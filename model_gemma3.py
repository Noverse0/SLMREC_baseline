import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import GemmaModel, GemmaForCausalLM, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model
)
import math
import os

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class Log2feats(torch.nn.Module):
    def __init__(self, user_emb_dim, item_emb_dim, seq_len):
        super(Log2feats, self).__init__()
        self.pos_emb = torch.nn.Embedding(seq_len, item_emb_dim) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=0.5)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)

        for _ in range(2):
            new_attn_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(user_emb_dim,
                                                            8,
                                                            0.5)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(user_emb_dim, 0.5)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, log_seqs):
        seqs = log_seqs
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).cuda())
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs.cpu() == 0).cuda()
        seqs *= ~timeline_mask # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device="cuda"))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

class SASRec(nn.Module):
    def __init__(self, args, device, dataset):
        super(SASRec, self).__init__()
        self.device = device
        self.m_item = dataset.m_item
        self.dim = args.emb_dim
        self.hid_dim = args.hid_dim
        self.embedding = nn.Embedding(self.m_item, self.dim)
        self.up_emb = nn.Linear(self.dim,self.hid_dim)
        self.down_emb = nn.Linear(self.hid_dim,self.dim)

        self.log2feats = Log2feats(self.hid_dim, self.dim, args.max_seq_length)

        self.loss = nn.CrossEntropyLoss()

    def seq_emb(self, seq):
        batch_size = seq.shape[0]
        item_embs = self.embedding(seq)
        item_embs = self.up_emb(item_embs)
        log_feats = self.log2feats(item_embs)
        log_feats = self.down_emb(log_feats)
        return log_feats

    def seq_emb_up(self, seq):
        batch_size = seq.shape[0]
        item_embs = self.embedding(seq)
        item_embs = self.up_emb(item_embs)
        log_feats = self.log2feats(item_embs)
        return log_feats

    def forward(self, seq, pos):
        # ce loss 
        log_feats = self.seq_emb(seq) # user, item, item_emb
        logits = self.embedding.weight[1:] # item_emb
        logits = torch.matmul(log_feats, logits.transpose(0, 1))

        loss = self.loss(logits, pos.squeeze(-1))
        return loss

class LLM4RecGemma3(nn.Module):
    def __init__(self, **args):
        super(LLM4RecGemma3, self).__init__()
        self.args = args
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']

        print(f'Initializing Gemma3 language decoder ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=self.args['lora_target_modules'],
            bias='none',
        )

        # Use Gemma3 model instead of LLaMA
        self.gemma_model = GemmaModel.from_pretrained(
            "google/gemma-2-9b",  # Gemma3 9B model
            torch_dtype=torch.float16,
            cache_dir=args['cache_dir'],
            device_map=self.args['device_map']
        )
        
        if self.args['drop_type'] == "trune":
            self.gemma_model.layers = nn.ModuleList(self.gemma_model.layers[:self.args['gemma_decoder_nums']])
        elif self.args['drop_type'] == "interval":
            # Interval for layer dropping
            interval_nums = self.args['interval_nums']
            # Keep layers with interval-based dropping
            self.gemma_model.layers = nn.ModuleList([layer for i, layer in enumerate(self.gemma_model.layers) if (i + 1) % (interval_nums + 1) != 0])
            num_layers = len(self.gemma_model.layers)
            print(f'Number of layers in the model: {num_layers}')

        if self.args['train_stargy'] == "lora":
            self.gemma_model = get_peft_model(self.gemma_model, peft_config)
            self.gemma_model.print_trainable_parameters()
        self.gemma_model.config.use_cache = False
        
        # Use AutoTokenizer for Gemma3
        self.gemma_tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-9b", 
            use_fast=False, 
            cache_dir=args['cache_dir']
        )
        
        # Set pad token for Gemma3
        if self.gemma_tokenizer.pad_token is None:
            self.gemma_tokenizer.pad_token = self.gemma_tokenizer.eos_token
        self.gemma_tokenizer.padding_side = "right"
        
        self.instruct_ids, self.instruct_mask = self.gemma_tokenizer(
            self.args['instruction_text'][0],
            truncation=True, 
            padding=False,
            return_tensors='pt', 
            add_special_tokens=False
        ).values()
        
        self.response_ids, self.response_mask = self.gemma_tokenizer(
            self.args['instruction_text'][1],
            truncation=True, 
            padding=False,
            return_tensors='pt', 
            add_special_tokens=False
        ).values()
        print('Gemma3 language decoder initialized.')

        self.task_type = args['task_type']
        self.input_embeds = nn.Embedding.from_pretrained(self.args['input_embeds'], freeze=True)
        self.input_proj = nn.Linear(self.input_dim, self.gemma_model.config.hidden_size)
        self.score = nn.Linear(self.gemma_model.config.hidden_size, self.input_dim, bias=False)
        self.loss = torch.nn.CrossEntropyLoss()

    def predict(self, inputs, inputs_mask, output_hidden_states=False, output_logits=True):
        bs = inputs.shape[0]
        if self.args['train_stargy'] == "lora":
            instruct_embeds = self.gemma_model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
            response_embeds = self.gemma_model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        else:
            instruct_embeds = self.gemma_model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
            response_embeds = self.gemma_model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        if self.task_type == 'general':
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.gemma_model(
            inputs_embeds=inputs, 
            attention_mask=attention_mask, 
            return_dict=True, 
            output_hidden_states=output_hidden_states
        )
        
        if output_logits:
            pooled_output = outputs.last_hidden_state[:, -1]
            pooled_logits = self.score(pooled_output)
        if not output_hidden_states:
            return pooled_logits
        else:
            if output_logits:
                return pooled_logits, outputs.hidden_states
            else:
                return outputs.hidden_states

    def multiple_predict(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        if self.args['train_stargy'] == "lora":
            instruct_embeds = self.gemma_model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
            response_embeds = self.gemma_model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        else:
            instruct_embeds = self.gemma_model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
            response_embeds = self.gemma_model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        if self.task_type == 'general':
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.gemma_model(
            inputs_embeds=inputs, 
            attention_mask=attention_mask, 
            return_dict=True, 
            output_hidden_states=True
        )
        
        # Multiple predictions from different layers
        multiple_logits = []
        for i in range(len(outputs.hidden_states) // 4, len(outputs.hidden_states), len(outputs.hidden_states) // 4):
            pooled_output = outputs.hidden_states[i][:, -1]
            pooled_logits = self.score(pooled_output)
            multiple_logits.append(pooled_logits)
        
        return multiple_logits

    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        loss = None
        if torch.max(data_type).item() == 0:  # all item ce loss
            pooled_logits = self.predict(inputs, inputs_mask, output_hidden_states=False, output_logits=True)
            test_item_emb = self.input_embeds.weight
            logits = torch.matmul(pooled_logits, test_item_emb.transpose(0, 1))
            loss = self.loss(logits, answers.squeeze(-1))
            predict = logits
            
            return {
                'loss': loss,
                'logits': predict,
            }
        elif torch.max(data_type).item() == 1:  # predict sample negative, bce loss
            pooled_logits = self.predict(inputs, inputs_mask, output_hidden_states=False, output_logits=True)
            log_feats = pooled_logits.unsqueeze(1)
            pos_embs = self.input_embeds(answers)
            neg_embs = self.input_embeds(neg_samples)
            pos_logits = torch.matmul(log_feats, pos_embs.permute(0, 2, 1))
            neg_logits = torch.matmul(log_feats, neg_embs.permute(0, 2, 1))

            pos_label = torch.ones_like(pos_logits).cuda()
            neg_label = torch.zeros_like(neg_logits).cuda()
            loss_real = nn.BCEWithLogitsLoss()(pos_logits, pos_label)
            loss_false = nn.BCEWithLogitsLoss()(neg_logits, neg_label)
            loss = loss_real + loss_false
            predict = torch.cat((pos_logits, neg_logits), -1).squeeze()

            return {
                'loss': loss,
                'logits': predict,
            }

class LLM4RecGemma3Teacher(LLM4RecGemma3):
    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        loss = None
        if torch.max(data_type).item() == 0:  # all item ce loss
            pooled_logits, teacher_output_states = self.predict(inputs, inputs_mask, output_hidden_states=True, output_logits=True)
            test_item_emb = self.input_embeds.weight
            logits = torch.matmul(pooled_logits, test_item_emb.transpose(0, 1))
            loss = self.loss(logits, answers.squeeze(-1))
            predict = logits
            
            return {
                'loss': loss,
                'logits': predict,
                'teacher_output_states': teacher_output_states,
                'data_type': data_type,
            }
        elif torch.max(data_type).item() == 1:  # predict sample negative, bce loss
            pooled_logits = self.predict(inputs, inputs_mask, output_hidden_states=False, output_logits=True)
            log_feats = pooled_logits.unsqueeze(1)
            pos_embs = self.input_embeds(answers)
            neg_embs = self.input_embeds(neg_samples)
            pos_logits = torch.matmul(log_feats, pos_embs.permute(0, 2, 1))
            neg_logits = torch.matmul(log_feats, neg_embs.permute(0, 2, 1))

            pos_label = torch.ones_like(pos_logits).cuda()
            neg_label = torch.zeros_like(neg_logits).cuda()
            loss_real = nn.BCEWithLogitsLoss()(pos_logits, pos_label)
            loss_false = nn.BCEWithLogitsLoss()(neg_logits, neg_label)
            loss = loss_real + loss_false
            predict = torch.cat((pos_logits, neg_logits), -1).squeeze()

            return {
                'loss': loss,
                'logits': predict,
            }

class LLM4RecGemma3Student(LLM4RecGemma3):
    def __init__(self, **args):
        super().__init__(**args)
        self.distill_block = args['distill_block']
        self.is_cls_multiple = args['is_cls_multiple']
        self.down_layer_list = nn.ModuleList()
        if self.is_cls_multiple:
            for _ in range(self.distill_block-1):
                self.down_layer_list.append(nn.Linear(self.gemma_model.config.hidden_size, self.input_dim, bias=False))

    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        loss = None
        if torch.max(data_type).item() == 0:  # all item ce loss
            pooled_logits, student_output_states = self.predict(inputs, inputs_mask, output_hidden_states=True, output_logits=True)
            test_item_emb = self.input_embeds.weight
            logits = torch.matmul(pooled_logits, test_item_emb.transpose(0, 1))
            loss = self.loss(logits, answers.squeeze(-1))
            predict = logits
            loss_cls_multiple = 0
            if self.is_cls_multiple:
                for i in range(0, self.distill_block-1): 
                    pooled_logits_tmp = self.down_layer_list[i](student_output_states[(len(student_output_states)//self.distill_block)*(i+1)][:, -1]) 
                    logits_tmp = torch.matmul(pooled_logits_tmp, test_item_emb.transpose(0, 1))
                    loss_tmp = self.loss(logits_tmp, answers.squeeze(-1))
                    loss_cls_multiple = loss_cls_multiple + loss_tmp
            return {
                'loss': loss,
                'logits': predict,
                'student_output_states': student_output_states,
                'data_type': data_type,
                'loss_cls_multiple': loss_cls_multiple,
            }
        elif torch.max(data_type).item() == 1:
            pooled_logits = self.predict(inputs, inputs_mask, output_hidden_states=False, output_logits=True)
            log_feats = pooled_logits.unsqueeze(1)
            pos_embs = self.input_embeds(answers)
            neg_embs = self.input_embeds(neg_samples)
            pos_logits = torch.matmul(log_feats, pos_embs.permute(0, 2, 1))
            neg_logits = torch.matmul(log_feats, neg_embs.permute(0, 2, 1))

            pos_label = torch.ones_like(pos_logits).cuda()
            neg_label = torch.zeros_like(neg_logits).cuda()
            loss_real = nn.BCEWithLogitsLoss()(pos_logits, pos_label)
            loss_false = nn.BCEWithLogitsLoss()(neg_logits, neg_label)
            loss = loss_real + loss_false
            predict = torch.cat((pos_logits, neg_logits), -1).squeeze()

            return {
                'loss': loss,
                'logits': predict,
            } 