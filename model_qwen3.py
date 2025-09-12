import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
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

class LLM4RecQwen3(nn.Module):
    def __init__(self, **args):
        super(LLM4RecQwen3, self).__init__()
        self.args = args
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']

        print(f'Initializing Qwen3 language decoder ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=self.args['lora_target_modules'],
            bias='none',
        )

        # Use Qwen3 model with AutoModelForCausalLM for proper compatibility
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            args['base_model'],
            torch_dtype=torch.bfloat16,
            cache_dir=args['cache_dir'],
            trust_remote_code=True,
            # device_map=self.args['device_map']
        )
        
        if self.args['drop_type'] == "trune":
            self.qwen_model.model.layers = nn.ModuleList(self.qwen_model.model.layers[:self.args['qwen_decoder_nums']])
        elif self.args['drop_type'] == "interval":
            # Interval for layer dropping
            interval_nums = self.args['interval_nums']
            # Keep layers with interval-based dropping
            self.qwen_model.model.layers = nn.ModuleList([layer for i, layer in enumerate(self.qwen_model.model.layers) if (i + 1) % (interval_nums + 1) != 0])
            num_layers = len(self.qwen_model.model.layers)
            print(f'Number of layers in the model: {num_layers}')

        if self.args['train_stargy'] == "lora":
            self.qwen_model = get_peft_model(self.qwen_model, peft_config)
            self.qwen_model.print_trainable_parameters()
        self.qwen_model.config.use_cache = False
        
        # Use AutoTokenizer for Qwen3-14B
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            args['base_model'], 
            use_fast=False, 
            cache_dir=args['cache_dir'],
            trust_remote_code=True
        )
        
        # Set pad token for Qwen3
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
        self.qwen_tokenizer.padding_side = "right"
        
        self.instruct_ids, self.instruct_mask = self.qwen_tokenizer(
            self.args['instruction_text'][0],
            truncation=True, 
            padding=False,
            return_tensors='pt', 
            add_special_tokens=False
        ).values()
        
        self.response_ids, self.response_mask = self.qwen_tokenizer(
            self.args['instruction_text'][1],
            truncation=True, 
            padding=False,
            return_tensors='pt', 
            add_special_tokens=False
        ).values()
        print('Qwen3-14B language decoder initialized.')

        self.task_type = args['task_type']
        self.input_embeds = nn.Embedding.from_pretrained(self.args['input_embeds'], freeze=True).to(torch.bfloat16)
        self.input_proj = nn.Linear(self.input_dim, self.qwen_model.config.hidden_size).to(torch.bfloat16)
        self.score = nn.Linear(self.qwen_model.config.hidden_size, self.input_dim, bias=False).to(torch.bfloat16)
        self.loss = torch.nn.CrossEntropyLoss()

    def predict(self, inputs, inputs_mask, output_hidden_states=False, output_logits=True):
        bs = inputs.shape[0]
        if self.args['train_stargy'] == "lora":
            instruct_embeds = self.qwen_model.get_input_embeddings()(self.instruct_ids.cuda()).expand(bs, -1, -1).to(torch.bfloat16)
            response_embeds = self.qwen_model.get_input_embeddings()(self.response_ids.cuda()).expand(bs, -1, -1).to(torch.bfloat16)
        else:
            instruct_embeds = self.qwen_model.get_input_embeddings()(self.instruct_ids.cuda()).expand(bs, -1, -1).to(torch.bfloat16)
            response_embeds = self.qwen_model.get_input_embeddings()(self.response_ids.cuda()).expand(bs, -1, -1).to(torch.bfloat16)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        if self.task_type == 'general':
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        
        # Ensure all tensors are bfloat16
        inputs = inputs.to(torch.bfloat16)
        instruct_embeds = instruct_embeds.to(torch.bfloat16)
        response_embeds = response_embeds.to(torch.bfloat16)
        
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.qwen_model(
            inputs_embeds=inputs, 
            attention_mask=attention_mask, 
            return_dict=True, 
            output_hidden_states=output_hidden_states
        )
        
        if output_logits:
            # For CausalLM models, we need to access hidden_states differently
            if hasattr(outputs, 'last_hidden_state'):
                pooled_output = outputs.last_hidden_state[:, -1]
            else:
                # For CausalLMOutputWithPast, we need to get the last layer's hidden states
                # First enable output_hidden_states if not already enabled
                if not output_hidden_states:
                    outputs = self.qwen_model(
                        inputs_embeds=inputs, 
                        attention_mask=attention_mask, 
                        return_dict=True, 
                        output_hidden_states=True
                    )
                pooled_output = outputs.hidden_states[-1][:, -1]
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
            instruct_embeds = self.qwen_model.get_input_embeddings()(self.instruct_ids.cuda()).expand(bs, -1, -1).to(torch.bfloat16)
            response_embeds = self.qwen_model.get_input_embeddings()(self.response_ids.cuda()).expand(bs, -1, -1).to(torch.bfloat16)
        else:
            instruct_embeds = self.qwen_model.get_input_embeddings()(self.instruct_ids.cuda()).expand(bs, -1, -1).to(torch.bfloat16)
            response_embeds = self.qwen_model.get_input_embeddings()(self.response_ids.cuda()).expand(bs, -1, -1).to(torch.bfloat16)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        if self.task_type == 'general':
            users = self.user_proj(self.user_embeds(inputs[:, 0].unsqueeze(1)))
            items = self.input_proj(self.input_embeds(inputs[:, 1:]))
            inputs = torch.cat([users, items], dim=1)
        else:
            inputs = self.input_proj(self.input_embeds(inputs))
        
        # Ensure inputs are bfloat16
        inputs = inputs.to(torch.bfloat16)
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.qwen_model(
            inputs_embeds=inputs, 
            attention_mask=attention_mask, 
            return_dict=True, 
            output_hidden_states=True
        )
        
        # Multiple predictions from different layers
        multiple_logits = []
        hidden_states = outputs.hidden_states
        for i in range(len(hidden_states) // 4, len(hidden_states), len(hidden_states) // 4):
            pooled_output = hidden_states[i][:, -1]
            pooled_logits = self.score(pooled_output)
            multiple_logits.append(pooled_logits)
        
        return multiple_logits

    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        loss = None
        if torch.max(data_type).item() == 0:  # all item ce loss
            pooled_logits = self.predict(inputs, inputs_mask, output_hidden_states=False, output_logits=True)
            test_item_emb = self.input_embeds.weight.to(torch.bfloat16)
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
            pos_embs = self.input_embeds(answers).to(torch.bfloat16)
            neg_embs = self.input_embeds(neg_samples).to(torch.bfloat16)
            pos_logits = torch.matmul(log_feats, pos_embs.permute(0, 2, 1))
            neg_logits = torch.matmul(log_feats, neg_embs.permute(0, 2, 1))

            pos_label = torch.ones_like(pos_logits).cuda().to(torch.bfloat16)
            neg_label = torch.zeros_like(neg_logits).cuda().to(torch.bfloat16)
            loss_real = nn.BCEWithLogitsLoss()(pos_logits, pos_label)
            loss_false = nn.BCEWithLogitsLoss()(neg_logits, neg_label)
            loss = loss_real + loss_false
            predict = torch.cat((pos_logits, neg_logits), -1).squeeze()

            return {
                'loss': loss,
                'logits': predict,
            }

class LLM4RecQwen3Teacher(LLM4RecQwen3):
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

class LLM4RecQwen3Student(LLM4RecQwen3):
    def __init__(self, **args):
        super().__init__(**args)
        self.distill_block = args['distill_block']
        self.is_cls_multiple = args['is_cls_multiple']
        self.down_layer_list = nn.ModuleList()
        if self.is_cls_multiple:
            for _ in range(self.distill_block-1):
                self.down_layer_list.append(nn.Linear(self.qwen_model.config.hidden_size, self.input_dim, bias=False))

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

class LLM4RecQwen3Teacher(LLM4RecQwen3):
    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        loss = None
        if torch.max(data_type).item() == 0:  # all item ce loss
            pooled_logits, teacher_output_states = self.predict(inputs, inputs_mask, output_hidden_states=True, output_logits=True)
            test_item_emb = self.input_embeds.weight.to(torch.bfloat16)
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
            pos_embs = self.input_embeds(answers).to(torch.bfloat16)
            neg_embs = self.input_embeds(neg_samples).to(torch.bfloat16)
            pos_logits = torch.matmul(log_feats, pos_embs.permute(0, 2, 1))
            neg_logits = torch.matmul(log_feats, neg_embs.permute(0, 2, 1))

            pos_label = torch.ones_like(pos_logits).cuda().to(torch.bfloat16)
            neg_label = torch.zeros_like(neg_logits).cuda().to(torch.bfloat16)
            loss_real = nn.BCEWithLogitsLoss()(pos_logits, pos_label)
            loss_false = nn.BCEWithLogitsLoss()(neg_logits, neg_label)
            loss = loss_real + loss_false
            predict = torch.cat((pos_logits, neg_logits), -1).squeeze()

            return {
                'loss': loss,
                'logits': predict,
            }

class LLM4RecQwen3Student(LLM4RecQwen3):
    def __init__(self, **args):
        super().__init__(**args)
        self.distill_block = args['distill_block']
        self.is_cls_multiple = args['is_cls_multiple']
        self.down_layer_list = nn.ModuleList()
        if self.is_cls_multiple:
            for _ in range(self.distill_block-1):
                self.down_layer_list.append(nn.Linear(self.qwen_model.config.hidden_size, self.input_dim, bias=False).to(torch.bfloat16))

    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        loss = None
        if torch.max(data_type).item() == 0:  # all item ce loss
            pooled_logits, student_output_states = self.predict(inputs, inputs_mask, output_hidden_states=True, output_logits=True)
            test_item_emb = self.input_embeds.weight.to(torch.bfloat16)
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
            pos_embs = self.input_embeds(answers).to(torch.bfloat16)
            neg_embs = self.input_embeds(neg_samples).to(torch.bfloat16)
            pos_logits = torch.matmul(log_feats, pos_embs.permute(0, 2, 1))
            neg_logits = torch.matmul(log_feats, neg_embs.permute(0, 2, 1))

            pos_label = torch.ones_like(pos_logits).cuda().to(torch.bfloat16)
            neg_label = torch.zeros_like(neg_logits).cuda().to(torch.bfloat16)
            loss_real = nn.BCEWithLogitsLoss()(pos_logits, pos_label)
            loss_false = nn.BCEWithLogitsLoss()(neg_logits, neg_label)
            loss = loss_real + loss_false
            predict = torch.cat((pos_logits, neg_logits), -1).squeeze()

            return {
                'loss': loss,
                'logits': predict,
            }

class LLM4RecQwen3Distill(nn.Module):
    def __init__(self, **args):
        super(LLM4RecQwen3Distill, self).__init__()
        self.args = args
        self.model_teacher = LLM4RecQwen3(
            base_model=self.args['base_model'],
            task_type=self.args['task_type'],
            cache_dir=self.args['cache_dir'],
            input_dim=128,
            output_dim=0,
            interval_nums=self.args['interval_nums'],
            drop_type=self.args['drop_type'],
            lora_r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            lora_target_modules=self.args['lora_target_modules'],
            device_map=self.args['device_map'],
            instruction_text=self.args['instruction_text'],
            train_stargy=self.args['train_stargy'],
            user_embeds=None,
            input_embeds=self.args['item_embed'],
            seq_len=30,
            qwen_decoder_nums=self.args['qwen_decoder_nums_teacher'],
        )
        self.model_student = LLM4RecQwen3(
            base_model=self.args['base_model'],
            task_type=self.args['task_type'],
            cache_dir=self.args['cache_dir'],
            input_dim=128,
            output_dim=0,
            interval_nums=self.args['interval_nums'],
            drop_type=self.args['drop_type'],
            lora_r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            lora_target_modules=self.args['lora_target_modules'],
            device_map=self.args['device_map'],
            instruction_text=self.args['instruction_text'],
            train_stargy=self.args['train_stargy'],
            user_embeds=None,
            input_embeds=self.args['item_embed'],
            seq_len=30,
            qwen_decoder_nums=self.args['qwen_decoder_nums_student'],
        )
        self.teacher_block = self.args['qwen_decoder_nums_teacher'] // 4
        self.student_block = self.args['qwen_decoder_nums_student'] // 4
        self.distill_lambda = self.args['distill_lambda']
        self.loss_cls = torch.nn.CrossEntropyLoss()
        self.is_cls_multiple_teacher = self.args['is_cls_multiple_teacher']
        self.is_cls_multiple_student = self.args['is_cls_multiple_student']

    def forward(self, input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type):
        # Teacher forward pass
        with torch.no_grad():
            teacher_output = self.model_teacher(input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type)
        
        # Student forward pass  
        student_output = self.model_student(input_ids, labels, inputs, inputs_mask, answers, neg_samples, data_type)
        
        # Distillation loss computation
        student_loss = student_output['loss']
        distill_loss = 0
        
        if torch.max(data_type).item() == 0 and 'teacher_output_states' in teacher_output and 'student_output_states' in student_output:
            teacher_states = teacher_output['teacher_output_states']
            student_states = student_output['student_output_states']
            
            # Knowledge distillation between corresponding layers
            for i in range(min(len(teacher_states), len(student_states))):
                if i % (len(teacher_states) // len(student_states)) == 0:
                    teacher_hidden = teacher_states[i][:, -1].to(torch.bfloat16)
                    student_hidden = student_states[i // (len(teacher_states) // len(student_states))][:, -1].to(torch.bfloat16)
                    distill_loss += F.mse_loss(student_hidden, teacher_hidden)
        
        total_loss = student_loss + self.distill_lambda * distill_loss
        
        return {
            'loss': total_loss,
            'logits': student_output['logits'],
            'student_loss': student_loss,
            'distill_loss': distill_loss,
        }