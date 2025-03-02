import numpy as np
import random
import os
import torch
import time
from collections import defaultdict
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import torch.nn.functional as F
from utils.data_utils import *
from utils.eval_utils import *
from model import SASRec
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm
from functools import partial
import logging
from utils import *
from utils.log_utils import *
from torch.utils.data import DataLoader, RandomSampler

logger = logging.getLogger()

def test(model, args, valLoader):
    """Evaluate model performance on validation/test set.
    
    Args:
        model: The SASRec model to evaluate
        args: Training arguments
        valLoader: DataLoader for validation/test data
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    stats = AverageMeter('loss','ndcg_1','ndcg_5','ndcg_10','hit_1','hit_5','hit_10','MRR')
    for k,sample in enumerate(tqdm(valLoader)):
        batch = tuple(t for t in sample)
        user_ids, input_ids, target_pos, target_neg, answers, neg_samples, _ = batch
        input_ids = input_ids.cuda()
        answers = answers.cuda()
        neg_samples = neg_samples.cuda()
        with torch.no_grad():
            pos_logits, neg_logits = model.predict_sample(input_ids, answers, neg_samples)
        pos_label = torch.ones_like(pos_logits).cuda()
        neg_label = torch.zeros_like(neg_logits).cuda()
        loss_real = nn.BCEWithLogitsLoss()(pos_logits, pos_label)
        loss_false = nn.BCEWithLogitsLoss()(neg_logits, neg_label)
        loss = loss_real + loss_false
        predict = torch.cat((pos_logits,neg_logits),-1).squeeze().cpu().detach().numpy().copy()
        pos_logits = pos_logits.squeeze()
        neg_logits = torch.mean(neg_logits.squeeze(),-1)
        
        HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = get_sample_scores(predict)
        stats.update(loss=loss.item(),ndcg_1=NDCG_1,ndcg_5=NDCG_5,ndcg_10=NDCG_10,hit_1=HIT_1,hit_5=HIT_5,hit_10=HIT_10,MRR=MRR)
    return stats.loss, stats.hit_1, stats.ndcg_1, stats.hit_5, stats.ndcg_5, stats.hit_10, stats.ndcg_10, stats.MRR

def train(model, device, trainLoader, args, valLoader, testLoader):
    """Train the SASRec model.
    
    Args:
        model: The SASRec model to train
        device: Device to train on (cuda/cpu)
        trainLoader: DataLoader for training data
        args: Training arguments
        valLoader: DataLoader for validation data
        testLoader: DataLoader for test data
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_ndcg = -1
    best_epoch = -1
    
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0.0
        
        for k, sample in enumerate(tqdm(trainLoader)):
            batch = tuple(t for t in sample)
            user_ids, input_ids, target_pos, target_neg, answers, neg_samples, _ = batch
            input_ids = input_ids.to(device)
            answers = answers.to(device)
            neg_samples = neg_samples.to(device)
            
            optimizer.zero_grad()
            loss = model(input_ids, answers)

            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Epoch {epoch} Train Loss: {total_loss/len(trainLoader):.4f}')
        
        # Evaluate on validation set
        metrics = test(model, args, valLoader)
        ndcg = metrics[6]
        
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pt'))
            
        print(f'Epoch {epoch} Val NDCG@10: {ndcg:.4f} Best: {best_ndcg:.4f} at epoch {best_epoch}')
    
    # Test final model
    print('Testing best model...')
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pt')))
    metrics = test(model, args, testLoader)
    print(f'Test metrics: {metrics}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequential Recommendation Training')
    parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hid_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--max_seq_length', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--model_dir', type=str, default='model/', help='Directory to save model checkpoints')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load data and create model
    datasetTrain = SASRecDataset(item_size=999, max_seq_length=args.max_seq_length,data_type='train',csv_path=args.data_path)
    trainLoader = data.DataLoader(datasetTrain, batch_size=args.batch_size, shuffle=True, num_workers=8)

    datasetVal = SASRecDataset(item_size=999, max_seq_length=args.max_seq_length,data_type='valid',csv_path=args.data_path)
    valLoader = data.DataLoader(datasetVal, batch_size=args.batch_size, shuffle=False, num_workers=8)

    datasetTest = SASRecDataset(item_size=999, max_seq_length=args.max_seq_length,data_type='test',csv_path=args.data_path)
    testLoader = data.DataLoader(datasetTest, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = SASRec(args, device='cuda', dataset=datasetTest).cuda()
    
    # Train model
    train(model, 'cuda', trainLoader, args, valLoader, testLoader)