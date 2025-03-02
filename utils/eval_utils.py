"""
Evaluation metrics for sequential recommendation.
This module contains implementations of common metrics used in recommendation systems:
- Recall and Precision@K
- Mean Reciprocal Rank (MRR)@K
- Mean Average Precision (MAP)@K
- Normalized Discounted Cumulative Gain (NDCG)@K
- Area Under the ROC Curve (AUC)
"""

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn as nn

def RecallPrecision_atK(test, r, k):
    """Calculate Recall and Precision at K.
    
    Args:
        test: Ground truth items for each user
        r: Binary relevance matrix
        k: Number of items to consider
        
    Returns:
        precision: Precision@K
        recall: Recall@K
    """
    tp = r[:, :k].sum(1)
    precision = np.sum(tp) / k
    recall_n = np.array([len(test[i]) for i in range(len(test))])
    recall = np.sum(tp / recall_n)
    return precision, recall

def MRR_atK(test, r, k):
    """Calculate Mean Reciprocal Rank at K.
    
    Args:
        test: Ground truth items for each user
        r: Binary relevance matrix
        k: Number of items to consider
        
    Returns:
        MRR: Mean Reciprocal Rank@K
    """
    pred = r[:, :k]
    weight = np.arange(1, k+1)
    MRR = np.sum(pred / weight, axis=1) / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    MRR = np.sum(MRR)
    return MRR

def MAP_atK(test, r, k):
    """Calculate Mean Average Precision at K.
    
    Args:
        test: Ground truth items for each user
        r: Binary relevance matrix
        k: Number of items to consider
        
    Returns:
        MAP: Mean Average Precision@K
    """
    pred = r[:, :k]
    rank = pred.copy()
    for i in range(k):
        rank[:, k - i - 1] = np.sum(rank[:, :k - i], axis=1)
    weight = np.arange(1, k+1)
    AP = np.sum(pred * rank / weight, axis=1)
    AP = AP / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    return np.sum(AP)

def NDCG_atK(test, r, k):
    """Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        test: Ground truth items for each user
        r: Binary relevance matrix
        k: Number of items to consider
        
    Returns:
        NDCG: Normalized Discounted Cumulative Gain@K
    """
    test_matrix = np.zeros((len(test), k))
    for i, items in enumerate(test):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = r * (1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test):
    """Calculate Area Under the ROC Curve.
    
    Args:
        all_item_scores: Predicted scores for all items
        dataset: Complete dataset
        test: Ground truth items
        
    Returns:
        auc: Area Under the ROC Curve
    """
    r = getLabel(test, dataset)
    r = r.flatten()
    all_item_scores = all_item_scores.flatten()
    return roc_auc_score(r, all_item_scores)

def getLabel(test, pred):
    """Convert test and predictions to binary relevance matrix.
    
    Args:
        test: Ground truth items
        pred: Predicted items
        
    Returns:
        Binary relevance matrix
    """
    r = []
    for i in range(len(test)):
        groundTrue = test[i]
        predictTopK = pred[i]
        pred_list = list(map(lambda x: x in groundTrue, predictTopK))
        pred_list = np.array(pred_list).astype("float")
        r.append(pred_list)
    return np.array(r).astype('float')

def compute_metrics(pred):
    """Compute all evaluation metrics for a prediction.
    
    Args:
        pred: Prediction object containing logits and labels
        
    Returns:
        Dictionary of computed metrics
    """
    logits = pred.predictions
    labels = pred.label_ids[0]
    metrics = {}
    
    # Calculate metrics at different K values
    for k in [1, 5, 10]:
        r = getLabel(labels, logits)
        metrics[f'ndcg_{k}'] = NDCG_atK(labels, r, k)
        metrics[f'hit_{k}'] = RecallPrecision_atK(labels, r, k)[1]
    
    metrics['mrr'] = MRR_atK(labels, r, 10)
    return metrics

def get_sample_scores(pred_list):
    pred_list = (-pred_list).argsort().argsort()[:, 0]
    HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
    HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
    HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
    return HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def choose_predict(predict_d1,predict_d2,domain_id):
    predict_d1_cse, predict_d2_cse = [], []
    for i in range(domain_id.shape[0]):
        if domain_id[i] == 0:
            predict_d1_cse.append(predict_d1[i,:])
        else:
            predict_d2_cse.append(predict_d2[i,:])
    if len(predict_d1_cse)!=0:
        predict_d1_cse = np.array(predict_d1_cse)
    if len(predict_d2_cse)!=0:
        predict_d2_cse = np.array(predict_d2_cse)
    return predict_d1_cse, predict_d2_cse

def choose_predict_overlap(predict_d1,predict_d2,domain_id,overlap_label):
    predict_d1_cse_over, predict_d1_cse_nono, predict_d2_cse_over, predict_d2_cse_nono = [], [], [], []
    for i in range(domain_id.shape[0]):
        if domain_id[i] == 0:
            if overlap_label[i][0]==0:
                predict_d1_cse_nono.append(predict_d1[i,:])
            else:
                predict_d1_cse_over.append(predict_d1[i,:])
        else:
            if overlap_label[i][0]==0:
                predict_d2_cse_nono.append(predict_d2[i,:])
            else:
                predict_d2_cse_over.append(predict_d2[i,:])
    if len(predict_d1_cse_over)!=0:
        predict_d1_cse_over = np.array(predict_d1_cse_over)
    if len(predict_d1_cse_nono)!=0:
        predict_d1_cse_nono = np.array(predict_d1_cse_nono)
    if len(predict_d2_cse_over)!=0:
        predict_d2_cse_over = np.array(predict_d2_cse_over)
    if len(predict_d2_cse_nono)!=0:
        predict_d2_cse_nono = np.array(predict_d2_cse_nono)
    return predict_d1_cse_over, predict_d1_cse_nono, predict_d2_cse_over, predict_d2_cse_nono

def compute_metrics_multiple(pred):
    logits = pred.predictions 
    print(logits.shape)
    loss = logits[:,:,-1]
    predict = logits[:,:,:-1]
    # if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
    #     HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = -1, -1, -1, -1, -1, -1, -1
    # else:
    HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = [],[],[],[],[],[],[]
    for i in range(predict.shape[1]):
        HIT_1_tmp, NDCG_1_tmp, HIT_5_tmp, NDCG_5_tmp, HIT_10_tmp, NDCG_10_tmp, MRR_tmp = get_sample_scores(predict[:,i,:])
        HIT_1.append(HIT_1_tmp)
        NDCG_1.append(NDCG_1_tmp)
        HIT_5.append(HIT_5_tmp)
        NDCG_5.append(NDCG_5_tmp)
        HIT_10.append(HIT_10_tmp)
        NDCG_10.append(NDCG_10_tmp)
        MRR.append(MRR_tmp)
    print("mrr:{}".format(MRR))
    return {
        'hit@1':HIT_1,
        'hit@5':HIT_5,
        'ndcg@5':NDCG_5,
        'hit@10':HIT_10,
        'ndcg@10':NDCG_10,
        'mrr':MRR,
        'loss':np.mean(loss,axis=0),
    }

def get_full_sort_score(answers, pred_list):
    recall, ndcg, mrr = [], [], []
    for k in [5, 10, 15, 20]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
        mrr.append(MRR_atK(answers, pred_list, k))
    return recall, ndcg, mrr