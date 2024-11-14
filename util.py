import torch
import evaluate
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_keys = {
    'mrpc': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'qnli': ('question', 'sentence'),
    'rte': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
    'qqp': ('question1', 'question2'),
    'stsb': ('sentence1', 'sentence2'),
}

def preprocess_function(examples, tokenizer, sentence1_key, sentence2_key):
    if sentence2_key is None:
        texts = (examples[sentence1_key],)
    else:
        texts = (examples[sentence1_key], examples[sentence2_key])
    # Tokenization
    result = tokenizer(*texts, truncation=True, max_length=128)
    # Add labels
    result['labels'] = examples['label']
    return result

def compute_metrics(eval_preds, dataset_name):
    predictions, labels = eval_preds
    if dataset_name == 'stsb':
        # Regression task
        predictions = predictions.squeeze()
        metric = evaluate.load('glue', dataset_name)
        result = metric.compute(predictions=predictions, references=labels)
        return result
    else:
        # Classification task
        predictions = np.argmax(predictions, axis=1)
        metric = evaluate.load('glue', dataset_name)
        result = metric.compute(predictions=predictions, references=labels)
        return result

def low_rank_approximation(W, X, rank):
    # SVD of W & X
    U_W, S_W, V_W_T = torch.linalg.svd(W, full_matrices=False)
    U_X, S_X, V_X_T = torch.linalg.svd(X, full_matrices=False)
        
    # Compute Z = S_W_r V_W_r^T U_X_t S_X_t and truncate
    Z = torch.diag(S_W) @ V_W_T @ U_X @ torch.diag(S_X)
    U_Z, S_Z, V_Z_T = torch.linalg.svd(Z, full_matrices=False)
    U_Z_k = U_Z[:, :rank]
    S_Z_k = S_Z[:rank]
    V_Z_k = V_Z_T[:rank, :]

    # Compute W
    W_approx = W @ V_W_T.T @ torch.diag(1 / S_W) @ U_Z_k @torch.diag(S_Z_k) @ V_Z_k @ torch.diag(1 / S_X) @ U_X.T
    return W_approx


import torch


def low_rank_approximation_attn(Q, K, Y_q, Y_k, rank):

    Q_approx = low_rank_approximation(Q, Y_q, rank)
    K_approx = low_rank_approximation(K, Y_k, rank)

    #print("Q_approx", Q_approx.shape)
    #print("K_approx", K_approx.shape)

    QY = Q @ Y_q
    KY = K @ Y_k

    U_W, S_W, V_W_T = torch.linalg.svd(QY.T, full_matrices=False)
    U_X, S_X, V_X_T = torch.linalg.svd(KY, full_matrices=False)

    Z = torch.diag(S_W) @ V_W_T @ U_X @ torch.diag(S_X)
    U_Z, S_Z, V_Z_T = torch.linalg.svd(Z, full_matrices=False)
    U_Z_k = U_Z[:, :rank]
    S_Z_k = S_Z[:rank]
    V_Z_k = V_Z_T[:rank, :]

    M = V_W_T.T @ torch.diag(1 / S_W) @ U_Z_k @ torch.diag(S_Z_k) @ V_Z_k @ torch.diag(1 / S_X) @ U_X.T

    '''
    Q_approx = U_W @ torch.diag(S_W) @ V_W_T @ M_approx
    K_approx = M_approx @ U_X @ torch.diag(S_X) @ V_X_T



    Y_q_inv = np.linalg.pinv(Y_q)
    Y_k_inv = np.linalg.pinv(Y_k)

    Q_approx = Q_approx @ Y_q_inv
    K_approx = K_approx @ Y_k_inv
    '''
    K_approx = M @ K_approx
    return Q_approx, K_approx

def low_rank_approximation_SVD(W, X, rank):
    U_W, S_W, V_W_T = torch.linalg.svd(W, full_matrices=False)
    U_W_k = U_W[:, :rank]
    S_W_k = S_W[:rank]
    V_W_k = V_W_T[:rank, :]
    
    W_approx = U_W_k @ torch.diag(S_W_k) @V_W_k
    return W_approx
