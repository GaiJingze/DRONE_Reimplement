import torch
import evaluate
import numpy as np

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
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

    # Compute U,V
    U=W @ V_W_T.T @ torch.diag(1 / S_W) @ U_Z_k @torch.diag(torch.sqrt(S_Z_k))
    V=torch.diag(torch.sqrt(S_Z_k)) @ V_Z_k @ torch.diag(1 / S_X) @ U_X.T
    M_U=V_W_T.T @ torch.diag(1 / S_W) @ U_Z_k @ torch.diag(torch.sqrt(S_Z_k))
    return U,V,M_U

def low_rank_approximation_attn(Q, K, Y_q, Y_k, rank):

    U_Q,V_Q,_ = low_rank_approximation(Q, Y_q, rank)
    U_K,V_K,_ = low_rank_approximation(K, Y_k, rank)

    QY_T = (Q @ Y_q).T
    KY = K @ Y_k

    _,M_V,M_U=low_rank_approximation(QY_T,KY,rank)
    
    return U_Q,V_Q,U_K,V_K,M_U,M_V

def low_rank_approximation_SVD(W, X, rank):
    U_W, S_W, V_W_T = torch.linalg.svd(W, full_matrices=False)
    U_W_k = U_W[:, :rank]
    S_W_k = S_W[:rank]
    V_W_k = V_W_T[:rank, :]
    
    U=U_W_k @ torch.diag(torch.sqrt(S_W_k))
    V=torch.diag(torch.sqrt(S_W_k)) @V_W_k
    return U,V,None

if __name__=='__main__':
    X=torch.randn((128,64),).double()@torch.randn((64,256)).double()
    Q=torch.randn((256,128)).double()
    K=torch.randn((256,128)).double()
    W=torch.randn((256,128)).double()
    rank=96
    #test low rank approximation
    U_prone,V_prone,_=low_rank_approximation(W,X,rank)
    U_svd,V_svd,_=low_rank_approximation_SVD(W,X,rank)
    y=W @ X
    y_prone=U_prone @ V_prone @ X
    y_svd=U_svd @ V_svd @ X
    e_prone,e_svd=torch.mean(torch.abs(y-y_prone)),torch.mean(torch.abs(y-y_svd))
    print(e_prone,e_svd)
    #test attention approximation
    U_Q,V_Q,U_K,V_K,M_U,M_V=low_rank_approximation_attn(Q,K,X,X,rank)
    U_Q_svd,V_Q_svd,_=low_rank_approximation_SVD(Q,X,rank)
    U_K_svd,V_K_svd,_=low_rank_approximation_SVD(K,X,rank)
    d=(Q@X).T @ (K@X)
    d_prone=(U_Q@V_Q@X).T @ M_U @ M_V @ (U_K@V_K@X)
    d_svd=(U_Q_svd@V_Q_svd@X).T @ (U_K_svd@V_K_svd@X)
    e_prone,e_svd=torch.mean(torch.abs(d-d_prone)),torch.mean(torch.abs(d-d_svd))
    print(e_prone,e_svd)