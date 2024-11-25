import torch
import evaluate
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
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
    
    result = tokenizer(
        *texts,
        padding='max_length',   
        truncation=True,
        max_length=128         
    )
    
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

def low_rank_approximation_attn(Q, K, V, X, rank, head_num=12 ,bias_q=0, bias_k=0 ):

    U_Q,V_Q,_ = low_rank_approximation(Q, X, rank*head_num)
    U_K,V_K,_ = low_rank_approximation(K, X, rank*head_num)
    U_V,V_V,_ = low_rank_approximation(V, X, rank*head_num)

    QY_T = (U_Q @ V_Q @ X).T+ bias_q  # [seq_len, d_model]
    KY = ((U_K @ V_K @ X).T + bias_k).T  # [d_model, seq_len]
        
    seq_len,d_model = QY_T.shape[-2:]
    d_k = d_model // head_num

    QY_T = QY_T.view(seq_len, head_num, d_k)  # [seq_len, head_num, d_k]
    KY = KY.view(head_num, d_k, seq_len)  # [head_num, d_k, seq_len]

    M_V_list = []
    M_U_list = []
    for i in range(head_num): # multi-head
        QY_T_i = QY_T[ :, i, :]  # [seq_len, d_k]
        KY_i = KY[i, :, :]  # [d_k, seq_len]
        _, M_V_i, M_U_i = low_rank_approximation(QY_T_i, KY_i, rank)
        M_V_list.append(M_V_i)
        M_U_list.append(M_U_i)
    
    M_V = torch.stack(M_V_list, dim=0)  # [batch_size, head_num, ...]
    M_U = torch.stack(M_U_list, dim=0)
    
    return U_Q, V_Q, U_K, V_K, M_U, M_V, U_V, V_V

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
    X=X.T
    Q=torch.randn((384,256)).double()
    K=torch.randn((384,256)).double()
    V=torch.randn((384,256)).double()
    W=torch.randn((384,256)).double()
    rank=96
    #test low rank approximation
    U_drone,V_drone,_=low_rank_approximation(W,X,rank)
    U_svd,V_svd,_=low_rank_approximation_SVD(W,X,rank)
    y=W @ X
    y_drone=U_drone @ V_drone @ X
    y_svd=U_svd @ V_svd @ X
    e_drone,e_svd=torch.mean(torch.abs(y-y_drone)),torch.mean(torch.abs(y-y_svd))
    print(e_drone,e_svd)


    #test attention approximation
    head_num=1
    rank=64
    U_Q,V_Q,U_K,V_K,M_U,M_V,_,_=low_rank_approximation_attn(Q,K,V,X,rank,head_num=head_num)
    U_Q_svd,V_Q_svd,_=low_rank_approximation_SVD(Q,X,rank)
    U_K_svd,V_K_svd,_=low_rank_approximation_SVD(K,X,rank)
    d=(Q@X).T @ (K@X)
    d_drone=(U_Q@V_Q@X).T @ M_U.squeeze(0) @ M_V.squeeze(0) @ (U_K@V_K@X)
    d_svd=(U_Q_svd@V_Q_svd@X).T @ (U_K_svd@V_K_svd@X)
    e_drone,e_svd=torch.mean(torch.abs(d-d_drone)),torch.mean(torch.abs(d-d_svd))
    print(e_drone,e_svd)