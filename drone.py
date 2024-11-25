import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from transformers import AutoModel, AutoTokenizer, BertTokenizer

from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification,
    BertSdpaSelfAttention,
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertEmbeddings,
    BertPooler
)
from module import LowRankLinear,LowRankAttention
from tqdm.auto import tqdm
from datasets import load_dataset
from util import low_rank_approximation, low_rank_approximation_attn, low_rank_approximation_SVD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json

n_layer = 12
tolerant = 2  # loss increase ratio
dataset_times=json.load(open('./time_ori_gpu.json','r'))
steps={
        BertSdpaSelfAttention:16,
        BertSelfOutput:96,
        BertIntermediate:96,
        BertOutput:96
}
max_ranks={
        BertSdpaSelfAttention:32,
        BertSelfOutput:384,
        BertIntermediate:384,
        BertOutput:384
}

def get_tolerances(dataset_name):
    times=dataset_times[dataset_name]
    
    time_attn = times[BertSdpaSelfAttention.__name__]
    time_self_output = times[BertSelfOutput.__name__]
    time_intermediate = times[BertIntermediate.__name__]
    time_output = times[BertOutput.__name__]

    minimal_time = min(time_attn, time_self_output, time_intermediate, time_output)
    multiplier = (time_attn + time_self_output + time_intermediate + time_output) / minimal_time

    basic_tolerance = np.exp(np.log(tolerant) / multiplier)

    tol_attn = np.exp(np.log(basic_tolerance ** (time_attn / minimal_time)) / n_layer)
    tol_self_output = np.exp(np.log(basic_tolerance ** (time_self_output / minimal_time)) / n_layer)
    tol_intermediate = np.exp(np.log(basic_tolerance ** (time_intermediate / minimal_time)) / n_layer)
    tol_output = np.exp(np.log(basic_tolerance ** (time_output / minimal_time)) / n_layer)
    
    tols={
        BertSdpaSelfAttention:tol_attn,
        BertSelfOutput:tol_self_output,
        BertIntermediate:tol_intermediate,
        BertOutput:tol_output
    }
    return tols

model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", num_labels=2)
model.to("cuda")
tols=get_tolerances('sst2')
# Load dataset and prepare dataloader
BATCH_SIZE = 32
dataset = load_dataset("glue", "sst2", split="validation")
tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

def tokenize(batch):
    return tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

def collect_activations(model, dataloader, layer):
    activations = []
    def hook_fn(module, input, output):
        activations.append(input[0].detach())
        
    handle = layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(next(model.parameters()).device)
            model(inputs)

    handle.remove()
    X = torch.cat(activations, dim=0)
    if X.ndim > 2:
        X = X.permute(2, 0, 1, *range(3, X.ndim))
        X = X.reshape(X.shape[0], -1)
    else:
        X = X.T
    return X

def compress_attn_layer_by_optimal_rank(model, dataloader, layer, initial_rank, max_rank, allowed_loss, increment):
    rank = initial_rank

    Q = layer.query.weight.data.clone()
    K = layer.key.weight.data.clone()
    V = layer.value.weight.data.clone()
    bias_q = layer.query.weight.bias.clone()
    bias_k = layer.key.weight.bias.clone()
    bias_v = layer.value.weight.bias.clone()
    X = collect_activations(model, dataloader, layer.query)

    while rank <= max_rank:
        
        U_Q,V_Q,U_K,V_K,M_U,M_V,U_V,V_V= low_rank_approximation_attn(Q,K,V,X,rank,bias_q=bias_q,bias_k=bias_k)
        
        layer.query.weight.data = Q_approx
        layer.key.weight.data = K_approx

        new_loss = evaluate_model_loss(model, dataloader)
        
        if new_loss < allowed_loss:
            setattr(layer,"dense",LowRankLinear(U.T,V.T,layer.dense.bias.data.clone()))
            return rank
        else:
            # Revert weights if loss tolerance is exceeded and increment the rank
            layer.query.weight.data = original_Q
            layer.key.weight.data = original_K
            rank += increment
    return max_rank  # Return max_rank if loss could not be met within rank limit

def compress_linear_layer_by_optimal_rank(model, dataloader, layer, initial_rank, max_rank, allowed_loss, increment):
    rank = initial_rank
    original_weight = layer.dense.weight.data.clone()
    X = collect_activations(model, dataloader, layer)

    while True:
        U,V,_ = low_rank_approximation(original_weight, X, rank)
        layer.dense.weight.data = U@V
        
        new_loss = evaluate_model_loss(model, dataloader)
        
        if new_loss < allowed_loss:
            setattr(layer,"dense",LowRankLinear(U.T,V.T,layer.dense.bias.data.clone()))
            return rank
        else:
            layer.dense.weight.data = original_weight
            rank += increment
        if rank>max_rank: return False

def compress_layer(model, dataloader, layer, initial_rank, max_rank, allowed_loss):
    if isinstance(layer, BertSdpaSelfAttention): #attention layer
        optimal_rank = compress_attn_layer_by_optimal_rank(model, dataloader, layer, initial_rank, max_rank, allowed_loss,initial_rank)
    else:
        optimal_rank = compress_linear_layer_by_optimal_rank(model, dataloader, layer, initial_rank, max_rank, allowed_loss,initial_rank)
    
    if optimal_rank:
        print("Layer compressed with optimal rank ", optimal_rank)
        return True
    else:
        print("Can not find optimal rank within the loss tolerance")
        return False

def overall_low_rank_approximation(model, dataloader):
    layer_names = [name for name, _ in model.named_modules()]

    original_loss = evaluate_model_loss(model, dataloader)
    allowed_loss = original_loss
       
    for layer_name in layer_names:
        layer = dict(model.named_modules())[layer_name]
        if isinstance(layer, (BertSdpaSelfAttention,BertSelfOutput,BertIntermediate,BertOutput)):
            tol = tols[type(layer)]
            initial_rank = steps[type(layer)]
            max_rank = max_ranks[type(layer)]
        else :
            continue

        allowed_loss = original_loss * tol

        # Compress the layer
        print("------------------")
        print("Layer Name:", layer_name)
        print("allowed_loss", allowed_loss)
        result = compress_layer(model, dataloader, layer, initial_rank, max_rank, allowed_loss)
        
        # If compression succeeded, evaluate and check for loss tolerance
        if result:
            new_loss = evaluate_model_loss(model, dataloader)
            print("Original_loss:", original_loss)
            print("New_loss:", new_loss)
        else:
            print("Layer not compressed")

    return model


def evaluate_model_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_model_accuracy(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy

if __name__=='__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())

    acc = evaluate_model_accuracy(model, dataloader)
    print(f"Average accuracy of the model: {acc}")

    compressed_model = overall_low_rank_approximation(model, dataloader)
    compressed_model.save_pretrained('prone')
    acc = evaluate_model_accuracy(compressed_model, dataloader)
    print(f"Average accuracy of the compressed model: {acc}")
    model1=AutoModel.from_pretrained('prone')
