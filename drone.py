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

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset
from util import low_rank_approximation, low_rank_approximation_attn, low_rank_approximation_SVD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Timing values for each module (BertSdpaSelfAttention, BertSelfOutput, BertIntermediate, BertOutput)
time_attn = 117.5
time_self_output = 34.27
time_intermediate = 133.11
time_output = 128.84
n_layer = 12
tolerant = 1.2  # Allowed total loss increase ratio


# Minimal time for the fastest module
minimal_time = min(time_attn, time_self_output, time_intermediate, time_output)
multiplier = (time_attn + time_self_output + time_intermediate + time_output) / minimal_time

# Base tolerance calculation
basic_tolerance = np.exp(np.log(tolerant) / multiplier)

# Tolerance values for each module
tol_attn = np.exp(np.log(basic_tolerance ** (time_attn / minimal_time)) / n_layer)
tol_self_output = np.exp(np.log(basic_tolerance ** (time_self_output / minimal_time)) / n_layer)
tol_intermediate = np.exp(np.log(basic_tolerance ** (time_intermediate / minimal_time)) / n_layer)
tol_output = np.exp(np.log(basic_tolerance ** (time_output / minimal_time)) / n_layer)

#test = tol_attn * tol_self_output * tol_intermediate * tol_output
#print("test", test ** n_layer)  #should be equal to tolerant

# Tolerance mapping for different types of layers
module_tolerance = {
    "BertSdpaSelfAttention": tol_attn,
    "BertSelfOutput": tol_self_output,
    "BertIntermediate": tol_intermediate,
    "BertOutput": tol_output,
}

# Load and move model to GPU
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2", num_labels=2)
model.to("cuda")

# Load dataset and prepare dataloader
BATCH_SIZE = 32
dataset = load_dataset("glue", "sst2", split="validation")
tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

def tokenize(batch):
    return tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Function definitions remain the same
def collect_activations(model, dataloader, layer):
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach())

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
    """
    Find the optimal rank for Q and K in attn layer

    Parameters:
        model (nn.Module): The model being compressed.
        dataloader (DataLoader): The data loader for the evaluation dataset.
        layer (nn.Module): The layer to compress.
        initial_rank (int): The initial rank to start the search.
        max_rank (int): The maximum allowable rank.
        allowed_loss (float): Allowed loss.
        increment (int): Rank increment per search iteration.

    Returns:
        int: Optimal rank satisfying the loss tolerance, or max_rank if tolerance cannot be met.
    """

    rank = initial_rank

    original_Q = layer.query.weight.data.clone()
    original_K = layer.key.weight.data.clone()

    Y_q = collect_activations(model, dataloader, layer.query)
    Y_k = collect_activations(model, dataloader, layer.key)

    while rank <= max_rank:
        # Attempt compression with the current rank
        Q_approx, K_approx = low_rank_approximation_attn(layer.query.weight.data, layer.key.weight.data, Y_q, Y_k, rank)

        layer.query.weight.data = Q_approx
        layer.key.weight.data = K_approx

        # Evaluate model loss after compression
        new_loss = evaluate_model_loss(model, dataloader)

        # Check if new loss is within allowed range
        if new_loss < allowed_loss:
            return rank
        else:
            # Revert weights if loss tolerance is exceeded and increment the rank
            layer.query.weight.data = original_Q
            layer.key.weight.data = original_K
            rank += increment
    return max_rank  # Return max_rank if loss could not be met within rank limit



def compress_layer_by_optimal_rank(model, dataloader, layer, initial_rank, max_rank, allowed_loss, increment):
    """
    Find the optimal rank by incrementally increasing it until the desired loss tolerance is met.
    
    Parameters:
        model (nn.Module): The model being compressed.
        dataloader (DataLoader): The data loader for the evaluation dataset.
        layer (nn.Module): The layer to compress.
        initial_rank (int): The initial rank to start the search.
        max_rank (int): The maximum allowable rank.
        allowed_loss (float): Allowed loss.
        increment (int): Rank increment per search iteration.
        
    Returns:
        int: Optimal rank satisfying the loss tolerance, or max_rank if tolerance cannot be met.
    """

    rank = initial_rank

    original_weight = layer.weight.data.clone()
    X = collect_activations(model, dataloader, layer)

    while rank <= max_rank:
        # Attempt compression with the current rank

        W_approx = low_rank_approximation(layer.weight.data, X, rank)
        layer.weight.data = W_approx
        
        # Evaluate model loss after compression
        new_loss = evaluate_model_loss(model, dataloader)
        
        # Check if new loss is within tolerance
        if new_loss < allowed_loss:
            return rank
        else:
            # Revert weights if loss tolerance is exceeded and increment the rank
            layer.weight.data = original_weight
            rank += increment

    return max_rank  # Return max_rank if tolerance could not be met within rank limit

def compress_layer(model, dataloader, layer, initial_rank, max_rank, allowed_loss):
    """
    Compresses a given layer in the model by finding the optimal rank that respects the loss tolerance.
    
    Parameters:
        model (nn.Module): The model being compressed.
        dataloader (DataLoader): The data loader for the evaluation dataset.
        layer_name (str): The name of the layer to compress.
        initial_rank (int): Starting rank for compression.
        max_rank (int): Max allowable rank for compression.
        allowed_loss (float): Allowed loss for compression.

    Returns:
        nn.Module: The updated model with the specified layer compressed.
        bool: Success flag indicating if the layer was successfully compressed.
    """
    if isinstance(layer, BertSdpaSelfAttention): #attention layer
        optimal_rank = compress_attn_layer_by_optimal_rank(
            model, dataloader, layer, initial_rank, max_rank, allowed_loss,
            increment=96
        )
    else:
        if not hasattr(layer, 'weight'):
            print("Skipping layer: No weight in this layer")
            return False

        W = layer.weight.data
        X = collect_activations(model, dataloader, layer)

        if  W.ndim < 2 or W.shape[1] != X.shape[0]:
            print("Skipping layer: Invalid weight shape")
            return False

        optimal_rank = compress_layer_by_optimal_rank(
            model, dataloader, layer, initial_rank, max_rank, allowed_loss,
            increment=96
        )
    if optimal_rank == max_rank:
        print("Can not find optimal rank within the loss tolerance")
        return False
    else:
        print("Layer compressed with optimal rank ", optimal_rank)
        return True

def overall_low_rank_approximation(model, dataloader, layer_names, module_tolerance):
    """
    Apply low-rank approximation across specified layers while keeping loss increase within allowed tolerance.
    
    Parameters:
        model (nn.Module): The model to compress.
        dataloader (DataLoader): DataLoader for the dataset used in evaluating loss.
        layer_names (list): List of layer names to apply compression.
        module_tolerance (dict): Dictionary specifying the allowed loss tolerance for each module type.
        
    Returns:
        nn.Module: The compressed model.
    """
    original_loss = evaluate_model_loss(model, dataloader)
    allowed_loss = original_loss
    # Define initial and maximum ranks for each layer type
    initial_ranks = {
        "BertSdpaSelfAttention": 96,
        "BertSelfOutput": 96,
        "BertIntermediate": 96,
        "BertOutput": 96,
    }
    max_ranks = {
        "BertSdpaSelfAttention": 768,
        "BertSelfOutput": 768,
        "BertIntermediate": 768,
        "BertOutput": 768,
    }
    
    for layer_name in layer_names:
        # Determine module type and tolerance
        layer = dict(model.named_modules())[layer_name]
        if isinstance(layer, BertSdpaSelfAttention):
            tol = module_tolerance["BertSdpaSelfAttention"]
            initial_rank = initial_ranks["BertSdpaSelfAttention"]
            max_rank = max_ranks["BertSdpaSelfAttention"]
        elif isinstance(layer, BertSelfOutput):
            tol = module_tolerance["BertSelfOutput"]
            initial_rank = initial_ranks["BertSelfOutput"]
            max_rank = max_ranks["BertSelfOutput"]
        elif isinstance(layer, BertIntermediate):
            tol = module_tolerance["BertIntermediate"]
            initial_rank = initial_ranks["BertIntermediate"]
            max_rank = max_ranks["BertIntermediate"]
        elif isinstance(layer, BertOutput):
            tol = module_tolerance["BertOutput"]
            initial_rank = initial_ranks["BertOutput"]
            max_rank = max_ranks["BertOutput"]
        elif layer_name.endswith("embeddings") or layer_name.endswith("attention.self.key") or layer_name.endswith("attention.self.query"):
            continue
        else: #try to compress
            tol = 1
            initial_rank = initial_ranks["BertOutput"]
            max_rank = max_ranks["BertOutput"]

        allowed_loss = allowed_loss * tol

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
    loss_fn = torch.nn.CrossEntropyLoss()
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

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

acc = evaluate_model_accuracy(model, dataloader)
print(f"Average accuracy of the model: {acc}")

layer_names = [name for name, module in model.named_modules()]

compressed_model = overall_low_rank_approximation(model, dataloader, layer_names, module_tolerance)
acc = evaluate_model_accuracy(compressed_model, dataloader)
print(f"Average accuracy of the compressed model: {acc}")
