from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset
from util import low_rank_approximation, low_rank_approximation_SVD
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
tolerant = 2.0  # Allowed total loss increase ratio

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

for name, module in model.named_modules():
    print(name)

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

def find_optimal_rank(model, dataloader, layer, initial_rank, max_rank, loss_tolerance, increment):
    """
    Find the optimal rank by incrementally increasing it until the desired loss tolerance is met.
    
    Parameters:
        model (nn.Module): The model being compressed.
        dataloader (DataLoader): The data loader for the evaluation dataset.
        layer (nn.Module): The layer to compress.
        initial_rank (int): The initial rank to start the search.
        max_rank (int): The maximum allowable rank.
        loss_tolerance (float): Allowed ratio increase in loss.
        increment (int): Rank increment per search iteration.
        
    Returns:
        int: Optimal rank satisfying the loss tolerance, or max_rank if tolerance cannot be met.
    """
    prev_loss = evaluate_model_loss(model, dataloader)
    rank = initial_rank

    while rank <= max_rank:
        # Attempt compression with the current rank
        original_weight = layer.weight.data.clone()
        X = collect_activations(model, dataloader, layer)
        W_approx = low_rank_approximation(layer.weight.data, X, rank)
        layer.weight.data = W_approx
        
        # Evaluate model loss after compression
        new_loss = evaluate_model_loss(model, dataloader)
        
        # Check if new loss is within tolerance
        if new_loss / prev_loss < 1 + loss_tolerance:
            return rank
        else:
            # Revert weights if loss tolerance is exceeded and increment the rank
            layer.weight.data = original_weight
            rank += increment

    return max_rank  # Return max_rank if tolerance could not be met within rank limit

def compress_layer(model, dataloader, layer_name, initial_rank, max_rank, loss_tolerance):
    """
    Compresses a given layer in the model by finding the optimal rank that respects the loss tolerance.
    
    Parameters:
        model (nn.Module): The model being compressed.
        dataloader (DataLoader): The data loader for the evaluation dataset.
        layer_name (str): The name of the layer to compress.
        initial_rank (int): Starting rank for compression.
        max_rank (int): Max allowable rank for compression.
        loss_tolerance (float): Allowed ratio increase in loss.

    Returns:
        nn.Module: The updated model with the specified layer compressed.
        bool: Success flag indicating if the layer was successfully compressed.
    """
    layer = dict(model.named_modules())[layer_name]
    if not hasattr(layer, 'weight'):
        return model, False
    
    optimal_rank = find_optimal_rank(
        model, dataloader, layer, initial_rank, max_rank, loss_tolerance, increment=16 if "Attention" in layer_name else 96
    )
    
    # Compress the layer with the determined optimal rank
    X = collect_activations(model, dataloader, layer)
    W_approx = low_rank_approximation(layer.weight.data, X, optimal_rank)
    layer.weight.data = W_approx
    return model, True

def revert_layer(model, layer_name, original_weights):
    """
    Revert a specific layer's weights to their original state.
    
    Parameters:
        model (nn.Module): The model with the layer to revert.
        layer_name (str): Name of the layer to revert.
        original_weights (dict): Dictionary containing the original weights of each layer.
        
    Returns:
        nn.Module: The model with the specified layer reverted.
    """
    layer = dict(model.named_modules())[layer_name]
    if hasattr(layer, 'weight') and layer_name in original_weights:
        layer.weight.data = original_weights[layer_name].clone()
    return model

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
    
    # Define initial and maximum ranks for each layer type
    initial_ranks = {
        "BertSdpaSelfAttention": 16,
        "BertSelfOutput": 96,
        "BertIntermediate": 96,
        "BertOutput": 96,
    }
    max_ranks = {
        "BertSdpaSelfAttention": 64,
        "BertSelfOutput": 768,
        "BertIntermediate": 768,
        "BertOutput": 768,
    }
    
    for layer_name in layer_names:
        # Determine module type and tolerance
        if "BertSdpaSelfAttention" in layer_name:
            allowed_loss_ratio = module_tolerance["BertSdpaSelfAttention"]
            initial_rank = initial_ranks["BertSdpaSelfAttention"]
            max_rank = max_ranks["BertSdpaSelfAttention"]
        elif "BertSelfOutput" in layer_name:
            allowed_loss_ratio = module_tolerance["BertSelfOutput"]
            initial_rank = initial_ranks["BertSelfOutput"]
            max_rank = max_ranks["BertSelfOutput"]
        elif "BertIntermediate" in layer_name:
            allowed_loss_ratio = module_tolerance["BertIntermediate"]
            initial_rank = initial_ranks["BertIntermediate"]
            max_rank = max_ranks["BertIntermediate"]
        else:
            allowed_loss_ratio = module_tolerance["BertOutput"]
            initial_rank = initial_ranks["BertOutput"]
            max_rank = max_ranks["BertOutput"]
        
        # Compress the layer
        print("------------------")
        print("Layer Name:", layer_name)
        model, result = compress_layer(model, dataloader, layer_name, initial_rank, max_rank, allowed_loss_ratio)
        
        # If compression succeeded, evaluate and check for loss tolerance
        if result:
            new_loss = evaluate_model_loss(model, dataloader)
            print("original_loss:", original_loss)
            print("new_loss:", new_loss)
            
            # Revert if loss tolerance exceeded
            if new_loss / original_loss >= 1 + allowed_loss_ratio:
                print(f"Compression of {layer_name} exceeded allowed loss ratio, reverting changes.")
                model = revert_layer(model, layer_name)
            else:
                print(f"Layer {layer_name} compressed within allowed loss ratio.")
                
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

acc = evaluate_model_accuracy(model, dataloader)
print(f"Average accuracy of the model: {acc}")

layer_names = [name for name, module in model.named_modules() if hasattr(module, 'weight')]

compressed_model = overall_low_rank_approximation(model, dataloader, layer_names, module_tolerance)
acc = evaluate_model_accuracy(compressed_model, dataloader)
print(f"Average accuracy of the compressed model: {acc}")
