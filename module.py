import torch
import torch.nn as nn
from util import low_rank_approximation,low_rank_approximation_attn
from transformers.models.bert.modeling_bert import BertSdpaSelfAttention
from typing import Optional,Tuple
import math
from transformers import AutoConfig

class LowRankLinear(nn.Module):
    def __init__(self, U, V, bias):
        super(LowRankLinear, self).__init__()
        self.U = nn.Parameter(U)  
        self.V = nn.Parameter(V) 
        self.bias = nn.Parameter(bias)   

    def forward(self, x):
        return x@self.V@self.U + self.bias

class LowRankAttention(nn.Module):
    def __init__(self, U_Q,V_Q,U_K,V_K,U_V,V_V,bias_q,bias_k,bias_v,M_U,M_V):
        super(LowRankAttention, self).__init__()
        self.embed_size=V_Q.shape[0]
        self.num_heads = 12
        self.head_dim = self.embed_size // self.num_heads

        self.Q = LowRankLinear(U_Q,V_Q,bias_q)
        self.K = LowRankLinear(U_K,V_K,bias_k)
        self.V = LowRankLinear(U_V,V_V,bias_v)
        self.M_U = nn.Parameter(M_U)
        self.M_V = nn.Parameter(M_V)
        self.bias_q=bias_q
        self.bias_k=bias_k
        self.bias_v=bias_v
        
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        N, seq_length, embed_size = hidden_states.shape
        scale_factor = 1 / math.sqrt(self.head_dim)
        queries = self.Q(hidden_states)
        keys = self.K(hidden_states)
        values = self.V(hidden_states)

        queries = queries.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_bias = torch.zeros(N,1,seq_length, seq_length, dtype=queries.dtype,device=queries.device)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attention_mask
                
        attn_weight = (queries @self.M_U) @ (self.M_V @ keys.transpose(-2, -1)) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout(attn_weight)
        out=attn_weight @ values
        out = out.transpose(1, 2).contiguous().view(N, seq_length, self.embed_size)

        return (out,)


class model(nn.Module):
    def __init__(self) -> None:
        super(model,self).__init__()
        self.linear=None

    def forward(self,x):
        return self.linear(x)

if __name__ == '__main__':
    # test linear
    X = torch.randn((128, 92)) @ torch.randn((92, 768))
    W = torch.randn((768, 768))
    bias = torch.randn(768)
    rank = 92

    # torch linear
    with torch.no_grad():
        linear_layer = nn.Linear(W.shape[1], W.shape[0])
        linear_layer.weight.copy_(W)
        linear_layer.bias.copy_(bias)
        y_linear = linear_layer(X)
    
    # LowRankLinear layer
    U_drone, V_drone, _ = low_rank_approximation(W, X.T, rank)
    low_rank_layer = LowRankLinear(U_drone.T, V_drone.T, bias)
    y_low_rank = low_rank_layer(X)

    # error
    error = torch.mean(torch.abs(y_low_rank - y_linear))
    print("Error between LowRankLinear and nn.Linear outputs:", error.item())

    # test attention
    X = torch.randn((128, 32)) @ torch.randn((32, 768)).unsqueeze(0)
    Q = torch.randn((768, 768))
    K = torch.randn((768, 768))
    V = torch.randn((768, 768))
    bias_q = torch.zeros(768)
    bias_k = torch.zeros(768)
    bias_v = torch.randn(768)
    rank = 32

    # torch attention
    layer = BertSdpaSelfAttention(AutoConfig.from_pretrained('bert-base-uncased'),'absolute')
    layer.eval()
    with torch.no_grad():
        layer.query.weight.copy_(Q)
        layer.query.bias.copy_(bias_q)
        layer.key.weight.copy_(K)
        layer.key.bias.copy_(bias_k)
        layer.value.weight.copy_(V)
        layer.value.bias.copy_(bias_v)
        y=layer(X)[0]
        
    # low rank attention
    U_Q,V_Q,U_K,V_K,M_U,M_V,U_V,V_V=low_rank_approximation_attn(Q,K,V,X.squeeze().T,rank,bias_q=bias_q,bias_k=bias_k)
    low_rank_layer = LowRankAttention(U_Q.T,V_Q.T,U_K.T,V_K.T,U_V.T,V_V.T,bias_q,bias_k,bias_v,M_U,M_V)
    low_rank_layer.eval()
    y_low_rank = low_rank_layer(X)[0]

    # error
    error = torch.mean(torch.abs(y_low_rank - y))
    print("Error between LowRankAttention and BertSdpaSelfAttention outputs:", error.item())
    
    # import timeit
    # def a():
    #     y = layer(X)
    
    # def b():
    #     y_low_rank = low_rank_layer(X)

    # t1=timeit.timeit(a,number=50)
    # t2=timeit.timeit(b,number=50)
    # print(t1,t2)