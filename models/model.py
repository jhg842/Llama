import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class ModelArgs:
    embed_dim: int = 512
    vocab_dim: int = 10000
    n_heads: int = 8
    n_layers: int = 8
    batch_size: int = 128
    max_seq_len: int = 512

class Transformer(nn.Module):
    def __init__(self,  embed_dim, n_heads, max_seq_len,dropout = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.attention = Attention(embed_dim, n_heads,max_seq_len, dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.ffn = feedforward(embed_dim)
        
        self.norm2 = RMSNorm(embed_dim)
        
        self.freqs_cis = precompute_freqs_cis(embed_dim // n_heads, 1024)

    def forward(self, x, start_pos, *args, **kwargs):
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, start_pos, self.freqs_cis)
        fx = attn_output + x
        norm_fx = self.norm2(fx)
        ffn_output = self.ffn(norm_fx)
        output = ffn_output + fx
        
        return output
        

class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_seq_len, dropout = 0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.n_kv_heads = self.n_heads // 2
        self.head_dim = embed_dim // n_heads
        
        self.wq = nn.Linear(embed_dim, embed_dim, bias = False)
        self.wk = nn.Linear(embed_dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(embed_dim, self.n_kv_heads * self.head_dim, bias = False)
        self.dropout = nn.Dropout(dropout)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        
        # self.register_buffer("tril", torch.tril(torch.ones(max_seq_len, max_seq_len)))
        # self.cache_k = torch.zeros(batch, max_seq, self.n_heads, self.embed_dim // self.n_heads)
        # self.cache_v = torch.zeros(batch, max_seq, self.n_heads, self.embed_dim // self.n_heads)
        
    def forward(self, x, start_pos, freqs_cis):
        bat, seq, dim = x.shape
       
        q = self.wq(x).view(bat, seq, self.n_heads, self.head_dim)
        k = self.wk(x).view(bat, seq, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bat, seq, self.n_kv_heads, self.head_dim)
        
        repeats_heads = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeats_heads, dim = 2)
        v = v.repeat_interleave(repeats_heads, dim = 2)
        
        q, k = rotary_embedding(q, k, freqs_cis[start_pos : start_pos + seq])
        # q, k = rotary_embedding(q, k, freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        outputs = F.scaled_dot_product_attention(q, k ,v, is_causal = True, dropout_p = self.dropout.p if self.training else 0.0)
        # attn_scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        
        # causal_mask = torch.tril(torch.ones(seq, seq, device=x.device, dtype = torch.bool))
        # causal_mask_float = torch.zeros(seq, seq, device=x.device, dtype = x.dtype).view(1, 1, seq, seq)
        # causal_mask_float.masked_fill_(~ causal_mask, float("-inf"))
        # attn_weights = attn_scores + causal_mask_float
  
        # attention_weights = F.softmax(attn_weights, dim=-1)
        # attention_weights = self.dropout(attention_weights)
        
        # outputs = attention_weights @ v
        outputs = outputs.transpose(1,2).reshape(bat, seq, self.embed_dim)
        
        return self.out_proj(outputs)
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
        
def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    return freqs_cis.view(*shape)
    
            
def rotary_embedding(q, k, freqs_cis):
    wq = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    wk = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, wq).to(wq.device)
    q_out = torch.view_as_real(wq * freqs_cis).flatten(3)
    k_out = torch.view_as_real(wk * freqs_cis).flatten(3)
    
    return q_out.type_as(q), k_out.type_as(k)
    
    
    

class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))
        
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

        return x * rms * self.weight
    
class feedforward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.intermediate_dim = int(embed_dim * (8 / 3))
        self.embed_dim = embed_dim
        self.weight1 = nn.Linear(embed_dim, self.intermediate_dim)
        self.weight2 = nn.Linear(self.intermediate_dim, embed_dim)
        self.weight3 = nn.Linear(embed_dim, self.intermediate_dim)
    
    def forward(self, x):
   
        return self.weight2(F.silu(self.weight1(x)) * self.weight3(x))



        # if cashe:
        #     q = self.q(x).view(bat, seq, self.n_heads, self.embed_dim // self.n_heads)
        #     k = self.k(x).view(bat, seq, self.n_heads, self.embed_dim // self.n_heads)
        #     v = self.v(x).view(bat, seq, self.n_heads, self.embed_dim // self.n_heads)
            
        #     q, k = rotary_embedding(q, k)
            
        #     self.cashe_k = self.cache_k.to(x.device)
        #     self.cache_v = self.cache_v.to(x.device)
            
        #     self.cashe_k[:, start_pos: seq + start_pos] = k
        #     self.cashe_v[:, start_pos: seq + start_pos] = v
            
        #     keys = self.cache_k[:, :seq + start_pos]
        #     values = self.cache_v[:, :seq + start_pos]
            
        #     q = q.transpose(1,2)
        #     keys = keys.transpose(1,2)
        #     values = values.transpose(1,2)
            
        #     scores = q @ keys.transpose(-2, -1) / (d_k ** 0.5)
            
        #     if mask is not None:
        #         scores = scores + mask
            
        #     scores = F.softmax(scores, dim = -1)
        #     outputs = scores @ values
            
        #     outputs = outputs.transpose(1, 2).contiguous().view(bat, seq, self.embed_dim)
            
        #     return outputs