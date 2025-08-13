import torch
import torch.nn as nn
from .model import Transformer, RMSNorm

class LLaMA(nn.Module):
    def __init__(self, transformer, vocab_dim, embed_dim, n_heads, max_seq_len, n_layers, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_dim, embed_dim)
        
        self.blocks = nn.ModuleList([transformer(embed_dim, n_heads, max_seq_len, dropout) for _ in range(n_layers)])
        self.rms = RMSNorm(embed_dim)

        self.output = nn.Linear(embed_dim, vocab_dim, bias = False)

        self.embedding.weight = self.output.weight

    def forward(self, x, start_pos):
        x = self.embedding(x)
        for layer in self.blocks:
            x = layer(x, start_pos)
        x = self.rms(x)
        outputs = self.output(x)
        
        return outputs

    @torch.inference_mode()
    def generate(self, tokenizer, prompt, max_seq_len, device = "cuda"):
        self.eval()
        input_ids = tokenizer.encode(prompt, True, False)
        input_ids = torch.tensor([input_ids], dtype = torch.long, device = device)
        for i in range(max_seq_len):
            outputs = self(input_ids, i)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim = -1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim = 1)

        return tokenizer.decode(input_ids[0].tolist())

    
    


  
        
  
        
        
        
