import argparse
import os
import sys
import numpy as np
import random
import argparse
# from gpt import GPT, Tokenizer, transformer, ModelArgs, positional_encoding

from llama import LLaMA, ModelArgs, Transformer, Tokenizer

from transformers import AutoTokenizer

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from dataset import charDataset, data_mode
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from engine import save_checkpoint
# import utils

def main():

    checkpoint = "/home/work/llemr/checkpoint/llama"
    file_name = "checkpoint.pt"
    tok_path = "/home/work/llemr/sentencepiece/sentencepiece.model"
    device = "cuda"
    seed = 777
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    args = ModelArgs(
        vocab_dim = 10000,
        embed_dim = 768,
        n_layers = 16,
        n_heads = 12,
        batch_size = 16,
        max_seq_len = 512,
    )


    tokenizer = Tokenizer(tok_path)
    text = data_mode('train')

    encoded_text = tokenizer.encode(text, True, True)
    train_data = charDataset(encoded_text, args.max_seq_len)
    data_loader_train = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    
    
    
    model = LLaMA(Transformer, args.vocab_dim, args.embed_dim, args.n_heads, args.max_seq_len, args.n_layers).to(device) # LLaMA

    total_params = sum(p.numel() for p in model.parameters())

    print(f"모델의 전체 파라미터 수: {total_params:,}개") 
    epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print('Starting LLaMA model training...')
    for epoch in tqdm(range(epochs), desc = "Training Epochs"):
        model.train()
        total_loss = 0.0
        
        for x, y in data_loader_train:
            
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x,0)

            bat, seq_len, vocab_dim = outputs.shape
            outputs = outputs.reshape(bat * seq_len, vocab_dim)
            y = y.reshape(bat * seq_len)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        total_loss /= len(data_loader_train)
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
        save_checkpoint(epoch, model, optimizer, total_loss, checkpoint, file_name, True)
        
    print("Training complete!")      
    dist.destroy_process_group()  

    
if __name__ == '__main__':


    main()

