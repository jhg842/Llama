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

    # os.environ['RANK'] = str(gpu_id)
    # os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
    # os.environ['LOCAL_RANK'] = str(gpu_id)
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12356'

    # current_rank, total_world_size, current_gpu_id, is_distributed_active = utils.init_distributed_mode(dist_url = 'env://', dist_backend='nccl')

    # checkpoint = "/home/jhg842/llama/checkpoint/gpt"
    checkpoint = "/home/work/llemr/checkpoint/llama"
    file_name = "checkpoint.pt"
    tok_path = "/home/work/llemr/sentencepiece/sentencepiece.model"
    # device = torch.device(f"cuda:{current_gpu_id}")
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
    # pe = positional_encoding(args.embed_dim, args.max_seq_len) #gpt
    # tokenizer = Tokenizer(args.vocab_dim, args.vocab_dim)
    # transformers = transformer(args.vocab_dim, args.embed_dim, args.n_heads, args.n_layers) # GPT
    # model = GPT(tokenizer, transformers, pe, args.vocab_dim, args.embed_dim, args.n_layers).to(device)

    # tokenizer = AutoTokenizer.from_pretrained("/home/work/llemr/tokenizer/huggingface_tokenizer")

    tokenizer = Tokenizer(tok_path)
    text = data_mode('train')

    encoded_text = tokenizer.encode(text, True, True)
    train_data = charDataset(encoded_text, args.max_seq_len)
    # sampler = DistributedSampler(train_data, num_replicas=total_world_size, rank = current_rank, shuffle = True)
    data_loader_train = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    
    
    # l_transformers = transformer(args.embed_dim, args.n_heads, args.max_seq_len) # LLaMA
    
    model = LLaMA(Transformer, args.vocab_dim, args.embed_dim, args.n_heads, args.max_seq_len, args.n_layers).to(device) # LLaMA
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_gpu_id] )

    total_params = sum(p.numel() for p in model.parameters())

    print(f"모델의 전체 파라미터 수: {total_params:,}개") # 콤마로 보기 좋게 출력
    epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print('Starting LLaMA model training...')
    for epoch in tqdm(range(epochs), desc = "Training Epochs"):
        model.train()
        # sampler.set_epoch(epoch)
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

def test():

    checkpoint = "/home/work/llemr/checkpoint/llama/"
    file_name = "best_checkpoint.pt"
    tok_path = "/home/work/llemr/sentencepiece/sentencepiece.model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = ModelArgs(
        vocab_dim = 10000,
        embed_dim = 512,
        n_layers = 8,
        n_heads = 8,
        batch_size = 8,
        max_seq_len = 256,
    )
    text = data_mode('test')
    # text = "회사 관계자는 이번 특허는 대표적인 차량 내적 사고 요인인"
    checkpoint_path = checkpoint + file_name
    checkpoint = torch.load(checkpoint_path, map_location=device)

    l_transformers = transformer(args.embed_dim, args.n_heads, args.max_seq_len) # LLaMA
    model = LLaMA(l_transformers, args.vocab_dim, args.embed_dim, args.n_layers).to(device) # LLaMA

    model.load_state_dict(checkpoint['model_state_dict'])

    tokenizer = Tokenizer(tok_path)

    new_text = model.generate(tokenizer, text, args.max_seq_len)
    print(new_text)
    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())
    # parsed_args = parser.parse_args() 

    # args = ModelArgs(
    #     vocab_dim = 10000,
    #     embed_dim = 512,
    #     n_layers = 8,
    #     n_heads = 8,
    #     batch_size = 128,
    #     max_seq_len = 256,
    # )

    main()
    # mp.spawn(
    #     main,
    #     nprocs = torch.cuda.device_count(),
    #     join = True,
    # )
    # test()
