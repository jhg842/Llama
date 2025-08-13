import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from llama import Tokenizer

class charDataset(Dataset):
    def __init__(self, encoded_text, block_size):
        
        self.encoded_text = encoded_text
        
        self.block_size = block_size
        self._length = len(self.encoded_text) - self.block_size
        
    def __len__(self):
      
      return self._length
    
    def __getitem__(self, idx):
        x = self.encoded_text[idx:idx + self.block_size]
        y = self.encoded_text[idx + 1: idx + self.block_size + 1]
        
        # x_tensor = torch.tensor(x, dtype = torch.long)
        # y_tensor = torch.tensor(y, dtype = torch.long)
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)      

def data_mode(mode):
    if mode == 'train':
        dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")
        text = "".join(dataset['train']['document'])
        # text = dataset['train']['document']
    elif mode == 'valid':
        dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")
        text = "".join(dataset['validation']['document'])
    elif mode == 'test':
        dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")
        text = "".join(dataset['test']['document'][0])
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")
    
    return text






    