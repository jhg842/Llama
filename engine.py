import torch
import os
import torch.nn as nn

from llama import LLaMA, Tokenizer, Transformer
from datasets import load_dataset


def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    avg_loss: float,
    checkpoint_path: str,
    file_name: str,
    is_best: bool = True,
):
    os.makedirs(checkpoint_path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'hyperparameters': {
            'lr': optimizer.param_groups[0]['lr'],
        }
    }
    
    if is_best:
        file_path = os.path.join(checkpoint_path, f"best_{file_name}")
    else:
        file_path = os.path.join(checkpoint_path, file_name)
        
    torch.save(checkpoint, file_path)
    
    if epoch % 5 == 0:
        epoch_file_path = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, epoch_file_path)

def sample_next_token(logits, temperature=0.5, top_k=50):
    # temperature 조정
    logits = logits / temperature

    # top-k 적용
    top_k = min(top_k, logits.size(-1))  # safety
    values, indices = torch.topk(logits, top_k)
    probs = torch.nn.functional.softmax(values, dim=-1)

    # 샘플링
    next_token = torch.multinomial(probs, num_samples=1)
    return indices[0][next_token]

@torch.inference_mode()
def generate(model, tokenizer, prompt, max_seq_len, device = "cuda"):
    model.eval()
    
    input_ids = tokenizer.encode(prompt, True, False)
    input_ids = torch.tensor(input_ids, dtype = torch.long, device = device)
    prompt_length = len(input_ids)
    print(prompt_length)
    input_ids = input_ids.unsqueeze(0)

    for i in range(max_seq_len - prompt_length):
        with torch.no_grad():
            outputs = model(input_ids, i)
            next_token_logits = outputs[:, -1, :]
            next_token = sample_next_token(next_token_logits, temperature = 0.7, top_k = 50)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())

