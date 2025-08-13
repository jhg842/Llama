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

# device = 'cuda'
# tok_path = "/home/work/llemr/sentencepiece/sentencepiece.model"

# model = LLaMA(Transformer, 10000, 768, 12, 512, 16).to(device)
# checkpoint = torch.load("/home/work/llemr/checkpoint/llama/best_checkpoint.pt", map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# tokenizer = Tokenizer(tok_path)

# prompt = "아이엘사이언스의 자회사 아이트로닉스는 차량용 복합기능형 졸음 방지 단말기 특허를 출원했다고 4일 밝혔다. 신규 특허는 자동차 주행 중 운전자의 졸음운전을 방지하는 상태 검출 기술에 관한 것이다. 해당 단말기는 가시광선 및 근적외선 광원을 조사하는 광원 모듈 운전자의 얼굴 영상을 촬영하는 가시광선 및 근적외선 카메라 차량 실내의 이산화탄소 농도를 측정하는 이산화탄소 센서로 구성됐다. 단말기는 광원에 반응하는 운전자의 얼굴 촬영 영상을 기반으로 심박 데이터와 눈의 깜빡임 횟수 눈을 감은 시간 등을 측정한다. 여기에 차내 졸음을 유발하는 이산화탄소 농도까지 종합적으로 분석해 운전자의 졸음 상태를 판단하고 결과값에 따라 경보 신호를 송출하도록 설계됐다. 아이트로닉스는"
# prompt = "아이엘사이언스의 자회사 아이트로닉스는 차량용 복합기능형 졸음 방지 단말기 특허를 출원했다고 4일 밝혔다. 신규 특허는 자동차 주행 중 운전자의 졸음운전을"
# generate_text = generate(
#     model,
#     tokenizer,
#     prompt,
#     max_seq_len = 512
# )
# print(generate_text)

# dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")
# text = "".join(dataset['test']['document'][0])
# print(text)
