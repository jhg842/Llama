import evaluate
from engine import generate
import torch
from llama import LLaMA, Tokenizer, Transformer
from datasets import load_dataset
import numpy as np

def eval(bleu, rouge, bert_score ,pred, ref):

    bleu_result = bleu.compute(predictions = pred, references = ref)
    rouge_result = rouge.compute(predictions = pred, references = ref)
    bert_result = bertscore.compute(predictions = pred, references = ref, lang = 'ko')

    result = {'bleu': bleu_result['bleu'],
    'rouge1': rouge_result['rouge1'],
    'rouge2': rouge_result['rouge2'],
    'rougeL': rouge_result['rougeL'],
    'bert_score': np.mean(bert_result['f1'])}
    

    return result

bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')



device = 'cuda'
tok_path = "/home/work/llemr/sentencepiece/sentencepiece.model"

model = LLaMA(Transformer, 10000, 768, 12, 512, 16).to(device)
checkpoint = torch.load("/home/work/llemr/checkpoint/llama/best_checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
tokenizer = Tokenizer(tok_path)

prompt = '아이엘사이언스의 자회사 아이트로닉스는 차량용 복합기능형 졸음 방지 단말기 특허를 출원했다고 4일 밝혔다. 신규 특허는 자동차 주행 중 운전자의 졸음운전을 방지하는 상태 검출 기술에 관한 것이다.'

dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")
true_text = "".join(dataset['test']['document'][0:2])
print(true_text)

generate_text = generate(
    model,
    tokenizer,
    prompt,
    max_seq_len = 512
)
generate_text = generate_text[:1020]
true_text = true_text[:1020]
scores = eval(bleu, rouge, bertscore, [generate_text], [[true_text]])
print(scores)