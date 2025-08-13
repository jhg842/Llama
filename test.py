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
tok_path = ""

model = LLaMA(Transformer, 10000, 768, 12, 512, 16).to(device)
checkpoint = torch.load("", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
tokenizer = Tokenizer(tok_path)

prompt = ''

dataset = load_dataset("")
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