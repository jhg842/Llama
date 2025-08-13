from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os
from datasets import load_dataset

logger = getLogger()

class Tokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file = model_path)
        logger.info(f"Reloaded SetencePiece model from {model_path}")

        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - Bos ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s:str, bos: bool, eos: bool)-> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, t: List[int])-> str:

        return self.sp_model.decode(t)


# model = Tokenizer('sentencepiece.model')
# dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")
# text = dataset['train']['document'][0]
# encoded = model.encode(text, True,True)
# decoded = model.decode(encoded)
# print(encoded, decoded)
# with open("train.text", "w", encoding='utf-8')as f:
#     for line in texts:
#         f.write(line.strip() + "\n")

# import sentencepiece as spm
# spm_train_command = (
#     f"--input=train.text "
#     f"--model_prefix=sentencepiece "
#     f"--vocab_size=10000 "
#     f"--model_type=unigram " # 여기 쉼표 제거
#     f"--character_coverage=1.0" # 여기도 쉼표 제거
# )
# spm.SentencePieceTrainer.Train(spm_train_command)

