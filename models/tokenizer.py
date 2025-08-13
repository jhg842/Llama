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




