
from typing import List
import numpy as np


class BaseReranker:
    """
    Base class for reranker
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        raise NotImplementedError


class BgeReranker(BaseReranker):
    """
    class for Bge reranker
    """

    def __init__(self, path: str = 'BAAI/bge-reranker-base') -> None:
        super().__init__(path)
        self._model, self._tokenizer = self.load_model(path)

    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        import mindspore
        pairs = [(text, c) for c in content]

        inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors='ms', max_length=512)
        inputs = {k: v for k, v in inputs.items()}
        scores = self._model(**inputs, return_dict=True).logits.view(-1, ).float()
        index = np.argsort(scores.tolist())[-k:][::-1]
        return [content[i] for i in index]

    def load_model(self, path: str):

        from mindnlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.set_train(False)
        return model, tokenizer
