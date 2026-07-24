"""基于 Embedding + 单层 GRU 的轻量 caption 文本编码器。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
TEXT_FEATURE_DIM = 128


class TextEncoder(nn.Module):
    """将 caption 编码为 128 维文本特征。"""

    output_dim = TEXT_FEATURE_DIM

    def __init__(
        self,
        vocab: Dict[str, int],
        embedding_dim: int = 128,
        hidden_dim: int = TEXT_FEATURE_DIM,
        max_length: int = 32,
    ) -> None:
        super().__init__()
        if max_length < 2:
            raise ValueError("max_length 至少应为 2。")

        self.vocab = dict(vocab)
        self.max_length = max_length
        self.embedding = nn.Embedding(
            num_embeddings=len(self.vocab),
            embedding_dim=embedding_dim,
            padding_idx=self.vocab[PAD_TOKEN],
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.projection = (
            nn.Identity()
            if hidden_dim == self.output_dim
            else nn.Linear(hidden_dim, self.output_dim)
        )

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """使用简单正则表达式进行小写分词。"""

        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())

    @classmethod
    def build_vocab(
        cls,
        captions: Sequence[str],
        min_frequency: int = 1,
    ) -> Dict[str, int]:
        """从 caption 列表构建词表。"""

        counter: Counter[str] = Counter()
        for caption in captions:
            counter.update(cls.tokenize(caption))

        vocabulary = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3,
        }
        for token in sorted(counter):
            if counter[token] >= min_frequency:
                vocabulary[token] = len(vocabulary)
        return vocabulary

    @classmethod
    def from_captions(
        cls,
        captions: Sequence[str],
        embedding_dim: int = 128,
        hidden_dim: int = TEXT_FEATURE_DIM,
        max_length: int = 32,
        min_frequency: int = 1,
    ) -> "TextEncoder":
        """从 caption 列表构建词表并创建编码器。"""

        vocab = cls.build_vocab(captions, min_frequency=min_frequency)
        return cls(
            vocab=vocab,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_length=max_length,
        )

    def _caption_to_ids(self, caption: str) -> List[int]:
        token_ids = [self.vocab[BOS_TOKEN]]
        unknown_id = self.vocab[UNK_TOKEN]
        token_ids.extend(self.vocab.get(token, unknown_id) for token in self.tokenize(caption))
        token_ids.append(self.vocab[EOS_TOKEN])
        return token_ids[: self.max_length]

    def encode(self, captions: Sequence[str]) -> Tensor:
        """将 caption 列表转换为 padded token id，输出形状为 [batch, max_length]。"""

        if isinstance(captions, str):
            captions = [captions]
        if not captions:
            raise ValueError("captions 不能为空。")

        encoded = [self._caption_to_ids(caption) for caption in captions]
        token_ids = torch.full(
            (len(encoded), self.max_length),
            fill_value=self.vocab[PAD_TOKEN],
            dtype=torch.long,
        )
        for row, ids in enumerate(encoded):
            token_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return token_ids

    def forward(self, captions: Sequence[str] | Tensor) -> Tensor:
        """输入 caption 文本或 token ids，输出 [batch_size, 128]。"""

        if isinstance(captions, torch.Tensor):
            token_ids = captions
            if token_ids.ndim != 2:
                raise ValueError("token ids 的形状必须是 [batch_size, sequence_length]。")
            token_ids = token_ids.to(device=self.embedding.weight.device, dtype=torch.long)
        else:
            token_ids = self.encode(captions).to(self.embedding.weight.device)

        padding_id = self.vocab[PAD_TOKEN]
        lengths = (token_ids != padding_id).sum(dim=1).clamp(min=1).cpu()
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.projection(hidden[-1])


def main() -> None:
    """运行文本编码器的简单 shape 测试。"""

    captions = [
        "a fluffy white cat sitting on a chair",
        "a spotted cat standing indoors",
        "a brown dog looking at the camera",
    ]
    encoder = TextEncoder.from_captions(captions)
    token_ids = encoder.encode(captions)
    features = encoder(captions)

    print(f"token ids shape: {tuple(token_ids.shape)}")
    print(f"text feature shape: {tuple(features.shape)}")
    print(f"vocabulary size: {len(encoder.vocab)}")


if __name__ == "__main__":
    main()
