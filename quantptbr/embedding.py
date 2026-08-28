"""Envelope de embedding congelado: os parâmetros que definem os vetores.

Trocar qualquer campo deste envelope produz vetores diferentes, logo um estudo
diferente. Ele é medido em S1.4, escrito em disco, e o passe completo de S1.5 o
*lê* em vez de redigitá-lo — é isso que impede que piloto e passe real divirjam
num parâmetro sem que ninguém perceba.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from quantptbr.corpus import REPO_ROOT

#: Escolhido em S1.4 por um motivo verificável: é o único recuperador denso de
#: estágio único com nDCG@10 publicado sobre o Quati 1M (0,3955, Tabela 7 de
#: Bueno et al.), nas mesmas 50 consultas que usamos. Esse número é alvo de
#: calibração do braço A0 — não comparação científica, e sim prova de que o
#: instrumento mede certo antes de medir quantização.
MODEL_ID = "intfloat/multilingual-e5-base"

#: Fixado no commit do repositório do modelo, não em "main". `main` se move.
MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"

DIMENSIONS = 768

#: Limite duro do modelo: XLM-R base tem 514 posições, das quais 512 utilizáveis.
#: Não é um parâmetro a sintonizar — é o teto que `max_length` não pode cruzar.
MODEL_MAX_LENGTH = 512

#: O E5 é treinado com estes prefixos e **degrada em silêncio** sem eles. Ficam no
#: envelope, e não soltos no código, porque são parte do que define o vetor.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

ENVELOPE_PATH = REPO_ROOT / "config" / "embedding_envelope.json"


@dataclass(frozen=True)
class Envelope:
    """Todo parâmetro que altera os vetores produzidos."""

    model_id: str
    model_revision: str
    dimensions: int
    max_length: int
    batch_size: int
    dtype: str
    pooling: str
    normalize: bool
    query_prefix: str
    passage_prefix: str

    def save(self, path: Path = ENVELOPE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path = ENVELOPE_PATH) -> Envelope:
        if not path.exists():
            raise FileNotFoundError(f"{path} ausente. Rode scripts/02_pilot_embeddings.py (S1.4).")
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def load_model(dtype: str, max_length: int):
    """Carrega o BGE-M3 denso na revisão fixada, em `cuda`.

    Importa `torch`/`sentence_transformers` aqui dentro para que os módulos que
    só precisam do envelope não paguem alguns segundos de import de GPU.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cuda",
        model_kwargs={"dtype": getattr(torch, dtype)},
    )
    model.max_seq_length = max_length

    # O pipeline denso é Transformer → Pooling → Normalize. Se o modules.json do
    # modelo mudar e trouxer outra cabeça junto, isto para o passe antes que
    # vetores de outra natureza entrem no artefato.
    names = [type(m).__name__ for m in model]
    if names != ["Transformer", "Pooling", "Normalize"]:
        raise RuntimeError(f"pipeline inesperado do modelo: {names}")
    if model.get_embedding_dimension() != DIMENSIONS:
        raise RuntimeError(f"dimensão {model.get_embedding_dimension()} != {DIMENSIONS} esperada")
    return model


def pooling_mode(model) -> str:
    """O modo de pooling efetivo, lido do módulo em vez de presumido."""
    return model[1].get_config_dict()["pooling_mode"]
