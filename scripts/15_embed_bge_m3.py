#!/usr/bin/env python3
"""Vetores do BGE-M3 para a trilha do documento de arquitetura.

O documento de arquitetura fixa o `BAAI/bge-m3`; o estudo do servidor usou o
E5-base. Este script roda o mesmo passe de embedding de `scripts/03` — mesma
ordem do corpus, mesmos blocos retomáveis, mesma renormalização em float32 —
trocando só o envelope. O resultado alimenta `scripts/14_exhaustive.py`:

    python scripts/15_embed_bge_m3.py
    python scripts/14_exhaustive.py --label bge-m3 \\
        --corpus data/vectors/bge_m3_1m_float32.npy \\
        --queries data/vectors/bge_m3_queries_float32.npy

**O envelope fica aqui, e não em `config/embedding_envelope.json`.** Aquele
arquivo é o envelope congelado do estudo, amarrado por teste à evidência que o
produziu; trocá-lo invalidaria a trilha do servidor. Este é gravado inteiro no
manifesto da saída, que é o que torna os vetores rastreáveis.

O BGE-M3 não usa prefixo de consulta nem de passagem, e o pooling é pelo token
CLS — lido do próprio `1_Pooling/config.json` do modelo na revisão fixada.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np

from quantptbr import corpus, embedding, vectors

ENVELOPE = embedding.Envelope(
    model_id="BAAI/bge-m3",
    #: Fixado no commit do repositório do modelo, não em "main".
    model_revision="5617a9f61b028005a4858fdac845db406aefb181",
    dimensions=1024,
    #: Mesmo teto do E5: as passagens do Quati têm ~1.000 caracteres, bem abaixo.
    max_length=512,
    #: Menor que os 32 do E5 para caber nos 6 GB de uma RTX 4050 com o modelo large.
    batch_size=16,
    dtype="float16",
    pooling="cls",
    normalize=True,
    query_prefix="",
    passage_prefix="",
)

ARRAY_PATH = vectors.VECTORS_DIR / "bge_m3_1m_float32.npy"
PROGRESS_PATH = vectors.VECTORS_DIR / "bge_m3_progress.json"
QUERIES_PATH = vectors.VECTORS_DIR / "bge_m3_queries_float32.npy"
MANIFEST_PATH = corpus.REPO_ROOT / "runs" / "s1.5_vectors_bge_m3.json"


def check(array: np.ndarray) -> float:
    """Recusa linha não finita ou não unitária; devolve o pior erro de norma."""
    worst = 0.0
    for start in range(0, len(array), vectors.CHUNK_SIZE):
        block = np.asarray(array[start : start + vectors.CHUNK_SIZE], dtype=np.float32)
        if not np.isfinite(block).all():
            raise RuntimeError(f"linhas não finitas no bloco que começa em {start:,}")
        worst = max(worst, float(np.abs(np.linalg.norm(block, axis=1) - 1).max()))
    if worst > 1e-5:
        raise RuntimeError(f"vetores não unitários: erro de norma {worst:.2e}")
    return worst


def embed_queries() -> np.ndarray:
    topics = corpus.load_topics()
    model = embedding.load_model(
        ENVELOPE.dtype,
        ENVELOPE.max_length,
        model_id=ENVELOPE.model_id,
        revision=ENVELOPE.model_revision,
        dimensions=ENVELOPE.dimensions,
    )
    try:
        raw = model.encode(
            [ENVELOPE.query_prefix + topics[q] for q in corpus.judged_query_ids()],
            batch_size=ENVELOPE.batch_size,
            normalize_embeddings=ENVELOPE.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    finally:
        del model
    return vectors.normalise(raw)


def main() -> int:
    corpus.verify_manifest()
    # Lido antes do passe de horas: chamar o git no fim, com a RAM cheia, já
    # falhou ao criar o processo.
    commit = vectors.git_commit()
    print(f"envelope: {ENVELOPE.model_id} @ {ENVELOPE.model_revision[:8]} ({ENVELOPE.dimensions}d)")

    timing = vectors.run_pass(ENVELOPE, ARRAY_PATH, PROGRESS_PATH)
    array = np.load(ARRAY_PATH, mmap_mode="r")
    if array.shape != (corpus.EXPECTED_PASSAGES, ENVELOPE.dimensions):
        raise RuntimeError(f"forma {array.shape} inesperada")
    worst_norm_error = check(array)

    queries = embed_queries()
    np.save(QUERIES_PATH, queries)

    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "finished_at": datetime.now(UTC).isoformat(),
                "git_commit": commit,
                "corpus_sha256": corpus.read_manifest()["files"]["corpus"]["sha256"],
                "envelope": ENVELOPE.__dict__,
                "array": {
                    "path": str(ARRAY_PATH.relative_to(corpus.REPO_ROOT)),
                    "shape": list(array.shape),
                    "sha256": corpus.sha256(ARRAY_PATH),
                    "worst_norm_error": worst_norm_error,
                },
                "queries": {
                    "path": str(QUERIES_PATH.relative_to(corpus.REPO_ROOT)),
                    "shape": list(queries.shape),
                    "sha256": corpus.sha256(QUERIES_PATH),
                },
                "timing": timing,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n→ {ARRAY_PATH}\n→ {QUERIES_PATH}\n→ {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
