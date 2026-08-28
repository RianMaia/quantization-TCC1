#!/usr/bin/env python3
"""S1.6 — Codifica as consultas julgadas sob o mesmo envelope do corpus.

Consulta e passagem precisam ser codificadas de forma consistente entre si. Uma
divergência aqui — outro `max_length`, outro dtype, o prefixo errado — degradaria
**todos os braços por igual**, o que a torna invisível na comparação entre braços
e fatal contra a linha de base publicada. Por isso o envelope é lido do disco, e
o prefixo `query: ` vem dele em vez de estar solto no código.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np

from quantptbr import corpus, embedding, retrieval, vectors


def main() -> int:
    corpus.verify_manifest()
    envelope = embedding.Envelope.load()
    judged = corpus.judged_query_ids()
    print(f"codificando {len(judged)} consultas julgadas com prefixo {envelope.query_prefix!r} …")

    query_vectors = retrieval.embed_queries(envelope)
    if query_vectors.shape != (len(judged), envelope.dimensions):
        raise RuntimeError(f"forma {query_vectors.shape} inesperada")
    norms = np.linalg.norm(query_vectors, axis=1)
    if np.abs(norms - 1).max() > 1e-5:
        raise RuntimeError(f"consultas não normalizadas: erro {np.abs(norms - 1).max():.2e}")

    vectors.VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(retrieval.QUERY_VECTORS_PATH, query_vectors)

    retrieval.QUERY_MANIFEST_PATH.parent.mkdir(exist_ok=True)
    retrieval.QUERY_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "issue": "RAF-55",
                "finished_at": datetime.now(UTC).isoformat(),
                "git_commit": vectors.git_commit(),
                "query_ids": judged,
                "envelope": json.loads(embedding.ENVELOPE_PATH.read_text(encoding="utf-8")),
                "array": {
                    "path": str(retrieval.QUERY_VECTORS_PATH.relative_to(corpus.REPO_ROOT)),
                    "shape": list(query_vectors.shape),
                    "sha256": corpus.sha256(retrieval.QUERY_VECTORS_PATH),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"vetores  → {retrieval.QUERY_VECTORS_PATH}")
    print(f"manifesto → {retrieval.QUERY_MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
