"""O único caminho de recuperação: um harness parametrizado por braço.

Se cada braço tivesse o seu script, eles divergiriam em coisas que ninguém
pretendeu — um `ef` diferente, outro top-k, outro tratamento de empate — e essas
diferenças seriam reportadas como efeito de quantização. Um harness só, com o
braço como parâmetro, torna a divergência acidental impossível em vez de apenas
improvável.

**Desempate.** A quantização binária produz distâncias de Hamming inteiras, com
muitos empates exatos. A ordem em que o servidor devolve empatados não é estável
e afetaria o braço binário desproporcionalmente. A regra aqui é fixa e igual em
todos os braços: pontuação decrescente, e em caso de empate o **menor ID de
ponto**, que é a ordem do corpus congelada em S1.0.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import models

from quantptbr import corpus, embedding, server, stats, vectors

QUERY_VECTORS_PATH = vectors.VECTORS_DIR / "queries_float32.npy"
QUERY_MANIFEST_PATH = corpus.REPO_ROOT / "runs" / "s1.6_queries.json"
RUNS_DIR = corpus.REPO_ROOT / "runs" / "retrieval"

#: Profundidade única de recuperação. O Recall@100 exige 100, e o nDCG@10 sai do
#: mesmo run — nunca de uma segunda passada mais rasa.
DEPTH = 100


@dataclass(frozen=True)
class Row:
    query_id: str
    passage_id: str
    score: float
    rank: int


def embed_queries(envelope: embedding.Envelope | None = None) -> np.ndarray:
    """Consultas julgadas sob o mesmo envelope do corpus, na ordem de `judged_query_ids`.

    O prefixo `query: ` vem do envelope, não do código. Uma inconsistência entre
    a codificação de consulta e a de passagem degradaria todos os braços por
    igual — invisível na comparação entre braços e fatal contra a linha de base
    publicada.
    """
    envelope = embedding.Envelope.load() if envelope is None else envelope
    topics = corpus.load_topics()
    judged = corpus.judged_query_ids()
    model = embedding.load_model(envelope.dtype, envelope.max_length)
    try:
        raw = model.encode(
            [envelope.query_prefix + topics[q] for q in judged],
            batch_size=envelope.batch_size,
            normalize_embeddings=envelope.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    finally:
        del model
    return vectors.normalise(raw)


def load_queries() -> tuple[list[str], np.ndarray]:
    if not QUERY_VECTORS_PATH.exists():
        raise FileNotFoundError(f"{QUERY_VECTORS_PATH} ausente. Rode scripts/06_embed_queries.py.")
    return corpus.judged_query_ids(), np.load(QUERY_VECTORS_PATH)


def arm_config(arm_id: str) -> dict:
    for arm in stats.matrix()["arms"]:
        if arm["id"] == arm_id:
            return arm
    raise KeyError(f"braço {arm_id} não está na matriz pré-registrada")


def search_params(
    arm_id: str,
    mode: str,
    ef: int | None = None,
    rescore: bool | None = None,
    oversampling: float | None = None,
) -> models.SearchParams:
    """Parâmetros de busca de um braço, derivados da matriz — nunca digitados aqui.

    `ef`, `rescore` e `oversampling` existem só para os diagnósticos que precisam
    afrouxar um held-constant de propósito e sob rótulo: a decomposição de M4.3
    varre `ef` com `rescore` desligado, e a varredura de M3.4 varre
    `oversampling`. Toda tabela reportada usa os valores da matriz, que é o que
    se obtém omitindo os três.
    """
    if mode not in stats.matrix()["search_modes"]:
        raise ValueError(f"modo {mode!r} não pré-registrado")
    arm = arm_config(arm_id)
    overrides = arm.get("search_overrides", {})
    quantization = None
    if arm["quantization"] is not None:
        declared_rescore = overrides.get("rescore", True) if rescore is None else rescore
        quantization = models.QuantizationSearchParams(
            rescore=declared_rescore,
            # Sem rescoring não há o que reordenar, e pedir oversampling assim
            # só ampliaria a lista de candidatos sem mudar o critério.
            oversampling=(
                (overrides.get("oversampling") if oversampling is None else oversampling)
                if declared_rescore
                else None
            ),
        )
    return models.SearchParams(
        exact=(mode == "exact"),
        hnsw_ef=stats.matrix()["held_constant"]["hnsw"]["ef_search"] if ef is None else ef,
        quantization=quantization,
    )


def retrieve(
    client,
    arm_id: str,
    mode: str,
    depth: int = DEPTH,
    ef: int | None = None,
    rescore: bool | None = None,
    oversampling: float | None = None,
    record_ms: list[float] | None = None,
) -> list[Row]:
    """Recupera para todas as consultas julgadas, com desempate determinístico."""
    arm = arm_config(arm_id)
    collection = arm["collection"]
    server.require_server(client)

    counted = client.count(collection, exact=True).count
    if counted != corpus.EXPECTED_PASSAGES:
        raise RuntimeError(
            f"{collection} tem {counted:,} pontos, esperados {corpus.EXPECTED_PASSAGES:,}"
        )
    if arm["quantization"] is not None:
        server.require_quantization(client, collection, arm["quantization"]["kind"])

    query_ids, query_vectors = load_queries()
    passage_ids = corpus.load_passage_ids()
    params = search_params(arm_id, mode, ef=ef, rescore=rescore, oversampling=oversampling)

    rows: list[Row] = []
    for query_id, vector in zip(query_ids, query_vectors, strict=True):
        # `record_ms` existe para que M3.3 cronometre **este** caminho, e não uma
        # cópia dele: latência medida por um segundo laço divergiria do que a
        # qualidade mediu sem que nada parecesse errado.
        started = time.perf_counter()
        hits = client.query_points(
            collection, query=vector.tolist(), limit=depth, search_params=params
        ).points
        if record_ms is not None:
            record_ms.append((time.perf_counter() - started) * 1000)
        # A ordem do servidor não é estável entre empates; esta é.
        ordered = sorted(hits, key=lambda h: (-h.score, h.id))
        rows.extend(
            Row(query_id, passage_ids[hit.id], float(hit.score), rank)
            for rank, hit in enumerate(ordered, start=1)
        )
    return rows


def as_run(rows: list[Row]) -> dict[str, dict[str, float]]:
    """Converte para o formato que o `ir_measures` consome.

    O score vai como `-rank` em vez do valor bruto: a ordenação já foi decidida
    pela regra de desempate, e passar o score do servidor deixaria a biblioteca
    reordenar empates por conta própria, desfazendo exatamente o que a regra
    existe para fixar.
    """
    run: dict[str, dict[str, float]] = {}
    for row in rows:
        run.setdefault(row.query_id, {})[row.passage_id] = float(-row.rank)
    return run


def save_run(rows: list[Row], arm_id: str, mode: str) -> Path:
    """Persiste por consulta, nunca agregado: o bootstrap de M4 lê exatamente isto."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{arm_id}_{mode}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
    return path


def load_run(arm_id: str, mode: str) -> list[Row]:
    path = RUNS_DIR / f"{arm_id}_{mode}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [Row(**json.loads(line)) for line in fh]
