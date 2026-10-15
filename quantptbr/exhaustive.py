"""Trilha exaustiva: as quatro condições do documento de arquitetura, sem índice.

O documento de arquitetura (seção 27) define F32, F16, I8 e B1 sob busca
exaustiva, com regras próprias de quantização e de métrica. A trilha do Qdrant
mede outra coisa — os códigos do servidor, sob HNSW — e não consegue isolar o
erro de quantização por busca exata, porque o `exact` do Qdrant varre os
originais. Aqui cada consulta é comparada com todas as passagens, de modo que a
única fonte de perda é a representação.

As regras são as do documento, não as da matriz de `config/arms.json`:

* **F16** guarda float16 e reconstrói em float32, bloco a bloco, sem
  renormalizar.
* **I8** usa escala simétrica por dimensão, `s_j = max_i |x_ij| / 127`, tirada
  do corpus e nunca dos qrels. A consulta fica em float32.
* **B1** guarda 1 bit por dimensão (1 se positiva) e compara por distância de
  Hamming, com consulta e corpus empacotados. É outra função de comparação, e o
  escore é o negativo da distância para que maior continue sendo melhor.

**Desempate.** Escore decrescente e, no empate, o menor índice de linha — a
mesma regra de `retrieval.py`. Importa sobretudo em B1, onde a distância é
inteira e os empates na fronteira do top-k são a regra, não a exceção.
"""

from __future__ import annotations

import time
import tracemalloc

import ir_measures
import numpy as np
from ir_measures import RR, R, nDCG

from quantptbr import corpus

CONDITIONS = ("F32", "F16", "I8", "B1")

#: Profundidade do ranking persistido (seção 27.3 do documento).
DEPTH = 100

#: Linhas reconstruídas por vez. Limita o buffer float32 a ~150 MB em 768d.
BLOCK = 50_000

#: Rótulos da tabela → medida do `ir_measures`, sob o protocolo do documento:
#: ganho 2^rel − 1 no nDCG e relevante = grau acima de zero em Recall e MRR.
MEASURES = {
    "nDCG@10": nDCG(gains={grade: 2**grade - 1 for grade in corpus.RELEVANCE_GRADES}) @ 10,
    "Recall@100": R(rel=1) @ DEPTH,
    "MRR@100": RR(rel=1) @ DEPTH,
}


def int8_scales(vectors: np.ndarray) -> np.ndarray:
    """Uma escala por dimensão, `max |x| / 127`. Dimensão toda nula recebe 1."""
    peak = np.zeros(vectors.shape[1], dtype=np.float32)
    for start in range(0, len(vectors), BLOCK):
        peak = np.maximum(peak, np.abs(vectors[start : start + BLOCK]).max(axis=0))
    scales = peak / 127
    scales[scales == 0] = 1.0
    return scales


def pack_bits(block: np.ndarray) -> np.ndarray:
    """Um bit por dimensão, em palavras de 64 bits para a contagem de população.

    O preenchimento até múltiplo de 64 é de zeros nos dois lados da comparação,
    então não entra na distância.
    """
    bits = np.packbits(block > 0, axis=1)
    padding = -bits.shape[1] % 8
    if padding:
        bits = np.pad(bits, ((0, 0), (0, padding)))
    return np.ascontiguousarray(bits).view(np.uint64)


def encode(vectors: np.ndarray, kind: str) -> dict:
    """Constrói a representação de uma condição, lendo o corpus bloco a bloco."""
    if kind not in CONDITIONS:
        raise ValueError(f"condição {kind!r} não está entre {CONDITIONS}")
    if kind == "F32":
        # Cópia para a RAM: medir sobre o array mapeado cronometraria o disco.
        return {"kind": kind, "codes": np.array(vectors, dtype=np.float32)}

    scales = int8_scales(vectors) if kind == "I8" else None
    blocks = []
    for start in range(0, len(vectors), BLOCK):
        block = np.asarray(vectors[start : start + BLOCK], dtype=np.float32)
        if kind == "F16":
            blocks.append(block.astype(np.float16))
        elif kind == "I8":
            # `rint` arredonda ao par mais próximo; é a convenção registrada.
            blocks.append(np.clip(np.rint(block / scales), -127, 127).astype(np.int8))
        else:
            blocks.append(pack_bits(block))
    representation = {"kind": kind, "codes": np.concatenate(blocks)}
    if scales is not None:
        representation["scales"] = scales
    return representation


def size_bytes(representation: dict) -> int:
    """O que a condição mantém em memória: códigos e, no I8, as escalas."""
    return int(sum(v.nbytes for v in representation.values() if isinstance(v, np.ndarray)))


def scores(representation: dict, query: np.ndarray) -> np.ndarray:
    """O escore de toda passagem contra uma consulta float32. Maior é melhor."""
    kind, codes = representation["kind"], representation["codes"]
    if kind == "B1":
        distance = np.bitwise_count(codes ^ pack_bits(query[None, :])).sum(axis=1, dtype=np.int32)
        return -distance
    out = np.empty(len(codes), dtype=np.float32)
    for start in range(0, len(codes), BLOCK):
        block = codes[start : start + BLOCK]
        if kind == "F16":
            block = block.astype(np.float32)
        elif kind == "I8":
            block = block * representation["scales"]
        out[start : start + BLOCK] = block @ query
    return out


def top_k(values: np.ndarray, k: int) -> np.ndarray:
    """Os `k` maiores escores, com empate decidido pelo menor índice.

    Um `argpartition` sozinho escolheria ao acaso entre os empatados na fronteira
    do top-k. Aqui entram todos os que alcançam o k-ésimo escore, e só depois o
    corte, para que o resultado não dependa da ordem interna da partição.
    """
    k = min(k, values.size)
    threshold = np.partition(values, -k)[-k]
    candidates = np.flatnonzero(values >= threshold)
    order = np.lexsort((candidates, -values[candidates]))[:k]
    return candidates[order]


def search(
    representation: dict, query: np.ndarray, k: int = DEPTH
) -> tuple[np.ndarray, np.ndarray]:
    """Uma chamada de busca: é exatamente isto que a latência cronometra."""
    values = scores(representation, query)
    rows = top_k(values, k)
    return rows, values[rows]


def rank(
    representation: dict, queries: np.ndarray, query_ids: list[str], passage_ids: list[str]
) -> list:
    """O ranking de todas as consultas, no formato de linhas que a trilha do Qdrant usa."""
    from quantptbr.retrieval import Row

    rows = []
    for query_id, query in zip(query_ids, queries, strict=True):
        found, values = search(representation, query)
        rows.extend(
            Row(query_id, passage_ids[row], float(value), position)
            for position, (row, value) in enumerate(zip(found, values, strict=True), start=1)
        )
    return rows


def measure(representation: dict, queries: np.ndarray, warmup: int, rounds: int, seed: int) -> dict:
    """Latência por chamada e vazão sequencial, com aquecimento descartado.

    A ordem das consultas é embaralhada a cada rodada pela mesma semente em
    todas as condições, então as quatro percorrem exatamente a mesma sequência.
    A vazão é sequencial — consultas sobre o tempo total de busca — e não diz
    nada sobre capacidade sob concorrência.
    """
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for round_index in range(warmup + rounds):
        for position in rng.permutation(len(queries)):
            started = time.perf_counter()
            search(representation, queries[position])
            elapsed = time.perf_counter() - started
            if round_index >= warmup:
                samples.append(elapsed * 1000)

    # ponytail: pico de buffers por tracemalloc numa única chamada, não RSS do
    # processo amostrado; trocar por amostragem do SO se o pico virar resultado.
    tracemalloc.start()
    search(representation, queries[0])
    working_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    ordered = sorted(samples)
    return {
        "latency_ms_p50": float(np.percentile(ordered, 50)),
        "latency_ms_p95": float(np.percentile(ordered, 95)),
        "qps_sequential": len(ordered) / (sum(ordered) / 1000),
        "samples": len(ordered),
        "warmup_rounds": warmup,
        "measured_rounds": rounds,
        "seed": seed,
        "representation_bytes": size_bytes(representation),
        "working_peak_bytes": int(working_peak),
    }


def evaluate(run: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]]) -> dict:
    """As três métricas do documento, por consulta e agregadas, sobre todas as julgadas.

    Nenhuma consulta é excluída: a que não tem passagem relevante pontua zero em
    todas as condições e é sinalizada em `unscoreable`. O denominador é o mesmo
    para as quatro, como a seção 27.3 exige.
    """
    missing = set(qrels) - set(run)
    if missing:
        raise ValueError(f"consultas julgadas sem resultados no run: {sorted(missing, key=int)}")
    scoped = {q: v for q, v in run.items() if q in qrels}

    label_of = {str(measure): label for label, measure in MEASURES.items()}
    selected = list(MEASURES.values())
    per_query: dict[str, dict[str, float]] = {label: {} for label in MEASURES}
    for result in ir_measures.iter_calc(selected, qrels, scoped):
        per_query[label_of[str(result.measure)]][result.query_id] = float(result.value)
    aggregate = {
        label_of[str(measure)]: float(value)
        for measure, value in ir_measures.calc_aggregate(selected, qrels, scoped).items()
    }
    return {
        "queries": len(qrels),
        "unscoreable": corpus.unscoreable_query_ids(qrels),
        "aggregate": aggregate,
        "per_query": per_query,
    }
