#!/usr/bin/env python3
"""S1.1 — Prova que os contadores de memória respondem, antes de apontá-los para 3 GB.

Um contador que não se move é um instrumento quebrado, e descobrir isso depois de
construir os seis braços custaria o sprint. Aqui ele é exercitado contra quatro
configurações cuja direção esperada é conhecida de antemão.

**O que este script decide e o que não decide.** Ele valida o *instrumento*: que
o contador resolve a carga de vetores, distingue configurações e responde ao
estado de residência. Só isso é portão. As comparações entre configurações saem
registradas como observações, nunca aprovadas ou reprovadas — 400 mil vetores
aleatórios com cinco consultas exaustivas não caracterizam braço nenhum. Isso é
trabalho de M3, a 1M, com os vetores e as consultas reais.

**Estado de residência declarado.** A primeira execução (2026-08-18) revelou que
a residência dos vetores do Qdrant aparece em `file` (page cache), não em `anon`,
porque os segmentos são mapeados em disco mesmo com `on_disk=False`. Medir logo
após a escrita em massa mede o rastro do build, não o conjunto de trabalho. Por
isso cada caso é medido em **três estados**: frio após reinício do container,
morno após carga de consultas HNSW, e após uma busca exaustiva — que toca todos
os vetores e portanto define o teto de residência.

Roda com o servidor de pé, **nenhuma outra coleção carregada** e a máquina
ociosa. Uma execução concorrente com outra carga pesada contamina o page cache.
"""

from __future__ import annotations

import json
import time

import numpy as np
from qdrant_client import models

from quantptbr import corpus, memory, server

COLLECTION = "quantptbr_smoke"
POINTS = 400_000
DIMS = 768
BATCH = 2_000
WARMUP_QUERIES = 200

INT8 = models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
)

CASES = [
    ("1. float32, vetores em RAM", None, False),
    ("2. float32, vetores on_disk", None, True),
    ("3. int8 SEM on_disk", INT8, False),
    ("4. int8 COM on_disk", INT8, True),
]


def build(client, quantization, on_disk: bool, vectors: np.ndarray) -> None:
    client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=DIMS, distance=models.Distance.COSINE, on_disk=on_disk
        ),
        quantization_config=quantization,
    )
    for start in range(0, len(vectors), BATCH):
        chunk = vectors[start : start + BATCH]
        client.upsert(
            collection_name=COLLECTION,
            points=models.Batch(ids=list(range(start, start + len(chunk))), vectors=chunk.tolist()),
            wait=True,
        )
    deadline = time.monotonic() + 600
    while client.get_collection(COLLECTION).status != models.CollectionStatus.GREEN:
        if time.monotonic() > deadline:
            raise TimeoutError("otimizador não terminou em 10 min")
        time.sleep(2)
    if quantization is not None:
        server.require_quantization(client, COLLECTION, "scalar")


def measure_states(vectors: np.ndarray) -> dict[str, dict]:
    """Mede a residência em três estados, porque ela não é um número só.

    Frio logo após reinício é o piso; morno sob HNSW é o que uma implantação
    real sustenta; exaustivo é o teto, porque toca todos os vetores.
    """
    server.restart()
    client = server.client()
    server.require_server(client)
    states = {"cold": memory.read().as_dict()}

    rng = np.random.default_rng(1)
    probes = vectors[rng.choice(len(vectors), WARMUP_QUERIES, replace=False)]
    for probe in probes:
        client.query_points(COLLECTION, query=probe.tolist(), limit=10)
    states["warm_hnsw"] = memory.read().as_dict()

    for probe in probes[:5]:
        client.query_points(
            COLLECTION,
            query=probe.tolist(),
            limit=10,
            search_params=models.SearchParams(exact=True),
        )
    states["exhaustive"] = memory.read().as_dict()
    return states


def main() -> int:
    print(f"gerando {POINTS:,} vetores aleatórios de {DIMS}d …")
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(POINTS, DIMS)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    payload_mb = POINTS * DIMS * 4 / 1024**2
    print(f"{payload_mb:.0f} MB de vetores float32 — é este o sinal que os contadores devem ver\n")

    results = []
    for label, quantization, on_disk in CASES:
        server.restart(fresh_storage=True)
        client = server.client()
        server.require_server(client)
        baseline = memory.read()

        build(client, quantization, on_disk, vectors)
        states = measure_states(vectors)
        results.append(
            {
                "case": label,
                "baseline_total_mb": baseline.total_mb,
                "states": states,
                "rss_mb": memory.rss_mb(),
            }
        )
        print(f"\n{label}")
        for name, state in states.items():
            print(
                f"  {name:<11} anon {state['anon_mb']:>7.1f}  file {state['file_mb']:>7.1f}  "
                f"total {state['total_mb']:>7.1f}  MB"
            )

    checks, observations = memory.smoke_verdict(results, payload_mb)

    print("\n=== validação do instrumento ===")
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'FALHA'}] {name}")
    print("\n=== observações entre configurações (a confirmar em M3) ===")
    for name, value in observations.items():
        if not name.startswith("_"):
            print(f"  {name}: {value:+.1f} MB")

    path = corpus.REPO_ROOT / "runs" / "s1.1_memory_smoke.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "issue": "RAF-50",
                "points": POINTS,
                "dims": DIMS,
                "float32_payload_mb": round(payload_mb, 1),
                "warmup_queries": WARMUP_QUERIES,
                "host_idle": "requerido; leituras de page cache não valem sob carga concorrente",
                "cases": results,
                "checks": checks,
                "observations": observations,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nevidência → {path}")

    server.client().delete_collection(COLLECTION)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
