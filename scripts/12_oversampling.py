#!/usr/bin/env python3
"""M3.4 — Varredura de oversampling sobre o braço binário com rescoring.

O A5 é reportado num único ponto de operação — `oversampling=4.0`, declarado na
matriz — e esse ponto carrega sozinho a recomendação de "use rescoring". Uma
recomendação de um ponto só não diz se ele é um bom ponto: não se sabe se 2x já
bastaria, nem se 8x compraria mais qualidade, nem quanto de latência cada passo
custa.

**Afrouxa um held-constant, de propósito e sob rótulo.** `oversampling` é
varrido; todo o resto — coleção, topologia, `ef`, rescoring — segue a matriz.
Nenhuma tabela reportada usa este script.

**Reaproveita a coleção do A4.** O A5 difere dele apenas em parâmetros de busca,
então a varredura não constrói nada: mede o mesmo índice sob candidatos
diferentes, e a diferença é atribuível ao oversampling e a nada mais.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime

from quantptbr import corpus, evaluation, manifest, retrieval, server

OUTPUT = corpus.REPO_ROOT / "runs" / "m3.4_oversampling.json"

#: A escada inclui o valor pré-registrado (4.0) para que uma das linhas seja
#: comparável ao que M3.1 reportou, e 1.0 como piso: rescoring sem ampliar a
#: lista de candidatos.
LADDER = (1.0, 2.0, 4.0, 8.0, 16.0)

ARM = "A5"
METRIC = "nDCG@10"

#: Repetições da latência. Menos que a varredura de M3.3 porque aqui a latência
#: é secundária: o que se procura é a forma da curva, não um p95 publicável.
REPEATS = 3


def measure(client, oversampling: float) -> dict:
    samples: list[float] = []
    rows = retrieval.retrieve(client, ARM, "hnsw", oversampling=oversampling, record_ms=samples)
    scored = evaluation.evaluate(retrieval.as_run(rows))
    for _ in range(REPEATS - 1):
        retrieval.retrieve(client, ARM, "hnsw", oversampling=oversampling, record_ms=samples)
    ordered = sorted(samples)
    return {
        "oversampling": oversampling,
        METRIC: scored["aggregate"][METRIC],
        "R(rel=2)@100": scored["aggregate"]["R(rel=2)@100"],
        "latency_ms_median": round(statistics.median(ordered), 2),
        "latency_ms_p95": round(ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)], 2),
        "latency_samples": len(ordered),
    }


def main() -> int:
    manifest.require_clean_tree()
    server.start()
    client = server.client()

    collection = retrieval.arm_config(ARM)["collection"]
    if not client.collection_exists(collection):
        raise SystemExit(
            f"{collection} não existe. Construa o A4 antes: "
            "`python scripts/05_create_collections.py A4`"
        )

    preregistered = retrieval.arm_config(ARM)["search_overrides"]["oversampling"]
    print(f"{ARM} sobre {collection}, oversampling pré-registrado {preregistered}\n")

    rows = []
    for value in LADDER:
        started = time.perf_counter()
        row = measure(client, value)
        rows.append(row)
        marker = "  ← pré-registrado" if value == preregistered else ""
        print(
            f"  oversampling {value:>5.1f}  {METRIC} {row[METRIC]:.4f}  "
            f"latência {row['latency_ms_median']:6.1f} ms  "
            f"({time.perf_counter() - started:5.1f}s){marker}"
        )

    reported = next(r for r in rows if r["oversampling"] == preregistered)
    best = max(rows, key=lambda r: r[METRIC])
    floor = next(r for r in rows if r["oversampling"] == LADDER[0])

    # Se ampliar a lista de candidatos não muda nada, o ganho atribuído ao
    # oversampling é na verdade do rescoring sozinho — e a recomendação do
    # artigo precisa dizer isso em vez de vender o parâmetro.
    spread = best[METRIC] - floor[METRIC]

    report = {
        "issues": ["RAF-69"],
        "arm": ARM,
        "collection": collection,
        "finished_at": datetime.now(UTC).isoformat(),
        "metric": METRIC,
        "_relaxed_held_constant": "oversampling",
        "preregistered_oversampling": preregistered,
        "ladder": list(LADDER),
        "latency_repeats": REPEATS,
        "by_oversampling": rows,
        "at_preregistered": reported,
        "best": best,
        "best_over_preregistered": best[METRIC] - reported[METRIC],
        "spread_over_ladder": spread,
        "_reading": (
            "A diferença entre o melhor e o piso da escada é o que o oversampling "
            "compra além do rescoring sozinho. Pequena, a recomendação do artigo "
            "é 'use rescoring' e não 'ajuste o oversampling'."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nmelhor: oversampling {best['oversampling']} com {METRIC} {best[METRIC]:.4f}")
    print(f"amplitude sobre a escada: {spread:+.4f}")
    if abs(spread) < 0.005:
        print("  A escada é plana: o ganho é do rescoring, não do oversampling.")
    print(f"\n→ {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
