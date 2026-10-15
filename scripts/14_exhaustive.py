#!/usr/bin/env python3
"""Trilha exaustiva do documento de arquitetura: F32, F16, I8 e B1, sem índice.

Preenche a Tabela 3 do documento. Cada consulta é comparada com todas as
passagens, então o que separa as condições é só a representação — sem grafo,
sem servidor, sem rescoring. As regras de quantização e de métrica são as da
seção 27 do documento e moram em `quantptbr/exhaustive.py`.

    python scripts/14_exhaustive.py --label e5
    python scripts/14_exhaustive.py --label bge-m3 \\
        --corpus data/vectors/bge_m3_1m.npy --queries data/vectors/bge_m3_queries.npy

O modelo é argumento, não código: a trilha recebe um arquivo de vetores do
corpus, na ordem de `corpus.iter_passages()`, e um das consultas julgadas, na
ordem de `corpus.judged_query_ids()`. `--label` só nomeia as saídas.

**Duas réguas de qualidade, de propósito.** As colunas principais seguem o
protocolo do documento (ganho 2^rel − 1, 50 consultas). A coluna "protocolo do
estudo" reavalia o mesmo ranking com as regras de `config/arms.json` (ganho
linear, 49 consultas), que é o que permite pôr esta trilha ao lado da do Qdrant
sem comparar grandezas diferentes.

Escreve os rankings em `runs/exhaustive/`, a evidência em
`runs/exhaustive_<label>.json` e a tabela em `paper/exhaustive_<label>.md`.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

from quantptbr import analysis, corpus, evaluation, exhaustive, manifest, retrieval, stats, vectors

RUNS = corpus.REPO_ROOT / "runs"
PAPER = corpus.REPO_ROOT / "paper"

#: Seção 27.3 do documento: três rodadas de aquecimento e dez medidas.
WARMUP_ROUNDS = 3
MEASURED_ROUNDS = 10

METRIC = "nDCG@10"
BASELINE = "F32"

#: Linhas amostradas para conferir que os vetores chegam unitários.
NORM_SAMPLE = 10_000


def load_inputs(corpus_path: Path, queries_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Abre os vetores e recusa o que não bate com o corpus e as consultas congelados."""
    passages = np.load(corpus_path, mmap_mode="r")
    queries = np.load(queries_path).astype(np.float32)
    passage_ids = corpus.load_passage_ids()

    if passages.shape[0] != len(passage_ids):
        raise RuntimeError(
            f"{corpus_path}: {passages.shape[0]:,} linhas, corpus tem {len(passage_ids):,}"
        )
    expected = (len(corpus.judged_query_ids()), passages.shape[1])
    if queries.shape != expected:
        raise RuntimeError(f"{queries_path}: forma {queries.shape}, esperada {expected}")

    rows = np.sort(np.random.default_rng(0).choice(len(passage_ids), NORM_SAMPLE, replace=False))
    sample = np.concatenate([np.asarray(passages[rows], dtype=np.float32), queries])
    if not np.isfinite(sample).all():
        raise RuntimeError("vetores não finitos na amostra")
    drift = float(np.abs(np.linalg.norm(sample, axis=1) - 1).max())
    if drift > 1e-3:
        raise RuntimeError(f"vetores não unitários: erro de norma {drift:.2e}")
    return passages, queries, passage_ids


def save_run(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")


def per_query(report: dict) -> np.ndarray:
    values = report["per_query"][METRIC]
    return np.array([values[q] for q in sorted(values, key=int)], dtype=np.float64)


def table(results: dict[str, dict]) -> str:
    baseline_bytes = results[BASELINE]["performance"]["representation_bytes"]
    header = [
        "Configuração",
        "nDCG@10",
        "Δ vs F32 (IC 95%)",
        "Recall@100",
        "MRR@100",
        "Top-10 igual ao F32",
        "Memória (GB)",
        "Redução",
        "Latência p50 (ms)",
        "p95 (ms)",
        "QPS",
        "nDCG@10, protocolo do estudo",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for kind, row in results.items():
        quality, speed, versus = row["quality"]["aggregate"], row["performance"], row["versus_f32"]
        delta = (
            "—"
            if versus is None
            else f"{versus['mean_difference']:+.4f} [{versus['ci_low']:+.4f}, {versus['ci_high']:+.4f}]"
        )
        cells = [
            kind,
            f"{quality['nDCG@10']:.4f}",
            delta,
            f"{quality['Recall@100']:.4f}",
            f"{quality['MRR@100']:.4f}",
            f"{row['overlap_at_10_with_f32']:.3f}",
            f"{speed['representation_bytes'] / 1e9:.3f}",
            f"{baseline_bytes / speed['representation_bytes']:.1f}x",
            f"{speed['latency_ms_p50']:.1f}",
            f"{speed['latency_ms_p95']:.1f}",
            f"{speed['qps_sequential']:.1f}",
            f"{row['study_protocol_ndcg10']:.4f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="nome do modelo nas saídas, ex.: e5, bge-m3")
    parser.add_argument("--corpus", type=Path, default=vectors.ARRAY_PATH)
    parser.add_argument("--queries", type=Path, default=retrieval.QUERY_VECTORS_PATH)
    parser.add_argument("--conditions", nargs="+", default=list(exhaustive.CONDITIONS))
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if BASELINE not in args.conditions:
        raise SystemExit(f"{BASELINE} é a referência de toda comparação e não pode ficar de fora")

    manifest.require_clean_tree(args.allow_dirty)
    passages, queries, passage_ids = load_inputs(args.corpus, args.queries)
    query_ids = corpus.judged_query_ids()
    qrels = corpus.load_qrels()
    seed = stats.matrix()["statistics"]["seed"]
    print(
        f"{args.label}: {passages.shape[0]:,} passagens de {passages.shape[1]}d, {len(query_ids)} consultas\n"
    )

    results: dict[str, dict] = {}
    runs: dict[str, list] = {}
    for kind in args.conditions:
        started = time.perf_counter()
        representation = exhaustive.encode(passages, kind)
        build_seconds = time.perf_counter() - started

        rows = exhaustive.rank(representation, queries, query_ids, passage_ids)
        run_path = RUNS / "exhaustive" / f"{args.label}_{kind}.jsonl"
        save_run(rows, run_path)
        run = retrieval.as_run(rows)

        results[kind] = {
            "build_seconds": round(build_seconds, 2),
            "run_file": str(run_path.relative_to(corpus.REPO_ROOT)),
            "quality": exhaustive.evaluate(run, qrels),
            "study_protocol_ndcg10": evaluation.evaluate(run)["aggregate"][METRIC],
            "performance": exhaustive.measure(
                representation, queries, WARMUP_ROUNDS, MEASURED_ROUNDS, seed
            ),
        }
        runs[kind] = rows
        del representation
        quality, speed = results[kind]["quality"]["aggregate"], results[kind]["performance"]
        print(
            f"  {kind}  nDCG@10 {quality['nDCG@10']:.4f}  Recall@100 {quality['Recall@100']:.4f}  "
            f"MRR@100 {quality['MRR@100']:.4f}  {speed['representation_bytes'] / 1e9:.3f} GB  "
            f"p50 {speed['latency_ms_p50']:.1f} ms"
        )

    baseline = per_query(results[BASELINE]["quality"])
    for kind, row in results.items():
        row["overlap_at_10_with_f32"] = analysis.overlap(runs[BASELINE], runs[kind], 10)
        row["versus_f32"] = None
        if kind != BASELINE:
            comparison = stats.paired_bootstrap(baseline, per_query(row["quality"]))
            row["versus_f32"] = comparison.__dict__ | {
                "sentence": comparison.sentence(kind, METRIC).replace("A0", BASELINE)
            }

    # O run da busca exata do A0 na trilha do servidor, medido em agosto sobre o
    # artefato original. Com vetores do mesmo modelo regenerados noutra máquina,
    # a sobreposição diz quanto o arquivo novo reproduz o antigo; com outro modelo,
    # diz quanto os dois modelos concordam.
    original = None
    if (RUNS / "retrieval" / "A0_exact.jsonl").exists():
        reference = retrieval.load_run("A0", "exact")
        original = {f"@{d}": analysis.overlap(reference, runs[BASELINE], d) for d in (10, 100)}
        print(f"\n  F32 contra o A0 exato original: sobreposição {original}")

    evidence = {
        "label": args.label,
        "finished_at": datetime.now(UTC).isoformat(),
        "git_commit": manifest.git_commit(),
        "git_dirty": manifest.git_is_dirty(),
        "protocol": "documento de arquitetura, seção 27: busca exaustiva, ganho 2^rel − 1, relevante = grau > 0",
        "inputs": {
            "corpus_vectors": {
                "path": str(args.corpus),
                "shape": list(passages.shape),
                "sha256": corpus.sha256(args.corpus),
            },
            "query_vectors": {
                "path": str(args.queries),
                "shape": list(queries.shape),
                "sha256": corpus.sha256(args.queries),
            },
            "qrels_sha256": corpus.read_manifest()["files"]["qrels"]["sha256"],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.processor(),
            "threadpools": [
                {key: pool[key] for key in ("internal_api", "num_threads")}
                for pool in threadpool_info()
            ],
        },
        "f32_overlap_with_original_a0_exact": original,
        "conditions": results,
    }
    evidence_path = RUNS / f"exhaustive_{args.label}.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    unscoreable = results[BASELINE]["quality"]["unscoreable"]
    report = "\n".join(
        [
            f"# Trilha exaustiva — {args.label}",
            "",
            "Gerado por `scripts/14_exhaustive.py`; nenhum número foi digitado. Busca",
            "exaustiva sobre as quatro condições do documento de arquitetura (seção 27).",
            "",
            table(results),
            "",
            f"> Qualidade sobre {len(qrels)} consultas, com ganho 2^rel − 1. Consultas sem passagem",
            f"> relevante, mantidas no denominador com zero: {', '.join(unscoreable) or 'nenhuma'}.",
            "> A última coluna reavalia o mesmo ranking com as regras de `config/arms.json`",
            "> (ganho linear, 49 consultas) e é a única comparável com `paper/tables.md`.",
            ">",
            f"> Latência por chamada, {WARMUP_ROUNDS} rodadas de aquecimento descartadas e",
            f"> {MEASURED_ROUNDS} medidas, ordem embaralhada pela mesma semente nas quatro condições.",
            "> QPS é vazão sequencial, não capacidade sob concorrência. Memória é o tamanho",
            "> da representação em RAM, em GB decimais; F16 e I8 são reconstruídos em float32",
            "> bloco a bloco, então a latência deles não demonstra aceleração nativa.",
            *(
                []
                if original is None
                else [
                    ">",
                    "> Sobreposição do top-10 e do top-100 do F32 com a busca exata do A0 da trilha",
                    "> do servidor (`runs/retrieval/A0_exact.jsonl`):",
                    f"> {original['@10']:.3f} e {original['@100']:.3f}.",
                ]
            ),
            "",
        ]
    )
    PAPER.mkdir(exist_ok=True)
    (PAPER / f"exhaustive_{args.label}.md").write_text(report, encoding="utf-8")
    print(f"\n→ {evidence_path}\n→ {PAPER / f'exhaustive_{args.label}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
