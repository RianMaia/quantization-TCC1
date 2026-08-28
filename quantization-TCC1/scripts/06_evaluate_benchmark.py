#!/usr/bin/env python3
"""S1.7 / M3.1 — Recupera, avalia e, no A0 exato, confronta o valor publicado.

Todo resultado de quantização deste artigo é uma **diferença** contra o A0. Se o
A0 estiver errado — qrels mal lidos, limiar trocado, consulta desalinhada — todas
as diferenças saem calculadas contra uma referência quebrada, e a coerência
interna da tabela não revelaria nada.

Daí o portão: o A0 sob busca exata é comparado contra o nDCG@10 = 0,3955
publicado para o `multilingual-e5-base` no Quati 1M. **Nada é ajustado para
acertar o número.** O valor do âncora está inteiramente em ser uma checagem
independente; sintonizar o pipeline até bater destruiria justamente isso.

    python scripts/06_evaluate_benchmark.py A0
    python scripts/06_evaluate_benchmark.py A1 A2 --modes hnsw

Os valores por consulta são persistidos junto do agregado, porque é deles que o
bootstrap pareado de M4 se alimenta — uma média já agregada não pareia nada.
"""

from __future__ import annotations

import argparse

from quantptbr import corpus, evaluation, manifest, retrieval, server, stats

RUNS_DIR = corpus.REPO_ROOT / "runs"


def calibrates(arm_id: str, mode: str) -> bool:
    target = stats.matrix()["evaluation"]["calibration_target"]
    return arm_id == target["arm"] and mode == target["search_mode"]


def report_calibration(report: dict) -> None:
    band = report["range_check"]
    print("\n=== calibração do aparelho (S1.7) ===")
    print(f"  protocolo: {report['protocol']}")
    print(f"  publicado (Quati, Tabela 6, E5-base)   {report['published']:.4f}")
    print(
        f"  pré-registrado, 49 consultas           {report['preregistered_over_49']:.4f}"
        f"   ({report['delta_49']:+.4f})"
    )
    print(
        f"  diagnóstico, 50 consultas              {report['diagnostic_over_50']:.4f}"
        f"   ({report['delta_50']:+.4f})"
    )
    print("\n  faixa da Tabela 7 (qrels de 1M, os do estudo)")
    print(f"    {band['floor']['system']:<28} {band['floor']['ndcg10']:.4f}")
    print(
        f"    A0 exato                     {report['study_qrels_ndcg10']:.4f}"
        f"   {'dentro' if band['inside'] else 'FORA'} da faixa"
    )
    print(f"    {band['nearest_above']['system']:<28} {band['nearest_above']['ndcg10']:.4f}")
    print(f"    {band['ceiling']['system']:<28} {band['ceiling']['ndcg10']:.4f}")
    print(f"\n  → {report['verdict']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", metavar="BRAÇO", help="ids da matriz, ex.: A0 A1")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=stats.matrix()["search_modes"],
        help="modos de busca; o padrão são os dois pré-registrados",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permite árvore git suja; a execução fica registrada como não citável",
    )
    args = parser.parse_args()

    manifest.require_clean_tree(args.allow_dirty)
    corpus.verify_manifest()
    server.start()
    client = server.client()

    halted = False
    for arm_id in args.arms:
        for mode in args.modes:
            print(f"\n{arm_id} / {mode}: recuperando {retrieval.DEPTH} por consulta …")
            rows = retrieval.retrieve(client, arm_id, mode)
            run_path = retrieval.save_run(rows, arm_id, mode)

            run = retrieval.as_run(rows)
            result = evaluation.evaluate(run)
            for name, value in result["aggregate"].items():
                print(f"  {name:<12} {value:.4f}   (n={result['queries']})")

            extra = {"evaluation": result, "run_file": str(run_path.relative_to(corpus.REPO_ROOT))}
            if calibrates(arm_id, mode):
                extra["calibration"] = evaluation.calibration_report(run)
                report_calibration(extra["calibration"])
                halted = halted or extra["calibration"]["verdict"].startswith("PARE")

            path = manifest.RunManifest.capture(
                arm_id=arm_id, search_mode=mode, allow_dirty=args.allow_dirty, **extra
            ).save(RUNS_DIR / f"eval_{arm_id.lower()}_{mode}.json")
            print(f"  manifesto → {path}")

    if halted:
        print(
            "\nO aparelho não está validado. Nenhum braço comprimido é interpretável até isto ser resolvido."
        )
    return 1 if halted else 0


if __name__ == "__main__":
    raise SystemExit(main())
