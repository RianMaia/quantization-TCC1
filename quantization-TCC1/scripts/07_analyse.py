#!/usr/bin/env python3
"""M4.1 — Bootstrap pareado contra o A0, com intervalo em toda diferença.

Nenhum verbo é escolhido aqui. A frase de cada linha sai de `Comparison.sentence`,
com o verbo amarrado ao intervalo pela regra pré-registrada em S1.2 — que é o
que torna a contenção final crível em vez de post-hoc.

Um intervalo contendo zero nunca é reportado como equivalência. Ao lado dele vai
a diferença mínima detectável, calculada da variância observada: sem ela, o
leitor não sabe se o intervalo é informativo ou se o desenho era cego para
diferenças desse tamanho.
"""

from __future__ import annotations

import json

from quantptbr import analysis, corpus, manifest

OUTPUT = corpus.REPO_ROOT / "runs" / "m4_bootstrap.json"


def main() -> int:
    manifest.require_clean_tree()
    report = analysis.report()

    for mode in sorted({row["mode"] for row in report["comparisons"]}):
        print(f"\n=== modo {mode} ===")
        for metric in sorted({r["metric"] for r in report["comparisons"] if r["mode"] == mode}):
            print(f"\n  {metric}")
            for row in report["comparisons"]:
                if row["mode"] != mode or row["metric"] != metric:
                    continue
                flag = "  [idêntico por construção]" if row["identical_by_construction"] else ""
                print(
                    f"    {row['arm']}  {row['variant_mean']:.4f}  "
                    f"{row['mean_difference']:+.4f}  "
                    f"IC95 [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  "
                    f"mde {row['minimum_detectable_difference']:.4f}  "
                    f"→ {row['verdict']}{flag}"
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n→ {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
