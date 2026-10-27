#!/usr/bin/env python3
"""Comparação entre as trilhas: busca exaustiva × Qdrant e BGE-M3 × E5.

Nada aqui mede: o script lê a evidência de `scripts/14_exhaustive.py` e da
trilha do Qdrant em `runs/`, e escreve `paper/comparacao_trilhas.md` e
`paper/figures/comparacao_trilhas.png`.

    python scripts/14_exhaustive.py --label e5
    python scripts/14_exhaustive.py --label bge-m3 --corpus ... --queries ...
    python scripts/16_compare_tracks.py

**Busca exaustiva × Qdrant, com o mesmo modelo (E5).** A trilha exaustiva não
tem índice nem reordenação, então a perda dela contra o F32 é só da compressão.
No Qdrant, a coluna "só quantização" é o platô da varredura de `ef` sem
reordenação (`runs/m4_decomposition.json`), em que o grafo já convergiu; a
coluna "como medido" é a configuração do estudo, com o grafo em `ef` 128 e a
reordenação padrão. As três colunas usam a régua do estudo (ganho linear, 49
consultas), a única comum às duas trilhas.

**BGE-M3 × E5, com o mesmo protocolo.** Usa a régua do documento de
arquitetura (ganho 2^rel − 1, 50 consultas). Sem `runs/exhaustive_bge-m3.json`,
essa parte fica de fora e o resto sai do mesmo jeito.
"""

from __future__ import annotations

import importlib.util
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantptbr import corpus

RUNS = corpus.REPO_ROOT / "runs"
OUTPUT = corpus.REPO_ROOT / "paper" / "comparacao_trilhas.md"

# Mesmo estilo preto e branco e mesma função de gravação das outras figuras.
_spec = importlib.util.spec_from_file_location(
    "figures", corpus.REPO_ROOT / "scripts" / "13_figures.py"
)
figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figures)

#: Representação, condição da trilha exaustiva e braço do Qdrant equivalente.
PAIRS = [("float32", "F32", "A0"), ("int8", "I8", "A1"), ("binária", "B1", "A4")]
CONDITIONS = ["F32", "F16", "I8", "B1"]
MODELS = {"e5": "E5", "bge-m3": "BGE-M3"}


def load(name: str) -> dict | None:
    path = RUNS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def qdrant_hnsw(arm: str) -> float:
    return load(f"eval_{arm.lower()}_hnsw.json")["evaluation"]["aggregate"]["nDCG@10"]


def tracks_table(e5: dict, decomposition: dict) -> tuple[str, list[dict]]:
    reference = decomposition["reference"]["nDCG@10"]
    base = e5["conditions"]["F32"]["study_protocol_ndcg10"]
    rows = []
    for label, kind, arm in PAIRS:
        rows.append(
            {
                "label": label,
                "exhaustive": e5["conditions"][kind]["study_protocol_ndcg10"],
                "plateau": decomposition["arms"][arm]["plateau_ndcg10"],
                "measured": qdrant_hnsw(arm),
            }
        )
    lines = [
        "| Representação | Exaustiva | Δ vs F32 | Qdrant, só quantização | Δ vs A0 exato "
        "| Qdrant, como medido | Δ vs A0 exato |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['exhaustive']:.4f} | {row['exhaustive'] - base:+.4f} "
            f"| {row['plateau']:.4f} | {row['plateau'] - reference:+.4f} "
            f"| {row['measured']:.4f} | {row['measured'] - reference:+.4f} |"
        )
    f16 = e5["conditions"]["F16"]["study_protocol_ndcg10"]
    lines.append(f"| float16 | {f16:.4f} | {f16 - base:+.4f} | — | — | — | — |")
    return "\n".join(lines), rows


def retention(evidence: dict) -> dict[str, float]:
    base = evidence["conditions"]["F32"]["quality"]["aggregate"]["nDCG@10"]
    return {
        kind: evidence["conditions"][kind]["quality"]["aggregate"]["nDCG@10"] / base
        for kind in CONDITIONS
    }


def models_table(evidence: dict[str, dict]) -> str:
    header = ["Condição"]
    for name in evidence:
        header += [f"nDCG@10 {name}", f"% do F32 ({name})", f"Δ vs F32 {name} (IC 95%)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    kept = {name: retention(row) for name, row in evidence.items()}
    for kind in CONDITIONS:
        cells = [kind]
        for name, row in evidence.items():
            condition = row["conditions"][kind]
            versus = condition["versus_f32"]
            cells += [
                f"{condition['quality']['aggregate']['nDCG@10']:.4f}",
                f"{100 * kept[name][kind]:.1f}%",
                "—"
                if versus is None
                else f"{versus['mean_difference']:+.4f} [{versus['ci_low']:+.4f}, {versus['ci_high']:+.4f}]",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(rows: list[dict], evidence: dict[str, dict]) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    x = range(len(rows))
    series = [
        ("exhaustive", "Exaustiva (sem índice)", "o", figures.PRIMARY),
        ("plateau", "Qdrant, só quantização", "s", figures.SURFACE),
        ("measured", "Qdrant, como medido", "^", figures.SECONDARY),
    ]
    for offset, (key, name, marker, fill) in zip((-0.15, 0, 0.15), series, strict=True):
        left.plot(
            [i + offset for i in x],
            [row[key] for row in rows],
            marker,
            markersize=8,
            markerfacecolor=fill,
            markeredgecolor=figures.INK,
            linestyle="none",
            label=name,
        )
    left.set_xticks(list(x), [row["label"] for row in rows])
    left.set_ylabel("nDCG@10 (régua do estudo)")
    left.set_ylim(0, None)
    left.set_title("E5: compressão sozinha × no Qdrant")
    left.legend(frameon=False, fontsize=8, loc="lower left")

    markers = iter([("o", figures.PRIMARY), ("s", figures.SURFACE)])
    for name, row in evidence.items():
        marker, fill = next(markers)
        kept = retention(row)
        right.plot(
            CONDITIONS,
            [100 * kept[kind] for kind in CONDITIONS],
            marker,
            markersize=8,
            markerfacecolor=fill,
            markeredgecolor=figures.INK,
            linestyle="-",
            color=figures.MUTED,
            label=name,
        )
    right.set_ylabel("nDCG@10 em % do F32 (régua do documento)")
    right.set_ylim(0, 105)
    right.set_title("Quanto cada modelo retém sob compressão")
    right.legend(frameon=False, fontsize=8, loc="lower left")

    fig.tight_layout()
    figures.save(fig, "comparacao_trilhas.png")


def main() -> int:
    e5 = load("exhaustive_e5.json")
    if e5 is None:
        raise SystemExit("falta runs/exhaustive_e5.json; rode scripts/14_exhaustive.py --label e5")
    decomposition = load("m4_decomposition.json")
    evidence = {
        name: row for label, name in MODELS.items() if (row := load(f"exhaustive_{label}.json"))
    }

    tracks, rows = tracks_table(e5, decomposition)
    overlap = e5["f32_overlap_with_original_a0_exact"]
    report = [
        "# Comparação entre as trilhas",
        "",
        "Gerado por `scripts/16_compare_tracks.py` a partir de `runs/`; nenhum número",
        "foi digitado.",
        "",
        "## Busca exaustiva × Qdrant, com o mesmo modelo (E5)",
        "",
        tracks,
        "",
        "> Régua do estudo: ganho linear, 49 consultas. A busca exata do A0 no Qdrant",
        f"> dá {decomposition['reference']['nDCG@10']:.4f}. \"Só quantização\" é o platô sem",
        "> reordenação, com o grafo já convergido; \"como medido\" é a Tabela T1 de",
        "> `paper/tables.md` (ef 128, reordenação padrão). A trilha exaustiva usa os",
        "> vetores regenerados; o top-10 e o top-100 do F32 coincidem com os da busca",
        f"> exata do A0 original em {overlap['@10']:.3f} e {overlap['@100']:.3f}.",
        "> As regras de quantização diferem: o I8 exaustivo usa o máximo por dimensão",
        "> (seção 27.1 do documento); o int8 do Qdrant, o quantil 0,99.",
        "",
        "## BGE-M3 × E5, com o mesmo protocolo",
        "",
    ]
    if "BGE-M3" in evidence:
        report += [
            models_table(evidence),
            "",
            "> Régua do documento de arquitetura: ganho 2^rel − 1, 50 consultas, busca",
            "> exaustiva. \"% do F32\" é o nDCG@10 da condição dividido pelo do F32 do",
            "> mesmo modelo.",
        ]
    else:
        report.append("Pendente: falta `runs/exhaustive_bge-m3.json`.")
    report.append("")

    OUTPUT.write_text("\n".join(report), encoding="utf-8")
    print(f"→ {OUTPUT}")
    figure(rows, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
