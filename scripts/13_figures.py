#!/usr/bin/env python3
"""Figuras e análise por consulta, a partir da evidência já medida.

Nada aqui mede: o script lê `paper/numbers.json` e os arquivos de `runs/`, e
desenha o que as tabelas já dizem. Roda offline, sem GPU e sem servidor.

    python scripts/11_regenerate.py     # antes, para ter o numbers.json em dia
    python scripts/13_figures.py

Escreve as figuras em `paper/figures/` e a tabela por consulta em
`paper/per_query.md`. Nenhum valor é digitado aqui — as coordenadas de todo
ponto vêm dos mesmos arquivos que alimentam `paper/tables.md`, de modo que
figura e tabela não podem divergir.

**A sobreposição de ranking não usa o gabarito.** Ela pergunta quanto do top-k
do A0 cada braço ainda devolve, e por isso não herda o limite das 49 consultas
julgadas por LLM: é uma segunda leitura do efeito da quantização, independente
da primeira.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantptbr import analysis, corpus, retrieval, stats

NUMBERS = corpus.REPO_ROOT / "paper" / "numbers.json"
GEOMETRY = corpus.REPO_ROOT / "runs" / "m2.8_geometry.json"
FIGURES = corpus.REPO_ROOT / "paper" / "figures"
PER_QUERY = corpus.REPO_ROOT / "paper" / "per_query.md"

METRIC = "nDCG@10"
MODE = "hnsw"
DEPTHS = (10, 100)

#: Preto e branco, como a seção 12 do documento de arquitetura promete. A
#: identidade de um ponto vem de forma, preenchimento, textura e rótulo direto,
#: nunca só do tom — o que também sobrevive a impressão e fotocópia.
PRIMARY = "#1a1a1a"
SECONDARY = "#a6a6a6"
INK = "#000000"
MUTED = "#7a7a7a"
GRID = "#e0e0e0"
SURFACE = "#ffffff"

#: O A3 e o A5 ficam vizinhos nos dois gráficos de Pareto; com o deslocamento
#: padrão os rótulos se sobrepõem.
LABEL_OFFSET = {"A5": (8, -14)}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"→ {FIGURES / name}")


def pareto_figure(numbers: dict, key: str, cost: dict[str, float], xlabel: str, name: str) -> None:
    """Qualidade contra um custo, com a fronteira que `11_regenerate.py` calculou."""
    quality = numbers["quality"]
    frontier = numbers["pareto"][key]["frontier"]

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for on_frontier, face, marker, label in (
        (True, PRIMARY, "o", "na fronteira de Pareto"),
        (False, SURFACE, "s", "dominado"),
    ):
        arms = [arm for arm in quality if (arm in frontier) == on_frontier]
        axis.scatter(
            [cost[arm] for arm in arms],
            [quality[arm]["ndcg10"] for arm in arms],
            s=70,
            marker=marker,
            facecolors=face,
            edgecolors=PRIMARY,
            linewidth=1.5,
            label=label,
            zorder=3,
        )
    ordered = sorted(frontier, key=lambda arm: cost[arm])
    axis.plot(
        [cost[arm] for arm in ordered],
        [quality[arm]["ndcg10"] for arm in ordered],
        color=PRIMARY,
        linewidth=1.2,
        zorder=2,
    )
    for arm in quality:
        axis.annotate(
            arm,
            (cost[arm], quality[arm]["ndcg10"]),
            xytext=LABEL_OFFSET.get(arm, (7, 6)),
            textcoords="offset points",
            color=INK,
        )
    axis.set_xlabel(f"{xlabel}  (menor é melhor)")
    axis.set_ylabel(f"{METRIC}  (maior é melhor)")
    axis.set_title(f"Fronteira de Pareto: {len(frontier)} de {len(quality)} braços")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 0.55)
    axis.legend(frameon=False, loc="lower right")
    save(figure, name)


def confidence_figure(numbers: dict) -> None:
    """A diferença de cada braço contra o A0, com o intervalo que decide o verbo."""
    rows = [(arm, row) for arm, row in numbers["quality"].items() if row["ci"] is not None]
    rows.reverse()

    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    for position, (arm, row) in enumerate(rows):
        low, high = row["ci"]
        crosses_zero = low <= 0 <= high
        colour = MUTED if crosses_zero else PRIMARY
        axis.plot([low, high], [position, position], color=colour, linewidth=2, zorder=2)
        axis.scatter(
            [row["delta_vs_a0"]],
            [position],
            s=50,
            facecolors=SURFACE if crosses_zero else PRIMARY,
            edgecolors=colour,
            linewidth=1.5,
            zorder=3,
        )
        axis.annotate(
            f"{row['delta_vs_a0']:+.4f}  [{low:+.4f}, {high:+.4f}]",
            (high, position),
            xytext=(8, -3),
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
    axis.axvline(0, color=INK, linewidth=0.8, zorder=1)
    axis.set_yticks(range(len(rows)), [arm for arm, _ in rows])
    axis.tick_params(axis="y", colors=INK)
    axis.grid(axis="y", visible=False)
    axis.set_xlim(-0.5, 0.2)
    axis.set_xlabel(f"Diferença de {METRIC} contra o A0, com IC 95%")
    axis.set_title("Vazado: o intervalo contém zero. Cheio: inteiramente abaixo de zero")
    save(figure, "confianca.png")


def oversampling_figure(numbers: dict) -> None:
    """Dois painéis, um eixo cada: qualidade e latência têm escalas diferentes."""
    sweep = numbers["oversampling"]
    rows = sweep["by_oversampling"]
    factors = [row["oversampling"] for row in rows]
    preregistered = sweep["preregistered_oversampling"]

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
    for axis, field, label, colour in (
        (top, METRIC, METRIC, PRIMARY),
        (bottom, "latency_ms_median", "Latência mediana (ms)", SECONDARY),
    ):
        values = [row[field] for row in rows]
        axis.plot(factors, values, color=colour, linewidth=2, marker="o", markersize=6)
        for factor, value in zip(factors, values, strict=True):
            axis.annotate(
                f"{value:.4f}" if field == METRIC else f"{value:.1f}",
                (factor, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=8,
            )
        axis.axvline(preregistered, color=MUTED, linewidth=0.8, linestyle="--")
        axis.set_ylabel(label)
        axis.margins(y=0.25)
    top.set_title(f"A5: oversampling varrido; tracejado = {preregistered:g}x pré-registrado")
    bottom.set_xscale("log", base=2)
    bottom.set_xticks(factors, [f"{factor:g}x" for factor in factors])
    bottom.set_xlabel("Oversampling")
    save(figure, "oversampling.png")


def geometry_figure() -> None:
    """Por que a binária colapsa: dimensões que dão quase sempre o mesmo bit."""
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    positive = np.array(geometry["per_dimension"]["positive_fraction"])
    majority = np.maximum(positive, 1 - positive)
    threshold = geometry["degeneracy_threshold"]

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    # Arredondado para que o limiar caia exatamente numa borda: com a borda em
    # 0,9500000000000001 uma dimensão em 0,95 exato seria pintada do lado errado.
    edges = np.round(np.linspace(0.5, 1.0, 21), 3)
    counts, _, patches = axis.hist(
        majority, bins=edges, color=PRIMARY, edgecolor=SURFACE, linewidth=1
    )
    highlighted = 0
    for patch, left, count in zip(patches, edges[:-1], counts, strict=True):
        if left >= threshold:
            patch.set_facecolor(SECONDARY)
            patch.set_hatch("///")
            highlighted += int(count)
    if highlighted != geometry["degenerate_dimensions"]:
        raise RuntimeError(
            f"a figura pinta {highlighted} dimensões como degeneradas, a evidência diz "
            f"{geometry['degenerate_dimensions']}"
        )
    axis.axvline(threshold, color=INK, linewidth=0.8, linestyle="--")
    axis.annotate(
        f"{geometry['degenerate_dimensions']} de {geometry['dimensions']} dimensões\n"
        f"degeneradas em sinal (≥ {threshold:.2f})",
        (threshold, axis.get_ylim()[1]),
        xytext=(-8, -6),
        textcoords="offset points",
        ha="right",
        va="top",
        color=INK,
        fontsize=9,
    )
    axis.set_xlabel("Fração dos vetores com o bit majoritário (0,5 = equilíbrio, 1,0 = bit inútil)")
    axis.set_ylabel("Dimensões")
    axis.set_title(
        f"Orçamento efetivo do código binário: {geometry['effective_bits']:.1f} "
        f"de {geometry['dimensions']} bits"
    )
    save(figure, "geometria.png")


def per_query_table() -> str:
    """Vitórias, empates, derrotas e sobreposição de ranking, cada braço contra o A0."""
    loaded = analysis.load_all(MODE)
    baseline = loaded[analysis.BASELINE]
    reference = retrieval.load_run(analysis.BASELINE, MODE)

    def overlaps(arm_id: str, mode: str) -> list[str]:
        run = retrieval.load_run(arm_id, mode)
        return [f"{analysis.overlap(reference, run, depth):.3f}" for depth in DEPTHS]

    header = ["Braço", "Configuração", "Vitórias", "Empates", "Derrotas"]
    header += [f"Sobreposição@{depth} com A0" for depth in DEPTHS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for arm in stats.matrix()["arms"]:
        if arm["id"] == analysis.BASELINE:
            continue
        left, right = analysis.aligned(baseline, loaded[arm["id"]], METRIC)
        cells = [arm["id"], arm["label"], *map(str, analysis.win_tie_loss(left, right))]
        lines.append("| " + " | ".join(cells + overlaps(arm["id"], MODE)) + " |")
    control = ["A0 exato", "float32, busca exaustiva (controle)", "—", "—", "—"]
    lines.append("| " + " | ".join(control + overlaps(analysis.BASELINE, "exact")) + " |")

    queries = len(analysis.aligned(baseline, baseline, METRIC)[0])
    return "\n".join(
        [
            "# Análise por consulta — gerada, não digitada",
            "",
            "Gerado por `scripts/13_figures.py` a partir de `runs/eval_*.json` e",
            "`runs/retrieval/*.jsonl`. Busca HNSW; referência é o A0.",
            "",
            f"Vitórias, empates e derrotas contam consultas em {METRIC} ({queries}",
            "pontuáveis). A sobreposição é a fração média do top-k do A0 que o braço",
            "também devolve; não usa o gabarito e cobre todas as consultas recuperadas.",
            "",
            *lines,
            "",
            "> A linha de controle compara o A0 sob HNSW com o A0 sob busca exata: é",
            "> quanto o grafo sozinho mexe no ranking, sem quantização nenhuma. Um braço",
            "> com sobreposição próxima desse valor perturba o top-k tanto quanto a busca",
            "> aproximada já perturbava.",
            "",
        ]
    )


def main() -> int:
    if not NUMBERS.exists():
        raise SystemExit(f"{NUMBERS} ausente. Rode scripts/11_regenerate.py antes.")
    numbers = json.loads(NUMBERS.read_text(encoding="utf-8"))

    floor = {arm: row["provisioning_floor_mb"] for arm, row in numbers["cost"].items()}
    latency = {arm: row["median"] for arm, row in numbers["latency"].items()}
    pareto_figure(
        numbers, "quality_vs_memory", floor, "Piso de provisionamento (MB)", "pareto_memoria.png"
    )
    pareto_figure(
        numbers, "quality_vs_latency", latency, "Latência mediana (ms)", "pareto_latencia.png"
    )
    confidence_figure(numbers)
    oversampling_figure(numbers)
    geometry_figure()

    PER_QUERY.write_text(per_query_table(), encoding="utf-8")
    print(f"→ {PER_QUERY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
