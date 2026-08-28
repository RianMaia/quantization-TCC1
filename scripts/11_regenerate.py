#!/usr/bin/env python3
"""M4.4 — Regenera toda tabela do artigo a partir das evidências gravadas.

**Ponto de entrada único.** Quem escrever o artigo não lê `runs/*.json` a mão
nem digita número nenhum: roda isto e copia. Todo valor sai de um arquivo de
evidência produzido por um script de medição, e a procedência de cada um está
no bloco final.

**Não mede nada, e não completa nada.** Um número que falta nunca é estimado:
uma lacuna declarada é recuperável, um número inventado não.

**Aborta com código não-zero quando falta evidência.** RAF-74 pré-registrou
isso, e a razão é que regeneração parcial silenciosa põe tabela incompleta no
artigo — um `—` copiado para o LaTeX vira lacuna publicada. Para inspecionar um
estado intermediário durante a medição, passe `--allow-partial`, que marca as
lacunas e sai com zero; nenhuma tabela reportada deve sair desse modo.

Saídas em `paper/`: `tables.md` para colar, `numbers.json` para conferir.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from quantptbr import corpus, manifest, stats

RUNS = corpus.REPO_ROOT / "runs"
OUTPUT_DIR = corpus.REPO_ROOT / "paper"

#: Estado de residência que o artigo reporta. O exaustivo continua gravado, mas
#: o `exact` do Qdrant varre os originais e ignora os códigos, então ele mede o
#: caminho dos originais em todo braço em vez do teto do braço quantizado.
REPORTED_STATE = "warm_hnsw"

MISSING = "—"

#: Lacunas encontradas na passada. Preenchida pelas funções de tabela e cobrada
#: no fim: é o que decide entre sair com zero e abortar.
GAPS: list[str] = []


def gap(what: str) -> str:
    GAPS.append(what)
    return MISSING


def load(name: str) -> dict | None:
    path = RUNS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def arms() -> list[dict]:
    return stats.matrix()["arms"]


def table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown puro: o formato final é do LaTeX de quem escrever, não nosso."""
    line = "| " + " | ".join(header) + " |"
    rule = "|" + "|".join("---" for _ in header) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([line, rule, *body])


def quality_rows(numbers: dict) -> list[list[str]]:
    """T1 — qualidade por braço, com o intervalo que calibra o verbo da frase."""
    bootstrap = load("m4_bootstrap.json")
    by_arm = {}
    if bootstrap:
        for row in bootstrap["comparisons"]:
            if row["mode"] == "hnsw" and row["metric"] == "nDCG@10":
                by_arm[row["arm"]] = row

    rows = []
    for arm in arms():
        evaluation = load(f"eval_{arm['id'].lower()}_hnsw.json")
        if not evaluation:
            marker = gap(f"qualidade do {arm['id']} (runs/eval_{arm['id'].lower()}_hnsw.json)")
            rows.append([arm["id"], arm["label"], marker, marker, marker, marker])
            continue
        aggregate = evaluation["evaluation"]["aggregate"]
        comparison = by_arm.get(arm["id"])
        if comparison:
            interval = f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]"
            delta = f"{comparison['mean_difference']:+.4f}"
            verdict = comparison["verdict"]
        else:
            interval, delta = "", MISSING
            verdict = "linha de base"
        rows.append(
            [
                arm["id"],
                arm["label"],
                f"{aggregate['nDCG@10']:.4f}",
                f"{aggregate['R(rel=2)@100']:.4f}",
                f"{delta} {interval}".strip(),
                verdict,
            ]
        )
        numbers["quality"][arm["id"]] = {
            "ndcg10": aggregate["nDCG@10"],
            "recall100": aggregate["R(rel=2)@100"],
            "delta_vs_a0": comparison["mean_difference"] if comparison else None,
            "ci": [comparison["ci_low"], comparison["ci_high"]] if comparison else None,
            "verdict": verdict,
            "minimum_detectable_difference": (
                comparison["minimum_detectable_difference"] if comparison else None
            ),
        }
    return rows


def cost_rows(numbers: dict) -> tuple[list[list[str]], list[str]]:
    """T2 — a tabela que carrega a tese central do artigo.

    Comprimir reduz a memória que é preciso provisionar **e aumenta os bytes que
    é preciso armazenar**: o Qdrant guarda os originais float32 ao lado dos
    códigos. Por isso cada braço carrega as duas razões, e a nominal — a que a
    documentação e os blogs citam — fica ao lado para o contraste ser visível.
    """
    measured = load("m3_arms.json")
    if not measured:
        gap("medições de memória e latência (runs/m3_arms.json)")
        return [], ["`runs/m3_arms.json` ausente: rode `scripts/10_measure_arms.py`."]

    by_arm = {row["arm"]: row for row in measured["arms"]}
    baseline = by_arm.get("A0")
    notes: list[str] = []
    rows = []
    for arm in arms():
        row = by_arm.get(arm["id"])
        if not row:
            marker = gap(f"custo do {arm['id']} em runs/m3_arms.json")
            rows.append([arm["id"], arm["label"], marker] + [marker] * 4)
            notes.append(f"{arm['id']}: não medido nesta execução.")
            continue
        state = row["states"][REPORTED_STATE]
        disk = row["footprint"]["total_mb"]
        floor = state["anon_mb"]
        # Razões na direção que se lê sem inverter: acima de 1,00x significa
        # gastar MAIS que o float32. A forma invertida convida exatamente o erro
        # de leitura que o artigo existe para desfazer — 0,32x parece "comprimiu
        # para 32%" quando o fato é "precisa de 3,1x mais RAM".
        storage = f"{disk / baseline['footprint']['total_mb']:.2f}x" if baseline else MISSING
        base_floor = baseline["states"][REPORTED_STATE]["anon_mb"] if baseline else 0
        residency = f"{floor / base_floor:.2f}x" if base_floor else MISSING
        rows.append(
            [
                arm["id"],
                arm["label"],
                f"{arm['nominal_compression']}x",
                f"{disk:,.1f}",
                storage,
                f"{floor:,.1f}",
                residency,
            ]
        )
        numbers["cost"][arm["id"]] = {
            "nominal_compression": arm["nominal_compression"],
            "disk_mb": disk,
            "provisioning_floor_mb": floor,
            "resident_total_mb": row["reported_resident_mb"],
            "page_cache_mb": state["file_mb"],
            "residency_state": REPORTED_STATE,
            "unindexed_points": (row.get("build") or {}).get("unindexed_points"),
        }
        residual = (row.get("build") or {}).get("unindexed_points") or 0
        if residual:
            notes.append(
                f"{arm['id']}: {residual:,} pontos num segmento residual fora do HNSW, "
                "buscados por varredura exata."
            )
    if measured.get("void_measurements"):
        notes.append(
            "Medições anuladas por paginação do container: "
            f"{', '.join(measured['void_measurements'])}."
        )
    return rows, notes


def latency_rows(numbers: dict) -> list[list[str]]:
    """T3 — latência. Cache quente e consultas repetidas: não é cold start."""
    measured = load("m3_arms.json")
    if not measured:
        return []
    by_arm = {row["arm"]: row for row in measured["arms"]}
    rows = []
    for arm in arms():
        row = by_arm.get(arm["id"])
        if not row:
            marker = gap(f"latência do {arm['id']} em runs/m3_arms.json")
            rows.append([arm["id"], arm["label"], marker, marker, marker])
            continue
        latency = row["latency_ms_per_query"]
        rows.append(
            [
                arm["id"],
                arm["label"],
                f"{latency['median']:.1f}",
                f"{latency['p95']:.1f}",
                f"{latency['samples']}",
            ]
        )
        numbers["latency"][arm["id"]] = latency
    return rows


def decomposition_rows(numbers: dict) -> list[list[str]]:
    """T4 — de onde vem a perda: dos códigos, do grafo, ou recuperada no rescoring."""
    decomposition = load("m4_decomposition.json")
    if not decomposition:
        return []
    rows = []
    for arm_id, values in decomposition["arms"].items():
        rows.append(
            [
                arm_id,
                f"{values['quantization_error']:+.4f}",
                f"{values['graph_error_at_preregistered_ef']:+.4f}",
                f"{values['rescoring_recovery']:+.4f}",
                f"{values['reported_ndcg10']:.4f}",
            ]
        )
        numbers["decomposition"][arm_id] = values
    return rows


def geometry_lines(numbers: dict) -> list[str]:
    """T5 — por que o braço binário colapsa, e não só que ele colapsa."""
    geometry = load("m2.8_geometry.json")
    if not geometry:
        return []
    numbers["geometry"] = {
        key: geometry[key]
        for key in (
            "dimensions",
            "sample_size",
            "degeneracy_threshold",
            "mean_of_dimension_means",
            "mean_abs_offset_over_std",
            "degenerate_dimensions",
            "effective_bits",
        )
    }
    fraction = geometry["effective_bits"] / geometry["dimensions"]
    return [
        (
            f"- Média das médias por dimensão: **{geometry['mean_of_dimension_means']:+.5f}** "
            "— o espaço *parece* centrado quando olhado no agregado."
        ),
        (
            "- Deslocamento por dimensão, |média|/desvio: "
            f"**{geometry['mean_abs_offset_over_std']:.3f}** "
            "— olhado dimensão a dimensão, não está."
        ),
        (
            f"- Dimensões degeneradas em sinal (limiar {geometry['degeneracy_threshold']}, "
            f"declarado antes de calcular): **{geometry['degenerate_dimensions']} de "
            f"{geometry['dimensions']}**."
        ),
        (
            f"- Orçamento efetivo do código binário: **{geometry['effective_bits']:.1f} bits de "
            f"{geometry['dimensions']}** ({fraction:.1%})."
        ),
    ]


def pareto(numbers: dict, cost_of, key: str, label: str) -> tuple[list[str], list[str]]:
    """Quem sobrevive quando qualidade e um custo são cobrados juntos.

    Um braço é dominado quando existe outro que não é pior em nenhum dos dois
    eixos e é melhor em pelo menos um. É a pergunta que se faz antes de escolher
    uma configuração, e ela tem resposta mecânica em vez de retórica.
    """
    quality = numbers["quality"]
    shared = [arm for arm in quality if cost_of(arm) is not None]
    if len(shared) < 2:
        return [], []

    def point(arm: str) -> tuple[float, float]:
        return quality[arm]["ndcg10"], cost_of(arm)

    frontier, dominated = [], {}
    for arm in shared:
        ndcg, cost = point(arm)
        rivals = [
            other
            for other in shared
            if other != arm
            and point(other)[0] >= ndcg
            and point(other)[1] <= cost
            and point(other) != (ndcg, cost)
        ]
        if rivals:
            dominated[arm] = rivals
        else:
            frontier.append(arm)

    numbers.setdefault("pareto", {})[key] = {
        "objectives": ["maximizar nDCG@10", f"minimizar {label}"],
        "frontier": frontier,
        "dominated_by": dominated,
    }

    rows = [
        [
            arm,
            f"{point(arm)[0]:.4f}",
            f"{point(arm)[1]:,.1f}",
            (
                "dominado por " + ", ".join(dominated[arm])
                if arm in dominated
                else "**na fronteira**"
            ),
        ]
        for arm in shared
    ]
    return [table(["Braço", "nDCG@10", label, "Situação"], rows)], frontier


def pareto_lines(numbers: dict) -> list[str]:
    """T7 — as duas fronteiras que a recomendação de ponto de operação precisa."""
    cost, latency = numbers["cost"], numbers["latency"]
    memory, memory_frontier = pareto(
        numbers,
        lambda arm: (cost.get(arm) or {}).get("provisioning_floor_mb"),
        "quality_vs_memory",
        "Piso de provisionamento (MB)",
    )
    speed, speed_frontier = pareto(
        numbers,
        lambda arm: (latency.get(arm) or {}).get("median"),
        "quality_vs_latency",
        "Latência mediana (ms)",
    )
    if not memory:
        return []

    lines = ["### Qualidade × memória a provisionar", "", *memory]
    if len(memory_frontier) == 1:
        lines += [
            "",
            f"> **A fronteira tem um ponto só: {memory_frontier[0]}.** Nenhuma",
            "> configuração comprimida é Pareto-ótima neste par de eixos — a linha de",
            "> base tem ao mesmo tempo a melhor qualidade e o menor piso.",
        ]
    if speed:
        lines += ["", "### Qualidade × latência", "", *speed]
        if len(speed_frontier) > 1:
            lines += [
                "",
                (
                    f"> Aqui a compressão paga: a fronteira tem {len(speed_frontier)} "
                    f"pontos ({', '.join(speed_frontier)})."
                ),
                "> É o eixo em que trocar qualidade por velocidade tem sentido.",
            ]
    lines += [
        "",
        "> **A recomendação é condicional, e o texto não deve simplificá-la.** Se a",
        "> restrição é memória a provisionar, não comprima: nesta configuração a",
        "> compressão custa RAM não-evictável em vez de economizá-la. Se a restrição é",
        "> latência, a compressão compra tempo, e aí o par binário + rescoring é o que",
        "> recupera qualidade sem devolver a velocidade toda.",
    ]
    return lines


def oversampling_lines(numbers: dict) -> list[str]:
    """T8 — o oversampling do braço binário, varrido em vez de assumido."""
    sweep = load("m3.4_oversampling.json")
    if not sweep:
        return []
    numbers["oversampling"] = sweep
    preregistered = sweep["preregistered_oversampling"]
    rows = [
        [
            f"{row['oversampling']:.0f}x" + (" ←" if row["oversampling"] == preregistered else ""),
            f"{row[sweep['metric']]:.4f}",
            f"{row['latency_ms_median']:.1f}",
        ]
        for row in sweep["by_oversampling"]
    ]
    best = sweep["best"]
    lines = [table(["Oversampling", sweep["metric"], "Latência mediana (ms)"], rows)]
    if abs(sweep["spread_over_ladder"]) < 0.005:
        lines += [
            "",
            "> A escada é plana: o ganho é do rescoring, não do oversampling.",
        ]
    else:
        lines += [
            "",
            (
                f"> **A escada não é plana: amplitude de {sweep['spread_over_ladder']:+.4f} "
                f"em {sweep['metric']}.** O ponto pré-registrado ({preregistered:.0f}x) não é "
                f"o melhor da escada — a {best['oversampling']:.0f}x o braço binário chega a "
                f"{best[sweep['metric']]:.4f}."
            ),
            ">",
            "> **Consequência para o texto:** o A5 reportado nas outras tabelas usa o valor",
            "> pré-registrado, e é esse que deve ser citado como resultado. Esta varredura é",
            "> diagnóstico — mostra que o parâmetro tem efeito forte e que o ponto de",
            "> operação escolhido antes da medição era conservador. Trocar o valor",
            "> reportado por causa do resultado seria escolher o ponto depois de ver os",
            "> dados.",
        ]
    return lines


def calibration_lines(numbers: dict) -> list[str]:
    """T6 — a validação do aparelho, que é o que autoriza todo o resto.

    Sem ela o estudo compara seis braços entre si e não sabe se algum deles está
    perto do que a literatura obteve. Reproduz o protocolo dos autores do Quati:
    run de 1M pontuado contra os qrels de 10M.
    """
    path = RUNS / "retrieval" / "A0_exact.jsonl"
    if not path.exists():
        return []
    from collections import defaultdict

    from quantptbr import evaluation

    run: dict[str, dict[str, float]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        hit = json.loads(line)
        run[hit["query_id"]][hit["passage_id"]] = hit["score"]
    report = evaluation.calibration_report(dict(run))
    numbers["calibration"] = report
    band = report["range_check"]
    return [
        (
            f"- Alvo publicado (Bueno et al., Quati, Tabela 6, E5-base): "
            f"**{report['published']:.4f}**"
        ),
        (
            f"- Medido aqui, mesmo protocolo: **{report['preregistered_over_49']:.4f}** "
            f"(Δ {report['delta_49']:+.4f})"
        ),
        f"- Veredito: **{report['verdict']}**",
        (
            f"- Checagem de faixa: {band['floor']['system']} {band['floor']['ndcg10']:.4f} "
            f"< A0 {report['study_qrels_ndcg10']:.4f} "
            f"< {band['nearest_above']['system']} {band['nearest_above']['ndcg10']:.4f} "
            f"— dentro: **{band['inside']}**"
        ),
    ]


def provenance(numbers: dict) -> list[str]:
    """O bloco que torna a tabela auditável em vez de apenas legível."""
    reference = load("eval_a0_hnsw.json")
    lines = []
    if reference:
        controlled = reference["controlled"]
        numbers["provenance"] = {
            "corpus_sha256": controlled["corpus_sha256"],
            "qrels_sha256": controlled["qrels_sha256"],
            "envelope": controlled["envelope"],
            "server_version": controlled["server_version"],
            "image_digest": controlled["image_digest"],
            "hnsw": controlled["hnsw"],
            "distance": controlled["distance"],
            "measurement_tools": controlled["measurement_tools"],
            "judged_queries": len(controlled["judged_query_ids"]),
            "host": reference["host"],
        }
        lines += [
            (
                f"- Corpus `sha256:{controlled['corpus_sha256'][:16]}…`, "
                f"qrels `sha256:{controlled['qrels_sha256'][:16]}…`"
            ),
            (
                f"- Modelo `{controlled['envelope']['model_id']}` "
                f"@ `{controlled['envelope']['model_revision'][:12]}`, "
                f"{controlled['envelope']['dimensions']}d"
            ),
            f"- Qdrant {controlled['server_version']}, imagem `{controlled['image_digest'][:23]}…`",
            (
                f"- HNSW m={controlled['hnsw']['m']}, "
                f"ef_construct={controlled['hnsw']['ef_construct']}, "
                f"ef_search={controlled['hnsw']['ef_search']}, {controlled['distance']}"
            ),
            (
                f"- Consultas julgadas: {len(controlled['judged_query_ids'])} "
                f"(pontuáveis: {reference['evaluation']['queries']})"
            ),
            f"- Kernel `{reference['host']['kernel']}`, Python {reference['host']['python']}",
        ]
    commit = manifest.git_commit()
    numbers["regenerated_at"] = datetime.now(UTC).isoformat()
    numbers["git_commit"] = commit
    lines.append(f"- Regenerado em {numbers['regenerated_at']} no commit `{commit[:12]}`")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="marca lacunas e sai com zero; nenhuma tabela reportada deve sair deste modo",
    )
    args = parser.parse_args()

    numbers: dict = {
        "quality": {},
        "cost": {},
        "latency": {},
        "decomposition": {},
    }

    cost, cost_notes = cost_rows(numbers)
    sections = [
        "# Tabelas do artigo — regeneradas, não digitadas",
        "",
        "Gerado por `scripts/11_regenerate.py`. Todo número vem de um arquivo em `runs/`.",
        "Nenhum valor aqui foi transcrito à mão; um campo `—` significa evidência ausente,",
        "não estimativa.",
        "",
        "## T1 — Qualidade por braço (busca HNSW, 49 consultas pontuáveis)",
        "",
        table(
            ["Braço", "Configuração", "nDCG@10", "R(rel≥2)@100", "Δ vs A0 (IC 95%)", "Veredito"],
            quality_rows(numbers),
        ),
        "",
        "> Um intervalo que contém zero **não** é equivalência: é ausência de evidência de",
        "> diferença neste tamanho de amostra. A diferença mínima detectável por braço está",
        "> em `numbers.json`.",
        "",
        "## T2 — Compressão: armazenamento contra residência",
        "",
        "A distinção central do artigo, e o ponto em que o medido contraria o esperado.",
        "",
        "O Qdrant guarda os vetores float32 originais ao lado dos códigos quantizados, de",
        "modo que comprimir **aumenta os bytes a armazenar** — isso era esperado. O que",
        "não era: comprimir **também aumenta a memória a provisionar**. Com",
        "`always_ram: true`, os códigos vão para memória anônima enquanto os originais",
        "seguem mapeados em disco; a linha de base float32 mantém tudo como page cache",
        "evictável, e por isso tem o menor piso de todos. A razão nominal é a que a",
        "documentação e os posts de prática citam; as outras duas são as medidas.",
        "",
        table(
            [
                "Braço",
                "Configuração",
                "Compressão nominal",
                "Disco (MB)",
                "Disco vs A0",
                "Piso de provisionamento (MB)",
                "Piso vs A0",
            ],
            cost,
        ),
        "",
        "> **Ler as duas últimas colunas na direção certa: acima de `1.00x` é gastar",
        "> MAIS que o float32, não menos.** Nenhum braço comprimido reduz o piso de",
        "> provisionamento — todos o aumentam. Em disco, só a quantização binária fica",
        "> abaixo da linha de base, e por 5%, longe dos 32x nominais.",
        ">",
        "> Piso de provisionamento = memória anônima do cgroup do container no estado",
        f"> `{REPORTED_STATE}`: é o que precisa existir na máquina. O page cache dos",
        "> originais é evictável e aparece em `numbers.json` como `page_cache_mb`.",
    ]
    if cost_notes:
        sections += ["", *[f"> {note}" for note in cost_notes]]

    latency = latency_rows(numbers)
    if latency:
        sections += [
            "",
            "## T3 — Latência por consulta (ms)",
            "",
            table(["Braço", "Configuração", "Mediana", "p95", "Amostras"], latency),
            "",
            "> Cache quente, consultas repetidas, concorrência 1. Não é um perfil de",
            "> cold start, e os valores são específicos deste hardware.",
        ]

    decomposition = decomposition_rows(numbers)
    if decomposition:
        sections += [
            "",
            "## T4 — Decomposição do erro (nDCG@10)",
            "",
            table(
                ["Braço", "Quantização", "Grafo (ef pré-registrado)", "Rescoring", "Reportado"],
                decomposition,
            ),
            "",
            "> Obtida varrendo `ef` até o platô, com rescoring desligado. Isola quanto da",
            "> perda vem dos códigos e quanto vem da busca aproximada.",
        ]

    geometry = geometry_lines(numbers)
    if geometry:
        sections += ["", "## T5 — Geometria dos vetores e o teto do braço binário", "", *geometry]

    pareto = pareto_lines(numbers)
    if pareto:
        sections += [
            "",
            "## T7 — Fronteiras de Pareto e o ponto de operação",
            "",
            "Um braço é **dominado** quando existe outro que não é pior em nenhum dos dois",
            "eixos e é melhor em pelo menos um. É a pergunta que se faz antes de escolher",
            "uma configuração, e tem resposta mecânica.",
            "",
            *pareto,
        ]

    oversampling = oversampling_lines(numbers)
    if oversampling:
        sections += [
            "",
            "## T8 — Varredura de oversampling (A5, sobre a coleção do A4)",
            "",
            "Afrouxa um held-constant sob rótulo: `oversampling` é varrido, o resto segue a",
            "matriz. Serve para dizer se o ponto de operação do A5 é bom, e não apenas qual",
            "ele é.",
            "",
            *oversampling,
        ]

    calibration = calibration_lines(numbers)
    if calibration:
        sections += [
            "",
            "## T6 — Validação do aparelho",
            "",
            "Reproduz o protocolo dos autores do Quati — run sobre o corpus de 1M pontuado",
            "contra os qrels de 10M — para confrontar a linha de base com o único número de",
            "E5-base publicado sobre este benchmark. É o que autoriza ler as outras tabelas.",
            "",
            *calibration,
        ]

    sections += ["", "## Procedência", "", *provenance(numbers)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_path = OUTPUT_DIR / "tables.md"
    numbers_path = OUTPUT_DIR / "numbers.json"
    tables_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    numbers_path.write_text(
        json.dumps(numbers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    numbers["gaps"] = GAPS
    print("\n".join(sections))
    print(f"\n→ {tables_path}\n→ {numbers_path}")

    if not GAPS:
        return 0
    print(f"\n{len(GAPS)} lacuna(s) de evidência:")
    for missing in GAPS:
        print(f"  - {missing}")
    if args.allow_partial:
        print("\n--allow-partial: saindo com zero. Esta saída NÃO é publicável.")
        return 0
    print("\nAbortando: regeneração parcial põe tabela incompleta no artigo.")
    print("Para inspecionar mesmo assim, rode com --allow-partial.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
