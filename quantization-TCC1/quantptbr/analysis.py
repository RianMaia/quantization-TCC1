"""Diferenças pareadas contra o A0, com intervalo — o protocolo de S1.2 aplicado.

Com 49 consultas, a tentação na hora da escrita é chamar uma diferença pequena
de degradação. Nada aqui decide verbo: o veredito sai de `stats.verdict`, que lê
o intervalo. O que este módulo faz é parear as consultas certas, recusar agregar
execuções incomparáveis, e calcular o que o desenho **não** teria conseguido ver.

Essa última parte não é enfeite. Reportar "não distinguível de ruído" sem dizer
qual diferença o estudo teria detectado deixa o leitor sem saber se o intervalo
é informativo ou se o experimento era cego. A diferença mínima detectável é o
que separa as duas leituras.
"""

from __future__ import annotations

import json
import math

import numpy as np

from quantptbr import corpus, manifest, stats

RUNS_DIR = corpus.REPO_ROOT / "runs"

#: Potência assumida ao reportar a diferença mínima detectável. Convenção, e
#: declarada porque muda o número: com potência maior, o desenho "vê" menos.
POWER = 0.80

BASELINE = "A0"


def evaluation_path(arm_id: str, mode: str):
    return RUNS_DIR / f"eval_{arm_id.lower()}_{mode}.json"


def load(arm_id: str, mode: str) -> dict:
    path = evaluation_path(arm_id, mode)
    if not path.exists():
        raise FileNotFoundError(f"{path} ausente. Rode scripts/06_evaluate_benchmark.py {arm_id}.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_all(mode: str) -> dict[str, dict]:
    """Todas as avaliações de um modo, já recusadas se incomparáveis entre si."""
    arms = [arm["id"] for arm in stats.matrix()["arms"]]
    loaded = {arm: load(arm, mode) for arm in arms}
    manifest.require_same_envelope(list(loaded.values()))
    return loaded


def aligned(baseline: dict, variant: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Os dois vetores por consulta, na mesma ordem.

    O pareamento é o que dá poder a n=49, e parear na ordem errada destruiria
    exatamente isso sem que nenhum número parecesse estranho.
    """
    left = baseline["evaluation"]["per_query"][metric]
    right = variant["evaluation"]["per_query"][metric]
    if set(left) != set(right):
        raise ValueError(f"{metric}: conjuntos de consultas divergem entre as duas execuções")
    query_ids = sorted(left, key=int)
    return (
        np.array([left[q] for q in query_ids], dtype=np.float64),
        np.array([right[q] for q in query_ids], dtype=np.float64),
    )


def minimum_detectable_difference(
    baseline: np.ndarray, variant: np.ndarray, power: float = POWER
) -> float:
    """A menor diferença que este desenho detectaria de forma confiável.

    Sai da variância *observada* das diferenças pareadas, não de uma suposição.
    É o número que diz se um intervalo contendo zero é informativo ou se o
    experimento simplesmente não enxergava nada desse tamanho.
    """
    from scipy.stats import norm

    differences = variant - baseline
    spread = float(differences.std(ddof=1))
    confidence = stats.matrix()["statistics"]["confidence"]
    z_alpha = norm.ppf(1 - (1 - confidence) / 2)
    z_beta = norm.ppf(power)
    return float((z_alpha + z_beta) * spread / math.sqrt(differences.size))


def compare(mode: str, metric: str) -> list[dict]:
    """Cada braço contra o A0, no mesmo modo e na mesma métrica."""
    loaded = load_all(mode)
    baseline = loaded[BASELINE]
    rows = []
    for arm_id, evaluation in loaded.items():
        if arm_id == BASELINE:
            continue
        left, right = aligned(baseline, evaluation, metric)
        comparison = stats.paired_bootstrap(left, right)
        rows.append(
            {
                "arm": arm_id,
                "mode": mode,
                "metric": metric,
                "baseline_mean": float(left.mean()),
                "variant_mean": float(right.mean()),
                "mean_difference": comparison.mean_difference,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "verdict": comparison.verdict,
                "sentence": comparison.sentence(arm_id, metric),
                "minimum_detectable_difference": minimum_detectable_difference(left, right),
                # Zero exato em todas as consultas não é um achado sobre
                # quantização: é o modo de busca devolvendo o mesmo run.
                "identical_by_construction": bool(np.abs(right - left).max() == 0.0),
                "n": comparison.n,
            }
        )
    return rows


def report() -> dict:
    """A tabela inteira: todo braço, todo modo, toda métrica pré-registrada."""
    matrix = stats.matrix()
    metrics = list(load(BASELINE, matrix["search_modes"][0])["evaluation"]["per_query"])
    rows = [
        row
        for mode in matrix["search_modes"]
        for metric in metrics
        for row in compare(mode, metric)
    ]
    return {
        "baseline": BASELINE,
        "protocol": matrix["statistics"],
        "power_for_detectable_difference": POWER,
        "comparisons": rows,
    }
