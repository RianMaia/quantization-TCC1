"""Protocolo estatístico pré-registrado em S1.2.

A regra do veredito é mecânica de propósito. Com 50 consultas, a tentação na
hora da escrita é chamar uma diferença pequena de degradação; um procedimento
que decide o verbo a partir do intervalo, sem julgamento no momento da redação,
é o que torna a contenção final crível em vez de post-hoc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from quantptbr.corpus import REPO_ROOT

MATRIX_PATH = REPO_ROOT / "config" / "arms.json"

DEGRADES = "degrada"
IMPROVES = "melhora"
INDISTINGUISHABLE = "não distinguível de ruído neste tamanho de amostra"


def matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Comparison:
    """Uma diferença pareada contra o A0, com o veredito já decidido pelo intervalo."""

    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    verdict: str

    def sentence(self, arm: str, metric: str) -> str:
        """A frase que vai ao artigo, com o verbo amarrado ao intervalo."""
        interval = f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
        if self.verdict is INDISTINGUISHABLE:
            return (
                f"{arm}: diferença de {metric} de {self.mean_difference:+.4f} contra o A0, "
                f"IC 95% {interval} — {INDISTINGUISHABLE}."
            )
        return (
            f"{arm} {self.verdict} em {metric}: {self.mean_difference:+.4f} contra o A0, "
            f"IC 95% {interval}."
        )


def verdict(ci_low: float, ci_high: float) -> str:
    """A regra pré-registrada, aplicada sem julgamento.

    Um intervalo que contém zero nunca é reportado como equivalência: ausência
    de evidência de diferença não é evidência de ausência de diferença.
    """
    if ci_high < 0:
        return DEGRADES
    if ci_low > 0:
        return IMPROVES
    return INDISTINGUISHABLE


def paired_bootstrap(
    baseline: np.ndarray,
    variant: np.ndarray,
    resamples: int | None = None,
    seed: int | None = None,
    confidence: float | None = None,
) -> Comparison:
    """Bootstrap pareado por consulta sobre a diferença de médias (variante − A0).

    O pareamento é o que dá poder a n=50: consultas difíceis são difíceis em
    todos os braços, e reamostrar a *diferença* remove essa variação entre
    consultas em vez de deixá-la dominar o intervalo.

    Os parâmetros omitidos vêm de `config/arms.json`, para que uma análise não
    possa usar silenciosamente outra contagem de reamostras ou outra semente.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    variant = np.asarray(variant, dtype=np.float64)
    if baseline.shape != variant.shape or baseline.ndim != 1:
        raise ValueError(
            f"esperados dois vetores pareados de mesmo tamanho, {baseline.shape} e {variant.shape}"
        )
    if baseline.size == 0:
        raise ValueError("nada a comparar: zero consultas")

    settings = matrix()["statistics"]
    resamples = settings["resamples"] if resamples is None else resamples
    seed = settings["seed"] if seed is None else seed
    confidence = settings["confidence"] if confidence is None else confidence

    differences = variant - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[indices].mean(axis=1)

    tail = (1 - confidence) / 2 * 100
    ci_low, ci_high = (float(x) for x in np.percentile(means, [tail, 100 - tail]))
    return Comparison(
        n=int(differences.size),
        mean_difference=float(differences.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        verdict=verdict(ci_low, ci_high),
    )
