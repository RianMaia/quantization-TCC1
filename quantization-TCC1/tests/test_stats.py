"""O bootstrap pareado, testado contra casos de resposta conhecida.

S1.2 exige isto explicitamente: o procedimento é validado em casos sintéticos
**antes** de tocar em resultado real. Um bootstrap com viés silencioso produziria
intervalos plausíveis e conclusões erradas, e não haveria como perceber olhando
para os números do estudo.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantptbr import stats


def test_identical_arms_are_indistinguishable():
    scores = np.random.default_rng(0).uniform(0, 1, size=50)
    result = stats.paired_bootstrap(scores, scores)
    assert result.mean_difference == 0.0
    assert result.ci_low == result.ci_high == 0.0
    assert result.verdict is stats.INDISTINGUISHABLE


def test_constant_offset_has_a_degenerate_interval():
    """Diferença sem variância: o intervalo colapsa no próprio deslocamento."""
    baseline = np.random.default_rng(1).uniform(0, 1, size=50)
    result = stats.paired_bootstrap(baseline, baseline - 0.05)
    assert result.mean_difference == pytest.approx(-0.05)
    assert result.ci_low == pytest.approx(-0.05)
    assert result.ci_high == pytest.approx(-0.05)
    assert result.verdict is stats.DEGRADES


def test_interval_matches_the_analytic_normal_answer():
    """O caso de resposta conhecida.

    Para diferenças normais e n grande, o IC percentílico do bootstrap tem de
    convergir para mu ± 1,96·sigma/√n. Se o reamostrador estiver pareando errado
    ou reamostrando o eixo errado, é aqui que aparece.
    """
    n, mu, sigma = 4000, -0.02, 0.15
    rng = np.random.default_rng(7)
    differences = rng.normal(mu, sigma, size=n)
    baseline = np.zeros(n)

    result = stats.paired_bootstrap(baseline, differences)
    half_width = 1.96 * differences.std(ddof=1) / np.sqrt(n)
    assert result.ci_low == pytest.approx(differences.mean() - half_width, abs=2e-3)
    assert result.ci_high == pytest.approx(differences.mean() + half_width, abs=2e-3)


def test_small_effect_under_large_noise_is_not_called_a_difference():
    """A situação que a regra existe para conter: efeito minúsculo, n=50."""
    rng = np.random.default_rng(3)
    baseline = rng.normal(0.4, 0.3, size=50)
    result = stats.paired_bootstrap(baseline, baseline + rng.normal(0.001, 0.3, size=50))
    assert result.verdict is stats.INDISTINGUISHABLE
    assert "não distinguível" in result.sentence("A1", "nDCG@10")


def test_large_effect_is_detected_at_fifty_queries():
    rng = np.random.default_rng(4)
    baseline = rng.normal(0.4, 0.1, size=50)
    assert stats.paired_bootstrap(baseline, baseline - 0.1).verdict is stats.DEGRADES
    assert stats.paired_bootstrap(baseline, baseline + 0.1).verdict is stats.IMPROVES


def test_the_same_seed_reproduces_the_same_interval():
    rng = np.random.default_rng(5)
    baseline, variant = rng.normal(size=50), rng.normal(size=50)
    assert stats.paired_bootstrap(baseline, variant) == stats.paired_bootstrap(baseline, variant)


def test_verdict_rule_is_mechanical():
    assert stats.verdict(-0.20, -0.01) is stats.DEGRADES
    assert stats.verdict(0.01, 0.20) is stats.IMPROVES
    assert stats.verdict(-0.01, 0.20) is stats.INDISTINGUISHABLE
    assert stats.verdict(0.0, 0.20) is stats.INDISTINGUISHABLE, "zero na borda não é diferença"


def test_mismatched_or_empty_input_is_refused():
    with pytest.raises(ValueError):
        stats.paired_bootstrap(np.zeros(50), np.zeros(49))
    with pytest.raises(ValueError):
        stats.paired_bootstrap(np.zeros(0), np.zeros(0))
