#!/usr/bin/env python3
"""M2.8 — Geometria dos vetores: o que decide se a quantização binária pode funcionar.

A binarização troca cada dimensão por um bit, cortando em zero. O que sobrevive
depende inteiramente de como os valores se distribuem **em torno de zero**, por
dimensão. Duas propriedades decidem:

* **Centragem.** Se uma dimensão fica quase toda de um lado do zero, quase todo
  vetor recebe o mesmo bit ali e a dimensão não distingue documento nenhum. Uma
  representação com muitas dimensões assim tem orçamento de bits efetivo muito
  menor que o nominal.
* **Dispersão.** Dimensões amontoadas junto do zero trocam de bit a qualquer
  perturbação, contribuindo ruído em vez de sinal.

**Este relatório é uma explicação, não uma predição.** A RAF-65 exigia registrar
a predição *antes* de olhar o nDCG do braço binário, e a ordem de execução foi
outra: o A4 já estava medido em 0,0773 quando esta análise rodou. O slot de
predição está perdido e fica registrado como perdido — apresentá-la agora como
predição seria falsificar a única coisa que a tornava uma.

O que o relatório ainda entrega: a propriedade da representação que *prevê* o
resultado, o que transfere para outros modelos e é útil a quem escolhe um
embedding pensando em compressão.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np

from quantptbr import corpus, embedding, vectors

OUTPUT = corpus.REPO_ROOT / "runs" / "m2.8_geometry.json"

#: Amostra. Grande o bastante para as estatísticas por dimensão estabilizarem,
#: pequena o bastante para não varrer os 3 GB inteiros.
SAMPLE = 200_000
SEED = 20260818

#: **Declarado antes de computar.** Uma dimensão é degenerada em sinal quando
#: esta fração ou mais dos vetores compartilha o mesmo bit ali. 0,95 é o corte;
#: escolhê-lo depois de ver a distribuição seria escolhê-lo para a história.
DEGENERACY_THRESHOLD = 0.95


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Bits de informação que o sinal de cada dimensão carrega.

    Uma dimensão perfeitamente equilibrada vale 1 bit; uma que sempre dá o mesmo
    sinal vale 0. A soma sobre as dimensões é o orçamento efetivo do código
    binário, ignorando correlação entre dimensões — que só pode reduzi-lo mais.
    """
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    return -(safe * np.log2(safe) + (1 - safe) * np.log2(1 - safe))


def main() -> int:
    envelope = embedding.Envelope.load()
    array = vectors.load()
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(array.shape[0], size=SAMPLE, replace=False))

    print(f"amostrando {SAMPLE:,} de {array.shape[0]:,} vetores de {envelope.dimensions}d …")
    sample = np.asarray(array[rows], dtype=np.float32)

    means = sample.mean(axis=0)
    stds = sample.std(axis=0)
    positive_fraction = (sample > 0).mean(axis=0)
    # Fração que compartilha o bit majoritário: 0,5 é equilíbrio, 1,0 é degenerada.
    majority = np.maximum(positive_fraction, 1 - positive_fraction)

    entropy = binary_entropy(positive_fraction)
    effective_bits = float(entropy.sum())
    degenerate = int((majority >= DEGENERACY_THRESHOLD).sum())

    # Quanto do desvio de cada dimensão é deslocamento da média, e não espalhamento
    # em torno dela. Alto significa que o corte em zero não passa pelo meio dos dados.
    offset_ratio = float(np.mean(np.abs(means) / stds))

    print(f"\n=== geometria de {envelope.model_id} ===")
    print(f"dimensões                          {envelope.dimensions}")
    print(f"média das médias por dimensão      {means.mean():+.5f}")
    print(f"|média|/desvio, média              {offset_ratio:.3f}")
    print(f"fração majoritária de sinal (mediana) {np.median(majority):.3f}")
    print(
        f"dimensões degeneradas (>= {DEGENERACY_THRESHOLD:.2f})   {degenerate} de {envelope.dimensions}"
    )
    print(f"bits efetivos                      {effective_bits:.1f} de {envelope.dimensions}")
    print(
        f"orçamento efetivo                  {effective_bits / envelope.dimensions:.1%} do nominal"
    )

    quartis = np.percentile(majority, [50, 75, 90, 99])
    print(f"fração majoritária p50/p75/p90/p99 {' '.join(f'{q:.3f}' for q in quartis)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "issue": "RAF-65",
                "computed_at": datetime.now(UTC).isoformat(),
                "_prediction_slot": (
                    "PERDIDO. A RAF-65 exigia a predição registrada antes de o nDCG do braço "
                    "binário ser examinado. O A4 já estava medido em 0,0773 quando esta análise "
                    "rodou, então isto é explicação post-hoc e está rotulado como tal."
                ),
                "model_id": envelope.model_id,
                "dimensions": envelope.dimensions,
                "sample_size": SAMPLE,
                "seed": SEED,
                "degeneracy_threshold": DEGENERACY_THRESHOLD,
                "_threshold_declared_before_computing": True,
                "mean_of_dimension_means": float(means.mean()),
                "mean_abs_offset_over_std": offset_ratio,
                "sign_majority": {
                    "median": float(np.median(majority)),
                    "p75": float(quartis[1]),
                    "p90": float(quartis[2]),
                    "p99": float(quartis[3]),
                },
                "degenerate_dimensions": degenerate,
                "effective_bits": effective_bits,
                "effective_fraction_of_nominal": effective_bits / envelope.dimensions,
                "per_dimension": {
                    "mean": means.tolist(),
                    "std": stds.tolist(),
                    "positive_fraction": positive_fraction.tolist(),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n→ {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
