#!/usr/bin/env python3
"""S1.4 — Piloto de throughput e congelamento do envelope de embedding.

Converte em números medidos as três incógnitas que precedem o passe de 1M:

1. **`max_length`** — decidido pelo comprimento real das passagens em tokens,
   confrontado com o custo medido de cobri-lo. O `sentence-transformers` faz
   padding dinâmico por lote, então `max_length` **trunca** mas quase não
   encarece: a varredura de referência mede esse custo em vez de presumi-lo.
2. **`batch_size`** — o que maximiza throughput dentro de 11 GB de VRAM.
3. **fp16 vs fp32** — a Turing não tem bf16, e o estudo inteiro mede deltas
   pequenos de qualidade. Se o fp16 girar os vetores na ordem de grandeza do
   efeito da quantização, ele é confundidor e o passe roda em fp32. Isso se
   decide aqui, com evidência, e não na fase de análise.

   A rotação em cosseno é uma grandeza abstrata: 1e-4 é pouco ou muito? A
   pergunta só tem resposta nas unidades do próprio estudo, então o piloto
   também mede o que o fp16 faz com o **ranking** — e compara com o que a
   quantização int8, o braço mais suave da matriz, faz com o mesmo ranking. É
   essa razão que decide se o fp16 é confundidor, não um limiar escolhido a
   priori.

Saídas: `config/embedding_envelope.json` (a decisão, consumida por S1.5) e
`runs/s1.4_pilot.json` (a evidência que a sustenta).
"""

from __future__ import annotations

import heapq
import json
import math
import random
import subprocess
import time
from datetime import UTC, datetime

import numpy as np
import torch

from quantptbr import corpus, embedding

SEED = 0
SAMPLE_SIZE = 10_000
SWEEP_SIZE = 2_048
FP32_BATCH_SIZE = 32
BATCH_CANDIDATES = (16, 32, 64)

#: Profundidade do ranking comparado. É o mesmo k do nDCG@10, a métrica primária:
#: uma perturbação que não muda o top-10 não pode mover a métrica.
RANK_K = 10

#: Quantas das passagens mais longas do corpus inteiro são tokenizadas para
#: limitar a cauda. A amostra aleatória descreve o corpo da distribuição; ela não
#: enxerga o passageiro mais longo de 1M, que é justamente quem `max_length`
#: precisa cobrir.
LONGEST_K = 500

#: Teto para `max_length`: o limite posicional do modelo. Se a cauda do corpus
#: ultrapassá-lo, truncar deixa de ser evitável e vira decisão declarada, com a
#: fração truncada registrada em vez de silenciada.
MAX_LENGTH_CEILING = embedding.MODEL_MAX_LENGTH

#: Alternativa curta usada só como referência de custo: se cobrir a cauda inteira
#: custasse throughput, este é o valor que teria sido escolhido em seu lugar.
REFERENCE_COVERAGE = 0.99

RUNS_DIR = corpus.REPO_ROOT / "runs"
PILOT_PATH = RUNS_DIR / "s1.4_pilot.json"


def ceil32(n: float) -> int:
    return int(math.ceil(n / 32) * 32)


def scan_corpus(
    sample_size: int = SAMPLE_SIZE, longest_k: int = LONGEST_K, seed: int = SEED
) -> tuple[list[str], list[str]]:
    """Numa única leitura do corpus, colhe a amostra aleatória e as mais longas.

    Amostrar as primeiras n linhas refletiria um segmento do crawl e enviesaria a
    distribuição de comprimentos — que é justamente o que o piloto vai medir. As
    mais longas vêm por heap para que o limite da cauda seja do corpus inteiro, e
    não da amostra.
    """
    wanted = set(random.Random(seed).sample(range(corpus.EXPECTED_PASSAGES), sample_size))
    sample: list[str] = []
    heap: list[tuple[int, int, str]] = []
    for i, (_, text) in enumerate(corpus.iter_passages()):
        if i in wanted:
            sample.append(text)
        entry = (len(text), i, text)
        if len(heap) < longest_k:
            heapq.heappush(heap, entry)
        elif len(text) > heap[0][0]:
            heapq.heapreplace(heap, entry)
    longest = [text for _, _, text in sorted(heap, reverse=True)]
    return sample, longest


def token_lengths(model, texts: list[str]) -> np.ndarray:
    """Comprimentos em tokens sem truncamento, para enxergar a cauda inteira."""
    return np.array(
        [len(ids) for ids in model.tokenizer(texts, add_special_tokens=True)["input_ids"]],
        dtype=np.int32,
    )


def encode(model, texts: list[str], batch_size: int) -> tuple[np.ndarray, float]:
    """Codifica e devolve (vetores, segundos), com a GPU sincronizada nas pontas."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    torch.cuda.synchronize()
    return vectors, time.perf_counter() - start


def sweep(model, texts: list[str], max_length: int, label: str) -> list[dict]:
    """Throughput por batch size num `max_length`. OOM vira linha, não crash."""
    print(f"\nvarredura em fp16 — {label} (max_length {max_length}, {len(texts):,} passagens):")
    model.max_seq_length = max_length
    rows = []
    for batch_size in BATCH_CANDIDATES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            _, seconds = encode(model, texts, batch_size)
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch_size": batch_size, "oom": True})
            print(f"  batch {batch_size:>3}  OOM")
            continue
        row = {
            "batch_size": batch_size,
            "oom": False,
            "seconds": round(seconds, 2),
            "passages_per_second": round(len(texts) / seconds, 1),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        }
        rows.append(row)
        print(
            f"batch {batch_size:>3}  {row['passages_per_second']:>7.1f} pass/s"
            f"pico {row['peak_vram_mib']:>5} MiB"
        )
    return rows


def fidelity(fp16_vectors: np.ndarray, fp32_vectors: np.ndarray) -> dict:
    """Perturbação **angular** entre a mesma passagem codificada em fp16 e fp32.

    Renormalizar em float64 antes do produto interno não é preciosismo: o fp16
    guarda ~3 dígitos decimais, então um vetor que saiu normalizado tem norma
    1 ± 1e-4 quando relido em precisão dupla. Sem renormalizar, o "cosseno"
    passa de 1 e a métrica mistura deriva de norma com rotação — e é só a
    rotação que compete com o efeito da quantização.

    A deriva de norma é reportada à parte porque S1.5 precisa dela: os vetores
    vão para o índice em float32 e têm de ser renormalizados na escrita.
    """
    a = fp16_vectors.astype(np.float64)
    b = fp32_vectors.astype(np.float64)
    fp16_norms = np.linalg.norm(a, axis=1)
    cosines = np.clip(
        np.einsum("ij,ij->i", a / fp16_norms[:, None], b / np.linalg.norm(b, axis=1)[:, None]),
        -1.0,
        1.0,
    )
    return {
        "n": int(cosines.size),
        "min": float(cosines.min()),
        "p01": float(np.percentile(cosines, 1)),
        "median": float(np.median(cosines)),
        "worst_angular_perturbation": float(1 - cosines.min()),
        "fp16_norm_drift": {
            "min": float(fp16_norms.min()),
            "max": float(fp16_norms.max()),
            "worst_abs": float(np.abs(fp16_norms - 1).max()),
        },
    }


def simulate_scalar_int8(vectors: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """Aproxima o quantizador escalar do Qdrant: mapa linear global para 256 níveis.

    É uma aproximação — o quantizador real vive no servidor e será medido em M2.
    Serve aqui só como régua: dá a perturbação do braço **mais suave** da matriz,
    contra a qual a perturbação do fp16 tem de ser desprezível para não ser
    confundidor.
    """
    lo, hi = np.quantile(vectors, [1 - quantile, quantile])
    scale = (hi - lo) / 255
    levels = np.clip(np.round((vectors - lo) / scale), 0, 255)
    return (levels * scale + lo).astype(np.float32)


def top_k(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Índices do top-k por produto interno. Os vetores já saem normalizados."""
    return np.argsort(-(queries @ docs.T), axis=1)[:, :RANK_K]


def rank_agreement(
    base_queries: np.ndarray,
    base_docs: np.ndarray,
    variant_queries: np.ndarray,
    variant_docs: np.ndarray,
) -> dict:
    """Quanto o top-k muda entre duas execuções do pipeline inteiro.

    Consultas e documentos entram como par porque é assim que o pipeline roda:
    trocar a precisão troca as duas pontas, e medir só o lado dos documentos
    subestimaria o efeito.
    """
    a = top_k(base_queries, base_docs)
    b = top_k(variant_queries, variant_docs)
    overlap = [len(set(x) & set(y)) / RANK_K for x, y in zip(a, b, strict=True)]
    return {
        "queries": int(base_queries.shape[0]),
        "documents": int(base_docs.shape[0]),
        "k": RANK_K,
        "identical_order": round(
            float(np.mean([list(x) == list(y) for x, y in zip(a, b, strict=True)])), 4
        ),
        "identical_set": round(
            float(np.mean([set(x) == set(y) for x, y in zip(a, b, strict=True)])), 4
        ),
        "mean_overlap": round(float(np.mean(overlap)), 6),
    }


def main() -> int:
    corpus.verify_manifest()
    RUNS_DIR.mkdir(exist_ok=True)

    print(f"lendo o corpus: amostra de {SAMPLE_SIZE:,} (seed {SEED}) + {LONGEST_K} mais longas …")
    texts, longest = scan_corpus()

    # Carregado longo primeiro: o tokenizer independe de max_length, e é a
    # distribuição medida que vai escolher o valor definitivo.
    model = embedding.load_model("float16", max_length=MAX_LENGTH_CEILING)

    lengths = token_lengths(model, [embedding.PASSAGE_PREFIX + t for t in texts])
    tail = token_lengths(model, [embedding.PASSAGE_PREFIX + t for t in longest])
    corpus_token_max = int(tail.max())
    max_length = min(ceil32(corpus_token_max), MAX_LENGTH_CEILING)
    reference_length = ceil32(float(np.quantile(lengths, REFERENCE_COVERAGE)))

    length_stats = {
        "sample_n": int(lengths.size),
        "mean": round(float(lengths.mean()), 1),
        "p50": int(np.percentile(lengths, 50)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "sample_max": int(lengths.max()),
        "longest_k": LONGEST_K,
        "corpus_token_max": corpus_token_max,
        "chosen_max_length": max_length,
        "reference_max_length": reference_length,
        "sample_fraction_truncated": round(float((lengths > max_length).mean()), 5),
        "tail_fraction_truncated": round(float((tail > max_length).mean()), 5),
    }
    print(
        f"\ntokens na amostra: mediana {length_stats['p50']}, p95 {length_stats['p95']}, "
        f"p99 {length_stats['p99']}, máx {length_stats['sample_max']}"
    )
    print(f"máximo entre as {LONGEST_K} passagens mais longas do corpus: {corpus_token_max}")
    print(
        f"max_length = {max_length} (trunca {length_stats['tail_fraction_truncated']:.2%} da cauda)"
    )

    sweep_texts = [embedding.PASSAGE_PREFIX + t for t in texts[:SWEEP_SIZE]]
    rows = sweep(model, sweep_texts, max_length, "escolhido")
    usable = [r for r in rows if not r["oom"]]
    if not usable:
        raise RuntimeError("todo batch size deu OOM; reveja max_length ou a VRAM disponível")
    best = max(usable, key=lambda r: r["passages_per_second"])
    batch_size = best["batch_size"]
    print(f"escolhido batch_size {batch_size}")

    # Sem esta referência, "cobrir a cauda inteira" seria uma afirmação sem
    # preço. Com ela, o artigo pode dizer quanto custou não truncar.
    reference_rows = (
        sweep(model, sweep_texts, reference_length, f"referência p{REFERENCE_COVERAGE:.0%}")
        if reference_length != max_length
        else []
    )

    model.max_seq_length = max_length
    judged = corpus.judged_query_ids()
    topics = corpus.load_topics()
    queries = [topics[q] for q in judged]

    passages = [embedding.PASSAGE_PREFIX + t for t in texts]
    queries = [embedding.QUERY_PREFIX + q for q in queries]

    print(f"\ncodificando {SAMPLE_SIZE:,} passagens em fp16 (medição de throughput) …")
    fp16_docs, seconds = encode(model, passages, batch_size)
    rate = len(passages) / seconds
    projection_hours = corpus.EXPECTED_PASSAGES / rate / 3600
    print(f"{rate:.1f} pass/s → projeção de {projection_hours:.2f} h para 1M")
    fp16_queries, _ = encode(model, queries, batch_size)
    encode_dtype = str(fp16_docs.dtype)

    pooling = embedding.pooling_mode(model)
    del model
    torch.cuda.empty_cache()

    print(f"codificando as mesmas {SAMPLE_SIZE:,} passagens em fp32 (contrafactual) …")
    fp32_model = embedding.load_model("float32", max_length=max_length)
    fp32_docs, _ = encode(fp32_model, passages, FP32_BATCH_SIZE)
    fp32_queries, _ = encode(fp32_model, queries, FP32_BATCH_SIZE)
    del fp32_model
    torch.cuda.empty_cache()

    agreement = fidelity(fp16_docs, fp32_docs)
    print(
        f"\nrotação fp16×fp32: pior 1−cos = {agreement['worst_angular_perturbation']:.2e}"
        f"  (p01 cos {agreement['p01']:.7f})"
    )
    print(
        f"deriva de norma do fp16: até {agreement['fp16_norm_drift']['worst_abs']:.2e} "
        f"— S1.5 renormaliza na escrita"
    )

    fp32_docs = fp32_docs.astype(np.float32)
    ranks = {
        "fp16": rank_agreement(
            fp32_queries, fp32_docs, fp16_queries.astype(np.float32), fp16_docs.astype(np.float32)
        ),
        # A régua: o braço mais suave da matriz, com a consulta em fp32 — é assim
        # que o Qdrant compara uma consulta contra vetores quantizados.
        "simulated_int8": rank_agreement(
            fp32_queries, fp32_docs, fp32_queries, simulate_scalar_int8(fp32_docs)
        ),
    }
    print(f"\ntop-{RANK_K} sobre {len(judged)} consultas julgadas × {SAMPLE_SIZE:,} passagens:")
    for name, r in ranks.items():
        print(
            f"  {name:<15} sobreposição {r['mean_overlap']:.4f}"
            f"  conjunto idêntico {r['identical_set']:.0%}"
            f"  ordem idêntica {r['identical_order']:.0%}"
        )

    envelope = embedding.Envelope(
        model_id=embedding.MODEL_ID,
        model_revision=embedding.MODEL_REVISION,
        dimensions=embedding.DIMENSIONS,
        max_length=max_length,
        batch_size=batch_size,
        dtype="float16",
        pooling=pooling,
        normalize=True,
        query_prefix=embedding.QUERY_PREFIX,
        passage_prefix=embedding.PASSAGE_PREFIX,
    )
    envelope.save()

    PILOT_PATH.write_text(
        json.dumps(
            {
                "issue": "RAF-53",
                "finished_at": datetime.now(UTC).isoformat(),
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=corpus.REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip(),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "corpus_sha256": corpus.read_manifest()["files"]["corpus"]["sha256"],
                "seed": SEED,
                "sample_size": SAMPLE_SIZE,
                "token_lengths": length_stats,
                "batch_sweep": rows,
                "batch_sweep_reference": reference_rows,
                "encode_output_dtype": encode_dtype,
                "rank_agreement": ranks,
                "throughput": {
                    "passages_per_second": round(rate, 1),
                    "seconds_for_sample": round(seconds, 1),
                    "projected_hours_for_1m": round(projection_hours, 2),
                },
                "fp16_vs_fp32": agreement,
                "judged_queries": len(judged),
                "envelope": json.loads(embedding.ENVELOPE_PATH.read_text(encoding="utf-8")),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nenvelope → {embedding.ENVELOPE_PATH}")
    print(f"evidência → {PILOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
