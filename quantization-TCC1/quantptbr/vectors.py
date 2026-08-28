"""O artefato de vetores: caminhos, contrato e leitura por memória mapeada.

A linha *i* do array é a passagem *i* na ordem de `corpus.iter_passages()`. Essa
correspondência é posicional e não tem tabela de tradução — é por isso que a
ordem do corpus é um contrato congelado em S1.0. Um braço que leia este array
com outra ordem produziria nDCG plausível e errado.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import time
from datetime import UTC, datetime

import numpy as np

from quantptbr import corpus, embedding
from quantptbr.corpus import REPO_ROOT

VECTORS_DIR = REPO_ROOT / "data" / "vectors"
ARRAY_PATH = VECTORS_DIR / "quati_1m_float32.npy"
PROGRESS_PATH = VECTORS_DIR / "progress.json"

#: Manifesto do passe. Fica em `runs/` porque é versionado: o array de 3 GB não
#: entra no git, mas o hash que o identifica entra.
MANIFEST_PATH = REPO_ROOT / "runs" / "s1.5_vectors.json"

#: Passagens por checkpoint. Define também os limites de retomada: um passe
#: interrompido recomeça no múltiplo anterior, e como a composição dos lotes
#: depende só do conteúdo do bloco, o resultado é idêntico ao de um passe único.
CHUNK_SIZE = 50_000


def load(mode: str = "r") -> np.memmap:
    """Abre o array por memória mapeada. Nunca carregue 3 GB para a RAM."""
    if not ARRAY_PATH.exists():
        raise FileNotFoundError(f"{ARRAY_PATH} ausente. Rode scripts/03_generate_embeddings.py.")
    return np.load(ARRAY_PATH, mmap_mode=mode)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=corpus.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def read_progress() -> int:
    if not PROGRESS_PATH.exists():
        return 0
    return int(json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))["rows_done"])


def write_progress(rows_done: int) -> None:
    PROGRESS_PATH.write_text(json.dumps({"rows_done": rows_done}) + "\n", encoding="utf-8")


def open_array(rows: int, dims: int) -> np.memmap:
    """Abre o array final, criando-o com o cabeçalho .npy se ainda não existir."""
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    if ARRAY_PATH.exists():
        return np.lib.format.open_memmap(ARRAY_PATH, mode="r+")
    return np.lib.format.open_memmap(ARRAY_PATH, mode="w+", dtype=np.float32, shape=(rows, dims))


def normalise(block: np.ndarray) -> np.ndarray:
    """Renormaliza em float32 antes de gravar.

    Não é redundante com `normalize_embeddings=True`: o modelo roda em fp16, e
    S1.4 mediu as normas de saída derivando até 4,86e-4. Gravar sem renormalizar
    entregaria ao índice vetores que não são unitários, e a métrica de distância
    pressupõe que sejam.
    """
    block = block.astype(np.float32)
    return block / np.linalg.norm(block, axis=1, keepdims=True)


def run_pass(envelope: embedding.Envelope) -> dict:
    total = corpus.EXPECTED_PASSAGES
    done = read_progress()
    array = open_array(total, envelope.dimensions)

    if done >= total:
        print(f"passe já completo ({done:,} linhas). Nada a fazer.")
        return {"resumed_from": done, "encoded": 0, "seconds": 0.0}

    print(f"retomando em {done:,} de {total:,}" if done else f"iniciando passe de {total:,}")
    model = embedding.load_model(envelope.dtype, envelope.max_length)

    stream = itertools.islice(corpus.iter_passages(), done, None)
    started = time.perf_counter()
    encoded = 0

    while done < total:
        size = min(CHUNK_SIZE, total - done)
        block = list(itertools.islice(stream, size))
        if len(block) != size:
            raise RuntimeError(f"corpus acabou em {done + len(block):,}, esperado {total:,}")

        texts = [envelope.passage_prefix + text for _, text in block]
        chunk_started = time.perf_counter()
        out = model.encode(
            texts,
            batch_size=envelope.batch_size,
            normalize_embeddings=envelope.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        array[done : done + size] = normalise(out)

        # Ordem obrigatória: os bytes primeiro, o progresso depois. O inverso
        # deixaria um bloco parcial marcado como concluído após uma queda.
        array.flush()
        done += size
        write_progress(done)

        encoded += size
        rate = size / (time.perf_counter() - chunk_started)
        remaining = (total - done) / rate / 60
        print(f"  {done:>9,}/{total:,}  {rate:6.1f} pass/s  faltam {remaining:5.1f} min")

    seconds = time.perf_counter() - started
    del model
    return {"resumed_from": read_progress() - encoded, "encoded": encoded, "seconds": seconds}


def verify(envelope: embedding.Envelope) -> None:
    """Verificações que abortam o passe em vez de deixar passar vetor ruim."""
    array = load()
    total = corpus.EXPECTED_PASSAGES

    print("\n=== verificação do artefato ===")
    if array.shape != (total, envelope.dimensions):
        raise RuntimeError(f"forma {array.shape} != ({total}, {envelope.dimensions})")
    print(f"forma                     {array.shape}")

    # Em blocos: o array tem 3 GB e não cabe em RAM de uma vez.
    worst_norm_error = 0.0
    zero_rows = nonfinite_rows = 0
    for start in range(0, total, CHUNK_SIZE):
        block = np.asarray(array[start : start + CHUNK_SIZE], dtype=np.float32)
        finite = np.isfinite(block).all(axis=1)
        nonfinite_rows += int((~finite).sum())
        norms = np.linalg.norm(block, axis=1)
        zero_rows += int((norms == 0).sum())
        worst_norm_error = max(worst_norm_error, float(np.abs(norms[finite] - 1).max()))

    print(f"linhas não finitas        {nonfinite_rows}")
    print(f"linhas nulas              {zero_rows}")
    print(f"pior erro de norma        {worst_norm_error:.2e}")
    if nonfinite_rows or zero_rows:
        raise RuntimeError("o artefato contém linhas inválidas")
    if worst_norm_error > 1e-5:
        raise RuntimeError(f"vetores não unitários: erro de norma {worst_norm_error:.2e}")

    check_row_correspondence(array, envelope)
    check_relevance_signal(array)


def check_row_correspondence(array: np.ndarray, envelope: embedding.Envelope) -> None:
    """A linha *i* é mesmo a passagem *i*?

    Recodifica uma amostra e exige que cada linha guardada seja mais parecida com
    a recodificação da **sua própria** passagem do que com a de qualquer outra da
    amostra. Compara o argmax, e não um limiar de cosseno: é isso que pega um
    deslocamento de uma posição, que um limiar frouxo deixaria passar.
    """
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(array.shape[0], size=256, replace=False))
    wanted = set(indices.tolist())
    texts = [t for i, (_, t) in enumerate(corpus.iter_passages()) if i in wanted]

    model = embedding.load_model(envelope.dtype, envelope.max_length)
    fresh = normalise(
        model.encode(
            [envelope.passage_prefix + t for t in texts],
            batch_size=envelope.batch_size,
            normalize_embeddings=envelope.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    del model

    stored = np.asarray(array[indices], dtype=np.float32)
    matches = int((np.argmax(stored @ fresh.T, axis=1) == np.arange(len(indices))).sum())
    print(f"correspondência linha↔passagem  {matches}/{len(indices)}")
    if matches != len(indices):
        raise RuntimeError(f"{len(indices) - matches} linhas não correspondem à sua passagem")


def check_relevance_signal(array: np.ndarray) -> None:
    """O modelo foi aplicado certo, ou produziu ruído bem formado?

    Usa o gabarito oficial: para cada consulta julgada, as passagens de grau 3
    têm de pontuar acima de passagens aleatórias. Não é uma métrica reportada —
    é a diferença entre "os vetores existem" e "os vetores significam algo".
    """
    envelope = embedding.Envelope.load()
    qrels = corpus.load_qrels()
    topics = corpus.load_topics()
    row_of = {pid: i for i, pid in enumerate(corpus.load_passage_ids())}

    queries = {q: v for q, v in qrels.items() if any(g == 3 for g in v.values())}
    model = embedding.load_model(envelope.dtype, envelope.max_length)
    query_vectors = normalise(
        model.encode(
            [envelope.query_prefix + topics[q] for q in queries],
            batch_size=envelope.batch_size,
            normalize_embeddings=envelope.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    del model

    rng = np.random.default_rng(1)
    background = np.asarray(array[np.sort(rng.choice(array.shape[0], 2000, replace=False))])
    wins = 0
    for vector, (_, judgements) in zip(query_vectors, queries.items(), strict=True):
        relevant = [row_of[p] for p, g in judgements.items() if g == 3]
        relevant_score = float((np.asarray(array[sorted(relevant)]) @ vector).mean())
        if relevant_score > float((background @ vector).mean()):
            wins += 1
    print(f"sinal de relevância             {wins}/{len(queries)} consultas")
    if wins != len(queries):
        raise RuntimeError(
            f"{len(queries) - wins} consultas pontuam passagens de grau 3 abaixo do acaso"
        )


def write_manifest(envelope: embedding.Envelope, timing: dict) -> None:
    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "issue": "RAF-54",
                "finished_at": datetime.now(UTC).isoformat(),
                "git_commit": git_commit(),
                "corpus_sha256": corpus.read_manifest()["files"]["corpus"]["sha256"],
                "corpus_revision": corpus.REVISION,
                "passages": corpus.EXPECTED_PASSAGES,
                "envelope": json.loads(embedding.ENVELOPE_PATH.read_text(encoding="utf-8")),
                "array": {
                    "path": str(ARRAY_PATH.relative_to(corpus.REPO_ROOT)),
                    "dtype": "float32",
                    "shape": [corpus.EXPECTED_PASSAGES, envelope.dimensions],
                    "bytes": ARRAY_PATH.stat().st_size,
                    "sha256": corpus.sha256(ARRAY_PATH),
                },
                "timing": timing,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
