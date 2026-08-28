#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from collections import Counter

from huggingface_hub import hf_hub_download

from quantptbr import corpus


def fetch() -> dict:
    files = {}
    for name, remote in corpus.REMOTE_FILES.items():
        print(f"baixando {remote} ...", flush=True)
        hf_hub_download(
            repo_id=corpus.DATASET,
            filename=remote,
            repo_type="dataset",
            revision=corpus.REVISION,
            local_dir=corpus.RAW_DIR,
        )
        path = corpus.local_path(name)
        files[name] = {
            "remote": remote,
            "bytes": path.stat().st_size,
            "sha256": corpus.sha256(path),
        }
        print(f"  {path.name}  {files[name]['bytes']:,} bytes  {files[name]['sha256'][:16]}…")
    return files


def write_manifest(files: dict) -> None:
    corpus.MANIFEST_PATH.write_text(
        json.dumps(
            {"dataset": corpus.DATASET, "revision": corpus.REVISION, "files": files},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nmanifesto escrito em {corpus.MANIFEST_PATH}")


def report() -> int:
    """Imprime o contrato do corpus e devolve o número de violações encontradas."""
    violations = 0

    passage_ids = corpus.load_passage_ids()
    qrels = corpus.load_qrels()
    topics = corpus.load_topics()
    judged = corpus.judged_query_ids(qrels)

    n_passages = len(passage_ids)
    unique = len(set(passage_ids))
    judged_counts = [len(v) for v in qrels.values()]
    mean_judged = sum(judged_counts) / len(judged_counts)
    grades = Counter(g for v in qrels.values() for g in v.values())
    with_positive = sum(1 for v in qrels.values() if any(g > 0 for g in v.values()))

    print("\n=== contrato do corpus ===")
    print(f"passagens                 {n_passages:,} (esperado {corpus.EXPECTED_PASSAGES:,})")
    print(f"passage_ids únicos        {unique:,}")
    print(f"tópicos no arquivo        {len(topics)}")
    print(f"consultas julgadas        {len(judged)} (esperado {corpus.EXPECTED_JUDGED_QUERIES})")
    print(f"média julgadas/consulta   {mean_judged:.2f} (esperado {corpus.EXPECTED_MEAN_JUDGED})")
    print(f"distribuição de graus     {dict(sorted(grades.items()))}")
    print(f"consultas com grau > 0    {with_positive}")

    if n_passages != corpus.EXPECTED_PASSAGES:
        print("VIOLAÇÃO: contagem de passagens diverge da figura publicada")
        violations += 1
    if unique != n_passages:
        print(f"VIOLAÇÃO: {n_passages - unique} passage_ids duplicados no corpus")
        violations += 1
    if len(judged) != corpus.EXPECTED_JUDGED_QUERIES:
        print("VIOLAÇÃO: número de consultas julgadas diverge da figura publicada")
        violations += 1

    known = set(passage_ids)
    orphans = {query_id: sorted(set(v) - known) for query_id, v in qrels.items() if set(v) - known}
    n_orphans = sum(len(v) for v in orphans.values())
    print(f"\ncobertura qrels→corpus    {n_orphans} IDs órfãos")
    if orphans:
        print("VIOLAÇÃO: qrels referenciam passagens ausentes do corpus")
        for query_id, ids in list(orphans.items())[:5]:
            print(f"  consulta {query_id}: {ids[:3]}{' …' if len(ids) > 3 else ''}")
        violations += 1

    missing_topics = [q for q in judged if q not in topics]
    if missing_topics:
        print(f"VIOLAÇÃO: consultas julgadas sem texto em topics: {missing_topics}")
        violations += 1

    return violations


def main() -> int:
    write_manifest(fetch())
    corpus.verify_manifest()
    violations = report()
    if violations:
        print(f"\n{violations} violação(ões) do contrato. Investigar antes de indexar.")
        return 1
    print("\ncontrato do corpus verificado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
