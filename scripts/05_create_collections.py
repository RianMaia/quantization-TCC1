#!/usr/bin/env python3
"""M2 — Constrói as coleções dos braços a partir da matriz pré-registrada.

Um script só para todos os braços, porque a lista de braços é argumento e não
código. Produzir o A1 exige passar `A1`; não exige editar nada. É essa
propriedade que garante que a diferença entre dois braços na tabela final seja a
diferença que a matriz declara, e não uma divergência acidental de build.

    python scripts/05_create_collections.py A0
    python scripts/05_create_collections.py A1 A2 A3 A4 --rebuild

O A5 não é construível: ele compartilha a coleção do A4 e difere apenas em
parâmetros de busca.
"""

from __future__ import annotations

import argparse

from quantptbr import corpus, index, manifest, server

RUNS_DIR = corpus.REPO_ROOT / "runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", metavar="BRAÇO", help="ids da matriz, ex.: A0 A1")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="descarta uma coleção já existente em vez de abortar",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continua um envio interrompido em vez de abortar; os IDs já presentes são conferidos",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permite árvore git suja; a execução fica registrada como não citável",
    )
    args = parser.parse_args()

    # Falha cedo: descobrir a árvore suja depois de meia hora de build é caro.
    manifest.require_clean_tree(args.allow_dirty)

    server.start()
    client = server.client()

    for arm_id in args.arms:
        report = index.build(client, arm_id, recreate=args.rebuild, resume=args.resume)
        path = manifest.RunManifest.capture(
            arm_id=arm_id, allow_dirty=args.allow_dirty, build=report
        ).save(RUNS_DIR / f"m2_{arm_id.lower()}_build.json")
        print(f"manifesto → {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
