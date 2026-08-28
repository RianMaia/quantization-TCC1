#!/usr/bin/env python3
"""M3.2 / M3.3 / M3.5 — Memória residente, latência e footprint, um braço por vez.

**Por que uma passada só.** As três medições exigem a mesma condição cara: uma
coleção sozinha no servidor, em container recém-subido. Coleções vizinhas
carregadas contaminam o page cache, que é exatamente onde a residência dos
vetores aparece. Como o contrato manda derrubar as outras, cada braço precisa
ser reconstruído depois que o anterior sai — fazer memória e latência em passadas
separadas custaria cinco reconstruções a mais sem medir nada de novo.

**Estados declarados.** A residência é lida em três: frio logo após o reinício,
morno após uma carga fixa de consultas no caminho de busca **do próprio braço**,
e exaustivo após busca exata. O estado vai junto de todo número.

**Emenda de 2026-08-19: o estado reportado é `warm_hnsw`, não `exhaustive`.**
S1.7 elegeu o exaustivo como teto por "tocar todos os vetores". Medido a 1M, ele
toca os vetores *errados*: o `exact` do Qdrant varre os originais float32 e
ignora os códigos, então a busca exaustiva pagina ~3 GB de originais em todo
braço quantizado — A1 saltou de 441 MB de page cache no morno para 3.042 no
exaustivo. É um caminho que nenhum deploy de braço comprimido executaria, e ele
apaga precisamente a compressão que o estudo mede. O exaustivo continua gravado,
como leitura do caminho dos originais, e não como teto.

**A janela de paginação anula a medição.** Não o volume de swap residente — a
taxa durante a janela. Ver `memory.SwapWatch`.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime

from quantptbr import corpus, index, manifest, memory, retrieval, server, stats

OUTPUT = corpus.REPO_ROOT / "runs" / "m3_arms.json"

#: Consultas de aquecimento, idênticas em todos os braços. Define o conjunto de
#: trabalho de forma reprodutível em vez de deixá-lo depender do acaso.
WARMUP_REPEATS = 4

#: Repetições da latência, depois do aquecimento descartado. 49 × 5 dá 245
#: amostras por braço, o bastante para um p95 que não dependa de uma consulta só.
LATENCY_REPEATS = 5


def owner_of(arm_id: str) -> str:
    """Quem constrói a coleção deste braço. O A5 mora na coleção do A4."""
    return retrieval.arm_config(arm_id).get("_shares_collection_with") or arm_id


def sole_collection(client, arm_id: str) -> dict | None:
    """Garante que só a coleção deste braço existe. Verificado, não suposto.

    Devolve o relatório do build quando precisou construir. Cada medição derruba
    as outras coleções, então esta passada reconstrói os mesmos braços várias
    vezes ao longo do estudo — e é de graça registrar isso, que é justamente a
    evidência de repetibilidade que M2.1 e M2.3 pedem.
    """
    wanted = retrieval.arm_config(arm_id)["collection"]
    for existing in [c.name for c in client.get_collections().collections]:
        if existing != wanted:
            print(f"    derrubando {existing}")
            client.delete_collection(existing)
    report = None
    if not client.collection_exists(wanted):
        print(f"    construindo {wanted}")
        report = index.build(client, owner_of(arm_id))
    remaining = [c.name for c in client.get_collections().collections]
    if remaining != [wanted]:
        raise RuntimeError(f"esperada só {wanted}, servidor tem {remaining}")
    return report


def latency_ms(client, arm_id: str, mode: str, repeats: int) -> list[float]:
    """Latência de cada consulta, cronometrada dentro do harness de recuperação.

    O p95 precisa de amostras por consulta: cronometrar passadas inteiras daria
    cinco números, e um p95 sobre cinco não é um p95.
    """
    samples: list[float] = []
    for _ in range(repeats):
        retrieval.retrieve(client, arm_id, mode, record_ms=samples)
    return samples


def summarise(samples: list[float]) -> dict:
    ordered = sorted(samples)
    index_p95 = min(int(0.95 * len(ordered)), len(ordered) - 1)
    return {
        "median": round(statistics.median(ordered), 2),
        "p95": round(ordered[index_p95], 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "samples": len(ordered),
    }


def measure(client, arm_id: str) -> dict:
    queries = len(retrieval.load_queries()[0])
    server.restart()
    client = server.client()
    server.require_server(client)

    # O build que acabou de rodar deixou o kernel recuperando páginas. Abrir a
    # janela agora anularia a medição pela nossa própria carga.
    if not memory.quiesce():
        print("    host ainda paginando após o prazo; a janela vai nascer suja")

    with memory.SwapWatch() as swap:
        states = {"cold": memory.read().as_dict()}

        # Aquecimento com os parâmetros do próprio braço: o A5 relê os originais
        # no rescoring, e é por isso que a residência dele difere da do A4 apesar
        # de compartilharem a coleção.
        for _ in range(WARMUP_REPEATS):
            retrieval.retrieve(client, arm_id, "hnsw")
        states["warm_hnsw"] = memory.read().as_dict()

        # Latência só depois do aquecimento: a primeira passada pagina os
        # segmentos do disco e mediria I/O de carga, não custo de serviço.
        samples = latency_ms(client, arm_id, "hnsw", LATENCY_REPEATS)

        retrieval.retrieve(client, arm_id, "exact")
        states["exhaustive"] = memory.read().as_dict()

    return {
        "arm": arm_id,
        "collection": retrieval.arm_config(arm_id)["collection"],
        "states": states,
        "reported_state": "warm_hnsw",
        "reported_resident_mb": states["warm_hnsw"]["total_mb"],
        "provisioning_floor_mb": states["warm_hnsw"]["anon_mb"],
        "_exhaustive_is_not_a_ceiling": (
            "O exact do Qdrant varre os originais e ignora os códigos, então o "
            "estado exaustivo mede o caminho dos originais em todo braço, não o "
            "teto do braço quantizado."
        ),
        "predicted_resident_mb": retrieval.arm_config(arm_id)["predicted_resident_mb"],
        "latency_ms_per_query": summarise(samples)
        | {
            "passes": LATENCY_REPEATS,
            "queries_per_pass": queries,
            "concurrency": 1,
            "residency_state": "warm_hnsw",
        },
        "rss_mb": memory.rss_mb(),
        "swap": swap.as_dict(),
        "footprint": index.footprint(retrieval.arm_config(arm_id)["collection"]),
    }


def persist(results: list[dict]) -> list[str]:
    """Grava o que já foi medido. Chamada a cada braço, não só no fim.

    Um braço trava de verdade — não só demora — e quando trava a exceção sobe
    do meio do laço. Escrever só no fim jogaria fora os braços já concluídos
    naquela mesma execução; a M4 não deveria pagar de novo por um braço que já
    saiu limpo só porque o vizinho travou.
    """
    ordered = sorted(results, key=lambda row: row["arm"])
    void = [r["arm"] for r in ordered if r["swap"]["is_void"]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "issues": ["RAF-67", "RAF-68", "RAF-70"],
                "finished_at": datetime.now(UTC).isoformat(),
                "warmup_repeats": WARMUP_REPEATS,
                "latency_repeats": LATENCY_REPEATS,
                "void_measurements": void,
                "arms": ordered,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return void


def main() -> int:
    manifest.require_clean_tree()
    server.start()

    # O A5 compartilha a coleção do A4: medir memória dele mediria o A4 outra
    # vez. A latência, essa sim, difere — o rescoring lê os originais do disco.
    arms = [a["id"] for a in stats.matrix()["arms"]]
    wanted = sys.argv[1:] or arms
    unknown = [a for a in wanted if a not in arms]
    if unknown:
        raise SystemExit(f"braço não pré-registrado: {unknown}")

    # Execução parcial não apaga medição válida: o arquivo é mesclado por braço.
    previous = json.loads(OUTPUT.read_text(encoding="utf-8"))["arms"] if OUTPUT.exists() else []
    results = [row for row in previous if row["arm"] not in wanted]

    for arm_id in wanted:
        print(f"\n{arm_id}")
        client = server.client()
        build_report = sole_collection(client, arm_id)
        row = measure(client, arm_id)
        row["build"] = build_report

        results.append(row)
        state = row["states"][row["reported_state"]]
        print(
            f"    residente {row['reported_resident_mb']:8.1f} MB em {row['reported_state']}   "
            f"(anon {state['anon_mb']:.1f} / file {state['file_mb']:.1f})   "
            f"previsto {row['predicted_resident_mb']}   "
            f"latência mediana {row['latency_ms_per_query']['median']:.1f} ms"
        )
        if row["swap"]["is_void"]:
            print("    MEDIÇÃO NULA: o host paginou durante a janela")
        persist(results)

    void = persist(results)
    print(f"\n→ {OUTPUT}")
    if void:
        print(f"braços a remedir: {void}")
    return 1 if void else 0


if __name__ == "__main__":
    raise SystemExit(main())
