#!/usr/bin/env python3
"""M4.3 — Separa o erro de quantização do erro do grafo, varrendo `ef`.

**Por que não é com o modo exato.** S1.2 pré-registrou a decomposição como
"exato do braço contra exato do A0". Medido em 2026-08-19, isso é impossível: o
`exact: true` do Qdrant varre os vetores **originais** e ignora a quantização.
Os seis braços devolveram ranks e scores byte-idênticos, inclusive o A4 com
`rescore=False`. A coluna é constante por construção, e teria ido para a tabela
como "quantização não custa nada sob busca exaustiva".

**O instrumento que sobra.** Conforme o `ef` cresce, a busca no grafo visita
mais candidatos e o erro de aproximação tende a zero — mas ela continua
pontuando pelos **códigos quantizados**. O que sobra no platô contra o A0 exato
é erro de quantização puro, medido sobre os códigos reais do servidor e não
sobre uma simulação nossa.

O A0 é varrido junto e serve de controle: se o A0 no maior `ef` não converge
para o A0 exato, a varredura não chegou ao platô e nenhuma leitura vale.

**Dois held-constants são afrouxados aqui, de propósito e sob rótulo:** `ef`, que
é varrido, e `rescore`, forçado a falso em todos os braços — com rescoring a
busca relê os originais e todo braço converge para o float32, que é exatamente o
efeito que apaga a medição. Nenhuma tabela reportada usa este script; ele produz
o diagnóstico de decomposição e nada mais.
"""

from __future__ import annotations

import json
import time

from quantptbr import corpus, evaluation, manifest, retrieval, server, stats

OUTPUT = corpus.REPO_ROOT / "runs" / "m4_decomposition.json"

#: Escada de `ef`. O primeiro valor é o pré-registrado, para que a linha de cima
#: da tabela seja comparável ao que M3.1 reportou.
EF_LADDER = (128, 512, 2048, 8192)

METRIC = "nDCG@10"


def sweep(client, arm_id: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ef in EF_LADDER:
        started = time.perf_counter()
        rows = retrieval.retrieve(client, arm_id, "hnsw", ef=ef, rescore=False)
        value = evaluation.evaluate(retrieval.as_run(rows))["aggregate"][METRIC]
        scores[ef] = value
        print(f"    ef {ef:>5}  {METRIC} {value:.4f}   ({time.perf_counter() - started:5.1f}s)")
    return scores


def main() -> int:
    manifest.require_clean_tree()
    server.start()
    client = server.client()

    # A referência é o float32 exaustivo: o teto que qualquer braço alcançaria
    # se a quantização não custasse nada e o grafo fosse perfeito.
    reference = evaluation.evaluate(retrieval.as_run(retrieval.load_run("A0", "exact")))
    ceiling = reference["aggregate"][METRIC]
    print(f"referência: A0 exato (float32 exaustivo)  {METRIC} {ceiling:.4f}\n")

    # O A5 compartilha a coleção do A4 e difere só em rescoring, que aqui está
    # desligado nos dois: varrê-lo mediria o A4 outra vez.
    arms = [a["id"] for a in stats.matrix()["arms"] if not a.get("_shares_collection_with")]

    results = {}
    for arm_id in arms:
        print(f"  {arm_id}")
        scores = sweep(client, arm_id)
        plateau = scores[EF_LADDER[-1]]

        # Terceira parcela: o que o rescoring recupera. Sai da diferença entre o
        # run reportado por M3.1 — que usa o rescoring declarado na matriz — e
        # este mesmo braço no mesmo `ef`, pontuando só pelos códigos.
        reported = evaluation.evaluate(retrieval.as_run(retrieval.load_run(arm_id, "hnsw")))
        reported_value = reported["aggregate"][METRIC]

        results[arm_id] = {
            "by_ef": scores,
            "plateau_ndcg10": plateau,
            "quantization_error": plateau - ceiling,
            "graph_error_at_preregistered_ef": scores[EF_LADDER[0]] - plateau,
            "reported_ndcg10": reported_value,
            "rescoring_recovery": reported_value - scores[EF_LADDER[0]],
        }

    converged = abs(results["A0"]["quantization_error"])
    print(f"\ncontrole: A0 no maior ef desvia {converged:+.4f} do A0 exato")
    if converged > 0.005:
        print("  A varredura NÃO chegou ao platô. Nenhuma decomposição abaixo é interpretável.")

    # O A4 já declara rescore=False na matriz, então o diagnóstico tem de
    # reproduzir o run reportado dele exatamente. Se não reproduzir, o override
    # não está chegando onde deveria e nenhuma linha acima vale.
    a4 = results["A4"]
    if abs(a4["rescoring_recovery"]) > 1e-9:
        raise RuntimeError(
            f"A4 declara rescore=False mas o diagnóstico difere do run reportado "
            f"em {a4['rescoring_recovery']:+.6f}: o override não está chegando à busca"
        )
    print("controle: A4 (rescore=False na matriz) reproduz o run reportado exatamente")

    print(
        f"\n{'braço':6} {'platô':>8} {'quantização':>13} {'grafo @128':>12} "
        f"{'rescoring':>11} {'reportado':>11}"
    )
    for arm_id, row in results.items():
        print(
            f"{arm_id:6} {row['plateau_ndcg10']:8.4f} {row['quantization_error']:+13.4f} "
            f"{row['graph_error_at_preregistered_ef']:+12.4f} "
            f"{row['rescoring_recovery']:+11.4f} {row['reported_ndcg10']:11.4f}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "issue": "RAF-73",
                "metric": METRIC,
                "reference": {"arm": "A0", "mode": "exact", METRIC: ceiling},
                "ef_ladder": list(EF_LADDER),
                "rescore": False,
                "_relaxed_held_constants": ["hnsw.ef_search", "search_overrides.rescore"],
                "_why": "O modo exact do Qdrant ignora quantização; sem afrouxar os dois, a decomposição é inmensurável.",
                "converged": converged <= 0.005,
                "arms": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n→ {OUTPUT}")
    return 0 if converged <= 0.005 else 1


if __name__ == "__main__":
    raise SystemExit(main())
