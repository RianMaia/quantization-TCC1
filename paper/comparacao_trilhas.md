# Comparação entre as trilhas

Gerado por `scripts/16_compare_tracks.py` a partir de `runs/`; nenhum número
foi digitado.

## Busca exaustiva × Qdrant, com o mesmo modelo (E5)

| Representação | Exaustiva | Δ vs F32 | Qdrant, só quantização | Δ vs A0 exato | Qdrant, como medido | Δ vs A0 exato |
|---|---|---|---|---|---|---|
| float32 | 0.4688 | +0.0000 | 0.4669 | +0.0000 | 0.4655 | -0.0014 |
| int8 | 0.4679 | -0.0010 | 0.4591 | -0.0078 | 0.4604 | -0.0065 |
| binária | 0.1148 | -0.3540 | 0.1139 | -0.3530 | 0.0773 | -0.3897 |
| float16 | 0.4653 | -0.0035 | — | — | — | — |

> Régua do estudo: ganho linear, 49 consultas. A busca exata do A0 no Qdrant
> dá 0.4669. "Só quantização" é o platô sem
> reordenação, com o grafo já convergido; "como medido" é a Tabela T1 de
> `paper/tables.md` (ef 128, reordenação padrão). A trilha exaustiva usa os
> vetores regenerados; o top-10 e o top-100 do F32 coincidem com os da busca
> exata do A0 original em 0.996 e 0.997.
> As regras de quantização diferem: o I8 exaustivo usa o máximo por dimensão
> (seção 27.1 do documento); o int8 do Qdrant, o quantil 0,99.

## BGE-M3 × E5, com o mesmo protocolo

Pendente: falta `runs/exhaustive_bge-m3.json`.
