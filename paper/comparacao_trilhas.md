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

| Condição | nDCG@10 E5 | % do F32 (E5) | Δ vs F32 E5 (IC 95%) | nDCG@10 BGE-M3 | % do F32 (BGE-M3) | Δ vs F32 BGE-M3 (IC 95%) |
|---|---|---|---|---|---|---|
| F32 | 0.4356 | 100.0% | — | 0.5065 | 100.0% | — |
| F16 | 0.4325 | 99.3% | -0.0030 [-0.0073, +0.0004] | 0.5079 | 100.3% | +0.0014 [-0.0001, +0.0043] |
| I8 | 0.4337 | 99.6% | -0.0019 [-0.0071, +0.0034] | 0.5082 | 100.3% | +0.0016 [-0.0002, +0.0048] |
| B1 | 0.1089 | 25.0% | -0.3267 [-0.3846, -0.2694] | 0.4377 | 86.4% | -0.0688 [-0.1049, -0.0334] |

> Régua do documento de arquitetura: ganho 2^rel − 1, 50 consultas, busca
> exaustiva. "% do F32" é o nDCG@10 da condição dividido pelo do F32 do
> mesmo modelo.
