# Análise por consulta — gerada, não digitada

Gerado por `scripts/13_figures.py` a partir de `runs/eval_*.json` e
`runs/retrieval/*.jsonl`. Busca HNSW; referência é o A0.

Vitórias, empates e derrotas contam consultas em nDCG@10 (49
pontuáveis). A sobreposição é a fração média do top-k do A0 que o braço
também devolve; não usa o gabarito e cobre todas as consultas recuperadas.

| Braço | Configuração | Vitórias | Empates | Derrotas | Sobreposição@10 com A0 | Sobreposição@100 com A0 |
|---|---|---|---|---|---|---|
| A1 | scalar int8 | 4 | 38 | 7 | 0.952 | 0.946 |
| A2 | product quantization 16x | 3 | 36 | 10 | 0.938 | 0.912 |
| A3 | product quantization 32x | 11 | 16 | 22 | 0.746 | 0.705 |
| A4 | binary | 1 | 3 | 45 | 0.100 | 0.102 |
| A5 | binary + oversampling and rescoring | 11 | 9 | 29 | 0.664 | 0.541 |
| A0 exato | float32, busca exaustiva (controle) | — | — | — | 0.960 | 0.932 |

> A linha de controle compara o A0 sob HNSW com o A0 sob busca exata: é
> quanto o grafo sozinho mexe no ranking, sem quantização nenhuma. Um braço
> com sobreposição próxima desse valor perturba o top-k tanto quanto a busca
> aproximada já perturbava.
