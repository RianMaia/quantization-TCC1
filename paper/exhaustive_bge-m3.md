# Trilha exaustiva — bge-m3

Gerado por `scripts/14_exhaustive.py`; nenhum número foi digitado. Busca
exaustiva sobre as quatro condições do documento de arquitetura (seção 27).

| Configuração | nDCG@10 | Δ vs F32 (IC 95%) | Recall@100 | MRR@100 | Top-10 igual ao F32 | Memória (GB) | Redução | Latência p50 (ms) | p95 (ms) | QPS | nDCG@10, protocolo do estudo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F32 | 0.5065 | — | 0.7267 | 0.8177 | 1.000 | 4.096 | 1.0x | 897.6 | 1150.1 | 1.1 | 0.5431 |
| F16 | 0.5079 | +0.0014 [-0.0001, +0.0043] | 0.7267 | 0.8177 | 0.996 | 2.048 | 2.0x | 8693.8 | 9825.8 | 0.1 | 0.5445 |
| I8 | 0.5082 | +0.0016 [-0.0002, +0.0048] | 0.7254 | 0.8177 | 0.988 | 1.024 | 4.0x | 6977.4 | 11686.7 | 0.1 | 0.5447 |
| B1 | 0.4377 | -0.0688 [-0.1049, -0.0334] | 0.6295 | 0.7667 | 0.572 | 0.128 | 32.0x | 292.8 | 569.0 | 3.0 | 0.4680 |

> Qualidade sobre 50 consultas, com ganho 2^rel − 1. Consultas sem passagem
> relevante, mantidas no denominador com zero: 2.
> A última coluna reavalia o mesmo ranking com as regras de `config/arms.json`
> (ganho linear, 49 consultas) e é a única comparável com `paper/tables.md`.
>
> Latência por chamada, 3 rodadas de aquecimento descartadas e
> 10 medidas, ordem embaralhada pela mesma semente nas quatro condições.
> QPS é vazão sequencial, não capacidade sob concorrência. Memória é o tamanho
> da representação em RAM, em GB decimais; F16 e I8 são reconstruídos em float32
> bloco a bloco, então a latência deles não demonstra aceleração nativa.
>
> Sobreposição do top-10 e do top-100 do F32 com a busca exata do A0 da trilha
> do servidor (`runs/retrieval/A0_exact.jsonl`):
> 0.442 e 0.385.
