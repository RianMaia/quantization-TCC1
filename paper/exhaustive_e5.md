# Trilha exaustiva — e5

Gerado por `scripts/14_exhaustive.py`; nenhum número foi digitado. Busca
exaustiva sobre as quatro condições do documento de arquitetura (seção 27).

| Configuração | nDCG@10 | Δ vs F32 (IC 95%) | Recall@100 | MRR@100 | Top-10 igual ao F32 | Memória (GB) | Redução | Latência p50 (ms) | p95 (ms) | QPS | nDCG@10, protocolo do estudo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F32 | 0.4356 | — | 0.6514 | 0.8253 | 1.000 | 3.072 | 1.0x | 525.8 | 1260.6 | 1.5 | 0.4688 |
| F16 | 0.4325 | -0.0030 [-0.0073, +0.0004] | 0.6535 | 0.8153 | 0.992 | 1.536 | 2.0x | 6644.1 | 6925.9 | 0.2 | 0.4653 |
| I8 | 0.4337 | -0.0019 [-0.0071, +0.0034] | 0.6525 | 0.8253 | 0.980 | 0.768 | 4.0x | 4692.3 | 4975.0 | 0.2 | 0.4679 |
| B1 | 0.1089 | -0.3267 [-0.3846, -0.2694] | 0.1469 | 0.3186 | 0.124 | 0.096 | 32.0x | 215.2 | 424.8 | 4.3 | 0.1148 |

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
> 0.996 e 0.997.
