# Tabelas do artigo — regeneradas, não digitadas

Gerado por `scripts/11_regenerate.py`. Todo número vem de um arquivo em `runs/`.
Nenhum valor aqui foi transcrito à mão; um campo `—` significa evidência ausente,
não estimativa.

## T1 — Qualidade por braço (busca HNSW, 49 consultas pontuáveis)

| Braço | Configuração | nDCG@10 | R(rel≥2)@100 | Δ vs A0 (IC 95%) | Veredito |
|---|---|---|---|---|---|
| A0 | float32 baseline | 0.4655 | 0.6726 | — | linha de base |
| A1 | scalar int8 | 0.4604 | 0.6631 | -0.0051 [-0.0217, +0.0082] | não distinguível de ruído neste tamanho de amostra |
| A2 | product quantization 16x | 0.4411 | 0.6197 | -0.0244 [-0.0533, -0.0012] | degrada |
| A3 | product quantization 32x | 0.3928 | 0.4880 | -0.0727 [-0.1165, -0.0332] | degrada |
| A4 | binary | 0.0773 | 0.1194 | -0.3883 [-0.4491, -0.3291] | degrada |
| A5 | binary + oversampling and rescoring | 0.3770 | 0.4109 | -0.0885 [-0.1280, -0.0525] | degrada |

> Um intervalo que contém zero **não** é equivalência: é ausência de evidência de
> diferença neste tamanho de amostra. A diferença mínima detectável por braço está
> em `numbers.json`.

## T2 — Compressão: armazenamento contra residência

A distinção central do artigo, e o ponto em que o medido contraria o esperado.

O Qdrant guarda os vetores float32 originais ao lado dos códigos quantizados, de
modo que comprimir **aumenta os bytes a armazenar** — isso era esperado. O que
não era: comprimir **também aumenta a memória a provisionar**. Com
`always_ram: true`, os códigos vão para memória anônima enquanto os originais
seguem mapeados em disco; a linha de base float32 mantém tudo como page cache
evictável, e por isso tem o menor piso de todos. A razão nominal é a que a
documentação e os posts de prática citam; as outras duas são as medidas.

| Braço | Configuração | Compressão nominal | Disco (MB) | Disco vs A0 | Piso de provisionamento (MB) | Piso vs A0 |
|---|---|---|---|---|---|---|
| A0 | float32 baseline | 1x | 3,928.3 | 1.00x | 410.6 | 1.00x |
| A1 | scalar int8 | 4x | 4,881.0 | 1.24x | 1,276.1 | 3.11x |
| A2 | product quantization 16x | 16x | 4,739.8 | 1.21x | 632.8 | 1.54x |
| A3 | product quantization 32x | 32x | 3,950.3 | 1.01x | 537.6 | 1.31x |
| A4 | binary | 32x | 3,714.0 | 0.95x | 554.1 | 1.35x |
| A5 | binary + oversampling and rescoring | 32x | 3,714.0 | 0.95x | 561.4 | 1.37x |

> **Ler as duas últimas colunas na direção certa: acima de `1.00x` é gastar
> MAIS que o float32, não menos.** Nenhum braço comprimido reduz o piso de
> provisionamento — todos o aumentam. Em disco, só a quantização binária fica
> abaixo da linha de base, e por 5%, longe dos 32x nominais.
>
> Piso de provisionamento = memória anônima do cgroup do container no estado
> `warm_hnsw`: é o que precisa existir na máquina. O page cache dos
> originais é evictável e aparece em `numbers.json` como `page_cache_mb`.

## T3 — Latência por consulta (ms)

| Braço | Configuração | Mediana | p95 | Amostras |
|---|---|---|---|---|
| A0 | float32 baseline | 16.6 | 19.1 | 250 |
| A1 | scalar int8 | 10.6 | 12.1 | 250 |
| A2 | product quantization 16x | 66.3 | 68.7 | 250 |
| A3 | product quantization 32x | 11.5 | 13.1 | 250 |
| A4 | binary | 4.5 | 6.4 | 250 |
| A5 | binary + oversampling and rescoring | 8.1 | 10.0 | 250 |

> Cache quente, consultas repetidas, concorrência 1. Não é um perfil de
> cold start, e os valores são específicos deste hardware.

## T4 — Decomposição do erro (nDCG@10)

| Braço | Quantização | Grafo (ef pré-registrado) | Rescoring | Reportado |
|---|---|---|---|---|
| A0 | +0.0000 | -0.0014 | +0.0000 | 0.4655 |
| A1 | -0.0078 | -0.0077 | +0.0090 | 0.4604 |
| A2 | -0.0427 | -0.0295 | +0.0463 | 0.4411 |
| A3 | -0.2010 | -0.0388 | +0.1657 | 0.3928 |
| A4 | -0.3530 | -0.0367 | +0.0000 | 0.0773 |

> Obtida varrendo `ef` até o platô, com rescoring desligado. Isola quanto da
> perda vem dos códigos e quanto vem da busca aproximada.

## T5 — Geometria dos vetores e o teto do braço binário

- Média das médias por dimensão: **-0.00018** — o espaço *parece* centrado quando olhado no agregado.
- Deslocamento por dimensão, |média|/desvio: **1.614** — olhado dimensão a dimensão, não está.
- Dimensões degeneradas em sinal (limiar 0.95, declarado antes de calcular): **323 de 768**.
- Orçamento efetivo do código binário: **324.7 bits de 768** (42.3%).

## T7 — Fronteiras de Pareto e o ponto de operação

Um braço é **dominado** quando existe outro que não é pior em nenhum dos dois
eixos e é melhor em pelo menos um. É a pergunta que se faz antes de escolher
uma configuração, e tem resposta mecânica.

### Qualidade × memória a provisionar

| Braço | nDCG@10 | Piso de provisionamento (MB) | Situação |
|---|---|---|---|
| A0 | 0.4655 | 410.6 | **na fronteira** |
| A1 | 0.4604 | 1,276.1 | dominado por A0 |
| A2 | 0.4411 | 632.8 | dominado por A0 |
| A3 | 0.3928 | 537.6 | dominado por A0 |
| A4 | 0.0773 | 554.1 | dominado por A0, A3 |
| A5 | 0.3770 | 561.4 | dominado por A0, A3 |

> **A fronteira tem um ponto só: A0.** Nenhuma
> configuração comprimida é Pareto-ótima neste par de eixos — a linha de
> base tem ao mesmo tempo a melhor qualidade e o menor piso.

### Qualidade × latência

| Braço | nDCG@10 | Latência mediana (ms) | Situação |
|---|---|---|---|
| A0 | 0.4655 | 16.6 | **na fronteira** |
| A1 | 0.4604 | 10.6 | **na fronteira** |
| A2 | 0.4411 | 66.3 | dominado por A0, A1 |
| A3 | 0.3928 | 11.5 | dominado por A1 |
| A4 | 0.0773 | 4.5 | **na fronteira** |
| A5 | 0.3770 | 8.1 | **na fronteira** |

> Aqui a compressão paga: a fronteira tem 4 pontos (A0, A1, A4, A5).
> É o eixo em que trocar qualidade por velocidade tem sentido.

> **A recomendação é condicional, e o texto não deve simplificá-la.** Se a
> restrição é memória a provisionar, não comprima: nesta configuração a
> compressão custa RAM não-evictável em vez de economizá-la. Se a restrição é
> latência, a compressão compra tempo, e aí o par binário + rescoring é o que
> recupera qualidade sem devolver a velocidade toda.

## T8 — Varredura de oversampling (A5, sobre a coleção do A4)

Afrouxa um held-constant sob rótulo: `oversampling` é varrido, o resto segue a
matriz. Serve para dizer se o ponto de operação do A5 é bom, e não apenas qual
ele é.

| Oversampling | nDCG@10 | Latência mediana (ms) |
|---|---|---|
| 1x | 0.2747 | 5.4 |
| 2x | 0.3388 | 6.6 |
| 4x ← | 0.3648 | 8.8 |
| 8x | 0.4013 | 13.1 |
| 16x | 0.4187 | 20.6 |

> **A escada não é plana: amplitude de +0.1439 em nDCG@10.** O ponto pré-registrado (4x) não é o melhor da escada — a 16x o braço binário chega a 0.4187.
>
> **Consequência para o texto:** o A5 reportado nas outras tabelas usa o valor
> pré-registrado, e é esse que deve ser citado como resultado. Esta varredura é
> diagnóstico — mostra que o parâmetro tem efeito forte e que o ponto de
> operação escolhido antes da medição era conservador. Trocar o valor
> reportado por causa do resultado seria escolher o ponto depois de ver os
> dados.

## T6 — Validação do aparelho

Reproduz o protocolo dos autores do Quati — run sobre o corpus de 1M pontuado
contra os qrels de 10M — para confrontar a linha de base com o único número de
E5-base publicado sobre este benchmark. É o que autoriza ler as outras tabelas.

- Alvo publicado (Bueno et al., Quati, Tabela 6, E5-base): **0.3955**
- Medido aqui, mesmo protocolo: **0.3824** (Δ -0.0131)
- Veredito: **aparelho validado**
- Checagem de faixa: BM25 0.3991 < A0 0.4669 < ColBERT-X mMARCO pt-BR 0.4927 — dentro: **True**

## Procedência

- Corpus `sha256:096b262a3444df34…`, qrels `sha256:aafab4294572a043…`
- Modelo `intfloat/multilingual-e5-base` @ `d12875059715`, 768d
- Qdrant 1.19.0, imagem `sha256:057ee3a8da769fe7…`
- HNSW m=16, ef_construct=100, ef_search=128, Cosine
- Consultas julgadas: 50 (pontuáveis: 49)
- Kernel `7.0.11-200.nobara.fc43.x86_64`, Python 3.14.4
- Regenerado em 2026-08-27T14:40:01.985624+00:00 no commit `fb40243aac69`
