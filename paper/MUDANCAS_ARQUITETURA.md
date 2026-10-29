# Mudanças em relação ao documento de arquitetura

O documento de arquitetura entregue (versão 1.1, 10/09/2026) descreve o sistema
planejado. Este arquivo registra onde a implementação seguiu o plano, onde
divergiu e por quê, e como cada divergência foi tratada. Ele existe para que a
comparação entre o documento e o repositório possa ser feita linha a linha.

## Resumo

O repositório tem **duas trilhas experimentais** sobre o mesmo conjunto de dados
(Quati 1M, 50 consultas julgadas):

| | Trilha do documento | Trilha do servidor |
|---|---|---|
| Onde está | `quantptbr/exhaustive.py`, `scripts/14_exhaustive.py` | `quantptbr/index.py`, `quantptbr/retrieval.py`, scripts 05 a 12 |
| Busca | exaustiva, em NumPy | HNSW, no Qdrant 1.19 |
| Condições | F32, F16, I8, B1 | float32, int8, PQ 16x, PQ 32x, binária, binária + rescoring |
| Regras de quantização | as da seção 27.1 do documento | as do Qdrant |
| Métricas | nDCG@10 (ganho 2^rel − 1), Recall@100, MRR@100, QPS | nDCG@10 (ganho linear), Recall@100 |
| Consultas no denominador | 50 | 49 |

A trilha do documento implementa o protocolo exatamente como a seção 27
especifica. A trilha do servidor corresponde à "trilha separada de backend
otimizado" que a seção 27.1 já previa como extensão. As duas são comparáveis
porque a trilha do documento também reavalia cada ranking com as regras da
trilha do servidor (coluna "protocolo do estudo" em `paper/exhaustive_<modelo>.md`).

## Divergências e tratamento

| Item | Documento | Implementação | Situação |
|---|---|---|---|
| Modelo de embedding | `BAAI/bge-m3`, 1.024 dimensões | `intfloat/multilingual-e5-base`, 768 dimensões, nas duas trilhas | **Aberta.** Ver abaixo. |
| Mecanismo de busca | "FAISS ou mecanismo equivalente" | NumPy exaustivo (trilha do documento) e Qdrant (trilha do servidor) | Coberto pela seção 27.4, que deixa o backend como decisão do piloto. |
| Condição F16 | prevista | implementada na trilha do documento | Fechada. |
| Quantização por produto | "extensão, não necessária" (seção 7.1) | dois braços na trilha do servidor | Extensão, como previsto. |
| Reordenação com F32 | fora das quatro condições principais (seção 27.2) | ausente na trilha do documento; presente na trilha do servidor | Fechada: as condições principais não usam reordenação. |
| Regra do I8 | `s_j = max_i \|x_ij\| / 127` por dimensão | igual na trilha do documento; quantil 0,99 do Qdrant na trilha do servidor | Fechada. |
| MRR@100 e QPS | previstos | implementados na trilha do documento | Fechada. |
| Ganho do nDCG | `2^rel − 1` | `2^rel − 1` na trilha do documento; linear na trilha do servidor | Fechada; as duas réguas são reportadas. |
| Relevante para Recall e MRR | grau > 0 | grau > 0 na trilha do documento; grau ≥ 2 na trilha do servidor | Fechada. |
| Consultas sem relevante | sinalizadas, sem exclusão | idem na trilha do documento; consulta 2 excluída na trilha do servidor | Fechada. |
| Latência | 3 aquecimentos + 10 rodadas, ordem embaralhada | idem na trilha do documento | Fechada. |
| Gráficos | preto e branco | preto e branco (`scripts/13_figures.py`) | Fechada. |
| Estrutura de pastas | `src/`, scripts 01 a 05 | `quantptbr/`, scripts 01 a 16 | Diferença de organização, sem efeito nos resultados. |

### Modelo de embedding

O documento fixa o BGE-M3. A implementação usa o E5-base por um motivo
verificável: é o único modelo denso com nDCG@10 publicado sobre o Quati 1M
(0,3955, Tabela 6 de Bueno et al.). Isso permitiu validar o aparelho de medição
antes de medir quantização: a linha de base reproduz o número publicado com
diferença de −0,0131, dentro da tolerância de 0,02 declarada antes da medição.
Com o BGE-M3 não haveria número publicado contra o qual conferir.

Para fechar a divergência, a trilha do documento roda também sobre vetores do
BGE-M3, gerados por `scripts/15_embed_bge_m3.py` na revisão fixada do modelo
(pooling CLS, sem prefixo, 1.024 dimensões). O envelope fica no próprio script
e no manifesto da saída, e não em `config/embedding_envelope.json`, que é o
envelope congelado da trilha do servidor. A comparação BGE-M3 × E5 sai de
`scripts/16_compare_tracks.py`, com o mesmo protocolo para os dois modelos.

## Proveniência

- As medições da trilha do servidor (`runs/`, agosto de 2026) foram executadas
  numa máquina Linux com GPU de 11 GB. O arquivo de vetores usado nelas não foi
  preservado; só o hash dele está registrado em `runs/s1.5_vectors.json`.
- Os vetores usados pela trilha do documento foram regenerados pelo grupo, com o
  mesmo modelo, revisão e parâmetros (`config/embedding_envelope.json`), numa
  RTX 4050 de 6 GB. Placa e sistema diferentes mudam as últimas casas decimais,
  então o arquivo novo não é idêntico byte a byte ao original. O registro dele
  fica em `runs/s1.5_vectors_regenerado.json`, sem sobrescrever o original.
  O efeito é pequeno e está medido: o top-10 e o top-100 do F32 exaustivo
  coincidem com os da busca exata do A0 original em 99,6% e 99,7%.
- Os vetores das 50 consultas foram regenerados do mesmo modo. O registro novo
  fica em `runs/s1.6_queries_regenerado.json`; `runs/s1.6_queries.json` segue
  sendo o da trilha do servidor.
- A latência da trilha do documento depende do estado da máquina, não só do
  código. A primeira execução da Tabela 3 do E5 (29/10) rodou com a RAM ocupada
  por outros processos, e o F32 mediu 525,8 ms por consulta, contra 148,8 ms
  numa execução de 27/10 com a máquina livre. A qualidade é idêntica nas duas,
  porque o ranking não depende do tempo. Por isso a latência do E5 é medida de
  novo nas mesmas condições da do BGE-M3, e só a comparação entre execuções
  feitas em sequência na mesma máquina é válida.
- Na trilha do documento, F16 e I8 são reconstruídos em float32 bloco a bloco
  antes do produto interno, como a seção 27.1 descreve. A latência deles mede
  essa reconstrução em NumPy, não aceleração nativa de meia precisão ou de
  inteiros, e por isso sai maior que a do F32.

## O que continua fora do escopo

- Controle em inglês. Sem ele, o estudo mede quantização num benchmark nativo
  em português, mas não responde se o resultado difere do inglês.
- Avaliação da resposta de um LLM (fora de escopo desde o documento).
