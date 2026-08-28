# Compressão de índices vetoriais em português brasileiro

Até onde dá para comprimir os vetores de um índice de busca semântica em pt-BR
antes que a busca piore de forma mensurável, e quanto cada unidade de memória
economizada custa em qualidade.

Artigo de medição. Nenhum método, arquitetura ou modelo novo é proposto — os
números são a entrega. O planejamento vive no Linear sob *Vector Index
Compression in Brazilian Portuguese*; os documentos **Mission**, **Tech Stack**,
**Measurement Contract** e **Pre-registration (S1.2)** são normativos.

## Para quem vai escrever o artigo

Comece por **[`paper/HANDOFF.md`](paper/HANDOFF.md)**: o que foi medido, com que
força cada afirmação pode ser feita, e onde estão os limites.

```bash
python scripts/11_regenerate.py     # regenera toda tabela; offline, sem GPU
```

Escreve `paper/tables.md` para colar e `paper/numbers.json` para conferir.
**Nenhum número do artigo deve ser digitado à mão.** O script aborta com código
não-zero se faltar evidência, em vez de emitir tabela com lacunas.

### O que os números dizem

- **O aparelho foi validado** antes de qualquer conclusão: a linha de base
  reproduz o único número de E5-base publicado sobre o Quati dentro de 0,0131.
- **A quantização escalar int8 é um nulo informativo** — sem degradação
  detectável neste tamanho de amostra, o que não é o mesmo que equivalência.
- **A quantização binária colapsa** (nDCG@10 de 0,4655 para 0,0773), e a
  geometria dos vetores explica por quê: 323 das 768 dimensões são degeneradas
  em sinal, restando 42,3% do orçamento de bits.
- **Comprimir aumentou a memória a provisionar em todos os braços**, contra o
  que o estudo pré-registrou. Com `always_ram: true` os códigos vão para memória
  anônima não-evictável enquanto os originais seguem mapeados; a linha de base
  float32, que não tem códigos, acaba com o menor piso dos seis.
- **A fronteira de Pareto de qualidade × memória tem um ponto só: o float32.**
  No par qualidade × latência ela tem quatro, e é aí que comprimir paga.

## Layout

| Diretório | Papel |
| -- | -- |
| `config/` | Decisões congeladas, legíveis por máquina, consumidas pelos scripts |
| `data/raw/` | Entradas congeladas, buscadas por revisão fixa. Só o `MANIFEST.json` vai ao git |
| `data/vectors/` | Artefatos gerados (3 GB). Fora do git; identificados por hash em `runs/` |
| `runs/` | Manifestos e evidências por execução. **Versionados** — é o que torna os artefatos verificáveis |
| `quantptbr/` | Os módulos: contrato do corpus, envelope, vetores, servidor, memória, recuperação, avaliação, estatística |
| `scripts/` | Etapas numeradas do pipeline |
| `tests/` | Contratos executáveis. `-m "not gpu"` pula o que exige GPU |

Nenhum artefato grande é rastreado pelo git, e nenhum artefato de que a análise
dependa fica ao mesmo tempo fora do git e sem hash.

## Reproduzir

Requer Python 3.12+, [uv](https://docs.astral.sh/uv/), podman e uma GPU NVIDIA
com 11 GB. Sem rede depois do passo 1.

```bash
uv sync

python scripts/01_fetch_quati.py        # corpus, tópicos e qrels na revisão fixa
python scripts/02_pilot_embeddings.py   # S1.4 — congela o envelope de embedding
python scripts/03_generate_embeddings.py  # S1.5 — 1M passagens, ~40 min de GPU
python scripts/04_embed_queries.py      # S1.6 — as 50 consultas julgadas

python scripts/qdrant_server.py start   # servidor fixado por digest
python scripts/memory_smoke_test.py     # S1.1 — prova que os contadores respondem

python scripts/05_create_collections.py A0   # M2.1 — braço de referência float32

python scripts/06_evaluate_benchmark.py     # M3.1 — qualidade e o portão de calibração
python scripts/07_analyse.py                # M4.1 — bootstrap pareado
python scripts/08_decompose.py              # M4.3 — quantização contra grafo, varrendo ef
python scripts/09_vector_geometry.py        # M2.8 — degenerescência de sinal por dimensão
python scripts/10_measure_arms.py           # M3.2/3.3/3.5 — memória, latência e disco
python scripts/12_oversampling.py           # M3.4 — escada de oversampling do A5
python scripts/11_regenerate.py             # M4.4 — todas as tabelas
```

As medições longas devem rodar sob `systemd-inhibit --what=sleep:idle`: se a
máquina suspender no meio, `time.monotonic()` congela junto e os prazos internos
param de correr.

O braço é argumento, não código: `05_create_collections.py A1 A2 A3 A4` constrói
os demais sem editar nada. O A5 não é construível — ele compartilha a coleção do
A4 e difere apenas em parâmetros de busca.

O passe de 1M é retomável: rodar de novo continua do último bloco completo e
produz resultado idêntico ao de uma execução ininterrupta.

## O que impede um resultado errado de parecer certo

Cada um destes existe porque a falha correspondente é **silenciosa** — produziria
uma tabela impecável que não mede nada.

- **Modo local do Qdrant aceita e ignora quantização.** Todo script de medição
  chama `server.require_server`, e todo build quantizado chama
  `server.require_quantization`, que relê a configuração do servidor.
- **Subamostrar o corpus destrói o gabarito.** `01_fetch_quati.py` falha se um
  único ID dos qrels não estiver no corpus.
- **O envelope de embedding pode divergir entre etapas.** Ele é lido de
  `config/embedding_envelope.json`, nunca redigitado, e um teste amarra a decisão
  à evidência que a produziu.
- **A ordem do corpus define os IDs dos pontos.** Verificada por recodificação de
  amostra, comparando argmax e não limiar.
- **Execuções incomparáveis podem ser somadas.** `manifest.require_same_envelope`
  aborta; nunca avisa e continua.
- **RSS não é memória residente.** O contrato está em `quantptbr/memory.py`.
- **O Qdrant indexa de forma assíncrona.** Uma coleção medida durante a
  construção do grafo não dá nem o custo de build nem o regime estacionário.
  `index.wait_until_indexed` exige verde **e** a contagem de vetores indexados.
- **Contagem de vetores indexados acima do total é otimizador em curso, não
  build pronto.** O Qdrant soma segmentos sobrepostos durante uma
  re-otimização; medido, um braço reportou 2.300.544 indexados para 1M de
  pontos. Limitar a contagem ao total não resolve — esconde a evidência em vez
  de detectá-la.
- **Escolher o ponto de operação depois de ver os dados.** A varredura de
  oversampling mostra que 16x supera o 4x pré-registrado, e o reportado continua
  sendo o pré-registrado.

## Testes

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -m "not gpu"   # offline, sem rede
uv run pytest -m gpu         # inclui o teste de retomada do passe
```
