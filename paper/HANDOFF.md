# Repasse — o que foi medido, e o que pode ser afirmado

Este documento existe para que quem escrever o artigo não precise ler o código
nem reconstruir o raciocínio. Ele diz o que o estudo mediu, com que força cada
afirmação pode ser feita, e onde os limites estão.

**Divisão de trabalho.** Este repositório entrega código, medições e evidência.
A redação, a revisão bibliográfica e a submissão são de outra pessoa. As issues
de M5 e M6.3 no Linear foram canceladas por essa razão, não por terem sido
descartadas.

---

## 1. Um comando regenera tudo

```bash
python scripts/11_regenerate.py
```

Escreve `paper/tables.md` (para colar) e `paper/numbers.json` (para conferir).
**Nenhum número do artigo deve ser digitado à mão.** Se um valor não aparece
nessa saída, ele não foi medido — e não deve entrar no texto.

Roda offline, sem GPU e sem servidor: lê apenas os arquivos de evidência em
`runs/`. Reproduzir as medições do zero é outra história — ver §7.

**Aborta com código 1 se faltar evidência**, em vez de emitir tabela com
lacunas: um `—` copiado para o LaTeX vira lacuna publicada. Verificado de
propósito removendo a medição de um braço. Existe `--allow-partial` para
inspecionar estado intermediário durante a medição; **nenhuma tabela reportada
deve sair desse modo**.

Rodar duas vezes produz números byte-idênticos — só a linha de carimbo
(timestamp) difere, e ela é metadado de procedência, não resultado.

---

## 2. O que o estudo mediu

Seis configurações de quantização vetorial sobre **Quati 1M** (1.000.000 de
passagens em português, 50 consultas julgadas das quais 49 são pontuáveis),
indexadas no **Qdrant 1.19.0**, com embeddings do
**`intfloat/multilingual-e5-base`** (768 dimensões).

| Braço | Configuração |
|---|---|
| A0 | float32, sem compressão — linha de base |
| A1 | quantização escalar int8 |
| A2 | product quantization 16x |
| A3 | product quantization 32x |
| A4 | quantização binária |
| A5 | quantização binária + oversampling e rescoring |

Para cada braço: qualidade (nDCG@10 e Recall@100), memória residente, latência
por consulta e ocupação em disco — todos sob topologia HNSW idêntica, para que
a única diferença entre braços seja a que a matriz declara.

> ⚠️ **Discrepância a corrigir na escrita.** Os documentos de planejamento no
> Linear (RAF-77) mencionam **BGE-M3** como modelo de embedding. **O estudo
> usou `intfloat/multilingual-e5-base`**, revisão `d128750597153bb5`. O valor
> medido é o que vale; os documentos de planejamento estão desatualizados. A
> escolha também é o que torna a calibração da §3 possível, já que E5-base é a
> única linha do Quati com número publicado que conseguimos reproduzir.

---

## 3. O aparelho foi validado antes de qualquer conclusão

Um estudo de degradação que não sabe se sua linha de base está certa não mede
degradação — mede o próprio erro. Por isso o A0 foi confrontado com o único
número de E5-base publicado sobre o Quati:

| | valor |
|---|---|
| Publicado (Bueno et al., Tabela 6, seção 1M, linha E5-base) | 0,3955 |
| Medido aqui, reproduzindo o protocolo | **0,3824** |
| Diferença | −0,0131 (dentro da tolerância pré-declarada de 0,02) |

E a checagem de faixa contra os sistemas da Tabela 7: BM25 0,3991 < **A0
0,4669** < ColBERT-X 0,4927. A linha de base cai onde deveria cair.

**Detalhe de protocolo que o texto precisa preservar:** o 0,3955 vem de um run
sobre o corpus de 1M pontuado contra os qrels de **10M**. Compará-lo contra o
nDCG do estudo sobre os qrels de 1M compararia grandezas diferentes — os pools
têm 97,78 e 38,66 julgamentos por consulta, e o mesmo run pontua mais baixo no
primeiro. A calibração reproduz o protocolo deles; o estudo usa o dele.

---

## 4. A contribuição conceitual central

> **Nesta configuração, comprimir um índice vetorial não reduziu nem os bytes
> armazenados nem a memória a provisionar. Aumentou as duas coisas.**

⚠️ **Isto contraria o que o próprio estudo pré-registrou.** A expectativa era a
formulação usual: compressão reduz residência e aumenta armazenamento, porque o
Qdrant mantém os vetores float32 originais ao lado dos códigos quantizados. A
primeira metade se confirmou. A segunda não.

**O mecanismo, que é o que torna o achado transferível.** Com
`always_ram: true`, os códigos quantizados vão para memória **anônima** — que
não pode ser despejada e precisa existir na máquina. Os originais float32
continuam mapeados em disco, como page cache **evictável**. A linha de base
float32 não tem códigos, então mantém tudo como page cache e acaba com o
**menor piso de provisionamento de todos os seis braços** (410,6 MB). Cada
braço comprimido soma um custo anônimo em cima de um custo mapeado que não
desapareceu.

Os números de 4x e 32x citados na documentação e em posts de prática descrevem
o tamanho do *código*, não o de nada que se possa provisionar ou faturar. A
tabela T2 põe as três razões lado a lado — nominal, disco e piso — para tornar
a distância visível: o A1 promete 4x e custa **3,11x mais** RAM não-evictável
que não comprimir nada.

**Como escrever isto sem exagerar.** É um resultado sobre *esta* configuração:
um motor (Qdrant 1.19), com `always_ram: true`, com os originais retidos.
Não é uma afirmação sobre quantização em geral, e um motor que descartasse os
originais teria outro perfil. O que é geral é a distinção — residência,
armazenamento e tamanho de código são três grandezas diferentes, e a prática
corrente cita uma como se fosse as três.

Isto custa quase nada em espaço de página e é o que um leitor leva embora mesmo
que esqueça todo o resto. Não é uma ressalva para a seção de limitações — é
para a introdução, a tabela de resultados e a conclusão.

---

## 5. Resultados e a força de cada afirmação

Números exatos em `paper/tables.md`. O que importa aqui é **qual verbo cada um
autoriza**.

**Calibração de verbo — regra mecânica, sem julgamento:**

| Situação | Verbo permitido |
|---|---|
| Intervalo de confiança inteiramente abaixo de zero | "degrada" |
| Intervalo contendo zero | "não apresenta degradação detectável neste tamanho de amostra" |

Um intervalo que contém zero **nunca** deve ser escrito como equivalência.
Ausência de evidência de diferença não é evidência de ausência de diferença, e
a diferença mínima detectável por braço está em `numbers.json` para quem quiser
declarar o poder do teste.

**O que os dados sustentam:**

- **A1 (int8) é um nulo informativo.** Diferença de −0,0051, intervalo
  [−0,0217, +0,0082], diferença mínima detectável 0,0218. A frase correta é que
  não há degradação detectável neste tamanho de amostra — não que os dois sejam
  equivalentes.
- **A2, A3, A4 e A5 degradam**, com intervalos inteiramente abaixo de zero.
- **A4 (binária) colapsa**: nDCG@10 de 0,0773 contra 0,4655 da linha de base. E
  o rescoring do A5 recupera boa parte disso (0,3770), o que faz do par A4/A5 o
  argumento mais forte do artigo a favor de oversampling com rescoring.

**A decomposição do erro (T4)** separa quanto da perda vem dos códigos e quanto
vem da busca aproximada, varrendo `ef` até o platô. O achado que vale destacar:
o HNSW sobre float32 custa apenas −0,0014, então a degradação medida é da
quantização, não do grafo.

**A fronteira de Pareto (T7) dá o resultado mais direto do artigo.** No par
qualidade × memória a provisionar, **a fronteira tem um ponto só: o A0**. Toda
configuração comprimida é dominada — existe uma alternativa que é melhor nos
dois eixos ao mesmo tempo. No par qualidade × latência a história muda: a
fronteira tem quatro pontos (A0, A1, A4, A5), e é aí que comprimir faz sentido.

**A recomendação de ponto de operação é condicional, e simplificá-la seria
desonesto:** se a restrição é memória a provisionar, não comprima; se é
latência, comprima e use rescoring para recuperar qualidade.

**A varredura de oversampling (T8) mostra que o ponto de operação
pré-registrado era conservador.** A escada não é plana: amplitude de +0,1439 em
nDCG@10 entre 1x e 16x. Em 4x — o valor pré-registrado — o A5 dá 0,3648; em 16x
chega a **0,4187**, a 20,6 ms.

⚠️ **Como escrever isso sem cometer o erro que o estudo evita.** O A5 reportado
em todas as outras tabelas usa o valor pré-registrado, e é esse que deve ser
citado como resultado. A varredura é diagnóstico. **Trocar o número reportado
por causa dela seria escolher o ponto de operação depois de ver os dados**, que
é exatamente a liberdade que o pré-registro existe para remover. A forma
honesta: reportar 4x como resultado e citar a varredura como evidência de que o
parâmetro tem efeito forte e merece ser sintonizado em produção.

**A geometria (T5) explica o colapso binário em vez de apenas relatá-lo.** O
espaço parece centrado no agregado (média das médias por dimensão −0,00018),
mas dimensão a dimensão o deslocamento é de 1,614 desvios: 323 das 768
dimensões são degeneradas em sinal, restando **324,7 bits efetivos de 768
(42,3%)**. O limiar de degenerescência foi declarado antes de calcular.

---

## 6. Limitações que precisam aparecer no texto

Estas não são opcionais e não devem ser cortadas para caber no limite de
páginas. Se faltar espaço, corte uma figura.

1. **É um artigo de medição, não de método.** O mérito está no rigor e na
   utilidade, não em novidade conceitual. O texto precisa dizer isso, não
   apenas o autor precisa saber disso.
2. **Cinquenta consultas julgadas, 49 pontuáveis.** Efeitos grandes são
   detectáveis; pequenos não. O artigo diz qual é qual por braço.
3. **Os rótulos de relevância do Quati são gerados por LLM (GPT-4).**
   ⚠️ **Reportar o número corretamente:** o artigo do Quati relata Kappa de
   Cohen de **0,31** contra anotadores humanos, que os autores descrevem como
   *abaixo* dos 0,41 observados entre humanos, embora consistente com a
   literatura. A frase da proposta original — "concordância entre anotadores
   comparável ao desempenho humano" — **superestima isso e não deve ser
   repetida**. Além do ruído, rótulos de LLM não descartam vieses sistemáticos
   correlacionados com as famílias de modelo avaliadas.
4. **O corpus é conteúdo web genérico do ClueWeb22**, não documentação técnica,
   jurídica ou administrativa. As conclusões não transferem automaticamente
   para domínios especializados.
5. **O corpus é predominantemente, mas não exclusivamente, brasileiro.** As
   amostras em português do ClueWeb22 incluem material de Portugal.
6. **Um único modelo de embedding, numa única dimensionalidade.** Os resultados
   podem não transferir. A análise de geometria da §5 é o que torna essa
   limitação informativa em vez de apenas declarada: ela dá o mecanismo pelo
   qual outro modelo se comportaria diferente.
7. **Uma máquina, um dispositivo de armazenamento, uma configuração de
   memória.** Os números de latência são específicos deste hardware.
8. **Latência medida com cache quente sobre consultas repetidas**, que não é um
   perfil de cold start de produção.
9. **As figuras de page cache refletem um conjunto de trabalho aquecido por 50
   consultas.** Uma carga mais diversa tocaria mais dos originais mapeados.
10. **Os números de qualidade e os de custo vêm de builds diferentes do mesmo
    índice.** As tabelas de qualidade saem dos runs de recuperação de
    2026-08-25; as de memória, latência e disco, da varredura de 2026-08-27,
    que reconstruiu cada coleção. São a mesma configuração, mas não a mesma
    instância do grafo HNSW, e a diferença é observável: o A5 dá 0,3770 no run
    reportado e 0,3648 no índice reconstruído. Está dentro da largura do
    intervalo de confiança, mas o texto deve dizer que qualidade e custo foram
    medidos em builds distintos em vez de deixar implícito que são o mesmo.
11. **O modo `exact` do Qdrant ignora a quantização.** Ele varre os vetores
    float32 originais, então não serve como teto de memória de braço
    quantizado, e a decomposição do erro precisou de uma varredura de `ef` no
    lugar dele. O estado reportado é `warm_hnsw`.

---

## 7. O que é reprodutível, e a que custo

| O quê | Precisa de | Custo aproximado |
|---|---|---|
| Todas as tabelas e números | nada além deste repositório | segundos |
| Medições de memória e latência | Qdrant em podman, ~25 GB de disco | ~35 min |
| Índices dos seis braços | idem, mais o artefato de vetores | ~4 min por braço |
| Artefato de vetores (3 GB) | **GPU** | horas |

O caminho sem GPU vai até a regeneração completa das tabelas a partir das
medições publicadas. Refazer a passagem de embeddings exige GPU.

Sementes fixadas: embedding, bootstrap (20260818) e a ordenação do corpus.
Hashes de corpus, qrels e artefato de vetores estão em `paper/numbers.json`.

---

## 8. O que não está feito

Honestidade sobre o estado da entrega:

- **RAF-64** — referência BM25 própria. A calibração passou sem ela, usando o
  BM25 publicado pelos autores do Quati como piso de faixa.
- **RAF-81** — revisão adversarial interna contra as limitações declaradas.

Nenhum desses bloqueia a escrita. O primeiro e o segundo, se houver tempo, são
os que mais agregariam.

---

## 9. Uma nota sobre repetibilidade de footprint

O critério original de M2.1 — reconstruir um braço reproduz o mesmo footprint
dentro de uma tolerância pequena — **não foi atendido como escrito**. A
ocupação em disco varia até 18% entre builds idênticos, porque o Qdrant
pré-aloca chunks de 32 MB por segmento e quantos chunks existem depende de como
o otimizador distribuiu os pontos naquela execução. O *conteúdo* é
reprodutível: braços que repetem repetem o `vector_storage` byte a byte.

**Consequência para o texto:** a afirmação de que comprimir aumenta o
armazenamento total sobrevive, porque as faixas não se sobrepõem. Mas o artigo
deve reportar **faixas**, não um número único por braço, e a margem é da ordem
de duas vezes o ruído de repetição, não de dez.
