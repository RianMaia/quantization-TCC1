# Resultados

Texto da seção de resultados do artigo. As tabelas não são redigitadas aqui:
cada uma é colada do arquivo gerado que a acompanha, e os números citados no
texto foram conferidos contra esses arquivos. A calibração dos verbos segue
`paper/HANDOFF.md`, seção 5: intervalo de confiança inteiramente abaixo de zero
autoriza "degrada"; intervalo que contém zero autoriza apenas "não apresenta
degradação detectável neste tamanho de amostra", nunca equivalência.

| Tabela no texto | Arquivo de origem |
|---|---|
| Tabela 3 (E5) | `paper/exhaustive_e5.md` |
| Tabela 3 (BGE-M3) | `paper/exhaustive_bge-m3.md` |
| Comparação entre trilhas e entre modelos | `paper/comparacao_trilhas.md`, figura `paper/figures/comparacao_trilhas.png` |
| T1 a T8 (trilha do servidor) | `paper/tables.md` |

## Validação do aparelho de medição

Antes de medir quantização, a linha de base foi confrontada com o único
resultado publicado de um modelo denso sobre o Quati 1M (T6). Pelo protocolo
dos autores do benchmark, o E5-base medido aqui dá nDCG@10 de 0,3824, contra
0,3955 publicado, uma diferença de −0,0131, dentro da tolerância de 0,02
declarada antes da medição. Na régua do estudo, a busca exata do A0 dá 0,4669 e
fica entre o BM25 (0,3991) e o ColBERT-X mMARCO pt-BR (0,4927), a faixa
esperada para um modelo denso desse porte. Os demais resultados só são lidos a
partir deste ponto.

## Quantização isolada: busca exaustiva

A Tabela 3 mede as quatro condições do documento de arquitetura com busca
exaustiva, de modo que a única fonte de perda é a representação dos vetores.
Com o E5 e o protocolo do documento (ganho 2^rel − 1, 50 consultas), o F32
atinge nDCG@10 de 0,4356, Recall@100 de 0,6514 e MRR@100 de 0,8253.

- **F16** não apresenta degradação detectável neste tamanho de amostra:
  Δ nDCG@10 = −0,0030, IC 95% [−0,0073; +0,0004]. O top-10 coincide com o do
  F32 em 99,2% das posições, com metade da memória (1,536 GB contra 3,072 GB).
- **I8** também não apresenta degradação detectável neste tamanho de amostra:
  Δ = −0,0019, IC 95% [−0,0071; +0,0034], com top-10 igual ao do F32 em 98,0%
  e um quarto da memória (0,768 GB). O MRR@100 é o mesmo do F32 (0,8253).
- **B1** degrada: Δ = −0,3267, IC 95% [−0,3846; −0,2694]. O nDCG@10 cai para
  0,1089, o Recall@100 para 0,1469 e o MRR@100 para 0,3186, e só 12,4% do
  top-10 coincide com o do F32. A compressão é de 32 vezes (0,096 GB).

Na busca exaustiva a memória escala exatamente com a razão nominal (2x, 4x e
32x), porque só os códigos ficam em memória. Esse ponto contrasta com o
servidor, discutido adiante.

Com o BGE-M3, modelo fixado pelo documento, o F32 atinge nDCG@10 de 0,5065,
Recall@100 de 0,7267 e MRR@100 de 0,8177:

- **F16** e **I8** não apresentam degradação detectável neste tamanho de
  amostra: Δ = +0,0014, IC 95% [−0,0001; +0,0043], e Δ = +0,0016, IC 95%
  [−0,0002; +0,0048]. As estimativas pontuais positivas não autorizam dizer
  que a compressão melhora a busca. O top-10 coincide com o do F32 em 99,6% e
  98,8%.
- **B1** degrada: Δ = −0,0688, IC 95% [−0,1049; −0,0334]. Mas a perda é muito
  menor que a do E5: o nDCG@10 fica em 0,4377, o Recall@100 em 0,6295, o
  MRR@100 em 0,7667, e 57,2% do top-10 coincide com o do F32. A memória cai de
  4,096 GB para 0,128 GB.

## Quantização no servidor: Qdrant com HNSW

Na trilha do servidor (T1, ganho linear, 49 consultas pontuáveis), o int8 (A1)
não apresenta degradação detectável neste tamanho de amostra: Δ = −0,0051,
IC 95% [−0,0217; +0,0082], com diferença mínima detectável de 0,0218. A
quantização por produto degrada a 16x (A2, Δ = −0,0244) e a 32x (A3,
Δ = −0,0727). A binária (A4) colapsa, de 0,4655 para 0,0773, e o
oversampling com reordenação em float32 (A5) recupera boa parte disso, até
0,3770, ainda abaixo da linha de base (Δ = −0,0885, IC 95% [−0,1280; −0,0525]).

## Busca exaustiva × Qdrant com o mesmo modelo

A comparação entre as trilhas separa o que é efeito da quantização do que é
efeito do mecanismo de busca. Com o E5 e a mesma régua (ganho linear, 49
consultas):

- **int8:** a busca exaustiva perde 0,0010 em relação ao F32, e o Qdrant, com o
  grafo convergido e sem reordenação, perde 0,0078. As duas perdas são menores
  que 0,01. A diferença entre elas é compatível com a regra de escala de cada
  trilha: o máximo por dimensão no documento, o quantil 0,99 no Qdrant.
- **binária:** as duas trilhas concordam quando só a quantização é isolada:
  0,1148 na busca exaustiva e 0,1139 no platô do Qdrant, perdas de 0,3540 e
  0,3530. O colapso é, portanto, propriedade do código binário sobre estes
  vetores, e não do mecanismo de busca.
- **O grafo custa pouco, exceto no binário.** O HNSW sobre float32 perde
  0,0014 em relação à busca exata. No binário, o Qdrant como medido (0,0773)
  fica 0,0367 abaixo do platô de quantização (T4): com códigos tão pobres, a
  busca aproximada perde vizinhos que a exaustiva acharia.

A geometria dos vetores explica o colapso em vez de apenas registrá-lo (T5). O
espaço do E5 parece centrado no agregado (média das médias por dimensão de
−0,00018), mas não dimensão a dimensão: o deslocamento médio é de 1,614 desvios,
e 323 das 768 dimensões são degeneradas em sinal. O código binário fica com
324,7 bits efetivos de 768 (42,3%).

## Custo: memória e latência

**Memória.** A economia de memória depende de onde os originais ficam. Na busca
exaustiva, que guarda só os códigos, a redução é a nominal. No Qdrant, que
mantém os vetores float32 ao lado dos códigos, comprimir aumentou a memória a
provisionar em todos os braços (T2): o int8 exige 3,11 vezes o piso do float32.
Por isso a fronteira de Pareto de qualidade × memória a provisionar tem um
ponto só, o A0 (T7).

**Latência no servidor.** No par qualidade × latência a fronteira tem quatro
pontos (A0, A1, A4 e A5). O int8 responde em 10,6 ms de mediana contra 16,6 ms
do float32, e o binário com reordenação em 8,1 ms (T3). É nesse eixo que
comprimir compensa. A recomendação é condicional: se a restrição é memória a
provisionar, não comprimir; se é latência, comprimir e reordenar.

**Latência na busca exaustiva.** Na busca exaustiva, F16 e I8 são
reconstruídos em float32 bloco a bloco antes do produto interno, como a seção
27.1 do documento descreve. A latência deles mede essa reconstrução em NumPy, e
não aceleração nativa, por isso sai maior que a do F32. O B1 compara por
distância de Hamming sobre palavras de 64 bits e é a única condição mais rápida
que o F32. A ordem é a mesma nos dois modelos. Com o E5, as medianas são de
215,2 ms (B1), 525,8 ms (F32), 4.692,3 ms (I8) e 6.644,1 ms (F16). Com o
BGE-M3, são de 292,8 ms, 897,6 ms, 6.977,4 ms e 8.693,8 ms.

Os valores absolutos não devem ser lidos como desempenho da máquina. As duas
execuções rodaram com a RAM ocupada por outros processos. Numa execução de
27/10 com a máquina livre, o F32 do E5 mediu 148,8 ms, um terço do valor
registrado na Tabela 3. Pelo mesmo motivo, a diferença de latência entre os
dois modelos não é interpretável.

**Ponto de operação do binário com reordenação (T8).** A varredura de
oversampling do A5 não é plana: de 0,2747 em 1x a 0,4187 em 16x, a 20,6 ms. O
valor reportado continua sendo o pré-registrado (4x, 0,3648 no índice
reconstruído). A varredura é diagnóstico: mostra que o parâmetro tem efeito
forte e merece ser ajustado em produção, sem trocar o resultado depois de ver os
dados.

## BGE-M3 × E5

Sob o mesmo protocolo, os dois modelos respondem igual à compressão moderada e
de forma oposta à agressiva (figura `comparacao_trilhas.png`, painel da
direita).

- **F16 e I8:** nenhum dos dois modelos apresenta degradação detectável neste
  tamanho de amostra. Os dois retêm mais de 99% do nDCG@10 do próprio F32.
- **B1:** os dois degradam, mas em escalas diferentes. O E5 retém 25,0% do
  nDCG@10 do F32, e o BGE-M3 retém 86,4%. Em estimativa pontual, o BGE-M3
  binário, com 0,128 GB, fica no patamar do E5 em float32, com 3,072 GB
  (0,4377 contra 0,4356). Essa diferença entre modelos não foi testada
  estatisticamente aqui.
- **Ponto de partida:** o F32 do BGE-M3 fica acima do F32 do E5 (0,5065 contra
  0,4356, e 0,5431 contra 0,4688 na régua do estudo). Essa diferença também
  não foi testada. Os modelos também ordenam de forma diferente: o top-10 do
  F32 do BGE-M3 coincide com o da busca exata do A0 (E5) em só 44,2%.

A conclusão sobre a quantização binária, portanto, depende do modelo. No E5, a
geometria explica o colapso: 323 das 768 dimensões são degeneradas em sinal
(T5). A mesma análise ainda não foi feita sobre os vetores do BGE-M3. Ela é o
próximo passo para dizer se a diferença entre os modelos vem do mesmo
mecanismo.

## Limitações que acompanham estes resultados

Resumo de `paper/HANDOFF.md`, seção 6, que deve aparecer no texto final:

1. É um estudo de medição, não de método.
2. São 50 consultas julgadas (49 pontuáveis na régua do estudo): efeitos grandes
   são detectáveis, pequenos não.
3. Os rótulos do Quati são gerados por LLM, com Kappa de 0,31 contra
   anotadores humanos, abaixo dos 0,41 observados entre humanos.
4. O corpus é conteúdo web genérico, não documentação técnica nem jurídica.
5. O corpus é predominantemente, mas não exclusivamente, brasileiro.
6. A latência é específica desta máquina e, na busca exaustiva, do estado dela
   no momento da medição. As duas tabelas exaustivas foram medidas com a RAM
   sob pressão.
7. Dois modelos na busca exaustiva (E5 e BGE-M3), mas só o E5 na trilha do
   servidor. O resultado de que o binário colapsa vale para o E5 e não se
   transfere para o BGE-M3.
8. No servidor, qualidade e custo vêm de builds distintos do mesmo índice.
9. O modo `exact` do Qdrant ignora a quantização, o que exigiu a varredura de
   `ef` para decompor o erro.
