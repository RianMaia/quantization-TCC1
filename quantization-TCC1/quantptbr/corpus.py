"""Acesso congelado ao corpus Quati 1M, aos tópicos e aos julgamentos de relevância.

Os IDs dos pontos em toda coleção Qdrant são o índice da passagem em
`iter_passages`. Essa ordem é um contrato: uma vez executada a passagem de
embedding, mudá-la invalida todos os índices construídos sobre ela.

O acesso é por arquivo direto numa revisão fixa, e não por
`load_dataset(..., trust_remote_code=True)`: assim a entrada é fixável,
hasheável e não executa código de terceiros.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

DATASET = "unicamp-dl/quati"
REVISION = "e5279055bba3e7ba1bece5c0eddd0ba232df49c3"

#: Caminhos relativos no repositório do dataset, preservados localmente.
REMOTE_FILES = {
    "corpus": "quati_1M.tsv",
    "qrels": "qrels/quati_1M_qrels.txt",
    "qrels_10m": "qrels/quati_10M_qrels.txt",
    "topics": "topics/quati_all_topics.tsv",
    "test_topics": "topics/quati_test_topics.tsv",
}

#: Os qrels de 10M não são gabarito deste estudo — o corpus é o de 1M e é contra
#: `qrels` que todo braço é medido. Eles existem aqui por um motivo só: a única
#: linha publicada de E5-base sobre o Quati (Tabela 6, seção "1M Passages") é um
#: run de 1M pontuado com estes julgamentos, e reproduzir esse número é o que
#: valida o aparelho em S1.7. Usá-los em qualquer outro lugar seria medir contra
#: um pool de relevantes que o corpus de 1M nem contém.
CALIBRATION_QRELS = "qrels_10m"

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "MANIFEST.json"

#: Figuras publicadas em Oliveira et al. (STIL 2024) para a versão de 1M.
#: Divergência não é arredondada: é investigada antes de qualquer indexação.
EXPECTED_PASSAGES = 1_000_000
EXPECTED_JUDGED_QUERIES = 50
EXPECTED_MEAN_JUDGED = 38.66

#: Escala ordinal observada nos qrels de 1M. São quatro graus, não dois.
RELEVANCE_GRADES = (0, 1, 2, 3)

#: A consulta 2 ("Por que os países Guiana e Suriname não são filiados a
#: Conmebol?") tem 53 julgamentos, todos de grau 0: nenhuma passagem relevante
#: existe para ela no corpus de 1M. Seu DCG ideal é nulo, o que deixa o nDCG
#: indefinido — 0 por convenção em algumas ferramentas, omitido em outras.
#: Excluí-la da média ou não é decisão de S1.2/S1.7. Aqui o fato apenas fica
#: exposto e travado contra uma mudança silenciosa do dataset.
KNOWN_UNSCOREABLE_QUERY_IDS = ("2",)


def remote_url(name: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{REMOTE_FILES[name]}"


def local_path(name: str) -> Path:
    return RAW_DIR / REMOTE_FILES[name]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} ausente. Rode `python scripts/01_fetch_quati.py` primeiro."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_manifest() -> None:
    """Confere que os arquivos em disco ainda são os que o manifesto registrou.

    Toda etapa a jusante chama isto antes de ler o corpus: uma entrada trocada
    silenciosamente contaminaria todos os braços de forma idêntica, o que é
    justamente o tipo de falha que a comparação entre braços não revelaria.
    """
    manifest = read_manifest()
    if manifest["revision"] != REVISION:
        raise ValueError(f"manifesto na revisão {manifest['revision']}, código espera {REVISION}")
    for name, entry in manifest["files"].items():
        path = local_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} ausente mas presente no manifesto")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"{path}: sha256 {actual} != manifesto {entry['sha256']}")


def iter_passages() -> Iterator[tuple[str, str]]:
    """Percorre (passage_id, passage) na ordem do arquivo, que define os IDs dos pontos."""
    path = local_path("corpus")
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{lineno}: esperados 2 campos separados por tab, obtidos {len(fields)}"
                )
            yield fields[0], fields[1]


def load_passage_ids() -> list[str]:
    """IDs das passagens na ordem do corpus. O índice na lista é o ID do ponto."""
    return [passage_id for passage_id, _ in iter_passages()]


def load_qrels(name: str = "qrels") -> dict[str, dict[str, int]]:
    """Julgamentos oficiais no formato TREC, como {query_id: {passage_id: grau}}.

    Os graus são ordinais e incluem 0. Qual grau conta como relevante é uma
    decisão fixada em S1.7 e registrada no dicionário de métricas — não aqui.
    """
    path = local_path(name)
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{lineno}: esperados 4 campos TREC, obtidos {len(fields)}")
            query_id, _iteration, passage_id, relevance = fields
            judgements = qrels.setdefault(query_id, {})
            if passage_id in judgements:
                raise ValueError(
                    f"{path}:{lineno}: julgamento duplicado para ({query_id}, {passage_id})"
                )
            judgements[passage_id] = int(relevance)
    return qrels


def load_topics(name: str = "topics") -> dict[str, str]:
    path = local_path(name)
    topics: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        header = next(fh).rstrip("\n").split("\t")
        if header != ["query_id", "query"]:
            raise ValueError(f"{path}: cabeçalho inesperado {header}")
        for lineno, line in enumerate(fh, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path}:{lineno}: esperados 2 campos, obtidos {len(fields)}")
            topics[fields[0]] = fields[1]
    return topics


def judged_query_ids(qrels: dict[str, dict[str, int]] | None = None) -> list[str]:
    """As consultas pontuáveis, ordenadas numericamente.

    Só estas entram em qualquer média. Incluir as 200 consultas do arquivo de
    tópicos diluiria a média com consultas sem gabarito.
    """
    qrels = load_qrels() if qrels is None else qrels
    return sorted(qrels, key=int)


def unscoreable_query_ids(qrels: dict[str, dict[str, int]] | None = None) -> list[str]:
    """Consultas julgadas sem nenhuma passagem relevante, logo sem DCG ideal.

    Elas pontuam 0 de forma idêntica em todos os braços, então não afetam as
    diferenças pareadas — mas puxam o nDCG absoluto para baixo e quebram a
    comparabilidade com as linhas de base publicadas.
    """
    qrels = load_qrels() if qrels is None else qrels
    return sorted((q for q, v in qrels.items() if not any(g > 0 for g in v.values())), key=int)
