"""Construção das coleções: um builder só, parametrizado pela matriz.

Seis coleções construídas por seis scripts divergiriam em coisas que ninguém
pretendeu — um `m` diferente, outra distância, outro tamanho de lote — e essas
diferenças apareceriam na tabela final como efeito de quantização. Aqui o
builder recebe a definição do braço vinda de `config/arms.json` e nada mais, de
modo que a única diferença entre dois braços é a diferença que a matriz declara.

**O texto da passagem não vai para o índice.** Ele acrescentaria gigabytes de
payload a todos os braços e não é parte do que este estudo mede. O payload
guarda só o `passage_id`, e mesmo esse fica em disco: o número do artigo é
memória residente, e um campo que nenhuma consulta lê não pode entrar nele.

**Indexação é assíncrona.** Uma coleção medida enquanto o otimizador ainda
constrói o grafo não dá nem o custo de construção nem o regime estacionário —
dá um número intermediário que não corresponde a nada. Por isso
`wait_until_indexed` exige verde *e* a contagem de vetores indexados, em vez de
confiar que o `upsert` terminou.
"""

from __future__ import annotations

import time

import numpy as np
from qdrant_client import models

from quantptbr import corpus, embedding, retrieval, server, stats, vectors

#: Pontos por requisição. Medido em S1.1: 2.000 × 768 float32 passa em gRPC sem
#: estourar, enquanto 5.000 vira 81 MB de JSON e o REST recusa. Também é o teto
#: de linhas que o array mapeado entrega à RAM de uma vez.
BATCH = 2_000

#: Teto para o otimizador terminar 1M de pontos. Generoso de propósito: estourar
#: o prazo aborta o build, e abortar um build correto custa mais que esperar.
INDEX_TIMEOUT = 4 * 3600

#: Amostra do confronto ponto↔passagem. Compara o payload e o vetor gravados no
#: servidor contra a ordem congelada do corpus.
SPOT_CHECK_POINTS = 256

#: Diretórios que o Qdrant cria dentro de um segmento, na ordem em que o
#: `footprint` os procura. O que não cair em nenhum deles vira "outros".
FOOTPRINT_COMPONENTS = (
    "vector_storage",
    "vector_index",
    "payload_storage",
    "payload_index",
    "wal",
)


def quantization_config(arm: dict):
    """Traduz a declaração da matriz para a configuração do cliente.

    Toda a diferença entre os braços passa por esta função. Ela é deliberadamente
    burra: lê os campos declarados e não inventa padrão nenhum, para que um campo
    esquecido na matriz vire erro em vez de virar um braço silenciosamente
    diferente do pré-registrado.
    """
    declared = arm["quantization"]
    if declared is None:
        return None
    kind = declared["kind"]
    if kind == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType(declared["type"]),
                quantile=declared["quantile"],
                always_ram=declared["always_ram"],
            )
        )
    if kind == "product":
        return models.ProductQuantization(
            product=models.ProductQuantizationConfig(
                compression=models.CompressionRatio(declared["compression"]),
                always_ram=declared["always_ram"],
            )
        )
    if kind == "binary":
        return models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=declared["always_ram"])
        )
    raise ValueError(f"quantização {kind!r} não está no vocabulário da matriz")


def hnsw_config() -> models.HnswConfigDiff:
    """Topologia fixa, igual em todos os braços.

    Sintonizar o HNSW por braço trocaria a variável em estudo: a diferença
    medida deixaria de ser da quantização e passaria a ser da topologia.
    """
    hnsw = stats.matrix()["held_constant"]["hnsw"]
    return models.HnswConfigDiff(m=hnsw["m"], ef_construct=hnsw["ef_construct"])


def distance() -> models.Distance:
    return models.Distance(stats.matrix()["held_constant"]["distance"])


def optimizers_config() -> models.OptimizersConfigDiff:
    """Número de segmentos fixado, não derivado da máquina.

    Deixado em 0, o Qdrant o deriva da contagem de CPUs. A topologia do índice
    passaria a depender do hardware, e com ela a taxa fixa de 32 MB por segmento
    que o armazenamento de payload pré-aloca.
    """
    return models.OptimizersConfigDiff(
        default_segment_number=stats.matrix()["held_constant"]["segments"]
    )


def require_topology(client, collection: str) -> None:
    """Relê do servidor o que a matriz manda manter constante.

    Mesmo motivo de `require_quantization`: uma configuração aceita e não
    aplicada daria seis coleções com a topologia padrão e uma coluna de HNSW que
    descreve o que pedimos, não o que rodou.
    """
    held = stats.matrix()["held_constant"]
    config = client.get_collection(collection).config
    actual = {
        "m": config.hnsw_config.m,
        "ef_construct": config.hnsw_config.ef_construct,
        "distance": config.params.vectors.distance.value,
        "segments": config.optimizer_config.default_segment_number,
    }
    expected = {
        "m": held["hnsw"]["m"],
        "ef_construct": held["hnsw"]["ef_construct"],
        "distance": held["distance"],
        "segments": held["segments"],
    }
    if actual != expected:
        raise RuntimeError(f"{collection}: topologia {actual} diverge da matriz {expected}")


def buildable(arm: dict) -> None:
    """Um braço que compartilha coleção não é construído — é um modo de busca."""
    if arm.get("_shares_collection_with"):
        raise ValueError(
            f"{arm['id']} compartilha a coleção de {arm['_shares_collection_with']} e difere "
            "apenas em parâmetros de busca; construí-la reconstruiria o outro braço"
        )


def create(client, arm: dict, dims: int) -> None:
    client.create_collection(
        collection_name=arm["collection"],
        vectors_config=models.VectorParams(
            size=dims, distance=distance(), on_disk=arm["vectors_on_disk"]
        ),
        hnsw_config=hnsw_config(),
        optimizers_config=optimizers_config(),
        quantization_config=quantization_config(arm),
        # Explícito e não herdado: o padrão do servidor pode mudar entre versões,
        # e este campo entra direto no número de memória residente do artigo.
        on_disk_payload=True,
    )


def resume_point(client, collection: str, total: int) -> int:
    """Onde retomar um upload interrompido, verificado em vez de suposto.

    O envio é sequencial por ID e cada lote espera confirmação, então uma coleção
    com N pontos tem exatamente os IDs 0..N-1. A suposição é conferida — o ponto
    N-1 existe, o N não — porque retomar do lugar errado deixaria buracos no
    índice que só apareceriam depois, como recall baixo num braço só.
    """
    done = client.count(collection, exact=True).count
    if done in (0, total):
        return done
    if done > total:
        raise RuntimeError(f"{collection}: {done:,} pontos, mais que os {total:,} do corpus")
    present = {int(record.id) for record in client.retrieve(collection, ids=[done - 1, done])}
    if present != {done - 1}:
        raise RuntimeError(
            f"{collection}: tem {done:,} pontos mas os IDs não são 0..{done - 1:,} "
            f"(sondagem devolveu {sorted(present)}). Reconstrua com --rebuild."
        )
    return done


def upload(
    client, collection: str, array, passage_ids: list[str], batch: int = BATCH, first: int = 0
) -> int:
    """Envia os pontos em blocos, lendo o array mapeado faixa por faixa.

    O array tem 3 GB. Materializá-lo em RAM para enviar colocaria o processo
    Python e o servidor disputando a mesma memória que o estudo está medindo.
    """
    total = len(passage_ids)
    for start in range(first, total, batch):
        stop = min(start + batch, total)
        block = np.asarray(array[start:stop], dtype=np.float32)
        client.upsert(
            collection_name=collection,
            points=models.Batch(
                ids=list(range(start, stop)),
                vectors=block.tolist(),
                payloads=[{"passage_id": pid} for pid in passage_ids[start:stop]],
            ),
            wait=True,
        )
    return total - first


#: Maior operação de indexação já observada num build desta matriz foi 94 s
#: (telemetria do Qdrant, build de A0 em 2026-08-26). Sem progresso por 5x isso
#: não é uma otimização lenta — é o segmento residual, e ele não vai se mover.
STALL_TIMEOUT = 480.0

#: Pontos que podem ficar fora do HNSW sem que o build seja recusado. O lote
#: final do upload pode cair sozinho num segmento abaixo do `indexing_threshold`
#: do Qdrant; esse segmento é resolvido por varredura exata, com resultado
#: correto, e um lote em 1M é 0,2% dos pontos. Acima disso não é resíduo, é
#: build incompleto.
RESIDUAL_TOLERANCE = BATCH


def wait_until_indexed(
    client,
    collection: str,
    expected: int,
    timeout: float = INDEX_TIMEOUT,
    poll: float = 5.0,
    stall_timeout: float = STALL_TIMEOUT,
    tolerance: int = RESIDUAL_TOLERANCE,
) -> tuple[float, int]:
    """Espera o otimizador terminar. Verde sozinho não basta.

    Logo após o último `upsert` o otimizador ainda não começou, e a coleção
    reporta verde por um instante. Exigir também a contagem de vetores indexados
    fecha essa janela: é a diferença entre afirmar que a indexação terminou e
    supor que terminou.

    **O segmento residual é aceito, não forçado.** Medido em 2026-08-26: o lote
    final pode cair num segmento `plain` abaixo do `indexing_threshold` (~10 MB),
    que por design do Qdrant nunca ganha HNSW e, sem mais escrita na coleção,
    nunca é fundido — `green` com 998.000/1.000.000 fica estável para sempre.
    Duas tentativas de forçar a indexação baixando o limiar terminaram pior que
    o problema: na criação, o otimizador reescreveu segmento sobre segmento
    durante o upload e o A4 foi a 43,3 GB em disco; depois do upload, entrou em
    re-otimização que não converge — 260 segmentos no lugar de 16, pontos
    duplicados, 30,4 GB, e `indexed_vectors_count` em 3.052.848 para 1M de
    pontos. Aceitar o resíduo declarado é mais barato e mais honesto: esses
    pontos são buscados por varredura exata, com resultado correto.

    **Contagem acima do total é prova de que o otimizador não terminou.** O
    Qdrant soma os segmentos, e durante uma re-otimização os mesmos pontos vivem
    em segmentos sobrepostos: o A4 reportou 2.300.544 indexados para 1M de
    pontos, com 144 segmentos no lugar de 16. Limitar a contagem a `expected`
    não resolve — `min(2.300.544, 1M) >= 1M` continua passando, e o build foi
    dado por concluído em 5 s sobre uma coleção que ainda ia churnar por meia
    hora. Enquanto a contagem estiver inflada, a coleção não está pronta.
    """
    started = time.monotonic()
    deadline = started + timeout
    best = -1
    progressed_at = started
    inflated_since: float | None = None
    while True:
        info = client.get_collection(collection)
        raw = info.indexed_vectors_count or 0
        if info.status == models.CollectionStatus.RED:
            raise RuntimeError(f"{collection}: coleção em estado RED")
        # Segmentos sobrepostos: o otimizador ainda está reescrevendo.
        converged = raw <= expected
        indexed = min(raw, expected)
        green = info.status == models.CollectionStatus.GREEN
        if green and converged and indexed >= expected:
            return time.monotonic() - started, int(indexed)
        now = time.monotonic()
        if not converged:
            # Reescrever segmento é trabalho em curso, mas não é ilimitado: uma
            # coleção pode encalhar inflada e ficar assim. Medido em 2026-08-26,
            # o A4 parou em 144 segmentos e 2.300.544 indexados e continuava lá
            # meia hora depois, com o otimizador reportando `ok`.
            if inflated_since is None:
                inflated_since = now
                print(
                    f"    {raw:,} vetores indexados para {expected:,} pontos: segmentos "
                    "sobrepostos, o otimizador ainda está reescrevendo"
                )
            elif now - inflated_since > stall_timeout:
                raise TimeoutError(
                    f"{collection}: {raw:,} vetores indexados para {expected:,} pontos há "
                    f"{stall_timeout / 60:.0f} min — o otimizador encalhou com segmentos "
                    "sobrepostos e não converge; reconstrua a coleção"
                )
            progressed_at = now
        elif indexed > best:
            inflated_since = None
            best, progressed_at = indexed, now
        elif now - progressed_at > stall_timeout:
            missing = expected - indexed
            if green and converged and missing <= tolerance:
                print(
                    f"    {missing:,} pontos em segmento residual, fora do HNSW e "
                    "buscados por varredura exata"
                )
                return time.monotonic() - started, int(indexed)
            raise TimeoutError(
                f"{collection}: travado em {indexed:,}/{expected:,} vetores indexados, "
                f"sem progresso há {stall_timeout / 60:.0f} min e {missing:,} pontos fora do "
                f"índice — acima da tolerância de {tolerance:,}; reconstrua a coleção"
            )
        if now > deadline:
            raise TimeoutError(
                f"{collection}: {info.status} com {indexed:,}/{expected:,} vetores indexados "
                f"após {timeout / 60:.0f} min"
            )
        time.sleep(poll)


def footprint(collection: str) -> dict:
    """Ocupação em disco, decomposta. O total sozinho engana.

    O Qdrant pré-aloca páginas de 32 MB por segmento: com 16 segmentos, o
    armazenamento de payload ocupa 512 MB em disco para guardar ~20 MB de IDs.
    Num braço float32 de 3 GB isso é ruído; num braço quantizado é a maior parte
    do número. Reportar só o total converteria uma taxa fixa do servidor em
    conclusão sobre compressão — e a compressão contra armazenamento total é
    exatamente uma das afirmações do artigo.
    """
    path = server.STORAGE_DIR / "collections" / collection
    if not path.exists():
        raise FileNotFoundError(f"{path} ausente: a coleção não está neste armazenamento")

    by_component = dict.fromkeys([*FOOTPRINT_COMPONENTS, "outros"], 0)
    for entry in path.rglob("*"):
        if not entry.is_file():
            continue
        parts = set(entry.relative_to(path).parts)
        component = next((c for c in FOOTPRINT_COMPONENTS if c in parts), "outros")
        by_component[component] += entry.stat().st_size

    total = sum(by_component.values())
    return {
        "total_bytes": total,
        "total_mb": round(total / 1024**2, 1),
        "by_component_mb": {k: round(v / 1024**2, 1) for k, v in by_component.items()},
    }


def check_points(client, collection: str, array, passage_ids: list[str]) -> dict:
    """O ponto *i* é mesmo a passagem *i*?

    O confronto é contra o **servidor**, não contra a nossa própria lista: o
    payload e o vetor gravados têm de reproduzir a ordem congelada em S1.0. Um
    deslocamento de uma posição produziria nDCG plausível e errado, e é
    exatamente o que a comparação entre braços não revelaria.
    """
    counted = client.count(collection, exact=True).count
    if counted != len(passage_ids):
        raise RuntimeError(f"{collection} tem {counted:,} pontos, esperados {len(passage_ids):,}")

    rng = np.random.default_rng(0)
    sample = min(SPOT_CHECK_POINTS, len(passage_ids))
    ids = sorted(int(i) for i in rng.choice(len(passage_ids), size=sample, replace=False))
    records = {
        int(record.id): record
        for record in client.retrieve(collection, ids=ids, with_payload=True, with_vectors=True)
    }
    if set(records) != set(ids):
        raise RuntimeError(f"{collection}: servidor não devolveu {sorted(set(ids) - set(records))}")

    worst_cosine = 1.0
    for point_id in ids:
        record = records[point_id]
        stored_id = record.payload.get("passage_id")
        if stored_id != passage_ids[point_id]:
            raise RuntimeError(
                f"{collection}: ponto {point_id} carrega {stored_id!r}, "
                f"esperado {passage_ids[point_id]!r}"
            )
        stored = np.asarray(record.vector, dtype=np.float64)
        original = np.asarray(array[point_id], dtype=np.float64)
        worst_cosine = min(worst_cosine, float(stored @ original))

    # O servidor renormaliza para Cosine; os vetores já chegam unitários, então o
    # que sobra é arredondamento de float32 e nada mais.
    if worst_cosine < 1 - 1e-5:
        raise RuntimeError(f"{collection}: vetor gravado diverge do original (cos {worst_cosine})")
    return {"sampled": sample, "worst_cosine": worst_cosine, "counted": counted}


def build(client, arm_id: str, recreate: bool = False, resume: bool = False, array=None) -> dict:
    """Constrói um braço inteiro a partir da sua definição na matriz."""
    arm = retrieval.arm_config(arm_id)
    buildable(arm)
    collection = arm["collection"]

    server.require_server(client)
    corpus.verify_manifest()
    envelope = embedding.Envelope.load()

    passage_ids = corpus.load_passage_ids()
    array = vectors.load() if array is None else array
    if array.shape != (len(passage_ids), envelope.dimensions):
        raise RuntimeError(
            f"artefato {array.shape} não bate com ({len(passage_ids)}, {envelope.dimensions})"
        )

    exists = client.collection_exists(collection)
    resume_from = 0
    if exists and recreate:
        client.delete_collection(collection)
        exists = False
    elif exists and resume:
        resume_from = resume_point(client, collection, len(passage_ids))
        print(f"{arm_id} → {collection}: retomando em {resume_from:,}")
    elif exists:
        raise RuntimeError(
            f"{collection} já existe. Passe recreate=True para reconstruir do zero, "
            "ou resume=True para continuar um envio interrompido."
        )
    if not exists:
        print(f"{arm_id} → {collection}: criando coleção …")
        create(client, arm, envelope.dimensions)

    print(f"{arm_id}: enviando {len(passage_ids) - resume_from:,} pontos em lotes de {BATCH:,} …")
    started = time.perf_counter()
    upload(client, collection, array, passage_ids, first=resume_from)
    upload_seconds = time.perf_counter() - started

    print(f"{arm_id}: aguardando o otimizador …")
    index_seconds, indexed = wait_until_indexed(client, collection, len(passage_ids))

    # Config lida de volta do servidor: uma coleção cuja quantização não pegou
    # não é um braço, é o A0 com outro nome.
    require_topology(client, collection)
    if arm["quantization"] is not None:
        server.require_quantization(client, collection, arm["quantization"]["kind"])

    spot_check = check_points(client, collection, array, passage_ids)
    disk = footprint(collection)

    report = {
        "arm": arm_id,
        "collection": collection,
        "points": len(passage_ids),
        # Um build retomado não tem tempo de parede comparável: `upload_seconds`
        # cobre só o que este processo enviou. `resumed_from` > 0 marca isso.
        "resumed_from": resume_from,
        "upload_seconds": round(upload_seconds, 1),
        "index_seconds": round(index_seconds, 1),
        "indexed_vectors_count": indexed,
        # Pontos fora do HNSW, num segmento abaixo do indexing_threshold. São
        # buscados por varredura exata; a tabela final tem de poder dizer isso.
        "unindexed_points": len(passage_ids) - indexed,
        "footprint": disk,
        "spot_check": spot_check,
    }
    print(
        f"{arm_id}: {report['points']:,} pontos, upload {upload_seconds / 60:.1f} min, "
        f"indexação {index_seconds / 60:.1f} min, disco {disk['total_mb']:,.1f} MB "
        f"({disk['by_component_mb']})"
    )
    return report
