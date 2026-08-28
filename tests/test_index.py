"""O builder das coleções: um caminho só, parametrizado pela matriz.

Estes testes travam as três propriedades que tornam a tabela final interpretável:
que a configuração de cada braço saia da matriz e não do código, que o índice
carregue só o que o estudo mede, e que uma coleção ainda em construção nunca
seja tratada como pronta.

Nenhum deles precisa de servidor: o que está sob teste é a tradução da matriz
para a configuração e o protocolo de espera, não o Qdrant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from qdrant_client import models

from quantptbr import index, retrieval, server, stats

DIMS = 4


@dataclass
class FakeInfo:
    status: models.CollectionStatus
    indexed_vectors_count: int | None


@dataclass
class FakeRecord:
    id: int
    payload: dict
    vector: list[float]


@dataclass
class FakeCount:
    count: int


@dataclass
class FakeClient:
    """O mínimo do cliente que o builder toca, com tudo registrado."""

    states: list[FakeInfo] = field(default_factory=list)
    records: list[FakeRecord] = field(default_factory=list)
    points: int = 0
    created: dict | None = None
    upserts: list[dict] = field(default_factory=list)

    def create_collection(self, **kwargs):
        self.created = kwargs

    def upsert(self, collection_name, points, wait):
        self.upserts.append({"collection": collection_name, "points": points, "wait": wait})

    def get_collection(self, _collection):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def count(self, _collection, exact):
        assert exact is True
        return FakeCount(self.points)

    def retrieve(self, _collection, ids, with_payload=False, with_vectors=False):
        self.retrieved_with = {"payload": with_payload, "vectors": with_vectors}
        wanted = set(ids)
        return [r for r in self.records if r.id in wanted]


class SliceRecorder:
    """Fica no lugar do array mapeado e recusa leituras maiores que um lote.

    É a forma direta de testar a proibição de S1.5/M2.1: carregar os 3 GB para a
    RAM colocaria o Python e o servidor disputando a memória que o estudo mede.
    """

    def __init__(self, rows: int, dims: int, limit: int):
        self.data = np.tile(np.arange(rows, dtype=np.float32)[:, None], (1, dims))
        self.shape = self.data.shape
        self.limit = limit
        self.slices: list[tuple[int, int]] = []

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, _ = key.indices(self.shape[0])
            if stop - start > self.limit:
                raise AssertionError(f"leitura de {stop - start} linhas excede o lote {self.limit}")
            self.slices.append((start, stop))
        return self.data[key]


def test_every_arm_translates_to_the_configuration_the_matrix_declares():
    """A tradução é a única diferença entre braços; ela sai da matriz, não daqui."""
    configs = {arm["id"]: index.quantization_config(arm) for arm in stats.matrix()["arms"]}
    assert configs["A0"] is None
    assert configs["A1"].scalar.type == models.ScalarType.INT8
    assert configs["A1"].scalar.quantile == 0.99
    assert configs["A2"].product.compression == models.CompressionRatio.X16
    assert configs["A3"].product.compression == models.CompressionRatio.X32
    assert configs["A4"].binary is not None
    for arm_id in ("A1", "A2", "A3", "A4"):
        declared = retrieval.arm_config(arm_id)["quantization"]
        applied = next(iter(configs[arm_id].model_dump().values()))
        assert applied["always_ram"] == declared["always_ram"], arm_id


def test_unknown_quantization_kind_is_refused():
    with pytest.raises(ValueError, match="vocabulário da matriz"):
        index.quantization_config({"quantization": {"kind": "ternary"}})


def test_topology_and_distance_are_held_constant_across_arms():
    held = stats.matrix()["held_constant"]
    hnsw = index.hnsw_config()
    assert (hnsw.m, hnsw.ef_construct) == (held["hnsw"]["m"], held["hnsw"]["ef_construct"])
    assert index.distance().value == held["distance"]
    assert index.optimizers_config().default_segment_number == held["segments"]


def test_the_indexing_threshold_is_left_alone():
    """Baixar o limiar saiu pior que o resíduo que ele resolvia.

    Na criação, o otimizador reescreveu segmento sobre segmento durante o upload
    e o A4 foi a 43,3 GB em disco. Depois do upload, entrou em re-otimização que
    não converge: 260 segmentos no lugar de 16 e 30,4 GB.
    """
    assert index.optimizers_config().indexing_threshold is None
    assert not hasattr(index, "sweep_residual_segments")


def test_segment_count_is_pinned_instead_of_derived_from_the_machine():
    """Em 0, o Qdrant o deriva das CPUs e a topologia passa a depender do host."""
    assert stats.matrix()["held_constant"]["segments"] > 0


@dataclass
class FakeConfig:
    hnsw_config: object
    params: object
    optimizer_config: object


def topology_client(**overrides) -> FakeClient:
    held = stats.matrix()["held_constant"]
    values = {
        "m": held["hnsw"]["m"],
        "ef_construct": held["hnsw"]["ef_construct"],
        "distance": models.Distance(held["distance"]),
        "segments": held["segments"],
        **overrides,
    }
    info = FakeConfig(
        hnsw_config=models.HnswConfigDiff(m=values["m"], ef_construct=values["ef_construct"]),
        params=models.CollectionParams(
            vectors=models.VectorParams(size=DIMS, distance=values["distance"])
        ),
        optimizer_config=models.OptimizersConfigDiff(default_segment_number=values["segments"]),
    )
    return FakeClient(states=[type("Info", (), {"config": info})()])


def test_topology_is_read_back_from_the_server():
    index.require_topology(topology_client(), "c")


@pytest.mark.parametrize(
    "override",
    [{"m": 32}, {"ef_construct": 200}, {"distance": models.Distance.DOT}, {"segments": 4}],
)
def test_topology_that_did_not_take_is_refused(override):
    """Config aceita e não aplicada daria uma coluna que descreve o pedido, não o run."""
    with pytest.raises(RuntimeError, match="topologia"):
        index.require_topology(topology_client(**override), "c")


def test_arm_that_shares_a_collection_is_not_buildable():
    """Construir o A5 reconstruiria o A4; ele é um modo de busca, não uma coleção."""
    index.buildable(retrieval.arm_config("A4"))
    with pytest.raises(ValueError, match="compartilha a coleção"):
        index.buildable(retrieval.arm_config("A5"))


def test_collection_is_created_from_the_arm_and_keeps_payload_off_ram():
    client = FakeClient()
    index.create(client, retrieval.arm_config("A0"), DIMS)
    created = client.created
    assert created["vectors_config"].size == DIMS
    assert created["vectors_config"].distance == index.distance()
    assert created["vectors_config"].on_disk is False
    assert created["quantization_config"] is None
    assert created["on_disk_payload"] is True, "payload residente entraria no número do artigo"
    assert created["hnsw_config"] == index.hnsw_config()
    assert created["optimizers_config"] == index.optimizers_config()


def test_upload_streams_the_array_and_never_materialises_it():
    client = FakeClient()
    rows, batch = 10, 4
    array = SliceRecorder(rows, DIMS, limit=batch)
    passage_ids = [f"p{i}" for i in range(rows)]

    sent = index.upload(client, "c", array, passage_ids, batch=batch)

    assert sent == rows
    assert array.slices == [(0, 4), (4, 8), (8, 10)], "as faixas têm de ladrilhar o array"
    assert [len(u["points"].ids) for u in client.upserts] == [4, 4, 2]
    assert [i for u in client.upserts for i in u["points"].ids] == list(range(rows))


def test_payload_carries_the_passage_id_and_nothing_else():
    """O texto da passagem custaria gigabytes por braço e não é o que se mede."""
    client = FakeClient()
    passage_ids = ["pa", "pb", "pc"]
    index.upload(client, "c", SliceRecorder(3, DIMS, 3), passage_ids, batch=3)
    payloads = client.upserts[0]["points"].payloads
    assert payloads == [{"passage_id": p} for p in passage_ids]
    assert all(set(p) == {"passage_id"} for p in payloads)


def test_point_ids_are_the_frozen_corpus_positions():
    client = FakeClient()
    index.upload(client, "c", SliceRecorder(5, DIMS, 2), [f"p{i}" for i in range(5)], batch=2)
    assert [u["points"].ids for u in client.upserts] == [[0, 1], [2, 3], [4]]


def test_upload_resumes_where_it_stopped_without_reuploading():
    client = FakeClient()
    array = SliceRecorder(10, DIMS, limit=4)
    sent = index.upload(client, "c", array, [f"p{i}" for i in range(10)], batch=4, first=4)
    assert sent == 6
    assert [i for u in client.upserts for i in u["points"].ids] == list(range(4, 10))
    assert array.slices == [(4, 8), (8, 10)], "não relê o que já foi enviado"


def test_resume_point_verifies_the_ids_instead_of_trusting_the_count():
    """Retomar do lugar errado deixaria buracos que só aparecem como recall baixo."""
    client = FakeClient(points=400, records=[FakeRecord(399, {}, [])])
    assert index.resume_point(client, "c", 1000) == 400

    with_hole = FakeClient(points=400, records=[FakeRecord(400, {}, [])])
    with pytest.raises(RuntimeError, match="IDs não são"):
        index.resume_point(with_hole, "c", 1000)


def test_resume_point_handles_the_empty_and_the_finished_collection():
    assert index.resume_point(FakeClient(points=0), "c", 1000) == 0
    assert index.resume_point(FakeClient(points=1000), "c", 1000) == 1000
    with pytest.raises(RuntimeError, match="mais que os"):
        index.resume_point(FakeClient(points=1001), "c", 1000)


def test_yellow_collection_is_not_treated_as_ready():
    client = FakeClient(
        states=[
            FakeInfo(models.CollectionStatus.YELLOW, 0),
            FakeInfo(models.CollectionStatus.GREY, 500),
            FakeInfo(models.CollectionStatus.GREEN, 1000),
        ]
    )
    _, indexed = index.wait_until_indexed(client, "c", expected=1000, poll=0)
    assert indexed == 1000
    assert client.states == [FakeInfo(models.CollectionStatus.GREEN, 1000)]


def test_green_with_an_incomplete_index_is_refused_not_assumed():
    """Verde logo após o upsert é o otimizador que ainda nem começou."""
    client = FakeClient(states=[FakeInfo(models.CollectionStatus.GREEN, 999)])
    with pytest.raises(TimeoutError, match="999/1,000"):
        index.wait_until_indexed(client, "c", expected=1000, timeout=0, poll=0)


def test_a_residual_segment_within_tolerance_is_accepted_not_forced():
    """0,2% dos pontos fora do HNSW são buscados por varredura exata, e isso basta."""
    client = FakeClient(states=[FakeInfo(models.CollectionStatus.GREEN, 998_000)])
    _, indexed = index.wait_until_indexed(client, "c", expected=1_000_000, stall_timeout=0, poll=0)
    assert indexed == 998_000


def test_a_shortfall_beyond_tolerance_is_still_a_failed_build():
    client = FakeClient(states=[FakeInfo(models.CollectionStatus.GREEN, 900_000)])
    with pytest.raises(TimeoutError, match="acima da tolerância"):
        index.wait_until_indexed(client, "c", expected=1_000_000, stall_timeout=0, poll=0)


def test_an_inflated_count_is_never_read_as_a_finished_build():
    """O A4 reportou 2.300.544 indexados para 1M de pontos, verde, em 5 s.

    Limitar a contagem a `expected` não bastava: `min(2.300.544, 1M) >= 1M`
    passava, e o build era dado por concluído sobre 144 segmentos sobrepostos
    que ainda iam churnar por meia hora.
    """
    inflated = FakeInfo(models.CollectionStatus.GREEN, 2_300_544)
    client = FakeClient(states=[inflated])
    with pytest.raises(TimeoutError):
        index.wait_until_indexed(
            client, "c", expected=1_000_000, timeout=0, stall_timeout=0, poll=0
        )


def test_a_collection_stuck_inflated_is_not_waited_on_for_hours():
    """O A4 encalhou em 144 segmentos e 2.300.544 indexados e ficou lá.

    Sem um teto próprio para o estado inflado, o relógio de travamento reinicia
    a cada volta e a espera corre as 4h inteiras do teto global.
    """
    stuck = FakeInfo(models.CollectionStatus.GREEN, 2_300_544)
    client = FakeClient(states=[stuck])
    with pytest.raises(TimeoutError, match="encalhou"):
        index.wait_until_indexed(client, "c", expected=1_000_000, stall_timeout=0, poll=0)


def test_the_optimizer_is_given_time_while_the_count_is_still_inflated():
    """Contagem inflada é trabalho em curso, não travamento: o relógio reinicia."""
    client = FakeClient(
        states=[
            FakeInfo(models.CollectionStatus.GREEN, 2_300_544),
            FakeInfo(models.CollectionStatus.GREEN, 1_500_000),
            FakeInfo(models.CollectionStatus.GREEN, 1_000_000),
        ]
    )
    _, indexed = index.wait_until_indexed(client, "c", expected=1_000_000, stall_timeout=30, poll=0)
    assert indexed == 1_000_000


def test_red_collection_aborts_immediately():
    client = FakeClient(states=[FakeInfo(models.CollectionStatus.RED, 0)])
    with pytest.raises(RuntimeError, match="RED"):
        index.wait_until_indexed(client, "c", expected=1, poll=0)


def make_spot_check_client(rows: int, array, passage_ids, shift: int = 0) -> FakeClient:
    return FakeClient(
        points=rows,
        records=[
            FakeRecord(
                id=i,
                payload={"passage_id": passage_ids[(i + shift) % rows]},
                vector=list(np.asarray(array[i], dtype=float)),
            )
            for i in range(rows)
        ],
    )


def test_spot_check_confronts_the_server_against_the_frozen_order(monkeypatch):
    monkeypatch.setattr(index, "SPOT_CHECK_POINTS", 8)
    rows = 16
    array = np.eye(rows, DIMS + rows, dtype=np.float32)[:, :rows]
    passage_ids = [f"p{i}" for i in range(rows)]
    client = make_spot_check_client(rows, array, passage_ids)

    result = index.check_points(client, "c", array, passage_ids)
    assert result["sampled"] == 8
    assert result["worst_cosine"] == pytest.approx(1.0)
    assert client.retrieved_with == {"payload": True, "vectors": True}, (
        "sem os dois o confronto seria contra a nossa própria lista, não contra o servidor"
    )


def test_spot_check_catches_an_off_by_one_in_the_payload(monkeypatch):
    """Um deslocamento de uma posição produziria nDCG plausível e errado."""
    monkeypatch.setattr(index, "SPOT_CHECK_POINTS", 8)
    rows = 16
    array = np.eye(rows, rows, dtype=np.float32)
    passage_ids = [f"p{i}" for i in range(rows)]
    client = make_spot_check_client(rows, array, passage_ids, shift=1)

    with pytest.raises(RuntimeError, match="esperado"):
        index.check_points(client, "c", array, passage_ids)


def test_spot_check_catches_a_wrong_count():
    array = np.eye(4, 4, dtype=np.float32)
    client = FakeClient(points=3)
    with pytest.raises(RuntimeError, match="esperados 4"):
        index.check_points(client, "c", array, ["a", "b", "c", "d"])


def test_footprint_separates_the_servers_fixed_preallocation_from_the_vectors(
    monkeypatch, tmp_path
):
    """O total sozinho vira conclusão sobre compressão que é taxa fixa do servidor.

    O Qdrant pré-aloca uma página de 32 MB de payload por segmento. Num braço
    float32 de 3 GB isso é ruído; num braço quantizado é a maior parte do número.
    """
    monkeypatch.setattr(server, "STORAGE_DIR", tmp_path)
    segment = tmp_path / "collections" / "quati_a0_float32" / "0" / "segments" / "abc"
    (segment / "vector_storage" / "vectors").mkdir(parents=True)
    (segment / "payload_storage").mkdir(parents=True)
    (tmp_path / "collections" / "quati_a0_float32" / "0" / "wal").mkdir(parents=True)
    # Esparsos: `st_size` reporta o tamanho lógico e o teste não escreve 38 MB.
    sizes = {
        segment / "vector_storage" / "vectors" / "chunk_0.mmap": 32 * 1024**2,
        segment / "payload_storage" / "page_0.dat": 4 * 1024**2,
        tmp_path / "collections" / "quati_a0_float32" / "0" / "wal" / "open-0": 2 * 1024**2,
    }
    for path, size in sizes.items():
        with path.open("wb") as handle:
            handle.truncate(size)
    (segment / "segment.json").write_bytes(b"{}")

    disk = index.footprint("quati_a0_float32")
    assert disk["total_bytes"] == sum(sizes.values()) + 2
    components = disk["by_component_mb"]
    assert components["vector_storage"] > components["payload_storage"] > 0
    assert components["wal"] > 0 and components["outros"] >= 0

    with pytest.raises(FileNotFoundError):
        index.footprint("quati_a1_int8")


REPEATABILITY = stats.MATRIX_PATH.parent.parent / "runs" / "m2_build_repeatability.json"


@pytest.mark.skipif(not REPEATABILITY.exists(), reason="repetição de build não registrada")
def test_footprint_repeatability_is_reported_as_a_range_not_a_number():
    """O critério de M2.1 não foi atendido, e o teste trava o fato em vez da esperança.

    A ocupação em disco varia até 18% entre builds idênticos: o conteúdo repete,
    mas a pré-alocação de chunks de 32 MB por segmento depende de como o
    otimizador distribuiu os pontos naquela execução.
    """
    import json

    record = json.loads(REPEATABILITY.read_text(encoding="utf-8"))
    deltas = record["footprint_repeatability"]["vector_storage_delta_pct"]
    assert max(abs(v) for v in deltas.values()) > 5, (
        "se a repetição virou apertada, o artigo pode reportar número em vez de faixa"
    )

    # A claim C8 só sobrevive porque as faixas não se sobrepõem.
    builds = record["builds"]
    a0 = [b["total_mb"] for b in builds["A0"]]
    a1 = [b["total_mb"] for b in builds["A1"]]
    assert max(a0) < min(a1), "C8 depende de o braço quantizado ocupar mais disco que o A0"


@pytest.mark.skipif(not REPEATABILITY.exists(), reason="repetição de build não registrada")
def test_product_quantization_optimizer_time_is_the_deterministic_one():
    """A evidência de que o tempo de PQ é dominado pelo treino do codebook.

    Determinismo é assinatura de trabalho fixo; variabilidade é assinatura do
    escalonamento do otimizador. Os braços de PQ repetem exato, todos os outros
    variam por um fator de 2 a 3.
    """
    import json

    builds = json.loads(REPEATABILITY.read_text(encoding="utf-8"))["builds"]
    spread = {arm: {b["index_minutes"] for b in runs} for arm, runs in builds.items()}
    for arm in ("A2", "A3"):
        assert len(spread[arm]) == 1, f"{arm} deixou de ser determinístico; reveja o limite"
    assert any(len(spread[arm]) > 1 for arm in ("A0", "A1", "A4"))
