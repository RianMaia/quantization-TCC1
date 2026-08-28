"""Guardas do servidor Qdrant.

O primeiro teste não verifica o nosso código: verifica a **premissa** em que a
issue inteira se apoia — que o modo local aceita e ignora quantização em
silêncio. Se uma versão futura do cliente passar a recusar, a guarda vira
redundante e o teste avisa. Se continuar ignorando, ele documenta o motivo de
tanta cerimônia com podman.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient, models

from quantptbr import server

DIMS = 8
QUANTIZED = models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
)


def make_collection(client: QdrantClient, name: str, quantization=None) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=DIMS, distance=models.Distance.COSINE),
        quantization_config=quantization,
    )


def test_local_mode_silently_ignores_quantization():
    """A premissa de S1.3, medida em vez de citada."""
    client = QdrantClient(":memory:")
    make_collection(client, "local", QUANTIZED)
    assert client.get_collection("local").config.quantization_config is None, (
        "o modo local passou a respeitar quantização; reveja a necessidade da guarda"
    )


def test_guard_rejects_a_local_mode_client():
    with pytest.raises(RuntimeError, match="modo local"):
        server.require_server(QdrantClient(":memory:"))


@pytest.fixture(scope="module")
def client() -> QdrantClient:
    if not server.is_running():
        pytest.skip("servidor parado; rode `python scripts/qdrant_server.py start`")
    return QdrantClient(url=server.URL)


def test_guard_accepts_the_pinned_server(client):
    server.require_server(client)


def test_server_reports_quantization_back(client):
    """Uma coleção cuja configuração não pegou não é um braço: é o A0 com outro nome."""
    name = "quantptbr_probe_quantized"
    client.delete_collection(name)
    make_collection(client, name, QUANTIZED)
    try:
        server.require_quantization(client, name, "scalar")
    finally:
        client.delete_collection(name)


def test_liveness_check_fails_on_an_unquantized_collection(client):
    name = "quantptbr_probe_plain"
    client.delete_collection(name)
    make_collection(client, name, quantization=None)
    try:
        with pytest.raises(RuntimeError, match="não reporta quantização"):
            server.require_quantization(client, name, "scalar")
    finally:
        client.delete_collection(name)


def test_client_timeout_is_wide_enough_for_a_cold_collection():
    """Consulta lenta é dado; timeout no meio da execução é execução perdida.

    O padrão de 5 s do cliente estourou na primeira consulta sobre o A0 frio,
    paginando os segmentos do disco. A latência é cronometrada do lado do
    cliente, então o teto largo não interfere no que M3.3 mede.
    """
    assert server.TIMEOUT >= 300, "teto apertado demais para um braço frio de 1M"
