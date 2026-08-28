"""Ciclo de vida do servidor Qdrant e as guardas que impedem medir o nada.

`qdrant_client.local.local_collection.LocalCollection` fixa
`quantization_config=None`. Em modo local — `:memory:` ou um caminho — uma
configuração de quantização é **aceita sem erro e ignorada em silêncio**. Um
estudo rodado assim construiria seis coleções float32 idênticas, produziria seis
nDCG idênticos e uma tabela impecável que não mede nada.

Por isso as duas guardas deste módulo não são opcionais: `require_server` antes
de qualquer medição, `require_quantization` depois de qualquer criação de
coleção quantizada. Falha de configuração silenciosa é o maior risco do projeto.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request

from quantptbr.corpus import REPO_ROOT

#: Fixado por digest, não por tag. Tags se movem; o digest é o artefato.
IMAGE = "docker.io/qdrant/qdrant"
DIGEST = "sha256:057ee3a8da769fe7310dd3537b4dc7583bf87a95ce8ac43c0af5a46bc580d1fc"
IMAGE_PINNED = f"{IMAGE}@{DIGEST}"
SERVER_VERSION = "1.19.0"

CONTAINER = "quantptbr-qdrant"
HOST = "127.0.0.1"
HTTP_PORT = 6333
GRPC_PORT = 6334
URL = f"http://{HOST}:{HTTP_PORT}"

#: Fora de `data/`: o armazenamento do servidor é estado do container, não um
#: artefato congelado do estudo, e não deve ser confundido com um.
STORAGE_DIR = REPO_ROOT / "qdrant_storage"


def podman(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["podman", *args], capture_output=True, text=True, check=check)


def is_running() -> bool:
    result = podman("ps", "--filter", f"name={CONTAINER}", "--format", "{{.Names}}", check=False)
    return CONTAINER in result.stdout.split()


#: O servidor relê os segmentos ao subir, e a medição reinicia o container com
#: a coleção já construída. Medido em 2026-08-26: 60 s não bastaram para uma
#: coleção grande, e a varredura morreu entre dois braços. Uma partida lenta é
#: um dado sobre o braço; um timeout no meio dela só apaga a medição.
READY_TIMEOUT = 300.0


def start(wait: float = READY_TIMEOUT) -> None:
    """Sobe o servidor se ainda não estiver de pé. Idempotente."""
    if is_running():
        return
    podman("rm", "-f", CONTAINER, check=False)
    STORAGE_DIR.mkdir(exist_ok=True)
    podman(
        "run",
        "-d",
        "--name",
        CONTAINER,
        "-p",
        f"{HOST}:{HTTP_PORT}:6333",
        "-p",
        f"{HOST}:{GRPC_PORT}:6334",
        # :Z reetiqueta para SELinux, que o Nobara aplica; sem isso o container
        # sobe e falha ao escrever, o que parece um erro do Qdrant.
        "-v",
        f"{STORAGE_DIR}:/qdrant/storage:Z",
        IMAGE_PINNED,
    )
    wait_until_ready(wait)


def stop(remove_storage: bool = False) -> None:
    podman("rm", "-f", CONTAINER, check=False)
    if remove_storage and STORAGE_DIR.exists():
        subprocess.run(["rm", "-rf", str(STORAGE_DIR)], check=True)


def restart(fresh_storage: bool = False) -> None:
    """Container novo por medição, como o contrato de memória residente exige."""
    stop(remove_storage=fresh_storage)
    start()


def wait_until_ready(timeout: float = READY_TIMEOUT) -> dict:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/", timeout=2) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.5)
    raise TimeoutError(f"servidor não respondeu em {timeout}s: {last}")


#: Teto por requisição, bem acima do padrão de 5 s do cliente. Este é um
#: aparelho de medição: uma consulta lenta é um dado, um timeout no meio da
#: execução é uma execução perdida. Medido em 2026-08-19, a primeira consulta
#: sobre o A0 frio estourou os 5 s paginando os segmentos, enquanto as
#: seguintes levaram de 0,02 a 2,7 s. A latência é cronometrada do lado do
#: cliente, então alargar o teto não interfere no que M3.3 mede.
TIMEOUT = 600.0


def client(prefer_grpc: bool = True, timeout: float = TIMEOUT):
    """Cliente apontado para o servidor fixado, em gRPC por padrão.

    O REST rejeita payloads acima de 32 MB, e um lote de 5.000 vetores de 768
    dimensões vira 81 MB de JSON. Em 1M de pontos isso não é uma otimização —
    é a diferença entre a inserção funcionar e não funcionar.
    """
    from qdrant_client import QdrantClient

    return QdrantClient(
        host=HOST, port=HTTP_PORT, grpc_port=GRPC_PORT, prefer_grpc=prefer_grpc, timeout=timeout
    )


def require_server(client) -> None:
    """Precondição de toda medição. Aborta antes de produzir número inválido."""
    if getattr(client, "_client", None).__class__.__name__ == "QdrantLocal":
        raise RuntimeError(
            "cliente em modo local: quantização seria aceita e ignorada em silêncio. "
            "Suba o servidor com `python scripts/qdrant_server.py start`."
        )
    version = wait_until_ready()["version"]
    if version != SERVER_VERSION:
        raise RuntimeError(f"servidor {version} != {SERVER_VERSION} fixado pelo digest")


def require_quantization(client, collection: str, expected: str) -> None:
    """Lê a configuração de volta do servidor. Config aplicada ou build falhou.

    `expected` é o nome do campo em `quantization_config`: "scalar", "product"
    ou "binary". Uma coleção cuja configuração não pegou não é um braço — é o
    A0 com outro nome, e comparar os dois mediria zero.
    """
    config = client.get_collection(collection).config.quantization_config
    if config is None:
        raise RuntimeError(
            f"{collection}: servidor não reporta quantização. "
            "A configuração não foi aplicada; a coleção é float32 disfarçada."
        )
    actual = type(config).__name__.lower()
    if expected not in actual:
        raise RuntimeError(f"{collection}: quantização {actual}, esperada {expected}")
