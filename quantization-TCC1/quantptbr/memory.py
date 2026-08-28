"""Leitura da memória residente sob o contrato de S1.1.

A métrica principal é `memory.current` do cgroup v2 do container do Qdrant —
`total` aqui — **num estado de residência declarado**, com `anon` e `file` ao
lado. O RSS é rejeitado: o Qdrant mapeia os vetores em disco, então o RSS conta
páginas *file-backed* e descartáveis cujo tamanho depende de quantas consultas
já rodaram.

**Emenda de 2026-08-18 (S1.7).** O contrato original nomeava `anon` como métrica
principal, supondo que page cache não é requisito de provisionamento. A medição
refutou a premissa: com 400 mil vetores float32 (1.172 MB), `file` marcou 1.172,8
MB e `anon` quase nada. Os segmentos são mapeados em disco mesmo com
`on_disk=False`, então a residência dos vetores mora no page cache. Reportar
`anon` subestimaria a métrica-título em cerca de três vezes.

Residência também não é um número só: ela depende do que já foi consultado. Daí
os três estados declarados — frio, morno sob HNSW, e exaustivo.

**Emenda de 2026-08-19 (M3.2).** O estado reportado é o **morno**, não o
exaustivo. A busca exata do Qdrant varre os vetores *originais* e ignora os
códigos, então o estado exaustivo pagina ~3 GB de originais em todo braço
comprimido e apaga a compressão sob medição — um caminho que nenhum deploy
comprimido executa. O `anon` volta a ser reportado com destaque, agora por um
motivo medido: é ele que rastreia os códigos fixados com `always_ram`, e é
independente da carga de consultas, enquanto o `file` depende inteiramente dela.

O processo do Qdrant vive num cgroup **folha** (`<scope>/container`), não no
scope. Ler o scope agregaria o que mais estiver pendurado nele.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from quantptbr import server

MIB = 1024**2


@dataclass(frozen=True)
class MemorySnapshot:
    """Uma linha de medição. Toda medida do estudo carrega uma destas."""

    anon_mb: float
    file_mb: float
    total_mb: float
    host_available_mb: float
    host_swap_used_mb: float
    cgroup: str
    taken_at: str

    def as_dict(self) -> dict:
        return asdict(self)


def swap_counters() -> tuple[int, int]:
    """Páginas que o **host** moveu para dentro e para fora do swap, desde o boot.

    Contexto, não portão: paginação de processos alheios não distorce uma leitura
    de cgroup. Ver `container_swap`, que é a escala que decide.
    """
    stat = _read_stat(Path("/proc/vmstat"))
    return stat["pswpin"], stat["pswpout"]


def container_swap() -> dict[str, int]:
    """Quanto o **container** foi paginado. É esta a escala que anula a medição.

    A leitura de memória vem do cgroup do container, então só a paginação *dele*
    pode distorcê-la. A primeira versão desta guarda lia `/proc/vmstat` e anulou
    três medições por atividade de fundo do host: o A5 foi anulado por 0,6 MB de
    paginação com 17 GB livres, enquanto o `memory.swap.peak` do próprio
    container marcava zero.

    `memory.swap.peak` é marca d'água desde o início do cgroup, e cada medição
    sobe um container novo — então ele cobre exatamente a janela, sem precisar de
    diferença entre dois instantes.
    """
    leaf = cgroup_path()
    stat = _read_stat(leaf / "memory.stat")
    peak = leaf / "memory.swap.peak"
    return {
        "swap_peak_bytes": int(peak.read_text()) if peak.exists() else 0,
        "swap_current_bytes": int((leaf / "memory.swap.current").read_text()),
        "zswapped_out_pages": stat.get("zswpout", 0),
    }


def quiesce(timeout: float = 180.0, still_for: float = 15.0, poll: float = 1.0) -> bool:
    """Espera o host parar de paginar antes de abrir a janela de medição.

    O build imediatamente anterior lê 3 GB e escreve 3,6, e o kernel segue
    trazendo páginas de volta do swap depois que ele termina. Não é o portão —
    esse é `container_swap` — mas medir sobre um host ainda em recuperação
    disputa I/O com a própria medição.

    Devolve False se não houver silêncio dentro do prazo, para quem chama decidir.
    """
    deadline = time.monotonic() + timeout
    last = swap_counters()
    quiet_since = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(poll)
        current = swap_counters()
        if current != last:
            last = current
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= still_for:
            return True
    return False


class SwapWatch:
    """Anula a medição se o **container** foi paginado durante a janela.

    Três regras foram tentadas aqui, e vale registrar por quê.

    A primeira testava `swap_used > 0 and available < 1024` no host: media volume
    acumulado, não paginação. Nesta máquina há 4 GB de swap residente com os
    contadores parados — rastro do passe de embedding — e a regra anularia
    medições válidas se a RAM apertasse, enquanto deixaria passar paginação ativa
    com RAM de sobra.

    A segunda passou a olhar a *taxa*, mas ainda no host, e anulou o A0 duas
    vezes e o A5 uma, sempre por frações de megabyte movidas por outros
    processos, com 17 GB livres. Escala errada: a leitura é do cgroup do
    container, então só a paginação dele pode distorcê-la.

    A terceira, esta, lê os contadores do próprio container.
    """

    def __enter__(self) -> Self:
        self._before = container_swap()
        self._after = self._before
        return self

    def __exit__(self, *_) -> None:
        self._after = container_swap()

    @property
    def zswapped_out_pages(self) -> int:
        return self._after["zswapped_out_pages"] - self._before["zswapped_out_pages"]

    @property
    def peak_bytes(self) -> int:
        return self._after["swap_peak_bytes"]

    @property
    def is_void(self) -> bool:
        return self.peak_bytes > 0 or self.zswapped_out_pages > 0

    def as_dict(self) -> dict:
        host_in, host_out = swap_counters()
        return {
            "container_swap_peak_bytes": self.peak_bytes,
            "container_zswapped_out_pages": self.zswapped_out_pages,
            "is_void": self.is_void,
            "_host_counters_are_context_only": {"pswpin": host_in, "pswpout": host_out},
        }


def cgroup_path() -> Path:
    """O cgroup folha que contém exatamente os processos do container."""
    scope = server.podman(
        "inspect", server.CONTAINER, "--format", "{{.State.CgroupPath}}"
    ).stdout.strip()
    if not scope:
        raise RuntimeError(f"{server.CONTAINER} não está de pé; não há cgroup para ler")
    leaf = Path("/sys/fs/cgroup") / scope.lstrip("/") / "container"
    if not (leaf / "memory.stat").exists():
        raise RuntimeError(f"{leaf}/memory.stat ausente; o contrato de S1.1 precisa de emenda")
    return leaf


def _read_stat(path: Path) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (line.split(maxsplit=1) for line in path.read_text().splitlines())
    }


def _host() -> tuple[float, float]:
    meminfo = {
        key.rstrip(":"): int(value.split()[0])
        for key, value in (
            line.split(maxsplit=1) for line in Path("/proc/meminfo").read_text().splitlines()
        )
    }
    available = meminfo["MemAvailable"] / 1024
    swap_used = (meminfo["SwapTotal"] - meminfo["SwapFree"]) / 1024
    return available, swap_used


def read() -> MemorySnapshot:
    """As três figuras de memória do container, mais o estado do host, numa chamada."""
    leaf = cgroup_path()
    stat = _read_stat(leaf / "memory.stat")
    total = int((leaf / "memory.current").read_text())
    available, swap_used = _host()
    return MemorySnapshot(
        anon_mb=round(stat["anon"] / MIB, 1),
        file_mb=round(stat["file"] / MIB, 1),
        total_mb=round(total / MIB, 1),
        host_available_mb=round(available, 1),
        host_swap_used_mb=round(swap_used, 1),
        cgroup=str(leaf),
        taken_at=datetime.now(UTC).isoformat(),
    )


def cross_check() -> dict:
    """Confere a leitura direta contra o `podman stats`.

    Divergência é explicada, nunca reconciliada por média: as duas fontes medem
    coisas ligeiramente diferentes, e a que vale é o cgroup.
    """
    snapshot = read()
    raw = server.podman("stats", "--no-stream", "--format", "json", server.CONTAINER).stdout
    # `mem_usage` vem formatado ("209MB / 25.02GB") e em unidades SI, enquanto o
    # cgroup é lido em bytes e convertido em MiB. A diferença de ~4% entre as
    # duas escalas é esperada e não é reconciliada — o cgroup é a fonte.
    used, _, _ = json.loads(raw)[0]["mem_usage"].partition("/")
    podman_mb = _parse_si(used.strip()) / MIB
    return {
        "cgroup_total_mb": snapshot.total_mb,
        "podman_stats_mb": round(podman_mb, 1),
        "difference_mb": round(snapshot.total_mb - podman_mb, 1),
        "_note": "podman reporta em unidades SI; o cgroup é convertido em MiB",
    }


def _parse_si(value: str) -> float:
    for suffix, factor in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3), ("B", 1.0)):
        if value.endswith(suffix):
            return float(value.removesuffix(suffix)) * factor
    raise ValueError(f"unidade não reconhecida em {value!r}")


def analytical_mb(passages: int, dims: int, quantization: dict | None) -> float:
    """A previsão, para ser impressa ao lado da medição — hipótese sob teste.

    Estimativas analíticas não substituem medição em lugar nenhum do estudo. Elas
    existem para que a diferença entre previsto e medido seja um resultado.
    """
    if quantization is None:
        return passages * dims * 4 / MIB
    kind = quantization["kind"]
    if kind == "scalar":
        return passages * dims * 1 / MIB
    if kind == "binary":
        return passages * dims / 8 / MIB
    if kind == "product":
        return passages * dims * 4 / int(quantization["compression"].lstrip("xX")) / MIB
    raise ValueError(f"quantização desconhecida: {kind}")


def rss_mb() -> float:
    """RSS do processo do Qdrant. **Nunca reportado como memória residente.**

    Existe só para que o artigo possa mostrar o número que teria sido reportado
    sob a definição rejeitada, e o quanto ele difere.
    """
    pid = subprocess.run(
        ["pgrep", "-x", "qdrant"], capture_output=True, text=True, check=False
    ).stdout.split()
    if not pid:
        raise RuntimeError("processo qdrant não encontrado")
    rollup = _read_stat_kb(Path(f"/proc/{pid[0]}/smaps_rollup"))
    return round(rollup["Rss"] / 1024, 1)


def _read_stat_kb(path: Path) -> dict[str, int]:
    values = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0])
    return values


def smoke_verdict(cases: list[dict], payload_mb: float) -> tuple[dict, dict]:
    """Separa validação de instrumento de caracterização de braço.

    A primeira execução misturou as duas, e as checagens falharam por codificarem
    uma expectativa que o dado refutou — não por o instrumento estar quebrado.
    Aqui só as **checagens** são portão. Elas perguntam se o contador resolve a
    carga e distingue configurações; nada mais.

    As **observações** são comparações entre configurações. Elas são registradas,
    nunca aprovadas ou reprovadas: 400 mil vetores aleatórios com cinco consultas
    exaustivas não caracterizam braço nenhum. Isso é trabalho de M3, a 1M, com os
    vetores reais e o conjunto de consultas real.
    """
    peak = [c["states"]["exhaustive"] for c in cases]
    cold = [c["states"]["cold"] for c in cases]
    float32_ram, float32_disk, int8_ram, int8_disk = peak

    checks = {
        # A prova: o page cache do container tem de refletir o volume de vetores.
        "o contador resolve a carga de vetores": abs(float32_ram["file_mb"] - payload_mb)
        < 0.2 * payload_mb,
        # Um contador que devolve o mesmo número para tudo não mede nada.
        "o contador distingue configurações": (
            max(p["total_mb"] for p in peak) - min(p["total_mb"] for p in peak) > 0.2 * payload_mb
        ),
        # Residência não é um número: depende do que já foi consultado.
        "o estado declarado importa": any(
            abs(p["total_mb"] - c["total_mb"]) > 0.1 * payload_mb
            for p, c in zip(peak, cold, strict=True)
        ),
        # Quantizado com always_ram é memória anônima; o original é page cache.
        "always_ram aparece em anon": int8_ram["anon_mb"] - float32_ram["anon_mb"]
        > 0.25 * payload_mb,
    }
    observations = {
        "float32_on_disk_muda_a_residencia_mb": round(
            float32_disk["total_mb"] - float32_ram["total_mb"], 1
        ),
        "int8_always_ram_muda_a_residencia_mb": round(
            int8_ram["total_mb"] - float32_ram["total_mb"], 1
        ),
        "int8_com_on_disk_muda_a_residencia_mb": round(
            int8_disk["total_mb"] - float32_ram["total_mb"], 1
        ),
        "_confirmar_em_M3": "vetores aleatórios a 400k não caracterizam braço; a 1M sim",
    }
    return checks, observations
