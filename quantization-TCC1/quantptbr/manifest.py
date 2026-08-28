"""Manifesto de execução: a superfície de controle experimental, gravada pelo código.

Um manifesto escrito à mão ou depois do fato pode divergir da execução que
descreve, e aí não vale nada. Este é emitido por `RunManifest.capture()` no
momento em que a execução acontece, sem o chamador ter de lembrar de pedir.

A checagem de consistência existe pelo motivo oposto: duas execuções que diferem
em qualquer campo controlado **não podem** ser agregadas numa estatística. A
análise aborta em vez de avisar, porque um aviso num log é um aviso que ninguém
lê.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from quantptbr import corpus, embedding, memory, server, stats

#: Campos cuja divergência torna duas execuções incomparáveis. Não é a lista de
#: tudo que o manifesto grava — é a lista do que, se mudar, invalida a agregação.
CONTROLLED_FIELDS = (
    "envelope",
    "corpus_sha256",
    "qrels_sha256",
    "judged_query_ids",
    "server_version",
    "image_digest",
    "hnsw",
    "distance",
    "measurement_tools",
)

#: Bibliotecas que decidem um número reportado, e não apenas o transportam.
MEASUREMENT_TOOLS = ("ir-measures", "pytrec-eval-terrier", "qdrant-client")


def measurement_tools() -> dict[str, str]:
    """Versões das bibliotecas que calculam as métricas.

    O nDCG tem variantes defensáveis de desconto e de DCG ideal: uma atualização
    de biblioteca pode mudar o número sem que o run mude uma linha. Dois braços
    avaliados por versões diferentes não podem entrar na mesma estatística, e é
    por isso que isto é campo controlado e não apenas registro.
    """
    return {name: metadata.version(name) for name in MEASUREMENT_TOOLS}


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=corpus.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


#: Saídas de execução, versionadas de propósito. Sujeira aqui não diz nada sobre
#: o código que rodou: a própria execução escreve nestes caminhos antes de pedir
#: o manifesto, e contá-la faria toda execução bloquear a si mesma.
EVIDENCE_PREFIXES = ("runs/",)


def git_is_dirty() -> bool:
    """A árvore *de código* está suja? Evidência de execução não conta.

    O que a citabilidade exige é que o commit gravado descreva o código
    executado. Um `runs/A0_hnsw.jsonl` recém-escrito não muda esse código.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=corpus.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    changed = (line[3:].strip('"') for line in result.stdout.splitlines() if line[3:])
    return any(not path.startswith(EVIDENCE_PREFIXES) for path in changed)


def require_clean_tree(allow_dirty: bool = False) -> None:
    """Um commit gravado tem de descrever o código que de fato rodou.

    O override existe para desenvolvimento; medições reportadas no artigo não o
    usam, e o manifesto registra quando ele foi usado.
    """
    if git_is_dirty() and not allow_dirty:
        raise RuntimeError(
            "árvore git suja: o commit gravado não descreveria o código que rodou. "
            "Commite, ou passe allow_dirty=True e aceite que a execução não é citável."
        )


@dataclass(frozen=True)
class RunManifest:
    """Tudo que precisa ser igual para que duas linhas possam ser somadas."""

    run: dict
    controlled: dict
    host: dict
    extra: dict = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        *,
        arm_id: str | None = None,
        search_mode: str | None = None,
        repetition: int = 0,
        concurrency: int = 1,
        allow_dirty: bool = False,
        **extra,
    ) -> RunManifest:
        require_clean_tree(allow_dirty)
        manifest = corpus.read_manifest()
        matrix = stats.matrix()
        arm = next((a for a in matrix["arms"] if a["id"] == arm_id), None)

        controlled = {
            "envelope": json.loads(embedding.ENVELOPE_PATH.read_text(encoding="utf-8")),
            "corpus_sha256": manifest["files"]["corpus"]["sha256"],
            "qrels_sha256": manifest["files"]["qrels"]["sha256"],
            "judged_query_ids": corpus.judged_query_ids(),
            "server_version": server.SERVER_VERSION,
            "image_digest": server.DIGEST,
            "hnsw": matrix["held_constant"]["hnsw"],
            "distance": matrix["held_constant"]["distance"],
            "measurement_tools": measurement_tools(),
        }
        run = {
            "started_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "arm": arm_id,
            "arm_config": arm,
            "search_mode": search_mode,
            "repetition": repetition,
            "concurrency": concurrency,
            "seed": matrix["statistics"]["seed"],
        }
        snapshot = memory.read() if server.is_running() else None
        host = {
            "python": platform.python_version(),
            "kernel": platform.release(),
            "memory": snapshot.as_dict() if snapshot else None,
        }
        return cls(run=run, controlled=controlled, host=host, extra=extra)

    def as_dict(self) -> dict:
        return {"run": self.run, "controlled": self.controlled, "host": self.host, **self.extra}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return path


def require_same_envelope(manifests: list[dict]) -> None:
    """Aborta se execuções incomparáveis forem agregadas. Nunca avisa e continua.

    Somar nDCG de dois braços que consumiram vetores diferentes, ou que rodaram
    contra versões diferentes do servidor, produz um número que não corresponde a
    experimento nenhum — e que parece perfeitamente normal na tabela.
    """
    if not manifests:
        raise ValueError("nada a comparar")
    reference = manifests[0]
    for index, other in enumerate(manifests[1:], start=1):
        divergent = [
            key
            for key in CONTROLLED_FIELDS
            if reference["controlled"].get(key) != other["controlled"].get(key)
        ]
        if divergent:
            raise RuntimeError(
                f"execuções 0 e {index} divergem em {divergent}: não podem ser agregadas"
            )
