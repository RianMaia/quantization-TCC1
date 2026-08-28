#!/usr/bin/env python3
"""Ciclo de vida do servidor Qdrant fixado por digest.

    python scripts/qdrant_server.py start | stop | restart | fresh | status

`fresh` derruba o container e apaga o armazenamento — é o estado inicial que o
contrato de memória residente exige antes de cada medição.
"""

from __future__ import annotations

import sys

from quantptbr import server


def main(action: str) -> int:
    if action == "start":
        server.start()
    elif action == "stop":
        server.stop()
    elif action == "restart":
        server.restart()
    elif action == "fresh":
        server.restart(fresh_storage=True)
    elif action != "status":
        print(__doc__)
        return 2

    if action == "stop":
        print("servidor parado.")
        return 0

    if not server.is_running():
        print("servidor parado.")
        return 1
    info = server.wait_until_ready()
    print(f"qdrant {info['version']} em {server.URL}")
    print(f"imagem  {server.IMAGE_PINNED}")
    print(f"storage {server.STORAGE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "status"))
