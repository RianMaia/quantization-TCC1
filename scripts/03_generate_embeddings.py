#!/usr/bin/env python3
"""S1.5 — Passe de embedding sobre 1M e publicação do artefato de vetores.

É a única etapa cara e irreversível do estudo. Todos os seis braços leem estes
vetores, então um defeito aqui contamina os seis de forma **idêntica** — o pior
tipo, porque as comparações entre braços continuariam internamente coerentes e
nada pareceria errado.

Daí a forma do script: envelope lido do disco em vez de redigitado, ordem do
corpus preservada por construção, checkpoint a cada bloco fixo, renormalização
na escrita, e uma bateria de verificações no fim que falha alto em vez de
degradar em silêncio.

Retomada: `python scripts/03_generate_embeddings.py` de novo. O passe recomeça no
último bloco completo. Como os blocos são faixas fixas e a composição dos lotes
depende só do conteúdo do bloco, o resultado é idêntico ao de um passe único.
"""

from __future__ import annotations

from quantptbr import corpus, embedding, vectors


def main() -> int:
    corpus.verify_manifest()
    envelope = embedding.Envelope.load()
    print(
        f"envelope: {envelope.model_id} @ {envelope.model_revision[:8]} "
        f"({envelope.dimensions}d, {envelope.dtype}, max_length {envelope.max_length})"
    )

    timing = vectors.run_pass(envelope)
    if timing["encoded"]:
        rate = timing["encoded"] / timing["seconds"]
        print(
            f"\n{timing['encoded']:,} passagens em {timing['seconds'] / 60:.1f} min "
            f"({rate:.1f} pass/s)"
        )

    vectors.verify(envelope)
    print("\ncalculando sha256 do artefato …")
    vectors.write_manifest(envelope, timing)
    print(f"manifesto → {vectors.MANIFEST_PATH}")
    print("\nartefato de vetores publicado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
