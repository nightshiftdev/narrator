# Licensing and dependencies

`narrator` itself is **MIT** (see `LICENSE`).

A default install pulls only permissive dependencies:

| package | licence |
|---|---|
| numpy | BSD-3-Clause |
| pypdfium2 | BSD-3-Clause / Apache-2.0 |
| soundfile | BSD-3-Clause |
| rich | MIT |

Verify at any time:

```bash
python - <<'PY'
import importlib.metadata as md
for d in md.distributions():
    t = ((d.metadata.get("License-Expression") or d.metadata.get("License") or "")
         + " " + " ".join(d.metadata.get_all("Classifier") or [])).upper()
    if "GPL" in t and "LGPL" not in t:
        print("copyleft:", d.metadata["Name"])
PY
```

## The optional Kokoro extra is GPL

`narrator[kokoro]` installs `kokoro-onnx`, which depends on `phonemizer`,
which drives **espeak-ng (GPL-3.0)**. Installing that extra makes *your
installation* GPL-3.0. The narrator source stays MIT, and nothing GPL is
distributed in this repository.

This is not avoidable by swapping the grapheme-to-phoneme engine. Kokoro's
own G2P (`misaki`) falls back to espeak for anything outside its lexicon, and
without that fallback it emits `❓` for ordinary proper nouns — character
names included. A voice that cannot pronounce names is not a usable narrator,
so the choice is espeak or nothing, and espeak is GPL.

Permissive alternatives, if you need an MIT-only install:

- `-e macsay` — macOS `say`, built in, no extra dependencies at all.
- `-e elevenlabs` — network API, no local G2P.

## Model weights

Kokoro-82M weights are Apache-2.0, downloaded at runtime and not
redistributed here.
