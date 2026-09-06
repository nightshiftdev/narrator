# narrator

Reads a `.md`, `.pdf` or `.txt` file aloud like someone who has already read it —
not like a machine seeing each sentence for the first time.

```bash
narrate book.pdf                    # local Kokoro voice; starts talking in seconds, saves a chaptered .m4b
narrate notes.md -e kokoro          # local neural voice
narrate paper.pdf --dry-run         # see the performance direction, synthesise nothing
```

## Why it doesn't sound robotic

Most text-to-speech is flat because the model sees one sentence at a time and
has to guess the prosody from that sentence alone. A human narrator has read
the paragraph: they know the line is the quiet turn after an argument, so they
slow down and drop their voice.

The pipeline buys that context back.

```
file ─▶ extract ─▶ normalise ─▶ DIRECT ─▶ synthesise ─▶ master ─▶ play + save
```

**extract** — structure-aware. Headings, quotes, lists and code survive as
distinct kinds, because a heading should be read as an arrival and a block
quote should be read differently from the prose around it. PDFs get font-size
heading detection, hyphen-break repair, page-number stripping and duplicate
text-layer collapsing.

**normalise** — a large share of the robot feeling lives here, not in the
voice. `1969` becomes *nineteen sixty-nine*, not *one thousand nine hundred
and sixty-nine*. `$4.2M` becomes *four point two million dollars*. `Dr.`,
`e.g.`, `21st`, `9:30 a.m.`, `°C`, `£20,000`, bare URLs, and `(Smith et al.,
2019)` citation noise all get handled before the voice ever sees them.

**direct** — the part that makes it an audiobook. An LLM reads a rolling
window of sentences *with the surrounding context* and annotates each one:

```
  10 somber      0.80  0.90s  *forty-three years*   He spent forty-three years on the problem.
       sit with the sacrifice
  15 tense       0.85  1.20s  *refused*             The Board of Longitude refused to pay him.
       bitter injustice, let it land
```

Emotion, pace, which word carries the contrast, and how long the silence
afterwards should be. Direction is cached on disk, so re-rendering a book is
free. With `--no-direct` a structural heuristic pass runs instead: slower and
weightier for headings, longer breaths at paragraph ends, a beat after a
question. Not a performance, but not a monotone.

**synthesise** — whichever engine you picked, fed the direction in the form it
understands.

**master** — engine-added lead-in and tail silence is trimmed so *our* pauses
set the rhythm; clips get 8 ms ramps so concatenation never clicks; the whole
programme is loudness-normalised to −19 LUFS (audiobook convention) with peak
headroom.

## Where your text goes

Only one stage of the pipeline has any reason to touch the network, and you
choose whether it does.

| `--director` | judgement | your text |
|---|---|---|
| `cloud` *(default)* | best | **sent to Anthropic**, in batches with context |
| `local` | good | **never leaves the machine** — an Ollama model on localhost |
| `off` | heuristics only | never leaves the machine |

```bash
narrate ch1.pdf --director local                    # qwen2.5:14b by default
narrate ch1.pdf --director local --director-model mistral-nemo
narrate ch1.pdf --director off                      # no model at all
```

Synthesis is a separate question: `kokoro`, `chatterbox` and `macsay` all run
locally, so `--director local -e kokoro` is a fully offline pipeline — you can
pull the ethernet cable and it still works. `-e elevenlabs` sends your text to
ElevenLabs regardless of the director setting.

Direction is cached on disk at `~/.cache/narrator/direction/`, keyed by model,
so re-rendering a chapter never re-sends it.

## Engines

| engine | quality | speed | cost | direction it honours |
|---|---|---|---|---|
| `kokoro` | very good | ~8× realtime | free, offline | pace, pauses |
| `elevenlabs` | best | network-bound | ~$1–3 / hour | emotion, emphasis, pace, look-ahead context |
| `chatterbox` | good, clonable | ~1× realtime | free, offline | emotional intensity, pace |
| `macsay` | baseline | ~5× realtime | free, built in | pace, pitch, emphasis |

```bash
narrate x.md -e kokoro --list-voices
narrate x.md -e elevenlabs -v charlotte        # needs ELEVENLABS_API_KEY
narrate x.md -e chatterbox -v ~/my-voice.wav   # clone from ~10s of audio
```

`kokoro` and `macsay` work out of the box. The others are opt-in:

```bash
uv add chatterbox-tts          # ~2.5 GB, pulls PyTorch
export ELEVENLABS_API_KEY=sk_...
```

## Options

```
-e, --engine        kokoro | elevenlabs | chatterbox | macsay
-v, --voice         voice name (or reference clip path, for chatterbox)
-o, --out           output path; .m4b (chaptered) or .wav
    --no-play       render only
    --no-save       play only
    --director      cloud | local | off   (see "Where your text goes")
    --director-model  override the director model
    --no-direct     alias for --director off
    --speed         playback rate multiplier
    --from N        start at sentence N
    --limit N       read only N sentences — good for auditioning a voice
    --dry-run       print the direction and stop
    --read-code     read code blocks instead of skipping them
    --list-voices
```

## Notes

No Homebrew and no ffmpeg. Encoding uses macOS's own `afconvert`, playback
uses `afplay`, and espeak-ng arrives as a pip wheel.

Audio is rendered ahead of the ear on a worker thread, so playback starts
within a couple of seconds and never stalls, while the finished programme is
assembled in parallel and written to disk.

The director uses the `claude` CLI if it's on your PATH, or `ANTHROPIC_API_KEY`
if that's set.

## Install

```bash
uv sync && uv pip install -e .
```

Kokoro weights (~330 MB, once):

```bash
mkdir -p ~/.cache/narrator/kokoro && cd ~/.cache/narrator/kokoro
B=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
curl -fLO $B/kokoro-v1.0.onnx && curl -fLO $B/voices-v1.0.bin
```
