"""narrate — read a document aloud, like someone who has read it before."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.progress import (BarColumn, Progress as RichProgress, SpinnerColumn,
                           TextColumn, TimeRemainingColumn)

from . import audio as A
from . import extract
from .cast import (Cast, apply_overrides, assign_voices, attribute,
                   build_roster, ensure_cast, track_pov, write_review)
from .director import (LOCAL_DEFAULT, DirectorConfig, direct,
                       ollama_available, profile_for)
from .engines import (AVAILABLE, DIRECTION_PROFILE, EXPRESSIVENESS,
                       load_engine)
from .normalize import segment
from .pipeline import Renderer, collect

console = Console()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="narrate",
        description="Read a md / pdf / txt file aloud with an actual performance.",
    )
    p.add_argument("file", type=Path, help="document to read")
    p.add_argument("-e", "--engine", default="kokoro", choices=AVAILABLE,
                   help="voice backend (default: kokoro — local, free, offline)")
    p.add_argument("-v", "--voice", default=None, help="voice name within the engine")
    p.add_argument("--dialogue-voice", default=None,
                   help="second voice for spoken lines (kokoro); off by default")
    p.add_argument("--cast", action="store_true",
                   help="infer who speaks each line and give characters their "
                        "own voices (kokoro; adds a second model pass)")
    p.add_argument("--cast-file", type=Path, default=None,
                   help="TOML overriding the inferred casting (see --cast-review)")
    p.add_argument("--cast-review", type=Path, default=None,
                   help="write an editable cast file describing what was "
                        "inferred, then stop")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="write audio here (.m4b, .wav); default: alongside the source")
    p.add_argument("--no-play", action="store_true", help="render only, don't play")
    p.add_argument("--no-save", action="store_true", help="play only, don't write a file")
    p.add_argument("--director", default="cloud",
                   choices=["cloud", "local", "off"],
                   help="cloud: best judgement, your text is sent to Anthropic. "
                        "local: an Ollama model on this machine, nothing leaves. "
                        "off: heuristics only")
    p.add_argument("--no-direct", action="store_true",
                   help="alias for --director off")
    p.add_argument("--director-model", default=None,
                   help="override the director model")
    p.add_argument("--director-jobs", type=int, default=None,
                   help="batches to direct concurrently (default 4 local, 3 cloud)")
    p.add_argument("--director-profile", default="auto",
                   choices=["auto", "timing", "colour", "prosody", "full"],
                   help="how much direction to generate; auto matches the engine")
    p.add_argument("--speed", type=float, default=1.0, help="playback rate multiplier")
    p.add_argument("--from", dest="start", type=int, default=0,
                   help="start at sentence N")
    p.add_argument("--limit", type=int, default=None,
                   help="read at most N sentences (handy for auditioning)")
    p.add_argument("--read-code", action="store_true", help="read code blocks aloud")
    p.add_argument("--dry-run", action="store_true",
                   help="show the direction, synthesise nothing")
    p.add_argument("--list-voices", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_voices:
        engine = load_engine(args.engine)
        for v in engine.voices():
            console.print(f"  {v}")
        engine.close()
        return 0

    # ---------------------------------------------------------------- extract
    with console.status("[dim]reading document…"):
        doc = extract.load(args.file)
        sentences = segment(doc, read_code=args.read_code)

    if args.start:
        sentences = sentences[args.start:]
    if args.limit:
        sentences = sentences[: args.limit]
    if not sentences:
        console.print("[red]nothing to read[/]")
        return 1

    words = sum(len(s.text.split()) for s in sentences)
    console.print(f"[bold]{doc.title}[/]")
    console.print(f"[dim]{len(doc.blocks)} blocks · {len(sentences)} sentences · "
                  f"~{words:,} words · ~{words / 155:.0f} min[/]\n")

    # ----------------------------------------------------------------- direct
    provider = "off" if args.no_direct else args.director
    # Generating direction the engine cannot honour costs real time and
    # changes no audio, so ask only for what this voice can perform.
    profile = (DIRECTION_PROFILE.get(args.engine)
               or profile_for(EXPRESSIVENESS.get(args.engine, 3))
               ) if args.director_profile == "auto" else args.director_profile
    if provider == "local":
        cfg = DirectorConfig.local(args.director_model or LOCAL_DEFAULT,
                                   profile=profile)
        if args.director_jobs:
            cfg.jobs = args.director_jobs
        served = ollama_available()
        if not served:
            console.print("[red]no local model server found[/] — start it with "
                          "[bold]ollama serve[/], or use --director off")
            return 1
        if cfg.model not in served:
            console.print(f"[red]{cfg.model} is not pulled.[/] available: "
                          f"{', '.join(served)}")
            return 1
        console.print(f"[green]local director[/] · {cfg.model} · {cfg.jobs} jobs · "
                      f"{cfg.profile} profile · "
                      "[dim]nothing leaves this machine[/]")
    elif provider == "off":
        cfg = DirectorConfig(enabled=False)
    else:
        cfg = DirectorConfig(provider="cloud",
                             model=args.director_model or "claude-sonnet-5",
                             jobs=args.director_jobs or 3, profile=profile)
        console.print("[yellow]cloud director[/] · [dim]document text is sent to "
                      "Anthropic; use --director local to keep it here[/]")

    if cfg.enabled:
        live = console.is_terminal
        with RichProgress(SpinnerColumn(), TextColumn("[dim]directing"), BarColumn(),
                          TextColumn("{task.completed}/{task.total}"),
                          TimeRemainingColumn(), console=console,
                          transient=live, disable=not live) as bar:
            task = bar.add_task("direct", total=len(sentences))
            step = max(1, len(sentences) // 10)
            last = [0]

            def report(done: int, total: int) -> None:
                bar.update(task, completed=done)
                if not live and done - last[0] >= step:
                    last[0] = done
                    console.print(f"  directing {done}/{total}")

            direct(sentences, doc.title, cfg, on_progress=report)
    elif not args.cast_review:
        direct(sentences, doc.title, DirectorConfig(enabled=False))

    # ------------------------------------------------------------------ cast
    cast_obj = None
    pov_voices: dict[str, str] = {}
    if args.cast:
        from .engines import EXPRESSIVENESS
        with console.status("[dim]reading the cast…"):
            roster = build_roster(sentences, doc.title, cfg)
        if not roster:
            console.print("[yellow]no characters identified; single voice[/]")
        else:
            track_pov(sentences, roster)
            with RichProgress(SpinnerColumn(), TextColumn("[dim]attributing"),
                              BarColumn(), TextColumn("{task.completed}/{task.total}"),
                              console=console, transient=console.is_terminal,
                              disable=not console.is_terminal) as bar:
                t = bar.add_task("attr", total=len(sentences))
                attribute(sentences, roster, doc.title, cfg,
                          on_progress=lambda d, tot: bar.update(t, completed=d))
            from .engines import load_engine as _le
            probe = _le(args.engine, voice=args.voice)
            available = set(probe.voices())
            narrator_voice = getattr(probe, "voice", "")
            probe.close()
            pov_order: list[str] = []
            for x in sentences:
                if x.pov and x.pov not in pov_order:
                    pov_order.append(x.pov)
            cast_obj = assign_voices(roster, narrator_voice, available, pov_order)
            pov_voices = {c.name.upper(): (c.voice or narrator_voice)
                          for c in roster.values() if c.is_narrator}
            if args.cast_file and args.cast_file.exists():
                import tomllib
                conf = tomllib.loads(args.cast_file.read_text())
                for name, v in (conf.get("narrator") or {}).items():
                    pov_voices[name.upper()] = v
                for name, v in (conf.get("characters") or {}).items():
                    key = name.upper()
                    voice = v if isinstance(v, str) else v.get("voice", "")
                    if key not in cast_obj.characters:
                        from .cast import Character
                        cast_obj.characters[key] = Character(name=name)
                    cast_obj.characters[key].voice = voice
                flat = conf.get("flat") or []
                if isinstance(flat, list):
                    cast_obj.flat = {str(x).upper() for x in flat}
                    if cast_obj.flat:
                        console.print("  [dim]read flat (no emotional "
                                      f"colour): {', '.join(sorted(cast_obj.flat))}[/]")
                changed, warnings = apply_overrides(sentences, conf,
                                                    cast_obj.characters)
                if changed:
                    console.print(f"  [dim]{changed} line(s) reassigned from "
                                  f"{args.cast_file.name}[/]")
                # A book-wide cast file names lines absent from one chapter,
                # which is normal; show a few and count the rest.
                for w in warnings[:5]:
                    console.print(f"  [yellow]{escape(w)}[/]")
                if len(warnings) > 5:
                    console.print(f"  [dim]…and {len(warnings) - 5} more lines "
                                  f"in the cast file not present here[/]")
                for note in ensure_cast(sentences, cast_obj, available, cfg,
                                        doc.title):
                    console.print(f"  [green]cast from file:[/] {escape(note)}")
            console.print("[bold]cast[/]")
            for line in cast_obj.summary():
                console.print(f"  [dim]{line}[/]")
            unknown = [c.name for c in cast_obj.characters.values()
                       if not c.voice and not c.is_narrator]
            if unknown:
                console.print(
                    f"  [yellow]no voice (gender unknown), reading as the "
                    f"narrator: {', '.join(sorted(unknown))}[/]")
                console.print('  [dim]assign them under \[characters] in the '
                              'cast file to give them their own voice[/]')
            if args.cast_review:
                write_review(args.cast_review, sentences, cast_obj, pov_voices)
                console.print(f"[green]\u2713[/] wrote {args.cast_review}")
                return 0
            attributed = sum(1 for x in sentences if x.role == "speech" and x.speaker)
            total_speech = sum(1 for x in sentences if x.role == "speech")
            console.print(f"  [dim]{attributed}/{total_speech} spoken lines attributed[/]\n")

    if args.dry_run:
        # When casting, show the voice each line will actually be read in:
        # verifying that by ear over hours of audio is not reasonable.
        voice_of = None
        if cast_obj is not None:
            probe2 = load_engine(args.engine, voice=args.voice,
                                 cast=cast_obj, pov_voices=pov_voices or None)

            def voice_of(sent):
                return getattr(probe2, "_base_voice", lambda x: "")(sent)

        for s in sentences:
            emph = " ".join(f"*{w}*" for w in s.emphasis)
            who = f" [green]{s.speaker}[/]" if s.speaker else ""
            if voice_of is not None:
                v = voice_of(s)
                flat = " flat" if cast_obj.is_flat(s) else ""
                who += f" [blue]<{v}{flat}>[/]"
            console.print(
                f"[dim]{s.index:>4}[/]{who} [cyan]{s.emotion:<14}[/]"
                f"[magenta]{s.pace:.2f}[/] [yellow]{s.pause_after:.2f}s[/] "
                f"{emph:<24} {s.text[:78]}"
            )
            if s.note:
                console.print(f"      [dim italic]{s.note}[/]")
        return 0

    # ------------------------------------------------------------------ voice
    try:
        engine = load_engine(args.engine, voice=args.voice,
                             dialogue_voice=args.dialogue_voice,
                             cast=cast_obj, pov_voices=pov_voices or None)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        return 1
    console.print(f"[dim]engine {engine.name} · voice {getattr(engine, 'voice', '—')}[/]\n")

    # ----------------------------------------------------------------- render
    started = time.monotonic()
    live = console.is_terminal
    bar = RichProgress(SpinnerColumn(), TextColumn("[dim]{task.description}"),
                       BarColumn(), TextColumn("{task.completed}/{task.total}"),
                       TimeRemainingColumn(), console=console,
                       transient=live, disable=not live)
    task_id = bar.add_task("narrating", total=len(sentences))

    def on_progress(p, s):
        bar.update(task_id, completed=p.rendered,
                   description=f"narrating [{p.realtime_factor:.1f}× realtime]")

    renderer = Renderer(engine, sentences, on_progress=on_progress)
    with bar:
        renderer.start()
        try:
            result = collect(renderer, doc, play=not args.no_play, speed=args.speed)
        except KeyboardInterrupt:
            renderer.stop()
            console.print("\n[yellow]stopped[/]")
            return 130
        finally:
            engine.close()

    elapsed = time.monotonic() - started
    console.print(f"[green]✓[/] {result.duration / 60:.1f} min of audio "
                  f"in {elapsed:.0f}s ({result.duration / max(elapsed, 1e-9):.1f}× realtime)")

    # ------------------------------------------------------------------- save
    if not args.no_save:
        out = args.out or args.file.with_suffix(".m4b")
        wav = out.with_suffix(".wav")
        A.write_wav(wav, result.samples, result.rate)
        if out.suffix.lower() in (".m4b", ".m4a"):
            try:
                A.to_m4b(wav, out, chapters=result.chapters, title=doc.title)
                wav.unlink(missing_ok=True)
            except Exception as exc:
                console.print(f"[yellow]m4b encode failed ({exc}); kept {wav}[/]")
                out = wav
        else:
            out = wav
        console.print(f"[green]✓[/] {out}")
        if result.chapters:
            console.print(f"[dim]  {len(result.chapters)} chapter marks[/]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
