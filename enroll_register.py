#!/usr/bin/env python3
"""
enroll_register.py — Multi-Register-Enrollment für den House-Mode-SpeakerStore.

Liest mehrere WAV-Clips eines Sprechers (verschiedene Stimmlagen/Register:
normal/laut/leise/schnell/locker …), berechnet je Clip ein 192-dim CAM++-
Embedding und baut daraus die Register-Wolke des Sprechers im Fingerprint-Store
(speakers.json). Match später = beste Similarity gegen IRGENDEINES der Register
-> Modulation/Emotion fällt nicht mehr aus dem Fingerprint.

Beispiel:
  python enroll_register.py --name Alice --role admin \
      --store house_data/speakers.json \
      --model house_data/models/campplus_multilingual.onnx \
      alice/alice_*.wav

Standardverhalten:
  • --replace  (Default): ersetzt die komplette Register-Wolke des Sprechers
                neu aus den angegebenen Clips (idempotent, kein Drift).
  • --append :  hängt die Clips als zusätzliche Register an eine bestehende Wolke.

Akzeptiert auch nicht-16k/Mono-WAV (oder andere Formate), solange ffmpeg da ist:
nicht-konforme Dateien werden on-the-fly nach 16 kHz/Mono/PCM konvertiert.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

import speaker_id as spk

SAMPLE_RATE = spk.SAMPLE_RATE  # 16000


def _read_wav_f32(path: Path) -> np.ndarray:
    """Liest ein WAV als float32 [-1,1] Mono @ 16 kHz.

    Wenn das WAV bereits 16 kHz/Mono/16-bit PCM ist, direkt über `wave`.
    Sonst über ffmpeg konvertieren (falls verfügbar).
    """
    try:
        with wave.open(str(path), "rb") as w:
            ch = w.getnchannels()
            sr = w.getframerate()
            sw = w.getsampwidth()
            if ch == 1 and sr == SAMPLE_RATE and sw == 2:
                raw = w.readframes(w.getnframes())
                a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                return a
    except (wave.Error, EOFError):
        pass

    # Fallback: ffmpeg -> 16k mono s16le
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"{path}: nicht 16k/Mono/16-bit und ffmpeg nicht gefunden für Konvertierung."
        )
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg-Konvertierung fehlgeschlagen für {path}: "
                           f"{proc.stderr.decode('utf-8', 'replace')}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def _label_from_path(path: Path) -> str:
    """alice_laut.wav -> 'laut', alice_normal.wav -> 'normal', sonst Stem."""
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[1]
    return stem


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-Register-Enrollment für den SpeakerStore.")
    ap.add_argument("clips", nargs="+", help="WAV-Clips (verschiedene Register).")
    ap.add_argument("--name", required=True, help="Sprecher-Name (z.B. Alice).")
    ap.add_argument("--role", default=None,
                    help="Freitext-Rolle, z.B. 'Admin', 'Vater', 'Freund' "
                         "(erscheint im Sprecher-Tag; auch in der Web-UI editierbar).")
    ap.add_argument("--relation", default=None,
                    help="Legacy-Feld Beziehung, z.B. 'Vater' — neu besser --role verwenden.")
    here = Path(__file__).resolve().parent
    ap.add_argument("--store", default=str(here / "house_data" / "speakers.json"),
                    help="Pfad zum Fingerprint-Store (JSON).")
    ap.add_argument("--model", default=str(here / "house_data" / "models" / "campplus_multilingual.onnx"),
                    help="Pfad zum CAM++ ONNX-Modell.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--replace", action="store_true", default=True,
                      help="Register-Wolke komplett ersetzen (Default).")
    mode.add_argument("--append", dest="replace", action="store_false",
                      help="Clips an bestehende Wolke anhängen.")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"❌ Modell nicht gefunden: {model_path}", file=sys.stderr)
        return 2

    # Clips einlesen + einbetten
    clip_paths: list[Path] = []
    for pat in args.clips:
        p = Path(pat)
        if not p.is_file():
            print(f"❌ Clip nicht gefunden: {p}", file=sys.stderr)
            return 2
        clip_paths.append(p)
    clip_paths.sort()

    embedder = spk.SpeakerEmbedder(str(model_path))
    store = spk.SpeakerStore(str(args.store))

    embs: list[np.ndarray] = []
    labels: list[str] = []
    for p in clip_paths:
        samples = _read_wav_f32(p)
        dur = len(samples) / SAMPLE_RATE
        emb = embedder.embed(samples, SAMPLE_RATE)
        if emb is None:
            print(f"⚠️  {p.name}: zu kurz/leer für ein Embedding — übersprungen.", file=sys.stderr)
            continue
        label = _label_from_path(p)
        embs.append(emb)
        labels.append(label)
        print(f"  ✓ {p.name:24s} [{label:8s}] {dur:5.1f}s -> Embedding ok")

    if not embs:
        print("❌ Keine verwertbaren Clips — Abbruch.", file=sys.stderr)
        return 1

    # Optional: Sanity — Register untereinander sollten ähnlich sein (gleiche Stimme)
    if len(embs) > 1:
        sims = [spk.cosine(embs[i], embs[j])
                for i in range(len(embs)) for j in range(i + 1, len(embs))]
        print(f"\n  Register-Kohärenz (paarweise Cosine): "
              f"min={min(sims):.3f} mittel={float(np.mean(sims)):.3f} max={max(sims):.3f}")

    existing = store.get(args.name)
    if args.replace or existing is None:
        store.set_registers(args.name, embs, role=args.role, labels=labels,
                            relation=args.relation)
        action = "ersetzt" if existing else "neu angelegt"
    else:
        for e, l in zip(embs, labels):
            store.add_register(args.name, e, role=args.role, label=l,
                               relation=args.relation)
        action = "angehängt"

    sp = store.get(args.name)
    _rel = f", relation={sp.relation}" if sp.relation else ""
    print(f"\n✅ '{args.name}' {action}: {sp.n_samples} Register "
          f"[{', '.join(sp.registers) if sp.registers else '—'}], role={sp.role}{_rel}")
    print(f"   Store: {args.store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
