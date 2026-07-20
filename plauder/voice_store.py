"""Local voice library — cloned voices as files on disk.

The counterpart to ``voices.VoiceLibrary`` (which is HTTP CRUD against the
OmniVoice wrapper): here the samples live in ``voices/`` next to the project and
cloning happens in-process, so a voice is just a reference recording plus its
transcript.

Layout::

    voices/
      library.json          index (see below)
      <voiceId>/<sampleId>.wav

Index::

    {"active": "<voiceId>|default",
     "voices": [{"id","name","created","activeSample",
                 "samples":[{"id","name","refText","seconds","created"}]}]}

A voice keeps SEVERAL samples but clones from exactly ONE at a time
(``activeSample``) — OmniVoice takes a single reference recording, so mixing
samples is not a thing; you collect takes and pick the one that sounds best.

These samples are deliberately NOT the House-Mode speaker-ID samples: those
identify who is speaking (multi-register fingerprint, all registers count),
these reproduce a voice (one clean take wins).

Pure storage — no server state, no TTS calls. ``ai_voice`` drives it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

LOG = logging.getLogger("voice-chat")

DEFAULT_VOICE_ID = "default"


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class LocalVoiceStore:
    """File-backed voice library. All methods are sync and cheap (small JSON +
    file moves); the callers wrap the genuinely slow parts (STT, cloning)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._index_path = self.root / "library.json"
        self._data: dict | None = None

    # ---- index ------------------------------------------------------------
    def _load(self) -> dict:
        if self._data is not None:
            return self._data
        data = {"active": DEFAULT_VOICE_ID, "voices": []}
        try:
            with open(self._index_path, encoding="utf-8") as fh:
                stored = json.load(fh)
            if isinstance(stored.get("voices"), list):
                data["voices"] = stored["voices"]
            if isinstance(stored.get("active"), str):
                data["active"] = stored["active"]
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            LOG.warning("voice library index unreadable (%s) — starting empty", exc)
        self._data = data
        return data

    def _save(self) -> None:
        data = self._load()
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self._index_path)
        except Exception:
            LOG.exception("could not persist the voice library index")

    # ---- lookup -----------------------------------------------------------
    def _find(self, voice_id: str) -> dict | None:
        for v in self._load()["voices"]:
            if v["id"] == voice_id:
                return v
        return None

    def list(self) -> list[dict]:
        """Voices for the UI, built-in default first. ``isDefault`` marks the
        entry that falls back to the backend's configured reference."""
        out = [{"id": DEFAULT_VOICE_ID, "name": "", "isDefault": True,
                "samples": [], "activeSample": None}]
        for v in self._load()["voices"]:
            out.append({
                "id": v["id"], "name": v.get("name") or "",
                "isDefault": False, "created": v.get("created"),
                "activeSample": v.get("activeSample"),
                "samples": [dict(s) for s in v.get("samples", [])],
            })
        return out

    def get_active(self) -> str:
        active = self._load().get("active") or DEFAULT_VOICE_ID
        if active != DEFAULT_VOICE_ID and self._find(active) is None:
            return DEFAULT_VOICE_ID       # voice was deleted underneath us
        return active

    def set_active(self, voice_id: str) -> str:
        vid = voice_id or DEFAULT_VOICE_ID
        if vid != DEFAULT_VOICE_ID and self._find(vid) is None:
            vid = DEFAULT_VOICE_ID
        self._load()["active"] = vid
        self._save()
        return vid

    def sample_path(self, voice_id: str, sample: dict) -> Path:
        """Where a sample's audio lives. Adopted files (see :meth:`discover`)
        carry an explicit ``path`` and stay where they are; recorded/uploaded
        ones live under ``<root>/<voiceId>/<sampleId>.wav``."""
        if sample.get("path"):
            p = Path(sample["path"])
            return p if p.is_absolute() else (self.root / p)
        return self.root / voice_id / f"{sample['id']}.wav"

    def reference(self, voice_id: str | None = None) -> tuple[str, str] | None:
        """``(wav_path, ref_text)`` of the voice's active sample, or None for
        the built-in default / a voice without usable samples."""
        vid = voice_id or self.get_active()
        if vid == DEFAULT_VOICE_ID:
            return None
        v = self._find(vid)
        if not v or not v.get("samples"):
            return None
        sid = v.get("activeSample") or v["samples"][0]["id"]
        for s in v["samples"]:
            if s["id"] == sid:
                path = self.sample_path(vid, s)
                if path.exists():
                    return str(path), (s.get("refText") or "")
        return None

    def set_ref_text(self, voice_id: str, sample_id: str, ref_text: str) -> bool:
        """Fill in a transcript later (adopted files arrive without one)."""
        v = self._find(voice_id)
        if v is None:
            return False
        for s in v.get("samples", []):
            if s["id"] == sample_id:
                s["refText"] = ref_text or ""
                self._save()
                return True
        return False

    def discover(self) -> list[dict]:
        """Adopt loose ``*.wav`` files sitting directly in the library root.

        Hand-placed references (the ones OMNIVOICE_REF_AUDIO points at, e.g.
        ``voices/xena.wav``) are cloned voices too — they were just never
        registered. They become normal voices named after the file, with the
        file itself as their single sample, referenced IN PLACE so nothing is
        moved or copied. Idempotent: already-adopted paths are skipped.
        """
        if not self.root.is_dir():
            return []
        known = set()
        for v in self._load()["voices"]:
            for s in v.get("samples", []):
                if s.get("path"):
                    known.add(str(self.sample_path(v["id"], s).resolve()))
        added: list[dict] = []
        for wav in sorted(self.root.glob("*.wav")):
            if str(wav.resolve()) in known:
                continue
            voice = self.create_voice(wav.stem)
            sid = _new_id("s")
            # refText stays empty — it is transcribed lazily on first use, so
            # adoption never blocks startup on the STT model.
            sample = {"id": sid, "name": wav.name, "refText": "",
                      "path": str(wav), "seconds": 0.0, "created": time.time(),
                      "adopted": True}
            voice["samples"].append(sample)
            voice["activeSample"] = sid
            added.append(voice)
        if added:
            self._save()
        return added

    def find_by_path(self, path: str) -> dict | None:
        """The voice whose active sample is this file (used to match the
        configured OMNIVOICE_REF_AUDIO against the adopted voices)."""
        try:
            target = Path(path).resolve()
        except Exception:  # noqa: BLE001
            return None
        for v in self._load()["voices"]:
            for s in v.get("samples", []):
                try:
                    if self.sample_path(v["id"], s).resolve() == target:
                        return v
                except Exception:  # noqa: BLE001
                    continue
        return None

    # ---- mutation ---------------------------------------------------------
    def create_voice(self, name: str) -> dict:
        voice = {"id": _new_id("voice"), "name": (name or "").strip(),
                 "created": time.time(), "activeSample": None, "samples": []}
        self._load()["voices"].append(voice)
        self._save()
        return voice

    def rename_voice(self, voice_id: str, name: str) -> bool:
        v = self._find(voice_id)
        if v is None:
            return False
        v["name"] = (name or "").strip()
        self._save()
        return True

    def delete_voice(self, voice_id: str) -> bool:
        data = self._load()
        before = len(data["voices"])
        data["voices"] = [v for v in data["voices"] if v["id"] != voice_id]
        if len(data["voices"]) == before:
            return False
        shutil.rmtree(self.root / voice_id, ignore_errors=True)
        if data.get("active") == voice_id:
            data["active"] = DEFAULT_VOICE_ID
        self._save()
        return True

    def add_sample(self, voice_id: str, wav: bytes, *, ref_text: str,
                   name: str = "", seconds: float = 0.0) -> dict | None:
        """Stores a WAV as a new sample. The first sample of a voice becomes
        its active reference automatically."""
        v = self._find(voice_id)
        if v is None:
            return None
        sid = _new_id("s")
        vdir = self.root / voice_id
        vdir.mkdir(parents=True, exist_ok=True)
        try:
            (vdir / f"{sid}.wav").write_bytes(wav)
        except Exception:
            LOG.exception("could not write voice sample %s/%s", voice_id, sid)
            return None
        sample = {"id": sid, "name": (name or "").strip(), "refText": ref_text or "",
                  "seconds": round(float(seconds), 2), "created": time.time()}
        v.setdefault("samples", []).append(sample)
        if not v.get("activeSample"):
            v["activeSample"] = sid
        self._save()
        return sample

    def set_active_sample(self, voice_id: str, sample_id: str) -> bool:
        v = self._find(voice_id)
        if v is None or not any(s["id"] == sample_id for s in v.get("samples", [])):
            return False
        v["activeSample"] = sample_id
        self._save()
        return True

    def rename_sample(self, voice_id: str, sample_id: str, name: str) -> bool:
        v = self._find(voice_id)
        if v is None:
            return False
        for s in v.get("samples", []):
            if s["id"] == sample_id:
                s["name"] = (name or "").strip()
                self._save()
                return True
        return False

    def delete_sample(self, voice_id: str, sample_id: str) -> bool:
        v = self._find(voice_id)
        if v is None:
            return False
        gone = next((s for s in v.get("samples", []) if s["id"] == sample_id), None)
        if gone is None:
            return False
        keep = [s for s in v.get("samples", []) if s["id"] != sample_id]
        v["samples"] = keep
        # Adopted files belong to the user, not to us — drop the entry, keep
        # the file. Only recordings we created are deleted from disk.
        if not gone.get("adopted"):
            self.sample_path(voice_id, gone).unlink(missing_ok=True)
        if v.get("activeSample") == sample_id:
            # Never leave a voice pointing at a deleted reference.
            v["activeSample"] = keep[0]["id"] if keep else None
        self._save()
        return True
