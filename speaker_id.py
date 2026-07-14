"""
speaker_id.py — Sprecher-Erkennung für die Haus-KI (House Mode).

Eigenständiges, vom Server entkoppeltes Modul. Wird NUR geladen/genutzt, wenn
HOUSE_SPEAKER_ID aktiv ist (siehe server.py). Hängt sich an die VAD-Segmente,
die der Server ohnehin produziert (float32 PCM @ 16 kHz), und liefert:

  • SpeakerEmbedder  — ONNX-Modell (3D-Speaker CAM++), audio_f32 -> 192-dim Vektor
  • SpeakerStore     — Fingerprint-Store (JSON): name -> {embedding, role, ...}
  • SpeakerIdentifier— Cosine-Similarity-Klassifikation + Kurzsegment-Regel

Bewusst KEINE Server-/Auth-/Memory-Logik hier — das ist reine Engine.
Das Fbank-Preprocessing ist Kaldi-kompatibel (80-dim, wie vom Modell erwartet).

Modell: 3D-Speaker CAM++ (zh_en multilingual advanced), 16 kHz, Input
[N, T, 80] Fbank, Output [N, 192] Embedding. feature_normalize_type=global-mean.
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- Defaults (werden von server.py via ENV überschrieben/übergeben) ---------
# Empirisch an echten Clips kalibriert (2026-06-19, Owner-Clip gegen Store):
#   Eigener Score des Owners: 3s=0.63, 5s=0.64, 22s=0.82 (kurz <1s nahe 0).
#   Fremdsprecher: 0.07-0.16. Das "Tal" liegt also ~0.2-0.6.
# Switch-Threshold MUSS erreichbar sein (0.8 war es nur bei ~20s Sprache ->
# nie ein current gesetzt -> Erkennung tot). 0.6 trennt sicher gegen Fremde.
DEFAULT_SIM_THRESHOLD = 0.5      # Bestaetigung des AKTUELLEN Sprechers (gleiche ID, klebt leichter)
DEFAULT_SWITCH_THRESHOLD = 0.6   # Wechsel zu einer NEUEN Identitaet (klar ueber Fremd-Max 0.16)
DEFAULT_MIN_DUR_S = 5.0          # Grenze: >= -> zuverlaessige Erkennung (gilt immer); < -> Sticky/Kleben
EMBED_DIM = 192
SAMPLE_RATE = 16000

# Kaldi-fbank-Parameter (müssen zum Trainings-Setup des Modells passen)
_FBANK_NUM_MEL = 80
_FBANK_FRAME_LENGTH_MS = 25.0
_FBANK_FRAME_SHIFT_MS = 10.0


# ============================================================================
# Fbank-Feature-Extraktion (Kaldi-kompatibel)
# ============================================================================
def compute_fbank(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Berechnet 80-dim Kaldi-style log-Mel-Fbank-Features.

    Nutzt kaldi-native-fbank, wenn verfügbar (exakt Kaldi-kompatibel);
    fällt sonst auf torchaudio.compliance.kaldi.fbank zurück.

    Rückgabe: np.float32 [T, 80].
    """
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    # Bevorzugt: kaldi-native-fbank (identisch zu Kaldi/3D-Speaker-Training)
    try:
        import kaldi_native_fbank as knf  # type: ignore

        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.samp_freq = float(sample_rate)
        opts.frame_opts.frame_length_ms = _FBANK_FRAME_LENGTH_MS
        opts.frame_opts.frame_shift_ms = _FBANK_FRAME_SHIFT_MS
        opts.mel_opts.num_bins = _FBANK_NUM_MEL
        opts.energy_floor = 0.0
        fbank = knf.OnlineFbank(opts)
        # Kaldi erwartet int16-Range-Samples (waveform * 32768)
        fbank.accept_waveform(float(sample_rate), (samples * 32768.0).tolist())
        fbank.input_finished()
        frames = [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
        if not frames:
            return np.zeros((0, _FBANK_NUM_MEL), dtype=np.float32)
        return np.asarray(frames, dtype=np.float32)
    except ImportError:
        pass

    # Fallback: torchaudio Kaldi-compliance
    import torch
    import torchaudio.compliance.kaldi as kaldi

    wav = torch.from_numpy(samples).unsqueeze(0) * 32768.0
    feats = kaldi.fbank(
        wav,
        num_mel_bins=_FBANK_NUM_MEL,
        frame_length=_FBANK_FRAME_LENGTH_MS,
        frame_shift=_FBANK_FRAME_SHIFT_MS,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=float(sample_rate),
    )
    return feats.numpy().astype(np.float32)


# ============================================================================
# Embedder (ONNX)
# ============================================================================
class SpeakerEmbedder:
    """Lädt das CAM++ ONNX-Modell lazy und berechnet 192-dim Embeddings."""

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self._sess = None
        self._lock = threading.Lock()

    def _ensure_session(self):
        if self._sess is None:
            with self._lock:
                if self._sess is None:
                    import onnxruntime as ort
                    so = ort.SessionOptions()
                    so.intra_op_num_threads = 1  # leichtgewichtig, CPU
                    self._sess = ort.InferenceSession(
                        self.model_path,
                        sess_options=so,
                        providers=["CPUExecutionProvider"],
                    )
        return self._sess

    def embed(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
        """audio_f32 [-1,1] -> L2-normalisiertes 192-dim Embedding (np.float32).

        Gibt None zurück, wenn das Audio zu kurz für brauchbare Features ist.
        """
        feats = compute_fbank(samples, sample_rate)
        if feats.shape[0] < 5:  # <~50ms Audio -> nutzlos
            return None
        # global-mean Normalisierung (laut Modell-Metadata)
        feats = feats - feats.mean(axis=0, keepdims=True)
        x = feats[np.newaxis, :, :].astype(np.float32)  # [1, T, 80]
        sess = self._ensure_session()
        emb = sess.run(["embedding"], {"x": x})[0][0]  # [192]
        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine-Similarity zweier (idealerweise L2-normalisierter) Vektoren."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================================
# Fingerprint-Store (JSON)
# ============================================================================
@dataclass
class Speaker:
    """Fingerprint eines Sprechers.

    Multi-Register (2026-06-17): statt EINES gemittelten Embeddings hält ein
    Sprecher jetzt eine LISTE von Register-Embeddings ("Stimmlagen-Wolke":
    normal/laut/leise/schnell/locker ...). Match = beste Similarity gegen
    IRGENDEINES der Register (siehe SpeakerIdentifier). Damit fällt Modulation/
    Emotion nicht mehr aus dem Fingerprint.

    Abwärtskompatibel: Stores im alten Format ({"embedding": [...]} als einzelner
    Vektor) werden beim Laden in eine 1-elementige Register-Liste migriert.
    Das Property `embedding` liefert weiterhin das gemittelte Gesamt-Embedding
    (für Alt-Code / rollendes Mittel).
    """
    name: str
    embeddings: list[np.ndarray]              # Register-Wolke (>=1 L2-norm. Vektoren)
    role: str = ""                            # Freitext-Rolle, z.B. "Admin"/"Vater"/"Freund"
                                              # (Legacy-Stores: "admin" | "guest")
    relation: str = ""                        # Legacy-Feld (Beziehung); neu wird alles in
                                              # `role` gepflegt, `relation` bleibt lesbar
    enrolled_at: float = field(default_factory=time.time)
    registers: list[str] = field(default_factory=list)  # optionale Labels je Register

    @property
    def n_samples(self) -> int:
        return len(self.embeddings)

    @property
    def embedding(self) -> np.ndarray:
        """Gemittelter Gesamt-Fingerprint (L2-normalisiert) über alle Register."""
        if not self.embeddings:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        return _l2(np.mean(np.stack(self.embeddings, axis=0), axis=0))

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "embeddings": [e.astype(np.float32).tolist() for e in self.embeddings],
            "registers": list(self.registers),
            "role": self.role,
            "relation": self.relation,
            "enrolled_at": self.enrolled_at,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Speaker":
        # Neues Format (Liste) bevorzugt, altes Format ({"embedding": vec}) migrieren.
        if "embeddings" in d and d["embeddings"]:
            embs = [_l2(np.asarray(e, dtype=np.float32)) for e in d["embeddings"]]
        elif d.get("embedding") is not None:
            embs = [_l2(np.asarray(d["embedding"], dtype=np.float32))]
        else:
            embs = []
        return cls(
            name=d["name"],
            embeddings=embs,
            registers=list(d.get("registers", [])),
            role=d.get("role", ""),
            relation=d.get("relation", ""),
            enrolled_at=d.get("enrolled_at", time.time()),
        )


class SpeakerStore:
    """Persistenter Fingerprint-Store: name -> Speaker. JSON auf Platte."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._speakers: dict[str, Speaker] = {}
        self._lock = threading.Lock()
        self.load()

    def load(self):
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except Exception:
            return
        with self._lock:
            self._speakers = {
                s["name"]: Speaker.from_json(s) for s in data.get("speakers", [])
            }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {"speakers": [s.to_json() for s in self._speakers.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self.path)

    def all(self) -> list[Speaker]:
        with self._lock:
            return list(self._speakers.values())

    def get(self, name: str) -> Speaker | None:
        with self._lock:
            return self._speakers.get(name)

    def upsert(self, sp: Speaker):
        with self._lock:
            self._speakers[sp.name] = sp
        self.save()

    def enroll_average(self, name: str, new_emb: np.ndarray, role: str | None = None):
        """Rollendes Mittel (Alt-Verhalten / Auto-Enroll): hängt das neue
        Embedding als zusätzliches Register an die Wolke des Sprechers an.
        Neu, falls unbekannt.

        Hinweis: Früher wurde gemittelt; mit Multi-Register sammeln wir die
        Register stattdessen einzeln (besser für Modulation). Der Name bleibt
        für Signatur-Kompatibilität."""
        self.add_register(name, new_emb, role=role)

    def add_register(
        self, name: str, new_emb: np.ndarray, role: str | None = None,
        label: str | None = None, relation: str | None = None,
    ):
        """Fügt ein einzelnes Register-Embedding zur Wolke des Sprechers hinzu
        (legt den Sprecher an, falls unbekannt)."""
        with self._lock:
            existing = self._speakers.get(name)
            if existing is None:
                self._speakers[name] = Speaker(
                    name=name, embeddings=[_l2(new_emb)], role=role or "",
                    relation=relation or "",
                    registers=[label] if label else [],
                )
            else:
                existing.embeddings.append(_l2(new_emb))
                if label:
                    existing.registers.append(label)
                if role:
                    existing.role = role
                if relation is not None:
                    existing.relation = relation
        self.save()

    def remove(self, name: str) -> bool:
        """Löscht einen Sprecher samt aller Register. True = existierte."""
        with self._lock:
            existed = self._speakers.pop(name, None) is not None
        if existed:
            self.save()
        return existed

    def rename(self, old: str, new: str) -> bool:
        """Benennt einen Sprecher um (Register/Rolle/Relation bleiben).
        False, wenn `old` fehlt oder `new` schon existiert."""
        with self._lock:
            sp = self._speakers.get(old)
            if sp is None or new in self._speakers:
                return False
            del self._speakers[old]
            sp.name = new
            self._speakers[new] = sp
        self.save()
        return True

    def set_role(self, name: str, role: str) -> bool:
        """Setzt die Freitext-Rolle eines Sprechers (z.B. "Admin", "Vater",
        "Freund"); leerer String löscht sie. Die Rolle ersetzt das Legacy-Feld
        `relation` als LLM-Tag-Qualifier — beim Setzen wird `relation` geleert,
        damit das Tag nicht doppelt qualifiziert ("Papa, Vater").
        False bei unbekanntem Sprecher."""
        with self._lock:
            sp = self._speakers.get(name)
            if sp is None:
                return False
            sp.role = role.strip()
            sp.relation = ""
        self.save()
        return True

    def remove_register(self, name: str, index: int) -> bool:
        """Entfernt EIN Register (Embedding + Label) eines Sprechers.
        False bei unbekanntem Sprecher oder Index außerhalb der Wolke."""
        with self._lock:
            sp = self._speakers.get(name)
            if sp is None or not (0 <= index < len(sp.embeddings)):
                return False
            del sp.embeddings[index]
            if index < len(sp.registers):
                del sp.registers[index]
        self.save()
        return True

    def rename_register(self, name: str, index: int, label: str) -> bool:
        """Benennt EIN Register (Label) eines Sprechers um; das Embedding
        bleibt unverändert. False bei unbekanntem Sprecher oder Index
        außerhalb der Wolke."""
        with self._lock:
            sp = self._speakers.get(name)
            if sp is None or not (0 <= index < len(sp.embeddings)):
                return False
            # Die Label-Liste darf kürzer sein als die Wolke (Labels sind
            # optional) — bis zum Index mit Leer-Labels auffüllen.
            while len(sp.registers) <= index:
                sp.registers.append("")
            sp.registers[index] = label
        self.save()
        return True

    def set_registers(
        self, name: str, embs: list[np.ndarray], role: str | None = None,
        labels: list[str] | None = None, relation: str | None = None,
    ):
        """Ersetzt die komplette Register-Wolke eines Sprechers (Bulk-Enroll).
        Legt den Sprecher an oder überschreibt seine Register vollständig."""
        embs = [_l2(e) for e in embs]
        with self._lock:
            existing = self._speakers.get(name)
            if existing is None:
                self._speakers[name] = Speaker(
                    name=name, embeddings=embs, role=role or "",
                    relation=relation or "",
                    registers=list(labels or []),
                )
            else:
                existing.embeddings = embs
                existing.registers = list(labels or [])
                if role:
                    existing.role = role
                if relation is not None:
                    existing.relation = relation
        self.save()


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ============================================================================
# Identifier (Klassifikation + Kurzsegment-Regel)
# ============================================================================
@dataclass
class IdentifyResult:
    name: str           # erkannter Name oder "unbekannt"
    role: str           # Freitext-Rolle des Sprechers ("" wenn keine; "unknown" bei Fremden)
    score: float        # beste Cosine-Similarity dieses Segments
    known: bool         # True = sicherer Match (>= Schwelle) ODER fortgeführt
    held: bool = False  # True = angenommen selber Sprecher (Fenster, kein sicherer Match)
    relation: str = ""  # optionale Beziehung (z.B. "Vater"/"Mutter"), leer wenn keine
    embedding: np.ndarray | None = None  # rohes Segment-Embedding (Auto-Enroll später)


class SpeakerIdentifier:
    """Klassifiziert VAD-Segmente gegen den Store — Sticky Speaker.

    Logik (Spec 2026-06-19) — "Default-to-Last-Known":

    Das System haelt einen aktiven Zustand `current_speaker` (Name/Rolle/
    Relation). Der Kontext "klebt" am letzten bekannten Sprecher, bis ein
    KLARER BRUCH kommt: eine andere Identitaet mit HOHER Konfidenz.

    Pro Segment (identify):
      1. Bestes Store-Match (best_name, best_score) bestimmen.
      2. WECHSEL nur, wenn best_name != current UND
         best_score >= switch_threshold (hoch, z.B. 0.80).
         -> current = best_name; known=True, held=False.
      3. BESTAETIGUNG des aktuellen Sprechers, wenn best_name == current UND
         best_score >= sim_threshold (normal). -> known=True, held=False.
      4. SONST (kurz, unsicher, geringe Konfidenz, Ueberlappung, Verzoegerung):
         -> KLEBEN am current_speaker (held=True), falls einer existiert.
         -> existiert noch keiner: "unbekannt" (kein Sprecher gesetzt).

    Damit ist die Zuordnung rein konfidenzbasiert und braucht KEINE Zeit-/
    Fenster-/playback.done-Logik mehr. Ueberschreiben des alten Sprechers
    passiert nur bei eindeutiger, hochkonfidenter Erkennung einer NEUEN Person.

    mark_playback_finished()/reset(): siehe unten (reset verwirft current).
    """

    def __init__(
        self,
        embedder: SpeakerEmbedder,
        store: SpeakerStore,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,      # Bestaetigung gleiche ID
        switch_threshold: float = DEFAULT_SWITCH_THRESHOLD,  # Wechsel zu NEUER ID (hoch)
        min_dur_s: float = DEFAULT_MIN_DUR_S,   # Dauergrenze lang/kurz (lang=verbindlich, kurz=Sticky)
        hold_s: float = 3.0,                    # nur noch Kompat (unbenutzt in Sticky-Logik)
        now_fn=None,
    ):
        self.embedder = embedder
        self.store = store
        self.sim_threshold = sim_threshold
        self.switch_threshold = switch_threshold
        self.min_dur_s = min_dur_s
        self.hold_s = hold_s
        self._now = now_fn or time.time
        # Aktiver Sprecher-Zustand (Sticky)
        self._cur_name: str | None = None
        self._cur_role: str = "unknown"
        self._cur_relation: str = ""

    # -- API-Kompat-Stubs (Zeitfenster-Logik entfernt) -----------------------
    def mark_playback_finished(self, now: float | None = None):
        """No-Op. Sticky-Logik braucht kein playback-Fenster mehr."""
        return None

    def reset(self):
        """Aktiven Sprecher verwerfen (z.B. Session-Reset)."""
        self._cur_name = None
        self._cur_role = "unknown"
        self._cur_relation = ""

    def _set_current(self, name: str):
        sp = self.store.get(name)
        self._cur_name = name
        self._cur_role = sp.role if sp else ""
        self._cur_relation = sp.relation if sp else ""

    # -- Hauptklassifikation (Sticky Speaker) --------------------------------
    def identify(
        self,
        samples: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        now: float | None = None,
        speech_start_ts: float | None = None,   # akzeptiert fuer Signatur-Kompat (unbenutzt)
        force_hold: bool = False,               # akzeptiert fuer Signatur-Kompat (unbenutzt)
    ) -> IdentifyResult:
        emb = self.embedder.embed(samples, sample_rate)
        dur_s = len(samples) / float(sample_rate) if sample_rate else 0.0

        # Bestes Store-Match (Multi-Register: beste Similarity gegen IRGENDEINES
        # der Register eines Sprechers zaehlt).
        best_name, best_score = None, -1.0
        if emb is not None:
            for sp in self.store.all():
                for reg in sp.embeddings:
                    s = cosine(emb, reg)
                    if s > best_score:
                        best_name, best_score = sp.name, s
        score = max(best_score, 0.0)

        # ====================================================================
        # LANGES Segment (dur >= min_dur_s): Das System erkennt den Sprecher
        # hier ZUVERLAESSIG. Ergebnis gilt IMMER verbindlich -> KEIN Kleben.
        # Sticky/Hold ist ausschliesslich fuer KURZE Ueberbrueckungssegmente
        # gedacht; lange Antworten sind davon ausgenommen (siehe Spec).
        # ====================================================================
        if dur_s >= self.min_dur_s:
            if best_name is not None and best_score >= self.sim_threshold:
                # Sicher erkannt -> dieser Sprecher gilt (setzt/wechselt current).
                if best_name != self._cur_name:
                    self._set_current(best_name)
                return IdentifyResult(
                    name=self._cur_name, role=self._cur_role, score=score,
                    known=True, held=False, relation=self._cur_relation, embedding=emb,
                )
            # Lang, aber kein Match (z.B. fremde/verstellte Stimme) -> ehrlich
            # unbekannt. current wird verworfen, damit nicht faelschlich der
            # alte Sprecher weiterklebt. Bewusst OHNE Zaehler: unbekannte
            # Sprecher sind nicht unterscheidbar, Hochzaehlen suggeriert nur
            # Scheinidentitaeten.
            self._cur_name = None
            self._cur_role = "unknown"
            self._cur_relation = ""
            return IdentifyResult(
                name="unbekannt", role="unknown",
                score=score, known=False, held=False, embedding=emb,
            )

        # ====================================================================
        # KURZES Segment (dur < min_dur_s): Embedding instabil -> Sticky.
        # Default-to-Last-Known: am aktuellen Sprecher kleben, AUSSER es gibt
        # einen hochkonfidenten klaren Bruch zu einer anderen Identitaet.
        # ====================================================================
        # Klarer Bruch: andere ID mit HOHER Konfidenz (selten bei kurz, aber
        # moeglich) -> wechseln.
        if (best_name is not None and best_name != self._cur_name
                and best_score >= self.switch_threshold):
            self._set_current(best_name)
            return IdentifyResult(
                name=self._cur_name, role=self._cur_role, score=score,
                known=True, held=False, relation=self._cur_relation, embedding=emb,
            )
        # Aktiver Sprecher vorhanden -> KLEBEN (das ist der Sinn der Regel).
        if self._cur_name is not None:
            return IdentifyResult(
                name=self._cur_name, role=self._cur_role, score=score,
                known=True, held=True, relation=self._cur_relation, embedding=emb,
            )
        # Kaltstart ohne aktiven Sprecher: kurzes Segment ist zu unsicher fuer
        # eine Identitaet -> unbekannt (erst ein langes Segment setzt current).
        return IdentifyResult(
            name="unbekannt", role="unknown",
            score=score, known=False, held=False, embedding=emb,
        )
