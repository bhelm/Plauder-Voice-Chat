"""Local voice library (plauder.voice_store): voices, their samples, and which
sample is THE reference to clone from. Pure file I/O — no server, no TTS."""

import json

import pytest

from plauder.voice_store import DEFAULT_VOICE_ID, LocalVoiceStore

WAV = b"RIFF....WAVEfake"


@pytest.fixture
def store(tmp_path):
    return LocalVoiceStore(tmp_path)


def test_empty_store_lists_only_the_builtin_default(store):
    lst = store.list()
    assert [v["id"] for v in lst] == [DEFAULT_VOICE_ID]
    assert store.get_active() == DEFAULT_VOICE_ID
    assert store.reference() is None


def test_first_sample_becomes_the_reference(store):
    v = store.create_voice("Xena")
    s1 = store.add_sample(v["id"], WAV, ref_text="hallo welt", seconds=3.0)
    assert store._find(v["id"])["activeSample"] == s1["id"]
    s2 = store.add_sample(v["id"], WAV, ref_text="zweite aufnahme", seconds=4.0)
    # A later take does NOT steal the reference — that is an explicit choice.
    assert store._find(v["id"])["activeSample"] == s1["id"]
    assert store.set_active_sample(v["id"], s2["id"]) is True
    assert store._find(v["id"])["activeSample"] == s2["id"]


def test_reference_returns_path_and_transcript_of_active_sample(store, tmp_path):
    v = store.create_voice("Xena")
    store.add_sample(v["id"], WAV, ref_text="erste", seconds=3.0)
    s2 = store.add_sample(v["id"], WAV, ref_text="zweite", seconds=3.0)
    store.set_active_sample(v["id"], s2["id"])
    store.set_active(v["id"])
    path, ref_text = store.reference()
    assert ref_text == "zweite"
    assert path == str(tmp_path / v["id"] / f"{s2['id']}.wav")
    assert (tmp_path / v["id"] / f"{s2['id']}.wav").read_bytes() == WAV


def test_reference_is_none_for_default_and_for_a_voice_without_samples(store):
    v = store.create_voice("Leer")
    store.set_active(v["id"])
    assert store.reference() is None          # voice exists but has no take
    store.set_active(DEFAULT_VOICE_ID)
    assert store.reference() is None


def test_deleting_the_reference_sample_promotes_another(store):
    v = store.create_voice("Xena")
    s1 = store.add_sample(v["id"], WAV, ref_text="a", seconds=1.0)
    s2 = store.add_sample(v["id"], WAV, ref_text="b", seconds=1.0)
    assert store.delete_sample(v["id"], s1["id"]) is True
    # Never leave a voice pointing at a deleted reference.
    assert store._find(v["id"])["activeSample"] == s2["id"]
    assert store.reference(v["id"])[1] == "b"


def test_deleting_the_last_sample_clears_the_reference(store):
    v = store.create_voice("Xena")
    s1 = store.add_sample(v["id"], WAV, ref_text="a", seconds=1.0)
    store.delete_sample(v["id"], s1["id"])
    assert store._find(v["id"])["activeSample"] is None
    assert store.reference(v["id"]) is None


def test_deleting_the_active_voice_falls_back_to_default(store, tmp_path):
    v = store.create_voice("Xena")
    store.add_sample(v["id"], WAV, ref_text="a", seconds=1.0)
    store.set_active(v["id"])
    assert store.delete_voice(v["id"]) is True
    assert store.get_active() == DEFAULT_VOICE_ID
    assert not (tmp_path / v["id"]).exists()   # sample files go too


def test_active_id_survives_a_reload(tmp_path):
    a = LocalVoiceStore(tmp_path)
    v = a.create_voice("Xena")
    a.add_sample(v["id"], WAV, ref_text="a", seconds=1.0)
    a.set_active(v["id"])
    b = LocalVoiceStore(tmp_path)               # fresh instance, same directory
    assert b.get_active() == v["id"]
    assert b.reference()[1] == "a"


def test_active_pointing_at_a_vanished_voice_reads_as_default(tmp_path):
    """Hand-edited/партially restored index: get_active must not hand out an id
    that no longer resolves."""
    (tmp_path).mkdir(exist_ok=True)
    (tmp_path / "library.json").write_text(
        json.dumps({"active": "voice-gone", "voices": []}), encoding="utf-8")
    assert LocalVoiceStore(tmp_path).get_active() == DEFAULT_VOICE_ID


def test_unknown_voice_operations_report_failure(store):
    assert store.add_sample("nope", WAV, ref_text="x") is None
    assert store.rename_voice("nope", "x") is False
    assert store.delete_voice("nope") is False
    assert store.set_active_sample("nope", "s") is False
    assert store.delete_sample("nope", "s") is False


def test_corrupt_index_starts_empty_instead_of_raising(tmp_path):
    (tmp_path / "library.json").write_text("{not json", encoding="utf-8")
    store = LocalVoiceStore(tmp_path)
    assert [v["id"] for v in store.list()] == [DEFAULT_VOICE_ID]


# --- adopting hand-placed reference recordings -------------------------------
def test_discover_adopts_loose_wavs_as_voices(tmp_path):
    (tmp_path / "xena.wav").write_bytes(WAV)
    (tmp_path / "joy.wav").write_bytes(WAV)
    store = LocalVoiceStore(tmp_path)
    added = store.discover()
    assert {v["name"] for v in added} == {"xena", "joy"}
    names = {v["name"] for v in store.list() if not v["isDefault"]}
    assert names == {"xena", "joy"}
    # Referenced in place — nothing is moved or copied.
    assert (tmp_path / "xena.wav").exists()
    v = next(v for v in store._load()["voices"] if v["name"] == "xena")
    assert store.reference(v["id"])[0] == str(tmp_path / "xena.wav")


def test_discover_is_idempotent(tmp_path):
    (tmp_path / "xena.wav").write_bytes(WAV)
    store = LocalVoiceStore(tmp_path)
    assert len(store.discover()) == 1
    assert store.discover() == []          # second boot adds nothing
    assert len([v for v in store.list() if not v["isDefault"]]) == 1


def test_discover_ignores_recorded_sample_dirs(tmp_path):
    """Recorded samples live in <root>/<voiceId>/ — only loose top-level files
    are adoption candidates, or every recording would spawn a second voice."""
    store = LocalVoiceStore(tmp_path)
    v = store.create_voice("Recorded")
    store.add_sample(v["id"], WAV, ref_text="a", seconds=1.0)
    assert store.discover() == []


def test_find_by_path_matches_the_configured_reference(tmp_path):
    (tmp_path / "xena.wav").write_bytes(WAV)
    store = LocalVoiceStore(tmp_path)
    store.discover()
    assert store.find_by_path(str(tmp_path / "xena.wav"))["name"] == "xena"
    assert store.find_by_path(str(tmp_path / "nope.wav")) is None


def test_deleting_an_adopted_sample_keeps_the_users_file(tmp_path):
    (tmp_path / "xena.wav").write_bytes(WAV)
    store = LocalVoiceStore(tmp_path)
    store.discover()
    v = next(v for v in store._load()["voices"] if v["name"] == "xena")
    sid = v["activeSample"]
    assert store.delete_sample(v["id"], sid) is True
    # The entry is gone, the file is NOT ours to delete.
    assert (tmp_path / "xena.wav").exists()


def test_ref_text_can_be_filled_in_later(tmp_path):
    (tmp_path / "xena.wav").write_bytes(WAV)
    store = LocalVoiceStore(tmp_path)
    store.discover()
    v = next(v for v in store._load()["voices"] if v["name"] == "xena")
    assert store.reference(v["id"])[1] == ""      # adopted without transcript
    assert store.set_ref_text(v["id"], v["activeSample"], "hallo welt") is True
    assert store.reference(v["id"])[1] == "hallo welt"
    assert LocalVoiceStore(tmp_path).reference(v["id"])[1] == "hallo welt"
