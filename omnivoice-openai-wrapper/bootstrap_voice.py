#!/usr/bin/env python
"""Bootstrap: generate voice candidates via OmniVoice voice-design (valid
instruct vocabulary only), so you can freeze one as a fixed clone reference.

Run once on the GPU box, listen to ref/cand_*.wav, pick the one you like, then:
  cp ref/cand_2.wav ref/ref.wav
  # put the EXACT spoken text below into ref/ref.txt
and (re)start the server. Prefer a clean, ~10-20 s reference for best cloning.
"""
import os

import numpy as np, soundfile as sf, torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(_HERE, "ref")
SR = 24000
REF_TEXT = ("Guten Tag und herzlich willkommen. Ich bin Ihre deutsche Sprachassistentin "
            "und lese Ihnen jeden beliebigen Text mit klarer, freundlicher Stimme vor.")

os.makedirs(REF_DIR, exist_ok=True)
print(">> loading model k2-fsa/OmniVoice ...", flush=True)
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", dtype=torch.bfloat16).to("cuda").eval()
print(">> model on", next(model.parameters()).device, flush=True)
gen = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0)

# instruct must use ONLY the supported vocabulary, comma+space separated.
instructs = [
    "female, young adult, moderate pitch",
    "female, middle-aged, moderate pitch",
    "female, young adult, high pitch",
    "female, moderate pitch",
]
for i, ins in enumerate(instructs):
    print(f">> candidate {i}: {ins}", flush=True)
    out = model.generate(text=REF_TEXT, language="de", instruct=ins, generation_config=gen)
    wav = np.asarray(out[0], dtype=np.float32)
    sf.write(os.path.join(REF_DIR, f"cand_{i}.wav"), wav, SR)
    print(f"   wrote cand_{i}.wav  ({len(wav)/SR:.1f}s, peak {abs(wav).max():.3f})", flush=True)
print(">> done. The spoken text for every candidate is REF_TEXT above —", flush=True)
print("   copy it verbatim into ref/ref.txt when you freeze one as ref/ref.wav.", flush=True)
