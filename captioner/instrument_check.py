"""Instrument check: same 40 FLEURS clips, transcribed one by one (no cutting), same normaliser as test_caption.py.
If this WER ~ the captioner's WER, the gap to the leaderboard is the normaliser, not the captioner."""
import io, json, os, numpy as np, soundfile as sf, jiwer, torch
from datasets import load_dataset, Audio
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from caption import MODELS, LANG, device
dev, dt = device()
res = {}
for lang, cfg in (("da", "da_dk"), ("en", "en_us")):
    ds = load_dataset("parquet", data_files={"test": f"hf://datasets/google/fleurs@refs/convert/parquet/{cfg}/test/*.parquet"}, split="test", streaming=True).cast_column("audio", Audio(decode=False))
    clips, refs = [], []
    for i, r in enumerate(ds):
        k = int(os.environ.get("SKIP", "40"))
        if i < k: continue
        if i >= k + 40: break
        y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32"); clips.append(y if y.ndim == 1 else y.mean(1)); refs.append(r["raw_transcription"])
    asr = pipeline("automatic-speech-recognition", model=MODELS[lang], dtype=dt, device=dev)
    hyp = [o["text"] for o in asr(clips, batch_size=8, generate_kwargs={"language": LANG[lang], "task": "transcribe"})]
    norm = BasicTextNormalizer() if lang == "da" else EnglishTextNormalizer({})
    res[lang] = round(jiwer.wer(norm(" ".join(refs)), norm(" ".join(hyp))) * 100, 2)
    print(lang, "per-clip WER, same normaliser:", res[lang], flush=True)
json.dump({"per_clip_wer_pct": res, "clips": "FLEURS test, SKIP=" + os.environ.get("SKIP", "40") + ", n=40"}, open(__file__.rsplit("/", 1)[0] + "/instrument_check_results.json", "w"), indent=1)
