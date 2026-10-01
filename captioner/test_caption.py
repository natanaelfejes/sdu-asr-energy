"""Test the captioner on public FLEURS test clips stitched into one ~7 minute 'lecture' per language.
Default SKIP=40: clips 41-80, which were NOT used to choose the captioner settings (clips 1-40 were).
Reports WER of the full transcript, speed on this machine, and checks the .srt is well-formed.
"""
import io, json, os, re, sys, numpy as np, soundfile as sf, jiwer
from datasets import load_dataset, Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from caption import caption

def build(cfg, n=40, skip=0):
    ds = load_dataset("parquet", data_files={"test": f"hf://datasets/google/fleurs@refs/convert/parquet/{cfg}/test/*.parquet"},
                      split="test", streaming=True).cast_column("audio", Audio(decode=False))
    parts, refs, gap = [], [], np.zeros(8000, np.float32)  # 0.5 s pause between utterances
    for i, r in enumerate(ds):
        if i < skip: continue
        if i >= skip + n: break
        y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if y.ndim > 1: y = y.mean(1)
        assert sr == 16000, sr
        parts += [y, gap]; refs.append(r["raw_transcription"])
    return np.concatenate(parts), " ".join(refs)

out = {}
for lang, cfg in (("da", "da_dk"), ("en", "en_us")):
    audio, ref = build(cfg, skip=int(os.environ.get("SKIP", "40")))
    wav = f"/tmp/fleurs_{lang}_lecture.wav"; sf.write(wav, audio, 16000)
    r = caption(wav, lang)
    norm = BasicTextNormalizer() if lang == "da" else EnglishTextNormalizer({})
    wer = jiwer.wer(norm(ref), norm(r["text"])) * 100
    srt = open(r["srt"], encoding="utf-8").read()
    blocks = [b for b in srt.strip().split("\n\n") if b]
    ok = all(re.match(r"\d+\n\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}\n.+", b, re.S) for b in blocks)
    out[lang] = {"wer_pct": round(wer, 2), **{k: r[k] for k in ("model", "device", "audio_sec", "seconds", "x_realtime", "segments")},
                 "srt_well_formed": ok, "srt_sample": "\n\n".join(blocks[:2])}
    print(lang, json.dumps({k: v for k, v in out[lang].items() if k != "srt_sample"}), flush=True)
json.dump(out, open(__file__.rsplit("/", 1)[0] + "/test_results_heldout.json" if os.environ.get("SKIP", "40") != "0" else __file__.rsplit("/", 1)[0] + "/test_results_tuning.json", "w"), indent=1, ensure_ascii=False)
