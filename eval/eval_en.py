"""English (accented) ASR eval with GPU energy: WER + Wh per audio hour.

Energy = NVML board energy counter (mJ) read right before and after inference
(model loading excluded). Same speed definition as the Danish harness:
speed_x = audio seconds / inference seconds.
"""
import argparse, json, os, time
import io
import numpy as np, torch, jiwer, pynvml, soundfile as sf, librosa
from datasets import load_dataset, Audio
from transformers import pipeline
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer

SETS = {  # name: (hf id, config, split, text key, extra group key)
    "fleurs_en": ("parquet:google/fleurs@refs/convert/parquet/en_us/test/*.parquet", None, "test", "raw_transcription", None),
    "edacc": ("edinburghcstr/edacc", None, "test", "text", "accent"),
    "ami_ihm": ("edinburghcstr/ami", "ihm", "test", "text", None),
}

def load(name, n):
    src, cfg, split, tk, gk = SETS[name]
    if src.startswith("parquet:"):
        ds = load_dataset("parquet", data_files={split: "hf://datasets/" + src[8:]}, split=split, streaming=True)
    else:
        ds = load_dataset(src, cfg, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    rows = []
    for r in ds:
        t = (r[tk] or "").strip()
        if not t or t.lower() in ("ignore_time_segment_in_scoring",):
            continue
        a = r["audio"]
        y, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        rows.append((y, t, r.get(gk) if gk else None))
        if n and len(rows) >= n:
            break
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", default="fleurs_en,edacc,ami_ihm")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="/work/team/sdu-asr/run/en")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
    asr = pipeline("automatic-speech-recognition", model=a.model, torch_dtype=torch.bfloat16, device="cuda:0")
    norm = EnglishTextNormalizer(asr.tokenizer.english_spelling_normalizer) if hasattr(asr.tokenizer, "english_spelling_normalizer") else BasicTextNormalizer()
    res = {"model": a.model, "batch_size": a.batch_size, "dtype": "bf16"}
    for s in a.sets.split(","):
        rows = load(s, a.max_samples)
        audio_s = sum(len(x[0]) for x in rows) / 16000
        torch.cuda.synchronize(); e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h); t0 = time.time()
        gk = {"language": "en", "task": "transcribe"}
        short = [i for i, x in enumerate(rows) if len(x[0]) <= 30 * 16000]
        long_ = [i for i, x in enumerate(rows) if len(x[0]) > 30 * 16000]
        out = [None] * len(rows)
        for i, o in zip(short, asr([rows[i][0] for i in short], batch_size=a.batch_size, generate_kwargs=gk)):
            out[i] = o
        if long_:  # only clips over Whisper's 30 s window get chunked
            for i, o in zip(long_, asr([rows[i][0] for i in long_], batch_size=a.batch_size, chunk_length_s=30, generate_kwargs=gk)):
                out[i] = o
        torch.cuda.synchronize(); t1 = time.time(); e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        hyp = [norm(o["text"]) for o in out]; ref = [norm(x[1]) for x in rows]
        keep = [i for i, r in enumerate(ref) if r.strip()]
        wer = jiwer.wer([ref[i] for i in keep], [hyp[i] for i in keep]) * 100
        wh = (e1 - e0) / 1000 / 3600
        d = {"n": len(rows), "audio_h": round(audio_s / 3600, 3), "infer_s": round(t1 - t0, 1),
             "speed_x": round(audio_s / (t1 - t0), 1), "energy_wh": round(wh, 2),
             "wh_per_audio_h": round(wh / (audio_s / 3600), 3), "wer": round(wer, 2), "n_chunked": len(long_)}
        if SETS[s][4]:
            g = {}
            for i in keep:
                g.setdefault(rows[i][2], ([], []))
                g[rows[i][2]][0].append(ref[i]); g[rows[i][2]][1].append(hyp[i])
            d["wer_by_group"] = {k: round(jiwer.wer(v[0], v[1]) * 100, 2) for k, v in g.items() if len(v[0]) >= 30}
        res[s] = d
        with open(f"{a.out}/{s}__{a.model.replace('/', '_')}.jsonl", "w") as f:
            for i, o in enumerate(out):
                f.write(json.dumps({"ref": rows[i][1], "hyp": o["text"], "group": rows[i][2]}) + "\n")
        print(s, json.dumps({k: v for k, v in d.items() if k != "wer_by_group"}), flush=True)
    json.dump(res, open(f"{a.out}/{a.model.replace('/', '_')}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
