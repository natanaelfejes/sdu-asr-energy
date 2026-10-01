"""Full CoRal conversation VAL WER with 95% bootstrap CI (resampling utterances).
python val_full.py MODEL_PATH_OR_ID OUT_JSON
"""
import io, json, sys, numpy as np, torch, soundfile as sf, librosa, jiwer
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

name, out = sys.argv[1], sys.argv[2]
proc = WhisperProcessor.from_pretrained(name)
model = WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
ds = load_dataset("CoRal-project/coral-v3", "conversation", split="val", streaming=True).cast_column("audio", Audio(decode=False))
norm = BasicTextNormalizer(); refs, hyps, fb, rb = [], [], [], []

def flush():
    global fb, rb
    with torch.no_grad():
        ids = model.generate(torch.tensor(np.stack(fb)).cuda().to(torch.bfloat16), language="danish", task="transcribe")
    hyps.extend(proc.batch_decode(ids, skip_special_tokens=True)); refs.extend(rb); fb, rb = [], []

for r in ds:
    t = (r.get("text") or "").strip()
    y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    if not t or len(y) > 30 * 16000: continue
    fb.append(proc.feature_extractor(y, sampling_rate=16000).input_features[0]); rb.append(t)
    if len(fb) == 64: flush()
if fb: flush()
R = [norm(x) for x in refs]; H = [norm(x) for x in hyps]
keep = [i for i, x in enumerate(R) if x.strip()]
R = [R[i] for i in keep]; H = [H[i] for i in keep]
m = jiwer.process_words(R, H)
# per-utterance errors and word counts for bootstrap
errs = np.array([jiwer.process_words(r, h).wer * len(r.split()) for r, h in zip(R, H)]); words = np.array([len(r.split()) for r in R])
rng = np.random.default_rng(0); bs = []
for _ in range(1000):
    idx = rng.integers(0, len(R), len(R)); bs.append(errs[idx].sum() / words[idx].sum() * 100)
res = {"model": name, "n": len(R), "wer": round(m.wer * 100, 2), "ci95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]}
json.dump(res, open(out, "w")); print("VALFULL", json.dumps(res), flush=True)
json.dump({"refs": R, "hyps": H}, open(out.replace(".json", "_pairs.json"), "w"))
