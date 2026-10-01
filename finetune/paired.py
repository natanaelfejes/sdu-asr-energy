import json, glob, numpy as np, jiwer
d = "/work/team/sdu-asr/valfull"
B = json.load(open(f"{d}/base_pairs.json"))
def per_utt(P):
    return np.array([jiwer.process_words(r, h).wer * len(r.split()) for r, h in zip(P["refs"], P["hyps"])])
eb = per_utt(B); w = np.array([len(r.split()) for r in B["refs"]])
rng = np.random.default_rng(0)
for f in sorted(glob.glob(f"{d}/*_pairs.json")):
    n = f.split("/")[-1].replace("_pairs.json", "")
    if n == "base": continue
    P = json.load(open(f)); assert P["refs"] == B["refs"], "clip order differs"
    e = per_utt(P); diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(w), len(w)); diffs.append((e[i].sum() - eb[i].sum()) / w[i].sum() * 100)
    diffs = np.array(diffs)
    print(f"PAIRED {n}: delta vs base = {(e.sum()-eb.sum())/w.sum()*100:+.2f} pp, 95% CI [{np.percentile(diffs,2.5):+.2f}, {np.percentile(diffs,97.5):+.2f}], P(worse) = {(diffs>0).mean():.3f}")
