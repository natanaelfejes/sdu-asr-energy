"""Combine all runs into results/summary.csv + results/summary.md.

Danish (harness): WER per dataset from run/{da,coral}/results/*.json.
  Energy = mean board power while GPU util >= 10% (power/*.csv, 200 ms samples) / speed_x.
English (eval_en.py): WER + Wh per audio hour from NVML energy counter, per dataset.
Usage: python3 aggregate.py <copied run dir>   (the 'run' folder from copy_results.sh)
"""
import csv, glob, json, os, sys

run = sys.argv[1]
out_dir = os.path.dirname(os.path.dirname(os.path.abspath(run)))
rows = []

def active_power(path):
    ps = []
    for line in open(path):
        parts = [p.strip() for p in line.split(",")]
        try:
            w, u = float(parts[1]), float(parts[2])
        except (ValueError, IndexError):
            continue
        if u >= 10:
            ps.append(w)
    return sum(ps) / len(ps) if ps else None

for kind, sets in (("da", ("fleurs_da", "ftspeech")), ("coral", ("coral_conversation", "coral_read_aloud"))):
    for f in glob.glob(f"{run}/{kind}/results/*.json"):
        d = json.load(open(f))
        model = d["model"].split("](")[0].strip("[")
        slug = model.replace("/", "_")
        pw = glob.glob(f"{run}/power/{'da' if kind == 'da' else 'coral'}_{slug}.csv")
        p = active_power(pw[0]) if pw else None
        wh = round(p / d["speed_x"], 2) if p and d.get("speed_x") else None
        for s in sets:
            if d.get(f"{s}_wer") is not None:
                rows.append({"lang": "da", "model": model, "dataset": s, "wer": d[f"{s}_wer"],
                             "wh_per_audio_h": wh, "speed_x": d.get("speed_x"), "energy_method": "active power / speed_x (per model run)"})

for f in glob.glob(f"{run}/en/*.json"):
    d = json.load(open(f))
    for s in ("fleurs_en", "edacc", "ami_ihm"):
        if s in d:
            rows.append({"lang": "en", "model": d["model"], "dataset": s, "wer": d[s]["wer"],
                         "wh_per_audio_h": d[s]["wh_per_audio_h"], "speed_x": d[s]["speed_x"], "energy_method": "NVML energy counter (per dataset)"})

rows.sort(key=lambda r: (r["lang"], r["dataset"], r["wer"]))
with open(f"{out_dir}/summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

with open(f"{out_dir}/summary.md", "w") as fh:
    for lang in ("da", "en"):
        fh.write(f"\n## {'Danish' if lang == 'da' else 'English (accented)'}\n\n| dataset | model | WER % | Wh / audio hour | speed x |\n|---|---|---|---|---|\n")
        for r in rows:
            if r["lang"] == lang:
                fh.write(f"| {r['dataset']} | {r['model']} | {r['wer']} | {r['wh_per_audio_h']} | {r['speed_x']} |\n")
print(open(f"{out_dir}/summary.md").read())
