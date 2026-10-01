import json, sys, glob
d, m = sys.argv[1], sys.argv[2]
slug = m.replace("/", "__").replace(".", "__")
f = glob.glob(f"{d}/{slug}.json")
ok = bool(f) and any(v for k, v in json.load(open(f[0])).items() if k.endswith("_wer") and k != "mean_wer")
print("OK" if ok else f"FAILED {m}"); sys.exit(0 if ok else 1)
