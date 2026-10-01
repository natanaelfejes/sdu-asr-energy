"""Fine-tune edda-v0.1 on CoRal-v3 train (streamed), time-boxed, DDP via torchrun.

torchrun --nproc_per_node=N ft_edda.py --run NAME --lr 1e-5 --mix conv --end 23:00
Writes /work/team/sdu-asr/ft/NAME/{final/, log.jsonl, val.json}. Checkpoint = HF save_pretrained.
Also logs throughput (audio hours processed per GPU-hour), for costing the next allocation.
"""
import argparse, io, json, os, subprocess, time, datetime as dt
import numpy as np, torch, torch.distributed as dist, soundfile as sf, librosa, jiwer
from datasets import load_dataset, Audio, interleave_datasets
from datasets.distributed import split_dataset_by_node
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

BASE = "danish-foundation-models/edda-v0.1"
SR = 16000


def decode(a):
    b = a.get("bytes")
    try:
        y, sr = sf.read(io.BytesIO(b) if b else a["path"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y if sr == SR else librosa.resample(y, orig_sr=sr, target_sr=SR)
    except Exception:
        src = ["-i", "pipe:0"] if b else ["-i", a["path"]]
        p = subprocess.run(["ffmpeg", "-v", "quiet", *src, "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                           input=b, capture_output=True)
        return np.frombuffer(p.stdout, np.float32)


def stream(config, split):
    ds = load_dataset("CoRal-project/coral-v3", config, split=split, streaming=True)
    return ds.cast_column("audio", Audio(decode=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--mix", default="conv", choices=["conv", "both"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--end", required=True, help="HH:MM local wall time to stop training")
    ap.add_argument("--val-n", type=int, default=400)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="-1 = no training, validation only (baseline)")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    out = f"/work/team/sdu-asr/ft/{a.run}"
    os.makedirs(out, exist_ok=True)
    now = dt.datetime.now()
    h, m = map(int, a.end.split(":"))
    end_t = now.replace(hour=h, minute=m, second=0).timestamp()
    if end_t < now.timestamp():
        end_t += 86400

    proc = WhisperProcessor.from_pretrained(BASE)
    proc.tokenizer.set_prefix_tokens(language="danish", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=torch.float32).cuda()
    model.config.forced_decoder_ids = None
    if a.freeze_encoder:
        for p_ in model.model.encoder.parameters():
            p_.requires_grad = False
    start_id = model.config.decoder_start_token_id

    if a.mix == "conv":
        ds = stream("conversation", "train")
    else:
        ds = interleave_datasets([stream("conversation", "train"), stream("read_aloud", "train")],
                                 probabilities=[0.6, 0.4], seed=0, stopping_strategy="all_exhausted")
    ds = ds.shuffle(seed=rank, buffer_size=300)
    ds = split_dataset_by_node(ds, rank=rank, world_size=world)

    def prep(r):
        y = decode(r["audio"])
        text = (r.get("text") or "").strip()
        ok = bool(text) and 0.5 * SR < len(y) <= 30 * SR
        feats = proc.feature_extractor(y if ok else np.zeros(SR, np.float32), sampling_rate=SR).input_features[0]
        return {"f": feats, "l": proc.tokenizer(text).input_ids, "ok": ok, "sec": len(y) / SR}

    ds = ds.map(prep, remove_columns=ds.column_names)

    def collate(b):
        b = [x for x in b if x["ok"]] or b[:1]  # never return an empty batch (would desync DDP ranks)
        f = torch.tensor(np.stack([x["f"] for x in b]))
        L = max(len(x["l"]) for x in b)
        lab = torch.full((len(b), L), -100, dtype=torch.long)
        for i, x in enumerate(b):
            lab[i, :len(x["l"])] = torch.tensor(x["l"])
        if (lab[:, 0] == start_id).all():
            lab = lab[:, 1:]
        return f, lab, sum(x["sec"] for x in b)

    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, collate_fn=collate, num_workers=6, prefetch_factor=2)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if ddp else model
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.lr, weight_decay=0.01)
    warm = 100
    net.train()
    step, audio_s, t0 = 0, 0.0, time.time()
    log = open(f"{out}/log.jsonl", "a") if rank == 0 else None
    for batch in (dl if a.max_steps != -1 else []):
        f, lab, sec = batch
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / warm)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net(input_features=f.cuda(non_blocking=True), labels=lab.cuda(non_blocking=True)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1; audio_s += sec
        stop = time.time() >= end_t or (a.max_steps and step >= a.max_steps)
        if ddp:
            s = torch.tensor([1.0 if stop else 0.0], device="cuda"); dist.all_reduce(s); stop = s.item() > 0
        if rank == 0 and (step % 50 == 0 or stop):
            el = time.time() - t0
            rec = {"step": step, "loss": round(loss.item(), 4), "elapsed_min": round(el / 60, 1),
                   "audio_h_per_gpu_h": round((audio_s * world / 3600) / (el * world / 3600), 1), "world": world,
                   "time": dt.datetime.now().strftime("%H:%M")}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if stop:
            break

    if rank == 0:
        model.save_pretrained(f"{out}/final", safe_serialization=True)
        proc.save_pretrained(f"{out}/final")
        # quick selection metric on CoRal VALIDATION conversation (test set is only used once, for the winner)
        model.eval(); norm = BasicTextNormalizer(); refs, hyps = [], []
        val = stream("conversation", "val")
        batch_f, batch_r = [], []
        for i, r in enumerate(val):
            if i >= a.val_n:
                break
            y = decode(r["audio"])
            if not (r.get("text") or "").strip() or len(y) > 30 * SR:
                continue
            batch_f.append(proc.feature_extractor(y, sampling_rate=SR).input_features[0]); batch_r.append(r["text"])
            if len(batch_f) == 32:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ids = model.generate(torch.tensor(np.stack(batch_f)).cuda(), language="danish", task="transcribe")
                hyps += proc.batch_decode(ids, skip_special_tokens=True); refs += batch_r; batch_f, batch_r = [], []
        wer = jiwer.wer([norm(x) for x in refs], [norm(x) for x in hyps]) * 100
        json.dump({"run": a.run, "lr": a.lr, "mix": a.mix, "freeze_encoder": a.freeze_encoder, "steps": step, "val_conv_wer": round(wer, 2), "n": len(refs)},
                  open(f"{out}/val.json", "w"))
        print("VAL", a.run, round(wer, 2), flush=True)
    if ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
