"""Offline lecture captioner: audio/video in, .srt subtitles out. Runs locally (Apple GPU, NVIDIA or CPU).

Model choice follows the 30 Sep 2026 UCloud measurements:
  Danish  -> danish-foundation-models/edda-v0.1 (up to 3x fewer errors than whisper-large-v3, ~1/4 energy)
  English -> openai/whisper-large-v3-turbo      (more accurate on accented English, ~6x less energy)

The audio is split at every pause into utterance-sized pieces (at most MAX_SEC), each piece is transcribed as
short-form audio (the setting the benchmark measured), and subtitle times come from the split points. This avoids
transformers' long-form chunking, which merged a 7-minute test file into a single subtitle.

usage: python caption.py LECTURE.mp4 --lang da|en [--out LECTURE.srt]
"""
import argparse, subprocess, sys, time
import numpy as np, torch
from transformers import pipeline

MODELS = {"da": "danish-foundation-models/edda-v0.1", "en": "openai/whisper-large-v3-turbo"}
LANG = {"da": "danish", "en": "english"}
SR = 16000
MAX_SEC = 25.0   # longest piece sent to the model (Whisper's window is 30 s)
MIN_SEC = {"da": 6.0, "en": 1.0}  # pieces shorter than this merge with the next one. edda (Danish) degrades on
# sentence fragments, turbo (English) prefers single utterances. Chosen on FLEURS test clips 1-40, reported on 41-80.
FRAME = 0.03     # 30 ms frames for the energy-based pause detector
MIN_PAUSE = 0.3  # a pause must last this long to be a split point


def load_audio(path):
    """Any audio/video file -> 16 kHz mono float32, via ffmpeg (no audio leaves the machine)."""
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32)


def split_at_pauses(audio, min_sec=6.0):
    """Return [(start_sample, end_sample)] speech pieces, one per pause-delimited utterance.

    Cuts at every pause >= MIN_PAUSE (models are trained and benchmarked on single utterances;
    packing several sentences into one 20 s piece cost ~4 WER points in testing). Pieces shorter
    than MIN_SEC are merged with the next one; pieces longer than MAX_SEC are hard-cut. Pieces whose
    energy is near silence are dropped, so silence/noise does not produce hallucinated subtitles.
    """
    n = int(FRAME * SR)
    frames = len(audio) // n
    if frames == 0:
        return []
    rms = np.sqrt(np.mean(audio[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    # a pause must be clearly quieter than speech: relative to the quiet floor, but never above 30% of the
    # loud level (otherwise steady, dense speech would count as one long pause and be dropped)
    thresh = max(min(np.percentile(rms, 20) * 2.0, np.percentile(rms, 90) * 0.3), 1e-4)
    quiet = rms < thresh
    # pieces are dropped only if, on average, they are as quiet as a pause (relative and capped, so quiet
    # recordings keep their speech and dense speech is never dropped; tested on both)
    silence = thresh
    need = int(round(MIN_PAUSE / FRAME))
    # speech regions = maximal runs of frames not inside a pause of >= need frames
    regions, i = [], 0
    in_pause = np.zeros(frames, bool)
    while i < frames:
        if quiet[i]:
            j = i
            while j < frames and quiet[j]:
                j += 1
            if j - i >= need:
                in_pause[i:j] = True
            i = j
        else:
            i += 1
    i = 0
    while i < frames:
        if not in_pause[i]:
            j = i
            while j < frames and not in_pause[j]:
                j += 1
            regions.append([i * n, min(j * n, len(audio))])
            i = j
        else:
            i += 1
    # merge too-short regions forward, hard-cut too-long ones
    merged = []
    for a, b in regions:
        if merged and (merged[-1][1] - merged[-1][0]) < min_sec * SR and (b - merged[-1][0]) <= MAX_SEC * SR:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    pieces, max_len = [], int(MAX_SEC * SR)
    for a, b in merged:
        while b - a > max_len:
            pieces.append((a, a + max_len)); a += max_len
        pieces.append((a, b))
    keep = []
    for a, b in pieces:
        seg = audio[a:b]
        if b - a > int(0.2 * SR) and np.sqrt(np.mean(seg ** 2)) > silence:
            keep.append((a, b))
    return keep


def ts(sec):
    ms = int(round(sec * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def device():
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def caption(path, lang, out=None):
    audio = load_audio(path)
    dev, dtype = device()
    asr = pipeline("automatic-speech-recognition", model=MODELS[lang], dtype=dtype, device=dev)
    pieces = split_at_pauses(audio, MIN_SEC[lang])
    if not pieces:
        raise SystemExit(f"no speech found in {path}")
    t0 = time.time()
    res = asr([audio[a:b] for a, b in pieces], batch_size=8,
              generate_kwargs={"language": LANG[lang], "task": "transcribe"})
    took = time.time() - t0
    out = out or path.rsplit(".", 1)[0] + ".srt"
    texts, i = [], 0
    with open(out, "w", encoding="utf-8") as f:
        for (a, b), r in zip(pieces, res):
            text = r["text"].strip()
            if not text:
                continue
            i += 1; texts.append(text)
            f.write(f"{i}\n{ts(a / SR)} --> {ts(b / SR)}\n{text}\n\n")
    dur = len(audio) / SR
    return {"srt": out, "text": " ".join(texts), "audio_sec": round(dur, 1), "seconds": round(took, 1),
            "x_realtime": round(dur / took, 1), "device": dev, "model": MODELS[lang], "segments": i}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--lang", choices=["da", "en"], required=True); ap.add_argument("--out")
    a = ap.parse_args()
    r = caption(a.file, a.lang, a.out)
    print({k: v for k, v in r.items() if k != "text"}, file=sys.stderr)
