# Danish and English speech recognition on UCloud: accuracy and energy

Measurements made on 30 September 2026 on SDU's UCloud (one NVIDIA B200 per run), comparing the
speech models offered by UCloud's Transcriber app against the open Danish model
[edda-v0.1](https://huggingface.co/danish-foundation-models/edda-v0.1), for word error rate (WER)
and GPU energy per hour of transcribed audio. Public test sets only.

**Two-page summary with all tables, method and sources:** [`results/SDU-ASR-results.pdf`](results/SDU-ASR-results.pdf)

## Main results

| Danish | WER default (whisper-large-v3) | WER edda-v0.1 | Energy default → edda (Wh / audio h) |
|---|---|---|---|
| CoRal conversation | 48.7% | 16.0% | 31.5 → 7.7 |
| CoRal read-aloud | 26.9% | 10.5% | 31.5 → 7.7 |
| FLEURS | 12.8% | 7.6% | 23.4 → 5.9 |
| FTSpeech | 16.5% | 6.0% | 23.4 → 5.9 |

Accented English (EdAcc): whisper-large-v3-turbo 19.0% vs default 21.1% WER, 3.9 vs 25.0 Wh per audio hour.

Fine-tuning edda further on CoRal (10 runs) did not beat it: the best run ties
(+0.14 WER points, 95% CI −0.10 to +0.39, paired bootstrap on 1,887 validation clips).
Full tables: [`results/summary.md`](results/summary.md).

## Offline captioner

`captioner/caption.py` turns a lecture recording into an `.srt` subtitle file, entirely on the local
machine, using edda-v0.1 for Danish and whisper-large-v3-turbo for English (the models chosen by the
measurements above).

```bash
pip install torch transformers   # and ffmpeg on PATH
python captioner/caption.py lecture.mp4 --lang da
```

Tested on a MacBook Pro (M4 Pro) on ~7 minutes of stitched public FLEURS test clips per language that
were **not** used to tune it (`captioner/test_caption.py`, results in `captioner/test_results_heldout.json`):

| | Captioner | Same clips one by one | Speed |
|---|---|---|---|
| Danish (edda-v0.1) | 9.8% WER | 8.5% | ~10× real-time |
| English (large-v3-turbo) | 6.8% WER | 6.5% | ~10× real-time |

The pause-merging setting was chosen per language on clips 1–40 (`instrument_check.py` gives the
one-by-one reference, saved in `instrument_check_results.json`). The captioner test and its reference use the same
normaliser (Whisper `BasicTextNormalizer` for Danish, `EnglishTextNormalizer` without the spelling map for English),
which differs slightly from `eval/eval_en.py`, so compare captioner numbers only with each other.
It has not yet been tested on real lecture recordings. Known limitation: a speaker much quieter than the main one
(about 20 dB, e.g. an audience question far from the microphone) can be treated as silence and left out.

## Contents

| Folder | What |
|---|---|
| `eval/` | English evaluation with GPU energy (`eval_en.py`), Danish run scripts and aggregation |
| `finetune/` | Fine-tuning script, full validation, paired bootstrap |
| `captioner/` | Offline captioner and its test |
| `results/` | Summary tables, PDF, fine-tuning validation results and training logs |

The cluster scripts contain absolute paths from the UCloud run (`/work/team/...`) and read a Hugging
Face token from a file; adjust both to your setup. Danish WER uses the public [Danish ASR leaderboard harness](https://github.com/Rye-A1/danish-asr-leaderboard)
(not included here). Per-utterance transcripts are deliberately not published, because the test-set
licences restrict redistributing their content.

## Data used (not redistributed)

CoRal-v3 (OpenRAIL, gated) · FLEURS (CC BY 4.0) · FTSpeech (evaluation only) · EdAcc (CC BY-SA) · AMI (CC BY 4.0).
See the PDF for links and licence notes.

## Limitations

Benchmark audio, not SDU lectures. Energy is GPU only (no CPU, memory or cooling).
The Danish and English energy figures use two different measurement methods (see the PDF), so compare energy within a language only. English speed/energy on the first test set (FLEURS) includes GPU warm-up; EdAcc, the set the energy claim uses, does not. Fine-tuning confidence intervals resample utterances, not speakers, so they are likely somewhat too narrow.

## Author

Natanael Fejes, BSc Software Engineering, SDU Sønderborg. Compute: SDU UCloud (DeiC Interactive HPC).

MIT licence for the code in this repository.
