cd /work/team/sdu-asr && mkdir -p run/en
[ -d venv-en ] || { uv venv -q venv-en; . venv-en/bin/activate; uv pip install -q torch transformers datasets jiwer nvidia-ml-py soundfile librosa "torchcodec" accelerate 2>&1 | tail -3; }
. venv-en/bin/activate
python -c "import torch,transformers,datasets;print('env ok',torch.__version__,transformers.__version__,datasets.__version__,torch.cuda.is_available())"
python /work/team/eval_en.py --model openai/whisper-small --max-samples 20 --out run/en_test 2>&1 | grep -a -v -i warn | tail -12
echo TEST_DONE
for m in openai/whisper-large-v3 openai/whisper-large-v3-turbo openai/whisper-medium openai/whisper-small distil-whisper/distil-large-v3.5; do
  s=$(echo $m | tr '/' '_'); [ -f run/en/$s.json ] && continue
  echo "START $m $(date +%T)"
  python /work/team/eval_en.py --model $m 2>&1 | grep -a -E "^(fleurs_en|edacc|ami_ihm) |Error|Traceback" | tail -8
  echo "END $m $(date +%T)"
done
echo EN_ALL_DONE
