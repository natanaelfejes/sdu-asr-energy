# Danish: leaderboard harness (published normalisation) + GPU power log per model.
cd /work/team/sdu-asr/harness && . .venv/bin/activate
mkdir -p ../run/da/results ../run/da/outputs ../run/power
for m in danish-foundation-models/edda-v0.1 openai/whisper-large-v3 openai/whisper-large-v3-turbo openai/whisper-medium openai/whisper-small; do
  s=$(echo $m | tr '/' '_')
  [ -f ../run/da/results/DONE_$s ] && continue
  nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu --format=csv,noheader,nounits -lms 200 > ../run/power/da_$s.csv &
  P=$!
  echo "START $m $(date +%T)"
  danish-asr-eval --model $m --backend transformers --datasets fleurs,ftspeech --out-dir ../run/da/results --outputs-dir ../run/da/outputs 2>&1 | grep -a -E "WER|speed_x|Error|Traceback" | tail -8
  kill $P; python3 /work/team/ok.py ../run/da/results "$m" && touch ../run/da/results/DONE_$s; echo "END $m $(date +%T)"
done
echo DA_ALL_DONE
