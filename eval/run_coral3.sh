# CoRal-v3 test (conversation + read-aloud) with published harness scoring + power log; then resume FLEURS/FTSpeech leftovers.
cd /work/team/sdu-asr/harness && . .venv/bin/activate
export HF_TOKEN=$(cat /work/team/.hf_token)
mkdir -p ../run/coral/results ../run/coral/outputs ../run/power
for m in danish-foundation-models/edda-v0.1 openai/whisper-large-v3 openai/whisper-large-v3-turbo openai/whisper-medium openai/whisper-small; do
  s=$(echo $m | tr '/' '_'); [ -f ../run/coral/results/DONE_$s ] && continue
  nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu --format=csv,noheader,nounits -lms 200 > ../run/power/coral_$s.csv & P=$!
  echo "START $m $(date +%T)"
  danish-asr-eval --model $m --backend transformers --datasets coral_conversation,coral_read_aloud --out-dir ../run/coral/results --outputs-dir ../run/coral/outputs --audio-cache-dir ../run/coral_cache 2>&1 | grep -a -E "_wer |speed_x|Error|Traceback" | tail -8
  kill $P; python3 /work/team/ok.py ../run/coral/results "$m" && touch ../run/coral/results/DONE_$s; echo "END $m $(date +%T)"
done
echo CORAL_ALL_DONE
