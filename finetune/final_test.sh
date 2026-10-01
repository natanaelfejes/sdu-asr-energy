# usage: bash final_test.sh RUN   -> harness test of ft/RUN/final on CoRal (conv+read), FLEURS, FTSpeech, with power log
RUN=$1; M=/work/team/sdu-asr/ft/$RUN/final
cd /work/team/sdu-asr/harness && . .venv/bin/activate && export HF_TOKEN=$(cat /work/team/.hf_token)
mkdir -p ../run/ft_test/results ../run/ft_test/outputs ../run/power
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu --format=csv,noheader,nounits -lms 200 > ../run/power/fttest_$RUN.csv & P=$!
danish-asr-eval --model $M --model-id sdu/edda-ft-$RUN --backend transformers --datasets coral_conversation,coral_read_aloud --out-dir ../run/ft_test/results/coral --outputs-dir ../run/ft_test/outputs --audio-cache-dir ../run/coral_cache 2>&1 | grep -a -E "WER:|speed_x"
danish-asr-eval --model $M --model-id sdu/edda-ft-$RUN --backend transformers --datasets fleurs,ftspeech --out-dir ../run/ft_test/results/other --outputs-dir ../run/ft_test/outputs 2>&1 | grep -a -E "WER:|speed_x"
kill $P; echo FINAL_TEST_DONE $RUN
