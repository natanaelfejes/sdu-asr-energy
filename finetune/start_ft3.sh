# usage: bash start_ft3.sh RUN LR MIX [extra args]
RUN=$1; LR=$2; MIX=$3; shift 3
(which ffmpeg >/dev/null || (sudo -n apt-get update -qq && sudo -n apt-get install -y -qq ffmpeg)) >/dev/null 2>&1
cd /work/team/sdu-asr && . venv-en/bin/activate && export HF_TOKEN=$(cat /work/team/.hf_token)
nohup python /work/team/ft_edda_v3.py --run $RUN --lr $LR --mix $MIX --end 23:30 --val-n 400 "$@" > /work/team/sdu-asr/ft_$RUN.log 2>&1 &
echo "started $RUN lr=$LR mix=$MIX $*"
