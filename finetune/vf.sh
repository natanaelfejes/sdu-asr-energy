# usage: bash vf.sh NAME:MODEL [NAME:MODEL ...]  (sequential, background)
cd /work/team/sdu-asr && . venv-en/bin/activate && export HF_TOKEN=$(cat /work/team/.hf_token) && mkdir -p valfull
(for spec in "$@"; do n=${spec%%:*}; m=${spec#*:}; python /work/team/val_full.py $m valfull/$n.json 2>&1 | grep -a -E "VALFULL|Error|Traceback" ; done; echo VF_DONE) > /work/team/sdu-asr/valfull_$(hostname).log 2>&1 &
echo started "$@"
