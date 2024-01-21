# 1547
i=0
while (( $i < 100000 ))
do
    python /root/autodl-fs/hnu-nlp/ccac/utils/CCAC-2023-AIDebater/Track2_Argument\ Generation/trans_data.py
    python /root/autodl-fs/hnu-nlp/run.py --config /root/autodl-fs/hnu-nlp/ccac/configs/baseline.yml
    python /root/autodl-fs/hnu-nlp/run.py --config /root/autodl-fs/hnu-nlp/ccac/configs/baseline.yml
    python /root/autodl-fs/hnu-nlp/run.py --config /root/autodl-fs/hnu-nlp/ccac/configs/baseline.yml
done
