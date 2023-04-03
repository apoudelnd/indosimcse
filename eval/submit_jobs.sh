#!/bin/bash
# batch submission for crc
target_dirs=()
# for model in 'bert' 'distilbert' 'roberta' 'xlnet' 'se_bert'; do
# for model in 'se_scibert' 'se_bert_pretrained'; do
for model in 'indobert-base' 'indobert-lite-base' 'indobert-lite-large' 'indobert-large' "xlm-r" "indosimcse" "indobertweet"; do
    dir_path="/scratch365/apoudel/indosimcse/eval/camlruns/${model}/"
    if [[ -d $dir_path ]]; then
        target_dirs+=($(realpath $dir_path))
    fi
done

for d in ${target_dirs[@]}; do
    cd $d
    echo $PWD
    chmod +x train.sh
    condor_submit submit_train.jdl
done