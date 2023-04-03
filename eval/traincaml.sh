#!/bin/bash
#$ -pe smp 12     # Specify parallel environment and legal core size
#$ -q gpu         # Specify queue
#$ -l gpu_card=4  # number of GPU
#$ -N c_b_c          # cchit_bert_completion

echo "Use bert model for compltion task of CCHIT dataset"
proj_root="/scratch365/apoudel/indosimcse"
if [ ! -d $proj_root ]; then
    if [ $(basename $PWD) == "SEBert" ]; then
        # When $PWD is project root
        proj_root=$PWD
    else
        # When $PWD is current direcotry
        proj_root=$(realpath $PWD/../../../../../)
    fi
fi
echo "project root:$proj_root"

cd $proj_root
source ${proj_root}/venv_caml/bin/activate

seed=${1:-12}
echo "seed:$seed"


cur_script_dir="${proj_root}/eval/"
py_file="${proj_root}/eval/train.py"
pretrained_model="/scratch365/apoudel/indosimcse/eval/output/2023_03_01_12_52_06_12"
cur_time=$(date +"%Y_%m_%d_%I_%M_%S")
output_dir="${cur_script_dir}/output/${cur_time}_${seed}"
log_dir="${cur_script_dir}/logs/${cur_time}_${seed}"
cache_dir="${cur_script_dir}/cache/${seed}"
dataset_config_name="NLI-indonesia"

mkdir -p $log_dir
mkdir -p $output_dir
mkdir -p $cache_dir


python $py_file \
    --model_name_or_path $pretrained_model \
    --do_train false \
    --do_predict true \
    --learning_rate 5e-5 \
    --logging_strategy "steps" \
    --save_strategy "steps" \
    --evaluation_strategy "steps" \
    --logging_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --output_dir $output_dir \
    --logging_dir $log_dir \
    --overwrite_output_dir true \
    --report_to "tensorboard" \
    --preprocessing_num_workers 4 \
    --load_best_model_at_end true \
    --greater_is_better false \
    --metric_for_best_model "eval_loss" \
    --seed $seed \
    --data_cache_dir $cache_dir \
    --fp16 true \