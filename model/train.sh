#!/bin/bash
#$ -pe smp 1      # Specify parallel environment and legal core size
#$ -q gpu             # Specify queue
#$ -l gpu_card=1  # number of GPU
#$ -N simcse      # cchit_scibert_completion

echo "Unsupervised finetune"
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

cur_script_dir="${proj_root}/model/"
py_file="${proj_root}/model/train.py"
data_folder="${proj_root}/data/output_pd.csv"
pretrained_model="indolem/indobertweet-base-uncased"
cur_time=$(date +"%Y_%m_%d_%I_%M_%S")
output_dir="${cur_script_dir}/output/${cur_time}_${seed}"
log_dir="${cur_script_dir}/logs/${cur_time}_${seed}"
image_dir="${cur_script_dir}/images/${cur_time}_${seed}"

mkdir -p $output_dir
mkdir -p $log_dir
mkdir -p $image_dir

python $py_file \
    --model_name_or_path $pretrained_model \
    --data_folder $data_folder \
    --reduction_teq 'UMAP' \
    --output_dir $output_dir \
    --batch_size  32 \
    --image_dir $image_dir \
    --max_seq_length 256 \
    --fp16 \
    "$@"
