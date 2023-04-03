import os
import argparse

dir_path = "/scratch365/apoudel/indosimcse/eval"

def script_gen(
    model_path="bert-base-uncased",
    model_type="bert",
    save_steps= 100,
    num_train_epochs = 4,
    gpu=2,
    cpu=2,
    do_train=True,
):

    run_name = f"{model_type}"

    return f"""#!/bin/bash
#$ -pe smp {cpu}      # Specify parallel environment and legal core size
#$ -q gpu             # Specify queue
#$ -l gpu_card={gpu}  # number of GPU
#$ -N {run_name}      # cm_bert_100
echo "Use {model_type} model for INDONLI"
proj_root="/scratch365/apoudel/indosimcse/"  
if [ ! -d $proj_root ]; then
    if [ $(basename $PWD) == "INDOSIMCSE" ]; then
        # When $PWD is project root
        proj_root=$PWD
    else
        # When $PWD is current direcotry
        proj_root=$(realpath $PWD/../../../../../)
    fi
fi
echo "project root:$proj_root"
cd $proj_root
source $proj_root/venv_caml/bin/activate

seed=${{1:-12}}
echo "seed:$seed"
cur_script_dir="${{proj_root}}/eval/camlruns/{model_type}"
py_file="${{proj_root}}/eval/train.py"
pretrained_model="{model_path}"
cur_time=$(date +"%Y_%m_%d_%I_%M_%S")
output_dir="${{cur_script_dir}}/output/${{cur_time}}_${{seed}}"
log_dir="${{cur_script_dir}}/logs/${{cur_time}}_${{seed}}"
cache_dir="${{cur_script_dir}}/cache/${{seed}}"
mkdir -p $output_dir
mkdir -p $cache_dir

python $py_file \\
    --data_folder $data_folder \\
    --model_arch "bert" \\
    --model_name_or_path $pretrained_model \\
    --do_train {str(do_train).lower()} \\
    --do_predict true \\
    --learning_rate 5e-5 \\
    --logging_strategy "steps" \\
    --save_strategy "steps" \\
    --evaluation_strategy "steps" \\
    --logging_steps {save_steps} \\
    --save_steps {save_steps} \\
    --eval_steps {save_steps} \\
    --save_total_limit 1 \\
    --num_train_epochs {num_train_epochs} \\
    --per_device_train_batch_size 4 \\
    --per_device_eval_batch_size 16 \\
    --gradient_accumulation_steps 8 \\
    --output_dir $output_dir \\
    --logging_dir $log_dir \\
    --overwrite_output_dir true \\
    --report_to "tensorboard" \\
    --preprocessing_num_workers 4 \\
    --load_best_model_at_end true \\
    --greater_is_better false \\
    --metric_for_best_model "eval_loss" \\
    --seed $seed \\
    --data_cache_dir $cache_dir \\
    --fp16 true
"""


def gen_condor_submit(queue_num=1):
    return f"""universe   = vanilla
executable = train.sh
arguments = $(PROCESS)
should_transfer_files = IF_NEEDED
when_to_transfer_output = ON_EXIT
Log    = $(Cluster)_$(PROCESS).log
Output = $(Cluster)_$(PROCESS).out
Error  = $(Cluster)_$(PROCESS).err
# Enable Singularity feature
## Notre Dame Images
+SingularityImage = "/afs/crc.nd.edu/x86_64_linux/s/singularity/images/pytorch-1.9.sif"
request_gpus   = 1
request_memory = 30 Gb
request_cpus   = 4
Queue {queue_num}
    """

def run_generation(model_type, model_path, overwrite=False, save_steps=100):
    script_dir = os.path.join(dir_path,
        "camlruns",
        model_type,
    )
    if os.path.isdir(script_dir):
        if not overwrite:
            print(
                "script dir already exist, abort to avoid override manually customized parameters"
            )
            return
        else:
            print(f"overwrite {script_dir}")
    else:
        print(f"create dir {script_dir}")
        os.makedirs(script_dir)
    run_script = os.path.join(script_dir, "train.sh")
    condor_script = os.path.join(script_dir, "submit_train.jdl")

    num_train_epochs =  10


    with open(run_script, "w") as fout:
        fout.write(
            script_gen(
                model_path=model_path,
                model_type=model_type,
                save_steps = save_steps,
                num_train_epochs = num_train_epochs
            )
        )

    with open(condor_script, 'w') as fout:
        fout.write(
            gen_condor_submit()
        )
    return script_dir


def gen_baseline(args):

    for (model_type, model) in [
            ("indobert-base", "indobenchmark/indobert-base-p2"),
            ("indobert-lite-base", "indobenchmark/indobert-lite-base-p2"),
            ("indobert-lite-large", "indobenchmark/indobert-lite-large-p2"),
            ("indobert-large", "indobenchmark/indobert-large-p2"),
            ("xlm-r", "xlm-roberta-base"),
            ("indosimcse", "/scratch365/apoudel/indosimcse/SimCSE/output/2023_01_20_12_41_05_12/checkpoint-4500"),
            ("indobertweet", "indolem/indobertweet-base-uncased")
            # ("xlnet", "xlnet-base-cased"),
            # ("gitlink", "/scratch365/jlin6/projects/SEBert/nl_trace/crc/git/commit_to_issue/bert/output/2022_02_21_11_53_34_12/checkpoint-23360"),
            # ("sebert", "/scratch365/jlin6/data/pretrain_output/bert_pretrained_v2_5e5/bert_pretrained_v2_5e5/output/checkpoint-254160")
        ]:

        print(model_type, model)
        save_steps = 100

        run_generation(
                    model_type = model_type,
                    model_path = model,
                    save_steps = save_steps, 
                    overwrite = args.overwrite
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate job script for experiments")
    parser.add_argument("--overwrite", '-O', help="Whether overwrite existing script if dir exists already", default=False)
    args = parser.parse_args()
    gen_baseline(args)