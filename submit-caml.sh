#!/bin/bash
#$ -q long     # Specify queue (use ‘debug’ for development)
#$ -N L2A      # Specify job name
if [ -r /opt/crc/Modules/current/init/bash ]; then
    source /opt/crc/Modules/current/init/bash
fi

proj_root="/scratch365/apoudel/indosimcse"
py_file="${proj_root}/data_process.py"
cd $proj_root
source ${proj_root}/venv_caml/bin/activate
python ./data_process.py 
