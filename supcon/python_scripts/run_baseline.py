#!/usr/bin/env python

import os

from cvar_pyutils.ccc import submit_job

os.environ['TPU'] = 'CCC'
os.environ['DATA_DIR'] = '~/dccstor/hansri/data/'
os.environ['MODEL_DIR'] = '~/dccstor/hansri/google-research/supcon/trained_models/'

submit_dependant_jobs(number_of_rolling_jobs=16, command_to_run='scripts/supcon_imagenet_resnet50.sh --mode=train_then_eval \
  --tpu_name=CCC --data_dir=~/dccstor/hansri/data/imagenet2012/ --model_dir=$~/dccstor/hansri/google-research/supcon/trained_models \ 
  --use_tpu=False', machine_type='x86', time='6h', 
  num_cores=32, num_gpus=4, mem='400g', gpu_type='v100 | a100', 'conda_env'='tf2.2',
  mail_log_file_when_done='seungwook.han@ibm.com')
