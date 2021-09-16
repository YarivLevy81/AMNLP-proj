import os
import subprocess
import time
import tensorflow.compat.v1 as tf

exp_name= None
dataset = None
eval_dataset= None
warmup = None
gamma = None
schedule_steps = None
accum= None
lr = None

for dataset, eval_dataset in [('squad/42-num-examples-512/', 'squad/dev/')]:
    for warmup in [200,1000,2000]:
        for gamma in [0.95,0.99]:
            for schedule_steps in [50,100,600]:
                for accum in [32,128]:
                    for lr in [1e-3,1e-4,1e-5]:
                        exp_name = f'{dataset}_warmup={warmup}_gamma={gamma}_schedule={schedule_steps}_accum={accum}_lr={lr}'
                        tf.print(f'sendin exp={exp_name.replace("/","_")}')
                        cmd = f"""#! /bin/sh
#SBATCH --job-name=finetune
#SBATCH --output=slurm/finetune.%j.out
#SBATCH --error=slurm/finetune.%j.err
#SBATCH --partition=studentkillable
#SBATCH --mail-user=sehaik@mail.tau.ac.il
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=600
python3 train.py \
--train_dataset=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/{dataset} \
--eval_dataset=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/{eval_dataset} \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--config=t5-small \
--num_train_steps=10000 \
--print_loss_steps=250 \
--save_checkpoints_steps=250 \
--train_batch_size=1 \
--eval_batch_size=1 \
--calc_f1=1 \
--skip_eval=0 \
--save_best=1 \
--save_latest=1 \
--save_step=0 \
--num_workers=30 \
--from_pretrained=1 \
--output_dir=/home/yandex/AMNLP2021/sehaik/exps/{exp_name} \
--warmup_steps={warmup} --gamma={gamma} \
--schedule_steps={schedule_steps} \
--gradient_accumulation_steps={accum} \
--learning_rate={lr}"""
                        with open('finetune.slurm', "w") as f:
                            f.write(cmd)
                        os.system('sbatch finetune.slurm')
                        while True:
                            out = subprocess.check_output(['squeue'])
                            n = str(out).count('sehaik')
                            if n<10:
                                break
                            tf.print(f'{n} is too many. waiting...')
                            time.sleep(60)
                        time.sleep(10) # make sure slurm queue updates