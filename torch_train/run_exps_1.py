import os
import subprocess
import time
import tensorflow.compat.v1 as tf

warmup = 200
gamma = 0.95
schedule_steps = 600
accum = 32
lr = 0.001

for name in ['squad', 'naturalquestions']:
    for seed in [42, 43, 44]:
        for d_size in [32, 128, 512]:
            dataset = f'{name}/{seed}-num-examples-{d_size}/'
            eval_dataset = f'{name}/dev/'
            exp_name = f'{dataset.replace("/","_")}'
            tf.print(f'sendin exp={exp_name}')
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
--num_train_steps=3000 \
--print_loss_steps=50 \
--save_checkpoints_steps=200 \
--train_batch_size=1 \
--eval_batch_size=1 \
--calc_f1=1 \
--skip_eval=0 \
--save_best=1 \
--save_latest=1 \
--save_step=0 \
--num_workers=30 \
--from_pretrained=1 \
--output_dir=/home/yandex/AMNLP2021/sehaik/exps/from_pretrained/{exp_name} \
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
                time.sleep(300)
            time.sleep(10) # make sure slurm queue updates