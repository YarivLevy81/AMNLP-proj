#! /bin/sh
#SBATCH --job-name=torch_train
#SBATCH --output=slurm/squad_512_seed=42.%j.out
#SBATCH --error=slurm/squad_512_seed=42.%j.err
#SBATCH --partition=studentkillable
#SBATCH --mail-user=sehaik@mail.tau.ac.il
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=60
python3 train.py \
--train_dataset=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/42-num-examples-512/ \
--eval_dataset=/home/yandex/AMNLP2021/sehaik/torch_mrqa_datasets/squad/dev/ \
--cache_dir=/home/yandex/AMNLP2021/sehaik/huggingface_cache \
--config=t5-small \
--num_train_steps=20000 \
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
--output_dir=/home/yandex/AMNLP2021/sehaik/squad_512_seed=42/ \
--warmup_steps=2000 --gamma=0.9999 \
--schedule_steps=1000 \
--gradient_accumulation_steps=12 \
--learning_rate=1e-4

