#! /bin/sh
#SBATCH --job-name=run_pretrain
#SBATCH --output=run_pretrain.out
#SBATCH --error=run_pretrain.err
#SBATCH --partition=studentkillable
#SBATCH --mail-user=sehaik@mail.tau.ac.il
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=600
./scripts/run_splinter_generative.sh
