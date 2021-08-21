#! /bin/sh
#SBATCH --job-name=test_finetune
#SBATCH --output=test_finetune.out
#SBATCH --error=test_finetune.err
#SBATCH --partition=studentkillable
#SBATCH --mail-user=sehaik@mail.tau.ac.il
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=60
./scripts/test_finetune.sh
