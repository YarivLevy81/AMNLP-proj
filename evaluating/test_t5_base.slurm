#! /bin/sh
#SBATCH --job-name=test_t5_base
#SBATCH --output=test_t5_base.out
#SBATCH --error=test_t5_base.err
#SBATCH --partition=studentkillable
#SBATCH --mail-user=sehaik@mail.tau.ac.il
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=60
./scripts/test_t5_base.sh
