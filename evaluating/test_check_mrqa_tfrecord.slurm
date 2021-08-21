#! /bin/sh
#SBATCH --job-name=test_check_mrqa_tfrecord
#SBATCH --output=test_check_mrqa_tfrecord.out
#SBATCH --error=test_check_mrqa_tfrecord.err
#SBATCH --partition=studentkillable
#SBATCH --mail-user=sehaik@mail.tau.ac.il
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=60
./scripts/test_check_mrqa_tfrecord.sh
