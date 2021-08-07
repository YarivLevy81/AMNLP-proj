#! /bin/sh
#SBATCH --job-name=create_data
#SBATCH --output=create_data.out
#SBATCH --error=create_data.err
#SBATCH --partition=studentbatch
#SBATCH --mail-user=sehaik@mail.tau.ac.il
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
./scripts/create_data_for_splinter_generative.sh