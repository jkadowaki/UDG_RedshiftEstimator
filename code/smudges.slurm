#!/bin/bash
### Requests 1 Standard Node on Puma with 94 CPUs with 5GB of memory each, totaling 470GB memory.
### https://public.confluence.arizona.edu/display/UAHPC/Compute+Resources

#SBATCH --job-name=SMUDGes_Redshift_CNN_train
#SBATCH --nodes=1
#SBATCH --ntasks=94             
#SBATCH --mem=470gb
###SBATCH --gres=gpu:1                   
#SBATCH --time=00:30:00   
#SBATCH --partition=standard
#SBATCH --account=<group_name>   
 
# SLURM Inherits your environment. cd $SLURM_SUBMIT_DIR not needed
pwd; hostname; date
 
module load python
python --version
