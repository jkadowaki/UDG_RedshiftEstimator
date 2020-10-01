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
#SBATCH --account=dfz 
 

PROJECT_DIRECTORY=/home/u11/jkadowaki/UDG_RedshiftEstimator
CODE=$PROJECT_DIRECTORY/code 

# Activates Python Virtual Environment
source $PROJECT_DIRECTORY/smudges_env/bin/activate

cd $CODE

echo "START: $(date +%Y/%m/%d' '%H:%M:%S)"

START_TIME=$(date +%s)
python3 ./smudges_cnn_model.py
END_TIME=$(date +%s)

echo "END:      $(date +%Y/%m/%d' '%H:%M:%S)"
echo "DURATION: $((ETIME-STIME)) seconds"
