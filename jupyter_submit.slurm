#!/bin/bash
#SBATCH --job-name="Jupyter"                # Job name
#SBATCH --mail-user=sujinesh@stanford.edu   # Email address    
#SBATCH --mail-type=NONE                    # Mail notification type (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition=gpu-pascal                  # Node partition, use gpu-pascal for gpus
#SBATCH --nodes=1                           # Number of nodes requested
#SBATCH --ntasks=1                          # Number of processes
#SBATCH --time=03:00:00                     # Time limit request

source ~/codes/python/python-venv/bin/activate
EXEC_DIR=$HOME/ME344
# hostname && apptainer exec --nv container.img jupyter-notebook --no-browser --port=9999 --notebook-dir=$EXEC_DIR
# apptainer exec container.img jupyter nbconvert final_project_train.ipynb --to python
apptainer exec --nv container.img python3 final_project_train.py
