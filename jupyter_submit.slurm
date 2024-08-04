#!/bin/bash
#SBATCH --job-name="Jupyter"                # Job name
#SBATCH --mail-user=sujinesh@stanford.edu   # Email address    
#SBATCH --mail-type=NONE                    # Mail notification type (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition=normal                  # Node partition
#SBATCH --nodes=1                           # Number of nodes requested
#SBATCH --ntasks=1                          # Number of processes
#SBATCH --time=01:00:00                     # Time limit request

source ~/ME344/python/python-venv/bin/activate
EXEC_DIR=$HOME/ME344
hostname && jupyter-notebook --no-browser --notebook-dir=$EXEC_DIR
