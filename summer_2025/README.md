# ME344 Project X - LLM + RAG Slang Translator

Author: [Sujeeth Jinesh](https://www.linkedin.com/in/SujeethJinesh/)

This project will involve building a RAG pipeline for a chatbot you will create. This tutorial is heavily inspired by content from [pixegami](https://www.youtube.com/watch?v=2TJxpyO3ei4).

## TODO: Add high level description of LLM + RAG project and overview.

In this project, you will build a slang translator. We will be using [this dataset](https://www.kaggle.com/datasets/therohk/urban-dictionary-words-dataset?resource=download) as our slang data.

## Accessing your cluster

Ssh into your assigned cluster.

Execute `su - student` to switch to the student user.

## Download requirements

### Create a virtual environment

If you have not already created a virtual environment before, paste the following into your terminal. This will create a python3.11 virtual environment called "python-venv", and will activate the virtual environment.

```
mkdir -p ~/codes/python && cd ~/codes/python
python3.11 -m venv python-venv
source python-venv/bin/activate
```

<details>
  <summary>Why use a virtual environment?</summary>
  
  Virtual environments in Python primarily allow us to resolve dependencies in a controlled environment that are separate from system wide packages. Imagine some packages needing some specific package versions, while others need some others, this could lead to messy problems if you don't resolve dependencies in an isolated environment.
</details>

You can also add `source python-venv/bin/activate` to your `.bashrc` so you'll activate the virtual environment on login (this is highly recommended).

<details>
  <summary>Why use bashrc?</summary>
  
  This script is run at the beginning of login, and allows us to define some actions we want to take before we get control of the terminal. In this case, we want to activate our python environment so we don't forget to do this later and mess with system packages accidentally.
</details>

## TODO: Download Ollama

```
curl -fsSL https://ollama.com/install.sh | sh
```

Then run ollama locally (Necessary for the embeddings step). This will download the 8B model.

```
ollama run llama3.1
```

You may even want to try the 70B model if your GPU has ~40GB of VRAM because it uses 4 bit quantization (basically reducing VRAM needed to run on a GPU). The P100s on your cluster have 16 GB of VRAM. All Llama 3.1 models were originally quantized with fp16, and if we use 4 bit quantization then we can expect to save ~75% (4/16 = 25% memory footprint) or about 10GB of our P100 card.

```
ollama run llama3.1:70b
```

Ollama will be running on `localhost:11434`.

Create a file called `jupyter_submit.slurm`, and put in the following.

```
#!/bin/bash
#SBATCH --job-name="Jupyter"                # Job name
#SBATCH --mail-user=<email>@stanford.edu    # Email address
#SBATCH --mail-type=NONE                    # Mail notification type (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition=gpu-pascal              # Node partition, use gpu-pascal for gpus
#SBATCH --nodes=1                           # Number of nodes requested
#SBATCH --ntasks=1                          # Number of processes
#SBATCH --time=03:00:00                     # Time limit request

source ~/codes/python/python-venv/bin/activate
EXEC_DIR=$HOME/ME344
# hostname && apptainer exec --nv container.img jupyter-notebook --no-browser --port=9999 --notebook-dir=$EXEC_DIR
# apptainer exec container.img jupyter nbconvert final_project_train.ipynb --to python
apptainer exec --nv container.img python3 final_project_train.py
```

Launch the job with `sbatch jupyter_submit.slurm` This will launch an apptainer with `final_project_train.py`. Note you may need to cancel any other pending jobs with `scancel <job-id>`.

This will produce a slurm-<job-id>.out. Then you can run the following command in a separate window:

`ssh -L 8888:localhost:8888 student@hpcc-cluster-[C] -t ssh -L 8888:localhost:8888 compute-1-1`.

Then you can find the port forwarded address of the jupyter notebook with `egrep -w 'compute|localhost'  slurm-*.out`. Now you should be able to connect with the cluster and run jobs.

From here you can open up final_project_train.ipynb and run all the cells.

### Download Model (Llama 3.1 or others)

## Build RAG pipeline

## Serve locally
