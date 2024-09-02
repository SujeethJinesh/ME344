# ME344 Project X - LLM + RAG Slang Translator

Author: [Sujeeth Jinesh](https://www.linkedin.com/in/SujeethJinesh/)

This project will teach you how to build a Retrieval Augmented Generation (RAG) chatbot. It can be used with text based data to enhance your Large Language Model (LLM). This technique is commonly referred to as LLM + RAG. This tutorial is heavily inspired by content from [pixegami](https://www.youtube.com/watch?v=2TJxpyO3ei4).

## Technical details of LLM + RAG

### What problem does RAG solve?

Standard LLMs do not have access to any outside data when generating responses to users, which is quite problematic when we want to augment our LLM to work with new information (e.g. breaking news).

One possible way to add this data is to continuously finetune our LLM (e.g. [LoRA](https://arxiv.org/abs/2106.09685)), but this is incredibly expensive, difficult, and may require downtime for an application. Often times, it may also not even be necessary as there's still a chance of hallucination.

RAG is designed with this in mind - being fast to add information without downtime, easy to deploy, and does not require significant compute.

### How does RAG work conceptually?

Think of a student as an LLM. That student will take an exam--think of the exam questions as a user's queries. The student is also allowed to build a cheatsheet before the exam.

When the student takes the exam (i.e. answers a query), they will look to their cheatsheet, and add in relevant information, potentially citing information.

Consider an example question of "What major event happened today?".

The student can "augment" the question by adding in information from their cheatsheet and transform it into something like "CONTEXT: Stanford researchers announce creation of an Artificial General Intelligence | QUESTION: What major event happened today?". This makes it much easier to answer the question.

The mechanics are slightly different for LLMs, but the concept is the same.

### How does RAG work mechanistically?

RAG enhances our LLM by passing the user's query to a "vector database", returning potentially relevant information, and then "augmenting" our user's query to cite this relevant information.

The vector database is key here, because when our system performs lookups, we are embedding the question in a vector space and looking for similar values near it. These similar values are then used to augment our prompt like the example above: "CONTEXT: Stanford researchers announce creation of an Artificial General Intelligence | QUESTION: What major event happened today?" (note, we can add as many entries as fits within our LLM's context length).

Our vector database is filled up with files or documents we give it. This is done by first chunking the data (making smaller chunks of the data) and then creating embeddings for it (vector representations) to place it in the database. The database can be updated anytime and independently of the LLM (for our purposes).

Simple right? Let's dive into the project!

## Project Overview

In this project, you will build a slang translator using Llama 3.1 and RAG. We will be using [this dataset](https://www.kaggle.com/datasets/therohk/urban-dictionary-words-dataset?resource=download) as our slang data.

We encourage you to use any model and data of your choice! [Ollama](https://ollama.com/library) is a great and easy place to try out various models, and [Kaggle](https://www.kaggle.com/datasets) is a great place to look for datasets.

# Getting Started

## Accessing your cluster

SSH into your assigned cluster.

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

## Download Ollama

Ollama is our model manager, and will make development using specific models significantly easier.

```
curl -fsSL https://ollama.com/install.sh | sh
```

Next, we need to run our Ollama locally. This will download the 8B model.

```
ollama run llama3.1
```

You may even want to try the 70B model if your GPU has ~40GB of VRAM because it uses 4 bit quantization (basically reducing VRAM needed to run on a GPU). The P100s on your cluster have 16 GB of VRAM. All Llama 3.1 models were originally quantized with fp16, and if we use 4 bit quantization then we can expect to save ~75% (4/16 = 25% memory footprint) or about 10GB of our P100 card.

```
ollama run llama3.1:70b
```

Once your model is downloaded, Ollama will run on `localhost:11434`.

This is what we'll be using to query our model

######

Create a file called `jupyter_submit.slurm`, and put in the following.

```
#!/bin/bash
#SBATCH --job-name="Jupyter"                # Job name
#SBATCH --mail-user=<email>@stanford.edu    # Email address
#SBATCH --mail-type=NONE                    # Mail notification type (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition=gpu-pascal              # Node partition, use gpu-pascal for Nvidia Pascal architecture gpus
#SBATCH --nodes=1                           # Number of nodes requested
#SBATCH --ntasks=1                          # Number of processes
#SBATCH --time=03:00:00                     # Time limit request

source ~/codes/python/python-venv/bin/activate
EXEC_DIR=$HOME/ME344
# hostname && apptainer exec --nv container.img jupyter-notebook --no-browser --port=9999 --notebook-dir=$EXEC_DIR
# apptainer exec container.img jupyter nbconvert final_project_train.ipynb --to python
apptainer exec --nv container.img python3 final_project_train.py
```

Launch the job with `sbatch jupyter_submit.slurm`. Note you may need to cancel any other pending jobs with `scancel <job-id>`.

This will produce a slurm-<job-id>.out. Then you can run the following command in a separate window:

`ssh -L 8888:localhost:8888 student@hpcc-cluster-[C] -t ssh -L 8888:localhost:8888 compute-1-1`.

Then you can find the port forwarded address of the jupyter notebook with `egrep -w 'compute|localhost'  slurm-*.out`. Now you should be able to connect with the cluster and run jobs.

From here you can open up rag.ipynb and run through the cells.

Come back when you finished setting up your RAG pipeline.

## Serve locally

Now we want to serve our model locally. To do this, we have provided an easy to interact with frontend so we can test out our model.

cd into `chat-gpt-clone`

To start the development server, run `npm install` and then run `npm start` this will create your dev site on `localhost:3000`.

We need to port forward port 3000 to our local 3000, so we can do that by doing the following:

`ssh -L 3000:localhost:3000 student@hpcc-cluster-[C] -t ssh -L 3000:localhost:3000 compute-1-1`

We'll also want to port forward the Ollama config to make sure that we are using the proper backend.

`ssh -L 11434:localhost:11434 student@hpcc-cluster-[C] -t ssh -L 11434:localhost:11434 compute-1-1`

If you navigate to `localhost:3000` on your computer, you'll see the chat interface!

Try playing around with the chat interface and even try making changes to the system prompt in `Chat.tsx`.

## Look at metrics

You'll be able to use `htop` in your cluster to see how demanding the usage is whenever you submit a query to your model, you should see high GPU usage from your python program. This indicates that you're using your GPU!

We encourage you to try out various different models, parameter sizes, and data to see how this effects the GPU usage.
