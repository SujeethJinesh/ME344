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

## Enabling Internet on your GPU cluster

Reach out to Professor Jones to enable internet on your cluster, otherwise you will not be able to download models and use the internet.

## Accessing your cluster

SSH into your assigned cluster as root. It's important to be a root user because we will need to install Ollama, which requires root access.

## Download requirements

### Download Ollama

[Ollama](https://ollama.com/download/linux) is our model manager, and will make development using specific models significantly easier, make sure you are running as root, and then execute the following command to download Ollama.

```
curl -fsSL https://ollama.com/install.sh | sh
```

Now switch back to the student user by running `su - student`. The rest of these commands will be run as the `student` user.

### Create a virtual environment

While running as the `student` user, you will want to create a virtual environment to manage dependencies.

If you have not already created a virtual environment before, paste the following into your terminal. This will create a python3.11 virtual environment called "python-venv", and will activate the virtual environment.

```
# Create a folder for your python environment
mkdir -p ~/codes/python && cd ~/codes/python

# Create the actual python environment & call it "python-venv"
python3.11 -m venv python-venv

# Activate your python environment right now.
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

Next, we need to run our Ollama locally. This will download the 8B model.

```
ollama run llama3.1
```

You may even want to try the 70B model if your GPU has ~40GB of VRAM because it uses 4 bit quantization (basically reducing VRAM needed to run on a GPU). The P100s on your cluster have 16 GB of VRAM. All Llama 3.1 models were originally quantized with fp16, and if we use 4 bit quantization then we can expect to save ~75% (4/16 = 25% memory footprint) or about 10GB of our P100 card.

```
ollama run llama3.1:70b
```

You may notice that Ollama doesn't detect any GPUs, and that's fine. We'll fix that later.

After the model is downloaded, you'll get a prompt to start interacting with the model. Go ahead and prompt it!

Whenever you want to exit the chat, you can type `/bye` to end the conversation.

Ollama will also be running on `localhost:11434` in the background.

This localhost address is what we'll be using to query our model.

### Downloading Embedding Model

We'll be using a special embedding model for our vector database. In order for us to get that model we need to pull it from ollama. Run the command `ollama pull nomic-embed-text`.

### Starting our Vector Database

We'll want to create our vector database and have it run locally so we can store our documents there. To do this, start it in a new terminal with `chroma run --host localhost --port 8000 --path ./chroma`

In order for us to be able to access this vector database locally as well, we'll want to port forward the vector database.

`ssh -L 8000:localhost:8000 student@hpcc-cluster-[C] -t ssh -L 8000:localhost:8000 compute-1-1`

## Implementing RAG

### Creating a SLURM script

Use `pip` to install the package named `notebook`.

You'll want to clone the directory for this project for convenience (TODO: Make the repository called project-5).

Change to the `project-5` directory.

You should also install all the requirements for the rest of the project using pip. You can do this by doing `pip install -r requirements.txt`

Create a Slurm script to launch Jupyter Notebook on compute-1-1. Make sure you are in the `$HOME/project-5` directory (command to execute is pwd). Name the script `jupyter_submit.slurm`:

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
EXEC_DIR=$HOME/project-5
hostname && jupyter-notebook --no-browser --notebook-dir=$EXEC_DIR
```

Launch the job with `sbatch jupyter_submit.slurm`. Note you may need to cancel any other pending jobs with `scancel <job-id>`.

This will produce a slurm-<job-id>.out. Then you can run the following command in a separate window:

`ssh -L 8888:localhost:8888 student@hpcc-cluster-[C] -t ssh -L 8888:localhost:8888 compute-1-1`.

Then you can find the port forwarded address of the jupyter notebook with `egrep -w 'compute|localhost'  slurm-*.out`.

The output will look something like this:

```
...
slurm-103.out:compute-1-1
slurm-103.out:[I 18:40:29.846 NotebookApp] http://localhost:8888/?token=c11be5afa5cddd73548d8ff73786291202a37868d5c18451
slurm-103.out:        http://localhost:8888/?token=c11be5afa5cddd73548d8ff73786291202a37868d5c18451
```

Now you should be able to connect with the cluster and run jobs by following the link. In this case it's `http://localhost:8888/?token=c11be5afa5cddd73548d8ff73786291202a37868d5c18451`.

From here you can copy the notebook from [here (TODO ADD LINK)]() named `rag.ipynb` and run through the cells.

Come back when you finished setting up your RAG pipeline!

## Serve locally

Now that we've updated our vector database, we want our model to use that when asked for queries. To do this, we will make a request to our vector database when the user gives us a query, append that information we received from the database, and then send that whole prompt off to the model!

We have provided an easy to interact with frontend so we can test out our model.

cd into `chat-gpt-clone`

To start the development server, run `npm install` and then run `npm start` this will create your dev site on `localhost:3000`.

We need to port forward port 3000 to our local 3000, so we can do that by doing the following:

`ssh -L 3000:localhost:3000 student@hpcc-cluster-[C] -t ssh -L 3000:localhost:3000 compute-1-1`

We'll also want to port forward the Ollama config to make sure that we are using the proper backend.

`ssh -L 11434:localhost:11434 student@hpcc-cluster-[C] -t ssh -L 11434:localhost:11434 compute-1-1`

If you navigate to `localhost:3000` on your computer, you'll see the chat interface!

Try playing around with the chat interface. This chat interface is missing a key feature though, it's connected to your backend model but does not actually use any of the data we stored in our vector database.

To use our vector database, we have a little bit of work to do. You will see that there's a file called `Rag.tsx`, this file is where you'll be making some small code changes.

We will update the code to connect to our backend vector database, get our embedding data, and augment it to our user generated query.

# TODO finish this explanation

Lastly, one way to tune the model is to try making changes to the system prompt in `Chat.tsx`. You can instruct the model to behave in some way.

## Look at metrics

You'll be able to use `htop` in your cluster to see how demanding the usage is whenever you submit a query to your model, you should see high GPU usage from your python program. This indicates that you're using your GPU!

We encourage you to try out various different models, parameter sizes, data, and system prompts to see how this effects the GPU usage, accuracy, speed, etc!

We're excited to see how you choose to adapt this code!
