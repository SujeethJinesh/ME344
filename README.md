# ME344

## Steps For Finetuning Model

Ssh into your assigned cluster.

Execute `su - student` to switch to the student user.

Create a file called `jupyter_submit.slurm`, and put in the following.

```
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
```

Launch the job with `sbatch jupyter_submit.slurm` This will launch an apptainer with `final_project_train.py`. Note you may need to cancel any other pending jobs with `scancel <job-id>`.

This will produce a slurm-<job-id>.out. Then you can run the following command in a separate window:

`ssh -L 8888:localhost:8888 student@hpcc-cluster-[C] -t ssh -L 8888:localhost:8888 compute-1-1`.

Then you can find the port forwarded address of the jupyter notebook with `egrep -w 'compute|localhost'  slurm-*.out`. Now you should be able to connect with the cluster and run jobs.

From here you can open up final_project_train.ipynb and run all the cells.

If you wish to use your own custom data, check out `gen_z_slangs_translations.json` and make sure your data fits in that format!

## Steps For Website

cd into `chat-gpt-clone`

To start the development server, run `npm install` and then run `npm start` this will create your dev site on `localhost:3000`.

If you navigate to the site `localhost:3000`, you'll see the chat interface! But if you try inputting any prompts into it, it won't work just yet.

## Steps for Backend

cd into `torchchat`, which is taken from [here](https://github.com/pytorch/torchchat)

You'll need to log in to huggingface to access models. You may also need to accept agreements for specific models like [Llama 3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).

Use this command to do so `huggingface-cli login`

Then run `./install_requirements.sh`. This will install the necessary requirements.

Then you can download llama 3.1 with `python3 torchchat.py download llama3.1`.

This requires a little under 20 GB of disk space.

After this you'll be able to run `python3 torchchat.py chat llama3.1` to chat with llama or `python3 torchchat.py generate llama3.1 --prompt "write me a story about a boy and his bear"` to generate a response.

We will use the server feature.

Open a separate terminal and run `python3 torchchat.py server llama3.1` This will run llama on port 5001.

TODO: We need to figure out how to allow you to use your own model as part of the backend

## Putting it together.

Once you have your frontend server running on port 3000 and your backend server running on port 5001, you should be able to send chats and see your response!

You'll want to explore and see how to change the model so you can use that, and play around with `Chat.tsx` to see if you can prompt engineer things for your own use case!
