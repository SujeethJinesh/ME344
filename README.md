# ME344

## Steps For Model

ssh into root user of assigned cluster

do `su - student` to switch to the student user.

cd into the ME344 folder.

launch job with `sbatch jupyter_submit.slurm`. Note you may need to cancel any other pending jobs with `scancel <job-id>`.

This will produce a slurm-<job-id>.out. Then you can run the following command in a separate window:

`ssh -L 8888:localhost:8888 student@hpcc-cluster-[C] -t ssh -L 8888:localhost:8888 compute-1-1`.

Then you can find the port forwarded address of the jupyter notebook with `egrep -w 'compute|localhost'  slurm-*.out`. Now you should be able to connect with the cluster and run jobs.

## Steps For Website

cd into `chat-gpt-clone`

To start the development server, run `npm install` and then run `npm start` this will create your dev site on `localhost:3000`
