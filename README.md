# AGT-MOE

# step-1 Mlflow setup

```bash
mlflow server --host 127.0.0.1 --port 8080
```



# setting up process on linux server 

1. I have created a python environment and run the jupyter notebook there. 

```bash
# get a gpu node allocated to you
# step-1 
srun --partition=gpu-all --gres=gpu:1 --cpus-per-task=1 --mem=16G --time=00:10:00 --pty bash -i
# activate the environment
source ~/moe_venv/bin/activate
# host the notebook
jupyter notebook --no-browser --port=6666 --ip=0.0.0.0
# note the url like this: http://gpu3:8888/tree?token=cc9a3d22a8bb5e80b514c457cc841b63d1411bbbf17f9530

# Open a new terminal in local and setup ssh tunnel to access the jupyter notebook - make sure the port is free
ssh -N -L 6666:localhost:6666 -J higoyal@linux7.cs.uchicago.edu higoyal@gpu3



# Step-2: do the same setup for the mlflow 
# do this from the terminal of jupyter notebook
mlflow server --host 0.0.0.0 --port 7777
ssh -N -L 7777:localhost:7777 -J higoyal@linux7.cs.uchicago.edu higoyal@gpu3
# go to http://localhost:7777 - for mlflow server


# optional- if not using the srun, just run this 
ssh -N -f -L 6666:localhost:6666 higoyal@linux7.cs.uchicago.edu

```

