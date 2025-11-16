# Basic structure of files
/app/modal_app.py - main driver script to run the training on Modal
/app/train_ddp_stream.py - handles the data distribution of the training dataset among different workers - parallel training
/app/dataset_chess.py - handles loading logic of the dataset to prevent memory overload - extends from a IterableDataset object to allow for traversing of the dataset
Data is loaded in batch_size amount into memory

# Setting up Modal
On your local machine, run
```
pip install modal
python3 -m modal setup
```
# Sign up for Modal account
1. Go to modal.com and sign up using GitHub

# Setting up Modal Workspaces
1. In your account dashboard, go to workspaces
2. To create a workspace that more than one person can collaborate on, click on create a workspace
3. You will have to attach an payment method
4. On your new workspace, you can manage members and add
5. Copy the commands prompted in your local terminal after setting up workspace

# Setting up Modal Volumes
1. Once activated workspace from above, use the following commands to set up this project
```
modal volume create chess-datasets
modal volume create chess-code
modal volume create chess-checkpoints

modal volume put chess-code /app/ /app/ <-- copy the app module into volume
modal volume put chess-datasets <your_first_pgn_dataset_file>
modal volume put chess-datasets <your_nth_pgn_dataset_file>
modal volume ls <volume_name> <-- to view files in volume
modal volume list <-- list all volumes
modal volume rm <volume_name> <-- remove file / directory in volume
```

With the ``` modal_app.py ``` located one level above /app, run the following in this directory (i.e., the directory of modal_app.py)
```
modal deploy modal_app.py <-- handles env setup and package installation
modal run modal_app.py <-- starts training
```
