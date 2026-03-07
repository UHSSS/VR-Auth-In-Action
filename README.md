# Authentication in Action: Interaction-Triggered Continuous User Authentication for Controller-Free Virtual Reality

This repository contains data and code for analysis of collected data.

## Running the Code
Python 3.11 or greater is recommended. Lower versions may work but have not been tested.

Use python to create a virtual environment (or you can use Conda).
```bash
python -m venv env
# Activate virtual env on Windows
env/Scripts/activate
# Activate virtual env on Linux/Mac
source env/bin/activate
```

Install required packages into the virtual environment once activated.
```bash
pip install -r requirements.txt
```

With your environment now set up, you can run the analysis code on the dataset. `VR_Auth.py` and a specifically organized `Data/` folder is required.

```bash
python VR_Auth.py
```

## Data Folder Format
Data folder is organized by user, and then by trial. 
```bash
Data/
-- 1/ # (User 1)
---- 1/ # (User 1, trial 1)
---- 2/ # (User 1, trial 2)
...
-- 2/ # (User 2)
...
```
Each trial folder should contain a `segmented_headset_data.csv`. This contains VR telemetry for each frame during the experiment, including the headset and hands transforms (three dimensional positional and rotational coordinates), finger curl information, and which object the user was interacting with at a given moment (`flag`)