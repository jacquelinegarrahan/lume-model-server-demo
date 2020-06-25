# Lume Model EPICS Server Demo

This repository contains a demo for serving a lume model using EPICS. 

## Installation
The environment for this project is managed using conda. The environment can be created directly from the environment.yml file.

`$ conda env create -f environment.yml`

`$ conda activate lume-model-server-demo`

## Run
First set up the kernel:

` $ python -m ipykernel install --user --name=lume-model-server-demo`

Once that is complete, launch the notebook:

` $ jupyter notebook Demo.ipynb`