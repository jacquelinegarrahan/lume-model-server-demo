# Jupyter notebook demo of lume-model and lume-epics

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jacquelinegarrahan/lume-model-server-demo/binder?urlpath=/proxy/5006/surrogate_model_client)

This repository contains demo notebooks for serving online models using the [lume-model](https://github.com/slaclab/lume-model) and [lume-epics](https://github.com/slaclab/lume-epics). A simple demo is included outlining an implementation of a minimal model using the tools, as well as an example of an more complicated surrogate model.

A conda installation is required for setting up the required packages and the code is executed inside jupyter notebooks.

## Clone the demo repository

``` $ git clone https://github.com/jacquelinegarrahan/lume-model-server-demo.git ```

## Navigate to the demo repository

``` $ cd lume-model-server-demo ```

## Set up and activate conda environment

Create the environment using conda and the environment.yml file included with the repository:

``` $ conda env create -f environment.yml ```

Activate the environment:

``` $ conda activate lume-model-server-demo```

## Set up the ipython kernel

```$ python -m ipykernel install --user --name=lume-model-server-demo ```

## Launch Jupyter

``` $ jupyter notebook ```

## Run simple demo

In two tabs, open the `SimpleServer` and `SimpleClient` notebooks. Begin by following the code outlined in the `SimpleServer` notebook, and then execute the code in the `SimpleClient` to render the application. Finally, terminate the server using the `Server.stop()` method in the `SimpleServer` notebook.

## Run surrogate model demo

The surrogate model demo uses the `MyModel` python class defined in the `MySurrogateModel.py`. The `MyModel` class uses a custom implementation to load a Keras model in __init__ and executes the model in the `evaluate` method. The model is loaded using an hdf5 file containing the model weights and architecture.

In two tabs, open the `SurrogateModelServer` and `SurrogateModelClient` notebooks. Begin by following the code outlined in the `SurrogateModelServer` notebook, and then execute the code in the `SurrogateModelClient` to render the application. Finally, terminate the server using the `Server.stop()` method in the `SurrogateModelServer` notebook.
