# FaaSterConfig
Automatic resource configuration generator for FaaS functions. 

This repo contains jupyter notebooks, openFaas functions, and script(s) to generate configs and interact with OpenFaas to determine which config is the best for a given function 

## Setup 

Lots of methods here will make use of various libraries, its good to just install everything from the requirements.txt file (optionally create a pyenv first)
```bash
pip install -r requirements.txt
```
### Notebooks

Start testing things with existing or create new jupyter notebooks and store them in the notebooks directory

### OpenFaaS

If you want to test out some of the functions by themselves, install faas-cli and use it to upload/run the functions in the openFaas directory

on mac: 

```bash
brew install faas-cli
cd openFaas
faas-cli local-run greeter
```

in a new terminal, send a curl request locally: 

```bash
curl http://0.0.0.0:8080 -X POST  --data 'ECHO'
```
This should return Echo to the terminal 

### FaaSterConfig

Still in progress, but run

```bash
cd FaaSterConfig
python3 FaaSterConfig.py
```

To generate a new config file (openfaas .yml files) based on an existing function which will duplicate that function many times with different configurations. 

Eventually, this will automatically run these configurations and report back the time and cost and provide an optimal set of configurations to choose from.

