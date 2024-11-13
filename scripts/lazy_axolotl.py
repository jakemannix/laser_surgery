import runpod
import requests
import json
import yaml
from google import userdata
import os
import requests

"""
# @title ## Training parameters
GPU = "NVIDIA GeForce RTX 3090" # @param ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA A30", "NVIDIA A40", "NVIDIA GeForce RTX 3070", "NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 3080 Ti", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090 Ti", "NVIDIA GeForce RTX 4070 Ti", "NVIDIA GeForce RTX 4080", "NVIDIA GeForce RTX 4090", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA L4", "NVIDIA L40", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX 4000 SFF Ada Generation", "NVIDIA RTX 5000 Ada Generation", "NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX A2000", "NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX A5000", "NVIDIA RTX A6000", "Tesla V100-FHHL-16GB", "Tesla V100-PCIE-16GB", "Tesla V100-SXM2-16GB", "Tesla V100-SXM2-32GB"]
NUMBER_OF_GPUS = 1 # @param {type:"slider", min:1, max:8, step:1}
CONTAINER_DISK = 75 # @param {type:"slider", min:50, max:500, step:25}
CLOUD_TYPE = "COMMUNITY" # @param ["COMMUNITY", "SECURE"]
SCRIPT = "https://gist.githubusercontent.com/mlabonne/9d85e1bb8fc3efe8649b677845c83bdb/raw" # @param {type:"string"}
LLM_AUTOEVAL = True # @param {type:"boolean"}
DEBUG = False # @param {type:"boolean"}

# @markdown ---

# @markdown ## Tokens
# @markdown Enter the name of your tokens in the Secrets tab.
USERNAME = "mlabonne" # @param {type:"string"}
RUNPOD_TOKEN = "runpod" # @param {type:"string"}
HUGGING_FACE_TOKEN = "HF_TOKEN" # @param {type:"string"}
WANDB_TOKEN = "wandb" # @param {type:"string"}
GITHUB_TOKEN = "github" # @param {type:"string"}

# Environment variables
runpod.api_key = userdata.get(RUNPOD_TOKEN)
WANDB_API_KEY = userdata.get(WANDB_TOKEN)
HF_TOKEN = userdata.get(HUGGING_FACE_TOKEN)
GITHUB_API_TOKEN = userdata.get(GITHUB_TOKEN)

"""


def get_summary(yaml_config):
    config = yaml.safe_load(yaml_config)
    base_model = config.get('base_model', 'Unknown model')
    dataset_info = []
    datasets = config.get('datasets', [])
    for dataset in datasets:
        path = dataset.get('path', 'Unknown path')
        dtype = dataset.get('type', 'Unknown type')
        dataset_info.append(f"{path} ({dtype})")
    datasets_summary = ', '.join(dataset_info)
    print(f"This runs trains {base_model} on {datasets_summary}.")


def get_var(var):
    return os.getenv(var, userdata.get(var))


# TODO: scripts and gist stuff.  Run from local, sure.  But maybe Maxime had it right, and UX should be Colab?
def create_pod(config, runpod_config, SCRIPT, gist_url):
    runpod.api_key = get_var("RUNPOD_TOKEN")
    pod = runpod.create_pod(
        name=f"LazyAxolotl - {runpod_config['model']}",
        image_name="winglian/axolotl-runpod:main-py3.10-cu118-2.0.1",
        gpu_type_id=runpod_config['gpu'],
        cloud_type=runpod_config['cloud_type'],
        gpu_count=runpod_config['num_gpus'],
        volume_in_gb=0,
        container_disk_in_gb=runpod_config['container_disk'],
        template_id="eul6o46pab",
        env={
            "HF_TOKEN": get_var("HF_TOKEN"),
            "SCRIPT": SCRIPT,
            "WANDB_API_KEY": get_var("WANDB_API_KEY"),
            "GIST_URL": gist_url,
            "MODEL": runpod_config("model"),
            "BASE_MODEL": config['base_model'],
            "USERNAME": runpod_config("username"),
            "LLM_AUTOEVAL": runpod_config("llm_autoeval"),
            "BENCHMARK": "nous",
            "GITHUB_API_TOKEN": get_var("GITHUB_API_TOKEN"),
            "TRUST_REMOTE_CODE": True,
            "DEBUG": runpod_config("debug"),
        }
    )
    print("https://www.runpod.io/console/pods")
    return pod
