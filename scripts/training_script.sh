#!/bin/bash

WORKSPACE=${WORKSPACE:-"/workspace"}
cd $WORKSPACE

# Install Axolotl and dependencies
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip3 install packaging huggingface_hub
pip3 install -e '.[flash-attn,deepspeed]'

# Install screen
apt install -y screen
screen -S main

# Train model
curl -o config.yaml $GIST_URL
mkdir out
accelerate launch -m axolotl.cli.train config.yaml

# Merge adapter
# CUDA_VISIBLE_DEVICES="" python3 -m axolotl.cli.merge_lora config.yaml --lora_model_dir=out

# Merge and upload model
# this is currently saved in merge_upload.py in this directory
wget https://gist.githubusercontent.com/mlabonne/522857e8d2aaf647451a09705fceb275/raw/05ac6b01a733065d90938c955bbd4a9eb7121d74/merge_upload.py
python merge_upload.py --base_model=$BASE_MODEL --peft_model="./out/" --hub_id=$MODEL_NAME

# Upload merged model
huggingface-cli upload --repo-type model $MODEL_NAME $MODEL_NAME .

# LLM AutoEval
if [ "$LLM_AUTOEVAL" == "True" ]; then
    git clone https://github.com/mlabonne/llm-autoeval.git
    bash $(basename https://github.com/mlabonne/llm-autoeval.git .git)/runpod.sh
fi

# Kill pod
if [ "$DEBUG" == "False" ]; then
    runpodctl remove pod $RUNPOD_POD_ID
fi

sleep infinity
