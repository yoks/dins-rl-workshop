# DINS Reinforced Learning Workshop project
This is testing RL environment for DINS RL Workshop. Repository contains basic scaffold and setup guide.

## Setup
Environment can be setup on Linux, MacOS or Windows (using linux subsystem) machines.
Guide assumes Ubuntu 16.04-LTS is used.

For Windows, XServer should be installed (like `Xming`)

`Python 3` should be installed on machine as pre-requisite
#### Step 1
Install Python Virtual Env.
```bash
pip install virtualenv
pip install virtualenvwrapper
export WORKON_HOME=~/Envs
source /usr/local/bin/virtualenvwrapper.sh
mkdir -p $WORKON_HOME
# Create virtual env
mkvirtualenv dins-workshop
```
#### Step 2
Install dependencies:
```bash
# absl
pip install absl-py
# Numpy
pip install numpy
# OpenCV Python
pip install opencv-python
# OpenAI gym
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```
#### Step 3
Install OpenAI baselines
```bash
sudo apt-get install libcr-dev mpich2 mpich2-doc zlib1g-dev cmake python-opencv
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```
#### Step 4
Install fceux and super-mario-gym environment
```bash
sudo apt-get install fceux
git clone https://github.com/yoks/gym-super-mario.git
pip install -e .
```
## Training
```bash
python train.py
```
## Running
```bash
python run.py
```
