#!/usr/bin/bash

# set up virtual environment
sudo apt-get update -y
sudo apt-get install python3.9-dev -y
sudo apt-get install python3-venv -y
python3.9 -m venv nlp-env
source ./nlp-env/bin/activate

# prettier outputs
GREEN='\033[1;32m'
NC='\033[0m'

# install packages
pip install pandas
pip install bertopic
pip install fasttext
pip install matplotlib
pip install seaborn
pip install stop_words
#pip install ipykernel
#python -m ipykernel install --user --name==nlp-env

# done
echo -e "[${GREEN}INFO:${NC}] Everything installed!"
