# BERTopic modeling on the 2022 Russian-Ukranian War
A Reddit Dataset for the Russo-Ukrainian Conflict in 2022

Welcome to our reposity used for BERTopic modeling on a reddit dataset for the Russian-Ukranian War in 2022.
The modeling is based on the dataset described in [this paper](https://arxiv.org/abs/2206.05107) using the [BERTopic package](https://github.com/MaartenGr/BERTopic/tree/master/bertopic). We performed the variations of BERTopic known as dynamic topic modeling and topics per class on a downsampled version of the original dataset containing XX documents (Reddit submissions and comments). Dynamic topic modeling was performed on a multilingual model as well as for three languages separately (English, Ukranian and Russian). Class-based modeling was performed per language and per subreddit for the multilingual model. We encourage exploration of out interactive [plotly](https://github.com/plotly) plots.

## Project Organization
The organization of the project is as follows:

├── LICENSE                  <- the license of this code
├── README.md                <- The top-level README for this project.
├── .github            
│   └── workflows            <- workflows to automatically run when code is pushed
│   │    └── pytest.yml      <- A workflow which runs pytests upon push
├── ner                      <- The main folder for scripts
|   ├── archive              <- Folder containing early drafts of the scripts.
|   ├── data.py              <- A script containing functions for loading and preprocessing data.
|   ├── LSTM.py              <- A script containing classes and functions used for initializing the LSTM as well as plotting and computing the loss.
|   └── main.py              <- A script containing the main function in which the model is trained, validated, and tested.
├── out                      <- Folder containing .png files of the loss-curves and .txt files with the test metrics.
├── run.sh                   <- setup script to run main.py and download required packages
├── .gitignore               <- A list of files not uploaded to git
├── requirement.txt          <- A requirements file of the required packages.
└── assignment_description.md<- the assignment description





