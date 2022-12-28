# BERTopic modeling on the 2022 Russian-Ukranian War
A Reddit Dataset for the Russo-Ukrainian Conflict in 2022

Welcome to our reposity used for BERTopic modeling on a reddit dataset for the Russian-Ukranian War in 2022.
The modeling is based on the dataset described in [this paper](https://arxiv.org/abs/2206.05107) using the [BERTopic package](https://github.com/MaartenGr/BERTopic/tree/master/bertopic). We performed the variations of BERTopic known as dynamic topic modeling and topics per class on a downsampled version of the original dataset containing XX documents (Reddit submissions and comments). Dynamic topic modeling was performed on a multilingual model as well as for three languages separately (English, Ukranian and Russian). Class-based modeling was performed per language and per subreddit for the multilingual model. We encourage exploration of out interactive [plotly](https://github.com/plotly) plots.

## Project Organization
The organization of the project is as follows:
```
├── data                      <- The folder in which the preprocessed dataframe is saved.
├── GPU                       <- Stores the zip-folder that was used for GPU acceleration attempts
├── models                    <- Stores the pre-trained embedding model
├── nbs                       <- Notebooks used for BERTopic modeling on Russian and Ukranian subsets
|   ├── model_russian.ipynb
|   └── model_ukraine.ipynb
├── out                      <- Stores the outputs from BERTopic modeling
├── raw                      <- Folder containing the raw data
├── src 
|   ├── bertopic_all.py      <- Script to run the multilingual model
|   ├── bertopic_eng.py      <- Script to run the english model
|   ├── data.py              <- Script of data pre-processing functions
|   ├── main.py              <- Main script
|   ├── summary.py           <- Script of data summary staticstic functions
│   └── viz_functions.py     <- Script of the main visualization functions
├── run.sh                   <- Setup script that downloads packages and creates the environment
├── .gitignore               <- A list of files not uploaded to git
└── README.md                <- Presentation of the repository
```




