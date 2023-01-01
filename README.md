# BERTopic modeling on the 2022 Russian-Ukranian War
A Reddit Dataset for the Russo-Ukrainian Conflict in 2022

Welcome to our reposity used for BERTopic modeling on a reddit dataset for the Russian-Ukranian War in 2022.
The modeling is based on the dataset described in [this paper](https://arxiv.org/abs/2206.05107) using the [BERTopic package](https://github.com/MaartenGr/BERTopic/tree/master/bertopic). We performed the variations of BERTopic known as dynamic topic modeling and topics per class on a downsampled version of the original dataset containing 326884 documents (Reddit submissions and comments). Dynamic topic modeling was performed on a multilingual model as well as for three languages separately (English, Ukranian and Russian). Class-based modeling was performed per language and per subreddit for the multilingual model. We encourage exploration of our interactive [plotly](https://github.com/plotly) plots.

## Investigate the interactive plots!
[Topics over time for all languages (global model)](https://htmlpreview.github.io/?https://github.com/Lina-Elkjaer/NLPexam/blob/main/out/topics_over_time_all.html)

[Topics over time for english](https://htmlpreview.github.io/?https://github.com/Lina-Elkjaer/NLPexam/blob/main/out/topics_over_time_eng.html)

[Topics over time for russian](https://htmlpreview.github.io/?https://github.com/Lina-Elkjaer/NLPexam/blob/main/out/topics_over_time_russian.html)

[Topics over time for ukranian](https://htmlpreview.github.io/?https://github.com/Lina-Elkjaer/NLPexam/blob/main/out/topics_over_time_ukrainian.html)

[Topics per language](https://htmlpreview.github.io/?https://github.com/Lina-Elkjaer/NLPexam/blob/main/out/topics_per_language_all.html)

[Topics per subreddit](https://htmlpreview.github.io/?https://github.com/Lina-Elkjaer/NLPexam/blob/main/out/topics_per_subreddit_all.html)

## Project Organization
The organization of the project is as follows:
```

├── GPU                       <- Stores the zip-folder that was used for GPU acceleration attempts.
├── archive                   <- Previous notebooks to test functions, installations and datastructures.
├── data                      <- The folder in which the preprocessed dataframe is saved.
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

## How to run the code
To replicate our analysis first clone this repository. Then type
```
bash run.sh
```
to install the neccesary packages and create the virual environment nlp-env. Subsequently, make sure to activate the environment by typing
```
source ./nlp-env/bin/activate
```
You are now ready to download the raw data from [here](https://github.com/James-ZYM/RussiaUkraineConflictDataset) and place the Comments and Submissions folders within the raw folder.
use main.py to perform preprocessing and generate the dataframe that can be used to perform BERTopic modeling
```
python3 src/main.py
```
Finally run the bertopic scripts to train the model and generate the html-files containing the interactive plots
```
python3 src/bertopic_all.py
python3 src/bertopic_eng.py
```




