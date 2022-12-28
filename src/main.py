import os
import pandas as pd
from data import *
import fasttext
from summary import *

# running this script took 7.65 hrs on all comments (9225973 rows) on 64 CPUs

def main():

    cwd = os.path.dirname(os.path.abspath(__file__))

    dirs = [os.path.join(".." , "raw")]
    file_list = get_files(current_path = cwd, data_dirs = dirs)
    pretrained_lang_model = os.path.join("..", "models", "lid.176.ftz")
    ft_model = fasttext.load_model(pretrained_lang_model)

    #file_list = file_list[:50] + file_list[1320:1370]
    #file_list = file_list[1320:1322]

    make_df(file_list, save = 0, ft_model = ft_model)

    # summarize data
    data = pd.read_pickle(os.path.join('..','data','full_df.pkl'))
    outpath = os.path.join('..', 'out')

    get_summaries(data, outpath)

if __name__ == "__main__":
    main()