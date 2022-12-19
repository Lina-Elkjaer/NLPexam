import os
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
import seaborn as sns
sns.set_theme()

def format_data(data: pd.DataFrame, groups: list[str]):
    df = data.groupby(groups[0])[groups[1]].value_counts()
    df = df.unstack(level=[1])
    return df

def get_volume(data: pd.DataFrame, groups: list[str], plot_title: str, outpath: str, file_name: str):
    data['date'] = pd.to_datetime(data['date'])
    df = data.groupby(groups[0])[groups[1]].value_counts()
    df = df.unstack(level=[1]).T
    
    # plot
    plt.figure()
    df.plot()
    plt.title(plot_title)
    plt.xlabel('date')
    plt.ylabel('document count')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=True)

    # save
    save_summary(df, file_name, outpath)

def get_language(data: pd.DataFrame, groups: list[str], outpath: str):
    df = format_data(data, groups)
    file_name = f'language_{groups[0]}'

    # get table of top language percentages across subreddits
    en_ru_uk = df[["en","ru","uk"]]
    x = data[groups[0]].value_counts()
    df = pd.concat([en_ru_uk, x.rename("total")], axis=1)

    # find percentages of top languages
    en_ru_uk['en_percent'] = df['en'] / df['total'] * 100
    en_ru_uk['ru_percent'] = df['ru'] / df['total'] * 100
    en_ru_uk['uk_percent'] = df['uk'] / df['total'] * 100

    # round decimals
    df = en_ru_uk.round(decimals = 2)

    # plot
    plot_df = df[['en_percent', 'ru_percent', 'uk_percent']]

    plt.figure()
    plot_df.plot(kind = "bar")
    plt.title('Languages by subreddit')
    plt.xlabel('Subreddit')
    plt.ylabel('Language in %')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=True)

    # save
    save_summary(df, file_name, outpath)

def get_doc_length(data: pd.DataFrame, outpath: str, file_name: str):
    data["doc_length"] = data["document"].apply(lambda x: len(x.split()))
    df = data.groupby("sub_reddit")["doc_length"].describe()
    
    # plot
    plt.figure()
    df['mean'].plot(kind = "bar")
    plt.title('Mean document length per subreddit')
    plt.xlabel('subreddit')
    plt.ylabel('Mean document length')
    plt.legend(fancybox=True, shadow=True)
    
    # save
    save_summary(df, file_name, outpath)

def get_types(data: pd.DataFrame, outpath: str):
    df = format_data(data, groups = ['sub_reddit', 'type'])
    df = pd.DataFrame(df)

    # plot
    plt.figure()
    df.plot(kind = 'bar', stacked = True)
    plt.title('Number of document type pr. subreddit')
    plt.xlabel('Subreddit')
    plt.ylabel('document count')
    plt.legend(fancybox=True, shadow=True)

    # save
    df['total'] = df['comment'] + df['submission']
    save_summary(df, file_name = 'doc_types', outpath = outpath)

def save_summary(df: pd.DataFrame, file_name: str, outpath: str):
    outfile = os.path.join(outpath, file_name)
    plt.savefig(f'{outfile}.png', dpi = 300, bbox_inches = 'tight')
    df.to_csv(path_or_buf = f'{outfile}.csv', header=True, index=True, sep='\t', mode='a')

def get_summaries(data: pd.DataFrame, outpath: str):
    # volume summaries
    volume_groups = [['type', 'date'], ['sub_reddit', 'date']]
    for groups in volume_groups:
        file_name = f'volume_{groups[0]}'
        get_volume(data = data, groups = groups, plot_title = f'Number of documents pr. day by {groups[0]}', file_name = file_name, outpath = outpath)

    # document length summaries
    get_doc_length(data, outpath = outpath, file_name = 'document_length')

    # language summaries
    language_groups = ['sub_reddit', 'language']
    get_language(data, language_groups, outpath)

    # document type summaries
    get_types(data, outpath)