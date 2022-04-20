"""
    Movie Recommendation System based on user profile
    Copyright 2022 Mahdie Rafiei. All Rights Reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import os.path
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_datasets(fixed_trainset_path, validation_set_path):
    fixed = pd.read_csv(fixed_trainset_path)
    fixed['label'] = (fixed['rate'] >= 7).map({False: 0, True: 1})
    dynam = pd.read_csv(validation_set_path)
    dynam['label'] = (dynam['rate'] >= 7).map({False: 0, True: 1})
    return fixed, dynam


def create_splits(fixed_trainset_path, validation_set_path):
    fixed, dynam = load_datasets(fixed_trainset_path, validation_set_path)
    # TBD: check Nan values for bssb-rs dataset
    # TBD: check feature's order in bssb-rs dataset
    # fixed = fixed.dropna()
    # dynam = dynam.dropna()

    print(len(fixed), len(dynam))
    print(fixed.film_id.nunique())
    print(dynam.film_id.nunique())
    print(len(set(fixed.film_id.unique()).intersection(dynam.film_id.unique())))

    return fixed, dynam


def create_paths(path: str, fixed_file: str, val_file: str, cluster: int or None):
    if cluster is not None:
        fixed_file = fixed_file.replace('.', f"{cluster}.")
        val_file = val_file.replace('.', f"{cluster}.")
    return f"{path}/data/{fixed_file}", f"{path}/data/{val_file}"


def load_and_split(fixed_path, val_path):
    train_set, test_set = create_splits(fixed_path, val_path)
    return train_set, test_set


def drop_columns(df):
    columns = list(set(df.columns).intersection({'Unnamed: 0', 'creation_year', 'early_bird', 'night_owl',
                                                 'prefered_hour', 'weekend_tweeter',
                                                 'prefered_weekday', 'friends_count',
                                                 'followers_count', 'favourites_count',
                                                 'tweets_count', 'geo_enabled'
                                                 }))
    return df.drop(columns=columns)


def typecast(df, columns):
    for column in columns:
        df[column] = df[column].astype(int)
    return df


def plot_skl_mtericts(df, metric, plot_name):
    avg = df.groupby(['cluster'])[metric].agg('sum').reset_index()
    # avg = typecast(avg, columns=['cluster', 'fold'])
    sns.lineplot(data=avg, x='cluster', y=metric)
    plt.savefig(plot_name)
    plt.close()


def plot_top_n(df, metric='recall', plot_name=''):
    avg = df.groupby(['cluster'])[metric].agg('sum').reset_index()
    # avg = typecast(avg, columns=['topn'])

    sns.lineplot(data=avg, x='cluster', y=metric)
    plt.savefig(plot_name)
    plt.close()


def plot_top_n_cluster(df, metric='recall', plot_name=''):
    avg = df.groupby(['topn'])[metric].agg('sum').reset_index()
    # avg = typecast(avg, columns=['topn'])

    sns.lineplot(data=avg, x='topn', y=metric)
    plt.savefig(plot_name)
    plt.close()


def create_run_path(path, method):
    res_path = f"{path}/results/{method}"
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    return res_path


def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    return pickle.load(open(filename, 'rb'))

