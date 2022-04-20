import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Utils import load_and_split, drop_columns
from sklearn.model_selection import KFold
from Metrics import top_n_recall, top_n_precision, top_n_f_measure, skl_metr, top_n_accuracy
import numpy as np

index = 0


def train_metric_group(train_set):
    x_train, y_train = filter_dataframe(train_set)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(x_train, y_train)
    return model


def predict(models, test_set):
    predictions = pd.DataFrame(data=[], columns=['film_id', 'user_id', 'pred', 'prob'])
    for film_id, model in models.items():
        filtered_films = test_set[test_set['film_id'] == film_id]
        if len(filtered_films) == 0:
            continue
        user_ids = filtered_films.user_id
        x_test, y_test = filter_dataframe(filtered_films)
        predict = model.predict_proba(x_test)
        prob = []
        # [0.4, 0.6]
        # [1.0]
        for p in predict:
            # pred.append(p[1] if len(p) > 1 else 1-p[0])
            prob.append(p[1] if len(p) > 1 else p[0])
        pred = model.predict(x_test)
        ids = [film_id] * len(pred)

        x = list(zip(user_ids, prob, pred, ids))
        predictions = predictions.append(pd.DataFrame(x, columns=['user_id',
                                                                  'prob',
                                                                  'pred',
                                                                  'film_id']))
    return predictions


def filter_dataframe(df):
    columns = list(set(df.columns).intersection(["user_id", "film_id", "rate", "label", "Unnamed: 0"]))
    return df.drop(columns=columns), df.label


def train_model(train_set, test_set):
    models = {}
    for film_id in train_set.film_id.unique():
        models[film_id] = train_metric_group(train_set[train_set['film_id'] == film_id])
    predictions = predict(models, test_set)
    return predictions


def avg_topn(data, weights):
    return np.sum([np.array(d) * weights[i] for i, d in enumerate(data)], axis=0) / sum(weights)


def cv_training(fixed_path, val_path, top_n, n_folds, leave_one_out):
    fixed, dynam = load_and_split(fixed_path, val_path)
    total_n_users = len(dynam)
    kf = KFold(n_splits=len(dynam) if leave_one_out else n_folds)
    folds = kf.split(dynam)
    total_topn_recall, total_topn_precision, total_topn_f_measure = [], [], []
    evals = pd.DataFrame(columns=['fold', 'n_users', 'acc'])
    top_n_evals = pd.DataFrame(columns=['fold', 'n_users', 'topn', 'recall', 'precision', 'fmeasure'])
    for fold_n, fold in enumerate(folds, 1):
        print(f"\n\t#### Training for fold number {fold_n} ###\n")
        train = fixed.append(dynam[dynam.index.isin(fold[0])])
        test = dynam[dynam.index.isin(fold[1])]
        test_users_liked_movies = drop_columns(test)
        pred = train_model(train, test)
        n_users = test.user_id.nunique()
        acc, precision, recall, f_measure = skl_metr(pred, test_users_liked_movies)
        print(acc, precision, recall, f_measure)
        evals = evals.append({'fold': fold_n,
                              'n_users': n_users,
                              'acc': acc * n_users / total_n_users,
                              'precision': precision * n_users / total_n_users,
                              'recall': recall * n_users / total_n_users,
                              'fmeasure': f_measure * n_users / total_n_users
                              }, ignore_index=True)
        topn_recall, topn_precision, topn_f_measure = [], [], []
        for n in range(1, top_n + 1):
            recall = top_n_recall(pred, test_users_liked_movies, n)
            print(f"Average Top-{n} Recall of fold {fold_n}: {recall}")
            topn_recall.append(recall)
            precision = top_n_precision(pred, test_users_liked_movies, n)
            print(f"Average Top-{n} Precision of fold {fold_n}: {precision}")
            topn_precision.append(precision)
            f_measure = top_n_f_measure(precision, recall)
            print(f"Average Top-{n} F-measure of fold {fold_n}: {f_measure}")
            topn_f_measure.append(f_measure)
            record = {'fold': fold_n,
                      'n_users': n_users,
                      'topn': n,
                      'recall': recall * n_users / total_n_users,
                      'precision': precision * n_users / total_n_users,
                      'fmeasure': f_measure * n_users / total_n_users}
            top_n_evals = top_n_evals.append(record, ignore_index=True)

        total_topn_f_measure.append(topn_f_measure)
        total_topn_recall.append(topn_recall)
        total_topn_precision.append(topn_precision)
    print(f"Total Average Accuracy: {evals.groupby('fold')['acc'].sum().values}")
    print(f"Total Average of Top-N recalls: {top_n_evals.groupby('topn')['recall'].sum().values}")
    print(f"Total Average of Top-N Precision: {top_n_evals.groupby('topn')['precision'].sum().values}")
    print(f"Total Average of Top-N F-measure: {top_n_evals.groupby('topn')['fmeasure'].sum().values}")
    return evals, top_n_evals
