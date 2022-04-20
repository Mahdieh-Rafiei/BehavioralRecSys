import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def filter_users(predictions, test_user_liked_movies, user_id, n):
    pred = predictions[
               predictions['user_id'] == user_id
               ].sort_values(by=['prob']).film_id.values[:n]
    liked = test_user_liked_movies[
        (test_user_liked_movies['label'] == 1) &
        (test_user_liked_movies['user_id'] == user_id)
        ].film_id.values
    disliked = test_user_liked_movies[
        (test_user_liked_movies['label'] == 0) &
        (test_user_liked_movies['user_id'] == user_id)
        ].film_id.values

    return pred, liked, disliked


def top_n_recall(predictions, test_user_liked_movies, n):
    test_user_ids = test_user_liked_movies.user_id.unique()
    user_top_n_acc = []
    for user_id in test_user_ids:
        pred, liked, _ = filter_users(predictions, test_user_liked_movies, user_id, n)
        n_liked_movies = len(liked)
        n_liked_movies_in_top_n = len(set(pred).intersection(set(liked)))
        user_top_n_acc.append(0 if n_liked_movies == 0 else n_liked_movies_in_top_n / n_liked_movies)
        # user_top_n_acc.append(n_liked_movies_in_top_n / n)
    return np.mean(user_top_n_acc)


def top_n_precision(predictions, test_user_liked_movies, n):
    test_user_ids = test_user_liked_movies.user_id.unique()
    user_top_n_acc = 0.0
    for user_id in test_user_ids:
        pred, liked, disliked = filter_users(predictions, test_user_liked_movies, user_id, n)
        n_liked_movies_in_top_n = len(set(pred).intersection(set(liked)))
        user_top_n_acc += n_liked_movies_in_top_n
    return user_top_n_acc / (len(test_user_ids) * n)


def top_n_f_measure(precision, recall):
    return (precision * recall) / (precision + recall)


def skl_metr(pred, true):
    y_pred = (pred['prob'] >= 0.5).map({False: 0, True: 1}).values
    y_true = true.label.values
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fmeas = f1_score(y_true, y_pred)
    return acc, prec, recall, fmeas

