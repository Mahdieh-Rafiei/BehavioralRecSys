import argparse
import os
from Item import cv_training
from Utils import create_paths, plot_skl_mtericts, plot_top_n, plot_top_n_cluster, create_run_path, load_model


def parse_arguments():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description="Federated Learning models",
                                     usage="python main.py --path 'home/' --fixed fix.csv")
    #   General
    parser.add_argument("--path", type=str, help="path to the dataset", default=os.getcwd())
    parser.add_argument("--method", "--m", type=str, choices=["cbrs", "bssb-rs"], default="cbrs")

    #   Dataset
    parser.add_argument("--fixed", type=str, help="name of the fixed train-set file", default="M&MfixtrainSet.csv")
    parser.add_argument("--validation", type=str, help="name of the validation file", default="M&MfixtrainSet.csv")
    parser.add_argument("--n_folds", type=int, help="Number of folds for CV", default=2)

    #   Model
    parser.add_argument("--n_clusters", "--K", type=int, help=" number of clusters", default=3)
    parser.add_argument("--top_n", type=int, help="Top N recommendation", default=2)
    parser.add_argument("--leave_one_out", dest="leave_one_out", help="Use leave one Out evaluation", default=False,
                        action='store_true')
    # parser.set_defaults(leave_one_out=True)
    return parser.parse_args()


#   Name: categorical behavioural recommendation system

def cbrs(path, fixed_file, val_file, n_clusters, top_n, n_folds, leave_one_out, res_path):
    skl_metrics = None
    top_n_metrics = None
    for cluster in range(n_clusters):
        print(f"Cluster {cluster + 1} execution:")
        fixed_path, val_path = create_paths(path, fixed_file, val_file, cluster)
        eval, topn_eval = cv_training(fixed_path, val_path, top_n, n_folds, leave_one_out)
        eval['cluster'] = cluster + 1
        if skl_metrics is None:
            skl_metrics = eval
        else:
            skl_metrics = skl_metrics.append(eval)
        topn_eval['cluster'] = cluster + 1
        if top_n_metrics is None:
            top_n_metrics = topn_eval
        else:
            top_n_metrics = top_n_metrics.append(topn_eval)
    skl_metrics['cluster'] = skl_metrics['cluster'].astype(int)
    skl_metrics['fold'] = skl_metrics['fold'].astype(int)
    skl_metrics.to_csv(f"{res_path}/sklearn_metrics.csv")
    top_n_metrics['cluster'] = top_n_metrics['cluster'].astype(int)
    top_n_metrics['fold'] = top_n_metrics['fold'].astype(int)
    top_n_metrics['topn'] = top_n_metrics['topn'].astype(int)

    top_n_metrics.to_csv(f"{res_path}/top_n_metrics.csv")
    for metric in ['recall', 'precision', 'fmeasure']:
        plot_top_n_cluster(top_n_metrics, metric, f"{res_path}/{metric}-c.png")
        plot_top_n(top_n_metrics, metric, f"{res_path}/{metric}.png")
        plot_skl_mtericts(skl_metrics, metric, f"{res_path}/sklearn_metrics_{metric}.png")


# def predict(user):
#     user_cluster = cluster(user)
#     models = load_model(filename=f"./cluster{user_cluster}/models")
#
#     cluster_dataset = import_cluster(user_cluster)
#     cv_training(path, fixed, validation, cluster_dataset, top_n, n_folds, leave_one_out)


def bssb_rs(path, fixed_file, val_file, top_n, n_folds, leave_one_out):
    fixed_path, val_path = create_paths(path, fixed_file, val_file, cluster=None)
    cv_training(fixed_path, val_path, top_n, n_folds, leave_one_out)


if __name__ == '__main__':
    # CLuster users

    # predict movies
    args = parse_arguments()
    res_path = create_run_path(args.path, args.method)
    if args.method == "cbrs":
        cbrs(args.path, args.fixed, args.validation, args.n_clusters, args.top_n, args.n_folds, args.leave_one_out, res_path)
    elif args.method == "bssb-rs":
        bssb_rs(args.path, args.fixed, args.validation, args.top_n, args.n_folds, args.leave_one_out, res_path)
