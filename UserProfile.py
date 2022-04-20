from Item import train_metric_group
index = 0
import pandas as pd

# def train_metric_group(grp):
#     global index
#     X_train, X_test, y_train, y_test = train_test_split(
#         grp.drop(columns=["user_id", "film_id", "rate", "label"]),
#         grp.label,
#         test_size=0.2,
#         random_state=42,
#     )
#
#     model = RandomForestClassifier(n_estimators=10)
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#     #     print(y_pred)
#     pickle.dump(model, open('models/RF_cluster' + str(index), "wb"))
#     index = index + 1
#
#     return pd.Series(
#         {
#             "accuracy": accuracy_score(y_test, y_pred),
#             "precision": precision_score(y_test, y_pred, zero_division=1),
#             "recall": recall_score(y_test, y_pred, zero_division=1),
#             "f1": f1_score(y_test, y_pred, zero_division=1),
#             "MAE": mean_absolute_error(y_test, y_pred),
#         }
#     )


def partition_users(file_name: str = "NNEW111.csv", ):
    df = pd.read_csv(file_name)
    # labels = df['rate']
    films_num_rates = df.film_id.value_counts()

    df_rates = df[df.film_id.isin(films_num_rates.index)]

    df_rates = df_rates.assign(label=0)
    df_rates.loc[(df_rates.rate >= 7), "label"] = 1


    result = df_rates.groupby("film_id").apply(train_metric_group)

    result.to_csv("New-result-Random-forest-cluster1.csv")
    result.mean()

#-------------------------Cluster 2---------------------------
df = pd.read_csv("NNEW222.csv")
labels = df['rate']
films_num_rates = df.film_id.value_counts()

df_rates = df[df.film_id.isin(films_num_rates.index)]

df_rates = df_rates.assign(label=0)
df_rates.loc[(df_rates.rate >= 7), "label"] = 1


result = df_rates.groupby("film_id").apply(train_metric_group)

result.to_csv("New-result-Random-forest-cluster2.csv")
result.mean()

#-------------------------Cluster 3---------------------------
df = pd.read_csv("NNEW333.csv")
labels = df['rate']
films_num_rates = df.film_id.value_counts()

df_rates = df[df.film_id.isin(films_num_rates.index)]

df_rates = df_rates.assign(label=0)
df_rates.loc[(df_rates.rate >= 7), "label"] = 1


result = df_rates.groupby("film_id").apply(train_metric_group)

result.to_csv("New-result-Random-forest-cluster3.csv")
result.mean()