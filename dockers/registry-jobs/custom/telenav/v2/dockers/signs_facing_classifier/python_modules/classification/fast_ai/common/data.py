import pandas as pd
from sklearn.utils import shuffle


def train_test_split_stratified(data_df, percentage, groupby):
    """ Given a dataframe containing a full set of metadata about ROI facing images, it splits it using the percentage
     argument in a stratified manner, based on the given groupby column.
    """

    train_list = []
    test_list = []

    for label_class, grouped_df in data_df.groupby(groupby):
        nr_train = int(percentage * len(grouped_df))
        grouped_df = shuffle(grouped_df, random_state=0)

        train_df = grouped_df[:nr_train]
        test_df = grouped_df[nr_train:]

        train_list.append(train_df)
        test_list.append(test_df)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df
