import itertools
from collections import Counter

import classification.scripts.utils as utils
import classification.scripts.validator as validator
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confidence_metrics(pred_data_df, way_id_pred_df, conf_levels):
    img_accs = [validator.get_confidence_accuracy(pred_data_df, conf_level) for conf_level in conf_levels]
    way_accs = [validator.get_confidence_accuracy(way_id_pred_df, conf_level) for conf_level in conf_levels]
    way_filtered_accs = [
        validator.get_confidence_accuracy(get_confidence_way_id_pred_df(pred_data_df, conf_level), conf_level) for
        conf_level in conf_levels]

    img_recalls = [get_confidence_recall(pred_data_df, conf_level) for conf_level in conf_levels]
    way_recalls = [get_confidence_recall(way_id_pred_df, conf_level) for conf_level in conf_levels]

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(121)

    ax.set_xlabel('confidence')
    ax.set_ylabel('accuracy')

    plt.plot(conf_levels, img_accs, 'b-', label='img_accuracy')

    for i, j in zip(conf_levels, img_accs):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.plot(conf_levels, img_recalls, 'y-', label='img_recalls')

    for i, j in zip(conf_levels, img_recalls):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.legend(loc='best')

    for key, acc, recall in zip(conf_levels, img_accs, img_recalls):
        print("%.2f --> Img acc %.2f / Img recall %.2f " % (key, acc, recall))

    ax = fig.add_subplot(122)

    ax.set_xlabel('confidence')
    ax.set_ylabel('accuracy')

    plt.plot(conf_levels, way_accs, 'r-', label='way_accuracy')

    for i, j in zip(conf_levels, way_accs):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.plot(conf_levels, way_filtered_accs, 'g-', label='way_filtered_accuracy')

    for i, j in zip(conf_levels, way_filtered_accs):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.plot(conf_levels, way_recalls, 'p-', label='way_recalls')

    for i, j in zip(conf_levels, way_recalls):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))
    plt.legend(loc='best')

    for key, acc, recall in zip(conf_levels, way_accs, way_recalls):
        print("%.2f --> Way acc %.2f / Way recall %.2f " % (key, acc, recall))

    plt.show()


def plot_confidence_metrics_by_class(pred_data_df, way_id_pred_df, conf_levels, classes, classIndex_2_class):
    pred_data_df.loc[:, 'pred_class'] = pred_data_df.loc[:, 'pred'].apply(
        lambda pred: label2text(pred, classIndex_2_class))
    way_id_pred_df.loc[:, 'pred_class'] = way_id_pred_df.loc[:, 'pred'].apply(
        lambda pred: label2text(pred, classIndex_2_class))

    class_2_img_acc = {}
    class_2_way_acc = {}

    for current_class in classes:
        class_pred_df = pred_data_df[pred_data_df['pred_class'] == current_class]
        img_accs = [validator.get_confidence_accuracy(class_pred_df, conf_level) for conf_level in conf_levels]
        class_2_img_acc[current_class] = img_accs

        way_class_pred_df = way_id_pred_df[way_id_pred_df['pred_class'] == current_class]
        way_accs = [validator.get_confidence_accuracy(way_class_pred_df, conf_level) for conf_level in conf_levels]
        class_2_way_acc[current_class] = way_accs

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(121)

    ax.set_xlabel('confidence')
    ax.set_ylabel('accuracy')

    for current_class in classes:
        plt.plot(conf_levels, class_2_img_acc[current_class], label=current_class)
        for i, j in zip(conf_levels, class_2_img_acc[current_class]):
            ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.legend(loc='best')

    ax = fig.add_subplot(122)

    ax.set_xlabel('confidence')
    ax.set_ylabel('accuracy')

    for current_class in classes:
        plt.plot(conf_levels, class_2_way_acc[current_class], label="way-" + current_class)
        for i, j in zip(conf_levels, class_2_way_acc[current_class]):
            ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.legend(loc='best')

    plt.show()


def plot_class_accuracy_by(way_id_pred_df, group_by_col, classes, classIndex_2_class):
    way_id_pred_df.loc[:, 'pred_class'] = way_id_pred_df.loc[:, 'pred'].apply(
        lambda pred: label2text(pred, classIndex_2_class))

    class_2_keyvals = {}

    for current_class in classes:
        class_way_id_pred_df = way_id_pred_df[way_id_pred_df['pred_class'] == current_class]

        keys, values = validator.get_accuracy_by_col(class_way_id_pred_df, group_by_col)
        class_2_keyvals[current_class] = (keys, values)

    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(121)

    ax.set_xlabel(group_by_col)
    ax.set_ylabel('accuracy')

    for current_class in classes:
        keys, values = class_2_keyvals[current_class]
        plt.plot(keys, values, label=current_class)

        for i, j in zip(keys, values):
            ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.legend(loc='best')
    plt.show()


def plot_count_per_conf_bucket(pred_data_df, classes, classIndex_2_class):
    pred_data_df.loc[:, 'pred_class'] = pred_data_df.loc[:, 'pred'].apply(
        lambda pred: label2text(pred, classIndex_2_class))

    class_2_keyvals = {}

    for current_class in classes:
        class_way_id_pred_df = pred_data_df[pred_data_df['pred_class'] == current_class]

        bucket_2_size = class_way_id_pred_df \
            .groupby("conf_bucket") \
            .size() \
            .to_dict()

        bu_2_acc = zip(bucket_2_size.keys(), bucket_2_size.values())
        bu_2_acc = sorted(bu_2_acc, key=lambda tup: tup[0])

        keys = [key for key, _ in bu_2_acc]
        values = [value for _, value in bu_2_acc]

        class_2_keyvals[current_class] = (keys, values)

    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(121)

    ax.set_xlabel('conf_bucket')
    ax.set_ylabel('accuracy')

    for current_class in classes:
        keys, values = class_2_keyvals[current_class]
        plt.plot(keys, values, label=current_class)

        for i, j in zip(keys, values):
            ax.annotate(str(int(round(j, 2))), xy=(i, j))

    plt.legend(loc='best')
    plt.show()


def plot_accuracy_by(way_id_pred_df, group_by_col):
    keys, values = validator.get_accuracy_by_col(way_id_pred_df, group_by_col)

    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(121)

    ax.set_xlabel(group_by_col)
    ax.set_ylabel('accuracy')

    plt.plot(keys, values, label='accuracy')

    for i, j in zip(keys, values):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.legend(loc='best')
    plt.show()


def plot_nr_ways_per_bucket(way_id_pred_df, group_by_col):
    bucket_2_accuracy = way_id_pred_df \
        .groupby(group_by_col) \
        .agg({'correct': 'count'}) \
        .rename(columns={'correct': 'count'}) \
        .to_dict()['count']

    bu_2_acc = zip(bucket_2_accuracy.keys(), bucket_2_accuracy.values())
    bu_2_acc = sorted(bu_2_acc, key=lambda tup: tup[0])

    plt.bar([key for key, _ in bu_2_acc], [value for _, value in bu_2_acc], label='nr_ways')
    plt.legend(loc='best')

    plt.xlabel(group_by_col)
    plt.ylabel('number of instances')

    plt.show()

    for key, value in bu_2_acc:
        print("%d --> Nr ways = %d" % (key, value))


def plot_metrics_by_feature_thresholds(data_df, feature_name, feature_thresholds, high_threshold=False):
    accuracies = [
        validator.compute_accuracy_by_feature_threshold(data_df, feature_name, feature_threshold, high_threshold) \
        for feature_threshold in feature_thresholds]

    recall_2_nr_elems = [
        validator.compute_recall_by_feature_threshold(data_df, feature_name, feature_threshold, high_threshold) \
        for feature_threshold in feature_thresholds]

    recalls = [recall for (recall, _) in recall_2_nr_elems]
    recall_nr_elems = [recall_nr_elem for (_, recall_nr_elem) in recall_2_nr_elems]

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(111)

    ax.set_xlabel(feature_name)
    ax.set_ylabel('accuracy/recall')

    plt.plot(feature_thresholds, accuracies, 'b-', label='accuracy')

    for i, j in zip(feature_thresholds, accuracies):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.plot(feature_thresholds, recalls, 'r-', label='recall')

    for i, j in zip(feature_thresholds, recalls):
        ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.legend(loc='best')
    plt.show()

    for index in range(len(feature_thresholds)):
        print("%.2f --> Acc = %.3f / Recall = %.3f ( %d )" % (feature_thresholds[index], accuracies[index], \
                                                              recalls[index], recall_nr_elems[index]))


def plot_class_metrics_by_feature_thresholds(data_df, feature_name, feature_thresholds, \
                                             classes, classIndex_2_class, high_threshold=False
                                             ):
    data_df.loc[:, 'pred_class'] = data_df.loc[:, 'pred'].apply(lambda pred: label2text(pred, classIndex_2_class))

    class_2_acc = {}

    for current_class in classes:
        class_data_df = data_df[data_df['pred_class'] == current_class]

        accuracies = [validator.compute_accuracy_by_feature_threshold(class_data_df, feature_name, feature_threshold,
                                                                      high_threshold) \
                      for feature_threshold in feature_thresholds]

        class_2_acc[current_class] = accuracies

    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(121)
    ax.set_xlabel(feature_name)
    ax.set_ylabel('accuracy/recall')

    for current_class in classes:
        accuracies = class_2_acc[current_class]
        plt.plot(feature_thresholds, accuracies, label=current_class)

        for i, j in zip(feature_thresholds, accuracies):
            ax.annotate(str(int(round(j, 2) * 100)), xy=(i, j))

    plt.legend(loc='best')
    plt.show()


def plot_predictions(pred_df, classIndex_2_class, label_class=None, pred_class=None, max_nr_results=None):
    if (max_nr_results is None):
        max_nr_results = len(pred_df)

    df = pred_df if label_class is None else pred_df[pred_df['label_class'] == label_class]
    df = df if pred_class is None else df[df['pred_class'] == pred_class]

    for index, row in df.iloc[:max_nr_results].iterrows():
        img = row['img']
        way_id = row['way_id']

        pred = label2text(row['pred'], classIndex_2_class)
        label = row['label_class']

        plt.imshow(img)
        plt.title("Pred = {} \nLabel = {} \nWay Id = {}".format(pred, label, way_id))
        plt.figure()

        plt.show()


# ------------------------------------------------------------------


def print_metrics_in_min_max_ranges(data_df, column, high_range, low_range):
    for high_value in high_range:

        print("----------")

        for low_value in low_range:

            if (low_value >= high_value):
                continue

            acc = compute_accuracy_by_min_max_feature_threshold(data_df, column, low_value, high_value)
            recall = compute_recall_by_min_max_feature_threshold(data_df, column, low_value, high_value)

            if (acc < 0.7 or recall < 0.7):
                continue

            print("%.2f to %.2f ==> Accuracy = %.3f \ Recall = %.3f" % (low_value, high_value, acc, recall))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')

    plt.show()


def plt_confusion_matrix(pred_df, classIndex_2_class, NR_CLASSES):
    preds = utils.numpify(pred_df['pred'])
    ground_truth = utils.numpify(pred_df['label'])

    cm_preds = [np.argmax(pred) for pred in preds]
    cm_ground_truth = [np.argmax(pred) for pred in ground_truth]

    cnf_matrix = confusion_matrix(cm_preds,cm_ground_truth)

    plot_confusion_matrix(cnf_matrix, classes=[classIndex_2_class[index] for index in range(NR_CLASSES)])


def print_metrics(pred_df):
    acc = validator.compute_accuracy(pred_df)

    print("Accuracy = %f" % acc)


def label2text(label, classIndex_2_class):
    return classIndex_2_class[np.argmax(label)]


def get_confidence_recall(way_id_pred_df, conf_level):
    nr_total = len(way_id_pred_df)
    conf_preds_indexes = way_id_pred_df['pred'].apply(lambda pred: max(pred) >= conf_level)
    nr_after_filter = float(Counter(conf_preds_indexes)[True])

    return nr_after_filter / nr_total


def compute_accuracy_by_min_max_feature_threshold(pred_df, feature_name, low_feature_threshold, high_feature_threshold):
    valid_indexes = pred_df[feature_name].apply(
        lambda feature_value: low_feature_threshold <= feature_value <= high_feature_threshold)

    valid_df = pred_df.loc[valid_indexes.tolist()]

    return validator.compute_accuracy(valid_df)


def compute_recall_by_min_max_feature_threshold(pred_df, feature_name, low_feature_threshold, high_feature_threshold):
    nr_total = len(pred_df)

    valid_indexes = pred_df[feature_name].apply(
        lambda feature_value: low_feature_threshold <= feature_value <= high_feature_threshold)

    nr_after_filter = float(Counter(valid_indexes)[True])

    return nr_after_filter / nr_total


def plot_confidence_images(data_df, classIndex_2_class, nr_imgs=None):
    if (nr_imgs == None):
        nr_imgs = len(data_df)

    for index, row in data_df.iloc[:nr_imgs].iterrows():
        img = row['img']
        conf = row['pred_conf']
        pred = label2text(row['pred'], classIndex_2_class)
        label = row['label_class']
        way_id = row['way_id']

        plt.imshow(img)
        plt.title("Pred = {} \nLabel = {} \nConf = {} \n".format(pred, label, conf))
        plt.figure()
    plt.show()


def get_data_2_bucket_count(data_df, target_col, bucket_size):
    data_df.loc[:, 'bucket'] = data_df.loc[:, target_col].apply(lambda feature_value: int(feature_value / bucket_size))

    bucket_counts = data_df \
        .groupby("bucket") \
        .agg({'correct': 'count'}) \
        .rename(columns={'correct': 'count'}) \
        .to_dict()['count']

    data_bu_2_acc = zip(bucket_counts.keys(), bucket_counts.values())
    data_bu_2_acc = sorted(data_bu_2_acc, key=lambda tup: tup[0])

    return data_bu_2_acc


def plot_nr_imgs_per_feature_bucket(data_df, column_name, bucket_size, nr_buckets=None):
    data_df.loc[:, 'bucket'] = data_df.loc[:, column_name].apply(lambda feature_value: int(feature_value / bucket_size))

    if (nr_buckets != None):
        data_df = data_df[data_df['bucket'] < nr_buckets]

    bucket_2_accuracy = data_df \
        .groupby("bucket") \
        .agg({'correct': 'count'}) \
        .rename(columns={'correct': 'count'}) \
        .to_dict()['count']

    bu_2_acc = zip(bucket_2_accuracy.keys(), bucket_2_accuracy.values())
    bu_2_acc = sorted(bu_2_acc, key=lambda tup: tup[0])

    plt.bar([key for key, _ in bu_2_acc], [value for _, value in bu_2_acc], label='nr_imgs')
    plt.legend(loc='best')
    plt.show()

    # for key,value in bu_2_acc:
    # print("%d --> Nr Imgs = %d"%(key,value))


def plot_feature_bucket_stats(data_df, feature_name, bucket_size, nr_buckets=None):
    correct_preds_df = data_df[data_df['correct'] == 1]
    incorrect_preds_df = data_df[data_df['correct'] == 0]

    data_bu_2_acc = get_data_2_bucket_count(data_df, feature_name, bucket_size)
    correct_bu_2_acc = get_data_2_bucket_count(correct_preds_df, feature_name, bucket_size)
    incorrect_bu_2_acc = get_data_2_bucket_count(incorrect_preds_df, feature_name, bucket_size)

    if (nr_buckets != None):
        data_bu_2_acc = data_bu_2_acc[:nr_buckets]
        correct_bu_2_acc = correct_bu_2_acc[:nr_buckets]
        incorrect_bu_2_acc = incorrect_bu_2_acc[:nr_buckets]

    all_keys = [key for key, _ in data_bu_2_acc]
    all_values = [value for _, value in data_bu_2_acc]
    correct_values = [value for _, value in correct_bu_2_acc]
    incorrect_values = [value for _, value in incorrect_bu_2_acc]

    plt.plot(all_keys, all_values, 'g-', label='all')
    plt.plot(all_keys, correct_values, 'b-', label='correct')
    plt.plot(all_keys, incorrect_values, 'r-', label='incorrect')

    plt.legend(loc='best')
    plt.figure()

    plt.bar(all_keys, [float(cv) / av for cv, av in zip(correct_values, all_values)], label='accuracy')
    plt.legend(loc='best')

    plt.show()
    plt.figure()

    plot_nr_imgs_per_feature_bucket(data_df, feature_name, bucket_size, nr_buckets)


def get_confidence_df(data_df, conf_level):
    conf_preds_indexes = data_df['pred'].apply(lambda pred: max(pred) >= conf_level)
    conf_df = data_df.loc[conf_preds_indexes.tolist()]

    return conf_df


def get_confidence_way_id_pred_df(pred_data_df, conf_level):
    conf_pred_data_df = get_confidence_df(pred_data_df, conf_level)
    conf_way_id_pred_df = utils.get_way_id_preds_df(conf_pred_data_df)
    return conf_way_id_pred_df


def plot_samples_from_each_class(df_path, img_path, classes, nr_pics_per_class, specific_class=None):
    data_df = utils.read_img_batches_in_df(df_path, img_path, nr_batches=3)

    for current_class in classes:

        if (specific_class != None and current_class != specific_class):
            continue

        class_df_indexes = data_df['label_class'] == current_class
        class_df = data_df[class_df_indexes][['img', 'label_class']]

        nr_elem = min(class_df.shape[0], nr_pics_per_class)

        for index in range(nr_elem):
            img = class_df.iloc[index]['img']
            label = class_df.iloc[index]['label_class']
            plt.imshow(img)
            plt.title(label)
            plt.figure()

            plt.show()
