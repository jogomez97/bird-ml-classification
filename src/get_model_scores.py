import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import LabelBinarizer

from custom_dataset_loader_and_splitter import CustomDatasetLoaderAndSplitter

seed_list = [-1, 41, 1024, 777, 73059]


def load_dataset(args, seed=-1, only_test=False):
    # load the images and get the splits
    data_loader = CustomDatasetLoaderAndSplitter(
        args.dataset_path,
        validation=args.validation_split,
        test=args.test_split,
        seed=seed,
    )

    if args.test_split > 0:
        train_x, train_y, val_x, val_y, test_x, test_y = data_loader.load_and_split()
    else:
        train_x, train_y, val_x, val_y = data_loader.load_and_split()

    # convert the labels from integers to vectors
    train_y = LabelBinarizer().fit_transform(train_y)
    val_y = LabelBinarizer().fit_transform(val_y)

    # set up and return test data if needed
    if args.test_split > 0:
        test_y = LabelBinarizer().fit_transform(test_y)
        if only_test:
            return test_x, test_y
        return train_x, train_y, val_x, val_y, test_x, test_y

    if only_test:
        return val_x, val_y
    return train_x, train_y, val_x, val_y


def evaluate_model(args, models_path, test_x, test_y, class_names, seed):
    model = load_model(models_path)

    # evaluate the network on the fine-tuned model
    print("[INFO] evaluating after fine-tuning...")
    predictions = model.predict(test_x, batch_size=args.batch_size)

    y_true = test_y.argmax(axis=1)
    y_pred = predictions.argmax(axis=1)

    with open("stats.csv", "a", newline="\n") as fd:
        fd.write("\n")
        fd.write(models_path.split("/")[-3])
        fd.write(",")
        fd.write(str(seed_list.index(seed) + 1))
        fd.write(",")
        fd.write(str(seed))
        fd.write(",")
        fd.write(str(accuracy_score(y_true, y_pred)))
        fd.write(",")
        fd.write(str(recall_score(y_true, y_pred, average="micro", zero_division=0)))
        fd.write(",")
        fd.write(str(recall_score(y_true, y_pred, average="macro", zero_division=0)))
        fd.write(",")
        fd.write(str(recall_score(y_true, y_pred, average="weighted", zero_division=0)))
        fd.write(",")
        fd.write(str(precision_score(y_true, y_pred, average="micro", zero_division=0)))
        fd.write(",")
        fd.write(str(precision_score(y_true, y_pred, average="macro", zero_division=0)))
        fd.write(",")
        fd.write(
            str(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        )
        fd.write(",")
        fd.write(str(f1_score(y_true, y_pred, average="micro", zero_division=0)))
        fd.write(",")
        fd.write(str(f1_score(y_true, y_pred, average="macro", zero_division=0)))
        fd.write(",")
        fd.write(str(f1_score(y_true, y_pred, average="weighted", zero_division=0)))

    if args.plot_matrix:
        c_dict = classification_report(
            test_y.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=class_names,
            output_dict=True,
        )
        avg_score = c_dict["macro avg"]["recall"]

        ax = skplt.metrics.plot_confusion_matrix(
            test_y.argmax(axis=1),
            predictions.argmax(axis=1),
            normalize=True,
            cmap="Blues",
            figsize=(12, 8),
        )
        ax.xaxis.set_ticklabels(class_names)
        ax.yaxis.set_ticklabels(class_names)
        plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment="right")
        plt.tight_layout()
        ax.set_xlabel("Predicted label\nAverage score: {:.03f}".format(avg_score))
        plt.show()


def main(args):
    # Group all models by dict with their path
    a = {k: [] for k in seed_list}
    i = 0
    prev_model = ""
    for subdir, dirs, files in os.walk(args.models_path):
        for file in files:
            if file == "output.h5":
                model_name = subdir.split("/")[-2]
                if model_name != prev_model:
                    i = 0
                    prev_model = model_name

                model_path = os.path.join(subdir, file)
                a[seed_list[i]].append(model_path)
                i += 1

    # write header for stats file
    with open("stats.csv", "w") as fd:
        fd.write(
            "model_name,fold,seed,accuracy,recall_micro,recall_macro,recall_weighted,precision_micro,precision_macro,precision_weighted,f1_micro,f1_macro,f1_weighted"
        )

    for seed, models in a.items():
        # get the test images and labels
        class_names = list(os.listdir(args.dataset_path))

        test_x, test_y = load_dataset(args, seed, only_test=True)

        # evaluate all models that share the same seed so we avoid loading the dataset every time
        for model_path in models:
            evaluate_model(args, model_path, test_x, test_y, class_names, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--dataset-path", type=str, help="Path to the dataset.", required=True
    )
    parser.add_argument(
        "-m",
        "--models-path",
        type=str,
        help="Path to the models. It will look for all output.h5 files.",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for training the model.",
    )
    parser.add_argument(
        "--validation-split",
        type=int,
        default=0.2,
        help="Percentage of dataset to split for validation.",
    )
    parser.add_argument(
        "--test-split",
        type=int,
        default=0,
        help="Percentage of dataset to split for test.",
    )
    parser.add_argument(
        "--plot-matrix",
        action="store_true",
        help="Plot the confusion matrix after a model evaluation.",
    )
    args = parser.parse_args()

    main(args)
