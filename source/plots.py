import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import seaborn as sns


def accuracy_loss_plot(
    history: tf.keras.callbacks.History,
    model: tf.keras.models.Model,
    model_name: str,
    save_dir: str,
) -> None:
    """
    This function plots the accuracy and loss for the training and validation and
    saves it to a file as "save_dir/model_name_accuracy_loss.png".
    Accuracy and loss can have arbitrary names, but there must be exactly 2.

    Args:
        history (tf.keras.callbacks.History): history object returned by model.fit()
        model (tf.keras.models.Model): model to plot the accuracy and loss for
        model_name (str): name of the model (used for the file name)
        save_dir (str): path to the directory to save the accuracy and loss plot

    Raises:
        ValueError: if there are not exactly 2 metrics
    """

    metrics_list = model.metrics_names
    num_metrics = len(metrics_list)
    if num_metrics != 2:
        raise ValueError("save_accuracy_loss requires exactly 2 metrics")

    metric_1, metric_2 = metrics_list

    training_values_1 = history.history[metric_1]
    validation_values_1 = history.history["val_" + metric_1]
    training_values_2 = history.history[metric_2]
    validation_values_2 = history.history["val_" + metric_2]

    epochs_range = range(len(training_values_1))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_values_1, label=f"Training {metric_1}")
    plt.plot(epochs_range, validation_values_1, label=f"Validation {metric_1}")
    plt.legend(loc="lower right")
    plt.title(f"Training and Validation {metric_1}")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_values_2, label=f"Training {metric_2}")
    plt.plot(epochs_range, validation_values_2, label=f"Validation {metric_2}")
    plt.legend(loc="upper right")
    plt.title(f"Training and Validation {metric_2}")


    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, model_name + "_accuracy_loss.png"))
        print(f"Saved accuracy and loss to {save_dir}/{model_name}_accuracy_loss.png")


def confusion_matrix_plot(
    matrix: np.ndarray,
    normalization: str,
    name_dict: dict,
    chosen_labels: list,
    model_name: str,
    save_dir: str,
) -> None:
    """
    This function plots the confusion matrix and saves it to a file as
    "save_dir/model_name_confusion_matrix.png".

    Normalization can be one of ["row","col","max",None].
    "row" and "col" normalization normalizes the sum of rows or columns to be 1.
    "max" normalization normalizes the matrix to have maximum value 1.
    None does not normalize the matrix.

    Args:
        matrix (np.ndarray): confusion matrix
        normalization (str): type of normalization
        name_dict (dict): dictionary mapping class index to class name
        chosen_labels (list): list of labels that were used for training
        model_name (str): name of the model (used for the file name)
        save_dir (str): path to the directory to save the confusion matrix plot
    """
    # check for valid normalization
    if normalization not in ["row", "col", "max", None]:
        raise ValueError("Normalization must be one of ['row','col','max', None]")

    # normalize the matrix
    if normalization == "row":
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
    elif normalization == "col":
        matrix = matrix / matrix.sum(axis=0, keepdims=True)
    elif normalization == "max":
        matrix = matrix / matrix.max()

    # generate class_names list
    class_names = [name_dict[label] for label in chosen_labels]

    sns.heatmap(
        matrix,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
        fmt=".2f",
        cmap="Blues",
    )
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    plt.title("Confusion Matrix")

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, model_name + "_confusion_matrix.png"))
        print(f"Saved confusion matrix to {save_dir}/{model_name}_confusion_matrix.png")