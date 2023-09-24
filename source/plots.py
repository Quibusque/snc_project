import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import seaborn as sns


def accuracy_loss_plot_and_save(history, model_name: str, save_dir: str) -> None:
    """
    This function plots the accuracy and loss for the training and validation
    sets and saves it to a file as "save_dir/model_name_accuracy_loss.png".

    Args:
        history: The history object returned by model.fit().
        model_name: The name of the model (used for the file name).
        save_dir: The directory to save the plot to.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()

    plt.savefig(os.path.join(save_dir, model_name + "_accuracy_loss.png"))
    plt.close()

def confusion_matrix_plot_and_save(
    matrix: np.ndarray,
    normalization: str,
    class_names: list[str],
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
        class_names (list[str]): list of class names to label the axes
        model_name (str): name of the model (used for the file name)
        save_dir (str): path to the directory to save the confusion matrix
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
        plt.savefig(os.path.join(save_dir, model_name + "_confusion_matrix.png"))
    plt.show()
