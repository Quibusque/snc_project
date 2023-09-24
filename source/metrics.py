import numpy as np
import os
import tensorflow as tf

def save_accuracy_loss(history, model_name: str, save_dir: str):
    """
    This function saves the accuracy and loss for the training and validation to
    a csv file as "save_dir/model_name_accuracy_loss.csv".

    Args:
        history: The history object returned by model.fit().
        model_name: The name of the model (used for the file name).
        save_dir: The directory to save the csv file to.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = len(acc)

    with open(os.path.join(save_dir, model_name + "_accuracy_loss.csv"), "w") as f:
        f.write("epoch,acc,val_acc,loss,val_loss\n")
        for i in range(epochs):
            f.write(f"{i},{acc[i]},{val_acc[i]},{loss[i]},{val_loss[i]}\n")


def confusion_matrix(
    dataset: tf.data.Dataset, model: tf.keras.Model, num_classes: int
) -> np.ndarray:
    """
    This function computes the confusion matrix for a given dataset and model.

    Args:
        dataset (tf.data.Dataset): dataset to compute confusion matrix for
        model (tf.keras.Model): model to use for predictions
        num_classes (int): number of classes for this model

    Returns:
        matrix (np.ndarray): confusion matrix
    """

    predicted_classes = np.array([])
    true_classes = np.array([])

    print(
        f"Computing confusion matrix for {len(dataset)} images, this may take a while..."
    )
    for x, y in dataset:
        predicted_classes = np.concatenate(
            [predicted_classes, np.argmax(model(x), axis=-1)]
        )
        true_classes = np.concatenate([true_classes, y])
    print("Done!")

    matrix = (
        tf.math.confusion_matrix(
            labels=true_classes, predictions=predicted_classes, num_classes=num_classes
        )
        .numpy()
        .astype("float32")
    )

    return matrix


def class_metrics_compute_and_save_(
    matrix: np.ndarray, save_dir: str, model_name: str
) -> None:
    """
    This function computes the precision, recall and f1 score for each class
    based on the confusion matrix and saves them to a csv file in the path
    "save_dir/model_name_precision_recall_f1.csv".

    Args:
        matrix: The confusion matrix.
        save_dir: The directory to save the csv file to.
        model_name: The name of the model (for the file name).
    """
    num_classes = matrix.shape[0]
    with open(
        os.path.join(save_dir, model_name + "_precision_recall_f1.csv"), "w"
    ) as f:
        f.write("class,precision,recall,f1\n")
        for i in range(num_classes):
            # compute precision recall and f1 for each class
            precision = matrix[i, i] / matrix[:, i].sum()
            recall = matrix[i, i] / matrix[i, :].sum()
            f1 = 2 * precision * recall / (precision + recall)

            f.write(f"{i},{precision},{recall},{f1}\n")
