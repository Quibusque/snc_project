import numpy as np
import os
import tensorflow as tf


def save_accuracy_loss(
    history: tf.keras.callbacks.History,
    model: tf.keras.models.Model,
    model_name: str,
    save_dir: str,
) -> None:
    """
    This function saves the accuracy and loss for the training and validation to a
    csv file as "save_dir/model_name_accuracy_loss.csv".
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
    val_metric_1, val_metric_2 = "val_" + metric_1, "val_" + metric_2

    training_values_1 = history.history[metric_1]
    validation_values_1 = history.history[val_metric_1]
    training_values_2 = history.history[metric_2]
    validation_values_2 = history.history[val_metric_2]

    epochs = len(training_values_1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, model_name + "_accuracy_loss.csv"), "w") as f:
        f.write(f"epoch,{metric_1},{val_metric_1},{metric_2},{val_metric_2}\n")
        for i in range(epochs):
            f.write(
                f"{i},{training_values_1[i]},{validation_values_1[i]},{training_values_2[i]},{validation_values_2[i]}\n"
            )


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


def save_class_metrics(matrix: np.ndarray, save_dir: str, model_name: str) -> None:
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
