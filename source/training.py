import os  # add exception handling

import numpy as np
import pandas as pd

import tensorflow as tf

from source.utils import labels_for_dataset


def build_dataset(
    img_dir: str,
    df_good: pd.DataFrame,
    shuffle: bool,
    seed: int,
    validation_split: float = None,
    image_size: tuple = (256, 256),
    crop_to_aspect_ratio: bool = True,
    batch_size: int = 64,
    label_key: str = "label",
) -> tf.data.Dataset:
    """
    This function builds a tf dataset from a directory of images and a dataframe
    with the image file names and labels.
    If validation_split is not None, the dataset is split into training and
    validation sets.

    Args:
        img_dir (str): path to image directory
        df_good (pd.DataFrame): dataframe with image file names and labels
        shuffle (bool): whether to shuffle the dataset
        seed (int): seed for shuffling
        validation_split (float): fraction of the dataset to use for
            validation. Defaults to None.
        image_size (tuple): size of the images in the dataset. Defaults to (256,256).
        crop_to_aspect_ratio (bool): whether to crop the images to the aspect
            ratio of image_size. Defaults to True.
        batch_size (int): batch size. Defaults to 64.
        label_key (str): name of the column containing the labels. Defaults to "label".

    Returns:
        if validation_split is None:
            dataset (tf.data.Dataset): dataset of images and labels
        if validation_split is not None:
            training (tf.data.Dataset): training dataset
            validation (tf.data.Dataset): validation dataset
    """
    # set subset variable based on validation_split
    if validation_split is None:
        subset = None
    else:
        subset = "both"

    # put labels in the [0,num_classes) range
    true_label_list = labels_for_dataset(df_good, label_key)

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=img_dir,
        labels=true_label_list,
        label_mode="int",
        batch_size=batch_size,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        color_mode="rgb",
        image_size=image_size,
        shuffle=shuffle,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )

    if validation_split is not None:
        training = dataset[0]
        validation = dataset[1]
        return training, validation

    return dataset


def build_model(
    num_classes: int,
    metric: tf.keras.metrics.Metric,
    loss: tf.keras.losses.Loss,
    dropout_rate: float = 0.2,
    input_shape: tuple = (256, 256, 3),
    print_summary: bool = True,
) -> tf.keras.models.Model:
    """
    This function builds a model with the EfficientNetV2B0 network as the base
    and a dense layer with softmax activation as the head. The model is compiled
    with the Adam optimizer, the specified loss function and the specified metric.

    Args:
        num_classes (int): number of classes
        metric (tf.keras.metrics.Metric): metric to use for evaluation
        loss (tf.keras.losses.Loss): loss function
        dropout_rate (float): dropout rate. Defaults to 0.2.
        input_shape (tuple): shape of the input images. Defaults to (256,256,3).
        print_summary (bool): whether to print the model summary. Defaults to True.

    Returns:
        model (tf.keras.models.Model): compiled model
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
                include_top=False,
                weights=None,
                input_tensor=None,
                input_shape=input_shape,
                pooling="max",
                include_preprocessing=True,
            ),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=metric,
    )

    if print_summary:
        model.summary()

    return model


def train_model(
    model,
    training_data,
    validation_data,
    epochs,
    model_name,
    checkpoint_dir,
    patience=3,
    save_weights_only=True,
    save_best_only=True,
) -> tf.keras.callbacks.History:
    """
    This function trains a model on the training data and evaluates it on the
    validation data. Early stopping is used to stop training if the validation
    loss does not improve for patience epochs. The model weights are saved to
    "checkpoint_dir/model_name" after each epoch.

    Args:
        model (tf.keras.models.Model): model to train
        training_data (tf.data.Dataset): training dataset
        validation_data (tf.data.Dataset): validation dataset
        epochs (int): number of epochs
        model_name (str): name of the model (used for saving the weights)
        checkpoint_dir (str): path to the directory to save the weights to
        patience (int): number of epochs to wait before stopping training if
            the validation loss does not improve. Defaults to 3.
        save_weights_only (bool): whether to save only the weights or the whole
            model. Defaults to True.
        save_best_only (bool): whether to save only the weights of the best
            model. Defaults to True.
    Returns:
        history (tf.keras.callbacks.History): history object containing training
            and validation metrics
    """
    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, model_name, "cp-{epoch:02d}.h5"),
                monitor="val_loss",
                save_best_only=save_best_only,
                save_weights_only=save_weights_only,
            ),
        ],
    )
    return history
