import os  # add exception handling

import numpy as np
import pandas as pd

import tensorflow as tf


def build_dataset(
    img_dir: str,
    image_size: tuple,
    true_label_list: list,
    crop_to_aspect_ratio: bool = True,
    batch_size: int = 64,
    validation_split: float = None,
    shuffle: bool = False,
    seed: int = None,
):
    # set subset variable based on validation_split
    if validation_split is None:
        subset = None
    else:
        subset = "both"

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


def build_model(num_classes, input_shape, metric, loss, dropout_rate, print_summary):
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
):
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
