import os
import tensorflow as tf
import numpy as np
import pytest

from .plots import accuracy_loss_plot, confusion_matrix_plot


def test_accuracy_loss_plot(tmpdir):
    # Create dummy history data

    # make a very simple model
    inputs = tf.keras.layers.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    x = np.random.random((2, 3))
    y = np.random.randint(0, 2, (2, 2))
    history = model.fit(x, y, epochs=3, validation_split=0.2)

    out_dir = os.path.join(str(tmpdir), "test_accuracy_loss_plot")
    # check that out_dir is not created yet
    assert not os.path.exists(out_dir)

    accuracy_loss_plot(history, model, "test_model", out_dir)

    # check that out_dir is created
    assert os.path.exists(out_dir)
    # check that the plot is saved
    assert os.path.exists(os.path.join(out_dir, "test_model_accuracy_loss.png"))

    # make a very simple model with 3 metrics
    inputs = tf.keras.layers.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
    x = np.random.random((2, 3))
    y = np.random.randint(0, 2, (2, 2))
    history = model.fit(x, y, epochs=3, validation_split=0.2)

    # check that the function raises an error if there are not exactly 2 metrics
    with pytest.raises(ValueError):
        accuracy_loss_plot(history, model, "test_model", out_dir)


def test_confusion_matrix_plot(tmpdir):
    # make a random 3x3 matrix with integer entries
    matrix = np.random.randint(0, 100, (3, 3))
    chosen_labels = [0, 1, 2]
    name_dict = {0: "label_0", 1: "label_1", 2: "label_2"}
    out_dir = os.path.join(str(tmpdir), "test_confusion_matrix_plot")
    model_name = "test_model"
    # check that out_dir is not created yet
    assert not os.path.exists(out_dir)

    # check that all 4 normalizations work
    # these are without saving
    for normalization in ["col", "max", None]:
        confusion_matrix_plot(
            matrix, normalization, name_dict, chosen_labels, model_name, save_dir=None
        )
    # this is with saving

    confusion_matrix_plot(matrix, "row", name_dict, chosen_labels, model_name, out_dir)

    # check that out_dir is created
    assert os.path.exists(out_dir)
    # check that the plot is saved
    assert os.path.exists(os.path.join(out_dir, "test_model_confusion_matrix.png"))


    #check that we raise an error if we call with wrong normalization string
    with pytest.raises(ValueError):
        confusion_matrix_plot(matrix, "any", name_dict, chosen_labels, model_name, out_dir)

    
