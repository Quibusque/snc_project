import pytest

import tensorflow as tf
import os
import numpy as np
from .metrics import confusion_matrix, save_class_metrics, save_accuracy_loss
import tensorflow as tf
def test_save_accuracy_loss(tmpdir):
    # Create dummy history data

    # make a very simple model
    n_entries = 10
    n_features = 3
    n_classes = 2
    inputs = tf.keras.layers.Input(shape=(n_features,))
    outputs = tf.keras.layers.Dense(n_classes)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    model.summary()
    x = np.random.random((n_entries, n_features))
    y = np.random.randint(0, n_classes, (n_entries))
    history = model.fit(x, y, epochs=1, validation_split=0.2)

    out_dir = os.path.join(str(tmpdir), "test_accuracy_loss")
    # check that out_dir is not created yet
    assert not os.path.exists(out_dir)

    save_accuracy_loss(history, model, "test_model", out_dir)

    # check that out_dir/model_name_accuracy_loss.csv is created
    assert os.path.exists(os.path.join(out_dir, "test_model_accuracy_loss.csv"))

    # make a very simple model
    model.compile(optimizer="Adam", loss="mse", metrics=["mae", "accuracy"])
    history = model.fit(x, y, epochs=1, validation_split=0.2)

    # check that the function raises an error if there are not exactly 2 metrics
    with pytest.raises(ValueError):
        save_accuracy_loss(history, model, "test_model", out_dir)

def test_confusion_matrix():

    # make a very simple model
    n_entries = 10
    n_features = 3
    n_classes = 2
    x = np.random.random((n_entries, n_features))
    y = np.random.randint(0, n_classes, (n_entries))

    inputs = tf.keras.layers.Input(shape=(None,n_features))
    outputs = tf.keras.layers.Dense(n_classes)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    model.summary()

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)

    model.fit(dataset, epochs=1)


    confusion_matrix(dataset, model, n_classes)

def test_save_class_metrics(tmpdir):
    #create dummy matrix
    matrix = np.array([[1,2],[3,4]])
    save_dir = os.path.join(str(tmpdir), "test_class_metrics")
    # check that save_dir is not created yet
    assert not os.path.exists(save_dir)

    save_class_metrics(matrix, save_dir, model_name = "test_model")
    #check the file save_dir/model_name_precision_recall.csv is created
    assert os.path.exists(os.path.join(save_dir, "test_model_precision_recall_f1.csv"))
