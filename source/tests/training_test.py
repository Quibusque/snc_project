import pytest
import numpy as np
import os
import pandas as pd
from PIL import Image
import tensorflow as tf

from ..training import build_dataset, build_model, train_model


@pytest.fixture
def sample_df():
    # make a dummy dataframe with 10 entries and 3 labels [1,2,3]
    df = pd.DataFrame(
        {
            "file_name": [f"img{i}.jpeg" for i in range(10)],
            "label": np.random.randint(1, 4, 10),
        }
    )
    return df


@pytest.fixture
def sample_img_dir(tmpdir):
    img_dir = tmpdir.mkdir("images")
    for i in range(10):
        img = Image.new("RGB", (256, 256), (255, 255, 255))
        img.save(os.path.join(img_dir, f"img{i}.jpeg"), "JPEG")
    return img_dir


def test_build_dataset(sample_df, sample_img_dir):
    seed = 1
    batch_size = 1

    img_dir = sample_img_dir
    df_good = sample_df
    label_map = {1: 0, 2: 1, 3: 2}

    # build training and validation datasets with shuffling
    tra_ds, val_ds = build_dataset(
        img_dir,
        df_good,
        label_map,
        True,
        seed,
        validation_split=0.2,
        batch_size=batch_size,
    )

    # test that tra_ds and val_ds have the correct number of entries
    assert tra_ds.cardinality().numpy() == 8
    assert val_ds.cardinality().numpy() == 2

    # check that the labels are 0,1,2
    for _, label in tra_ds:
        assert np.isin(label.numpy(), [0, 1, 2])
    for _, label in val_ds:
        assert np.isin(label.numpy(), [0, 1, 2])

    # build a single dataset with no shuffling and no validation split
    dataset = build_dataset(
        img_dir,
        df_good,
        label_map,
        False,
        seed=None,
        validation_split=None,
        batch_size=batch_size,
    )

    # check that its labels are 0,1,2
    for _, label in dataset:
        assert np.isin(label.numpy(), [0, 1, 2])


@pytest.fixture
def sample_ds(sample_df, sample_img_dir):
    seed = 1
    batch_size = 1

    img_dir = sample_img_dir
    df_good = sample_df
    label_map = {1: 0, 2: 1, 3: 2}

    # build training and validation datasets with shuffling
    tra_ds, val_ds = build_dataset(
        img_dir,
        df_good,
        label_map,
        True,
        seed,
        validation_split=0.2,
        batch_size=batch_size,
    )

    return tra_ds, val_ds


def test_build_model(sample_ds, capfd):
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    num_classes = 3
    model = build_model(num_classes, metric, loss, print_summary=True)

    # #check that summary was printed
    out, err = capfd.readouterr()
    # Model: "sequential"
    assert 'Model: "sequential"' in out

    # check that the model has the correct number of layers
    assert len(model.layers) == 3

    # fit the model to the dataset
    # (fitting is required to get models metrics)
    tra_ds, val_ds = sample_ds
    model.fit(tra_ds, validation_data=val_ds, epochs=1)

    # check the models metrics
    assert model.metrics_names == ["loss", "accuracy"]


def test_train_model(sample_ds, tmpdir):
    # set up the model
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    num_classes = 3

    # create a simple CNN model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=(256, 256, 3)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # compile the model
    model.compile(optimizer="adam", loss=loss, metrics=[metric])

    # train the model
    tra_ds, val_ds = sample_ds
    checkpoint_dir = tmpdir.mkdir("checkpoints")
    train_model(
        model,
        tra_ds,
        val_ds,
        epochs=3,
        model_name="test_model",
        checkpoint_dir=checkpoint_dir,
        save_best_only=False,
        patience=4,
    )

    # check that there are 3 checkpoints in checkpoint_dir/model_name
    assert len(os.listdir(checkpoint_dir.join("test_model"))) == 3
