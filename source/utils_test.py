from .utils import (
    format_id_to_filename,
    remove_wrong_entries,
    delete_wrong_files,
    sample_labels,
    prepare_dataframe_and_files_for_training,
    reset_images_position,
    labels_for_dataset,
    make_labelled_dataframe
)
import pytest
import pandas as pd
import os


def test_format_id_to_filename():
    sample_unformatted_dataframe = pd.DataFrame({"id": ["image1", "image2", "image3"]})
    file_name_key = "file_name"
    result_df = format_id_to_filename(sample_unformatted_dataframe, file_name_key)

    # Check if the result_df has file_name column
    assert file_name_key in result_df.columns.tolist()

    # Check if the values in the file_name column have the ".jpeg" extension
    assert result_df[file_name_key].str.endswith(".jpeg").all()

    # Check if the original id column is no longer present
    assert "id" not in result_df.columns.tolist()


# Setting up a temporary directory with files
@pytest.fixture
def temp_img_dir(tmpdir):
    img_dir = str(tmpdir)

    dir_files = ["image1.jpeg", "image2.jpeg", "image3.jpeg"]
    for file in dir_files:
        with open(os.path.join(img_dir, file), "w") as f:
            f.write("sample content")

    return img_dir


# Sample test data
@pytest.fixture
def sample_dataframe():
    data = {"file_name": ["image1.jpeg", "image2.jpeg", "image3.jpeg"]}
    return pd.DataFrame(data)


def test_remove_wrong_entries(sample_dataframe, temp_img_dir):
    # add a wrong entry to the dataframe with concat
    wrong_entry = pd.DataFrame({"file_name": ["image4.jpeg"]})
    sample_dataframe = pd.concat([sample_dataframe, wrong_entry], ignore_index=True)

    result_df = remove_wrong_entries(sample_dataframe, temp_img_dir)

    # Check that the wrong entry is removed
    assert "image4.jpeg" not in result_df["file_name"].tolist()

    # Check that the correct entries are still present
    original_entries = ["image1.jpeg", "image2.jpeg", "image3.jpeg"]
    assert set(original_entries) == set(result_df["file_name"].tolist())


def test_delete_wrong_files(sample_dataframe, temp_img_dir):
    # add a wrong file to the directory
    with open(os.path.join(temp_img_dir, "image4.jpeg"), "w") as f:
        f.write("sample content")

    delete_wrong_files(sample_dataframe, temp_img_dir)

    # Check that the wrong file is deleted
    assert "image4.jpeg" not in os.listdir(temp_img_dir)

    # Check that the correct files are still present
    original_files = ["image1.jpeg", "image2.jpeg", "image3.jpeg"]
    assert set(original_files) == set(os.listdir(temp_img_dir))


@pytest.fixture
def big_sample_dataframe():
    # this dataframe has 1000 entries and 10 labels
    data = {"file_name": [f"image{i}.jpeg" for i in range(0, 1000)]}
    data["label"] = [i for i in range(0, 10)] * 100
    dataframe = pd.DataFrame(data)
    # shuffle the dataframe
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    return dataframe


def test_sample_labels_big(big_sample_dataframe):
    # num_samples < 100 to test that entries are removed
    num_samples = 34
    seed = 0

    result_df = sample_labels(big_sample_dataframe, num_samples, seed)

    # Check that the result_df has the correct number of entries
    assert len(result_df) == num_samples * 10

    # Check that the result_df has the correct number of entries for each label
    assert result_df.groupby("label").size().tolist() == [num_samples] * 10

    # Check that the dataframe is sorted by "file_name"
    assert result_df["file_name"].tolist() == sorted(result_df["file_name"].tolist())


def test_sample_labels_small(big_sample_dataframe):
    # num_samples > 100 to test that entries are not removed
    num_samples = 150
    seed = 0

    result_df = sample_labels(big_sample_dataframe, num_samples, seed)

    # Check that the dataframe is sorted by "file_name"
    assert result_df["file_name"].tolist() == sorted(result_df["file_name"].tolist())

    # check that result_df is the same as big_sample_dataframe (after sorting it)
    assert result_df.equals(
        big_sample_dataframe.sort_values(by=["file_name"]).reset_index(drop=True)
    )


@pytest.fixture
def sample_img_dir(tmpdir):
    img_dir = str(tmpdir)

    dir_files = [f"image{i}.jpeg" for i in range(0, 1000)]
    for file in dir_files:
        with open(os.path.join(img_dir, file), "w") as f:
            f.write("sample content")

    return img_dir


def test_prepare_dataframe_and_files_for_training(
    big_sample_dataframe, sample_img_dir, tmpdir
):
    num_samples = 42
    seed = 0
    chosen_labels = [0, 1, 2, 3, 4]

    img_dir = sample_img_dir
    bad_img_dir = os.path.join(str(tmpdir), "bad_img_dir")
    test_img_dir = os.path.join(str(tmpdir), "test_img_dir")

    df_good, df_test = prepare_dataframe_and_files_for_training(
        big_sample_dataframe,
        chosen_labels,
        img_dir,
        bad_img_dir,
        test_img_dir,
        num_samples,
        seed,
    )

    # check that all the images in df_bad are in bad_img_dir
    df_bad = big_sample_dataframe[~big_sample_dataframe["label"].isin(chosen_labels)]
    assert set(df_bad["file_name"].tolist()).issubset(set(os.listdir(bad_img_dir)))

    # check that the df_good has the correct number of entries
    assert len(df_good) == num_samples * len(chosen_labels)
    # check that all the labels in df_good are in chosen_labels
    assert set(df_good["label"].tolist()).issubset(set(chosen_labels))
    # Check that images in df_good are all in img_dir
    assert set(df_good["file_name"].tolist()).issubset(set(os.listdir(img_dir)))

    # Check that number of total entries is preserved
    assert len(df_good) + len(df_bad) + len(df_test) == len(big_sample_dataframe)

    # check that all the labels in df_test are in chosen_labels
    assert set(df_test["label"].tolist()).issubset(set(chosen_labels))
    # Check that images in df_test are all in test_img_dir
    assert set(df_test["file_name"].tolist()).issubset(set(os.listdir(test_img_dir)))

    # test that we raise an error if we run this function on non-empty
    # bad_img_dir or test_img_dir

    # make an empty dir
    os.makedirs(empty_dir := os.path.join(str(tmpdir), "empty_dir"))

    with pytest.raises(ValueError):
        prepare_dataframe_and_files_for_training(
            big_sample_dataframe,
            chosen_labels,
            img_dir,
            empty_dir,
            test_img_dir,
            num_samples,
            seed,
        )
    with pytest.raises(ValueError):
        prepare_dataframe_and_files_for_training(
            big_sample_dataframe,
            chosen_labels,
            img_dir,
            bad_img_dir,
            empty_dir,
            num_samples,
            seed,
        )


def test_reset_images_position(sample_img_dir, tmpdir):
    img_dir = sample_img_dir
    bad_img_dir = os.path.join(str(tmpdir), "bad_img_dir")
    test_img_dir = os.path.join(str(tmpdir), "test_img_dir")

    # create the bad_img_dir and test_img_dir if they don't exist
    if not os.path.exists(bad_img_dir):
        os.makedirs(bad_img_dir)
    if not os.path.exists(test_img_dir):
        os.makedirs(test_img_dir)

    # add 20 images to bad_img_dir and 20 images to test_img_dir
    for i in range(0, 20):
        with open(os.path.join(bad_img_dir, f"bad_image{i}.jpeg"), "w") as f:
            f.write("sample content")
        with open(os.path.join(test_img_dir, f"test_image{i}.jpeg"), "w") as f:
            f.write("sample content")

    # check that bad_img_dir and test_img_dir have 20 images each
    assert len(os.listdir(bad_img_dir)) == 20
    assert len(os.listdir(test_img_dir)) == 20

    # save the original number of images in img_dir
    original_num_images = len(os.listdir(img_dir))

    reset_images_position(img_dir, bad_img_dir, test_img_dir)

    # check that bad_img_dir and test_img_dir are empty
    assert len(os.listdir(bad_img_dir)) == 0
    assert len(os.listdir(test_img_dir)) == 0

    # check that img_dir size is 40 longer
    assert len(os.listdir(img_dir)) == original_num_images + 40


def test_labels_for_dataset():
    # make a dummy dataframe
    data = {"label": [0, 1, 2], "id": ["image1", "image2", "image3"]}
    df = pd.DataFrame(data)
    # make a dummy label map
    label_map = {0: "label_0", 1: "label_1", 2: "label_2"}
    # check that the function works
    assert labels_for_dataset(df, label_map, "label") == [
        "label_0",
        "label_1",
        "label_2",
    ]



def test_make_labelled_dataframe(tmpdir, capfd):
    # make a dummy csv file
    csv_path = os.path.join(str(tmpdir), "dummy.csv")
    # csv should have a lng,lat,id columns
    data = {
        "lng": [1, 2, 3],
        "lat": [1, 2, 3],
        "id": ["image1", "image2", "image3"],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    # make a dummy image directory
    img_dir = os.path.join(str(tmpdir), "images")
    os.makedirs(img_dir)
    # fill it with some images
    for file in ["image1.jpeg", "image2.jpeg", "image4.jpeg"]:
        with open(os.path.join(img_dir, file), "w") as f:
            f.write("sample content")

    df, name_dict = make_labelled_dataframe(csv_path, img_dir)
    # check that the dataframe has the correct entries
    assert df["file_name"].tolist() == ["image1.jpeg", "image2.jpeg"]
    # check that the dataframe has the correct columns
    assert set(df.columns) == set(["lng", "lat", "file_name", "label"])
    # check that image4.jpeg has been removed from files
    assert "image4.jpeg" not in os.listdir(img_dir)

    out, err = capfd.readouterr()
    # check that the function prints the correct message
    assert "Computing region label for images, this may take a while..." in out
    assert "Done!" in out
