import pandas as pd
import os

from .geo_regions import get_poly_list, get_point_region


def format_id_to_filename(
    df: pd.DataFrame, file_name_key: str = "file_name"
) -> pd.DataFrame:
    """
    Changes the name of the "id" column of the dataframe to file_name_key and
    appends ".jpeg" to each entry in the column.

    Args:
        df (pd.DataFrame): dataframe to modify
        file_name_key (str): new name of the "id" column. Defaults to "file_name".

    Returns:
        pd.DataFrame: modified dataframe
    """
    df["id"] = df["id"].apply(lambda x: x + ".jpeg")

    df = df.rename(columns={"id": file_name_key})
    return df


def remove_wrong_entries(
    df: pd.DataFrame, img_dir: str, file_name_key: str = "file_name"
) -> pd.DataFrame:
    """
    Removes entries from the dataframe that do not have corresponding files in
    the image directory.

    Args:
        df (pd.DataFrame): dataframe
        img_dir (str): path to image directory
        file_name_key (str): name of the column containing the file names. Defaults to "file_name".
    """
    dir_files_set = set(os.listdir(img_dir))
    df_files_set = set(df[file_name_key].tolist())

    # missing files are in the dataframe but not in the directory
    missing_files = df_files_set - dir_files_set

    # remove entries with df[file_name_key] in missing_files
    df = df[~df[file_name_key].isin(missing_files)]
    return df


def delete_wrong_files(
    df: pd.DataFrame, img_dir: str, file_name_key: str = "file_name"
) -> None:
    """
    Deletes files in the image directory that do not have corresponding entries
    in the dataframe.

    Args:
        df (pd.DataFrame): dataframe
        img_dir (str): path to image directory
        file_name_key (str): name of the column containing the file names. Defaults to "file_name".
    """
    dir_files_set = set(os.listdir(img_dir))
    df_files_set = set(df[file_name_key].tolist())

    # extra files are in the directory but not in the dataframe
    extra_files = dir_files_set - df_files_set

    # delete files in extra_files
    for file_name in extra_files:
        os.remove(os.path.join(img_dir, file_name))


def sample_labels(
    df: pd.DataFrame,
    num_samples: int,
    seed: int,
    label_key: str = "label",
    file_name_key: str = "file_name",
) -> pd.DataFrame:
    """
    This function takes a dataframe with a column of labels and returns
    a dataframe with at most num_samples entries for each label.

    Before sampling, the dataframe is randomly shuffled with seed.
    The dataframe is sorted by alphabetical order of file_name_key after sampling
    because this is the order that the images are loaded in by the tf dataset.

    Args:
        df (pd.DataFrame): dataframe to sample from
        num_samples (int): maximum number of samples for each label
        seed (int): seed for random shuffling
        label_key (str): name of the column containing the labels. Defaults to "label".
        file_name_key (str): name of the column containing the file names. Defaults to "file_name".



    Returns:
        pd.DataFrame: sampled dataframe
    """
    # shuffle the dataframe with seed
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Create a mask to filter rows so that each class has at most num_samples entries
    mask = df.groupby(label_key).cumcount() < num_samples
    df = df[mask]

    # sort dataframe by alphabetical order of file_name column
    df = df.sort_values(by=file_name_key).reset_index(drop=True)

    return df


def prepare_dataframe_and_files_for_training(
    df: pd.DataFrame,
    chosen_labels: list,
    img_dir: str,
    bad_img_dir: str,
    test_img_dir: str,
    num_samples: int,
    seed: int,
    label_key: str = "label",
    file_name_key: str = "file_name",
):
    """
    Given a list of labels, this function:
    1. Moves all images whose labels are not in the list to the bad_img_dir
    2. Creates a dataframe with at most num_samples images for each label
    3. Moves excess images that were not sampled to the test_img_dir

    Args:
        df (pd.DataFrame): dataframe with the labels
        chosen_labels (list): list of labels to keep
        img_dir (str): path to the image directory
        bad_img_dir (str): path to the directory to move bad images to
        test_img_dir (str): path to the directory to move test images to
        num_samples (int): maximum number of samples for each label
        seed (int): seed for random shuffling
        label_key (str): name of the column containing the labels. Defaults to "label".
        file_name_key (str): name of the column containing the file names. Defaults to "file_name".

    Returns:
        df_good (pd.DataFrame): dataframe with the sampled good labels
        df_test (pd.DataFrame): dataframe with the test images
    """
    # create the bad_img_dir and test_img_dir if they don't exist
    if not os.path.exists(bad_img_dir):
        os.makedirs(bad_img_dir)
    if not os.path.exists(test_img_dir):
        os.makedirs(test_img_dir)

    # if the directories bad_img_dir and test_img_dir are not empty, raise an error
    if len(os.listdir(bad_img_dir)) != 0:
        raise ValueError("bad_img_dir must be empty")
    if len(os.listdir(test_img_dir)) != 0:
        raise ValueError("test_img_dir must be empty")

    # create the dataframe with the images that have the chosen labels
    df_chosen = df[df[label_key].isin(chosen_labels)]

    # move all images that have bad labels to the bad_img_dir
    df_bad = df[~df[label_key].isin(chosen_labels)]
    for file_name in df_bad[file_name_key]:
        os.rename(
            os.path.join(img_dir, file_name), os.path.join(bad_img_dir, file_name)
        )

    # df_good samples num_samples images from each class
    df_good = sample_labels(df_chosen, num_samples, seed)

    # df_test contains the images of df_chosen that were not sampled
    df_test = df_chosen[~df_chosen[file_name_key].isin(df_good[file_name_key])]
    for file_name in df_test[file_name_key]:
        os.rename(
            os.path.join(img_dir, file_name), os.path.join(test_img_dir, file_name)
        )

    # sort the dataframes by alphabetical order of file_name_key
    df_good = df_good.sort_values(by=file_name_key).reset_index(drop=True)
    df_test = df_test.sort_values(by=file_name_key).reset_index(drop=True)

    return df_good, df_test


def labels_for_dataset(df: pd.DataFrame, map: dict, label_key: str = "label") -> list:
    """
    This function returns a list of labels for a given dataframe, where the labels
    are mapped according to the dictionary map.
    """
    # list of labels
    label_list = df[label_key].tolist()

    # true_label_list is the list of labels in the range [0,num_classes)
    true_label_list = [map[value] for value in label_list]

    return true_label_list


def reset_images_position(img_dir: str, bad_img_dir: str, test_img_dir: str) -> None:
    """
    This function moves all images in bad_img_dir and test_img_dir back to img_dir
    after checking that directories bad_img_dir and test_img_dir exist.

    Args:
        img_dir (str): path to the image directory
        bad_img_dir (str): path to the directory bad images were moved to
        test_img_dir (str): path to the directory test images were moved to
    """

    if os.path.exists(test_img_dir):
        for file in os.listdir(test_img_dir):
            os.rename(os.path.join(test_img_dir, file), os.path.join(img_dir, file))
    if os.path.exists(bad_img_dir):
        for file in os.listdir(bad_img_dir):
            os.rename(os.path.join(bad_img_dir, file), os.path.join(img_dir, file))


def make_labelled_dataframe(
    csv_path: str,
    img_dir: str,
    label_key: str = "label",
    file_name_key: str = "file_name",
):
    """
    This function takes the csv file with the list of images in the img_dir and
    the coordinates of the images and returns a dataframe with the labels for each image.

    The input csv must have a specific format where the "id" column contains the file names
    without the ".jpeg" extension. The lng and lat columns contain the longitude and
    latitude coordinates of the images.
    The dataframe is "sanitized" because images in the img_dir that are not in the csv
    are deleted and images in the csv that are not in the img_dir are deleted.
    The regions to label the images are computed using the geo_regions module.
    In the output dataframe, the "id" column is renamed to file_name_key where filenames
    have the ".jpeg" extension appended to them. The label for each image is stored in
    a new column with the name label_key.
    name_dict is a dictionary mapping region numbers to region acronyms (see geo_regions.py)

    Args:
        csv_path (str): path to the csv file
        img_dir (str): path to the image directory
        label_key (str): name of the column containing the labels. Defaults to "label".
        file_name_key (str): name of the column containing the file names. Defaults to "file_name".

    Returns:
        df (pd.DataFrame): dataframe with the labels
        name_dict (dict): dictionary mapping region names to region labels
    """
    # create a dataframe with the labels
    df = pd.read_csv(csv_path)
    df = format_id_to_filename(df, file_name_key)

    # delete entries in the dataframe that are not in the images folder
    df = remove_wrong_entries(df, img_dir, file_name_key)

    # delete images in the images folder that are not in the dataframe
    delete_wrong_files(df, img_dir, file_name_key)

    polygons, name_dict = get_poly_list()
    print(f"Computing region label for images, this may take a while...")
    df[label_key] = df.apply(lambda x: get_point_region(x.lng, x.lat, polygons), axis=1)
    print(f"Done!")

    return df, name_dict
