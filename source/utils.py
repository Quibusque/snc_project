import pandas as pd
import os


def format_id_to_filename(df: pd.DataFrame, file_name_key: str) -> pd.DataFrame:
    """
    Changes the name of the "id" column of the dataframe to file_name_key and
    appends ".jpeg" to each entry in the column.

    Args:
        df (pd.DataFrame): dataframe to modify
        file_name_key (str): new name of the "id" column

    Returns:
        pd.DataFrame: modified dataframe
    """
    df["id"] = df["id"].apply(lambda x: x + ".jpeg")

    df = df.rename(columns={"id": file_name_key})
    return df


def remove_wrong_entries(df: pd.DataFrame, img_dir: str, file_name_key: str) -> None:
    """
    Removes entries from the dataframe that do not have corresponding files in
    the image directory.

    Args:
        df (pd.DataFrame): dataframe
        img_dir (str): path to image directory
    """
    dir_files_set = set(os.listdir(img_dir))
    df_files_set = set(df[file_name_key].tolist())

    # missing files are in the dataframe but not in the directory
    missing_files = df_files_set - dir_files_set

    # remove entries with df[file_name_key] in missing_files
    df = df[~df[file_name_key].isin(missing_files)]
    return df


def sample_classes(
    df: pd.DataFrame, label_key: str, num_samples: int, file_name_key: str, seed: int
) -> pd.DataFrame:
    """
    This function takes a dataframe with a column of labels and returns
    a dataframe with at most num_samples entries for each label.

    Before sampling, the dataframe is randomly shuffled with seed.
    The dataframe is sorted by alphabetical order of file_name_key after sampling
    because this is the order that the images are loaded in by the tf dataset.

    Args:
        df (pd.DataFrame): dataframe to sample from
        label_key (str): name of the column containing the labels
        num_samples (int): maximum number of samples for each label
        file_name_key (str): name of the column containing the file names
        seed (int): seed for random shuffling

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
