import pandas as pd
import os


def format_id_to_filename(df: pd.DataFrame, file_name_key:str) -> pd.DataFrame:
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


def remove_wrong_entries(df: pd.DataFrame, img_dir: str, file_name_key:str) -> None:
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