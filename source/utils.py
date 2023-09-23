import pandas as pd
import os


def format_id_to_filename(df: pd.DataFrame, file_name_key:str) -> pd.DataFrame:
    # add a ".jpeg" to the id column of the dataframe
    df["id"] = df["id"].apply(lambda x: x + ".jpeg")
    # and rename the id to file_name
    df = df.rename(columns={"id": file_name_key})
    return df


def remove_wrong_entries(df: pd.DataFrame, img_dir: str) -> None:
    dir_files_set = set(os.listdir(img_dir))
    df_files_set = set(df["file_name"].tolist())

    # missing files are in the dataframe but not in the directory
    missing_files = df_files_set - dir_files_set

    # remove the missing files from the dataframe
    df.query("file_name not in @missing_files", inplace=True)