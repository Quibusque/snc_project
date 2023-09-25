import pandas as pd

from source.geo_regions import get_poly_list, get_point_region
from source.utils import format_id_to_filename, remove_wrong_entries, delete_wrong_files


def make_labelled_dataframe(csv_path, img_dir, label_key, file_name_key):
    # create a dataframe with the labels
    df = pd.read_csv(csv_path)
    df = format_id_to_filename(df, file_name_key)
    # delete entries in the dataframe that are not in the images folder
    df = remove_wrong_entries(df,img_dir,file_name_key)
    # delete images in the images folder that are not in the dataframe
    delete_wrong_files(df,img_dir,file_name_key)

    polygons, name_dict = get_poly_list()
    print(f"Computing region label for {len(df)} images, this may take a while...")
    df[label_key] = df.apply(lambda x: get_point_region(x.lng, x.lat, polygons), axis=1)
    print(f"Done!")

    return df, name_dict