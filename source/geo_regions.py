##GEO REGIONS


##pandas for csv reading and shapely for polygons
from pandas import read_csv
import shapely

def get_poly_list() -> (list, dict):
    csv_url = "https://raw.githubusercontent.com/SantanderMetGroup/ATLAS/main/reference-regions/IPCC-WGI-reference-regions-v4_coordinates.csv"
    regions_df = read_csv(csv_url)
    polygons = []
    for index, row in regions_df.iterrows():
        points = []
        for i in range(4, len(row)):
            if type(row[i]) == float:  # NaN
                break
            points.append(row[i].split("|"))
        polygons.append(shapely.Polygon(points))
    name_dict = {acronym: num for acronym, num in enumerate(regions_df["Acronym"])}
    return (polygons, name_dict)


def get_point_region(x, y, polygons, default=99) -> int:
    point = shapely.Point(x, y)
    for i in range(len(polygons)):
        if shapely.contains(polygons[i], point):
            return i
    return default
