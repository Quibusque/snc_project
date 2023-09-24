##GEO REGIONS


##pandas for csv reading and shapely for polygons
from pandas import read_csv
import shapely


def get_poly_list() -> (list, dict):
    """
    Returns a list of polygons and a dictionary of corresponding region names.

    The list of polygons is a list of shapely.Polygon, each representing a
    climate region. The dictionary of region names is a dictionary mapping
    region index to region acronyms.

    The source for the csv file is: https://github.com/IPCC-WG1/Atlas
    """
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


def get_point_region(x: float, y: float, polygons: list, default: int = 99) -> int:
    """
    Given a list of polygons and a point (x, y), returns the index of the
    polygon that contains the point. Default is returned if no polygon
    contains the point.

    Args:
        x (float): x coordinate of point
        y (float): y coordinate of point
        polygons (list): list of shapely.Polygon
        default (int): default value to return if no polygon contains point
    
    Returns:
        int: index of polygon that contains point
    """
    point = shapely.Point(x, y)
    for i in range(len(polygons)):
        if shapely.contains(polygons[i], point):
            return i
    return default
