import shapely
from ..geo_regions import get_poly_list, get_point_region


def test_get_poly_list():
    polygons, name_dict = get_poly_list()
    assert len(polygons) == 62
    assert len(name_dict) == 62

    true_name_dict = {
        0: 'GIC,0',  1: 'NWN,1',  2: 'NEN,2',  3: 'WNA,3',  4: 'CNA,4',
        5: 'ENA,5',  6: 'NCA,6',  7: 'SCA,7',  8: 'CAR,8',  9: 'NWS,9',
        10: 'NSA,10', 11: 'NES,11', 12: 'SAM,12', 13: 'SWS,13', 14: 'SES,14',
        15: 'SSA,15', 16: 'NEU,16', 17: 'WCE,17', 18: 'EEU,18', 19: 'MED,19',
        20: 'SAH,20', 21: 'WAF,21', 22: 'CAF,22', 23: 'NEAF,23', 24: 'SEAF,24',
        25: 'WSAF,25', 26: 'ESAF,26', 27: 'MDG,27', 28: 'RAR,28', 29: 'RAR*,29',
        30: 'WSB,30', 31: 'ESB,31', 32: 'RFE,32', 33: 'WCA,33', 34: 'ECA,34',
        35: 'TIB,35', 36: 'EAS,36', 37: 'ARP,37', 38: 'SAS,38', 39: 'SEA,39',
        40: 'NAU,40', 41: 'CAU,41', 42: 'EAU,42', 43: 'SAU,43', 44: 'NZ,44',
        45: 'EAN,45', 46: 'WAN,46', 47: 'ARO,47', 48: 'NPO,48', 49: 'NPO*,49',
        50: 'EPO,50', 51: 'EPO*,51', 52: 'SPO,52', 53: 'SPO*,53', 54: 'NAO,54',
        55: 'EAO,55', 56: 'SAO,56', 57: 'ARS,57', 58: 'BOB,58', 59: 'EIO,59',
        60: 'SIO,60', 61: 'SOO,61'}

    assert name_dict == true_name_dict

    true_poly_list = [
        shapely.Polygon([shapely.Point(-10.0, 62.0), shapely.Point(-38.0, 62.0), shapely.Point(-42.0, 58.0), shapely.Point(-50.0, 58.0), shapely.Point(-82.0, 85.0), shapely.Point(-10.0, 85.0), shapely.Point(-10.0, 62.0), ]),
        shapely.Polygon([shapely.Point(-105.0, 50.0), shapely.Point(-130.0, 50.0), shapely.Point(-143.0, 58.0), shapely.Point(-168.0, 52.5), shapely.Point(-168.0, 72.6), shapely.Point(-129.0, 72.6), shapely.Point(-125.0, 77.6), shapely.Point(-105.0, 81.0), shapely.Point(-105.0, 50.0), ]),
        shapely.Polygon([shapely.Point(-50.0, 50.0), shapely.Point(-50.0, 58.0), shapely.Point(-82.0, 85.0), shapely.Point(-105.0, 81.0), shapely.Point(-105.0, 50.0), shapely.Point(-50.0, 50.0), ]),
        shapely.Polygon([shapely.Point(-130.0, 50.0), shapely.Point(-122.5, 33.8), shapely.Point(-105.0, 33.8), shapely.Point(-105.0, 50.0), shapely.Point(-130.0, 50.0), ]),
        shapely.Polygon([shapely.Point(-90.0, 50.0), shapely.Point(-90.0, 25.0), shapely.Point(-105.0, 33.8), shapely.Point(-105.0, 50.0), shapely.Point(-90.0, 50.0), ]),
        shapely.Polygon([shapely.Point(-70.0, 25.0), shapely.Point(-90.0, 25.0), shapely.Point(-90.0, 50.0), shapely.Point(-50.0, 50.0), shapely.Point(-77.0, 31.0), shapely.Point(-70.0, 25.0), ]),
        shapely.Polygon([shapely.Point(-90.0, 25.0), shapely.Point(-104.5, 16.0), shapely.Point(-122.5, 33.8), shapely.Point(-105.0, 33.8), shapely.Point(-90.0, 25.0), ]),
        shapely.Polygon([shapely.Point(-75.0, 12.0), shapely.Point(-83.4, 2.2), shapely.Point(-104.5, 16.0), shapely.Point(-90.0, 25.0), shapely.Point(-75.0, 12.0), ]),
        shapely.Polygon([shapely.Point(-75.0, 12.0), shapely.Point(-90.0, 25.0), shapely.Point(-70.0, 25.0), shapely.Point(-55.0, 12.0), shapely.Point(-75.0, 12.0), ]),
        shapely.Polygon([shapely.Point(-75.0, 12.0), shapely.Point(-83.4, 2.2), shapely.Point(-83.4, -10.0), shapely.Point(-79.0, -15.0), shapely.Point(-72.0, -15.0), shapely.Point(-72.0, 12.0), shapely.Point(-75.0, 12.0), ]),
        shapely.Polygon([shapely.Point(-72.0, 12.0), shapely.Point(-72.0, -8.0), shapely.Point(-50.0, -8.0), shapely.Point(-50.0, 7.6), shapely.Point(-55.0, 12.0), shapely.Point(-72.0, 12.0), ]),
        shapely.Polygon([shapely.Point(-34.0, -20.0), shapely.Point(-50.0, -20.0), shapely.Point(-50.0, 0.0), shapely.Point(-34.0, 0.0), shapely.Point(-34.0, -20.0), ]),
        shapely.Polygon([shapely.Point(-66.4, -20.0), shapely.Point(-72.0, -15.0), shapely.Point(-72.0, -8.0), shapely.Point(-50.0, -8.0), shapely.Point(-50.0, -20.0), shapely.Point(-66.4, -20.0), ]),
        shapely.Polygon([shapely.Point(-72.0, -15.0), shapely.Point(-66.4, -20.0), shapely.Point(-71.5, -47.0), shapely.Point(-79.0, -47.0), shapely.Point(-74.6, -20.0), shapely.Point(-79.0, -15.0), shapely.Point(-72.0, -15.0), ]),
        shapely.Polygon([shapely.Point(-34.0, -20.0), shapely.Point(-56.0, -40.0), shapely.Point(-70.2, -40.0), shapely.Point(-66.4, -20.0), shapely.Point(-34.0, -20.0), ]),
        shapely.Polygon([shapely.Point(-79.0, -56.0), shapely.Point(-79.0, -47.0), shapely.Point(-71.5, -47.0), shapely.Point(-70.2, -40.0), shapely.Point(-56.0, -40.0), shapely.Point(-56.0, -56.0), shapely.Point(-79.0, -56.0), ]),
        shapely.Polygon([shapely.Point(-10.0, 48.0), shapely.Point(-10.0, 72.6), shapely.Point(40.0, 72.6), shapely.Point(40.0, 61.3), shapely.Point(-10.0, 48.0), ]),
        shapely.Polygon([shapely.Point(-10.0, 45.0), shapely.Point(-10.0, 48.0), shapely.Point(40.0, 61.3), shapely.Point(40.0, 45.0), shapely.Point(-10.0, 45.0), ]),
        shapely.Polygon([shapely.Point(40.0, 45.0), shapely.Point(40.0, 65.0), shapely.Point(60.0, 65.0), shapely.Point(60.0, 45.0), shapely.Point(40.0, 45.0), ]),
        shapely.Polygon([shapely.Point(-10.0, 30.0), shapely.Point(-10.0, 45.0), shapely.Point(40.0, 45.0), shapely.Point(40.0, 30.0), shapely.Point(-10.0, 30.0), ]),
        shapely.Polygon([shapely.Point(-20.0, 14.7), shapely.Point(-20.0, 30.0), shapely.Point(33.0, 30.0), shapely.Point(42.1, 14.7), shapely.Point(-20.0, 14.7), ]),
        shapely.Polygon([shapely.Point(-20.0, 7.6), shapely.Point(-20.0, 14.7), shapely.Point(15.0, 14.7), shapely.Point(8.0, 0.0), shapely.Point(-20.0, 7.6), ]),
        shapely.Polygon([shapely.Point(8.0, -10.0), shapely.Point(8.0, 0.0), shapely.Point(15.0, 14.7), shapely.Point(27.0, 14.7), shapely.Point(27.0, -10.0), shapely.Point(8.0, -10.0), ]),
        shapely.Polygon([shapely.Point(27.0, 2.3), shapely.Point(27.0, 14.7), shapely.Point(42.1, 14.7), shapely.Point(43.7, 12.0), shapely.Point(53.0, 15.0), shapely.Point(53.0, 7.0), shapely.Point(46.5, 2.3), shapely.Point(27.0, 2.3), ]),
        shapely.Polygon([shapely.Point(27.0, -10.0), shapely.Point(27.0, 2.3), shapely.Point(46.5, 2.3), shapely.Point(46.5, -10.0), shapely.Point(27.0, -10.0), ]),
        shapely.Polygon([shapely.Point(8.0, -36.0), shapely.Point(8.0, -10.0), shapely.Point(25.0, -10.0), shapely.Point(25.0, -36.0), shapely.Point(8.0, -36.0), ]),
        shapely.Polygon([shapely.Point(25.0, -10.0), shapely.Point(25.0, -36.0), shapely.Point(31.0, -36.0), shapely.Point(46.5, -10.0), shapely.Point(25.0, -10.0), ]),
        shapely.Polygon([shapely.Point(36.2, -27.0), shapely.Point(46.5, -10.0), shapely.Point(53.0, -10.0), shapely.Point(53.0, -27.0), shapely.Point(36.2, -27.0), ]),
        shapely.Polygon([shapely.Point(40.0, 65.0), shapely.Point(40.0, 72.6), shapely.Point(94.0, 82.0), shapely.Point(180.0, 73.8), shapely.Point(180.0, 65.0), shapely.Point(40.0, 65.0), ]),
        shapely.Polygon([shapely.Point(-180.0, 73.8), shapely.Point(-180.0, 65.0), shapely.Point(-168.0, 65.0), shapely.Point(-168.0, 72.6), shapely.Point(-180.0, 73.8), ]),
        shapely.Polygon([shapely.Point(60.0, 45.0), shapely.Point(60.0, 65.0), shapely.Point(90.0, 65.0), shapely.Point(90.0, 45.0), shapely.Point(60.0, 45.0), ]),
        shapely.Polygon([shapely.Point(90.0, 45.0), shapely.Point(90.0, 65.0), shapely.Point(130.0, 65.0), shapely.Point(130.0, 45.0), shapely.Point(90.0, 45.0), ]),
        shapely.Polygon([shapely.Point(130.0, 45.0), shapely.Point(130.0, 65.0), shapely.Point(180.0, 65.0), shapely.Point(180.0, 59.9), shapely.Point(157.0, 50.0), shapely.Point(152.0, 45.0), shapely.Point(130.0, 45.0), ]),
        shapely.Polygon([shapely.Point(40.0, 30.0), shapely.Point(40.0, 45.0), shapely.Point(75.0, 45.0), shapely.Point(75.0, 30.0), shapely.Point(60.0, 30.0), shapely.Point(60.0, 23.5), shapely.Point(47.6, 30.0), shapely.Point(40.0, 30.0), ]),
        shapely.Polygon([shapely.Point(75.0, 37.0), shapely.Point(75.0, 45.0), shapely.Point(117.0, 45.0), shapely.Point(108.0, 37.0), shapely.Point(75.0, 37.0), ]),
        shapely.Polygon([shapely.Point(75.0, 30.0), shapely.Point(75.0, 37.0), shapely.Point(100.0, 37.0), shapely.Point(100.0, 30.0), shapely.Point(88.0, 26.0), shapely.Point(75.0, 30.0), ]),
        shapely.Polygon([shapely.Point(100.0, 19.5), shapely.Point(100.0, 37.0), shapely.Point(108.0, 37.0), shapely.Point(117.0, 45.0), shapely.Point(152.0, 45.0), shapely.Point(132.0, 25.0), shapely.Point(132.0, 19.5), shapely.Point(100.0, 19.5), ]),
        shapely.Polygon([shapely.Point(33.0, 30.0), shapely.Point(47.6, 30.0), shapely.Point(60.0, 23.5), shapely.Point(60.0, 19.5), shapely.Point(53.0, 15.0), shapely.Point(43.7, 12.0), shapely.Point(33.0, 30.0), ]),
        shapely.Polygon([shapely.Point(60.0, 23.5), shapely.Point(60.0, 30.0), shapely.Point(75.0, 30.0), shapely.Point(88.0, 26.0), shapely.Point(100.0, 30.0), shapely.Point(100.0, 19.5), shapely.Point(95.0, 19.5), shapely.Point(87.0, 19.5), shapely.Point(79.0, 7.0), shapely.Point(76.0, 7.0), shapely.Point(70.0, 19.5), shapely.Point(66.5, 23.5), shapely.Point(60.0, 23.5), ]),
        shapely.Polygon([shapely.Point(93.0, -10.0), shapely.Point(93.0, 19.5), shapely.Point(132.0, 19.5), shapely.Point(132.0, 5.0), shapely.Point(155.0, -10.0), shapely.Point(93.0, -10.0), ]),
        shapely.Polygon([shapely.Point(110.0, -20.0), shapely.Point(110.0, -10.0), shapely.Point(155.0, -10.0), shapely.Point(155.0, -20.0), shapely.Point(110.0, -20.0), ]),
        shapely.Polygon([shapely.Point(110.0, -30.0), shapely.Point(110.0, -20.0), shapely.Point(145.5, -20.0), shapely.Point(145.5, -32.9), shapely.Point(140.0, -30.0), shapely.Point(110.0, -30.0), ]),
        shapely.Polygon([shapely.Point(145.5, -32.9), shapely.Point(145.5, -20.0), shapely.Point(155.0, -20.0), shapely.Point(155.0, -38.0), shapely.Point(145.5, -32.9), ]),
        shapely.Polygon([shapely.Point(110.0, -36.0), shapely.Point(110.0, -30.0), shapely.Point(140.0, -30.0), shapely.Point(155.0, -38.0), shapely.Point(155.0, -50.0), shapely.Point(110.0, -36.0), ]),
        shapely.Polygon([shapely.Point(155.0, -50.0), shapely.Point(155.0, -30.0), shapely.Point(180.0, -30.0), shapely.Point(180.0, -50.0), shapely.Point(155.0, -50.0), ]),
        shapely.Polygon([shapely.Point(-180.0, -90.0), shapely.Point(-180.0, -83.0), shapely.Point(-56.0, -83.0), shapely.Point(-56.0, -75.0), shapely.Point(-25.0, -75.0), shapely.Point(5.0, -64.0), shapely.Point(180.0, -64.0), shapely.Point(180.0, -90.0), shapely.Point(-180.0, -90.0), ]),
        shapely.Polygon([shapely.Point(-180.0, -83.0), shapely.Point(-180.0, -70.0), shapely.Point(-80.0, -70.0), shapely.Point(-65.0, -62.0), shapely.Point(-56.0, -62.0), shapely.Point(-56.0, -83.0), shapely.Point(-180.0, -83.0), ]),
        shapely.Polygon([shapely.Point(-180.0, 90.0), shapely.Point(-180.0, 73.8), shapely.Point(-168.0, 72.6), shapely.Point(-129.0, 72.6), shapely.Point(-125.0, 77.6), shapely.Point(-82.0, 85.0), shapely.Point(-10.0, 85.0), shapely.Point(-10.0, 72.6), shapely.Point(40.0, 72.6), shapely.Point(94.0, 82.0), shapely.Point(180.0, 73.8), shapely.Point(180.0, 90.0), shapely.Point(-180.0, 90.0), ]),
        shapely.Polygon([shapely.Point(132.0, 7.6), shapely.Point(132.0, 25.0), shapely.Point(157.0, 50.0), shapely.Point(180.0, 59.9), shapely.Point(180.0, 7.6), shapely.Point(132.0, 7.6), ]),
        shapely.Polygon([shapely.Point(-180.0, 7.6), shapely.Point(-180.0, 65.0), shapely.Point(-168.0, 65.0), shapely.Point(-168.0, 52.5), shapely.Point(-143.0, 58.0), shapely.Point(-130.0, 50.0), shapely.Point(-122.5, 33.8), shapely.Point(-104.5, 16.0), shapely.Point(-91.7, 7.6), shapely.Point(-180.0, 7.6), ]),
        shapely.Polygon([shapely.Point(155.0, -10.0), shapely.Point(132.0, 5.0), shapely.Point(132.0, 7.6), shapely.Point(180.0, 7.6), shapely.Point(180.0, -10.0), shapely.Point(155.0, -10.0), ]),
        shapely.Polygon([shapely.Point(-180.0, -10.0), shapely.Point(-180.0, 7.6), shapely.Point(-91.7, 7.6), shapely.Point(-83.4, 2.2), shapely.Point(-83.4, -10.0), shapely.Point(-180.0, -10.0), ]),
        shapely.Polygon([shapely.Point(180.0, -10.0), shapely.Point(180.0, -30.0), shapely.Point(155.0, -30.0), shapely.Point(155.0, -10.0), shapely.Point(180.0, -10.0), ]),
        shapely.Polygon([shapely.Point(-180.0, -56.0), shapely.Point(-180.0, -10.0), shapely.Point(-83.4, -10.0), shapely.Point(-74.6, -20.0), shapely.Point(-79.0, -47.0), shapely.Point(-79.0, -56.0), shapely.Point(-180.0, -56.0), ]),
        shapely.Polygon([shapely.Point(-50.0, 7.6), shapely.Point(-77.0, 31.0), shapely.Point(-50.0, 50.0), shapely.Point(-50.0, 58.0), shapely.Point(-42.0, 58.0), shapely.Point(-38.0, 62.0), shapely.Point(-10.0, 62.0), shapely.Point(-10.0, 30.0), shapely.Point(-20.0, 30.0), shapely.Point(-20.0, 7.6), shapely.Point(-50.0, 7.6), ]),
        shapely.Polygon([shapely.Point(-34.0, -10.0), shapely.Point(-34.0, 0.0), shapely.Point(-50.0, 0.0), shapely.Point(-50.0, 7.6), shapely.Point(-20.0, 7.6), shapely.Point(8.0, 0.0), shapely.Point(8.0, -10.0), shapely.Point(-34.0, -10.0), ]),
        shapely.Polygon([shapely.Point(-56.0, -56.0), shapely.Point(-56.0, -40.0), shapely.Point(-34.0, -20.0), shapely.Point(-34.0, -10.0), shapely.Point(8.0, -10.0), shapely.Point(8.0, -36.0), shapely.Point(-56.0, -56.0), ]),
        shapely.Polygon([shapely.Point(53.0, 7.0), shapely.Point(53.0, 15.0), shapely.Point(60.0, 19.5), shapely.Point(60.0, 23.5), shapely.Point(66.5, 23.5), shapely.Point(70.0, 19.5), shapely.Point(76.0, 7.0), shapely.Point(53.0, 7.0), ]),
        shapely.Polygon([shapely.Point(79.0, 7.0), shapely.Point(87.0, 19.5), shapely.Point(93.0, 19.5), shapely.Point(93.0, 7.0), shapely.Point(79.0, 7.0), ]),
        shapely.Polygon([shapely.Point(46.5, -10.0), shapely.Point(46.5, 2.3), shapely.Point(53.0, 7.0), shapely.Point(93.0, 7.0), shapely.Point(93.0, -10.0), shapely.Point(46.5, -10.0), ]),
        shapely.Polygon([shapely.Point(36.2, -27.0), shapely.Point(53.0, -27.0), shapely.Point(53.0, -10.0), shapely.Point(110.0, -10.0), shapely.Point(110.0, -36.0), shapely.Point(31.0, -36.0), shapely.Point(36.2, -27.0), ]),
        shapely.Polygon([shapely.Point(-180.0, -56.0), shapely.Point(-180.0, -70.0), shapely.Point(-80.0, -70.0), shapely.Point(-65.0, -62.0), shapely.Point(-56.0, -62.0), shapely.Point(-56.0, -75.0), shapely.Point(-25.0, -75.0), shapely.Point(5.0, -64.0), shapely.Point(180.0, -64.0), shapely.Point(180.0, -50.0), shapely.Point(155.0, -50.0), shapely.Point(110.0, -36.0), shapely.Point(8.0, -36.0), shapely.Point(-56.0, -56.0), shapely.Point(-180.0, -56.0), ]),
    ]
    assert polygons == true_poly_list


# sample polygon list
def poly_list():
    # two simple squares
    poly_list = [
        shapely.Polygon(
            [
                shapely.Point(0, 0),
                shapely.Point(0, 1),
                shapely.Point(1, 1),
                shapely.Point(1, 0),
            ]
        ),
        shapely.Polygon(
            [
                shapely.Point(2, 2),
                shapely.Point(2, 3),
                shapely.Point(3, 3),
                shapely.Point(3, 2),
            ]
        ),
    ]
    return poly_list


# sample point list
def point_list():
    point_list = [
        (0.5, 0.5),
        (2.5, 2.5),
        (0.5, 2.5),
        (2.5, 0.5),
        (0, 0),
        (2, 2),
    ]
    return point_list


def test_get_point_region():
    result_list = []
    default = 99
    for x, y in point_list():
        result_list.append(get_point_region(x, y, poly_list(), default))

    true_result_list = [0,1,default,default,default,default]
    assert result_list == true_result_list
