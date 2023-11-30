import pyproj


def convert_geographic_to_projection(lat, lon):
    n_zone = lon//6 + 1
    # Определение системы координат СК-42
    crs_src = pyproj.CRS.from_epsg(4740)

    # Определение системы координат WGS84 (географические координаты)
    crs_dst = pyproj.CRS.from_epsg(28400 + n_zone)

    # Создание преобразователя координат
    transformer = pyproj.Transformer.from_crs(crs_src, crs_dst)

    x, y = transformer.transform(lat, lon)
    return x, y


def convert_projection_to_geographic(x, y):
    n_zone = y // 10 ** 6
    # Определение системы координат СК-42
    crs_src = pyproj.CRS.from_epsg(28400 + n_zone)

    # Определение системы координат WGS84 (географические координаты)
    crs_dst = pyproj.CRS.from_epsg(4740)

    # Создание преобразователя координат
    transformer = pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    lon, lat = transformer.transform(y, x)
    return lat, lon

