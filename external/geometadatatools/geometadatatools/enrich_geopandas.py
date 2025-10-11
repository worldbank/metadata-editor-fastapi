# import geopandas as gpd
# import pygeohash as Geohash


# def geohash(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
#     """ "
#     Generate a geohash for each row in the GeoDataFrame.

#     Args:
#         gdf (gpd.GeoDataFrame): The GeoDataFrame to generate geohashes for.

#     Returns:
#         gpd.GeoSeries: A GeoSeries of geohashes.

#     Example:
#     ```python
#     import geopandas as gpd
#     import pygeohash as Geohash

#     # Create a sample GeoDataFrame
#     data = {'geometry': [Point(1, 2), Point(3, 4)]}
#     gdf = gpd.GeoDataFrame(data, geometry='geometry')
#     # Generate geohashes
#     gdf['geohash'] = geohash(gdf)
#     ```
#     """
#     # beware area and centroid calculations since: "UserWarning: Geometry is in a geographic CRS. Results from
#     # 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before
#     # this operation."

#     return Geohash.encode(gdf.geometry.centroid.y, gdf.geometry.centroid.x)


# def enrich_geopandas(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     """
#     Enriches a GeoDataFrame with additional metadata.

#     Args:
#         gdf (gpd.GeoDataFrame): The GeoDataFrame to be enriched.

#     Returns:
#         gpd.GeoDataFrame: The enriched GeoDataFrame.
#     """
#     # for consistency, move the geometry column to the last position
#     geometry = gdf.pop("geometry")
#     gdf["geometry"] = geometry

#     # beware area and centroid calculations since: "UserWarning: Geometry is in a geographic CRS. Results from
#     # 'centroid' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before
#     # this operation."
#     # This also prevents use of geohashs since they encode the centroid.
#     # gdf['geohash'] = geohash(gdf)

#     bounds = gdf.bounds
#     bounds.columns = [f"geometry_bbox_{col}" for col in bounds.columns]
#     return gdf.join(bounds)
