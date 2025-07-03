import osmnx as ox
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point
import pickle

metro_stations = gpd.read_file("location_data/metro_stations.geojson")
metro_df = metro_stations.reset_index()
metro_df['lat'] = metro_df.geometry.centroid.y
metro_df['lon'] = metro_df.geometry.centroid.x
metro_df[['name', 'lat', 'lon']].dropna(subset=['name'])

key_locations = {
    'airport': {'lat': 13.1979, 'lon': 77.7063, 'name': 'Kempegowda International Airport'},
    'railway_stations': [
        {'lat': 12.9767, 'lon': 77.5703, 'name': 'KSR Bengaluru City Junction'},
        {'lat': 13.0287, 'lon': 77.5340, 'name': 'Yeshwantpur Junction'}
    ]
}

major_roads = gpd.read_file("location_data/major_roads.shp")
major_roads = major_roads[major_roads["name"].notna()]
major_roads

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def distance_to_linestring(point_lat, point_lon, linestring_geom):
    """
    Calculate minimum distance from a point to a LINESTRING geometry
    """
    
    point = Point(point_lon, point_lat)  # Note: shapely uses (lon, lat)
    
    # Calculate distance in degrees, then convert to km
    distance_degrees = point.distance(linestring_geom)
    
    # Rough conversion: 1 degree ≈ 111 km (at equator)
    # For Bangalore latitude (~13°N), 1 degree lon ≈ 108 km
    distance_km = distance_degrees * 111
    
    return distance_km

def get_nearest_metro(lat, lon):
    metro_distances = []

    for _, station in metro_df.iterrows():
        dist = calculate_distance(lat, lon, station['lat'], station['lon'])
        name = station.get('name', 'Unknown')
        metro_distances.append((dist, name))

    metro_distances.sort()
    metro_dist = metro_distances[0][0]
    nearest_metro = metro_distances[0][1]

    return metro_dist, nearest_metro 

def get_nearest_main_road(lat, lon):
    major_road_distances = []
    for _, road in major_roads.iterrows():
        dist = distance_to_linestring(lat, lon, road.geometry)
        name = road.get('name', 'Unknown')
        major_road_distances.append((dist, name))
    major_road_distances.sort(key=lambda x: x[0])    
    dist = major_road_distances[0][0]
    name = major_road_distances[0][1]
    return dist, name

def get_taluk(lat, lon):
    model_dir = "models"
    name = "taluk"

    with open(f'{model_dir}/{name}_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open(f'{model_dir}/{name}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    input_features = scaler.transform([[lat, lon]])
    taluk = model.predict(input_features)[0]
    return taluk

def get_all_location_details(lat, lon):
    loc_dict = {}
    loc_dict["airport_distance_kms"] = calculate_distance(lat, lon,
                                                            key_locations['airport']['lat'], key_locations['airport']['lon'])
    loc_dict["ksr_jn_distance_kms"] = calculate_distance(lat, lon,
                                                            key_locations['railway_stations'][0]['lat'], key_locations['railway_stations'][0]['lon'])
    loc_dict["yeshwantpur_jn_distance_kms"] = calculate_distance(lat, lon,
                                                                    key_locations['railway_stations'][1]['lat'], key_locations['railway_stations'][1]['lon'])
    loc_dict["nearest_metro"] = get_nearest_metro(lat, lon)
    loc_dict["nearest_major_road"] = get_nearest_main_road(lat, lon)
    loc_dict["taluk"] = get_taluk(lat, lon)
    return loc_dict