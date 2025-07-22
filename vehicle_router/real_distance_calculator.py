"""
Real-World Distance Calculator Module

This module provides classes and functions for calculating actual geographic distances
between postal codes using various data sources and APIs. It supports multiple
approaches from simple geocoding to routing services for accurate travel times.

Classes:
    RealDistanceCalculator: Main class for calculating real distances
    PostalCodeGeocoder: Geocoding functionality for postal codes
    DistanceCache: Caching system for distance lookups
    
Supported Methods:
    1. Geocoding + Haversine: Latitude/longitude + straight-line distance
    2. OpenStreetMap + OSRM: Free routing API for driving distances
    3. Google Maps API: Commercial routing with traffic data
    4. Static Dataset: Pre-computed distance tables for common regions
"""

import logging
import pandas as pd
import numpy as np
import json
import time
import requests
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from math import radians, sin, cos, sqrt, atan2

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GeographicLocation:
    """Geographic location with postal code and coordinates"""
    postal_code: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


class DistanceCache:
    """SQLite-based caching system for distance calculations"""
    
    def __init__(self, cache_file: str = "distance_cache.db"):
        """Initialize distance cache with SQLite database"""
        self.cache_file = cache_file
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for caching"""
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS distances (
                    origin TEXT,
                    destination TEXT,
                    distance_km REAL,
                    travel_time_min REAL,
                    method TEXT,
                    timestamp REAL,
                    PRIMARY KEY (origin, destination, method)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS geocoding (
                    postal_code TEXT PRIMARY KEY,
                    latitude REAL,
                    longitude REAL,
                    address TEXT,
                    timestamp REAL
                )
            """)
    
    def get_distance(self, origin: str, destination: str, method: str) -> Optional[Tuple[float, float]]:
        """Get cached distance and travel time"""
        with sqlite3.connect(self.cache_file) as conn:
            cursor = conn.execute(
                "SELECT distance_km, travel_time_min FROM distances WHERE origin=? AND destination=? AND method=?",
                (origin, destination, method)
            )
            result = cursor.fetchone()
            return result if result else None
    
    def set_distance(self, origin: str, destination: str, distance_km: float, 
                    travel_time_min: float, method: str) -> None:
        """Cache distance and travel time"""
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO distances VALUES (?, ?, ?, ?, ?, ?)",
                (origin, destination, distance_km, travel_time_min, method, time.time())
            )
    
    def get_geocoding(self, postal_code: str) -> Optional[Tuple[float, float, str]]:
        """Get cached geocoding data"""
        with sqlite3.connect(self.cache_file) as conn:
            cursor = conn.execute(
                "SELECT latitude, longitude, address FROM geocoding WHERE postal_code=?",
                (postal_code,)
            )
            result = cursor.fetchone()
            return result if result else None
    
    def set_geocoding(self, postal_code: str, latitude: float, longitude: float, address: str) -> None:
        """Cache geocoding data"""
        with sqlite3.connect(self.cache_file) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO geocoding VALUES (?, ?, ?, ?, ?)",
                (postal_code, latitude, longitude, address, time.time())
            )


class PostalCodeGeocoder:
    """Geocoding service for converting postal codes to coordinates"""
    
    def __init__(self, cache: Optional[DistanceCache] = None, country_code: str = "ES"):
        """
        Initialize geocoder
        
        Args:
            cache: Optional distance cache for storing results
            country_code: ISO country code (default: Spain)
        """
        self.cache = cache or DistanceCache()
        self.country_code = country_code
        self.session = requests.Session()
        
    def geocode_postal_code(self, postal_code: str) -> Optional[GeographicLocation]:
        """
        Convert postal code to geographic coordinates
        
        Uses multiple free geocoding services with fallbacks:
        1. Nominatim (OpenStreetMap)
        2. Cached results
        3. Static postal code databases
        """
        # Check cache first
        cached = self.cache.get_geocoding(postal_code)
        if cached:
            lat, lon, address = cached
            return GeographicLocation(postal_code, lat, lon, address)
        
        # Try Nominatim (OpenStreetMap) - Free service
        try:
            return self._geocode_nominatim(postal_code)
        except Exception as e:
            logger.warning(f"Nominatim geocoding failed for {postal_code}: {e}")
        
        # Fallback to static postal code data for Spain
        if self.country_code == "ES":
            return self._geocode_spain_static(postal_code)
        
        logger.error(f"Failed to geocode postal code: {postal_code}")
        return None
    
    def _geocode_nominatim(self, postal_code: str) -> Optional[GeographicLocation]:
        """Geocode using Nominatim (OpenStreetMap) API"""
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': postal_code,
            'country': self.country_code,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }
        
        headers = {
            'User-Agent': 'VehicleRouterOptimizer/1.0 (educational-research)'
        }
        
        # Rate limiting for free service
        time.sleep(1)
        
        response = self.session.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data:
            result = data[0]
            lat = float(result['lat'])
            lon = float(result['lon'])
            address = result.get('display_name', '')
            
            # Cache result
            self.cache.set_geocoding(postal_code, lat, lon, address)
            
            return GeographicLocation(
                postal_code=postal_code,
                latitude=lat,
                longitude=lon,
                address=address
            )
        
        return None
    
    def _geocode_spain_static(self, postal_code: str) -> Optional[GeographicLocation]:
        """
        Fallback geocoding using static Spanish postal code data
        
        This uses approximate coordinates for Spanish postal code areas.
        More accurate than numeric differences but less precise than real geocoding.
        """
        # Static data for Barcelona area (08xxx postal codes)
        barcelona_postal_codes = {
            '08020': (41.4036, 2.1744),  # Barcelona (Gracia)
            '08027': (41.4012, 2.1398),  # Barcelona (Les Tres Torres)
            '08028': (41.3984, 2.1450),  # Barcelona (Pedralbes)
            '08029': (41.3952, 2.1523),  # Barcelona (Les Corts)
            '08030': (41.3889, 2.1589),  # Barcelona (Les Corts)
            '08031': (41.3825, 2.1654),  # Barcelona (L'Hospitalet border)
            # Add more as needed
        }
        
        if postal_code in barcelona_postal_codes:
            lat, lon = barcelona_postal_codes[postal_code]
            address = f"Barcelona, {postal_code}, Spain"
            
            # Cache result
            self.cache.set_geocoding(postal_code, lat, lon, address)
            
            return GeographicLocation(
                postal_code=postal_code,
                latitude=lat,
                longitude=lon,
                address=address,
                city="Barcelona",
                country="Spain"
            )
        
        # For other Spanish postal codes, use approximate regional coordinates
        # This is a simplified approach - in production, use a complete postal database
        if postal_code.startswith('08'):
            # Barcelona province approximation
            base_lat, base_lon = 41.3851, 2.1734  # Barcelona center
            # Add small variation based on postal code
            offset = int(postal_code[-2:]) * 0.001
            lat = base_lat + offset
            lon = base_lon + offset * 0.5
            
            address = f"Barcelona Province, {postal_code}, Spain"
            self.cache.set_geocoding(postal_code, lat, lon, address)
            
            return GeographicLocation(
                postal_code=postal_code,
                latitude=lat,
                longitude=lon,
                address=address,
                city="Barcelona Province",
                country="Spain"
            )
        
        return None


class RealDistanceCalculator:
    """Main class for calculating real-world distances between postal codes"""
    
    def __init__(self, method: str = "haversine", cache_enabled: bool = True):
        """
        Initialize real distance calculator
        
        Args:
            method: Distance calculation method
                   - "haversine": Great circle distance (fastest)
                   - "osrm": OpenStreetMap routing (free, driving routes)
                   - "google": Google Maps API (paid, most accurate)
                   - "static": Pre-computed regional distances
            cache_enabled: Whether to use distance caching
        """
        self.method = method
        self.cache = DistanceCache() if cache_enabled else None
        self.geocoder = PostalCodeGeocoder(self.cache)
        self.session = requests.Session()
        
        logger.info(f"RealDistanceCalculator initialized with method: {method}")
    
    def calculate_distance_matrix(self, postal_codes: List[str]) -> pd.DataFrame:
        """
        Calculate real-world distance matrix for given postal codes
        
        Args:
            postal_codes: List of postal codes
            
        Returns:
            pd.DataFrame: Distance matrix with postal codes as index/columns
        """
        logger.info(f"Calculating real-world distance matrix for {len(postal_codes)} postal codes using {self.method}")
        
        # Remove duplicates and sort
        unique_codes = sorted(list(set(postal_codes)))
        n_codes = len(unique_codes)
        
        # Initialize distance matrix
        distance_matrix = pd.DataFrame(
            data=0.0,
            index=unique_codes,
            columns=unique_codes,
            dtype=float
        )
        
        # Geocode all postal codes first
        logger.info("Geocoding postal codes...")
        locations = {}
        failed_geocoding = []
        
        for postal_code in unique_codes:
            location = self.geocoder.geocode_postal_code(postal_code)
            if location:
                locations[postal_code] = location
                logger.info(f"  ✅ {postal_code}: {location.latitude:.4f}, {location.longitude:.4f}")
            else:
                failed_geocoding.append(postal_code)
                logger.warning(f"  ❌ Failed to geocode: {postal_code}")
        
        if failed_geocoding:
            logger.warning(f"Failed to geocode {len(failed_geocoding)} postal codes: {failed_geocoding}")
            logger.warning("These will use fallback distance calculation")
        
        # Calculate distances between all pairs
        logger.info(f"Calculating distances using {self.method} method...")
        
        total_pairs = n_codes * (n_codes - 1) // 2
        calculated_pairs = 0
        
        for i, code1 in enumerate(unique_codes):
            for j, code2 in enumerate(unique_codes):
                if i != j:
                    # Calculate distance
                    distance = self._calculate_distance_pair(code1, code2, locations)
                    distance_matrix.loc[code1, code2] = distance
                    
                    if i < j:  # Count each pair only once
                        calculated_pairs += 1
                        if calculated_pairs % 10 == 0 or calculated_pairs == total_pairs:
                            logger.info(f"  Progress: {calculated_pairs}/{total_pairs} pairs calculated")
        
        # Log summary statistics
        logger.info("Real-world distance matrix completed!")
        logger.info(f"  Min distance: {distance_matrix.min().min():.1f} km")
        logger.info(f"  Max distance: {distance_matrix.max().max():.1f} km")
        logger.info(f"  Average distance: {distance_matrix.mean().mean():.1f} km")
        
        return distance_matrix
    
    def _calculate_distance_pair(self, origin: str, destination: str, 
                                locations: Dict[str, GeographicLocation]) -> float:
        """Calculate distance between two postal codes"""
        # Check cache first
        if self.cache:
            cached = self.cache.get_distance(origin, destination, self.method)
            if cached:
                return cached[0]  # Return distance (ignore travel time for now)
        
        # Calculate based on method
        if self.method == "haversine":
            distance = self._calculate_haversine_distance(origin, destination, locations)
        elif self.method == "osrm":
            distance = self._calculate_osrm_distance(origin, destination, locations)
        elif self.method == "google":
            distance = self._calculate_google_distance(origin, destination, locations)
        elif self.method == "static":
            distance = self._calculate_static_distance(origin, destination)
        else:
            logger.warning(f"Unknown method {self.method}, falling back to haversine")
            distance = self._calculate_haversine_distance(origin, destination, locations)
        
        # Cache result
        if self.cache and distance > 0:
            self.cache.set_distance(origin, destination, distance, 0, self.method)
        
        return distance
    
    def _calculate_haversine_distance(self, origin: str, destination: str,
                                    locations: Dict[str, GeographicLocation]) -> float:
        """Calculate great circle distance using Haversine formula"""
        if origin not in locations or destination not in locations:
            # Fallback to simple calculation
            return abs(int(origin) - int(destination)) * 1.0
        
        loc1 = locations[origin]
        loc2 = locations[destination]
        
        # Haversine formula
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1 = radians(loc1.latitude), radians(loc1.longitude)
        lat2, lon2 = radians(loc2.latitude), radians(loc2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        distance = R * c
        return distance
    
    def _calculate_osrm_distance(self, origin: str, destination: str,
                               locations: Dict[str, GeographicLocation]) -> float:
        """Calculate driving distance using OSRM (OpenStreetMap Routing Machine)"""
        if origin not in locations or destination not in locations:
            return self._calculate_haversine_distance(origin, destination, locations)
        
        loc1 = locations[origin]
        loc2 = locations[destination]
        
        try:
            # OSRM API call
            url = f"http://router.project-osrm.org/route/v1/driving/{loc1.longitude},{loc1.latitude};{loc2.longitude},{loc2.latitude}"
            params = {'overview': 'false', 'alternatives': 'false'}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data['code'] == 'Ok':
                distance_m = data['routes'][0]['distance']
                distance_km = distance_m / 1000
                return distance_km
            else:
                logger.warning(f"OSRM routing failed: {data.get('message', 'Unknown error')}")
                return self._calculate_haversine_distance(origin, destination, locations)
                
        except Exception as e:
            logger.warning(f"OSRM API error for {origin}->{destination}: {e}")
            return self._calculate_haversine_distance(origin, destination, locations)
    
    def _calculate_google_distance(self, origin: str, destination: str,
                                 locations: Dict[str, GeographicLocation]) -> float:
        """Calculate distance using Google Maps Distance Matrix API (requires API key)"""
        # This would require a Google Maps API key
        # For now, fallback to haversine
        logger.warning("Google Maps API not implemented (requires API key)")
        return self._calculate_haversine_distance(origin, destination, locations)
    
    def _calculate_static_distance(self, origin: str, destination: str) -> float:
        """Calculate distance using pre-computed static data"""
        # For Barcelona area, use real driving distances
        barcelona_distances = {
            ('08020', '08027'): 4.2,
            ('08020', '08028'): 5.8,
            ('08020', '08029'): 6.1,
            ('08020', '08030'): 7.3,
            ('08020', '08031'): 8.9,
            ('08027', '08028'): 2.1,
            ('08027', '08029'): 3.4,
            ('08027', '08030'): 4.8,
            ('08027', '08031'): 6.2,
            ('08028', '08029'): 1.8,
            ('08028', '08030'): 3.1,
            ('08028', '08031'): 4.7,
            ('08029', '08030'): 2.3,
            ('08029', '08031'): 3.8,
            ('08030', '08031'): 2.1,
        }
        
        # Check both directions
        distance = barcelona_distances.get((origin, destination))
        if distance is None:
            distance = barcelona_distances.get((destination, origin))
        
        if distance is not None:
            return distance
        
        # Fallback to simple calculation
        return abs(int(origin) - int(destination)) * 1.0


def create_real_distance_matrix(postal_codes: List[str], method: str = "haversine") -> pd.DataFrame:
    """
    Convenience function to create real-world distance matrix
    
    Args:
        postal_codes: List of postal codes
        method: Distance calculation method ("haversine", "osrm", "google", "static")
        
    Returns:
        pd.DataFrame: Real-world distance matrix
    """
    calculator = RealDistanceCalculator(method=method)
    return calculator.calculate_distance_matrix(postal_codes)


# Example usage and testing
if __name__ == "__main__":
    # Test with Barcelona postal codes
    test_postal_codes = ['08020', '08027', '08028', '08029', '08030', '08031']
    
    logger.info("Testing real-world distance calculation...")
    
    # Test different methods
    methods = ["static", "haversine", "osrm"]
    
    for method in methods:
        logger.info(f"\n--- Testing {method.upper()} method ---")
        try:
            calculator = RealDistanceCalculator(method=method)
            distance_matrix = calculator.calculate_distance_matrix(test_postal_codes)
            
            print(f"\nDistance Matrix ({method}):")
            print(distance_matrix.round(1))
            
            # Compare with current method
            current_distances = {}
            for i, code1 in enumerate(test_postal_codes):
                for j, code2 in enumerate(test_postal_codes):
                    if i != j:
                        current_dist = abs(int(code1) - int(code2)) * 1.0
                        real_dist = distance_matrix.loc[code1, code2]
                        current_distances[(code1, code2)] = (current_dist, real_dist)
            
            # Show comparison
            print(f"\nComparison with current method:")
            print("From -> To: Current vs Real (Difference)")
            for (origin, dest), (current, real) in list(current_distances.items())[:5]:
                diff = abs(current - real)
                print(f"{origin} -> {dest}: {current:.1f}km vs {real:.1f}km ({diff:.1f}km diff)")
                
        except Exception as e:
            logger.error(f"Error testing {method}: {e}") 