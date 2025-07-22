"""
Distance Calculator Module

This module provides distance calculation between postal codes using:
1. OpenStreetMap geocoding + Haversine distance (primary method)
2. Postal code difference (fallback method)

Simple, reliable, and efficient approach for real-world distance calculation.
"""

import logging
import pandas as pd
import requests
import time
from typing import Dict, List, Optional
from math import radians, sin, cos, sqrt, atan2

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DistanceCalculator:
    """Distance calculator using OpenStreetMap geocoding and static fallback"""
    
    def __init__(self, country_code: str = "ES"):
        """
        Initialize distance calculator
        
        Args:
            country_code: ISO country code for geocoding (default: Spain)
        """
        self.country_code = country_code
        self.session = requests.Session()
        self.geocoding_cache = {}  # In-memory cache for coordinates
        
        # Set respectful user agent for OpenStreetMap Nominatim
        self.session.headers.update({
            'User-Agent': 'VehicleRouterOptimizer/1.0 (educational-research)'
        })
        
        logger.info(f"DistanceCalculator initialized for country: {country_code}")
    
    def calculate_distance_matrix(self, postal_codes: List[str]) -> pd.DataFrame:
        """
        Calculate distance matrix for given postal codes
        
        Args:
            postal_codes: List of postal codes
            
        Returns:
            pd.DataFrame: Distance matrix with real-world distances in km
        """
        logger.info(f"Calculating distances for {len(postal_codes)} postal codes using OpenStreetMap")
        
        # Remove duplicates and sort
        unique_codes = sorted(list(set(postal_codes)))
        
        # Geocode all postal codes
        logger.info("Geocoding postal codes...")
        coordinates = {}
        failed_geocoding = []
        
        for i, postal_code in enumerate(unique_codes, 1):
            logger.info(f"  📍 Geocoding {postal_code} ({i}/{len(unique_codes)})...")
            coords = self._geocode_postal_code(postal_code)
            if coords:
                coordinates[postal_code] = coords
                logger.info(f"     ✅ {postal_code}: {coords[0]:.4f}, {coords[1]:.4f}")
            else:
                failed_geocoding.append(postal_code)
                logger.warning(f"     ❌ Failed to geocode: {postal_code}")
        
        if failed_geocoding:
            logger.warning(f"Using fallback distances for {len(failed_geocoding)} postal codes")
        
        # Initialize distance matrix
        distance_matrix = pd.DataFrame(
            data=0.0,
            index=unique_codes,
            columns=unique_codes,
            dtype=float
        )
        
        # Calculate distances between all pairs
        logger.info("Calculating distances...")
        for i, code1 in enumerate(unique_codes):
            for j, code2 in enumerate(unique_codes):
                if i != j:
                    distance = self._calculate_distance(code1, code2, coordinates)
                    distance_matrix.loc[code1, code2] = distance
        
        # Log summary
        logger.info("Distance matrix completed!")
        logger.info(f"  Min distance: {distance_matrix.min().min():.1f} km")
        logger.info(f"  Max distance: {distance_matrix.max().max():.1f} km") 
        logger.info(f"  Average distance: {distance_matrix.mean().mean():.1f} km")
        
        return distance_matrix
    
    def _geocode_postal_code(self, postal_code: str) -> Optional[tuple]:
        """
        Get coordinates for a postal code using OpenStreetMap
        
        Args:
            postal_code: The postal code to geocode
            
        Returns:
            tuple: (latitude, longitude) or None if failed
        """
        # Check cache first
        if postal_code in self.geocoding_cache:
            return self.geocoding_cache[postal_code]
        
        try:
            # OpenStreetMap Nominatim geocoding API
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'postalcode': postal_code,
                'country': self.country_code,
                'format': 'json',
                'limit': 1,
                'addressdetails': 0
            }
            
            # Rate limiting - be respectful to free service (reduced for CLI)
            time.sleep(0.5)
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data:
                result = data[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                coords = (lat, lon)
                
                # Cache the result
                self.geocoding_cache[postal_code] = coords
                return coords
            
        except Exception as e:
            logger.warning(f"Geocoding failed for {postal_code}: {e}")
        
        return None
    
    def _calculate_distance(self, origin: str, destination: str, coordinates: Dict[str, tuple]) -> float:
        """Calculate distance between two postal codes"""
        
        # Use real coordinates if available
        if origin in coordinates and destination in coordinates:
            return self._haversine_distance(coordinates[origin], coordinates[destination])
        
        # Fallback to postal code difference
        try:
            return abs(int(origin) - int(destination)) * 0.8  # 0.8km per unit
        except ValueError:
            return 10.0  # Default for non-numeric postal codes
    
    def _haversine_distance(self, coord1: tuple, coord2: tuple) -> float:
        """
        Calculate great circle distance using Haversine formula
        
        Args:
            coord1: (latitude, longitude) of first point
            coord2: (latitude, longitude) of second point
            
        Returns:
            float: Distance in kilometers
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        # Earth's radius in km
        return 6371 * c


def create_distance_matrix(postal_codes: List[str], country_code: str = "ES") -> pd.DataFrame:
    """
    Convenience function to create real-world distance matrix
    
    Args:
        postal_codes: List of postal codes
        country_code: ISO country code for geocoding
        
    Returns:
        pd.DataFrame: Real-world distance matrix
    """
    calculator = DistanceCalculator(country_code=country_code)
    return calculator.calculate_distance_matrix(postal_codes)


# Test the distance calculator
if __name__ == "__main__":
    test_codes = ['08020', '08027', '08028', '08029', '08030', '08031']
    
    logger.info("Testing DistanceCalculator...")
    calculator = DistanceCalculator(country_code="ES")
    distance_matrix = calculator.calculate_distance_matrix(test_codes)
    
    print("\nReal Distance Matrix:")
    print(distance_matrix.round(1))
    
    print(f"\n✅ Successfully geocoded {len(calculator.geocoding_cache)} postal codes")
    
    # Show example distances
    print("\nExample distances:")
    for i in range(3):
        for j in range(i+1, min(i+3, len(test_codes))):
            code1, code2 = test_codes[i], test_codes[j]
            distance = distance_matrix.loc[code1, code2]
            print(f"  {code1} -> {code2}: {distance:.1f} km") 