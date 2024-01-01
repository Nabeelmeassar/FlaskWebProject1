import heapq
import json
import math
import folium
from math import radians, cos, sin, asin, sqrt

class RoutePlanner:
    def __init__(self, cities, coordinates_with_names):
        self.cities = cities
        self.coordinates_with_names = coordinates_with_names

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points on the earth (specified in decimal degrees).
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r
    
    def calculate_score(self, city, current_city):
        rating = self.cities[city]['rating']
        distance = self.haversine(
            self.cities[city]['GPS'][1], self.cities[city]['GPS'][0],
            self.cities[current_city]['GPS'][1], self.cities[current_city]['GPS'][0]
        )

        # Exponentielles Gewicht für das Rating und logarithmische Skalierung für die Entfernung.
        rating_exponent = 2
        weighted_rating = rating ** rating_exponent
    
        log_distance = math.log(distance + 1)

        # Berechnen des kombinierten gewichteten Scores.
        score = weighted_rating / log_distance
        return score

    def best_first_search(self, start):
        visited = set()
        priority_queue = [(-self.calculate_score(start, start), start)]

        route = []
        city_score = {}
        while priority_queue:
            _, current_city = heapq.heappop(priority_queue)
            if current_city not in visited:
                visited.add(current_city)
                route.append(current_city)
                for city in self.cities:
                    if city not in visited:
                        score = -self.calculate_score(city, current_city)
                        if city not in city_score or city_score[city]['score'] > score:
                            city_score[city] = {'score': score}
                            heapq.heappush(priority_queue, (score, city))

        return route, city_score

    def create_rout_map(self, route, route_color='blue', route_weight=5):
        # Pfad zur GeoJSON-Datei von Deutschland
        germany_geojson_path = 'C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\scripts\\2_hoch.geo.json'
    
        # Laden der GeoJSON-Datei
        with open(germany_geojson_path, 'r', encoding='utf-8') as f:
            germany_geojson = json.load(f)
        # Erstellen einer folium Karte, zentriert auf Deutschland
        m = folium.Map(location=[51.1657, 10.4515], zoom_start=6)

        # Hinzufügen der GeoJSON-Grenzen von Deutschland zur Karte
        folium.GeoJson(
            germany_geojson,
            name='Germany GeoJSON'
        ).add_to(m)
        # Hinzufügen von Markern für jede Stadt in der Route
        for index, city in enumerate(route):
            index = index + 1
            if city in self.coordinates_with_names:
                coords = self.coordinates_with_names[city]
                folium.Marker(
                    location=coords,
                    tooltip=f'{index} - {city}',
                    icon=folium.Icon(icon=f'futbol', prefix='fa', color='red')
                ).add_to(m)

        # Erstellen der Koordinaten für die Polyline, die die Route darstellt
        route_coords = [self.coordinates_with_names[city] for city in route if city in self.coordinates_with_names]
    
        # Hinzufügen der Route als Polyline zur Karte
        folium.PolyLine(route_coords, color=route_color, weight=route_weight, opacity=0.8).add_to(m)

        # Anpassen der Kartenansicht, um alle Marker einzuschließen
        bounds = [[47.2701114, 5.8663425], [55.0815, 15.0418962]]
        m.fit_bounds(bounds)

        # Rückgabe der erstellten Karte
        return m