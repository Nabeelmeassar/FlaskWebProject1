import heapq
import json
import math
import folium
from geopy.distance import geodesic
class TravelPlanner:
    def __init__(self, cities, gewicht,freie_tage,budget):
        self.cities = cities
        self.gewicht = gewicht  # Dieser Wert bestimmt, wie stark das Rating gewichtet wird.
        self.freie_tage = freie_tage
        self.budget = budget
        
    def calculate_cost_heuristic(self, city_name):
            # Berechnet eine Heuristik für die Kosten einer Stadt, die bei der Entscheidungsfindung hilft.
            cost = self.cities[city_name].get_cost()
            # Je niedriger die Kosten, desto höher die Heuristik. Es wird eine Normalisierung auf das Budget durchgeführt.
            cost_factor = max(0, (self.budget - cost) / self.budget)
            return cost_factor
    
    def calculate_cost_score(self, city, current_city):
        # Berechnet einen Score für jede Stadt basierend auf ihrer Bewertung, Entfernung zur aktuellen Stadt und den Kosten.
        rating = self.cities[city].rating
        distance = geodesic(self.cities[city].gps, self.cities[current_city].gps).km
        cost = self.cities[city].get_cost()
        # Anwendung exponentiellen Gewichts auf die Bewertung und logarithmische Skalierung auf die Entfernung.
        weighted_rating = rating ** self.gewicht
        log_distance = math.log(distance + 1)  # +1 verhindert Division durch Null.
        # Einbeziehung der Kosten in den Score.
        cost_factor = self.calculate_cost_heuristic(city)
        # Der Score wird berechnet als gewichtete Bewertung dividiert durch die logarithmische Entfernung, multipliziert mit dem Kostenfaktor.
        return (weighted_rating / log_distance) * cost_factor

    def best_first_search(self, start):
        # Implementiert die Best-First-Suche, um die Route zu finden, die den besten Score bietet.
        visited = set()
        priority_queue = [(-self.calculate_cost_score(start, start), start)]
        route = []
        city_score = {}
        while priority_queue:
            # Nimmt die Stadt mit dem höchsten Score aus der Warteschlange.
            _, current_city = heapq.heappop(priority_queue)
            if current_city not in visited:
                # Fügt die Stadt zur Route hinzu und markiert sie als besucht.
                visited.add(current_city)
                route.append(current_city)
                for city in self.cities:
                    # Fügt Nachbarstädte zur Warteschlange hinzu, wenn sie nicht besucht wurden.
                    if city not in visited:
                        score = self.calculate_cost_score(city, current_city)
                        if city not in city_score or city_score[city] < score:
                            city_score[city] = score
                            heapq.heappush(priority_queue, (-score, city))
        return route

    def create_rout_map(self, route, germany_geojson_path, route_color='blue', route_weight=5):
        # Laden der GeoJSON-Datei
        with open(germany_geojson_path, 'r', encoding='utf-8') as f:
            germany_geojson = json.load(f)
        # Erstellen einer folium Karte, zentriert auf Deutschland
        football_map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)

        # Hinzuf�gen der GeoJSON-Grenzen von Deutschland zur Karte
        folium.GeoJson(
            germany_geojson,
            name='Germany GeoJSON'
        ).add_to(football_map)
        for city_name, city_obj in self.cities.items():
            coords = city_obj.gps
            rating = city_obj.rating
            folium.Marker(
                location=coords,
                tooltip=f'{city_name}, Bewertung = {round(rating, 2)}',
                icon=folium.Icon(icon='futbol', prefix='fa', color='red')
            ).add_to(football_map)
            
        # Hinzuf�gen von Markern f�r jede Stadt in der Route
        for index, city in enumerate(route):
            if city in self.cities:
                coords = self.cities[city].gps
                rating = self.cities[city].rating
                folium.Marker(
                    location=coords,
                    tooltip=f'{index} - {city}, Bewertung = {round(rating,2)}',
                    icon=folium.Icon(icon=f'futbol', prefix='fa', color='green')
                ).add_to(football_map)
            index = index + 1


        # Correct the creation of route coordinates
        route_coords = [tuple(map(float, self.cities[city].gps)) for city in route if city in self.cities]

        # Ensure there is more than one city to form a polyline
        if len(route_coords) > 1:
            # Add the polyline to the map
            folium.PolyLine(route_coords, color=route_color, weight=route_weight).add_to(football_map)
        else:
            print("Not enough coordinates to form a route")

        # Anpassen der Kartenansicht, um alle Marker einzuschlie�en
        bounds = [[47.2701114, 5.8663425], [55.0815, 15.0418962]]
        football_map.fit_bounds(bounds)

        # R�ckgabe der erstellten Karte
        return football_map
    def create_google_maps_route_link(self, cities):
        base_url = "https://www.google.com/maps/dir/"
        route_parts = [city.replace(' ', '+') for city in cities]  # Ersetzt Leerzeichen durch '+' für URL-Kodierung
        route_url = base_url + '/'.join(route_parts)
        return route_url

        # Beispiel: Erstellen eines Links für eine Route, die Berlin, München und Köln umfasst.
        cities_route = ["Berlin", "München", "Köln"]
        route_link = create_google_maps_route_link(cities_route)
        return route_link
    def calculate_driving_cost(self, city1, city2):
        fuel_factor = 0.1
        distance = geodesic(city1.gps, city2.gps).km
        driving_cost = distance * fuel_factor
        return driving_cost

    def calculate_total_price(self, routes):
        total_price = 0
        new_route = []
        route = []
        current_hotel_cost = 0.0
        current_ticket_cost = 0.0
        current_total_cost = 0.0
        tage = -1

        for city_name in routes:
            if city_name in self.cities:
                self.cities[city_name].update_driving_cost(0.0)
                self.cities[city_name].update_distance_km(0.0)
                route.append(self.cities[city_name])
    
        for city in route:
            current_hotel_cost = float(city.hotel_cost)
            current_ticket_cost = float(city.ticket_cost)
            if city != route[0]:
                current_total_cost = current_hotel_cost + current_ticket_cost

            if (current_total_cost + total_price) <= self.budget and tage < self.freie_tage:
                total_price += current_total_cost
                new_route.append(city.name)
                if route.index(city) < len(route) - 1:
                    next_city = route[route.index(city) + 1]
                    current_driving_cost = self.calculate_driving_cost(city, next_city)
                    distance = geodesic(city.gps, next_city.gps).km
                    if (current_driving_cost + total_price) <= self.budget and tage < self.freie_tage:
                        total_price += current_driving_cost
                        tage += 1
                        next_city.update_driving_cost(current_driving_cost)
                        next_city.update_distance_km(distance)
                    else:
                        break
            else:
                next_city.update_driving_cost(0.0)
                break
            cities_new_route = {}
            for city_name in new_route:
                if city_name in self.cities:
                    cities_new_route[city_name] = self.cities[city_name].rating
            # Gesamtbewertung berechnen
            total_rating = sum(cities_new_route.values())
            average_rating = total_rating / len(cities_new_route) if cities_new_route else None
        return total_price, new_route, tage, average_rating