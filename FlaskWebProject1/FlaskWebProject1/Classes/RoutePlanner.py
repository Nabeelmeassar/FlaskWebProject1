from geopy.distance import geodesic

class RoutePlanner:
    def __init__(self, cities, budget, freie_tage, routes):
        self.cities = cities  # Dies sollte ein Dictionary sein, das City-Objekte enthält
        self.budget = budget
        self.freie_tage = freie_tage
        self.routes = routes
    
    def calculate_driving_cost(self, city1, city2):
        fuel_factor = 0.1
        distance = geodesic(city1.gps, city2.gps).km
        driving_cost = distance * fuel_factor
        return driving_cost

    def calculate_total_price(self):
        total_price = 0
        new_route = []
        route = []
        current_hotel_cost = 0.0
        current_ticket_cost = 0.0
        current_total_cost = 0.0
        tage = -1

        for city_name in self.routes:
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
        return total_price, new_route, tage