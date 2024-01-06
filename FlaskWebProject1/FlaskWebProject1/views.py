"""
Routes and views for the flask application.
"""
from email import message
from geopy.distance import geodesic
from FlaskWebProject1 import app
from flask import request, jsonify
from flask import render_template, request
from datetime import datetime
from FlaskWebProject1.Classes.TravelAssistant import TravelAssistant
from FlaskWebProject1.Classes.city import get_cities
from FlaskWebProject1.Classes.TravelPlanner import TravelPlanner


cities = get_cities()
csv_data_path = 'C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\data.csv'

@app.route('/post_preference_json', methods=['POST'])

def post_preference_json_handler():

    content = request.get_json()
    select_start = content.get('Select_start')
    tage = int(content.get('Tage'))
    gewicht = int(content.get('Gewicht'))
    budget = int(content.get('Person_Budget'))
    preis_hoch = int(content.get('Preis_hoch'))
    travelAssistant = TravelAssistant(csv_data_path, content, cities )
    city_with_rating, mse, average_rating = travelAssistant.predict_user_preferences()
    if preis_hoch == 1:
        content.update({'budget': budget - 200})
        budget -= 200  # Verringere das Budget
        travelAssistant = TravelAssistant(csv_data_path, content, cities )
        city_with_rating, mse, average_rating = travelAssistant.predict_user_preferences()
    travelPlanner = TravelPlanner(cities, gewicht, budget )
    routes = travelPlanner.best_first_search(select_start)
    route = {city.id: city for city in cities.values() if city.name in routes}
    total_price, new_route = calculate_total_price(routes, budget)
    # Karte erstellen
    football_map = travelPlanner.create_rout_map(new_route)
    cities_serializable = {city_name: city_obj.to_dict() for city_name, city_obj in cities.items()}

    # Konvertieren Sie die Karte in eine HTML-Zeichenkette
    map_html = football_map._repr_html_()    
    content.update({
            'city_with_rating': cities_serializable,
            'm_html': map_html,
            'route': new_route,
            'mse': mse,
            'total_price': round(total_price, 2),
            'average_rating':average_rating
    })
    # Return the modified content as JSON
    return jsonify(content)

@app.route('/', methods = ['GET','POST'])
def home():
   return render_template(
        'index.html',
        title = 'Home Page',
        cities = cities,
        year=datetime.now().strftime('%Y.%d.%h'),        
        )


@app.route('/contact')
def contact():
        return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message= message
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

def calculate_driving_cost(city1, city2):
    fuel_factor = 0.1
    distance = geodesic(city1.gps, city2.gps).km
    driving_cost = distance * fuel_factor
    return driving_cost

def calculate_total_price(route_, budget):
    # Define a variable for the total price
    total_price = 0
    new_route = []
    route = []
    current_hotel_cost = 0.0
    current_ticket_cost = 0.0
    current_total_cost = 0.0
    for city_name in route_:
        if city_name in cities:  # Prüfen Sie, ob der Stadtname ein Schlüssel im cities Dictionary ist
            cities[city_name].update_driving_cost(0.0)
            cities[city_name].update_distance_km(0.0)
            route.append(cities[city_name])  # Fügen Sie das City-Objekt zur Liste r hinzu
    for city in route:
        current_hotel_cost = float(city.hotel_cost)
        current_ticket_cost = float(city.ticket_cost)
        if city != route[0]:
            current_total_cost = current_hotel_cost + current_ticket_cost

        # Prüfen, ob das aktuelle Gesamtkostenbudget überschritten wird
        if (current_total_cost + total_price) <= budget:
            total_price += current_total_cost
            new_route.append(city.name)
            
            # Wenn dies nicht die letzte Stadt ist, berechnen Sie die Fahrtkosten zur nächsten Stadt
            if route.index(city) < len(route) - 1:
                next_city = route[route.index(city) + 1]
                current_driving_cost = calculate_driving_cost(city, next_city) 
                distance = geodesic(city.gps, next_city.gps).km
                # Prüfen, ob das Hinzufügen der Fahrtkosten das Budget überschreitet
                if (current_driving_cost + total_price) <= budget:
                    total_price += current_driving_cost
                    next_city.update_driving_cost(current_driving_cost)
                    next_city.update_distance_km(distance)
                else:
                    # Wenn das Budget überschritten wird, beenden Sie die Schleife
                    city.update_driving_cost(0.0)
                    city.update_distance_km(0.0)
                    break
        else:
            # Wenn das Budget überschritten wird, beenden Sie die Schleife
            city.update_driving_cost(0.0)
            city.update_distance_km(0.0)

            break

    return total_price, new_route
