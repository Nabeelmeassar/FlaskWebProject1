"""
Routes and views for the flask application.
"""
from email import message
from geopy.distance import geodesic
from FlaskWebProject1 import app
from flask import request, jsonify
from flask import render_template, request
from datetime import datetime
from FlaskWebProject1.Classes.city import get_cities
from FlaskWebProject1.Classes.TravelAssistant import TravelAssistant
from FlaskWebProject1.Classes.TravelPlanner import TravelPlanner
from FlaskWebProject1.Classes.RoutePlanner import RoutePlanner


cities = get_cities()
csv_data_path = 'C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\training_data.csv'

@app.route('/post_preference_json', methods=['POST'])

def post_preference_json_handler():
    # JSON-Inhalt aus der Anfrage extrahieren
    content = request.get_json()
    
    # Verschiedene Parameter aus dem JSON-Körper extrahieren
    select_start = content.get('Select_start')
    freie_tage = int(content.get('Tage'))
    gewicht = int(content.get('Gewicht'))
    budget = int(content.get('Person_Budget'))
    preis_hoch = int(content.get('Preis_hoch'))
    
    # Erstellt eine Instanz von TravelAssistant, um Benutzerpräferenzen zu berechnen
    travelAssistant = TravelAssistant(csv_data_path, content, cities)
    
    # Berechnet die Benutzerpräferenzen basierend auf den angegebenen Daten
    city_with_rating, mse, average_rating = travelAssistant.predict_user_preferences()
    
    # Wenn der Benutzer bereit ist, mehr zu bezahlen, aktualisiere das Budget
    if preis_hoch == 1:
        content.update({'budget': budget - 200})
        budget -= 200  # Verringere das Budget
        travelAssistant = TravelAssistant(csv_data_path, content, cities)
        city_with_rating, mse, average_rating = travelAssistant.predict_user_preferences()
    
    # Erstellt eine Instanz von TravelPlanner, um die beste Route zu finden
    travelPlanner = TravelPlanner(cities, gewicht, budget)
    
    # Verwendet den Algorithmus "Best-First-Search" zur Routenfindung
    routes = travelPlanner.best_first_search(select_start)
    
    # Erstellt ein Wörterbuch der Städte auf der Route
    route = {city.id: city for city in cities.values() if city.name in routes}
    
    # Erstellt eine Instanz von RoutePlanner, um die Gesamtkosten der Route zu berechnen
    planner = RoutePlanner(cities, budget, freie_tage, routes)
    
    # Berechnet die Gesamtkosten, die neue Route und die Anzahl der Tage
    total_price, new_route, tage = planner.calculate_total_price()
    
    # Erstellt eine Karte für die Route mit den definierten Städten
    football_map = travelPlanner.create_rout_map(new_route)
    
    # Macht die Stadtinformationen serialisierbar für die JSON-Antwort
    cities_serializable = {city_name: city_obj.to_dict() for city_name, city_obj in cities.items()}
    
    # Konvertiert die Karte in eine HTML-String-Repräsentation
    map_html = football_map._repr_html_()
    
    # Aktualisiert den Inhalt mit zusätzlichen Informationen für die Antwort
    content.update({
        'city_with_rating': cities_serializable,
        'm_html': map_html,
        'route': new_route,
        'mse': mse,
        'total_price': round(total_price, 2),
        'average_rating': average_rating,
        'tage': tage
    })
    
    # Gibt den aktualisierten Inhalt als JSON zurück
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
