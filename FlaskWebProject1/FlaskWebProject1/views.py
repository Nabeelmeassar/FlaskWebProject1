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


cities = get_cities()
csv_data_path = 'C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\training_data.csv'
 # Pfad zur GeoJSON-Datei von Deutschland
germany_geojson_path = 'C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\scripts\\2_hoch.geo.json'

@app.route('/post_preference_json', methods=['POST'])

def post_preference_json_handler():
    # JSON-Inhalt aus der Anfrage extrahieren
    content = request.get_json()
    
    # Verschiedene Parameter aus dem JSON-Körper extrahieren
    select_start = content.get('Select_start')
    freie_tage = int(content.get('Tage'))
    gewicht = int(content.get('Gewicht'))
    budget = int(content.get('Person_Budget'))
    
    # Erstellt eine Instanz von TravelAssistant, um Benutzerpräferenzen zu berechnen
    travelAssistant = TravelAssistant(csv_data_path, content, cities)
    
    # Berechnet die Benutzerpräferenzen basierend auf den angegebenen Daten
    city_with_rating, mse = travelAssistant.predict_user_preferences()
    
    # Erstellt eine Instanz von TravelPlanner, um die beste Route zu finden
    travelPlanner = TravelPlanner(cities, gewicht,freie_tage, budget)
    
    # Verwendet den Algorithmus "Best-First-Search" zur Routenfindung
    routes = travelPlanner.best_first_search(select_start)
    
    # Erstellt ein Wörterbuch der Städte auf der Route
    route = {city.id: city for city in cities.values() if city.name in routes}
     
    # Berechnet die Gesamtkosten, die neue Route und die Anzahl der Tage
    total_price, new_route, tage, average_rating = travelPlanner.calculate_total_price(routes)
    
    # Erstellt eine Karte für die Route mit den definierten Städten
    football_map = travelPlanner.create_rout_map(new_route, germany_geojson_path)
    google_map_route = travelPlanner.create_google_maps_route_link(new_route)
    
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
        'tage': tage,
        'mse': mse,
        'google_map_route': google_map_route,
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
