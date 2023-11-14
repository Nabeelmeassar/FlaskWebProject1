"""
Routes and views for the flask application.
"""

from datetime import datetime
from tokenize import Double
from flask import render_template, request
from FlaskWebProject1 import app
from flask import Flask, current_app
from math import radians, sin, cos, sqrt, atan2
from FlaskWebProject1.Classes.CSVFile import CSVFile
from FlaskWebProject1.Classes.Verein import Verein
from flask import Flask, request, jsonify


@app.route('/postjson', methods=['POST'])
def post_json_handler():
    content = request.get_json()
    selected_start = content.get('select_start')
    selected_ziel = content.get('select_ziel')
    selected_wochentag = content.get('wochentag')

    max_distance = float(content.get('max_distance'))
    budget = float(content.get('budget'))
    bevorzugtes_wetter = content.get('bevorzugtes_wetter')

    distance = calculate_distance(selected_start, selected_ziel)

    # Add all the new and processed data to the original content
    content.update({
        'distance': distance,
        'start': selected_start,
        'ziel': selected_ziel,
        'max_distance': max_distance,
        'budget': budget,
        'bevorzugtes_wetter': bevorzugtes_wetter,
    })

    # Return the modified content as JSON
    return jsonify(content)

csv_file_verein = CSVFile('C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\csv\\verein.csv')
csv_file_cities = CSVFile('C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\csv\\cities_data.csv')
vereinen = csv_file_verein.read_verein_from_csv()
cities = csv_file_cities.read_cities_from_csv()


@app.route('/', methods = ['GET','POST'])
def home():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        # Daten aus den Dropdown-Menüs
        selected_start = request.form['select_start']
        selected_ziel = request.form['select_ziel']
        selected_wochentag = request.form['wochentag']

        # Daten aus den Eingabefeldern
        max_distance = float(request.form['max_distance'])
        budget = float(request.form['budget'])
        bevorzugtes_wetter = request.form['bevorzugtes_wetter']
        distance = calculate_distance(selected_start, selected_ziel)
       
        return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,   
        cities = cities,
        vereinen = vereinen,
        distance = distance,
        start = selected_start,
        ziel = selected_ziel,
        max_distance = max_distance,
        budget = budget,
        bevorzugtes_wetter = bevorzugtes_wetter,
    )
    
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
        cities = cities,
        vereinen = vereinen,
        )


@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
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

def calculate_distance(coord1, coord2):
    
    # Remove parentheses and split by comma for each coordinate
    coord1 = coord1.replace('(','').replace(')','').split(',')
    coord2 = coord2.replace('(','').replace(')','').split(',')

    # Convert strings to floats
    lat1, lon1 = map(float, coord1)
    lat2, lon2 = map(float, coord2)
    # Umrechnung von Grad zu Radian
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

   # Haversine Formel
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    # Radius der Erde in Kilometern (mittlerer Radius)
    r = 6371.0

    # Berechnung der Entfernung
    distance = r * c
    distance = distance * 1.6
    # Rückgabe der Entfernung als Dictionary mit Wert und Einheit
    return round(distance, 2)
