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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from joblib import dump



# with open("C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\csv\\Vereins.csv", 'r') as file:
#     csvreader = file.read().split(';')
#     print(file.read())
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
        spielart = request.form['spielart']
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
        ziel = selected_ziel
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
