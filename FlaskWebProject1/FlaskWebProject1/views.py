"""
Routes and views for the flask application.
"""

from datetime import datetime
from pprint import pprint
from tokenize import Double
from flask import json, render_template, request
from FlaskWebProject1 import app
from flask import Flask, current_app
from math import radians, sin, cos, sqrt, atan2
from FlaskWebProject1.Classes.CSVFile import CSVFile
from FlaskWebProject1.Classes.Verein import Verein
from flask import Flask, request, jsonify
import heapq


@app.route('/postjson', methods=['POST'])
def post_json_handler():
    content = request.get_json()
    start_vertex = int(content.get('select_start'))
    target_vertex = int(content.get('select_ziel'))
    max_distance = float(content.get('max_distance'))
    budget = float(content.get('budget'))

    # Call shortest_path and save the result
    path = shortest_path(club_graph, start_vertex, target_vertex)
    distances, _ = calculate_distances(club_graph, start_vertex)  # Get the distances dictionary

    # Get the distance of the shortest path
    shortest_distance_km = distances[target_vertex]  # Ensure this distance is in kilometers

    # Add all the new and processed data to the original content
    content.update({
        'start': start_vertex,
        'ziel': target_vertex,
        'max_distance': max_distance,
        'budget': budget,
        'shortest_path': path,  # Add the shortest path here
        'shortest_distance_km': shortest_distance_km  # Add the shortest distance in kilometers here
    })

    # Return the modified content as JSON
    return jsonify(content)

csv_file_verein = CSVFile('C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\verein.csv')
csv_file_cities = CSVFile('C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\cities_data.csv')
csv_file_clubs = CSVFile('C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\clubs.csv')
vereinen = csv_file_verein.read_verein_from_csv()
cities = csv_file_cities.read_cities_from_csv()
clubs = csv_file_clubs.read_clubs_from_csv()

# Erstelle ein Dictionary für einen schnelleren Zugriff auf die Vereine
club_dict = {club.id: club for club in clubs}

for club in clubs:
    for _id in club.nachbarn_ids:
        # Zugriff auf den Nachbarverein über das Dictionary
        nachbar_club = club_dict.get(int(_id))
        if nachbar_club:
            club.nachbarns.append(nachbar_club)
            club.add_distance(nachbar_club)
 # Erstelle ein Dictionary von Clubs, das die Nachbarn und Entfernungen abbildet
club_graph = {club.id: {distance.other.id: distance.distance_in_km for distance in club.distances} for club in clubs}
      
@app.route('/', methods = ['GET','POST'])
def home():

    if request.method == 'POST':  #this block is only entered when the form is submitted
        # Daten aus den Dropdown-Menüs
        selected_start = request.form['select_start']
        selected_ziel = request.form['select_ziel']

        # Daten aus den Eingabefeldern
        max_distance = float(request.form['max_distance'])
        budget = float(request.form['budget'])
        bevorzugtes_wetter = request.form['bevorzugtes_wetter']
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
        clubs = clubs,
)
    
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
        cities = cities,
        vereinen = vereinen,
        clubs = clubs,
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
def calculate_distances(graph, start_vertex):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start_vertex] = 0

    predecessors = {vertex: None for vertex in graph}

    pq = [(0, start_vertex)]
    while len(pq) > 0:
        current_distance, current_vertex = heapq.heappop(pq)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, predecessors

def shortest_path(graph, start_vertex, target_vertex):
    distances, predecessors = calculate_distances(graph, start_vertex)
    path = []
    while target_vertex is not None:
        path.append(target_vertex)
        target_vertex = predecessors[target_vertex]
    path = path[::-1]  # reverse the path
    return path




# Genetic Algorithm
def genetic_algorithm(attractions, population_size, generations, mutation_rate):
    # Create the initial population
    population = create_initial_population(attractions)

    for generation in range(generations):
        # Calculate the fitness of each route in the population
        fitness = [1 / calculate_distance(route) for route in population]

        # Select the top routes based on their fitness
        num_top_routes = max(2, int(population_size * 0.2))
        top_routes_indices = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)[:num_top_routes]
        top_routes = [population[i] for i in top_routes_indices]

        # Create the next generation through crossover and mutation
        next_generation = top_routes

        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(top_routes, 2)
            child = simple_crossover(parent1, parent2)
            if random.random() < mutation_rate:
                mutate(child)
            next_generation.append(child)

        population = next_generation

    # Find the best route in the final population
    best_route_index = top_routes_indices[0]
    best_route = population[best_route_index]



