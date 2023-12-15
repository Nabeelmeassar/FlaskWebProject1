"""
Routes and views for the flask application.
"""

from datetime import datetime
from email import message
from operator import iconcat
from pprint import pprint
from tokenize import Double
from flask import json, render_template, request
from folium.map import Icon
from FlaskWebProject1 import app
from flask import Flask, current_app
from math import radians, sin, cos, sqrt, atan2
from FlaskWebProject1.Classes.CSVFile import CSVFile
from FlaskWebProject1.Classes.Map import Map
from FlaskWebProject1.Classes.Verein import Verein
from flask import Flask, request, jsonify
import heapq
import folium
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import json
import networkx as nx




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
    coordinates = list(map(get_club_coordinates_and_city, path))
    best_route = list(map(get_city_name, path))
    # Gesamtroute erstellen
    coordinates_with_names = {club.stadt: (club.lat, club.long) for club in clubs}
     
 
    # Karte erstellen
    football_map = Map().create_football_clubs_map(coordinates_with_names, routes, best_route)
    # Konvertieren Sie die Karte in eine HTML-Zeichenkette
    map_html = football_map._repr_html_()
    # Get the distance of the shortest path
    shortest_distance_km = distances[target_vertex]  # Ensure this distance is in kilometers

    # Add all the new and processed data to the original content
    content.update({
        'start': club_dict[start_vertex].stadt,
        'ziel': club_dict[target_vertex].stadt,
        'max_distance': round(max_distance, 2),
        'budget': budget,
        'Kuerzester_Weg': list(map(get_club_name, path)),  # Convert club IDs to names
        'Kuerzeste_Entfernung_km': round(shortest_distance_km, 2),
        'm_html': map_html

    })

    # Return the modified content as JSON
    return jsonify(content)

# csv_file_verein = CSVFile('C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\verein.csv')
# csv_file_cities = CSVFile('C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\cities_data.csv')
csv_file_clubs = CSVFile('C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\clubs.csv')
# vereinen = csv_file_verein.read_verein_from_csv()
# cities = csv_file_cities.read_cities_from_csv()
clubs = csv_file_clubs.read_clubs_from_csv()

# Erstelle ein Dictionary fuer einen schnelleren Zugriff auf die Vereine
club_dict = {club.id: club for club in clubs}

for club in clubs:
    for _id in club.nachbarn_ids:
        # Zugriff auf den Nachbarverein ueber das Dictionary
        nachbar_club = club_dict.get(int(_id))
        if nachbar_club:
            club.nachbarns.append(nachbar_club)
            club.add_distance(nachbar_club)
 # Erstelle ein Dictionary von Clubs, das die Nachbarn und Entfernungen abbildet
club_graph = {club.id: {distance.other.id: distance.distance_in_km for distance in club.distances} for club in clubs}
# Dictionary zum Speichern der Routen:
routes = []

# Schleife ueber die Liste von Stadien
for club in clubs:
    # Schleife ueber die Liste von Nachbarstadien fuer jedes Stadion
    for x in club.nachbarns:
       rout = (club.stadt, x.stadt)
       routes.append(rout)


@app.route('/', methods = ['GET','POST'])
def home():

    if request.method == 'POST':  #this block is only entered when the form is submitted
        # Daten aus den Dropdown-Menues
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
        clubs = clubs,
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
    path = path[::-1]  
    return path


def get_club_name(club_id):
    return ' => ' + str(club_id) + ' ' + club_dict[club_id].stadt 

def get_club_coordinates_and_city(club_id):
    # Holen Sie die Koordinaten und den Stadtnamen des Clubs aus dem club_dict
    if club_id in club_dict:
        club_info = club_dict[club_id]
        return [club_info.lat, club_info.long], club_info.stadt
    else:
        return None, None  # Oder eine andere Form der Fehlerbehandlung
    
def get_city_name(club_id):
    # Holen Sie die Koordinaten und den Stadtnamen des Clubs aus dem club_dict
    if club_id in club_dict:
        club_info = club_dict[club_id]
        return club_info.stadt
    else:
        return None, None  # Oder eine andere Form der Fehlerbehandlung
    









############################################################################################
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error

# # CSV-Dateien einlesen
# spielplan_df = pd.read_csv('C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\spielplan.csv')
# kostendaten_df = pd.read_csv('C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\kostendaten.csv')

# # Konvertieren Sie die 'Datum'-Spalte in DateTime-Objekte
# spielplan_df['Datum'] = pd.to_datetime(spielplan_df['Datum'], format='%d.%m.%Y')

# # Zusammenfuehren der Datensaetze anhand der 'ID'-Spalte
# df = pd.merge(spielplan_df, kostendaten_df, on='ID')

# # Umbenennen der Spalte 'Stadt_x' in 'Stadt' und Entfernen der Spalte 'Stadt_y'
# df = df.rename(columns={'Stadt_x': 'Stadt'}).drop('Stadt_y', axis=1)

# # Vorbereitung der Merkmale (X) und der Zielvariable (y)
# # # X = df.drop('Beste Route', axis=1)  # Entfernen Sie die Zielvariable, um das Merkmalsset zu erstellen
# y = df['Beste Route']               # Festlegen der Zielvariablen

# # Vorverarbeitung fuer kategorische Daten
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# # Erstellen Sie einen Vorverarbeiter mit OneHotEncoder fuer kategorische Merkmale
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', categorical_transformer, ['Stadt', 'Verein'])  # Achten Sie darauf, dass diese Spalten in Ihrem DataFrame existieren
#     ])

# # Erstellen einer Pipeline mit Vorverarbeiter und einem Regressor
# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('regressor', RandomForestRegressor(random_state=42))])

# Aufteilung der Daten in Trainings- und Testsets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Trainieren des Modells
# pipeline.fit(X_train, y_train)

# # Prognose fuer den Testdatensatz
# y_pred = pipeline.predict(X_test)

# # Bewertung des Modells
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mittlerer quadratischer Fehler: {mse}')


##################################################################################################################################
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler

# # Angenommen, die 'Entscheidung' ist jetzt eine kontinuierliche Variable, die eine Bewertung oder Wahrscheinlichkeit repraesentiert
# # ...

# # Daten vorbereiten
# features = df[['Budget', 'Distanz', 'Entertainment', 'Tradition']]
# labels = df['Entscheidung']  # Dies ist jetzt kontinuierlich

# # Daten aufteilen in Trainings- und Testset
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# # Feature-Scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Modell erstellen
# model = RandomForestRegressor(n_estimators=100, random_state=42)

# # Modell trainieren
# model.fit(X_train_scaled, y_train)

# # Modell validieren
# predictions = model.predict(X_test_scaled)

# # Genauigkeit berechnen als mittleren quadratischen Fehler (MSE)
# mse = mean_squared_error(y_test, predictions)
# print(f'MSE: {mse}')

# # Modell benutzen, um Vorhersagen zu treffen
# # Beispiel: Vorhersage fuer einen neuen Zielort
# new_data = [[8, 6, 9, 9]]  # Beispielwerte fuer 'Budget', 'Distanz', 'Entertainment', 'Tradition'
# new_data_scaled = scaler.transform(new_data)  # Skalierung der neuen Daten
# prediction = model.predict(new_data_scaled)
# print(f'Vorhergesagte Bewertung: {prediction[0]}')  # Vorhersage ist jetzt eine kontinuierliche Groesse

##############################################################################################################################################
# Erstellen eines DataFrame aus den bereitgestellten Daten
# Hier nehmen wir an, dass 'Bewertung' die Zielvariable ist, die Sie vorhersagen moechten


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Assuming that 'data.csv' is in the same directory as your script, 
# # modify the path if it's located elsewhere.

# # Load the data
# df = pd.read_csv(csv_file_path)

# # Select features and target
# # Assuming that all the remaining columns after dropping 'Ziel' and 'Bewertung' are numeric and relevant features.
# X = df.drop(['Ziel', 'Bewertung'], axis=1)
# y = df['Bewertung'].astype(float)  # Ensure the target is float

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the model
# model = LinearRegression()
# model.fit(X_train, y_train)
# print(f'X_train {X_train}')
# print(f'y_train {y_train}')
# print(f'model {model.__doc__}')

# # Make predictions on the test data
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Der mittlere quadratische Fehler (MSE) des Modells: {mse}")
# mse = mean_squared_error(y_test, y_pred)
# print(f"Der mittlere quadratische Fehler (MSE) des Modells: {mse}")

###########################################################################################################
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_squared_error
# from sklearn.impute import SimpleImputer

# # Erstellen Sie einen fiktiven Datensatz
# np.random.seed(42)  # Fuer reproduzierbare Ergebnisse

# # Beispiel Features
# costs = np.random.randint(100, 1000, 100)  # Kosten fuer einen Aufenthalt
# experiences = np.random.randint(1, 11, 100)  # Erfahrungsbewertung
# qualities = np.random.randint(1, 11, 100)  # Qualitaetsbewertung
# cities = np.random.choice(['CityA', 'CityB', 'CityC'], 100)  # Stadt
# city_ratings = np.random.randint(1, 11, 100)  # Gesamtbewertung der Stadt

# # DataFrame erstellen
# df = pd.DataFrame({
#     'cost': costs,
#     'experience': experiences,
#     'quality': qualities,
#     'city': cities,
#     'city_rating': city_ratings
# })

# # Aufteilung in Feature und Zielvariablen
# X = df.drop('city_rating', axis=1)
# y = df['city_rating']

# # Aufteilung in Trainings- und Testdatensatz
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Numerische Features und kategoriale Features definieren
# numeric_features = ['cost', 'experience', 'quality']
# categorical_features = ['city']

# # Vorverarbeitungsschritte fuer numerische und kategoriale Daten
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# # ColumnTransformer, der unterschiedliche Transformationen auf unterschiedliche Features anwendet
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)])

# # Pipeline erstellen, die die Vorverarbeitung und das Modell umfasst
# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('regressor', RandomForestRegressor(random_state=42))])

# # Parameter Grid fuer GridSearchCV erstellen
# param_grid = {
#     'regressor__n_estimators': [50, 100, 200],
#     'regressor__max_features': ['auto', 'sqrt', 'log2'],
#     'regressor__max_depth': [None, 10, 20, 30]
# }

# # GridSearchCV-Objekt erstellen
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)

# # Model training und Evaluation
# grid_search.fit(X_train, y_train)

# # Ergebnisse
# print("Best parameters found:")
# print(grid_search.best_params_)
# best_model = grid_search.best_estimator_

# # Vorhersagen auf dem Testset
# y_pred = best_model.predict(X_test)

# # RMSE berechnen
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# print(f"Test RMSE: {rmse}")


##################################################################################################################################

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np
# import joblib

# csv_file_path = 'C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\data.csv'

# # Daten laden
# data = pd.read_csv(csv_file_path)

# # Nehmen wir an, dass die Bewertungsspalte fehlt, und fuellen sie mit zufaelligen Werten
# # Dies ist nur fuer das Beispiel; Sie sollten Ihre tatsaechlichen Bewertungsdaten verwenden
# # data['Bewertung'] = np.random.rand(len(data))

# # Kategoriale und numerische Spalten identifizieren
# categorical_cols = [cname for cname in data.columns if data[cname].dtype == "object"]
# numerical_cols = [cname for cname in data.columns if cname not in categorical_cols and cname != 'Bewertung']

# # Zielvariable (y) und Features (X) festlegen
# y = data['Bewertung']
# X = data.drop('Bewertung', axis=1)

# # Vorverarbeitung fuer numerische Daten
# numerical_transformer = SimpleImputer(strategy='mean')

# # Vorverarbeitung fuer kategoriale Daten
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# # Vorverarbeitungstransformator
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])

# # Modell
# model = RandomForestRegressor(n_estimators=100, random_state=0)

# # Pipeline erstellen
# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('model', model)])

# # Trainings- und Testdaten aufteilen
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Modell trainieren
# pipeline.fit(X_train, y_train)

# # Modell speichern
# model_filename = 'travel_recommendation_model.joblib'
# joblib.dump(pipeline, model_filename)

# # Vorhersagen treffen
# preds = pipeline.predict(X_test)

# # Modellbewertung
# rmse = mean_squared_error(y_test, preds, squared=False)
# mae = mean_absolute_error(y_test, preds)
# print('Test RMSE:', rmse)
# print('Test MAE:', mae)

# # Funktion, die die neuen Benutzerdaten akzeptiert und eine Vorhersage erstellt
# def update_preferences_and_predict(model_path, current_user_data, new_preferences):
#     # Modell laden
#     trained_model = joblib.load(model_path)
    
#     # Aktualisieren Sie die Benutzerdaten mit den neuen Praeferenzen
#     for preference, value in new_preferences.items():
#         if preference in current_user_data:
#             current_user_data[preference] = value
#         else:
#             raise ValueError(f"Preference '{preference}' is not recognized.")
    
#     # Konvertieren Sie die aktualisierten Daten in das richtige Format fuer das Modell
#     transformed_data = trained_model.named_steps['preprocessor'].transform(pd.DataFrame([current_user_data]))
    
#     # Erstellen Sie eine neue Vorhersage mit dem Modell
#     new_prediction = trained_model.named_steps['model'].predict(transformed_data)
    
#     return new_prediction[0]

# # Beispielbenutzerdaten (muessen den Daten im Modell entsprechen)
# user_data_example = X.iloc[0].to_dict()

# Neue Praeferenzen einfuegen
# new_preferences = {
#     'Zielort_Besucherzahl': 75000,
#     'Zielort_Topteam': 'Hoch',
#     'Zielort_Hotelkosten': 292,
#     'Zielort_Ticketkosten': 'Hoch',
#     'Zielort_Nachtleben': 'Exzellent',
#     'Zielort_Jahre in der 1.Bundesliga': 59,
#     'Zielort_Trophen letzten 5 Jahre': 30,
#     'User_Budget': 1200,
#     'User_Max. Distanz': 1000,
#     'User_Entertainment Fussballfan': 'Hoch',
#     'User_Traditionsfussballfan': 'Hoch'
# }


# # Neue Vorhersage basierend auf aktualisierten Praeferenzen
# new_prediction = update_preferences_and_predict(model_filename, user_data_example, new_preferences)

# # Ausgabe der aktualisierten Vorhersage
# print(f"Basierend auf Ihren neuen Praeferenzen ist die neue Bewertung: {new_prediction}")
##################################################################################################################


# Pfad zur CSV-Datei
# csv_file_path = 'C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\data.csv'

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_squared_error
# import joblib


# # Daten laden
# data = pd.read_csv(csv_file_path)

# # Kategoriale und numerische Spalten identifizieren
# categorical_cols = [cname for cname in data.columns if 
#                     data[cname].dtype == "object" and 
#                     cname not in ['Ziel', 'Bewertung']]
# numerical_cols = [cname for cname in data.columns if 
#                   data[cname].dtype in ['int64', 'float64'] and 
#                   cname not in ['Ziel', 'Bewertung']]

# # Zielvariable (y) und Features (X) festlegen
# y = data['Bewertung']
# X = data.drop(['Ziel', 'Bewertung'], axis=1)

# # Vorverarbeitung fuer numerische Daten
# numerical_transformer = SimpleImputer(strategy='mean')

# # Vorverarbeitung fuer kategoriale Daten
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# # Vorverarbeitungstransformator
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])

# # Modell
# model = RandomForestRegressor(n_estimators=100, random_state=0)

# # Pipeline erstellen
# pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                            ('model', model)])

# # Trainings- und Testdaten aufteilen
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Modell trainieren
# pipeline.fit(X_train, y_train)

# # Modellbewertung
# preds_test = pipeline.predict(X_test)
# rmse = mean_squared_error(y_test, preds_test, squared=False)
# print(f'Test RMSE: {rmse}')

# # Benutzerpraeferenzen definieren
# user_preferences = {
#     'Zielort_Besucherzahl': 75000,
#     'Zielort_Topteam': 'Hoch',
#     'Zielort_Hotelkosten': 292,
#     'Zielort_Ticketkosten': 'Hoch',
#     'Zielort_Nachtleben': 'Exzellent',
#     'Zielort_Jahre in der 1.Bundesliga': 59,
#     'Zielort_Trophen letzten 5 Jahre': 30,
#     'User_Budget': 1200,
#     'User_Max. Distanz': 1000,
#     'User_Entertainment Fussballfan': 'Hoch',
#     'User_Traditionsfussballfan': 'Hoch'
# }

# # Erstellen Sie eine Kopie von 'X' und aktualisieren Sie sie mit den Benutzerpraeferenzen
# X_pref = X.copy()
# for pref_name, pref_value in user_preferences.items():
#     if pref_name in X.columns:
#         X_pref[pref_name] = pref_value
#     # else:
      
#         # Fuer kategoriale Praeferenzen, setzen Sie die entsprechende dummy-Variable
#         # cat_pref_name = f"{pref_name}_{pref_value}"
#         # if cat_pref_name in X.columns:
#         #     X_pref[cat_pref_name] = 1
#         #     # Setzen Sie alle anderen Kategorien dieser Variablen auf 0
#         #     for col in X.columns:
#         #         if col.startswith(pref_name + "_") and col != cat_pref_name:
#         #             X_pref[col] = 0
# feature_importances = pipeline.named_steps['model'].feature_importances_
# feature_names = numerical_cols + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols))
# for name, importance in sorted(zip(feature_names, user_preferences), key=lambda x: x[1], reverse=True):
#     print(f"{name}: {importance}")
# user_pref_df = pd.DataFrame(columns=X.columns)
# for col in numerical_cols:
#     user_pref_df[col] = [user_preferences.get(col, X_train[col].median())]
# for col in categorical_cols:
#     user_pref_df[col] = [user_preferences.get(col, X_train[col].mode()[0])]
# user_predicted_rating = pipeline.predict(user_pref_df)[0] # Benutzen Sie hier nicht 'user_pref_df_encoded'
# print(f'Vorhergesagte Bewertung fuer Benutzerpraeferenzen: {user_predicted_rating}')   
# # Vorhersagen fuer alle Staedte mit Benutzerpraeferenzen treffen
# predicted_ratings = pipeline.predict(X_pref)
# # Vorhersagen zu den Originaldaten hinzufuegen
# data_with_predictions = data.copy()
# data_with_predictions['Predicted_Bewertung'] = predicted_ratings
# # Die Top 5 Zielorte basierend auf den Vorhersagen auswaehlen
# top_cities = data_with_predictions.nlargest(5, 'Predicted_Bewertung')

# # Ausgabe der Top 5 Zielorte
# print("Top 5 empfohlene Zielstaedte basierend auf Benutzerpraeferenzen:")
# print(top_cities[['Ziel', 'Predicted_Bewertung']])

# # Fuegen Sie die Vorhersagen zum urspruenglichen DataFrame hinzu und sortieren Sie die Staedte
# data['Predicted_Bewertung'] = predicted_ratings

# print(data)
# cities_with_low_rating = data[data['Predicted_Bewertung'] < user_predicted_rating]
# print(f'cities_with_low_rating {cities_with_low_rating}')
###########################################################################################################################################
csv_file_path = 'C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\data.csv' 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib

# Daten laden
data = pd.read_csv(csv_file_path)

# Kategoriale und numerische Spalten identifizieren
categorical_cols = [cname for cname in data.columns if data[cname].dtype == "object" and cname not in ['Ziel', 'Bewertung']]
numerical_cols = [cname for cname in data.columns if cname not in categorical_cols and cname != 'Bewertung' and cname != 'Ziel']

# Zielvariable (y) und Features (X) festlegen
y = data['Bewertung']
X = data.drop(['Ziel', 'Bewertung'], axis=1)

# Vorverarbeitung faer numerische Daten
numerical_transformer = SimpleImputer(strategy='mean')

# Vorverarbeitung faer kategoriale Daten
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Vorverarbeitungstransformator
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Modell
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Pipeline erstellen
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
pipeline.fit(X_train, y_train)

# Modellbewertung
preds_test = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, preds_test, squared=False)
print(f'Test RMSE: {rmse}')

# Modell speichern
model_filename = 'travel_recommendation_model.joblib'
joblib.dump(pipeline, model_filename)

# Neue Benutzerpraeferenzen
new_preferences = {
        'Entfernung_km': 600,
        'Zielort_Nachtleben': 'Hoch',
        'User_Budget': 1200,
        'User_Max. Distanz': 1000,
        'User_Entertainment Fussballfan': 'Hoch',
        'User_Traditionsfussballfan': 'Hoch'  
}

# DataFrame fuer neue Benutzerpraeferenzen erstellen
user_pref_df = pd.DataFrame(columns=X.columns)
for col in numerical_cols:
    user_pref_df[col] = [new_preferences.get(col, X_train[col].median())]
for col in categorical_cols:
    user_pref_df[col] = [new_preferences.get(col, X_train[col].mode()[0])]

# Vorhersage der Bewertung mit den neuen Benutzerpraeferenzen
user_pref_df_encoded = pipeline.named_steps['preprocessor'].transform(user_pref_df)

feature_importances = pipeline.named_steps['model'].feature_importances_
feature_names = numerical_cols + \
                list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols))
for name, importance in sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance}")

user_predicted_rating = pipeline.predict(user_pref_df)[0]  # Benutzen Sie hier nicht 'user_pref_df_encoded'
print(f'Vorhergesagte Bewertung fuer Benutzerpraeferenzen: {user_predicted_rating}')

# Vorhersagen faer alle Staedte treffen
data['Predicted_Bewertung'] = pipeline.predict(X)
######################################################

print('#######################################################################')
# Die Staedte auswaehlen, deren vorhergesagte Bewertungen nahe der Benutzerbewertung sind
# Dies sind die Staedte, die wahrscheinlich am besten zu den Praeferenzen des Benutzers passen
suitable_cities = data[abs(data['Predicted_Bewertung'] - user_predicted_rating) < 0.3]

# Ausgabe der Staedte, die gut zu den Benutzerpraeferenzen passen
print('Staedte, die gut zu den Benutzerpraeferenzen passen:')
print(suitable_cities[['Ziel', 'Predicted_Bewertung']])

# Die Staedte auswaehlen, deren vorhergesagte Bewertungen deutlich von der Benutzerbewertung abweichen
# Diese Staedte sind wahrscheinlich weniger geeignet fuer den Benutzer
unsuitable_cities = data[abs(data['Predicted_Bewertung'] - user_predicted_rating) >= 0.3]

# Ausgabe der Staedte, die weniger gut zu den Benutzerpraeferenzen passen
print('Staedte, die weniger gut zu den Benutzerpraeferenzen passen:')
print(unsuitable_cities[['Ziel', 'Predicted_Bewertung']])
print('#######################################################################')

######################################################

# # Die Top-5-Zielstaedte basierend auf den Vorhersagen auswaehlen
# top_cities = data.nlargest(5, 'Predicted_Bewertung')[['Ziel', 'Predicted_Bewertung']]
# print('Top 5 Zielstaedte basierend auf den Vorhersagen:')
# print(top_cities)
# cities_with_low_rating = data[data['Predicted_Bewertung'] <= user_predicted_rating]

# # Ausgabe dieser Staedte
# print(cities_with_low_rating[['Ziel', 'Predicted_Bewertung']])

# print(data)
# print(f'new_preferences {new_preferences}')



########################################################################################################################################