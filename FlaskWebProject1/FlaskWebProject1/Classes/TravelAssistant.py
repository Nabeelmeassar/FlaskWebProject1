import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

class TravelAssistant:
    def __init__(self, csv_file_path, new_user_preferences, cities):
        self.csv_file_path = csv_file_path
        self.new_user_preferences = new_user_preferences
        self.cities = cities
        
    # Funktion zum Laden und Vorbereiten der Daten
    def load_and_prepare_data(self):
        try:
            data = pd.read_csv(self.csv_file_path)
            features = data.drop('Bewertung', axis=1)
            labels = data['Bewertung']
            return features, labels
        except FileNotFoundError as e:
            print(f"Datei nicht gefunden: {e}")
            return None, None
        except KeyError as e:
            print(f"Fehlende Spalte: {e}")
            return None, None

    # Hauptfunktion fuer Vorhersagen
    def predict_user_preferences(self):
        X, y = self.load_and_prepare_data()
        if X is None or y is None:
            return None

        # Identifizieren kategorischer Spalten
        categorical_features = ['Ziel']  # Beispiel, sollte angepasst werden
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Erstellen des ColumnTransformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
            ],
            remainder='passthrough'
        )

        # Aufteilen in Trainings- und Testdaten
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Erstellen der Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Modell trainieren
        pipeline.fit(X_train, y_train)

        # Modell bewerten
        predictions = pipeline.predict(X_test)
        model_mse = mean_squared_error(y_test, predictions)
        

        # Vorhersagen vorbereiten
        predictions = {}
        for city in X['Ziel'].unique():
            city_data = X[X['Ziel'] == city]        
            # Benutzerpraeferenzen aktualisieren
            for feature, value in self.new_user_preferences.items():
                if feature in city_data:
                    city_data[feature] = value
            city_pred = pipeline.predict(city_data)[0]
            predictions[city] = city_pred
        
        # Sortieren der Vorhersagen
        predictions = OrderedDict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))

        # Vorhersagen ausgeben
        for city, pred in predictions.items():
            print(f"Stadt: {city}, Vorhergesagte Bewertung: {pred} ")
            if self.cities[city].name == city:
                self.cities[city].update_rating(pred)

        return self.cities, model_mse