import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

class UserPreferencePredictor:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        
    def load_and_prepare_data(self):
        try:
            data = pd.read_csv(self.csv_file_path)
            X = data.drop('Bewertung', axis=1)
            y = data['Bewertung']
            return X, y
        except FileNotFoundError as e:
            print(f"Datei nicht gefunden: {e}")
            return None, None
        except KeyError as e:
            print(f"Fehlende Spalte: {e}")
            return None, None

    def evaluate_model(self):
        predictions = self.pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        print(f"Modell MSE: {mse}")
        return mse

    def train_model(self, X_train, y_train):
        categorical_features = ['Ziel']  # Beispiel, sollte angepasst werden
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
            ],
            remainder='passthrough'
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        self.pipeline.fit(X_train, y_train)

    def predict_user_preferences(self, new_user_preferences):
        X, y = self.load_and_prepare_data()
        if X is None or y is None:
            return None

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train_model(X_train, y_train)

        mse = self.evaluate_model()

        predictions = {}
        for city in X['Ziel'].unique():
            city_data = X[X['Ziel'] == city].copy()
            for feature, value in new_user_preferences.items():
                if feature in city_data:
                    city_data[feature] = value
            city_pred = self.pipeline.predict(city_data)[0]
            predictions[city] = city_pred

        predictions = OrderedDict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))

        for city, pred in predictions.items():
            print(f"Stadt: {city}, Vorhergesagte Bewertung: {pred}")

        return predictions, mse