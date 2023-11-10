import csv
from FlaskWebProject1.Classes.Verein import Verein
from FlaskWebProject1.Classes.city import City

class CSVFile:
    """description of class"""
    
    def __init__(self, file_path):
        self.file_path = file_path

    def read_verein_from_csv(self):
        verein = []
        with open(self.file_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter=';')
            first_row = next(reader)
            if first_row:
                # Wenn die erste Zeile nicht leer ist, verarbeiten Sie sie
                print(first_row)
            for row in reader:
                # Verarbeitet die restlichen Zeilen
                name = row[0]
                stadionname = row[1]
                lat = float(row[2])
                lon = float(row[3])
                city = Verein(name,stadionname, lat, lon)
                verein.append(city)
            # cities.append(city)
        return verein
    
    def read_cities_from_csv(self):
        cities = []
        with open(self.file_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter=',')
            first_row = next(reader)
            if first_row:
                # Wenn die erste Zeile nicht leer ist, verarbeiten Sie sie
                print(first_row)
            for row in reader:
                # Verarbeitet die restlichen Zeilen name, lat, lng, country, population
                name = row[0]
                lat = float(row[1])
                lon = float(row[2])
                country = row[3]
                population = int(row[4])
                city = City(name, lat, lon,country, population)
                cities.append(city)
            # cities.append(city)
        return cities