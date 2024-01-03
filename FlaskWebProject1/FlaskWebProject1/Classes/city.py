# Importiere die Bibliothek geopy, um die Distanz zu berechnen
import csv
from geopy.distance import geodesic
# Eine Klasse, die die Informationen über eine Stadt speichert
class City:
    cities = {}
    def __init__(self,id, name, gps, rating, hotel_cost, ticket_cost):
        self.name = name
        self.id = id
        self.gps = gps # Ein Tupel aus (latitude, longitude)
        self.rating = rating
        self.hotel_cost = float(hotel_cost) # Pro Nacht
        self.ticket_cost = float(ticket_cost)# Pro Spiel
    def to_dict(self):
        # Convert the City object to a dictionary
        return {
            'id': self.id,
            'name': self.name,
            'gps': self.gps,
            'rating': self.rating,
            'hotel_cost': self.hotel_cost,
            'ticket_cost': self.ticket_cost
        }
        
    def update_hotel_cost(self, new_hotel_cost):
        self.hotel_cost = float(new_hotel_cost)

    def update_ticket_cost(self, new_ticket_cost):
        self.ticket_cost = float(new_ticket_cost)

    def update_rating(self, new_rating):
        self.rating = new_rating
        
    def get_hotel_tick_cost(self):
        cost = self.hotel_cost + self.ticket_cost
        return cost
    
def get_cities():
    cities = {} 
    # Schritt 2: Öffnen der CSV-Datei und Erstellen von City-Objekten
    csv_path = 'C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\csv\\clubs.csv'
    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        # next(reader)  # Überspringen Sie die Kopfzeile, falls vorhanden
        # Schritt 3: Erstellen von City-Objekten und Hinzufügen zur Liste
        for row in reader:
            # Angenommen, jede Zeile hat Stadtname, Bevölkerung und Land in dieser Reihenfolge
            city = City(row[0], row[1],(row[4],row[5]),0 , row[6], row[7])
            cities[city.name] = city
    return cities


