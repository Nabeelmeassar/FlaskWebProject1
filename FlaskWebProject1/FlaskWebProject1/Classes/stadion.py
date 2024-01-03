import heapq
from math import radians, cos, sin, asin, sqrt
from FlaskWebProject1.Classes.distance import Distance

class Stadion:
    def __init__():
        pass
    def __init__(self, id, stadt, verein, stadion, lat, long, hotelkosten, ticketkosten, stadt_qualitaet, stadion_stimmung, nachbarn_ids, nachbarns=None, distance = None):
        self.id = id
        self.stadt = stadt
        self.verein = verein
        self.stadion = stadion
        self.lat = float(lat)
        self.long = float(long)
        self.hotelkosten = hotelkosten
        self.ticketkosten = ticketkosten
        self.stadt_qualitaet = stadt_qualitaet
        self.stadion_stimmung = stadion_stimmung
        self.nachbarn_ids = nachbarn_ids.split(';') if isinstance(nachbarn_ids, str) else nachbarn_ids        
        self.nachbarns = nachbarns if nachbarns is not None else []
        self.distances = distance if nachbarns is not None else []
        
    def add_distance(self, stadion):
        distance = Distance(self,stadion, self.calculate_distance(stadion))
        self.distances.append(distance)
    def calculate_distance(self, other):
        # Umwandlung von Dezimalgraden in Radian
        lon1, lat1, lon2, lat2 = map(radians, [self.long, self.lat, other.long, other.lat])
        
        # Haversine Formel
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371 # Radius der Erde in Kilometern
        return round((c * r),2)
    def add_nachbar(self, stadion):
        self.nachbarns.append(stadion)

    def print_stadion_tree(self, level=0):
        print(' ' * level + self.stadion)
        for nachbar in self.nachbarns:
            nachbar.print_stadion_tree(level + 1)
    def to_dict(self):
        return {
            'id': self.id,
            'stadt': self.stadt,
            'verein': self.verein,
            'stadion': self.stadion,
            'lat': self.lat,
            'long': self.long,
            'hotelkosten': self.hotelkosten,
            'ticketkosten': self.ticketkosten,
            'stadt_qualitaet': self.stadt_qualitaet,
            'stadion_stimmung': self.stadion_stimmung,
            'nachbarn_ids': self.nachbarn_ids,
             'nachbarns': [n.id for n in self.nachbarns],
            'distances': [d.to_dict() for d in self.distances]
        }
