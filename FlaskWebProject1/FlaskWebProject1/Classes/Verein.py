# Definieren Sie die Klasse Verein
class Verein:
    # Der Konstruktor der Klasse, der aufgerufen wird, wenn ein neues Objekt dieser Klasse erstellt wird
    def __init__(self, name, stadionname, lat, lon):
        # `self` ist eine Referenz auf das aktuelle Objekt der Klasse
        # `name` ist der Name des Vereins
        self.name = name
        # `stadionname` ist der Name des Stadions des Vereins
        self.stadionname = stadionname
        # `lat` und `lon` sind die Breiten- und Längengrade des Stadions
        self.lat = lat
        self.lon = lon