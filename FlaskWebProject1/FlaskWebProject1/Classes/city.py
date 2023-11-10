# Definition der Klasse City
class City:
    # Die __init__ Methode ist der Konstruktor der Klasse. 
    # Sie wird aufgerufen, wenn ein neues Objekt dieser Klasse erstellt wird.
    def __init__(self, name, lat, lon, country, population):
        self.name = name  # Der Name der Stadt
        self.lat = lat  # Der Breitengrad der Stadt
        self.lon = lon  # Der L�ngengrad der Stadt
        self.country = country  # Das Land, in dem sich die Stadt befindet
        self.population = population  # Die Bev�lkerungszahl der Stadt

    # Die __str__ Methode wird aufgerufen, wenn das Objekt in einen String umgewandelt wird.
    # Sie gibt eine lesbare Darstellung des Objekts zur�ck.
    def __str__(self):
        # Verwendung von f-Strings f�r eine einfache Formatierung
        return f'{self.name}, {self.country} ({self.lat}, {self.lon}): {self.population} Einwohner'
