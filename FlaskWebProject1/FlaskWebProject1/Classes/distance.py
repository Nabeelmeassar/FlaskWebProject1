class Distance:
    def __init__(self, obj, other, distance_in_km):
        self.obj = obj
        self.other = other
        self.distance_in_km = distance_in_km
    def to_dict(self):
        return {
            'obj_id': self.obj.id,
            'other_id': self.other.id,
            'distance_in_km': self.distance_in_km
        }