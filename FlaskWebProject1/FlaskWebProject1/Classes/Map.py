import folium
import json

class Map:
    def __init__(self):
        # Load the GeoJSON for Germany when initializing the class
        germany_geojson_path = 'C:\\Users\\nn\\source\\repos\\FlaskWebProject1\\FlaskWebProject1\\FlaskWebProject1\\static\\scripts\\2_hoch.geo.json'
        with open(germany_geojson_path, 'r', encoding='utf-8') as f:
            self.germany_geojson = json.load(f)

    def create_football_clubs_map(self, clubs_coords, routes, best_route=None, route_color='blue', best_route_color='green', best_route_weight=5):
        # Create a Map instance centered on Germany
        m = folium.Map(location=[51.1657, 10.4515], zoom_start=6)

        # Add the boundaries of Germany
        geojson_layer = folium.GeoJson(
            self.germany_geojson,
            name='Germany GeoJSON'
        ).add_to(m)

        # Add markers for the best route
        if best_route:
            for index, club in enumerate(best_route, start=1):
                if club in clubs_coords:
                    coords = clubs_coords[club]
                    folium.Marker(
                        location=coords,
                        tooltip=club,
                        icon=folium.Icon(icon=str(index), prefix='fa', color='orange'),
                    ).add_to(m)

        # Draw lines between the clubs for all routes
        for route in routes:
            start_coords = clubs_coords.get(route[0])
            end_coords = clubs_coords.get(route[1])
            if start_coords and end_coords:
                folium.PolyLine(
                    locations=[start_coords, end_coords],
                    color=route_color,
                    weight=2.5,
                    opacity=0.5,
                    name='Regular Route'
                ).add_to(m)

        # Draw the best route in a different color and with a different weight
        if best_route:
            best_route_coords = [clubs_coords[club] for club in best_route if club in clubs_coords]
            folium.PolyLine(
                locations=best_route_coords,
                color=best_route_color,
                weight=best_route_weight,
                opacity=1,
                name='Best Route'
            ).add_to(m)

        # Limit the map to Germany
        bounds = [[47.2701114, 5.8663425], [55.0815, 15.0418962]]
        m.fit_bounds(bounds)
        return m
