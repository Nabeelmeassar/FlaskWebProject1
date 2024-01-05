function sendPreferenceFormData() {
    var xhr = new XMLHttpRequest();

    xhr.open("POST", '/post_preference_json', true);
    xhr.setRequestHeader("Content-Type", "application/json");
    var entscheidungContentElement = document.getElementById('entscheidung_content');
    var mymap = document.getElementById('mapdiv');
    var city_score_div = document.getElementById('city_score');
    entscheidungContentElement.innerHTML = '';

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var jsonResponse = JSON.parse(xhr.responseText);
            var cityRatings = jsonResponse.city_with_rating;
            // Angenommen, Sie haben ein Objekt mit Städten als Schlüsseln und Scores als Werten.
            var cityScores = jsonResponse.city_score
            console.log(cityScores)
            mymap.innerHTML = '<h2>Kartenuebersicht</h2>'; 
            mymap.innerHTML += jsonResponse.route; 
            mymap.innerHTML += '<h3 class = "alert alert-primary">Gesamtkosten = ' + jsonResponse.total_price + ' €  </h3>'; 
            console.log(jsonResponse.total_price)
            console.log(jsonResponse)
            mymap.innerHTML += jsonResponse.m_html;
            mymap.style.visibility = 'visible';

            var ratingsArray = [];
            for (var city in cityRatings) {
                if (cityRatings.hasOwnProperty(city)) {
                    ratingsArray.push([city, cityRatings[city].rating]); // Assuming each city object has a 'rating'
                }
            }

            // Sort the array by rating in descending order (highest rating first)
            ratingsArray.sort(function (a, b) {
                return b[1] - a[1];
            });

            // Start building the HTML string for a table
            //alert('Der mittlere quadratische Fehler (Mean Squared Error, MSE) ist eine Metrik zur Beurteilung der Qualität eines Regressionsmodells. MSE=n1​∑i=1n​(yi​−y^​i​)2 = ' + jsonResponse.mse + ' city_score ' + jsonResponse.city_score)
            console.log(jsonResponse.city_score)
            console.log(jsonResponse)
            var htmlContent = '<h2>Entscheidungsinformationen</h2>' 
            htmlContent += '<table class="table">'; // Add border for visibility

            // Add table headers
            htmlContent += '<tr><th>Plaz</th><th>ID</th><th>Stadt</th><th>Bewertung</th><th>Hotelkosten</th><th>Ticketkosten</th><th>Fahrtkosten</th><th>Gesamtkosten</th><th>KM</th></tr>';
            var startCity = data.Select_start 
            var dista_in_km = 0 
            // Loop through the sorted array and add rows to the table
            for (let i = 0; i < ratingsArray.length; i++) {
                var city = ratingsArray[i][0];
                var cityData = cityRatings[city]; // Access the city data
                dista_in_km += cityData.distance_km
                htmlContent += '<tr>';
                htmlContent += '<td>' + (i + 1) + '</td>'; // Count, i starts from 0, hence (i + 1).
                htmlContent += '<td>' + cityData.id + '</td>'; // City ID
                if (startCity == city)
                    htmlContent += '<td class ="text-success bold"> Start ' + city + '</td>'; // City Name
                else
                    htmlContent += '<td>' + city + '</td>'; // City Name
                htmlContent += '<td>' + cityData.rating.toFixed(2) +' <br />'+ createStars(cityData.rating.toFixed(0)) + '</td>'; // Rating
                htmlContent += '<td>' + cityData.hotel_cost + ' €</td>';
                htmlContent += '<td>' + cityData.ticket_cost + ' €</td>';
                if (cityData.driving_cost != 0.0) {
                    htmlContent += '<td>' + cityData.driving_cost.toFixed(2) + ' €</td>';
                    if (startCity == city) {
                        var cost = cityData.driving_cost
                    } else {
                        var cost = cityData.hotel_cost + cityData.ticket_cost + cityData.driving_cost
                    }
                    htmlContent += '<td>' + cost.toFixed(2) + ' €</td>';
                    htmlContent += '<td>' + cityData.distance_km.toFixed(2) + ' Km</td>';
                } else {
                    htmlContent += '<td>-€</td>';
                    htmlContent += '<td>-€</td>';
                    htmlContent += '<td>-</td>';
                }

                htmlContent += '</tr>';
            }

            // Close the table tag
            htmlContent += '</table>';
            mymap.innerHTML += '<h3 class = "alert">Gesamtdistanz = ' + dista_in_km.toFixed(2) + ' Km  </h3>';
            alert(dista_in_km.toFixed(2) + 'km')

            // Assuming 'entscheidungContentElement' is a valid DOM element.
            entscheidungContentElement.innerHTML = htmlContent;
            entscheidungContentElement.style.visibility = 'visible';
        }
    };

    // Get the form element by its ID
    var formElement = document.getElementById('preferenceForm');
    var formData = new FormData(formElement);

    // Construct the data object with the form values
    var data = {
        'Person_Budget': formData.get('Person_Budget'),
        'Select_start': formData.get('select_start'),
        'Tage': formData.get('tage'),
        //'Person_Max_Distanz': formData.get('Person_Max_Distanz'),
        'Person_Entertainment_Fussballfan': formData.get('Person_Entertainment_Fussballfan'),
        'Person_Traditionsfussballfan': formData.get('Person_Traditionsfussballfan'),
        'Person_Schnaeppchenjaeger': formData.get('Person_Schnaeppchenjaeger'),
        'Partygaenger': formData.get('Partygaenger'),
        'Gewicht': formData.get('bewertung_gewicht'),
        'Preis_hoch': formData.get('preis_hoch'),
    };

    // Send the JSON string to the server
    xhr.send(JSON.stringify(data));
}
function createStars(rating) {
    // Definieren Sie die maximale Anzahl von Sternen
    const maxStars = 5;
    let starsHTML = '';

    // Fügen Sie gefüllte Sterne hinzu, basierend auf der Bewertung
    for (let i = 0; i < rating; i++) {
        starsHTML += '&#9733;'; // Gefüllter Stern
    }

    // Fügen Sie leere Sterne hinzu, um die maximale Anzahl zu erreichen
    for (let i = rating; i < maxStars; i++) {
        starsHTML += '&#9734;'; // Leerstern
    }

    return starsHTML;
}