function sendPreferenceFormData() {
    var xhr = new XMLHttpRequest();

    xhr.open("POST", '/post_preference_json', true);
    xhr.setRequestHeader("Content-Type", "application/json");
    var entscheidungContentElement = document.getElementById('entscheidung_content');
    var mymap = document.getElementById('mapdiv');
    //var feedback_div = document.getElementById('feedback_div');
    entscheidungContentElement.innerHTML = '';

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var jsonResponse = JSON.parse(xhr.responseText);
            var cityRatings = jsonResponse.city_with_rating;
            // Angenommen, Sie haben ein Objekt mit Städten als Schlüsseln und Scores als Werten.
            var cityScores = jsonResponse.city_score
            console.log(cityScores)
            mymap.style.visibility = 'visible';
            //feedback_div.style.visibility = 'visible';

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
                    htmlContent += '<td class ="text-success bold"> Start von ' + city + '</td>'; // City Name
                else
                    htmlContent += '<td>' + city + '</td>'; // City Name
                htmlContent += '<td>' + cityData.rating.toFixed(2) +' <br />'+ createStars(cityData.rating.toFixed(2)) + '</td>'; // Rating
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
                    htmlContent += '<td>-</td>';
                    htmlContent += '<td>-</td>';
                    htmlContent += '<td>-</td>';
                }

                htmlContent += '</tr>';
            }

            // Close the table tag
            htmlContent += '</table>';
            mymap.innerHTML = `
                          <h3>
                            Gesamtkosten ist ${jsonResponse.total_price} €,
                            Reisebewertung ist ${createStars(jsonResponse.average_rating.toFixed(2))} ${jsonResponse.average_rating.toFixed(2)}, 
                            Gesamte Distanz ist ${dista_in_km.toFixed(2) } Km
                            Dauer ${jsonResponse.tage } Tage
                          </h4>
                <p> Route => ${ jsonResponse.route}</p>`;

            mymap.innerHTML +=`
            <a href="${jsonResponse.google_map_route}/target="_blank" rel="noopener noreferrer">Google Map</a>
            `;
            mymap.innerHTML += jsonResponse.google_map_route;
            mymap.innerHTML += jsonResponse.m_html;

            // Assuming 'entscheidungContentElement' is a valid DOM element.
            entscheidungContentElement.innerHTML = htmlContent;
            entscheidungContentElement.style.visibility = 'visible';
        }
    };

    // Get the form element by its ID
    var formData = new FormData(preferenceForm);
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
    };
    // Send the JSON string to the server
    xhr.send(JSON.stringify(data));
}
function createStars(rating) {
    // Definieren Sie die maximale Anzahl von Sternen und die HTML für die Sterne
    const maxStars = 5;
    let starsHTML = '';

    // Ganze gefüllte Sterne hinzufügen
    for (let i = 0; i < Math.floor(rating); i++) {
        starsHTML += '<span style="color: gold;">&#9733;</span>'; // Gefüllter Stern in Goldfarbe
    }

    // Einen teilweise gefüllten Stern hinzufügen, wenn es einen Dezimalanteil gibt
    if (rating % 1 !== 0) {
        const widthPercent = Math.round((rating % 1) * 100); // Prozent des teilweise gefüllten Sterns
        starsHTML += `<span style="display: inline-block; position: relative; color: #ccc;">&#9733;`; // Leerstern in hellem Grau
        starsHTML += `<span style="position: absolute; left: 0; top: 0; width: ${widthPercent}%; overflow: hidden; color: gold;">&#9733;</span>`; // Überlagerung mit teilweise gefülltem Stern in Goldfarbe
        starsHTML += `</span>`;
    }

    // Leere Sterne hinzufügen, um die maximale Anzahl zu erreichen
    const totalStarsDisplayed = Math.ceil(rating);
    for (let i = totalStarsDisplayed; i < maxStars; i++) {
        starsHTML += '<span style="color: #ccc;">&#9734;</span>'; // Leerstern in hellem Grau
    }

    return starsHTML;
}