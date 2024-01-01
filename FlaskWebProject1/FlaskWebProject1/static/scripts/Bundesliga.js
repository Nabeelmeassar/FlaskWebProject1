function sendFormData() {
    var xhr = new XMLHttpRequest();

    xhr.open("POST", '/postjson', true);
    xhr.setRequestHeader("Content-Type", "application/json");
    var formElement = document.getElementById('myForm');

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var jsonArray = JSON.parse(xhr.responseText);
            console.log(jsonArray); // Prints the received JSON data to the console

            // Here you can process the data and use it in your application
            var processedData = {
                'Startstadt': jsonArray.start,
                'Zielstadt': jsonArray.ziel,
                'Max Entfernung ': jsonArray.max_distance + ' Km',
                'budget': jsonArray.budget,
                'Kuerzeste Entfernung in km': jsonArray.Kuerzeste_Entfernung_km + ' Km',
                'Kuerzester Weg': jsonArray.Kuerzester_Weg,
            };

            console.log(processedData); // Output the processed data to the console
            // Use 'processedData' in your application as needed
            // Convert 'processedData' to a string and set it as the innerHTML of the element with the ID 'entscheidung_content'
            // Get the element with the ID 'entscheidung_content'
            var entscheidungContentElement = document.getElementById('entscheidung_content');
            var mymap = document.getElementById('mapdiv');

            entscheidungContentElement.style.visibility = 'visible';
            entscheidungContentElement.innerHTML = '';
            mymap.innerHTML = jsonArray.m_html;
            // Go through all properties in 'processedData'
            Object.keys(processedData).forEach(function (key) {
                // Create a new <p> element
                var newParagraph = document.createElement('p');

                // Set the content of the new paragraph
                newParagraph.textContent = key + ': ' + processedData[key];

                // Append the new paragraph to the 'entscheidung_content' element
                entscheidungContentElement.appendChild(newParagraph);
            });
        }
    };

    var formData = new FormData(formElement); // Assuming 'formElement' is a reference to your form

    var data = {
        'select_start': formData.get('select_start'),
        'select_ziel': formData.get('select_ziel'),
        'wochentag': formData.get('wochentag'),
        'bevorzugtes_wetter': formData.get('bevorzugtes_wetter'),
        'max_distance': formData.get('max_distance'),
        'budget': formData.get('budget')
    };

    xhr.send(JSON.stringify(data));
}
function sendPreferenceFormData() {
    var xhr = new XMLHttpRequest();

    xhr.open("POST", '/post_preference_json', true);
    xhr.setRequestHeader("Content-Type", "application/json");
    var entscheidungContentElement = document.getElementById('entscheidung_content');
    var mymap = document.getElementById('mapdiv');
    var city_score_div = document.getElementById('city_score');
    entscheidungContentElement.style.visibility = 'visible';
    entscheidungContentElement.innerHTML = '';

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var jsonResponse = JSON.parse(xhr.responseText);
            var cityRatings = jsonResponse.city_with_rating;
            var html = '<ul>'; // Starten Sie mit einer ungeordneten Liste
            for (var city in jsonResponse.city_score) {
                if (jsonResponse.city_score.hasOwnProperty(city)) {
                    var score = jsonResponse.city_score[city].score;
                    html += '<li>' + city + ': ' + score + '</li>'; // Fügen Sie jedes Stadt-Score-Paar als Listenelement hinzu
                }
            }
            html += '</ul>';
            mymap.innerHTML = jsonResponse.route; 
            mymap.innerHTML += jsonResponse.m_html; // Assuming 'mymap' is a valid DOM element.
            mymap.innerHTML += 'Die Funktion `calculate_score` berechnet einen Punktwert für eine Stadt. Dieser Wert hängt davon ab, wie gut die Stadt bewertet ist und wie weit sie von einer anderen Stadt entfernt ist. Je besser die Bewertung und je näher die Stadt, desto höher der Punktwert.'; 
            mymap.innerHTML += html; 

            // Convert object to an array of [city, rating] pairs
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
            alert('Der mittlere quadratische Fehler (Mean Squared Error, MSE) ist eine Metrik zur Beurteilung der Qualität eines Regressionsmodells. MSE=n1​∑i=1n​(yi​−y^​i​)2 = ' + jsonResponse.mse + ' city_score ' + jsonResponse.city_score)
            console.log(jsonResponse.city_score)
            var htmlContent = '' 
            htmlContent += '<table class="table">'; // Add border for visibility

            // Add table headers
            htmlContent += '<tr><th>Plaz</th><th>ID</th><th>Stadt</th><th>Bewertung</th><th>GPS</th></tr>';

            // Loop through the sorted array and add rows to the table
            for (let i = 0; i < ratingsArray.length; i++) {
                var city = ratingsArray[i][0];
                var cityData = cityRatings[city]; // Access the city data
                htmlContent += '<tr>';
                htmlContent += '<td>' + (i + 1) + '</td>'; // Count, i starts from 0, hence (i + 1).
                htmlContent += '<td>' + cityData.ID + '</td>'; // City ID
                htmlContent += '<td>' + city + '</td>'; // City Name
                htmlContent += '<td>' + cityData.rating.toFixed(2) + '</td>'; // Rating
                htmlContent += '<td>' + cityData.GPS + '</td>'; // GPS Coordinates, changed 'club_coordinate' to 'gps'
                //htmlContent += '<td>' + jsonResponse.city_score[city].score.toFixed(2) + '</td>';
                htmlContent += '</tr>';
            }

            // Close the table tag
            htmlContent += '</table>';

            // Assuming 'entscheidungContentElement' is a valid DOM element.
            entscheidungContentElement.innerHTML = htmlContent;
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
        'Partygaenger': formData.get('Partygaenger')
    };

    // Send the JSON string to the server
    xhr.send(JSON.stringify(data));
}