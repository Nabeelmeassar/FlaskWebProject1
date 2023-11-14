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
                'distance': jsonArray.distance,
                'start': jsonArray.select_start,
                'ziel': jsonArray.select_ziel,
                'max_distance': jsonArray.max_distance,
                'budget': jsonArray.budget,
                'bevorzugtes_wetter': jsonArray.bevorzugtes_wetter
            };

            console.log(processedData); // Output the processed data to the console
            // Use 'processedData' in your application as needed
            // Convert 'processedData' to a string and set it as the innerHTML of the element with the ID 'entscheidung_content'
            // Get the element with the ID 'entscheidung_content'
            var entscheidungContentElement = document.getElementById('entscheidung_content');
            entscheidungContentElement.style.visibility = 'visible';
            entscheidungContentElement.innerHTML = '';
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