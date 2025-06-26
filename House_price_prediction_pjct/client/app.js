function getBathValue() {
    let uiBathrooms = document.getElementsByName("uiBathrooms");
    for (let i in uiBathrooms) {
        if (uiBathrooms[i].checked) {
            // CORRECTED: Get the actual 'value' attribute of the checked radio button
            return parseInt(uiBathrooms[i].value);
        }
    }
    // Consider returning a default value (e.g., 2) or handling this case specifically
    return -1; // Invalid Value if no bathroom is selected
}

// Function to get the currently selected number of bedrooms
function getBedroomsValue() {
    let bd = document.getElementsByName("uiBEDR");
    for (let i in bd) {
        if (bd[i].checked) {
            // CORRECTED: Get the actual 'value' attribute of the checked radio button
            return parseInt(bd[i].value);
        }
    }
    // Consider returning a default value (e.g., 2) or handling this case specifically
    return -1; // Invalid Value if no bedroom is selected
}

// Function triggered when the "Estimate Price" button is clicked
function onClickedEstimatePrice() {
    console.log("Estimate Price button clicked");

    let sqft = document.getElementById("uiSqft");
    let bedrooms = getBedroomsValue();
    let bathrooms = getBathValue();
    let location = document.getElementById("uiLocations");
    let estPrice = document.getElementById("uiEstimatedPrice");

    // IMPORTANT: Ensure this URL is correct for your deployed Shiny-Carnival app
    let url = "https://shiny-carnival-j7qp574wpr525gj7.github.dev/predict_home_price";

    $.post(url, {
        total_sqft: parseFloat(sqft.value),
        bedrooms: bedrooms, // This key name 'bedrooms' should match your Flask app's request.form["bedrooms"]
        bath: bathrooms,
        location: location.value
    }, function(data, status) {
        // Callback function when the request is successful
        console.log("Prediction data received:", data);
        console.log("Status:", status);

        // Check if data.predicted_price exists and is a valid number
        if (data && typeof data.predicted_price !== 'undefined' && !isNaN(data.predicted_price)) {
            // Update the HTML element to display the predicted price
            estPrice.innerHTML = "<h2>" + data.predicted_price.toString() + " kwacha</h2>";
        } else {
            // Handle cases where the prediction might be missing or invalid
            estPrice.innerHTML = "<h2>Error: Could not get estimated price.</h2>";
            console.error("Invalid prediction data:", data);
        }
    }).fail(function(jqXHR, textStatus, errorThrown) {
        // Error handling for the AJAX request
        console.error("AJAX Error during prediction:", textStatus, errorThrown);
        console.error("Response Text:", jqXHR.responseText);
        estPrice.innerHTML = "<h2>Error: Failed to connect to prediction service or invalid input.</h2>";
    });
}

// Function to load location names into the dropdown when the page loads
function onPageLoad() {
    console.log("document loaded: fetching locations...");

    // IMPORTANT: Ensure this URL is correct for your deployed Shiny-Carnival app
    let url = "https://shiny-carnival-j7qp574wpr525gj7.github.dev/get_loc_data";

    $.get(url, function(data, status) {
        console.log("Got response for get_location_names request");
        if (data && data.locations) { // Check if data and data.locations exist
            let locations = data.locations;
            let uiLocations = document.getElementById("uiLocations");

            // Clear existing options
            $(uiLocations).empty(); // Use jQuery for simpler empty()

            // Add back the default "Choose a Location" option
            $(uiLocations).append(new Option("Choose a Location", "", true, true));

            // Populate the dropdown with locations received from the backend
            for (let i = 0; i < locations.length; i++) { // Use standard for loop for array iteration
                let opt = new Option(locations[i]);
                $(uiLocations).append(opt);
            }
        } else {
            console.warn("No locations data received or data format is unexpected.");
            // Optionally display an error to the user if locations can't be loaded
            let uiLocations = document.getElementById("uiLocations");
            $(uiLocations).empty();
            $(uiLocations).append(new Option("Error loading locations", "", true, true));
        }
    }).fail(function(jqXHR, textStatus, errorThrown) {
        console.error("AJAX Error fetching locations:", textStatus, errorThrown);
        console.error("Response Text:", jqXHR.responseText);
        let uiLocations = document.getElementById("uiLocations");
        $(uiLocations).empty();
        $(uiLocations).append(new Option("Error loading locations", "", true, true));
    });
}

// Ensure onPageLoad is called when the DOM is ready using jQuery
$(document).ready(function() {
    onPageLoad();
});
