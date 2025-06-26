function getBathValue() {
  var uiBathrooms = document.getElementsByName("uiBathrooms");
  for(var i in uiBathrooms) {
    if(uiBathrooms[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function getBedroomsValue() {
  var bd = document.getElementsByName("uiBEDR");
  for(var i in bd) {
    if(bd[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function onClickedEstimatePrice() {
  console.log("predicted price button clicked");
  var sqft = document.getElementById("uiSqft");
  var bedrooms= getBedroomsValue();
  var bathrooms = getBathValue();
  var location = document.getElementById("uiLocations");
  var estPrice = document.getElementById("uiEstimatedPrice");

  var url = "https://shiny-carnival-j7qp574wpr525gj7-5000.app.github.dev/predict_home_price"; //Use this if you are NOT using nginx
  //var url = "/api/predict_home_price"; // Use this if  you are using nginx

  $.post(url, {
      total_sqft: parseFloat(sqft.value),
      bedrooms: bedrooms,
      bath: bathrooms,
      location: location.value
  },function(data, status) {
<<<<<<< HEAD
      console.log(data.predicted_price);
      estPrice.innerHTML = "<h2>" + data.predicted_price.toString() + " kwacha</h2>";
=======
      console.log(data.predited_price);
      estPrice.innerHTML = "<h2>" + data.predited_price.toString() + " kwacha</h2>";
>>>>>>> e20c46ce4a45568213ce225f2a0543adff26c98e
      console.log(status);
  });
}

function onPageLoad() {
  console.log( "document loaded" );
  var url = "https://shiny-carnival-j7qp574wpr525gj7-5000.app.github.dev/get_loc_data";// Use this if you are NOT using nginx which is first 7 tutorials
  
  $.get(url,function(data, status) {
      console.log("got response for get_location_names request");
      if(data) {
          var locations = data.locations;
          var uiLocations = document.getElementById("uiLocations");
          $('#uiLocations').empty();
          for(var i in locations) {
              var opt = new Option(locations[i]);
              $('#uiLocations').append(opt);
          }
      }
  });
}

window.onload = onPageLoad;
