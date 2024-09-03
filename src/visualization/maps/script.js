// Create Leaflet map instance with custom tile layers
const map = L.map('map').setView([0, 0], 2); // Centered at [0, 0] with zoom level 2

// const map = L.map('map', {
//     center: [0, 0], // Centered at [0, 0]
//     zoom: 3, // Initial zoom level
//     layers: [osmLayer] // Initial tile layer
// });

// Add OpenStreetMap tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Sample data for scatterplot
// const data = [
//     { name: "City1", coordinates: [10, 20] },
//     { name: "City2", coordinates: [30, 40] },
//     // Add more data points as needed
// ];

// // Draw scatterplot
// data.forEach(d => {
//     L.circle(d.coordinates, {
//         color: 'red',
//         fillColor: '#f03',
//         fillOpacity: 0.8,
//         radius: 50000 // Adjust radius as needed
//     }).addTo(map).bindPopup(d.name); // Add a popup with the city name
// });

// Create a div element to display coordinates
const coordinatesBox = L.control({position: 'topright'});

coordinatesBox.onAdd = function (map) {
    this._div = L.DomUtil.create('div', 'coordinates-box');
    this.update();
    return this._div;
};

coordinatesBox.update = function (latlng) {
    this._div.innerHTML = latlng ? `Latitude: ${latlng.lat.toFixed(4)}<br>Longitude: ${latlng.lng.toFixed(4)}` : 'Hover over the map';
};

coordinatesBox.addTo(map);

d3.csv("http://localhost:8080/data_20201231.csv").then(function(data) {
    console.log("CSV data:", data);
}).catch(function(error) {
    console.error("Error loading CSV:", error);
});

// Load CSV data
// d3.csv("file://Users/z6f/code/ehr-env/data/interim/data_20201231.csv").then(function(data) {
//     // Draw scatterplot markers
//     data.forEach(function(d) {
//         const lat = parseFloat(d.Lat);
//         const lng = parseFloat(d.Lon);
//         L.circleMarker([lat, lng], {
//             radius: 5, // Adjust radius as needed
//             fillColor: "red",
//             color: "#000",
//             weight: 1,
//             opacity: 1,
//             fillOpacity: 0.8
//         }).addTo(map); // Assuming you have a "name" column in your CSV
//     });
// })
// d3.csv("/Users/z6f/code/ehr-env/data/interim/data_20201231.csv").then(function(data) {
//     // Draw scatterplot markers
//     data.forEach(function(d) {
//         const lat = parseFloat(d.Lat);
//         const lng = parseFloat(d.Lon);
//         L.circleMarker([lat, lng], {
//             radius: 5, // Adjust radius as needed
//             fillColor: "red",
//             color: "#000",
//             weight: 1,
//             opacity: 1,
//             fillOpacity: 0.8
//         }).addTo(map).bindPopup(d.name); // Assuming you have a "name" column in your CSV
//     });
// }).catch(function(error) {
//     console.log("Error loading CSV:", error);
// });

// Listen for mousemove event on the map
// map.on('mousemove', function(event) {
//     const { lat, lng } = event.latlng; // Get latitude and longitude from event
//     coordinatesBox.update({ lat, lng });
// });
