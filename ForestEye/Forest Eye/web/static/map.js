document.addEventListener('DOMContentLoaded', function () {
    var mymap = L.map('map', {
        center: [-6.235, -51.9253], // Centered on South America
        zoom: 5, // Zoom level
        fullscreenControl: true // Enable fullscreen control
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(mymap);

    var markers = [
        { name: 'Point 1', latitude: -9.598418379, longitude: -52.03797437 },
        { name: 'Point 2', latitude: -3.71999392, longitude: -56.948192 },
        
        { name: 'Point 3', latitude: -2.402538472, longitude: -48.66149455 },
        { name: 'Point 4', latitude: -1.219139101, longitude:-55.74756095 },
        { name: 'Point 5', latitude: -1.563140788, longitude: -48.07568556 },
        { name: 'Point 6', latitude: -6.589639966, longitude: -54.90057544 },
    ];

    markers.forEach(function (marker) {
        var popupContent = `
            <b>${marker.name}</b><br>
            Latitude: ${marker.latitude}<br>
            Longitude: ${marker.longitude}<br>
            <a href="/predict?lat=${marker.latitude}&lon=${marker.longitude}">Predict</a>
        `;
        L.marker([marker.latitude, marker.longitude]).addTo(mymap)
            .bindPopup(popupContent);
    });
});