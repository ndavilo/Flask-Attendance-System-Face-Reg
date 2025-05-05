let currentLocation = null;
let locationVerified = false;
let cameraActive = false;

const videoElement = document.getElementById('videoElement');
const canvasElement = document.getElementById('canvasElement');
const ctx = canvasElement.getContext('2d');
const locationResult = document.getElementById('locationResult');
const locationStatus = document.getElementById('locationStatus');
const cameraStatus = document.getElementById('cameraStatus');
const clockInBtn = document.getElementById('clockInBtn');
const clockOutBtn = document.getElementById('clockOutBtn');

async function startCamera() {
    try {
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            }, 
            audio: false
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        
        // Adjust video element size based on container
        const resizeVideo = () => {
            const container = videoElement.parentElement;
            const aspectRatio = 4/3; // Default to 4:3
            
            if (container.clientWidth > 0) {
                videoElement.width = container.clientWidth;
                videoElement.height = container.clientWidth / aspectRatio;
                canvasElement.width = container.clientWidth;
                canvasElement.height = container.clientWidth / aspectRatio;
            }
        };
        
        resizeVideo();
        window.addEventListener('resize', resizeVideo);
        
        cameraStatus.textContent = 'Camera active';
        cameraStatus.className = 'status-message status-success';
        cameraActive = true;
    } catch (err) {
        cameraStatus.textContent = 'Camera access denied';
        cameraStatus.className = 'status-message status-error';
        showAlert('error', 'Camera access denied - please enable permissions');
        cameraActive = false;
    } finally {
        updateButtonStates();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    startCamera();
    if (!navigator.geolocation) {
        locationResult.innerHTML = '<p>Geolocation not supported by your browser</p>';
        locationStatus.innerHTML = '<p class="status-error">Location services unavailable</p>';
        return;
    }
    fetchLocation();
});

async function fetchLocation() {
    locationResult.innerHTML = '<p>Detecting location...</p>';
    locationStatus.innerHTML = '<p class="status-warning">Verifying location...</p>';

    try {
        const position = await new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(resolve, reject, {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            });
        });

        const { latitude: lat, longitude: lon, accuracy } = position.coords;

        if (accuracy > 100) {
            locationStatus.innerHTML = '<p class="status-error">Location too inaccurate (disable VPN or move outdoors)</p>';
            locationVerified = false;
            updateButtonStates();
            return;
        }

        const response = await fetch(locationUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon })
        });

        const data = await response.json();

        if (data.status === 'success') {
            currentLocation = data.location;
            locationResult.innerHTML = `
                <p><strong>Latitude:</strong> ${data.location.latitude.toFixed(6)}</p>
                <p><strong>Longitude:</strong> ${data.location.longitude.toFixed(6)}</p>
                <p><strong>Address:</strong> ${data.location.address}</p>
                <p><strong>Accuracy:</strong> ±${Math.round(accuracy)} meters</p>
            `;
            locationStatus.innerHTML = '<p class="status-success">Location verified</p>';
            locationVerified = true;
        } else {
            locationResult.innerHTML = `<p>${data.error || 'Location verification failed'}</p>`;
            locationStatus.innerHTML = '<p class="status-error">Location not verified</p>';
            showAlert('error', 'Location verification failed: ' + (data.reason || 'Unknown error'));
        }
    } catch (error) {
        handleLocationError(error);
    } finally {
        updateButtonStates();
    }
}

function handleLocationError(error) {
    let message = "Location unavailable: ";
    switch (error.code) {
        case error.PERMISSION_DENIED:
            message = "Location access denied - please enable permissions"; break;
        case error.POSITION_UNAVAILABLE:
            message = "Location unavailable - check network connection"; break;
        case error.TIMEOUT:
            message = "Location request timed out - try again"; break;
        default:
            message = "Location detection failed";
    }

    locationResult.innerHTML = `<p>${message}</p>`;
    locationStatus.innerHTML = '<p class="status-error">Location not verified</p>';
    showAlert('error', message);
}

function updateButtonStates() {
    const enabled = locationVerified && cameraActive;
    clockInBtn.disabled = !enabled;
    clockOutBtn.disabled = !enabled;
}

async function submitWithLocation(actionUrl) {
    if (!locationVerified || !cameraActive) {
        showAlert('error', !locationVerified ? 'Location not verified - cannot proceed' : 'Camera not available - cannot proceed');
        return;
    }

    ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    const imageData = canvasElement.toDataURL('image/jpeg', 0.7).split(',')[1];

    try {
        const response = await fetch(actionUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                latitude: currentLocation.latitude,
                longitude: currentLocation.longitude,
                image: imageData
            })
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.reason || 'Unknown error occurred');

        if (data.status === 'success') {
            showAlert('success', `${data.message} - ${data.data.name} at ${data.data.time}`);
            setTimeout(fetchLocation, 2000);
        } else {
            showAlert('error', `${data.message}: ${data.reason}`);
        }
    } catch (error) {
        showAlert('error', `Request failed: ${error.message}`);
    }
}

function showAlert(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `status-message status-${type}`;
    alertDiv.textContent = message;

    const statusContainer = document.getElementById('statusMessages');
    statusContainer.innerHTML = '';
    statusContainer.appendChild(alertDiv);

    setTimeout(() => alertDiv.remove(), 5000);
}