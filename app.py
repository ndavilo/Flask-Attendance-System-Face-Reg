import os
import json
import base64
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import face_utils
from datetime import datetime
from dotenv import load_dotenv
from geopy.distance import geodesic
import logging

load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load allowed locations from JSON file
def load_locations():
    try:
        with open('allowedLocations.json', 'r') as f:
            data = json.load(f)
            return data.get('locations', {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading locations: {str(e)}")
        return {}

ALLOWED_LOCATIONS = load_locations()

# Initialize face recognition
try:
    real_time_pred = face_utils.RealTimePrediction()
    staff_df = face_utils.retrive_data(name='staff:register')
    logger.info("Face recognition initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize face recognition: {str(e)}")
    raise

def verify_location(browser_lat, browser_lon):
    """Verify if coordinates are within any allowed location"""
    if not ALLOWED_LOCATIONS:
        return False, "No allowed locations configured"
    
    for loc_name, loc_data in ALLOWED_LOCATIONS.items():
        try:
            distance = geodesic(
                (browser_lat, browser_lon),
                (loc_data['lat'], loc_data['lon'])
            ).km
            
            if distance <= loc_data['radius_km']:
                return True, f"Verified in {loc_name} ({(distance*1000):.0f}m from center)"
        except KeyError as e:
            logger.warning(f"Invalid location data for {loc_name}: {str(e)}")
            continue
    
    return False, "Location not in any allowed area"

def get_nearest_location_name(lat, lon):
    """Get the name of the nearest allowed location"""
    if not ALLOWED_LOCATIONS:
        return "Unknown location"
    
    nearest = None
    min_distance = float('inf')
    
    for loc_name, loc_data in ALLOWED_LOCATIONS.items():
        try:
            distance = geodesic((lat, lon), (loc_data['lat'], loc_data['lon'])).km
            if distance < min_distance:
                min_distance = distance
                nearest = loc_name
        except KeyError:
            continue
    
    if nearest:
        return f"Near {nearest} ({(min_distance*1000):.0f}m)"
    return "Unknown location"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                             'EFCC1.png', mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Placeholder endpoint for compatibility"""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', blank_frame)
    return Response(
        b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n',
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/reload_locations', methods=['POST'])
def reload_locations():
    """Reload locations from file"""
    global ALLOWED_LOCATIONS
    ALLOWED_LOCATIONS = load_locations()
    return jsonify({
        'status': 'success',
        'message': f'Reloaded {len(ALLOWED_LOCATIONS)} locations'
    })

@app.route('/clock_in', methods=['POST'])
def clock_in():
    """Handle clock-in with browser-captured image"""
    try:
        data = request.json
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
        image_data = data.get('image')

        if not image_data:
            return jsonify({
                'status': 'error',
                'message': 'No image provided',
                'reason': 'Camera frame missing'
            }), 400

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                'status': 'error',
                'message': 'Invalid image data',
                'reason': 'Failed to decode image'
            }), 400

        # Verify location
        is_valid, reason = verify_location(latitude, longitude)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': 'Clock In blocked',
                'reason': reason
            }), 403

        # Process face recognition
        frame = real_time_pred.face_prediction(
            test_image=frame,
            dataframe=staff_df,
            feature_column='Facial_features',
            thresh=0.5
        )

        logs = real_time_pred.logs
        if not logs.get('name'):
            return jsonify({
                'status': 'error',
                'message': 'Clock In failed',
                'reason': 'No face detected'
            }), 400

        name = logs['name'][0]
        if name == 'Unknown':
            return jsonify({
                'status': 'error',
                'message': 'Clock In failed',
                'reason': 'Unknown user'
            }), 401

        if not real_time_pred.check_last_action(name, 'Clock_In'):
            return jsonify({
                'status': 'error',
                'message': 'Clock In failed',
                'reason': f'{name} already clocked in'
            }), 409

        # Save logs with location
        logs['latitude'] = [latitude] * len(logs['name'])
        logs['longitude'] = [longitude] * len(logs['name'])
        real_time_pred.saveLogs_redis(Clock_In_Out='Clock_In')
        
        return jsonify({
            'status': 'success',
            'message': 'Clock In successful',
            'data': {
                'name': name,
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'address': get_nearest_location_name(latitude, longitude)
                }
            }
        })

    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': 'Invalid coordinates',
            'reason': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Clock In failed',
            'reason': str(e)
        }), 500

@app.route('/clock_out', methods=['POST'])
def clock_out():
    """Handle clock-out with browser-captured image"""
    try:
        data = request.json
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
        image_data = data.get('image')

        if not image_data:
            return jsonify({
                'status': 'error',
                'message': 'No image provided',
                'reason': 'Camera frame missing'
            }), 400

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                'status': 'error',
                'message': 'Invalid image data',
                'reason': 'Failed to decode image'
            }), 400

        # Verify location
        is_valid, reason = verify_location(latitude, longitude)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': 'Clock Out blocked',
                'reason': reason
            }), 403

        # Process face recognition
        frame = real_time_pred.face_prediction(
            test_image=frame,
            dataframe=staff_df,
            feature_column='Facial_features',
            thresh=0.5
        )

        logs = real_time_pred.logs
        if not logs.get('name'):
            return jsonify({
                'status': 'error',
                'message': 'Clock Out failed',
                'reason': 'No face detected'
            }), 400

        name = logs['name'][0]
        if name == 'Unknown':
            return jsonify({
                'status': 'error',
                'message': 'Clock Out failed',
                'reason': 'Unknown user'
            }), 401

        if not real_time_pred.check_last_action(name, 'Clock_Out'):
            return jsonify({
                'status': 'error',
                'message': 'Clock Out failed',
                'reason': f'{name} must clock in first'
            }), 409

        # Save logs with location
        logs['latitude'] = [latitude] * len(logs['name'])
        logs['longitude'] = [longitude] * len(logs['name'])
        real_time_pred.saveLogs_redis(Clock_In_Out='Clock_Out')
        
        return jsonify({
            'status': 'success',
            'message': 'Clock Out successful',
            'data': {
                'name': name,
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'address': get_nearest_location_name(latitude, longitude)
                }
            }
        })

    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': 'Invalid coordinates',
            'reason': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Clock Out failed',
            'reason': str(e)
        }), 500

@app.route('/get_location', methods=['POST'])
def get_location():
    """Resolve coordinates to location"""
    try:
        data = request.json
        latitude = float(data.get('lat'))
        longitude = float(data.get('lon'))

        return jsonify({
            'status': 'success',
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'address': get_nearest_location_name(latitude, longitude)
            }
        })

    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Invalid coordinates',
            'reason': 'Latitude and longitude must be numbers'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Location resolution failed',
            'reason': str(e)
        }), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv('DEBUG', 'False') == 'True',
        threaded=True
    )