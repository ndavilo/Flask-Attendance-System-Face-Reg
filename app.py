import os
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_from_directory
import cv2
import face_utils
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')

# Initialize face recognition
real_time_pred = face_utils.RealTimePrediction()
staff_df = face_utils.retrive_data(name='staff:register')

def generate_frames():
    """Generate video frames with face recognition."""
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                frame = real_time_pred.face_prediction(
                    test_image=frame,
                    dataframe=staff_df,
                    feature_column='Facial_features',
                    thresh=0.5
                )
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                             'EFCC1.png', mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clock_in', methods=['POST'])
def clock_in():
    """Handle clock-in requests with location data."""
    try:
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        
        if not latitude or not longitude:
            return jsonify({
                'status': 'error',
                'message': 'Clock In failed',
                'reason': 'Location data missing'
            }), 400

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
                'reason': 'Unknown user - Face not recognized'
            }), 401

        if not real_time_pred.check_last_action(name, 'Clock_In'):
            return jsonify({
                'status': 'error',
                'message': 'Clock In failed',
                'reason': f'{name} already clocked in today without clocking out first'
            }), 409

        # Add location to all entries
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
                    'address': currentLocation.address if 'currentLocation' in globals() else 'Location not resolved'
                }
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Clock In failed',
            'reason': str(e)
        }), 500

@app.route('/clock_out', methods=['POST'])
def clock_out():
    """Handle clock-out requests with location data."""
    try:
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        
        if not latitude or not longitude:
            return jsonify({
                'status': 'error',
                'message': 'Clock Out failed',
                'reason': 'Location data missing'
            }), 400

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
                'reason': 'Unknown user - Face not recognized'
            }), 401

        if not real_time_pred.check_last_action(name, 'Clock_Out'):
            return jsonify({
                'status': 'error',
                'message': 'Clock Out failed',
                'reason': f'{name} must clock in first before clocking out'
            }), 409

        # Add location to all entries
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
                    'address': currentLocation.address if 'currentLocation' in globals() else 'Location not resolved'
                }
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Clock Out failed',
            'reason': str(e)
        }), 500

@app.route('/get_location', methods=['POST'])
def get_location():
    """Handle location resolution from coordinates."""
    try:
        data = request.json
        if not data or 'lat' not in data or 'lon' not in data:
            return jsonify({'error': 'Invalid location data'}), 400

        latitude = float(data['lat'])
        longitude = float(data['lon'])

        # Here you would typically call a geocoding service
        # For now, we'll just return the coordinates
        return jsonify({
            'status': 'success',
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'address': f"Approximate location: {latitude:.6f}, {longitude:.6f}"
            }
        })

    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Invalid coordinates',
            'reason': 'Latitude and longitude must be valid numbers'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Location resolution failed',
            'reason': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)