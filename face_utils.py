import numpy as np
import pandas as pd
import redis
import cv2
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class FaceRecognitionError(Exception):
    """Custom exception for face recognition errors"""
    pass

class RedisConnectionError(Exception):
    """Custom exception for Redis connection errors"""
    pass

# Redis connection configuration from environment variables
REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')

def get_redis_connection():
    """Create and return a Redis connection with error handling"""
    try:
        return redis.StrictRedis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=False,
            socket_timeout=5,
            socket_connect_timeout=5
        )
    except Exception as e:
        raise RedisConnectionError(f"Failed to connect to Redis: {str(e)}")

r = get_redis_connection()

def retrive_data(name):
    """
    Retrieve facial recognition data from Redis database and format it into a DataFrame.
    
    Args:
        name (str): Redis hash key name
        
    Returns:
        pd.DataFrame: DataFrame containing staff information
        
    Raises:
        RedisConnectionError: If there's a problem connecting to Redis
        ValueError: If the data format is invalid
    """
    try:
        retrive_dict = r.hgetall(name)
        if not retrive_dict:
            raise ValueError(f"No data found in Redis for key: {name}")

        retrive_series = pd.Series(retrive_dict)
        
        # Convert byte strings to numpy arrays
        retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
        
        # Properly decode byte keys to strings
        index = [key.decode('utf-8') for key in retrive_series.index]
        retrive_series.index = index
        
        retrive_df = retrive_series.to_frame().reset_index()
        retrive_df.columns = ['ID_Name_Role', 'Facial_features']
        
        # Initialize default values
        retrive_df['File No. Name'] = ''
        retrive_df['Role'] = ''
        retrive_df['Zone'] = 'Lagos Zone 2'  # Default zone
        
        # Process each record safely
        for i, row in retrive_df.iterrows():
            try:
                # Ensure we're working with string, not bytes
                id_name_role = str(row['ID_Name_Role'])
                parts = id_name_role.split('@')
                
                # Extract file no and name (first part before @)
                file_name_role = parts[0]
                file_no, name = file_name_role.split('.', 1)
                
                # Extract role (second part)
                role = parts[1] if len(parts) > 1 else ''
                
                # Extract zone if available (third part in new format)
                zone = parts[2] if len(parts) > 2 else 'Lagos Zone 2'
                
                # Update the row
                retrive_df.at[i, 'File No. Name'] = f"{file_no}.{name}"
                retrive_df.at[i, 'Role'] = role
                retrive_df.at[i, 'Zone'] = zone
                
            except Exception as e:
                print(f"Warning: Error processing record {row['ID_Name_Role']}: {str(e)}")
                continue
        
        return retrive_df[['ID_Name_Role', 'File No. Name', 'Role', 'Facial_features', 'Zone']]
    
    except Exception as e:
        raise RedisConnectionError(f"Error retrieving data from Redis: {str(e)}")

def load_logs(name, end=-1):
    """
    Load logs from Redis list.
    
    Args:
        name (str): Redis list key (e.g., 'attendance:logs')
        end (int): Index of last item to retrieve (default -1 for all items)
        
    Returns:
        list: List of decoded log entries
        
    Raises:
        RedisConnectionError: If there's a problem connecting to Redis
    """
    try:
        logs_list = r.lrange(name, start=0, end=end)
        return [log.decode('utf-8') if isinstance(log, bytes) else log for log in logs_list]
    except Exception as e:
        raise RedisConnectionError(f"Error loading logs from Redis: {str(e)}")

# Configure face analysis model
try:
    faceapp = FaceAnalysis(name='buffalo_sc',
                         root='insightface_model',
                         providers=['CPUExecutionProvider'])
    faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
except Exception as e:
    raise FaceRecognitionError(f"Failed to initialize face recognition model: {str(e)}")

def ml_search_algorithm(dataframe, feature_column, test_vector, name_role=['File No. Name', 'Role'], thresh=0.5):
    """
    Perform facial recognition search using cosine similarity.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing staff facial features
        feature_column (str): Name of column containing facial embeddings
        test_vector (np.array): Facial embedding to compare against
        name_role (list): Column names for name and role in dataframe
        thresh (float): Similarity threshold for positive match (0-1)
        
    Returns:
        tuple: (matched_name, matched_role) or ('Unknown', 'Unknown') if no match
        
    Raises:
        ValueError: If input data is invalid
    """
    if dataframe.empty or feature_column not in dataframe.columns:
        return 'Unknown', 'Unknown'

    dataframe = dataframe.copy()
    X_list = dataframe[feature_column].tolist()
    X_cleaned = []
    indices = []

    # Clean and validate input vectors
    for i, item in enumerate(X_list):
        if isinstance(item, (list, np.ndarray)) and len(item) > 0:
            item_arr = np.array(item)
            if item_arr.shape == test_vector.shape:
                X_cleaned.append(item_arr)
                indices.append(i)

    if len(X_cleaned) == 0:
        return 'Unknown', 'Unknown'

    dataframe = dataframe.iloc[indices].reset_index(drop=True)
    x = np.array(X_cleaned)
    similar = cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = similar.flatten()
    dataframe['cosine'] = similar_arr
    data_filter = dataframe[dataframe['cosine'] >= thresh]

    if not data_filter.empty:
        best_match = data_filter.sort_values(by='cosine', ascending=False).iloc[0]
        person_name, person_role = best_match[name_role[0]], best_match[name_role[1]]
    else:
        person_name, person_role = 'Unknown', 'Unknown'

    return person_name, person_role

class RealTimePrediction:
    """
    Class for handling real-time facial recognition predictions and attendance logging.
    """
    
    def __init__(self):
        """Initialize with empty logs dictionary"""
        self.logs = dict(name=[], role=[], current_time=[])
    
    def reset_dict(self):
        """Reset the logs dictionary to empty state"""
        self.logs = dict(name=[], role=[], current_time=[])

    def check_last_action(self, name, current_action):
        """
        Check if the current action is valid based on previous logs.
        
        Args:
            name (str): Staff name to check
            current_action (str): Either 'Clock_In' or 'Clock_Out'
            
        Returns:
            bool: True if action is valid, False otherwise
        """
        try:
            if name == 'Unknown':
                return True
            
            logs = load_logs('attendance:logs', end=10)
            last_action = None
            last_date = None
            
            for log in logs:
                parts = log.split('@')
                if len(parts) >= 4:  # Modified to handle location data
                    log_name = parts[0]
                    log_action = parts[3] if len(parts) > 3 else ''
                    
                    if log_name == name:
                        try:
                            log_timestamp = parts[2]
                            log_datetime = datetime.strptime(log_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                            log_date = log_datetime.date()
                            
                            if last_date is None or log_datetime > last_date:
                                last_action = log_action
                                last_date = log_datetime
                        except ValueError:
                            continue
            
            if last_action is None:
                return True
            
            current_date = datetime.now().date()
            same_day = (last_date.date() == current_date) if last_date else False
            
            if not same_day:
                return True
            
            if last_action == 'Clock_In' and current_action == 'Clock_Out':
                return True
            elif last_action == 'Clock_Out' and current_action == 'Clock_In':
                return True
            elif last_action == current_action:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in check_last_action: {str(e)}")
            return False

    def saveLogs_redis(self, Clock_In_Out):
        """
        Save recognition logs to Redis after validation.
        
        Args:
            Clock_In_Out (str): Either 'Clock_In' or 'Clock_Out'
            
        Raises:
            RedisConnectionError: If there's a problem saving to Redis
        """
        try:
            # Ensure all arrays have the same length
            max_length = max(len(self.logs.get(key, [])) for key in ['name', 'role', 'current_time'])
            
            # Create a clean dictionary with consistent array lengths
            clean_logs = {
                'name': self.logs.get('name', ['Unknown'] * max_length),
                'role': self.logs.get('role', ['Unknown'] * max_length),
                'current_time': self.logs.get('current_time', [str(datetime.now())] * max_length),
                'latitude': self.logs.get('latitude', [None] * max_length),
                'longitude': self.logs.get('longitude', [None] * max_length)
            }
            
            dataframe = pd.DataFrame(clean_logs)
            dataframe.drop_duplicates('name', inplace=True)
            
            name_list = dataframe['name'].tolist()
            role_list = dataframe['role'].tolist()
            current_time_list = dataframe['current_time'].tolist()
            latitude_list = dataframe['latitude'].tolist()
            longitude_list = dataframe['longitude'].tolist()
            
            encoded_data = []

            for name, role, current_time, lat, lon in zip(name_list, role_list, current_time_list, latitude_list, longitude_list):
                if name != 'Unknown':
                    if self.check_last_action(name, Clock_In_Out):
                        # Include location if available
                        location_str = f"@{lat},{lon}" if lat is not None and lon is not None else ""
                        concat_string = f"{name}@{role}@{current_time}@{Clock_In_Out}{location_str}"
                        encoded_data.append(concat_string)
                    else:
                        print(f"Action blocked: {name} attempted {Clock_In_Out} after previous action")

            if len(encoded_data) > 0:
                r.lpush('attendance:logs', *encoded_data)

            self.reset_dict()
            
        except Exception as e:
            raise RedisConnectionError(f"Failed to save logs to Redis: {str(e)}")

    def face_prediction(self, test_image, dataframe, feature_column, name_role=['File No. Name', 'Role'], thresh=0.5):
        """
        Perform face detection and recognition on an input image.
        
        Args:
            test_image (np.array): Input image frame
            dataframe (pd.DataFrame): Staff data with facial features
            feature_column (str): Column name containing facial embeddings
            name_role (list): Column names for name and role
            thresh (float): Similarity threshold for recognition
            
        Returns:
            np.array: Image with detection boxes and labels
            
        Raises:
            FaceRecognitionError: If face detection fails
        """
        try:
            current_time = str(datetime.now())
            results = faceapp.get(test_image)
            test_copy = test_image.copy()
            
            for res in results:
                x1, y1, x2, y2 = res['bbox'].astype(int)
                embeddings = res['embedding']
                person_name, person_role = ml_search_algorithm(dataframe, 
                                                              feature_column,
                                                              test_vector=embeddings,
                                                              name_role=name_role,
                                                              thresh=thresh)
                
                if person_name == 'Unknown':
                    color = (0, 0, 255)  # Red for unknown
                else:
                    color = (0, 255, 0)  # Green for known
                
                cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)
                text_gen = person_name
                cv2.putText(test_copy, text_gen, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                cv2.putText(test_copy, current_time, (x1, y2+10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

                self.logs['name'].append(person_name)
                self.logs['role'].append(person_role)
                self.logs['current_time'].append(current_time)
            
            return test_copy
            
        except Exception as e:
            raise FaceRecognitionError(f"Face prediction failed: {str(e)}")