import os
import face_recognition
import numpy as np
import cv2
from PIL import Image
import sqlite3
import pickle
import uuid
from datetime import datetime

class FaceEngine:
    def __init__(self, db_path='instance/facesnap.sqlite'):
        self.db_path = db_path
        self.face_similarity_threshold = 0.6  # Lower is more strict
        
    def _get_db_connection(self):
        """Get a connection to the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def detect_faces(self, image_path):
        """Detect faces in an image and return their locations"""
        # Load image using face_recognition library
        image = face_recognition.load_image_file(image_path)
        
        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image, model="hog")
        
        # Get face encodings for each face found
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        return image, face_locations, face_encodings
    
    def save_face_crop(self, image, face_location, event_id, cluster_id):
        """Crop a face from an image and save it to the appropriate directory"""
        # Ensure directory exists
        cluster_dir = os.path.join('static', 'faces', str(event_id), f'cluster_{cluster_id}')
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Extract face coordinates
        top, right, bottom, left = face_location
        
        # Add some margin to the face crop (20% of face size)
        height = bottom - top
        width = right - left
        margin_h = int(height * 0.2)
        margin_w = int(width * 0.2)
        
        # Adjust coordinates with margin, ensuring they stay within image bounds
        img_height, img_width = image.shape[:2]
        top = max(0, top - margin_h)
        bottom = min(img_height, bottom + margin_h)
        left = max(0, left - margin_w)
        right = min(img_width, right + margin_w)
        
        # Crop the face from the image
        face_image = image[top:bottom, left:right]
        
        # Convert from BGR to RGB (if using OpenCV)
        if isinstance(image, np.ndarray) and image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face_image)
        
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        file_path = os.path.join(cluster_dir, filename)
        
        # Save the face image
        pil_image.save(file_path)
        
        return file_path
    
    def find_or_create_cluster(self, event_id, face_encoding):
        """Find an existing face cluster or create a new one based on face similarity"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Get all existing clusters for this event
        cursor.execute(
            "SELECT id FROM face_clusters WHERE event_id = ?", 
            (event_id,)
        )
        clusters = cursor.fetchall()
        
        best_match_cluster_id = None
        best_match_distance = float('inf')
        
        # Compare face encoding with representative faces from each cluster
        for cluster in clusters:
            cluster_id = cluster['id']
            
            # Get a sample image from this cluster
            cursor.execute(
                "SELECT file_path FROM images WHERE event_id = ? AND cluster_id = ? LIMIT 1",
                (event_id, cluster_id)
            )
            sample_image = cursor.fetchone()
            
            if sample_image:
                # Load the sample image and get its encoding
                try:
                    sample_img = face_recognition.load_image_file(sample_image['file_path'])
                    sample_encodings = face_recognition.face_encodings(sample_img)
                    
                    if sample_encodings:
                        # Calculate face distance
                        face_distance = face_recognition.face_distance([sample_encodings[0]], face_encoding)[0]
                        
                        # Update best match if this is better
                        if face_distance < best_match_distance:
                            best_match_distance = face_distance
                            best_match_cluster_id = cluster_id
                except Exception as e:
                    print(f"Error processing sample image: {e}")
        
        # If we found a good match, use that cluster
        if best_match_cluster_id is not None and best_match_distance < self.face_similarity_threshold:
            conn.close()
            return best_match_cluster_id
        
        # Otherwise, create a new cluster
        cursor.execute(
            "INSERT INTO face_clusters (event_id, created_at) VALUES (?, ?)",
            (event_id, datetime.now().isoformat())
        )
        new_cluster_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return new_cluster_id
    
    def process_image(self, image_path, event_id):
        """Process an uploaded image, detect faces, and assign to clusters"""
        # Detect faces in the image
        image, face_locations, face_encodings = self.detect_faces(image_path)
        
        # Convert image from RGB to BGR for OpenCV processing
        if len(face_locations) > 0:
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process each detected face
        results = []
        for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
            # Find or create a cluster for this face
            cluster_id = self.find_or_create_cluster(event_id, face_encoding)
            
            # Save the face crop
            face_path = self.save_face_crop(image_cv, face_location, event_id, cluster_id)
            
            # Store the image and face information in the database
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Convert face encoding to binary for storage
            face_encoding_binary = pickle.dumps(face_encoding)
            
            # Store the relative path for database storage
            db_image_path = os.path.relpath(image_path, 'static').replace('\\', '/')
            db_face_path = os.path.relpath(face_path, 'static').replace('\\', '/')

            # Store in images table
            cursor.execute(
                "INSERT INTO images (file_path, cluster_id, event_id, created_at) VALUES (?, ?, ?, ?)",
                (db_image_path, cluster_id, event_id, datetime.now().isoformat())
            )
            image_id = cursor.lastrowid

            # Store face crop information in face_crops table
            cursor.execute(
                "INSERT INTO face_crops (file_path, cluster_id, image_id, created_at) VALUES (?, ?, ?, ?)",
                (db_face_path, cluster_id, image_id, datetime.now().isoformat())
            )
            
            conn.commit()
            conn.close()
            
            # Add to results
            results.append({
                'face_location': face_location,
                'cluster_id': cluster_id,
                'face_path': face_path
            })
        
        return results
    
    def verify_user(self, selfie_path, event_id):
        """Verify a user by matching their selfie with existing face clusters"""
        # Detect face in selfie
        try:
            image, face_locations, face_encodings = self.detect_faces(selfie_path)
            
            if not face_locations or len(face_locations) == 0:
                return {'success': False, 'message': 'No face detected in selfie'}
            
            if len(face_locations) > 1:
                return {'success': False, 'message': 'Multiple faces detected in selfie. Please submit a selfie with only your face.'}
            
            # Get the face encoding from the selfie
            selfie_encoding = face_encodings[0]
            
            # Find the best matching cluster
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Get all clusters for this event
            cursor.execute(
                "SELECT id FROM face_clusters WHERE event_id = ?", 
                (event_id,)
            )
            clusters = cursor.fetchall()
            
            best_match_cluster_id = None
            best_match_distance = float('inf')
            
            # Compare selfie with representative faces from each cluster
            for cluster in clusters:
                cluster_id = cluster['id']
                
                # Get a sample image from this cluster
                cursor.execute(
                    "SELECT file_path FROM images WHERE event_id = ? AND cluster_id = ? LIMIT 1",
                    (event_id, cluster_id)
                )
                sample_image = cursor.fetchone()
                
                if sample_image:
                    try:
                        sample_img = face_recognition.load_image_file(sample_image['file_path'])
                        sample_encodings = face_recognition.face_encodings(sample_img)
                        
                        if sample_encodings:
                            # Calculate face distance
                            face_distance = face_recognition.face_distance([sample_encodings[0]], selfie_encoding)[0]
                            
                            # Update best match if this is better
                            if face_distance < best_match_distance:
                                best_match_distance = face_distance
                                best_match_cluster_id = cluster_id
                    except Exception as e:
                        print(f"Error processing sample image: {e}")
            
            conn.close()
            
            # If we found a good match, return success
            if best_match_cluster_id is not None and best_match_distance < self.face_similarity_threshold:
                return {
                    'success': True, 
                    'cluster_id': best_match_cluster_id,
                    'confidence': 1.0 - best_match_distance  # Convert distance to confidence score
                }
            else:
                return {'success': False, 'message': 'No matching face found in our database'}
                
        except Exception as e:
            return {'success': False, 'message': f'Error during verification: {str(e)}'}