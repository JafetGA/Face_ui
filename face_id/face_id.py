import face_recognition
import cv2
import numpy as np
import time
import pickle

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Initialize our variables
cv_scaler = 4  # this has to be a whole number

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0


def process_frame(frame):
    global face_locations, face_encodings, face_names

    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))

    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    face_names = []
    for face_encoding in face_encodings:
        # Calcular distancias a todos los rostros conocidos
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Encontrar la distancia mínima
        min_distance = np.min(face_distances)
        best_match_index = np.argmin(face_distances)
        
        name = "Unknown"
        # Tolerancia más estricta para reducir falsos positivos
        tolerance = 0.45  # Más estricto que el default (0.6)
        
        # Solo asignar nombre si la distancia está dentro de la tolerancia
        if min_distance <= tolerance:
            # Verificación adicional: comprobar que hay múltiples encodings que coinciden
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
            
            if matches[best_match_index]:
                candidate_name = known_face_names[best_match_index]
                
                # Contar cuántos encodings de esta persona coinciden
                same_person_indices = [i for i, n in enumerate(known_face_names) if n == candidate_name]
                same_person_matches = [matches[i] for i in same_person_indices]
                match_ratio = sum(same_person_matches) / len(same_person_matches)
                
                # Requerir que al menos 50% de los encodings de la persona coincidan
                if match_ratio >= 0.5:
                    name = candidate_name
                    confidence = (1 - min_distance) * 100  # Convertir a porcentaje de confianza
                    print(f"[DEBUG] Recognized: {name} (Confidence: {confidence:.1f}%, Distance: {min_distance:.3f}, Ratio: {match_ratio:.2f})")
                else:
                    print(f"[DEBUG] Rejected: {candidate_name} (Ratio too low: {match_ratio:.2f})")
            else:
                print(f"[DEBUG] Rejected by tolerance (Distance: {min_distance:.3f} > {tolerance})")
        else:
            print(f"[DEBUG] Unknown face (Min distance: {min_distance:.3f} > {tolerance})")
        
        face_names.append(name)

    return frame


def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

    return frame


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps


while True:
    # Capture a frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    # Process the frame with the function
    processed_frame = process_frame(frame)

    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)

    # Calculate and update FPS
    current_fps = calculate_fps()

    # Attach FPS counter to the text and boxes
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display everything over the video feed.
    cv2.imshow('Video', display_frame)

    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# By breaking the loop we run this code here which closes everything
cap.release()
cv2.destroyAllWindows()