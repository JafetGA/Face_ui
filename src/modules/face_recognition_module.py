import cv2
import numpy as np
import pickle
import face_recognition
import os


class FaceRecognitionModule:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.cv_scaler = 4  # Reducir para mejor rendimiento
        self.tolerance = 0.45  # Tolerancia más estricta para reducir falsos positivos
        
        # Cargar encodings al inicializar
        self.load_face_encodings()

    def load_face_encodings(self):
        """Cargar encodings de rostros conocidos"""
        encodings_path = os.path.join(os.path.dirname(__file__), "..", "encodings", "encodings.pickle")
        try:
            with open(encodings_path, "rb") as f:
                data = pickle.loads(f.read())
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]
            print(f"[INFO] Loaded {len(self.known_face_encodings)} face encodings")
            return True
        except FileNotFoundError:
            print("[WARNING] encodings.pickle not found. Face recognition disabled.")
            self.known_face_encodings = []
            self.known_face_names = []
            return False

    def reload_face_encodings(self):
        """Recargar encodings de rostros conocidos dinámicamente"""
        print("[INFO] Recargando encodings de rostros...")
        success = self.load_face_encodings()
        if success:
            print(f"[INFO] Encodings recargados exitosamente. Total: {len(self.known_face_encodings)} rostros")
        else:
            print("[ERROR] No se pudieron recargar los encodings")
        return success

    def process_face_recognition(self, frame):
        """Procesar reconocimiento facial en el frame"""
        if not self.known_face_encodings:
            return frame, []
        
        # Redimensionar frame para mejor rendimiento
        small_frame = cv2.resize(frame, (0, 0), fx=(1/self.cv_scaler), fy=(1/self.cv_scaler))
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Encontrar rostros y sus encodings - usar modelo consistente con training
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model="large")
        
        face_names = []
        
        for face_encoding in face_encodings:
            # Calcular distancias a todos los rostros conocidos
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            # Encontrar la distancia mínima
            min_distance = np.min(face_distances)
            best_match_index = np.argmin(face_distances)
            
            name = "Desconocido"
            confidence = 0.0
            
            # Solo asignar nombre si la distancia está dentro de la tolerancia
            if min_distance <= self.tolerance:
                # Verificación adicional: comprobar que hay múltiples encodings que coinciden
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
                
                if matches[best_match_index]:
                    candidate_name = self.known_face_names[best_match_index]
                    
                    # Contar cuántos encodings de esta persona coinciden
                    same_person_indices = [i for i, n in enumerate(self.known_face_names) if n == candidate_name]
                    same_person_matches = [matches[i] for i in same_person_indices]
                    match_ratio = sum(same_person_matches) / len(same_person_matches)
                    
                    # Requerir que al menos 50% de los encodings de la persona coincidan
                    if match_ratio >= 0.5:
                        name = candidate_name
                        confidence = (1 - min_distance) * 100  # Convertir a porcentaje de confianza
                        print(f"[DEBUG] Reconocido: {name} (Confianza: {confidence:.1f}%, Distancia: {min_distance:.3f}, Ratio: {match_ratio:.2f})")
                    else:
                        print(f"[DEBUG] Rechazado: {candidate_name} (Ratio muy bajo: {match_ratio:.2f})")
                else:
                    print(f"[DEBUG] Rechazado por tolerancia (Distancia: {min_distance:.3f} > {self.tolerance})")
            else:
                print(f"[DEBUG] Rostro desconocido (Distancia mínima: {min_distance:.3f} > {self.tolerance})")
            
            face_names.append((name, confidence))
        
        return frame, list(zip(face_locations, face_names))

    def draw_face_boxes(self, frame, face_data, primary_color="#26a69a", unknown_color="#b71c1c", text_color="#0f172a"):
        """Dibujar cajas y nombres en los rostros detectados"""
        for (top, right, bottom, left), (name, confidence) in face_data:
            # Escalar coordenadas de vuelta al tamaño original
            top *= self.cv_scaler
            right *= self.cv_scaler
            bottom *= self.cv_scaler
            left *= self.cv_scaler
            
            # Convertir colores hex a BGR para OpenCV
            if name != "Desconocido":
                # Convertir primary_color de hex a BGR
                hex_color = primary_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                color = (b, g, r)  # BGR para OpenCV
                
                # Mostrar nombre con confianza si está identificado
                display_text = f"{name} ({confidence:.0f}%)"
            else:
                # Convertir unknown_color de hex a BGR
                hex_color = unknown_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                color = (b, g, r)  # BGR para OpenCV
                display_text = name
            
            # Convertir text_color de hex a BGR
            text_hex = text_color.lstrip('#')
            text_r, text_g, text_b = tuple(int(text_hex[i:i+2], 16) for i in (0, 2, 4))
            text_color_bgr = (text_b, text_g, text_r)  # BGR para OpenCV
            
            # Dibujar rectángulo alrededor del rostro con línea más gruesa
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            
            # Calcular dimensiones del texto para centrar mejor
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
            
            # Crear rectángulo para el fondo del texto más elegante
            label_height = text_height + baseline + 10
            cv2.rectangle(frame, (left, bottom - label_height), (left + text_width + 10, bottom), color, cv2.FILLED)
            
            # Agregar borde al rectángulo del texto
            cv2.rectangle(frame, (left, bottom - label_height), (left + text_width + 10, bottom), color, 2)
            
            # Dibujar el texto con mejor posicionamiento
            cv2.putText(frame, display_text, (left + 5, bottom - baseline - 5), font, font_scale, text_color_bgr, thickness, cv2.LINE_AA)
        
        return frame

    def has_face_encodings(self):
        """Verificar si hay encodings cargados"""
        return len(self.known_face_encodings) > 0

    def get_face_count(self):
        """Obtener número de rostros conocidos"""
        return len(self.known_face_encodings)