import cv2
import numpy as np
import pickle
import face_recognition
import os
import serial
import time


class FaceRecognitionModule:
    def __init__(self, arduino_port='COM5', baud_rate=9600):
        self.known_face_encodings = []
        self.known_face_names = []
        self.cv_scaler = 4  # Reducir para mejor rendimiento
        self.tolerance = 0.45  # Tolerancia más estricta para reducir falsos positivos
        
        # Configuración Arduino
        self.arduino = None
        self.arduino_connected = False
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # Segundos entre detecciones para evitar spam
        
        # Cargar encodings al inicializar
        self.load_face_encodings()
        
        # Conectar con Arduino
        self.connect_arduino(arduino_port, baud_rate)

    def connect_arduino(self, port, baud_rate):
        """Conectar con Arduino"""
        try:
            self.arduino = serial.Serial(port, baud_rate, timeout=1)
            time.sleep(2)  # Esperar que Arduino se inicialice
            self.arduino_connected = True
            print(f"[INFO] Arduino conectado en {port}")
            
            # Enviar señal de prueba
            self.send_arduino_signal('TEST')
            
        except serial.SerialException as e:
            print(f"[WARNING] No se pudo conectar con Arduino en {port}: {e}")
            print("[INFO] El sistema funcionará sin control de LEDs")
            self.arduino_connected = False
        except Exception as e:
            print(f"[ERROR] Error inesperado al conectar Arduino: {e}")
            self.arduino_connected = False

    def send_arduino_signal(self, signal):
        """Enviar señal al Arduino"""
        if not self.arduino_connected:
            return False
            
        try:
            # Enviar señal como string terminado en newline
            self.arduino.write(f"{signal}\n".encode())
            print(f"[DEBUG] Señal enviada a Arduino: {signal}")
            return True
        except Exception as e:
            print(f"[ERROR] Error enviando señal a Arduino: {e}")
            return False

    def control_leds(self, face_detected, is_known=False):
        """Controlar LEDs basado en detección facial"""
        current_time = time.time()
        
        # Verificar cooldown para evitar spam de señales
        if current_time - self.last_detection_time < self.detection_cooldown:
            return
            
        if face_detected:
            if is_known:
                # Persona conocida - LED verde (pin 11)
                self.send_arduino_signal('KNOWN')
                print("[INFO] LED verde encendido - Persona conocida")
            else:
                # Persona desconocida - LED rojo (pin 10)
                self.send_arduino_signal('UNKNOWN')
                print("[INFO] LED rojo encendido - Persona desconocida")
                
            self.last_detection_time = current_time
        else:
            # No hay detección - apagar LEDs
            self.send_arduino_signal('OFF')

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
        known_person_detected = False
        
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
                        known_person_detected = True
                        print(f"[DEBUG] Reconocido: {name} (Confianza: {confidence:.1f}%, Distancia: {min_distance:.3f}, Ratio: {match_ratio:.2f})")
                    else:
                        print(f"[DEBUG] Rechazado: {candidate_name} (Ratio muy bajo: {match_ratio:.2f})")
                else:
                    print(f"[DEBUG] Rechazado por tolerancia (Distancia: {min_distance:.3f} > {self.tolerance})")
            else:
                print(f"[DEBUG] Rostro desconocido (Distancia mínima: {min_distance:.3f} > {self.tolerance})")
            
            face_names.append((name, confidence))
        
        # Controlar LEDs basado en detección
        if len(face_locations) > 0:
            self.control_leds(True, known_person_detected)
        else:
            self.control_leds(False)
        
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

    def download_and_reload_encodings(self, api_url="http://localhost:8000/api/face_attendance/v1/download"):
        """
        Descarga nuevos encodings desde la API y los recarga
        
        Args:
            api_url (str): URL de la API para descargar los encodings
            
        Returns:
            bool: True si la descarga y recarga fueron exitosas, False en caso contrario
        """
        try:
            # Importar la función de descarga
            from src.api.download_encodings import download_encodings_from_api
            
            # Descargar los nuevos encodings
            print("[INFO] Descargando encodings desde la API...")
            if download_encodings_from_api(api_url):
                # Recargar los encodings
                print("[INFO] Recargando encodings...")
                return self.reload_face_encodings()
            else:
                print("[ERROR] No se pudieron descargar los encodings desde la API")
                return False
                
        except ImportError as e:
            print(f"[ERROR] Error al importar módulo de descarga: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Error inesperado al descargar y recargar encodings: {e}")
            return False

    def close_arduino_connection(self):
        """Cerrar conexión con Arduino"""
        if self.arduino_connected and self.arduino:
            try:
                self.send_arduino_signal('OFF')  # Apagar LEDs antes de cerrar
                self.arduino.close()
                print("[INFO] Conexión con Arduino cerrada")
            except Exception as e:
                print(f"[ERROR] Error cerrando conexión Arduino: {e}")

    def __del__(self):
        """Destructor para cerrar conexión Arduino"""
        self.close_arduino_connection()


# Ejemplo de configuración en tu WebcamUI:
# 
# Modifica tu WebcamUI para crear la instancia así:
# self.face_module = FaceRecognitionModule(arduino_port='COM3', baud_rate=9600)
#
# Y asegúrate de llamar face_module.close_arduino_connection() al cerrar la aplicación