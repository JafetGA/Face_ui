import customtkinter as ctk
import cv2
import numpy as np
import pickle
import face_recognition
from PIL import Image
import os


class WebcamWidget(ctk.CTkFrame):
    def __init__(self, parent, width=640, height=480, **kwargs):
        super().__init__(parent, width=width, height=height, fg_color="#1a1a1a", corner_radius=10, **kwargs)
        self.pack_propagate(False)

        # Variables de cámara
        self.cap = None
        self.running = False
        
        # Variables de reconocimiento facial
        self.known_face_encodings = []
        self.known_face_names = []
        self.cv_scaler = 7  # Reducir para mejor rendimiento
        self.load_face_encodings()

        # Label para mostrar video
        self.video_label = ctk.CTkLabel(
            self,
            text="Presiona 'Iniciar Cámara' para comenzar",
            font=ctk.CTkFont(size=16),
            text_color="#666666"
        )
        self.video_label.pack(expand=True, fill="both")

    def load_face_encodings(self):
        """Cargar encodings de rostros conocidos"""
        encodings_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src/encodings/encodings.pickle")
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
        
        # Encontrar rostros y sus encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # Comparar con rostros conocidos
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Desconocido"
            
            # Usar el rostro con menor distancia
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)
        
        return frame, list(zip(face_locations, face_names))

    def draw_face_boxes(self, frame, face_data):
        """Dibujar cajas y nombres en los rostros detectados"""
        for (top, right, bottom, left), name in face_data:
            # Escalar coordenadas de vuelta al tamaño original
            top *= self.cv_scaler
            right *= self.cv_scaler
            bottom *= self.cv_scaler
            left *= self.cv_scaler
            
            # Dibujar rectángulo alrededor del rostro
            color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Dibujar etiqueta con el nombre
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame

    def start_camera(self):
        """Iniciar cámara"""
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.running = True
                self.update_frame()
                return True
            else:
                self.video_label.configure(text="Error: No se pudo acceder a la cámara")
                return False
        return False

    def stop_camera(self):
        """Detener cámara"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        # Limpiar la imagen y restaurar el texto predeterminado
        self.video_label.configure(
            image=None,
            text="Cámara detenida\nPresiona 'Iniciar Cámara' para reanudar"
        )
        # Asegurar que no quede referencia a la imagen anterior
        self.video_label.image = None

    def update_frame(self):
        """Actualizar frame de video"""
        if not self.running or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (620, 460))
            
            # Procesar reconocimiento facial
            processed_frame, face_data = self.process_face_recognition(frame)
            
            # Dibujar cajas y nombres en rostros detectados
            display_frame = self.draw_face_boxes(processed_frame, face_data)
            
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            ctk_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=(620, 460)
            )
            self.video_label.configure(image=ctk_image, text="")
            self.video_label.image = ctk_image

        self.after(30, self.update_frame)  # ~33 FPS

    def cleanup(self):
        """Limpiar recursos al cerrar"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()