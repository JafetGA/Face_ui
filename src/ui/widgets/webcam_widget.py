import customtkinter as ctk
import cv2
import numpy as np
import pickle
import face_recognition
from PIL import Image
import os


class WebcamWidget(ctk.CTkFrame):
    def __init__(self, parent, width=640, height=480, primary_color="#26a69a", unknown_color="#b71c1c", text_color="#0f172a", **kwargs):
        super().__init__(parent, width=width, height=height, fg_color="#1a1a1a", corner_radius=10, **kwargs)
        self.pack_propagate(False)

        # Variables de cámara
        self.cap = None
        self.running = False
        self.current_image = None  # Mantener referencia a la imagen actual
        
        # Variables de reconocimiento facial
        self.known_face_encodings = []
        self.known_face_names = []
        self.cv_scaler = 4  # Reducir para mejor rendimiento
        
        # Colores personalizados
        self.primary_color = primary_color
        self.unknown_color = unknown_color
        self.text_color = text_color
        
        self.load_face_encodings()

        # Label para mostrar video
        self.video_label = ctk.CTkLabel(
            self,
            text="Iniciando cámara...",
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
            
            # Convertir colores hex a BGR para OpenCV
            if name != "Desconocido":
                # Convertir primary_color de hex a BGR
                hex_color = self.primary_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                color = (b, g, r)  # BGR para OpenCV
            else:
                # Convertir unknown_color de hex a BGR
                hex_color = self.unknown_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                color = (b, g, r)  # BGR para OpenCV
            
            # Convertir text_color de hex a BGR
            text_hex = self.text_color.lstrip('#')
            text_r, text_g, text_b = tuple(int(text_hex[i:i+2], 16) for i in (0, 2, 4))
            text_color = (text_b, text_g, text_r)  # BGR para OpenCV
            
            # Dibujar rectángulo alrededor del rostro con línea más gruesa
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            
            # Calcular dimensiones del texto para centrar mejor
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, thickness)
            
            # Crear rectángulo para el fondo del texto más elegante
            label_height = text_height + baseline + 10
            cv2.rectangle(frame, (left, bottom - label_height), (left + text_width + 10, bottom), color, cv2.FILLED)
            
            # Agregar borde al rectángulo del texto
            cv2.rectangle(frame, (left, bottom - label_height), (left + text_width + 10, bottom), color, 2)
            
            # Dibujar el texto con mejor posicionamiento
            cv2.putText(frame, name, (left + 5, bottom - baseline - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        return frame

    def start_camera(self):
        """Iniciar cámara"""
        if not self.running:
            try:
                # Limpiar estado previo
                self._clear_video_display()
                self.video_label.configure(text="Iniciando cámara...")
                
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    # Verificar que realmente puede leer frames
                    ret, _ = self.cap.read()
                    if ret:
                        self.running = True
                        self.update_frame()
                        print("[INFO] Cámara iniciada exitosamente")
                        return True
                    else:
                        self.cap.release()
                        self.cap = None
                        error_msg = "Error: La cámara no puede capturar imágenes"
                        print(f"[ERROR] {error_msg}")
                        self._set_error_message(error_msg)
                        return False
                else:
                    self.cap = None
                    error_msg = "Error: No se pudo acceder a la cámara\nVerifica que no esté siendo usada por otra aplicación"
                    print(f"[ERROR] {error_msg}")
                    self._set_error_message(error_msg)
                    return False
            except Exception as e:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                error_msg = f"Error inesperado al iniciar cámara:\n{str(e)}"
                print(f"[ERROR] {error_msg}")
                self._set_error_message(error_msg)
                return False
        return False

    def _clear_video_display(self):
        """Limpiar display de video de forma segura"""
        try:
            # Liberar referencia a imagen actual
            self.current_image = None
            # Forzar actualización del widget
            self.video_label.update()
        except Exception as e:
            print(f"[DEBUG] Error clearing display: {e}")

    def _set_error_message(self, message):
        """Establecer mensaje de error de forma segura"""
        try:
            self._clear_video_display()
            # Crear un nuevo label si es necesario
            self.video_label.configure(image="", text=message)
        except Exception as e:
            print(f"[ERROR] No se pudo mostrar mensaje de error: {e}")
            # Como último recurso, recrear el label
            self._recreate_video_label(message)

    def _recreate_video_label(self, text=""):
        """Recrear el video label como último recurso"""
        try:
            self.video_label.destroy()
            self.video_label = ctk.CTkLabel(
                self,
                text=text if text else "Error de visualización",
                font=ctk.CTkFont(size=16),
                text_color="#666666"
            )
            self.video_label.pack(expand=True, fill="both")
            self.current_image = None
        except Exception as e:
            print(f"[ERROR] No se pudo recrear video label: {e}")

    def stop_camera(self):
        """Detener cámara"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        # Limpiar estado de forma segura
        try:
            self._clear_video_display()
            self.video_label.configure(
                image="",
                text="Cámara detenida\nPresiona 'Iniciar Cámara' para reanudar"
            )
        except Exception as e:
            print(f"[ERROR] Error al limpiar video label: {e}")
            self._recreate_video_label("Cámara detenida\nPresiona 'Iniciar Cámara' para reanudar")

    def update_frame(self):
        """Actualizar frame de video"""
        if not self.running or not self.cap or not self.cap.isOpened():
            return

        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (620, 460))
                
                # Procesar reconocimiento facial
                processed_frame, face_data = self.process_face_recognition(frame)
                
                # Dibujar cajas y nombres en rostros detectados
                display_frame = self.draw_face_boxes(processed_frame, face_data)
                
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Crear nueva imagen CTk
                ctk_image = ctk.CTkImage(
                    light_image=pil_image,
                    dark_image=pil_image,
                    size=(620, 460)
                )
                
                # Actualizar imagen de forma segura
                try:
                    self.current_image = ctk_image  # Mantener referencia
                    self.video_label.configure(image=ctk_image, text="")
                except Exception as e:
                    print(f"[ERROR] Error configurando imagen: {e}")
                    # Continuar sin mostrar la imagen
                    pass

            if self.running:  # Solo continuar si todavía está corriendo
                self.after(30, self.update_frame)  # ~33 FPS
        except Exception as e:
            print(f"[ERROR] Error en update_frame: {e}")
            # En caso de error, detener la cámara
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self._set_error_message("Error durante la captura de video")

    def cleanup(self):
        """Limpiar recursos al cerrar"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()