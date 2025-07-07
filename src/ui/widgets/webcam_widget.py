import customtkinter as ctk
import cv2
from PIL import Image
from src.modules.face_recognition_module import FaceRecognitionModule


class WebcamWidget(ctk.CTkFrame):
    def __init__(self, parent, width=640, height=480, primary_color="#26a69a", unknown_color="#b71c1c", text_color="#0f172a", **kwargs):
        super().__init__(parent, width=width, height=height, fg_color="#1a1a1a", corner_radius=10, **kwargs)
        self.pack_propagate(False)

        # Variables de cámara
        self.cap = None
        self.running = False
        self.current_image = None  # Mantener referencia a la imagen actual
        
        # Instancia del módulo de reconocimiento facial
        self.face_recognition = FaceRecognitionModule()
        
        # Colores personalizados
        self.primary_color = primary_color
        self.unknown_color = unknown_color
        self.text_color = text_color

        # Label para mostrar video
        self.video_label = ctk.CTkLabel(
            self,
            text="Iniciando cámara...",
            font=ctk.CTkFont(size=16),
            text_color="#666666"
        )
        self.video_label.pack(expand=True, fill="both")

    def reload_face_encodings(self):
        """Recargar encodings de rostros conocidos dinámicamente"""
        return self.face_recognition.reload_face_encodings()

    def process_face_recognition(self, frame):
        """Procesar reconocimiento facial en el frame"""
        return self.face_recognition.process_face_recognition(frame)

    def draw_face_boxes(self, frame, face_data):
        """Dibujar cajas y nombres en los rostros detectados"""
        return self.face_recognition.draw_face_boxes(
            frame, 
            face_data, 
            self.primary_color, 
            self.unknown_color, 
            self.text_color
        )

    def download_and_reload_encodings(self):
        """Descargar encodings desde la API y recargarlos"""
        return self.face_recognition.download_and_reload_encodings()

    def start_camera(self):
        """Iniciar cámara"""
        if not self.running:
            try:
                # Limpiar estado previo
                self._clear_video_display()
                self.video_label.configure(text="Iniciando cámara...")
                
                self.cap = cv2.VideoCapture(1)
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