import customtkinter as ctk
import cv2
import time
from PIL import Image
import os
from src.modules.face_recognition_module import FaceRecognitionModule


class WebcamControlWidget(ctk.CTkFrame):
    def __init__(self, parent, width=640, height=480, primary_color="#26a69a", unknown_color="#b71c1c", text_color="#0f172a", access_status_widget=None, **kwargs):
        super().__init__(parent, fg_color="#1a1a1a", corner_radius=10, **kwargs)
        
        # Variables de configuración
        self.width = width
        self.height = height
        self.primary_color = primary_color
        self.unknown_color = unknown_color
        self.text_color = text_color
        self.access_status_widget = access_status_widget
        
        # Variables de cámara
        self.cap = None
        self.running = False
        self.current_image = None
        
        # Variable para controlar el tiempo del estado
        self.last_detection_time = 0
        self.status_timeout = 3000  # 3 segundos
        
        # Instancia del módulo de reconocimiento facial
        self.face_recognition = FaceRecognitionModule()
        
        # Callbacks para botones
        self.start_callback = None
        self.stop_callback = None
        self.reload_callback = None
        self.download_callback = None
        
        # Variables de animación
        self.reload_animation_running = False
        
        # Crear interfaz
        self.setup_layout()
        self.load_icons()
        self.create_webcam_area()
        self.create_control_buttons()
        
    def setup_layout(self):
        """Configurar el layout principal del widget"""
        # Configurar grid principal
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        
    def load_icons(self):
        """Cargar iconos para los botones"""
        try:
            # Cargar icono de play
            play_path = os.path.join(os.path.dirname(__file__), "..", "assets", "play.png")
            self.play_icon = ctk.CTkImage(
                light_image=Image.open(play_path),
                dark_image=Image.open(play_path),
                size=(20, 20)
            )
            
            # Cargar icono de stop
            stop_path = os.path.join(os.path.dirname(__file__), "..", "assets", "stop.png")
            self.stop_icon = ctk.CTkImage(
                light_image=Image.open(stop_path),
                dark_image=Image.open(stop_path),
                size=(20, 20)
            )
            
            # Cargar icono de reload
            reload_path = os.path.join(os.path.dirname(__file__), "..", "assets", "reload.png")
            self.reload_icon = ctk.CTkImage(
                light_image=Image.open(reload_path),
                dark_image=Image.open(reload_path),
                size=(20, 20)
            )
        except Exception as e:
            print(f"Error cargando iconos: {e}")
            self.play_icon = None
            self.stop_icon = None
            self.reload_icon = None
            
    def create_webcam_area(self):
        """Crear el área de la webcam"""
        # Frame para la webcam
        self.webcam_frame = ctk.CTkFrame(
            self,
            width=self.width,
            height=self.height,
            fg_color="#2a2a2a",
            corner_radius=10
        )
        self.webcam_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.webcam_frame.pack_propagate(False)
        
        # Label para mostrar video
        self.video_label = ctk.CTkLabel(
            self.webcam_frame,
            text="Iniciando cámara...",
            font=ctk.CTkFont(size=16),
            text_color="#666666"
        )
        self.video_label.pack(expand=True, fill="both")
        
    def create_control_buttons(self):
        """Crear los botones de control"""
        # Frame para los botones
        self.buttons_frame = ctk.CTkFrame(
            self,
            height=80,
            fg_color="#2a2a2a",
            corner_radius=10
        )
        self.buttons_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.buttons_frame.pack_propagate(False)
        
        # Configurar grid para los botones
        self.buttons_frame.grid_columnconfigure(0, weight=1)
        self.buttons_frame.grid_columnconfigure(1, weight=1)
        self.buttons_frame.grid_columnconfigure(2, weight=1)
        self.buttons_frame.grid_columnconfigure(3, weight=1)
        
        # Botón de iniciar
        self.start_button = ctk.CTkButton(
            self.buttons_frame,
            text="▶️" if not self.play_icon else "",
            image=self.play_icon,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#174d55",
            hover_color="#1e8e7f",
            text_color="white",
            width=60, height=50,
            command=self._on_start_click,
            state="disabled"
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=15)
        
        # Botón de detener
        self.stop_button = ctk.CTkButton(
            self.buttons_frame,
            text="⏹️" if not self.stop_icon else "",
            image=self.stop_icon,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.primary_color,
            hover_color="#1e8e7f",
            text_color="white",
            width=60, height=50,
            command=self._on_stop_click,
            state="normal"
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=15)
        
        # Botón de recargar
        self.reload_button = ctk.CTkButton(
            self.buttons_frame,
            text="🔄" if not self.reload_icon else "",
            image=self.reload_icon,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#174d55",
            hover_color="#1e8e7f",
            text_color="white",
            width=60, height=50,
            command=self._on_reload_click
        )
        self.reload_button.grid(row=0, column=2, padx=5, pady=15)
        
        # Botón de descargar
        self.download_button = ctk.CTkButton(
            self.buttons_frame,
            text="📥",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#0d47a1",
            hover_color="#1565c0",
            text_color="white",
            width=60, height=50,
            command=self._on_download_click
        )
        self.download_button.grid(row=0, column=3, padx=5, pady=15)
        
    # Métodos de los botones
    def _on_start_click(self):
        """Callback interno para botón start"""
        success = self.start_camera()
        if success:
            self.set_camera_started()
        else:
            self.set_camera_stopped()
            
    def _on_stop_click(self):
        """Callback interno para botón stop"""
        self.stop_camera()
        self.set_camera_stopped()
        
    def _on_reload_click(self):
        """Callback interno para botón reload"""
        if not self.reload_animation_running:
            success = self.reload_face_encodings()
            if success:
                self._animate_reload_success()
            
    def _on_download_click(self):
        """Callback interno para botón download"""
        success = self.download_and_reload_encodings()
        if success:
            self._animate_download_success()
        else:
            self._animate_download_error()
            
    def set_camera_started(self):
        """Configurar estados cuando la cámara inicia"""
        self.start_button.configure(state="disabled", fg_color="#174d55")
        self.stop_button.configure(state="normal", fg_color=self.primary_color)
        
    def set_camera_stopped(self):
        """Configurar estados cuando la cámara se detiene"""
        self.start_button.configure(state="normal", fg_color=self.primary_color)
        self.stop_button.configure(state="disabled", fg_color="#174d55")
        
    def _animate_reload_success(self):
        """Animación para recarga exitosa"""
        self.reload_animation_running = True
        self.reload_button.configure(fg_color="#4caf50", text="✅")
        self.after(1000, self._restore_reload_button)
        
    def _restore_reload_button(self):
        """Restaurar botón de reload"""
        self.reload_button.configure(fg_color="#174d55", text="🔄" if not self.reload_icon else "")
        self.reload_animation_running = False
        
    def _animate_download_success(self):
        """Animación para descarga exitosa"""
        self.download_button.configure(fg_color="#4caf50", text="✅")
        self.after(2000, self._restore_download_button)
        
    def _animate_download_error(self):
        """Animación para error de descarga"""
        self.download_button.configure(fg_color="#f44336", text="❌")
        self.after(2000, self._restore_download_button)
        
    def _restore_download_button(self):
        """Restaurar botón de descarga"""
        self.download_button.configure(fg_color="#0d47a1", text="📥")
        
    # Métodos de la webcam
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
                self._clear_video_display()
                self.video_label.configure(text="Iniciando cámara...")
                
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
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

    def stop_camera(self):
        """Detener cámara"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        # Limpiar estado de acceso
        if self.access_status_widget:
            self.access_status_widget.set_waiting_status()

        # Limpiar estado de forma segura
        try:
            self._clear_video_display()
            self.video_label.configure(
                image="",
                text="Cámara detenida\nPresiona 'Iniciar Cámara' para reanudar"
            )
        except Exception as e:
            print(f"[ERROR] Error al limpiar video label: {e}")

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
                
                # Actualizar estado de acceso basado en detecciones
                self.update_access_status(face_data)
                
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
                    self.current_image = ctk_image
                    self.video_label.configure(image=ctk_image, text="")
                except Exception as e:
                    print(f"[ERROR] Error configurando imagen: {e}")

            if self.running:
                self.after(30, self.update_frame)  # ~33 FPS
        except Exception as e:
            print(f"[ERROR] Error en update_frame: {e}")
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self._set_error_message("Error durante la captura de video")

    def update_access_status(self, face_data):
        """Actualizar el estado de acceso basado en las detecciones de rostros"""
        if not self.access_status_widget:
            return
        
        current_time = time.time() * 1000
        
        if face_data:
            recognized_faces = []
            unknown_faces = []
            
            for (location, (name, confidence)) in face_data:
                if name != "Desconocido":
                    recognized_faces.append(name)
                else:
                    unknown_faces.append(name)
            
            if recognized_faces:
                person_name = recognized_faces[0]
                self.access_status_widget.set_access_granted(person_name)
                self.last_detection_time = current_time
            elif unknown_faces:
                self.access_status_widget.set_access_denied()
                self.last_detection_time = current_time
        else:
            if current_time - self.last_detection_time > self.status_timeout:
                self.access_status_widget.set_waiting_status()

    def _clear_video_display(self):
        """Limpiar display de video de forma segura"""
        try:
            self.current_image = None
            self.video_label.update()
        except Exception as e:
            print(f"[DEBUG] Error clearing display: {e}")

    def _set_error_message(self, message):
        """Establecer mensaje de error de forma segura"""
        try:
            self._clear_video_display()
            self.video_label.configure(image="", text=message)
        except Exception as e:
            print(f"[ERROR] No se pudo mostrar mensaje de error: {e}")

    def cleanup(self):
        """Limpiar recursos al cerrar"""
        self.running = False
        self.face_recognition.close_arduino_connection()
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.access_status_widget:
            self.access_status_widget.set_waiting_status()
            
        cv2.destroyAllWindows()
