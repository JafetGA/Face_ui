import customtkinter as ctk
import cv2
from PIL import Image


class WebcamWidget(ctk.CTkFrame):
    def __init__(self, parent, width=640, height=480, **kwargs):
        super().__init__(parent, width=width, height=height, fg_color="#1a1a1a", corner_radius=10, **kwargs)
        self.pack_propagate(False)

        # Variables de cámara
        self.cap = None
        self.running = False

        # Label para mostrar video
        self.video_label = ctk.CTkLabel(
            self,
            text="Presiona 'Iniciar Cámara' para comenzar",
            font=ctk.CTkFont(size=16),
            text_color="#666666"
        )
        self.video_label.pack(expand=True, fill="both")

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

        self.video_label.configure(
            image=None,
            text="Cámara detenida\nPresiona 'Iniciar Cámara' para reanudar"
        )

    def update_frame(self):
        """Actualizar frame de video"""
        if not self.running or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (620, 460))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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