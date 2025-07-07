import customtkinter as ctk
from src.ui.widgets import LogoWidget, ClockWidget, WebcamWidget, ControlButtonsWidget


class WebcamUI:
    def __init__(self):
        # Configuración de CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Colores personalizados
        self.primary_color = "#26a69a"
        self.disabled_color = "#174d55"
        self.text_color = "#0f172a"
        self.bg_color = "#020617"
        self.unknown_color = "#b71c1c"  # Color para rostros desconocidos

        # Crear ventana principal
        self.root = ctk.CTk()
        self.root.title("Face ID")
        self.root.iconbitmap("src/ui/assets/icon.ico")
        self.root.configure(fg_color=self.bg_color)

        # Crear interfaz
        self.create_interface()

        self.logo_widget.load_logo("src/ui/assets/luminilogo.svg")

        try:
            self.root.state('zoomed')
        except Exception as e:
            print(f"Error al maximizar la ventana: {e}")
            self.root.attributes('-zoomed', True)

    def create_interface(self):
        """Crear la interfaz usando widgets"""
        # Main frame principal
        self.main_frame = ctk.CTkFrame(self.root, fg_color=self.bg_color, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Widget de logo posicionado en esquina superior izquierda
        self.logo_widget = LogoWidget(
            self.main_frame,
            primary_color=self.primary_color
        )
        self.logo_widget.place(x=0, y=0)

        # Widget de reloj centrado en la parte superior
        self.clock_widget = ClockWidget(
            self.main_frame,
            primary_color=self.primary_color
        )
        self.clock_widget.pack(pady=(80, 30))

        # Widget de webcam
        self.webcam_widget = WebcamWidget(
            self.main_frame,
            width=640,
            height=480,
            primary_color=self.primary_color,
            unknown_color=self.unknown_color,
            text_color=self.text_color
        )
        self.webcam_widget.pack(pady=20, padx=20)

        # Widget de botones de control
        self.control_buttons = ControlButtonsWidget(
            self.main_frame,
            primary_color=self.primary_color,
            disabled_color=self.disabled_color,
            text_color=self.text_color
        )
        self.control_buttons.pack(pady=20)

        # Configurar callbacks de los botones
        self.control_buttons.set_callbacks(
            start_callback=self.webcam_widget.start_camera,
            stop_callback=self.webcam_widget.stop_camera,
            reload_callback=self.webcam_widget.reload_face_encodings,
            download_callback=self.webcam_widget.download_and_reload_encodings
        )

        # Iniciar la cámara automáticamente
        self.auto_start_camera()

    def auto_start_camera(self):
        """Iniciar cámara automáticamente al arrancar la aplicación"""
        # Establecer estado inicial de botones (cámara iniciada)
        self.control_buttons.set_camera_started()
        
        # Intentar iniciar la cámara
        success = self.webcam_widget.start_camera()
        
        # Si falla, revertir estado de botones
        if not success:
            self.control_buttons.set_camera_stopped()

    def on_closing(self):
        """Manejar cierre de aplicación"""
        self.webcam_widget.cleanup()
        self.root.destroy()

    def run(self):
        """Ejecutar aplicación"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()