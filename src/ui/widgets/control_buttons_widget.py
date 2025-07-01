import customtkinter as ctk
from PIL import Image
import os


class ControlButtonsWidget(ctk.CTkFrame):
    def __init__(self, parent, primary_color="#26a69a", disabled_color="#174d55", text_color="#0f172a", **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.primary_color = primary_color
        self.disabled_color = disabled_color
        self.text_color = text_color

        # Callbacks
        self.start_callback = None
        self.stop_callback = None
        self.reload_callback = None

        # Animation variables
        self.reload_animation_running = False

        # Load icons
        self.load_icons()
        self.create_buttons()

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

    def create_buttons(self):
        """Crear botones de control"""
        # Configurar grid para mejor distribución
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Botón de iniciar con icono
        self.start_button = ctk.CTkButton(
            self,
            text="",
            image=self.play_icon,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=self.primary_color,
            hover_color="#1e8e7f",
            text_color=self.text_color,
            width=50, height=50,
            command=self._on_start_click
        )
        self.start_button.grid(row=0, column=0, padx=10, pady=5)

        # Botón de detener con icono
        self.stop_button = ctk.CTkButton(
            self,
            text="",
            image=self.stop_icon,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=self.disabled_color,
            text_color=self.text_color,
            width=50, height=50,
            command=self._on_stop_click,
            state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=10, pady=5)

        # Botón de recarga con ícono
        self.reload_button = ctk.CTkButton(
            self,
            text="",
            image=self.reload_icon,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#174d55",
            hover_color="#174d55",
            text_color=self.text_color,
            width=50, height=50,            
            command=self._on_reload_click
        )
        self.reload_button.grid(row=0, column=2, padx=10, pady=5)

    def set_callbacks(self, start_callback, stop_callback, reload_callback=None):
        """Establecer callbacks para los botones"""
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.reload_callback = reload_callback

    def _on_start_click(self):
        """Callback interno para botón start"""
        if self.start_callback:
            success = self.start_callback()
            if success:
                self.set_camera_started()

    def _on_stop_click(self):
        """Callback interno para botón stop"""
        if self.stop_callback:
            self.stop_callback()
            self.set_camera_stopped()

    def _on_reload_click(self):
        """Callback interno para botón reload"""
        if self.reload_callback and not self.reload_animation_running:
            success = self.reload_callback()
            if success:
                self._animate_reload_success()

    def _animate_reload_success(self):
        """Animar el botón de reload cuando es exitoso"""
        self.reload_animation_running = True
        
        # Cambiar a color de éxito
        self.reload_button.configure(
            fg_color="#1e8e7f",
            hover_color="#1e8e7f",
            )
        
        self.after(1000, self._start_color_transition)

    def _start_color_transition(self):
        """Iniciar transición gradual de color"""
        # Colores en formato RGB
        success_color = (30, 142, 127)  # #1e8e7f
        original_color = (23, 77, 85)   # #174d55
        
        steps = 15  # Número de pasos para la transición
        current_step = 0
        
        def transition_step():
            nonlocal current_step
            if current_step < steps:
                # Calcular color intermedio
                ratio = current_step / steps
                r = int(success_color[0] + (original_color[0] - success_color[0]) * ratio)
                g = int(success_color[1] + (original_color[1] - success_color[1]) * ratio)
                b = int(success_color[2] + (original_color[2] - success_color[2]) * ratio)
                
                color_hex = f"#{r:02x}{g:02x}{b:02x}"
                self.reload_button.configure(fg_color=color_hex)
                
                current_step += 1
                self.after(100, transition_step)  # 100ms entre pasos
            else:
                # Asegurar color final exacto
                self.reload_button.configure(
                    fg_color="#174d55",
                    hover_color="#174d55",
                    )
                self.reload_animation_running = False
        
        transition_step()

    def set_camera_started(self):
        """Actualizar estado de botones cuando se inicia la cámara"""
        self.start_button.configure(
            state="disabled",
            fg_color=self.disabled_color
        )
        self.stop_button.configure(
            state="normal",
            fg_color=self.primary_color,
            hover_color="#1e8e7f"
        )

    def set_camera_stopped(self):
        """Actualizar estado de botones cuando se detiene la cámara"""
        self.start_button.configure(
            state="normal",
            fg_color=self.primary_color,
            hover_color="#1e8e7f"
        )
        self.stop_button.configure(
            state="disabled",
            fg_color=self.disabled_color
        )