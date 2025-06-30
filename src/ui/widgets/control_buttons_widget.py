import customtkinter as ctk


class ControlButtonsWidget(ctk.CTkFrame):
    def __init__(self, parent, primary_color="#26a69a", disabled_color="#174d55", text_color="#0f172a", **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.primary_color = primary_color
        self.disabled_color = disabled_color
        self.text_color = text_color

        # Callbacks
        self.start_callback = None
        self.stop_callback = None

        self.create_buttons()

    def create_buttons(self):
        """Crear botones de control"""
        self.start_button = ctk.CTkButton(
            self,
            text="Iniciar Cámara",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.primary_color,
            hover_color="#1e8e7f",
            text_color=self.text_color,
            width=120, height=35,
            command=self._on_start_click
        )
        self.start_button.pack(side="left", padx=10)

        self.stop_button = ctk.CTkButton(
            self,
            text="Detener Cámara",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.disabled_color,
            text_color=self.text_color,
            width=120, height=35,
            command=self._on_stop_click,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=10)

    def set_callbacks(self, start_callback, stop_callback):
        """Establecer callbacks para los botones"""
        self.start_callback = start_callback
        self.stop_callback = stop_callback

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

    def set_camera_started(self):
        """Actualizar estado de botones cuando se inicia la cámara"""
        self.start_button.configure(
            state="disabled",
            fg_color=self.disabled_color,
            text_color=self.text_color
        )
        self.stop_button.configure(
            state="normal",
            fg_color=self.primary_color,
            hover_color="#1e8e7f",
            text_color=self.text_color
        )

    def set_camera_stopped(self):
        """Actualizar estado de botones cuando se detiene la cámara"""
        self.start_button.configure(
            state="normal",
            fg_color=self.primary_color,
            hover_color="#1e8e7f",
            text_color=self.text_color
        )
        self.stop_button.configure(
            state="disabled",
            fg_color=self.disabled_color,
            text_color=self.text_color
        )