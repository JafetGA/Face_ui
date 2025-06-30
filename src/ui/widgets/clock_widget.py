import customtkinter as ctk
from datetime import datetime
import pytz


class ClockWidget(ctk.CTkLabel):
    def __init__(self, parent, primary_color="#26a69a", **kwargs):
        super().__init__(
            parent,
            text="00:00:00",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=primary_color,
            **kwargs
        )
        self.primary_color = primary_color
        self.update_clock()

    def update_clock(self):
        """Actualizar reloj cada segundo"""
        try:
            now = datetime.now(pytz.timezone('America/Mexico_City'))
        except:
            now = datetime.now()

        self.configure(text=now.strftime("%H:%M:%S"))
        self.after(1000, self.update_clock)