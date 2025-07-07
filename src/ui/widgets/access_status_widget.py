import customtkinter as ctk


class AccessStatusWidget(ctk.CTkFrame):
    def __init__(self, parent, primary_color="#26a69a", denied_color="#b71c1c", **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.primary_color = primary_color
        self.denied_color = denied_color
        
        # Título del sistema
        self.title_label = ctk.CTkLabel(
            self,
            text="SISTEMA DE CONTROL DE ACCESO CON INTELIGENCIA ARTIFICIAL (VISIÓN)",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#ffffff",
            wraplength=800,  # Permitir salto de línea si es necesario
            justify="center"
        )
        self.title_label.pack(pady=(0, 15))
        
        # Estado de acceso
        self.status_label = ctk.CTkLabel(
            self,
            text="ESTADO: ESPERANDO DETECCIÓN",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#888888"
        )
        self.status_label.pack(pady=(0, 10))
        
        # Estado inicial
        self.current_status = "waiting"
    
    def set_access_granted(self, person_name="Usuario"):
        """Mostrar estado de acceso permitido"""
        self.status_label.configure(
            text=f"ACCESO PERMITIDO - {person_name.upper()}",
            text_color=self.primary_color
        )
        self.current_status = "granted"
    
    def set_access_denied(self):
        """Mostrar estado de acceso denegado"""
        self.status_label.configure(
            text="ACCESO DENEGADO - PERSONA NO RECONOCIDA",
            text_color=self.denied_color
        )
        self.current_status = "denied"
    
    def set_waiting_status(self):
        """Mostrar estado de espera"""
        self.status_label.configure(
            text="ESTADO: ESPERANDO DETECCIÓN",
            text_color="#888888"
        )
        self.current_status = "waiting"
    
    def get_status(self):
        """Obtener estado actual"""
        return self.current_status
