import customtkinter as ctk


class TestControlButtonsWidget(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="#ff0000", corner_radius=10, height=100, **kwargs)
        self.pack_propagate(False)
        
        # Label de prueba para verificar que el widget sea visible
        test_label = ctk.CTkLabel(
            self,
            text="BOTONES DE CONTROL - WIDGET DE PRUEBA",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        )
        test_label.pack(expand=True)
        
        # Botones de prueba simples
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=10)
        
        button1 = ctk.CTkButton(button_frame, text="START", width=80, height=40)
        button1.pack(side="left", padx=5)
        
        button2 = ctk.CTkButton(button_frame, text="STOP", width=80, height=40)
        button2.pack(side="left", padx=5)
        
        button3 = ctk.CTkButton(button_frame, text="RELOAD", width=80, height=40)
        button3.pack(side="left", padx=5)
        
        button4 = ctk.CTkButton(button_frame, text="DOWNLOAD", width=80, height=40)
        button4.pack(side="left", padx=5)
