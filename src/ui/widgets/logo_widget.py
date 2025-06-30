import io
import cairosvg
import customtkinter as ctk
from PIL import Image


class LogoWidget(ctk.CTkLabel):
    def __init__(self, parent, primary_color="#26a69a", **kwargs):
        super().__init__(
            parent,
            text="[LOGO AQUÍ]",
            font=ctk.CTkFont(size=16),
            text_color=primary_color,
            **kwargs
        )
        self.primary_color = primary_color

    def load_logo(self, image_path):
        """Cargar logo desde archivo SVG o imagen"""
        try:
            if image_path.lower().endswith('.svg'):
                png_data = cairosvg.svg2png(url=image_path, output_width=200, output_height=200)
                pil_image = Image.open(io.BytesIO(png_data))
            else:
                pil_image = Image.open(image_path)
                pil_image.thumbnail((200, 200), Image.Resampling.LANCZOS)

            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
            self.configure(image=ctk_image, text="")
            self.image = ctk_image
        except Exception as e:
            print(f"Error al cargar el logo: {e}")