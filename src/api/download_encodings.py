import requests
import os
from datetime import datetime

path = 'src/encodings/'

def download_encodings_from_api(api_url="http://localhost:8000/api/face_attendance/v1/download"):
    """
    Descarga el archivo de encodings desde la API y lo guarda en el path especificado
    
    Args:
        api_url (str): URL de la API para descargar los encodings
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    try:
        # Asegurar que el directorio existe
        os.makedirs(path, exist_ok=True)
        
        print(f"[INFO] Conectando a la API: {api_url}")
        
        # Realizar la petición GET a la API
        response = requests.get(api_url, timeout=30)
        
        # Verificar que la respuesta sea exitosa
        response.raise_for_status()
        
        # Obtener el nombre del archivo desde los headers o usar uno por defecto
        filename = "encodings.pickle"
        
        # Verificar si hay un header Content-Disposition para obtener el nombre del archivo
        if 'Content-Disposition' in response.headers:
            content_disposition = response.headers['Content-Disposition']
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')
        
        # Ruta completa donde guardar el archivo
        file_path = os.path.join(path, filename)
        
        # Crear backup del archivo existente si existe
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(path, f"encodings_backup_{timestamp}.pickle")
            os.rename(file_path, backup_path)
            print(f"[INFO] Archivo existente respaldado como: {backup_path}")
        
        # Guardar el archivo descargado
        with open(file_path, 'wb') as file:
            file.write(response.content)
        
        print(f"[INFO] Archivo descargado exitosamente: {file_path}")
        print(f"[INFO] Tamaño del archivo: {len(response.content)} bytes")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] No se pudo conectar a la API. Verifica que el servidor esté ejecutándose.")
        return False
        
    except requests.exceptions.Timeout:
        print("[ERROR] Tiempo de espera agotado al conectar con la API.")
        return False
        
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Error HTTP al descargar el archivo: {e}")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error en la petición: {e}")
        return False
        
    except OSError as e:
        print(f"[ERROR] Error al guardar el archivo: {e}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Error inesperado: {e}")
        return False

def download_encodings_with_retry(max_retries=3, api_url="http://localhost:8000/api/face_attendance/v1/download"):
    """
    Descarga los encodings con reintentos en caso de fallo
    
    Args:
        max_retries (int): Número máximo de reintentos
        api_url (str): URL de la API para descargar los encodings
        
    Returns:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    for attempt in range(1, max_retries + 1):
        print(f"[INFO] Intento {attempt} de {max_retries}")
        
        if download_encodings_from_api(api_url):
            return True
        
        if attempt < max_retries:
            print("[INFO] Esperando antes del siguiente intento...")
            import time
            time.sleep(2)  # Esperar 2 segundos antes del siguiente intento
    
    print("[ERROR] Falló la descarga después de todos los intentos")
    return False

if __name__ == "__main__":
    # Ejemplo de uso
    success = download_encodings_with_retry()
    if success:
        print("[SUCCESS] Encodings descargados exitosamente")
    else:
        print("[FAILED] No se pudieron descargar los encodings")