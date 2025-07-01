import cv2
import os
import time
from datetime import datetime

def capture_training_photos(person_name, num_photos=10):
    """
    Capturar fotos de entrenamiento con indicaciones para mejorar la diversidad
    """
    
    # Crear directorio si no existe
    dataset_dir = os.path.join("dataset", person_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return False
    
    print(f"[INFO] Capturando {num_photos} fotos para {person_name}")
    print("[INFO] Sigue las instrucciones en pantalla para obtener fotos diversas")
    print("[INFO] Presiona ESPACIO para tomar foto, 'q' para salir")
    
    # Lista de poses/instrucciones para diversidad
    instructions = [
        "Mira directamente a la cámara",
        "Inclina ligeramente la cabeza a la izquierda",
        "Inclina ligeramente la cabeza a la derecha", 
        "Mira ligeramente hacia arriba",
        "Mira ligeramente hacia abajo",
        "Sonríe naturalmente",
        "Expresión seria",
        "Aleja tu cara de la cámara",
        "Acerca tu cara a la cámara",
        "Pose natural como prefieras"
    ]
    
    photos_taken = 0
    current_instruction = 0
    
    while photos_taken < num_photos:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] No se pudo capturar frame")
            break
        
        # Mostrar frame con instrucciones
        display_frame = frame.copy()
        
        # Agregar texto con instrucciones
        instruction = instructions[current_instruction % len(instructions)]
        cv2.putText(display_frame, f"Foto {photos_taken + 1}/{num_photos}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, instruction, 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "ESPACIO: tomar foto | Q: salir", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Dibujar marco guía para el rostro
        h, w = display_frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        face_w, face_h = 200, 240
        
        # Rectángulo guía
        cv2.rectangle(display_frame, 
                     (center_x - face_w//2, center_y - face_h//2),
                     (center_x + face_w//2, center_y + face_h//2),
                     (0, 255, 0), 2)
        cv2.putText(display_frame, "Centra tu rostro aqui", 
                   (center_x - 100, center_y - face_h//2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Captura de Fotos de Entrenamiento', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Espacio para tomar foto
            # Generar nombre de archivo único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_name}_{timestamp}_{photos_taken + 1:02d}.jpg"
            filepath = os.path.join(dataset_dir, filename)
            
            # Guardar foto original (sin texto)
            cv2.imwrite(filepath, frame)
            
            print(f"[INFO] Foto guardada: {filename}")
            photos_taken += 1
            current_instruction += 1
            
            # Pausa breve para evitar fotos duplicadas accidentales
            time.sleep(0.5)
            
        elif key == ord('q'):
            print("[INFO] Captura cancelada por el usuario")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"[INFO] Captura completa: {photos_taken} fotos guardadas para {person_name}")
    print(f"[INFO] Fotos guardadas en: {dataset_dir}")
    
    if photos_taken >= 5:
        print("[INFO] Ahora ejecuta training.py para actualizar el modelo")
        return True
    else:
        print("[WARNING] Se necesitan al menos 5 fotos para un buen entrenamiento")
        return False

def list_existing_people():
    """Listar personas existentes en el dataset"""
    dataset_dir = "dataset"
    
    if not os.path.exists(dataset_dir):
        print("[INFO] No existe el directorio dataset")
        return []
    
    people = []
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            photo_count = len([f for f in os.listdir(item_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            people.append((item, photo_count))
    
    return people

def main():
    print("=== CAPTURA DE FOTOS DE ENTRENAMIENTO ===")
    print()
    
    # Mostrar personas existentes
    existing_people = list_existing_people()
    
    if existing_people:
        print("Personas existentes en el dataset:")
        for name, count in existing_people:
            print(f"  {name}: {count} fotos")
        print()
    
    # Solicitar nombre de persona
    person_name = input("Ingresa el nombre de la persona (o 'salir' para terminar): ").strip()
    
    if person_name.lower() in ['salir', 'exit', 'quit', '']:
        print("Saliendo...")
        return
    
    # Validar nombre
    if not person_name.replace('_', '').replace('-', '').isalnum():
        print("[ERROR] El nombre solo debe contener letras, números, guiones y guiones bajos")
        return
    
    # Verificar si ya existe
    existing_count = 0
    for name, count in existing_people:
        if name.lower() == person_name.lower():
            existing_count = count
            break
    
    if existing_count > 0:
        print(f"[INFO] {person_name} ya tiene {existing_count} fotos")
        add_more = input("¿Agregar más fotos? (s/n): ").strip().lower()
        if add_more not in ['s', 'si', 'yes', 'y']:
            return
    
    # Solicitar número de fotos
    try:
        num_photos = int(input("¿Cuántas fotos capturar? (recomendado: 8-12): ") or "10")
        if num_photos < 1 or num_photos > 50:
            print("[ERROR] Número de fotos debe estar entre 1 y 50")
            return
    except ValueError:
        print("[ERROR] Número inválido")
        return
    
    print()
    print("CONSEJOS PARA BUENAS FOTOS DE ENTRENAMIENTO:")
    print("- Asegúrate de tener buena iluminación")
    print("- Evita sombras fuertes en el rostro")
    print("- Mantén el rostro centrado en el marco")
    print("- Sigue las instrucciones en pantalla para variedad")
    print("- No uses gafas de sol o accesorios que oculten el rostro")
    print()
    
    input("Presiona ENTER cuando estés listo para comenzar...")
    
    # Capturar fotos
    success = capture_training_photos(person_name, num_photos)
    
    if success:
        print()
        print("PRÓXIMOS PASOS:")
        print("1. Ejecuta 'python face_id/training.py' para entrenar el modelo")
        print("2. Ejecuta 'python face_id/validate_training.py' para validar el entrenamiento")
        print("3. Prueba el reconocimiento con tu aplicación")

if __name__ == "__main__":
    main()
