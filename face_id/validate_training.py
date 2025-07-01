import os
import pickle
import face_recognition
import cv2
import numpy as np
from imutils import paths

def validate_encodings():
    """Validar la calidad de los encodings entrenados"""
    
    # Cargar encodings
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.loads(f.read())
            encodings = data["encodings"]
            names = data["names"]
    except FileNotFoundError:
        print("[ERROR] No se encontró encodings.pickle. Ejecuta training.py primero.")
        return
    
    print(f"[INFO] Validando {len(encodings)} encodings...")
    
    # Estadísticas por persona
    name_counts = {}
    for name in names:
        name_counts[name] = name_counts.get(name, 0) + 1
    
    print("\n[INFO] Estadísticas por persona:")
    for name, count in name_counts.items():
        print(f"  {name}: {count} encodings")
    
    # Análisis de distancias internas (qué tan similares son los encodings de la misma persona)
    print("\n[INFO] Análisis de consistencia interna:")
    for person in name_counts.keys():
        person_encodings = [encodings[i] for i, n in enumerate(names) if n == person]
        
        if len(person_encodings) > 1:
            distances = []
            for i in range(len(person_encodings)):
                for j in range(i + 1, len(person_encodings)):
                    dist = face_recognition.face_distance([person_encodings[i]], person_encodings[j])[0]
                    distances.append(dist)
            
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            min_distance = np.min(distances)
            
            print(f"  {person}:")
            print(f"    Distancia promedio entre encodings: {avg_distance:.3f}")
            print(f"    Distancia máxima: {max_distance:.3f}")
            print(f"    Distancia mínima: {min_distance:.3f}")
            
            if avg_distance > 0.4:
                print("    [WARNING] Distancia promedio alta - considera agregar más fotos consistentes")
            elif avg_distance < 0.2:
                print("    [INFO] Excelente consistencia")
            else:
                print("    [INFO] Buena consistencia")
    
    # Análisis de separación entre personas
    print("\n[INFO] Análisis de separación entre personas:")
    unique_names = list(name_counts.keys())
    
    if len(unique_names) > 1:
        for i in range(len(unique_names)):
            for j in range(i + 1, len(unique_names)):
                person1 = unique_names[i]
                person2 = unique_names[j]
                
                encodings1 = [encodings[k] for k, n in enumerate(names) if n == person1]
                encodings2 = [encodings[k] for k, n in enumerate(names) if n == person2]
                
                # Calcular distancia mínima entre las dos personas
                min_cross_distance = float('inf')
                for enc1 in encodings1:
                    for enc2 in encodings2:
                        dist = face_recognition.face_distance([enc1], enc2)[0]
                        min_cross_distance = min(min_cross_distance, dist)
                
                print(f"  {person1} vs {person2}: distancia mínima = {min_cross_distance:.3f}")
                
                if min_cross_distance < 0.4:
                    print("    [WARNING] Distancia muy baja - pueden confundirse")
                elif min_cross_distance < 0.5:
                    print("    [CAUTION] Distancia moderada - posible confusión ocasional")
                else:
                    print("    [INFO] Buena separación")

def test_recognition_accuracy():
    """Probar la precisión del reconocimiento con las imágenes de entrenamiento"""
    
    # Cargar encodings
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.loads(f.read())
            known_encodings = data["encodings"]
            known_names = data["names"]
    except FileNotFoundError:
        print("[ERROR] No se encontró encodings.pickle.")
        return
    
    print("\n[INFO] Probando precisión con imágenes de entrenamiento...")
    
    # Obtener todas las imágenes del dataset
    imagePaths = list(paths.list_images("dataset"))
    
    correct = 0
    total = 0
    tolerance = 0.45  # Misma tolerancia que en producción
    
    for imagePath in imagePaths:
        true_name = imagePath.split(os.path.sep)[-2]
        
        # Cargar y procesar imagen
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encontrar rostros
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes, model="large")
        
        for encoding in encodings:
            total += 1
            
            # Aplicar la misma lógica que en producción
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            min_distance = np.min(face_distances)
            best_match_index = np.argmin(face_distances)
            
            predicted_name = "Unknown"
            
            if min_distance <= tolerance:
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=tolerance)
                
                if matches[best_match_index]:
                    candidate_name = known_names[best_match_index]
                    
                    # Contar coincidencias
                    same_person_indices = [i for i, n in enumerate(known_names) if n == candidate_name]
                    same_person_matches = [matches[i] for i in same_person_indices]
                    match_ratio = sum(same_person_matches) / len(same_person_matches)
                    
                    if match_ratio >= 0.5:
                        predicted_name = candidate_name
            
            if predicted_name == true_name:
                correct += 1
            else:
                print(f"[ERROR] {imagePath}: Esperado '{true_name}', obtuvo '{predicted_name}' (distancia: {min_distance:.3f})")
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n[INFO] Precisión en dataset de entrenamiento: {accuracy:.1f}% ({correct}/{total})")
    
    if accuracy < 90:
        print("[WARNING] Precisión baja - considera:")
        print("  - Agregar más fotos variadas de cada persona")
        print("  - Verificar la calidad de las fotos existentes")
        print("  - Asegurar buena iluminación y ángulos diversos")

if __name__ == "__main__":
    print("=== VALIDACIÓN DE ENTRENAMIENTO ===")
    validate_encodings()
    test_recognition_accuracy()
    
    print("\n=== RECOMENDACIONES ===")
    print("1. Para mejorar la precisión:")
    print("   - Agrega al menos 5-10 fotos por persona")
    print("   - Usa fotos con diferentes ángulos, iluminación y expresiones")
    print("   - Evita fotos borrosas o de muy baja calidad")
    print("2. Si ves confusión entre personas:")
    print("   - Verifica que las fotos estén correctamente etiquetadas")
    print("   - Agrega más fotos distintivas de cada persona")
    print("3. Para reducir falsos positivos:")
    print("   - El sistema ahora usa tolerancia más estricta (0.45)")
    print("   - Requiere múltiples encodings que coincidan")
