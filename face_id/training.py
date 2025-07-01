import os
from imutils import paths
import face_recognition
import pickle
import cv2

def preprocess_image(image):
    """Preprocesar imagen para mejor calidad de encoding"""
    # Mejorar contraste
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L = clahe.apply(L)
    enhanced = cv2.merge([L, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Reducir ruido
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return denoised

def augment_face_region(image, box):
    """Generar variaciones de la región del rostro para encodings más robustos"""
    top, right, bottom, left = box
    face_region = image[top:bottom, left:right]
    
    variations = [face_region]  # Original
    
    # Variación con rotación ligera
    h, w = face_region.shape[:2]
    center = (w//2, h//2)
    
    # Rotaciones pequeñas (-5 a 5 grados)
    for angle in [-3, 3]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_region, M, (w, h))
        variations.append(rotated)
    
    # Variación con ajuste de brillo
    brightened = cv2.convertScaleAbs(face_region, alpha=1.1, beta=10)
    darkened = cv2.convertScaleAbs(face_region, alpha=0.9, beta=-10)
    variations.extend([brightened, darkened])
    
    return variations

print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))
knownEncodings = []
knownNames = []

# Configuración para un entrenamiento más robusto
MIN_FACES_PER_PERSON = 3  # Mínimo número de caras requeridas por persona
TARGET_ENCODINGS_PER_IMAGE = 2  # Número objetivo de encodings por imagen

for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    
    # Preprocesar imagen para mejor calidad
    processed_image = preprocess_image(image)
    rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Detectar rostros con configuración consistente con reconocimiento
    boxes = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=1)
    
    if len(boxes) == 0:
        print(f"[WARNING] No face found in {imagePath}")
        continue
    
    # Para cada rostro detectado
    for box_idx, box in enumerate(boxes):
        print(f"[INFO] Processing face {box_idx + 1} in {imagePath}")
        
        # Generar encoding principal con más jitters para mayor robustez
        main_encodings = face_recognition.face_encodings(rgb, [box], num_jitters=15, model="large")
        
        if len(main_encodings) > 0:
            # Agregar encoding principal
            knownEncodings.append(main_encodings[0])
            knownNames.append(name)
            
            # Solo crear encoding adicional si no hay demasiados ya
            current_count = sum(1 for n in knownNames if n == name)
            if current_count < TARGET_ENCODINGS_PER_IMAGE:
                # Generar una variación adicional con menos jitters
                additional_encoding = face_recognition.face_encodings(rgb, [box], num_jitters=5, model="large")
                if len(additional_encoding) > 0:
                    knownEncodings.append(additional_encoding[0])
                    knownNames.append(name)

# Validar que tenemos suficiente data para cada persona
print(f"[INFO] Generated {len(knownEncodings)} total encodings")

# Mostrar estadísticas por persona y validar
name_counts = {}
for name in knownNames:
    name_counts[name] = name_counts.get(name, 0) + 1

print("[INFO] Encodings per person:")
valid_training = True
for name, count in name_counts.items():
    print(f"  {name}: {count} encodings")
    if count < MIN_FACES_PER_PERSON:
        print(f"[WARNING] {name} has only {count} encodings (minimum recommended: {MIN_FACES_PER_PERSON})")
        print(f"[WARNING] Consider adding more photos of {name} for better accuracy")
        valid_training = False

if not valid_training:
    print("[WARNING] Training data may be insufficient for reliable recognition")
    print("[WARNING] Consider adding more diverse photos of each person")
else:
    print("[INFO] Training data validation passed")

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Training complete. Encodings saved to 'encodings.pickle'")
print(f"[INFO] Total encodings: {len(knownEncodings)}")