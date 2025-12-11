from PIL import Image
import numpy as np

# Nome del file che ti hanno passato
IMAGE_PATH = "Chest_Xray_PA_3-8-2010.png"  # <--- Assicurati che il nome sia giusto

try:
    # 1. Carica l'immagine
    img = Image.open(IMAGE_PATH)
    
    print("--- REPORT IMMAGINE ---")
    print(f"Nome file: {IMAGE_PATH}")
    print(f"Formato: {img.format}")      # Es: PNG, JPEG, TIFF
    print(f"Dimensioni originali: {img.size}") # Es: (1024, 1024)
    print(f"Modalità colore: {img.mode}") # Es: L (Grigio), RGB, RGBA, I (16bit)
    
    # 2. Controlliamo i valori dei pixel
    img_array = np.array(img)
    print(f"Valore Minimo pixel: {img_array.min()}")
    print(f"Valore Massimo pixel: {img_array.max()}")
    print(f"Tipo dati numpy: {img_array.dtype}") # Es: uint8, uint16
    
    # 3. Verifica visuale veloce
    img.show() # Apre l'immagine col visualizzatore di default del tuo PC

except FileNotFoundError:
    print(f"ERRORE: Non trovo il file '{IMAGE_PATH}' nella cartella.")