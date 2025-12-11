import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Resize,
    ScaleIntensity,
    ToTensor,
)
from monai.networks.layers import Norm

# --- CONFIGURAZIONE ---
MODEL_PATH = "best_metric_model.pth" # Pesi da utilizzare 
IMAGE_PATH = "Chest_Xray_PA_3-8-2010.png" # Immagine da testare
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Dove far girare il modello

print(f"Usando dispositivo: {DEVICE}")

# --- 1. DEFINIZIONE ARCHITETTURA ---
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,   
).to(DEVICE)

# --- 2. CARICAMENTO PESI ---
try:
    print("Caricamento pesi modello...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Modalità valutazione (congela i pesi)
    print("✅ Modello caricato con successo.")
except FileNotFoundError:
    print(f"❌ ERRORE: Non trovo '{MODEL_PATH}'.")
    exit()

# --- 3. PRE-PROCESSING DELL'IMMAGINE ---
# Trasformo l'immagine PNG in ciò che la rete si aspetta (Tensore 1x1x256x256)
def preprocess(img_path):
    # Apre e converte forzatamente in scala di grigi (L)
    img_pil = Image.open(img_path).convert("L")
    img_np = np.array(img_pil)
    
    # Pipeline di trasformazione MONAI (uguale alla validazione)
    transforms = Compose([
        ToTensor(),
        EnsureChannelFirst(channel_dim="no_channel"), # (H, W) -> (1, H, W)
        Resize((IMG_SIZE, IMG_SIZE)),                 # Ridimensiona a 256x256
        ScaleIntensity(),                             # Normalizza tra 0 e 1
    ])
    
    input_tensor = transforms(img_np)
    # Aggiunge la dimensione del batch: (1, H, W) -> (1, 1, H, W)
    return input_tensor.unsqueeze(0), img_pil

# --- 4. INFERENZA ---
try:
    input_tensor, original_pil = preprocess(IMAGE_PATH)
    input_tensor = input_tensor.to(DEVICE)
    
    print("Esecuzione inferenza...")
    with torch.no_grad():
        # Passaggio nella rete
        logits = model(input_tensor)
        
        # Post-processing: Sigmoide -> Soglia 0.5 -> Float
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()

    # Prepara per la visualizzazione (rimuove dimensioni batch/canale)
    # Porta su CPU e converte in numpy
    mask_np = mask.squeeze().cpu().numpy()
    
    print("✅ Segmentazione completata.")

    # --- 5. VISUALIZZAZIONE ---
    plt.figure(figsize=(12, 6))

    # Immagine originale (ridimensiono per confronto)
    plt.subplot(1, 3, 1)
    plt.title("Input X-Ray (Resized)")
    # Converto l'input tensor in numpy per vederlo
    input_show = input_tensor.squeeze().cpu().numpy()
    plt.imshow(input_show, cmap="gray")
    plt.axis("off")

    # Maschera Predetta
    plt.subplot(1, 3, 2)
    plt.title("AI Predicted Mask")
    plt.imshow(mask_np, cmap="gray")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(input_show, cmap="gray")
    plt.imshow(mask_np, cmap="jet", alpha=0.5) # Maschera semitrasparente
    plt.axis("off")

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"❌ ERRORE: Immagine '{IMAGE_PATH}' non trovata.")
except Exception as e:
    print(f"❌ ERRORE IMPREVISTO: {e}")