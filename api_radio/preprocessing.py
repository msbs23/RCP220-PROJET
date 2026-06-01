from torchvision import transforms
from PIL import Image
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # radio en niveaux de gris → 3 canaux
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # valeurs ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Prend une image en bytes (reçue via l'API)
    Retourne un tensor prêt pour le modèle
    """
    from io import BytesIO
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0)  # ajoute la dimension batch → [1, 3, 224, 224]