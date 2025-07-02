import io
import base64
import string
from PIL import Image
from flask import Flask, request, jsonify

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ==== Encode / Decode ====
ALL_CHARS = string.digits + string.ascii_uppercase
CHAR_TO_INDEX = {c: i for i, c in enumerate(ALL_CHARS)}
INDEX_TO_CHAR = {i: c for c, i in CHAR_TO_INDEX.items()}

def decode_label(indices):
    return ''.join(INDEX_TO_CHAR[i] for i in indices)

# ==== Model ====
class CaptchaModel(nn.Module):
    def __init__(self, num_chars=36, seq_length=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(128 * 6 * 25, seq_length * num_chars)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1, 4, 36)

# ==== Load model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CaptchaModel().to(device)
model.load_state_dict(torch.load("captcha_model.pth", map_location=device))
model.eval()

# ==== Transform ====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==== Flask App ====
app = Flask(__name__)

@app.route("/", methods=["POST"])
def predict_base64():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({"error": "Missing 'image' field (base64 encoded PNG)"}), 400

    try:
        # Giải mã base64 -> PIL Image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))

        # Tiền xử lý
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Dự đoán
        with torch.no_grad():
            output = model(image_tensor)
            pred = output.argmax(dim=2)[0]
            result = decode_label(pred.cpu().numpy())

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
