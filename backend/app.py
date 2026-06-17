
##Load the model  
import torch
import timm
import os
from PIL import Image
from torchvision import transforms


from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=2
)

model.eval()
model.to(device)

# Load model weights if file exists and is not empty
model_path = "deepguard_efficientnet.pth"
if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

#Step 3: Create Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])


## Step 4: Create Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():

    try:

        if 'image' not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        image_file = request.files['image']

        image = Image.open(image_file).convert('RGB')

        image = transform(image)

        image = image.unsqueeze(0)

        image = image.to(device)

        with torch.no_grad():

            output = model(image)

            probs = torch.softmax(output, dim=1)

            confidence, prediction = torch.max(
                probs,
                dim=1
            )

        result = (
            "AI Generated"
            if prediction.item() == 1
            else "Real Image"
        )

        return jsonify({
            "prediction": result,
            "confidence": round(
                confidence.item() * 100,
                2
            )
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

@app.route("/")
def home():
    return "DeepGuard Backend Running"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)