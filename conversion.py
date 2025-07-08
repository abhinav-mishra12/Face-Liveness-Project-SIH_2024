import torch
import torchvision.models as models

# Load the pretrained MobileNetV2 model
model = models.mobilenet_v2()

# Modify the classifier to match the saved model's classifier (2 output classes)
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=2)

# Load the saved state dict (weights)
model.load_state_dict(torch.load('C:/Users/Aman/SIH Hackathon/Liveness_weights.pth', map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Dummy input for model tracing
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "C:/Users/Aman/SIH Hackathon/Liveness.onnx",  # Save path for the ONNX model
    input_names=["input"], 
    output_names=["output"], 
    opset_version=11
)

print("Model successfully exported to ONNX format!")
