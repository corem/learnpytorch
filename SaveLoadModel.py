import torch
import torch.onnx as onnx
import torchvision.models as models

# Load pretrained model
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Save model
torch.save(model, 'model.pth')

# Load saved model
model = torch.load('model.pth')

# Export in ONNX standard format
input_image = torch.zeros((1,3,224,224))
onnx.export(model, input_image, 'model.onnx')