import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model_name = "patrickjohncyh/fashion-clip"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# create dummy image
img = Image.new('RGB', (224, 224), color = 'red')

inputs = processor(images=[img], return_tensors="pt", padding=True)
print("Keys in inputs:", inputs.keys())

with torch.no_grad():
    feats = model.get_image_features(**inputs)

print("Type of feats:", type(feats))
if hasattr(feats, 'keys'):
    print("Keys in feats:", feats.keys())
elif hasattr(feats, 'shape'):
    print("Shape of feats:", feats.shape)
else:
    print("Dir feats:", dir(feats))
