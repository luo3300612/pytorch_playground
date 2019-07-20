import torchvision
import torch
import PIL
import cv2

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

img = cv2.imread('timg.jpeg')

image_tensor = torchvision.transforms.functional.to_tensor(img)
predictions = model([image_tensor])

print(predictions)
print(predictions[0].keys())