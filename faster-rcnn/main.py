import torchvision
import torch
import PIL
import cv2

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

img = cv2.imread('timg.jpeg')
# image = PIL.Image.open('timg.jpeg')

image_tensor = torchvision.transforms.functional.to_tensor(img)
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

predictions = model([image_tensor])

res = torchvision.ops.nms(predictions[0]['boxes'], predictions[0]['scores'], iou_threshold=0.4)

print(res)

for box in predictions[0]['boxes'][res]:
    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))

cv2.imshow("Window1", img)
cv2.waitKey(0)
print(len(predictions))
print(predictions)
