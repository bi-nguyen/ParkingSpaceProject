from load_and_save import load_checkpoint
from model import architecture
import cv2
import torch
from PIL import Image
import numpy as np
import os
LOAD_MODEL_FILE = "weights\model_car_parking.pth.tar"
detection = architecture()
load_checkpoint(torch.load(LOAD_MODEL_FILE), detection)
image = cv2.imread("clf-data\empty\\00000000_00000161.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
# model.requires_grad_(False)
image = cv2.resize(image, (69,32)) # width,height (69,29)
image_tensor = torch.from_numpy(image).to(torch.float32).unsqueeze(0).permute(0,3,1,2)

print(detection(image_tensor).item())
count = 0
for i in os.listdir("clf-data\empty"):
    image = cv2.imread(f"clf-data\empty\\{i}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (69,32))
    image_tensor = torch.from_numpy(image).to(torch.float32).unsqueeze(0).permute(0,3,1,2)
    if detection(image_tensor).item()<0.5:
        count+=1
print(count/len(os.listdir("clf-data\empty")))

    

