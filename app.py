import cv2
from utils import get_parking_spots_bboxes
from load_and_save import load_checkpoint
import torch
import numpy as np
from model import architecture
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
LOAD_MODEL_FILE = "weights\model_car_parking.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5
detection = architecture()
load_checkpoint(torch.load(LOAD_MODEL_FILE), detection)
detection.to(DEVICE)

def available_parking_checking(sub_frame:torch.Tensor):
    predict = detection(sub_frame)
    return predict.detach().cpu().numpy()>THRESHOLD

def crop_image(frame,bboxes):
    crop_list = np.zeros((len(bboxes),3,32,69))
    # converting frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    for idx,bbox in enumerate(bboxes):
        sub_frame = cv2.resize(rgb_frame[bbox[1]:bbox[3],bbox[0]:bbox[2]],(69,32))
        crop_list[idx]=sub_frame.reshape(-1,32,69)
    return crop_list.astype(np.float32)

def diff(img1,img2):
    img1 = img1.reshape(img1.shape[0],-1)
    img2 = img2.reshape(img2.shape[0],-1)
    return abs(np.mean(img1,axis=-1)-np.mean(img2,axis=-1))

mask = 'mask\mask_1920_1080.png'
cap = cv2.VideoCapture("Data\data\parking_1920_1080_loop.mp4")
mask = cv2.imread(mask, 0)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

bboxes = get_parking_spots_bboxes(connected_components)

print(len(bboxes))

count = 0
ratio = 29
prev_crop_images= np.array([])
percentile_threshold = 97
changing_threshold = 0.6
predict_value = None
while True:
    rep,frame = cap.read()
    if rep:
        if count%ratio==0:
            crop_images = crop_image(frame,bboxes) # tensor
            count=0
            if len(prev_crop_images)==0:
                predict_value = available_parking_checking(torch.tensor(crop_images,device=DEVICE))
            else:
                differ = diff(crop_images,prev_crop_images)
                changing_box = np.nonzero(differ/np.max(differ,axis=0)>changing_threshold)[0]
                predict_value[changing_box] = available_parking_checking(torch.tensor(crop_images[changing_box],device=DEVICE))
            prev_crop_images = crop_images
        for idx,bbox in enumerate(bboxes):
            if predict_value[idx]:
                cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3)
            else:
                cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
        count+=1
        cv2.rectangle(frame,(0,0),(700,80),(0,0,0),-1)
        cv2.putText(frame,f"Available parking spots: {np.count_nonzero(predict_value)}/{len(predict_value)}",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow("frame",frame)
        cv2.waitKey(10)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()