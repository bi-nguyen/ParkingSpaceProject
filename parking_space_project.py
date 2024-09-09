import cv2  
import pickle
import numpy as np
width,height = 107, 42
location_box = None
try:
    with open("coordinate/CarParkPos","rb") as f:
        location_box = pickle.load(f)
except:
    print("The file is not exist.")

cap = cv2.VideoCapture("carPark.mp4")

def drawing_boudingbox(frame,gray_frame,bboxes,threshold=300):
    count = 0
    for pos in bboxes:
        crop_image = gray_frame[pos[1]:pos[1]+height,pos[0]:pos[0]+width]
        cv2.rectangle(frame,pos,(pos[0]+width,pos[1]+height),(0,0,255),2)
        value = cv2.countNonZero(crop_image)
        cv2.putText(frame,str(value),(pos[0],pos[1]+height-6),cv2.FONT_HERSHEY_PLAIN,1,(50,180,200),2)
        if value<300:
            count+=1
    cv2.rectangle(frame,(0,0),(250,50),(0,0,0),-1)
    cv2.putText(frame,f"Available spots : {count}/{len(crop_image)}",(25,25),cv2.FONT_HERSHEY_PLAIN,1,(50,180,200),2)
    



while True:
    ret,frame = cap.read()
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == cap.get(cv2.CAP_PROP_POS_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    if ret:
        # convert image to binary
        binary_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # smoothing image 
        smoothing_image= cv2.GaussianBlur(binary_image,(9,9),0)
        # applying thresold
        thresold_image = cv2.adaptiveThreshold(smoothing_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV,25,16)
        # kernel = np.ones((3, 3), np.uint8)
        # imgThres = cv2.dilate(thresold_image, kernel, iterations=1)
        drawing_boudingbox(frame,thresold_image,location_box)
        
        cv2.imshow("smoothing_image",thresold_image)
        cv2.imshow("image",frame)
        
        cv2.waitKey(10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

