import cv2 
import pickle
width,height =107, 42

try:
    with open("coordinate/CarParkPos","rb") as f:
        points = pickle.load(f)
except:
    points=[]

def mouse_click(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
    elif  event == cv2.EVENT_RBUTTONDOWN:
        for idx,point in enumerate(points):
            x1,y1 = point
            if x1<x<x1+width and y1<y<y1+height:
                points.pop(idx)
                break
    with open("Car_coordiante","wb") as f:
        pickle.dump(points,f)



while True:
    image = cv2.imread("carParkImg.png")
    for point in points:
        cv2.rectangle(image,point,(point[0]+width,point[1]+height),(0,0,255),2)
    cv2.imshow("image",image)
    cv2.setMouseCallback("image",mouse_click)
    cv2.waitKey(1)

    