import cv2

# Open the video file
cap = cv2.VideoCapture('carPark.mp4')

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
print(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # taking the number of frames in one video
# Read and display the video frames
while True:
    # Capture frame-by-frame
    # setting video image
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == cap.get(cv2.CAP_PROP_POS_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    ret, frame = cap.read()

    # If the frame was read successfully
    if ret:
        # Display the frame
        cv2.imshow('Video', frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # blur_image= cv2.GaussianBlur(gray,(5,5),1)
        blur_image = cv2.bilateralFilter(gray,9,65,65)
        adaptive = cv2.adaptiveThreshold(blur_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,16)
        cv2.imshow("gray",gray)
        cv2.imshow("blur",blur_image)
        cv2.imshow("adaptive",adaptive)
        cv2.waitKey(10)
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()