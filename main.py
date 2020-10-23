from cv2 import cv2

# Must pass valid file directory
img = cv2.imread('images/man.png') 

# Converting the image into grayscale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

face_cascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")

faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (255, 255, 250),3)


# Displaying the image
cv2.imshow("ImageWindow", img) 

# WaitKey is set to 0, 
# that means the image window will close as soon as any key is pressed.
cv2.waitKey(0)
cv2.destroyAllWindows()