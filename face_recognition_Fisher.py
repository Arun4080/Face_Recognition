import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognise=cv2.createFisherFaceRecognizer(5,600)
recognise.load("Recogniser/trainingDataFisher.xml")

cap=cv2.VideoCapture(0)
id=0
while True:
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
	gray_face=cv2.resize((gray[y:y+h,x:x+h]),(100,100))
    	id,conf = recognise.predict(gray_face)
        if id==0:id="Arun"
	    elif id==2:id="Akash"
	    else: id="Unknown"
	cv2.cv.PutText(cv2.cv.fromarray(img), str(id),(x+w,x+w+1),font,255)
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
	break;
cap.release()
cv2.destroyAllWindows()
