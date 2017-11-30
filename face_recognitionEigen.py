import cv2

faceDetect= cv2.CascadeClassifier('Haar//haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
rec=cv2.createEigenFaceRecognizer()
rec.load("Recogniser/trainingDataEigan.xml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	id,conf=rec.predict(gray[y:y+h,x:x+w])
	if id==0:id="Arun"
	elif id==2:id="Akash"
	else: id="Unknown"
	cv2.cv.PutText(cv2.cv.fromarray(img), str(id),(x+w,x+w+1),font,255)
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
	break;    


cap.release()
cv2.destroyAllWindows()
