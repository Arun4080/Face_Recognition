import cv2

faceDetect= cv2.CascadeClassifier('Haar//haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
rec=cv2.face.EigenFaceRecognizer_create(15)
rec.read("Recogniser/trainingDataEigan.xml")
id=0
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(cv2.resize(gray[y:y+h,x:x+w], (190,190)))
        # Change ID names as per given 
        if id==1:id="Arun"
        elif id==2:id="xyz"
        else: id="Unknown"
        cv2.putText(img, str(id),(x+w,y),font,1,(255,0,0))
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
        break  
cap.release()
cv2.destroyAllWindows()
