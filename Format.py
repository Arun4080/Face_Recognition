import os
import glob
import cv2
a=glob.glob("dataset/.jpg")
for i in a:
    print i
    img=cv2.imread(""+i,cv2.IMREAD_GRAYSCALE)
    c=cv2.resize(img, (190,190))
    cv2.imwrite("a/"+str(i)+".jpg", c)
    print "converted %s"%i
