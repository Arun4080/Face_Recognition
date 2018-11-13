import glob
import os
import cv2
import numpy as np
from PIL import Image

EigenFace=cv2.face.EigenFaceRecognizer_create(15)
path='dataset'

def getImagesWithID(path):
    imagePaths=glob.glob('dataset/*.jpg')
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        print(imagePath)
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(1)
    return np.array(IDs),faces

Ids,faces=getImagesWithID(path)
print('Training.......')
EigenFace.train(faces, Ids)
EigenFace.save('Recogniser/trainingDataEigan.xml')

print('All files Saved')

cv2.destroyAllWindows()
