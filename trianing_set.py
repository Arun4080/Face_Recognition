import os
import cv2
import numpy as np
from PIL import Image

EigenFace=cv2.createEigenFaceRecognizer(15)
FisherFace=cv2.createFisherFaceRecognizer(5)
LBPHFace=cv2.createLBPHFaceRecognizer(2, 2, 7, 7)
path='dataset'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
	faceImg=faceImg.resize((110,110))        
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

FisherFace.train(faces, Ids)
FisherFace.save('Recogniser/trainingDataFisher.xml')

LBPHFace.train(faces, Ids)
LBPHFace.save('Recogniser/trainingDataLBPH.xml')

print('All files Saved')

cv2.destroyAllWindows()
