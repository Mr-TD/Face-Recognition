import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'DemoDataset'


def GetImageId(path):
    imgpath = [os.path.join(path, file) for file in os.listdir(path)]
    image = []
    ids = []
    for photo in imgpath:
        img = Image.open(photo).convert('L')
        imgnp = np.array(img, dtype='uint8')
        id = int(os.path.split(photo)[1].split('.')[1])
        image.append(imgnp)
        ids.append(id)
        print(id)
    return ids, image


ids,image = GetImageId(path)
recognizer.train(image, np.array(ids))
recognizer.write('train.yml')
