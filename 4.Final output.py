import cv2
import numpy as np
import os
from gtts import gTTS

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cnn_sgn

IMG_SIZE = 96
LR = 1e-3

nb_classes=28

MODEL_NAME = 'handsign.model'

model=cnn_sgn.cnn_model()

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

# organize imports
import cv2
import imutils
import numpy as np

from collections import Counter

import time

            # 0    1       2     3      4      5       6       7         8
out_label=['C', 'FOUR', 'HAND', 'L', 'LIKE', 'OK', 'THREE', 'VICTORY','ZERO']
pre=[]

s=''
cchar=[0,0]
c1=''

aWeight = 0.5


camera = cv2.VideoCapture(0)


top, right, bottom, left = 170, 150, 425, 450


num_frames = 0

flag=0
flag1=0

timer=0
temp_list = []
res_ = ''
while(True):

    (grabbed, frame) = camera.read()


    frame = imutils.resize(frame, width=700)


    frame = cv2.flip(frame, 1)

    clone = frame.copy()

    (height, width) = frame.shape[:2]

    roi = frame[top:bottom, right:left]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    
    # cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
    
    img=gray

    
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    test_data =img

    orig = img
    data = img.reshape(IMG_SIZE,IMG_SIZE,1)

    model_out = model.predict([data])[0]

    model_out = model.predict([data])[0]

    pnb=np.argmax(model_out)
    print(str(np.argmax(model_out))+" "+str(out_label[pnb]))

    pre.append(out_label[pnb]) 

    temp_list.append(str(out_label[pnb]))
    timer+=1
    if(timer%10 == 0):
        m = 0
        res_ = temp_list[0] 
        for i in temp_list: 
            freq = temp_list.count(i) 
            if freq > m: 
                m = freq 
                res_ = i 
        temp_list = []
    cv2.putText(clone,
           '%s ' % (res_),
           (450, 150), cv2.FONT_HERSHEY_PLAIN,5,(255, 0, 0))


    cv2.rectangle(clone, (110, 70), (300, 250), (0,0,255), 4)

    cv2.putText(clone,
                   '%s ' % (str(s)),
                   (10, 60), cv2.FONT_HERSHEY_PLAIN,3,(0, 0, 0))

    num_frames += 1

    cv2.imshow("Video Feed", clone)


    keypress = cv2.waitKey(1) & 0xFF


    if keypress == ord("q"):
        break

    elif keypress == ord("s"):
        text2speech = gTTS(text=res_, lang='en')
        text2speech.save('sample.mp3')

camera.release()
cv2.destroyAllWindows()
