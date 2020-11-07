import cv2
import numpy as np
import os

IMG_SIZE=96

top, right, bottom, left = 200, 150, 400, 450

exit_con='**'

a=''
dir0=input('enter the directory name : ')

try:
    os.mkdir(dir0)
except:
    print('contain folder in same name')


camera = cv2.VideoCapture(0)

while(True):
    a=input('exit: ** or enter the label name : ')

    if a==exit_con:
        break

    dir1=str(dir0)+'/'+str(a)
    print(dir1)

    try:
        os.mkdir(dir1)
    except:
        print('contain folder')

    i=0
    while(True):
        (t, frame) = camera.read()

        frame = cv2.flip(frame, 1)


        roi = frame[top:bottom, right:left]


        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)


        gray = cv2.resize(gray, (IMG_SIZE,IMG_SIZE))


        cv2.imwrite("%s/%s/%d.jpg"%(dir0,a,i),gray)
        i+=1
        print(i)
        if i>500:
            break


        cv2.rectangle(frame, (150, 70), (350, 250), (0,255,0), 2)

        cv2.imshow("Video Feed 1", gray)

        cv2.imshow("Video Feed", frame)

        keypress = cv2.waitKey(1)


        if keypress == 27:
            break

camera.release()
cv2.destroyAllWindows()

    


