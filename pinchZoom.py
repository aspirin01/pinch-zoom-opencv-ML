import cv2
import mediapipe as mp
import time
import numpy as np
import HandDetectorModule as hdm
import math
import pyautogui 

pyautogui.FAILSAFE=False
pTime=0
wCam,hCam=640,480
time.sleep(3)

cap =cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
detector=hdm.HandDetector(detectionCon=0.7)
arr=[0,0]
i=0
#great work
while True:
    pyautogui.keyDown('ctrl') 

    success,img=cap.read()
    img=detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if(len(lmList)!=0):
        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        length=math.hypot(x2-x1,y2-y1)
        arr.append(length)
        diff=(arr[i-1]-arr[i-2])

        pyautogui.scroll(diff/7)
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)

        i+=1
    cTime = time.time()
    fps= 1/(cTime-pTime)
    pTime=cTime
    pyautogui.keyUp('ctrl')  

    cv2.putText(img, f'FPS: {int(fps)}',(20,50),cv2.FONT_HERSHEY_COMPLEX,1,(200,0,0),3)
    cv2.imshow("img",img)
    cv2.waitKey(1)


