# Importing necessary modules
import cv2
import mediapipe as mp
import time
#great

# Defining a Hand Detector Class that can be used over and over, for detecting hands, and hand landmarks.
class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)

        self.mpDraw =  mp.solutions.drawing_utils

    def findHands(self,img,draw=True):

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for hlmks in self.results.multi_hand_landmarks:
                if draw:            
                    self.mpDraw.draw_landmarks(img,hlmks,self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self,img,handNo=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                        # print(id,lm)
                        h,w,c=img.shape
                        cx,cy=int(lm.x*w),int(lm.y*h)
                        lmList.append([id,cx,cy])
                        if draw:
                            cv2.circle(img,(cx,cy),6,(22,0,110),cv2.FILLED)

        return lmList
def main():
    cap=cv2.VideoCapture(0)
    cTime=0
    pTime=0
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if (len(lmList)!=0):
            print(lmList[2])
        cTime = time.time()
        fps= 1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,100,0),3)
        cv2.imshow("image",img)
        cv2.waitKey(1)

if __name__== "__main__":
    main()
