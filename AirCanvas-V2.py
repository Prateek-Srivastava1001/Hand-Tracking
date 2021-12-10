import cv2
import mediapipe as mp
import numpy as np
import os
import HandTrackingModule as htm

folderpath = "Headers"
mylist = os.listdir(folderpath)
print(mylist)
overlayList = []
for imPath in mylist:
    image = cv2.imread(f'{folderpath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (0, 0, 255)
brushThickness = 10
eraserThickness = 50
xp, yp = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectCon=0.5)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    #Importing image from camera
    success, img = cap.read()
    img = cv2.flip(img, 1)
    #Finding Hand Landmarks
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #Checking which fingers are up
        fingers = detector.fingersUp()
        #If Selection Mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1<124:
                #220-480->Red
                if 220<x1<470:
                    header=overlayList[0]
                    drawColor = (0, 0, 255)
                #480-760->Blue
                elif 490<x1<750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                #760-1050->Green
                elif 770<x1<1040:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                #>1050 ->Eraser
                elif 1060<x1<1270:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        #If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            if xp ==0 and yp == 0:
                xp, yp = x1, y1
            if(drawColor==(0, 0, 0)):
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 25, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    # Setting the header image
    img[0:124, 0:1280] = header
    cv2.imshow("Output", img)
    # cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # To Exit on pressing q
        break
# IMPORTANT STEPS
cap.release()
cv2.destroyAllWindows()
