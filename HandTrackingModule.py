import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False,maxHands = 2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon
        self.tipIds = [4, 8, 12, 16, 20]

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for Lms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, Lms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img, handNo=0, draw=False):
        self.lmList = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 15, (255, 0, 255),cv2.FILLED)

        return self.lmList
    def fingersUp(self):
        fingers = []
        #Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #Other fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers



def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[8])
        cv2.imshow("Video", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # To Exit on pressing q
            break

if __name__ == "__main__":
    main()
