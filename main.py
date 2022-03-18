import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

CAP_WIDTH = 640
CAP_HEIGHT = 480

cap = cv2.VideoCapture(0)
cap.set(3, CAP_WIDTH)#(3, 1280) # Wight
cap.set(4, CAP_HEIGHT)#(4, 720) # Height

detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = [] # all point of the snake
        self.lengths = [] # distances between each point
        self.currentLength = 0 # total length of the snake
        self.allowedLength = 150 # total allowed length
        self.previousHead = 0,0 # previous head point

        # Remove border pixels from png imgs e resizing img:
        self.imgFood = cv2.resize(cv2.imread(pathFood, cv2.IMREAD_UNCHANGED),
                        (40,40), interpolation=cv2.INTER_AREA)

        self.hFood, self.wFood, _ = self.imgFood.shape # height, width, channels
        self.foodPoint = 0, 0
        self.randomFoodLocation()


    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, CAP_WIDTH), random.randint(40, CAP_HEIGHT - 80)
        


    def update(self, imgMain, currentHead):
        px, py = self.previousHead
        cx, cy = currentHead

        self.points.append([cx, cy])
        distace = math.hypot(cx - px, cy - py)
        self.lengths.append(distace)
        self.currentLength += distace
        self.previousHead = cx, cy


        # Length reduction
        while self.currentLength > self.allowedLength:
            self.currentLength -= self.lengths.pop(0)
            self.points.pop(0)


        # Check if snake ate the food
        rx, ry = self.foodPoint
        if rx - self.wFood//2 < cx < rx + self.wFood//2 and \
            ry - self.hFood//2 < cy < ry + self.hFood//2:
            print("Ate")


        # Draw snake
        if self.points:
            for i, point in enumerate(self.points):
                if i != 0:
                    cv2.line(imgMain, self.points[i-1],point, (0,0,255), 20)
            cv2.circle(imgMain, self.points[-1], 12, (200, 0, 200), cv2.FILLED)

        # Draw food
        rx, ry = self.foodPoint
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx-self.wFood//2, ry-self.hFood//2))

        return imgMain
                



game = SnakeGameClass("apple.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) # Inverter a imagem horizontalmente
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList'] # Lista dos pontos detectados da mão
        pointIndex = lmList[8][0:2] # Pontos [x, y] do dedo indicador, sem incluir z. [x, y, z]
        img = game.update(img, pointIndex)

    cv2.imshow("Image",img)
    cv2.waitKey(1)