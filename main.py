import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import os

cap = cv2.VideoCapture(0) # Change here according to your webcam id (0, 1, 2...)

CAP_WIDTH = int(cap.get(3))
CAP_HEIGHT = int(cap.get(4))

detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self):
        self.points = [] # all point of the snake
        self.lengths = [] # distances between each point
        self.currentLength = 0 # total length of the snake
        self.allowedLength = 150 # total allowed length
        self.previousHead = 0,0 # previous head point

        # Remove border pixels from png imgs e resizing img:
        self.imgFood = self.updateFood()

        self.hFood, self.wFood, _ = self.imgFood.shape # height, width, channels
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False


    def updateFood(self):
        imgsPaths = ["apple", "coffee", "donut", "iceCream", "pizza"]
        curr = imgsPaths[random.randint(0, len(imgsPaths)-1)]
        return cv2.resize(cv2.imread(os.path.join("images", f'{curr}.png'), cv2.IMREAD_UNCHANGED),
                        (50,50), interpolation=cv2.INTER_AREA)

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, CAP_WIDTH - 40), random.randint(40, CAP_HEIGHT - 80)
        


    def update(self, imgMain, currentHead):

        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", [50, 200], scale= 5, 
                    thickness = 5, offset = 20)
            cvzone.putTextRect(imgMain, f'Your score: {self.score}', [50, 300], scale= 3, 
                    thickness = 5, offset = 20)

            cvzone.putTextRect(imgMain, "Press R to restart", [50, 400], scale= 2, 
                    thickness = 2, offset = 10)

            return imgMain

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
            
            self.randomFoodLocation()
            self.imgFood = self.updateFood()
            self.allowedLength += 50
            self.score += 1



        # Draw score
        cvzone.putTextRect(imgMain, f'Score: {self.score}', [50, 80], scale= 3, 
                    thickness = 3, offset = 10)


        # Draw food
        rx, ry = self.foodPoint
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx-self.wFood//2, ry-self.hFood//2))


        # Draw snake
        if self.points:
            for i, point in enumerate(self.points):
                if i != 0:
                    cv2.line(imgMain, self.points[i-1],point, (0,0,255), 20)
            cv2.circle(imgMain, self.points[-1], 12, (0, 180, 40), cv2.FILLED)

        
        # Check for collision
        pts = np.array(self.points[:-2], np.int32) # converting into a numpy array of integers
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
            
            # Check if head [cx, cy] is hitting any of the other points:
        minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

        if -2 <= minDist <= 2:
            self.gameOver = True
            self.points = []
            self.lengths = []
            self.currentLength = 0
            self.allowedLength = 150 
            self.previousHead = 0,0
            self.randomFoodLocation()



        return imgMain
                



game = SnakeGameClass()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) # Flip image horizontally
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList'] # List of hand detected points (landmarkers)
        pointIndex = lmList[8][0:2] # Index finger points [x, y], without z. [x, y, z]
        img = game.update(img, pointIndex)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)

    if key == ord('r'):
        game.gameOver = False
        game.score = 0