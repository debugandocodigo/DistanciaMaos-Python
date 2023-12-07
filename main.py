import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

video = cv2.VideoCapture(0)  # inicia a captura de video

video.set(3,1280)  # seta a largura do video
video.set(4,720)  # seta a altura do video

detector = HandDetector(detectionCon=0.8,maxHands=1)  # inicia o detector de mãos

## Distância entre os dedos
distPixels = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
distCM = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coef = np.polyfit(distPixels,distCM,2)  # calcula os coeficientes da função de segundo grau

while True:
    check, img = video.read()  # lê o frame do video
    img = cv2.flip(img,1)  # espelha a imagem
    hands = detector.findHands(img,draw=False)  # detecta as mãos

    if hands:  # se houver mãos
        lmlist = hands[0]['lmList']  # lista com as posições dos landmarks
        x,y,w,h = hands[0]['bbox']  # bounding box da mão
        x1,y1,_ = lmlist[5]  # posição do dedo indicador
        x2,y2, _ = lmlist[17]  # posição do dedo médio
        dist = (abs(x2 - x1))  # calcula a distância entre os dedos indicador e médio
        A,B,C = coef  # coeficientes da função de segundo grau
        distCMT = (A*dist**2)+(B*dist)+C  # calcula a distância em cm

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)  # desenha o bounding box da mão
        cvzone.putTextRect(img,f'{int(distCMT)} cm',(x+5,y-10))  # escreve a distância em cm

    cv2.imshow('Imagem',img)  # mostra a imagem
    cv2.waitKey(1)  # espera 1ms