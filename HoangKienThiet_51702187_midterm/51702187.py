import sys
import cv2
# import numpy as np
fOutput = open('output.txt', 'w')
img = cv2.imread('input.png')
#------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# -----------------
thresh = cv2.adaptiveThreshold(blur,127,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,2)
ret, markers = cv2.connectedComponents(thresh) # lable_num, lable_img
# -----------------
convertBinary = cv2.bitwise_not(thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
# -----------------
opening = cv2.morphologyEx(convertBinary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
# -----------------
revertBinary = cv2.bitwise_not(closing)
edged = cv2.Canny(revertBinary, 100, 127, 127, L2gradient=True)
#------------------
cnts, hier = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = lambda x: cv2.contourArea(x), reverse = True)
# ------------- Draw
for c in cnts:
    if cv2.contourArea(c) > 100 and cv2.contourArea(c) < 500:
        [X, Y, W, H] = cv2.boundingRect(c)
        cv2.rectangle(img, (X, Y), (X + W, Y + H), (0,0,255), 1)
        fOutput.write(str(X))
        fOutput.write(',')
        fOutput.write(str(Y))
        fOutput.write(' ')
        fOutput.write(str(W))
        fOutput.write(' ')
        fOutput.write(str(H))
        fOutput.write('\n')

fOutput.close()
cv2.imshow('Output',img)
cv2.imwrite('output.jpg',img)
cv2.waitKey(0)
