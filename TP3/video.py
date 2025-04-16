#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print('Pass a filename as first argument')
    sys.exit(0)

cap = cv2.VideoCapture(filename)
fourcc = cv2.VideoWriter_fourcc(*'H264')

#Obtener framesize
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framesize = (width, height)
fps = cap.get(cv2.CAP_PROP_FPS)
delay =  int(1000 / fps)  # Calcular el delay en milisegundos

out = cv2.VideoWriter('output.mp4', fourcc, fps, framesize)



while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convertir escala de grises a BGR
        out.write(gray_bgr)  # Escribir el fotograma en formato BGR
        cv2.imshow('Image gray', gray)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()