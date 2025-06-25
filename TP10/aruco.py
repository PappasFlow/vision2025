#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import cv2.aruco as aruco
import datetime  # Importar el módulo datetime

# Diccionario de empleados con información asociada
empleado = {
    1: {"name": "Joaquin", "status": False},
    2: {"name": "Melina", "status": False }
}

# Verificar si se pasa un argumento para usar un archivo de video
if len(sys.argv) > 1:
    filename = sys.argv[1]
    cap = cv2.VideoCapture(filename)
else:
    print('No se pasó un archivo de video, abriendo la cámara...\n')
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Forzar el uso de V4L2

if not cap.isOpened():
    print("Error al abrir el video o la cámara.")
    sys.exit()

# Instrucciones para el usuario
print("Instrucciones:")
print("Deteccion automatica entrada/salida por ArUco\n")
print("Presiona 'q' para salir.\n")


# Cargar el diccionario de marcadores ArUco
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Crear la ventana una sola vez
cv2.namedWindow('Detección de ArUco', cv2.WINDOW_NORMAL)

# Dibujar sobre ARuCo 
def draw_markers(corners, ids, frame):
    for corner, marker_id in zip(corners, ids):
        # Convertir las esquinas a enteros
        corner = corner.reshape((4, 2)).astype(int)
        # Centro del cuadrado
        center_x = int(corner[:, 0].mean())
        center_y = int(corner[:, 1].mean())

        # Verificar si el marcador está en el diccionario de empleados
        if marker_id[0] not in empleado:
            # Agregar el marcador como "Desconocido" si no está en el diccionario
            empleado[marker_id[0]] = {"name": "Desconocido", "status": False}
            print(f"Nuevo marcador detectado: ID {marker_id[0]} agregado como 'Desconocido'.")

        # Obtener el nombre del empleado
        name = empleado[marker_id[0]]["name"]

        # Mostrar el nombre del empleado o el ID
        if name != "Desconocido":
            cv2.putText(frame, f"{name}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Dibujar el cuadrado verde alrededor del marcador
            cv2.polylines(frame, [corner], isClosed=True, color=(0, 255, 0), thickness=2)  # Color y grosor
        else:
            cv2.putText(frame, f"ID: {marker_id[0]}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Dibujar el cuadrado rojo alrededor del marcador
            cv2.polylines(frame, [corner], isClosed=True, color=(0, 0, 255), thickness=2)  # Color y grosor

def check_position(corners, ids, frame, center_x):
    for corner, marker_id in zip(corners, ids):
        # Convertir las esquinas a enteros
        corner = corner.reshape((4, 2)).astype(int)
        # Calcular el centro del marcador
        marker_center_x = int(corner[:, 0].mean())
        
        # Actualizar el estado del marcador si cruza la línea central
        current_status = empleado[marker_id[0]]["status"]
        current_time = datetime.datetime.now().strftime("[%H:%M:%S del %d/%m/%Y]")  # Obtener fecha y hora actual
        if marker_center_x < center_x and current_status != False:
            empleado[marker_id[0]]["status"] = False
            print(f"{empleado[marker_id[0]]['name']} (id:{marker_id[0]}) salió {current_time}.")
        elif marker_center_x > center_x and current_status != True:
            empleado[marker_id[0]]["status"] = True
            print(f"{empleado[marker_id[0]]['name']} (id:{marker_id[0]}) entró {current_time}.")

# Variable global para contar empleados dentro
active_count = 0
def count_active_employees(frame):
    global active_count  # Declarar active_count como global
    active_count = sum(1 for emp in empleado.values() if emp["status"])
    cv2.putText(frame, f"Empleados dentro: {active_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Mostrar el video en tiempo real con detección de ArUco
while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame. Finalizando...")
        break

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar marcadores ArUco
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Obtener dimensiones del frame y calcular el centro en el eje X
    height, width = frame.shape[:2]
    center_x = width // 2

    # Dibujar una línea vertical en el centro del frame
    cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 4)

    # Si se detectan marcadores, dibujar los bordes y verificar posición
    if ids is not None:
        draw_markers(corners, ids, frame)
        check_position(corners, ids, frame, center_x)

    # Contar empleados activos y mostrar el número en la pantalla
    count_active_employees(frame)

    # Mostrar ventana
    cv2.imshow('Detección de ArUco', frame)

    # Salir al presionar la tecla 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(f"\nEmpleados dentro antes de cerrar: {active_count}")
        break

# Liberar recursos y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()