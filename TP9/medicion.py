import cv2  # Librería OpenCV para procesamiento de imágenes
import numpy as np  # Librería NumPy para manejo de matrices
import sys

#lee argumento escrito para elegir la imagen
if(len(sys.argv)>1):
    image_path = sys.argv[1]
else:
    image_path = 'p1.jpg'  # Ruta por defecto de la imagen a procesar

# Mensajes de ayuda para el usuario
print("<r> Image restored.")  # Mensaje para restaurar la imagen
print("<c> Ventana calibrada para medición.")  # Mensaje para aplicar lcalibracion
print("<q> Exit.")  # Mensaje para salir del programa

# Variables globales
points_homography = []  # Lista para almacenar los puntos seleccionados para la homografía
img = None  # Imagen base
original_img = None  # Copia de la imagen base original

# Variables globales para medir distancias
measurement_points = []  # Lista para almacenar los puntos seleccionados en la ventana calibrada
scale_factor = None  # Factor de escala para convertir píxeles a metros (se calculará en la primera medición)

def order_points(pts):
    """
    Ordena los puntos en el siguiente orden: superior izquierda, superior derecha,
    inferior derecha, inferior izquierda.

    Parámetros:
    - pts: Lista de puntos seleccionados.

    Retorna:
    - Lista de puntos ordenados.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Suma y resta de coordenadas para identificar las esquinas
    s = np.sum(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # Superior izquierda
    rect[2] = pts[np.argmax(s)]  # Inferior derecha

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Superior derecha
    rect[3] = pts[np.argmax(diff)]  # Inferior izquierda

    return rect

def select_homography_points(event, x, y, flags, param):
    """
    Función de callback para seleccionar puntos con el mouse para la homografía.
    """
    global points_homography, img  # Variables globales necesarias

    if event == cv2.EVENT_LBUTTONDOWN:  # Si se presiona el botón izquierdo del mouse
        if len(points_homography) < 4:  # Solo permitimos seleccionar 4 puntos
            points_homography.append((x, y))  # Agrega el punto seleccionado a la lista
            # Dibuja un círculo en el punto seleccionado
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # Color azul
            cv2.imshow('image', img)  # Actualiza la ventana de la imagen

def compute_homography(src_points, dst_points):
    """
    Calcula la matriz de homografía entre dos conjuntos de puntos.

    Parámetros:
    - src_points: Lista de 4 puntos en la imagen base.
    - dst_points: Lista de 4 puntos correspondientes en la imagen destino.

    Retorna:
    - Matriz de homografía (3x3).
    """
    return cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))  # Calcula la homografía

def rectify_image(base_img, homography_matrix, output_size):
    """
    Aplica una homografía para rectificar una región de la imagen.

    Parámetros:
    - base_img: Imagen base.
    - homography_matrix: Matriz de homografía.
    - output_size: Tamaño de la imagen de salida (ancho, alto).

    Retorna:
    - Imagen rectificada.
    """
    return cv2.warpPerspective(base_img, homography_matrix, output_size)  # Aplica la homografía

def compute_rectified_size(points):
    """
    Calcula el tamaño de la ventana rectificada en función de los puntos seleccionados.

    Parámetros:
    - points: Lista de puntos seleccionados (ordenados).

    Retorna:
    - (ancho, alto): Dimensiones de la ventana rectificada.
    """
    width = int(np.linalg.norm(points[1] - points[0]))  # Distancia entre el punto superior izquierdo y superior derecho
    height = int(np.linalg.norm(points[2] - points[1]))  # Distancia entre el punto superior derecho y el inferior derecho
    return width, height

def measure_distance(event, x, y, flags, param):
    """
    Función de callback para medir la distancia entre dos puntos seleccionados en la ventana calibrada.
    """
    global measurement_points, rectified_img, scale_factor  # Variables globales necesarias

    if event == cv2.EVENT_LBUTTONDOWN:  # Si se presiona el botón izquierdo del mouse
        measurement_points.append((x, y))  # Agrega el punto seleccionado a la lista
        # Dibuja un círculo en el punto seleccionado
        cv2.circle(rectified_img, (x, y), 5, (0, 0, 255), -1)  # Color rojo
        cv2.imshow('calibrada', rectified_img)  # Actualiza la ventana calibrada

        if len(measurement_points) == 2:  # Si se seleccionaron dos puntos
            # Calcular la distancia en píxeles
            dist_pixels = np.linalg.norm(np.array(measurement_points[0]) - np.array(measurement_points[1]))
            
            if scale_factor is None:  # Si el factor de escala no está definido
                print(f"Distancia en píxeles: {dist_pixels:.2f}")
                while True:  # Bucle para validar la entrada del usuario
                    try:
                        real_distance = float(input("\nIngrese la distancia real en metros entre los puntos seleccionados: "))
                        if real_distance <= 0:  # Validar que la distancia sea positiva
                            raise ValueError("La distancia debe ser un número positivo.")
                        scale_factor = real_distance / dist_pixels  # Calcular el factor de escala
                        print(f"Factor de escala calculado: {scale_factor:.6f} metros/píxel")

                        # Dibujar la línea entre los puntos seleccionados
                        cv2.line(rectified_img, measurement_points[0], measurement_points[1], (255, 0, 0), 2)  # Color azul

                        # Mostrar el valor de calibración sobre la línea
                        mid_point = (
                            (measurement_points[0][0] + measurement_points[1][0]) // 2,
                            (measurement_points[0][1] + measurement_points[1][1]) // 2,
                        )
                        cv2.putText(
                            rectified_img,
                            f"cal={real_distance:.2f} m",
                            mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),  # Color rojo
                            1,
                            cv2.LINE_AA,
                        )  # Color rojo
                        cv2.imshow('calibrada', rectified_img)  # Actualiza la ventana calibrada
                        break  # Salir del bucle si la entrada es válida
                    except ValueError as e:
                        print(f"Entrada inválida: {e}. Por favor, intente nuevamente.")
            else:
                # Convertir la distancia a metros usando el factor de escala
                dist_meters = dist_pixels * scale_factor
                print(f"Distancia en píxeles: {dist_pixels:.2f}, Distancia en metros: {dist_meters:.2f}")

                # Dibujar la línea entre los puntos seleccionados
                cv2.line(rectified_img, measurement_points[0], measurement_points[1], (255, 0, 0), 2)  # Color azul

                # Mostrar el valor de la medición sobre la línea
                mid_point = (
                    (measurement_points[0][0] + measurement_points[1][0]) // 2,
                    (measurement_points[0][1] + measurement_points[1][1]) // 2,
                )
                cv2.putText(
                    rectified_img,
                    f"{dist_meters:.2f} m",
                    mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),  # Color rojo
                    1,
                    cv2.LINE_AA,
                )  # Color rojo
                cv2.imshow('calibrada', rectified_img)  # Actualiza la ventana calibrada
            
            measurement_points = []  # Reinicia la lista de puntos seleccionados

# Valores fijos para el ancho y alto en metros de la zona seleccionada
ZONE_WIDTH_METERS = 3.08 # Ancho en metros
ZONE_HEIGHT_METERS =1.23  # Alto en metros

def compute_scale_factor(rect_width, rect_height):
    """
    Calcula el factor de escala (metros por píxel) basado en las dimensiones rectificadas.

    Parámetros:
    - rect_width: Ancho de la ventana rectificada en píxeles.
    - rect_height: Alto de la ventana rectificada en píxeles.

    Retorna:
    - Factor de escala en metros por píxel.
    """
    scale_x = ZONE_WIDTH_METERS / rect_width
    scale_y = ZONE_HEIGHT_METERS / rect_height
    return (scale_x + scale_y) / 2  # Promedio de las escalas en X e Y

# Valores fijos para el ancho y alto en píxeles de la zona seleccionada
ZONE_WIDTH_PIXELS = 1000  # Ancho en píxeles (puedes ajustar según tu preferencia)
ZONE_HEIGHT_PIXELS = int(ZONE_WIDTH_PIXELS * (ZONE_HEIGHT_METERS / ZONE_WIDTH_METERS))  # Alto proporcional en píxeles

# Cargar la imagen base desde el archivo
img = cv2.imread(image_path)  # Carga la imagen desde la ruta especificada
if img is None:  # Verifica si la imagen se cargó correctamente
    raise FileNotFoundError(f"No se pudo cargar la imagen desde {image_path}")

original_img = img.copy()  # Guarda una copia de la imagen base original

# Configurar la ventana y el callback del mouse
cv2.namedWindow('image')  # Crea una ventana para mostrar la imagen
cv2.setMouseCallback('image', select_homography_points)  # Asigna la función de callback para manejar eventos del mouse

print("Seleccione 4 puntos no colineales en la imagen base con el mouse.")  # Mensaje para el usuario

# Bucle principal del programa
while True:
    cv2.imshow('image', img)  # Muestra la imagen en la ventana
    key = cv2.waitKey(1) & 0xFF  # Espera una tecla presionada

    if key == ord('c'):  # Si se presiona la tecla 'c'
        if len(points_homography) == 4:  # Verifica que se hayan seleccionado 4 puntos
            # Ordenar los puntos seleccionados
            ordered_points = order_points(np.array(points_homography))

            # Define los puntos correspondientes en la imagen destino (rectangular)
            points_dst = [(0, 0), (ZONE_WIDTH_PIXELS - 1, 0), 
                          (ZONE_WIDTH_PIXELS - 1, ZONE_HEIGHT_PIXELS - 1), (0, ZONE_HEIGHT_PIXELS - 1)]

            # Calcular la homografía
            homography_matrix = compute_homography(ordered_points, points_dst)

            # Rectificar la imagen
            rectified_img = rectify_image(original_img, homography_matrix, (ZONE_WIDTH_PIXELS, ZONE_HEIGHT_PIXELS))
            cv2.namedWindow('calibrada')  # Crea una nueva ventana para la imagen rectificada
            cv2.imshow('calibrada', rectified_img)  # Muestra la imagen rectificada
            cv2.setMouseCallback('calibrada', measure_distance)  # Asigna el callback para medir distancias

            # Calcular el factor de escala
            scale_factor = compute_scale_factor(ZONE_WIDTH_PIXELS, ZONE_HEIGHT_PIXELS)
            print(f"Factor de escala calculado automáticamente: {scale_factor:.6f} metros/píxel")
            print("\nVentana calibrada para medición. Seleccione dos puntos para medir la distancia.")
        else:
            print("Debe seleccionar exactamente 4 puntos no colineales.")  # Mensaje de error

    elif key == ord('r'):  # Si se presiona la tecla 'r'
        cv2.destroyWindow('calibrada')  # Cierra la ventana calibrada
        img = original_img.copy()  # Restaura la imagen base original
        points_homography = []  # Reinicia la lista de puntos seleccionados para homografía
        print("Selección reiniciada.")  # Mensaje para el usuario

    elif key == ord('q'):  # Si se presiona la tecla 'q'
        break  # Sale del bucle principal

cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV
