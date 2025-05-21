import cv2  # Librería OpenCV para procesamiento de imágenes
import numpy as np  # Librería NumPy para manejo de matrices
import sys

#lee argumento escrito para elegir la imagen
if(len(sys.argv)>1):
    image_path = sys.argv[1]
else:
    image_path = '/home/ubuntu/Desktop/vision2025/TP2/hoja.png'

# Mensajes de ayuda para el usuario
print("<r> Image restored.")  # Mensaje para restaurar la imagen
print("<h> Homography rectification.")  # Mensaje para aplicar la homografía
print("<q> Exit.")  # Mensaje para salir del programa

# Variables globales
points_homography = []  # Lista para almacenar los puntos seleccionados para la homografía
img = None  # Imagen base
original_img = None  # Copia de la imagen base original

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

    if key == ord('h'):  # Si se presiona la tecla 'h'
        if len(points_homography) == 4:  # Verifica que se hayan seleccionado 4 puntos
            # Ordenar los puntos seleccionados
            ordered_points = order_points(np.array(points_homography))

            # Define los puntos correspondientes en la imagen destino (rectangular)
            rect_width, rect_height = 400, 300  # Dimensiones de la imagen rectificada
            points_dst = [(0, 0), (rect_width - 1, 0), (rect_width - 1, rect_height - 1), (0, rect_height - 1)]

            # Calcular la homografía
            homography_matrix = compute_homography(ordered_points, points_dst)

            # Rectificar la imagen
            rectified_img = rectify_image(original_img, homography_matrix, (rect_width, rect_height))
            cv2.imshow('rectified', rectified_img)  # Muestra la imagen rectificada
        else:
            print("Debe seleccionar exactamente 4 puntos no colineales.")  # Mensaje de error

    elif key == ord('r'):  # Si se presiona la tecla 'r'
        img = original_img.copy()  # Restaura la imagen base original
        points_homography = []  # Reinicia la lista de puntos seleccionados para homografía
        print("Selección reiniciada.")  # Mensaje para el usuario

    elif key == ord('q'):  # Si se presiona la tecla 'q'
        break  # Sale del bucle principal

cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV