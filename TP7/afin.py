import cv2  # Librería OpenCV para procesamiento de imágenes
import numpy as np  # Librería NumPy para manejo de matrices
import sys  # Librería para manejar argumentos de línea de comandos

# Mensajes de ayuda para el usuario
print("<r> Image restored.")  # Mensaje para restaurar la imagen
print("<a> afin incrustion.")  # Mensaje para aplicar la transformación afín
print("<q> Exit.")  # Mensaje para salir del programa

# Variables globales
points_src = []  # Lista para almacenar los puntos seleccionados en la imagen base
points_dst = []  # Lista para almacenar los puntos correspondientes en la imagen destino
drawing = False  # Variable para indicar si se está dibujando
img = None  # Imagen base
original_img = None  # Copia de la imagen base original
second_img = None  # Imagen que será incrustada

def select_points(event, x, y, flags, param):
    """
    Función de callback para seleccionar puntos con el mouse.
    Se activa al hacer clic en la ventana de la imagen.

    Parámetros:
    - event: Tipo de evento del mouse (por ejemplo, clic izquierdo).
    - x, y: Coordenadas del punto donde ocurrió el evento.
    - flags: Indicadores adicionales del evento.
    - param: Parámetros adicionales (no se usa aquí).
    """
    global points_src, drawing, img  # Variables globales necesarias

    if event == cv2.EVENT_LBUTTONDOWN:  # Si se presiona el botón izquierdo del mouse
        if len(points_src) < 3:  # Solo permitimos seleccionar 3 puntos
            points_src.append((x, y))  # Agrega el punto seleccionado a la lista
            # Dibuja un círculo en el punto seleccionado
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Color verde
            cv2.imshow('image', img)  # Actualiza la ventana de la imagen

def compute_affine_transformation(src_points, dst_points):
    """
    Calcula la matriz de transformación afín entre dos conjuntos de puntos.

    Parámetros:
    - src_points: Lista de 3 puntos en la imagen base.
    - dst_points: Lista de 3 puntos correspondientes en la imagen destino.

    Retorna:
    - Matriz de transformación afín (2x3).
    """
    return cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))  # Calcula la matriz afín

def embed_image(base_img, overlay_img, affine_matrix, mask):
    """
    Incrusta una imagen en otra usando una transformación afín.

    Parámetros:
    - base_img: Imagen base donde se incrustará la otra imagen.
    - overlay_img: Imagen a incrustar.
    - affine_matrix: Matriz de transformación afín.
    - mask: Máscara para definir las áreas de incrustación.

    Retorna:
    - Imagen combinada.
    """
    rows, cols, _ = base_img.shape  # Obtiene las dimensiones de la imagen base
    # Aplica la transformación afín a la imagen a incrustar
    transformed_overlay = cv2.warpAffine(overlay_img, affine_matrix, (cols, rows))
    # Aplica la transformación afín a la máscara
    transformed_mask = cv2.warpAffine(mask, affine_matrix, (cols, rows))

    # Combina las imágenes usando la máscara
    combined_img = base_img.copy()  # Crea una copia de la imagen base
    # Reemplaza los píxeles en las áreas definidas por la máscara
    combined_img[transformed_mask > 0] = transformed_overlay[transformed_mask > 0]

    return combined_img  # Retorna la imagen combinada

# Crear un fondo blanco como imagen base
height, width = 600, 800  # Dimensiones del fondo blanco
img = np.ones((height, width, 3), dtype=np.uint8) * 255  # Crea una imagen blanca
original_img = img.copy()  # Guarda una copia de la imagen base original

# Cargar la imagen a incrustar
# Verifica si se pasó un argumento al ejecutar el script
if len(sys.argv) > 1:
    overlay_img_path = sys.argv[1]  # Ruta de la imagen pasada como argumento
else:
    overlay_img_path = '/home/ubuntu/Desktop/vision2025/TP2/hoja.png'  # Ruta predeterminada
second_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)  # Carga la imagen con todos sus canales
if second_img is None:  # Verifica si la imagen se cargó correctamente
    raise FileNotFoundError("No se pudo cargar la imagen a incrustar.")  # Lanza un error si no se encuentra

# Si la imagen tiene 4 canales (BGRA), conviértela a 3 canales (BGR)
if second_img.shape[2] == 4:  # Verifica si la imagen tiene un canal alfa
    second_img = cv2.cvtColor(second_img, cv2.COLOR_BGRA2BGR)  # Convierte a formato BGR

# Crear una máscara para la imagen a incrustar
overlay_gray = cv2.cvtColor(second_img, cv2.COLOR_BGR2GRAY)  # Convierte la imagen a escala de grises
_, mask = cv2.threshold(overlay_gray, 1, 255, cv2.THRESH_BINARY)  # Crea una máscara binaria

# Configurar la ventana y el callback del mouse
cv2.namedWindow('image')  # Crea una ventana para mostrar la imagen
cv2.setMouseCallback('image', select_points)  # Asigna la función de callback para manejar eventos del mouse

print("Seleccione 3 puntos no colineales en la imagen base con el mouse.")  # Mensaje para el usuario

# Bucle principal del programa
while True:
    cv2.imshow('image', img)  # Muestra la imagen en la ventana
    key = cv2.waitKey(1) & 0xFF  # Espera una tecla presionada

    if key == ord('a'):  # Si se presiona la tecla 'a'
        if len(points_src) == 3:  # Verifica que se hayan seleccionado 3 puntos
            # Define los puntos correspondientes en la imagen destino
            points_dst = [(0, 0), (second_img.shape[1] - 1, 0), (0, second_img.shape[0] - 1)]

            # Calcular la transformación afín
            affine_matrix = compute_affine_transformation(points_dst, points_src)

            # Incrustar la imagen
            img = embed_image(original_img, second_img, affine_matrix, mask)
            cv2.imshow('image', img)  # Muestra la imagen combinada
        else:
            print("Debe seleccionar exactamente 3 puntos no colineales.")  # Mensaje de error

    elif key == ord('r'):  # Si se presiona la tecla 'r'
        img = original_img.copy()  # Restaura la imagen base original
        points_src = []  # Reinicia la lista de puntos seleccionados
        print("Selección reiniciada.")  # Mensaje para el usuario

    elif key == ord('q'):  # Si se presiona la tecla 'q'
        break  # Sale del bucle principal

cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV