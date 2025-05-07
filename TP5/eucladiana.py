# Importación de módulos necesarios
import cv2  # OpenCV para procesamiento de imágenes
import numpy as np  # Numpy para manejo de matrices
import sys  # Para manejar argumentos de línea de comandos
import os  # Para manejar archivos y directorios

# Mensajes de ayuda para el usuario
print("<r> Image restored.")  # Mensaje para restaurar la imagen
print("<e> Euclidean transformation.")  # Mensaje apply_euclidean_transformation
print("<q> Exit.")  # Mensaje para salir del programa

# Verifica si se pasó un argumento al ejecutar el script
if len(sys.argv) > 1:
    png = sys.argv[1]
else:
    png = '/home/ubuntu/Desktop/vision2025/TP2/hoja.png'

# Variables globales para manejar el dibujo
drawing = False  # Indica si el mouse está presionado
ix, iy = -1, -1  # Coordenadas iniciales del dibujo
fx, fy = -1, -1  # Coordenadas finales del dibujo


# Carga la imagen desde la ruta especificada
img = cv2.imread(png)  # Lee la imagen
if img is None:
    raise FileNotFoundError("Image not found. Please check the path.")
original_img = img.copy()  # Crea una copia de la imagen original

#------------------------------------------------------------------------
def apply_euclidean_transformation(image, angle, tx, ty):
    """
    Aplica una transformación euclidiana (rotación + traslación) a una imagen.

    Parámetros:
    - image: Imagen a transformar (matriz de píxeles).
    - angle: Ángulo en radianes.
    - tx: Traslación en x.
    - ty: Traslación en y.

    Retorna:
    - Imagen transformada.
    """
    # Obtén las dimensiones de la imagen
    rows, cols = image.shape[:2]

    # Calcula el tamaño del nuevo lienzo
    cos_angle = abs(np.cos(angle))
    sin_angle = abs(np.sin(angle))
    new_width = int(cols * cos_angle + rows * sin_angle)
    new_height = int(cols * sin_angle + rows * cos_angle)

    # Calcula el centro de la imagen original
    original_center = (cols / 2, rows / 2)

    # Calcula la matriz de rotación alrededor del centro original
    rotation_matrix = cv2.getRotationMatrix2D(original_center, np.degrees(angle), 1)

    # Ajusta la traslación para centrar la imagen en el nuevo lienzo
    rotation_matrix[0, 2] += (new_width / 2) - original_center[0] + tx
    rotation_matrix[1, 2] += (new_height / 2) - original_center[1] + ty

    # Aplica la transformación a la imagen con el nuevo tamaño
    transformed_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    return transformed_image


def get_transformation_parameters():
    """
    Solicita al usuario los valores de ángulo, traslación en x y traslación en y.

    Retorna:
    - angle: Ángulo en radianes.
    - tx: Traslación en x.
    - ty: Traslación en y.
    """
    try:
        # Solicita el ángulo al usuario (en grados) y lo convierte a radianes
        angle = float(input("Ingrese el ángulo de rotación (en grados): "))
        angle = np.radians(angle)  # Convierte a radianes

        # Solicita la traslación en x e y
        tx = float(input("Ingrese la traslación en x: "))
        ty = float(input("Ingrese la traslación en y: "))

        return angle, tx, ty
    except ValueError:
        print("Entrada inválida. Por favor, ingrese valores numéricos.")
        return get_transformation_parameters()  # Reintenta en caso de error

# Función para manejar eventos del mouse
def draw(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, mode, img  # Variables globales necesarias

    if event == cv2.EVENT_LBUTTONDOWN:  # Evento: botón izquierdo presionado
        drawing = True  # Inicia el dibujo
        ix, iy = x, y  # Guarda las coordenadas iniciales

    elif event == cv2.EVENT_MOUSEMOVE:  # Evento: movimiento del mouse
        if drawing is True:  # Si se está dibujando
            img = original_img.copy()  # Restaura la imagen original para evitar superposiciones
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)  # Dibuja un rectángulo


    elif event == cv2.EVENT_LBUTTONUP:  # Evento: botón izquierdo soltado
        drawing = False  # Finaliza el dibujo
        fx, fy = x, y  # Guarda las coordenadas finales
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 0), 2)  # Dibuja el rectángulo final


# Configuración de la ventana de OpenCV
cv2.namedWindow('image')  # Crea una ventana llamada 'image'
cv2.setMouseCallback('image', draw)  # Asocia la función de eventos del mouse

# Bucle principal del programa
while True:
    cv2.imshow('image', img)  # Muestra la imagen en la ventana
    k = cv2.waitKey(1) & 0xFF  # Espera por una tecla presionada

    if k == ord('e'):  # Si se presiona 'g', guarda la selección
        if ix != -1 and iy != -1 and fx != -1 and fy != -1:  # Verifica que las coordenadas sean válidas
            x1, y1 = min(ix, fx), min(iy, fy)  # Calcula las coordenadas superiores izquierdas
            x2, y2 = max(ix, fx), max(iy, fy)  # Calcula las coordenadas inferiores derechas
            cropped_img = original_img[y1:y2, x1:x2]  # Recorta la imagen seleccionada
            if cropped_img.size > 0:  # Verifica que la selección no esté vacía
                # Solicita los parámetros de transformación al usuario
                angle, tx, ty = get_transformation_parameters()
                cropped_img = apply_euclidean_transformation(cropped_img, angle, tx, ty)
                # Guarda la imagen recortada
                i = 1
                while os.path.exists(f'copia{i}.png'):  # Verifica si el archivo ya existe
                    i += 1  # Incrementa el número
                filename = f'copia{i}.png'
                cv2.imwrite(filename, cropped_img)  # Guarda la imagen recortada
                print(f"Cropped image saved as '{filename}'")
            else:
                print("Invalid selection. No image saved.")  # Mensaje de error si la selección es inválida
        img = original_img.copy() 
    elif k == ord('r'):  # Si se presiona 'r', restaura la imagen original
        img = original_img.copy() 
        print("Image restored. You can make a new selection.")  

    elif k == ord('q'):  # Si se presiona 'q', sale del programa
        break  # Rompe el bucle principal

# Cierra todas las ventanas de OpenCV
cv2.destroyAllWindows()




