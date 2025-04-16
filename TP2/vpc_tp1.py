#TP1 (Mascara ByN de una imagen)
#Joaquin Pappano Meinardi | Leg:86730
#Año: 2025 
#Materia: Visión por Computadora (UTN-FRC)
#----------------------------------------------
import cv2
import sys

#lee argumento escrito al llamar 
if(len(sys.argv)>1):
    png = sys.argv[1]
else:
    png = 'hoja.png'

# Definir un umbral para la máscara
if (len(sys.argv)>2):
    threshold = int(sys.argv[2])
else:
    threshold = 170



img = cv2.imread(png, 0)  # Carga la imagen en escala de grises


# Recorremos la imagen píxel a píxel
for row_idx, row in enumerate(img):
    for col_idx, col in enumerate(row):
        # Si el valor del píxel es mayor que el umbral, lo hacemos blanco (255), si no, negro (0)
        img[row_idx, col_idx] = 255 if col > threshold else 0

# Guardar la imagen resultante
cv2.imwrite('resultado.png', img)