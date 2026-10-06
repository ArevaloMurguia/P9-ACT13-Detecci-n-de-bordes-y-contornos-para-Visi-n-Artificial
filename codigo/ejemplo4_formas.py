print("=== Arevalo Rafael NC:0031 ===")
import cv2

# Cargar imagen
imagen = cv2.imread("imagenes/Elefante.jpg")

# Comprobar imagen
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Umbralización
_, binaria = cv2.threshold(
    gris,
    127,
    255,
    cv2.THRESH_BINARY
)

# Buscar contornos
contornos, _ = cv2.findContours(
    binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Crear copia
resultado = imagen.copy()

for contorno in contornos:

    # Eliminar objetos pequeños
    area = cv2.contourArea(contorno)

    if area < 500:
        continue

    # Perímetro
    perimetro = cv2.arcLength(
        contorno,
        True
    )

    # Aproximar contorno
    aproximacion = cv2.approxPolyDP(
        contorno,
        0.04 * perimetro,
        True
    )

    # Número de vértices
    vertices = len(aproximacion)

    # Identificar forma
    if vertices == 3:
        forma = "Triangulo"

    elif vertices == 4:

        x, y, ancho, alto = cv2.boundingRect(
            aproximacion
        )

        relacion = ancho / float(alto)

        if 0.90 <= relacion <= 1.10:
            forma = "Cuadrado"
        else:
            forma = "Rectangulo"

    elif vertices == 5:
        forma = "Pentagono"

    elif vertices > 5:
        forma = "Circulo"

    else:
        forma = "Desconocida"

    # Dibujar contorno
    cv2.drawContours(
        resultado,
        [aproximacion],
        -1,
        (0, 255, 0),
        2
    )

    # Obtener posición
    x, y, ancho, alto = cv2.boundingRect(
        aproximacion
    )

    # Escribir nombre de la forma
    cv2.putText(
        resultado,
        forma,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

# Mostrar resultado
cv2.imshow(
    "Formas identificadas-0031",
    resultado
)

# Guardar resultado
cv2.imwrite(
    "resultados/ejemplo4_formas.jpg",
    resultado
)

print("Identificación de formas terminada.")
print("Resultado guardado en resultados/ejemplo4_formas-0031.jpg")

# Esperar
cv2.waitKey(0)

# Cerrar
cv2.destroyAllWindows()

print("=== Arevalo Rafael NC:0031 ===")