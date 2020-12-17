cod = ['r', 'g', 'b']
cod_val = 'na'

# En caso de que se detecte un semáforo y se recorte en la variable crop
inp = np.float32(crop.reshape((-1,3)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
# Label: id del cluster al que pertenece cada píxel de la imagen
# Center: valores de los píxeles de los centros en cada canal
_, label, center = cv2.kmeans(inp, K, None, criteria,10, cv2.KMEANS_RANDOM_CENTERS)
# Toma el id del cluster al que pertenece el píxel y le asigna el valor del centro
res = np.uint8(center[label.flatten()]).reshape((crop.shape)) # Reconstruye la imagen
# Selecciona los valores de píxeles mayor a 200
res = res * (res > 200)
# Suma los pixeles mayores a 200
rgb_sig = [res[:, :, 0].sum(), res[:, :, 1].sum(), res[:, :, 2].sum()]
rc, gc, bc = rgb_sig
# si los canales rojo y verde son mayores a cero
if rc > 0 and gc > 0:
    if bc > 0: # si lo es también azúl
        cod_val = 'g' # es verde
    else:
        cod_val = 'r' # sino es rojo
elif rc > 0 or gc > 0 or bc > 0: # si alguno de los 3 es mayor a cero
    cod_val = cod[np.argmax(rgb_sig)] # se toma el color con más ocurrencias
