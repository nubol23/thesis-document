alpha = 0.75
ema = None
threshold = 0.2

while True:
    # ... código de control y predicción en cada iteración
    
    # se obitnen las predicciones de cada red
    out = model_drive(X, action_tensor).cpu()
    depth_map = model_depth(X_D)[0, 0].cpu().numpy()
    segmentation = model_semseg(X).cpu().numpy()[0].argmax(axis=0).astype(np.uint8)

    # Redondeo de valores para reducir precisión y oscilaciones
    # t: throttle (acelerador), s: (steer) dirección
    t, s = round(float(out[0, 0]), 3), round(float(out[0, 1]), 3)
    # Limitación de la velocidad
    t = min(t, 0.5)
    
    # Si estamos fuera de una intersección
    if isinstance(ema, type(None)):  # Si ema está en t=0
        ema = s                      # inicializar
    else:
        ema = alpha*s + (1-alpha)*ema  # Dar un paso de EMA
        
    if abs(s) <= threshold:  # Si está en el rango
        s = ema              # Aplicar el valor del EMA
