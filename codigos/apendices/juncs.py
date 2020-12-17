import pandas as pd
import os
from shutil import copy
from tqdm import tqdm

if __name__ == '__main__':
  df = pd.read_csv('path/to/train_dataset.csv')

  # Crear colúmnas para la carpeta y el archivo
  df['folder'] = [f.split('/')[0] for f in df['filenames'].tolist()]
  df['file'] = [f.split('/')[1].split('.')[0] for f in df['filenames'].tolist()]

  # Se elimina la colúmna que contiene ambos en uno
  df = df.drop(['filenames'], axis=1)

  # Seleccionar las filas de intersecciones
  df_junc = df.query('junction == True')

  # Extraer la lísta de índices
  idxs = df_junc.index.tolist()

  # Se buscan las intersecciones
  diffs = [0]*(len(idxs)-1)
  junc_idxs = []
  temp_idxs = []
  for i in range(len(idxs)-1):
    # Si la diferencia de índices es grande
    # son intersecciones distintas
    diff = idxs[i+1] - idxs[i]
    
    temp_idxs.append(idxs[i])
    if diff != 1 and (idxs[i+1] != 7999 and idxs[i] != 0):
        junc_idxs.append(temp_idxs)
        temp_idxs = []
      
  # Ubicación de las carpetas para cada intersección
  base_path = 'path/to/junctions'
  # Ubicación de las imágenes originales
  origin_path = 'path/to/Images'

  for i, junc_img_idxs in tqdm(enumerate(junc_idxs)):
    # Crear carpeta por intersección
    os.makedirs(f'{base_path}/{i}')
    for idx in junc_img_idxs:
      row = df.iloc[idx]
      folder, file = row['filenames'].split('/')
      
      # Copiar la imagen a su respectiva carpeta
      copy(f'{origin_path}/{folder}/rgb/{file}', f'{base_path}/{i}/{folder}_{file}')
