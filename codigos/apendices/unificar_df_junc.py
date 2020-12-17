import pandas as pd
import os

if __name__ == '__main__':
  # CSV del conjunto de datos
  dataset = pd.read_csv('path/to/train_dataset.csv')
  # CSV de intersecciones etiquetadas
  junctions = pd.read_csv('path/to/juncs.csv')

  # Separar la colúmna filenames en una para carpetas y otra para nombre de archivos
  dataset['folder'] = [f.split('/')[0] for f in dataset['filenames'].tolist()]
  dataset['file'] = [f.split('/')[1].split('.')[0] for f in dataset['filenames'].tolist()]
  dataset = dataset.drop(['filenames'], axis=1)

  # Crear una lista vacía para cada etiqueta llena de ceros
  # (valores negativos en clasificación binaria)
  path_left = [0]*len(dataset)
  path_right = [0]*len(dataset)
  path_forward = [0]*len(dataset)

  action_left = [0]*len(dataset)
  action_right = [0]*len(dataset)
  action_forward = [0]*len(dataset)
  # Por defecto la "No acción" está marcada.
  no_action = [1]*len(dataset)

  # Ubicación de las imágenes de intersecciones
  junc_path = 'path/to/junctions'

  # Ordenar carpetas de intersección por el número de menor a mayor
  juncs = sorted(os.listdir(junc_path), key=lambda x: int(x))

  for junc in juncs:
    # Ordenar los archivos numéricamente
    files = sorted(os.listdir(f'{junc_path}/{junc}'),
                   key=lambda x: int(x.split('.')[0].split('_')[1]))
                   
    # Indexar la fila correspondiente a la intersección
    row = junctions.iloc[int(junc)]
    for file in files:
      folder, img = list(map(int, file.split('.')[0].split('_')))
      idx = 8000*folder + img
      
      # Asignar los valores etiquetados
      path_left[idx] = row['path_left']
      path_right[idx] = row['path_right']
      path_forward[idx] = row['path_forward']

      action_left[idx] = row['action_left']
      action_right[idx] = row['action_right']
      action_forward[idx] = row['action_forward']
      # Como es una intersección se desmarca la "No acción"
      no_action[idx] = 0
