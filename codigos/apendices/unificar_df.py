import pandas as pd
import os
from pathlib import Path

# Listar los CSVs en orden de creación
def listdir_date(dirpath):
    return map(lambda p: str(p).split('/')[-1], 
               sorted(Path(dirpath).iterdir(), key=os.path.getmtime))
    
if __name__ == '__main__':
    path = 'path/to/dfs'
    
    # Cargar los CSVs ordenados por fecha como una tupla(nombre, objeto)
    dfs = [(file, pd.read_csv(f'{path}/{file}')) for file in listdir_date(path)]

    for filename, df in dfs:
        # Insertar colúmna de los nombres de archivos y número de simulación
        # correspondiente a cada csv
        names = [f'{filename.split(".")[0]}/{i}.png' for i in range(len(df))]
        df['filenames'] = names
    
    # Concatenar CSVs en uno solo
    whole_dataset = pd.concat(list(map(lambda x: x[1], dfs)))

    # Guardar como un nuevo archivo
    whole_dataset.to_csv('path/to/train_dataset.csv', index=False)
