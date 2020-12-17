path = '...'  # Dirección de los CSVs por simulación
# Cargar una lista de CSVs
dfs = [(file, pd.read_csv(f'{path}/{file}')) for file in listdir_date(path)]

for filename, df in dfs:
    names = [f'{filename.split(".")[0]}/{i}.png' for i in range(len(df))]
    df['filenames'] = names

whole_dataset = pd.concat(list(map(lambda x: x[1], dfs)))

whole_dataset.to_csv(f'{dest}/train_dataset.csv', index=False)
