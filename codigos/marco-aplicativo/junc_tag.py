data = defaultdict(list)
for junc in juncs: # Iterando por las carpetas
    data['folder'].append(int(junc))
    # Lectura de imagenes
    files = sorted(os.listdir(f'junctions/{junc}'),
                   key=lambda x: int(x.split('.')[0].split('_')[1]))
    for file in files: # Visualización
        img = cv2.imread(f'junctions/{junc}/{file}')
        cv2.imshow('window', img)
        sleep(0.01)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    print(f'junc: {junc}')
    junc_desc = input() # Registro de etiquetas
    data['path_left'].append(1 if 'a' in junc_desc else 0)
    data['path_right'].append(1 if 'd' in junc_desc else 0)
    data['path_forward'].append(1 if 'w' in junc_desc else 0)

    junc_action = input()
    data['action_left'].append(1 if 'a' in junc_action else 0)
    data['action_right'].append(1 if 'd' in junc_action else 0)
    data['action_forward'].append(1 if 'w' in junc_action else 0)

df = pd.DataFrame.from_dict(data)
df.to_csv('juncs_final.csv', index=False)
