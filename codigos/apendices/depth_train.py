import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from models import CustomMobilenet

from PIL import Image, ImageOps
import os
import numpy as np
import pandas as pd
import time

from tqdm import tqdm

import matplotlib.pyplot as plt

# Clase encargada de cargar los datos
class DriveDepthDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir, transform=None, stills=True):
    temp = root_dir.split('/')
    self.root_dir = '/'.join(temp[:-1])
    self.transform = transform

    self.data = pd.read_csv(root_dir)
    # Si se desean mantener las imágenes donde el vehículo está quieto
    if stills:
        self.data = self.data['n_id'].tolist()
    else:
        self.data = self.data.query('throttle != 0.0')['n_id'].tolist()

  def __len__(self):
    return len(self.data)
  
  # Sobrecarga del operador [] para indexado
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    
    # Decidir si espejar la imagen aleatoriamente
    h_flip = np.random.random() < 0.5

    # Cargar imagen RGB
    img_rgb_path = os.path.join(self.root_dir, 'rgb', f'{self.data[idx]}.png')
    img_rgb = Image.open(img_rgb_path)
    if h_flip:
        img_rgb = ImageOps.mirror(img_rgb)
    if self.transform:
        img_rgb = self.transform(img_rgb)

    # Cargar imagen de profundidad
    img_depth_path = os.path.join(self.root_dir, 'depth', f'{self.data[idx]}.png')
    img_depth = Image.open(img_depth_path)
    if h_flip:
        img_depth = ImageOps.mirror(img_depth)
        
    # Convertir la imagen de 24 bits a un mapa de profundidades [0, 1000]
    img_depth = np.transpose(np.asarray(img_depth, dtype=np.float32), (2, 0, 1))
    target = img_depth[0, :, :] + img_depth[1, :, :] * 256 + img_depth[2, :, :] * 256 * 256
    
    # Truncar las distancias hasta 30 metros como máximo
    target = np.clip((target / (256 * 256 * 256 - 1)) * 1000, None, 30)
    target = torch.from_numpy(target).float()
    
    # Crear tupla de la muestra
    sample = (img_rgb, target.view(1, target.shape[0], target.shape[1]))

    return sample

if __name__ == '__main__':
  # Valores de normalización de la imagen
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  # Instanciar la red de profundidades
  model = CustomMobilenetDepth((180, 240), pretrained=False)
  # Enviar parámetros a la GPU
  model.cuda()

  train_dir = 'path/to/train_data.csv'
  val_dir = 'path/to/val_data.csv'
  
  # Instanciar el DataLoader para el conjunto de entrenamiento
  train_loader = torch.utils.data.DataLoader(
    dataset=DriveDepthDataset(train_dir, transforms.Compose([
        transforms.ToTensor(),
        lambda T: T[:3],
        normalize
    ])),
    batch_size=64,
    shuffle=True,
    num_workers=12,
    pin_memory=True
  )
  
  # Instanciar el DataLoader para el conjunto de validación
  val_loader = torch.utils.data.DataLoader(
    dataset=DriveDepthDataset(val_dir, transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])),
    batch_size=64,
    shuffle=True,
    num_workers=12,
    pin_memory=True
  )
  
  # Definir función de costo a optimizar
  criterion = nn.MSELoss().cuda()
  # Definir optimizador
  optimizer = torch.optim.Adam(model.parameters())
  
  # Historial de errores
  losses = []
  
  # Entrenar por un número de iteraciones o hasta detener el proceso
  for epoch in range(50):
    # Guardar tiempo de inicio de iteración
    start = time.time()
    
    # Modelo en modo entrenamiento
    model.train()
    train_loss = 0
    # Iterar por los bloques de datos de 64 en 64
    for i, (X, y) in tqdm(enumerate(train_loader)):
      X = X.cuda(non_blocking=True)
      y = y.cuda(non_blocking=True)
      # Realizar la predicción
      y_hat = model(X)

      # Calcular el error
      loss = criterion(y_hat, y)
      # Acumular el error
      train_loss += float(loss.detach())
      
      # Derivar y actualizar los parámetros
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    # Evaluar la red en el conjunto de validación cruzada
    val_loss = 0
    with torch.no_grad():
      model.eval()
      for i, (X, y) in enumerate(val_loader):
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        y_hat = model(X)

        loss = criterion(y_hat, y)
        val_loss += float(loss)
  
    # Guardar el tiempo de finalización de la iteración
    end = time.time()
    
    # Calcular el error medio de la iteración
    t_loss = train_loss / len(train_loader)
    v_loss = val_loss / len(val_loader)
    print('epoch:', epoch, 'L:', t_loss, v_loss, 'Time:', end - start)
    
    # Guardar los parámetros de la i-ésima iteración
    torch.save(
      {
        'epoch': epoch,
        'arch': 'mobilenet_depth',
        'state_dict': model.state_dict()
      },
      f'weights/c_mob_{epoch}.pth.tar')
    # Guardar los errores
    losses.append([epoch, t_loss, v_loss])
    np.save('hist', np.array(losses))
