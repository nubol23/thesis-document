import os
import time
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from custom_mobilenet import CustomMobileNet

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

class DriveDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir, transform=None):
    temp = root_dir.split('/')
    self.root_dir = '/'.join(temp[:-1])
    self.transform = transform

    self.data = pd.read_csv(root_dir)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    row = self.data.iloc[idx]
    
    # Cargar imagen RGB
    folder, file = row['filenames'].split('/')
    img_rgb_path = os.path.join(
        self.root_dir, 'Images', folder, 'rgb', file)
    img_rgb = Image.open(img_rgb_path)

    if self.transform:
        img_rgb = self.transform(img_rgb)

    # leer etiquetas de la i-ésima muestra
    throttle = row['throttle']
    steering = row['steer']
    action_left = row['action_left']
    action_right = row['action_right']
    action_forward = row['action_forward']
    no_action = row['no_action']
    
    # Crear muestra
    sample = (img_rgb,
              # Tensor de parámetros de acción.
              torch.tensor([
                  float(action_left), float(action_right),
                  float(action_forward), float(no_action)
              ]),
              # Tensor de etiquetas de aceleración y giro.
              torch.tensor([
                  float(throttle), float(steering)
              ]))

    return sample


if __name__ == '__main__':
  model = CustomMobileNet(pretrained=True)

  model.cuda()

  train_dir = 'path/to/train_dataset_final.csv'
  val_dir = 'path/to/val_dataset_final.csv'

  train_loader = torch.utils.data.DataLoader(
    dataset=DriveDataset(train_dir, transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        lambda T: T[:3]
    ])),
    batch_size=64,
    shuffle=True,
    num_workers=12,
    pin_memory=True
  )

  val_loader = torch.utils.data.DataLoader(
    dataset=DriveDataset(val_dir, transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])),
    batch_size=64,
    shuffle=False,
    num_workers=12,
    pin_memory=True
  )

  criterion = nn.MSELoss().cuda()
  optimizer = torch.optim.Adam(model.parameters())

  losses = []

  # Iterar para entrenar la red
  for epoch in range(50):
    start = time.time()

    model.train()
    train_loss = 0
    # Definir barra de progreso interactiva
    train_progress = tqdm(enumerate(train_loader),
                          desc="train",
                          total=len(train_loader))
                          
    # Iterar por cada minibatch de 64 muestras
    for i, (X, actions, y) in train_progress:
      # Copiar los datos a la GPU
      X = X.cuda(non_blocking=True)
      actions = actions.cuda(non_blocking=True)
      y = y.cuda(non_blocking=True)
      y_hat = model(X, actions)
      
      # Calcular el error cuadrático medio para la aceleración
      loss1 = criterion(y_hat[:, 0], y[:, 0])
      # Calcular el error cuadrático medio para la dirección
      loss2 = criterion(torch.tanh(y_hat[:, 1]), y[:, 1])
      # Combinar ambos errores
      loss = (loss1 + loss2)/2

      # Paso de optimización
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss += float(loss.detach())
      train_progress.set_postfix(loss=(train_loss/(i+1)))

    model.eval()

    val_loss = 0
    with torch.no_grad():
      model.eval()
      val_progress = tqdm(enumerate(val_loader),
                          desc="val",
                          total=len(val_loader))
      for i, (X, actions, y) in val_progress:
        X = X.cuda(non_blocking=True)
        actions = actions.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        y_hat = model(X, actions)

        loss1 = criterion(y_hat[:, 0], y[:, 0])
        loss2 = criterion(y_hat[:, 1], y[:, 1])
        loss = (loss1 + loss2) / 2

        val_loss += float(loss)
        val_progress.set_postfix(loss=(val_loss/(i+1)))

    end = time.time()

    t_loss = train_loss / len(train_loader)
    v_loss = val_loss / len(val_loader)
    print('epoch:', epoch, 'L:', t_loss, v_loss, 'Time:', end-start)

    torch.save(
      {
          'epoch': epoch,
          'arch': 'mobilenet_custom',
          'state_dict': model.state_dict()
      },
      f'weights/mob_drive_{epoch}.pth.tar')
    losses.append([epoch, t_loss, v_loss])
    np.save('hist_drive', np.array(losses))
