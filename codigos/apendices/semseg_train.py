import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from semseg_model import CustomMobilenetSemseg

from PIL import Image, ImageOps
import os
import numpy as np
import pandas as pd
import time

from tqdm import tqdm


class DriveSemsegDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir, transform=None, stills=True, train=True):
    temp = root_dir.split('/')
    self.root_dir = '/'.join(temp[:-1])
    self.transform = transform

    self.data = pd.read_csv(root_dir)
    if stills:
        self.data = self.data['filenames'].tolist()
    else:
        self.data = self.data.query('throttle != 0.0')
        self.data = self.data['filenames'].tolist()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    h_flip = np.random.random() < 0.5

    # Imagen RGB
    folder, file = self.data[idx].split('/')
    img_rgb_path = os.path.join(self.root_dir, 'Images', folder, 'rgb', file)
    img_rgb = Image.open(img_rgb_path)
    if h_flip:
        img_rgb = ImageOps.mirror(img_rgb)
    if self.transform:
        img_rgb = self.transform(img_rgb)

    # Máscara de segmentación
    img_semseg_path = os.path.join(self.root_dir, 'Images', folder, 'mask', file)
    img_semseg = Image.open(img_semseg_path)
    if h_flip:
        img_semseg = ImageOps.mirror(img_semseg)
    # Se extrae el canal R de la imagen en el que está codificada la máscara
    img_semseg = np.asarray(img_semseg)[:, :, 0].copy()
    sample = (img_rgb, torch.from_numpy(img_semseg))

    return sample


if __name__ == '__main__':
  # Se fija una semilla para hacer el experimento reproducible
  np.random.seed(42)
  
  model = CustomMobilenetSemseg((180, 240), pretrained=False)
  model = model.cuda()

  train_dir = 'path/to/train_dataset.csv'
  val_dir = 'path/to/val_dataset.csv'

  train_loader = torch.utils.data.DataLoader(
    dataset=DriveSemsegDataset(train_dir, transforms.Compose([
        transforms.ToTensor(),
        lambda T: T[:3],
    ]), stills=False),
    batch_size=64,
    shuffle=True,
    num_workers=12,
    pin_memory=True
  )

  val_loader = torch.utils.data.DataLoader(
    dataset=DriveSemsegDataset(val_dir, transforms.Compose([
        transforms.ToTensor(),
    ]), stills=False, train=False),
    batch_size=64,
    shuffle=True,
    num_workers=12,
    pin_memory=True
  )
  
  # Entropía cruzada como criterio de erro (Log loss)
  criterion = nn.CrossEntropyLoss().cuda()
  # Optimizador Adam con un learning rate de 0.01
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, amsgrad=True)

  losses = []
  
  train_len = len(train_loader)
  val_len = len(val_loader)
  for epoch in range(50):
    start = time.time()

    model.train()
    train_loss = 0
    
    train_progress = tqdm(enumerate(train_loader), desc="train", total=train_len)
    for i, (X, y) in train_progress:
      X = X.cuda(non_blocking=True)
      y = y.cuda(non_blocking=True)
      y_hat = model(X)
      
      # Se calcula el error softmax 2D
      loss = criterion(y_hat, y.long())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      loss_val = loss.detach()
      train_loss += float(loss_val)
      
      train_progress.set_postfix(loss=(train_loss/(i+1)))
    
    torch.save({
      'epoch': epoch,
      'arch': 'mobilenet_depth',
      'state_dict': model.state_dict()
    },
    f'weights/s_mob_{epoch}.pth.tar')
    
    val_loss = 0
    with torch.no_grad():
      model.eval()
      val_progress = tqdm(enumerate(val_loader), desc="val", total=val_len)
      for i, (X, y) in val_progress:
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        y_hat = model(X)

        loss = criterion(y_hat, y.long())

        val_loss += float(loss)
        
        val_progress.set_postfix(loss=(val_loss/(i+1))) # Loss info

    end = time.time()

    t_loss = train_loss / len(train_loader)
    v_loss = val_loss / len(val_loader)
    print('epoch:', epoch, 'L:', t_loss, v_loss, 'Time:', end - start)

    losses.append([epoch, t_loss, v_loss])
    np.save('hist', np.array(losses))
