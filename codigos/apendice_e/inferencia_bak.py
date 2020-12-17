import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torch.nn as nn
import glob
import os
import sys

from scipy import stats

sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.version_info.major,
    sys.version_info.minor,
    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
sys.path.append('../carla')
sys.path.append('path/to/DriveNet')
sys.path.append('path/to/DepthNet')
sys.path.append('path/to/SemsegNet')

import carla
from custom_mobilenet import CustomMobileNet, CustomMobileNetExt
from models import CustomMobilenetDepth
from semseg_model import CustomMobilenetSemseg

from carla.libcarla import Transform

from collections import Counter
import pygame
import numpy as np
import queue
import re
import weakref
import collections
import cv2
from PIL import Image
from numba import njit
from priority_queue import PriorityQueue
import imutils


def should_quit():
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      return True
    elif event.type == pygame.KEYUP:
      if event.key == pygame.K_ESCAPE:
        return True
  return False


def find_weather_presets():
  # Método disponible en el repositorio oficial
  """Method to find weather presets"""
  rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

  def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

  presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
  return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def img_to_array(image):
  array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
  array = np.reshape(array, (image.height, image.width, 4))
  array = array[:, :, :3]
  return array


def show_window(surface, array):
  if len(array.shape) > 2:
    array = array[:, :, ::-1]
  image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
  surface.blit(image_surface, (0, 0))


def create_camera(cam_type, vehicle, pos, h, w, lib, world):
  cam = lib.find(f'sensor.camera.{cam_type}')
  cam.set_attribute('image_size_x', str(w))
  cam.set_attribute('image_size_y', str(h))
  camera = world.spawn_actor(
    cam,
    pos,
    attach_to=vehicle,
    attachment_type=carla.AttachmentType.Rigid)
  return camera

def set_random_weather(world):
  weather_presets = find_weather_presets()
  weather_index = np.random.choice([0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14])
  preset = weather_presets[weather_index]
  world.set_weather(preset[0])
    
def spawn_player(world):
  world_map = world.get_map()
  player = None
  while player is None:
    spawn_points = world_map.get_spawn_points()
    spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
        
    player = world.try_spawn_actor(world.get_blueprint_library().filter('model3')[0], 
                                   spawn_point)
        
  return player
      
def load_network(model_class, path, shape=None, pretrained=None):
  if isinstance(shape, type(None)):
    model = model_class()
  else:
    model = model_class((180, 240), pretrained=False)
  epoch, arch, state_dict = torch.load(path).values()
  model.load_state_dict(state_dict)
  model = model.cuda()
  model.eval()
    
  return model
    
def take_action(world, junction_data, road_id, lane_id, position):
  wp = world.get_map().get_waypoint(position)
  if wp.is_junction:
    junc = wp.get_junction()
    k = (road_id, lane_id, junc.id)
        
    # Si llega a una intersección por lado desconocido
    if not k in junction_data.keys():  
      action = [0, 0, 0, 1] # sin acción.
    else:
      choice = np.random.choice(junction_data[k])  
      if action == [0, 0, 0, 1]:
        if choice == 'l':
          action = [1, 0, 0, 0]
        elif choice == 'r':
          action = [0, 1, 0, 0]
        elif choice == 'f':
          action = [0, 0, 1, 0]
  else:
    road_id, lane_id = wp.road_id, wp.lane_id
    action = [0, 0, 0, 1]
        
  return action
    
def steer_correction(action, mutable_params, threshold):
  if action == [1, 0, 0, 0]: # Forzar izquierda
    if not mutable_params['s'] < -threshold:                  
      mutable_params['s'] = max(-threshold - mutable_params['ac'], -0.8)
      mutable_params['ac'] += 0.02          
    else:
      mutable_params['ac'] = 0              
  elif action == [0, 1, 0, 0]: # Forzar derecha
    if not mutable_params['s'] > threshold:                     
      mutable_params['s'] = min(threshold + mutable_params['ac'], 0.8)
      mutable_params['ac'] += 0.02
    else:
      mutable_params['ac'] = 0
  elif action == [0, 0, 1, 0]: # Forzar recto
    if not abs(mutable_params['s']) <= threshold:             
      mutable_params['s'] = -mutable_params['s']
    mutable_params['ac'] = 0                  
  elif action == [0, 0, 0, 1]:
    if isinstance(mutable_params['ema'], type(None)): # inicializar EMA
      mutable_params['ema'] = mutable_params['s']             
    else:
      mutable_params['ema'] = mutable_params['alpha']*mutable_params['s']+\
       (1-mutable_params['alpha'])*mutable_params['ema']
    if abs(mutable_params['s']) <= threshold:  
      s = mutable_params['ema']              
    else:
      mutable_params['ac'] = 0
    
def get_class_semseg(segmentation, class_id):
  seg = ((segmentation == class_id)*255).astype(np.uint8)
  seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((5, 5)))
  seg = cv2.dilate(seg, np.ones((5, 5)), iterations = 2)
  seg = cv2.erode(seg, np.ones((5, 5)), iterations = 1)
   
  return seg
    
def extract_objects_depth(depth_map, contours, veh, poles):
  objects_depth = np.round(depth_map).astype(np.uint8)
  depth_vals_veh = cv2.bitwise_and(objects_depth, veh)
  depth_vals_pol = cv2.bitwise_and(objects_depth, poles)
  objects_depth = cv2.bitwise_or(depth_vals_veh, depth_vals_pol)
  cv2.fillPoly(objects_depth, pts=[contours], color=0)

  return objects_depth

def bounding_box(min_area, x_min_thresh, signals):
  pq = PriorityQueue()
  if signals.sum() > 0:  # Comprobar si contiene semáforos
    bbox = find_contours(signals)
    for x1, y1, x2, y2 in bbox:
      area = abs(x1 - x2)*abs(y1 - y2)
      if area > min_area and x1 > x_min_thresh:
        pq.push((-area, x1, y1, x2, y2))
    
  valid = True if not pq.empty() and abs(pq.top()[0]) > min_area else False
    
  if valid:
    _, x1, y1, x2, y2 = pq.top()
    x, y = round(x1), round(y1)
    w, h = round(abs(x1 - x2)), round(abs(y1 - y2))
        
    return x, y, w, h
  return None

def cluster(crop, color_code):
  cod_val = 'na'
  inp = np.float32(crop.reshape((-1,3)))
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 4
  _, label, center = cv2.kmeans(inp, K, None, criteria,10, cv2.KMEANS_RANDOM_CENTERS)
  res = np.uint8(center[label.flatten()]).reshape((crop.shape))
  res = res * (res > 200)
  rgb_sig = [res[:, :, 0].sum(), res[:, :, 1].sum(), res[:, :, 2].sum()]
  rc, gc, bc = rgb_sig
  if rc > 0 and gc > 0:
    if bc > 0:
      cod_val = 'g'
    else:
      cod_val = 'r'
  elif rc > 0 or gc > 0 or bc > 0:
    cod_val = color_code[np.argmax(rgb_sig)]
        
  return cod_val, res, rc, gc, bc

@njit
def ff(mat, i, j, x1, y1, x2, y2):
  if mat[i, j] != 0:
    directions = np.asarray([
        (-1, -1), (0, -1), (1, -1), (-1, 0), 
        (1, 0), (-1, 1), (0, 1), (1, 1)
    ])

    mat[i, j] = 0
    x1 = min(x1, j)
    y1 = min(y1, i)
    x2 = max(x2, j)
    y2 = max(y2, i)

    for dx, dy in directions:
      x1, y1, x2, y2 = ff(mat, i+dy, j+dx, x1, y1, x2, y2)
    
  return x1, y1, x2, y2

@njit
def find_contours(img, pad=1, copy=True):
  if copy:
    local_img = img.copy()
  else:
    local_img = img
  boxes = []

  for i in range(0, local_img.shape[0]):
    for j in range(120, local_img.shape[1]):
      if local_img[i, j] != 0:
        x1, y1, x2, y2 = ff(local_img, i, j, np.inf, np.inf, -np.inf, -np.inf)
        boxes.append((int(max(x1-pad, 0)),
                      int(max(y1-pad, 0)), 
                      int(min(x2+pad, local_img.shape[1])),
                      int(min(y2+pad, local_img.shape[0]))))
                              
  return boxes

if __name__ == '__main__':
  try:
    actor_list = []
    
    # 1. Definir la pantalla de visualización de imágenes mediante la librería PyGame.
    pygame.init()
    w, h = 240, 180
    display = pygame.display.set_mode((2*w, 2*h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    try:
    # 2. Conectar con el simulador e inicializar el vehículo en el mapa.
      client = carla.Client('localhost', 2000)
      client.load_world('Town02')
          
      client.set_timeout(2.0)
      world = client.get_world()
          
      player = spawn_player(world)
      actor_list.append(player)
      set_random_weather(world)

      blueprint_library = world.get_blueprint_library()
          
      font = cv2.FONT_HERSHEY_SIMPLEX

      cam_pos = carla.Transform(carla.Location(x=1.6, z=1.7))
      camera_rgb = create_camera(cam_type='rgb',
                                 vehicle=player,
                                 pos=cam_pos,
                                 h=h, w=w,
                                 lib=blueprint_library,
                                 world=world)
      actor_list.append(camera_rgb)
          
    # 3. Cargar los parámetros de las redes.
      model_depth = load_network(CustomMobilenetDepth, 
                                 'path/to/weights',
                                 (180, 240), False)
                                     
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
      depth_preprocess = transforms.Compose([
          transforms.ToTensor(),
          lambda T: T[:3],
          normalize
      ])
          
      model_drive = load_network(CustomMobileNet, 
                                 'path/to/weights')
      
      drive_preprocess = transforms.Compose([
          transforms.Resize((224, 224), interpolation=Image.BICUBIC),
          transforms.ToTensor(),
      ])
          
      model_semseg = load_network(CustomMobilenetSemseg,
                                  'path/to/weights',
                                  (180, 240), False)
          
    # 4. Inicializar la información de las intersecciones (debido a que son de tipo T)
      junction_data = {
          # (id calle, id_carril, id_interseccion)
          (1, 1, 230): ['l', 'f'],
          (18, -1, 195): ['l', 'f'],
          (13, -1, 20): ['l', 'f'],
          (14, -1, 160): ['l', 'f'],
          (0, -1, 230): ['r', 'f'],
          (19, 1, 195): ['r', 'f'],
          (14, 1, 20): ['r', 'f'],
          (15, 1, 160): ['r', 'f'],
          (4, 1, 230): ['l', 'r'],
          (4, -1, 125): ['r', 'f'],
          (8, -1, 125): ['l', 'r'],
          (5, 1, 125): ['l', 'f'],
          (7, 1, 195): ['l', 'r'],
          (10, 1, 20): ['l', 'r'],
          (6, -1, 160): ['l', 'r'],
          (6, 1, 265): ['l', 'f'],
          (5, -1, 265): ['r', 'f'],
          (9, -1, 265): ['l', 'r'],
          (9, 1, 90): ['l', 'r'],
          (10, -1, 90): ['r', 'f'],
          (11, 1, 90): ['l', 'f'],
          (11, -1, 55): ['l', 'r'],
          (8, 1, 55): ['l', 'f'],
          (7, -1, 55): ['r', 'f']
      }
          
      contours = np.array([
          [0,139], [0, 0], [240, 0],
          [240, 139], [191, 139], [135, 90],
          [110,90], [51, 139], [0, 139],
          [0, 180], [240, 180], [240, 139], [0, 139]
      ])
          
    # 5. Iniciar un bucle de iteraciones de la simulación a 30 fps.
      with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:
        road_id, lane_id = 0, 0
        action = [0, 0, 0, 1]
              
        mutable_params = {'ema': None, 'alpha': 0.75, 'ac': 0, 
                          's': None, 't': None, 'b': 0}
        threshold = 0.2            
        min_area = 95
        x_min_thresh = 120
        color_code = ['r', 'g', 'b']
              
        # Iterando por la simulación
        while True:
          if should_quit():
            break
          clock.tick()
          snapshot, image_rgb = sync_mode.tick(timeout=2.0)
          rgb_arr = img_to_array(image_rgb)

          # Convertir a RGB
          X_img = Image.fromarray(cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB))
          X_drive = drive_preprocess(X_img).view(1, 3, 224, 224).cuda()
          X_depth = depth_preprocess(X_img).view(1, 3, 180, 240).cuda()
                  
          position = player.get_location()
      # 6. Verifica con el simulador si está en una intersección y decide el camino
          action = take_action(world, junction_data, road_id, lane_id, position)
          action_tensor = torch.tensor(action).view(1, -1).cuda()
                  
      # 7. Ingresar los valores de entrada a cada red neuronal y recibir su salida.
          with torch.no_grad():
            pred_drive = model_drive(X_drive, action_tensor).cpu()
            depth_map = model_depth(X_depth)[0, 0].cpu().numpy()
            segmentation = model_semseg(X_drive).cpu().numpy()[0]\
                                                .argmax(axis=0).astype(np.uint8)
                  
          mutable_params['t'] = round(float(pred_drive[0, 0]), 3)
          mutable_params['s'] = round(float(pred_drive[0, 1]), 3)
          mutable_params['t'] = min(mutable_params['t'], 0.5)
                  
      # 8. Corregir las oscilaciones con EMA.
      # 9. Si el vehículo gira lo suficiente, se da un impulso mediante un acumulador.
          steer_correction(action, mutable_params, threshold)
                  
      # 10. Se procesan las predicciones de la segmentación semántica para obstáculos.
          vehicles = get_class_semseg(segmentation, 10)
          poles = get_class_semseg(segmentation, 5)
                  
          objects_depth = extract_objects_depth(depth_map, contours, vehicles, poles)
                  
      # 11. Se cuenta la moda de las distancias de objetos cercanos.
          if np.sum(objects_depth) != 0:
            c = sorted(Counter(list(objects_depth.ravel())).items(), key=lambda x: -x[1])
            if c[0][0] == 0:
              moda = c[1][0]
              count = c[1][1]
            else:
              moda = c[0][0]
              count = c[0][1]
            else:
              moda = np.inf
              count = np.inf
                  
          if moda <= 4 and 40 <= count < np.inf:
            mutable_params['s'], mutable_params['t'] = 0, 0
            mutable_params['b'] += 0.15
            mutable_params['b'] = round(min(max(0, mutable_params['b']), 1), 3)
          else:
            mutable_params['b'] = 0
                  
      # 12.  Se detecta la posición de los semáforos.
          signals = get_class_semseg(segmentation, 12)
          box = bounding_box(min_area, x_min_thresh, signals)
                  
      # 13. Se predice un código de color, r para rojo o g para verde.
          code_val = 'na'
          signal_img = np.ones((180, 60, 3))
          rc, gc, bc = 0, 0, 0
          if isinstance(box, tuple):
            x, y, w, h = box
            crop = cv2.resize(cv2.cvtColor(rgb_arr[y:y+h, x:x+w, :], cv2.COLOR_BGR2RGB), 
                             (8, 24), interpolation=cv2.INTER_CUBIC)
            code_val, _, rc, gc, bc = cluster(crop, color_code) # -> 60x180
                      
      # 14. Se usa el código de color para decidir si frenar o no.
            if code_val == 'r':
              mutable_params['b'] = 1
              mutable_params['s'] = 0
              mutable_params['t'] = 0
                  
      # 15. Se envían las decisiones finales de control al vehículo.
            player.apply_control(carla.VehicleControl(throttle= mutable_params['t'], 
                                                      steer= mutable_params['s'], 
                                                      brake= mutable_params['b']))
                  
            # Visualizando la cámara y predicciones
            line1 = f"t: {round(mutable_params['t'], 2)} \
                      b: {round(mutable_params['b'], 2)} \
                      s: {round(mutable_params['s'], 2)} \
                      color: {code_val}"
            line2 = f"moda: {moda}, r: {int(rc>0)} g: {int(gc>0)} b: {int(bc>0)}"
            rgb_arr = cv2.resize(rgb_arr, None, 2, 2, cv2.INTER_CUBIC)
            pos1 = (10, 30)
            pos2 = (10, 60)
            black, green = (0, 0, 0), (0, 255, 0)
            cv2.putText(rgb_arr, line1, pos1, font, 0.6, black, 3, cv2.LINE_AA)
            cv2.putText(rgb_arr, line1, pos1, font, 0.6, green, 1, cv2.LINE_AA)
            cv2.putText(rgb_arr, line2, pos2, font, 0.6, black, 3, cv2.LINE_AA)
            cv2.putText(rgb_arr, line2, pos2, font, 0.6, green, 1, cv2.LINE_AA)
            show_window(display, rgb_arr)
            pygame.display.flip()
    finally:
      for actor in actor_list:
        actor.destroy()
      pygame.quit()
  except KeyboardInterrupt:
        print('\nFin')
