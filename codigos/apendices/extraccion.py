import pickle
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torch.nn as nn
import glob
import os
import sys

sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
  sys.version_info.major,
  sys.version_info.minor,
  'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
sys.path.append('../carla')
sys.path.append('path/to/DriveNet')
sys.path.append('path/to/DepthNet')
sys.path.append('path/to/SemsegNet')

from sync_mode import CarlaSyncMode # Clase incluida en los ejemplos de Carla
import carla

import pygame
import numpy as np
import re
import cv2

def should_quit():
  # función incluida con la API de Carla
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      return True
    elif event.type == pygame.KEYUP:
      if event.key == pygame.K_ESCAPE:
        return True
  return False

def img_to_array(image):
  array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
  array = np.reshape(array, (image.height, image.width, 4))
  array = array[:, :, :3]
  return array


def show_window(surface, array, pos=(0,0)):
  if len(array.shape) > 2:
    array = array[:, :, ::-1]
  image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
  surface.blit(image_surface, pos)


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

def find_weather_presets():
  # función incluida con la API de Carla
  rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

  def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

  presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
  return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


if __name__ == '__main__':
  try:
    actor_list = []
    base_path = 'path/to/dataset'
    # Número de simulación
    n_sim = 1
    
    # Crear las carpetas para las imágenes de la simulación
    os.makedirs(f'{base_path}/Images/{n_sim}/rgb/')
    os.makedirs(f'{base_path}/Images/{n_sim}/depth/')
    os.makedirs(f'{base_path}/Images/{n_sim}/mask/')
    
    pygame.init()
    w, h = 240, 180
    display = pygame.display.set_mode((w, h), 
                                      pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    
    data = {'throttle': [], 'brake': [], 'steer': [], 'junction': []}
    try:
      client = carla.Client('localhost', 2000)
      client.set_timeout(2.0)
      
      world = client.get_world()
      player = spawn_player(world)
      actor_list.append(player)
      
      weather_index = 0
      weather_presets = find_weather_presets()
      preset = weather_presets[weather_index]
      world.set_weather(preset[0])

      blueprint_library = world.get_blueprint_library()

      cam_pos = carla.Transform(carla.Location(x=1.6, z=1.7))
      
      camera_rgb = create_camera(cam_type='rgb',
                   vehicle=player,
                   pos=cam_pos,
                   h=h, w=w,
                   lib=blueprint_library,
                   world=world)
      actor_list.append(camera_rgb)
      
      camera_semseg = create_camera(cam_type='semantic_segmentation', 
                                    vehicle=vehicle, 
                                    pos=cam_pos, 
                                    h=h, w=w,
                                    lib=blueprint_library, 
                                    world=world)
      actor_list.append(camera_semseg)

      camera_depth = create_camera(cam_type='depth', 
                                   vehicle=vehicle, 
                                   pos=cam_pos, 
                                   h=h, w=w,
                                   lib=blueprint_library, 
                                   world=world)
      actor_list.append(camera_depth)
      
      # Ceder el control del vahículo al Traffic Manager
      player.set_autopilot(True)
                
      with CarlaSyncMode(world, camera_rgb, camera_semseg, camera_depth, 
                         fps=20) as sync_mode:
        frame = 0
        while True:
          if should_quit():
            return
          clock.tick()
          
          # Obtener los fotogramas síncronos de las cámaras
          snapshot, image_rgb, image_semseg, image_depth = sync_mode.tick(timeout=2.0)
          
          # Convertir la imagen a un 2D array
          rgb_arr = img_to_array(image_rgb)
          # Visualizar la cámara RGB
          show_window(display, rgb_arr)
          pygame.display.flip()
                  
          c = player.get_control()

          if c.throttle != 0:
            location = player.get_location()
            wp = world.get_map().get_waypoint(location)
            
            # Convetir las otras cámaras en arreglos BGR
            depth_arr = img_to_array(image_depth)
            mask_arr = img_to_array(image_semseg)

            # Guardar las imagenes como archivos
            cv2.imwrite(f'{base_path}/Images/{n_sim}/rgb/{frame}.png', rgb_arr)
            cv2.imwrite(f'{base_path}/Images/{n_sim}/mask/{frame}.png', mask_arr)
            
            # Crear una entrada de datos en el diccionario
            data['throttle'].append(min(c.throttle, 0.4))
            data['brake'].append(c.brake)
            data['steer'].append(c.steer)
            data['junction'].append(wp.is_junction)

            frame += 1

            if frame == 8000:
              # Si se llega a los 8000 fotogramas, terminar simulacion
              return
            if frame % 1000 == 0:
              print(f'Frame: {frame}')
    finally:
      print('destroying actors.')
      for actor in actor_list:
          actor.destroy()
      world.destroy()
      pygame.quit()
      
      # Convertir el diccionario en un dataframe
      df = pd.DataFrame.from_dict(data)
      # Exportar el dataframe como csv para la simulación
      df.to_csv(f'{base_path}/Dfs/{n_sim}.csv', index=False)
      print('done.')
      
  except KeyboardInterrupt:
    print('\nFin')
