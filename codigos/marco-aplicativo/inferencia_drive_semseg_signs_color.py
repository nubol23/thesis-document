import pickle
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torch.nn as nn
import glob
import os
import sys

from scipy import stats

try:
    sys.path.append(glob.glob('/home/nubol23/Desktop/Installers/CARLA_0.9.9.4/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
try:
    sys.path.append('/home/nubol23/Desktop/Installers/CARLA_0.9.9.4/PythonAPI/carla')
except IndexError:
    pass
try:
    sys.path.append('/home/nubol23/Desktop/Codes/Tesis/NNTrain/Drive')
    sys.path.append('/home/nubol23/Desktop/Codes/Tesis/Train/Depth')
    sys.path.append('/home/nubol23/Desktop/Codes/Tesis/NNTrain/Segmentation')
except IndexError:
    pass
from custom_mobilenet import CustomMobileNet, CustomMobileNetExt
from models import CustomMobilenetDepth
from semseg_model import CustomMobilenetSemseg

import carla
from carla.libcarla import Waypoint, Transform
from agents.navigation.behavior_agent import BehaviorAgent

from collections import Counter
import random
import pygame
import numpy as np
import queue
import re
import math
import weakref
import collections
import cv2
from PIL import Image
from skimage.measure import find_contours
from priority_queue import PriorityQueue
import imutils


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class World(object):
    """ Class representing the surrounding environment """

    # def __init__(self, carla_world, args):
    def __init__(self, carla_world):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        # self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = -1
        # self._gamma = args.gamma
        # self.restart(args)
        self.recording_enabled = False
        self.recording_start = 0

        blueprint = self.world.get_blueprint_library().filter('model3')[0]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        print("Spawning the player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player)
        self.gnss_sensor = GnssSensor(self.player)
        # self.camera_manager = CameraManager(self.player, self._gamma)
        # self.camera_manager.transform_index = cam_pos_id
        # self.camera_manager.set_sensor(cam_index, notify=False)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.player.get_world().set_weather(preset[0])

    def random_weather(self):
        if self._weather_index == -1:
            idx = -1
            while not idx in [-1, 5, 6, 10, 11, 12, 13]:
                idx = random.randint(0, len(self._weather_presets)-1)
            # self._weather_index = random.randint(0, len(self._weather_presets)-1)
            self._weather_index = idx
            preset = self._weather_presets[self._weather_index]
            self.player.get_world().set_weather(preset[0])
            
            # with open('transforms', 'a') as f:
            #         f.write(str({'Weather': self._weather_index})+'\n')
        else:
            preset = self._weather_presets[self._weather_index]
            self.player.get_world().set_weather(preset[0])

    # def render(self, display):
    #     """Render world"""
    #     self.camera_manager.render(display)

    # def destroy_sensors(self):
    #     """Destroy sensors"""
    #     self.camera_manager.sensor.destroy()
    #     self.camera_manager.sensor = None
    #     self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            # self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]


def draw_image(surface, image, pos=(0, 0), blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    # print(array.shape)
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    # array = cv2.resize(array, (224, 224))
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, pos)


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

import matplotlib.pyplot as plt
def main():
    #np.random.seed(42)
    
    actor_list = []
    pygame.init()
    world = None

    w, h = 240, 180

    # display = pygame.display.set_mode((w, h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    # display = pygame.display.set_mode((w*3, h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    display = pygame.display.set_mode((w*4 + 60, h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    # display = pygame.display.set_mode((224, 224), pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    try:
        client = carla.Client('localhost', 2000)
        # client.load_world('Town04')
        client.set_timeout(2.0)
        world = World(client.get_world())

        world.random_weather()

        vehicle = world.player
        blueprint_library = world.world.get_blueprint_library()

        cam_pos = carla.Transform(carla.Location(x=1.6, z=1.7))

        camera_rgb = create_camera(cam_type='rgb',
                                   vehicle=vehicle,
                                   pos=cam_pos,
                                   h=h, w=w,
                                   lib=blueprint_library,
                                   world=world.world)
        actor_list.append(camera_rgb)
    
        # DEPTH
        model_depth = CustomMobilenetDepth((180, 240), pretrained=False)
        # epoch, arch, state_dict = torch.load('Depth/weights_mse/d_mob_66.pth.tar').values()
        epoch, arch, state_dict = torch.load('/home/nubol23/Desktop/Codes/Tesis/Train/Depth/weights/c_mob_20.pth.tar').values()
        model_depth.load_state_dict(state_dict)
        model_depth = model_depth.cuda()
        model_depth.eval()
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([
            transforms.ToTensor(),
            lambda T: T[:3],
            normalize
        ])

        preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
        
        # DRIVE
        # model_drive_path = 'Drive/weights_extended_ams/mob_drive_30.pth.tar'
        model_drive_path = 'Drive/weights/mob_drive_24.pth.tar'
        epoch, arch, state_dict = torch.load(model_drive_path).values()
        model_drive = CustomMobileNet()
        # model_drive = CustomMobileNetExt()
        model_drive.load_state_dict(state_dict)
        model_drive = model_drive.cuda()
        model_drive.eval()
        
        # SEMSEG
        model_semseg = CustomMobilenetSemseg((180, 240), pretrained=False)
        model_semseg = model_semseg.cuda()
        epoch, arch, state_dict = torch.load('Segmentation/weights/s_mob_7.pth.tar').values()
        model_semseg.load_state_dict(state_dict)
        
        
        junction_data = {
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
        
        classes = { 
             0: [0, 0, 0],         # None 
             1: [70, 70, 70],      # Buildings 
             2: [190, 153, 153],   # Fences 
             3: [72, 0, 90],       # Other 
             4: [220, 20, 60],     # Pedestrians 
             5: [153, 153, 153],   # Poles 
             6: [157, 234, 50],    # RoadLines 
             7: [128, 64, 128],    # Roads 
             8: [244, 35, 232],    # Sidewalks 
             9: [107, 142, 35],    # Vegetation 
             10: [0, 0, 255],      # Vehicles 
             11: [102, 102, 156],  # Walls 
             12: [220, 220, 0]     # TrafficSigns 
        }
        
        # Create a synchronous mode context.
        with CarlaSyncMode(world.world, camera_rgb, fps=30) as sync_mode:
        # with CarlaSyncMode(world.world, camera_rgb, camera_semseg, camera_depth, fps=20) as sync_mode:
            # frame = 0
            road_id, lane_id = 0, 0
            action = [0, 0, 0, 1]
            ac = 0              # acumulador incremento forzado de giro

            ema = None          # Ema iniciado en None
            alpha = 0.75        # Parámetro del EMA
            prev_throttle = 0.7 # EXT
            b = 0
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                # snapshot, image_rgb, image_semseg, image_depth = sync_mode.tick(timeout=2.0)
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # Draw the display.
                rgb_arr = img_to_array(image_rgb)
                #plt.imshow(rgb_arr)
                #plt.show()
                #break

                # DRIVE
                X_img = Image.fromarray(cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB))
                X = preprocess(X_img).view(1, 3, 224, 224).cuda()
                X_D = trans(X_img).view(1, 3, 180, 240).cuda()
                
                loc = world.player.get_location()
                wp = world.map.get_waypoint(loc)
                if wp.is_junction:
                    junc = wp.get_junction()
                    k = (road_id, lane_id, junc.id)
                    
                    if not k in junction_data.keys():  # Si llega a una intersección por lado desconocido
                        action == [0, 0, 0, 1]
                    else:
                        choice = np.random.choice(junction_data[k])  # Seleccionar una dirección
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
                
                action_tensor = torch.tensor(action).view(1, -1)
                action_tensor = action_tensor.cuda()
                prev_throttle_tensor = torch.tensor(prev_throttle).view(1, -1).cuda() # EXT
                with torch.no_grad():
                    out = model_drive(X, action_tensor).cpu()
                    # out = model_drive(X, action_tensor, prev_throttle_tensor).cpu()
                    depth_map = model_depth(X_D)[0, 0].cpu().numpy()
                    segmentation = model_semseg(X).cpu().numpy()[0].argmax(axis=0).astype(np.uint8)
                
                t, s = round(float(out[0, 0]), 3), round(float(out[0, 1]), 3)
                # t, s = float(out[0, 0]), float(out[0, 1])
                t = min(t, 0.5)  # Clip de la aceleración para evitar carrera
                
                pre_s = s       # Valor previo del steer con fines de debug
                threshold = 0.2
                if action == [1, 0, 0, 0]:
                    action_str = 'left'
                    if not s < -threshold:  # Si no está en un rango de giro mínimo
                        s = -threshold      # forzar giro
                        s -= ac             # incrementar el angulo conforme pasa el tiempo si no se corrige
                        s = max(s, -0.8)    # clip del valor
                        ac += 0.02          # acumular el incremento 0.05
                    else:
                        ac = 0 # 0.02
                    ema_str = ''
                elif action == [0, 1, 0, 0]:
                    action_str = 'right'
                    if not s > threshold:  # similar al caso de arriba pero al otro lado
                        s = threshold
                        s += ac
                        s = min(s, 0.8)
                        ac += 0.02
                    else:
                        ac = 0
                    ema_str = ''
                elif action == [0, 0, 1, 0]:
                    action_str = 'forward'
                    if not abs(s) <= threshold:  # Si debe ir recto pero gira a la fuerza
                        s = -s                   # Dar volantazo para corregir la posición
                    ac = 0                       # Restaurar acumulador de giro
                    ema_str = ''
                elif action == [0, 0, 0, 1]:
                    if isinstance(ema, type(None)):  # Si ema está en t=0
                        ema = s                      # inicializar
                    else:
                        ema = alpha*s + (1-alpha)*ema  # Dar un paso de EMA
                        
                    if abs(s) <= threshold:  # Si tiene poco giro (esta en una recta)
                        ema_str = 'ema'
                        s = ema              # Usar el valor del EMA
                    else:                    # Si está en una curva del trayecto que no es intersección
                        ema_str = ''         # No hacer nada
                        
                    action_str = 'no action'
                    ac = 0      # Reiniciar acumulador

                # DEPTH
                #roi = depth_map[90:135, 105:145]
                #moda = stats.mode(np.round(roi).astype(np.int32), axis=None)[0][0]
                # b = 0
                #if moda <= 4:
                #    t, s, b = 0, 0, 1

                # print(round(t, 3), round(s, 3), pre_s, s == pre_s, action_str, ema_str, moda, sep='\t')  # Debug

                # show_window(display, pred_show_merge)
                # show_window(display, rgb_arr)
                
                depth_vals = depth_map.copy()
                depth_map = np.round(depth_map/depth_map.max()*255).astype(np.uint8)
                depth_img = cv2.merge([depth_map, depth_map, depth_map])
                
                ped = ((segmentation == 4)*255).astype(np.uint8)  # red
                ped = cv2.morphologyEx(ped, cv2.MORPH_OPEN, np.ones((5, 5)))
                ped = cv2.dilate(ped, np.ones((5, 5)), iterations = 2)
                ped = cv2.erode(ped, np.ones((5, 5)), iterations = 1)
                
                veh = ((segmentation == 10)*255).astype(np.uint8) # green
                veh = cv2.morphologyEx(veh, cv2.MORPH_OPEN, np.ones((5, 5)))
                veh = cv2.dilate(veh, np.ones((5, 5)), iterations = 2)
                veh = cv2.erode(veh, np.ones((5, 5)), iterations = 1)
                
                poles = ((segmentation == 5)*255).astype(np.uint8)
                poles = cv2.morphologyEx(poles, cv2.MORPH_OPEN, np.ones((5, 5)))
                poles = cv2.dilate(poles, np.ones((5, 5)), iterations = 2)
                poles = cv2.erode(poles, np.ones((5, 5)), iterations = 1)
                
                sig = ((segmentation == 12)*255).astype(np.uint8) # blue
                sig = cv2.morphologyEx(sig, cv2.MORPH_OPEN, np.ones((5, 5)))
                sig = cv2.dilate(sig, np.ones((5, 5)), iterations = 2)
                sig = cv2.erode(sig, np.ones((5, 5)), iterations = 1)
                
                # semseg_img = cv2.merge([ped, (np.logical_or(veh, poles)*255).astype(np.uint8), sig])
                # semseg_img = cv2.cvtColor(semseg_img, cv2.COLOR_RGB2BGR)
                
                # semseg_img = np.round(segmentation/segmentation.max()*255).astype(np.uint8)
                # semseg_img = cv2.merge([semseg_img, semseg_img, semseg_img])
                
                # mix with depth
                #semseg_img = cv2.bitwise_and(depth_map, veh)
                #semseg_img = ((segmentation == 7)*255).astype(np.uint8)
                
                contours = np.array([
                    [0,180],
                    [0, 0],
                    [240, 0],
                    [240, 180],
                    [135, 90],
                    [110,90],
                    [0, 180]
                ])
                #cv2.fillPoly(semseg_img, pts =[contours], color=0)
                
                depth_vals_and = np.round(depth_vals).astype(np.uint8)
                depth_vals_veh = cv2.bitwise_and(depth_vals_and, veh)
                depth_vals_pol = cv2.bitwise_and(depth_vals_and, poles)
                depth_vals_and = cv2.bitwise_or(depth_vals_veh, depth_vals_pol)
                cv2.fillPoly(depth_vals_and, pts=[contours], color=0)
                cv2.fillPoly(depth_vals_and, pts=[np.array([[0,180],[0,140],[240,140],[240,180],[0,180]])], color=0)
                
                # cv2.fillPoly(depth_map, pts=[contours], color=0)
                # depth_img = cv2.merge([depth_map, depth_map, depth_map])
                
                dv = cv2.bitwise_and(depth_map, veh)
                dp = cv2.bitwise_and(depth_map, poles)
                ds = cv2.bitwise_or(dv, dp)
                # cv2.fillPoly(ds, pts=[contours], color=0)
                # cv2.fillPoly(ds, pts=[np.array([[0,180],[0,140],[240,140],[240,180],[0,180]])], color=0)
                semseg_img = cv2.merge([ds, ds, ds])
                
                colored = np.apply_along_axis(lambda p: classes[p[0]], 1, segmentation.reshape(1, 180*240).T)
                colored = colored.reshape(180, 240, 3)
                # seg = np.round(segmentation/segmentation.max() * 255).astype(np.uint8)
                # colored = cv2.merge([seg, seg, seg])
                
                if np.sum(depth_vals_and) != 0:
                    c = sorted(Counter(list(depth_vals_and.ravel())).items(), key=lambda x: -x[1])
                    # print(c)
                    if c[0][0] == 0:
                        moda = c[1][0]
                        count = c[1][1]
                    else:
                        moda = c[0][0]
                        count = c[0][1]
                else:
                    moda = np.inf
                    count = np.inf
                
                prev_throttle = t
                
                # if moda <= 2 and 40 <= count < np.inf:
                if moda <= 4 and 40 <= count < np.inf:
                    s, t, b = 0, 0, b+0.15
                    b = max(0, b)
                    b = min(b, 1)
                    b = round(b, 3)
                else:
                    b = 0
                
                # BOUNDING BOX DETECTOR
                # min_area = 90
                min_area = 95
                x_min_thresh = 120
                pq = PriorityQueue()
                if sig.sum() > 0:  # Check if the image contains traffic signs
                    contours = find_contours(sig, 0.8)  # Find contours
                    for contour in contours:
                        Ymin = np.min(contour[:,0])
                        Ymax = np.max(contour[:,0])
                        Xmin = np.min(contour[:,1])
                        Xmax = np.max(contour[:,1])
                        area = abs(Xmin - Xmax)*abs(Ymin - Ymax)
                        if area > min_area and Xmin > x_min_thresh:
                            pq.push((-area, Xmin, Ymin, Xmax, Ymax))
                
                semseg_arr = cv2.merge([sig, sig, sig])
                
                valid = True if not pq.empty() and abs(pq.top()[0]) > min_area else False
                
                crop = np.ones((180, 240, 3))
                cod_val = 'na'
                rc, gc, bc = 0, 0, 0
                if valid:
                    _, x1, y1, x2, y2 = pq.top()
                    x, y = round(x1), round(y1)
                    w, h = round(abs(x1 - x2)), round(abs(y1 - y2))
                    cv2.rectangle(semseg_arr, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    crop = cv2.resize(cv2.cvtColor(rgb_arr[y:y+h, x:x+w, :], cv2.COLOR_BGR2RGB), (8, 24), interpolation=cv2.INTER_CUBIC)
                    
                    # K-MEANS
                    Z = crop.reshape((-1,3))
                    Z = np.float32(Z)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    K = 4
                    ret, label, center = cv2.kmeans(Z, K, None, criteria,10, cv2.KMEANS_RANDOM_CENTERS)
                    center = np.uint8(center)
                    res = center[label.flatten()]
                    res2 = res.reshape((crop.shape))
                    res2 = res2 * (res2 > 200)
                    crop = res2
                    rgb_sig = [res2[:, :, 0].sum(), res2[:, :, 1].sum(), res2[:, :, 2].sum()] # Sumamos los pixeles mayores a 200
                    rc, gc, bc = rgb_sig
                    cod = ['r', 'g', 'b']
                    if rc > 0 and gc > 0:
                        if bc > 0:
                            cod_val = 'g'
                        else:
                            cod_val = 'r'
                    elif rc > 0 or gc > 0 or bc > 0:
                        cod_val = cod[np.argmax(rgb_sig)]
                    
                    
                if crop.shape != (180, 240, 3):
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    crop = imutils.resize(crop, height=180)                    
                    #dx = abs(crop.shape[1] - 240)
                    #filler = np.zeros((180, dx, 3))
                    #crop = np.hstack([crop, filler])
                    
                if cod_val == 'r':
                    b = 1
                    s = 0
                    t = 0
                
                # print(round(t, 3), b, round(s, 3), action_str, ema_str, moda, count, cod_val, sep='\t')  # Debug
                print(round(t, 3), b, round(s, 3), cod_val, [rc, gc, bc], moda, sep='\t')  # Debug
                world.player.apply_control(carla.VehicleControl(throttle=t, steer=s, brake=b))  # Aplicar control
                
                show_window(display, np.hstack([rgb_arr, depth_img, colored, semseg_img, crop]))
                # show_window(display, np.hstack([rgb_arr, depth_img, semseg_arr]))
                # show_window(display, np.hstack([rgb_arr, depth_img, crop]))
                pygame.display.flip()
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        world.destroy()
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
