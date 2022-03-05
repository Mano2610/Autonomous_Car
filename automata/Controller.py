import glob
import os
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH = 254
IM_HEIGHT = 254


class Controller:
    def __init__(self, world, model_num, spawn_num, vehicle_id='vehicle.tesla.model3'):
        # Get vehicle blueprint
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(vehicle_id)

        # Pick a spawn point and spawn vehicle
        spawn_point = world.get_map().get_spawn_points()[spawn_num]
        self.vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        # Get models
        self.lane_model = load_model('models/lane_detection.h5')
        self.is_single_branch = model_num < 5
        self.control_model = load_model(f'models/car_control_v{model_num}{".h5" if not self.is_single_branch else ""}')

        # Set cameras
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')

        spawn_point = carla.Transform(carla.Location(x=2.5, z=1))
        self.sensor = world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
        self.sensor.listen(lambda data: self.process_img(data))

    def apply_control(self, ctrl):
        self.vehicle.apply_control(carla.VehicleControl(steer=ctrl[0], throttle=ctrl[1], brake=ctrl[2]))

    def process_single_branch(self, img):
        # Prepare image for lane detection model
        img = cv2.resize(img, (160, 80))
        img = np.array(img)
        img = img[None, :, :, :]

        # Generate lane mask and concatenate to original
        lane = self.lane_model.predict(img)[0] * 255
        enhanced_img = np.concatenate((img[0], lane), axis=2)
        enhanced_img = enhanced_img[None, :, :, :]

        control = self.control_model.predict(enhanced_img)[0]
        control = [float(x) for x in control]
        control[2] = round(control[2])
        self.apply_control(control)

    def process_triple_branch(self, img):
        # Prepare image for lane detection model
        new_size = (300, 300)
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA) / 127.5 - 1.0

        control = np.concatenate(np.concatenate(self.control_model.predict(np.expand_dims(img, axis=0))))
        control = [float(x) for x in control]
        control[2] = round(control[2])
        self.apply_control(control)

    def process_img(self, data):
        img = np.array(data.raw_data)
        img = np.reshape(img, (IM_HEIGHT, IM_WIDTH, 4))  # BGRA
        img = img[:, :, :3]  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        if self.is_single_branch:
            self.process_single_branch(img)
        else:
            self.process_triple_branch(img)

    def destroy(self):
        self.sensor.destroy()
        self.vehicle.destroy()
