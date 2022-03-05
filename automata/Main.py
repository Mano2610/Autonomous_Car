import glob
import os
import sys
from signal import signal, SIGINT

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
from Controller import Controller


def handler(sign, frame):
    for actor in actor_list:
        actor.destroy()
    sys.exit(0)


actor_list = []
signal(SIGINT, handler)

parser = argparse.ArgumentParser(description='Instantiates controller')
parser.add_argument(
        '--model',
        metavar='m',
        default='2',
        help='Model number (1-6)',
        type=int)
parser.add_argument(
        '--spawn',
        metavar='s',
        default='3',
        help='Spawn point (0-255)',
        type=int)
args = parser.parse_args()

# Instantiate client
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(2)

world = client.get_world()
controller = Controller(world, args.model, args.spawn)
actor_list.append(controller)

while True:
    pass
