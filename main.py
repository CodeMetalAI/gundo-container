import airsim
import numpy as np
from PIL import Image
import os, time
from multiprocessing import Process
from control.gamepad import XboxController
from datetime import datetime
from common.motion import radius_to_motion_parameters

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
# client.moveToPositionAsync(-10, 10, -10, 5).join()

CAPTURE_MODE = False
controller = XboxController() if CAPTURE_MODE else None

yaw_rate = 0.01
max_roll = np.deg2rad(75)
max_pitch = np.deg2rad(75)
input_rate = 0.01
capture_rate = 0.2
neutral_rate = 0.2

Pgain = .01
Igain = .01
throttle = float(.63)
yaw = 0
roll = float(0)
pitch = .1


def controller_task(inputs):
    yaw = (inputs["yaw+"] - inputs["yaw-"]) * yaw_rate
    roll = inputs["rx"] * max_roll
    pitch = inputs["ry"] * max_pitch
    throttle = float(np.clip(inputs["y"], 0, 1))

    client.moveByRollPitchYawrateThrottleAsync(roll=roll, pitch=pitch, yaw_rate=yaw,
                                               throttle=throttle, duration=input_rate)

def target_task():
    # control code to fly towards target
    return


def neutral_task():
    # fly around in a circle
    radius = 10.0
    velocity = 2.0

    parameters = radius_to_motion_parameters(radius, velocity)
    client.moveByRollPitchYawrateThrottleAsync(roll=parameters["roll"],
                                               pitch=parameters["pitch"],
                                               yaw_rate=parameters["yaw_rate"],
                                               throttle=parameters["throttle"],
                                               duration=input_rate)


def capture_loop():
    now = datetime.now()
    ts = now.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
    os.mkdir(f"capture_data/{ts}")
    os.mkdir(f"capture_data/{ts}/images")
    os.mkdir(f"capture_data/{ts}/masks")
    i = 0

    while True:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        airsim.write_png(f"capture_data/{ts}/images/cap_{i}.png", img_rgb)

        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        airsim.write_png(f"capture_data/{ts}/masks/cap_{i}.png", img_rgb)

        i += 1

        time.sleep(capture_rate)

if __name__ == '__main__':
    p1 = Process(target=control_loop)
    p1.start()

    if CAPTURE_MODE:
        capture_loop()


def motion_loop():
    while True:
        if controller:
            inputs = controller.read()
            controller_task(inputs)
        else:
            neutral_task()
        time.sleep(input_rate)


if __name__ == '__main__':
    p1 = Process(target=motion_loop)
    p1.start()

    if CAPTURE_MODE:
        capture_loop()

# # take images
# responses = client.simGetImages([
#     airsim.ImageRequest("0", airsim.ImageType.DepthVis),
#     airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
# print('Retrieved images: %d', len(responses))

# # do something with the images
# for response in responses:
#     if response.pixels_as_float:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#         airsim.write_pfm(os.path.normpath('temp/py1.pfm'), airsim.get_pfm_array(response))
#     else:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         airsim.write_file(os.path.normpath('temp/py1.png'), response.image_data_uint8)
