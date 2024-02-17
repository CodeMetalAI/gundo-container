import airsim
import numpy as np
from PIL import Image
import os, time
from multiprocessing import Process
from control.gamepad import XboxController
from datetime import datetime

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
# client.moveToPositionAsync(-10, 10, -10, 5).join()

CAPTURE_MODE = True

controller = XboxController()
yaw_rate = np.pi
max_roll = np.deg2rad(75)
max_pitch = np.deg2rad(75)
input_rate = 0.01

capture_rate = 0.2

def control_loop():
    while True:
        inputs = controller.read()
        roll = inputs["rx"] * max_roll
        pitch = inputs["ry"] * max_pitch
        yaw = (inputs["yaw+"] - inputs["yaw-"]) * yaw_rate
        throttle = float(np.clip(inputs["y"], 0, 1))

        client.moveByRollPitchYawrateThrottleAsync(roll=roll, pitch=pitch, yaw_rate=yaw, throttle=throttle, duration=input_rate)
        time.sleep(input_rate)

def capture_loop():
    now = datetime.now()
    ts = now.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
    os.mkdir(f"capture_data/{ts}")
    i = 0

    while True:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img = Image.fromarray(img_rgb)
        img.save(f"capture_data/{ts}/cap_{i}.png")
        i += 1

        time.sleep(capture_rate)

if __name__ == '__main__':
    p1 = Process(target=control_loop)
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
