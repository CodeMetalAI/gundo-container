import airsim
import numpy as np
import os, time
from control.gamepad import XboxController

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
# client.moveToPositionAsync(-10, 10, -10, 5).join()

controller = XboxController()
yaw_rate = np.pi
max_roll = np.deg2rad(45)
max_pitch = np.deg2rad(45)
input_rate = 0.01

while True:
    inputs = controller.read()
    roll = inputs["rx"] * max_roll
    pitch = inputs["ry"] * max_pitch
    yaw = (inputs["yaw+"] - inputs["yaw-"]) * yaw_rate
    throttle = float(np.clip(inputs["y"], 0, 1))

    client.moveByRollPitchYawrateThrottleAsync(roll=roll, pitch=pitch, yaw_rate=yaw, throttle=throttle, duration=input_rate)
    time.sleep(input_rate)

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
