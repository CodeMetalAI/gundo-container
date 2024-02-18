import math

import airsim
import numpy as np
import os
import time
import configargparse
from multiprocessing import Process
from control.gamepad import XboxController
from datetime import datetime
from common.motion import radius_to_motion_parameters
from yolov5.detection import get_model, detect

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
vehicleNames = client.listVehicles()
vehicleCapture = {}
vehicleStates = {}
for name in vehicleNames:
    client.enableApiControl(True, name)
    client.armDisarm(True, name)
    vehicleStates.update({name: "chase_orange"})
    vehicleCapture.update({name: False})
    takeoff = client.takeoffAsync(vehicle_name=name)
    takeoff.join()

yaw_rate = 0.01
max_roll = np.deg2rad(75)
max_pitch = np.deg2rad(75)
input_rate = 0.01
capture_rate = 0.2
neutral_rate = 0.2

throttle = float(.73)
yaw = 0
roll = float(0)
pitch = .1

model = get_model()


def get_image(vehicle_name):
    """ query client for current observation """
    responses = client.simGetImages([
        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)], vehicle_name=vehicle_name)
    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    return img_rgb


def get_detections(img, thresh=0.8):
    """ detect objects with conf above threshold """
    bboxes = []
    boxes = detect(img, model)
    for box in boxes:
        bbox = box["box"]
        conf = box["conf"]
        label = box["label"].split(" ")[0]
        if conf > thresh:
            bboxes.append(bbox)
    return bboxes


def controller_task(inputs, vehicle_name):
    yaw = (inputs["yaw+"] - inputs["yaw-"]) * yaw_rate
    roll = inputs["rx"] * max_roll
    pitch = inputs["ry"] * max_pitch
    throttle = float(np.clip(inputs["y"], 0, 1))

    client.moveByRollPitchYawrateThrottleAsync(roll=roll, pitch=pitch, yaw_rate=yaw,
                                               throttle=throttle, duration=input_rate,
                                               vehicle_name=vehicle_name)


def chaseOrange(vehicle_name):
    img_rgb = get_image(vehicle_name=vehicle_name)
    target = np.array([86, 120, 190])  # B,G,R
    target = target.reshape(1, 1, 3)
    distanc = img_rgb - target
    distanc = distanc.astype(np.float32)
    sum_across_3rd_dim = np.sum(distanc ** 2, axis=2)
    targetPixel = np.unravel_index(np.argmin(sum_across_3rd_dim), sum_across_3rd_dim.shape)

    Pgain = .01
    roll = float(0)
    pitch = .4

    # elErr = targetPixel[0] - 72
    azErr = targetPixel[1] - 128
    yaw_rate = float(-azErr * Pgain)

    client.moveByRollPitchYawrateZAsync(roll=roll, pitch=pitch, yaw_rate=yaw_rate, z=-5,
                                        duration=input_rate, vehicle_name=vehicle_name)
    print("chasing orange")


def chaseTarget(vehicle_name):
    """ chases target and returns true if we have positive detections, else false """
    img_rgb = get_image(vehicle_name=vehicle_name)
    bboxes = get_detections(img_rgb)
    if not len(bboxes):
        return False
    # TODO: update target based on bboxes
    target = np.array([86, 120, 190])  # B,G,R
    target = target.reshape(1, 1, 3)
    distanc = img_rgb - target
    distanc = distanc.astype(np.float32)
    sum_across_3rd_dim = np.sum(distanc ** 2, axis=2)
    targetPixel = np.unravel_index(np.argmin(sum_across_3rd_dim), sum_across_3rd_dim.shape)

    Pgain = .01
    roll = float(0)
    pitch = .4

    # elErr = targetPixel[0] - 72
    azErr = targetPixel[1] - 128
    yaw_rate = float(-azErr * Pgain)

    client.moveByRollPitchYawrateZAsync(roll=roll, pitch=pitch, yaw_rate=yaw_rate, z=-5,
                                        duration=input_rate, vehicle_name=vehicle_name)
    return True


def neutral_task(vehicle_name):
    """ fly in circles and return True if we identify a target """
    # fly around in a circle
    radius = 10.0
    velocity = 2.0

    parameters = radius_to_motion_parameters(radius, velocity)
    client.moveByRollPitchYawrateThrottleAsync(roll=parameters["roll"],
                                               pitch=parameters["pitch"],
                                               yaw_rate=parameters["yaw_rate"],
                                               throttle=parameters["throttle"],
                                               duration=input_rate,
                                               vehicle_name=vehicle_name)
    img_rgb = get_image(vehicle_name=vehicle_name)
    bboxes = get_detections(img_rgb)
    return bool(len(bboxes) > 0)


def capture_loop():
    now = datetime.now()
    ts = now.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
    os.mkdir(f"capture_data/{ts}")
    os.mkdir(f"capture_data/{ts}/images")
    os.mkdir(f"capture_data/{ts}/masks")
    i = 0

    while True:
        for vehicle_name in vehicleNames:
            if vehicleCapture[vehicle_name]:
                responses = client.simGetImages(
                    [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
                    vehicle_name=vehicle_name)
                response = responses[0]
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                airsim.write_png(f"capture_data/{ts}/images/cap_{i}.png", img_rgb)

                responses = client.simGetImages(
                    [airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)],
                    vehicle_name=vehicle_name)
                response = responses[0]
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(response.height, response.width, 3)
                airsim.write_png(f"capture_data/{ts}/masks/cap_{i}.png", img_rgb)

                i += 1

        time.sleep(capture_rate)


def motion_loop(captureMode=False):
    controller = XboxController() if captureMode else None
    while True:
        state = client.getMultirotorState(vehicle_name=vehicleNames[0])
        northA = state.kinematics_estimated.linear_acceleration.x_val
        eastA = state.kinematics_estimated.linear_acceleration.y_val
        downA = state.kinematics_estimated.linear_acceleration.z_val

        # large acceleration estimate collision
        acc = math.sqrt(northA * northA + eastA * eastA + downA * downA)
        threshold = 3.0
        if acc > 3.0:
            vehicleStates[vehicleNames[1]] = 'neutral'
            vehicleStates[vehicleNames[2]] = 'neutral'
        # don't want to run logging loop too often
        for vehicle_name in vehicleNames:
            if vehicleStates[vehicle_name] == 'capture':
                inputs = controller.read()
                controller_task(inputs, vehicle_name)
            elif vehicleStates[vehicle_name] == 'chase_orange':
                chaseOrange(vehicle_name)
            elif vehicleStates[vehicle_name] == 'chase_target':
                target_locked = chaseTarget(vehicle_name)
                if not target_locked:
                    vehicleStates[vehicle_name] = 'neutral'
            else:
                target_locked = neutral_task(vehicle_name)
                if target_locked:
                    vehicleStates[vehicle_name] = 'chase_target'
            time.sleep(input_rate)


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--capture", type=int, default=0,
                        help='specify 1 for running capture')
    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    p1 = Process(target=motion_loop, args=[bool(args.capture)])
    p1.start()

    if bool(args.capture):
        capture_loop()
