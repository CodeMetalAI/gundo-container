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

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
# client.moveToPositionAsync(-10, 10, -10, 5).join()

yaw_rate = 0.01
max_roll = np.deg2rad(75)
max_pitch = np.deg2rad(75)
input_rate = 0.01
capture_rate = 0.2
neutral_rate = 0.2

throttle = float(.63)
yaw = 0
roll = float(0)
pitch = .1

model = get_model()


def get_image():
    """ query client for current observation """
    responses = client.simGetImages([
        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  # vehicle_name=vehicleName
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


def controller_task(inputs):
    yaw = (inputs["yaw+"] - inputs["yaw-"]) * yaw_rate
    roll = inputs["rx"] * max_roll
    pitch = inputs["ry"] * max_pitch
    throttle = float(np.clip(inputs["y"], 0, 1))

    client.moveByRollPitchYawrateThrottleAsync(roll=roll, pitch=pitch, yaw_rate=yaw,
                                               throttle=throttle, duration=input_rate)


def chaseOrange():
    img_rgb = get_image()
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
                                        duration=input_rate)  # , vehicle_name=vehicleName
    print("chasing orange")


def chaseTarget():
    """ chases target and returns true if we have positive detections, else false """
    img_rgb = get_image()
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
                                        duration=input_rate)  # , vehicle_name=vehicleName
    print("chasing target")
    return True


def neutral_task():
    """ fly in circles and return True if we identify a target """
    # fly around in a circle
    radius = 10.0
    velocity = 2.0

    parameters = radius_to_motion_parameters(radius, velocity)
    client.moveByRollPitchYawrateThrottleAsync(roll=parameters["roll"],
                                               pitch=parameters["pitch"],
                                               yaw_rate=parameters["yaw_rate"],
                                               throttle=parameters["throttle"],
                                               duration=input_rate)
    img_rgb = get_image()
    bboxes = get_detections(img_rgb)
    return bool(len(bboxes) > 0)


def capture_loop():
    now = datetime.now()
    ts = now.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
    os.mkdir(f"capture_data/{ts}")
    os.mkdir(f"capture_data/{ts}/images")
    os.mkdir(f"capture_data/{ts}/masks")
    os.mkdir(f"capture_data/{ts}/motion")

    i = 0

    camera_info = client.simGetCameraInfo(str(0))

    with open(f"capture_data/{ts}/motion/camera_info.txt", 'a') as the_file:
        the_file.write(str(camera_info.fov) + "\n")
        the_file.write(str(camera_info.proj_mat.matrix) + "\n")
        the_file.write(str([[256 // 2, 0, 256 // 2],
                            [0, 256 // 2, 256 // 2],
                            [0, 0, 1]]))

    while True:
        camera_info = client.simGetCameraInfo(str(0))
        pos = str([camera_info.pose.position.x_val,
                   camera_info.pose.position.y_val,
                   camera_info.pose.position.z_val,
                   camera_info.pose.orientation.x_val,
                   camera_info.pose.orientation.y_val,
                   camera_info.pose.orientation.z_val,
                   camera_info.pose.orientation.w_val])
        with open(f"capture_data/{ts}/motion/pos_{i}.txt", 'a') as the_file:
            the_file.write(pos)

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


def motion_loop(captureMode=False, chaseOrange=False, chaseTargetFlag=True):
    controller = XboxController() if captureMode else None
    target_locked = False
    while True:
        print("Capture: {}, Chase Orange: {}, Chase Target: {}".format(captureMode, chaseOrange, chaseTargetFlag))
        if controller:
            inputs = controller.read()
            controller_task(inputs)
        elif chaseOrange:
            chaseOrange()
        elif chaseTargetFlag:
            target_locked = chaseTarget()
            if not target_locked:
                chaseTargetFlag = False
        else:
            target_locked = neutral_task()
            if target_locked:
                chaseTargetFlag = True
        time.sleep(input_rate)


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--chase_orange", type=int, default=0,
                        help='specify 1 for chasing orange')
    parser.add_argument("--chase_target", type=int, default=1,
                        help='specify 1 for chasing target')
    parser.add_argument("--capture", type=int, default=0,
                        help='specify 1 for running capture')
    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    p1 = Process(target=motion_loop, \
                 args=(bool(args.capture), \
                       bool(args.chase_orange), \
                       bool(args.chase_target),) \
                 )
    p1.start()

    if bool(args.capture):
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
