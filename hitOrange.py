import airsim
import numpy as np
import os, time
from yolov5.detection import get_model, detect
import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--detect", type=int, default=0,
                        help='specify 1 for running detection')
    return parser
parser = config_parser()
args = parser.parse_args()

# delete the group that includes the boxes

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
# client.moveToPositionAsync(-10, 10, -10, 5).join()

ts = float(.2)
Pgain = .01
Igain = .01
throttle = float(.6)
roll = float(0)
pitch = .1
yaw_rate = .01

# z = 20
yaw = 0 # yaw error integrated 

if args.detect:
    model = get_model()

while True:
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])

    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
    img_rgb = img1d.reshape(response.height, response.width, 3)
    if args.detect:
        boxes = detect(img_rgb, model)
        for box in boxes:
            bbox = box["box"]
            conf = box["conf"]
            label = box["label"].split(" ")[0]
            print(label)
    target = np.array([86,120,190])
    target = target.reshape(1, 1, 3)
    distanc = img_rgb - target
    distanc = distanc.astype(np.float32)
    sum_across_3rd_dim = np.sum(distanc ** 2, axis=2)
    index_of_smallest_sum = np.unravel_index(np.argmin(sum_across_3rd_dim), sum_across_3rd_dim.shape)
    elErr = index_of_smallest_sum[0] - 72
    azErr = index_of_smallest_sum[1] - 128

    
    # pitch = float(-elErr*gain)
    yaw_rate = float(-azErr*Pgain)
    
    yaw_rate = yaw_rate + yaw*Igain

    
    # yaw_rate = 0

    client.moveByRollPitchYawrateThrottleAsync(roll=roll, pitch=pitch, yaw_rate=yaw_rate, throttle=.6, duration=ts)
    
    yaw += yaw_rate*ts

    # client.moveByRollPitchYawrateZAsync(roll=roll, pitch=pitch, yaw_rate=yaw_rate, z=z, duration=ts)
    time.sleep(ts)

# while True:
#     client.moveByRollPitchYawrateThrottleAsync(roll=roll, pitch=pitch, yaw_rate=yaw_rate, z = 2, duration=ts)
#     time.sleep(ts)

# client.moveByRollPitchYawZAsync(roll=roll, pitch=pitch, yaw_rate=yaw_rate, throttle=throttle, duration=ts)

# # Example usage
# tracker = Tracker()
# tracker.update_target([650, 370])  # New target position
# tracker.track()  # Adjust roll and pitch to track the new target

# do something with the images
# for response in responses:
#     if response.pixels_as_float:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#         airsim.write_pfm(os.path.normpath('C:/Users/bill/Desktop/py1.pfm'), airsim.get_pfm_array(response))
#     else:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         airsim.write_file(os.path.normpath('C:/Users/bill/Desktop/py1.png'), response.image_data_uint8)


# class Tracker:
#     def __init__(self):
#         self.target_pixel = np.array([640, 360])  # Example target in pixel space (x, y)
#         self.current_position = np.array([320, 180])  # Starting position (x, y)
#         self.roll = 0.0
#         self.pitch = 0.0
#         self.yaw_rate = 0.0

#     def update_target(self, new_target):
#         """Update the target pixel position."""
#         self.target_pixel = np.array(new_target)

#     def track(self):
#         """Adjust roll and pitch to track the target pixel."""
#         error = self.target_pixel - self.current_position

#         # Simple proportional control
#         kp = 0.1  # Proportional gain. This needs to be tuned to your specific scenario

#         # Calculate roll and pitch adjustments
#         # Assuming positive roll implies moving right and positive pitch implies moving up
#         self.roll = kp * error[0]  # Error in x influences roll
#         self.pitch = kp * error[1]  # Error in y influences pitch

#         # Update current_position for demonstration purposes
#         # In a real scenario, this would be updated based on the vehicle's movement
#         self.current_position += error * kp

#         print(f"Roll: {self.roll}, Pitch: {self.pitch}")


        # fig = plt.imshow(img_rgb)
# plt.show()

# [86,120,190]

# original image is fliped vertically
# img_rgb = np.flipud(img_rgb)

# # write to png 
# airsim.write_png(os.path.normpath(filename + '.png'), img_rgb) 



# NPimage = np.fromstring(responses[0].image_data_uint8,dtype = np.uint8)

# temp = np.zeros([2500])
# temp[:2498] = NPimage

# # take images
# responses = client.simGetImages([
#     airsim.ImageRequest("0", airsim.ImageType.DepthVis),
#     airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
# print('Retrieved images: %d', len(responses))

# responses = client.simGetImages([
#     # png format
#     airsim.ImageRequest(0, airsim.ImageType.Scene), 
#     # uncompressed RGB array bytes
#     airsim.ImageRequest(1, airsim.ImageType.Scene, False, False),
#     # floating point uncompressed image
#     airsim.ImageRequest(1, airsim.ImageType.DepthPlanar, True)])

# import plotly.express as px
# import matplotlib.pyplot as plt
