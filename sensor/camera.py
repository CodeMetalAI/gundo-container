import unittest
import airsim
import numpy as np
import cv2


class CameraGeneric:
    def __init__(self, client):
        self.client = client

    @staticmethod
    def target_az_el(obj_pixel_x,
                     obj_pixel_y,
                     intrinsic_params,
                     extrinsic_params):
        # Assuming intrinsic_params is a dictionary containing 'focal_length_x', 'focal_length_y', 'principal_point_x',
        # and 'principal_point_y' Assuming extrinsic_params is a dictionary containing 'rotation_vector' and
        # 'translation_vector'

        # Convert pixel coordinates to normalized camera coordinates
        px = (obj_pixel_x - intrinsic_params['principal_point_x']) / intrinsic_params['focal_length_x']
        py = (obj_pixel_y - intrinsic_params['principal_point_y']) / intrinsic_params['focal_length_y']

        rotation_matrix, _ = cv2.Rodrigues(np.array(extrinsic_params['rotation_vector']))
        translation_vector = np.array(extrinsic_params['translation_vector'])

        camera_coordinates = np.array([px, py, 1])
        world_coordinates = rotation_matrix @ camera_coordinates + translation_vector

        az = np.arctan2(world_coordinates[1], world_coordinates[0])
        el = np.arctan2(world_coordinates[2], np.sqrt(world_coordinates[0] ** 2 + world_coordinates[1] ** 2))

        az_deg = np.degrees(az)
        el_deg = np.degrees(el)

        return az_deg, el_deg


        # Example usage:
        # obj_pixel_x, obj_pixel_y = ... # Detected object coordinates
        # intrinsic_params = {'focal_length_x': ..., 'focal_length_y': ..., 'principal_point_x': ..., 'principal_point_y': ...}
        # extrinsic_params = {'rotation_vector': ..., 'translation_vector': ...}
        # az, el = object_az_el(frame, obj_pixel_x, obj_pixel_y, intrinsic_params, extrinsic_params)
        # print(f"Azimuth: {az}, Elevation: {el}")

class Camera(CameraGeneric):
    def get_frame(self):
        # Whatever drone API or external web cameras
        return


class SimCamera(CameraGeneric):
    def get_frame(self):
        # if self.client.isConnected():
            return self.client.simGetImage("0", airsim.ImageType.Scene)


if __name__ == '__main__':
    unittest.main()
