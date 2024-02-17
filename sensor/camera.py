import unittest
import airsim

class RealCamera:
    def __init__(self):
        --

class SimCamera:
    def __init__(self, client):
        self.client = client

    def get_frame(self):
        if self.client.isConnected():
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis),
                airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
            return responses


class DroneCamera:
    def __init__(self, client):
        self.client = client


if __name__ == '__main__':
    unittest.main()
