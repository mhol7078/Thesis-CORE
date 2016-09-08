import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D

__author__ = 'Michael Holmes'


# ----------------------------------------------------------------
#
# Handles all visualisation of reconstruction/local tracking in
# the main thread to avoid threading issues with window creation.
#
# ----------------------------------------------------------------

class Visualiser:
    def __init__(self):
        self.reconFig = plt.figure().gca(projection='3d')
        self.reconFig.set_xlabel('X Axis')
        self.reconFig.set_ylabel('Y Axis')
        self.reconFig.set_zlabel('Z Axis')
        self.reconFig.set_xlim3d(-2000, 2000)
        self.reconFig.set_ylim3d(-2000, 2000)
        self.reconFig.set_zlim3d(-2000, 2000)
        self.reconFig.view_init(azim=-90, elev=90)
        self.reconFig.legend()

        self.observFig = plt.figure().gca(projection='3d')
        self.observFig.set_xlabel('X Axis')
        self.observFig.set_ylabel('Y Axis')
        self.observFig.set_zlabel('Z Axis')
        self.observFig.set_xlim3d(-2000, 2000)
        self.observFig.set_ylim3d(-2000, 2000)
        self.observFig.set_zlim3d(-2000, 2000)
        self.observFig.view_init(azim=-90, elev=90)
        self.observFig.legend()
        return

    def render_reconstruction(self, posData):
        # Close reconstruction figure if open
        self.reconFig.close()
        # Separate position data into separate targets
        sepTargets = {}
        for targNum, position, capTime in posData:
            if targNum in sepTargets:
                sepTargets[targNum].append((position, capTime))
            else:
                sepTargets[targNum] = [(position, capTime)]

    def visualise_observation(self, segment1, segment2):
        # Close figure if open
        self.observFig.close()
        # Fix system global coords to upright coords
        rotMat = np.zeros((3, 3))
        rotMat[0, 0] = 1
        rotMat[1, 2] = 1
        rotMat[2, 1] = 1

        seg1Point1 = np.dot(rotMat, segment1[:, 0])
        seg1Point2 = np.dot(rotMat, segment1[:, 1])
        seg2Point1 = np.dot(rotMat, segment2[:, 0])
        seg2Point2 = np.dot(rotMat, segment2[:, 1])

        # plot the lines
        self.observFig.plot([seg1Point1[0], seg1Point2[0]], [seg1Point1[1], seg1Point2[1]],
                            [seg1Point1[2], seg1Point2[2]], label='line 1')
        self.observFig.plot([seg2Point1[0], seg2Point2[0]], [seg2Point1[1], seg2Point2[1]],
                            [seg2Point1[2], seg2Point2[2]], label='line 2')

        self.observFig.show()
