import cv2
import numpy as np
import time
import threading

__author__ = 'Michael Holmes'

##----------------------------------------------------------------##
#
# Class to handle all camera-related operations such as
# calibration, updating frames and responding to frame requests
#
##----------------------------------------------------------------##

# Global for mouse callback
target0 = (-1, -1)
camLock = threading.Lock()


def onMouse(event, x, y, flags, param):
    global target0
    if flags & cv2.EVENT_FLAG_LBUTTON:
        target0 = x, y
    return


class CamHandler(threading.Thread):
    def __init__(self, camID=None):
        threading.Thread.__init__(self)
        # If ID is given, check if valid and open camera
        self.frame = None
        self.lastTimestamp = 0  # Time in seconds since epoch (epoch and accuracy dependent on platform)
        self.deltaTime = 0  # Time in seconds between camera frames
        if camID is not None:
            self.camObj = cv2.VideoCapture(camID)
            if not self.camObj.isOpened():
                self.camObj = self._assign_cam()
        else:
            self.camObj = self._assign_cam()
        # Lock resolution at 480p for now if possible
        self.camObj.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
        self.camObj.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
        self._get_frame()
        self.updateFrame = False
        self.killThread = False

    def run(self):
        while not self.killThread:
            if self.updateFrame:
                camLock.acquire(1)
                self._get_frame()
                self.updateFrame = False
                camLock.release()
        return

    def stop(self):
        self.killThread = True
        self._release_cam()
        return

    def get_frame(self):
        camLock.acquire(1)
        self.updateFrame = True
        camLock.release()
        return

    def _get_frame(self):
        self.camObj.grab()
        newTimestamp = time.time()
        self.frame = self.camObj.retrieve()[1]
        self.deltaTime = newTimestamp - self.lastTimestamp
        self.lastTimestamp = newTimestamp
        return

    def current_frame(self):
        camLock.acquire(1)
        frame = self.frame
        camLock.release()
        return frame

    def _assign_cam(self):
        # Enumerate available cameras
        imgList = []
        currID = 0
        camRef = cv2.VideoCapture()
        while True:
            camRef.open(currID)
            if not camRef.isOpened():
                break
            imgList.append(camRef.read()[1])
            camRef.release()
            currID += 1
            # Limit preview to 6 cameras
            if currID == 6:
                break
        # If no cameras available raise exception
        if not len(imgList):
            print 'IOError: No cameras found.'
            raise IOError
        # Spawn camera images and choose appropriate camera
        camRef.open(self._choose_cam(imgList, 1024, 768))
        return camRef

    def _choose_cam(self, imgList, xPx, yPx):
        global target0
        # Tessellates up to six images into a single image and returns that
        # image of size xPx pixels (width) by yPx pixels (height)
        imX = xPx / len(imgList)
        outImg = np.zeros((yPx, xPx, 3), np.uint8)
        for idx, img in enumerate(imgList):
            img = cv2.resize(img, (imX, yPx))
            outImg[:, (idx * imX):((idx + 1) * imX), :] = img
        cv2.imshow("Camera Options - Click image to choose", outImg)
        cv2.setMouseCallback("Camera Options - Click image to choose", onMouse)
        while target0[0] == -1:
            cv2.waitKey(1000)
        cv2.destroyAllWindows()
        return target0[0] / imX

    def is_opened(self):
        return self.camObj.isOpened()

    def _release_cam(self):
        return self.camObj.release()

    def local_2D_to_3D(self, point, cameraMatrix, extrinsicMatrix):
        # Isolate intrinsic parameters
        fx = cameraMatrix[0, 0]
        fy = cameraMatrix[1, 1]
        cx = cameraMatrix[0, 2]
        cy = cameraMatrix[1, 2]
        # Construct direction vector
        lineVect = np.zeros((3, 1))
        lineVect[0] = (point[0] - cx) / fx
        lineVect[1] = (point[1] - cy) / fy
        lineVect[2] = 1
        lineVect = lineVect / np.linalg.norm(lineVect)
        # Rotate to global frame
        lineVect = np.dot(extrinsicMatrix[0:2, 0:2].T, lineVect)
        return
