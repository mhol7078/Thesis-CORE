import cv2
import numpy as np
import time

__author__ = 'Michael Holmes'

# Global for mouse callback
target0 = (-1, -1)


def onMouse(event, x, y, flags, param):
    global target0
    if flags & cv2.EVENT_FLAG_LBUTTON:
        target0 = x, y


class CamHandler:
    def __init__(self, camID=None):
        # If ID is given, check if valid and open camera
        self.frame = None
        self.lastTimestamp = 0  # Time in seconds since epoch (epoch and accuracy dependent on platform)
        self.deltaTime = 0  # Time in seconds between camera frames
        if camID is not None:
            self.camObj = cv2.VideoCapture(camID)
            if not self.camObj.isOpened():
                self.camObj = self.assign_cam()
        else:
            self.camObj = self.assign_cam()
        # Lock resolution at 480p for now if possible
        self.camObj.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
        self.camObj.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    def assign_cam(self):
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
        camRef.open(self.choose_cam(imgList, 1024, 768))
        return camRef

    def choose_cam(self, imgList, xPx, yPx):
        global target0
        # Tessellates up to six images into a single image and returns that image of size xPx pixels (width) by yPx pixels (height)
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

    def get_frame(self):
        self.camObj.grab()
        newTimestamp = time.time()
        self.frame = self.camObj.retrieve()[1]
        self.deltaTime = newTimestamp - self.lastTimestamp
        self.lastTimestamp = newTimestamp

    def is_opened(self):
        return self.camObj.isOpened()

    def release_cam(self):
        return self.camObj.release()

    # Function to perform intrinsic calibration of camera using square or circular pattern
    def calibrate_int(self, patternType, patternSize, patternDimensions, numImages, refineWindow, numIter, epsIter):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, numIter, epsIter)
        if patternType == 'Square':
            # Initialise 3D object grid for chessboard
            objectPoints = np.zeros((np.prod(patternSize), 3), np.float32)
            objectPoints[:, :2] = np.indices(patternSize).T.reshape(-1, 2)
            objectPoints *= patternDimensions[0]
            # Initialise point buffers
            objPArray = []
            imgPArray = []
            calibCount = 0
            self.get_frame()
            h, w = self.frame.shape[:2]
            cv2.namedWindow('Calibration Capture')
            # Capture calibration images
            while calibCount < numImages:
                self.get_frame()
                cv2.imshow('Calibration Capture', self.frame)
                userIn = cv2.waitKey(50)
                # 'c' to capture frame
                if userIn & 0xFF == ord('c'):
                    grayFrame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(grayFrame, (patternSize[1], patternSize[0]), None)
                    if ret:
                        corners2 = cv2.cornerSubPix(grayFrame, corners, (refineWindow, refineWindow), (-1, -1),
                                                    criteria)
                        if corners2 is None:
                            corners2 = corners
                            cv2.putText(self.frame, 'Unable to refine corners', (10, grayFrame.shape[0] - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                        cv2.drawChessboardCorners(self.frame, (patternSize[1], patternSize[0]), corners2, True)
                        cv2.imshow('Calibration Capture', self.frame)
                        userToggle = False
                        while not userToggle:
                            userIn = cv2.waitKey(50)
                            # 'y' to confirm good corner determination
                            if userIn & 0xFF == ord('y'):
                                userToggle = True
                                objPArray.append(objectPoints)
                                imgPArray.append(corners2.reshape(-1, 2))
                                calibCount += 1
                            # 'n' to abandon selected image
                            elif userIn & 0xFF == ord('n'):
                                userToggle = True
                    else:
                        cv2.putText(grayFrame, 'Unable to locate chessboard', (10, grayFrame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                        cv2.imshow('Calibration Capture', grayFrame)
                        cv2.waitKey(1000)
                elif userIn & 0xFF == ord('q'):
                    break
            # Run calibration
            ret, intrins, distCoefs, rotVecs, transVecs = cv2.calibrateCamera(objPArray, imgPArray, (w, h))
            return
        elif patternType == 'Circle':
            return
        else:
            assert ("Calibration pattern type not expected.")

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
