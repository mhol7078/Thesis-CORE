import cv2
import numpy as np
import json


##----------------------------------------------------------------##
#
# Class to handle all intrinsic and extrinsic camera calibration
# operations.
#
##----------------------------------------------------------------##


class Calibration:
    def __init__(self, camRef, intrinFile=None, calibParams=None):
        self.camRef = camRef
        parameters = None
        if intrinFile is not None:
            parameters = json.load(intrinFile)
        elif self.params_not_none(calibParams):
            parameters = self.members_from_params(**calibParams)
        self.patternType = parameters[0]
        self.patternSize = parameters[1]
        self.patternDimensions = parameters[2]
        self.numImages = parameters[3]
        self.refineWindow = parameters[4]
        self.criteria = parameters[5]

    def params_not_none(self, params):
        flag = True
        for x in params:
            if x is None:
                flag = False
        return flag

    def members_from_params(self, patternType, patternSize,
                            patternDimensions, numImages, refineWindow,
                            numIter, epsIter):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, numIter, epsIter)
        return patternType, patternSize, patternDimensions, numImages, refineWindow, criteria

    # Function to perform intrinsic calibration of camera using square or circular pattern
    def calibrate_int(self, writeFlag=None):
        if self.patternType == 'Square':
            # Initialise 3D object grid for chessboard
            objectPoints = np.zeros((np.prod(self.patternSize), 3), np.float32)
            objectPoints[:, :2] = np.indices(self.patternSize).T.reshape(-1, 2)
            objectPoints *= self.patternDimensions[0]
            # Initialise point buffers
            objPArray = []
            imgPArray = []
            calibCount = 0
            self.camRef.get_frame()
            h, w = self.camRef.current_frame().shape[:2]
            cv2.namedWindow('Calibration Capture')
            # Capture calibration images
            while calibCount < self.numImages:
                self.camRef.get_frame()
                cv2.imshow('Calibration Capture', self.camRef.current_frame())
                userIn = cv2.waitKey(50)
                # 'c' to capture frame
                if userIn & 0xFF == ord('c'):
                    grayFrame = cv2.cvtColor(self.camRef.current_frame(), cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(grayFrame,
                                                             (self.patternSize[1], self.patternSize[0]),
                                                             None)
                    if ret:
                        corners2 = cv2.cornerSubPix(grayFrame,
                                                    corners,
                                                    (self.refineWindow, self.refineWindow),
                                                    (-1, -1),
                                                    self.criteria)
                        if corners2 is None:
                            corners2 = corners
                            cv2.putText(self.camRef.current_frame(),
                                        'Unable to refine corners',
                                        (10, grayFrame.shape[0] - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                        cv2.drawChessboardCorners(self.camRef.current_frame(),
                                                  (self.patternSize[1], self.patternSize[0]),
                                                  corners2,
                                                  True)
                        cv2.imshow('Calibration Capture', self.camRef.current_frame())
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
                elif userIn & 0xFF == ord(' '):
                    break
            # Run calibration
            if calibCount:
                ret, intrins, distCoefs, rotVecs, transVecs = cv2.calibrateCamera(objPArray, imgPArray, (w, h))
            cv2.destroyWindow('Calibration Capture')
        elif self.patternType == 'Circle':
            pass
        else:
            assert "Calibration pattern type not expected."
        if writeFlag is not None and writeFlag:
            parameters = [self.patternType, self.patternSize,
                          self.patternDimensions, self.numImages,
                          self.refineWindow, self.criteria]
            f = open('intrinsicparams.config', 'w')
            json.dump(parameters, f)
            f.close()
        return
