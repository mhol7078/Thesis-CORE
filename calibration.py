import cv2
import numpy as np
import json
import time
from scipy.spatial.distance import pdist, squareform


# Thesis Notes:
# Went for 4 grid 3d pattern because of correspondence problem
# Went for contour-based fiducial like in QR code because of keypoint false-positive problem
# Didn't embed characters inside circles because it caused problems at longer range
# Didn't use custom circle pattern per side because it would require scanning for each different pattern per frame

##----------------------------------------------------------------##
#
# Class to handle all intrinsic and extrinsic camera calibration
# operations.
#
##----------------------------------------------------------------##


class Calibration:
    def __init__(self, camRef=None, intrinFile=None, calibParams=None):
        self.camRef = camRef
        parameters = None
        if intrinFile is not None or calibParams is not None:
            if intrinFile is not None:
                parameters = json.load(intrinFile)
            elif self.params_not_none(calibParams):
                parameters = self.members_from_params(**calibParams)
            self.patternType = parameters[0]
            self.patternSize = parameters[1]
            self.patternDimension = parameters[2]
            self.numImages = parameters[3]
            self.refineWindow = parameters[4]
            self.criteria = parameters[5]
            self.camMatrix = parameters[6]
        else:
            self.patternType = None
            self.patternSize = None
            self.patternDimension = None
            self.numImages = None
            self.refineWindow = None
            self.criteria = None
            self.camMatrix = None
        self.hierarchy = None
        self.nestMin = 4
        self.calTime = 120.0
        self.capInterval = 2.0
        # offsets as a percentage of the length of the circular grid longest sides for each marker corner, clockwise
        # from top left
        self.markerOffsets = [-0.5, -0.4, 0.7, -0.4, 0.7, 0.4, -0.5, 0.4]
        return

    def params_not_none(self, params):
        flag = True
        for x in params:
            if x is None:
                flag = False
        return flag

    def members_from_params(self, patternType, patternSize,
                            patternDimension, numImages, refineWindow,
                            numIter, epsIter, camMatrix):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, numIter, epsIter)
        return patternType, patternSize, patternDimension, numImages, refineWindow, criteria, camMatrix

    # Function to perform intrinsic calibration of camera using square or circular pattern
    def calibrate_int(self, writeFlag=None):

        # Initialise 3D object grid
        objectPoints = self.gen_objp_grid()

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
            frameCopy = self.camRef.current_frame()
            cv2.imshow('Calibration Capture', frameCopy)
            userIn = cv2.waitKey(50)
            ret = None
            # 'c' to capture frame
            if userIn & 0xFF == ord('c'):
                greyFrame = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
                if self.patternType == 'Square':
                    ret, corners = cv2.findChessboardCorners(greyFrame,
                                                             (self.patternSize[1], self.patternSize[0]),
                                                             None)
                elif self.patternType == 'Circle':
                    ret, corners = cv2.findCirclesGrid(greyFrame,
                                                       (self.patternSize[1], self.patternSize[0]),
                                                       cv2.CALIB_CB_ASYMMETRIC_GRID)
                else:
                    assert "Calibration pattern type not expected."
                if ret:
                    cv2.cornerSubPix(greyFrame,
                                     corners,
                                     (self.refineWindow, self.refineWindow),
                                     (-1, -1),
                                     self.criteria)
                    cv2.drawChessboardCorners(self.camRef.current_frame(),
                                              (self.patternSize[1], self.patternSize[0]),
                                              corners,
                                              True)
                    cv2.imshow('Calibration Capture', frameCopy)
                    userToggle = False
                    while not userToggle:
                        userIn = cv2.waitKey(50)
                        # 'y' to confirm good corner determination
                        if userIn & 0xFF == ord('y'):
                            userToggle = True
                            objPArray.append(objectPoints)
                            imgPArray.append(corners.reshape(-1, 2))
                            calibCount += 1
                        # 'n' to abandon selected image
                        elif userIn & 0xFF == ord('n'):
                            userToggle = True
                else:
                    cv2.putText(greyFrame, 'Unable to locate chessboard', (10, greyFrame.shape[0] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.imshow('Calibration Capture', greyFrame)
                    cv2.waitKey(1000)
            elif userIn & 0xFF == ord(' '):
                break
        # Run calibration
        if calibCount:
            ret, self.camMatrix, distCoefs, rotVecs, transVecs = cv2.calibrateCamera(objPArray, imgPArray, (w, h))
        cv2.destroyWindow('Calibration Capture')

        if writeFlag is not None and writeFlag:
            parameters = [self.patternType, self.patternSize,
                          self.patternDimension, self.numImages,
                          self.refineWindow, self.criteria, self.camMatrix]
            f = open('intrinsicparams.config', 'w')
            json.dump(parameters, f)
            f.close()
        return

    def gen_objp_grid(self, patternType=None):
        if patternType is None:
            patternType = self.patternType
        objectPoints = np.zeros((np.prod(self.patternSize), 3), np.float32)
        if patternType == 'Square':
            # Square checkerboard grid
            objectPoints[:, :2] = np.indices(self.patternSize).T.reshape(-1, 2)
            objectPoints *= self.patternDimension
        elif patternType == 'Circle':
            # Asymmetric circle grid
            xvals = np.mgrid[0:self.patternSize[0], 0:self.patternSize[1]][0].T
            xvals *= self.patternDimension
            xvals[1::2, :] += self.patternDimension / 2
            xvals = xvals.flatten(order='C').reshape((-1, 1))
            yvals = np.mgrid[0:self.patternSize[0], 0:self.patternSize[1]][1].T
            yvals *= self.patternDimension / 2
            yvals = yvals.flatten(order='C').reshape((-1, 1))
            objectPoints[:, :1] = xvals
            objectPoints[:, 1:2] = yvals
        return objectPoints

    def calibrate_ext(self, calTime=None):
        # Initialise 3D object grid
        objectPoints = self.gen_objp_grid()
        # Initialise Calibration Timer
        if calTime is None:
            calTime = self.calTime
        startTime = time.time()
        currTime = startTime
        lastCheckTime = startTime - self.capInterval - 1
        # while currTime - startTime < calTime:
        while True:
            self.camRef.get_frame()
            frameCopy = self.camRef.current_frame()
            cv2.imshow('Extrinsic Calibration', frameCopy)
            userIn = cv2.waitKey(50)
            # if currTime - lastCheckTime > self.capInterval:
            if userIn & 0xFF == ord('c'):
                # Get new frame
                greyFrame = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
                # Search for circle grid
                ret, centers = cv2.findCirclesGrid(greyFrame,
                                                   (self.patternSize[0], self.patternSize[1]),
                                                   flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                if ret:
                    cv2.cornerSubPix(greyFrame,
                                     centers,
                                     (self.refineWindow, self.refineWindow),
                                     (-1, -1),
                                     self.criteria)
                    cv2.drawChessboardCorners(frameCopy,
                                              (self.patternSize[0], self.patternSize[1]),
                                              centers,
                                              True)
                    markers = self.find_markers(greyFrame, centers)
                    if markers is not None:
                        for markerNo, idx, center in markers:
                            cv2.circle(frameCopy, center, 5, (0, 255, 0))
                        if len(markers) == 2:
                            markerString = 'Found Markers: %d & %d' % (markers[0][0], markers[1][0])
                        else:
                            markerString = 'Found Marker: %d' % markers[0][0]
                        cv2.putText(frameCopy, markerString, (10, frameCopy.shape[0] - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.imshow('Extrinsic Calibration', frameCopy)
                    cv2.waitKey(2000)
                    lastCheckTime = time.time()
                else:
                    cv2.putText(frameCopy, 'Unable to locate calibration grid', (10, frameCopy.shape[0] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.imshow('Extrinsic Calibration', frameCopy)
                    cv2.waitKey(1000)
            currTime = time.time()

    def find_markers(self, image, imgPoints):
        # Set reference points
        boundPoints = np.zeros((4, 2), dtype="float32")
        boundPoints[0] = imgPoints[0].reshape((1, 2))
        boundPoints[1] = imgPoints[3].reshape((1, 2))
        boundPoints[2] = imgPoints[43].reshape((1, 2))
        boundPoints[3] = imgPoints[40].reshape((1, 2))
        # Elongate target area
        srcPoints = self.expand_bounds(boundPoints)
        # Find Perspective Transformation and its inverse for later unwarping
        srcDist = squareform(pdist(srcPoints))
        destPoints = np.array([0, 0, srcDist[0, 1], 0,
                               srcDist[0, 1], srcDist[1, 2],
                               0, srcDist[1, 2]], dtype='float32').reshape((-1, 2))
        warpMatrix = cv2.getPerspectiveTransform(srcPoints, destPoints)
        invWarpMatrix = cv2.getPerspectiveTransform(destPoints, srcPoints)
        # Isolate region of interest from image
        warpedImg = cv2.warpPerspective(image, warpMatrix, (destPoints[1, 0], destPoints[3, 1]))
        # Extract contours
        edged = cv2.Canny(warpedImg, 30, 200)
        contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.hierarchy = hierarchy[0]
        markers = []
        for idx, h in enumerate(self.hierarchy):
            if h[3] != -1:
                continue
            else:
                # Check if contour is quadrilateral
                peri = cv2.arcLength(contours[idx], True)
                approx = cv2.approxPolyDP(contours[idx], 0.02 * peri, True)
                if len(approx) == 4:
                    count = self.parse_contour(idx, 0) + 1
                    if count >= self.nestMin:
                        moments = cv2.moments(contours[idx])
                        if moments['m00'] == 0:
                            continue
                        roughCenter = np.array([moments['m10'] / moments['m00'],
                                                moments['m01'] / moments['m00'], 1]).reshape((-1, 1))
                        roughCenter = np.dot(invWarpMatrix, roughCenter)[:2]
                        # Correct center for perspective transform
                        markers.append((count - self.nestMin, idx, (roughCenter[0], roughCenter[1])))
        if len(markers):
            return markers
        else:
            return None

    def parse_contour(self, index, count):
        if self.hierarchy[index][2] != -1 and count < self.nestMin:
            count = self.parse_contour(self.hierarchy[index][2], count + 1)
        elif count >= self.nestMin and self.hierarchy[index][0] != -1:
            count = self.parse_contour(self.hierarchy[index][0], count + 1)
        return count

    # Expands bounds of circle grid to include markers
    def expand_bounds(self, boundPoints):
        # Find distance between corners for ratio
        boundDist = squareform(pdist(boundPoints, 'euclidean'))
        # Translate points wrt circle grid origin (Top left corner of circle grid)
        boundSub = boundPoints[0].copy()
        for idx in range(boundPoints.shape[0]):
            boundPoints[idx] -= boundSub
        # Correct rotation of bound points to Global Coords
        angleOffset = -(np.arctan2(boundPoints[1, 1], boundPoints[1, 0]) % (2 * np.pi))
        c, s = np.cos(angleOffset), np.sin(angleOffset)
        rotMatrix = np.array([[c, -s], [s, c]])
        boundPoints = np.dot(rotMatrix, boundPoints.T).T
        # Apply offsets of markers
        xDist = boundDist[0, 1]
        yDist = boundDist[0, 3]
        offsetsX = np.array([xDist * x for x in self.markerOffsets[0::2]]).reshape((-1, 1))
        offsetsY = np.array([yDist * y for y in self.markerOffsets[1::2]]).reshape((-1, 1))
        boundPoints += np.hstack([offsetsX, offsetsY])
        # Rotate back to image coordinates
        rotMatrix = np.array([[c, s], [-s, c]])
        boundPoints = np.dot(rotMatrix, boundPoints.T).T
        # Translate points wrt to image
        for idx in range(boundPoints.shape[0]):
            boundPoints[idx] += boundSub
        return boundPoints.astype('float32')


if __name__ == '__main__':
    myCal = Calibration()
    image = cv2.imread('test.png')
    greyFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, centers = cv2.findCirclesGrid(greyFrame,
                                       (4, 11),
                                       flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    markers = myCal.find_markers(greyFrame, centers)
    if markers is not None:
        for markerNo, idx, center in markers:
            cv2.circle(image, center, 5, (0, 255, 0))
        if len(markers) == 2:
            markerString = 'Found Markers: %d & %d' % (markers[0][0], markers[1][0])
        else:
            markerString = 'Found Marker: %d' % markers[0][0]
        cv2.putText(image, markerString, (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.drawChessboardCorners(image,
                              (4, 11),
                              centers,
                              True)
    cv2.imshow('Marker Test', image)
    while True:
        cv2.waitKey(1000)
