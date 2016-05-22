import cv2
import numpy as np
import json
import time
from scipy import spatial

# Thesis Notes:
# Went for 4 grid 3d pattern because of correspondence problem
# Went for contour-based fiducial like in QR code because of key-point false-positive problem
# Didn't embed characters inside circles because it caused problems at longer range
# Didn't use custom circle pattern per side because it would require scanning for each different pattern per frame
# Had issue identifying 2 separate grids in single frame, even if not both symmetric/asymmetric
# Dumped circle grids altogether due to clashes with square grids/other circle grids

##----------------------------------------------------------------##
#
# Class to handle all intrinsic and extrinsic camera calibration
# operations.
#
##----------------------------------------------------------------##

# GLOBAL to drive mouse events
target0 = None


# Mouse event for local mode markup functions
def onMouse(event, x, y, flags, param):
    global target0
    if event == cv2.EVENT_LBUTTONDOWN:
        target0 = (x, y)


class Calibration:
    def __init__(self, camRef=None, calibFilename=None, calibParams=None, targetFilename=None):
        self._camRef = camRef
        self._intPatternType = None
        self._extPatternType = None
        self._intPatternSize = None
        self._extPatternSize = None
        self._intPatternDimension = None
        self._extPatternDimension = None
        self._numImages = None
        self._refineWindow = None
        self._numIter = None
        self._epsIter = None
        self._criteria = None
        self.camMatrix = None
        self.distCoefs = None
        self._hierarchy = None
        self._nestMin = None
        self._calTime = None
        self._capInterval = None
        # offsets list as a percentage of the length of the circular grid longest sides for each marker corner,
        # clockwise from top left
        self._markerOffsets = None
        self._gridOffset = None
        # Relating markers to sides of the 3d rig
        self._markerSides = None
        if calibFilename is not None:
            calibParams = self.read_calib_from_file(calibFilename)
            if self._params_not_none(calibParams):
                self._members_from_params(**calibParams)
                print 'Loaded intrinsic parameters from %s.' % calibFilename
        elif calibParams is not None and self._params_not_none(calibParams):
            self._members_from_params(**calibParams)
            print 'Loaded intrinsic parameters from initialisation script - intrinsic calibration required.'
        else:
            assert 'Failed to load intrinsic parameters.'
        if targetFilename is not None:
            self.target = self._load_target(targetFilename)
        return

    def _params_not_none(self, params):
        flag = True
        for x in params.values():
            if x is None:
                flag = False
        return flag

    def _members_from_params(self, intPatternType, extPatternType, intPatternSize, extPatternSize,
                             intPatternDimension, extPatternDimension, numImages, refineWindow,
                             numIter, epsIter, camMatrix, distCoefs, nestMin, calTime, capInterval,
                             markerOffsets, gridOffset, markerSides):
        self._intPatternType = intPatternType
        self._extPatternType = extPatternType
        self._intPatternSize = intPatternSize
        self._extPatternSize = extPatternSize
        self._intPatternDimension = intPatternDimension
        self._extPatternDimension = extPatternDimension
        self._numImages = numImages
        self._refineWindow = refineWindow
        self._numIter = numIter
        self._epsIter = epsIter
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, numIter, epsIter)
        self.camMatrix = camMatrix
        self.distCoefs = distCoefs
        self._nestMin = nestMin
        self._calTime = calTime
        self._capInterval = capInterval
        self._markerOffsets = markerOffsets
        self._gridOffset = gridOffset
        self._markerSides = markerSides
        return

    def write_calib_to_file(self, filename):
        camMatrixList = []
        distCoefsList = []
        for x in self.camMatrix.flat:
            camMatrixList.append(x)
        for x in self.distCoefs.flat:
            distCoefsList.append(x)
        calibParams = dict(intPatternType=self._intPatternType,
                           extPatternType=self._extPatternType,
                           intPatternSize=self._intPatternSize,
                           extPatternSize=self._extPatternSize,
                           intPatternDimension=self._intPatternDimension,
                           extPatternDimension=self._extPatternDimension,
                           numImages=self._numImages,
                           refineWindow=self._refineWindow,
                           numIter=self._numIter,
                           epsIter=self._epsIter,
                           camMatrix=camMatrixList,
                           distCoefs=distCoefsList,
                           nestMin=self._nestMin,
                           calTime=self._calTime,
                           capInterval=self._capInterval,
                           markerOffsets=self._markerOffsets,
                           gridOffset=self._gridOffset,
                           markerSides=self._markerSides)
        fd = open(filename, 'w')
        json.dump(calibParams, fd)
        fd.close()
        print 'Successfully written intrinsic calibration to file.'
        return

    def read_calib_from_file(self, filename):
        # Load params from .cfg
        fd = open(filename, 'r')
        calibParams = json.load(fd)
        fd.close()
        # rectify intrinsic matrix to numpy array
        camMatrixList = calibParams['camMatrix']
        camMatrix = np.float64(camMatrixList).reshape((3, 3))
        distCoefsList = calibParams['distCoefs']
        distCoefs = np.float64(distCoefsList).reshape((5, 1))
        calibParams['distCoefs'] = distCoefs
        calibParams['camMatrix'] = camMatrix
        # rectify lists to tuples
        calibParams['intPatternSize'] = tuple(calibParams['intPatternSize'])
        calibParams['extPatternSize'] = tuple(calibParams['extPatternSize'])
        calibParams['refineWindow'] = tuple(calibParams['refineWindow'])
        return calibParams

    def _gen_objp_grid(self, patternType=None, patternSize=None, patternDimension=None):
        if patternType is None:
            patternType = self._intPatternType
            patternSize = self._intPatternSize
            patternDimension = self._intPatternDimension
        objectPoints = np.zeros((np.prod(patternSize), 3), np.float32)
        if patternType == 'Square':
            # Square checkerboard grid
            objectPoints[:, :2] = np.indices(patternSize).T.reshape(-1, 2)
            objectPoints *= patternDimension
        elif patternType == 'Circle':
            # Asymmetric circle grid
            xvals = np.mgrid[0:patternSize[0], 0:patternSize[1]][0].T
            xvals *= patternDimension
            xvals[1::2, :] += patternDimension / 2
            xvals = xvals.flatten(order='C').reshape((-1, 1))
            yvals = np.mgrid[0:patternSize[0], 0:patternSize[1]][1].T
            yvals *= patternDimension / 2
            yvals = yvals.flatten(order='C').reshape((-1, 1))
            objectPoints[:, :1] = xvals
            objectPoints[:, 1:2] = yvals
        return objectPoints

    # Function to perform intrinsic calibration of camera using square or circular pattern
    def calibrate_int(self):
        # Initialise 3D object grid
        objectPoints = self._gen_objp_grid(self._intPatternType, self._intPatternSize, self._intPatternDimension)
        # Initialise point buffers
        objPArray = []
        imgPArray = []
        calibCount = 0
        self._camRef.get_frame()
        h, w = self._camRef.current_frame().shape[:2]
        cv2.namedWindow('Calibration Capture')
        # Capture calibration images
        while calibCount < self._numImages:
            self._camRef.get_frame()
            frameCopy = self._camRef.current_frame()
            cv2.imshow('Calibration Capture', frameCopy)
            userIn = cv2.waitKey(50)
            ret = None
            # 'c' to capture frame
            if userIn & 0xFF == ord('c'):
                greyFrame = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
                if self._intPatternType == 'Square':
                    ret, corners = cv2.findChessboardCorners(greyFrame, self._intPatternSize, None)
                else:
                    ret, corners = cv2.findCirclesGrid(greyFrame, self._intPatternSize,
                                                       flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                if ret:
                    cv2.cornerSubPix(greyFrame, corners, self._refineWindow, (-1, -1), self._criteria)
                    cv2.drawChessboardCorners(frameCopy, self._intPatternSize, corners, True)
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
                            cv2.putText(frameCopy, 'Image Count: %d of %d accepted' % (calibCount, self._numImages),
                                        (10, frameCopy.shape[0] - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            cv2.imshow('Calibration Capture', frameCopy)
                            print 'Image Count: %d of %d accepted.' % (calibCount, self._numImages)
                            cv2.waitKey(1000)
                        # 'n' to abandon selected image
                        elif userIn & 0xFF == ord('n'):
                            userToggle = True
                            cv2.putText(frameCopy, 'Image discarded',
                                        (10, frameCopy.shape[0] - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            cv2.imshow('Calibration Capture', frameCopy)
                            print 'Image discarded.'
                            cv2.waitKey(1000)
                else:
                    cv2.putText(greyFrame, 'Unable to locate grid', (10, greyFrame.shape[0] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    print 'Unable to locate grid.'
                    cv2.imshow('Calibration Capture', greyFrame)
                    cv2.waitKey(1000)
            elif userIn & 0xFF == ord(' '):
                break
        # Run calibration
        if calibCount:
            cv2.putText(greyFrame, 'Calibrating intrinsics...', (10, greyFrame.shape[0] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            print 'Calibrating intrinsics...'
            cv2.imshow('Calibration Capture', greyFrame)
            cv2.waitKey(1000)
            rotVecs = None
            transVecs = None
            ret, self.camMatrix, self.distCoefs, rotVecs, transVecs = cv2.calibrateCamera(objPArray, imgPArray, (w, h),
                                                                                          self.camMatrix,
                                                                                          self.distCoefs, rotVecs,
                                                                                          transVecs)
        cv2.destroyWindow('Calibration Capture')
        return

    # Function to gather circle grid calibration data for a nominated time calTime then send data to the Master Node
    def calibrate_ext(self, debug=False):
        foundTargets = []
        # Initialise Timer
        startTime = time.time()
        currTime = startTime
        lastCheckTime = startTime - self._capInterval - 1
        # while currTime - startTime < self.calTime:
        while True:
            self._camRef.get_frame()
            frameCopy = self._camRef.current_frame()
            currTime = time.time()
            if debug:
                cv2.imshow('Extrinsic Calibration', frameCopy)
            # if currTime - lastCheckTime > self.capInterval:
            userIn = cv2.waitKey(50)
            if userIn & 0xFF == ord('c'):
                # Get new frame
                greyFrame = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
                # Search for square grid
                ret, corners = cv2.findChessboardCorners(greyFrame, self._extPatternSize, None)
                if ret:
                    cv2.cornerSubPix(greyFrame, corners, self._refineWindow, (-1, -1), self._criteria)
                    side = self._find_side(greyFrame, corners)
                    if side[0] is not None:
                        foundTargets.append((side, corners, currTime))
                        if debug:
                            cv2.drawChessboardCorners(frameCopy, self._extPatternSize, corners, True)
                            sideString = 'Found Side: %s: INV %d' % (side[0], side[1])
                            cv2.putText(frameCopy, sideString, (10, frameCopy.shape[0] - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            cv2.imshow('Extrinsic Calibration', frameCopy)
                            cv2.waitKey(100)
                            lastCheckTime = time.time()
                elif debug:
                    cv2.putText(frameCopy, 'Unable to locate calibration grid', (10, frameCopy.shape[0] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.imshow('Extrinsic Calibration', frameCopy)
                    cv2.waitKey(100)
        if debug:
            cv2.destroyWindow('Extrinsic Calibration')
        return foundTargets if len(foundTargets) else None

    def calibrate_target(self, filename, debug=False):
        targetSides = []
        capImages = self._numImages
        while True and capImages:
            self._camRef.get_frame()
            frameCopy = self._camRef.current_frame()
            if debug:
                cv2.imshow('Target Calibration', frameCopy)
            userIn = cv2.waitKey(50)
            if userIn & 0xFF == ord('c'):
                # Get new frame
                greyFrame = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY)
                # Search for checkerboard grid
                ret1, corners1 = cv2.findChessboardCorners(greyFrame, self._extPatternSize, None)
                if ret1:
                    cv2.cornerSubPix(greyFrame, corners1, self._refineWindow, (-1, -1), self._criteria)
                    side1 = self._find_side(greyFrame, corners1)
                    if side1 is None:
                        if debug:
                            cv2.putText(frameCopy, 'Unable to locate side 1 markers', (10, frameCopy.shape[0] - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            cv2.imshow('Target Calibration', frameCopy)
                            cv2.waitKey(1000)
                        continue
                    gridBounds = np.zeros((4, 2))
                    gridBounds[0] = corners1[-1]
                    gridBounds[1] = corners1[np.prod(self._extPatternSize) - self._extPatternSize[0]]
                    gridBounds[2] = corners1[0]
                    gridBounds[3] = corners1[self._extPatternSize[0] - 1]
                    # Scrub first board from image
                    cv2.fillConvexPoly(greyFrame, gridBounds.astype(np.int32), (255, 255, 255))
                    # Search for second checkerboard grid
                    ret2, corners2 = cv2.findChessboardCorners(greyFrame, self._extPatternSize, None)
                else:
                    if debug:
                        cv2.putText(frameCopy, 'Unable to locate any sides', (10, frameCopy.shape[0] - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                        cv2.imshow('Target Calibration', frameCopy)
                        cv2.waitKey(1000)
                    continue
                if ret1 and not ret2 and corners2 is not None:
                    # Complete partial second side
                    ret2, corners2 = self._complete_grid(frameCopy.copy(), corners2)
                if ret1 and ret2:
                    cv2.cornerSubPix(greyFrame, corners2, self._refineWindow, (-1, -1), self._criteria)
                    side2 = self._find_side(greyFrame, corners2)
                    if self._parse_sides(side1, side2):
                        targetSides.append((side1, corners1, side2, corners2))
                        capImages -= 1
                        if debug:
                            cv2.drawChessboardCorners(frameCopy, self._extPatternSize, corners1, True)
                            cv2.drawChessboardCorners(frameCopy, self._extPatternSize, corners2, True)
                            sideString = 'Found Sides: %s:INV %d   %s:INV %d' % (side1[0], side1[1], side2[0], side2[1])
                            cv2.putText(frameCopy, sideString, (10, frameCopy.shape[0] - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            cv2.imshow('Target Calibration', frameCopy)
                            cv2.waitKey(500)
                            lastCheckTime = time.time()
                    elif debug:
                        cv2.putText(frameCopy, 'Unable to locate side 2 markers', (10, frameCopy.shape[0] - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                        cv2.imshow('Target Calibration', frameCopy)
                        cv2.waitKey(1000)
                elif debug:
                    cv2.putText(frameCopy, 'Unable to locate side 2', (10, frameCopy.shape[0] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.imshow('Target Calibration', frameCopy)
                    cv2.waitKey(1000)
            elif userIn & 0xFF == ord(' '):
                break
        # Calibrate Sides wrt North face
        calibTarget = self._parse_pnp_pairs(targetSides)
        if calibTarget is not None:
            self._save_target(filename, calibTarget)
        if debug:
            cv2.destroyWindow('Target Calibration')
        return

    def _save_target(self, filename, target):
        fd = open(filename, 'w')
        targetCopy = target.copy()
        for key in targetCopy.keys():
            targetCopy[key] = list(targetCopy[key].flatten(1))
        json.dump(targetCopy, fd)
        fd.close()
        return

    def _load_target(self, filename):
        fd = open(filename)
        target = json.load(fd)
        for key in target.keys():
            target[key] = np.array(target[key], dtype=np.float32).reshape((4, 4))
        fd.close()
        return target

    def _complete_grid(self, image, foundPoints):
        imgCopy = image.copy()
        candidates = foundPoints.copy().reshape((-1, 2))
        cv2.drawChessboardCorners(imgCopy, self._extPatternSize, candidates, False)
        cv2.putText(imgCopy, 'Complete grid manually? y/n', (10, imgCopy.shape[0] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow('Incomplete Grid', imgCopy)
        while True:
            userIn = cv2.waitKey(1000)
            if userIn & 0xFF == ord('y'):
                break
            elif userIn & 0xFF == ord('n'):
                return None
        cv2.destroyWindow('Incomplete Grid')
        # Resize ROI
        boundRect = cv2.boundingRect(foundPoints)
        boundPoints = np.array([boundRect[0], boundRect[1],
                                boundRect[0] + boundRect[2], boundRect[1],
                                boundRect[0] + boundRect[2], boundRect[1] + boundRect[3],
                                boundRect[0], boundRect[1] + boundRect[3]], dtype=np.float32).reshape((-1, 2))
        boundPoints = self._expand_bounds(boundPoints)
        boundRect = cv2.boundingRect(boundPoints)
        roiImg = image[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]]
        scaleParams = (boundRect[0], boundRect[1], 300.0 / roiImg.shape[1], 300.0 / roiImg.shape[0])
        roiImg = cv2.resize(roiImg, (300, 300))
        adjCandidates = self._transform_candidates(scaleParams, candidates)
        # Try rectify image and rescan with drawChessboardCorners
        rectifiedCandidates = self._rectify_refine_corners(roiImg)
        if rectifiedCandidates is not None:
            # Invert Transform points and return
            rectifiedCandidates = self._transform_candidates(scaleParams, rectifiedCandidates, True)
            return True, rectifiedCandidates.reshape((-1, 1, 2)).astype(np.float32)
        # Alternative manual method
        # Delete incorrect markers
        adjCandidates = self._manual_candidates(roiImg, adjCandidates, 'del')
        # Add correct markers
        adjCandidates = self._manual_candidates(roiImg, adjCandidates, 'add')
        ret, adjCandidates = cv2.findChessboardCorners(roiImg, self._extPatternSize, adjCandidates)
        if ret:
            adjCandidates = self._transform_candidates(scaleParams, adjCandidates, True)
            return True, adjCandidates.reshape((-1, 1, 2)).astype(np.float32)
        else:
            return False, None

    def _rectify_refine_corners(self, roiImg):
        global target0
        cornerLabels = ('top left', 'top right', 'bottom right', 'bottom left')
        corners = []
        for corner in cornerLabels:
            titleStr = 'Click on the %s corner, y/n to confirm, space to cancel' % corner
            cv2.namedWindow(titleStr)
            cv2.setMouseCallback(titleStr, onMouse)
            reDraw = True
            continueFlag = False
            while True:
                if reDraw:
                    cv2.imshow(titleStr, roiImg)
                    reDraw = False
                if target0 is not None:
                    imgCopy = roiImg.copy()
                    cv2.circle(imgCopy, (target0[0], target0[1]), 5, (255, 0, 0), -1)
                    cv2.imshow(titleStr, imgCopy)
                    while not reDraw:
                        userIn = cv2.waitKey(100)
                        if userIn & 0xFF == ord('y'):
                            corners.append([target0[0], target0[1]])
                            continueFlag = True
                            target0 = None
                            break
                        elif userIn & 0xFF == ord('n'):
                            reDraw = True
                            target0 = None
                if continueFlag:
                    cv2.destroyWindow(titleStr)
                    break
                if cv2.waitKey(100) & 0xFF == ord(' '):
                    cv2.destroyWindow(titleStr)
                    return None
        corners = np.array(corners, dtype=np.float32).reshape((-1, 2))
        objCandidates = self._gen_objp_grid(self._extPatternType, self._extPatternSize, self._extPatternDimension)[:,
                        :2]
        objWarps = np.zeros((4, 2), dtype=np.float32)
        objWarps[0] = objCandidates[0]
        objWarps[1] = objCandidates[self._extPatternSize[0] - 1]
        objWarps[2] = objCandidates[-1]
        objWarps[3] = objCandidates[np.prod(self._extPatternSize) - self._extPatternSize[0]]
        # Offset object points to center of image
        warpDist = spatial.distance.squareform(spatial.distance.pdist(corners))
        warpOffset = np.zeros((1, 2), dtype=np.float32)
        warpOffset[0, 0] = (roiImg.shape[1] - warpDist[0, 1]) / 2
        warpOffset[0, 1] = (roiImg.shape[0] - warpDist[0, 3]) / 2
        for idx in range(objWarps.shape[0]):
            objWarps[idx, 0] += warpOffset[0, 0]
            objWarps[idx, 1] += warpOffset[0, 1]
        # Find Perspective Transformation
        warpMatrix = cv2.getPerspectiveTransform(corners, objWarps)
        invWarpMatrix = cv2.getPerspectiveTransform(objWarps, corners)
        warpedImg = cv2.warpPerspective(roiImg, warpMatrix, (roiImg.shape[1], roiImg.shape[0]))
        ret, cornersFixed = cv2.findChessboardCorners(warpedImg, self._extPatternSize)
        if ret:
            cornersFixed = np.hstack((cornersFixed.reshape((-1, 2)), np.ones((cornersFixed.shape[0], 1)))).T
            cornersFixed = np.dot(invWarpMatrix, cornersFixed)
            cornersFixed = cornersFixed[:2, :] / np.tile(cornersFixed[2, :], (2, 1))
            return cornersFixed.T
        else:
            return None

    def _manual_candidates(self, roiImg, adjCandidates, operation):
        global target0
        target0 = None
        if operation == 'add':
            opFlag = True
        elif operation == 'del':
            opFlag = False
        else:
            assert 'Incorrect operation type, not add or del'
        titleStr = 'Click to %s, y/n to confirm, space to finish' % operation
        if not opFlag:
            kdTree = spatial.cKDTree(adjCandidates)
        cv2.imshow(titleStr, roiImg)
        cv2.setMouseCallback(titleStr, onMouse)
        reDraw = True
        while True:
            if reDraw:
                # Reset the delete prompt window
                imgCopy = roiImg.copy()
                drawnCorners = adjCandidates.copy().reshape((-1, 1, 2)).astype(np.float32)
                cv2.drawChessboardCorners(imgCopy, self._extPatternSize, drawnCorners, False)
                cv2.imshow(titleStr, imgCopy)
                reDraw = False
            if target0 is not None:
                # Highlight closest point to click, highlight and prompt for removal
                if opFlag:
                    cv2.circle(imgCopy, (target0[0], target0[1]), 5, (255, 0, 0), -1)
                else:
                    dist, idx = kdTree.query(np.array(target0).reshape((1, 2)))
                    cv2.circle(imgCopy, (adjCandidates[idx, 0], adjCandidates[idx, 1]), 5, (255, 0, 0), -1)
                cv2.imshow(titleStr, imgCopy)
                while not reDraw:
                    userIn = cv2.waitKey(100)
                    if userIn & 0xFF == ord('y'):
                        if opFlag:
                            # Add point to candidates
                            adjCandidates = np.vstack((adjCandidates, np.array(target0).reshape((1, 2))))
                        else:
                            # Remove point from candidates
                            adjCandidates = np.delete(adjCandidates, idx, axis=0)
                            kdTree = spatial.cKDTree(adjCandidates)
                        reDraw = True
                        target0 = None
                    elif userIn & 0xFF == ord('n'):
                        reDraw = True
                        target0 = None
            if cv2.waitKey(100) & 0xFF == ord(' '):
                cv2.destroyWindow(titleStr)
                break
        return adjCandidates

    def _transform_candidates(self, scaleParams, points, invert=False):
        returnPoints = points.copy()
        translator = np.tile(np.array([scaleParams[0], scaleParams[1]]).reshape((1, 2)),
                             (returnPoints.shape[0], 1))
        if not invert:
            scaler = np.tile(np.array([scaleParams[2], scaleParams[3]]).reshape((1, 2)),
                             (returnPoints.shape[0], 1))
            returnPoints -= translator
            returnPoints *= scaler
        else:
            scaler = np.tile(np.array([1.0 / scaleParams[2], 1.0 / scaleParams[3]]).reshape((1, 2)),
                             (returnPoints.shape[0], 1))
            returnPoints *= scaler
            returnPoints += translator
        return returnPoints

    def _check_target_sides(self, targets):
        # Return true if every side of the rig is captured
        sideFlags = dict(N=False, S=False, E=False, W=False)
        for side1, corners1, side2, corners2 in targets:
            sideFlags[side1[0]] = True
            sideFlags[side2[0]] = True
        allSides = True
        for x in sideFlags.values():
            if not x:
                allSides = False
        return allSides

    def _parse_pnp_pairs(self, targets):
        # Make sure all sides have been captured
        if not self._check_target_sides(targets):
            return False
        pairsDict = {}
        # Computer homologous transformation matrix for each pair and amalgamate
        objPoints = self._gen_objp_grid(self._extPatternType, self._extPatternSize, self._extPatternDimension)
        for side1, center1, side2, center2 in targets:
            # Calc position wrt to camera
            ret1, rot1, trans1 = cv2.solvePnP(objPoints, center1, self.camMatrix, self.distCoefs)
            ret2, rot2, trans2 = cv2.solvePnP(objPoints, center2, self.camMatrix, self.distCoefs)
            if ret1 and ret2:
                # Calc position of 2 wrt 1
                rot1 = cv2.Rodrigues(rot1)[0]
                rot2 = cv2.Rodrigues(rot2)[0]
                rotComp = np.dot(rot1.T, rot2)
                transComp = -trans1 + np.dot(rot1.T, trans2)
                transform2wrt1 = np.hstack((np.vstack((rotComp, np.zeros((1, 3)))), np.vstack((transComp, 1))))
                keyStr = side1[0] + side2[0]
                if keyStr not in pairsDict and keyStr[::-1] not in pairsDict:
                    # Add new transformation matrix to dictionary
                    pairsDict[keyStr] = (transform2wrt1, 1)
                elif keyStr in pairsDict:
                    # Add new matrix to previous sum to average out later
                    pairsDict[keyStr] = (pairsDict[keyStr][0] + transform2wrt1, pairsDict[keyStr][1] + 1)
                elif keyStr[::-1] in pairsDict:
                    # Reverse transformation matrix and add to previous sum
                    transform1wrt2 = self._invert_transform_matrix(transform2wrt1)
                    pairsDict[keyStr[::-1]] = (
                    pairsDict[keyStr[::-1]][0] + transform1wrt2, pairsDict[keyStr[::-1]][1] + 1)
        # Compute averages
        if len(pairsDict.keys()) == 4:
            for key, value in pairsDict.iteritems():
                pairsDict[key] = value[0] / float(value[1])
            # Generate output dictionary of transformation matrices wrt N
            transformDict = {'N': np.eye(4)}
            if 'NE' in pairsDict:
                transformDict['E'] = pairsDict['NE']
            else:
                transformDict['E'] = self._invert_transform_matrix(pairsDict['EN'])
            if 'NW' in pairsDict:
                transformDict['W'] = pairsDict['NW']
            else:
                transformDict['W'] = self._invert_transform_matrix(pairsDict['WN'])
            if 'SE' in pairsDict:
                transformDict['S'] = np.dot(transformDict['E'], self._invert_transform_matrix(pairsDict['SE']))
            else:
                transformDict['S'] = np.dot(transformDict['E'], pairsDict['ES'])
            return transformDict
        else:
            assert 'Failed target calibration: Could not generate PnP pairs.'
            return None

    def _invert_transform_matrix(self, matrix):
        rotInv = matrix[0:3, 0:3].reshape((3, 3)).T
        transInv = np.dot(-rotInv, matrix[0:3, 3].reshape((3, 1)))
        return np.hstack((np.vstack((rotInv, np.zeros((1, 3)))), np.vstack((transInv, 1))))

    def _parse_sides(self, side1, side2):
        # Check orientations match
        if side2 is None or side1[1] != side2[1]:
            return False
        # Check sides are neighbouring
        check1 = True if side1[0] == 'N' or side1[0] == 'S' else False
        check2 = True if side2[0] == 'N' or side2[0] == 'S' else False
        return True if check1 != check2 else False

    # detects sides in image given detected grid as a ROI
    def _find_side(self, image, imgPoints):
        objCandidates = self._gen_objp_grid(self._extPatternType, self._extPatternSize, self._extPatternDimension)[:,
                        :2]
        imgCandidates = imgPoints.reshape((-1, 2))
        objWarps = np.zeros((4, 2), dtype=np.float32)
        imgWarps = objWarps.copy()
        objWarps[0] = objCandidates[0]
        objWarps[1] = objCandidates[self._extPatternSize[0] - 1]
        objWarps[2] = objCandidates[-1]
        objWarps[3] = objCandidates[np.prod(self._extPatternSize) - self._extPatternSize[0]]
        # changed order because of stupid corner indexing update in OpenCV 3.1
        imgWarps[0] = imgCandidates[-1]
        imgWarps[1] = imgCandidates[np.prod(self._extPatternSize) - self._extPatternSize[0]]
        imgWarps[2] = imgCandidates[0]
        imgWarps[3] = imgCandidates[self._extPatternSize[0] - 1]
        # Offset object points to center of image
        warpDist = spatial.distance.squareform(spatial.distance.pdist(imgWarps))
        warpOffset = np.zeros((1, 2), dtype=np.float32)
        warpOffset[0, 0] = (image.shape[1] - warpDist[0, 1]) / 2
        warpOffset[0, 1] = (image.shape[0] - warpDist[0, 3]) / 2
        for idx in range(objWarps.shape[0]):
            objWarps[idx, 0] += warpOffset[0, 0]
            objWarps[idx, 1] += warpOffset[0, 1]
        # Find Perspective Transformation
        warpMatrix = cv2.getPerspectiveTransform(imgWarps, objWarps)
        # Elongate ROI to include markers
        boundPoints = self._expand_bounds(objWarps)
        boundRect = cv2.boundingRect(boundPoints)
        # Warp image
        warpedImg = cv2.warpPerspective(image, warpMatrix, (image.shape[1], image.shape[0]))
        # Isolate ROI
        roiImg = warpedImg[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]]
        # cv2.imshow('ROI', roiImg)
        # Determine side using contour method
        side, sideInverted = self._side_from_contours(roiImg)
        if side is not None:
            return side, sideInverted
        else:
            # If contours inconclusive determine side based on slower blobs method
            roiGridBounds = objWarps - np.tile(np.array(boundRect[0:2]).reshape((1, 2)), (4, 1))
            side, sideInverted = self._side_from_blobs(roiImg, roiGridBounds)
        if side is not None:
            return side, sideInverted
        else:
            return None

    # Parse based on contours
    def _side_from_contours(self, image):
        edged = cv2.Canny(image, 30, 200)
        _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                    count = self._parse_contour(idx, 0) + 1
                    if count >= self._nestMin:
                        moments = cv2.moments(contours[idx])
                        if moments['m00'] == 0:
                            continue
                        roughCenter = np.array([moments['m10'] / moments['m00'],
                                                moments['m01'] / moments['m00'], 1]).reshape((-1, 1))
                        # Correct center for perspective transform
                        markers.append((count - self._nestMin, roughCenter.reshape((-1, 1))))
        # Determine side from markers
        gridCenter = (image.shape[1] // 2, image.shape[0] // 2)
        return self._side_from_markers(markers, gridCenter)

    def _side_from_markers(self, markers, gridCenter):
        if len(markers) == 2:
            sideM1, inverted1 = self._marker_to_side(markers[0], gridCenter)
            sideM2, inverted2 = self._marker_to_side(markers[1], gridCenter)
            # prevent conflict in side determination
            if sideM1 is not None and sideM2 is not None and sideM1 == sideM2:
                return sideM1, inverted1
        return None, None

    def _side_from_blobs(self, image, gridBounds):
        markers = []
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 100
        detector = cv2.SimpleBlobDetector_create(params)
        keyPoints = detector.detect(image)
        boundDist = spatial.distance.squareform(spatial.distance.pdist(gridBounds, 'euclidean'))
        # upper wrt image orientation, not position in array
        upperB = gridBounds[0, 1] - self._gridOffset * boundDist[0, 3]
        lowerB = gridBounds[3, 1] + self._gridOffset * boundDist[0, 3]
        upperPoints = []
        lowerPoints = []
        for keyPoint in keyPoints:
            if keyPoint.pt[1] < upperB:
                upperPoints.append(keyPoint.pt)
            elif keyPoint.pt[1] > lowerB:
                lowerPoints.append(keyPoint.pt)
        upperPoints = np.array(upperPoints).reshape(-1, 2)
        lowerPoints = np.array(lowerPoints).reshape(-1, 2)
        if not upperPoints.size and len(lowerPoints) == 7:
            markers.append((0, np.array([image.shape[1] // 2, upperB]).reshape((-1, 1))))
            roughCenter = np.mean(lowerPoints, axis=0).reshape((-1, 1))
            markers.append((len(lowerPoints), roughCenter))
        elif not lowerPoints.size and len(upperPoints) == 7:
            markers.append((0, np.array([image.shape[1] // 2, lowerB]).reshape((-1, 1))))
            roughCenter = np.mean(upperPoints, axis=0).reshape((-1, 1))
            markers.append((len(upperPoints), roughCenter))
        elif upperPoints.size and lowerPoints.size:
            roughCenter = np.mean(lowerPoints, axis=0).reshape((-1, 1))
            markers.append((len(lowerPoints), roughCenter))
            roughCenter = np.mean(upperPoints, axis=0).reshape((-1, 1))
            markers.append((len(upperPoints), roughCenter))
        gridCenter = (image.shape[1] // 2, image.shape[0] // 2)
        return self._side_from_markers(markers, gridCenter)

    def _marker_to_side(self, marker, gridCenter):
        for side, markers in self._markerSides.iteritems():
            if marker[0] == markers[0]:
                inverted = self._is_side_inverted(marker[1], True, gridCenter)
                return side, inverted
            elif marker[0] == markers[1]:
                inverted = self._is_side_inverted(marker[1], False, gridCenter)
                return side, inverted
        return None, None

    def _is_side_inverted(self, markerCenter, isTop, gridCenter):
        if isTop:
            return True if markerCenter[1] >= gridCenter[1] else False
        else:
            return True if markerCenter[1] < gridCenter[1] else False

    # recursive function to count nested contours
    def _parse_contour(self, index, count):
        if self.hierarchy[index][2] != -1 and count < self._nestMin:
            count = self._parse_contour(self.hierarchy[index][2], count + 1)
        elif count >= self._nestMin and self.hierarchy[index][0] != -1:
            count = self._parse_contour(self.hierarchy[index][0], count + 1)
        return count

    # Expands bounds of grid to include markers based on ratio of known checkerboard lengths in image
    def _expand_bounds(self, boundPoints):
        returnedPoints = boundPoints.copy()
        # Find distance between corners for ratio
        boundDist = spatial.distance.squareform(spatial.distance.pdist(boundPoints, 'euclidean'))
        # Apply offsets of markers
        xDist = boundDist[0, 1]
        yDist = boundDist[0, 3]
        offsetsX = np.array([xDist * x for x in self._markerOffsets[0::2]]).reshape((-1, 1))
        offsetsY = np.array([yDist * y for y in self._markerOffsets[1::2]]).reshape((-1, 1))
        returnedPoints += np.hstack([offsetsX, offsetsY])
        return returnedPoints.astype('float32')


# Test Script
if __name__ == '__main__':
    myCal = Calibration(calibFilename='intrinsic.cfg')
    target = myCal._load_target('target.cfg')
    test = 1
