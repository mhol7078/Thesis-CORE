import cv2
import numpy as np
import json
import time
import os
from sys import exit
from scipy import spatial
from HSConfig import calibCommonParams as cCP, calibIntParams as cIP, calibExtParams as cEP, miscParams as mP
from rpp import rpp
import shelve

# Thesis Notes:
# Went for 4 grid 3d pattern because of correspondence problem
# Went for contour-based fiducial like in QR code because of key-point false-positive problem
# Didn't embed characters inside circles because it caused problems at longer range
# Didn't use custom circle pattern per side because it would require scanning for each different pattern per frame
# Had issue identifying 2 separate grids in single frame, even if not both symmetric/asymmetric
# Dumped circle grids altogether due to clashes with square grids/other circle grids
# Rewrote calibration capture to suit offline processing

# TODO: Add multi-zonal support post-thesis
# TODO: Take calibration suite offline.
# TODO: Employ RANSAC backed PnP

##----------------------------------------------------------------##
#
# Class to handle all intrinsic and extrinsic camera calibration
# operations.
#
##----------------------------------------------------------------##

# GLOBAL to drive mouse events
target0 = None


# Mouse event for manual calibration functions
def onMouse(event, x, y, flags, param):
    global target0
    if event == cv2.EVENT_LBUTTONDOWN:
        target0 = (x, y)


class Calibration:
    def __init__(self, camRef, netRef, commonFilename=None, intrinFilename=None, extrinFilename=None,
                 targetFilename=None, noExtrin=False):
        self._camRef = camRef
        self._netRef = netRef
        self._intPatternType = None
        self._extPatternType = None
        self._intPatternSize = None
        self._extPatternSize = None
        self._intPatternDimension = None
        self._extPatternDimension = None
        self._intNumImages = None
        self._refineWindow = None
        self._numIter = None
        self._epsIter = None
        self._criteria = None
        self._camMatrix = None
        self._distCoefs = None
        self._calibTarget = None
        self._hierarchy = None
        self._nestMin = None
        self._objErrTol = None
        self.extCalTime = None
        self._extCapInterval = None
        # offsets list as a percentage of the length of the circular grid longest sides for each marker corner,
        # clockwise from top left
        self._markerOffsets = None
        self._gridOffset = None
        # Relating markers to sides of the 3d rig
        self._markerSides = None

        # Load common calibration parameters
        if commonFilename is not None and os.path.isfile(commonFilename):
            ret, calibCommonParams = self._read_common_from_file(commonFilename)
            if ret:
                self._members_from_params('common', **calibCommonParams)
                print 'Loaded common calibation parameters from %s' % commonFilename
            else:
                print 'Unable to load provided common parameters from file, reverting to defaults.'
                self._members_from_params('common', **cCP)
                self._write_common_to_file('commonCalib.cfg')
                print 'Written common calibration file to <commonCalib.cfg>.'
        else:
            print 'No common calibration parameters file given, reverting to defaults.'
            self._members_from_params('common', **cCP)
            self._write_common_to_file('commonCalib.cfg')
            print 'Written common calibration file to <commonCalib.cfg>.'

        # Load intrinsics
        if intrinFilename is not None and os.path.isfile(intrinFilename):
            ret, calibIntParams = self._read_intrin_from_file(intrinFilename)
            if ret:
                self._members_from_params('intrin', **calibIntParams)
                print 'Loaded intrinsic parameters from %s.' % intrinFilename
            else:
                print 'Unable to load intrinsic calibration from file, please remove %s to re-calibrate camera.' % \
                      intrinFilename
                exit()
        else:
            print 'Intrinsic calibration filename missing. Intrinsic re-calibration required.'
            userIn = raw_input('Perform intrinsic calibration? Requires non-headless node. (y/n): ')
            while userIn != 'y' and userIn != 'n':
                userIn = raw_input('Perform intrinsic calibration? Requires non-headless node. (y/n): ')
            if userIn == 'y':
                self._members_from_params('intrin', **cIP)
                ret = self._calibrate_int()
                if ret:
                    self._write_intrin_to_file('intrinCalib.cfg')
                    print 'Written intrinsic calibration file to <intrinCalib.cfg>.'
                else:
                    print 'Cannot continue without calibration.'
                    exit()
            else:
                print 'Cannot continue without intrinsic calibration.'
                exit()

        # Quick exit for not dealing with extrinsic calibration in main program/networking requirements.
        if noExtrin:
            self._members_from_params('extrin', **cEP)
            print 'Finished non-extrinsic calibration.'
            return

        # Load extrinsics
        # Load from file
        if self._netRef.name == 'Master' and extrinFilename is not None and os.path.isfile(extrinFilename):
            # Load extrinsic calibration data and tell slaves no extrinsic calibration is required.
            ret, calibExtParams = self._read_extrin_from_file(extrinFilename)
            if not ret:
                print 'Failed to load extrinsic calibration from file. Please remove %s to recalibrate camera.' % \
                      extrinFilename
                exit()
            else:
                self._netRef.send_comms_all('NoCalibrate')
                print 'Loaded extrinsic calbration from %s.' % extrinFilename

        # Else perform new extrinsic calibration
        # Master side
        elif self._netRef.name == 'Master':
            print 'Extrinsic calibration file missing. Extrinsic re-calibration required.'
            # Load base extrinsic parameters from defaults
            self._members_from_params('extrin', **cEP)
            print 'Loaded default extrinsic parameters.'
            # Load target configuration
            if targetFilename is not None and os.path.isfile(targetFilename):
                ret = self._load_target(targetFilename)
                if ret:
                    print 'Loaded calibration target from %s' % targetFilename
                else:
                    print 'Unable to load target calibration from file. ' \
                          'Please remove %s to re-calibrate target.' % targetFilename
                    exit()
            else:
                print 'Target calibration file missing. Target calibration required.'
                userIn = raw_input('Perform target calibration? Requires non-headless node. (y/n): ')
                while userIn != 'y' and userIn != 'n':
                    userIn = raw_input('Perform target calibration? Requires non-headless node. (y/n): ')
                if userIn == 'y':
                    ret = self._calibrate_target()
                    if ret:
                        self._save_target('targetCalib.cfg')
                        print 'Target calibration successful, saved calibration to <targetCalib.cfg>'
                    else:
                        print 'Failed to calibrate target. Cannot continue without target.'
                        exit()

            # Prompt for extrinsic calibration and filename
            userIn = raw_input('Perform extrinsic calibration? (y/n): ')
            while userIn != 'y' and userIn != 'n':
                userIn = raw_input('Perform extrinsic calibration? (y/n): ')
            if userIn == 'y':
                # Begin master extrinsic calibration
                self._calibrate_ext_master()
                # If successful, save calibration and alert slave nodes to continue
            else:
                print 'Cannot continue without extrinsic calibration.'
                exit()

        # Slave side
        elif self._netRef.name == 'Slave':
            # Wait to see if Master node requires new extrinsic calibration
            print 'Waiting for extrinsic calibration instructions.'
            ret = self._netRef.netEventDict['calibSync'].wait(self._netRef.waitTimeout)
            if not ret:
                print 'Network wait timeout. Cannot continue.'
                exit()
            self._netRef.netEventDict['calibSync'].clear()
            # If commanded to calibrate
            if self._netRef.calibFlag:
                print 'Extrinsic calibration required.'
                # Load base extrinsic parameters from defaults
                self._members_from_params('extrin', **cEP)
                print 'Loaded default extrinsic parameters.'
                print 'Commencing extrinsic calibration...'
                extrinTransforms = self._calibrate_ext_slave()
                print 'Waiting for command to transmit calibration data.'
                ret = self._netRef.netEventDict['calibSync'].wait(self._netRef.waitTimeout)
                if not ret:
                    print 'Network wait timeout. Cannot continue.'
                    exit()
                self._netRef.netEventDict['calibSync'].clear()
                # Send extrinsic transforms to Master
                if not self._netRef.calibFlag:
                    print 'Master aborted extrinsic calibration. Exiting.'
                    exit()
                print 'Transferring extrinsic calibration data...'
                if extrinTransforms is None:
                    print 'Unable to transfer data: failed to capture datapoints.'
                    exit()
                for side, transform, capTime, objErr in extrinTransforms:
                    data = self._package_extrin_data(side, transform, capTime, objErr)
                    self._netRef.sendQueue.put(('Calibrate', data))
                print 'Sending transfer complete signal.'
                self._netRef.sendQueue.put(('Success', None))
                # Wait for calibration complete signal from Master
                print 'Waiting for Master to compile extrinsic calibration.'
                ret = self._netRef.netEventDict['calibSync'].wait(self._netRef.waitTimeout)
                if not ret:
                    print 'Network wait timeout. Cannot continue.'
                    exit()
                self._netRef.netEventDict['calibSync'].clear()
                print 'Extrinsic calibration complete.'
            else:
                print 'No extrinsic calibration required.'
        return

    def _members_from_params(self, fileType, intPatternType=None, extPatternType=None, intPatternSize=None,
                             extPatternSize=None, intPatternDimension=None, extPatternDimension=None, intNumImages=None,
                             refineWindow=None, numIter=None, epsIter=None, camMatrix=None, distCoefs=None,
                             nestMin=None, objErrTol=None, extCalTime=None, extCapInterval=None, markerOffsets=None,
                             gridOffset=None, markerSides=None):
        if fileType == 'common':
            self._refineWindow = refineWindow
            self._numIter = numIter
            self._epsIter = epsIter
            self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, numIter, epsIter)
        elif fileType == 'intrin':
            self._intPatternType = intPatternType
            self._intPatternSize = intPatternSize
            self._intPatternDimension = intPatternDimension
            self._intNumImages = intNumImages
            self._camMatrix = camMatrix
            self._distCoefs = distCoefs
        elif fileType == 'extrin':
            self._extPatternType = extPatternType
            self._extPatternSize = extPatternSize
            self._extPatternDimension = extPatternDimension
            self._nestMin = nestMin
            self._objErrTol = objErrTol
            self.extCalTime = extCalTime
            self._extCapInterval = extCapInterval
            self._markerOffsets = markerOffsets
            self._gridOffset = gridOffset
            self._markerSides = markerSides
        return

    def _write_common_to_file(self, filename):
        calibParams = dict(refineWindow=self._refineWindow,
                           numIter=self._numIter,
                           epsIter=self._epsIter)
        fd = open(filename, 'w')
        json.dump(calibParams, fd)
        fd.close()
        print 'Successfully written common calibration to file.'
        return

    def _read_common_from_file(self, filename):
        # Load params from .cfg
        fd = open(filename, 'r')
        calibParams = json.load(fd)
        fd.close()
        if calibParams is None:
            return False, None
        calibParams['refineWindow'] = tuple(calibParams['refineWindow'])
        return True, calibParams

    def _write_intrin_to_file(self, filename):
        camMatrixList = []
        distCoefsList = []
        for x in self._camMatrix.flat:
            camMatrixList.append(x)
        for x in self._distCoefs.flat:
            distCoefsList.append(x)
        calibParams = dict(intPatternType=self._intPatternType,
                           intPatternSize=self._intPatternSize,
                           intPatternDimension=self._intPatternDimension,
                           intNumImages=self._intNumImages,
                           camMatrix=camMatrixList,
                           distCoefs=distCoefsList)
        fd = open(filename, 'w')
        json.dump(calibParams, fd)
        fd.close()
        print 'Successfully written intrinsic calibration to file.'
        return

    def _read_intrin_from_file(self, filename):
        # Load params from .cfg
        fd = open(filename, 'r')
        calibParams = json.load(fd)
        fd.close()
        if calibParams is None:
            return False, None
        # rectify intrinsic matrix to numpy array
        camMatrixList = calibParams['camMatrix']
        camMatrix = np.float64(camMatrixList).reshape((3, 3))
        distCoefsList = calibParams['distCoefs']
        distCoefs = np.float64(distCoefsList).reshape((5, 1))
        calibParams['distCoefs'] = distCoefs
        calibParams['camMatrix'] = camMatrix
        # rectify lists to tuples
        calibParams['intPatternSize'] = tuple(calibParams['intPatternSize'])
        return True, calibParams

    def _write_extrin_to_file(self, filename):
        calibParams = dict(extPatternType=self._extPatternType,
                           extPatternSize=self._extPatternSize,
                           extPatternDimension=self._extPatternDimension,
                           nestMin=self._nestMin,
                           extCalTime=self.extCalTime,
                           extCapInterval=self._extCapInterval,
                           markerOffsets=self._markerOffsets,
                           gridOffset=self._gridOffset,
                           markerSides=self._markerSides)
        fd = open(filename, 'w')
        json.dump(calibParams, fd)
        fd.close()
        print 'Successfully written extrinsic calibration to file.'
        return

    def _read_extrin_from_file(self, filename):
        # Load params from .cfg
        fd = open(filename, 'r')
        calibParams = json.load(fd)
        fd.close()
        if calibParams is None:
            return False, None
        calibParams['extPatternSize'] = tuple(calibParams['extPatternSize'])
        return True, calibParams

    def get_intrinsics(self):
        return self._camMatrix, self._distCoefs

    def _gen_objp_grid(self, patternType=None, patternSize=None, patternDimension=None):
        if patternType is None:
            patternType = self._intPatternType
            patternSize = self._intPatternSize
            patternDimension = self._intPatternDimension
        objectPoints = np.zeros((np.prod(patternSize), 3), np.float32)
        xvals = np.mgrid[0:patternSize[0], 0:patternSize[1]][0].T
        xvals *= patternDimension
        yvals = np.mgrid[0:patternSize[0], 0:patternSize[1]][1].T
        if patternType == 'Circle':
            # Asymmetric circle grid
            xvals[1::2, :] += patternDimension / 2
            yvals *= patternDimension / 2
        else:
            yvals *= patternDimension
        xvals = xvals.flatten(order='C').reshape((-1, 1))
        yvals = yvals.flatten(order='C').reshape((-1, 1))
        objectPoints[:, :1] = xvals
        objectPoints[:, 1:2] = yvals
        return objectPoints

    # Function to perform intrinsic calibration of camera using square or circular pattern
    def _calibrate_int(self):
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
        while calibCount < self._intNumImages:
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
                            cv2.putText(frameCopy, 'Image Count: %d of %d accepted' % (calibCount, self._intNumImages),
                                        (10, frameCopy.shape[0] - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                            cv2.imshow('Calibration Capture', frameCopy)
                            print 'Image Count: %d of %d accepted.' % (calibCount, self._intNumImages)
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
                    cv2.putText(greyFrame, 'Unable perfoto locate grid', (10, greyFrame.shape[0] - 10),
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
            ret, self._camMatrix, self._distCoefs, rotVecs, transVecs = cv2.calibrateCamera(objPArray, imgPArray,
                                                                                            (w, h),
                                                                                            self._camMatrix,
                                                                                            self._distCoefs, rotVecs,
                                                                                            transVecs)
            print 'Intrinsic calibration complete.'
            cv2.destroyWindow('Calibration Capture')
            return True
        else:
            print 'Unable to calibrate intrinsics.'
            cv2.destroyWindow('Calibration Capture')
            return False

    # Function to gather local extrinsic calibration data, receive data from slaves, then correlate to find all slave
    # extrinsic positions wrt the Master
    def _calibrate_ext_master(self):
        # Tell slaves to begin calibration
        self._netRef.send_comms_all('Calibrate')
        # Crunch local target data while slaves crunch theirs
        localTargets = self._calibrate_ext_slave()
        # Send command for slaves to begin transmission if gathered local data successfully.
        if localTargets is not None:
            print 'Sending slave transmission command.'
            self._netRef.send_comms_all('Success')
        else:
            print 'Unable to calibrate local scene data, cannot continue.'
            return 1
        # Wait for slave data to arrive
        slaveData = {}
        startTime = time.time()
        currTime = startTime
        finishedSlaves = 0
        print 'Waiting for slaves to finish local data capture and transmit.'
        while currTime - startTime < self._netRef.waitTimeout and finishedSlaves != self._netRef.num_connected():
            slavePacket = self._netRef.get_slave_data()
            if slavePacket is not None and slavePacket[1] == self._netRef.commMsgPrefixes['Calibrate']:
                addr, prefix, data = slavePacket
                if addr[0] in slaveData:
                    slaveData[addr[0]].append(data)
                else:
                    slaveData[addr[0]] = [data]
            elif slavePacket is not None and slavePacket[1] == self._netRef.commMsgPrefixes['Success']:
                finishedSlaves += 1
            currTime = time.time()

        # If not all slaves submitted data, end calibration
        if len(slaveData) != self._netRef.num_connected():
            print 'Did not receive local data from all slaves. Cannot continue.'
            return 1
        # Unpack data packets and correct time wrt Master
        for addr in slaveData.keys():
            offsetTime = self._netRef.slaveSyncTimes[addr]
            for idx, packet in enumerate(slaveData[addr]):
                side, transform, capTime, objErr = self._depackage_extrin_data(packet)
                slaveData[addr][idx] = (side, transform, capTime - offsetTime, objErr)
        extrinData = slaveData.copy()
        extrinData['localhost'] = localTargets
        del slaveData, localTargets
        saveData = shelve.open('extrinData')
        saveData['extrinData'] = extrinData
        saveData.close()
        # Determine optimal calibration sequence by identifying pairs of nodes, from Master out
        # Return transforms wrt Master and send calibration complete command to slaves.
        print 'Extrinsic calibration successful.'
        self._netRef.send_comms_all('Success')
        return 0

    # Function to gather extrinsic calibration data for a nominated time calTime then send data to the Master Node
    def _calibrate_ext_slave(self):
        if __debug__:
            print 'Running in DEBUG MODE!'
        # Setup capture file and timing list
        # codec = cv2.VideoWriter_fourcc(*mP['capCodec'])
        # capFile = cv2.VideoWriter('extrinSlaveCap.avi', codec, mP['capFrameRate'], (mP['capWidth'], mP['capHeight']))
        imgList = []
        timeList = []

        foundTargets = []
        print 'Capturing scene for %f seconds...' % self.extCalTime
        # Initialise Timer
        startTime = time.time()
        currTime = startTime
        lastCheckTime = startTime - self._extCapInterval - 1
        while currTime - startTime < self.extCalTime:
            self._camRef.get_frame()
            frameCopy = self._camRef.current_frame()
            currTime = time.time()
            if currTime - lastCheckTime > self._extCapInterval:
                # capFile.write(frameCopy)
                imgList.append(cv2.cvtColor(frameCopy, cv2.COLOR_BGR2GRAY))
                timeList.append(currTime)
                lastCheckTime = time.time()

        # Release file for reading
        # capFile.release()

        frameCount = 0
        numFrames = len(imgList)
        print 'Searching captured scene (%d frames) for valid frames...' % numFrames
        currTime = time.time()
        lastCheckTime = currTime
        for frame, currTime in zip(imgList, timeList):
            # Search for square grid
            ret, corners = cv2.findChessboardCorners(frame, self._extPatternSize, None)
            if ret:
                cv2.cornerSubPix(frame, corners, self._refineWindow, (-1, -1), self._criteria)
                side = self._find_side(frame, corners)
                if side is not None:
                    foundTargets.append((side, corners, currTime))
                    print 'Count %d\t Found Side: %s\tINV %d' % (len(foundTargets), side[0], side[1])
                    sideString = 'Found Side: %s\tINV %d' % (side[0], side[1])
                else:
                    print 'Unable to determine side in frame %d.' % frameCount
                    sideString = 'Unable to find side in frame %d' % frameCount

                if __debug__:
                    cv2.drawChessboardCorners(frameCopy, self._extPatternSize, corners, True)
                    cv2.putText(frameCopy, sideString, (10, frameCopy.shape[0] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    cv2.imshow('Extrinsic Calibration', frameCopy)
                    cv2.waitKey(100)

            elif __debug__:
                cv2.putText(frameCopy, 'Unable to locate calibration grid', (10, frameCopy.shape[0] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                cv2.imshow('Extrinsic Calibration', frameCopy)
                cv2.waitKey(100)
            frameCount += 1
            currTime = time.time()
            if currTime - lastCheckTime > 10:
                lastCheckTime = time.time()
                print '%d%% complete.' % int(frameCount * 100.0 / numFrames)

        if __debug__:
            cv2.destroyWindow('Extrinsic Calibration')

        # Parse valid frames for extrinsic data
        if len(foundTargets):
            print 'Parsing valid frames for intrinsic data...'
            returnTargets = []
            objPoints = self._gen_objp_grid(self._extPatternType, self._extPatternSize, self._extPatternDimension)
            for side, corners, capTime in foundTargets:
                rot, trans, objErr = self._solve_rpp(objPoints, corners)
                if objErr < self._objErrTol:
                    transform = np.hstack((np.vstack((rot, np.zeros((1, 3)))), np.vstack((trans, 1))))
                    returnTargets.append((side, transform, capTime, objErr))
            return returnTargets
        else:
            print 'Unable to find valid frames in scene capture.'
            return None

    def _package_extrin_data(self, side, transform, capTime, objErr):
        packet = '%s:%d' % (side[0], side[1])
        for val in transform.flatten(1):
            packet += ':%s' % repr(val)
        packet += ':%s' % repr(capTime)
        packet += ':%s' % repr(objErr)
        return packet

    def _depackage_extrin_data(self, packet):
        splitPack = packet.split(':')
        side = (splitPack[0], int(splitPack[1]))
        transform = []
        for idx in range(2, 18):
            transform.append(float(splitPack[idx]))
        transform = np.array(transform).reshape((4, 4))
        capTime = float(splitPack[-2])
        objErr = float(splitPack[-1])
        return side, transform, capTime, objErr

    def _calibrate_target(self):
        targetSides = []
        capImages = self._intNumImages
        while True and capImages:
            self._camRef.get_frame()
            frameCopy = self._camRef.current_frame()
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
                        cv2.putText(frameCopy, 'Unable to locate side 1 markers', (10, frameCopy.shape[0] - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                        print 'Unable to locate side 1 markers.'
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
                    cv2.putText(frameCopy, 'Unable to locate any grids', (10, frameCopy.shape[0] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    print 'Unable to locate any grids.'
                    cv2.imshow('Target Calibration', frameCopy)
                    cv2.waitKey(1000)
                    continue
                if ret1 and not ret2 and corners2 is not None:
                    # Complete partial second side
                    ret2, corners2 = self._complete_grid(frameCopy, corners2, 300)
                if ret1 and ret2:
                    cv2.cornerSubPix(greyFrame, corners2, self._refineWindow, (-1, -1), self._criteria)
                    side2 = self._find_side(greyFrame, corners2)
                    if self._parse_sides(side1, side2):
                        targetSides.append((side1, corners1, side2, corners2))
                        cv2.drawChessboardCorners(frameCopy, self._extPatternSize, corners2, True)
                        cv2.drawChessboardCorners(frameCopy, self._extPatternSize, corners1, True)
                        capImages -= 1
                        sideString = 'Found Sides: %s:INV %d   %s:INV %d' % (side1[0], side1[1], side2[0], side2[1])
                    else:
                        sideString = 'Unable to locate side 2 markers.'
                    cv2.putText(frameCopy, sideString, (10, frameCopy.shape[0] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    print sideString
                    cv2.imshow('Target Calibration', frameCopy)
                    cv2.waitKey(1000)
                else:
                    cv2.putText(frameCopy, 'Unable to locate side 2 grid', (10, frameCopy.shape[0] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    print 'Unable to locate side 2 grid.'
                    cv2.imshow('Target Calibration', frameCopy)
                    cv2.waitKey(1000)
            elif userIn & 0xFF == ord(' '):
                break

        # Calibrate Sides wrt North face
        greyFrame = self._camRef.current_frame()
        greyFrame = cv2.cvtColor(greyFrame, cv2.COLOR_BGR2GRAY)
        cv2.putText(greyFrame, 'Performing target calibration...', (10, greyFrame.shape[0] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow('Target Calibration', greyFrame)

        # savedData = shelve.open('targetData')
        # savedData['targetData'] = targetSides
        # savedData.close()
        # return False
        #
        # savedData = shelve.open('targetData')
        # targetSides = savedData['targetData']
        # savedData.close()

        print 'Performing target calibration...'
        self._calibTarget = self._parse_target_pnp_pairs(targetSides)
        cv2.destroyWindow('Target Calibration')
        if self._calibTarget is not None:
            print 'Target calibration successful.'
            return True
        else:
            print 'Target calibration failed.'
            return False

    def _save_target(self, filename):
        fd = open(filename, 'w')
        targetCopy = self._calibTarget.copy()
        for key in targetCopy.keys():
            targetCopy[key] = list(targetCopy[key].flatten(1))
        json.dump(targetCopy, fd)
        fd.close()
        return

    def _load_target(self, filename):
        fd = open(filename)
        target = json.load(fd)
        for key in target.keys():
            target[key] = np.array(target[key], dtype=np.float32).reshape((4, 4)).T
        fd.close()
        if len(target) == 4:
            self._calibTarget = target
            return True
        else:
            self._calibTarget = None
            return False

    def get_target(self):
        return self._calibTarget

    def _complete_grid(self, image, foundPoints, roiSize):
        imgCopy = image.copy()
        candidates = foundPoints.copy().reshape((-1, 2))
        # Resize ROI
        boundRect = cv2.boundingRect(foundPoints)
        boundPoints = np.array([boundRect[0], boundRect[1],
                                boundRect[0] + boundRect[2], boundRect[1],
                                boundRect[0] + boundRect[2], boundRect[1] + boundRect[3],
                                boundRect[0], boundRect[1] + boundRect[3]], dtype=np.float32).reshape((-1, 2))
        boundPoints = self._expand_bounds(boundPoints)
        boundRect = cv2.boundingRect(boundPoints)
        roiImg = image[boundRect[1]:boundRect[1] + boundRect[3], boundRect[0]:boundRect[0] + boundRect[2]]
        scaleParams = (boundRect[0], boundRect[1], float(roiSize) / roiImg.shape[1], float(roiSize) / roiImg.shape[0])
        roiImg = cv2.resize(roiImg, (roiSize, roiSize))

        # Try rescan with drawChessboardCorners from ROI image
        ret, scaledCorners = cv2.findChessboardCorners(roiImg, self._extPatternSize)
        if ret:
            greyFrame = cv2.cvtColor(roiImg, cv2.COLOR_BGR2GRAY)
            cv2.cornerSubPix(greyFrame, scaledCorners, self._refineWindow, (-1, -1), self._criteria)
            scaledCorners = np.hstack(
                (scaledCorners[:, 0, 0].reshape((-1, 1)), scaledCorners[:, 0, 1].reshape((-1, 1))))
            cornersUnscaled = self._transform_candidates(scaleParams, scaledCorners, True)
            return True, cornersUnscaled.reshape((-1, 1, 2)).astype(np.float32)

        # Try rectify ROI image and rescan with drawChessboardCorners
        rectifiedCandidates = self._rectify_refine_corners(roiImg)
        if rectifiedCandidates is not None:
            # Invert Transform points and return
            rectifiedCandidates = self._transform_candidates(scaleParams, rectifiedCandidates, True)
            return True, rectifiedCandidates.reshape((-1, 1, 2)).astype(np.float32)

        # Alternative manual method
        cv2.drawChessboardCorners(imgCopy, self._extPatternSize, candidates, False)
        cv2.putText(imgCopy, 'Complete grid manually? y/n', (10, imgCopy.shape[0] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow('Incomplete Grid', imgCopy)
        while True:
            userIn = cv2.waitKey(1000)
            if userIn & 0xFF == ord('y'):
                break
            elif userIn & 0xFF == ord('n'):
                return False, None
        cv2.destroyWindow('Incomplete Grid')
        adjCandidates = self._transform_candidates(scaleParams, candidates)
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
        while True:
            corners = []
            for corner in cornerLabels:
                titleStr = 'Click on the %s corner, space to cancel' % corner
                cv2.namedWindow(titleStr)
                cv2.setMouseCallback(titleStr, onMouse)
                reDraw = True
                while True:
                    if reDraw:
                        imgCopy = roiImg.copy()
                        if len(corners):
                            for x, y in corners:
                                cv2.circle(imgCopy, (x, y), 5, (255, 0, 0), -1)
                        cv2.imshow(titleStr, imgCopy)
                        reDraw = False
                    if target0 is not None:
                        corners.append([target0[0], target0[1]])
                        target0 = None
                        cv2.destroyWindow(titleStr)
                        break
                    if cv2.waitKey(100) & 0xFF == ord(' '):
                        cv2.destroyWindow(titleStr)
                        return None
            imgCopy = roiImg.copy()
            for x, y in corners:
                cv2.circle(imgCopy, (x, y), 5, (255, 0, 0), -1)
            confirmStr = 'Press (y/n) to confirm or try again, space to cancel'
            cv2.imshow(confirmStr, imgCopy)
            continueFlag = False
            while True:
                userIn = cv2.waitKey(50)
                if userIn & 0xFF == ord('y'):
                    continueFlag = True
                    cv2.destroyWindow(confirmStr)
                    break
                elif userIn & 0xFF == ord('n'):
                    cv2.destroyWindow(confirmStr)
                    break
                elif userIn & 0xFF == ord(' '):
                    cv2.destroyWindow(confirmStr)
                    return None
            if continueFlag:
                break
        corners = np.array(corners, dtype=np.float32).reshape((-1, 2))
        objCandidates = self._gen_objp_grid(self._extPatternType, self._extPatternSize, self._extPatternDimension)[:, :2]
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

    def _parse_target_pnp_pairs(self, targets):
        # Make sure all sides have been captured
        if not self._check_target_sides(targets):
            return False
        pairsDict = {}
        # Computer homologous transformation matrix for each pair and amalgamate
        objPoints = self._gen_objp_grid(self._extPatternType, self._extPatternSize, self._extPatternDimension).T
        for side1, center1, side2, center2 in targets:
            # Calc position wrt to camera
            center1 = center1[::-1]
            center2 = center2[::-1]
            rot1, trans1, objErr1 = self._solve_rpp(objPoints, center1)
            rot2, trans2, objErr2 = self._solve_rpp(objPoints, center2)
            if objErr1 < self._objErrTol and objErr2 < self._objErrTol:
                # Calc transform of 2 to 1
                # Invert 2nd side
                side2Inverted = self._invert_transform_matrix(self._form_transform_matrix(rot2, trans2))
                # Compound homologous transforms
                transform1 = self._form_transform_matrix(rot1, trans1)
                transform2to1 = np.dot(side2Inverted, transform1)

                keyStr = side1[0] + side2[0]
                if keyStr not in pairsDict and keyStr[::-1] not in pairsDict:
                    # Add new transformation matrix to dictionary
                    pairsDict[keyStr] = (transform2to1, 1)
                elif keyStr in pairsDict:
                    # Add new matrix to previous sum to average out later
                    pairsDict[keyStr] = (pairsDict[keyStr][0] + transform2to1, pairsDict[keyStr][1] + 1)
                elif keyStr[::-1] in pairsDict:
                    # Reverse transformation matrix and add to previous sum
                    transform1to2 = self._invert_transform_matrix(transform2to1)
                    pairsDict[keyStr[::-1]] = (pairsDict[keyStr[::-1]][0] + transform1to2,
                                               pairsDict[keyStr[::-1]][1] + 1)
        # Compute averages
        if len(pairsDict.keys()) == 4:
            for key, value in pairsDict.iteritems():
                pairsDict[key] = value[0] / float(value[1])
            # Generate output dictionary of transformation matrices from <Side> to North
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
            print 'Could not generate target face pair RPP solutions. (Generated %d of %d)' % (len(pairsDict.keys()), 4)
            return None

    def _invert_transform_matrix(self, matrix):
        rotInv = matrix[0:3, 0:3].reshape((3, 3)).T
        transInv = np.dot(-rotInv, matrix[0:3, 3].reshape((3, 1)))
        return np.hstack((np.vstack((rotInv, np.zeros((1, 3)))), np.vstack((transInv, 1))))

    def _form_transform_matrix(self, rot, trans):
        return np.hstack((np.vstack((rot, np.zeros((1, 3)))), np.vstack((trans, 1))))


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
        objCandidates = self._gen_objp_grid(self._extPatternType, self._extPatternSize, self._extPatternDimension)[:, :2]
        imgCandidates = imgPoints.reshape((-1, 2))
        objWarps = np.zeros((4, 2), dtype=np.float32)
        imgWarps = objWarps.copy()
        objWarps[0] = objCandidates[0]
        objWarps[1] = objCandidates[self._extPatternSize[0] - 1]
        objWarps[2] = objCandidates[-1]
        objWarps[3] = objCandidates[np.prod(self._extPatternSize) - self._extPatternSize[0]]
        # changed order because of stupid corner indexing update in OpenCV 3.0
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
        # cv2.waitKey(3000)
        # cv2.destroyWindow('ROI')
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
        if hierarchy is None:
            return None, None
        self._hierarchy = hierarchy[0]
        markers = []
        for idx, h in enumerate(self._hierarchy):
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
        if self._hierarchy[index][2] != -1 and count < self._nestMin:
            count = self._parse_contour(self._hierarchy[index][2], count + 1)
        elif count >= self._nestMin and self._hierarchy[index][0] != -1:
            count = self._parse_contour(self._hierarchy[index][0], count + 1)
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

    # Return pose (R, t, objErr) from input 3xn object points and 2xn image points
    def _solve_rpp(self, objPoints, imgPoints):
        imgPointsUndistorted = cv2.undistortPoints(imgPoints, self._camMatrix, self._distCoefs)
        imgPointsUndistorted = np.hstack(
            (imgPointsUndistorted[:, 0, 0].reshape((-1, 1)), imgPointsUndistorted[:, 0, 1].reshape((-1, 1)))).T
        return rpp(objPoints.copy(), imgPointsUndistorted)


def extrin_calibration(extrinData, binSize):
    # Construct data matrix monotonically ordered by time and normalise times/binSize
    binSizeCopy = binSize
    sortedData = []
    for addr in extrinData.keys():
        idxCount = 0
        for side, transform, capTime in extrinData[addr]:
            sortedData.append((addr, idxCount, capTime))
            idxCount += 1
    sortedData = np.array(sortedData, dtype=[('addr', 'S16'), ('idx', 'i4'), ('time', 'f8')])
    sortedData.sort(order='time')
    sortedTime = sortedData['time']
    sortedTime -= sortedTime.min() * np.ones_like(sortedTime)
    binSizeCopy = binSizeCopy / (sortedTime.max() - sortedTime.min())
    sortedTime /= (sortedTime.max() - sortedTime.min()) * np.ones_like(sortedTime)
    # Group pairs of data together based on time proximity, assign weighting as function of time discrepancy
    obsPairs = {}
    maxIdxNums = len(str(len(sortedTime)))
    for idx in range(len(sortedTime)):
        idxOffset = 1
        while idx + idxOffset < len(sortedTime) and sortedTime[idx + idxOffset] - sortedTime[idx] < binSizeCopy:
            str1 = str(idx).zfill(maxIdxNums)
            str2 = str(idx + idxOffset).zfill(maxIdxNums)
            if sortedData['addr'][idx] != sortedData['addr'][
                        idx + idxOffset] and str1 + str2 not in obsPairs and str2 + str1 not in obsPairs:
                weighting = 1.0 - ((sortedTime[idx + idxOffset] - sortedTime[idx]) / binSizeCopy)
                obsPairs[str1 + str2] = (idx, idx + idxOffset, weighting)
            idxOffset += 1
    # Prepare observation pair data for calibration
    calibPairs = {}
    for idx1, idx2, weighting in obsPairs.values():
        ob1 = sortedData[['addr', 'idx']][idx1]
        ob2 = sortedData[['addr', 'idx']][idx2]
        if ob1[0] + ':' + ob2[0] in calibPairs:
            calibPairs[ob1[0] + ':' + ob2[0]].append(((ob1[0], ob2[0]),
                                                      (extrinData[ob1[0]][ob1[1]], extrinData[ob2[0]][ob2[1]]),
                                                      weighting))
        elif ob2[0] + ':' + ob1[0] in calibPairs:
            calibPairs[ob2[0] + ':' + ob1[0]].append(((ob2[0], ob1[0]),
                                                      (extrinData[ob2[0]][ob2[1]], extrinData[ob1[0]][ob1[1]]),
                                                      weighting))
        else:
            calibPairs[ob1[0] + ':' + ob2[0]] = [((ob1[0], ob2[0]),
                                                  (extrinData[ob1[0]][ob1[1]], extrinData[ob2[0]][ob2[1]]),
                                                  weighting)]
    # Construct expanding node tree from Master out mapping node connections, check if there is a valid path to master
    # from each slave Node
    nodePairs = {}
    for pairKey, pair in calibPairs.iteritems():
        nodePairs[pairKey] = pair[0][0]
    treeHierarchy = construct_node_tree(nodePairs, 'localhost', len(extrinData))
    if treeHierarchy is None:
        print 'Unable to map each slave node back to the master, please adjust camera layout.'
        exit()
    # Collapse multiples of same pair (ie same 2 node pairings) using time window weightings
    collapse_compute_extrin_multiples(calibPairs, 10)
    # Compute position of each camera with respect to Master and return

    return 0


# Collapse multiple observations for the same node pairing based on weighting and compute direct extrinsic relation
def collapse_compute_extrin_multiples(calibPairs, nTopCandidates):
    for nodeKey, nodePairs in calibPairs.iteritems():
        # Isolate weightings, find index of min(nTopCandidates,all)
        weightings = np.array(list(enumerate(map(list, zip(*nodePairs))[2])),
                              dtype=[('idx', 'i4'), ('weighting', 'f8')])
        if len(nodePairs) < nTopCandidates:
            topWeightings = np.sort(weightings, order='weighting')[::-1]
        else:
            topWeightings = np.sort(weightings, order='weighting')[-1:-1 - nTopCandidates:-1]
        # Reject outliers in top candidates and determine extrinsic PnP solution of remaining pairs
        topWeightingsOutliers = mad_based_outlier(topWeightings['weighting'])
        topNodes = []
        loopCount = 0
        for idx, weighting in topWeightings:
            if not topWeightingsOutliers[loopCount]:
                pnpPosition = calc_extrin_pnp(nodePairs[idx][1:])
                topNodes.append(pnpPosition)
            loopCount += 1

    return 0


# Uses the 2 extrinsic position information sets to calculate the position of node 2 (in the namekey) wrt node 1
def calc_extrin_pnp(nodePair):
    # temporary line to load target without main program (target loader already integrated)
    target = tmp_load_target('target.cfg')

    test = 1


def tmp_load_target(filename):
    fd = open(filename)
    target = json.load(fd)
    for key in target.keys():
        target[key] = np.array(target[key], dtype=np.float32).reshape((4, 4)).T
    fd.close()
    if len(target) == 4:
        return target
    else:
        return None


def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    medAbsDeviation = np.median(diff)
    modifiedZScore = 0.6745 * diff / medAbsDeviation
    return modifiedZScore > thresh


# Checks entire tree for membership, returns level on tree
def get_node_level(targetNode, treeDict):
    nodeLevel = 0
    node = targetNode
    while node in treeDict:
        if treeDict[node][0] is None:
            break
        else:
            node = treeDict[node][0]
            nodeLevel += 1
    return nodeLevel


# Add node to tree
def add_node_to_tree(node1, node2, treeDict, checkedNodes, candidateNodes):
    if node2 not in treeDict:
        treeDict[node2] = (node1, get_node_level(node1, treeDict) + 1)
    else:
        newNodeLevel = get_node_level(node1, treeDict) + 1
        if newNodeLevel < treeDict[node2][1]:
            treeDict[node2] = (node1, newNodeLevel)
    if node2 not in candidateNodes and node2 not in checkedNodes:
        candidateNodes.append(node2)
    return treeDict, checkedNodes, candidateNodes


# Build tree down from the root master node to map connections efficiently
def construct_node_tree(calibPairs, rootNode, numNodes):
    treeDict = {rootNode: (None, 0)}
    pairsCopy = calibPairs.copy()
    keysToDelete = []
    candidateNodes = [rootNode]
    checkedNodes = []
    while len(pairsCopy) and len(candidateNodes):
        # Get next candidate node
        targetNode = candidateNodes[0]
        for currKey in keysToDelete:
            if currKey in pairsCopy:
                del pairsCopy[currKey]
        keysToDelete = []
        for pairKey, pairList in pairsCopy.iteritems():
            node1 = pairList[0]
            node2 = pairList[1]
            if node1 == targetNode:
                treeDict, checkedNodes, candidateNodes = add_node_to_tree(node1, node2, treeDict, checkedNodes,
                                                                          candidateNodes)
                keysToDelete.append(pairKey)
            elif node2 == targetNode:
                treeDict, checkedNodes, candidateNodes = add_node_to_tree(node2, node1, treeDict, checkedNodes,
                                                                          candidateNodes)
                keysToDelete.append(pairKey)
        # Move target node from candidate list to checked list
        del candidateNodes[0]
        checkedNodes.append(targetNode)

    if len(treeDict) == numNodes:
        return treeDict
    else:
        return None


# def test_node_tree_constructor(numTests, numNodes, nodePC):
#     import random
#     testSummary = []
#     for testIdx in range(numTests):
#         testNodes = [str(x) for x in range(numNodes)]
#         testPairs = {}
#         testKey = 0
#         for targetNode in testNodes:
#             for subNode in testNodes:
#                 randNum = random.randint(0, 100)
#                 if subNode != targetNode and randNum < nodePC:
#                     testPairs[str(testKey)] = (targetNode, subNode)
#                     testKey += 1
#         treeDict = construct_node_tree(testPairs, '0', numNodes)
#         if treeDict is None:
#             testSummary.append((False, treeDict, testPairs))
#         else:
#             testSummary.append((True, treeDict, testPairs))
#     return testSummary


if __name__ == '__main__':
    from camOps import CamHandler

    cam = CamHandler(0)
    net = None
    calib = Calibration(cam, net, commonFilename='commonCalib.cfg', intrinFilename='intrinCalib1.cfg', noExtrin=True)

    test = tmp_load_target('target.cfg')

    # ret = calib._calibrate_target()
    # if ret:
    #     calib._save_target('target.cfg')

    # numTrials = 100
    # numNodes = 10
    # linkChance = 15
    # summary = test_node_tree_constructor(numTrials, numNodes, linkChance)
    # successNum = 0
    # for fullLink, foo, bar in summary:
    #     if fullLink == True:
    #         successNum += 1
    # print 'For %d trials of %d nodes with a %d percent chance of node linkage, full linkage success rate = %f percent.' % (numTrials, numNodes, linkChance, float(successNum)/numTrials*100.0)

    # savedData = shelve.open('extrinData')
    # extrinData = savedData['extrinData']
    # savedData.close()
    # extrin_calibration(extrinData, 0.1)
