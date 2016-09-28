import cv2
import numpy as np

__author__ = 'Michael Holmes'

# ----------------------------------------------------------------
#
# Configuration file for the system
#
# ----------------------------------------------------------------

# Configuration Filenames
masterConfigFiles = dict(commonFilename='commonCalib.cfg',
                         intrinFilename='intrinCalib1-480.cfg',
                         extrinFilename='extrinCalib3Parallel.cfg',
                         targetFilename='targetCalib.cfg')

slaveConfigFiles = dict(commonFilename='commonCalib.cfg',
                        intrinFilename='intrinCalib3-480.cfg')

# Miscellaneous Parameters
mainParams = dict(intrinCalibOnly=0,
                  noNetwork=0,
                  viewLocalTargets=0,
                  maxRayDist=15000)

# Local Tracking Parameters
trackingParams = dict(numTargets=3,
                      maxFailedCycles=5,
                      deadZone=20,
                      shiftMulti=2)

# Network Parameters
netParams = dict(nodeType='Master',
                 numSlaves=2,
                 port=42681,
                 bcAddr='192.168.1.255',
                 bcReattempts=5,
                 msgTimeout=10,
                 waitTimeout=600)

# Camera Parameters
camParams = dict(camIndex=1,
                 capHeight=480,
                 capWidth=640,
                 capFrameRate=30.0)

# Kalman Filter / Sparse Optical Flow Parameters
kalmanParams = dict(initState=np.zeros((6, 1)),
                    initControl=np.zeros((1, 1)),
                    initP=np.spacing(1) * np.eye(6),
                    posSTD=12.0,  # pos 12, vel 10, accel 1.5, range 6
                    velSTD=10.0,
                    accelSTD=1.5,
                    rangeSTD=6.0,
                    predictStep=0.02,
                    updateStep=0.1)

flowParams = dict(winSize=(15, 15),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                  trackLen=20)

featureParams = dict(maxCorners=500,
                     qualityLevel=0.3,
                     minDistance=7,
                     blockSize=7)

# Colour Segmentation Parameters
colourParams = dict(hueWindow=10,
                    satWindow=20,
                    kernelSize=5,
                    colourTargetRadius=15)

# Initialise Calibration Parameters
calibCommonParams = dict(refineWindow=(5, 5),
                         numIter=30,
                         epsIter=0.1)

calibIntParams = dict(intPatternType='Circle',
                      intPatternSize=(4, 11),
                      intPatternDimension=74,
                      intNumImages=20,
                      camMatrix=np.zeros((3, 3)),
                      distCoefs=np.zeros((5, 1)))

calibExtParams = dict(extPatternType='Square',
                      extPatternSize=(6, 9),
                      extPatternDimension=18,
                      markerOffsets=[-0.66, -0.5, 0.66, -0.5, 0.66, 0.5, -0.66, 0.5],
                      gridOffset=0.21,
                      nestMin=4,
                      objErrTol=1.0,
                      extCalTime=15.0,
                      extCapInterval=0.1,
                      staticTarget=0,
                      markerSides=dict(N=(6, 1),
                                       S=(5, 2),
                                       E=(4, 3),
                                       W=(7, 0)),
                      extrinTransforms=dict())

