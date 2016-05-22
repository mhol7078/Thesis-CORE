import cv2
import numpy as np

# Initialise Kalman Filter/Sparse Optical Flow settings
kalmanParams = dict(initState=np.zeros((6, 1)),
                    initControl=np.zeros((1, 1)),
                    initP=np.spacing(1) * np.eye(6),
                    posSTD=12.0,
                    velSTD=10.0,
                    accelSTD=1.5,
                    rangeSTD=6.0,
                    predictStep=0.1,
                    updateStep=0.5)

flowParams = dict(winSize=(15, 15),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                  trackLen=10,
                  )

featureParams = dict(maxCorners=500,
                     qualityLevel=0.3,
                     minDistance=7,
                     blockSize=7)

colourParams = dict(hueWindow=10,
                    satWindow=10,
                    kernelSize=5)

calibParams = dict(intPatternType='Circle',
                   extPatternType='Square',
                   intPatternSize=(4, 11),
                   extPatternSize=(6, 9),
                   intPatternDimension=72,
                   extPatternDimension=19,
                   numImages=30,
                   refineWindow=(5, 5),
                   numIter=30,
                   epsIter=0.1,
                   camMatrix=np.zeros((3, 3)),
                   distCoefs=np.zeros((5, 1)),
                   nestMin=4,
                   calTime=120.0,
                   capInterval=2.0,
                   markerOffsets=[-0.66, -0.5, 0.66, -0.5, 0.66, 0.5, -0.66, 0.5],
                   gridOffset=0.21,
                   markerSides=dict(N=(6, 1),
                                    S=(5, 2),
                                    E=(4, 3),
                                    W=(7, 0)))

netParams = dict(nodeType='Slave',
                 numSlaves=1,
                 port=42680,
                 bcAddr='192.168.1.255')
