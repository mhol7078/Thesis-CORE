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

calibParams = dict(patternType='Circle',
                   patternSize=(4, 11),
                   patternDimension=30,
                   numImages=5,
                   refineWindow=5,
                   numIter=30,
                   epsIter=0.1,
                   camMatrix=0)

netParams = dict(nodeType='Slave',
                 port=42680,
                 maxQueueSize=15)
