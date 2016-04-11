from imageOps import LocalModeOne
from camOps import CamHandler
import cv2
import numpy as np

__author__ = 'michael'

# ---------------------------------------------------------------------
#               MAIN FUNCTION BLOCK
# ---------------------------------------------------------------------

# TODO
# Put webcam in its own thread to reduce delay
# Extrinsic Calibration: Cube with 4 calibration circle grids

if __name__ == '__main__':

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

    calibParams = dict(patternType='Square',
                       patternSize=(8, 6),
                       patternDimensions=(30, 30),
                       numImages=5,
                       refineWindow=11,
                       numIter=30,
                       epsIter=0.01)

    netParams = dict(nodeType='Slave',
                     port=42680,
                     maxQueueSize=15)

    # Initialise Local Camera
    myCam = CamHandler(0)
    myCam.start()

    # Undertake calibration
    myCam.calibrate_int(**calibParams)

    # Initialise Local Tracking Mode
    localMode = LocalModeOne(myCam, kalmanParams, flowParams, featureParams, colourParams)

    # Setup test window
    cv2.namedWindow('Filter Test', cv2.WINDOW_AUTOSIZE)

    # Run simulation, tracking ball position with the filter
    while myCam.is_opened():

        # Get new image
        myCam.get_frame()

        # Update tracker elapsed times
        localMode.tracker.update_elapsed_counters(myCam.deltaTime)

        # Run prediction stage if prediction increment has elapsed
        if localMode.tracker.predict_stage_elapsed():
            localMode.tracker.predict()

        # Run update stage if update increment has elapsed
        if localMode.tracker.update_stage_elapsed():
            localMode.new_obs_from_im(myCam.current_frame())
            localMode.tracker.update(localMode.currObs)

        # Push latest filter estimate to image window along with new image
        cv2.circle(myCam.current_frame(), (localMode.tracker.x[0], localMode.tracker.x[1]), 10, (0, 255, 0), 1)
        cv2.imshow('Filter Test', myCam.current_frame())

        # Check for termination
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Close up and return
            myCam.stop()
            while myCam.isAlive():
                pass
            cv2.destroyAllWindows()
            break
