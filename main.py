from imageOps import LocalModeOne
from camOps import CamHandler
from calibration import Calibration
import sys
import cv2
from HSConfig import kalmanParams, flowParams, featureParams, colourParams, calibParams, netParams

__author__ = 'michael'

# ---------------------------------------------------------------------
#               MAIN FUNCTION BLOCK
# ---------------------------------------------------------------------

# TODO
# Extrinsic Calibration: Cube with 4 calibration circle grids

if __name__ == '__main__':

    # Initialise Local Camera
    myCam = CamHandler(0)
    myCam.start()

    # Undertake calibration
    # myCalib = Calibration(myCam, calibParams=calibParams)
    myCalib = Calibration(myCam, calibFilename='intrinsic.cfg', targetFilename='target1.cfg')
    # myCalib.calibrate_int()
    # myCalib.write_calib_to_file('intrinsic.cfg')
    myCalib.calibrate_target('target1.cfg')
    # Initialise Local Tracking Mode
    localMode = LocalModeOne(myCam, kalmanParams, flowParams, featureParams, colourParams)

    # Setup test window
    cv2.namedWindow('HindSight Main', cv2.WINDOW_AUTOSIZE)

    # Run system, tracking target position with the filter
    while myCam.is_opened():

        # Get new image
        myCam.get_frame()

        # Update tracker elapsed times
        localMode.update_elapsed_counters(myCam.current_deltaTime())

        # Run prediction stage if prediction increment has elapsed
        if localMode.predict_stage_elapsed():
            localMode.predict()

        # Run update stage if update increment has elapsed
        if localMode.update_stage_elapsed():
            localMode.new_obs_from_im(myCam.current_frame().copy())
            localMode.update(localMode.get_current_obs())

        # Push latest filter estimate to image window along with new image
        frameCopy = myCam.current_frame().copy()
        currEstimate = localMode.observe_model()
        cv2.circle(frameCopy, (currEstimate[0], currEstimate[1]), 10, (0, 255, 0), 1)
        cv2.imshow('HindSight Main', frameCopy)

        # Check for termination
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Close up and return
            myCam.stop()
            while myCam.isAlive():
                pass
            cv2.destroyAllWindows()
            break