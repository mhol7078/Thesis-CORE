from imageOps import LocalModeOne
from camOps import CamHandler
from calibration import Calibration
import cv2
from networking import MasterNode, SlaveNode
from HSConfig import netParams, miscParams
from sys import exit

__author__ = 'michael'


# ---------------------------------------------------------------------
#               MAIN FUNCTION BLOCK
# ---------------------------------------------------------------------
def main():
    # Initialise system
    cam, net, calib, localMode = initialise(netParams['nodeType'])

    # Run system, tracking target position with the filter
    while cam.is_opened():

        # Get new image
        cam.get_frame()

        # Update tracker elapsed times
        localMode.update_elapsed_counters(cam.current_deltaTime())

        # Run prediction stage if prediction increment has elapsed
        if localMode.predict_stage_elapsed():
            localMode.predict()

        # Run update stage if update increment has elapsed
        if localMode.update_stage_elapsed():
            localMode.new_obs_from_im(cam.current_frame().copy())
            localMode.update(localMode.get_current_obs())

        # Push latest filter estimate to image window along with new image
        frameCopy = cam.current_frame().copy()
        currEstimate = localMode.observe_model()
        cv2.circle(frameCopy, (currEstimate[0], currEstimate[1]), 10, (0, 255, 0), 1)
        cv2.imshow('HindSight Main', frameCopy)

        # Check for termination
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Close up and return
            cam.stop()
            while cam.isAlive():
                pass
            cv2.destroyAllWindows()
            break


def initialise(nodeType):
    # Initialise local camera thread
    cam = CamHandler(miscParams['camIndex'])

    # Initialise networking threads
    if nodeType == 'Master':
        net = MasterNode(**netParams)
    else:
        net = SlaveNode(**netParams)
    ret = net.netEventDict['calibSync'].wait(netParams['waitTimeout'])
    net.netEventDict['calibSync'].clear()
    if not ret:
        print 'Network wait timeout. Cannot continue.'
        exit()
    # Load parameters or undertake calibration
    if nodeType == 'Master':
        calib = Calibration(cam, net, commonFilename='commonCalib.cfg', intrinFilename='intrinCalib1.cfg',
                            targetFilename='targetCalib.cfg')
    else:
        calib = Calibration(cam, net, commonFilename='commonCalib.cfg', intrinFilename='intrinCalib2.cfg')

    print 'Finished calibration.'
    exit(0)

    # Initialise Local Tracking Mode
    localMode = LocalModeOne(cam)

    # Setup test window
    cv2.namedWindow('HindSight Main', cv2.WINDOW_AUTOSIZE)

    return cam, net, calib, localMode


if __name__ == '__main__':
    main()
