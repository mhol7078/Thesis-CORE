import cv2
from reconstruction import Reconstruction
import time
from HSConfig import netParams, mainParams

__author__ = 'Michael Holmes'


# ---------------------------------------------------------------------
#               MAIN FUNCTION BLOCK
# ---------------------------------------------------------------------
def main():

    # Initialise system
    if netParams['nodeType'] == 'Master':
        Reconstructor = Reconstruction('Master', noNetwork=mainParams['noNetwork'])
    else:
        Reconstructor = Reconstruction('Slave')

    # Shutdown if only intrinsic calibration was required
    if mainParams['intrinCalibOnly']:
        return True

    # Spawn Window For User Interface
    cv2.namedWindow('UserWindow')

    # User interface loop
    lastRefresh = time.time()
    while True:
        # Check for Reconstructor shutdown (for slaves)
        if not Reconstructor.isAlive():
            break

        # Post new update of locks
        if time.time() - lastRefresh > 0.2:
            currLocks = Reconstructor.collect_estimates()
            cv2.imshow('UserWindow', currLocks)
            lastRefresh = time.time()

        # Check for user input
        userIn = cv2.waitKey(1) & 0xFF
        if userIn == ord('q'):  # Shutdown system
            Reconstructor.shutdown()
            break
        elif userIn == ord(' ') and Reconstructor.name == 'Master':  # Toggle capture
            Reconstructor.toggle_capture()
        elif userIn == ord('r') and Reconstructor.name == 'Master':  # Begin reconstruction
            Reconstructor.toggle_reconstruct()
        elif userIn == ord('p') and Reconstructor.name == 'Master':  # Purge capture data
            Reconstructor.purge_capture_data()
    return True


if __name__ == '__main__':
    main()
    # Pig X:0,Y:310,Z:1000
    # Small Cube: X:1000,Y:380,Z:2000
    # Large Cube: X:0,Y:230,Z:3000
