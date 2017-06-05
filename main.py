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
    # Camera Positions Parallel Trial:
    # Pig X:0,Y:310,Z:1000
    # Small Cube: X:1000,Y:380,Z:2000
    # Large Cube: X:0,Y:230,Z:3000

    # Camera Positions Wide Trial:
    # Pig X:0,Y:435,Z:1000
    # Kettle: X:400,Y:435,Z:2000
    # Hotpack: X:-300,Y:435,Z:3000
    # Cam 1: X:0,Y:0,Z:0, Straight on
    # Cam 2: X:870,Y:75,Z:700, 60deg left
    # Cam 3: X:-850,Y:75,Z:640, 60deg right
