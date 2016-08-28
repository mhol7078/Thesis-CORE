import cv2
from reconstruction import Reconstruction
from HSConfig import netParams

__author__ = 'Michael Holmes'


# ---------------------------------------------------------------------
#               MAIN FUNCTION BLOCK
# ---------------------------------------------------------------------
def main():
    # Initialise system
    if netParams['nodeType'] == 'Master':
        Reconstructor = Reconstruction('Master')
    else:
        Reconstructor = Reconstruction('Slave')

    # User interface loop
    while True:
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
