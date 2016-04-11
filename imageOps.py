import cv2
import numpy as np
from scipy import spatial
from kalmanFilt import KalmanTrack

__author__ = 'Michael Holmes'

##----------------------------------------------------------------##
#
# Class to handle all tracking related queries, utilises the
# Kalman filter class
#
##----------------------------------------------------------------##

# GLOBAL to drive mouse events
target0 = [(-1, -1), (-1, -1), False, (-1, -1)]


# TODO:
# Outlier rejection in minimum enclosing circle calc
# Add toggle for showing track history
# Make markup a boxed average instead of single pixel

# Mouse event for local mode markup functions
def onMouse(event, x, y, flags, param):
    global target0
    if event == cv2.EVENT_LBUTTONDOWN:
        target0[0] = (x, y)
        target0[2] = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if target0[2]:
            target0[3] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        target0[1] = (x, y)
        cv2.destroyWindow('Choose Target')


class LocalModeOne:
    def __init__(self, camRef, kalmanParams, flowParams, featureParams, colourParams):
        # Markup Target
        self.tracker = KalmanTrack(**kalmanParams)
        self.currObs = np.zeros((2, 1))
        self.colourParams = colourParams
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (colourParams['kernelSize'], colourParams['kernelSize']))
        self.hsvLimits = np.zeros((4, 1))
        self.markup_target(camRef)
        del self.colourParams
        self.prevIm = cv2.cvtColor(camRef.current_frame(), cv2.COLOR_BGR2GRAY)
        # Update filter with initial positions, image resolution and target colour profile
        self.tracker.x[0] = self.currObs[0]
        self.tracker.x[1] = self.currObs[1]
        # Add Optical Flow and feature detection params
        self.colourLock = False
        self.colourTargetRadius = 5
        self.flowLock = False
        self.flowTrackHistoryMax = flowParams['trackLen']
        del flowParams['trackLen']
        self.flowTracks = []
        self.flowParams = flowParams
        self.featureParams = featureParams
        return

    def markup_target(self, camRef):
        global target0, frameImg
        cv2.waitKey(3000)
        camRef.get_frame()
        camRef.get_frame()  # Doubled to initialise time-step
        self.tracker.maxUV = camRef.current_frame().shape
        cv2.imshow('Choose Target', camRef.current_frame())
        cv2.setMouseCallback('Choose Target', onMouse)
        while target0[1][0] == -1:
            if not target0[2]:
                camRef.get_frame()
                cv2.imshow('Choose Target', camRef.current_frame())
            else:
                if not (target0[3][0] == -1):
                    frameCopy = camRef.current_frame()
                    cv2.rectangle(frameCopy, target0[0], target0[3], (0, 255, 0))
                    cv2.imshow('Choose Target', frameCopy)
                else:
                    cv2.imshow('Choose Target', camRef.current_frame())
            if cv2.waitKey(50) & 0xFF == ord(' '):
                cv2.destroyWindow('Choose Target')
                return
        bounds = self.set_roi(target0)
        self.currObs[0] = bounds[2]
        self.currObs[1] = bounds[5]
        if bounds[0] == bounds[1] or bounds[3] == bounds[4]:
            self.set_hsv_limits(camRef.current_frame()[int(bounds[2]), int(bounds[5])])
        else:
            self.set_hsv_limits(camRef.current_frame()[bounds[0]:bounds[1], bounds[3]:bounds[4]])
        return

    # Get bounding box from selected points
    def set_roi(self, target0):
        bounds = np.zeros((6, 1))
        bounds[0] = np.minimum(target0[0][0], target0[1][0])
        bounds[1] = np.maximum(target0[0][0], target0[1][0])
        bounds[2] = np.mean((target0[0][0], target0[1][0]))
        bounds[3] = np.minimum(target0[0][1], target0[1][1])
        bounds[4] = np.maximum(target0[0][1], target0[1][1])
        bounds[5] = np.mean((target0[0][1], target0[1][1]))
        return bounds

    # converts lone pixel from BGR->HSV
    def set_hsv_limits(self, bgr):
        # Correct for lone pixel selection
        if len(bgr) == 3:
            bgr = bgr.reshape(1, 1, 3)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hueMean = np.mean(hsv[:, :, 0])
        satMean = np.mean(hsv[:, :, 1])
        hsvLimits = np.zeros((4, 1))
        hsvLimits[0] = self.bound_hue_limits(hueMean - self.colourParams['hueWindow'])
        hsvLimits[1] = self.bound_hue_limits(hueMean + self.colourParams['hueWindow'])
        hsvLimits[2] = self.bound_sat_limits(satMean - self.colourParams['satWindow'])
        hsvLimits[3] = self.bound_sat_limits(satMean + self.colourParams['satWindow'])
        self.hsvLimits = hsvLimits
        return

    def bound_hue_limits(self, hue):
        while hue < 0 or hue > 255:
            if hue < 0:
                hue += 255
            elif hue > 255:
                hue -= 255
        return hue

    def bound_sat_limits(self, sat):
        if sat < 0:
            sat = 0
        elif sat > 255:
            sat = 255
        return sat

    def new_obs_from_im(self, image):
        # Get new measurement from colour extraction
        colourObs = self.new_colour_obs(image)
        # Get new measurement from sparse optical flow
        flowObs = self.new_flow_obs(image, colourObs)
        # Return point and update current member
        if self.colourLock and self.flowLock:
            self.currObs[0] = np.mean((colourObs[0], flowObs[0]))
            self.currObs[1] = np.mean((colourObs[1], flowObs[1]))
        elif self.colourLock:
            self.currObs = colourObs
        elif self.flowLock:
            self.currObs = flowObs
        else:
            self.currObs[0] = -1
            self.currObs[1] = -1
        return colourObs

    def new_colour_obs(self, image):
        # Convert new frame to HSV
        newHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Threshold Target
        newHSV = self.threshold_target(newHSV)
        cv2.imshow('Threshold', newHSV)
        # Morphological Opening
        newHSV = cv2.morphologyEx(newHSV, cv2.MORPH_OPEN, self.kernel)
        # Morphological Closing
        newHSV = cv2.morphologyEx(newHSV, cv2.MORPH_CLOSE, self.kernel)
        # Find largest blob
        contours, hierarchy = cv2.findContours(newHSV, cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE)
        targetOut = self.check_contours(contours)
        return targetOut

    def threshold_target(self, origHSV):
        # Circular colour spaces are a pain
        if self.hsvLimits[1] >= self.hsvLimits[0]:
            newHSV = cv2.inRange(origHSV, cv2.cv.Scalar(self.hsvLimits[0], self.hsvLimits[2], 0),
                                 cv2.cv.Scalar(self.hsvLimits[1], self.hsvLimits[3], 255))
        else:
            newHSV = cv2.inRange(origHSV, cv2.cv.Scalar(self.hsvLimits[1], self.hsvLimits[2], 0),
                                 cv2.cv.Scalar(255, self.hsvLimits[3], 255)) & cv2.inRange(origHSV, cv2.cv.Scalar(0,
                                                                                                                  self.hsvLimits[
                                                                                                                      2],
                                                                                                                  0),
                                                                                           cv2.cv.Scalar(
                                                                                               self.hsvLimits[0],
                                                                                               self.hsvLimits[3], 255))
        return newHSV

    def check_contours(self, contours):
        targetOut = np.zeros((2, 1))
        if len(contours):
            # Generate centroids list
            candidates = np.zeros((len(contours), 2))
            for idx in range(len(contours)):
                moments = cv2.moments(contours[0])
                if moments['m00'] == 0:
                    continue
                candidates[idx, 0] = moments['m10'] / moments['m00']
                candidates[idx, 1] = moments['m01'] / moments['m00']
            # Remove failed candidates
            candidates = candidates[candidates > 0].reshape(-1, 2)
            if len(candidates) == 0:
                self.colourLock = False
                return targetOut
            # Find centroid closest to current estimate
            kdTree = spatial.cKDTree(candidates)
            prevEst = self.tracker.x[:2].T
            dist, idxs = kdTree.query(prevEst)
            targetOut = candidates[idxs[0], :2].reshape((2, 1))
            self.colourLock = True
            self.colourTargetRadius = int(np.sqrt(moments['m00'] / np.pi)) + 5
        else:
            self.colourLock = False
        return targetOut

    def new_flow_obs(self, image, colourObs):
        targetOut = np.zeros((2, 1))
        greyIm = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        newTracks = []
        # Run tracker if points present, used from OpenCV example
        if len(self.flowTracks) > 0:
            img0, img1 = self.prevIm, greyIm
            p0 = np.float32([tr[-1] for tr in self.flowTracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.flowParams)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.flowParams)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1
            for tr, (x, y), good_flag in zip(self.flowTracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.flowTrackHistoryMax:
                    del tr[0]
                newTracks.append(tr)
            self.flowTracks = newTracks
        else:
            mask = np.zeros_like(greyIm)
            mask[:] = 255
            if self.colourLock:  # Base search on new lock
                cv2.circle(mask, (colourObs[0], colourObs[1]), self.colourTargetRadius, 0, -1)
            else:  # Base search on current estimate
                cv2.circle(mask, (self.tracker.x[0], self.tracker.x[1]), self.colourTargetRadius, 0, -1)
            p = cv2.goodFeaturesToTrack(greyIm, mask=mask, **self.featureParams)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.flowTracks.append([(x, y)])
            self.flowLock = False
        self.prevIm = greyIm
        if len(newTracks):
            cent, rad = cv2.minEnclosingCircle(np.float32([tr[-1] for tr in newTracks]).reshape(-1, 1, 2))
            self.flowLock = True
            targetOut[0] = cent[0]
            targetOut[1] = cent[1]
        return targetOut
