import cv2
import numpy as np
import threading
import time
from itertools import combinations
from sys import exit
from camOps import CamHandler
from networking import MasterNode, SlaveNode
from calibration import Calibration
from localTracking import LocalModeOne
from HSConfig import camParams, netParams, trackingParams, masterConfigFiles as mCF, slaveConfigFiles as sCF
import shelve

__author__ = 'Michael Holmes'


# ----------------------------------------------------------------
#
# Top level class for the system handling camera, network and
# calibration initialisation
# Uses networking.py for data transfer and collation
# Uses calibration.py for rectification of data to global frame
# Uses localTracking.py for local frame tracking
# Uses camOps.py for camera handling
#
# TODO: Integrate matplotlib 3D plotter
# TODO: Test n>=4 nodes
# ----------------------------------------------------------------


class Reconstruction(threading.Thread):
    def __init__(self, nodeType):
        # Initialise own thread
        threading.Thread.__init__(self)

        # Thread params
        self._killThread = False
        self.setDaemon(True)

        # Reconstruction Params
        self.name = nodeType
        self.posList = []

        # Initialise local camera thread
        print 'Initialising camera link.'
        self._cam = CamHandler(camParams['camIndex'])
        print 'Camera link up.'

        # Initialise Network
        print 'Initialising network links.'
        if nodeType == 'Master':
            self._net = MasterNode(**netParams)
        elif nodeType == 'Slave':
            self._net = SlaveNode(**netParams)
        else:
            print "Nodetype not designated 'Master' or 'Slave'. Cannot continue."
            exit()
        ret = self._net.netEventDict['calibSync'].wait(netParams['waitTimeout'])
        self._net.netEventDict['calibSync'].clear()
        if ret:
            print 'Network link up.'
        else:
            print 'Network wait timeout. Cannot continue.'
            exit()

        # Initialise Calibration
        print 'Initialising calibration parameters'
        if nodeType == 'Master':
            self._calib = Calibration(self._cam, self._net, **mCF)
        else:
            self._calib = Calibration(self._cam, self._net, **sCF)
        print 'Finished calibration initialisation.'

        # Initialise Tracking
        print 'Initialising local target tracking.'
        self._trackers = []
        for idx in range(trackingParams['numTargets']):
            self._trackers.append(LocalModeOne(self._cam, idx))
        print 'Finished local tracking initialisation.'

        # Start thread
        self.start()
        return

    def run(self):
        if self._net.name == 'Master':  # Master loop
            while not self._killThread:
                # If capture flag load tracking data from slaves
                if self._net.netEventDict['capture'].isSet():
                    self._master_append_local_tracker_data()
                    self._master_append_slave_tracker_data()
                # If recon flag reconstruct from collected tracking data
                if self._net.netEventDict['reconstruct'].isSet():
                    compiledData = self._master_compile_data()
                    savedData = shelve.open('reconData')
                    savedData['reconData'] = compiledData
                    savedData.close()
                    self.toggle_reconstruct()
        else:  # Slave loop
            while not self._killThread:
                # If capture flag start transmitting positions
                if self._net.netEventDict['capture'].isSet():
                    self._slave_send_tracker_data()
                elif self._net.netEventDict['reconSync'].isSet():
                    self._net.send_comms('Success')
                    self._net.netEventDict['reconSync'].clear()

        return

    # TODO: PLACE FOR RECONSTRUCT

    def _master_compile_data(self):
        # Collect remaining slave data in queue
        # Wait for slave data to arrive
        startTime = time.time()
        currTime = startTime
        finishedSlaves = []
        print 'Waiting for slaves to finish local data capture and transmit.'
        while currTime - startTime < self._net.waitTimeout and len(finishedSlaves) != self._net.num_connected():
            slavePacket = self._net.get_slave_data()
            if slavePacket is not None and slavePacket[1] == self._net.commMsgPrefixes['Position']:
                addr, prefix, data = slavePacket
                self.posList.append((addr[0], data))
            elif slavePacket is not None and slavePacket[1] == self._net.commMsgPrefixes['Success'] and slavePacket[0] \
                    not in finishedSlaves:
                finishedSlaves.append(slavePacket[0])
            currTime = time.time()
        # For each element in the position list:
        compiledData = []
        idx = 0
        for addr, packet in self.posList:
            targNum, capTime, xPos, yPos, localWeight = self._depackage_pos_data(packet)
            # Correct time wrt master
            if addr != 'localhost':
                capTime -= self._net.slaveSyncTimes[addr]
            lineVect = np.zeros((4, 2))
            lineVect[0, 1] = xPos
            lineVect[1, 1] = yPos
            lineVect[2, 1] = 1
            lineVect[3, 0] = 1
            lineVect[3, 1] = 1
            if addr != 'localhost':  # correct position to global frame vector if not master node observation
                lineVect = np.dot(self._calib.extrinTransforms[addr][0], lineVect)
                point1 = lineVect[:3, 0]
                point2 = lineVect[:3, 1]
                newVect = (point2 - point1).reshape((3, 1))
                lineVect = np.hstack((point1.reshape((3, 1)), newVect))
                # Get corresponding extrinsic weighting
                extrinWeight = self._calib.extrinTransforms[addr][1]
                compiledData.append((idx, targNum, addr, capTime, lineVect, localWeight, extrinWeight))
            else:
                compiledData.append((idx, targNum, addr, capTime, lineVect[:3, :], localWeight, 1))
            idx += 1
        return compiledData

    def _get_local_tracker_data(self):
        # Get each position from local trackers
        packets = []
        for tracker in self._trackers:
            packet = tracker.get_pos_data()
            if packet is not None:
                packets.append(packet)
        # For each target-locked tracker
        stringPackets = []
        for packet in packets:
            # Correct for camera intrinsic parameters
            pos = np.array([packet[2], packet[3]]).reshape((1, 1, 2))
            pos = cv2.undistortPoints(pos, self._calib.camMatrix, self._calib.distCoefs)
            pos = pos.reshape((2,))
            # Package packet and return
            stringPackets.append(self._package_pos_data(packet[0], packet[1], pos[0], pos[1], packet[4]))
        if len(stringPackets):
            return stringPackets
        else:
            return None

    def _master_append_slave_tracker_data(self):
        slavePacket = self._net.get_slave_data()
        if slavePacket is not None and slavePacket[1] == self._net.commMsgPrefixes['Position']:
            addr, prefix, data = slavePacket
            self.posList.append((addr[0], data))

    def _master_append_local_tracker_data(self):
        localPackets = self._get_local_tracker_data()
        if localPackets is not None:
            for packet in localPackets:
                self.posList.append(('localhost', packet))
            return True
        else:
            return False

    def _slave_send_tracker_data(self):
        stringPackets = self._get_local_tracker_data()
        if stringPackets is not None:
            for packet in stringPackets:
                self._net.send_comms('Position', packet)
            return True
        else:
            return False

    def _package_pos_data(self, targNumber, timeStamp, xPos, yPos, weight):
        packet = '%d:%s:%s:%s:%s' % (targNumber, repr(timeStamp), repr(xPos), repr(yPos), repr(weight))
        return packet

    def _depackage_pos_data(self, packet):
        splitPack = packet.split(':')
        targNumber = int(splitPack[0])
        timeStamp = float(splitPack[1])
        xPos = float(splitPack[2])
        yPos = float(splitPack[3])
        weight = float(splitPack[4])
        return targNumber, timeStamp, xPos, yPos, weight

    def shutdown(self):
        # Shutdown local trackers
        for tracker in self._trackers:
            tracker.stop()
            while tracker.isAlive():
                pass
        # Shutdown camera feed
        self._cam.stop()
        while self._cam.isAlive():
            pass
        # Shutdown network
        self._net.stop()
        while self._net.isAlive():
            pass
        # Kill any remaining OpenCV windows
        cv2.destroyAllWindows()
        return True

    # Toggle capture mode on or off if master, slaves will follow upon comm send
    def toggle_capture(self):
        if self._net.name == 'Slave':
            return False
        if self._net.netEventDict['capture'].isSet():
            print 'Capture off.'
            self._net.netEventDict['capture'].clear()
        else:
            print 'Capture on.'
            self._net.netEventDict['capture'].set()
        self._net.send_comms_all('Capture')
        return True

    # Toggle reconstruction mode on or off if master, slaves will follow upon comm send
    def toggle_reconstruct(self):
        if self._net.name == 'Slave':
            return False
        if self._net.netEventDict['reconstruct'].isSet():
            print 'Finishing reconstruction.'
            self._net.netEventDict['reconstruct'].clear()
        else:
            print 'Starting reconstruction.'
            self._net.netEventDict['reconstruct'].set()
        self._net.send_comms_all('Reconstruct')
        return True

    def purge_capture_data(self):
        if self._net.name == 'Slave':
            return False
        if len(self.posList):
            self.posList = []
            print 'Purged capture data.'
            return True
        else:
            print 'No capture data to purge.'
            return False


# Reconstruct 3D path of targets using compiled 3D vector data and nominated time window 'binSize'
def master_reconstruct(compiledData, binSize):
    # Isolate timing data
    binSizeCopy = binSize
    timingData = []
    for idx, targNum, addr, capTime, lineVect, localWeight, extrinWeight in compiledData:
        timingData.append((idx, targNum, addr, capTime))
    # Normalise times
    timingData = np.array(timingData, dtype=[('idx', 'i8'), ('targNum', 'i8'), ('addr', 'S16'), ('time', 'f8')])
    timingData.sort(order='time')
    sortedTime = timingData['time']
    sortedTime -= sortedTime.min() * np.ones_like(sortedTime)
    binSizeCopy = binSizeCopy / (sortedTime.max() - sortedTime.min())
    sortedTime /= (sortedTime.max() - sortedTime.min()) * np.ones_like(sortedTime)
    # Group according observation window
    obsGroups = []
    idx = 0
    lastGroupMaxIdx = 0
    while idx < len(sortedTime):
        idxOffset = 1
        newObGroup = [timingData['idx'][idx]]
        newObGroupAddr = [timingData['addr'][idx]]
        newObGroupTiming = [sortedTime[idx]]
        while idx + idxOffset < len(sortedTime) and sortedTime[idx + idxOffset] - sortedTime[idx] < binSizeCopy:
            if timingData['targNum'][idx] == timingData['targNum'][idx + idxOffset]:
                newObGroup.append(timingData['idx'][idx + idxOffset])
                newObGroupAddr.append(timingData['addr'][idx + idxOffset])
                newObGroupTiming.append(sortedTime[idx + idxOffset])
            idxOffset += 1
        # If new group is not a subset of a previous group add to observation groups
        newObGroupAddr = set(newObGroupAddr)
        if idx + idxOffset > lastGroupMaxIdx and len(newObGroup) > 1 and len(newObGroupAddr) > 1:
            # Compute timing weightings
            groupTimeMean = np.mean(newObGroupTiming)
            for idx2 in range(len(newObGroupTiming)):
                newObGroupTiming[idx2] = 1.0 - np.fabs(newObGroupTiming[idx2] - groupTimeMean) / binSizeCopy
            # Append to group
            obsGroups.append((newObGroup, newObGroupTiming))
            lastGroupMaxIdx = idx + idxOffset
        idx += 1
    # Collapse node-multiples in groups according to weightings
    reconstructedPoints = []
    for obsGroup in obsGroups:
        addrSets = {}
        for idx, timeWeight in zip(obsGroup[0], obsGroup[1]):
            idxSet = compiledData[idx]
            if idxSet[2] not in addrSets:
                addrSets[idxSet[2]] = [(idxSet, timeWeight)]
            else:
                addrSets[idxSet[2]].append((idxSet, timeWeight))
        # Compute combined weight and collapse if multiples
        refinedObsGroup = []
        for addr, addrSet in addrSets.iteritems():
            # Compute weight score/s from localWeight, extrinWeight, timingWeight and collate vectors/times
            # if more than one
            weightScores = []
            capTimes = []
            vectors = np.zeros((3, 1))
            for idxSet, timeWeight in addrSet:
                weightScores.append(np.average([idxSet[5], idxSet[6], timeWeight], weights=[0.5, 0.3, 0.2]))
                capTimes.append(idxSet[3])
                vectors = np.hstack((vectors, idxSet[4][:, 1].reshape((3, 1))))
            # Collapse lineVect according to weight score and assign weight
            vector = np.average(vectors[:, 1:].reshape((3, -1)), axis=1, weights=weightScores).reshape((3, 1))
            avgCapTime = np.average(capTimes, weights=weightScores)
            lineVect = np.hstack((idxSet[4][:, 0].reshape((3, 1)), vector))
            weighting = np.mean(weightScores)
            refinedObsGroup.append((idxSet[1], addr, avgCapTime, lineVect, weighting))
        # For each observation group, compute a position/error estimate and time
        groupEstimates = []
        for idx1, idx2 in combinations(range(len(refinedObsGroup)), 2):
            line1 = refinedObsGroup[idx1][3]
            line2 = refinedObsGroup[idx2][3]
            weighting1 = refinedObsGroup[idx1][4]
            weighting2 = refinedObsGroup[idx2][4]
            capTime1 = refinedObsGroup[idx1][2]
            capTime2 = refinedObsGroup[idx2][2]
            pairPosEstimate = point_between_skew_lines(line1, line2, weighting1, weighting2)
            if pairPosEstimate is not None:
                avgCapTime = np.average([capTime1, capTime2], weights=[weighting1, weighting2])
                avgWeight = np.mean([weighting1, weighting2])
                groupEstimates.append((pairPosEstimate, avgCapTime, avgWeight))
        # Collapse multiple estimates per group
        positions = np.zeros((3, 1))
        capTimes = []
        weights = []
        for estimate in groupEstimates:
            positions = np.hstack((positions, estimate[0]))
            capTimes.append(estimate[1])
            weights.append(estimate[2])
        avgPosition = np.average(positions[:, 1:].reshape((3, -1)), axis=1, weights=weights)
        avgTime = np.average(capTimes, weights=weights)
        # Append new observation position
        reconstructedPoints.append((avgPosition, avgTime))

    return reconstructedPoints


def point_between_skew_lines(line1, line2, weighting1=None, weighting2=None):
    # Initialise components
    point1 = line1[:, 0].astype(np.float64)
    point2 = line2[:, 0].astype(np.float64)
    vect1 = line1[:, 1].astype(np.float64)
    vect2 = line2[:, 1].astype(np.float64)
    w0 = point1 - point2
    a = np.dot(vect1, vect1)
    b = np.dot(vect1, vect2)
    c = np.dot(vect2, vect2)
    d = np.dot(vect1, w0)
    e = np.dot(vect2, w0)

    # Check for parallel lines
    if (a * c - b ** 2) == 0:
        return None

    # Calculate parameter values s,t along vectors
    t = (b * e - c * d) / (a * c - b ** 2)
    s = (a * e - b * d) / (a * c - b ** 2)

    # Calculate bounding shortest distance line
    close1 = point1 + t * vect1
    close2 = point2 + s * vect2

    # If no weighting is designated, take the midpoint
    if weighting1 is None or weighting2 is None:
        weighting = 0.5
    else:
        weighting = 1.0 - weighting1 / (weighting1 + weighting2)

    return (close1 + weighting * (close2 - close1)).reshape((3, 1))


if __name__ == '__main__':
    # Test script for the reconstruction module
    savedData = shelve.open('reconData')
    compiledData = savedData['reconData']
    savedData.close()
    reconPoints = master_reconstruct(compiledData, 0.1)
    test = 1
    pass
