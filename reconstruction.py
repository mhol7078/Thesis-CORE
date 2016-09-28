import cv2
import numpy as np
import threading
import time
from os import path
from itertools import combinations
from sys import exit
from camOps import CamHandler
from networking import MasterNode, SlaveNode
from calibration import Calibration
from localTracking import LocalModeOne
from HSConfig import camParams, netParams, trackingParams, masterConfigFiles as mCF, slaveConfigFiles as sCF, mainParams
# from visualise import Visualiser
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
# TODO: Test n>=4 nodes
# ----------------------------------------------------------------


class Reconstruction(threading.Thread):
    def __init__(self, nodeType, noNetwork=False):
        # Initialise own thread
        threading.Thread.__init__(self)

        # Thread params
        self._killThread = False
        self.setDaemon(True)

        # Reconstruction Params
        self.name = nodeType
        self.posList = []
        self.rayMax = mainParams['maxRayDist']  # Maximum viewing range for each camera in mm

        # Run no-network calibration load
        if noNetwork and nodeType == 'Master':
            self._calib = Calibration(None, None, noNetwork=True, **mCF)
            return

        # Initialise local camera thread
        print 'Initialising camera link.'
        self._cam = CamHandler(camParams['camIndex'])
        print 'Camera link up.'

        # Run intrinsic calibration only
        if mainParams['intrinCalibOnly']:
            print 'Running forced intrinsic calibration.'
            self._calib = Calibration(self._cam, None, intrinOnly=True)
            # Close camera
            self._cam.stop()
            return

        # Otherwise normal operation

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
                    posData = self._master_reconstruct(compiledData, 0.1)
                    self._write_recon_to_file(posData)
                    self.toggle_reconstruct()
        else:  # Slave loop
            while not self._killThread:
                # Check see if a shutdown has been ordered by Master
                if not self._net.isAlive():
                    self.shutdown()
                # If capture flag start transmitting positions
                if self._net.netEventDict['capture'].isSet():
                    self._slave_send_tracker_data()
                elif self._net.netEventDict['reconSync'].isSet():
                    self._net.send_comms('Success')
                    self._net.netEventDict['reconSync'].clear()
        return

    def collect_estimates(self):
        currFrame = self._cam.get_frame()[0]
        trackerEstimates = []
        for tracker in self._trackers:
            trackerEstimates.append(tracker.get_current_estimate())
        for estimate, lockStatus in trackerEstimates:
            if lockStatus:
                cv2.circle(currFrame, (estimate[0], estimate[1]), 20, (0, 255, 0), 3)
            else:
                cv2.circle(currFrame, (estimate[0], estimate[1]), 20, (0, 0, 255), 3)
        return currFrame

    def _write_recon_to_file(self, posData, filename=None, append=False):
        # Use default if no name given
        if filename is None:
            filename = 'reconstruction.csv'
        # Open or append unique file for processing
        if append:
            fd = open(filename, 'a')
        else:
            while path.isfile(filename):
                splitName = filename.split('.')
                scoreIdx = splitName[0][::-1].find('_')
                if scoreIdx != -1:
                    counter = int(splitName[0][::-1].split('_')[0][::-1]) + 1
                    filename = splitName[0][:-(scoreIdx + 1)] + '_' + str(counter) + '.csv'
                else:
                    counter = 1
                    filename = splitName[0] + '_1' + '.csv'
            fd = open(filename, 'w')
        # Dump CSV variables to file
        fd.write(time.ctime() + '\n')
        for targNum, capTime, position in posData:
            writeStr = repr(targNum) + ',' + repr(capTime)
            for idx in range(len(position)):
                writeStr += ',' + repr(position[idx])
            writeStr += '\n'
            fd.write(writeStr)
        fd.close()
        return

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
        # End own thread
        self._killThread = True
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
            print 'Finished reconstruction.'
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
    def _master_reconstruct(self, compiledData, binSize):
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
                pairPosEstimate = self._point_between_skew_rays(line1, line2, weighting1, weighting2)
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
            reconstructedPoints.append((refinedObsGroup[0][0], avgTime, avgPosition))
        return reconstructedPoints

    # Credit Dan Sunday, geomalgorithms.com
    def _point_between_skew_rays(self, line1, line2, weighting1=None, weighting2=None):
        eps = 1e-8
        # Extend ray out to max viewing distance
        line1Point2 = self._extend_ray(line1)
        line2Point2 = self._extend_ray(line2)

        line1Point1 = line1[:, 0].astype(np.float64)
        line2Point1 = line2[:, 0].astype(np.float64)

        # Visualiser
        # segment1 = np.hstack((line1Point1.reshape((3, 1)), line1Point2.reshape((3, 1))))
        # segment2 = np.hstack((line2Point1.reshape((3, 1)), line2Point2.reshape((3, 1))))
        # visualise_observation(segment1, segment2)

        u = line1Point2 - line1Point1
        v = line2Point2 - line2Point1
        w = line1Point1 - line2Point1
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w)
        e = np.dot(v, w)
        D = a * c - b ** 2
        sD = D
        tD = D

        if D < eps:  # Lines are almost parallel
            sN = 0.0
            sD = 1.0
            tN = e
            tD = c
        else:
            sN = b * e - c * d
            tN = a * e - b * d
            if sN < 0.0:
                sN = 0.0
                tN = e
                tD = c
            elif sN > sD:
                sN = sD
                tN = e + b
                tD = c

        if tN < 0.0:
            tN = 0.0
            if -d < 0.0:
                sN = 0.0
            elif -d > a:
                sN = sD
            else:
                sN = -d
                sD = a
        elif tN > tD:
            tN = tD
            if b - d < 0.0:
                sN = 0.0
            elif b - d > a:
                sN = sD
            else:
                sN = b - d
                sD = a

        # sc is line1 vector scalar, tc is line2
        sc = 0.0 if np.abs(sN) < eps else sN / sD
        tc = 0.0 if np.abs(tN) < eps else tN / tD

        # Closest points on each line
        close1 = line1Point1 + sc * u
        close2 = line2Point1 + tc * v

        if weighting1 is None or weighting2 is None:
            weighting = 0.5
        else:
            weighting = 1.0 - weighting1 / (weighting1 + weighting2)

        return (close1 + weighting * (close2 - close1)).reshape((3, 1))

    def _extend_ray(self, line):
        pointLine = line[:, 0].astype(np.float64)
        vectLine = line[:, 1].astype(np.float64)
        sI = self.rayMax / np.linalg.norm(vectLine)
        return pointLine + sI * vectLine


if __name__ == '__main__':
    # Test script for the reconstruction module
    # savedData = shelve.open('compiledData')
    # compiledData = savedData['compiledData']
    # savedData.close()
    #
    #
    # # # visualHdl = Visualiser()
    reconHdl = Reconstruction('Master', noNetwork=True)
    # posData = reconHdl._master_reconstruct(compiledData, 0.1)

    pass
