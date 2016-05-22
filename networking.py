import threading
import Queue
import socket
import select
import sys
import time

# from HSConfig import netParams

netParams = dict(nodeType='Master',
                 numSlaves=1,
                 port=42680,
                 bcAddr='192.168.1.255')

__author__ = 'Michael Holmes'

if netParams['nodeType'] == 'Master':
    # Global Queues for passing Time and Position Info
    timeQueue = Queue.Queue()
    positionQueue = Queue.Queue()


# Class to hold various protocol strings, identifiers, format etc.
class NetProto:
    def __init__(self):
        self._sockTimeout = 0.01
        self._msgTimeout = 10.0
        self._bcReattempts = 3
        self._bcDelay = self._msgTimeout / self._bcReattempts
        self._recBuffSize = 1024
        self._commMsgPrefixes = {'Calibrate': 'C', 'Success': 'S', 'Position': 'P', 'Time': 'T'}
        self._broadMsgPrefixes = {'Connect': 'C', 'Disconnect': 'D', 'Time': 'T', 'ForceTime': 'F'}


# Base class for Master and Slave Nodes/ Worker Threads, handles low level socket actions
class NetBase(threading.Thread, NetProto):
    def __init__(self, nodeType, port, bcAddr):
        threading.Thread.__init__(self)
        NetProto.__init__(self)
        self._nodeType = nodeType
        self._port = port
        self._commReadable = False
        self._commWriteable = False
        self._broadReadable = False
        self._broadWriteable = False
        self._hostname = socket.getfqdn()
        self._hostAddr = socket.gethostbyname(self._hostname)
        self._bcAddr = bcAddr
        self._killThread = False
        self._connect_socks()
        return

    def _connect_socks(self):
        try:
            self._commSock = socket.socket()
            self._broadSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except (KeyboardInterrupt, SystemExit):
            raise
        except socket.error, msg:
            print 'Failed to create socket. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
            self._killThread = True
            sys.exit()
        self._commSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._broadSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self._nodeType == 'Master':
            self._broadSock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            self._commSock.bind(('', self._port))
            if self._nodeType == 'Slave':
                self._broadSock.bind(('', self._port))
        except (KeyboardInterrupt, SystemExit):
            raise
        except socket.error, msg:
            print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
            self._killThread = True
            sys.exit()
        return

    def stop(self):
        self._killThread = True
        return

    def _close_socks(self):
        self._commSock.close()
        if self._nodeType == 'Slave':
            self._broadSock.close()
        return

    def _check_comm_sock(self):
        readableSocks, writeableSocks, errorSocks = select.select([self._commSock],
                                                                  [self._commSock], [],
                                                                  self._sockTimeout)
        if self._commSock in readableSocks:
            self._commReadable = True
        else:
            self._commReadable = False
        if self._commSock in writeableSocks:
            self._commWriteable = True
        else:
            self._commWriteable = False
        return

    def _check_broad_sock(self):
        readableSocks, writeableSocks, errorSocks = select.select([self._broadSock],
                                                                  [self._broadSock], [],
                                                                  self._sockTimeout)
        if self._broadSock in readableSocks:
            self._broadReadable = True
        else:
            self._broadReadable = False
        if self._broadSock in writeableSocks:
            self._broadWriteable = True
        else:
            self._broadWriteable = False
        return

    def _send_data(self, prefixKey, data=None):
        totalSent = 0
        if data is None:
            msgLen = 4
            msg = self._commMsgPrefixes[prefixKey] + '%03d' % msgLen
        else:
            msgLen = 4 + len(data)
            msg = self._commMsgPrefixes[prefixKey] + '%03d' % msgLen + data
        while totalSent < msgLen:
            sent = self._commSock.send(msg[totalSent:])
            if sent == 0:
                raise RuntimeError("Communications connection broken")
            totalSent += sent
        return

    def _rec_data(self):
        chunks = []
        bytesRec = 0
        while bytesRec < 4:
            chunk = self._commSock.recv(self._recBuffSize)
            if chunk == '':
                raise RuntimeError("Communications connection broken")
            chunks.append(chunk)
            bytesRec += len(chunk)
        msgPrefix = ''.join(chunks)
        numBytes = int(msgPrefix[1:4])
        if len(msgPrefix) > 4:
            chunks = list(msgPrefix[4:])
            msgPrefix = msgPrefix[:4]
            bytesRec = len(chunks)
        else:
            chunks = []
            bytesRec = 0
        while bytesRec < numBytes - 4:
            chunk = self._commSock.recv(min((numBytes - 4) - bytesRec, self._recBuffSize))
            if chunk == '':
                raise RuntimeError("Communications connection broken")
            chunks.append(chunk)
            bytesRec += len(chunk)
        return msgPrefix + ''.join(chunks)

    # Broadcast master node address to local network
    def _send_broadcast(self, prefixKey):
        totalSent = 0
        msg = self._broadMsgPrefixes[prefixKey] + '%03d' % 4
        while totalSent < 4:
            sent = self._broadSock.sendto(msg[totalSent:], (self._bcAddr, self._port))
            if sent == 0:
                raise RuntimeError("Broadcast connection broken")
            totalSent += sent
        return

    def _rec_broadcast(self):
        chunks = []
        bytesRec = 0
        addr = None
        while bytesRec < 4:
            chunk, addr = self._broadSock.recvfrom(self._recBuffSize)
            if chunk == '':
                raise RuntimeError("Broadcast connection broken")
            chunks.append(chunk)
            bytesRec += len(chunk)
        msgPrefix = ''.join(chunks)
        numBytes = int(msgPrefix[1:4])  # +1 ?
        if len(msgPrefix) > 4:
            chunks = list(msgPrefix[4:])
            msgPrefix = msgPrefix[:4]
            bytesRec = len(chunks)
        else:
            chunks = []
            bytesRec = 0
        while bytesRec < numBytes - 4:
            chunk = self._broadSock.recv(min((numBytes - 4) - bytesRec, self._recBuffSize))
            if chunk == '':
                raise RuntimeError("Broadcast connection broken")
            chunks.append(chunk)
            bytesRec += len(chunk)
        return msgPrefix + ''.join(chunks), addr

    def _flush_socket(self, socketType):

        if socketType == 'comm':
            # Flush comm tcp socket
            self._check_comm_sock()
            while self._commReadable and not self._killThread:
                self._commSock.recv(self._recBuffSize)
                self._check_comm_sock()
        else:
            # Flush broad udp socket
            self._check_broad_sock()
            while self._broadReadable and not self._killThread:
                self._broadSock.recv(self._recBuffSize)
                self._check_broad_sock()
        return


# Class for Master Server Node
class MasterNode(NetBase):
    def __init__(self, nodeType, numSlaves, port, bcAddr):
        global timeQueue, positionQueue
        NetBase.__init__(self, nodeType, port, bcAddr)
        print 'Initialising in Master mode.'
        self._clients = {}
        self._threadReg = {}
        self._name = nodeType
        self._threadID = 0
        self._numSlaves = numSlaves
        self._timeQueue = timeQueue
        self._positionQueue = positionQueue
        self._timeSync = False
        self._timeOffsets = None
        self.setDaemon(True)
        return

    def run(self):
        # Flush sockets
        self._flush_socket('broad')
        # Connect to client nodes
        self._connect_to_slaves()
        # If any nodes, synchronise time
        if self.num_connected() == self._numSlaves:
            self.start_time_sync()
            while not self._killThread:
                if self._timeSync:
                    self._timeSync = False
                    self._timeOffsets = self._sync_time()
            self._disconnect_from_slaves()
        return

    def _connect_to_slaves(self):
        lastTime = time.time()
        currTime = lastTime
        castTime = currTime - 1
        self._commSock.listen(5)
        while not self._killThread \
                and currTime - lastTime < self._msgTimeout \
                and self.num_connected() != self._numSlaves:
            if currTime - castTime > self._bcDelay:
                print 'Broadcasting connection request.'
                self._send_broadcast('Connect')
                self._check_comm_sock()
                castTime = time.time()
            if self._commReadable:
                # If there is a new inbound connection, answer it
                conn, addr = self._commSock.accept()
                # Check if addr matches a current thread and delete it, otherwise create new thread
                if addr in self._clients:
                    self._clients[addr].stop()
                    while self._clients[addr].isAlive():
                        pass
                    del self._clients[addr]
                self._clients[addr] = MasterThread(self._threadID, conn, addr, self._timeQueue, self._positionQueue)
                self._clients[addr].start()
                self._threadReg[self._threadID] = addr
                print 'New connection: %s' % addr[0]
                self._threadID += 1
            currTime = time.time()
        return

    def _disconnect_from_slaves(self):
        # Kill remaining worker threads then exit
        self._send_broadcast('Disconnect')
        self._close_socks()
        for addr in self._clients.keys():
            self._clients[addr].stop()
            while self._clients[addr].isAlive():
                pass
        return

    # Returns integer number of connected slave nodes
    def num_connected(self):
        return len(self._clients)

    # Set flag for time synchronisation to slaves
    def start_time_sync(self):
        self._timeSync = True
        return

    def _sync_time(self, forced=False):
        ret = False
        failCount = 0
        slaveTimes = {}
        while not ret and failCount < self._bcReattempts:
            sendTime = time.time()
            if forced and failCount == 0:
                self._send_broadcast('ForceTime')
            else:
                self._send_broadcast('Time')
            print 'Broadcasting time sync request.'
            currTime = sendTime
            # Wait for responses from slave nodes
            while currTime - sendTime < self._msgTimeout and len(slaveTimes) != len(self._clients):
                while not self._timeQueue.empty():
                    threadID, slaveTime = self._timeQueue.get()
                    addr, port = self._threadReg[threadID]
                    slaveTimes[addr] = slaveTime - sendTime
                    self._timeQueue.task_done()
                currTime = time.time()
            failCount += 1
            ret, failedAddr = self._check_time_returns(slaveTimes)
        if ret:
            print 'Synchronised time with all nodes.'
            return slaveTimes
        elif failCount == self._bcReattempts:
            assert 'Unable to time synchronise all nodes after max attempts.'
        else:
            assert 'Failed to synchronise node times.'

    def _check_time_returns(self, slaveTimes):
        if len(slaveTimes) != len(self._clients):
            print 'Error: Time synchronisation failure, completed %d/%d nodes.' % (len(slaveTimes), len(self._clients))
            clientsCopy = self._clients.copy()
            for clientAddr in slaveTimes.keys():
                if clientAddr in clientsCopy:
                    del clientsCopy[clientAddr]
            for clientAddr in clientsCopy.keys():
                print 'Error: Did not receive time sync for client address: %s:%d' % (clientAddr[0], clientAddr[1])
            return False, clientsCopy.keys()
        else:
            return True, None

# Worker thread for Master Server Node
class MasterThread(NetBase):
    def __init__(self, threadID, conn, addr, timeQueue, positionQueue):
        threading.Thread.__init__(self)
        NetProto.__init__(self)
        self._killThread = False
        self._threadID = threadID
        self._commSock = conn
        self._nodeType = 'Master'
        self._name = addr
        self._timeQueue = timeQueue
        self._positionQueue = positionQueue
        return

    def run(self):
        global exitEvent
        # Send connection confirmation to slave node
        self._send_data('Success')
        while not self._killThread:
            self._check_comm_sock()
            if self._commReadable:
                dataIn = self._rec_data()
                if dataIn[0] == self._commMsgPrefixes['Time']:
                    msgLen = int(dataIn[1:4])
                    self._timeQueue.put((self._threadID, float(dataIn[4:4 + msgLen])))
        return


# Class for Slave Client Node
class SlaveNode(NetBase):
    def __init__(self, nodeType, numSlaves, port, bcAddr):
        NetBase.__init__(self, nodeType, port, bcAddr)
        print 'Initialising in Slave mode.'
        self._name = nodeType
        self._masterAddr = None
        self._slaved = False
        self._timeSynced = False
        self.setDaemon(True)
        return

    def run(self):
        self._flush_socket('broad')
        self._sync_to_master()
        while not self._killThread:
            self._check_comm_sock()
            if self._commReadable:
                print self._rec_data()
            self._check_broad_sock()
            if self._broadReadable:
                dataIn, inAddr = self._rec_broadcast()
                if dataIn[0] == self._broadMsgPrefixes['Time']:
                    print 'Received time sync request.'
                    if not self._timeSynced:
                        currTime = repr(time.time())
                        self._send_data('Time', currTime)
                        print 'Sending current time: %s' % currTime
                        self._timeSynced = True
                    else:
                        print 'Ignoring time sync request: already synced.'
                elif dataIn[0] == self._broadMsgPrefixes['ForceTime']:
                    print 'Received forced time sync request.'
                    currTime = repr(time.time())
                    self._send_data('Time', currTime)
                    print 'Sending current time: %s' % currTime
                    self._timeSynced = True
                elif dataIn[0] == self._broadMsgPrefixes['Disconnect']:
                    print 'Received disconnect request.'
                    self._killThread = True
        return

    def _sync_to_master(self):
        print 'Waiting for master broadcast.'
        while not self._slaved and not self._killThread:
            self._check_broad_sock()
            if self._broadReadable:
                broadIn, addr = self._rec_broadcast()
                if broadIn[0] == self._broadMsgPrefixes['Connect']:
                    self._masterAddr = addr[0]
                    print 'Received master connection request.'
                    self._connect_to_master()
        return

    def _connect_to_master(self):
        connected = False
        print 'Connecting to master...'
        self._commSock.connect((self._masterAddr, self._port))
        while not connected and not self._killThread:
            self._check_comm_sock()
            if self._commReadable:
                dataIn = self._rec_data()
                if dataIn[0] == self._commMsgPrefixes['Success']:
                    connected = True
        print 'Connected'
        self._slaved = True
        return

    def _disconnect_from_master(self):
        self._close_socks()
        print 'Disconnected from master.'
        return


def test_script():
    if netParams['nodeType'] == 'Master':
        myNode = MasterNode(**netParams)
    elif netParams['nodeType'] == 'Slave':
        myNode = SlaveNode(**netParams)
    else:
        myNode = SlaveNode(**netParams)
    myNode.start()
    lastTime = time.time()
    startTime = lastTime
    while myNode.isAlive():
        currTime = time.time()
        if currTime - lastTime > 10 and myNode._nodeType == 'Master':
            lastTime = currTime
            print 'Clients Connected: %d/%d' % (myNode.num_connected(), myNode._numSlaves)
        if currTime - startTime > 240:
            myNode.stop()
    return 0


if __name__ == '__main__':
    test_script()
