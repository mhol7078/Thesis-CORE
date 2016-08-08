import threading
import Queue
import socket
import select
import sys
import time

__author__ = 'Michael Holmes'

# ---------------------------------------------------------------------
#               GLOBALS
# ---------------------------------------------------------------------
netEventDict = {'calibSync': threading.Event(), 'recDataSync': threading.Event()}


# Class to hold various protocol strings, identifiers, format etc.
class NetProto:
    def __init__(self):
        self._sockTimeout = 0.01
        self._recBuffSize = 1024
        self.commMsgPrefixes = {'Calibrate': 'C', 'Success': 'S', 'Position': 'P', 'Time': 'T', 'ForceTime': 'F',
                                'NoCalibrate': 'N'}
        self.broadMsgPrefixes = {'Connect': 'C', 'Disconnect': 'D', 'Shutdown': 'S'}


# Base class for Master and Slave Nodes/ Worker Threads, handles low level socket actions
class NetBase(threading.Thread, NetProto):
    def __init__(self, nodeType, port, bcAddr, bcReattempts, msgTimeout, waitTimeout):
        global netEventDict
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
        self.msgTimeout = msgTimeout
        self.waitTimeout = waitTimeout
        self._bcReattempts = bcReattempts
        self._killThread = False
        self._socksOpen = False
        self._recOverflow = []
        self._connect_socks()
        self.netEventDict = netEventDict
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
        self._socksOpen = True
        return

    def stop(self):
        self._killThread = True
        return

    def _close_socks(self):
        self._socksOpen = False
        self._commSock.close()
        if self._nodeType == 'Slave':
            self._broadSock.close()
        return

    def _check_socks(self, socketType):
        if socketType == 'comm':
            readableSocks, writeableSocks, errorSocks = select.select([self._commSock],
                                                                      [self._commSock], [],
                                                                      self._sockTimeout)
        elif socketType == 'broad':
            readableSocks, writeableSocks, errorSocks = select.select([self._broadSock],
                                                                      [self._broadSock], [],
                                                                  self._sockTimeout)
        else:
            readableSocks, writeableSocks, errorSocks = select.select([self._commSock, self._broadSock],
                                                                      [self._commSock, self._broadSock], [],
                                                                      self._sockTimeout)
        if self._commSock in readableSocks:
            self._commReadable = True
        else:
            self._commReadable = False
        if self._commSock in writeableSocks:
            self._commWriteable = True
        else:
            self._commWriteable = False
        if self._nodeType == 'Slave':
            if self._broadSock in readableSocks:
                self._broadReadable = True
            else:
                self._broadReadable = False
            if self._broadSock in writeableSocks:
                self._broadWriteable = True
            else:
                self._broadWriteable = False
        return 0

    def _send_data(self, socketType, prefixKey, data=None):
        totalSent = 0
        if data is None:
            msgLen = 4
            if socketType == 'comm':
                msg = self.commMsgPrefixes[prefixKey] + '%03d' % msgLen
            else:
                msg = self.broadMsgPrefixes[prefixKey] + '%03d' % msgLen
        else:
            msgLen = 4 + len(data)
            if socketType == 'comm':
                msg = self.commMsgPrefixes[prefixKey] + '%03d' % msgLen + data
            else:
                msg = self.broadMsgPrefixes[prefixKey] + '%03d' % msgLen + data
        while totalSent < msgLen:
            if socketType == 'comm':
                sent = self._commSock.send(msg[totalSent:])
            else:
                sent = self._broadSock.sendto(msg[totalSent:], (self._bcAddr, self._port))
            if sent == 0:
                print 'Communications connection broken.'
                self._close_socks()
                self._socksOpen = False
                return 1
            totalSent += sent
        return 0

    def _rec_data(self, socketType):
        # Make sure sockets are still connected
        if not self._socksOpen:
            return None
        if len(self._recOverflow):
            chunks = self._recOverflow
            bytesRec = len(chunks)
        else:
            chunks = []
            bytesRec = 0
        while bytesRec < 4:
            if socketType == 'comm':
                chunk = self._commSock.recv(4)
            else:
                chunk, addr = self._broadSock.recvfrom(4)
            if chunk == '':
                print 'Communications connection broken.'
                self._close_socks()
                self._socksOpen = False
                return None
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
            if socketType == 'comm':
                chunk = self._commSock.recv(numBytes - 4 - bytesRec)
            else:
                chunk = self._broadSock.recv(numBytes - 4 - bytesRec)
            if chunk == '':
                print 'Communications connection broken.'
                self._close_socks()
                self._socksOpen = False
                return None
            chunks.append(chunk)
            bytesRec += len(chunk)
        if bytesRec > numBytes - 4:
            self._recOverflow = chunks[(numBytes - 4):]
            chunks = chunks[:(numBytes - 4)]
        if socketType == 'comm':
            return msgPrefix + ''.join(chunks)
        else:
            return msgPrefix + ''.join(chunks), addr

    def _flush_socket(self, socketType):
        if socketType == 'comm':
            # Flush comm tcp socket
            self._check_socks('comm')
            while self._commReadable and not self._killThread:
                self._commSock.recv(self._recBuffSize)
                self._check_socks('comm')
        elif socketType == 'broad':
            # Flush broad udp socket
            self._check_socks('broad')
            while self._broadReadable and not self._killThread:
                self._broadSock.recv(self._recBuffSize)
                self._check_socks('broad')
        return 0


# Class for Master Server Node
class MasterNode(NetBase):
    def __init__(self, nodeType, numSlaves, port, bcAddr, bcReattempts, msgTimeout, waitTimeout):
        NetBase.__init__(self, nodeType, port, bcAddr, bcReattempts, msgTimeout, waitTimeout)
        print 'Initialising in Master mode.'
        self._clients = {}
        self._threadReg = {}
        self.name = nodeType
        self._threadID = 0
        self._threadCounter = 1
        self._numSlaves = numSlaves
        self._recQueue = Queue.Queue()
        self._timeSync = False
        self._dataSync = False
        self._newSlaveData = None
        self.slaveSyncTimes = None
        self._timeOffsets = None
        self.setDaemon(True)
        self.start()
        return

    def run(self):
        # Flush sockets
        self._flush_socket('broad')
        # Connect to client nodes
        ret = self._connect_to_slaves()
        if ret:
            self._disconnect_from_slaves()
            return 1
        # If successfully connected, synchronise time (forced)
        ret = self._sync_time(True)
        if ret:
            self._disconnect_from_slaves()
            return 1
        while not self._killThread:
            if self._timeSync:
                self._timeSync = False
                self._timeOffsets = self._sync_time()
            if self._dataSync:
                self._dataSync = False
                if not self._recQueue.empty():
                    self._newSlaveData = self._recQueue.get()
                    self._recQueue.task_done()
                else:
                    self._newSlaveData = None
                self.netEventDict['recDataSync'].set()
        self._disconnect_from_slaves()
        return 0

    # Command main thread to check if slave data has arrived and collect one packet, block until updated.
    def get_slave_data(self):
        self.netEventDict['recDataSync'].clear()
        self._dataSync = True
        self.netEventDict['recDataSync'].wait()
        if self._newSlaveData is not None:
            threadID, prefix, data = self._newSlaveData
            addr = self._threadReg[threadID]
            return addr, prefix, data
        else:
            return None

    def _connect_to_slaves(self):
        self._commSock.listen(5)
        for castCount in range(self._bcReattempts):
            print 'Broadcasting connection request %d/%d.' % (castCount + 1, self._bcReattempts)
            ret = self._send_data('broad', 'Connect')
            if ret:
                print 'Failed to connect to slave nodes. Exiting.'
                return 1
            castTime = time.time()
            currTime = castTime
            while not self._killThread \
                    and currTime - castTime < self.msgTimeout \
                    and self.num_connected() != self._numSlaves:
                self._check_socks('comm')
                if self._commReadable:
                    # If there is a new inbound connection, answer it
                    conn, addr = self._commSock.accept()
                    # Check if addr matches a current thread and delete it, otherwise create new thread
                    if addr in self._clients:
                        self._clients[addr].stop()
                        while self._clients[addr].isAlive():
                            pass
                        del self._clients[addr]
                    self._clients[addr] = MasterThread(self._threadCounter, conn, addr, self._recQueue)
                    self._threadReg[self._threadCounter] = addr
                    print 'New connection: %s' % addr[0]
                    self._threadCounter += 1
                currTime = time.time()
            if self.num_connected() == self._numSlaves:
                break
        if self.num_connected() == self._numSlaves:
            print 'Successfully connected to all client nodes.'
            return 0
        else:
            print 'Failed to connect to all slave nodes. Exiting.'
            return 1

    def _disconnect_from_slaves(self):
        # Kill remaining worker threads then exit
        self._send_data('broad', 'Disconnect')
        self._close_socks()
        for addr in self._clients.keys():
            self._clients[addr].stop()
            while self._clients[addr].isAlive():
                pass
        return 0

    # Returns integer number of connected slave nodes
    def num_connected(self):
        return len(self._clients)

    # Set flag for time synchronisation to slaves
    def start_time_sync(self):
        self._timeSync = True
        return 0

    def send_comms_all(self, prefixKey, data=None):
        for clientAddr in self._clients.keys():
            self._clients[clientAddr].sendQueue.put((prefixKey, data))

    def _sync_time(self, forced=False):
        slaveTimes = {}
        sendTime = time.time()
        if forced:
            print 'Communicating forced time sync request.'
            self.send_comms_all('ForceTime')
        else:
            print 'Communicating time sync request.'
            self.send_comms_all('Time')
        currTime = sendTime
        # Wait for responses from slave nodes
        while currTime - sendTime < self.msgTimeout and len(slaveTimes) != len(self._clients):
            if self._recQueue.empty():
                threadID, prefix, slaveTime = self._recQueue.get()
                addr, port = self._threadReg[threadID]
                slaveTimes[addr] = float(slaveTime) - sendTime
                self._recQueue.task_done()
            currTime = time.time()
        ret = self._check_time_returns(slaveTimes)
        # Flush data queue
        while not self._recQueue.empty():
            self._recQueue.get()
            self._recQueue.task_done()
        if ret:
            print 'Failed to synchronise node times.'
            return 1
        else:
            print 'Synchronised time with all nodes.'
            self.slaveSyncTimes = slaveTimes
            self.netEventDict['calibSync'].set()
            return 0

    def _check_time_returns(self, slaveTimes):
        if len(slaveTimes) != len(self._clients):
            print 'Error: Time synchronisation failure, completed %d/%d nodes.' % (len(slaveTimes), len(self._clients))
            clientsCopy = self._clients.copy()
            for clientAddr in slaveTimes.keys():
                if clientAddr in clientsCopy:
                    del clientsCopy[clientAddr]
            for clientAddr in clientsCopy.keys():
                print 'Error: Did not receive time sync for client address: %s:%d' % (clientAddr[0], clientAddr[1])
            return 1
        else:
            return 0


# Worker thread for Master Server Node
class MasterThread(NetBase):
    def __init__(self, threadID, conn, addr, recQueue):
        threading.Thread.__init__(self)
        NetProto.__init__(self)
        self._killThread = False
        self._threadID = threadID
        self._commSock = conn
        self._socksOpen = True
        self._nodeType = 'Master'
        self._name = addr
        self._recOverflow = []
        self.sendQueue = Queue.Queue()
        self._recQueue = recQueue
        self.start()
        return

    def run(self):
        # Send connection confirmation to slave node
        ret = self._send_data('comm', 'Success')
        if ret:
            return 1
        while not self._killThread:
            # Send data/commands to Slave node from Master if available
            if not self.sendQueue.empty():
                prefixKey, data = self.sendQueue.get()
                ret = self._send_data('comm', prefixKey, data)
                if ret:
                    return 1
                self.sendQueue.task_done()
            # Send data to Master thread from Slaves if available
            self._check_socks('comm')
            if self._commReadable:
                dataIn = self._rec_data('comm')
                if dataIn is None:
                    return 1
                msgLen = int(dataIn[1:4])
                if msgLen == 4:
                    self._recQueue.put((self._threadID, dataIn[0], None))
                else:
                    self._recQueue.put((self._threadID, dataIn[0], dataIn[4:4 + msgLen]))
        return 0


# Class for Slave Client Node
class SlaveNode(NetBase):
    def __init__(self, nodeType, numSlaves, port, bcAddr, bcReattempts, msgTimeout, waitTimeout):
        NetBase.__init__(self, nodeType, port, bcAddr, bcReattempts, msgTimeout, waitTimeout)
        print 'Initialising in Slave mode.'
        self.name = nodeType
        self._masterAddr = None
        self._slaved = False
        self._timeSynced = False
        self.sendQueue = Queue.Queue()
        self.calibFlag = None
        self.setDaemon(True)
        self.start()
        return

    def run(self):
        self._flush_socket('broad')
        ret = self._sync_to_master()
        if ret:
            return 1
        while not self._killThread:
            # Send any queued comms data (usually calibration or position) to Master
            if not self.sendQueue.empty():
                prefixKey, data = self.sendQueue.get()
                ret = self._send_data('comm', prefixKey, data)
                if ret:
                    return 1
                self.sendQueue.task_done()
            # Respond to commands from Master
            self._check_socks('comm')
            if self._commReadable:
                dataIn = self._rec_data('comm')
                if dataIn is None:
                    return 1
                elif dataIn[0] == self.commMsgPrefixes['Time']:
                    print 'Received time sync request.'
                    if not self._timeSynced:
                        currTime = repr(time.time())
                        self._send_data('comm', 'Time', currTime)
                        print 'Sending current time: %s' % currTime
                        self._timeSynced = True
                    else:
                        print 'Ignoring time sync request: already synced.'
                elif dataIn[0] == self.commMsgPrefixes['ForceTime']:
                    print 'Received forced time sync request.'
                    currTime = repr(time.time())
                    self._send_data('comm', 'Time', currTime)
                    print 'Sending current time: %s' % currTime
                    self._timeSynced = True
                    self.netEventDict['calibSync'].set()
                elif dataIn[0] == self.commMsgPrefixes['Calibrate']:
                    self.calibFlag = True
                    self.netEventDict['calibSync'].set()
                    print "Received 'Calibrate' instruction."
                elif dataIn[0] == self.commMsgPrefixes['NoCalibrate']:
                    print "Received 'NoCalibrate' instruction."
                    self.calibFlag = False
                    self.netEventDict['calibSync'].set()
                elif dataIn[0] == self.commMsgPrefixes['Success']:
                    self.netEventDict['calibSync'].set()
            self._check_socks('broad')
            if self._broadReadable:
                dataIn, inAddr = self._rec_data('broad')
                if dataIn is None:
                    return 1
                elif dataIn[0] == self.broadMsgPrefixes['Connect']:
                    print 'Received connection request.'
                    if self._slaved:
                        self._disconnect_from_master()
                    self._sync_to_master()
                elif dataIn[0] == self.broadMsgPrefixes['Disconnect']:
                    print 'Received disconnect request.'
                    self._disconnect_from_master()
                elif dataIn[0] == self.broadMsgPrefixes['Shutdown']:
                    print 'Received shutdown request.'
                    self._disconnect_from_master()
                    self._killThread = True
        return 0

    def _sync_to_master(self):
        print 'Waiting for master broadcast.'
        if not self._socksOpen:
            self._connect_socks()
        startTime = time.time()
        currTime = startTime
        while not self._slaved and currTime - startTime < self.msgTimeout and not self._killThread:
            self._check_socks('broad')
            if self._broadReadable:
                broadIn, addr = self._rec_data('broad')
                if broadIn is None:
                    return 1
                elif broadIn[0] == self.broadMsgPrefixes['Connect']:
                    self._masterAddr = addr[0]
                    print 'Received master connection request.'
                    self._connect_to_master()
            currTime = time.time()
        if self._slaved:
            return 0
        else:
            return 1

    def _connect_to_master(self):
        connected = False
        print 'Connecting to master...'
        self._commSock.connect((self._masterAddr, self._port))
        while not connected and not self._killThread:
            self._check_socks('comm')
            if self._commReadable:
                dataIn = self._rec_data('comm')
                if dataIn[0] == self.commMsgPrefixes['Success']:
                    connected = True
        print 'Connected'
        self._slaved = True
        return

    def _disconnect_from_master(self):
        self._slaved = False
        self._close_socks()
        print 'Disconnected from master.'
        return
