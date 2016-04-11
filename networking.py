import threading
import socket
# import struct
# import fcntl
import select
import sys
import time

__author__ = 'Michael Holmes'


# Class to hold various protocol strings, identifiers, format etc.
class NetProto:
    def __init__(self):
        self.sockTimeout = 0.01
        self.recBuffSize = 1024
        self.commMsgPrefixes = {'Calibrate': 'C', 'Disconnect': 'D', 'Success': 'Y', 'Position': 'P'}
        self.broadMsgPrefixes = {'Connect': 'C', 'Disconnect': 'D'}
        self.commMsgPrefixesInv = {}
        for key, value in self.commMsgPrefixes.iteritems():
            self.commMsgPrefixesInv[value] = key
        self.broadMsgPrefixesInv = {}
        for key, value in self.broadMsgPrefixes.iteritems():
            self.broadMsgPrefixesInv[value] = key


# Base class for Master and Slave Nodes/ Worker Threads, handles low level socket actions
class NetBase(threading.Thread, NetProto):
    def __init__(self, nodeType, port):
        threading.Thread.__init__(self)
        NetProto.__init__(self)
        self.nodeType = nodeType
        try:
            self.commSock = socket.socket()
            self.broadSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error, msg:
            print 'Failed to create socket. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
            sys.exit()
        self.commSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.broadSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.nodeType == 'Master':
            self.broadSock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.hostname = socket.getfqdn()
        self.hostAddr = socket.gethostbyname(self.hostname)
        self.broadAddr = '192.168.1.255'
        # self.subnetMask = socket.inet_ntoa(fcntl.ioctl(socket.socket(socket.AF_INET, socket.SOCK_DGRAM), 35099, struct.pack('256s', "wlan0"))[20:24])
        self.port = port
        self.commReadable = False
        self.commWriteable = False
        self.broadReadable = False
        self.broadWriteable = False
        try:
            self.commSock.bind(('', self.port))
            if self.nodeType == 'Slave':
                self.broadSock.bind(('', self.port))
        except socket.error, msg:
            print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
            sys.exit()
        self.killThread = False
        return

    def stop(self):
        self.killThread = True
        return

    def close_socks(self):
        self.commSock.close()
        self.broadSock.close()

    def check_comm_sock(self):
        readableSocks, writeableSocks, errorSocks = select.select([self.commSock], [self.commSock], [],
                                                                  self.sockTimeout)
        if self.commSock in readableSocks:
            self.commReadable = True
        else:
            self.commReadable = False
        if self.commSock in writeableSocks:
            self.commWriteable = True
        else:
            self.commWriteable = False
        return

    def check_broad_sock(self):
        readableSocks, writeableSocks, errorSocks = select.select([self.broadSock], [self.broadSock], [],
                                                                  self.sockTimeout)
        if self.broadSock in readableSocks:
            self.broadReadable = True
        else:
            self.broadReadable = False
        if self.broadSock in writeableSocks:
            self.broadWriteable = True
        else:
            self.broadWriteable = False
        return

    def send_data(self, prefixKey, data=None):
        totalSent = 0
        if data is None:
            msgLen = 4
            msg = self.commMsgPrefixes[prefixKey] + '%03d' % msgLen
        else:
            msgLen = 4 + len(data)
            msg = self.commMsgPrefixes[prefixKey] + '%03d' % msgLen + data
        while totalSent < msgLen:
            sent = self.commSock.send(msg[totalSent:])
            if sent == 0:
                raise RuntimeError("Communications connection broken")
            totalSent += sent
        return

    def rec_data(self):
        chunks = []
        bytesRec = 0
        while bytesRec < 4:
            chunk = self.commSock.recv(12)
            if chunk == '':
                raise RuntimeError("Communications connection broken")
            chunks.append(chunk)
            bytesRec += len(chunk)
        msgPrefix = ''.join(chunks)
        numBytes = int(msgPrefix[1:4])
        bytesRec = 0
        if len(msgPrefix) > 4:
            chunks = list(msgPrefix[4:])
        else:
            chunks = []
        while bytesRec < numBytes - 4:
            chunk = self.commSock.recv(min(numBytes - bytesRec, self.recBuffSize))
            if chunk == '':
                raise RuntimeError("Communications connection broken")
            chunks.append(chunk)
            bytesRec += len(chunk)
        return msgPrefix + ''.join(chunks)

    # Broadcast master node address to local network
    def send_broadcast(self, prefixKey):
        totalSent = 0
        msg = self.broadMsgPrefixes[prefixKey] + '%03d' % 4
        while totalSent < 4:
            sent = self.broadSock.sendto(msg[totalSent:], (self.broadAddr, self.port))
            if sent == 0:
                raise RuntimeError("Broadcast connection broken")
            totalSent += sent
        return

    def rec_broadcast(self):
        chunks = []
        bytesRec = 0
        addr = None
        while bytesRec < 4:
            chunk, addr = self.broadSock.recvfrom(self.recBuffSize)
            if chunk == '':
                raise RuntimeError("Broadcast connection broken")
            chunks.append(chunk)
            bytesRec += len(chunk)
        msgPrefix = ''.join(chunks)
        numBytes = int(msgPrefix[1:4])
        bytesRec = 0
        if len(msgPrefix) > 4:
            chunks = list(msgPrefix[4:])
        else:
            chunks = []
        while bytesRec < numBytes - 4:
            chunk = self.broadSock.recv(min(numBytes - bytesRec, self.recBuffSize))
            if chunk == '':
                raise RuntimeError("Broadcast connection broken")
            chunks.append(chunk)
            bytesRec += len(chunk)
        return msgPrefix + ''.join(chunks), addr


# Class for Master Server Node
class MasterNode(NetBase):
    def __init__(self, port):
        NetBase.__init__(self, 'Master', port)
        self.clients = {}
        self.name = 'Master'
        self.threadID = 0
        return

    def run(self):
        self.send_broadcast('Connect')
        self.commSock.listen(5)
        conn = None
        while not self.killThread:
            self.check_comm_sock()
            if self.commReadable:
                # If there is a new inbound connection, answer it
                conn, addr = self.commSock.accept()
                # Check if addr matches a current thread and delete it, otherwise create new thread
                if addr in self.clients:
                    self.clients[addr].stop()
                    while self.clients[addr].isAlive():
                        pass
                    del self.clients[addr]
                self.clients[addr] = MasterThread(self.threadID, conn, addr)
                self.clients[addr].start()
                print 'New connection: %s' % addr[0]
                self.threadID += 1
                conn = None
        # Kill remaining worker threads then exit
        self.send_broadcast('Disconnect')
        self.close_socks()
        for addr in self.clients.keys():
            self.clients[addr].stop()
            while self.clients[addr].isAlive():
                pass
        return


# Worker thread for Master Server Node
class MasterThread(NetBase):
    def __init__(self, threadID, conn, addr):
        threading.Thread.__init__(self)
        NetProto.__init__(self)
        self.threadID = threadID
        self.commSock = conn
        self.name = addr
        self.killThread = False
        return

    def run(self):
        self.send_data('Success')
        while not self.killThread:
            self.check_comm_sock()
            if self.commReadable:
                print self.rec_data()
        return

    def close_socks(self):
        self.commSock.close()


# Class for Slave Client Node
class SlaveNode(NetBase):
    def __init__(self, port):
        NetBase.__init__(self, 'Slave', port)
        self.name = 'Slave'
        self.masterAddr = None
        return

    def run(self):
        while self.masterAddr is None and not self.killThread:
            self.check_broad_sock()
            if self.broadReadable:
                broadIn, addr = self.rec_broadcast()
                if broadIn[0] == self.broadMsgPrefixes['Connect']:
                    self.masterAddr = addr[0]
                    connected = False
                    print 'Broadcast received, connecting.'
                    self.commSock.connect((self.masterAddr, self.port))
                    while not connected and not self.killThread:
                        self.check_comm_sock()
                        if self.commReadable:
                            dataIn = self.rec_data()
                            if dataIn[0] == self.commMsgPrefixes['Success']:
                                connected = True
                    print 'Connected'
        while not self.killThread:
            self.check_comm_sock()
            if self.commReadable:
                print self.rec_data()
        return


# node = 'Master'
node = 'Slave'

if __name__ == '__main__' and node == 'Master':
    myMaster = MasterNode(44623)
    myMaster.start()
    lastTime = time.time()
    startTime = lastTime
    while not myMaster.killThread:
        currTime = time.time()
        if currTime - lastTime > 10:
            lastTime = currTime
            print 'Number of clients: %d' % len(myMaster.clients)
        if currTime - startTime > 120:
            myMaster.stop()

if __name__ == '__main__' and node == 'Slave':
    mySlave = SlaveNode(44623)
    mySlave.start()
    lastTime = time.time()
    while not mySlave.killThread:
        currTime = time.time()
        if currTime - lastTime > 90:
            mySlave.stop()
