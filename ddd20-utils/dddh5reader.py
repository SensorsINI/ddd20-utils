"""
Reads DDD hdf5 dvs data and return aps frames + events.
@author: Zhe He, Tobi Delbruck
@contact:hezhehz@live.cn, tobi@ini.uzh.ch
@latest update: 2019-May-31
"""

import queue as Queue
import time
# import pdb
import numpy as np
import logging
import h5py
from tqdm import tqdm

from view import HDF5Stream, MergedStream
from interfaces.caer import unpack_data
from interfaces import caer



class DDD20SimpleReader(object):
    '''
    Simple reader with no multiprocessing threads to read in DDD recording and
    extract data
    '''
    ETYPE_DVS = 'polarity_event'
    ETYPE_APS = 'frame_event'
    ETYPE_IMU = 'imu6_event'

    def __init__(self, fname, startTimeS=None, stopTimeS=None):
        """Init

        Parameters
        ----------
        fname: str
            path of input hdf5 file.
        startTimeS: float
            start time of the stream in seconds from start of recording.
        stopTimeS: float
            stop time of the stream in seconds from start of recording.
        """
        self.f_in =h5py.File(fname, 'r')
        # self.m = MergedStream(self.f_in)
        # self.start = int(self.m.tmin + 1e6 * startTimeS) if startTimeS else 0
        # self.stop = (self.m.tmin + 1e6 * stopTimeS) if stopTimeS else self.m.tmax
        # self.m.search(self.start)
        logging.info(str(fname)+' contains following keys')
        hasDavisData=False
        dvsKey='dvs'
        for key in self.f_in.keys():
            if key==dvsKey: hasDavisData=True
            print(key)
        if not hasDavisData: raise('file does not contain DAVIS data (key dvs)')

        dvsGroup=self.f_in[dvsKey]
        logging.info('group dvs contains following keys')
        for key in dvsGroup.keys():
            print(key)

        logging.info('group dvs contains following items')
        for item in dvsGroup.items():
            print(item)

        self.davisData=dvsGroup['data']
        logging.info('The DAVIS data has the shape '+str(self.davisData.shape))

        self.numPackets=self.davisData.shape[0] # start here, this is not actual size
        firstPacket=self.readPacket(0)
        self.startTimeS=firstPacket['timestamp']
        # the last packets in file are actually empty (some consequence of how file is written)
        # just go backards until we get a packet with some data
        lastPacket=self.readPacket(self.numPackets-1)
        while not lastPacket:
            self.numPackets-=1
            lastPacket = self.readPacket(self.numPackets-1)
        self.endTimeS=lastPacket['timestamp']
        logging.info('file has '+str(self.numPackets)+' packets with start time='+str(self.startTimeS)+'s and end time='+str(self.endTimeS)+'s')

        # logging.info('Sample DAVIS data is the following')
        # i=0
        # for dat in self.davisData:
        #     headerDat=dat[1] # caer header
        #     header=caer.unpack_header(headerDat) # gets the packet type
        #     data = {'dvs_header': dat[1]} # put it to the dict as header
        #     data.update(caer.unpack_header(data['dvs_header'])) # update the dict?
        #     dat0=dat[0]   # timestamp of the packet?
        #     data['dvs_data'] = dat[2] # put the data payload, dvs_data refers to DAVIS camera data, can be frames or IMU data too
        #     data=caer.unpack_data(data) # use caer to unpack it, store it back to data, which gets timestamp and cooked data
        #     # print some info
        #     if data: # if could not unpack, is False
        #         print('packet #'+str(i)
        #           +' timestamp: '+str(data['timestamp'])
        #           +' etype: '+str(data['etype'])
        #           +' esize: '+str(data['esize'])
        #           +' enumber: '+str(data['enumber'])
        #           )
        #     i+=1
        #     if i>50: break


    def readPacket(self, number):
        """
        Reads packet k in the dataset
        Parameters
        ----------
        number: number of packet, in range(0,numPackets)

        Returns
        -------
        packet of data, or False if packet is outside of range or cannot be extracted
        """
        if number >= self.numPackets or number<0: return False
        dat = self.davisData[number]
        headerDat = dat[1]  # caer header
        if headerDat.shape[0]==0: return False  # empty packet, can happen at end of recording
        packet = {'dvs_header': dat[1]}  # put it to the dict as header
        packet.update(caer.unpack_header(packet['dvs_header']))  # update the dict?
        # dat0 = dat[0]  # timestamp of the packet?
        packet['dvs_data'] = dat[2]  # put the data payload, dvs_data refers to DAVIS camera data, can be frames or IMU data too
        packet = caer.unpack_data(packet)  # use caer to unpack it, store it back to data, which gets timestamp and cooked data
        packet.timestamp=packet['timestamp']
        packet.etype=packet['etype']

        # # print some info
        # if data:  # if could not unpack, is False
        #     print('packet #' + str(k)
        #           + ' timestamp: ' + str(data['timestamp'])
        #           + ' etype: ' + str(data['etype'])
        #           + ' esize: ' + str(data['esize'])
        #           + ' enumber: ' + str(data['enumber'])
        #           )
        return packet

    def search(self,timeS):
        """
        Search for a starting time
        Parameters
        ----------
        timeS time in s from start of recording (self.startTimeS)

        Returns
        -------
        packet number

        """
        logging.info('searching for start time {}'.format(timeS))
        for k in tqdm(range(0,self.numPackets)):
            data=self.readPacket(k)
            if not data: # maybe cannot parse this particular type of packet (e.g. imu6)
                continue
            t=data['timestamp']
            if t>=self.startTimeS+timeS:
                logging.info('\nfound start time '+str(timeS)+' at packet '+str(k))
                return k
        logging.warning('\ncould not find start time '+str(timeS)+' before end of file')
        return False


class DDD20ReaderMultiProcessing(object):
    """
    Read aps frames and events from hdf5 files in DDD
    @author: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-May-31
    """

    def __init__(self, fname, startTimeS=None, stopTimeS=None):
        """Init

        Parameters
        ----------
        fname: str
            path of input hdf5 file.
        startTimeS: float
            start time of the stream in seconds.
        stopTimeS: float
            stop time of the stream in seconds.
        """
        self.f_in = HDF5Stream(fname, {'dvs'})
        self.m = MergedStream(self.f_in)
        self.start = int(self.m.tmin + 1e6 * startTimeS) if startTimeS else 0
        self.stop = (self.m.tmin + 1e6 * stopTimeS) if stopTimeS else self.m.tmax
        self.m.search(self.start)

    def readEntire(self):
        """
        Read entire file to memory.

        Returns
        -------
        aps_ts: np.array,
            timestamps of aps frames.
        aps_frame: np.ndarray, [n, width, height]
            aps frames
        events: numpy record array.
            events, col names: ["ts", "y", "x", "polarity"], \
                data types: ["<f8", "<i8", "<i8", "<i8"]
        """
        sys_ts, t_offset, current = 0, 0, 0
        timestamp = 0
        frames, events = [], []
        while self.m.has_data and sys_ts <= self.stop * 1e-6:
            try:
                sys_ts, d = self.m.get()
            except Queue.Empty:
                # wait for queue to fill up
                time.sleep(0.01)
                continue
            if not d or sys_ts < self.start * 1e-6:
                # skip unused data
                continue
            if d['etype'] == 'special_event':
                unpack_data(d)
                # this is a timestamp reset
                if any(d['data'] == 0):
                    print('ts reset detected, setting offset', timestamp)
                    t_offset += current
                    # NOTE the timestamp of this special event is not meaningful
                continue
            if d['etype'] == 'frame_event':
                ts = d['timestamp'] + t_offset
                frame = filter_frame(unpack_data(d))
                data = np.array(
                    [(ts, frame)],
                    dtype=np.dtype(
                        [('ts', np.float64),
                         ('frame', np.uint8, frame.shape)]
                    )
                )
                frames.append(data)
                current = ts
                continue
            if d['etype'] == 'polarity_event':
                unpack_data(d)
                data = d["data"]
                data = np.hstack(
                    (data[:, 0][:, None] * 1e-6 + t_offset,
                     data[:, 1][:, None],
                     data[:, 2][:, None],
                     data[:, 3].astype(np.int)[:, None] * 2 - 1)
                )
                events.append(data)
                continue
        frames = np.hstack(frames)
        events = np.vstack(events)
        frames["ts"] -= frames["ts"][0]
        events[:, 0] -= events[0][0]
        self.f_in.exit.set()
        self.m.exit.set()
        self.f_in.join()
        self.m.join()

        return frames, events

def filter_frame(d):
    '''
    receives 16 bit frame,
    needs to return unsigned 8 bit img
    '''
    # add custom filters here...
    # d['data'] = my_filter(d['data'])
    frame8 = (d['data'] / 256).astype(np.uint8)
    return frame8
