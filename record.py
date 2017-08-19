#!/usr/bin/python

'''
Recorder for DAVIS + OpenXC data
Author: J. Binas <jbinas@gmail.com>, 2017

This software is released under the
GNU LESSER GENERAL PUBLIC LICENSE Version 3.

Usage:
 Record in a specific directory
 $ ./record.py <recording_dir>

 Record into a specific file
 $ ./record.py <recording_file.hdf5>
'''

import time, sys, os, signal, thread
import numpy as np
import interfaces, datasets
from reporting import Stats
from view import Viewer, unpack_data
import Queue


BUFSIZE_DS = 32384
BUFSIZE_AER = 8192
BUFSIZE_OXC = 1024

dtypes = {
        'dvs/data': (datasets.h5py.special_dtype(vlen=np.uint8), (3,)),
        'dvs/timestamp': int,
        }

dtypes_vi = {
        'accelerator_pedal_position': float,
        'brake_pedal_status': bool,
        'engine_speed': float,
        'fine_odometer_since_restart': float,
        'fuel_consumed_since_restart': float,
        'fuel_level': float,
        'headlamp_status': bool,
        'high_beam_status': bool,
        'ignition_status': datasets.h5py.special_dtype(vlen=unicode),
        'lateral_acceleration': float,
        'latitude': float,
        'longitude': float,
        'longitudinal_acceleration': float,
        'odometer': float,
        'parking_brake_status': bool,
        'steering_wheel_angle': float,
        'torque_at_transmission': float,
        'transmission_gear_position': datasets.h5py.special_dtype(vlen=unicode),
        'gear_lever_position': datasets.h5py.special_dtype(vlen=unicode),
        #'turn_signal_status': datasets.h5py.special_dtype(vlen=unicode),
        'vehicle_speed': float,
        'windshield_wiper_status': bool,
        }

ignition_status = {
        'off': 0,
        'accessory': 1,
        'run': 2,
        'start': 3
        }

gear_position = {
        'neutral': 0,
        'nuetral': 0, # misspelled in ford VI version 4 firmware
        'first': 1,
        'second': 2,
        'third': 3,
        'fourth': 4,
        'fifth': 5,
        'sixth': 6,
        'seventh': 7,
        'eighth': 8,
        'ninth': 9,
        'tenth': 10,
        'drive': 3,
        'sport': 2,
        'low': 1,
        'reverse': -1,
        'park': -2,
        }

conversions_vi = {
        'brake_pedal_status': float,
        'headlamp_status': float,
        'high_beam_status': float,
        'parking_brake_status': float,
        'windshield_wiper_status': float,
        'ignition_status': lambda v: ignition_status.get(v, 99),
        'transmission_gear_position': lambda v: gear_position.get(v, 99),
        'gear_lever_position': lambda v: gear_position.get(v, 99), # added defauult of 99 for unknown values rather than None
        }

# -- end of config --


for k, dt in dtypes_vi.iteritems():
    #dtypes[k + '/data'] = dt
    dtypes[k + '/data'] = (float, (2,))
    dtypes[k + '/timestamp'] = int

def save_aer(ds, data):
    ''' send aer data dict to dataset buffer '''
    row = [ np.fromstring(np.bytes_(data['dvs_timestamp']), dtype=np.uint8),
            np.fromstring(data['dvs_header'], dtype=np.uint8),
            np.fromstring(data['dvs_data'], dtype=np.uint8), ]
    ds.save({'dvs_timestamp': data['dvs_timestamp'], 'dvs_data': row})

def save_vi(ds, data):
    ''' send vi data dict to dataset buffer '''
    if data['name'] not in dtypes_vi:
        return False
    conv = conversions_vi.get(data['name'], False)
    val = conv(data['value']) if conv else data['value']
    ds.save({
        data['name'] + '_data': [data['timestamp'], val],
        data['name'] + '_timestamp': data['timestamp']
    })
    return True

def get_filename():
    ''' generate file name of the recording file '''
    filename = 'rec%s.hdf5' % int(time.time())
    if len(sys.argv) > 1:
        path = sys.argv.pop(-1).strip()
        if path.endswith('hdf5'):
            filename = path
        else:
            filename = os.path.join(path, filename)
    return filename


def input_thread(l):
    raw_input('Press enter to start recording...')
    l.append(None)


if __name__ == '__main__':
    filename = get_filename()
    aer = interfaces.caer.Monitor(bufsize=BUFSIZE_AER)
    vi = interfaces.openxc.Monitor(bufsize=BUFSIZE_OXC)
    exposure = interfaces.caer.ExposureCtl()
    # flush buffers
    t = time.time()
    while time.time() - t < 1:
        aer.get()
        vi.get()
    
    # pre-recording loop
    viewer = Viewer(zoom=1.41,rotate180=True)
    inp_detect = []
    thread.start_new_thread(input_thread, (inp_detect,))
    while not inp_detect:
        res = aer.get()
        if res and res['etype'] in interfaces.caer.EVENT_TYPES and res['evalid']:
            viewer.show(res)
            exposure.update(res)
        res = vi.get()
        if res:
            viewer.show(res)
    # end of pre-recording loop

    # init recording file
    dataset = datasets.HDF5(filename, dtypes, bufsize=BUFSIZE_DS)
    count_aer = {k: 0 for k in interfaces.caer.EVENT_TYPES}
    count_vi = {k: 0 for k in dtypes_vi}
    stats = Stats(filename,
            counters={'aer': count_aer, 'vi': count_vi},
            buffers={'aer': aer, 'vi': vi, 'dataset': dataset})

    # flush buffers
    t = time.time()
    while time.time() - t < 0.2:
        aer.get()
        vi.get()


    # wait for keyboard input
    def end_thread(list):
        raw_input('hit enter to end recording...')
        list.append(None)

    #start recording
    raw_inp = []
    viewer.set_fps(5)
    
    
    thread.start_new_thread(end_thread, (raw_inp,))
    while not raw_inp:
#    while not dataset.exit.is_set():
        try:
            # get aer data
            res = aer.get()
            if res and res['etype'] in interfaces.caer.EVENT_TYPES and res['evalid']:
                save_aer(dataset, res)
                count_aer[res['etype']] += res['ecapacity']
                viewer.show(res)
                exposure.update(res)
            # get vi data
            res = vi.get()
            if res:
                if save_vi(dataset, res):
                  count_vi[res['name']] += 1
                  viewer.show(res)
            stats.report()
        except KeyboardInterrupt:
            print '\ninterrupt, exiting...'
            dataset.exit.set()
            viewer.close()
        except Queue.Full:
            print('queue full, ignoring')
            pass

    print '\nexiting...'
    dataset.exit.set()
    aer.exit.set()
    vi.exit.set()
    viewer.close()


