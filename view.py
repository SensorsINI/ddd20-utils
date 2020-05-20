#!/usr/bin/env python

'''
Experimental viewer for DAVIS + OpenXC data
Author: J. Binas <jbinas@gmail.com>, 2017

This software is released under the
GNU LESSER GENERAL PUBLIC LICENSE Version 3.

Usage:
 Play a file from the beginning:
 $ ./view.py <recorded_file.hdf5>

 Play a file, starting at X percent:
 $ ./view.py <recorded_file.hdf5> -s X%

 Play a file starting at second X
 $ ./view.py <recorded_file.hdf5> -s Xs
'''

from __future__ import print_function
import argparse
import ctypes
from argparse import RawTextHelpFormatter
import numpy as np
import h5py
import cv2
import time
from queue import Empty
from ddd20_interfaces.caer import DVS_SHAPE, unpack_header, unpack_data
from ddd20_utils.datasets import CHUNK_SIZE
from ddd20_utils.dddh5reader import MergedStream, HDF5Stream

VIEW_DATA = {
        'dvs',
        'steering_wheel_angle',
        'engine_speed',
        'accelerator_pedal_position',
        'brake_pedal_status',
        'vehicle_speed',
        }


# this changed in version 3
CV_AA = cv2.LINE_AA if int(cv2.__version__[0]) > 2 else cv2.CV_AA

class Interface(object):
    def __init__(self,
                 tmin=0, tmax=0,
                 search_callback=None,
                 update_callback=None,
                 create_callback=None,
                 destroy_callback=None):
        self.tmin, self.tmax = tmin, tmax
        self.search_callback = search_callback
        self.update_callback = update_callback
        self.create_callback = create_callback
        self.destroy_callback = destroy_callback

    def _set_t(self, t):
        self.t_now = int(t - self.tmin)
        if self.update_callback is not None:
            self.update_callback(t)

    def close(self):
        if self.close_callback is not None:
            self.close_callback


class Viewer(Interface):
    ''' Simple visualizer for events '''
    def __init__(self, max_fps=40, zoom=1, rotate180=False, **kwargs):
        super(Viewer, self).__init__(**kwargs)
        self.zoom = zoom
        cv2.namedWindow('frame')
        # tobi added from https://stackoverflow.com/questions/21810452/
        # cv2-imshow-command-doesnt-work-properly-in-opencv-python/
        # 24172409#24172409
        cv2.startWindowThread()
        cv2.namedWindow('polarity')
        ox = 0
        oy = 0
        cv2.moveWindow('frame', ox, oy)
        cv2.moveWindow('polarity', ox + int(448*self.zoom), oy)
        self.set_fps(max_fps)
        self.pol_img = 0.5 * np.ones(DVS_SHAPE)
        self.t_now = 0
        self.t_pre = {}
        self.count = {}
        self.cache = {}
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.display_info = True
        self.display_color = 0
        self.playback_speed = 1.  # seems to do nothing
        self.rotate180 = rotate180
        # sets contrast for full scale event count for white/black
        self.dvs_contrast = 2
        self.paused = False

    def set_fps(self, max_fps):
        self.min_dt = 1. / max_fps

    def show(self, d, t=None):
        # handle keyboad input
        key_pressed = cv2.waitKey(1) & 0xFF  # http://www.asciitable.com/
        if key_pressed != -1:
            if key_pressed == ord('i'):  # 'i' pressed
                if self.display_color == 0:
                    self.display_color = 255
                elif self.display_color == 255:
                    self.display_color = 0
                    self.display_info = not self.display_info
                print('rotated car info display')
            elif key_pressed == ord('x'):  # exit
                print('exiting from x key')
                raise SystemExit
            elif key_pressed == ord('f'):  # f (faster) key pressed
                self.min_dt = self.min_dt*1.2
                print('increased min_dt to ', self.min_dt, ' s')
                # self.playback_speed = min(self.playback_speed + 0.2, 5.0)
                # print('increased playback speed to ',self.playback_speed)
            elif key_pressed == ord('s'):  # s (slower) key pressed
                self.min_dt = self.min_dt/1.2
                print('decreased min_dt to ', self.min_dt, ' s')
                # self.playback_speed = max(self.playback_speed - 0.2, 0.2)
                # print('decreased playback speed to ',self.playback_speed)
            elif key_pressed == ord('b'):  # brighter
                self.dvs_contrast = max(1, self.dvs_contrast-1)
                print('increased DVS contrast to ', self.dvs_contrast,
                      ' full scale event count')
            elif key_pressed == ord('d'):  # brighter
                self.dvs_contrast = self.dvs_contrast+1
                print('decreased DVS contrast to ', self.dvs_contrast,
                      ' full scale event count')
            elif key_pressed == ord(' '):  # toggle paused
                self.paused = not self.paused
                print('decreased DVS contrast to ', self.dvs_contrast,
                      ' full scale event count')
        if self.paused:
            while True:
                key_paused = cv2.waitKey(1) or 0xff
                if key_paused == ord(' '):
                    self.paused = False
                    break

        ''' receive and handle single event '''
        if 'etype' not in d:
            d['etype'] = d['name']
        etype = d['etype']
        if not self.t_pre.get(etype):
            self.t_pre[etype] = -1
        self.count[etype] = self.count.get(etype, 0) + 1
        if etype == 'frame_event' and \
                time.time() - self.t_pre[etype] > self.min_dt:
            if 'data' not in d:
                unpack_data(d)
            img = (d['data'] / 256).astype(np.uint8)
            if self.rotate180 is True:
                # grab the dimensions of the image and calculate the center
                # of the image
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)

                # rotate the image by 180 degrees
                M = cv2.getRotationMatrix2D(center, 180, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
            if self.display_info:
                self._plot_steering_wheel(img)
                self._print(img, (50, 220), 'accelerator_pedal_position', '%')
                self._print(img, (100, 220), 'brake_pedal_status',
                            'brake', True)
                self._print(img, (200, 220), 'vehicle_speed', 'km/h')
                self._print(img, (300, 220), 'engine_speed', 'rpm')
            if t is not None:
                self._plot_timeline(img)
            if self.zoom != 1:
                img = cv2.resize(
                    img, None, fx=self.zoom, fy=self.zoom,
                    interpolation=cv2.INTER_CUBIC)
            cv2.imshow('frame', img)
            # cv2.waitKey(1)
            self.t_pre[etype] = time.time()
        elif etype == 'polarity_event':
            if 'data' not in d:
                unpack_data(d)
            # makes DVS image, but only from latest message
            self.pol_img[d['data'][:, 2], d['data'][:, 1]] += \
                (d['data'][:, 3]-.5)/self.dvs_contrast
            if time.time() - self.t_pre[etype] > self.min_dt:
                if self.zoom != 1:
                    self.pol_img = cv2.resize(
                            self.pol_img, None,
                            fx=self.zoom, fy=self.zoom,
                            interpolation=cv2.INTER_CUBIC)
                    if self.rotate180 is True:
                        # grab the dimensions of the image and
                        # calculate the center
                        # of the image
                        (h, w) = self.pol_img.shape[:2]
                        center = (w // 2, h // 2)

                        # rotate the image by 180 degrees
                        M = cv2.getRotationMatrix2D(center, 180, 1.0)
                        self.pol_img = cv2.warpAffine(self.pol_img, M, (w, h))
                        if self.display_info:
                            self._print_string(self.pol_img, (25, 25), "%.2fms"%(self.min_dt*1000))
                cv2.imshow('polarity', self.pol_img)
                # cv2.waitKey(1)
                self.pol_img = 0.5 * np.ones(DVS_SHAPE)
                
                self.t_pre[etype] = time.time()
        elif etype in VIEW_DATA:
            if 'data' not in d:
                d['data'] = d['value']
            self.cache[etype] = d['data']
            self.t_pre[etype] = time.time()
        if t is not None:
            self._set_t(t)

    def _plot_steering_wheel(self, img):
        if 'steering_wheel_angle' not in self.cache:
            return
        c, r = (173, 130), 65  # center, radius
        a = self.cache['steering_wheel_angle']
        a_rad = + a / 180. * np.pi + np.pi / 2
        if self.rotate180:
            a_rad = np.pi-a_rad
        t = (c[0] + int(np.cos(a_rad) * r), c[1] - int(np.sin(a_rad) * r))
        cv2.line(img, c, t, self.display_color, 2, CV_AA)
        cv2.circle(img, c, r, self.display_color, 1, CV_AA)
        cv2.line(img, (c[0]-r+5, c[1]), (c[0]-r, c[1]),
                 self.display_color, 1, CV_AA)
        cv2.line(img, (c[0]+r-5, c[1]), (c[0]+r, c[1]),
                 self.display_color, 1, CV_AA)
        cv2.line(img, (c[0], c[1]-r+5), (c[0], c[1]-r),
                 self.display_color, 1, CV_AA)
        cv2.line(img, (c[0], c[1]+r-5), (c[0], c[1]+r),
                 self.display_color, 1, CV_AA)
        cv2.putText(
            img, '%0.1f deg' % a,
            (c[0]-35, c[1]+30), self.font, 0.4, self.display_color, 1, CV_AA)

    def _print(self, img, pos, name, unit, autohide=False):
        if name not in self.cache:
            return
        v = self.cache[name]
        if autohide and v == 0:
            return
        cv2.putText(
            img, '%d %s' % (v, unit),
            (pos[0]-40, pos[1]+20), self.font, 0.4,
            self.display_color, 1, CV_AA)

    def _print_string(self, img, pos, string):
         cv2.putText(
            img, '%s' % string,
            (pos[0], pos[1]), self.font, 0.4, self.display_color, 1, CV_AA)

    def _plot_timeline(self, img):
        pos = (50, 10)
        p = int(346 * self.t_now / (self.tmax - self.tmin))
        cv2.line(img, (0, 2), (p, 2), 255, 1, CV_AA)
        cv2.putText(
            img, '%d s' % self.t_now,
            (pos[0]-40, pos[1]+20), self.font, 0.4, self.display_color, 1, CV_AA)

    def close(self):
        cv2.destroyAllWindows()


class Controller(Interface):
    def __init__(self, filename, **kwargs):
        super(Controller, self).__init__(**kwargs)
        cv2.namedWindow('control')
        cv2.moveWindow('control', 400, 698)
        self.f = h5py.File(filename, 'r')
        self.tmin, self.tmax = self._get_ts()
        self.len = int(self.tmax - self.tmin) + 1
        img = np.zeros((100, self.len))
        self.plot_pixels(img, 'headlamp_status', 0, 10)
        self.plot_line(img, 'steering_wheel_angle', 20, 30)
        self.plot_line(img, 'vehicle_speed', 69, 30)
        self.width = 978
        self.img = cv2.resize(
            img, (self.width, 100), interpolation=cv2.INTER_NEAREST)
        cv2.setMouseCallback('control', self._set_search)
        self.t_pre = 0
        self.update(0)
        self.f.close()

    def update(self, t):
        self._set_t(t)
        t = int(float(self.width) / self.len * (t - self.tmin))
        if t == self.t_pre:
            return
        self.t_pre = t
        img = self.img.copy()
        img[:, :t+1] = img[:, :t+1] * 0.5 + 0.5
        cv2.imshow('control', img)
        cv2.waitKey(1)

    def plot_line(self, img, name, offset, height):
        x, y = self.get_xy(name)
        if x is None:
            return
        y -= y.min()
        y = y / y.max() * height
        x = x.clip(0, self.len - 1)
        img[offset+height-y.astype(int), x] = 1

    def plot_pixels(self, img, name, offset=0, height=1):
        x, y = self.get_xy(name)
        if x is None:
            return
        img[offset:offset+height, x] = y

    def _set_search(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        t = self.len * 1e6 * x / float(self.width) + self.tmin * 1e6
        self._search_callback(t)

    def _get_ts(self):
        ts = self.f['dvs']['timestamp']
        tmin = ts[0]
        i = -1
        while ts[i] == 0:
            i -= 1
        tmax = ts[i]
        print('tmin/tmax', tmin, tmax)
        return int(tmin * 1e-6), int(tmax * 1e-6)

    def get_xy(self, name):
        d = self.f[name]['data']
        print('name', name)
        gtz_ids = d[:, 0] > 0
        if not gtz_ids.any():
            return None, 0
        gtz = d[gtz_ids, :]
        return (gtz[:, 0] * 1e-6 - self.tmin).astype(int), gtz[:, 1]


def caer_event_from_row(row):
    '''
    Takes binary dvs data as input,
    returns unpacked event data or False if event type does not exist.
    '''
    sys_ts, head, body = (v.tobytes() for v in row)
    if not sys_ts:
        # rows with 0 timestamp do not contain any data
        return 0, False
    d = unpack_header(head)
    d['dvs_data'] = body
    return int(sys_ts) * 1e-6, unpack_data(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('filename')
    parser.add_argument('--start', '-s', type=str, default=0.,
                        help="Examples:\n"
                             "-s 50%% - play file starting at 50%%\n"
                             "-s 66s - play file starting at 66s")
    parser.add_argument('--rotate', '-r', type=bool, default=False,
                        help="Rotate the scene 180 degrees if True, "
                             "Otherwise False")
    args = parser.parse_args()

    fname = args.filename
    c = Controller(fname,)
    m = MergedStream(HDF5Stream(fname, VIEW_DATA))
    c._search_callback = m.search
    t = time.time()
    t_pre = 0
    t_offset = 0
    r180 = args.rotate
    #  r180arg = "-r180"

    print('recording duration', (m.tmax - m.tmin) * 1e-6, 's')
    # direct skip by command line
    # parse second argument
    try:
        second_opt = args.start
        n_, type_ = second_opt[:-1], second_opt[-1]
        if type_ == '%':
            m.search((m.tmax - m.tmin) * 1e-2 * float(n_) + m.tmin)
        elif type_ == 's':
            m.search(float(n_) * 1e6 + m.tmin)
    except:
        pass
    v = Viewer(tmin=m.tmin * 1e-6, tmax=m.tmax * 1e-6,
               zoom=1.41, rotate180=r180, update_callback=c.update)
    # run main loop
    ts_reset = False
    while m.has_data:
        try:
            sys_ts, d = m.get()
        #  except Queue.Empty:
        except Empty:
            continue
        if not d:
            continue
        if d['etype'] == 'timestamp_reset':
            ts_reset = True
            continue
        if not d['etype'] in {'frame_event', 'polarity_event'}:
            v.show(d)
            continue
        if d['timestamp'] < t_pre:
            print('[WARN] negative dt detected!')
        t_pre = d['timestamp']
        if ts_reset:
            print('resetting timestamp')
            t_offset = 0
            ts_reset = False
        if not t_offset:
            t_offset = time.time() - d['timestamp']
            print('setting offset', t_offset)
        t_sleep = max(d['timestamp'] - time.time() + t_offset, 0)
        time.sleep(t_sleep)
        v.show(d, sys_ts)
        if time.time() - t > 1:
            print(chr(27) + "[2J")
            t = time.time()
            print('fps:\n', '\n'.join(
                ['  %s %s' % (
                 k.ljust(20), v_) for k, v_ in v.count.items()]))
            v.count = {k: 0 for k in v.count}
