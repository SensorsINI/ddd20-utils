#!/usr/bin/python

from __future__ import print_function
import sys
import h5py
import numpy as np
import scipy.interpolate as ip


tstep = 1


if __name__ == '__main__':
    k = sys.argv[1]
    fnames = sys.argv[2:]

    vout = []

    for fname in fnames:
        try:
            f = h5py.File(fname, 'r')
        except:
            print('could not open', fname)
            continue
        print('processing', fname)
        nnz = f[k]['timestamp'][:] > 0
        if not any(nnz):
            print('no data...')
            continue
        data = f[k]['data'][nnz, :]
        t, v = data[:, 0] * 1e-6, data[:, 1]
        curve = ip.interp1d(t, v)
        tnew = np.linspace(t[0], t[-1], int((t[-1] - t[0]) / tstep))
        vout.append(curve(tnew))

    out = np.hstack(vout)
    np.savetxt(k + '.dat', out)
