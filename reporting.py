
import os, sys, time


txt_norm = '\033[0;37m'
txt_bold = '\033[1;37m'
txt_grn = '\033[0;32m'
txt_red = '\033[1;31m'


class Stats(object):
    def __init__(self, filename, counters, buffers, cls=True):
        self.buffers = buffers
        self.counters = counters
        self.filename = filename
        self.max_qsize = {k: 0 for k in self.buffers}
        self.cls = cls
        self.t0 = time.time()
        self.t_pre = time.time()

    def report(self):
        ''' print some stats '''
        for k,o in self.buffers.iteritems():
            self.max_qsize[k] = max(self.max_qsize[k], o.q.qsize())
        if time.time() - self.t_pre < 1:
            return
        self.t_pre = time.time()
        if self.cls:
            os.system('clear')
        # print counter stats
        for name, counter in self.counters.iteritems():
            print txt_bold, '\nrecording', name, 'events:', txt_norm
            for k in counter:
                c = txt_red if counter[k] == 0 else txt_norm
                print '%s  %s: %s Hz%s' % (c, k, counter[k], txt_norm)
                counter[k] = 0
        # print buffer stats
        print txt_bold, '\nbuffer status:', txt_norm
        for k,o in self.buffers.iteritems():
            p = 100. * self.max_qsize[k] / o.q._maxsize
            p60 = int(0.6 * p)
            out = '  %s: %d/%d ' % (k, self.max_qsize[k], o.q._maxsize)
            out = out.ljust(p60, '|').ljust(60, '.') + ' %0.2f %%' % p
            out = txt_red + out[:p60] + txt_grn + out[p60:] + txt_norm
            self.max_qsize[k] = 0
            print out
            if not o.is_alive():
                print txt_red + \
                        '\n  Problem encountered in %s module. Exiting.\n' % k \
                        + txt_norm
                sys.exit()
        print '\n\nrecording into %s (%d s)\n' % (self.filename, time.time() - self.t0)
