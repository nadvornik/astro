import json
import numpy as np
import time
import logging

log = logging.getLogger()

class TempSeries:
    def __init__(self, tc, status):
        self.status = status
        self.tc = tc
#        print(tc)
        self.status.setdefault('val', 0)
        self.status.setdefault('prev_t', None)
        
        
    def reset(self):
        self.status['val'] = 0
        self.status['prev_t'] = None
    
    def add(self, val, t = None):
        if t is None:
            t = time.time()
    
        if self.status['prev_t'] is None:
            self.status['val'] = val
            self.status['prev_t'] = t
            return
        
        dt = t - self.status['prev_t']
        r = np.exp(-dt / self.tc)
        
        self.status['val'] = self.status['val'] * r + val * (1.0 - r)
        self.status['prev_t'] = t

    def getval(self):
        return self.status['val']

class TempModel:
    def __init__(self, status = None):
        if status is None:
            status = {}
        #self.n_ser = 13
        self.status = status
        self.n_ser = 8
        self.n_inp = 2
        self.ser = []
        self.offset = 0
        self.status.setdefault('series', [None] * self.n_inp)
        if len(self.status['series']) != self.n_inp:
            self.status['series'] = [None] * self.n_inp
        	
        for j in range(self.n_inp):
            self.ser.append([])
            if not self.status['series'][j] or len(self.status['series'][j]) != self.n_ser:
            	self.status['series'][j] = [None] * self.n_ser
            for i in range(self.n_ser):
#                self.ser[j].append(TempSeries(10 * 1.7**i))
                if not self.status['series'][j][i]:
                    self.status['series'][j][i] = {}
                self.ser[j].append(TempSeries(10 * 2**i, self.status['series'][j][i]))


        self.m = np.array([  -411.85594942,   2217.2011034 ,  -6184.16365289,  11450.68024559,
       -13268.30039248,   8877.1265101 ,  -2976.20681318,    189.24613256,
          -81.5995854 ,     47.51171379,    691.78160701,  -2864.94795918,
         3660.18695886,  -2355.46844124,    127.04236913,    923.38084718])
        
    def add(self, i, val, t = None):
        if t is None:
            t = time.time()
        for s in self.ser[i]:
            s.add(val, t)
            #val = s.val

    def vals(self):
        res = []
        for j in range(self.n_inp):
    	    res += list(map(lambda s: s.getval(), self.ser[j]))
        return res

    def res(self):
        res = np.dot(self.vals(), self.m) - self.offset
        log.info("tempmodel res %.0f", res)
        return res
        
    def set_offset(self, v):
        self.offset = np.dot(self.vals(), self.m) - v
        log.info("tempmodel off %.0f  %.0f", v, self.offset)

    def add_offset(self, v):
    	self.offset += v

def proc_series(tempname1, tempname2, focusname, valid_from, valid_to):
    with open(tempname1, 'r') as f:
        temp1 = json.load(f)

    with open(tempname2, 'r') as f:
        temp2 = json.load(f)

    with open(focusname, 'r') as f:
        focus = json.load(f)
   
    temp1 = temp1['data']['result'][0]['values']
    temp1.reverse()
    temp2 = temp2['data']['result'][0]['values']
    temp2.reverse()
    focus=focus['data']['result'][0]['values']


    tm = TempModel()

    res = []
    j = 0
    t_temp = [[], []]
    t_foc = []
    while j < len(focus):
        for i, temp in enumerate([temp1, temp2]):
            if len(temp) and (j >= len(focus) or temp[-1][0] <= focus[j][0]):
                
                t , tr = temp.pop()
                tr = float(tr)
                tm.add(i, tr, t)
          
                if t >= valid_from - 2*3600 and t <= valid_to:
                    t_temp[i].append((t, tr))
                
        else:
            if focus[j][0] >= valid_from and focus[j][0] <= valid_to:
                res.append((focus[j][0], float(focus[j][1]), tm.vals()))
                t_foc.append((focus[j][0], float(focus[j][1])))
            j += 1
    return res, t_temp[0], t_temp[1], t_foc

def proc_series2(fn):
    with open(fn, 'r') as f:
       lines = f.readlines()

    res = []
    t_temp = [[], []]
    t_foc = []

    tm = TempModel()

    for l in lines:
        sl = l.split(' ')
        #1619635930.218231 15.920000000000002 13.370000000000005 1760 ba_run 0.11976221315862048 -1.731914027918022
        t = float(sl[0])
        ta = float(sl[1])
        tb = float(sl[2])

        foc = float(sl[3])

        mode = sl[4]
        ba_int = float(sl[5])
        ba_off = float(sl[6])

        valid = (mode == 'ba_run') and (abs(ba_off) < 1.5) and (ba_off != 0.0)

        tm.add(0, ta, t)
        tm.add(1, tb, t)

        t_temp[0].append((t, ta))
        t_temp[1].append((t, tb))

        if valid:
            res.append((t, foc, tm.vals()))
            t_foc.append((t, foc))

    return res, t_temp[0], t_temp[1], t_foc

def fit_temp_model(series):
    A = []
    b = []
    for i, s in enumerate(series):
        for v in s:
            c = [0.0] * len(series)
            c[i] = 1.0
            A.append(v[2] + c)
            b.append(v[1])
    m = np.linalg.lstsq(A, b, rcond=None)[0]
    return m[0:-len(series)]


if __name__ == "__main__":
#    ser1, t_temp1, t_foc1 = proc_series("temp/temp",  "temp/focus",  1564170751,1564192846)
#    ser2, t_temp2, t_foc2 = proc_series("temp/temp23", "temp/focus23", 1565811226, 1565825671)
#    ser3, t_temp3, t_foc3 = proc_series("temp/temp23", "temp/focus23", 1565984656, 1565998441)
#    ser4, t_temp4, t_foc4 = proc_series("temp/temp4", "temp/focus4", 1566414286, 1566441256)
#    ser5, t_temp5, t_foc5 = proc_series("temp/temp5", "temp/focus5", 1566500386, 1566528511)
#    ser6, t_temp6, t_foc6 = proc_series("temp/temp6", "temp/focus6", 1566674266, 1566698551)

#    ser1, t_temp1a, t_temp1b, t_foc1 = proc_series("temp/temp1", "temp/tempref1", "temp/focus1",  1568484360, 1568520000)
#    ser2, t_temp2a, t_temp2b, t_foc2 = proc_series("temp/temp2", "temp/tempref2", "temp/focus2",  1568570040, 1568606100)
#    ser3, t_temp3a, t_temp3b, t_foc3 = proc_series("temp/temp3", "temp/tempref3", "temp/focus3",  1568829120, 1568862000)


    ser1, t_temp1a, t_temp1b, t_foc1 = proc_series2("ba_log/4/ba_log_1619593214")
    ser2, t_temp2a, t_temp2b, t_foc2 = proc_series2("ba_log/2/ba_log_1619426865")
    ser3, t_temp3a, t_temp3b, t_foc3 = proc_series2("ba_log/3_nf/ba_log_1619506815")

    m = fit_temp_model([ser1, ser2])
    m = TempModel().m
    print(repr(m))
    print(np.sum(m))

    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    import datetime
    plots = [ser1, ser2]
    fig = plt.figure()
    for i,ser in enumerate(plots):

        src = np.array(list(map(lambda s: s[1], ser)))
        res = np.dot(list(map(lambda s: s[2], ser)), m)
        res += src[5000] - res[5000]
        ax1 = fig.add_subplot(len(plots), 1, i + 1)
        ax1.plot(src)
        ax1.plot(res)

    pl_ts = [(t_temp1a, t_temp1b, t_foc1)]
    
    fig = plt.figure()
    for i, (t_temp_a, t_temp_b, t_foc) in enumerate(pl_ts): 
        ax1 = fig.add_subplot(len(pl_ts), 1, i + 1)
        
        ax1.xaxis.set_major_formatter( md.DateFormatter( '%H:%M' ) )
        
        x, y = np.array(t_temp_a).T
        x=[datetime.datetime.fromtimestamp(ts) for ts in x]
        ax1.plot(x, y, color='blue')
        x, y = np.array(t_temp_b).T
        x=[datetime.datetime.fromtimestamp(ts) for ts in x]
        ax1.plot(x, y, color='green')
        ax2 = fig.add_subplot(len(pl_ts), 1, i + 1, sharex=ax1, frameon=False)

        ax2.xaxis.set_major_formatter( md.DateFormatter( '%H:%M' ) )

        ax2.yaxis.tick_right()
        x, y = np.array(t_foc).T
        x=[datetime.datetime.fromtimestamp(ts) for ts in x]
        ax2.plot(x, y - y[0], color='red')

    #for i in range(9):
    #    plt.plot(list(map(lambda s: s[2][i], ser3)))
    plt.show()

