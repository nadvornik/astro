import json
import numpy as np
import time
import logging

log = logging.getLogger()

class TempSeries:
    def __init__(self, tc):
        self.tc = tc
#        print(tc)
        self.reset()
        
    def reset(self):
        self.val = 0
        self.prev_t = None
    
    def add(self, val, t = None):
        if t is None:
            t = time.time()
    
        if self.prev_t is None:
            self.val = val
            self.prev_t = t
            return
        
        dt = t - self.prev_t
        r = np.exp(-dt / self.tc)
        
        self.val = self.val * r + val * (1.0 - r)
        self.prev_t = t
        

class TempModel:
    def __init__(self):
        self.n_ser = 13
        self.ser = []
        self.offset = 0
        for i in range(self.n_ser):
            self.ser.append(TempSeries(10 * 1.7**i))
        self.m = np.array([ -1881.87094835,   7547.79105662, -14516.2088065 ,  18634.52020501,
       -18974.44140905,  18117.99394501, -16390.83537827,  14003.74561443,
       -11264.33186464,   8162.08664113,  -5002.21407288,   1985.38835968,
         -430.20444172])
    
    def add(self, val, t = None):
        if t is None:
            t = time.time()
        for s in self.ser:
            s.add(val, t)
            #val = s.val

    def vals(self):
    	return list(map(lambda s: s.val, self.ser))

    def res(self):
        res = np.dot(self.vals(), self.m) - self.offset
        log.info("tempmodel res %.0f", res)
        return res
        
    def set_offset(self, v):
        self.offset = np.dot(self.vals(), self.m) - v
        log.info("tempmodel off %.0f  %.0f", v, self.offset)


def proc_series(tempname, focusname, valid_from, valid_to):
    with open(tempname, 'r') as f:
        temp = json.load(f)

    with open(focusname, 'r') as f:
        focus = json.load(f)
   
    temp=temp['data']['result'][0]['values']
    focus=focus['data']['result'][0]['values']


    tm = TempModel()

    res = []
    i = 0
    j = 0
    while i < len(temp) or j < len(focus):
        if i < len(temp) and (j >= len(focus) or temp[i][0] <= focus[j][0]):
            tm.add(float(temp[i][1]), temp[i][0])
            i += 1
        else:
            if focus[j][0] >= valid_from and focus[j][0] <= valid_to:
                res.append((focus[j][0], float(focus[j][1]), tm.vals()))
            j += 1
    return res

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
    ser1 = proc_series("temp/temp",  "temp/focus",  1564170751,1564192846)
    ser2 = proc_series("temp/temp23", "temp/focus23", 1565811226, 1565825671)
    ser3 = proc_series("temp/temp23", "temp/focus23", 1565984656, 1565998441)
    ser4 = proc_series("temp/temp4", "temp/focus4", 1566414286, 1566441256)
    ser5 = proc_series("temp/temp5", "temp/focus5", 1566500386, 1566528511)


    m = fit_temp_model([ser4, ser5])
    print(repr(m))

    import matplotlib.pyplot as plt
    for ser in [ser1, ser2, ser3, ser4, ser5]:

        src = list(map(lambda s: s[1], ser))
        res = np.dot(list(map(lambda s: s[2], ser)), m)
        res += src[500] - res[500]

        plt.plot(src)
        plt.plot(res)

    #for i in range(9):
    #    plt.plot(list(map(lambda s: s[2][i], ser3)))
    plt.show()

