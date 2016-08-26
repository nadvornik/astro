#!/usr/bin/python

import numpy as np

def polyfit2d(x, y, f, deg):
	deg = np.asarray(deg)
	vander = np.polynomial.polynomial.polyvander2d(x, y, deg)
	vander = vander.reshape((-1,vander.shape[-1]))
	f = f.reshape((vander.shape[0],))
	c = np.linalg.lstsq(vander, f)[0]
	return c.reshape(deg+1)

def interpolate2d(a, x, y):
	x0=np.array(x, dtype=np.int)
	y0=np.array(y, dtype=np.int)
	x0 = np.clip(x0, 0, a.shape[1] - 2)
	y0 = np.clip(y0, 0, a.shape[0] - 2)
	xf = np.array(x - x0)
	yf = np.array(y - y0)
	
	vn = (a[y0, x0] * (1.0 - yf) + a[y0 + 1, x0] * yf) * (1.0 - xf) + (a[y0, x0 + 1] * (1.0 - yf) + a[y0 + 1, x0 + 1] * yf) * xf
	return vn



