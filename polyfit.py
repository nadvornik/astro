#!/usr/bin/python

import numpy as np

def polyfit2d(x, y, f, deg):
	vander = np.polynomial.polynomial.polyvander2d(x, y, (deg, deg))[:, np.where(np.flipud(np.tri(deg + 1)).ravel())[0]]
	f = f.reshape((vander.shape[0],))
	c = np.linalg.lstsq(vander, f)[0]
	res = np.zeros((deg + 1, deg + 1))
	res[np.where(np.flipud(np.tri(deg + 1)))] = c
	return res

def interpolate2d(a, x, y):
	x0=np.array(x, dtype=np.int)
	y0=np.array(y, dtype=np.int)
	x0 = np.clip(x0, 0, a.shape[1] - 2)
	y0 = np.clip(y0, 0, a.shape[0] - 2)
	xf = np.array(x - x0)
	yf = np.array(y - y0)
	
	vn = (a[y0, x0] * (1.0 - yf) + a[y0 + 1, x0] * yf) * (1.0 - xf) + (a[y0, x0 + 1] * (1.0 - yf) + a[y0 + 1, x0 + 1] * yf) * xf
	return vn




if __name__ == '__main__':
	x = [ 1.0, 2.0, 3.0]
	y = [ 1.0, 1.0, 1.0]
	r = np.array([ 2.0, 3.0, 4.0])
	print(polyfit2d(x, y, r, 2))