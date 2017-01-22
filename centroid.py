import cv2
import numpy as np




centroid_mat_cache = {}
def centroid(a):
	h, w = a.shape
	key = "%d %d" % a.shape
	if key not in centroid_mat_cache:
		x = np.arange(0, w, dtype = np.float32) - w / 2.0 + 0.5
		y = np.arange(0, h, dtype = np.float32) - h / 2.0 + 0.5
		centroid_mat_x, centroid_mat_y = np.meshgrid(x, y)
		centroid_mat_cache[key] = (centroid_mat_x, centroid_mat_y)
	else:
		(centroid_mat_x, centroid_mat_y) = centroid_mat_cache[key]
		
	s = np.sum(a)
	if s == 0.0:
		return 0, 0
	x = cv2.sumElems(cv2.multiply(a, centroid_mat_x, dtype=cv2.CV_32FC1))[0] / s
	y = cv2.sumElems(cv2.multiply(a, centroid_mat_y, dtype=cv2.CV_32FC1))[0] / s
	return x, y
	

def centerfit(m, b, w):
        wm2p1 = cv2.divide(w, m*m + 1, dtype=cv2.CV_32FC1)
        sw  = np.sum(wm2p1)
        smmw = np.sum(m * m * wm2p1)
        smw  = np.sum(m * wm2p1)
        smbw = np.sum(m * b * wm2p1)
        sbw  = np.sum(b * wm2p1)
        det = smw*smw - smmw*sw
        if det == 0.0:
		return 0.0, 0.0
        xc = (smbw*sw - smw*sbw)/det; 
        yc = (smbw*smw - smmw*sbw)/det;
        if np.isnan(xc) or np.isnan(yc):
        	return 0.0, 0.0
        return xc, yc



def sym_center(I):
	I = np.array(I, dtype = np.float64)
	h,w = I.shape
	x = np.arange(0.5, w - 1) - (w - 1) / 2.0
	y = np.arange(0.5, h - 1) - (h - 1) / 2.0
	xm, ym = np.meshgrid(x, y)
	
	ru = I[1:, 1:] - I[:-1, :-1]
	rv = I[1:, :-1] - I[:-1, 1:]
	
	ru = cv2.blur(ru, (3,3))
	rv = cv2.blur(rv, (3,3))
	
	r2 = ru * ru + rv * rv
	rcx, rcy = centroid(r2)
	w = r2 / ((xm - rcx) **2 + (ym - rcy) ** 2 + 0.00001)**0.5

	m = cv2.divide(ru + rv, ru - rv)
	m[np.where(np.isinf(m))] = 10000

	b = ym - m*xm
	return centerfit(m, b, w)


hfr_mat_cache = {}
def hfr(a, sub_bg = False):
	h, w = a.shape
	key = "%d %d" % a.shape
	if key not in hfr_mat_cache:
		x = np.arange(0, w, dtype = np.float32) - w / 2.0 + 0.5
		y = np.arange(0, h, dtype = np.float32) - h / 2.0 + 0.5
		mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))
		xx, yy = np.meshgrid(x, y)
		mat = cv2.multiply((xx**2 + yy**2)**0.5, mask, dtype=cv2.CV_32FC1)
		hfr_mat_cache[key] = (mat, mask)
	else:
		(mat, mask) = hfr_mat_cache[key]
	
	if sub_bg:
		bg = np.median(a[np.where(mask == 0)])
		a = cv2.subtract(a, bg)
	
	s = cv2.sumElems(cv2.multiply(a,  mask, dtype=cv2.CV_32FC1))[0]
	if s == 0.0:
		return h / 2
	r = cv2.sumElems(cv2.multiply(a,  mat, dtype=cv2.CV_32FC1))[0] / s
	return r


if __name__ == "__main__":

	I = np.array([  [ 0,   0,   0,   0,   0  , 0],
			[ 0,   0,   0,   0,   0  , 0],
			[ 0,   0,   0,   0,   0  , 0],
			[ 0,   0,   1.0, 1.0, 0  , 0],
			[ 0,   0,   1.0, 1.0, 0  , 0],
			[ 0,   0,   0,   0,   0  , 0],
			[ 0,   0,   0,   0,   0  , 0],
			])

	print I
	print sym_center(I)
	print centroid(I)
	print hfr(I)
