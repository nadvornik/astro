import cv2
import numpy as np
#from scipy.optimize import curve_fit



centroid_mat_cache = {}
def centroid(a):
	h, w = a.shape
	key = "%d %d" % a.shape
	if key not in centroid_mat_cache:
		x = np.arange(0, w, dtype = np.float32) - w / 2.0 + 0.5
		y = np.arange(0, h, dtype = np.float32) - h / 2.0 + 0.5
		mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))
		centroid_mat_x, centroid_mat_y = np.meshgrid(x, y)
		centroid_mat_cache[key] = (centroid_mat_x * mask, centroid_mat_y * mask)
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

	m[(np.isinf(m))] = 10000
	m[(np.isnan(m))] = 0

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
		bg = np.median(a[(mask == 0)])
		a = cv2.subtract(a, bg, dtype=cv2.CV_32FC1)
	
	s = cv2.sumElems(cv2.multiply(a,  mask, dtype=cv2.CV_32FC1))[0]
	if s == 0.0:
		return h / 2
	r = cv2.sumElems(cv2.multiply(a,  mat, dtype=cv2.CV_32FC1))[0] / s
	return r

ell_mat_cache = {}
def fit_ellipse(a):
	h, w = a.shape
	key = "%d %d" % a.shape
	if key not in ell_mat_cache:
		x = np.arange(0, w, dtype = np.float32) - w / 2.0 + 0.5
		y = np.arange(0, h, dtype = np.float32) - h / 2.0 + 0.5
		mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))
		ell_mat_x, ell_mat_y = np.meshgrid(x, y)
		ell_mat_x2 = ell_mat_x ** 2
		ell_mat_y2 = ell_mat_y ** 2
		ell_mat_xy = ell_mat_x * ell_mat_y
		ell_mat_cache[key] = (ell_mat_x2 * mask, ell_mat_y2 * mask, ell_mat_xy * mask)
	else:
		(ell_mat_x2, ell_mat_y2, ell_mat_xy) = ell_mat_cache[key]
		
	s = np.sum(a)
	if s == 0.0:
		return np.array([0.0, 0.0]), np.array([(0.0, 0.0), (0.0, 0.0)])
	vx = cv2.sumElems(cv2.multiply(a, ell_mat_x2, dtype=cv2.CV_32FC1))[0] / s
	vy = cv2.sumElems(cv2.multiply(a, ell_mat_y2, dtype=cv2.CV_32FC1))[0] / s
	cov = -cv2.sumElems(cv2.multiply(a, ell_mat_xy, dtype=cv2.CV_32FC1))[0] / s
	covmat = np.array([[vx, cov], [cov, vy]])
	w, v = np.linalg.eig(covmat)
	w **= 0.5
	return w, v


def gaussian2d((y, x), my, mx, sig, mag, shift):
	r2 = (x - mx) ** 2 + (y - my) ** 2
	return (np.exp(-r2 / (2 * sig * sig)) * mag + shift)

#def get_fwhm(a):
#	x0 = np.arange(a.shape[1])
#	y0 = np.arange(a.shape[0])
#	y, x = np.meshgrid(y0, x0)
#	yx = (y.ravel(), x.ravel())
#	z = a.ravel()
#	mx = a.shape[1] / 2.0
#	my = a.shape[0] / 2.0
#
#	shift = np.amin(z)
#	mag = np.amax(z) - shift
#	#print yx
#	try:
#		popt, pcov = curve_fit(gaussian2d, yx, z, p0 = [my, mx, 1, mag, shift], bounds = ([my - 3, mx - 3, 0, 0, 0], [my + 3, mx + 3, np.inf, np.inf, np.inf]))
#		print popt
#		return popt[2] * 2.355
#	except:
#		return 1.0

if __name__ == "__main__":

	I = np.array([  [ 0,   1,   0,   0,   0  , 0],
			[ 0,   1,   1,   0,   0  , 0],
			[ 0,   1,   1,   0,   0  , 0],
			[ 0,   0,   1.0, 1.0, 0  , 0],
			[ 0,   0,   1.0, 0,   0  , 0],
			[ 0,   0,   0,   0,   0  , 0],
			[ 0,   0,   0,   0,   0  , 0],
			])

	print I
	print sym_center(I)
	print centroid(I)
	print hfr(I)
	#print fwhm(I)
	
