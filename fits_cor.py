import cv2
import numpy as np
import pyfits


def read_wcs(fn):
	hdulist = pyfits.open(fn)
	return hdulist[0].header

def get_image_info(wcs):
	return wcs['IMAGEW'], wcs['IMAGEH'], wcs['CRPIX1'], wcs['CRPIX2']

def get_coef(wcs, tab):
	order = wcs[tab + '_ORDER']
	c = np.zeros((order + 1, order + 1))
	for i in range(0, order + 1):
		for j in range(0, order + 1 - i):
			c[i, j] = wcs["%s_%d_%d" % (tab, i, j)]
	return c
	

def cor_map(wcs, orig_h):
	w, h, crx, cry = get_image_info(wcs)
	
	A = get_coef(wcs, 'A')
	B = get_coef(wcs, 'B')
	
#	mx = np.zeros((orig_h, w), np.float32)
#	my = np.zeros((orig_h, w), np.float32)
#	
#	for y0 in range(0, orig_h):
#		y = float(y0) / orig_h * h - cry
#		for x0 in range(0, w):
#			x = x0 - crx
#			cor_x = 0
#			for (p, q), c in np.ndenumerate(A):
#				cor_x += c * x**p * y ** q
#			mx[y0, x0] = x0 + cor_x
#			cor_y = 0
#			for (p, q), c in np.ndenumerate(B):
#				cor_y += c * x**p * y ** q
#			my[y0, x0] = y0 + cor_y + y / h * (orig_h - h)
	
	m = np.empty((orig_h, w, 2), np.float32)
	
	xr = np.arange(0, w, dtype=np.float32) - crx
	yr = np.arange(0, orig_h, dtype=np.float32) / orig_h * h - cry
	
	print xr
	print yr
	
	A[0, 0] += crx
	A[1, 0] += 1
	
	m[:,:,0] = np.polynomial.polynomial.polygrid2d(xr, yr, A).T

	B[0, 0] += cry
	B[0, 1] += 1 + float(orig_h - h) / h
	print B
	
	m[:,:,1] = np.polynomial.polynomial.polygrid2d(xr, yr, B).T * (float(orig_h) / h)
	

#	print "mx"
#	print mx
	print "m"
	print m[:,:,0]
	
	
	
#	print "my"
#	print my
	print "m"
	print m[:,:,1]
	
	return m
	

wcs = read_wcs("x1.wcs")
print get_image_info(wcs)
print get_coef(wcs, 'A')
print get_coef(wcs, 'B')

#img = cv2.imread("../guider1466281863.tif", cv2.IMREAD_UNCHANGED)
img = cv2.imread("navigator1467788659.tif", cv2.IMREAD_UNCHANGED)
#img = cv2.imread("xxx.tif", cv2.IMREAD_UNCHANGED)
print img
print img.shape

m = cor_map(wcs, 960)
np.save("oag_cor.npy", m)

img = cv2.remap(img, m, None, cv2.INTER_CUBIC)
cv2.imwrite("ooo.tif", img)
