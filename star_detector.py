import numpy as np
import cv2
import threading
from centroid import centroid, sym_center
import sys
import logging
log = logging.getLogger()


class MaxDetector(threading.Thread):
	def __init__(self, img, d, n, y1, y2, no_over = False):
		threading.Thread.__init__(self)
		self.d = d
		self.n = n
		self.y1 = y1
		self.y2 = y2
		
		self.y1e = max(0, y1 - d)
		self.y2e = min(img.shape[0], y2 + d)
		
		self.y1e0 = y1 - self.y1e
		self.y2e0 = y2 - self.y1e
		
		self.img = img
		self.no_over = no_over
		self.found = []
		
	
	def run(self):
		(h, w) = self.img.shape
		imge = np.array(self.img[self.y1e:self.y2e, : ], dtype = np.float32)
	
		imge = cv2.GaussianBlur(imge, (9, 9), 0)


		dilkernel = np.ones((self.d,self.d),np.uint8)
		dil = cv2.dilate(imge, dilkernel)
		img = imge[self.y1e0:self.y2e0, : ]
		dil = dil[self.y1e0:self.y2e0, : ]
		
		locmax = np.where(img >= dil)
		valmax = img[locmax]

		if self.no_over:
			maxv = np.amax(valmax) * 0.8
			valmax[valmax > maxv] = 0.0

		ordmax = np.argsort(valmax)[::-1]
		ordmax = ordmax[:self.n]

		
		centroid_size = 7
	
		for (y, x, v) in zip(locmax[0][ordmax], locmax[1][ordmax], valmax[ordmax]):
			if (v <= 0.0):
				break
			if (x < centroid_size):
				continue
			if (y + self.y1 < centroid_size):
				continue
			if (x > w - centroid_size - 1):
				continue
			if (y + self.y1 > h - centroid_size - 1):
				continue
			xs, ys = sym_center(imge[y + self.y1e0 - centroid_size : y + self.y1e0 + centroid_size + 1, x - centroid_size : x + centroid_size + 1])
			#print "centroid", xs, ys, xs2, ys2
			
			self.found.append((y + self.y1 + ys, x + xs, v))

	

def find_max(img, d, n = 40, no_over = False):
	(h, w) = img.shape
	par = 4
	step = (h + par - 1) / par
	mds = []
	joined = []
	for y in xrange(0, h, step):
		try:
			md = MaxDetector(img, d, n / par + 1, y, min(y + step, h), no_over)
			#md.run()
			md.start()
			mds.append(md)
		except:
			log.exception('Unexpected error')

	for md in mds:
		try:
			md.join()
			joined += md.found
		except:
			log.exception('Unexpected error')
	if len(joined) == 0:
		return []

	joined = np.array(joined)
	ordmax = np.argsort(joined[:, 2])[::-1]
	ordmax = ordmax[:n]
	joined = joined[ordmax]
	
	return joined

def centroid_list(im, pt0, off):
	(h, w) = im.shape

	pt = []
	match = []

	centroid_size = 10
	
	for i, (y, x, v) in enumerate(pt0):
		x = int(x + off[1] + 0.5)
		y = int(y + off[0] + 0.5)
		
		if (x < centroid_size):
			continue
		if (y < centroid_size):
			continue
		if (x > w - centroid_size - 1):
			continue
		if (y > h - centroid_size - 1):
			continue
		cm = np.array(im[y - centroid_size : y + centroid_size + 1, x - centroid_size : x + centroid_size + 1], dtype = np.float32)
		
		xs, ys = sym_center(cm)
		#print "xs, ys", xs, ys
		if abs(xs) > 5:
			continue
		if abs(ys) > 5:
			continue
			
		#print "centroid", v, cm[centroid_size + ys, centroid_size + xs], mean, stddev
		
		cm = cv2.GaussianBlur(cm, (9, 9), 0)
		mean, stddev = cv2.meanStdDev(cm)
		if cm[int(centroid_size + ys + 0.5), int(centroid_size + xs + 0.5)] < mean + stddev * 3:
			continue
		
		i2 = len(pt)
		pt.append((y + ys, x + xs, v))
		match.append([i, i2])
	return match_take(pt0, pt, match)

def centroid_mean(im, pt0, off):
	(h, w) = im.shape

	mean = []

	centroid_size = 10
	
	for i, (y, x, v) in enumerate(pt0):
		x = int(x + off[1] + 0.5)
		y = int(y + off[0] + 0.5)
		
		if (x < centroid_size):
			continue
		if (y < centroid_size):
			continue
		if (x > w - centroid_size - 1):
			continue
		if (y > h - centroid_size - 1):
			continue
		cm = np.array(im[y - centroid_size : y + centroid_size + 1, x - centroid_size : x + centroid_size + 1], dtype = np.float32)
		
		mean.append(cm)
	
	mean = np.mean(mean, axis = 0)
		
	xs, ys = sym_center(mean)
	return (ys, xs)




def match_take(pt1, pt2, match, ord1 = None, ord2 = None):
	match = np.array(match)
	if match.shape[0] == 0:
		return np.array([]), np.array([]), np.array([])
	
	pt1m = np.array(np.take(pt1, match[:, 0], axis=0), np.float)
	pt2m = np.array(np.take(pt2, match[:, 1], axis=0), np.float)
	
	if ord1 is not None:
		match[:, 0] = ord1[match[:, 0]]
	if ord2 is not None:
		match[:, 1] = ord2[match[:, 1]]

	return pt1m, pt2m, match

def check_drift(pt1, pt2, match, maxdif, maxdrift, off):
	pt1m, pt2m, match = match_take(pt1, pt2, match)
	
	dist = pt2m[:, 0:2] - pt1m[:, 0:2]
	med = np.median(dist, axis = 0)
	dif = np.max(np.abs(dist - [[med] * dist.shape[0]] ))
	if (dif > maxdif):
		return False
	drift = np.linalg.norm(dist[0, 0:2] - off)
	return drift < maxdrift
		

def find_nearest(array, val):
	diff = np.abs(array.flatten() - val)
	idx = diff.argmin()
	return np.unravel_index(idx, array.shape), diff[idx]

def pairwise_dist(x):
	b = np.dot(x, x.T)
	q = np.diag(b)[:, None]
	return np.sqrt(q + q.T - 2 * b)

def match_triangle(pt1, pt2, maxdif = 5.0, maxdrift = 20, off = (0.0, 0.0), n_max = 12, n_stop = 4):
	if len(pt1) == 0 or len(pt2) == 0:
		return match_take(pt1, pt2, [])
	
	ord1 = np.argsort(pt1[:, 2])[::-1]
	ord2 = np.argsort(pt2[:, 2])[::-1]
	
	pt1s = pt1[ord1][:min(n_max, len(ord1))]
	pt2s = pt2[ord2][:min(n_max, len(ord2))]
	
	dist1 = pairwise_dist(pt1s[:, 0:2])
	dist2 = pairwise_dist(pt2s[:, 0:2])
	
	bestmatch = []

	for a1 in xrange(0, len(pt1) - 2):
		for b1 in xrange(a1 + 1, len(pt1s) - 1):
			#print a1, b1, len(bestmatch)
			ab1 = dist1[a1, b1]
			((a2, b2), dif) = find_nearest(dist2, ab1)
			if dif > maxdif:
				continue
			match = []
			for c1 in xrange(b1 + 1, len(pt1s)):
				ac1 = dist1[a1, c1]
				bc1 = dist1[b1, c1]
				
				((c2_1,), dif1) = find_nearest(dist2[a2], ac1)
				((c2_2,), dif2) = find_nearest(dist2[b2], bc1)
				if c2_1 == c2_2 and dif1 < maxdif and dif2 < maxdif:
					#print "  match c1", a1, b1, c1, ac1, bc1, c2_1, c2_2, dif1, dif2
					match = [[a1, a2], [b1, b2], [c1, c2_1]]
					c2 = c2_1
					break

				((c2_1,), dif1) = find_nearest(dist2[a2], bc1)
				((c2_2,), dif2) = find_nearest(dist2[b2], ac1)
				if c2_1 == c2_2 and dif1 < maxdif and dif2 < maxdif:
					#print "  match c2", a1, b1, c1, ac1, bc1, c2_1, c2_2, dif1, dif2
					match = [[a1, b2], [b1, a2], [c1, c2_1]]
					tmp = a2
					a2 = b2
					b2 = tmp
					c2 = c2_1
					break
			
			for d1 in xrange(c1 + 1, len(pt1s)):
				ad1 = dist1[a1, d1]
				bd1 = dist1[b1, d1]
				cd1 = dist1[c1, d1]
				
				((d2_1,), dif1) = find_nearest(dist2[a2], ad1)
				((d2_2,), dif2) = find_nearest(dist2[b2], bd1)
				((d2_3,), dif3) = find_nearest(dist2[c2], cd1)
				if d2_1 == d2_2 and d2_2 == d2_3 and dif1 < maxdif and dif2 < maxdif and dif3 < maxdif:
					match.append([d1, d2_1])

			if len(match) == 3 and len(bestmatch) < 3 and check_drift(pt1s, pt2s, match, maxdif, maxdrift, off):
				bestmatch = match
			
			if len(match) >= n_stop:
				# 2 triangles are enough with default n_stop = 4
				return match_take(pt1s, pt2s, match, ord1, ord2)

			if len(bestmatch) < 2:
				if check_drift(pt1s, pt2s, [[a1, a2], [b1, b2]], maxdif, maxdrift, off):
					bestmatch = [[a1, a2], [b1, b2]]
				elif check_drift(pt1s, pt2s, [[a1, b2], [b1, a2]], maxdif, maxdrift, off):
					bestmatch = [[a1, b2], [b1, a2]]

			if len(match) > len(bestmatch):
				bestmatch = match
	
	if len(bestmatch) >= 4:
		return match_take(pt1s, pt2s, bestmatch, ord1, ord2)
	
	if len(bestmatch) == 0 and len(pt1s) > 0 and len(pt2s) > 0:
		for i in xrange(0, min(3, len(pt1s))):
			for j in xrange(0, min(3, len(pt2s))):
				if check_drift(pt1s, pt2s, [[i, j]], maxdif, maxdrift, off):
					bestmatch.append([i, j])
	return match_take(pt1s, pt2s, bestmatch, ord1, ord2)
	

def match_closest(pt1, pt2, d, off = (0.0, 0.0), M = None):
	if len(pt1) == 0 or len(pt2) == 0:
		return match_take(pt1, pt2, [])
	
	if M is None:
		M = np.matrix(np.concatenate((np.array([[1.0, 0], [0, 1.0]]), np.array([off]))))
	
	pt1t = np.hstack(( np.insert(pt1[:, 0:2], 2, 1.0, axis=1).dot(M).A , pt1[:, 2].reshape((-1,1)) ))
	ord2 = np.argsort(pt2[:, 0])
	pt2s = pt2[ord2]
	match = []
	l = len(pt2s)
	for i1, (y1, x1, flux1) in enumerate(pt1t):
		i2 = np.searchsorted(pt2s[:, 0], y1)
		closest_dist = d ** 2;
		closest_idx = -1
		ii2 = i2;
		while (ii2 >=0 and ii2 < l):
			(y2, x2, flux2) = pt2s[ii2]
			if (y2 < y1 - d):
				break
			dist = (y1 - y2) ** 2 + (x1 - x2) ** 2
			if (dist < closest_dist):
				closest_dist = dist
				closest_idx = ii2
			ii2 = ii2 - 1


		ii2 = i2;
		while (ii2 >=0 and ii2 < l):
			(y2, x2, flux2) = pt2s[ii2]
			if (y2 > y1 + d):
				break
			dist = (y1 - y2) ** 2 + (x1 - x2) ** 2
			if (dist < closest_dist):
				closest_dist = dist
				closest_idx = ii2
			ii2 = ii2 + 1

		if (closest_idx >= 0):
			match.append((i1, ord2[closest_idx]))
	return match_take(pt1, pt2, match)


def avg_pt(pt1m, pt2m, noise = 2):
	if pt1m.shape[0] > 1:
		dif = pt2m[:, 0:2] - pt1m[:, 0:2]
		weights = pt2m[:, 2] * pt1m[:, 2]
		sumw = np.sum(weights)
		if sumw > 0:
			v = np.average(dif, axis = 0, weights = weights)
			difdif2 = np.sum((dif - v)**2, axis = 1)
			var = np.sum(difdif2 * weights) / sumw
			weights[(difdif2 > var * noise**2)] = 0.0
			v = np.average(dif, axis = 0, weights = weights)
			return v, weights
	elif pt1m.shape[0] == 1:
		v = (pt2m - pt1m)[0, 0:2]
		weights = np.sqrt(pt2m[:, 2] * pt1m[:, 2])
		return v, weights
	
	v = np.array([0.0, 0.0])
	weights = np.array([0.0])
	return v, weights


def pt_translation(pt1, pt2, weights):
	t = pt2[:, 0:2] - pt1[:, 0:2]
	return np.matrix([[1., 0], 
	                  [0, 1.],
	                  np.average(t, axis = 0, weights = weights)])

def pt_translation_scale(pt1, pt2, weights):
	pt1 = pt1.reshape((-1, 2))
	pt2 = pt2.reshape((-1, 2))
	c1 = np.average(pt1[:, 0:2], axis = 0, weights = weights)
	c2 = np.average(pt2[:, 0:2], axis = 0, weights = weights)

	centered_pt1 = pt1[:, 0:2] - c1
	centered_pt2 = pt2[:, 0:2] - c2
	
	s, z = np.polyfit(np.concatenate((centered_pt1[:, 0], centered_pt1[:, 1])),
	                  np.concatenate((centered_pt2[:, 0], centered_pt2[:, 1])), 
	                  1, 
	                  w = np.concatenate((weights, weights)))

	t = c2 - c1 * float(s)
	
	m = np.concatenate((np.array([[float(s), 0], [0, float(s)]]), t.reshape((1,2))))
	m = np.matrix(m)
	#print m
	return m

def pt_translation_rotate(pt1, pt2, weights):
	c1 = np.average(pt1[:, 0:2], axis = 0, weights = weights)
	c2 = np.average(pt2[:, 0:2], axis = 0, weights = weights)

	centered_pt1 = pt1[:, 0:2] - c1
	centered_pt2 = pt2[:, 0:2] - c2
	
	
	c00 = np.array(centered_pt1[:,0]).reshape(-1)
	c10 = np.array(centered_pt2[:,0]).reshape(-1)
	c01 = np.array(centered_pt1[:,1]).reshape(-1)
	c11 = np.array(centered_pt2[:,1]).reshape(-1)
	
	cov = np.array([
	  [ np.average(c00 * c10, axis = 0, weights = weights),
	    np.average(c01 * c10, axis = 0, weights = weights)],
	  [ np.average(c00 * c11, axis = 0, weights = weights),
	    np.average(c01 * c11, axis = 0, weights = weights)]])
	w, u, vt = cv2.SVDecomp(cov)
	
	r = np.matrix(np.transpose(vt)).dot(np.transpose(u))
	t = c2 - c1 * r
	m = np.matrix(np.concatenate((r, t)))
	#print m
	return m

def pt_transform_opt(pt1m, pt2m, noise = 2, pt_func = pt_translation):
	if len(pt1m) == 0:
		return np.matrix([[1., 0], [0, 1.], [0, 0]]), []
	pt1m = np.array(pt1m)
	pt2m = np.array(pt2m)
	pt1 = pt1m[:, 0:2]
	pt2 = pt2m[:, 0:2]
	weights = pt2m[:, 2] * pt1m[:, 2] + 1
	sumw = np.sum(weights)
	
	if pt_func == pt_translation_scale and len(pt1m) < 2:
		pt_func = pt_translation

	if pt_func == pt_translation_rotate and len(pt1m) < 4:
		pt_func = pt_translation
	
	try:
		m = pt_func(pt1, pt2, weights)
	except:
		log.exception('Unexpected error')
		pt_func = pt_translation
		m = pt_func(pt1, pt2, weights)
	
	pt1t = np.insert(pt1, 2, 1.0, axis=1).dot(m).A
	
	d2 = np.sum((pt2 - pt1t)**2, axis = 1)
	var = np.sum(d2 * weights) / sumw
	weights[(d2 > var * noise**2)] = 1.0
	
	m = pt_func(pt1, pt2, weights)
	
	return m, weights
