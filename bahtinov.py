import numpy as np
import cv2
from smooth import smooth
from centroid import centroid, hfr
import logging


log = logging.getLogger()

def normalize(img):
        dst = np.empty_like(img)
        return cv2.normalize(img, dst, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

def intersect(l1, l2):
	(p, r) = np.array(l1[0]), np.array(l1[1])
	(q, s) = np.array(l2[0]), np.array(l2[1])
	
	crs = np.cross(r,s)
	if crs == 0:
		return None
	u = np.cross((q - p), r) / crs
	return q + u * s

def sig_dist(l, pt):
	p, v = l
	return (v[0]*pt[1] - v[1]*pt[0] + (p[0]*v[1]  - v[0] * p[1]))  / (v[0] ** 2 + v[1] ** 2) ** 0.5
		

def get_rect(img, size, c):
	x1 = c[1] - size[1] // 2
	y1 = c[0] - size[0] // 2
	x2 = x1 + size[1]
	y2 = y1 + size[0]
	
	offx = x1
	offy = y1
	
	clr = False
	if x1 < 0:
		x1 = 0
		clr = True
	if y1 < 0:
		y1 = 0
		clr = True
	if x2 > img.shape[1]:
		x2 = img.shape[1]
		clr = True
	if y2 > img.shape[0]:
		y2 = img.shape[0]
		clr = True

	if clr:
		res = np.zeros(size, dtype = img.dtype)
	else:
		res = np.empty(size, dtype = img.dtype)
	res[y1 - offy: y2 - offy, x1 - offx: x2 - offx] = img[y1:y2, x1:x2]
	return res 

class Bahtinov:
	def ba_skel(self, img):
		skel = np.zeros_like(img)
 
		element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
 
		while(np.any(img)):
			eroded = cv2.erode(img,element)
			temp = cv2.dilate(eroded,element)
			temp = cv2.subtract(img,temp)
			skel = cv2.bitwise_or(skel,temp)
			img = eroded
 		skel = cv2.dilate(skel,element)
		return skel

	def ba_center(self, img):
		bl = cv2.blur(img, (10, 10))
		bl = cv2.blur(bl, (10, 10))
		bl = cv2.blur(bl, (10, 10))
		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(bl)
		self.center = (maxLoc[1], maxLoc[0]) 
		self.roi_center = self.center
	
	def ba_roi(self, img):
		a = get_rect(img, (100, 100), self.center)
		self.radius = int(3 * hfr(a))
		log.info("roi hfr %d" % self.radius)

	def ba_angle(self, skel_pts):
		cov = np.cov(skel_pts)
		w, v = np.linalg.eig(cov)
	
		log.info("eig %s %s" % (w, v))
		if w[0] > w[1]:
			return -np.arctan2(v[0][0], v[0][1])
		else:
			return -np.arctan2(v[1][0], v[1][1])

	def ba_lines_coarse(self, skel_pts):
		hist_off = np.pi / 2 - self.angle
		skel_pts = skel_pts - np.array(self.roi_center)[:, np.newaxis]
		a_pts = (np.arctan2(skel_pts[0], skel_pts[1]) + hist_off) % np.pi
		log.info("skel pts %d" % len(skel_pts[0]))
		bins = 180 // 3
		hist, edges = np.histogram(a_pts, bins = bins, range = (0, np.pi))
		hist = smooth(hist, window_len = 5)
		log.info("hist %s" % hist)
		angles = []
		for i in range(3):
			mi = np.argmax(hist)
			v0 = hist[mi]
			hist[mi] = 0
			ii = mi - 1
			v = v0
			while ii >= 0 and hist[ii] < v:
				v = hist[ii]
				hist[ii] = 0
				ii -= 1
	
			ii = mi + 1
			v = v0
			while (ii < bins and hist[ii] < v):
				v = hist[ii]
				hist[ii] = 0
				ii += 1
			
			angles.append(np.pi * mi / bins)
			log.info("mi %d" % mi)

		angles = np.sort(angles)
		angles = (np.array(angles) - hist_off) % np.pi 
		
		log.info("angles %s", angles  / np.pi * 180)
		return [(self.roi_center, (np.sin(a), np.cos(a))) for a in angles]
	
		
	
	
	def ba_draw_line(self, img, l):
		(p, v) = l
		x1 = int(p[1] + 1000*(v[1]))
		y1 = int(p[0] + 1000*(v[0]))
		x2 = int(p[1] - 1000*(v[1]))
		y2 = int(p[0] - 1000*(v[0]))
	
	    	cv2.line(img,(x1,y1),(x2,y2),(255,),10)
	    	cv2.circle(img, (int(self.roi_center[1]), int(self.roi_center[0])), 5, (0,), 10)
	
	
	def ba_fit_line(self, img, l):
		(p, v) = l
	
		mask = np.zeros_like(img)
		self.ba_draw_line(mask, l)
		
	
		pts = np.where(mask)
		
		weights = np.array(img[pts], dtype=np.float32)
		
		
		dist2 = (v[0]*pts[1] - v[1]*pts[0] + (p[0]*v[1]  - v[0] * p[1])) ** 2 / (v[0] ** 2 + v[1] ** 2)
		dist = dist2 ** 0.5
		var = np.average(dist2, weights=weights)
		#log.info("sigma %f" % (var**0.5))
	
	
		weights /= dist + var
		
		pts = np.array(pts)
		
		r = np.sum((pts - np.array(self.roi_center)[:, np.newaxis]) ** 2, axis = 0) ** 0.5
		
		weights *= r
		
		A = np.ones((len(weights), 2), dtype=np.float32)
		A[:,0] = pts[self.x_axis]
		y = np.array(pts[1 - self.x_axis], dtype=np.float32)
	
	
		weights_s = weights**0.5
		Aw = A * weights_s[:, np.newaxis]
		yw = y * weights_s
	
		line = np.linalg.lstsq(Aw, yw)[0]
	
		if self.x_axis:
			return (line[1], 0.0), (line[0], 1.0)
		else:
			return (0.0, line[1]), (1.0, line[0])


	def fit_all(self, img):
		for l in range(3):
			for i in range(20):
				self.lines[l] = self.ba_fit_line(img, self.lines[l])

	def result(self):
		try:
			intr = intersect(self.lines[0], self.lines[2])
			if intr is None:
				return 0.0
			ret = sig_dist(self.lines[1], intr)
			log.info("bathinov %f" % ret)
			return ret
		except:
			log.exception('Unexpected error')
			return 0.0



	def prepare(self, img, bg = False):
		try:
			self.ba_center(img)
			mean, sigma = cv2.meanStdDev(img)
			if bg:
				bl = cv2.blur(img, (5, 5))
				bl = cv2.erode(bl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20)))
				bl = cv2.blur(bl, (30, 30))
				img = cv2.subtract(img, bl)

			skel = cv2.subtract(img, sigma)
			self.ba_roi(skel)

			skel = get_rect(skel, (self.radius * 4, self.radius * 4), self.center)
			self.roi_center = (self.radius * 2, self.radius * 2)
		
			skel = self.ba_skel(skel)

			skel_pts = np.array(np.where(skel))
			self.angle = self.ba_angle(skel_pts)
			log.info("main angle %d", self.angle / np.pi * 180)
			
			if self.angle > np.pi / 4 and self.angle < np.pi * 3 / 4:
				self.x_axis = 0
			else:
				self.x_axis = 1
		
			self.lines = self.ba_lines_coarse(skel_pts)

			img = get_rect(img, (self.radius * 4, self.radius * 4), self.center)

			self.fit_all(img)
			return True

		except:
			log.exception('Unexpected error')
			return False

	def update(self, img, bg = False):
		try:
			self.ba_center(img)

			if bg:
				bl = cv2.blur(img, (5, 5))
				bl = cv2.erode(bl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20)))
				bl = cv2.blur(bl, (30, 30))
				img = cv2.subtract(img, bl)

			img = get_rect(img, (self.radius * 4, self.radius * 4), self.center)
			self.fit_all(img)
			return True
		except:
			log.exception('Unexpected error')
			return False
	
	def plot(self, img):
		try:
			for i, (p, v) in enumerate(self.lines):
				v = np.array(v) / np.linalg.norm(v)
				r = self.radius * 2
				if i == 1:
					r *= 1.2
			
				x1 = int(self.center[1] + r*(v[1]))
				y1 = int(self.center[0] + r*(v[0]))
				x2 = int(self.center[1] - r*(v[1]))
				y2 = int(self.center[0] - r*(v[0]))

				cv2.line(img,(x1,y1),(x2,y2),(255,),1)
		except:
			log.exception('Unexpected error')
			
	

if __name__ == "__main__":
	logging.basicConfig(format="%(filename)s:%(lineno)d: %(message)s", level=logging.INFO)
	
	imgc = cv2.imread("b.jpg")
	#imgc = cv2.imread("ba1.jpg")
	#imgc = cv2.imread("Bhatinov.jpg")
	img = np.amin(imgc, axis = 2)
	
	b = Bahtinov()
	
	b.prepare(img, bg=True)
	print "result", b.result()
	b.plot(imgc)
	cv2.imshow("res", imgc)
	cv2.waitKey(0)
	

	imgc = cv2.imread("ba2.jpg")
	img = np.amin(imgc, axis = 2)
	b.update(img, bg=True)
	print "result", b.result()
	#cv2.waitKey(0)

	imgc = cv2.imread("ba3.jpg")
	img = np.amin(imgc, axis = 2)
	b.update(img, bg=True)
	print "result", b.result()

	#cv2.waitKey(0)
	
	
	
	