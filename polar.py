#!/usr/bin/python

import numpy as np
from quat import Quaternion
from astrometry.util.util import Tan
import math
import cv2
import time
import os

from am import Plotter


def quat_axis_to_ra_dec(q):
	(w,x,y,z) = q.a
	print "xyz", x,y,z
	ra = math.degrees(math.atan2(y,x))
	dec = math.degrees(math.atan2(z, (x ** 2 + y ** 2)**0.5))
	return (ra, dec)

def ra_dec_to_xyz(rd):
	rd = np.deg2rad(rd)
	v = [ math.cos(rd[0]) * math.cos(rd[1]), math.sin(rd[0]) * math.cos(rd[1]), math.sin(rd[1]) ]
	return v

def julian_date(t = None):
	if t is None:
		t = time.time()
	return ( t / 86400.0 ) + 2440587.5;

def celestial_rot(t = None):
	jd = julian_date(t)
	jh = (jd + 0.5 - math.floor(jd + 0.5)) * 24
	jd0 = math.floor(jd + 0.5) - 0.5
	T = (jd0 - 2451545.0) / 36525
	S0 = (6.697374558 + 2400.05133691 * T + 0.0000258622 * T ** 2 - 0.0000000017 * T**3) % 24
	S = S0 + 1.0027379093 * jh
	rot = S * 15 + 15
	print "cel time", T, S0, jh, S
	return rot

def precession():
	T = (julian_date() - 2451545.0) / 36525
	e0 = 84381.406
	a1 = 5038.481507 * T - 1.0790069 * T**2 - 0.00114045 * T**3 + 0.000132851 * T**4 - 0.0000000951 * T**5
	a2 = e0 - 0.025754 * T + 0.0512623 * T**2 - 0.00772503 * T**3 - 0.000000467 * T**4 + 0.0000003337 * T**5
	a3 = 10.556403 * T - 2.3814292 * T**2 - 0.00121197 * T**3 + 0.000170663 * T**4 - 0.0000000560 * T**5

	#q = Quaternion([0., 0., a3 / 3600]) * Quaternion([-a2 / 3600, 0., 0.]) * Quaternion([0., 0., -a1 / 3600]) * Quaternion([e0 / 3600, 0., 0.])
	q = Quaternion([a3 / 3600, 0., 0.]) * Quaternion([0., 0., -a2 / 3600]) * Quaternion([-a1 / 3600, 0., 0.]) * Quaternion([0., 0., e0 / 3600])
	return q

class Polar:
	def __init__(self):
		self.pos = []
		self.t0 = None
		self.p2_from = None
		self.prec_q = precession()
		self.prec_ra, self.prec_dec = self.prec_q.inv().transform_ra_dec([0, 90])
		
	def add(self, ra, dec, roll, t):
		if self.t0 is None:
			self.t0 = t
		ha = (t - self.t0) / 240.0
		qha = self.prec_q * Quaternion([-ha, 0, 0]) / self.prec_q
		#print "qha", quat_axis_to_ra_dec(qha), "prec", self.prec_ra, self.prec_dec 
		#print Quaternion([ra, dec, roll]).to_euler(), (qha * Quaternion([ra, dec, roll])).to_euler()
		self.pos.append(qha * Quaternion([ra, dec, roll]))
		#self.pos.append(Quaternion([ra, dec, roll]))

	def tan_to_euler(self, tan, off=(0,0)):
		ra, dec = tan.radec_center()
		# the field moved by given offset pixels from the position in self.wcs
		(crpix1, crpix2) = tan.crpix
		ra, dec = tan.pixelxy2radec(crpix1 - off[1], crpix2 - off[0])

		cd11, cd12, cd21, cd22 = tan.cd
		
		det = cd11 * cd22 - cd12 * cd21
		if det >= 0:
			parity = 1.
		else:
			parity = -1.
		T = parity * cd11 + cd22
		A = parity * cd21 - cd12
		orient = math.degrees(math.atan2(A, T))
		#orient = math.degrees(math.atan2(cd21, cd11))
		
		return ra, dec, orient

	def add_tan(self, tan, t):
		ra, dec, orient = self.tan_to_euler(tan)
		print ra, dec, orient
		self.add(ra, dec, orient, t)
		print "added ", t


	def compute2(self, i = 0, j = None):
		if j is None:
			j = len(self.pos) - 1
		q1 = self.pos[i]
		q2 = self.pos[j]
		
		print q1.a
		print q2.a
		
		c = q2 / q1

		ra, dec = quat_axis_to_ra_dec(c)
		
		if dec < 0:
			dec = -dec
			ra -= 180
		if ra < 0.0 :
			ra += 360
		
		print ra, dec
		self.ra = ra
		self.dec = dec
		return True, ra, dec

	def compute(self):
		if len(self.pos) < 2:
			return False, None, None
		qa = np.array([p.a for p in self.pos])
		
		qamin = np.amin(qa, axis = 0)
		qamax = np.amax(qa, axis = 0)
		qarange = qamax - qamin
		ao = np.argsort(qarange) #axis order, ao[0], ao[1] are computed from ao[2] and ao[3]

		ones = np.ones(len(qa))
		
		A = np.column_stack((qa[:,ao[2]], qa[:,ao[3]]))
		res0 = np.linalg.lstsq(A, qa[:,ao[0]])[0]
		res1 = np.linalg.lstsq(A, qa[:,ao[1]])[0]
		
		#for q in qa:
		#	print q[0] * res2[0] + q[1] * res2[1] + res2[2]  - q[2] , q[0] * res3[0] + q[1] * res3[1] + res3[2]  - q[3]
		qa1 = np.zeros(4)
		qa1[ao[2]] = 1.
		qa1[ao[0]] = res0[0]
		qa1[ao[1]] = res1[0]
		
		qa2 = np.zeros(4)
		qa2[ao[3]] = 1.
		qa2[ao[0]] = res0[1]
		qa2[ao[1]] = res1[1]
		
		#print ao, qa1, qa2
		#q1 = Quaternion([1., 0., res2[0], res3[0]], normalize=True)
		#q2 = Quaternion([0., 1., res2[1], res3[1]], normalize=True)
		
		q1 = Quaternion(qa1, normalize=True)
		q2 = Quaternion(qa2, normalize=True)
		
		c = q2 / q1

		ra, dec = quat_axis_to_ra_dec(c)
		
		if dec < 0:
			dec = -dec
			ra -= 180
		if ra < 0.0 :
			ra += 360
		
		self.ra = ra
		self.dec = dec
		#print "rotation center", ra, dec
		return True, ra, dec

		
		#fig = plt.figure()
		#ax = fig.add_subplot(111, projection='3d')

		#ax.scatter(qa[:,0], qa[:,1], qa[:,2], c=qa[:,3], cmap=plt.hot())
		#ax.scatter(qa[:,0], qa[:,1], qa[:,3], c=qa[:,3], cmap=plt.hot())
		#ax.scatter(qa[:,0], qa[:,1], qa[:,0] * res2[0] + qa[:,1] * res2[1] + res2[2], c=qa[:,3], cmap=plt.hot())
		#plt.show()

	def plot(self):
		extra = []
		extra.append((self.ra, self.dec, "A"))
		
		extra.append((self.prec_ra, self.prec_dec, "P"))
		
		ra = 0.0 #self.ra
		dec = 90.0 #self.dec
		size = 2.0
		w = 640
		h = 640
		pixscale = size / w
	
		wcs = Tan(*[float(x) for x in [
			ra, dec, 0.5 + (w / 2.), 0.5 + (h / 2.),
			-pixscale, 0., 0., -pixscale, w, h,
			]])
		plot = Plotter(wcs)
		return plot.plot(extra=extra)

	def transform_ra_dec_list(self, l):
		t = Quaternion.from_ra_dec_pair([self.ra, self.dec], [self.prec_ra, self.prec_dec])
		print "transform_ra_dec_list", t.transform_ra_dec([self.ra, self.dec])
		res = []
		for rd in l:
			res.append(t.transform_ra_dec(rd))
		return res

	def phase2_set_ref_pos(self, ra, dec, roll):
		self.p2_from = Quaternion([ra, dec, roll])
		self.ref_ra = self.ra
		self.ref_dec = self.dec
		
	def phase2_set_ref_tan(self, tan):
		ra, dec, orient = self.tan_to_euler(tan)
		self.phase2_set_ref_pos(ra, dec, orient)
	
	def phase2_set_pos(self, ra, dec, roll):
		if self.p2_from is None:
			return self.phase2_set_ref_pos(ra, dec, roll)
			
		pos = Quaternion([ra, dec, roll])
		t = pos / self.p2_from
		self.ra, self.dec = t.transform_ra_dec([self.ref_ra, self.ref_dec])
	
	def phase2_set_tan(self, tan, off = (0, 0)):
		ra, dec, orient = self.tan_to_euler(tan, off)
		self.phase2_set_pos(ra, dec, orient)


	def plot2(self, size = 540, area = 0.1):
		ha = celestial_rot()
		qha = Quaternion([90-ha, 0, 0])
		
		img = np.zeros((size, size, 3), dtype=np.uint8)
		c = size / 2
		scale = size / area
		
		t = Quaternion.from_ra_dec_pair([self.ra, self.dec], [self.prec_ra, self.prec_dec])

		
		polaris = [37.9529,  89.2642]
		
		polaris_target = self.prec_q.transform_ra_dec(polaris)
		prec = t.transform_ra_dec([0, 90])
		polaris_real = t.transform_ra_dec(polaris_target)

		polaris_target = qha.transform_ra_dec(polaris_target)
		prec = qha.transform_ra_dec(prec)
		polaris_real = qha.transform_ra_dec(polaris_real)

		polaris_target_xyz = ra_dec_to_xyz(polaris_target)
		polaris_r = (polaris_target_xyz[0] ** 2 + polaris_target_xyz[1] ** 2)**0.5
		prec_xyz = ra_dec_to_xyz(prec)
		polaris_real_xyz = ra_dec_to_xyz(polaris_real)

		cv2.circle(img, (c,c), int(polaris_r * scale), (0, 255, 0), 1)
		for i in range (0, 24):
			a = np.deg2rad([i * 360.0 / 24.0])
			sa = math.sin(a)
			ca = math.cos(a)
			cv2.line(img, (int(c + sa * polaris_r * scale), int(c + ca * polaris_r * scale)), (int(c + sa * (polaris_r * scale + 8)), int(c + ca * (polaris_r * scale + 8))), (0, 255, 0), 1)
			
		cv2.circle(img, (int(c + polaris_target_xyz[0] * scale), int(c + polaris_target_xyz[1] * scale)), 4, (0, 255, 0), 2)
		
		cv2.line(img, (0, c), (size, c), (0, 255, 0), 1)
		cv2.line(img, (c, 0), (c, size), (0, 255, 0), 1)
		
		
		cv2.circle(img, (int(c + prec_xyz[0] * scale), int(c + prec_xyz[1] * scale)), 4, (255, 255, 255), 2)
		cv2.circle(img, (int(c + polaris_real_xyz[0] * scale), int(c + polaris_real_xyz[1] * scale)), 4, (255, 255, 255), 2)
		
		pole_dist = (prec_xyz[0] ** 2 + prec_xyz[1] ** 2) ** 0.5
		if pole_dist >= area / 2:
			cv2.putText(img, "%0.1fdeg" % (90 - prec[1]), (int(c + prec_xyz[0] / pole_dist * area / 5 * scale - 50), int(c + prec_xyz[1] / pole_dist * area / 5 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
			cv2.arrowedLine(img, (int(c + prec_xyz[0] / pole_dist * area / 3 * scale), int(c + prec_xyz[1] / pole_dist * area / 3 * scale)), (int(c + prec_xyz[0] / pole_dist * area / 2 * scale), int(c + prec_xyz[1] / pole_dist * area / 2 * scale)), (255, 255, 255), 2)

		return img

if __name__ == "__main__":
	
#	import random
#	import sys
#	for tra in range(0, 360, 10):
#		for tdec in range(5, 95, 10):
#			for troll in range (-170, 170, 10):
#				q = Quaternion([153, 44,73])
#				rot = Quaternion([tra, tdec, troll])
#				p = Polar()
#				for i in range(0, 20):
#					q = rot * q
#					rra, rdec, rroll = q.to_euler()
#					p.add(rra + random.random() * 2 - 1, rdec + random.random() * 2 - 1, rroll + random.random() * 2 - 1, 0)
#					#p.add(rra, rdec, rroll, 0)
#	
#				print tra, tdec, troll,
#				res = np.array((p.compute()))[1:3]
#				ra, dec = quat_axis_to_ra_dec(rot)
#				
#				if dec * p.pos[0].to_euler()[1] < 0:
#					dec = -dec
#					ra -= 180
#				if ra < 0.0 :
#					ra += 360
#				orig = np.array([ra, dec])
#				print  np.linalg.norm(res - orig)

	extra = [(0.0, 90.0, "z")]

	for r in [(169, 243), # 0x
	          (274, 351), # 1x
	          (377, 450), # 2x
	          (457, 484), # 6x (457)485-516
	          (274, 516) #all
	          ]:
		p = Polar()
		for i in range(*r):
			tan = Tan('converted/IMG_%04d.wcs' % (i),0)
			t = os.path.getmtime('converted/IMG_%04d.wcs' % (i))
			#tan = Tan('log_%d.wcs' % (i),0)
			p.add_tan(tan, t)
#		for i in range(1, len(p.pos)):
#			ra, dec = p.compute2(0,i)
#			extra.append((ra, dec, ""))
		res, ra, dec = p.compute()
		extra.append((ra, dec, "1"))
		

	for r in [(124, 628), # 0x
	          (748, 872), # 1x
	          (1017, 1418), # 2x
	          (1424, 1432), # 6x (457)485-516
	          (748, 1432) #all
	          ]:
		p = Polar()
		for i in range(*r):
			tan = Tan('converted/log_%d.wcs' % (i),0)
			t = os.path.getmtime('converted/log_%d.wcs' % (i))
			#tan = Tan('log_%d.wcs' % (i),0)
			p.add_tan(tan, t)
#		for i in range(1, len(p.pos) - 1):
#			ra, dec = p.compute2(0,i)
#			extra.append((ra, dec, ""))
#			ra, dec = p.compute2(i,len(p.pos) - 1)
#			extra.append((ra, dec, ""))
		res, ra, dec = p.compute()
		extra.append((ra, dec, "2"))

	
	p = Polar()
	for t in [1446322430,
	          1446322444,
	          1446322479,
	          1446322512,
	          1446322563,
	          1446322596]:
		tan = Tan('t/capture_gphoto%d.wcs' % t , 0)
		p.add_tan(tan, t)
	res, ra, dec = p.compute()
	extra.append((ra, dec, "gphoto"))
	
	p = Polar()
	for t in [
	          1446322479,
	          1446322511,
	          1446322562,
	          1446322596]:
		tan = Tan('t/capture_v4l%d.wcs' % t , 0)
		p.add_tan(tan, t)
	res, ra, dec = p.compute()
	extra.append((ra, dec, "v4l2"))
	cv2.imshow("polar", p.plot2())
	
	p.transform_ra_dec_list([])
	
	#for i in range(100, 1400,500):
	#	pp = Polar()
	#	for j in range(i, i+2000):
	#		tan = Tan('log_%d.wcs' % (j),0)
	#		pp.add_tan(tan, 0)
	#	ra, dec = pp.compute()
	#	extra.append((ra, dec, ""))

	
	#for i in range (20, 50):
	#	ra, dec = p.compute2(0, i)
	#	extra.append((ra, dec, ""))
	qp = precession()
	ra, dec = qp.inv().transform_ra_dec([0, 90])
	extra.append((ra, dec, "p"))
	print "precession", ra, dec
	
	
	ra = 0.
	dec = 90.
	size = 70.0
	w = 800
	h = 800
	pixscale = size / w
	
	wcs = Tan(*[float(x) for x in [
		ra, dec, 0.5 + (w / 2.), 0.5 + (h / 2.),
		-pixscale, 0., 0., pixscale, w, h,
		]])
	plot = Plotter(wcs)
	cv2.imshow("plot", plot.plot(extra=extra))
	cv2.waitKey(0)