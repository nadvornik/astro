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
	print x,y,z
	ra = math.degrees(math.atan2(y,x))
	dec = math.degrees(math.atan2(z, (x ** 2 + y ** 2)**0.5))
	return (ra, dec)

def precession():
	T = (time.time() - time.mktime((2000, 1, 1, 12, 0, 0, 0, 0, 0))) / 3600 / 24 / 36525
	print T
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
		print "qha", quat_axis_to_ra_dec(qha), "prec", self.prec_ra, self.prec_dec 
		print Quaternion([ra, dec, roll]).to_euler(), (qha * Quaternion([ra, dec, roll])).to_euler()
		self.pos.append(qha * Quaternion([ra, dec, roll]))

	def tan_to_euler(self, tan):
		ra, dec = tan.radec_center()
		cd11, cd12, cd21, cd22 = tan.cd
		
		det = cd11 * cd22 - cd12 * cd21
		if det >= 0:
			parity = 1.
		else:
			parity = -1.
		T = parity * cd11 + cd22
		A = parity * cd21 - cd12
		orient = -math.degrees(math.atan2(A, T))
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
		
		print q1.q
		print q2.q
		
		c = q2 / q1

		ra, dec = quat_axis_to_ra_dec(c)
		
		if dec * self.pos[i].to_euler()[1] < 0:
			dec = -dec
			ra -= 180
		if ra < 0.0 :
			ra += 360
		
		print ra, dec
		return ra, dec

	def compute(self):
		if len(self.pos) < 4:
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
		
		#q1 = Quaternion([1., 0., res2[0], res3[0]], normalize=True)
		#q2 = Quaternion([0., 1., res2[1], res3[1]], normalize=True)
		
		q1 = Quaternion(qa1, normalize=True)
		q2 = Quaternion(qa2, normalize=True)
		
		c = q2 / q1

		ra, dec = quat_axis_to_ra_dec(c)
		
		if dec * self.pos[0].to_euler()[1] < 0:
			dec = -dec
			ra -= 180
		if ra < 0.0 :
			ra += 360
		
		self.ra = ra
		self.dec = dec
		print "rotation center", ra, dec
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
		size = 7.0
		w = 640
		h = 640
		pixscale = size / w
	
		wcs = Tan(*[float(x) for x in [
			ra, dec, 0.5 + (w / 2.), 0.5 + (h / 2.),
			-pixscale, 0., 0., pixscale, w, h,
			]])
		plot = Plotter(wcs)
		return plot.plot(extra=extra, grid = False)

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
		
	def pase2_set_ref_tan(self, tan):
		ra, dec, orient = self.tan_to_euler(tan)
		self.phase2_set_ref_pos(ra, dec, orient)
	
	def phase2_set_pos(self, ra, dec, roll):
		if self.p2_from is None:
			return self.phase2_set_ref_pos(ra, dec, roll)
			
		pos = Quaternion([ra, dec, roll])
		t = pos / self.p2_from
		self.ra, self.dec = t.transform_ra_dec([self.ref_ra, self.ref_dec])
	
	def phase2_set_tan(self, tan):
		ra, dec, orient = self.tan_to_euler(tan)
		self.phase2_set_pos(ra, dec, orient)


if __name__ == "__main__":
	

	extra = [(0.0, 90.0, "z")]

	for r in [#(169, 243), # 0x
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
		

	for r in [#(124, 628), # 0x
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
	
	
	
	ra = 0.
	dec = 90.
	size = 5.0
	w = 1200
	h = 1200
	pixscale = size / w
	
	wcs = Tan(*[float(x) for x in [
		ra, dec, 0.5 + (w / 2.), 0.5 + (h / 2.),
		-pixscale, 0., 0., pixscale, w, h,
		]])
	plot = Plotter(wcs)
	cv2.imshow("plot", plot.plot(extra=extra))
	cv2.waitKey(0)