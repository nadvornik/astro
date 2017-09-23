#!/usr/bin/python

import numpy as np
from quat import Quaternion
from astrometry.util.util import Tan
import math
import cv2
import time
import os

import logging

log = logging.getLogger()

from am import Plotter


def quat_axis_to_ra_dec(q):
	(w,x,y,z) = q.a
	#print "xyz", x,y,z
	ra = math.degrees(math.atan2(y,x))
	dec = math.degrees(math.atan2(z, (x ** 2 + y ** 2)**0.5))
	return (ra, dec)

def xyz_to_ra_dec(v):
	(x,y,z) = v
	#print "xyz", x,y,z
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
	rot = S * 15.0
	#print "cel time", T, S0, jh, S
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
	def __init__(self, status, cameras):
		self.status = status
		self.pos = []
		self.campos = []
		self.cameras = {}
		self.campos_avg = []
		self.campos_adjust = []
		for i, c in enumerate(cameras):
			self.cameras[c] = i
			self.pos.append(None)
			self.campos.append(None)
			self.campos_avg.append(None)
			self.campos_adjust.append(None)
		self.prec_q = precession()
		self.status.setdefault('gps', (50.0, 15.0))
		self.prec_ra, self.prec_dec = self.prec_q.inv().transform_ra_dec([0, 90])
		self.reset()
		
	def reset(self):
		self.t0 = None
		self.ra = None
		self.dec = None
		self.solved = False
		self.mode = 'solve'
		for i in range(0, len(self.pos)):
			self.pos[i] = []
			self.campos[i] = []
			self.campos_avg[i] = None
			self.campos_adjust[i] = None
		self.p2_from = None
		self.ref_ra = None
		self.ref_dec = None
		
	def set_gps(self, gps):
		self.status['gps'] = gps

	def set_mode(self, mode):
		if mode == 'adjust' and self.solved:
			self.mode = 'adjust'
		else:
			self.mode = 'solve'
			
	
	def set_pos(self, ra, dec, roll, t, camera):
		if self.mode == 'solve':
			self.mode_solve_set_pos(ra, dec, roll, t, camera)
		elif self.mode == 'adjust':
			self.mode_adjust_set_pos(ra, dec, roll, t, camera)
	
	def mode_solve_set_pos(self, ra, dec, roll, t, camera):
		ci = self.cameras[camera]
		pos_orig = Quaternion([ra, dec, roll])
		self.campos_adjust[ci] = (pos_orig, t)
		if self.t0 is None:
			self.t0 = t
		ha = (t - self.t0) / 240.0
		qha = self.prec_q * Quaternion([-ha, 0, 0]) / self.prec_q
		#print "qha", quat_axis_to_ra_dec(qha), "prec", self.prec_ra, self.prec_dec 
		#print Quaternion([ra, dec, roll]).to_euler(), (qha * Quaternion([ra, dec, roll])).to_euler()
		
		pos = qha * pos_orig
		self.pos[ci].append((pos, t))
		
		if ci > 0 and len(self.pos[0]) > 0:
			prev_pos, prev_t = self.campos_adjust[0]
			if abs(t - prev_t) < 10:
				self.campos[ci].append(pos_orig.inv() * prev_pos)
		elif ci == 0:
			for i in range(1, len(self.pos)):
				if len(self.pos[i]) > 0:
					prev_pos, prev_t = self.campos_adjust[i]
					if abs(t - prev_t) < 10:
						self.campos[i].append(prev_pos.inv() * pos_orig)

			

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

	def set_pos_tan(self, tan, t, camera, off = (0, 0)):
		ra, dec, orient = self.tan_to_euler(tan, off)
		#print ra, dec, orient
		self.set_pos(ra, dec, orient, t, camera)
		#print "added ", t


	def compute2(self, i = 0, j = None):
		if j is None:
			j = len(self.pos) - 1
		q1 = self.pos[i]
		q2 = self.pos[j]
		
		log.info(q1.a)
		log.info(q2.a)
		
		c = q2 / q1

		ra, dec = quat_axis_to_ra_dec(c)
		
		if dec < 0:
			dec = -dec
			ra -= 180
		if ra < 0.0 :
			ra += 360
		
		log.info(ra, dec)
		self.ra = ra
		self.dec = dec
		return True, ra, dec
	
	def camera_position(self, ci, noise = 2):
		if len(self.campos[ci]) < 2:
			return None
		avg = Quaternion.average(self.campos[ci])
		
		d = np.array([avg.distance_metric(q) for q in self.campos[ci]])
		
		d2 = d**2
		var = np.mean(d2)
		#print var * noise**2
		#print d2
		#print np.where(d2 < var * noise**2)[0]
		self.campos[ci] = [self.campos[ci][i] for i in np.where(d2 < var * noise**2)[0]]
		
		avg = Quaternion.average(self.campos[ci])
		self.campos_avg[ci] = avg
		return avg

	def solve_(self, noise=2):
		if self.mode == 'adjust':
			return self.ra, self.dec
		if len(self.pos[0]) < 2:
			return None, None

		qlist = [p.a for (p, t) in self.pos[0]]
		
		for ci in range(1, len(self.pos)):
			q_trans = self.camera_position(ci, noise)
			if q_trans is not None:
				log.info("q_trans", q_trans.to_euler())
			
			#q_trans=Quaternion([0,1,0])
			if q_trans is not None:
				qlist_ci = [ (p * q_trans).a for (p, t) in self.pos[ci]]
				#print "qlist",  [ Quaternion(q).to_euler() for q in qlist]
				#print "qlist_ci", [ Quaternion(q).to_euler() for q in qlist_ci]
				#print "orig_ci", [ q.to_euler() for (q, t) in self.pos[ci]]
				qlist.extend(qlist_ci)
				#if len(qlist_ci) > 2:
				#	qlist = qlist_ci
		
				
		qa = np.array(qlist)
		
		qamin = np.amin(qa, axis = 0)
		qamax = np.amax(qa, axis = 0)
		qarange = qamax - qamin
		ao = np.argsort(qarange) #axis order, ao[0], ao[1] are computed from ao[2] and ao[3]

		A = np.column_stack((qa[:,ao[2]], qa[:,ao[3]]))
		res0 = np.linalg.lstsq(A, qa[:,ao[0]])[0]
		res1 = np.linalg.lstsq(A, qa[:,ao[1]])[0]
		
		dif0 = qa[:,ao[0]] - (qa[:,ao[2]] * res0[0] + qa[:,ao[3]] * res0[1])
		dif1 = qa[:,ao[1]] - (qa[:,ao[2]] * res1[0] + qa[:,ao[3]] * res1[1])

		d2 = dif0**2 + dif1**2
		var = np.mean(d2)
		qa = qa[(d2 < var * noise**2)]

		# again, with filtered values
		A = np.column_stack((qa[:,ao[2]], qa[:,ao[3]]))
		res0 = np.linalg.lstsq(A, qa[:,ao[0]])[0]
		res1 = np.linalg.lstsq(A, qa[:,ao[1]])[0]

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
		self.solved = True
		#print "rotation center", ra, dec
		return ra, dec

		
		#fig = plt.figure()
		#ax = fig.add_subplot(111, projection='3d')

		#ax.scatter(qa[:,0], qa[:,1], qa[:,2], c=qa[:,3], cmap=plt.hot())
		#ax.scatter(qa[:,0], qa[:,1], qa[:,3], c=qa[:,3], cmap=plt.hot())
		#ax.scatter(qa[:,0], qa[:,1], qa[:,0] * res2[0] + qa[:,1] * res2[1] + res2[2], c=qa[:,3], cmap=plt.hot())
		#plt.show()

	def solve(self, noise=2):
		if self.mode == 'adjust':
			return self.ra, self.dec
		if len(self.pos[0]) < 3:
			return None, None

		weights = np.matrix(np.zeros((2, 2)))
		wsum = np.zeros((2,1))

		for ci in range(0, len(self.pos)):
			if len(self.pos[ci]) < 3:
				continue
			try:
				qlist = [ p for (p, t) in self.pos[ci]]
				
				avg = Quaternion.average(qlist).inv()
				alist = []
				for q in qlist:
					q0 = q * avg
					if abs(q0.a[0]) < 0.7:
						ax, roll = q0.to_axis_roll()
						q0 = Quaternion.from_axis_roll(ax, roll - 180)

					a = q0.a[1:4] / q0.a[0]
					alist.append(a)
				aa = np.array(alist)


				d2 = np.sum(aa[:, 0:2] ** 2, axis = 1)
				var = np.mean(d2)

				aa = aa[(d2 < var * 4)]


				line = cv2.fitLine(aa / aa[0,2] * 100, cv2.DIST_L2, 0.001, 0.000001, 0.000001)[0:3].reshape(3)

				line2d = line[0:2] / line[2]
				aa2d = aa[:, 0:2] - np.outer(aa[:, 2],  line2d)
				cov =np.cov(aa2d.T)
				#print cov
				w = np.matrix(cov).I
				#print ci, "w", w
				weights += w
				wsum += w * line2d.reshape((2,1))
	
				#print ci, "weights", weights
				#print ci, "wsum", wsum
			except:
				continue
	
		try:
			line2d = weights.I * wsum
			#print "res", line2d
			line = np.array([line2d[0,0], line2d[1,0], 1])

			ra, dec = xyz_to_ra_dec(line)
		
			if dec < 0:
				dec = -dec
				ra -= 180
			if ra < 0.0 :
				ra += 360
		
			self.ra = ra
			self.dec = dec
			self.solved = True
		except:
			return None, None
		#print "rotation center", ra, dec
		#print "prec", self.prec_ra, self.prec_dec
		return ra, dec

		#fig = plt.figure()
		#ax = fig.add_subplot(111, projection='3d')

		#ax.scatter(qa[:,0], qa[:,1], qa[:,2], c=qa[:,3], cmap=plt.hot())
		#ax.scatter(qa[:,0], qa[:,1], qa[:,3], c=qa[:,3], cmap=plt.hot())
		#ax.scatter(qa[:,0], qa[:,1], qa[:,0] * res2[0] + qa[:,1] * res2[1] + res2[2], c=qa[:,3], cmap=plt.hot())
		#plt.show()

	def save(self):
		alist = []

		for ci in range(0, len(self.pos)):
			qlist = [ p.a for (p, t) in self.pos[ci]]
			alist.append(np.array(qlist))
		np.savez("polar_%d.npz" % self.t0, *alist)

	def plot(self):
		extra = []
		if self.ra is not None and self.dec is not None:
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
		t = Quaternion.from_ra_dec_pair([self.prec_ra, self.prec_dec], [self.ra, self.dec])
		#print "transform_ra_dec_list", t.transform_ra_dec([self.ra, self.dec])
		res = []
		for rd in l:
			res.append(t.transform_ra_dec(rd))
		return res

	def mode_adjust_set_ref_pos(self):
		self.ref_ra = self.ra
		self.ref_dec = self.dec
		
	def mode_adjust_set_pos(self, ra, dec, roll, t, camera):
		ci = self.cameras[camera]
		pos_orig = Quaternion([ra, dec, roll])
		self.campos_adjust[ci] = (pos_orig, t)
		
		poslist = []
		for i in range(0, len(self.pos)):
			if self.campos_adjust[i] is None:
				continue
			prev_pos, prev_t = self.campos_adjust[i]
			if abs(t - prev_t) > 10:
				continue
			if i > 0  and self.campos_avg[i] is None:
				self.camera_position(i)
			if i > 0  and self.campos_avg[i] is not None:
				prev_pos =  prev_pos * self.campos_avg[i]
			poslist.append(prev_pos)
		
		log.info([p.to_euler() for p in poslist])
		
		pos = Quaternion.average(poslist)

		if self.p2_from is None:
			self.mode_adjust_set_ref_pos()
			self.p2_from = pos
			self.save()
			
		t = pos / self.p2_from
		log.info(t.to_euler())
		self.ra, self.dec = t.transform_ra_dec([self.ref_ra, self.ref_dec])
	

	def plot2(self, size = 960, area = 0.1):
		ha = celestial_rot() + self.status['gps'][1]
		qha = Quaternion([90-ha, 0, 0])
		
		img = np.zeros((size, size, 3), dtype=np.uint8)
		c = size / 2
		scale = size / area
		
		if self.ra is not None and self.dec is not None:
			t = Quaternion.from_ra_dec_pair([self.ra, self.dec], [self.prec_ra, self.prec_dec])
		else:
			t = Quaternion.from_ra_dec_pair([0.0, 90.0], [self.prec_ra, self.prec_dec])
		
		
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
		
		
		if self.ra is not None and self.dec is not None:
			cv2.circle(img, (int(c + prec_xyz[0] * scale), int(c + prec_xyz[1] * scale)), 4, (255, 255, 255), 2)
			cv2.circle(img, (int(c + polaris_real_xyz[0] * scale), int(c + polaris_real_xyz[1] * scale)), 4, (255, 255, 255), 2)
		
			pole_dist = (prec_xyz[0] ** 2 + prec_xyz[1] ** 2) ** 0.5
			if pole_dist >= area / 2:
				cv2.putText(img, "%0.1fdeg" % (90 - prec[1]), (int(c + prec_xyz[0] / pole_dist * area / 5 * scale - 50), int(c + prec_xyz[1] / pole_dist * area / 5 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
				cv2.arrowedLine(img, (int(c + prec_xyz[0] / pole_dist * area / 3 * scale), int(c + prec_xyz[1] / pole_dist * area / 3 * scale)), (int(c + prec_xyz[0] / pole_dist * area / 2 * scale), int(c + prec_xyz[1] / pole_dist * area / 2 * scale)), (255, 255, 255), 2)

		return img
	
	def zenith(self):
		ha = (celestial_rot() + self.status['gps'][1]) % 360.0
		dec = self.status['gps'][0]
		return (ha, dec)

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
		res, ra, dec = p.solve()
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
		res, ra, dec = p.solve()
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
	res, ra, dec = p.solve()
	extra.append((ra, dec, "gphoto"))
	
	p = Polar()
	for t in [
	          1446322479,
	          1446322511,
	          1446322562,
	          1446322596]:
		tan = Tan('t/capture_v4l%d.wcs' % t , 0)
		p.add_tan(tan, t)
	res, ra, dec = p.solve()
	extra.append((ra, dec, "v4l2"))
	cv2.imshow("polar", p.plot2())
	
	p.transform_ra_dec_list([])
	
	#for i in range(100, 1400,500):
	#	pp = Polar()
	#	for j in range(i, i+2000):
	#		tan = Tan('log_%d.wcs' % (j),0)
	#		pp.add_tan(tan, 0)
	#	ra, dec = pp.solve()
	#	extra.append((ra, dec, ""))

	
	#for i in range (20, 50):
	#	ra, dec = p.compute2(0, i)
	#	extra.append((ra, dec, ""))
	qp = precession()
	ra, dec = qp.inv().transform_ra_dec([0, 90])
	extra.append((ra, dec, "p"))
	log.info("precession", ra, dec)
	
	
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
	
	log.info(p.zenith())
	
	cv2.waitKey(0)
