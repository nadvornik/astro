#!/usr/bin/env python

# Copyright (C) 2015 Vladimir Nadvornik
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import math
import numpy as np
import cv2
import logging
from time import sleep
import bisect
import subprocess

from am import Solver, Plotter
from polar import Polar

import sys
import io
import os.path
import time
import threading

from v4l2_camera import *
from camera_gphoto import *

from serial_control import GuideOutBase, GuideOut

import random
from line_profiler import LineProfiler

from gui import ui
from cmd import cmdQueue

def normalize(img):
	dst = np.empty_like(img)
	return cv2.normalize(img, dst, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	

class Median:
	def __init__(self, n):
		self.n = n
		self.i = 0
		self.list = []
		self.res = None
		self.bg_thread = None
	
	def _add(self, im):
		if (self.i < len(self.list)):
			self.list[self.i] = im
		else:
			self.list.append(im)
		self.i = (self.i + 1) % self.n

		a = np.array(self.list)
		for i in range(a.shape[1]):
			a[:, i, :] = cv2.sort(a[:, i, :], cv2.SORT_DESCENDING | cv2.SORT_EVERY_COLUMN)
		self.res = np.array(a[a.shape[0] / 2, :, :])
		#a = np.median(self.list, axis = 0)
		#self.res = np.empty_like(self.list[0])
		#self.res[:,:] = a

	def _add_masked(self, im, pts):
		mask = np.zeros_like(im)
	
		white = np.iinfo(im.dtype).max
		for p in pts:
			cv2.circle(mask, p, 20, (white), -1)

		mask = cv2.blur(mask, (30, 30))
		mask = cv2.blur(mask, (30, 30))
		inv_mask = cv2.bitwise_not(mask)
		
		res = cv2.add(cv2.multiply(im, inv_mask, scale = 1.0 / white), cv2.multiply(self.res, mask, scale = 1.0 / white))
		
		self._add(res)
		#ui.imshow("dark", normalize(inv_mask))

	def add(self, *args, **kwargs):
		if self.bg_thread is not None:
			self.bg_thread.join()
		
		self.bg_thread = threading.Thread(target=self._add, args = args, kwargs = kwargs)
		self.bg_thread.start()

	def add_masked(self, *args, **kwargs):
		if self.bg_thread is not None:
			self.bg_thread.join()
		
		self.bg_thread = threading.Thread(target=self._add_masked, args = args, kwargs = kwargs)
		self.bg_thread.start()

		
	def get(self):
		if self.bg_thread is not None:
			self.bg_thread.join()
			self.bg_thread = None
			
		return self.res

	def len(self):
		if self.bg_thread is not None:
			self.bg_thread.join()
			self.bg_thread = None
		return len(self.list)

	def reset(self):
		if self.bg_thread is not None:
			self.bg_thread.join()
			self.bg_thread = None
		self.i = 0
		self.list = []


class MaxDetector(threading.Thread):
	def __init__(self, img, d, n, y1, y2):
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
		ordmax = np.argsort(valmax)[::-1]
		ordmax = ordmax[:self.n]

		self.found = []
	
		for (y, x, v) in zip(locmax[0][ordmax], locmax[1][ordmax], valmax[ordmax]):
			if (x < 1):
				continue
			if (y + self.y1 < 1):
				continue
			if (x > w - 2):
				continue
			if (y + self.y1 > h - 2):
				continue
			dx = imge[y + self.y1e0, x - 1] - 2 * v + imge[y + self.y1e0, x + 1]
			dy = imge[y + self.y1e0 - 1, x] - 2 * v + imge[y + self.y1e0 + 1, x]
			if dx != 0:
				xs = 0.5*(imge[y + self.y1e0, x - 1] - imge[y + self.y1e0, x + 1]) / dx
			else:
				xs = 0
			if dy != 0:
				ys = 0.5*(imge[y + self.y1e0 - 1, x] - imge[y + self.y1e0 + 1, x]) / dy
			else:
				ys = 0
			# y, x, flux
			self.found.append((y + self.y1 + ys, x + xs, v))

	

def find_max(img, d, n = 40):
	(h, w) = img.shape
	par = 4
	step = (h + par - 1) / par
	mds = []
	for y in range(0, h, step):
		md = MaxDetector(img, d, n / par + 1, y, min(y + step, h))
		mds.append(md)
		#md.run()
		md.start()
		

	joined = []
	for md in mds:
		md.join()
		joined += md.found
	
	joined = np.array(joined)
	ordmax = np.argsort(joined[:, 2])[::-1]
	ordmax = ordmax[:n]
	joined = joined[ordmax]
	
	return joined

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

def match_triangle(pt1, pt2, maxdif = 5.0, maxdrift = 10, off = (0.0, 0.0)):
	if len(pt1) == 0 or len(pt2) == 0:
		return match_take(pt1, pt2, [])
	
	ord1 = np.argsort(pt1[:, 2])[::-1]
	ord2 = np.argsort(pt2[:, 2])[::-1]
	
	pt1s = pt1[ord1][:20]
	pt2s = pt2[ord2][:20]
	
	dist1 = np.array([ [ ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 for p1 in pt1s ] for p2 in pt1s ])
	dist2 = np.array([ [ ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 for p1 in pt2s ] for p2 in pt2s ])

	bestmatch = []

	for a1 in range(0, len(pt1) - 2):
		for b1 in range(a1 + 1, len(pt1s) - 1):
			ab1 = dist1[a1, b1]
			((a2, b2), dif) = find_nearest(dist2, ab1)
			if dif > maxdif:
				continue
			match = []
			if len(bestmatch) < 2:
				if check_drift(pt1s, pt2s, [[a1, a2], [b1, b2]], maxdif, maxdrift, off):
					bestmatch = [[a1, a2], [b1, b2]]
				elif check_drift(pt1s, pt2s, [[a1, b2], [b1, a2]], maxdif, maxdrift, off):
					bestmatch = [[a1, b2], [b1, a2]]
			
			for c1 in range(b1 + 1, len(pt1s)):
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
			
			if len(match) == 3 and len(bestmatch) < 3 and check_drift(pt1s, pt2s, match, maxdif, maxdrift, off):
				bestmatch = match
			
			for d1 in range(c1 + 1, len(pt1s)):
				ad1 = dist1[a1, d1]
				bd1 = dist1[b1, d1]
				cd1 = dist1[c1, d1]
				
				((d2_1,), dif1) = find_nearest(dist2[a2], ad1)
				((d2_2,), dif2) = find_nearest(dist2[b2], bd1)
				((d2_3,), dif3) = find_nearest(dist2[c2], cd1)
				if d2_1 == d2_2 and d2_2 == d2_3 and dif1 < maxdif and dif2 < maxdif and dif3 < maxdif:
					match.append([d1, d2_1])
			
			if len(match) > 3:
				# 2 triangles are enough
				return match_take(pt1s, pt2s, match, ord1, ord2)
	
	if len(bestmatch) == 0 and len(pt1s) > 0 and len(pt2s) > 0:
		for i in range(0, min(3, len(pt1s))):
			for j in range(0, min(3, len(pt2s))):
				if check_drift(pt1s, pt2s, [[i, j]], maxdif, maxdrift, off):
					bestmatch.append([i, j])
	return match_take(pt1s, pt2s, bestmatch, ord1, ord2)
	

def match_closest(pt1, pt2, d, off = (0.0, 0.0)):
	if len(pt1) == 0 or len(pt2) == 0:
		return match_take(pt1, pt2, [])
	
	ord2 = np.argsort(pt2[:, 0])
	pt2s = pt2[ord2]
	match = []
	l = len(pt2s)
	for i1, (y1orig, x1orig, flux1) in enumerate(pt1):
		y1 = y1orig + off[0]
		x1 = x1orig + off[1]
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
			weights[np.where(difdif2 > var * noise**2)] = 0.0
			v = np.average(dif, axis = 0, weights = weights)
			return v, weights
	elif pt1m.shape[0] == 1:
		v = (pt2m - pt1m)[0, 0:2]
		weights = np.sqrt(pt2m[:, 2] * pt1m[:, 2])
		return v, weights
	
	v = np.array([0.0, 0.0])
	weights = np.array([0.0])
	return v, weights


class Stack:
	def __init__(self, ratio = 0.1):
		self.img = None
		self.prev_pt = []
		self.prev_pt_verified = []
		self.xy = None
		self.ratio = ratio
	
	def add(self, im, show_match = False):
		if im.dtype == np.uint8:
			im = cv2.multiply(im, 255.0, dtype=cv2.CV_16UC1)
		if (self.img is None):
			self.img = im
			return (0.0, 0.0)
			
		pt2 = find_max(im, 12, n = 40)

		pt1 = self.prev_pt_verified
		pt1m, pt2m, match = match_triangle(pt1, pt2, 5, 15)
		#print "match1",match
		if len(match) == 0:
			pt1 = self.get_xy()
			pt1m, pt2m, match = match_triangle(pt1, pt2, 5, 15)
			#print "match2",match
		
		
		if len(match) == 0:
			self.img = cv2.multiply(im, self.ratio, dtype=cv2.CV_16UC1)
			self.prev_pt_verified = pt2
			self.prev_pt = pt2
			off = np.array([0., 0.])
		else:
			off = np.median(pt2m[:, 0:2] - pt1m[:, 0:2], axis = 0)
			pt1 = self.prev_pt
	
			pt1m, pt2m, match = match_closest(pt1, pt2, 5, off)
			off, weights = avg_pt(pt1m, pt2m)
			#print "off2", off 
			#print match
			self.prev_pt_verified = pt2m
			self.prev_pt = pt2

			M = np.array([[1.0, 0.0, off[1]],
			             [0.0, 1.0, off[0]]])


			self.img = cv2.warpAffine(self.img, M[0:2,0:3], (im.shape[1], im.shape[0]));
			
			self.img = cv2.addWeighted(self.img, 1.0 - self.ratio, im, self.ratio, 0, dtype=cv2.CV_16UC1)

		
		self.xy = None

		if show_match:
			self.match = normalize(self.img)
			for p in pt1:
				cv2.circle(self.match, (int(p[1]), int(p[0])), 13, (255), 1)
		
			for p in pt2:
				cv2.circle(self.match, (int(p[1]), int(p[0])), 5, (255), 1)
			for p in pt2m:
				cv2.circle(self.match, (int(p[1]), int(p[0])), 10, (255), 1)
		
		return off

	def add_simple(self, im):
		if im.dtype == np.uint8:
			im = cv2.multiply(im, 255, dtype=cv2.CV_16UC1)
		if (self.img is None or self.img.shape != im.shape):
			self.img = im
			return
		self.img = cv2.addWeighted(self.img, 1.0 - self.ratio, im, self.ratio, 0, dtype=cv2.CV_16UC1)
		self.xy = None
		return (0.0, 0.0)

	def get(self, dtype = np.uint8):
		if dtype == np.uint8:
			return cv2.divide(self.img, 255, dtype=cv2.CV_8UC1)
		else:
			return self.img

	def get_xy(self):
		if self.xy is None:
			self.xy = np.array(find_max(self.img, 12, n = 20))

		return self.xy
	
	def reset(self):
		self.img = None

def _plot_bg(window, func, *args, **kwargs):
	ui.imshow(window, func(*args, **kwargs))

def plot_bg(*args, **kwargs):
	#_plot_bg(*args, **kwargs)
	threading.Thread(target=_plot_bg, args = args, kwargs = kwargs).start()

class Navigator:
	def __init__(self, dark, ui_capture):
		self.dark = dark
		self.stack = Stack()
		self.solver = None
		self.solver_off = np.array([0.0, 0.0])
		self.dispmode = 'disp-normal'
		self.ra = None
		self.dec = None
		self.field_deg = None
		self.plotter = None
		self.plotter_off = np.array([0.0, 0.0])
		self.ii = 0
		self.ui_capture = ui_capture
		self.polar = Polar()
		self.polar_mode = 1
		self.polar_solved = False
		self.index_sources = []

	def proc_frame(self,im, i, t = None):
	
		self.im = im
		
		if t == None:
			t = time.time()
		if (self.dark.len() > 2):
			im_sub = cv2.subtract(im, self.dark.get())
		else:
			im_sub = im
			
		bg = cv2.blur(im_sub, (30, 30))
		bg = cv2.blur(bg, (30, 30))
		im_sub = cv2.subtract(im_sub, bg)


		if (self.dark.len() == 0):
			self.dark.add(im)
	
		off = self.stack.add(im_sub, show_match=(self.dispmode == 'disp-match'))
		filtered = self.stack.get()
		
		self.solver_off += off
		self.plotter_off += off

		if (self.dispmode == 'disp-orig'):
			ui.imshow(self.ui_capture, normalize(im))
		elif (self.dispmode == 'disp-df-cor'):
			ui.imshow(self.ui_capture, normalize(im_sub))
		elif (self.dispmode == 'disp-normal'):
			if self.plotter is not None:
				nm = normalize(filtered)
				for p in self.stack.get_xy():
					cv2.circle(nm, (int(p[1]), int(p[0])), 13, (255), 1)
		
				extra = []
				if self.polar_solved:
					transf_index = self.polar.transform_ra_dec_list(self.index_sources)
					extra = [ (ti[0], ti[1], "") for ti in transf_index ]
					#print "extra: ", extra
					
				plot_bg(self.ui_capture, self.plotter.plot, nm, self.plotter_off, extra = extra)
			else:
				ui.imshow(self.ui_capture, normalize(filtered))
		elif (self.dispmode.startswith('disp-zoom-')):
			if self.plotter is not None:
				zoom = self.dispmode[len('disp-zoom-'):]
				plot_bg(self.ui_capture, self.plotter.plot, normalize(filtered), self.plotter_off, scale=zoom)
			else:
				ui.imshow(self.ui_capture, normalize(filtered))
				
		elif (self.dispmode == 'disp-match'):
			ui.imshow(self.ui_capture, normalize(self.stack.match))
	
		if self.solver is not None and not self.solver.is_alive():
			self.solver.join()
			if self.solver.solved:
				self.ra = self.solver.ra
				self.dec = self.solver.dec
				self.field_deg = self.solver.field_deg
			
				self.dark.add_masked(self.solved_im, self.solver.ind_sources)
				
				self.index_sources = self.solver.ind_radec
				#print "self.solver.ind_radec", self.solver.ind_radec
				#self.solver.wcs.write_to("log_%d.wcs" % self.ii)
				#subprocess.call(['touch', '-r', "testimg17_" + str(i) + ".tif", "log_%d.wcs" % self.ii])
				if self.polar_mode == 1:
					self.polar.add_tan(self.solver.wcs, self.solver_time)
					if self.polar.compute()[0]:
						self.polar_solved = True
						ui.imshow(self.ui_capture + '_polar', self.polar.plot2())
						#ui.imshow(self.ui_capture + '_polar', self.polar.plot())
				elif self.polar_mode == 2:
					self.polar.phase2_set_tan(self.solver.wcs)
					#ui.imshow(self.ui_capture + '_polar', self.polar.plot())
					ui.imshow(self.ui_capture + '_polar', self.polar.plot2())
					
				self.ii += 1
				self.plotter = Plotter(self.solver.wcs)
				self.plotter_off = self.solver_off
			else:
				self.ra = None
				self.dec = None
			self.solver = None
			self.solved_im = None

		if self.solver is None and i > 20 :
			xy = self.stack.get_xy()
			print "len", len(xy)
			if len(xy) > 6:
				self.solver_time = t
				self.solved_im = im
				self.solver = Solver(sources_list = xy, field_w = im.shape[1], field_h = im.shape[0], ra = self.ra, dec = self.dec, field_deg = self.field_deg)
				#self.solver = Solver(sources_img = filtered, field_w = im.shape[1], field_h = im.shape[0], ra = self.ra, dec = self.dec, field_deg = self.field_deg)
				self.solver.start()
				self.solver_off = np.array([0.0, 0.0])
		
	def cmd(self, cmd):
		if cmd == 'solver-reset' and self.solver is not None:
			self.solver.terminate(wait=True)
			self.field_deg = None
			self.solver = None

		if cmd == 'dark':
			self.dark.add(self.im)
		
		if cmd.startswith('disp-'):
			self.dispmode = cmd
		if cmd == 'save':
			cv2.imwrite(self.ui_capture + str(int(time.time())) + ".tif", self.stack.get())

		if cmd == 'polar-reset':
			self.polar = Polar()
			self.polar_mode = 1
			self.polar_solved = False

		if cmd == 'polar-align' and self.polar_solved:
			self.polar_mode = 2


def fit_line(xylist):
	a = np.array(xylist)
	x = a[:, 0]
	y = a[:, 1]
	return np.polyfit(x, y, 1)

class Guider:
	def __init__(self, go, dark, ui_capture):
		self.go = go
		self.dark = dark
		self.reset()
		self.t0 = 0
		self.resp0 = []
		self.pt0 = []
		self.ui_capture = ui_capture

	def reset(self):
		self.mode = 1
		self.off = (0.0, 0.0)
		self.off_t = None
		self.go.out(0)
		self.cnt = 0
		self.pt0 = []
		self.ok = False
		self.capture_in_progress = False

	def dark_add_masked(self, im):
		h, w = im.shape
		pts = []
		for p in self.pt0:
			(x, y) = (int(p[1] + self.off[1]), int(p[0] + self.off[0]))
			if (x < 0):
				continue
			if (y < 0):
				continue
			if (x > w - 1):
				continue
			if (y > h - 1):
				continue
			pts.append((x,y))
		return self.dark.add_masked(im, pts)

	def cmd(self, cmd):
		if cmd == "capture":
			self.capture_in_progress = True
		if cmd == "capture-finished":
			self.capture_in_progress = False

	def proc_frame(self, im, i):
		t = time.time()
		print "mode", self.mode
		
		if len(self.pt0) == 0:
			cmdQueue.put('navigator')
			self.go.out(0)

		if (self.dark.len() >= 4):
			print "dark"
			im_sub = cv2.subtract(im, self.dark.get())
		else:
			print "no dark"
			im_sub = im


		bg = cv2.blur(im_sub, (30, 30))
		bg = cv2.blur(bg, (30, 30))
		im_sub = cv2.subtract(im_sub, bg)

		debug = normalize(im_sub)

		if self.mode==1:
			self.used_cnt = []
			self.cnt = 0
			self.dist = 1.0
			self.go.out(1)
			self.mode = 2
			self.t0 = t
			self.resp0 = []

		elif self.mode==2:
				
			self.cnt += 1
			pt = find_max(im_sub, 20, n = 30)
			
			# ignore hotpixels
			pt1m, pt2m, match = match_closest(self.pt0, pt, 5)
			if len(match) > 0:
				pt = np.delete(pt, match[:, 1], axis = 0)
			
			if self.off_t is None:
				off = (0., 0.)
			else:
				off = self.off * (t - self.t0) / (self.off_t - self.t0)
			pt1m, pt2m, match = match_triangle(self.pt0, pt, 5, 50, off)
			if len(match) > 0:
				off, weights = avg_pt(pt1m, pt2m)
				print "triangle", off, match
			
				pt0, pt, match = match_closest(self.pt0, pt, 5, off)
			
			if len(match) > 0:
			
				off, weights = avg_pt(pt0, pt, noise = 3)
				print "weights", weights 
				dist = np.linalg.norm(off)

				if (dist > 20):
					self.dark.add(im)
			
				self.resp0.append((t - self.t0, dist))
			
				if (dist > self.dist):
					self.dist = dist
					self.off = off
					self.off_t = t
			
				print off, dist
				pt_ok = match[np.where(weights > 0), 0][0]
				self.used_cnt.extend(pt_ok)

				for i in pt_ok:
					p = self.pt0[i]
					cv2.circle(debug, (int(p[1] + self.off[1]), int(p[0] + self.off[0])), 13, (255), 1)

				if (self.dist > 100 and self.cnt > 12):
					self.t1 = t
					dt = self.t1 - self.t0
					self.go.out(-1)
				
					aresp = np.array(self.resp0)
					aresp1 = aresp[aresp[:, 1] > 10]
					m, c = np.polyfit(aresp1[:, 0], aresp1[:, 1], 1)

					self.pixpersec = m
					self.t_delay1 = -c / m
					self.pixperframe = self.pixpersec * dt / self.cnt
					self.dist = m * dt + c
					self.ref_off = complex(*self.off) / dist
				
					print "pixpersec", self.pixpersec, "pixperframe", self.pixperframe, "t_delay1", self.t_delay1
				
					self.pt0 = np.array(self.pt0)[np.where(np.bincount(self.used_cnt) > self.cnt / 3)]
				
					self.cnt = 0
					self.mode = 3
				
					self.go.out(-1, self.dist / self.pixpersec)
			for p in pt:
				cv2.circle(debug, (int(p[1]), int(p[0])), 10, (255), 1)

		elif self.mode==3:
			self.cnt += 1
			pt = find_max(im_sub, 20, n = 30)
			print pt
			pt1m, pt2m, match = match_triangle(self.pt0, pt, 5, 50, self.off)
			if len(match) > 0:
				off, weights = avg_pt(pt1m, pt2m)
				print "triangle", off, match
			
				pt0, pt, match = match_closest(self.pt0, pt, 5, off)
				
			if len(match) > 0:
				self.off, weights = avg_pt(pt0, pt)
				err = complex(*self.off) / self.ref_off
				
				self.resp0.append((t - self.t0, err.real))
			
				print "err:", err, err.real

				if (err.real > 30):
					self.dark_add_masked(im)

				for p in pt:
					cv2.circle(debug, (int(p[1]), int(p[0])), 10, (255), 1)
				self.go.out(-1, err.real / self.pixpersec)
				
				if err.real < self.pixpersec * self.t_delay1 + self.pixperframe:
					self.t2 = t
					dt = self.t2 - self.t1
					
					aresp = np.array(self.resp0)
					aresp1 = aresp[aresp[:, 0] > self.t1 + self.t_delay1 - self.t0]
					m, c = np.polyfit(aresp1[:, 0], aresp1[:, 1], 1)

					self.pixpersec_neg = m
					self.t_delay2 = (c + self.t_delay1 * self.pixpersec) / (self.pixpersec - self.pixpersec_neg) - self.t1 + self.t0


					self.pixperframe_neg = self.pixpersec_neg * dt / self.cnt
				
					print "pixpersec_neg", self.pixpersec_neg, "pixperframe_neg", self.pixperframe_neg, "t_delay2", self.t_delay2
					self.t_delay = (self.t_delay1 + self.t_delay2) / 2
					if (self.t_delay < 0):
						self.t_delay = 0
				
					self.mode = 4


		elif self.mode==4:
			pt = find_max(im_sub, 20, n = 30)
			pt1m, pt2m, match = match_triangle(self.pt0, pt, 5, 30, self.off)
			if len(match) > 0:
				off, weights = avg_pt(pt1m, pt2m)
				print "triangle", off, match
			
				pt0, pt, match = match_closest(self.pt0, pt, 5, off)
				
			if len(match) > 0:
				self.off, weights = avg_pt(pt0, pt)
				err = complex(*self.off) / self.ref_off
				self.resp0.append((t - self.t0, err.real))

				err_corr = err.real + self.go.recent_avg(self.t_delay) * self.pixpersec
				
				aggresivnes = 0.6
				err_corr *= aggresivnes
				print "err:", err, err.real, "corr:", err_corr, "t_delay: ", self.t_delay
				if err_corr > 0.1:
					self.go.out(-1, -err_corr / self.pixpersec_neg)
				elif err_corr < -0.1:
					self.go.out(1, -err_corr / self.pixpersec)
				else:
					self.go.out(0)
				
				self.ok = (err.real < 2 and err.real > -2)
				if self.ok and not self.capture_in_progress:
					cmdQueue.put('capture')
					self.capture_in_progress = True
				
				for p in pt:
					cv2.circle(debug, (int(p[1]), int(p[0])), 10, (255), 1)
				

				if i % 100 == 0:
					np.save("resp0_%d.npy" % self.t0, np.array(self.resp0))
					self.go.save("go_%d.npy" % self.t0)
					print "SAVED" 
				
		if len(self.pt0) > 0:
			for p in self.pt0:
				cv2.circle(debug, (int(p[1]), int(p[0])), 13, (255), 1)

		ui.imshow(self.ui_capture, debug)


class Focuser:
	def __init__(self, ui_capture):
		self.stack = Stack()
		self.dark = Median(3)
		self.ui_capture = ui_capture
		self.dispmode = 3


	def cmd(self, cmd):
		if cmd == 'dark':
			self.dark.add(self.im)
		if cmd.startswith('disp-'):
			self.dispmode = cmd

	def proc_frame(self, im, i):
		self.im = im

		if (self.dark.len() > 0):
			im_sub = cv2.subtract(im, self.dark.get())
			minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(im_sub)
			im_sub = cv2.add(im_sub, -minVal, dtype=cv2.CV_8UC1)
		else:
			im_sub = im


		self.stack.add_simple(im_sub)
		if (self.dispmode == 'disp-orig'):
			ui.imshow(self.ui_capture, normalize(im))
		elif (self.dispmode == 'disp-df-cor'):
			ui.imshow(self.ui_capture, normalize(im_sub))
		elif (self.dispmode == 'disp-normal'):
			ui.imshow(self.ui_capture, normalize(self.stack.get()))
		else:
			filtered = self.stack.get()
			filtered = normalize(filtered)
	
			mask = cv2.compare(filtered, 128, cv2.CMP_GE)
	
			print "Nonzero size: ", cv2.countNonZero(mask)
	
			rgb = cv2.cvtColor(filtered,cv2.COLOR_GRAY2RGB)
	
			rgb[:,:, 1] = cv2.bitwise_and(filtered, cv2.bitwise_not(mask))
			ui.imshow(self.ui_capture, rgb)

class Runner(threading.Thread):
	def __init__(self, camera, navigator = None, guider = None, focuser = None):
                threading.Thread.__init__(self)
		self.camera = camera
		self.navigator = navigator
		self.guider = guider
		self.focuser = focuser
		
	def run(self):
		profiler = LineProfiler()
		profiler.add_function(Navigator.proc_frame)
		profiler.add_function(Stack.add)
		profiler.add_function(Median.add)
		profiler.add_function(Median.add_masked)
		profiler.add_function(find_max)
		profiler.add_function(match_triangle)
		profiler.add_function(Plotter.plot)
		
		profiler.enable_by_count()
		
		
		cmdQueue.register(self)
		
		i = 0
		if self.navigator is not None:
			mode = 'navigator'
		else:
			mode = 'guider'

		while True:
			while True:
				cmd=cmdQueue.get(self, 1)
				if cmd is None:
					break
				if cmd == 'exit':
					if self.navigator.solver is not None and self.navigator.solver.is_alive():
						self.navigator.solver.terminate()
					profiler.print_stats()

					return
				elif cmd == 'navigator' and self.navigator is not None:
					mode = 'navigator'
				elif cmd == 'guider' and self.guider is not None:
					self.guider.reset()
					self.guider.pt0 = self.navigator.stack.get_xy()
					mode = 'guider'
				elif cmd == 'z1':
					if self.focuser is not None:
						mode = 'focuser'
						maxx = 300
						maxy = 300
						if self.navigator:
							minVal, maxVal, (minx, miny), (maxx, maxy) = cv2.minMaxLoc(self.navigator.stack.get())
						self.camera.cmd(cmd, x=maxx, y=maxy)
				elif cmd == 'z0' and self.navigator is not None:
					mode = 'navigator'
					self.camera.cmd(cmd)
				elif cmd == 'z0' and self.guider is not None:
					mode = 'guider'
					self.camera.cmd(cmd)
				else:
					self.camera.cmd(cmd)
					
				if mode == 'navigator':
					self.navigator.cmd(cmd)
				if mode == 'guider':
					self.guider.cmd(cmd)
				if mode == 'focuser':
					self.focuser.cmd(cmd)
	
			im, t = self.camera.capture()
			#cv2.imwrite("testimg20_" + str(i) + ".tif", im)
			im = np.amin(im, axis = 2)
			if mode == 'navigator':
				self.navigator.proc_frame(im, i, t)
			if mode == 'guider':
				self.guider.proc_frame(im, i)
			if mode == 'focuser':
				self.focuser.proc_frame(im, i)
			i += 1
			if i == 300:
				cmdQueue.put('exit')
		cmdQueue.put('exit')
		
from PIL import Image;


class Camera_test:
	def __init__(self):
		self.i = 0
		self.step = 1
		self.x = 0
		self.y = 0
	
	def cmd(self, cmd):
		if cmd == 'left':
			self.x -= 1
		if cmd == 'right':
			self.x += 1
		if cmd == 'up':
			self.y += 1
		if cmd == 'down':
			self.y -= 1
		if cmd == 'test-capture':
			self.step = 1 - self.step
	
	def capture(self):
		#time.sleep(0.9)
		print self.i
		#pil_image = Image.open("converted/IMG_%04d.jpg" % (146+self.i))
		#pil_image.thumbnail((1000,1000), Image.ANTIALIAS)
		#im = np.array(pil_image)
		im = cv2.imread("testimg16_" + str(self.i) + ".tif")
		#im = apply_gamma(im, 2.2)
		M = np.array([[1.0, 0.0, self.x],
		              [0.0, 1.0, self.y]])
		bg = cv2.blur(im, (30, 30))
		im = cv2.warpAffine(im, M[0:2,0:3], (im.shape[1], im.shape[0]), bg, borderMode=cv2.BORDER_TRANSPARENT);

		#t = os.path.getmtime("testimg16_" + str(self.i) + ".tif")
		self.i += self.step
		return im, 0

class Camera_test_g:
	def __init__(self, go):
		self.i = 0
		self.err = 0.0
		self.go = go
	
	def cmd(self, cmd):
		print "camera:", cmd
	
	def capture(self):
		time.sleep(0.5)
		self.err += random.random() * 2 - 1.1
		corr = self.go.recent_avg()
		i = int((corr - self.go.recent_avg(1))  + self.err)
		print self.err, corr * 3, i
		im = cv2.imread("testimg16_" + str(i + 50) + ".tif")
		return im, None


def run_v4l2():
	ui.namedWindow('capture')
	cam = Camera("/dev/video1")
	cam.prepare(1280, 960)
	dark = Median(5)
	nav = Navigator(dark, 'capture')

	runner = Runner(cam, navigator = nav)
	runner.start()
	runner.join()

def run_gphoto():
	cam = Camera_gphoto()
	cam.prepare()
	ui.namedWindow('capture')
	ui.namedWindow('capture_polar')
	ui.namedWindow('full_res')
	dark = Median(5)
	nav = Navigator(dark, 'capture')
	focuser = Focuser('capture')

	runner = Runner(cam, navigator = nav, focuser = focuser)
	runner.start()
	runner.join()


def run_v4l2_g():
	ui.namedWindow('capture')
	ui.namedWindow('capture_polar')
	cam = Camera("/dev/video1")
	cam.prepare(1280, 960)

	dark = Median(5)
	nav = Navigator(dark, 'capture')
	go = GuideOut()
	guider = Guider(go, dark, 'capture')

	runner = Runner(cam, navigator = nav, guider = guider)
	runner.start()
	runner.join()

def run_test_g():
	ui.namedWindow('capture')
	ui.namedWindow('capture_polar')
	dark = Median(5)
	nav = Navigator(dark, 'capture')
	go = GuideOutBase()
	guider = Guider(go, dark, 'capture')
	cam = Camera_test_g(go)

	runner = Runner(cam, navigator = nav, guider = guider)
	runner.start()
	runner.join()

def run_test():
	ui.namedWindow('capture')
	ui.namedWindow('capture_polar')
	
	cam = Camera_test()
	dark = Median(5)
	nav = Navigator(dark, 'capture')

	runner = Runner(cam, navigator = nav)
	runner.start()
	runner.join()

def run_test_2():
	ui.namedWindow('capture')
	ui.namedWindow('capture_v4l')
	ui.namedWindow('capture_polar')
	ui.namedWindow('capture_v4l_polar')

	dark1 = Median(5)
	dark2 = Median(5)

	cam1 = Camera_test()
	nav1 = Navigator(dark1, 'capture')

	nav = Navigator(dark2, 'capture_v4l')
	go = GuideOutBase()
	guider = Guider(go, dark2, 'capture_v4l')
	cam = Camera_test_g(go)

	runner = Runner(cam1, navigator = nav1)
	runner.start()
	
	runner2 = Runner(cam, navigator = nav, guider = guider)
	runner2.start()
	
	
	runner.join()
	runner2.join()

def run_test_2_gphoto():
	
	cam1 = Camera_gphoto()
	cam1.prepare()

	dark1 = Median(5)
	dark2 = Median(5)
	
	nav1 = Navigator(dark1, 'capture')
	focuser = Focuser('capture')

	nav = Navigator(dark2, 'capture_v4l')
	go = GuideOutBase()
	guider = Guider(go, dark2, 'capture_v4l')
	cam = Camera_test_g(go)

	ui.namedWindow('capture')
	ui.namedWindow('capture_v4l')
	ui.namedWindow('capture_polar')
	ui.namedWindow('capture_v4l_polar')
	ui.namedWindow('full_res')

	runner = Runner(cam1, navigator = nav1, focuser=focuser)
	runner.start()
	
	runner2 = Runner(cam, navigator = nav, guider = guider)
	runner2.start()
	
	
	runner.join()
	runner2.join()


def run_2():
	cam = Camera("/dev/video1")
	cam.prepare(1280, 960)
	
	cam1 = Camera_gphoto()
	cam1.prepare()

	dark1 = Median(5)
	dark2 = Median(5)
	
	nav1 = Navigator(dark1, 'capture')
	focuser = Focuser('capture')

	nav = Navigator(dark2, 'capture_v4l')
	go = GuideOut()
	guider = Guider(go, dark2, 'capture_v4l')

	ui.namedWindow('capture')
	ui.namedWindow('capture_v4l')
	ui.namedWindow('capture_polar')
	ui.namedWindow('capture_v4l_polar')
	ui.namedWindow('full_res')

	runner = Runner(cam1, navigator = nav1, focuser=focuser)
	runner.start()
	
	runner2 = Runner(cam, navigator = nav, guider = guider)
	runner2.start()
	
	
	runner.join()
	runner2.join()



if __name__ == "__main__":
	os.environ["LC_NUMERIC"] = "C"
	
	with ui:
		#run_gphoto()
	#run_test_2()
	#run_v4l2()
	#run_test_2_gphoto()
	#run_v4l2_g()
	#run_2()
		run_test()








