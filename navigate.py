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
	return cv2.normalize(img, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	

class Median:
	def __init__(self, n):
		self.n = n
		self.i = 0
		self.list = []
		self.res = None
	
	def add(self, im):
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

	def add_masked(self, im, pts):
		mask = np.zeros_like(im)
	
		white = np.iinfo(im.dtype).max
		for p in pts:
			cv2.circle(mask, p, 20, (white), -1)

		mask = cv2.blur(mask, (30, 30))
		mask = cv2.blur(mask, (30, 30))
		inv_mask = cv2.bitwise_not(mask)
		
		res = cv2.add(cv2.multiply(im, inv_mask, scale = 1.0 / white), cv2.multiply(self.res, mask, scale = 1.0 / white))
		
		self.add(res)
		#ui.imshow("dark", normalize(inv_mask))


	def get(self):
		return self.res

	def len(self):
		return len(self.list)

	def reset(self):
		self.i = 0
		self.list = []




def find_max(img, d, noise = 4, filt = False):
	#img = cv2.medianBlur(img, 5)
	bg = cv2.blur(img, (30, 30))
	bg = cv2.blur(bg, (30, 30))
	img = cv2.subtract(img, bg, dtype=cv2.CV_32FC1)

	img = cv2.GaussianBlur(img, (9, 9), 0)


	(mean, stddev) = cv2.meanStdDev(img)

	img = cv2.subtract(img, mean + stddev * noise)

	filt_img = img

	dilkernel = np.ones((d,d),np.uint8)
	dil = cv2.dilate(img, dilkernel)

	(h, w) = img.shape

	r,dil = cv2.threshold(dil,0,255,cv2.THRESH_TOZERO)
	
	locmax = cv2.compare(img, dil, cv2.CMP_GE)

	nonzero = None
	if stddev > 0.0:
		#nonzero = zip(*locmax.nonzero())
		nonzero = cv2.findNonZero(locmax)
	if nonzero is  not None:
		nonzero = nonzero[:, 0, :]
	else:
		nonzero = []
	ret = []
	
	#for (y, x) in nonzero:
	for (x, y) in nonzero:
		if (x < 1):
			continue
		if (y < 1):
			continue
		if (x > w - 2):
			continue
		if (y > h - 2):
			continue
		dx = img[y, x - 1] - 2 * img[y, x] + img[y, x + 1]
		dy = img[y - 1, x] - 2 * img[y, x] + img[y + 1, x]
		if dx != 0:
			xs = 0.5*(img[y, x - 1] - img[y, x + 1]) / dx
		else:
			xs = x
		if dy != 0:
			ys = 0.5*(img[y - 1, x] - img[y + 1, x]) / dy
		else:
			ys = y
		# y, x, flux, certainity: n_sigma
		ret.append((y + ys, x + xs, img[y, x] + mean + stddev * noise, math.erf((img[y, x] / stddev + noise) / 2**0.5) ** (w * h) ))
	ret = sorted(ret, key=lambda pt: pt[2], reverse=True)[:40]
	ret = sorted(ret, key=lambda pt: pt[0])
	
	if (filt):
		return ret, filt_img
	else:
		return ret

def match_idx(pt1, pt2, d, off = (0.0, 0.0)):
	if len(pt1) == 0 or len(pt2) == 0:
		return np.array([])
	maxflux1 = max(1, np.amax(np.array(pt1)[:, 2]))
	maxflux2 = max(1, np.amax(np.array(pt2)[:, 2]))
	match = []
	l = len(pt2)
	for i1, (y1orig, x1orig, flux1, cert1) in enumerate(pt1):
		y1 = y1orig + off[0]
		x1 = x1orig + off[1]
		i2 = bisect.bisect_left(pt2, (y1, x1))
		closest_dist = d ** 2 * 2;
		closest_idx = -1
		ii2 = i2;
		while (ii2 >=0 and ii2 < l):
			(y2, x2, flux2, cert2) = pt2[ii2]
			if (y2 < y1 - d):
				break
			dist = (y1 - y2) ** 2 + (x1 - x2) ** 2 + ((flux1 / maxflux1 - flux2 / maxflux2) * d) ** 2
			if (dist < closest_dist):
				closest_dist = dist
				closest_idx = ii2
			ii2 = ii2 - 1


		ii2 = i2;
		while (ii2 >=0 and ii2 < l):
			(y2, x2, flux2, cert2) = pt2[ii2]
			if (y2 > y1 + d):
				break
			dist = (y1 - y2) ** 2 + (x1 - x2) ** 2 + ((flux1 / maxflux1 - flux2 / maxflux2) * d) ** 2
			if (dist < closest_dist):
				closest_dist = dist
				closest_idx = ii2
			ii2 = ii2 + 1

		if (closest_idx >= 0):
			match.append((i1, closest_idx))
	return np.array(match)

def filt_match_idx(pt1, pt2, d, off = (0.0, 0.0)):
	match = match_idx(pt1, pt2, d, off)

	if match.shape[0] == 0:
		return np.array([]), np.array([]), np.array([])
		
	pt1m = np.array(np.take(pt1, match[:, 0], axis=0), np.float)
	pt2m = np.array(np.take(pt2, match[:, 1], axis=0), np.float)
	return pt1m, pt2m, match

def avg_pt(pt1m, pt2m, noise = 1):
	if pt1m.shape[0] > 1:
		dif = pt2m[:, 0:2] - pt1m[:, 0:2]
		weights = pt2m[:, 3] * pt1m[:, 3]
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
		weights = np.sqrt(pt2m[:, 3] * pt1m[:, 3])
		return v, weights
	
	v = np.array([0.0, 0.0])
	weights = np.array([0.0])
	return v, weights


class Stack:
	def __init__(self, dist = 20, ratio = 0.1):
		self.img = None
		self.dist = dist
		self.prev_pt = None
		self.xy = None
		self.off = np.array([0.0, 0.0])
		self.ratio = ratio
	
	def add(self, im):
		if im.dtype == np.uint8:
			im = cv2.multiply(im, 255.0, dtype=cv2.CV_16UC1)
		if (self.img is None):
			self.img = im
			return (0.0, 0.0)
			
		pt1 = self.prev_pt
		if pt1 is None:
			pt1 = self.get_xy()
		pt2, self.filt_img = find_max(im, 30, noise=5, filt=True)
		self.prev_pt = pt2
		
		self.pts = normalize(self.img)
		for p in pt1:
			cv2.circle(self.pts, (int(p[1]), int(p[0])), 13, (255), 1)
		
		for p in pt2:
			cv2.circle(self.pts, (int(p[1]), int(p[0])), 5, (255), 1)
	
		pt1m, pt2m, match = filt_match_idx(pt1, pt2, self.dist, self.off)
		off, weights = avg_pt(pt1m, pt2m)
		
		if np.max(weights) >= 0.99:
			self.off = off
		else:
			self.off = self.off * 0.9
		
		M = np.array([[1.0, 0.0, self.off[1]],
		              [0.0, 1.0, self.off[0]]])

		for p in pt2m:
			cv2.circle(self.pts, (int(p[1]), int(p[0])), 10, (255), 1)

		bg = cv2.blur(self.img, (30, 30))
		self.img = cv2.warpAffine(self.img, M[0:2,0:3], (im.shape[1], im.shape[0]), bg, borderMode=cv2.BORDER_TRANSPARENT);
		
		self.img = cv2.addWeighted(self.img, 1.0 - self.ratio, im, self.ratio, 0, dtype=cv2.CV_16UC1)
		self.xy = None
		return self.off

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
			self.xy = np.array(find_max(self.img, 20, noise=2))
		return self.xy
	
	def reset(self):
		self.img = None

class Navigator:
	def __init__(self, ui_capture):
		self.dark = Median(10)
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
			minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(im_sub)
			im_sub = cv2.add(im_sub, -minVal)
		else:
			im_sub = im
		
		if (self.dark.len() == 0):
			self.dark.add(im)
	
		off = self.stack.add(im_sub)
		filtered = self.stack.get()
		
		self.solver_off += off
		self.plotter_off += off

		if (self.dispmode == 'disp-orig'):
			ui.imshow(self.ui_capture, normalize(im))
		if (self.dispmode == 'disp-df-cor'):
			ui.imshow(self.ui_capture, normalize(im_sub))
		if (self.dispmode == 'disp-normal'):
			if self.plotter is not None:
				nm = normalize(filtered)
				for p in self.stack.get_xy():
					cv2.circle(nm, (int(p[1]), int(p[0])), 13, (255), 1)
		
				extra = []
				if self.polar_solved:
					transf_index = self.polar.transform_ra_dec_list(self.index_sources)
					extra = [ (ti[0], ti[1], "") for ti in transf_index ]
					#print "extra: ", extra
					
				ui.imshow(self.ui_capture, self.plotter.plot(nm, self.plotter_off, extra = extra))
			else:
				ui.imshow(self.ui_capture, normalize(filtered))
		if (self.dispmode == 'disp-match'):
			ui.imshow(self.ui_capture, normalize(self.stack.pts))
	
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
						ui.imshow(self.ui_capture + '_polar2', self.polar.plot2())
						ui.imshow(self.ui_capture + '_polar', self.polar.plot())
				elif self.polar_mode == 2:
					self.polar.phase2_set_tan(self.solver.wcs)
					ui.imshow(self.ui_capture + '_polar', self.polar.plot())
					ui.imshow(self.ui_capture + '_polar2', self.polar.plot2())
					
				self.ii += 1
				self.plotter = Plotter(self.solver.wcs)
				if (self.dispmode.startswith('disp-zoom-')):
					zoom = int(self.dispmode[len('disp-zoom-'):])
					plot = self.plotter.plot_viewfinder(normalize(filtered), zoom)
					ui.imshow(self.ui_capture, plot)
				self.plotter_off = self.solver_off
			else:
				self.ra = None
				self.dec = None
			self.solver = None
			self.solved_im = None

		if self.solver is None and i > 20 :
			xy = self.stack.get_xy()
			print "len", len(xy)
			if len(xy) > 4:
				self.solver_time = t
				self.solved_im = im
				self.filt_im = self.stack.filt_img
				self.solver = Solver(sources_list = xy, field_w = im.shape[1], field_h = im.shape[0], ra = self.ra, dec = self.dec, field_deg = self.field_deg)
				#self.solver = Solver(sources_img = filtered, field_w = im.shape[1], field_h = im.shape[0], ra = self.ra, dec = self.dec, field_deg = self.field_deg)
				self.solver.start()
				self.solver_off = np.array([0.0, 0.0])
		
	def cmd(self, cmd):
		if cmd == ' ' and self.solver is not None:
			self.solver.terminate(wait=True)
			self.field_deg = None

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
	def __init__(self, go, ui_capture):
		self.go = go
		self.reset()
		self.t0 = 0
		self.resp0 = []
		self.pt0 = []
		self.ui_capture = ui_capture

	def reset(self):
		self.mode = 0
		self.dark = Median(10)
		self.stack = Stack(4)
		self.off = (0.0, 0.0)
		self.go.out(0)
		self.cnt = 0
		self.pt0 = []
		self.ok = False
		self.capture_in_progress = False

	def get_df(self, im, filt_img):
		mask = np.zeros_like(im)
		h, w = mask.shape
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
		return darkframe(im, filt_img, pts)

	def cmd(self, cmd):
		if cmd == 'r':
			self.reset()
			self.mode = 1
		if cmd == "capture":
			self.capture_in_progress = True
		if cmd == "capture-finished":
			self.capture_in_progress = False

	def proc_frame(self, im, i):
		t = time.time()
		

		if (self.dark.len() >= 4):
			print "dark"
			im_sub = cv2.subtract(im, self.dark.get())
			minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(im)
			im_sub = cv2.add(im_sub, -minVal)
		else:
			print "no dark"
			im_sub = im

		debug = normalize(im)

		if self.mode==1:
			self.stack.add(im_sub)
			self.cnt = self.cnt + 1
			if self.cnt == 5:
				self.pt0 = find_max(self.stack.get(), 50, 4)
				if len(self.pt0) < 1:
					print "not enough pt", len(self.pt0)
					self.reset()
					return
				self.used_cnt = []
				self.cnt = 0
				self.dist = 1.0
				self.go.out(1)
				self.mode = 2
				self.t0 = t
				self.resp0 = []

		elif self.mode==2:
				
			self.dark.add(im)
			self.cnt += 1
			pt = find_max(im_sub, 50, 2)
			pt0, pt, match = filt_match_idx(self.pt0, pt, 30, self.off)
			if len(match) == 0:
				return
			
			distplus = np.where(np.linalg.norm(pt[:, 0:2] - pt0[:, 0:2], axis = 1) > 1)
			
			off, weights = avg_pt(pt0[distplus], pt[distplus], noise = 2)
			dist = np.linalg.norm(off)

			if (dist > 20):
				self.dark.add(im)
			
			self.resp0.append((t - self.t0, dist))
			
			if (dist > self.dist):
				self.dist = dist
				self.off = off
			
			print self.off, dist
			for p in pt:
				cv2.circle(debug, (int(p[1]), int(p[0])), 10, (255), 1)
			pt_ok = match[np.where(weights > 0), 0][0]
			self.used_cnt.extend(pt_ok)

			for i in pt_ok:
				p = self.pt0[i]
				cv2.circle(debug, (int(p[1] + self.off[1]), int(p[0] + self.off[0])), 13, (255), 1)

			if (dist > 100 and self.cnt > 12):
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
				
				self.pt0 = np.array(self.pt0)[np.where(np.bincount(self.used_cnt) > self.cnt / 2)]
				self.cnt = 0
				self.mode = 3
				
				self.go.out(-1, self.dist / self.pixpersec)

		elif self.mode==3:
			self.cnt += 1
			pt, filt_img = find_max(im_sub, 50, filt = True)
			pt0, pt, match = filt_match_idx(self.pt0, pt, 30, self.off)
			if match.shape[0] > 0:
				self.off, weights = avg_pt(pt0, pt)
				err = complex(*self.off) / self.ref_off
				
				self.resp0.append((t - self.t0, err.real))
			
				print "err:", err, err.real

				if (err.real > 30):
					self.dark.add(self.get_df(im, filt_img))

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
			pt = find_max(im_sub, 50)
			pt0, pt, match = filt_match_idx(self.pt0, pt, 30, self.off)
			if match.shape[0] > 0:
				self.off, weights = avg_pt(pt0, pt)
				err = complex(*self.off) / self.ref_off
				self.resp0.append((t - self.t0, err.real))

				err_corr = err.real + self.go.recent_avg(self.t_delay) * self.pixpersec
				
				aggresivnes = 0.2 + (t - self.t0) / 3600
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
		self.dark = Median(10)
		self.ui_capture = ui_capture
		self.dispmode = 3


	def cmd(self, cmd):
		if cmd == 'dark':
			self.dark.add(self.im)
		if cmd == '1':
			self.dispmode = 1
		if cmd == '2':
			self.dispmode = 2
		if cmd == '3':
			self.dispmode = 3
		if cmd == '4':
			self.dispmode = 4
			
	def proc_frame(self, im, i):
		self.im = im

		if (self.dark.len() > 0):
			im_sub = cv2.subtract(im, self.dark.get())
			minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(im_sub)
			im_sub = cv2.add(im_sub, -minVal, dtype=cv2.CV_8UC1)
		else:
			im_sub = im


		self.stack.add_simple(im_sub)
	
		if (self.dispmode == 1):
			ui.imshow(self.ui_capture, normalize(im))
		if (self.dispmode == 2):
			ui.imshow(self.ui_capture, normalize(im_sub))
		if (self.dispmode == 3):
			ui.imshow(self.ui_capture, normalize(self.stack.get()))
		if (self.dispmode == 4):
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
		profiler.add_function(Plotter.plot)
		profiler.add_function(Plotter.plot_viewfinder)
		
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
					profiler.print_stats()

					return
				elif cmd == 'navigator' and self.navigator is not None:
					mode = 'navigator'
				elif cmd == 'guider' and self.guider is not None:
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
		#time.sleep(0.5)
		print self.i
		#pil_image = Image.open("converted/IMG_%04d.jpg" % (146+self.i))
		#pil_image.thumbnail((1000,1000), Image.ANTIALIAS)
		#im = np.array(pil_image)
		im = cv2.imread("testimg17_" + str(self.i) + ".tif")
		M = np.array([[1.0, 0.0, self.x],
		              [0.0, 1.0, self.y]])
		bg = cv2.blur(im, (30, 30))
		im = cv2.warpAffine(im, M[0:2,0:3], (im.shape[1], im.shape[0]), bg, borderMode=cv2.BORDER_TRANSPARENT);

		t = os.path.getmtime("testimg17_" + str(self.i) + ".tif")
		self.i += self.step
		return im, t

class Camera_test_g:
	def __init__(self, go):
		self.i = 0
		self.err = 0.0
		self.go = go
	
	def cmd(self, cmd):
		print "camera:", cmd
	
	def capture(self):
		time.sleep(0.5)
		self.err += random.random() * 2 - 1.01
		corr = self.go.recent_avg()
		i = int((corr - self.go.recent_avg(1))  + self.err)
		print self.err, corr * 3, i
		im = cv2.imread("testimg16_" + str(i + 50) + ".tif")
		return im, None


def run_v4l2():
	ui.namedWindow('capture')
	cam = Camera("/dev/video1")
	cam.prepare(1280, 960)
	nav = Navigator('capture')

	runner = Runner(cam, navigator = nav)
	runner.start()
	runner.join()

def run_gphoto():
	cam = Camera_gphoto()
	cam.prepare()
	ui.namedWindow('capture')
	ui.namedWindow('full_res')
	nav = Navigator('capture')
	focuser = Focuser('capture')

	runner = Runner(cam, navigator = nav, focuser = focuser)
	runner.start()
	runner.join()


def run_v4l2_g():
	ui.namedWindow('capture')
	cam = Camera("/dev/video1")
	cam.prepare(1280, 960)

	nav = Navigator('capture')
	go = GuideOut()
	guider = Guider(go, 'capture')

	runner = Runner(cam, navigator = nav, guider = guider)
	runner.start()
	runner.join()

def run_test_g():
	ui.namedWindow('capture')
	nav = Navigator('capture')
	go = GuideOutBase()
	guider = Guider(go, 'capture')
	cam = Camera_test_g(go)

	runner = Runner(cam, navigator = nav, guider = guider)
	runner.start()
	runner.join()

def run_test():
	ui.namedWindow('capture')
	
	cam = Camera_test()
	nav = Navigator('capture')

	runner = Runner(cam, navigator = nav)
	runner.start()
	runner.join()

def run_test_2():
	ui.namedWindow('capture_gphoto')
	ui.namedWindow('capture_v4l')
	
	cam1 = Camera_test()
	nav1 = Navigator('capture_gphoto')

	nav = Navigator('capture_v4l')
	go = GuideOutBase()
	guider = Guider(go, 'capture_v4l')
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
	nav1 = Navigator('capture_gphoto')
	focuser = Focuser('capture_gphoto')

	nav = Navigator('capture_v4l')
	go = GuideOutBase()
	guider = Guider(go, 'capture_v4l')
	cam = Camera_test_g(go)

	ui.namedWindow('capture_gphoto')
	ui.namedWindow('capture_v4l')

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
	nav1 = Navigator('capture_gphoto')
	focuser = Focuser('capture_gphoto')

	nav = Navigator('capture_v4l')
	go = GuideOut()
	guider = Guider(go, 'capture_v4l')

	ui.namedWindow('capture_gphoto')
	ui.namedWindow('capture_v4l')

	runner = Runner(cam1, navigator = nav1, focuser=focuser)
	runner.start()
	
	runner2 = Runner(cam, navigator = nav, guider = guider)
	runner2.start()
	
	
	runner.join()
	runner2.join()



if __name__ == "__main__":
#    sys.exit(run_gphoto())
    #sys.exit(test_g())

    #run_v4l2_g()
    #run_v4l2()
    with ui:
    	#run_gphoto()
    	run_test()
    	#run_test_2_gphoto()
    	#run_v4l2_g()
    	#run_2()








