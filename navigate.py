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


from am import Solver, Plotter

import gphoto2 as gp
import sys
import io
import os.path
import time
from v4l2_camera import *

from serial_control import GuideOutBase, GuideOut

import random
from line_profiler import LineProfiler

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from gui import ui


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


	def get(self):
		return self.res

	def len(self):
		return len(self.list)

	def reset(self):
		self.i = 0
		self.list = []

def darkframe(im, pts):
	mask = np.zeros_like(im)
	
	fill = np.array(im, copy=True)
	
	maxv = np.iinfo(mask.dtype).max
	
	for p in pts:
		cv2.circle(fill, p, 10, (0), -1)
		cv2.circle(mask, p, 10, (maxv), -1)
		
	inv_mask = cv2.bitwise_not(mask)
	fill = cv2.blur(fill, (30,30))
	fill = cv2.blur(fill, (30,30))
	fill = cv2.blur(fill, (30,30))
	inv_mask = cv2.blur(inv_mask, (30,30))
	inv_mask = cv2.blur(inv_mask, (30,30))
	inv_mask = cv2.blur(inv_mask, (30,30))
	fill = cv2.divide(fill, inv_mask, scale=maxv)
	idx = (mask==0)
        fill[idx] = im[idx]
	return fill



def find_max(img, d, noise = 4, debug = False):
	#img = cv2.medianBlur(img, 5)
	bg = cv2.blur(img, (30, 30))
	bg = cv2.blur(bg, (30, 30))
	img = cv2.subtract(img, bg, dtype=cv2.CV_32FC1)

	img = cv2.GaussianBlur(img, (9, 9), 0)

	(mean, stddev) = cv2.meanStdDev(img)
	if stddev == 0:
		return []
	img = cv2.subtract(img, mean + stddev * noise)

	dilkernel = np.ones((d,d),np.uint8)
	dil = cv2.dilate(img, dilkernel)


	r,dil = cv2.threshold(dil,0,255,cv2.THRESH_TOZERO)
	
	locmax = cv2.compare(img, dil, cv2.CMP_GE)

	(h, w) = locmax.shape
	nonzero = zip(*locmax.nonzero())
	ret = []
	
	for (y, x) in nonzero:
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
	ret = sorted(ret, key=lambda pt: pt[2], reverse=True)[:50]
	ret = sorted(ret, key=lambda pt: pt[0])
	
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
	def __init__(self, dist = 20):
		self.img = None
		self.dist = dist
		self.xy = None
		self.off = np.array([0.0, 0.0])
	
	def add(self, im):
		if im.dtype == np.uint8:
			im = cv2.multiply(im, 255, dtype=cv2.CV_16UC1)
		if (self.img is None):
			self.img = im
			return (0.0, 0.0)
			
		pt1 = self.get_xy()
		pt2 = find_max(im, 30, noise=4, debug=True)
		
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
		
		self.img = cv2.addWeighted(self.img, 0.95, im, 0.05, 0, dtype=cv2.CV_16UC1)
		self.xy = None
		return self.off

	def add_simple(self, im):
		if im.dtype == np.uint8:
			im = cv2.multiply(im, 255, dtype=cv2.CV_16UC1)
		if (self.img is None or self.img.shape != im.shape):
			self.img = im
			return
		self.img = cv2.addWeighted(self.img, 0.9, im, 0.1, 0, dtype=cv2.CV_16UC1)
		self.xy = None
		return (0.0, 0.0)

	def get(self, dtype = np.uint8):
		if dtype == np.uint8:
			return cv2.divide(self.img, 255, dtype=cv2.CV_8UC1)
		else:
			return self.img

	def get_xy(self):
		if self.xy is None:
			self.xy = np.array(find_max(self.img, 20, noise=1))
		return self.xy
	
	def reset(self):
		self.img = None

class Navigator:
	def __init__(self):
		self.dark = Median(10)
		self.stack = Stack()
		self.solver = None
		self.solver_off = np.array([0.0, 0.0])
		self.dispmode = 3
		self.ra = None
		self.dec = None
		self.field_deg = None
		self.plotter = None
		self.plotter_off = np.array([0.0, 0.0])

	def proc_frame(self,im, i, key):
	
		if (self.dark.len() > 0):
			im_sub = cv2.subtract(im, self.dark.get())
			minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(im_sub)
			im_sub = cv2.add(im_sub, -minVal)
		else:
			im_sub = im
	
		off = self.stack.add(im_sub)
		med = self.stack.get()
		
		self.solver_off += off
		self.plotter_off += off

		if (self.dispmode == 1):
			ui.imshow('capture', normalize(im))
		if (self.dispmode == 2):
			ui.imshow('capture', normalize(im_sub))
		if (self.dispmode == 3):
			if self.plotter is not None:
				nm = normalize(med)
				for p in self.stack.get_xy():
					cv2.circle(nm, (int(p[1]), int(p[0])), 13, (255), 1)
		
				ui.imshow('capture', self.plotter.plot(nm, self.plotter_off))
			else:
				ui.imshow('capture', normalize(med))
		if (self.dispmode == 4):
			ui.imshow('capture', normalize(self.stack.pts))
	
		if self.solver is not None and not self.solver.is_alive():
			self.solver.join()
			if self.solver.solved:
				self.ra = self.solver.ra
				self.dec = self.solver.dec
				self.field_deg = self.solver.field_deg
			
				self.dark.add(darkframe(self.solved_im, self.solver.ind_sources))
				
				self.plotter = Plotter(self.solver.wcs)
				plot = self.plotter.plot_viewfinder(normalize(med), 5)
				ui.imshow('plot', plot)
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
				self.solved_im = im
				self.solver = Solver(sources_list = xy, field_w = im.shape[1], field_h = im.shape[0], ra = self.ra, dec = self.dec, field_deg = self.field_deg)
				self.solver.start()
				self.solver_off = np.array([0.0, 0.0])
			
		if key == 32 and self.solver is not None:
			self.solver.terminate(wait=True)
			self.field_deg = None

		if key == ord('d'):
			self.dark.add(im)
		
		if key == ord('1'):
			self.dispmode = 1
		if key == ord('2'):
			self.dispmode = 2
		if key == ord('3'):
			self.dispmode = 3
		if key == ord('4'):
			self.dispmode = 4

def fit_line(xylist):
	a = np.array(xylist)
	x = a[:, 0]
	y = a[:, 1]
	return np.polyfit(x, y, 1)

class Guider:
	def __init__(self, go):
		self.go = go
		self.reset()
		self.t0 = 0
		self.resp0 = []
		self.pt0 = []
	
	def reset(self):
		self.mode = 0
		self.dark = Median(10)
		self.stack = Stack(4)
		self.off = (0.0, 0.0)
		self.go.out(0)
		self.cnt = 0
		self.pt0 = []

	def get_df(self, im):
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
		return darkframe(im, pts)

	def proc_frame(self, im, i, key):
		t = time.time()
		if key == ord('r'):
			self.reset()
			self.mode = 1
		elif key == ord('s'):
			np.save("resp0_%d.npy" % self.t0, np.array(self.resp0))
			self.go.save("go_%d.npy" % self.t0)
			print "SAVED" 

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
				plt.ion()
				plt.axvline(x=0)
				plt.axhline(y=0)

		elif self.mode==2:
			plt.figure(1)
				
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
			
			plt.scatter(t - self.t0, dist)
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
				
				plt.plot([self.t_delay1, dt], [0, m * dt + c])
				plt.axhline(y=self.dist)
				plt.axvline(x=dt)
				plt.axvline(x=self.t_delay1)
				
				print "pixpersec", self.pixpersec, "pixperframe", self.pixperframe, "t_delay1", self.t_delay1
				
				self.pt0 = np.array(self.pt0)[np.where(np.bincount(self.used_cnt) > self.cnt / 2)]
				self.cnt = 0
				self.mode = 3
				
				self.go.out(-1, self.dist / self.pixpersec)

		elif self.mode==3:
			plt.figure(1)
			self.cnt += 1
			pt = find_max(im_sub, 50)
			pt0, pt, match = filt_match_idx(self.pt0, pt, 30, self.off)
			if match.shape[0] > 0:
				self.off, weights = avg_pt(pt0, pt)
				err = complex(*self.off) / self.ref_off
				
				plt.scatter(t - self.t0, err.real)
				self.resp0.append((t - self.t0, err.real))
			
				print "err:", err, err.real

				if (err.real > 30):
					self.dark.add(self.get_df(im))

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
				
					plt_x = np.array([self.t1 - self.t0 + self.t_delay2, self.t2 - self.t0])
					plt.plot(plt_x, plt_x * m + c)
					plt.plot([(self.dist - c) / m, dt], [self.dist, m * dt + c])
					plt.axvline(x=self.t1 - self.t0 + self.t_delay2)
					


					print "pixpersec_neg", self.pixpersec_neg, "pixperframe_neg", self.pixperframe_neg, "t_delay2", self.t_delay2
					self.t_delay = (self.t_delay1 + self.t_delay2) / 2
					if (self.t_delay < 0):
						self.t_delay = 0
				
					self.mode = 4


		elif self.mode==4:
			plt.figure(1)
			pt = find_max(im_sub, 50)
			pt0, pt, match = filt_match_idx(self.pt0, pt, 30, self.off)
			if match.shape[0] > 0:
				self.off, weights = avg_pt(pt0, pt)
				err = complex(*self.off) / self.ref_off
				self.resp0.append((t - self.t0, err.real))

				err_corr = err.real + self.go.recent_avg(self.t_delay) * self.pixpersec
				
				err_corr *= 0.5
				print "err:", err, err.real, "corr:", err_corr, "t_delay: ", self.t_delay
				if err_corr > 0.1:
					self.go.out(-1, -err_corr / self.pixpersec_neg)
				elif err_corr < -0.1:
					self.go.out(1, -err_corr / self.pixpersec)
				else:
					self.go.out(0)

				for p in pt:
					cv2.circle(debug, (int(p[1]), int(p[0])), 10, (255), 1)
				
				plt.figure(2)
				plt.plot(t - self.t0, self.go.recent_avg(), "go")
				
		if len(self.pt0) > 0:
			for p in self.pt0:
				cv2.circle(debug, (int(p[1]), int(p[0])), 13, (255), 1)

		cv2.imshow("capture", debug)


def set_config_choice(gp, camera, context, name, num):
	config = gp.check_result(gp.gp_camera_get_config(camera, context))
	OK, widget = gp.gp_widget_get_child_by_name(config, name)
	if OK >= gp.GP_OK:
		# set value
		value = gp.check_result(gp.gp_widget_get_choice(widget, num))
		print name, value
		gp.check_result(gp.gp_widget_set_value(widget, value))
	# set config
	gp.check_result(gp.gp_camera_set_config(camera, config, context))

def set_config_value(gp, camera, context, name, value):
	config = gp.check_result(gp.gp_camera_get_config(camera, context))
	OK, widget = gp.gp_widget_get_child_by_name(config, name)
	if OK >= gp.GP_OK:
		# set value
		print name, value
		gp.check_result(gp.gp_widget_set_value(widget, value))
	# set config
	gp.check_result(gp.gp_camera_set_config(camera, config, context))

def handle_focus_keys(gp, camera, context, key):
	if ('x' == chr(key & 255)):
		set_config_choice(gp, camera, context, 'manualfocusdrive', 2)
	if ('c' == chr(key & 255)):
		set_config_choice(gp, camera, context, 'manualfocusdrive', 1)
	if ('v' == chr(key & 255)):
		set_config_choice(gp, camera, context, 'manualfocusdrive', 0)
	if ('b' == chr(key & 255)):
		set_config_choice(gp, camera, context, 'manualfocusdrive', 4)
	if ('n' == chr(key & 255)):
		set_config_choice(gp, camera, context, 'manualfocusdrive', 5)
	if ('m' == chr(key & 255)):
		set_config_choice(gp, camera, context, 'manualfocusdrive', 6)

def capture_bulb(gp, camera, context, sec):
	set_config_value(gp, camera, context, 'shutterspeed', 'bulb')
	set_config_value(gp, camera, context, 'eosremoterelease', 'Immediate')
	time.sleep(sec)
	file_path = gp.check_result(gp.gp_camera_capture(camera, gp.GP_CAPTURE_IMAGE, context))
	print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
	target = os.path.join('/tmp', file_path.name)
	print('Copying image to', target)
	camera_file = gp.check_result(gp.gp_camera_file_get(camera, file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL, context))
	gp.check_result(gp.gp_file_save(camera_file, target))


zimage = Stack()
zdark = Median(10)

def proc_zoom_frame(im, i, key):
	global zimage

	if key == ord('d'):
		zdark.add(im)

	if (zdark.len() > 0):
		im_sub = cv2.subtract(im, zdark.get())
		minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(im_sub)
		im_sub = cv2.add(im_sub, -minVal, dtype=cv2.CV_8UC1)
	else:
		im_sub = im


	zimage.add_simple(im_sub)
	med = zimage.get()
	med = normalize(med)
	
	mask = cv2.compare(med, 128, cv2.CMP_GE)
	
	print "Nonzero size: ", cv2.countNonZero(mask)
	
	rgb = cv2.cvtColor(med,cv2.COLOR_GRAY2RGB)
	
	rgb[:,:, 1] = cv2.bitwise_and(med, cv2.bitwise_not(mask))
	
	ui.imshow('capture', rgb)


def run_gphoto():
	global cmd
	subprocess.call(['killall', 'gvfsd-gphoto2'])

	logging.basicConfig(
		format='%(levelname)s: %(name)s: %(message)s', level=logging.ERROR)
	gp.check_result(gp.use_python_logging())
	context = gp.gp_context_new()
	camera = gp.check_result(gp.gp_camera_new())
	gp.check_result(gp.gp_camera_init(camera, context))
	# required configuration will depend on camera type!
	set_config_choice(gp, camera, context, 'capturesizeclass', 2)

	set_config_choice(gp, camera, context, 'output', 1)
	set_config_choice(gp, camera, context, 'output', 0)
	sleep(2)

	ui.namedWindow('capture')
	ui.namedWindow('plot')
	nav = Navigator()

    	i = 0
	zoom = 1
	x = 3000
	y = 2000
	set_config_value(gp, camera, context, 'eoszoomposition', "%d,%d" % (x, y))

	while True:
		key=ui.waitKey(1)
		if (key == 27):
			break
		if key == ord('q'):
			break

		try:
			if i == 0:
				camera_file = gp.check_result(gp.gp_camera_capture_preview(camera, context))
			file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))

			handle_focus_keys(gp, camera, context, key)
			if key == ord('z'):
				if zoom == 1:
					zoom = 5
					minVal, maxVal, (minx, miny), (maxx, maxy) = cv2.minMaxLoc(nav.image.get())
					
					x = maxx * zoom - 300
					y = maxy * zoom - 300
					set_config_value(gp, camera, context, 'eoszoomposition', "%d,%d" % (x, y))
					set_config_value(gp, camera, context, 'eoszoom', '5')
					
				elif zoom == 5:
					zoom = 1
					set_config_value(gp, camera, context, 'eoszoom', '1')

			if key == ord('j'):
				x = max(100, x - 100)
				set_config_value(gp, camera, context, 'eoszoomposition', "%d,%d" % (x, y))
				zdark.reset()
			if key == ord('l'):
				x = x + 100
				set_config_value(gp, camera, context, 'eoszoomposition', "%d,%d" % (x, y))
				zdark.reset()
			if key == ord('i'):
				y = max(100, y - 100)
				set_config_value(gp, camera, context, 'eoszoomposition', "%d,%d" % (x, y))
				zdark.reset()
			if key == ord('k'):
				y = y + 100
				set_config_value(gp, camera, context, 'eoszoomposition', "%d,%d" % (x, y))
				zdark.reset()

			if key == ord('t'):
				capture_bulb(gp, camera, context, 5)

			camera_file = gp.check_result(gp.gp_camera_capture_preview(camera, context))
			pil_image = Image.open(io.BytesIO(file_data))
			#pil_image.save("testimg2_" + str(i) + ".tif")
			im = np.amin(np.array(pil_image), axis = 2)

			if zoom == 1:
				nav.proc_frame(im, i, key)
			else:
				proc_zoom_frame(im, i, key)
			
			i = i + 1

		except KeyboardInterrupt:
			break
		except:
			print "Unexpected error:", sys.exc_info()


	gp.check_result(gp.gp_camera_exit(camera, context))
	subprocess.call(['killall', 'astrometry-engine'])
	return 0

def run_v4l2():
	ui.namedWindow('plot')
	ui.namedWindow('capture')
	cam = Camera("/dev/video1")
	cam.prepare(1280, 960)
	nav = Navigator()

	i = 0
	while True:
#		try:
			key=ui.waitKey(50)
			if (key == 27):
				break
			im = cam.capture()
			#cv2.imwrite("testimg17_" + str(i) + ".tif", im)
			im = np.amin(im, axis = 2)
			nav.proc_frame(im, i, key)
			i = i + 1
#		except:
#			print "Unexpected error:", sys.exc_info()


def run_v4l2_g():
	ui.namedWindow('plot')
	ui.namedWindow('capture')
	cam = Camera("/dev/video1")
	cam.prepare(1280, 960)
	go = GuideOut()
	guider = Guider(go)

	i = 0
	while True:
		print i
		key=ui.waitKey(50)
		if (key == 27):
			break
		im = cam.capture()
		#cv2.imwrite("testimg10_" + str(i) + ".tif", im)
		im = np.amin(im, axis = 2)
		guider.proc_frame(im, i, key)
		i = i + 1


def test():
	ui.namedWindow('plot')
	ui.namedWindow('capture')
	nav = Navigator()
	i = 0
	while True:
		key=ui.waitKey(20)
		if (key == 27):
			break
		#pil_image = Image.open("testimg10_" + str(i) + ".tif")
		#pil_image = Image.open("/home/nadvornik/rawmk/collection/2015-09-12/img_" + str(7780+i) + ".jpg")
		#pil_image.thumbnail((1000,1000), Image.ANTIALIAS)
		#im = np.amin(np.array(pil_image), axis = 2)
		print i
		im = cv2.imread("testimg16_" + str(i) + ".tif")
		im = np.amin(im, axis = 2)
		#im = cv2.multiply(im, 255, dtype=cv2.CV_16UC1)

		nav.proc_frame(im, i, key)
		i = i + 1
		#if (i == 200):
		#	break

def test_g():
	ui.namedWindow('plot')
	ui.namedWindow('capture')
	
	go = GuideOutBase()
	guider = Guider(go)
	err = 0
	while True:
		err += random.random() * 2 - 1.01
		#err += 0.2
		corr = go.recent_avg()
		i = int((corr - go.recent_avg(1))  + err)
		print err, corr * 3, i
		key=ui.waitKey(500)
		if (key == 27):
			break
#		pil_image = Image.open("testimg2_" + str(i + 377) + ".tif")
#		pil_image = Image.open("testimg11_" + str(i + 60) + ".tif")
#		pil_image = Image.open("testimg" + str(i) + ".tif")
#		pil_image.thumbnail((1000,1000), Image.ANTIALIAS)
		#im = np.amin(np.array(pil_image), axis = 2)
		im = cv2.imread("testimg12_" + str(i + 50) + ".tif")
		im = np.amin(im, axis = 2)
		im = cv2.multiply(im, 255, dtype=cv2.CV_16UC1)

		guider.proc_frame(im, i, key)
		if guider.t0 > 0:
			plt.figure(2)
			plt.plot(time.time() - guider.t0, err + guider.go.recent_avg(), "ro")





if __name__ == "__main__":
#    sys.exit(run_gphoto())
    #sys.exit(test_g())
    profiler = LineProfiler()
    profiler.add_function(Navigator.proc_frame)
    profiler.add_function(Stack.add)
    profiler.add_function(find_max)
    profiler.enable_by_count()

    #run_v4l2_g()
    #run_v4l2()
    with ui:
    	test()
    profiler.print_stats()


#pil_image = Image.open("img_4316.jpg")
#cmd = astrometry_start(pil_image)

#while cmd.poll() is None:
#	sleep(1)

#astrometry_finish(cmd)

#print cmd.poll






