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

from guide_out import GuideOut

import random
from line_profiler import LineProfiler

from gui import ui
from cmd import cmdQueue

from stacktraces import stacktraces
import json

from PIL import Image;


class Status:
	def __init__(self, conf_file):
		try:
			with open(conf_file) as data_file:    
				self.status = json.load(data_file)
			self.status["conf_file"] = conf_file
		except:
			self.status = {"conf_file" : conf_file}
		ui.set_status(self)
		
	def save(self):
		with open(self.status["conf_file"], "w") as outfile:
			json.dump(self.status, outfile, indent=4)
	
	def to_json(self):
		return json.dumps(self.status)

	def path(self, p):
		def get_or_create(d, k):
			if k not in d:
				d[k] = {}
			return d[k]
			
		return reduce(get_or_create, p, self.status)


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
		if self.res is None:
			return
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

centroid_mat_cache = {}
def centroid(a, centroid_size):
	if centroid_size not in centroid_mat_cache:
		centroid_mat_x = np.array([[x for x in range(-centroid_size, centroid_size + 1) ] for y in range(-centroid_size, centroid_size + 1) ], dtype=np.float)
		centroid_mat_y = np.array([[y for x in range(-centroid_size, centroid_size + 1) ] for y in range(-centroid_size, centroid_size + 1) ], dtype=np.float)
		centroid_mat_cache[centroid_size] = (centroid_mat_x, centroid_mat_y)
	else:
		(centroid_mat_x, centroid_mat_y) = centroid_mat_cache[centroid_size]
		

	s = cv2.sumElems(a)[0]
	if s == 0.0:
		return 0, 0
	x = cv2.sumElems(cv2.multiply(a, centroid_mat_x, dtype=cv2.CV_32FC1))[0] / s
	y = cv2.sumElems(cv2.multiply(a, centroid_mat_y, dtype=cv2.CV_32FC1))[0] / s
	return x, y
	

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
		
		centroid_size = 7
	
		for (y, x, v) in zip(locmax[0][ordmax], locmax[1][ordmax], valmax[ordmax]):
			if (v <= 0.0):
				continue
			if (x < centroid_size):
				continue
			if (y + self.y1 < centroid_size):
				continue
			if (x > w - centroid_size - 1):
				continue
			if (y + self.y1 > h - centroid_size - 1):
				continue
			xs, ys = centroid(imge[y + self.y1e0 - centroid_size : y + self.y1e0 + centroid_size + 1, x - centroid_size : x + centroid_size + 1], centroid_size)
			#print "centroid", xs, ys, xs2, ys2
			
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
	if len(joined) == 0:
		return []

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

def pairwise_dist(x):
	b = np.dot(x, x.T)
	q = np.diag(b)[:, None]
	return np.sqrt(q + q.T - 2 * b)

def match_triangle(pt1, pt2, maxdif = 5.0, maxdrift = 10, off = (0.0, 0.0)):
	if len(pt1) == 0 or len(pt2) == 0:
		return match_take(pt1, pt2, [])
	
	ord1 = np.argsort(pt1[:, 2])[::-1]
	ord2 = np.argsort(pt2[:, 2])[::-1]
	
	pt1s = pt1[ord1][:12]
	pt2s = pt2[ord2][:12]
	
	dist1 = pairwise_dist(pt1s[:, 0:2])
	dist2 = pairwise_dist(pt2s[:, 0:2])
	
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
	
	m = pt_func(pt1, pt2, weights)
	
	pt1t = np.insert(pt1, 2, 1.0, axis=1).dot(m).A
	
	d2 = np.sum((pt2 - pt1t)**2, axis = 1)
	var = np.sum(d2 * weights) / sumw
	weights[np.where(d2 > var * noise**2)] = 1.0
	
	m = pt_func(pt1, pt2, weights)
	
	return m, weights
	
class Stack:
	def __init__(self, ratio = 0.1):
		self.img = None
		self.match = None
		self.prev_pt = []
		self.prev_pt_verified = []
		self.xy = None
		self.ratio = ratio
	
	def add(self, im, show_match = False):
		if im.dtype == np.uint8:
			im = cv2.multiply(im, 255.0, dtype=cv2.CV_16UC1)
		if (self.img is None):
			self.img = im
			self.match = im
			return np.matrix([[1., 0], [0, 1.], [0, 0]])
			
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
			M = np.matrix([[1., 0], [0, 1.], [0, 0]])
		else:
			M, weights = pt_transform_opt(pt1m, pt2m, pt_func = pt_translation_rotate)
			
			pt1 = self.prev_pt
	
			pt1m, pt2m, match = match_closest(pt1, pt2, 5, M = M)
			#off, weights = avg_pt(pt1m, pt2m)
			#print "off2", off 
			#print match
			self.prev_pt_verified = pt2m
			self.prev_pt = pt2

			M, weights = pt_transform_opt(pt1m, pt2m, pt_func = pt_translation_rotate)
			
			Mt = np.array([[ M[0, 0], M[0, 1], M[2,1] ],
			               [ M[1, 0], M[1, 1], M[2,0] ]])

			self.img = cv2.warpAffine(self.img, Mt, (im.shape[1], im.shape[0]));
			
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
		return M

	def add_simple(self, im):
		if im.dtype == np.uint8:
			im = cv2.multiply(im, 255, dtype=cv2.CV_16UC1)
		if (self.img is None or self.img.shape != im.shape):
			self.img = im
			return
		self.img = cv2.addWeighted(self.img, 1.0 - self.ratio, im, self.ratio, 0, dtype=cv2.CV_16UC1)
		self.xy = None
		return (0.0, 0.0)

	def get(self, dtype = None):
		if dtype == np.uint8:
			return cv2.divide(self.img, 255, dtype=cv2.CV_8UC1)
		else:
			return self.img

	def get_xy(self):
		if self.xy is None:
			self.xy = np.array(find_max(self.img, 12, n = 30))

		return self.xy

	def get_xy_verified(self):
		return self.prev_pt_verified

	def reset(self):
		self.img = None

def _plot_bg(window, status, func, *args, **kwargs):
	disp = func(*args, **kwargs)
	cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

	ui.imshow(window, disp)

def plot_bg(*args, **kwargs):
	#_plot_bg(*args, **kwargs)
	threading.Thread(target=_plot_bg, args = args, kwargs = kwargs).start()

class Navigator:
	def __init__(self, status, dark, polar, tid, polar_tid = None):
		self.status = status
		self.dark = dark
		self.stack = Stack()
		self.solver = None
		self.solver_off = np.array([0.0, 0.0])
		self.status.setdefault("dispmode", 'disp-normal')
		try:
			if self.status['lensname'] == self.status['camera']['lensname']:
				self.status.setdefault('field_deg', None)
			else:
				self.status['field_deg'] = None
		except:
			self.status['field_deg'] = None
		try:
			self.status['lensname'] = self.status['camera']['lensname']
		except:
			pass
		
		self.wcs = None
		self.plotter = None
		self.plotter_off = np.array([0.0, 0.0])
		self.tid = tid
		self.polar = polar
		self.polar_tid = polar_tid
		self.index_sources = []
		self.status['i_solved'] = 0
		self.status['i_solver'] = 0
		self.status['t_solved'] = 0
		self.status['t_solver'] = 0
		self.prev_t = 0
		self.status['ra'], self.status['dec'] = self.polar.zenith()
		self.status['max_radius'] = 100
		self.status['radius'] = self.status['max_radius']

	def proc_frame(self,im, i, t = None):
		self.i = i
		if im.ndim > 2:
			im = cv2.min(cv2.min(im[:, :, 0], im[:, :, 1]), im[:, :, 2])

		self.im = im
		
		if t is None:
			t = time.time()
		if (self.dark.len() > 2):
			im_sub = cv2.subtract(im, self.dark.get())
		else:
			im_sub = im
			
		bg = cv2.blur(im_sub, (30, 30))
		bg = cv2.blur(bg, (30, 30))
		im_sub = cv2.subtract(im_sub, bg)

		if i < 6:
			self.dark.add(im)

		M = self.stack.add(im_sub, show_match=(self.status['dispmode'] == 'disp-match'))
		filtered = self.stack.get()
		
		self.solver_off = np.insert(self.solver_off, 2, 1.0).dot(M).A1
		self.plotter_off = np.insert(self.plotter_off, 2, 1.0).dot(M).A1

		try:
			fps = 1.0 / (t - self.prev_t)
		except:
			fps = 0

		if self.solver is not None and not self.solver.is_alive():
			self.solver.join()
			if self.solver.solved:
				self.status['ra'] = self.solver.ra
				self.status['dec'] = self.solver.dec
				self.status['field_deg'] = self.solver.field_deg
				self.status['radius'] = self.status['field_deg']
				self.wcs = self.solver.wcs
			
				self.dark.add_masked(self.solved_im, self.solver.ind_sources)
				
				self.index_sources = self.solver.ind_radec
				self.plotter = Plotter(self.wcs)
				self.plotter_off = self.solver_off
				self.status['i_solved'] = self.status['i_solver']
				self.status['t_solved'] = self.status['t_solver']
				#print "self.solver.ind_radec", self.solver.ind_radec
				#self.solver.wcs.write_to("log_%d.wcs" % self.ii)
				#subprocess.call(['touch', '-r', "testimg17_" + str(i) + ".tif", "log_%d.wcs" % self.ii])
				if self.polar.mode == 'solve':
					self.polar.set_pos_tan(self.wcs, self.status['t_solver'], self.tid)
				if self.polar_tid is not None:
					self.polar.solve()
					
			else:
				if self.status['radius'] > 0 and self.status['radius'] < 70:
					self.status['radius'] = self.status['radius'] * 2 + 15
				else:
					self.status['ra'], self.status['dec'] = self.polar.zenith()
					self.status['radius'] = self.status['max_radius']
					self.wcs = None
			self.solver = None
			self.solved_im = None

		if self.solver is None and i > 20 :
			xy = self.stack.get_xy()
			#print "len", len(xy)
			if len(xy) > 8:
				self.status['i_solver'] = i
				self.status['t_solver'] = t
				self.solved_im = im
				self.solver = Solver(sources_list = xy, field_w = im.shape[1], field_h = im.shape[0], ra = self.status['ra'], dec = self.status['dec'], field_deg = self.status['field_deg'], radius = self.status['radius'])
				#self.solver = Solver(sources_img = filtered, field_w = im.shape[1], field_h = im.shape[0], ra = self.status['ra'], dec = self.status['dec'], field_deg = self.status['field_deg'])
				self.solver.start()
				self.solver_off = np.array([0.0, 0.0])
		if self.polar.mode == 'solve' and self.polar_tid is not None:
			polar_plot = self.polar.plot2()
			p_status = "#%d %s solv#%d r:%.1f fps:%.1f" % (i, self.polar.mode, i - self.status['i_solver'], self.status['radius'], fps)
			cv2.putText(polar_plot, p_status, (10, polar_plot.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
			ui.imshow(self.polar_tid, polar_plot)
		elif self.polar.mode == 'adjust' and self.wcs is not None:
			self.polar.set_pos_tan(self.wcs, self.status['t_solver'], self.tid, off = self.plotter_off)
			if self.polar_tid is not None:
				polar_plot = self.polar.plot2()
				p_status = "#%d %s solv#%d r:%.1f fps:%.1f" % (i, self.polar.mode, i - self.status['i_solved'], self.status['radius'], fps)
				cv2.putText(polar_plot, p_status, (10, polar_plot.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
				ui.imshow(self.polar_tid, polar_plot)
			
		status = "#%d %s %s  solv#%d r:%.1f fps:%.1f" % (i, self.status['dispmode'], self.polar.mode, i - self.status['i_solver'], self.status['radius'], fps)
		if (self.status['dispmode'] == 'disp-orig'):
			disp = normalize(im)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			ui.imshow(self.tid, disp)
		elif (self.status['dispmode'] == 'disp-df-cor'):
			disp = normalize(im_sub)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			ui.imshow(self.tid, disp)
		elif (self.status['dispmode'] == 'disp-normal'):
			disp = normalize(filtered)
			for p in self.stack.get_xy():
				cv2.circle(disp, (int(p[1]), int(p[0])), 13, (255), 1)
			if self.plotter is not None:
		
				extra_lines = []
				
				if self.polar.mode == 'adjust':
					transf_index = self.polar.transform_ra_dec_list(self.index_sources)
					extra_lines = [ (si[0], si[1], ti[0], ti[1]) for si, ti in zip(self.index_sources, transf_index) ]
					
				plot_bg(self.tid, status, self.plotter.plot, disp, self.plotter_off, extra_lines = extra_lines)
			else:
				cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
				ui.imshow(self.tid, disp)
		elif (self.status['dispmode'].startswith('disp-zoom-')):
			if self.plotter is not None:
				zoom = self.status['dispmode'][len('disp-zoom-'):]
				plot_bg(self.tid, status, self.plotter.plot, normalize(filtered), self.plotter_off, scale=zoom)
			else:
				disp = normalize(filtered)
				cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
				ui.imshow(self.tid, disp)
				
		elif (self.status['dispmode'] == 'disp-match'):
			disp = self.stack.match
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			ui.imshow(self.tid, disp)
	


		self.prev_t = t
		
	def cmd(self, cmd):
		if cmd == 'solver-reset':
			if self.solver is not None:
				self.solver.terminate(wait=True)
			self.status['field_deg'] = None
			self.solver = None
			self.plotter = None
			self.status['ra'], self.status['dec'] = self.polar.zenith()
			self.status['radius'] = self.status['max_radius']

		if cmd == 'solver-retry':
			if self.solver is not None:
				self.solver.terminate(wait=False)
			self.status['ra'], self.status['dec'] = self.polar.zenith()
			self.status['radius'] = self.status['max_radius']

		if cmd == 'dark':
			self.dark.add(self.im)
		
		if cmd.startswith('disp-'):
			self.status['dispmode'] = cmd
		if cmd == 'save':
			cv2.imwrite(self.tid + str(int(time.time())) + ".tif", self.stack.get())

		if cmd == 'polar-reset':
			if self.polar_tid is not None:
				self.polar.reset()

		if cmd == 'polar-align':
			self.polar.set_mode('adjust')

		if cmd.startswith('gps') and self.polar_tid is not None:
			try:
				str_gps = cmd[len('gps'):]
				(lat, lon) = [float(n) for n in str_gps.split(',')]
				self.polar.set_gps((lat, lon))
			except:
				print "Error: " +  sys.exc_info().__str__()
	
	def proc_full_res(self, jpg):
		t = time.time()
		pil_image = Image.open(jpg)
		im_c = np.array(pil_image)
		im = np.amin(im_c, 2)

		pts = find_max(im, 12, 100)

		solver = Solver(sources_list = pts, field_w = im.shape[1], field_h = im.shape[0], ra = self.status['ra'], dec = self.status['dec'], field_deg = self.status['field_deg'], radius = 100)
		solver.start()
		solver.join()
		if solver.solved:
			print "full-res solved:", solver.ra, solver.dec
			self.status['ra'] = solver.ra
			self.status['dec'] = solver.dec
			self.status['field_deg'] = solver.field_deg
			self.status['radius'] = solver.field_deg
			self.polar.set_pos_tan(solver.wcs, t, "full-res")

			if (self.status['dispmode'].startswith('disp-zoom-')):
				zoom = self.status['dispmode'][len('disp-zoom-'):]
			else:
				zoom = 1
			
			plotter=Plotter(solver.wcs)
        		plot = plotter.plot(im_c, scale = zoom)
        		ui.imshow('full_res', plot)

		else:
			print "full-res not solved"

def fit_line(xylist):
	a = np.array(xylist)
	x = a[:, 0]
	y = a[:, 1]
	return np.polyfit(x, y, 1)

class Guider:
	def __init__(self, status, go, dark, tid):
		self.status = status
		self.status.setdefault('aggressivness', 0.6)
		self.go = go
		self.dark = dark
		self.reset()
		self.t0 = 0
		self.resp0 = []
		self.pt0 = []
		self.tid = tid
		self.prev_t = 0

	def reset(self):
		self.status['mode'] = 'start'
		self.status['t_delay'] = None
		self.status['t_delay1'] = None
		self.status['t_delay2'] = None
		self.status['pixpersec'] = None
		self.status['pixpersec_neg'] = None
		self.status['seq'] = 'seq-stop'
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
		if cmd.startswith('aggressivness-'):
			try:
				self.status['aggressivness'] = float(cmd[len('aggressivness-'):])
			except:
				pass

		if cmd.startswith('seq-'):
			self.status['seq'] = cmd

	def proc_frame(self, im, i):

		t = time.time()

		if im.ndim > 2:
			im = cv2.min(cv2.min(im[:, :, 0], im[:, :, 1]), im[:, :, 2])
		
		if len(self.pt0) == 0:
			cmdQueue.put('navigator')
			self.go.out(0)

		if (self.dark.len() >= 4):
			im_sub = cv2.subtract(im, self.dark.get())
		else:
			im_sub = im


		bg = cv2.blur(im_sub, (30, 30))
		bg = cv2.blur(bg, (30, 30))
		im_sub = cv2.subtract(im_sub, bg)

		disp = normalize(im_sub)
		pt = find_max(im_sub, 20, n = 30)


		try:
			fps = 1.0 / (t - self.prev_t)
		except:
			fps = 0
		
		status = "#%d Guider:%s fps:%.1f" % (i, self.status['mode'], fps)

		if self.status['mode'] == 'start':
			self.used_cnt = []
			self.cnt = 0
			self.dist = 1.0
			self.go.out(1)
			self.status['mode'] = 'move'
			self.t0 = time.time()
			self.resp0 = []

		elif self.status['mode'] == 'move':
				
			self.cnt += 1
			
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
				#print "triangle", off, match
			
				pt0, pt, match = match_closest(self.pt0, pt, 5, off)
			
			if len(match) > 0:
			
				off, weights = avg_pt(pt0, pt, noise = 3)
				#print "weights", weights 
				dist = np.linalg.norm(off)

				if (dist > 20):
					self.dark.add(im)
					self.resp0.append((t - self.t0, dist, 0))
			
				if (dist > self.dist):
					self.dist = dist
					self.off = off
					self.off_t = t
			
				#print off, dist
				pt_ok = match[np.where(weights > 0), 0][0]
				self.used_cnt.extend(pt_ok)

				for i in pt_ok:
					p = self.pt0[i]
					cv2.circle(disp, (int(p[1] + self.off[1]), int(p[0] + self.off[0])), 13, (255), 1)

				status += " dist:%.1f" % (dist)

				if (self.dist > 100 and len(self.resp0) > 12):
					self.t1 = time.time()
					dt = t - self.t0
					self.go.out(-1)
				
					aresp = np.array(self.resp0)
					aresp1 = aresp[aresp[:, 1] > 10]
					m, c = np.polyfit(aresp1[:, 0], aresp1[:, 1], 1)

					self.status['pixpersec'] = m
					self.status['t_delay1'] = max(-c / m, 0.5)
					
					self.pixperframe = self.status['pixpersec'] * dt / self.cnt
					self.dist = m * dt + c
					self.ref_off = complex(*self.off) / dist
				
					print "pixpersec", self.status['pixpersec'], "pixperframe", self.pixperframe, "t_delay1", self.status['t_delay1']
				
					self.pt0 = np.array(self.pt0)[np.where(np.bincount(self.used_cnt) > self.cnt / 3)]
				
					self.cnt = 0
					self.status['mode'] = 'back'
				
					self.go.out(-1, self.dist / self.status['pixpersec'])
			for p in pt:
				cv2.circle(disp, (int(p[1]), int(p[0])), 10, (255), 1)

		elif self.status['mode'] == 'back':
			self.cnt += 1
			pt1m, pt2m, match = match_triangle(self.pt0, pt, 5, 50, self.off)
			if len(match) > 0:
				off, weights = avg_pt(pt1m, pt2m)
				#print "triangle", off, match
			
				pt0, pt, match = match_closest(self.pt0, pt, 5, off)
				
			if len(match) > 0:
				self.off, weights = avg_pt(pt0, pt)
				err = complex(*self.off) / self.ref_off
				
				self.resp0.append((t - self.t0, err.real, err.imag))
			
				status += " err:%.1f %.1f t_delay1:%.1f" % (err.real, err.imag, self.status['t_delay1'])

				if (err.real > 30):
					self.dark_add_masked(im)

				for p in pt:
					cv2.circle(disp, (int(p[1]), int(p[0])), 10, (255), 1)
				self.go.out(-1, err.real / self.status['pixpersec'])
				
				if err.real < self.status['pixpersec'] * self.status['t_delay1'] + self.pixperframe:
					self.t2 = t
					dt = self.t2 - self.t1
					
					aresp = np.array(self.resp0)
					aresp1 = aresp[aresp[:, 0] > self.t1 + self.status['t_delay1'] - self.t0]
					m, c = np.polyfit(aresp1[:, 0], aresp1[:, 1], 1)

					self.status['pixpersec_neg'] = m
					self.status['t_delay2'] = max(0.5, (c + self.status['t_delay1'] * self.status['pixpersec']) / (self.status['pixpersec'] - self.status['pixpersec_neg']) - self.t1 + self.t0)


					self.pixperframe_neg = self.status['pixpersec_neg'] * dt / self.cnt
				
					print "pixpersec_neg", self.status['pixpersec_neg'], "pixperframe_neg", self.pixperframe_neg, "t_delay2", self.status['t_delay2']
					self.status['t_delay'] = (self.status['t_delay1'] + self.status['t_delay2']) / 2
				
					self.status['mode'] = 'track'


		elif self.status['mode'] == 'track':
			pt1m, pt2m, match = match_triangle(self.pt0, pt, 5, 30, self.off)
			if len(match) > 0:
				off, weights = avg_pt(pt1m, pt2m)
				#print "triangle", off, match
			
				pt0, pt, match = match_closest(self.pt0, pt, 5, off)
				
			if len(match) > 0:
				self.off, weights = avg_pt(pt0, pt)
				err = complex(*self.off) / self.ref_off
				self.resp0.append((t - self.t0, err.real, err.imag))

				t_proc = time.time() - t

				err_corr = err.real + self.go.recent_avg(self.status['t_delay'] + t_proc, self.status['pixpersec'], self.status['pixpersec_neg'])
				
				
				err_corr *= self.status['aggressivness']
				status += " err:%.1f %.1f corr:%.1f t_d:%.1f t_p:%.1f" % (err.real, err.imag, err_corr, self.status['t_delay'], t_proc)
				if err_corr > 0.1:
					self.go.out(-1, -err_corr / self.status['pixpersec_neg'])
				elif err_corr < -0.1:
					self.go.out(1, -err_corr / self.status['pixpersec'])
				else:
					self.go.out(0)
				
				self.ok = (err.real < 2 and err.real > -2)
				if not self.capture_in_progress and (self.status['seq'] == 'seq-guided' and self.ok or self.status['seq'] == 'seq-unguided'):
					cmdQueue.put('capture')
					self.capture_in_progress = True
				
				for p in pt:
					cv2.circle(disp, (int(p[1]), int(p[0])), 10, (255), 1)
				

				if i % 100 == 0:
					np.save("resp0_%d.npy" % self.t0, np.array(self.resp0))
					self.go.save("go_%d.npy" % self.t0)
					print "SAVED" 
				
		if len(self.pt0) > 0:
			for p in self.pt0:
				cv2.circle(disp, (int(p[1]), int(p[0])), 13, (255), 1)

		cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
		ui.imshow(self.tid, disp)
		self.prev_t = t

def smooth(x,window_len=11,window='hanning'):

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.append(np.append([x[0] for i in range(0, window_len)], x),[x[-1] for i in range(0, window_len)])
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='full')
    return y[window_len + window_len / 2: window_len + window_len / 2 + x.size]

class Focuser:
	def __init__(self, tid, dark = None):
		self.stack = Stack(ratio=0.3)
		if dark is None:
			self.dark = Median(3)
		else:
			self.dark = dark
		self.tid = tid
		self.dispmode = 'disp-orig'
		self.phase = 'wait'
		self.phase_wait = 0
		self.hfr = Focuser.hfr_size
		self.focus_yx = None
		self.prev_t = 0

	hfr_size = 30
	hfr_mat_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hfr_size * 2 + 1, hfr_size * 2 + 1))
	hfr_mat = cv2.multiply(np.array([[(x**2 + y**2)**0.5 for x in range(-hfr_size, hfr_size + 1) ] for y in range(-hfr_size, hfr_size + 1) ], dtype=np.float), hfr_mat_mask, dtype=cv2.CV_32FC1)
	@staticmethod
	def hfr(a):
		s = cv2.sumElems(cv2.multiply(a,  Focuser.hfr_mat_mask, dtype=cv2.CV_32FC1))[0]
		if s == 0.0:
			return Focuser.hfr_size
		r = cv2.sumElems(cv2.multiply(a,  Focuser.hfr_mat, dtype=cv2.CV_32FC1))[0] / s
		return r

	@staticmethod
	def v_param(v_curve):
		v_len = len(v_curve)
		side_len = int(v_len * 0.4)

		smooth_size = side_len / 3 * 2 + 1
		v_curve_s = smooth(v_curve, smooth_size, 'flat')
		v_curve_s = smooth(v_curve_s, smooth_size, 'flat')
		
		derived = np.gradient(v_curve_s)
		#print derived.tolist()
				
		i1 = np.argmin(derived)
		i2 = np.argmax(derived)
				
		m1 = derived[i1]
		m2 = derived[i2]
				
		c1 = v_curve_s[i1] - i1 * m1
		c2 = v_curve_s[i2] - i2 * m2
				
		#m1, c1 = np.polyfit(range(0, side_len), self.v_curve[0:side_len], 1)
		#m2, c2 = np.polyfit(range(v_len - side_len, v_len), self.v_curve[v_len - side_len: v_len], 1)
		xmin =  (c2 - c1) / (m1 - m2)
		side_len = xmin * 0.8
		print "v_len", v_len, "side_len", side_len, "m1", m1, "c1", c1, "m2", m2, "c2", c2, "xmin", xmin
		
		return xmin, side_len, smooth_size, c1, m1, c2, m2, v_curve_s
	
	@staticmethod
	def v_shift(v_curve2, smooth_size, c1, m1):
		v_curve2_s = smooth(np.array(v_curve2), smooth_size, 'flat')
		v_curve2_s = smooth(v_curve2_s, smooth_size, 'flat')
		derived = np.gradient(v_curve2_s)
		i1 = np.argmin(derived)
		y = v_curve2_s[i1]
		print "i1", i1
		hyst = (y - c1) / m1 - i1
		print "hyst", hyst
		return hyst, v_curve2_s
	

	def cmd(self, cmd):
		if cmd == 'dark':
			self.dark.add(self.im)
		if cmd.startswith('disp-'):
			self.dispmode = cmd
		if cmd == 'focus':
			self.phase = 'seek'

	def reset(self):
		self.phase = 'wait'

	def get_max_flux(self, im, xy, stddev):
		ret = []
		hfr = None
		(h, w) = im.shape
		for p in xy:
			if p[2] < stddev * 3:
				#print "under 3stddev:", p[2], stddev * 3
				continue
			x = int(p[1])
			y = int(p[0])
			if (x < Focuser.hfr_size * 2):
				continue
			if (y < Focuser.hfr_size * 2):
				continue
			if (x > w - Focuser.hfr_size * 2 - 1):
				continue
			if (y > h - Focuser.hfr_size * 2 - 1):
				continue
			if hfr is None:
				hfr = Focuser.hfr(im[y - Focuser.hfr_size : y + Focuser.hfr_size + 1, x - Focuser.hfr_size : x + Focuser.hfr_size + 1])
				if hfr > Focuser.hfr_size * 0.5:
					hfr = None
					continue
				ret.append(p)
			else:
				if Focuser.hfr(im[y - Focuser.hfr_size : y + Focuser.hfr_size + 1, x - Focuser.hfr_size : x + Focuser.hfr_size + 1]) < hfr + 1:
					ret.append(p)
		print "hfr", hfr, ret
				
		if len(ret) > 0:
			return ret[0][2], hfr, np.array(ret)
		else:
			return 0, None, None

	def get_hfr(self, im):
		hfr = 0
		(h, w) = im.shape
		if self.focus_yx is None or len(self.focus_yx) == 0:
			return Focuser.hfr_size
		
		centroid_size = 20
		filtered = []
		original = []
		
		hfr_list = []
		
		sum_w = 0.0
		for p in self.focus_yx:
			(y, x, v) = p
			if (x < Focuser.hfr_size):
				continue
			if (y < Focuser.hfr_size):
				continue
			if (x > w - Focuser.hfr_size - 1):
				continue
			if (y > h - Focuser.hfr_size - 1):
				continue
			xs, ys = centroid(im[y  - centroid_size : y + centroid_size + 1, x - centroid_size : x + centroid_size + 1], centroid_size)
			x += xs
			y += ys
			ix = int(x)
			iy = int(y)
			if (ix < Focuser.hfr_size):
				continue
			if (iy < Focuser.hfr_size):
				continue
			if (ix > w - Focuser.hfr_size - 1):
				continue
			if (iy > h - Focuser.hfr_size - 1):
				continue

			filtered.append( (y, x, v) )
			original.append( p )
			hfr_list.append( Focuser.hfr(im[iy - Focuser.hfr_size : iy + Focuser.hfr_size + 1, ix - Focuser.hfr_size : ix + Focuser.hfr_size + 1]) )

		if len(filtered) == 0:
			return Focuser.hfr_size

		filtered = np.array(filtered)
		original = np.array(original)
		M, weights = pt_transform_opt(original, filtered, pt_func = pt_translation_scale)
		filtered[:, 0:2] = np.insert(original[:, 0:2], 2, 1.0, axis=1).dot(M).A

		self.focus_yx = filtered
		print "hfr_list", hfr_list, weights
		
		hfr = np.average(hfr_list, weights = weights)
		d2 = (np.array(hfr_list) - hfr) ** 2
		var = np.average(d2, weights = weights)
		noise = 2
		weights[np.where(d2 > var * noise**2)] = 1.0
		hfr = np.average(hfr_list, weights = weights)
		print "hfr_list_filt", hfr_list, weights
		return hfr


	def set_xy_from_stack(self, stack):
		im = stack.get()
		mean, self.stddev = cv2.meanStdDev(im)
		self.max_flux, self.min_hfr, self.focus_yx = self.get_max_flux(im, stack.get_xy(), 0)

	def proc_frame(self, im, i):
		t = time.time()

		try:
			fps = 1.0 / (t - self.prev_t)
		except:
			fps = 0
		
		if im.ndim > 2:
			im = cv2.min(cv2.min(im[:, :, 0], im[:, :, 1]), im[:, :, 2])

		self.im = im
		
		im = cv2.medianBlur(im, 3)

		if (self.dark.len() > 0):
			im_sub = cv2.subtract(im, self.dark.get())
		else:
			im_sub = im

		bg = cv2.blur(im_sub, (200, 200))
		bg = cv2.blur(bg, (200, 200))
		im_sub = cv2.subtract(im_sub, bg)

		self.stack.add_simple(im_sub)
		self.stack_im = self.stack.get()

		if self.phase_wait > 0:
			self.phase_wait -= 1
		elif self.phase == 'seek': # move near, out of focus
			self.hfr = self.get_hfr(im_sub)
			print "in-focus hfr ", self.hfr
			if self.hfr < Focuser.hfr_size / 3:
				self.phase = 'prep_record_v'
				self.phase_wait = 3
				cmdQueue.put('f+3')
			else:
				self.focus_yx = None
				for i in range (0, 12):
					cmdQueue.put('f-3')
				self.phase = 'dark'
				self.phase_wait = 5
				self.max_flux = 0
				self.min_hfr = Focuser.hfr_size
				self.dark_add = self.dark.n
		elif self.phase == 'dark': # use current image as darkframes
			if self.dark_add > 0:
				self.dark_add -= 1
				self.dark.add(self.im)
			else:
				mean, self.stddev = cv2.meanStdDev(self.stack_im)
				print "mean, stddev: ", mean, self.stddev
				for i in range (0, 9):
					cmdQueue.put('f+3')
				self.phase_wait = 5
				self.search_steps = 0
				self.phase = 'search'
		elif self.phase == 'search': # step far, record max flux
			flux, hfr, yx = self.get_max_flux(self.stack_im, self.stack.get_xy(), self.stddev)
			if flux < self.max_flux * 0.7 or hfr > self.min_hfr * 2 or self.search_steps > 120:
				self.phase = 'prep_record_v'
				cmdQueue.put('f-1')
			else:
				if flux > self.max_flux:
					self.focus_yx = yx
					self.max_flux = flux
					self.min_hfr = hfr
				else:
					cmdQueue.put('f+2')
				self.search_steps += 1
				self.hfr = self.get_hfr(im_sub)
			#self.phase_wait = 2
			print "max", flux, self.max_flux
		elif self.phase == 'prep_record_v': # record v curve
			self.hfr = self.get_hfr(im_sub)
			if self.hfr < Focuser.hfr_size / 2:
				self.v_curve = []
				self.phase = 'record_v'
			cmdQueue.put('f-1')
		elif self.phase == 'record_v': # record v curve
			self.hfr = self.get_hfr(im_sub)
			self.v_curve.append(self.hfr)
			if len(self.v_curve) > 20 and self.hfr > self.v_curve[0]:
				self.phase = 'focus_v'
				print "v_curve", self.v_curve[::-1]
				
				self.v_curve = np.array(self.v_curve)[::-1] # reverse

				self.xmin, self.side_len, self.smooth_size, self.c1, self.m1, c2, m2, v_curve_s = Focuser.v_param(self.v_curve)

				self.v_curve2 = []
				
				cmdQueue.put('f+1')
			else:
				cmdQueue.put('f-1')
		elif self.phase == 'focus_v': # go back, record first part of second v curve
			self.hfr = self.get_hfr(im_sub)
			if len(self.v_curve2) < self.side_len:
				self.v_curve2.append(self.hfr)
				cmdQueue.put('f+1')
				self.phase_wait = 1
			else:
				hyst, v_curve2_s = Focuser.v_shift(np.array(self.v_curve2), self.smooth_size, self.c1, self.m1)
				self.remaining_steps = round(self.xmin - self.side_len - hyst)
				print "remaining", self.remaining_steps
				self.phase = 'focus_v2'
				self.v_curve2.append(self.hfr)
		elif self.phase == 'focus_v2': # estimate maximum, go there
			self.hfr = self.get_hfr(im_sub)
			self.v_curve2.append(self.hfr)
			if self.remaining_steps > 0:
				self.remaining_steps -= 1
				cmdQueue.put('f+1')
			else:
				t = time.time()
				np.save("v_curve1_%d.npy" % t, np.array(self.v_curve))
				np.save("v_curve2_%d.npy" % t, np.array(self.v_curve2))
				self.phase = 'wait'
				
			print "hfr", self.hfr

		else:
			if self.focus_yx is not None:
				self.hfr = self.get_hfr(im_sub)
			
			
			

		status = "#%d Focuser: %s %s fps:%.1f hfr:%.2f" % (i, self.phase, self.dispmode, fps, self.hfr)
	

		if (self.dispmode == 'disp-orig'):
			disp = normalize(im)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			if self.focus_yx is not None:
				for p in self.focus_yx:
					cv2.circle(disp, (int(p[1]), int(p[0])), 20, (255), 1)
			ui.imshow(self.tid, disp)
		elif (self.dispmode == 'disp-df-cor'):
			disp = normalize(im_sub)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			if self.focus_yx is not None:
				for p in self.focus_yx:
					cv2.circle(disp, (int(p[1]), int(p[0])), 20, (255), 1)
			ui.imshow(self.tid, disp)
		elif (self.dispmode == 'disp-normal'):
			disp = normalize(self.stack_im)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			if self.focus_yx is not None:
				for p in self.focus_yx:
					cv2.circle(disp, (int(p[1]), int(p[0])), 20, (255), 1)
			ui.imshow(self.tid, disp)
		else:
			disp = cv2.cvtColor(normalize(self.stack_im), cv2.COLOR_GRAY2RGB)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
			ui.imshow(self.tid, disp)
		self.prev_t = t

class Runner(threading.Thread):
	def __init__(self, tid, camera, navigator = None, guider = None, zoom_focuser = None, focuser = None):
                threading.Thread.__init__(self)
                self.tid = tid
		self.camera = camera
		self.navigator = navigator
		self.guider = guider
		self.zoom_focuser = zoom_focuser
		self.focuser = focuser
		
	def run(self):
		profiler = LineProfiler()
		profiler.add_function(Navigator.proc_frame)
		profiler.add_function(Stack.add)
		profiler.add_function(Median.add)
		profiler.add_function(Median.add_masked)
		profiler.add_function(find_max)
		profiler.add_function(match_triangle)
		profiler.add_function(Runner.run)
		profiler.add_function(Camera_test.capture)
		profiler.add_function(Polar.solve)
		#profiler.add_function(Polar.camera_position)
		
		profiler.enable_by_count()
		
		
		cmdQueue.register(self.tid)
		
		i = 0
		if self.navigator is not None:
			mode = 'navigator'
		else:
			mode = 'guider'

		while True:
			while True:
				cmd=cmdQueue.get(self.tid, 0.0001)
				if cmd is None:
					break
				if cmd == 'exit' or cmd == 'shutdown':
					profiler.print_stats()

					return
				elif cmd == 'navigator' and self.navigator is not None:
					if mode == 'zoom_focuser':
						self.camera.cmd('z0')
					mode = 'navigator'
				elif cmd == 'guider' and self.guider is not None:
					if mode == 'zoom_focuser':
						self.camera.cmd('z0')
					self.guider.reset()
					self.guider.pt0 = self.navigator.stack.get_xy()
					mode = 'guider'
				elif cmd == 'z1':
					if self.zoom_focuser is not None:
						self.zoom_focuser.reset()
						maxx = 300
						maxy = 300
						if mode == 'navigator':
							(maxy, maxx, maxv) = self.navigator.stack.get_xy()[0]
						elif mode == 'focuser':
							(maxy, maxx, maxv) = self.focuser.stack.get_xy()[0]
						self.camera.cmd(cmd, x=maxx, y=maxy)
						mode = 'zoom_focuser'
				elif cmd == 'z0':
					if mode == 'zoom_focuser':
						self.camera.cmd(cmd)
						mode = 'navigator'
					elif mode == 'focuser':
						mode = 'navigator'
				elif cmd == 'focus' and mode != 'zoom_focuser' and self.focuser is not None:
					if mode == 'navigator':
						self.focuser.set_xy_from_stack(self.navigator.stack)
					mode = 'focuser'
				elif cmd == 'capture' or cmd == 'test-capture':
					if mode == 'zoom_focuser':
						self.camera.cmd('z0')
						mode = 'navigator'

					cmdQueue.put('capture-started')
					try:
						self.camera.capture_bulb(test=(cmd == 'test-capture'), callback = self.capture_cb)
					except AttributeError:
						pass
					except:
						print "Unexpected error: " + sys.exc_info().__str__()

					break
				else:
					self.camera.cmd(cmd)
					
				if mode == 'navigator':
					self.navigator.cmd(cmd)
				if mode == 'guider':
					self.guider.cmd(cmd)
				if mode == 'focuser':
					self.focuser.cmd(cmd)
				if mode == 'zoom_focuser':
					self.zoom_focuser.cmd(cmd)
	
			im, t = self.camera.capture()
			
			#cv2.imwrite("testimg20_" + str(i) + ".tif", im)
			if mode == 'navigator':
				self.navigator.proc_frame(im, i, t)
			if mode == 'guider':
				self.guider.proc_frame(im, i)
			if mode == 'focuser':
				self.focuser.proc_frame(im, i)
			if mode == 'zoom_focuser':
				self.zoom_focuser.proc_frame(im, i)
			i += 1
			#if i == 300:
			#	cmdQueue.put('exit')
		cmdQueue.put('exit')
		self.camera.shutdown()
	
	def capture_cb(self, jpg):
		cmdQueue.put('capture-finished')
		ui.imshow_jpg("full_res", io.BytesIO(jpg))
		threading.Thread(target=self.navigator.proc_full_res, args = [io.BytesIO(jpg)] ).start()

def main_loop():
	global status

	cmdQueue.register('main')
	while True:
		cmd=cmdQueue.get('main')
		if cmd == 'exit' or cmd == 'shutdown':
			stacktraces()

			if cmd == 'shutdown':
				subprocess.call(['shutdown', '-h', "now"])
			break
		if cmd == 'save':
			status.save()
		
		if cmd == 'interrupt':
			camera = status.path(["navigator", "camera"])
			camera['interrupt'] = True

	status.save()
	


class Camera_test:
	def __init__(self, status):
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
		#time.sleep(3)
		print self.i
		#pil_image = Image.open("converted/IMG_%04d.jpg" % (146+self.i))
		#pil_image.thumbnail((1000,1000), Image.ANTIALIAS)
		#im = np.array(pil_image)
		#im = cv2.imread("testimg16_" + str(self.i % 100 * 3 + int(self.i / 100) * 10) + ".tif")
		im = cv2.imread("testimg19_" + str(self.i) + ".tif")
		#im = apply_gamma(im, 2.2)
		if self.x != 0 or self.y != 0:
			M = np.array([[1.0, 0.0, self.x],
		        	      [0.0, 1.0, self.y]])
			bg = cv2.blur(im, (30, 30))
			im = cv2.warpAffine(im, M[0:2,0:3], (im.shape[1], im.shape[0]), bg, borderMode=cv2.BORDER_TRANSPARENT);

		#t = os.path.getmtime("testimg16_" + str(self.i) + ".tif")
		self.i += self.step
		return im, None

	def shutdown(self):
		pass

class Camera_test_shift:
	def __init__(self, status, cam0, shift):
		self.cam0 = cam0
		self.shift = shift
	
	def cmd(self, cmd):
		pass
	
	def capture(self):
		i =  self.cam0.i + self.shift
		im = cv2.imread("testimg19_" + str(i) + ".tif")
		return im, None

	def shutdown(self):
		pass


class Camera_test_g:
	def __init__(self, status, go):
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

	def shutdown(self):
		pass

def run_v4l2():
	global status
	status = Status("run_v4l2.conf")
	ui.namedWindow('navigator')
	ui.namedWindow('polar')

        polar = Polar(status.path(["polar"]), ['navigator'])
	cam = Camera(status.path(["navigator", "camera"]))
	cam.prepare(1280, 960)
	dark = Median(5)
	nav = Navigator(status.path(["navigator"]), dark, polar, 'navigator', polar_tid = 'polar')

	runner = Runner('navigator', cam, navigator = nav)
	runner.start()
	
	main_loop()
	runner.join()

def run_gphoto():
	global status
	status = Status("run_gphoto.conf")
	cam = Camera_gphoto(status.path(["navigator", "camera"]))
	cam.prepare()
	ui.namedWindow('navigator')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

        polar = Polar(status.path(["polar"]), ['navigator', 'full-res'])

	dark = Median(5)
	nav = Navigator(status.path(["navigator"]), dark, polar, 'navigator', polar_tid = 'polar')
	focuser = Focuser('navigator', dark = dark)
	zoom_focuser = Focuser('navigator')

	runner = Runner('navigator', cam, navigator = nav, focuser = focuser, zoom_focuser = zoom_focuser)
	runner.start()

	main_loop()
	runner.join()


def run_v4l2_g():
	global status
	status = Status("run_v4l2_g.conf")
	ui.namedWindow('navigator')
	ui.namedWindow('polar')

        polar = Polar(status.path(["polar"]), ['navigator'])

	cam = Camera(status.path(["guider", "navigator", "camera"]))
	cam.prepare(1280, 960)

	dark = Median(5)
	nav = Navigator(status.path(["guider", "navigator"]), dark, polar, 'navigator', polar_tid = 'polar')
	go = GuideOut()
	guider = Guider(status.path(["guider"]), go, dark, 'navigator')

	runner = Runner('navigator', cam, navigator = nav, guider = guider)
	runner.start()

	main_loop()
	runner.join()

def run_gphoto_g():
	global status
	status = Status("run_gphoto_g.conf")
	ui.namedWindow('navigator')
	ui.namedWindow('polar')

        polar = Polar(status.path(["polar"]), ['navigator', 'full-res'])

	cam = Camera_gphoto(status.path(["guider", "navigator", "camera"]))
	cam.prepare()

	dark = Median(5)
	nav = Navigator(status.path(["guider", "navigator"]), dark, polar, 'navigator', polar_tid = 'polar')
	go = GuideOut()
	guider = Guider(status.path(["guider"]), go, dark, 'navigator')

	runner = Runner('navigator', cam, navigator = nav, guider = guider)
	runner.start()
	main_loop()
	runner.join()

def run_test_g():
	global status
	status = Status("run_test_g.conf")
	ui.namedWindow('navigator')
	ui.namedWindow('polar')

        polar = Polar(status.path(["polar"]), ['navigator'])

	dark = Median(5)
	nav = Navigator(status.path(["guider", "navigator"]), dark, polar, 'navigator', polar_tid = 'polar')
	go = GuideOut()
	guider = Guider(status.path(["guider"]), go, dark, 'navigator')
	cam = Camera_test_g(status.path(["guider", "navigator", "camera"]), go)

	runner = Runner('navigator', cam, navigator = nav, guider = guider)
	runner.start()
	main_loop()
	runner.join()

def run_test():
	global status
	status = Status("run_test.conf")
	ui.namedWindow('navigator')
	ui.namedWindow('polar')

        polar = Polar(status.path(["polar"]), ['navigator'])

	cam = Camera_test(status.path(["navigator", "camera"]))
	dark = Median(5)
	nav = Navigator(status.path(["navigator"]), dark, polar, 'navigator', polar_tid = 'polar')

	runner = Runner('navigator', cam, navigator = nav)
	runner.start()
	main_loop()
	runner.join()

def run_test_2():
	global status
	status = Status("run_test_2.conf")
	ui.namedWindow('navigator')
	ui.namedWindow('guider')
	ui.namedWindow('polar')

        polar = Polar(status.path(["polar"]), ['navigator', 'guider'])

	dark1 = Median(5)
	dark2 = Median(5)

	cam1 = Camera_test(status.path(["navigator", "camera"]))
	nav1 = Navigator(status.path(["navigator"]), dark1, polar, 'navigator', polar_tid = 'polar')

	nav = Navigator(status.path(["guider", "navigator"]), dark2, polar, 'guider')
	go = GuideOut()
	guider = Guider(status.path(["guider"]), go, dark2, 'guider')
	#cam = Camera_test_g(status.path(["guider", "navigator", "camera"]), go)
	cam = Camera_test_shift(status.path(["guider", "navigator", "camera"]), cam1, 3000)
	
	runner = Runner('navigator', cam1, navigator = nav1)
	runner.start()
	
	runner2 = Runner('guider', cam, navigator = nav, guider = guider)
	runner2.start()
	
	main_loop()
	
	runner.join()
	runner2.join()

def run_test_2_gphoto():
	global status
	status = Status("run_test_2_gphoto.conf")
	
	cam1 = Camera_gphoto(status.path(["navigator", "camera"]))
	cam1.prepare()

        polar = Polar(status.path(["polar"]), ['navigator', 'guider', 'full-res'])

	dark1 = Median(5)
	dark2 = Median(5)
	
	nav1 = Navigator(status.path(["navigator"]), dark1, polar, 'navigator', polar_tid = 'polar')
	focuser = Focuser('navigator', dark = dark1)
	zoom_focuser = Focuser('navigator')

	nav = Navigator(status.path(["guider", "navigator"]), dark2, polar, 'guider')
	go = GuideOut()
	guider = Guider(status.path(["guider"]), go, dark2, 'guider')
	cam = Camera_test_g(status.path(["guider", "navigator", "camera"]), go)

	ui.namedWindow('navigator')
	ui.namedWindow('guider')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

	go.out(1, 10) # move aside for 10s to collect darkframes

	runner = Runner('navigator', cam1, navigator = nav1, focuser=focuser, zoom_focuser = zoom_focuser)
	runner.start()
	
	runner2 = Runner('guider', cam, navigator = nav, guider = guider)
	runner2.start()
	
	main_loop()
	
	runner.join()
	runner2.join()


def run_2():
	global status
	status = Status("run_2.conf")
	cam = Camera(status.path(["guider", "navigator", "camera"]))
	cam.prepare(1280, 960)
	
	cam1 = Camera_gphoto(status.path(["navigator", "camera"]))
	cam1.prepare()

        polar = Polar(status.path(["polar"]), ['navigator', 'guider', 'full-res'])

	dark1 = Median(5)
	dark2 = Median(5)
	
	nav1 = Navigator(status.path(["navigator"]), dark1, polar, 'navigator', polar_tid = 'polar')
	focuser = Focuser('navigator', dark = dark1)
	zoom_focuser = Focuser('navigator')

	nav = Navigator(status.path(["guider", "navigator"]), dark2, polar, 'guider')
	go = GuideOut()
	guider = Guider(status.path(["guider"]), go, dark2, 'guider')

	ui.namedWindow('navigator')
	ui.namedWindow('guider')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

	
	go.out(1, 10) # move aside for 10s to collect darkframes

	runner = Runner('navigator', cam1, navigator = nav1, focuser=focuser, zoom_focuser = zoom_focuser)
	runner.start()
	
	runner2 = Runner('guider', cam, navigator = nav, guider = guider)
	runner2.start()
	
	main_loop()
	
	runner.join()
	runner2.join()

if __name__ == "__main__":
	os.environ["LC_NUMERIC"] = "C"
	
	mystderr = os.fdopen(os.dup(sys.stderr.fileno()), 'w', 0)
	devnull = open(os.devnull,"w")
	os.dup2(devnull.fileno(), sys.stdout.fileno())
	os.dup2(devnull.fileno(), sys.stderr.fileno())
	
	sys.stdout = mystderr
	sys.stderr = mystderr
	

	#run_gphoto()
	run_test_2()
	#run_v4l2()
	#run_test_2_gphoto()
	#run_v4l2_g()
	#run_2()
	#run_test_g()
	#run_2()
	#run_test()








