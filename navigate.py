#!/usr/bin/env python3

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

from am import Solver, Plotter, scale_wcs
from polar import Polar

import sys
import io
import os.path
import time
import threading
import os
import psutil
import signal

from v4l2_camera import *
from camera_gphoto import *
from camera_indi import *

from guide_out import GuideOut

import queue
import random
#from line_profiler import LineProfiler

from gui import ui
from cmd import cmdQueue

from stacktraces import stacktraces
import json

from focuser_out import FocuserOut
from temp_sensor import TempSensor
from ext_trigger import ExtTrigger
from centroid import centroid, sym_center, hfr, fit_ellipse, getRectSubPix
from polyfit import *
from quat import Quaternion
from star_detector import *
from bahtinov import Bahtinov
from smooth import smooth

import indi_python.indi_base as indi
from indi_python.indi_loop import IndiLoop


import logging
from functools import reduce
logging.basicConfig(format="%(filename)s:%(lineno)d: %(message)s", level=logging.INFO)

log = logging.getLogger()

class Status:
	def __init__(self, conf_file):
		try:
			with open(conf_file) as data_file:    
				self.status = json.load(data_file)
			self.status["conf_file"] = conf_file
		except:
			self.status = {"conf_file" : conf_file}
			log.exception("conf load")
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


class IndiDriver(IndiLoop):
	def __init__(self):
		super().__init__(driver=True, client_addr='localhost')
		self.queues = {'snoop': {}, 'new': {}}
		self.setQueue = queue.Queue()
		self.timeout = 0.2
		self.reply_timeout = 60
		
		self.sendClient(indi.getProperties())
	
	
	def loop1(self):
		super().loop1()
		
		try:
			while True:
				prop = self.setQueue.get(block=False)
				self.sendDriver(prop.setMessage())
		except queue.Empty:
			pass
		
	def register(self, device, msg_type = 'new'):
		self.queues[msg_type][device] = queue.Queue()
		log.info("register %s", device)
	
	def get(self, device, block=True, timeout=None, msg_type = 'new'):
		q = self.queues[msg_type][device]
		try:
			pd = q.get(block=block, timeout=timeout)
			log.info("%s:%s", device, pd[1].getAttr('name'))
			return pd
		except queue.Empty:
			return (None, None)
			
	def handleNewValue(self, msg, prop, from_client_socket=False):
		if from_client_socket:
			return

		device = prop.getAttr('device')

		if device in self.queues['new']:
			self.queues['new'][device].put((msg, prop))
			prop.setAttr('state', 'Busy')
			self.enqueueSetMessage(prop)
		else:
			prop.newFromEtree(msg)
			prop.setAttr('state', 'Ok')
			self.enqueueSetMessage(prop)

	def handleSnoop(self, msg, prop):
		device = prop.getAttr('device')

		if prop.itype == 'BLOB':
			msg = None
		if device in self.queues['snoop']:
			self.queues['snoop'][device].put((msg, prop))
	
	def handleEOF(self):
		cmdQueue.put('exit')

	def enqueueSetMessage(self, prop):
		self.setQueue.put(prop)


def normalize(img):
	dst = np.empty_like(img)
	return cv2.normalize(img, dst, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	

def cv2_dtype(dtype):
	if np.issubdtype(np.uint8, dtype):
		return cv2.CV_8UC1
	else:
		return cv2.CV_16UC1
	

def subtract_bg(im, scale):
	bg = cv2.resize(im, ((im.shape[1] + scale - 1) // scale, (im.shape[0] + scale - 1) // scale), interpolation=cv2.INTER_AREA)
	bg = cv2.erode(bg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
	bg = cv2.blur(bg, (20,20))
	bg = cv2.blur(bg, (20,20))
	bg = cv2.blur(bg, (20,20))
	bg = cv2.resize(bg, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)
	bg = cv2.subtract(bg, 1)
	im = cv2.subtract(im, bg)
	return im

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
		self.res = np.array(a[a.shape[0] // 2, :, :])
		#a = np.median(self.list, axis = 0)
		#self.res = np.empty_like(self.list[0])
		#self.res[:,:] = a

	def _add_masked(self, im, pts):
		if self.res is None:
			return

		if im.dtype == np.uint8:
			cv_dtype = cv2.CV_8UC1
		elif im.dtype == np.int8:
			cv_dtype = cv2.CV_8SC1
		elif im.dtype == np.int16:
			cv_dtype = cv2.CV_16SC1
		else: # im.dtype == np.uint16:
			cv_dtype = cv2.CV_16UC1

		mask = np.zeros_like(im, dtype=np.uint16)
	
		white = np.iinfo(mask.dtype).max
		for p in pts:
			cv2.circle(mask, p, 20, (white), -1)

		mask = cv2.blur(mask, (30, 30))
		mask = cv2.blur(mask, (30, 30))
		inv_mask = cv2.bitwise_not(mask)
		
		res = cv2.add(cv2.multiply(im, inv_mask, scale = 1.0 / white, dtype=cv_dtype), cv2.multiply(self.res, mask, scale = 1.0 / white, dtype=cv_dtype))
		
		self._add(res)
		#ui.imshow("dark", normalize(inv_mask))

	def _add_mean(self, im, pts=None):
		if im.dtype == np.uint8:
			cv_dtype = cv2.CV_8SC1
		else:
			cv_dtype = cv2.CV_16SC1
	
		mean, stddev = cv2.meanStdDev(im)
		im = cv2.subtract(im, mean, dtype=cv_dtype)
		
		im = np.clip(im, -3 * stddev, 3 * stddev)
		
		mean, stddev = cv2.meanStdDev(im)
		im = cv2.subtract(im, mean, dtype=cv_dtype)
		if pts is None:
			self._add(im)
		else:
			self._add_masked(im, pts)
		

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

	def add_mean(self, *args, **kwargs):
		if self.bg_thread is not None:
			self.bg_thread.join()
		
		self.bg_thread = threading.Thread(target=self._add_mean, args = args, kwargs = kwargs)
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


def get_hfr_field(im, pts, hfr_size = 20, sub_bg = False):
	cur_hfr = hfr_size
	(h, w) = im.shape
		
	hfr_list = []
		
	sum_w = 0.0
	for p in pts:
			(y, x) = p[:2]
			ix = int(x + 0.5)
			iy = int(y + 0.5)
			if (ix < hfr_size):
				continue
			if (iy < hfr_size):
				continue
			if (ix > w - hfr_size - 1):
				continue
			if (iy > h - hfr_size - 1):
				continue

			hf = hfr(im[iy - hfr_size : iy + hfr_size + 1, ix - hfr_size : ix + hfr_size + 1], sub_bg)
			if hf < 0.9:
				continue
			
			if hf > hfr_size * 0.7:
				continue

			hfr_list.append((y, x, hf) )

	if len(hfr_list) == 0:
		hfr_list.append((h / 2, w / 2, hfr_size) )
	return hfr_list

def filter_hfr_list(hfr_list):
	hfr_list = np.array(hfr_list)
	if len(hfr_list) < 3:
		return hfr_list

	for deg, kappa in enumerate([3, 2, 2]):
		A = np.polynomial.polynomial.polyvander2d(hfr_list[:, 0], hfr_list[:, 1], (deg, deg))[:, np.where(np.flipud(np.tri(deg + 1)).ravel())[0]]
		#log.info A
	
		for i in range(4):
			coef = np.linalg.lstsq(A, hfr_list[:, 2])[0]
	
			cur_hfr = np.dot(A, coef)
			cur_hfr[(cur_hfr < 1)] = 1
			d2 = (hfr_list[:,2] - cur_hfr) ** 2 / cur_hfr**2
			var = np.average(d2)
			keep = np.where(d2 <= var * kappa**2 + 0.001)
			hfr_list = hfr_list[keep]
			A = A[keep]
	return hfr_list

	
def get_hfr_list(im, pts, hfr_size = 20, sub_bg = False):
	hfr_list = get_hfr_field(im, pts, hfr_size, sub_bg)

	if len(hfr_list) == 0:
		return hfr_size

	hfr_list = filter_hfr_list(hfr_list)
	cur_hfr = np.average(hfr_list[:,2])
	return cur_hfr


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
		#log.info "match1",match
		if len(match) == 0:
			pt1 = self.get_xy()
			pt1m, pt2m, match = match_triangle(pt1, pt2, 5, 15)
			#log.info "match2",match
		
		
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
			#log.info "off2", off 
			#log.info match
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
				cv2.circle(self.match, (int(p[1] + 0.5), int(p[0] + 0.5)), 13, (255), 1)
		
			for p in pt2:
				cv2.circle(self.match, (int(p[1] + 0.5), int(p[0] + 0.5)), 5, (255), 1)
			for p in pt2m:
				cv2.circle(self.match, (int(p[1] + 0.5), int(p[0] + 0.5)), 10, (255), 1)
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
		if self.img is None:
			return None
		if dtype == np.uint8:
			return cv2.divide(self.img, 255, dtype=cv2.CV_8UC1)
		else:
			return self.img

	def get_xy(self):
		if self.xy is None:
			self.xy = np.array(find_max(self.img, 12, n = 15))

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

def apply_gamma8(img, gamma):
	lut = np.fromiter( ( (x / 255.0)**gamma * 255.0 for x in range(256)), dtype=np.uint8 )
	return np.take(lut, img)

def apply_gamma16_to8(img, gamma):
	lut = np.fromiter( ( (x / 65535.0)**gamma * 255.0 for x in range(65535)), dtype=np.uint8 )
	return np.take(lut, img)

def format_coords(ra, dec):
	ret = {}
	ra = ra % 360.0

	ret['ra_deg'] = '{:03.2f}'.format(ra)
	ret['dec_deg'] = '{:03.2f}'.format(dec)
	sign = 1
	if dec < 0:
		sign = -1
		dec = -dec
		
	i_dec_deg = int(dec)
	i_dec_min = int((dec - i_dec_deg) * 60 + 1e-61)
	i_dec_sec = int((((dec - i_dec_deg) * 60) - i_dec_min) * 60 + 1e-6)
	ret['dec_dms'] = '{:03d}:{:02d}:{:02d}'.format(i_dec_deg * sign, i_dec_min, i_dec_sec)
	i_ra_deg = int(ra)
	i_ra_min = int((ra - i_ra_deg) * 60 + 1e-6)
	i_ra_sec = int((((ra - i_ra_deg) * 60) - i_ra_min) * 60 + 1e-6)
	ret['ra_dms'] = '{:03d}:{:02d}:{:02d}'.format(i_ra_deg, i_ra_min, i_ra_sec)
	hra = ra / 15
	ret['ra_h'] = '{:03.2f}'.format(hra)
	i_hra_deg = int(hra)
	i_hra_min = int((hra - i_hra_deg) * 60 + 1e-6)
	i_hra_sec = int((((hra - i_hra_deg) * 60) - i_hra_min) * 60 + 1e-6)
	ret['ra_hms'] = '{:3d}:{:02d}:{:02d}'.format(i_hra_deg, i_hra_min, i_hra_sec)
	return ret


class Navigator:
	def __init__(self, driver, device, status, dark, mount, tid, polar_tid = None, full_res = None):
		self.status = status
		
		self.driver = driver
		self.device = device
		driver.defineProperties("""
		<INDIDriver>

			<defNumberVector device="{0}" name="solver_time" label="solver_time" group="Solver Control" state="Idle" perm="ro">
				<defNumber name="i_solver" label="i_solver" format="%5.0f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="i_solved" label="i_solved" format="%5.0f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="t_solver" label="t_solver" format="%4.1f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="t_solved" label="t_solved" format="%4.1f" min="0" max="0" step="0">0</defNumber>
			</defNumberVector>

			<defTextVector device="{0}" name="field_corr" label="Field correction file" group="Solver Control" state="Idle" perm="rw">
				<defText name="file" label="file"></defText>
			</defTextVector>

			<defNumberVector device="{0}" name="filed_corr_limit" label="Field correction limit" group="Solver Control" state="Idle" perm="rw">
				<defNumber name="limit" label="limit" format="%5.0f" min="0" max="0" step="0">10</defNumber>
			</defNumberVector>

			<defNumberVector device="{0}" name="coord" label="Coord" group="Solver Control" state="Idle" perm="rw">
				<defNumber name="RA" label="RA" format="%10.6m" min="0" max="0" step="0">0</defNumber>
				<defNumber name="DEC" label="Dec" format="%10.6m" min="0" max="0" step="0">0</defNumber>
			</defNumberVector>

			<defNumberVector device="{0}" name="field" label="Field" group="Solver Control" state="Idle" perm="ro">
				<defNumber name="current" label="Current" format="%4.2f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="radius" label="Radius" format="%4.2f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="max_radius" label="Max Radius" format="%4.2f" min="0" max="0" step="0">100</defNumber>
			</defNumberVector>

			<defSwitchVector device="{0}" name="commands" label="Commands" group="Solver Control" state="Idle" perm="rw" rule="AtMostOne">
				<defSwitch name="reset" label="reset">Off</defSwitch>
				<defSwitch name="retry" label="retry">Off</defSwitch>
				<defSwitch name="darkframe" label="dark">Off</defSwitch>
				<defSwitch name="hotpixels" label="hotpixels">Off</defSwitch>
			</defSwitchVector>

		</INDIDriver>
		""".format(device))

		self.props = driver[device]
		
		self.dark = dark
		self.stack = Stack()
		self.solvedlock = threading.Lock()
		self.solver = None
		self.full_res_solver = None
		self.full_res_lock = threading.Lock()
		self.solver_off = np.array([0.0, 0.0])
		self.status.setdefault("field_corr_limit", 10)
		self.status.setdefault("field_corr", None)
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
		self.mount = mount
		self.polar_tid = polar_tid
		self.index_sources = []
		self.status['i_solved'] = 0
		self.status['i_solver'] = 0
		self.status['t_solved'] = 0
		self.status['t_solver'] = 0
		self.prev_t = 0
		self.status['ra'], self.status['dec'] = self.mount.polar.zenith()
		self.status['max_radius'] = 100
		if tid == 'guider':
			self.status['ra'], self.status['dec'], self.status['max_radius'] = self.mount.get_oag_pos()
		self.status.update(format_coords(self.status['ra'], self.status['dec']))
		self.status['radius'] = self.status['max_radius']
		self.hotpixels = None
		self.hotpix_cnt = None
		
		self.field_corr = None
		self.field_corr_list = []
		if self.status['field_corr'] is not None:
			try:
				self.field_corr = np.load(self.status['field_corr'])
			except:
				self.field_corr = None
		
		self.i_dark = 0
		self.full_res = full_res
		if self.full_res is not None:
			self.full_res['full_hfr'] = []
			self.full_res['full_temp'] = []
			self.full_res['full_hum'] = []
			self.full_res['full_name'] = []
		self.status.setdefault('go_by', 0.1)
		self.status.setdefault('profiles', [{}])
		self.status.setdefault('profile', 0)
		self.im = None

	def set_profile(self, i):
		try:
			profile = self.status['profiles'][i]
		except:
			return
		if "field_corr_limit" in profile:
			self.status["field_corr_limit"] = profile["field_corr_limit"]
		if "field_corr" in profile:
			self.status["field_corr"] = profile["field_corr"]
		else:
			self.status["field_corr"] = None
		
		if self.status['field_corr'] is not None:
			try:
				self.field_corr = np.load(self.status['field_corr'])
			except:
				log.exception("load field_corr");
				self.field_corr = None

		if "field_deg" in profile:
			self.status["field_deg"] = profile["field_deg"]
		else:
			self.status["field_deg"] = None

		if "oag_pos" in profile:
			self.mount.status['oag_pos'] = profile["oag_pos"]
			self.mount.status['t_dif'] = 120

		if "main_pos" in profile:
			self.mount.status['main_pos'] = profile["main_pos"]
			self.mount.status['t_dif'] = 120

		self.status['profile'] = i
		

	def hotpix_find(self):
		bg = cv2.GaussianBlur(self.im, (7, 7), 0)
		im = cv2.subtract(self.im, bg)
		
		mean, stddev = cv2.meanStdDev(im)
		
		if self.hotpix_cnt is None:
			self.hotpix_cnt = np.zeros_like(im, dtype=np.uint8)
		
		self.hotpix_cnt[(im > stddev * 10)] += 1
		
	def hotpix_update(self):
		self.hotpixels = list(zip(*np.where(self.hotpix_cnt > 2)))
		
	
	def proc_frame(self,im, i, t = None):
		self.i = i
		dispmode = self.props['dispmode'].getActiveSwitch()
		if im.ndim > 2:
			im = cv2.min(cv2.min(im[:, :, 0], im[:, :, 1]), im[:, :, 2])

		self.im = im
		
		if t is None:
			t = time.time()
		if (self.dark.len() > 2):
			im_sub = cv2.subtract(im, self.dark.get(), dtype=cv2_dtype(im.dtype))
		else:
			im_sub = im
		
		im_sub = subtract_bg(im_sub, 10)

		n_hotpixels = 0
		if self.hotpixels is not None:
			n_hotpixels = len(self.hotpixels)
			for p in self.hotpixels:
				cv2.circle(im_sub, (int(p[1] + 0.5), int(p[0] + 0.5)), 1, (0), -1)

		if i < 6:
			self.dark.add_mean(im)
			self.hotpix_find()
		
		if i == 6:
			self.hotpix_update()
			
		if self.field_corr is not None:
			im_sub = cv2.remap(im_sub, self.field_corr, None, cv2.INTER_LINEAR)

		M = self.stack.add(im_sub, show_match=(dispmode == 'match'))
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
				save_conf = self.status['field_deg'] is None
				with self.solvedlock:
					self.status['ra'] = self.solver.ra
					self.status['dec'] = self.solver.dec
					self.status.update(format_coords(self.status['ra'], self.status['dec']))
					self.status['field_deg'] = self.solver.field_deg
					self.status['radius'] = self.status['field_deg']
					self.wcs = self.solver.wcs
			
					if i - self.i_dark > 12:
						self.dark.add_mean(self.solved_im, self.solver.ind_sources)
						self.i_dark = i
				
					self.index_sources = self.solver.ind_radec
					self.plotter = Plotter(self.wcs)
					self.plotter_off = self.solver_off
					self.status['i_solved'] = self.status['i_solver']
					self.status['t_solved'] = self.status['t_solver']
					self.mount.set_pos_tan(self.wcs, self.status['t_solver'], self.tid)
					if self.mount.polar.mode == 'solve':
						self.mount.polar.set_pos_tan(self.wcs, self.status['t_solver'], self.tid)
				
					self.props['solver_time']['i_solved'].setValue(self.status['i_solved'])
					self.props['solver_time']['t_solved'].setValue(self.status['t_solved'])
					self.props['coord'].setValue((self.status['ra'] / 15.0, self.status['dec']))
					self.props['field']['current'].setValue(self.status['field_deg'] or 0)
					self.props['field']['radius'].setValue(self.status['radius'])
					self.driver.enqueueSetMessage(self.props['solver_time'])
					self.driver.enqueueSetMessage(self.props['coord'])
					self.driver.enqueueSetMessage(self.props['field'])
					
				if self.polar_tid is not None:
					self.mount.polar.solve()
				if save_conf:
					cmdQueue.put('save')
					
				log.info("field corr len %d", len(self.field_corr_list))
				if len(self.field_corr_list) > self.status['field_corr_limit']:
					self.update_field_cor()
					self.status['field_corr_limit'] *= 2
					
			else:
				if self.status['radius'] > 0 and self.status['radius'] * 2 + 15 < self.status['max_radius']:
					self.status['radius'] = self.status['radius'] * 2 + 15
					self.props['field']['radius'].setValue(self.status['radius'])
					self.driver.enqueueSetMessage(self.props['field'])
				else:
					if self.tid == 'guider':
						self.status['ra'], self.status['dec'], self.status['max_radius'] = self.mount.get_oag_pos()
					else:
						self.status['ra'], self.status['dec'] = self.mount.polar.zenith()
					self.status['radius'] = self.status['max_radius']
					self.wcs = None
					#self.props['coord'].setValue((self.status['ra'] / 15.0, self.status['dec']))
					self.props['field']['current'].setValue(self.status['field_deg'] or 0)
					self.props['field']['radius'].setValue(self.status['radius'])
					self.props['field']['max_radius'].setValue(self.status['max_radius'])
					#self.driver.enqueueSetMessage(self.props['coord'])
					self.driver.enqueueSetMessage(self.props['field'])
			self.solver = None
			self.solved_im = None

		if self.solver is None and i > 20 and dispmode != 'orig' and self.status['dispmode'] != 'df-cor':
			xy = self.stack.get_xy()
			#log.info "len", len(xy)
			if len(xy) > 8:
				self.status['i_solver'] = i
				self.status['t_solver'] = t
				self.solved_im = im
				self.solver = Solver(sources_list = xy, field_w = im.shape[1], field_h = im.shape[0], ra = self.status['ra'], dec = self.status['dec'], field_deg = self.status['field_deg'], radius = self.status['radius'], field_corr = self.field_corr_list)
				#self.solver = Solver(sources_img = filtered, field_w = im.shape[1], field_h = im.shape[0], ra = self.status['ra'], dec = self.status['dec'], field_deg = self.status['field_deg'])
				self.solver.start()
				self.solver_off = np.array([0.0, 0.0])
				self.props['solver_time']['i_solver'].setValue(self.status['i_solver'])
				self.props['solver_time']['t_solver'].setValue(self.status['t_solver'])
				self.driver.enqueueSetMessage(self.props['solver_time'])
		if self.mount.polar.mode == 'solve' and self.polar_tid is not None:
			polar_plot = self.mount.polar.plot2()
			p_status = "#%d %s solv#%d r:%.1f fps:%.1f" % (i, self.mount.polar.mode, i - self.status['i_solver'], self.status['radius'], fps)
			cv2.putText(polar_plot, p_status, (10, polar_plot.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
			ui.imshow(self.polar_tid, polar_plot)
		elif self.mount.polar.mode == 'adjust' and self.wcs is not None:
			self.mount.polar.set_pos_tan(self.wcs, self.status['t_solver'], self.tid, off = self.plotter_off)
			if self.polar_tid is not None:
				polar_plot = self.mount.polar.plot2()
				p_status = "#%d %s solv#%d r:%.1f fps:%.1f" % (i, self.mount.polar.mode, i - self.status['i_solved'], self.status['radius'], fps)
				cv2.putText(polar_plot, p_status, (10, polar_plot.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
				ui.imshow(self.polar_tid, polar_plot)
			
		status = "#%d %s %s  solv#%d r:%.1f fps:%.1f hp:%d" % (i, dispmode, self.mount.polar.mode, i - self.status['i_solver'], self.status['radius'], fps, n_hotpixels)
		if (dispmode == 'orig'):
			disp = normalize(im)

			try:
				zp = self.status['camera']['zoom_pos']
				cv2.rectangle(disp, (zp[0], zp[1]), (zp[2], zp[3]), (200), 1)
			except:
				pass

			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			ui.imshow(self.tid, disp)
		elif (dispmode == 'df-cor'):
			disp = normalize(im_sub)
			
			try:
				zp = self.status['camera']['zoom_pos']
				cv2.rectangle(disp, (zp[0], zp[1]), (zp[2], zp[3]), (200), 1)
			except:
				pass

			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			ui.imshow(self.tid, disp)
		elif (dispmode == 'normal'):
			disp = normalize(filtered)
			for p in self.stack.get_xy():
				cv2.circle(disp, (int(p[1] + 0.5), int(p[0] + 0.5)), 13, (255), 1)
			
			try:
				zp = self.status['camera']['zoom_pos']
				cv2.rectangle(disp, (zp[0], zp[1]), (zp[2], zp[3]), (200), 1)
			except:
				pass
			
			if self.plotter is not None:
		
				extra_lines = []
				
				if self.mount.polar.mode == 'adjust':
					transf_index = self.mount.polar.transform_ra_dec_list(self.index_sources)
					extra_lines = [ (si[0], si[1], ti[0], ti[1]) for si, ti in zip(self.index_sources, transf_index) ]
					
				plot_bg(self.tid, status, self.plotter.plot, disp, self.plotter_off, extra_lines = extra_lines)
			else:
				cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
				ui.imshow(self.tid, disp)
		elif (dispmode.startswith('zoom-')):
			if self.plotter is not None:
				zoom = dispmode[len('zoom-'):]
				extra_lines = []
				if self.tid == 'navigator':
					extra_lines = self.mount.get_guider_plot()
				elif self.tid == 'guider':
                                        extra_lines = self.mount.get_main_plot()
				plot_bg(self.tid, status, self.plotter.plot, normalize(filtered), self.plotter_off, scale=zoom, extra_lines = extra_lines)
			else:
				disp = normalize(filtered)
				cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
				ui.imshow(self.tid, disp)
				
		elif (dispmode == 'match'):
			disp = self.stack.match
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			ui.imshow(self.tid, disp)
	


		self.prev_t = t
	
	def handleNewProp(self, msg, prop):
		name = prop.getAttr('name')
		if name == 'field_corr':
			self.status['field_corr'] = prop['file'].getValue()
			prop.setAttr('state', 'Ok')
			
		elif name == 'commands':
			if prop['reset'] == True:
				prop['reset'].setValue(False)
				if self.solver is not None:
					self.solver.terminate(wait=True)
				self.status['field_deg'] = None
				self.solver = None
				self.plotter = None
				if self.tid == 'guider':
					self.status['ra'], self.status['dec'], self.status['max_radius'] = self.mount.get_oag_pos()
				else:
					self.status['ra'], self.status['dec'] = self.mount.polar.zenith()
				self.status['radius'] = self.status['max_radius']
				self.field_corr = None
				self.field_corr_limit = 10
				self.props['coord'].setValue((self.status['ra'] / 15.0, self.status['dec']))
				self.props['field']['current'].setValue(self.status['field_deg'] or 0)
				self.props['field']['radius'].setValue(self.status['radius'])
				self.props['field']['max_radius'].setValue(self.status['max_radius'])
				self.driver.enqueueSetMessage(self.props['coord'])
				self.driver.enqueueSetMessage(self.props['field'])
				prop.setAttr('state', 'Ok')

			if prop['retry'] == True:
				prop['retry'].setValue(False)

				if self.solver is not None:
					self.solver.terminate(wait=False)
				if self.full_res_solver is not None:
                        	        self.full_res_solver.terminate(wait=False)
				if self.tid == 'guider':
					self.status['ra'], self.status['dec'], self.status['max_radius'] = self.mount.get_oag_pos()
				else:
					self.status['ra'], self.status['dec'] = self.mount.polar.zenith()
				self.status['radius'] = self.status['max_radius']
				prop.setAttr('state', 'Ok')


			if prop['darkframe'] == True:
				prop['darkframe'].setValue(False)
				self.dark.add_mean(self.im)
				prop.setAttr('state', 'Ok')

			if prop['hotpixels'] == True:
				prop['hotpixels'].setValue(False)
				self.hotpix_find()
				self.hotpix_update()
				if len(self.hotpixels) > 1000:
					self.hotpix_cnt = None
				prop.setAttr('state', 'Ok')

		
		
	
	def cmd(self, cmd):
		if cmd == 'solver-reset':
			if self.solver is not None:
				self.solver.terminate(wait=True)
			self.status['field_deg'] = None
			self.solver = None
			self.plotter = None
			if self.tid == 'guider':
				self.status['ra'], self.status['dec'], self.status['max_radius'] = self.mount.get_oag_pos()
			else:
				self.status['ra'], self.status['dec'] = self.mount.polar.zenith()
			self.status['radius'] = self.status['max_radius']
			self.field_corr = None
			self.field_corr_limit = 10

		if cmd == 'solver-retry':
			if self.solver is not None:
				self.solver.terminate(wait=False)
			if self.full_res_solver is not None:
                                self.full_res_solver.terminate(wait=False)
			if self.tid == 'guider':
				self.status['ra'], self.status['dec'], self.status['max_radius'] = self.mount.get_oag_pos()
			else:
				self.status['ra'], self.status['dec'] = self.mount.polar.zenith()
			self.status['radius'] = self.status['max_radius']

		if cmd == 'dark':
			self.dark.add_mean(self.im)

		if cmd == 'hotpixels':
			self.hotpix_find()
			self.hotpix_update()
			if len(self.hotpixels) > 1000:
				self.hotpix_cnt = None
		
#		if cmd.startswith('disp-'):
#			self.status['dispmode'] = cmd[len('disp-'):]

#		if cmd.startswith('full-disp-'):
#			self.status['full_dispmode'] = cmd

		if cmd == 'save':
			img = self.stack.get()
			if img is not None:
				cv2.imwrite(self.tid + str(int(time.time())) + ".tif", img)

		if cmd == 'polar-reset':
			if self.polar_tid is not None:
				self.mount.polar.reset()

		if cmd == 'polar-align':
			self.mount.polar.set_mode('adjust')

		if cmd.startswith('gps') and self.polar_tid is not None:
			try:
				str_gps = cmd[len('gps'):]
				(lat, lon) = [float(n) for n in str_gps.split(',')]
				self.mount.polar.set_gps((lat, lon))
			except:
				log.exception('Unexpected error')
		if cmd.startswith('go-left'):
				self.mount.move_main_px(self.im.shape[1] * self.status['go_by'], 0, self.tid)
		if cmd.startswith('go-right'):
				self.mount.move_main_px(-self.im.shape[1] * self.status['go_by'], 0, self.tid)
		if cmd.startswith('go-down'):
				self.mount.move_main_px(0, -self.im.shape[0] * self.status['go_by'], self.tid)
		if cmd.startswith('go-up') and self.mount.go_dec is not None:
				self.mount.move_main_px(0, self.im.shape[0] * self.status['go_by'], self.tid)
		
		if cmd.startswith('go-stop'):
			self.mount.stop()
		
		if cmd.startswith('go-by-'):
			try:
				self.status['go_by'] = float(cmd[len('go-by-'):])
			except:
				pass
		if cmd.startswith('center-'):
			w = self.im.shape[1]
			h = self.im.shape[0]
			reg = (0, 0, w, h)
			if cmd == 'center-ul':
				reg = (0, 0, w * 0.51, h * 0.51)
			if cmd == 'center-ur':
				reg = (w * 0.49, 0, w, h * 0.51)
			if cmd == 'center-ll':
				reg = (0, h * 0.49, w * 0.51, h)
			if cmd == 'center-lr':
				reg = (w * 0.49, h * 0.49, w, h)
			
			for y, x, v in self.stack.get_xy():
				if x >= reg[0] and y >= reg[1] and x < reg[2] and y < reg[3] and (x - w / 2)**2 + (y - h / 2)**2 > 400:
					self.mount.move_main_px(-x + w / 2, -y + h / 2, self.tid)
					break

		if cmd.startswith('profile-'):
			try:
				self.set_profile(int(cmd[len('profile-'):]))
			except:
				pass
		
	
	def proc_full_res(self, imgdata, name = None):
		with self.full_res_lock:
			full_dispmode = self.props['full_dispmode'].getActiveSwitch()
			process = psutil.Process(os.getpid())
			t = time.time()
			(temp, hum) = self.mount.temp_sensor.get()

			hdulist = None
			
			try:
				if name.endswith('.fits'):
					hdulist=fits.open(imgdata)
					im_c = hdulist[0].data
					
					ui.imshow('full_res', apply_gamma16_to8(im_c, 0.6))
#			log.error("shape %s", im.shape) #(3, 720, 1280)
#			im = im[1]

				if (full_dispmode == 'full-disp-orig'):
					if self.full_res is not None:
						self.full_res['full_hfr'].append(0)
						self.full_res['full_name'].append(name)
						self.full_res['full_temp'].append(temp)
						self.full_res['full_hum'].append(hum)
						self.full_res['full_ts'] = t
					cmdQueue.put('capture-full-res-done')
					if hdulist is not None:
						hdulist.close()
					return

				if name.endswith('.jpg'):
					im_c = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), -1)

				log.info("full_res decoded")
				#mean, stddev = cv2.meanStdDev(im_c)
				#im_c[:,:,0] = cv2.subtract(im_c[:,:,0], mean[0])
				#im_c[:,:,1] = cv2.subtract(im_c[:,:,1], mean[1])
				#im_c[:,:,2] = cv2.subtract(im_c[:,:,2], mean[2])
				im_c = subtract_bg(im_c, 15)

				if name.endswith('.jpg'):
					im = cv2.cvtColor(im_c, cv2.COLOR_RGB2GRAY);
					im = apply_gamma8(im, 2.2)
				else:
					im = im_c #cv2.add(im_c, 0, dtype=cv2.CV_8UC1)
		
				log.info("full_res bg")
				pts = find_max(im, 12, 200)
		
				w = im.shape[1]
				h = im.shape[0]
				log.info("full_res max %d", len(pts))
			except:
				log.exception('full_res bg')
				if self.full_res is not None:
					self.full_res['full_hfr'].append(0)
					self.full_res['full_name'].append(name)
					self.full_res['full_temp'].append(temp)
					self.full_res['full_hum'].append(hum)
					self.full_res['full_ts'] = t
				cmdQueue.put('capture-full-res-done')
				if hdulist is not None:
					hdulist.close()
				return

			if len(pts) < 1:
				log.info("full_res no sources detected")
				if self.full_res is not None:
					self.full_res['full_hfr'].append(0)
					self.full_res['full_name'].append(name)
					self.full_res['full_temp'].append(temp)
					self.full_res['full_hum'].append(hum)
					self.full_res['full_ts'] = t
				cmdQueue.put('capture-full-res-done')
				if hdulist is not None:
					hdulist.close()
				return
		
			try:
				pts_v_max = pts[0, 2]
				pts_v_min = pts[-1, 2]
				pts_v_thr = pts_v_min + (pts_v_max - pts_v_min) * 0.8
			
				pts_no_over = pts[(pts[:, 2] <= pts_v_thr)]
				log.info("pts min %f max %f thr %f len %d %d", pts_v_min, pts_v_max, pts_v_thr, len(pts), len(pts_no_over))
		
				hfr_list = get_hfr_field(im, pts_no_over, sub_bg = True)
				log.info("full_res get hfr")
				hfr_list = filter_hfr_list(hfr_list)
		
				full_hfr = np.mean(np.array(hfr_list)[:,2])
			
				if self.full_res is not None:
					self.full_res['full_hfr'].append(full_hfr)
					self.full_res['full_name'].append(name)
					self.full_res['full_temp'].append(temp)
					self.full_res['full_hum'].append(hum)
					self.full_res['full_ts'] = t
		
				log.info("full_res filter hfr %f", full_hfr)

				ell_list = []
				for p in hfr_list:
					patch_size = int(p[2] * 4 + 2)
					a = getRectSubPix(im, (patch_size, patch_size), (p[1] - 0.5, p[0] - 0.5), patchType=cv2.CV_32FC1)
					ell_list.append(fit_ellipse(a))
				
				del im
			except:
				log.exception('full_res hfr')
				cmdQueue.put('capture-full-res-done')
				if hdulist is not None:
					hdulist.close()
				return

			log.info("full_res ell")

				
			if len(pts) < 7:
				log.info("full_res no sources detected")
				cmdQueue.put('capture-full-res-done')
				if hdulist is not None:
					hdulist.close()
				return

			try:
				if (full_dispmode != 'full-disp-hfr'):
					solver = Solver(sources_list = pts, field_w = w, field_h = h, ra = self.status['ra'], dec = self.status['dec'], field_deg = self.status['field_deg'], radius = 100)
					self.full_res_solver = solver
					solver.start()
				else:
					self.full_res_solver = None
		
				im_c = cv2.normalize(im_c,  im_c, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
				im_c = apply_gamma8(im_c, 0.6)

				log.info("full_res norm")

				for i, p in enumerate(hfr_list):
					val, vec = ell_list[i]
					#log.info("%s %s", val, vec)
					#fwhm = get_fwhm(a)
					#log.info "fwhm", fwhm, p[2]
					cv2.circle(im_c, (int(p[1]), int(p[0])), int(p[2] * 10), (100,100,100), 2)
					#cv2.circle(im_c, (int(p[1]), int(p[0])), int(fwhm * 10), (255,255,255), 2)
					if val[0] > val[1]:
						v = val[0] * 10
						vec = vec[0]
						v2 = val[1] * 10
					else:
						v = val[1] * 10
						vec = vec[1]
						v2 = val[0] * 10
		
					v += (v - v2) * 5
		
					p11 = (p[1], p[0]) - vec * v
					p12 = (p[1], p[0]) - vec * v2
					p13 = (p[1], p[0]) + vec * v
					p14 = (p[1], p[0]) + vec * v2
					cv2.line(im_c, (int(p11[0]), int(p11[1])), (int(p12[0]), int(p12[1])), (255,0,0), 2)
					cv2.line(im_c, (int(p13[0]), int(p13[1])), (int(p14[0]), int(p14[1])), (255,0,0), 2)

				log.info("full_res plot")

				ui.imshow('full_res', im_c)
		
				if (full_dispmode == 'full-disp-hfr'):
					cmdQueue.put('capture-full-res-done')
					if hdulist is not None:
						hdulist.close()
					return
				solver.join()
				if solver.solved:
					log.info("full-res solved: %f %f", solver.ra, solver.dec)
					with self.solvedlock:
						self.status['ra'] = solver.ra
						self.status['dec'] = solver.dec
						self.status['field_deg'] = solver.field_deg
						self.status['radius'] = solver.field_deg
						self.mount.polar.set_pos_tan(solver.wcs, t, "full-res")
				
						if self.im is not None:
							self.wcs = scale_wcs(solver.wcs, (self.im.shape[1], self.im.shape[0]))
							self.plotter_off = np.array([0.0, 0.0])
							self.plotter = Plotter(self.wcs)
							self.mount.set_pos_tan(self.wcs, t, self.tid)


					if (full_dispmode.startswith('full-disp-zoom-')):
						zoom = full_dispmode[len('full-disp-zoom-'):]
					else:
						zoom = 1
			

			
					plotter=Plotter(solver.wcs)
					plot = plotter.plot(im_c, scale = zoom)
					ui.imshow('full_res', plot)

				else:
					log.info("full-res not solved")
		
			except:
				log.exception('full_res solver')
			cmdQueue.put('capture-full-res-done')
			if hdulist is not None:
				hdulist.close()
	
	def get_xy_cor(self):
		xy = self.stack.get_xy()
		if self.field_corr is not None:
			xnew = interpolate2d(self.field_corr[:, :, 0], xy[:, 1], xy[:, 0])
			ynew = interpolate2d(self.field_corr[:, :, 1], xy[:, 1], xy[:, 0])
			
			xy[:, 0] = ynew
			xy[:, 1] = xnew
		return xy

	def update_field_cor(self):
		field_corr_list = np.array(self.field_corr_list)
		if self.field_corr is not None:
			xnew = interpolate2d(self.field_corr[:, :, 0], field_corr_list[:, 0], field_corr_list[:, 1])
			ynew = interpolate2d(self.field_corr[:, :, 1], field_corr_list[:, 0], field_corr_list[:, 1])
			
			field_corr_list[:, 0] = xnew
			field_corr_list[:, 1] = ynew
		log.info(field_corr_list)


		M = pt_translation_rotate(field_corr_list[:, 2:4], field_corr_list[:, 0:2], np.ones((field_corr_list.shape[0])))
		log.info("%s", M)
#		#M2 = cv2.estimateRigidTransform(np.array(field_corr_list[:, 2:4]), np.array(field_corr_list[:, 0:2]), False)
		M2, inl = cv2.estimateAffinePartial2D(np.array(field_corr_list[:, 2:4]), np.array(field_corr_list[:, 0:2]))
		M2 = np.matrix(M2.T)
		log.info("M2")
		log.info("%s", M2)
		
		field_corr_list[:, 2:4] = np.insert(field_corr_list[:, 2:4], 2, 1.0, axis=1).dot(M2).A
		log.info(field_corr_list)

		if len(field_corr_list) < 1000:
			deg = 1
		else:
			deg = 2

		xcorr = polyfit2d(field_corr_list[:, 2], field_corr_list[:, 3], field_corr_list[:, 0], deg)
		ycorr = polyfit2d(field_corr_list[:, 2], field_corr_list[:, 3], field_corr_list[:, 1], deg)
		
		#log.info "xcorr"
		#log.info xcorr
		#log.info "ycorr"
		#log.info ycorr
		h, w = self.im.shape
		xr = np.arange(0, w, dtype=np.float32)
		yr = np.arange(0, h, dtype=np.float32)
		
		m = np.empty((h,w,2), dtype=np.float32)
		m[:,:,0] = np.polynomial.polynomial.polygrid2d(xr, yr, xcorr).T
		m[:,:,1] = np.polynomial.polynomial.polygrid2d(xr, yr, ycorr).T
		
		log.info("field_corr")
		log.info("%s", self.field_corr)
		log.info("new")
		log.info(m)
		fn = self.status['field_corr']
		if fn is not None:
			if fn.endswith('.npy'):
				fn = fn[:-4]
			fn = "%s.%d.npy" % (fn, int(time.time()))
			np.save(fn, m)
		self.field_corr = m
		#sys.exit(1)

		self.field_corr_list = []

def fit_line(xylist, sigma = 2):
	a = np.array(xylist)
	m, c = np.polyfit(a[:, 0], a[:, 1], 1)
	log.info("fit_line res1 %f %f" , m ,c)
	
	for i in range(1, 5):
		d2 = (a[:, 0] * m + c - a[:, 1]) ** 2
		var = np.mean(d2)
		a = a[(d2 < var * sigma ** 2)]
		m, c = np.polyfit(a[:, 0], a[:, 1], 1)
		log.info("fit_line res2 %f %f" , m ,c)
	return m, c

class GuiderAlg(object):
	def __init__(self, go, status):
		self.go = go
		self.status = status
		self.status.setdefault('min_move', 0.1)
		self.status.setdefault('aggressivness', 0.5)
		self.status['t_delay'] = 0.5
		self.status['last_move'] = 0
		self.corr = 0
		self.status['restart'] = False
		self.parity = 0

	def set_params(self, pixpersec, pixpersec_neg, parity = 1):
		self.pixpersec = pixpersec
		self.pixpersec_neg = pixpersec_neg
		self.parity = parity
        	

	
	def get_corr_delay(self, t_proc):
		return self.go.recent_avg(self.status['t_delay'] + t_proc, self.pixpersec, -self.pixpersec_neg)
	

class GuiderAlgDec(GuiderAlg):
	def __init__(self, driver, device, go, status):
		super(GuiderAlgDec, self).__init__(go, status)
		self.status.setdefault('rev_move', 2.0)
		self.status.setdefault('smooth_c', 0.1)
		self.corr_acc = 0.0

		self.driver = driver
		self.device = device
		driver.defineProperties("""
		<INDIDriver>

			<defNumberVector device="{0}" name="guider_dec" label="Guider Dec" group="Guider" state="Idle" perm="rw">
				<defNumber name="aggressivness" label="aggressivness" format="%1.1f" min="0" max="1.5" step="0.1">0.5</defNumber>
				<defNumber name="min_move" label="min_move" format="%1.1f" min="0" max="5" step="0.1">0.1</defNumber>
				<defNumber name="rev_move" label="rev_move" format="%1.1f" min="0" max="5" step="0.1">2.0</defNumber>
				<defNumber name="t_delay" label="t_delay" format="%1.1f" min="0" max="10" step="0.1">0.5</defNumber>
				<defNumber name="smooth" label="smooth" format="%1.2f" min="0.01" max="1" step="0.01">0.1</defNumber>
			</defNumberVector>

			<defNumberVector device="{0}" name="guider_dec_move" label="Guider Dec Output" group="Guider" state="Idle" perm="ro">
				<defNumber name="current" label="move" format="%2.3f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="acc" label="acc" format="%2.3f" min="0" max="0" step="0">0</defNumber>
			</defNumberVector>

			<defSwitchVector device="{0}" name="guider_dec_commands" label="Guider Dec Commands" group="Guider" state="Idle" perm="rw" rule="AtMostOne">
				<defSwitch name="reset" label="reset">Off</defSwitch>
			</defSwitchVector>

		</INDIDriver>
		""".format(device))

		self.props = driver[device]
		
	def handleNewProp(self, msg, prop):
		name = prop.getAttr('name')
		if name == 'guider_dec':
			self.status['aggressivness'], self.status['min_move'], self.status['rev_move'], self.status['t_delay'], self.status['smooth_c'] = prop.to_array()
			prop.setAttr('state', 'Ok')
			
		elif name == 'guider_dec_commands':
			if prop['reset'] == True:
				prop['reset'].setValue(False)
				prop.setAttr('state', 'Ok')


	def get_corr(self, err, err2, t0):
		corr1 = err * self.parity + self.get_corr_delay(time.time() - t0) 
		
		corr1 *= self.status['aggressivness']
		
		smooth_c = self.status['smooth_c']

		if np.abs(corr1) < self.status['rev_move']:
			corr_acc = self.corr_acc + corr1 * smooth_c
			if np.abs(corr_acc) < 3:
				self.corr_acc = corr_acc
		else:
			self.corr_acc = 0
		
		corr = corr1 + self.corr_acc

		if self.status['restart']:
			log.info("dec err %f, corr1 %f, corr_acc %f, corr %f, restart", err, corr1, self.corr_acc, corr)
			self.status['restart'] = False
			self.corr_acc = 0
			return corr
			
		if corr > 0 and self.status['last_move'] < 0 and corr < self.status['rev_move']:
			corr = 0
		elif corr < 0 and self.status['last_move'] > 0 and corr > -self.status['rev_move']:
			corr = 0
		
		if corr * self.status['last_move'] < 0:
			self.corr_acc = 0
		
		
		log.info("dec err %f, corr1 %f, corr_acc %f, corr %f", err, corr1, self.corr_acc, corr)

		return corr

	def proc(self, err, err2, t0):
		if self.parity == 0:
			return

		corr = self.get_corr(err, err2, t0)
		self.corr = corr
		
		if corr > self.status['min_move']:
			self.go.out(-1, corr / self.pixpersec_neg)
			self.status['last_move'] = corr
		elif corr < -self.status['min_move']:
			self.go.out(1, -corr / self.pixpersec)
			self.status['last_move'] = corr
		else:
			self.go.out(0)
			self.corr = 0

		self.props["guider_dec_move"]["current"].setValue(self.corr)
		self.props["guider_dec_move"]["acc"].setValue(self.corr_acc)
		self.driver.enqueueSetMessage(self.props["guider_dec_move"])


class GuiderAlgRa(GuiderAlg):
	def __init__(self, driver, device, go, status):
		super(GuiderAlgRa, self).__init__(go, status)
		self.status.setdefault('smooth_c', 0.1)
		self.smooth_var2 = 1.0
		self.corr_acc = 0.0

		self.driver = driver
		self.device = device
		driver.defineProperties("""
		<INDIDriver>

			<defNumberVector device="{0}" name="guider_ra" label="Guider RA" group="Guider" state="Idle" perm="rw">
				<defNumber name="aggressivness" label="aggressivness" format="%1.1f" min="0" max="1.5" step="0.1">0.5</defNumber>
				<defNumber name="min_move" label="min_move" format="%1.1f" min="0" max="5" step="0.1">0.1</defNumber>
				<defNumber name="t_delay" label="t_delay" format="%1.1f" min="0" max="10" step="0.1">0.5</defNumber>
				<defNumber name="smooth" label="smooth" format="%1.3f" min="0.01" max="1" step="0.01">0.1</defNumber>
			</defNumberVector>

			<defNumberVector device="{0}" name="guider_ra_move" label="Guider RA Output" group="Guider" state="Idle" perm="ro">
				<defNumber name="current" label="move" format="%2.1f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="acc" label="acc" format="%2.3f" min="0" max="0" step="0">0</defNumber>
			</defNumberVector>

			<defSwitchVector device="{0}" name="guider_ra_commands" label="Guider RA Commands" group="Guider" state="Idle" perm="rw" rule="AtMostOne">
				<defSwitch name="reset" label="reset">Off</defSwitch>
			</defSwitchVector>

		</INDIDriver>
		""".format(device))
		self.props = driver[device]

	def handleNewProp(self, msg, prop):
		name = prop.getAttr('name')
		if name == 'guider_ra':
			self.status['aggressivness'], self.status['min_move'], self.status['t_delay'], self.status['smooth_c'] = prop.to_array()
			prop.setAttr('state', 'Ok')
			
		elif name == 'guider_ra_commands':
			if prop['reset'] == True:
				prop['reset'].setValue(False)
				prop.setAttr('state', 'Ok')


	def get_corr(self, err, err2, t0):
		corr = err * self.parity + self.get_corr_delay(time.time() - t0)
		corr *= self.status['aggressivness']
	
		smooth_c = self.status['smooth_c']
		corr_acc = self.corr_acc + corr * smooth_c
		if np.abs(corr_acc) < np.abs(self.pixpersec / 2.0):
			self.corr_acc = corr_acc
	
		err2norm = (err2 ** 2 / self.smooth_var2) ** 0.5
		corr *= 1.0 / (1.0 + err2norm)
		corr += self.corr_acc

		self.smooth_var2 = self.smooth_var2 * (1.0 - smooth_c) + err2**2 * smooth_c
		
		log.info("ra err %f, err2 %f, err2norm %f, err2agg %f, corr_acc %f, corr %f", err, err2, err2norm, 1.0 / (1.0 + err2norm), self.corr_acc, corr)
		return corr

	def proc(self, err, err2, t0):
		if self.parity == 0:
			return

		corr = self.get_corr(err, err2, t0)
		self.corr = corr
		
		if corr > self.status['min_move']:
			self.go.out(-1, corr / self.pixpersec_neg)
			self.status['last_move'] = corr
		elif corr < -self.status['min_move']:
			self.go.out(1, -corr / self.pixpersec)
			self.status['last_move'] = corr
		else:
			self.go.out(0)
			self.corr = 0

		self.props["guider_ra_move"]["current"].setValue(self.corr)
		self.props["guider_ra_move"]["acc"].setValue(self.corr_acc)
		self.driver.enqueueSetMessage(self.props["guider_ra_move"])


class Guider:
	def __init__(self, driver, device, status, mount, dark, tid, full_res = None):
		self.status = status

		self.driver = driver
		self.device = device
		driver.defineProperties("""
		<INDIDriver>
			<defSwitchVector device="{0}" name="guider_phase" label="Phase" group="Guider" state="Idle" perm="ro" rule="OneOfMany">
				<defSwitch name="inactive">On</defSwitch>
				<defSwitch name="start">Off</defSwitch>
				<defSwitch name="move">Off</defSwitch>
				<defSwitch name="back">Off</defSwitch>
				<defSwitch name="move_dec">Off</defSwitch>
				<defSwitch name="track">Off</defSwitch>
				<defSwitch name="close">Off</defSwitch>
			</defSwitchVector>

			<defSwitchVector device="{0}" name="expose" label="Expose" group="Guider" state="Idle" perm="rw" rule="OneOfMany">
				<defSwitch name="stop">On</defSwitch>
				<defSwitch name="guided">Off</defSwitch>
				<defSwitch name="guided2">Off</defSwitch>
				<defSwitch name="free">Off</defSwitch>
			</defSwitchVector>


			<defNumberVector device="{0}" name="callibration" label="callibration" group="Guider" state="Idle" perm="rw">
				<defNumber name="pixpersec_plus"  format="%2.2f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="pixpersec_minus" format="%2.2f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="pixpersec_dec"  format="%2.2f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="orientation" format="%4.1f" min="0" max="0" step="0">0</defNumber>
			</defNumberVector>

			<defNumberVector device="{0}" name="offset" label="Offset" group="Guider" state="Idle" perm="ro">
				<defNumber name="RA" label="RA" format="%4.2f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="DEC" label="Dec" format="%4.2f" min="0" max="0" step="0">0</defNumber>
			</defNumberVector>

		</INDIDriver>
		""".format(device))

		self.props = driver[device]




		self.status.setdefault('ra_alg', {})
		self.status.setdefault('dec_alg', {})

		self.mount = mount

		self.alg_ra = GuiderAlgRa(driver, device, self.mount.go_ra, self.status['ra_alg'])
		self.alg_dec = GuiderAlgDec(driver, device, self.mount.go_dec, self.status['dec_alg'])
		
		self.dark = dark
		self.tid = tid
		self.full_res = full_res
		self.status['seq'] = 'seq-stop'
		
		
		if self.full_res is not None:
			self.full_res['ra_err_list'] = []
			self.full_res['dec_err_list'] = []
			self.full_res['guider_hfr'] = []
			self.full_res['focuser_pos'] = []
			self.full_res['guider_ts'] = None
			self.full_res['full_ts'] = None
			
			self.full_res.setdefault('guider_hfr_cov', 0)
			self.full_res.setdefault('last_step', 0)
			self.full_res.setdefault('diff_thr', 0.5)
			self.full_res['diff_acc'] = 0
			self.full_res.setdefault('hyst', 0)
			self.full_res['temp_cor'] = 0
			

		self.reset()
		self.t0 = 0
		self.resp0 = []
		self.pt0 = []
		self.prev_t = 0

	def reset(self):
		self.changePhase('start')
		self.status['t_delay'] = None
		self.status['t_delay1'] = None
		self.status['t_delay2'] = None
		self.status['pixpersec'] = None
		self.status['pixpersec_neg'] = None
		self.status['pixpersec_dec'] = None
		self.status['curr_ra_err_list'] = []
		self.status['curr_dec_err_list'] = []
		self.status['curr_hfr_list'] = []
		self.off = (0.0, 0.0)
		self.off_t = None
		self.mount.go_ra.out(0)
		self.mount.go_dec.out(0)
		self.cnt = 0
		self.pt0 = []
		self.pt0base = []
		self.capture_in_progress = 0
		self.capture_proc_in_progress = 0
		self.capture_init = False
		self.countdown = 5
		self.parity = 0
		self.status['alarm_count'] = 0
		
		self.i0 = 0
		self.dither = complex(0, 0)
		self.pos_corr = [0, 0]

	def dark_add_masked(self, im):
		h, w = im.shape
		pts = []
		for p in self.pt0:
			(x, y) = (int(p[1] + 0.5 + self.off[1]), int(p[0] + 0.5 + self.off[0]))
			if (x < 0):
				continue
			if (y < 0):
				continue
			if (x > w - 1):
				continue
			if (y > h - 1):
				continue
			pts.append((x,y))
		return self.dark.add_mean(im, pts)

	def update_pt0(self):
		try:
			dither_off = self.dither * self.ref_off
			self.pt0 = np.array(self.pt0base, copy=True)
			self.pt0[:, 0] += dither_off.real + self.pos_corr[0]
			self.pt0[:, 1] += dither_off.imag + self.pos_corr[1]
		except:
			pass
		

	def handleNewProp(self, msg, prop):
		name = prop.getAttr('name')
		if name == 'expose':
			self.status['seq'] = "seq-" + prop.getActiveSwitch()
			prop.setAttr('state', 'Ok')
		elif name == 'callibration':
			self.status['pixpersec'], self.status['pixpersec_neg'], self.status['pixpersec_dec'], fixme = prop.to_array()
			prop.setAttr('state', 'Ok')
		else:
			self.alg_ra.handleNewProp(msg, prop)
			self.alg_dec.handleNewProp(msg, prop)


	def changePhase(self, mode):
		self.status['mode'] = mode
		self.props['guider_phase'].enforceRule(mode, True)
		self.driver.enqueueSetMessage(self.props['guider_phase'])

		
	def cmd(self, cmd):
		if cmd == "stop":
			self.mount.go_ra.out(0)
			self.mount.go_dec.out(0)
			

		elif cmd == "capture-started":
			self.capture_in_progress += 1
			self.capture_proc_in_progress += 1
			self.capture_init = False

		elif cmd == "capture-finished":
			self.capture_in_progress -= 1
			if self.capture_in_progress < 0:
				self.capture_in_progress = 0
				log.info("capture_in_progress negative")
		
			if self.capture_in_progress == 0:
				try:
					dither_dec = self.dither.imag
					if self.status['pixpersec_dec'] is not None:
						if self.status['pixpersec_dec'] * self.status['dec_alg']['last_move'] * self.parity > 0:
							dither_dec -= 1
						else:
							dither_dec += 1
						if np.abs(dither_dec) > 50:
							dither_dec = 0
					self.dither = complex((self.dither.real + 11) % 37, dither_dec)
					self.update_pt0()
					self.countdown = 4
				except:
					log.exception('dither')
					pass
				
				#self.status['dec_alg']['restart'] = True
				
				if self.full_res is not None:
					if len(self.status['curr_ra_err_list' ]) > 0:
						self.full_res['ra_err_list' ].append(np.mean(np.array(self.status['curr_ra_err_list' ]) ** 2) ** 0.5)
					else:
						self.full_res['ra_err_list' ].append(0.0)
					if len(self.status['curr_dec_err_list']) > 0:
						self.full_res['dec_err_list'].append(np.mean(np.array(self.status['curr_dec_err_list']) ** 2) ** 0.5)
					else:
						self.full_res['dec_err_list'].append(0.0)
					if len(self.status['curr_hfr_list']) > 0:
						self.full_res['guider_hfr'].append(np.mean(self.status['curr_hfr_list']))
					else:
						self.full_res['guider_hfr'].append(0.0)
					self.full_res['guider_ts'] = time.time()
			
		elif cmd == 'guider-up':
			self.pos_corr[0] -= 5
			self.update_pt0()
		elif cmd == 'guider-down':
			self.pos_corr[0] += 5
			self.update_pt0()
		elif cmd == 'guider-left':
			self.pos_corr[1] -= 5
			self.update_pt0()
		elif cmd == 'guider-right':
			self.pos_corr[1] += 5
			self.update_pt0()
			
		elif cmd == "capture-full-res-done":
			self.capture_proc_in_progress -= 1
			if self.capture_proc_in_progress < 0:
				self.capture_proc_in_progress = 0
				log.info("capture_proc_in_progress negative")
			if self.capture_proc_in_progress == 0 and self.full_res is not None:
				if self.full_res['diff_thr'] >= 0:
					self.focus_loop()
		
		elif cmd.startswith('ra-') or cmd.startswith('dec-'):
			try:
				if cmd.startswith('ra-'):
					g_alg = 'ra_alg'
					cmd = cmd[len('ra-'):]
				else:
					g_alg = 'dec_alg'
					cmd = cmd[len('dec-'):]
				field, val = cmd.split('-')
				
				self.status[g_alg][field] = float(val)
			except:
				pass

		elif cmd.startswith('diff-thr-'):
			try:
				if self.full_res is not None:
					self.full_res['diff_thr'] = float(cmd[len('diff-thr-'):])
			except:
				pass


		elif cmd.startswith('seq-'):
			self.status['seq'] = cmd

	def focus_loop(self):
		if self.full_res['guider_ts'] is None or self.full_res['full_ts'] is None or abs(self.full_res['guider_ts'] - self.full_res['full_ts']) > 60:
			self.full_res['guider_ts'] = None
			self.full_res['full_ts'] = None
			return
		
		guider_hfr_diff = 0.0
		if len(self.full_res['guider_hfr']) > 1:
			guider_hfr_diff = self.full_res['guider_hfr'][-1] - self.full_res['guider_hfr'][-2]
			cov = guider_hfr_diff * self.full_res['last_step']
			self.full_res['guider_hfr_cov'] = self.full_res['guider_hfr_cov'] * 0.8 + cov
		
		full_hfr_diff = 0.0
		if len(self.full_res['full_hfr']) > 1:
			full_hfr_diff = self.full_res['full_hfr'][-1] - self.full_res['full_hfr'][-2]

		g_diff = 0.0
		if len(self.full_res['dec_err_list']) > 1 and len(self.full_res['ra_err_list']) > 1:
			g1 = (self.full_res['dec_err_list'][-2] ** 2 + self.full_res['ra_err_list'][-2] ** 2) ** 0.5
			g2 = (self.full_res['dec_err_list'][-1] ** 2 + self.full_res['ra_err_list'][-1] ** 2) ** 0.5
			g_diff = g2 - g1
		
		
		log.info("focus_loop full: %f  guider: %f,  g: %f, cov %f", full_hfr_diff, guider_hfr_diff, g_diff, self.full_res['guider_hfr_cov'])
		
		self.full_res['diff_acc'] = max(self.full_res['diff_acc'] + full_hfr_diff, 0.0)

		temp_coef = self.full_res.get('temp_coef', 0)
		temp_pos = (self.full_res['full_temp'][-1] - self.full_res['full_temp'][0]) * temp_coef
		
		if self.full_res['diff_thr'] == 0 or self.full_res['full_hfr'][-1] == 0:
			temp_diff = temp_pos - self.full_res['temp_cor']
			last_step = 0
			if abs(temp_diff) > 0.5:
				if self.full_res['last_step'] * temp_diff < 0:
					for st in range(0, self.full_res['hyst']):
						if temp_diff < 0:
							log.info("focus_loop temp hyst -1")
							cmdQueue.put('f-1')
							last_step += -1.0
						else:
							log.info("focus_loop temp hyst +1")
							cmdQueue.put('f+1')
							last_step += 1.0
				for st in range(0, int(abs(temp_diff) + 0.5)):
					if temp_diff < 0:
						log.info("focus_loop temp -1")
						cmdQueue.put('f-1')
						last_step += -1.0
						self.full_res['temp_cor'] -= 1
					else:
						log.info("focus_loop temp +1")
						cmdQueue.put('f+1')
						last_step += 1.0
						self.full_res['temp_cor'] += 1
			self.full_res['last_step'] = last_step
			
		else:
			self.full_res['temp_cor'] = int(temp_pos) 
		
			if self.full_res['diff_acc'] > self.full_res['diff_thr']:
				self.full_res['diff_acc'] = 0 #reset
				if self.full_res['last_step'] < 0:
					log.info("focus_loop rev 1")
					for st in range(0, 1 + self.full_res['hyst']):
						cmdQueue.put('f+1')
					self.full_res['last_step'] = 1.0 * (1 + self.full_res['hyst'])
				else:
					log.info("focus_loop rev -1")
					for st in range(0, 1 + self.full_res['hyst']):
						cmdQueue.put('f-1')
					self.full_res['last_step'] = -1.0 * (1 + self.full_res['hyst'])
			else:
				if self.full_res['last_step'] < 0:
					log.info("focus_loop keep -1")
					cmdQueue.put('f-1')
					self.full_res['last_step'] = -1.0
				else:
					log.info("focus_loop keep +1")
					cmdQueue.put('f+1')
					self.full_res['last_step'] = 1.0
		
		if len(self.full_res['focuser_pos']) > 0:
			focuser_pos = self.full_res['focuser_pos'][-1]
		else:
			focuser_pos = 0
		focuser_pos += self.full_res['last_step']
		self.full_res['focuser_pos'].append(focuser_pos)

	def proc_frame(self, im, i):
		t = time.time()

		if im.ndim > 2:
			im = cv2.min(cv2.min(im[:, :, 0], im[:, :, 1]), im[:, :, 2])
		
		if len(self.pt0) == 0:
			cmdQueue.put('navigator')
			self.mount.go_ra.out(0)
			self.mount.go_dec.out(0)

		if (self.dark.len() >= 4):
			im_sub = cv2.subtract(im, self.dark.get(), dtype=cv2_dtype(im.dtype))
		else:
			im_sub = im

		if self.status['mode'] == 'close':
			pt0, pt, match = centroid_list(im_sub, self.pt0, self.off)
			if len(match) == 0:
				self.changePhase('track')		
		
		if self.status['mode'] != 'close':
			im_sub = subtract_bg(im_sub, 10)

			pt = find_max(im_sub, 20, n = 30)

		disp = normalize(im_sub)


		try:
			fps = 1.0 / (t - self.prev_t)
		except:
			fps = 0
		
		status = "#%d Guider:%s fps:%.1f" % (i, self.status['mode'], fps)

		if self.status['mode'] == 'start':
			self.used_cnt = []
			self.cnt = 0
			self.dist = 1.0
			self.mount.go_ra.out(1)
			self.changePhase('move')
			self.t0 = time.time()
			self.resp0 = []
			self.i0 = i
			cmdQueue.put('disp-df-cor')

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
				#log.info "triangle", off, match
			
				pt0, pt, match = match_closest(self.pt0, pt, 5, off)
			
			if len(match) > 0:
			
				off, weights = avg_pt(pt0, pt, noise = 3)
				#log.info "weights", weights 
				dist = np.linalg.norm(off)

				if (dist > 20):
					if i % int(2 + (i - self.i0) / self.dist) == 0:
						self.dark.add_mean(im)
					self.resp0.append((t - self.t0, dist, 0))
				max_move = 30 * (t - self.t0)
				if self.off_t is not None:
					max_move =  self.dist + 30 * (t - self.off_t)
				if (dist > self.dist and dist < max_move):
					self.dist = dist
					self.off = off
					self.off_t = t
			
				#log.info off, dist
				pt_ok = match[np.where(weights > 0), 0][0]
				self.used_cnt.extend(pt_ok)

				for i in pt_ok:
					p = self.pt0[i]
					cv2.circle(disp, (int(p[1] + self.off[1] + 0.5), int(p[0] + self.off[0] + 0.5)), 13, (255), 1)

				status += " dist:%.1f" % (dist)

				if self.dist > 100 and len(self.resp0) > 15 or len(self.resp0) > 60:
					self.t1 = time.time()
					dt = t - self.t0
					self.mount.go_ra.out(-1)
				
					aresp = np.array(self.resp0)
					aresp1 = aresp[aresp[:, 1] > 10]
					if len(aresp1) < 3:
						aresp1 = aresp
					m, c = fit_line(aresp1)

					self.status['pixpersec'] = m
					self.status['t_delay1'] = max(-c / m, 0.5)
					
					self.pixperframe = self.status['pixpersec'] * dt / self.cnt
					self.dist = m * dt + c
					self.ref_off = complex(*self.off) / dist
				
					log.info("pixpersec %f pixperframe %f t_delay1 %f", self.status['pixpersec'], self.pixperframe, self.status['t_delay1'])
				
					bincnt = np.bincount(self.used_cnt, minlength=len(self.pt0))
					cntlimit = min(5, np.mean(bincnt))
					self.pt0 = np.array(self.pt0)[(bincnt > cntlimit)]
					self.pt0base = self.pt0
				
					self.cnt = 0
					self.changePhase('back')
				
					self.mount.go_ra.out(-1, self.dist / self.status['pixpersec'])
					cmdQueue.put('interrupt')

			for p in pt:
				cv2.circle(disp, (int(p[1] + 0.5), int(p[0] + 0.5)), 10, (255), 1)

		elif self.status['mode'] == 'back':
			self.cnt += 1

			pt1m, pt2m, match = match_triangle(self.pt0, pt, 5, 50, self.off)
			if len(match) > 0:
				off, weights = avg_pt(pt1m, pt2m)
				#log.info "triangle", off, match
			
				pt0, pt, match = match_closest(self.pt0, pt, 5, off)
				
			if len(match) > 0:
				self.off, weights = avg_pt(pt0, pt)
				err = complex(*self.off) / self.ref_off
				
				self.resp0.append((t - self.t0, err.real, err.imag))
			
				status += " err:%.1f %.1f t_delay1:%.1f" % (err.real, err.imag, self.status['t_delay1'])

				if (err.real > 30):
					self.dark_add_masked(im)

				for p in pt:
					cv2.circle(disp, (int(p[1] + 0.5), int(p[0] + 0.5)), 10, (255), 1)
				self.mount.go_ra.out(-1, err.real / self.status['pixpersec'])
				
				if err.real < self.status['pixpersec'] * self.status['t_delay1'] + self.pixperframe:
					self.t2 = t
					dt = self.t2 - self.t1
					
					aresp = np.array(self.resp0)
					aresp1 = aresp[aresp[:, 0] > self.t1 + self.status['t_delay1'] - self.t0]
					m, c = fit_line(aresp1)

					self.status['pixpersec_neg'] = -m
					self.status['t_delay2'] = max(0.5, (c + self.status['t_delay1'] * self.status['pixpersec']) / (self.status['pixpersec'] + self.status['pixpersec_neg']) - self.t1 + self.t0)


					self.pixperframe_neg = self.status['pixpersec_neg'] * dt / self.cnt
				
					log.info("pixpersec_neg %f pixperframe_neg %f t_delay2 %f", self.status['pixpersec_neg'], self.pixperframe_neg, self.status['t_delay2'])
					self.status['t_delay'] = (self.status['t_delay1'] + self.status['t_delay2']) / 2
					
					#testing
					try:
						self.status['t_delay'] = self.status['navigator']['camera']['exp-sec'] * 1.0
					except:
						pass
				
					self.err0_dec = err.imag
					
					if self.mount.go_dec is not None:
						self.mount.go_dec.out(1, self.status['t_delay'] * 2 + 12)
						self.changePhase('move_dec')
					else:
						self.changePhase('track')
						self.status['pixpersec_dec'] = None
						self.mount.set_guider_calib(np.angle(self.ref_off, deg=True), 0, self.status['pixpersec'], self.status['pixpersec_neg'], 0, 0)
						#self.alg_ra = GuiderAlgRa(self.mount.go_ra, self.status['t_delay'], self.status['pixpersec'], self.status['pixpersec_neg'], self.status['ra_alg'])
						self.alg_ra.set_params(self.status['pixpersec'], self.status['pixpersec_neg'])

						self.alg_dec.set_params(0, 0, 0)
						cmdQueue.put('interrupt')

					self.props['callibration']['pixpersec_plus'].setValue(self.status['pixpersec'] or 0)
					self.props['callibration']['pixpersec_minus'].setValue(self.status['pixpersec_neg'] or 0)
					self.props['callibration']['pixpersec_dec'].setValue(self.status['pixpersec_dec'] or 0)
					self.driver.enqueueSetMessage(self.props['callibration'])

		elif self.status['mode'] == 'move_dec':
			pt1m, pt2m, match = match_triangle(self.pt0, pt, 5, 50, self.off)
			if len(match) > 0:
				off, weights = avg_pt(pt1m, pt2m)
				#log.info "triangle", off, match
			
				pt0, pt, match = match_closest(self.pt0, pt, 5, off)
				
			if len(match) > 0:
				self.off, weights = avg_pt(pt0, pt)
				err = complex(*self.off) / self.ref_off
				
				self.resp0.append((t - self.t0, err.real, err.imag))
			
				status += " err:%.1f %.1f t_delay:%.1f" % (err.real, err.imag, self.status['t_delay'])
				
				log.info(status)
				log.info("%f %f", abs(err.imag - self.err0_dec), 2 * self.status['pixpersec'])
			
				if t > self.t2 + self.status['t_delay'] * 2 + 10 or abs(err.imag - self.err0_dec) > 50:
					aresp = np.array(self.resp0)
					aresp1 = aresp[aresp[:, 0] > self.t2 + self.status['t_delay1'] - self.t0, ::2]
					m, c = fit_line(aresp1)

					if abs(err.imag - self.err0_dec) < min(2 * self.status['pixpersec'], 10):
						log.info("no dec axis")
						self.parity = 0
						self.status['pixpersec_dec'] = None
						
					elif m > 0:
						log.info("dec_pos")
						self.parity = 1
						self.status['pixpersec_dec'] = m
					else:
						log.info("dec_neg")
						self.parity = -1
						self.status['pixpersec_dec'] = -m

						log.info("move_dec test2 %f %f", self.status['pixpersec_dec'], m)

					self.changePhase('track')
					cmdQueue.put('interrupt')
					self.mount.set_guider_calib(np.angle(self.ref_off, deg=True), self.parity, self.status['pixpersec'], self.status['pixpersec_neg'], self.status['pixpersec_dec'], self.status['pixpersec_dec'])
					self.alg_ra.set_params(self.status['pixpersec'], self.status['pixpersec_neg'])
					if self.status['pixpersec_dec'] is not None:
						self.alg_dec.set_params(self.status['pixpersec_dec'], self.status['pixpersec_dec'], parity = self.parity)
					else:
						self.alg_dec.set_params(0, 0, 0)
					
					self.props['callibration']['pixpersec_plus'].setValue(self.status['pixpersec'] or 0)
					self.props['callibration']['pixpersec_minus'].setValue(self.status['pixpersec_neg'] or 0)
					self.props['callibration']['pixpersec_dec'].setValue(self.status['pixpersec_dec'] or 0)
					self.driver.enqueueSetMessage(self.props['callibration'])


				for p in pt:
					cv2.circle(disp, (int(p[1] + 0.5), int(p[0] + 0.5)), 10, (255), 1)


		elif self.status['mode'] == 'track' or self.status['mode'] == 'close':
			if self.status['mode'] == 'track':
				pt1m, pt2m, match = match_triangle(self.pt0, pt, 5, 80, self.off)
				if len(match) > 0:
					off, weights = avg_pt(pt1m, pt2m)
					#log.info "triangle", off, match
			
					pt0, pt, match = match_closest(self.pt0, pt, 5, off)

			if len(match) > 0:
				self.off, weights = avg_pt(pt0, pt)
				log.info("centroid off1 %s", self.off)
				self.off += centroid_mean(im_sub, pt0, self.off)
				log.info("centroid off2 %s", self.off)
				
				err = complex(*self.off) / self.ref_off
				self.resp0.append((t - self.t0, err.real, err.imag))
				t_proc = time.time() - t

				self.props["offset"]["RA"].setValue(err.real)
				self.props["offset"]["DEC"].setValue(err.imag)
				self.driver.enqueueSetMessage(self.props["offset"])

				self.alg_ra.proc(err.real, err.imag, t)

				if self.alg_dec is not None:
					self.alg_dec.proc(err.imag, err.real, t)
					status += " err:%.1f %.1f corr:%.1f %.1f t_d:%.1f t_p:%.1f" % (err.real, err.imag, self.alg_ra.corr, self.alg_dec.corr, self.status['t_delay'], t_proc)
				else:
					status += " err:%.1f %.1f corr:%.1f t_d:%.1f t_p:%.1f" % (err.real, err.imag, self.alg_ra.corr, self.status['t_delay'], t_proc)
				
				ok = (np.abs(err.real) < max(0.5, np.abs(self.status['t_delay'] * self.status['pixpersec'])))
				if self.parity != 0:
					ok = (ok and np.abs(err.imag) < np.abs(self.status['t_delay'] * self.status['pixpersec_dec']))
				
				if ok or self.status['seq'] == 'seq-free':
					if self.countdown > 0:
						self.countdown -= 1
				else:
					if self.status['seq'] == 'seq-guided':
						self.countdown = 4
				ready = (self.countdown == 0)
				
				if not self.capture_init and self.capture_proc_in_progress == 0 and ready and self.status['seq'] != 'seq-stop':
					cmdQueue.put('capture')
					self.capture_init = True
					self.status['curr_ra_err_list'] = []
					self.status['curr_dec_err_list'] = []
					self.status['curr_hfr_list'] = []
				
				if self.capture_in_progress > 0:
					self.status['curr_ra_err_list'].append(err.real)
					self.status['curr_dec_err_list'].append(err.imag)
					self.status['curr_hfr_list'].append(get_hfr_list(im_sub, pt, sub_bg=True))
				
				log.info("capture %d %d %d", self.capture_init, self.capture_in_progress, self.capture_proc_in_progress)
				
				if ok:
					self.changePhase('close')
				
				for p in pt:
					cv2.circle(disp, (int(p[1] + 0.5), int(p[0] + 0.5)), 10, (255), 1)
				

				if i % 100 == 0:
					np.save("resp0_%d.npy" % self.t0, np.array(self.resp0))
					self.mount.go_ra.save("go_ra_%d.npy" % self.t0)
					self.mount.go_dec.save("go_dec_%d.npy" % self.t0)
					log.info("SAVED") 
				self.status['alarm_count'] = 0
			else:
				self.status['alarm_count'] += 1
				
		if len(self.pt0) > 0:
			for p in self.pt0:
				cv2.circle(disp, (int(p[1] + 0.5), int(p[0] + 0.5)), 13, (255), 1)
				cv2.circle(disp, (int(p[1] + self.off[1] + 0.5), int(p[0] + self.off[0] + 0.5)), 5, (255), 1)

		cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
		ui.imshow(self.tid, disp)
		self.prev_t = t

class Focuser:
	def __init__(self, driver, device, tid, status, dark = None, full_res = None):
		self.status = status

		self.driver = driver
		self.device = device
		driver.defineProperties("""
		<INDIDriver>
			<defSwitchVector device="{0}" name="focus_control" label="Control" group="Focuser" state="Idle" perm="rw" rule="AtMostOne">
				<defSwitch name="Full AF">Off</defSwitch>
				<defSwitch name="Fast AF">Off</defSwitch>
				<defSwitch name="Bahtinov">Off</defSwitch>
				<defSwitch name="Stop">Off</defSwitch>
			</defSwitchVector>

			<defSwitchVector device="{0}" name="focuser_phase" label="Phase" group="Focuser" state="Idle" perm="ro" rule="OneOfMany">
				<defSwitch name="wait">On</defSwitch>
				<defSwitch name="start">Off</defSwitch>
				<defSwitch name="seek">Off</defSwitch>
				<defSwitch name="fast_search_start">Off</defSwitch>
				<defSwitch name="get_hfr_start">Off</defSwitch>
				<defSwitch name="get_hfr">Off</defSwitch>
				<defSwitch name="dark">Off</defSwitch>
				<defSwitch name="fast_search">Off</defSwitch>
				<defSwitch name="search">Off</defSwitch>
				<defSwitch name="prep_record_v">Off</defSwitch>
				<defSwitch name="record_v">Off</defSwitch>
				<defSwitch name="focus_v">Off</defSwitch>
				<defSwitch name="focus_v2">Off</defSwitch>
				<defSwitch name="ba_start">Off</defSwitch>
				<defSwitch name="ba_init">Off</defSwitch>
				<defSwitch name="ba_run">Off</defSwitch>
			</defSwitchVector>


			<defNumberVector device="{0}" name="focuser_callibration" label="Focuser Callibration" group="Focuser" state="Idle" perm="ro">
				<defNumber name="hyst"  format="%2.2f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="m1" format="%2.2f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="m2"  format="%2.2f" min="0" max="0" step="0">0</defNumber>
			</defNumberVector>

			<defNumberVector device="{0}" name="focus" label="focus" group="Focuser" state="Idle" perm="ro">
				<defNumber name="HFR" format="%2.1f" min="0" max="0" step="0">0</defNumber>
				<defNumber name="Bahtinov" format="%2.1f" min="0" max="0" step="0">0</defNumber>
			</defNumberVector>

		</INDIDriver>
		""".format(device))

		self.props = driver[device]


		self.stack = Stack(ratio=0.3)
		self.tid = tid
		self.status['phase'] = 'wait'
		self.phase_wait = 0
		self.hfr = Focuser.hfr_size
		self.focus_yx = None
		self.prev_t = 0
		self.cmdtab = ['f+3', 'f+2', 'f+1', '', 'f-1', 'f-2', 'f-3']
		self.full_res = full_res
		self.bahtinov = Bahtinov()
		self.ba_int = 0.0
		self.ba_pos = 0.0
		self.ba_step = 0
		self.ba_dir = 0
		self.reset(dark)

	hfr_size = 30

	@staticmethod
	def v_param(v_curve):
		v_curve = np.array(v_curve)
		v_len = len(v_curve)
		side_len = int(v_len * 0.4)

		smooth_size = side_len // 4 * 2 + 1
		v_curve_s = smooth(v_curve, smooth_size, 'flat')
		v_curve_s = smooth(v_curve_s, smooth_size, 'flat')
		
		derived = np.gradient(v_curve_s)
		#log.info derived.tolist()
				
		i1 = np.argmin(derived)
		i2 = np.argmax(derived)
				
		m1 = derived[i1]
		m2 = derived[i2]
				
		c1 = v_curve_s[i1] - i1 * m1
		c2 = v_curve_s[i2] - i2 * m2
				
		#m1, c1 = np.polyfit(range(0, side_len), self.status['v_curve'][0:side_len], 1)
		#m2, c2 = np.polyfit(range(v_len - side_len, v_len), self.status['v_curve'][v_len - side_len: v_len], 1)
		xmin =  (c2 - c1) / (m1 - m2)
		side_len = xmin * 0.8
		smooth_size = int(side_len) // 4 * 2 + 1
		log.info("v_len %f side_len %f m1 %f c1 %f m2 %f c2 %f xmin %f", v_len, side_len, m1, c1, m2, c2, xmin)
		
		return xmin, side_len, smooth_size, c1, m1, c2, m2, v_curve_s
	
	@staticmethod
	def v_shift(v_curve1, v_curve2, smooth_size, c1, m1, c2, m2):
		v_curve1_s = smooth(np.array(v_curve1), smooth_size, 'flat')
		v_curve1_s = smooth(v_curve1_s, smooth_size, 'flat')
		v_curve2_s = smooth(np.array(v_curve2), smooth_size, 'flat')
		v_curve2_s = smooth(v_curve2_s, smooth_size, 'flat')
		derived2 = np.gradient(v_curve2_s)
		derived1 = np.gradient(v_curve1_s)

		i2 = np.argmin(derived2)
		m2 = derived2[i2]
		c2 = v_curve2_s[i2] - i2 * m2
		log.info("i2 m2 c2 %f %f %f", i2, m2, c2)

		i1 = np.argmin(derived1)
		m1 = derived1[i1]
		c1 = v_curve1_s[i1] - i1 * m1
		log.info("i1 m1 c1 %f %f %f", i1, m1, c1)
		
		
		y = v_curve2_s[i2]
		hyst2 = (y - c1) / m1 - i2

		y = v_curve1_s[i1]
		hyst1 = (y - c2) / m2 - i1
		
		hyst = (hyst2 - hyst1) / 2.0
		log.info("hyst %f %f %f", hyst, hyst1, hyst2)
		return hyst, v_curve2_s
	

	def handleNewProp(self, msg, prop):
		name = prop.getAttr('name')
		if name == 'focus_control':
			if prop['Full AF'] == True:
				self.changePhase('seek')
				prop['Full AF'].setValue(False)
			elif prop['Fast AF'] == True:
				self.changePhase('fast_search_start')
				prop['Fast AF'].setValue(False)
			elif prop['Bahtinov'] == True:
				self.changePhase('ba_start')
				prop['Bahtinov'].setValue(False)
			elif prop['Stop'] == True:
				self.changePhase('wait')
				prop['Stop'].setValue(False)
			
			prop.setAttr('state', 'Ok')


	def changePhase(self, mode):
		self.status['phase'] = mode
		self.props['focuser_phase'].enforceRule(mode, True)
		self.driver.enqueueSetMessage(self.props['focuser_phase'])

	def cmd(self, cmd):
		if cmd == 'dark':
			self.dark.add_mean(self.im)
		if cmd == 'af':
			self.changePhase('seek')
		if cmd == 'af_fast':
			self.changePhase('fast_search_start')

		if cmd == 'stop':
			self.changePhase('wait')
		if cmd == 'bahtinov':
			self.changePhase('ba_start')

	def reset(self, dark = None):
		self.changePhase('wait')
		self.stack = Stack(ratio=0.3)
		if dark is None:
			self.dark = Median(3)
		else:
			self.dark = dark

	def get_max_flux(self, im, xy, stddev):
		ret = []
		cur_hfr = None
		(h, w) = im.shape
		for p in xy:
			if p[2] < stddev * 3:
				#log.info "under 3stddev:", p[2], stddev * 3
				continue
			x = int(p[1] + 0.5)
			y = int(p[0] + 0.5)
			if (x < Focuser.hfr_size * 2):
				continue
			if (y < Focuser.hfr_size * 2):
				continue
			if (x > w - Focuser.hfr_size * 2 - 1):
				continue
			if (y > h - Focuser.hfr_size * 2 - 1):
				continue
			if cur_hfr is None:
				cur_hfr = hfr(im[y - Focuser.hfr_size : y + Focuser.hfr_size + 1, x - Focuser.hfr_size : x + Focuser.hfr_size + 1])
				if cur_hfr > Focuser.hfr_size * 0.5:
					cur_hfr = None
					continue
				ret.append(p)
			else:
				if hfr(im[y - Focuser.hfr_size : y + Focuser.hfr_size + 1, x - Focuser.hfr_size : x + Focuser.hfr_size + 1]) < cur_hfr + 1:
					ret.append(p)
		log.info("hfr %s %s", cur_hfr, ret)
				
		if len(ret) > 0:
			return ret[0][2], cur_hfr, np.array(ret)
		else:
			return 0, None, None

	def get_hfr(self, im, min_hfr = None):
		cur_hfr = 0
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
			x = int(x + 0.5)
			y = int(y + 0.5)
			if (x < Focuser.hfr_size):
				continue
			if (y < Focuser.hfr_size):
				continue
			if (x > w - Focuser.hfr_size - 1):
				continue
			if (y > h - Focuser.hfr_size - 1):
				continue
			xs, ys = sym_center(im[y  - centroid_size : y + centroid_size + 1, x - centroid_size : x + centroid_size + 1])
			log.info("sym_center %f %f", xs, ys)
			x += xs
			y += ys
			ix = int(x + 0.5)
			iy = int(y + 0.5)
			if (ix < Focuser.hfr_size):
				continue
			if (iy < Focuser.hfr_size):
				continue
			if (ix > w - Focuser.hfr_size - 1):
				continue
			if (iy > h - Focuser.hfr_size - 1):
				continue
			hf = hfr(im[iy - Focuser.hfr_size : iy + Focuser.hfr_size + 1, ix - Focuser.hfr_size : ix + Focuser.hfr_size + 1], sub_bg = True)

			if min_hfr is not None and hf < min_hfr:
				continue
			filtered.append( (y, x, v) )
			original.append( p )
			hfr_list.append( hf )

		if len(filtered) == 0:
			return Focuser.hfr_size

		filtered = np.array(filtered)
		original = np.array(original)
		M, weights = pt_transform_opt(original, filtered, pt_func = pt_translation_scale)
		

		log.info("transf M %s", M)
		filtered[:, 0:2] = np.insert(original[:, 0:2], 2, 1.0, axis=1).dot(M).A

		self.focus_yx = filtered
		log.info("hfr_list %s %s", hfr_list, weights)
		
		cur_hfr = np.average(hfr_list, weights = weights)
		d2 = (np.array(hfr_list) - cur_hfr) ** 2
		var = np.average(d2, weights = weights)
		noise = 2
		weights[(d2 > var * noise**2)] = 1.0
		cur_hfr = np.average(hfr_list, weights = weights)
		log.info("hfr_list_filt %s %s", hfr_list, weights)
		return cur_hfr


	def set_xy_from_stack(self, stack):
		im = stack.get()
		mean, self.stddev = cv2.meanStdDev(im)
#		self.max_flux, self.min_hfr, self.focus_yx = self.get_max_flux(im, stack.get_xy(), 0)
		yx = stack.get_xy()
		log.info("hfr_list_00 %s", yx)
		yx = [p for p in yx if p[2] > self.stddev * 4]
		log.info("hfr_list_00 stddev*4  %f", self.stddev * 4)

		log.info("hfr_list_01 %s", yx)
		if yx:
			max_flux = np.amax([p[2] for p in yx])
			yx = [p for p in yx if p[2] > max_flux / 3]
		log.info("hfr_list_02 %s", yx)


		self.focus_yx = get_hfr_field(im, yx, hfr_size = Focuser.hfr_size, sub_bg = True)
		log.info("hfr_list_1 %s", self.focus_yx)
		self.focus_yx = filter_hfr_list(self.focus_yx)
		log.info("hfr_list_2 %s", self.focus_yx)

	def step(self, s):
		cmdQueue.put(self.cmdtab[s + 3])

	def proc_frame(self, im, i):
		t = time.time()
		dispmode = self.props['dispmode'].getActiveSwitch()

		try:
			fps = 1.0 / (t - self.prev_t)
		except:
			fps = 0
		
		if im.ndim > 2:
			im = cv2.add(cv2.add(im[:, :, 0], im[:, :, 1]), im[:, :, 2])

		self.im = im
		
#		im = cv2.medianBlur(im, 3)

		if (self.dark.len() > 0):
			log.info(im.shape)
			log.info(self.dark.get().shape)
			im_sub = cv2.subtract(im, self.dark.get(), dtype=cv2_dtype(im.dtype))
		else:
			im_sub = im

		im_sub = subtract_bg(im_sub, 10)

		self.stack.add_simple(im_sub)
		self.stack_im = self.stack.get()

		if self.phase_wait > 0:
			self.phase_wait -= 1
		elif self.status['phase'] == 'get_hfr_start':
			self.phase_wait = 3
			self.changePhase('get_hfr')
		elif self.status['phase'] == 'get_hfr':
			self.set_xy_from_stack(self.stack)
			if self.focus_yx is not None:
				self.hfr = self.get_hfr(im_sub)
			self.changePhase('wait')
		elif self.status['phase'] == 'seek': # move near, out of focus
			self.hfr = self.get_hfr(im_sub)
			log.info("in-focus hfr %f", self.hfr)
			if self.hfr < Focuser.hfr_size / 3:
				self.changePhase('prep_record_v')
				self.phase_wait = 3
				self.step(3)
			else:
				self.focus_yx = None
				for i in range (0, 12):
					self.step(-3)
				self.changePhase('dark')
				self.phase_wait = 5
				self.max_flux = 0
				self.min_hfr = Focuser.hfr_size
				self.dark_add = self.dark.n
		elif self.status['phase'] == 'dark': # use current image as darkframes
			if self.dark_add > 0:
				self.dark_add -= 1
				self.dark.add_mean(self.im)
			else:
				mean, self.stddev = cv2.meanStdDev(self.stack_im)
				log.info("mean, stddev: %f %f", mean, self.stddev)
				for i in range (0, 9):
					self.step(3)
				self.phase_wait = 5
				self.search_steps = 0
				self.changePhase('search')
		elif self.status['phase'] == 'search': # step far, record max flux
			flux, hfr, yx = self.get_max_flux(self.stack_im, self.stack.get_xy(), self.stddev)
			if hfr is not None and (flux < self.max_flux * 0.7 or hfr > self.min_hfr * 2) or self.search_steps > 120:
				self.changePhase('prep_record_v')
				self.step(-1)
			else:
				if flux > self.max_flux:
					#self.focus_yx = yx
					self.max_flux = flux
					self.min_hfr = hfr
					self.set_xy_from_stack(self.stack)
				else:
					self.step(2)
				self.search_steps += 1
				self.hfr = self.get_hfr(im_sub)
			#self.phase_wait = 2
			log.info("max %f %f", flux, self.max_flux)
		elif self.status['phase'] == 'fast_search_start':
			self.phase_wait = 3
			self.changePhase('fast_search')
		elif self.status['phase'] == 'fast_search':
			self.set_xy_from_stack(self.stack)
			if self.focus_yx is None or len(self.focus_yx) == 0:
				self.status['phase'] = 'wait' #stop
			else:
				self.step(3)
				self.phase_wait = 1
				self.changePhase('prep_record_v')
		elif self.status['phase'] == 'prep_record_v': # record v curve
			self.hfr = self.get_hfr(im_sub)
			if self.hfr < Focuser.hfr_size / 2:
				self.status['v_curve'] = []
				self.status['v_curve2'] = []
				self.status['xmin'] = None
				self.status['side_len'] = None
				self.status['smooth_size'] = None
				self.status['c1'] = None
				self.status['m1'] = None
				self.status['c2'] = None
				self.status['m2'] = None
				self.status['v_curve_s'] = None
				self.status['v_curve2_s'] = None
				self.status['hyst'] = None
				self.status['remaining_steps'] = None

				self.hfr = self.get_hfr(im_sub, min_hfr = 5)

				self.changePhase('record_v')
			self.step(-1)
		elif self.status['phase'] == 'record_v': # record v curve
			self.hfr = self.get_hfr(im_sub)
			self.status['v_curve'].append(self.hfr)

			if len(self.status['v_curve']) == 15:
				self.status['start_hfr'] = np.median(self.status['v_curve'])
				self.status['min_hfr'] = self.status['start_hfr']
				self.status['cur_hfr'] = self.status['start_hfr']

			if len(self.status['v_curve']) > 15:
				self.status['cur_hfr'] = np.median(self.status['v_curve'][-15:])
				log.info('cur_hfr %f %f %f', self.status['cur_hfr'], self.status['min_hfr'], self.status['start_hfr'])

				if self.status['cur_hfr'] < self.status['min_hfr']:
					self.status['min_hfr'] = self.status['cur_hfr']
					self.status['side_len'] = len(self.status['v_curve'])

			if len(self.status['v_curve']) > 30:
				self.status['prev_hfr'] = np.median(self.status['v_curve'][-30:-15])

				if (self.status['cur_hfr'] > self.status['start_hfr'] or 
				    self.status['cur_hfr'] > max(8, self.status['min_hfr'] * 3) or
				    len(self.status['v_curve']) > 2 * self.status['side_len']):
					self.changePhase('focus_v')
					for i in range(0, len(self.status['v_curve']) - 16):
						start_hfr = np.median(self.status['v_curve'][i:i+15])
						if start_hfr < self.status['cur_hfr']:
							self.status['v_curve'] = self.status['v_curve'][i:]
							break
					
					log.info("v_curve %s", self.status['v_curve'][::-1])

					self.status['v_curve'] = self.status['v_curve'][::-1] # reverse

					self.status['xmin'], self.status['side_len'], self.status['smooth_size'], self.status['c1'], self.status['m1'], self.status['c2'], self.status['m2'], v_curve_s = Focuser.v_param(self.status['v_curve'])
					self.status['v_curve_s'] = v_curve_s.tolist()

					self.props["focuser_callibration"]["m1"].setValue(self.status['m1'])
					self.props["focuser_callibration"]["m2"].setValue(self.status['m2'])
					self.driver.enqueueSetMessage(self.props["focuser_callibration"])

					self.status['v_curve2'] = []
					if  self.status['side_len'] < 5:
						self.changePhase('wait')

			self.step(-1)
		elif self.status['phase'] == 'focus_v': # go back, record first part of second v curve
			self.hfr = self.get_hfr(im_sub)
			if len(self.status['v_curve2']) > self.status['side_len'] or self.hfr <= self.status['min_hfr'] and len(self.status['v_curve2']) > 4:
				self.status['hyst'], v_curve2_s = Focuser.v_shift(np.array(self.status['v_curve']), np.array(self.status['v_curve2']), self.status['smooth_size'], self.status['c1'], self.status['m1'], self.status['c2'], self.status['m2'])
				self.status['v_curve2_s'] = v_curve2_s.tolist()

				self.status['remaining_steps'] = round(self.status['xmin'] - len(self.status['v_curve2']) - self.status['hyst'])
				log.info("remaining %d", self.status['remaining_steps'])
				if self.status['remaining_steps'] < 5:
					self.changePhase('focus_v2')
					
					self.props["focuser_callibration"]["hyst"].setValue(self.status['hyst'])
					self.driver.enqueueSetMessage(self.props["focuser_callibration"])

					if self.full_res is not None:
						self.full_res['hyst'] = max(0, int(round(- self.status['hyst'])))
			self.step(1)
			self.phase_wait = 1
			self.status['v_curve2'].append(self.hfr)
		elif self.status['phase'] == 'focus_v2': # estimate maximum, go there
			self.hfr = self.get_hfr(im_sub)
			self.status['v_curve2'].append(self.hfr)
			if self.status['remaining_steps'] > 0:
				self.status['remaining_steps'] -= 1
				self.step(1)
			else:
				t = time.time()
				np.save("v_curve1_%d.npy" % t, np.array(self.status['v_curve']))
				np.save("v_curve2_%d.npy" % t, np.array(self.status['v_curve2']))
				self.changePhase('wait')
				
			log.info("hfr %f", self.hfr)

		elif self.status['phase'] == 'ba_start':
			if self.bahtinov.prepare(self.stack_im):
				self.ba_pos = self.bahtinov.result()
				if self.ba_dir == 0:
					self.step(2)
					self.phase_wait = 3
					self.changePhase('ba_init')
				else:
					self.ba_int = 0.0
					self.changePhase('ba_run')
			else:
				self.changePhase('wait')
		elif self.status['phase'] == 'ba_init':
			if self.bahtinov.update(im_sub):
				ba_pos = self.bahtinov.result()

				self.props["focus"]["Bahtinov"].setValue(ba_pos)
				self.driver.enqueueSetMessage(self.props["focus"])

				if ba_pos - self.ba_pos > 0:
					self.ba_dir = 1
				else:
					self.ba_dir = -1
				self.ba_pos = ba_pos
				self.ba_int = 0.0
				self.changePhase('ba_run')
			
		elif self.status['phase'] == 'ba_run':
			if self.bahtinov.update(im_sub):
				self.ba_pos = self.bahtinov.result()

				self.props["focus"]["Bahtinov"].setValue(self.ba_pos)
				self.driver.enqueueSetMessage(self.props["focus"])

				self.ba_int += self.ba_pos * 0.1
				if np.abs(self.ba_int) > 1.0:
					if self.ba_pos * self.ba_dir > 0:
						self.step(-1)
						self.ba_step -= 1
					else:
						self.step(1)
						self.ba_step += 1
					self.ba_int = 0.0
					log.info("ba_step %d" % self.ba_step)
			
		else:
			if self.focus_yx is not None:
				self.hfr = self.get_hfr(im_sub)
			else:
				self.changePhase('get_hfr_start')
			
		self.props["focus"]["HFR"].setValue(self.hfr)
		self.driver.enqueueSetMessage(self.props["focus"])
			
		if 'ba_' in self.status['phase']:
			status = "#%d F: %s %s ba:%.2f %.2f fps:%.1f" % (i, self.status['phase'], dispmode, self.ba_pos, self.ba_int, fps)
		else:
			status = "#%d F: %s %s hfr:%.2f fps:%.1f" % (i, self.status['phase'], dispmode, self.hfr, fps)
	

		if (dispmode == 'orig'):
			disp = normalize(im)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			if 'ba_' in self.status['phase']:
				self.bahtinov.plot(disp)
			if self.focus_yx is not None:
				for p in self.focus_yx:
					cv2.circle(disp, (int(p[1] + 0.5), int(p[0] + 0.5)), 20, (255), 1)
			ui.imshow(self.tid, disp)
		elif (dispmode == 'df-cor'):
			disp = normalize(im_sub)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			if 'ba_' in self.status['phase']:
				self.bahtinov.plot(disp)
			if self.focus_yx is not None:
				for p in self.focus_yx:
					cv2.circle(disp, (int(p[1] + 0.5), int(p[0] + 0.5)), 20, (255), 1)
			ui.imshow(self.tid, disp)
		elif (dispmode == 'normal'):
			disp = normalize(self.stack_im)
			if 'ba_' in self.status['phase']:
				self.bahtinov.plot(disp)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255), 2)
			if self.focus_yx is not None:
				for p in self.focus_yx:
					cv2.circle(disp, (int(p[1] + 0.5), int(p[0] + 0.5)), 20, (255), 1)
			ui.imshow(self.tid, disp)
		elif (dispmode.startswith('zoom-')):
			zoom = int(dispmode[len('zoom-'):])
			rect = np.array(self.stack_im.shape) // zoom
			shift = np.array(self.stack_im.shape) // 2 - rect // 2
			disp = self.stack_im[shift[0]:shift[0]+rect[0], shift[1]:shift[1]+rect[1]]
			disp = normalize(disp)
			disp = cv2.resize(disp, (self.stack_im.shape[1], self.stack_im.shape[0]))
			disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
			ui.imshow(self.tid, disp)
		else:
			disp = cv2.cvtColor(normalize(self.stack_im), cv2.COLOR_GRAY2RGB)
			cv2.putText(disp, status, (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
			if 'ba_' in self.status['phase']:
				self.bahtinov.plot(disp)
			ui.imshow(self.tid, disp)
		self.prev_t = t

class Mount:
	def __init__(self, status, polar, go_ra = None, go_dec = None):
		self.status = status
		self.polar = polar
		self.go_ra = go_ra
		self.go_dec = go_dec
		self.status.setdefault('oag', True)
		if self.status['oag']:
			self.status.setdefault('oag_pos', None)
			self.status.setdefault('main_pos', None)
			self.status.setdefault('t_dif', 120)
			if self.status['oag_pos'] is None:
				self.status['t_dif'] = 120
			self.status.setdefault('guider_roll', None)
		else:
			self.status['oag_pos'] = None
			self.status['main_pos'] = None
			self.status['t_dif'] = 120
			self.status['guider_roll'] = None

		self.status.setdefault('guider_pixscale', None)
		self.status.setdefault('guider_parity', 1)

		self.status.setdefault('arcsec_per_sec_ra_plus', 1)
		self.status.setdefault('arcsec_per_sec_ra_minus', 1)
		self.status.setdefault('arcsec_per_sec_dec_plus', 1)
		self.status.setdefault('arcsec_per_sec_dec_minus', 1)


		self.main_t = None
		self.guider_t = None
		self.main_tan = None
		self.guider_tan = None
		self.status['temp_sensor'] = {}
		self.temp_sensor = TempSensor(self.status['temp_sensor'])
		#self.ext_trigger = ExtTrigger()
		
		
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
		pixscale = 3600.0 * math.sqrt(abs(det))
		
		return ra, dec, orient, pixscale, parity

	def set_pos_tan(self, tan, t, camera):
		#ra, dec, orient = self.tan_to_euler(tan, off)
		#log.info ra, dec, orient
		#self.set_pos(ra, dec, orient, t, camera)

		if camera == 'navigator':
			self.main_tan = tan
			self.main_t = t
			
		elif camera == 'guider':
			self.guider_tan = tan
			self.guider_t = t
			ra, dec, roll, pixscale, parity = self.tan_to_euler(tan)
			self.status['guider_pixscale'] = pixscale
			self.status['guider_parity'] = parity
			self.status['guider_roll'] = roll
			
		
		if self.main_t is not None and self.guider_t is not None and (np.abs(self.main_t - self.guider_t) < self.status['t_dif'] or 
		   self.guider_t >= self.main_t and self.guider_t - self.main_t < 20) :
			self.status['t_dif'] = np.abs(self.main_t - self.guider_t)

			guider_w = self.guider_tan.get_width()
			guider_h = self.guider_tan.get_height()
			
			mra, mdec, mroll, mpixscale, mparity = self.tan_to_euler(self.main_tan)
			mq = Quaternion([mra, mdec, mroll])
			
			res = []
			for x, y in [(0, 0), (guider_w - 1, 0), (guider_w - 1, guider_h - 1), (0, guider_h - 1), (guider_w / 2.0 - 0.5, guider_h / 2.0 - 0.5)]:
				ra, dec = self.guider_tan.pixelxy2radec(x, y)
				q = Quaternion([ra, dec, mroll])
				
				sq = mq.inv() * q
				
				res.append(sq.to_euler().tolist())
			self.status['oag_pos'] = res

			main_w = self.main_tan.get_width()
			main_h = self.main_tan.get_height()
			
			gra, gdec, groll, gpixscale, gparity = self.tan_to_euler(self.guider_tan)
			gq = Quaternion([gra, gdec, groll])
			
			res = []
			for x, y in [(0, 0), (main_w - 1, 0), (main_w - 1, main_h - 1), (0, main_h - 1), (main_w / 2.0 - 0.5, main_h / 2.0 - 0.5)]:
				ra, dec = self.main_tan.pixelxy2radec(x, y)
				q = Quaternion([ra, dec, groll])
				
				sq = gq.inv() * q
				
				res.append(sq.to_euler().tolist())
			self.status['main_pos'] = res
	def get_guider_plot(self):
		if self.status['oag_pos'] is not None and self.main_tan is not None:
			res = []
			for e in self.status['oag_pos'][0:4]:
				sq = Quaternion(e)
				mra, mdec, mroll, mpixscale, mparity = self.tan_to_euler(self.main_tan)
				mq = Quaternion([mra, mdec, mroll])
				q =  mq * sq
				ra, dec, roll = q.to_euler()
				res.append((ra, dec))
			res2 = []
			for i, e in enumerate(res):
				res2.append((res[i - 1][0], res[i - 1][1], e[0], e[1]))
			return res2
		else:
			return []

	def get_main_plot(self):
		if self.status['main_pos'] is not None and self.guider_tan is not None:
			res = []
			for e in self.status['main_pos'][0:4]:
				sq = Quaternion(e)
				gra, gdec, groll, gpixscale, gparity = self.tan_to_euler(self.guider_tan)
				mq = Quaternion([gra, gdec, groll])
				q =  mq * sq
				ra, dec, roll = q.to_euler()
				res.append((ra, dec))
			res2 = []
			for i, e in enumerate(res):
				res2.append((res[i - 1][0], res[i - 1][1], e[0], e[1]))
			return res2
		else:
			return []
				
	def get_oag_pos(self):
		if self.status['oag_pos'] is not None and self.main_tan is not None:
			sq = Quaternion(self.status['oag_pos'][4])
			mra, mdec, mroll, mpixscale, mparity = self.tan_to_euler(self.main_tan)
			mq = Quaternion([mra, mdec, mroll])
			q =  mq * sq
			gra, gdec, groll = q.to_euler()
			return gra, gdec, 1.0
		elif self.main_tan is not None and self.status['oag']:
			mra, mdec, mroll, mpixscale, mparity = self.tan_to_euler(self.main_tan)
			return mra, mdec, 5.0
		else:
			zra, zdec = self.polar.zenith()
			return zra,zdec, 100
	
	
	def set_guider_calib(self, roll, parity, pixpersec_ra_plus, pixpersec_ra_minus, pixpersec_dec_plus, pixpersec_dec_minus):
		if parity != 0:
			log.info('parity %f %f', parity, self.status['guider_parity'])
			self.status['guider_parity'] = parity
		log.info('roll %f %f', 90 + roll * self.status['guider_parity'], self.status['guider_roll'])
		self.status['guider_roll'] = 90 + roll * self.status['guider_parity']
		if self.status['guider_pixscale'] is not None:
			if self.guider_tan is not None and time.time() - self.guider_t < 60:
				gra, gdec, groll, gpixscale, gparity = self.tan_to_euler(self.guider_tan)
			elif self.status['oag_pos'] is not None and self.main_tan is not None:
				sq = Quaternion(self.status['oag_pos'][4])
				mra, mdec, mroll, mpixscale, mparity = self.tan_to_euler(self.main_tan)
				mq = Quaternion([mra, mdec, mroll])
				q =  mq * sq
				gra, gdec, groll = q.to_euler()
			elif self.main_tan is not None:
				gra, gdec, groll, gpixscale, gparity = self.tan_to_euler(self.main_tan)
			else:
				return
			gdec = np.deg2rad(gdec)
		
			self.status['arcsec_per_sec_ra_plus'] = pixpersec_ra_plus * self.status['guider_pixscale'] / np.max([np.cos(gdec), 0.2])
			self.status['arcsec_per_sec_ra_minus'] = pixpersec_ra_minus * self.status['guider_pixscale'] / np.max([np.cos(gdec), 0.2])
			if pixpersec_dec_plus is not None:
				self.status['arcsec_per_sec_dec_plus'] = pixpersec_dec_plus * self.status['guider_pixscale']
			if pixpersec_dec_minus is not None:
				self.status['arcsec_per_sec_dec_minus'] = pixpersec_dec_minus * self.status['guider_pixscale']

	def get_guider_calib(self):
		pass
		# return roll, parity, pixpersec_ra_plus, pixpersec_ra_minus, pixpersec_dec_plus, pixpersec_dec_minus

	def move_main_px(self, dx, dy, camera):
		if camera == 'navigator':
			if self.main_tan is None:
				return
			log.info("move pix %f %f", dx, dy)

			mra, mdec, mroll, mpixscale, mparity = self.tan_to_euler(self.main_tan)
			
			mroll = np.deg2rad(mroll)
			ra = (np.cos(mroll) * dx - np.sin(mroll) * dy) * mpixscale / np.max([np.cos(np.deg2rad(mdec)), 0.2])
			dec =(np.sin(mroll) * dx + np.cos(mroll) * dy) * mpixscale
                        
			dec *= mparity
                        
			log.info("move arcsec %f %f", ra, dec)
			if self.go_ra is not None:
				if ra > 0:
					log.info("move_ra sec %f", ra / self.status['arcsec_per_sec_ra_plus'])
					self.go_ra.out(1, ra / self.status['arcsec_per_sec_ra_plus'])
				elif ra < 0:
					log.info("move_ra sec %f", ra / self.status['arcsec_per_sec_ra_minus'])
					self.go_ra.out(-1, -ra / self.status['arcsec_per_sec_ra_minus'])
				else:
					self.go_ra.out(0)

			if self.go_dec is not None:
				if dec > 0:
					log.info("move_dec sec %f", dec / self.status['arcsec_per_sec_dec_plus'])
					self.go_dec.out(-1, dec / self.status['arcsec_per_sec_dec_plus'])
				elif dec < 0:
					log.info("move_dec sec %f", dec / self.status['arcsec_per_sec_dec_minus'])
					self.go_dec.out(1, -dec / self.status['arcsec_per_sec_dec_minus'])
				else:
					self.go_dec.out(0)

	def stop(self):
		if self.go_dec is not None:
			self.go_dec.out(0)
		if self.go_ra is not None:
			self.go_ra.out(0)

	def move_to(self, ra, dec):
		pass





class Runner(threading.Thread):
	def __init__(self, driver, device, tid, camera, navigator = None, guider = None, focuser = None, video_tid = None):
		threading.Thread.__init__(self)


		self.driver = driver
		self.device = device
		driver.defineProperties("""
		<INDIDriver>
			<defSwitchVector device="{0}" name="CONNECTION" label="Camera Connection" group="Main Control" state="Idle" perm="rw" rule="OneOfMany">
				<defSwitch name="CONNECT">Off</defSwitch>
				<defSwitch name="DISCONNECT">On</defSwitch>
			</defSwitchVector>

			<defSwitchVector device="{0}" name="dispmode" label="Display mode" group="Main Control" state="Idle" perm="rw" rule="OneOfMany">
				<defSwitch name="normal" label="normal">On</defSwitch>
				<defSwitch name="zoom-2" label="zoom-2">Off</defSwitch>
				<defSwitch name="zoom-3" label="zoom-3">Off</defSwitch>
				<defSwitch name="zoom-4" label="zoom-4">Off</defSwitch>
				<defSwitch name="zoom-8" label="zoom-8">Off</defSwitch>
				<defSwitch name="zoom-16" label="zoom16">Off</defSwitch>
				<defSwitch name="zoom-deg50" label="zoom-deg50">Off</defSwitch>
				<defSwitch name="zoom-deg100" label="zoom-deg100">Off</defSwitch>
				<defSwitch name="zoom-deg180" label="zoom-deg180">Off</defSwitch>
				<defSwitch name="orig" label="orig">Off</defSwitch>
				<defSwitch name="df-cor" label="df-cor">Off</defSwitch>
				<defSwitch name="match" label="match">Off</defSwitch>
			</defSwitchVector>

			<defSwitchVector device="{0}" name="full_dispmode" label="FullRes Display mode" group="FullRes" state="Idle" perm="rw" rule="OneOfMany">
				<defSwitch name="normal" label="normal">On</defSwitch>
				<defSwitch name="zoom-2" label="zoom-2">Off</defSwitch>
				<defSwitch name="zoom-3" label="zoom-3">Off</defSwitch>
				<defSwitch name="zoom-4" label="zoom-4">Off</defSwitch>
				<defSwitch name="zoom-8" label="zoom-8">Off</defSwitch>
				<defSwitch name="zoom-16" label="zoom16">Off</defSwitch>
				<defSwitch name="zoom-deg50" label="zoom-deg50">Off</defSwitch>
				<defSwitch name="zoom-deg100" label="zoom-deg100">Off</defSwitch>
				<defSwitch name="zoom-deg180" label="zoom-deg180">Off</defSwitch>
				<defSwitch name="orig" label="orig">Off</defSwitch>
				<defSwitch name="df-cor" label="df-cor">Off</defSwitch>
				<defSwitch name="match" label="match">Off</defSwitch>
			</defSwitchVector>

			<defSwitchVector device="{0}" name="camera_control" label="Camera" group="Main Control" state="Idle" perm="rw" rule="AtMostOne">
				<defSwitch name="capture">Off</defSwitch>
				<defSwitch name="test_capture">Off</defSwitch>
			</defSwitchVector>

			<defNumberVector device="{0}" name="EXPOSURE" label="Expose" group="Main Control" state="Idle" perm="rw" timeout="60">
				<defNumber name="EXP_TIME" label="Duration (s)" format="%5.3f" min="0.001" max="3600" step="1">1</defNumber>
				<defNumber name="TEST_EXP_TIME" label="Test Duration (s)" format="%5.3f" min="0.001" max="3600" step="1">1</defNumber>
			</defNumberVector>

			<defSwitchVector device="{0}" name="run_control" label="Control" group="Main Control" state="Idle" perm="rw" rule="AtMostOne">
				<defSwitch name="exit">Off</defSwitch>
				<defSwitch name="shutdown">Off</defSwitch>
				<defSwitch name="save">Off</defSwitch>
				<defSwitch name="stop">Off</defSwitch>
			</defSwitchVector>

		</INDIDriver>
		""".format(device))
		
		prop_str = """
		<INDIDriver>
			<defSwitchVector device="{0}" name="run_mode" label="Mode" group="Main Control" state="Idle" perm="rw" rule="AtMostOne">
		"""
		if guider:
			prop_str += '<defSwitch name="guider">Off</defSwitch>'
		if focuser:
			prop_str += '<defSwitch name="focuser">Off</defSwitch>'
			prop_str += '<defSwitch name="zoom_focuser">Off</defSwitch>'
		
		prop_str += """
			</defSwitchVector>
		</INDIDriver>
		"""
		
		driver.defineProperties(prop_str.format(device))
		
		if focuser:
			driver.defineProperties("""
			<INDIDriver>
				<defSwitchVector device="{0}" name="focus_plus" label="Focus" group="Main Control" state="Idle" perm="rw" rule="AtMostOne">
					<defSwitch name="f+3">Off</defSwitch>
					<defSwitch name="f+2">Off</defSwitch>
					<defSwitch name="f+1">Off</defSwitch>
				</defSwitchVector>

				<defSwitchVector device="{0}" name="focus_minus" label="Focus" group="Main Control" state="Idle" perm="rw" rule="AtMostOne">
					<defSwitch name="f-3">Off</defSwitch>
					<defSwitch name="f-2">Off</defSwitch>
					<defSwitch name="f-1">Off</defSwitch>
				</defSwitchVector>

				<defNumberVector device="{0}" name="focus_pos" label="Focus" group="Main Control" state="Idle" perm="ro" timeout="60">
					<defNumber name="pos" label="Pos" format="%5.0f">0</defNumber>
				</defNumberVector>
			</INDIDriver>
			""".format(device))

		
		self.props = driver[device]
		driver.register(device)

		self.tid = tid
		self.camera = camera
		self.navigator = navigator
		self.guider = guider
		self.focuser = focuser
		self.capture_in_progress = False
		self.video_tid = video_tid
		self.video_capture = False
		self.camera_run = False
		
		
	def run(self):
#		profiler = LineProfiler()
#		profiler.add_function(Navigator.proc_frame)
#		profiler.add_function(Guider.proc_frame)
#		profiler.add_function(Stack.add)
#		profiler.add_function(Median.add)
#		profiler.add_function(Median.add_masked)
#		profiler.add_function(find_max)
#		profiler.add_function(match_triangle)
#		profiler.add_function(Runner.run)
#		profiler.add_function(Camera_test.capture)
#		profiler.add_function(Polar.solve)
#		#profiler.add_function(Polar.camera_position)
		
#		profiler.enable_by_count()
		
		
		cmdQueue.register(self.tid)
		
		i = 0
		if self.navigator is not None:
			mode = 'navigator'
		else:
			mode = 'guider'

		process = psutil.Process(os.getpid())
		
		

		
		while True:
		
			mem_info = process.memory_info()
			log.info("mem_used %d %d" % (mem_info.rss,mem_info.vms) )

			while True:
				msg, prop = self.driver.get(self.device, block=False)
				if prop is None:
					break
				
				prop.newFromEtree(msg)

				name = prop.getAttr('name')

				if name == 'CONNECTION':
					if prop['DISCONNECT'] == True:
						if self.camera_run:
							try:
								self.camera.shutdown()
							except:
								log.exception("camera shutdown")
						self.camera_run = False
						prop.setAttr('state', 'Ok')
					elif prop['CONNECT'] == True:
						if not self.camera_run:
							try:
								self.camera.prepare()
							except:
								log.exception("camera shutdown")
						self.camera_run = True
						prop.setAttr('state', 'Ok')

				elif name == 'dispmode':
					prop.setAttr('state', 'Ok')

				elif name == 'full_dispmode':
					prop.setAttr('state', 'Ok')
			
				elif name == 'run_mode':
					if self.guider and prop['guider'] == True:
						prop.setAttr('state', 'Ok')
						if mode == 'zoom_focuser':
							self.camera.cmd('z0')
						self.guider.reset()
						self.guider.pt0 = self.navigator.get_xy_cor()
						mode = 'guider'

					elif self.focuser and prop['zoom_focuser'] == True:
						prop.setAttr('state', 'Ok')
						self.focuser.reset()
						self.camera.cmd('z1')
						mode = 'zoom_focuser'
					elif self.focuser and prop['focuser'] == True:
						prop.setAttr('state', 'Ok')
						if mode == 'zoom_focuser':
							self.camera.cmd('z0')
						self.focuser.reset(dark = self.navigator.dark)
						self.camera.cmd('z0')
						mode = 'focuser'
					else:
						if self.guider is not None:
							self.guider.cmd('stop')

						prop.setAttr('state', 'Ok')
						if mode == 'zoom_focuser':
							self.camera.cmd('z0')
						mode = 'navigator'

				elif name == 'camera_control':
				
					if prop['capture'] == True or prop['test_capture'] == True:
						if self.camera_run:
							if mode == 'zoom_focuser':
								self.camera.cmd('z0')
								mode = 'navigator'
								self.props['run_mode'].enforceRule(mode, True)
								self.driver.enqueueSetMessage(self.props['run_mode'])

							try:
								self.camera.capture_bulb(test=(prop['test_capture'] == True), callback_start = self.capture_start_cb, callback_end = self.capture_end_cb)
							except AttributeError:
								pass
							except:
								log.exception('Unexpected error')					
							if self.capture_in_progress:
								log.info("runner: capture_in_progress not finished")
								log.info("capture_finished_fix")
	
								cmdQueue.put('capture-finished')
								cmdQueue.put('capture-full-res-done')
								self.capture_in_progress = False
						
							prop['capture'].setValue(False)
							prop['test_capture'].setValue(False)
							prop.setAttr('state', 'Ok')
						else:
							prop['capture'].setValue(False)
							prop['test_capture'].setValue(False)
							prop.setAttr('state', 'Failed')
							self.driver.message("Camera is paused", self.device)
				elif name == 'EXPOSURE':
					self.camera.cmd('exp-sec-' + str(prop['EXP_TIME']))
					self.camera.cmd('test-exp-sec-' + str(prop['TEST_EXP_TIME']))
					prop.setAttr('state', 'Ok')
				elif name == 'focus_plus' or name == 'focus_minus':
					cmd = prop.getActiveSwitch()
					cmdQueue.put(cmd)
					prop.setAttr('state', 'Ok')
					prop[cmd].setValue(False)

				else:
					if self.navigator:
						self.navigator.handleNewProp(msg, prop)
					if self.guider:
						self.guider.handleNewProp(msg, prop)
					if self.focuser:
						self.focuser.handleNewProp(msg, prop)
		
				self.driver.enqueueSetMessage(prop)

			while True:
				cmd=cmdQueue.get(self.tid, 0.0001)
				if cmd is None:
					break
				if cmd == 'exit' or cmd == 'shutdown':
					if self.guider is not None:
						self.guider.cmd('stop')
					#profiler.print_stats()
					try:
						self.camera.shutdown()
					except:
						pass
						

					return
				elif cmd == 'navigator' and self.navigator is not None:
					if self.guider is not None:
						self.guider.cmd('stop')
					if mode == 'zoom_focuser':
						self.camera.cmd('z0')
					mode = 'navigator'
					self.props['run_mode'].enforceRule(mode, True)
					self.driver.enqueueSetMessage(self.props['run_mode'])

				elif cmd == 'guider' and self.guider is not None:
					if mode == 'zoom_focuser':
						self.camera.cmd('z0')
					self.guider.reset()
					self.guider.pt0 = self.navigator.get_xy_cor()
					mode = 'guider'
					self.props['run_mode'].enforceRule(mode, True)
					self.driver.enqueueSetMessage(self.props['run_mode'])
				elif cmd == 'z1':
					if self.focuser is not None:
						self.focuser.reset()
						self.camera.cmd(cmd)
						mode = 'zoom_focuser'
				elif cmd == 'z0':
					if mode == 'zoom_focuser':
						self.focuser.reset(dark = self.navigator.dark)
						self.camera.cmd(cmd)
						mode = 'navigator'
					elif mode == 'focuser':
						mode = 'navigator'
					self.props['run_mode'].enforceRule(mode, True)
					self.driver.enqueueSetMessage(self.props['run_mode'])
				elif cmd == 'zcenter':
					if self.focuser is not None:
						self.camera.cmd(cmd)
				elif cmd == 'zpos':
					if self.focuser is not None:
						pts = []
						try:
							
							if mode == 'navigator':
								pts = self.navigator.stack.get_xy()
							elif mode == 'focuser':
								pts = self.focuser.stack.get_xy()
							log.info("%s", pts)
							if len(pts) > 0:
								i = random.randint(0,len(pts) - 1)
								(maxy, maxx, maxv) = pts[i]
								self.camera.cmd(cmd, maxx, maxy)
							else:
								self.camera.cmd('zcenter')
						except:
							pass
							
				elif (cmd == 'af' or cmd == 'af_fast') and mode != 'zoom_focuser' and self.focuser is not None:
					#if mode == 'navigator':
					#	self.focuser.set_xy_from_stack(self.navigator.stack)
					mode = 'focuser'
					self.props['run_mode'].enforceRule(mode, True)
					self.driver.enqueueSetMessage(self.props['run_mode'])
					self.focuser.cmd(cmd)
				elif cmd == 'dark':
					if mode == 'navigator':
						self.navigator.cmd(cmd)
					elif mode == 'guider':
						self.guider.cmd(cmd)
					elif mode == 'focuser':
						self.focuser.cmd(cmd)
				elif cmd == 'capture' or cmd == 'test-capture':
					if mode == 'zoom_focuser':
						self.camera.cmd('z0')
						mode = 'navigator'
						self.props['run_mode'].enforceRule(mode, True)
						self.driver.enqueueSetMessage(self.props['run_mode'])
					try:
						self.camera.capture_bulb(test=(cmd == 'test-capture'), callback_start = self.capture_start_cb, callback_end = self.capture_end_cb)
					except AttributeError:
						pass
					except:
						log.exception('Unexpected error')					
					if self.capture_in_progress:
						log.info("runner: capture_in_progress not finished")
						log.info("capture_finished_fix")

						cmdQueue.put('capture-finished')
						cmdQueue.put('capture-full-res-done')
						self.capture_in_progress = False
					break
				elif cmd == 'capture_start':
					self.camera.cmd(cmd)
					self.video_capture = True
				elif cmd == 'capture_stop':
					self.camera.cmd(cmd)
					self.video_capture = False
				else:
					self.camera.cmd(cmd)
					
					if self.navigator is not None:
						self.navigator.cmd(cmd)
					if self.guider is not None:
						self.guider.cmd(cmd)
			
					if mode == 'focuser':
						self.focuser.cmd(cmd)
			
			try:
				if self.props["focus_pos"]["pos"] != self.camera.focuser.pos:
					self.props["focus_pos"]["pos"].setValue(self.camera.focuser.pos)
					self.props["focus_pos"].setAttr('state', 'Ok')
					self.driver.enqueueSetMessage(self.props["focus_pos"])
			except:
				pass
			
			if not self.camera_run:
				time.sleep(1)
				continue
			
			im, t = self.camera.capture()
			
			if im is None:
				continue
			
			log.info("%d %f", i, t)
			if self.video_tid is not None:
				log.info("%s %s", im.shape, im.dtype)
				
				show = im
				max_v = np.iinfo(im.dtype).max
				if max_v > 255:
					show = np.array(show / ((max_v + 1) / 256), dtype = np.uint8)
				log.info("%s %s", show.shape, show.dtype)
				ui.imshow(self.video_tid, show)
			elif self.video_capture:
				time.sleep(5)
				
			if not self.video_capture:
				try:
					#cv2.imwrite("testimg23_" + str(i) + ".tif", im)
					if mode == 'navigator':
						self.navigator.proc_frame(im, i, t)
					if mode == 'guider':
						self.guider.proc_frame(im, i)
					if mode == 'focuser' or mode == 'zoom_focuser':
						self.focuser.proc_frame(im, i)
				except:
					log.exception('proc_frame')

			i += 1
			#if i == 300:
			#	cmdQueue.put('exit')
		cmdQueue.put('exit')
		self.camera.shutdown()
	
	def capture_start_cb(self):
		cmdQueue.put('capture-started')
		log.info("gc.collect %d" % gc.collect())
		self.capture_in_progress = True
		if self.navigator.full_res_solver is not None:
                                self.navigator.full_res_solver.terminate(wait=False)
	
	def capture_end_cb(self, img, name):
		self.capture_in_progress = False
		log.info("capture_finished_cb")
		cmdQueue.put('capture-finished')
		if img is not None:
			if name.endswith('.jpg'):
				ui.imshow_jpg("full_res", img)
			#self.navigator.proc_full_res(jpg, name)
			threading.Thread(target=self.navigator.proc_full_res, args = [img, name] ).start()
		else:
			cmdQueue.put('capture-full-res-done')

def main_loop():
	global status

	cmdQueue.register('main')
	while True:
		driver.loop1()
		cmd=cmdQueue.get('main', timeout=0.01)
		if cmd is None:
			continue
		
		if cmd == 'exit' or cmd == 'shutdown':
			stacktraces()

			if cmd == 'shutdown':
				subprocess.call(['shutdown', '-h', "now"])

			t0 = time.time()
			while time.time() < t0 + 5:
				driver.loop1()
			
			log.info("INDI loop shutdown")
			break
		if cmd == 'save':
			status.save()
		
		if cmd == 'interrupt':
			camera = status.path(["navigator", "camera"])
			camera['interrupt'] = True
		
		if cmd == 'sync':
			try:
				from indi_python.basic_indi_client import BasicIndiClient
				import indi_python.indi_xml as indiXML
				client = BasicIndiClient("localhost", 7624, 0.01)
				client.sendMessage(indiXML.newSwitchVector([indiXML.oneSwitch("On", indi_attr = {"name" : "SYNC"})],
					indi_attr = {"name" : "ON_COORD_SET", "device" : "EQMod Mount"}))
				ra = status.path(["navigator"]).get('ra', 0.0) / 15.0
				dec = status.path(["navigator"]).get('dec', 0.0)
				client.sendMessage(indiXML.newNumberVector([indiXML.oneNumber(ra, indi_attr = {"name" : "RA"}), indiXML.oneNumber(dec, indi_attr = {"name" : "DEC"})],
					indi_attr = {"name" : "EQUATORIAL_EOD_COORD", "device" : "EQMod Mount"}))
				client.sendMessage(indiXML.newSwitchVector([indiXML.oneSwitch("On", indi_attr = {"name" : "TRACK"})],
					indi_attr = {"name" : "ON_COORD_SET", "device" : "EQMod Mount"}))

				client.close()
			except:
				log.exception('sync')


	status.save()
	


class Camera_test:
	def __init__(self, status):
		self.status = status
		self.i = 0
		self.step = 1
		self.x = 0
		self.y = 0
		self.status['exp-sec'] = 60
	
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
		if cmd.startswith('exp-sec-'):
			self.status['exp-sec'] = float(cmd[len('exp-sec-'):])

	def capture(self):
		#time.sleep(2)
		log.info("%d", self.i)
		#pil_image = Image.open("converted/IMG_%04d.jpg" % (146+self.i))
		#pil_image.thumbnail((1000,1000), Image.ANTIALIAS)
		#im = np.array(pil_image)
		im = cv2.imread("test/testimg23_" + str(self.i % 100 * 3 + int(self.i / 100) * 10) + ".tif")
		#im = cv2.imread("testimg23_" + str(self.i) + ".tif")
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
		self.status = status
	
	def cmd(self, cmd):
		pass
	
	def capture(self):
		i =  self.cam0.i + self.shift
		im = cv2.imread("test/testimg23_" + str(i) + ".tif")
		return im, None

	def shutdown(self):
		pass


class Camera_test_g:
	def __init__(self, status, go_ra, go_dec):
		self.status = status
		self.i = 0
		self.err = 0.0
		self.go_ra = go_ra
		self.go_dec = go_dec
		self.status['exp-sec'] = 0.5
	
	def cmd(self, cmd):
		log.info("camera: %s", cmd)
		if cmd.startswith('exp-sec-'):
			self.status['exp-sec'] = float(cmd[len('exp-sec-'):])

	
	def capture(self):
		time.sleep(self.status['exp-sec'])
		self.err += random.random() * 2 - 1.5
		corr = self.go_ra.recent_avg() * 5
		i = int((corr - self.go_ra.recent_avg(1))  + self.err)
		log.info("%f %f", self.err, corr * 3, i)
		im = cv2.imread("test/testimg23_" + str(i + 100) + ".tif")
		log.info("test/testimg23_" + str(i + 100) + ".tif")
		corr_dec = self.go_dec.recent_avg()
		im = im[50 + int(corr_dec * 3):-50 + int(corr_dec * 3)]
		#im = cv2.flip(im, 1)
		return im, None

	def shutdown(self):
		pass

def run_v4l2():
	global status
	status = Status("run_v4l2.conf")
	ui.namedWindow('navigator')
	ui.namedWindow('polar')

	polar = Polar(status.path(["polar"]), ['navigator'])
	mount = Mount(status.path(["mount"]), polar)
	cam = Camera(status.path(["navigator", "camera"]))
	cam.prepare(1280, 960)
	dark = Median(5)
	nav = Navigator(status.path(["navigator"]), dark, mount, 'navigator', polar_tid = 'polar')

	runner = Runner('navigator', cam, navigator = nav)
	runner.start()
	
	main_loop()
	runner.join()

def run_gphoto():
	global status
	status = Status("run_gphoto.conf")
	fo = FocuserOut()
	cam = Camera_gphoto(status.path(["navigator", "camera"]), fo)
	cam.prepare()
	ui.namedWindow('navigator')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

	polar = Polar(status.path(["polar"]), ['navigator', 'full-res'])
	mount = Mount(status.path(["mount"]), polar)

	dark = Median(5)
	nav = Navigator(status.path(["navigator"]), dark, mount, 'navigator', polar_tid = 'polar')
	focuser = Focuser('navigator', status.path(["navigator", "focuser"]), dark = dark)

	runner = Runner('navigator', cam, navigator = nav, focuser = focuser)
	runner.start()

	main_loop()
	runner.join()


def run_v4l2_g():
	global status
	status = Status("run_v4l2_g.conf")
	cam = Camera(status.path(["guider", "navigator", "camera"]))
	cam.prepare(1280, 960)

	ui.namedWindow('guider')
	ui.namedWindow('polar')

	polar = Polar(status.path(["polar"]), ['guider'])
        
	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")
	
	mount = Mount(status.path(["mount"]), polar, go_ra, go_dec)


	dark = Median(5)
	nav = Navigator(status.path(["guider", "navigator"]), dark, mount, 'guider', polar_tid = 'polar')
	guider = Guider(status.path(["guider"]), mount, dark, 'guider')

	runner = Runner('guider', cam, navigator = nav, guider = guider)
	runner.start()

	main_loop()
	runner.join()

def run_gphoto_g():
	global status
	status = Status("run_gphoto_g.conf")
	ui.namedWindow('navigator')
	ui.namedWindow('polar')

	polar = Polar(status.path(["polar"]), ['navigator', 'full-res'])
	
	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")

	mount = Mount(status.path(["mount"]), polar, go_ra, go_dec)

	cam = Camera_gphoto(status.path(["guider", "navigator", "camera"]))
	cam.prepare()

	dark = Median(5)
	nav = Navigator(status.path(["guider", "navigator"]), dark, mount, 'navigator', polar_tid = 'polar')
	
	guider = Guider(status.path(["guider"]), mount, dark, 'navigator')

	runner = Runner('navigator', cam, navigator = nav, guider = guider)
	runner.start()
	main_loop()
	runner.join()

def run_test_g():
	global status
	status = Status("run_test_g.conf")
	ui.namedWindow('guider')
	ui.namedWindow('polar')

	polar = Polar(status.path(["polar"]), ['guider'])
	
	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")
        
	mount = Mount(status.path(["mount"]), polar, go_ra, go_dec)

	dark = Median(5)
	nav = Navigator(status.path(["guider", "navigator"]), dark, mount, 'guider', polar_tid = 'polar')
	guider = Guider(status.path(["guider"]), mount, dark, 'guider')
	cam = Camera_test_g(status.path(["guider", "navigator", "camera"]), go_ra, go_dec)

	runner = Runner('guider', cam, navigator = nav, guider = guider)
	runner.start()
	main_loop()
	runner.join()

def run_test():
	global status
	status = Status("run_test.conf")
	ui.namedWindow('navigator')
	ui.namedWindow('polar')

	polar = Polar(status.path(["polar"]), ['navigator'])
	mount = Mount(status.path(["mount"]), polar)

	cam = Camera_test(status.path(["navigator", "camera"]))
	dark = Median(5)
	nav = Navigator(status.path(["navigator"]), dark, mount, 'navigator', polar_tid = 'polar')

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

	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")

	mount = Mount(status.path(["mount"]), polar, go_ra, go_dec)

	dark1 = Median(5)
	dark2 = Median(5)

	cam1 = Camera_test(status.path(["navigator", "camera"]))
	nav1 = Navigator(status.path(["navigator"]), dark1, mount, 'navigator', polar_tid = 'polar')

	nav = Navigator(status.path(["guider", "navigator"]), dark2, mount, 'guider')

	guider = Guider(status.path(["guider"]), mount, dark2, 'guider')
	#cam = Camera_test(status.path(["guider", "navigator", "camera"]))
	cam = Camera_test_g(status.path(["guider", "navigator", "camera"]), go_ra, go_dec)
	
	runner = Runner('navigator', cam1, navigator = nav1)
	runner.start()
	
	runner2 = Runner('guider', cam, navigator = nav, guider = guider)
	runner2.start()
	
	main_loop()
	
	runner.join()
	runner2.join()


def run_test_2_kstars():
	from kstars_camera import Camera_test_kstars, Camera_test_kstars_g
	global status
	status = Status("run_test_2.conf")
 
	global driver
	driver = IndiDriver()
 

	ui.namedWindow('navigator')
	ui.namedWindow('guider')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

	polar = Polar(status.path(["polar"]), ['navigator', 'guider', 'full-res'])

	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")

	mount = Mount(status.path(["mount"]), polar, go_ra, go_dec)

	dark1 = Median(5)
	dark2 = Median(5)

	fo = FocuserOut()
	cam1 = Camera_test_kstars(status.path(["navigator", "camera"]), go_ra, go_dec, fo)
	nav1 = Navigator(driver, "Navigator", status.path(["navigator"]), dark1, mount, 'navigator', polar_tid = 'polar', full_res = status.path(["full_res"]))

	nav = Navigator(driver, "Guider", status.path(["guider", "navigator"]), dark2, mount, 'guider')

	guider = Guider(driver, "Guider", status.path(["guider"]), mount, dark2, 'guider', full_res = status.path(["full_res"]))
	#cam = Camera_test(status.path(["guider", "navigator", "camera"]))
	cam = Camera_test_kstars_g(status.path(["guider", "navigator", "camera"]), cam1)


	focuser = Focuser(driver, "Navigator", 'navigator', status.path(["navigator", "focuser"]), dark = dark1, full_res = status.path(["full_res"]))
	
	runner = Runner(driver, "Navigator", 'navigator', cam1, navigator = nav1, focuser = focuser)
	runner.start()
	
	runner2 = Runner(driver, "Guider", 'guider', cam, navigator = nav, guider = guider)
	runner2.start()
	
	main_loop()
	
	runner.join()
	runner2.join()

def run_indi():
	global status
	status = Status("run_indi.conf")

	global driver
	driver = IndiDriver()

	ui.namedWindow('navigator')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

	polar = Polar(status.path(["polar"]), ['navigator'])
	mount = Mount(status.path(["mount"]), polar)
	fo = FocuserOut()

	#cam = Camera_indi(driver, "CCD Simulator", status.path(["navigator", "camera"]), focuser=fo)
	#cam = Camera_indi(driver, "V4L2 CCD", status.path(["navigator", "camera"]), focuser=fo)
	cam = Camera_indi(driver, "ZWO CCD ASI1600MM Pro", status.path(["navigator", "camera"]), focuser=fo)
	dark = Median(5)
	
	nav = Navigator(driver, "Navigator", status.path(["navigator"]), dark, mount, 'navigator', polar_tid = 'polar')

	focuser = Focuser(driver, "Navigator", 'navigator', status.path(["navigator", "focuser"]))

	runner = Runner(driver, "Navigator", 'navigator', cam, navigator = nav, focuser = focuser)
	runner.start()
	main_loop()
	runner.join()

def run_test_2_gphoto():
	global status
	status = Status("run_test_2_gphoto.conf")
	
	cam1 = Camera_gphoto(status.path(["navigator", "camera"]))
	cam1.prepare()

	polar = Polar(status.path(["polar"]), ['navigator', 'guider', 'full-res'])

	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")

	mount = Mount(status.path(["mount"]), polar, go_ra, go_dec)

	dark1 = Median(5)
	dark2 = Median(5)
	
	nav1 = Navigator(status.path(["navigator"]), dark1, mount, 'navigator', polar_tid = 'polar', full_res = status.path(["full_res"]))
	focuser = Focuser('navigator', status.path(["navigator", "focuser"]), dark = dark1)

	nav = Navigator(status.path(["guider", "navigator"]), dark2, mount, 'guider')

	guider = Guider(status.path(["guider"]), mount, dark2, 'guider', full_res = status.path(["full_res"]))
	cam = Camera_test_g(status.path(["guider", "navigator", "camera"]), go)

	ui.namedWindow('navigator')
	ui.namedWindow('guider')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

	go.out(1, 10) # move aside for 10s to collect darkframes

	runner = Runner('navigator', cam1, navigator = nav1, focuser=focuser)
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
	
	fo = FocuserOut()
	cam1 = Camera_gphoto(status.path(["navigator", "camera"]), fo)
	cam1.prepare()

	polar = Polar(status.path(["polar"]), ['navigator', 'guider', 'full-res'])

	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")

	mount = Mount(status.path(["mount"]), polar, go_ra, go_dec)

	dark1 = Median(5)
	dark2 = Median(5)
	
	nav1 = Navigator(status.path(["navigator"]), dark1, mount, 'navigator', polar_tid = 'polar', full_res = status.path(["full_res"]))
	focuser = Focuser('navigator', status.path(["navigator", "focuser"]), dark = dark1, full_res = status.path(["full_res"]))

	nav = Navigator(status.path(["guider", "navigator"]), dark2, mount, 'guider')

	guider = Guider(status.path(["guider"]), mount, dark2, 'guider', full_res = status.path(["full_res"]))

	ui.namedWindow('navigator')
	ui.namedWindow('guider')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

	
	go_ra.out(1, 10) # move aside for 10s to collect darkframes

	runner = Runner('navigator', cam1, navigator = nav1, focuser=focuser)
	runner.start()
	
	runner2 = Runner('guider', cam, navigator = nav, guider = guider)
	runner2.start()
	
	main_loop()
	
	runner.join()
	runner2.join()

def run_2_indi():
	global status
	status = Status("run_2_indi.conf")

	global driver
	driver = IndiDriver()

	ui.namedWindow('navigator')
	ui.namedWindow('guider')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

	polar = Polar(status.path(["polar"]), ['navigator', 'guider', 'full-res'])

	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")

	mount = Mount(status.path(["mount"]), polar, go_ra, go_dec)

	cam2 = Camera(status.path(["guider", "navigator", "camera"]))
	cam2.prepare(1280, 960)

	fo = FocuserOut()


	cam = Camera_indi(driver, "CCD Simulator", status.path(["navigator", "camera"]), focuser=fo)
	#cam = Camera_indi(driver, "V4L2 CCD", status.path(["navigator", "camera"]), focuser=fo)
	#cam = Camera_indi(driver, "ZWO CCD ASI1600MM Pro", status.path(["navigator", "camera"]), focuser=fo)
	dark = Median(5)
	
	nav = Navigator(driver, "Navigator", status.path(["navigator"]), dark, mount, 'navigator', polar_tid = 'polar')

	focuser = Focuser(driver, "Navigator", 'navigator', status.path(["navigator", "focuser"]))

	runner = Runner(driver, "Navigator", 'navigator', cam, navigator = nav, focuser = focuser)
	runner.start()

	dark2 = Median(5)
	nav2 = Navigator(driver, "Guider", status.path(["guider", "navigator"]), dark2, mount, 'guider')
	guider = Guider(driver, "Guider", status.path(["guider"]), mount, dark2, 'guider', full_res = status.path(["full_res"]))

	runner2 = Runner(driver, "Guider", 'guider', cam2, navigator = nav2, guider = guider)
	runner2.start()

	main_loop()
	runner.join()
	runner2.join()

def run_calibrate_v4l2_g():
	global status
	status = Status("run_calibrate_v4l2_g.conf")
	cam = Camera(status.path(["camera"]))
	status.save()
	cam.prepare(1280, 960)

	ui.namedWindow('guider')
	go_dec = GuideOut("./guide_out_dec")
	dark = Median(5)

	try:
		from pyA20.gpio import gpio
		from pyA20.gpio import port
		from pyA20.gpio import connector

		gpio.init() #Initialize module. Always called first
		pins = [ port.PA8, port.PA9, port.PA10, port.PA20 ]

		for p in pins:
			gpio.setcfg(p, gpio.OUTPUT)  #Configure LED1 as output
			gpio.output(p, 0)

	except:
		pass

	vals = []
	for exp in np.arange(0.1, 1.5, 0.1):
		go_dec.out(0)
		cam.cmd('exp-sec-%f' % exp)
		out = 0

		for i in range(0,10):
			im, t = cam.capture()
			if im.ndim > 2:
				im = cv2.min(cv2.min(im[:, :, 0], im[:, :, 1]), im[:, :, 2])
			dark.add_mean(im)
			ui.imshow('guider', normalize(im))

		for test in range(0, 100):
			im, t = cam.capture()
			t = time.time()
			if im.ndim > 2:
				im = cv2.min(cv2.min(im[:, :, 0], im[:, :, 1]), im[:, :, 2])

			im = cv2.subtract(im, dark.get(), dtype=cv2_dtype(im.dtype))
			ui.imshow('guider', normalize(im))
			val = int(np.amax(im))
			log.info("%f %f, %f, %f", exp, test, out, val)

			if test == 10:
				val0 = val
				go_dec.out(1)
			elif test == 20:
				val1 = val
				go_dec.out(0)
				t0 = time.time()
				log.info("%f %d %d", exp, val0, val1)
				valm = (val1 + val0) / 2
				out = 0

			elif test > 20 and out == 0 and val < valm:
				log.info("change %f %f", exp, t - t0)
				vals.append((exp, t - t0))
				go_dec.out(1)
				t0 = time.time()
				out = 1
			elif test > 20 and out == 1 and val > valm:
				log.info("change %f %f", exp, t - t0)
				vals.append((exp, t - t0))
				go_dec.out(0)
				t0 = time.time()
				out = 0

	log.info("line", fit_line(vals))


	cmdQueue.put('exit')

def run_2_v4l2():
	global status
	status = Status("run_2_v4l2.conf")
	cam = Camera(status.path(["guider", "navigator", "camera"]))
	cam.prepare(1280, 960)
	
	fo = FocuserOut()
	cam1 = Camera(status.path(["navigator", "camera"]), fo)
	cam1.prepare(1280, 960)

	polar = Polar(status.path(["polar"]), ['navigator', 'guider', 'full-res'])

	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")

	mount = Mount(status.path(["mount"]), polar, go_ra, go_dec)

	dark1 = Median(5)
	dark2 = Median(5)
	
	nav1 = Navigator(status.path(["navigator"]), dark1, mount, 'navigator', polar_tid = 'polar')
	focuser = Focuser('navigator', status.path(["navigator", "focuser"]), dark = dark1)

	nav = Navigator(status.path(["guider", "navigator"]), dark2, mount, 'guider')

	guider = Guider(status.path(["guider"]), mount, dark2, 'guider')

	ui.namedWindow('navigator')
	ui.namedWindow('guider')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

	
	go_ra.out(1, 10) # move aside for 10s to collect darkframes

	runner = Runner('navigator', cam1, navigator = nav1, focuser=focuser, video_tid = 'full_res')
	runner.start()
	
	runner2 = Runner('guider', cam, navigator = nav, guider = guider)
	runner2.start()
	
	main_loop()
	
	runner.join()
	runner2.join()

def run_test_full_res():
	from kstars_camera import Camera_test_kstars, Camera_test_kstars_g
	global status
	status = Status("run_test_2.conf")

	global driver
	driver = IndiDriver()

	ui.namedWindow('navigator')
	ui.namedWindow('guider')
	ui.namedWindow('polar')
	ui.namedWindow('full_res')

	polar = Polar(status.path(["polar"]), ['navigator', 'guider', 'full-res'])

	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")

	mount = Mount(status.path(["mount"]), polar, go_ra, go_dec)

	dark1 = Median(5)
	dark2 = Median(5)

	fo = FocuserOut()
	cam1 = Camera_test_kstars(status.path(["navigator", "camera"]), go_ra, go_dec, fo)
	nav1 = Navigator(driver, "Navigator", status.path(["navigator"]), dark1, mount, 'navigator', polar_tid = 'polar', full_res = status.path(["full_res"]))

	nav = Navigator(driver, "Guider", status.path(["guider", "navigator"]), dark2, mount, 'guider')

	guider = Guider(driver, "Guider", status.path(["guider"]), mount, dark2, 'guider', full_res = status.path(["full_res"]))
	#cam = Camera_test(status.path(["guider", "navigator", "camera"]))
	cam = Camera_test_kstars_g(status.path(["guider", "navigator", "camera"]), cam1)


	focuser = Focuser(driver, "Navigator", 'navigator', status.path(["navigator", "focuser"]), dark = dark1)
	
#	profiler = LineProfiler()
#	profiler.add_function(Navigator.proc_full_res)
#	profiler.enable_by_count()
		
	for i in range(0,10000):
		fn = "../data/IMAGE_%03d.fits" % (i % 17);
		nav1.proc_full_res(fn, fn)
		time.sleep(1)
	
#	profiler.print_stats()
					

if __name__ == "__main__":
	os.environ["LC_NUMERIC"] = "C"
	signal.signal(signal.SIGINT, cmdQueue.send_exit)
	signal.signal(signal.SIGTERM, cmdQueue.send_exit)
	
	#mystderr = os.fdopen(os.dup(sys.stderr.fileno()), 'w', 0)
	#devnull = open(os.devnull,"w")
	#os.dup2(devnull.fileno(), sys.stdout.fileno())
	#os.dup2(devnull.fileno(), sys.stderr.fileno())
	
	#sys.stdout = mystderr
	#sys.stderr = mystderr
	

	#run_gphoto()
	#run_test_2_kstars()
	run_2_indi()
	#run_v4l2()
	#run_test_2_gphoto()
	#run_v4l2()
	#run_2()
	#run_test_g()
	#run_2()
	#run_test()
	#run_test_full_res()








