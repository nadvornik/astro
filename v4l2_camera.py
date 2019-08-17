#!/usr/bin/env python

# Copyright (C) 2015 Vladimir Nadvornik
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import os
import mmap
from errno import EINVAL as _EINVAL
import ctypes
import numpy as np
import fcntl
from v4l2 import *
import select
import cv2
import time
import sys

from uvc_xu_control import *
import logging

log = logging.getLogger()


# this file contains hacks for ELP-USB130W01MT camera with AR0130 sensor

#######################################################################
# based on info from
# http://comments.gmane.org/gmane.linux.drivers.uvc.devel/3190
def sonix_write_sensor(fd, addr, val):
	data = (ctypes.c_uint8 * 8)(0x10, 3, (addr >> 8) & 0xff, addr & 0xff, (val >> 8) & 0xff, val & 0xff, 0x00, 0x00)
	control = uvc_xu_control_query(3, 2, UVC_SET_CUR, 8, data)
	fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, control)

def sonix_write_asic(fd, addr, val):
	data = (ctypes.c_uint8 * 4)(addr & 0xff, (addr >> 8) & 0xff, val & 0xff, 0x00)
	control = uvc_xu_control_query(3, 1, UVC_SET_CUR, 4, data)
	fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, control)

def sonix_read_asic(fd, addr):
	data = (ctypes.c_uint8 * 4)(addr & 0xff, (addr >> 8) & 0xff, 0x00, 0xff)
	control = uvc_xu_control_query(3, 1, UVC_SET_CUR, 4, data)
	fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, control)
	data[3] = 0
	control.query = UVC_GET_CUR
	fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, control)
	return data[2]
#######################################################################


class Camera:
	def __init__(self, status, focuser = None):
		self.status = status
		self.buffers = []
		self.i = 0
		self.dev_name = self.status.setdefault("device", "/dev/video0")
		self.status['lensname'] = 'default'
		self.status.setdefault('exp-sec', 0.5)
		self.vd = None
		self.mm = []
		self.focuser = focuser
		self.status.setdefault('width', 1280)
		self.status.setdefault('height', 960)
		self.status.setdefault('format', V4L2_PIX_FMT_Y16)
		self.status['capture_idx'] = 0
		self.status['capture'] = False
		self.status.setdefault('capture_path', None)

	def _prepare(self, width = None, height = None, format = None, decode=True):


		log.info("v4l open {}".format(self.dev_name))
		self.vd = os.open(self.dev_name, os.O_RDWR | os.O_NONBLOCK, 0)
		log.info("v4l open fd {}".format(self.vd))
		self.fmt = format
		self.decode = decode
	
		cp = v4l2_capability()
		fcntl.ioctl(self.vd, VIDIOC_QUERYCAP, cp)

		fmt = v4l2_format()
		fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
		fmt.fmt.pix.width = width
		fmt.fmt.pix.height = height
		fmt.fmt.pix.field = V4L2_FIELD_NONE

		#fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_JPEG
		#fcntl.ioctl(self.vd, VIDIOC_S_FMT, fmt)
		#time.sleep(0.01)

		if (self.fmt != V4L2_PIX_FMT_JPEG):
			fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV
		else:
			fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_JPEG
		fcntl.ioctl(self.vd, VIDIOC_S_FMT, fmt)
		time.sleep(0.01)
		self.control(V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_APERTURE_PRIORITY)
		#self.control(V4L2_CID_EXPOSURE_ABSOLUTE, 50)

		self.width = width

		req = v4l2_requestbuffers()
		req.count = 2
		req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
		req.memory = V4L2_MEMORY_MMAP
		fcntl.ioctl(self.vd, VIDIOC_REQBUFS, req)
		time.sleep(0.01)
		for ind in range(req.count):
			buf = v4l2_buffer()
			buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
			buf.memory = V4L2_MEMORY_MMAP
			buf.index = ind
			fcntl.ioctl(self.vd, VIDIOC_QUERYBUF, buf)
			#buffer = ImageBuffer()
			#buffer.length = buf.length
			mm = mmap.mmap(self.vd, buf.length, mmap.MAP_SHARED, 
				mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset)
			self.mm.append(mm)
			buffer = np.ndarray((buf.length,), dtype=np.uint8, buffer=mm)
			self.buffers.append(buffer)
		for ind in range(req.count):
			buf = v4l2_buffer()
			buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
			buf.memory = V4L2_MEMORY_MMAP
			buf.index = ind
			fcntl.ioctl(self.vd, VIDIOC_QBUF, buf)
			time.sleep(0.01)
		type = v4l2_buf_type(V4L2_BUF_TYPE_VIDEO_CAPTURE)


		t0 = time.time()
		max_t = 30

		fcntl.ioctl(self.vd, VIDIOC_STREAMON, type)

		ready_to_read, ready_to_write, in_error = ([], [], [])
		while len(ready_to_read) == 0 and time.time() - t0 < max_t:
			try:
				ready_to_read, ready_to_write, in_error = select.select([self.vd], [], [], max_t)
			except (OSError, select.error) as why:
				log.exception('Unexpected error')
				continue

		if time.time() - t0 >= max_t:
			fcntl.ioctl(self.vd, VIDIOC_STREAMOFF, type)

			os.close(self.vd)
			self.vd = None
			return False
	
		return True


	def prepare(self, width = None, height = None, format = None, decode = True):
		if self.vd is not None:
			log.error("already initialized, run shutdown first")
			return True

		if width == None:
			width = self.status['width']
		if height == None:
			height = self.status['height']

		if format == None:
			format = self.status['format']
	
		self.status['width'] = width
		self.status['height'] = height
		self.status['format'] = format

		self.w = width
		self.h = height

		i = 0
		while True:
			try:
				if self._prepare(width, height, format, decode):
					break
			except:
				log.exception('Unexpected error')
				i += 1
			log.info("camera init failed, retry %d" % i)
			self.shutdown()
			time.sleep(1)
		if self.vd is None:
			return False

		try:
			# longest manual exposure available via uvc driver
			self.control(V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL)
			self.control(V4L2_CID_EXPOSURE_ABSOLUTE, 500)
		
		except: 
			log.exception('Unexpected error')

		time.sleep(0.2)
	
		try:
			if (self.fmt == V4L2_PIX_FMT_SBGGR16 or self.fmt == V4L2_PIX_FMT_Y16):
				# the registers seems to be similar to gspca/sn9c20x.c driver
				# but the actual addresses are different.
				#
				# this register changed it's value between mjpeg and yuv
				# so I tried other values ...
				# this one worked:
				sonix_write_asic(self.vd, 0x1100, 0xff) # switch to raw mode

				time.sleep(0.2)
		
				# modify sensor registers
				# AR0130 Register Reference
				# http://www.onsemi.com/pub_link/Collateral/AND9214-D.PDF
		#		sonix_write_sensor(self.vd, 0x3070, 3) #test pattern
	
				sonix_write_sensor(self.vd, 0x305e, 0xffff) # digital gain
				sonix_write_sensor(self.vd, 0x30b0, 0x1030) # analog gain

				#sonix_write_sensor(self.vd, 0x30ea, 0x8c00) #disable black level compensation
				#sonix_write_sensor(self.vd, 0x3044, 0x0000) #disable row noise compensation
				#sonix_write_sensor(self.vd, 0x3012, int(0x2400 * self.status['exp-sec'])) #exposure time coarse
		except:
			log.exception('Unexpected error')		

		self.cmd("exp-sec-%f" % self.status['exp-sec'])
		# skip incorrectly set  frames at the beginning
		self.capture()
		self.capture()
		self.capture()
		return True
	
#######################################################################


	def _capture(self):
		ready_to_read, ready_to_write, in_error = ([], [], [])
		while len(ready_to_read) == 0:
			try:
				ready_to_read, ready_to_write, in_error = select.select([self.vd], [], [])
			except (OSError, select.error) as why:
				continue

		buf = v4l2_buffer()
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
		buf.memory = V4L2_MEMORY_MMAP
		fcntl.ioctl(self.vd, VIDIOC_DQBUF, buf)
		img = self.buffers[buf.index]
		if (self.fmt == V4L2_PIX_FMT_SBGGR16):
			img = img.reshape((-1, self.width, 2))
			img16 = np.array(img[:,:, 0], dtype=np.uint16)
			img = img16 * 256 + img[:,:, 1] * 64
		
			if self.status['capture'] and self.status['capture_path'] is not None:
				i = self.status['capture_idx']
				while os.path.isfile(self.status['capture_path'] + 'capture%04d.tif' % i):
					i += 1
				cv2.imwrite(self.status['capture_path'] + 'capture%04d.tif' % i, img)
				log.info("saved {}".format(i))
				i += 1
				self.status['capture_idx'] = i

		
			if self.decode:
				img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB)
		elif (self.fmt == V4L2_PIX_FMT_Y16):
			img = img.reshape((-1, self.width, 2))
			img16 = np.array(img[:,:, 0], dtype=np.uint16)
			img = img16 * 256 + img[:,:, 1] * 64
		
			if self.status['capture'] and self.status['capture_path'] is not None:
				i = self.status['capture_idx']
				while os.path.isfile(self.status['capture_path'] + 'capture%04d.tif' % i):
					i += 1
				cv2.imwrite(self.status['capture_path'] + 'capture%04d.tif' % i, img)
				log.info("saved {}".format(i))
				i += 1
				self.status['capture_idx'] = i

		
			if self.decode:
				img = cv2.blur(img, (2, 2))
		elif (self.fmt == V4L2_PIX_FMT_YUYV):
			img = img.reshape((-1, self.width, 2))
			if self.decode:
				img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
		elif (self.fmt == V4L2_PIX_FMT_MJPEG):
			if self.decode:
				img = cv2.imdecode(img, cv2.CV_LOAD_IMAGE_COLOR)
			else:
				img = np.array(img, copy=True)
		else:
			#unsupported format
			log.exception('Unexpected error')	

		fcntl.ioctl(self.vd, VIDIOC_QBUF, buf)
		return img, time.time()

	def control(self, id, value):
		control = v4l2_control(id, value)
		fcntl.ioctl(self.vd, VIDIOC_S_CTRL, control)
		time.sleep(0.01)


	def capture(self):
		for i in range(0, 20):
			try:
				return self._capture()
			except:
				log.exception('Unexpected error')
				self.shutdown()
				self.prepare(self.w, self.h, self.fmt, self.decode)


	def cmd(self, cmd):
		try:
			if cmd in ["f-3", "f-2", "f-1", "f+3", "f+2", "f+1"]:
				if self.focuser is not None:
					self.focuser.cmd(cmd)
		except:
			pass

		try:
			if cmd == 'capture_start' and self.status['capture_path'] is not None:
				self.status['capture'] = True
			elif cmd == 'capture_stop':
				self.status['capture'] = False
		
		except:
			pass



		try:
			if cmd.startswith('exp-sec-'):
				exp_sec = float(cmd[len('exp-sec-'):])
				exp_uvc = min(int(exp_sec * 10000), 5000)
				log.info("exp_sec {}, exp_uvc {}".format(exp_sec, exp_uvc))
				self.control(V4L2_CID_EXPOSURE_ABSOLUTE, exp_uvc)
				if exp_sec > 0.5:
					sonix_write_sensor(self.vd, 0x3012, int(0x2400 * exp_sec)) #exposure time coarse
				self.status['exp-sec'] = exp_sec
		except:
			log.exception('Unexpected error')
			self.shutdown()
			self.prepare(self.w, self.h, self.fmt, self.decode)

	def shutdown(self):
		if self.vd is not None:
			type = v4l2_buf_type(V4L2_BUF_TYPE_VIDEO_CAPTURE)
			try:
				fcntl.ioctl(self.vd, VIDIOC_STREAMOFF, type)
			except:
				log.exception('Unexpected error')		
		del self.buffers
		for mm in self.mm:
			mm.close()
		self.mm = []
		self.buffers = []
		if self.vd is not None:
			log.info("v4l close fd %s", self.vd)
			os.close(self.vd)
			self.vd = None
			time.sleep(1)

if __name__ == "__main__":
	exp = 0.5
	cam = Camera({"device":"/dev/video2", 'exp-sec': exp})
	cam.prepare(1280, 960,V4L2_PIX_FMT_SBGGR16, False)

	capt = False
	i = 0
	while True:
		img,t = cam.capture()
		cv2.imshow('capture', cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB))
		if capt:
			while os.path.isfile('capture%04d.tif' % i):
				i += 1
			cv2.imwrite('capture%04d.tif' % i, img)
			log.info("saved {}".format(i))
			i += 1
	
		ch = 0xFF & cv2.waitKey(1)
		if ch == 27:
			break
		elif ch == ord('a'):
			exp *= 1.1
			cam.cmd("exp-sec-%f" % exp)
		elif ch == ord('z'):
			exp /= 1.1
			cam.cmd("exp-sec-%f" % exp)
		elif ch == ord(' '):
			capt = True
		elif ch == ord('s'):
			capt = False
			
			


	cam.shutdown()
	cv2.destroyAllWindows()

	

