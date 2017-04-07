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

from uvc_xu_control import *

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
    def __init__(self, status):
        self.status = status
        self.buffers = []
        self.i = 0
        self.dev_name = self.status.setdefault("device", "/dev/video0")
        self.status['lensname'] = 'default'
        self.status.setdefault('exp-sec', 1.5)
        self.vd = None
        self.mm = []

    def _prepare(self, width, height, format = V4L2_PIX_FMT_SBGGR16):


        self.vd = os.open(self.dev_name, os.O_RDWR | os.O_NONBLOCK, 0)
        print "v4l open fd", self.vd
	self.fmt = format


	
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

	if (self.fmt == V4L2_PIX_FMT_SBGGR16):
		fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV
	else:
        	fmt.fmt.pix.pixelformat = format
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
	max_t = 5

        fcntl.ioctl(self.vd, VIDIOC_STREAMON, type)

        ready_to_read, ready_to_write, in_error = ([], [], [])
        while len(ready_to_read) == 0 and time.time() - t0 < max_t:
            try:
                ready_to_read, ready_to_write, in_error = select.select([self.vd], [], [], max_t)
            except (OSError, select.error) as why:
                continue

	if time.time() - t0 >= max_t:
		fcntl.ioctl(self.vd, VIDIOC_STREAMOFF, type)

		os.close(self.vd)
		self.vd = None
		return False
	
	return True


    def prepare(self, width, height, format = V4L2_PIX_FMT_SBGGR16):
	self.w = width
	self.h = height

	i = 0
	while True:
		try:
			if self._prepare(width, height, format):
				break
		except:
			pass
		i += 1
		print "camera init failed, retry %d" % i
		self.shutdown()
		time.sleep(1)
	if self.vd is None:
		return False

	try:
		# longest manual exposure available via uvc driver
		self.control(V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL)
		self.control(V4L2_CID_EXPOSURE_ABSOLUTE, 5000)
		
	except: 
		pass

#######################################################################
        time.sleep(0.2)
	
	try:
		if (self.fmt == V4L2_PIX_FMT_SBGGR16):
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
	#        sonix_write_sensor(self.vd, 0x3070, 3) #test pattern
	
	        sonix_write_sensor(self.vd, 0x305e, 0xffff) # digital gain
	        sonix_write_sensor(self.vd, 0x30b0, 0x1030) # analog gain

	        #sonix_write_sensor(self.vd, 0x30ea, 0x8c00) #disable black level compensation
	        #sonix_write_sensor(self.vd, 0x3044, 0x0000) #disable row noise compensation
	        sonix_write_sensor(self.vd, 0x3012, int(0x2400 * self.status['exp-sec'])) #exposure time coarse
	except:
		pass
        
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
		img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB)
	elif (self.fmt == V4L2_PIX_FMT_YUYV):
		img = img.reshape((-1, self.width, 2))
		img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
	elif (self.fmt == V4L2_PIX_FMT_MJPEG):
		img = cv2.imdecode(img, cv2.CV_LOAD_IMAGE_COLOR)
	else:
		#unsupported format
		pass
	
        fcntl.ioctl(self.vd, VIDIOC_QBUF, buf)
        return img, None

    def control(self, id, value):
        control = v4l2_control(id, value)
        fcntl.ioctl(self.vd, VIDIOC_S_CTRL, control)
        time.sleep(0.01)


    def capture(self):
    	for i in range(0, 20):
    		try:
    			return self._capture()
    		except:
    			self.shutdown()
    			self.prepare(self.w, self.h)


    def cmd(self, cmd):
    	try:
		if cmd.startswith('exp-sec-'):
			exp_sec = float(cmd[len('exp-sec-'):])
			sonix_write_sensor(self.vd, 0x3012, int(0x2400 * exp_sec)) #exposure time coarse
			self.status['exp-sec'] = exp_sec
	except:
		self.shutdown()
		self.prepare(self.w, self.h)

    def shutdown(self):
	#for mm in self.mm:
	#	mm.close()
	self.mm = []
	if self.vd is not None:
		print "v4l close fd", self.vd
		os.close(self.vd)
		self.vd = None
		time.sleep(1)

if __name__ == "__main__":
	cam = Camera({"device":"/dev/video1", 'exp-sec':0.1})
	cam.prepare(1280, 960,V4L2_PIX_FMT_YUYV)

	for i in range(0,20000):
		img,t = cam.capture()
		print img
		cv2.imshow('capture', img)
		cv2.imwrite('capture%d.jpg'%i, img)
		ch = 0xFF & cv2.waitKey(1)
		if ch == 27:
			break



	cv2.destroyAllWindows()

	

