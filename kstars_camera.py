#!/usr/bin/python
import sys
import time
import select
import os
import random
import math
import cv2
import numpy as np
from PIL import Image
import io

import logging

log = logging.getLogger()

def apply_gamma(img, gamma):
	lut = np.fromiter( ( (x / 255.0)**gamma * 65535.0 for x in xrange(256)), dtype=np.uint16 )
	return np.take(lut, img)





class Camera_test_kstars:
	def __init__(self, status, go_ra, go_dec, focuser):
		self.status = status
		self.go_ra = go_ra
		self.go_dec = go_dec
		self.focuser = focuser
		self.status['exp-sec'] = 60
		self.status['test-exp-sec'] = 1
		self.e_ra = 70
		self.e_dec = 65
		self.status['exp_in_progress'] = False
		self.t0 = time.time()
		self.hyst = 2
		self.bahtinov = False
		
		import gobject

		gobject.threads_init()

		from dbus import glib
		glib.init_threads()

		# Create a session bus.
		import dbus
		bus = dbus.SessionBus()

		remote_object = bus.get_object("org.kde.kstars", # Connection name
                               "/KStars" # Object's path
                              )

		self.iface = dbus.Interface(remote_object, 'org.kde.kstars')

		
		self.capture()
	
	def cmd(self, cmd):
		log.info("camera: %s", cmd)
		if cmd.startswith('exp-sec-'):
			self.status['exp-sec'] = float(cmd[len('exp-sec-'):])

		if cmd == "f-1" and self.hyst > 0:
			self.hyst -= 1
			return

		if cmd == "f+1" and self.hyst < 5:
			self.hyst += 1
			return

		if cmd in ["f-3", "f-2", "f-1", "f+3", "f+2", "f+1"]:
			self.focuser.cmd(cmd)
		
		if cmd == 'z1':
			self.bahtinov = True
		
		if cmd == 'z0':
			self.bahtinov = False
		

	
	def capture(self):
		
		self.e_ra = 70 + np.random.normal(0, 0.002) + 0.001 * (time.time() - self.t0) + np.sin((time.time() - self.t0) / 3.0) * 0.002
		self.e_dec = 65 + np.random.normal(0, 0.002) - 0.0005 * (time.time() - self.t0)
		
		ra = -(self.go_ra.recent_avg() - self.go_ra.recent_avg(0.001)) / 3600.0 * 60.0  + self.e_ra
		dec = (self.go_dec.recent_avg() - self.go_dec.recent_avg(0.001)) / 3600.0 * 60.0  + self.e_dec
		
		log.info("set ra, dec %f,%f" % (ra,dec))
		self.iface.setRaDec(ra / 360.0 * 24.0, dec)
		time.sleep(0.5)
		
		self.iface.exportImage("/tmp/kstars.jpg") #,2000,2000, signature='sii')

		im = cv2.imread("/tmp/kstars.jpg")
		if self.bahtinov:
			d = self.focuser.pos / 5.0
			id = int(d)
			
			im2 = np.zeros((500, 500, 3), dtype = im.dtype)
		
			cv2.line(im2,(0,200),(500,400),(200, 200, 200),3)
			cv2.line(im2,(0,400),(500,200),(200, 200, 200),3)
			cv2.line(im2,(0,300 + id),(500,300+id),(200,200,200),3)
			
			M = cv2.getRotationMatrix2D((250,250),60,0.2)
			im2 = cv2.warpAffine(im2, M, (500, 500), flags=cv2.INTER_LANCZOS4)
			print im.shape
			im[0:500, 0:500] = cv2.add(im[0:500, 0:500], im2)

		self.im = apply_gamma(im, 2.2)
		h, w, c = self.im.shape
		im = np.rot90(self.im[:, 0:w/2])
		
		log.info("focuser %d" % self.focuser.pos)
		bl = np.abs(self.focuser.pos / 150.0)**2 + 1
		
		
		
		ibl = int(bl + 1)
		#im = cv2.blur(im, (ibl, ibl))
		#im = cv2.blur(im, (ibl, ibl))
		im = cv2.GaussianBlur(im, (51, 51), bl)
		im = cv2.add(im, np.random.normal(3, 3, im.shape), dtype = cv2.CV_16UC3)
		
		
		return im, time.time()


	def capture_bulb(self, test = False, callback_start = None, callback_end = None):

		if callback_start is not None:
			callback_start()
		if test:
			sec = self.status['test-exp-sec']
		else:
			sec = self.status['exp-sec']
		self.status['exp_in_progress'] = True
		for i in range(0, int(sec + 0.5)):
			time.sleep(1)
			self.status['cur_time'] = i
		
		self.status['exp_in_progress'] = False
		h, w, c = self.im.shape
		im = np.rot90(self.im[:, 0:w/2])

		bl = np.abs(self.focuser.pos / 150.0)**2 + 1
		ibl = int(bl + 1)
		#im = cv2.blur(im, (ibl, ibl))
		#im = cv2.blur(im, (ibl, ibl))
		im = cv2.GaussianBlur(im, (51, 51), bl)
		im = cv2.add(im, np.random.normal(3, 3, im.shape), dtype = cv2.CV_16UC3)

		if callback_end is not None:
			im = np.array(im, dtype=np.uint8)
			tmpFile = io.BytesIO()
			pil_image = Image.fromarray(im)
			#pil_image = Image.open('preview2.jpg')
			pil_image.save(tmpFile,'JPEG')
			file_data = tmpFile.getvalue()
			callback_end(file_data, "img_{}.jpg".format(time.time()))


	def shutdown(self):
		pass

class Camera_test_kstars_g:
	def __init__(self, status, cam0):
		self.status = status
		self.status['exp-sec'] = 0.5
		self.cam0 = cam0
	
	def cmd(self, cmd):
		log.info("camera: %s", cmd)
		if cmd.startswith('exp-sec-'):
			self.status['exp-sec'] = float(cmd[len('exp-sec-'):])

	
	def capture(self):
		time.sleep(self.status['exp-sec'])
		self.cam0.capture()
		im = self.cam0.im
		h, w, c = im.shape
		im = im[h/4:h/4*3, w/2:]
		#im = cv2.flip(im, 1)

		bl = np.abs((self.cam0.focuser.pos + 200) / 150.0)**2 + 1
		ibl = int(bl + 1)
		#im = cv2.blur(im, (ibl, ibl))
		#im = cv2.blur(im, (ibl, ibl))
		im = cv2.GaussianBlur(im, (51, 51), bl)
		im = cv2.add(im, np.random.normal(3, 3, im.shape), dtype = cv2.CV_16UC3)
		return im, time.time()

	def shutdown(self):
		pass


if __name__ == "__main__":
	from guide_out import GuideOut
	
	go_ra = GuideOut("./guide_out_ra")
        go_dec = GuideOut("./guide_out_dec")
        
        cam = Camera_test_kstars({}, go_ra, go_dec)
        im, t = cam.capture()
        