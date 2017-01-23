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
		print "camera:", cmd
		if cmd.startswith('exp-sec-'):
			self.status['exp-sec'] = float(cmd[len('exp-sec-'):])

		if cmd in ["f-3", "f-2", "f-1", "f+3", "f+2", "f+1"]:
			self.focuser.cmd(cmd)

	
	def capture(self):
		
		self.e_ra += random.random() * 0.001 - 0.0005
		self.e_dec += random.random() * 0.001 - 0.0005
		
		ra = (self.go_ra.recent_avg() - self.go_ra.recent_avg(0.001)) / 3600.0 * 60.0  + self.e_ra
		dec = -(self.go_dec.recent_avg() - self.go_dec.recent_avg(0.001)) / 3600.0 * 60.0  + self.e_dec
		
		print "set ra, dec %f,%f" % (ra,dec)
		self.iface.setRaDec(ra / 360.0 * 24.0, dec)
		time.sleep(0.5)
		
		self.iface.exportImage("/tmp/kstars.jpg") #,2000,2000, signature='sii')

		im = cv2.imread("/tmp/kstars.jpg")
		self.im = apply_gamma(im, 2.2)
		h, w, c = self.im.shape
		im = np.rot90(self.im[:, 0:w/2])
		
		print "focuser %d" % self.focuser.pos
		bl = np.abs(self.focuser.pos / 150.0)**2 + 0.3
		ibl = int(bl + 1)
		#im = cv2.blur(im, (ibl, ibl))
		#im = cv2.blur(im, (ibl, ibl))
		im = cv2.GaussianBlur(im, (51, 51), bl)
		
		return im, time.time()


	def capture_bulb(self, test = False, callback = None):
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
		if callback is not None:
			im = np.array(im, dtype=np.uint8)
			tmpFile = io.BytesIO()
			pil_image = Image.fromarray(im)
			pil_image.save(tmpFile,'JPEG')
			file_data = tmpFile.getvalue()
			callback(file_data)


	def shutdown(self):
		pass

class Camera_test_kstars_g:
	def __init__(self, status, cam0):
		self.status = status
		self.status['exp-sec'] = 0.5
		self.cam0 = cam0
	
	def cmd(self, cmd):
		print "camera:", cmd
		if cmd.startswith('exp-sec-'):
			self.status['exp-sec'] = float(cmd[len('exp-sec-'):])

	
	def capture(self):
		time.sleep(self.status['exp-sec'])
		self.cam0.capture()
		im = self.cam0.im
		h, w, c = im.shape
		return cv2.flip(im[h/4:h/4*3, w/2:], 1), time.time()

	def shutdown(self):
		pass


if __name__ == "__main__":
	from guide_out import GuideOut
	
	go_ra = GuideOut("./guide_out_ra")
        go_dec = GuideOut("./guide_out_dec")
        
        cam = Camera_test_kstars({}, go_ra, go_dec)
        im, t = cam.capture()
        print np.where(im > 0)
        