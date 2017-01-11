#!/usr/bin/python
import sys
import time
import select
import os
import random
import math
import cv2
import numpy as np

def apply_gamma(img, gamma):
	lut = np.fromiter( ( (x / 255.0)**gamma * 65535.0 for x in xrange(256)), dtype=np.uint16 )
	return np.take(lut, img)





class Camera_test_kstars:
	def __init__(self, status, go_ra, go_dec):
		self.status = status
		self.go_ra = go_ra
		self.go_dec = go_dec
		self.status['exp-sec'] = 0.5
		self.e_ra = 70
		self.e_dec = 65
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

	
	def capture(self):
		
		self.e_ra += random.random() * 0.001 - 0.0005
		self.e_dec += random.random() * 0.001 - 0.0005
		
		ra = (self.go_ra.recent_avg() - self.go_ra.recent_avg(0.001)) / 3600.0 * 60.0  + self.e_ra
		dec = (self.go_dec.recent_avg() - self.go_dec.recent_avg(0.001)) / 3600.0 * 60.0  + self.e_dec
		
		print "set ra, dec %f,%f" % (ra,dec)
		self.iface.setRaDec(ra / 360.0 * 24.0, dec)
		time.sleep(self.status['exp-sec'])
		
		self.iface.exportImage("/tmp/kstars.jpg") #,2000,2000, signature='sii')

		im = cv2.imread("/tmp/kstars.jpg")
		self.im = apply_gamma(im, 2.2)
		h, w, c = self.im.shape
		return np.rot90(self.im[:, 0:w/2]), time.time()

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
        