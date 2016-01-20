#!/usr/bin/python
import gphoto2 as gp
import subprocess
import logging
import os
import sys
import time
import io
from PIL import Image
import numpy as np

from gui import ui
from cmd import cmdQueue
from stacktraces import stacktraces

def apply_gamma(img, gamma):
	lut = np.fromiter( ( (x / 255.0)**gamma * 65535.0 for x in xrange(256)), dtype=np.uint16 )
	return np.take(lut, img)



class Camera_gphoto:
	def __init__(self):
		self.iso = 400
		self.test_iso = 3200
		self.exp_sec = 120
		self.test_exp_sec = 15

	def get_config_value(self, name):
		config = gp.check_result(gp.gp_camera_get_config(self.camera, self.context))
		OK, widget = gp.gp_widget_get_child_by_name(config, name)
		if OK >= gp.GP_OK:
			# set value
			value = gp.check_result(gp.gp_widget_get_value(widget))
			print >> sys.stderr,name, value
			return value

	def set_config_choice(self, name, num):
		for t in range(0, 20):
			try:
				config = gp.check_result(gp.gp_camera_get_config(self.camera, self.context))
				OK, widget = gp.gp_widget_get_child_by_name(config, name)
				if OK >= gp.GP_OK:
					# set value
					value = gp.check_result(gp.gp_widget_get_choice(widget, num))
					print >> sys.stderr,name, value
					gp.check_result(gp.gp_widget_set_value(widget, value))
				# set config
				gp.check_result(gp.gp_camera_set_config(self.camera, config, self.context))
				break
			except gp.GPhoto2Error as ex:
				print ex.code
				time.sleep(0.1)
				continue

	def set_config_value(self, name, value):
		for t in range(0, 20):
			try:
				config = gp.check_result(gp.gp_camera_get_config(self.camera, self.context))
				OK, widget = gp.gp_widget_get_child_by_name(config, name)
				if OK >= gp.GP_OK:
					# set value
					print >> sys.stderr,name, value
					gp.check_result(gp.gp_widget_set_value(widget, value))
				# set config
				gp.check_result(gp.gp_camera_set_config(self.camera, config, self.context))
				break
			except gp.GPhoto2Error as ex:
				print ex.code
				time.sleep(0.1)
				continue
	
	def capture_bulb(self, sec, iso = None, card = False):
		if iso is not None:
			self.set_config_value('iso', str(iso))
			
		if card:
			self.set_config_choice('capturetarget', 1) #card
			#self.set_config_choice('imageformat', 24) #RAW 
			self.set_config_choice('imageformat', 7) #RAW + Large Normal JPEG 
			
		else:
			self.set_config_choice('capturetarget', 0) #mem
			self.set_config_choice('imageformat', 1) #Large Normal JPEG
		
		self.set_config_value('eosremoterelease', 'Immediate')
		time.sleep(sec)
		self.set_config_value('eosremoterelease', 'Release Full')
		while True:
			e, file_path =  gp.check_result(gp.gp_camera_wait_for_event(self.camera, 1000,self.context))
			print >> sys.stderr, "wait for event", e
			if e == gp.GP_EVENT_FILE_ADDED:
				print >> sys.stderr, "filepath:", file_path.folder, file_path.name
				filename, file_extension = os.path.splitext(file_path.name)
				if file_extension == ".jpg" or file_extension == ".JPG":
					break
				
		
		target = os.path.join('/tmp', file_path.name)
		
		n = 20
		while True:
			n -= 1
			try:
				camera_file = gp.check_result(gp.gp_camera_file_get(self.camera, file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL, self.context))
				file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
				ui.imshow_jpg("full_res", io.BytesIO(file_data));
				break
			except gp.GPhoto2Error as ex:
				if ex.code == gp.GP_ERROR_CAMERA_BUSY:
					time.sleep(1)

					if (n > 0):
						continue
				raise

		#stop review on display
		self.set_config_value('eosremoterelease', 'Press Half')
		self.set_config_value('eosremoterelease', 'Release Half')
	
		time.sleep(.2)
		self.set_config_choice('output', 1)
		time.sleep(.2)
		self.set_config_choice('output', 0)
		time.sleep(3)
		
	
	
	def prepare(self):
		self.shape = (704, 1056)
		self.zoom_shape = (680, 1024)
	
		subprocess.call(['killall', 'gvfsd-gphoto2'])
		subprocess.call(['killall', 'gvfs-gphoto2-volume-monitor'])
	
		logging.basicConfig(
			format='%(levelname)s: %(name)s: %(message)s', level=logging.ERROR)
		gp.check_result(gp.use_python_logging())
		self.context = gp.gp_context_new()
		while True:
			try:
				self.camera = gp.check_result(gp.gp_camera_new())
				gp.check_result(gp.gp_camera_init(self.camera, self.context))
				break
			except gp.GPhoto2Error as ex:
				print "gphoto2 camera is not ready"
				time.sleep(2)
				continue
		
		# wake up the camera
		self.set_config_value('eosremoterelease', 'Press Half')
		self.set_config_value('eosremoterelease', 'Release Half')
		self.set_config_value('eosremoterelease', 'Release Full')
		
		cur_time = self.get_config_value('datetime')
		subprocess.call(['date', '--set', '@' + str(cur_time) ])
		
		# required configuration will depend on camera type!
		self.set_config_choice('capturesizeclass', 2)
	
		time.sleep(.2)
		self.set_config_choice('output', 1)
		time.sleep(.2)
		self.set_config_choice('output', 0)
	
		self.zoom = 1
		self.x = 3000
		self.y = 2000
		self.set_config_value('eoszoomposition', "%d,%d" % (self.x, self.y))
		time.sleep(3)
	
	def cmd(self, cmd, x = None, y = None):
		try:
			if cmd == "f-3":
				self.set_config_choice('manualfocusdrive', 2)
			if cmd == "f-2":
				self.set_config_choice('manualfocusdrive', 1)
			if cmd == "f-1":
				self.set_config_choice('manualfocusdrive', 0)
			if cmd == "f+1":
				self.set_config_choice('manualfocusdrive', 4)
			if cmd == "f+2":
				self.set_config_choice('manualfocusdrive', 5)
			if cmd == "f+3":
				self.set_config_choice('manualfocusdrive', 6)
			if cmd == "z1":
        
				zoom = 5
							
				self.x = x * zoom - self.zoom_shape[1] / 2
				self.y = y * zoom - self.zoom_shape[0] / 2
				self.set_config_value('eoszoomposition', "%d,%d" % (self.x, self.y))
				self.set_config_value('eoszoom', '5')
				time.sleep(.2)
				self.set_config_choice('output', 1)
				time.sleep(.2)
				self.set_config_choice('output', 0)
				time.sleep(12)
				self.capture()
        
			if cmd == "z0":
				zoom = 1
				self.set_config_value('eoszoom', '1')
				time.sleep(.2)
				self.set_config_choice('output', 1)
				time.sleep(.2)
				self.set_config_choice('output', 0)
				time.sleep(12)
				self.capture()
		
			if cmd == 'left':
				self.x = max(100, self.x - 100)
				self.set_config_value('eoszoomposition', "%d,%d" % (self.x, self.y))
			if cmd == 'right':
				self.x = self.x + 100
				self.set_config_value('eoszoomposition', "%d,%d" % (self.x, self.y))
			if cmd == 'up':
				self.y = max(100, self.y - 100)
				self.set_config_value('eoszoomposition', "%d,%d" % (self.x, self.y))
			if cmd == 'down':
				self.y = self.y + 100
				self.set_config_value('eoszoomposition', "%d,%d" % (self.x, self.y))
		
			if cmd.startswith('iso-'):
				self.iso = cmd[len('iso-'):]
        
			if cmd.startswith('test-iso-'):
				self.test_iso = cmd[len('test-iso-'):]
        
			if cmd.startswith('exp-sec-'):
				self.exp_sec = int(cmd[len('exp-sec-'):])
        
			if cmd.startswith('test-exp-sec-'):
				self.test_exp_sec = int(cmd[len('test-exp-sec-'):])
			
			if cmd.startswith('f-number-'):
				self.set_config_value('aperture', cmd[len('f-number-'):])
        
			if cmd == 'test-capture':
				self.capture_bulb(self.test_exp_sec, self.test_iso)
			
			if cmd == 'capture':
				self.capture_bulb(self.exp_sec, self.iso, card=True)
				cmdQueue.put('capture-finished')
			
		except gp.GPhoto2Error as ex:
			print "Unexpected error:", sys.exc_info()
			print "code:", ex.code
			stacktraces()
			time.sleep(1)
			if ex.code == -7 or ex.code == -1:
				gp.gp_camera_exit(self.camera, self.context)
				self.prepare()

	def capture(self):
		while True:
			try:
				for i in range(0,20):
					try:
						camera_file = gp.check_result(gp.gp_camera_capture_preview(self.camera, self.context))
						break
					except gp.GPhoto2Error as ex:
						if i < 19:
							continue
					raise
						
				file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
	
				pil_image = Image.open(io.BytesIO(file_data))
				#pil_image.save("testimg2_" + str(i) + ".tif")
				im = np.array(pil_image)
				im = apply_gamma(im, 2.2)
	
				return im, None
	
			except KeyboardInterrupt:
				break
			except gp.GPhoto2Error as ex:
				print "Unexpected error:", sys.exc_info()
				print "code:", ex.code
				stacktraces()
				time.sleep(1)
				if ex.code == -7 or ex.code == -1:
					gp.gp_camera_exit(self.camera, self.context)
					self.prepare()

	
	def __del__(self):
		gp.check_result(gp.gp_camera_exit(self.camera, self.context))
	
	
	
	
if __name__ == "__main__":
	cam = Camera_gphoto()
	cam.prepare()
	cam.capture()