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

from cmd import cmdQueue

class Camera_gphoto:

	def set_config_choice(self, name, num):
		config = gp.check_result(gp.gp_camera_get_config(self.camera, self.context))
		OK, widget = gp.gp_widget_get_child_by_name(config, name)
		if OK >= gp.GP_OK:
			# set value
			value = gp.check_result(gp.gp_widget_get_choice(widget, num))
			print name, value
			gp.check_result(gp.gp_widget_set_value(widget, value))
		# set config
		gp.check_result(gp.gp_camera_set_config(self.camera, config, self.context))

	def set_config_value(self, name, value):
		config = gp.check_result(gp.gp_camera_get_config(self.camera, self.context))
		OK, widget = gp.gp_widget_get_child_by_name(config, name)
		if OK >= gp.GP_OK:
			# set value
			print name, value
			gp.check_result(gp.gp_widget_set_value(widget, value))
		# set config
		gp.check_result(gp.gp_camera_set_config(self.camera, config, self.context))
	
	
	def capture_bulb(self, sec, card = False):
		if self.camera_file is not None:
			file_data = gp.check_result(gp.gp_file_get_data_and_size(self.camera_file))
			self.camera_file = None
		while True:
			try:
				if card:
					self.set_config_choice('capturetarget', 1) #card
					self.set_config_choice('imageformat', 24) #RAW 
				else:
					self.set_config_choice('capturetarget', 0) #mem
					self.set_config_choice('imageformat', 1) #Large Normal JPEG
			
				break
			except gp.GPhoto2Error as ex:
            			if ex.code == gp.GP_ERROR_CAMERA_BUSY:
                			time.sleep(0.1)
               				continue
            			# some other error we can't handle here
            			raise

		self.set_config_value('shutterspeed', 'bulb')
		time.sleep(.1)
		self.set_config_value('eosremoterelease', 'Immediate')
		time.sleep(sec)
		while True:
			try:
				file_path = gp.check_result(gp.gp_camera_capture(self.camera, gp.GP_CAPTURE_IMAGE, self.context))
				break
			except gp.GPhoto2Error as ex:
				if ex.code == gp.GP_ERROR_CAMERA_BUSY:
					continue
				raise
		
		target = os.path.join('/tmp', file_path.name)
		
		n = 20
		while True:
			n -= 1
			try:
				camera_file = gp.check_result(gp.gp_camera_file_get(self.camera, file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL, self.context))
				gp.check_result(gp.gp_file_save(camera_file, target))
				break
			except gp.GPhoto2Error as ex:
				if ex.code == gp.GP_ERROR_CAMERA_BUSY:
					time.sleep(1)

					if (n > 0):
						continue
				raise

	
		self.set_config_choice('capturetarget', 0) #mem
		time.sleep(.1)
		self.set_config_choice('output', 1)
		self.set_config_choice('output', 0)
		time.sleep(2)
		
	
	
	def prepare(self):
		subprocess.call(['killall', 'gvfsd-gphoto2'])
		subprocess.call(['killall', 'gvfs-gphoto2-volume-monitor'])
	
		logging.basicConfig(
			format='%(levelname)s: %(name)s: %(message)s', level=logging.INFO)
		gp.check_result(gp.use_python_logging())
		self.context = gp.gp_context_new()
		self.camera = gp.check_result(gp.gp_camera_new())
		gp.check_result(gp.gp_camera_init(self.camera, self.context))
		# required configuration will depend on camera type!
		self.set_config_choice('capturesizeclass', 2)
	
		self.set_config_choice('output', 1)
		self.set_config_choice('output', 0)
		time.sleep(2)
	
		self.zoom = 1
		self.x = 3000
		self.y = 2000
		self.set_config_value('eoszoomposition', "%d,%d" % (self.x, self.y))
		self.camera_file = None
	
	def cmd(self, cmd, x = None, y = None):
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
			if self.camera_file is not None:
				file_data = gp.check_result(gp.gp_file_get_data_and_size(self.camera_file))
				self.camera_file = None

			zoom = 5
						
			self.x = x * zoom - 300
			self.y = y * zoom - 300
			self.set_config_value('eoszoomposition', "%d,%d" % (self.x, self.y))
			self.set_config_value('eoszoom', '5')
			time.sleep(.1)
			self.set_config_choice('output', 1)
			time.sleep(.1)
			self.set_config_choice('output', 0)
			time.sleep(12)

		if cmd == "z0":
			if self.camera_file is not None:
				file_data = gp.check_result(gp.gp_file_get_data_and_size(self.camera_file))
				self.camera_file = None
			zoom = 1
			self.set_config_value('eoszoom', '1')
			time.sleep(.1)
			self.set_config_choice('output', 1)
			time.sleep(.1)
			self.set_config_choice('output', 0)
			time.sleep(12)
	
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
	
		if cmd == 'test-capture':
			self.capture_bulb(3)
		
		if cmd == 'capture':
			self.capture_bulb(300, card=True)
			cmdQueue.put('capture-finished')
		
	
	def capture(self):
		while True:
			try:
				if self.camera_file is None:
					for i in range(0,20):
						try:
							self.camera_file = gp.check_result(gp.gp_camera_capture_preview(self.camera, self.context))
							break
						except gp.GPhoto2Error as ex:
							if i < 19:
								continue
						raise
						
				file_data = gp.check_result(gp.gp_file_get_data_and_size(self.camera_file))
	
	
				for i in range(0,20):
					try:
						self.camera_file = gp.check_result(gp.gp_camera_capture_preview(self.camera, self.context))
						break
					except gp.GPhoto2Error as ex:
						if i < 19:
							continue
					raise

				pil_image = Image.open(io.BytesIO(file_data))
				#pil_image.save("testimg2_" + str(i) + ".tif")
				im = np.array(pil_image)
	
				return im
	
			except KeyboardInterrupt:
				break
			except:
				print "Unexpected error:", sys.exc_info()

	
	def __del__(self):
		gp.check_result(gp.gp_camera_exit(self.camera, self.context))
	
	
	
	
if __name__ == "__main__":
	cam = Camera_gphoto()
	cam.prepare()
	cam.capture()