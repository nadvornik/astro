#!/usr/bin/python
import gphoto2 as gp
import subprocess
import logging
import os
import sys
import time
import io
import numpy as np
import cv2

from gui import ui
from cmd import cmdQueue
from stacktraces import stacktraces

import gc

log = logging.getLogger()

def apply_gamma(img, gamma):
	lut = np.fromiter( ( (x / 255.0)**gamma * 65535.0 for x in xrange(256)), dtype=np.uint16 )
	return np.take(lut, img)



class Camera_gphoto:
	def __init__(self, status, focuser = None):
		self.status = status
		self.status.setdefault('iso', 400)
		self.status.setdefault('test-iso', 3200)
		self.status.setdefault('exp-sec', 120)
		self.status.setdefault('test-exp-sec', 15)
		self.status.setdefault('f-number', '5.6')
		self.status['cur_time'] = 0
		self.status['exp_in_progress'] = False
		self.status['interrupt'] = False
		self.focuser = focuser
		self.fpshack = ''
		self.fpshackiso = 0
		self.status.setdefault('capture_idx', 0)
		self.status['capture'] = False
		self.status.setdefault('capture_path', None)

	def get_config_value(self, name):
		config = gp.check_result(gp.gp_camera_get_config(self.camera, self.context))
		OK, widget = gp.gp_widget_get_child_by_name(config, name)
		if OK >= gp.GP_OK:
			# set value
			value = gp.check_result(gp.gp_widget_get_value(widget))
			log.info("get %s => %s", name, value)
			return value

	def set_config_choice(self, name, num):
		for t in range(0, 20):
			try:
				config = gp.check_result(gp.gp_camera_get_config(self.camera, self.context))
				OK, widget = gp.gp_widget_get_child_by_name(config, name)
				if OK >= gp.GP_OK:
					# set value
					value = gp.check_result(gp.gp_widget_get_choice(widget, num))
					log.info("set %s => %s", name, value)
					gp.check_result(gp.gp_widget_set_value(widget, value))
				# set config
				gp.check_result(gp.gp_camera_set_config(self.camera, config, self.context))
				break
			except gp.GPhoto2Error as ex:
				log.exception('failed')
				time.sleep(0.1)
				continue

	def set_config_value(self, name, value):
		for t in range(0, 20):
			try:
				config = gp.check_result(gp.gp_camera_get_config(self.camera, self.context))
				OK, widget = gp.gp_widget_get_child_by_name(config, name)
				if OK >= gp.GP_OK:
					log.info("set %s => %s", name, value)
					# set value
					gp.check_result(gp.gp_widget_set_value(widget, value))
				# set config
				gp.check_result(gp.gp_camera_set_config(self.camera, config, self.context))
				break
			except gp.GPhoto2Error as ex:
				log.exception('failed')
				time.sleep(0.1)
				continue

	def set_config_value_checked(self, name, value):
		value = str(value)
		ret = False
		for t in range(0, 20):
			try:
				config = gp.check_result(gp.gp_camera_get_config(self.camera, self.context))
				OK, widget = gp.gp_widget_get_child_by_name(config, name)
				
				if OK >= gp.GP_OK:
					num = None
					choice_count = gp.check_result(gp.gp_widget_count_choices(widget))
					log.info("count %d", choice_count)
					for i in range(choice_count):
						vi = gp.check_result(gp.gp_widget_get_choice(widget, i))
						if vi.lower() == value.lower():
							num = i
							value = vi
							break
						try:
							if abs(float(vi) - float(value)) < 0.000001:
								value = vi
								num = i
								break
						except ValueError:
							pass
						try:
							if '/' in vi:
								fr = vi.split('/')
								fr = float(fr[0]) / float(fr[1])
								if abs(fr - float(value)) < abs(fr * 0.001):
									value = vi
									num = i
									break
						except:
							pass
					
					if num is not None:
						log.info("set %s => %s (choice %d)" % (name, value, num))
						# set value
						gp.check_result(gp.gp_widget_set_value(widget, value))
						ret = True
					else:
						log.info("cant't set %s => %s" % (name, value))
				# set config
				gp.check_result(gp.gp_camera_set_config(self.camera, config, self.context))
				break
			except gp.GPhoto2Error as ex:
				log.exception('failed')
				time.sleep(0.1)
				ret = False
				continue
		return ret
	
	def do_fps_hack(self):
		if self.status['exp-sec'] < 1.0:
			return

		if self.fpshack == 'output':
			time.sleep(.2)
			self.set_config_choice('output', 1)
			time.sleep(.2)
			self.set_config_choice('output', 0)
			time.sleep(3)
		elif self.fpshack == 'iso':
			self.fpshackiso = 5
			

	def capture_bulb(self, test = False, callback_start = None, callback_end = None):
		if test:
			sec = self.status['test-exp-sec']

			self.set_config_value_checked('iso', self.status['test-iso'])
			try:
				self.status['test-iso'] = int(self.get_config_value('iso'))
			except:
				pass
			#self.set_config_choice('capturetarget', 0) #mem
			self.set_config_choice('capturetarget', 1) #card
			self.set_config_choice('imageformat', 1) #Large Normal JPEG
		else:
			sec = self.status['exp-sec']

			self.set_config_value_checked('iso', self.status['iso'])
			try:
				self.status['iso'] = int(self.get_config_value('iso'))
			except:
				pass
			self.set_config_choice('capturetarget', 1) #card
			#self.set_config_choice('imageformat', 24) #RAW 
			self.set_config_choice('imageformat', 7) #RAW + Large Normal JPEG 

		self.set_config_value('aperture', self.status['f-number'])
		self.status['f-number'] = self.get_config_value('aperture')

		if callback_start is not None:
			try:
				callback_start()
			except:
				log.exception('Unexpected error')					


		self.t_start = time.time()
		while True:
			e, file_path =  gp.check_result(gp.gp_camera_wait_for_event(self.camera, 100,self.context))
			t = time.time() - self.t_start
			log.info("camera event %f %s %s", t, e, file_path)
			if t > 1 or file_path is None:
				break


		if sec <= 4:
			bulbmode = None
			self.set_config_value_checked('autoexposuremode', 'Manual')
			self.set_config_value_checked('shutterspeed', sec)
			for t in range(0, 1):
				try:
					log.info("trgger capture")
					gp.check_result(gp.gp_camera_trigger_capture(self.camera, self.context))
				except gp.GPhoto2Error as ex:
					log.exception('failed')
					time.sleep(0.1)
					continue


		else:
			self.set_config_value_checked('autoexposuremode', 'Manual')
			if not self.set_config_value_checked('shutterspeed', 'Bulb'):
				self.set_config_value_checked('autoexposuremode', 'Bulb')
		
			bulbmode = 'eosremoterelease'
			if not self.set_config_value_checked('eosremoterelease', 'Immediate'):
				self.set_config_value('bulb', 1)
				bulbmode = 'bulb'
		self.t_start = time.time()
		t = 0
		self.status['exp_in_progress'] = True
		self.status['interrupt'] = False
		while True:
			#if t < sec - 4 and not self.status['interrupt']:
			#	time.sleep(3)
			e, file_path =  gp.check_result(gp.gp_camera_wait_for_event(self.camera, 1000,self.context))
			t = time.time() - self.t_start
			log.info("camera event %f %s %s", t, e, file_path)
			try:
				ev = str(file_path)
				if ev.startswith("BulbExposureTime "):
					tp = int(ev[len("BulbExposureTime "):])
					log.info("exp time parsed %f", tp)
					self.t_start += t - tp
					t = tp
			except:
				pass
			
			if self.status['exp_in_progress']:
				self.status['cur_time'] = int(t)

			if self.status['exp_in_progress'] and (t >= sec or self.status['interrupt']):
				if bulbmode == 'bulb':
					self.set_config_value('bulb', 0)
				elif  bulbmode == 'eosremoterelease':
					self.set_config_value_checked('eosremoterelease', 'Release Full')
				self.status['exp_in_progress'] = False

			if not self.status['exp_in_progress']:
				log.info("waiting for image")
			
			if e == gp.GP_EVENT_FILE_ADDED:
				log.info("filepath: %s %s", file_path.folder, file_path.name)
				filename, file_extension = os.path.splitext(file_path.name)
				if file_extension == ".jpg" or file_extension == ".JPG":
					break
			if t > sec + 60:
				file_path = None
				log.info("image timeout")
				break
		
		self.status['exp_in_progress'] = False
		self.status['cur_time'] = 0
		self.status['interrupt'] = False
			
		file_data = None
		filename = None
		if file_path is not None:
			filename = file_path.name
			target = os.path.join('/tmp', file_path.name)
			n = 20
			while True:
				n -= 1
				try:
					camera_file = gp.check_result(gp.gp_camera_file_get(self.camera, file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL, self.context))
					file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
					break
				except gp.GPhoto2Error as ex:
					if ex.code == gp.GP_ERROR_CAMERA_BUSY:
						time.sleep(1)
	
						if (n > 0):
							continue

		if callback_end is not None:
			try:
				callback_end(memoryview(file_data).tobytes(), filename)
			except:
				log.exception('Unexpected error')					
					

		
		log.info('callback end')
		#stop review on display
		self.set_config_value_checked('eosremoterelease', 'Press Half')
		self.set_config_value_checked('eosremoterelease', 'Release Half')
		log.info('review end')
	
		self.do_fps_hack()
		self.set_config_value_checked('iso', 100)

		log.info('fps hack end')
		
	def prepare(self):
		self.shape = (704, 1056)
		self.zoom_shape = (680, 1024)
	
		subprocess.call(['killall', 'gvfsd-gphoto2'])
		subprocess.call(['killall', 'gvfs-gphoto2-volume-monitor'])
	
		gp.check_result(gp.use_python_logging())
		self.context = gp.gp_context_new()
		while True:
			try:
				self.camera = gp.check_result(gp.gp_camera_new())
				gp.check_result(gp.gp_camera_init(self.camera, self.context))
				break
			except gp.GPhoto2Error as ex:
				log.info("gphoto2 camera is not ready")
				time.sleep(2)
				continue
		
		# wake up the camera
		self.set_config_value_checked('eosremoterelease', 'Press Half')
		self.set_config_value_checked('eosremoterelease', 'Release Half')
		self.set_config_value_checked('eosremoterelease', 'Release Full')
		self.set_config_value('bulb', 0)
		
		self.cameramodel = self.get_config_value('cameramodel')
		log.info(self.cameramodel)
		if self.cameramodel == "Canon EOS 40D":
			self.shape = (680, 1024)
			self.zoom_shape = (800, 768)
			self.fpshack = 'iso'
			self.zoom_scale = 3.8
		elif self.cameramodel == "Canon EOS 7D":
			self.shape = (704, 1056)
			self.zoom_shape = (680, 1024)
	                self.fpshack = 'output'
	                self.zoom_scale = 5

		
		
		cur_time = self.get_config_value('datetime')
		if cur_time is None:
			cur_time = self.get_config_value('datetimeutc')
		try:
			cur_time = int(cur_time)
			if cur_time - time.time() > 1500:
				log.info("adjusting time %s %s", time.time(), cur_time)
				subprocess.call(['date', '--set', '@' + str(cur_time) ])
		except:
			pass
			
		self.set_config_value_checked('aperture', self.status['f-number'])
		self.status['f-number'] = self.get_config_value('aperture')
		self.status['lensname'] = self.get_config_value('lensname')
		
		self.set_config_choice('drivemode', 0)
		
		self.set_config_value_checked('autoexposuremode', 'Manual')
		if not self.set_config_value_checked('shutterspeed', 'Bulb'):
			self.set_config_value_checked('autoexposuremode', 'Bulb')
		
		# required configuration will depend on camera type!
		self.set_config_choice('capturesizeclass', 2)
	
		self.do_fps_hack()
	
		self.zoom = 1
		self.x = (self.shape[1] * self.zoom_scale - self.zoom_shape[1]) / 2
		self.y = (self.shape[0] * self.zoom_scale - self.zoom_shape[0]) / 2
		
		
		self.status['zoom_pos'] = [int(v) for v in [ self.x / self.zoom_scale, self.y / self.zoom_scale, (self.x + self.zoom_shape[1]) / self.zoom_scale, (self.y + self.zoom_shape[0]) / self.zoom_scale]]
		
		self.set_config_value('eoszoomposition', "%d,%d" % (self.x, self.y))
		time.sleep(3)
	
	def cmd(self, cmd, x = None, y = None):
		try:
			if cmd in ["f-3", "f-2", "f-1", "f+3", "f+2", "f+1"]:
				if self.focuser is not None:
					self.focuser.cmd(cmd)
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
							
				#self.x = x * zoom - self.zoom_shape[1] / 2
				#self.y = y * zoom - self.zoom_shape[0] / 2
				self.set_config_value('eoszoomposition', "%d,%d" % (self.x, self.y))
				self.set_config_value('eoszoom', '5')
				time.sleep(.2)
				self.do_fps_hack()
				im, t = self.capture()
				while im.shape[0] != self.zoom_shape[0] or im.shape[1] != self.zoom_shape[1]:
					log.info("zoom shape %s %s", im.shape, self.zoom_shape)
					im, t = self.capture()
				
        
			if cmd == "z0":
				zoom = 1
				self.set_config_value('eoszoom', '1')
				time.sleep(.2)
				self.do_fps_hack()
				im, t = self.capture()
				while im.shape[0] != self.shape[0] or im.shape[1] != self.shape[1]:
					log.info("shape %s %s", im.shape, self.shape)
					im, t = self.capture()

			if cmd == "zcenter":
				self.x = (self.shape[1] * self.zoom_scale - self.zoom_shape[1]) / 2
				self.y = (self.shape[0] * self.zoom_scale - self.zoom_shape[0]) / 2
		
				self.status['zoom_pos'] = [int(v) for v in [ self.x / self.zoom_scale, self.y / self.zoom_scale, (self.x + self.zoom_shape[1]) / self.zoom_scale, (self.y + self.zoom_shape[0]) / self.zoom_scale]]

			if cmd == "zpos":
				self.x = x * self.zoom_scale - self.zoom_shape[1] / 2
				self.y = y * self.zoom_scale - self.zoom_shape[0] / 2
		
				self.status['zoom_pos'] = [int(v) for v in [ self.x / self.zoom_scale, self.y / self.zoom_scale, (self.x + self.zoom_shape[1]) / self.zoom_scale, (self.y + self.zoom_shape[0]) / self.zoom_scale]]

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
				self.status['iso'] = cmd[len('iso-'):]
        
			if cmd.startswith('test-iso-'):
				self.status['test-iso'] = cmd[len('test-iso-'):]
        
			if cmd.startswith('exp-sec-'):
				try:
					self.status['exp-sec'] = float(cmd[len('exp-sec-'):])
				except:
					pass
        
			if cmd.startswith('test-exp-sec-'):
				try:
					self.status['test-exp-sec'] = float(cmd[len('test-exp-sec-'):])
				except:
					pass
			
			if cmd.startswith('f-number-'):
				self.set_config_value('aperture', cmd[len('f-number-'):])
				self.status['f-number'] = self.get_config_value('aperture')


			if cmd == 'capture_start' and self.status['capture_path'] is not None:
				self.status['capture'] = True
			elif cmd == 'capture_stop':
				self.status['capture'] = False
		

			
		except gp.GPhoto2Error as ex:
			log.exception('Unexpected error')
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
				if self.status['capture'] and self.status['capture_path'] is not None:
					i = self.status['capture_idx']
					while os.path.isfile(self.status['capture_path'] + 'capt%04d.jpg' % i):
						i += 1
					f = open(self.status['capture_path'] + 'capt%04d.jpg' % i, "wb")
					f.write(file_data)
					f.close()

					log.info("saved {}".format(i))
					i += 1
					self.status['capture_idx'] = i

				im = cv2.imdecode(np.fromstring(memoryview(file_data).tobytes(), dtype=np.uint8), -1)
				im = apply_gamma(im, 2.2)
				if self.fpshackiso > 0:
					self.set_config_value_checked('iso', 1600)
					self.set_config_value_checked('iso', 100)
					self.fpshackiso -= 1

				return im, time.time()
	
			except KeyboardInterrupt:
				break
			except gp.GPhoto2Error as ex:
				log.exception('Unexpected error')
				time.sleep(1)
				if ex.code == -7 or ex.code == -1 or ex.code == -52:
					gp.gp_camera_exit(self.camera, self.context)
					self.prepare()

	
	def shutdown(self):
		time.sleep(1)
		self.set_config_value_checked('eosremoterelease', 'Release Full')
		self.set_config_value('viewfinder', 0)
		gp.check_result(gp.gp_camera_exit(self.camera, self.context))
		
	
	
	
	
if __name__ == "__main__":
	cam = Camera_gphoto()
	cam.prepare()
	cam.capture()