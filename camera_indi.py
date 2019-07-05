#!/usr/bin/python
import sys
import time
import select
import os
import random
import math
import cv2
import numpy as np
import io
from astropy.io import fits
import indi_python.indi_base as indi

import logging

log = logging.getLogger()

def apply_gamma(img, gamma):
	lut = np.fromiter( ( (x / 255.0)**gamma * 65535.0 for x in range(256)), dtype=np.uint16 )
	return np.take(lut, img)





class Camera_indi:
	def __init__(self, driver, device, status, focuser = None):
		self.status = status
		self.focuser = focuser
		self.status['exp-sec'] = 0.1
		self.status['test-exp-sec'] = 1
		self.status['exp_in_progress'] = False

		self.status['cur_time'] = 0
		self.status['interrupt'] = False

		self.status['binning'] = 4
		self.status['zoom_shape'] = (768, 1024)
		self.status['zoom_pos'] = [0, 0, 768, 1024]
		
		self.driver = driver
		self.device = device
		self.driver.register(device, msg_type='snoop')

		self.max_width = 1280
		self.max_height = 1024
		
		self.status['mode'] = 'z0'

	def set_mode(self, mode):
		if mode == 'z0':
			self.driver.sendClientMessageWait(self.device, "CCD_BINNING", {'HOR_BIN': self.status['binning'], 'VER_BIN': self.status['binning']})
			self.driver.sendClientMessageWait(self.device, "CCD_STREAM_FRAME", {
				'X': 0,
				'Y': 0,
				'WIDTH': self.max_width // self.status['binning'],
				'HEIGHT': self.max_height // self.status['binning']
			})
			self.status['mode'] = 'z0'
		elif mode == 'z1':
			self.driver.sendClientMessageWait(self.device, "CCD_BINNING", {'HOR_BIN': 1, 'VER_BIN': 1})
			self.driver.sendClientMessageWait(self.device, "CCD_STREAM_FRAME", {
				'X': self.status['zoom_pos'][0] * self.status['binning'],
				'Y': self.status['zoom_pos'][1] * self.status['binning'],
				'WIDTH': self.status['zoom_shape'][1],
				'HEIGHT': self.status['zoom_shape'][0]
			})
			self.status['mode'] = 'z1'
	
	def cmd(self, cmd, x = None, y = None):
		log.info("camera: %s", cmd)
		if cmd.startswith('exp-sec-'):
			self.status['exp-sec'] = float(cmd[len('exp-sec-'):])

		if cmd in ["f-3", "f-2", "f-1", "f+3", "f+2", "f+1"] and self.focuser:
			self.focuser.cmd(cmd)
		
		if cmd == "z1":
			self.driver.sendClientMessageWait(self.device, "CCD_VIDEO_STREAM", {"STREAM_ON": "Off"})
			self.set_mode('z1')
			self.driver.sendClientMessageWait(self.device, "CCD_VIDEO_STREAM", {"STREAM_ON": "On"})
			im, t = self.capture()
			while im is not None and (im.shape[0] != self.status['zoom_shape'][0] or im.shape[1] != self.status['zoom_shape'][1]):
				log.info("zoom shape %s %s", im.shape, self.status['zoom_shape'])
				im, t = self.capture()
			

		if cmd == "z0":
			self.driver.sendClientMessageWait(self.device, "CCD_VIDEO_STREAM", {"STREAM_ON": "Off"})
			self.set_mode('z0')
			
			self.driver.sendClientMessageWait(self.device, "CCD_VIDEO_STREAM", {"STREAM_ON": "On"})
			im, t = self.capture()

		if cmd == "zcenter":
			self.x = (self.max_width - self.status['zoom_shape'][1]) // 2
			self.y = (self.max_height - self.status['zoom_shape'][0]) // 2
	
			self.status['zoom_pos'] = [int(v) for v in [ self.x // self.status['binning'], self.y / self.status['binning'], (self.x + self.status['zoom_shape'][1]) // self.status['binning'], (self.y + self.status['zoom_shape'][0]) // self.status['binning']]]

		if cmd == "zpos":
			self.x = np.clip(x * self.status['binning'] - self.status['zoom_shape'][1] // 2, 0, self.max_width - self.status['zoom_shape'][1])
			self.y = np.clip(y * self.status['binning'] - self.status['zoom_shape'][0] // 2, 0, self.max_height - self.status['zoom_shape'][0])
	
			self.status['zoom_pos'] = [int(v) for v in [ self.x // self.status['binning'], self.y / self.status['binning'], (self.x + self.status['zoom_shape'][1]) // self.status['binning'], (self.y + self.status['zoom_shape'][0]) // self.status['binning']]]

		if cmd.startswith('exp-sec-'):
			try:
				self.status['exp-sec'] = float(cmd[len('exp-sec-'):])
			except:
				pass


	def check_stream(self):
		if self.driver[self.device]["CCD_VIDEO_STREAM"]["STREAM_ON"] == False:
			while True: # empty queue
				msg, prop = self.driver.get(self.device, block=False, msg_type = 'snoop')
				if prop is None:
					break

			self.max_width = self.driver[self.device]["CCD_INFO"]["CCD_MAX_X"].native()
			self.max_height = self.driver[self.device]["CCD_INFO"]["CCD_MAX_Y"].native()
			

			self.driver.sendClientMessageWait(self.device, "CCD_VIDEO_FORMAT", {'ASI_IMG_RAW16': 'On'})
			
#			self.driver.sendClientMessageWait(self.device, "CCD_COLOR_SPACE", {"CCD_COLOR_RGB": "On"})
			self.driver.sendClientMessageWait(self.device, "Stack", {"Mean": "On"})
			
			
			self.cmd('zcenter')

			self.set_mode(self.status['mode'])

			self.driver.sendClient(indi.enableBLOB(self.device, "CCD1"))
			self.driver.sendClientMessageWait(self.device, "CCD_VIDEO_STREAM", {"STREAM_ON": "On"})

	def capture_(self):

		if self.driver[self.device]["CONNECTION"]["CONNECT"] == False:
			self.driver.sendClientMessageWait(self.device, "CONNECTION", {"CONNECT": "On"})
		

		while True:
			self.check_stream()
			msg, prop = self.driver.get(self.device, block=True, timeout=5, msg_type = 'snoop')
			
			if prop is None:
				log.error("wait1 timeout")
				return None, time.time()

			log.error("wait1")
			if prop:
				log.error(prop.getAttr('name'))
			
			
			if prop and prop.getAttr('name') == 'CCD1':
				break
		
		if prop['CCD1'].getAttr('format') == '.stream':
			im = np.ndarray((int(prop['CCD1'].getAttr('size')),), dtype=np.uint8, buffer=prop['CCD1'].native())
			width = int(self.driver[self.device]["CCD_STREAM_FRAME"]["WIDTH"].getValue())
			height = int(self.driver[self.device]["CCD_STREAM_FRAME"]["HEIGHT"].getValue())
			try:
				if self.driver[self.device]["CCD_COLOR_SPACE"]["CCD_COLOR_RGB"] == True:
					im = im.reshape((height, width, 3))
				else:
					im = im.reshape((height, width))
			except:
				im = im.reshape((-1, width))
			log.info("shape %s", im.shape)
		return im, time.time()

	def capture(self):
		while True:
			try:
				return self.capture_()
			except:
				log.exception("capture")
				time.sleep(1)
				return None, time.time()

	def capture_bulb(self, test = False, callback_start = None, callback_end = None):
		while self.driver[self.device]["CCD_VIDEO_STREAM"]["STREAM_ON"] == True:
			self.driver.sendClientMessageWait(self.device, "CCD_VIDEO_STREAM", {"STREAM_ON": "Off"})
			time.sleep(0.1)

		while True: # empty queue
			msg, prop = self.driver.get(self.device, block=False, msg_type = 'snoop')
			log.error(prop)
			if prop is None:
				break

		#self.driver.sendClientMessageWait(self.device, "Stack", {"Mean": "On"})
		#self.driver.sendClientMessageWait(self.device, "CCD_COLOR_SPACE", {"CCD_COLOR_RGB": "On"})
		
		
		self.driver.sendClient(indi.enableBLOB(self.device, "CCD1"))
		self.driver.sendClientMessageWait(self.device, "CCD_BINNING", {'HOR_BIN': 1, 'VER_BIN': 1})

		self.driver.sendClientMessageWait(self.device, "CCD_VIDEO_FORMAT", {'ASI_IMG_RAW16': 'On'})

		self.driver.sendClientMessageWait(self.device, "CCD_FRAME_RESET", {'RESET': 'On'})


		self.driver.sendClientMessageWait(self.device, "CCD_EXPOSURE", {"CCD_EXPOSURE_VALUE": self.status['exp-sec']})
		self.status['exp_in_progress'] = True

		if callback_start is not None:
			try:
				callback_start()
			except:
				log.exception('Unexpected error')



		try:
			while True:
				msg, prop = self.driver.get(self.device, block=True, timeout=10, msg_type = 'snoop')
				log.error(prop)
				if prop and prop.getAttr('name') == 'CCD1' and prop['CCD1'].getAttr('format') == '.fits':
					break
					
				if prop and prop.getAttr('name') == 'CCD_EXPOSURE' and prop.getAttr('state') in ['Alert', 'Idle'] :
					callback_end(None, None)
					self.status['exp_in_progress'] = False
					return

				if prop and prop.getAttr('name') == 'CCD_EXPOSURE' and prop.getAttr('state') == 'Busy':
					self.status['cur_time'] = int(self.status['exp-sec'] - prop["CCD_EXPOSURE_VALUE"])
			
		
			blobfile=io.BytesIO(prop['CCD1'].native())
#			hdulist=fits.open(blobfile)
#			im = hdulist[0].data
#			log.error("shape %s", im.shape) #(3, 720, 1280)
#			im = im[1]

			if callback_end is not None:
#				im = np.array(im, dtype=np.uint8)
#				ret, file_data = cv2.imencode('.jpg', im)
#				file_data = file_data.tobytes()
			
				callback_end(blobfile, "img_{}.fits".format(time.time()))
		except:
			log.exception('Unexpected error')
		self.status['exp_in_progress'] = False


	def shutdown(self):
		self.driver.sendClientMessageWait(self.device, "CCD_VIDEO_STREAM", {"STREAM_ON": "Off"})
		
		self.driver.sendClientMessageWait(self.device, "CCD_BINNING", {'HOR_BIN': 1, 'VER_BIN': 1})

		self.driver.sendClientMessageWait(self.device, "CCD_VIDEO_FORMAT", {'ASI_IMG_RAW16': 'On'})

		self.driver.sendClientMessageWait(self.device, "CCD_FRAME_RESET", {'RESET': 'On'})




if __name__ == "__main__":
	from guide_out import GuideOut
	
	go_ra = GuideOut("./guide_out_ra")
	go_dec = GuideOut("./guide_out_dec")
        
	cam = Camera_test_kstars({}, go_ra, go_dec)
	im, t = cam.capture()
        