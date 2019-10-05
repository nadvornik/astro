import time
import sys

import indi_python.indi_base as indi

import logging

log = logging.getLogger()

class FocuserIndi:
	def __init__(self, driver, device):
		self.driver = driver
		self.device = device
		self.pos0 = None
	
	
	def cmd(self, cmd):
		try:
			if self.driver.checkValue(self.device, "CONNECTION", "CONNECT") == 'Off':
				self.driver.sendClientMessageWait(self.device, "CONNECTION", {"CONNECT": "On"})
			
		
			if cmd == "f-3":
				self.driver.sendClientMessage(self.device, "FOCUS_MOTION", {"FOCUS_INWARD": "On"})
				self.driver.sendClientMessage(self.device, "REL_FOCUS_POSITION", {"FOCUS_RELATIVE_POSITION": 384})
			if cmd == "f-2":
				self.driver.sendClientMessage(self.device, "FOCUS_MOTION", {"FOCUS_INWARD": "On"})
				self.driver.sendClientMessage(self.device, "REL_FOCUS_POSITION", {"FOCUS_RELATIVE_POSITION": 128})
			if cmd == "f-1":
				self.driver.sendClientMessage(self.device, "FOCUS_MOTION", {"FOCUS_INWARD": "On"})
				self.driver.sendClientMessage(self.device, "REL_FOCUS_POSITION", {"FOCUS_RELATIVE_POSITION": 16})
			if cmd == "f+3":
				self.driver.sendClientMessage(self.device, "FOCUS_MOTION", {"FOCUS_OUTWARD": "On"})
				self.driver.sendClientMessage(self.device, "REL_FOCUS_POSITION", {"FOCUS_RELATIVE_POSITION": 384})
			if cmd == "f+2":
				self.driver.sendClientMessage(self.device, "FOCUS_MOTION", {"FOCUS_OUTWARD": "On"})
				self.driver.sendClientMessage(self.device, "REL_FOCUS_POSITION", {"FOCUS_RELATIVE_POSITION": 128})
			if cmd == "f+1":
				self.driver.sendClientMessage(self.device, "FOCUS_MOTION", {"FOCUS_OUTWARD": "On"})
				self.driver.sendClientMessage(self.device, "REL_FOCUS_POSITION", {"FOCUS_RELATIVE_POSITION": 16})
		except:
			log.exception("focuser move")

	def get_pos(self):
		try:
			if self.driver.checkValue(self.device, "CONNECTION", "CONNECT") == 'Off':
				self.driver.sendClientMessageWait(self.device, "CONNECTION", {"CONNECT": "On"})

			pos = int(self.driver[self.device]["ABS_FOCUS_POSITION"]["FOCUS_ABSOLUTE_POSITION"].getValue())
			if self.pos0 is None:
				self.pos0 = pos
			return pos - self.pos0
		except:
			log.exception("focuser pos")
			return 0
