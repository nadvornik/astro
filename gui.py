#!/usr/bin/python

import webserver
import cv2
from PIL import Image
import Queue

class MyGUI_CV2:
	def waitKey(self, timeout):
		return cv2.waitKey(timeout)

	def namedWindow(self, name):
		return cv2.namedWindow(name)
	
	def imshow(self, name, img):
		return cv2.imshow(name, img)
	def __enter__(self):
		pass

	def __exit__(self, type, value, traceback):
		pass
		

class MyGUI_Web:
	def __init__(self):
		self.server = webserver.ServerThread()
		self.server.start()
	
	def waitKey(self, timeout):
		try:
			cmd = webserver.cmdqueue.get(block=True, timeout= timeout/1000.0)
			return ord(cmd)
		except Queue.Empty:
			return 0

	def namedWindow(self, name):
		webserver.mjpeglist.add(name)
	
	def imshow(self, name, img):
		pil_image = Image.fromarray(img)
		webserver.mjpeglist.update(name, pil_image)

	def __enter__(self):
		pass

	def __exit__(self, type, value, traceback):
		self.server.shutdown()

ui = MyGUI_CV2()
#ui = MyGUI_Web()
