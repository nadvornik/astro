#!/usr/bin/python

import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import cgi
import threading
import io
import time
import os
import sys
from cmd import cmdQueue
import subprocess
from stacktraces import stacktraces
import exceptions
import logging

log = logging.getLogger()

class MjpegBuf:
	def __init__(self):
		self.condition = threading.Condition()
		self.buf = None
		self.encoded = True
		self.seq = -1

	def update(self, image):
		with self.condition:
			self.buf = image
			self.encoded = False
			self.condition.notify_all()
			self.seq += 1

	def update_jpg(self, jpg):
		with self.condition:
			self.buf = jpg
			self.encoded = True
			self.condition.notify_all()
			self.seq += 1

	def serve(self, handler, seq):
	        t0 = time.time()
		with self.condition:
			while self.buf is None or seq == self.seq + 1:
				if time.time() > t0 + 80:
					log.info("req timeout")
					raise exceptions.EOFError()
				self.condition.wait(30)
			if not self.encoded:
				ret, file_data = cv2.imencode('.jpg', self.buf)
                                self.buf = file_data.tobytes()
                                del file_data
				self.encoded = True
				
			buf = self.buf
			seq = self.seq
				
		l = len(buf)
		handler.send_header('Content-type','image/jpeg')
		handler.send_header('Content-length',str(l))
		handler.send_header('X-seq',str(seq))
		handler.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
		handler.send_header('Pragma', 'no-cache')
		handler.send_header('Expires', '0')
		handler.end_headers()
		handler.wfile.write(buf)
		handler.wfile.flush()


class MjpegList:
	def __init__(self):
		self.dict = {}
	
	def add(self, name):
		self.dict[name] = MjpegBuf()
	
	def get(self, name):
		if name in self.dict:
			return self.dict[name]
		else:
			return None
	
	def update(self, name, image):
		self.dict[name].update(image)

	def update_jpg(self, name, jpg):
		self.dict[name].update_jpg(jpg)



class Handler(BaseHTTPRequestHandler):

	wbufsize = 1024 * 1024
	timeout = 120

	def do_GET(self):
		s_path = self.path.split('?')
		path = self.path
		args = ''
		if len(s_path) == 2:
			path, args = s_path
		
		base = os.path.basename(path)
		name, ext = os.path.splitext(base)
		if self.path == '/':
			self.send_response(301)
			self.send_header('Location', 'index.html')
			self.end_headers()
			return
		elif ext == '.jpg':
			s_args = args.split('=')
			if len(s_args) == 2 and s_args[0] == 'seq':
				seq = int(s_args[1])
			else:
				seq = 0
			
			mjpeg = mjpeglist.get(name)
			if mjpeg is None:
				self.send_response(404)
				self.end_headers()
				return
			
			self.send_response(200)
			mjpeg.serve(self, seq)
			return
		elif base == 'log.html':
			self.send_response(200)
			self.send_header('Content-type','text/html')
			self.end_headers()
			self.wfile.write("<html><head><title>Log</title></head>")
			self.wfile.write("<body><pre>")
			self.wfile.write(subprocess.check_output(['journalctl', '-u', 'navigate.service', '-n', '300']))
			self.wfile.write("</bre></body></html>")
			return
		elif base == 'status.json':
			global status
			if status is not None:
				self.send_response(200)
				self.send_header('Content-type','application/json')
				self.end_headers()
				self.wfile.write(status.to_json())
			else:
				self.send_response(404)
				self.end_headers()
				return
			return
		elif ext == '.html' or ext == '.js' or  ext == '.css':
			try:
				f = open(base)
			except:
				self.send_response(404)
				self.end_headers()
				return
			c = f.read()
			f.close()
			self.send_response(200)
			if ext == '.html':
				self.send_header('Content-type','text/html; charset=utf-8')
			elif ext == '.js':
				self.send_header('Content-type','application/javascript')
			elif ext == '.css':
				self.send_header('Content-type','text/css')
			self.end_headers()
			self.wfile.write(c)
			return
		self.send_response(404)
		self.end_headers()

	def do_POST(self):
		ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
		if ctype == 'multipart/form-data':
			postvars = cgi.parse_multipart(self.rfile, pdict)
		elif ctype == 'application/x-www-form-urlencoded':
			length = int(self.headers.getheader('content-length'))
			postvars = cgi.parse_qs(self.rfile.read(length), keep_blank_values=1)
		else:
			postvars = {}
	
		base = os.path.basename(self.path)
		if base == 'button':
			self.send_response(200)
			self.send_header('Content-type','text/text')
			self.end_headers()
			self.wfile.write("ok")
			
			cmd = postvars['cmd'][0]
			try:
				tgt = postvars['tgt'][0]
			except:
				tgt = None
			if cmd == 'exit' or cmd == 'shutdowm' or cmd == 'stacktrace':
				stacktraces()
			cmdQueue.put(cmd, target = tgt)
			return
		self.send_response(404)
		self.end_headers()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
	"""Handle requests in a separate thread."""

	def handle_error(self, request, client_address):
		ex, val = sys.exc_info()[:2]
		#if 'Broken pipe' not in val:
		log.exception('Unexpected error')
		#pass

class ServerThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.server = ThreadedHTTPServer(('', 8080), Handler)
		self.server.daemon_threads = True
		self.daemon = True

	def run(self):
		self.server.serve_forever()

	def shutdown(self):
		time.sleep(10)
		self.server.shutdown()
		
mjpeglist = MjpegList()
status = None




if __name__ == '__main__':
	server = ServerThread()
	server.start()
	
	mjpeglist.add('capture')

	i = 0
	while True:
		pil_image = Image.open("testimg2_" + str(i % 5 * 4 + 3) + ".tif")
		mjpeglist.update('capture', pil_image)
		time.sleep(10)

		i += 1
