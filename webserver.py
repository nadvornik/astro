#!/usr/bin/python

from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
import cgi
import threading
from PIL import Image
import StringIO
import time
import os
from cmd import cmdQueue

class MjpegBuf:
	def __init__(self):
		self.condition = threading.Condition()
		self.buf = None
		self.encoded = True

	def update(self, pil_image):
		with self.condition:
			self.buf = pil_image
			self.encoded = False
			self.condition.notify_all()

	def update_jpg(self, jpg):
		with self.condition:
			self.buf = jpg.getvalue()
			self.encoded = True
			self.condition.notify_all()

	def serve(self, handler):
		with self.condition:
			i = 0
			while True:
				handler.wfile.write("--jpegBoundary\r\n")
				if self.buf is None or i > 0:
					self.condition.wait()
				if not self.encoded:
					tmpFile = StringIO.StringIO()
					self.buf.save(tmpFile,'JPEG')
					self.buf = tmpFile.getvalue()
					self.encoded = True
					
				l = len(self.buf)
				handler.send_header('Content-type','image/jpeg')
				handler.send_header('Content-length',str(l))
				handler.end_headers()
				handler.wfile.write(self.buf)
				i +=  1

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
	
	def update(self, name, pil_image):
		self.dict[name].update(pil_image)

	def update_jpg(self, name, jpg):
		self.dict[name].update_jpg(jpg)



class Handler(BaseHTTPRequestHandler):
    
	def do_GET(self):
		base = os.path.basename(self.path)
		name, ext = os.path.splitext(base)
		print name, ext
		if self.path == '/':
			self.send_response(301)
			self.send_header('Location', 'index.html')
			self.end_headers()
			return
		elif ext == '.mjpg':
			print name
			mjpeg = mjpeglist.get(name)
			if mjpeg is None:
				self.send_response(404)
				self.end_headers()
				return
			
			self.send_response(200)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpegBoundary')
			self.end_headers()
			mjpeg.serve(self)
			return
		elif ext == '.html':
			self.send_response(200)
			self.send_header('Content-type','text/html')
			self.end_headers()
			f = open(base)
			self.wfile.write(f.read())
			f.close()
			return
		elif base == 'jquery.min.js':
			self.send_response(200)
			self.send_header('Content-type','application/javascript')
			self.end_headers()
			f = open(base)
			self.wfile.write(f.read())
			f.close()
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
			cmdQueue.put(postvars['key'][0])
			return
		self.send_response(404)
		self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class ServerThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.server = ThreadedHTTPServer(('', 8080), Handler)
		self.server.daemon_threads = True

	def run(self):
		print 'Starting server, use <Ctrl-C> to stop'
		self.server.serve_forever()

	def shutdown(self):
		self.server.shutdown()
		
mjpeglist = MjpegList()

if __name__ == '__main__':
	server = ServerThread()
	server.start()
	
	mjpeglist.add('cam')

	i = 0
	while True:
		pil_image = Image.open("testimg12_" + str(i % 5 * 4 + 3) + ".tif")
		mjpeglist.update('cam', pil_image)
		time.sleep(1)

		i += 1
