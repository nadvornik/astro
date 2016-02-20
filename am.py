import cv2
import numpy as np

import os
import sys
import subprocess
import signal

import pyfits
from astrometry.util.util import Tan, anwcs_new_tan
from astrometry.blind.plotstuff import *
import threading
import math
import tempfile
import shutil
import time
import Queue
import atexit

class Engine(threading.Thread):
	def __init__(self, conf, queue):
		threading.Thread.__init__(self)
		self.daemon = True
		self.conf = conf
		self.cmd = None
		self.ready = True
		self.queue = queue
		self.terminating = False
		self.start()

	

	def run(self):
		#log = open(self.conf + self.name + ".log", "w")
		while True:
			if self.cmd is None or self.cmd.poll() is not None:
				restart = self.cmd is not None
				if restart:
					print "%s engine exited with %d\n" % (self.conf, self.cmd.poll()),
				#log.write("<<<\n")
				#log.close()
				if self.terminating:
					return
				args = ['astrometry-engine', '--config', self.conf, '-f', '-', '-v' ]
				self.cmd = subprocess.Popen(args, close_fds=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1 )
				if not restart:
					atexit.register(self.terminate)
				if not self.ready:
					self.ready = True
					self.queue.put(self) # back to free engines queue
				
			line = self.cmd.stdout.readline()
			#log.write(line)
			if "seconds on this field" in line:
				print line
				#log.write(">>>\n")
				self.ready = True
				self.queue.put(self) # back to free engines queue
		
			
	
	def solve(self, axy):
		self.ready = False
		while True:
			try:
				self.cmd.stdin.write(axy + "\n")
				break
			except:
				print "Error: " +  sys.exc_info().__str__()
				time.sleep(0.1)
		
	
	def check(self):
		return self.ready
	
	def terminate(self):
		self.terminating = True
		if self.cmd is not None and self.cmd.poll() is None:
			self.cmd.terminate()
			time.sleep(0.2)
		if self.cmd is not None and self.cmd.poll() is None:
			self.cmd.kill()
		self.join()
	
		

class EnginePool:
	def __init__(self):
		self.engines = {}
		for (conf, n) in [('conf-all', 2), ('conf-41', 1), ('conf-42-1', 1), ('conf-42-2', 1) ]:
			self.engines[conf] = Queue.Queue()
			for i in range(0, n):
				engine = Engine(conf + ".cfg", self.engines[conf])
				self.engines[conf].put(engine)

			
		
	def get(self, conf):
		return self.engines[conf].get(block = True)
	
engines = EnginePool()



class Solver(threading.Thread):
	def __init__(self, sources_list = None, field_w = None, field_h = None, ra = None, dec = None, field_deg = None, radius = None):
		threading.Thread.__init__(self)
		self.sources_list = sources_list
		self.field_w = field_w
		self.field_h = field_h
		self.ra = ra
		self.dec = dec
		self.field_deg = field_deg
		self.radius = radius
		if self.ra is not None and field_deg is not None and radius is None:
			self.radius = field_deg * 2.0
		self.solved = False
		self.engines = []
		
	
	def run(self):
		tmp_dir = tempfile.mkdtemp()
		tbhdu = pyfits.BinTableHDU.from_columns([
			pyfits.Column(name='X', format='E', array=self.sources_list[:, 1]),
			pyfits.Column(name='Y', format='E', array=self.sources_list[:, 0]),
			pyfits.Column(name='FLUX', format='E', array=self.sources_list[:, 2])
			])
		prihdr = pyfits.Header()
		prihdr['IMAGEW'] = self.field_w
		prihdr['IMAGEH'] = self.field_h
		
		prihdr['ANRUN'] = True
		prihdr['ANVERUNI'] = True
		prihdr['ANVERDUP'] = False
		prihdr['ANCRPIXC'] = True
		prihdr['ANTWEAK'] = True
		prihdr['ANTWEAKO'] = 1
		prihdr['ANSOLVED'] = tmp_dir + '/field.solved'
		#prihdr['ANMATCH'] = tmp_dir + '/field.match'
		prihdr['ANRDLS'] = tmp_dir + '/field.rdls'
		prihdr['ANWCS'] = tmp_dir + '/field.wcs'
		#prihdr['ANCORR'] = tmp_dir + '/field.corr'
		prihdr['ANCANCEL'] = tmp_dir + '/field.solved'
		
		
		prihdr['ANPOSERR'] = 2
		if self.ra is not None:
			prihdr['ANERA'] = self.ra
			prihdr['ANEDEC'] = self.dec
			if self.radius is not None and self.radius > 0:
				prihdr['ANERAD'] = self.radius
			
		prihdr['ANDPL1'] = 1
		prihdr['ANDPU1'] = 20
		
		if self.field_deg is not None:
			prihdr['ANAPPL1'] = self.field_deg * 0.95 * 3600 / self.field_w
			prihdr['ANAPPU1'] = self.field_deg * 1.05 * 3600 / self.field_w
		
		if self.radius is not None and self.radius > 0 and self.radius < 5:
			prihdr['ANODDSSL'] = 1e6
		else:
			prihdr['ANODDSSL'] = 1e8
		
		
		prihdu = pyfits.PrimaryHDU(header=prihdr)
		thdulist = pyfits.HDUList([prihdu, tbhdu])
		thdulist.writeto(tmp_dir + '/field.axy', clobber=True)
	
		if self.radius is not None and self.radius > 0 and self.radius < 5:
			conf_list = ['conf-all']
		else:
			conf_list = ['conf-41', 'conf-42-1', 'conf-42-2' ]
		
		self.cancel_file = tmp_dir + '/field.solved'
		
		solved = None
		global engines
		for conf in conf_list:
			engine = engines.get(conf)
			engine.solve(tmp_dir + '/field.axy')
			self.engines.append(engine)
		
		while True:
			running = False
			for engine in self.engines:
				if not engine.check():
					running = True
			if not running:
				break
			time.sleep(0.1)
		
		if os.path.exists(tmp_dir + '/field.wcs'):
			solved = tmp_dir + '/field'

		self.cancel_file = None

		if solved is None or not os.path.exists(solved + ".wcs"):
			self.ra = None
			self.dec = None
			self.field_deg = None
			shutil.rmtree(tmp_dir)
			return
	
		self.wcs = Tan(solved + ".wcs", 0)
		self.ra, self.dec = self.wcs.radec_center()
		self.field_deg = self.field_w * self.wcs.pixel_scale() / 3600
		
		ind = pyfits.open(solved + '.rdls')
		tbdata = ind[1].data
		self.ind_sources = []
		self.ind_radec = []
		for l in tbdata:
			ok, x, y = self.wcs.radec2pixelxy(l['ra'], l['dec'])
			x = np.clip(int(x), 0, self.field_w - 1)
			y = np.clip(int(y), 0, self.field_h - 1)
			self.ind_sources.append((x,y))
			self.ind_radec.append((l['ra'], l['dec']))
		shutil.rmtree(tmp_dir)
		self.solved = True


	def terminate(self, wait = True):
		
		try:
			with open(self.cancel_file, 'a'):
				pass
		except:
			pass
		
		if wait:
			self.join()
		

class Plotter:
	def __init__(self, wcs):
		self.wcs = wcs

	def plot(self, img = None, off = [0., 0.], extra = [], extra_lines = [], grid = True, scale = 1):
		field_w = self.wcs.get_width()
		field_h = self.wcs.get_height()

		try:
			scale = float(scale)
		except ValueError:
			if scale.startswith('deg'):
				scale = float(scale[3:]) / (self.wcs.pixel_scale() * field_w / 3600.0)
			else:
				raise

		# the field moved by given offset pixels from the position in self.wcs
		(crpix1, crpix2) = self.wcs.crpix
		ra2, dec2 = self.wcs.pixelxy2radec(crpix1 - off[1], crpix2 - off[0])
		

		# plot extra area with the original image in the center
		xborder = field_w * (scale - 1) / 2.0
		yborder = field_h * (scale - 1) / 2.0
		nw = field_w * scale
		nh = field_h * scale
	
		new_wcs = Tan(self.wcs)
		new_wcs.set_crval(ra2, dec2)

		new_wcs.set_width(nw)
		new_wcs.set_height(nh)
		
		new_wcs.set_crpix(crpix1 + xborder, crpix2 + yborder)

		#thumbnail size and pos
		tw = int(field_w / scale)
		th = int(field_h / scale)
		tx = int(xborder / scale)
		ty = int(yborder / scale)


		
		plot = Plotstuff(outformat=PLOTSTUFF_FORMAT_PPM)
		plot.wcs = anwcs_new_tan(new_wcs)
		plot.scale_wcs(1.0 / scale)

		plot.set_size_from_wcs()

		plot.color = 'black'
		plot.alpha = 1.0
		plot.plot('fill')

		plot.color = 'gray'
		
		plot_area_deg = new_wcs.pixel_scale() * scale * field_w / 3600.0
		#print "plot area", plot_area_deg
		
		
		if grid:
			log_step = [1, 2, 5];
			log_ra = int(np.floor(np.log10(plot_area_deg / (0.01 + math.cos(math.radians(dec2)))) * 3)) - 1
			log_dec = int(np.floor(np.log10(plot_area_deg) * 3)) - 1
			grid_step_ra = 10 ** int(log_ra / 3) * log_step[log_ra % 3]
			grid_step_dec = 10 ** int(log_dec / 3) * log_step[log_dec % 3]
			grid_step_ra = min(10, grid_step_ra)
			grid_step_dec = min(10, grid_step_dec)
			#print "grid", plot_area_deg, log_ra, log_dec, grid_step_ra, grid_step_dec
			plot.plot_grid(grid_step_ra, grid_step_dec, grid_step_ra, grid_step_dec)

		ann = plot.annotations
		ann.NGC = (plot_area_deg < 60)
		ann.constellations = (plot_area_deg > 10)
		ann.constellation_labels = (plot_area_deg > 10)
		ann.constellation_labels_long = False
		ann.bright = (plot_area_deg < 60)
		ann.ngc_fraction = 0.05 / scale

		ann.HD = (plot_area_deg < 9)
		ann.HD_labels = (plot_area_deg < 3)
		ann.hd_catalog = "hd.fits"

		for (r, d, name) in extra:
			ann.add_target(r, d, name)

		for (r1, d1, r2, d2) in extra_lines:
			plot.move_to_radec(r1, d1)
			plot.line_to_radec(r2, d2)
			plot.stroke()

		plot.color = 'green'
		plot.lw = 2
		plot.valign = 'B'
		plot.halign = 'C'
		plot.label_offset_x = 0;
		plot.label_offset_y = -20;

		plot.plot('annotations')
		
		# frame around the image
		if scale > 1:
			plot.color = 'blue'
			plot.polygon([(tx, ty), (tx + tw, ty), (tx + tw, ty + th), (tx, ty + th)])
            		plot.close_path()
            		plot.stroke()

		plot_image = plot.get_image_as_numpy()

		bg = np.zeros_like(plot_image)
		if img is not None:
			if scale == 1:
				thumb = img
			else:
				thumb = cv2.resize(img, (tw, th))
			bg[ty:ty+th, tx:tx+tw, 0] = bg[ty:ty+th, tx:tx+tw, 1] = bg[ty:ty+th, tx:tx+tw, 2] = thumb
		
		res=cv2.add(plot_image, bg)

		return res



if __name__ == "__main__":
	img = cv2.imread("field.tif")
	img = np.amin(img, axis = 2)

	solver = Solver(sources_img = img)
	solver.start()
	solver.join()
	
	plotter=Plotter(solver.wcs)
	plot = plotter.plot(img, scale = 5)
	cv2.imshow("plot", plot)
	cv2.waitKey(0)
