import numpy as np
from math import *

class Quaternion:
	def __init__(self, a, normalize = False):
		if len(a) == 4:
			if normalize:
				a1 = np.array(a)
				norm = np.linalg.norm(a1)
				self.a = a1 / norm
			else:
				self.a = np.array(a, copy=True)
		elif len(a) == 3:
			self.set_from_euler(a)
	
	def set_from_euler(self, a):
		yaw, pitch, roll = np.deg2rad(a)
		cr = cos(roll/2);
		cp = cos(pitch/2);
		cy = cos(yaw/2);
		sr = sin(roll/2);
		sp = sin(-pitch/2);
		sy = sin(yaw/2);
		cpcy = cp * cy;
		spsy = sp * sy;
		w = cr * cpcy + sr * spsy;
		x = sr * cpcy - cr * spsy;
		y = cr * sp * cy + sr * cp * sy;
		z = cr * cp * sy - sr * sp * cy;
		self.a = [ w, x, y, z ]

	def to_euler(self):
		w, x, y, z = self.a
		roll = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
		pitch = -asin(2 * (w * y - z * x));
		yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
		return np.rad2deg([yaw, pitch, roll]);

#	def to_axis_roll

	def inv(self):
		w, x, y, z = self.a
		return Quaternion([w, -x, -y, -z])

	def __mul__(self, q2):
		w1, x1, y1, z1 = self.a
		w2, x2, y2, z2 = q2.a
		w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2 
		x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2 
		y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2 
		z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
		return Quaternion([w, x, y, z])

	def __div__(self, q2):
		return self * q2.inv()
	
	def transform_vector(self, v):
		v2 = (self * Quaternion([0, v[0], v[1], v[2]]) * self.inv()).a[1:4]
		return v2
	
	def transform_ra_dec(self, rd):
		rd = np.deg2rad(rd)
		v = [ cos(rd[0]) * cos(rd[1]), sin(rd[0]) * cos(rd[1]), sin(rd[1]) ]
		x, y, z = self.transform_vector(v)
		ra = degrees(atan2(y,x))
		dec = degrees(atan2(z, (x ** 2 + y ** 2)**0.5))
		return [ra, dec]
	
	@classmethod
	def from_vector_pair(cls, v1, v2):
		c = np.cross(v1, v2)
		d = np.dot(v1, v2)
		a = [d, c[0], c[1], c[2]]
		
		a[0] += np.linalg.norm(a)
		
		return cls(a, normalize = True)
	
	@classmethod
	def from_ra_dec_pair(cls, rd1, rd2):
		rd1 = np.deg2rad(rd1)
		rd2 = np.deg2rad(rd2)
		
		v1 = [ cos(rd1[0]) * cos(rd1[1]), sin(rd1[0]) * cos(rd1[1]), sin(rd1[1]) ]
		v2 = [ cos(rd2[0]) * cos(rd2[1]), sin(rd2[0]) * cos(rd2[1]), sin(rd2[1]) ]
		
		return cls.from_vector_pair(v1, v2)


