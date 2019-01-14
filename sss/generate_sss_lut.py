from argparse import ArgumentParser
from struct import pack
from math import exp, acos, cos, sin, fabs, pi
from sys import float_info

# Common constants.
SRGB_EXPONENT = 1 / 2.4
EIGHT_PI = 8 * pi
ONE_OVER_TWO_PI = 1.0 / (2 * pi)

def get_args():
	parser = ArgumentParser(description='Generates a lookup table in .tga format for use with the provided SSS shader.')
	parser.add_argument('name', help='Output file name for the resulting table')
	parser.add_argument('--samples', type=int, default=200, help='Number of scattering samples per texel. Larger sample counts improve overall quality, but take longer to compute. Must be > 0.')
	parser.add_argument('--lut_width', type=int, default=128, help='Width of the lookup texture, in texels. Larger widths give accurate results across a wider range of light angles. Must be > 0.')
	parser.add_argument('--lut_height', type=int, default=128, help='Height of the lookup texture, in texels. Larger heights give accurate results across a wider range of object sizes. Must be > 0.')
	parser.add_argument('--min_radius', type=float, default=0.1, help='Minimum object radius to precompute, in meters. Must be > 0.0.')
	parser.add_argument('--max_radius', type=float, default=1.0, help='Maximum object radius to precompute, in meters. Must be > min_radius.')
	parser.add_argument('--mfp_r', type=float, default=1.0, help='Mean free path length for red channel, in millimeters. Must be > 0.0.')
	parser.add_argument('--mfp_g', type=float, default=1.0, help='Mean free path length for green channel, in millimeters. Must be > 0.0.')
	parser.add_argument('--mfp_b', type=float, default=1.0, help='Mean free path length for blue channel, in millimeters. Must be > 0.0.')
	parser.add_argument('--albedo_r', type=float, default=1.0, help='Base reflectance for red channel. Must be > 0.0 and < 1.0.')
	parser.add_argument('--albedo_g', type=float, default=1.0, help='Base reflectance for green channel. Must be > 0.0 and < 1.0.')
	parser.add_argument('--albedo_b', type=float, default=1.0, help='Base reflectance for blue channel. Must be > 0.0 and < 1.0.')
	return parser.parse_args()

def pack_tga_header(width, height):
	header = bytearray()
	# Field 6 length/field 1. Always a 0 byte.
	header += b'\0'
	# Colormap type/field 2. We never include a colormap, so this is always a 0 byte.
	header += b'\0'
	# Image type/field 3. We always use a 24-bit uncompressed image, so this is always a 2 byte.
	header += b'\2'
	# Colormap specifier/field 4. We never include a colormap, so this is always 5 0 bytes.
	header += pack('<HHB', 0, 0, 0)
	# Image specification/field 5. NWN wants TGA images in bottom-left read order.
	header += pack('<HHHHBB', 0, 0, int(width), int(height), 24, 0b0)
	return header

def generate_lut(lut_width, lut_height, min_radius, max_radius, samples, mfp_r, mfp_g, mfp_b, albedo_r, albedo_g, albedo_b, **kwargs):
	MFP_R = mfp_r * 1000
	MFP_G = mfp_g * 1000
	MFP_B = mfp_b * 1000
	def scale_mfp(albedo):
		return 1.85 - albedo + 7 * fabs(albedo - 0.8)**3
	def scale_dmfp(albedo):
		return 1.9 - albedo + 3.5 * fabs(albedo - 0.8)**2
	def bssrdf(r, s, mfp_length):
		# Christensen-Burley Approximate BSSRDF.
		exp3 = exp(-s*r / (3.0 * mfp_length))
		expd = exp3 * exp3 * exp3
		return min(s * (exp3 + expd) / max(EIGHT_PI * r * mfp_length, float_info.epsilon), 1.0)
	def to_srgb_unorm(val):
		srgb = 12.92 * val if val <= 0.0031308 else 1.055 * (val**SRGB_EXPONENT) - 0.055
		return max(min(int(srgb * 255), 255), 0)
	def integrate_scattering(radius, cos_theta):
		sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
		s_r, s_g, s_b = scale_dmfp(albedo_r), scale_dmfp(albedo_g), scale_dmfp(albedo_b)

		sample_scale = (2 * pi) / samples
		theta = acos(cos_theta)
		for sample in xrange(samples):
			a = pi - sample_scale * sample
			distance = fabs(2 * radius * sin(a * 0.5))
			bssrdf_r = bssrdf(distance, s_r, mfp_r)
			bssrdf_g = bssrdf(distance, s_g, mfp_g)
			bssrdf_b = bssrdf(distance, s_b, mfp_b)

			cos_delta = max(cos(theta + a), 0.0)
			sum_r += cos_delta * bssrdf_r
			sum_g += cos_delta * bssrdf_g
			sum_b += cos_delta * bssrdf_b
		return (to_srgb_unorm(sum_r * sample_scale), to_srgb_unorm(sum_g * sample_scale), to_srgb_unorm(sum_b * sample_scale))

	def integrate_ring(lut_width, radius):
		cos_step = 2.0 / lut_width
		return [integrate_scattering(radius, cos_step * cosine - 1) for cosine in xrange(lut_width)]

	radius_step = (max_radius - min_radius) / lut_height
	# OpenGL texture origin is bottom-left, so we need to flip the Y coordinate ordering (large -> small radii)
	return [integrate_ring(lut_width=lut_width, radius=max_radius - radius_step * radius) for radius in xrange(lut_height)]

if __name__ == "__main__":
	args = get_args()
	table = generate_lut(**vars(args))
	with open("{0:s}_lut.tga".format(args.name), "wb") as tga:
		# Write TGA header.
		tga.write(pack_tga_header(width=len(table[0]), height=len(table)))
		# Write texels.
		for scanline in table:
			for pixel in scanline:
				tga.write(pack('<BBB', *pixel))
		# Write TGA footer.
		tga.write("TRUEVISION-XFILE.\0".encode('ascii'))
	with open("{0:s}.mtr".format(args.name), "w") as material:
		radiusScale = 1 / (args.max_radius - args.min_radius)
		# Write NWN material with precomputed uniforms.
		material.write("// Shaders\ncustomshaderVS vslit_sm_nm\ncustomshaderFS fslit_sss_nm")
		material.write("\n\n// Textures".encode('ascii'))
		material.write("\ntexture0 // <base map here>\ntexture1 // <normal map here>\ntexture2 // <specular map here>".encode('ascii'))
		material.write("\ntexture3 {0:s}_lut".format(args.name).encode('ascii'))
		material.write("\n// Parameters\n// min_radius: {0:f}, max_radius: {1:f}".format(args.min_radius, args.max_radius).encode('ascii'))
		material.write("\nparameter float radiusBias {0:f}".format(-args.min_radius * radiusScale).encode('ascii'))
		material.write("\nparameter float radiusScale {0:f}".format(radiusScale).encode('ascii'))