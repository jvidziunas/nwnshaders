from argparse import ArgumentParser
from math import exp, cos, acos, fabs, pi

def get_args():
	parser = ArgumentParser(description='Generates a lookup table in .tga format for use with the provided SSS shader.')
	parser.add_argument('--samples', type=int, default=200, help='Number of scattering samples per texel. Larger sample counts improve overall quality, but take longer to compute.')
	parser.add_argument('--width', type=int, default=128, help='Width of the lookup texture, in texels. Larger widths give accurate results across a wider range of light angles.')
	parser.add_argument('--height', type=int, default=128, help='Height of the lookup texture, in texels. Larger heights give accurate results across a wider range of object sizes.')
	parser.add_argument('--min_radius', type=float, default=0.01, help='Minimum object radius to precompute. Must be > 0.')
	parser.add_argument('--max_radius', type=float, default=10.0, help='Maximum object radius to precompute. Must be > min_radius.')
	return parser.parse_args()

def christensen_burley_scale_mfp(albedo):
	return 1.85 - albedo + 7 * fabs(albedo - 0.8)**3

def christensen_burley_scale_dmfp(albedo):
	return 1.9 - albedo + 3.5 * fabs(albedo - 0.8)**2

def as_rgb(red, green, blue):
	return (int(red*255), int(green*255), int(blue*255))

def generate_lut(width, height, min_radius, max_radius, samples, red_dmfp, green_dmfp, blue_dmfp):
	EIGHT_PI = 8.0 * pi

	angle_scale = 2.0 / width
	angle_bias = angle_scale / 2 - 1
	radius_scale = max_radius - min_radius
	radius_bias = radius_scale / 2 + min_radius

	def christensen_burley_bssrdf(distance, dmfp_length):
		return (exp(-distance / dmfp_length) + exp(-distance / (3 * dmfp_length))) / (EIGHT_PI * distance * dmfp_length)

	def integrate_scattering(ring_radius, angle):
		cos_theta = angle * angle_scale + angle_bias
		theta = acos(cos_theta)

		curvature = ring_radius * radius_scale + radius_bias
		bound = pi / curvature

		integration_scale = 2 * bound / samples
		integration_bias = -bound + integration_scale / 2

		# Monte Carlo integration for scattering estimation.
		# To consider: Is there a way to improve precision here? Repeated adds are where ULPs go to die
		# To consider: The Christensen-Burley BSSRDF can be importance sampled, potentially requiring fewer
		#	samples per texel for equivalent to superior quality
		red, green, blue = 0.0, 0.0, 0.0
		for sample in xrange(samples):
			delta = sample * integration_scale + integration_bias
			cos_delta = max(cos(theta - delta * curvature), 0.0)

			red += cos_delta * christensen_burley_bssrdf(delta, red_dmfp)
			green += cos_delta * christensen_burley_bssrdf(delta, green_dmfp)
			blue += cos_delta * christensen_burley_bssrdf(delta, blue_dmfp)
		return as_rgb(red * integration_scale, green* integration_scale, blue* integration_scale)

	def integrate_ring(angular_resolution, radius):
		return [integrate_scattering(radius, angle) for angle in xrange(angular_resolution)]

	return [integrate_ring(angular_resolution=width, radius=radius) for radius in xrange(height)]

args = get_args()
generate_lut(width=args.width, height=args.height, min_radius=args.min_radius, max_radius=args.max_radius, samples=args.samples)