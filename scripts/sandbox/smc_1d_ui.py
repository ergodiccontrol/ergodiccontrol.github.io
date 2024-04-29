import numpy as np


## Rendering utilities
# =====================================

GAUSSIANS_HEIGHT = 0.29
HISTOGRAM_HEIGHT = 0.05
MARGIN = 0.05

GAUSSIANS_POSITION = 1.0 - GAUSSIANS_HEIGHT - 0.1
PATH_1D_POSITION = GAUSSIANS_POSITION - MARGIN

PATH_2D_HEIGHT = PATH_1D_POSITION - HISTOGRAM_HEIGHT - MARGIN


def clear_screen():
	ctx.setTransform(canvas.width, 0, 0, -canvas.height, 0, canvas.height)
	ctx.fillStyle = 'white'
	ctx.fillRect(0, 0, 1, 1)


def draw_gaussian(x, y, param, color, color2):
	ctx.setTransform(canvas.width, 0, 0, -canvas.height, 0, canvas.height)
	ctx.translate(0, GAUSSIANS_POSITION)

	ctx.lineWidth = 0.005
	ctx.fillStyle = color
	ctx.strokeStyle = color2

	ctx.beginPath()
	ctx.moveTo(x[0], 0)
	for i in range(y.shape[0]):
		ctx.lineTo(x[i], y[i] * GAUSSIANS_HEIGHT)
	ctx.lineTo(x[-1], 0)
	ctx.closePath()
	ctx.fill()
	ctx.stroke()


def draw_scene(param):
	clear_screen()

	# Draw Gaussians
	x = np.linspace(param.xlim[0], param.xlim[1], int((param.xlim[1] - param.xlim[0]) / 0.005))
	y = np.ndarray((param.nbGaussian, x.shape[0]))
	for k in range(param.nbGaussian):
		mu = param.Mu[k]
		sigma = param.Sigma[k] * 4
		y[k, :] = 1. / (np.sqrt(2. * np.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.) / 2)
	y = y / np.max(y)
	
	for k in range(param.nbGaussian):
		draw_gaussian(x, y[k, :], param, '#FFA50066', '#FFA500')

	# Draw ergodic control 1D path
	ctx.setTransform(canvas.width, 0, 0, -canvas.height, 0, canvas.height)
	ctx.translate(0, PATH_1D_POSITION)

	ctx.lineWidth = 0.005
	ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)'
	ctx.stroke(path_1d)

	# Draw current position
	ctx.fillStyle = '#000000FF'
	ctx.beginPath()
	ctx.arc(r_x[-1], 0.0, 0.01, 0, 2*np.pi)
	ctx.fill()

	# Draw histogram
	ctx.setTransform(canvas.width, 0, 0, -canvas.height, 0, canvas.height)
	if hist is not None:
		for k in range(len(hist)):
			color = (1.0 - (hist[k] / np.max(hist))) * 255
			ctx.fillStyle = f'rgb({color}, {color}, {color})'
			ctx.fillRect(bins[k], 0.0, bins[k+1] - bins[k], HISTOGRAM_HEIGHT)

	# Draw ergodic control 2D path
	ctx.translate(0, PATH_1D_POSITION - t / param.nbData * PATH_2D_HEIGHT)
	ctx.lineWidth = 0.002
	ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)'
	ctx.stroke(path_2d)


def register_listeners():
	pass


def unregister_listeners():
	pass
