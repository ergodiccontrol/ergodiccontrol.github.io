import numpy as np
from pyodide.ffi import create_proxy


## Variables
# ===============================

controls = None
mouse_pos = None
manipulated_point = None
paths = None
listeners = {}


## Gaussians handling (using the mouse)
# =====================================

def onMouseMove(event):
    global mouse_pos, controls, manipulated_point, param

    rect = canvas.getBoundingClientRect()

    mouse_pos = [
        (event.clientX - rect.left) / (rect.right - rect.left),
        1.0 - (event.clientY - rect.top) / (rect.bottom - rect.top)
    ]

    if (manipulated_point is not None) and manipulated_point[2]:
        id, i, _ = manipulated_point

        if i == 0:
            controls.xPos[:, id] += mouse_pos - controls.Mu[:, id]
            controls.yPos[:, id] += mouse_pos - controls.Mu[:, id]
            controls.Mu[:, id] = mouse_pos
            param.Mu[:, id] = mouse_pos

        else:
            if i == 1:
                controls.xPos[:, id] += mouse_pos - controls.xPos[:, id]

                xDir = controls.xPos[:, id] - controls.Mu[:, id]
                xDir /= np.linalg.norm(xDir)

                yDir = np.array([ -xDir[1], xDir[0] ])

                controls.yPos[:, id] = yDir * np.linalg.norm(controls.yPos[:, id] - controls.Mu[:, id]) + controls.Mu[:, id]

            else:
                controls.yPos[:, id] += mouse_pos - controls.yPos[:, id]

                yDir = controls.yPos[:, id] - controls.Mu[:, id]
                yDir /= np.linalg.norm(yDir)

                xDir = np.array([ -yDir[1], yDir[0] ])

                controls.xPos[:, id] = xDir * np.linalg.norm(controls.xPos[:, id] - controls.Mu[:, id]) + controls.Mu[:, id]

            RG = np.ndarray((2, 2))
            RG[:,0] = controls.xPos[:,id] - controls.Mu[:,id]
            RG[:,1] = controls.yPos[:,id] - controls.Mu[:,id]

            param.Sigma[:, :, id] = (RG @ RG.T) / 2.0

        return

    elif mouse_pos is not None:
        for id in range(controls.nbGaussian):
            for i, point in enumerate([ controls.Mu[:, id], controls.xPos[:, id], controls.yPos[:, id] ]):
                over = np.linalg.norm(point - mouse_pos) <= controls.radius

                if over:
                    manipulated_point = (id, i, False)
                    canvas.style.cursor = 'move'
                    return

    manipulated_point = None
    canvas.style.cursor = 'default'


def onMouseDown(event):
    global manipulated_point
    if manipulated_point is not None:
        manipulated_point = (manipulated_point[0], manipulated_point[1], True)


def onMouseUp(event):
    global manipulated_point
    if manipulated_point is not None:
        manipulated_point = (manipulated_point[0], manipulated_point[1], False)
        reset(reset_state=False)


def register_listeners():
    global listeners

    entries = [
        ('mousemove', onMouseMove),
        ('mousedown', onMouseDown),
        ('mouseup', onMouseUp),
    ]

    for event, listener in entries:
        listener = create_proxy(listener)
        canvas.addEventListener(event, listener)
        listeners[event] = listener


def unregister_listeners():
    global listeners

    for event, listener in listeners.items():
        canvas.removeEventListener(event, listener)

    listeners = {}


def create_gaussian_controls(param):
    controls = lambda: None # Lazy way to define an empty class in python
    controls.nbGaussian = param.nbGaussian
    controls.Mu = np.array(param.Mu)
    controls.xPos = np.zeros((param.nbVar, param.nbGaussian))
    controls.yPos = np.zeros((param.nbVar, param.nbGaussian))
    controls.radius = 0.01

    for id in range(controls.nbGaussian):
        s, U = np.linalg.eig(param.Sigma[:2, :2, id])
        D = np.diag(s) * 2 # Contours are drawn with two standard deviations
        R = np.real(U @ np.sqrt(D+0j))

        controls.xPos[:2,id] = (R @ np.array([1.0, 0.0])).T + param.Mu[:2,id]
        controls.yPos[:2,id] = (R @ np.array([0.0, 1.0])).T + param.Mu[:2,id]

    RG = np.ndarray((2, 2))
    RG[:,0] = (controls.xPos[:,0] - controls.Mu[:,0])
    RG[:,1] = (controls.yPos[:,0] - controls.Mu[:,0])

    return controls


## Helpers
# =====================================

def line_segment_and_circle_intersect(cx, cy, radius, x1, y1, x2, y2):
    # First, we find the equation of the line that passes through the two points (x1, y1) and (x2, y2)
    # The equation of a line in the form y = mx + b is given by:
    #   y - y1 = m(x - x1)
    # We can solve for m as follows:
    m = (y2 - y1) / ((x2 - x1)+1e-30)

    # The equation of the line can then be written as:
    #   y = mx - mx1 + y1
    # We can solve for b as follows:
    b = y1 - m * x1

    # The distance between a point (x0, y0) and a line y = mx + b is given by:
    #   distance = abs(y0 - mx0 - b) / sqrt(m**2 + 1)
    distance = abs(cy - m * cx - b) / np.sqrt(m**2 + 1)

    # If the distance is greater than the radius of the circle, the line segment and the circle do not intersect
    if distance > radius:
        return False
    else:
        # If the distance is less than the radius, we need to check if one of the endpoints of the line segment is inside the circle
        d1 = np.sqrt((cx - x1)**2 + (cy - y1)**2)
        d2 = np.sqrt((cx - x2)**2 + (cy - y2)**2)
        return d1 <= radius or d2 <= radius


## Rendering utilities
# =====================================

def clear_screen():
    ctx.setTransform(canvas.width, 0, 0, -canvas.height, 0, canvas.height)
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, 1, 1)

    ctx_histogram.setTransform(canvas_histogram.width, 0, 0, -canvas_histogram.height, 0, canvas_histogram.height)
    ctx_histogram.fillStyle = 'white'
    ctx_histogram.fillRect(0, 0, 1, 1)


def draw_Gaussian(id, param, color, color2):
    ctx.setTransform(canvas.width, 0, 0, -canvas.height, 0, canvas.height)
    ctx.translate(param.Mu[0,id], param.Mu[1,id])

    s, U = np.linalg.eig(param.Sigma[:2, :2, id])

    # Draw Gaussian
    al = np.linspace(-np.pi, np.pi, 50)
    D = np.diag(s) * 2 # Draw contours with two standard deviations
    R = np.real(U @ np.sqrt(D+0j))

    msh = (R @ np.array([np.cos(al), np.sin(al)])).T #+ param.Mu[:2,id]

    ctx.lineWidth = 0.005
    ctx.fillStyle = color
    ctx.strokeStyle = color2

    ctx.beginPath()
    ctx.moveTo(msh[0,0], msh[0,1])
    for i in range(msh.shape[0]-1):
        ctx.lineTo(msh[i+1,0], msh[i+1,1])
    ctx.closePath()
    ctx.fill()
    ctx.stroke()


def draw_Gaussian_controls(controls, color, color2):
    ctx.setTransform(canvas.width, 0, 0, -canvas.height, 0, canvas.height)

    for id in range(controls.nbGaussian):
        is_manipulating_gaussian = (manipulated_point is not None) and (id == manipulated_point[0]) and manipulated_point[2]

        for i, point in enumerate([ controls.Mu[:, id], controls.xPos[:, id], controls.yPos[:, id] ]):
            is_over = (manipulated_point is not None) and (id == manipulated_point[0]) and (i == manipulated_point[1])

            obj = Path2D.new()
            obj.arc(point[0], point[1], controls.radius * 2.0 if is_over else controls.radius, 0, 2*np.pi)
            ctx.fillStyle = color2 if is_over or is_manipulating_gaussian else color
            ctx.fill(obj)

    if (manipulated_point is not None) and manipulated_point[2]:
        id, _, _ = manipulated_point
        ctx.lineWidth = '0.005'
        ctx.strokeStyle = color2
        ctx.beginPath()
        ctx.moveTo(controls.xPos[0, id], controls.xPos[1, id])
        ctx.lineTo(controls.Mu[0, id], controls.Mu[1, id])
        ctx.lineTo(controls.yPos[0, id], controls.yPos[1, id])
        ctx.stroke()


def draw_scene(param):
    clear_screen()

    # Draw Gaussians
    for k in range(param.nbGaussian):
        draw_Gaussian(k, param, '#FFA50066', '#FFA500')

    # Draw initial points
    ctx.setTransform(canvas.width, 0, 0, -canvas.height, 0, canvas.height)
    ctx.fillStyle = 'black'
    ctx.lineWidth = '0.01'
    if len(param.x0.shape) == 1:
        ctx.beginPath()
        ctx.arc(param.x0[0], param.x0[1], 0.006, 0, 2*np.pi)
        ctx.fill()
    else:
        for y in range(param.x0.shape[0]):
            ctx.beginPath()
            ctx.arc(param.x0[y, 0], param.x0[y, 1], 0.006, 0, 2*np.pi)
            ctx.fill()

    # Draw ergodic control paths
    ctx.lineWidth = 0.005
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)'

    for path in paths:
        ctx.stroke(path)

    # Draw the controls allowing to manipulate the gaussians
    draw_Gaussian_controls(controls, '#AA1166', '#FFFF00')

    # Draw histogram
    ctx_histogram.setTransform(canvas_histogram.width, 0, 0, -canvas_histogram.height, 0, canvas_histogram.height)
    if hist is not None:
        normalized = hist.copy() / np.sum(hist)
        normalized = normalized / np.max(normalized)
        for ky in range(hist.shape[0]):
            for kx in range(hist.shape[1]):
                color = (1.0 - normalized[ky, kx]) * 255
                ctx_histogram.fillStyle = f'rgb({color}, {color}, {color})'
                ctx_histogram.fillRect(xbins[kx], ybins[ky], xbins[kx+1] - xbins[kx], ybins[ky+1] - ybins[ky])
