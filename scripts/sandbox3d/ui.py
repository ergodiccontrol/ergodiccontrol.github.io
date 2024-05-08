from pyodide.ffi import create_proxy


if 'gaussians' not in globals():
    gaussians = []


def on_gaussian_modified(gaussian, inProgress):
    if inProgress:
        return

    for i, gaussian in enumerate(gaussians):
        gaussian_params.Mu[:,i] = gaussian.position
        gaussian_params.Sigma[:,:,i] = gaussian.sigma

    reset(reset_state=False)


def reset_rendering(param):
    global gaussians

    viewer3D.removePath('trajectory')

    for gaussian in gaussians:
        viewer3D.removeGaussian(gaussian.name)

    gaussians = []
    for n in range(param.nbGaussian):
        gaussian = viewer3D.addGaussian(
            f'gaussian{n}',
            gaussian_params.Mu[:,n],
            gaussian_params.Sigma[:,:,n],
            '#0000aa',
            on_gaussian_modified
        )
        gaussians.append(gaussian)

    viewer3D.setRenderingCallback(start_rendering if start_rendering is not None else update_rendering)
    viewer3D.physicsSimulatorPaused = False


def start_rendering(delta, time):
    global start_rendering

    run_buttons = document.getElementsByClassName("py-repl-run-button")
    for button in run_buttons:
        if button.getAttribute('hacked') is None:
            button.addEventListener('click', create_proxy(lambda evt: reset()))
            button.setAttribute('hacked', True)

    viewer3D.setRenderingCallback(update_rendering)

    update_rendering(delta, time)

    start_rendering = None


def update_rendering(delta, time):
    update()

    if trajectory.shape[0] > 1:
        viewer3D.removePath('trajectory')
        viewer3D.addPath(
            'trajectory',
            trajectory,
            0.005,
            '#ffff00',
            False,      # No shading
            True        # Transparent
        )
