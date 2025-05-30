"""
visualization.py

Batch visualization utility for torchcontrol.

This module provides high-level functions for visualizing batches of time-series data (2D or 3D) as static images or animated GIFs.
It supports efficient multiprocessing rendering, flexible grid layout, and both 2D/3D/multi-curve plotting,
making it suitable for large-scale simulation or control experiments.

Main API:
    - render_batch_gif: Render a batch grid of 2D/3D time-series plots as an animated GIF.
    - render_batch_img: Render a batch grid of 2D/3D time-series plots as a static image.

Features:
    - Supports multiple curves per subplot: x_hist, y_hist, z_hist can all be [num_envs, num_steps, n_curves] or [num_envs, num_steps].
    - Each curve can have its own label and line style.
    - 2D: Each subplot can show multiple time series, XY trajectories, or any batch multi-curve data.
    - 3D: Each subplot can show multiple XYZ trajectories or batch 3D time series.
    - Efficient multiprocessing rendering for large batches (GIF).
    - Flexible grid layout and axis/legend customization.
    - Helper functions ensure input arrays and metadata are always valid and consistent.
"""

import io
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def render_batch_gif(
    gif_path: str,
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    z_hist: np.ndarray | None = None,
    width: int = 1,
    height: int = 1,
    labels: list[str] | None = None,
    line_styles: list[str] | None = None,
    titles: list[str] | None = None,
    frame_stride: int = 1,
    duration: float = 0.04,
    xlim: list[float] | tuple[float, float] | None = None,
    ylim: list[float] | tuple[float, float] | None = None,
    zlim: list[float] | tuple[float, float] | None = None,
    xlabel: str = 'X',
    ylabel: str = 'Y',
    zlabel: str | None = None,
):
    """
    Render a batch grid of 2D or 3D time-series plots as an animated GIF using multiprocessing.
    Supports multiple curves per subplot. Each of x_hist, y_hist, z_hist (if present) should be
    shape [num_envs, num_steps, n_curves] or [num_envs, num_steps] (will be broadcasted to n_curves=1).

    Args:
        gif_path (str): Output path for the GIF file.
        x_hist (np.ndarray): [num_envs, num_steps, n_curves] or [num_envs, num_steps] X data for each curve in each environment.
        y_hist (np.ndarray): [num_envs, num_steps, n_curves] or [num_envs, num_steps] Y data for each curve in each environment.
        z_hist (np.ndarray | None): [num_envs, num_steps, n_curves] or [num_envs, num_steps] for 3D. If None, do 2D plot.
        width (int): Number of columns in the batch grid.
        height (int): Number of rows in the batch grid.
        labels (list[str] | None): Labels for each curve. Defaults to ['curve0', ...].
        line_styles (list[str] | None): Line styles for each curve. Defaults to ['-']*n_curves.
        titles (list [str] | None): Titles for each subplot. Defaults to ['Env 0', ...].
        frame_stride (int): Stride for frame saving (controls GIF speed). Defaults to 1 (save every frame).
        duration (float): Duration per frame in seconds. Defaults to 0.04 (25 FPS).
        xlim (list[float] | tuple[float, float] | None): x-axis limits [xmin, xmax].
        ylim (list[float] | tuple[float, float] | None): y-axis limits [ymin, ymax].
        zlim (list[float] | tuple[float, float] | None): z-axis limits [zmin, zmax] (for 3D).
        xlabel (str): x-axis label. Defaults to 'X'.
        ylabel (str): y-axis label. Defaults to 'Y'.
        zlabel (str | None): z-axis label (for 3D).

    Example:
        >>> render_batch_gif(
        ...     gif_path='results/demo.gif',
        ...     x_hist=x_hist, # shape [16, 100, 3]
        ...     y_hist=y_hist, # shape [16, 100, 3]
        ...     z_hist=None,
        ...     width=4,
        ...     height=4,
        ...     labels=['x1', 'x2', 'ref'],
        ...     line_styles=['-', '--', 'r--'],
        ...     titles=None,  # Defaults to ['Env 0', 'Env 1', ...]
        ...     frame_stride=10,
        ...     duration=0.04, # 25 FPS
        ...     xlim=None,
        ...     ylim=None,
        ...     zlim=None,
        ...     xlabel='Time (s)',
        ...     ylabel='Value',
        ...     zlabel=None,  # Only for 3D plots
        ... )
    """
    # Ensure x_hist, y_hist, z_hist are 3D arrays with the same shape
    x_hist, y_hist, z_hist = ensure_xyz_hist(x_hist, y_hist, z_hist)
    
    # Get num_envs, num_steps, n_curves
    num_envs, num_steps, n_curves = x_hist.shape
    
    # Ensure width and height are enough to fit num_envs in a grid
    width, height = ensure_width_height(width, height, num_envs)
   
    # Set default labels and check lengths
    labels = ensure_labels(labels, n_curves)
    
    # Set default line styles and check lengths
    line_styles = ensure_line_styles(line_styles, n_curves)
    
    # Set default titles and check lengths
    titles = ensure_titles(titles, num_envs)
    
    # Create argument list for multiprocessing
    frame_indices = list(range(0, num_steps, frame_stride))
    args_list = [
        (None,  # img_path is None for GIF frames
         x_hist[:, :frame_idx+1, :],
         y_hist[:, :frame_idx+1, :],
         z_hist[:, :frame_idx+1, :] if z_hist is not None else None,
         width, height, labels, line_styles, titles, xlim, ylim, zlim, ylabel, xlabel, zlabel)
        for frame_idx in frame_indices
    ]
    # Use ProcessPoolExecutor + as_completed for live tqdm
    frames = [None] * len(args_list)
    with ProcessPoolExecutor() as executor:
        future_to_idx = {executor.submit(render_batch_img, *args): idx for idx, args in enumerate(args_list)}
        with tqdm(total=len(args_list), desc="Rendering GIF frames") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                frames[idx] = future.result()
                pbar.update(1)
    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"GIF saved to {gif_path}")

def render_batch_img(
    img_path: str | None,
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    z_hist: np.ndarray | None,
    width: int = 1,
    height: int = 1,
    labels: list[str] | None = None,
    line_styles: list[str] | None = None,
    titles: list[str] | None = None,
    xlim: list[float] | tuple[float, float] | None = None,
    ylim: list[float] | tuple[float, float] | None = None,
    zlim: list[float] | tuple[float, float] | None = None,
    ylabel: str = 'Y',
    xlabel: str = 'X',
    zlabel: str | None = None,
) -> np.ndarray | None:
    """
    Render a static batch image for all data points provided.

    Args:
        img_path (str | None): If given, save the image to this path. Otherwise, return the image array.
        x_hist, y_hist, z_hist: [num_envs, num_steps, n_curves] arrays (all data to plot).
        width, height: Grid size.
        labels: List of curve labels.
        line_styles: List of line styles.
        titles: List of subplot titles.
        xlim, ylim, zlim: Axis limits.
        ylabel, xlabel, zlabel: Axis labels.
    Returns:
        np.ndarray | None: The rendered frame as an image array if img_path is None, else None.

    Example:
        >>> render_batch_img(
        ...     img_path='results/final.png',
        ...     x_hist=x_hist,  # shape [16, 100, 3]
        ...     y_hist=y_hist,  # shape [16, 100, 3]
        ...     z_hist=None,
        ...     width=4,
        ...     height=4,
        ...     labels=['x1', 'x2', 'ref'],
        ...     line_styles=['-', '--', 'r--'],
        ...     titles=None, # Defaults to ['Env 0', 'Env 1', ...]
        ...     xlim=None,
        ...     ylim=None,
        ...     zlim=None,
        ...     xlabel='Time (s)',
        ...     ylabel='Value',
        ...     zlabel=None,
        ... )
    """
    # Ensure x_hist, y_hist, z_hist are 3D arrays with the same shape
    x_hist, y_hist, z_hist = ensure_xyz_hist(x_hist, y_hist, z_hist)
    
    # Get num_envs, num_steps, n_curves
    num_envs, num_steps, n_curves = x_hist.shape
    
    # Ensure width and height are enough to fit num_envs in a grid
    width, height = ensure_width_height(width, height, num_envs)
   
    # Set default labels and check lengths
    labels = ensure_labels(labels, n_curves)
    
    # Set default line styles and check lengths
    line_styles = ensure_line_styles(line_styles, n_curves)
    
    # Set default titles and check lengths
    titles = ensure_titles(titles, num_envs)
    
    is_3d = z_hist is not None
    figsize = (4 * width, 3.5 * height) if is_3d else (4 * width, 3 * height) # Adjust height for 3D plots
    fig = plt.figure(figsize=figsize)
    for idx in range(num_envs):
        if is_3d:
            ax = fig.add_subplot(height, width, idx + 1, projection='3d')
            for k in range(n_curves):
                style = line_styles[k]
                label = labels[k]
                ax.plot(x_hist[idx, :, k], y_hist[idx, :, k], z_hist[idx, :, k], style, label=label)
            if zlabel:
                ax.set_zlabel(zlabel)
            if zlim is not None:
                ax.set_zlim(zlim)
        else:
            ax = fig.add_subplot(height, width, idx + 1)
            for k in range(n_curves):
                style = line_styles[k]
                label = labels[k]
                ax.plot(x_hist[idx, :, k], y_hist[idx, :, k], style, label=label)
        ax.set_title(titles[idx])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid()
        ax.legend()
    plt.tight_layout()
    
    # Save to file or return as image array
    if img_path is not None:
        fig.savefig(img_path)
        plt.close(fig)
        print(f"Image saved to {img_path}")
        return None
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = imageio.imread(buf)
        buf.close()
        return img

"""
Helper functions to ensure input arrays are in the correct shape and format.
"""

def ensure_3d(array: np.ndarray | None) -> np.ndarray | None:
    """
    Ensure the input array is 3D. If it is 1D or 2D, add new axiss to make it 3D.
    If it is already 3D, return as is. If None, return None.
    Args:
        array (np.ndarray | None): Input array to ensure 3D shape.
    Returns:
        array (np.ndarray | None): The input array reshaped to 3D. Shape: [num_envs, num_steps, n_curves]
    """
    if array is None: # If the array is None, return None
        return None
    assert 0 < array.ndim <= 3, "Input array must be 1D, 2D, or 3D."
    
    # If 1D or 2D, add new axes to make it 3D
    if array.ndim == 1:
        return array[np.newaxis, :, np.newaxis] # Add new axes for num_envs=1, n_curves=1
    elif array.ndim == 2:
        return array[:, :, np.newaxis]  # Add a new axis for n_curves=1
    else:
        return array # Already 3D, return as is

def ensure_xyz_hist(x_hist: np.ndarray | None, y_hist: np.ndarray | None, z_hist: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure x_hist, y_hist, z_hist are all 3D arrays with the same shape.
    Args:
        x_hist (np.ndarray | None): X data for each curve in each environment.
        y_hist (np.ndarray | None): Y data for each curve in each environment.
        z_hist (np.ndarray | None): Z data for each curve in each environment (optional, for 3D plots).
    Returns:
        (x_hist, y_hist, z_hist): Tuple of (x_hist, y_hist, z_hist) all ensured to be 3D.
    """
    # Ensure x_hist, y_hist, z_hist are 3D arrays
    x_hist = ensure_3d(x_hist)
    y_hist = ensure_3d(y_hist)
    z_hist = ensure_3d(z_hist)

    # x_hist and y_hist must not be None
    if x_hist is None or y_hist is None:
        raise ValueError("x_hist and y_hist must not be None.")
    
    # Validate shapes
    if z_hist is not None:
        assert x_hist.shape == y_hist.shape == z_hist.shape, "x_hist, y_hist, and z_hist must have the same shape, got shapes: {}, {}, {}".format(
            x_hist.shape, y_hist.shape, z_hist.shape)
    else:
        assert x_hist.shape == y_hist.shape, "x_hist and y_hist must have the same shape, got shapes: {}, {}".format(
            x_hist.shape, y_hist.shape)
    
    return x_hist, y_hist, z_hist

def ensure_width_height(width: int | None, height: int | None, num_envs: int) -> tuple[int, int]:
    """
    Ensure the width and height are enough to fit num_envs in a grid.
    If width or height is None, calculate the other dimension based on num_envs.
    Args:
        width (int | None): Desired width of the grid.
        height (int | None): Desired height of the grid.
        num_envs (int): Number of environments to fit in the grid.
    Returns:
        (width, height): (width, height) that can fit num_envs in a grid.
    """
    if width is None and height is None:
        raise ValueError("Either width or height must be specified.")
    
    # Calculate width and height based on num_envs, if one of them is None
    if width is None:
        # Calculate width based on height
        width = np.ceil(num_envs / height).astype(int)
    elif height is None:
        # Calculate height based on width
        height = np.ceil(num_envs / width).astype(int)
    
    # Ensure width and height are positive integers
    assert width > 0 and height > 0, "Width and height must be positive integers."
    
    # Ensure the grid can fit all environments
    assert width * height >= num_envs, f"Grid size {width}x{height} must be at least {num_envs} to fit all environments."
    
    return width, height

def ensure_labels(labels: list[str] | None, n_curves: int) -> list[str]:
    """
    Ensure the labels list has the correct length for the number of curves.
    If labels is None, generate default labels.
    Args:
        labels (list[str] | None): Input labels.
        n_curves (int): Number of curves.
    Returns:
        labels (list[str]): Labels with length equal to n_curves.
    """
    if labels is None:
        return [f'curve{i}' for i in range(n_curves)]
    assert len(labels) == n_curves, f"labels must have length {n_curves}, got {len(labels)}."
    return labels

def ensure_line_styles(line_styles: list[str] | None, n_curves: int) -> list[str]:
    """
    Ensure the line styles list has the correct length for the number of curves.
    If line_styles is None, generate default styles.
    Args:
        line_styles (list[str] | None): Input line styles.
        n_curves (int): Number of curves.
    Returns:
        line_styles (list[str]): Line styles with length equal to n_curves.
    """
    if line_styles is None:
        return ['-'] * n_curves
    assert len(line_styles) == n_curves, f"line_styles must have length {n_curves}, got {len(line_styles)}."
    return line_styles

def ensure_titles(titles: list[str] | None, num_envs: int) -> list[str]:
    """
    Ensure the titles list has the correct length for the number of environments.
    If titles is None, generate default titles.
    Args:
        titles (list[str] | None): Input titles.
        num_envs (int): Number of environments.
    Returns:
        titles (list[str]): Titles with length equal to num_envs.
    """
    if titles is None:
        return [f'Env {i}' for i in range(num_envs)]
    assert len(titles) == num_envs, f"titles must have length {num_envs}, got {len(titles)}."
    return titles
