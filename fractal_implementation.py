"""
Fractal implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("PyTorch Device:", device)

"""
The code below was generated using ChatGPT:

*AI Prompt Used*
Generate a python script to plot the Koch Snowflake fractal using Pytorch, Numpy and Matplotlib using the GPU
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import sys

# ---------- User parameters ----------
iterations = 6           # recommended: 0..6 (each iteration multiplies segment count by 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
figsize = (8, 8)
dpi = 150
line_width = 0.6
save_file = "koch_snowflake.png"
# --------------------------------------

if device.type == "cuda":
    print("Using GPU:", torch.cuda.get_device_name(device))
else:
    print("Using device:", device)

def initial_triangle(scale=1.0):
    """
    Produce an equilateral triangle centered at origin in numpy (3 x 2).
    We'll scale it so it looks nice in the plot.
    """
    # Equilateral triangle vertices centered around origin
    angles = np.array([0.0, 2*np.pi/3, 4*np.pi/3])
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * scale
    return pts.astype(np.float32)

def rotate_tensor(vecs, angle):
    """
    Rotate 2D vectors by given angle.
    vecs: (N,2) tensor
    angle: scalar (radians)
    """
    c = math.cos(angle)
    s = math.sin(angle)
    R = torch.tensor([[c, -s], [s, c]], dtype=vecs.dtype, device=vecs.device)
    # (N,2) x (2,2) -> (N,2)
    return vecs.matmul(R.T)

def koch_iteration(points):
    """
    Perform one Koch iteration on a closed polygon given by `points` (torch tensor Nx2).
    Returns new_points (M x 2) where M ~ 4 * N (minus duplicates handling).
    """
    # Ensure points is (N,2)
    if points.dim() != 2 or points.size(1) != 2:
        raise ValueError("points must be (N,2)")

    # close polygon by appending first point
    A = points
    B = torch.cat([points[1:], points[:1]], dim=0)  # shifted

    # vectors
    vec = B - A  # (N,2)

    third = vec / 3.0
    one_third = A + third              # A + (B-A)/3
    two_third = A + 2.0 * third       # A + 2*(B-A)/3

    # apex / peak: rotate the segment between one_third and two_third by +60 degrees about one_third
    # vector between one_third and two_third is third
    peak = one_third + rotate_tensor(third, math.pi / 3.0)

    # For each original segment, produce points [A, one_third, peak, two_third]
    # We'll stack them and flatten in order, then append final closing point at the end
    segments = torch.stack([A, one_third, peak, two_third], dim=1)  # (N,4,2)
    N = segments.size(0)
    segments = segments.view(N * 4, 2)  # linearized but doesn't include final closing B

    # We also need to append the last original point B_last to close properly
    B_last = B[-1:].clone()  # (1,2)
    new_points = torch.cat([segments, B_last], dim=0)

    return new_points

def generate_koch(initial_pts_np, iterations, device):
    """
    initial_pts_np: numpy array (K,2)
    iterations: number of iterations
    device: torch device
    returns: final_points as numpy array (M,2)
    """
    pts = torch.from_numpy(initial_pts_np).to(device)

    # If initial polygon should be closed we will treat it as closed in function
    for i in range(iterations):
        pts = koch_iteration(pts)
        # after each iteration, many segments; we keep them all as float32
        if pts.numel() > 50_000_000:
            # safety guard to avoid exhausting memory
            raise MemoryError("Too many points; reduce iterations.")

    # Move to CPU and numpy for plotting
    pts_cpu = pts.detach().cpu().numpy()
    return pts_cpu

def plot_snowflake(points_np, figsize=(8,8), dpi=150, save_file=None, linewidth=0.8):
    plt.figure(figsize=figsize, dpi=dpi)
    # points_np is a list of vertices in order; connect them
    x = points_np[:,0]
    y = points_np[:,1]
    plt.plot(x, y, linewidth=linewidth)
    plt.axis('equal')
    plt.axis('off')
    if save_file:
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved to {save_file}")
    plt.show()

def main():
    print("Generating Koch Snowflake with iterations =", iterations)
    # Choose scale so the snowflake fits nicely
    scale = 1.0
    init = initial_triangle(scale=scale)
    # The polygon must be closed for interpretation; initial_triangle returns 3 points
    final_pts = generate_koch(init, iterations, device)
    print("Final number of points:", final_pts.shape[0])
    plot_snowflake(final_pts, figsize=figsize, dpi=dpi, save_file=save_file, linewidth=line_width)

if __name__ == "__main__":
    main()
