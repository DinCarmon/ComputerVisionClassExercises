import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import networkx as nx

def graph_cut_labeling(ssdd_tensor: np.ndarray):
    """Estimate the depth map using Graph Cuts with Alpha-Expansion.

    Args:
        ssdd_tensor: A tensor of the sum of squared differences for
        every pixel in a window of size win_size X win_size, for the
        2*dsp_range + 1 possible disparity values.

    Returns:
        Mapping depth estimation matrix of shape HxW.
    """
    ssdd_tensor /= 255.0
    num_rows, num_cols, num_disparities = ssdd_tensor.shape

    # Initialize naive labeling (smallest SSD value for each pixel)
    labeling = np.argmin(ssdd_tensor, axis=2)

    # Alpha-expansion algorithm for multi-label optimization
    for alpha in range(num_disparities):
        print(f'handling alpha {alpha}')
        labeling = alpha_expansion(ssdd_tensor, labeling, alpha)

    return labeling

def alpha_expansion(ssdd_tensor, labeling, alpha):
    """Perform a single alpha-expansion step using networkx for max-flow/min-cut.

    Args:
        ssdd_tensor: A tensor of SSD values (HxWxD).
        labeling: Current disparity map.
        alpha: The current label for expansion.

    Returns:
        Updated labeling after alpha-expansion.
    """
    num_rows, num_cols, num_disparities = ssdd_tensor.shape

    # Build the graph
    graph = nx.DiGraph()
    source = "source"
    sink = "sink"

    for i in range(num_rows):
        for j in range(num_cols):
            current_label = labeling[i, j]
            # costs: 1=low, 0=high
            cost_current = ssdd_tensor[i, j, current_label]
            cost_alpha = ssdd_tensor[i, j, alpha]

            # Add terminal edges
            pixel = (i, j)
            graph.add_edge(source, pixel, capacity=math.exp(-(cost_alpha**2)))
            if alpha != current_label:
                graph.add_edge(pixel, sink, capacity=math.exp(-(cost_current**2)))
            else:
                graph.add_edge(pixel, sink, capacity=0)

            # Add pairwise edges
            if i + 1 < num_rows:  # Vertical neighbor
                neighbor = (i + 1, j)
                add_pairwise_edges(graph, pixel, neighbor)
            if j + 1 < num_cols:  # Horizontal neighbor
                neighbor = (i, j + 1)
                add_pairwise_edges(graph, pixel, neighbor)

    # Solve max-flow/min-cut
    flow_value, partition = nx.minimum_cut(graph, source, sink)
    reachable, non_reachable = partition

    # Update labeling based on the min-cut
    for i in range(num_rows):
        for j in range(num_cols):
            pixel = (i, j)
            if pixel in reachable:  # Source side of the cut
                labeling[i, j] = alpha

    return labeling

def add_pairwise_edges(graph, pixel1, pixel2):
    """Add pairwise edges to the graph based on smoothness cost."""
    cost=0.99999
    graph.add_edge(pixel1, pixel2, capacity=(1-cost))
    graph.add_edge(pixel2, pixel1, capacity=(1-cost))

def ncc_distance(left_image: np.ndarray,
                 right_image: np.ndarray,
                 win_size: int,
                 dsp_range: int) -> np.ndarray:
    """Compute the NCC - normalized cross correlation distances tensor.

    Args:
        left_image: Left image of shape: HxWx3, and type np.double64.
        right_image: Right image of shape: HxWx3, and type np.double64.
        win_size: Window size odd integer.
        dsp_range: Half of the disparity range. The actual range is
        -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

    Returns:
        A tensor of the sum of squared differences for every pixel in a
        window of size win_size X win_size, for the 2*dsp_range + 1
        possible disparity values. The tensor shape should be:
        HxWx(2*dsp_range+1).
    """
    num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
    disparity_values = range(-dsp_range, dsp_range + 1)
    ncc_tensor = np.zeros((num_of_rows,
                            num_of_cols,
                            len(disparity_values)))

    if win_size < 0 or win_size % 2 == 0:
        raise TypeError('window size must be odd integer')

    padding_size_left = int((win_size - 1) / 2)
    # Another padding is needed for the horizontal dimension of the second image
    # An example: given we want to calculate the ssd for pixel (0x0), with a windows size of 3,
    # and a depth of -1. We may choose to define such a case with an ssd value of 0 because it is outside
    # the column boundary of the second image. However, a better solution is to compare only the third column of
    # the windows which is in the boundaries of the second image.
    padding_size_right = int((win_size - 1))
    if len(left_image.shape) > 2:  # handle image 3d case
        padded_left_image = np.pad(left_image, (
        (padding_size_left, padding_size_left), (padding_size_left, padding_size_left), (0, 0)))
        padded_right_image = np.pad(right_image, (
        (padding_size_left, padding_size_left), (padding_size_right, padding_size_right), (0, 0)))
    else:  # handle image 2d case
        padded_left_image = np.pad(left_image,
                                   ((padding_size_left, padding_size_left), (padding_size_left, padding_size_left)))
        padded_right_image = np.pad(right_image, (
        (padding_size_left, padding_size_left), (padding_size_right, padding_size_right)))

    for i in range(num_of_rows):
        for j in range(num_of_cols):
            left_window = padded_left_image[i:i + win_size, j:j + win_size]
            left_window_mean = [np.mean(left_window[:,:,i]) for i in range(left_window.shape[2])]
            left_window_std = [np.std(left_window[:,:,i]) for i in range(left_window.shape[2])]

            for d_idx, d in enumerate(disparity_values):
                j_right = j + d + (padding_size_right - padding_size_left)
                if 0 <= j_right <= (num_of_cols + 2 * padding_size_right - win_size):
                    right_window = padded_right_image[i:i + win_size, j_right:j_right + win_size]
                    right_window_mean = [np.mean(right_window[:,:,i]) for i in range(right_window.shape[2])]
                    right_window_std = [np.std(right_window[:,:,i]) for i in range(right_window.shape[2])]
                    a1 = -(left_window - left_window_mean)
                    a2 = (right_window - right_window_mean)
                    a = [( np.sum(a1[:,:,i] * a2[:,:,i]) / (max(left_window_std[i] * right_window_std[i], 0.00000000000000001))) for i in range(right_window.shape[2])]
                    ncc_tensor[i, j, d_idx] = np.mean(a)
                else:
                    ncc_tensor[i, j, d_idx] = 0

    if ncc_tensor.max() == ncc_tensor.min():
        raise ValueError('ssd tensor is all identical. Cannot normalize')
    ncc_tensor -= ncc_tensor.min()
    ncc_tensor /= ncc_tensor.max()
    ncc_tensor *= 255.0
    return ncc_tensor
