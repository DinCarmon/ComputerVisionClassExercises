"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

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
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))

        if win_size < 0 or win_size % 2 == 0:
            raise TypeError('window size must be odd integer')

        padding_size = int((win_size-1)/2)
        if len(left_image.shape) > 2: # handle image 3d case
            padded_left_image = np.pad(left_image, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)))
            padded_right_image = np.pad(right_image, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)))
        else: # handle image 2d case
            padded_left_image = np.pad(left_image, ((padding_size, padding_size), (padding_size, padding_size)))
            padded_right_image = np.pad(right_image, ((padding_size, padding_size), (padding_size, padding_size)))


        for i in range(num_of_rows):
            for j in range(num_of_cols):
                left_window = padded_left_image[i:i+win_size, j:j+win_size]
                for d_idx, d in enumerate(disparity_values):
                    j_right = j+d
                    if j_right >= 0 and j_right < num_of_cols:
                        right_window = padded_right_image[i:i + win_size, j_right:j_right + win_size]
                        ssdd_tensor[i, j, d_idx] = np.sum((left_window-right_window) ** 2)
                    else:
                        ssdd_tensor[i, j, d_idx] = np.sum(left_window ** 2)

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def min_excluding_rows(l_slice, p, exclude_list):
        rows_to_check = np.setdiff1d(np.arange(l_slice.shape[0]), exclude_list)
        return np.min(l_slice[rows_to_check, p])

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))

        # Initialize first column
        l_slice[:, 0] = c_slice[:, 0]
        l_slice[:, 0] -= np.min(l_slice[:, 0])

        # Use padding to handle edge cases
        l_slice_padded = np.pad(l_slice, ((1, 1), (0, 0)), constant_values=np.inf)

        for p in range(1, num_of_cols):
            for d in range(1, num_labels + 1):
                same_disp = l_slice_padded[d, p - 1]
                lower_p1_disp = l_slice_padded[d - 1, p - 1] + p1
                higher_p1_disp = l_slice_padded[d + 1, p - 1] + p1
                lower_p2_disp = np.min(l_slice_padded[:d - 1, p - 1]) + p2 if d - 1 > 0 else np.inf
                above_p2_disp = np.min(l_slice_padded[d + 2:, p - 1]) + p2 if d + 2 < num_labels + 2 else np.inf

                # Update l_slice; c_slice is in d-1 since l_slice is padded
                l_slice_padded[d, p] = c_slice[d - 1, p] + min(same_disp, lower_p1_disp, higher_p1_disp, lower_p2_disp, above_p2_disp)

                # Normalize
                l_slice_padded[d, p] -= min(l_slice_padded[:,p-1])

        # Remove padding
        l_slice = l_slice_padded[1:-1, :]

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)

        num_of_rows = ssdd_tensor.shape[0]
        for i in range(num_of_rows):
            c_slice = ssdd_tensor[i, :, :].transpose()
            l[i, :, :] = self.dp_grade_slice(c_slice, p1, p2).transpose()

        return self.naive_labeling(l)

    def dp_get_direction_slices(self,
                    ssdd_tensor: np.ndarray,
                    direction: int):

        c_slices = []
        if direction == 1: # loop over rows
            num_of_rows, num_of_cols = ssdd_tensor.shape[0], ssdd_tensor.shape[1]
            for i in range(num_of_rows):
                c_slice = ssdd_tensor[i, :, :].transpose()
                c_slices.append(c_slice)
        elif direction == 2: # loop over diags
            num_of_rows, num_of_cols = ssdd_tensor.shape[0], ssdd_tensor.shape[1]
            for offset in range(-(num_of_rows - 1), num_of_cols):
                c_slice = np.diagonal(ssdd_tensor, offset=offset)
                c_slices.append(c_slice)
        elif direction == 3: # transpose; loop over rows
            ssdd_tensor_transformed = np.transpose(ssdd_tensor, axes=[1, 0, 2])
            num_of_rows, num_of_cols = ssdd_tensor_transformed.shape[0], ssdd_tensor_transformed.shape[1]
            for i in range(num_of_rows):
                c_slice = ssdd_tensor_transformed[i, :, :].transpose()
                c_slices.append(c_slice)
        elif direction == 4: # flip cols; loop over diags
            ssdd_tensor_transformed = np.flip(ssdd_tensor, axis=1)
            num_of_rows, num_of_cols = ssdd_tensor_transformed.shape[0], ssdd_tensor_transformed.shape[1]
            for offset in range(-(num_of_rows - 1), num_of_cols):
                c_slice = np.diagonal(ssdd_tensor, offset=offset)
                c_slices.append(c_slice)
        elif direction == 5: # flip cols; transpose; loop over rows
            ssdd_tensor_transformed = np.flip(ssdd_tensor, axis=1)
            num_of_rows, num_of_cols = ssdd_tensor_transformed.shape[0], ssdd_tensor_transformed.shape[1]
            for i in range(num_of_rows):
                c_slice = ssdd_tensor[i, :, :].transpose()
                c_slices.append(c_slice)
        elif direction == 6: # flip cols and flip rows; loop over diags
            ssdd_tensor_transformed = np.flip(np.flip(ssdd_tensor, axis=1), axis=0)
            num_of_rows, num_of_cols = ssdd_tensor_transformed.shape[0], ssdd_tensor_transformed.shape[1]
            for offset in range(-(num_of_rows - 1), num_of_cols):
                c_slice = np.diagonal(ssdd_tensor, offset=offset)
                c_slices.append(c_slice)
        elif direction == 7: # transpose and flip rows; loop over rows
            ssdd_tensor_transformed = np.flip(np.transpose(ssdd_tensor, axes=[1, 0, 2]), axis=0)
            num_of_rows, num_of_cols = ssdd_tensor_transformed.shape[0], ssdd_tensor_transformed.shape[1]
            for i in range(num_of_rows):
                c_slice = ssdd_tensor_transformed[i, :, :].transpose()
                c_slices.append(c_slice)
        elif direction == 8: # flip rows; loop over diags
            ssdd_tensor_transformed = np.flip(ssdd_tensor, axis=0)
            num_of_rows, num_of_cols = ssdd_tensor_transformed.shape[0], ssdd_tensor_transformed.shape[1]
            for offset in range(-(num_of_rows - 1), num_of_cols):
                c_slice = np.diagonal(ssdd_tensor, offset=offset)
                c_slices.append(c_slice)

        return c_slices

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)
