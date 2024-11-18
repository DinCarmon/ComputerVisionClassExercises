"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        if match_p_src.shape != match_p_dst.shape:
            raise TypeError("Mismatch in dimensions of source pairs matrix, and destination pairs matrix")
        num_of_pairs = match_p_src.shape[1]
        if match_p_src.shape[0] != 2:
            raise TypeError("Src pairs matrix should have 2 rows")
        if num_of_pairs < 4:
            raise TypeError("Src pairs matrix should have at least 4 pairs of points")

        # The matrix A is built of rows repesenting equations H need to hold.
        A = np.zeros((2 * num_of_pairs, 9))
        for i in range(num_of_pairs):
            A[2 * i, 0] = match_p_src[0, i]
            A[2 * i, 1] = match_p_src[1, i]
            A[2 * i, 2] = 1
            A[2 * i, 6] = -match_p_dst[0,i] * match_p_src[0,i]
            A[2 * i, 7] = -match_p_dst[0, i] * match_p_src[1, i]
            A[2 * i, 8] = -match_p_dst[0, i]
            A[2 * i + 1, 3] = match_p_src[0, i]
            A[2 * i + 1, 4] = match_p_src[1, i]
            A[2 * i + 1, 5] = 1
            A[2 * i + 1, 6] = -match_p_dst[1, i] * match_p_src[0, i]
            A[2 * i + 1, 7] = -match_p_dst[1, i] * match_p_src[1, i]
            A[2 * i + 1, 8] = -match_p_dst[1, i]

        M = np.transpose(A) @ A

        eigenvalues, eigenvectors = np.linalg.eig(M)

        if eigenvalues.shape[0] < 9:
            raise RuntimeError("The equations matrix for finding H is defective. Warning")
        minimal_eigenvalue_eigenvector = eigenvectors[:, np.argmin(eigenvalues)]
        if not np.isclose(np.linalg.norm(minimal_eigenvalue_eigenvector), 1):
            raise RuntimeError("The chosen eigenvector repesenting H is not of magnitude 1. Aborting.")

        H = np.zeros((3, 3))
        H[0, 0] = minimal_eigenvalue_eigenvector[0]
        H[0, 1] = minimal_eigenvalue_eigenvector[1]
        H[0, 2] = minimal_eigenvalue_eigenvector[2]
        H[1, 0] = minimal_eigenvalue_eigenvector[3]
        H[1, 1] = minimal_eigenvalue_eigenvector[4]
        H[1, 2] = minimal_eigenvalue_eigenvector[5]
        H[2, 0] = minimal_eigenvalue_eigenvector[6]
        H[2, 1] = minimal_eigenvalue_eigenvector[7]
        H[2, 2] = minimal_eigenvalue_eigenvector[8]

        # Intend for h22 to be 1. This is a common arbitrary default, in some places. However,
        # a better rule of the thumb is to keep the norm of H to 1. Therefore, the following line is
        # commented out.
        # H = H / minimal_eigenvalue_eigenvector[8]

        return H

    @staticmethod
    def compute_homography_naive_basic_test():
        src_points = np.transpose(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        trg_points = np.transpose(np.array([[0, 0], [2, 0], [2, 2], [0, 2]]))
        H = Solution.compute_homography_naive(src_points, trg_points)
        print("H: \n", H)
        expected_H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        print("expected H: \n", expected_H)
        if np.allclose(H / H[2,2], expected_H):
            print("Matrices are equal!!")
        else:
            print("Matrices are not equal!!")

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.


        Returns:
            The forward homography of the source image to its destination.
        """
        if homography.shape != (3, 3):
            raise TypeError("Homographic transform should be of size 3x3")
        if src_image.shape[2] != dst_image_shape[2]:
            raise TypeError("mismatch in number of colors")

        projected_image = np.zeros(shape = dst_image_shape, dtype = np.uint8)

        for height_idx in range(src_image.shape[0]):
            for width_idx in range(src_image.shape[1]):
                projected_pixel_idx_vector = homography @ np.transpose([[width_idx, height_idx, 1]])
                normalized_projected_pixel_idx_vector = projected_pixel_idx_vector / projected_pixel_idx_vector[2, 0]
                rounded_projected_coord_vector = np.round(normalized_projected_pixel_idx_vector).astype(int)
                # only copy content if coord is withing range
                if 0 <= rounded_projected_coord_vector[0] < dst_image_shape[1] and \
                    0 <= rounded_projected_coord_vector[1] < dst_image_shape[0]:
                    projected_image[rounded_projected_coord_vector[1], rounded_projected_coord_vector[0], :] = src_image[height_idx, width_idx, :]
        return projected_image


    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        if homography.shape != (3, 3):
            raise TypeError("Homographic transform should be of size 3x3")
        if src_image.shape[2] != dst_image_shape[2]:
            raise TypeError("mismatch in number of colors")

        projected_image = np.zeros(shape=dst_image_shape, dtype=np.uint8)

        # xx is of pattern: 0,0,0,...0    ,1,1,...,1  ,2,...
        # yy is of pattern: 0,1,2,...100  ,0,1,...,100,0,...
        xx, yy = np.meshgrid(range(src_image.shape[1]), range(src_image.shape[0]))
        xx = xx.reshape((xx.shape[0] * xx.shape[1]))
        yy = yy.reshape((yy.shape[0] * yy.shape[1]))
        ones = np.ones(xx.shape[0])
        all_coords = np.stack((xx, yy, ones))
        projected_coords = homography @ all_coords
        projected_coords = np.round(projected_coords / projected_coords[2])[:2,:]

        # each column is the projected coords and the original ones after them. Dimensions: [4, num_of_projected_pixels]
        projected_coords_and_original_coords = np.append(projected_coords, all_coords[:2, :], axis=0).astype(int)

        # filter points which are outside the boundary
        projected_coords_and_original_coords = projected_coords_and_original_coords[:, projected_coords_and_original_coords[0] > 0]
        projected_coords_and_original_coords = projected_coords_and_original_coords[:, projected_coords_and_original_coords[1] > 0]
        projected_coords_and_original_coords = projected_coords_and_original_coords[:, projected_coords_and_original_coords[0] < dst_image_shape[1]]
        projected_coords_and_original_coords = projected_coords_and_original_coords[:, projected_coords_and_original_coords[1] < dst_image_shape[0]]

        # Copy the data from src to dst
        projected_image[projected_coords_and_original_coords[1, :],\
                        projected_coords_and_original_coords[0, :], :] = \
            src_image[projected_coords_and_original_coords[3, :],\
                        projected_coords_and_original_coords[2, :], :]

        return projected_image


    @staticmethod
    def calculate_distance(cord1, cord2) -> float:
        """Calculate the pixel distance between cord1 and cord2

        Args:
            cord1: array of size 2 representing a pixel
            cord2: array of size 2 representing a pixel

        Returns:
            A float representing the distance between the points
        """
        if cord1.shape[0] != 2 or cord2.shape[0] != 2:
            raise TypeError("cords are 2D")

        square_dist = (cord1[0] - cord2[0]) ** 2 + \
                      (cord1[1] - cord2[1]) ** 2
        return square_dist ** 0.5


    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        if homography.shape != (3, 3):
            raise TypeError("Homographic transform should be of size 3x3")
        if match_p_src.shape[0] != 2:
            raise TypeError("src points matrix should be of diensions 2xN")
        if match_p_src.shape != match_p_dst.shape:
            raise TypeError("Mismatch in size of matrices")

        num_of_pairs = match_p_src.shape[1]
        match_p_src_vector = np.stack((match_p_src[0], match_p_src[1], np.ones(num_of_pairs)))
        projected_coords_vector = homography @ match_p_src_vector
        projected_coords_vector_normalized = projected_coords_vector / projected_coords_vector[2]
        projected_coords = projected_coords_vector_normalized[:2]
        num_of_inliers = 0
        sum_of_inliers_square_error = 0
        for i in range(num_of_pairs):
            pixel_distance = Solution.calculate_distance(match_p_dst[:, i], projected_coords[:, i])
            # if it is an inlier
            if pixel_distance <= max_err:
                num_of_inliers += 1
                sum_of_inliers_square_error += pixel_distance ** 2

        if num_of_inliers == 0:
            return 0, 10 ** 9

        return num_of_inliers / num_of_pairs, \
                sum_of_inliers_square_error / num_of_inliers


    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points that meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        if homography.shape != (3, 3):
            raise TypeError("Homographic transform should be of size 3x3")
        if match_p_src.shape[0] != 2:
            raise TypeError("src points matrix should be of dimensions 2xN")
        if match_p_src.shape != match_p_dst.shape:
            raise TypeError("Mismatch in size of matrices")

        num_of_pairs = match_p_src.shape[1]
        match_p_src_vector = np.stack((match_p_src[0], match_p_src[1], np.ones(num_of_pairs)))
        projected_coords_vector = homography @ match_p_src_vector
        projected_coords_vector_normalized = projected_coords_vector / projected_coords_vector[2]
        projected_coords = projected_coords_vector_normalized[:2]

        mp_src_meets_model = []
        mp_dst_meets_model = []

        num_of_inliers = 0
        for i in range(num_of_pairs):
            pixel_distance = Solution.calculate_distance(match_p_dst[:, i], projected_coords[:, i])
            # if it is an inlier
            if pixel_distance <= max_err:
                num_of_inliers += 1
                mp_src_meets_model.append(match_p_src[:, i])
                mp_dst_meets_model.append(match_p_dst[:, i])

        if num_of_inliers > 0:
            mp_src_meets_model = np.array(mp_src_meets_model).T
            mp_dst_meets_model = np.array(mp_dst_meets_model).T
        else:
            mp_src_meets_model = np.empty(shape=(2, 0))
            mp_dst_meets_model = np.empty(shape=(2, 0))

        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1

        if match_p_src.shape[0] != 2:
            raise TypeError("src points matrix should be of dimensions 2xN")
        if match_p_src.shape != match_p_dst.shape:
            raise TypeError("Mismatch in size of matrices")

        homography = np.zeros((3, 3))
        best_mse = np.inf
        num_of_pairs = match_p_src.shape[1]

        for iteration in range(k):
            points_sample = sample(range(num_of_pairs), 4)
            match_p_src_sample = match_p_src[:, points_sample]
            match_p_dst_sample = match_p_dst[:, points_sample]
            initial_homography = self.compute_homography_naive(match_p_src_sample, match_p_dst_sample)
            fit_percent, _ = self.test_homography(initial_homography, match_p_src, match_p_dst, max_err)

            # update model if # inliers > d
            if fit_percent > d:
                mp_src_meets_model, mp_dst_meets_model = self.meet_the_model_points(initial_homography, match_p_src, match_p_dst, max_err)
                improved_homography = self.compute_homography_naive(mp_src_meets_model, mp_dst_meets_model)
                fit_percent, dist_mse = self.test_homography(improved_homography, match_p_src, match_p_dst, max_err)
                if dist_mse < best_mse:
                    best_mse = dist_mse
                    homography = improved_homography

        if best_mse == np.inf:
            raise RuntimeError("RANSAC algorithm failed.")
        return homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        if backward_projective_homography.shape != (3, 3):
            raise TypeError("Homographic transform should be of size 3x3")
        if src_image.shape[2] != dst_image_shape[2]:
            raise TypeError("mismatch in number of colors")

        projected_image = np.zeros(shape=dst_image_shape, dtype=np.uint8)

        # xx is of pattern: 0,0,0,...0    ,1,1,...,1  ,2,...
        # yy is of pattern: 0,1,2,...100  ,0,1,...,100,0,...
        xx, yy = np.meshgrid(range(dst_image_shape[1]), range(dst_image_shape[0]))
        xx = xx.reshape((xx.shape[0] * xx.shape[1]))
        yy = yy.reshape((yy.shape[0] * yy.shape[1]))
        ones = np.ones(xx.shape[0])
        all_coords = np.stack((xx, yy, ones))
        src_projected_coords = backward_projective_homography @ all_coords
        src_projected_coords = (src_projected_coords / src_projected_coords[2])[:2,:]

        # each column is the projected coords and the original ones after them. Dimensions: [4, num_of_projected_pixels]
        projected_coords_and_original_coords = np.append(src_projected_coords, all_coords[:2, :], axis=0)

        # filter points which are outside the boundary
        projected_coords_and_original_coords = projected_coords_and_original_coords[:, projected_coords_and_original_coords[0] > 0]
        projected_coords_and_original_coords = projected_coords_and_original_coords[:, projected_coords_and_original_coords[1] > 0]
        projected_coords_and_original_coords = projected_coords_and_original_coords[:, projected_coords_and_original_coords[0] < src_image.shape[1]]
        projected_coords_and_original_coords = projected_coords_and_original_coords[:, projected_coords_and_original_coords[1] < src_image.shape[0]]

        # mesh-grid of source image coordinates
        src_x, src_y = projected_coords_and_original_coords[0], projected_coords_and_original_coords[1]
        dst_x, dst_y = projected_coords_and_original_coords[2], projected_coords_and_original_coords[3]

        meshgrid_x_cords, meshgrid_y_cords = np.meshgrid(range(src_image.shape[1]), range(src_image.shape[0]))
        meshgrid_x_cords = meshgrid_x_cords.reshape((meshgrid_x_cords.shape[0] * meshgrid_x_cords.shape[1]))
        meshgrid_y_cords = meshgrid_y_cords.reshape((meshgrid_y_cords.shape[0] * meshgrid_y_cords.shape[1]))

        for channel in range(3):
            interpolated_values = griddata(
                points=np.stack((meshgrid_x_cords, meshgrid_y_cords), axis=-1),
                values=src_image[meshgrid_y_cords, meshgrid_x_cords, channel],
                xi=np.stack((src_x, src_y), axis=-1),
                method='cubic',
                fill_value=0
            )
            interpolated_values = np.clip(np.round(interpolated_values), 0, 255).astype(int)
            projected_image[dst_y.astype(int), dst_x.astype(int), channel] = interpolated_values

        return projected_image

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        translation_matrix = np.array([
            [1, 0, -pad_left],
            [0, 1, -pad_up],
            [0, 0, 1]
        ])

        final_homography = backward_homography @ translation_matrix
        final_homography = final_homography / np.linalg.norm(final_homography)

        return final_homography

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        forward_homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        backward_homography = self.compute_homography(match_p_dst, match_p_src, inliers_percent, max_err)

        panorama_rows_num, panorama_cols_num, pad_struct = self.find_panorama_shape(src_image, dst_image, forward_homography)

        # Add a linear projection from the dst image space to the built panorama image space
        pad_left = pad_struct.pad_left
        pad_up = pad_struct.pad_up
        homography_with_translation = self.add_translation_to_backward_homography(backward_homography,
                                                                                  pad_left, pad_up)

        # Compute final projection of the src image to the destination panorama space
        panorama_shape = (panorama_rows_num, panorama_cols_num, 3)
        img_panorama = self.compute_backward_mapping(homography_with_translation, src_image, panorama_shape)

        # Add the pixels from the dst image to the panorama
        img_panorama[pad_up:(dst_image.shape[0] + pad_up), pad_left:(dst_image.shape[1] + pad_left), :] = dst_image[:, :, :]

        return np.clip(img_panorama, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    Solution.compute_homography_naive_basic_test()