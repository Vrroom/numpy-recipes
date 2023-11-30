import numpy as np

def perspective_multiply(M, x):
    """
    Apply a perspective transformation to a set of 2D points.

    Args:
        M (numpy.ndarray): A 3x3 matrix representing the perspective transformation.
                          The matrix should be in the form of a numpy array with shape [3, 3].
        x (numpy.ndarray): An array of 2D points to be transformed.
                          Each point is represented as a row in this array, with the array having shape [N, 2],
                          where N is the number of points.

    Returns:
        numpy.ndarray: An array of transformed 2D points, with shape [N, 2].
                      The transformation is applied using the matrix M, and the points are then converted back to non-homogeneous coordinates.
    """
    N, _ = x.shape 
    x_hom = np.concatenate((x, np.ones((N, 1))), axis=1)
    y_hom = ((M @ x_hom.T).T)
    y = y_hom[:, :-1] / y_hom[:, -1:]
    return y

def find_first_true(arr):
    """
    Find the index of the first True value in a numpy array of boolean values.

    Args:
        arr (numpy.ndarray): A 1D numpy array of boolean values with shape (N,).

    Returns:
        int or None: The index of the first occurrence of a True value in the array.
                     If no True value is found, returns None.
    """
    indices = np.where(arr)[0]
    return indices[0] if indices.size > 0 else None


def find_first_false(arr):
    """
    Find the index of the first False value in a numpy array of boolean values.

    Args:
        arr (numpy.ndarray): A 1D numpy array of boolean values with shape (N,).

    Returns:
        int or None: The index of the first occurrence of a False value in the array.
                     If no False value is found, returns None.
    """
    indices = np.where(~arr)[0]
    return indices[0] if indices.size > 0 else None
    
def nd_argmin(arr):
    """
    Find the N-dimensional index of the minimum value in an N-dimensional array.

    Parameters:
    arr (np.ndarray): An N-dimensional NumPy array.

    Returns:
    tuple: An N-dimensional index tuple of the minimum value in the array.
    """
    # np.argmin returns the index of the minimum value in the flattened array
    flat_index_min = np.argmin(arr)

    # np.unravel_index converts the flat index to an N-dimensional index
    return np.unravel_index(flat_index_min, arr.shape)

def nd_argmax(arr):
    """
    Find the N-dimensional index of the maximum value in an N-dimensional array.

    Parameters:
    arr (np.ndarray): An N-dimensional NumPy array.

    Returns:
    tuple: An N-dimensional index tuple of the maximum value in the array.
    """
    # np.argmax returns the index of the maximum value in the flattened array
    flat_index_max = np.argmax(arr)

    # np.unravel_index converts the flat index to an N-dimensional index
    return np.unravel_index(flat_index_max, arr.shape)

