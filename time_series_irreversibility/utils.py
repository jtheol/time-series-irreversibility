import numpy as np
from ts2vg import HorizontalVG


def logistic_map(size: int, x_0: float = 0.3, r: float = 4) -> np.ndarray:
    """Generates a data set using the logistic map.

    Args:
        n (int): Number of time steps
        x (float, optional): Initial population value. Defaults to 0.3.
        r (float, optional): Growth rate. Defaults to 4.

    Returns:

        data (np.ndarray): An array containing the sequence of population values over n iterations.
    """
    data = np.zeros(size)
    data[0] = x_0

    for i in range(1, size):
        # x_n+1 = r*x_n * ( 1 - x_n)
        data[i] = r * data[i - 1] * (1 - data[i - 1])

    return data


def henon_map(
    size: int, x: float = 0.0, y: float = 0.0, a: float = 1.4, b: float = 0.3
) -> np.ndarray:
    """Generates a data set using the Hénon map.
    The henon map is chaotic at values a = 1.4 and b = 0.3

    Args:
        size (int): Number of data points to generate
        x (float, optional): Defaults to 0.0.
        y (float, optional): Defaults to 0.0.
        a (float, optional): Defaults to 1.4.
        b (float, optional): Defaults to 0.3.

    Returns:
        data (np.ndarray)
    """
    data = np.zeros((size, 2))

    for i in range(size):
        x_n = 1 - a * x**2 + y
        y_n = b * x
        data[i] = [x_n, y_n]

        x, y = x_n, y_n

    return data[:, 0]


def calc_outgoing_links(adj_matrix: np.array) -> np.ndarray:
    """Calculate the outgoing links from an adjacency matrix

    Args:
        adj_matrix (np.ndarray): The adjacency matrix from a horizontal visibility graph.

    Returns:
        outgoing_links (np.ndarray): Array of the number of outgoing links per node.
    """
    n = len(adj_matrix)
    sums = []

    for i in range(n):
        sums.append(np.sum(adj_matrix[i, i + 1 :]))

    return np.array(sums)


def adjacency_matrix(forward_series: np.array):
    """Calculate adjacency matrix for the forward and reversed time series.

    Args:
        data (np.ndarray): Time series array of values.

    Returns:
        adj_matrix_forward (np.ndarray), adj_matrix_reversed (np.array): Returns both the adjacency matrix for the forward and reversed series.
    """

    reversed_series = np.flip(forward_series)

    vg_forward = HorizontalVG()
    vg_forward.build(forward_series)
    adj_matrix_forward = vg_forward.adjacency_matrix()

    vg_reversed = HorizontalVG()
    vg_reversed.build(reversed_series)
    adj_matrix_reversed = vg_reversed.adjacency_matrix()

    return adj_matrix_forward, adj_matrix_reversed


def calc_async_index(out_forward: np.array, out_reversed: np.array) -> float:
    """Calculate the asynchronous index, which can be used to estimate the synchronization difference between two time series.
    Large asynchronous index values indicate weak synchronization between one sequence and another.

    Args:
        forward_series (np.ndarray): The adjacency matrix of the forward series
        reversed_series (np.ndarray): The adjacency matrix of the reversed series

    Returns:
        ai (float): Returns the asynchronous index.
    """

    n = len(out_forward)
    forward_sort = np.argsort(out_forward)

    delta = (n * (n - 1)) / 2

    inversion_n = 0

    for i in range(n):
        for j in range(i + 1, n):
            diff = out_reversed[forward_sort[i]] - out_reversed[forward_sort[j]]
            if diff > 0:
                inversion_n += 1

    return inversion_n / delta


def calc_relative_async_index(out_forward: np.array, out_reversed: np.array) -> float:
    """Calculate the relative asynchronous index.

    Args:
        out_forward (np.ndarray): Array of the outward links for the forward data.
        out_reversed (np.ndarray): Array of the outward links for the reversed data.

    Returns:
        rai (float) Returns the relative asynchronous index.
    """

    async_forward_reversed = calc_async_index(out_forward, out_reversed)
    async_reversed_forward = calc_async_index(out_reversed, out_forward)

    rai = -np.log(
        min(async_forward_reversed, async_reversed_forward)
        / max(async_forward_reversed, async_reversed_forward)
    )

    return rai
