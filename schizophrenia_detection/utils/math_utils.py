"""
Mathematical utility functions
"""

import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform


def calculate_correlation_matrix(data, method="pearson"):
    """
    Calculate correlation matrix

    Args:
        data: Input data (samples x features)
        method (str): Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation matrix
    """
    if method == "pearson":
        return np.corrcoef(data.T)
    elif method == "spearman":
        return stats.spearmanr(data, axis=0).correlation
    elif method == "kendall":
        return stats.kendalltau(data, axis=0).correlation
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def calculate_partial_correlation(data, method="pearson"):
    """
    Calculate partial correlation matrix

    Args:
        data: Input data (samples x features)
        method (str): Correlation method ('pearson', 'spearman')

    Returns:
        Partial correlation matrix
    """
    # Calculate precision matrix (inverse of covariance matrix)
    cov_matrix = np.cov(data.T)
    precision_matrix = np.linalg.inv(cov_matrix)

    # Calculate partial correlations
    partial_corr = np.zeros_like(precision_matrix)
    for i in range(precision_matrix.shape[0]):
        for j in range(precision_matrix.shape[1]):
            if i != j:
                partial_corr[i, j] = -precision_matrix[i, j] / np.sqrt(
                    precision_matrix[i, i] * precision_matrix[j, j]
                )

    return partial_corr


def calculate_connectivity_matrix(data, method="correlation"):
    """
    Calculate connectivity matrix from time series data

    Args:
        data: Input data (regions x time)
        method (str): Connectivity method ('correlation', 'partial_correlation', 'coherence', 'phase_locking')

    Returns:
        Connectivity matrix
    """
    if method == "correlation":
        return calculate_correlation_matrix(data.T)
    elif method == "partial_correlation":
        return calculate_partial_correlation(data.T)
    elif method == "coherence":
        # Calculate coherence between all pairs of time series
        n_regions = data.shape[0]
        connectivity = np.zeros((n_regions, n_regions))

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                f, Cxy = signal.coherence(data[i], data[j], fs=1.0)
                # Use mean coherence across all frequencies
                connectivity[i, j] = np.mean(Cxy)
                connectivity[j, i] = connectivity[i, j]

        return connectivity
    elif method == "phase_locking":
        # Calculate phase locking value between all pairs of time series
        n_regions = data.shape[0]
        connectivity = np.zeros((n_regions, n_regions))

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                # Compute analytic signal using Hilbert transform
                from scipy.signal import hilbert

                phase_i = np.angle(hilbert(data[i]))
                phase_j = np.angle(hilbert(data[j]))

                # Calculate phase locking value
                phase_diff = phase_i - phase_j
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))

                connectivity[i, j] = plv
                connectivity[j, i] = plv

        return connectivity
    else:
        raise ValueError(f"Unknown connectivity method: {method}")


def calculate_graph_metrics(connectivity_matrix):
    """
    Calculate graph metrics from connectivity matrix

    Args:
        connectivity_matrix: Connectivity matrix

    Returns:
        Dictionary of graph metrics
    """
    # Calculate degree
    degree = np.sum(connectivity_matrix > 0, axis=1)

    # Calculate strength (weighted degree)
    strength = np.sum(connectivity_matrix, axis=1)

    # Calculate clustering coefficient
    clustering = np.zeros(connectivity_matrix.shape[0])
    for i in range(connectivity_matrix.shape[0]):
        neighbors = np.where(connectivity_matrix[i] > 0)[0]
        if len(neighbors) > 1:
            subgraph = connectivity_matrix[np.ix_(neighbors, neighbors)]
            clustering[i] = np.sum(subgraph) / (len(neighbors) * (len(neighbors) - 1))

    # Calculate betweenness centrality
    betweenness = np.zeros(connectivity_matrix.shape[0])
    try:
        import networkx as nx

        G = nx.from_numpy_array(connectivity_matrix)
        betweenness_dict = nx.betweenness_centrality(G, weight="weight")
        betweenness = np.array(
            [betweenness_dict.get(i, 0) for i in range(connectivity_matrix.shape[0])]
        )
    except ImportError:
        pass

    # Calculate efficiency
    efficiency = np.zeros(connectivity_matrix.shape[0])
    for i in range(connectivity_matrix.shape[0]):
        # Calculate shortest path lengths from node i
        path_lengths = np.zeros(connectivity_matrix.shape[0])
        for j in range(connectivity_matrix.shape[0]):
            if i != j:
                # Use inverse of connectivity as distance
                distance = 1 / (connectivity_matrix[i, j] + 1e-8)
                path_lengths[j] = distance

        # Calculate efficiency as mean of inverse path lengths
        path_lengths[path_lengths == 0] = np.inf
        efficiency[i] = np.mean(1 / path_lengths[path_lengths != np.inf])

    return {
        "degree": degree,
        "strength": strength,
        "clustering": clustering,
        "betweenness": betweenness,
        "efficiency": efficiency,
    }


def calculate_spectral_power(data, sampling_rate, method="welch"):
    """
    Calculate spectral power of time series data

    Args:
        data: Input time series data
        sampling_rate (float): Sampling rate in Hz
        method (str): Method for spectral estimation ('welch', 'periodogram')

    Returns:
        Frequencies and power spectral density
    """
    if method == "welch":
        frequencies, psd = signal.welch(data, fs=sampling_rate, axis=-1)
    elif method == "periodogram":
        frequencies, psd = signal.periodogram(data, fs=sampling_rate, axis=-1)
    else:
        raise ValueError(f"Unknown spectral estimation method: {method}")

    return frequencies, psd


def calculate_band_power(psd, frequencies, bands):
    """
    Calculate power in specific frequency bands

    Args:
        psd: Power spectral density
        frequencies: Frequency array
        bands (dict): Dictionary of frequency bands

    Returns:
        Dictionary of band powers
    """
    band_powers = {}

    for band_name, (low_freq, high_freq) in bands.items():
        # Find frequency indices
        idx = np.logical_and(frequencies >= low_freq, frequencies <= high_freq)

        # Calculate mean power in the band
        if len(psd.shape) > 1:
            band_powers[band_name] = np.mean(psd[:, idx], axis=1)
        else:
            band_powers[band_name] = np.mean(psd[idx])

    return band_powers


def calculate_entropy(data, method="shannon"):
    """
    Calculate entropy of data

    Args:
        data: Input data
        method (str): Entropy method ('shannon', 'sample', 'approximate', 'spectral')

    Returns:
        Entropy value
    """
    if method == "shannon":
        # Calculate histogram
        hist, _ = np.histogram(data, bins=256, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities

        # Calculate Shannon entropy
        return -np.sum(hist * np.log2(hist))

    elif method == "sample":
        try:
            from antropy import sample_entropy

            return sample_entropy(data)
        except ImportError:
            raise ImportError("Antropy package is required for sample entropy")

    elif method == "approximate":
        try:
            from antropy import app_entropy

            return app_entropy(data)
        except ImportError:
            raise ImportError("Antropy package is required for approximate entropy")

    elif method == "spectral":
        # Calculate power spectral density
        frequencies, psd = signal.welch(data, fs=1.0)

        # Normalize PSD
        psd_norm = psd / np.sum(psd)

        # Calculate spectral entropy
        return -np.sum(psd_norm * np.log2(psd_norm))

    else:
        raise ValueError(f"Unknown entropy method: {method}")


def calculate_fractal_dimension(data, method="higuchi"):
    """
    Calculate fractal dimension of data

    Args:
        data: Input data
        method (str): Fractal dimension method ('higuchi', 'boxcount', 'dfa')

    Returns:
        Fractal dimension value
    """
    if method == "higuchi":
        return higuchi_fd(data)
    elif method == "boxcount":
        return boxcount_fd(data)
    elif method == "dfa":
        return dfa(data)
    else:
        raise ValueError(f"Unknown fractal dimension method: {method}")


def higuchi_fd(data, k_max=10):
    """
    Calculate Higuchi fractal dimension

    Args:
        data: Input data
        k_max (int): Maximum k value

    Returns:
        Higuchi fractal dimension
    """
    N = len(data)
    L = []

    for k in range(1, k_max + 1):
        Lk = 0

        for m in range(k):
            # Construct k subseries
            idx = np.arange(1, int((N - m) / k), dtype=int)
            subseries = data[m + idx * k]

            # Calculate length of subseries
            if len(subseries) > 1:
                diff = np.abs(np.diff(subseries))
                Lk += np.sum(diff)

        # Normalize length
        Lk = Lk * N / (len(idx) * k)
        L.append(np.log(Lk))

    # Fit line to log(L) vs log(1/k)
    k_values = np.arange(1, k_max + 1)
    coeffs = np.polyfit(np.log(1 / k_values), L, 1)

    return coeffs[0]


def boxcount_fd(data, box_sizes=None):
    """
    Calculate box-counting fractal dimension

    Args:
        data: Input data
        box_sizes (list): List of box sizes

    Returns:
        Box-counting fractal dimension
    """
    if box_sizes is None:
        box_sizes = [2, 4, 8, 16, 32, 64]

    N = len(data)
    counts = []

    for box_size in box_sizes:
        # Calculate number of boxes needed
        count = 0
        for i in range(0, N, box_size):
            # Find min and max in the box
            box_data = data[i : i + box_size]
            if len(box_data) > 0:
                min_val = np.min(box_data)
                max_val = np.max(box_data)

                # Calculate number of boxes in y direction
                if max_val > min_val:
                    y_boxes = int(np.ceil((max_val - min_val) / box_size))
                    count += y_boxes
                else:
                    count += 1

        counts.append(count)

    # Fit line to log(count) vs log(1/box_size)
    coeffs = np.polyfit(np.log(1 / np.array(box_sizes)), np.log(counts), 1)

    return coeffs[0]


def dfa(data):
    """
    Calculate detrended fluctuation analysis

    Args:
        data: Input data

    Returns:
        DFA exponent
    """
    N = len(data)

    # Calculate cumulative sum
    cumsum = np.cumsum(data - np.mean(data))

    # Calculate fluctuation for different window sizes
    window_sizes = np.logspace(np.log10(10), np.log10(N // 4), 20).astype(int)
    fluctuations = []

    for window_size in window_sizes:
        # Split data into windows
        n_windows = N // window_size
        if n_windows < 2:
            continue

        # Calculate fluctuation for each window
        fluctuation = 0
        for i in range(n_windows):
            start = i * window_size
            end = (i + 1) * window_size

            # Fit linear trend
            x = np.arange(window_size)
            y = cumsum[start:end]
            coeffs = np.polyfit(x, y, 1)
            trend = np.polyval(coeffs, x)

            # Calculate RMS fluctuation
            fluctuation += np.sqrt(np.mean((y - trend) ** 2))

        fluctuations.append(fluctuation / n_windows)

    # Fit line to log(fluctuation) vs log(window_size)
    coeffs = np.polyfit(
        np.log(window_sizes[: len(fluctuations)]), np.log(fluctuations), 1
    )

    return coeffs[0]


def calculate_mutual_information(x, y, bins=10):
    """
    Calculate mutual information between two variables

    Args:
        x: First variable
        y: Second variable
        bins (int): Number of bins for histogram

    Returns:
        Mutual information value
    """
    # Calculate joint histogram
    hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    # Calculate marginal histograms
    hist_x, _ = np.histogram(x, bins=x_edges)
    hist_y, _ = np.histogram(y, bins=y_edges)

    # Convert to probabilities
    p_xy = hist_xy / np.sum(hist_xy)
    p_x = hist_x / np.sum(hist_x)
    p_y = hist_y / np.sum(hist_y)

    # Calculate mutual information
    mi = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

    return mi
