import numpy as np

def likelihood_ratio_test(phases, coherence):
    """
    Implements the Likelihood Ratio Test (LRT) for phase-based noise detection.
    """
    gamma = np.abs(np.mean(np.exp(1j * phases), axis=0))  # Temporal coherence estimate
    lrt = np.log(coherence) - np.log(gamma)  # LRT formula
    return lrt
