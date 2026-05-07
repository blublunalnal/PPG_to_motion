import numpy as np
import torch


def generate_differential_signal_image(ppg_signal):
    """
    Generates a 3-channel signal image from a 1D PPG signal.

    Returns:
    np.ndarray: shape (3, N) — [original, first-order gradient, second-order gradient]
    """
    channel_1 = ppg_signal
    channel_2 = np.gradient(ppg_signal)
    channel_3 = np.gradient(channel_2)
    return np.stack([channel_1, channel_2, channel_3], axis=0)


def normalize_multichannel_image(signal_image, method='minmax'):
    """
    Normalizes the signal image channel-wise.

    Parameters:
    signal_image (np.ndarray): shape (C, ...) — any number of trailing dims.
    method (str): 'minmax' for [0, 1] or 'zscore' for zero-mean/unit-variance.

    Returns:
    np.ndarray: Normalized signal image, float32.
    """
    if method not in ('minmax', 'zscore'):
        raise ValueError(f"Unknown normalization method: {method!r}. Use 'minmax' or 'zscore'.")

    normalized_image = np.zeros_like(signal_image, dtype=np.float32)

    for i in range(signal_image.shape[0]):
        channel = signal_image[i]

        if method == 'minmax':
            c_min = np.min(channel)
            c_max = np.max(channel)
            if (c_max - c_min) > 1e-9:
                normalized_image[i] = (channel - c_min) / (c_max - c_min)
            else:
                normalized_image[i] = 0.0

        elif method == 'zscore':
            mean = np.mean(channel)
            std = np.std(channel)
            if std > 1e-9:
                normalized_image[i] = (channel - mean) / std
            else:
                normalized_image[i] = 0.0

    return normalized_image


_EPS = 1e-9


def generate_stft_features(ppg_signal, fs=40, n_fft=128, hop_length=16, win_length=None):
    """
    Generates STFT features matching tune_dinov3.py _build_stft_complex3:
      I1 = z(log(|X(f,t)| + eps))   — z-scored log-magnitude
      I2 = z(cos(phi(f,t)))          — z-scored cosine phase
      I3 = z(sin(phi(f,t)))          — z-scored sine phase
    where phi = atan2(Im X, Re X).

    Returns:
    np.ndarray: shape (3, n_freqs, n_frames), float32, each channel z-scored
    """
    win_length = win_length or n_fft
    x = torch.from_numpy(np.asarray(ppg_signal, dtype=np.float32))
    hann = torch.hann_window(win_length)

    with torch.no_grad():
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=hann,
            return_complex=True,
            center=True,
            normalized=False,
            onesided=True,
        )  # [F, T]

        mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2))
        log_mag = torch.log(mag + _EPS)

        phase = torch.atan2(spec.imag, spec.real)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        def _z(t):
            return (t - t.mean()) / (t.std() + _EPS)

        out = torch.stack([_z(log_mag), _z(cos_phase), _z(sin_phase)], dim=0)

    return out.numpy().astype(np.float32)


def create_signal_image(ppg_window, fs=40):
    """
    Creates signal images for a PPG window.

    Returns:
    dict with:
      'diff_signal_image': np.ndarray (3, N), float32, z-scored per channel
      'stft_image':        np.ndarray (3, n_freqs, n_frames), float32, z-scored per channel
    """
    diff_signal_image = generate_differential_signal_image(ppg_window)
    diff_signal_image = normalize_multichannel_image(diff_signal_image, method='zscore')

    stft_image = generate_stft_features(ppg_window, fs=fs)

    return {
        'diff_signal_image': diff_signal_image,
        'stft_image': stft_image,
    }
