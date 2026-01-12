"""
Ultra-high-performance live EEG frequency band plotter using PyQtGraph
Displays power in frequency bands (Delta, Theta, Alpha, Beta, Gamma) in real-time
Can easily achieve 200+ FPS for real-time visualization
Install: pip install pyqtgraph PyQt5 scipy
"""
import sys
import argparse
import time
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from collections import deque
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    print("ERROR: PyQtGraph not installed!")
    print("Install with: pip install pyqtgraph PyQt5")
    sys.exit(1)

# EEG Frequency Bands (Hz)
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50)
}

# Band descriptions
BAND_DESCRIPTIONS = {
    'Delta': 'Sleep',
    'Theta': 'Memory',
    'Alpha': 'Relaxed',
    'Beta': 'Focus',
    'Gamma': 'Attention'
}

# Parse arguments
parser = argparse.ArgumentParser(description='Ultra-fast Muse EEG frequency band plotter')
parser.add_argument('-w', '--window', type=float, default=10.0,
                    help='Window length in seconds for display (default: 10)')
parser.add_argument('--fft-window', type=float, default=2.0,
                    help='Window length in seconds for FFT computation (default: 2.0)')
parser.add_argument('--update-rate', type=float, default=10.0,
                    help='Hz rate for computing band power (default: 10, range: 1-60)')
parser.add_argument('-f', '--fps', type=int, default=60,
                    help='Target FPS (default: 60)')
parser.add_argument('-c', '--channels', type=str, nargs='+', default=['TP9', 'AF7', 'AF8', 'TP10'],
                    help='Channels to analyze: TP9, AF7, AF8, TP10 (default: all)')
parser.add_argument('-b', '--bands', type=str, nargs='+',
                    default=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
                    help='Frequency bands to plot: Delta, Theta, Alpha, Beta, Gamma (default: all)')
parser.add_argument('--minimal', action='store_true',
                    help='Minimal UI - waveforms only')
parser.add_argument('--dark', action='store_true', default=True,
                    help='Dark theme (default)')
parser.add_argument('--show-fps', action='store_true',
                    help='Display live FPS counter on screen')
parser.add_argument('--average-channels', action='store_true',
                    help='Average selected channels before computing band power')
parser.add_argument('--smooth', type=int, default=0,
                    help='Smoothing window size in samples (0=no smoothing, default: 0)')
parser.add_argument('--colors', type=str, nargs='+', default=None,
                    help='Hex color codes for bands (e.g., FF0000 for red). Specify one per band.')
parser.add_argument('--crop', type=int, default=0,
                    help='Number of samples to crop from each edge to hide smoothing artifacts (default: 0)')
parser.add_argument('--linewidth', type=float, default=2.0,
                    help='Line width for waveforms (default: 2.0)')
parser.add_argument('--log-scale', action='store_true',
                    help='Use logarithmic scale for power (dB)')
parser.add_argument('--relative-power', action='store_true',
                    help='Show bands as percentage of total power (0-100%)')
parser.add_argument('--auto-scale', action='store_true',
                    help='Automatically scale Y-axis based on data')
parser.add_argument('--y-max', type=float, default=None,
                    help='Maximum Y-axis value (default: 100 for %, 100 for linear, 40 for dB)')
args = parser.parse_args()

# Check for conflicting options
if args.log_scale and args.relative_power:
    print("WARNING: --log-scale and --relative-power are both specified.")
    print("         Using --relative-power (percentage mode), ignoring --log-scale")
    print()

# Channel name to index mapping
CHANNEL_NAMES = ['TP9', 'AF7', 'AF8', 'TP10']
CHANNEL_MAP = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}

# Convert channel names to indices
CHANNELS_TO_ANALYZE = []
for ch_name in args.channels:
    ch_upper = ch_name.upper()
    if ch_upper not in CHANNEL_MAP:
        print(f"ERROR: Invalid channel name '{ch_name}'")
        print(f"Valid channels: {', '.join(CHANNEL_NAMES)}")
        sys.exit(1)
    CHANNELS_TO_ANALYZE.append(CHANNEL_MAP[ch_upper])

# Validate bands
BANDS_TO_PLOT = []
for band_name in args.bands:
    band_title = band_name.capitalize()
    if band_title not in BANDS:
        print(f"ERROR: Invalid band name '{band_name}'")
        print(f"Valid bands: {', '.join(BANDS.keys())}")
        sys.exit(1)
    BANDS_TO_PLOT.append(band_title)

# Parse and validate colors
def parse_hex_color_to_rgb(hex_str):
    """Convert hex color to RGB tuple for PyQtGraph"""
    hex_str = hex_str.strip()
    if hex_str.startswith('#'):
        hex_str = hex_str[1:]
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color: {hex_str} (must be 6 digits)")
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (r, g, b)
    except ValueError:
        raise ValueError(f"Invalid hex color: {hex_str}")

# Handle color specification
custom_colors = None
# Default colors for each band
DEFAULT_BAND_COLORS = {
    'Delta': (128, 0, 128),    # Purple
    'Theta': (0, 0, 255),       # Blue
    'Alpha': (0, 255, 0),       # Green
    'Beta': (255, 255, 0),      # Yellow
    'Gamma': (255, 0, 0)        # Red
}

if args.colors:
    if len(args.colors) != len(BANDS_TO_PLOT):
        print(f"ERROR: Expected {len(BANDS_TO_PLOT)} color(s) but got {len(args.colors)}")
        print(f"Specify one color per band: {', '.join(BANDS_TO_PLOT)}")
        sys.exit(1)

    try:
        custom_colors = [parse_hex_color_to_rgb(c) for c in args.colors]
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Example: --colors FF0000 00FF00 0000FF")
        sys.exit(1)

# Configuration
WINDOW_SECONDS = args.window
FFT_WINDOW_SECONDS = args.fft_window
TARGET_FPS = args.fps
SAMPLE_RATE = 256

DISPLAY_BUFFER_SIZE = int(WINDOW_SECONDS * TARGET_FPS)  # Store computed band powers
FFT_BUFFER_SIZE = int(FFT_WINDOW_SECONDS * SAMPLE_RATE)  # Raw data for FFT
UPDATE_INTERVAL_MS = int(1000.0 / TARGET_FPS)

# Band power computation interval (seconds)
# Only recompute band power this often to avoid over-smoothing from overlapping windows
BAND_POWER_UPDATE_INTERVAL = 1.0 / args.update_rate

# Determine Y-axis limits
if args.y_max is not None:
    Y_MAX = args.y_max
elif args.relative_power:
    Y_MAX = 100  # Percentage (0-100%)
elif args.log_scale:
    Y_MAX = 40  # dB
else:
    Y_MAX = 100  # Arbitrary units for power

print("=" * 70)
print("MUSE EEG FREQUENCY BAND PLOTTER - PyQtGraph Backend")
print("=" * 70)
print(f"Configuration:")
print(f"  - Display window: {WINDOW_SECONDS}s ({DISPLAY_BUFFER_SIZE} samples)")
print(f"  - FFT window: {FFT_WINDOW_SECONDS}s ({FFT_BUFFER_SIZE} samples)")
print(f"  - Band power update rate: {args.update_rate} Hz (every {BAND_POWER_UPDATE_INTERVAL*1000:.1f}ms)")
print(f"  - Target FPS: {TARGET_FPS}")
if args.relative_power:
    print(f"  - Y-axis: 0 to {Y_MAX}%")
else:
    print(f"  - Y-axis: 0 to {Y_MAX} {'dB' if args.log_scale else 'µV²'}")
print(f"  - Channels: {[CHANNEL_NAMES[i] for i in CHANNELS_TO_ANALYZE]}")
print(f"  - Bands: {BANDS_TO_PLOT}")
for band in BANDS_TO_PLOT:
    print(f"    • {band}: {BANDS[band][0]}-{BANDS[band][1]} Hz")
print(f"  - Update interval: {UPDATE_INTERVAL_MS}ms")
print(f"  - Minimal mode: {args.minimal}")
print(f"  - Show FPS: {args.show_fps}")
print(f"  - Average channels: {args.average_channels}")
print(f"  - Relative power: {args.relative_power}")
print(f"  - Log scale: {args.log_scale}")
print(f"  - Auto scale: {args.auto_scale}")
if args.smooth > 0:
    print(f"  - Smoothing: {args.smooth} samples")
if args.crop > 0:
    print(f"  - Edge crop: {args.crop} samples")
if custom_colors:
    print(f"  - Custom colors: {args.colors}")
print(f"  - Line width: {args.linewidth}")
print("=" * 70)

# Find EEG stream
print("\nSearching for Muse EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=10)
if not streams:
    print("ERROR: No EEG stream found!")
    sys.exit(1)

inlet = StreamInlet(streams[0], max_buflen=1)
print(f"✓ Connected to: {streams[0].name()}")
print(f"✓ Sampling rate: {SAMPLE_RATE} Hz")

# Initialize Qt Application
app = QtWidgets.QApplication(sys.argv)

# Create window
win = pg.GraphicsLayoutWidget(show=True)
win.resize(1400, 800)
win.setWindowTitle(f'Muse EEG Frequency Bands - {SAMPLE_RATE}Hz')

# In minimal mode, remove all layout margins
if args.minimal:
    win.ci.layout.setContentsMargins(0, 0, 0, 0)
    win.ci.layout.setSpacing(0)

# Set dark theme
if args.dark:
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')

# Initialize buffers
n_bands = len(BANDS_TO_PLOT)

# Band power buffers - store computed power values over time
band_power_buffers = [deque(maxlen=DISPLAY_BUFFER_SIZE) for _ in range(n_bands)]
for buf in band_power_buffers:
    buf.extend([0] * DISPLAY_BUFFER_SIZE)

# Raw EEG buffers for FFT computation - one per channel
if args.average_channels:
    raw_eeg_buffers = [deque(maxlen=FFT_BUFFER_SIZE)]  # Single averaged buffer
    raw_eeg_buffers[0].extend([0] * FFT_BUFFER_SIZE)
else:
    raw_eeg_buffers = [deque(maxlen=FFT_BUFFER_SIZE) for _ in CHANNELS_TO_ANALYZE]
    for buf in raw_eeg_buffers:
        buf.extend([0] * FFT_BUFFER_SIZE)

time_axis = np.linspace(-WINDOW_SECONDS, 0, DISPLAY_BUFFER_SIZE)

# Cropped time axis if crop is enabled
if args.crop > 0:
    time_axis_display = time_axis[args.crop:-args.crop] if args.crop < DISPLAY_BUFFER_SIZE // 2 else time_axis
else:
    time_axis_display = time_axis

# Create plots - one per band
plots = []
curves = []
fps_text = None

for idx, band_name in enumerate(BANDS_TO_PLOT):
    p = win.addPlot(row=idx, col=0)

    if not args.minimal:
        ylabel = f'{band_name} ({BANDS[band_name][0]}-{BANDS[band_name][1]} Hz)'
        if args.relative_power:
            yunit = '%'
        elif args.log_scale:
            yunit = 'dB'
        else:
            yunit = 'µV²'
        p.setLabel('left', ylabel, units=yunit)
        p.showGrid(x=True, y=True, alpha=0.3)
        if idx == len(BANDS_TO_PLOT) - 1:
            p.setLabel('bottom', 'Time', units='s')
        else:
            p.hideAxis('bottom')
    else:
        # Minimal mode - hide everything
        p.hideAxis('left')
        p.hideAxis('bottom')
        p.showGrid(x=False, y=False)

    if not args.auto_scale:
        p.setYRange(0, Y_MAX)

    # In minimal mode, extend waveform to edges
    if args.minimal:
        p.setXRange(time_axis_display[0], time_axis_display[-1], padding=0)
        if not args.auto_scale:
            p.setYRange(0, Y_MAX, padding=0)
        p.getViewBox().setDefaultPadding(0)
    else:
        p.setXRange(-WINDOW_SECONDS, 0)

    if not args.auto_scale:
        p.disableAutoRange()

    # Use custom color if provided, otherwise use default band color
    color = custom_colors[idx] if custom_colors else DEFAULT_BAND_COLORS[band_name]
    curve = p.plot(pen=pg.mkPen(color=color, width=args.linewidth))

    # Add band name label in minimal mode
    if args.minimal:
        label_text = f"{band_name} - {BAND_DESCRIPTIONS[band_name]}"
        band_label = pg.TextItem(text=label_text, anchor=(0, 0), color=color)
        band_label.setFont(pg.QtGui.QFont('Arial', 24, pg.QtGui.QFont.Bold))
        p.addItem(band_label)
        # Position at top-left (closer to edge and lower down)
        band_label.setPos(time_axis_display[0] + WINDOW_SECONDS * 0.005, Y_MAX * 0.80)

    plots.append(p)
    curves.append(curve)

# Add FPS counter if requested
if args.show_fps:
    fps_text = pg.TextItem(anchor=(1, 0), color='lime')
    fps_text.setFont(pg.QtGui.QFont('Arial', 16, pg.QtGui.QFont.Bold))
    plots[0].addItem(fps_text)
    # Position in top-right corner
    fps_text.setPos(-WINDOW_SECONDS * 0.02, Y_MAX * 0.85)

# Statistics
samples_received = 0
frames_rendered = 0
start_time = time.time()
last_stats_time = start_time
last_band_power_update = start_time
last_computed_band_powers = None  # Store last computed values

def compute_band_power(eeg_data, fs, band):
    """
    Compute power in a frequency band using Welch's method

    Parameters:
    -----------
    eeg_data : array-like
        Raw EEG data
    fs : int
        Sampling frequency
    band : tuple
        (low_freq, high_freq) in Hz

    Returns:
    --------
    power : float
        Band power (µV² or dB if log_scale enabled)
    """
    # Use Welch's method for power spectral density
    # nperseg should be at least 2x the lowest frequency period
    nperseg = min(len(eeg_data), int(fs * 2))

    freqs, psd = welch(eeg_data, fs=fs, nperseg=nperseg, scaling='density')

    # Find frequencies in the band
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])

    # Integrate power in band (trapezoidal rule)
    band_power = np.trapezoid(psd[idx_band], freqs[idx_band])

    # Convert to dB if requested (but not if using relative power)
    if args.log_scale and not args.relative_power:
        # 10*log10(power), with small epsilon to avoid log(0)
        band_power = 10 * np.log10(band_power + 1e-12)

    return band_power

def update():
    global samples_received, frames_rendered, last_stats_time, last_band_power_update, last_computed_band_powers

    # Pull all available samples
    chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=512)

    if chunk:
        n_new = len(chunk)
        samples_received += n_new

        # Update raw EEG buffers
        chunk_array = np.array(chunk)

        if args.average_channels:
            # Average across selected channels, then add to buffer
            for sample in chunk_array:
                avg_value = np.mean([sample[ch_idx] for ch_idx in CHANNELS_TO_ANALYZE])
                raw_eeg_buffers[0].append(avg_value)
        else:
            # Update each channel buffer separately
            for idx, ch_idx in enumerate(CHANNELS_TO_ANALYZE):
                for sample in chunk_array:
                    raw_eeg_buffers[idx].append(sample[ch_idx])

    # Compute band powers from raw EEG data
    # Only recompute at specified interval to avoid over-smoothing from overlapping windows
    current_time = time.time()
    if (current_time - last_band_power_update >= BAND_POWER_UPDATE_INTERVAL and
        len(raw_eeg_buffers[0]) >= FFT_BUFFER_SIZE // 2):  # Need enough data

        band_powers = []

        for band_name in BANDS_TO_PLOT:
            band_range = BANDS[band_name]

            if args.average_channels:
                # Compute from single averaged buffer
                eeg_data = np.array(list(raw_eeg_buffers[0]))
                power = compute_band_power(eeg_data, SAMPLE_RATE, band_range)
                band_powers.append(power)
            else:
                # Compute for each channel and average the powers
                channel_powers = []
                for buf in raw_eeg_buffers:
                    eeg_data = np.array(list(buf))
                    power = compute_band_power(eeg_data, SAMPLE_RATE, band_range)
                    channel_powers.append(power)
                # Average powers across channels
                avg_power = np.mean(channel_powers)
                band_powers.append(avg_power)

        # Convert to relative power (percentage) if requested
        if args.relative_power:
            total_power = sum(band_powers)
            if total_power > 0:
                band_powers = [(power / total_power) * 100 for power in band_powers]
            else:
                band_powers = [0] * len(band_powers)

        # Store the computed values
        last_computed_band_powers = band_powers

        # Add computed powers to band power buffers
        for idx, power in enumerate(band_powers):
            band_power_buffers[idx].append(power)

        last_band_power_update = current_time
    elif last_computed_band_powers is not None:
        # Reuse last computed values to keep traces smooth
        for idx, power in enumerate(last_computed_band_powers):
            band_power_buffers[idx].append(power)

    # Update curves
    for idx, curve in enumerate(curves):
        # Get band power data
        data = np.array(list(band_power_buffers[idx]))

        # Apply smoothing if enabled
        if args.smooth > 0:
            data = uniform_filter1d(data, size=args.smooth, mode='reflect')

        # Crop edges if enabled
        if args.crop > 0:
            data = data[args.crop:-args.crop]

        curve.setData(time_axis_display, data)

    frames_rendered += 1

    # Update FPS display
    if args.show_fps:
        elapsed = time.time() - start_time
        current_fps = frames_rendered / elapsed if elapsed > 0 else 0
        fps_text.setText(f'{current_fps:.1f} FPS')

    # Print stats every 2 seconds
    current_time = time.time()
    if current_time - last_stats_time >= 2.0:
        elapsed = current_time - start_time
        actual_fps = frames_rendered / elapsed
        sample_rate = samples_received / elapsed
        print(f"[{elapsed:.1f}s] Samples/s: {sample_rate:.1f} | "
              f"Display FPS: {actual_fps:.1f} | "
              f"Frames: {frames_rendered}")
        last_stats_time = current_time

# Setup timer for updates
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_INTERVAL_MS)

print("\n" + "=" * 70)
print("STREAMING STARTED - Close window to stop")
print("=" * 70 + "\n")

# Start Qt event loop
try:
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
except KeyboardInterrupt:
    print("\n\nStopped by user")
finally:
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("STREAMING STATISTICS")
    print("=" * 70)
    print(f"Duration: {elapsed:.2f} seconds")
    print(f"Samples received: {samples_received}")
    print(f"Average sample rate: {samples_received/elapsed:.1f} Hz")
    print(f"Frames rendered: {frames_rendered}")
    print(f"Average FPS: {frames_rendered/elapsed:.1f}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"FPS achievement: {(frames_rendered/elapsed)/TARGET_FPS*100:.1f}%")
    print("=" * 70)
