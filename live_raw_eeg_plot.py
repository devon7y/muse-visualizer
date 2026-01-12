"""
Ultra-high-performance live EEG plotter using PyQtGraph
Can easily achieve 200+ FPS for real-time visualization
Install: pip install pyqtgraph PyQt5
"""
import sys
import argparse
import time
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from collections import deque
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, butter, filtfilt

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    print("ERROR: PyQtGraph not installed!")
    print("Install with: pip install pyqtgraph PyQt5")
    sys.exit(1)

# Parse arguments
parser = argparse.ArgumentParser(description='Ultra-fast Muse EEG plotter (200+ FPS capable)')
parser.add_argument('-w', '--window', type=float, default=5.0,
                    help='Window length in seconds (default: 5)')
parser.add_argument('-f', '--fps', type=int, default=60,
                    help='Target FPS (default: 60, can go 200+)')
parser.add_argument('-s', '--scale', type=float, default=200.0,
                    help='Y-axis scale in microvolts (default: 200)')
parser.add_argument('-c', '--channels', type=str, nargs='+', default=['TP9', 'AF7', 'AF8', 'TP10'],
                    help='Channels to plot: TP9, AF7, AF8, TP10 (default: all)')
parser.add_argument('--minimal', action='store_true',
                    help='Minimal UI - waveforms only')
parser.add_argument('--dark', action='store_true', default=True,
                    help='Dark theme (default)')
parser.add_argument('--show-fps', action='store_true',
                    help='Display live FPS counter on screen')
parser.add_argument('--average', action='store_true',
                    help='Average the selected channels into a single waveform')
parser.add_argument('--smooth', type=int, default=0,
                    help='Smoothing window size in samples (0=no smoothing, default: 0)')
parser.add_argument('--colors', type=str, nargs='+', default=None,
                    help='Hex color codes for waveforms (e.g., 47BBFF or #47BBFF). Specify one per channel, or one if averaging.')
parser.add_argument('--center', action='store_true',
                    help='Automatically center waveform by removing DC offset (calculated from first few seconds)')
parser.add_argument('--center-duration', type=float, default=10.0,
                    help='Duration in seconds to calculate DC offset for centering (default: 10)')
parser.add_argument('--crop', type=int, default=0,
                    help='Number of samples to crop from each edge to hide smoothing artifacts (default: 0)')
parser.add_argument('--linewidth', type=float, default=None,
                    help='Line width for waveforms (default: 1.5 for average, 1 for individual channels)')
parser.add_argument('--heart-rate', action='store_true',
                    help='Display heart rate calculated from PPG sensor')
args = parser.parse_args()

# Channel name to index mapping
CHANNEL_NAMES = ['TP9', 'AF7', 'AF8', 'TP10']
CHANNEL_MAP = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}

# Convert channel names to indices
CHANNELS_TO_PLOT = []
for ch_name in args.channels:
    ch_upper = ch_name.upper()
    if ch_upper not in CHANNEL_MAP:
        print(f"ERROR: Invalid channel name '{ch_name}'")
        print(f"Valid channels: {', '.join(CHANNEL_NAMES)}")
        sys.exit(1)
    CHANNELS_TO_PLOT.append(CHANNEL_MAP[ch_upper])

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
if args.colors:
    expected_num_colors = 1 if args.average else len(CHANNELS_TO_PLOT)
    if len(args.colors) != expected_num_colors:
        print(f"ERROR: Expected {expected_num_colors} color(s) but got {len(args.colors)}")
        if args.average:
            print("When using --average, specify only 1 color")
        else:
            print(f"When plotting {len(CHANNELS_TO_PLOT)} channel(s), specify {len(CHANNELS_TO_PLOT)} color(s)")
        sys.exit(1)

    try:
        custom_colors = [parse_hex_color_to_rgb(c) for c in args.colors]
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Example: --colors 47BBFF or --colors #47BBFF FF5733")
        sys.exit(1)

# Configuration
WINDOW_SECONDS = args.window
TARGET_FPS = args.fps
Y_SCALE = args.scale
SAMPLE_RATE = 256
DEFAULT_COLORS = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]  # RGB

BUFFER_SIZE = int(WINDOW_SECONDS * SAMPLE_RATE)
UPDATE_INTERVAL_MS = int(1000.0 / TARGET_FPS)

print("=" * 70)
print("MUSE EEG ULTRA-FAST PLOTTER - PyQtGraph Backend")
print("=" * 70)
print(f"Configuration:")
print(f"  - Window: {WINDOW_SECONDS}s ({BUFFER_SIZE} samples)")
print(f"  - Target FPS: {TARGET_FPS}")
print(f"  - Y-scale: ±{Y_SCALE} µV")
print(f"  - Channels: {[CHANNEL_NAMES[i] for i in CHANNELS_TO_PLOT]}")
print(f"  - Update interval: {UPDATE_INTERVAL_MS}ms")
print(f"  - Minimal mode: {args.minimal}")
print(f"  - Show FPS: {args.show_fps}")
print(f"  - Average channels: {args.average}")
if args.average:
    print(f"  - Averaging: {[CHANNEL_NAMES[i] for i in CHANNELS_TO_PLOT]}")
print(f"  - Smoothing: {args.smooth} samples" if args.smooth > 0 else "  - Smoothing: disabled")
if args.center:
    print(f"  - Auto-center: enabled (calibrating from first {args.center_duration}s)")
else:
    print(f"  - Auto-center: disabled")
if args.crop > 0:
    print(f"  - Edge crop: {args.crop} samples from each edge")
else:
    print(f"  - Edge crop: disabled")
if custom_colors:
    print(f"  - Custom colors: {args.colors}")
if args.linewidth:
    print(f"  - Line width: {args.linewidth}")
print(f"  - Heart rate display: {args.heart_rate}")
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

# Connect to PPG stream if heart rate monitoring is enabled
ppg_inlet = None
if args.heart_rate:
    print("\nSearching for Muse PPG stream...")
    ppg_streams = resolve_byprop('type', 'PPG', timeout=5)
    if ppg_streams:
        ppg_inlet = StreamInlet(ppg_streams[0], max_buflen=1)
        ppg_info = ppg_inlet.info()
        ppg_srate = int(ppg_info.nominal_srate())
        print(f"✓ Connected to PPG stream")
        print(f"✓ PPG sampling rate: {ppg_srate} Hz")
    else:
        print("⚠ Warning: No PPG stream found. Heart rate monitoring disabled.")
        args.heart_rate = False

# Initialize Qt Application
app = QtWidgets.QApplication(sys.argv)

# Create window
win = pg.GraphicsLayoutWidget(show=True, title="Muse EEG - Ultra Fast Mode")
win.resize(1400, 800)
win.setWindowTitle(f'Muse EEG Live - {SAMPLE_RATE}Hz')

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
n_channels = len(CHANNELS_TO_PLOT)
if args.average:
    # When averaging, we only need one buffer
    n_plots = 1
    buffers = [deque(maxlen=BUFFER_SIZE)]
    buffers[0].extend([0] * BUFFER_SIZE)
else:
    # Normal mode: one buffer per channel
    n_plots = n_channels
    buffers = [deque(maxlen=BUFFER_SIZE) for _ in range(n_channels)]
    for buf in buffers:
        buf.extend([0] * BUFFER_SIZE)

time_axis = np.linspace(-WINDOW_SECONDS, 0, BUFFER_SIZE)

# Cropped time axis if crop is enabled
if args.crop > 0:
    time_axis_display = time_axis[args.crop:-args.crop] if args.crop < BUFFER_SIZE // 2 else time_axis
else:
    time_axis_display = time_axis

# Create plots
plots = []
curves = []
fps_text = None

if args.average:
    # Averaged mode: single plot
    p = win.addPlot(row=0, col=0)

    if not args.minimal:
        avg_label = 'AVG[' + ','.join([CHANNEL_NAMES[i] for i in CHANNELS_TO_PLOT]) + ']'
        p.setLabel('left', avg_label, units='µV')
        p.setLabel('bottom', 'Time', units='s')
        p.showGrid(x=True, y=True, alpha=0.3)
    else:
        p.hideAxis('left')
        p.hideAxis('bottom')
        p.showGrid(x=False, y=False)

    p.setYRange(-Y_SCALE, Y_SCALE)

    # In minimal mode, extend waveform to edges
    if args.minimal:
        p.setXRange(time_axis_display[0], time_axis_display[-1], padding=0)
        p.setYRange(-Y_SCALE, Y_SCALE, padding=0)
        p.getViewBox().setDefaultPadding(0)  # Remove all padding
    else:
        p.setXRange(-WINDOW_SECONDS, 0)
    p.disableAutoRange()

    color = custom_colors[0] if custom_colors else (0, 255, 0)
    linewidth = args.linewidth if args.linewidth is not None else 1.5
    curve = p.plot(pen=pg.mkPen(color=color, width=linewidth))

    plots.append(p)
    curves.append(curve)
else:
    # Normal mode: one plot per channel
    for idx, ch_idx in enumerate(CHANNELS_TO_PLOT):
        p = win.addPlot(row=idx, col=0)

        if not args.minimal:
            p.setLabel('left', CHANNEL_NAMES[ch_idx], units='µV')
            p.showGrid(x=True, y=True, alpha=0.3)
            if idx == len(CHANNELS_TO_PLOT) - 1:
                p.setLabel('bottom', 'Time', units='s')
            else:
                p.hideAxis('bottom')
        else:
            # Minimal mode - hide everything
            p.hideAxis('left')
            p.hideAxis('bottom')
            p.showGrid(x=False, y=False)

        p.setYRange(-Y_SCALE, Y_SCALE)

        # In minimal mode, extend waveform to edges
        if args.minimal:
            p.setXRange(time_axis_display[0], time_axis_display[-1], padding=0)
            p.setYRange(-Y_SCALE, Y_SCALE, padding=0)
            p.getViewBox().setDefaultPadding(0)  # Remove all padding
        else:
            p.setXRange(-WINDOW_SECONDS, 0)
        p.disableAutoRange()

        # Use custom color if provided, otherwise use default
        color = custom_colors[idx] if custom_colors else DEFAULT_COLORS[ch_idx % len(DEFAULT_COLORS)]
        linewidth = args.linewidth if args.linewidth is not None else 1
        curve = p.plot(pen=pg.mkPen(color=color, width=linewidth))

        plots.append(p)
        curves.append(curve)

# Add FPS counter if requested
if args.show_fps:
    fps_text = pg.TextItem(anchor=(1, 0), color='lime')
    fps_text.setFont(pg.QtGui.QFont('Arial', 16, pg.QtGui.QFont.Bold))
    plots[0].addItem(fps_text)
    # Position in top-right corner
    fps_text.setPos(-WINDOW_SECONDS * 0.02, Y_SCALE * 0.85)

# Add heart rate display if requested
hr_text = None
if args.heart_rate and ppg_inlet:
    hr_text = pg.TextItem(anchor=(0, 0), color='#FF6B6B')  # Red/pink color for HR
    hr_text.setFont(pg.QtGui.QFont('Arial', 20, pg.QtGui.QFont.Bold))
    plots[0].addItem(hr_text)
    # Position in top-left corner
    if args.minimal:
        hr_text.setPos(time_axis_display[0] + WINDOW_SECONDS * 0.02, Y_SCALE * 0.85)
    else:
        hr_text.setPos(-WINDOW_SECONDS * 0.98, Y_SCALE * 0.85)
    hr_text.setText('HR: -- BPM')

# Statistics
samples_received = 0
frames_rendered = 0
start_time = time.time()
last_stats_time = start_time

# DC offset centering variables
dc_offsets = None  # Will store fixed DC offset for each channel
calibration_complete = False
calibration_data = []  # Collect data during calibration period

# Heart rate calculation variables
if args.heart_rate and ppg_inlet:
    PPG_BUFFER_SECONDS = 10  # Keep 10 seconds of PPG data for HR calculation
    ppg_buffer_size = PPG_BUFFER_SECONDS * ppg_srate
    ppg_buffer = deque(maxlen=ppg_buffer_size)
    ppg_buffer.extend([0] * ppg_buffer_size)

    # Bandpass filter for heart rate (0.5-4 Hz = 30-240 BPM)
    nyquist = ppg_srate / 2
    low_cutoff = 0.5 / nyquist  # 30 BPM
    high_cutoff = 4.0 / nyquist  # 240 BPM
    ppg_b, ppg_a = butter(2, [low_cutoff, high_cutoff], btype='band')

    current_hr = 0
    last_hr_update = time.time()
    hr_update_interval = 2.0  # Update HR every 2 seconds
else:
    ppg_buffer = None
    current_hr = 0

def update():
    global samples_received, frames_rendered, last_stats_time, dc_offsets, calibration_complete, calibration_data
    global current_hr, last_hr_update

    # Pull all available samples
    chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=512)

    if chunk:
        n_new = len(chunk)
        samples_received += n_new

        # Update buffers
        chunk_array = np.array(chunk)

        # Extract the channels we want
        chunk_channels = chunk_array[:, CHANNELS_TO_PLOT]

        # Collect calibration data if centering is enabled and not yet complete
        if args.center and not calibration_complete:
            if args.average:
                # For averaged mode, store the averaged values
                avg_values = np.mean(chunk_channels, axis=1, keepdims=True)
                calibration_data.append(avg_values)
            else:
                calibration_data.append(chunk_channels)

            elapsed_cal = time.time() - start_time
            if elapsed_cal >= args.center_duration:
                # Calibration period complete - calculate fixed DC offsets
                all_cal_data = np.vstack(calibration_data)
                dc_offsets = np.mean(all_cal_data, axis=0)
                calibration_complete = True
                calibration_data = []  # Free memory
                print(f"✓ DC offset calibration complete! Offsets: {dc_offsets.flatten()}")

        if args.average:
            # Average across selected channels
            for sample in chunk_array:
                avg_value = np.mean([sample[ch_idx] for ch_idx in CHANNELS_TO_PLOT])
                buffers[0].append(avg_value)
        else:
            # Normal mode: update each channel separately
            for idx, ch_idx in enumerate(CHANNELS_TO_PLOT):
                for sample in chunk_array:
                    buffers[idx].append(sample[ch_idx])

    # Collect PPG data and calculate heart rate
    if args.heart_rate and ppg_inlet:
        ppg_chunk, ppg_timestamps = ppg_inlet.pull_chunk(timeout=0.0, max_samples=512)
        if ppg_chunk:
            # PPG typically has 3 channels (ambient, infrared, red) - use infrared (index 1)
            ppg_array = np.array(ppg_chunk)
            if ppg_array.shape[1] >= 2:
                # Use infrared channel (usually channel 1)
                for sample in ppg_array:
                    ppg_buffer.append(sample[1])

        # Calculate heart rate every hr_update_interval seconds
        current_time = time.time()
        if current_time - last_hr_update >= hr_update_interval:
            if len(ppg_buffer) > ppg_srate * 3:  # Need at least 3 seconds of data
                # Convert buffer to numpy array
                ppg_data = np.array(list(ppg_buffer))

                # Apply bandpass filter (0.5-4 Hz for 30-240 BPM)
                try:
                    filtered_ppg = filtfilt(ppg_b, ppg_a, ppg_data)

                    # Find peaks in the filtered signal
                    # Minimum distance between peaks: 60/max_HR in samples
                    # For max 180 BPM: 60/180 = 0.333s between peaks
                    min_peak_distance = int(ppg_srate * 0.4)  # ~150 BPM max

                    # Prominence helps avoid false peaks from noise
                    peaks, properties = find_peaks(
                        filtered_ppg,
                        distance=min_peak_distance,
                        prominence=np.std(filtered_ppg) * 0.5
                    )

                    # Calculate HR from peak intervals
                    if len(peaks) >= 3:  # Need at least 3 peaks
                        # Calculate inter-beat intervals in seconds
                        peak_intervals = np.diff(peaks) / ppg_srate

                        # Remove outliers (unrealistic intervals)
                        # Valid range: 0.33s (180 BPM) to 2s (30 BPM)
                        valid_intervals = peak_intervals[(peak_intervals > 0.33) & (peak_intervals < 2.0)]

                        if len(valid_intervals) > 0:
                            # Average interval in seconds
                            avg_interval = np.median(valid_intervals)  # Use median for robustness
                            # Convert to BPM
                            current_hr = int(60 / avg_interval)

                            # Final sanity check
                            if current_hr < 30 or current_hr > 200:
                                current_hr = 0  # Invalid, show as unavailable
                except Exception as e:
                    # If filtering fails, keep previous HR
                    pass

            last_hr_update = current_time

            # Update HR display
            if hr_text:
                if current_hr > 0:
                    hr_text.setText(f'♥ {current_hr} BPM')
                else:
                    hr_text.setText('♥ -- BPM')

    # Update curves
    for idx, curve in enumerate(curves):
        # Get data for this channel
        data = np.array(list(buffers[idx]))

        # Apply centering (DC offset removal) if enabled and calibrated
        if args.center and calibration_complete and dc_offsets is not None:
            data = data - dc_offsets[idx]

        # Apply smoothing if enabled
        if args.smooth > 0:
            # Smooth the data using 'reflect' mode for better edge handling
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
if args.center:
    print(f"Calibrating DC offset from first {args.center_duration} seconds...")
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
    print(f"Expected samples at 256Hz: {int(elapsed * 256)}")
    print(f"Sample capture rate: {(samples_received/(elapsed*256))*100:.1f}%")
    print("=" * 70)
