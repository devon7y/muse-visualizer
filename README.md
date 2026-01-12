# Muse EEG Visualizer

Real-time EEG visualization tools for the Muse headband, featuring ultra-high-performance plotting of raw waveforms and frequency band analysis.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

Two powerful visualization scripts for analyzing EEG data from Muse headbands in real-time:

- **`live_raw_eeg_plot.py`** - Ultra-fast raw EEG waveform visualization (200+ FPS capable)
- **`live_oscillations_plot.py`** - Real-time frequency band analysis (Delta, Theta, Alpha, Beta, Gamma)

Both scripts use PyQtGraph for GPU-accelerated rendering and Lab Streaming Layer (LSL) for low-latency data acquisition.

## Features

### Raw EEG Plotter (`live_raw_eeg_plot.py`)

- **Ultra-high performance**: Achieves 200+ FPS with PyQtGraph backend
- **Real-time streaming**: 256 Hz sampling from Muse headband
- **Channel selection**: Plot individual channels (TP9, AF7, AF8, TP10) or average them
- **Signal processing**:
  - Adjustable smoothing with edge artifact handling
  - DC offset removal with calibration period
  - Edge cropping for clean visualization
- **Customization**:
  - Custom colors per channel (hex codes)
  - Adjustable line width
  - Configurable Y-axis scale
  - Variable time window
- **Minimal mode**: Clean, axis-free display for recording/streaming
- **Heart rate monitoring**: Calculate and display HR from PPG sensor (Muse 2/S)
- **Performance monitoring**: Live FPS counter option

### Frequency Band Plotter (`live_oscillations_plot.py`)

- **Frequency band analysis**: Real-time computation of Delta, Theta, Alpha, Beta, Gamma power
- **Multiple display modes**:
  - Absolute power (µV²)
  - Logarithmic scale (dB)
  - Relative power (percentage of total)
- **Configurable FFT**: Adjustable window length for spectral analysis
- **Update rate control**: Balance between responsiveness and smoothness
- **Band selection**: Choose which frequency bands to display
- **Labeled minimal mode**: Color-coded band labels with cognitive state descriptions
- **Signal processing**: Smoothing and edge cropping options
- **Custom color palettes**: Define colors for each frequency band

## Installation

### Requirements

- Python 3.8+
- Muse headband (2016, 2, or S)
- Windows: BlueMuse or Linux/Mac: muse-lsl

### Install Dependencies

```bash
pip install pyqtgraph PyQt5 pylsl scipy numpy
```

### Setup Muse Streaming

**Windows (BlueMuse):**
1. Download and install [BlueMuse](https://github.com/kowalej/BlueMuse/releases)
2. Connect your Muse headband
3. Start streaming in BlueMuse

**Mac/Linux (muse-lsl):**
```bash
pip install muselsl
muselsl stream
```

## Usage

### Raw EEG Visualization

**Basic usage:**
```bash
python live_raw_eeg_plot.py
```

**Advanced examples:**

Average frontal channels with smoothing:
```bash
python live_raw_eeg_plot.py -c AF7 AF8 --average --smooth 100 --center --colors 47BBFF
```

Minimal mode for clean display:
```bash
python live_raw_eeg_plot.py --minimal -f 120 --smooth 50 -s 15 -w 8
```

With heart rate monitoring:
```bash
python live_raw_eeg_plot.py --heart-rate --show-fps
```

Full-featured setup:
```bash
python live_raw_eeg_plot.py -c AF7 AF8 --average --center --smooth 100 --crop 40 --colors 47BBFF --minimal -f 60 -s 15 -w 8 --linewidth 5
```

### Frequency Band Analysis

**Basic usage:**
```bash
python live_oscillations_plot.py
```

**Advanced examples:**

Relative power (percentage) display:
```bash
python live_oscillations_plot.py --relative-power --minimal
```

High-speed updates with custom colors:
```bash
python live_oscillations_plot.py -c AF7 AF8 --average-channels --minimal -f 60 --smooth 5 -w 15 --linewidth 5 --relative-power --fft-window 0.5 --update-rate 250 --colors 225A8C 317CA3 3F9EB9 4EBFD0 5CE1E6
```

Specific bands only:
```bash
python live_oscillations_plot.py -b Alpha Beta --relative-power
```

Logarithmic scale:
```bash
python live_oscillations_plot.py --log-scale --show-fps
```

## Parameters

### Raw EEG Plotter

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-w, --window` | Time window in seconds | 5.0 |
| `-f, --fps` | Target frame rate | 60 |
| `-s, --scale` | Y-axis scale in µV | 200.0 |
| `-c, --channels` | Channels to plot (TP9/AF7/AF8/TP10) | All |
| `--average` | Average selected channels | False |
| `--smooth` | Smoothing window size | 0 |
| `--center` | Auto-center waveform (DC offset removal) | False |
| `--center-duration` | Calibration period in seconds | 10.0 |
| `--crop` | Crop samples from edges | 0 |
| `--colors` | Hex color codes | Default |
| `--linewidth` | Line thickness | 1.5 (avg), 1 (individual) |
| `--minimal` | Minimal UI mode | False |
| `--show-fps` | Display FPS counter | False |
| `--heart-rate` | Show heart rate from PPG | False |

### Frequency Band Plotter

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-w, --window` | Display window in seconds | 10.0 |
| `--fft-window` | FFT computation window in seconds | 2.0 |
| `--update-rate` | Band power update rate in Hz | 10.0 |
| `-f, --fps` | Display frame rate | 60 |
| `-c, --channels` | Channels to analyze | All |
| `-b, --bands` | Bands to plot (Delta/Theta/Alpha/Beta/Gamma) | All |
| `--average-channels` | Average channels before FFT | False |
| `--relative-power` | Show as percentage (0-100%) | False |
| `--log-scale` | Use logarithmic scale (dB) | False |
| `--auto-scale` | Auto-adjust Y-axis | False |
| `--y-max` | Maximum Y-axis value | Auto |
| `--smooth` | Smoothing window size | 0 |
| `--crop` | Crop samples from edges | 0 |
| `--colors` | Hex color codes for bands | Default |
| `--linewidth` | Line thickness | 2.0 |
| `--minimal` | Minimal UI with band labels | False |
| `--show-fps` | Display FPS counter | False |

## Frequency Bands

| Band | Frequency Range | Mental State | Color (Default) |
|------|----------------|--------------|-----------------|
| **Delta** | 0.5-4 Hz | Deep sleep | Purple (128, 0, 128) |
| **Theta** | 4-8 Hz | Memory, drowsiness | Blue (0, 0, 255) |
| **Alpha** | 8-13 Hz | Relaxed, calm | Green (0, 255, 0) |
| **Beta** | 13-30 Hz | Focus, alertness | Yellow (255, 255, 0) |
| **Gamma** | 30-50 Hz | Attention, processing | Red (255, 0, 0) |

## Technical Details

### Raw EEG Processing
- **Sampling Rate**: 256 Hz from Muse headband
- **Filtering**: Optional DC offset removal with calibration
- **Smoothing**: Uniform filter with reflect mode for edge handling
- **Display**: PyQtGraph with GPU acceleration

### Frequency Band Analysis
- **Method**: Welch's periodogram for power spectral density
- **FFT Window**: Configurable (default 2 seconds)
- **Frequency Resolution**: Determined by FFT window length
- **Update Mechanism**: Rate-limited to prevent over-smoothing
- **Normalization**: Optional relative power (percentage of total)

### Heart Rate Calculation (Raw Plotter Only)
- **Sensor**: PPG (photoplethysmography) on Muse 2/S
- **Sampling**: 64 Hz from PPG sensor
- **Processing**:
  - Bandpass filter: 0.5-4 Hz (30-240 BPM)
  - Peak detection with prominence threshold
  - Inter-beat interval calculation
  - Median-based averaging for robustness
  - Valid range: 30-200 BPM

## Performance Tips

1. **For smoothest display**: Use `--minimal` mode and target 60 FPS
2. **For responsiveness**: Reduce smoothing and FFT window size
3. **For clean traces**: Increase smoothing, enable DC centering, use crop
4. **For frequency bands**: Use `--relative-power` for equal visibility
5. **For artifacts**: Enable `--center` with sufficient calibration duration

## Troubleshooting

**No stream found:**
- Ensure BlueMuse (Windows) or muse-lsl (Mac/Linux) is running
- Check that Muse headband is connected and streaming
- Verify LSL is properly installed: `python -c "from pylsl import resolve_streams; print(resolve_streams())"`

**Low FPS:**
- Reduce window length or smoothing amount
- Lower target FPS if hardware limited
- Close other applications using GPU

**Steppy frequency bands:**
- Increase `--update-rate` (default 10 Hz, try 30-60 Hz)
- Reduce `--fft-window` for faster updates
- Decrease `--smooth` parameter

**PPG/Heart rate not working:**
- Muse 2 or Muse S required (original Muse doesn't have PPG)
- Enable PPG streaming in BlueMuse: `start bluemuse://setting?key=ppg_enabled!value=true`
- Ensure good headband fit for optical sensor contact

## Examples

### Meditation Monitoring
```bash
python live_oscillations_plot.py -b Alpha Theta --relative-power --minimal -w 30 --smooth 20
```
Track relaxation (Alpha) and deep meditation (Theta) states.

### Focus Training
```bash
python live_oscillations_plot.py -b Beta Gamma --relative-power --minimal --update-rate 30
```
Monitor attention (Gamma) and concentration (Beta) during work.

### Sleep Analysis
```bash
python live_oscillations_plot.py -b Delta Theta Alpha --relative-power -w 60 --fft-window 4
```
Observe transition from wakefulness (Alpha) to sleep (Delta/Theta).

### Clean Recording Setup
```bash
python live_raw_eeg_plot.py -c AF7 AF8 --average --smooth 100 --center --crop 40 --minimal --colors 47BBFF --linewidth 5 -w 8 -s 15 -f 60
```
Production-ready minimal display for video recording or streaming.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built for the [Muse headband](https://choosemuse.com/) by InteraXon
- Uses [BlueMuse](https://github.com/kowalej/BlueMuse) for Windows LSL streaming
- Uses [muse-lsl](https://github.com/alexandrebarachant/muse-lsl) for Mac/Linux
- Powered by [PyQtGraph](http://www.pyqtgraph.org/) for high-performance visualization
- Lab Streaming Layer ([LSL](https://github.com/sccn/labstreaminglayer)) for real-time data

## Citation

If you use this software in your research, please cite:

```
@software{muse_eeg_visualizer,
  title = {Muse EEG Visualizer},
  author = {Devon},
  year = {2026},
  url = {https://github.com/devon7y/muse-eeg-visualizer}
}
```

## Support

For questions or issues:
- Open an issue on GitHub
- Check the [BlueMuse documentation](https://github.com/kowalej/BlueMuse)
- Review the [muse-lsl guide](https://github.com/alexandrebarachant/muse-lsl)
