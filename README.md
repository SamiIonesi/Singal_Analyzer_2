# Singal_Analyzer_2

This repository describes some operations that are performed to an audio signal.

## 1. Task 1

> From the original file, only the signal between second 2 and second 3 is retained.

### 1.1 Python function

```Python
def trim_audio(self, output_file, start_second, end_second):
    """
    Extracts a portion of the audio between start_second and end_second.
    """
    with wave.open(self.input_file, 'rb') as wav:
        # Get audio parameters
        params = wav.getparams()
        framerate = params.framerate
        n_channels = params.nchannels
        sampwidth = params.sampwidth

        # Calculate start and end frames
        start_frame = int(start_second * framerate)
        end_frame = int(end_second * framerate)

        # Read and extract desired frames
        wav.setpos(start_frame)
        frames = wav.readframes(end_frame - start_frame)

        # Write trimmed audio to a new file
        with wave.open(output_file, 'wb') as trimmed_wav:
            trimmed_wav.setparams(params)
            trimmed_wav.writeframes(frames)

    print(f"Trimmed audio saved as '{output_file}'.")
    self.trimmed_file = output_file
```

### 1.2 Task description

> The trim_audio function extracts a specific portion of an audio file (between start_second and end_second) and saves the trimmed audio to a new file.

This function has the next steps for solving the task:

#### 1.2.1 Open the Input Audio File <br>
#### 1.2.2 Retrieve Audio Parameters <br>
#### 1.2.3 Calculate Start and End Frames <br>
#### 1.2.4 Extract the Desired Frames <br>
#### 1.2.5 Write the Trimmed Audio to a New File <br>
#### 1.2.6 Save the Output File Path <br>
#### 1.2.7 Print Confirmation <br>

## 2. Task 2

> The audio signal plays for one second.

### 2.1 Python function

```Python
def play_audio(self):
    """
    Plays the trimmed audio file asynchronously.
    """
    if not hasattr(self, 'trimmed_file'):
        raise ValueError("Trimmed audio not created. Run `trim_audio` first.")

    with wave.open(self.trimmed_file, 'rb') as wav:
        # Extract audio parameters and data
        params = wav.getparams()
        frames = wav.readframes(params.nframes)

    # Play audio asynchronously
    play_obj = sa.play_buffer(frames, num_channels=params.nchannels, bytes_per_sample=params.sampwidth, sample_rate=params.framerate)

```

### 2.2 Task description

> The play_audio function plays the trimmed audio file stored in self.trimmed_file.

This function has the next steps for solving the task:

#### 2.2.1. Checks if the trimmed file exists to avoid runtime errors.
#### 2.2.2. Opens the trimmed file and extracts its audio data and parameters using the wave module.
#### 2.2.3. Plays the audio data using the simpleaudio library, ensuring accurate playback with proper parameters.

## 3. Task 3

> For the signal considered for one second, plot the signal envelope and detect the peaks and corresponding time points.

### 3.1 Python function

```Python
def plot_envelope_and_detect_peaks(self, input_file):
    """
    Plots the signal envelope and detects peaks with corresponding time points.
    """

    if not hasattr(self, 'trimmed_file'):
        raise ValueError("Trimmed audio not created. Run `trim_audio` first.")

    # Read audio data
    with wave.open(input_file, 'rb') as wav:
        params = wav.getparams()
        framerate = params.framerate
        frames = wav.readframes(params.nframes)
        audio_data = np.frombuffer(frames, dtype=np.int16)

    if audio_data.size == 0:
        print("Error: Audio data is empty.")
        return

    print(f"Audio data loaded successfully. Length: {len(audio_data)} samples")
    print(f"Sample rate: {framerate} Hz")

    # Calculate time axis
    time_axis = np.linspace(0, len(audio_data) / framerate, num=len(audio_data))

    # Compute absolute value of the signal
    abs_signal = np.abs(audio_data)

    # Smooth the envelope with a moving average
    window_size = int(framerate * 0.01)  # 10ms window
    envelope = np.convolve(abs_signal, np.ones(window_size) / window_size, mode='same')

    # Detect peaks in the smoothed envelope
    peaks, _ = find_peaks(envelope, height=np.max(envelope) * 0.5)
    peak_times = time_axis[peaks]

    print(f"Number of peaks detected: {len(peaks)}")
    
    if len(peaks) > 0:
        print(f"Peak times (s): {peak_times}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, audio_data, label="Audio Signal", alpha=0.6)
    plt.plot(time_axis, envelope, label="Smoothed Envelope", color='red', linewidth=2)
    plt.scatter(peak_times, envelope[peaks], color='green', label="Peaks", zorder=3)
    plt.title("Signal Envelope and Peaks")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    print("Displaying plot...")
    plt.show()
```

### 3.2 Task description

> This function processes an audio file to plot its signal envelope and detect peaks, which are significant points of interest in the signal. 

This function has the next steps for solving the task:

#### 3.2.1 Input Validation
#### 3.2.2 Reading Audio Data
#### 3.2.3 Check for Empty Audio
#### 3.2.4 Time Axis Calculation
#### 3.2.5 Envelope Computation
#### 3.2.6 Peak Detection
#### 3.2.7 Plotting

![singal_evelope_and_peaks](https://github.com/user-attachments/assets/148c59c6-18e2-4956-825a-18b159089654)
