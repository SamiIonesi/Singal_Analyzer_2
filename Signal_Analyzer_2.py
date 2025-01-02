import wave
import numpy as np
import simpleaudio as sa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert

class AudioProcessor:
    def __init__(self, input_file):
        self.input_file = input_file

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
        
        #if len(peaks) > 0:
        #    print(f"Peak times (s): {peak_times}")

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



if __name__ == "__main__":
    input_wav = "wav13.wav"
    trimmed_wav = "one_second_audio.wav"

    # Create an instance of the class
    audio_processor = AudioProcessor(input_wav)

    # Perform the tasks
    audio_processor.trim_audio(trimmed_wav, 2, 3)  # Task 1
    #audio_processor.play_audio()  # Task 2
    audio_processor.plot_envelope_and_detect_peaks(trimmed_wav)  # Task 3
