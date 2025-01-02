import wave
import numpy as np
import simpleaudio as sa
import matplotlib.pyplot as plt
from scipy.fft import dct, idct  # DCT and IDCT from scipy
from scipy.signal import find_peaks, hilbert, spectrogram

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

    def apply_cosine_transform_and_save(self, output_transformed_file, output_reconstructed_file):
        """
        Applies the Cosine Transform, saves the transformed data, and reconstructs the signal.
        Also plots the residuals between the original and reconstructed signals.
        """
        if not hasattr(self, 'trimmed_file'):
            raise ValueError("Trimmed audio not created. Run `trim_audio` first.")

        # Read the trimmed audio data
        with wave.open(self.trimmed_file, 'rb') as wav:
            params = wav.getparams()
            framerate = params.framerate
            frames = wav.readframes(params.nframes)
            audio_data = np.frombuffer(frames, dtype=np.int16)

        if audio_data.size == 0:
            print("Error: Audio data is empty.")
            return

        # Apply the Discrete Cosine Transform (DCT)
        transformed = dct(audio_data, type=2, norm='ortho')

        # Save the transformed audio (DCT)
        with wave.open(output_transformed_file, 'wb') as transformed_wav:
            transformed_wav.setparams(params)
            transformed_wav.writeframes(transformed.astype(np.int16))

        # Reconstruct the signal by applying the Inverse DCT
        reconstructed_audio = idct(transformed, type=2, norm='ortho')

        # Save the reconstructed audio
        with wave.open(output_reconstructed_file, 'wb') as reconstructed_wav:
            reconstructed_wav.setparams(params)
            reconstructed_wav.writeframes(reconstructed_audio.astype(np.int16))

        # Calculate residuals
        residuals = audio_data - reconstructed_audio

        # Plot the original audio, reconstructed audio, and residuals
        time_axis = np.linspace(0, len(audio_data) / framerate, num=len(audio_data))

        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(time_axis, audio_data, label="Original Audio")
        plt.title("Original Audio Signal")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(time_axis, reconstructed_audio, label="Reconstructed Audio", color='orange')
        plt.title("Reconstructed Audio Signal")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(time_axis, residuals, label="Residuals", color='red')
        plt.title("Residuals (Original - Reconstructed)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_spectrogram(self, input_file):
        """
        Plots the spectrogram of the audio signal.
        """
        if not hasattr(self, 'trimmed_file'):
            raise ValueError("Trimmed audio not created. Run `trim_audio` first.")

        # Read the trimmed audio data
        with wave.open(input_file, 'rb') as wav:
            params = wav.getparams()
            framerate = params.framerate
            frames = wav.readframes(params.nframes)
            audio_data = np.frombuffer(frames, dtype=np.int16)

        if audio_data.size == 0:
            print("Error: Audio data is empty.")
            return

        # Compute the spectrogram
        f, t, Sxx = spectrogram(audio_data, framerate)

        # Plot the spectrogram
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto')
        plt.title("Spectrogram of the Audio Signal")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Power (dB)")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    input_wav = "wav13.wav"
    trimmed_wav = "one_second_audio.wav"
    transformed_wav = "transformed_audio.wav"
    reconstructed_wav = "reconstructed_audio.wav"

    # Create an instance of the class
    audio_processor = AudioProcessor(input_wav)

    # Perform the tasks
    audio_processor.trim_audio(trimmed_wav, 2, 3)  # Task 1
    #audio_processor.play_audio()  # Task 2
    audio_processor.plot_envelope_and_detect_peaks(trimmed_wav) # Task 3
    audio_processor.apply_cosine_transform_and_save(transformed_wav, reconstructed_wav)  # Task 4
    audio_processor.plot_spectrogram(trimmed_wav)  # Task 5
