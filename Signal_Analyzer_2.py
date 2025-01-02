import wave
import numpy as np
import simpleaudio as sa

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

    def make_one_second_audio(self, output_file):
        """
        Ensures the audio signal is exactly one second long.
        """
        if not hasattr(self, 'trimmed_file'):
            raise ValueError("Trimmed audio not created. Run `trim_audio` first.")

        with wave.open(self.trimmed_file, 'rb') as wav:
            # Get audio parameters
            params = wav.getparams()
            framerate = params.framerate
            n_channels = params.nchannels
            sampwidth = params.sampwidth

            # Read all frames
            frames = wav.readframes(wav.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)

            # Determine length of one second in samples
            one_second_frames = framerate * n_channels

            # Adjust audio to be exactly one second
            if len(audio_data) > one_second_frames:
                # Trim to one second
                audio_data = audio_data[:one_second_frames]
            elif len(audio_data) < one_second_frames:
                # Repeat to fill one second
                repeat_count = one_second_frames // len(audio_data)
                remainder = one_second_frames % len(audio_data)
                audio_data = np.tile(audio_data, repeat_count)
                audio_data = np.append(audio_data, audio_data[:remainder])

            # Convert back to bytes
            output_frames = audio_data.tobytes()

            # Write to a new file
            with wave.open(output_file, 'wb') as one_sec_wav:
                one_sec_wav.setparams(params)
                one_sec_wav.writeframes(output_frames)

        print(f"One-second audio saved as '{output_file}'.")
        self.one_second_file = output_file

    def play_audio(self):
        """
        Plays the one-second audio file.
        """
        if not hasattr(self, 'one_second_file'):
            raise ValueError("One-second audio not created. Run `make_one_second_audio` first.")

        with wave.open(self.one_second_file, 'rb') as wav:
            # Extract audio parameters and data
            params = wav.getparams()
            frames = wav.readframes(params.nframes)

        # Play audio using simpleaudio
        play_obj = sa.play_buffer(frames, num_channels=params.nchannels, bytes_per_sample=params.sampwidth, sample_rate=params.framerate)
        play_obj.wait_done()  # Wait for playback to finish


if __name__ == "__main__":
    input_wav = "wav13.wav"
    trimmed_wav = "trimmed_audio.wav"
    one_second_wav = "one_second_audio.wav"

    # Create an instance of the class
    audio_processor = AudioProcessor(input_wav)

    # Perform the tasks
    audio_processor.trim_audio(trimmed_wav, 2, 3)  # Task 1
    audio_processor.make_one_second_audio(one_second_wav)  # Task 2
    audio_processor.play_audio()  # Play the one-second audio
