import math
import os
import random
import tempfile

import numpy as np
import torch
import torchaudio
from cog import BasePredictor, Input, Path
from torch.nn.functional import pad
from tqdm import tqdm

from audiosr import build_model, download_checkpoint, super_resolution

MODELS = ["basic", "speech"]

# " Warning: audio is longer than 10.24 seconds, may degrade the
# model performance. It's recommand to truncate your audio to 5.12
# seconds before input to AudioSR to get the best performance."
WINDOW_AUDIO_LENGTH = 5.12
OUTPUT_SAMPLE_RATE = 48000

# THIS IS WHY WE CAN'T HAVE NICE THINGS, audiosr
# returns audio that is a different length than expected.
# For reasons I don't understand and that aren't documented:
# 5.12 sec * 48000 = 245760 samples.
# This is also divisible by 2048, the STFT window size they use.
# Nonetheless, for 5.12 sec audio input, audiosr returns 245776
# samples. They don't explain whether this is center or end padded
# but my current best guess is that it's end padded.
ACTUAL_AUDIOSR_OUTPUT_WINDOW_SAMPLES = 245776

AUDIOSR_STFT_WINDOW_LENGTH = 2048

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")


class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        for model in MODELS:
            download_checkpoint(model)
        self.audiosrs = {}

    def predict(
        self,
        input_file: Path = Input(description="Audio to upsample"),
        model: str = Input(
            description="Checkpoint (basic or speech)",
            default="basic",
            choices=["basic", "speech"],
        ),
        ddim_steps: int = Input(
            description="Number of inference steps", default=50, ge=10, le=500
        ),
        guidance_scale: float = Input(
            description="Scale for classifier free guidance",
            default=3.5,
            ge=1.0,
            le=20.0,
        ),
        overlap: float = Input(
            description="Window overlap, higher will give better predictions, between 0 and 1",
            default=0.75,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        output_bitrate: int = Input(
            description="WAV output bitrate", default=32, choices=[32, 16]
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Setting seed to: {seed}")

        if model not in self.audiosrs:
            self.audiosrs[model] = build_model(model_name=model, device=self.device)

        waveform = self.apply_model(
            input_file=input_file,
            model=model,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            overlap=overlap,
            seed=seed,
        )

        if waveform.max().abs() > 1:
            waveform /= waveform.max().abs()
        if output_bitrate == 16:
            waveform = (waveform[0] * 32767).astype(np.int16).T
        else:
            assert output_bitrate == 32, "Unsupported output bitrate"

        torchaudio.save("out.wav", waveform.cpu(), OUTPUT_SAMPLE_RATE)

        return Path("out.wav")

    def apply_model(self, input_file, model, ddim_steps, guidance_scale, overlap, seed):
        waveform, sample_rate = torchaudio.load(input_file)
        assert (
            sample_rate < OUTPUT_SAMPLE_RATE
        ), f"sample_rate: {sample_rate} should not exceed output_sample_rate: {OUTPUT_SAMPLE_RATE}"

        num_channels, num_samples = waveform.shape

        # Define window and hop length
        window_size = int(WINDOW_AUDIO_LENGTH * sample_rate)
        hop_length = int(window_size * (1 - overlap))

        # Prepare windows and padding amounts
        pad_start, pad_end, windows = prepare_windows(
            num_channels, num_samples, window_size, hop_length
        )

        hanning_window = torch.hann_window(AUDIOSR_STFT_WINDOW_LENGTH, periodic=False).to(self.device)

        def upsample(n):
            return n * OUTPUT_SAMPLE_RATE // sample_rate

        def upsample_and_stft_pad(n):
            # Since STFT padding will occur, we need to adjust the num samples
            n = upsample(n)
            return int(
                round(n / AUDIOSR_STFT_WINDOW_LENGTH) * AUDIOSR_STFT_WINDOW_LENGTH
            )

        upsampled_num_samples = upsample(num_samples)

        upsampled_window_size = upsample_and_stft_pad(window_size)

        # Apply padding to the waveform
        padded_waveform = self._apply_padding(waveform, pad_start, pad_end)

        upsampled_pad_start = upsample(pad_start)

        upsampled_padded_num_samples_from_audiosr_padding = upsample(windows[-1][0]) + ACTUAL_AUDIOSR_OUTPUT_WINDOW_SAMPLES
        #upsampled_padded_num_samples = upsample(windows[-1][1])

        # Doesn't work because audiosr doesn't explain why
        # the padding changes on the audio
        #upsampled_padded_num_samples = upsample_and_stft_pad(upsampled_num_samples)

        upsampled_waveform_shape = (num_channels, upsampled_padded_num_samples_from_audiosr_padding)
        print(f"upsampled_waveform_shape: {upsampled_waveform_shape}")

        # Initialize the output waveform and window sums for averaging
        output_waveform = torch.zeros(upsampled_waveform_shape).to(self.device)
        window_sums = torch.zeros(upsampled_waveform_shape[1]).to(self.device)

        # Process each window without batching
        # TODO: This could probably work in a batched way, if we dug into
        # the super_resolution API. Actually just writing a file with a
        # lot of channels might work, except the super_resolution
        # normalization is opinionated and is not per-channel.
        # (Except I updated that so we CAN try.)
        for start, end in tqdm(windows):
            upsampled_start = upsample(start)
            upsampled_end = upsample_and_stft_pad(end)
            assert (
                upsampled_end - upsampled_start == upsampled_window_size
            ), f"{upsampled_end - upsampled_start} != {upsampled_window_size}"
            print(f"{start} -> {end}, {upsampled_start} -> {upsampled_end}")

            # Extract the current window from the padded waveform
            windowed_waveform = padded_waveform[:, start:end]

            # Assert that each windowed_waveform is exactly window_size
            assert (
                windowed_waveform.shape[1] == window_size
            ), f"windowed_waveform shape: {windowed_waveform.shape[1]}, expected: {window_size}"

            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                # We do this weird thing because the API of super_resolution
                # is a bit opinionated
                torchaudio.save(f.name, windowed_waveform, sample_rate)

                with torch.no_grad():
                    print(f"input shape: {windowed_waveform.shape}")
                    output_window = super_resolution(
                        self.audiosrs[model],
                        f.name,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        ddim_steps=ddim_steps,
                        latent_t_per_second=12.8,
                    )
                    print(f"output shape: {output_window.shape}")
                    # Gross, audiosr should fix this too
                    output_window = torch.tensor(output_window, device=self.device)
                    assert (
                        output_window.ndim == 3 and output_window.shape[0] == 1
                    ), f"{output_window.ndim} != 3"
                    output_window = output_window[0, :, :]
                    print(f"output shape: {output_window.shape}")
                    assert (
                        output_window.shape[0] == num_channels
                    ), f"{output_window.shape[0]} != {num_channels}, output = {output_window.shape}"
                    # assert (
                    #    output_window.shape[1] == upsampled_end - upsampled_start
                    # ), f"{output_window.shape[1]} > {upsampled_end - upsampled_start}"
                    assert (
                        output_window.shape[1] == ACTUAL_AUDIOSR_OUTPUT_WINDOW_SAMPLES
                    ), f"{output_window.shape[1]} != {ACTUAL_AUDIOSR_OUTPUT_WINDOW_SAMPLES}"

            if output_window.shape[1] != upsampled_end - upsampled_start:
                print(
                    f"Waveform shape changed slightly during upsampling: {output_window.shape[1]}, expected: {upsampled_end - upsampled_start}"
                )
                # Pad the end
                upsampled_end = upsampled_start + output_window.shape[1]
                # print(f"Upsampled end: {upsampled_end}")
                # print(f"Upsampled start: {upsampled_start}")

            # Accumulate the output for overlapping windows, with hanning
            assert upsampled_end - upsampled_start == hanning_window.shape[0], f"{upsampled_end - upsampled_start} != {hanning_window.shape[0]}"
            output_waveform[:, upsampled_start:upsampled_end] += output_window * hanning_window
            window_sums[upsampled_start:upsampled_end] += hanning_window

        # Avoid division by zero in window sums
        window_sums[window_sums == 0] = 1e-4

        # Average the overlapping windows
        output_waveform = output_waveform / window_sums

        # Remove the padding to return to the original waveform length
        output_waveform = output_waveform[
            :, upsampled_pad_start : upsampled_num_samples + upsampled_pad_start
        ]

        return output_waveform

    def _apply_padding(self, waveform, pad_start, pad_end):
        try:
            padded_waveform = pad(waveform, (pad_start, pad_end), mode="reflect")
        except:
            # Fallback to constant padding if reflect padding fails
            padded_waveform = pad(waveform, (pad_start, pad_end))
        return padded_waveform


def prepare_windows(
    num_channels: int, num_samples: int, window_size: int, hop_length: int
):
    """
    Prepare windows for batched processing and overlap add.

    Let's just keep this simple and clean.
    We will definitely add at least window_size padding on each side.
    (Slightly inefficient for short audio.)

    We then construct windows at least this length, of window_size length.

    Finally, we reverse engineer the padding.
    """
    # print(f"num_samples={num_samples}, window_size={window_size}, hop_length={hop_length}")

    # We will definitely add at least window_size padding on each side
    min_total_samples = num_samples + 2 * window_size

    windows = [(0, window_size)]
    while windows[-1][1] < min_total_samples:
        windows.append((windows[-1][0] + hop_length, windows[-1][1] + hop_length))

    # print(f"windows = {windows}")

    final_total_samples = windows[-1][1]
    assert (
        final_total_samples >= min_total_samples
    ), f"{final_total_samples} < {min_total_samples}"

    # We might end up with some blank extra windows at the beginning and end
    # but whatever, this implementation is CORRECT and SAFEa
    total_padding_needed = final_total_samples - num_samples

    # print(f"total_padding_needed = {total_padding_needed}")

    # Determine the padding for the start and end
    pad_start = total_padding_needed // 2
    pad_end = total_padding_needed - pad_start

    # print(f"pad_start = {pad_start}, pad_end = {pad_end}")

    return pad_start, pad_end, windows


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    out = p.predict("example/music.wav", ddim_steps=50, guidance_scale=3.5, seed=42)
