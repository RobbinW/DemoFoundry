"""Read proxy frames on a 24 Hz grid and encode rendered video."""

import av
import numpy as np
from PIL import Image


def load_proxy(path, width, height):
    images, timestamps = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        clock_tick = float(stream.time_base)
        for frame in container.decode(stream):
            timestamps.append(float(frame.time))
            images.append(
                frame.to_image().convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            )
    if not images:
        raise ValueError("Proxy contains no video frames.")
    pts = np.asarray(timestamps) - timestamps[0]
    times = np.arange(len(images)) / 24
    # Some containers store 24 Hz timestamps in whole milliseconds.
    if np.allclose(pts, times, rtol=0, atol=min(clock_tick, 1e-3) + 1e-8):
        return images
    times = np.arange(int(np.floor(pts[-1] * 24 + 1e-6)) + 1) / 24
    return resample(images, pts, times)


def resample(images, timestamps, times):
    right = np.searchsorted(timestamps, times).clip(0, len(images) - 1)
    left = (right - 1).clip(0)
    indices = np.where(abs(timestamps[left] - times) < abs(timestamps[right] - times), left, right)
    return [images[index] for index in indices]


def write_video(frames, path, size, fps=24):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
