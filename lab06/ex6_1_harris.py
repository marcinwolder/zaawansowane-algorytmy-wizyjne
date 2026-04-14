from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib.pyplot as plt


def harris_response(
    image_gray: np.ndarray,
    sobel_ksize: int = 7,
    gauss_ksize: int = 7,
    k: float = 0.05,
) -> np.ndarray:
    """Compute normalized Harris response H in range [0, 1]."""
    image_f = image_gray.astype(np.float32)

    ix = cv2.Sobel(image_f, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    iy = cv2.Sobel(image_f, cv2.CV_32F, 0, 1, ksize=sobel_ksize)

    ixx = cv2.GaussianBlur(ix * ix, (gauss_ksize, gauss_ksize), sigmaX=0)
    iyy = cv2.GaussianBlur(iy * iy, (gauss_ksize, gauss_ksize), sigmaX=0)
    ixy = cv2.GaussianBlur(ix * iy, (gauss_ksize, gauss_ksize), sigmaX=0)

    det_m = ixx * iyy - ixy * ixy
    trace_m = ixx + iyy
    h = det_m - k * (trace_m * trace_m)

    h_min = float(h.min())
    h_max = float(h.max())
    if h_max == h_min:
        return np.zeros_like(h, dtype=np.float32)
    return ((h - h_min) / (h_max - h_min)).astype(np.float32)


def find_max(image: np.ndarray, size: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (ys, xs) for local maxima above threshold."""
    kernel = np.ones((size, size), dtype=np.uint8)
    data_max = cv2.dilate(image, kernel)
    maxima = (image == data_max) & (image > threshold)
    return np.nonzero(maxima)


def draw_points_with_stars(
    image_gray: np.ndarray,
    points: tuple[np.ndarray, np.ndarray],
    title: str,
    output_path: Path,
) -> None:
    ys, xs = points

    plt.figure(figsize=(10, 7))
    plt.imshow(image_gray, cmap="gray")
    plt.plot(xs, ys, "r*", markersize=4)
    plt.title(f"{title} | points: {len(xs)}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def process_pair(
    left_path: Path,
    right_path: Path,
    output_dir: Path,
    sobel_size: int = 7,
    gauss_size: int = 7,
    max_size: int = 7,
    threshold: float = 0.6,
) -> None:
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

    if left is None:
        raise FileNotFoundError(f"Cannot read image: {left_path}")
    if right is None:
        raise FileNotFoundError(f"Cannot read image: {right_path}")

    h_left = harris_response(left, sobel_ksize=sobel_size, gauss_ksize=gauss_size)
    h_right = harris_response(right, sobel_ksize=sobel_size, gauss_ksize=gauss_size)

    pts_left = find_max(h_left, size=max_size, threshold=threshold)
    pts_right = find_max(h_right, size=max_size, threshold=threshold)

    stem = left_path.stem.replace("1", "")

    draw_points_with_stars(
        left,
        pts_left,
        f"{left_path.name}",
        output_dir / f"{stem}_1_points.png",
    )
    draw_points_with_stars(
        right,
        pts_right,
        f"{right_path.name}",
        output_dir / f"{stem}_2_points.png",
    )

    print(f"{left_path.name}: {len(pts_left[0])} points")
    print(f"{right_path.name}: {len(pts_right[0])} points")


def main() -> None:
    data_dir = BASE_DIR / "data"
    out_dir = BASE_DIR / "out_ex6_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    process_pair(data_dir / "fontanna1.jpg", data_dir / "fontanna2.jpg", out_dir)
    process_pair(data_dir / "budynek1.jpg", data_dir / "budynek2.jpg", out_dir)

    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
