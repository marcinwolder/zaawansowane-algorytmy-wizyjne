# Laboratory 7 — Optical Flow: Detailed TODO List

This document breaks down **Laboratory 7 (Przepływ optyczny / Optical Flow)** from the course skrypt into concrete, ordered tasks. Each task includes a short **explanation** (what it means and why) and a list of **concrete sub-steps** so that nothing is left ambiguous.

---

## Preliminary concepts you must understand before coding

Optical flow is a **vector field** that describes how each pixel (or a selected subset of pixels) moves between two consecutive frames of a video. For every pixel in frame `I_{n-1}` you want to find a displacement vector `(u, v)` such that, ideally, `I_{n-1}(x, y) ≈ I_n(x + u, y + v)`. You are free to compute flow between non-adjacent frames (e.g. every 2nd or 5th frame), which is sometimes useful when motion between consecutive frames is too small.

Key distinctions to keep in mind:

- **Dense vs. sparse flow.** Dense flow assigns a vector to every pixel. Sparse flow only computes vectors for a chosen subset (regular grid or feature points such as Harris corners, SIFT, SURF, FAST).
- **Brightness constancy assumption.** Classical optical flow assumes that the intensity of a tracked pixel (or small patch) does not change between frames. This breaks down in uniform regions (a large gray wall has no unique match), under illumination changes, and around occlusions.
- **Why block matching struggles with large regions of uniform color.** Inside a uniform patch, many candidate positions in the next frame look equally good, so the match is ambiguous. This is why feature-based (sparse) methods are often preferred.
- **Why multi-scale matters.** Large displacements require a wide search window, which is computationally expensive. Computing flow on a downscaled image first and then refining it at higher resolutions lets you capture large motions cheaply.

Reference methods mentioned in the skrypt: Horn–Schunck (HS), Lucas–Kanade (LK), Farneback, Dual TV-L1, PCA Flow, DIS Flow, Simple Flow, Deep Flow. Benchmarks worth knowing: Middlebury, KITTI, Sintel.

---

## Task 1 — Implement the block-matching optical flow method

**What this is.** Block matching is the simplest way to compute dense optical flow. For every pixel `(j, i)` in frame `I`, you cut out a small patch (e.g. 7×7) centered on that pixel and then scan a small neighborhood around the same coordinates in frame `J` looking for the most similar patch. The displacement from `(j, i)` to the location of the best match is the flow vector for that pixel. The rationale: motion between consecutive frames is usually small, so restricting the search saves computation and reduces false matches.

### 1.1 Script setup and frame loading

- [x] Create a new Python script for Task 1.
- [x] Load two consecutive frames from a sequence (e.g. frames **150** and **151** from the `highway` sequence). Call the earlier one `I` and the later one `J`.
- [x] Optionally downsample both frames (e.g. by a factor of 4) while developing and testing — this makes iteration fast.
- [x] Convert both frames to grayscale using `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`.
- [x] Display `I`, `J`, and the absolute difference `cv2.absdiff(I, J)` to visually confirm that there is meaningful motion between the two frames.

### 1.2 Parameters of the block-matching method

- [x] Fix the patch (window) size to **7×7**. Define `W2 = 3` (half window size, integer division of 7/2).
- [x] Define the search radius: `dX = dY = 3` pixels in each direction. This means each pixel's match is searched in a 7×7 area around its original location in `J`.

### 1.3 Outer loops over the image

- [x] Loop over every pixel `(j, i)` of the image with two nested `for` loops (outer over rows `j`, inner over columns `i`).
- [x] Use the boundary condition `W2 <= j < H - W2` and `W2 <= i < W - W2` so that the patch never leaves the image. Do **not** compute flow on the borders.

### 1.4 Cut out the reference patch from `I`

- [x] Inside the loops, extract the patch:

  ```python
  IO = np.float32(I[j - W2 : j + W2 + 1, i - W2 : i + W2 + 1])
  ```

  Note the `+1` (slicing is exclusive on the right). Converting to `float32` is required for the distance computation that follows.

### 1.5 Inner loops over the search neighborhood in `J`

- [x] Add two more nested loops, with offsets `dj` from `-dY` to `+dY` and `di` from `-dX` to `+dX`.
- [x] Inside, check that the candidate center `(j + dj, i + di)` is still a valid patch center (boundary check), **or** adjust the outer loop ranges so this check becomes unnecessary.
- [x] Extract the candidate patch `JO` from `J` with the same size as `IO` and convert it to `float32`.

### 1.6 Compute the similarity between patches

- [x] Compute a distance between `IO` and `JO`. The skrypt suggests:

  ```python
  dist = np.sqrt(np.sum(np.square(JO - IO)))
  ```

  (Sum of squared differences — SSD — followed by a square root.)
- [x] Track the offset `(dj, di)` that yields the **minimum** distance over the whole `(dX, dY)` search area. That offset is the optical-flow vector for pixel `(j, i)`.

### 1.7 Store the flow field

- [x] Before the outer loops, allocate two 2D arrays `u` and `v` with the same height/width as `I`, initialized to zero.
- [x] After finding the best offset, store the horizontal component in `u[j, i]` and the vertical component in `v[j, i]` (be consistent about which axis is which — the skrypt uses `u` for horizontal and `v` for vertical).

### 1.8 Visualize the flow field using HSV color coding

The idea is to turn the two-channel `(u, v)` field into a color image where **hue encodes the direction** of motion and **saturation encodes the magnitude** (brighter = faster). See the color wheel figure 7.2 in the skrypt.

- [x] Convert the Cartesian flow `(u, v)` to polar form:

  ```python
  mag, angle = cv2.cartToPolar(u, v)
  ```

- [x] Create an empty HSV image with the same spatial shape as `I`, 3 channels, `dtype=np.uint8`.
- [x] Set channel 0 (H) to `angle * 90 / np.pi` — OpenCV's hue range is 0–180, so this maps 0–2π onto 0–180.
- [x] Set channel 1 (S) to the magnitude normalized to the range 0–255 (e.g. `cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)`).
- [x] Set channel 2 (V) to 255.
- [x] **Swap S and V** so that zero motion renders as black instead of white (this is the note in the skrypt).
- [x] Convert HSV → BGR with `cv2.cvtColor(..., cv2.COLOR_HSV2BGR)` and display the image.

### 1.9 Experiments and analysis

- [x] Try different combinations of image resolution, `W2`, `dX`, `dY` (e.g. half-size image with `W2 = dX = dY = 5`).
- [x] At the **original** image resolution, check whether `W2 = dX = dY = 3` is enough or whether you need a larger search area for correct flow. This illustrates why larger displacements require larger windows (and why multi-scale is needed — see Task 2).
- [x] Remember: block matching gives **integer-pixel** displacements only — it cannot produce sub-pixel flow like Horn–Schunck or Lucas–Kanade.

### 1.10 Reduce spurious matches using the frame difference

- [x] Compute the absolute difference between `I` and `J` (you already did this for visualization).
- [x] Binarize it with an appropriate threshold.
- [x] Dilate the binary mask (try several kernel sizes and iteration counts).
- [x] Only output flow vectors where the dilated difference mask is non-zero — this suppresses flow estimates in static background regions where any match is likely spurious.

---

## Task 2 — Multi-scale (pyramidal) block-matching flow

**What this is.** To detect large displacements without using a huge (expensive) search window, you build an image pyramid — a stack of progressively downscaled copies of the frame — and compute flow from the coarsest level to the finest. At each level you (a) estimate the flow, (b) upscale it to the next level, (c) **warp** the second frame using the current flow estimate so that motion is already mostly compensated, and then (d) compute only the small residual flow at the higher resolution. The final flow is the sum of the upscaled estimates from every scale. Refer to figures 7.4 and 7.5 in the skrypt.

### 2.1 Refactor Task 1 into reusable functions

- [x] Wrap the block-matching computation from Task 1 in a function with signature:

  ```python
  def of(J_org, I, J, W2=3, dY=3, dX=3):
      ...
      return u, v
  ```

  `J_org` is only passed in for visualization (to compute `absdiff` against `I` and display the three images). The actual matching happens between `I` and the possibly-warped `J`.
- [x] Wrap the HSV visualization from step 1.8 in a function:

  ```python
  def vis_flow(u, v, YX, name):
      ...
  ```

  where `YX` is the `(height, width)` of the image and `name` is the window title (e.g. `'of scale 2'`).
- [x] Verify the refactored code produces the same result as Task 1 when run at a single scale.

### 2.2 Build the image pyramid

- [x] Implement the helper:

  ```python
  def pyramid(im, max_scale):
      images = [im]
      for k in range(1, max_scale):
          images.append(cv2.resize(images[k-1], (0, 0), fx=0.5, fy=0.5))
      return images
  ```

  This produces `[original, halved, quartered, ...]` up to `max_scale` levels.
- [x] Call `pyramid(...)` on both `I` and `J`. For the experiment, limit yourself to 2 or 3 scales — more levels tend to accumulate interpolation errors in practice.

### 2.3 Main multi-scale loop

The algorithm iterates from the smallest scale up to the original. At each step you add the flow from the current scale to a running total.

- [x] Pick the smallest-scale frames as the starting point:

  ```python
  I = IP[-1]
  J = JP[-1]
  ```

- [x] Create two accumulator arrays `U` and `V` (full image size, zeros) for the total flow.
- [x] Loop over scales from smallest to largest. For each scale `s` (with `s = max_scale - 1` at the coarsest, down to `s = 0` at the original):
  - [x] Compute `u, v = of(J_org, I, J, ...)` at the current scale.
  - [x] **Resize the flow** to the next larger scale, doubling the spatial dimensions **and** doubling the numerical values of the vectors:

    ```python
    u_up = cv2.resize((2**s) * u, (0, 0), fx=2**s, fy=2**s, interpolation=cv2.INTER_LINEAR)
    v_up = cv2.resize((2**s) * v, (0, 0), fx=2**s, fy=2**s, interpolation=cv2.INTER_LINEAR)
    ```

    Doubling the *values* is critical: a displacement of 5 px at scale 1/2 corresponds to 10 px at the original resolution.
  - [x] Add `u_up`, `v_up` into the accumulators `U`, `V`.

### 2.4 Warping the second image

After estimating flow at a coarse scale, you want to **move the pixels of `J` back towards their positions in `I`** according to that flow, so that at the next finer scale only the small residual motion remains to be estimated.

- [x] Make a copy `J_new` of `J` at the next (larger) scale.
- [x] For each pixel `(j, i)` (use two `for` loops, and guard against going out of bounds), set:

  ```python
  J_new[j, i] = J[j + v_estimate, i + u_estimate]
  ```

  using the integer flow values already computed and upscaled for this scale.
- [x] Visually check that `J_new` now looks close to `I` — if so, the warping step is correct.
- [x] Assign `J = J_new` for the next iteration, and set `I` to the original frame at the new scale.
- [x] **Do not warp** at the finest (original) scale — at that point you just compute the final residual flow and add it to the accumulator.

### 2.5 Final visualization and comparison

- [x] Visualize the **total** flow `(U, V)` (summed over all scales) with `vis_flow(...)`.
- [x] Compare with:
  - Single-scale block matching using a **large** window (expensive).
  - Single-scale block matching using a **small** window (fast but fails on large motion).
  - Multi-scale block matching using a **small** window (fast *and* handles large motion).
- [x] Note the runtime of each configuration.
- [x] Check on a frame pair with large displacements (e.g. the `highway` sequence): did the multi-scale version successfully recover the flow where the small-window single-scale version failed?
- [x] Be aware that, due to interpolation errors when upscaling flow and downscaling images, pyramidal flow is never perfect. 2–4 scales is usually the sweet spot.

---

## Task 3 — Use OpenCV's built-in optical-flow methods

**What this is.** OpenCV ships with multiple optical flow algorithms that are faster and more accurate than the hand-rolled block-matching from Tasks 1 and 2. You should try several of them, visualize the output the same way, and compare quality and runtime.

### 3.1 Preparation

- [ ] Reuse (or adapt) your sequence-reading script from previous labs. Make sure it accepts a parameter `iStep` that controls which frames are processed (e.g. every frame, every 2nd, etc.).
- [ ] Work in grayscale for simplicity.
- [ ] Read the OpenCV optical flow tutorial and documentation for the API details.

### 3.2 Dense methods — sub-task (a)

Implement the following pattern for each of these algorithms: **Farneback**, **Dual TV-L1**, **PCA Flow**, **DIS Flow**, **Simple Flow**, **Deep Flow**, and the **dense Lucas–Kanade** variant.

- [ ] Load two consecutive frames and convert both to grayscale.
- [ ] Allocate a two-channel `flow` array to hold `(u, v)`.
- [ ] Create an instance of the chosen algorithm (e.g. `cv2.FarnebackOpticalFlow_create()` / `cv2.optflow.createOptFlow_DIS()` / etc.).
- [ ] Call its `.calc(prev, next, flow)` method.
- [ ] Visualize with the same HSV function from Task 1.
- [ ] For dense LK specifically, clip the magnitude before HSV mapping to avoid oversaturation:

  ```python
  mag[mag > 10] = 10
  ```

For each algorithm note:

- [ ] Runtime per frame pair.
- [ ] Accuracy around object edges.
- [ ] Robustness to uniform regions.

### 3.3 Sparse Lucas–Kanade — sub-task (b)

Here you compute flow only for a chosen set of points. The skrypt wants a **regular grid** of tracked points rather than Harris/SIFT corners.

- [ ] Generate a grid of points spaced every 10 pixels across the image. The data structure needed by `cv2.calcOpticalFlowPyrLK` is a float array of shape `(N, 1, 2)`.
- [ ] Call the sparse LK function, e.g.:

  ```python
  new_points, st, err = cv2.calcOpticalFlowPyrLK(prev, next, points, None, winSize=(15, 15), maxLevel=2, criteria=...)
  ```

- [ ] Understand each parameter (window size, pyramid depth, termination criteria for the iterative solver). Tune them if needed.
- [ ] Visualize by drawing a line from each `points[i]` to `new_points[i]` only where `st[i] == 1` (successfully tracked):

  ```python
  for jj in range(len(points)):
      if st[jj] == 1:
          cv2.line(img, tuple(points[jj, 0]), tuple(new_points[jj, 0]), (0, 0, 255))
  ```

- [ ] Optionally mark the arrow tail or head with a small dot to show direction.
- [ ] Optionally filter out vectors with tiny magnitude.
- [ ] At the end of each frame iteration, **don't forget** to copy the current frame into the "previous frame" variable for the next iteration.

### 3.4 Comparison

- [ ] Compare the sparse grid-LK output with the dense methods from step 3.2. Which methods cope well with uniform regions? Which are fastest? Which give the cleanest object boundaries?

---

## Task 4 — Extra: motion-direction detection and object classification using optical flow

**What this is.** The idea is that a rigid body (e.g. a car) exhibits coherent motion — all its pixels move in roughly the same direction — while a walking person shows incoherent motion because the limbs swing in opposite directions depending on the gait phase. By computing per-object statistics (mean and standard deviation of flow magnitude and angle), you may be able to distinguish cars from pedestrians.

### 4.1 Add foreground-object segmentation

- [ ] Start from your optical-flow script from Task 3 (use Farneback's dense flow, for example).
- [ ] Add MOG/GMM foreground segmentation (from an earlier lab). Instantiate with `cv2.createBackgroundSubtractorMOG2(detectShadows=False)` — disabling shadows prevents them being picked up as separate objects.

### 4.2 Filter the mask

- [ ] Apply morphological operations (erode + dilate, opening/closing) and/or median filtering to clean up the mask, just as in the earlier background-subtraction lab.

### 4.3 Connected-component labeling (indexation)

- [ ] Run `cv2.connectedComponentsWithStats(mask)` to label each foreground blob with a unique index.
- [ ] Visualize the result (scale the label image to 0–255 for display).

### 4.4 Per-object flow statistics

This is the core of the analysis.

- [ ] Only proceed if `retval > 0` (i.e. at least one object exists).
- [ ] For each object label, maintain **two Python lists**: one for per-pixel flow **magnitudes** and one for per-pixel flow **angles**. Using two separate lists (rather than a list of tuples) makes the later statistics computation simpler.
- [ ] Loop over every pixel in the image. For each pixel:
  - [ ] Skip if its label is 0 (background).
  - [ ] Skip if the flow magnitude at that pixel is below a small threshold (e.g. 1 pixel) — this removes noise and pixels that are effectively static.
  - [ ] Compute the angle with `math.atan2(v, u)`.
  - [ ] Append the magnitude and the angle to the per-object lists (indexed by the object label).
- [ ] After the loop, for each object with at least 2 samples, compute:
  - [ ] Mean and standard deviation of the magnitude list (use `statistics.mean` and `statistics.stdev`).
  - [ ] Mean and standard deviation of the angle list.
- [ ] Handle the edge case where an object has fewer than 2 pixels of meaningful flow (otherwise `stdev` will raise).

### 4.5 Visualization

- [ ] Reuse the bounding-box-drawing code from the background-subtraction lab.
- [ ] On top of each bounding box, overlay the four numbers (mag mean, mag std, angle mean, angle std) using `cv2.putText`.
- [ ] Optionally filter out very small blobs (likely noise) before drawing.
- [ ] Optionally draw a single arrow per object representing its mean motion vector.

### 4.6 Analysis questions to answer

- [ ] How do mean and standard deviation compare between the two sequences (`highway` with cars vs. `pedestrians` with people)?
- [ ] Does the mean motion direction match the real-world direction of travel?
- [ ] Is the angular standard deviation noticeably higher for pedestrians than for cars (the core hypothesis of this exercise)?
- [ ] Are there systematic segmentation problems? If so, which?

---

## Summary checklist

- [ ] Task 1 completed: working single-scale block-matching flow with HSV visualization, plus frame-difference-based filtering of spurious matches.
- [ ] Task 2 completed: multi-scale pyramidal block-matching flow that handles large displacements without a huge search window, with warping of the second frame between scales and final accumulation of flow across scales.
- [ ] Task 3 completed: comparison of at least several OpenCV dense methods (Farneback and others) plus sparse Lucas–Kanade on a regular grid, including runtime and quality notes.
- [ ] Task 4 (extra) completed: per-object flow statistics combined with MOG-based foreground segmentation, with an attempt to distinguish cars from pedestrians by the standard deviation of their motion angles.
- [ ] Each task's script saved in the expected format (`.py` or `.ipynb`) and uploaded to the UPeL course platform after the instructor approves the work.
