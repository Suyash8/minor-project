# Findings - Object Detection for Autonomous Vehicles

This document synthesizes the literature survey recorded in `sources.md`. All claims
below are traceable to a specific source listed there; where I generalize across sources,
the relevant section numbers from `sources.md` are cited in parentheses.

## Scope and narrowing

"Object detection for autonomous vehicles" is too broad to survey exhaustively, so this
survey was narrowed to six concrete use cases that, together, cover the perception stack
of a real AV from sensor input to deployment constraint:

1. **2D camera-based real-time detection** - the baseline problem: find and classify cars,
   pedestrians, cyclists, signs, etc. in a single camera frame fast enough to drive on.
2. **3D / LiDAR point-cloud detection** - recovering depth and precise 3D location/size,
   which 2D image detection alone cannot provide.
3. **Multi-sensor fusion** (camera + LiDAR, and camera + radar as a noted adjacent area) -
   combining sensors to cover each one's blind spots.
4. **Robustness under adverse weather** - rain, fog, night, snow - conditions where clean-
   weather-trained detectors degrade.
5. **Vulnerable road user (VRU) / pedestrian detection** - treated as a distinct sub-problem
   because of small object size, occlusion, and behavioral unpredictability.
6. **Edge/embedded deployment** - the constraint that all of the above must run in real
   time on limited, power-constrained, in-vehicle compute, not a data-center GPU.

Benchmark datasets (KITTI, nuScenes) are treated as a cross-cutting seventh area since
almost every paper above is evaluated against them.

## Development timeline observed across sources

Reconstructing publication order from the sources gathered:

- **2012** - KITTI benchmark suite (sources.md §4.1) establishes the first standardized,
  multi-task (2D/3D detection, stereo, flow, SLAM) driving benchmark, recorded from a
  single car with camera + LiDAR in one German city.
- **2016** - SSD (§1.1) demonstrates that single-stage detectors can match two-stage
  accuracy at real-time speed, a prerequisite for any onboard use.
- **2017–2019** - VoxelNet (§2.1) then PointPillars (§2.2) move 3D detection from
  hand-crafted point-cloud features to end-to-end learned representations, with
  PointPillars in particular hitting a speed/accuracy point usable in real time.
- **2019** - nuScenes (§3.2) generalizes the benchmark idea from KITTI to a full sensor
  suite (6 cameras, LiDAR, 5 radars) across 1,000 scenes, becoming the standard for
  fusion research; VRU-specific surveys (§6.1, §6.2) formalize pedestrian/cyclist detection
  as its own sub-field with intent estimation as the natural next step.
- **2020–2022** - Adverse-weather robustness becomes a dedicated research thread (rain
  review, §5.1); YOLOP (§1.2) and MobileYOLO (§1.3) show the shift toward multi-task and
  compressed-for-embedded detection, respectively; the camera+LiDAR fusion taxonomy survey
  (§3.1) organizes a fast-growing fusion literature.
- **2021–2024** - 3D detection survey (§2.3) takes stock of the field and explicitly adds
  runtime/robustness analysis as an evaluation axis alongside accuracy; adverse-weather
  work matures from "note the problem" to "engineer a fix" (YOLOv8 + merged-dataset
  fine-tuning, §5.3); edge-device benchmarking (§1.4) formalizes the accuracy/latency/
  energy trade-off across concrete embedded hardware (Raspberry Pi, Jetson).

Read together, the field's trajectory is: **establish a benchmark → get single-stage 2D
detection fast enough → extend to 3D via LiDAR → fuse sensors → confront real-world
robustness gaps (weather, VRUs) → compress/optimize for actual embedded hardware.** Each
stage exists because the previous one is a precondition for it, not a random research
trend - you cannot productively study weather robustness or embedded deployment for a
detector that isn't first real-time-capable and multi-sensor-aware.

## Findings by use case

### 1. 2D real-time detection
One-stage detectors (SSD, YOLO family) dominate AV perception because AV detection is
latency-critical: a two-stage detector's separate region-proposal step adds sequential
computation that one-stage detectors avoid by predicting boxes and classes directly from
feature maps in a single pass (§1.1; see also the architecture diagram in §7.3). The
trend within one-stage detection is now toward *multi-task* single networks - YOLOP
(§1.2) performs detection + drivable-area segmentation + lane detection together, and
still hits real-time speed on embedded Jetson hardware - because an AV needs all three
outputs anyway, and sharing one encoder across tasks is cheaper than running three
separate networks.

### 2. 3D / LiDAR detection
2D image detection cannot tell you how far away an object is or its true 3D extent, which
is why LiDAR-based 3D detection is treated as a separate, necessary capability rather than
a refinement of 2D detection (§2.3). The dominant architectural question in this area is
how to encode a sparse, irregular point cloud into something a standard CNN can process
efficiently: VoxelNet's 3D voxel grid (§2.1) was the first end-to-end learned answer, and
PointPillars' vertical-column ("pillar") encoding (§2.2) proved that you can drop the
expensive 3D convolutions and still get strong accuracy by projecting to a 2D
pseudo-image - this pillar-based approach is why PointPillars remains a standard baseline
in later benchmarks.

### 3. Multi-sensor fusion
No single sensor is sufficient: cameras give rich semantic detail but no reliable depth
and degrade in poor light/weather; LiDAR gives precise 3D geometry but is sparse and
expensive; radar is robust to weather but low-resolution. The key organizing question in
fusion research is not "which sensors" but "at what stage do you combine them" - the
camera+LiDAR fusion survey (§3.1) proposes a taxonomy based on fusion stage (raw
data/early vs. feature-level vs. proposal/decision-level vs. hybrids) precisely because
that design choice, more than sensor choice, determines a fusion architecture's
accuracy/latency/robustness trade-offs. The nuScenes dataset (§3.2) is the standard
testbed for this area because it is the first large public dataset to include radar
alongside camera and LiDAR.

### 4. Adverse weather robustness
Clean-weather-trained detectors reliably lose accuracy in rain, fog, snow, and low light
(§5.1, §5.2) - this is treated in the literature as an expected, well-documented failure
mode, not an edge case. Two mitigation strategies recur: (a) pre-process/enhance the image
before detection (deraining, contrast restoration, domain adaptation/translation, per
§5.1), or (b) fine-tune the detector directly on weather-degraded data, which the YOLOv8 +
ACDC/DAWN dataset-merging study (§5.3) shows produces steady accuracy gains as more
weather-diverse training data is merged in - i.e., dataset diversity is currently a more
reliable lever than architectural changes for this specific problem.

### 5. VRU / pedestrian detection
Pedestrians and cyclists are harder to detect reliably than vehicles because they are
smaller, more deformable, more often partially occluded, and behaviorally less
predictable (§6.1). Because a detection alone doesn't tell an AV whether a pedestrian is
about to step into the road, the field has moved toward pairing detection with tracking,
motion modeling, and pose-based intent estimation (§6.2) - detection is increasingly
framed as the necessary first stage of a "detect → track → predict intent" pipeline rather
than a standalone deliverable.

### 6. Edge/embedded deployment
Every architectural gain above (multi-task heads, pillar encodings, fusion) is
constrained by what can actually run within an AV's power and compute budget. The
edge-device benchmarking study (§1.4) puts hard numbers on this: SSD-MobileNetV1 is
fastest and most energy-efficient but least accurate; YOLOv8-Medium is most accurate but
most compute-hungry; hardware accelerators (Coral TPU) help lightweight models (SSD,
EfficientDet-Lite) more than they help YOLOv8; and the Jetson Orin Nano currently offers
the best overall accuracy/latency/energy balance among the tested embedded platforms.
MobileYOLO (§1.3) is a model-level answer to the same pressure - a MobileNetv2 backbone
plus depthwise-separable convolutions cut YOLOv4's parameter count by over 50M while
increasing detection speed by 70% on KITTI, illustrating the general recipe (lighter
backbone + separable convolutions + lightweight attention) used across this literature to
compress detectors for onboard hardware.

## Cross-cutting observations

- **Benchmarks shape the field as much as architectures do.** KITTI (§4.1) and nuScenes
  (§3.2) are referenced by nearly every other paper surveyed; a new detection/fusion
  method is judged largely by how it compares on these two datasets. KITTI's limitation -
  one city, daytime, clear weather - is part of why adverse-weather and VRU-specific
  research had to build (or reuse) additional datasets (ACDC, DAWN, per §5.3) rather than
  relying on KITTI alone.
- **Real-time constraint is the common thread across all six use cases.** Whether the
  topic is 2D detection, 3D detection, fusion, weather robustness, or VRU detection, nearly
  every paper surveyed reports a speed metric (FPS, ms latency, or explicit embedded-device
  benchmarking) alongside accuracy - accuracy without a latency number is treated as an
  incomplete result in this domain.
- **The "detect accurately" problem is gradually being absorbed into larger pipelines.**
  YOLOP folds detection into a shared multi-task network (§1.2); VRU research folds
  detection into a detect+track+predict-intent pipeline (§6.2); adverse-weather research
  folds detection into an enhance-then-detect or domain-adapt pipeline (§5.1, §5.3). Plain
  single-frame, single-task object detection is treated in current literature as a solved
  enough sub-problem that most active research composes it with something else, rather
  than trying to push single-task accuracy further in isolation.

## Gaps and open challenges noted in the surveyed literature

- Occlusion and small/far-away object detection remain open problems for VRUs specifically
  (§6.1).
- Weather robustness solutions are mostly dataset- and fine-tuning-driven rather than
  architecturally solved, meaning performance is bounded by how representative the
  training data is of the deployment weather conditions (§5.1, §5.3).
- Embedded-hardware benchmarking (§1.4) shows accuracy still degrades measurably as scene
  complexity (object count) increases, even on the best-performing hardware/model
  combinations - a concern for dense urban driving scenes specifically.
- Two MDPI-hosted reviews relevant to adverse weather (§5.2, §5.3 partially) and one
  MDPI/PMC-hosted paper on embedded YOLO variants (§1.3) could only be reviewed via their
  abstracts/introductions in this survey because their publishers block automated PDF
  downloads; a full read of these three papers would require manually downloading them
  through a browser (see `sources.md` §9 for the exact links and reasoning).

## Folder contents summary

```
literature survey/
├── sources.md          <- full source log (this survey's paper trail)
├── findings.md          <- this document
├── papers/               <- 12 downloaded PDFs (see sources.md §1-6 for per-file details)
└── images/               <- 3 downloaded, openly licensed images (see sources.md §7)
```
