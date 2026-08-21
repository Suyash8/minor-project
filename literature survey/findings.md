# Findings - Object Detection for Autonomous Vehicles (Software Scope)

This document synthesizes the literature survey recorded in `sources.md`. All claims
below are traceable to a specific source listed there; where I generalize across sources,
the relevant section numbers from `sources.md` are cited in parentheses.

**Scope note:** this survey is deliberately limited to the **software** side of AV object
detection - algorithms, network architectures, training strategies, datasets, and
software-level optimization. Hardware concerns (physical sensor units and their mounting,
embedded board/accelerator selection, sensor cost) are explicitly out of scope. Where a
surveyed source also covers hardware, only its software-relevant findings are used here,
and that boundary is called out at the point of use.

## Scope and narrowing

"Object detection for autonomous vehicles" is too broad to survey exhaustively, so this
survey was narrowed to six concrete software use cases that together cover the detection
pipeline from input data to a deployable model:

1. **2D camera-based real-time detection** - the baseline problem: find and classify cars,
   pedestrians, cyclists, signs, etc. in a single camera frame fast enough to drive on.
2. **3D detection from point-cloud data** - recovering depth and precise 3D location/size,
   which 2D image detection alone cannot provide. Treated here as a data-representation
   and network-architecture problem, not a LiDAR hardware problem.
3. **Multi-modal fusion** - the software architecture question of how and where to combine
   heterogeneous input streams (image data + point-cloud data) inside a detection network.
4. **Robustness under adverse weather** - rain, fog, night, snow - conditions where clean-
   weather-trained detectors degrade, and the preprocessing/training-side techniques used
   to counter that.
5. **Vulnerable road user (VRU) / pedestrian detection** - treated as a distinct sub-problem
   because of small object size, occlusion, and behavioral unpredictability.
6. **Model efficiency and software-level optimization** - architectural compression
   (lightweight backbones, separable convolutions, attention modules, multi-task weight
   sharing) that makes real-time inference achievable, independent of which hardware it
   eventually runs on.

Benchmark datasets (KITTI, nuScenes) are treated as a cross-cutting seventh area since
almost every paper above is evaluated against them, and datasets are themselves software
artifacts that shape the field.

## Development timeline observed across sources

Reconstructing publication order from the sources gathered:

- **2012** - The KITTI benchmark suite (sources.md §4.1) establishes the first
  standardized, multi-task (2D/3D detection, stereo, flow, SLAM) driving benchmark,
  creating a common yardstick that makes cross-method comparison possible at all.
- **2016** - SSD (§1.1) demonstrates that single-stage detectors can match two-stage
  accuracy at real-time speed, a prerequisite for any onboard use.
- **2017-2019** - VoxelNet (§2.1) then PointPillars (§2.2) move 3D detection from
  hand-crafted point-cloud features to end-to-end learned representations, with
  PointPillars in particular reaching a speed/accuracy point usable in real time.
- **2019** - nuScenes (§3.2) generalizes the benchmark idea from KITTI to a much larger,
  fully multi-modal annotated dataset across 1,000 scenes, becoming the standard testbed
  for fusion research; VRU-specific surveys (§6.1, §6.2) formalize pedestrian/cyclist
  detection as its own sub-field with intent estimation as the natural next step.
- **2020-2022** - Adverse-weather robustness becomes a dedicated research thread (rain
  review, §5.1); YOLOP (§1.2) and MobileYOLO (§1.3) show the shift toward multi-task
  networks and architecturally compressed detectors respectively; the multi-modal fusion
  taxonomy survey (§3.1) organizes a fast-growing fusion literature.
- **2021-2024** - The 3D detection survey (§2.3) takes stock of the field and explicitly
  adds runtime and robustness analysis as evaluation axes alongside accuracy;
  adverse-weather work matures from "note the problem" to "engineer a fix" (YOLOv8 +
  merged-dataset fine-tuning, §5.3); systematic benchmarking of detector families under
  constrained inference budgets formalizes the accuracy/latency trade-off (§1.4).

Read together, the field's software trajectory is: **establish a benchmark → get
single-stage 2D detection fast enough → extend to 3D via learned point-cloud encodings →
fuse modalities inside the network → confront real-world robustness gaps (weather, VRUs) →
compress the resulting models for real-time inference.** Each stage is a precondition for
the next, not a random research trend - weather robustness or model compression are not
productively studyable for a detector that isn't first real-time-capable and
multi-modal-aware.

## Findings by use case

### 1. 2D real-time detection
One-stage detectors (SSD, YOLO family) dominate AV perception because AV detection is
latency-critical: a two-stage detector's separate region-proposal step adds sequential
computation that one-stage detectors avoid by predicting boxes and classes directly from
feature maps in a single pass (§1.1; see also the architecture diagram in §7.1). The
trend within one-stage detection is toward *multi-task* single networks - YOLOP (§1.2)
performs detection + drivable-area segmentation + lane detection together from one shared
encoder - because an AV needs all three outputs anyway, and sharing an encoder across
tasks is computationally cheaper than running three separate networks. Multi-task weight
sharing is therefore both an accuracy story and an efficiency story.

### 2. 3D detection from point-cloud data
2D image detection cannot recover how far away an object is or its true 3D extent, which
is why 3D detection is treated as a separate, necessary capability rather than a
refinement of 2D detection (§2.3). The dominant software question in this area is how to
encode sparse, irregular point-cloud data into a form a standard CNN can process
efficiently. VoxelNet's learned 3D voxel encoding (§2.1) was the first end-to-end answer,
replacing hand-engineered features; PointPillars' vertical-column ("pillar") encoding
(§2.2) then showed that projecting to a 2D pseudo-image lets you drop expensive 3D
convolutions entirely while retaining strong accuracy. That representation choice - voxel
vs. pillar vs. raw-point - is the main axis along which 3D detection architectures differ,
and is why PointPillars remains a standard baseline in later benchmarks.

### 3. Multi-modal fusion
Image data and point-cloud data have complementary weaknesses: image features carry rich
semantics but no reliable depth and degrade in poor illumination; point-cloud data carries
precise geometry but is sparse and semantically thin. The key organizing question in the
fusion literature is not *which* modalities to combine but *at what stage of the network*
to combine them - the fusion survey (§3.1) proposes a taxonomy built specifically on
fusion stage (raw-data/early, feature-level, proposal/decision-level, and hybrids),
because that architectural decision is what actually determines a fusion model's
accuracy/latency/robustness trade-offs. nuScenes (§3.2) is the standard testbed here,
providing large-scale synchronized multi-modal data with 3D boxes across 23 classes and
roughly 7x KITTI's annotation volume.

### 4. Adverse weather robustness
Clean-weather-trained detectors reliably lose accuracy in rain, fog, snow, and low light
(§5.1, §5.2); the literature treats this as a well-documented, expected failure mode
rather than an edge case. Two software mitigation strategies recur: (a) insert an
image-enhancement stage before detection (deraining, contrast restoration, domain
adaptation, or image-to-image translation, per §5.1), or (b) change the training data
rather than the architecture - fine-tune the detector on weather-degraded imagery, which
the YOLOv8 + ACDC/DAWN dataset-merging study (§5.3) shows produces steady accuracy gains
as more weather-diverse data is merged in. The notable finding is that **dataset diversity
is currently a more reliable lever than architectural cleverness** for this specific
problem, which also means performance is bounded by how well the training data represents
deployment conditions.

### 5. VRU / pedestrian detection
Pedestrians and cyclists are harder to detect reliably than vehicles because they are
smaller in the image, more deformable, more often partially occluded, and behaviorally
less predictable (§6.1). Because a bounding box alone doesn't tell an AV whether a
pedestrian is about to step into the road, the field has moved toward composing detection
with tracking, motion modeling, and pose-based intent estimation (§6.2) - detection is
increasingly framed as the first stage of a "detect → track → predict intent" software
pipeline rather than a standalone deliverable.

### 6. Model efficiency and software-level optimization
Every architectural gain above is bounded by the inference budget available in a vehicle,
which makes model compression a first-class software concern rather than an afterthought.
MobileYOLO (§1.3) is the clearest worked example in this survey: swapping YOLOv4's
backbone for MobileNetv2, replacing standard convolutions with depthwise-separable
convolutions in the feature-fusion path and detection head, and adding lightweight channel
attention (ECA) cut the parameter count by over 50M and reduced model size to roughly a
fifth while increasing detection speed by about 70% on KITTI - at 90.7% reported accuracy.
The general recipe visible across this literature is therefore: **lighter backbone +
separable convolutions + lightweight attention + multi-task weight sharing.**

Comparative benchmarking of detector families under constrained inference budgets (§1.4)
reinforces the trade-off at the model level: lightweight SSD-MobileNet variants achieve
the lowest latency but the lowest accuracy, while larger YOLOv8 variants achieve the
highest accuracy at proportionally higher computational cost - i.e. there is no free
lunch, and model selection is an explicit accuracy-vs-latency decision. That study also
compares specific hardware platforms and accelerators; **that hardware dimension is out of
scope for this survey** and is not relied on here. Its software-relevant finding that does
carry over: the accuracy gap between light and heavy models widens as scene complexity
(object count) grows, so lightweight models degrade disproportionately in dense scenes.

## Cross-cutting observations

- **Benchmarks and datasets shape the field as much as architectures do.** KITTI (§4.1)
  and nuScenes (§3.2) are referenced by nearly every other paper surveyed, and a new
  method is judged largely by how it scores on them. KITTI's narrowness - one city,
  daytime, clear weather - is a direct cause of why adverse-weather and VRU research had
  to build or adopt additional datasets (ACDC, DAWN, per §5.3) rather than relying on it.
  Dataset choice is a research design decision, not a neutral detail.
- **Latency is reported as a first-class result, not an afterthought.** Across all six use
  cases, nearly every paper surveyed reports a speed metric (FPS or ms) alongside accuracy;
  accuracy without a corresponding latency figure is treated as an incomplete result in
  this domain.
- **Plain single-task detection is increasingly composed into larger pipelines.** YOLOP
  folds detection into a shared multi-task network (§1.2); VRU work folds it into a
  detect+track+predict-intent pipeline (§6.2); adverse-weather work folds it into an
  enhance-then-detect or domain-adapt pipeline (§5.1, §5.3). Current research mostly
  composes detection with something else rather than pushing isolated single-frame,
  single-task accuracy further.
- **Representation choice recurs as the central design lever.** Whether it's one-stage vs.
  two-stage heads (§1.1), voxel vs. pillar point-cloud encoding (§2.1, §2.2), or fusion
  stage (§3.1), the most consequential decisions in this literature are about how data is
  represented inside the network rather than about raw model scale.

## Gaps and open challenges noted in the surveyed literature

- Occlusion and small/distant-object detection remain open problems, most acutely for VRUs
  (§6.1).
- Weather robustness is largely dataset- and fine-tuning-driven rather than architecturally
  solved, so performance is capped by training-data representativeness (§5.1, §5.3).
- Detection accuracy still degrades measurably as scene complexity (object count) rises,
  with lightweight models affected disproportionately (§1.4) - a specific concern for
  dense urban scenes.
- Three sources could only be reviewed via their abstracts/introductions rather than full
  text, because their publishers (MDPI, PMC) block automated PDF downloads: the
  adverse-weather review (§5.2), the YOLOv8 adverse-weather study (§5.3), and the
  MobileYOLO paper (§1.3). Claims drawn from these three are limited to what those
  abstracts/introductions state. A full read requires manually downloading them through a
  browser (see `sources.md` §9 for links and details).

## Folder contents summary

```
literature survey/
├── sources.md            <- full source log (this survey's paper trail)
├── findings.md           <- this document
├── papers/               <- 12 downloaded PDFs (see sources.md §1-6 for per-file details)
└── images/               <- downloaded, openly licensed images (see sources.md §7)
```
