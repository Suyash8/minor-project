# Sources Log — Literature Survey on Object Detection for Autonomous Vehicles

This document records every paper, article, dataset page, and image consulted during this
literature survey. For each source it lists: the link, whether/where it was downloaded,
the saved filename, what the source contains, and what was learned from it (in my own
words, not verbatim reproduction). Papers are grouped by the use case they were researched
under (see `findings.md` for the narrowed scope and synthesis).

**Scope note:** the survey is scoped to the **software** side of AV object detection -
algorithms, architectures, training strategies, datasets, and software-level optimization.
Hardware topics (physical sensor units and mounting, embedded board/accelerator selection)
are out of scope. Source summaries below still describe each source's full contents
accurately, including any hardware material it covers, but only software-relevant findings
are carried into `findings.md`; where a source straddles the boundary that is flagged in
its entry.

Research method: search engine queries were run per use-case area (2D detection, 3D/LiDAR
detection, sensor fusion, adverse weather, VRU/pedestrian detection, edge deployment,
benchmark datasets). Candidate papers were shortlisted from arXiv, MDPI, PMC, and
Wikimedia Commons. Open-access PDFs were downloaded directly into `papers/`. Where a
publisher blocked automated downloads, the abstract/introduction was read via web fetch
instead, and that limitation is noted explicitly below rather than glossed over.

---

## 1. Foundational 2D object detectors (real-time single-stage detection)

### 1.1 SSD: Single Shot MultiBox Detector
- Link: https://arxiv.org/abs/1512.02325 (PDF: https://arxiv.org/pdf/1512.02325v3.pdf)
- Downloaded as: `papers/SSD_Liu2016_arxiv1512.02325.pdf` (2.5 MB, verified genuine PDF)
- Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
  Cheng-Yang Fu, Alexander C. Berg (UNC Chapel Hill / Google / Univ. of Michigan), ECCV 2016.
- Contents: Introduces SSD, a single-pass CNN detector that discretizes bounding-box
  predictions into a fixed set of default boxes at multiple feature-map scales, avoiding
  the region-proposal stage used by two-stage detectors. Reports 74.3% mAP at 59 FPS on
  VOC2007 with 300x300 input on a Titan X.
- Takeaway: SSD is one of the two foundational lineages (alongside YOLO) that made
  single-stage, real-time detection viable — directly relevant to why AV perception
  stacks favor one-stage detectors over slower two-stage R-CNN variants.

### 1.2 YOLOP: You Only Look Once for Panoptic Driving Perception
- Link: https://arxiv.org/abs/2108.11250 (PDF: https://arxiv.org/pdf/2108.11250.pdf)
- Downloaded as: `papers/YOLOP_Wu2021_arxiv2108.11250.pdf` (940 KB, verified genuine PDF)
- Authors: Dong Wu, Manwen Liao, Weitian Zhang, Xinggang Wang, et al. Published in
  Machine Intelligence Research, 2022 (arXiv v7 is the version matching the journal text).
- Contents: A single encoder / three-decoder network that performs traffic object
  detection, drivable-area segmentation, and lane detection simultaneously in one forward
  pass. Demonstrated at 23 FPS on an embedded Jetson TX2 while matching or beating
  single-task state of the art on the BDD100K dataset.
- Takeaway: Shows the trend of driving perception moving from "detect objects only" to
  multi-task panoptic perception on embedded hardware — an important development for the
  "use cases" narrowing (perception is rarely detection-only in production AV stacks).

### 1.3 MobileYOLO: Real-Time Object Detection Algorithm in Autonomous Driving Scenarios
- Link (journal): https://www.mdpi.com/1424-8220/22/9/3349
- Link (PMC mirror): https://pmc.ncbi.nlm.nih.gov/articles/PMC9100546/
- Download status: NOT downloaded as a file. Both the MDPI direct-PDF link and the PMC
  PDF endpoint returned bot-protection pages instead of the PDF (MDPI returned an Akamai
  "Access Denied" HTML page; PMC returned a proof-of-work JS challenge page) when fetched
  programmatically. Content below was read via the web-fetch tool against the PMC HTML
  article page instead.
- Authors: Yan Zhou, Sijie Wen, Delong Wang, Jinshan Meng, et al., Sensors 22(9):3349, 2022.
- Contents: Proposes MobileYOLO, a YOLOv4 variant that swaps in a MobileNetv2 backbone,
  replaces standard convolutions with depthwise-separable convolutions in PANet and the
  detection head, and adds an Efficient Channel Attention (ECA) module plus an SSH context
  module for small-object detection. Reports 90.7% accuracy on KITTI, a 52.11M parameter
  reduction vs YOLOv4, model size cut to one-fifth, and a 70% increase in detection speed.
- Takeaway: A concrete example of the accuracy/speed/model-size trade-off work needed to
  get YOLO-class detectors running on in-vehicle compute — feeds into the edge-deployment
  use case.

### 1.4 A Comprehensive Evaluation of Deep Learning Object Detection Models on Heterogeneous Edge Devices
- Link: https://arxiv.org/abs/2409.16808 (PDF: https://arxiv.org/pdf/2409.16808v3.pdf)
- Downloaded as: `papers/EdgeDeviceBenchmark_Alqahtani2024_arxiv2409.16808.pdf` (5.5 MB,
  verified genuine PDF)
- Authors: Daghash K. Alqahtani, Muhammad Aamir Cheema, Maria A. Rodriguez, Adel N. Toosi.
- Contents: Benchmarks YOLOv8 (Nano/Small/Medium), EfficientDet Lite (Lite0-2), and SSD
  (MobileNet V1, MobileDet) across Raspberry Pi 3/4/5 (with/without Coral TPU and AI HAT+),
  Jetson Nano, and Jetson Orin Nano, measuring energy use, inference time, and accuracy,
  including how accuracy degrades as scene object-count grows.
- Takeaway (software-relevant, used in findings): lightweight SSD-MobileNet variants are
  fastest but least accurate while larger YOLOv8 variants are most accurate at higher
  compute cost, and the accuracy gap between light and heavy models widens as scene
  object-count rises — so model-family selection is an explicit accuracy-vs-latency
  decision, and lightweight models degrade disproportionately in dense scenes.
- **Scope flag:** this paper also ranks specific hardware platforms and accelerators
  (Raspberry Pi variants, Coral TPU, AI HAT+, Jetson Nano / Orin Nano). That hardware
  comparison is **out of scope** for this survey and is deliberately not carried into
  `findings.md`; only the model-level trade-offs above are used.

---

## 2. 3D / LiDAR point-cloud object detection

### 2.1 VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
- Link: https://arxiv.org/abs/1711.06396 (PDF: https://arxiv.org/pdf/1711.06396.pdf)
- Downloaded as: `papers/VoxelNet_Zhou2017_arxiv1711.06396.pdf` (12 MB, verified genuine PDF)
- Authors: Yin Zhou, Oncel Tuzel (Apple), CVPR 2018.
- Contents: Divides a raw LiDAR point cloud into equally spaced 3D voxels, encodes the
  points inside each voxel with a novel Voxel Feature Encoding (VFE) layer, then applies
  3D convolution and a region proposal network to output 3D bounding boxes — the first
  major end-to-end trainable 3D detector that skips hand-crafted point-cloud features such
  as bird's-eye-view projections.
- Takeaway: This is the architectural ancestor of nearly all modern voxel-based LiDAR
  detectors; understanding it is necessary before PointPillars makes sense.

### 2.2 PointPillars: Fast Encoders for Object Detection from Point Clouds
- Link: https://arxiv.org/abs/1812.05784 (PDF: https://arxiv.org/pdf/1812.05784v2.pdf)
- Downloaded as: `papers/PointPillars_Lang2019_arxiv1812.05784.pdf` (5.5 MB, verified
  genuine PDF)
- Authors: Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar
  Beijbom (nuTonomy/Aptiv), CVPR 2019.
- Contents: Organizes the point cloud into vertical columns ("pillars") instead of 3D
  voxels, encodes each pillar with a PointNet-style network to build a 2D pseudo-image,
  then runs a standard 2D CNN detection head on it. This avoids expensive 3D convolutions.
- Takeaway: PointPillars is the accuracy/speed sweet spot that made LiDAR-only 3D detection
  practical for real-time onboard use; it is one of the most-cited baselines in every
  later 3D-detection benchmark I found, including the KITTI/nuScenes/Waymo leaderboards.

### 2.3 3D Object Detection for Autonomous Driving: A Survey
- Link: https://arxiv.org/abs/2106.10823 (PDF: https://arxiv.org/pdf/2106.10823v2.pdf)
- Downloaded as: `papers/3DObjectDetectionSurvey_Qian2021_arxiv2106.10823.pdf` (4.2 MB,
  verified genuine PDF)
- Authors: Rui Qian, Xin Lai, Xirong Li (Renmin University of China). Accepted to Pattern
  Recognition, May 2022.
- Contents: A comprehensive survey covering sensors (camera/LiDAR/radar), datasets,
  metrics, and state-of-the-art 3D detection methods (image-based, point-cloud-based, and
  multi-modal fusion-based), plus a case study on 15 representative methods with runtime,
  error, and robustness analysis.
- Takeaway: Used as the map of the 3D-detection landscape — it is what let me confirm that
  the field cleanly splits into monocular/stereo, LiDAR-only (voxel vs. pillar vs.
  point-based), and fusion-based approaches, and that runtime/robustness (not just mAP) is
  now treated as a first-class evaluation axis.

---

## 3. Multi-sensor fusion (camera + LiDAR + radar)

### 3.1 Multi-modal Sensor Fusion for Auto Driving Perception: A Survey
- Link: https://arxiv.org/abs/2202.02703 (PDF: http://arxiv.org/pdf/2202.02703v2.pdf)
- Downloaded as: `papers/MultiModalSensorFusionSurvey_Cui2022_arxiv2202.02703.pdf` (2.9 MB,
  verified genuine PDF)
- Authors: Keli Huang, Botian Shi, Xiang Li, Xin Li, Siyuan Huang, Yikang Li (Shanghai AI
  Laboratory et al.).
- Contents: Reviews over 50 papers on camera+LiDAR fusion for object detection and
  semantic segmentation. Proposes a new taxonomy that splits fusion methods into two major
  classes and four minor classes based on *where* in the pipeline fusion happens
  (early/data-level, feature-level, late/decision-level, and variants), rather than the
  traditional early/late split.
- Takeaway: Gave me the fusion-stage taxonomy used in `findings.md` — the "where you fuse"
  question (raw data vs. features vs. proposals vs. final boxes) is the central design
  decision that separates fusion architectures, more so than which sensors are combined.

### 3.2 nuScenes: A Multimodal Dataset for Autonomous Driving
- Link: https://arxiv.org/abs/1903.11027 (PDF: https://arxiv.org/pdf/1903.11027.pdf)
- Downloaded as: `papers/nuScenes_Caesar2019_arxiv1903.11027.pdf` (4.8 MB, verified genuine
  PDF)
- Authors: Holger Caesar, Varun Bankiti, Alex H. Lang, et al. (nuTonomy/Aptiv), CVPR 2020.
- Contents: Introduces the nuScenes dataset: 1000 driving scenes of 20 seconds each, with
  full sensor suite (6 cameras, 1 LiDAR, 5 radars, GPS, IMU), 3D bounding boxes for 23
  object classes and 8 attributes, and novel 3D detection/tracking metrics. Roughly 7x the
  annotations and 100x the images of KITTI.
- Takeaway: This is the dataset that made large-scale, multi-sensor (not just
  camera+LiDAR, but also radar) benchmarking standard in the field, and it underlies most
  of the fusion and 3D-detection papers surveyed here.

---

## 4. Benchmark datasets for autonomous-driving perception

### 4.1 Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite
- Link: https://www.cvlibs.net/publications/Geiger2012CVPR.pdf (author's own site, CVPR 2012)
- Downloaded as: `papers/KITTI_Geiger2012CVPR.pdf` (875 KB, verified genuine PDF, 8 pages)
- Authors: Andreas Geiger, Philip Lenz, Raquel Urtasun (Karlsruhe Institute of
  Technology / Toyota Research Institute).
- Contents: Introduces the KITTI benchmark suite — stereo, optical flow, visual
  odometry/SLAM, and 2D/3D object detection benchmarks recorded from a car-mounted
  camera+LiDAR rig driving around Karlsruhe, Germany. Includes 7,481 training / 7,518 test
  images with over 80,000 labeled objects for the detection task.
- Takeaway: KITTI is the original standardized benchmark that every detector in this
  survey (SSD-derivatives, MobileYOLO, VoxelNet, PointPillars) is evaluated against; it's
  the common yardstick that makes cross-paper comparison possible, though its scale (one
  city, daytime, clear weather) is now recognized as a limitation compared to nuScenes/Waymo.

---

## 5. Robustness under adverse weather / low visibility

### 5.1 Object Detection Under Rainy Conditions for Autonomous Vehicles: A Review of
State-of-the-Art and Emerging Techniques
- Link: https://arxiv.org/abs/2006.16471 (PDF: https://arxiv.org/pdf/2006.16471v4.pdf)
- Downloaded as: `papers/RainyOD_Hnewa2021_arxiv2006.16471.pdf` (9.9 MB, verified genuine
  PDF)
- Authors: Mazin Hnewa, Hayder Radha (Michigan State University). Published in IEEE Signal
  Processing Magazine, vol. 38, no. 1, pp. 53-67, Jan. 2021.
- Contents: A tutorial-style review of how rain degrades object-detection performance, and
  a survey of image-deraining methods, deep-learning domain adaptation, and image-to-image
  translation approaches proposed to counter it, with experimental comparisons across
  clear vs. rainy visual data.
- Takeaway: Establishes that rain (and adverse weather generally) is not a minor edge case
  but a well-studied, significant failure mode for camera-based detectors, and that the
  main mitigation strategies are (a) pre-process/enhance the image before detection, or
  (b) domain-adapt/retrain the detector directly on weather-degraded data.

### 5.2 Object Detection in Autonomous Vehicles under Adverse Weather: A Review of
Traditional and Deep Learning Approaches
- Link: https://www.mdpi.com/1999-4893/17/3/103
- Download status: NOT downloaded. The MDPI `/pdf` endpoint and a `staging.core.mdpi.com`
  mirror both returned bot-protection HTML stubs (Access Denied) instead of a PDF when
  fetched programmatically. Reviewed via the article's indexed abstract/snippet content
  instead (Algorithms 2024, 17(3), 103).
- Contents (from abstract/metadata): Surveys both traditional computer-vision and deep
  learning approaches for detecting vehicles, pedestrians, and road lanes specifically
  under adverse weather, connecting classical techniques to the modern deep-learning
  literature.
- Takeaway: Reinforces that adverse-weather detection research spans a spectrum from
  classical image processing (contrast/visibility restoration) through to modern deep
  detectors, and that most reviews treat vehicles/pedestrians/lanes as the three
  minimum-viable object classes an AV must keep detecting reliably regardless of weather.

### 5.3 Object Detection in Adverse Weather for Autonomous Driving through Data Merging
and YOLOv8
- Link: https://www.mdpi.com/1424-8220/23/20/8471 / PMC mirror:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10611033/
- Download status: NOT downloaded, same reason as above (MDPI PDF endpoint blocked;
  PMC PDF endpoint served a JS proof-of-work challenge page rather than a file). Reviewed
  via the PMC HTML article page (abstract + introduction) through web fetch.
- Authors: Debasis Kumar, Naveed Muhammad (University of Tartu), Sensors 23(20):8471, 2023.
- Contents: Fine-tunes YOLOv8 via transfer learning on two open adverse-weather datasets,
  ACDC (fog/rain/snow/night) and DAWN (fog/rain/snow/sand), plus a merged version of both,
  and compares detection performance across the individual and merged training sets.
- Takeaway: Concrete evidence that combining multiple weather-specific datasets during
  fine-tuning steadily improves detection performance over the stock YOLOv8 weights —
  supports the "dataset diversity beats architecture cleverness" trend also visible in
  the robustness survey above.

---

## 6. Pedestrian / vulnerable road user (VRU) detection

### 6.1 Vulnerable Road User Detection: State-of-the-Art and Open Challenges
- Link: https://arxiv.org/abs/1902.03601 (PDF: http://arxiv.org/pdf/1902.03601v1.pdf)
- Downloaded as: `papers/VRU_Detection_Rasouli2019_arxiv1902.03601.pdf` (833 KB, verified
  genuine PDF)
- Author: Amir Rasouli (submitted via Patrick Mannion), 2019.
- Contents: Surveys VRU (pedestrian and cyclist) detection specifically, covering
  benchmarks/datasets, detection techniques, and relevant machine-learning algorithms, and
  closes with open challenges (occlusion, small-scale objects, nighttime, and behavior
  unpredictability).
- Takeaway: Confirms VRU detection is treated as a distinct sub-problem from general
  object detection because pedestrians/cyclists are smaller, more deformable, more
  frequently occluded, and behaviorally less predictable than vehicles — motivating the
  intent-estimation work below.

### 6.2 Pedestrian and Cyclist Detection and Intent Estimation for Autonomous Vehicles
- Link: https://www.mdpi.com/2076-3417/9/11/2335 (author repository copy used for download)
- Downloaded as: `papers/PedestrianCyclistIntent_Ahmed2019_strathprints.pdf` (1.2 MB,
  verified genuine PDF, retrieved from the University of Strathclyde's open-access
  repository at
  https://strathprints.strath.ac.uk/87232/1/Ahmed-etal-AS-2019-Pedestrian-and-cyclist-detection-and-intent-estimation-for-autonomous-vehicles.pdf)
- Authors: Sarfraz Ahmed, M. Nazmul Huda, Sajjad Rajbhandari, Chitta Saha, Maysam Elshaw,
  Stratis Kanarachos. Applied Sciences 9(11):2335, 2019.
- Contents: Surveys deep-learning methods for pedestrian and cyclist detection and argues
  for combining detection with tracking, motion modeling, and pose estimation to estimate
  a VRU's *intent* (e.g., about to cross) rather than only their current position.
- Takeaway: Detection alone is not the end goal for VRU safety — the field is moving
  toward "detect + track + predict intent" pipelines, which reframes plain object
  detection as the first stage of a larger behavior-prediction problem.

---

## 7. Images consulted and downloaded

All images below were sourced from Wikimedia Commons and verified to carry an open license
(CC BY 4.0 or CC0) before download, with attribution captured here as required by the
license terms.

Because this survey is scoped to software only (see the scope note at the top of
`findings.md`), the software-relevant image is listed first as §7.1 and is the only one
cited in the findings. The two sensor-hardware photographs originally gathered are
retained below under §7.2 for provenance and completeness, but are marked out of scope and
are not relied on for any claim in `findings.md`.

### 7.1 One-stage vs two-stage object detector architecture diagram
- Link: https://commons.wikimedia.org/wiki/File:Object_detector_1stage_vs_2_stage.png
- Downloaded as: `images/ObjectDetector_1stage_vs_2stage_CC-BY4.0.png` (110 KB)
- License: CC BY 4.0. Credited to Licheng Jiao, Fan Zhang, Fang Liu, Shuyuan Yang, sourced
  from their paper "A Survey of Deep Learning-Based Object Detection" (IEEE Access, 2019,
  DOI: 10.1109/ACCESS.2019.2939201). Attribution required.
- Contents: Schematic diagram contrasting the general architecture of a two-stage detector
  (e.g., Faster R-CNN: region-proposal network followed by classification/regression) with
  a one-stage detector (e.g., SSD/YOLO: direct dense prediction from feature maps).
- Takeaway: A clean visual explanation of exactly why one-stage detectors (SSD, YOLO
  family) are preferred for AV real-time use over two-stage detectors — fewer sequential
  steps means lower latency, at some accuracy cost that later architectures try to recover.

### 7.2 Sensor-hardware photographs (OUT OF SCOPE — retained for provenance only)

These two images were downloaded earlier, before the survey was narrowed to software only.
They document physical sensor hardware, which is outside the current scope, so they are
**not cited anywhere in `findings.md`**. They are left in `images/` and logged here rather
than silently deleted, so the record of what was downloaded stays accurate. They can be
removed if a strictly software-only artifact set is wanted.

- **Zoox AV sensor close-up** — `images/Zoox_AV_sensor_closeup_CC-BY4.0.jpg` (6.3 MB).
  Link: https://commons.wikimedia.org/wiki/File:Zoox_Toyota_Highlander_Test_Vehicle_-_Sensor_Closeup_-_San_Francisco,_May_2025_07.jpg
  License: CC BY 4.0, author Wikimedia Commons user "9yz", attribution required.
  Contents: close-up of the camera and LiDAR sensor cluster on a Toyota Highlander used as
  a Zoox self-driving test vehicle. Out of scope: sensor hardware, not software.
- **Yandex AV roof LiDAR unit** — `images/Yandex_AV_roof_lidar_unit_CC0.jpg` (338 KB).
  Link: https://commons.wikimedia.org/wiki/File:Moscow,_Yandex_self-driving_Hyundai_Sonata,_Aug_2025_roof_unit_01.jpg
  License: CC0 / Public Domain Dedication, author Wikimedia Commons user "Retired
  electrician"; credited here as good practice though not legally required.
  Contents: roof-mounted LiDAR/sensor unit on a Yandex self-driving Hyundai Sonata in
  Moscow. Out of scope: sensor hardware, not software.

---

## 8. Notes on sources considered but not used in depth

The following were surfaced during search but only skimmed at the snippet/abstract level
because they were redundant with a source already logged above, or because they fell
outside the six narrowed use cases (see `findings.md` section "Scope and narrowing"):
- "A Review of Multi-Sensor Fusion in Autonomous Driving" (MDPI Sensors 25(19):6033) —
  redundant with the Cui/Huang et al. fusion survey already logged in 3.1.
- "Multi-Modal 3D Object Detection in Autonomous Driving: A Survey" (Springer IJCV, 2023) —
  redundant with the Qian et al. 3D survey already logged in 2.3.
- "A Survey of Deep Learning-Based Radar and Vision Fusion" (arXiv 2406.00714) — noted as
  evidence that radar-camera fusion is an active sub-area, but not read in full since
  camera+LiDAR fusion was chosen as the representative fusion use case for this survey.
- Pedestrian intent papers beyond 6.2 (e.g., "On-Board Detection of Pedestrian Intentions",
  PMC5676781) — noted as further reading but out of scope since intent *prediction* is a
  downstream task from detection, not detection itself.

## 9. A note on access restrictions encountered

Several MDPI-hosted PDFs and PMC (PubMed Central) PDF endpoints actively block
non-browser HTTP clients: MDPI returned Akamai "Access Denied" pages, and PMC returned a
JavaScript proof-of-work challenge page ("Preparing to download..."), for every
programmatic PDF request attempted against them in this session. This is a bot-detection
measure on their end, not a broken link. In every such case the paper's abstract and
introduction were still readable and were reviewed via the publisher's HTML article page,
and that is reflected honestly above rather than presenting these as full downloads. If a
local copy of these three papers is needed, they should be downloaded manually through a
browser from the links given in sections 1.3, 5.2, and 5.3.

---

## 10. Oriented Object Detection (OOD) in Aerial and Remote Sensing Imagery

### 10.1 DOTA: A Large-scale Dataset for Object Detection in Aerial Images
- Link: https://arxiv.org/abs/1711.10398 (PDF: https://openaccess.thecvf.com/content_cvpr_2018/papers/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.pdf)
- Downloaded as: `papers/DOTA_Xia2018_CVPR.pdf` (1.3 MB, verified genuine PDF, 10 pages)
- Authors: Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, Liangpei Zhang, CVPR 2018.
- Contents: Introduces DOTA, a benchmark dataset specifically designed for object detection in aerial scenes with arbitrary orientations. Contains 2,806 large aerial images (approx. 4000x4000 pixels) annotated with 188,282 instances across 15 common categories using 8-dof oriented bounding boxes (quadrilaterals). Provides baseline evaluations of horizontal vs. oriented detectors.
- Takeaway: Foundational dataset and evaluation protocol that established oriented bounding box (OBB) benchmark standards across earth vision and remote sensing.

### 10.2 Learning RoI Transformer for Oriented Object Detection in Aerial Images
- Link: https://arxiv.org/abs/1812.00155 (PDF: https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.pdf)
- Downloaded as: `papers/RoITransformer_Ding2019_CVPR.pdf` (1.35 MB, verified genuine PDF, 10 pages)
- Authors: Jian Ding, Nan Xue, Yang Long, Gui-Song Xia, Qikai Lu, CVPR 2019.
- Contents: Proposes RoI Transformer to address mismatch and misalignment between standard horizontal region proposals (HRoIs) and oriented objects. Uses a Rotated RoI (RRoI) learner to transform HRoIs into rotated RoIs and a Rotated Position Sensitive RoI Align (RPS-RoI-Align) module to extract rotation-invariant features without anchor proliferation.
- Takeaway: Landmark architectural contribution demonstrating that learning geometric transformations of region proposals resolves spatial misalignment in two-stage oriented detectors with minimal computation overhead.

### 10.3 Mask OBB: A Semantic Attention-Based Mask Oriented Bounding Box Representation for Multi-Category Object Detection in Aerial Images
- Link: https://doi.org/10.3390/rs11242930 (Archive Mirror: http://web.archive.org/web/20220618194628/https://mdpi-res.com/d_attachment/remotesensing/remotesensing-11-02930/article_deploy/remotesensing-11-02930-v2.pdf?version=1576747117)
- Downloaded as: `papers/MaskOBB_Wang2019_RemoteSensing.pdf` (35.5 MB, verified genuine PDF, 10 pages)
- Authors: Jinwang Wang, Jian Ding, Haowen Guo, Wensheng Cheng, Ting Pan, Wen Yang, Remote Sensing 2019.
- Contents: Formulates oriented object detection as an instance segmentation / pixel-level mask classification task to eliminate the parameter definition ambiguities (angle boundary discontinuities, vertex ordering) inherent in standard regression-based OBB approaches. Employs an Inception Lateral Connection Network (ILCN) for scale variations and a Semantic Attention Network (SAN) to separate objects from cluttered aerial backgrounds.
- Takeaway: Demonstrates that segmentation-guided representation resolves angle periodicity and boundary discontinuities in dense multi-category aerial object detection.

### 10.4 Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors
- Link: https://arxiv.org/abs/2008.07043 (PDF: https://arxiv.org/pdf/2008.07043.pdf)
- Downloaded as: `papers/BBAVectors_Yi2021_arxiv2008.07043.pdf` (6.74 MB, verified genuine PDF, 6 pages)
- Authors: Jingru Yi, Pengxiang Wu, Bo Liu, Qiaoying Huang, Hui Qu, Dimitris Metaxas, WACV 2021.
- Contents: Introduces BBAVectors, an anchor-free keypoint-based oriented object detector. Detects object center points and regresses box boundary-aware vectors across the four quadrants of the Cartesian coordinate plane to reconstruct oriented bounding boxes, eliminating positive/negative anchor imbalances and avoiding direct angle/aspect-ratio regression ambiguities.
- Takeaway: Key one-stage anchor-free approach that proves vector regression from central keypoints is superior to direct $(x, y, w, h, \theta)$ coordinate regression.

### 10.5 Oriented Object Detection in Optical Remote Sensing Images using Deep Learning: A Survey
- Link: https://arxiv.org/abs/2302.10473 (Journal DOI: https://doi.org/10.1007/s10462-025-11256-0)
- Downloaded as: `papers/OOD_Survey_Wang2023_arxiv2302.10473.pdf` (8.35 MB, verified genuine PDF, 46 pages)
- Authors: Kun Wang, Zi Wang, Zhang Li, Ang Su, Xichao Teng, Erting Pan, Minhao Liu, Qifeng Yu, Artificial Intelligence Review 2025 / arXiv:2302.10473v6.
- Contents: Comprehensive 46-page survey tracing the technical evolution from horizontal bounding box (HBB) detection to oriented bounding box (OBB) detection in remote sensing. Categorizes methods across detection frameworks (two-stage, one-stage, anchor-free, DETR/transformer-based), OBB regression schemes, loss functions, and rotation-invariant feature representations, addressing feature misalignment and angle periodicity.
- Takeaway: Definitive and exhaustive modern reference synthesizing architectures, datasets, loss functions, and benchmarks in remote sensing oriented object detection.

### 10.6 A Comprehensive Survey of Oriented Object Detection in Remote Sensing Images
- Link: https://doi.org/10.1016/j.eswa.2023.119960 (ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0957417423004621)
- Download status: NOT downloaded as a file (Publisher paywall). Published under Elsevier (*Expert Systems with Applications*, Vol. 224, Aug 2023, 119960) under subscription access; `is_oa: false`. No open-access preprint exists on arXiv, Research Square, TechRxiv, SSRN, or open university repositories.
- Authors: Long Wen, Yu Cheng, Yi Fang, Xinyu Li.
- Contents: Comprehensive review of deep learning oriented object detection in remote sensing images. Reviews the transition from horizontal to oriented detection, compares anchor-based and anchor-free frameworks, analyzes rotation-sensitive losses, and evaluates performance across public benchmarks.
- Takeaway: Important journal survey covering rotation-invariant modeling and loss formulation. Available via institutional subscription at the DOI link above.

