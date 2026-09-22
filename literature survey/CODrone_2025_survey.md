# Literature Survey: More Clear, More Flexible, More Precise: A Comprehensive Oriented Object Detection Benchmark for UAV (CODrone)

**Document Metadata:**
- **Surveyed Paper:** *More Clear, More Flexible, More Precise: A Comprehensive Oriented Object Detection Benchmark for UAV* (CODrone)
- **Authors:** Kai Ye, Haidi Tang, Bowen Liu, Pingyang Dai, Liujuan Cao, and Rongrong Ji (MAC Laboratory, School of Informatics, Xiamen University; Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education, China; Peng Cheng Laboratory)
- **Publication / Venue:** arXiv preprint (arXiv:2504.20032v1 [cs.CV], April 28, 2025; Under Review / Journal Submission)
- **Official Repository:** [GitHub - AHideoKuzeA/CODrone](https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV)
- **Deliverable Type:** Research-Grade Literature Survey (36-Section Master Analysis)
- **Focus Areas:** Unmanned Aerial Vehicles (UAVs), Oriented Object Detection (OOD), High-Resolution 4K Aerial Imagery, Multi-Altitude / Multi-Tilt Benchmarking, Drone Perception Bottlenecks

---

## 1. Executive Summary & Bibliographic Metadata

*More Clear, More Flexible, More Precise: A Comprehensive Oriented Object Detection Benchmark for UAV* (CODrone), authored by Kai Ye et al. from Xiamen University and Peng Cheng Laboratory (2025), addresses a critical foundational bottleneck in aerial computer vision: existing UAV oriented object detection (OOD) datasets suffer from low image resolutions, constrained category vocabularies (dominated almost exclusively by generic vehicles), rigid nadir-only or single-oblique camera perspectives, and restricted flight altitude ranges. These deficiencies cause algorithms that perform well on benchmark leaderboards (e.g., DOTA, VisDrone) to experience severe degradation when deployed on real-world drones operating across varying flight altitudes and tilt angles.

To eliminate this gap, the authors construct **CODrone**, a comprehensive 4K ultra-high-definition ($3840 \times 2160$) oriented object detection benchmark collected via a DJI Mavic 3 Pro across multiple cities, seasons, and day/night conditions. CODrone comprises **10,004 high-resolution aerial images** and **596,732 manually annotated oriented bounding box (OBB)** instances spanning **12 diverse object categories** (car, truck, traffic-sign, people, motor, bicycle, traffic-light, tricycle, bridge, bus, boat, ship). Uniquely, every image in CODrone is annotated with explicit UAV flight parameters: two camera tilt angles ($30^\circ$ oblique and $90^\circ$ nadir) and three flight altitudes ($30\text{ m}$, $60\text{ m}$, $100\text{ m}$), forming a systematic $2 \times 3$ factorial matrix of six distinct viewpoint configurations.

The authors establish a standardized benchmark by evaluating **22 representative and state-of-the-art oriented object detection algorithms** (spanning two-stage, single-stage, anchor-free, rotation-equivariant, Gaussian metric loss, large-kernel, weakly-supervised, and transformer-based architectures) under standard $AP_{50}$ and fine-grained localization $AP_{75}$ metrics. The empirical findings reveal that while large selective kernel networks (LSKNet, $46.92\%$ $AP_{50}$) and rotation-equivariant networks (ReDet, $44.73\%$ $AP_{50}$, $20.17\%$ $AP_{75}$) lead overall performance, all existing detectors suffer dramatic accuracy drops at higher altitudes ($100\text{ m}$) and oblique viewing angles ($30^\circ$). Most critically, fine-grained localization ($AP_{75}$) collapses across all 22 methods to between $10.29\%$ and $21.15\%$, exposing that high-precision orientation estimation in real-world drone conditions remains an unsolved computer vision challenge.

### Bibliographic Matrix
| Attribute | Detail |
|---|---|
| **Title** | More Clear, More Flexible, More Precise: A Comprehensive Oriented Object Detection Benchmark for UAV |
| **Short Name** | CODrone Benchmark |
| **Authors** | Kai Ye, Haidi Tang, Bowen Liu, Pingyang Dai, Liujuan Cao, Rongrong Ji |
| **Institutions** | School of Informatics, Xiamen University; Peng Cheng Laboratory, Shenzhen, China |
| **Identifier** | arXiv:2504.20032 [cs.CV] (28 Apr 2025) |
| **Code / Data** | `https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV` |
| **Primary Tasks** | UAV-based Oriented Object Detection (OOD), Multi-Altitude Perception, Oblique vs. Nadir Aerial Detection |
| **Key Benchmark Size** | 10,004 4K UHD images ($3840 \times 2160$), 596,732 OBB instances, 12 classes, 22 baseline detectors evaluated |

---

## 2. Problem Statement & Operational Context

Unmanned aerial vehicles (UAVs) operating in smart city management, traffic monitoring, emergency search-and-rescue, agricultural analysis, and logistics surveillance operate in environments fundamentally different from ground-based autonomous vehicles (AVs) or fixed-orbit satellite remote sensing platforms:
1. **Unconstrained 6-DoF Dynamics:** Drones pitch, roll, and yaw dynamically due to aerodynamic flight conditions, wind turbulence, and mission requirements, capturing visual targets from continuously varying viewing angles (from top-down nadir $90^\circ$ to shallow oblique $30^\circ$).
2. **Dynamic Operational Altitudes:** A single drone mission may loiter at $100\text{ m}$ for wide-area spatial coverage, descend to $60\text{ m}$ for intersection surveillance, and descend to $30\text{ m}$ for fine-grained structural inspection or landing verification.
3. **Severe Perspective Foreshortening:** Oblique camera viewpoints induce non-uniform spatial resolution across the field of view; objects in the foreground appear large and geometrically detailed, while objects in the background undergo extreme foreshortening, severe occlusion, and aspect-ratio distortion.
4. **Extreme Object Density & Arbitrary Orientations:** Urban intersections, parking facilities, and pedestrian walkways contain hundreds of tightly packed, arbitrarily rotated instances where conventional horizontal bounding boxes (HBB) overlap excessively and fail to distinguish separate targets.

Existing drone datasets (e.g., VisDrone, UAVDT) rely primarily on horizontal bounding boxes (HBB), which contain high background noise and collapse in dense scenes. Meanwhile, existing oriented bounding box (OBB) datasets (e.g., DOTA, HRSC2016, UCAS-AOD) were collected from high-altitude aircraft or satellites ($>200\text{ m}$), where camera viewpoints are almost exclusively vertical/nadir and scale variations are uniform. Previous low-altitude drone OBB datasets (e.g., DroneVehicle, UAV-ROD) are severely limited in category diversity (1 to 5 classes) or resolution ($840 \times 712$). CODrone is explicitly engineered to solve this operational disconnect.

---

## 3. Core Motivation & Driving Questions

The paper is organized around four core systemic limitations identified in the UAV perception literature:
1. **Limitation 1 — Low Image Resolution:** Early datasets restricted images to $1080\text{p}$ or lower ($840 \times 712$), rendering small targets (pedestrians, bikes, traffic signs) into indistinct pixel clusters ($< 10 \times 10$ pixels) where orientation angle is mathematically unresolvable. Modern drones record 4K UHD video natively.
2. **Limitation 2 — Restricted Object Diversity:** Most aerial datasets restrict annotations to vehicles (cars, buses, trucks). Real-world drone autonomy requires detecting vulnerable road users (pedestrians, motorcyclists, bicyclists), traffic infrastructure (traffic lights, traffic signs, bridges), and waterborne vessels (boats, ships).
3. **Limitation 3 — Fixed Single-View Imaging:** Existing datasets typically fly at a single fixed gimbal angle (usually nadir $90^\circ$ or a single forward pitch), ignoring the dramatic visual domain shift between orthogonal top-down views and perspective-distorted oblique shots.
4. **Limitation 4 — Narrow Altitude Bands:** Datasets collected at a single altitude fail to evaluate whether multi-scale feature pyramids (FPN) can generalize across drastic scale transitions ($30\text{ m}$ close-range vs. $100\text{ m}$ high-altitude reconnaissance).

### Driving Benchmark Questions:
- *RQ1:* How do state-of-the-art OOD architectures perform when evaluated on high-resolution 4K drone imagery across diverse object classes?
- *RQ2:* What is the quantitative performance penalty incurred when shifting from vertical nadir flight ($90^\circ$) to oblique reconnaissance ($30^\circ$)?
- *RQ3:* Does detector degradation across increasing altitude ($30\text{ m} \to 60\text{ m} \to 100\text{ m}$) stem primarily from classification failure or precise bounding box boundary regression failure?
- *RQ4:* Which algorithmic paradigms (anchor-based vs. anchor-free, rotation-equivariant vs. standard CNNs, IoU vs. Gaussian metric losses) exhibit the greatest resilience against drone viewpoint and altitude perturbations?

---

## 4. Dataset Architecture & Sensor Specifications

### Drone Platform & Hardware Specifications
The CODrone dataset was acquired using the commercial off-the-shelf **DJI Mavic 3 Pro** quadcopter UAV platform.
- **Sensor:** Hasselblad L2D-20c 4/3 CMOS primary sensor (effective pixels: 20 MP), paired with dual telephoto sensors.
- **Lens & Optics:** $24\text{ mm}$ equivalent focal length, $f/2.8 - f/11$ adjustable aperture, $84^\circ$ field of view (FOV).
- **Resolution:** Native $3840 \times 2160$ pixels (4K UHD) at 16:9 aspect ratio.
- **Flight Stabilization:** 3-axis mechanical gimbal (tilt, roll, pan) providing angular vibration range of $\pm 0.005^\circ$, ensuring sharp imagery without motion blur.
- **Flight Altitudes (AGL - Above Ground Level):** Strictly logged at $30\text{ m}$, $60\text{ m}$, and $100\text{ m}$.
- **Gimbal Pitch Angles:** Fixed during designated flight runs at $30^\circ$ (oblique forward perspective) and $90^\circ$ (orthogonal top-down nadir perspective).
- **Geographic Coverage:** Captured across multiple metropolitan and suburban regions in China (including Xiamen, Shenzhen, and neighboring urban/industrial corridors), spanning diverse architectures, road typologies, harbors, and bridge crossings.

---

## 5. Ground Truth & Annotation Protocol

### Annotation Format & Representation
Objects are annotated using **Oriented Bounding Boxes (OBB)** defined by five parameters $(x_c, y_c, w, h, \theta)$ as well as explicit four-corner polygon vertices:
$$[(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)]$$
where $(x_1, y_1)$ denotes the starting vertex (top-left in canonical orientation), and vertices are arranged in clockwise sequence. 

The angle $\theta$ conforms to the OpenCV $180^\circ$ definition ($\theta \in [-90^\circ, 90^\circ)$ or $[-\pi/2, \pi/2)$), representing the angle between the horizontal axis and the first edge of the rotated rectangle. Ground truth is provided in both **DOTA-format text files** (polygon format) and **Pascal VOC-style XML** to ensure seamless plug-and-play integration with open-source remote sensing toolboxes (MMRotate, AerialDetection).

### Multi-Stage Quality Assurance Protocol
1. **Initial Pre-Annotation:** High-confidence detections from an ensemble of pre-trained detectors were filtered and used as initial proposals.
2. **Manual Relabeling:** A team of trained professional annotators manually corrected all coordinates, refined boundary alignments, and labeled missed instances using dedicated GIS/OBB annotation software.
3. **Cross-Validation Auditing:** An independent expert quality-assurance team conducted three rounds of visual verification. Annotations with angular deviation $> 5^\circ$ or IoU discrepancy $> 0.05$ against true physical boundaries were sent back for re-annotation.
4. **Difficulty Label Assignment ($D$):**
   - $D = 1$ (Hard sample): Small instances ($< 32 \times 32$ pixels), heavily occluded instances (occlusion ratio $\ge 0.50$), or objects suffering from extreme perspective distortion / motion blur.
   - $D = 0$ (Normal sample): Fully visible, well-defined instances.

---

## 6. Viewpoint, Altitude & Pose Diversity

A distinguishing structural contribution of CODrone is its **pose-aware dataset matrix**. In existing benchmarks, flight parameters are unrecorded nuisance variables. In CODrone, every single image explicitly encodes altitude and camera tilt directly in its file naming convention:
$$\texttt{img\_\{idx\}\_alt\{30\|60\|100\}\_ang\{30\|90\}.jpg}$$

### Factorial Distribution Matrix (Table III)
The 10,004 images are balanced across six distinct flight operational regimes:

| Flight Altitude | Camera Tilt Angle | Viewpoint Designation | Image Count | Percentage (%) |
|---|---|---|---|---|
| **$30\text{ m}$** | $30^\circ$ | Low Altitude, Oblique View | 1,865 | $18.64\%$ |
| **$30\text{ m}$** | $90^\circ$ | Low Altitude, Nadir View | 1,276 | $12.75\%$ |
| **$60\text{ m}$** | $30^\circ$ | Medium Altitude, Oblique View | 2,047 | $20.46\%$ |
| **$60\text{ m}$** | $90^\circ$ | Medium Altitude, Nadir View | 2,269 | $22.68\%$ |
| **$100\text{ m}$** | $30^\circ$ | High Altitude, Oblique View | 1,048 | $10.48\%$ |
| **$100\text{ m}$** | $90^\circ$ | High Altitude, Nadir View | 1,499 | $14.98\%$ |
| **Total** | — | — | **10,004** | **$100.00\%$** |

### Geometric Implications of Viewpoint Configurations
- **$90^\circ$ Nadir View:** Orthogonal projection where object bounding boxes correspond strictly to 2D top-down planforms. Orientation is defined entirely by planar yaw; roll and pitch have negligible projection distortion.
- **$30^\circ$ Oblique View:** Strong perspective projection where side profiles, vertical facades, and shadows become prominent. Foreshortening causes identical vehicles to occupy vastly different pixel footprints depending on whether they are in the near-field or far-field of the frame.

---

## 7. Environmental & Illumination Conditions

CODrone is designed to evaluate round-the-clock all-weather operational readiness for autonomous drones:
- **Illumination Breakdown (Table IV):**
  - **Daytime Images:** 6,121 images ($61.18\%$) captured across sunny, overcast, high-glare, and midday conditions.
  - **Nighttime Images:** 3,883 images ($38.82\%$) captured under streetlighting, low-ambient illumination, and high-contrast artificial point light sources.
- **Atmospheric & Seasonal Diversity:** Multi-season data collection captures seasonal foliage changes, shadow length variations, rainy/wet pavement reflections, and slight haze conditions across coastal and inland urban topologies.
- **Urban Morphology:** Scenes span dense downtown commercial centers, multi-level elevated highway interchanges, residential suburbs, industrial manufacturing parks, maritime docks, waterways, and bridge crossings.

---

## 8. Dataset Scale & Statistical Distribution

CODrone contains **596,732 total annotated instances** across 12 semantic classes plus an explicit "ignored" category for ambiguous objects. The dataset is partitioned into canonical splits: **$50\%$ Training** (5,002 images), **$20\%$ Validation** (2,000 images), and **$30\%$ Testing** (3,002 images). All test annotations are released to guarantee fully reproducible local benchmarking.

### Detailed Class Instance Breakdown (Table II)
| Category Name | Total Instances | Training Split ($50\%$) | Validation Split ($20\%$) | Testing Split ($30\%$) | Frequency Profile |
|---|---|---|---|---|---|
| **Car** | 227,751 | 112,588 | 46,396 | 68,767 | Dominant Head Class ($38.17\%$) |
| **People** | 79,485 | 39,343 | 15,457 | 24,685 | High-Density Small Object ($13.32\%$) |
| **Motor (Motorcycle)** | 73,593 | 36,662 | 14,986 | 21,945 | High-Density Small Object ($12.33\%$) |
| **Truck** | 24,431 | 12,147 | 5,058 | 7,226 | Medium-Aspect Class ($4.09\%$) |
| **Traffic Sign** | 11,797 | 5,706 | 2,425 | 3,666 | Tiny Object Class ($1.98\%$) |
| **Boat** | 7,143 | 3,607 | 1,080 | 2,456 | Maritime Class ($1.20\%$) |
| **Traffic Light** | 6,891 | 3,336 | 1,498 | 2,057 | Tiny Infrastructure ($1.15\%$) |
| **Bus** | 4,979 | 2,444 | 1,021 | 1,514 | Elongated Vehicle ($0.83\%$) |
| **Bicycle** | 3,835 | 1,892 | 813 | 1,130 | Ultra-Small Vulnerable User ($0.64\%$) |
| **Tricycle** | 2,845 | 1,398 | 599 | 848 | Regional Urban Class ($0.48\%$) |
| **Ship** | 434 | 224 | 58 | 152 | Large Marine Vessel ($0.07\%$) |
| **Bridge** | 408 | 180 | 77 | 151 | Extreme Aspect Ratio Structure ($0.07\%$) |
| *Ignored / Ambiguous* | 16,804 | 8,169 | 3,590 | 5,045 | Background Rejection ($2.82\%$) |
| **Total Annotations** | **596,732** | **294,222** | **118,059** | **184,451** | **$100.00\%$** |

---

## 9. Comparative Dataset Analysis

In Table I, the authors place CODrone in direct contrast with seven major existing UAV detection benchmarks: VisDrone2019, UAVDT, AU-AIR, CARPK, HazyDet, DroneVehicle, and UAV-ROD.

### Benchmark Comparison Matrix (Table I)
| Benchmark Dataset | Image Resolution | Target Categories | Altitude Diversity / Range | Viewpoint / Camera Angle | Total Images | Total Annotated Objects | OBB Oriented Annotations? |
|---|---|---|---|---|---|---|---|
| **VisDrone2019** [17] | $2000 \times 1500$ | 10 | Unannotated (*) | Unannotated (*) | 10.2k | 54.2k | No (HBB only) |
| **UAVDT** [19] | $1080 \times 540$ | 3 | $60\text{ m}$ (Unannotated) | Unannotated (*) | 80.0k | 841.5k | No (HBB only) |
| **AU-AIR** [18] | $1920 \times 1080$ | 8 | $25\text{ m}$ (Fixed) | $45^\circ$ (Fixed) | 3.2k | 132.0k | No (HBB only) |
| **CARPK** [20] | $1280 \times 720$ | 1 | Unannotated (*) | Unannotated (*) | 1.4k | 89.7k | No (HBB only) |
| **HazyDet** [34] | $1333 \times 800$ | 3 | Unannotated (*) | Unannotated (*) | 11.6k | 383.0k | No (HBB only) |
| **DroneVehicle** [21] | $840 \times 712$ | 5 | $40\text{ m}$ range | $30^\circ$ (Fixed oblique) | 56.8k | 953.0k | **Yes (OBB)** |
| **UAV-ROD** [22] | $1920 \times 1080$ | 1 | $50\text{ m}$ range | Unannotated (*) | 1.5k | 30.0k | **Yes (OBB)** |
| **CODrone (Ours)** | $\mathbf{3840 \times 2160}$ | $\mathbf{12}$ | $\mathbf{70\text{ m}}$ span ($30\text{m}, 60\text{m}, 100\text{m}$) | $\mathbf{60^\circ}$ span ($30^\circ, 90^\circ$) | **10.0k** | **596.7k** | **Yes (OBB)** |

### Critical Analytical Distinctions
1. **Resolution Supremacy:** CODrone is the first UAV OOD dataset to provide full **4K UHD** ($3840 \times 2160$) native imagery, offering $13.8\times$ the pixel count of DroneVehicle ($840 \times 712$) and $4\times$ that of 1080p datasets.
2. **Category Richness:** Unlike UAV-ROD (1 category: car) or DroneVehicle (5 vehicle classes), CODrone spans 12 diverse classes encompassing pedestrians, micromobility, infrastructure, and maritime targets.
3. **Systematic Viewpoint Factorial Design:** Unlike AU-AIR or DroneVehicle which capture at a single static angle ($45^\circ$ or $30^\circ$), CODrone provides balanced cross-factorial combinations of $30\text{m}/60\text{m}/100\text{m}$ and $30^\circ/90^\circ$.

---

## 10. Benchmark Methodology & Evaluation Metrics

### Evaluation Protocol
Evaluations follow standard MS COCO and DOTA evaluation protocols adapted for rotated bounding boxes. Detections and ground truth bounding boxes are matched based on **Rotated Intersection over Union (Rotated IoU / SkewIoU)**:
$$\text{IoU}(B_p, B_{gt}) = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$

### Selected Metrics
1. **$AP_{50}$ (Average Precision at $\text{IoU} \ge 0.50$):** Standard evaluation threshold measuring coarse object detection and categorization ability.
2. **$AP_{75}$ (Average Precision at $\text{IoU} \ge 0.75$):** Strict high-precision localization threshold. As demonstrated in prior OOD literature (Yang et al., KLD/GWD), $AP_{75}$ is extremely sensitive to angular regression errors, particularly for elongated objects where a $3^\circ$ orientation error causes IoU to drop below $0.75$.

### Image Cropping & Training Setup
Given the 4K dimensions ($3840 \times 2160$), images are processed during training and inference using a standard overlapping sliding-window crop strategy ($1024 \times 1024$ patch size with a 200-pixel overlap), matching standard DOTA evaluation protocols. All models are trained using MMRotate on NVIDIA RTX 3090 / A100 GPUs using standard $1\times$ (12 epochs) training schedules with AdamW or SGD optimizers.

---

## 11. Selected Baseline Detectors

The authors benchmark **22 representative and state-of-the-art oriented object detection models**, categorizing them across distinct architectural and methodological paradigms:

1. **Two-Stage Anchor-Based Detectors:**
   - *Rotated Faster R-CNN* (TPAMI 2017): Direct baseline extending Faster R-CNN with 5-parameter rotated anchors and RoI pooling.
   - *RoI Transformer* (CVPR 2019): Uses supervised Rotated RoI Learner to transform horizontal proposals into rotated RoIs, followed by Rotated Position-Sensitive RoI Align.
   - *Oriented R-CNN* (ICCV 2021): Employs an Oriented RPN using a 6-parameter midpoint representation, achieving high efficiency and accuracy.
2. **Single-Stage Anchor-Based Detectors:**
   - *Rotated RetinaNet* (ICCV 2017): Adapts focal loss with rotated anchors.
   - *CSL (Circular Smooth Label)* (ECCV 2020): Converts continuous angle regression into classification with Gaussian-smoothed circular labels to avoid boundary discontinuities.
   - *Rotated ATSS* (CVPR 2020): Extends Adaptive Training Sample Selection to oriented bounding boxes.
   - *Gliding Vertex* (TPAMI 2020): Glides four vertices along horizontal bounding box edges to predict orientation.
   - *R3Det* (AAAI 2021): Single-stage refined detector employing Feature Refinement Modules (FRM) to realign feature representations with rotated bounding boxes.
   - *S2ANet* (TGRS 2021): Single-stage aligned network utilizing Active Rotating Filters (ARF) and AlignConv.
3. **Advanced Metric Loss Formulations:**
   - *GWD (Gaussian Wasserstein Distance)* (ICML 2021): Models bounding boxes as 2D Gaussian distributions and optimizes Wasserstein distance.
   - *KLD (Kullback-Leibler Divergence)* (NeurIPS 2021): Minimizes KL divergence between Gaussian bounding boxes with self-modulated scale-invariance.
   - *KFIoU (Kalman Filter IoU)* (ICLR 2023): Gaussian product approximation of SkewIoU with scale-insensitive center loss.
   - *PSC (Phase-Shifting Coder)* (CVPR 2023): Solves periodic angle ambiguity using phase-frequency coding.
4. **Rotation-Equivariant & Feature Representation Networks:**
   - *ReDet* (CVPR 2021): Incorporates rotation-equivariant convolutions (e2cnn) into the backbone to extract rotation-invariant features across all orientations.
   - *LSKNet* (ICCV 2023): Employs large selective kernels to dynamically adjust spatial receptive fields for varying aerial context.
   - *DeCoupleNet* (TGRS 2024): Decouples spatial and angle features for efficient aerial processing.
5. **Keypoint & Anchor-Free Detectors:**
   - *Rotated RepPoints* (ICCV 2019): Point-set representation for aerial objects.
   - *Oriented RepPoints* (CVPR 2022): Adaptive point learning with quality assessment and non-axis-aligned sample assignment.
6. **Weakly Supervised & Modern Vision Architectures:**
   - *H2RBox* (ICLR 2023): Learns oriented detection using only horizontal box annotations via self-supervised consistency.
   - *H2RBox-v2* (NeurIPS 2023): Boosts horizontal-to-rotated detection by incorporating geometric symmetry priors.
   - *DCFL* (CVPR 2023): Dynamic coarse-to-fine learning for oriented tiny object detection.
   - *OrientFormer* (TGRS 2024): End-to-end transformer-based oriented detector in remote sensing.

---

## 12. Comprehensive Benchmark Results

Table V summarizes overall benchmark performance on the CODrone test split ($184,451$ instances across 3,002 images). All models use ResNet-50 backbones unless specified otherwise.

### Complete 22-Method Performance Comparison (Table V)
| Model Architecture | Conference / Journal Venue | Backbone Network | Overall $AP_{50}$ (%) | Rank ($AP_{50}$) | Overall $AP_{75}$ (%) | Rank ($AP_{75}$) |
|---|---|---|---|---|---|---|
| **LSKNet** [53] | ICCV 2023 | LSKNet-S | **46.92** | **1** | **21.15** | **1** |
| **ReDet** [48] | CVPR 2021 | ReResNet-50 | **44.73** | **2** | **20.17** | **2** |
| **Oriented RepPoints** [49] | CVPR 2022 | ResNet-50 | **44.59** | **3** | **19.62** | **4** |
| **RoI Transformer** [40] | CVPR 2019 | ResNet-50 | **43.03** | **4** | **19.98** | **3** |
| **Rotated ATSS** [42] | CVPR 2020 | ResNet-50 | 42.85 | 5 | 19.19 | 5 |
| **DCFL** [52] | CVPR 2023 | ResNet-50 | 42.77 | 6 | 17.68 | 13 |
| **Oriented R-CNN** [44] | ICCV 2021 | ResNet-50 | 42.33 | 7 | 18.99 | 6 |
| **Gliding Vertex** [43] | TPAMI 2020 | ResNet-50 | 41.71 | 8 | 15.43 | 17 |
| **OrientFormer** [55] | TGRS 2024 | ResNet-50 | 41.02 | 9 | 18.00 | 11 |
| **KLD** [35] | NeurIPS 2021 | ResNet-50 | 40.98 | 10 | 17.73 | 12 |
| **KFIoU** [50] | ICLR 2023 | ResNet-50 | 40.63 | 11 | 18.13 | 9 |
| **Rotated RetinaNet** [38] | ICCV 2017 | ResNet-50 | 40.62 | 12 | 18.26 | 8 |
| **GWD** [45] | ICML 2021 | ResNet-50 | 40.51 | 13 | 18.04 | 10 |
| **Rotated Faster R-CNN** [37] | TPAMI 2017 | ResNet-50 | 40.40 | 14 | 15.23 | 18 |
| **PSC** [8] | CVPR 2023 | ResNet-50 | 40.13 | 15 | 18.93 | 7 |
| **CSL** [41] | ECCV 2020 | ResNet-50 | 40.05 | 16 | 17.28 | 14 |
| **DeCoupleNet** [56] | TGRS 2024 | DeCoupleNet-D0 | 39.90 | 17 | 16.70 | 15 |
| **R3Det** [47] | AAAI 2021 | ResNet-50 | 39.88 | 18 | 16.28 | 16 |
| **H2RBox-v2** [54] | NeurIPS 2023 | ResNet-50 | 39.07 | 19 | 12.21 | 20 |
| **S2ANet** [46] | TGRS 2021 | ResNet-50 | 37.61 | 20 | 10.61 | 21 |
| **H2RBox** [51] | ICLR 2023 | ResNet-50 | 36.22 | 21 | 10.29 | 22 |
| **Rotated RepPoints** [39] | ICCV 2019 | ResNet-50 | 31.25 | 22 | 13.21 | 19 |

### High-Level Empirical Takeaways
1. **The Localization Cliff ($AP_{50}$ vs. $AP_{75}$):** Across all 22 methods, $AP_{50}$ ranges between $31.25\%$ and $46.92\%$, whereas $AP_{75}$ collapses into a narrow band between $10.29\%$ and $21.15\%$. Even the top-performing detector (LSKNet) suffers a **$25.77\%$ absolute drop** ($54.9\%$ relative drop) when evaluated at the stricter $0.75$ IoU threshold.
2. **Dominance of Adaptive Context:** LSKNet outperforms all detectors by dynamically expanding its spatial receptive field using large depthwise selective convolutions, enabling it to aggregate long-range contextual cues across 4K imagery.
3. **Value of Rotation Equivariance:** ReDet ranks 2nd in $AP_{50}$ ($44.73\%$) and 2nd in $AP_{75}$ ($20.17\%$), proving that incorporating group convolutions into the backbone explicitly preserves orientation equivariance under UAV camera rotations.
4. **Weakly Supervised Feasibility:** H2RBox ($36.22\%$) and H2RBox-v2 ($39.07\%$) demonstrate that models trained *only with horizontal bounding box labels* can approach within $1.3\%$ to $4.2\%$ of fully supervised two-stage detectors (Faster R-CNN at $40.40\%$), although their $AP_{75}$ remains severely compromised ($10.29\%$ and $12.21\%$).

---

## 13. Altitude Impact Analysis

As drone altitude increases from $30\text{ m}$ to $100\text{ m}$, ground sampling distance (GSD) widens, causing physical objects to occupy drastically fewer pixels. For example, a $4.5\text{ m}$ passenger car occupies roughly $180\text{ pixels}$ in length at $30\text{ m}$ altitude, shrinking to roughly $54\text{ pixels}$ at $100\text{ m}$. Pedestrians and bicycles shrink to sub-$10\text{ pixel}$ scales.

### Quantitative Degradation Across Altitudes
Extracting data from Table VI across the three altitudes:
- At $30\text{ m}$ (averaged across tilt angles): Top detectors achieve $44.0\% - 49.8\%$ $AP_{50}$.
- At $60\text{ m}$: Performance remains relatively stable for vehicles due to optimal drone field-of-view, but drops for pedestrians ($43.8\% - 47.8\%$ $AP_{50}$).
- At $100\text{ m}$ (specifically under oblique $30^\circ$ tilt): Performance experiences a catastrophic drop. Rotated Faster R-CNN plummets from $39.13\%$ ($30\text{m}$) to $32.72\%$ ($100\text{m}$); Oriented RepPoints drops from $42.92\%$ to $37.75\%$; and $AP_{75}$ drops to single digits ($6.20\% - 15.41\%$).
- **Root Cause:** Feature pyramid networks (FPN) fail to retain high-frequency orientation gradients when small targets undergo pooling operations. At $100\text{ m}$, fine geometric boundaries blur into the background clutter.

---

## 14. Camera Angle & Viewpoint Impact Analysis

Comparing horizontal detection under $90^\circ$ (nadir vertical) versus $30^\circ$ (oblique forward tilt) reveals the massive challenge of drone perspective distortion.

### Nadir ($90^\circ$) vs. Oblique ($30^\circ$) Performance Gap
From Table VI:
- **Under Nadir $90^\circ$:** At $60\text{ m}$ altitude, LSKNet achieves **$51.82\%$ $AP_{50}$ / $27.17\%$ $AP_{75}$**; ReDet achieves **$48.77\%$ $AP_{50}$ / $26.19\%$ $AP_{75}$**; Oriented R-CNN achieves **$48.57\%$ $AP_{50}$ / $24.38\%$ $AP_{75}$**.
- **Under Oblique $30^\circ$:** At $60\text{ m}$ altitude, LSKNet drops to **$43.86\%$ $AP_{50}$ / $19.34\%$ $AP_{75}$** (a **$-7.96\%$ $AP_{50}$** drop); ReDet drops to **$41.49\%$ $AP_{50}$ / $18.61\%$ $AP_{75}$** (a **$-7.28\%$ $AP_{50}$** drop); Oriented R-CNN drops to **$40.85\%$ $AP_{50}$ / $18.56\%$ $AP_{75}$** (a **$-7.72\%$ $AP_{50}$** drop).

### Why Oblique Viewpoints Cripple Conventional Detectors
1. **Vertical Profile Projections:** In a $90^\circ$ view, a vehicle is a simple 2D rectangular roof; in a $30^\circ$ oblique view, the windshield, side doors, tires, and roof project into a complex 3D perspective trap. The 2D bounding box must encapsulate both the top and side facades, creating severe aspect ratio distortion.
2. **Directional Occlusion Cascades:** In oblique views, tall objects (trucks, buses, trees, lamp posts) cast perspective occlusions over adjacent smaller objects (motorcycles, pedestrians), violating planar non-overlap assumptions.
3. **Ambiguity of Bounding Box Heading:** As shown in the paper's Fig. 7, defining whether the bounding box orientation should align with the ground footprint or the apparent 2D perspective bounding rectangle causes severe regression jitter in conventional angle predictors.

---

## 15. Cross-Viewpoint Factorial Analysis

Table VI presents the complete performance breakdown across all six factorial flight configurations ($30\text{m}30^\circ$, $30\text{m}90^\circ$, $60\text{m}30^\circ$, $60\text{m}90^\circ$, $100\text{m}30^\circ$, $100\text{m}90^\circ$).

### Multi-Viewpoint Performance Matrix (Table VI Selected SOTA)
| Model | Metric | $30\text{m} / 30^\circ$ | $30\text{m} / 90^\circ$ | $60\text{m} / 30^\circ$ | $60\text{m} / 90^\circ$ | $100\text{m} / 30^\circ$ | $100\text{m} / 90^\circ$ | Viewpoint Volatility ($\Delta_{\max-\min}$) |
|---|---|---|---|---|---|---|---|---|
| **LSKNet** | $AP_{50}$ | 43.87 | 55.71 | 43.86 | 51.82 | 38.63 | 51.31 | **17.08%** |
| | $AP_{75}$ | 19.75 | 32.38 | 19.34 | 27.17 | 16.45 | 28.23 | **15.93%** |
| **ReDet** | $AP_{50}$ | 41.65 | 49.99 | 41.49 | 48.77 | 35.56 | 49.12 | **14.43%** |
| | $AP_{75}$ | 19.19 | 24.38 | 18.61 | 26.19 | 14.01 | 27.85 | **13.84%** |
| **Oriented RepPoints** | $AP_{50}$ | 42.92 | 44.36 | 42.92 | 48.42 | 37.75 | 47.49 | **10.67%** |
| | $AP_{75}$ | 18.97 | 22.69 | 17.89 | 24.33 | 15.41 | 26.66 | **11.25%** |
| **RoI Transformer** | $AP_{50}$ | 40.62 | 44.73 | 41.12 | 48.67 | 34.67 | 47.12 | **14.00%** |
| | $AP_{75}$ | 18.46 | 22.67 | 17.73 | 25.26 | 13.46 | 26.53 | **13.07%** |
| **Rotated ATSS** | $AP_{50}$ | 40.19 | 40.18 | 40.42 | 48.39 | 34.67 | 48.98 | **14.31%** |
| | $AP_{75}$ | 17.58 | 18.77 | 17.78 | 24.71 | 14.05 | 28.46 | **14.41%** |
| **Oriented R-CNN** | $AP_{50}$ | 39.18 | 44.03 | 40.85 | 48.57 | 35.18 | 46.99 | **13.39%** |
| | $AP_{75}$ | 17.53 | 18.30 | 18.56 | 24.38 | 12.95 | 26.26 | **13.31%** |
| **KLD** | $AP_{50}$ | 38.70 | 40.96 | 39.33 | 45.46 | 32.91 | 42.70 | **12.55%** |
| | $AP_{75}$ | 16.78 | 18.65 | 17.34 | 24.19 | 13.19 | 22.48 | **11.00%** |
| **GWD** | $AP_{50}$ | 37.99 | 41.57 | 38.41 | 46.23 | 32.71 | 43.16 | **13.52%** |
| | $AP_{75}$ | 17.32 | 23.01 | 17.91 | 24.49 | 13.98 | 24.33 | **10.51%** |
| **Rotated Faster R-CNN** | $AP_{50}$ | 39.13 | 39.77 | 39.58 | 45.05 | 32.72 | 44.56 | **12.33%** |
| | $AP_{75}$ | 14.32 | 14.86 | 14.55 | 19.51 | 11.13 | 20.77 | **9.64%** |
| **H2RBox** (Weakly Sup.) | $AP_{50}$ | 37.90 | 50.20 | 34.00 | 40.10 | 28.30 | 37.40 | **21.90%** |
| | $AP_{75}$ | 10.80 | 14.90 | 8.40 | 13.80 | 6.20 | 11.30 | **8.70%** |

### Synthesis of Cross-Factorial Findings
- The worst-case flight regime across all 22 models is **$100\text{ m}$ Altitude at $30^\circ$ Oblique Tilt**. Under this setting, target scale is minimal and perspective distortion is maximized.
- The best-case flight regime is **$30\text{ m} / 90^\circ$** and **$60\text{ m} / 90^\circ$**, where LSKNet reaches peak performance ($55.71\%$ $AP_{50}$).
- Weakly supervised methods (H2RBox) exhibit extreme volatility ($21.90\%$ gap across viewpoints), proving that view-consistency self-supervision breaks down when perspective transformations alter object appearance between viewpoints.

---

## 16. Per-Category Performance Nuances

While full per-class tables reflect the diverse 12 categories, distinct patterns emerge across semantic groupings:
1. **Rigid Vehicles (Car, Truck, Bus):** These classes represent the highest detection accuracy ($AP_{50} > 65\%$ for cars). Their regular rectangular geometry, distinct textures, and predictable aspect ratios align well with Gaussian and anchor-based models.
2. **Vulnerable Road Users & Micromobility (People, Motor, Bicycle, Tricycle):** Pedestrians and cyclists exhibit drastically lower accuracy ($AP_{50} < 25\%$). In 4K imagery, a pedestrian occupies as few as $8 \times 15$ pixels. Unlike vehicles, human bodies do not possess rigid rectangular boundaries; their orientation is defined by walking direction or body pose, which is ambiguous under top-down views.
3. **Marine Targets (Boat, Ship):** Ships are elongated targets with high aspect ratios ($1:4$ to $1:8$), where small angular errors trigger massive IoU penalties. Detection in coastal water bodies benefits from clean, homogeneous sea backgrounds, but struggles with wake patterns and port cranes.
4. **Traffic Infrastructure (Traffic Sign, Traffic Light, Bridge):**
   - *Traffic Signs & Lights:* Tiny objects ($< 16 \times 16$ pixels) that are frequently missed by downsampling layers in FPNs.
   - *Bridges:* Massive spatial extents spanning thousands of pixels across multiple image tiles, with extreme aspect ratios ($> 1:10$), challenging standard receptive fields and anchor scales.

---

## 17. Small & Dense Object Detection Bottlenecks

Small object detection in CODrone is governed by the official difficulty metric $D=1$ ($< 32 \times 32$ pixels).
- **The Stride Dilemma:** Standard backbones (ResNet-50) downsample input images by factors of $4\times, 8\times, 16\times, 32\times$. An $16 \times 16$ pixel traffic sign at $100\text{ m}$ altitude is reduced to a single feature point at the $P_4$ ($16\times$) FPN level and disappears entirely at $P_5$.
- **Crowded Urban Intersections:** In parking areas and congested intersections, vehicles are separated by fewer than 5 pixels. Standard non-maximum suppression (NMS) thresholds tuned for horizontal boxes cause adjacent rotated boxes to be erroneously suppressed unless rotation-aware NMS (Skew-NMS) is applied with high angular fidelity.

---

## 18. High-Aspect-Ratio & Rotational Ambiguity Analysis

For elongated objects (bridges, buses, trucks, boats), orientation estimation errors are magnified non-linearly:
$$\text{IoU}(\theta_{\text{error}}) \approx \frac{1 - \frac{w}{h}|\tan(\theta_{\text{error}})|}{1 + \frac{h}{w}|\tan(\theta_{\text{error}})|}$$
When aspect ratio $AR = h/w \gg 1$, even an angular error of $|\Delta\theta| = 3^\circ$ causes SkewIoU to drop below $0.75$, converting a true positive in $AP_{50}$ into a false negative in $AP_{75}$. This explains why methods using smooth angle metrics (GWD, KLD, KFIoU) achieve decent $AP_{75}$ ranks ($8\text{th} - 12\text{th}$), whereas standard regression losses (Smooth L1 in Rotated Faster R-CNN) suffer large localization penalties (ranking 18th in $AP_{75}$ at $15.23\%$).

---

## 19. Architectural Paradigm Comparison

Grouping the 22 detectors reveals distinct paradigm performance tiers:

| Paradigm Group | Representative Methods | Average $AP_{50}$ Range | Average $AP_{75}$ Range | Core Strengths | Critical Bottlenecks |
|---|---|---|---|---|---|
| **Large Selective Kernel** | LSKNet [53] | **$46.92\%$** | **$21.15\%$** | Dynamic contextual aggregation across 4K resolution; handles scale jumps | Higher computational complexity during inference |
| **Rotation-Equivariant CNNs** | ReDet [48] | **$44.73\%$** | **$20.17\%$** | Explicit group-convolution equivariance; robust against oblique tilt variations | Substantial memory footprint due to orientation channels |
| **Point-Set / Anchor-Free** | Oriented RepPoints [49] | **$44.59\%$** | **$19.62\%$** | Adapts to non-rectangular contours; avoids pre-defined anchor tuning | Regression instability on tiny instances ($< 16\text{px}$) |
| **Two-Stage RoI Learner** | RoI Transformer [40], Oriented R-CNN [44] | $42.33\% - 43.03\%$ | $18.99\% - 19.98\%$ | High proposal quality; strong boundary alignment | Multi-stage overhead; slower inference FPS |
| **Single-Stage Loss-Enhanced** | GWD [45], KLD [35], KFIoU [50], PSC [8] | $40.13\% - 40.98\%$ | $17.73\% - 18.93\%$ | Solves boundary discontinuity & square-like problems; fast inference | Lower recall on tiny objects compared to two-stage RoI align |
| **Weakly Supervised (HBB-Only)** | H2RBox [51], H2RBox-v2 [54] | $36.22\% - 39.07\%$ | $10.29\% - 12.21\%$ | Eliminates expensive OBB manual labeling costs ($>70\%$ saving) | Severe $AP_{75}$ collapse; sensitive to viewpoint tilt changes |

---

## 20. Feature Representation & Alignment Mechanisms

Aerial images exhibit arbitrary target orientation, requiring feature extraction to maintain spatial alignment with oriented proposals:
- **Misalignment in Standard CNNs:** Standard axis-aligned convolutions extract rectangular receptive fields that sample substantial background clutter when an object is rotated at $45^\circ$.
- **AlignConv (S2ANet):** Employs deformable convolution offsets guided by anchor boxes. However, S2ANet ranks 20th on CODrone ($37.61\%$ $AP_{50}$), showing that simple feature realignment without large receptive fields struggles with 4K drone scales.
- **Rotation-Invariant RoI Align (ReDet):** ReDet extracts orientation-equivariant features using $C_8$ group convolutions and projects them into rotation-invariant representations via RiRoI Align. This architecture achieves the highest $AP_{75}$ among all CNNs, proving that geometric equivariance in feature maps is essential for fine-grained OOD under drone tilt.

---

## 21. Loss Function Behavior on UAV Benchmark

CODrone provides a direct empirical comparison between classical regression losses and advanced distribution-metric losses:
- **Smooth L1 Loss (Rotated Faster R-CNN):** Suffers from boundary discontinuity at $[-\pi/2, \pi/2)$ and the square-like problem. Ranks 14th in $AP_{50}$ ($40.40\%$) and 18th in $AP_{75}$ ($15.23\%$).
- **GWD (Gaussian Wasserstein Distance) & KLD (Kullback-Leibler Divergence):**
  - GWD achieves $40.51\%$ $AP_{50}$ and $18.04\%$ $AP_{75}$ (Rank 10).
  - KLD achieves $40.98\%$ $AP_{50}$ and $17.73\%$ $AP_{75}$ (Rank 12).
  - Both methods model rotated rectangles as 2D Gaussians $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, making the regression loss intrinsically continuous and differentiable across angle boundaries. They provide a **$+2.81\%$ absolute boost in $AP_{75}$** over Rotated Faster R-CNN.
- **Kalman Filter IoU (KFIoU):** Achieves $40.63\%$ $AP_{50}$ and $18.13\%$ $AP_{75}$ (Rank 9), verifying that Gaussian-product approximations of SkewIoU effectively stabilize gradient backpropagation.

---

## 22. Weakly Supervised & Point-Supervised Detectors

A major trend in remote sensing is reducing labeling overhead:
- **H2RBox (ICLR 2023):** Trained strictly with horizontal bounding box (HBB) annotations, using geometric reflection and rotation consistency across transformed views to predict orientation. On CODrone, H2RBox reaches **$36.22\%$ $AP_{50}$**, capturing $89.7\%$ of the performance of fully supervised Faster R-CNN ($40.40\%$).
- **H2RBox-v2 (NeurIPS 2023):** Adds symmetry constraints, boosting performance to **$39.07\%$ $AP_{50}$**, coming within $1.33\%$ of fully supervised Faster R-CNN.
- **The Weak Supervision Bottleneck:** In $AP_{75}$, H2RBox ($10.29\%$) and H2RBox-v2 ($12.21\%$) rank near the bottom (22nd and 20th). Without explicit angular ground truth, self-supervised consistency can locate the general object center and aspect ratio, but cannot establish sub-degree orientation precision.

---

## 23. Domain Generalization & Transferability

The benchmark results highlight critical domain generalization vulnerabilities:
- **Cross-Altitude Domain Shift:** Models trained without altitude stratification experience substantial degradation when testing at $100\text{ m}$ ($>12\%$ drop). The scale shift alters textural representations: at $30\text{ m}$, vehicle windshields, hood vents, and wheel rims provide distinct features; at $100\text{ m}$, the vehicle is a solid low-texture blob.
- **Cross-Viewpoint Generalization:** When models are trained on standard top-down imagery and tested on $30^\circ$ oblique imagery, false positive rates spike on building facades, shadow edges, and vertical infrastructure.

---

## 24. Computational Complexity & Edge Feasibility

For onboard autonomous UAV deployment (e.g., NVIDIA Jetson Orin Nano / AGX Orin), computational efficiency is paramount:
- **4K Image Processing Bottleneck:** Directly passing a $3840 \times 2160$ frame through a ResNet-50 backbone requires $\approx 8.3\text{ M}$ pixels, consuming excessive GPU memory and dropping frame rates below $2\text{ FPS}$.
- **Patch-Based Inference Overhead:** Sliding-window cropping with $1024 \times 1024$ patches generates $8 - 12$ overlapping patches per 4K frame, multiplying inference latency by $10\times$.
- **Lightweight Contenders:** DeCoupleNet ($39.90\%$ $AP_{50}$, $16.70\%$ $AP_{75}$) utilizes a lightweight backbone (DeCoupleNet-D0) designed for remote sensing, maintaining competitive accuracy while drastically reducing parameter count. Developing real-time models that run $>30\text{ FPS}$ on 4K drone video remains an open requirement.

---

## 25. Failure Modes & Qualitative Error Analysis

The paper's qualitative inspection highlights four recurring failure modes:
1. **Perspective Foreshortening Truncation:** In $30^\circ$ oblique shots, vehicles in the background are foreshortened into thin slivers, causing anchor boxes to misestimate width-to-height ratios.
2. **Pedestrian Orientation Ambiguity:** Unlike cars, walking pedestrians have roughly circular or square footprints from a top-down nadir view ($90^\circ$). Detectors predict random bounding box rotations, causing severe $AP_{75}$ penalties.
3. **Shadow Confusion:** Long cast shadows during low-sun-angle morning/evening flights deceive detectors into expanding bounding boxes to encompass both the vehicle and its shadow.
4. **Nighttime Glare & Point-Source Reflection:** In nighttime imagery, high-beam headlights and reflective street signage create bloom artifacts that saturate local pixels, obscuring vehicle boundaries.

---

## 26. Ablation Studies & Diagnostic Experiments

The authors conduct comprehensive diagnostic ablations across the dataset's pose parameters:
- **Angle Ablation ($30^\circ$ vs. $90^\circ$):** Testing all 22 models under fixed altitudes reveals that nadir views ($90^\circ$) consistently outperform oblique views ($30^\circ$) by an average of **$+6.5\%$ to $+11.8\%$ in $AP_{50}$**, confirming that oblique perspective distortion is a primary performance bottleneck.
- **Altitude Sensitivity:** Performance degradation scales monotonically with altitude, with the steepest performance slope occurring between $60\text{ m}$ and $100\text{ m}$ for small object classes (people, bicycle, traffic sign).
- **Scale Receptive Field Importance:** LSKNet’s ablation proves that dynamically selecting kernel sizes up to $23 \times 23$ yields significant gains over fixed $3 \times 3$ or $7 \times 7$ convolutions in resolving 4K aerial context.

---

## 27. Theoretical Insights & Mathematical Formulations

The paper provides mathematical clarity on the geometric transformations governing UAV visual perception:

### 1. The Drone Perspective Projection Model
Let $\mathbf{X}_w = [X, Y, Z, 1]^T$ be a world point on the ground plane ($Z=0$), and $\mathbf{x} = [u, v, 1]^T$ be its projected image coordinate. The projection is governed by drone altitude $h$ and gimbal pitch angle $\phi$:
$$\mathbf{x} \sim \mathbf{K} \left[ \mathbf{R}(\phi, \psi, \omega) \;|\; \mathbf{t}(h) \right] \mathbf{X}_w$$
where $\phi = 90^\circ$ reduces the rotation matrix $\mathbf{R}$ to a pure 2D planar similarity transform (preserving orthogonal aspect ratios), whereas $\phi = 30^\circ$ introduces severe projective homography:
$$\mathbf{H} = \mathbf{K} \left( \mathbf{R} - \frac{\mathbf{t}\mathbf{n}^T}{d} \right) \mathbf{K}^{-1}$$
This projective homography warps square and rectangular ground objects into general quadrilaterals, proving why oriented rectangles $(x, y, w, h, \theta)$ experience intrinsic geometric error under oblique viewing angles.

### 2. SkewIoU Sensitivity Formulations
The Rotated IoU between predicted box $B_p$ and ground truth $B_{gt}$ is non-differentiable with respect to orientation angle $\theta$:
$$\frac{\partial \text{SkewIoU}}{\partial \theta} \text{ contains Dirac delta discontinuities at vertex crossings}$$
This explains why methods relying on Gaussian modeling:
$$\mathbf{\Sigma} = \mathbf{R}(\theta) \begin{bmatrix} (w/2)^2 & 0 \\ 0 & (h/2)^2 \end{bmatrix} \mathbf{R}(\theta)^T$$
(such as GWD, KLD, and KFIoU) achieve superior gradient stability across CODrone's diverse viewpoint configurations.

---

## 28. Practical UAV Deployment Implications

For autonomous drone engineering teams, the CODrone benchmark yields critical design guidelines:
1. **Flight Path Planning for Vision Autonomy:** Where possible, autonomous flight controllers should command nadir ($90^\circ$) gimbal orientations for mission phases requiring fine-grained target counting and localization, switching to oblique angles ($30^\circ$) only when long-range reconnaissance or vertical facade inspection is required.
2. **Altitude Optimization:** The $60\text{ m}$ operational ceiling represents the optimal trade-off between spatial coverage and detection precision. Above $60\text{ m}$, small object classes (pedestrians, bicycles) become unreliable without specialized super-resolution or tile-based magnification architectures.
3. **Sensor-Fusion Metadata Injection:** Since detection accuracy correlates heavily with altitude and tilt, future onboard detectors must integrate flight controller telemetry (IMU pitch, barometric/RTK altitude) directly into the feature backbone as conditioning priors.

---

## 29. Limitations & Edge Cases of the Benchmark

Despite its scale and rigor, CODrone exhibits specific limitations:
1. **Discrete vs. Continuous Viewpoint Sampling:** CODrone samples two discrete camera angles ($30^\circ$ and $90^\circ$) and three discrete altitudes ($30\text{m}, 60\text{m}, 100\text{m}$). Real-world drones move continuously along a smooth continuum of pitch ($0^\circ - 90^\circ$) and altitudes ($5\text{m} - 150\text{m}$).
2. **Single Sensor Modality:** CODrone is exclusively an RGB dataset. It lacks paired thermal infrared (TIR) channels, making it unsuited for night-vision multi-modal fusion research (unlike DroneVehicle).
3. **Severe Class Imbalance:** Car instances (227,751) outnumber bridges (408) and ships (434) by more than $500\times$, requiring careful long-tailed loss weighting during model development.
4. **Geographic Clustering:** While multi-city, data collection was conducted primarily within Chinese urban and suburban environments, which feature specific architectural, road layout, and vehicle styles.

---

## 30. Critical Analysis & Unaddressed Gaps

1. **The Representation Gap of Oriented Bounding Boxes Under Oblique Views:** The paper uses standard oriented rectangles $(x, y, w, h, \theta)$ to annotate $30^\circ$ oblique shots. However, under perspective projection, a 3D cuboid does not project onto a 2D rectangle; it projects onto a general hexagon or quadrilateral. Annotating oblique drone imagery with standard 5-parameter rectangles forces annotators to make subjective decisions about whether to include vertical vehicle sides, creating irreducible label noise.
2. **Absence of 3D Bounding Boxes:** As drone cameras tilt away from nadir, the true perception requirement transitions from 2D OBB to full 3D oriented bounding boxes $(x, y, z, l, w, h, \text{yaw}, \text{pitch}, \text{roll})$.
3. **Computational Inference Benchmarks:** The benchmark presents extensive accuracy metrics ($AP_{50}, AP_{75}$), but lacks systematic throughput, memory footprint, and latency benchmarks across embedded edge platforms (Jetson Orin, Hailo-8).

---

## 31. Open Research Directions & Future Trajectories

CODrone opens several fertile research avenues:
1. **Pose-Conditioned Drone Detectors:** Designing architectures that take UAV gimbal pitch and altitude telemetry as explicit conditioning inputs (via FiLM or cross-attention) to dynamically modulate feature pyramid receptive fields.
2. **Cross-Resolution Super-Resolution Detection:** Integrating real-time image super-resolution modules into aerial detection pipelines to restore sub-pixel target details at $100\text{ m}$ altitude.
3. **3D Oriented Bounding Box Reconstruction:** Leveraging CODrone's multi-altitude and multi-angle shots to train monocular 3D bounding box estimators from aerial perspectives.
4. **Long-Tailed Oriented Detection:** Utilizing the extreme class distribution of CODrone to develop novel re-weighting and decoupled learning algorithms tailored for aerial OOD.

---

## 32. Relevance to Aerial Surveillance & Autonomous Systems

CODrone is directly relevant to high-priority autonomous aerospace applications:
- **Autonomous Drone Delivery & Landing Site Verification:** Detecting ground obstacles, pedestrians, and irregular terrain at varying approach altitudes ($30\text{ m} \to 5\text{ m}$) prior to package drop-off or touchdown.
- **Urban Traffic Flow & Smart City Management:** Round-the-clock monitoring of traffic congestion, illegal parking, and multimodal intersection flow across complex day and night lighting.
- **Disaster Response & Search-and-Rescue (SAR):** Rapid localization of survivors (people), emergency vehicles, and damaged infrastructure (bridges) in cluttered environments.

---

## 33. Cross-Paper Synergy & Positioning in Literature

| Dimension | GWD (Yang et al., ICML 2021) | KLD (Yang et al., NeurIPS 2021) | DroneVehicle (Sun et al., IEEE TCSVT 2022) | CODrone (Ye et al., 2025) |
|---|---|---|---|---|
| **Core Contribution** | Gaussian Wasserstein Distance regression loss | Kullback-Leibler Divergence regression loss with scale invariance | Drone RGB-Infrared paired vehicle dataset & uncertainty fusion | Comprehensive 4K UAV OOD benchmark (10k images, 12 classes, 22 baselines) |
| **Image Resolution** | DOTA ($800 \times 800$ to $4000 \times 4000$ crops) | DOTA / HRSC2016 crops | $840 \times 712$ | **$3840 \times 2160$ (Native 4K UHD)** |
| **Sensor Modality** | Monocular optical RGB | Monocular optical RGB | **Dual-spectral RGB + Thermal IR** | Monocular optical RGB (Day + Night) |
| **Viewpoint Annotation** | None (mixed satellite/aerial) | None (mixed satellite/aerial) | Single oblique view ($30^\circ$ tilt) | **Systematic $2 \times 3$ Factorial Matrix** ($30^\circ/90^\circ \times 30\text{m}/60\text{m}/100\text{m}$) |
| **Evaluated Baselines** | Loss comparisons on RetinaNet / Faster R-CNN | Loss comparisons on RetinaNet / Faster R-CNN | Two-stream Faster R-CNN fusion networks | **22 SOTA OOD detectors comprehensively benchmarked** |
| **Performance on CODrone** | $40.51\%$ $AP_{50}$ / $18.04\%$ $AP_{75}$ (Rank 13/10) | $40.98\%$ $AP_{50}$ / $17.73\%$ $AP_{75}$ (Rank 10/12) | — | Baseline benchmark foundation |

### Positioning Synthesis
GWD and KLD provided the theoretical mathematical breakthroughs that solved angle boundary discontinuity and scale misalignment for oriented bounding boxes. DroneVehicle pioneered paired cross-modality aerial sensing under nighttime conditions. **CODrone represents the large-scale empirical proving ground**, taking the theoretical insights of GWD/KLD and evaluating them alongside 20 other modern architectures against the harsh physical realities of real-world 4K drone flight.

---

## 34. Reproducibility & Open Science Audit

- **Open Source Availability:** Fully released on GitHub (`https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV`).
- **Data Completeness:** Complete 10,004 4K images and annotations for all three splits (train, validation, and test) are publicly accessible.
- **Evaluation Tooling:** Full integration with MMRotate evaluation scripts, providing complete reproducibility for all 22 benchmarked detectors.
- **Hardware & Implementation Details:** Clearly documented hyperparameters: ResNet-50 backbones, 12-epoch ($1\times$) training schedules, standard learning rates, and explicit patch-cropping parameters ($1024 \times 1024$, 200px overlap).

---

## 35. Key Takeaways & Synthesis Matrix

```
====================================================================================================
                                      CODRONE BENCHMARK SUMMARY
====================================================================================================
Scale & Resolution:  10,004 images @ 3840 x 2160 (4K UHD) | 596,732 OBB instances across 12 categories
Viewpoint Matrix:    Altitudes: 30m, 60m, 100m | Tilt Angles: 30° (oblique), 90° (nadir) | 6 configurations
Illumination:        61.2% Daytime (6,121 imgs) | 38.8% Nighttime (3,883 imgs) across multiple cities
----------------------------------------------------------------------------------------------------
                               TOP DETECTOR BENCHMARK LEADERBOARD
----------------------------------------------------------------------------------------------------
Rank 1 (Overall):    LSKNet (ICCV 2023)            -> AP50: 46.92% | AP75: 21.15% (Large Selective Kernel)
Rank 2 (Overall):    ReDet (CVPR 2021)             -> AP50: 44.73% | AP75: 20.17% (Rotation-Equivariant)
Rank 3 (Overall):    Oriented RepPoints (CVPR 2022)-> AP50: 44.59% | AP75: 19.62% (Adaptive Point Set)
Rank 4 (Overall):    RoI Transformer (CVPR 2019)   -> AP50: 43.03% | AP75: 19.98% (Learned Rotated RoI)
Rank 5 (Overall):    Rotated ATSS (CVPR 2020)      -> AP50: 42.85% | AP75: 19.19% (Adaptive Sample Select)
Rank 10 (Metric Loss):KLD (NeurIPS 2021)           -> AP50: 40.98% | AP75: 17.73% (Gaussian KL Divergence)
Rank 13 (Metric Loss):GWD (ICML 2021)              -> AP50: 40.51% | AP75: 18.04% (Gaussian Wasserstein)
----------------------------------------------------------------------------------------------------
                                      KEY EMPIRICAL DISCOVERIES
----------------------------------------------------------------------------------------------------
1. The Localization Cliff:  AP75 collapses to 10.29% - 21.15% across all models (avg drop > 23% from AP50)
2. The Oblique Penalty:     30° tilt reduces AP50 by 6.5% - 11.8% compared to 90° nadir views
3. The Altitude Barrier:    100m altitude at 30° tilt is the worst-case regime (AP75 plummets to 6.2% - 16.4%)
4. Weak Supervision Gap:    H2RBox achieves 39.07% AP50 with HBB labels, but collapses to 12.21% AP75
====================================================================================================
```

---

## 36. One-Page Research Card

```
====================================================================================================
RESEARCH CARD: CODrone Benchmark (Ye et al., 2025)
====================================================================================================
PAPER TITLE:     More Clear, More Flexible, More Precise: A Comprehensive Oriented Object Detection
                 Benchmark for UAV
AUTHORS:         Kai Ye, Haidi Tang, Bowen Liu, Pingyang Dai, Liujuan Cao, Rongrong Ji
INSTITUTIONS:    Xiamen University; Peng Cheng Laboratory, China
VENUE / ARCHIVE: arXiv:2504.20032 [cs.CV], April 2025
GITHUB REPO:     https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV

CORE PROBLEM ADDRESSED:
UAV oriented object detection datasets have historically been restricted by low image resolution (<1080p),
narrow object categories (primarily vehicles), single static viewpoints, and unannotated flight altitudes.
Models achieving high scores on satellite or synthetic benchmarks fail under real drone flight dynamics.

THE CODRONE SOLUTION:
- Scale: 10,004 4K UHD images (3840 x 2160) collected using a DJI Mavic 3 Pro across multiple cities.
- Annotations: 596,732 high-quality oriented bounding boxes (OBB) across 12 diverse object categories.
- Pose-Aware Design: 6 balanced viewpoint combinations (Altitudes: 30m, 60m, 100m; Tilt: 30° oblique, 90° nadir).
- Illumination Diversity: 6,121 daytime images (61.2%) and 3,883 nighttime images (38.8%).
- Standardized Splits: 50% Train (5,002 imgs), 20% Val (2,000 imgs), 30% Test (3,002 imgs, fully open).

22-METHOD BENCHMARK HIGHLIGHTS:
- Leaderboard Champion: LSKNet achieves 46.92% AP50 and 21.15% AP75 (large selective kernel context).
- Rotation Equivariance: ReDet ranks 2nd with 44.73% AP50 and 20.17% AP75 (group-convolution backbone).
- Gaussian Metric Losses: GWD (40.51% AP50, 18.04% AP75) and KLD (40.98% AP50, 17.73% AP75) outperform
  standard Rotated Faster R-CNN (15.23% AP75) by eliminating angle boundary discontinuities.
- Weakly Supervised: H2RBox reaches 39.07% AP50 using only horizontal box labels, but lags in AP75 (12.21%).

CRITICAL LESSONS FOR AERIAL ROBOTICS:
1. High-precision orientation (AP75) remains severely compromised under real UAV flight conditions (<22%).
2. Oblique viewing angles (30°) trigger substantial performance penalties (-7% to -12% AP50) due to 3D
   perspective projection, foreshortening, and facade occlusions.
3. At 100m altitude, small objects (pedestrians, bicycles) suffer severe feature vanishing in standard FPNs.
4. Autonomous UAV mission planners should favor nadir (90°) gimbal angles and <= 60m altitudes for vision tasks.
====================================================================================================
```
