# Literature Survey: UAV-OBB: An Aerial Urban Vehicle Dataset with Oriented Bounding Boxes for Remote Sensing Object Detection in Smart Cities

**Document Metadata:**
- **Surveyed Paper:** *UAV-OBB: An Aerial Urban Vehicle Dataset with Oriented Bounding Boxes for Remote Sensing Object Detection in Smart Cities*
- **Authors:** Israr Ahmad, Shang Fengjun, Kiran Bibi, and Muhammad Salman Pathan
- **Affiliations:** School of Computer Science and Technology & Key Laboratory of Computer Network and Communication Technology, Chongqing University of Posts and Telecommunications (CQUPT), Chongqing, China; Beijing Language and Culture University, Beijing, China; ADAPT SFI Research Centre, School of Computing, Dublin City University (DCU), Dublin, Ireland
- **Publication / Venue:** *Data in Brief* (Elsevier), Volume 66, Article 112710, March 2026 (Published Online: March 25, 2026)
- **Identifiers:** DOI: [10.1016/j.dib.2026.112710](https://doi.org/10.1016/j.dib.2026.112710) | PubMed Central: [PMC13092195](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13092195/) | PMID: 42011240
- **Data Repository:** Mendeley Data (DOI: [10.17632/6snrjwcpkh.4](https://doi.org/10.17632/6snrjwcpkh.4)) | License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Deliverable Type:** Research-Grade Literature Survey (36-Section Master Analysis)
- **Focus Areas:** Unmanned Aerial Vehicles (UAVs), Oriented Bounding Box (OBB) Detection, Smart City Traffic Surveillance, YOLOv8-OBB, Aerial Vehicle Classification, Urban Remote Sensing

---

## 1. Executive Summary & Bibliographic Metadata

*UAV-OBB: An Aerial Urban Vehicle Dataset with Oriented Bounding Boxes for Remote Sensing Object Detection in Smart Cities*, authored by Israr Ahmad et al. (CQUPT, Beijing Language and Culture University, and Dublin City University, 2026), presents a dedicated, high-precision aerial traffic perception dataset designed to eliminate the severe limitations of standard axis-aligned horizontal bounding box (HBB) annotations in low-altitude drone surveillance. In dense urban environments—such as multi-lane expressways, roundabouts, bridges, and complex intersections—traditional HBB representations encompass massive amounts of redundant background noise (road surfaces, lane dividers, tree shadows) and cause overlapping bounding boxes between adjacent vehicles, severely confusing non-maximum suppression (NMS) algorithms. Furthermore, HBB fails to capture the physical orientation (heading angle) of vehicles, which is indispensable for tracking trajectories, calculating turning movements, and modeling traffic flow in smart city intelligent transportation systems (ITS).

To resolve these challenges, the authors construct **UAV-OBB**, comprising **1,617 high-resolution RGB images** ($1920 \times 1080$) and two raw aerial video sequences captured via a DJI Mavic 3 drone operating at operational altitudes between **$75\text{ m}$ and $108\text{ m}$** across major road networks in **Chongqing** and **Wuhan**, China. The dataset features **46,807 manually annotated oriented bounding box (OBB)** instances spanning six granular vehicle categories: `bike`, `bus`, `car`, `other_vehicle`, `taxi`, and `truck`. Crucially, the authors separate public taxis (`taxi`, 3,859 instances) from private cars (`car`, 27,212 instances) and include vulnerable two-wheelers (`bike`, 8,605 instances) and irregular regional transport (`other_vehicle`, 2,135 instances), providing fine-grained urban traffic discrimination.

Annotations are provided in normalized **YOLOv8-OBB 4-corner vertex format** ($[x_1, y_1, x_2, y_2, x_3, y_4]$), backed by rigorous inter-annotator agreement testing ($\text{IoU} > 0.85$ across $>95\%$ of sampled instances). A diagnostic baseline using a fine-tuned small-target YOLOv8-OBB detector establishes a strong benchmark: achieving **$0.786$ ($78.6\%$) $mAP_{50}$** and **$0.618$ ($61.8\%$) $mAP_{50-95}$** across all classes, with high heading precision on rigid vehicles (`taxi`: $0.929$ $mAP_{50}$, `bus`: $0.890$, `car`: $0.893$) and highlighting open research bottlenecks on micromobility targets (`bike`: $0.608$ $mAP_{50}$, $0.395$ $mAP_{50-95}$).

### Bibliographic Matrix
| Attribute | Detail |
|---|---|
| **Title** | UAV-OBB: An Aerial Urban Vehicle Dataset with Oriented Bounding Boxes for Remote Sensing Object Detection in Smart Cities |
| **Short Name** | UAV-OBB Dataset |
| **Authors** | Israr Ahmad, Shang Fengjun, Kiran Bibi, Muhammad Salman Pathan |
| **Institutions** | Chongqing University of Posts and Telecommunications (China); Dublin City University (Ireland) |
| **Publication** | *Data in Brief* (Elsevier), Vol. 66, Article 112710 (March 2026) |
| **DOI & Identifiers** | DOI: 10.1016/j.dib.2026.112710 | PMC: PMC13092195 | PMID: 42011240 |
| **Open Data Repository** | Mendeley Data: `https://doi.org/10.17632/6snrjwcpkh.4` | License: CC BY 4.0 |
| **Primary Tasks** | Rotation-Aware Aerial Vehicle Detection, Smart City ITS, YOLOv8-OBB Benchmarking |
| **Dataset Scale** | 1,617 images ($1920 \times 1080$), 46,807 OBB instances, 6 classes, 2 video sequences |

---

## 2. Problem Statement & Operational Context

Smart city transportation management increasingly relies on low-altitude Unmanned Aerial Vehicles (UAVs) to monitor traffic congestion, detect illegal driving maneuvers, supervise multi-modal intersections, and dispatch emergency response vehicles. Compared to fixed CCTV roadside gantries, drones offer dynamic 3D repositioning, wide fields of view, and rapid deployment without roadside infrastructure dependencies.

However, aerial object detection algorithms face severe operational failure modes when trained on conventional drone datasets:
1. **Horizontal Bounding Box (HBB) Flaws:** When vehicles travel diagonally across an intersection or curve around a roundabout, an axis-aligned bounding box expands to cover an area significantly larger than the physical vehicle. In dense traffic queues, adjacent HBBs overlap heavily, causing standard Greedy NMS to suppress legitimate adjacent vehicles.
2. **Loss of Heading & Kinematics:** HBB does not record vehicle yaw angle ($\theta$). Without orientation, downstream tracking algorithms cannot distinguish whether a vehicle is reversing, drifting across lanes, or executing a tight turn.
3. **Severe Class Coarseness:** Existing aerial vehicle datasets aggregate all automobiles into a monolithic "car" class. In actual smart city operations, distinguishing commercial public transit (taxis, ride-hailing) from private sedans, and separating two-wheel micromobility (e-bikes, scooters) from larger traffic is critical for bus-lane enforcement, taxi queue management, and pedestrian safety auditing.

UAV-OBB addresses these gaps by delivering an oriented bounding box dataset captured under real-world Chinese megacity traffic dynamics.

---

## 3. Core Motivation & Driving Questions

The dataset was engineered to address four primary research gaps in smart city remote sensing:
1. **Bridging the OBB Gap in Urban Drone Sensing:** While satellite datasets like DOTA and HRSC2016 use OBB, their extreme flight altitudes ($>200\text{ m}$) make them poorly representative of low-altitude urban drone dynamics ($75-108\text{ m}$), where camera perspectives, building occlusions, and tree canopies dominate.
2. **Fine-Grained Vehicle Typology:** Differentiating yellow/liveried taxis from private cars and isolating light electric two-wheelers (e-bikes/scooters) that share urban roadways.
3. **Standardizing on Modern Edge Formats:** Existing OBB datasets use legacy DOTA polygon text formats that require heavy preprocessing. UAV-OBB standardizes directly on the normalized 8-value quadrilateral format utilized natively by **YOLOv8-OBB**, enabling instant fine-tuning on edge devices.
4. **Providing Temporal Continuity:** Releasing full-motion 4K/30fps aerial video sequences alongside static extracted frames to support multi-frame temporal tracking and video object detection research.

---

## 4. Dataset Architecture & Sensor Specifications

### Flight Platform & Hardware Configuration
- **UAV Platform:** DJI Mavic 3 commercial quadcopter.
- **Primary Imaging Sensor:** Hasselblad L2D-20c camera featuring a 4/3 CMOS sensor (20 MP).
- **Lens & Optics:** $24\text{ mm}$ equivalent focal length, $84^\circ$ FOV, $f/2.8$ aperture.
- **Flight Altitudes:** Strictly logged at altitudes between **$75\text{ m}$ and $108\text{ m}$** Above Ground Level (AGL), balancing wide spatial road coverage with sufficient pixel resolution to resolve two-wheelers.
- **Camera Perspective:** Predominantly **nadir (top-down vertical, $\approx 90^\circ$)**, providing an orthogonal projection that minimizes perspective occlusions while capturing true physical vehicle planar footprints.
- **Raw Video Capture:** 4K UHD video recorded at $3840 \times 2160$ resolution at 30 frames per second (fps) in MP4 format.
- **Extracted Frame Resolution:** Static frames were subsampled from video clips and resized/exported at **$1920 \times 1080$ Full HD (FHD)** in standard JPEG format to balance spatial detail against edge GPU memory constraints.

---

## 5. Ground Truth & Annotation Protocol

### Annotation Standard & Coordinate System
UAV-OBB utilizes the **YOLOv8-OBB format**, where each object instance is encoded as a single line in a plain-text file matching the image basename:
$$\langle\text{class\_index}\rangle \quad x_1 \quad y_1 \quad x_2 \quad y_2 \quad x_3 \quad y_3 \quad x_4 \quad y_4$$
where:
- $\text{class\_index} \in \{0, 1, 2, 3, 4, 5\}$ corresponding alphabetically to: `bike` (0), `bus` (1), `car` (2), `other_vehicle` (3), `taxi` (4), `truck` (5).
- $(x_i, y_i)$ represent the four corner vertices of the oriented quadrilateral, normalized to the range $[0, 1]$ by dividing by image width ($W=1920$) and height ($H=1080$).
- Vertices are arranged in consistent clockwise order starting from the front-left or top-left corner.

### Quality Control & Inter-Annotator Verification
1. **Annotation Software:** Objects were annotated using X-AnyLabeling, an open-source tool tailored for rotated bounding box drafting.
2. **Two-Stage Curation Pass:** After initial annotator labeling, an experienced senior vision annotator performed an exhaustive second-pass verification, adjusting bounding box edge rotations, correcting fine boundary misalignments, and relabeling misclassified targets.
3. **Occlusion Handling:** Vehicles partially occluded under roadside tree canopies or bridge overpasses were explicitly retained and annotated for their visible portions, forcing models to learn through visual occlusion.
4. **Quantitative Inter-Annotator Agreement Test:** Fifty randomly selected images were labeled independently by two annotators blinded to each other's work. Over **$95\%$ of instances achieved an $\text{IoU} > 0.85$**, confirming exceptionally low annotation noise and high rotational consistency.

---

## 6. Viewpoint, Altitude & Pose Diversity

- **Altitude Band ($75\text{ m} - 108\text{ m}$):** Represents the realistic operational flight band authorized for urban commercial drones in China (below the standard $120\text{ m}$ regulatory ceiling). Within this band, a standard sedan occupies approximately $40 \times 20$ pixels, a city bus spans $120 \times 35$ pixels, and an electric scooter occupies approximately $12 \times 6$ pixels.
- **Nadir-Dominant Viewing:** Focusing on top-down views eliminates complex 3D perspective distortion (trapezoidal warping), ensuring that bounding boxes represent true 2D bounding rectangles that align with road coordinate systems.
- **Spatial Distribution Heatmaps:** As analyzed in the paper's Figure 4, vehicle centroids across train, validation, and test splits show realistic spatial clustering along linear transportation corridors, highway lanes, roundabout rings, and bridge choke points.

---

## 7. Environmental & Illumination Conditions

- **Geographic Testbeds:** Imagery was collected across two major Chinese megacities:
  - **Chongqing:** Renowned for mountainous 3D topography, complex multi-tier highway flyovers, suspension bridges, and dense urban canyons.
  - **Wuhan:** Characterized by wide multi-lane arterial boulevards, extensive river bridge crossings, and high-density commercial intersections.
- **Diurnal Variations:** Data collection was conducted across morning rush hour, midday sunlight, and late afternoon/evening twilight.
- **Atmospheric Conditions:** Images feature diverse atmospheric clarity including clear skies, overcast lighting, light rain, mist, and urban atmospheric haze, providing natural visual robustness against illumination and contrast shifts.

---

## 8. Dataset Scale & Statistical Distribution

UAV-OBB contains **1,617 images** and **46,807 annotated vehicle instances**. The dataset is partitioned into:
- **Training Set:** 1,383 images ($85.53\%$) containing 40,072 vehicle instances.
- **Validation Set:** 218 images ($13.48\%$) containing 6,223 vehicle instances.
- **Test Set:** 16 static images ($0.99\%$) containing 512 instances (supplemented by continuous video sequences for deployment evaluation).

### Comprehensive Instance Distribution (Table 1)
| Class Label | Class ID | Training Set | Validation Set | Test Set | Total Instances | Overall Percentage (%) |
|---|---|---|---|---|---|---|
| **Car** | 2 | 23,407 | 3,523 | 282 | **27,212** | **$58.14\%$** |
| **Bike** | 0 | 7,390 | 1,089 | 126 | **8,605** | **$18.38\%$** |
| **Bus** | 1 | 3,828 | 750 | 42 | **4,620** | **$9.87\%$** |
| **Taxi** | 4 | 3,332 | 498 | 33 | **3,859** | **$8.24\%$** |
| **Other Vehicle** | 3 | 1,801 | 313 | 21 | **2,135** | **$4.56\%$** |
| **Truck** | 5 | 314 | 50 | 8 | **376** | **$0.80\%$** |
| **Total** | — | **40,072** | **6,223** | **512** | **46,807** | **$100.00\%$** |

### Semantic Visual Cues & Class Definitions (Table 2)
- **`bike` (Class 0, 8,605 instances):** Motorcycles, mopeds, electric scooters, bicycles. Visual cues: distinct two-wheel longitudinal profile, rider helmet/torso silhouette, ultra-narrow footprint ($< 15\text{px}$).
- **`bus` (Class 1, 4,620 instances):** Large municipal transit buses, airport shuttles, commercial coaches ($>8\text{ m}$). Visual cues: elongated rectangular roof, repeated window arrays, roof ventilation/air-conditioning units.
- **`car` (Class 2, 27,212 instances):** Private sedans, hatchbacks, SUVs, compact passenger cars. Visual cues: standard roof profile, windshield/rear window geometries, uniform aspect ratios ($1:2$ to $1:2.2$).
- **`other_vehicle` (Class 3, 2,135 instances):** Delivery three-wheelers, auto-rickshaws, customized cargo vans, municipal sweepers, tractors. Visual cues: non-standard irregular geometries, open cargo beds, unique aspect ratios.
- **`taxi` (Class 4, 3,859 instances):** Commercial metered taxis and licensed urban cabs. Visual cues: identical chassis to sedans but distinguished by bright yellow/cyan livery and prominent illuminated rooftop advertising/fare light bars.
- **`truck` (Class 5, 376 instances):** Heavy freight lorries, flatbed trucks, concrete mixers, dump trucks. Visual cues: multi-segment chassis, visible cab-to-trailer articulation gaps, textured open cargo beds or cylindrical cement drums.

---

## 9. Comparative Dataset Analysis

| Dataset | Platform | Altitude Band | Resolution | Instances | Classes | OBB? | Fine-Grained Taxi/Bike? |
|---|---|---|---|---|---|---|---|
| **VisDrone2019** [17] | Drone | Variable ($20-150\text{m}$) | $2000 \times 1500$ | 54.2k | 10 | No (HBB) | Aggregated |
| **UAVDT** [19] | Drone | Low/Medium | $1080 \times 540$ | 841.5k | 3 | No (HBB) | Aggregated vehicles |
| **CARPK** [20] | Drone | $40\text{ m}$ (Parking) | $1280 \times 720$ | 89.7k | 1 | No (HBB) | Cars only |
| **DOTA-v2.0** [11] | Satellite/Aerial | $>200\text{ m}$ | Diverse | 1.79M | 18 | **Yes (OBB)** | General categories |
| **DroneVehicle** [21] | Drone | $80-120\text{ m}$ | $840 \times 712$ | 953.0k | 5 | **Yes (OBB)** | Car, truck, bus, van, freight |
| **CODrone** (2025) | Drone | $30, 60, 100\text{ m}$ | **$3840 \times 2160$** | 596.7k | 12 | **Yes (OBB)** | Diverse urban |
| **UAV-OBB** (2026) | Drone | **$75-108\text{ m}$** | **$1920 \times 1080$** | **46.8k** | **6** | **Yes (OBB)** | **Dedicated `taxi`, `bike`, `other`** |

### Strategic Niche of UAV-OBB
While CODrone is a massive multi-altitude benchmark and DroneVehicle is a dual-modal RGB-TIR dataset, **UAV-OBB serves as a lean, highly specialized, ready-to-train benchmark for urban intelligent traffic systems (ITS)**. By strictly targeting the $75-108\text{ m}$ flight ceiling with native YOLOv8-OBB annotations, it allows smart city developers to deploy real-time vehicle classification models without undergoing multi-stage coordinate transformations or sliding-window tiling.

---

## 10. Benchmark Methodology & Evaluation Metrics

### Evaluation Metrics
Performance is measured using standard Pascal VOC / MS COCO Object Detection metrics adapted for Rotated Bounding Boxes:
- **$\text{Box}(P)$ (Precision):** Ratio of true positive detections over all predicted bounding boxes.
- **$\text{Box}(R)$ (Recall):** Ratio of true positive detections over all ground truth annotations.
- **$mAP_{50}$:** Mean Average Precision calculated at a Rotated IoU threshold of $\ge 0.50$.
- **$mAP_{50-95}$:** Mean Average Precision averaged across 10 IoU thresholds from $0.50$ to $0.95$ with step size $0.05$. Represents the definitive metric for fine-grained boundary alignment and orientation fidelity.

---

## 11. Selected Baseline Detector: YOLOv8-OBB

To validate dataset learnability and establish a baseline, the authors fine-tuned an enhanced **YOLOv8-OBB** model:
- **Architecture:** Single-stage anchor-free oriented detector incorporating a modified C2f backbone with small-target feature enhancement layers (P2 feature pyramid level).
- **Bounding Box Head:** Predicts normalized center coordinates, width, height, and rotation angle $\theta$, converted internally to 4-corner polygon coordinates for Rotated IoU loss optimization.
- **Training Setup:** Initialized with pre-trained weights, fine-tuned on the 1,383 training images for 25 diagnostic epochs at input resolution $1024 \times 1024$ using SGD with momentum and cosine learning rate decay.

---

## 12. Comprehensive Baseline Results

Table 3 presents the quantitative baseline detection performance evaluated on the independent validation split (218 images, 6,223 annotated instances).

### Baseline Performance Breakdown (Table 3)
| Class | Evaluation Images | Ground Truth Instances | Precision ($\text{Box } P$) | Recall ($\text{Box } R$) | $mAP_{50}$ (%) | $mAP_{50-95}$ (%) |
|---|---|---|---|---|---|---|
| **All Classes (Mean)** | **218** | **6,223** | **0.848** | **0.722** | **0.786 (78.6%)** | **0.618 (61.8%)** |
| **Taxi** | 175 | 498 | **0.906** | **0.903** | **0.929 (92.9%)** | **0.795 (79.5%)** |
| **Car** | 217 | 3,523 | 0.876 | 0.857 | **0.893 (89.3%)** | **0.736 (73.6%)** |
| **Bus** | 198 | 750 | **0.941** | 0.824 | **0.890 (89.0%)** | **0.766 (76.6%)** |
| **Truck** | 48 | 50 | 0.869 | 0.625 | 0.757 (75.7%) | 0.632 (63.2%) |
| **Other Vehicle** | 150 | 313 | 0.881 | 0.511 | 0.637 (63.7%) | 0.512 (51.2%) |
| **Bike** | 197 | 1,089 | 0.616 | 0.614 | **0.608 (60.8%)** | **0.395 (39.5%)** |

### Critical Analytical Observations
1. **High Overall Baseline Accuracy ($78.6\%$ $mAP_{50}$):** Demonstrates that the dataset possesses clean, consistent annotations that modern anchor-free oriented detectors can learn rapidly.
2. **Taxi Disentanglement Success:** The detector achieves an exceptional **$0.929$ $mAP_{50}$** and **$0.795$ $mAP_{50-95}$** on `taxi`, outperforming standard `car` ($0.893$ / $0.736$). This confirms that distinct livery colors and roof advertisement signs provide strong, separable visual features from an aerial perspective.
3. **Rigid Target Precision:** Large, structured vehicles (`bus`, `car`, `taxi`) achieve high localization precision ($mAP_{50-95} > 73\%$), indicating that oriented boxes tightly encapsulate vehicle contours without background noise.
4. **The Micromobility Bottleneck (`bike`):** Two-wheelers achieve the lowest accuracy ($0.608$ $mAP_{50}$ and $0.395$ $mAP_{50-95}$). Their tiny spatial footprint ($< 15\text{ pixels}$) and visual similarity to road textures create frequent false negatives.
5. **Recall Deficit in `other_vehicle` ($0.511$):** Due to high intra-class morphological variance (rickshaws vs. delivery vans vs. tractors), the detector struggles to generalize across non-standard vehicles.

---

## 13. Altitude Impact Analysis

At the designated altitude range of **$75\text{ m} - 108\text{ m}$**:
- Pixel ground sampling distance (GSD) ranges between $3.0\text{ cm/px}$ and $4.5\text{ cm/px}$.
- This GSD is optimal for vehicle detection: it is low enough to prevent vehicles from degenerating into sub-pixel blur, yet high enough to capture an entire 6-lane urban intersection within a single $1920 \times 1080$ frame.
- Minor altitude variations between $75\text{ m}$ and $108\text{ m}$ introduce moderate scale shifts, training feature pyramid networks to handle multi-scale detection without extreme scale-collapse issues.

---

## 14. Camera Angle & Viewpoint Impact Analysis

By restricting camera pitch to predominantly **nadir ($90^\circ$)**:
- Visual appearances conform strictly to 2D orthogonal geometry.
- Vehicle side doors, windows, and wheels are excluded from the projection, eliminating perspective foreshortening and trapezoidal distortion.
- Yaw orientation corresponds directly to the angle of the long edge of the rectangle with respect to the road lane markers, enabling accurate lane-discipline auditing.

---

## 15. Cross-Viewpoint Factorial Analysis

Unlike CODrone, which deliberately evaluated extreme cross-viewpoint shifts ($30^\circ$ oblique vs. $90^\circ$ nadir), UAV-OBB intentionally focuses on the vertical nadir domain. This structural choice optimizes the dataset for automated traffic counting, intersection throughput measurement, and lane-specific queue length estimation, where orthogonal top-down perspectives are mandated by municipal traffic authorities.

---

## 16. Per-Category Performance Nuances

- **`car` vs. `taxi`:** Distinguishing taxis from cars is a major contribution. Taxis in Chongqing are predominantly yellow/orange with light bars. The detector achieves $0.906$ Precision and $0.903$ Recall on taxis, validating that aerial models can accurately audit commercial ride-hailing traffic.
- **`truck`:** Suffers from low recall ($0.625$) despite high precision ($0.869$). This is directly attributable to sample scarcity (only 376 total instances in the entire dataset, representing $0.8\%$ frequency).
- **`other_vehicle`:** Characterized by high precision ($0.881$) but low recall ($0.511$). Delivery three-wheelers and tricycles often carry irregular cargo bundles covered with tarpaulins, creating high visual entropy that confuses standard convolution backbones.

---

## 17. Small & Dense Object Detection Bottlenecks

- **Micromobility at Scale:** Bicycles and e-bikes occupy fewer than $100\text{ pixels}^2$ in $1920 \times 1080$ images. Downsampling by $16\times$ (P4 layer) reduces them to less than a single feature cell. Adding a high-resolution P2 feature pyramid layer ($4\times$ stride) is essential to preserve spatial gradients.
- **Traffic Congestion Queues:** At signalized intersections, cars queue bumper-to-bumper with less than 2 pixels of physical separation. Oriented bounding boxes prevent the false box merging that plagues horizontal detectors in these dense queues.

---

## 18. High-Aspect-Ratio & Rotational Ambiguity Analysis

Buses ($120 \times 35\text{ px}$, aspect ratio $\approx 3.5:1$) and trucks ($100 \times 30\text{ px}$, aspect ratio $\approx 3.3:1$) possess high aspect ratios where small angular regression errors trigger large IoU penalties. YOLOv8-OBB overcomes this by predicting rotated quadrilateral vertices directly, avoiding the angle boundary discontinuities that occur in 5-parameter angle regression frameworks.

---

## 19. Architectural Paradigm Comparison

UAV-OBB is evaluated using the single-stage anchor-free YOLOv8-OBB paradigm:
- **Anchor-Free Advantage:** Eliminates predefined aspect ratio anchors, allowing the network to regress arbitrarily oriented bounding boxes for both squarish passenger cars and highly elongated city buses.
- **Computational Efficiency:** Enables single-pass inference at $>50\text{ FPS}$ on desktop GPUs and $>15\text{ FPS}$ on embedded edge devices (NVIDIA Jetson), fulfilling smart city edge-compute requirements.

---

## 20. Feature Representation & Alignment Mechanisms

In YOLOv8-OBB, feature representations are extracted via the C2f backbone with cross-stage partial connections. To ensure rotational alignment, the detection head employs rotated decoupled branches where classification scores and bounding box coordinates are regressed independently, preventing classification confidence from degrading when target orientation varies wildly.

---

## 21. Loss Function Behavior on UAV Benchmark

The baseline model utilizes a multi-task loss formulation:
$$\mathcal{L} = \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{box}} \mathcal{L}_{\text{Rotated-IoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$
- **Rotated IoU / Proba-IoU:** Directly computes the intersection of rotated quadrilaterals, penalizing center offset, aspect ratio mismatch, and angular deviation simultaneously.
- **Distribution Focal Loss (DFL):** Models vertex coordinates as probability distributions, providing smooth gradients for occluded vehicle boundaries under tree canopies.

---

## 22. Weakly Supervised & Point-Supervised Detectors

While UAV-OBB provides fully supervised OBB annotations, the dataset authors note that the availability of precise 4-corner coordinates enables researchers to benchmark horizontal-to-oriented weakly supervised models (such as H2RBox) by stripping orientation annotations and measuring recovery accuracy.

---

## 23. Domain Generalization & Transferability

- **Cross-City Robustness:** By sourcing imagery from both Chongqing (mountainous, multi-layer expressways) and Wuhan (flat, wide arterial boulevards), the dataset incorporates built-in urban diversity.
- **Transfer Learning Potential:** The authors demonstrate that pre-training on large aerial datasets (DOTA, VisDrone) and fine-tuning on UAV-OBB yields rapid convergence within 25 epochs, proving high transferability to urban edge applications.

---

## 24. Computational Complexity & Edge Feasibility

- **Native Resolution ($1920 \times 1080$):** Unlike 4K benchmarks that require sliding-window tiling (e.g., CODrone), $1080\text{p}$ images can be downscaled to $1024 \times 1024$ and processed in a single forward pass on embedded platforms (Jetson Orin Nano, Jetson Xavier NX) with minimal memory footprint ($< 2.5\text{ GB}$ VRAM).
- **YOLOv8 Edge Optimization:** Compatible with TensorRT INT8 quantization and ONNX runtime export, enabling real-time onboard drone perception.

---

## 25. Failure Modes & Qualitative Error Analysis

The paper identifies three primary failure modes:
1. **Canopy Occlusion:** Vehicles parked or traveling under dense roadside trees experience partial occlusion. While annotators labeled visible segments, detectors occasionally predict truncated boxes with incorrect aspect ratios.
2. **Micromobility Group Confusion:** Groups of closely traveling electric scooters and bicycles frequently merge into a single detection box.
3. **Truck-to-Van Confusion:** Due to the severe scarcity of heavy trucks ($0.8\%$), delivery vans are occasionally misclassified as trucks.

---

## 26. Ablation Studies & Diagnostic Experiments

The 25-epoch baseline experiment serves as a diagnostic validation:
- Demonstrates that even without extensive hyperparameter optimization, the model converges to $0.786$ $mAP_{50}$.
- Proves that orientation angle is rapidly learnable within 10 epochs for high-aspect-ratio classes (`bus`, `car`).

---

## 27. Theoretical Insights & Mathematical Formulations

### Normalized YOLOv8-OBB Quadrilateral Representation
A bounding box $B$ is represented as four ordered vertices in normalized image coordinates:
$$B = \left\{ (x_i, y_i) \in [0, 1]^2 \;\big|\; i \in \{1, 2, 3, 4\} \right\}$$
The Rotated IoU between predicted box $B_p$ and ground truth $B_{gt}$ is computed via polygon clipping (Sutherland-Hodgman algorithm):
$$\text{Area}(B_p \cap B_{gt}) = \frac{1}{2} \left| \sum_{k=1}^{M} (x_k y_{k+1} - x_{k+1} y_k) \right|$$
This formulation avoids trigonometric singularities ($\tan\theta$) associated with 5-parameter representations $(x, y, w, h, \theta)$.

---

## 28. Practical UAV Deployment Implications

For municipal traffic management systems:
1. **Dedicated Taxi Monitoring:** The high precision on `taxi` ($92.9\%$) enables automated auditing of illegal passenger pickups in bus lanes or yellow-curb zones.
2. **Intersection Queue Measurement:** Tight OBBs permit exact calculation of vehicle counts per lane, providing ground-truth queue lengths for adaptive traffic light timing algorithms.
3. **Temporal Tracking:** The supplementary 4K/30fps video files provide an immediate foundation for benchmarking multi-object tracking (MOT) algorithms using oriented bounding boxes (e.g., ByteTrack-OBB).

---

## 29. Limitations & Edge Cases of the Benchmark

1. **Moderate Dataset Size:** 1,617 images is modest compared to large-scale benchmarks (CODrone: 10,004; DroneVehicle: 56,878).
2. **Small Static Test Split:** The static test split contains only 16 images (512 instances), meaning held-out evaluation relies on validation set metrics or continuous video evaluation.
3. **Class Imbalance:** Extreme frequency skew between cars ($58.1\%$) and trucks ($0.8\%$), reflecting realistic urban fleet distributions but requiring class-balanced loss weighting.
4. **Single Modality:** RGB-only; lacks paired thermal infrared imagery for nighttime operations.

---

## 30. Critical Analysis & Unaddressed Gaps

While UAV-OBB excels in providing ready-to-train YOLOv8-OBB annotations for smart cities, it does not evaluate advanced metric losses (such as GWD or KLD) or rotation-equivariant backbones (such as ReDet). Extending the benchmark across MMRotate detectors would further illuminate architectural trade-offs.

---

## 31. Open Research Directions & Future Trajectories

1. **Video Multi-Object Tracking (MOT-OBB):** Developing rotation-aware Kalman filters to track oriented vehicles across the provided video clips.
2. **Class-Imbalanced Long-Tailed Learning:** Benchmarking re-weighting strategies to boost truck recall without hurting taxi precision.
3. **Cross-Modality Transfer:** Using UAV-OBB to pre-train lightweight edge models for deployment on low-cost surveillance micro-drones.

---

## 32. Relevance to Aerial Surveillance & Autonomous Systems

- **Smart City Traffic Surveillance:** Real-time monitoring of urban choke points, bridges, and highway interchanges.
- **Autonomous Drone Patrols:** Enabling autonomous security drones to patrol industrial parks and report parking violations or lane obstructions.
- **Urban Density Analytics:** Generating spatial heatmaps of traffic congestion to inform municipal road design.

---

## 33. Cross-Paper Synergy & Positioning in Literature

| Dimension | GWD (ICML 2021) | KLD (NeurIPS 2021) | DroneVehicle (TCSVT 2022) | CODrone (2025) | UAV-OBB (2026) |
|---|---|---|---|---|---|
| **Core Focus** | Wasserstein metric loss | Scale-invariant KL loss | Multimodal RGB-TIR fusion | Comprehensive 4K benchmark | Lean Smart City ITS dataset |
| **Scale** | Loss evaluation on DOTA | Loss evaluation on DOTA | 56,878 images / 953k boxes | 10,004 images / 596k boxes | 1,617 images / 46.8k boxes |
| **Classes** | DOTA 15 classes | DOTA 15 classes | 5 vehicle classes | 12 diverse classes | **6 urban classes (with `taxi`, `bike`)** |
| **Altitude** | Satellite / High aerial | Satellite / High aerial | $80-120\text{ m}$ | $30\text{m}, 60\text{m}, 100\text{m}$ | **$75-108\text{ m}$** |
| **Format** | 5-param $(x, y, w, h, \theta)$ | 5-param $(x, y, w, h, \theta)$ | DOTA polygon format | DOTA / VOC format | **Native YOLOv8-OBB 4-corner** |

---

## 34. Reproducibility & Open Science Audit

- **Open Data:** Fully published under **CC BY 4.0** on Mendeley Data (`https://doi.org/10.17632/6snrjwcpkh.4`).
- **Complete Packaging:** Contains raw images, raw 4K videos, YOLOv8-OBB label text files, and coordinate conversion scripts.
- **Indexed Literature:** Published in Elsevier's *Data in Brief* (Vol. 66, 112710) and indexed in PubMed Central (PMC13092195).

---

## 35. Key Takeaways & Synthesis Matrix

```
====================================================================================================
                                      UAV-OBB DATASET SUMMARY
====================================================================================================
Dataset Scale:       1,617 images (1920 x 1080) | 46,807 OBB instances | 2 4K/30fps aerial video clips
Sensors & Platform:  DJI Mavic 3 (Hasselblad 4/3 CMOS) | Altitude: 75m - 108m AGL | Predominantly Nadir
Geography:           Chongqing City & Wuhan, China (bridges, expressways, complex intersections)
Annotation Format:   YOLOv8-OBB 4-corner normalized vertices: <class> x1 y1 x2 y2 x3 y3 x4 y4
----------------------------------------------------------------------------------------------------
                               CLASS DISTRIBUTION & BASELINE (TABLE 3)
----------------------------------------------------------------------------------------------------
Class 0: Bike        8,605 instances (18.4%) -> P: 0.616 | R: 0.614 | mAP50: 60.8% | mAP50-95: 39.5%
Class 1: Bus         4,620 instances ( 9.9%) -> P: 0.941 | R: 0.824 | mAP50: 89.0% | mAP50-95: 76.6%
Class 2: Car        27,212 instances (58.1%) -> P: 0.876 | R: 0.857 | mAP50: 89.3% | mAP50-95: 73.6%
Class 3: Other_Veh   2,135 instances ( 4.6%) -> P: 0.881 | R: 0.511 | mAP50: 63.7% | mAP50-95: 51.2%
Class 4: Taxi        3,859 instances ( 8.2%) -> P: 0.906 | R: 0.903 | mAP50: 92.9% | mAP50-95: 79.5%
Class 5: Truck         376 instances ( 0.8%) -> P: 0.869 | R: 0.625 | mAP50: 75.7% | mAP50-95: 63.2%
OVERALL (Mean):     46,807 instances (100%)  -> P: 0.848 | R: 0.722 | mAP50: 78.6% | mAP50-95: 61.8%
----------------------------------------------------------------------------------------------------
                                      KEY RESEARCH INSIGHTS
----------------------------------------------------------------------------------------------------
1. Taxi Disentanglement:  Taxis achieve 92.9% mAP50, proving aerial livery/roof-sign features are separable.
2. High Overall Accuracy: 78.6% mAP50 confirms clean annotations and rapid learnability in 25 epochs.
3. Micromobility Gap:     Bikes/scooters lag at 39.5% mAP50-95 due to tiny scale (<15px) and canopy clutter.
4. Edge Deployment Ready: 1080p native format + YOLOv8-OBB allows instant edge inference (>50 FPS).
====================================================================================================
```

---

## 36. One-Page Research Card

```
====================================================================================================
RESEARCH CARD: UAV-OBB (Ahmad et al., Data in Brief 2026)
====================================================================================================
PAPER TITLE:     UAV-OBB: An Aerial Urban Vehicle Dataset with Oriented Bounding Boxes for Remote
                 Sensing Object Detection in Smart Cities
AUTHORS:         Israr Ahmad, Shang Fengjun, Kiran Bibi, Muhammad Salman Pathan
INSTITUTIONS:    Chongqing University of Posts and Telecommunications (CQUPT), China; Dublin City
                 University (DCU), Ireland
JOURNAL / VENUE: Data in Brief (Elsevier), Volume 66, Article 112710, March 2026
IDENTIFIERS:     DOI: 10.1016/j.dib.2026.112710 | PMC: PMC13092195 | PMID: 42011240
OPEN DATA:       Mendeley Data: https://doi.org/10.17632/6snrjwcpkh.4 (CC BY 4.0)

CORE OBJECTIVE & CONTRIBUTIONS:
- Solves HBB background noise and lack of orientation in urban aerial vehicle surveillance.
- Constructs a high-quality oriented bounding box (OBB) benchmark captured via DJI Mavic 3 at 75-108m
  over complex road networks in Chongqing and Wuhan, China.
- Delivers 1,617 1080p images and 46,807 OBB annotations in native YOLOv8-OBB 4-corner format.
- Separates commercial taxis (3,859 instances) from private cars (27,212 instances) and includes
  vulnerable road users (8,605 bikes/scooters) and non-standard urban delivery vehicles (2,135).

BASELINE BENCHMARK PERFORMANCE (YOLOv8-OBB Fine-Tuned):
- Overall Accuracy:  0.848 Precision | 0.722 Recall | 78.6% mAP50 | 61.8% mAP50-95
- Taxi Performance:  0.906 Precision | 0.903 Recall | 92.9% mAP50 | 79.5% mAP50-95 (Highest)
- Car Performance:   0.876 Precision | 0.857 Recall | 89.3% mAP50 | 73.6% mAP50-95
- Bus Performance:   0.941 Precision | 0.824 Recall | 89.0% mAP50 | 76.6% mAP50-95
- Truck Performance: 0.869 Precision | 0.625 Recall | 75.7% mAP50 | 63.2% mAP50-95
- Other Vehicle:     0.881 Precision | 0.511 Recall | 63.7% mAP50 | 51.2% mAP50-95
- Bike Performance:  0.616 Precision | 0.614 Recall | 60.8% mAP50 | 39.5% mAP50-95 (Lowest)

DEPLOYMENT & RESEARCH TAKEAWAYS:
1. Native YOLOv8-OBB compatibility eliminates preprocessing, enabling direct edge deployment on Jetson.
2. Separating taxis from cars provides actionable data for municipal bus-lane enforcement and cab auditing.
3. Two-wheelers remain the critical perception bottleneck in aerial traffic vision, demanding high-resolution
   P2 feature pyramids and fine-grained loss functions.
====================================================================================================
```
