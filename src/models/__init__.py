"""
Model registry and factory for Oriented Bounding Box Detectors.
Exposes a unified interface across all CNN, Vision Transformer, and DETR architectures.
"""

from __future__ import annotations

from typing import Optional, Dict, Type
from src.models.base import BaseOBBDetector
from src.models.yolo_obb import YoloOBBDetector
from src.models.custom_obb import CustomOBBDetector
from src.models.std import STDDetector
from src.models.rvsa import RVSADetector
from src.models.ars_detr import ARSDETRDetector
from src.models.oriented_former import OrientedFormerDetector
from src.models.rio_detr import RioDETRDetector
from src.models.rhino import RHINODetector
from src.models.ao2_detr import AO2DETRDetector
from src.models.swin_obb import SwinOBBDetector
from src.models.lsknet import LSKNetDetector
from src.models.yolo11_obb import YOLO11OBBDetector

MODEL_REGISTRY: Dict[str, Type[BaseOBBDetector]] = {
    # 1. Spatial Transform Decoupling (AAAI 2024)
    "std": STDDetector,
    "std-detector": STDDetector,

    # 2. Rotated Varied-Size Attention (CVPR 2023 / TPAMI 2024)
    "rvsa": RVSADetector,
    "rvsa-detector": RVSADetector,

    # 3. Aspect Ratio-Sensitive DETR (IEEE TGRS 2024)
    "ars-detr": ARSDETRDetector,
    "ars_detr": ARSDETRDetector,

    # 4. OrientedFormer (IEEE TGRS 2024)
    "oriented-former": OrientedFormerDetector,
    "orientedformer": OrientedFormerDetector,

    # 5. Real-Time Oriented DETR (ECCV 2024)
    "rio-detr": RioDETRDetector,
    "rio_detr": RioDETRDetector,

    # 6. Rotated DINO (CVPR 2024)
    "rhino": RHINODetector,
    "rhino-detector": RHINODetector,

    # 7. Arbitrary-Oriented DETR (IEEE TCSVT 2023)
    "ao2-detr": AO2DETRDetector,
    "ao2_detr": AO2DETRDetector,

    # 8. Swin-Transformer OBB (2023)
    "swin-obb": SwinOBBDetector,
    "swin": SwinOBBDetector,

    # 9. Large Selective Kernel Network (ICCV 2023 / CODrone 2025 Rank 1)
    "lsknet": LSKNetDetector,
    "lsk-net": LSKNetDetector,

    # 10. YOLO11-OBB (Ultralytics 2024)
    "yolo11-obb": YOLO11OBBDetector,
    "yolo11n-obb": YOLO11OBBDetector,
    "yolo11m-obb": YOLO11OBBDetector,

    # Foundational Baselines
    "custom-obb": CustomOBBDetector,
    "yolov8n-obb": YoloOBBDetector,
    "yolov8s-obb": YoloOBBDetector,
    "yolov8m-obb": YoloOBBDetector,
}


def get_model(
    model_name: str,
    device: str = "cpu",
    num_classes: int = 10,
    weights_path: Optional[str] = None,
) -> BaseOBBDetector:
    """
    Factory function returning the appropriate detector wrapper.
    Guarantees that every detector inherits BaseOBBDetector and adheres
    to the uniform input (images: List[Image.Image]) and output contracts.
    """
    m_lower = model_name.lower().strip()

    if m_lower in MODEL_REGISTRY:
        detector_cls = MODEL_REGISTRY[m_lower]
        detector = detector_cls(model_name=m_lower, device=device, num_classes=num_classes)
    elif "yolo11" in m_lower:
        detector = YOLO11OBBDetector(model_name=m_lower, device=device, num_classes=num_classes)
    elif "yolo" in m_lower:
        detector = YoloOBBDetector(model_name=m_lower, device=device)
    elif "custom" in m_lower:
        detector = CustomOBBDetector(model_name=m_lower, device=device, num_classes=num_classes)
    else:
        supported = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model name: '{model_name}'.\n"
            f"Supported models: {supported}"
        )

    if weights_path:
        detector.load(weights_path)
    return detector


__all__ = [
    "BaseOBBDetector",
    "MODEL_REGISTRY",
    "get_model",
    # 10 Key Survey Models (Post-2022, Transformer-Focused)
    "STDDetector",
    "RVSADetector",
    "ARSDETRDetector",
    "OrientedFormerDetector",
    "RioDETRDetector",
    "RHINODetector",
    "AO2DETRDetector",
    "SwinOBBDetector",
    "LSKNetDetector",
    "YOLO11OBBDetector",
    # Additional Baselines
    "CustomOBBDetector",
    "YoloOBBDetector",
]
