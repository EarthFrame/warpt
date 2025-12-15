"""Framework detection utilities."""

import warnings

from warpt.backends.software.frameworks.base import FrameworkDetector
from warpt.backends.software.frameworks.bionemo import BioNeMoDetector
from warpt.backends.software.frameworks.deepspeed import DeepSpeedDetector
from warpt.backends.software.frameworks.einops import EinopsDetector
from warpt.backends.software.frameworks.einx import EinxDetector
from warpt.backends.software.frameworks.fairscale import FairScaleDetector
from warpt.backends.software.frameworks.jax import JAXDetector
from warpt.backends.software.frameworks.keras import KerasDetector
from warpt.backends.software.frameworks.matplotlib import MatplotlibDetector
from warpt.backends.software.frameworks.mxnet import MXNetDetector
from warpt.backends.software.frameworks.numba import NumbaDetector
from warpt.backends.software.frameworks.numpy import NumPyDetector
from warpt.backends.software.frameworks.onnx import ONNXDetector
from warpt.backends.software.frameworks.pandas import PandasDetector
from warpt.backends.software.frameworks.polars import PolarsDetector
from warpt.backends.software.frameworks.pytorch import PyTorchDetector
from warpt.backends.software.frameworks.pytorch_lightning import (
    PyTorchLightningDetector,
)
from warpt.backends.software.frameworks.scikit_learn import ScikitLearnDetector
from warpt.backends.software.frameworks.scipy import SciPyDetector
from warpt.backends.software.frameworks.tensorflow import TensorFlowDetector
from warpt.backends.software.frameworks.tensorrt import TensorRTDetector
from warpt.backends.software.frameworks.transformers import TransformersDetector
from warpt.backends.software.frameworks.wandb import WandBDetector
from warpt.backends.software.frameworks.zarr import ZarrDetector
from warpt.models.list_models import FrameworkInfo

__all__ = [
    "BioNeMoDetector",
    "DeepSpeedDetector",
    "EinopsDetector",
    "EinxDetector",
    "FairScaleDetector",
    "FrameworkDetector",
    "JAXDetector",
    "KerasDetector",
    "MXNetDetector",
    "MatplotlibDetector",
    "NumPyDetector",
    "NumbaDetector",
    "ONNXDetector",
    "PandasDetector",
    "PolarsDetector",
    "PyTorchDetector",
    "PyTorchLightningDetector",
    "SciPyDetector",
    "ScikitLearnDetector",
    "TensorFlowDetector",
    "TensorRTDetector",
    "TransformersDetector",
    "WandBDetector",
    "ZarrDetector",
    "detect_all_frameworks",
    "detect_framework",
]


# Registry of all available framework detectors
_FRAMEWORK_DETECTORS = [
    # Deep Learning Frameworks
    PyTorchDetector(),
    JAXDetector(),
    TensorFlowDetector(),
    KerasDetector(),
    MXNetDetector(),
    # PyTorch Ecosystem
    PyTorchLightningDetector(),
    FairScaleDetector(),
    # Model Optimization & Serving
    ONNXDetector(),
    TensorRTDetector(),
    # Specialized Deep Learning
    BioNeMoDetector(),
    DeepSpeedDetector(),
    TransformersDetector(),
    # Tensor & Neural Network Operations
    EinopsDetector(),
    EinxDetector(),
    # Scientific Computing & Data Processing
    NumPyDetector(),
    SciPyDetector(),
    ZarrDetector(),
    PandasDetector(),
    PolarsDetector(),
    # Compilation & Performance
    NumbaDetector(),
    # Visualization & ML
    MatplotlibDetector(),
    ScikitLearnDetector(),
    # Experiment Tracking
    WandBDetector(),
]


def detect_all_frameworks() -> dict[str, FrameworkInfo]:
    """Detect all available ML frameworks.

    Returns
    -------
        Dictionary mapping framework names to their FrameworkInfo objects.
        Includes both installed and non-installed frameworks with
        appropriate `installed` field values.
    """
    detected = {}
    # Suppress warnings during framework detection (frameworks may print
    # initialization warnings when imported that are not relevant to the user)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for detector in _FRAMEWORK_DETECTORS:
            info = detector.detect()
            if info is not None:
                detected[detector.framework_name] = info
            else:
                # Framework not installed, create entry with installed=False
                detected[detector.framework_name] = FrameworkInfo(
                    installed=False,
                    version=None,
                    cuda_support=False,
                )
    return detected


def detect_framework(framework_name: str) -> FrameworkInfo | None:
    """Detect a specific framework by name.

    Args:
        framework_name: Name of the framework to detect (e.g., 'pytorch')

    Returns
    -------
        FrameworkInfo object if framework is installed, None otherwise.
    """
    for detector in _FRAMEWORK_DETECTORS:
        if detector.framework_name == framework_name:
            return detector.detect()
    return None
