"""Library detection for MKL, cuBLAS, cuDNN, OpenBLAS, etc."""

import ctypes.util
import os
import platform
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from warpt.backends.software.base import SoftwareDetector
from warpt.models.list_models import LibraryInfo


class LibraryDetector(SoftwareDetector, ABC):
    """Base class for detecting libraries."""

    @abstractmethod
    def detect(self) -> LibraryInfo | None:
        """Detect the library.

        Returns
        -------
            LibraryInfo if detected, None otherwise.
        """
        pass

    def _find_library_path(self, lib_name: str) -> str | None:
        """Find the path to a library.

        Args:
            lib_name: Name of the library (e.g., 'mkl_rt', 'openblas')

        Returns
        -------
            Path to the library if found, None otherwise.
        """
        # Try ctypes.util.find_library first
        path = ctypes.util.find_library(lib_name)
        if path:
            # If it's just a name, try to resolve it to a full path
            if not os.path.isabs(path):
                # On Linux, find_library often returns the soname
                # We can try to find it using ldconfig
                if platform.system() == "Linux":
                    try:
                        cmd = ["ldconfig", "-p"]
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, check=False
                        )
                        if result.returncode == 0:
                            for line in result.stdout.splitlines():
                                if path in line and "=>" in line:
                                    return line.split("=>")[1].strip()
                    except (OSError, subprocess.SubprocessError):
                        pass
            return path

        # Common paths to check manually if find_library fails
        search_paths = []
        if platform.system() == "Linux":
            search_paths = [
                "/usr/lib",
                "/usr/lib64",
                "/usr/local/lib",
                "/usr/local/lib64",
                "/usr/lib/x86_64-linux-gnu",
            ]
            # Add CUDA paths
            cuda_path = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
            search_paths.extend(
                [
                    f"{cuda_path}/lib64",
                    f"{cuda_path}/targets/x86_64-linux/lib",
                ]
            )
        elif platform.system() == "Darwin":
            search_paths = [
                "/usr/local/lib",
                "/opt/homebrew/lib",
                "/opt/homebrew/opt/openblas/lib",
                "/usr/lib",
            ]

        # Add LD_LIBRARY_PATH
        ld_path = os.environ.get("LD_LIBRARY_PATH")
        if ld_path:
            search_paths.extend(ld_path.split(os.pathsep))

        extensions = []
        if platform.system() == "Linux":
            extensions = [".so"]
        elif platform.system() == "Darwin":
            extensions = [".dylib"]
        elif platform.system() == "Windows":
            extensions = [".dll"]

        for base_path in search_paths:
            if not os.path.isdir(base_path):
                continue
            for ext in extensions:
                # Try both libNAME.ext and NAME.ext
                for prefix in ["lib", ""]:
                    full_path = Path(base_path) / f"{prefix}{lib_name}{ext}"
                    if full_path.exists():
                        return str(full_path)
                    # Also check for versioned ones like .so.1
                    if platform.system() == "Linux":
                        matches = list(
                            Path(base_path).glob(f"{prefix}{lib_name}{ext}.*")
                        )
                        if matches:
                            return str(matches[0])

        return None


class MKLDetector(LibraryDetector):
    """Detector for Intel MKL."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "mkl"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("mkl_rt")
        if not path:
            return None

        # Try to get version if possible
        # MKL doesn't have a simple CLI for version, often encoded in path or
        # requires calling a function in the lib. For now, just report installed.
        return LibraryInfo(installed=True, version=None, path=path)


class OpenBLASDetector(LibraryDetector):
    """Detector for OpenBLAS."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "openblas"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("openblas")
        if not path:
            return None

        return LibraryInfo(installed=True, version=None, path=path)


class CuBLASDetector(LibraryDetector):
    """Detector for NVIDIA cuBLAS."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "cublas"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("cublas")
        if not path:
            return None

        return LibraryInfo(installed=True, version=None, path=path)


class CuDNNDetector(LibraryDetector):
    """Detector for NVIDIA cuDNN."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "cudnn"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("cudnn")
        if not path:
            return None

        # cuDNN version is often in cudnn_version.h if we can find the include dir
        version = None
        if path:
            # Try to find version header relative to lib path
            # lib path is usually .../lib64/libcudnn.so
            # header is usually .../include/cudnn_version.h
            lib_dir = Path(path).parent
            include_dir = lib_dir.parent / "include"
            header = include_dir / "cudnn_version.h"
            if header.exists():
                try:
                    content = header.read_text()
                    major, minor, patch = None, None, None
                    for line in content.splitlines():
                        if "#define CUDNN_MAJOR" in line:
                            major = line.split()[-1]
                        elif "#define CUDNN_MINOR" in line:
                            minor = line.split()[-1]
                        elif "#define CUDNN_PATCHLEVEL" in line:
                            patch = line.split()[-1]
                    if major and minor and patch:
                        version = f"{major}.{minor}.{patch}"
                except Exception:
                    pass

        return LibraryInfo(installed=True, version=version, path=path)


class NCCLDetector(LibraryDetector):
    """Detector for NVIDIA NCCL."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "nccl"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("nccl")
        if not path:
            return None
        return LibraryInfo(installed=True, version=None, path=path)


class RCCLDetector(LibraryDetector):
    """Detector for AMD RCCL."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "rccl"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("rccl")
        if not path:
            return None
        return LibraryInfo(installed=True, version=None, path=path)


class MPIDetector(LibraryDetector):
    """Detector for MPI (OpenMPI, MPICH, etc.)."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "mpi"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        # Try a few common MPI library names
        for lib in ["mpi", "mpich", "openmpi"]:
            path = self._find_library_path(lib)
            if path:
                # Try to get version from mpirun if available
                version = None
                mpirun_path = self._which("mpirun")
                if mpirun_path:
                    cmd_result = self._run_command([mpirun_path, "--version"])
                    if cmd_result and cmd_result[0] == 0:
                        # Extract first line for version
                        version = cmd_result[1].splitlines()[0].strip()

                return LibraryInfo(installed=True, version=version, path=path)
        return None


class OneDNNDetector(LibraryDetector):
    """Detector for Intel oneDNN."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "onednn"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("dnnl")
        if not path:
            return None
        return LibraryInfo(installed=True, version=None, path=path)


class TBBDetector(LibraryDetector):
    """Detector for Intel TBB."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "tbb"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("tbb")
        if not path:
            return None
        return LibraryInfo(installed=True, version=None, path=path)


class CuFFTDetector(LibraryDetector):
    """Detector for NVIDIA cuFFT."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "cufft"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("cufft")
        if not path:
            return None
        return LibraryInfo(installed=True, version=None, path=path)


class CuSPARSEDetector(LibraryDetector):
    """Detector for NVIDIA cuSPARSE."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "cusparse"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("cusparse")
        if not path:
            return None
        return LibraryInfo(installed=True, version=None, path=path)


class CuSOLVERDetector(LibraryDetector):
    """Detector for NVIDIA cuSOLVER."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "cusolver"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("cusolver")
        if not path:
            return None
        return LibraryInfo(installed=True, version=None, path=path)


class MAGMADetector(LibraryDetector):
    """Detector for MAGMA."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "magma"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("magma")
        if not path:
            return None
        return LibraryInfo(installed=True, version=None, path=path)


class BLISDetector(LibraryDetector):
    """Detector for BLIS."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "blis"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        path = self._find_library_path("blis")
        if not path:
            return None
        return LibraryInfo(installed=True, version=None, path=path)


class AccelerateDetector(LibraryDetector):
    """Detector for Apple Accelerate framework."""

    @property
    def software_name(self) -> str:
        """Return the canonical name of the software."""
        return "accelerate"

    def detect(self) -> LibraryInfo | None:
        """Detect if the software is installed and gather its information."""
        if platform.system() != "Darwin":
            return None

        # Accelerate is always present on modern macOS
        # It's not a single .dylib we can find easily with find_library in some cases
        # but we can check for its existence in /System/Library/Frameworks
        path = "/System/Library/Frameworks/Accelerate.framework"
        if os.path.exists(path):
            return LibraryInfo(installed=True, version=None, path=path)
        return None


def detect_all_libraries() -> dict[str, LibraryInfo]:
    """Detect all supported libraries.

    Returns
    -------
        Dictionary mapping library names to LibraryInfo.
        Includes both installed and non-installed libraries.
    """
    detectors: list[LibraryDetector] = [
        MKLDetector(),
        OpenBLASDetector(),
        CuBLASDetector(),
        CuDNNDetector(),
        AccelerateDetector(),
        NCCLDetector(),
        RCCLDetector(),
        MPIDetector(),
        OneDNNDetector(),
        TBBDetector(),
        CuFFTDetector(),
        CuSPARSEDetector(),
        CuSOLVERDetector(),
        MAGMADetector(),
        BLISDetector(),
    ]

    results = {}
    for detector in detectors:
        info = detector.detect()
        if info:
            results[detector.software_name] = info
        else:
            # Library not installed, create entry with installed=False
            results[detector.software_name] = LibraryInfo(
                installed=False,
                version=None,
                path=None,
            )

    return results
