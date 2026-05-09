"""Build configuration for C extensions.

Setuptools merges this with pyproject.toml automatically.
Only ext_modules lives here; everything else stays in pyproject.toml.
"""

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="warpt.stress._pointer_chase_ext",
            sources=["warpt/stress/_pointer_chase_ext.cpp"],
            language="c++",
            extra_compile_args=["-O2", "-std=c++17"],
        ),
    ],
)
