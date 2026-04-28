from setuptools import setup, find_packages

setup(
    name="vigil-ml",
    version="0.1.0",
    description="ML training observability SDK — like Sentry for PyTorch failures",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "pynvml>=11.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
        ],
    },
)
