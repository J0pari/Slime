"""Package trained models for distribution (Windows .exe, installers)"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict
import logging
import shutil
import json

logger = logging.getLogger(__name__)


class ModelPackager:
    """Package Slime Mold models for distribution"""

    def __init__(self, model_dir: Path, output_dir: Path):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_pip_package(
        self,
        package_name: str = "slime-mold-transformer",
        version: str = "0.1.0",
        author: str = "Anonymous",
        description: str = "Biologically-inspired transformer with dynamic architecture",
    ) -> Path:
        """Create pip-installable package.

        Args:
            package_name: PyPI package name
            version: Package version
            author: Package author
            description: Package description

        Returns:
            Path to created package
        """
        logger.info(f"Creating pip package {package_name}=={version}")

        # Create package directory structure
        pkg_dir = self.output_dir / package_name
        pkg_dir.mkdir(exist_ok=True)

        # Copy source files
        src_dir = self.model_dir.parent
        shutil.copytree(
            src_dir / "slime",
            pkg_dir / "slime",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'),
        )

        # Create setup.py
        setup_py = f"""
from setuptools import setup, find_packages

setup(
    name="{package_name}",
    version="{version}",
    author="{author}",
    description="{description}",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
    ],
    extras_require={{
        "triton": ["triton-windows>=2.0.0"] if sys.platform == "win32" else ["triton>=2.0.0"],
        "viz": ["matplotlib>=3.5.0", "seaborn>=0.11.0"],
        "export": ["onnx>=1.12.0", "onnxruntime>=1.12.0"],
        "datasets": ["datasets>=2.0.0", "transformers>=4.20.0"],
    }},
    entry_points={{
        "console_scripts": [
            "slime-train=slime.training.trainer:main",
        ],
    }},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
"""
        (pkg_dir / "setup.py").write_text(setup_py)

        # Create pyproject.toml
        pyproject_toml = f"""
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "{package_name}"
version = "{version}"
description = "{description}"
readme = "README.md"
requires-python = ">=3.9"
license = {{text = "MIT"}}

[tool.setuptools]
packages = ["slime"]
"""
        (pkg_dir / "pyproject.toml").write_text(pyproject_toml)

        # Create README
        readme = f"""# {package_name}

{description}

## Installation

```bash
pip install {package_name}
```

## Usage

```python
from slime.core.organism import Organism

model = Organism(
    sensory_dim=128,
    latent_dim=256,
    head_dim=64,
)

output = model(input_tensor)
```

## Features

- Sub-quadratic attention with Triton GPU kernels
- Dynamic component lifecycle (birth/death based on fitness)
- MAP-Elites behavioral archive for quality-diversity
- Multi-objective loss functions
- Production-ready with SLO monitoring

## Citation

If you use this model in your research, please cite:

```bibtex
@software{{slime_mold_transformer,
  title = {{Slime Mold Transformer}},
  author = {{{author}}},
  year = {{2025}},
  version = {{{version}}},
}}
```
"""
        (pkg_dir / "README.md").write_text(readme)

        # Build package
        logger.info("Building wheel...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "build"],
            cwd=pkg_dir,
            check=False,
            capture_output=True,
        )
        result = subprocess.run(
            [sys.executable, "-m", "build"],
            cwd=pkg_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(result.stdout)

        dist_dir = pkg_dir / "dist"
        if dist_dir.exists():
            wheels = list(dist_dir.glob("*.whl"))
            if wheels:
                logger.info(f"Created pip package: {wheels[0]}")
                return wheels[0]

        logger.error("Failed to create pip package")
        return pkg_dir

    def create_windows_exe(
        self,
        script_path: Path,
        exe_name: str = "slime_mold",
        icon_path: Optional[Path] = None,
        onefile: bool = True,
    ) -> Path:
        """Create Windows executable using PyInstaller.

        Args:
            script_path: Path to main Python script
            exe_name: Name for executable
            icon_path: Optional icon file (.ico)
            onefile: Create single-file executable

        Returns:
            Path to created executable
        """
        try:
            import PyInstaller.__main__
        except ImportError:
            logger.error("PyInstaller not installed. Install with: pip install pyinstaller")
            raise

        logger.info(f"Creating Windows executable: {exe_name}.exe")

        args = [
            str(script_path),
            f"--name={exe_name}",
            f"--distpath={self.output_dir / 'dist'}",
            f"--workpath={self.output_dir / 'build'}",
            f"--specpath={self.output_dir}",
            "--clean",
        ]

        if onefile:
            args.append("--onefile")

        if icon_path:
            args.append(f"--icon={icon_path}")

        # Hidden imports
        hidden_imports = [
            "torch",
            "numpy",
            "yaml",
            "slime",
        ]
        for module in hidden_imports:
            args.append(f"--hidden-import={module}")

        # Run PyInstaller
        PyInstaller.__main__.run(args)

        exe_path = self.output_dir / "dist" / f"{exe_name}.exe"
        if exe_path.exists():
            logger.info(f"Created executable: {exe_path}")
            return exe_path
        else:
            logger.error("Failed to create executable")
            raise FileNotFoundError(f"Executable not found at {exe_path}")

    def create_docker_image(
        self,
        image_name: str = "slime-mold-transformer",
        tag: str = "latest",
        base_image: str = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
    ) -> str:
        """Create Docker image.

        Args:
            image_name: Docker image name
            tag: Docker image tag
            base_image: Base Docker image

        Returns:
            Full image name with tag
        """
        logger.info(f"Creating Docker image {image_name}:{tag}")

        # Create Dockerfile
        dockerfile_content = f"""
FROM {base_image}

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY slime/ ./slime/
COPY config/ ./config/

# Install triton if on Linux
RUN if [ "$(uname -s)" = "Linux" ]; then \\
        pip install --no-cache-dir triton>=2.0.0; \\
    fi

# Expose port for inference server
EXPOSE 8000

# Run inference server
CMD ["python", "-m", "slime.api.native"]
"""

        dockerfile_path = self.output_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)

        # Create requirements.txt
        requirements = [
            "torch>=2.0.0",
            "numpy>=1.20.0",
            "pyyaml>=6.0",
        ]
        (self.output_dir / "requirements.txt").write_text("\n".join(requirements))

        # Build Docker image
        try:
            result = subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    f"{image_name}:{tag}",
                    "-f",
                    str(dockerfile_path),
                    str(self.model_dir.parent),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(result.stdout)
            logger.info(f"Created Docker image: {image_name}:{tag}")
            return f"{image_name}:{tag}"

        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("Docker not found. Install Docker Desktop for Windows.")
            raise

    def create_deployment_package(
        self,
        include_examples: bool = True,
        include_tests: bool = False,
    ) -> Path:
        """Create complete deployment package.

        Args:
            include_examples: Include example scripts
            include_tests: Include test suite

        Returns:
            Path to deployment package
        """
        logger.info("Creating deployment package...")

        deploy_dir = self.output_dir / "deployment"
        deploy_dir.mkdir(exist_ok=True)

        # Copy source
        shutil.copytree(
            self.model_dir.parent / "slime",
            deploy_dir / "slime",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'),
        )

        # Copy config
        shutil.copytree(
            self.model_dir.parent / "config",
            deploy_dir / "config",
            dirs_exist_ok=True,
        )

        if include_examples:
            examples_dir = deploy_dir / "examples"
            examples_dir.mkdir(exist_ok=True)

            # Create example training script
            train_example = """
import torch
from slime.core.organism import Organism
from slime.training.trainer import Trainer, TrainingConfig
from slime.bench.datasets import GLUEDataset

# Create model
model = Organism(
    sensory_dim=128,
    latent_dim=256,
    head_dim=64,
)

# Create dataset
dataset = GLUEDataset(task_name="sst2", split="train")

# Create trainer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
config = TrainingConfig(num_epochs=10)
trainer = Trainer(model, optimizer, config=config)

# Train
trainer.train(dataset)
"""
            (examples_dir / "train.py").write_text(train_example)

        if include_tests:
            shutil.copytree(
                self.model_dir.parent / "tests",
                deploy_dir / "tests",
                dirs_exist_ok=True,
            )

        # Create deployment manifest
        manifest = {
            "package": "slime-mold-transformer",
            "version": "0.1.0",
            "includes": {
                "source": True,
                "config": True,
                "examples": include_examples,
                "tests": include_tests,
            },
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
        (deploy_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Create archive
        archive_path = self.output_dir / "slime_mold_deployment"
        shutil.make_archive(str(archive_path), 'zip', deploy_dir)

        logger.info(f"Created deployment package: {archive_path}.zip")
        return Path(f"{archive_path}.zip")


def package_all(
    model_dir: Path,
    output_dir: Path,
    script_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Create all package formats.

    Args:
        model_dir: Directory containing trained model
        output_dir: Directory for output packages
        script_path: Path to main script for executable

    Returns:
        Dictionary mapping format to created package path
    """
    packager = ModelPackager(model_dir, output_dir)
    packages = {}

    # Pip package
    try:
        packages['pip'] = packager.create_pip_package()
    except Exception as e:
        logger.error(f"Failed to create pip package: {e}")

    # Windows executable
    if script_path:
        try:
            packages['exe'] = packager.create_windows_exe(script_path)
        except Exception as e:
            logger.error(f"Failed to create Windows exe: {e}")

    # Deployment package
    try:
        packages['deployment'] = packager.create_deployment_package()
    except Exception as e:
        logger.error(f"Failed to create deployment package: {e}")

    logger.info(f"Created {len(packages)} package formats")
    return packages
