import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict
import logging
import shutil
import json
logger = logging.getLogger(__name__)

class ModelPackager:

    def __init__(self, model_dir: Path, output_dir: Path):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_pip_package(self, package_name: str='slime-mold-transformer', version: str='0.1.0', author: str='Anonymous', description: str='Biologically-inspired transformer with dynamic architecture') -> Path:
        logger.info(f'Creating pip package {package_name}=={version}')
        pkg_dir = self.output_dir / package_name
        pkg_dir.mkdir(exist_ok=True)
        src_dir = self.model_dir.parent
        shutil.copytree(src_dir / 'slime', pkg_dir / 'slime', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
        setup_py = f'\nfrom setuptools import setup, find_packages\n\nsetup(\n    name="{package_name}",\n    version="{version}",\n    author="{author}",\n    description="{description}",\n    packages=find_packages(),\n    python_requires=">=3.9",\n    install_requires=[\n        "torch>=2.0.0",\n        "numpy>=1.20.0",\n        "pyyaml>=6.0",\n    ],\n    extras_require={{\n        "triton": ["triton-windows>=2.0.0"] if sys.platform == "win32" else ["triton>=2.0.0"],\n        "viz": ["matplotlib>=3.5.0", "seaborn>=0.11.0"],\n        "export": ["onnx>=1.12.0", "onnxruntime>=1.12.0"],\n        "datasets": ["datasets>=2.0.0", "transformers>=4.20.0"],\n    }},\n    entry_points={{\n        "console_scripts": [\n            "slime-train=slime.training.trainer:main",\n        ],\n    }},\n    classifiers=[\n        "Development Status :: 3 - Alpha",\n        "Intended Audience :: Science/Research",\n        "Programming Language :: Python :: 3",\n        "Programming Language :: Python :: 3.9",\n        "Programming Language :: Python :: 3.10",\n        "Programming Language :: Python :: 3.11",\n    ],\n)\n'
        (pkg_dir / 'setup.py').write_text(setup_py)
        pyproject_toml = f'\n[build-system]\nrequires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "{package_name}"\nversion = "{version}"\ndescription = "{description}"\nreadme = "README.md"\nrequires-python = ">=3.9"\nlicense = {{text = "MIT"}}\n\n[tool.setuptools]\npackages = ["slime"]\n'
        (pkg_dir / 'pyproject.toml').write_text(pyproject_toml)
        readme = f'# {package_name}\n\n{description}\n\n## Installation\n\n```bash\npip install {package_name}\n```\n\n## Usage\n\n```python\nfrom slime.core.organism import Organism\n\nmodel = Organism(\n    sensory_dim=128,\n    latent_dim=256,\n    head_dim=64,\n)\n\noutput = model(input_tensor)\n```\n\n## Features\n\n- Sub-quadratic attention with Triton GPU kernels\n- Dynamic component lifecycle (birth/death based on fitness)\n- MAP-Elites behavioral archive for quality-diversity\n- Multi-objective loss functions\n- Production-ready with SLO monitoring\n\n## Citation\n\nIf you use this model in your research, please cite:\n\n```bibtex\n@software{{slime_mold_transformer,\n  title = {{Slime Mold Transformer}},\n  author = {{{author}}},\n  year = {{2025}},\n  version = {{{version}}},\n}}\n```\n'
        (pkg_dir / 'README.md').write_text(readme)
        logger.info('Building wheel...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'build'], cwd=pkg_dir, check=False, capture_output=True)
        result = subprocess.run([sys.executable, '-m', 'build'], cwd=pkg_dir, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        dist_dir = pkg_dir / 'dist'
        if dist_dir.exists():
            wheels = list(dist_dir.glob('*.whl'))
            if wheels:
                logger.info(f'Created pip package: {wheels[0]}')
                return wheels[0]
        logger.error('Failed to create pip package')
        return pkg_dir

    def create_windows_exe(self, script_path: Path, exe_name: str='slime_mold', icon_path: Optional[Path]=None, onefile: bool=True) -> Path:
        try:
            import PyInstaller.__main__
        except ImportError:
            logger.error('PyInstaller not installed. Install with: pip install pyinstaller')
            raise
        logger.info(f'Creating Windows executable: {exe_name}.exe')
        args = [str(script_path), f'--name={exe_name}', f"--distpath={self.output_dir / 'dist'}", f"--workpath={self.output_dir / 'build'}", f'--specpath={self.output_dir}', '--clean']
        if onefile:
            args.append('--onefile')
        if icon_path:
            args.append(f'--icon={icon_path}')
        hidden_imports = ['torch', 'numpy', 'yaml', 'slime']
        for module in hidden_imports:
            args.append(f'--hidden-import={module}')
        PyInstaller.__main__.run(args)
        exe_path = self.output_dir / 'dist' / f'{exe_name}.exe'
        if exe_path.exists():
            logger.info(f'Created executable: {exe_path}')
            return exe_path
        else:
            logger.error('Failed to create executable')
            raise FileNotFoundError(f'Executable not found at {exe_path}')

    def create_docker_image(self, image_name: str='slime-mold-transformer', tag: str='latest', base_image: str='pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime') -> str:
        logger.info(f'Creating Docker image {image_name}:{tag}')
        dockerfile_content = f'\nFROM {base_image}\n\nWORKDIR /app\n\n# Install dependencies\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n\n# Copy source code\nCOPY slime/ ./slime/\nCOPY config/ ./config/\n\n# Install triton if on Linux\nRUN if [ "$(uname -s)" = "Linux" ]; then \\\n        pip install --no-cache-dir triton>=2.0.0; \\\n    fi\n\n# Expose port for inference server\nEXPOSE 8000\n\n# Run inference server\nCMD ["python", "-m", "slime.api.native"]\n'
        dockerfile_path = self.output_dir / 'Dockerfile'
        dockerfile_path.write_text(dockerfile_content)
        requirements = ['torch>=2.0.0', 'numpy>=1.20.0', 'pyyaml>=6.0']
        (self.output_dir / 'requirements.txt').write_text('\n'.join(requirements))
        try:
            result = subprocess.run(['docker', 'build', '-t', f'{image_name}:{tag}', '-f', str(dockerfile_path), str(self.model_dir.parent)], check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            logger.info(f'Created Docker image: {image_name}:{tag}')
            return f'{image_name}:{tag}'
        except subprocess.CalledProcessError as e:
            logger.error(f'Docker build failed: {e.stderr}')
            raise
        except FileNotFoundError:
            logger.error('Docker not found. Install Docker Desktop for Windows.')
            raise

    def create_deployment_package(self, include_examples: bool=True, include_tests: bool=False) -> Path:
        logger.info('Creating deployment package...')
        deploy_dir = self.output_dir / 'deployment'
        deploy_dir.mkdir(exist_ok=True)
        shutil.copytree(self.model_dir.parent / 'slime', deploy_dir / 'slime', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
        shutil.copytree(self.model_dir.parent / 'config', deploy_dir / 'config', dirs_exist_ok=True)
        if include_examples:
            examples_dir = deploy_dir / 'examples'
            examples_dir.mkdir(exist_ok=True)
            train_example = '\nimport torch\nfrom slime.core.organism import Organism\nfrom slime.training.trainer import Trainer, TrainingConfig\nfrom slime.bench.datasets import GLUEDataset\n\n# Create model\nmodel = Organism(\n    sensory_dim=128,\n    latent_dim=256,\n    head_dim=64,\n)\n\n# Create dataset\ndataset = GLUEDataset(task_name="sst2", split="train")\n\n# Create trainer\noptimizer = torch.optim.Adam(model.parameters(), lr=0.001)\nconfig = TrainingConfig(num_epochs=10)\ntrainer = Trainer(model, optimizer, config=config)\n\n# Train\ntrainer.train(dataset)\n'
            (examples_dir / 'train.py').write_text(train_example)
        if include_tests:
            shutil.copytree(self.model_dir.parent / 'tests', deploy_dir / 'tests', dirs_exist_ok=True)
        manifest = {'package': 'slime-mold-transformer', 'version': '0.1.0', 'includes': {'source': True, 'config': True, 'examples': include_examples, 'tests': include_tests}, 'python_version': f'{sys.version_info.major}.{sys.version_info.minor}'}
        (deploy_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        archive_path = self.output_dir / 'slime_mold_deployment'
        shutil.make_archive(str(archive_path), 'zip', deploy_dir)
        logger.info(f'Created deployment package: {archive_path}.zip')
        return Path(f'{archive_path}.zip')

def package_all(model_dir: Path, output_dir: Path, script_path: Optional[Path]=None) -> Dict[str, Path]:
    packager = ModelPackager(model_dir, output_dir)
    packages = {}
    try:
        packages['pip'] = packager.create_pip_package()
    except Exception as e:
        logger.error(f'Failed to create pip package: {e}')
    if script_path:
        try:
            packages['exe'] = packager.create_windows_exe(script_path)
        except Exception as e:
            logger.error(f'Failed to create Windows exe: {e}')
    try:
        packages['deployment'] = packager.create_deployment_package()
    except Exception as e:
        logger.error(f'Failed to create deployment package: {e}')
    logger.info(f'Created {len(packages)} package formats')
    return packages