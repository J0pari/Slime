import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import json
logger = logging.getLogger(__name__)

class ModelExporter:

    def __init__(self, model: nn.Module, device: Optional[torch.device]=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def export_to_torchscript(self, save_path: Path, example_input: Optional[torch.Tensor]=None, input_shape: Optional[Tuple[int, ...]]=None, method: str='trace') -> Path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if example_input is None:
            if input_shape is None:
                raise ValueError('Must provide either example_input or input_shape')
            example_input = torch.randn(*input_shape, device=self.device)
        else:
            example_input = example_input.to(self.device)
        logger.info(f'Exporting to TorchScript using {method} method...')
        try:
            if method == 'trace':
                traced_model = torch.jit.trace(self.model, example_input)
            elif method == 'script':
                traced_model = torch.jit.script(self.model)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")
            torch.jit.save(traced_model, str(save_path))
            logger.info(f'Saved TorchScript model to {save_path}')
            loaded = torch.jit.load(str(save_path))
            with torch.no_grad():
                original_out = self.model(example_input)
                loaded_out = loaded(example_input)
                diff = torch.abs(original_out - loaded_out).max().item()
                logger.info(f'Verification max diff: {diff}')
            return save_path
        except Exception as e:
            logger.error(f'Failed to export to TorchScript: {e}')
            raise

    def export_to_onnx(self, save_path: Path, example_input: Optional[torch.Tensor]=None, input_shape: Optional[Tuple[int, ...]]=None, input_names: Optional[list]=None, output_names: Optional[list]=None, dynamic_axes: Optional[Dict[str, Dict[int, str]]]=None, opset_version: int=17) -> Path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if example_input is None:
            if input_shape is None:
                raise ValueError('Must provide either example_input or input_shape')
            example_input = torch.randn(*input_shape, device=self.device)
        else:
            example_input = example_input.to(self.device)
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        logger.info(f'Exporting to ONNX (opset {opset_version})...')
        try:
            torch.onnx.export(self.model, example_input, str(save_path), input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=opset_version, do_constant_folding=True, export_params=True)
            logger.info(f'Saved ONNX model to {save_path}')
            try:
                import onnx
                import onnxruntime as ort
                onnx_model = onnx.load(str(save_path))
                onnx.checker.check_model(onnx_model)
                logger.info('ONNX model validation passed')
                ort_session = ort.InferenceSession(str(save_path))
                ort_inputs = {input_names[0]: example_input.cpu().numpy()}
                ort_outputs = ort_session.run(None, ort_inputs)
                with torch.no_grad():
                    torch_output = self.model(example_input).cpu().numpy()
                diff = abs(torch_output - ort_outputs[0]).max()
                logger.info(f'ONNX verification max diff: {diff}')
            except ImportError:
                logger.warning('onnx/onnxruntime not installed - skipping verification')
            return save_path
        except Exception as e:
            logger.error(f'Failed to export to ONNX: {e}')
            raise

    def export_weights(self, save_path: Path, include_optimizer: bool=False, optimizer: Optional[torch.optim.Optimizer]=None) -> Path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {'model_state_dict': self.model.state_dict(), 'model_config': self._get_model_config()}
        if include_optimizer:
            if optimizer is None:
                raise ValueError('Must provide optimizer if include_optimizer=True')
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, save_path)
        logger.info(f'Saved checkpoint to {save_path}')
        return save_path

    def export_config(self, save_path: Path) -> Path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        config = self._get_model_config()
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f'Saved config to {save_path}')
        return save_path

    def _get_model_config(self) -> Dict[str, Any]:
        config = {}
        for attr in ['sensory_dim', 'latent_dim', 'head_dim', 'device']:
            if hasattr(self.model, attr):
                value = getattr(self.model, attr)
                if isinstance(value, torch.device):
                    value = str(value)
                config[attr] = value
        config['num_parameters'] = sum((p.numel() for p in self.model.parameters()))
        config['num_trainable_parameters'] = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
        return config

    def export_all(self, output_dir: Path, model_name: str='slime_mold', example_input: Optional[torch.Tensor]=None, input_shape: Optional[Tuple[int, ...]]=None, optimizer: Optional[torch.optim.Optimizer]=None) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exports = {}
        try:
            exports['torchscript'] = self.export_to_torchscript(output_dir / f'{model_name}.pt', example_input=example_input, input_shape=input_shape)
        except Exception as e:
            logger.error(f'TorchScript export failed: {e}')
        try:
            exports['onnx'] = self.export_to_onnx(output_dir / f'{model_name}.onnx', example_input=example_input, input_shape=input_shape)
        except Exception as e:
            logger.error(f'ONNX export failed: {e}')
        try:
            exports['checkpoint'] = self.export_weights(output_dir / f'{model_name}_checkpoint.pth', include_optimizer=optimizer is not None, optimizer=optimizer)
        except Exception as e:
            logger.error(f'Checkpoint export failed: {e}')
        try:
            exports['config'] = self.export_config(output_dir / f'{model_name}_config.json')
        except Exception as e:
            logger.error(f'Config export failed: {e}')
        logger.info(f'Exported {len(exports)} formats to {output_dir}')
        return exports

def load_checkpoint(checkpoint_path: Path, model: nn.Module) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f'Loaded checkpoint from {checkpoint_path}')
    return model

def load_checkpoint_with_optimizer(checkpoint_path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[nn.Module, torch.optim.Optimizer]:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f'Loaded model and optimizer from {checkpoint_path}')
    else:
        logger.warning(f'No optimizer state in checkpoint {checkpoint_path}')
    return (model, optimizer)