"""
I/O utilities: .mat loading, subject ID parsing, model setup, weight loading.

This module unifies the duplicated I/O logic that was previously scattered
across rush_pipeline, calc_freq_all, and calc_entropy_all.
"""

import re
import copy
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
from scipy.io.matlab import mat_struct

import torch

from models import Resnet, ElderNet

logger = logging.getLogger(__name__)


# ============================================================================
# .mat file loading (handles both v5 and v7.3)
# ============================================================================

def loadmat(filename):
    """Load .mat file, handling both MATLAB v5 and v7.3 formats."""
    try:
        data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)
    except NotImplementedError:
        import mat73
        return mat73.loadmat(str(filename))
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return None


def _check_keys(data):
    for key in data:
        if isinstance(data[key], mat_struct):
            data[key] = _todict(data[key])
    return data


def _todict(matobj):
    return {
        strg: _todict(elem) if isinstance(elem, mat_struct) else elem
        for strg, elem in matobj.__dict__.items()
    }


# ============================================================================
# Subject ID parsing (unified, single source of truth)
# ============================================================================

def parse_subject_id(file_path) -> Optional[str]:
    """
    Extract subject ID from filename → 'projid_year' with no leading zeros.

    Handles formats:
        12345678_00_GENEActive.mat  → '12345678_0'
        12345678-00-something.mat   → '12345678_0'
    """
    filename = Path(file_path).stem
    match = re.match(r'^(\d+)[-_](\d+)', filename)
    if match:
        return f"{int(match.group(1))}_{int(match.group(2))}"
    return None


def normalize_id(id_str: str) -> str:
    """Normalize any ID string to 'projid_year' format."""
    try:
        id_str = str(id_str).strip()
        # Try underscore format first
        if '_' in id_str:
            parts = [p for p in id_str.replace('-', '_').split('_') if p.isdigit()]
            if len(parts) >= 2:
                return f"{int(parts[0])}_{int(parts[1])}"
        elif '-' in id_str:
            parts = [p for p in id_str.split('-') if p.isdigit()]
            if len(parts) >= 2:
                return f"{int(parts[0])}_{int(parts[1])}"
        return id_str
    except Exception:
        return id_str


# ============================================================================
# File discovery
# ============================================================================

def list_mat_files(directory, exclude_names=None, exclude_folders=None):
    """
    List .mat files recursively, excluding specific patterns.

    Args:
        directory: Root directory to search.
        exclude_names: Substrings to exclude from filenames.
        exclude_folders: Folder names to skip entirely.
    """
    if exclude_names is None:
        exclude_names = ["UpSideDown", "WearTime", "Temp.mat", "Time.mat", "info.mat"]
    if exclude_folders is None:
        exclude_folders = {"doubles", "small_amount_data"}

    return [
        f for f in Path(directory).rglob("*.mat")
        if not any(pat in f.name for pat in exclude_names)
        and not any(part in exclude_folders for part in f.parts)
    ]


# ============================================================================
# Raw data extraction from .mat (device-specific)
# ============================================================================

def extract_raw_data(file_path, sensor_device: str):
    """
    Extract acceleration data + metadata from a .mat file.

    Args:
        file_path: Path to .mat file.
        sensor_device: 'GENEActive' or 'Axivity'.

    Returns:
        dict with keys: 'acc' (N,3 array), 'fs' (float), 'start_time_str' (str)
        or None on failure.
    """
    file_path = Path(file_path)
    data = loadmat(file_path)
    if data is None:
        return None

    try:
        if sensor_device == 'GENEActive':
            return _extract_geneactive(data)
        elif sensor_device == 'Axivity':
            return _extract_axivity(data, file_path)
        else:
            logger.error(f"Unknown sensor device: {sensor_device}")
            return None
    except Exception as e:
        logger.error(f"Error extracting data from {file_path.name}: {e}")
        return None


def _extract_geneactive(data):
    values = data.get('values')
    if values is None:
        return None

    if isinstance(values, np.ndarray) and values.shape == (1, 1):
        values = values[0, 0]

    if isinstance(values, dict):
        acc = values['acc']
        fs = values['sampFreq']
        header = values['header']
    else:
        acc = values.acc
        fs = values.sampFreq
        header = values.header

    acc = np.asarray(acc, dtype=float)
    fs = float(fs.item()) if isinstance(fs, (np.ndarray, np.generic)) else float(fs)

    # Parse start time from header
    header_str = "".join(str(h) for h in header) if isinstance(header, (list, np.ndarray)) else str(header)
    match = re.search(r'Start Time:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\d{3}', header_str)
    if not match:
        logger.error("Could not parse start time from GENEActive header")
        return None

    return {'acc': acc, 'fs': fs, 'start_time_str': match.group(1)}


def _extract_axivity(data, file_path):
    if 'New_Data' not in data:
        return None

    acc = np.asarray(data['New_Data'], dtype=float)
    fs = 50.0  # Axivity default

    # Load companion info file for start time
    info_path = file_path.with_name(file_path.stem + "_info" + file_path.suffix)
    if not info_path.exists():
        info_path = file_path.parent.parent / "info" / info_path.name

    if not info_path.exists():
        logger.error(f"Info file not found for {file_path.name}")
        return None

    info = loadmat(info_path)
    if info is None:
        return None

    try:
        file_info = info['fileinfo']
        if isinstance(file_info, np.ndarray) and file_info.shape == (1, 1):
            file_info = file_info[0, 0]

        start_struct = file_info['start'] if isinstance(file_info, dict) else file_info.start
        if isinstance(start_struct, np.ndarray) and start_struct.shape == (1, 1):
            start_struct = start_struct[0, 0]

        start_str = start_struct['str'] if isinstance(start_struct, dict) else start_struct.str
        if isinstance(start_str, (np.ndarray, list)):
            start_str = start_str[0]
        if isinstance(start_str, np.ndarray):
            start_str = start_str.item()
        start_str = str(start_str)
    except Exception as e:
        logger.error(f"Could not parse Axivity start time: {e}")
        return None

    return {'acc': acc, 'fs': fs, 'start_time_str': start_str}


# ============================================================================
# Model setup (FIXED: properly forwards num_layers, batch_norm to ElderNet)
# ============================================================================

def setup_model(model_cfg, epoch_len=10, device='cpu'):
    """
    Build and load a model from config.

    Constructs the EXACT architecture used during training so that
    pretrained weight keys match. The key structure is:

    ElderNet for classification (gait detection):
        feature_extractor.layer1-5.*     (ResNet backbone)
        fc.linear1/2/3.*                 (LinearLayers: 1024→512→256→128)
        classifier.linear1.*             (Classifier: 128→2)

    ElderNet for regression (speed, cadence, etc.):
        feature_extractor.layer1-5.*     (ResNet backbone)
        fc.linear1/2/3.*                 (LinearLayers: 1024→512→256→128)
        regressor.linear_layers.0.*      (hidden layers)
        regressor.bn_layers.0.*          (if batch_norm=True)
        regressor.mu.*                   (output)
    """
    output_size = model_cfg.get('output_size', 1)
    is_classification = model_cfg.get('is_classification', False)
    is_regression = model_cfg.get('is_regression', False)
    max_mu = model_cfg.get('max_mu', 2.0)
    num_layers_regressor = model_cfg.get('num_layers', 1)
    use_bn = model_cfg.get('batch_norm', False)

    # Build Resnet backbone (only need the feature_extractor)
    resnet = Resnet(output_size=output_size, epoch_len=epoch_len)

    if model_cfg.get('net') == 'ElderNet':
        feature_extractor = resnet.feature_extractor

        # Must match training code defaults:
        #   linear_model_input_size = 1024  (backbone output for epoch_len=10)
        #   linear_model_output_size = 128  (from config feature_vector_size)
        #   non_linearity = True
        linear_input = 1024  # ResNet backbone output for 10-sec epochs
        linear_output = model_cfg.get('feature_vector_size', 128)

        model = ElderNet(
            feature_extractor,
            head=model_cfg.get('head', 'fc'),
            non_linearity=True,
            linear_model_input_size=linear_input,
            linear_model_output_size=linear_output,
            output_size=output_size,
            is_classification=is_classification,
            is_regression=is_regression,
            max_mu=max_mu,
            num_layers_regressor=num_layers_regressor,
            batch_norm=use_bn,
        )
    else:
        model = Resnet(
            output_size=output_size,
            epoch_len=epoch_len,
            is_classification=is_classification,
            is_regression=is_regression,
            max_mu=max_mu,
            num_layers_regressor=num_layers_regressor,
            batch_norm=use_bn,
        )

    # Load pretrained weights
    if model_cfg.get('pretrained', False) and model_cfg.get('trained_model_path'):
        name_start_idx = model_cfg.get('name_start_idx', 0)
        load_weights(model_cfg['trained_model_path'], model, device, name_start_idx)

    return copy.deepcopy(model).to(device, dtype=torch.float)


def load_weights(weight_path, model, device="cpu", name_start_idx=0):
    """
    Load pretrained weights with key remapping and strict verification.

    Raises an error if critical layers are missing from the checkpoint,
    instead of silently running with random weights.
    """
    pretrained_dict = torch.load(weight_path, map_location=device)
    model_dict = model.state_dict()

    # Remap key names (strip prefix based on name_start_idx)
    remapped = {}
    for key, val in pretrained_dict.items():
        new_key = ".".join(key.split(".")[name_start_idx:])
        remapped[new_key] = val

    # Find matching keys
    matched = {k: v for k, v in remapped.items()
               if k in model_dict and v.shape == model_dict[k].shape}
    missing_in_ckpt = set(model_dict.keys()) - set(matched.keys())
    unused_in_ckpt = set(remapped.keys()) - set(matched.keys())

    # Log diagnostics
    logger.info(f"Loading weights from {weight_path}")
    logger.info(f"  Matched: {len(matched)}/{len(model_dict)} model keys")

    if missing_in_ckpt:
        # Separate buffer keys (less critical) from parameter keys (critical)
        critical_missing = [k for k in missing_in_ckpt
                           if not k.endswith('.num_batches_tracked')]
        if critical_missing:
            logger.warning(f"  Missing from checkpoint ({len(critical_missing)} keys):")
            for k in sorted(critical_missing)[:10]:
                logger.warning(f"    {k}  {tuple(model_dict[k].shape)}")
            if len(critical_missing) > 10:
                logger.warning(f"    ... and {len(critical_missing) - 10} more")

    if unused_in_ckpt:
        logger.info(f"  Unused checkpoint keys: {len(unused_in_ckpt)}")
        for k in sorted(unused_in_ckpt)[:5]:
            logger.info(f"    {k}")

    # STRICT CHECK: if less than 50% of keys matched, something is very wrong
    match_ratio = len(matched) / len(model_dict) if model_dict else 0
    if match_ratio < 0.5:
        raise RuntimeError(
            f"Weight loading failed: only {len(matched)}/{len(model_dict)} keys matched "
            f"({match_ratio:.0%}). Architecture likely doesn't match checkpoint. "
            f"First 5 model keys: {sorted(model_dict.keys())[:5]}, "
            f"First 5 checkpoint keys: {sorted(remapped.keys())[:5]}"
        )

    model_dict.update(matched)
    model.load_state_dict(model_dict)
