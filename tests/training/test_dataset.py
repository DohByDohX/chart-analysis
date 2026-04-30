import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

def test_chart_dataset_init(tmp_path):
    # Mock modules because torch/PIL might be missing
    with patch.dict('sys.modules', {
        'torch': MagicMock(),
        'torch.utils.data': MagicMock(),
        'PIL': MagicMock(),
        'numpy': MagicMock(),
        'pandas': MagicMock(),
    }):
        # Mock torch.utils.data.Dataset manually to inherit from it
        import sys
        sys.modules['torch.utils.data'].Dataset = object

        from src.training.dataset import ChartDataset

        # Setup test data
        windows_dir = tmp_path / "windows"
        images_dir = tmp_path / "images"
        windows_dir.mkdir()
        images_dir.mkdir()

        for i in range(3):
            with open(windows_dir / f"window_{i:05d}.json", "w") as f:
                json.dump({"dummy": i}, f)

        tokenizer_mock = MagicMock()

        # Test dataset init
        dataset = ChartDataset(windows_dir, images_dir, tokenizer_mock)

        assert len(dataset.window_files) == 3
        assert len(dataset.window_data_cache) == 3
        assert dataset.get_window_data(0) == {"dummy": 0}
        assert dataset.get_window_data(1) == {"dummy": 1}
        assert dataset.get_window_data(2) == {"dummy": 2}

def test_chart_dataset_empty_dir(tmp_path):
    with patch.dict('sys.modules', {
        'torch': MagicMock(),
        'torch.utils.data': MagicMock(),
        'PIL': MagicMock(),
        'numpy': MagicMock(),
        'pandas': MagicMock(),
    }):
        import sys
        sys.modules['torch.utils.data'].Dataset = object

        from src.training.dataset import ChartDataset

        windows_dir = tmp_path / "windows"
        images_dir = tmp_path / "images"
        windows_dir.mkdir()
        images_dir.mkdir()

        tokenizer_mock = MagicMock()

        with pytest.raises(ValueError, match="No window files found"):
            ChartDataset(windows_dir, images_dir, tokenizer_mock)
