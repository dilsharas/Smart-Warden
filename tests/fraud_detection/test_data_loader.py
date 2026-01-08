"""
Unit tests for TransactionDataLoader
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.fraud_detection.data_loader import TransactionDataLoader


@pytest.fixture
def loader():
    """Create a data loader instance."""
    return TransactionDataLoader()


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample data."""
    data = {
        'sender': ['0x123', '0x456', '0x789'],
        'receiver': ['0xabc', '0xdef', '0xghi'],
        'value': [1.5, 2.3, 0.8],
        'gas_used': [21000, 21000, 21000],
        'timestamp': [1000000, 1000001, 1000002],
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_json_file():
    """Create a temporary JSON file with sample data."""
    data = [
        {'sender': '0x123', 'receiver': '0xabc', 'value': 1.5, 'gas_used': 21000, 'timestamp': 1000000},
        {'sender': '0x456', 'receiver': '0xdef', 'value': 2.3, 'gas_used': 21000, 'timestamp': 1000001},
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


class TestTransactionDataLoader:
    """Test cases for TransactionDataLoader."""

    def test_load_csv(self, loader, sample_csv_file):
        """Test loading CSV file."""
        df = loader.load_csv(sample_csv_file)
        assert len(df) == 3
        assert list(df.columns) == ['sender', 'receiver', 'value', 'gas_used', 'timestamp']

    def test_load_json(self, loader, sample_json_file):
        """Test loading JSON file."""
        df = loader.load_json(sample_json_file)
        assert len(df) == 2
        assert 'sender' in df.columns

    def test_validate_schema_valid(self, loader, sample_csv_file):
        """Test schema validation with valid data."""
        df = loader.load_csv(sample_csv_file)
        assert loader.validate_schema(df) is True

    def test_validate_schema_missing_fields(self, loader):
        """Test schema validation with missing fields."""
        df = pd.DataFrame({'sender': ['0x123'], 'receiver': ['0xabc']})
        with pytest.raises(ValueError, match="Missing required fields"):
            loader.validate_schema(df)

    def test_get_required_fields(self, loader):
        """Test getting required fields."""
        fields = loader.get_required_fields()
        assert 'sender' in fields
        assert 'receiver' in fields
        assert 'value' in fields
        assert 'gas_used' in fields
        assert 'timestamp' in fields

    def test_load_and_validate(self, loader, sample_csv_file):
        """Test load and validate in one step."""
        df = loader.load_and_validate(sample_csv_file)
        assert len(df) == 3
        assert loader.validate_schema(df) is True

    def test_file_not_found(self, loader):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            loader.load_csv('nonexistent_file.csv')

    def test_unsupported_format(self, loader):
        """Test error handling for unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                loader.load_and_validate(temp_path)
        finally:
            os.unlink(temp_path)

    def test_get_data_summary(self, loader, sample_csv_file):
        """Test getting data summary."""
        loader.load_csv(sample_csv_file)
        summary = loader.get_data_summary()
        assert summary['rows'] == 3
        assert 'columns' in summary
        assert 'dtypes' in summary
