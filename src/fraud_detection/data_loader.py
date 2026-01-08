"""
Transaction Data Loader Module

Handles loading and validation of blockchain transaction data from CSV and JSON files.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import REQUIRED_FIELDS, OPTIONAL_FIELDS

logger = logging.getLogger(__name__)


class TransactionDataLoader:
    """
    Loads and validates blockchain transaction data from various file formats.
    
    Supports CSV and JSON formats with comprehensive schema validation.
    """

    def __init__(self):
        """Initialize the data loader."""
        self.required_fields = REQUIRED_FIELDS
        self.optional_fields = OPTIONAL_FIELDS
        self.loaded_data: Optional[pd.DataFrame] = None

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load transaction data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame containing transaction data
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is empty or has invalid format
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded CSV file: {filepath} with {len(df)} rows")
            self.loaded_data = df
            return df
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {filepath}")
        except Exception as e:
            raise ValueError(f"Error reading CSV file {filepath}: {str(e)}")

    def load_json(self, filepath: str) -> pd.DataFrame:
        """
        Load transaction data from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            DataFrame containing transaction data
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is empty or has invalid format
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("JSON must contain a list of objects or a single object")
            
            logger.info(f"Loaded JSON file: {filepath} with {len(df)} rows")
            self.loaded_data = df
            return df
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {filepath}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file {filepath}: {str(e)}")

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame contains all required fields.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if schema is valid
            
        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = [field for field in self.required_fields if field not in df.columns]
        
        if missing_fields:
            raise ValueError(
                f"Missing required fields: {missing_fields}. "
                f"Required fields: {self.required_fields}"
            )
        
        logger.info(f"Schema validation passed. Found all required fields: {self.required_fields}")
        return True

    def get_required_fields(self) -> List[str]:
        """
        Get the list of required fields for transaction data.
        
        Returns:
            List of required field names
        """
        return self.required_fields.copy()

    def get_optional_fields(self) -> List[str]:
        """
        Get the list of optional fields for transaction data.
        
        Returns:
            List of optional field names
        """
        return self.optional_fields.copy()

    def load_and_validate(self, filepath: str) -> pd.DataFrame:
        """
        Load data from file and validate schema in one step.
        
        Args:
            filepath: Path to the data file (CSV or JSON)
            
        Returns:
            Validated DataFrame
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If schema is invalid
        """
        filepath = Path(filepath)
        
        # Determine file type and load accordingly
        if filepath.suffix.lower() == '.csv':
            df = self.load_csv(str(filepath))
        elif filepath.suffix.lower() == '.json':
            df = self.load_json(str(filepath))
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}. Use .csv or .json")
        
        # Validate schema
        self.validate_schema(df)
        
        logger.info(f"Successfully loaded and validated data from {filepath}")
        return df

    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            Dictionary with data summary information
        """
        if self.loaded_data is None:
            return {"status": "No data loaded"}
        
        return {
            "rows": len(self.loaded_data),
            "columns": list(self.loaded_data.columns),
            "dtypes": self.loaded_data.dtypes.to_dict(),
            "missing_values": self.loaded_data.isnull().sum().to_dict(),
            "numeric_summary": self.loaded_data.describe().to_dict(),
        }
