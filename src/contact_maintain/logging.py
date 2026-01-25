"""Logging utilities for contact maintenance experiments."""
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np


class DataLogger:
    """Logger for recording simulation/experiment data.
    
    Parameters
    ----------
    log_dir : str or Path, optional
        Directory to save log files. Default is current directory.
    experiment_name : str, optional
        Name prefix for the log files.
    """
    
    def __init__(self, log_dir=None, experiment_name="experiment"):
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Data storage
        self.data: Dict[str, list] = {}
        self.metadata: Dict[str, Any] = {}
        
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the log.
        
        Parameters
        ----------
        key : str
            Metadata key.
        value : Any
            Metadata value (must be JSON serializable or numpy array).
        """
        if isinstance(value, np.ndarray):
            value = value.tolist()
        self.metadata[key] = value
    
    def log(self, **kwargs):
        """Log data points.
        
        Parameters
        ----------
        **kwargs : dict
            Key-value pairs to log. Values can be scalars or numpy arrays.
        """
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            
            if isinstance(value, np.ndarray):
                value = value.tolist()
            
            self.data[key].append(value)
    
    def log_from_observer(self, observer):
        """Log data from a ContactObserver.
        
        Parameters
        ----------
        observer : ContactObserver
            Observer to extract data from.
        """
        state = observer.current_state
        self.log(
            in_contact=state.in_contact,
            force_magnitude=state.force_magnitude,
            contact_force=state.contact_force,
            contact_position=state.contact_position,
        )
    
    def get_filename(self, extension: str) -> Path:
        """Get a filename for saving.
        
        Parameters
        ----------
        extension : str
            File extension (without dot).
        
        Returns
        -------
        Path
            Full path to the file.
        """
        return self.log_dir / f"{self.experiment_name}_{self.timestamp}.{extension}"
    
    def save_pickle(self, filename: Optional[str] = None):
        """Save all data as a pickle file.
        
        Parameters
        ----------
        filename : str, optional
            Custom filename. If None, auto-generates.
        
        Returns
        -------
        Path
            Path to the saved file.
        """
        if filename:
            filepath = self.log_dir / filename
        else:
            filepath = self.get_filename("pkl")
        
        save_data = {
            'metadata': self.metadata,
            'data': self.data,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved data to {filepath}")
        return filepath
    
    def save_json(self, filename: Optional[str] = None):
        """Save all data as a JSON file.
        
        Parameters
        ----------
        filename : str, optional
            Custom filename. If None, auto-generates.
        
        Returns
        -------
        Path
            Path to the saved file.
        """
        if filename:
            filepath = self.log_dir / filename
        else:
            filepath = self.get_filename("json")
        
        save_data = {
            'metadata': self.metadata,
            'data': self.data,
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Saved data to {filepath}")
        return filepath
    
    def save_numpy(self, filename: Optional[str] = None):
        """Save data as numpy arrays in .npz file.
        
        Parameters
        ----------
        filename : str, optional
            Custom filename. If None, auto-generates.
        
        Returns
        -------
        Path
            Path to the saved file.
        """
        if filename:
            filepath = self.log_dir / filename
        else:
            filepath = self.get_filename("npz")
        
        # Convert lists to arrays
        arrays = {}
        for key, values in self.data.items():
            try:
                arrays[key] = np.array(values)
            except ValueError:
                # Skip if can't convert to array
                pass
        
        np.savez(filepath, **arrays, metadata=self.metadata)
        print(f"Saved data to {filepath}")
        return filepath
    
    def clear(self):
        """Clear all logged data."""
        self.data.clear()
    
    @staticmethod
    def load_pickle(filepath):
        """Load data from a pickle file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the pickle file.
        
        Returns
        -------
        dict
            Dictionary containing 'metadata' and 'data'.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_json(filepath):
        """Load data from a JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the JSON file.
        
        Returns
        -------
        dict
            Dictionary containing 'metadata' and 'data'.
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_numpy(filepath):
        """Load data from a numpy .npz file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the .npz file.
        
        Returns
        -------
        dict
            Dictionary of numpy arrays.
        """
        return dict(np.load(filepath, allow_pickle=True))

