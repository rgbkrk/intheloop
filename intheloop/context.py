"""
This module handles gathering and formatting context from the IPython environment
for use by a model.
"""

from typing import TypedDict, Dict, Any, List, Optional
from types import ModuleType
import sys
import inspect
from dataclasses import dataclass
from IPython.core.interactiveshell import InteractiveShell

@dataclass
class DataFrameInfo:
    """Information about a pandas DataFrame in the namespace"""
    name: str
    shape: tuple
    dtypes: Dict[str, str]
    sample: str # Will contain DataFrame.head() in a text format
    memory_usage: Dict[str, int]
    columns: List[str]

@dataclass
class VariableInfo:
    """Information about a variable in the namespace"""
    name: str
    type_name: str
    summary: str
    size: Optional[int] = None

class NotebookContext:
    """Gathers and manages context from the IPython environment"""
    
    def __init__(self, shell: InteractiveShell):
        self.shell = shell
        self._pandas_available = 'pandas' in sys.modules
        self._numpy_available = 'numpy' in sys.modules
        
    def get_imported_modules(self) -> Dict[str, ModuleType]:
        """Get information about currently imported modules"""
        return {
            name: module for name, module in self.shell.user_ns.items()
            if isinstance(module, ModuleType)
        }

    def get_dataframe_info(self) -> List[DataFrameInfo]:
        """Gather information about pandas DataFrames in the namespace"""
        if not self._pandas_available:
            return []
            
        import pandas as pd
        dataframes = []
        
        for name, obj in self.shell.user_ns.items():
            if isinstance(obj, pd.DataFrame):
                try:
                    df_info = DataFrameInfo(
                        name=name,
                        shape=obj.shape,
                        dtypes={str(k): str(v) for k, v in obj.dtypes.items()},
                        sample=obj.sample(3).to_string(),
                        memory_usage=obj.memory_usage(deep=True).to_dict(),
                        columns=obj.columns.tolist()
                    )
                    dataframes.append(df_info)
                except Exception as e:
                    # Log error but continue with other DataFrames
                    print(f"Error gathering info for DataFrame {name}: {e}")
                    
        return dataframes

    def get_array_info(self) -> List[VariableInfo]:
        """Gather information about numpy arrays in the namespace"""
        if not self._numpy_available:
            return []
            
        import numpy as np # type: ignore
        arrays = []
        
        for name, obj in self.shell.user_ns.items():
            if isinstance(obj, np.ndarray):
                info = VariableInfo(
                    name=name,
                    type_name=str(obj.dtype),
                    summary=f"shape={obj.shape}, min={obj.min():.2f}, max={obj.max():.2f}",
                    size=obj.nbytes
                )
                arrays.append(info)
                
        return arrays

    def get_recent_history(self, n_entries: int = 5) -> List[str]:
        """Get the most recent command history"""
        if self.shell.history_manager:
            # Get the last n entries from history
            return [
                entry[2] for entry in self.shell.history_manager.get_tail(n_entries)
            ]
        return []

    def get_current_namespace_summary(self) -> List[VariableInfo]:
        """Get a summary of current variables in namespace"""
        summary = []
        for name, obj in self.shell.user_ns.items():
            # Skip modules, private vars, and IPython's internal vars
            if (name.startswith('_') or 
                isinstance(obj, ModuleType) or
                name in ['In', 'Out', 'exit', 'quit']):
                continue
                
            try:
                type_name = type(obj).__name__
                if hasattr(obj, 'shape'):  # For numpy arrays, pandas objects
                    summary_text = f"shape={obj.shape}"
                elif hasattr(obj, '__len__'):
                    summary_text = f"len={len(obj)}"
                else:
                    summary_text = str(obj)[:100]  # Truncate long strings
                    
                info = VariableInfo(
                    name=name,
                    type_name=type_name,
                    summary=summary_text
                )
                summary.append(info)
            except Exception:
                # Skip problematic variables
                continue
                
        return summary

    def format_context(self) -> Dict[str, Any]:
        """Format all context information into a dictionary"""
        return {
            'dataframes': self.get_dataframe_info(),
            'arrays': self.get_array_info(),
            'recent_history': self.get_recent_history(),
            'namespace': self.get_current_namespace_summary(),
            'imported_modules': list(self.get_imported_modules().keys())
        }
