"""
This module handles gathering and formatting context from the IPython environment
for use by a model.
"""

from typing import Dict, Any, List, Optional
from types import ModuleType
import sys
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
                try:
                    # For numeric arrays, include min/max
                    if np.issubdtype(obj.dtype, np.number):
                        summary = f"shape={obj.shape}, min={obj.min():.2f}, max={obj.max():.2f}"
                    else:
                        # For non-numeric arrays (e.g. strings), just show shape and dtype
                        summary = f"shape={obj.shape}, dtype={obj.dtype}"
                        
                    info = VariableInfo(
                        name=name,
                        type_name=str(obj.dtype),
                        summary=summary,
                        size=obj.nbytes
                    )
                    arrays.append(info)
                except Exception as e:
                    # Skip problematic arrays but log the error
                    print(f"Error gathering info for array {name}: {e}")
                
        return arrays

    def get_in_out_history(self, n_entries: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent In/Out history, excluding %%ai commands"""
        history = []
        
        # Try multiple sources for input/output history
        In = self.shell.user_ns.get('In', self.shell.user_ns.get('_ih', []))
        Out = self.shell.user_ns.get('Out', self.shell.user_ns.get('_oh', {}))
        
        if not In:  # If we still don't have input history
            return history

        # Start from the most recent and work backwards
        for i in range(len(In) - 1, max(-1, len(In) - n_entries - 1), -1):
            input_cmd = In[i]
            # Skip empty inputs and %%ai commands
            if not input_cmd or input_cmd.strip().startswith('%%ai'):
                continue
            
            # Get output if it exists
            output = Out.get(i, '')
            
            # Only include non-empty outputs or inputs that aren't magic commands
            if output or not (input_cmd.startswith('%') or input_cmd.startswith('get_ipython()')):
                entry = {
                    'In': input_cmd,
                    'Out': output
                }
                history.append(entry)
            
        return history

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
            'in_out_history': self.get_in_out_history(),
            'namespace': self.get_current_namespace_summary(),
            'imported_modules': list(self.get_imported_modules().keys())
        }

    def format_context_for_prompt(self) -> str:
        """Format context information into a string suitable for model prompts"""
        context = self.format_context()
        sections = []
        
        if context['dataframes']:
            df_info = "\n".join(
                f"- {df.name}: {df.shape}, columns={df.columns}"
                for df in context['dataframes']
            )
            sections.append(f"DataFrames:\n{df_info}")
            
        if context['arrays']:
            array_info = "\n".join(
                f"- {arr.name}: {arr.summary}"
                for arr in context['arrays']
            )
            sections.append(f"NumPy Arrays:\n{array_info}")
            
        if context['in_out_history']:
            history_entries = []
            for i, entry in enumerate(reversed(context['in_out_history'])):
                history_entries.append(f"In[{i}]: {entry['In']}")
                if entry['Out']:  # Only show Out if there's actual output
                    history_entries.append(f"Out[{i}]: {entry['Out']}")
            history = "\n".join(history_entries)
            sections.append(f"Recent In/Out History:\n{history}")
            
        if context['imported_modules']:
            modules = ", ".join(context['imported_modules'])
            sections.append(f"Imported Modules: {modules}")
            
        return "\n\n".join(sections)
