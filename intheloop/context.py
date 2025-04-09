"""
This module handles gathering and formatting context from the IPython environment
for use by a model.
"""

from typing import TypedDict, Dict, Any, List, Optional, cast, Tuple
from types import ModuleType
import sys
import inspect
from dataclasses import dataclass
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.formatters import DisplayFormatter
import base64
from PIL import Image
from io import BytesIO

from .messages import TextContent, ImageContent

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

@dataclass
class OutputInfo:
    """Information about cell output"""
    output_type: str
    content: List[TextContent | ImageContent]

class NotebookContext:
    """Gathers and manages context from the IPython environment"""
    
    def __init__(self, shell: InteractiveShell):
        self.shell = shell
        self._pandas_available = 'pandas' in sys.modules
        self._numpy_available = 'numpy' in sys.modules

    def _process_output_data(self, output_data: Dict[str, Any]) -> List[TextContent | ImageContent]:
        """Process output data into appropriate content types"""
        contents = []
        
        # Handle text/plain
        if 'text/plain' in output_data:
            contents.append(TextContent(text=str(output_data['text/plain'])))
            
        # Handle image/png
        if 'image/png' in output_data:
            img_data = output_data['image/png']
            if isinstance(img_data, str):
                # If it's already base64 encoded
                image_url = f"data:image/png;base64,{img_data}"
            else:
                # If it's bytes, encode it
                image_url = f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
            contents.append(ImageContent(image_url=image_url))
            
        return contents

    def _get_rich_output(self, obj: Any) -> Dict[str, Any]:
        """Get rich output formats for an object using IPython's display formatter"""
        if not hasattr(self.shell, 'display_formatter'):
            return {}
            
        formatter = cast(DisplayFormatter, self.shell.display_formatter)
        result = formatter.format(obj)
        if result:
            data, metadata = result
            return data
        return {}

    def _process_notebook_output(self, output: Dict[str, Any]) -> List[TextContent | ImageContent]:
        """Process output from a notebook cell"""
        contents = []
        
        # Handle different output types
        output_type = output.get('output_type')
        
        if output_type == 'execute_result' or output_type == 'display_data':
            if 'data' in output:
                contents.extend(self._process_output_data(output['data']))
        elif output_type == 'stream':
            if 'text' in output:
                contents.append(TextContent(text=output['text']))
                
        return contents

    def get_output_history(self, n_entries: int = 5) -> List[OutputInfo]:
        """Get the output history including display data"""
        outputs = []
        
        # First try to get outputs from the current notebook if available
        nb = self.shell.user_ns.get('nb')
        if nb is not None and hasattr(nb, 'cells'):
            for cell in nb.cells:
                if 'outputs' in cell:
                    for output in cell['outputs']:
                        contents = self._process_notebook_output(output)
                        if contents:
                            outputs.append(OutputInfo(
                                output_type=output.get('output_type', 'unknown'),
                                content=contents
                            ))
        
        # Then get outputs from the current session
        Out = self.shell.user_ns.get('Out', {})
        current_execution = self.shell.execution_count
        
        # Look at recent outputs
        for i in range(current_execution - n_entries, current_execution + 1):
            if i in Out:
                output = Out[i]
                contents = []
                
                # Try using IPython's display formatter first
                data = self._get_rich_output(output)
                if data:
                    contents.extend(self._process_output_data(data))
                # Special handling for matplotlib figures
                elif hasattr(output, 'get_figure'):  # For matplotlib axes
                    fig_data = self._get_rich_output(output.get_figure())
                    contents.extend(self._process_output_data(fig_data))
                elif hasattr(output, 'canvas'):  # For matplotlib figures
                    fig_data = self._get_rich_output(output)
                    contents.extend(self._process_output_data(fig_data))
                # Fallback to string representation
                elif hasattr(output, '__str__'):
                    contents.append(TextContent(text=str(output)))
                
                if contents:
                    outputs.append(OutputInfo(
                        output_type='execute_result',
                        content=contents
                    ))
                        
        return outputs
        
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
            'imported_modules': list(self.get_imported_modules().keys()),
            'outputs': self.get_output_history()
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
