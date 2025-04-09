"""In-the-loop: generative AI tooling for IPython and Jupyter notebooks

## Usage

  `%load_ext intheloop` to load the extension in other notebook
  projects
"""

from .recommendations import register as register_exception_handler
from .magic import load_ipython_extension as register_magics

def load_ipython_extension(ipython):
    """Register both the exception handler and magic commands."""
    register_exception_handler(ipython)
    register_magics(ipython)

def unload_ipython_extension(ipython):
    """Unload both the exception handler and magic commands."""
    # Unload the custom exception handler
    ipython.set_custom_exc((Exception,), None)
    # Magic commands are automatically unloaded by IPython