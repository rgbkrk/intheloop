"""
This module implements IPython magic commands for AI assistance with context awareness.
"""

from typing import Dict, Any, Optional, cast
from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, Markdown, DisplayHandle, HTML
from openai import OpenAI
from .context import NotebookContext

@magics_class
class AIContextMagics(Magics):
    """Magic commands for AI assistance with notebook context awareness."""
    
    def __init__(self, shell: Any) -> None:
        # Cast shell to InteractiveShell since we know Magics only works with it
        super().__init__(cast(InteractiveShell, shell))
        self.client = OpenAI()
        self.shell = cast(InteractiveShell, shell)

    @magic_arguments()
    @argument('-m', '--model', default='gpt-4o',
              help='The model to use for completion')
    @argument('-s', '--system',
              help='Optional system message to override default')
    @cell_magic
    def ai(self, line: str, cell: str) -> None:
        """
        Magic that sends the cell contents to an AI model along with current notebook context.
        
        Usage:
            %%ai
            Your question or code here
            
        Options:
            -m, --model: Specify the model to use (default: gpt-4o)
            -s, --system: Provide a custom system message
        """
        args = parse_argstring(self.ai, line)
        
        # Gather context
        context = NotebookContext(self.shell)
        context_str = context.format_context_for_prompt()
        
        # Default system message
        system_msg = args.system or (
            "You are a helpful AI assistant with access to the current notebook context. "
            "Use this context to provide more relevant and accurate responses."
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Current Notebook Context:\n{context_str}\n\nUser Query:\n{cell}"}
        ]

        # Display the context in a collapsible details element
        details_html = """
        <details>
            <summary>Sent context for the model</summary>
            <pre style="white-space: pre-wrap; padding: 16px; border-radius: 6px; font-family: monospace;">
{content}
            </pre>
        </details>
        """.format(content="\n".join(str(m['content']) for m in messages if 'content' in m))
        
        display(HTML(details_html))
        
        # Create a placeholder for streaming output
        display_handle = display(Markdown("_Thinking..._"), display_id=True)
        
        try:
            response = self.client.responses.create(
                model=args.model,
                input=messages,
                stream=True
            )
            
            # Stream the response
            full_response = ""
            for chunk in response:
                if chunk.type == "response.output_text.delta":
                    full_response += chunk.delta
                    if display_handle:
                        display_handle.update(Markdown(full_response))
                    
        except Exception as e:
            if display_handle:
                display_handle.update(Markdown(f"Error: {str(e)}"))

def load_ipython_extension(ipython: Optional[InteractiveShell]) -> None:
    """
    Register the magic when the extension is loaded.
    """
    if ipython is not None:
        ipython.register_magics(AIContextMagics) 