"""
This module implements IPython magic commands for AI assistance with context awareness.
"""

from typing import Dict, Any, Optional, cast, List, Union
from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic)
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, Markdown, DisplayHandle, HTML, Image
from openai import OpenAI

from .context import NotebookContext, OutputInfo
from .messages import create_system_message, create_user_message, Message, TextContent, ImageContent
from openai.types.responses.response_input_param import ResponseInputItemParam

MessageContent = Union[TextContent, ImageContent]

@magics_class
class AIContextMagics(Magics):
    """Magic commands for AI assistance with notebook context awareness."""
    
    def __init__(self, shell: Any) -> None:
        # Cast shell to InteractiveShell since we know Magics only works with it
        super().__init__(cast(InteractiveShell, shell))
        self.client = OpenAI()
        self.shell = cast(InteractiveShell, shell)

    def _format_outputs_for_display(self, outputs: List[OutputInfo]) -> str:
        """Format outputs for display in the details view"""
        parts = []
        for output in outputs:
            for content in output.content:
                if isinstance(content, TextContent):
                    parts.append(f"Text output: {content.text}")
                elif isinstance(content, ImageContent):
                    parts.append("[Image output included]")
        return "\n".join(parts)

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
        
        # Get output history
        outputs = context.get_output_history()
        
        # Default system message
        system_msg = args.system or (
            "You are a helpful AI data assistant with access to the current Jupyter/IPython notebook context. "
            "Use this context to provide more relevant and accurate responses."
        )
        
        # Create base message content
        message_content: List[MessageContent] = [TextContent(text=f"Current Notebook Context:\n{context_str}\n\nUser Query:\n{cell}")]
        
        # Add any output content
        for output in outputs:
            message_content.extend(output.content)
        
        # Create messages
        messages: List[ResponseInputItemParam] = [
            create_system_message(system_msg),
            create_user_message(message_content)
        ]

        # Display the context in a collapsible details element
        details_html = """
        <details>
            <summary>Sent context for the model</summary>
            <div style="padding: 16px; border-radius: 6px;">
                <pre style="white-space: pre-wrap; font-family: monospace;">
{text_content}
                </pre>
                <div style="margin-top: 10px;">
                    <strong>Outputs included:</strong>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
{image_previews}
                    </div>
                </div>
            </div>
        </details>
        """
        
        # Collect text and images
        text_parts = []
        image_previews = []
        
        for content in message_content:
            if isinstance(content, TextContent):
                text_parts.append(content.text)
            elif isinstance(content, ImageContent):
                image_previews.append(
                    f'<img src="{content.image_url}" '
                    'style="max-width: 200px; max-height: 200px; object-fit: contain;">'
                )
        
        display(HTML(details_html.format(
            text_content="\n".join(text_parts),
            image_previews="\n".join(image_previews)
        )))
        
        # Create a placeholder for streaming output
        display_handle = display(Markdown("_Thinking..._"), display_id=True)
        
        try:
            response = self.client.responses.create(
                model=args.model,
                input=messages,
                stream=True,
                max_tokens=4096
            )
            
            # Stream the response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
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