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
from openai.types.chat import ChatCompletionToolParam
from .tools import CreateCell
import json

@magics_class
class AIContextMagics(Magics):
    """Magic commands for AI assistance with notebook context awareness."""
    
    def __init__(self, shell: Any) -> None:
        # Cast shell to InteractiveShell since we know Magics only works with it
        super().__init__(cast(InteractiveShell, shell))
        self.client = OpenAI()
        self.shell = cast(InteractiveShell, shell)
        self.function_call_buffers = {}
        self.debug = False  # Debug flag

    def debug_print(self, message: str) -> None:
        """Print debug messages only if debug mode is enabled."""
        if self.debug:
            print(f"DEBUG: {message}")

    def error_print(self, message: str) -> None:
        """Print error messages only if debug mode is enabled."""
        if self.debug:
            print(f"ERROR: {message}")

    @magic_arguments()
    @argument('-m', '--model', default='gpt-4o',
              help='The model to use for completion')
    @argument('-s', '--system',
              help='Optional system message to override default')
    @argument('-d', '--debug', action='store_true',
              help='Enable debug logging')
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
            -d, --debug: Enable debug logging
        """
        args = parse_argstring(self.ai, line)
        self.debug = args.debug  # Set debug mode based on argument
        
        # Gather context
        context = NotebookContext(self.shell)
        
        # TODO: Gather images as the proper format for the OpenAI API to send _actual_ images over

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

        tools= []

        tools.append(CreateCell.function_schema())
        
        try:
            response = self.client.responses.create(
                model=args.model,
                input=messages,
                tools=tools,
                tool_choice='auto',
                stream=True
            )
            
            # Stream the response
            full_response = ""
            for chunk in response:
                if chunk.type == "response.output_text.delta":
                    full_response += chunk.delta
                    if display_handle:
                        display_handle.update(Markdown(full_response))
                else:
                    # Process non-text chunks (like function calls)
                    chunk_dict = chunk.model_dump()
                    self.debug_print(f"Processing chunk type: {chunk.type}")
                    
                    # Initialize function call when it starts
                    if chunk.type == "response.output_item.added" and chunk_dict.get('item', {}).get('type') == 'function_call':
                        item = chunk_dict['item']
                        call_id = item.get('id')
                        name = item.get('name')
                        self.debug_print(f"Initializing function call at start - id: {call_id}, name: {name}")
                        if call_id:
                            self.function_call_buffers[call_id] = {
                                'name': name,
                                'arguments': '',
                                'call_id': call_id
                            }
                            self.debug_print(f"Created initial function call buffer: {self.function_call_buffers[call_id]}")
                    
                    # Handle function call argument streaming
                    elif chunk.type == "response.function_call_arguments.delta":
                        # If this is part of a function call, append to its arguments
                        current_call_id = chunk_dict.get('item_id')
                        if current_call_id in self.function_call_buffers:
                            self.debug_print(f"Appending to function call arguments for call_id {current_call_id}")
                            self.function_call_buffers[current_call_id]['arguments'] += chunk_dict['delta']
                        else:
                            self.debug_print(f"No buffer found for delta with call_id {current_call_id}")
                        
                    # When a function call is complete, execute it
                    elif chunk.type == "response.function_call_arguments.done":
                        call_id = chunk_dict.get('item_id')
                        self.debug_print(f"Function call arguments complete for call_id {call_id}")
                        if call_id in self.function_call_buffers:
                            try:
                                func_call = self.function_call_buffers[call_id]
                                self.debug_print(f"Found function call buffer for execution: {func_call}")
                                # Parse the complete function call arguments
                                args = json.loads(func_call['arguments'])
                                self.debug_print(f"Successfully parsed arguments: {args}")
                                
                                # Execute based on the stored function name
                                if func_call['name'] == 'CreateCell' and 'cell' in args:
                                    self.debug_print(f"Executing CreateCell with cell content length: {len(args['cell'])}")
                                    # Create a new cell with the provided code
                                    self.shell.set_next_input(args['cell'], replace=False)
                                    self.debug_print("Called set_next_input")
                                    # Clear the display handle since we're creating a new cell
                                    if display_handle:
                                        display_handle.update(Markdown(""))
                                        self.debug_print("Cleared display handle")
                                else:
                                    self.debug_print(f"Function call conditions not met. name={func_call['name']}, has_cell={'cell' in args}")
                                
                                # Clean up the buffer
                                del self.function_call_buffers[call_id]
                                self.debug_print(f"Cleaned up buffer for call_id {call_id}")
                            except json.JSONDecodeError as e:
                                self.error_print(f"Failed to parse function call arguments: {e}")
                            except Exception as e:
                                self.error_print(f"Unexpected error during function execution: {str(e)}")
                                import traceback
                                self.error_print(f"Traceback: {traceback.format_exc()}")
                        else:
                            self.debug_print(f"No function call buffer found for call_id {call_id}")
                    else:
                        if self.debug:
                            print(f"Unhandled chunk type ({chunk.type}): {chunk_dict}")
                    
        except Exception as e:
            if display_handle:
                display_handle.update(Markdown(f"Error: {str(e)}"))

def load_ipython_extension(ipython: Optional[InteractiveShell]) -> None:
    """
    Register the magic when the extension is loaded.
    """
    if ipython is not None:
        ipython.register_magics(AIContextMagics) 