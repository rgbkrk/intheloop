"""
This module creates a custom exception handler that will send the error to a model in order to help debug the code.
It will also display the error in the notebook as usual.
"""

from traceback import TracebackException
from types import TracebackType
from typing import Iterable, Type, TypedDict, List, Dict, Any

from IPython.core.interactiveshell import InteractiveShell
from IPython.core.getipython import get_ipython

from spork import Markdown

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam

from .context import NotebookContext


class InTheLoop:
    def __init__(self):
        self.client = OpenAI()
        self.messages = []

    def gather_context(self, shell: InteractiveShell) -> Dict[str, Any]:
        """Gather current notebook context"""
        context = NotebookContext(shell)
        return context.format_context()

    def form_exception_messages(self, code: str | None, etype: Type[BaseException], 
                              evalue: BaseException, plaintext_traceback: str,
                              context: Dict[str, Any]) -> Iterable[ChatCompletionMessageParam]:
        """Enhanced message formation with context"""
        code_str = str(code) if code is not None else "<no code available>"
        
        # Format the context information
        context_str = self._format_context_for_message(context)

        return [ChatCompletionSystemMessageParam(
            role="system",
            content=(
                f"Current Notebook Context:\n{context_str}\n\n"
                f"Error occurred in:\nIn[#]: {code_str}\n\n"
                f"Error:\n{evalue}\n\n"
                f"Traceback:\n{plaintext_traceback}"
            )
        )]


    def _format_context_for_message(self, context: Dict[str, Any]) -> str:
        """Format context information for inclusion in messages"""
        sections = []
        
        if context['dataframes']:
            df_info = "\n".join(
                f"- {df.name}: {df.shape}, columns={list(df.dtypes.keys())}"
                for df in context['dataframes']
            )
            sections.append(f"DataFrames:\n{df_info}")
            
        if context['arrays']:
            array_info = "\n".join(
                f"- {arr.name}: {arr.summary}"
                for arr in context['arrays']
            )
            sections.append(f"NumPy Arrays:\n{array_info}")
            
        if context['recent_history']:
            history = "\n".join(
                f"{i+1}. {cmd}" for i, cmd in enumerate(context['recent_history'])
            )
            sections.append(f"Recent Commands:\n{history}")
            
        if context['imported_modules']:
            modules = ", ".join(context['imported_modules'])
            sections.append(f"Imported Modules: {modules}")
            
        return "\n\n".join(sections)

    def custom_exc(
        self,
        shell: "InteractiveShell",
        etype: Type[BaseException],
        evalue: BaseException,
        tb: TracebackType,
        tb_offset=None,
    ):
        # still show the error within the notebook, don't just swallow it
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

        # On interrupt, just let it be
        if etype == KeyboardInterrupt:
            return
        # On exit, release the user
        elif etype == SystemExit:
            return

        try:
            code = self._get_current_code(shell)
            context = self.gather_context(shell)
            
            gm = Markdown(content="Analyzing context and seeking solution...")
            gm.display()
            
            formatted = TracebackException(etype, evalue, tb, limit=3).format(chain=True)
            plaintext_traceback = "\n".join(formatted)

            messages = self.form_exception_messages(
                code, etype, evalue, plaintext_traceback, context
            )
            gm.content = ""

            # Display the messages in a collapsible details element
            details_html = """
            <details>
                <summary>Sent context for the model</summary>
                <pre style="white-space: pre-wrap; padding: 16px; border-radius: 6px; font-family: monospace;">
{content}
                </pre>
            </details>
            """.format(content="\n".join(str(m['content']) for m in messages if 'content' in m))
            
            from IPython.display import display, HTML
            display(HTML(details_html))
            
            resp = self.client.chat.completions.create(
                model='gpt-4-turbo-preview',  # Using latest model for better context understanding
                messages=messages,
                tools=[],
                tool_choice='auto',
                stream=True,
            )

            gm = Markdown(content="Investigating...")
            gm.display()

            for chunk in resp:
                if chunk.choices[0].delta.content is not None:
                    gm.append(chunk.choices[0].delta.content)

        except Exception as e:
            print("Error while trying to provide a suggestion: ", e)
        except KeyboardInterrupt:
            
            if "gm" in locals():
                gm.append("\n\n> **Interrupted** ⚠️")

    def _get_current_code(self, shell: InteractiveShell) -> str | None:
        execution_count = shell.execution_count
        In = shell.user_ns["In"]
        history_manager = shell.history_manager

        # If the history is available, use that as it has the raw inputs (including magics)
        if (history_manager is not None) and (
            execution_count == len(history_manager.input_hist_raw) - 1
        ):
            return history_manager.input_hist_raw[execution_count]
        # Fallback on In
        elif In is not None and execution_count == len(In) - 1:
            # Otherwise, use the current input buffer
            return In[execution_count]
        # Otherwise history may not have been stored (store_history=False), so we should not send the
        # code to GPT.
        else:
            return None


def register(ipython=None):
    """Register the exception handler with the given IPython instance."""
    ipython = ipython or get_ipython()
    if not ipython:
        return

    itl = InTheLoop()
        
    ipython.set_custom_exc((Exception,), itl.custom_exc)