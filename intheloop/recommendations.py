"""
This module creates a custom exception handler that will send the error to OpenAI's
ChatGPT in order to help debug the code. It will also display the error in the
notebook as usual.
"""

from traceback import TracebackException
from types import TracebackType
from typing import Iterable, Type, TypedDict, List, Dict, Any

from IPython.core.interactiveshell import InteractiveShell
from IPython.core.getipython import get_ipython

from spork import Markdown

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class InTheLoop:
    def __init__(self):
        self.client = OpenAI()
        self.messages = []

    def form_exception_messages(self, code:str | None, etype:Type[BaseException], evalue:BaseException, plaintext_traceback:str) -> Iterable[ChatCompletionMessageParam]:

        # Plan:
        # Show that the user ran {code} and got the exception {etype} with value {evalue}.
        # The traceback should be trimmed down to 1024 characters.
        # It can show up as system messages entirely
        # We then return the messages to be used in the OpenAI API call

        code_str = str(code) if code is not None else "<no code available>"
        return [{
            "role": "system",
            "content": f"In[#]: {code_str}\n\nOut[#]: {evalue}\n\nTraceback:\n{plaintext_traceback}"
        }]
        
    # this function will be called on exceptions in any cell
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
            code = None

            execution_count = shell.execution_count

            In = shell.user_ns["In"]
            history_manager = shell.history_manager

            # If the history is available, use that as it has the raw inputs (including magics)
            if (history_manager is not None) and (
                execution_count == len(history_manager.input_hist_raw) - 1
            ):
                code = history_manager.input_hist_raw[execution_count]
            # Fallback on In
            elif In is not None and execution_count == len(In) - 1:
                # Otherwise, use the current input buffer
                code = In[execution_count]
            # Otherwise history may not have been stored (store_history=False), so we should not send the
            # code to GPT.
            else:
                code = None

            gm = Markdown(
                content="Seeking suggestion...",
                #content="Let's see how we can fix this... 🔧",
                # Note: stages were done with metadata in the past. We'll want to do that again at a later _stage_ (pun intended)
                # stage=Stage.STARTING,
            )
            gm.display()

            # Highly colorized tracebacks do not help GPT as much as a clean plaintext traceback.
            formatted = TracebackException(etype, evalue, tb, limit=3).format(chain=True)
            plaintext_traceback = "\n".join(formatted)


            resp = self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=self.form_exception_messages(code, etype, evalue, plaintext_traceback),
                tools=[],
                tool_choice='auto',
                stream=True,
            )

            gm.content = ""

            for chunk in resp:
                if chunk.choices[0].delta.content is not None:
                    gm.append(chunk.choices[0].delta.content)

        except Exception as e:
            print("Error while trying to provide a suggestion: ", e)
        except KeyboardInterrupt:
            # If we have our heading, replace it with empty text and take out any stage information
            if "gm" in locals():
                gm.content = " "
                # gm.stage = None


def register(ipython=None):
    """Register the exception handler with the given IPython instance."""
    ipython = ipython or get_ipython()
    if not ipython:
        return

    itl = InTheLoop()
        
    ipython.set_custom_exc((Exception,), itl.custom_exc)