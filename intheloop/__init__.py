"""In-the-loop: generative AI tooling for IPython and Jupyter notebooks

## Usage

  `%load_ext intheloop` to load the extension in other notebook
  projects
"""


def load_ipython_extension(ipython):
    # Example code from the genai package
    # import genai.suggestions
    # from genai.magics import assist, prompt

    # ipython.register_magic_function(assist, "cell")
    # ipython.register_magic_function(prompt, "cell")

    # genai.suggestions.register()
    from .recommendations import register
    register(ipython)


def unload_ipython_extension(ipython):
    # Unload the custom exception handler
    ipython.set_custom_exc((Exception,), None)