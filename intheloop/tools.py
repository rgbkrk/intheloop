from pydantic import BaseModel, Field
from openai.lib._pydantic import to_strict_json_schema

class CreateCell(BaseModel):
    cell: str = Field(description="The cell to create")

    def run(self) -> str:
        """Create a new cell in the notebook"""
        return "Cell created"

    @classmethod
    def function_schema(cls):
        return {
            "type": "function",
            "name": cls.__name__,
            "description": cls.__doc__,
            "parameters": to_strict_json_schema(cls),
        }
