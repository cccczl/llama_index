"""Test Output parsers."""


from langchain.output_parsers import ResponseSchema
from langchain.schema import BaseOutputParser as LCOutputParser

from gpt_index.output_parsers.langchain import LangchainOutputParser


class MockOutputParser(LCOutputParser):
    """Mock output parser.

    Similar to langchain's StructuredOutputParser, but better for testing.

    """

    response_schema: ResponseSchema

    def get_format_instructions(self) -> str:
        """Get format instructions."""
        return f"{{ {self.response_schema.name}, {self.response_schema.description} }}"

    def parse(self, text: str) -> str:
        """Parse the output of an LLM call."""
        # TODO: make this better
        return text


def test_lc_output_parser() -> None:
    """Test langchain output parser."""
    response_schema = ResponseSchema(
        name="Education",
        description="education experience",
    )
    lc_output_parser = MockOutputParser(response_schema=response_schema)
    output_parser = LangchainOutputParser(lc_output_parser)

    query_str = "Hello world."
    output_instructions = output_parser.format(query_str)
    assert output_instructions == (
        "Hello world.\n\n" "{ Education, education experience }"
    )
    query_str = "foo {bar}."
    output_instructions = output_parser.format(query_str)
    assert output_instructions == (
        "foo {bar}.\n\n" "{{ Education, education experience }}"
    )
