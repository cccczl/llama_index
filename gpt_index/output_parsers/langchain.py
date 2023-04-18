"""Base output parser class."""

from string import Formatter
from typing import Any, Optional

from langchain.schema import BaseOutputParser as LCOutputParser

from gpt_index.output_parsers.base import BaseOutputParser


class LangchainOutputParser(BaseOutputParser):
    """Langchain output parser."""

    def __init__(
        self, output_parser: LCOutputParser, format_key: Optional[str] = None
    ) -> None:
        """Init params."""
        self._output_parser = output_parser
        self._format_key = format_key

    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""
        # TODO: this object may be stringified by our upstream llmpredictor,
        # figure out better
        # ways to "convert" the object to a proper string format.
        return self._output_parser.parse(output)

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        format_instructions = self._output_parser.get_format_instructions()

        if query_tmpl_vars := {
            v for _, v, _, _ in Formatter().parse(query) if v is not None
        }:
            format_instructions = format_instructions.replace("{", "{{")
            format_instructions = format_instructions.replace("}", "}}")

        return (
            query.format(**{self._format_key: format_instructions})
            if self._format_key is not None
            else query + "\n\n" + format_instructions
        )
