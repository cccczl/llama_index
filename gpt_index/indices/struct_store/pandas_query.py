"""Default query for GPTPandasIndex."""

import logging
from typing import Any, Callable, Optional

import pandas as pd
from langchain.input import print_text

from gpt_index.data_structs.table_v2 import PandasStructTable
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.prompts.default_prompts import DEFAULT_PANDAS_PROMPT
from gpt_index.prompts.prompts import PandasPrompt
from gpt_index.response.schema import Response

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTION_STR = (
    "We wish to convert this query to executable Python code using Pandas.\n"
    "The final line of code should be a Python expression that can be called "
    "with the `eval()` function. This expression should represent a solution "
    "to the query."
)


def default_output_processor(
    output: str, df: pd.DataFrame, **output_kwargs: Any
) -> str:
    """Process outputs in a default manner."""
    import ast
    import sys
    import traceback

    if sys.version_info < (3, 9):
        logger.warn(
            "Python version must be >= 3.9 in order to use "
            "the default output processor, which executes "
            "the Python query. Instead, we will return the "
            "raw Python instructions as a string."
        )
        return output

    local_vars = {"df": df}

    # NOTE: inspired from langchain's tool
    # see langchain.tools.python.tool (PythonAstREPLTool)
    try:
        tree = ast.parse(output)
        module = ast.Module(tree.body[:-1], type_ignores=[])
        exec(ast.unparse(module), {}, local_vars)  # type: ignore
        module_end = ast.Module(tree.body[-1:], type_ignores=[])
        module_end_str = ast.unparse(module_end)  # type: ignore
        try:
            return str(eval(module_end_str, {}, local_vars))
        except Exception as e:
            raise e
    except Exception as e:
        traceback.print_exc()
        return f"There was an error running the output as Python code. Error message: {e}"


class GPTNLPandasIndexQuery(BaseGPTIndexQuery[PandasStructTable]):
    """GPT Pandas query.

    Convert natural language to Pandas python code.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

    Args:
        df (pd.DataFrame): Pandas dataframe to use.
        instruction_str (Optional[str]): Instruction string to use.
        output_processor (Optional[Callable[[str], str]]): Output processor.
            A callable that takes in the output string, pandas DataFrame,
            and any output kwargs and returns a string.
        pandas_prompt (Optional[PandasPrompt]): Pandas prompt to use.
        head (int): Number of rows to show in the table context.

    """

    def __init__(
        self,
        index_struct: PandasStructTable,
        df: Optional[pd.DataFrame] = None,
        instruction_str: Optional[str] = None,
        output_processor: Optional[Callable] = None,
        pandas_prompt: Optional[PandasPrompt] = None,
        output_kwargs: Optional[dict] = None,
        head: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        if df is None:
            raise ValueError("df must be provided.")
        self.df = df
        self._head = head
        self._pandas_prompt = pandas_prompt or DEFAULT_PANDAS_PROMPT
        self._instruction_str = instruction_str or DEFAULT_INSTRUCTION_STR
        self._output_processor = output_processor or default_output_processor
        self._output_kwargs = output_kwargs or {}

    def _get_table_context(self) -> str:
        """Get table context."""
        return str(self.df.head(self._head))

    def query(self, query_bundle: QueryBundle) -> Response:
        """Answer a query."""
        context = self._get_table_context()

        pandas_response_str, _ = self._service_context.llm_predictor.predict(
            self._pandas_prompt,
            df_str=context,
            query_str=query_bundle.query_str,
            instruction_str=self._instruction_str,
        )
        if self._verbose:
            print_text(f"> Pandas Instructions:\n" f"```\n{pandas_response_str}\n```\n")
        pandas_output = self._output_processor(
            pandas_response_str,
            self.df,
            **self._output_kwargs,
        )
        if self._verbose:
            print_text(f"> Pandas Output: {pandas_output}\n")

        response_extra_info = {
            "pandas_instruction_str": pandas_response_str,
        }

        return Response(response=pandas_output, extra_info=response_extra_info)
