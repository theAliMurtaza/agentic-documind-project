"""Tools the agent can call: document search and a safe calculator."""
import ast                                        # Parses text into a syntax tree so we can inspect it safely.
import operator                                   # Gives us + - * / as ready-made functions.

from langchain_core.tools import tool             # Decorator that turns a function into a LangChain tool.

from documind import config                       # Our settings (TOP_K, MAX_DISTANCE).
from documind.ingest import get_collection        # Reuses the database opener from Step 2.

# ---------------------------------------------------------------- search

NO_RESULTS = "NO_RELEVANT_RESULTS"                # Exact marker the agent's instructions will look for.


def search_documents(query: str, k: int = config.TOP_K) -> str:   # Pure logic, no LangChain, easy to test.
    """Return the closest chunks as text, or NO_RELEVANT_RESULTS."""
    if not query or not query.strip():             # Guards against an empty search.
        return "ERROR: the search query is empty."

    collection = get_collection()                  # Opens the Chroma database from disk.
    stored = collection.count()                    # How many chunks exist.
    if stored == 0:                                # Nothing ingested yet.
        return "ERROR: the document index is empty. Run: python -m documind.ingest"

    result = collection.query(                     # Embeds the question and finds the nearest chunks.
        query_texts=[query],                       # One question in a list (Chroma accepts batches).
        n_results=min(k, stored),                  # Never ask for more chunks than exist.
    )
    docs = result["documents"][0]                  # Chunk texts for our single question.
    metas = result["metadatas"][0]                 # Their source and page.
    dists = result["distances"][0]                 # Their distances (lower = closer in meaning).

    blocks = []                                    # Formatted chunks that pass the threshold.
    for text, meta, dist in zip(docs, metas, dists):   # Walks the three lists together.
        if dist <= config.MAX_DISTANCE:            # Keeps only chunks that are close enough.
            blocks.append(f"[{meta['source']} p.{meta['page']}] (distance {dist:.2f})\n{text}")

    if not blocks:                                 # Every chunk was too far away.
        return NO_RESULTS
    return "\n\n".join(blocks)                     # One string the model can read, citations included.


@tool                                              # Wraps the function so the model can call it.
def search_docs(query: str) -> str:
    """Search the user's uploaded PDF documents for passages relevant to a query.

    Use this for any question about the content of the documents. Write the
    query as a short, specific phrase of keywords, not a full sentence. Each
    result starts with [filename p.page], which you must cite in your answer.
    If the result is NO_RELEVANT_RESULTS, the documents do not cover the topic.
    """
    try:                                           # Any failure becomes text, never a crash.
        return search_documents(query)
    except Exception as exc:                       # Catches database or embedding problems.
        return f"ERROR: search failed ({exc})"


# ------------------------------------------------------------ calculator

_OPERATORS = {                                     # The only operations we allow.
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,                        # Unary minus, as in -5.
    ast.UAdd: operator.pos,                        # Unary plus, as in +5.
}


def _eval_node(node):                              # Recursively evaluates one node of the syntax tree.
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):   # A plain number (bools excluded).
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:   # Two operands with an operator, like 2 * 3.
        left = _eval_node(node.left)               # Evaluate the left side first.
        right = _eval_node(node.right)             # Then the right side.
        if isinstance(node.op, ast.Pow) and abs(right) > 100:   # Blocks 9**9**9 style freezes.
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:   # A sign in front of a value.
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("only numbers and + - * / % ** ( ) are allowed")   # Anything else is rejected.


def safe_eval(expression: str):                    # Pure logic, easy to test.
    """Evaluate a math expression without using eval()."""
    if not expression or len(expression) > 200:    # Reject empty or absurdly long input.
        raise ValueError("expression is empty or too long")
    tree = ast.parse(expression, mode="eval")      # Turns the text into a syntax tree.
    return _eval_node(tree.body)                   # Walks the tree and computes the value.


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result.

    Use this for any calculation instead of doing math yourself. Supports
    + - * / % ** and parentheses, for example: (240 * 0.15) + 3.
    Percentages must be written as decimals (15% is 0.15).
    """
    try:
        return str(safe_eval(expression))          # Tools return strings to the model.
    except ZeroDivisionError:
        return "ERROR: division by zero."
    except Exception as exc:                       # Bad syntax or blocked content.
        return f"ERROR: {exc}"


TOOLS = [search_docs, calculator]                  # Step 5 passes this list to the model and to the graph.