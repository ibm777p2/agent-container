"""Ask the agent about listings.

One-shot:     uv run pi-ask "3 bed homes under 500k in Austin"
Interactive:  uv run pi-ask
"""

import argparse


def ask(graph, question: str, verbose: bool) -> None:
    inputs = {"messages": [{"role": "user", "content": question}]}
    last = None
    for chunk in graph.stream(inputs, stream_mode="values"):
        last = chunk["messages"][-1]
        if verbose:
            last.pretty_print()
    if not verbose and last is not None:
        print(last.text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every graph step")
    args = parser.parse_args()

    from property_intelligence_rag.graph import graph

    if args.question:
        ask(graph, " ".join(args.question), args.verbose)
        return

    print("Property Intelligence — ask about listings (Ctrl-D or 'exit' to quit)")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit"}:
            return
        if question:
            ask(graph, question, args.verbose)
