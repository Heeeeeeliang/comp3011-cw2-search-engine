"""
Command-line entry point for the COMP3011 Coursework 2 search tool.

Provides an interactive shell with four commands:
    build           crawl the site, build the index, save to disk
    load            load the previously-built index from disk
    print <word>    print the inverted-index entry for <word>
    find <words>    list pages containing ALL given words

We use the standard-library `cmd.Cmd` class because it gives us a `>` prompt,
command dispatch, built-in `help`, and clean `EOF`/exit handling for free.
"""

import cmd


SEED_URL = "https://quotes.toscrape.com/"
INDEX_PATH = "data/index.json"


class SearchShell(cmd.Cmd):
    intro = "Search engine ready. Type 'help' for commands, 'exit' to quit."
    prompt = "> "

    def __init__(self) -> None:
        super().__init__()
        self.indexer = None  # populated by build/load

    def do_build(self, _arg: str) -> None:
        """build : crawl the site and build the inverted index."""
        raise NotImplementedError

    def do_load(self, _arg: str) -> None:
        """load : load the previously saved index from disk."""
        raise NotImplementedError

    def do_print(self, arg: str) -> None:
        """print <word> : show the inverted-index entry for one word."""
        raise NotImplementedError

    def do_find(self, arg: str) -> None:
        """find <word1> <word2> ... : list pages containing ALL given words."""
        raise NotImplementedError

    def do_exit(self, _arg: str) -> bool:
        """exit : quit the shell."""
        return True

    do_EOF = do_exit  # Ctrl-D also exits


if __name__ == "__main__":
    SearchShell().cmdloop()
