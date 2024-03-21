from typing import Tuple

import rich
import typer

from core import main

app = typer.Typer(add_completion=False)


@app.command()
def classify(
    ngram_range: Tuple[int, int] = typer.Option(
        (2, 2),
        help="Token size in words. e.g. (1,3) creates token of 1 word, 2 words and 3 words.",
    ),
    n_components: int = typer.Option(
        3, help="Number of categories to classify text in."
    ),
    max_iter: int = typer.Option(100, help="Number of max iteration for the LDA."),
):

    main(
        ngram_range=tuple(ngram_range), n_components=n_components, max_iter=max_iter
    )


app()
