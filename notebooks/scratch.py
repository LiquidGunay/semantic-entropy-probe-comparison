import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    title_md = mo.md("# Scratch notebook")
    title_md
    return (mo,)


if __name__ == "__main__":
    app.run()
