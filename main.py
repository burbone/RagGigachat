from __future__ import annotations
import logging
import typer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

cli = typer.Typer(help="RAG Ассистент CLI", add_completion=False)


@cli.command()
def index(path: str):
    from src.pipeline import RAGPipeline
    result = RAGPipeline().index(path)
    typer.echo(f"\n✓ Готово")
    typer.echo(f"  Страниц:       {result.pages}")
    typer.echo(f"  Чанков:        {result.chunks}")
    typer.echo(f"  Таблиц:        {result.tables_found}")
    typer.echo(f"  Пропущено стр: {result.skipped_pages}")


@cli.command()
def ask(path: str, question: str):
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline.index(path)
    answer, sources = pipeline.ask(question)
    typer.echo(f"\n{answer}")
    if sources:
        typer.echo("\nИсточники:")
        for s in sources:
            tag = "[таблица]" if s.is_table else "[текст]"
            typer.echo(f"  {tag} стр.{s.page}  score={s.score:.3f}  {s.source}")


@cli.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    import uvicorn
    uvicorn.run("api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    cli()