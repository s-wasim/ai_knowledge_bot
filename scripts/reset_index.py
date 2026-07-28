"""Drop and recreate the chunk index.

The move to 768-dimension local embeddings and AST-aware chunking makes any
previously ingested index incompatible: vector dimensions differ, chunk
boundaries differ, and older rows carry absolute filesystem paths. Rather than
migrate rows that would still be wrong, this drops the table and rebuilds the
schema so repos can be re-ingested cleanly.

Usage:
    docker compose run --rm app python -m scripts.reset_index
    docker compose run --rm app python -m scripts.reset_index --keep-repos
"""

import argparse
import sys

from sqlalchemy import text

from app.db import Base, _create_search_objects, init_db


def reset_index(keep_repos: bool = False) -> None:
    engine, _ = init_db()

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
        if not keep_repos:
            conn.execute(text("DELETE FROM repos"))
        conn.commit()

    Base.metadata.create_all(engine)

    if engine.dialect.name == "postgresql":
        _create_search_objects(engine)

    with engine.connect() as conn:
        remaining = conn.execute(text("SELECT count(*) FROM repos")).scalar()

    print(f"Index reset. chunks table rebuilt; {remaining} repo row(s) remain.")
    if remaining:
        print("Re-ingest each repo to repopulate chunks.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drop and recreate the chunk index.")
    parser.add_argument(
        "--keep-repos",
        action="store_true",
        help="Keep repo rows (their chunk counts will be stale until re-ingest).",
    )
    args = parser.parse_args(argv)
    reset_index(keep_repos=args.keep_repos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
