import argparse
import importlib
import json
from pathlib import Path


def load_db_config(path: Path):
    with path.open("r", encoding="utf-8-sig") as fp:
        cfg = json.load(fp)
    required = ("host", "port", "database", "user", "password")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing database config keys: {missing}")
    return cfg


def apply_sql(conn, sql_path: Path):
    sql_text = sql_path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql_text)
    conn.commit()


def load_psycopg2():
    try:
        return importlib.import_module("psycopg2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "psycopg2 is not installed. Run: pip install psycopg2-binary"
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description="Create subopt schema/tables in PostgreSQL."
    )
    parser.add_argument(
        "--config",
        default="notebooks/postgres_config.json",
        help="DB config JSON path (host, port, database, user, password)",
    )
    parser.add_argument(
        "--sql",
        default="database/schema_subopt.sql",
        help="SQL schema file path",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    sql_path = Path(args.sql)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")
    if not sql_path.is_file():
        raise FileNotFoundError(f"SQL file not found: {sql_path.resolve()}")

    cfg = load_db_config(config_path)
    psycopg2 = load_psycopg2()

    conn = psycopg2.connect(
        host=cfg["host"],
        port=int(cfg["port"]),
        dbname=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        connect_timeout=10,
    )
    try:
        apply_sql(conn, sql_path)
    finally:
        conn.close()

    print("Schema setup complete.")
    print(f"  database: {cfg['database']}")
    print("  schema: subopt")
    print(f"  sql: {sql_path.resolve()}")


if __name__ == "__main__":
    main()
