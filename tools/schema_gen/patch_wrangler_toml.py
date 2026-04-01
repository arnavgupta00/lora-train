import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrangler_toml", required=True)
    ap.add_argument("--binding", required=True)
    ap.add_argument("--db_name", required=True)
    ap.add_argument("--db_id", required=True)
    args = ap.parse_args()

    path = Path(args.wrangler_toml)
    txt = path.read_text(encoding="utf-8")

    # Idempotency: if binding already exists, do nothing.
    if f'binding = "{args.binding}"' in txt:
        print(f"binding already present: {args.binding}")
        return

    block = (
        "\n\n[[d1_databases]]\n"
        f'binding = "{args.binding}"\n'
        f'database_name = "{args.db_name}"\n'
        f'database_id = "{args.db_id}"\n'
    )
    path.write_text(txt.rstrip() + block + "\n", encoding="utf-8")
    print(f"patched {path} with binding {args.binding}")


if __name__ == "__main__":
    main()

