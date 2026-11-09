"""Prepare persistent sample comparisons through the running arena API."""

import argparse

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Upload samples to LlamaParse with configured Mistral judging (uses API credits).",
    )
    parser.add_argument(
        "--models", nargs="+", help="Open-model IDs; defaults to all runnable ones."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Explicitly create new runs instead of reusing results.",
    )
    args = parser.parse_args()
    with httpx.Client(base_url=args.api, timeout=120) as client:
        response = client.get("/api/models")
        response.raise_for_status()
        catalog = response.json()
        models = args.models or [
            m["id"] for m in catalog["models"] if m["availability"]["runnable"]
        ]
        if args.include_reference:
            models.append(catalog["reference"]["id"])
        if not models:
            parser.error("No runnable models are available.")
        response = client.get("/api/samples")
        response.raise_for_status()
        samples = response.json()["samples"]
        if not samples:
            print("No supported documents found in data/sample-docs.")
        for sample in samples:
            response = client.post(
                f"/api/samples/{sample['id']}/prepare",
                json={"models": models, "refresh": args.refresh},
            )
            response.raise_for_status()
            experiment = response.json()
            print(f"{sample['title']}: {experiment['status']} /experiments/{experiment['id']}")


if __name__ == "__main__":
    main()
