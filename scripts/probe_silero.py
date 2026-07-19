from pathlib import Path

from pytts.silero_probe import run_probe


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = run_probe(project_root)
    print(f"speakers={','.join(result.speakers)}")
    print(f"sha256={result.digest}")
    print(f"max_text_chars={result.max_text_chars}")


if __name__ == "__main__":
    main()
