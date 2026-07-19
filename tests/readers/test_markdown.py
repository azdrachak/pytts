from pathlib import Path

from pytts.domain import BlockKind
from pytts.readers.markdown import MarkdownReader


def _write(tmp_path: Path, text: str) -> Path:
    source = tmp_path / "article.md"
    source.write_text(text, encoding="utf-8")
    return source


def test_preserves_article_structure_and_link_label(tmp_path: Path) -> None:
    source = _write(
        tmp_path,
        """# Заголовок

Первый [абзац](https://example.com).

- Первый пункт
- Второй пункт

> Важная цитата.
""",
    )

    article = MarkdownReader().read(source)

    assert [(block.kind, block.text) for block in article.blocks] == [
        (BlockKind.HEADING, "Заголовок"),
        (BlockKind.PARAGRAPH, "Первый абзац."),
        (BlockKind.LIST_ITEM, "Первый пункт"),
        (BlockKind.LIST_ITEM, "Второй пункт"),
        (BlockKind.QUOTE, "Важная цитата."),
    ]


def test_drops_non_article_markdown_constructs(tmp_path: Path) -> None:
    source = _write(
        tmp_path,
        """---
title: Метаданные
---
# Статья

Текст ![картинка](image.png) после картинки.[^1]

```python
print("не читать")
```

    и этот блок кода тоже не читать

| Колонка | Значение |
| -- | -- |
| Код | 42 |

<aside>не читать</aside>

[^1]: Сноска, которую не читаем.
""",
    )

    article = MarkdownReader().read(source)

    assert [(block.kind, block.text) for block in article.blocks] == [
        (BlockKind.HEADING, "Статья"),
        (BlockKind.PARAGRAPH, "Текст после картинки."),
    ]


def test_decodes_html_entities_but_ignores_inline_code(tmp_path: Path) -> None:
    source = _write(tmp_path, "Текст &amp; ещё \\*важно\\* и `rm -rf example`.\n")

    article = MarkdownReader().read(source)

    assert article.blocks[0].text == "Текст & ещё *важно* и ."


def test_preserves_text_special_tokens_while_omitting_inline_code(tmp_path: Path) -> None:
    source = _write(tmp_path, "Текст &amp; ещё `код`.\n")

    article = MarkdownReader().read(source)

    assert article.blocks[0].text == "Текст & ещё ."
