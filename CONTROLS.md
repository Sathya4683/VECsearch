# CONTROLS.md

## Interactive CLI Navigation (Prompt Toolkit)

When you run the VECsearch app and choose sentence-wise or paragraph-wise results, the output is shown using an interactive terminal interface powered by `prompt_toolkit`. This allows smooth navigation and highlighting of results.

---

## Navigation Controls

### `↓` Arrow Key

Move to the **next** highlighted sentence or paragraph.

### `↑` Arrow Key

Move to the **previous** highlighted sentence or paragraph.

### `q` Key

Exit the interactive view and return to the main prompt.

---

## Notes

- The currently selected sentence or paragraph is styled with **bold and underline**, and is surrounded by `>>> <<<`.
- Non-selected matches are shown in a different color for easy readability.
- Non-matching content (if any) is displayed in plain text.
- Useful for reviewing multiple matches and choosing the most relevant context for RAG-based responses.

---
