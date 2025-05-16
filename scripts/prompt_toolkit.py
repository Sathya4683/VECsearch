from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window, HSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

# Interactive UI using prompt_toolkit
def interactive_highlight_view(highlight_ids, id_map):
    """Create an interactive view to navigate through highlighted text."""
    index = [0]  # Use a list for mutable state

    def get_text():
        """Generate formatted text for display."""
        display = []
        for id_, text in id_map.items():
            if id_ in highlight_ids:
                if highlight_ids[index[0]] == id_:
                    display.append(('[SetCursorPosition]', ''))
                    display.append(('class:selected', f">>> {text} <<<\n\n"))
                else:
                    display.append(('class:highlight', f"{text}\n\n"))
            else:
                display.append(('', f"{text}\n\n"))
        return display

    # Set up key bindings
    kb = KeyBindings()

    @kb.add('down')
    def next_highlight(event):
        index[0] = min(index[0] + 1, len(highlight_ids) - 1)
        event.app.invalidate()

    @kb.add('up')
    def prev_highlight(event):
        index[0] = max(index[0] - 1, 0)
        event.app.invalidate()

    @kb.add('q')
    def exit_(event):
        event.app.exit()

    # Create the UI layout
    content_control = FormattedTextControl(get_text)
    root_container = HSplit([Window(content_control, always_hide_cursor=False, wrap_lines=True)])
    layout = Layout(root_container)

    # Define styles
    style = Style.from_dict({
        "highlight": "#00ff00",
        "selected": "bold underline #ff0066",
    })

    # Create and run the application
    app = Application(layout=layout, key_bindings=kb, full_screen=True, style=style)
    app.run()
