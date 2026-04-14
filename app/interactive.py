import os
import sys
import termios
import tty
from dataclasses import dataclass
from typing import Iterable, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table


@dataclass(frozen=True)
class MenuOption:
    key: str
    label: str
    description: str = ""


class TerminalUI:
    def __init__(self):
        self.console = Console()
        self._interactive = (
            os.name == "posix" and sys.stdin.isatty() and sys.stdout.isatty()
        )

    def select_one(
        self,
        title: str,
        options: Sequence[MenuOption],
        subtitle: str = "",
        allow_cancel: bool = False,
    ) -> str | None:
        if not options:
            raise ValueError("select_one requires at least one option.")

        if not self._interactive:
            return self._fallback_select_one(title, options, allow_cancel=allow_cancel)

        cursor = 0
        while True:
            self._reset_screen()
            self.console.print(self._build_panel(title, subtitle))
            self.console.print(self._build_single_select_table(options, cursor))
            self.console.print(
                "[dim]Up/Down: move | Enter: confirm"
                + (" | Del/Backspace/Esc/q: back[/dim]" if allow_cancel else "[/dim]")
            )

            key = self._read_key()
            if key == "up":
                cursor = (cursor - 1) % len(options)
            elif key == "down":
                cursor = (cursor + 1) % len(options)
            elif key == "enter":
                return options[cursor].key
            elif key == "cancel" and allow_cancel:
                return None

    def select_many(
        self,
        title: str,
        options: Sequence[MenuOption],
        selected_keys: Iterable[str] | None = None,
        subtitle: str = "",
        allow_cancel: bool = False,
    ) -> list[str]:
        if not options:
            raise ValueError("select_many requires at least one option.")

        initial_keys = set(selected_keys or [])
        if not self._interactive:
            return self._fallback_select_many(
                title,
                options,
                initial_keys,
                allow_cancel=allow_cancel,
            )

        cursor = 0
        selected = {index for index, option in enumerate(options) if option.key in initial_keys}

        while True:
            self._reset_screen()
            self.console.print(self._build_panel(title, subtitle))
            self.console.print(self._build_multi_select_table(options, cursor, selected))
            self.console.print(
                "[dim]Up/Down: move | Space: toggle | Enter: confirm | a: all | n: none"
                + (" | Del/Backspace/Esc/q: back[/dim]" if allow_cancel else "[/dim]")
            )

            key = self._read_key()
            if key == "up":
                cursor = (cursor - 1) % len(options)
            elif key == "down":
                cursor = (cursor + 1) % len(options)
            elif key == "space":
                if cursor in selected:
                    selected.remove(cursor)
                else:
                    selected.add(cursor)
            elif key == "a":
                selected = set(range(len(options)))
            elif key == "n":
                selected.clear()
            elif key == "enter":
                return [options[index].key for index in sorted(selected)]
            elif key == "cancel" and allow_cancel:
                return [option.key for option in options if option.key in initial_keys]

    def ask_text(
        self,
        title: str,
        prompt_text: str,
        default: str | None = None,
        allow_empty: bool = False,
        allow_cancel: bool = False,
    ) -> str | None:
        while True:
            self._reset_screen()
            subtitle = "Enter a value"
            if allow_cancel:
                subtitle += " | leave empty to go back"
            self.console.print(self._build_panel(title, subtitle))
            answer = Prompt.ask(prompt_text, default=default or "")
            if allow_cancel and not answer:
                return None
            if answer or allow_empty:
                return answer

    def pause(self, message: str = "Press Enter to continue") -> None:
        Prompt.ask(f"[bold]{message}[/bold]", default="")

    def show_message(self, title: str, body: str, style: str = "cyan") -> None:
        self._reset_screen()
        self.console.print(Panel(body, title=title, border_style=style))
        self.pause()

    def _reset_screen(self) -> None:
        self.console.clear(home=True)
        self.console.print()

    def _build_panel(self, title: str, subtitle: str = "") -> Panel:
        body = title if not subtitle else f"{title}\n[dim]{subtitle}[/dim]"
        return Panel(body, border_style="cyan")

    def _build_single_select_table(
        self,
        options: Sequence[MenuOption],
        cursor: int,
    ) -> Table:
        table = Table(show_header=False, box=None, expand=True)
        table.add_column(width=3)
        table.add_column(ratio=2)
        table.add_column(ratio=5)

        for index, option in enumerate(options):
            prefix = "[bold cyan]>[/bold cyan]" if index == cursor else " "
            label = f"[bold]{option.label}[/bold]" if index == cursor else option.label
            table.add_row(prefix, label, option.description)

        return table

    def _build_multi_select_table(
        self,
        options: Sequence[MenuOption],
        cursor: int,
        selected: set[int],
    ) -> Table:
        table = Table(show_header=False, box=None, expand=True)
        table.add_column(width=3)
        table.add_column(width=5)
        table.add_column(ratio=2)
        table.add_column(ratio=5)

        for index, option in enumerate(options):
            prefix = "[bold cyan]>[/bold cyan]" if index == cursor else " "
            mark = "[green][x][/green]" if index in selected else "[dim][ ][/dim]"
            label = f"[bold]{option.label}[/bold]" if index == cursor else option.label
            table.add_row(prefix, mark, label, option.description)

        return table

    def _fallback_select_one(
        self,
        title: str,
        options: Sequence[MenuOption],
        allow_cancel: bool = False,
    ) -> str | None:
        self.console.print(Panel(title, border_style="cyan"))
        for index, option in enumerate(options, start=1):
            self.console.print(f"{index}. {option.label} - {option.description}")

        while True:
            answer = Prompt.ask(
                "Choose a number" + (" or q to go back" if allow_cancel else "")
            ).strip()
            if allow_cancel and answer.lower() == "q":
                return None
            if answer.isdigit():
                selected_index = int(answer) - 1
                if 0 <= selected_index < len(options):
                    return options[selected_index].key

    def _fallback_select_many(
        self,
        title: str,
        options: Sequence[MenuOption],
        initial_keys: set[str],
        allow_cancel: bool = False,
    ) -> list[str]:
        self.console.print(Panel(title, border_style="cyan"))
        for index, option in enumerate(options, start=1):
            marker = "x" if option.key in initial_keys else " "
            self.console.print(f"{index}. [{marker}] {option.label} - {option.description}")

        answer = Prompt.ask(
            "Enter comma-separated numbers"
            + (" (empty goes back)" if allow_cancel else " (empty keeps current selection)"),
            default="",
        ).strip()
        if not answer:
            return [option.key for option in options if option.key in initial_keys]

        selected_indexes = set()
        for chunk in answer.split(","):
            value = chunk.strip()
            if value.isdigit():
                index = int(value) - 1
                if 0 <= index < len(options):
                    selected_indexes.add(index)

        return [options[index].key for index in sorted(selected_indexes)]

    def _read_key(self) -> str:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            char = sys.stdin.read(1)
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x1b":
                next_char = sys.stdin.read(1)
                if next_char == "[":
                    final_char = sys.stdin.read(1)
                    if final_char == "3":
                        sys.stdin.read(1)
                        return "cancel"
                    return {
                        "A": "up",
                        "B": "down",
                        "C": "right",
                        "D": "left",
                    }.get(final_char, "cancel")
                return "cancel"
            if char in ("\r", "\n"):
                return "enter"
            if char == " ":
                return "space"
            if char in ("\x7f", "\x08"):
                return "cancel"
            if char.lower() == "q":
                return "cancel"
            return char.lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
