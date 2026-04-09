from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from .dataset_registry import (
    append_dataset_row,
    assign_dataset,
    dataset_summary,
    duplicate_active_dataset,
    get_active_dataset_paths,
    list_local_dataset_candidates,
    load_dataset_config,
)
from .interactive import MenuOption, TerminalUI
from .model_workbench import METHOD_SPECS, ModelWorkbench


class SexismDetectionCLI:
    def __init__(self):
        self.ui = TerminalUI()
        self.workbench = ModelWorkbench()
        self.selected_methods: list[str] = []

    def run(self) -> None:
        while True:
            dataset_paths = get_active_dataset_paths()
            subtitle = (
                f"Selected methods: {len(self.selected_methods)} | "
                f"train={dataset_paths['train'].name} | "
                f"test={dataset_paths['test'].name} | "
                f"dev={dataset_paths['dev'].name}"
            )
            action = self.ui.select_one(
                title="Sexism Detection CLI",
                subtitle=subtitle,
                options=[
                    MenuOption(
                        key="methods",
                        label="Choose methods",
                        description="Select one or more evaluation/prediction methods.",
                    ),
                    MenuOption(
                        key="evaluate",
                        label="Evaluate methods",
                        description="Train or run the selected methods and display stats.",
                    ),
                    MenuOption(
                        key="datasets",
                        label="Manage datasets",
                        description="Switch active CSVs or append custom examples.",
                    ),
                    MenuOption(
                        key="predict",
                        label="Predict text",
                        description="Classify a custom input with the selected methods.",
                    ),
                    MenuOption(
                        key="quit",
                        label="Quit",
                        description="Exit the CLI.",
                    ),
                ],
            )

            if action == "methods":
                self._choose_methods()
            elif action == "evaluate":
                self._evaluate_methods()
            elif action == "datasets":
                self._manage_datasets()
            elif action == "predict":
                self._predict_text()
            elif action == "quit":
                self.ui.console.clear()
                return

    def _choose_methods(self) -> None:
        options = [
            MenuOption(
                key=spec.key,
                label=spec.name,
                description=spec.description,
            )
            for spec in METHOD_SPECS
        ]
        self.selected_methods = self.ui.select_many(
            title="Choose methods",
            subtitle="Use Space to toggle one or more methods.",
            options=options,
            selected_keys=self.selected_methods,
        )

    def _evaluate_methods(self) -> None:
        if not self._ensure_methods_selected():
            return

        dataset_paths = get_active_dataset_paths()
        self.ui.console.clear()
        try:
            with self.ui.console.status("Evaluating selected methods..."):
                reports = self.workbench.evaluate_methods(self.selected_methods, dataset_paths)
        except Exception as exc:
            self.ui.show_message("Evaluation error", str(exc), style="red")
            return

        table = Table(title="Evaluation summary", expand=True)
        table.add_column("Method")
        table.add_column("Accuracy")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1")
        table.add_column("Confusion")

        for report in reports:
            table.add_row(
                report.method_name,
                f"{report.accuracy:.4f}",
                f"{report.precision:.4f}",
                f"{report.recall:.4f}",
                f"{report.f1:.4f}",
                self._format_confusion(report.confusion),
            )

        self.ui.console.print(table)
        self.ui.pause()

    def _manage_datasets(self) -> None:
        while True:
            action = self.ui.select_one(
                title="Manage datasets",
                subtitle="Switch active CSV files or append new labeled rows.",
                options=[
                    MenuOption(
                        key="overview",
                        label="Show active datasets",
                        description="Display current train/test/dev files and row counts.",
                    ),
                    MenuOption(
                        key="assign",
                        label="Assign dataset to split",
                        description="Point train/test/dev to another CSV file.",
                    ),
                    MenuOption(
                        key="duplicate",
                        label="Duplicate active dataset",
                        description="Create a working copy in the dataset folder.",
                    ),
                    MenuOption(
                        key="append",
                        label="Append labeled row",
                        description="Add a custom text example to a split.",
                    ),
                    MenuOption(
                        key="back",
                        label="Back",
                        description="Return to the main menu.",
                    ),
                ],
                allow_cancel=True,
            )

            if action in {None, "back"}:
                return
            if action == "overview":
                self._show_dataset_overview()
            elif action == "assign":
                self._assign_dataset()
            elif action == "duplicate":
                self._duplicate_dataset()
            elif action == "append":
                self._append_dataset_row()

    def _show_dataset_overview(self) -> None:
        try:
            config = load_dataset_config()
            table = Table(title="Active datasets", expand=True)
            table.add_column("Split")
            table.add_column("File")
            table.add_column("Rows")
            table.add_column("Sexist")
            table.add_column("Not sexist")

            for split, path in get_active_dataset_paths(config).items():
                summary = dataset_summary(path)
                table.add_row(
                    split,
                    str(Path(summary["path"]).name),
                    str(summary["rows"]),
                    str(summary["sexist"]),
                    str(summary["not_sexist"]),
                )
        except Exception as exc:
            self.ui.show_message("Dataset error", str(exc), style="red")
            return

        self.ui.console.clear()
        self.ui.console.print(table)
        self.ui.pause()

    def _assign_dataset(self) -> None:
        split = self._choose_split("Choose split to update")
        candidates = list_local_dataset_candidates()
        options = [
            MenuOption(
                key=str(path),
                label=path.name,
                description=str(path),
            )
            for path in candidates
        ]
        options.append(
            MenuOption(
                key="manual",
                label="Manual path",
                description="Type a relative or absolute CSV path.",
            )
        )

        selected = self.ui.select_one(
            title="Choose dataset file",
            subtitle=f"Assign a CSV file to the {split} split.",
            options=options,
            allow_cancel=True,
        )
        if selected is None:
            return

        if selected == "manual":
            dataset_path = self.ui.ask_text(
                title="Manual dataset path",
                prompt_text="CSV path",
            )
        else:
            dataset_path = selected

        try:
            assigned_path = assign_dataset(split, dataset_path)
        except Exception as exc:
            self.ui.show_message("Dataset error", str(exc), style="red")
            return

        self.ui.show_message(
            "Dataset updated",
            f"{split} now points to:\n{assigned_path}",
            style="green",
        )

    def _duplicate_dataset(self) -> None:
        split = self._choose_split("Choose split to duplicate")
        file_name = self.ui.ask_text(
            title="Duplicate dataset",
            prompt_text="New file name",
        )

        try:
            created_path = duplicate_active_dataset(split, file_name)
        except Exception as exc:
            self.ui.show_message("Duplicate error", str(exc), style="red")
            return

        self.ui.show_message(
            "Dataset duplicated",
            f"Created:\n{created_path}",
            style="green",
        )

    def _append_dataset_row(self) -> None:
        split = self._choose_split("Choose target split")
        text = self.ui.ask_text(
            title="Append labeled row",
            prompt_text="Input text",
        )
        label = self.ui.select_one(
            title="Choose label",
            options=[
                MenuOption("not sexist", "not sexist", "Negative class"),
                MenuOption("sexist", "sexist", "Positive class"),
            ],
        )

        try:
            dataset_path = append_dataset_row(split, text, label)
        except Exception as exc:
            self.ui.show_message("Append error", str(exc), style="red")
            return

        self.ui.show_message(
            "Row appended",
            f"New row added to:\n{dataset_path}",
            style="green",
        )

    def _predict_text(self) -> None:
        if not self._ensure_methods_selected():
            return

        user_text = self.ui.ask_text(
            title="Predict text",
            prompt_text="Text to classify",
        )
        dataset_paths = get_active_dataset_paths()

        self.ui.console.clear()
        try:
            with self.ui.console.status("Running predictions..."):
                predictions = self.workbench.predict_text(
                    self.selected_methods,
                    user_text,
                    dataset_paths,
                )
        except Exception as exc:
            self.ui.show_message("Prediction error", str(exc), style="red")
            return

        table = Table(title="Prediction results", expand=True)
        table.add_column("Method")
        table.add_column("Prediction")
        for prediction in predictions:
            style = "red" if prediction.label == "sexist" else "green"
            table.add_row(prediction.method_name, f"[{style}]{prediction.label}[/{style}]")

        self.ui.console.print(Panel(user_text, title="Input text", border_style="cyan"))
        self.ui.console.print(table)
        self.ui.pause()

    def _ensure_methods_selected(self) -> bool:
        if self.selected_methods:
            return True

        self.ui.show_message(
            "No methods selected",
            "Choose at least one method before evaluating or predicting.",
            style="yellow",
        )
        self._choose_methods()
        return bool(self.selected_methods)

    def _choose_split(self, title: str) -> str:
        return self.ui.select_one(
            title=title,
            options=[
                MenuOption("train", "train", "Used for classical model training."),
                MenuOption("test", "test", "Used for evaluation."),
                MenuOption("dev", "dev", "Optional validation split."),
            ],
        )

    def _format_confusion(self, confusion: list[list[int]]) -> str:
        if len(confusion) != 2 or any(len(row) != 2 for row in confusion):
            return str(confusion)
        tn, fp = confusion[0]
        fn, tp = confusion[1]
        return f"TN={tn} FP={fp} FN={fn} TP={tp}"


def run_cli() -> None:
    SexismDetectionCLI().run()
