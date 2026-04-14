from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from .dataset_registry import (
    DATASET_DIR,
    DEFAULT_DATASETS,
    append_dataset_row_to_path,
    assign_dataset,
    dataset_summary,
    duplicate_active_dataset,
    duplicate_dataset_file,
    get_active_dataset_paths,
    list_local_dataset_candidates,
    load_app_options,
    load_dataset_config,
    load_selected_methods,
    remove_dataset_file,
    save_app_options,
    save_dataset_config,
    save_selected_methods,
)
from .interactive import MenuOption, TerminalUI
from .model_workbench import METHOD_SPECS, ModelWorkbench


class SexismDetectionCLI:
    def __init__(self):
        self.ui = TerminalUI()
        self.options = load_app_options()
        self.workbench = ModelWorkbench(debug=self.options.get("debug", False))
        valid_method_keys = {spec.key for spec in METHOD_SPECS}
        self.selected_methods = [
            method_key
            for method_key in load_selected_methods()
            if method_key in valid_method_keys
        ]
        self._session_duplicated_datasets: set[Path] = set()

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
                        key="options",
                        label="Options",
                        description="CLI behavior and cleanup settings.",
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
            elif action == "options":
                self._manage_options()
            elif action == "quit":
                self._cleanup_session_duplicates()
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
            allow_cancel=True,
        )
        save_selected_methods(self.selected_methods)

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
                        description="Create a working copy, append a row, and switch the split to it.",
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
        if split is None:
            return

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
                allow_cancel=True,
            )
            if dataset_path is None:
                return
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
        if split is None:
            return

        file_name = self.ui.ask_text(
            title="Duplicate dataset",
            prompt_text="New file name",
            allow_cancel=True,
        )
        if file_name is None:
            return

        try:
            created_path = duplicate_active_dataset(split, file_name)
            self._track_session_duplicate(created_path)
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
        if split is None:
            return

        text = self.ui.ask_text(
            title="Append labeled row",
            prompt_text="Input text",
            allow_cancel=True,
        )
        if text is None:
            return

        label = self.ui.select_one(
            title="Choose label",
            options=[
                MenuOption("not sexist", "not sexist", "Negative class"),
                MenuOption("sexist", "sexist", "Positive class"),
            ],
            allow_cancel=True,
        )
        if label is None:
            return

        try:
            source_path = get_active_dataset_paths()[split]
            duplicate_name = self._build_append_dataset_name(split, source_path)
            duplicated_path = duplicate_dataset_file(source_path, duplicate_name)
            self._track_session_duplicate(duplicated_path)
            dataset_path = append_dataset_row_to_path(duplicated_path, text, label, split=split)
            assign_dataset(split, dataset_path)
        except Exception as exc:
            self.ui.show_message("Append error", str(exc), style="red")
            return

        self.ui.show_message(
            "Row appended",
            f"New row added to a working copy and {split} now points to:\n{dataset_path}",
            style="green",
        )

    def _predict_text(self) -> None:
        if not self._ensure_methods_selected():
            return

        user_text = self.ui.ask_text(
            title="Predict text",
            prompt_text="Text to classify",
            allow_cancel=True,
        )
        if user_text is None:
            return

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

    def _manage_options(self) -> None:
        selected = self.ui.select_many(
            title="Options",
            subtitle="Toggle persistent CLI settings.",
            options=[
                MenuOption(
                    key="delete_duplicated_datasets_on_exit",
                    label="Delete duplicated datasets on exit",
                    description="Enabled by default for copies created during this CLI session.",
                ),
                MenuOption(
                    key="debug",
                    label="Debug",
                    description="Enable verbose logs for preprocessing and BERTweet loading/prediction.",
                ),
            ],
            selected_keys=[key for key, value in self.options.items() if value],
            allow_cancel=True,
        )
        self.options["delete_duplicated_datasets_on_exit"] = "delete_duplicated_datasets_on_exit" in selected
        self.options["debug"] = "debug" in selected
        save_app_options(self.options)
        self.workbench.set_debug(self.options["debug"])

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

    def _choose_split(self, title: str) -> str | None:
        return self.ui.select_one(
            title=title,
            options=[
                MenuOption("train", "train", "Used for classical model training."),
                MenuOption("test", "test", "Used for evaluation."),
                MenuOption("dev", "dev", "Optional validation split."),
            ],
            allow_cancel=True,
        )

    def _format_confusion(self, confusion: list[list[int]]) -> str:
        if len(confusion) != 2 or any(len(row) != 2 for row in confusion):
            return str(confusion)
        tn, fp = confusion[0]
        fn, tp = confusion[1]
        return f"TN={tn} FP={fp} FN={fn} TP={tp}"

    def _track_session_duplicate(self, path: Path) -> None:
        self._session_duplicated_datasets.add(path.resolve())

    def _cleanup_session_duplicates(self) -> None:
        if not self.options.get("delete_duplicated_datasets_on_exit", True):
            return

        config = load_dataset_config()
        active_paths = get_active_dataset_paths(config)
        config_changed = False

        for split, path in active_paths.items():
            if path.resolve() in self._session_duplicated_datasets:
                config[split] = DEFAULT_DATASETS[split]
                config_changed = True

        if config_changed:
            save_dataset_config(config)

        for duplicated_path in list(self._session_duplicated_datasets):
            try:
                remove_dataset_file(duplicated_path)
            except FileNotFoundError:
                pass
            except OSError:
                continue

    def _build_append_dataset_name(self, split: str, source_path: Path) -> str:
        stem = source_path.stem
        suffix = source_path.suffix or ".csv"
        index = 1
        while True:
            candidate = f"{stem}_{split}_working_copy_{index}{suffix}"
            if not (DATASET_DIR / candidate).exists():
                return candidate
            index += 1


def run_cli() -> None:
    SexismDetectionCLI().run()
