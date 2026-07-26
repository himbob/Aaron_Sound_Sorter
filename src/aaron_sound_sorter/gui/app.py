"""Tkinter desktop app for previewing and approving sort results."""

from __future__ import annotations

import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Literal

from aaron_sound_sorter.gui.models import PreviewRow, SortPreviewSession
from aaron_sound_sorter.gui.neural_explanations import neural_evidence_lines
from aaron_sound_sorter.gui.preview_service import SortPlanExporter, SortPreviewService, gui_worker_count

APP_BG = "#f5f5f2"
PANEL_BG = "#ffffff"
TEXT = "#1e1e1e"
MUTED = "#5f6368"
ACCENT = "#2458a6"
ACCENT_DARK = "#173f7a"
WARN = "#9f6112"
BORDER = "#c7c7c7"


class AaronSoundSorterApp:
    """Desktop GUI for review-before-export sorting.

    Args:
        root: Tk root window owned by the launcher.

    Side Effects:
        Opens native file dialogs, runs classification in a background thread,
        and writes approved exports when requested.

    Important Constraints:
        The app passes only selected paths to the existing sorter. It does not
        inspect filenames as classification evidence.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.preview_service = SortPreviewService()
        self.exporter = SortPlanExporter()
        self.brain_config = self.preview_service.load_brain_family_config()
        self.session: SortPreviewSession | None = None
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.input_var = tk.StringVar()
        self.brain_status_var = tk.StringVar(value=self.brain_config.display_summary())
        self.destination_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="copy")
        self.status_var = tk.StringVar(value="Choose a folder, ZIP, or audio file to begin.")
        self.selected_index: int | None = None
        self.row_indexes: list[int] = []
        self.play_process: subprocess.Popen[str] | None = None

        self.configure_root()
        self.build_menu()
        self.build_ui()
        self.root.after(200, self.poll_worker_queue)

    def configure_root(self) -> None:
        """Configure the root window with stable classic Tk colors."""
        self.root.title("Aaron Sound Sorter")
        self.root.geometry("1220x760")
        self.root.minsize(980, 620)
        self.root.configure(background=APP_BG)
        self.root.option_add("*Font", "TkDefaultFont 13")
        self.root.option_add("*Button.Font", "TkDefaultFont 13")
        self.root.option_add("*Entry.Font", "TkDefaultFont 13")
        self.root.option_add("*Listbox.Font", "TkFixedFont 12")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def build_menu(self) -> None:
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open Folder...", command=self.choose_folder)
        file_menu.add_command(label="Open ZIP or Audio File...", command=self.choose_file)
        file_menu.add_separator()
        file_menu.add_command(label="Choose Destination...", command=self.choose_destination)
        file_menu.add_separator()
        file_menu.add_command(label="Preview Sort", command=self.start_preview)
        file_menu.add_command(label="Export Approved Sort", command=self.start_export)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.close)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label="Play Selected", command=self.play_selected)
        edit_menu.add_command(label="Override Selected", command=self.override_selected)
        edit_menu.add_command(label="Reset Selected", command=self.reset_selected)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        self.root.config(menu=menubar)

    def build_ui(self) -> None:
        """Create widgets and layout."""
        self.container = tk.Frame(self.root, bg=APP_BG, padx=14, pady=12)
        self.container.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            self.container,
            text="Aaron Sound Sorter",
            bg=APP_BG,
            fg=TEXT,
            font=("TkDefaultFont", 20, "bold"),
            anchor="w",
        )
        title.pack(fill=tk.X)
        subtitle = tk.Label(
            self.container,
            text="Preview the sorter, fix folders before export, then copy, move, or symlink the approved plan.",
            bg=APP_BG,
            fg=MUTED,
            anchor="w",
        )
        subtitle.pack(fill=tk.X, pady=(2, 12))

        self.build_input_panel()
        self.build_preview_panel()
        self.build_export_panel()
        self.build_status_bar()

    def build_input_panel(self) -> None:
        """Create input and brain-family status controls."""
        panel = self.panel(self.container)
        panel.pack(fill=tk.X)
        self.row_label(panel, "Input").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        self.path_entry(panel, self.input_var).grid(row=1, column=0, sticky="ew", padx=(10, 6), pady=(0, 10))
        self.primary_button(panel, "Folder", self.choose_folder).grid(row=1, column=1, padx=3, pady=(0, 10))
        self.button(panel, "ZIP/File", self.choose_file).grid(row=1, column=2, padx=3, pady=(0, 10))
        self.primary_button(panel, "Preview Sort", self.start_preview).grid(row=1, column=3, padx=(8, 10), pady=(0, 10))

        self.row_label(panel, "Brains").grid(row=2, column=0, sticky="w", padx=10, pady=(0, 4))
        tk.Label(
            panel,
            textvariable=self.brain_status_var,
            bg=PANEL_BG,
            fg=MUTED,
            anchor="w",
        ).grid(row=3, column=0, columnspan=4, sticky="ew", padx=10, pady=(0, 10))
        panel.columnconfigure(0, weight=1)

    def build_preview_panel(self) -> None:
        """Create preview list and details controls."""
        panel = self.panel(self.container)
        panel.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        header = tk.Frame(panel, bg=PANEL_BG)
        header.pack(fill=tk.X, padx=10, pady=(10, 6))
        tk.Label(header, text="Preview Queue", bg=PANEL_BG, fg=TEXT, font=("TkDefaultFont", 15, "bold")).pack(
            side=tk.LEFT
        )
        self.button(header, "Override Selected", self.override_selected).pack(side=tk.RIGHT, padx=(6, 0))
        self.button(header, "Reset Selected", self.reset_selected).pack(side=tk.RIGHT)
        self.button(header, "Play Selected", self.play_selected).pack(side=tk.RIGHT, padx=(0, 6))

        body = tk.Frame(panel, bg=PANEL_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        list_frame = tk.Frame(body, bg=PANEL_BG)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preview_list = tk.Listbox(
            list_frame,
            bg="#fbfbfb",
            fg=TEXT,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            activestyle="dotbox",
            borderwidth=1,
            relief=tk.SOLID,
            exportselection=False,
        )
        yscroll = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.preview_list.yview)
        self.preview_list.configure(yscrollcommand=yscroll.set)
        self.preview_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.preview_list.bind("<<ListboxSelect>>", self.on_preview_select)
        self.preview_list.bind("<Double-Button-1>", lambda _event: self.override_selected())

        detail_frame = tk.Frame(body, bg="#fafafa", highlightbackground=BORDER, highlightthickness=1)
        detail_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(12, 0))
        detail_frame.configure(width=390)
        detail_frame.pack_propagate(False)
        tk.Label(
            detail_frame,
            text="Selected File",
            bg="#fafafa",
            fg=TEXT,
            font=("TkDefaultFont", 14, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(10, 4))
        self.details_text = tk.Text(
            detail_frame,
            bg="#fafafa",
            fg=TEXT,
            height=18,
            wrap=tk.WORD,
            borderwidth=0,
            padx=10,
            pady=8,
        )
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 8))
        self.details_text.configure(state=tk.DISABLED)

    def build_export_panel(self) -> None:
        """Create export controls."""
        panel = self.panel(self.container)
        panel.pack(fill=tk.X, pady=(10, 0))
        self.row_label(panel, "Destination").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        self.path_entry(panel, self.destination_var).grid(row=1, column=0, sticky="ew", padx=(10, 6), pady=(0, 10))
        self.button(panel, "Destination", self.choose_destination).grid(row=1, column=1, padx=3, pady=(0, 10))
        self.mode_button(panel, "Copy", "copy").grid(row=1, column=2, padx=3, pady=(0, 10))
        self.mode_button(panel, "Move", "move").grid(row=1, column=3, padx=3, pady=(0, 10))
        self.mode_button(panel, "Symlink", "symlink").grid(row=1, column=4, padx=3, pady=(0, 10))
        self.primary_button(panel, "Export Approved Sort", self.start_export).grid(
            row=1, column=5, padx=(8, 10), pady=(0, 10)
        )
        panel.columnconfigure(0, weight=1)

    def build_status_bar(self) -> None:
        """Create bottom status bar."""
        self.status_label = tk.Label(
            self.container,
            textvariable=self.status_var,
            bg=APP_BG,
            fg=MUTED,
            anchor="w",
        )
        self.status_label.pack(fill=tk.X, pady=(8, 0))

    def panel(self, parent: tk.Widget) -> tk.Frame:
        """Return a framed panel."""
        return tk.Frame(parent, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)

    def row_label(self, parent: tk.Widget, text: str) -> tk.Label:
        """Return a small row label."""
        return tk.Label(parent, text=text, bg=PANEL_BG, fg=MUTED, anchor="w")

    def path_entry(self, parent: tk.Widget, variable: tk.StringVar) -> tk.Entry:
        """Return a path entry widget."""
        return tk.Entry(parent, textvariable=variable, bg="#ffffff", fg=TEXT, insertbackground=TEXT, relief=tk.SOLID)

    def button(self, parent: tk.Widget, label: str, command: Any) -> tk.Button:
        """Return a secondary button."""
        return tk.Button(
            parent,
            text=label,
            command=command,
            bg="#ececec",
            fg=TEXT,
            activebackground="#dedede",
            relief=tk.RAISED,
            padx=10,
            pady=4,
        )

    def primary_button(self, parent: tk.Widget, label: str, command: Any) -> tk.Button:
        """Return a primary action button."""
        return tk.Button(
            parent,
            text=label,
            command=command,
            bg=ACCENT,
            fg="#ffffff",
            activebackground=ACCENT_DARK,
            activeforeground="#ffffff",
            relief=tk.RAISED,
            padx=12,
            pady=5,
        )

    def mode_button(self, parent: tk.Widget, label: str, value: str) -> tk.Radiobutton:
        """Return an export-mode radio button."""
        return tk.Radiobutton(
            parent,
            text=label,
            value=value,
            variable=self.mode_var,
            bg=PANEL_BG,
            fg=TEXT,
            activebackground=PANEL_BG,
            selectcolor="#ffffff",
        )

    def choose_folder(self) -> None:
        """Choose a folder input."""
        selected = filedialog.askdirectory(title="Choose sample folder")
        if selected:
            self.input_var.set(selected)

    def choose_file(self) -> None:
        """Choose a ZIP or audio-file input."""
        selected = filedialog.askopenfilename(
            title="Choose ZIP or audio file",
            filetypes=[
                ("Audio or ZIP", "*.zip *.wav *.aif *.aiff *.flac *.ogg *.au"),
                ("All files", "*"),
            ],
        )
        if selected:
            self.input_var.set(selected)

    def choose_destination(self) -> None:
        """Choose an export destination."""
        selected = filedialog.askdirectory(title="Choose destination folder")
        if selected:
            self.destination_var.set(selected)

    def start_preview(self) -> None:
        """Start classification preview in a worker thread."""
        input_text = self.input_var.get().strip()
        if not input_text:
            messagebox.showwarning("Missing Input", "Choose a folder, ZIP, or audio file first.")
            return
        self.status_var.set("Classifying files for preview...")
        self.clear_rows()
        thread = threading.Thread(
            target=self.preview_worker,
            args=(Path(input_text),),
            daemon=True,
        )
        thread.start()

    def preview_worker(self, input_path: Path) -> None:
        """Worker target for preview classification."""
        try:
            session = self.preview_service.classify_input(
                input_path,
                sort_workers=gui_worker_count(),
            )
            self.worker_queue.put(("preview_done", session))
        except Exception as exc:
            self.worker_queue.put(("error", exc))

    def start_export(self) -> None:
        """Start approved-plan export in a worker thread."""
        if self.session is None:
            messagebox.showwarning("No Preview", "Run Preview Sort before exporting.")
            return
        destination_text = self.destination_var.get().strip()
        if not destination_text:
            messagebox.showwarning("Missing Destination", "Choose a destination folder first.")
            return
        self.status_var.set("Exporting approved sort plan...")
        thread = threading.Thread(
            target=self.export_worker,
            args=(self.session, Path(destination_text), self.mode_var.get()),
            daemon=True,
        )
        thread.start()

    def export_worker(self, session: SortPreviewSession, destination: Path, mode: str) -> None:
        """Worker target for approved-plan export."""
        try:
            summary = self.exporter.export(session, destination, mode=mode)  # type: ignore[arg-type]
            self.worker_queue.put(("export_done", summary))
        except Exception as exc:
            self.worker_queue.put(("error", exc))

    def poll_worker_queue(self) -> None:
        """Apply worker results on the Tk main thread."""
        try:
            while True:
                event, payload = self.worker_queue.get_nowait()
                if event == "preview_done":
                    self.set_session(payload)
                elif event == "export_done":
                    self.handle_export_done(payload)
                elif event == "error":
                    self.status_var.set("Error")
                    messagebox.showerror("Aaron Sound Sorter", str(payload))
        except queue.Empty:
            pass
        self.root.after(200, self.poll_worker_queue)

    def set_session(self, session: SortPreviewSession) -> None:
        """Display a completed preview session."""
        self.session = session
        self.clear_rows()
        for index, row in enumerate(session.rows):
            self.row_indexes.append(index)
            self.preview_list.insert(tk.END, self.format_preview_line(row))
        if session.rows:
            self.preview_list.selection_set(0)
            self.show_row_details(session.rows[0])
        self.status_var.set(f"Preview ready: {len(session.rows)} files. Run folder: {session.run_dir}")

    def format_preview_line(self, row: PreviewRow) -> str:
        """Return one visible preview-list line."""
        corrected = " *" if row.is_corrected else "  "
        return f"{corrected} {row.display_name}  ->  {row.approved_folder}"

    def clear_rows(self) -> None:
        """Clear the preview table."""
        self.preview_list.delete(0, tk.END)
        self.row_indexes.clear()
        self.selected_index = None
        self.set_details_text("No file selected.")

    def on_preview_select(self, _event: tk.Event[Any]) -> None:
        """Update the details panel for the selected row."""
        selected = self.preview_list.curselection()
        if not selected or self.session is None:
            return
        self.selected_index = selected[0]
        row = self.session.rows[self.selected_index]
        self.show_row_details(row)

    def selected_row(self) -> tuple[int, PreviewRow] | None:
        """Return the selected preview index and row."""
        if self.session is None:
            return None
        selected = self.preview_list.curselection()
        if not selected:
            return None
        index = int(selected[0])
        if index < 0 or index >= len(self.session.rows):
            return None
        return index, self.session.rows[index]

    def show_row_details(self, row: PreviewRow) -> None:
        """Show selected-row details."""
        lines = [
            f"File: {row.display_name}",
            "",
            f"Approved: {row.approved_folder}",
            f"Proposed:  {row.proposed_folder}",
            f"Decision:  {row.consensus_status}",
            f"Top:       {row.final_top}",
            f"Duration:  {row.duration_sec:.2f}s",
            "",
            "Neural Audio (plain English):",
            *neural_evidence_lines(row),
            "",
            "Voter Summary:",
            row.diagnostic_summary or "(none)",
            "",
            "Reason:",
            row.decision_reason or "(none)",
        ]
        self.set_details_text("\n".join(lines))

    def set_details_text(self, text: str) -> None:
        """Set the read-only details text."""
        self.details_text.configure(state=tk.NORMAL)
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert("1.0", text)
        self.details_text.configure(state=tk.DISABLED)

    def override_selected(self) -> None:
        """Open an override dialog for the selected row."""
        selected = self.selected_row()
        if selected is None or self.session is None:
            messagebox.showinfo("No Selection", "Select a row to override.")
            return
        index, row = selected
        dialog = OverrideDialog(self.root, row=row, labels=self.session.available_labels)
        self.root.wait_window(dialog.window)
        if dialog.selected_folder:
            row.approved_folder = dialog.selected_folder
            self.refresh_preview_row(index, row)

    def reset_selected(self) -> None:
        """Reset selected row to the sorter proposal."""
        selected = self.selected_row()
        if selected is None:
            return
        index, row = selected
        row.approved_folder = row.proposed_folder
        self.refresh_preview_row(index, row)

    def refresh_preview_row(self, index: int, row: PreviewRow) -> None:
        """Refresh displayed values for one preview row."""
        self.preview_list.delete(index)
        self.preview_list.insert(index, self.format_preview_line(row))
        self.preview_list.selection_clear(0, tk.END)
        self.preview_list.selection_set(index)
        self.preview_list.activate(index)
        self.show_row_details(row)

    def play_selected(self) -> None:
        """Play the currently selected row through macOS audio playback."""
        selected = self.selected_row()
        if selected is None:
            messagebox.showinfo("No Selection", "Select a row to play.")
            return
        _index, row = selected
        if not row.source_path.exists():
            messagebox.showwarning("Missing Audio", f"Audio file not found:\n{row.source_path}")
            return
        player_path = Path("/usr/bin/afplay")
        if not player_path.exists():
            messagebox.showinfo("Playback Unavailable", "Use the browser GUI for built-in audio playback.")
            return
        self.stop_playback()
        self.play_process = subprocess.Popen(
            [str(player_path), str(row.source_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.status_var.set(f"Playing {row.display_name}")

    def stop_playback(self) -> None:
        """Stop active Tk fallback playback if one is running."""
        if self.play_process is None:
            return
        if self.play_process.poll() is None:
            self.play_process.terminate()
        self.play_process = None

    def close(self) -> None:
        """Stop playback and close the app."""
        self.stop_playback()
        self.root.destroy()

    def handle_export_done(self, summary: Any) -> None:
        """Show export result and optionally write correction evidence."""
        status = f"Exported {summary.exported_count} files to {summary.sorted_root}"
        if summary.errors:
            status += f" with {len(summary.errors)} errors"
        self.status_var.set(status)
        if summary.errors:
            messagebox.showwarning("Export Finished With Errors", "\n".join(summary.errors[:10]))
        else:
            messagebox.showinfo("Export Complete", status)
        if self.session is not None and self.session.corrected_count:
            if messagebox.askyesno(
                "Save Correction Evidence",
                "Manual corrections exist. Save a measured-evidence correction pack for future brain training?",
            ):
                path = self.exporter.write_correction_evidence_package(self.session)
                messagebox.showinfo("Correction Evidence Saved", str(path))


class OverrideDialog:
    """Folder override dialog that uses classic Tk widgets."""

    def __init__(self, parent: tk.Tk, *, row: PreviewRow, labels: list[str]) -> None:
        self.window = tk.Toplevel(parent)
        self.window.title("Override Folder")
        self.window.geometry("820x360")
        self.window.minsize(680, 300)
        self.window.configure(bg=APP_BG)
        self.window.transient(parent)
        self.window.grab_set()
        self.selected_folder = ""
        self.folder_var = tk.StringVar(value=row.approved_folder)
        self.labels = labels

        outer = tk.Frame(self.window, bg=APP_BG, padx=12, pady=12)
        outer.pack(fill=tk.BOTH, expand=True)
        tk.Label(outer, text=row.display_name, bg=APP_BG, fg=TEXT, font=("TkDefaultFont", 14, "bold"), anchor="w").pack(
            fill=tk.X
        )
        tk.Label(outer, text="Approved folder", bg=APP_BG, fg=MUTED, anchor="w").pack(fill=tk.X, pady=(10, 4))
        entry = tk.Entry(outer, textvariable=self.folder_var, bg="#ffffff", fg=TEXT, insertbackground=TEXT)
        entry.pack(fill=tk.X)
        entry.focus_set()

        list_frame = tk.Frame(outer, bg=APP_BG)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.label_list = tk.Listbox(
            list_frame,
            bg="#ffffff",
            fg=TEXT,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            exportselection=False,
        )
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.label_list.yview)
        self.label_list.configure(yscrollcommand=scrollbar.set)
        self.label_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        for label in labels:
            self.label_list.insert(tk.END, label)
        self.label_list.bind("<<ListboxSelect>>", self.use_selected_label)
        self.label_list.bind("<Double-Button-1>", lambda _event: self.apply())

        buttons = tk.Frame(outer, bg=APP_BG)
        buttons.pack(fill=tk.X, pady=(10, 0))
        tk.Button(buttons, text="Cancel", command=self.window.destroy, padx=12, pady=4).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(
            buttons,
            text="Apply",
            command=self.apply,
            bg=ACCENT,
            fg="#ffffff",
            activebackground=ACCENT_DARK,
            activeforeground="#ffffff",
            padx=12,
            pady=4,
        ).pack(side=tk.RIGHT)
        self.window.bind("<Return>", lambda _event: self.apply())
        self.window.bind("<Escape>", lambda _event: self.window.destroy())

    def use_selected_label(self, _event: tk.Event[Any]) -> None:
        """Copy selected label into the editable folder entry."""
        selected = self.label_list.curselection()
        if selected:
            self.folder_var.set(self.label_list.get(int(selected[0])))

    def apply(self) -> None:
        """Accept the typed or selected folder."""
        value = self.folder_var.get().strip()
        if value:
            self.selected_folder = value
        self.window.destroy()


def create_root() -> tk.Tk:
    """Create the Tk root with macOS deprecation noise suppressed by caller."""
    root = tk.Tk()
    return root


def probe_layout() -> int:
    """Build the app once and print widget counts for command-line validation.

    Returns:
        Zero when the app creates visible controls; nonzero when no visible
        controls are attached to the root.
    """
    root = create_root()
    app = AaronSoundSorterApp(root)
    root.update_idletasks()
    visible_widgets = [
        widget for widget in app.container.winfo_children() if widget.winfo_ismapped() or str(widget.winfo_manager())
    ]
    print(f"root_geometry={root.winfo_geometry()}")
    print(f"visible_top_level_widgets={len(visible_widgets)}")
    root.destroy()
    return 0 if len(visible_widgets) >= 5 else 2


def main(mode: Literal["run", "probe"] = "run") -> int:
    """Launch the desktop app.

    Args:
        mode: ``"probe"`` builds the layout once for validation; ``"run"``
            starts the normal GUI loop.

    Returns:
        Process exit code.
    """
    if mode == "probe":
        return probe_layout()
    root = create_root()
    AaronSoundSorterApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
