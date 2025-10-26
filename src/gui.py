"""PyQt-based GUI for the style transfer agent."""

from __future__ import annotations

import shutil
import string
from pathlib import Path
from typing import Dict

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal, QThreadPool, QRunnable
from PyQt6.QtGui import QPixmap, QColor, QPalette, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGraphicsOpacityEffect,
    QGraphicsDropShadowEffect,
)


def _color_to_rgba(color: QColor, alpha: int | None = None) -> str:
    if alpha is None:
        return color.name()
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"

from .session import StyleTransferSession
from .audio_capture import AudioCaptureError, VoiceCommandListener, VoiceCommandConfig
from .voice_transcriber import FasterWhisperTranscriber


class FeedbackSignals(QObject):
    finished = pyqtSignal(bool, list)

    def __init__(self) -> None:
        super().__init__()


class FeedbackTask(QRunnable):
    def __init__(self, session: StyleTransferSession, feedback: str, input_path: Path) -> None:
        super().__init__()
        self.session = session
        self.feedback = feedback
        self.input_path = input_path
        self.signals = FeedbackSignals()

    def run(self) -> None:  # pragma: no cover - executed in worker thread
        try:
            # Run the expensive feedback + restyle logic away from the UI thread.
            changed, messages = self.session.apply_feedback(self.feedback, self.input_path)
        except Exception as exc:
            self.signals.finished.emit(False, [str(exc)])
            return
        self.signals.finished.emit(changed, messages)


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.current_path: Path | None = None

    def mousePressEvent(self, event):  # pragma: no cover - GUI interaction
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ImagePreviewWindow(QWidget):
    def __init__(self, main_window: "StyleTransferWindow", label: ClickableLabel) -> None:
        super().__init__(main_window)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._main_window = main_window
        self.target_label = label
        # Prevent recursive textChanged calls while syncing both editors.
        self._sync_guard = False

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._image_label, 1)

        bottom_wrapper = QWidget()
        bottom_wrapper.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        bottom_layout = QHBoxLayout(bottom_wrapper)
        bottom_layout.setContentsMargins(0, 0, 40, 40)
        bottom_layout.addStretch()

        self._theme = main_window.theme.copy()

        self._prompt_container = QWidget()
        self._prompt_container.setObjectName("previewPromptContainer")
        self._prompt_shadow = QGraphicsDropShadowEffect()
        self._prompt_shadow.setOffset(0)
        self._prompt_shadow.setBlurRadius(45)
        self._prompt_container.setGraphicsEffect(self._prompt_shadow)

        prompt_layout = QVBoxLayout(self._prompt_container)
        prompt_layout.setContentsMargins(24, 16, 24, 16)
        prompt_layout.setSpacing(8)

        caption = QLabel("Prompt")
        caption_font = QFont()
        caption_font.setPointSize(caption.font().pointSize() + 2)
        caption_font.setBold(True)
        caption.setFont(caption_font)
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt_layout.addWidget(caption)
        self._prompt_caption = caption

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Describe adjustments…")
        self.prompt_edit.setMinimumWidth(420)
        self.prompt_edit.setFixedHeight(120)
        prompt_layout.addWidget(self.prompt_edit)

        bottom_layout.addWidget(self._prompt_container)
        bottom_layout.addStretch()

        layout.addWidget(bottom_wrapper, 0, Qt.AlignmentFlag.AlignBottom)

        self.update_theme(self._theme)
        self.prompt_edit.setPlainText(main_window.feedback_edit.toPlainText())
        self.prompt_edit.textChanged.connect(self._sync_to_main)
        main_window.feedback_edit.textChanged.connect(self._sync_from_main)

        self.update_image(label.current_path)

    def update_image(self, path: Path | None) -> None:
        if path is None or not path.exists():
            self._image_label.setText("Image not available")
            return
        pixmap = QPixmap(str(path))
        target_size = self.screen().availableSize()
        self._image_label.setPixmap(
            pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def _sync_to_main(self) -> None:
        if self._sync_guard:
            return
        self._sync_guard = True
        text = self.prompt_edit.toPlainText()
        # Block signals so we do not trigger another sync round-trip.
        self._main_window.feedback_edit.blockSignals(True)
        self._main_window.feedback_edit.setPlainText(text)
        self._main_window.feedback_edit.blockSignals(False)
        self._sync_guard = False

    def _sync_from_main(self) -> None:
        if self._sync_guard:
            return
        self._sync_guard = True
        self.prompt_edit.blockSignals(True)
        self.prompt_edit.setPlainText(self._main_window.feedback_edit.toPlainText())
        self.prompt_edit.blockSignals(False)
        self._sync_guard = False

    def showFull(self) -> None:
        self.showFullScreen()

    def mousePressEvent(self, event):  # pragma: no cover - GUI interaction
        if event.button() == Qt.MouseButton.LeftButton:
            self.close()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):  # pragma: no cover - GUI interaction
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Space):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):  # pragma: no cover - GUI interaction
        try:
            self._main_window.feedback_edit.textChanged.disconnect(self._sync_from_main)
        except Exception:
            pass
        try:
            self.prompt_edit.textChanged.disconnect(self._sync_to_main)
        except Exception:
            pass
        super().closeEvent(event)

    def update_theme(self, theme: Dict[str, QColor]) -> None:
        self._theme = theme.copy()
        accent = self._theme["accent"]
        accent_text = self._theme["accent_text"]
        overlay = QColor(self._theme.get("overlay", QColor(15, 25, 35)))
        overlay.setAlpha(200)
        glow_color = QColor(self._theme.get("glow", accent))
        glow_color.setAlpha(190)

        self._prompt_container.setStyleSheet(
            f"#previewPromptContainer{{background-color:{_color_to_rgba(overlay, overlay.alpha())};border-radius:16px;}}"
        )
        if isinstance(self._prompt_shadow, QGraphicsDropShadowEffect):
            self._prompt_shadow.setColor(glow_color)

        self.prompt_edit.setStyleSheet(
            "QTextEdit{"
            f"color:{accent_text.name()};"
            f"background-color:{_color_to_rgba(QColor(5,5,5),160)};"
            f"border:1px solid {_color_to_rgba(accent,160)};"
            "border-radius:12px;"
            "padding:8px;}"
            f"QTextEdit:focus{{border:1px solid {accent.name()};}}"
        )

        if hasattr(self, "_prompt_caption") and self._prompt_caption is not None:
            self._prompt_caption.setStyleSheet(f"color:{accent_text.name()};")

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        self.setPalette(palette)


class StyleTransferWindow(QMainWindow):
    def __init__(
        self,
        session: StyleTransferSession,
    ) -> None:
        super().__init__()
        self.session = session
        self.setWindowTitle("Style Transfer Agent")

        self.slider_config = {
            "strength": (-100, 100, 1.0, 0.0),
            "saturation_scale": (-100, 100, 1.0, 0.0),
            "brightness_shift": (-100, 100, 1.0, 0.0),
            "shadow_lift": (-100, 100, 1.0, 0.0),
            "highlight_compress": (-100, 100, 1.0, 0.0),
            "contrast": (-100, 100, 1.0, 0.0),
            "clarity": (-100, 100, 1.0, 0.0),
            "color_temperature": (-100, 100, 1.0, 0.0),
            "grain_strength": (-100, 100, 1.0, 0.0),
        }
        self.slider_labels = {
            "saturation_scale": "Saturatiom",
            "brightness_shift": "Exposure",
            "shadow_lift": "Shadows",
            "highlight_compress": "Highlights",
            "color_temperature": "White Balance",
            "grain_strength": "Grain",
        }
        self.slider_map: dict[str, QSlider] = {}
        self.thread_pool = QThreadPool.globalInstance()
        self._styled_opacity_effect: QGraphicsOpacityEffect | None = None
        self._active_feedback_jobs = 0
        self.theme = self._default_theme()

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self._on_selection_changed)

        self.original_label = ClickableLabel("Original")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(320, 320)

        self.styled_label = ClickableLabel("Styled")
        self.styled_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.styled_label.setMinimumSize(320, 320)
        self.original_label.clicked.connect(lambda: self._toggle_preview(self.original_label))
        self.styled_label.clicked.connect(lambda: self._toggle_preview(self.styled_label))
        self._preview_window: ImagePreviewWindow | None = None

        self.feedback_edit = QTextEdit()
        self.feedback_edit.setPlaceholderText("Describe adjustments, e.g. 'give it a sunset vibe'")
        self.feedback_edit.setFixedHeight(80)

        self.apply_button = QPushButton("Apply Feedback")
        self.apply_button.setObjectName("applyButton")
        self.apply_button.clicked.connect(self.apply_feedback)

        reset_button = QPushButton("Reset Parameters")
        reset_button.clicked.connect(self.reset_parameters)

        restylise_button = QPushButton("Re-style All")
        restylise_button.clicked.connect(self.restylise_all)

        load_refs_button = QPushButton("Load References")
        load_refs_button.clicked.connect(self.load_reference_images)

        load_inputs_button = QPushButton("Load Inputs")
        load_inputs_button.clicked.connect(self.load_input_images)

        prev_button = QPushButton("← Previous")
        prev_button.clicked.connect(self.select_previous)

        next_button = QPushButton("Next →")
        next_button.clicked.connect(self.select_next)

        button_row = QHBoxLayout()
        button_row.addWidget(self.apply_button)
        button_row.addWidget(reset_button)
        button_row.addWidget(restylise_button)
        button_row.addWidget(load_refs_button)
        button_row.addWidget(load_inputs_button)

        nav_row = QHBoxLayout()
        nav_row.addWidget(prev_button)
        nav_row.addWidget(next_button)

        self.voice_toggle = QCheckBox("Voice Mode")
        self.voice_toggle.setChecked(False)
        self.voice_toggle.toggled.connect(self.toggle_voice_mode)

        self.voice_status = QLabel("Voice off")
        self.voice_status.setObjectName("voiceStatus")
        self.voice_status.setAlignment(Qt.AlignmentFlag.AlignLeft)

        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice_toggle)
        voice_row.addWidget(self.voice_status, 1)

        images_layout = QHBoxLayout()
        images_layout.addWidget(self.original_label, 1)
        images_layout.addWidget(self.styled_label, 1)

        right_layout = QVBoxLayout()
        right_layout.addLayout(images_layout)
        right_layout.addWidget(self.feedback_edit)
        right_layout.addLayout(button_row)
        right_layout.addLayout(nav_row)
        right_layout.addLayout(voice_row)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_list, 1)
        container = QWidget()
        container.setLayout(right_layout)
        main_layout.addWidget(container, 3)

        self.slider_panel = self._build_slider_panel()
        self.slider_toggle_button = QPushButton("▶")
        self.slider_toggle_button.setFixedWidth(28)
        self.slider_toggle_button.clicked.connect(self.toggle_slider_panel)

        slider_wrapper_layout = QHBoxLayout()
        slider_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        slider_wrapper_layout.addWidget(self.slider_panel)
        slider_wrapper_layout.addWidget(self.slider_toggle_button)
        slider_wrapper_layout.setAlignment(self.slider_toggle_button, Qt.AlignmentFlag.AlignVCenter)

        slider_wrapper = QWidget()
        slider_wrapper.setLayout(slider_wrapper_layout)
        main_layout.addWidget(slider_wrapper)

        self.slider_panel.hide()
        self.slider_toggle_button.setText("◀")

        wrapper = QWidget()
        wrapper.setLayout(main_layout)
        self.setCentralWidget(wrapper)

        self.image_list.clear()
        self._sync_sliders()

        self._pending_input: Path | None = None
        self.restylise_timer = QTimer(self)
        self.restylise_timer.setSingleShot(True)
        self.restylise_timer.timeout.connect(self._restylise_pending)

        self.voice_controller = VoiceFeedbackController(parent=self)
        self.voice_controller.transcript_ready.connect(self._on_voice_transcript)
        self.voice_controller.error.connect(self._on_voice_error)
        self.voice_controller.state_changed.connect(self._on_voice_state)

        self._voice_listen_timer = QTimer(self)
        self._voice_listen_timer.setSingleShot(True)
        self._voice_listen_timer.timeout.connect(self._set_voice_listening)

        self._apply_theme_palette()
        QTimer.singleShot(0, self._run_onboarding)

    # ------------------------------------------------------------------
    def _load_inputs(self) -> None:
        self.image_list.clear()
        for path in self.session.list_inputs():
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.image_list.addItem(item)
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)

    def _current_paths(self) -> tuple[Path, Path] | None:
        item = self.image_list.currentItem()
        if not item:
            return None
        input_path = item.data(Qt.ItemDataRole.UserRole)
        output_path = self.session.output_dir / input_path.name
        return input_path, output_path

    def _display_images(self) -> None:
        paths = self._current_paths()
        if not paths:
            return
        input_path, output_path = paths
        self._set_pixmap(self.original_label, input_path)
        self._set_pixmap(self.styled_label, output_path)
        self._sync_sliders()

    def _set_pixmap(self, label: QLabel, path: Path) -> None:
        if not path.exists():
            label.setText("Image not available")
            if isinstance(label, ClickableLabel):
                label.current_path = None
            return
        image = Image.open(path)
        qt_image = ImageQt(image)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        if isinstance(label, ClickableLabel):
            label.current_path = path
        self._update_preview_image(label)

    # ------------------------------------------------------------------
    def _build_slider_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        panel.setLayout(layout)

        for name, (minimum, maximum, scale, default) in self.slider_config.items():
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(minimum)
            slider.setMaximum(maximum)
            slider.setTickInterval(max(1, (maximum - minimum) // 4))
            slider.valueChanged.connect(lambda value, key=name, s=scale: self._slider_changed(key, value / s))
            label_text = self.slider_labels.get(name, name.replace("_", " ").title())
            caption = QLabel(label_text)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            block = QVBoxLayout()
            block.addWidget(caption)
            block.addWidget(slider)
            layout.addLayout(block)
            slider.setValue(int(default * scale))
            self.slider_map[name] = slider

        layout.addStretch()
        return panel

    def _default_theme(self) -> Dict[str, QColor]:
        accent = QColor(32, 152, 255)
        accent_light = QColor(accent)
        accent_light = accent_light.lighter(140)
        accent_text = QColor("white")
        overlay = QColor(20, 28, 40)
        overlay.setAlpha(210)
        glow = QColor(accent)
        glow.setAlpha(190)
        return {
            "accent": accent,
            "accent_light": accent_light,
            "accent_text": accent_text,
            "overlay": overlay,
            "glow": glow,
        }

    def _apply_theme_palette(self) -> None:
        accent = self.theme["accent"]
        accent_light = self.theme["accent_light"]
        accent_text = self.theme["accent_text"]

        style = (
            "QPushButton#applyButton {"
            f"background-color:{accent.name()};"
            f"color:{accent_text.name()};"
            "border-radius:6px;"
            "padding:6px 14px;"
            "font-weight:600;"
            "}"
            "QPushButton#applyButton:hover {"
            f"background-color:{_color_to_rgba(accent_light, 230)};"
            "}"
            "QPushButton#applyButton:disabled {"
            f"background-color:{_color_to_rgba(accent_light, 170)};"
            f"color:{_color_to_rgba(accent_text, 140)};"
            "}"
            "QLabel#voiceStatus {"
            f"color:{accent.name()};"
            "font-weight:600;"
            "}"
        )
        self.setStyleSheet(style)

        if self._preview_window:
            self._preview_window.update_theme(self.theme)

    def _run_onboarding(self) -> None:
        if not self._has_inputs():
            self._show_input_prompt()
        if not self.session.has_references():
            self._show_reference_prompt()

    def _has_inputs(self) -> bool:
        try:
            return bool(self.session.list_inputs())
        except ValueError:
            return False

    def _show_input_prompt(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Images")
        dialog.setModal(True)
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        message = QLabel("Upload the photos you'd like to stylize.")
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(message)
        add_button = QPushButton("Add Images")
        add_button.setMinimumWidth(160)
        add_button.clicked.connect(lambda: self._handle_onboarding_inputs(dialog))
        layout.addWidget(add_button, 0, Qt.AlignmentFlag.AlignCenter)
        dialog.setLayout(layout)
        dialog.exec()

    def _handle_onboarding_inputs(self, dialog: QDialog) -> None:
        dialog.accept()
        before = self._has_inputs()
        self.load_input_images()
        if not self._has_inputs() and not before:
            QMessageBox.information(self, "Inputs", "No images were added. You can add them later from the main window.")

    def _show_reference_prompt(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Reference Images")
        dialog.setModal(True)
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        message = QLabel("Would you like to import reference image(s) to style from?")
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(message)
        button_row = QHBoxLayout()
        import_button = QPushButton("Import References")
        skip_button = QPushButton("Skip for now")
        import_button.clicked.connect(lambda: self._handle_onboarding_references(dialog))
        skip_button.clicked.connect(dialog.reject)
        button_row.addWidget(import_button)
        button_row.addWidget(skip_button)
        layout.addLayout(button_row)
        dialog.setLayout(layout)
        dialog.exec()

    def _handle_onboarding_references(self, dialog: QDialog) -> None:
        self.load_reference_images()
        if self.session.has_references():
            dialog.accept()
        else:
            QMessageBox.information(self, "References", "No references were selected. You can add them later.")

    def _refresh_theme_from_fingerprint(self) -> None:
        try:
            fingerprint = self.session.fingerprint()
        except ValueError:
            return

        mean = fingerprint.color_mean
        accent = QColor(
            max(0, min(255, int(mean[0] * 255))),
            max(0, min(255, int(mean[1] * 255))),
            max(0, min(255, int(mean[2] * 255))),
        )
        if accent.red() == accent.green() == accent.blue():
            accent = accent.lighter(120)

        accent_light = QColor(accent)
        accent_light = accent_light.lighter(140)
        luminance = 0.299 * accent.red() + 0.587 * accent.green() + 0.114 * accent.blue()
        accent_text = QColor("black" if luminance > 180 else "white")

        if accent_text == QColor("black"):
            overlay = QColor(238, 240, 244)
            overlay.setAlpha(220)
        else:
            overlay = QColor(20, 28, 40)
            overlay.setAlpha(210)
        glow = QColor(accent)
        glow.setAlpha(200)

        self.theme = {
            "accent": accent,
            "accent_light": accent_light,
            "accent_text": accent_text,
            "overlay": overlay,
            "glow": glow,
        }
        self._apply_theme_palette()

    def _slider_changed(self, name: str, value: float) -> None:
        paths = self._current_paths()
        if not paths:
            return
        input_path, _ = paths
        self.session.set_parameter(input_path, name, value)
        self._schedule_restylise(input_path)
        self._sync_sliders()

    def _sync_sliders(self) -> None:
        paths = self._current_paths()
        params = None
        if paths:
            input_path, _ = paths
            params = self.session.current_parameters(input_path)

        for key, slider in self.slider_map.items():
            _, _, scale, default = self.slider_config[key]
            value = getattr(params, key) if params else default
            slider.blockSignals(True)
            slider.setValue(int(value * scale))
            slider.blockSignals(False)

    def _schedule_restylise(self, input_path: Path, immediate: bool = False) -> None:
        if not self.session.has_fingerprint():
            return
        self._pending_input = input_path
        self.restylise_timer.stop()
        if immediate:
            self._restylise_pending()
        else:
            self.restylise_timer.start(150)

    def _restylise_pending(self) -> None:
        if self._pending_input is None:
            return
        path = self._pending_input
        self._pending_input = None
        styled = self.session.stylise_image(path)
        self._refresh_theme_from_fingerprint()
        current = self._current_paths()
        if current and current[0] == path:
            self._set_pixmap(self.styled_label, styled)
        self._update_preview_image(self.styled_label)

    def toggle_voice_mode(self, enabled: bool) -> None:
        if enabled:
            if not self.voice_controller.start():
                self.voice_toggle.blockSignals(True)
                self.voice_toggle.setChecked(False)
                self.voice_toggle.blockSignals(False)
                self.voice_status.setText("Voice off")
            return
        self._voice_listen_timer.stop()
        self.voice_controller.stop()

    def _queue_feedback_application(self, feedback: str, origin: str) -> None:
        feedback = feedback.strip()
        if not feedback:
            if origin == "manual":
                QMessageBox.information(self, "Feedback", "Please enter feedback before applying.")
            else:
                self.voice_status.setText("Voice: empty command")
                self._schedule_voice_listening()
            return

        paths = self._current_paths()
        if not paths:
            if origin == "manual":
                QMessageBox.information(self, "Feedback", "Load and select an image first.")
            else:
                self.voice_status.setText("Voice: select an image to apply")
                self._schedule_voice_listening()
            return

        input_path, _ = paths
        self._start_processing_visuals()

        task = FeedbackTask(self.session, feedback, input_path)
        # Route the worker's result back onto the GUI thread via Qt signals.
        task.signals.finished.connect(
            lambda success, messages, fb=feedback, path=input_path, org=origin: self._on_feedback_finished(
                fb, path, org, success, messages
            )
        )
        self.thread_pool.start(task)

        if origin == "voice":
            self.voice_status.setText("Processing voice…")
        self.feedback_edit.setPlainText(feedback)

    def _on_feedback_finished(self, feedback: str, input_path: Path, origin: str, success: bool, messages: list[str]) -> None:
        self._stop_processing_visuals()

        if success:
            self._schedule_restylise(input_path, immediate=True)
            self._sync_sliders()
            display_message = messages[0] if messages else None
            if origin == "voice":
                if not display_message:
                    plain = feedback.translate(str.maketrans("", "", string.punctuation)).strip() or feedback
                    display_message = f"Ok, I am now {plain.lower()}"
                self.feedback_edit.setPlainText(display_message)
                self.voice_status.setText(self._format_voice_status("Applied", feedback))
                self._schedule_voice_listening()
            else:
                self.feedback_edit.setPlainText(display_message or "Feedback applied.")
        else:
            if origin == "voice":
                self.voice_status.setText(messages[0] if messages else "Voice: command not recognised")
                self._schedule_voice_listening()
            else:
                QMessageBox.information(
                    self,
                    "Feedback",
                    messages[0] if messages else "Could not interpret feedback; please try rephrasing.",
                )

    def _start_processing_visuals(self) -> None:
        if self._styled_opacity_effect is None:
            self._styled_opacity_effect = QGraphicsOpacityEffect(self.styled_label)
            self.styled_label.setGraphicsEffect(self._styled_opacity_effect)
        if self._active_feedback_jobs == 0:
            # Dim the preview and disable the button so the user sees work in progress.
            self._styled_opacity_effect.setOpacity(0.4)
            self.styled_label.repaint()
            self.apply_button.setEnabled(False)
        self._active_feedback_jobs += 1

    def _stop_processing_visuals(self) -> None:
        if self._active_feedback_jobs == 0:
            return
        self._active_feedback_jobs -= 1
        if self._active_feedback_jobs == 0 and self._styled_opacity_effect is not None:
            self._styled_opacity_effect.setOpacity(1.0)
            self.styled_label.repaint()
            self.apply_button.setEnabled(True)

    def _toggle_preview(self, label: ClickableLabel) -> None:
        if self._active_feedback_jobs:
            return
        path = label.current_path
        if path is None:
            return
        if self._preview_window is not None:
            self._preview_window.close()
            self._preview_window = None
            return
        self._preview_window = ImagePreviewWindow(self, label)
        self._preview_window.destroyed.connect(lambda: setattr(self, "_preview_window", None))
        self._preview_window.showFull()

    def _update_preview_image(self, label: ClickableLabel) -> None:
        if self._preview_window and self._preview_window.target_label is label:
            self._preview_window.update_image(label.current_path)

    # ------------------------------------------------------------------
    def apply_feedback(self) -> None:
        self._queue_feedback_application(self.feedback_edit.toPlainText(), origin="manual")

    def _on_voice_transcript(self, transcript: str) -> None:
        normalized = " ".join(transcript.strip().split())
        command = normalized if normalized else transcript.strip()
        if not command:
            self.voice_status.setText("Voice: empty command")
            self._schedule_voice_listening()
            return

        self._queue_feedback_application(command, origin="voice")

    def _on_voice_error(self, message: str) -> None:
        self.voice_controller.stop()
        self.voice_status.setText("Voice error")
        self._voice_listen_timer.stop()
        self.voice_toggle.blockSignals(True)
        self.voice_toggle.setChecked(False)
        self.voice_toggle.blockSignals(False)
        QMessageBox.warning(self, "Voice Input", message)

    def _on_voice_state(self, state: str) -> None:
        if state == "listening":
            self.voice_status.setText("Listening…")
        elif state == "processing":
            self.voice_status.setText("Processing voice…")
        else:
            self.voice_status.setText("Voice off")

    @staticmethod
    def _format_voice_status(prefix: str, transcript: str) -> str:
        snippet = transcript.strip()
        if len(snippet) > 48:
            snippet = snippet[:45].rstrip() + "…"
        if not snippet:
            return prefix
        return f"{prefix}: \"{snippet}\""

    def _schedule_voice_listening(self, delay_ms: int = 1600) -> None:
        if not self.voice_toggle.isChecked():
            return
        self._voice_listen_timer.stop()
        # A small delay lets the UI surface feedback before the mic resumes listening.
        self._voice_listen_timer.start(delay_ms)

    def _set_voice_listening(self) -> None:
        if self.voice_toggle.isChecked():
            self.voice_status.setText("Listening…")

    def reset_parameters(self) -> None:
        paths = self._current_paths()
        if not paths:
            QMessageBox.information(self, "Reset", "Select an image to reset its controls.")
            return
        input_path, _ = paths
        self.session.set_parameter(input_path, "strength", 0.0)
        self.session.set_parameter(input_path, "saturation_scale", 0.0)
        self.session.set_parameter(input_path, "brightness_shift", 0.0)
        self.session.set_parameter(input_path, "shadow_lift", 0.0)
        self.session.set_parameter(input_path, "highlight_compress", 0.0)
        self.session.set_parameter(input_path, "contrast", 0.0)
        self.session.set_parameter(input_path, "clarity", 0.0)
        self.session.set_parameter(input_path, "color_temperature", 0.0)
        self.session.set_parameter(input_path, "grain_strength", 0.0)
        self._sync_sliders()
        self._schedule_restylise(input_path, immediate=True)

    def restylise_all(self, initial: bool = False) -> None:
        if not self.session.has_fingerprint():
            if not initial:
                QMessageBox.information(self, "Stylise", "Load reference images before styling inputs.")
            return
        try:
            self.session.stylise_all()
        except Exception as exc:  # pragma: no cover - surfaced to user
            if not initial:
                QMessageBox.critical(self, "Error", str(exc))
            return
        self._refresh_theme_from_fingerprint()
        self._display_images()

    def _on_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        if current is None:
            return
        paths = self._current_paths()
        if not paths:
            return
        input_path, output_path = paths
        if not output_path.exists():
            self.session.stylise_image(input_path)
        self._display_images()
        self._sync_sliders()

    def load_reference_images(self) -> None:
        files = self._choose_files("Select reference images")
        if not files:
            return
        for src in files:
            dst = self.session.config.reference_dir / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        try:
            self.session.refresh_fingerprint()
            self.restylise_all()
        except Exception as exc:
            QMessageBox.warning(self, "References", str(exc))
        else:
            QMessageBox.information(self, "References", f"Loaded {len(files)} reference images.")

    def load_input_images(self) -> None:
        files = self._choose_files("Select input images")
        if not files:
            return
        for src in files:
            dst = self.session.config.input_dir / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        self._load_inputs()
        if self.session.has_fingerprint():
            self.restylise_all()
        else:
            QMessageBox.information(self, "Inputs", "Inputs loaded. Add reference images to apply the style.")

    def toggle_slider_panel(self) -> None:
        if self.slider_panel.isVisible():
            self.slider_panel.hide()
            self.slider_toggle_button.setText("◀")
        else:
            self.slider_panel.show()
            self.slider_toggle_button.setText("▶")

    # Navigation helpers -------------------------------------------------
    def select_previous(self) -> None:
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)

    def select_next(self) -> None:
        row = self.image_list.currentRow()
        if row < self.image_list.count() - 1:
            self.image_list.setCurrentRow(row + 1)

    def _choose_files(self, title: str) -> list[Path]:
        files, _ = QFileDialog.getOpenFileNames(self, title, str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp)")
        return [Path(path) for path in files]

    # Qt lifecycle -----------------------------------------------------
    def closeEvent(self, event) -> None:  # pragma: no cover - GUI hook
        try:
            self.voice_controller.stop()
        except Exception:
            pass
        self._cleanup_voice_debug()
        super().closeEvent(event)

    @staticmethod
    def _cleanup_voice_debug() -> None:
        debug_dir = Path("outputs/voice_debug")
        if debug_dir.exists():
            try:
                shutil.rmtree(debug_dir)
            except Exception:
                pass


class VoiceFeedbackController(QObject):
    transcript_ready = pyqtSignal(str)
    error = pyqtSignal(str)
    state_changed = pyqtSignal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        transcriber: FasterWhisperTranscriber | None = None,
        config: VoiceCommandConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self._transcriber = transcriber or FasterWhisperTranscriber()
        self._config = config or VoiceCommandConfig()
        self._listener: VoiceCommandListener | None = None

    def start(self) -> bool:
        if self._listener and self._listener.is_running:
            return True
        listener = VoiceCommandListener(
            # Feed every transcript/error back into Qt signals so the GUI can react.
            transcriber=self._transcriber,
            on_transcript=self._handle_transcript,
            on_error=self._handle_error,
            config=self._config,
            debug=True,
        )
        try:
            listener.start()
        except AudioCaptureError as exc:
            self.error.emit(str(exc))
            return False
        except Exception as exc:  # pragma: no cover - unexpected microphone errors
            self.error.emit(str(exc))
            return False
        self._listener = listener
        self.state_changed.emit("listening")
        return True

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
        self.state_changed.emit("idle")

    # ------------------------------------------------------------------
    def _handle_transcript(self, transcript: str) -> None:
        self.state_changed.emit("processing")
        # Mirror transcripts in the terminal to help debug voice interactions.
        print(f"[voice] transcript: {transcript}")  # Debug visibility in terminal
        self.transcript_ready.emit(transcript)

    def _handle_error(self, exc: Exception) -> None:
        self.error.emit(str(exc))


def launch_gui(session: StyleTransferSession) -> None:
    app = QApplication([])
    window = StyleTransferWindow(session)
    window.resize(1200, 700)
    window.show()
    app.exec()
