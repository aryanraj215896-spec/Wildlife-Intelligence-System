from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
import torch

from wildlife_ai.automatic_learning import (
    INITIAL_CHECKPOINT,
    INITIAL_MODEL_VERSION,
    AUTOMATIC_BATCH_SIZE,
    automatic_update_if_ready,
    get_selected_model,
    initialize_selected_model,
)
from wildlife_ai.config.settings import ProjectSettings, SUPPORTED_CLASSES
from wildlife_ai.feedback.service import submit_feedback
from wildlife_ai.inference import predict_image
from wildlife_ai.models.checkpoint import load_checkpoint
from wildlife_ai.utils.images import decode_sanitized_image

_APP_ROOT = Path(__file__).resolve().parents[2]
_VALIDATION_REPORT = (
    _APP_ROOT
    / "reports"
    / "model_information"
    / f"{INITIAL_MODEL_VERSION}-validation.json"
)
_DISPLAY_CLASSES = tuple(
    "Gray Wolf" if label == "Gray wolf" else label for label in SUPPORTED_CLASSES
)
_INTERNAL_BY_DISPLAY = {
    display: internal
    for display, internal in zip(_DISPLAY_CLASSES, SUPPORTED_CLASSES)
}
_FEEDBACK_OPTIONS = (*_DISPLAY_CLASSES, "Other / none of these")

_STYLE = """
<style>
:root {
  color-scheme: light;
  --forest: #244535;
  --moss: #496b54;
  --paper: #f5f6f1;
  --ink: #1f3029;
  --muted: #64736a;
}
.stApp, [data-testid="stAppViewContainer"] {
  background: var(--paper);
  color: var(--ink);
  font-family: Inter, "Segoe UI", Arial, sans-serif;
}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
  display: none;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
  max-width: 860px;
  padding: 2rem clamp(1rem, 4vw, 3rem) 4rem;
}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {
  color: var(--ink) !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: #ffffff !important;
  border: 1px dashed #a9b9ac !important;
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] * {
  color: var(--ink) !important;
  opacity: 1 !important;
}
[data-testid="stFileUploaderDropzone"] button {
  background: #ffffff !important;
  border-color: #a9b9ac !important;
  color: var(--forest) !important;
}
.wild-heading {
  border-bottom: 1px solid #dce4dc;
  margin-bottom: 1.5rem;
  padding-bottom: 1.1rem;
}
.wild-heading small {
  color: var(--moss);
  font-size: .75rem;
  font-weight: 800;
  letter-spacing: .15em;
  text-transform: uppercase;
}
.wild-heading h1 {
  color: var(--forest);
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 600;
  margin: .4rem 0;
}
.wild-heading p, .wild-muted { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--forest); }
div.stButton > button[kind="primary"] {
  background: var(--forest);
  border-color: var(--forest);
}
</style>
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get_settings() -> ProjectSettings:
    return ProjectSettings.from_repo_root(os.environ.get("WILDLIFE_AI_ROOT"))


@st.cache_resource(show_spinner="Loading the selected ResNet18 model...")
def _load_model(checkpoint_path: str, checkpoint_sha256: str):
    del checkpoint_sha256
    return load_checkpoint(checkpoint_path, device=torch.device("cpu"))


def _initialize_selection() -> dict[str, object]:
    settings = _get_settings()
    checkpoint = settings.root_dir / INITIAL_CHECKPOINT
    selected_row = None
    from wildlife_ai.storage.database import connect_database

    connection = connect_database(settings.database_path)
    try:
        selected_row = connection.execute(
            "SELECT 1 FROM application_model_selection WHERE singleton = 1"
        ).fetchone()
    finally:
        connection.close()
    if selected_row is None:
        local_report = (
            settings.root_dir
            / "reports"
            / "model_information"
            / f"{INITIAL_MODEL_VERSION}-validation.json"
        )
        if not local_report.is_file():
            local_report = _VALIDATION_REPORT
        return initialize_selected_model(
            settings.root_dir,
            settings.database_path,
            checkpoint,
            local_report,
        )
    return get_selected_model(settings.root_dir, settings.database_path)


def _save_feedback(
    *,
    response: str,
    predicted_label: str,
    corrected_label: str | None,
    uploaded_bytes: bytes,
    prediction: dict[str, object],
    retention_consent: bool,
    training_consent: bool,
) -> str:
    settings = _get_settings()
    return submit_feedback(
        settings.database_path,
        settings.feedback_dir,
        response=response,
        predicted_label=predicted_label,
        corrected_label=corrected_label,
        model_version_id=str(prediction["model_version"]),
        model_score=float(prediction["model_score"]),
        image_bytes=uploaded_bytes,
        retention_consent=retention_consent,
        training_use_consent=training_consent and retention_consent,
    )


def _show_update_result(result: dict[str, object]) -> None:
    status = result.get("status")
    if status == "collecting":
        st.info(
            f"Feedback saved. {result['eligible_count']} of "
            f"{AUTOMATIC_BATCH_SIZE} consented, unique examples are ready; "
            "the model updates automatically when the batch is complete."
        )
    elif status == "accepted":
        st.success(str(result["message"]))
    elif status == "rejected":
        st.warning(str(result["message"]))
    elif status == "failed":
        st.error(str(result["message"]))


def _record_answer(
    *,
    response: str,
    corrected_label: str | None,
    upload: bytes,
    prediction: dict[str, object],
    retention_consent: bool,
    training_consent: bool,
) -> None:
    settings = _get_settings()
    _save_feedback(
        response=response,
        predicted_label=str(prediction["predicted_label"]),
        corrected_label=corrected_label,
        uploaded_bytes=upload,
        prediction=prediction,
        retention_consent=retention_consent,
        training_consent=training_consent,
    )
    st.session_state["wildlife_feedback_saved_for"] = _sha256(upload)
    st.success("Thank you. Your feedback has been saved.")
    if response in {"yes", "no"} and retention_consent and training_consent:
        try:
            result = automatic_update_if_ready(
                settings.root_dir,
                settings.database_path,
                settings.root_dir / "data" / "manifests" / "dataset_manifest.csv",
            )
        except Exception as exc:
            st.error(
                "Feedback was saved, but the automatic model update could not start: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            _show_update_result(result)
    elif response == "unsupported":
        st.caption("This image is outside the supported classes and will not be used for training.")
    elif not retention_consent:
        st.caption("Feedback metadata was saved; the image was not retained or used for training.")
    elif not training_consent:
        st.caption("The image was retained with consent but will not be used for training.")


def main() -> None:
    settings = _get_settings()
    st.set_page_config(
        page_title="Wildlife Intelligence",
        page_icon="🐾",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_STYLE, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="wild-heading">
          <small>Wildlife Intelligence System</small>
          <h1>Identify an animal</h1>
          <p>Upload an image, see the model's prediction, and help improve it with your feedback.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        selected = _initialize_selection()
        model, metadata = _load_model(
            str(selected["checkpoint_path"]),
            str(selected["checkpoint_sha256"]),
        )
    except Exception as exc:
        st.error(f"The local model could not be loaded: {type(exc).__name__}: {exc}")
        st.stop()

    uploaded = st.file_uploader(
        "Upload an animal image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="JPG, JPEG, or PNG. Images remain on this computer.",
    )
    if uploaded is None:
        st.caption(f"Selected model: {selected['model_version_id']} · ResNet18")
        return

    upload_bytes = uploaded.getvalue()
    upload_hash = _sha256(upload_bytes)
    existing_hash = st.session_state.get("wildlife_prediction_hash")
    existing_version = st.session_state.get("wildlife_prediction_version")
    new_image = existing_hash != upload_hash
    new_model = existing_version != selected["model_version_id"]
    try:
        preview = decode_sanitized_image(upload_bytes)
        if new_image or new_model:
            prediction = predict_image(
                model,
                metadata,
                upload_bytes,
                device=torch.device("cpu"),
            )
        else:
            prediction = st.session_state["wildlife_prediction"]
    except Exception as exc:
        st.error(f"This image could not be classified: {type(exc).__name__}: {exc}")
        return

    st.image(preview, caption=uploaded.name, width="stretch")
    if new_image or new_model:
        st.session_state["wildlife_prediction_hash"] = upload_hash
        st.session_state["wildlife_prediction_version"] = selected["model_version_id"]
        st.session_state["wildlife_prediction"] = prediction
    if new_image:
        st.session_state.pop("wildlife_feedback_saved_for", None)
        st.session_state.pop("wildlife_feedback_mode", None)
    prediction = st.session_state["wildlife_prediction"]
    display_label = (
        "Gray Wolf" if prediction["predicted_label"] == "Gray wolf"
        else prediction["predicted_label"]
    )
    st.markdown("### Prediction")
    st.subheader(str(display_label))
    st.caption(
        f"Uncalibrated model score: {float(prediction['model_score']):.1%} "
        f"· Model: {prediction['model_version']} (ResNet18)"
    )
    alternatives = [
        row for row in prediction["top_outputs"]
        if row["label"] != prediction["predicted_label"]
    ]
    if alternatives:
        st.markdown("**Other possible matches**")
        for row in alternatives:
            label = "Gray Wolf" if row["label"] == "Gray wolf" else row["label"]
            st.caption(f"{label} — uncalibrated score {float(row['score']):.1%}")

    st.markdown("### Is this prediction correct?")
    saved_for = st.session_state.get("wildlife_feedback_saved_for")
    if saved_for == upload_hash:
        st.success("Feedback for this image has already been saved.")
        return
    retention_consent = st.checkbox(
        "I consent to keeping a sanitized copy of this image on this computer."
    )
    training_consent = st.checkbox(
        "I separately consent to using the retained image and my confirmed label for model training.",
        disabled=not retention_consent,
    )
    if not retention_consent:
        st.caption("Without image-retention consent, only feedback metadata is saved.")

    correct_col, incorrect_col = st.columns(2)
    if correct_col.button("Correct", width="stretch"):
        _record_answer(
            response="yes",
            corrected_label=str(prediction["predicted_label"]),
            upload=upload_bytes,
            prediction=prediction,
            retention_consent=retention_consent,
            training_consent=training_consent,
        )
        return
    if incorrect_col.button("Incorrect", width="stretch"):
        st.session_state["wildlife_feedback_mode"] = "incorrect"

    if st.session_state.get("wildlife_feedback_mode") == "incorrect":
        chosen = st.selectbox("Choose the correct animal", _FEEDBACK_OPTIONS)
        if st.button("Save correction", type="primary"):
            if chosen == "Other / none of these":
                _record_answer(
                    response="unsupported",
                    corrected_label=None,
                    upload=upload_bytes,
                    prediction=prediction,
                    retention_consent=retention_consent,
                    training_consent=False,
                )
            else:
                corrected = _INTERNAL_BY_DISPLAY[chosen]
                _record_answer(
                    response="no",
                    corrected_label=corrected,
                    upload=upload_bytes,
                    prediction=prediction,
                    retention_consent=retention_consent,
                    training_consent=training_consent,
                )
            return


if __name__ == "__main__":
    main()
