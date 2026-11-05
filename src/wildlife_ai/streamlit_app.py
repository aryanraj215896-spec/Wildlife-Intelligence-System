from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import streamlit as st

from wildlife_ai.curation.eligibility import (
    check_training_eligibility,
    export_curated_manifest,
)
from wildlife_ai.config.settings import ProjectSettings
from wildlife_ai.explainability.gradcam import generate_gradcam
from wildlife_ai.feedback.service import (
    list_feedback,
    review_feedback,
    submit_feedback,
    withdraw_retained_image,
)
from wildlife_ai.inference import predict_image, prepare_model_input
from wildlife_ai.models.checkpoint import load_checkpoint
from wildlife_ai.registry.service import (
    create_promotion_policy,
    list_candidate_models,
    promote_candidate,
    register_candidate,
    reject_candidate,
    rollback_model,
)
from wildlife_ai.storage.database import connect_database
from wildlife_ai.taxonomy.resolver import resolve_name
from wildlife_ai.training.engine import resolve_device
from wildlife_ai.utils.images import decode_sanitized_image

settings = ProjectSettings.from_repo_root(os.environ.get("WILDLIFE_AI_ROOT"))

_APP_CSS = """
<style>
:root {
  color-scheme: light;
  --wild-ink: #1f3029;
  --wild-muted: #607068;
  --wild-green: #315b47;
  --wild-green-dark: #244535;
  --wild-sage: #e8eee8;
  --wild-paper: #f5f6f1;
  --wild-card: #ffffff;
  --wild-line: #dce4dc;
  --wild-gold: #b7893f;
}
.stApp, [data-testid="stAppViewContainer"] {
  --text-color: var(--wild-ink) !important;
  --background-color: var(--wild-paper) !important;
  --secondary-background-color: var(--wild-card) !important;
  --primary-color: var(--wild-green) !important;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--wild-paper) !important;
  color: var(--wild-ink) !important;
  font-family: Inter, "Segoe UI", Arial, sans-serif;
}
[data-testid="stHeader"] {
  background: transparent !important;
}
[data-testid="stMainBlockContainer"] {
  max-width: 1260px;
  padding: 2rem clamp(1.1rem, 4vw, 3.2rem) 4rem;
}
[data-testid="stSidebar"] {
  background: var(--wild-sage) !important;
  border-right: 1px solid var(--wild-line);
  color: var(--wild-ink) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.4rem;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: var(--wild-ink) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  border-radius: 0.65rem;
  padding: 0.45rem 0.7rem;
  transition: background-color 120ms ease, color 120ms ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: #dce7dc;
}
.wild-brand {
  border-bottom: 1px solid #cbd8cc;
  margin: 0 0 1.3rem;
  padding: 0.3rem 0.3rem 1.1rem;
}
.wild-brand-mark {
  color: var(--wild-green);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.17em;
  text-transform: uppercase;
}
.wild-brand-name {
  color: var(--wild-ink);
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.32rem;
  font-weight: 700;
  line-height: 1.2;
  margin-top: 0.45rem;
}
.wild-brand-status {
  color: var(--wild-muted);
  font-size: 0.78rem;
  margin-top: 0.6rem;
}
.wild-page-head {
  border-bottom: 1px solid var(--wild-line);
  margin: 0 0 1.6rem;
  padding: 0 0 1rem;
}
.wild-eyebrow {
  color: var(--wild-green);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.wild-page-head h1 {
  color: var(--wild-ink);
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(1.8rem, 3vw, 2.45rem);
  font-weight: 600;
  letter-spacing: -0.025em;
  line-height: 1.14;
  margin: 0.35rem 0 0.45rem;
}
.wild-page-head p {
  color: var(--wild-muted);
  font-size: 1rem;
  line-height: 1.55;
  margin: 0;
  max-width: 68rem;
}
.wild-masthead {
  align-items: center;
  border-bottom: 1px solid var(--wild-line);
  color: var(--wild-green);
  display: flex;
  font-size: 0.76rem;
  font-weight: 800;
  justify-content: space-between;
  letter-spacing: 0.12em;
  margin-bottom: 1.7rem;
  padding-bottom: 0.75rem;
  text-transform: uppercase;
}
.wild-masthead span {
  color: var(--wild-muted);
  font-size: 0.68rem;
  letter-spacing: 0.1em;
}
[data-testid="stMetric"] {
  background: var(--wild-card);
  border: 1px solid var(--wild-line);
  border-radius: 0.8rem;
  padding: 0.9rem 1rem;
}
[data-testid="stMetricLabel"] {
  color: var(--wild-muted);
  font-size: 0.83rem;
}
[data-testid="stMetricValue"] {
  color: var(--wild-ink);
  font-family: Georgia, "Times New Roman", serif;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--wild-card) !important;
  border-color: var(--wild-line) !important;
  border-radius: 0.9rem;
}
[data-testid="stAlert"] {
  border-radius: 0.75rem;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
  color: var(--wild-ink) !important;
}
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
  color: var(--wild-ink) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background: #fff !important;
  border-color: var(--wild-line);
  border-radius: 0.65rem;
  color: var(--wild-ink) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  color: var(--wild-ink) !important;
}
[data-testid="stSelectbox"] [role="group"] {
  background: #fff !important;
  border: 1px solid var(--wild-line);
  border-radius: 0.65rem;
}
[data-testid="stSelectbox"] input[role="combobox"] {
  background: #fff !important;
  color: var(--wild-ink) !important;
}
[data-testid="stSelectbox"] button {
  background: #fff !important;
  color: var(--wild-ink) !important;
}
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] section {
  color: var(--wild-ink) !important;
}
[data-testid="stFileUploader"] section {
  background: #fbfcf9;
  border: 1.5px dashed #9aaf9e;
  border-radius: 0.85rem;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--wild-green);
}
[data-testid="stFileUploader"] button {
  background: #fff !important;
  color: var(--wild-green-dark) !important;
}
div.stButton > button,
[data-testid="stDownloadButton"] button {
  background: #fff;
  border: 1px solid #cbd7cd;
  border-radius: 0.65rem;
  color: var(--wild-ink);
  font-weight: 650;
  min-height: 2.55rem;
  transition: background-color 120ms ease, border-color 120ms ease;
}
div.stButton > button[kind="primary"] {
  background: var(--wild-green);
  border-color: var(--wild-green);
  color: #fff;
}
div.stButton > button[kind="primary"]:hover {
  background: var(--wild-green-dark);
  border-color: var(--wild-green-dark);
  color: #fff;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--wild-line);
  border-radius: 0.75rem;
  overflow: hidden;
}
h2, h3 {
  color: var(--wild-ink);
}
@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"] {
    padding: 1.2rem 1rem 2.5rem;
  }
  .wild-masthead {
    padding-right: 2rem;
  }
  .wild-masthead span {
    display: none;
  }
  .wild-page-head {
    margin-bottom: 1.15rem;
  }
  [data-testid="stSidebar"] {
    border-right: 0;
  }
}
</style>
"""


@st.cache_resource
def _load_model(checkpoint_path: str, device_name: str):
    device = resolve_device(device_name)
    return load_checkpoint(checkpoint_path, device=device), device


def _active_model() -> dict[str, object] | None:
    connection = connect_database(settings.database_path)
    try:
        row = connection.execute(
            """SELECT mv.model_version_id, mv.artifact_path, mv.artifact_sha256,
                      mv.architecture, mv.metadata_json
               FROM active_model am JOIN model_versions mv
                 ON mv.model_version_id = am.model_version_id
               WHERE am.singleton = 1"""
        ).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def _taxonomy_results(query: str) -> dict[str, object]:
    connection = connect_database(settings.database_path)
    try:
        return resolve_name(connection, query)
    finally:
        connection.close()


def _render_page_heading(title: str, description: str, eyebrow: str) -> None:
    st.markdown(
        f"""
        <div class="wild-page-head">
          <div class="wild-eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_taxon_search(query_key: str) -> tuple[str | None, str, str]:
    query = st.text_input("Animal name", key=query_key).strip()
    if not query:
        return None, "", ""
    result = _taxonomy_results(query)
    candidates = result["candidates"]
    st.caption(f"Resolver method: {result['method']}; confirmation is always required.")
    if not candidates:
        st.info("No taxonomy candidates were found. The text remains unresolved.")
        return None, query, str(result["method"])
    labels = [
        (
            f"{item['accepted_common_name'] or item['accepted_scientific_name']} — "
            f"{item['accepted_scientific_name']} ({item['rank']})"
        )
        for item in candidates
    ]
    selected_index = st.radio(
        "Choose the intended taxon",
        options=list(range(len(candidates))),
        format_func=lambda index: labels[index],
        key=f"{query_key}-candidate",
    )
    selected = candidates[selected_index]
    st.caption(
        f"Source: {selected['source_namespace']} {selected['source_release']}; "
        f"candidate score {selected['score']:.0f}."
    )
    confirmed = st.checkbox(
        "I confirm this is the animal I intended",
        key=f"{query_key}-confirm",
    )
    return (
        selected["taxon_version_id"] if confirmed else None,
        query,
        str(result["method"]),
    )


def _submit_feedback(
    response: str,
    prediction: dict[str, object],
    *,
    image_bytes: bytes,
    retain: bool,
    training_use: bool,
    original_name: str = "",
    taxon_id: str | None = None,
    resolver_method: str = "",
) -> None:
    source = license_name = attribution = ""
    if retain:
        source = st.text_input(
            "Image source or permitted source reference",
            key="feedback-source",
            help="Do not enter a URL if the source terms prohibit storing it.",
        ).strip()
        license_name = st.text_input(
            "Image licence or rights basis",
            key="feedback-license",
        ).strip()
        attribution = st.text_input(
            "Required attribution, if any",
            key="feedback-attribution",
        ).strip()
    if st.button("Submit feedback", key="submit-feedback"):
        feedback_id = submit_feedback(
            settings.database_path,
            settings.feedback_dir / "images",
            response=response,
            predicted_label=str(prediction["predicted_label"]),
            model_version_id=str(prediction["model_version_id"]),
            model_score=float(prediction["model_score"]),
            original_typed_name=original_name,
            resolved_taxon_version_id=taxon_id,
            resolver_method=resolver_method,
            image_bytes=image_bytes,
            retention_consent=retain,
            training_use_consent=training_use,
            source=source,
            license=license_name,
            attribution=attribution,
        )
        st.success(f"Feedback saved as pending review ({feedback_id}).")
        st.session_state.pop("feedback-response", None)
        st.session_state.pop("submitted-image-hash", None)


def _page_classify() -> None:
    _render_page_heading(
        "Classify an animal image",
        "Run a local model on a single image. Experimental candidate results are kept separate from production activation.",
        "Field analysis / image classification",
    )
    active = _active_model()
    inference_modes = [
        "Production (active model)",
        "Local development/demo (candidate)",
    ]
    mode_key = "inference-mode"
    if active is None:
        st.session_state[mode_key] = inference_modes[1]
    inference_mode = st.radio(
        "Inference mode",
        inference_modes,
        index=0 if active is not None else 1,
        key=mode_key,
        help=(
            "Development/demo mode runs a registered candidate locally without "
            "promoting it or changing the production active-model pointer."
        ),
        horizontal=True,
    )
    is_demo = inference_mode == "Local development/demo (candidate)"
    selected_model = active
    if is_demo:
        st.warning(
            "EXPERIMENTAL LOCAL DEMO — predictions are not production-approved. "
            "This mode never promotes or activates a model."
        )
        candidates = list_candidate_models(settings.database_path)
        valid_candidates: list[dict[str, str]] = []
        invalid_candidates: list[str] = []
        for candidate in candidates:
            checkpoint = (
                settings.root_dir / str(candidate["artifact_path"])
            ).resolve()
            if not checkpoint.is_relative_to(settings.root_dir):
                invalid_candidates.append(
                    f"{candidate['model_version_id']}: checkpoint path is outside the project"
                )
                continue
            if not checkpoint.is_file():
                invalid_candidates.append(
                    f"{candidate['model_version_id']}: checkpoint file is missing"
                )
                continue
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != candidate["artifact_sha256"]:
                invalid_candidates.append(
                    f"{candidate['model_version_id']}: checkpoint SHA-256 does not match registry"
                )
                continue
            valid_candidates.append(candidate)

        for issue in invalid_candidates:
            st.error(f"Candidate unavailable: {issue}")
        if not candidates:
            st.info("No registered candidate models are available for the local demo.")
        elif not valid_candidates:
            st.error(
                "Registered candidates exist, but none has a valid checkpoint. "
                "The uploader remains available, but inference requires a valid candidate."
            )
        else:
            architecture_names = {"resnet18": "ResNet18", "scratch_cnn": "ScratchCNN"}
            candidate_labels = [
                (
                    f"{architecture_names.get(str(candidate['architecture']), candidate['architecture'])}"
                    f" — {candidate['model_version_id']}"
                )
                for candidate in valid_candidates
            ]
            selected_label = st.selectbox(
                "Candidate model",
                candidate_labels,
                key="demo-candidate-model",
            )
            selected_model = next(
                candidate
                for candidate, label in zip(valid_candidates, candidate_labels)
                if label == selected_label
            )
    elif active is None:
        st.warning(
            "No model is registered as active. Production inference is unavailable. "
            "Choose Local development/demo (candidate) to run a registered candidate "
            "without promoting it."
        )
        return

    model_available = selected_model is not None
    checkpoint_path: Path | None = None
    if model_available:
        checkpoint_path = (
            settings.root_dir / str(selected_model["artifact_path"])
        ).resolve()
        if not checkpoint_path.is_relative_to(settings.root_dir):
            st.error("The selected checkpoint path is outside the project directory.")
            model_available = False
        elif not checkpoint_path.is_file():
            st.error(f"The selected model artifact is missing: {checkpoint_path}")
            model_available = False
        else:
            actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if actual_hash != selected_model["artifact_sha256"]:
                st.error("The selected model artifact hash does not match its registry record.")
                model_available = False

    model_identity = (
        str(selected_model["model_version_id"]) if selected_model is not None else "unavailable"
    )
    selection_key = f"{inference_mode}:{model_identity}"
    if st.session_state.get("selected-model-key") != selection_key:
        st.session_state.pop("current-prediction", None)
        st.session_state["selected-model-key"] = selection_key

    device_choice = st.selectbox("Inference device", ["auto", "cuda", "cpu"], index=0)
    uploaded = st.file_uploader(
        "Upload an animal image (JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
        key="classify-image-upload",
    )
    run_classification = st.button(
        "Classify image",
        disabled=uploaded is None or not model_available,
        type="primary",
    )
    if uploaded is None:
        return
    image_bytes = uploaded.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    try:
        preview = decode_sanitized_image(image_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.image(preview, caption="Image preview", use_container_width=True)
    if not model_available:
        st.info("Image preview is available. Register or repair a candidate model to run inference.")
        return
    prediction_key = "current-prediction"
    if st.session_state.get("current-image-hash") != image_hash:
        st.session_state.pop(prediction_key, None)
        st.session_state["current-image-hash"] = image_hash
        st.session_state.pop("feedback-response", None)

    if run_classification:
        try:
            assert checkpoint_path is not None
            (model, metadata), device = _load_model(str(checkpoint_path), device_choice)
            result = predict_image(model, metadata, image_bytes, device=device)
            assert selected_model is not None
            result["model_version_id"] = selected_model["model_version_id"]
            st.session_state[prediction_key] = result
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(f"Could not run inference: {exc}")

    prediction = st.session_state.get(prediction_key)
    if not prediction:
        return
    with st.container(border=True):
        prediction_columns = st.columns([1.5, 1])
        with prediction_columns[0]:
            st.caption("PREDICTED ANIMAL")
            st.subheader(str(prediction["predicted_label"]))
            st.write(f"Model version: `{prediction['model_version']}`")
            st.write(
                f"Architecture: {prediction['architecture']} · device: {prediction['device']}"
            )
        with prediction_columns[1]:
            st.metric(
                "Confidence score · uncalibrated",
                f"{float(prediction['model_score']):.1%}",
            )
            st.caption("Softmax model score; not a calibrated probability.")
    st.caption(
        "This model only selects among its eight supported classes. No approved "
        "abstention threshold or out-of-distribution detector is available; this result "
        "does not establish that the image contains a supported animal."
    )
    alternatives = [
        item
        for item in prediction["top_outputs"]
        if item["label"] != prediction["predicted_label"]
    ]
    if alternatives:
        st.markdown("#### Other class predictions")
        for item in alternatives:
            st.progress(
                float(item["score"]),
                text=f"{item['label']} · {float(item['score']):.1%}",
            )
    if prediction["architecture"] == "resnet18":
        if st.button("View experimental Grad-CAM"):
            try:
                (model, metadata), device = _load_model(
                    str(checkpoint_path), device_choice
                )
                model_input, model_view = prepare_model_input(metadata, image_bytes)
                heatmap = generate_gradcam(
                    model,
                    model_input.to(device),
                    model_view,
                    class_index=next(
                        index
                        for index, label in enumerate(metadata["class_names"])
                        if label == prediction["predicted_label"]
                    ),
                )
                st.image(heatmap, use_container_width=True)
                st.caption(
                    "Experimental Grad-CAM on the processed model input. "
                    "This is not a causal explanation or proof of identification."
                )
            except (RuntimeError, ValueError) as exc:
                st.error(f"Could not generate Grad-CAM: {exc}")
    else:
        st.caption("Grad-CAM is not implemented for this architecture.")

    if is_demo:
        st.info(
            "Demo inference only. Feedback submission is disabled in this mode so "
            "experimental candidate outputs are not mixed with reviewed project feedback."
        )
        return

    response_options = {
        "Yes": "yes",
        "No": "no",
        "Not sure": "unsure",
        "Unsupported / not listed": "unsupported",
    }
    columns = st.columns(4)
    for column, (label, response) in zip(columns, response_options.items()):
        if column.button(label, key=f"response-{response}"):
            st.session_state["feedback-response"] = response
    response = st.session_state.get("feedback-response")
    if response is None:
        return

    resolved_taxon = None
    typed_name = ""
    resolver_method = ""
    if response in {"no", "unsupported"}:
        st.markdown("#### Correction (optional)")
        resolved_taxon, typed_name, resolver_method = _render_taxon_search(
            f"correction-{image_hash[:10]}"
        )
    retain = st.checkbox(
        "I consent to retain this sanitized image for project review.",
        key="retain-feedback-image",
    )
    training_use = st.checkbox(
        "I separately consent to use the retained image in a future training dataset.",
        disabled=not retain,
        key="training-feedback-image",
    )
    if not retain:
        st.caption("Feedback is saved without retaining the uploaded image.")
    if retain:
        st.caption(
            "The retained copy is sanitized and stored separately from SQLite. Source, "
            "licence and attribution must still be reviewed before curation."
        )
    _submit_feedback(
        str(response),
        prediction,
        image_bytes=image_bytes,
        retain=retain,
        training_use=training_use,
        original_name=typed_name,
        taxon_id=resolved_taxon,
        resolver_method=resolver_method,
    )


def _page_taxonomy() -> None:
    _render_page_heading(
        "Taxonomy catalogue",
        "Search a versioned animal-name reference independently of what the image model can recognize.",
        "Reference / names and taxonomy",
    )
    connection = connect_database(settings.database_path)
    try:
        imports = connection.execute(
            """SELECT source_namespace, release, license, attribution, imported_at
               FROM taxonomy_imports ORDER BY imported_at DESC"""
        ).fetchall()
        count = connection.execute("SELECT COUNT(*) FROM taxa").fetchone()[0]
    finally:
        connection.close()
    summary_columns = st.columns([1, 2])
    summary_columns[0].metric("Imported taxon records", f"{count:,}")
    summary_columns[1].metric("Versioned source imports", len(imports))
    if not imports:
        st.info(
            "No reviewed taxonomy snapshot is installed. Name searches remain unresolved "
            "until a source and release are reviewed and imported."
        )
    else:
        st.markdown("#### Imported source releases")
        st.dataframe(
            [
                {
                    "Source": item["source_namespace"],
                    "Release": item["release"],
                    "Licence": item["license"],
                    "Attribution": item["attribution"],
                    "Imported": item["imported_at"],
                }
                for item in imports
            ],
            hide_index=True,
            use_container_width=True,
        )
    st.markdown("#### Find an animal name")
    st.caption(
        "Exact and fuzzy candidates are suggestions only. Confirm the intended taxon before use."
    )
    with st.container(border=True):
        taxon_id, query, method = _render_taxon_search("catalogue-search")
    if taxon_id:
        st.success(f"Confirmed selection: `{taxon_id}` for “{query}” ({method}).")


def _page_curation(records: list[dict[str, object]]) -> None:
    st.subheader("Curated dataset export")
    protected_manifest = settings.manifests_dir / "dataset_manifest.csv"
    if not protected_manifest.is_file():
        st.info("The frozen supported dataset manifest is required before curation export.")
        return
    eligible_ids = []
    try:
        for item in records:
            if item["verification_status"] != "verified":
                continue
            eligibility = check_training_eligibility(
                settings.database_path,
                str(item["feedback_id"]),
                protected_test_manifest=protected_manifest,
            )
            if eligibility["eligible"]:
                eligible_ids.append(str(item["feedback_id"]))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.error(f"Could not check curation eligibility: {exc}")
        return
    if not eligible_ids:
        st.info("No feedback currently satisfies all curation and frozen-test checks.")
        return
    selected_ids = st.multiselect("Eligible feedback", eligible_ids)
    if st.button("Export selected curation manifest", disabled=not selected_ids):
        destination = settings.manifests_dir / f"curated-feedback-{uuid.uuid4().hex[:12]}.csv"
        try:
            result = export_curated_manifest(
                settings.database_path,
                selected_ids,
                protected_test_manifest=protected_manifest,
                destination=destination,
            )
            st.success(
                f"Created {result['item_count']} item manifest at "
                f"`{result['manifest_path']}` (SHA-256 `{result['manifest_sha256']}`)."
            )
            st.download_button(
                "Download curated manifest",
                data=destination.read_bytes(),
                file_name=destination.name,
                mime="text/csv",
            )
        except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
            st.error(f"Could not export curation: {exc}")


def _page_review() -> None:
    _render_page_heading(
        "Feedback review",
        "Inspect submitted corrections, record review decisions, and export only eligible, consented examples.",
        "Curation / human review",
    )
    st.warning(
        "This local reviewer screen has no authentication or authorization. A reviewer "
        "name is an attribution note, not proof of identity."
    )
    records = list_feedback(settings.database_path)
    pending = [
        item for item in records
        if item["verification_status"] in {"pending", "needs_review"}
        or (
            item["verification_status"] == "verified"
            and item.get("image_path")
            and (
                item.get("rights_review_status") != "approved"
                or item.get("quality_review_status") != "approved"
            )
        )
    ]
    review_counts = {
        status: sum(item["verification_status"] == status for item in records)
        for status in ("pending", "needs_review", "verified", "rejected")
    }
    count_columns = st.columns(4)
    for column, status in zip(count_columns, review_counts):
        column.metric(status.replace("_", " ").title(), review_counts[status])
    st.markdown("#### Queue awaiting review")
    if not pending:
        st.info("There are no feedback items awaiting review.")
        _page_curation(records)
        return
    selected_id = st.selectbox(
        "Feedback item",
        [str(item["feedback_id"]) for item in pending],
        format_func=lambda key: (
            f"{key[:8]} · {next(item['response'] for item in pending if item['feedback_id'] == key)}"
        ),
    )
    record = next(item for item in pending if item["feedback_id"] == selected_id)
    with st.container(border=True):
        st.markdown("#### Submission details")
        st.json({
            "response": record["response"],
            "predicted_label": record["predicted_label"],
            "model_version_id": record["model_version_id"],
            "model_score": record["model_score"],
            "typed_name": record["original_typed_name"],
            "resolved_taxon_version_id": record["resolved_taxon_version_id"],
            "retention_consent": bool(record["retention_consent"]),
            "training_use_consent": bool(record["training_use_consent"]),
            "source": record.get("source"),
            "license": record.get("license"),
            "duplicate_of_feedback_id": record["duplicate_of_feedback_id"],
        })
    if record.get("image_path"):
        image_path = (settings.root_dir / str(record["image_path"])).resolve()
        if image_path.is_relative_to(settings.root_dir) and image_path.is_file():
            st.image(image_path, caption="Sanitized retained image", width=400)
    with st.form(f"review-{selected_id}"):
        decision = st.selectbox("Review decision", ["verified", "needs_review", "rejected"])
        reviewer = st.text_input("Reviewer reference (self-declared)")
        notes = st.text_area("Review notes")
        quality_approved = st.checkbox("I reviewed image quality and suitability")
        rights_approved = st.checkbox(
            "I reviewed the source and licence and confirm they permit this use"
        )
        submit_review = st.form_submit_button("Save review decision")
    if submit_review:
        try:
            review_feedback(
                settings.database_path,
                selected_id,
                decision=decision,
                reviewer_reference=reviewer,
                notes=notes,
                rights_approved=rights_approved and decision == "verified",
                quality_approved=quality_approved and decision == "verified",
            )
            st.success("Review decision saved.")
            st.rerun()
        except (KeyError, ValueError) as exc:
            st.error(str(exc))

    if record.get("image_path"):
        confirmed = st.checkbox(
            "I confirm withdrawal of image retention and deletion of this retained copy.",
            key=f"withdraw-confirm-{selected_id}",
        )
        if st.button("Withdraw retained image", disabled=not confirmed):
            try:
                result = withdraw_retained_image(settings.database_path, selected_id)
                st.warning(
                    "The retained image was removed. Affected manifests: "
                    + json.dumps(result["affected_manifests"])
                    + ". Any models trained from those manifests require lineage review."
                )
                st.rerun()
            except (KeyError, OSError) as exc:
                st.error(f"Could not withdraw the retained image: {exc}")
    _page_curation(records)


def _page_models() -> None:
    _render_page_heading(
        "Models and experiments",
        "Inspect local checkpoints, run candidates in demo mode, and keep production activation behind explicit review gates.",
        "Registry / versions and lifecycle",
    )
    active = _active_model()
    if active:
        with st.container(border=True):
            st.success(
                f"Production active model: {active['model_version_id']} "
                f"({active['architecture']})"
            )
            st.write(f"Artifact: {active['artifact_path']}")
            st.write(f"Artifact SHA-256: `{active['artifact_sha256']}`")
            st.json(json.loads(str(active["metadata_json"])))
    else:
        st.info(
            "No model is registered as active; production inference is disabled. "
            "Registered candidates may still be run in Local development/demo mode "
            "from the Classify view."
        )

    connection = connect_database(settings.database_path)
    try:
        versions = [
            dict(row)
            for row in connection.execute(
                """SELECT model_version_id, architecture, lifecycle_state, artifact_path,
                          artifact_sha256, metadata_json, decision_by, decision_rationale
                   FROM model_versions ORDER BY created_at DESC"""
            )
        ]
        policies = [
            dict(row)
            for row in connection.execute(
                """SELECT policy_id, approved_by, approval_reference, criteria_json
                   FROM promotion_policies ORDER BY created_at DESC"""
            )
        ]
    finally:
        connection.close()

    counts = {
        lifecycle: sum(version["lifecycle_state"] == lifecycle for version in versions)
        for lifecycle in ("candidate", "active", "retired", "rejected")
    }
    model_columns = st.columns(4)
    for column, lifecycle in zip(model_columns, counts):
        column.metric(lifecycle.title(), counts[lifecycle])

    st.markdown("#### Registered model versions")
    if versions:
        st.dataframe(
            [
                {
                    "Version": version["model_version_id"],
                    "Architecture": version["architecture"],
                    "Lifecycle": version["lifecycle_state"],
                    "Artifact": version["artifact_path"],
                    "SHA-256": version["artifact_sha256"],
                    "Test report": (
                        "Attached"
                        if "test_evaluation"
                        in json.loads(version["metadata_json"])
                        else "Not attached"
                    ),
                }
                for version in versions
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Register a trained checkpoint")
    checkpoints = sorted((settings.root_dir / "models" / "runs").glob("**/best-*.pt"))
    registered_paths = {
        str(Path(str(version["artifact_path"])).resolve()).casefold()
        for version in versions
    }
    unregistered_checkpoints = [
        path
        for path in checkpoints
        if str(path.resolve()).casefold() not in registered_paths
    ]
    if unregistered_checkpoints:
        checkpoint_paths = {
            str(path.relative_to(settings.root_dir)): path
            for path in unregistered_checkpoints
        }
        selected_checkpoint = st.selectbox(
            "Local checkpoint",
            list(checkpoint_paths),
            key="registry-checkpoint",
        )
        if st.button("Validate and register candidate"):
            checkpoint_path = checkpoint_paths[selected_checkpoint]
            architecture = checkpoint_path.stem.removeprefix("best-")
            report_path = (
                settings.root_dir
                / "reports"
                / "metrics"
                / f"{architecture}-test-metrics.json"
            )
            try:
                result = register_candidate(
                    settings.root_dir,
                    settings.database_path,
                    checkpoint_path,
                    evaluation_path=report_path if report_path.is_file() else None,
                )
                st.success(
                    f"Registered {result['model_version_id']} as a candidate; "
                    f"test evaluation attached: {result['has_test_evaluation']}."
                )
                st.rerun()
            except (FileNotFoundError, ValueError) as exc:
                st.error(f"Could not register candidate: {exc}")
    elif checkpoints:
        st.info("All local best checkpoints are already registered.")
    else:
        st.info("No local run checkpoints were found.")
    st.caption(
        "Local run checkpoints are not active until a manually approved policy passes "
        "the frozen-test and regression checks. Reviewer references are self-declared, "
        "not authenticated."
    )

    st.subheader("Approved promotion criteria")
    if policies:
        for policy in policies:
            st.write(
                f"`{policy['policy_id']}` · approval reference: "
                f"{policy['approval_reference']} · approver: {policy['approved_by']}"
            )
            st.json(json.loads(policy["criteria_json"]))
    else:
        st.info("No promotion criteria are configured; no model can be promoted.")
    with st.form("promotion-policy-form"):
        approver = st.text_input("Approver reference")
        approval_reference = st.text_input("Reference to the prior team decision")
        criterion_fields = {
            "minimum_test_accuracy": "Minimum test accuracy (0 to 1)",
            "minimum_test_macro_f1": "Minimum test macro F1 (0 to 1)",
            "minimum_per_class_recall": "Minimum per-class recall (0 to 1)",
            "maximum_accuracy_drop": "Maximum accuracy drop from active (0 to 1)",
            "maximum_macro_f1_drop": "Maximum macro-F1 drop from active (0 to 1)",
            "maximum_per_class_recall_drop": "Maximum per-class recall drop (0 to 1)",
            "maximum_inference_ms_per_image": "Maximum inference milliseconds per image",
            "maximum_checkpoint_size_bytes": "Maximum checkpoint size in bytes",
        }
        criterion_values = {
            key: st.text_input(label, key=f"promotion-policy-{key}")
            for key, label in criterion_fields.items()
        }
        policy_approval = st.checkbox(
            "I confirm these criteria were approved before evaluating this candidate."
        )
        create_policy = st.form_submit_button("Save promotion criteria")
    if create_policy:
        if not policy_approval:
            st.error("Prior criteria approval must be confirmed before saving.")
        else:
            try:
                criteria = {
                    key: (
                        int(value)
                        if key == "maximum_checkpoint_size_bytes"
                        else float(value)
                    )
                    for key, value in criterion_values.items()
                }
                policy_id = create_promotion_policy(
                    settings.database_path,
                    criteria,
                    approved_by=approver,
                    approval_reference=approval_reference,
                )
                st.success(f"Saved promotion policy {policy_id}.")
                st.rerun()
            except (ValueError, TypeError) as exc:
                st.error(f"Could not save promotion criteria: {exc}")

    candidates = [
        version for version in versions if version["lifecycle_state"] == "candidate"
    ]
    if candidates and policies:
        st.subheader("Promote candidate")
        with st.form("promotion-form"):
            selected_candidate = st.selectbox(
                "Candidate",
                [version["model_version_id"] for version in candidates],
            )
            selected_policy = st.selectbox(
                "Approved policy",
                [policy["policy_id"] for policy in policies],
            )
            promotion_actor = st.text_input("Decision-maker reference")
            promotion_rationale = st.text_area("Promotion rationale")
            submit_promotion = st.form_submit_button("Evaluate criteria and promote")
        if submit_promotion:
            try:
                result = promote_candidate(
                    settings.root_dir,
                    settings.database_path,
                    selected_candidate,
                    policy_id=selected_policy,
                    actor_reference=promotion_actor,
                    rationale=promotion_rationale,
                )
                st.success(f"Model {result['model_version_id']} is now active.")
                st.rerun()
            except (FileNotFoundError, KeyError, ValueError) as exc:
                st.error(f"Promotion was not completed: {exc}")

    if candidates:
        rejected_candidate = st.selectbox(
            "Candidate to reject",
            [version["model_version_id"] for version in candidates],
        )
        rejection_actor = st.text_input("Rejection decision-maker reference")
        rejection_rationale = st.text_area("Rejection rationale")
        if st.button("Reject candidate"):
            try:
                reject_candidate(
                    settings.database_path,
                    rejected_candidate,
                    actor_reference=rejection_actor,
                    rationale=rejection_rationale,
                )
                st.success(f"Rejected {rejected_candidate}.")
                st.rerun()
            except (KeyError, ValueError) as exc:
                st.error(f"Candidate rejection was not completed: {exc}")

    retired = [
        version for version in versions if version["lifecycle_state"] == "retired"
    ]
    if retired:
        st.subheader("Rollback")
        rollback_target = st.selectbox(
            "Previously active model",
            [version["model_version_id"] for version in retired],
        )
        rollback_actor = st.text_input("Rollback decision-maker reference")
        rollback_rationale = st.text_area("Rollback rationale")
        if st.button("Restore selected model"):
            try:
                rollback_model(
                    settings.root_dir,
                    settings.database_path,
                    rollback_target,
                    actor_reference=rollback_actor,
                    rationale=rollback_rationale,
                )
                st.success(f"Restored {rollback_target} as active.")
                st.rerun()
            except (FileNotFoundError, KeyError, ValueError) as exc:
                st.error(f"Rollback was not completed: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="Wildlife Intelligence",
        page_icon="W",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_APP_CSS, unsafe_allow_html=True)
    connection = connect_database(settings.database_path)
    connection.close()
    with st.sidebar:
        st.markdown(
            """
            <div class="wild-brand">
              <div class="wild-brand-mark">Field research tools</div>
              <div class="wild-brand-name">Wildlife<br>Intelligence</div>
              <div class="wild-brand-status">Local academic research prototype</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Workspace",
            ["Classify", "Taxonomy", "Review", "Models"],
            label_visibility="visible",
        )
        st.divider()
        st.caption("Local inference · Project data stays on this machine")

    st.markdown(
        """
        <div class="wild-masthead">
          <div>Wildlife &amp; Animal Intelligence System</div>
          <span>Local research workspace</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if page == "Classify":
        _page_classify()
    elif page == "Taxonomy":
        _page_taxonomy()
    elif page == "Review":
        _page_review()
    else:
        _page_models()
