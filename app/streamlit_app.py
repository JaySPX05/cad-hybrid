
"""CardioPredict - ECG and Tabular Research Prototype.

The tabular branch predicts general heart disease.
The ECG branch predicts CAD and arrhythmia probabilities.
Both predictions are displayed independently.
"""

import tempfile
from pathlib import Path

import streamlit as st

from cad_hybrid.predict import (
    explain_tabular,
    load_config,
    predict,
    tabular_threshold,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CardioPredict | Research Prototype",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL UI STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --ink: #203746;
    --muted: #71818B;
    --teal: #267B75;
    --line: #DFE7EA;
    --canvas: #F3F6F7;
}

.stApp {
    background: var(--canvas);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1320px;
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 22px;
}

.brand-mark {
    width: 43px;
    height: 43px;
    border-radius: 11px;
    background: #E4F0EE;
    color: #267B75;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 17px;
}

.brand-name {
    font-size: 19px;
    font-weight: 800;
    letter-spacing: 1px;
}

.brand-sub {
    color: #7B8A93;
    font-size: 10px;
    letter-spacing: 1px;
}

.page-heading {
    font-family: 'Manrope', sans-serif;
    font-size: 29px;
    font-weight: 800;
    margin-bottom: 7px;
}

.page-subheading {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 24px;
}

.section-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 13px;
    padding: 20px;
    margin-bottom: 15px;
}

.section-title {
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 5px;
}

.section-description {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 15px;
}

.result-panel {
    background: white;
    border: 1px solid var(--line);
    border-radius: 13px;
    padding: 20px;
    min-height: 180px;
}

.result-heading {
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.result-value {
    color: var(--teal);
    font-family: 'Manrope', sans-serif;
    font-size: 34px;
    font-weight: 800;
    margin-top: 12px;
}

.helper-box {
    background: #F7F9FA;
    border: 1px solid #E4EAED;
    border-radius: 10px;
    padding: 15px;
    color: #60747F;
    font-size: 12px;
    line-height: 1.7;
}

.stButton > button {
    background: #267B75;
    color: white;
    border: 1px solid #267B75;
    border-radius: 8px;
    min-height: 45px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #1E655F;
    color: white;
    border-color: #1E655F;
}

.footer {
    color: #89969D;
    font-size: 10px;
    text-align: center;
    padding: 25px 0 5px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="brand-row">
    <div class="brand-mark">CP</div>
    <div>
        <div class="brand-name">CARDIOPREDICT</div>
        <div class="brand-sub">CARDIOVASCULAR RESEARCH PLATFORM</div>
    </div>
</div>

<div class="page-heading">
    Cardiovascular Health Assessment
</div>

<div class="page-subheading">
    Independent machine-learning predictions using clinical
    measurements and ECG image analysis.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

try:
    tab_cfg = load_config("tabular")
    features = (
        tab_cfg["data"]["features"]["numeric"]
        + tab_cfg["data"]["features"]["categorical"]
    )

except (FileNotFoundError, KeyError, TypeError) as error:
    st.error(f"Could not load tabular configuration: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("Analysis Options")

    explain_labs = st.toggle(
        "Show SHAP explanation",
        value=False,
        help="Explain the tabular model prediction."
    )

    st.divider()

    st.caption("Research prototype — not for clinical diagnosis.")


# --------------------------------------------------
# INPUT SECTIONS
# --------------------------------------------------

lab_col, ecg_col = st.columns(2, gap="large")

lab_values = {}

with lab_col:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">01. Patient Measurements</div>
        <div class="section-description">
            Enter available clinical and laboratory measurements.
        </div>
    </div>
    """, unsafe_allow_html=True)

    use_labs = st.checkbox(
        "Enable tabular prediction",
        value=False
    )

    if use_labs:
        with st.container(border=True):
            st.markdown("**Clinical and laboratory inputs**")

            for name in features:
                value = st.text_input(
                    name.replace("_", " ").title(),
                    key=f"lab_{name}",
                    placeholder="Enter value"
                )

                if value.strip():
                    lab_values[name] = value.strip()


with ecg_col:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">02. ECG Image Analysis</div>
        <div class="section-description">
            Upload a 12-lead ECG image for CAD and arrhythmia analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    ecg_file = st.file_uploader(
        "Upload ECG image",
        type=["jpg", "jpeg", "png"]
    )

    if ecg_file:
        st.image(
            ecg_file,
            caption="Uploaded ECG image",
            use_container_width=True
        )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.write("")

if st.button(
    "Run Cardiovascular Analysis",
    type="primary",
    use_container_width=True
):

    selected_labs = (
        lab_values
        if use_labs and features
        else None
    )

    if selected_labs is None and ecg_file is None:
        st.warning("Enter patient measurements or upload an ECG image.")

    else:
        ecg_path = None

        try:
            if ecg_file is not None:
                suffix = Path(ecg_file.name).suffix or ".png"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as tmp:
                    tmp.write(ecg_file.getvalue())
                    ecg_path = tmp.name

            with st.spinner("Analyzing the available inputs..."):

                result = predict(
                    lab_values=selected_labs,
                    ecg_image_path=ecg_path,
                )

            # --------------------------------------
            # RESULTS
            # --------------------------------------

            st.divider()
            st.subheader("03. Prediction Results")

            result_lab, result_ecg = st.columns(2, gap="large")

            # --------------------------------------
            # TABULAR RESULTS
            # --------------------------------------

            with result_lab:
                st.markdown("""
                <div class="result-panel">
                    <div class="result-heading">
                        TABULAR MODEL
                    </div>
                    <div class="section-title">
                        General Heart Disease
                    </div>
                """, unsafe_allow_html=True)

                if result.tabular_probability is not None:
                    st.markdown(
                        f'<div class="result-value">'
                        f'{result.tabular_probability:.1%}</div>',
                        unsafe_allow_html=True,
                    )

                    threshold = tabular_threshold()

                    if threshold is not None:
                        st.caption(
                            f"Model decision threshold: {threshold:.1%}"
                        )

                else:
                    st.info("Tabular prediction was not requested.")

                st.markdown("</div>", unsafe_allow_html=True)

            # --------------------------------------
            # ECG RESULTS
            # --------------------------------------

            with result_ecg:
                st.markdown("""
                <div class="result-panel">
                    <div class="result-heading">
                        ECG MODEL
                    </div>
                    <div class="section-title">
                        ECG-based CAD Prediction
                    </div>
                """, unsafe_allow_html=True)

                if result.cad_probability is not None:

                    # CAD probability
                    st.markdown(
                        f'<div class="result-value">'
                        f'{result.cad_probability:.1%}</div>',
                        unsafe_allow_html=True,
                    )

                    st.caption("Calibrated CAD probability")

                    # Arrhythmia probability
                    if result.arrhythmia_probability is not None:

                        st.divider()

                        st.markdown("**Arrhythmia Probability**")

                        st.metric(
                            label="ECG arrhythmia score",
                            value=f"{result.arrhythmia_probability:.1%}",
                        )

                        st.warning(
                            "The ECG model has produced a separate "
                            "arrhythmia probability. This is a "
                            "research-model estimate, not a clinical "
                            "diagnosis. Please consult a qualified "
                            "healthcare professional for interpretation."
                        )

                    else:
                        st.info(
                            "Arrhythmia probability is unavailable "
                            "for this prediction."
                        )

                else:
                    st.info("ECG prediction was not requested.")

                st.markdown("</div>", unsafe_allow_html=True)

            # --------------------------------------
            # EXPLANATION
            # --------------------------------------

            st.markdown("""
            <div class="helper-box">
                <b>Important distinction:</b><br>
                The tabular model predicts general heart disease.
                The ECG model provides separate CAD and arrhythmia
                probabilities. These outputs are not combined into
                one score. The predictions are intended for research
                use only and are not a substitute for clinical assessment.
            </div>
            """, unsafe_allow_html=True)

            # --------------------------------------
            # SHAP EXPLANATION
            # --------------------------------------

            if explain_labs and selected_labs is not None:
                st.divider()
                st.subheader("Tabular Model Explanation")

                from cad_hybrid.explain.shap_tabular import plot_patient

                with st.spinner("Generating SHAP explanation..."):
                    explanation = explain_tabular(selected_labs)

                st.pyplot(plot_patient(explanation))

        except (ValueError, FileNotFoundError, NotImplementedError) as error:
            st.error(f"Prediction could not be completed: {error}")

        except Exception as error:
            st.error(f"Unexpected prediction error: {error}")

        finally:
            if ecg_path and Path(ecg_path).exists():
                Path(ecg_path).unlink()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    CARDIOPREDICT · MACHINE LEARNING RESEARCH PROTOTYPE<br>
    Independent ECG and tabular model predictions · Research use only
</div>
""", unsafe_allow_html=True)
