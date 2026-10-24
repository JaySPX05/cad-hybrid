"""Internal test frontend for trying the model by hand.

    streamlit run app/streamlit_app.py

Not a product UI. Works with lab values only, an ECG image only, or both. Until the
models are trained it shows a notice instead of a prediction.
"""

import tempfile

import streamlit as st

from cad_hybrid.predict import load_config, predict, tabular_threshold

st.set_page_config(page_title="cad-hybrid test app", layout="wide")
st.title("cad-hybrid: model test app")
st.caption("Research prototype. Not for clinical use. Trained on South Asian datasets "
           "(see ATTRIBUTIONS.md).")

with st.sidebar:
    st.header("Options")
    st.selectbox("Model version", ["baseline", "hybrid"], index=1,
                 help="Switches between saved model versions once they exist.")
    show_branches = st.toggle("Show each branch's result", value=True)

col_lab, col_ecg = st.columns(2)

with col_lab:
    st.subheader("Lab values")
    tab_cfg = load_config("tabular")
    features = tab_cfg["data"]["features"]["numeric"] + tab_cfg["data"]["features"]["categorical"]
    use_labs = st.checkbox("Use lab values", value=bool(features))
    lab_values = {}
    if not features:
        st.info("Feature list is empty. Fill it in configs/tabular.yaml after the data audit.")
    for name in features:
        value = st.text_input(name, help="Leave blank if not available")
        if value.strip():
            lab_values[name] = value.strip()
    # TODO: numeric inputs with units/ranges, 'load test-set row', CSV batch scoring

with col_ecg:
    st.subheader("ECG image")
    ecg_file = st.file_uploader("Upload a photo or scan of a 12-lead ECG",
                                type=["jpg", "jpeg", "png"])
    if ecg_file:
        st.image(ecg_file, use_container_width=True)
    # TODO: 'load test image' dropdown

if st.button("Predict", type="primary"):
    ecg_path = None
    if ecg_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(ecg_file.getvalue())
            ecg_path = tmp.name
    try:
        result = predict(lab_values=lab_values if use_labs and features else None,
                         ecg_image_path=ecg_path)
    except (NotImplementedError, FileNotFoundError) as exc:
        st.warning(f"Model not available yet: {exc}")
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.metric("Probability of heart disease", f"{result.probability:.0%}", result.category)
        lab_prob = result.branch_probabilities.get("tabular")
        if lab_prob is not None:
            threshold = tabular_threshold()
            if threshold is not None:
                flag = "above" if lab_prob >= threshold else "below"
                st.caption(f"Lab result {lab_prob:.0%} is {flag} the lab model's decision "
                           f"threshold ({threshold:.0%}). Risk categories are placeholders "
                           "until fusion is set up.")
        if show_branches:
            st.json(result.branch_probabilities)
        # TODO: SHAP bar chart (lab) and Grad-CAM overlay (ECG)
