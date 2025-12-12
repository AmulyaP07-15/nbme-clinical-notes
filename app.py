"""
Streamlit Web App for Clinical Note Summarization
Run with: streamlit run app.py
"""

import streamlit as st
import time
from src.model import ClinicalNoteSummarizer
from src.utils import get_statistics, get_example_notes

st.set_page_config(
    page_title="Clinical Note Summarizer",
    page_icon="🏥",
    layout="wide"
)


@st.cache_resource
def load_model(model_name='t5-base'):  # Changed from t5-small
    """Load model (cached so it only loads once)"""
    return ClinicalNoteSummarizer(model_name=model_name)


def main():
    st.title("🏥 Clinical Note Summarizer")
    st.markdown("### Automated abstractive summarization using T5 transformer")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")

        # t5-base as default (index=0 means first option)
        model_choice = st.selectbox("Model", ["t5-base", "t5-small"], index=0)

        # Increased limits and defaults
        max_length = st.slider("Max Summary Length (tokens)", 100, 500, 400, 20)
        min_length = st.slider("Min Summary Length (tokens)", 50, 200, 150, 10)

        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("This app uses T5 to generate abstractive summaries of clinical notes. "
                "400 tokens ≈ 250-300 words.")

        st.markdown("### ⚙️ Current Settings")
        st.code(f"Model: {model_choice}\nMax: {max_length} tokens\nMin: {min_length} tokens")

    with st.spinner(f"Loading {model_choice} model..."):
        summarizer = load_model(model_choice)

    model_info = summarizer.get_model_info()
    st.success(f"✓ Model loaded on {model_info['device']}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Input: Clinical Note")
        examples = get_example_notes()
        example_choice = st.selectbox("Load Example", ["Custom"] + list(examples.keys()))

        default_text = "" if example_choice == "Custom" else examples[example_choice]

        clinical_note = st.text_area(
            "Enter clinical note:",
            value=default_text,
            height=300,
            placeholder="Paste or type the clinical note here..."
        )

        st.caption(f"📊 Characters: {len(clinical_note)} | Words: {len(clinical_note.split())}")
        generate_btn = st.button("🚀 Generate Summary", type="primary", use_container_width=True)

    with col2:
        st.subheader("✨ Output: Abstractive Summary")

        if generate_btn:
            if not clinical_note.strip():
                st.error("⚠️ Please enter a clinical note first!")
            else:
                with st.spinner("Generating summary..."):
                    start_time = time.time()

                    # Pass the slider values to the summarizer
                    summary = summarizer.summarize(
                        clinical_note,
                        max_length=max_length,
                        min_length=min_length
                    )

                    elapsed_time = time.time() - start_time

                st.markdown("### Generated Summary:")
                st.info(summary)

                stats = get_statistics(clinical_note, summary)
                st.markdown("### 📊 Statistics:")
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Original", f"{stats['original_length']} chars")
                with col_b:
                    st.metric("Summary", f"{stats['summary_length']} chars")
                with col_c:
                    st.metric("Reduction", f"{stats['reduction_percentage']:.1f}%")

                # Additional stats
                st.caption(f"⏱️ Generated in {elapsed_time:.2f} seconds | "
                          f"Summary words: {len(summary.split())}")

                st.download_button(
                    label="📥 Download Summary",
                    data=f"ORIGINAL NOTE:\n{clinical_note}\n\nSUMMARY:\n{summary}",
                    file_name="clinical_summary.txt",
                    mime="text/plain"
                )
        else:
            st.info("👆 Enter a clinical note and click 'Generate Summary'")
            st.markdown("**Tip:** For complex notes, use higher max_length (400-500 tokens)")


if __name__ == "__main__":
    main()