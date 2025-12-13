# NBME Clinical Notes Analysis


An AI system that converts long, unstructured clinical notes into concise, structured SOAP summaries using abstractive transformer models (T5).

What It Does:

-Reduces clinical note length by ~70%
-Generates human-readable summaries in SOAP format
-Handles noisy, abbreviation-heavy medical text
-Runs on CPU (no GPU required)

Tech Stack:

-Python
-Hugging Face Transformers (T5-base)
-PyTorch
-scikit-learn (TF-IDF)
-Streamlit

How to Run:

pip install -r requirements.txt
python demo.py
# or
streamlit run app.py


Why It’s Interesting:

-Demonstrates real-world abstractive summarization
-Focuses on clinical NLP, not generic text
-Shows practical trade-offs between accuracy, speed, and hallucination control

Disclaimer:

-For research and educational use only. Not for clinical decision-making.
