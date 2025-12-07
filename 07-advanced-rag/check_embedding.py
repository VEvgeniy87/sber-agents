from sentence_transformers import SentenceTransformer
import sys

print("Loading multilingual-e5-base...")
try:
    model = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')
    print("Model loaded successfully.")
    print(f"Model dimension: {model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)