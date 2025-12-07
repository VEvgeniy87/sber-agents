import sys
sys.path.append('.')
from sentence_transformers import SentenceTransformer

print("Testing model loading...")
try:
    model = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')
    print("Model loaded successfully.")
    test_embedding = model.encode(["Hello world"])
    print(f"Embedding shape: {test_embedding.shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()