from app import load_saved_model
m, e, r = load_saved_model()
if m and e:
    print(f"✓ Model and encoder loaded successfully")
    print(f"  Classes: {r.get('classes', [])}")
    print(f"  Accuracy: {r.get('accuracy', 0):.4f}")
else:
    print("✗ Failed to load model")
