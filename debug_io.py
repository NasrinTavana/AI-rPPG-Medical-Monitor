# debug_io.py
import onnxruntime as ort
import json

session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

print(f"تعداد inputs: {len(session.get_inputs())}")
print(f"تعداد outputs: {len(session.get_outputs())}")

print("\n=== OUTPUTS ===")
for i, o in enumerate(session.get_outputs()):
    print(f"[{i}] {o.name} | {o.shape}")

with open("state_full.json") as f:
    state = json.load(f)
print(f"\nتعداد کلیدهای state_full.json: {len(state)}")
