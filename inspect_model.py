# import onnxruntime as ort
# import json
# import numpy as np

# session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

# print("=== ALL INPUTS ===")
# for i, inp in enumerate(session.get_inputs()):
#     print(f"[{i}] name={inp.name} | shape={inp.shape} | type={inp.type}")

# print("\n=== ALL OUTPUTS ===")
# for i, out in enumerate(session.get_outputs()):
#     print(f"[{i}] name={out.name} | shape={out.shape} | type={out.type}")

# print("\n=== STATE.JSON KEYS ===")
# with open("state.json") as f:
#     state = json.load(f)
# for k, v in state.items():
#     print(f"  {k} -> shape={np.array(v).shape}")

# model_keys = {inp.name for inp in session.get_inputs()}
# state_keys = set(state.keys()) | {'arg_0.1'}
# missing = model_keys - state_keys
# print(f"\ کلیدهای گمشده: {missing}")
