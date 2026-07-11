# # fix_state.py
# import onnxruntime as ort
# import json
# import numpy as np

# session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

# # بارگذاری state فعلی
# with open("state.json") as f:
#     old_state = json.load(f)

# # ساخت state کامل با صفر برای کلیدهای گمشده
# new_state = {}
# for inp in session.get_inputs():
#     name = inp.name
#     if name == 'arg_0.1':
#         continue  # این ورودی تصویره، state نیست
    
#     shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    
#     if name in old_state:
#         new_state[name] = old_state[name]
#         print(f"✅ موجود: {name} -> {shape}")
#     else:
#         new_state[name] = np.zeros(shape, dtype=np.float32).tolist()
#         print(f"➕ اضافه شد: {name} -> {shape}")

# with open("state_full.json", "w") as f:
#     json.dump(new_state, f)

# print(f"\n✅ state_full.json ساخته شد با {len(new_state)} کلید")
