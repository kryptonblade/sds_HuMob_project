import torch
from geoformer.model import build_model
model = build_model("tiny").to("mps")
try:
    model = torch.compile(model)
    x = torch.randint(0, 100, (2, 32)).to("mps")
    tod = torch.randint(0, 48, (2, 32)).to("mps")
    dow = torch.randint(0, 7, (2, 32)).to("mps")
    y, _, _ = model(x, tod, dow)
    print("COMPILE SUCCESS! Output shape:", y.shape)
except Exception as e:
    print("COMPILE FAILED:", e)
