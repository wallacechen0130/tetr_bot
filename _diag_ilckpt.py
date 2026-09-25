import torch

p = torch.load("checkpoints/il/best.pt", map_location="cpu", weights_only=False)
print("IL checkpoint keys:", list(p.keys()))
extra = p.get("extra", {})
print("extra.config:", {k: v for k, v in extra.get("config", {}).items() if k in ("network", "hidden_dim", "epochs", "topk_soft_targets", "limit")})
print("extra.network:", extra.get("network"))
print("metrics:", p.get("metrics"))
sd = p["model"]
keys = list(sd.keys())
print("張量數:", len(keys))
print("前 5 個 key:", keys[:5])
# 判斷是哪個 encoder
print("是 resnet:", any(k.startswith("board_encoder.stem") for k in keys))
print("是 small_cnn:", any(k.startswith("board_encoder.conv.0") for k in keys))
