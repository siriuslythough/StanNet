import argparse
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
from einops import rearrange

# Your utilities and models
from utils import ToHSV, ToComplex
from stanNet import stanNet_complex

def build_model(arch: str, num_classes: int):
    a = arch.lower()
    if a == "stannet":
        return stanNet_complex(num_classes)   
    raise ValueError(f"Unknown arch in checkpoint: {arch}")

def build_transform(image_size: int):
    return T.Compose([
        T.ToTensor(),
        T.Resize(image_size),
        T.CenterCrop(image_size),
        ToHSV(),
        ToComplex(),
    ])

@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Infer yoga pose class on a single image (complex CNN)")
    p.add_argument("--model", type=str, required=True, help="Path to model.pth")
    p.add_argument("--image", type=str, required=True, help="Path to image to classify")
    p.add_argument("--topk", type=int, default=5, help="Top-K to display")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = p.parse_args()

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    model = torch.load(args.model, map_location=device)
    classes = model.get("classes")
    num_classes = model.get("num_classes")
    image_size = model.get("image_size", 224)
    arch = model.get("arch", "yoga_copomix")
    print(f"Checkpoint: arch={arch} classes={len(classes)} image_size={image_size}")

    # Build model and load weights
    model = build_model(arch, num_classes=num_classes).to(device)
    model.load_state_dict(model["model"])
    model.eval()

    # Build transform, load image
    tf = build_transform(image_size)
    img = Image.open(args.image).convert("RGB")
    x = tf(img)               # complex tensor, shape (3,H,W)
    x = x.unsqueeze(0).to(device)  # (1,3,H,W) complex

    # Forward
    z = model(x)              # (1, C, 1, 1) complex
    z = rearrange(z, "b c h w -> b (c h w)")
    mag = z.abs()             # (1, C)
    prob = torch.softmax(mag, dim=1).squeeze(0)

    # Top-K
    k = min(args.topk, prob.numel())
    vals, idxs = torch.topk(prob, k)
    print("Top-{}:".format(k))
    for r, (p, i) in enumerate(zip(vals.tolist(), idxs.tolist()), 1):
        name = classes[i] if classes and i < len(classes) else str(i)
        print(f"{r:>2d}. {name:<20s} {p*100:6.2f}%")

if __name__ == "__main__":
    main()