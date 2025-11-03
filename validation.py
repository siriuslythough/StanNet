import argparse
from pathlib import Path
from typing import Optional, List

import torch
import torchvision.transforms as T
from PIL import Image
from einops import rearrange

# Your utilities and models
from utils import ToHSV, ToComplex
from stanNet import stanNet_complex
import networks


def build_model(arch: str, num_classes: int) -> torch.nn.Module:
    """Build model architecture based on architecture name."""
    a = arch.lower()
    
    # Simple architectures
    if a == "stannet":
        return stanNet_complex(num_classes)
    elif a == "cds_e":
        return networks.CDS_E(num_classes=num_classes)
    elif a == "alexnet":
        return networks.AlexNet(num_classes=num_classes)
    
    # VGG variants
    elif a.startswith("vgg"):
        vgg_name = a.replace("vgg", "Vgg")
        if vgg_name not in networks.cfg:
            raise ValueError(f"Unknown VGG variant: {a}. Available: vgg11, vgg13, vgg16, vgg19")
        return networks.VGG(vgg_name, num_classes=num_classes)
    
    # ResNet variants
    elif a == "resnet18":
        return networks.resnet18(num_classes=num_classes)
    elif a == "resnet34":
        return networks.resnet34(num_classes=num_classes)
    elif a == "resnet50":
        return networks.resnet50(num_classes=num_classes)
    elif a == "resnet101":
        return networks.resnet101(num_classes=num_classes)
    elif a == "resnet152":
        return networks.resnet152(num_classes=num_classes)
    
    # ResNeXt variants
    elif a == "resnext50_32x4d":
        return networks.resnext50_32x4d(num_classes=num_classes)
    elif a == "resnext101_32x8d":
        return networks.resnext101_32x8d(num_classes=num_classes)
    
    # Wide ResNet variants
    elif a == "wide_resnet50_2":
        return networks.wide_resnet50_2(num_classes=num_classes)
    elif a == "wide_resnet101_2":
        return networks.wide_resnet101_2(num_classes=num_classes)
    
    else:
        available_archs = [
            "stannet", "cds_e", "alexnet",
            "vgg11", "vgg13", "vgg16", "vgg19",
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "resnext50_32x4d", "resnext101_32x8d",
            "wide_resnet50_2", "wide_resnet101_2"
        ]
        raise ValueError(
            f"Unknown architecture: {a}\n"
            f"Available architectures: {', '.join(available_archs)}"
        )


def build_transform(image_size: int) -> T.Compose:
    """Build image transformation pipeline."""
    return T.Compose([
        T.ToTensor(),
        T.Resize(image_size),
        T.CenterCrop(image_size),
        ToHSV(),
        ToComplex(),
    ])


def load_model_from_path(model_path: str, device: torch.device):
    """Load a saved model checkpoint from file."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract metadata from checkpoint
    classes = checkpoint.get("classes")
    num_classes = checkpoint.get("num_classes")
    image_size = checkpoint.get("image_size", 224)
    arch = checkpoint.get("arch", "stannet")
    
    # Build model architecture
    model = build_model(arch, num_classes=num_classes).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    return model, classes, image_size


def predict_single_image(model, image_path: str, image_size: int, 
                         device: torch.device, classes: Optional[List[str]] = None, 
                         topk: int = 5):
    """Make prediction on a single image."""
    # Build transform and load image
    tf = build_transform(image_size)
    
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    x = tf(img)  # complex tensor, shape (3,H,W)
    x = x.unsqueeze(0).to(device)  # (1,3,H,W) complex
    
    # Forward pass
    with torch.no_grad():
        z = model(x)  # output shape depends on architecture
    
    # Handle different output shapes
    if z.dim() == 4:
        z = rearrange(z, "b c h w -> b (c h w)")
    
    mag = z.abs() if z.is_complex() else z  # (1, C)
    prob = torch.softmax(mag, dim=1).squeeze(0)
    
    # Get top-K predictions
    k = min(topk, prob.numel())
    vals, idxs = torch.topk(prob, k)
    
    return vals.cpu().tolist(), idxs.cpu().tolist(), classes


def find_saved_models(model_dir: str) -> List[str]:
    """Find all .pth model files in a directory."""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    model_files = sorted(model_path.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No .pth files found in {model_dir}")
    
    return [str(f) for f in model_files]


def print_available_architectures():
    """Print all available model architectures."""
    architectures = {
        "Simple Models": ["stannet", "cds_e", "alexnet"],
        "VGG Variants": ["vgg11", "vgg13", "vgg16", "vgg19"],
        "ResNet Variants": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        "ResNeXt Variants": ["resnext50_32x4d", "resnext101_32x8d"],
        "Wide ResNet Variants": ["wide_resnet50_2", "wide_resnet101_2"]
    }
    
    print("\n" + "="*60)
    print("Available Model Architectures")
    print("="*60)
    for category, models in architectures.items():
        print(f"\n{category}:")
        for model in models:
            print(f"  - {model}")
    print("\n" + "="*60 + "\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Infer yoga pose class on an image using saved models (complex CNN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validation.py --model_dir ./saved_models --image test.jpg
  python validation.py --model_dir ./saved_models --image test.jpg --model_name best.pth
  python validation.py --model_dir ./saved_models --image test.jpg --device cuda
  python validation.py --list-archs  # Show all available architectures
        """
    )
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default=None,
        help="Path to directory containing saved model.pth files"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=None,
        help="Path to image to classify"
    )
    parser.add_argument(
        "--topk", 
        type=int, 
        default=5, 
        help="Top-K predictions to display (default: 5)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for inference (default: auto)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Specific model filename to use (optional, defaults to latest)"
    )
    parser.add_argument(
        "--list-archs",
        action="store_true",
        help="List all available model architectures and exit"
    )
    
    args = parser.parse_args()
    
    # List architectures if requested
    if args.list_archs:
        print_available_architectures()
        return
    
    # Validate required arguments
    if not args.model_dir or not args.image:
        parser.print_help()
        print("\nError: --model_dir and --image are required arguments")
        return
    
    # Determine device
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
    
    # Find saved models
    model_files = find_saved_models(args.model_dir)
    print(f"Found {len(model_files)} model(s) in {args.model_dir}")
    
    # Select model to use
    if args.model_name:
        selected_model = None
        for model_file in model_files:
            if args.model_name in model_file:
                selected_model = model_file
                break
        if selected_model is None:
            raise FileNotFoundError(
                f"Model matching '{args.model_name}' not found in {args.model_dir}"
            )
    else:
        # Use the latest model (last in sorted list)
        selected_model = model_files[-1]
    
    print(f"Using model: {Path(selected_model).name}")
    
    # Load model
    model, classes, image_size = load_model_from_path(selected_model, device)
    print(f"Checkpoint: image_size={image_size} classes={len(classes) if classes else 'unknown'}")
    
    # Make prediction
    print(f"\nPredicting on: {args.image}")
    vals, idxs, classes = predict_single_image(
        model, 
        args.image, 
        image_size, 
        device, 
        classes, 
        topk=args.topk
    )
    
    # Display results
    print(f"\nTop-{len(vals)} predictions:")
    print("-" * 50)
    for rank, (prob, class_idx) in enumerate(zip(vals, idxs), 1):
        class_name = classes[class_idx] if classes and class_idx < len(classes) else f"Class {class_idx}"
        print(f"{rank:>2d}. {class_name:<30s} {prob*100:6.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()
