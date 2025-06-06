# import os
# import argparse
# import random
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split

# from torchvision import transforms

# from scripts.dataloader import LocomotionDataset
# from scripts.model import LocomotionClassifier


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", type=str, default="data/train")
#     parser.add_argument(
#         "--input_mode", type=str, choices=["depth", "rgb", "both"], default="both"
#     )
#     parser.add_argument(
#         "--use_lstm", action="store_true", help="Enable temporal input with LSTM"
#     )
#     parser.add_argument("--img_size", type=int, default=128)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pth")
#     parser.add_argument("--name", type=str, default="run")
#     parser.add_argument(
#         "--use_pretrained", action="store_true", help="Use pretrained CNN backbone"
#     )
#     return parser.parse_args()


# def get_transforms(input_mode, img_size):
#     common = transforms.Compose(
#         [
#             transforms.Resize((img_size, img_size)),
#             transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
#             transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
#             transforms.ToTensor(),
#         ]
#     )

#     rgb_tf = None
#     depth_tf = None

#     if input_mode in ["rgb", "both"]:
#         rgb_tf = transforms.Compose(
#             [
#                 transforms.ColorJitter(
#                     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
#                 ),
#                 common,
#             ]
#         )

#     if input_mode in ["depth", "both"]:
#         depth_tf = common  # could also normalize later

#     return depth_tf, rgb_tf


# def evaluate(model, dataloader, criterion, device):
#     model.eval()
#     correct = 0
#     total = 0
#     total_loss = 0.0

#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = {}
#             if "rgb" in batch:
#                 inputs["rgb"] = batch["rgb"].to(device)
#             if "depth" in batch:
#                 inputs["depth"] = batch["depth"].to(device)
#             labels = batch["label"].to(device)

#             outputs = model(rgb=inputs.get("rgb"), depth=inputs.get("depth"))
#             loss = criterion(outputs, labels)
#             total_loss += loss.item() * labels.size(0)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     acc = correct / total
#     avg_loss = total_loss / total
#     return acc, avg_loss


# def save_curves(
#     train_acc, val_acc, train_loss, val_loss, save_dir="outputs", name="run"
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     # Accuracy plot
#     plt.figure()
#     plt.plot(train_acc, label="Train Accuracy")
#     plt.plot(val_acc, label="Val Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy Curve")
#     plt.legend()
#     plt.savefig(os.path.join(save_dir, f"{name}_acc.png"))
#     plt.close()

#     # Loss plot
#     plt.figure()
#     plt.plot(train_loss, label="Train Loss")
#     plt.plot(val_loss, label="Val Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Loss Curve")
#     plt.legend()
#     plt.savefig(os.path.join(save_dir, f"{name}_loss.png"))
#     plt.close()


# def main():
#     args = parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Transforms
#     depth_tf, rgb_tf = get_transforms(args.input_mode, args.img_size)

#     # Dataset

#     seq_len = 3 if args.use_lstm else 1
#     full_dataset = LocomotionDataset(
#         data_dir=args.data_dir,
#         input_mode=args.input_mode,
#         depth_transform=depth_tf,
#         rgb_transform=rgb_tf,
#         seq_len=seq_len,
#     )

#     val_size = int(0.2 * len(full_dataset))
#     train_size = len(full_dataset) - val_size
#     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

#     # Model
#     model = LocomotionClassifier(
#         input_mode=args.input_mode,
#         use_lstm=args.use_lstm,
#         use_pretrained=args.use_pretrained,
#     )
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     best_val_acc = 0.0
#     os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

#     train_acc_list = []
#     val_acc_list = []
#     train_loss_list = []
#     val_loss_list = []
#     exp_name = args.name

#     for epoch in range(args.epochs):
#         model.train()
#         for batch in train_loader:
#             inputs = {}
#             if "rgb" in batch:
#                 inputs["rgb"] = batch["rgb"].to(device)
#             if "depth" in batch:
#                 inputs["depth"] = batch["depth"].to(device)
#             labels = batch["label"].to(device)

#             outputs = model(rgb=inputs.get("rgb"), depth=inputs.get("depth"))
#             loss = criterion(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         val_acc, val_loss = evaluate(model, val_loader, criterion, device)
#         print(
#             f"Epoch {epoch + 1}/{args.epochs} | Val Acc: {val_acc:.3f} | Val Loss: {val_loss:.4f}"
#         )

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), args.save_path)
#             print(f"Best model saved with accuracy: {val_acc:.3f}")

#         train_acc, train_loss = evaluate(model, train_loader, criterion, device)
#         print(f"Train Acc: {train_acc:.3f} | Train Loss: {train_loss:.4f}")
#         train_acc_list.append(train_acc)
#         train_loss_list.append(train_loss)

#         val_acc_list.append(val_acc)
#         val_loss_list.append(val_loss)

#     save_curves(
#         train_acc_list, val_acc_list, train_loss_list, val_loss_list, name=exp_name
#     )

#     print(f"Training complete. Best val acc: {best_val_acc:.3f}")


# if __name__ == "__main__":
#     main()

import os
import argparse
import random
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from scripts.dataloader import LocomotionDataset
from scripts.model import LocomotionClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument(
        "--input_mode", type=str, choices=["depth", "rgb", "both"], default="both"
    )
    parser.add_argument(
        "--use_lstm", action="store_true", help="Enable temporal input with LSTM"
    )
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--name", type=str, default="run")
    parser.add_argument(
        "--use_pretrained", action="store_true", help="Use pretrained CNN backbone"
    )
    return parser.parse_args()


def get_transforms(input_mode, img_size):
    common = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
            transforms.ToTensor(),
        ]
    )

    rgb_tf = None
    depth_tf = None

    if input_mode in ["rgb", "both"]:
        rgb_tf = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                common,
            ]
        )
    if input_mode in ["depth", "both"]:
        depth_tf = common

    return depth_tf, rgb_tf


def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {}
            if "rgb" in batch:
                inputs["rgb"] = batch["rgb"].to(device)
            if "depth" in batch:
                inputs["depth"] = batch["depth"].to(device)
            labels = batch["label"].to(device)

            outputs = model(rgb=inputs.get("rgb"), depth=inputs.get("depth"))
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / total
    return acc, avg_loss


def save_curves(
    train_acc, val_acc, train_loss, val_loss, save_dir="outputs", name="run"
):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{name}_acc.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{name}_loss.png"))
    plt.close()


def measure_inference_speed(model, dataloader, device, warmup=10, trials=50):
    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup + trials:
                break

            inputs = {}
            if "rgb" in batch:
                inputs["rgb"] = batch["rgb"].to(device)
            if "depth" in batch:
                inputs["depth"] = batch["depth"].to(device)

            if i < warmup:
                _ = model(rgb=inputs.get("rgb"), depth=inputs.get("depth"))
                continue

            start = time.perf_counter()
            _ = model(rgb=inputs.get("rgb"), depth=inputs.get("depth"))
            end = time.perf_counter()

            times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"\n=== Inference Speed ===")
    print(f"Avg time per batch: {avg_time:.6f} sec")
    print(f"Avg time per sample: {avg_time / dataloader.batch_size:.6f} sec")
    print(f"Throughput: {1 / (avg_time / dataloader.batch_size):.2f} samples/sec\n")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth_tf, rgb_tf = get_transforms(args.input_mode, args.img_size)
    seq_len = 3 if args.use_lstm else 1

    full_dataset = LocomotionDataset(
        data_dir=args.data_dir,
        input_mode=args.input_mode,
        depth_transform=depth_tf,
        rgb_transform=rgb_tf,
        seq_len=seq_len,
    )

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = LocomotionClassifier(
        input_mode=args.input_mode,
        use_lstm=args.use_lstm,
        use_pretrained=args.use_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            inputs = {}
            if "rgb" in batch:
                inputs["rgb"] = batch["rgb"].to(device)
            if "depth" in batch:
                inputs["depth"] = batch["depth"].to(device)
            labels = batch["label"].to(device)

            start_time = time.time()

            outputs = model(rgb=inputs.get("rgb"), depth=inputs.get("depth"))

            end_time = time.time()
            print(f"Epoch {epoch + 1}, Inference Time: {end_time - start_time:.4f} sec")
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Val Acc: {val_acc:.3f} | Val Loss: {val_loss:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Best model saved with accuracy: {val_acc:.3f}")

        train_acc, train_loss = evaluate(model, train_loader, criterion, device)
        print(f"Train Acc: {train_acc:.3f} | Train Loss: {train_loss:.4f}")
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

    save_curves(
        train_acc_list, val_acc_list, train_loss_list, val_loss_list, name=args.name
    )

    print(f"Training complete. Best val acc: {best_val_acc:.3f}")

    print("Measuring inference speed...")
    # measure_inference_speed(model, val_loader, device)


if __name__ == "__main__":
    main()
