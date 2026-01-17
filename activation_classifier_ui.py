import os
import json
import torch
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datasets import load_from_disk, concatenate_datasets
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F


def configure_torch_for_speed():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


class FlattenMLPClassifier(nn.Module):
    def __init__(self, input_shape=(4096, 32), num_classes=10):
        super().__init__()
        L, C = input_shape
        self.net = nn.Sequential(
            nn.Linear(L * C, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


class RowPoolMLPClassifier(nn.Module):
    def __init__(self, input_shape=(4096, 32), num_classes=10):
        super().__init__()
        _, C = input_shape
        self.row_mlp = nn.Sequential(
            nn.Linear(C, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.row_mlp(x)
        pooled = z.mean(dim=1)
        return self.head(pooled)


class Conv1DClassifier(nn.Module):
    def __init__(self, input_shape=(4096, 32), num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.conv(x)
        z = self.pool(z).squeeze(-1)
        return self.head(z)


def build_model(model_type, num_classes):
    if model_type == "Flatten MLP":
        return FlattenMLPClassifier(num_classes=num_classes)
    if model_type == "Row-wise MLP + Pool":
        return RowPoolMLPClassifier(num_classes=num_classes)
    if model_type == "Conv1D":
        return Conv1DClassifier(num_classes=num_classes)
    raise ValueError(model_type)


def decode_activation(row):
    arr = np.frombuffer(
        row["activation_matrix_bytes"],
        dtype=np.dtype(row["activation_matrix_dtype"])
    )
    return arr.reshape(row["activation_matrix_shape"])


class ActivationDataset(Dataset):
    def __init__(self, hf_dataset, cache_decoded=False):
        self.ds = hf_dataset
        self.cache = {} if cache_decoded else None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        row = self.ds[idx]
        x = torch.from_numpy(decode_activation(row)).float()
        y = torch.tensor(row["user_index"], dtype=torch.long)
        out = (x, y)

        if self.cache is not None:
            self.cache[idx] = out
        return out


def parse_dirs(text):
    return [d.strip() for d in text.split(",") if d.strip()]


def load_datasets_from_dirs(dir_list):
    datasets = [load_from_disk(d) for d in dir_list]
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def split_dataset(ds, val_ratio):
    n_val = int(len(ds) * val_ratio)
    return random_split(ds, [len(ds) - n_val, n_val])


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


def accuracy(logits, y):
    return (logits.argmax(dim=-1) == y).float().mean().item()


def train_model(
    dataset_dirs_text,
    model_type,
    save_dir,
    epochs,
    batch_size,
    learning_rate,
    val_ratio,
    num_workers,
    use_amp,
    cache_decoded,
    progress=gr.Progress(),
):
    configure_torch_for_speed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = use_amp and device == "cuda"

    yield "🚀 Loading datasets...", None

    hf_ds = load_datasets_from_dirs(parse_dirs(dataset_dirs_text))
    num_classes = len(set(hf_ds["user_index"]))

    full_ds = ActivationDataset(hf_ds, cache_decoded)
    train_ds, val_ds = split_dataset(full_ds, val_ratio)

    train_loader = make_loader(train_ds, batch_size, True, num_workers)
    val_loader = make_loader(val_ds, batch_size, False, num_workers)

    model = build_model(model_type, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = dict(epoch=[], train_loss=[], val_loss=[], train_acc=[], val_acc=[])

    total_steps = max(1, epochs * len(train_loader))
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tl, ta = 0.0, 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tl += loss.item()
            ta += accuracy(logits, y)

            step += 1
            progress(step / total_steps)

        tl /= len(train_loader)
        ta /= len(train_loader)

        model.eval()
        vl, va = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                vl += loss.item()
                va += accuracy(logits, y)

        vl /= len(val_loader)
        va /= len(val_loader)

        history["epoch"].append(epoch)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        df = pd.DataFrame(history)
        df_long = df.melt(id_vars=["epoch"], var_name="metric", value_name="value")

        msg = (
            f"Epoch {epoch}/{epochs}\n"
            f"Train loss {tl:.4f}, acc {ta:.4f}\n"
            f"Val   loss {vl:.4f}, acc {va:.4f}"
        )
        yield msg, df_long

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump({"model_type": model_type, "num_classes": num_classes}, f, indent=2)

    df = pd.DataFrame(history)
    df_long = df.melt(id_vars=["epoch"], var_name="metric", value_name="value")
    yield "✅ Training complete", df_long


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)
    model = build_model(meta["model_type"], meta["num_classes"])
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.to(device).eval()
    return model


def test_model_ui(model_dir, dataset_dirs_text, num_workers, progress=gr.Progress()):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_model(model_dir, device)
    except Exception as e:
        return f"Error loading model: {e}", None, None

    hf_ds = load_datasets_from_dirs(parse_dirs(dataset_dirs_text))
    ds = ActivationDataset(hf_ds)
    loader = make_loader(ds, 256, False, num_workers)

    all_preds = []
    all_labels = []

    for i, (x, y) in enumerate(loader, 1):
        x = x.to(device)
        y_cpu = y.numpy()

        with torch.no_grad():
            preds = model(x).argmax(dim=-1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(y_cpu)
        progress(i / len(loader))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = (all_preds == all_labels).mean()
    status_msg = f"📊 Accuracy: {acc:.4f}"

    unique, counts = np.unique(all_preds, return_counts=True)
    hist_df = pd.DataFrame({"label": unique, "count": counts})
    hist_df["label"] = hist_df["label"].astype(str)

    cm = confusion_matrix(all_labels, all_preds)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    return status_msg, hist_df, fig


def predict_single(model_dir, dataset_dir, index):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_dir, device)
    row = load_from_disk(dataset_dir)[int(index)]
    x = torch.from_numpy(decode_activation(row)).float().unsqueeze(0).to(device)
    pred = model(x).argmax(dim=-1).item()
    return pred, {k: row[k] for k in row if k != "activation_matrix_bytes"}


with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Activation Matrix Classifier")

    with gr.Tab("Train"):
        train_dirs = gr.Textbox(label="Dataset directories")
        model_type = gr.Dropdown(
            ["Flatten MLP", "Row-wise MLP + Pool", "Conv1D"],
            value="Row-wise MLP + Pool"
        )
        save_dir = gr.Textbox(label="Save directory")

        epochs = gr.Slider(1, 1000, 10, 1, label="Epochs")
        batch_size = gr.Slider(16, 2048, 256, 16, label="Batch size")
        learning_rate = gr.Number(1e-3, label="Learning rate")
        val_ratio = gr.Slider(0.05, 0.5, 0.2, 0.05, label="Validation split")

        num_workers = gr.Slider(0, 32, 8, 1, label="DataLoader workers")
        use_amp = gr.Checkbox(True, label="Use AMP")
        cache_decoded = gr.Checkbox(False, label="Cache decoded samples")

        train_btn = gr.Button("Train")

        train_status = gr.Textbox(label="Status", lines=4)

        train_plot = gr.LinePlot(
            x="epoch",
            y="value",
            color="metric",
            title="Training Metrics",
            tooltip=["epoch", "metric", "value"],
            height=350,
            width=None
        )

        train_btn.click(
            train_model,
            inputs=[
                train_dirs,
                model_type,
                save_dir,
                epochs,
                batch_size,
                learning_rate,
                val_ratio,
                num_workers,
                use_amp,
                cache_decoded,
            ],
            outputs=[train_status, train_plot],
            show_progress="hidden"
        )

    with gr.Tab("Test"):
        test_dirs = gr.Textbox(label="Dataset directories")
        model_dir = gr.Textbox(label="Model directory")
        test_workers = gr.Slider(0, 32, 8, 1, label="DataLoader workers")
        test_btn = gr.Button("Test")

        test_status = gr.Textbox(label="Results")

        with gr.Row():
            test_hist = gr.BarPlot(
                x="label",
                y="count",
                title="Predicted Label Distribution",
                tooltip=["label", "count"],
                height=400,
                width=None
            )
            test_cm = gr.Plot(label="Confusion Matrix")

        test_btn.click(
            test_model_ui,
            inputs=[model_dir, test_dirs, test_workers],
            outputs=[test_status, test_hist, test_cm]
        )

    with gr.Tab("Single Sample"):
        model_dir_ss = gr.Textbox(label="Model directory")
        dataset_dir_ss = gr.Textbox(label="Dataset directory")
        index = gr.Number(label="Sample index", precision=0)
        pred_btn = gr.Button("Predict")
        pred_out = gr.Number(label="Predicted class")
        info_out = gr.JSON()
        pred_btn.click(predict_single, [model_dir_ss, dataset_dir_ss, index], [pred_out, info_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
