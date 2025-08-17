# ================================================================
# ArtForgerNet — FULL PIPELINE with Metrics/Tables/Figures
# ================================================================
# - SpatialBranch (ResNet50 with offline-cache fallback)
# - FrequencyBranch (FFT mag CNN)
# - AttentionFusion
# - Few-shot cosine head (episodic regularization)
# - (Optional) BERT text prior stub
# - Strong augs, label smoothing, WD, early stop, grad clip, SWA optional
# - Exports EVERYTHING needed for the paper/report:
#   * Split counts (per split & per class), SEED=42 confirmation
#   * Val/Test metrics: Accuracy/Precision/Recall/F1/ROC-AUC + confusion matrix CSV+PNG
#   * Test predictions CSV: y_true,y_prob,path
#   * Ablations (4 configs): Spatial / Freq / SF(no attn) / SF+Attn(best)
#   * Robustness: JPEG {90,70,50}, Gaussian {0.05,0.10}, Blur
#   * Grad-CAM overlays grid (no external libs)
#   * Training setup JSON
#   * Failure cases thumbnails + notes
# ---------------------------------------------------------------

import os, io, csv, math, time, json, random, warnings
from urllib.error import URLError
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Optional deps (best-effort)
try:
    import cv2
except Exception:
    cv2 = None
try:
    from transformers import AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

warnings.filterwarnings("ignore")

# ----------------------- Repro & Device ------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[OK] SEED confirmed = {SEED}")

# -------------------- Spatial (ResNet50) -----------------------
class SpatialBranch(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            try:
                weights = ResNet50_Weights.DEFAULT
                backbone = resnet50(weights=weights)
            except URLError:
                backbone = resnet50(weights=None)
                cache_root = os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch'))
                ckpt_dir = os.path.join(cache_root, 'hub', 'checkpoints')
                fname = ResNet50_Weights.DEFAULT.url.rpartition('/')[-1]
                path = os.path.join(ckpt_dir, fname)
                if os.path.exists(path):
                    state = torch.load(path, map_location='cpu')
                    backbone.load_state_dict(state)
        else:
            backbone = resnet50(weights=None)
        backbone.fc = nn.Identity()  # -> (B,2048)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)  # (B,2048)

# -------------------- Frequency (FFT) --------------------------
class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x_gray = x.mean(dim=1, keepdim=True)             # (B,1,H,W)
        fft    = torch.fft.fft2(x_gray)
        mag    = torch.fft.fftshift(torch.abs(fft))      # (B,1,H,W)
        out    = self.cnn(mag)                           # (B,32,1,1)
        return out.view(x.size(0), -1)                   # (B,32)

# -------------------- Attention Fusion ------------------------
class AttentionFusion(nn.Module):
    def __init__(self, spatial_dim=2048, freq_dim=32, hidden_dim=512):
        super().__init__()
        self.query_fc = nn.Linear(spatial_dim, hidden_dim)
        self.key_fc   = nn.Linear(freq_dim,    hidden_dim)
        self.value_fc = nn.Linear(freq_dim,    hidden_dim)
        self.out_fc   = nn.Linear(spatial_dim + hidden_dim, 512)

    def forward(self, spatial_feat, freq_feat):
        Q = self.query_fc(spatial_feat)  # (B,H)
        K = self.key_fc(freq_feat)       # (B,H)
        V = self.value_fc(freq_feat)     # (B,H)
        attn_score = (Q * K).sum(dim=1, keepdim=True)    # (B,1)
        weights = torch.sigmoid(attn_score)              # gated [0,1]
        attended = weights * V                           # (B,H)
        fused = torch.cat([spatial_feat, attended], dim=1)
        return self.out_fc(fused)                        # (B,512)

# -------------------- Few-Shot Cosine Head --------------------
class CosineClassifier(nn.Module):
    def __init__(self, in_dim=512, num_classes=2, init_s=10.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, 512, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(init_s)))
        self.register_buffer("prototypes", torch.randn(num_classes, 512))
        nn.init.normal_(self.prototypes, std=0.02)

    def set_prototypes(self, proto_tensor):
        with torch.no_grad():
            self.prototypes.copy_(proto_tensor)

    def forward(self, feats):
        x = F.normalize(self.proj(feats), dim=-1)        # (B,512)
        w = F.normalize(self.prototypes, dim=-1)         # (C,512)
        logits = self.scale * x @ w.t()                  # (B,C)
        return logits

def episodic_prototypes(feats, labels, num_classes=2):
    C = num_classes
    protos = []
    for c in range(C):
        mask = (labels == c)
        if mask.any():
            proto = F.normalize(feats[mask].mean(dim=0, keepdim=True), dim=-1)
        else:
            proto = F.normalize(torch.randn_like(feats[:1]), dim=-1)
        protos.append(proto)
    return torch.cat(protos, dim=0)  # (C,512)

# -------------------- BERT Text Regularizer -------------------
class TextPriorBERT(nn.Module):
    def __init__(self, prompts=None, model_name="bert-base-uncased", device=DEVICE):
        super().__init__()
        self.enabled = _HAS_TRANSFORMERS
        if prompts is None:
            prompts = [
                "a human-made real artwork or painting",
                "an AI-generated image or artwork"
            ]
        self.prompts = prompts
        self.device = device
        if self.enabled:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.bert = AutoModel.from_pretrained(model_name, local_files_only=True).eval().to(device)
                for p in self.bert.parameters():
                    p.requires_grad = False
                with torch.no_grad():
                    toks = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
                    out = self.bert(**toks).last_hidden_state[:,0,:]  # CLS
                    self.text_embeds = F.normalize(out, dim=-1).detach()  # (2,768)
                self.vis_proj = nn.Linear(512, 768, bias=False)
                self.scale = nn.Parameter(torch.tensor(10.0))
            except Exception:
                self.enabled = False

    def forward(self, visual_512):
        if not self.enabled:
            return None, None
        v = F.normalize(self.vis_proj(visual_512), dim=-1)   # (B,768)
        t = F.normalize(self.text_embeds, dim=-1)            # (2,768)
        logits = self.scale * (v @ t.t())                    # (B,2)
        return logits, t

# -------------------- Main Model ------------------------------
class ArtForgerNet(nn.Module):
    def __init__(self, pretrained_spatial=True, use_freq=True, use_attention=True):
        super().__init__()
        self.use_freq = use_freq
        self.use_attention = use_attention

        self.spatial    = SpatialBranch(pretrained=pretrained_spatial)
        spatial_dim = 2048

        if use_freq:
            self.frequency = FrequencyBranch()
            freq_dim = 32
        else:
            freq_dim = 0

        if use_freq and use_attention:
            self.fusion = AttentionFusion(spatial_dim, freq_dim, 512)
            fused_dim = 512
        elif use_freq:
            fused_dim = spatial_dim + freq_dim
        else:
            fused_dim = spatial_dim

        self.fused_dim = fused_dim

        # Main classifier (binary)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Few-shot cosine head over fused features
        self.fewshot = CosineClassifier(in_dim=fused_dim, num_classes=2, init_s=10.0)

        # BERT text prior head
        self.text_prior = TextPriorBERT()

    def forward_features(self, x):
        s = self.spatial(x)                   # (B,2048)
        if self.use_freq:
            f = self.frequency(x)             # (B,32)
            if self.use_attention:
                v = self.fusion(s, f)         # (B,512)
            else:
                v = torch.cat([s, f], dim=1)  # (B,2048+32)
        else:
            v = s                              # (B,2048)
        return v  # fused feature

    def forward(self, x):
        v = self.forward_features(x)
        prob = self.classifier(v)              # (B,1), sigmoid probs
        logits_few = self.fewshot(v)           # (B,2) cosine logits
        logits_txt, _ = (self.text_prior(v) if self.text_prior.enabled else (None, None))
        return prob, logits_few, logits_txt, v

# -------------------- Dataset / Metrics -----------------------
class FauxFinderDataset(Dataset):
    def __init__(self, base_dir, transform=None, return_path=False):
        """
        base_dir:
          Data/
            REAL/
            FAKE/
        """
        self.samples = []
        self.transform = transform
        self.return_path = return_path
        for label, cls in enumerate(['REAL', 'FAKE']):
            cls_dir = os.path.join(base_dir, cls)
            if not os.path.isdir(cls_dir): 
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    self.samples.append((os.path.join(cls_dir, fname), label))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {base_dir}/REAL and {base_dir}/FAKE")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # labels: 0=REAL, 1=FAKE
        if self.return_path:
            return img, torch.tensor([label], dtype=torch.float32), path
        return img, torch.tensor([label], dtype=torch.float32)

def binary_accuracy(preds, labels):
    return (torch.round(preds) == labels).float().mean()

# -------------------- Grad-CAM (built-in) ---------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model.eval()
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_image, class_idx=1):
        # class_idx=1 -> FAKE class
        prob, _, _, _ = self.model(input_image)
        score = prob[:, 0] if class_idx == 1 else (1.0 - prob[:, 0])
        self.model.zero_grad(set_to_none=True)
        score.mean().backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2,3), keepdim=True)  # GAP on grads
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        if cv2 is not None:
            cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam

# -------------------- Logging & Utils -------------------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def save_confusion_png(cm, out_path="confusion_matrix.png", classes=("Real","Fake")):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4,4), dpi=150)
    import itertools
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel().tolist()
    return dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, ROC_AUC=auc,
                TN=tn, FP=fp, FN=fn, TP=tp), cm

def dump_training_setup(params, out_json="training_setup.json"):
    setup = dict(
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        device="cuda" if torch.cuda.is_available() else "cpu",
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        SEED=SEED,
        **params
    )
    with open(out_json,"w") as f:
        json.dump(setup, f, indent=2)
    return setup

# -------------------- Run-flag helpers (prevents mismatches) ---
def _infer_flags_from_dirname(run_dir: str):
    """
    Infer flags from the directory name. Matches names produced in run_all():
    spatial_only, freq_only, sf_noattn, sf_attn.
    Also tolerates older variants like 'spatialonly'/'freqonly'.
    """
    name = Path(run_dir).name.lower()
    if ("spatial_only" in name) or ("spatialonly" in name):
        return dict(use_freq=False, use_attention=False)
    if ("freq_only" in name) or ("freqonly" in name):
        return dict(use_freq=True,  use_attention=False)
    if "sf_noattn" in name:
        return dict(use_freq=True,  use_attention=False)
    if "sf_attn" in name:
        return dict(use_freq=True,  use_attention=True)
    # Fallback heuristic for unforeseen names
    return dict(
        use_freq=("freq" in name) or ("sf" in name) or ("attn" in name),
        use_attention=("attn" in name)
    )

def _load_flags(run_dir: str, default_pretrained_spatial: bool = True):
    """
    Prefer reading run_meta.json saved at training time; if missing, fall back to dir-name inference.
    """
    meta_path = os.path.join(run_dir, "run_meta.json")
    if os.path.exists(meta_path):
        m = json.load(open(meta_path, "r"))
        return dict(
            pretrained_spatial=bool(m.get("pretrained_spatial", default_pretrained_spatial)),
            use_freq=bool(m.get("use_freq", False)),
            use_attention=bool(m.get("use_attention", False)),
        )
    base = _infer_flags_from_dirname(run_dir)
    base["pretrained_spatial"] = default_pretrained_spatial
    return base

# -------------------- Loss Compositor -------------------------
class BCEWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.eps = 1e-7
    def forward(self, pred_sigmoid, target01):
        target = target01 * (1 - self.smoothing) + 0.5 * self.smoothing
        pred_sigmoid = torch.clamp(pred_sigmoid, self.eps, 1 - self.eps)
        loss = - (target*torch.log(pred_sigmoid) + (1-target)*torch.log(1-pred_sigmoid))
        return loss.mean()

class CompositeLoss:
    def __init__(self, lambda_bce=1.0, lambda_fs=0.5, lambda_txt=0.5, smoothing=0.05):
        self.bce = BCEWithLabelSmoothing(smoothing=smoothing)
        self.ce  = nn.CrossEntropyLoss()
        self.lambda_bce = lambda_bce
        self.lambda_fs  = lambda_fs
        self.lambda_txt = lambda_txt

    def __call__(self, prob_sigmoid, logits_few, logits_txt, labels01, labels_int):
        loss = 0.0
        loss_main = self.bce(prob_sigmoid, labels01)
        loss += self.lambda_bce * loss_main
        loss_fs = torch.tensor(0.0, device=prob_sigmoid.device)
        if logits_few is not None:
            loss_fs = self.ce(logits_few, labels_int)
            loss += self.lambda_fs * loss_fs
        loss_txt = torch.tensor(0.0, device=prob_sigmoid.device)
        if (logits_txt is not None):
            loss_txt = self.ce(logits_txt, labels_int)
            loss += self.lambda_txt * loss_txt
        return loss, {"bce": float(loss_main.detach().cpu()),
                      "fewshot": float(loss_fs.detach().cpu()),
                      "text": float(loss_txt.detach().cpu())}

# -------------------- Dataloaders & Split Counts ---------------
def make_loaders_with_counts(data_root, img_size=256, batch_size=16):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    base_ds = FauxFinderDataset(data_root, transform=None, return_path=True)
    total = len(base_ds)
    tr = int(0.70 * total); va = int(0.15 * total); te = total - tr - va
    g = torch.Generator().manual_seed(SEED)
    idxs = list(range(total))
    train_idx, val_idx, test_idx = random_split(idxs, [tr,va,te], generator=g)

    # Per-split per-class counts
    def count_by_class(indices):
        c = defaultdict(int)
        for i in indices:
            _, lbl = base_ds.samples[i]
            c[lbl]+=1
        return c
    ct_train = count_by_class(train_idx.indices)
    ct_val   = count_by_class(val_idx.indices)
    ct_test  = count_by_class(test_idx.indices)
    split_counts = {
        "train": {"REAL": ct_train.get(0,0), "FAKE": ct_train.get(1,0), "TOTAL": len(train_idx)},
        "val":   {"REAL": ct_val.get(0,0),   "FAKE": ct_val.get(1,0),   "TOTAL": len(val_idx)},
        "test":  {"REAL": ct_test.get(0,0),  "FAKE": ct_test.get(1,0),  "TOTAL": len(test_idx)},
    }
    # Wrap with transform views
    class _View(Dataset):
        def __init__(self, indices, tf):
            self.indices=list(indices); self.tf=tf
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            img, y, p = base_ds[self.indices[i]]
            img = self.tf(img)
            return img, y, p
    train_ds = _View(train_idx.indices, train_tf)
    val_ds   = _View(val_idx.indices,   test_tf)
    test_ds  = _View(test_idx.indices,  test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader, split_counts

# -------------------- Train/Eval (single config) --------------
def train_eval_once(
    data_root,
    out_dir,
    use_freq=True,
    use_attention=True,
    pretrained_spatial=True,
    epochs=10,
    img_size=256,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    lambda_fs=0.5,
    lambda_txt=0.5,
    grad_clip=1.0,
    use_swa=False,
    patience=5
):
    ensure_dir(out_dir)

    train_loader, val_loader, test_loader, split_counts = make_loaders_with_counts(
        data_root, img_size, batch_size
    )

    # dump split counts
    with open(os.path.join(out_dir, "split_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split","REAL","FAKE","TOTAL","SEED"])
        for s, row in split_counts.items():
            w.writerow([s, row["REAL"], row["FAKE"], row["TOTAL"], SEED])

    model = ArtForgerNet(pretrained_spatial=pretrained_spatial, use_freq=use_freq, use_attention=use_attention).to(DEVICE)

    # Save meta so evaluation can recreate the exact architecture
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(dict(
            pretrained_spatial=pretrained_spatial,
            use_freq=use_freq,
            use_attention=use_attention
        ), f, indent=2)

    criterion = CompositeLoss(lambda_bce=1.0, lambda_fs=lambda_fs, lambda_txt=lambda_txt, smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=lr*0.5)

    best_val = -1.0
    wait = 0
    history_rows = []
    print(f"==> Training [{out_dir}] ...")
    for epoch in range(1, epochs+1):
        t0 = time.time()
        # ---------------- TRAIN ----------------
        model.train()
        tr_loss = tr_acc = 0.0
        for imgs, labels01, _ in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels01 = labels01.to(DEVICE, non_blocking=True)          # (B,1) float
            labels_int = labels01.long().squeeze(1)                    # (B,) int

            optimizer.zero_grad(set_to_none=True)
            prob, logits_few, logits_txt, feats = model(imgs)
            loss, parts = criterion(prob, logits_few, logits_txt, labels01, labels_int)

            with torch.no_grad():
                proj_feats = F.normalize(model.fewshot.proj(feats), dim=-1)
                proto = episodic_prototypes(proj_feats, labels_int)
                model.fewshot.set_prototypes(proto)

            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            tr_loss += float(loss.detach().cpu())
            tr_acc  += float(binary_accuracy(prob.detach(), labels01.detach()).cpu())

        tr_loss /= len(train_loader)
        tr_acc  /= len(train_loader)

        # ---------------- VALID ----------------
        model.eval()
        va_loss = va_acc = 0.0
        with torch.no_grad():
            for imgs, labels01, _ in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels01 = labels01.to(DEVICE, non_blocking=True)
                labels_int = labels01.long().squeeze(1)
                prob, logits_few, logits_txt, feats = model(imgs)
                loss, _ = criterion(prob, logits_few, logits_txt, labels01, labels_int)
                va_loss += float(loss.cpu())
                va_acc  += float(binary_accuracy(prob, labels01).cpu())
        va_loss /= len(val_loader)
        va_acc  /= len(val_loader)

        scheduler.step(va_acc)
        if use_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        dt = time.time() - t0
        print(f"[{epoch:02d}] TrainLoss {tr_loss:.4f} Acc {tr_acc:.4f} | ValLoss {va_loss:.4f} Acc {va_acc:.4f} | {dt:.1f}s")

        history_rows.append([epoch, tr_loss, tr_acc, va_loss, va_acc])
        # Early stopping
        if va_acc > best_val + 1e-4:
            best_val = va_acc
            wait = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # Save history CSV
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc"])
        w.writerows(history_rows)

    # SWA: update bn and save
    if use_swa:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
        torch.save(swa_model.state_dict(), os.path.join(out_dir, "swa.pth"))

    # ---------------- TEST + Predictions CSV ----------------
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pth"), map_location=DEVICE))
    model.eval()

    all_probs, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for imgs, labels01, paths in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels01 = labels01.to(DEVICE, non_blocking=True)
            prob, _, _, _ = model(imgs)
            all_probs.extend(prob[:,0].detach().cpu().numpy().tolist())
            all_labels.extend(labels01[:,0].detach().cpu().numpy().astype(int).tolist())
            all_paths.extend(list(paths))

    y_true = np.array(all_labels, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    metrics, cm = compute_metrics(y_true, y_prob)

    # Save predictions CSV
    with open(os.path.join(out_dir, "preds_test.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["y_true","y_prob","path"])
        for yt, yp, p in zip(all_labels, all_probs, all_paths):
            w.writerow([yt, yp, p])

    # Save metrics & confusion matrix
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_png(cm, out_path=os.path.join(out_dir, "confusion_matrix.png"))

    # Return essentials
    return dict(
        out_dir=out_dir,
        split_counts=split_counts,
        val_acc=best_val,
        test_metrics=metrics,
        test_confusion=cm,
        best_ckpt=os.path.join(out_dir, "best.pth")
    )

# -------------------- Robustness (on best model) --------------
def run_robustness(best_run_dir, data_root, img_size=256, batch_size=16):
    print(f"==> Robustness on {best_run_dir}")
    # ensure same split by using the seeded splitter
    _, _, test_loader, _ = make_loaders_with_counts(data_root, img_size, batch_size)

    # Build model from saved flags (or from dirname fallback)
    flags = _load_flags(best_run_dir)
    model = ArtForgerNet(**flags).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(best_run_dir, "best.pth"), map_location=DEVICE))
    model.eval()

    ROB_CFGS = [
        ("clean",      {}),
        ("jpeg_90",    {"jpeg_quality":90}),
        ("jpeg_70",    {"jpeg_quality":70}),
        ("jpeg_50",    {"jpeg_quality":50}),
        ("gauss_0.05", {"gauss_sigma":0.05}),
        ("gauss_0.10", {"gauss_sigma":0.10}),
        ("blur",       {"blur_radius":1.5}),
    ]

    def pil_perturb(img, spec):
        if "jpeg_quality" in spec:
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=int(spec["jpeg_quality"]))
            buf.seek(0); return Image.open(buf).convert("RGB")
        if "blur_radius" in spec:
            return img.filter(ImageFilter.GaussianBlur(radius=float(spec["blur_radius"])))
        return img

    class PerturbedDS(Dataset):
        def __init__(self, base_ds, spec):
            self.base_ds = base_ds; self.spec = spec
            self.resize = transforms.Resize((img_size,img_size))
            self.to_tensor = transforms.ToTensor()
            self.norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        def __len__(self): return len(self.base_ds)
        def __getitem__(self, i):
            img, y, p = self.base_ds[i]
            # base_ds returns already normalized! we need raw path image
            img_raw = Image.open(p).convert("RGB")
            img_raw = self.resize(pil_perturb(img_raw, self.spec))
            x = self.to_tensor(img_raw)
            if "gauss_sigma" in self.spec:
                sigma = float(self.spec["gauss_sigma"])
                x = torch.clamp(x + torch.randn_like(x)*sigma, 0.0, 1.0)
            x = self.norm(x)
            return x, y, p

    # Rebuild a raw-path test dataset for perturbations
    base_ds = FauxFinderDataset(data_root, transform=None, return_path=True)
    total = len(base_ds)
    tr = int(0.70 * total); va = int(0.15 * total); te = total - tr - va
    g = torch.Generator().manual_seed(SEED)
    idxs = list(range(total))
    _, _, test_idx = random_split(idxs, [tr,va,te], generator=g)

    class TestView(Dataset):
        def __init__(self, indices): self.indices=list(indices)
        def __len__(self): return len(self.indices)
        def __getitem__(self, i): return base_ds[self.indices[i]]
    raw_test = TestView(test_idx.indices)

    rows = []
    for name, spec in ROB_CFGS:
        ds = PerturbedDS(raw_test, spec)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y, _ in dl:
                x = x.to(DEVICE)
                prob, _, _, _ = model(x)
                all_probs.extend(prob[:,0].detach().cpu().numpy().tolist())
                all_labels.extend(y[:,0].detach().cpu().numpy().astype(int).tolist())
        y_true = np.array(all_labels, dtype=int)
        y_prob = np.array(all_probs, dtype=float)
        mets, _ = compute_metrics(y_true, y_prob)
        rows.append([name, mets["Accuracy"], mets["ROC_AUC"]])
        print(f"  {name:10s}  Acc={mets['Accuracy']:.4f}  AUC={mets['ROC_AUC']:.4f}")

    with open(os.path.join(best_run_dir, "robustness_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["setting","Accuracy","ROC_AUC"]); w.writerows(rows)

# -------------------- Grad-CAM grid & Failures ----------------
def make_gradcam_and_failures(run_dir, data_root, img_size=256, batch_size=16, k_fail=5):
    print(f"==> Grad-CAM & failure cases for {run_dir}")
    # Build model from saved flags (or dirname)
    flags = _load_flags(run_dir)
    model = ArtForgerNet(**flags).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=DEVICE))
    model.eval()

    # Build test loader (normalized) + raw loader (for visualization)
    _, _, test_loader, _ = make_loaders_with_counts(data_root, img_size, batch_size)

    # Pick a target conv layer from spatial backbone
    target_layer = None
    for m in model.spatial.backbone.modules():
        if isinstance(m, nn.Conv2d):
            target_layer = m
    if target_layer is None:
        target_layer = list(model.spatial.backbone.modules())[-1]

    cam = GradCAM(model, target_layer)

    # Collect a few samples, their probs, and create overlays
    samples = []
    with torch.no_grad():
        for x, y, p in test_loader:
            x = x.to(DEVICE)
            prob, _, _, _ = model(x)
            probs = prob[:,0].detach().cpu().numpy()
            for i in range(min(len(p), 8 - len(samples))):
                samples.append((p[i], int(y[i,0].item()), float(probs[i])))
            if len(samples) >= 8:
                break

    # make overlays grid
    overlays = []
    for path, lbl, pr in samples:
        img = Image.open(path).convert("RGB").resize((img_size, img_size))
        t = transforms.ToTensor()(img)
        t = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(t)
        cammap = cam.generate(t.unsqueeze(0).to(DEVICE), class_idx=1)
        # colorize simple heatmap
        heat = (cammap*255).astype(np.uint8)
        heat = Image.fromarray(heat).resize((img_size, img_size))
        heat = heat.convert("L")
        heat_rgb = Image.merge("RGB", (heat, Image.new("L", heat.size), Image.new("L", heat.size)))  # red-ish
        overlay = Image.blend(img, heat_rgb, alpha=0.4)
        draw = ImageDraw.Draw(overlay)
        draw.text((5,5), f"T:{lbl} p1:{pr:.2f}", fill=(255,255,255))
        overlays.append(overlay)

    if overlays:
        cols = 4
        rows = math.ceil(len(overlays)/cols)
        W,H = overlays[0].size
        grid = Image.new("RGB", (cols*W, rows*H), (255,255,255))
        for i, im in enumerate(overlays):
            r, c = divmod(i, cols); grid.paste(im, (c*W, r*H))
        grid_path = os.path.join(run_dir, "gradcam_grid.png")
        grid.save(grid_path)
        print(f"  saved {grid_path}")

    # Failure cases (needs preds_test.csv)
    pred_csv = os.path.join(run_dir, "preds_test.csv")
    if os.path.exists(pred_csv):
        import pandas as pd
        df = pd.read_csv(pred_csv)
        if "y_prob" in df.columns:
            df["y_pred"] = (df["y_prob"]>=0.5).astype(int)
            wrong = df[df["y_true"]!=df["y_pred"]].copy()
            if not wrong.empty:
                wrong["margin"] = (df["y_prob"]-0.5).abs()
                worst = wrong.sort_values("margin").head(k_fail)
                thumbs = []
                notes = []
                for _, r in worst.iterrows():
                    pth = r["path"]
                    try:
                        im = Image.open(pth).convert("RGB").resize((img_size,img_size))
                        thumbs.append(im)
                        notes.append(f"T:{int(r['y_true'])} P:{int(r['y_pred'])} p1={float(r['y_prob']):.2f}")
                    except Exception:
                        pass
                if thumbs:
                    cols = min(5, len(thumbs))
                    rows = math.ceil(len(thumbs)/cols)
                    W,H = thumbs[0].size
                    grid = Image.new("RGB", (cols*W, rows*(H+24)), (255,255,255))
                    draw = ImageDraw.Draw(grid)
                    for i, im in enumerate(thumbs):
                        r, c = divmod(i, cols)
                        grid.paste(im, (c*W, r*(H+24)))
                        draw.text((c*W+4, r*(H+24)+H+4), notes[i], fill=(0,0,0))
                    outp = os.path.join(run_dir, "failure_cases.png")
                    grid.save(outp)
                    with open(os.path.join(run_dir, "failure_cases_notes.txt"), "w") as f:
                        for n in notes: f.write(n+"\n")
                    print(f"  saved {outp}")

# -------------------- Orchestrator: 4 Ablations ----------------
def run_all(
    data_root,
    epochs=10,
    img_size=256,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    use_swa=False
):
    runs = [
        ("spatial_only",     dict(use_freq=False, use_attention=False)),
        ("freq_only",        dict(use_freq=True,  use_attention=False)),  # note: still spatial+freq (concat)
        ("sf_noattn",        dict(use_freq=True,  use_attention=False)),
        ("sf_attn",          dict(use_freq=True,  use_attention=True)),
    ]
    results_rows = []
    best_run = None
    best_auc = -1.0
    for name, flags in runs:
        out_dir = f"runs/{name}"
        res = train_eval_once(
            data_root=data_root,
            out_dir=out_dir,
            epochs=epochs,
            img_size=img_size,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            use_swa=use_swa,
            pretrained_spatial=True,
            **flags
        )
        va = res["val_acc"]
        mets = res["test_metrics"]
        results_rows.append([
            name, flags["use_freq"], flags["use_attention"],
            va, mets["Accuracy"], mets["ROC_AUC"], res["out_dir"]
        ])
        auc = mets["ROC_AUC"] if not np.isnan(mets["ROC_AUC"]) else -1.0
        if auc > best_auc:
            best_auc = auc
            best_run = res["out_dir"]

    # Save ablation summary
    ensure_dir("runs")
    with open("runs/ablation_summary.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Config","use_freq","use_attention","Val_Acc","Test_Acc","Test_ROC_AUC","dir"])
        w.writerows(results_rows)

    # Training setup dump
    setup = dump_training_setup(dict(
        batch_size=batch_size, epochs=epochs, lr=lr, weight_decay=weight_decay,
        grad_clip=grad_clip, SWA=use_swa
    ), out_json="training_setup.json")
    print("=== Training/Env setup ===")
    print(json.dumps(setup, indent=2))

    # Robustness + Grad-CAM + Failures on best run
    if best_run is not None and os.path.exists(os.path.join(best_run,"best.pth")):
        run_robustness(best_run, data_root, img_size, batch_size)
        make_gradcam_and_failures(best_run, data_root, img_size, batch_size)

    # Console recap
    print("\n================ FINAL RECAP ================\n")
    # Print split counts from the last run (also present per-run)
    last_split = os.path.join(results_rows[-1][-1], "split_counts.csv")
    if os.path.exists(last_split):
        with open(last_split, "r") as f:
            print(f"Split counts (from {last_split}):\n{f.read()}")
    # Print best test metrics
    if best_run and os.path.exists(os.path.join(best_run,"test_metrics.json")):
        best_metrics = json.load(open(os.path.join(best_run,"test_metrics.json"),"r"))
        print(f"Best Test (picked by ROC-AUC): dir={best_run}")
        for k in ["Accuracy","Precision","Recall","F1","ROC_AUC","TN","FP","FN","TP"]:
            print(f"  {k}: {best_metrics.get(k)}")
    # List artifacts
    print("\nArtifacts:")
    print(" - runs/ablation_summary.csv")
    for name,_ in runs:
        d = f"runs/{name}"
        print(f" - {d}/split_counts.csv")
        print(f" - {d}/history.csv")
        print(f" - {d}/best.pth")
        print(f" - {d}/preds_test.csv")
        print(f" - {d}/test_metrics.json")
        print(f" - {d}/confusion_matrix.png")
    if best_run:
        print(f" - {best_run}/robustness_summary.csv")
        print(f" - {best_run}/gradcam_grid.png")
        print(f" - {best_run}/failure_cases.png")
        print(f" - training_setup.json")
    print("\n=============================================\n")

# -------------------- Entry Point -----------------------------
if __name__ == "__main__":
    # Kaggle default path (adjust if needed):
    # /kaggle/input/real-and-fake-ai-generated-art-images-dataset/Data
    DATA_ROOT = os.environ.get(
        "DATA_ROOT",
        "/kaggle/input/real-and-fake-ai-generated-art-images-dataset/Data"
    )

    print("Device:", DEVICE)
    print("Transformers available:", _HAS_TRANSFORMERS)

    # ⚠️ Set small values first to smoke-test; then crank up
    run_all(
        data_root=DATA_ROOT,
        epochs=10,           # 🔧 adjust
        img_size=256,
        batch_size=16,
        lr=1e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_swa=False
    )
