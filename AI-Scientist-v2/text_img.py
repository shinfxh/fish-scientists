import os
import torch
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

class ClipToVaeMapper(nn.Module):
    def __init__(self, clip_dim=1024, vae_latent_dim=4096):
        super().__init__()
        self.b2_res = nn.Linear(clip_dim, vae_latent_dim)

    def forward(self, x):
    # Block1 residual MLP
        return self.b2_res(x)

# --- Data Loading & Preprocessing ---
def load_latents_and_caps(data_dir, batch_size=256):
    dataset = StreamingDataset(local=data_dir, shuffle=False, batch_size=batch_size)
    captions, latents = [], []
    for sample in dataset:
        captions.append(sample['caption'])
        flat = np.frombuffer(sample['latents_256'], dtype=np.float16)
        arr = flat.reshape(4, 32, 32)
        lat = torch.from_numpy(arr).to(dtype=torch.bfloat16)
        latents.append(lat)
        break
    return captions, torch.stack(latents)


def load_clip_embeddings(cache_dir, device):
    files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.pt')],
                   key=lambda x: int(os.path.splitext(x)[0]))
    embs = []
    for fname in tqdm(files, desc=f'Loading CLIP embeddings from {cache_dir}'):
        emb = torch.load(os.path.join(cache_dir, fname))
        embs.append(emb)
        break
    return torch.stack(embs).to(device=device, dtype=torch.bfloat16)

# --- Training Script ---
def main():
    # --- Paths and Device ---
    data_dir = 'datadir/textcaps/mds_latents_sdxl1_dfnclipH14/0'
    text_cache_dir = 'clip_text_embed'
    image_cache_dir = 'clip_image_embed/clip_cache'
    model_ckpt = 'mixed_clip_mapper.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)

    # --- Hyperparameters ---
    BATCH_SIZE = 1
    EPOCHS = 300
    WARMUP_STEPS = 5
    LEARNING_RATE = 1e-4
    PATIENCE = 50               # epochs to wait for improvement
    STAGNANT_THRESHOLD = 1e-6   # minimal decrease to count as improvement

    # --- Load Data ---
    # captions, latents = load_latents_and_caps(data_dir, batch_size=BATCH_SIZE)
    # N = len(captions)
    # latents = latents.view(N, -1).to(device=device)
    captions = [1]
    N = len(captions)
    latents = torch.randn(1, 4096).to(device=device)

    # --- Load CLIP Embeddings ---
    # text_embs = load_clip_embeddings(text_cache_dir, device)
    # image_embs = load_clip_embeddings(image_cache_dir, device)
    text_embs = torch.randn(1, 1024).to(device=device) * 0.1
    image_embs = torch.randn(1, 1024).to(device=device) * 0.1
    text_std = text_embs.std()
    image_std = image_embs.std()
    text_embs = text_embs / (text_std + 1e-6)
    image_embs = image_embs / (image_std + 1e-6)
    assert text_embs.shape == image_embs.shape, "Text and image embedding counts must match"

    print(f'[INFO] Loaded {N} samples, both text and image embeddings.')

    assert not torch.any(torch.isnan(latents)), "Latents contain NaN values before standardization"
    assert latents.std() > 1e-6, "Latents have zero variance, cannot standardize"
    print(f"Latents std: {latents.std(dim=0, keepdim=True)}")  # Print the std for latents
    print(latents[:10]) 

    # --- Model Setup ---
    model = ClipToVaeMapper().to(device=device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] Trainable parameters: {total_params:,}')

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = (N // BATCH_SIZE) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    mse_loss = nn.MSELoss()


    # --- Early Stopping Vars ---
    best_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for i in tqdm(range(0, N, BATCH_SIZE), desc=f'Epoch {epoch}'):
            # Batch indices
            idx_end = min(i + BATCH_SIZE, N)
            idx = slice(i, idx_end)
            batch_size_actual = idx_end - i

            # Mix embeddings: half text, half image
            mask = torch.rand(batch_size_actual, device=device) < 0
            emb_text = text_embs[idx].to(torch.float32)
            emb_image = image_embs[idx].to(torch.float32)
            batch_x = torch.where(mask.unsqueeze(1), emb_image, emb_text)
            print(batch_x)
            print(batch_x.isnan().sum())

            batch_y = latents[idx].to(torch.float32)
            print(batch_y.isnan().sum())
            # if epoch > 10:
            #     batch_y = batch_y + torch.randn_like(batch_y) * 0.01

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = mse_loss(pred, batch_y)
            print(pred.isnan().sum())
            print(pred, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * batch_x.size(0)
            global_step += 1

        avg_loss = running_loss / N
        print(f'Epoch {epoch}/{EPOCHS} - MSE Loss: {avg_loss:.4f}')

        # Early stopping
        if best_loss - avg_loss > STAGNANT_THRESHOLD:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'[INFO] No improvement for {epochs_no_improve} epochs')

        if epochs_no_improve >= PATIENCE:
            print(f'[INFO] Early stopping: no improvement in {PATIENCE} epochs')
            break

        # Evaluation logging on mixed embeddings
        model.eval()
        with torch.no_grad():
            eval_mask = torch.rand(BATCH_SIZE, device=device) < 0.5
            eval_text = text_embs[:BATCH_SIZE].to(torch.float32)
            eval_image = image_embs[:BATCH_SIZE].to(torch.float32)
            eval_x = torch.where(eval_mask.unsqueeze(1), eval_image, eval_text)
            eval_pred = model(eval_x)
            pred_mean = eval_pred.mean().item()
            pred_std = eval_pred.std().item()

    # --- Save Model ---
    torch.save(model.state_dict(), model_ckpt)
    print(f'[✅] Model saved to {model_ckpt}')

if __name__ == '__main__':
    main()
