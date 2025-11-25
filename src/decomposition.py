from transformers import AutoTokenizer, AutoModel
import torch


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_label(target_phrase, sentence):
    
    encoded = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    sent_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    target_tokens = tokenizer.tokenize(target_phrase)
    target_positions = []
    for i in range(len(sent_tokens) - len(target_tokens) + 1):
        if sent_tokens[i : i + len(target_tokens)] == target_tokens:
            target_positions.extend(range(i, i + len(target_tokens)))
    if len(target_positions) == 0:
        raise ValueError(f"Phrase '{target_phrase}' not found in tokenized sentence.")
    with torch.no_grad():
        outputs = model(**encoded)
    hidden = outputs.last_hidden_state
    word_embedding = hidden[0, target_positions, :].mean(dim=0)

    return word_embedding


def compute_mean_and_cov(embeddings: torch.Tensor):
    mu = embeddings.mean(dim=0)  # (D,)
    centered = embeddings - mu
    cov = (centered.T @ centered) / (embeddings.shape[0] - 1)
    return mu, cov

def fit_pca(embeddings: torch.Tensor,  mu,n_components=2):
    X = embeddings - mu
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    W = Vt[:n_components].T
    return W

def project_mean_and_cov(mu: torch.Tensor, cov: torch.Tensor, W: torch.Tensor):
    mu_2d = W.T @ mu
    cov_2d = W.T @ cov @ W
    return mu_2d, cov_2d

def covariance_to_ellipse_params(cov_2d: torch.Tensor, n_std=1.0):
    eigvals, eigvecs = torch.linalg.eigh(cov_2d)
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    width, height = 2 * n_std * torch.sqrt(eigvals)
    angle = torch.atan2(eigvecs[1, 0], eigvecs[0, 0]) * (180.0 / torch.pi)
    return width.item(), height.item(), angle.item()

def get_center_and_cov(label, sentences):
    embeddings = torch.stack([embed_label(label, s) for s in sentences]) 
    mu, cov = compute_mean_and_cov(embeddings)
    W = fit_pca(embeddings, n_components=2, mu = mu)
    mu_2d, cov_2d = project_mean_and_cov(mu, cov, W)
    return mu_2d, cov_2d