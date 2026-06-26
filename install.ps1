Write-Host "Analisi dell'hardware in corso..." -ForegroundColor Cyan

# Controlla se il sistema riconosce i driver della scheda video (NVIDIA)
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host ">>> GPU NVIDIA RILEVATA! Procedo con l'installazione di PyTorch + CUDA 12.6" -ForegroundColor Green
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Host ">>> Nessuna GPU NVIDIA rilevata. Procedo con l'installazione di PyTorch per CPU" -ForegroundColor Yellow
    pip install torch torchvision torchaudio
}

Write-Host ">>> Installo le librerie di base (numpy, scipy, matplotlib)..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ">>> Installazione completata con successo! L'ambiente e' pronto." -ForegroundColor Green