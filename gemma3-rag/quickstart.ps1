# Gemma3 RAG KasenSabo MVP - Quick Start Script
# このスクリプトは順番に実行する各ステップのコマンドをまとめたものです

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Gemma3 RAG KasenSabo MVP - Quick Start" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ステップ1: 環境確認
Write-Host "[Step 1] Checking Environment..." -ForegroundColor Yellow
Write-Host ""

# Python確認
Write-Host "Checking Python version..." -ForegroundColor Green
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.9 or later." -ForegroundColor Red
    exit 1
}

# Ollama確認
Write-Host "`nChecking Ollama..." -ForegroundColor Green
ollama --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Ollama not found. Please install Ollama from https://ollama.ai/" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ Environment check passed!" -ForegroundColor Green
Write-Host ""

# ステップ2: Ollamaモデルのダウンロード
Write-Host "[Step 2] Downloading Ollama Models..." -ForegroundColor Yellow
Write-Host ""

Write-Host "Checking if gemma:2b-instruct-q4_K_M is available..." -ForegroundColor Green
$model1 = ollama list | Select-String "gemma:2b-instruct-q4_K_M"
if (-not $model1) {
    Write-Host "Downloading gemma:2b-instruct-q4_K_M (this may take a few minutes)..." -ForegroundColor Cyan
    ollama pull gemma:2b-instruct-q4_K_M
} else {
    Write-Host "✓ Model already exists" -ForegroundColor Green
}

Write-Host "`nChecking if gemma:2b-instruct-q8_0 is available..." -ForegroundColor Green
$model2 = ollama list | Select-String "gemma:2b-instruct-q8_0"
if (-not $model2) {
    Write-Host "Downloading gemma:2b-instruct-q8_0 (this may take a few minutes)..." -ForegroundColor Cyan
    ollama pull gemma:2b-instruct-q8_0
} else {
    Write-Host "✓ Model already exists" -ForegroundColor Green
}

Write-Host "`n✓ Models ready!" -ForegroundColor Green
Write-Host ""

# ステップ3: Python仮想環境のセットアップ
Write-Host "[Step 3] Setting up Python Environment..." -ForegroundColor Yellow
Write-Host ""

# 既存の仮想環境を使用
$venvPath = "C:\Users\yasun\.virtualenvs\multimodal-raptor-colvbert-blip-3yGxnN3T"

if (Test-Path $venvPath) {
    Write-Host "Using existing virtual environment: multimodal-raptor-colvbert-blip" -ForegroundColor Green
    Write-Host "`nActivating virtual environment..." -ForegroundColor Cyan
    & "$venvPath\Scripts\Activate.ps1"
} else {
    Write-Host "ERROR: Virtual environment not found at $venvPath" -ForegroundColor Red
    Write-Host "Please check the path or create a new venv." -ForegroundColor Red
    exit 1
}

Write-Host "`nUpgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

Write-Host "`nInstalling dependencies (this may take a few minutes)..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet

Write-Host "`nDownloading NLTK data..." -ForegroundColor Cyan
python -c "import nltk; nltk.download('punkt', quiet=True); print('✓ NLTK data downloaded')"

Write-Host "`n✓ Python environment ready!" -ForegroundColor Green
Write-Host ""

# ステップ4: インデックス構築
Write-Host "[Step 4] Building Index..." -ForegroundColor Yellow
Write-Host ""

if (Test-Path "index/chroma_index") {
    Write-Host "Index already exists. Do you want to rebuild it? (y/N): " -ForegroundColor Cyan -NoNewline
    $rebuild = Read-Host
    if ($rebuild -eq "y" -or $rebuild -eq "Y") {
        Write-Host "Rebuilding index..." -ForegroundColor Cyan
        python scripts/build_index.py
    } else {
        Write-Host "✓ Using existing index" -ForegroundColor Green
    }
} else {
    Write-Host "Building index for the first time (this may take 5-10 minutes)..." -ForegroundColor Cyan
    python scripts/build_index.py
}

Write-Host ""

# ステップ5: 動作確認
Write-Host "[Step 5] Testing RAG System..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Do you want to run a quick test? (Y/n): " -ForegroundColor Cyan -NoNewline
$test = Read-Host

if ($test -ne "n" -and $test -ne "N") {
    Write-Host "`nRunning test queries..." -ForegroundColor Cyan
    python scripts/run_rag.py
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Run full benchmark:  python scripts/run_benchmark.py" -ForegroundColor White
Write-Host "  2. Test single query:   python scripts/run_rag.py" -ForegroundColor White
Write-Host "  3. Test evaluation:     python scripts/evaluate.py" -ForegroundColor White
Write-Host ""
Write-Host "For detailed instructions, see:" -ForegroundColor Yellow
Write-Host "  - README.md" -ForegroundColor White
Write-Host "  - SETUP_GUIDE.md" -ForegroundColor White
Write-Host ""
