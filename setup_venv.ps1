$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ が見つかりません。python.org 等から Python をインストールしてください。"
}

try {
    $versionText = & python --version 2>&1
} catch {
    throw "python コマンドを実行できません。Python 3.10+ をインストールし、PATH を確認してください。"
}
if ($LASTEXITCODE -ne 0) {
    throw "python コマンドを実行できません。Python 3.10+ をインストールし、PATH を確認してください。"
}
Write-Host "Using $versionText"

python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m ipykernel install --user --name hamiltonian-resources --display-name "Python (hamiltonian-resources)"

Write-Host "準備完了: .\.venv\Scripts\Activate.ps1"
Write-Host "Notebook: jupyter lab notebooks\resource_comparison.ipynb"
