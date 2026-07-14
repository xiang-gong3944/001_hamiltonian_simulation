param(
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ が見つかりません。python.org 等から Python をインストールしてください。"
}

try {
    $versionText = & $PythonCommand --version 2>&1
    & $PythonCommand -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
} catch {
    throw "python コマンドを実行できません。Python 3.10+ をインストールし、PATH を確認してください。"
}
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10 以上が必要です。検出された Python: $versionText"
}
Write-Host "Using $versionText"

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $PythonCommand -m venv (Join-Path $projectRoot ".venv")
}

# --upgrade-strategy eager also fixes incompatible packages left in an existing venv.
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install --upgrade --upgrade-strategy eager -r requirements.txt
& $venvPython -m pip check

Write-Host ""
Write-Host "準備完了。VS Code で notebooks\resource_comparison.ipynb を開いてください。"
Write-Host "カーネルが自動選択されない場合は、次を指定してください:"
Write-Host "  $venvPython"
Write-Host "このスクリプトはブラウザ版 Jupyter を起動しません。"
