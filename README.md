# Hamiltonian simulation resource comparison

Pauli 和で与えたハミルトニアンについて、次の回路レベルの比較を行う Python パッケージです。

- Lie–Trotter / 高次 Suzuki 公式
- Pauli-LCU block encoding と QSVT 射影位相列
- well-conditioned multiproduct formula (MPF) の coherent LCU
- 小規模系の statevector 計算と厳密行列指数との比較
- 大規模系の、許容誤差を固定した T / CNOT / 補助量子ビット数の比較

回路構成関数はハミルトニアンの密行列を作りません。密行列を使うのは、小規模検証の厳密解だけです。

## セットアップ（VS Code）

Python 3.10 以上と、VS Code の Python・Jupyter 拡張を用意します。スクリプトは `.venv` の作成と依存関係の導入だけを行い、ブラウザ版 Jupyter は起動しません。

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_venv.ps1
```

`python` 以外のコマンド名を使う場合は、たとえば次のように指定できます。

```powershell
.\setup_venv.ps1 -PythonCommand py
```

### macOS / Linux

```bash
bash setup_venv.sh
```

特定の Python を使う場合:

```bash
PYTHON=python3.12 bash setup_venv.sh
```

セットアップ後、VS Code で `notebooks/resource_comparison.ipynb` を直接開きます。カーネルが自動選択されない場合は、Notebook 右上のカーネル選択から次を指定してください。

```text
Windows: .venv\Scripts\python.exe
macOS/Linux: .venv/bin/python
```

`.vscode/settings.json` が workspace 内の `.venv` を検索対象として指定します。ユーザー領域への Jupyter kernelspec 登録は行わないため、環境はプロジェクト内で完結します。

## 構造

```text
src/hamiltonian_resources/
  hamiltonians.py    # Pauli 和入力、TFIM、Heisenberg 鎖
  trotter.py         # 積公式回路
  circuit_utils.py   # PREPARE / SELECT / block encoding
  qsvt.py            # QSVT 回路、次数見積り、pyqsp 位相合成
  multiproduct.py    # MPF 係数と coherent-LCU 回路
  simulation.py      # 小規模な厳密解との比較
  resources.py       # transpile 後の CX と T 見積り
  benchmark.py       # 固定誤差のパラメータ選択とスケーリング
notebooks/
  resource_comparison.ipynb
tests/
```

## 最小例

```python
from hamiltonian_resources import (
    BenchmarkConfig, benchmark_scaling, transverse_field_ising
)

config = BenchmarkConfig(time=1.0, target_error=1e-3)
table = benchmark_scaling(
    list(range(2, 21, 2)),
    lambda n: transverse_field_ising(n, coupling=1.0, field=0.7),
    config,
)
print(table[["system_size", "algorithm", "t_count", "cnot_count"]])
```

`benchmark_scaling(..., transpile_circuits=False)`（既定）は大規模向けの明示的な分解コストモデルです。`True` は実回路を Qiskit で基底ゲートへ分解するので、小規模な校正に使ってください。

## 比較上の重要な前提

1. **誤差予算**: アルゴリズム誤差と単一量子ビット回転合成誤差に分けます。`synthesis_error_fraction` で後者の割合を指定します。
2. **T 数**: 任意角 `Rz` は無料ではありません。各非 Clifford 回転に均等に精度を配り、`3 log2(1/epsilon_rot) + log2 log2(1/epsilon_rot)` 型の ancilla-free 合成コストで見積もります。
3. **固定誤差の次数選択**: `choose_parameters` は比較可能な保守的スケーリング proxy であり、ハミルトニアン固有の厳密誤差上界ではありません。小規模では `compare_with_exact` で校正してください。
4. **QSVT**: `exp(-iHt)=cos(Ht)-i sin(Ht)` の偶・奇成分は別の definite-parity QSP 列です。`synthesize_hamsim_phases` は各成分を生成し、`build_qsvt_circuit` は一成分、`build_hamiltonian_qsvt_circuit` は追加 LCU qubit で両成分を結合した回路を作ります。
5. **MPF**: 回路の zero-branch block は weighted sum を係数 1-norm で割ったものです。したがって postselection 成功率（または振幅増幅コスト）を無視して Trotter と比較してはいけません。回路 metadata に係数、1-norm、postselection 条件を保存します。
6. **大規模モデル**: 多重制御ゲートの CNOT 数はアーキテクチャ、clean/dirty ancilla、コンパイラで変わります。この実装の解析値は比較用の明記された分解モデルです。

## テスト

```powershell
pytest
ruff check src tests
```

## 参考文献

- G. H. Low, V. Kliuchnikov, N. Wiebe, [Well-conditioned multiproduct Hamiltonian simulation](https://arxiv.org/abs/1907.11679)
- A. Gilyén et al., [Quantum singular value transformation and beyond](https://arxiv.org/abs/1806.01838)
- [Qiskit `PauliEvolutionGate`](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.PauliEvolutionGate)
- [pyqsp](https://pypi.org/project/pyqsp/)（QSP 位相合成）
