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
  circuit_utils.py   # PREPARE / SELECT / block encoding / robust OAA
  qsvt.py            # Jacobi--Anger 位相合成、QSVT、LCU、robust OAA
  multiproduct.py    # MPF 係数、segment LCU、robust OAA
  simulation.py      # 小規模な厳密解との比較
  resources.py       # transpile 後の CX と T 見積り
  benchmark.py       # 固定誤差のパラメータ選択とスケーリング
notebooks/
  resource_comparison.ipynb
  qsvt_validation.ipynb
  mpf_validation.ipynb
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

QSVT回路は時刻と回路近似誤差を直接指定します。既定では、cos/sinの
coherent LCUに続けて1回のrobust oblivious amplitude amplificationを行います。

```python
from hamiltonian_resources import (
    build_hamiltonian_qsvt_circuit,
    compare_with_exact,
    transverse_field_ising,
)

H = transverse_field_ising(2)
circuit = build_hamiltonian_qsvt_circuit(H, time=0.2, epsilon=1e-3)
check = compare_with_exact(H, 0.2, method="qsvt", qsvt_epsilon=1e-3)
print(check)
```

`amplitude_amplification=False`では、all-zero ancilla blockは概ね
`scale * exp(-i H t) / 2`で、成功確率は`scale**2 / 4`です。増幅後は同じblockが
`exp(-i H t)`に近づきます。

`benchmark_scaling(..., transpile_circuits=False)`（既定）は大規模向けの明示的な分解コストモデルです。`True` は実回路を Qiskit で基底ゲートへ分解するので、小規模な校正に使ってください。

MPF では LCU の項数 `m` を指定すると、登録済みの well-conditioned な
Trotter 分割数が自動的に選ばれます。対応範囲は `m=2` から `m=15` です。
既定の`schedule="new"`は3-step OAA込みのquery数を抑える表で、
`schedule="legacy"`を指定すると以前の1-normが小さい表を使用できます。
時間を `segments` 個に分け、各segment内で
`sum_j a_j S_2(step_time / k_j)**k_j` をcoherent LCUとして作ります。
係数1-normは正負の相殺identity branchで2へpaddingされ、既定では各segmentを
1回の3-step robust OAAで増幅してから同じbranch register上で反復します。

```python
from hamiltonian_resources import (
    build_multiproduct_circuit, optimal_mpf_exponents, transverse_field_ising,
)

print(optimal_mpf_exponents(3))                    # (1, 2, 4)
print(optimal_mpf_exponents(3, schedule="legacy")) # (1, 2, 6)
H = transverse_field_ising(2)
mpf = build_multiproduct_circuit(H, time=0.2, m=3, segments=2)
```

`amplitude_amplification=False`は単一segmentのLCU検証専用です。このとき
zero-branch blockはMPF stepのちょうど1/2になります。

## 比較上の重要な前提

1. **誤差予算**: アルゴリズム誤差と単一量子ビット回転合成誤差に分けます。`synthesis_error_fraction` で後者の割合を指定します。
2. **T 数**: 任意角 `Rz` は無料ではありません。各非 Clifford 回転に均等に精度を配り、`3 log2(1/epsilon_rot) + log2 log2(1/epsilon_rot)` 型の ancilla-free 合成コストで見積もります。解析モデルでは多重制御ゲートの Toffoli 相当コストも temporary-AND（1 対あたり 4 T）で `t_count` に計上し、`toffoli_count` 列に対数を保存します。
3. **固定誤差の次数選択**: `choose_parameters` は次数 1・2 の積公式に対して、対象ハミルトニアンの入れ子交換子から計算した厳密上界 `W1 t^2 / r`、`W2 t^3 / r^2`（Childs–Su–Tran–Wiebe–Zhu, PRX 11, 011020 (2021)）を使います。局所ハミルトニアンでは `W2 = O(n)` であり、緩い 1-norm proxy `(alpha t)^3`（TFIM では `O(n^3)`）と違って QSVT の厳密な Jacobi–Anger 次数と同じ「タイトさ」で比較できます。MPF の segment 数は `alpha_eff = min(alpha, W2^(1/3))` による交換子校正 proxy で、これは証明付き上界ではありません。小規模では `compare_with_exact` で校正してください（テスト `test_chosen_*_meet_the_error_budget` が予算充足を検証します）。
4. **QSVT**: `exp(-iHt)=cos(Ht)-i sin(Ht)` の偶・奇成分は、Jacobi--Anger展開から同じscaleで生成します。`sym_qsp`の目標多項式はWx応答の虚部に現れるため、各成分は`V`と`V^dagger`のLCUで実blockとして抽出します。その後cos/sinを結合し、all-zero ancilla subspaceに対する3-step robust OAAを行います。生の位相配列を受け取る公開APIはありません。
5. **MPF**: 各segmentの増幅前zero-branch blockを`B=M/2`とすると、3-step OAA後のblockは厳密に`3B - 4 B B^dagger B`です。これは`M`がunitaryに近い範囲で`M`へ近づきます。同じbranch register上で増幅step unitaryを反復するため、複数segmentの最終blockを単純な`M**segments`と同一視はしません。metadataにはschedule、係数1-norm、padding、論理Gate数、controlled-`U2` query数を保存します。
6. **大規模モデル**: 多重制御ゲートの CNOT 数はアーキテクチャ、clean/dirty ancilla、コンパイラで変わります。この実装の解析値は比較用の明記された分解モデルです。
7. **解析コストと具体回路の対応**: 既定の解析式は具体回路の構造（QSVT の quadrature 抽出、cos/sin LCU、3-step OAA、MPF の identity padding・branch 幅・segment ごとの OAA factor 3）を反映します。したがって 3 手法とも「決定的動作あたり」の比較です。ただし controlled 応答回路については、`V` と `V^dagger` が block-encoding query を共有し projector 位相だけを選択する効率的コンパイルを仮定します。`transpile_circuits=True` は Qiskit の汎用 `.control()` 分解を使うため、これよりかなり大きな数値になります（校正時はこの差に注意）。

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
