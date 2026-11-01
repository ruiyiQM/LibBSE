# LibBSE regression tests

测试分为人工小算例和真实材料算例。测试通过表示对应固定输入与检查项通过，不等于所有 BSE 功能或物理近似都已验证。

| 目录 | 用途 |
|---|---|
| [general_testcase](general_testcase) | C++／Python 快速测试：ELPA/Haydock、GAGQ、分布式 Krylov、FFT 卷积、TR 拟合、缓存预算、文件转换与报告门禁 |
| [physical_testcases](physical_testcases) | 真实 Si 输入、冻结参考输出和 TOML 算例配置 |
| [backend](backend) | 执行、结果比较、报告汇总、依赖安装脚本 |
| [doc](doc) | 测试说明、GitHub 配置和验证记录 |

配置采用 TOML；数值数据采用压缩归档／NumPy；报告采用文本和 JUnit XML。数值比较会处理本征态整体复相位和简并态子空间，检查归一化、正交性与非有限值，不直接逐元素比较未对齐的振幅。

## 运行

在仓库根目录编译后：

```bash
python3 -m pip install -r regression_test/backend/requirements.txt
ctest --test-dir build -LE 'integration|real_si|lri_kernel' --output-on-failure
ctest --test-dir build -L 'real_si|lri_kernel' --output-on-failure
```

第二条执行快速测试，第三条执行自带 Si 光谱与 LRI 核回归。运行需要 MPI、ELPA 和配套编译依赖；Python 至少为 3.11。真实 Si 光谱测试默认注册，可用 `-DLIBBSE_TEST_REAL_SI=OFF` 关闭。

Si 测试包括 ELPA、Haydock 的三方向与单方向，以及单进程／2 MPI 的 GAGQ
光谱对照。设置 `-DLIBBSE_ENABLE_FFTW=ON` 后另注册单进程与
2 MPI 的 FFT-Haydock 光谱对照；FFT 单元测试在关闭 FFTW 时检查明确的
不可用错误，在开启时检查实际复数卷积。

[LRI 核回归](physical_testcases/lri_kernels/README.md) 使用冻结 H-chain
coarse4→fine8 与现有 Si222 输入，比对显式矩阵 Hx，并检查实际缓存、TR
复用和低内存回退。包含 8/32 MPI、OpenMP 线程受限、复数 W 与 rank 间
预算不一致；不会在 CI 设置速度门槛。GitHub 分别验证 FFTW ON/OFF。

真实材料回归的 `mpirun` 默认使用 `--bind-to core`。由调度器限制 CPU 集合时，
可设置 `LIBBSE_MPI_BIND_TO=none` 交由调度环境分配；实际命令保存在各运行的
`execution.txt` 中。性能基准应另行确认绑核，不能直接用此设置推断并行效率。

原先十组外部模板回归仍保留，配置见 [templates.toml](physical_testcases/templates.toml)。通过 CMake 的 `LIBBSE_REGRESSION_FIXTURES`、`LIBBSE_REGRESSION_REFERENCE_EXE` 注册，标签为 `integration`。该流程使用外部冻结程序；自带 Si 流程使用归档的参考数值，CI 不会自动重建参考答案。

## 文档

- [快速测试清单](doc/UNIT_TESTS.md)
- [真实 Si 测试与数据来源](physical_testcases/si222/README.md)
- [GitHub Actions](doc/GITHUB_ACTIONS.md)
- [Si 222 验证记录](doc/SI222_VALIDATION.md)
- [初次本地回归验证记录](doc/VALIDATION.md)
