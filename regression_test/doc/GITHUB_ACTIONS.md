# GitHub 自动测试

[工作流](../../.github/workflows/regression.yml) 使用标准 `ubuntu-24.04`，不访问集群，不需要 self-hosted runner、SSH 密钥或 `LIBBSE_CI_ENV`。

每次 push、PR 或手动触发：

1. 下载固定 LibRPA commit，安装 [Python 依赖](../backend/requirements.txt)。
2. 用 [构建脚本](../backend/build_ci_elpa.sh) 编译 ELPA 2024.05.001，核对源包 SHA256，启用 OpenMP 和 skew 接口。SIMD 编译参数适配当前 runner，构建与测试在同一机器进行。
3. 安装 FFTW3 开发包，以 `LIBBSE_ENABLE_FFTW=ON/OFF` 两个独立矩阵任务编译 LibBSE；一个失败不取消另一个。
4. **General testcases**：运行快速测试，包括实际 FFT 复数卷积或明确的不可用错误、GAGQ、分布式 Krylov、TR 拟合与缓存预算边界。
5. **Physical testcases**：运行自带 Si 222 光谱与 8 组 [LRI 核回归](../physical_testcases/lri_kernels/README.md)。Si 包括 R/FFT/k-MPI Haydock、x/xyz 与 GAGQ；核回归检查显式 Hx、缓存/回退、实际 TR 命中、线程受限及复数 W。关闭 FFTW 时只去掉依赖 FFT 后端的光谱测试。
6. 保存 CTest JUnit、构建日志和真实算例文本比较报告。失败返回非零，不静默跳过缺失数据。

本地 commit 不触发 GitHub，push 才触发；一次 push 多个 commit 时测试最新提交。数据与参考输出已经随仓库保存，CI 不重新运行 AIMS，也不重建参考答案。

首次启用需在仓库 Settings → Actions → General 允许 GitHub 官方 actions。推送后在 Actions → LibBSE regression 查看结果。分支规则可将 `Build and regression tests (FFTW ON)` 和 `Build and regression tests (FFTW OFF)` 设为合并所需检查。

外部十组模板算例仍可本地运行，配置见 [templates.toml](../physical_testcases/templates.toml)；它们不是本工作流默认运行内容。

参考：[标准 runner](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)、[触发事件](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)。

## 阶段和结果报告

主要阶段依次为 `Compile LibBSE`、`Test General Cases`、`Test Physical Cases`、`Postrun`、`Report`。准备依赖的步骤保留在编译之前。

- `Postrun` 仅上传诊断日志和 JUnit，失败时也执行，供定位问题。
- `Report` 始终尝试读取两组 JUnit，在 Actions Summary 显示通过数量、所有测试名称、🟢 PASS／🔴 FAIL 和失败原因，失败输出可展开查看。
- 缺失、跳过、损坏的报告不能算通过。general 测试失败后，只要编译成功且任务未取消，physical 测试仍执行。
- 编译后保存 `ctest --show-only=json-v1` 的实际注册清单，报告按与执行相同的标签表达式划分 general/physical；新增测试未出现在 JUnit 会失败，不用手工维护名称列表。清单缺失或损坏也失败。报告脚本与核门禁本身都有拒绝缺失/非有限/跳过结果的快速测试。
- 两种 FFTW 构建的 artifacts 分别命名；保存测试清单、两组 JUnit、核比较 JSON、执行命令/程序哈希及 per-rank LRI 日志。CI 不重建冻结参考，不用共享 runner 的耗时判断提速。
- CI 用 `LIBBSE_MPI_BIND_TO=none`，允许小输入超订阅到标准 runner；避免在 Slurm/cpuset 等受限环境中强制绑定到不可用 core。此设置只用于正确性回归，不能证明并行效率。核回归为共享 runner 留出 1200 秒上限，超时仍算失败。
- 删除重复比较器 job；比较器仍由 general CTest 执行。
- 只保留一次 `actions/checkout`；LibRPA 用固定 commit 的只读 Git 拉取，Python 使用 Ubuntu Python 的虚拟环境。由 checkout 生成的 `Post …` 属于 GitHub 自动清理，仍可能显示在自定义 `Report` 后面，无法合并到 `Postrun`。

报告脚本为 [ci_report.py](../backend/ci_report.py)。它仅使用 Python 标准库，不需要 JSON 输入。
