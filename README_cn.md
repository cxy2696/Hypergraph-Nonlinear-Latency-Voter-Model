# Hypergraph-Nonlinear-Latency Voter Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**中文** · [English](README_en.md) · [语言入口](README.md)

## 概述

本仓库是 **Hybrid Hypergraph-Nonlinear-Latency Voter Model（HNL Voter Model）** 的论文原型：用超图刻画群体互动，用非线性从众强度 `q` 刻画“跟随多数”的强度，用异质时延刻画“不是所有人同时反应”。

传统 Voter 模型假设两两对话、即时反应；HNL 增加了三件现实中常见的事：

- 人们往往在**群体**中互相影响（群聊、论坛、会议），而不只是两两交流；
- 采纳是**非线性**的，局部多数对个体的拉力由 `q` 控制；
- 每个人的反应速度不同，消息可能在发布很久后仍继续扩散。

论文全文：[Hypergraph-Nonlinear-Latency Voter Model.pdf](https://cxy2696.github.io/Hypergraph-Nonlinear-Latency-Voter-Model/HNL%20Voter%20Model.pdf)

## 适合研究什么

- **危机沟通**：什么时候辟谣，时机会改变结果吗？
- **社区讨论**：群体是在形成共识，还是被压成沉默的多数？
- **组织变革**：宣布节奏与反应延迟如何影响情绪？
- **产品发布 / 品牌**：早期声量等于长期口碑吗？
- **教学与研究**：演示从众与时延如何改变共识，复现经典 Voter 极限。

## 生产级后继项目

本仓库保留论文原型，便于复现原始结果。需要可复现、可服务化、带 API/报告/合规护栏的实现，请使用
**[OpinionLab](https://github.com/cxy2696/opinionlab)**：

- 节点中心、顺序无关的更新（原型的超边循环依赖顺序）；
- 确定性 RNG 流、运行清单与金标基线；
- CSR 向量化与事件驱动引擎，带蒙特卡洛置信区间；
- CLI/REST 服务、合规护栏与离线 HTML 报告。

## 在线体验

打开：**https://cxy2696.github.io/opinionlab/**

[![OpinionLab demo](https://raw.githubusercontent.com/cxy2696/opinionlab/main/docs/demo.png)](https://github.com/cxy2696/opinionlab/blob/main/docs/DEMO_cn.md)

- 15 个中英双语场景模板，可调 `q`、时延与副本数；
- 3D 工业风追踪视图，右上角切换中/英；
- 三张图带粒子流动演示，系统操作舱布局适配桌面/平板/手机；
- 一句话结论、决策提示与后续验证清单；
- 下载单文件离线报告。

本地运行：

```bash
git clone https://github.com/cxy2696/opinionlab.git
cd opinionlab
make demo          # http://127.0.0.1:8000
```

## 具体案例

- [辟谣时点](https://github.com/cxy2696/opinionlab/blob/main/examples/rumor_timing.py)：
  `uv run python examples/rumor_timing.py --repeats 8`
- [社区从众](https://github.com/cxy2696/opinionlab/blob/main/examples/community_consensus.py)：
  `uv run python examples/community_consensus.py --repeats 6`
- [使用场景与解读](https://github.com/cxy2696/opinionlab/blob/main/docs/USE_CASES_cn.md)

## 范围与局限

本仓库是论文原型：结果依赖超边顺序，使用全局随机状态，仅用于参考与复现。
模型不预测真实个人或事件，也不建模推荐系统、媒体议程或线下行为。
