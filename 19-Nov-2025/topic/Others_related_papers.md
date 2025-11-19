# Robust Verification of Controllers under State Uncertainty via Hamilton-Jacobi Reachability Analysis 

**Title (ZH)**: 基于Hamilton-Jacobi可达性分析的控制器在状态不确定性下的健壮验证 

**Authors**: Albert Lin, Alessandro Pinto, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2511.14755)  

**Abstract**: As perception-based controllers for autonomous systems become increasingly popular in the real world, it is important that we can formally verify their safety and performance despite perceptual uncertainty. Unfortunately, the verification of such systems remains challenging, largely due to the complexity of the controllers, which are often nonlinear, nonconvex, learning-based, and/or black-box. Prior works propose verification algorithms that are based on approximate reachability methods, but they often restrict the class of controllers and systems that can be handled or result in overly conservative analyses. Hamilton-Jacobi (HJ) reachability analysis is a popular formal verification tool for general nonlinear systems that can compute optimal reachable sets under worst-case system uncertainties; however, its application to perception-based systems is currently underexplored. In this work, we propose RoVer-CoRe, a framework for the Robust Verification of Controllers via HJ Reachability. To the best of our knowledge, RoVer-CoRe is the first HJ reachability-based framework for the verification of perception-based systems under perceptual uncertainty. Our key insight is to concatenate the system controller, observation function, and the state estimation modules to obtain an equivalent closed-loop system that is readily compatible with existing reachability frameworks. Within RoVer-CoRe, we propose novel methods for formal safety verification and robust controller design. We demonstrate the efficacy of the framework in case studies involving aircraft taxiing and NN-based rover navigation. Code is available at the link in the footnote. 

**Abstract (ZH)**: 基于感知的自主系统鲁棒验证框架：通过哈密尔顿-雅各比可达性分析 

---
# Mutation Testing for Industrial Robotic Systems 

**Title (ZH)**: 工业机器人系统中的突变测试 

**Authors**: Marcela Gonçalves dos Santos, Sylvain Hallé, Fábio Petrillo  

**Link**: [PDF](https://arxiv.org/pdf/2511.14432)  

**Abstract**: Industrial robotic systems (IRS) are increasingly deployed in diverse environments, where failures can result in severe accidents and costly downtime. Ensuring the reliability of the software controlling these systems is therefore critical. Mutation testing, a technique widely used in software engineering, evaluates the effectiveness of test suites by introducing small faults, or mutants, into the code. However, traditional mutation operators are poorly suited to robotic programs, which involve message-based commands and interactions with the physical world. This paper explores the adaptation of mutation testing to IRS by defining domain-specific mutation operators that capture the semantics of robot actions and sensor readings. We propose a methodology for generating meaningful mutants at the level of high-level read and write operations, including movement, gripper actions, and sensor noise injection. An empirical study on a pick-and-place scenario demonstrates that our approach produces more informative mutants and reduces the number of invalid or equivalent cases compared to conventional operators. Results highlight the potential of mutation testing to enhance test suite quality and contribute to safer, more reliable industrial robotic systems. 

**Abstract (ZH)**: 工业机器人系统（IRS）的变异测试 Adaptation of Mutation Testing for Industrial Robotic Systems 

---
# FICO: Finite-Horizon Closed-Loop Factorization for Unified Multi-Agent Path Finding 

**Title (ZH)**: FICO: 有限时域闭环因子分解方法统一多代理路径规划 

**Authors**: Jiarui Li, Alessandro Zanardi, Runyu Zhang, Gioele Zardini  

**Link**: [PDF](https://arxiv.org/pdf/2511.13961)  

**Abstract**: Multi-Agent Path Finding is a fundamental problem in robotics and AI, yet most existing formulations treat planning and execution separately and address variants of the problem in an ad hoc manner. This paper presents a system-level framework for MAPF that integrates planning and execution, generalizes across variants, and explicitly models uncertainties. At its core is the MAPF system, a formal model that casts MAPF as a control design problem encompassing classical and uncertainty-aware formulations. To solve it, we introduce Finite-Horizon Closed-Loop Factorization (FICO), a factorization-based algorithm inspired by receding-horizon control that exploits compositional structure for efficient closed-loop operation. FICO enables real-time responses -- commencing execution within milliseconds -- while scaling to thousands of agents and adapting seamlessly to execution-time uncertainties. Extensive case studies demonstrate that it reduces computation time by up to two orders of magnitude compared with open-loop baselines, while delivering significantly higher throughput under stochastic delays and agent arrivals. These results establish a principled foundation for analyzing and advancing MAPF through system-level modeling, factorization, and closed-loop design. 

**Abstract (ZH)**: 多智能体路径规划是机器人学和AI中的一个基础问题，但大多数现有形式将计划和执行分开处理，并针对问题的不同变体采取非系统的方法。本文提出了一种多智能体路径规划的系统级框架，该框架将计划和执行整合，跨不同变体进行推广，并明确建模不确定性。其核心是多智能体路径规划系统，这是一种形式模型，将多智能体路径规划问题表述为一个控制设计问题，涵盖了经典的和不确定性意识的形式模型。为了解决这个问题，我们引入了有限时域闭环因子分解（FICO）算法，这是一种受回溯_horizon控制启发、利用组合结构实现高效闭环操作的因子分解算法。FICO能够实现实时响应——在毫秒内开始执行，同时能扩展到数千个智能体，并无缝适应执行时的不确定性。广泛的实际案例研究表明，与开环基准相比，它将计算时间减少了两个数量级，同时在随机延迟和智能体到达时显著提高了吞吐量。这些结果为通过系统级建模、因子分解和闭环设计分析和推进多智能体路径规划奠定了原理性的基础。 

---
# Identifying Time-varying Costs in Finite-horizon Linear Quadratic Gaussian Games 

**Title (ZH)**: 有限区间线性二次高斯博弈中时变成本的识别 

**Authors**: Kai Ren, Maryam Kamgarpour  

**Link**: [PDF](https://arxiv.org/pdf/2511.14358)  

**Abstract**: We address cost identification in a finite-horizon linear quadratic Gaussian game. We characterize the set of cost parameters that generate a given Nash equilibrium policy. We propose a backpropagation algorithm to identify the time-varying cost parameters. We derive a probabilistic error bound when the cost parameters are identified from finite trajectories. We test our method in numerical and driving simulations. Our algorithm identifies the cost parameters that can reproduce the Nash equilibrium policy and trajectory observations. 

**Abstract (ZH)**: 我们在有限_horizon 线性二次高斯博弈中解决成本识别问题。我们刻画生成给定纳什均衡策略的成本参数集。我们提出一种反向传播算法来识别时间变化的成本参数。我们推导出成本参数从有限轨迹中被识别时的概率误差上界。我们在数值和驾驶模拟中测试了我们的方法。我们的算法能够识别出能够重现纳什均衡策略和轨迹观察的成本参数。 

---
# Multi-Timescale Model Predictive Control for Slow-Fast Systems 

**Title (ZH)**: 多时间尺度模型预测控制方法及其在慢-快系统中的应用 

**Authors**: Lukas Schroth, Daniel Morton, Amon Lahr, Daniele Gammelli, Andrea Carron, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2511.14311)  

**Abstract**: Model Predictive Control (MPC) has established itself as the primary methodology for constrained control, enabling autonomy across diverse applications. While model fidelity is crucial in MPC, solving the corresponding optimization problem in real time remains challenging when combining long horizons with high-fidelity models that capture both short-term dynamics and long-term behavior. Motivated by results on the Exponential Decay of Sensitivities (EDS), which imply that, under certain conditions, the influence of modeling inaccuracies decreases exponentially along the prediction horizon, this paper proposes a multi-timescale MPC scheme for fast-sampled control. Tailored to systems with both fast and slow dynamics, the proposed approach improves computational efficiency by i) switching to a reduced model that captures only the slow, dominant dynamics and ii) exponentially increasing integration step sizes to progressively reduce model detail along the horizon. We evaluate the method on three practically motivated robotic control problems in simulation and observe speed-ups of up to an order of magnitude. 

**Abstract (ZH)**: 基于指数衰减灵敏度的多时间尺度模型预测控制 

---
# Hessians in Birkhoff-Theoretic Trajectory Optimization 

**Title (ZH)**: Birkhoff论域轨迹优化中的海森矩阵 

**Authors**: I. M. Ross  

**Link**: [PDF](https://arxiv.org/pdf/2511.13963)  

**Abstract**: This paper derives various Hessians associated with Birkhoff-theoretic methods for trajectory optimization. According to a theorem proved in this paper, approximately 80% of the eigenvalues are contained in the narrow interval [-2, 4] for all Birkhoff-discretized optimal control problems. A preliminary analysis of computational complexity is also presented with further discussions on the grand challenge of solving a million point trajectory optimization problem. 

**Abstract (ZH)**: 本文推导了与Birkhoff理论方法相关的各种Hessian矩阵。根据本文证明的一个定理，所有Birkhoff离散化的最优控制问题中约80%的特征值包含在狭窄的区间[-2, 4]内。还提供了初步的计算复杂度分析，并进一步讨论了一百万点轨迹优化问题的挑战。 

---
# A Trajectory-free Crash Detection Framework with Generative Approach and Segment Map Diffusion 

**Title (ZH)**: 基于生成方法和段落地图扩散的无轨迹碰撞检测框架 

**Authors**: Weiying Shen, Hao Yu, Yu Dong, Pan Liu, Yu Han, Xin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2511.13795)  

**Abstract**: Real-time crash detection is essential for developing proactive safety management strategy and enhancing overall traffic efficiency. To address the limitations associated with trajectory acquisition and vehicle tracking, road segment maps recording the individual-level traffic dynamic data were directly served in crash detection. A novel two-stage trajectory-free crash detection framework, was present to generate the rational future road segment map and identify crashes. The first-stage diffusion-based segment map generation model, Mapfusion, conducts a noisy-to-normal process that progressively adds noise to the road segment map until the map is corrupted to pure Gaussian noise. The denoising process is guided by sequential embedding components capturing the temporal dynamics of segment map sequences. Furthermore, the generation model is designed to incorporate background context through ControlNet to enhance generation control. Crash detection is achieved by comparing the monitored segment map with the generations from diffusion model in second stage. Trained on non-crash vehicle motion data, Mapfusion successfully generates realistic road segment evolution maps based on learned motion patterns and remains robust across different sampling intervals. Experiments on real-world crashes indicate the effectiveness of the proposed two-stage method in accurately detecting crashes. 

**Abstract (ZH)**: 实时碰撞检测对于发展前瞻性安全管理策略和提升整体交通效率至关重要。为了解决轨迹获取和车辆跟踪的限制，采用了记录个体级交通动态数据的道路段落地图进行碰撞检测。提出了一种新颖的两阶段轨迹无导向碰撞检测框架，以生成合理的未来道路段落地图并识别碰撞。在第一阶段，基于扩散的段落地图生成模型Mapfusion通过逐步向道路段落地图添加噪声，直到地图被腐蚀为纯高斯噪声，实现从嘈杂到正常的转换过程。去噪过程由捕捉段落地图序列时空动态的顺序嵌入组件引导。此外，生成模型通过ControlNet整合背景上下文，以增强生成控制。碰撞检测通过将监测的段落地图与第二阶段扩散模型的生成结果进行比较来实现。Mapfusion在非碰撞车辆运动数据上训练，能够基于学习到的运动模式生成真实的道路段落演化地图，并且能够跨越不同的采样间隔保持鲁棒性。实验表明，所提出的两阶段方法在准确检测碰撞方面是有效的。 

---
# Who Moved My Distribution? Conformal Prediction for Interactive Multi-Agent Systems 

**Title (ZH)**: 谁移走了我的分布？交互式多智能体系统的同核预测 

**Authors**: Allen Emmanuel Binny, Anushri Dixit  

**Link**: [PDF](https://arxiv.org/pdf/2511.11567)  

**Abstract**: Uncertainty-aware prediction is essential for safe motion planning, especially when using learned models to forecast the behavior of surrounding agents. Conformal prediction is a statistical tool often used to produce uncertainty-aware prediction regions for machine learning models. Most existing frameworks utilizing conformal prediction-based uncertainty predictions assume that the surrounding agents are non-interactive. This is because in closed-loop, as uncertainty-aware agents change their behavior to account for prediction uncertainty, the surrounding agents respond to this change, leading to a distribution shift which we call endogenous distribution shift. To address this challenge, we introduce an iterative conformal prediction framework that systematically adapts the uncertainty-aware ego-agent controller to the endogenous distribution shift. The proposed method provides probabilistic safety guarantees while adapting to the evolving behavior of reactive, non-ego agents. We establish a model for the endogenous distribution shift and provide the conditions for the iterative conformal prediction pipeline to converge under such a distribution shift. We validate our framework in simulation for 2- and 3- agent interaction scenarios, demonstrating collision avoidance without resulting in overly conservative behavior and an overall improvement in success rates of up to 9.6% compared to other conformal prediction-based baselines. 

**Abstract (ZH)**: 基于不确定性预测的安全运动规划至关重要，尤其是在使用学习模型预测周围代理行为时。内生分布迁移下的迭代自适应典型预测框架为反应性非ego代理的 evolving 行为提供了概率安全保证。我们建立了内生分布迁移模型，并提供了在该分布迁移下迭代典型预测管道收敛的条件。我们在模拟的2-和3-代理交互场景中验证了该框架，展示了有效的碰撞避免且未导致过于保守的行为，并在成功率上提高了最多9.6%，优于其他基于典型预测的基线方法。 

---
# Heterogeneous Multi-Agent Proximal Policy Optimization for Power Distribution System Restoration 

**Title (ZH)**: 异构多代理近端策略优化在电力配网恢复中的应用 

**Authors**: Parya Dolatyabi, Mahdi Khodayar  

**Link**: [PDF](https://arxiv.org/pdf/2511.14730)  

**Abstract**: Restoring power distribution systems (PDS) after large-scale outages requires sequential switching operations that reconfigure feeder topology and coordinate distributed energy resources (DERs) under nonlinear constraints such as power balance, voltage limits, and thermal ratings. These challenges make conventional optimization and value-based RL approaches computationally inefficient and difficult to scale. This paper applies a Heterogeneous-Agent Reinforcement Learning (HARL) framework, instantiated through Heterogeneous-Agent Proximal Policy Optimization (HAPPO), to enable coordinated restoration across interconnected microgrids. Each agent controls a distinct microgrid with different loads, DER capacities, and switch counts, introducing practical structural heterogeneity. Decentralized actor policies are trained with a centralized critic to compute advantage values for stable on-policy updates. A physics-informed OpenDSS environment provides full power flow feedback and enforces operational limits via differentiable penalty signals rather than invalid action masking. The total DER generation is capped at 2400 kW, and each microgrid must satisfy local supply-demand feasibility. Experiments on the IEEE 123-bus and IEEE 8500-node systems show that HAPPO achieves faster convergence, higher restored power, and smoother multi-seed training than DQN, PPO, MAES, MAGDPG, MADQN, Mean-Field RL, and QMIX. Results demonstrate that incorporating microgrid-level heterogeneity within the HARL framework yields a scalable, stable, and constraint-aware solution for complex PDS restoration. 

**Abstract (ZH)**: 基于异构代理强化学习的发电系统大规模中断后恢复 

---
# Rate-Distortion Guided Knowledge Graph Construction from Lecture Notes Using Gromov-Wasserstein Optimal Transport 

**Title (ZH)**: 基于重新率-失真指导的知识图谱从讲座笔记构建方法：使用Gromov-Wasserstein最优传输 

**Authors**: Yuan An, Ruhma Hashmi, Michelle Rogers, Jane Greenberg, Brian K. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2511.14595)  

**Abstract**: Task-oriented knowledge graphs (KGs) enable AI-powered learning assistant systems to automatically generate high-quality multiple-choice questions (MCQs). Yet converting unstructured educational materials, such as lecture notes and slides, into KGs that capture key pedagogical content remains difficult. We propose a framework for knowledge graph construction and refinement grounded in rate-distortion (RD) theory and optimal transport geometry. In the framework, lecture content is modeled as a metric-measure space, capturing semantic and relational structure, while candidate KGs are aligned using Fused Gromov-Wasserstein (FGW) couplings to quantify semantic distortion. The rate term, expressed via the size of KG, reflects complexity and compactness. Refinement operators (add, merge, split, remove, rewire) minimize the rate-distortion Lagrangian, yielding compact, information-preserving KGs. Our prototype applied to data science lectures yields interpretable RD curves and shows that MCQs generated from refined KGs consistently surpass those from raw notes on fifteen quality criteria. This study establishes a principled foundation for information-theoretic KG optimization in personalized and AI-assisted education. 

**Abstract (ZH)**: 面向任务的知识图谱（KGs）使AI驱动的学习助手系统能够自动生成高质量的多项选择题（MCQs）。然而，将讲座笔记和幻灯片等非结构化教育资源转换为能够捕捉关键教学内容的知识图谱仍然具有挑战性。我们提出了一种基于率失真（RD）理论和最优运输几何的知识图谱构建与精炼框架。在该框架中，讲座内容被建模为度量测度空间，捕捉语义和关系结构，而候选知识图谱则通过融合格罗莫夫-瓦兹松耦合（FGW）对齐以量化语义失真。率项通过知识图谱的大小体现，反映复杂性和紧凑性。精炼操作（添加、合并、拆分、删除、重连）通过最小化率失真拉格朗日表达式，生成紧凑且信息保留的知识图谱。我们的原型应用于数据科学讲座，生成可解释的RD曲线，并展示了从精炼知识图谱生成的多项选择题在十五项质量标准上均优于原始笔记生成的多项选择题。本研究奠定了信息论知识图谱优化在个性化和AI辅助教育中的原则性基础。 

---
# Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation 

**Title (ZH)**: 听 like 一位教师：利用自适应层注意力和知识蒸馏减轻耳语幻觉 

**Authors**: Kumud Tripathi, Aditya Srinivas Menon, Aman Gaurav, Raj Prakash Gohil, Pankaj Wasnik  

**Link**: [PDF](https://arxiv.org/pdf/2511.14219)  

**Abstract**: The Whisper model, an open-source automatic speech recognition system, is widely adopted for its strong performance across multilingual and zero-shot settings. However, it frequently suffers from hallucination errors, especially under noisy acoustic conditions. Previous works to reduce hallucinations in Whisper-style ASR systems have primarily focused on audio preprocessing or post-processing of transcriptions to filter out erroneous content. However, modifications to the Whisper model itself remain largely unexplored to mitigate hallucinations directly. To address this challenge, we present a two-stage architecture that first enhances encoder robustness through Adaptive Layer Attention (ALA) and further suppresses hallucinations using a multi-objective knowledge distillation (KD) framework. In the first stage, ALA groups encoder layers into semantically coherent blocks via inter-layer correlation analysis. A learnable multi-head attention module then fuses these block representations, enabling the model to jointly exploit low- and high-level features for more robust encoding. In the second stage, our KD framework trains the student model on noisy audio to align its semantic and attention distributions with a teacher model processing clean inputs. Our experiments on noisy speech benchmarks show notable reductions in hallucinations and word error rates, while preserving performance on clean speech. Together, ALA and KD offer a principled strategy to improve Whisper's reliability under real-world noisy conditions. 

**Abstract (ZH)**: 一种增强 Whisper 模型鲁棒性以减轻幻听错误的两阶段架构 

---
# Beyond Accuracy: A Multi-Dimensional Framework for Evaluating Enterprise Agentic AI Systems 

**Title (ZH)**: 超越准确性：评估企业代理人工智能系统的多维度框架 

**Authors**: Sushant Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2511.14136)  

**Abstract**: Current agentic AI benchmarks predominantly evaluate task completion accuracy, while overlooking critical enterprise requirements such as cost-efficiency, reliability, and operational stability. Through systematic analysis of 12 main benchmarks and empirical evaluation of state-of-the-art agents, we identify three fundamental limitations: (1) absence of cost-controlled evaluation leading to 50x cost variations for similar precision, (2) inadequate reliability assessment where agent performance drops from 60\% (single run) to 25\% (8-run consistency), and (3) missing multidimensional metrics for security, latency, and policy compliance. We propose \textbf{CLEAR} (Cost, Latency, Efficacy, Assurance, Reliability), a holistic evaluation framework specifically designed for enterprise deployment. Evaluation of six leading agents on 300 enterprise tasks demonstrates that optimizing for accuracy alone yields agents 4.4-10.8x more expensive than cost-aware alternatives with comparable performance. Expert evaluation (N=15) confirms that CLEAR better predicts production success (correlation $\rho=0.83$) compared to accuracy-only evaluation ($\rho=0.41$). 

**Abstract (ZH)**: 当前的企业AI基准主要评估任务完成准确性，忽视了成本效率、可靠性和操作稳定性等关键企业需求。通过对12个主要基准的系统分析和对先进代理的实证评估，我们识别出三种根本性限制：（1）缺乏成本控制评估导致相似精度下成本变化50倍，（2）可靠性评估不足，导致代理性能从单一运行的60%下降到八次运行一致性的25%，（3）缺少关于安全、延迟和政策合规性的多维度指标。我们提出了一种综合评价框架CLEAR（Cost, Latency, Efficacy, Assurance, Reliability），该框架专门设计用于企业部署。在300个企业任务上对六种领先代理的评估表明，仅仅优化准确性会导致代理比成本意识型替代方案高出4.4-10.8倍的成本，且具有可比性能。专家评估（N=15）确认，与仅仅基于准确性的评估相比，CLEAR更能预测生产成功（相关性ρ=0.83 vs ρ=0.41）。 

---
# PRISM: Prompt-Refined In-Context System Modelling for Financial Retrieval 

**Title (ZH)**: PRISM: 基于提示精炼的上下文建模方法在金融检索中的应用 

**Authors**: Chun Chet Ng, Jia Yu Lim, Wei Zeng Low  

**Link**: [PDF](https://arxiv.org/pdf/2511.14130)  

**Abstract**: With the rapid progress of large language models (LLMs), financial information retrieval has become a critical industrial application. Extracting task-relevant information from lengthy financial filings is essential for both operational and analytical decision-making. The FinAgentBench dataset formalizes this problem through two tasks: document ranking and chunk ranking. We present PRISM, a training-free framework that integrates refined system prompting, in-context learning (ICL), and a lightweight multi-agent system. Each component is examined extensively to reveal their synergies: prompt engineering provides precise task instructions, ICL supplies semantically relevant few-shot examples, and the multi-agent system models coordinated scoring behaviour. Our best configuration achieves an NDCG@5 of 0.71818 on the restricted validation split. We further demonstrate that PRISM is feasible and robust for production-scale financial retrieval. Its modular, inference-only design makes it practical for real-world use cases. The source code is released at this https URL. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的迅速发展，金融信息检索已成为关键的工业应用。从长篇金融文件中提取与任务相关的信息对于操作和分析决策至关重要。FinAgentBench数据集通过两个任务——文档排名和片段排名——正式化了这一问题。我们介绍了PRISM，这是一种无需训练的框架，结合了精炼的系统提示、类条件学习（ICL）和轻量级多智能体系统。每个组件都被详细分析以揭示它们的协同作用：提示工程提供了精确的任务说明，类条件学习提供了语义相关的少量示例，而多智能体系统则模拟了协调评分行为。我们最佳配置在受限验证分割上的NDCG@5达到0.71818。我们进一步证明，PRISM在生产规模的金融检索中是可行且稳健的。其模块化、仅推理的设计使其适用于实际应用场景。源代码发布于此链接。 

---
# Making Evidence Actionable in Adaptive Learning 

**Title (ZH)**: 让证据在适应性学习中发挥作用 

**Authors**: Amirreza Mehrabi, Jason W. Morphew, Breejha Quezada, N. Sanjay Rebello  

**Link**: [PDF](https://arxiv.org/pdf/2511.14052)  

**Abstract**: Adaptive learning often diagnoses precisely yet intervenes weakly, yielding help that is mistimed or misaligned. This study presents evidence supporting an instructor-governed feedback loop that converts concept-level assessment evidence into vetted micro-interventions. The adaptive learning algorithm contains three safeguards: adequacy as a hard guarantee of gap closure, attention as a budgeted constraint for time and redundancy, and diversity as protection against overfitting to a single resource. We formalize intervention assignment as a binary integer program with constraints for coverage, time, difficulty windows informed by ability estimates, prerequisites encoded by a concept matrix, and anti-redundancy enforced through diversity. Greedy selection serves low-richness and tight-latency regimes, gradient-based relaxation serves rich repositories, and a hybrid method transitions along a richness-latency frontier. In simulation and in an introductory physics deployment with one thousand two hundred four students, both solvers achieved full skill coverage for essentially all learners within bounded watch time. The gradient-based method reduced redundant coverage by approximately twelve percentage points relative to greedy and harmonized difficulty across slates, while greedy delivered comparable adequacy with lower computational cost in scarce settings. Slack variables localized missing content and supported targeted curation, sustaining sufficiency across subgroups. The result is a tractable and auditable controller that closes the diagnostic-pedagogical loop and delivers equitable, load-aware personalization at classroom scale. 

**Abstract (ZH)**: 自适应学习往往精确诊断但轻微干预，导致帮助时机不当或不对齐。本研究提供了支持教师监管的反馈循环的证据，将概念级评估证据转化为经过验证的微干预。自适应学习算法包含三个保障措施：充分性作为确保空白填补的硬性保证，注意度作为时间冗余的预算约束，多样性作为防止过度拟合单一资源的保护。我们将干预分配形式化为具有覆盖、时间、基于能力估计的难度窗口、通过概念矩阵编码的先修条件以及通过多样性防止冗余的二元整数规划问题。贪婪选择服务于低丰富度和紧延迟区域，基于梯度的松弛服务于丰富的资源库，混合方法在丰富度-延迟前沿上进行转换。在模拟及一万零四名学生的大学物理课部署中，两种求解器均在限定的观察时间内实现了几乎所有学习者的关键技能全覆盖。基于梯度的方法将冗余覆盖减少了约12个百分点，并在整个题集中实现了难度的和谐性，而贪婪方法在资源稀缺时以较低的计算成本实现了相当的充分性。松弛变量定位了缺失的内容并支持了有针对性的内容筛选，确保了亚组的充分性。结果是可管理且可审计的控制器，可以封闭诊断-教学循环，并在教室规模上提供公平、负担感知的个性化。 

---
# AISAC: An Integrated multi-agent System for Transparent, Retrieval-Grounded Scientific Assistance 

**Title (ZH)**: AISAC：一种集成的透明化检索驱动科学辅助多-agent系统 

**Authors**: Chandrachur Bhattacharya, Sibendu Som  

**Link**: [PDF](https://arxiv.org/pdf/2511.14043)  

**Abstract**: AI Scientific Assistant Core (AISAC) is an integrated multi-agent system developed at Argonne National Laboratory for scientific and engineering workflows. AISAC builds on established technologies - LangGraph for orchestration, FAISS for vector search, and SQLite for persistence - and integrates them into a unified system prototype focused on transparency, provenance tracking, and scientific adaptability.
The system implements a Router-Planner-Coordinator workflow and an optional Evaluator role, using prompt-engineered agents coordinated via LangGraph's StateGraph and supported by helper agents such as a Researcher. Each role is defined through custom system prompts that enforce structured JSON outputs. A hybrid memory approach (FAISS + SQLite) enables both semantic retrieval and structured conversation history. An incremental indexing strategy based on file hashing minimizes redundant re-embedding when scientific corpora evolve. A configuration-driven project bootstrap layer allows research teams to customize tools, prompts, and data sources without modifying core code.
All agent decisions, tool invocations, and retrievals are logged and visualized through a custom Gradio interface, providing step-by-step transparency for each reasoning episode. The authors have applied AISAC to multiple research areas at Argonne, including specialized deployments for waste-to-products research and energy process safety, as well as general-purpose scientific assistance, demonstrating its cross-domain applicability. 

**Abstract (ZH)**: AI科学助手核心（AISAC）是一款在阿贡国家实验室开发的集成多智能体系统，用于科学和工程工作流。AISAC基于已有的技术——LangGraph进行编排、FAISS进行向量搜索以及SQLite进行持久化，并将它们整合成为一个专注于透明性、可溯源性和科学适应性的统一系统原型。系统实现了路由器-规划者-协调员工作流以及可选的评估员角色，利用LangGraph的StateGraph协调由提示工程构建的智能体，并由如研究员这样的辅助智能体支持。每个角色通过定制的系统提示来定义，并强制输出结构化的JSON数据。混合记忆方法（FAISS + SQLite）实现语义检索和结构化对话历史的结合。基于文件哈希的渐进式索引策略在科学语料库演化时最小化重复嵌入。配置驱动的项目启动层允许研究团队自定义工具、提示和数据源而无需修改核心代码。所有智能体的决策、工具调用和检索都被记录并通过自定义的Gradio界面可视化，为每一步推理提供了逐步的透明性。作者已将AISAC应用于阿贡的多个研究领域，包括废弃物到产品的研究和能源过程安全的专用部署，以及通用科学研究，展示了其跨领域的适用性。 

---
# CORGI: Efficient Pattern Matching With Quadratic Guarantees 

**Title (ZH)**: CORGI: 具有二次保证的有效模式匹配 

**Authors**: Daniel Weitekamp  

**Link**: [PDF](https://arxiv.org/pdf/2511.13942)  

**Abstract**: Rule-based systems must solve complex matching problems within tight time constraints to be effective in real-time applications, such as planning and reactive control for AI agents, as well as low-latency relational database querying. Pattern-matching systems can encounter issues where exponential time and space are required to find matches for rules with many underconstrained variables, or which produce combinatorial intermediate partial matches (but are otherwise well-constrained). When online AI systems automatically generate rules from example-driven induction or code synthesis, they can easily produce worst-case matching patterns that slow or halt program execution by exceeding available memory. In our own work with cognitive systems that learn from example, we've found that aggressive forms of anti-unification-based generalization can easily produce these circumstances. To make these systems practical without hand-engineering constraints or succumbing to unpredictable failure modes, we introduce a new matching algorithm called CORGI (Collection-Oriented Relational Graph Iteration). Unlike RETE-based approaches, CORGI offers quadratic time and space guarantees for finding single satisficing matches, and the ability to iteratively stream subsequent matches without committing entire conflict sets to memory. CORGI differs from RETE in that it does not have a traditional $\beta$-memory for collecting partial matches. Instead, CORGI takes a two-step approach: a graph of grounded relations is built/maintained in a forward pass, and an iterator generates matches as needed by working backward through the graph. This approach eliminates the high-latency delays and memory overflows that can result from populating full conflict sets. In a performance evaluation, we demonstrate that CORGI significantly outperforms RETE implementations from SOAR and OPS5 on a simple combinatorial matching task. 

**Abstract (ZH)**: 规则基础系统必须在严格的时间限制内解决复杂的匹配问题，以在如AI代理的即时规划和反应控制以及低延迟关系数据库查询等实际应用中发挥作用。模式匹配系统可能会遇到需要指数时间复杂度和空间来匹配具有许多欠约束变量的规则，或产生组合中间部分匹配的情况（但其他方面受良好约束）。当在线AI系统从实例驱动的归纳或代码合成自动生成规则时，它们可能会轻易地生成最坏情况下的匹配模式，这些模式通过超出行存内存限制而减慢或停止程序执行。在我们自己的认知系统工作中，我们发现基于反统一的强大形式的泛化很容易产生这些情况。为了使这些系统在无需手工工程约束或陷入不可预测的失败模式的情况下变得实用，我们提出了一种新的匹配算法，称为CORGI（Collection-Oriented Relational Graph Iteration）。与RETE基于的方法不同，CORGI为查找单个使满足匹配提供了二次时间与空间保证，并且能够迭代地流式传输后续匹配而无需将整个冲突集提交至内存。与RETE不同的是，CORGI没有传统的β-内存来收集部分匹配。相反，CORGI采用两步方法：在前向传递过程中构建/维护一个关系图，并且迭代器根据需要通过反向遍历该图来生成匹配。这种方法消除了因填充完整冲突集而导致的高延迟延迟和内存溢出。在性能评估中，我们展示出CORGI在一项简单的组合匹配任务中显著优于SOAR和OPS5的RETE实现。 

---
# Causal computations in Semi Markovian Structural Causal Models using divide and conquer 

**Title (ZH)**: Semi马尔可夫结构因果模型中的因果计算及其分而治之方法 

**Authors**: Anna Rodum Bjøru, Rafael Cabañas, Helge Langseth, Antonio Salmerón  

**Link**: [PDF](https://arxiv.org/pdf/2511.13852)  

**Abstract**: Recently, Bjøru et al. proposed a novel divide-and-conquer algorithm for bounding counterfactual probabilities in structural causal models (SCMs). They assumed that the SCMs were learned from purely observational data, leading to an imprecise characterization of the marginal distributions of exogenous variables. Their method leveraged the canonical representation of structural equations to decompose a general SCM with high-cardinality exogenous variables into a set of sub-models with low-cardinality exogenous variables. These sub-models had precise marginals over the exogenous variables and therefore admitted efficient exact inference. The aggregated results were used to bound counterfactual probabilities in the original model. The approach was developed for Markovian models, where each exogenous variable affects only a single endogenous variable. In this paper, we investigate extending the methodology to \textit{semi-Markovian} SCMs, where exogenous variables may influence multiple endogenous variables. Such models are capable of representing confounding relationships that Markovian models cannot. We illustrate the challenges of this extension using a minimal example, which motivates a set of alternative solution strategies. These strategies are evaluated both theoretically and through a computational study. 

**Abstract (ZH)**: 最近，Bjøru等人提出了一种新的分而治之算法，用于结构因果模型（SCMs）中反事实概率的边界估计。该方法假定SCMs是从纯观察数据中学习得到的，导致对外生变量边缘分布的刻画不够精确。该方法利用结构方程的典范表示将具有高基数外生变量的通用SCM分解为具有低基数外生变量的子模型集。这些子模型对外生变量具有精确的边缘分布，因此允许高效的精确推理。汇总的结果被用于在原始模型中估计反事实概率。该方法应用于马尔可夫模型，其中每个外生变量仅影响一个内生变量。本文探讨将该方法扩展到半马尔可夫模型（exogenous变量可能影响多个内生变量）的可行性。此类模型能够表示马尔可夫模型无法表示的混杂关系。利用一个最小示例说明这种扩展的挑战，并提出若干替代解决方案策略。这些策略从理论上和计算上进行了评估。 

---
# When AI Does Science: Evaluating the Autonomous AI Scientist KOSMOS in Radiation Biology 

**Title (ZH)**: 当AI进行科学研究：评估自主AI科学家KOSMOS在辐射生物学中的表现 

**Authors**: Humza Nusrat, Omar Nusrat  

**Link**: [PDF](https://arxiv.org/pdf/2511.13825)  

**Abstract**: Agentic AI "scientists" now use language models to search the literature, run analyses, and generate hypotheses. We evaluate KOSMOS, an autonomous AI scientist, on three problems in radiation biology using simple random-gene null benchmarks. Hypothesis 1: baseline DNA damage response (DDR) capacity across cell lines predicts the p53 transcriptional response after irradiation (GSE30240). Hypothesis 2: baseline expression of OGT and CDO1 predicts the strength of repressed and induced radiation-response modules in breast cancer cells (GSE59732). Hypothesis 3: a 12-gene expression signature predicts biochemical recurrence-free survival after prostate radiotherapy plus androgen deprivation therapy (GSE116918). The DDR-p53 hypothesis was not supported: DDR score and p53 response were weakly negatively correlated (Spearman rho = -0.40, p = 0.76), indistinguishable from random five-gene scores. OGT showed only a weak association (r = 0.23, p = 0.34), whereas CDO1 was a clear outlier (r = 0.70, empirical p = 0.0039). The 12-gene signature achieved a concordance index of 0.61 (p = 0.017) but a non-unique effect size. Overall, KOSMOS produced one well-supported discovery, one plausible but uncertain result, and one false hypothesis, illustrating that AI scientists can generate useful ideas but require rigorous auditing against appropriate null models. 

**Abstract (ZH)**: 代理型AI科学家现在使用语言模型搜索文献、运行分析并生成假设。我们使用简单的随机基因缺失基准评估了自主AI科学家KOSMOS在辐射生物学中的三个问题。假设1：各细胞系的基础DNA损伤反应（DDR）能力预测电离辐射后p53转录响应（GSE30240）。假设2：基础OGT和CDO1表达预测乳腺癌细胞中沉默和诱导辐射响应模块的强度（GSE59732）。假设3：12基因表达特征预测前列腺放疗加雄激素剥夺治疗后无生化复发的生存概率（GSE116918）。DDR-p53假设未得到支持：DDR得分和p53响应之间相关性较弱（ Spearman rho = -0.40，p = 0.76），与随机五基因得分无显著差异。OGT仅表现出弱相关性（r = 0.23，p = 0.34），而CDO1是一个明显的异常值（r = 0.70，经验p = 0.0039）。12基因特征获得了0.61的一致性指标（p = 0.017），但非唯一的效果大小。总体而言，KOSMOS产生了一个受支持的发现，一个可能是正确的但不确定的结果，以及一个虚假的假设，这表明代理型AI科学家可以生成有用的想法，但需要通过对适当的随机模型进行严格的审计来进行验证。 

---
# Automated proving in planar geometry based on the complex number identity method and elimination 

**Title (ZH)**: 基于复数恒等方法与消元的平面几何自动证明 

**Authors**: Zoltán Kovács, Xicheng Peng  

**Link**: [PDF](https://arxiv.org/pdf/2511.14728)  

**Abstract**: We improve the complex number identity proving method to a fully automated procedure, based on elimination ideals. By using declarative equations or rewriting each real-relational hypothesis $h_i$ to $h_i-r_i$, and the thesis $t$ to $t-r$, clearing the denominators and introducing an extra expression with a slack variable, we eliminate all free and relational point variables. From the obtained ideal $I$ in $\mathbb{Q}[r,r_1,r_2,\ldots]$ we can find a conclusive result. It plays an important role that if $r_1,r_2,\ldots$ are real, $r$ must also be real if there is a linear polynomial $p(r)\in I$, unless division by zero occurs when expressing $r$. Our results are presented in Mathematica, Maple and in a new version of the Giac computer algebra system. Finally, we present a prototype of the automated procedure in an experimental version of the dynamic geometry software GeoGebra. 

**Abstract (ZH)**: 我们基于消元理想改进了复数恒等式证明方法，使其成为完全自动化的程序。通过使用声明性方程或将每个实关系假设 $h_i$ 转换为 $h_i-r_i$，并将论题 $t$ 转换为 $t-r$，清除分母并引入一个带松弛变量的额外表达式，我们消除了所有自由和关系点变量。从在 $\mathbb{Q}[r,r_1,r_2,\ldots]$ 中获得的理想 $I$ 中，我们可以找到结论性结果。如果 $r_1, r_2, \ldots$ 是实数，除非在表示 $r$ 时出现除零情况，否则 $r$ 也必须是实数。我们的结果在 Mathematica、Maple 和 Giac 计算机代数系统的最新版本中呈现。最后，我们在动态几何软件 GeoGebra 的实验版本中展示了该自动程序的原型。 

---
# \textit{FLARE}: Adaptive Multi-Dimensional Reputation for Robust Client Reliability in Federated Learning 

**Title (ZH)**: FLARE：用于联邦学习中稳健客户端可靠性的自适应多维声誉系统 

**Authors**: Abolfazl Younesi, Leon Kiss, Zahra Najafabadi Samani, Juan Aznar Poveda, Thomas Fahringer  

**Link**: [PDF](https://arxiv.org/pdf/2511.14715)  

**Abstract**: Federated learning (FL) enables collaborative model training while preserving data privacy. However, it remains vulnerable to malicious clients who compromise model integrity through Byzantine attacks, data poisoning, or adaptive adversarial behaviors. Existing defense mechanisms rely on static thresholds and binary classification, failing to adapt to evolving client behaviors in real-world deployments. We propose FLARE, an adaptive reputation-based framework that transforms client reliability assessment from binary decisions to a continuous, multi-dimensional trust evaluation. FLARE integrates: (i) a multi-dimensional reputation score capturing performance consistency, statistical anomaly indicators, and temporal behavior, (ii) a self-calibrating adaptive threshold mechanism that adjusts security strictness based on model convergence and recent attack intensity, (iii) reputation-weighted aggregation with soft exclusion to proportionally limit suspicious contributions rather than eliminating clients outright, and (iv) a Local Differential Privacy (LDP) mechanism enabling reputation scoring on privatized client updates. We further introduce a highly evasive Statistical Mimicry (SM) attack, a benchmark adversary that blends honest gradients with synthetic perturbations and persistent drift to remain undetected by traditional filters. Extensive experiments with 100 clients on MNIST, CIFAR-10, and SVHN demonstrate that FLARE maintains high model accuracy and converges faster than state-of-the-art Byzantine-robust methods under diverse attack types, including label flipping, gradient scaling, adaptive attacks, ALIE, and SM. FLARE improves robustness by up to 16% and preserves model convergence within 30% of the non-attacked baseline, while achieving strong malicious-client detection performance with minimal computational overhead. this https URL 

**Abstract (ZH)**: 联邦学习（FL）能够在保护数据隐私的同时实现模型协作训练。然而，它仍然容易受到通过拜占庭攻击、数据投毒或适应性对抗行为的恶意客户端的攻击。现有的防御机制依赖于静态阈值和二元分类，难以适应实际部署中不断演化的客户端行为。我们提出了一种适应性声誉框架FLARE，将客户端可靠性评估从二元决策转化为连续的多维度信任评估。FLARE集成了：(i) 多维度声誉得分，捕捉性能一致性、统计异常指标和时间行为；(ii) 自校准的自适应阈值机制，根据模型收敛性和近期攻击强度调整安全严苛程度；(iii) 基于声誉加权聚合的软排除机制，以比例限制可疑贡献而不过早排除客户端；(iv) 局部差分隐私(LDP)机制，使声誉评分能够在私有化客户端更新上进行。我们进一步介绍了一种高度规避的统计模仿（SM）攻击，这是一种基准对手，它将诚实梯度与合成扰动和持续漂移结合起来，以逃避传统过滤器的检测。在MNIST、CIFAR-10和SVHN数据集上的100个客户端的广泛实验表明，FLARE在多种攻击类型（包括标签翻转、梯度缩放、适应性攻击、ALIE和SM）下能够保持高模型准确性并更快地收敛，其鲁棒性提高高达16%，同时能够在30%的非攻击基准下保持模型收敛，且具有较低的计算开销实现强大的恶意客户端检测性能。 

---
# Seeing Beyond the Image: ECG and Anatomical Knowledge-Guided Myocardial Scar Segmentation from Late Gadolinium-Enhanced Images 

**Title (ZH)**: 超越图像视角：基于晚期钆增强影像的心脏解剖知识指导的心肌瘢痕分割 

**Authors**: Farheen Ramzan, Yusuf Kiberu, Nikesh Jathanna, Meryem Jabrane, Vicente Grau, Shahnaz Jamil-Copley, Richard H. Clayton, Chen, Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.14702)  

**Abstract**: Accurate segmentation of myocardial scar from late gadolinium enhanced (LGE) cardiac MRI is essential for evaluating tissue viability, yet remains challenging due to variable contrast and imaging artifacts. Electrocardiogram (ECG) signals provide complementary physiological information, as conduction abnormalities can help localize or suggest scarred myocardial regions. In this work, we propose a novel multimodal framework that integrates ECG-derived electrophysiological information with anatomical priors from the AHA-17 atlas for physiologically consistent LGE-based scar segmentation. As ECGs and LGE-MRIs are not acquired simultaneously, we introduce a Temporal Aware Feature Fusion (TAFF) mechanism that dynamically weights and fuses features based on their acquisition time difference. Our method was evaluated on a clinical dataset and achieved substantial gains over the state-of-the-art image-only baseline (nnU-Net), increasing the average Dice score for scars from 0.6149 to 0.8463 and achieving high performance in both precision (0.9115) and sensitivity (0.9043). These results show that integrating physiological and anatomical knowledge allows the model to "see beyond the image", setting a new direction for robust and physiologically grounded cardiac scar segmentation. 

**Abstract (ZH)**: 从延迟钆增强心脏MRI中基于生理和解剖知识的心肌疤痕准确分割：一种整合心电图信息和AHA-17解剖先验的时空意识特征融合框架 

---
# Adapformer: Adaptive Channel Management for Multivariate Time Series Forecasting 

**Title (ZH)**: Adapformer: 自适应通道管理多变量时间序列预测 

**Authors**: Yuchen Luo, Xinyu Li, Liuhua Peng, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2511.14632)  

**Abstract**: In multivariate time series forecasting (MTSF), accurately modeling the intricate dependencies among multiple variables remains a significant challenge due to the inherent limitations of traditional approaches. Most existing models adopt either \textbf{channel-independent} (CI) or \textbf{channel-dependent} (CD) strategies, each presenting distinct drawbacks. CI methods fail to leverage the potential insights from inter-channel interactions, resulting in models that may not fully exploit the underlying statistical dependencies present in the data. Conversely, CD approaches often incorporate too much extraneous information, risking model overfitting and predictive inefficiency. To address these issues, we introduce the Adaptive Forecasting Transformer (\textbf{Adapformer}), an advanced Transformer-based framework that merges the benefits of CI and CD methodologies through effective channel management. The core of Adapformer lies in its dual-stage encoder-decoder architecture, which includes the \textbf{A}daptive \textbf{C}hannel \textbf{E}nhancer (\textbf{ACE}) for enriching embedding processes and the \textbf{A}daptive \textbf{C}hannel \textbf{F}orecaster (\textbf{ACF}) for refining the predictions. ACE enhances token representations by selectively incorporating essential dependencies, while ACF streamlines the decoding process by focusing on the most relevant covariates, substantially reducing noise and redundancy. Our rigorous testing on diverse datasets shows that Adapformer achieves superior performance over existing models, enhancing both predictive accuracy and computational efficiency, thus making it state-of-the-art in MTSF. 

**Abstract (ZH)**: 多变量时间序列预测中的自适应forecasting变压器（Adapformer）：通过有效的通道管理融合独立通道和依赖通道方法 

---
# Expert-Guided POMDP Learning for Data-Efficient Modeling in Healthcare 

**Title (ZH)**: 专家引导的POMDP学习在医疗健康领域的数据高效建模 

**Authors**: Marco Locatelli, Arjen Hommersom, Roberto Clemens Cerioli, Daniela Besozzi, Fabio Stella  

**Link**: [PDF](https://arxiv.org/pdf/2511.14619)  

**Abstract**: Learning the parameters of Partially Observable Markov Decision Processes (POMDPs) from limited data is a significant challenge. We introduce the Fuzzy MAP EM algorithm, a novel approach that incorporates expert knowledge into the parameter estimation process by enriching the Expectation Maximization (EM) framework with fuzzy pseudo-counts derived from an expert-defined fuzzy model. This integration naturally reformulates the problem as a Maximum A Posteriori (MAP) estimation, effectively guiding learning in environments with limited data. In synthetic medical simulations, our method consistently outperforms the standard EM algorithm under both low-data and high-noise conditions. Furthermore, a case study on Myasthenia Gravis illustrates the ability of the Fuzzy MAP EM algorithm to recover a clinically coherent POMDP, demonstrating its potential as a practical tool for data-efficient modeling in healthcare. 

**Abstract (ZH)**: 从有限数据中学习部分可观测马尔可夫决策过程（POMDP）的参数是一个显著的挑战。我们提出了Fuzzy MAP EM算法，这是一种通过将模糊伪计数集成到由专家定义的模糊模型中丰富期望最大化（EM）框架中的方法，从而将专家知识引入参数估计过程的新方法。这种集成自然地将问题重新表述为最大后验（MAP）估计，有效地在数据有限的环境中指导学习。在合成的医疗模拟中，我们的方法在低数据和高噪声条件下均优于标准EM算法。此外，肌无力病例研究说明了Fuzzy MAP EM算法能够恢复出临床一致的POMDP的能力，展示了其在医疗保健中进行高效数据建模的潜在应用价值。 

---
# A Method for Characterizing Disease Progression from Acute Kidney Injury to Chronic Kidney Disease 

**Title (ZH)**: 一种从急性肾损伤到慢性肾病的疾病进展表征方法 

**Authors**: Yilu Fang, Jordan G. Nestor, Casey N. Ta, Jerard Z. Kneifati-Hayek, Chunhua Weng  

**Link**: [PDF](https://arxiv.org/pdf/2511.14603)  

**Abstract**: Patients with acute kidney injury (AKI) are at high risk of developing chronic kidney disease (CKD), but identifying those at greatest risk remains challenging. We used electronic health record (EHR) data to dynamically track AKI patients' clinical evolution and characterize AKI-to-CKD progression. Post-AKI clinical states were identified by clustering patient vectors derived from longitudinal medical codes and creatinine measurements. Transition probabilities between states and progression to CKD were estimated using multi-state modeling. After identifying common post-AKI trajectories, CKD risk factors in AKI subpopulations were identified through survival analysis. Of 20,699 patients with AKI at admission, 3,491 (17%) developed CKD. We identified fifteen distinct post-AKI states, each with different probabilities of CKD development. Most patients (75%, n=15,607) remained in a single state or made only one transition during the study period. Both established (e.g., AKI severity, diabetes, hypertension, heart failure, liver disease) and novel CKD risk factors, with their impact varying across these clinical states. This study demonstrates a data-driven approach for identifying high-risk AKI patients, supporting the development of decision-support tools for early CKD detection and intervention. 

**Abstract (ZH)**: 急性肾损伤（AKI）患者发展为慢性肾病（CKD）的风险较高，但识别高风险个体仍具挑战性。通过电子健康记录（EHR）数据动态追踪AKI患者的临床演变，研究AKI向CKD的进展。 

---
# MRI Embeddings Complement Clinical Predictors for Cognitive Decline Modeling in Alzheimer's Disease Cohorts 

**Title (ZH)**: MRI嵌入补充临床预测因子以建模阿尔茨海默病队列的认知衰退 

**Authors**: Nathaniel Putera, Daniel Vilet Rodríguez, Noah Videcrantz, Julia Machnio, Mostafa Mehdipour Ghazi  

**Link**: [PDF](https://arxiv.org/pdf/2511.14601)  

**Abstract**: Accurate modeling of cognitive decline in Alzheimer's disease is essential for early stratification and personalized management. While tabular predictors provide robust markers of global risk, their ability to capture subtle brain changes remains limited. In this study, we evaluate the predictive contributions of tabular and imaging-based representations, with a focus on transformer-derived Magnetic Resonance Imaging (MRI) embeddings. We introduce a trajectory-aware labeling strategy based on Dynamic Time Warping clustering to capture heterogeneous patterns of cognitive change, and train a 3D Vision Transformer (ViT) via unsupervised reconstruction on harmonized and augmented MRI data to obtain anatomy-preserving embeddings without progression labels. The pretrained encoder embeddings are subsequently assessed using both traditional machine learning classifiers and deep learning heads, and compared against tabular representations and convolutional network baselines. Results highlight complementary strengths across modalities. Clinical and volumetric features achieved the highest AUCs of around 0.70 for predicting mild and severe progression, underscoring their utility in capturing global decline trajectories. In contrast, MRI embeddings from the ViT model were most effective in distinguishing cognitively stable individuals with an AUC of 0.71. However, all approaches struggled in the heterogeneous moderate group. These findings indicate that clinical features excel in identifying high-risk extremes, whereas transformer-based MRI embeddings are more sensitive to subtle markers of stability, motivating multimodal fusion strategies for AD progression modeling. 

**Abstract (ZH)**: 准确建模阿尔茨海默病的认知衰退对于早期分层和个性化管理至关重要。虽然表格式预测因子提供了全球风险的稳健标志，但它们在捕捉细微的大脑变化方面仍有限制。在本研究中，我们评估了表格式表示和基于成像的表示的预测贡献，重点关注基于变压器提取的磁共振成像（MRI）嵌入。我们引入了一种基于动态时间规整聚类的轨迹感知标签策略，以捕获认知变化的异质模式，并通过无监督重建对协调和扩增的MRI数据进行3D视觉变换器（ViT）训练，以获得保留解剖结构的嵌入，而无需进展标签。预训练的编码嵌入随后使用传统机器学习分类器和深度学习头部进行评估，并与表格式表示和卷积网络基线进行比较。结果表明，不同模态具有互补的优势。临床和体积特征在预测轻度和重度进展方面获得了约0.70的最高AUC值，突显了它们在捕捉全局衰退轨迹方面的用途。相比之下，ViT模型的MRI嵌入在区分认知稳定个体方面最有效，AUC值为0.71。然而，所有方法在异质中度组中表现均不佳。这些发现表明，临床特征在识别高风险极端方面表现优秀，而基于变压器的MRI嵌入对细微的稳定性标志更为敏感，这促使了AD进展建模中的多模态融合策略。 

---
# Biased Minds Meet Biased AI: How Class Imbalance Shapes Appropriate Reliance and Interacts with Human Base Rate Neglect 

**Title (ZH)**: 有偏的思维遇到有偏的AI：类别不平衡如何塑造适当的依赖并与其交互作用影响人类的基本概率忽视 

**Authors**: Nick von Felten, Johannes Schöning, Klaus Opwis, Nicolas Scharowksi  

**Link**: [PDF](https://arxiv.org/pdf/2511.14591)  

**Abstract**: Humans increasingly interact with artificial intelligence (AI) in decision-making. However, both AI and humans are prone to biases. While AI and human biases have been studied extensively in isolation, this paper examines their complex interaction. Specifically, we examined how class imbalance as an AI bias affects people's ability to appropriately rely on an AI-based decision-support system, and how it interacts with base rate neglect as a human bias. In a within-subject online study (N= 46), participants classified three diseases using an AI-based decision-support system trained on either a balanced or unbalanced dataset. We found that class imbalance disrupted participants' calibration of AI reliance. Moreover, we observed mutually reinforcing effects between class imbalance and base rate neglect, offering evidence of a compound human-AI bias. Based on these findings, we advocate for an interactionist perspective and further research into the mutually reinforcing effects of biases in human-AI interaction. 

**Abstract (ZH)**: 人类越来越多地在决策中与人工智能（AI）互动，然而AI和人类都容易受到偏见的影响。尽管AI偏见和人类偏见在孤立的情况下已经得到了广泛的研究所，本文探讨了它们的复杂互动。具体来说，本文研究了作为AI偏见的类别不平衡如何影响人们适当依赖基于AI的决策支持系统的能力，以及它与人类偏见中的基本率忽视之间的互动。在一个被试内在线研究中（N=46），参与者使用训练于平衡或不平衡数据集的基于AI的决策支持系统对三种疾病进行分类。我们发现类别不平衡干扰了参与者对AI依赖性的校准。此外，我们观察到类别不平衡和基本率忽视之间的相互加强效应，提供了人类-AI偏见复合效应的证据。基于这些发现，本文倡导交互主义视角，并进一步研究人类-AI互动中偏见的相互加强效应。 

---
# Deep Learning-Based Regional White Matter Hyperintensity Mapping as a Robust Biomarker for Alzheimer's Disease 

**Title (ZH)**: 基于深度学习的区域白质高信号图映射作为阿尔茨海默病的稳健生物标志物 

**Authors**: Julia Machnio, Mads Nielsen, Mostafa Mehdipour Ghazi  

**Link**: [PDF](https://arxiv.org/pdf/2511.14588)  

**Abstract**: White matter hyperintensities (WMH) are key imaging markers in cognitive aging, Alzheimer's disease (AD), and related dementias. Although automated methods for WMH segmentation have advanced, most provide only global lesion load and overlook their spatial distribution across distinct white matter regions. We propose a deep learning framework for robust WMH segmentation and localization, evaluated across public datasets and an independent Alzheimer's Disease Neuroimaging Initiative (ADNI) cohort. Our results show that the predicted lesion loads are in line with the reference WMH estimates, confirming the robustness to variations in lesion load, acquisition, and demographics. Beyond accurate segmentation, we quantify WMH load within anatomically defined regions and combine these measures with brain structure volumes to assess diagnostic value. Regional WMH volumes consistently outperform global lesion burden for disease classification, and integration with brain atrophy metrics further improves performance, reaching area under the curve (AUC) values up to 0.97. Several spatially distinct regions, particularly within anterior white matter tracts, are reproducibly associated with diagnostic status, indicating localized vulnerability in AD. These results highlight the added value of regional WMH quantification. Incorporating localized lesion metrics alongside atrophy markers may enhance early diagnosis and stratification in neurodegenerative disorders. 

**Abstract (ZH)**: 白质高信号的深度学习分割与定位：跨公共数据集和独立阿尔茨海默病神经影像学倡议(ADNI)队列的评估 

---
# Examining the Metrics for Document-Level Claim Extraction in Czech and Slovak 

**Title (ZH)**: 研究捷克语和斯洛伐克语文档级声明提取的指标 

**Authors**: Lucia Makaiová, Martin Fajčík, Antonín Jarolím  

**Link**: [PDF](https://arxiv.org/pdf/2511.14566)  

**Abstract**: Document-level claim extraction remains an open challenge in the field of fact-checking, and subsequently, methods for evaluating extracted claims have received limited attention. In this work, we explore approaches to aligning two sets of claims pertaining to the same source document and computing their similarity through an alignment score. We investigate techniques to identify the best possible alignment and evaluation method between claim sets, with the aim of providing a reliable evaluation framework. Our approach enables comparison between model-extracted and human-annotated claim sets, serving as a metric for assessing the extraction performance of models and also as a possible measure of inter-annotator agreement. We conduct experiments on newly collected dataset-claims extracted from comments under Czech and Slovak news articles-domains that pose additional challenges due to the informal language, strong local context, and subtleties of these closely related languages. The results draw attention to the limitations of current evaluation approaches when applied to document-level claim extraction and highlight the need for more advanced methods-ones able to correctly capture semantic similarity and evaluate essential claim properties such as atomicity, checkworthiness, and decontextualization. 

**Abstract (ZH)**: 文档级别声明抽取仍然是事实核查领域的开放挑战，随之而来的抽取声明的评估方法也受到了有限的关注。在本文中，我们探索将两组关于同一源文档的声明进行对齐并通过对齐分数计算它们相似性的方法。我们研究了识别声明集之间最佳对齐和评估方法的技术，旨在提供一个可靠的评估框架。我们的方法可以比较模型抽取和人工标注的声明集，作为评估模型抽取性能的指标，也可能作为不同标注者之间一致性的一种度量。我们在新收集的数据集上进行实验，该数据集来源于捷克和 Slovak 新闻文章评论下的声明，由于这些语言的非正式语言、强烈的地域背景以及语言细微差异，这些领域带来额外的挑战。实验结果强调了当前评估方法在应用于文档级别声明抽取时的局限性，并突显了需要更先进方法的重要性——这些方法能够正确捕捉语义相似性并评估诸如原子性、查证价值性和脱域性等核心声明属性。 

---
# DecNefLab: A Modular and Interpretable Simulation Framework for Decoded Neurofeedback 

**Title (ZH)**: DecNefLab: 一个模块化和可解释的解码神经反馈模拟框架 

**Authors**: Alexander Olza, Roberto Santana, David Soto  

**Link**: [PDF](https://arxiv.org/pdf/2511.14555)  

**Abstract**: Decoded Neurofeedback (DecNef) is a flourishing non-invasive approach to brain modulation with wide-ranging applications in neuromedicine and cognitive neuroscience. However, progress in DecNef research remains constrained by subject-dependent learning variability, reliance on indirect measures to quantify progress, and the high cost and time demands of experimentation.
We present DecNefLab, a modular and interpretable simulation framework that formalizes DecNef as a machine learning problem. Beyond providing a virtual laboratory, DecNefLab enables researchers to model, analyze and understand neurofeedback dynamics. Using latent variable generative models as simulated participants, DecNefLab allows direct observation of internal cognitive states and systematic evaluation of how different protocol designs and subject characteristics influence learning.
We demonstrate how this approach can (i) reproduce empirical phenomena of DecNef learning, (ii) identify conditions under which DecNef feedback fails to induce learning, and (iii) guide the design of more robust and reliable DecNef protocols in silico before human implementation.
In summary, DecNefLab bridges computational modeling and cognitive neuroscience, offering a principled foundation for methodological innovation, robust protocol design, and ultimately, a deeper understanding of DecNef-based brain modulation. 

**Abstract (ZH)**: DecNefLab：一种模块化可解释的模拟框架，用于脑调控的机器学习问题建模与分析 

---
# MissHDD: Hybrid Deterministic Diffusion for Hetrogeneous Incomplete Data Imputation 

**Title (ZH)**: MissHDD: 混合确定性扩散异构不完全数据插补 

**Authors**: Youran Zhou, Mohamed Reda Bouadjenek, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2511.14543)  

**Abstract**: Incomplete data are common in real-world tabular applications, where numerical, categorical, and discrete attributes coexist within a single dataset. This heterogeneous structure presents significant challenges for existing diffusion-based imputation models, which typically assume a homogeneous feature space and rely on stochastic denoising trajectories. Such assumptions make it difficult to maintain conditional consistency, and they often lead to information collapse for categorical variables or instability when numerical variables require deterministic updates. These limitations indicate that a single diffusion process is insufficient for mixed-type tabular imputation.
We propose a hybrid deterministic diffusion framework that separates heterogeneous features into two complementary generative channels. A continuous DDIM-based channel provides efficient and stable deterministic denoising for numerical variables, while a discrete latent-path diffusion channel, inspired by loopholing-based discrete diffusion, models categorical and discrete features without leaving their valid sample manifolds. The two channels are trained under a unified conditional imputation objective, enabling coherent reconstruction of mixed-type incomplete data.
Extensive experiments on multiple real-world datasets show that the proposed framework achieves higher imputation accuracy, more stable sampling trajectories, and improved robustness across MCAR, MAR, and MNAR settings compared with existing diffusion-based and classical methods. These results demonstrate the importance of structure-aware diffusion processes for advancing deep learning approaches to incomplete tabular data. 

**Abstract (ZH)**: 异构特征共存的混合类型表征不完备数据的混合确定性扩散框架 

---
# IMSE: Efficient U-Net-based Speech Enhancement using Inception Depthwise Convolution and Amplitude-Aware Linear Attention 

**Title (ZH)**: IMSE：基于Inception Depthwise卷积和幅度意识线性注意力的高效U-Net语音增强方法 

**Authors**: Xinxin Tang, Bin Qin, Yufang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.14515)  

**Abstract**: Achieving a balance between lightweight design and high performance remains a significant challenge for speech enhancement (SE) tasks on resource-constrained devices. Existing state-of-the-art methods, such as MUSE, have established a strong baseline with only 0.51M parameters by introducing a Multi-path Enhanced Taylor (MET) transformer and Deformable Embedding (DE). However, an in-depth analysis reveals that MUSE still suffers from efficiency bottlenecks: the MET module relies on a complex "approximate-compensate" mechanism to mitigate the limitations of Taylor-expansion-based attention, while the offset calculation for deformable embedding introduces additional computational burden. This paper proposes IMSE, a systematically optimized and ultra-lightweight network. We introduce two core innovations: 1) Replacing the MET module with Amplitude-Aware Linear Attention (MALA). MALA fundamentally rectifies the "amplitude-ignoring" problem in linear attention by explicitly preserving the norm information of query vectors in the attention calculation, achieving efficient global modeling without an auxiliary compensation branch. 2) Replacing the DE module with Inception Depthwise Convolution (IDConv). IDConv borrows the Inception concept, decomposing large-kernel operations into efficient parallel branches (square, horizontal, and vertical strips), thereby capturing spectrogram features with extremely low parameter redundancy. Extensive experiments on the VoiceBank+DEMAND dataset demonstrate that, compared to the MUSE baseline, IMSE significantly reduces the parameter count by 16.8\% (from 0.513M to 0.427M) while achieving competitive performance comparable to the state-of-the-art on the PESQ metric (3.373). This study sets a new benchmark for the trade-off between model size and speech quality in ultra-lightweight speech enhancement. 

**Abstract (ZH)**: 实现轻量化设计与高性能之间的平衡仍然是资源受限设备上语音增强任务的一项重大挑战。现有最先进的方法，如MUSE，通过引入Multi-path Enhanced Taylor（MET）变压器和变形嵌入（DE），建立了仅有513K参数的强大基线。然而，深入分析表明，MUSE仍然存在效率瓶颈：MET模块依赖一种复杂的“近似补偿”机制来缓解基于泰勒展开的注意力限制，而变形嵌入的偏移计算引入了额外的计算负担。本文提出IMSE，一种系统优化和超轻量级网络。我们引入了两项核心创新：1）用幅度感知线性注意力（MALA）替代MET模块，从根本上纠正了线性注意力中的“幅度忽略”问题，在注意力计算中明确保留了查询向量的范数信息，从而实现高效的全局建模而不需辅助补偿分支；2）用Inception深度卷积（IDConv）替代DE模块。IDConv借鉴Inception概念，将大内核操作分解为高效的并行分支（平方、水平和垂直条带），从而以极低的参数冗余捕捉频谱图特征。在VoiceBank+DEMAND数据集上的广泛实验表明，与MUSE基线相比，IMSE在参数量显著减少16.8%（从513K降至427K）的同时，在PESQ指标上实现竞争力的表现。本研究为超轻量级语音增强中的模型大小与语音质量之间的权衡设立了新标准。 

---
# Towards Stable and Structured Time Series Generation with Perturbation-Aware Flow Matching 

**Title (ZH)**: 面向扰动感知流匹配的稳定结构时间序列生成 

**Authors**: Jintao Zhang, Mingyue Cheng, Zirui Liu, Xianquan Wang, Yitong Zhou, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.14488)  

**Abstract**: Time series generation is critical for a wide range of applications, which greatly supports downstream analytical and decision-making tasks. However, the inherent temporal heterogeneous induced by localized perturbations present significant challenges for generating structurally consistent time series. While flow matching provides a promising paradigm by modeling temporal dynamics through trajectory-level supervision, it fails to adequately capture abrupt transitions in perturbed time series, as the use of globally shared parameters constrains the velocity field to a unified representation. To address these limitations, we introduce \textbf{PAFM}, a \textbf{P}erturbation-\textbf{A}ware \textbf{F}low \textbf{M}atching framework that models perturbed trajectories to ensure stable and structurally consistent time series generation. The framework incorporates perturbation-guided training to simulate localized disturbances and leverages a dual-path velocity field to capture trajectory deviations under perturbation, enabling refined modeling of perturbed behavior to enhance the structural coherence. In order to further improve sensitivity to trajectory perturbations while enhancing expressiveness, a mixture-of-experts decoder with flow routing dynamically allocates modeling capacity in response to different trajectory dynamics. Extensive experiments on both unconditional and conditional generation tasks demonstrate that PAFM consistently outperforms strong baselines. Code is available at this https URL. 

**Abstract (ZH)**: 基于扰动感知的流匹配时间序列生成框架 

---
# Agentic AI Systems in Electrical Power Systems Engineering: Current State-of-the-Art and Challenges 

**Title (ZH)**: 电气电力系统工程中代理型AI系统：现状与挑战 

**Authors**: Soham Ghosh, Gaurav Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2511.14478)  

**Abstract**: Agentic AI systems have recently emerged as a critical and transformative approach in artificial intelligence, offering capabilities that extend far beyond traditional AI agents and contemporary generative AI models. This rapid evolution necessitates a clear conceptual and taxonomical understanding to differentiate this new paradigm. Our paper addresses this gap by providing a comprehensive review that establishes a precise definition and taxonomy for "agentic AI," with the aim of distinguishing it from previous AI paradigms. The concepts are gradually introduced, starting with a highlight of its diverse applications across the broader field of engineering. The paper then presents four detailed, state-of-the-art use case applications specifically within electrical engineering. These case studies demonstrate practical impact, ranging from an advanced agentic framework for streamlining complex power system studies and benchmarking to a novel system developed for survival analysis of dynamic pricing strategies in battery swapping stations. Finally, to ensure robust deployment, the paper provides detailed failure mode investigations. From these findings, we derive actionable recommendations for the design and implementation of safe, reliable, and accountable agentic AI systems, offering a critical resource for researchers and practitioners. 

**Abstract (ZH)**: 代理型AI系统 recently emerged as一个关键且 transformative的approach在人工智能中，提供了超越传统AI代理和当代生成型AI模型的能力。这种迅速演变需要一个清晰的概念和分类理解以区分这一新范式。本文通过提供一个全面的回顾，为“代理型AI”建立了一个精确的定义和分类体系，旨在将其与之前的AI范式区分开来。本文逐步介绍了其在更广泛的工程领域中的多样应用，然后提出了四个具体且最新的案例研究，这些案例研究集中在电气工程领域。这些案例研究展示了代理型AI的实际影响，从复杂电力系统研究和基准测试的先进代理框架到电池更换站动态定价策略生存分析的新系统。最后，为了确保部署的稳健性，本文提供了详细的故障模式调查。从这些发现中，我们推导出有关设计和实现安全、可靠和负责任的代理型AI系统的可操作建议，为研究人员和实践者提供关键资源。 

---
# nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers 

**Title (ZH)**: nnterp：Transformer机理可解释性的标准化接口 

**Authors**: Clément Dumas  

**Link**: [PDF](https://arxiv.org/pdf/2511.14465)  

**Abstract**: Mechanistic interpretability research requires reliable tools for analyzing transformer internals across diverse architectures. Current approaches face a fundamental tradeoff: custom implementations like TransformerLens ensure consistent interfaces but require coding a manual adaptation for each architecture, introducing numerical mismatch with the original models, while direct HuggingFace access through NNsight preserves exact behavior but lacks standardization across models. To bridge this gap, we develop nnterp, a lightweight wrapper around NNsight that provides a unified interface for transformer analysis while preserving original HuggingFace implementations. Through automatic module renaming and comprehensive validation testing, nnterp enables researchers to write intervention code once and deploy it across 50+ model variants spanning 16 architecture families. The library includes built-in implementations of common interpretability methods (logit lens, patchscope, activation steering) and provides direct access to attention probabilities for models that support it. By packaging validation tests with the library, researchers can verify compatibility with custom models locally. nnterp bridges the gap between correctness and usability in mechanistic interpretability tooling. 

**Abstract (ZH)**: 机制可解释性研究需要可靠的工具来分析跨多种架构的变换器内部。当前的方法面临一个基本的权衡：自定义实现如TransformerLens确保一致性接口，但需要为每个架构手动编撰适应代码，引入与原始模型的数值差异，而直接通过NNsight访问HuggingFace保留了精确行为，但在模型之间缺乏标准化。为弥合这一差距，我们开发了nnterp，这是一个围绕NNsight的轻量级包装器，提供了一致的接口用于变换器分析，同时保留原始的HuggingFace实现。通过自动模块重命名和全面的验证测试，nnterp使研究人员能够编写一次干预代码并在50多种模型变体（横跨16个架构家族）中部署。库中包括常见可解释性方法（logit lens、patchscope、激活引导）的内置实现，并为支持的模型直接提供注意力概率访问。通过将验证测试与库打包，研究人员可以本地验证自定义模型的兼容性。nnterp在机制可解释性工具的正确性和可用性之间架起桥梁。 

---
# Effective Diversification of Multi-Carousel Book Recommendation 

**Title (ZH)**: 多轮盘书推荐的有效多样化 

**Authors**: Daniël Wilten, Gideon Maillette de Buy Wenniger, Arjen Hommersom, Paul Lucassen, Emiel Poortman  

**Link**: [PDF](https://arxiv.org/pdf/2511.14461)  

**Abstract**: Using multiple carousels, lists that wrap around and can be scrolled, is the basis for offering content in most contemporary movie streaming platforms. Carousels allow for highlighting different aspects of users' taste, that fall in categories such as genres and authors. However, while carousels offer structure and greater ease of navigation, they alone do not increase diversity in recommendations, while this is essential to keep users engaged. In this work we propose several approaches to effectively increase item diversity within the domain of book recommendations, on top of a collaborative filtering algorithm. These approaches are intended to improve book recommendations in the web catalogs of public libraries. Furthermore, we introduce metrics to evaluate the resulting strategies, and show that the proposed system finds a suitable balance between accuracy and beyond-accuracy aspects. 

**Abstract (ZH)**: 使用多个轮播列表是大多数当代电影流媒体平台提供内容的基础。轮播列表可以滚动显示并围绕内容卷绕，有助于突出用户口味的不同方面，如类型和作者。然而，虽然轮播列表提供了结构并增强了导航的便捷性，但它们本身并不能增加推荐的多样性，而多样性对于保持用户参与是必不可少的。在本工作中，我们提出了一种在基于协同过滤算法的基础上有效增加书籍推荐多样性的多种方法。这些方法旨在改进公共图书馆网络目录中的书籍推荐。此外，我们引入了评估这些策略的指标，并展示了所提系统在准确性和超越准确性方面找到了合适的平衡。 

---
# Analyzing the Impact of Participant Failures in Cross-Silo Federated Learning 

**Title (ZH)**: 分析参与者故障对跨域联邦学习的影响 

**Authors**: Fabian Stricker, David Bermbach, Christian Zirpins  

**Link**: [PDF](https://arxiv.org/pdf/2511.14456)  

**Abstract**: Federated learning (FL) is a new paradigm for training machine learning (ML) models without sharing data. While applying FL in cross-silo scenarios, where organizations collaborate, it is necessary that the FL system is reliable; however, participants can fail due to various reasons (e.g., communication issues or misconfigurations). In order to provide a reliable system, it is necessary to analyze the impact of participant failures. While this problem received attention in cross-device FL where mobile devices with limited resources participate, there is comparatively little research in cross-silo FL.
Therefore, we conduct an extensive study for analyzing the impact of participant failures on the model quality in the context of inter-organizational cross-silo FL with few participants. In our study, we focus on analyzing generally influential factors such as the impact of the timing and the data as well as the impact on the evaluation, which is important for deciding, if the model should be deployed. We show that under high skews the evaluation is optimistic and hides the real impact. Furthermore, we demonstrate that the timing impacts the quality of the trained model. Our results offer insights for researchers and software architects aiming to build robust FL systems. 

**Abstract (ZH)**: 联邦学习（FL）是一种无需共享数据即训练机器学习（ML）模型的新范式。在跨机构场景下应用联邦学习时，为了确保FL系统的可靠性，参与者可能因各种原因（如通信问题或配置错误）而失败。为了提供一个可靠的系统，有必要分析参与者失败的影响。虽然在跨设备联邦学习（涉及资源有限的移动设备）中，已经对此问题给予了较多关注，但在跨机构联邦学习（涉及较少参与者）中，对此的研究相对较少。因此，我们开展了广泛的研究，分析参与者失败对跨机构联邦学习中少参与者场景下模型质量的影响。在研究中，我们专注于分析一般影响因素，如时间影响和数据影响，以及评估影响，这对于决定模型是否应部署至关重要。我们发现，在高度偏斜的情况下，评估是乐观的，掩盖了真实影响。此外，我们展示了时间影响训练模型的质量。我们的结果为希望构建稳健联邦学习系统的研究人员和软件架构师提供了见解。 

---
# Hybrid Modeling of Photoplethysmography for Non-invasive Monitoring of Cardiovascular Parameters 

**Title (ZH)**: 非侵入性心血管参数监测的光电容积脉搏波混合建模 

**Authors**: Emanuele Palumbo, Sorawit Saengkyongam, Maria R. Cervera, Jens Behrmann, Andrew C. Miller, Guillermo Sapiro, Christina Heinze-Deml, Antoine Wehenkel  

**Link**: [PDF](https://arxiv.org/pdf/2511.14452)  

**Abstract**: Continuous cardiovascular monitoring can play a key role in precision health. However, some fundamental cardiac biomarkers of interest, including stroke volume and cardiac output, require invasive measurements, e.g., arterial pressure waveforms (APW). As a non-invasive alternative, photoplethysmography (PPG) measurements are routinely collected in hospital settings. Unfortunately, the prediction of key cardiac biomarkers from PPG instead of APW remains an open challenge, further complicated by the scarcity of annotated PPG measurements. As a solution, we propose a hybrid approach that uses hemodynamic simulations and unlabeled clinical data to estimate cardiovascular biomarkers directly from PPG signals. Our hybrid model combines a conditional variational autoencoder trained on paired PPG-APW data with a conditional density estimator of cardiac biomarkers trained on labeled simulated APW segments. As a key result, our experiments demonstrate that the proposed approach can detect fluctuations of cardiac output and stroke volume and outperform a supervised baseline in monitoring temporal changes in these biomarkers. 

**Abstract (ZH)**: 连续心血管监测在精准健康中扮演着关键角色。然而，一些感兴趣的基金心脏病理标记物，包括搏出量和心输出量，需要通过有创测量，如动脉压力波形（APW）进行测量。作为无创替代方案，脉搏血氧图（PPG）测量在医院环境中常规收集。不幸的是，直接从PPG而非APW预测关键心脏标记物仍然是一个开放性的挑战，并且由标注的PPG测量数据稀缺性进一步复杂化。作为解决方案，我们提出了一种结合血流动力学仿真和未标注临床数据的混合方法，用于从PPG信号直接估计心血管标记物。我们的混合模型结合了在配对PPG-APW数据上训练的条件变分自编码器和在标注仿真APW片段上训练的心脏标记物条件密度估计器。实验结果表明，所提出的方法能够检测搏出量和心输出量的变化，并在监测这些标记物的时空变化时优于监督基线方法。 

---
# MiAD: Mirage Atom Diffusion for De Novo Crystal Generation 

**Title (ZH)**: MiAD: 幻象原子扩散方法在从头生成晶体中的应用 

**Authors**: Andrey Okhotin, Maksim Nakhodnov, Nikita Kazeev, Andrey E Ustyuzhanin, Dmitry Vetrov  

**Link**: [PDF](https://arxiv.org/pdf/2511.14426)  

**Abstract**: In recent years, diffusion-based models have demonstrated exceptional performance in searching for simultaneously stable, unique, and novel (S.U.N.) crystalline materials. However, most of these models don't have the ability to change the number of atoms in the crystal during the generation process, which limits the variability of model sampling trajectories. In this paper, we demonstrate the severity of this restriction and introduce a simple yet powerful technique, mirage infusion, which enables diffusion models to change the state of the atoms that make up the crystal from existent to non-existent (mirage) and vice versa. We show that this technique improves model quality by up to $\times2.5$ compared to the same model without this modification. The resulting model, Mirage Atom Diffusion (MiAD), is an equivariant joint diffusion model for de novo crystal generation that is capable of altering the number of atoms during the generation process. MiAD achieves an $8.2\%$ S.U.N. rate on the MP-20 dataset, which substantially exceeds existing state-of-the-art approaches. The source code can be found at \href{this https URL}{\texttt{this http URL}}. 

**Abstract (ZH)**: 近年来，基于扩散的模型在寻找同时稳定、唯一和新颖（S.U.N.）的晶体材料方面展现了卓越性能。然而，大多数这些模型在生成过程中不具备改变晶体中原子数量的能力，这限制了模型采样轨迹的多样性。本文展示了这一限制的严重性，并介绍了一种简单而强大的技术——幻象注入，该技术使扩散模型能够改变构成晶体的原子从存在到不存在（幻象）及其相反状态。我们表明，这种技术将模型质量改进了高达2.5倍。由此产生的模型Mirage Atom Diffusion (MiAD) 是一个用于从头生成晶体的不变联合扩散模型，能够在生成过程中改变原子的数量。MiAD 在MP-20数据集上的S.U.N. 速率达到8.2%，远超现有最先进的方法。源代码可在 \href{this https URL}{\texttt{这个链接}} 获取。 

---
# Sigil: Server-Enforced Watermarking in U-Shaped Split Federated Learning via Gradient Injection 

**Title (ZH)**: Sigil: 在U型拆分联邦学习中通过梯度注入实现服务器强制水印标记 

**Authors**: Zhengchunmin Dai, Jiaxiong Tang, Peng Sun, Honglong Chen, Liantao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.14422)  

**Abstract**: In decentralized machine learning paradigms such as Split Federated Learning (SFL) and its variant U-shaped SFL, the server's capabilities are severely restricted. Although this enhances client-side privacy, it also leaves the server highly vulnerable to model theft by malicious clients. Ensuring intellectual property protection for such capability-limited servers presents a dual challenge: watermarking schemes that depend on client cooperation are unreliable in adversarial settings, whereas traditional server-side watermarking schemes are technically infeasible because the server lacks access to critical elements such as model parameters or labels.
To address this challenge, this paper proposes Sigil, a mandatory watermarking framework designed specifically for capability-limited servers. Sigil defines the watermark as a statistical constraint on the server-visible activation space and embeds the watermark into the client model via gradient injection, without requiring any knowledge of the data. Besides, we design an adaptive gradient clipping mechanism to ensure that our watermarking process remains both mandatory and stealthy, effectively countering existing gradient anomaly detection methods and a specifically designed adaptive subspace removal attack. Extensive experiments on multiple datasets and models demonstrate Sigil's fidelity, robustness, and stealthiness. 

**Abstract (ZH)**: 在分布式机器学习范式如拆分联邦学习（SFL）及其变体U形SFL中，服务器的能力严重受限。尽管这提高了客户端的隐私性，但也使服务器极易受到恶意客户端的模型窃取攻击。为这类能力有限的服务器保护知识产权提出了双重挑战：依赖客户端合作的水印方案在对抗性环境中不可靠，而传统的服务器端水印方案因服务器缺乏对关键元素（如模型参数或标签）的访问而技术上不可行。

为此，本文提出Sigil，一种专为能力有限的服务器设计的强制水印框架。Sigil将水印定义为服务器可见激活空间上的统计约束，并通过梯度注入将水印嵌入客户端模型中，无需任何数据知识。此外，我们设计了一种自适应梯度裁剪机制，以确保我们的水印过程既具有强制性又具有隐匿性，有效地抵御了现有的梯度异常检测方法和专门设计的自适应子空间移除攻击。在多个数据集和模型上的广泛实验表明，Sigil具有高保真度、稳定性和隐匿性。 

---
# Clinically-Validated Innovative Mobile Application for Assessing Blinking and Eyelid Movements 

**Title (ZH)**: 临床验证的创新移动应用，用于评估眨眼和眼睑运动 

**Authors**: Gustavo Adolpho Bonesso, Carlos Marcelo Gurjão de Godoy, Tammy Hentona Osaki, Midori Hentona Osaki, Bárbara Moreira Ribeiro Trindade dos Santos, Regina Célia Coelho  

**Link**: [PDF](https://arxiv.org/pdf/2511.14361)  

**Abstract**: Blinking is a vital physiological process that protects and maintains the health of the ocular surface. Objective assessment of eyelid movements remains challenging due to the complexity, cost, and limited clinical applicability of existing tools. This study presents the clinical validation of Bapp (Blink Application), a mobile application developed using the Flutter framework and integrated with Google ML Kit for on-device, real-time analysis of eyelid movements. The validation occurred using 45 videos from real patients, whose blinks were manually annotated by ophthalmology specialists from the Paulista School of Medicine of the Federal University of Sao Paulo (EPM-UNIFESP) to serve as the ground truth. Bapp's performance was evaluated using standard metrics, including Precision, Recall, and F1-Score, with results demonstrating 98.4% precision, 96.9% recall, and an overall accuracy of 98.3%. These outcomes confirm the reliability of Bapp as a portable, accessible, and objective tool for monitoring both normal and abnormal eyelid movements. The application offers a promising alternative to traditional manual blink counting, supporting continuous ocular health monitoring and postoperative evaluation in clinical environments. 

**Abstract (ZH)**: 眨眼是保护和维持眼表健康的关键生理过程。由于现有工具的复杂性、成本及临床适用性限制，客观评估眼睑运动仍然具有挑战性。本研究介绍了使用 Flutter 框架开发并集成了 Google ML Kit 的移动应用 Bapp (Blink Application) 的临床验证，该应用实现了实时眼睑运动分析。验证使用了保罗里巴特医学院联邦大学圣保罗分校 (EPM-UNIFESP) 眼科学专家手动标注的 45 个患者视频作为ground truth。Bapp 的性能通过 Precision、Recall 和 F1-Score 等标准指标进行评估，结果显示其精确率为 98.4%、召回率为 96.9%、总体准确率为 98.3%。这些结果证明 Bapp 是一个便携、易获取且客观的眼睑运动监测工具。该应用为临床环境中正常和异常眨眼监测提供了有 promise 的替代方案，支持持续的眼球健康监测和术后评估。 

---
# H-LDM: Hierarchical Latent Diffusion Models for Controllable and Interpretable PCG Synthesis from Clinical Metadata 

**Title (ZH)**: H-LDM：分层隐含扩散模型在临床元数据指导下的心源性音频合成可控性和可解释性研究 

**Authors**: Chenyang Xu, Siming Li, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14312)  

**Abstract**: Phonocardiogram (PCG) analysis is vital for cardiovascular disease diagnosis, yet the scarcity of labeled pathological data hinders the capability of AI systems. To bridge this, we introduce H-LDM, a Hierarchical Latent Diffusion Model for generating clinically accurate and controllable PCG signals from structured metadata. Our approach features: (1) a multi-scale VAE that learns a physiologically-disentangled latent space, separating rhythm, heart sounds, and murmurs; (2) a hierarchical text-to-biosignal pipeline that leverages rich clinical metadata for fine-grained control over 17 distinct conditions; and (3) an interpretable diffusion process guided by a novel Medical Attention module. Experiments on the PhysioNet CirCor dataset demonstrate state-of-the-art performance, achieving a Fréchet Audio Distance of 9.7, a 92% attribute disentanglement score, and 87.1% clinical validity confirmed by cardiologists. Augmenting diagnostic models with our synthetic data improves the accuracy of rare disease classification by 11.3\%. H-LDM establishes a new direction for data augmentation in cardiac diagnostics, bridging data scarcity with interpretable clinical insights. 

**Abstract (ZH)**: 心脏听诊图（PCG）分析对于心血管疾病诊断至关重要，但由于标注的病理数据稀缺限制了AI系统的性能。为此，我们引入了H-LDM，一种层次潜在扩散模型，用于从结构化元数据生成临床准确且可控制的心脏听诊图信号。该方法的特点包括：（1）多尺度VAE学习生理脱混的潜在空间，分离心律、心音和杂音；（2）一种利用丰富临床元数据的层次文本到生物信号管道，实现对17种不同条件的精细控制；（3）一种由新型医疗注意模块引导的可解释扩散过程。在PhysioNet CirCor数据集上的实验展示了业界领先的表现，实现了9.7的Fréchet音频距离、92%的属性脱混分数，并且经过心脏病专家确认的临床有效性为87.1%。将我们的合成数据 aug 量到诊断模型中，可提高罕见疾病分类的准确率11.3%。H-LDM 为心脏诊断中的数据扩充建立了一个新的方向，解决了数据稀缺与可解释临床洞察之间的鸿沟。 

---
# SAM-Fed: SAM-Guided Federated Semi-Supervised Learning for Medical Image Segmentation 

**Title (ZH)**: SAM-Fed: SAM 引导的联邦半监督学习在医学图像分割中的应用 

**Authors**: Sahar Nasirihaghighi, Negin Ghamsarian, Yiping Li, Marcel Breeuwer, Raphael Sznitman, Klaus Schoeffmann  

**Link**: [PDF](https://arxiv.org/pdf/2511.14302)  

**Abstract**: Medical image segmentation is clinically important, yet data privacy and the cost of expert annotation limit the availability of labeled data. Federated semi-supervised learning (FSSL) offers a solution but faces two challenges: pseudo-label reliability depends on the strength of local models, and client devices often require compact or heterogeneous architectures due to limited computational resources. These constraints reduce the quality and stability of pseudo-labels, while large models, though more accurate, cannot be trained or used for routine inference on client devices. We propose SAM-Fed, a federated semi-supervised framework that leverages a high-capacity segmentation foundation model to guide lightweight clients during training. SAM-Fed combines dual knowledge distillation with an adaptive agreement mechanism to refine pixel-level supervision. Experiments on skin lesion and polyp segmentation across homogeneous and heterogeneous settings show that SAM-Fed consistently outperforms state-of-the-art FSSL methods. 

**Abstract (ZH)**: 医学图像分割在临床上非常重要，但数据隐私和专家标注的成本限制了标注数据的可用性。联邦半监督学习（FSSL）提供了一种解决方案，但面临两大挑战：伪标签的可靠性取决于局部模型的强度，而客户端设备由于计算资源有限，往往需要使用紧凑或异构的架构。这些约束降低了伪标签的质量和稳定性，而大型模型虽然更为准确，却无法在客户端设备上进行训练或常规推理。我们提出了一种名为SAM-Fed的联邦半监督框架，该框架利用高性能的分割基础模型在训练过程中引导轻量级客户端。SAM-Fed结合了双重知识蒸馏与自适应一致机制来精炼像素级监督。在一致性和异构设置下的皮肤病变和息肉分割实验中，SAM-Fed始终优于最先进的FSSL方法。 

---
# Weight Variance Amplifier Improves Accuracy in High-Sparsity One-Shot Pruning 

**Title (ZH)**: 权重方差放大器提高高稀疏度一次性剪枝的准确性 

**Authors**: Vincent-Daniel Yun, Junhyuk Jo, Sunwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.14282)  

**Abstract**: Deep neural networks achieve outstanding performance in visual recognition tasks, yet their large number of parameters makes them less practical for real-world applications. Recently, one-shot pruning has emerged as an effective strategy for reducing model size without additional training. However, models trained with standard objective functions often suffer a significant drop in accuracy after aggressive pruning. Some existing pruning-robust optimizers, such as SAM, and CrAM, mitigate this accuracy drop by guiding the model toward flatter regions of the parameter space, but they inevitably incur non-negligible additional computations. We propose a Variance Amplifying Regularizer (VAR) that deliberately increases the variance of model parameters during training. Our study reveals an intriguing finding that parameters with higher variance exhibit greater pruning robustness. VAR exploits this property by promoting such variance in the weight distribution, thereby mitigating the adverse effects of pruning. We further provide a theoretical analysis of its convergence behavior, supported by extensive empirical results demonstrating the superior pruning robustness of VAR. 

**Abstract (ZH)**: 深度神经网络在视觉识别任务中表现出色，但其庞大的参数数量使其在实际应用中不够实用。最近，单-shot 自裁剪已成为一种有效的方法，可以在不进行额外训练的情况下减小模型大小。然而，使用标准目标函数训练的模型在进行激进裁剪后往往会遭受显著的准确率下降。现有的某些自裁剪稳健优化器，如SAM和CrAM，通过引导模型向参数空间的平坦区域发展来缓解这种准确率下降，但这些方法不可避免地会带来额外的非忽视性计算开销。我们提出了一种方差放大正则化器（VAR），该正则化器在训练过程中故意增加模型参数的方差。我们的研究表明，高方差的参数展现出更强的自裁剪稳健性。VAR 利用这一特性，通过促进权重分布中的这种方差来减轻裁剪的不利影响。我们还提供了其收敛行为的理论分析，并通过广泛的经验结果证明了VAR的优越自裁剪稳健性。 

---
# Comparing Task-Agnostic Embedding Models for Tabular Data 

**Title (ZH)**: 面向表格数据的无任务嵌入模型对比研究 

**Authors**: Frederik Hoppe, Lars Kleinemeier, Astrid Franz, Udo Göbel  

**Link**: [PDF](https://arxiv.org/pdf/2511.14276)  

**Abstract**: Recent foundation models for tabular data achieve strong task-specific performance via in-context learning. Nevertheless, they focus on direct prediction by encapsulating both representation learning and task-specific inference inside a single, resource-intensive network. This work specifically focuses on representation learning, i.e., on transferable, task-agnostic embeddings. We systematically evaluate task-agnostic representations from tabular foundation models (TabPFN and TabICL) alongside with classical feature engineering (TableVectorizer) across a variety of application tasks as outlier detection (ADBench) and supervised learning (TabArena Lite). We find that simple TableVectorizer features achieve comparable or superior performance while being up to three orders of magnitude faster than tabular foundation models. The code is available at this https URL. 

**Abstract (ZH)**: 最近的表格数据基础模型通过上下文学习在特定任务上取得了强大性能，然而它们侧重于直接预测，将表示学习和特定任务推理封装在一个资源密集型网络中。本文专注于表示学习，即具有可转移性、任务无关的嵌入。我们系统地评估了表格基础模型（TabPFN和TabICL）以及经典特征工程（TableVectorizer）在多个应用任务（如异常检测ADBench和监督学习TabArena Lite）中的任务无关表示。我们发现简单的TableVectorizer特征在性能上可与表格基础模型媲美甚至更优，同时速度更快，快了1到3个数量级。代码可从此处获取。 

---
# ArbESC+: Arabic Enhanced Edit Selection System Combination for Grammatical Error Correction Resolving conflict and improving system combination in Arabic GEC 

**Title (ZH)**: arbESC+：阿拉伯语增强的编辑选择系统组合以解决语法错误修正中的冲突并提高系统组合性能 

**Authors**: Ahlam Alrehili, Areej Alhothali  

**Link**: [PDF](https://arxiv.org/pdf/2511.14230)  

**Abstract**: Grammatical Error Correction (GEC) is an important aspect of natural language processing. Arabic has a complicated morphological and syntactic structure, posing a greater challenge than other languages. Even though modern neural models have improved greatly in recent years, the majority of previous attempts used individual models without taking into account the potential benefits of combining different systems. In this paper, we present one of the first multi-system approaches for correcting grammatical errors in Arabic, the Arab Enhanced Edit Selection System Complication (ArbESC+). Several models are used to collect correction proposals, which are represented as numerical features in the framework. A classifier determines and implements the appropriate corrections based on these features. In order to improve output quality, the framework uses support techniques to filter overlapping corrections and estimate decision reliability. A combination of AraT5, ByT5, mT5, AraBART, AraBART+Morph+GEC, and Text editing systems gave better results than a single model alone, with F0.5 at 82.63% on QALB-14 test data, 84.64% on QALB-15 L1 data, and 65.55% on QALB-15 L2 data. As one of the most significant contributions of this work, it's the first Arab attempt to integrate linguistic error correction. Improving existing models provides a practical step towards developing advanced tools that will benefit users and researchers of Arabic text processing. 

**Abstract (ZH)**: 阿拉伯语语法错误纠正（GEC）方法研究 

---
# Parallelizing Tree Search with Twice Sequential Monte Carlo 

**Title (ZH)**: 使用两次顺序蒙特卡洛并行化树搜索 

**Authors**: Yaniv Oren, Joery A. de Vries, Pascal R. van der Vaart, Matthijs T. J. Spaan, Wendelin Böhmer  

**Link**: [PDF](https://arxiv.org/pdf/2511.14220)  

**Abstract**: Model-based reinforcement learning (RL) methods that leverage search are responsible for many milestone breakthroughs in RL. Sequential Monte Carlo (SMC) recently emerged as an alternative to the Monte Carlo Tree Search (MCTS) algorithm which drove these breakthroughs. SMC is easier to parallelize and more suitable to GPU acceleration. However, it also suffers from large variance and path degeneracy which prevent it from scaling well with increased search depth, i.e., increased sequential compute. To address these problems, we introduce Twice Sequential Monte Carlo Tree Search (TSMCTS). Across discrete and continuous environments TSMCTS outperforms the SMC baseline as well as a popular modern version of MCTS. Through variance reduction and mitigation of path degeneracy, TSMCTS scales favorably with sequential compute while retaining the properties that make SMC natural to parallelize. 

**Abstract (ZH)**: 基于模型的强化学习（RL）方法利用搜索在RL中产生了许多里程碑式的突破。最近， sequential Monte Carlo (SMC) 作为 Monte Carlo Tree Search (MCTS) 的一种替代算法崭露头角。SMC 更易于并行化且更适合 GPU 加速。然而，它也面临方差大和路径退化的问题，这阻碍了其在增加搜索深度时的扩展性，即增加顺序计算量。为了解决这些问题，我们提出了一种名为 Twice Sequential Monte Carlo Tree Search (TSMCTS) 的方法。在离散和连续环境中，TSMCTS 在性能上优于 SMC 基线以及一种流行的现代 MCTS 版本。通过减少方差和缓解路径退化，TSMCTS 在增加顺序计算量时具有可扩展性，同时保持了 SMC 易于并行化的特点。 

---
# Bridging the Gap Between Bayesian Deep Learning and Ensemble Weather Forecasts 

**Title (ZH)**: Bayesian深度学习与 ensemble天气预报的桥梁 

**Authors**: Xinlei Xiong, Wenbo Hu, Shuxun Zhou, Kaifeng Bi, Lingxi Xie, Ying Liu, Richang Hong, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2511.14218)  

**Abstract**: Weather forecasting is fundamentally challenged by the chaotic nature of the atmosphere, necessitating probabilistic approaches to quantify uncertainty. While traditional ensemble prediction (EPS) addresses this through computationally intensive simulations, recent advances in Bayesian Deep Learning (BDL) offer a promising but often disconnected alternative. We bridge these paradigms through a unified hybrid Bayesian Deep Learning framework for ensemble weather forecasting that explicitly decomposes predictive uncertainty into epistemic and aleatoric components, learned via variational inference and a physics-informed stochastic perturbation scheme modeling flow-dependent atmospheric dynamics, respectively. We further establish a unified theoretical framework that rigorously connects BDL and EPS, providing formal theorems that decompose total predictive uncertainty into epistemic and aleatoric components under the hybrid BDL framework. We validate our framework on the large-scale 40-year ERA5 reanalysis dataset (1979-2019) with 0.25° spatial resolution. Experimental results show that our method not only improves forecast accuracy and yields better-calibrated uncertainty quantification but also achieves superior computational efficiency compared to state-of-the-art probabilistic diffusion models. We commit to making our code open-source upon acceptance of this paper. 

**Abstract (ZH)**: 一种统一的混合贝叶斯深度学习框架用于分解和建模集合天气预报中的认识不确定性与偶然不确定性 

---
# Multi-Scale Correlation-Aware Transformer for Maritime Vessel Re-Identification 

**Title (ZH)**: 多尺度关联 Awareness Transformer 用于 maritime 船舶再识别 

**Authors**: Yunhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.14203)  

**Abstract**: Maritime vessel re-identification (Re-ID) plays a crucial role in advancing maritime monitoring and intelligent situational awareness systems. However, some existing vessel Re-ID methods are directly adapted from pedestrian-focused algorithms, making them ill-suited for mitigating the unique problems present in vessel images, particularly the greater intra-identity variations and more severe missing of local parts, which lead to the emergence of outlier samples within the same identity. To address these challenges, we propose the Multi-scale Correlation-aware Transformer Network (MCFormer), which explicitly models multi-scale correlations across the entire input set to suppress the adverse effects of outlier samples with intra-identity variations or local missing, incorporating two novel modules, the Global Correlation Module (GCM), and the Local Correlation Module (LCM). Specifically, GCM constructs a global similarity affinity matrix across all input images to model global correlations through feature aggregation based on inter-image consistency, rather than solely learning features from individual images as in most existing approaches. Simultaneously, LCM mines and aligns local features of positive samples with contextual similarity to extract local correlations by maintaining a dynamic memory bank, effectively compensating for missing or occluded regions in individual images. To further enhance feature robustness, MCFormer integrates global and local features that have been respectively correlated across multiple scales, effectively capturing latent relationships among image features. Experiments on three benchmarks demonstrate that MCFormer achieves state-of-the-art performance. 

**Abstract (ZH)**: 海事船舶再识别（Re-ID）在推进海事监控和智能态势感知系统中发挥着重要作用。然而，一些现有的船舶Re-ID方法直接从行人关注的算法中借鉴而来，使其不适用于缓解船舶图像中特有的问题，特别是更大的身份内差异和更严重的局部缺失，导致同一身份内出现异常样本。为了应对这些挑战，我们提出了多尺度相关性感知变换网络（MCFormer），该网络明确建模输入集中的多尺度相关性，通过基于跨图像一致性进行特征聚合而非仅从单个图像中学习特征来抑制身份内差异或局部缺失导致的不良影响，并引入了两个新的模块：全局相关性模块（GCM）和局部相关性模块（LCM）。GCM构建了所有输入图像之间的全局相似性亲和矩阵，通过基于跨图像一致性的特征聚合来建模全局相关性。同时，LCM通过维护一个动态记忆库来挖掘和对齐正样本的局部特征，以基于上下文相似性提取局部相关性，从而有效补充单个图像中缺失或遮挡的区域。为了进一步增强特征的鲁棒性，MCFormer结合了在多尺度上分别建模的全局和局部特征，有效地捕捉图像特征之间的潜在关系。在三个基准上的实验结果表明，MCFormer达到了最先进的性能。 

---
# Certified Signed Graph Unlearning 

**Title (ZH)**: 认证签名图删除 

**Authors**: Junpeng Zhao, Lin Li, Kaixi Hu, Kaize Shi, Jingling Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2511.14168)  

**Abstract**: Signed graphs model complex relationships through positive and negative edges, with widespread real-world applications. Given the sensitive nature of such data, selective removal mechanisms have become essential for privacy protection. While graph unlearning enables the removal of specific data influences from Graph Neural Networks (GNNs), existing methods are designed for conventional GNNs and overlook the unique heterogeneous properties of signed graphs. When applied to Signed Graph Neural Networks (SGNNs), these methods lose critical sign information, degrading both model utility and unlearning effectiveness. To address these challenges, we propose Certified Signed Graph Unlearning (CSGU), which provides provable privacy guarantees while preserving the sociological principles underlying SGNNs. CSGU employs a three-stage method: (1) efficiently identifying minimal influenced neighborhoods via triangular structures, (2) applying sociological theories to quantify node importance for optimal privacy budget allocation, and (3) performing importance-weighted parameter updates to achieve certified modifications with minimal utility degradation. Extensive experiments demonstrate that CSGU outperforms existing methods, achieving superior performance in both utility preservation and unlearning effectiveness on SGNNs. 

**Abstract (ZH)**: 签名图通过正负边建模复杂关系，具有广泛的实际应用。鉴于此类数据的敏感性，选择性移除机制已成为隐私保护的必备手段。虽然图遗忘能够从图神经网络（GNNs）中删除特定数据的影响，但现有方法主要针对常规GNNs，忽略了签名图的独特异质特性。当应用于签名图神经网络（SGNNs）时，这些方法会丢失关键的符号信息，导致模型效用和遗忘效果下降。为了解决这些挑战，我们提出了认证签名图遗忘（Certified Signed Graph Unlearning，CSGU），它提供了可证明的隐私保证，同时保留了支撑SGNNs的社会学原则。CSGU采用三阶段方法：（1）通过三角结构高效识别最小影响区域，（2）应用社会学理论量化节点的重要性以优化隐私预算分配，（3）执行重要性加权参数更新以实现认证修改并尽量减少效用下降。 extensive实验证明，CSGU在保留效用和遗忘效果方面优于现有方法，表现出更好的性能。 

---
# Selective Weak-to-Strong Generalization 

**Title (ZH)**: 选择性弱到强的泛化 

**Authors**: Hao Lang, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.14166)  

**Abstract**: Future superhuman models will surpass the ability of humans and humans will only be able to \textit{weakly} supervise superhuman models. To alleviate the issue of lacking high-quality data for model alignment, some works on weak-to-strong generalization (W2SG) finetune a strong pretrained model with a weak supervisor so that it can generalize beyond weak supervision. However, the invariable use of weak supervision in existing methods exposes issues in robustness, with a proportion of weak labels proving harmful to models. In this paper, we propose a selective W2SG framework to avoid using weak supervision when unnecessary. We train a binary classifier P(IK) to identify questions that a strong model can answer and use its self-generated labels for alignment. We further refine weak labels with a graph smoothing method. Extensive experiments on three benchmarks show that our method consistently outperforms competitive baselines. Further analyses show that P(IK) can generalize across tasks and difficulties, which indicates selective W2SG can help superalignment. 

**Abstract (ZH)**: 未来超人类模型将超越人类能力，人类仅能对超人类模型进行弱监督。为了解决模型对高质量数据需求不足的问题，一些弱到强泛化的（W2SG）方法通过使用弱监督者微调强预训练模型，使其能够超越弱监督进行泛化。然而，现有方法中不变地使用弱监督暴露了鲁棒性问题，部分弱标签对模型有害。在本文中，我们提出了一种选择性的W2SG框架，避免在不必要的时候使用弱监督。我们训练一个二分类器P(IK)来识别强模型可以回答的问题，并使用其自生成的标签进行对齐。我们进一步通过图平滑方法细化弱标签。在三个基准上的广泛实验表明，我们的方法始终优于竞争baseline。进一步的分析表明，P(IK)可以在任务和困难程度上泛化，这表明选择性的W2SG有助于超对齐。 

---
# Fair-GNE : Generalized Nash Equilibrium-Seeking Fairness in Multiagent Healthcare Automation 

**Title (ZH)**: Fair-GNE：多代理健康care自动化中的广义纳什均衡寻求公平性 

**Authors**: Promise Ekpo, Saesha Agarwal, Felix Grimm, Lekan Molu, Angelique Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2511.14135)  

**Abstract**: Enforcing a fair workload allocation among multiple agents tasked to achieve an objective in learning enabled demand side healthcare worker settings is crucial for consistent and reliable performance at runtime. Existing multi-agent reinforcement learning (MARL) approaches steer fairness by shaping reward through post hoc orchestrations, leaving no certifiable self-enforceable fairness that is immutable by individual agents at runtime. Contextualized within a setting where each agent shares resources with others, we address this shortcoming with a learning enabled optimization scheme among self-interested decision makers whose individual actions affect those of other agents. This extends the problem to a generalized Nash equilibrium (GNE) game-theoretic framework where we steer group policy to a safe and locally efficient equilibrium, so that no agent can improve its utility function by unilaterally changing its decisions. Fair-GNE models MARL as a constrained generalized Nash equilibrium-seeking (GNE) game, prescribing an ideal equitable collective equilibrium within the problem's natural fabric. Our hypothesis is rigorously evaluated in our custom-designed high-fidelity resuscitation simulator. Across all our numerical experiments, Fair-GNE achieves significant improvement in workload balance over fixed-penalty baselines (0.89 vs.\ 0.33 JFI, $p < 0.01$) while maintaining 86\% task success, demonstrating statistically significant fairness gains through adaptive constraint enforcement. Our results communicate our formulations, evaluation metrics, and equilibrium-seeking innovations in large multi-agent learning-based healthcare systems with clarity and principled fairness enforcement. 

**Abstract (ZH)**: 在学习驱动的供需医护人员分配中确保多个代理任务客观的工作量分配对于运行时的一致可靠表现至关重要。现有的多代理强化学习（MARL）方法通过事后编排塑造奖励来引导公平，但无法提供在运行时由个体代理自我执行且不可变更的可验证公平性。在每个代理与其他代理共享资源的环境中，我们通过将自我 interested 决策制定者之间的学习驱动优化方案扩展到广义纳什均衡（GNE）博弈框架来解决这一不足，使群体策略达到一种安全且局部高效的均衡状态，使得任一代理单独改变决策都无法提高其效用函数。Fair-GNE将MARL建模为一个受限的广义纳什均衡（GNE）寻求博弈，规定了一个在问题自然结构中的理想公平集体均衡。我们在我们自定义设计的高保真复苏模拟器中严格评估了我们的假设。在所有数值实验中，Fair-GNE在固定惩罚基准（0.89 vs. 0.33 JFI，$p < 0.01$）上实现了显著的工作量平衡改善，同时保持了86%的任务成功率，通过自适应约束执行实现了统计显著的公平性收益。我们的结果清晰地传达了我们在大规模多代理学习驱动的医疗系统中的形式化表述、评估指标以及均衡寻求的创新，并进行了原则性的公平性执行。 

---
# Soft-Label Training Preserves Epistemic Uncertainty 

**Title (ZH)**: Soft-Label Training保留了 epistemic 不确定性 

**Authors**: Agamdeep Singh, Ashish Tiwari, Hosein Hasanbeig, Priyanshu Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.14117)  

**Abstract**: Many machine learning tasks involve inherent subjectivity, where annotators naturally provide varied labels. Standard practice collapses these label distributions into single labels, aggregating diverse human judgments into point estimates. We argue that this approach is epistemically misaligned for ambiguous data--the annotation distribution itself should be regarded as the ground truth. Training on collapsed single labels forces models to express false confidence on fundamentally ambiguous cases, creating a misalignment between model certainty and the diversity of human perception. We demonstrate empirically that soft-label training, which treats annotation distributions as ground truth, preserves epistemic uncertainty. Across both vision and NLP tasks, soft-label training achieves 32% lower KL divergence from human annotations and 61% stronger correlation between model and annotation entropy, while matching the accuracy of hard-label training. Our work repositions annotation distributions from noisy signals to be aggregated away, to faithful representations of epistemic uncertainty that models should learn to reproduce. 

**Abstract (ZH)**: 许多机器学习任务包含固有的主观性，其中注释员自然提供不同的标签。标准做法是将这些标签分布压缩为单一标签，将多样的人类判断聚合为点估计。我们认为，在模糊数据的情况下，这种做法在认识论上是错位的——注释分布本身应被视为ground truth。在压缩的单一标签上进行训练迫使模型在基本模糊的情况下表达虚假的信心，从而在模型的确定性和人类感知的多样性之间造成错位。我们实证证明，将注释分布视为ground truth的软标签训练保留了认识论上的不确定性。在视觉和自然语言处理任务中，软标签训练的KL散度比人类注释低32%，模型与注释熵的相关性比硬标签训练强61%，同时保持相同的准确度。我们的工作重新定位了注释分布，从嘈杂的信号聚合为忠实的认识论不确定性表示，模型应该学习再现这些表示。 

---
# Synthetic Clinical Notes for Rare ICD Codes: A Data-Centric Framework for Long-Tail Medical Coding 

**Title (ZH)**: 基于数据为中心的框架：用于长尾医学编码的罕见ICD编码合成临床笔记 

**Authors**: Truong Vo, Weiyi Wu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.14112)  

**Abstract**: Automatic ICD coding from clinical text is a critical task in medical NLP but remains hindered by the extreme long-tail distribution of diagnostic codes. Thousands of rare and zero-shot ICD codes are severely underrepresented in datasets like MIMIC-III, leading to low macro-F1 scores. In this work, we propose a data-centric framework that generates high-quality synthetic discharge summaries to mitigate this imbalance. Our method constructs realistic multi-label code sets anchored on rare codes by leveraging real-world co-occurrence patterns, ICD descriptions, synonyms, taxonomy, and similar clinical notes. Using these structured prompts, we generate 90,000 synthetic notes covering 7,902 ICD codes, significantly expanding the training distribution. We fine-tune two state-of-the-art transformer-based models, PLM-ICD and GKI-ICD, on both the original and extended datasets. Experiments show that our approach modestly improves macro-F1 while maintaining strong micro-F1, outperforming prior SOTA. While the gain may seem marginal relative to the computational cost, our results demonstrate that carefully crafted synthetic data can enhance equity in long-tail ICD code prediction. 

**Abstract (ZH)**: 自动从临床文本编码ICD代码是医学NLP中的关键任务，但受到诊断代码极端长尾分布的限制。在MIMIC-III等数据集中，成千上万的罕见和零样本ICD代码严重欠代表，导致低的宏F1分数。在本工作中，我们提出了一种以数据为中心的框架，生成高质量的合成出院总结以缓解这种不平衡。我们的方法通过利用现实世界共现模式、ICD描述、同义词、分类学和类似的临床笔记构建现实的多标签代码集。使用这些结构化的提示，我们生成了涵盖7,902个ICD代码的90,000个合成笔记，极大地扩展了训练分布。我们在原始数据集和扩展数据集上分别对两个最先进的基于变压器的模型PLM-ICD和GKI-ICD进行了微调。实验表明，我们的方法在轻微提高宏F1的同时维持了较强的微观F1，超越了先前的SOTA。尽管相对于计算成本而言收益似乎微小，但我们的结果表明，精心设计的合成数据可以增强长尾ICD代码预测的公平性。 

---
# Error-Driven Scene Editing for 3D Grounding in Large Language Models 

**Title (ZH)**: 基于错误的场景编辑以改善大型语言模型中的3D语义对齐 

**Authors**: Yue Zhang, Zun Wang, Han Lin, Jialu Li, Jianing Yang, Yonatan Bitton, Idan Szpektor, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2511.14086)  

**Abstract**: Despite recent progress in 3D-LLMs, they remain limited in accurately grounding language to visual and spatial elements in 3D environments. This limitation stems in part from training data that focuses on language reasoning rather than spatial understanding due to scarce 3D resources, leaving inherent grounding biases unresolved. To address this, we propose 3D scene editing as a key mechanism to generate precise visual counterfactuals that mitigate these biases through fine-grained spatial manipulation, without requiring costly scene reconstruction or large-scale 3D data collection. Furthermore, to make these edits targeted and directly address the specific weaknesses of the model, we introduce DEER-3D, an error-driven framework following a structured "Decompose, Diagnostic Evaluation, Edit, and Re-train" workflow, rather than broadly or randomly augmenting data as in conventional approaches. Specifically, upon identifying a grounding failure of the 3D-LLM, our framework first diagnoses the exact predicate-level error (e.g., attribute or spatial relation). It then executes minimal, predicate-aligned 3D scene edits, such as recoloring or repositioning, to produce targeted counterfactual supervision for iterative model fine-tuning, significantly enhancing grounding accuracy. We evaluate our editing pipeline across multiple benchmarks for 3D grounding and scene understanding tasks, consistently demonstrating improvements across all evaluated datasets through iterative refinement. DEER-3D underscores the effectiveness of targeted, error-driven scene editing in bridging linguistic reasoning capabilities with spatial grounding in 3D LLMs. 

**Abstract (ZH)**: 尽管近期在3D大语言模型方面取得了进展，但它们在将语言精准地映射到3D环境中的视觉和空间元素方面仍存在局限性。这一局限部分源于训练数据侧重于语言推理而非空间理解，因缺乏3D资源而未能解决内在的映射偏差问题。为了解决这一问题，我们提出3D场景编辑作为一种关键机制，通过精细的空间操控生成精确的视觉假设以减轻这些偏差，而无需进行耗时的场景重建或大规模3D数据采集。此外，为了使这些编辑具有针对性，直接针对模型的具体薄弱环节进行，我们引入了DEER-3D，这是一种基于结构化的“分解、诊断评估、编辑和重训”工作流的错误驱动框架，而不是像传统方法那样广泛或随机地扩充数据。具体而言，当识别出3D大语言模型的映射失败时，我们的框架首先诊断出具体的谓词级错误（例如属性或空间关系）。然后执行最小化且与谓词对齐的3D场景编辑，如重新着色或重新定位，从而生成定向的假设监督，以促进迭代的模型微调，显著提高映射精度。我们在多个3D映射和场景理解任务基准上评估了我们的编辑管道，通过迭代完善，在所有测试数据集上均展示了改进效果。DEER-3D突显了目标导向、错误驱动的场景编辑在将语言推理能力与3D大语言模型的空间映射结合方面的有效性。 

---
# Automated glenoid bone loss measurement and segmentation in CT scans for pre-operative planning in shoulder instability 

**Title (ZH)**: 基于CT扫描的肩关节不稳定手术前计划中肱骨关节面骨丢失的自动化测量与分割 

**Authors**: Zhonghao Liu, Hanxue Gu, Qihang Li, Michael Fox, Jay M. Levin, Maciej A. Mazurowski, Brian C. Lau  

**Link**: [PDF](https://arxiv.org/pdf/2511.14083)  

**Abstract**: Reliable measurement of glenoid bone loss is essential for operative planning in shoulder instability, but current manual and semi-automated methods are time-consuming and often subject to interreader variability. We developed and validated a fully automated deep learning pipeline for measuring glenoid bone loss on three-dimensional computed tomography (CT) scans using a linear-based, en-face view, best-circle method. Shoulder CT images of 91 patients (average age, 40 years; range, 14-89 years; 65 men) were retrospectively collected along with manual labels including glenoid segmentation, landmarks, and bone loss measurements. The multi-stage algorithm has three main stages: (1) segmentation, where we developed a U-Net to automatically segment the glenoid and humerus; (2) anatomical landmark detection, where a second network predicts glenoid rim points; and (3) geometric fitting, where we applied principal component analysis (PCA), projection, and circle fitting to compute the percentage of bone loss. The automated measurements showed strong agreement with consensus readings and exceeded surgeon-to-surgeon consistency (intraclass correlation coefficient (ICC) 0.84 vs 0.78), including in low- and high-bone-loss subgroups (ICC 0.71 vs 0.63 and 0.83 vs 0.21, respectively; P < 0.001). For classifying patients into low, medium, and high bone-loss categories, the pipeline achieved a recall of 0.714 for low and 0.857 for high severity, with no low cases misclassified as high or vice versa. These results suggest that our method is a time-efficient and clinically reliable tool for preoperative planning in shoulder instability and for screening patients with substantial glenoid bone loss. Code and dataset are available at this https URL. 

**Abstract (ZH)**: 可靠的盂骨骨质丢失测量对于肩关节不稳手术计划至关重要，但当前的手动和半自动化方法耗时且常受阅读者间变异性的影响。我们开发并验证了一种基于深度学习的完全自动化管道，用于通过线性基于的面视最佳圆方法在三维计算机断层扫描（CT）图像上测量盂骨骨质丢失。回顾性收集了91例患者的肩关节CT图像（平均年龄40岁；年龄范围14-89岁；男性65例），包括盂骨分割、标志点和骨质丢失测量的手动标签。多阶段算法分为三个主要阶段：（1）分割，我们开发了一个U-Net自动分割盂骨和肱骨；（2）解剖标志点检测，第二个网络预测盂骨边缘点；（3）几何拟合，我们应用主成分分析（PCA）、投影和圆拟合计算骨质丢失的百分比。自动测量与共识读数表现出强烈的一致性，并且超过了外科医生之间的一致性（ICC 0.84 vs 0.78），包括在低骨丢失和高骨丢失亚组中（ICC分别为0.71 vs 0.63和0.83 vs 0.21；P < 0.001）。对于将患者分类为低、中、高骨质丢失类别，管道在低度严重性中的召回率为0.714，在高度严重性中的召回率为0.857，没有将低病例误分类为高病例或反之。这些结果表明，我们的方法是一种高效且临床可靠的工具，可用于肩关节不稳的术前计划以及筛选有显著盂骨骨质丢失的患者。代码和数据集可通过以下链接获取。 

---
# Zero-Training Task-Specific Model Synthesis for Few-Shot Medical Image Classification 

**Title (ZH)**: 基于零训练的任务特定模型合成的少样本医学图像分类 

**Authors**: Yao Qin, Yangyang Yan, YuanChao Yang, Jinhua Pang, Huanyong Bi, Yuan Liu, HaiHua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14082)  

**Abstract**: Deep learning models have achieved remarkable success in medical image analysis but are fundamentally constrained by the requirement for large-scale, meticulously annotated datasets. This dependency on "big data" is a critical bottleneck in the medical domain, where patient data is inherently difficult to acquire and expert annotation is expensive, particularly for rare diseases where samples are scarce by definition. To overcome this fundamental challenge, we propose a novel paradigm: Zero-Training Task-Specific Model Synthesis (ZS-TMS). Instead of adapting a pre-existing model or training a new one, our approach leverages a large-scale, pre-trained generative engine to directly synthesize the entire set of parameters for a task-specific classifier. Our framework, the Semantic-Guided Parameter Synthesizer (SGPS), takes as input minimal, multi-modal task information as little as a single example image (1-shot) and a corresponding clinical text description to directly synthesize the entire set of parameters for a task-specific classifier.
The generative engine interprets these inputs to generate the weights for a lightweight, efficient classifier (e.g., an EfficientNet-V2), which can be deployed for inference immediately without any task-specific training or fine-tuning. We conduct extensive evaluations on challenging few-shot classification benchmarks derived from the ISIC 2018 skin lesion dataset and a custom rare disease dataset. Our results demonstrate that SGPS establishes a new state-of-the-art, significantly outperforming advanced few-shot and zero-shot learning methods, especially in the ultra-low data regimes of 1-shot and 5-shot classification. This work paves the way for the rapid development and deployment of AI-powered diagnostic tools, particularly for the long tail of rare diseases where data is critically limited. 

**Abstract (ZH)**: 零训练任务特定模型合成（ZS-TMS）：面向医疗图像分析的新型范式 

---
# CafeMed: Causal Attention Fusion Enhanced Medication Recommendation 

**Title (ZH)**: CafeMed: 因果注意力融合增强的药品推荐 

**Authors**: Kelin Ren, Chan-Yang Ju, Dong-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.14064)  

**Abstract**: Medication recommendation systems play a crucial role in assisting clinicians with personalized treatment decisions. While existing approaches have made significant progress in learning medication representations, they suffer from two fundamental limitations: (i) treating medical entities as independent features without modeling their synergistic effects on medication selection; (ii) employing static causal relationships that fail to adapt to patient-specific contexts and health states. To address these challenges, we propose CafeMed, a framework that integrates dynamic causal reasoning with cross-modal attention for safe and accurate medication recommendation. CafeMed introduces two key components: the Causal Weight Generator (CWG) that transforms static causal effects into dynamic modulation weights based on individual patient states, and the Channel Harmonized Attention Refinement Module (CHARM) that captures complex interdependencies between diagnoses and procedures. This design enables CafeMed to model how different medical conditions jointly influence treatment decisions while maintaining medication safety constraints. Extensive experiments on MIMIC-III and MIMIC-IV datasets demonstrate that CafeMed significantly outperforms state-of-the-art baselines, achieving superior accuracy in medication prediction while maintaining the lower drug--drug interaction rates. Our results indicate that incorporating dynamic causal relationships and cross-modal synergies leads to more clinically-aligned and personalized medication recommendations. Our code is released publicly at this https URL. 

**Abstract (ZH)**: 基于动态因果推理和跨模态注意力的药物推荐系统CafeMed 

---
# Radial Compensation: Stable and Semantically Decoupled Generative Models on Riemannian Manifolds 

**Title (ZH)**: 径向补偿：黎曼流形上稳定且语义解耦的生成模型 

**Authors**: Marios Papamichals, Regina Ruane  

**Link**: [PDF](https://arxiv.org/pdf/2511.14056)  

**Abstract**: Generative models on curved spaces rely on charts to map Euclidean spaces to manifolds. Exponential maps preserve geodesics but have stiff, radius-dependent Jacobians, while volume-preserving charts maintain densities but distort geodesic distances. Both approaches entangle curvature with model parameters, inflating gradient variance. In high-dimensional latent normalizing flows, the wrapped exponential prior can stretch radii far beyond the curvature scale, leading to poor test likelihoods and stiff solvers. We introduce Radial Compensation (RC), an information-geometric method that selects the base density in the tangent space so that the likelihood depends only on geodesic distance from a pole, decoupling parameter semantics from curvature. RC lets radial parameters retain their usual meaning in geodesic units, while the chart can be tuned as a numerical preconditioner. We extend RC to manifolds with known geodesic polar volume and show that RC is the only construction for geodesic-radial likelihoods with curvature-invariant Fisher information. We derive the Balanced-Exponential (bExp) chart family, balancing volume distortion and geodesic error. Under RC, all bExp settings preserve the same manifold density and Fisher information, with smaller dial values reducing gradient variance and flow cost. Empirically, RC yields stable generative models across densities, VAEs, flows on images and graphs, and protein models. RC improves likelihoods, restores clean geodesic radii, and prevents radius blow-ups in high-dimensional flows, making RC-bExp a robust default for likelihood-trained generative models on manifolds. 

**Abstract (ZH)**: Radial Compensation for Generative Models on Manifolds 

---
# Training-free Detection of AI-generated images via Cropping Robustness 

**Title (ZH)**: 基于剪裁鲁棒性的无训练AI生成图像检测 

**Authors**: Sungik Choi, Hankook Lee, Moontae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.14030)  

**Abstract**: AI-generated image detection has become crucial with the rapid advancement of vision-generative models. Instead of training detectors tailored to specific datasets, we study a training-free approach leveraging self-supervised models without requiring prior data knowledge. These models, pre-trained with augmentations like RandomResizedCrop, learn to produce consistent representations across varying resolutions. Motivated by this, we propose WaRPAD, a training-free AI-generated image detection algorithm based on self-supervised models. Since neighborhood pixel differences in images are highly sensitive to resizing operations, WaRPAD first defines a base score function that quantifies the sensitivity of image embeddings to perturbations along high-frequency directions extracted via Haar wavelet decomposition. To simulate robustness against cropping augmentation, we rescale each image to a multiple of the models input size, divide it into smaller patches, and compute the base score for each patch. The final detection score is then obtained by averaging the scores across all patches. We validate WaRPAD on real datasets of diverse resolutions and domains, and images generated by 23 different generative models. Our method consistently achieves competitive performance and demonstrates strong robustness to test-time corruptions. Furthermore, as invariance to RandomResizedCrop is a common training scheme across self-supervised models, we show that WaRPAD is applicable across self-supervised models. 

**Abstract (ZH)**: 基于自监督模型的无训练AI生成图像检测算法（WaRPAD） 

---
# MRI Plane Orientation Detection using a Context-Aware 2.5D Model 

**Title (ZH)**: 基于上下文感知的2.5D模型的MRI切片方向检测 

**Authors**: SangHyuk Kim, Daniel Haehn, Sumientra Rampersad  

**Link**: [PDF](https://arxiv.org/pdf/2511.14021)  

**Abstract**: Humans can easily identify anatomical planes (axial, coronal, and sagittal) on a 2D MRI slice, but automated systems struggle with this task. Missing plane orientation metadata can complicate analysis, increase domain shift when merging heterogeneous datasets, and reduce accuracy of diagnostic classifiers. This study develops a classifier that accurately generates plane orientation metadata. We adopt a 2.5D context-aware model that leverages multi-slice information to avoid ambiguity from isolated slices and enable robust feature learning. We train the 2.5D model on both 3D slice sequences and static 2D images. While our 2D reference model achieves 98.74% accuracy, our 2.5D method raises this to 99.49%, reducing errors by 60%, highlighting the importance of 2.5D context. We validate the utility of our generated metadata in a brain tumor detection task. A gated strategy selectively uses metadata-enhanced predictions based on uncertainty scores, boosting accuracy from 97.0% with an image-only model to 98.0%, reducing misdiagnoses by 33.3%. We integrate our plane orientation model into an interactive web application and provide it open-source. 

**Abstract (ZH)**: 人类可以轻易地在2D MRI切片上识别出轴位、冠状位和矢状位等解剖平面，但自动化系统在这一任务上面临挑战。缺失的平面方向元数据可能使分析复杂化，增加合并异构数据集时的领域偏移，并降低诊断分类器的准确性。本研究开发了一种能准确生成平面方向元数据的分类器。我们采用了2.5D上下文感知模型，利用多切片信息来避免孤立切片产生的歧义，从而实现稳健的特征学习。我们同时在3D切片序列和静态2D图像上训练2.5D模型。我们的2D参考模型准确率达到98.74%，而2.5D方法将其提升到99.49%，错误率降低了60%，突显了2.5D上下文的重要性。我们验证了生成的元数据在脑肿瘤检测任务中的实用性。基于不确定性分数的门控策略选择性地使用元数据增强的预测，将仅使用图像的模型准确率从97.0%提升到98.0%，误诊率降低了33.3%。我们将平面方向模型集成到一个交互式网络应用中，并提供开源。 

---
# Developing a Grounded View of AI 

**Title (ZH)**: 基于地面视角的人工智能发展 

**Authors**: Bifei Mao, Lanqing Hong  

**Link**: [PDF](https://arxiv.org/pdf/2511.14013)  

**Abstract**: As a capability coming from computation, how does AI differ fundamentally from the capabilities delivered by rule-based software program? The paper examines the behavior of artificial intelligence (AI) from engineering points of view to clarify its nature and limits. The paper argues that the rationality underlying humanity's impulse to pursue, articulate, and adhere to rules deserves to be valued and preserved. Identifying where rule-based practical rationality ends is the beginning of making it aware until action. Although the rules of AI behaviors are still hidden or only weakly observable, the paper has proposed a methodology to make a sense of discrimination possible and practical to identify the distinctions of the behavior of AI models with three types of decisions. It is a prerequisite for human responsibilities with alternative possibilities, considering how and when to use AI. It would be a solid start for people to ensure AI system soundness for the well-being of humans, society, and the environment. 

**Abstract (ZH)**: 作为一种来自计算的能力，人工智能（AI）与基于规则的软件程序提供的能力有何根本不同？本文从工程的角度探讨AI的行为，以阐明其本质和局限性。文章主张，人类追求、阐述并遵守规则的合理性值得被珍视和保留。识别基于规则的实践理性结束之处是使其意识到自身行为的开始。尽管AI行为的规则仍然隐藏或仅弱可观测，本文提出了一种方法论，使其能够对AI模型的三种决定类型的行为差异进行区分和实践识别。这对于考虑如何及何时使用AI，以确保人类的福祉、社会和环境的AI系统的健全性是必要的前提。这将为人们确保AI系统的健全性提供坚实的基础。 

---
# Can Artificial Intelligence Accelerate Technological Progress? Researchers' Perspectives on AI in Manufacturing and Materials Science 

**Title (ZH)**: 人工智能能否加速技术进步？研究人员对人工智能在制造业和材料科学中的看法 

**Authors**: John P. Nelson, Olajide Olugbade, Philip Shapira, Justin B. Biddle  

**Link**: [PDF](https://arxiv.org/pdf/2511.14007)  

**Abstract**: Artificial intelligence (AI) raises expectations of substantial increases in rates of technological and scientific progress, but such anticipations are often not connected to detailed ground-level studies of AI use in innovation processes. Accordingly, it remains unclear how and to what extent AI can accelerate innovation. To help to fill this gap, we report results from 32 interviews with U.S.-based academic manufacturing and materials sciences researchers experienced with AI and machine learning (ML) techniques. Interviewees primarily used AI for modeling of materials and manufacturing processes, facilitating cheaper and more rapid search of design spaces for materials and manufacturing processes alike. They report benefits including cost, time, and computation savings in technology development. However, interviewees also report that AI/ML tools are unreliable outside design spaces for which dense data are already available; that they require skilled and judicious application in tandem with older research techniques; and that AI/ML tools may detrimentally circumvent opportunities for disruptive theoretical advancement. Based on these results, we suggest there is reason for optimism about acceleration in sustaining innovations through the use of to AI/ML; but that support for conventional empirical, computational, and theoretical research is required to maintain the likelihood of further major advances in manufacturing and materials science. 

**Abstract (ZH)**: 人工智能（AI）提高了对技术与科学进步显著增加的期望，但这种预期通常未与关于AI在创新过程中的使用情况的详细实地研究相连。因此，目前尚不清楚AI在多大程度上可以加速创新。为了填补这一空白，我们报道了对32位美国学术制造和材料科学研究人员的访谈结果，这些研究人员经验丰富且熟悉AI和机器学习（ML）技术。受访者主要使用AI进行材料和制造过程的建模，以促进在设计空间中更快更便宜地搜索材料和制造过程的设计。他们报告了技术开发中的成本、时间和计算节约等好处。然而，受访者还表示，AI/ML工具在设计空间外往往是不可靠的；它们需要与传统的研究技术相结合并谨慎应用；并且AI/ML工具可能会绕过破坏性的理论进步的机会。基于这些结果，我们认为，通过使用AI/ML可以在持续创新方面保持乐观态度；但需要支持传统的实证、计算和理论研究，以维持在制造与材料科学领域实现进一步重大进展的可能性。 

---
# FlakyGuard: Automatically Fixing Flaky Tests at Industry Scale 

**Title (ZH)**: FlakyGuard: 大规模自动修复 flaky 测试 

**Authors**: Chengpeng Li, Farnaz Behrang, August Shi, Peng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.14002)  

**Abstract**: Flaky tests that non-deterministically pass or fail waste developer time and slow release cycles. While large language models (LLMs) show promise for automatically repairing flaky tests, existing approaches like FlakyDoctor fail in industrial settings due to the context problem: providing either too little context (missing critical production code) or too much context (overwhelming the LLM with irrelevant information). We present FlakyGuard, which addresses this problem by treating code as a graph structure and using selective graph exploration to find only the most relevant context. Evaluation on real-world flaky tests from industrial repositories shows that FlakyGuard repairs 47.6 % of reproducible flaky tests with 51.8 % of the fixes accepted by developers. Besides it outperforms state-of-the-art approaches by at least 22 % in repair success rate. Developer surveys confirm that 100 % find FlakyGuard's root cause explanations useful. 

**Abstract (ZH)**: 非确定性通过或失败的 flakes 测试浪费开发者时间并延误发布周期。尽管大型语言模型（LLMs）显示出自动修复 flakes 测试的潜力，但现有方法如 FlakyDoctor 在工业环境中因上下文问题而失效：提供的上下文要么太少（缺少关键生产代码），要么太多（使 LLM 沉浸于无关信息中）。我们提出 FlakyGuard，通过将代码视为图结构并使用选择性的图探索来找到最相关的上下文，从而解决这一问题。在现实世界工业仓库中的 flaky 测试上的评估表明，FlakyGuard 修复了可重复的 flaky 测试的 47.6%，且 51.8% 的修复得到了开发者的接受。此外，FlakyGuard 在修复成功率上至少比最佳现有方法高 22%。开发者调查确认，所有开发者都认为 FlakyGuard 的根本原因解释非常有用。 

---
# How to Marginalize in Causal Structure Learning? 

**Title (ZH)**: 在因果结构学习中如何实现边际化？ 

**Authors**: William Zhao, Guy Van den Broeck, Benjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14001)  

**Abstract**: Bayesian networks (BNs) are a widely used class of probabilistic graphical models employed in numerous application domains. However, inferring the network's graphical structure from data remains challenging. Bayesian structure learners approach this problem by inferring a posterior distribution over the possible directed acyclic graphs underlying the BN. The inference process often requires marginalizing over probability distributions, which is typically done using dynamic programming methods that restrict the set of possible parents for each node. Instead, we present a novel method that utilizes tractable probabilistic circuits to circumvent this restriction. This method utilizes a new learning routine that trains these circuits on both the original distribution and marginal queries. The architecture of probabilistic circuits then inherently allows for fast and exact marginalization on the learned distribution. We then show empirically that utilizing our method to answer marginals allows Bayesian structure learners to improve their performance compared to current methods. 

**Abstract (ZH)**: 贝叶斯网络（BNs）是广泛应用于众多领域的一种概率图形模型。然而，从数据中推断网络的图形结构仍然具有挑战性。贝叶斯结构学习方法通过推断BN所 underlying 的有向无环图的可能性后验分布来解决这个问题。推断过程中通常需要对概率分布进行边缘化，这通常通过动态规划方法来实现，这些方法会限制每个节点的可能父节点集合。相反，我们提出了一种新颖的方法，利用可计算的概率电路来克服这一限制。该方法利用一种新的学习机制，在原始分布和边缘查询上训练这些电路。概率电路的架构本质上允许在学习的分布上进行快速且精确的边缘化。随后，我们通过实验展示了利用此方法回答边缘查询可以使贝叶斯结构学习方法优于现有方法。 

---
# Data Whitening Improves Sparse Autoencoder Learning 

**Title (ZH)**: 数据 whitening 提高稀疏自编码器学习效果 

**Authors**: Ashwin Saraswatula, David Klindt  

**Link**: [PDF](https://arxiv.org/pdf/2511.13981)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as a promising approach for learning interpretable features from neural network activations. However, the optimization landscape for SAE training can be challenging due to correlations in the input data. We demonstrate that applying PCA Whitening to input activations -- a standard preprocessing technique in classical sparse coding -- improves SAE performance across multiple metrics. Through theoretical analysis and simulation, we show that whitening transforms the optimization landscape, making it more convex and easier to navigate. We evaluate both ReLU and Top-K SAEs across diverse model architectures, widths, and sparsity regimes. Empirical evaluation on SAEBench, a comprehensive benchmark for sparse autoencoders, reveals that whitening consistently improves interpretability metrics, including sparse probing accuracy and feature disentanglement, despite minor drops in reconstruction quality. Our results challenge the assumption that interpretability aligns with an optimal sparsity--fidelity trade-off and suggest that whitening should be considered as a default preprocessing step for SAE training, particularly when interpretability is prioritized over perfect reconstruction. 

**Abstract (ZH)**: 基于PCA白化的稀疏自编码器在多个度量指标上提高了性能，特别是在神经网络激活值上应用PCA白化预处理技术改善了稀疏自编码器的表现。 

---
# Preference-Based Learning in Audio Applications: A Systematic Analysis 

**Title (ZH)**: 基于偏好学习的音频应用：系统分析 

**Authors**: Aaron Broukhim, Yiran Shen, Prithviraj Ammanabrolu, Nadir Weibel  

**Link**: [PDF](https://arxiv.org/pdf/2511.13936)  

**Abstract**: Despite the parallel challenges that audio and text domains face in evaluating generative model outputs, preference learning remains remarkably underexplored in audio applications. Through a PRISMA-guided systematic review of approximately 500 papers, we find that only 30 (6%) apply preference learning to audio tasks. Our analysis reveals a field in transition: pre-2021 works focused on emotion recognition using traditional ranking methods (rankSVM), while post-2021 studies have pivoted toward generation tasks employing modern RLHF frameworks. We identify three critical patterns: (1) the emergence of multi-dimensional evaluation strategies combining synthetic, automated, and human preferences; (2) inconsistent alignment between traditional metrics (WER, PESQ) and human judgments across different contexts; and (3) convergence on multi-stage training pipelines that combine reward signals. Our findings suggest that while preference learning shows promise for audio, particularly in capturing subjective qualities like naturalness and musicality, the field requires standardized benchmarks, higher-quality datasets, and systematic investigation of how temporal factors unique to audio impact preference learning frameworks. 

**Abstract (ZH)**: 尽管音频和文本领域在评估生成模型输出时面临着类似的挑战，但在音频应用中，偏好学习依然被明显忽视。通过遵循PRISMA指南对约500篇论文进行系统性回顾，我们发现仅有30篇（6%）研究将偏好学习应用于音频任务。我们的分析揭示了一个正在转型的领域：2021年以前的工作主要集中在使用传统排名方法（rankSVM）进行情感识别，而2021年以后的研究则转向使用现代RLHF框架进行生成任务。我们识别出三个关键模式：（1）多维度评估策略的出现，结合了合成、自动化和人工偏好的方法；（2）传统指标（WER、PESQ）与不同上下文中的人类判断之间的一致性不明显；（3）多阶段训练管道的趋同，这些管道结合了奖励信号。我们的研究结果表明，尽管偏好学习在音频领域，特别是在捕捉自然度和音乐性等主观品质方面显示出了前景，但该领域仍需标准化基准、高质量数据集，并系统性地探讨时间因素对偏好学习框架的特殊影响。 

---
# Compute-in-Memory Implementation of State Space Models for Event Sequence Processing 

**Title (ZH)**: 内存计算实现的状态空间模型在事件序列处理中的应用 

**Authors**: Xiaoyu Zhang, Mingtao Hu, Sen Lu, Soohyeon Kim, Eric Yeu-Jer Lee, Yuyang Liu, Wei D. Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13912)  

**Abstract**: State space models (SSMs) have recently emerged as a powerful framework for long sequence processing, outperforming traditional methods on diverse benchmarks. Fundamentally, SSMs can generalize both recurrent and convolutional networks and have been shown to even capture key functions of biological systems. Here we report an approach to implement SSMs in energy-efficient compute-in-memory (CIM) hardware to achieve real-time, event-driven processing. Our work re-parameterizes the model to function with real-valued coefficients and shared decay constants, reducing the complexity of model mapping onto practical hardware systems. By leveraging device dynamics and diagonalized state transition parameters, the state evolution can be natively implemented in crossbar-based CIM systems combined with memristors exhibiting short-term memory effects. Through this algorithm and hardware co-design, we show the proposed system offers both high accuracy and high energy efficiency while supporting fully asynchronous processing for event-based vision and audio tasks. 

**Abstract (ZH)**: 基于能量高效的计算内存硬件实现状态空间模型以实现实时事件驱动处理 

---
# Can QE-informed (Re)Translation lead to Error Correction? 

**Title (ZH)**: 基于QE指导的(重)翻译能否实现错误修正？ 

**Authors**: Govardhan Padmanabhan  

**Link**: [PDF](https://arxiv.org/pdf/2511.13884)  

**Abstract**: The paper presents two approaches submitted to the WMT 2025 Automated Translation Quality Evaluation Systems Task 3 - Quality Estimation (QE)-informed Segment-level Error Correction. While jointly training QE systems with Automatic Post-Editing (APE) has shown improved performance for both tasks, APE systems are still known to overcorrect the output of Machine Translation (MT), leading to a degradation in performance. We investigate a simple training-free approach - QE-informed Retranslation, and compare it with another within the same training-free paradigm. Our winning approach selects the highest-quality translation from multiple candidates generated by different LLMs. The second approach, more akin to APE, instructs an LLM to replace error substrings as specified in the provided QE explanation(s). A conditional heuristic was employed to minimise the number of edits, with the aim of maximising the Gain-to-Edit ratio. The two proposed approaches achieved a Delta COMET score of 0.0201 and -0.0108, respectively, leading the first approach to achieve the winning position on the subtask leaderboard. 

**Abstract (ZH)**: 面向WMT 2025 自动翻译质量评估系统任务3的质量估计（QE）导向的段落级错误修正的两种方法：一种无训练的自动重写方法及其应用 

---
# Randomized Controlled Trials for Conditional Access Optimization Agent 

**Title (ZH)**: 条件访问优化代理的随机控制试验 

**Authors**: James Bono, Beibei Cheng, Joaquin Lozano  

**Link**: [PDF](https://arxiv.org/pdf/2511.13865)  

**Abstract**: AI agents are increasingly deployed to automate complex enterprise workflows, yet evidence of their effectiveness in identity governance is limited. We report results from the first randomized controlled trial (RCT) evaluating an AI agent for Conditional Access (CA) policy management in Microsoft Entra. The agent assists with four high-value tasks: policy merging, Zero-Trust baseline gap detection, phased rollout planning, and user-policy alignment. In a production-grade environment, 162 identity administrators were randomly assigned to a control group (no agent) or treatment group (agent-assisted) and asked to perform these tasks. Agent access produced substantial gains: accuracy improved by 48% and task completion time decreased by 43% while holding accuracy constant. The largest benefits emerged on cognitively demanding tasks such as baseline gap detection. These findings demonstrate that purpose-built AI agents can significantly enhance both speed and accuracy in identity administration. 

**Abstract (ZH)**: AI代理在身份治理中的有效性研究：Microsoft Entra条件下访问控制策略管理的随机对照试验 

---
# Randomized Controlled Trials for Phishing Triage Agent 

**Title (ZH)**: 随机对照试验在钓鱼邮件处置代理中的应用 

**Authors**: James Bono  

**Link**: [PDF](https://arxiv.org/pdf/2511.13860)  

**Abstract**: Security operations centers (SOCs) face a persistent challenge: efficiently triaging a high volume of user-reported phishing emails while maintaining robust protection against threats. This paper presents the first randomized controlled trial (RCT) evaluating the impact of a domain-specific AI agent - the Microsoft Security Copilot Phishing Triage Agent - on analyst productivity and accuracy. Our results demonstrate that agent-augmented analysts achieved up to 6.5 times as many true positives per analyst minute and a 77% improvement in verdict accuracy compared to a control group. The agent's queue prioritization and verdict explanations were both significant drivers of efficiency. Behavioral analysis revealed that agent-augmented analysts reallocated their attention, spending 53% more time on malicious emails, and were not prone to rubber-stamping the agent's malicious verdicts. These findings offer actionable insights for SOC leaders considering AI adoption, including the potential for agents to fundamentally change the optimal allocation of SOC resources. 

**Abstract (ZH)**: 面向 phishing 邮件的专用 AI 助手对分析师生产力和准确性的影响：一项随机控制试验 

---
# ScoresActivation: A New Activation Function for Model Agnostic Global Explainability by Design 

**Title (ZH)**: ScoresActivation: 一种用于设计导向的整体可解释性的新激活函数 

**Authors**: Emanuel Covaci, Fabian Galis, Radu Balan, Daniela Zaharie, Darian Onchis  

**Link**: [PDF](https://arxiv.org/pdf/2511.13809)  

**Abstract**: Understanding the decision of large deep learning models is a critical challenge for building transparent and trustworthy systems. Although the current post hoc explanation methods offer valuable insights into feature importance, they are inherently disconnected from the model training process, limiting their faithfulness and utility. In this work, we introduce a novel differentiable approach to global explainability by design, integrating feature importance estimation directly into model training. Central to our method is the ScoresActivation function, a feature-ranking mechanism embedded within the learning pipeline. This integration enables models to prioritize features according to their contribution to predictive performance in a differentiable and end-to-end trainable manner. Evaluations across benchmark datasets show that our approach yields globally faithful, stable feature rankings aligned with SHAP values and ground-truth feature importance, while maintaining high predictive performance. Moreover, feature scoring is 150 times faster than the classical SHAP method, requiring only 2 seconds during training compared to SHAP's 300 seconds for feature ranking in the same configuration. Our method also improves classification accuracy by 11.24% with 10 features (5 relevant) and 29.33% with 16 features (5 relevant, 11 irrelevant), demonstrating robustness to irrelevant inputs. This work bridges the gap between model accuracy and interpretability, offering a scalable framework for inherently explainable machine learning. 

**Abstract (ZH)**: 理解大型深度学习模型的决策是构建透明和可信赖系统的关键挑战。尽管当前的后 hoc 解释方法提供了有价值的特征重要性见解，但它们与模型训练过程本质上是脱节的，限制了其忠实性和实用性。在本工作中，我们提出了一种新的设计导向的可微分方法，通过将特征重要性估计直接集成到模型训练中来实现全局可解释性。我们方法的核心是ScoresActivation函数，这是一种嵌入在学习管道中的特征排名机制。这种集成使得模型能够在可微分和端到端可训练的方式下根据特征对预测性能的贡献来优先排序特征。在基准数据集上的评估表明，我们的方法能够提供与SHAP值和真实特征重要性一致的、全局忠实且稳定的特征排名，同时保持高预测性能。此外，特征评分比经典的SHAP方法快150倍，训练时仅需2秒，而SHAP在同一配置下对特征排名需要300秒。我们的方法还在10个特征（5个相关）和16个特征（5个相关，11个不相关）的情况下分别提高了分类准确性11.24%和29.33%，显示出对无关输入的稳健性。本工作在模型准确性和可解释性之间架起了桥梁，提供了一种可扩展的内在可解释性机器学习框架。 

---
# GAEA: Experiences and Lessons Learned from a Country-Scale Environmental Digital Twin 

**Title (ZH)**: GAEA：国家尺度环境数字孪生的经验与启示 

**Authors**: Andreas Kamilaris, Chirag Padubidri, Asfa Jamil, Arslan Amin, Indrajit Kalita, Jyoti Harti, Savvas Karatsiolis, Aytac Guley  

**Link**: [PDF](https://arxiv.org/pdf/2511.13807)  

**Abstract**: This paper describes the experiences and lessons learned after the deployment of a country-scale environmental digital twin on the island of Cyprus for three years. This digital twin, called GAEA, contains 27 environmental geospatial services and is suitable for urban planners, policymakers, farmers, property owners, real-estate and forestry professionals, as well as insurance companies and banks that have properties in their portfolio. This paper demonstrates the power, potential, current and future challenges of geospatial analytics and environmental digital twins on a large scale. 

**Abstract (ZH)**: 本文介绍了在塞浦路斯岛上部署国家规模的环境数字孪生体三年后的体验与教训。该数字孪生体名为GAEA，包含27项环境地理空间服务，适用于城市规划师、政策制定者、农民、地产所有者、房地产和林业专业人士，以及其他拥有投资组合中财产的保险公司和银行。本文展示了大规模地理空间分析和环境数字孪生体的力量、潜力，以及当前和未来的挑战。 

---
# Passive Dementia Screening via Facial Temporal Micro-Dynamics Analysis of In-the-Wild Talking-Head Video 

**Title (ZH)**: 野生环境下对话头视频面部时空微动态分析的被动痴呆筛查 

**Authors**: Filippo Cenacchi. Longbing Cao, Mitchell McEwan, Deborah Richards  

**Link**: [PDF](https://arxiv.org/pdf/2511.13802)  

**Abstract**: We target passive dementia screening from short camera-facing talking head video, developing a facial temporal micro dynamics analysis for language free detection of early neuro cognitive change. This enables unscripted, in the wild video analysis at scale to capture natural facial behaviors, transferrable across devices, topics, and cultures without active intervention by clinicians or researchers during recording. Most existing resources prioritize speech or scripted interviews, limiting use outside clinics and coupling predictions to language and transcription. In contrast, we identify and analyze whether temporal facial kinematics, including blink dynamics, small mouth jaw motions, gaze variability, and subtle head adjustments, are sufficient for dementia screening without speech or text. By stabilizing facial signals, we convert these micro movements into interpretable facial microdynamic time series, smooth them, and summarize short windows into compact clip level statistics for screening. Each window is encoded by its activity mix (the relative share of motion across streams), thus the predictor analyzes the distribution of motion across streams rather than its magnitude, making per channel effects transparent. We also introduce YT DemTalk, a new dataset curated from publicly available, in the wild camera facing videos. It contains 300 clips (150 with self reported dementia, 150 controls) to test our model and offer a first benchmarking of the corpus. On YT DemTalk, ablations identify gaze lability and mouth/jaw dynamics as the most informative cues, and light weighted shallow classifiers could attain a dementia prediction performance of (AUROC) 0.953, 0.961 Average Precision (AP), 0.851 F1-score, and 0.857 accuracy. 

**Abstract (ZH)**: 基于短摄像头面向头部对话视频的被动痴呆筛查：Temporal面部微动态分析在语言自由检测早期神经认知变化中的应用 

---
# Synergizing Multigrid Algorithms with Vision Transformer: A Novel Approach to Enhance the Seismic Foundation Model 

**Title (ZH)**: 多级算法与视觉变换器协同优化：一种增强地震基础模型的新方法 

**Authors**: Huiwen Wu, Shuo Zhang, Yi Liu, Hongbin Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.13800)  

**Abstract**: Due to the emergency and homogenization of Artificial Intelligence (AI) technology development, transformer-based foundation models have revolutionized scientific applications, such as drug discovery, materials research, and astronomy. However, seismic data presents unique characteristics that require specialized processing techniques for pretraining foundation models in seismic contexts with high- and low-frequency features playing crucial roles. Existing vision transformers (ViTs) with sequential tokenization ignore the intrinsic pattern and fail to grasp both the high- and low-frequency seismic information efficiently and effectively. This work introduces a novel adaptive two-grid foundation model training strategy (ADATG) with Hilbert encoding specifically tailored for seismogram data, leveraging the hierarchical structures inherent in seismic data. Specifically, our approach employs spectrum decomposition to separate high- and low-frequency components and utilizes hierarchical Hilbert encoding to represent the data effectively. Moreover, observing the frequency principle observed in ViTs, we propose an adaptive training strategy that initially emphasizes coarse-level information and then progressively refines the model's focus on fine-level features. Our extensive experiments demonstrate the effectiveness and efficiency of our training methods. This research highlights the importance of data encoding and training strategies informed by the distinct characteristics of high- and low-frequency features in seismic images, ultimately contributing to the enhancement of visual seismic foundation models pretraining. 

**Abstract (ZH)**: 基于希尔伯特编码的适应性双网格基础模型训练策略（ADATG）：面向地震数据的独特特征 

---
# MAT-MPNN: A Mobility-Aware Transformer-MPNN Model for Dynamic Spatiotemporal Prediction of HIV Diagnoses in California, Florida, and New England 

**Title (ZH)**: MAT-MPNN：一种基于移动性的变换器-MPNN模型，用于加利福尼亚、佛罗里达和新英格兰地区HIV诊断的动态时空预测 

**Authors**: Zhaoxuan Wang, Weichen Kang, Yutian Han, Lingyuan Zhao, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.13797)  

**Abstract**: Human Immunodeficiency Virus (HIV) has posed a major global health challenge for decades, and forecasting HIV diagnoses continues to be a critical area of research. However, capturing the complex spatial and temporal dependencies of HIV transmission remains challenging. Conventional Message Passing Neural Network (MPNN) models rely on a fixed binary adjacency matrix that only encodes geographic adjacency, which is unable to represent interactions between non-contiguous counties. Our study proposes a deep learning architecture Mobility-Aware Transformer-Message Passing Neural Network (MAT-MPNN) framework to predict county-level HIV diagnosis rates across California, Florida, and the New England region. The model combines temporal features extracted by a Transformer encoder with spatial relationships captured through a Mobility Graph Generator (MGG). The MGG improves conventional adjacency matrices by combining geographic and demographic information. Compared with the best-performing hybrid baseline, the Transformer MPNN model, MAT-MPNN reduced the Mean Squared Prediction Error (MSPE) by 27.9% in Florida, 39.1% in California, and 12.5% in New England, and improved the Predictive Model Choice Criterion (PMCC) by 7.7%, 3.5%, and 3.9%, respectively. MAT-MPNN also achieved better results than the Spatially Varying Auto-Regressive (SVAR) model in Florida and New England, with comparable performance in California. These results demonstrate that applying mobility-aware dynamic spatial structures substantially enhances predictive accuracy and calibration in spatiotemporal epidemiological prediction. 

**Abstract (ZH)**: 基于移动性意识的变换器-消息传递神经网络（MAT-MPNN）框架预测加利福尼亚、佛罗里达和新英格兰地区的县级艾滋病毒诊断率 

---
# Modeling Fairness in Recruitment AI via Information Flow 

**Title (ZH)**: 基于信息流建模招聘AI中的公平性 

**Authors**: Mattias Brännström, Themis Dimitra Xanthopoulou, Lili Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13793)  

**Abstract**: Avoiding bias and understanding the real-world consequences of AI-supported decision-making are critical to address fairness and assign accountability. Existing approaches often focus either on technical aspects, such as datasets and models, or on high-level socio-ethical considerations - rarely capturing how these elements interact in practice. In this paper, we apply an information flow-based modeling framework to a real-world recruitment process that integrates automated candidate matching with human decision-making. Through semi-structured stakeholder interviews and iterative modeling, we construct a multi-level representation of the recruitment pipeline, capturing how information is transformed, filtered, and interpreted across both algorithmic and human components. We identify where biases may emerge, how they can propagate through the system, and what downstream impacts they may have on candidates. This case study illustrates how information flow modeling can support structured analysis of fairness risks, providing transparency across complex socio-technical systems. 

**Abstract (ZH)**: 避免偏见和理解AI支持的决策在现实生活中的后果对于解决公平性和分配责任至关重要。现有方法通常侧重于技术层面，如数据集和模型，或高层次的社科伦理考量——很少捕捉到这些要素在实际操作中的相互作用。在本文中，我们采用基于信息流的建模框架对一个实际招聘过程进行建模，该过程整合了自动化候选人匹配和人类决策。通过半结构化的利益相关者访谈和迭代建模，我们构建了一个多层级的招聘流程表示，捕捉了信息在算法和人类组件之间如何被转换、过滤和解读。我们确定了偏见可能产生的地方，它们如何在系统中传播，以及它们可能对候选人的下游影响。本案例研究展示了信息流建模如何支持对复杂社技系统中公平性风险的结构化分析，提供透明度。 

---
# XAI-Driven Deep Learning for Protein Sequence Functional Group Classification 

**Title (ZH)**: 基于XAI的深度学习在蛋白质序列功能团分类中的应用 

**Authors**: Pratik Chakraborty, Aryan Bhargava  

**Link**: [PDF](https://arxiv.org/pdf/2511.13791)  

**Abstract**: Proteins perform essential biological functions, and accurate classification of their sequences is critical for understanding structure-function relationships, enzyme mechanisms, and molecular interactions. This study presents a deep learning-based framework for functional group classification of protein sequences derived from the Protein Data Bank (PDB). Four architectures were implemented: Convolutional Neural Network (CNN), Bidirectional Long Short-Term Memory (BiLSTM), CNN-BiLSTM hybrid, and CNN with Attention. Each model was trained using k-mer integer encoding to capture both local and long-range dependencies. Among these, the CNN achieved the highest validation accuracy of 91.8%, demonstrating the effectiveness of localized motif detection. Explainable AI techniques, including Grad-CAM and Integrated Gradients, were applied to interpret model predictions and identify biologically meaningful sequence motifs. The discovered motifs, enriched in histidine, aspartate, glutamate, and lysine, represent amino acid residues commonly found in catalytic and metal-binding regions of transferase enzymes. These findings highlight that deep learning models can uncover functionally relevant biochemical signatures, bridging the gap between predictive accuracy and biological interpretability in protein sequence analysis. 

**Abstract (ZH)**: 蛋白质执行基本生物功能，准确分类其序列对于理解结构-功能关系、酶机制和分子相互作用至关重要。本研究提出了一种基于深度学习的框架，用于分类来自蛋白质数据银行（PDB）的蛋白质序列的功能组。四种架构被实现：卷积神经网络（CNN）、双向长短期记忆网络（BiLSTM）、CNN与BiLSTM的混合模型，以及带有注意机制的CNN。每个模型均使用k-mer整数编码进行训练，以捕获局部和长距离依赖关系。在这其中，CNN在验证集中取得了最高准确率91.8%，证明了局部模式检测的有效性。应用可解释的人工智能技术，如Grad-CAM和集成梯度，来解释模型预测并识别生物意义的序列模式。发现的模式富含组氨酸、天冬氨酸、谷氨酸和赖氨酸，代表常见于转移酶催化和金属结合区域的氨基酸残基。这些发现强调了深度学习模型可以揭示功能相关的生物化学特征，桥梁了预测准确性和生物可解释性之间的差距，在蛋白质序列分析中的应用。 

---
# GeoPl@ntNet: A Platform for Exploring Essential Biodiversity Variables 

**Title (ZH)**: GeoPl@ntNet：一个探索基本生物多样性变量的平台 

**Authors**: Lukas Picek, César Leblanc, Alexis Joly, Pierre Bonnet, Rémi Palard, Maximilien Servajean  

**Link**: [PDF](https://arxiv.org/pdf/2511.13790)  

**Abstract**: This paper describes GeoPl@ntNet, an interactive web application designed to make Essential Biodiversity Variables accessible and understandable to everyone through dynamic maps and fact sheets. Its core purpose is to allow users to explore high-resolution AI-generated maps of species distributions, habitat types, and biodiversity indicators across Europe. These maps, developed through a cascading pipeline involving convolutional neural networks and large language models, provide an intuitive yet information-rich interface to better understand biodiversity, with resolutions as precise as 50x50 meters. The website also enables exploration of specific regions, allowing users to select areas of interest on the map (e.g., urban green spaces, protected areas, or riverbanks) to view local species and their coverage. Additionally, GeoPl@ntNet generates comprehensive reports for selected regions, including insights into the number of protected species, invasive species, and endemic species. 

**Abstract (ZH)**: GeoPl@ntNet：一种交互式web应用，通过动态地图和数据表使关键生物多样性变量对所有人可见和易理解 

---
# Quantifying Distribution Shift in Traffic Signal Control with Histogram-Based GEH Distance 

**Title (ZH)**: 基于直方图GEH距离的交通信号控制中分布偏移量化研究 

**Authors**: Federico Taschin, Ozan K. Tonguz  

**Link**: [PDF](https://arxiv.org/pdf/2511.13785)  

**Abstract**: Traffic signal control algorithms are vulnerable to distribution shift, where performance degrades under traffic conditions that differ from those seen during design or training. This paper introduces a principled approach to quantify distribution shift by representing traffic scenarios as demand histograms and comparing them with a GEH-based distance function. The method is policy-independent, interpretable, and leverages a widely used traffic engineering statistic. We validate the approach on 20 simulated scenarios using both a NEMA actuated controller and a reinforcement learning controller (FRAP++). Results show that larger scenario distances consistently correspond to increased travel time and reduced throughput, with particularly strong explanatory power for learning-based control. Overall, this method can predict performance degradation under distribution shift better than previously published techniques. These findings highlight the utility of the proposed framework for benchmarking, training regime design, and monitoring in adaptive traffic signal control. 

**Abstract (ZH)**: 交通信号控制算法易受分布偏移的影响，当交通条件与设计或训练时的不同时，性能会下降。本文通过将交通场景表示为需求直方图，并使用基于GEH的距离函数进行比较，提出了一种基于原理的方法来量化分布偏移。该方法独立于策略，具有可解释性，并利用了广泛使用的交通工程统计量。我们在使用NEMA自适应控制器和强化学习控制器（FRAP++）模拟的20种场景中验证了该方法。结果显示，较大的场景距离始终对应更高的旅行时间和更低的通过能力，尤其是在基于学习的控制方面有很强的解释力。总体而言，该方法在预测分布偏移下的性能下降方面优于已发表的技术。这些发现强调了所提出框架在自适应交通信号控制中的基准测试、训练制度设计和监控方面的实用性。 

---
# Semantic Multiplexing 

**Title (ZH)**: 语义多复用 

**Authors**: Mohammad Abdi, Francesca Meneghello, Francesco Restuccia  

**Link**: [PDF](https://arxiv.org/pdf/2511.13779)  

**Abstract**: Mobile devices increasingly require the parallel execution of several computing tasks offloaded at the wireless edge. Existing communication systems only support parallel transmissions at the bit level, which fundamentally limits the number of tasks that can be concurrently processed. To address this bottleneck, this paper introduces the new concept of Semantic Multiplexing. Our approach shifts stream multiplexing from bits to tasks by merging multiple task-related compressed representations into a single semantic representation. As such, Semantic Multiplexing can multiplex more tasks than the number of physical channels without adding antennas or widening bandwidth by extending the effective degrees of freedom at the semantic layer, without contradicting Shannon capacity rules. We have prototyped Semantic Multiplexing on an experimental testbed with Jetson Orin Nano and millimeter-wave software-defined radios and tested its performance on image classification and sentiment analysis while comparing to several existing baselines in semantic communications. Our experiments demonstrate that Semantic Multiplexing allows jointly processing multiple tasks at the semantic level while maintaining sufficient task accuracy. For example, image classification accuracy drops by less than 4% when increasing from 2 to 8 the number of tasks multiplexed over a 4$\times$4 channel. Semantic Multiplexing reduces latency, energy consumption, and communication load respectively by up to 8$\times$, 25$\times$, and 54$\times$ compared to the baselines while keeping comparable performance. We pledge to publicly share the complete software codebase and the collected datasets for reproducibility. 

**Abstract (ZH)**: 移动设备越来越多地需要在无线边缘并行执行多个计算任务。现有的通信系统仅在位级支持并行传输，从根本上限制了可以同时处理的任务数量。为了应对这一瓶颈，本文引入了新的概念——语义复用。我们的方法将流复用从位级转向任务级，通过将多个任务相关的压缩表示合并成一个语义表示。因此，语义复用可以在不增加天线或扩展带宽的情况下，通过在语义层扩展有效的自由度来复用比物理信道更多的任务，而不违背香农容量规则。我们已经在实验测试台上使用Jetson Orin Nano和毫米波软件定义无线电原型实现语义复用，并在其上进行图像分类和情感分析性能测试，将其与语义通信中现有的几种基线进行比较。我们的实验表明，语义复用允许在语义层面上同时处理多个任务，同时保持足够的任务准确性。例如，在一个4×4信道上，将复用的任务数从2增加到8时，图像分类准确性仅下降不到4%。与基线相比，语义复用分别将延迟、能耗和通信负载降低了最多8倍、25倍和54倍，同时保持了相当的性能。我们将致力于公开分享完整的软件代码库和收集的数据集，以实现可重复性。 

---
# Known Meets Unknown: Mitigating Overconfidence in Open Set Recognition 

**Title (ZH)**: 已知遇未知：减轻开放集识别中的过度自信 

**Authors**: Dongdong Zhao, Ranxin Fang, Changtian Song, Zhihui Liu, Jianwen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13775)  

**Abstract**: Open Set Recognition (OSR) requires models not only to accurately classify known classes but also to effectively reject unknown samples. However, when unknown samples are semantically similar to known classes, inter-class overlap in the feature space often causes models to assign unjustifiably high confidence to them, leading to misclassification as known classes -- a phenomenon known as overconfidence. This overconfidence undermines OSR by blurring the decision boundary between known and unknown classes. To address this issue, we propose a framework that explicitly mitigates overconfidence caused by inter-class overlap. The framework consists of two components: a perturbation-based uncertainty estimation module, which applies controllable parameter perturbations to generate diverse predictions and quantify predictive uncertainty, and an unknown detection module with distinct learning-based classifiers, implemented as a two-stage procedure, which leverages the estimated uncertainty to improve discrimination between known and unknown classes, thereby enhancing OSR performance. Experimental results on three public datasets show that the proposed framework achieves superior performance over existing OSR methods. 

**Abstract (ZH)**: 开放集识别（OSR）要求模型不仅要准确分类已知类别，还要有效拒绝未知样本。然而，当未知样本在语义上与已知类别相似时，特征空间中的类别间重叠往往会导致模型对它们赋予不合理的高置信度，从而将其错误地分类为已知类别——这一现象被称为过自信心。过自信心会削弱OSR，模糊已知和未知类别之间的决策边界。为此，我们提出了一种框架，以明确减轻由类别间重叠引起的过自信心。该框架由两个组成部分组成：基于扰动的不确定性估计模块，该模块应用可控参数扰动以生成多样化的预测并量化预测不确定性；以及一个具有不同学习分类器的未知检测模块，该模块实现为两阶段过程，利用估计的不确定性以提高已知和未知类别的区分性，从而提升OSR性能。在三个公开数据集上的实验结果表明，所提出的框架在现有的OSR方法上表现出更好的性能。 

---
# Dynamic Temperature Scheduler for Knowledge Distillation 

**Title (ZH)**: 动态温度调度器用于知识蒸馏 

**Authors**: Sibgat Ul Islam, Jawad Ibn Ahad, Fuad Rahman, Mohammad Ruhul Amin, Nabeel Mohammed, Shafin Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2511.13767)  

**Abstract**: Knowledge Distillation (KD) trains a smaller student model using a large, pre-trained teacher model, with temperature as a key hyperparameter controlling the softness of output probabilities. Traditional methods use a fixed temperature throughout training, which is suboptimal. Moreover, architectural differences between teacher and student often result in mismatched logit magnitudes. We demonstrate that students benefit from softer probabilities early in training but require sharper probabilities in later stages. We introduce Dynamic Temperature Scheduler (DTS), which adjusts temperature dynamically based on the cross-entropy loss gap between teacher and student. To our knowledge, this is the first temperature scheduling method that adapts based on the divergence between teacher and student distributions. Our method integrates seamlessly with existing KD frameworks. We validate DTS across multiple KD strategies on vision (CIFAR-100, Tiny-ImageNet) and NLP tasks (GLUE, Dolly, SelfIns, UnNI, S-NI), consistently outperforming static-temperature baselines. Code is available at this https URL. 

**Abstract (ZH)**: 知识蒸馏（KD）使用一个大型预训练教师模型来训练一个较小的学生模型，并使用温度作为关键超参数来控制输出概率的软硬度。传统的做法在整个训练过程中使用固定的温度，这往往是次优选择。此外，教师模型和学生模型之间的架构差异通常会导致对数概率幅度不匹配。我们证明了学生在训练早期从更软的概率中受益，但在训练后期需要更硬的概率。我们引入了动态温度调度器（DTS），它根据教师模型和学生模型之间的交叉熵损失差距动态调整温度。据我们所知，这是第一个基于教师模型和学生模型分布差异进行自适应的温度调度方法。该方法无缝集成到现有的知识蒸馏框架中。我们在视觉（CIFAR-100, Tiny-ImageNet）和自然语言处理任务（GLUE, Dolly, SelfIns, UnNI, S-NI）上对多种知识蒸馏策略进行了验证，结果表明，与固定温度基线相比表现更优。代码已发布于此 <https://> 地址。 

---
# Credal Ensemble Distillation for Uncertainty Quantification 

**Title (ZH)**: 证据集合蒸馏在不确定性量化中的应用 

**Authors**: Kaizheng Wang, Fabio Cuzzolin, David Moens, Hans Hallez  

**Link**: [PDF](https://arxiv.org/pdf/2511.13766)  

**Abstract**: Deep ensembles (DE) have emerged as a powerful approach for quantifying predictive uncertainty and distinguishing its aleatoric and epistemic components, thereby enhancing model robustness and reliability. However, their high computational and memory costs during inference pose significant challenges for wide practical deployment. To overcome this issue, we propose credal ensemble distillation (CED), a novel framework that compresses a DE into a single model, CREDIT, for classification tasks. Instead of a single softmax probability distribution, CREDIT predicts class-wise probability intervals that define a credal set, a convex set of probability distributions, for uncertainty quantification. Empirical results on out-of-distribution detection benchmarks demonstrate that CED achieves superior or comparable uncertainty estimation compared to several existing baselines, while substantially reducing inference overhead compared to DE. 

**Abstract (ZH)**: 深度集成（DE）已 emerge 作为一种 Powerful 方法，用于量化预测不确定性并区分其 aleatoric 和 epistemic 组件，从而增强模型的 robustness 和可靠性。然而，其在推断过程中的高计算和内存成本为广泛的实际部署带来了重大挑战。为克服这一问题，我们提出了一种新的框架——信念集成蒸馏（CED），该框架将 DE 压缩为一个单一模型 CREDIT，用于分类任务。与单一的 softmax 概率分布不同，CREDIT 预测类别的概率区间，这些区间定义了一个概率分布的凸集合（即credal集），用于不确定性量化。基于离分布检测基准的实验证明，CED 在不确定性估计方面优于几种现有基线方法，同时相比 DE 显著降低了推断开销。 

---
# Gene Incremental Learning for Single-Cell Transcriptomics 

**Title (ZH)**: 单细胞转录组学的基因增量学习 

**Authors**: Jiaxin Qi, Yan Cui, Jianqiang Huang, Gaogang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2511.13762)  

**Abstract**: Classes, as fundamental elements of Computer Vision, have been extensively studied within incremental learning frameworks. In contrast, tokens, which play essential roles in many research fields, exhibit similar characteristics of growth, yet investigations into their incremental learning remain significantly scarce. This research gap primarily stems from the holistic nature of tokens in language, which imposes significant challenges on the design of incremental learning frameworks for them. To overcome this obstacle, in this work, we turn to a type of token, gene, for a large-scale biological dataset--single-cell transcriptomics--to formulate a pipeline for gene incremental learning and establish corresponding evaluations. We found that the forgetting problem also exists in gene incremental learning, thus we adapted existing class incremental learning methods to mitigate the forgetting of genes. Through extensive experiments, we demonstrated the soundness of our framework design and evaluations, as well as the effectiveness of our method adaptations. Finally, we provide a complete benchmark for gene incremental learning in single-cell transcriptomics. 

**Abstract (ZH)**: 基因作为生物信息学中 fundament 元素，在增量学习框架下得到了广泛研究。相比之下，尽管 tokens 在许多研究领域中发挥着重要作用，并表现出类似的增长特性，但针对它们的增量学习研究仍然极为稀缺。这一研究缺口主要源于 tokens 在语言中的整体性质，这为它们设计增量学习框架带来了显著的挑战。为克服这一障碍，本研究转向大型生物数据集——单细胞转录组学中的基因，以构建基因增量学习的管线并建立相应的评估。我们发现基因增量学习中也存在遗忘问题，因此我们适应现有的类别增量学习方法来减轻基因的遗忘问题。通过广泛的实验，我们展示了我们框架设计和评估的稳健性，以及我们方法改进的有效性。最后，我们提供了一个完整的单细胞转录组学中基因增量学习基准。 

---
# MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with MoE-LayerNorm 

**Title (ZH)**: MoETTA: 采用MoE-LayerNorm在混合分布偏移下的测试时自适应 

**Authors**: Xiao Fan, Jingyan Jiang, Zhaoru Chen, Fanding Huang, Xiao Chen, Qinting Jiang, Bowen Zhang, Xing Tang, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13760)  

**Abstract**: Test-Time adaptation (TTA) has proven effective in mitigating performance drops under single-domain distribution shifts by updating model parameters during inference. However, real-world deployments often involve mixed distribution shifts, where test samples are affected by diverse and potentially conflicting domain factors, posing significant challenges even for SOTA TTA methods. A key limitation in existing approaches is their reliance on a unified adaptation path, which fails to account for the fact that optimal gradient directions can vary significantly across different domains. Moreover, current benchmarks focus only on synthetic or homogeneous shifts, failing to capture the complexity of real-world heterogeneous mixed distribution shifts. To address this, we propose MoETTA, a novel entropy-based TTA framework that integrates the Mixture-of-Experts (MoE) architecture. Rather than enforcing a single parameter update rule for all test samples, MoETTA introduces a set of structurally decoupled experts, enabling adaptation along diverse gradient directions. This design allows the model to better accommodate heterogeneous shifts through flexible and disentangled parameter updates. To simulate realistic deployment conditions, we introduce two new benchmarks: potpourri and potpourri+. While classical settings focus solely on synthetic corruptions, potpourri encompasses a broader range of domain shifts--including natural, artistic, and adversarial distortions--capturing more realistic deployment challenges. Additionally, potpourri+ further includes source-domain samples to evaluate robustness against catastrophic forgetting. Extensive experiments across three mixed distribution shifts settings show that MoETTA consistently outperforms strong baselines, establishing SOTA performance and highlighting the benefit of modeling multiple adaptation directions via expert-level diversity. 

**Abstract (ZH)**: 基于混合分布切换的MoETTA：一种新型熵导向的测试时自适应框架 

---
# Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection 

**Title (ZH)**: 多代理VLMs引导的自训练在PNU损失函数下的低资源 Offensive内容检测 

**Authors**: Han Wang, Deyi Ji, Junyu Lu, Lanyun Zhu, Hailong Zhang, Haiyang Wu, Liqun Liu, Peng Shu, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.13759)  

**Abstract**: Accurate detection of offensive content on social media demands high-quality labeled data; however, such data is often scarce due to the low prevalence of offensive instances and the high cost of manual annotation. To address this low-resource challenge, we propose a self-training framework that leverages abundant unlabeled data through collaborative pseudo-labeling. Starting with a lightweight classifier trained on limited labeled data, our method iteratively assigns pseudo-labels to unlabeled instances with the support of Multi-Agent Vision-Language Models (MA-VLMs). Un-labeled data on which the classifier and MA-VLMs agree are designated as the Agreed-Unknown set, while conflicting samples form the Disagreed-Unknown set. To enhance label reliability, MA-VLMs simulate dual perspectives, moderator and user, capturing both regulatory and subjective viewpoints. The classifier is optimized using a novel Positive-Negative-Unlabeled (PNU) loss, which jointly exploits labeled, Agreed-Unknown, and Disagreed-Unknown data while mitigating pseudo-label noise. Experiments on benchmark datasets demonstrate that our framework substantially outperforms baselines under limited supervision and approaches the performance of large-scale models 

**Abstract (ZH)**: 社交媒体中精确检测 offensive 内容需要高质量标注数据；然而，由于 offensive 实例出现频率低且人工标注成本高，这类数据往往稀缺。为应对这一低资源挑战，我们提出一种自训练框架，该框架通过协作伪标签利用丰富的未标注数据。我们的方法从少量标注数据训练的轻量级分类器开始，迭代为未标注实例分配伪标签，并借助多代理视觉-语言模型（MA-VLM）的支持。分类器和 MA-VLMs 同意的未标注数据被标记为一致未知集，而分歧样本则构成分歧未知集。为了提高标签可靠性，MA-VLMs 模拟双视角，即管理员和用户，捕捉监管和主观视角。我们使用一种新颖的正样本-负样本-未标注（PNU）损失函数优化分类器，该损失函数同时利用标注、一致未知和分歧未知数据，同时减少伪标签噪声。在基准数据集上的实验表明，在有限监督下，我们的框架显著优于基线模型，并接近大规模模型的性能。 

---
# ChemFixer: Correcting Invalid Molecules to Unlock Previously Unseen Chemical Space 

**Title (ZH)**: ChemFixer: 修正无效分子以解锁先前未见的化学空间 

**Authors**: Jun-Hyoung Park, Ho-Jun Song, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.13758)  

**Abstract**: Deep learning-based molecular generation models have shown great potential in efficiently exploring vast chemical spaces by generating potential drug candidates with desired properties. However, these models often produce chemically invalid molecules, which limits the usable scope of the learned chemical space and poses significant challenges for practical applications. To address this issue, we propose ChemFixer, a framework designed to correct invalid molecules into valid ones. ChemFixer is built on a transformer architecture, pre-trained using masking techniques, and fine-tuned on a large-scale dataset of valid/invalid molecular pairs that we constructed. Through comprehensive evaluations across diverse generative models, ChemFixer improved molecular validity while effectively preserving the chemical and biological distributional properties of the original outputs. This indicates that ChemFixer can recover molecules that could not be previously generated, thereby expanding the diversity of potential drug candidates. Furthermore, ChemFixer was effectively applied to a drug-target interaction (DTI) prediction task using limited data, improving the validity of generated ligands and discovering promising ligand-protein pairs. These results suggest that ChemFixer is not only effective in data-limited scenarios, but also extensible to a wide range of downstream tasks. Taken together, ChemFixer shows promise as a practical tool for various stages of deep learning-based drug discovery, enhancing molecular validity and expanding accessible chemical space. 

**Abstract (ZH)**: 基于深度学习的分子生成模型已表现出高效探索广阔化学空间的潜力，通过生成具有 Desired Properties 的潜在药物候选物。然而，这些模型经常生成化学上无效的分子，这限制了学习到的化学空间的可用范围，并对实际应用提出了重大挑战。为解决这一问题，我们提出了一种名为 ChemFixer 的框架，旨在将无效分子纠正为有效分子。ChemFixer 利用变压器架构构建，通过掩码技术预训练，并在我们构建的大量有效/无效分子pair数据集上进行微调。经过全面评估，ChemFixer 提高了分子的有效性同时有效地保留了原始输出的化学和生物学分布特性。这表明 ChemFixer 可以恢复之前无法生成的分子，从而扩展潜在药物候选物的多样性。此外，ChemFixer 在使用有限数据进行药物-靶点相互作用预测任务中得到了有效应用，提高了生成配体的有效性并发现了有前景的配体-蛋白对。这些结果表明，ChemFixer 不仅在数据受限场景中有效，还能扩展到广泛下游任务。总体而言，ChemFixer 显示出作为基于深度学习药物发现各阶段实用工具的潜力，提升分子有效性并扩展可访问的化学空间。 

---
# Multi-Horizon Time Series Forecasting of non-parametric CDFs with Deep Lattice Networks 

**Title (ZH)**: 非参数CDF的多步时序预测深格网网络 

**Authors**: Niklas Erdmann, Lars Bentsen, Roy Stenbro, Heine Nygard Riise, Narada Dilp Warakagoda, Paal E. Engelstad  

**Link**: [PDF](https://arxiv.org/pdf/2511.13756)  

**Abstract**: Probabilistic forecasting is not only a way to add more information to a prediction of the future, but it also builds on weaknesses in point prediction. Sudden changes in a time series can still be captured by a cumulative distribution function (CDF), while a point prediction is likely to miss it entirely. The modeling of CDFs within forecasts has historically been limited to parametric approaches, but due to recent advances, this no longer has to be the case. We aim to advance the fields of probabilistic forecasting and monotonic networks by connecting them and propose an approach that permits the forecasting of implicit, complete, and nonparametric CDFs. For this purpose, we propose an adaptation to deep lattice networks (DLN) for monotonically constrained simultaneous/implicit quantile regression in time series forecasting. Quantile regression usually produces quantile crossovers, which need to be prevented to achieve a legitimate CDF. By leveraging long short term memory units (LSTM) as the embedding layer, and spreading quantile inputs to all sub-lattices of a DLN with an extended output size, we can produce a multi-horizon forecast of an implicit CDF due to the monotonic constraintability of DLNs that prevent quantile crossovers. We compare and evaluate our approach's performance to relevant state of the art within the context of a highly relevant application of time series forecasting: Day-ahead, hourly forecasts of solar irradiance observations. Our experiments show that the adaptation of a DLN performs just as well or even better than an unconstrained approach. Further comparison of the adapted DLN against a scalable monotonic neural network shows that our approach performs better. With this adaptation of DLNs, we intend to create more interest and crossover investigations in techniques of monotonic neural networks and probabilistic forecasting. 

**Abstract (ZH)**: 概率预测不仅能够为对未来预测增加更多信息，还能弥补点预测的不足。时间序列中的突变仍可被累积分布函数（CDF）捕捉，而点预测很可能完全错过这些突变。历史上来讲，预测中的CDF建模通常局限于参数方法，但得益于近期的进步，这种情况不再必然。我们旨在通过结合概率预测和单调网络来推进这两个领域，并提出一种方法，允许预测隐含的、完整的和非参数的CDF。为此，我们提出了一种深度格网网络（DLN）的适应方法，用于时间序列预测中的单调约束同时/隐式分位数回归。分位数回归通常会产生分位数交叉，必须防止这种现象以生成合法的CDF。通过利用长短期记忆单元（LSTM）作为嵌入层，并将分位数输入扩展到DLN的所有子格网，我们可以在确保DLN的单调约束防止分位数交叉的情况下，生成多时段的隐含CDF的预测。我们在高相关的时序预测应用：一天前每小时的太阳能辐照度观测预测中，将我们的方法与其他相关最先进的方法进行了比较和评估。实验表明，适应后的DLN与无约束方法表现相当，甚至更好。进一步将适应后的DLN与可扩展的单调神经网络进行比较，证明了我们方法的优势。通过这种DLN的适应，我们希望在单调神经网络技术和概率预测方法之间创造更多的兴趣和交叉研究。 

---
# Motor Imagery Classification Using Feature Fusion of Spatially Weighted Electroencephalography 

**Title (ZH)**: 基于空间加权脑电特征融合的 MOTOR IMAGERY 分类 

**Authors**: Abdullah Al Shiam, Md. Khademul Islam Molla, Abu Saleh Musa Miah, Md. Abdus Samad Kamal  

**Link**: [PDF](https://arxiv.org/pdf/2511.13752)  

**Abstract**: A Brain Computer Interface (BCI) connects the human brain to the outside world, providing a direct communication channel. Electroencephalography (EEG) signals are commonly used in BCIs to reflect cognitive patterns related to motor function activities. However, due to the multichannel nature of EEG signals, explicit information processing is crucial to lessen computational complexity in BCI systems. This study proposes an innovative method based on brain region-specific channel selection and multi-domain feature fusion to improve classification accuracy. The novelty of the proposed approach lies in region-based channel selection, where EEG channels are grouped according to their functional relevance to distinct brain regions. By selecting channels based on specific regions involved in motor imagery (MI) tasks, this technique eliminates irrelevant channels, reducing data dimensionality and improving computational efficiency. This also ensures that the extracted features are more reflective of the brain actual activity related to motor tasks. Three distinct feature extraction methods Common Spatial Pattern (CSP), Fuzzy C-means clustering, and Tangent Space Mapping (TSM), are applied to each group of channels based on their brain region. Each method targets different characteristics of the EEG signal: CSP focuses on spatial patterns, Fuzzy C means identifies clusters within the data, and TSM captures non-linear patterns in the signal. The combined feature vector is used to classify motor imagery tasks (left hand, right hand, and right foot) using Support Vector Machine (SVM). The proposed method was validated on publicly available benchmark EEG datasets (IVA and I) from the BCI competition III and IV. The results show that the approach outperforms existing methods, achieving classification accuracies of 90.77% and 84.50% for datasets IVA and I, respectively. 

**Abstract (ZH)**: 一种基于脑区特异性通道选择和多域特征融合的脑机接口方法 

---
# SCALEX: Scalable Concept and Latent Exploration for Diffusion Models 

**Title (ZH)**: SCALEX：可扩展的概念和潜在空间探索在扩散模型中 

**Authors**: E. Zhixuan Zeng, Yuhao Chen, Alexander Wong  

**Link**: [PDF](https://arxiv.org/pdf/2511.13750)  

**Abstract**: Image generation models frequently encode social biases, including stereotypes tied to gender, race, and profession. Existing methods for analyzing these biases in diffusion models either focus narrowly on predefined categories or depend on manual interpretation of latent directions. These constraints limit scalability and hinder the discovery of subtle or unanticipated patterns.
We introduce SCALEX, a framework for scalable and automated exploration of diffusion model latent spaces. SCALEX extracts semantically meaningful directions from H-space using only natural language prompts, enabling zero-shot interpretation without retraining or labelling. This allows systematic comparison across arbitrary concepts and large-scale discovery of internal model associations. We show that SCALEX detects gender bias in profession prompts, ranks semantic alignment across identity descriptors, and reveals clustered conceptual structure without supervision. By linking prompts to latent directions directly, SCALEX makes bias analysis in diffusion models more scalable, interpretable, and extensible than prior approaches. 

**Abstract (ZH)**: 图像生成模型经常编码社会偏见，包括与性别、种族和职业相关的刻板印象。现有方法在分析这些偏见时要么专注于预先定义的类别，要么依赖于对潜在方向的手动解释。这些限制限制了可扩展性并阻碍了对微妙或未预见模式的发现。

我们提出了SCALEX框架，一种对扩散模型潜在空间进行可扩展和自动化探索的方法。SCALEX仅使用自然语言提示从H空间中提取具有语义意义的方向，从而实现零样本解释，无需重新训练或标注。这使得对任意概念进行系统比较成为可能，并大规模发现内部模型关联。我们展示了SCALEX在职业提示中检测性别偏见、在身份描述符间排名语义对齐以及在无监督的情况下揭示概念结构集群。通过直接将提示与潜在方向关联起来，SCALEX使扩散模型中的偏见分析相比此前方法更具有可扩展性、可解释性和可扩展性。 

---
# DeepDefense: Layer-Wise Gradient-Feature Alignment for Building Robust Neural Networks 

**Title (ZH)**: DeepDefense: 层级梯度-特征对齐构建稳健的神经网络 

**Authors**: Ci Lin, Tet Yeap, Iluju Kiringa, Biwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13749)  

**Abstract**: Deep neural networks are known to be vulnerable to adversarial perturbations, which are small and carefully crafted inputs that lead to incorrect predictions. In this paper, we propose DeepDefense, a novel defense framework that applies Gradient-Feature Alignment (GFA) regularization across multiple layers to suppress adversarial vulnerability. By aligning input gradients with internal feature representations, DeepDefense promotes a smoother loss landscape in tangential directions, thereby reducing the model's sensitivity to adversarial noise.
We provide theoretical insights into how adversarial perturbation can be decomposed into radial and tangential components and demonstrate that alignment suppresses loss variation in tangential directions, where most attacks are effective. Empirically, our method achieves significant improvements in robustness across both gradient-based and optimization-based attacks. For example, on CIFAR-10, CNN models trained with DeepDefense outperform standard adversarial training by up to 15.2% under APGD attacks and 24.7% under FGSM attacks. Against optimization-based attacks such as DeepFool and EADEN, DeepDefense requires 20 to 30 times higher perturbation magnitudes to cause misclassification, indicating stronger decision boundaries and a flatter loss landscape. Our approach is architecture-agnostic, simple to implement, and highly effective, offering a promising direction for improving the adversarial robustness of deep learning models. 

**Abstract (ZH)**: 深度神经网络known to be vulnerable to adversarial perturbations，这些是小且精心构造的输入，会导致错误预测。在本文中，我们提出了一种新颖的防御框架DeepDefense，该框架在多层中应用梯度-特征对齐（GFA）正则化以抑制对抗性漏洞。通过梯度输入与内部特征表示对齐，DeepDefense促进沿切向方向更平滑的损失景观，从而降低模型对对抗噪声的敏感性。 

---
# Review of Passenger Flow Modelling Approaches Based on a Bibliometric Analysis 

**Title (ZH)**: 基于文献计量分析的乘客流模型研究综述 

**Authors**: Jonathan Hecht, Weilian Li, Ziyue Li, Youness Dehbi  

**Link**: [PDF](https://arxiv.org/pdf/2511.13742)  

**Abstract**: This paper presents a bibliometric analysis of the field of short-term passenger flow forecasting within local public transit, covering 814 publications that span from 1984 to 2024. In addition to common bibliometric analysis tools, a variant of a citation network was developed, and topic modelling was conducted. The analysis reveals that research activity exhibited sporadic patterns prior to 2008, followed by a marked acceleration, characterised by a shift from conventional statistical and machine learning methodologies (e.g., ARIMA, SVM, and basic neural networks) to specialised deep learning architectures. Based on this insight, a connection to more general fields such as machine learning and time series modelling was established. In addition to modelling, spatial, linguistic, and modal biases were identified and findings from existing secondary literature were validated and quantified. This revealed existing gaps, such as constrained data fusion, open (multivariate) data, and underappreciated challenges related to model interpretability, cost-efficiency, and a balance between algorithmic performance and practical deployment considerations. In connection with the superordinate fields, the growth in relevance of foundation models is also noteworthy. 

**Abstract (ZH)**: 本文对地方公共交通领域短期乘客流量forecasting的文献进行了bibliometric分析，涵盖了从1984年到2024年的814篇出版物。除了使用常规的bibliometric分析工具，还开发了引文网络变体，并进行了主题建模。分析揭示了在2008年之前研究活动呈现出断续模式，随后出现显著加速，特征是从传统的统计和机器学习方法（如ARIMA、SVM和基础神经网络）转向专门的深度学习架构。基于此见解，建立了与更广泛领域如机器学习和时间序列建模的联系。除了建模，还识别了空间、语言和模式偏见，并验证和量化了现有次级文献的研究成果。这揭示了现有空白，如数据融合受限、开放（多变量）数据以及模型可解释性、成本效益和算法性能与实际部署考虑之间的平衡不足的问题。与上位领域相关，值得注意的是，基础模型的重要性也在增长。 

---
# Subject-Independent Imagined Speech Detection via Cross-Subject Generalization and Calibration 

**Title (ZH)**: 基于跨被试泛化和校准的独立于主体的想象言语检测 

**Authors**: Byung-Kwan Ko, Soowon Kim, Seo-Hyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.13739)  

**Abstract**: Achieving robust generalization across individuals remains a major challenge in electroencephalogram based imagined speech decoding due to substantial variability in neural activity patterns. This study examined how training dynamics and lightweight subject specific adaptation influence cross subject performance in a neural decoding framework. A cyclic inter subject training approach, involving shorter per subject training segments and frequent alternation among subjects, led to modest yet consistent improvements in decoding performance across unseen target data. Furthermore, under the subject calibrated leave one subject out scheme, incorporating only 10 % of the target subjects data for calibration achieved an accuracy of 0.781 and an AUC of 0.801, demonstrating the effectiveness of few shot adaptation. These findings suggest that integrating cyclic training with minimal calibration provides a simple and effective strategy for developing scalable, user adaptive brain computer interface systems that balance generalization and personalization. 

**Abstract (ZH)**: 基于电生理信号的想象语音解码中实现跨个体鲁棒泛化仍是一项重大挑战，由于神经活动模式存在显著差异。本研究探讨了训练动力学和轻量级个体特定适应如何影响神经解码框架下的跨个体性能。通过涉及较短的个体训练段和频繁的主体交替的一种循环跨个体训练方法，在未见过的目标数据上实现了解码性能的温和但持续的提升。此外，在个体校准的“留一主体外”方案下，仅使用目标主体数据的10%进行校准，达到了0.781的准确率和0.801的AUC值，展示了少量样本adaptation的有效性。这些发现表明，将循环训练与最小化校准相结合提供了一种简单且有效的策略，用于开发可扩展且用户自适应的大脑计算机接口系统，平衡泛化与个性化。 

---
# DualLaguerreNet: A Decoupled Spectral Filter GNN and the Uncovering of the Flexibility-Stability Trade-off 

**Title (ZH)**: DualLaguerreNet: 一个解耦的谱过滤GNN和柔性-稳定性权衡的揭示 

**Authors**: Huseyin Goksu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13729)  

**Abstract**: Graph Neural Networks (GNNs) based on spectral filters, such as the Adaptive Orthogonal Polynomial Filter (AOPF) class (e.g., LaguerreNet), have shown promise in unifying the solutions for heterophily and over-smoothing. However, these single-filter models suffer from a "compromise" problem, as their single adaptive parameter (e.g., alpha) must learn a suboptimal, averaged response across the entire graph spectrum. In this paper, we propose DualLaguerreNet, a novel GNN architecture that solves this by introducing "Decoupled Spectral Flexibility." DualLaguerreNet splits the graph Laplacian into two operators, L_low (low-frequency) and L_high (high-frequency), and learns two independent, adaptive Laguerre polynomial filters, parameterized by alpha_1 and alpha_2, respectively. This work, however, uncovers a deeper finding. While our experiments show DualLaguerreNet's flexibility allows it to achieve state-of-the-art results on complex heterophilic tasks (outperforming LaguerreNet), it simultaneously underperforms on simpler, homophilic tasks. We identify this as a fundamental "Flexibility-Stability Trade-off". The increased parameterization (2x filter parameters and 2x model parameters) leads to overfitting on simple tasks, demonstrating that the "compromise" of simpler models acts as a crucial regularizer. This paper presents a new SOTA architecture for heterophily while providing a critical analysis of the bias-variance trade-off inherent in adaptive GNN filter design. 

**Abstract (ZH)**: 基于谱滤波器的图神经网络（GNNs），如适应性正交多项式滤波器（AOPF）类（例如LaguerreNet），在统一异质性和过度平滑的解决方案方面显示出潜力。然而，这些单滤波器模型存在一个“妥协”问题，因为它们的单一适应参数（例如alpha）必须在整个图谱上学习一个次优的、平均的响应。本文提出了一种新颖的GNN架构DualLaguerreNet，通过引入“解耦的谱灵活性”来解决这一问题。DualLaguerreNet将图拉普拉斯算子划分为两个操作符L_low（低频）和L_high（高频），并学习两个独立的、适应性的拉普拉斯多项式滤波器，分别参数化为alpha_1和alpha_2。然而，本项工作揭示了一个更深层次的发现。尽管我们的实验表明，DualLaguerreNet的灵活性使其在复杂的异质性任务上达到了最先进的性能（超越了LaguerreNet），但在更简单的同质性任务上表现不佳。我们将其识别为一种根本性的“灵活性-稳定性权衡”。增加的参数化（2倍的滤波器参数和2倍的模型参数）导致在简单任务上过拟合，表明更简单的模型的“妥协”起到了关键的正则化作用。本文提出了一种最新的异质性架构，同时对自适应GNN滤波器设计固有的偏差-方差权衡进行了关键分析。 

---
# Refine Thought: A Test-Time Inference Method for Embedding Model Reasoning 

**Title (ZH)**: 精炼思维：一种嵌入模型推理的测试时推理方法 

**Authors**: Guangzhi Wang, Kai Li, Yinghao Jiao, Zhi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13726)  

**Abstract**: We propose RT (Refine Thought), a method that can enhance the semantic rea-soning ability of text embedding models. The method obtains the final semanticrepresentation by running multiple forward passes of the text embedding this http URL show that RT achieves significant improvements on semantic reason-ing tasks in BRIGHT and the person job matching benchmark PJBenchmark1, while maintaining consistent performance on general-purpose semantic under-standing tasks such as C-MTEB. Our results indicate that RT is effective becauseit further activates the semantic reasoning ability learned during pretraining bydecoder-only text embedding models(e.g., Qwen3-Embedding-8B). RT canbe seen as a test-time inference method. 

**Abstract (ZH)**: 我们提出RT（Refine Thought），一种能够增强文本嵌入模型语义推理能力的方法。该方法通过多次运行文本嵌入的前向传播过程获得最终的语义表示。实验结果显示，RT在BRIGHT和人职匹配基准PJ Benchmark1上的语义推理任务中实现了显著的改进，同时在如C-MTEB等通用语义理解任务中保持了一致的表现。我们的结果表明，RT有效的原因在于它进一步激活了解码器-only文本嵌入模型（例如Qwen3-Embedding-8B）在预训练期间学习到的语义推理能力。RT可以被视为一种测试时的推理方法。 

---
# Preparation Meets Opportunity: Enhancing Data Preprocessing for ML Training With Seneca 

**Title (ZH)**: 准备成就机遇：通过Seneca增强ML训练数据预处理 

**Authors**: Omkar Desai, Ziyang Jiao, Shuyi Pei, Janki Bhimani, Bryan S. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.13724)  

**Abstract**: Input data preprocessing is a common bottleneck when concurrently training multimedia machine learning (ML) models in modern systems. To alleviate these bottlenecks and reduce the training time for concurrent jobs, we present Seneca, a data loading system that optimizes cache partitioning and data sampling for the data storage and ingestion (DSI) pipeline. The design of Seneca contains two key techniques. First, Seneca uses a performance model for the data pipeline to optimally partition the cache for three different forms of data (encoded, decoded, and augmented). Second, Seneca opportunistically serves cached data over uncached ones during random batch sampling so that concurrent jobs benefit from each other. We implement Seneca by modifying PyTorch and demonstrate its effectiveness by comparing it against several state-of-the-art caching systems for DNN training. Seneca reduces the makespan by 45.23% compared to PyTorch and increases data processing throughput by up to 3.45x compared to the next best dataloader. 

**Abstract (ZH)**: 多媒体机器学习模型现代系统中并发训练时的数据预处理是一个常见的瓶颈。为了缓解这些瓶颈并减少并发作业的训练时间，我们提出了Seneca，一种优化数据存储和 ingestion (DSI) 管道中的缓存分区和数据采样的数据加载系统。Seneca的设计包含两项关键技术。首先，Seneca使用数据管道性能模型来最优地为三种不同形式的数据（编码、解码和增强）分区缓存。其次，Seneca在随机批次采样时机会性地提供缓存数据而非未缓存数据，从而使并发作业互惠受益。我们通过修改PyTorch实现Seneca，并通过将其与几种最新的DNN训练缓存系统进行比较，展示了其有效性。与PyTorch相比，Seneca将总体执行时间缩短了45.23%，并与下一个最佳的数据加载器相比，数据处理吞吐量提高了高达3.45倍。 

---
