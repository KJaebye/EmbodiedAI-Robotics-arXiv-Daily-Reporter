# Scalable Offline Metrics for Autonomous Driving 

**Title (ZH)**: 可扩展的自主驾驶离线指标 

**Authors**: Animikh Aich, Adwait Kulkarni, Eshed Ohn-Bar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08571)  

**Abstract**: Real-World evaluation of perception-based planning models for robotic systems, such as autonomous vehicles, can be safely and inexpensively conducted offline, i.e., by computing model prediction error over a pre-collected validation dataset with ground-truth annotations. However, extrapolating from offline model performance to online settings remains a challenge. In these settings, seemingly minor errors can compound and result in test-time infractions or collisions. This relationship is understudied, particularly across diverse closed-loop metrics and complex urban maneuvers. In this work, we revisit this undervalued question in policy evaluation through an extensive set of experiments across diverse conditions and metrics. Based on analysis in simulation, we find an even worse correlation between offline and online settings than reported by prior studies, casting doubts on the validity of current evaluation practices and metrics for driving policies. Next, we bridge the gap between offline and online evaluation. We investigate an offline metric based on epistemic uncertainty, which aims to capture events that are likely to cause errors in closed-loop settings. The resulting metric achieves over 13% improvement in correlation compared to previous offline metrics. We further validate the generalization of our findings beyond the simulation environment in real-world settings, where even greater gains are observed. 

**Abstract (ZH)**: 基于感知的规划模型在机器人系统中的实际评估可以通过离线计算模型预测误差来安全且经济地进行，但将离线模型性能 extrapolate 到在线设置仍然是一项挑战。在这些设置中，看似轻微的错误可能会积累并导致测试时的违规或碰撞。这一关系在不同的闭环度量和复杂的城市机动中被忽视。在这项工作中，我们通过一系列广泛实验重新审视这一被低估的问题，分析表明，离线和在线设置之间的相关性比先前研究报道的更差，这引发了对当前评估实践和驾驶策略度量有效性的质疑。接下来，我们弥合了离线和在线评估之间的差距。我们研究了一种基于证伪不确定性（epistemic uncertainty）的离线度量，旨在捕捉可能导致闭环环境中错误的事件。该度量在相关性上相比之前的离线度量实现了超过13%的改进。我们进一步验证了我们在模拟环境之外的实际环境中的一般性结论，在这些环境中观察到更大的益处。 

---
# Reliability of Single-Level Equality-Constrained Inverse Optimal Control 

**Title (ZH)**: 单水平约束最优控制的可靠性 

**Authors**: Filip Bečanović, Kosta Jovanović, Vincent Bonnet  

**Link**: [PDF](https://arxiv.org/pdf/2510.08406)  

**Abstract**: Inverse optimal control (IOC) allows the retrieval of optimal cost function weights, or behavioral parameters, from human motion. The literature on IOC uses methods that are either based on a slow bilevel process or a fast but noise-sensitive minimization of optimality condition violation. Assuming equality-constrained optimal control models of human motion, this article presents a faster but robust approach to solving IOC using a single-level reformulation of the bilevel method and yields equivalent results. Through numerical experiments in simulation, we analyze the robustness to noise of the proposed single-level reformulation to the bilevel IOC formulation with a human-like planar reaching task that is used across recent studies. The approach shows resilience to very large levels of noise and reduces the computation time of the IOC on this task by a factor of 15 when compared to a classical bilevel implementation. 

**Abstract (ZH)**: 基于单层改革的逆最优控制：快速且鲁棒的人类运动最优成本函数权重检索方法 

---
# Accurate and Noise-Tolerant Extraction of Routine Logs in Robotic Process Automation (Extended Version) 

**Title (ZH)**: 准确且抗噪声的机器人过程自动化常规日志提取（扩展版） 

**Authors**: Massimiliano de Leoni, Faizan Ahmed Khan, Simone Agostinelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.08118)  

**Abstract**: Robotic Process Mining focuses on the identification of the routine types performed by human resources through a User Interface. The ultimate goal is to discover routine-type models to enable robotic process automation. The discovery of routine-type models requires the provision of a routine log. Unfortunately, the vast majority of existing works do not directly focus on enabling the model discovery, limiting themselves to extracting the set of actions that are part of the routines. They were also not evaluated in scenarios characterized by inconsistent routine execution, hereafter referred to as noise, which reflects natural variability and occasional errors in human performance. This paper presents a clustering-based technique that aims to extract routine logs. Experiments were conducted on nine UI logs from the literature with different levels of injected noise. Our technique was compared with existing techniques, most of which are not meant to discover routine logs but were adapted for the purpose. The results were evaluated through standard state-of-the-art metrics, showing that we can extract more accurate routine logs than what the state of the art could, especially in the presence of noise. 

**Abstract (ZH)**: 基于机器人流程挖掘：通过用户界面识别常规任务类型以发现常规类型模型 

---
# Orientation Learning and Adaptation towards Simultaneous Incorporation of Multiple Local Constraints 

**Title (ZH)**: 面向多局部约束同时整合的定向学习与适应 

**Authors**: Gaofeng Li, Peisen Xu, Ruize Wang, Qi Ye, Jiming Chen, Dezhen Song, Yanlong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07986)  

**Abstract**: Orientation learning plays a pivotal role in many tasks. However, the rotation group SO(3) is a Riemannian manifold. As a result, the distortion caused by non-Euclidean geometric nature introduces difficulties to the incorporation of local constraints, especially for the simultaneous incorporation of multiple local constraints. To address this issue, we propose the Angle-Axis Space-based orientation representation method to solve several orientation learning problems, including orientation adaptation and minimization of angular acceleration. Specifically, we propose a weighted average mechanism in SO(3) based on the angle-axis representation method. Our main idea is to generate multiple trajectories by considering different local constraints at different basepoints. Then these multiple trajectories are fused to generate a smooth trajectory by our proposed weighted average mechanism, achieving the goal to incorporate multiple local constraints simultaneously. Compared with existing solution, ours can address the distortion issue and make the off-theshelf Euclidean learning algorithm be re-applicable in non-Euclidean space. Simulation and Experimental evaluations validate that our solution can not only adapt orientations towards arbitrary desired via-points and cope with angular acceleration constraints, but also incorporate multiple local constraints simultaneously to achieve extra benefits, e.g., achieving smaller acceleration costs. 

**Abstract (ZH)**: 基于Angle-Axis Space的方向学习方法及其实现多重局部约束的研究 

---
# Injecting Hallucinations in Autonomous Vehicles: A Component-Agnostic Safety Evaluation Framework 

**Title (ZH)**: 在自主车辆中注入幻觉：一种不依赖组件的安全评估框架 

**Authors**: Alexandre Moreira Nascimento, Gabriel Kenji Godoy Shimanuki, Lúcio Flavio Vismari, João Batista Camargo Jr, Jorge Rady de Almeida Jr, Paulo Sergio Cugnasca, Anna Carolina Muller Queiroz, Jeremy Noah Bailenson  

**Link**: [PDF](https://arxiv.org/pdf/2510.07749)  

**Abstract**: Perception failures in autonomous vehicles (AV) remain a major safety concern because they are the basis for many accidents. To study how these failures affect safety, researchers typically inject artificial faults into hardware or software components and observe the outcomes. However, existing fault injection studies often target a single sensor or machine perception (MP) module, resulting in siloed frameworks that are difficult to generalize or integrate into unified simulation environments. This work addresses that limitation by reframing perception failures as hallucinations, false perceptions that distort an AV situational awareness and may trigger unsafe control actions. Since hallucinations describe only observable effects, this abstraction enables analysis independent of specific sensors or algorithms, focusing instead on how their faults manifest along the MP pipeline. Building on this concept, we propose a configurable, component-agnostic hallucination injection framework that induces six plausible hallucination types in an iterative open-source simulator. More than 18,350 simulations were executed in which hallucinations were injected while AVs crossed an unsignalized transverse street with traffic. The results statistically validate the framework and quantify the impact of each hallucination type on collisions and near misses. Certain hallucinations, such as perceptual latency and drift, significantly increase the risk of collision in the scenario tested, validating the proposed paradigm can stress the AV system safety. The framework offers a scalable, statistically validated, component agnostic, and fully interoperable toolset that simplifies and accelerates AV safety validations, even those with novel MP architectures and components. It can potentially reduce the time-to-market of AV and lay the foundation for future research on fault tolerance, and resilient AV design. 

**Abstract (ZH)**: 自主车辆（AV）感知故障对安全的影响研究 

---
# GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control 

**Title (ZH)**: GATO：加速并批处理轨迹优化的边端模型预测控制 

**Authors**: Alexander Du, Emre Adabag, Gabriel Bravo, Brian Plancher  

**Link**: [PDF](https://arxiv.org/pdf/2510.07625)  

**Abstract**: While Model Predictive Control (MPC) delivers strong performance across robotics applications, solving the underlying (batches of) nonlinear trajectory optimization (TO) problems online remains computationally demanding. Existing GPU-accelerated approaches typically (i) parallelize a single solve to meet real-time deadlines, (ii) scale to very large batches at slower-than-real-time rates, or (iii) achieve speed by restricting model generality (e.g., point-mass dynamics or a single linearization). This leaves a large gap in solver performance for many state-of-the-art MPC applications that require real-time batches of tens to low-hundreds of solves. As such, we present GATO, an open source, GPU-accelerated, batched TO solver co-designed across algorithm, software, and computational hardware to deliver real-time throughput for these moderate batch size regimes. Our approach leverages a combination of block-, warp-, and thread-level parallelism within and across solves for ultra-high performance. We demonstrate the effectiveness of our approach through a combination of: simulated benchmarks showing speedups of 18-21x over CPU baselines and 1.4-16x over GPU baselines as batch size increases; case studies highlighting improved disturbance rejection and convergence behavior; and finally a validation on hardware using an industrial manipulator. We open source GATO to support reproducibility and adoption. 

**Abstract (ZH)**: GATO：一种面向中等批次大小领域的并行轨迹优化求解器 

---
# Inspection Planning Primitives with Implicit Models 

**Title (ZH)**: 隐式模型下的检查规划 primitives 

**Authors**: Jingyang You, Hanna Kurniawati, Lashika Medagoda  

**Link**: [PDF](https://arxiv.org/pdf/2510.07611)  

**Abstract**: The aging and increasing complexity of infrastructures make efficient inspection planning more critical in ensuring safety. Thanks to sampling-based motion planning, many inspection planners are fast. However, they often require huge memory. This is particularly true when the structure under inspection is large and complex, consisting of many struts and pillars of various geometry and sizes. Such structures can be represented efficiently using implicit models, such as neural Signed Distance Functions (SDFs). However, most primitive computations used in sampling-based inspection planner have been designed to work efficiently with explicit environment models, which in turn requires the planner to use explicit environment models or performs frequent transformations between implicit and explicit environment models during planning. This paper proposes a set of primitive computations, called Inspection Planning Primitives with Implicit Models (IPIM), that enable sampling-based inspection planners to entirely use neural SDFs representation during planning. Evaluation on three scenarios, including inspection of a complex real-world structure with over 92M triangular mesh faces, indicates that even a rudimentary sampling-based planner with IPIM can generate inspection trajectories of similar quality to those generated by the state-of-the-art planner, while using up to 70x less memory than the state-of-the-art inspection planner. 

**Abstract (ZH)**: 基础设施的老化和复杂性增加使得高效的检查规划更加关键，以确保安全。基于采样的运动规划使得许多检查规划器快速，但它们通常需要大量的内存。尤其是当被检查的结构庞大而复杂，由各种几何形状和大小的桁架和支柱组成时，这种结构可以用隐式模型，如神经符号距离函数（SDF）进行有效表示。然而，大多数用于采样基检查规划器的基本计算是为有效处理显式环境模型设计的，这反过来要求规划器使用显式环境模型，或者在规划过程中频繁地在隐式和显式环境模型之间进行转换。本文提出了一组称为隐式模型检查规划基元（IPIM）的基本计算，使得采样基检查规划器在规划过程中可以完全使用神经SDF表示。在三个场景下的评估，包括一个复杂的真实世界结构，其三角网格面超过9200万个，表明即使是最简单的带有IPIM的采样基规划器也能生成与最先进的规划器相似质量的检查轨迹，使用的内存是最新检查规划器的1/70。 

---
# DEAS: DEtached value learning with Action Sequence for Scalable Offline RL 

**Title (ZH)**: DEAS: 分离价值学习与动作序列以实现可扩展的离线RL 

**Authors**: Changyeon Kim, Haeone Lee, Younggyo Seo, Kimin Lee, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07730)  

**Abstract**: Offline reinforcement learning (RL) presents an attractive paradigm for training intelligent agents without expensive online interactions. However, current approaches still struggle with complex, long-horizon sequential decision making. In this work, we introduce DEtached value learning with Action Sequence (DEAS), a simple yet effective offline RL framework that leverages action sequences for value learning. These temporally extended actions provide richer information than single-step actions and can be interpreted through the options framework via semi-Markov decision process Q-learning, enabling reduction of the effective planning horizon by considering longer sequences at once. However, directly adopting such sequences in actor-critic algorithms introduces excessive value overestimation, which we address through detached value learning that steers value estimates toward in-distribution actions that achieve high return in the offline dataset. We demonstrate that DEAS consistently outperforms baselines on complex, long-horizon tasks from OGBench and can be applied to enhance the performance of large-scale Vision-Language-Action models that predict action sequences, significantly boosting performance in both RoboCasa Kitchen simulation tasks and real-world manipulation tasks. 

**Abstract (ZH)**: 离线强化学习（RL）为训练智能代理提供了无需昂贵在线交互的有吸引力范式。然而，当前方法仍然难以应对复杂的、长期序列决策任务。在本文中，我们引入了基于动作序列的分离值学习（DEAS），这是一种简单而有效的离线RL框架，利用动作序列进行值学习。这些时间延长的动作提供了比单步动作更丰富的信息，并可以通过半马尔可夫决策过程Q学习通过选项框架进行解释，从而通过一次考虑更长的序列来减少有效的规划时滞。然而，直接在演员-批评家算法中采用这样的序列会导致价值过估计，我们通过分离值学习的方法将价值估计导向离线数据集中实现高回报的在分布动作来解决这一问题。我们的研究结果表明，DEAS在OGBench的复杂、长期序列任务中始终优于基线方法，并且可以应用于增强预测动作序列的大型Vision-Language-Action模型的性能，在RoboCasa Kitchen模拟任务和现实世界的操作任务中显著提升了性能。 

---
# FlowSearch: Advancing deep research with dynamic structured knowledge flow 

**Title (ZH)**: FlowSearch：动态结构化知识流推动的深度研究进展 

**Authors**: Yusong Hu, Runmin Ma, Yue Fan, Jinxin Shi, Zongsheng Cao, Yuhao Zhou, Jiakang Yuan, Xiangchao Yan, Wenlong Zhang, Lei Bai, Bo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08521)  

**Abstract**: Deep research is an inherently challenging task that demands both breadth and depth of thinking. It involves navigating diverse knowledge spaces and reasoning over complex, multi-step dependencies, which presents substantial challenges for agentic systems. To address this, we propose FlowSearch, a multi-agent framework that actively constructs and evolves a dynamic structured knowledge flow to drive subtask execution and reasoning. FlowSearch is capable of strategically planning and expanding the knowledge flow to enable parallel exploration and hierarchical task decomposition, while also adjusting the knowledge flow in real time based on feedback from intermediate reasoning outcomes and insights. FlowSearch achieves state-of-the-art performance on both general and scientific benchmarks, including GAIA, HLE, GPQA and TRQA, demonstrating its effectiveness in multi-disciplinary research scenarios and its potential to advance scientific discovery. The code is available at this https URL. 

**Abstract (ZH)**: 深度研究是一项艰巨的任务，需要广度和深度的思考。它涉及导航多元的知识空间并推理复杂的多步依赖关系，这对自主系统提出了重大挑战。为此，我们提出FlowSearch，这是一种多代理框架，能够主动构建和演化动态的知识流程结构，以驱动子任务执行和推理。FlowSearch能够战略性地规划和扩展知识流程，以实现并行探索和层次化任务分解，并根据中间推理结果和见解实时调整知识流程。FlowSearch在包括GAIA、HLE、GPQA和TRQA在内的通用和科学基准测试中实现了最先进的性能，展示了其在多学科研究场景中的有效性及其促进科学发现的潜力。代码可在以下链接获取。 

---
# Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries 

**Title (ZH)**: 超越Pass@k：推理边界广度-深度度量 

**Authors**: Marius Dragoi, Ioana Pintilie, Florin Gogianu, Florin Brad  

**Link**: [PDF](https://arxiv.org/pdf/2510.08325)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm to improve Large Language Models on reasoning tasks such as coding, math or logic. To assess the reasoning boundary (the fraction of problems a model can solve) researchers often report Pass@k at large sampling budgets. Recent results reveal a crossover phenomenon: while RLVR models outperform the base model at small k values, the base model usually outperforms them when sampling a very large number of completions. This has been interpreted as evidence that base models have a larger reasoning boundary. We argue that on tasks with discrete answer spaces, such as math with numeric outputs, Pass@k at large k reflects the increasingly higher chance of success in the limit of the number of trials rather than genuine reasoning, and can therefore be misleading. We propose Cover@tau, which measures the fraction of problems that a model can solve for which at least a tau proportion of completions are correct. Unlike Pass@k, Cover@tau captures reasoning under an explicit reliability threshold: models that rely on random guessing degrade rapidly as tau increases. We evaluate several RLVR models using Cover@tau-based metrics and illustrate how the relative rankings of popular algorithms change compared to Pass@1, offering a different perspective on reasoning boundaries. 

**Abstract (ZH)**: Verifiable奖励下的强化学习（RLVR）在提高大型语言模型在编码、数学或逻辑等推理任务上的能力方面已成为一种强大的范式。在大规模采样预算下，研究人员常报告Pass@k以评估推理边界（模型能解决的问题比例）。近期结果揭示了一个交叉现象：虽然在较小的k值下，RLVR模型优于基模型，但在大规模采样下，基模型通常表现更优。这被认为表明基模型具有更大的推理边界。我们认为，在具有离散答案空间的任务中，如带有数值输出的数学问题中，较大的k值下的Pass@k反映的是在大量试次极限下的成功概率越来越高的趋势，而不是真正的推理能力，因此可能会误导人。我们提出了Cover@tau来衡量模型可以解决的一类问题的比例，其中至少有tau比例的完成是正确的。与Pass@k不同，Cover@tau捕捉了基于显式可靠性的推理：依赖随机猜测的模型随着tau的增加会迅速退化。我们使用基于Cover@tau的指标评估了多个RLVR模型，并展示了相比于Pass@1时的相对排名的变化，提供了对推理边界的另一种视角。 

---
# Symmetry-Aware Fully-Amortized Optimization with Scale Equivariant Graph Metanetworks 

**Title (ZH)**: 对称意识全拟优化与尺度等变图元网络 

**Authors**: Bart Kuipers, Freek Byrman, Daniel Uyterlinde, Alejandro García-Castellanos  

**Link**: [PDF](https://arxiv.org/pdf/2510.08300)  

**Abstract**: Amortized optimization accelerates the solution of related optimization problems by learning mappings that exploit shared structure across problem instances. We explore the use of Scale Equivariant Graph Metanetworks (ScaleGMNs) for this purpose. By operating directly in weight space, ScaleGMNs enable single-shot fine-tuning of existing models, reducing the need for iterative optimization. We demonstrate the effectiveness of this approach empirically and provide a theoretical result: the gauge freedom induced by scaling symmetries is strictly smaller in convolutional neural networks than in multi-layer perceptrons. This insight helps explain the performance differences observed between architectures in both our work and that of Kalogeropoulos et al. (2024). Overall, our findings underscore the potential of symmetry-aware metanetworks as a powerful approach for efficient and generalizable neural network optimization. Open-source code: this https URL 

**Abstract (ZH)**: 使用权重空间中的Scale不变图元网络加速相关优化问题的求解通过学习利用问题实例间共享结构的映射。我们探索使用Scale不变图元网络（ScaleGMNs）为此目的。通过直接在权重空间中操作，ScaleGMNs允许对现有模型进行单次调整微调，减少迭代优化的需要。我们通过实证方法展示了该方法的有效性，并提供了一个理论结果：由尺度对称性引起的规范自由度在卷积神经网络中比在多层感知机中严格较小。这一洞见有助于解释我们在本文和Kalogeropoulos等人（2024）工作中观察到的架构性能差异。总体而言，我们的研究强调了具备对称意识的元网络作为高效且泛化能力强的神经网络优化方法的潜力。开源代码：详见此处。 

---
# Co-TAP: Three-Layer Agent Interaction Protocol Technical Report 

**Title (ZH)**: 共三层代理交互协议技术报告 

**Authors**: Shunyu An, Miao Wang, Yongchao Li, Dong Wan, Lina Wang, Ling Qin, Liqin Gao, Congyao Fan, Zhiyong Mao, Jiange Pu, Wenji Xia, Dong Zhao, Rui Hu, Ji Lu, Guiyue Zhou, Baoyu Tang, Yanqin Gao, Yongsheng Du, Daigang Xu, Lingjun Huang, Baoli Wang, Xiwen Zhang, Luyao Wang, Shilong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08263)  

**Abstract**: This paper proposes Co-TAP (T: Triple, A: Agent, P: Protocol), a three-layer agent interaction protocol designed to address the challenges faced by multi-agent systems across the three core dimensions of Interoperability, Interaction and Collaboration, and Knowledge Sharing. We have designed and proposed a layered solution composed of three core protocols: the Human-Agent Interaction Protocol (HAI), the Unified Agent Protocol (UAP), and the Memory-Extraction-Knowledge Protocol (MEK). HAI focuses on the interaction layer, standardizing the flow of information between users, interfaces, and agents by defining a standardized, event-driven communication paradigm. This ensures the real-time performance, reliability, and synergy of interactions. As the core of the infrastructure layer, UAP is designed to break down communication barriers among heterogeneous agents through unified service discovery and protocol conversion mechanisms, thereby enabling seamless interconnection and interoperability of the underlying network. MEK, in turn, operates at the cognitive layer. By establishing a standardized ''Memory (M) - Extraction (E) - Knowledge (K)'' cognitive chain, it empowers agents with the ability to learn from individual experiences and form shareable knowledge, thereby laying the foundation for the realization of true collective intelligence. We believe this protocol framework will provide a solid engineering foundation and theoretical guidance for building the next generation of efficient, scalable, and intelligent multi-agent applications. 

**Abstract (ZH)**: Co-TAP（交互、代理、协议三位一体）：一种针对多代理系统在互操作性、交互与合作以及知识共享三大核心维度挑战的三层代理交互协议 

---
# The Tournament Tree Method for preference elicitation in Multi-criteria decision-making 

**Title (ZH)**: Tournament Tree 方法在多准则决策中的偏好 elicitation 

**Authors**: Diego García-Zamora, Álvaro Labella, José Rui Figueira  

**Link**: [PDF](https://arxiv.org/pdf/2510.08197)  

**Abstract**: Pairwise comparison methods, such as Fuzzy Preference Relations and Saaty's Multiplicative Preference Relations, are widely used to model expert judgments in multi-criteria decision-making. However, their application is limited by the high cognitive load required to complete $m(m-1)/2$ comparisons, the risk of inconsistency, and the computational complexity of deriving consistent value scales. This paper proposes the Tournament Tree Method (TTM), a novel elicitation and evaluation framework that overcomes these limitations. The TTM requires only $m-1$ pairwise comparisons to obtain a complete, reciprocal, and consistent comparison matrix. The method consists of three phases: (i) elicitation of expert judgments using a reduced set of targeted comparisons, (ii) construction of the consistent pairwise comparison matrix, and (iii) derivation of a global value scale from the resulting matrix. The proposed approach ensures consistency by design, minimizes cognitive effort, and reduces the dimensionality of preference modeling from $m(m-1)/2$ to $m$ parameters. Furthermore, it is compatible with the classical Deck of Cards method, and thus it can handle interval and ratio scales. We have also developed a web-based tool that demonstrates its practical applicability in real decision-making scenarios. 

**Abstract (ZH)**: 基于锦标赛树的方法：一种新的专家判断 elicitation 和评价框架 

---
# Measuring What Matters: The AI Pluralism Index 

**Title (ZH)**: 衡量重要的东西：AI多元指数 

**Authors**: Rashid Mushkani  

**Link**: [PDF](https://arxiv.org/pdf/2510.08193)  

**Abstract**: Artificial intelligence systems increasingly mediate knowledge, communication, and decision making. Development and governance remain concentrated within a small set of firms and states, raising concerns that technologies may encode narrow interests and limit public agency. Capability benchmarks for language, vision, and coding are common, yet public, auditable measures of pluralistic governance are rare. We define AI pluralism as the degree to which affected stakeholders can shape objectives, data practices, safeguards, and deployment. We present the AI Pluralism Index (AIPI), a transparent, evidence-based instrument that evaluates producers and system families across four pillars: participatory governance, inclusivity and diversity, transparency, and accountability. AIPI codes verifiable practices from public artifacts and independent evaluations, explicitly handling "Unknown" evidence to report both lower-bound ("evidence") and known-only scores with coverage. We formalize the measurement model; implement a reproducible pipeline that integrates structured web and repository analysis, external assessments, and expert interviews; and assess reliability with inter-rater agreement, coverage reporting, cross-index correlations, and sensitivity analysis. The protocol, codebook, scoring scripts, and evidence graph are maintained openly with versioned releases and a public adjudication process. We report pilot provider results and situate AIPI relative to adjacent transparency, safety, and governance frameworks. The index aims to steer incentives toward pluralistic practice and to equip policymakers, procurers, and the public with comparable evidence. 

**Abstract (ZH)**: 人工智能系统 increasingly mediate知识,沟通和决策。开发和治理仍然集中在少数几家企业和国家手中，这引发了技术可能编码狭隘利益并限制公众参与的担忧。语言、视觉和编码能力基准常见，但包容性和治理的公共、可审计衡量标准罕见。我们定义人工智能多元主义为受影响的利益相关者能够塑造目标、数据实践、保护措施和部署的程度。我们提出了人工智能多元主义指数（AIPI），这是一种透明的、基于证据的工具，评估生产者和系统家族在四个支柱：参与式治理、包容性和多样性、透明度和问责制方面的表现。AIPI 通过公共材料和独立评估验证可验证的做法，并明确处理“未知”证据，报告最低界限、“证据”和仅已知分数的覆盖率。我们正式化了测量模型；实施了一个可重复的操作流水线，该流水线将结构化网页和存储库分析、外部评估和专家访谈整合在一起；并通过跨评价者一致性、覆盖率报告、跨指标相关性和敏感性分析评估其可靠性。协议、代码手册、评分脚本和证据图以受版本控制的发布形式公开维护，并具有公开裁决流程。我们报告了试点供应商的结果，并将 AIPI 相对于相邻的透明度、安全性和治理框架进行了定位。该指数旨在引导激励措施朝着多元主义实践的方向，并为政策制定者、采购者和公众提供可比的证据。 

---
# Prepared mind, fast response: A temporal decoupling framework for adaptive knowledge orchestration in open-domain dialogue 

**Title (ZH)**: 未雨绸缪，迅疾响应：一种面向开放域对话的时空解耦适应性知识 orchestration 框架 

**Authors**: Jinling Gan, Churong Liang, Runnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08175)  

**Abstract**: The latency-quality tradeoff is a fundamental constraint in open-domain dialogue AI systems, since comprehensive knowledge access necessitates prohibitive response delays. Contemporary approaches offer two inadequate solutions: lightweight instruct models achieve sub-second latency but lack reasoning depth, while tool-augmented ReAct agents enhance factuality through external knowledge at the cost of synchronous execution that blocks interaction during re- trieval processes. PMFR is thus proposed, with a tempo- ral decoupling framework that fundamentally resolves the contradiction through asynchronous knowledge orchestra- tion. PMFR employs three coordinated components: (1) a Knowledge Adequacy Evaluator for real-time sufficiency assessment, (2) a Lightweight Response Generator for imme- diate user interaction, and (3) an Asynchronous Knowledge Refinement Agent for background knowledge enhancement. This architecture maintains continuous conversational flow while progressively enriching knowledge coverage through intelligent triggering mechanisms. Evaluation results on Top- iOCQA demonstrate PMFR outperforms brute-force scaling: PMFR achieves 95.3% latency reduction (23.38s -> 1.09s) while preserving response quality comparable to heavyweight synchronous baselines (GEval-C: 0.613 vs. 0.620). 

**Abstract (ZH)**: 开放域对话AI系统中的延迟-质量权衡是基本限制，因为它要求全面的知识访问，从而导致不可接受的响应延迟。当代方法提供了两种不充分的解决方案：轻量级指令模型实现了亚秒级延迟，但缺乏推理深度；而工具增强的ReAct代理通过外部知识增强了事实性，但以同步执行为代价，在检索过程中阻塞了交互。因此提出了PMFR，这是一种通过异步知识编排根本解决矛盾的时域解耦框架。PMFR采用三个协调的组件：（1）知识充足性评估器进行实时充分性评估，（2）轻量级响应生成器进行即时用户交互，（3）背景知识增强的异步知识精炼代理。该架构保持了连续的对话流，并通过智能化触发机制逐步丰富知识覆盖面。Top-iOCQA上的评估结果显示，PMFR优于 brute-force 扩容：PMFR实现了95.3%的延迟减少（23.38s -> 1.09s），同时保持与重量级同步基线（GEval-C: 0.613 vs. 0.620）相当的响应质量。 

---
# From Ethical Declarations to Provable Independence: An Ontology-Driven Optimal-Transport Framework for Certifiably Fair AI Systems 

**Title (ZH)**: 从伦理声明到可验证独立性：一种面向本体的最优输运框架，用于认证公平的AI系统 

**Authors**: Sukriti Bhattacharya, Chitro Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08086)  

**Abstract**: This paper presents a framework for provably fair AI that overcomes the limits of current bias mitigation methods by systematically removing all sensitive information and its proxies. Using ontology engineering in OWL 2 QL, it formally defines sensitive attributes and infers their proxies through logical reasoning, constructing a sigma algebra G that captures the full structure of biased patterns. Fair representations are then obtained via Delbaen Majumdar optimal transport, which generates variables independent of G while minimizing L2 distance to preserve accuracy. This guarantees true independence rather than mere decorrelation. By modeling bias as dependence between sigma algebras, compiling ontological knowledge into measurable structures, and using optimal transport as the unique fair transformation, the approach ensures complete fairness in tasks like loan approval, where proxies such as ZIP code reveal race. The result is a certifiable and mathematically grounded method for trustworthy AI. 

**Abstract (ZH)**: 本文提出了一种基于证明公平性的AI框架，通过系统地移除所有敏感信息及其代理，克服了当前偏见缓解方法的局限。利用OWL 2 QL本体工程，形式化定义了敏感属性并通过逻辑推理推断其代理，构建了一个σ代数G，捕获了偏见模式的完整结构。随后通过Delbaen和Majumdar最优运输生成与G独立的变量，同时最小化L2距离以保留准确性。这确保了真正的独立性而非简单的降相关。通过将偏见建模为σ代数之间的依赖关系，编译本体知识为可测量结构，并使用最优运输作为唯一的公平转换，该方法确保了诸如贷款审批等任务中的完全公平性，其中代理如邮政编码可能揭示种族信息。结果，本文提供了一种可验证且具有数学基础的值得信赖AI方法。 

---
# Multi-Condition Conformal Selection 

**Title (ZH)**: 多条件一致性选择 

**Authors**: Qingyang Hao, Wenbo Liao, Bingyi Jing, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.08075)  

**Abstract**: Selecting high-quality candidates from large-scale datasets is critically important in resource-constrained applications such as drug discovery, precision medicine, and the alignment of large language models. While conformal selection methods offer a rigorous solution with False Discovery Rate (FDR) control, their applicability is confined to single-threshold scenarios (i.e., y > c) and overlooks practical needs for multi-condition selection, such as conjunctive or disjunctive conditions. In this work, we propose the Multi-Condition Conformal Selection (MCCS) algorithm, which extends conformal selection to scenarios with multiple conditions. In particular, we introduce a novel nonconformity score with regional monotonicity for conjunctive conditions and a global Benjamini-Hochberg (BH) procedure for disjunctive conditions, thereby establishing finite-sample FDR control with theoretical guarantees. The integration of these components enables the proposed method to achieve rigorous FDR-controlled selection in various multi-condition environments. Extensive experiments validate the superiority of MCCS over baselines, its generalizability across diverse condition combinations, different real-world modalities, and multi-task scalability. 

**Abstract (ZH)**: 从大规模数据集选择高质量候选人在药物发现、精准医疗和大语言模型对齐等资源受限应用中至关重要。虽然符合性选择方法通过控制虚假发现率（FDR）提供了严格的解决方案，但它们的应用局限于单一阈值场景（即，y > c），且忽视了 conjunctive 和 disjunctive 条件下的多条件选择需求。本工作提出了一种多条件符合性选择（MCCS）算法，将符合性选择扩展到多条件场景。特别地，我们引入了一种具有区域单调性的新型非一致性评分用于 conjunctive 条件，并提出了一种全局 Benjamini-Hochberg (BH) 过程用于 disjunctive 条件，从而在理论上确保了有限样本下的 FDR 控制。这些组件的整合使提出的方法能够在各种多条件环境中实现严格的 FDR 控制选择。广泛的实验证明了 MCCS 在基线方法上的优越性，以及在不同条件组合、不同的现实世界模态和多任务扩展方面的普适性。 

---
# PEAR: Phase Entropy Aware Reward for Efficient Reasoning 

**Title (ZH)**: PEAR：相位熵感知奖励促进高效推理 

**Authors**: Chen Huang, Wei Lu, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08026)  

**Abstract**: Large Reasoning Models (LRMs) have achieved impressive performance on complex reasoning tasks by generating detailed chain-of-thought (CoT) explanations. However, these responses are often excessively long, containing redundant reasoning steps that inflate inference cost and reduce usability. Controlling the length of generated reasoning without sacrificing accuracy remains an open challenge. Through a systematic empirical analysis, we reveal a consistent positive correlation between model entropy and response length at different reasoning stages across diverse LRMs: the thinking phase exhibits higher entropy, reflecting exploratory behavior of longer responses, while the final answer phase shows lower entropy, indicating a more deterministic this http URL observation suggests that entropy at different reasoning stages can serve as a control knob for balancing conciseness and performance. Based on this insight, this paper introduces Phase Entropy Aware Reward (PEAR), a reward mechanism that incorporating phase-dependent entropy into the reward design. Instead of treating all tokens uniformly, PEAR penalize excessive entropy during the thinking phase and allowing moderate exploration at the final answer phase, which encourages models to generate concise reasoning traces that retain sufficient flexibility to solve the task correctly. This enables adaptive control of response length without relying on explicit length targets or rigid truncation rules. Extensive experiments across four benchmarks demonstrate that PEAR consistently reduces response length while sustaining competitive accuracy across model scales. In addition, PEAR demonstrates strong out-of-distribution (OOD) robustness beyond the training distribution. Our code is available at: this https URL. 

**Abstract (ZH)**: 大型推理模型中的阶段熵感知奖励机制（Phase Entropy Aware Reward, PEAR）：平衡简洁性和性能的研究 

---
# ReInAgent: A Context-Aware GUI Agent Enabling Human-in-the-Loop Mobile Task Navigation 

**Title (ZH)**: ReInAgent：一种支持人类在环的移动任务导航的上下文感知GUI代理 

**Authors**: Haitao Jia, Ming He, Zimo Yin, Likang Wu, Jianping Fan, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07988)  

**Abstract**: Mobile GUI agents exhibit substantial potential to facilitate and automate the execution of user tasks on mobile phones. However, exist mobile GUI agents predominantly privilege autonomous operation and neglect the necessity of active user engagement during task execution. This omission undermines their adaptability to information dilemmas including ambiguous, dynamically evolving, and conflicting task scenarios, leading to execution outcomes that deviate from genuine user requirements and preferences. To address these shortcomings, we propose ReInAgent, a context-aware multi-agent framework that leverages dynamic information management to enable human-in-the-loop mobile task navigation. ReInAgent integrates three specialized agents around a shared memory module: an information-managing agent for slot-based information management and proactive interaction with the user, a decision-making agent for conflict-aware planning, and a reflecting agent for task reflection and information consistency validation. Through continuous contextual information analysis and sustained user-agent collaboration, ReInAgent overcomes the limitation of existing approaches that rely on clear and static task assumptions. Consequently, it enables more adaptive and reliable mobile task navigation in complex, real-world scenarios. Experimental results demonstrate that ReInAgent effectively resolves information dilemmas and produces outcomes that are more closely aligned with genuine user preferences. Notably, on complex tasks involving information dilemmas, ReInAgent achieves a 25% higher success rate than Mobile-Agent-v2. 

**Abstract (ZH)**: 基于上下文的多智能体框架ReInAgent：解决移动设备上的任务导航问题 

---
# TaoSR-SHE: Stepwise Hybrid Examination Reinforcement Learning Framework for E-commerce Search Relevance 

**Title (ZH)**: TaoSR-SHE：电子商务搜索相关性逐步混合增强学习框架 

**Authors**: Pengkun Jiao, Yiming Jin, Jianhui Yang, Chenhe Dong, Zerui Huang, Shaowei Yao, Xiaojiang Zhou, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07972)  

**Abstract**: Query-product relevance analysis is a foundational technology in e-commerce search engines and has become increasingly important in AI-driven e-commerce. The recent emergence of large language models (LLMs), particularly their chain-of-thought (CoT) reasoning capabilities, offers promising opportunities for developing relevance systems that are both more interpretable and more robust. However, existing training paradigms have notable limitations: SFT and DPO suffer from poor generalization on long-tail queries and from a lack of fine-grained, stepwise supervision to enforce rule-aligned reasoning. In contrast, reinforcement learning with verification rewards (RLVR) suffers from sparse feedback, which provides insufficient signal to correct erroneous intermediate steps, thereby undermining logical consistency and limiting performance in complex inference scenarios.
To address these challenges, we introduce the Stepwise Hybrid Examination Reinforcement Learning framework for Taobao Search Relevance (TaoSR-SHE). At its core is Stepwise Reward Policy Optimization (SRPO), a reinforcement learning algorithm that leverages step-level rewards generated by a hybrid of a high-quality generative stepwise reward model and a human-annotated offline verifier, prioritizing learning from critical correct and incorrect reasoning steps. TaoSR-SHE further incorporates two key techniques: diversified data filtering to encourage exploration across varied reasoning paths and mitigate policy entropy collapse, and multi-stage curriculum learning to foster progressive capability growth. Extensive experiments on real-world search benchmarks show that TaoSR-SHE improves both reasoning quality and relevance-prediction accuracy in large-scale e-commerce settings, outperforming SFT, DPO, GRPO, and other baselines, while also enhancing interpretability and robustness. 

**Abstract (ZH)**: 淘宝搜索相关性逐步混合检验强化学习框架（TaoSR-SHE） 

---
# Agent-Based Genetic Algorithm for Crypto Trading Strategy Optimization 

**Title (ZH)**: 基于代理的遗传算法在加密交易策略优化中的应用 

**Authors**: Qiushi Tian, Churong Liang, Kairan Hong, Runnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.07943)  

**Abstract**: Cryptocurrency markets present formidable challenges for trading strategy optimization due to extreme volatility, non-stationary dynamics, and complex microstructure patterns that render conventional parameter optimization methods fundamentally inadequate. We introduce Cypto Genetic Algorithm Agent (CGA-Agent), a pioneering hybrid framework that synergistically integrates genetic algorithms with intelligent multi-agent coordination mechanisms for adaptive trading strategy parameter optimization in dynamic financial environments. The framework uniquely incorporates real-time market microstructure intelligence and adaptive strategy performance feedback through intelligent mechanisms that dynamically guide evolutionary processes, transcending the limitations of static optimization approaches. Comprehensive empirical evaluation across three cryptocurrencies demonstrates systematic and statistically significant performance improvements on both total returns and risk-adjusted metrics. 

**Abstract (ZH)**: 加密货币市场由于极端波动性、非平稳动态以及复杂的微观结构模式，为交易策略优化带来了巨大挑战。我们提出了一种名为Crypto Genetic Algorithm Agent (CGA-Agent) 的创新性混合框架，该框架将遗传算法与智能多agent协调机制结合，以适应在动态金融市场中的自适应交易策略参数优化。该框架通过智能机制实时整合市场微观结构智能和自适应策略性能反馈，动态引导进化过程，超越了静态优化方法的局限性。全面的实证研究表明，在三种加密货币上的表现系统性且统计学上显著优于基准方法，提高了总投资回报和风险调整后的指标。 

---
# Towards Meaningful Transparency in Civic AI Systems 

**Title (ZH)**: 面向公民AI系统的有意义透明度 

**Authors**: Dave Murray-Rust, Kars Alfrink, Cristina Zaga  

**Link**: [PDF](https://arxiv.org/pdf/2510.07889)  

**Abstract**: Artificial intelligence has become a part of the provision of governmental services, from making decisions about benefits to issuing fines for parking violations. However, AI systems rarely live up to the promise of neutral optimisation, creating biased or incorrect outputs and reducing the agency of both citizens and civic workers to shape the way decisions are made. Transparency is a principle that can both help subjects understand decisions made about them and shape the processes behind those decisions. However, transparency as practiced around AI systems tends to focus on the production of technical objects that represent algorithmic aspects of decision making. These are often difficult for publics to understand, do not connect to potential for action, and do not give insight into the wider socio-material context of decision making. In this paper, we build on existing approaches that take a human-centric view on AI transparency, combined with a socio-technical systems view, to develop the concept of meaningful transparency for civic AI systems: transparencies that allow publics to engage with AI systems that affect their lives, connecting understanding with potential for action. 

**Abstract (ZH)**: 人工智能已成为政府服务的一部分，从关于福利的决策到对停车违规行为罚款的发放。然而，AI系统很少兑现中立优化的承诺，产生有偏向或错误的输出，并减少公民和公务人员塑造决策方式的自主权。透明性作为一项原则，既能帮助主体理解关于他们的决策，也能塑造这些决策背后的进程。然而，围绕AI系统实践的透明性往往侧重于生产代表决策过程中算法方面的技术对象。这些对象通常难以公众理解，不与潜在行动能力相连，也不揭示决策过程中更广泛的社会物质背景。本文在此基础上，结合以人为本的AI透明性方法和 socio-technical 系统视角，发展了公民AI系统有意义透明性的概念：允许公众参与影响他们生活的AI系统，将理解与潜在行动能力连接起来。 

---
# Understanding DeepResearch via Reports 

**Title (ZH)**: 通过报告理解DeepResearch 

**Authors**: Tianyu Fan, Xinyao Niu, Yuxiang Zheng, Fengji Zhang, Chengen Huang, Bei Chen, Junyang Lin, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07861)  

**Abstract**: DeepResearch agents represent a transformative AI paradigm, conducting expert-level research through sophisticated reasoning and multi-tool integration. However, evaluating these systems remains critically challenging due to open-ended research scenarios and existing benchmarks that focus on isolated capabilities rather than holistic performance. Unlike traditional LLM tasks, DeepResearch systems must synthesize diverse sources, generate insights, and present coherent findings, which are capabilities that resist simple verification. To address this gap, we introduce DeepResearch-ReportEval, a comprehensive framework designed to assess DeepResearch systems through their most representative outputs: research reports. Our approach systematically measures three dimensions: quality, redundancy, and factuality, using an innovative LLM-as-a-Judge methodology achieving strong expert concordance. We contribute a standardized benchmark of 100 curated queries spanning 12 real-world categories, enabling systematic capability comparison. Our evaluation of four leading commercial systems reveals distinct design philosophies and performance trade-offs, establishing foundational insights as DeepResearch evolves from information assistants toward intelligent research partners. Source code and data are available at: this https URL. 

**Abstract (ZH)**: 深度研究代理代表了一种 transformative 的人工智能范式，通过复杂的推理和多工具集成来进行专家级研究。然而，由于开放的研究场景和现有的主要关注孤立能力而非整体性能的基准，这些系统的评估仍然极具挑战性。与传统的语言模型任务不同，深度研究系统必须综合多种资源、生成见解并呈现连贯的研究成果，这些能力难以简单验证。为解决这一问题，我们引入了深度研究报告评估（DeepResearch-ReportEval），这是一种全面的框架，旨在通过研究报告这一最具代表性的输出来评估深度研究系统。我们的方法系统性地评估了三个维度：质量、冗余和事实性，采用一种创新的LLM作为法官的方法，实现了强烈的专业一致性。我们贡献了一个包含100个精心策划的查询的标准基准，覆盖12个真实世界类别，使系统的能力比较得以系统化。对于我们对四家领先商用系统的评估揭示了不同的设计哲学和性能权衡，这些洞察为深度研究从信息助手向智能研究伙伴演进奠定了基础。源代码和数据可在以下链接获取：this https URL。 

---
# FinMR: A Knowledge-Intensive Multimodal Benchmark for Advanced Financial Reasoning 

**Title (ZH)**: FinMR：一种知识密集型多模态基准用于高级金融推理 

**Authors**: Shuangyan Deng, Haizhou Peng, Jiachen Xu, Rui Mao, Ciprian Doru Giurcăneanu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07852)  

**Abstract**: Multimodal Large Language Models (MLLMs) have made substantial progress in recent years. However, their rigorous evaluation within specialized domains like finance is hindered by the absence of datasets characterized by professional-level knowledge intensity, detailed annotations, and advanced reasoning complexity. To address this critical gap, we introduce FinMR, a high-quality, knowledge-intensive multimodal dataset explicitly designed to evaluate expert-level financial reasoning capabilities at a professional analyst's standard. FinMR comprises over 3,200 meticulously curated and expertly annotated question-answer pairs across 15 diverse financial topics, ensuring broad domain diversity and integrating sophisticated mathematical reasoning, advanced financial knowledge, and nuanced visual interpretation tasks across multiple image types. Through comprehensive benchmarking with leading closed-source and open-source MLLMs, we highlight significant performance disparities between these models and professional financial analysts, uncovering key areas for model advancement, such as precise image analysis, accurate application of complex financial formulas, and deeper contextual financial understanding. By providing richly varied visual content and thorough explanatory annotations, FinMR establishes itself as an essential benchmark tool for assessing and advancing multimodal financial reasoning toward professional analyst-level competence. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在近年来取得了显著进展。然而，它们在金融等专业领域中的严格评估受到缺乏具有专业级知识强度、详细注释和高级推理复杂度的数据集的限制。为了解决这一关键差距，我们介绍了FinMR，一个高质量的专业级知识密集型多模态数据集，专门用于以专业分析师的标准评估专家级金融推理能力。FinMR 包含了超过 3,200 个精心选择和专业注释的问题-答案对，覆盖了 15 个不同的金融主题，确保了广泛的主题多样性，并结合了复杂的数学推理、高级金融知识和多类型图像的细微视觉解释任务。通过与领先的专业闭源和开源 MLLMs 的全面基准测试，我们突显了这些模型与专业金融分析师之间的显著性能差异，揭示了模型改进的关键领域，如精确的图像分析、复杂金融公式的准确应用以及更深入的金融背景理解。凭借丰富多样的视觉内容和详尽的解释性注释，FinMR 成为了评估并推动多模态金融推理达到专业分析师水平的重要基准工具。 

---
# Strategic Communication under Threat: Learning Information Trade-offs in Pursuit-Evasion Games 

**Title (ZH)**: 威胁下的战略沟通：追求逃避博弈中信息权衡的学习 

**Authors**: Valerio La Gatta, Dolev Mutzari, Sarit Kraus, VS Subrahmanian  

**Link**: [PDF](https://arxiv.org/pdf/2510.07813)  

**Abstract**: Adversarial environments require agents to navigate a key strategic trade-off: acquiring information enhances situational awareness, but may simultaneously expose them to threats. To investigate this tension, we formulate a PursuitEvasion-Exposure-Concealment Game (PEEC) in which a pursuer agent must decide when to communicate in order to obtain the evader's position. Each communication reveals the pursuer's location, increasing the risk of being targeted. Both agents learn their movement policies via reinforcement learning, while the pursuer additionally learns a communication policy that balances observability and risk. We propose SHADOW (Strategic-communication Hybrid Action Decision-making under partial Observation for Warfare), a multi-headed sequential reinforcement learning framework that integrates continuous navigation control, discrete communication actions, and opponent modeling for behavior prediction. Empirical evaluations show that SHADOW pursuers achieve higher success rates than six competitive baselines. Our ablation study confirms that temporal sequence modeling and opponent modeling are critical for effective decision-making. Finally, our sensitivity analysis reveals that the learned policies generalize well across varying communication risks and physical asymmetries between agents. 

**Abstract (ZH)**: 对抗环境要求智能体在关键的战略权衡中导航：获取信息可以增强态势感知，但同时可能会使智能体暴露于威胁之中。为了探讨这一矛盾，我们提出了一个追逐-逃逸-暴露-隐蔽游戏（PursuitEvasion-Exposure-Concealment Game, PEEC）模型，其中追逐智能体需要决定何时进行通信以获取逃逸者的方位。每次通信都会揭示追逐者的方位，增加被目标锁定的风险。两个智能体通过强化学习学习其运动策略，追逐者还通过学习一个平衡可检测性和风险的通信策略。我们提出了SHADOW（基于部分观测的战略通信混合动作决策框架用于战争），这是一种多头序列强化学习框架，结合了连续导航控制、离散通信动作以及对手建模以进行行为预测。实证评估表明，SHADOW追逐者在成功率上超过了六个竞争性基线。我们的消融研究证实了时间序列建模和对手建模对于有效决策是至关重要的。最后，我们的敏感性分析表明，学习到的策略能够很好地泛化到不同的通信风险和智能体之间的物理不对称性。 

---
# Safely Exploring Novel Actions in Recommender Systems via Deployment-Efficient Policy Learning 

**Title (ZH)**: 通过部署高效的策略学习安全探索新颖行动在推荐系统中的应用 

**Authors**: Haruka Kiyohara, Yusuke Narita, Yuta Saito, Kei Tateno, Takuma Udagawa  

**Link**: [PDF](https://arxiv.org/pdf/2510.07635)  

**Abstract**: In many real recommender systems, novel items are added frequently over time. The importance of sufficiently presenting novel actions has widely been acknowledged for improving long-term user engagement. A recent work builds on Off-Policy Learning (OPL), which trains a policy from only logged data, however, the existing methods can be unsafe in the presence of novel actions. Our goal is to develop a framework to enforce exploration of novel actions with a guarantee for safety. To this end, we first develop Safe Off-Policy Policy Gradient (Safe OPG), which is a model-free safe OPL method based on a high confidence off-policy evaluation. In our first experiment, we observe that Safe OPG almost always satisfies a safety requirement, even when existing methods violate it greatly. However, the result also reveals that Safe OPG tends to be too conservative, suggesting a difficult tradeoff between guaranteeing safety and exploring novel actions. To overcome this tradeoff, we also propose a novel framework called Deployment-Efficient Policy Learning for Safe User Exploration, which leverages safety margin and gradually relaxes safety regularization during multiple (not many) deployments. Our framework thus enables exploration of novel actions while guaranteeing safe implementation of recommender systems. 

**Abstract (ZH)**: 在推荐系统中确保新型行动探索的安全框架 

---
# A Case for Leveraging Generative AI to Expand and Enhance Training in the Provision of Mental Health Services 

**Title (ZH)**: 利用生成式AI扩展和增强精神健康服务培训的案例分析 

**Authors**: Hannah R. Lawrence, Shannon Wiltsey Stirman, Samuel Dorison, Taedong Yun, Megan Jones Bell  

**Link**: [PDF](https://arxiv.org/pdf/2510.07623)  

**Abstract**: Generative artificial intelligence (Generative AI) is transforming healthcare. With this evolution comes optimism regarding the impact it will have on mental health, as well as concern regarding the risks that come with generative AI operating in the mental health domain. Much of the investment in, and academic and public discourse about, AI-powered solutions for mental health has focused on therapist chatbots. Despite the common assumption that chatbots will be the most impactful application of GenAI to mental health, we make the case here for a lower-risk, high impact use case: leveraging generative AI to enhance and scale training in mental health service provision. We highlight key benefits of using generative AI to help train people to provide mental health services and present a real-world case study in which generative AI improved the training of veterans to support one another's mental health. With numerous potential applications of generative AI in mental health, we illustrate why we should invest in using generative AI to support training people in mental health service provision. 

**Abstract (ZH)**: 生成式人工智能（生成AI）正在变革医疗保健。随着这一变革的到来，人们对于生成AI在心理健康方面可能产生的影响既充满期待，也对其可能带来的风险表示担忧。关于利用人工智能解决方案解决心理健康问题的投资和学术及公共讨论，大多集中在心理治疗聊天机器人上。尽管普遍认为聊天机器人将是生成AI在心理健康领域最具影响力的應用，但我们在这里提出一种更低风险、更高影响力的用例：利用生成AI增强和扩展心理健康服务提供人员的培训。我们强调利用生成AI帮助培训提供心理健康服务人员的关键益处，并展示了一个实际案例，说明生成AI如何改善退伍军人支持彼此心理健康的服务培训。鉴于生成AI在心理健康领域的众多潜在应用，我们阐明了为什么应当投资利用生成AI支持培训心理健康服务提供人员的重要性。 

---
# Benchmarking is Broken - Don't Let AI be its Own Judge 

**Title (ZH)**: 基准测试出了问题 - 别让AI自我评审 

**Authors**: Zerui Cheng, Stella Wohnig, Ruchika Gupta, Samiul Alam, Tassallah Abdullahi, João Alves Ribeiro, Christian Nielsen-Garcia, Saif Mir, Siran Li, Jason Orender, Seyed Ali Bahrainian, Daniel Kirste, Aaron Gokaslan, Mikołaj Glinka, Carsten Eickhoff, Ruben Wolff  

**Link**: [PDF](https://arxiv.org/pdf/2510.07575)  

**Abstract**: The meteoric rise of Artificial Intelligence (AI), with its rapidly expanding market capitalization, presents both transformative opportunities and critical challenges. Chief among these is the urgent need for a new, unified paradigm for trustworthy evaluation, as current benchmarks increasingly reveal critical vulnerabilities. Issues like data contamination and selective reporting by model developers fuel hype, while inadequate data quality control can lead to biased evaluations that, even if unintentionally, may favor specific approaches. As a flood of participants enters the AI space, this "Wild West" of assessment makes distinguishing genuine progress from exaggerated claims exceptionally difficult. Such ambiguity blurs scientific signals and erodes public confidence, much as unchecked claims would destabilize financial markets reliant on credible oversight from agencies like Moody's.
In high-stakes human examinations (e.g., SAT, GRE), substantial effort is devoted to ensuring fairness and credibility; why settle for less in evaluating AI, especially given its profound societal impact? This position paper argues that the current laissez-faire approach is unsustainable. We contend that true, sustainable AI advancement demands a paradigm shift: a unified, live, and quality-controlled benchmarking framework robust by construction, not by mere courtesy and goodwill. To this end, we dissect the systemic flaws undermining today's AI evaluation, distill the essential requirements for a new generation of assessments, and introduce PeerBench, a community-governed, proctored evaluation blueprint that embodies this paradigm through sealed execution, item banking with rolling renewal, and delayed transparency. Our goal is to pave the way for evaluations that can restore integrity and deliver genuinely trustworthy measures of AI progress. 

**Abstract (ZH)**: 人工智能的迅猛崛起及其市场资本的迅速扩张，带来了变革性的机遇和关键性的挑战。其中最为紧迫的问题是需要构建一个新的统一可信赖评估范式，因为当前的基准测试越来越多地揭示出关键的漏洞。数据污染和模型开发者的选择性报告加剧了这种炒作现象，而不尽如人意的数据质量控制可能导致带有偏见的评估，即使这一偏见是无意的，也可能偏好特定的方法。随着大量参与者涌入AI领域，“野蛮生长”的评估体系使得辨别真实进展与夸大宣传变得异常困难。这种模糊性模糊了科学信号，侵蚀了公众信心，就如同未经管控的索赔会动摇依赖于可靠监管机构（例如穆迪）的金融市场一样。

在高风险的人类考试（如SAT、GRE）中，投入了大量精力确保公平性和可信度；为什么在评估AI时变得不够呢，特别是考虑到AI对社会的巨大影响？本文认为，目前的放任自流的做法是不可持续的。我们认为，真正的可持续AI进步需要范式转变，即构建一个统一、实时、质量可控的基准测试框架，这一框架应是通过系统设计而非仅凭好意和善意来实现的坚固框架。为此，我们剖析了当前AI评估体系的系统性缺陷，提炼出新一代评估体系的必要要求，并介绍了PeerBench，这是一种由社区管理、监考的评估蓝图，通过封闭执行、题库滚动更新和延迟透明化来体现这一范式。我们的目标是为评估铺平道路，从而恢复诚信并提供真正可信的AI进步衡量标准。 

---
# An Evaluation Study of Hybrid Methods for Multilingual PII Detection 

**Title (ZH)**: 混合方法多语言PII检测评价研究 

**Authors**: Harshit Rajgarhia, Suryam Gupta, Asif Shaik, Gulipalli Praveen Kumar, Y Santhoshraj, Sanka Nithya Tanvy Nishitha, Abhishek Mukherji  

**Link**: [PDF](https://arxiv.org/pdf/2510.07551)  

**Abstract**: The detection of Personally Identifiable Information (PII) is critical for privacy compliance but remains challenging in low-resource languages due to linguistic diversity and limited annotated data. We present RECAP, a hybrid framework that combines deterministic regular expressions with context-aware large language models (LLMs) for scalable PII detection across 13 low-resource locales. RECAP's modular design supports over 300 entity types without retraining, using a three-phase refinement pipeline for disambiguation and filtering. Benchmarked with nervaluate, our system outperforms fine-tuned NER models by 82% and zero-shot LLMs by 17% in weighted F1-score. This work offers a scalable and adaptable solution for efficient PII detection in compliance-focused applications. 

**Abstract (ZH)**: 低资源语言中个人可识别信息的检测对于隐私合规至关重要但依然具有挑战性，因语言多样性及标注数据有限。我们提出RECAP，一种结合确定性正则表达式和上下文感知的大语言模型（LLM）的混合框架，用于在13种低资源语言环境中可扩展地检测个人可识别信息。RECAP的模块化设计无需重新训练即可支持超过300种实体类型，并通过三阶段细化流水线进行消歧和过滤。经nervaluate评测，我们的系统在加权F1分数上比微调的NER模型高出82%，比零样本的LLM高出17%。本研究提供了针对合规应用高效个人可识别信息检测的可扩展和适应性解决方案。 

---
# Optimizing Ethical Risk Reduction for Medical Intelligent Systems with Constraint Programming 

**Title (ZH)**: 基于约束编程优化医疗智能系统伦理风险减少 

**Authors**: Clotilde Brayé, Aurélien Bricout, Arnaud Gotlieb, Nadjib Lazaar, Quentin Vallet  

**Link**: [PDF](https://arxiv.org/pdf/2510.07491)  

**Abstract**: Medical Intelligent Systems (MIS) are increasingly integrated into healthcare workflows, offering significant benefits but also raising critical safety and ethical concerns. According to the European Union AI Act, most MIS will be classified as high-risk systems, requiring a formal risk management process to ensure compliance with the ethical requirements of trust- worthy AI. In this context, we focus on risk reduction optimization problems, which aim to reduce risks with ethical considerations by finding the best balanced assignment of risk assessment values according to their coverage of trustworthy AI ethical requirements. We formalize this problem as a constrained optimization task and investigate three resolution paradigms: Mixed Integer Programming (MIP), Satisfiability (SAT), and Constraint Pro- gramming(CP).Our contributions include the mathematical formulation of this optimization problem, its modeling with the Minizinc constraint modeling language, and a comparative experimental study that analyzes the performance, expressiveness, and scalability of each ap- proach to solving. From the identified limits of the methodology, we draw some perspectives of this work regarding the integration of the Minizinc model into a complete trustworthy AI ethical risk management process for MIS. 

**Abstract (ZH)**: 医疗智能系统（MIS）在医疗保健工作流程中的应用日益增多，虽然带来了显著的好处，但也引发了重要的安全和伦理问题。根据欧盟人工智能法案，大多数MIS将被归类为高风险系统，需要通过正式的风险管理过程来确保符合可信人工智能的伦理要求。在这种背景下，我们关注的是伦理考量下的风险减少优化问题，即通过找到最佳平衡的风险评估值分配方案，以覆盖可信人工智能的伦理要求来降低风险。我们将这个问题形式化为一个约束优化任务，并研究了三种解决范式：混合整数规划（MIP）、满足性（SAT）和约束编程（CP）。我们的贡献包括对该优化问题的数学表述、使用Minizinc约束建模语言进行建模以及对每种方法求解性能、表示能力和可扩展性的比较实验研究。通过对方法论的识别限制，我们对该工作在MIS中集成Minizinc模型以实现完整的可信人工智能伦理风险管理过程提出了几点展望。 

---
# ExpertAgent: Enhancing Personalized Education through Dynamic Planning and Retrieval-Augmented Long-Chain Reasoning 

**Title (ZH)**: ExpertAgent: 通过动态规划和检索增强长链推理提升个性化教育 

**Authors**: Binrong Zhu, Guiran Liu, Nina Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07456)  

**Abstract**: The application of advanced generative artificial intelligence in education is often constrained by the lack of real-time adaptability, personalization, and reliability of the content. To address these challenges, we propose ExpertAgent - an intelligent agent framework designed for personalized education that provides reliable knowledge and enables highly adaptive learning experiences. Therefore, we developed ExpertAgent, an innovative learning agent that provides users with a proactive and personalized learning experience. ExpertAgent dynamic planning of the learning content and strategy based on a continuously updated student model. Therefore, overcoming the limitations of traditional static learning content to provide optimized teaching strategies and learning experience in real time. All instructional content is grounded in a validated curriculum repository, effectively reducing hallucination risks in large language models and improving reliability and trustworthiness. 

**Abstract (ZH)**: 高级生成人工智能在教育中的应用往往受限于内容的实时适应性、个性化和可靠性不足。为应对这些挑战，我们提出了一种名为ExpertAgent的智能代理框架，该框架旨在提供个性化的教育服务，并提供可靠的知识，从而实现高度适应性的学习体验。因此，我们开发了ExpertAgent这一创新的学习代理，为用户提供主动且个性化的学习体验。ExpertAgent根据不断更新的学生模型动态规划学习内容和策略，克服了传统静态学习内容的局限性，提供实时优化的教学策略和学习体验。所有教学内容均基于经过验证的课程资源库，有效降低了大型语言模型的幻觉风险，提高了可靠性和可信度。 

---
# Less is More: Strategic Expert Selection Outperforms Ensemble Complexity in Traffic Forecasting 

**Title (ZH)**: fewer is More: 策略性专家选择在交通预测中优于ensemble复杂性 

**Authors**: Walid Guettala, Yufan Zhao, László Gulyás  

**Link**: [PDF](https://arxiv.org/pdf/2510.07426)  

**Abstract**: Traffic forecasting is fundamental to intelligent transportation systems, enabling congestion mitigation and emission reduction in increasingly complex urban environments. While recent graph neural network approaches have advanced spatial temporal modeling, existing mixture of experts frameworks like Time Enhanced Spatio Temporal Attention Model (TESTAM) lack explicit incorporation of physical road network topology, limiting their spatial capabilities. We present TESTAM+, an enhanced spatio temporal forecasting framework that introduces a novel SpatioSemantic Expert integrating physical road topology with data driven feature similarity through hybrid graph construction. TESTAM+ achieves significant improvements over TESTAM: 1.3% MAE reduction on METR LA (3.10 vs. 3.14) and 4.1% improvement on PEMS BAY (1.65 vs. 1.72). Through comprehensive ablation studies, we discover that strategic expert selection fundamentally outperforms naive ensemble aggregation. Individual experts demonstrate remarkable effectiveness: the Adaptive Expert achieves 1.63 MAE on PEMS BAY, outperforming the original three expert TESTAM (1.72 MAE), while the SpatioSemantic Expert matches this performance with identical 1.63 MAE. The optimal Identity + Adaptive configuration achieves an 11.5% MAE reduction compared to state of the art MegaCRN on METR LA (2.99 vs. 3.38), while reducing inference latency by 53.1% compared to the full four expert TESTAM+. Our findings reveal that fewer, strategically designed experts outperform complex multi expert ensembles, establishing new state of the art performance with superior computational efficiency for real time deployment. 

**Abstract (ZH)**: 基于时空语义专家的城市交通流预测增强框架：Physical Road Topology Integration for Improved Spatial Capabilities 

---
# Position: AI Will Transform Neuropsychology Through Mental Health Digital Twins for Dynamic Mental Health Care, Especially for ADHD 

**Title (ZH)**: 位置：AI将通过心理健康数字孪生实现动态心理健康护理，尤其适用于注意力缺陷多动障碍（ADHD） 

**Authors**: Neil Natarajan, Sruthi Viswanathan, Xavier Roberts-Gaal, Michelle Marie Martel  

**Link**: [PDF](https://arxiv.org/pdf/2510.07409)  

**Abstract**: Static solutions don't serve a dynamic mind. Thus, we advocate a shift from static mental health diagnostic assessments to continuous, artificial intelligence (AI)-driven assessment. Focusing on Attention-Deficit/Hyperactivity Disorder (ADHD) as a case study, we explore how generative AI has the potential to address current capacity constraints in neuropsychology, potentially enabling more personalized and longitudinal care pathways. In particular, AI can efficiently conduct frequent, low-level experience sampling from patients and facilitate diagnostic reconciliation across care pathways. We envision a future where mental health care benefits from continuous, rich, and patient-centered data sampling to dynamically adapt to individual patient needs and evolving conditions, thereby improving both accessibility and efficacy of treatment. We further propose the use of mental health digital twins (MHDTs) - continuously updated computational models that capture individual symptom dynamics and trajectories - as a transformative framework for personalized mental health care. We ground this framework in empirical evidence and map out the research agenda required to refine and operationalize it. 

**Abstract (ZH)**: 静态解决方案不适合动态思维。因此，我们建议从静态心理健康诊断评估转向连续的、基于人工智能（AI）的评估。以注意缺陷多动障碍（ADHD）为例，我们探讨了生成式AI如何有可能解决神经心理学领域的当前容量限制， potentially 推动更为个性化和纵向的护理路径。特别是，AI可以高效地从患者那里进行频繁的、低层次的经验抽样，并促进不同护理路径中的诊断一致。我们设想一个未来，在这个未来中，心理健康护理可以从连续的、丰富的和以患者为中心的数据抽样中受益，以动态适应个别患者的需求及其不断变化的状况，从而提高治疗的可及性和有效性。我们进一步提出使用心理健康数字双胞胎（MHDTs）——不断更新的计算模型，捕捉个体症状动态和轨迹——作为实现个性化心理健康护理的变革性框架。我们基于实证证据来构建这一框架，并指出为了细化和实施这一框架所需的研究议程。 

---
# Truth-Aware Decoding: A Program-Logic Approach to Factual Language Generation 

**Title (ZH)**: 真相意识解码：基于程序逻辑的事实语言生成方法 

**Authors**: Faruk Alpay, Hamdi Alakkad  

**Link**: [PDF](https://arxiv.org/pdf/2510.07331)  

**Abstract**: This paper introduces Truth-Aware Decoding (TAD), a verification-oriented decoding scheme that aligns neural language generation with knowledge bases. Situated in the tradition of probabilistic program semantics for sequence models, TAD augments modern instruction-tuned systems with a lattice of semantic guards that operate at decode time. Our contributions are fourfold: (i) a constraint-based semantics that renders oracle filtering as a program-logic judgment, (ii) a proof that greedy selection enjoys local likelihood dominance under sound and complete guards (Theorem 2.7), (iii) an entropy-style invariant that quantifies factual risk via knowledge-aware safe mass, and (iv) a multi-agent operational calculus with verified Lean artefacts to certify implementation behaviour. Numerical and algorithmic case studies confirm that the resulting guardrails reduce hallucinations without sacrificing throughput, yielding a pragmatic bridge between large-scale empirical models and formal verification. 

**Abstract (ZH)**: 这篇论文介绍了基于真实性的解码（TAD），这是一种验证导向的解码方案，将神经语言生成与知识库对齐。TAD 位于概率程序语义对序列模型的传统之上，在现代指令调优系统中加入了在解码时操作的语义门控层次结构。我们的贡献包括：(i) 基于约束的语义学，将或然过滤视为程序逻辑判断，(ii) 证明在正确和完备的门控条件下，贪婪选择具有局部似然性主导权（定理2.7），(iii) 一种熵风格的不变量，通过知识感知的安全质量量化事实风险，以及(iv) 验证过的Lean artefacts多代理操作微积分以认证实现行为。数值和算法案例研究证实，由此产生的门控结构减少了幻觉现象，同时不牺牲吞吐量，从而为大规模经验模型与形式验证之间架起了一座务实的桥梁。 

---
# Platform-Agnostic Modular Architecture for Quantum Benchmarking 

**Title (ZH)**: 基于平台的量子基准测试模块化架构 

**Authors**: Neer Patel, Anish Giri, Hrushikesh Pramod Patil, Noah Siekierski, Avimita Chatterjee, Sonika Johri, Timothy Proctor, Thomas Lubinski, Siyuan Niu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08469)  

**Abstract**: We present a platform-agnostic modular architecture that addresses the increasingly fragmented landscape of quantum computing benchmarking by decoupling problem generation, circuit execution, and results analysis into independent, interoperable components. Supporting over 20 benchmark variants ranging from simple algorithmic tests like Bernstein-Vazirani to complex Hamiltonian simulation with observable calculations, the system integrates with multiple circuit generation APIs (Qiskit, CUDA-Q, Cirq) and enables diverse workflows. We validate the architecture through successful integration with Sandia's $\textit{pyGSTi}$ for advanced circuit analysis and CUDA-Q for multi-GPU HPC simulations. Extensibility of the system is demonstrated by implementing dynamic circuit variants of existing benchmarks and a new quantum reinforcement learning benchmark, which become readily available across multiple execution and analysis modes. Our primary contribution is identifying and formalizing modular interfaces that enable interoperability between incompatible benchmarking frameworks, demonstrating that standardized interfaces reduce ecosystem fragmentation while preserving optimization flexibility. This architecture has been developed as a key enhancement to the continually evolving QED-C Application-Oriented Performance Benchmarks for Quantum Computing suite. 

**Abstract (ZH)**: 我们提出了一种平台无关的模块化架构，通过将问题生成、电路执行和结果分析解耦为独立且可互操作的组件，应对量子计算基准测试日益碎片化的景观。该系统支持超过20种基准测试变体，从简单的伯恩斯坦-瓦zigani算法测试到复杂的哈密尔顿量模拟及可观测量计算。该系统与多个电路生成API（Qiskit、CUDA-Q、Cirq）兼容，并支持多种工作流。通过成功将该架构与桑迪亚国家实验室的pyGSTi集成进行高级电路分析，以及与CUDA-Q集成进行多GPU高性能计算模拟，我们验证了该架构的有效性。系统的可扩展性通过实现现有基准的动态电路变体和一个新的量子强化学习基准来展示，这些基准可以在多种执行和分析模式下立即使用。我们的主要贡献在于识别并形式化了模块化接口，这些接口能够使不兼容的基准测试框架相互兼容，证明了标准化接口可以减少生态系统碎片化，同时保持优化的灵活性。该架构是不断演化的量子计算套件QED-C应用程序导向性能基准的关键增强之一。 

---
# Integral Signatures of Activation Functions: A 9-Dimensional Taxonomy and Stability Theory for Deep Learning 

**Title (ZH)**: 激活函数的积分签名：深度学习中9维分类学与稳定性理论 

**Authors**: Ankur Mali, Lawrence Hall, Jake Williams, Gordon Richards  

**Link**: [PDF](https://arxiv.org/pdf/2510.08456)  

**Abstract**: Activation functions govern the expressivity and stability of neural networks, yet existing comparisons remain largely heuristic. We propose a rigorous framework for their classification via a nine-dimensional integral signature S_sigma(phi), combining Gaussian propagation statistics (m1, g1, g2, m2, eta), asymptotic slopes (alpha_plus, alpha_minus), and regularity measures (TV(phi'), C(phi)). This taxonomy establishes well-posedness, affine reparameterization laws with bias, and closure under bounded slope variation. Dynamical analysis yields Lyapunov theorems with explicit descent constants and identifies variance stability regions through (m2', g2). From a kernel perspective, we derive dimension-free Hessian bounds and connect smoothness to bounded variation of phi'. Applying the framework, we classify eight standard activations (ReLU, leaky-ReLU, tanh, sigmoid, Swish, GELU, Mish, TeLU), proving sharp distinctions between saturating, linear-growth, and smooth families. Numerical Gauss-Hermite and Monte Carlo validation confirms theoretical predictions. Our framework provides principled design guidance, moving activation choice from trial-and-error to provable stability and kernel conditioning. 

**Abstract (ZH)**: 激活函数 Governs 神经网络的表达能力和稳定性，现有比较大多still保留直觉性。我们提出了一种通过九维积分签名S_sigma(φ)的严谨分类框架，结合高斯传播统计量（m1, g1, g2, m2, η），渐近斜率（α_plus, α_minus）和正则性度量（TV(φ'), C(φ)）。这种分类体系确立了良定义性、带有偏置的仿射重参数化法则以及在有界斜率变化下的封闭性。动力学分析得到了显式下降常数的Lyapunov定理，并通过（m2', g2)确定了方差稳定性区域。从核的角度来看，我们推导出了维数无关的Hessian边界，并将平滑度与φ'的有界变分联系起来。应用此框架，我们分类了八种标准激活函数（ReLU, 泄漏ReLU, 双曲正切, Sigmoid, Swish, GELU, Mish, TeLU），证明了饱和、线性增长和平滑家族之间的严格区别。数值的Gauss-Hermite和蒙特卡洛验证确认了理论预测。此框架提供了原则性的设计指导，使激活函数的选择从试错转变为可证明的稳定性和核条件。 

---
# gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity 

**Title (ZH)**: gLSTM：通过增加存储容量减轻过度压缩 

**Authors**: Hugh Blayney, Álvaro Arroyo, Xiaowen Dong, Michael M. Bronstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.08450)  

**Abstract**: Graph Neural Networks (GNNs) leverage the graph structure to transmit information between nodes, typically through the message-passing mechanism. While these models have found a wide variety of applications, they are known to suffer from over-squashing, where information from a large receptive field of node representations is collapsed into a single fixed sized vector, resulting in an information bottleneck. In this paper, we re-examine the over-squashing phenomenon through the lens of model storage and retrieval capacity, which we define as the amount of information that can be stored in a node's representation for later use. We study some of the limitations of existing tasks used to measure over-squashing and introduce a new synthetic task to demonstrate that an information bottleneck can saturate this capacity. Furthermore, we adapt ideas from the sequence modeling literature on associative memories, fast weight programmers, and the xLSTM model to develop a novel GNN architecture with improved capacity. We demonstrate strong performance of this architecture both on our capacity synthetic task, as well as a range of real-world graph benchmarks. 

**Abstract (ZH)**: 图神经网络（GNNs）通过图结构在节点之间传递信息，通常通过消息传递机制实现。尽管这些模型在广泛的应用中找到了用途，但它们会遭受过压缩的现象，即节点表示的大 receptive field 中的信息被压缩到一个固定大小的向量中，形成信息瓶颈。在本文中，我们从模型存储和检索能力的角度重新审视过压缩现象，我们将存储和检索能力定义为可以在节点表示中储存并稍后使用的信息量。我们研究了现有用于衡量过压缩的一些限制，并引入了一个新的合成任务来证明信息瓶颈可以饱和这种能力。此外，我们借鉴序列建模领域中关联记忆、快速权重编程和xLSTM模型的想法，开发了一种新的GNN架构，具有改进的能力。我们在这项能力合成任务以及一系列实际图基准上都展示了该架构的出色性能。 

---
# Synthetic Series-Symbol Data Generation for Time Series Foundation Models 

**Title (ZH)**: 合成序列符号数据生成用于时间序列基础模型 

**Authors**: Wenxuan Wang, Kai Wu, Yujian Betterest Li, Dan Wang, Xiaoyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08445)  

**Abstract**: Foundation models for time series analysis (TSA) have attracted significant attention. However, challenges such as training data scarcity and imbalance continue to hinder their development. Inspired by complex dynamic system theories, we design a series-symbol data generation mechanism, enabling the unrestricted creation of high-quality time series data paired with corresponding symbolic expressions. To leverage series-symbol data pairs with strong correlations, we develop \texttt{SymTime}, a pre-trained foundation model for enhancing time series representation using symbolic information. \texttt{SymTime} demonstrates competitive performance across five major TSA tasks when fine-tunes with downstream tasks, rivaling foundation models pre-trained on real-world datasets. This approach underscores the potential of series-symbol data generation and pretraining mechanisms in overcoming data scarcity and enhancing task performance. The code is available at this https URL. 

**Abstract (ZH)**: 基于时间序列分析的础模型（TSA）吸引了显著的关注。然而，训练数据稀缺性和不平衡性等挑战仍阻碍其发展。受复杂动力系统理论启发，我们设计了一种序列-符号数据生成机制，能够不受限制地生成高质量的时间序列数据及其对应的符号表达。为了利用强相关性的时间序列-符号数据配对，我们开发了\texttt{SymTime}，这是一种预训练基础模型，用于通过符号信息增强时间序列表示。在下游任务微调后，\texttt{SymTime}在五个主要的TSA任务中展示了竞争力，与基于真实数据集预训练的基础模型不相上下。该方法强调了序列-符号数据生成和预训练机制在克服数据稀缺性和提升任务性能方面的潜力。代码见此链接：this https URL。 

---
# ClauseLens: Clause-Grounded, CVaR-Constrained Reinforcement Learning for Trustworthy Reinsurance Pricing 

**Title (ZH)**: ClauseLens: 基于条款、CVaR约束的可信再保险定价 reinforcement learning方法 

**Authors**: Stella C. Dong, James R. Finlay  

**Link**: [PDF](https://arxiv.org/pdf/2510.08429)  

**Abstract**: Reinsurance treaty pricing must satisfy stringent regulatory standards, yet current quoting practices remain opaque and difficult to audit. We introduce ClauseLens, a clause-grounded reinforcement learning framework that produces transparent, regulation-compliant, and risk-aware treaty quotes.
ClauseLens models the quoting task as a Risk-Aware Constrained Markov Decision Process (RA-CMDP). Statutory and policy clauses are retrieved from legal and underwriting corpora, embedded into the agent's observations, and used both to constrain feasible actions and to generate clause-grounded natural language justifications.
Evaluated in a multi-agent treaty simulator calibrated to industry data, ClauseLens reduces solvency violations by 51%, improves tail-risk performance by 27.9% (CVaR_0.10), and achieves 88.2% accuracy in clause-grounded explanations with retrieval precision of 87.4% and recall of 91.1%.
These findings demonstrate that embedding legal context into both decision and explanation pathways yields interpretable, auditable, and regulation-aligned quoting behavior consistent with Solvency II, NAIC RBC, and the EU AI Act. 

**Abstract (ZH)**: 再保险条约定价必须满足严格的监管标准，但当前的报价实践依然不透明且难以审计。我们介绍了ClauseLens，这是一种基于条款的强化学习框架，生成透明、合规且风控的条约报价。 

---
# Prompts Generalize with Low Data: Non-vacuous Generalization Bounds for Optimizing Prompts with More Informative Priors 

**Title (ZH)**: 低数据量下提示通用化：具有更多信息先验的优化提示的非空泛泛化界 

**Authors**: David Madras, Joshua Safyan, Qiuyi, Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08413)  

**Abstract**: Many prompt engineering techniques have been successful in practice, even when optimizing over a large prompt space with with a small amount of task-specific data. Recent work has partially explained this success by showing generalization bounds which apply PAC-Bayes theory to the discrete prompt space, but they are non-vacuous only in data-rich scenarios. We argue that such widespread success can be more fully explained through more carefully considering data- or distribution-dependent perplexity, which acts as an effective prior and steers the optimization towards prompts that are more ``natural'' for the task at hand. We derive novel generalization bounds that are non-vacuous for data-scarce prompt optimization via more useful priors, formally analyzing how perplexity regularization tightens these bounds by limiting exploration. Empirically, we explore both the bounds' effectiveness and the practical benefits of perplexity regularization in improving prompt generalization. 

**Abstract (ZH)**: 许多提示工程技巧在实践中的表现都很成功，即使在仅使用少量任务特定数据的情况下优化一个大的提示空间也是如此。近期的研究部分地通过将PAC-Bayes理论应用于离散提示空间来解释这种成功，并且只有在数据丰富的情况下这些解释才非空泛。我们认为，这种广泛的成功可以通过更仔细地考虑数据或分布相关的困惑度来更全面地解释，困惑度作为有效的先验能够引导优化向着更适合任务的“自然”提示。我们推导出新的泛化界，通过更有用的先验在数据稀缺的提示优化中保持非空泛，并形式化分析困惑度正则化如何通过限制探索来收紧这些界限。从实验上，我们研究了这些界的有效性以及困惑度正则化在提高提示泛化方面的实际益处。 

---
# Detecting Legend Items on Historical Maps Using GPT-4o with In-Context Learning 

**Title (ZH)**: 使用GPT-4o和上下文学习检测历史地图中的图例项 

**Authors**: Sofia Kirsanova, Yao-Yi Chiang, Weiwei Duan  

**Link**: [PDF](https://arxiv.org/pdf/2510.08385)  

**Abstract**: Historical map legends are critical for interpreting cartographic symbols. However, their inconsistent layouts and unstructured formats make automatic extraction challenging. Prior work focuses primarily on segmentation or general optical character recognition (OCR), with few methods effectively matching legend symbols to their corresponding descriptions in a structured manner. We present a method that combines LayoutLMv3 for layout detection with GPT-4o using in-context learning to detect and link legend items and their descriptions via bounding box predictions. Our experiments show that GPT-4 with structured JSON prompts outperforms the baseline, achieving 88% F-1 and 85% IoU, and reveal how prompt design, example counts, and layout alignment affect performance. This approach supports scalable, layout-aware legend parsing and improves the indexing and searchability of historical maps across various visual styles. 

**Abstract (ZH)**: 历史地图图例对于解析制图符号至关重要。然而，它们不一致的布局和无结构的格式使得自动提取变得具有挑战性。先前的工作主要集中在分段或一般的光学字符识别（OCR）上，很少有方法能够有效地将图例符号与其对应的描述以结构化的方式匹配。我们提出了一种方法，该方法结合使用LayoutLMv3进行布局检测与GPT-4o进行上下文学习，通过边界框预测检测并链接图例项及其描述。我们的实验表明，GPT-4在结构化JSON提示下的表现优于基线，达到88%的F-1和85%的IoU，并揭示了提示设计、示例数量和布局对齐如何影响性能。该方法支持可扩展、布局感知的图例解析，并提高了各种视觉风格的历史地图的索引和可搜索性。 

---
# DeepEN: Personalized Enteral Nutrition for Critically Ill Patients using Deep Reinforcement Learning 

**Title (ZH)**: DeepEN: 采用深度强化学习的重症患者个性化肠内营养供给方法 

**Authors**: Daniel Jason Tan, Jiayang Chen, Dilruk Perera, Kay Choong See, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.08350)  

**Abstract**: We introduce DeepEN, a deep reinforcement learning (RL) framework for personalized enteral nutrition (EN) in critically ill patients. Trained offline on over 11,000 ICU patients from the MIMIC-IV database, DeepEN generates 4-hourly recommendations for caloric, protein, and fluid intake tailored to each patient's evolving physiology. The model integrates a curated, clinically informed state space with a custom reward function that balances short-term physiological and nutrition-related goals with long-term survival outcomes. Using a dueling double deep Q-network with conservative Q-learning regularization, DeepEN learns clinically realistic policies that align with high-value clinician actions while discouraging unsafe deviations. Across various qualitative and quantitative metrics, DeepEN outperforms clinician-derived and guideline-based policies, achieving a 3.7 $\pm$ 0.17 percentage-point reduction in estimated mortality (18.8% vs 22.5%) and improvements in key nutritional biomarkers. These findings highlight the potential of safe, data-driven personalization of EN therapy to improve outcomes beyond traditional guideline- or heuristic-based approaches. 

**Abstract (ZH)**: DeepEN：一种用于重症患者个性化肠内营养的深度强化学习框架 

---
# Learning What's Missing: Attention Dispersion and EMA Stabilization in Length Generalization 

**Title (ZH)**: 学习所缺失的：注意力分散与长度泛化中的EMA稳定化 

**Authors**: Pál Zsámboki, Benjamin Levi, David Ansel Josef Smith, Mitansh Kagalwala, Arlington Kell, Samuel Liechty, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08341)  

**Abstract**: We study length generalization in transformers through the set complement task, where a model must predict a uniform distribution over tokens absent from an input sequence -- an ability central to board-game style reasoning. Our main theoretical result establishes two statements. First, we prove tight bounds on embedding and value dimensions for single-layer attention-only transformers. Second, we show that if such a model achieves balanced logit displacement at lengths 1 and 2, then it must generalize to longer sequences, though with reduced precision. A mechanistic reading of the proof explains this limitation: as more tokens are attended to, softmax compresses logit displacements, eroding separation between valid and invalid outputs. Training dynamics also suggest a second obstacle: when many next tokens are possible, updates become noisy. We hypothesize that dropout can counteract the first effect and Exponential Moving Average (EMA) the second. We validate these hypotheses through random hyperparameter search on the set complement task, which confirms both mechanisms. We then test OthelloGPT, a GPT-1 style model trained on random Othello moves, and find that EMA again improves length generalization in this more complex setting. 

**Abstract (ZH)**: 我们通过集合补任务研究transformers的长度泛化能力，其中模型必须预测输入序列中缺失的标记的均匀分布——这是类似棋盘游戏推理的核心能力。我们的主要理论结果建立了两个陈述。首先，我们证明了单层自注意变换器的嵌入和价值维度的紧密边界。其次，我们证明如果该模型在长度1和2上实现了平衡的logit位移，那么它必须泛化到更长的序列，尽管精度有所降低。对证明的机械解读解释了这一局限：随着更多标记被关注，softmax压缩logit位移，削弱了有效和无效输出之间的区分。训练动态也表明存在第二个障碍：当有许多可能的下一个标记时，更新变得嘈杂。我们假设dropout可以抵消第一个效应，而Exponential Moving Average (EMA)可以抵消第二个效应。我们通过集合补任务中的随机超参数搜索验证了这些假设，这证实了这两种机制。然后，我们测试了一个基于随机井字棋移动训练的OthelloGPT模型，并发现在这个更复杂的环境中，EMA再次改善了长度泛化。 

---
# Counterfactual Identifiability via Dynamic Optimal Transport 

**Title (ZH)**: 动态最优传输下的反事实可识别性 

**Authors**: Fabio De Sousa Ribeiro, Ainkaran Santhirasekaram, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2510.08294)  

**Abstract**: We address the open question of counterfactual identification for high-dimensional multivariate outcomes from observational data. Pearl (2000) argues that counterfactuals must be identifiable (i.e., recoverable from the observed data distribution) to justify causal claims. A recent line of work on counterfactual inference shows promising results but lacks identification, undermining the causal validity of its estimates. To address this, we establish a foundation for multivariate counterfactual identification using continuous-time flows, including non-Markovian settings under standard criteria. We characterise the conditions under which flow matching yields a unique, monotone and rank-preserving counterfactual transport map with tools from dynamic optimal transport, ensuring consistent inference. Building on this, we validate the theory in controlled scenarios with counterfactual ground-truth and demonstrate improvements in axiomatic counterfactual soundness on real images. 

**Abstract (ZH)**: 我们针对可观测数据中的高维多变量结果的反事实识别这一开放问题提出了解决方案。佩尔（2000）认为，为了证实因果关系，反事实必须是可以从观测数据分布中恢复的。近期关于反事实推断的研究显示出了有希望的结果，但缺乏识别性，这削弱了其因果有效性。为了解决这一问题，我们利用连续时间流建立了多变量反事实识别的基础，包括在标准条件下非马氏过程下的识别性。我们通过动力最优传输的工具刻画了流匹配在何种条件下可以生成唯一的、单调的且保持秩的反事实传输映射，并确保推断的一致性。在此基础上，我们在有反事实真相控制的场景中验证了该理论，并在实际图像上展示了在公理反事实准确性方面的改进。 

---
# FuelCast: Benchmarking Tabular and Temporal Models for Ship Fuel Consumption 

**Title (ZH)**: FuelCast: 评估表结构和时间序列模型在船舶燃油消耗预测中的性能 

**Authors**: Justus Viga, Penelope Mueck, Alexander Löser, Torben Weis  

**Link**: [PDF](https://arxiv.org/pdf/2510.08217)  

**Abstract**: In the shipping industry, fuel consumption and emissions are critical factors due to their significant impact on economic efficiency and environmental sustainability. Accurate prediction of ship fuel consumption is essential for further optimization of maritime operations. However, heterogeneous methodologies and limited high-quality datasets hinder direct comparison of modeling approaches. This paper makes three key contributions: (1) we introduce and release a new dataset (this https URL) comprising operational and environmental data from three ships; (2) we define a standardized benchmark covering tabular regression and time-series regression (3) we investigate the application of in-context learning for ship consumption modeling using the TabPFN foundation model - a first in this domain to our knowledge. Our results demonstrate strong performance across all evaluated models, supporting the feasibility of onboard, data-driven fuel prediction. Models incorporating environmental conditions consistently outperform simple polynomial baselines relying solely on vessel speed. TabPFN slightly outperforms other techniques, highlighting the potential of foundation models with in-context learning capabilities for tabular prediction. Furthermore, including temporal context improves accuracy. 

**Abstract (ZH)**: 在航运业中，燃料消耗和排放是关键因素，因为它们对经济效率和环境可持续性有重大影响。准确预测船舶燃料消耗对于进一步优化海上运营至关重要。然而，方法异质性和高质量数据集有限阻碍了建模方法的直接比较。本文做出了三项关键贡献：（1）我们引入并发布了新的数据集（https://...），包含三艘船舶的运营和环境数据；（2）我们定义了一个标准化基准，涵盖表格回归和时间序列回归；（3）我们研究了使用TabPFN基础模型进行船舶消耗建模的上下文学习应用——据我们所知，这是该领域的首创。我们的结果显示，所有评估模型均表现出色，支持了船上基于数据的燃料预测的可行性。包含环境条件的模型一贯优于仅依赖船舶速度的简单多项式基线。TabPFN 略微优于其他技术，突显了具有上下文学习能力的基础模型在表格预测中的潜力。此外，包含时间上下文可以提高准确性。 

---
# Robust Canonicalization through Bootstrapped Data Re-Alignment 

**Title (ZH)**: 通过-bootstraped数据重对齐的稳健 canonical化 

**Authors**: Johann Schmidt, Sebastian Stober  

**Link**: [PDF](https://arxiv.org/pdf/2510.08178)  

**Abstract**: Fine-grained visual classification (FGVC) tasks, such as insect and bird identification, demand sensitivity to subtle visual cues while remaining robust to spatial transformations. A key challenge is handling geometric biases and noise, such as different orientations and scales of objects. Existing remedies rely on heavy data augmentation, which demands powerful models, or on equivariant architectures, which constrain expressivity and add cost. Canonicalization offers an alternative by shielding such biases from the downstream model. In practice, such functions are often obtained using canonicalization priors, which assume aligned training data. Unfortunately, real-world datasets never fulfill this assumption, causing the obtained canonicalizer to be brittle. We propose a bootstrapping algorithm that iteratively re-aligns training samples by progressively reducing variance and recovering the alignment assumption. We establish convergence guarantees under mild conditions for arbitrary compact groups, and show on four FGVC benchmarks that our method consistently outperforms equivariant, and canonicalization baselines while performing on par with augmentation. 

**Abstract (ZH)**: 细粒度视觉分类任务中的范化对齐算法：处理几何偏置和噪声以提高模型鲁棒性 

---
# Leveraging Whisper Embeddings for Audio-based Lyrics Matching 

**Title (ZH)**: 基于 whisper 向量的音频歌词匹配 

**Authors**: Eleonora Mancini, Joan Serrà, Paolo Torroni, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2510.08176)  

**Abstract**: Audio-based lyrics matching can be an appealing alternative to other content-based retrieval approaches, but existing methods often suffer from limited reproducibility and inconsistent baselines. In this work, we introduce WEALY, a fully reproducible pipeline that leverages Whisper decoder embeddings for lyrics matching tasks. WEALY establishes robust and transparent baselines, while also exploring multimodal extensions that integrate textual and acoustic features. Through extensive experiments on standard datasets, we demonstrate that WEALY achieves a performance comparable to state-of-the-art methods that lack reproducibility. In addition, we provide ablation studies and analyses on language robustness, loss functions, and embedding strategies. This work contributes a reliable benchmark for future research, and underscores the potential of speech technologies for music information retrieval tasks. 

**Abstract (ZH)**: 基于音频的歌词匹配可以成为内容基于的检索方法的有吸引力的替代方案，但现有方法往往存在重现性有限和基准不一致的问题。在本文中，我们介绍了WEALY，一个完全可重现的管道，利用Whisper解码器嵌入进行歌词匹配任务。WEALY建立了稳健且透明的基准，并探索了结合文本和声学特征的多模态扩展。通过在标准数据集上的广泛实验，我们展示了WEALY在重现性较差的领先方法中的性能相当。此外，我们提供了消融研究和语言鲁棒性、损失函数和嵌入策略的分析。本工作为未来研究提供了一个可靠的基准，并强调了语音技术在音乐信息检索任务中的潜在应用。 

---
# Quantum Agents for Algorithmic Discovery 

**Title (ZH)**: 量子代理进行算法发现 

**Authors**: Iordanis Kerenidis, El-Amine Cherrat  

**Link**: [PDF](https://arxiv.org/pdf/2510.08159)  

**Abstract**: We introduce quantum agents trained by episodic, reward-based reinforcement learning to autonomously rediscover several seminal quantum algorithms and protocols. In particular, our agents learn: efficient logarithmic-depth quantum circuits for the Quantum Fourier Transform; Grover's search algorithm; optimal cheating strategies for strong coin flipping; and optimal winning strategies for the CHSH and other nonlocal games. The agents achieve these results directly through interaction, without prior access to known optimal solutions. This demonstrates the potential of quantum intelligence as a tool for algorithmic discovery, opening the way for the automated design of novel quantum algorithms and protocols. 

**Abstract (ZH)**: 通过基于奖励的经验学习训练的量子代理自主重新发现多个经典量子算法和协议 

---
# Approximate Domain Unlearning for Vision-Language Models 

**Title (ZH)**: 视觉-语言模型的近似域忘怀 

**Authors**: Kodai Kawamura, Yuta Goto, Rintaro Yanagi, Hirokatsu Kataoka, Go Irie  

**Link**: [PDF](https://arxiv.org/pdf/2510.08132)  

**Abstract**: Pre-trained Vision-Language Models (VLMs) exhibit strong generalization capabilities, enabling them to recognize a wide range of objects across diverse domains without additional training. However, they often retain irrelevant information beyond the requirements of specific downstream tasks, raising concerns about computational efficiency and potential information leakage. This has motivated growing interest in approximate unlearning, which aims to selectively remove unnecessary knowledge while preserving overall model performance. Existing approaches to approximate unlearning have primarily focused on class unlearning, where a VLM is retrained to fail to recognize specified object classes while maintaining accuracy for others. However, merely forgetting object classes is often insufficient in practical applications. For instance, an autonomous driving system should accurately recognize real cars while avoiding misrecognition of illustrated cars depicted in roadside advertisements as real cars, which could be hazardous. In this paper, we introduce Approximate Domain Unlearning (ADU), a novel problem setting that requires reducing recognition accuracy for images from specified domains (e.g., illustration) while preserving accuracy for other domains (e.g., real). ADU presents new technical challenges: due to the strong domain generalization capability of pre-trained VLMs, domain distributions are highly entangled in the feature space, making naive approaches based on penalizing target domains ineffective. To tackle this limitation, we propose a novel approach that explicitly disentangles domain distributions and adaptively captures instance-specific domain information. Extensive experiments show that our approach outperforms baselines built upon VLM tuning techniques, paving the way for practical and fine-grained unlearning in VLMs. Code: this https URL. 

**Abstract (ZH)**: 预训练多模态模型的近似领域卸载（Approximate Domain Unlearning for Pre-trained Vision-Language Models） 

---
# Bayesian Decision Making around Experts 

**Title (ZH)**: 专家周围的贝叶斯决策制定 

**Authors**: Daniel Jarne Ornia, Joel Dyer, Nicholas Bishop, Anisoara Calinescu, Michael Wooldridge  

**Link**: [PDF](https://arxiv.org/pdf/2510.08113)  

**Abstract**: Complex learning agents are increasingly deployed alongside existing experts, such as human operators or previously trained agents. However, it remains unclear how should learners optimally incorporate certain forms of expert data, which may differ in structure from the learner's own action-outcome experiences. We study this problem in the context of Bayesian multi-armed bandits, considering: (i) offline settings, where the learner receives a dataset of outcomes from the expert's optimal policy before interaction, and (ii) simultaneous settings, where the learner must choose at each step whether to update its beliefs based on its own experience, or based on the outcome simultaneously achieved by an expert. We formalize how expert data influences the learner's posterior, and prove that pretraining on expert outcomes tightens information-theoretic regret bounds by the mutual information between the expert data and the optimal action. For the simultaneous setting, we propose an information-directed rule where the learner processes the data source that maximizes their one-step information gain about the optimal action. Finally, we propose strategies for how the learner can infer when to trust the expert and when not to, safeguarding the learner for the cases where the expert is ineffective or compromised. By quantifying the value of expert data, our framework provides practical, information-theoretic algorithms for agents to intelligently decide when to learn from others. 

**Abstract (ZH)**: 复杂学习代理与现有专家（如人类操作员或先前训练的代理）共同部署的情况下，如何高效地整合专家数据仍不明确，尤其是当专家数据的结构与学习代理自身的行动-结果经验不同时。我们通过贝叶斯多臂 bandit 框架研究了这个问题，考虑了两种情况：(i) 在线设置，学习者在交互之前接收专家最优策略的结果数据集；(ii) 同步设置，学习者在每一步需要决定是基于自身的经验还是同时实现的专家结果更新其信念。我们形式化了专家数据如何影响学习者的后验分布，并证明了基于专家数据的预训练通过专家数据与最优行动之间的互信息，收紧了信息论意义下的后悔上界。对于同步设置，我们提出了一种基于信息指导的原则，该原则使学习者能够最大化其对最优行动的一步信息增益来处理数据源。最后，我们提出策略以帮助学习者判断何时信任专家以及何时不应，从而保障在专家无效或被篡改的情况下保护学习者。通过量化专家数据的价值，我们的框架为代理提供了实用的信息论算法，以智能地决定何时向他人学习。 

---
# VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving Documents 

**Title (ZH)**: 版本意识增强型检索增益生成：面向演进文档的版本感知生成方法 

**Authors**: Daniel Huwiler, Kurt Stockinger, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2510.08109)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems fail when documents evolve through versioning-a ubiquitous characteristic of technical documentation. Existing approaches achieve only 58-64% accuracy on version-sensitive questions, retrieving semantically similar content without temporal validity checks. We present VersionRAG, a version-aware RAG framework that explicitly models document evolution through a hierarchical graph structure capturing version sequences, content boundaries, and changes between document states. During retrieval, VersionRAG routes queries through specialized paths based on intent classification, enabling precise version-aware filtering and change tracking. On our VersionQA benchmark-100 manually curated questions across 34 versioned technical documents-VersionRAG achieves 90% accuracy, outperforming naive RAG (58%) and GraphRAG (64%). VersionRAG reaches 60% accuracy on implicit change detection where baselines fail (0-10%), demonstrating its ability to track undocumented modifications. Additionally, VersionRAG requires 97% fewer tokens during indexing than GraphRAG, making it practical for large-scale deployment. Our work establishes versioned document QA as a distinct task and provides both a solution and benchmark for future research. 

**Abstract (ZH)**: 版本感知生成（VersionRAG）系统：面向技术文档版本演化的检索增强生成框架 

---
# Development of Mental Models in Human-AI Collaboration: A Conceptual Framework 

**Title (ZH)**: 人类与人工智能协作中心理模型的发展：一个概念框架 

**Authors**: Joshua Holstein, Gerhard Satzger  

**Link**: [PDF](https://arxiv.org/pdf/2510.08104)  

**Abstract**: Artificial intelligence has become integral to organizational decision-making and while research has explored many facets of this human-AI collaboration, the focus has mainly been on designing the AI agent(s) and the way the collaboration is set up - generally assuming a human decision-maker to be "fixed". However, it has largely been neglected that decision-makers' mental models evolve through their continuous interaction with AI systems. This paper addresses this gap by conceptualizing how the design of human-AI collaboration influences the development of three complementary and interdependent mental models necessary for this collaboration. We develop an integrated socio-technical framework that identifies the mechanisms driving the mental model evolution: data contextualization, reasoning transparency, and performance feedback. Our work advances human-AI collaboration literature through three key contributions: introducing three distinct mental models (domain, information processing, complementarity-awareness); recognizing the dynamic nature of mental models; and establishing mechanisms that guide the purposeful design of effective human-AI collaboration. 

**Abstract (ZH)**: 人工智能已成为组织决策不可或缺的一部分，尽管已有研究探讨了许多人机协作方面的内容，但主要集中在设计AI代理及其合作方式上，通常假定人类决策者是固定的。然而，决策者的心智模型通过与AI系统持续互动而演变这一点却很少被关注。本文通过阐述人机协作设计如何影响这一合作所需发展的三种互补且相互依赖的心智模型的发展，弥补了这一缺口。我们构建了一个整合的社会技术框架，识别出驱动心智模型演变的机制：数据语境化、推理透明度和绩效反馈。我们的研究通过三个关键贡献推进了人机协作文献：引入了三种独特的心智模型（领域心智模型、信息处理心智模型、互补性意识心智模型）；承认心智模型的动态特性；并建立了指导目的性设计有效人机协作的机制。 

---
# A Novel Ensemble Learning Approach for Enhanced IoT Attack Detection: Redefining Security Paradigms in Connected Systems 

**Title (ZH)**: 一种增强物联网攻击检测的新型集成学习方法：重新定义连接系统中的安全范式 

**Authors**: Hikmat A. M. Abdeljaber, Md. Alamgir Hossain, Sultan Ahmad, Ahmed Alsanad, Md Alimul Haque, Sudan Jha, Jabeen Nazeer  

**Link**: [PDF](https://arxiv.org/pdf/2510.08084)  

**Abstract**: The rapid expansion of Internet of Things (IoT) devices has transformed industries and daily life by enabling widespread connectivity and data exchange. However, this increased interconnection has introduced serious security vulnerabilities, making IoT systems more exposed to sophisticated cyber attacks. This study presents a novel ensemble learning architecture designed to improve IoT attack detection. The proposed approach applies advanced machine learning techniques, specifically the Extra Trees Classifier, along with thorough preprocessing and hyperparameter optimization. It is evaluated on several benchmark datasets including CICIoT2023, IoTID20, BotNeTIoT L01, ToN IoT, N BaIoT, and BoT IoT. The results show excellent performance, achieving high recall, accuracy, and precision with very low error rates. These outcomes demonstrate the model efficiency and superiority compared to existing approaches, providing an effective and scalable method for securing IoT environments. This research establishes a solid foundation for future progress in protecting connected devices from evolving cyber threats. 

**Abstract (ZH)**: 物联网设备的迅速扩张通过实现广泛的连接和数据交换改变了各行各业和日常生活。然而，这种增加的互联性引入了严重的安全漏洞，使物联网系统更容易受到复杂的网络攻击。本研究提出了一种新颖的集成学习架构，旨在提高物联网攻击检测效果。提出的方案应用了先进的机器学习技术，特别是Extra Trees Classifier，并结合了彻底的预处理和超参数优化。该方法在包括CICIoT2023、IoTID20、BotNeTIoT L01、ToN IoT、N BaIoT和BoT IoT等多个基准数据集上进行了评估。结果表明，该模型具有出色的性能，实现了高召回率、准确率和精确率，并且错误率极低。这些结果展示了该模型相较于现有方法的高效性和优越性，为保护物联网环境提供了一种有效且可扩展的方法。本研究为未来防范不断演变的网络威胁保护连接设备奠定了坚实的基础。 

---
# Attribution-by-design: Ensuring Inference-Time Provenance in Generative Music Systems 

**Title (ZH)**: 设计归因：确保生成音乐系统推理时间可追溯性 

**Authors**: Fabio Morreale, Wiebke Hutiri, Joan Serrà, Alice Xiang, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2510.08062)  

**Abstract**: The rise of AI-generated music is diluting royalty pools and revealing structural flaws in existing remuneration frameworks, challenging the well-established artist compensation systems in the music industry. Existing compensation solutions, such as piecemeal licensing agreements, lack scalability and technical rigour, while current data attribution mechanisms provide only uncertain estimates and are rarely implemented in practice. This paper introduces a framework for a generative music infrastructure centred on direct attribution, transparent royalty distribution, and granular control for artists and rights' holders. We distinguish ontologically between the training set and the inference set, which allows us to propose two complementary forms of attribution: training-time attribution and inference-time attribution. We here favour inference-time attribution, as it enables direct, verifiable compensation whenever an artist's catalogue is used to condition a generated output. Besides, users benefit from the ability to condition generations on specific songs and receive transparent information about attribution and permitted usage. Our approach offers an ethical and practical solution to the pressing need for robust compensation mechanisms in the era of AI-generated music, ensuring that provenance and fairness are embedded at the core of generative systems. 

**Abstract (ZH)**: AI生成音乐的兴起正在稀释版权池并揭示现有回报框架的结构性缺陷，挑战音乐行业已建立的艺术家补偿体系。现有的补偿解决方案，如零散的许可协议，缺乏扩展性和技术严谨性，而当前的数据归属机制只能提供不确定的估计且很少实际实施。本文提出了一种以直接归属、透明版权分配和艺术家及权利持有者的细粒度控制为中心的生成音乐基础设施框架。我们从本体论上区分训练集和推理集，这使我们能够提出两种互补形式的归属：训练时归属和推理时归属。我们更倾向于推理时归属，因为它可以在使用艺术家目录来条件生成输出时提供直接且可验证的补偿。此外，用户可以从有条件生成特定歌曲的能力中受益，并获得有关归属和许可使用情况的透明信息。我们的方法为AI生成音乐时代迫切需要的稳健补偿机制提供了一种伦理和实用的解决方案，确保来源和公平性在生成系统的核心得到体现。 

---
# FedDTRE: Federated Dialogue Generation Models Powered by Trustworthiness Evaluation 

**Title (ZH)**: FedDTRE：基于可靠性评估的联邦对话生成模型 

**Authors**: Shule Lu, Lingxiang Wang, Sijia Wen, Ziwei Wang, Hainan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08058)  

**Abstract**: With the rapid development of artificial intelligence, dialogue systems have become a prominent form of human-computer interaction. However, traditional centralized or fully local training approaches face challenges in balancing privacy preservation and personalization due to data privacy concerns and heterogeneous device capabilities. Federated learning, as a representative distributed paradigm, offers a promising solution. However, existing methods often suffer from overfitting under limited client data and tend to forget global information after multiple training rounds, leading to poor generalization. To address these issues, we propose FedDTRE, a Federated adaptive aggregation strategy for Dialogue generation based on Trustworthiness Evaluation. Instead of directly replacing local models with the global model, FedDTRE leverages trustworthiness scores of both global and local models on a fairness-oriented evaluation dataset to dynamically regulate the global model's contribution during local updates. Experimental results demonstrate that FedDTRE can improve dialogue model performance and enhance the quality of dialogue generation. 

**Abstract (ZH)**: 基于可信度评估的联邦自适应聚合策略FedDTRE：对话生成中的联邦学习 

---
# Verifying Graph Neural Networks with Readout is Intractable 

**Title (ZH)**: 验证带有读出的图神经网络是不可行的 

**Authors**: Artem Chernobrovkin, Marco Sälzer, François Schwarzentruber, Nicolas Troquard  

**Link**: [PDF](https://arxiv.org/pdf/2510.08045)  

**Abstract**: We introduce a logical language for reasoning about quantized aggregate-combine graph neural networks with global readout (ACR-GNNs). We provide a logical characterization and use it to prove that verification tasks for quantized GNNs with readout are (co)NEXPTIME-complete. This result implies that the verification of quantized GNNs is computationally intractable, prompting substantial research efforts toward ensuring the safety of GNN-based systems. We also experimentally demonstrate that quantized ACR-GNN models are lightweight while maintaining good accuracy and generalization capabilities with respect to non-quantized models. 

**Abstract (ZH)**: 我们介绍了一种逻辑语言用于推理量化聚合结合图神经网络（ACR-GNN）及其全局读出操作。我们提供了逻辑特征化，并利用其证明了具有读出操作的量化图神经网络的验证任务是(co)NEXPTIME完全的。这一结果表明，验证量化图神经网络在计算上是不可行的，从而促使了对基于图神经网络系统的安全性保证的大量研究工作。我们还实验证明了量化ACR-GNN模型具有轻量级结构，同时在非量化模型方面保持了良好的准确性和泛化能力。 

---
# MRI-derived quantification of hepatic vessel-to-volume ratios in chronic liver disease using a deep learning approach 

**Title (ZH)**: 使用深度学习方法的慢性肝病中肝血管与体积比的MRI定量分析 

**Authors**: Alexander Herold, Daniel Sobotka, Lucian Beer, Nina Bastati, Sarah Poetter-Lang, Michael Weber, Thomas Reiberger, Mattias Mandorfer, Georg Semmler, Benedikt Simbrunner, Barbara D. Wichtmann, Sami A. Ba-Ssalamah, Michael Trauner, Ahmed Ba-Ssalamah, Georg Langs  

**Link**: [PDF](https://arxiv.org/pdf/2510.08039)  

**Abstract**: Background: We aimed to quantify hepatic vessel volumes across chronic liver disease stages and healthy controls using deep learning-based magnetic resonance imaging (MRI) analysis, and assess correlations with biomarkers for liver (dys)function and fibrosis/portal hypertension.
Methods: We assessed retrospectively healthy controls, non-advanced and advanced chronic liver disease (ACLD) patients using a 3D U-Net model for hepatic vessel segmentation on portal venous phase gadoxetic acid-enhanced 3-T MRI. Total (TVVR), hepatic (HVVR), and intrahepatic portal vein-to-volume ratios (PVVR) were compared between groups and correlated with: albumin-bilirubin (ALBI) and model for end-stage liver disease-sodium (MELD-Na) score, and fibrosis/portal hypertension (Fibrosis-4 [FIB-4] score, liver stiffness measurement [LSM], hepatic venous pressure gradient [HVPG], platelet count [PLT], and spleen volume).
Results: We included 197 subjects, aged 54.9 $\pm$ 13.8 years (mean $\pm$ standard deviation), 111 males (56.3\%): 35 healthy controls, 44 non-ACLD, and 118 ACLD patients. TVVR and HVVR were highest in controls (3.9; 2.1), intermediate in non-ACLD (2.8; 1.7), and lowest in ACLD patients (2.3; 1.0) ($p \leq 0.001$). PVVR was reduced in both non-ACLD and ACLD patients (both 1.2) compared to controls (1.7) ($p \leq 0.001$), but showed no difference between CLD groups ($p = 0.999$). HVVR significantly correlated indirectly with FIB-4, ALBI, MELD-Na, LSM, and spleen volume ($\rho$ ranging from -0.27 to -0.40), and directly with PLT ($\rho = 0.36$). TVVR and PVVR showed similar but weaker correlations.
Conclusions: Deep learning-based hepatic vessel volumetry demonstrated differences between healthy liver and chronic liver disease stages and shows correlations with established markers of disease severity. 

**Abstract (ZH)**: 背景:我们旨在使用基于深度学习的磁共振成像(MRI)分析定量慢性肝病各阶段和健康对照的肝血管体积，并评估其与肝功能(失)常和纤维化/门脉高压生物标志物的相关性。方法:我们使用3D U-Net模型对门静脉期gadoxetic酸增强的3-T MRI进行肝血管分割，回顾性评估健康对照、非晚期和晚期慢性肝病(ACLD)患者。比较各组的总体积比(TVVR)、肝体积比(HVVR)和肝内门静脉体积比(PVVR)，并与白蛋白-胆红素(ALBI)评分和终末期肝病-钠(MELD-Na)评分，以及纤维化/门脉高压(Fibrosis-4 [FIB-4]评分、肝硬度测量[LSM]、肝静脉压力梯度[HVPG]、血小板计数[PLT]和脾脏体积)进行相关性分析。结果:共纳入197名受试者，年龄54.9岁 $\pm$ 13.8岁，男性111名(56.3%)：35名健康对照、44名非ACLD和118名ACLD患者。在健康对照中TVVR和HVVR最高(3.9；2.1)，非ACLD中居中(2.8；1.7)，ACLD患者中最低(2.3；1.0)($p \leq 0.001$)。PVVR在非ACLD和ACLD患者中均低于健康对照(1.7)($p \leq 0.001$)，但在不同CLD组之间无显著差异($p = 0.999$)。HVVR与FIB-4、ALBI、MELD-Na、LSM和脾脏体积呈间接相关($\rho$从-0.27到-0.40)，与PLT呈直接相关($\rho = 0.36$)。TVVR和PVVR的相关性相似但较弱。结论:基于深度学习的肝脏血管容积测量显示健康肝脏与慢性肝病各阶段之间的差异，并与疾病严重性的已建立标志物显示出相关性。 

---
# Backdoor Vectors: a Task Arithmetic View on Backdoor Attacks and Defenses 

**Title (ZH)**: 后门向量：后门攻击与防御的任务算术视角 

**Authors**: Stanisław Pawlak, Jan Dubiński, Daniel Marczak, Bartłomiej Twardowski  

**Link**: [PDF](https://arxiv.org/pdf/2510.08016)  

**Abstract**: Model merging (MM) recently emerged as an effective method for combining large deep learning models. However, it poses significant security risks. Recent research shows that it is highly susceptible to backdoor attacks, which introduce a hidden trigger into a single fine-tuned model instance that allows the adversary to control the output of the final merged model at inference time. In this work, we propose a simple framework for understanding backdoor attacks by treating the attack itself as a task vector. $Backdoor\ Vector\ (BV)$ is calculated as the difference between the weights of a fine-tuned backdoored model and fine-tuned clean model. BVs reveal new insights into attacks understanding and a more effective framework to measure their similarity and transferability. Furthermore, we propose a novel method that enhances backdoor resilience through merging dubbed $Sparse\ Backdoor\ Vector\ (SBV)$ that combines multiple attacks into a single one. We identify the core vulnerability behind backdoor threats in MM: $inherent\ triggers$ that exploit adversarial weaknesses in the base model. To counter this, we propose $Injection\ BV\ Subtraction\ (IBVS)$ - an assumption-free defense against backdoors in MM. Our results show that SBVs surpass prior attacks and is the first method to leverage merging to improve backdoor effectiveness. At the same time, IBVS provides a lightweight, general defense that remains effective even when the backdoor threat is entirely unknown. 

**Abstract (ZH)**: Model Merging中的后门攻击理解与防御：Sparse Backdoor Vector与Injection BV Subtraction 

---
# Leveraging Author-Specific Context for Scientific Figure Caption Generation: 3rd SciCap Challenge 

**Title (ZH)**: 利用作者特定上下文进行科学图表标题生成：第3届SciCap挑战赛 

**Authors**: Watcharapong Timklaypachara, Monrada Chiewhawan, Nopporn Lekuthai, Titipat Achakulvisut  

**Link**: [PDF](https://arxiv.org/pdf/2510.07993)  

**Abstract**: Scientific figure captions require both accuracy and stylistic consistency to convey visual information. Here, we present a domain-specific caption generation system for the 3rd SciCap Challenge that integrates figure-related textual context with author-specific writing styles using the LaMP-Cap dataset. Our approach uses a two-stage pipeline: Stage 1 combines context filtering, category-specific prompt optimization via DSPy's MIPROv2 and SIMBA, and caption candidate selection; Stage 2 applies few-shot prompting with profile figures for stylistic refinement. Our experiments demonstrate that category-specific prompts outperform both zero-shot and general optimized approaches, improving ROUGE-1 recall by +8.3\% while limiting precision loss to -2.8\% and BLEU-4 reduction to -10.9\%. Profile-informed stylistic refinement yields 40--48\% gains in BLEU scores and 25--27\% in ROUGE. Overall, our system demonstrates that combining contextual understanding with author-specific stylistic adaptation can generate captions that are both scientifically accurate and stylistically faithful to the source paper. 

**Abstract (ZH)**: 科学图表说明文字需要兼具准确性和风格一致性以传达视觉信息。本文介绍了针对第3届SciCap挑战赛的领域特定说明文字生成系统，该系统结合了图表相关文本上下文和作者特定的写作风格，使用了LaMP-Cap数据集。我们的方法采用两阶段管道：第一阶段结合上下文过滤、通过DSPy的MIPROv2和SIMBA进行类别特定提示优化，并选择说明文字候选；第二阶段使用实例提示进行风格 refinement。实验结果表明，类别特定提示在ROUGE-1召回率上提高了8.3%，同时将精确率损失控制在2.8%，BLEU-4减少控制在10.9%。基于个人风格的风格 refinement 将BLEU分数提高了40-48%，ROUGE分数提高了25-27%。总体而言，我们的系统证明了结合上下文理解与作者特定的风格适应可以生成既科学准确又忠实于原始论文风格的说明文字。 

---
# Is Architectural Complexity Always the Answer? A Case Study on SwinIR vs. an Efficient CNN 

**Title (ZH)**: 建筑复杂性always是解决方案吗？SwinIR与高效CNN的案例研究 

**Authors**: Chandresh Sutariya, Nitin Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.07984)  

**Abstract**: The simultaneous restoration of high-frequency details and suppression of severe noise in low-light imagery presents a significant and persistent challenge in computer vision. While large-scale Transformer models like SwinIR have set the state of the art in performance, their high computational cost can be a barrier for practical applications. This paper investigates the critical trade-off between performance and efficiency by comparing the state-of-the-art SwinIR model against a standard, lightweight Convolutional Neural Network (CNN) on this challenging task. Our experimental results reveal a nuanced but important finding. While the Transformer-based SwinIR model achieves a higher peak performance, with a Peak Signal-to-Noise Ratio (PSNR) of 39.03 dB, the lightweight CNN delivers a surprisingly competitive PSNR of 37.4 dB. Crucially, the CNN reached this performance after converging in only 10 epochs of training, whereas the more complex SwinIR model required 132 epochs. This efficiency is further underscored by the model's size; the CNN is over 55 times smaller than SwinIR. This work demonstrates that a standard CNN can provide a near state-of-the-art result with significantly lower computational overhead, presenting a compelling case for its use in real-world scenarios where resource constraints are a primary concern. 

**Abstract (ZH)**: 低光照图像中高频细节的恢复和严重噪声抑制的同时实现是计算机视觉中一个持续存在的重大挑战。尽管基于Transformer的大规模模型如SwinIR在性能上达到最新水平，但其高昂的计算成本可能成为实际应用的障碍。本文通过将最新的SwinIR模型与标准的轻量级卷积神经网络（CNN）在这一挑战性任务上进行比较，探讨了性能与效率之间的关键权衡。实验结果显示，虽然基于Transformer的SwinIR模型在峰值信噪比（PSNR）为39.03 dB的情况下实现了较高的峰值性能，但轻量级CNN的PSNR同样达到了令人惊讶的37.4 dB。更重要的是，CNN仅在10个训练周期后就达到了这一性能，而更为复杂的SwinIR模型则需要132个训练周期。此外，模型的大小差异进一步凸显了CNN的优势；CNN比SwinIR小约55倍。这项工作展示了标准CNN可以在显著降低计算开销的情况下提供接近最新水平的结果，为资源限制为主要关注点的真实世界应用场景提供了有力的使用案例。 

---
# ZeroCard: Cardinality Estimation with Zero Dependence on Target Databases -- No Data, No Query, No Retraining 

**Title (ZH)**: ZeroCard: 不依赖目标数据库基数估计——无数据，无查询，无需重新训练 

**Authors**: Xianghong Xu, Rong Kang, Xiao He, Lei Zhang, Jianjun Chen, Tieying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07983)  

**Abstract**: Cardinality estimation is a fundamental task in database systems and plays a critical role in query optimization. Despite significant advances in learning-based cardinality estimation methods, most existing approaches remain difficult to generalize to new datasets due to their strong dependence on raw data or queries, thus limiting their practicality in real scenarios. To overcome these challenges, we argue that semantics in the schema may benefit cardinality estimation, and leveraging such semantics may alleviate these dependencies. To this end, we introduce ZeroCard, the first semantics-driven cardinality estimation method that can be applied without any dependence on raw data access, query logs, or retraining on the target database. Specifically, we propose to predict data distributions using schema semantics, thereby avoiding raw data dependence. Then, we introduce a query template-agnostic representation method to alleviate query dependence. Finally, we construct a large-scale query dataset derived from real-world tables and pretrain ZeroCard on it, enabling it to learn cardinality from schema semantics and predicate representations. After pretraining, ZeroCard's parameters can be frozen and applied in an off-the-shelf manner. We conduct extensive experiments to demonstrate the distinct advantages of ZeroCard and show its practical applications in query optimization. Its zero-dependence property significantly facilitates deployment in real-world scenarios. 

**Abstract (ZH)**: 基于语义的基数估计：零依赖方法 

---
# Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training 

**Title (ZH)**: 揭示多层次闲聊步的重要性：基于稳定性的去中心化训练泛化分析 

**Authors**: Qinglun Li, Yingqi Liu, Miao Zhang, Xiaochun Cao, Quanjun Yin, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.07980)  

**Abstract**: Decentralized training removes the centralized server, making it a communication-efficient approach that can significantly improve training efficiency, but it often suffers from degraded performance compared to centralized training. Multi-Gossip Steps (MGS) serve as a simple yet effective bridge between decentralized and centralized training, significantly reducing experiment performance gaps. However, the theoretical reasons for its effectiveness and whether this gap can be fully eliminated by MGS remain open questions. In this paper, we derive upper bounds on the generalization error and excess error of MGS using stability analysis, systematically answering these two key questions. 1). Optimization Error Reduction: MGS reduces the optimization error bound at an exponential rate, thereby exponentially tightening the generalization error bound and enabling convergence to better solutions. 2). Gap to Centralization: Even as MGS approaches infinity, a non-negligible gap in generalization error remains compared to centralized mini-batch SGD ($\mathcal{O}(T^{\frac{c\beta}{c\beta +1}}/{n m})$ in centralized and $\mathcal{O}(T^{\frac{2c\beta}{2c\beta +2}}/{n m^{\frac{1}{2c\beta +2}}})$ in decentralized). Furthermore, we provide the first unified analysis of how factors like learning rate, data heterogeneity, node count, per-node sample size, and communication topology impact the generalization of MGS under non-convex settings without the bounded gradients assumption, filling a critical theoretical gap in decentralized training. Finally, promising experiments on CIFAR datasets support our theoretical findings. 

**Abstract (ZH)**: 去中心化训练消除了中心服务器，使其成为一种通信高效的训练方法，能够显著提高训练效率，但通常会相比中心化训练表现出降级的性能。多闲聊步长（MGS）作为一种简单有效的去中心化与中心化训练之间的桥梁，显著减少了实验性能差距。然而，其有效性的理论原因以及MGS是否能完全消除这一差距仍是悬而未决的问题。在本文中，我们通过稳定性分析推导出MGS的一般化误差和超额误差的上界，系统地回答了这两个关键问题。1) 最优化误差减少：MGS以指数级减少最优化误差边界，从而以指数级紧化一般化误差边界，并使收敛于更好的解。2) 与中心化差距：即便MGS趋向无穷大，在一般化误差上与中心化小批量SGD仍存在不可忽视的差距（在中心化设置中为$\mathcal{O}(T^{\frac{c\beta}{c\beta +1}}/{nm})$，在去中心化设置中为$\mathcal{O}(T^{\frac{2c\beta}{2c\beta +2}}/{nm^{\frac{1}{2c\beta +2}}})$）。此外，我们在非凸设置下，没有假设梯度有界的情况下，首次对影响MGS一般化的关键因素（如学习率、数据异质性、节点数目、每个节点样本大小和通信拓扑）进行统一分析，填补了去中心化训练领域的重要理论空白。最后，CIFAR数据集上的有希望的实验结果支持了我们的理论发现。 

---
# A Systematic Evaluation of Self-Supervised Learning for Label-Efficient Sleep Staging with Wearable EEG 

**Title (ZH)**: 自我监督学习在可标记数据高效睡眠分期中的系统评估——基于可穿戴EEG信号 

**Authors**: Emilio Estevan, María Sierra-Torralba, Eduardo López-Larraz, Luis Montesano  

**Link**: [PDF](https://arxiv.org/pdf/2510.07960)  

**Abstract**: Wearable EEG devices have emerged as a promising alternative to polysomnography (PSG). As affordable and scalable solutions, their widespread adoption results in the collection of massive volumes of unlabeled data that cannot be analyzed by clinicians at scale. Meanwhile, the recent success of deep learning for sleep scoring has relied on large annotated datasets. Self-supervised learning (SSL) offers an opportunity to bridge this gap, leveraging unlabeled signals to address label scarcity and reduce annotation effort. In this paper, we present the first systematic evaluation of SSL for sleep staging using wearable EEG. We investigate a range of well-established SSL methods and evaluate them on two sleep databases acquired with the Ikon Sleep wearable EEG headband: BOAS, a high-quality benchmark containing PSG and wearable EEG recordings with consensus labels, and HOGAR, a large collection of home-based, self-recorded, and unlabeled recordings. Three evaluation scenarios are defined to study label efficiency, representation quality, and cross-dataset generalization. Results show that SSL consistently improves classification performance by up to 10% over supervised baselines, with gains particularly evident when labeled data is scarce. SSL achieves clinical-grade accuracy above 80% leveraging only 5% to 10% of labeled data, while the supervised approach requires twice the labels. Additionally, SSL representations prove robust to variations in population characteristics, recording environments, and signal quality. Our findings demonstrate the potential of SSL to enable label-efficient sleep staging with wearable EEG, reducing reliance on manual annotations and advancing the development of affordable sleep monitoring systems. 

**Abstract (ZH)**: 可穿戴EEG设备作为多导睡眠图(PSG)的有前途的替代方案已经 emergence 

---
# DISCO: Diversifying Sample Condensation for Efficient Model Evaluation 

**Title (ZH)**: DISCO: 多样化样本凝缩以实现高效模型评估 

**Authors**: Alexander Rubinstein, Benjamin Raible, Martin Gubri, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.07959)  

**Abstract**: Evaluating modern machine learning models has become prohibitively expensive. Benchmarks such as LMMs-Eval and HELM demand thousands of GPU hours per model. Costly evaluation reduces inclusivity, slows the cycle of innovation, and worsens environmental impact. The typical approach follows two steps. First, select an anchor subset of data. Second, train a mapping from the accuracy on this subset to the final test result. The drawback is that anchor selection depends on clustering, which can be complex and sensitive to design choices. We argue that promoting diversity among samples is not essential; what matters is to select samples that $\textit{maximise diversity in model responses}$. Our method, $\textbf{Diversifying Sample Condensation (DISCO)}$, selects the top-k samples with the greatest model disagreements. This uses greedy, sample-wise statistics rather than global clustering. The approach is conceptually simpler. From a theoretical view, inter-model disagreement provides an information-theoretically optimal rule for such greedy selection. $\textbf{DISCO}$ shows empirical gains over prior methods, achieving state-of-the-art results in performance prediction across MMLU, Hellaswag, Winogrande, and ARC. Code is available here: this https URL. 

**Abstract (ZH)**: 评估现代机器学习模型已经变得极为昂贵。基准测试如LMMs-Eval和HELM每模型需求数千个GPU小时。高昂的评估成本降低了包容性，减缓了创新周期，并恶化了环境影响。典型方法遵循两个步骤。首先，选择一个基准子数据集。其次，训练一个从该子集上的准确率到最终测试结果的映射。缺点是基准选择依赖于聚类，这可能复杂且对设计选择敏感。我们argue促进样本多样性并非必不可少；关键在于选择能够最大化模型响应多样性的样本。我们的方法**Diversifying Sample Condensation (DISCO)**选择具有最大模型分歧的前k个样本。该方法使用基于样本的贪心统计而非全局聚类。该方法概念上更简单。从理论上看，模型间分歧提供了这种贪心选择的信息论最佳规则。**DISCO**在MMLU、Hellaswag、Winogrande和ARC的性能预测中实现了优于前方法的结果。代码可在以下链接获取：this https URL。 

---
# A Large-scale Dataset for Robust Complex Anime Scene Text Detection 

**Title (ZH)**: 大规模数据集用于健壮的复杂动画场景文字检测 

**Authors**: Ziyi Dong, Yurui Zhang, Changmao Li, Naomi Rue Golding, Qing Long  

**Link**: [PDF](https://arxiv.org/pdf/2510.07951)  

**Abstract**: Current text detection datasets primarily target natural or document scenes, where text typically appear in regular font and shapes, monotonous colors, and orderly layouts. The text usually arranged along straight or curved lines. However, these characteristics differ significantly from anime scenes, where text is often diverse in style, irregularly arranged, and easily confused with complex visual elements such as symbols and decorative patterns. Text in anime scene also includes a large number of handwritten and stylized fonts. Motivated by this gap, we introduce AnimeText, a large-scale dataset containing 735K images and 4.2M annotated text blocks. It features hierarchical annotations and hard negative samples tailored for anime scenarios. %Cross-dataset evaluations using state-of-the-art methods demonstrate that models trained on AnimeText achieve superior performance in anime text detection tasks compared to existing datasets. To evaluate the robustness of AnimeText in complex anime scenes, we conducted cross-dataset benchmarking using state-of-the-art text detection methods. Experimental results demonstrate that models trained on AnimeText outperform those trained on existing datasets in anime scene text detection tasks. AnimeText on HuggingFace: this https URL 

**Abstract (ZH)**: 当前的文字检测数据集主要针对自然场景或文档场景，其中文字通常以常规字体和形状、单调的颜色和有序的布局出现。文字通常沿直线或曲线排列。然而，这些特征与动漫场景中的文字特性有显著差异，后者中的文字风格多样、排列不规则、容易与符号和装饰图案等复杂的视觉元素混淆。动漫场景中的文字包含大量手写和艺术化的字体。为弥合这一差距，我们引入了AnimeText数据集，包含735K张图像和4.2M个标注的文字区块。该数据集具有面向动漫场景的层次化标注和困难的负样本。跨数据集评估表明，使用AnimeText训练的模型在动漫文字检测任务上的表现优于现有数据集。为了评估AnimeText在复杂动漫场景中的鲁棒性，我们使用最先进的文字检测方法进行了跨数据集基准测试。实验结果表明，使用AnimeText训练的模型在动漫场景文字检测任务上的表现优于使用现有数据集训练的模型。AnimeText在HuggingFace：这个 https URL。 

---
# MMM: Quantum-Chemical Molecular Representation Learning for Combinatorial Drug Recommendation 

**Title (ZH)**: MMM：量子化学分子表示学习在组合药物推荐中的应用 

**Authors**: Chongmyung Kwon, Yujin Kim, Seoeun Park, Yunji Lee, Charmgil Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.07910)  

**Abstract**: Drug recommendation is an essential task in machine learning-based clinical decision support systems. However, the risk of drug-drug interactions (DDI) between co-prescribed medications remains a significant challenge. Previous studies have used graph neural networks (GNNs) to represent drug structures. Regardless, their simplified discrete forms cannot fully capture the molecular binding affinity and reactivity. Therefore, we propose Multimodal DDI Prediction with Molecular Electron Localization Function (ELF) Maps (MMM), a novel framework that integrates three-dimensional (3D) quantum-chemical information into drug representation learning. It generates 3D electron density maps using the ELF. To capture both therapeutic relevance and interaction risks, MMM combines ELF-derived features that encode global electronic properties with a bipartite graph encoder that models local substructure interactions. This design enables learning complementary characteristics of drug molecules. We evaluate MMM in the MIMIC-III dataset (250 drugs, 442 substructures), comparing it with several baseline models. In particular, a comparison with the GNN-based SafeDrug model demonstrates statistically significant improvements in the F1-score (p = 0.0387), Jaccard (p = 0.0112), and the DDI rate (p = 0.0386). These results demonstrate the potential of ELF-based 3D representations to enhance prediction accuracy and support safer combinatorial drug prescribing in clinical practice. 

**Abstract (ZH)**: 基于分子电子局域函数图的多模态药物-药物相互作用预测框架 

---
# Self-Supervised Learning Strategies for a Platform to Test the Toxicity of New Chemicals and Materials 

**Title (ZH)**: 自主监督学习策略用于测试新型化学物质和材料的毒性平台 

**Authors**: Thomas Lautenschlager, Nils Friederich, Angelo Jovin Yamachui Sitcheu, Katja Nau, Gaëlle Hayot, Thomas Dickmeis, Ralf Mikut  

**Link**: [PDF](https://arxiv.org/pdf/2510.07853)  

**Abstract**: High-throughput toxicity testing offers a fast and cost-effective way to test large amounts of compounds. A key component for such systems is the automated evaluation via machine learning models. In this paper, we address critical challenges in this domain and demonstrate how representations learned via self-supervised learning can effectively identify toxicant-induced changes. We provide a proof-of-concept that utilizes the publicly available EmbryoNet dataset, which contains ten zebrafish embryo phenotypes elicited by various chemical compounds targeting different processes in early embryonic development. Our analysis shows that the learned representations using self-supervised learning are suitable for effectively distinguishing between the modes-of-action of different compounds. Finally, we discuss the integration of machine learning models in a physical toxicity testing device in the context of the TOXBOX project. 

**Abstract (ZH)**: 高通量毒性测试提供了一种快速且成本有效的大量化合物测试方法。此类系统的关键组件是通过机器学习模型进行的自动化评估。在本文中，我们针对该领域的关键挑战进行了探讨，并展示了通过自监督学习学习的表示能够有效识别毒物诱导的变化。我们利用公开可用的EmbryoNet数据集提供了概念验证，该数据集包含由作用于早期胚胎发育不同过程的各种化学物质诱导的十种斑马鱼胚胎表型。我们的分析表明，通过自监督学习学习的表示适合于有效区分不同化合物的作用机制。最后，我们在TOXBOX项目背景下讨论了机器学习模型在物理毒性测试设备中的集成。 

---
# Meta-Learning Based Few-Shot Graph-Level Anomaly Detection 

**Title (ZH)**: 基于元学习的少样本图级别异常检测 

**Authors**: Liting Li, Yumeng Wang, Yueheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.07847)  

**Abstract**: Graph-level anomaly detection aims to identify anomalous graphs or subgraphs within graph datasets, playing a vital role in various fields such as fraud detection, review classification, and biochemistry. While Graph Neural Networks (GNNs) have made significant progress in this domain, existing methods rely heavily on large amounts of labeled data, which is often unavailable in real-world scenarios. Additionally, few-shot anomaly detection methods based on GNNs are prone to noise interference, resulting in poor embedding quality and reduced model robustness. To address these challenges, we propose a novel meta-learning-based graph-level anomaly detection framework (MA-GAD), incorporating a graph compression module that reduces the graph size, mitigating noise interference while retaining essential node information. We also leverage meta-learning to extract meta-anomaly information from similar networks, enabling the learning of an initialization model that can rapidly adapt to new tasks with limited samples. This improves the anomaly detection performance on target graphs, and a bias network is used to enhance the distinction between anomalous and normal nodes. Our experimental results, based on four real-world biochemical datasets, demonstrate that MA-GAD outperforms existing state-of-the-art methods in graph-level anomaly detection under few-shot conditions. Experiments on both graph anomaly and subgraph anomaly detection tasks validate the framework's effectiveness on real-world datasets. 

**Abstract (ZH)**: 图级别异常检测旨在识别图数据集中异常的图或子图，在欺诈检测、评论分类和生物化学等领域发挥着重要作用。尽管图神经网络（GNNs）在该领域取得了显著进展，但现有方法严重依赖大量标记数据，而在实际场景中此类数据往往不可用。此外，基于GNN的少样本异常检测方法容易受到噪声干扰，导致嵌入质量较差和模型鲁棒性降低。为应对这些挑战，我们提出了一种新的基于元学习的图级别异常检测框架（MA-GAD），结合了图压缩模块以减小图的大小，减轻噪声干扰同时保留关键节点信息。我们还利用元学习从相似网络中提取元异常信息，从而学习一种初始模型，该模型能够在有限样本的情况下快速适应新任务。这可以提高目标图上的异常检测性能，并使用偏差网络增强异常节点和正常节点之间的区别。基于四个真实的生物化学数据集的实验结果表明，在少样本条件下，MA-GAD在图级别异常检测中的性能优于现有最先进的方法。实验结果验证了该框架在真实数据集上的有效性。 

---
# The Rise of the Knowledge Sculptor: A New Archetype for Knowledge Work in the Age of Generative AI 

**Title (ZH)**: 知识塑造者崛起：生成式AI时代的新知识工作者原型 

**Authors**: Cathal Doyle  

**Link**: [PDF](https://arxiv.org/pdf/2510.07829)  

**Abstract**: In the Generative Age, the nature of knowledge work is transforming. Traditional models that emphasise the organisation and retrieval of pre-existing information are increasingly inadequate in the face of generative AI (GenAI) systems capable of autonomous content creation. This paper introduces the Knowledge Sculptor (KS), a new professional archetype for Human-GenAI collaboration that transforms raw AI output into trustworthy, actionable knowledge. Grounded in a socio-technical perspective, the KS is conceptualised through a framework of competencies, including architecting a vision, iterative dialogue, information sculpting, and curiosity-driven synthesis. A practice-based vignette illustrates the KS role in action, and in a self-referential approach, the paper itself serves as an artefact of the sculpting process it describes. 

**Abstract (ZH)**: 生成时代，知识工作的本质正在转变。传统的侧重组织和检索现有信息的模型在面对能够自主内容创造的生成型AI系统时越来越显得不足。本文介绍了知识雕刻师（KS），这是一种新的专业形象，旨在促进人类与生成型AI的合作，将原始AI输出转化为可信赖且可操作的知识。基于社会技术视角，KS通过愿景架构、迭代对话、信息雕刻和好奇心驱动的综合等能力框架进行概念化。通过基于实践的实例展示KS的角色，并采用一种自指性方法，本文本身作为描述过程中的一种产物。 

---
# A Unified Multi-Task Learning Framework for Generative Auto-Bidding with Validation-Aligned Optimization 

**Title (ZH)**: 统一的多任务学习框架：带有验证对齐优化的生成性自动出价 

**Authors**: Yiqin Lv, Zhiyu Mou, Miao Xu, Jinghao Chen, Qi Wang, Yixiu Mao, Yun Qu, Rongquan Bai, Chuan Yu, Jian Xu, Bo Zheng, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.07760)  

**Abstract**: In online advertising, heterogeneous advertiser requirements give rise to numerous customized bidding tasks that are typically optimized independently, resulting in extensive computation and limited data efficiency. Multi-task learning offers a principled framework to train these tasks jointly through shared representations. However, existing multi-task optimization strategies are primarily guided by training dynamics and often generalize poorly in volatile bidding environments. To this end, we present Validation-Aligned Multi-task Optimization (VAMO), which adaptively assigns task weights based on the alignment between per-task training gradients and a held-out validation gradient, thereby steering updates toward validation improvement and better matching deployment objectives. We further equip the framework with a periodicity-aware temporal module and couple it with an advanced generative auto-bidding backbone to enhance cross-task transfer of seasonal structure and strengthen bidding performance. Meanwhile, we provide theoretical insights into the proposed method, e.g., convergence guarantee and alignment analysis. Extensive experiments on both simulated and large-scale real-world advertising systems consistently demonstrate significant improvements over typical baselines, illuminating the effectiveness of the proposed approach. 

**Abstract (ZH)**: 基于验证对齐的多任务优化方法(VAMO)在在线广告中的应用 

---
# MeSH: Memory-as-State-Highways for Recursive Transformers 

**Title (ZH)**: MeSH: 记忆作为状态高速公路的递归变压器 

**Authors**: Chengting Yu, Xiaobo Shu, Yadao Wang, Yizhen Zhang, Haoyi Wu, Jiaang Li, Rujiao Long, Ziheng Chen, Yuchi Xu, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07739)  

**Abstract**: Recursive transformers reuse parameters and iterate over hidden states multiple times, decoupling compute depth from parameter depth. However, under matched compute, recursive models with fewer parameters often lag behind non-recursive counterparts. By probing hidden states, we trace this performance gap to two primary bottlenecks: undifferentiated computation, where the core is forced to adopt a similar computational pattern at every iteration, and information overload, where long-lived and transient information must coexist in a single hidden state. To address the issues, we introduce a Memory-as-State-Highways (MeSH) scheme, which externalizes state management into an explicit memory buffer and employs lightweight routers to dynamically diversify computation across iterations. Probing visualizations confirm that MeSH successfully resolves the pathologies by inducing functional specialization across iterations. On the Pythia suite (160M-1.4B), MeSH-enhanced recursive transformers consistently improve over recursive baselines and outperforms its larger non-recursive counterpart at the 1.4B scale, improving average downstream accuracy by +1.06% with 33% fewer non-embedding parameters. Our analysis establishes MeSH as a scalable and principled architecture for building stronger recursive models. 

**Abstract (ZH)**: 递归Transformer通过参数共享和多轮迭代隐藏状态，解耦计算深度和参数深度。然而，在匹配计算资源的情况下，参数较少的递归模型往往落后于非递归模型。通过探究隐藏状态，我们将性能差距归因于两个主要瓶颈：未分化的计算模式，即核心在每次迭代中被迫采用类似的计算模式，以及信息过载，长寿命和临时信息必须在同一隐藏状态中共存。为了解决这些问题，我们提出了Memory-as-State-Highways (MeSH)方案，该方案将状态管理外部化到显式的内存缓冲区中，并使用轻量级路由器在迭代间动态多样化计算。视觉探针证实MeSH成功解决了这些病态问题，通过在迭代间诱导功能特化。在Pythia套件（160M-1.4B规模）上，MeSH增强的递归变压器在所有基准上表现出优性能，并在1.4B规模上优于其更大的非递归模型，平均下游准确率提高了1.06%，同时非嵌入参数减少了33%。我们的分析确立了MeSH作为一种可扩展且合理的递归模型架构。 

---
# Causality Guided Representation Learning for Cross-Style Hate Speech Detection 

**Title (ZH)**: 因果引导的表示学习在跨风格仇恨言论检测中的应用 

**Authors**: Chengshuai Zhao, Shu Wan, Paras Sheth, Karan Patwa, K. Selçuk Candan, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07707)  

**Abstract**: The proliferation of online hate speech poses a significant threat to the harmony of the web. While explicit hate is easily recognized through overt slurs, implicit hate speech is often conveyed through sarcasm, irony, stereotypes, or coded language -- making it harder to detect. Existing hate speech detection models, which predominantly rely on surface-level linguistic cues, fail to generalize effectively across diverse stylistic variations. Moreover, hate speech spread on different platforms often targets distinct groups and adopts unique styles, potentially inducing spurious correlations between them and labels, further challenging current detection approaches. Motivated by these observations, we hypothesize that the generation of hate speech can be modeled as a causal graph involving key factors: contextual environment, creator motivation, target, and style. Guided by this graph, we propose CADET, a causal representation learning framework that disentangles hate speech into interpretable latent factors and then controls confounders, thereby isolating genuine hate intent from superficial linguistic cues. Furthermore, CADET allows counterfactual reasoning by intervening on style within the latent space, naturally guiding the model to robustly identify hate speech in varying forms. CADET demonstrates superior performance in comprehensive experiments, highlighting the potential of causal priors in advancing generalizable hate speech detection. 

**Abstract (ZH)**: 在线仇恨言论的泛滥对网络和谐构成显著威胁。虽然明确的仇恨言论通过明显的污言秽语易于识别，但隐含的仇恨言论常常通过讽刺、反语、刻板印象或隐含语言来传达——使其更难检测。现有的仇恨言论检测模型主要依赖于表面语言线索，未能有效地跨不同风格变体进行推广。此外，不同平台上仇恨言论的传播往往针对不同的群体，并采用独特的风格，这可能会在平台和标签之间诱导虚假相关性，进一步挑战当前的检测方法。受这些观察的启发，我们假设仇恨言论的生成可以建模为涉及关键因素的因果图：上下文环境、创作者动机、目标和风格。基于这一图，我们提出了CADET因果表示学习框架，将仇恨言论分解为可解释的潜在因素，然后控制混杂变量，从而将真实的仇恨意图与表面的语言线索区分开来。此外，CADET通过在潜在空间中干预风格来支持假设性推理，自然引导模型在不同形式中稳健地检测仇恨言论。在全面的实验中，CADET展示了优越的性能，突显了因果先验在推动泛化仇恨言论检测方面的潜力。 

---
# IKNet: Interpretable Stock Price Prediction via Keyword-Guided Integration of News and Technical Indicators 

**Title (ZH)**: IKNet: 基于关键词引导的新闻与技术指标集成的可解释股票价格预测 

**Authors**: Jinwoong Kim, Sangjin Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.07661)  

**Abstract**: The increasing influence of unstructured external information, such as news articles, on stock prices has attracted growing attention in financial markets. Despite recent advances, most existing newsbased forecasting models represent all articles using sentiment scores or average embeddings that capture the general tone but fail to provide quantitative, context-aware explanations of the impacts of public sentiment on predictions. To address this limitation, we propose an interpretable keyword-guided network (IKNet), which is an explainable forecasting framework that models the semantic association between individual news keywords and stock price movements. The IKNet identifies salient keywords via FinBERTbased contextual analysis, processes each embedding through a separate nonlinear projection layer, and integrates their representations with the time-series data of technical indicators to forecast next-day closing prices. By applying Shapley Additive Explanations the model generates quantifiable and interpretable attributions for the contribution of each keyword to predictions. Empirical evaluations of S&P 500 data from 2015 to 2024 demonstrate that IKNet outperforms baselines, including recurrent neural networks and transformer models, reducing RMSE by up to 32.9% and improving cumulative returns by 18.5%. Moreover, IKNet enhances transparency by offering contextualized explanations of volatility events driven by public sentiment. 

**Abstract (ZH)**: 非结构化外部信息（如新闻文章）对股票价格日益增长的影响在金融市场上引起了广泛关注。尽管近期取得了一些进展，但现有的大多数基于新闻的预测模型通过情感评分或平均词嵌入来表示所有文章，这些方法虽然捕捉了整体情绪，但无法提供定量且具有上下文 awareness 的解释，说明公众情绪对预测的影响。为解决这一局限，我们提出了一种可解释的关键词导向网络（IKNet），这是一种可解释的预测框架，用于建模单个新闻关键词与股票价格变动之间的语义关联。IKNet 通过基于 FinBERT 的上下文分析识别关键关键词，每个嵌入通过单独的非线性投影层处理，并将它们的表示与技术指标的时间序列数据集成以预测次日收盘价。通过应用 Shapley 加性解释，该模型为每个关键词对预测的贡献生成了可量化且可解释的归因。对2015年至2024年标普500指数数据的实证评估表明，IKNet 的表现优于包括循环神经网络和变换器模型在内的基线模型，减少了高达32.9%的 RMSE，并提高了累计回报率18.5%。此外，IKNet 提高了透明度，通过对由公众情绪驱动的波动事件提供上下文化的解释。 

---
# Value Flows 

**Title (ZH)**: 价值流动 

**Authors**: Perry Dong, Chongyi Zheng, Chelsea Finn, Dorsa Sadigh, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2510.07650)  

**Abstract**: While most reinforcement learning methods today flatten the distribution of future returns to a single scalar value, distributional RL methods exploit the return distribution to provide stronger learning signals and to enable applications in exploration and safe RL. While the predominant method for estimating the return distribution is by modeling it as a categorical distribution over discrete bins or estimating a finite number of quantiles, such approaches leave unanswered questions about the fine-grained structure of the return distribution and about how to distinguish states with high return uncertainty for decision-making. The key idea in this paper is to use modern, flexible flow-based models to estimate the full future return distributions and identify those states with high return variance. We do so by formulating a new flow-matching objective that generates probability density paths satisfying the distributional Bellman equation. Building upon the learned flow models, we estimate the return uncertainty of distinct states using a new flow derivative ODE. We additionally use this uncertainty information to prioritize learning a more accurate return estimation on certain transitions. We compare our method (Value Flows) with prior methods in the offline and online-to-online settings. Experiments on $37$ state-based and $25$ image-based benchmark tasks demonstrate that Value Flows achieves a $1.3\times$ improvement on average in success rates. Website: this https URL Code: this https URL 

**Abstract (ZH)**: 虽然当前大多数增强学习方法将未来回报分布压平为单一标量值，分布型增强学习方法利用回报分布提供更强的学习信号，并能够促进探索和安全增强学习的应用。尽管估计回报分布的主要方法是将其建模为离散区间上的分类分布或估计有限数量的分位数，但这些方法未能回答回报分布的细微结构以及如何区分具有高回报不确定性状态以辅助决策的问题。本文的关键思想是使用现代灵活的流动模型估计完整的未来回报分布，并识别具有高回报方差的状态。我们通过制定一个新的流动匹配目标来实现这一点，该目标生成满足分布贝尔曼方程的概率密度路径。基于学习到的流动模型，我们使用一个新的流动导数常微分方程来估计不同状态的回报不确定性。此外，我们还利用这些不确定性信息来优先学习某些转换上的更准确回报估计。我们在离线和在线到在线设置中将我们的方法（价值流动）与前期方法进行了比较。在37个基于状态和25个基于图像的基准任务上的实验表明，价值流动在成功率上平均提高了1.3倍。网址：这个 https URL 代码：这个 https URL。 

---
# Retentive Relevance: Capturing Long-Term User Value in Recommendation Systems 

**Title (ZH)**: 持久相关性：捕获推荐系统中的长期用户价值 

**Authors**: Saeideh Bakhshi, Phuong Mai Nguyen, Robert Schiller, Tiantian Xu, Pawan Kodandapani, Andrew Levine, Cayman Simpson, Qifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07621)  

**Abstract**: Recommendation systems have traditionally relied on short-term engagement signals, such as clicks and likes, to personalize content. However, these signals are often noisy, sparse, and insufficient for capturing long-term user satisfaction and retention. We introduce Retentive Relevance, a novel content-level survey-based feedback measure that directly assesses users' intent to return to the platform for similar content. Unlike other survey measures that focus on immediate satisfaction, Retentive Relevance targets forward-looking behavioral intentions, capturing longer term user intentions and providing a stronger predictor of retention. We validate Retentive Relevance using psychometric methods, establishing its convergent, discriminant, and behavioral validity. Through large-scale offline modeling, we show that Retentive Relevance significantly outperforms both engagement signals and other survey measures in predicting next-day retention, especially for users with limited historical engagement. We develop a production-ready proxy model that integrates Retentive Relevance into the final stage of a multi-stage ranking system on a social media platform. Calibrated score adjustments based on this model yield substantial improvements in engagement, and retention, while reducing exposure to low-quality content, as demonstrated by large-scale A/B experiments. This work provides the first empirically validated framework linking content-level user perceptions to retention outcomes in production systems. We offer a scalable, user-centered solution that advances both platform growth and user experience. Our work has broad implications for responsible AI development. 

**Abstract (ZH)**: 一种基于内容的调查反馈度量Retentive Relevance及其在提升用户留存中的应用 

---
# DGTEN: A Robust Deep Gaussian based Graph Neural Network for Dynamic Trust Evaluation with Uncertainty-Quantification Support 

**Title (ZH)**: DGTEN：一种具有不确定性量化支持的鲁棒深度高斯图神经网络动态信任评价方法 

**Authors**: Muhammad Usman, Yugyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.07620)  

**Abstract**: Dynamic trust evaluation in large, rapidly evolving graphs requires models that can capture changing relationships, express calibrated confidence, and resist adversarial manipulation. DGTEN (Deep Gaussian-based Trust Evaluation Network) introduces a unified graph framework that achieves all three by combining uncertainty-aware message passing, expressive temporal modeling, and built-in defenses against trust-targeted attacks. It represents nodes and edges as Gaussian distributions so that both semantic signals and epistemic uncertainty propagate through the graph neural network, enabling risk-aware trust decisions rather than overconfident guesses. To model how trust evolves, it employs hybrid Absolute-Gaussian-Hourglass (HAGH) positional encoding with Kolmogorov-Arnold network-based unbiased multi-head attention, followed by an ordinary differential equation (ODE)-based residual learning module to jointly capture abrupt shifts and smooth trends. Robust adaptive ensemble coefficient analysis prunes or down-weights suspicious interactions using complementary cosine and Jaccard similarity measures, mitigating reputation laundering, sabotage, and on/off attacks. On two signed Bitcoin trust networks, DGTEN delivers significant improvements: in single-timeslot prediction on Bitcoin-Alpha, it improves MCC by 10.77% over the best dynamic baseline; in the cold-start scenario, it achieves a 16.41% MCC gain - the largest across all tasks and datasets. Under adversarial on/off attacks, it surpasses the baseline by up to 11.63% MCC. These results validate the effectiveness of the unified DGTEN framework. 

**Abstract (ZH)**: 动态大型快速演化图中的信任评估需要能够捕捉关系变化、表达校准的信心并抵抗 adversarial 操作的模型。DGTEN（基于深度高斯的信任评估网络）通过结合不确定性意识的消息传递、表现力时间建模以及内置的针对信任目标攻击的防护机制，实现上述所有目标。它将节点和边表示为高斯分布，使得语义信号和Epistemic不确定性通过图神经网络传播，从而实现风险意识的信任决策，而非过度自信的猜测。为了建模信任如何演变，它使用了混合绝寂数学-高斯-漏斗（HAGH）位置编码与基于Kolmogorov-Arnold神经网络的无偏多头注意机制，随后通过基于常微分方程（ODE）的残差学习模块来共同捕捉突变和平稳趋势。鲁棒自适应集成系数分析使用互补的余弦相似性和Jaccard相似性度量来消除或降低可疑交互的影响，从而减轻声誉洗钱、破坏和开停攻击。在两个带有签名的比特币信任网络上，DGTEN实现了显著的改进：在单时隙预测方面，它在Bitcoin-Alpha中将MCC提高了10.77%，这是最佳动态基线的最佳表现；在冷启动场景中，它实现了16.41%的MCC增益，这是所有任务和数据集中最大的增益。在对抗开停攻击的情况下，它将MCC提高了最多11.63%。这些结果验证了统一的DGTEN框架的有效性。 

---
# TGM: a Modular and Efficient Library for Machine Learning on Temporal Graphs 

**Title (ZH)**: TGM：用于时序图机器学习的模块化高效库 

**Authors**: Jacob Chmura, Shenyang Huang, Tran Gia Bao Ngo, Ali Parviz, Farimah Poursafaei, Jure Leskovec, Michael Bronstein, Guillaume Rabusseau, Matthias Fey, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2510.07586)  

**Abstract**: Well-designed open-source software drives progress in Machine Learning (ML) research. While static graph ML enjoys mature frameworks like PyTorch Geometric and DGL, ML for temporal graphs (TG), networks that evolve over time, lacks comparable infrastructure. Existing TG libraries are often tailored to specific architectures, hindering support for diverse models in this rapidly evolving field. Additionally, the divide between continuous- and discrete-time dynamic graph methods (CTDG and DTDG) limits direct comparisons and idea transfer. To address these gaps, we introduce Temporal Graph Modelling (TGM), a research-oriented library for ML on temporal graphs, the first to unify CTDG and DTDG approaches. TGM offers first-class support for dynamic node features, time-granularity conversions, and native handling of link-, node-, and graph-level tasks. Empirically, TGM achieves an average 7.8x speedup across multiple models, datasets, and tasks compared to the widely used DyGLib, and an average 175x speedup on graph discretization relative to available implementations. Beyond efficiency, we show in our experiments how TGM unlocks entirely new research possibilities by enabling dynamic graph property prediction and time-driven training paradigms, opening the door to questions previously impractical to study. TGM is available at this https URL 

**Abstract (ZH)**: 精心设计的开源软件推动机器学习研究进展。虽然静态图机器学习享有成熟的框架如PyTorch Geometric和DGL，但时间图（随时间演变的网络）的机器学习领域缺乏类似的基础设施。现有的时间图库往往针对特定架构，阻碍了对该迅速发展的领域中各种模型的支持。此外，连续时间和离散时间动态图方法（CTDG和DTDG）之间的差距限制了直接比较和思想转移。为了解决这些差距，我们介绍了时间图建模（TGM），这是一个面向研究的时间图机器学习库，首次统一了CTDG和DTDG方法。TGM提供了一流的支持动态节点特征、时间粒度转换以及对边级、节点级和图级任务的原生处理。实证研究表明，与广泛使用的DyGLib相比，TGM在多个模型、数据集和任务上的平均加速倍数为7.8倍，而在图离散化方面的平均加速倍数为175倍。除了效率，我们的实验展示了TGM如何通过启用动态图属性预测和时间驱动的训练范式，解锁全新的研究可能性，开启以前难以研究的问题。TGM可在以下链接获取：this https URL 

---
# Linguistic Patterns in Pandemic-Related Content: A Comparative Analysis of COVID-19, Constraint, and Monkeypox Datasets 

**Title (ZH)**: 疫情相关语料中的语言模式：COVID-19、Pandemics和Monkeypox数据集的比较分析 

**Authors**: Mkululi Sikosana, Sean Maudsley-Barton, Oluwaseun Ajao  

**Link**: [PDF](https://arxiv.org/pdf/2510.07579)  

**Abstract**: This study conducts a computational linguistic analysis of pandemic-related online discourse to examine how language distinguishes health misinformation from factual communication. Drawing on three corpora: COVID-19 false narratives (n = 7588), general COVID-19 content (n = 10700), and Monkeypox-related posts (n = 5787), we identify significant differences in readability, rhetorical markers, and persuasive language use. COVID-19 misinformation exhibited markedly lower readability scores and contained over twice the frequency of fear-related or persuasive terms compared to the other datasets. It also showed minimal use of exclamation marks, contrasting with the more emotive style of Monkeypox content. These patterns suggest that misinformation employs a deliberately complex rhetorical style embedded with emotional cues, a combination that may enhance its perceived credibility. Our findings contribute to the growing body of work on digital health misinformation by highlighting linguistic indicators that may aid detection efforts. They also inform public health messaging strategies and theoretical models of crisis communication in networked media environments. At the same time, the study acknowledges limitations, including reliance on traditional readability indices, use of a deliberately narrow persuasive lexicon, and reliance on static aggregate analysis. Future research should therefore incorporate longitudinal designs, broader emotion lexicons, and platform-sensitive approaches to strengthen robustness. 

**Abstract (ZH)**: 本研究通过计算语言学分析与疫情相关的在线 discourse，以探讨语言如何区分健康 misinformation 和事实性沟通。本研究基于三个语料库：COVID-19 错误叙事（n = 7588）、一般 COVID-19 内容（n = 10700）和猴痘相关帖子（n = 5787），识别出流畅度、修辞标志和说服性语言使用方面的显著差异。COVID-19  misinformation 的流畅度明显较低，同时包含比其他数据集两倍以上的恐惧相关或说服性术语。此外，它在使用感叹号方面也很少，与猴痘内容更具情绪化的风格形成对比。这些模式表明，misinformation 采用了一种刻意复杂的修辞风格，其中嵌入了情感线索，这种结合可能增强其感知可信度。研究结果为数字健康 misinformation 的研究领域增添了新的语言指标，有助于检测努力。它们还为公共卫生信息沟通策略提供了依据，并为网络媒体环境中危机沟通的理论模型提供建议。同时，本研究承认其局限性，包括依赖传统的流畅度指标、使用刻意狭窄的说服词汇库以及依赖于静态的总体分析。未来研究应采用纵向设计、更广泛的情绪词汇库和平台敏感的方法，以增强研究的稳健性。

---

This study conducts a computational linguistic analysis of pandemic-related online discourse to examine how language distinguishes health misinformation from factual communication. 

---
# Accuracy, Memory Efficiency and Generalization: A Comparative Study on Liquid Neural Networks and Recurrent Neural Networks 

**Title (ZH)**: 准确度、内存效率和泛化能力：液态神经网络与循环神经网络的比较研究 

**Authors**: Shilong Zong, Alex Bierly, Almuatazbellah Boker, Hoda Eldardiry  

**Link**: [PDF](https://arxiv.org/pdf/2510.07578)  

**Abstract**: This review aims to conduct a comparative analysis of liquid neural networks (LNNs) and traditional recurrent neural networks (RNNs) and their variants, such as long short-term memory networks (LSTMs) and gated recurrent units (GRUs). The core dimensions of the analysis include model accuracy, memory efficiency, and generalization ability. By systematically reviewing existing research, this paper explores the basic principles, mathematical models, key characteristics, and inherent challenges of these neural network architectures in processing sequential data. Research findings reveal that LNN, as an emerging, biologically inspired, continuous-time dynamic neural network, demonstrates significant potential in handling noisy, non-stationary data, and achieving out-of-distribution (OOD) generalization. Additionally, some LNN variants outperform traditional RNN in terms of parameter efficiency and computational speed. However, RNN remains a cornerstone in sequence modeling due to its mature ecosystem and successful applications across various tasks. This review identifies the commonalities and differences between LNNs and RNNs, summarizes their respective shortcomings and challenges, and points out valuable directions for future research, particularly emphasizing the importance of improving the scalability of LNNs to promote their application in broader and more complex scenarios. 

**Abstract (ZH)**: 本综述旨在比较分析液态神经网络（LNNs）和传统循环神经网络（RNNs）及其变种（如长短期记忆网络LSTMs和门控循环单元GRUs）在处理序列数据方面的模型精度、内存效率和泛化能力。通过系统回顾现有研究，本文探讨了这些神经网络架构的基本原理、数学模型、关键特性和固有挑战。研究表明，作为一种新兴的、受生物学启发的连续时间动态神经网络，LNN显现出在处理噪声和非稳态数据以及实现领域外泛化方面的显著潜力。此外，某些LNN变种在参数效率和计算速度方面优于传统RNN。然而，由于其成熟生态系统和在多种任务中的成功应用，RNN仍然是序列建模的基石。本文指出了LNNs和RNNs之间的共同点与异同，总结了各自的不足和挑战，并指出了未来研究的重要方向，尤其是强调了提高LNNs可扩展性的重要性，以促进其在更广泛和更复杂场景中的应用。 

---
# EEG Sleep Stage Classification with Continuous Wavelet Transform and Deep Learning 

**Title (ZH)**: 基于连续小波变换和深度学习的EEG睡眠阶段分类 

**Authors**: Mehdi Zekriyapanah Gashti, Ghasem Farjamnia  

**Link**: [PDF](https://arxiv.org/pdf/2510.07524)  

**Abstract**: Accurate classification of sleep stages is crucial for the diagnosis and management of sleep disorders. Conventional approaches for sleep scoring rely on manual annotation or features extracted from EEG signals in the time or frequency domain. This study proposes a novel framework for automated sleep stage scoring using time-frequency analysis based on the wavelet transform. The Sleep-EDF Expanded Database (sleep-cassette recordings) was used for evaluation. The continuous wavelet transform (CWT) generated time-frequency maps that capture both transient and oscillatory patterns across frequency bands relevant to sleep staging. Experimental results demonstrate that the proposed wavelet-based representation, combined with ensemble learning, achieves an overall accuracy of 88.37 percent and a macro-averaged F1 score of 73.15, outperforming conventional machine learning methods and exhibiting comparable or superior performance to recent deep learning approaches. These findings highlight the potential of wavelet analysis for robust, interpretable, and clinically applicable sleep stage classification. 

**Abstract (ZH)**: 准确的睡眠阶段分类对于睡眠障碍的诊断和管理至关重要。传统的睡眠评分方法依赖于手动注释或从EEG信号的时间或频率域提取的特征。本研究提出了一种基于小波变换的时频分析框架，以实现自动化的睡眠阶段评分。采用睡眠-EDF扩展数据库（睡眠磁带记录）进行评估。连续小波变换（CWT）生成了时频图，捕捉了与睡眠阶段划分相关的频率带上的瞬态和振荡模式。实验结果表明，所提出的小波基表示方法结合集成学习实现了88.37%的整体准确率和73.15%的宏平均F1分数，优于传统的机器学习方法，并且在性能上与最近的深度学习方法相当或更优。这些发现突显了小波分析在实现稳健、可解释且临床可适用的睡眠阶段分类中的潜力。 

---
# A Denoising Framework for Real-World Ultra-Low Dose Lung CT Images Based on an Image Purification Strategy 

**Title (ZH)**: 基于图像净化策略的实时超低剂量肺部CT图像去噪框架 

**Authors**: Guoliang Gong, Man Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07492)  

**Abstract**: Ultra-low dose CT (uLDCT) significantly reduces radiation exposure but introduces severe noise and artifacts. It also leads to substantial spatial misalignment between uLDCT and normal dose CT (NDCT) image pairs. This poses challenges for directly applying existing denoising networks trained on synthetic noise or aligned data. To address this core challenge in uLDCT denoising, this paper proposes an innovative denoising framework based on an Image Purification (IP) strategy. First, we construct a real clinical uLDCT lung dataset. Then, we propose an Image Purification strategy that generates structurally aligned uLDCT-NDCT image pairs, providing a high-quality data foundation for network training. Building upon this, we propose a Frequency-domain Flow Matching (FFM) model, which works synergistically with the IP strategy to excellently preserve the anatomical structure integrity of denoised images. Experiments on the real clinical dataset demonstrate that our IP strategy significantly enhances the performance of multiple mainstream denoising models on the uLDCT task. Notably, our proposed FFM model combined with the IP strategy achieves state-of-the-art (SOTA) results in anatomical structure preservation. This study provides an effective solution to the data mismatch problem in real-world uLDCT denoising. Code and dataset are available at this https URL. 

**Abstract (ZH)**: 基于图像净化策略的超低剂量CT去噪框架 

---
# HEMERA: A Human-Explainable Transformer Model for Estimating Lung Cancer Risk using GWAS Data 

**Title (ZH)**: HEMERA：一种用于估计肺癌风险的人类可解释的变压器模型（基于GWAS数据） 

**Authors**: Maria Mahbub, Robert J. Klein, Myvizhi Esai Selvan, Rowena Yip, Claudia Henschke, Providencia Morales, Ian Goethert, Olivera Kotevska, Mayanka Chandra Shekar, Sean R. Wilkinson, Eileen McAllister, Samuel M. Aguayo, Zeynep H. Gümüş, Ioana Danciu, VA Million Veteran Program  

**Link**: [PDF](https://arxiv.org/pdf/2510.07477)  

**Abstract**: Lung cancer (LC) is the third most common cancer and the leading cause of cancer deaths in the US. Although smoking is the primary risk factor, the occurrence of LC in never-smokers and familial aggregation studies highlight a genetic component. Genetic biomarkers identified through genome-wide association studies (GWAS) are promising tools for assessing LC risk. We introduce HEMERA (Human-Explainable Transformer Model for Estimating Lung Cancer Risk using GWAS Data), a new framework that applies explainable transformer-based deep learning to GWAS data of single nucleotide polymorphisms (SNPs) for predicting LC risk. Unlike prior approaches, HEMERA directly processes raw genotype data without clinical covariates, introducing additive positional encodings, neural genotype embeddings, and refined variant filtering. A post hoc explainability module based on Layer-wise Integrated Gradients enables attribution of model predictions to specific SNPs, aligning strongly with known LC risk loci. Trained on data from 27,254 Million Veteran Program participants, HEMERA achieved >99% AUC (area under receiver characteristics) score. These findings support transparent, hypothesis-generating models for personalized LC risk assessment and early intervention. 

**Abstract (ZH)**: 人类可解释变压器模型在GWAS数据中估计肺癌风险（.HEMERA：人类可解释变压器模型用于基于GWAS数据估计肺癌风险） 

---
# MoGU: Mixture-of-Gaussians with Uncertainty-based Gating for Time Series Forecasting 

**Title (ZH)**: MoGU：基于不确定性门控的混合高斯模型的时间序列预测 

**Authors**: Yoli Shavit, Jacob Goldberger  

**Link**: [PDF](https://arxiv.org/pdf/2510.07459)  

**Abstract**: We introduce Mixture-of-Gaussians with Uncertainty-based Gating (MoGU), a novel Mixture-of-Experts (MoE) framework designed for regression tasks and applied to time series forecasting. Unlike conventional MoEs that provide only point estimates, MoGU models each expert's output as a Gaussian distribution. This allows it to directly quantify both the forecast (the mean) and its inherent uncertainty (variance). MoGU's core innovation is its uncertainty-based gating mechanism, which replaces the traditional input-based gating network by using each expert's estimated variance to determine its contribution to the final prediction. Evaluated across diverse time series forecasting benchmarks, MoGU consistently outperforms single-expert models and traditional MoE setups. It also provides well-quantified, informative uncertainties that directly correlate with prediction errors, enhancing forecast reliability. Our code is available from: this https URL 

**Abstract (ZH)**: 基于不确定性门控的混合高斯模型（MoGU）：一种用于时间序列预测的新型混合专家框架 

---
# Minimizing the Value-at-Risk of Loan Portfolio via Deep Neural Networks 

**Title (ZH)**: 通过深度神经网络最小化贷款组合的价值-at-风险 

**Authors**: Albert Di Wang, Ye Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.07444)  

**Abstract**: Risk management is a prominent issue in peer-to-peer lending. An investor may naturally reduce his risk exposure by diversifying instead of putting all his money on one loan. In that case, an investor may want to minimize the Value-at-Risk (VaR) or Conditional Value-at-Risk (CVaR) of his loan portfolio. We propose a low degree of freedom deep neural network model, DeNN, as well as a high degree of freedom model, DSNN, to tackle the problem. In particular, our models predict not only the default probability of a loan but also the time when it will default. The experiments demonstrate that both models can significantly reduce the portfolio VaRs at different confidence levels, compared to benchmarks. More interestingly, the low degree of freedom model, DeNN, outperforms DSNN in most scenarios. 

**Abstract (ZH)**: P2P借贷中的风险管理是一个突出的问题。投资者可以通过多元化而不是将所有资金集中在一笔贷款上来自然地降低风险暴露。在这种情况下，投资者可能希望最小化其贷款组合的价值-at-风险（VaR）或条件价值-at-风险（CVaR）。我们提出了一种低自由度的深度神经网络模型DeNN以及一种高自由度模型DSNN来解决这一问题。特别是，我们的模型不仅能预测一笔贷款的违约概率，还能预测其违约时间。实验结果表明，这两种模型都能在不同的置信水平下显著降低组合VaR，相较于基准模型。更有趣的是，在大多数情景下，低自由度模型DeNN表现优于DSNN。 

---
# Quantum Grid Path Planning Using Parallel QAOA Circuits Based on Minimum Energy Principle 

**Title (ZH)**: 基于最小能量原则的并行QAOA电路量子网格路径规划 

**Authors**: Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07413)  

**Abstract**: To overcome the bottleneck of classical path planning schemes in solving NP problems and address the predicament faced by current mainstream quantum path planning frameworks in the Noisy Intermediate-Scale Quantum (NISQ) era, this study attempts to construct a quantum path planning solution based on parallel Quantum Approximate Optimization Algorithm (QAOA) architecture. Specifically, the grid path planning problem is mapped to the problem of finding the minimum quantum energy state. Two parallel QAOA circuits are built to simultaneously execute two solution processes, namely connectivity energy calculation and path energy calculation. A classical algorithm is employed to filter out unreasonable solutions of connectivity energy, and finally, the approximate optimal solution to the path planning problem is obtained by merging the calculation results of the two parallel circuits. The research findings indicate that by setting appropriate filter parameters, quantum states corresponding to position points with extremely low occurrence probabilities can be effectively filtered out, thereby increasing the probability of obtaining the target quantum state. Even when the circuit layer number p is only 1, the theoretical solution of the optimal path coding combination can still be found by leveraging the critical role of the filter. Compared with serial circuits, parallel circuits exhibit a significant advantage, as they can find the optimal feasible path coding combination with the highest probability. 

**Abstract (ZH)**: 针对经典路径规划方案在解决NP问题时遇到的瓶颈以及当前主流量子路径规划框架在Noisy Intermediate-Scale Quantum (NISQ) 时代面临的困境，本研究尝试基于并行量子近似优化算法（QAOA）架构构建一种量子路径规划解决方案。具体而言，将网格路径规划问题映射为寻找最低量子能态的问题。构建两个并行的QAOA电路，同时执行连接能计算和路径能计算两个解决方案过程。采用经典算法对连接能的不合理解进行筛选，最终通过合并两个并行电路的计算结果获得路径规划问题的近似最优解。研究结果表明，通过设置合适的筛选参数，可以有效滤除发生概率极低的位置点对应的量子态，从而提高获得目标量子态的概率。即使在电路层数p仅为1的情况下，通过充分发挥筛选的 critical 角色，仍可以找到最优路径编码组合的理论解决方案。与串行电路相比，并行电路展现出显著优势，能够以最高概率找到最优可行路径编码组合。 

---
# Attention to Order: Transformers Discover Phase Transitions via Learnability 

**Title (ZH)**: 关注顺序：Transformer通过可学习性发现相变 

**Authors**: Şener Özönder  

**Link**: [PDF](https://arxiv.org/pdf/2510.07401)  

**Abstract**: Phase transitions mark qualitative reorganizations of collective behavior, yet identifying their boundaries remains challenging whenever analytic solutions are absent and conventional simulations fail. Here we introduce learnability as a universal criterion, defined as the ability of a transformer model containing attention mechanism to extract structure from microscopic states. Using self-supervised learning and Monte Carlo generated configurations of the two-dimensional Ising model, we show that ordered phases correspond to enhanced learnability, manifested in both reduced training loss and structured attention patterns, while disordered phases remain resistant to learning. Two unsupervised diagnostics, the sharp jump in training loss and the rise in attention entropy, recover the critical temperature in excellent agreement with the exact value. Our results establish learnability as a data-driven marker of phase transitions and highlight deep parallels between long-range order in condensed matter and the emergence of structure in modern language models. 

**Abstract (ZH)**: 相变标记集体行为的质变重组，但在缺乏解析解且常规模拟失效时，确定其边界依然具有挑战性。在这里我们引入可学习性作为普适标准，定义为包含注意机制的变换器模型从微观状态中提取结构的能力。使用自监督学习和蒙特卡洛生成的二维伊辛模型配置，我们证明有序相对应于增强的可学习性，表现为训练损失降低和受结构化的注意力模式，而无序相则对学习保持抵抗。两种无监督诊断，训练损失的sharp跃变和注意力熵的上升，能够以极佳的精度恢复临界温度，与准确值吻合良好。我们的结果确立了可学习性作为相变的数据驱动标志，并强调了凝聚物质中的长程有序与现代语言模型中结构的涌现之间的深刻相似性。 

---
# Local MAP Sampling for Diffusion Models 

**Title (ZH)**: 局部MAP采样用于扩散模型 

**Authors**: Shaorong Zhang, Rob Brekelmans, Greg Ver Steeg  

**Link**: [PDF](https://arxiv.org/pdf/2510.07343)  

**Abstract**: Diffusion Posterior Sampling (DPS) provides a principled Bayesian approach to inverse problems by sampling from $p(x_0 \mid y)$. However, in practice, the goal of inverse problem solving is not to cover the posterior but to recover the most accurate reconstruction, where optimization-based diffusion solvers often excel despite lacking a clear probabilistic foundation. We introduce Local MAP Sampling (LMAPS), a new inference framework that iteratively solving local MAP subproblems along the diffusion trajectory. This perspective clarifies their connection to global MAP estimation and DPS, offering a unified probabilistic interpretation for optimization-based methods. Building on this foundation, we develop practical algorithms with a probabilistically interpretable covariance approximation, a reformulated objective for stability and interpretability, and a gradient approximation for non-differentiable operators. Across a broad set of image restoration and scientific tasks, LMAPS achieves state-of-the-art performance, including $\geq 2$ dB gains on motion deblurring, JPEG restoration, and quantization, and $>1.5$ dB improvements on inverse scattering benchmarks. 

**Abstract (ZH)**: 局部最大后验采样（LMAPS）提供了一种新的推理框架，通过沿扩散轨迹迭代求解局部MAP子问题，从而澄清了这些方法与全局MAP估计和DPS之间的联系，为基于优化的方法提供了一致的概率解释。 

---
# Deep Learning Based Approach to Enhanced Recognition of Emotions and Behavioral Patterns of Autistic Children 

**Title (ZH)**: 基于深度学习的方法以增强自闭症儿童情绪和行为模式识别 

**Authors**: Nelaka K.A.R, Peiris M.K.V, Liyanage R.P.B  

**Link**: [PDF](https://arxiv.org/pdf/2510.07320)  

**Abstract**: Autism Spectrum Disorder significantly influences the communication abilities, learning processes, behavior, and social interactions of individuals. Although early intervention and customized educational strategies are critical to improving outcomes, there is a pivotal gap in understanding and addressing nuanced behavioral patterns and emotional identification in autistic children prior to skill development. This extended research delves into the foundational step of recognizing and mapping these patterns as a prerequisite to improving learning and soft skills. Using a longitudinal approach to monitor emotions and behaviors, this study aims to establish a baseline understanding of the unique needs and challenges faced by autistic students, particularly in the Information Technology domain, where opportunities are markedly limited. Through a detailed analysis of behavioral trends over time, we propose a targeted framework for developing applications and technical aids designed to meet these identified needs. Our research underscores the importance of a sequential and evidence-based intervention approach that prioritizes a deep understanding of each child's behavioral and emotional landscape as the basis for effective skill development. By shifting the focus toward early identification of behavioral patterns, we aim to foster a more inclusive and supportive learning environment that can significantly improve the educational and developmental trajectory of children with ASD. 

**Abstract (ZH)**: 自闭症谱系障碍显著影响个体的沟通能力、学习过程、行为和社会互动。尽管早期干预和定制化教育策略对改善结果至关重要，但在技能发展之前理解和解决自闭症儿童细微的行为模式和情绪识别方面仍存在关键缺口。本扩展研究致力于通过识别和绘制这些模式，为改善学习和软技能奠定基础。采用纵向方法监测情绪和行为，本研究旨在建立对自闭学生独特需求和挑战的基础理解，特别是在信息技术领域，机会明显受限。通过对时间序列的行为趋势进行详细分析，我们提出了一个针对这些识别需求开发应用和技术辅助的定向框架。研究表明，采用顺序性和基于证据的干预方法，优先了解每个孩子的行为和情感环境对于有效技能发展至关重要。通过转向早期识别行为模式，我们旨在培养一个更具包容性和支持性的学习环境，从而显著改善自闭症谱系障碍儿童的教育和发展轨迹。 

---
# DUA-D2C: Dynamic Uncertainty Aware Method for Overfitting Remediation in Deep Learning 

**Title (ZH)**: DUA-D2C：动态不确定性意识方法在深度学习中防治过拟合 

**Authors**: Md. Saiful Bari Siddiqui, Md Mohaiminul Islam, Md. Golam Rabiul Alam  

**Link**: [PDF](https://arxiv.org/pdf/2411.15876)  

**Abstract**: Overfitting remains a significant challenge in deep learning, often arising from data outliers, noise, and limited training data. To address this, the Divide2Conquer (D2C) method was previously proposed, which partitions training data into multiple subsets and trains identical models independently on each. This strategy enables learning more consistent patterns while minimizing the influence of individual outliers and noise. However, D2C's standard aggregation typically treats all subset models equally or based on fixed heuristics (like data size), potentially underutilizing information about their varying generalization capabilities. Building upon this foundation, we introduce Dynamic Uncertainty-Aware Divide2Conquer (DUA-D2C), an advanced technique that refines the aggregation process. DUA-D2C dynamically weights the contributions of subset models based on their performance on a shared validation set, considering both accuracy and prediction uncertainty. This intelligent aggregation allows the central model to preferentially learn from subsets yielding more generalizable and confident edge models, thereby more effectively combating overfitting. Empirical evaluations on benchmark datasets spanning multiple domains demonstrate that DUA-D2C significantly improves generalization. Our analysis includes evaluations of decision boundaries, loss curves, and other performance metrics, highlighting the effectiveness of DUA-D2C. This study demonstrates that DUA-D2C improves generalization performance even when applied on top of other regularization methods, establishing it as a theoretically grounded and effective approach to combating overfitting in modern deep learning. Our codes are publicly available at: this https URL. 

**Abstract (ZH)**: Dynamic Uncertainty-Aware Divide2Conquer for Improving Generalization in Deep Learning 

---
