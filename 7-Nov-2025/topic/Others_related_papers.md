# Temporal Action Selection for Action Chunking 

**Title (ZH)**: 时间动作选择在动作切片中的应用 

**Authors**: Yueyang Weng, Xiaopeng Zhang, Yongjin Mu, Yingcong Zhu, Yanjie Li, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04421)  

**Abstract**: Action chunking is a widely adopted approach in Learning from Demonstration (LfD). By modeling multi-step action chunks rather than single-step actions, action chunking significantly enhances modeling capabilities for human expert policies. However, the reduced decision frequency restricts the utilization of recent observations, degrading reactivity - particularly evident in the inadequate adaptation to sensor noise and dynamic environmental changes. Existing efforts to address this issue have primarily resorted to trading off reactivity against decision consistency, without achieving both. To address this limitation, we propose a novel algorithm, Temporal Action Selector (TAS), which caches predicted action chunks from multiple timesteps and dynamically selects the optimal action through a lightweight selector network. TAS achieves balanced optimization across three critical dimensions: reactivity, decision consistency, and motion coherence. Experiments across multiple tasks with diverse base policies show that TAS significantly improves success rates - yielding an absolute gain of up to 73.3%. Furthermore, integrating TAS as a base policy with residual reinforcement learning (RL) substantially enhances training efficiency and elevates the performance plateau. Experiments in both simulation and physical robots confirm the method's efficacy. 

**Abstract (ZH)**: 基于时间的行动选择器（TAS）：一种平衡反应性、决策一致性与运动连贯性的新算法 

---
# Studying the Effect of Explicit Interaction Representations on Learning Scene-level Distributions of Human Trajectories 

**Title (ZH)**: 研究显式交互表示对学习人类轨迹场景级分布的影响 

**Authors**: Anna Mészáros, Javier Alonso-Mora, Jens Kober  

**Link**: [PDF](https://arxiv.org/pdf/2511.04375)  

**Abstract**: Effectively capturing the joint distribution of all agents in a scene is relevant for predicting the true evolution of the scene and in turn providing more accurate information to the decision processes of autonomous vehicles. While new models have been developed for this purpose in recent years, it remains unclear how to best represent the joint distributions particularly from the perspective of the interactions between agents. Thus far there is no clear consensus on how best to represent interactions between agents; whether they should be learned implicitly from data by neural networks, or explicitly modeled using the spatial and temporal relations that are more grounded in human decision-making. This paper aims to study various means of describing interactions within the same network structure and their effect on the final learned joint distributions. Our findings show that more often than not, simply allowing a network to establish interactive connections between agents based on data has a detrimental effect on performance. Instead, having well defined interactions (such as which agent of an agent pair passes first at an intersection) can often bring about a clear boost in performance. 

**Abstract (ZH)**: 有效捕获场景中所有代理的联合分布对于预测场景的真实演化并进而为自主车辆的决策过程提供更准确的信息是相关的。虽然近年来为此目的开发了新模型，但从代理间交互的角度来看，如何最好地表示联合分布仍不清楚。目前尚无明确共识，代理间的交互是应该通过神经网络隐式从数据中学习，还是通过更基于人类决策制定的空间和时间关系显式建模。本文旨在研究在相同网络结构中描述交互的各种方法及其对最终学习到的联合分布的影响。我们的研究发现，通常情况下，仅仅允许网络基于数据建立代理间的交互连接会对性能产生负面影响。相反，具有良好定义的交互（例如，过交叉口时哪一只代理先通过）往往能显著提升性能。 

---
# GraSP-VLA: Graph-based Symbolic Action Representation for Long-Horizon Planning with VLA Policies 

**Title (ZH)**: 基于图的符号动作表示：具有VLA策略的长时规划 

**Authors**: Maëlic Neau, Zoe Falomir, Paulo E. Santos, Anne-Gwenn Bosser, Cédric Buche  

**Link**: [PDF](https://arxiv.org/pdf/2511.04357)  

**Abstract**: Deploying autonomous robots that can learn new skills from demonstrations is an important challenge of modern robotics. Existing solutions often apply end-to-end imitation learning with Vision-Language Action (VLA) models or symbolic approaches with Action Model Learning (AML). On the one hand, current VLA models are limited by the lack of high-level symbolic planning, which hinders their abilities in long-horizon tasks. On the other hand, symbolic approaches in AML lack generalization and scalability perspectives. In this paper we present a new neuro-symbolic approach, GraSP-VLA, a framework that uses a Continuous Scene Graph representation to generate a symbolic representation of human demonstrations. This representation is used to generate new planning domains during inference and serves as an orchestrator for low-level VLA policies, scaling up the number of actions that can be reproduced in a row. Our results show that GraSP-VLA is effective for modeling symbolic representations on the task of automatic planning domain generation from observations. In addition, results on real-world experiments show the potential of our Continuous Scene Graph representation to orchestrate low-level VLA policies in long-horizon tasks. 

**Abstract (ZH)**: 基于连续场景图的神经符号方法GraSP-VLA：用于自动规划领域生成的任务中符号表示建模 

---
# Enhancing Fault-Tolerant Space Computing: Guidance Navigation and Control (GNC) and Landing Vision System (LVS) Implementations on Next-Gen Multi-Core Processors 

**Title (ZH)**: 增强容错太空计算：下一代多核处理器上指导导航控制（GNC）和着陆视觉系统（LVS）的实现 

**Authors**: Kyongsik Yun, David Bayard, Gerik Kubiak, Austin Owens, Andrew Johnson, Ryan Johnson, Dan Scharf, Thomas Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04052)  

**Abstract**: Future planetary exploration missions demand high-performance, fault-tolerant computing to enable autonomous Guidance, Navigation, and Control (GNC) and Lander Vision System (LVS) operations during Entry, Descent, and Landing (EDL). This paper evaluates the deployment of GNC and LVS algorithms on next-generation multi-core processors--HPSC, Snapdragon VOXL2, and AMD Xilinx Versal--demonstrating up to 15x speedup for LVS image processing and over 250x speedup for Guidance for Fuel-Optimal Large Divert (GFOLD) trajectory optimization compared to legacy spaceflight hardware. To ensure computational reliability, we present ARBITER (Asynchronous Redundant Behavior Inspection for Trusted Execution and Recovery), a Multi-Core Voting (MV) mechanism that performs real-time fault detection and correction across redundant cores. ARBITER is validated in both static optimization tasks (GFOLD) and dynamic closed-loop control (Attitude Control System). A fault injection study further identifies the gradient computation stage in GFOLD as the most sensitive to bit-level errors, motivating selective protection strategies and vector-based output arbitration. This work establishes a scalable and energy-efficient architecture for future missions, including Mars Sample Return, Enceladus Orbilander, and Ceres Sample Return, where onboard autonomy, low latency, and fault resilience are critical. 

**Abstract (ZH)**: 未来行星探索任务需要高性能、容错计算能力，以实现进入、下降和着陆（EDL）期间自主姿态控制、导航与制导（GNC）和着陆视觉系统（LVS）的操作。本文评估了GNC和LVS算法在下一代多核处理器——HPSC、Snapdragon VOXL2和AMD Xilinx Versal上的部署，结果显示LVS图像处理速度提高了15倍，而燃料最优大偏航轨迹优化（GFOLD）指导的速度提高了超过250倍，相较于传统的太空飞行硬件。为了确保计算可靠性，我们提出了ARBITER（异步冗余行为检查以实现可信执行和恢复）机制，这是一种多核投票（MV）机制，能够在冗余核心之间进行实时故障检测与修正。ARBITER在静态优化任务（GFOLD）和动态闭环控制（姿态控制系统）中进行了验证。进一步的压力注入研究表明，GFOLD的梯度计算阶段对位错误最为敏感，从而推动了选择性保护策略和向量基输出仲裁的发展。本文为未来的火星采样返回、土卫二轨道探测器和谷神星采样返回任务建立了可扩展且能效高的架构，其中载上自治功能、低延迟和抗故障能力至关重要。 

---
# HACI: A Haptic-Audio Code Interface to Improve Educational Outcomes for Visually Impaired Introductory Programming Students 

**Title (ZH)**: HACI: 一种触觉-音频代码界面，以提高视障 introductory 编程学生教育成果 

**Authors**: Pratham Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2511.03733)  

**Abstract**: This thesis introduces the Haptic-Audio Code Interface (HACI), an educational tool designed to enhance programming education for visually impaired (VI) students by integrating haptic and audio feedback to compensate for the absence of visual cues. HACI consists of a non-resource-intensive web application supporting JavaScript program development, execution, and debugging, connected via a cable to an Arduino-powered glove with six integrated haptic motors to provide physical feedback to VI programmers. Motivated by the need to provide equitable educational opportunities in computer science, HACI aims to improve non-visual code navigation, comprehension, summarizing, editing, and debugging for students with visual impairments while minimizing cognitive load. This work details HACI's design principles, technical implementation, and a preliminary evaluation through a pilot study conducted with undergraduate Computer Science students. Findings indicate that HACI aids in the non-visual navigation and understanding of programming constructs, although challenges remain in refining feedback mechanisms to ensure consistency and reliability, as well as supplementing the current functionality with a more feature-reach and customizable accessible learning experience which will allow visually impaired students to fully utilize interleaved haptic and audio feedback. The study underscores the transformative potential of haptic and audio feedback in educational practices for the visually impaired, setting a foundation for future research and development in accessible programming education. This thesis contributes to the field of accessible technology by demonstrating how tactile and auditory feedback can be effectively integrated into educational tools, thereby broadening accessibility in STEM education. 

**Abstract (ZH)**: 用于视障学生的触觉-音频代码接口（HACI）：一种增强编程教育的教育工具 

---
# Question the Questions: Auditing Representation in Online Deliberative Processes 

**Title (ZH)**: 质疑问题：在线 deliberative 过程中表示审计 

**Authors**: Soham De, Lodewijk Gelauff, Ashish Goel, Smitha Milli, Ariel Procaccia, Alice Siu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04588)  

**Abstract**: A central feature of many deliberative processes, such as citizens' assemblies and deliberative polls, is the opportunity for participants to engage directly with experts. While participants are typically invited to propose questions for expert panels, only a limited number can be selected due to time constraints. This raises the challenge of how to choose a small set of questions that best represent the interests of all participants. We introduce an auditing framework for measuring the level of representation provided by a slate of questions, based on the social choice concept known as justified representation (JR). We present the first algorithms for auditing JR in the general utility setting, with our most efficient algorithm achieving a runtime of $O(mn\log n)$, where $n$ is the number of participants and $m$ is the number of proposed questions. We apply our auditing methods to historical deliberations, comparing the representativeness of (a) the actual questions posed to the expert panel (chosen by a moderator), (b) participants' questions chosen via integer linear programming, (c) summary questions generated by large language models (LLMs). Our results highlight both the promise and current limitations of LLMs in supporting deliberative processes. By integrating our methods into an online deliberation platform that has been used for over hundreds of deliberations across more than 50 countries, we make it easy for practitioners to audit and improve representation in future deliberations. 

**Abstract (ZH)**: 许多公民会议和辩论式民意测验等审议过程的核心特征是参与者直接与专家互动的机会。尽管参与者通常被邀请提出问题供专家小组解答，但由于时间限制，只能选择有限的问题。这提出了如何选择一组最能代表所有参与者利益的问题的挑战。我们介绍了一种基于社会选择概念“正当代表”（JR）的审计框架，用于衡量问题名单提供的代表水平。我们提出了在一般效用设置中审计JR的第一种算法，最高效算法的运行时间为$O(mn\log n)$，其中$n$为参与者数量，$m$为提出的问题数量。我们将审计方法应用于历史审议，比较了（a）由主持人选择的实际提出给专家小组的问题，（b）通过整数线性规划选择的参与者问题，（c）由大型语言模型生成的摘要问题的代表性。我们的结果突显了大型语言模型在支持审议过程中的潜力和当前局限性。通过将我们的方法集成到一个已用于超过几百场审议的在线审议平台中，我们使实践者能够在未来审议中轻松审计和改进代表水平。 

---
# Are We Asking the Right Questions? On Ambiguity in Natural Language Queries for Tabular Data Analysis 

**Title (ZH)**: 我们在问对问题了吗？关于表格数据查询中自然语言歧义性的问题 

**Authors**: Daniel Gomm, Cornelius Wolff, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2511.04584)  

**Abstract**: Natural language interfaces to tabular data must handle ambiguities inherent to queries. Instead of treating ambiguity as a deficiency, we reframe it as a feature of cooperative interaction, where the responsibility of query specification is shared among the user and the system. We develop a principled framework distinguishing cooperative queries, i.e., queries that yield a resolvable interpretation, from uncooperative queries that cannot be resolved. Applying the framework to evaluations for tabular question answering and analysis, we analyze the queries in 15 popular datasets, and observe an uncontrolled mixing of query types neither adequate for evaluating a system's execution accuracy nor for evaluating interpretation capabilities. Our framework and analysis of queries shifts the perspective from fixing ambiguity to embracing cooperation in resolving queries. This reflection enables more informed design and evaluation for natural language interfaces for tabular data, for which we outline implications and directions for future research. 

**Abstract (ZH)**: 自然语言与表格数据接口必须处理查询中原生存在的歧义。我们不将歧义视为缺陷，而是将其重新定义为用户与系统之间协同交互的特征，其中查询规格化的责任由用户和系统共同承担。我们开发了一个基于原则的框架，区分可解析的协同查询和无法解析的非协同查询。将该框架应用于表格问答和分析的评估中，我们分析了15个流行数据集中的查询，并观察到查询类型的不受控混合，既不足以评估系统执行准确性，也不足以评估解释能力。我们的框架和查询分析从纠正歧义转向在解决查询过程中拥抱协作。这种反思为自然语言与表格数据接口的设计和评估提供了更有力的指导，并概述了未来研究的方向。 

---
# Optimizing Sensor Placement in Urban Storm Sewers: A Data-Driven Sparse Sensing Approach 

**Title (ZH)**: 城市暴雨排水管道中传感器布设优化：一种数据驱动的稀疏传感方法 

**Authors**: Zihang Ding, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04556)  

**Abstract**: Urban surface water flooding, triggered by intense rainfall overwhelming drainage systems, is increasingly frequent and widespread. While flood prediction and monitoring in high spatial-temporal resolution are desired, practical constraints in time, budget, and technology hinder its full implementation. How to monitor urban drainage networks and predict flow conditions under constrained resource is a major challenge. This study presents a data-driven sparse sensing (DSS) framework, integrated with EPA-SWMM, to optimize sensor placement and reconstruct peak flowrates in a stormwater system, using the Woodland Avenue catchment in Duluth, Minnesota, as a case study. We utilized a SWMM model to generate a training dataset of peak flowrate profiles across the stormwater network. Furthermore, we applied DSS - leveraging singular value decomposition for dimensionality reduction and QR factorization for sensor allocation - to identify the optimal monitoring nodes based on the simulated training dataset. We then validated the representativeness of these identified monitoring nodes by comparing the DSS-reconstructed peak flowrate profiles with those obtained from SWMM. Three optimally placed sensors among 77 nodes achieved satisfactory reconstruction performance with Nash-Sutcliffe Efficiency (NSE) values of 0.92-0.95 (25th to 75th percentiles). In addition, the model showed good robustness to uncertainty in measurements. Its robustness to sensor failures is location-dependent and improves with the number of sensors deployed. The framework balances computational efficiency and physical interpretability, enabling high-accuracy flow reconstruction with minimal sensors. This DSS framework can be further integrated with predictive models to realize flood early warning and real-time control under limited sensing and monitoring resource. 

**Abstract (ZH)**: 基于稀疏感知的城市排水网络流条件监控与预测框架 

---
# Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption through Empirical and Theoretical Analysis 

**Title (ZH)**: 促进可持续网络代理：基于实证和理论分析的能效benchmarking与估算 

**Authors**: Lars Krupp, Daniel Geißler, Vishal Banwari, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2511.04481)  

**Abstract**: Web agents, like OpenAI's Operator and Google's Project Mariner, are powerful agentic systems pushing the boundaries of Large Language Models (LLM). They can autonomously interact with the internet at the user's behest, such as navigating websites, filling search masks, and comparing price lists. Though web agent research is thriving, induced sustainability issues remain largely unexplored. To highlight the urgency of this issue, we provide an initial exploration of the energy and $CO_2$ cost associated with web agents from both a theoretical -via estimation- and an empirical perspective -by benchmarking. Our results show how different philosophies in web agent creation can severely impact the associated expended energy, and that more energy consumed does not necessarily equate to better results. We highlight a lack of transparency regarding disclosing model parameters and processes used for some web agents as a limiting factor when estimating energy consumption. Our work contributes towards a change in thinking of how we evaluate web agents, advocating for dedicated metrics measuring energy consumption in benchmarks. 

**Abstract (ZH)**: 基于网络的智能代理，如OpenAI的Operator和Google的Project Mariner，是强大的代理系统，正推动大型语言模型（LLM）的边界。它们可以在用户授权下自主与互联网交互，如导航网站、填充搜索框和比较价格列表。尽管基于网络的智能代理研究进展迅速，但其诱发的可持续性问题仍未得到充分探讨。为了突显这一问题的紧迫性，我们从理论（通过估算）和实证（通过基准测试）角度初期探索了基于网络的智能代理的能耗与二氧化碳成本。研究结果显示，不同的基于网络的智能代理创建哲学对其能耗有严重影响，且更高的能耗并不一定意味着更好的结果。我们指出了在估算能耗时对某些基于网络的智能代理的模型参数和使用过程缺乏透明度造成的限制因素。我们的工作有助于改变评估基于网络的智能代理的方式，呼吁在基准测试中采用专门衡量能耗的指标。 

---
# Probing the Probes: Methods and Metrics for Concept Alignment 

**Title (ZH)**: 探查探针：概念对齐的方法与度量 

**Authors**: Jacob Lysnæs-Larsen, Marte Eggen, Inga Strümke  

**Link**: [PDF](https://arxiv.org/pdf/2511.04312)  

**Abstract**: In explainable AI, Concept Activation Vectors (CAVs) are typically obtained by training linear classifier probes to detect human-understandable concepts as directions in the activation space of deep neural networks. It is widely assumed that a high probe accuracy indicates a CAV faithfully representing its target concept. However, we show that the probe's classification accuracy alone is an unreliable measure of concept alignment, i.e., the degree to which a CAV captures the intended concept. In fact, we argue that probes are more likely to capture spurious correlations than they are to represent only the intended concept. As part of our analysis, we demonstrate that deliberately misaligned probes constructed to exploit spurious correlations, achieve an accuracy close to that of standard probes. To address this severe problem, we introduce a novel concept localization method based on spatial linear attribution, and provide a comprehensive comparison of it to existing feature visualization techniques for detecting and mitigating concept misalignment. We further propose three classes of metrics for quantitatively assessing concept alignment: hard accuracy, segmentation scores, and augmentation robustness. Our analysis shows that probes with translation invariance and spatial alignment consistently increase concept alignment. These findings highlight the need for alignment-based evaluation metrics rather than probe accuracy, and the importance of tailoring probes to both the model architecture and the nature of the target concept. 

**Abstract (ZH)**: 可解释AI中，概念激活向量（CAVs）通常通过训练线性分类探针来检测人类可理解的概念作为深 neural 网络激活空间的方向来获得。广泛假设探针的分类准确性表明CAV忠实于其目标概念。然而，我们证明了探针的分类准确性本身是概念对齐的不可靠度量，即CAV捕获预期概念的程度。实际上，我们认为探针更可能捕获虚假相关性，而不是仅代表预期概念。作为我们分析的一部分，我们展示了故意构建以利用虚假相关性的刻意对齐偏差探针，其准确性接近于标准探针。为解决这一严重问题，我们引入了一种基于空间线性归因的概念定位方法，并提供了它与现有特征可视化技术的全面比较，用于检测和减轻概念对齐偏差。我们进一步提出了三种类别的度量标准来定量评估概念对齐：硬准确度、分割评分和增强鲁棒性。我们的分析表明，具有平移不变性和空间对齐性的探针一致地增加概念对齐。这些发现突显了需要基于对齐的评估度量而非探针准确性，并强调了要根据模型架构和目标概念的性质来定制探针的重要性。 

---
# RLoop: An Self-Improving Framework for Reinforcement Learning with Iterative Policy Initialization 

**Title (ZH)**: RLoop：一种迭代策略初始化的自我提升强化学习框架 

**Authors**: Zeng Zhiyuan, Jiashuo Liu, Zhangyue Yin, Ge Zhang, Wenhao Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04285)  

**Abstract**: While Reinforcement Learning for Verifiable Rewards (RLVR) is powerful for training large reasoning models, its training dynamics harbor a critical challenge: RL overfitting, where models gain training rewards but lose generalization. Our analysis reveals this is driven by policy over-specialization and catastrophic forgetting of diverse solutions generated during training. Standard optimization discards this valuable inter-step policy diversity. To address this, we introduce RLoop, a self-improving framework built on iterative policy initialization. RLoop transforms the standard training process into a virtuous cycle: it first uses RL to explore the solution space from a given policy, then filters the successful trajectories to create an expert dataset. This dataset is used via Rejection-sampling Fine-Tuning (RFT) to refine the initial policy, creating a superior starting point for the next iteration. This loop of exploration and exploitation via iterative re-initialization effectively converts transient policy variations into robust performance gains. Our experiments show RLoop mitigates forgetting and substantially improves generalization, boosting average accuracy by 9% and pass@32 by over 15% compared to vanilla RL. 

**Abstract (ZH)**: 强化学习验证奖励的自我提升框架：RLoop 

---
# Opus: A Quantitative Framework for Workflow Evaluation 

**Title (ZH)**: Opus：工作流评估的定量框架 

**Authors**: Alan Seroul, Théo Fagnoni, Inès Adnani, Dana O. Mohamed, Phillip Kingston  

**Link**: [PDF](https://arxiv.org/pdf/2511.04220)  

**Abstract**: This paper introduces the Opus Workflow Evaluation Framework, a probabilistic-normative formulation for quantifying Workflow quality and efficiency. It integrates notions of correctness, reliability, and cost into a coherent mathematical model that enables direct comparison, scoring, and optimization of Workflows. The framework combines the Opus Workflow Reward, a probabilistic function estimating expected performance through success likelihood, resource usage, and output gain, with the Opus Workflow Normative Penalties, a set of measurable functions capturing structural and informational quality across Cohesion, Coupling, Observability, and Information Hygiene. It supports automated Workflow assessment, ranking, and optimization within modern automation systems such as Opus and can be integrated into Reinforcement Learning loops to guide Workflow discovery and refinement. In this paper, we introduce the Opus Workflow Reward model that formalizes Workflow success as a probabilistic expectation over costs and outcomes. We define measurable Opus Workflow Normative Penalties capturing structural, semantic, and signal-related properties of Workflows. Finally, we propose a unified optimization formulation for identifying and ranking optimal Workflows under joint Reward-Penalty trade-offs. 

**Abstract (ZH)**: Opus 工作流评估框架：一种用于量化工作流质量与效率的概率规范性建模 

---
# When Empowerment Disempowers 

**Title (ZH)**: 当赋权现象剥夺权力 

**Authors**: Claire Yang, Maya Cakmak, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2511.04177)  

**Abstract**: Empowerment, a measure of an agent's ability to control its environment, has been proposed as a universal goal-agnostic objective for motivating assistive behavior in AI agents. While multi-human settings like homes and hospitals are promising for AI assistance, prior work on empowerment-based assistance assumes that the agent assists one human in isolation. We introduce an open source multi-human gridworld test suite Disempower-Grid. Using Disempower-Grid, we empirically show that assistive RL agents optimizing for one human's empowerment can significantly reduce another human's environmental influence and rewards - a phenomenon we formalize as disempowerment. We characterize when disempowerment occurs in these environments and show that joint empowerment mitigates disempowerment at the cost of the user's reward. Our work reveals a broader challenge for the AI alignment community: goal-agnostic objectives that seem aligned in single-agent settings can become misaligned in multi-agent contexts. 

**Abstract (ZH)**: 基于赋能的多人类辅助行为：一个开放源代码的多人类网格世界测试套件及其影响 

---
# Testing the Testers: Human-Driven Quality Assessment of Voice AI Testing Platforms 

**Title (ZH)**: 测试测试者：语音AI测试平台的人工质量评估 

**Authors**: Miguel E. Andres, Vadim Fedorov, Rida Sadek, Enric Spagnolo-Arrizabalaga, Nadescha Trudel  

**Link**: [PDF](https://arxiv.org/pdf/2511.04133)  

**Abstract**: Voice AI agents are rapidly transitioning to production deployments, yet systematic methods for ensuring testing reliability remain underdeveloped. Organizations cannot objectively assess whether their testing approaches (internal tools or external platforms) actually work, creating a critical measurement gap as voice AI scales to billions of daily interactions.
We present the first systematic framework for evaluating voice AI testing quality through human-centered benchmarking. Our methodology addresses the fundamental dual challenge of testing platforms: generating realistic test conversations (simulation quality) and accurately evaluating agent responses (evaluation quality). The framework combines established psychometric techniques (pairwise comparisons yielding Elo ratings, bootstrap confidence intervals, and permutation tests) with rigorous statistical validation to provide reproducible metrics applicable to any testing approach.
To validate the framework and demonstrate its utility, we conducted comprehensive empirical evaluation of three leading commercial platforms focused on Voice AI Testing using 21,600 human judgments across 45 simulations and ground truth validation on 60 conversations. Results reveal statistically significant performance differences with the proposed framework, with the top-performing platform, Evalion, achieving 0.92 evaluation quality measured as f1-score versus 0.73 for others, and 0.61 simulation quality using a league based scoring system (including ties) vs 0.43 for other platforms.
This framework enables researchers and organizations to empirically validate the testing capabilities of any platform, providing essential measurement foundations for confident voice AI deployment at scale. Supporting materials are made available to facilitate reproducibility and adoption. 

**Abstract (ZH)**: 语音AI代理正快速过渡到生产部署阶段，但确保测试可靠性的系统方法仍处于起步阶段。组织无法客观评估其测试方法（内部工具或外部平台）是否有效，随着语音AI每日交互量增至数十亿次，这一关键测量缺口变得尤为重要。

我们提出了首个基于以人为本的标准框架，用于评估语音AI测试质量。该方法学解决了测试平台的基本双重挑战：生成现实对话（模拟质量）和准确评估代理响应（评估质量）。框架结合了成熟的心理测量技术（成对比较产生Elo评分、靴strapping置信区间以及置换检验），并通过严格的统计验证提供可重复的衡量指标，适用于任何测试方法。

为验证该框架并展示其实用性，我们使用21,600个人类判断，针对45次模拟进行了全面的经验性评估，并对60次对话进行了事实验证。结果表明，提出的框架显示出统计上显著的性能差异，表现最佳的平台Evalion在f1分数上的评估质量为0.92，而其他平台为0.73；在基于联赛的评分系统（包括平局）中的模拟质量为0.61，而其他平台为0.43。

该框架使研究人员和组织能够经验性地验证任何平台的测试能力，为其大规模语音AI部署提供必要的测量基础。支持材料已提供以促进可重复性和采用。 

---
# Extracting Causal Relations in Deep Knowledge Tracing 

**Title (ZH)**: 提取深度知识追踪中的因果关系 

**Authors**: Kevin Hong, Kia Karbasi, Gregory Pottie  

**Link**: [PDF](https://arxiv.org/pdf/2511.03948)  

**Abstract**: A longstanding goal in computational educational research is to develop explainable knowledge tracing (KT) models. Deep Knowledge Tracing (DKT), which leverages a Recurrent Neural Network (RNN) to predict student knowledge and performance on exercises, has been proposed as a major advancement over traditional KT methods. Several studies suggest that its performance gains stem from its ability to model bidirectional relationships between different knowledge components (KCs) within a course, enabling the inference of a student's understanding of one KC from their performance on others. In this paper, we challenge this prevailing explanation and demonstrate that DKT's strength lies in its implicit ability to model prerequisite relationships as a causal structure, rather than bidirectional relationships. By pruning exercise relation graphs into Directed Acyclic Graphs (DAGs) and training DKT on causal subsets of the Assistments dataset, we show that DKT's predictive capabilities align strongly with these causal structures. Furthermore, we propose an alternative method for extracting exercise relation DAGs using DKT's learned representations and provide empirical evidence supporting our claim. Our findings suggest that DKT's effectiveness is largely driven by its capacity to approximate causal dependencies between KCs rather than simple relational mappings. 

**Abstract (ZH)**: 长期以来，计算教育研究的一个目标是开发可解释的知识追踪（KT）模型。深度知识追踪（DKT），通过递归神经网络（RNN）预测学生在练习中的知识和表现，被认为是一种传统KT方法的重大进步。已有研究表明，其性能提升源于其能够建模课程内不同知识组件（KCs）之间的双向关系，从而能够从学生在其他知识组件上的表现推断其对该知识组件的理解。在本文中，我们挑战了这一普遍解释，并证明DKT的优势在于其隐含地能够将先修关系建模为因果结构，而非双向关系。通过对练习关系图进行修剪以形成有向无环图（DAG），并在助理机器数据集的因果子集上训练DKT，我们展示了DKT的预测能力与这些因果结构高度一致。此外，我们提出了一种使用DKT学习表示提取练习关系DAG的替代方法，并提供了支持我们观点的实证证据。我们的研究结果表明，DKT的有效性主要由其能够近似知识组件之间的因果依赖关系驱动，而非简单的关系映射。 

---
# Addressing divergent representations from causal interventions on neural networks 

**Title (ZH)**: 解决因果干预对神经网络的分歧表示问题 

**Authors**: Satchel Grant, Simon Jerome Han, Alexa Tartaglini, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2511.04638)  

**Abstract**: A common approach to mechanistic interpretability is to causally manipulate model representations via targeted interventions in order to understand what those representations encode. Here we ask whether such interventions create out-of-distribution (divergent) representations, and whether this raises concerns about how faithful their resulting explanations are to the target model in its natural state. First, we demonstrate empirically that common causal intervention techniques often do shift internal representations away from the natural distribution of the target model. Then, we provide a theoretical analysis of two classes of such divergences: `harmless' divergences that occur in the null-space of the weights and from covariance within behavioral decision boundaries, and `pernicious' divergences that activate hidden network pathways and cause dormant behavioral changes. Finally, in an effort to mitigate the pernicious cases, we modify the Counterfactual Latent (CL) loss from Grant (2025) that regularizes interventions to remain closer to the natural distributions, reducing the likelihood of harmful divergences while preserving the interpretive power of interventions. Together, these results highlight a path towards more reliable interpretability methods. 

**Abstract (ZH)**: 一种常见的机制可解释性方法是通过目标模型中的针对性干预来因果操控模型表示，以理解这些表示所编码的内容。本文探讨了这些干预是否会产生分布外（发散）的表示，以及这种现象是否会影响其对目标模型自然状态下的解释的忠实性。首先，我们通过实验证明，常见的因果干预技术往往会使内部表示偏离目标模型的自然分布。然后，我们提供了两类发散的理论分析：无害的发散发生在权重的零空间和行为决策边界内的协方差中，而有害的发散则激活了隐藏网络路径并导致潜在的行为变化。最后，为了缓解有害发散的情况，我们修改了Grant（2025）提出的Counterfactual Latent（CL）损失函数，使其干预保持更接近自然分布，从而减少有害发散的可能性，同时保留干预的解释力。这些结果共同指出了更可靠的可解释性方法的途径。 

---
# Integrating Temporal and Structural Context in Graph Transformers for Relational Deep Learning 

**Title (ZH)**: 集成图变压器中的时空和结构上下文以进行关系深度学习 

**Authors**: Divyansha Lachi, Mahmoud Mohammadi, Joe Meyer, Vinam Arora, Tom Palczewski, Eva L. Dyer  

**Link**: [PDF](https://arxiv.org/pdf/2511.04557)  

**Abstract**: In domains such as healthcare, finance, and e-commerce, the temporal dynamics of relational data emerge from complex interactions-such as those between patients and providers, or users and products across diverse categories. To be broadly useful, models operating on these data must integrate long-range spatial and temporal dependencies across diverse types of entities, while also supporting multiple predictive tasks. However, existing graph models for relational data primarily focus on spatial structure, treating temporal information merely as a filtering constraint to exclude future events rather than a modeling signal, and are typically designed for single-task prediction. To address these gaps, we introduce a temporal subgraph sampler that enhances global context by retrieving nodes beyond the immediate neighborhood to capture temporally relevant relationships. In addition, we propose the Relational Graph Perceiver (RGP), a graph transformer architecture for relational deep learning that leverages a cross-attention-based latent bottleneck to efficiently integrate information from both structural and temporal contexts. This latent bottleneck integrates signals from different node and edge types into a common latent space, enabling the model to build global context across the entire relational system. RGP also incorporates a flexible cross-attention decoder that supports joint learning across tasks with disjoint label spaces within a single model. Experiments on RelBench, SALT, and CTU show that RGP delivers state-of-the-art performance, offering a general and scalable solution for relational deep learning with support for diverse predictive tasks. 

**Abstract (ZH)**: 在医疗保健、金融和电子商务等领域，关系数据的时间动态源自复杂交互，如患者与提供者之间或跨不同类别用户与产品的交互。为了广泛适用，这些数据上的模型必须整合不同实体类型之间的长程空间和时间依赖关系，同时支持多种预测任务。然而，现有关系数据图形模型主要关注空间结构，将时间信息视为排除未来事件的过滤约束，而非建模信号，并且通常针对单一任务预测设计。为弥补这些不足，我们引入了一种时间子图采样器，通过检索超出即时邻域的节点来增强全局上下文，以捕捉时间相关的关联。此外，我们提出了关系图形感知机（RGP），这是一种利用交叉注意力机制的潜在瓶颈来高效整合结构和时间上下文信息的图形转换器架构。这一潜在瓶颈将不同节点和边类型的信息整合到一个共同的潜在空间，使模型能够在整个关系系统中构建全局上下文。RGP还集成了灵活的交叉注意力解码器，支持在具有不相交标签空间的多个任务中进行联合学习。实验结果表明，RGP在RelBench、SALT和CTU上表现出色，提供了一种支持多类型预测任务的通用和可扩展的关系深度学习解决方案。 

---
# Alternative Fairness and Accuracy Optimization in Criminal Justice 

**Title (ZH)**: 刑事司法中的替代公平与准确性的优化 

**Authors**: Shaolong Wu, James Blume, Geshi Yeung  

**Link**: [PDF](https://arxiv.org/pdf/2511.04505)  

**Abstract**: Algorithmic fairness has grown rapidly as a research area, yet key concepts remain unsettled, especially in criminal justice. We review group, individual, and process fairness and map the conditions under which they conflict. We then develop a simple modification to standard group fairness. Rather than exact parity across protected groups, we minimize a weighted error loss while keeping differences in false negative rates within a small tolerance. This makes solutions easier to find, can raise predictive accuracy, and surfaces the ethical choice of error costs. We situate this proposal within three classes of critique: biased and incomplete data, latent affirmative action, and the explosion of subgroup constraints. Finally, we offer a practical framework for deployment in public decision systems built on three pillars: need-based decisions, Transparency and accountability, and narrowly tailored definitions and solutions. Together, these elements link technical design to legitimacy and provide actionable guidance for agencies that use risk assessment and related tools. 

**Abstract (ZH)**: 算法公平研究迅速发展，但刑事司法领域的核心概念仍存争议。我们回顾群体公平、个体公平和过程公平，并分析它们之间的冲突条件。随后，我们提出了一种标准群体公平的简单修改方法。我们不是追求保护群体之间的精确平等，而是最小化加权误差损失，同时保持假阴性率差异在较小的容许范围内。这使得解决方案更容易找到，可以提高预测准确性，并揭示错误成本的伦理选择。我们将此提议置于三种批评类别的框架内：有偏和不完整数据、隐含的积极行动以及子群体约束的爆炸性增长。最后，我们提出了一种实用框架，用于基于三大支柱在公共决策系统中的部署：基于需求的决策、透明度与问责制、以及精确定制的定义与解决方案。这些元素将技术设计与合法性相连，并为使用风险评估及相关工具的机构提供可操作的指导。 

---
# Q3R: Quadratic Reweighted Rank Regularizer for Effective Low-Rank Training 

**Title (ZH)**: Q3R：二次重新加权秩正则化符有效低秩训练 

**Authors**: Ipsita Ghosh, Ethan Nguyen, Christian Kümmerle  

**Link**: [PDF](https://arxiv.org/pdf/2511.04485)  

**Abstract**: Parameter-efficient training, based on low-rank optimization, has become a highly successful tool for fine-tuning large deep-learning models. However, these methods fail at low-rank pre-training tasks where maintaining the low-rank structure and the objective remains a challenging task. We propose the Quadratic Reweighted Rank Regularizer dubbed Q3R, which leads to a novel low-rank inducing training strategy inspired by the iteratively reweighted least squares (IRLS) framework. Q3R is based on a quadratic regularizer term which majorizes a smoothed log determinant serving as rank surrogate objective. Unlike other low-rank training techniques, Q3R is able to train weight matrices with prescribed, low target ranks of models that achieve comparable predictive performance as dense models, with small computational overhead, while remaining fully compatible with existing architectures. For example, we demonstrated one experiment where we are able to truncate $60\%$ and $80\%$ of the parameters of a ViT-Tiny model with $~1.3\%$ and $~4\%$ accuracy drop in CIFAR-10 performance respectively. The efficacy of Q3R is confirmed on Transformers across both image and language tasks, including for low-rank fine-tuning. 

**Abstract (ZH)**: 基于二次重权秩正则化的低秩诱导训练策略：Q3R方法及其在Transformer模型中的应用 

---
# Fraud-Proof Revenue Division on Subscription Platforms 

**Title (ZH)**: 骗保防范收入分配：订阅平台上的公平分摊 

**Authors**: Abheek Ghosh, Tzeh Yuan Neoh, Nicholas Teh, Giannis Tyrovolas  

**Link**: [PDF](https://arxiv.org/pdf/2511.04465)  

**Abstract**: We study a model of subscription-based platforms where users pay a fixed fee for unlimited access to content, and creators receive a share of the revenue. Existing approaches to detecting fraud predominantly rely on machine learning methods, engaging in an ongoing arms race with bad actors. We explore revenue division mechanisms that inherently disincentivize manipulation. We formalize three types of manipulation-resistance axioms and examine which existing rules satisfy these. We show that a mechanism widely used by streaming platforms, not only fails to prevent fraud, but also makes detecting manipulation computationally intractable. We also introduce a novel rule, ScaledUserProp, that satisfies all three manipulation-resistance axioms. Finally, experiments with both real-world and synthetic streaming data support ScaledUserProp as a fairer alternative compared to existing rules. 

**Abstract (ZH)**: 我们研究一种基于订阅的平台模型，用户支付固定费用以无限访问内容，创作者获得部分内容收益。现有的欺诈检测方法主要依赖机器学习方法，与不良行为者进行一种持续的 arms race。我们探索内置抑制操纵机制的收益分配机制。我们形式化了三种类型的抗操纵公理，并检查现有规则是否满足这些公理。我们证明一种广泛用于流媒体平台的机制不仅未能防止欺诈，反而使检测操纵变得计算上不可行。我们还引入了一种新的规则 ScaledUserProp，该规则满足所有三种抗操纵公理。最后，实验结果支持 ScaledUserProp 相比现有规则是一种更为公平的替代方案。 

---
# Deep Dictionary-Free Method for Identifying Linear Model of Nonlinear System with Input Delay 

**Title (ZH)**: 深层字典自由方法用于具有输入延迟的非线性系统线性模型识别 

**Authors**: Patrik Valábek, Marek Wadinger, Michal Kvasnica, Martin Klaučo  

**Link**: [PDF](https://arxiv.org/pdf/2511.04451)  

**Abstract**: Nonlinear dynamical systems with input delays pose significant challenges for prediction, estimation, and control due to their inherent complexity and the impact of delays on system behavior. Traditional linear control techniques often fail in these contexts, necessitating innovative approaches. This paper introduces a novel approach to approximate the Koopman operator using an LSTM-enhanced Deep Koopman model, enabling linear representations of nonlinear systems with time delays. By incorporating Long Short-Term Memory (LSTM) layers, the proposed framework captures historical dependencies and efficiently encodes time-delayed system dynamics into a latent space. Unlike traditional extended Dynamic Mode Decomposition (eDMD) approaches that rely on predefined dictionaries, the LSTM-enhanced Deep Koopman model is dictionary-free, which mitigates the problems with the underlying dynamics being known and incorporated into the dictionary. Quantitative comparisons with extended eDMD on a simulated system demonstrate highly significant performance gains in prediction accuracy in cases where the true nonlinear dynamics are unknown and achieve comparable results to eDMD with known dynamics of a system. 

**Abstract (ZH)**: 具有输入延迟的非线性动力学系统由于其固有复杂性和延迟对系统行为的影响，预测、估测和控制面临着重大挑战。传统的线性控制技术在这些情况下通常会失效，需要创新的方法。本文提出了一种使用LSTM增强的Deep Koopman模型来近似Koopman算子的方法，从而能够对具有时间延迟的非线性系统进行线性表示。通过引入LSTM层，所提出框架捕捉历史依赖性并有效地将带时间延迟的系统动力学编码到潜在空间中。与依赖预定义字典的传统扩展动态模态分解(eDMD)方法不同，LSTM增强的Deep Koopman模型是无字典的，这减轻了系统潜在动力学已知并被纳入字典的问题。在模拟系统上的定量比较表明，在真非线性动力学未知的情况下，预测精度有了显著的提升，并且在系统已知动力学的情况下，获得了与eDMD相当的结果。 

---
# On the Equivalence of Regression and Classification 

**Title (ZH)**: 回归与分类的等价性 

**Authors**: Jayadeva, Naman Dwivedi, Hari Krishnan, N.M. Anoop Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.04422)  

**Abstract**: A formal link between regression and classification has been tenuous. Even though the margin maximization term $\|w\|$ is used in support vector regression, it has at best been justified as a regularizer. We show that a regression problem with $M$ samples lying on a hyperplane has a one-to-one equivalence with a linearly separable classification task with $2M$ samples. We show that margin maximization on the equivalent classification task leads to a different regression formulation than traditionally used. Using the equivalence, we demonstrate a ``regressability'' measure, that can be used to estimate the difficulty of regressing a dataset, without needing to first learn a model for it. We use the equivalence to train neural networks to learn a linearizing map, that transforms input variables into a space where a linear regressor is adequate. 

**Abstract (ZH)**: 一种回归和分类之间的正式联系尚不稳固。尽管支持向量回归中使用了边际最大化项$\|w\|$，但最多只能将其作为正则化项进行解释。我们展示了具有M个样本在线性超平面上的回归问题，等价于一个具有2M个样本的线性可分分类任务。我们证明了在等价分类任务中进行边际最大化会导致不同于传统使用的回归公式。利用这种等价性，我们展示了“可回归性”度量，可以用来估计回归数据集的难度，而无需首先学习一个模型。我们利用这种等价性训练神经网络学习一个线性化映射，将输入变量转换到一个线性回归器适用的空间。 

---
# Spurious Correlation-Aware Embedding Regularization for Worst-Group Robustness 

**Title (ZH)**: aware Embedding Regularization for Worst-Group Robustness 

**Authors**: Subeen Park, Joowang Kim, Hakyung Lee, Sunjae Yoo, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.04401)  

**Abstract**: Deep learning models achieve strong performance across various domains but often rely on spurious correlations, making them vulnerable to distribution shifts. This issue is particularly severe in subpopulation shift scenarios, where models struggle in underrepresented groups. While existing methods have made progress in mitigating this issue, their performance gains are still constrained. They lack a rigorous theoretical framework connecting the embedding space representations with worst-group error. To address this limitation, we propose Spurious Correlation-Aware Embedding Regularization for Worst-Group Robustness (SCER), a novel approach that directly regularizes feature representations to suppress spurious cues. We show theoretically that worst-group error is influenced by how strongly the classifier relies on spurious versus core directions, identified from differences in group-wise mean embeddings across domains and classes. By imposing theoretical constraints at the embedding level, SCER encourages models to focus on core features while reducing sensitivity to spurious patterns. Through systematic evaluation on multiple vision and language, we show that SCER outperforms prior state-of-the-art studies in worst-group accuracy. Our code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 深度学习模型在各种领域中表现出强大的性能，但往往依赖于虚假相关性，使其在分布变化时变得脆弱。这一问题在次人口转移场景中尤为严重，模型在少数群体中表现挣扎。尽管现有方法在缓解这一问题方面已经取得了进展，但其性能提升仍然受到限制。它们缺乏将嵌入空间表示与最差群体错误联系起来的严格理论框架。为解决这一局限性，我们提出了一种新的方法——感知虚假相关性的嵌入正则化以增强最差群体鲁棒性（SCER），该方法直接通过对特征表示施加正则化来抑制虚假线索。我们理论上证明，最差群体错误受分类器依赖虚假方向相对于核心方向的强度影响，这些方向是从不同群体间均值嵌入的跨域和跨类差异中识别出来的。通过在嵌入级别施加理论约束，SCER促使模型专注于核心特征并减少对虚假模式的敏感性。通过在多个视觉和语言领域的系统评估，我们展示了SCER在最差群体准确率方面优于先前的最优研究结果。我们的代码可在\href{this https URL}{this https URL}获取。 

---
# LUME-DBN: Full Bayesian Learning of DBNs from Incomplete data in Intensive Care 

**Title (ZH)**: LUME-DBN: 全贝叶斯学习不完整ICU数据的DBN 

**Authors**: Federico Pirola, Fabio Stella, Marco Grzegorczyk  

**Link**: [PDF](https://arxiv.org/pdf/2511.04333)  

**Abstract**: Dynamic Bayesian networks (DBNs) are increasingly used in healthcare due to their ability to model complex temporal relationships in patient data while maintaining interpretability, an essential feature for clinical decision-making. However, existing approaches to handling missing data in longitudinal clinical datasets are largely derived from static Bayesian networks literature, failing to properly account for the temporal nature of the data. This gap limits the ability to quantify uncertainty over time, which is particularly critical in settings such as intensive care, where understanding the temporal dynamics is fundamental for model trustworthiness and applicability across diverse patient groups. Despite the potential of DBNs, a full Bayesian framework that integrates missing data handling remains underdeveloped. In this work, we propose a novel Gibbs sampling-based method for learning DBNs from incomplete data. Our method treats each missing value as an unknown parameter following a Gaussian distribution. At each iteration, the unobserved values are sampled from their full conditional distributions, allowing for principled imputation and uncertainty estimation. We evaluate our method on both simulated datasets and real-world intensive care data from critically ill patients. Compared to standard model-agnostic techniques such as MICE, our Bayesian approach demonstrates superior reconstruction accuracy and convergence properties. These results highlight the clinical relevance of incorporating full Bayesian inference in temporal models, providing more reliable imputations and offering deeper insight into model behavior. Our approach supports safer and more informed clinical decision-making, particularly in settings where missing data are frequent and potentially impactful. 

**Abstract (ZH)**: 动态贝叶斯网络在处理 longitudinal 临床数据中的缺失值方面的新方法：基于吉布斯采样的学习框架 

---
# Differentially Private In-Context Learning with Nearest Neighbor Search 

**Title (ZH)**: 差异隐私条件学习中的最近邻搜索 

**Authors**: Antti Koskela, Tejas Kulkarni, Laith Zumot  

**Link**: [PDF](https://arxiv.org/pdf/2511.04332)  

**Abstract**: Differentially private in-context learning (DP-ICL) has recently become an active research topic due to the inherent privacy risks of in-context learning. However, existing approaches overlook a critical component of modern large language model (LLM) pipelines: the similarity search used to retrieve relevant context data. In this work, we introduce a DP framework for in-context learning that integrates nearest neighbor search of relevant examples in a privacy-aware manner. Our method outperforms existing baselines by a substantial margin across all evaluated benchmarks, achieving more favorable privacy-utility trade-offs. To achieve this, we employ nearest neighbor retrieval from a database of context data, combined with a privacy filter that tracks the cumulative privacy cost of selected samples to ensure adherence to a central differential privacy budget. Experimental results on text classification and document question answering show a clear advantage of the proposed method over existing baselines. 

**Abstract (ZH)**: 差分隐私上下文学习（DP-ICL）由于内在的隐私风险已成为一个活跃的研究课题。然而，现有的方法忽视了现代大型语言模型（LLM）管道中的一个关键组件：用于检索相关上下文数据的相似性搜索。在本文中，我们引入了一种差分隐私框架，以隐私意识的方式集成相关的示例的最近邻搜索。我们的方法在所有评估基准上的表现显著优于现有的基线，实现了更为有利的隐私-效用 trade-offs。为了实现这一点，我们在上下文数据数据库中进行最近邻检索，并结合一个隐私过滤器来跟踪所选样本的累积隐私成本，以确保符合中心差分隐私预算。实验结果表明，在文本分类和文档问答任务中，所提出的方法明显优于现有基线。 

---
# AIM: Software and Hardware Co-design for Architecture-level IR-drop Mitigation in High-performance PIM 

**Title (ZH)**: 目标：高性能PIM架构级IR-drop共设软硬件设计方法研究 

**Authors**: Yuanpeng Zhang, Xing Hu, Xi Chen, Zhihang Yuan, Cong Li, Jingchen Zhu, Zhao Wang, Chenguang Zhang, Xin Si, Wei Gao, Qiang Wu, Runsheng Wang, Guangyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.04321)  

**Abstract**: SRAM Processing-in-Memory (PIM) has emerged as the most promising implementation for high-performance PIM, delivering superior computing density, energy efficiency, and computational precision. However, the pursuit of higher performance necessitates more complex circuit designs and increased operating frequencies, which exacerbate IR-drop issues. Severe IR-drop can significantly degrade chip performance and even threaten reliability. Conventional circuit-level IR-drop mitigation methods, such as back-end optimizations, are resource-intensive and often compromise power, performance, and area (PPA). To address these challenges, we propose AIM, comprehensive software and hardware co-design for architecture-level IR-drop mitigation in high-performance PIM. Initially, leveraging the bit-serial and in-situ dataflow processing properties of PIM, we introduce Rtog and HR, which establish a direct correlation between PIM workloads and IR-drop. Building on this foundation, we propose LHR and WDS, enabling extensive exploration of architecture-level IR-drop mitigation while maintaining computational accuracy through software optimization. Subsequently, we develop IR-Booster, a dynamic adjustment mechanism that integrates software-level HR information with hardware-based IR-drop monitoring to adapt the V-f pairs of the PIM macro, achieving enhanced energy efficiency and performance. Finally, we propose the HR-aware task mapping method, bridging software and hardware designs to achieve optimal improvement. Post-layout simulation results on a 7nm 256-TOPS PIM chip demonstrate that AIM achieves up to 69.2% IR-drop mitigation, resulting in 2.29x energy efficiency improvement and 1.152x speedup. 

**Abstract (ZH)**: SRAM Processing-in-Memory (PIM) 架构级IR-drop缓解的全面软件硬件协同设计 

---
# On the Brittleness of CLIP Text Encoders 

**Title (ZH)**: CLIP文本编码器的脆弱性研究 

**Authors**: Allie Tran, Luca Rossetto  

**Link**: [PDF](https://arxiv.org/pdf/2511.04247)  

**Abstract**: Multimodal co-embedding models, especially CLIP, have advanced the state of the art in zero-shot classification and multimedia information retrieval in recent years by aligning images and text in a shared representation space. However, such modals trained on a contrastive alignment can lack stability towards small input perturbations. Especially when dealing with manually expressed queries, minor variations in the query can cause large differences in the ranking of the best-matching results. In this paper, we present a systematic analysis of the effect of multiple classes of non-semantic query perturbations in an multimedia information retrieval scenario. We evaluate a diverse set of lexical, syntactic, and semantic perturbations across multiple CLIP variants using the TRECVID Ad-Hoc Video Search queries and the V3C1 video collection. Across models, we find that syntactic and semantic perturbations drive the largest instabilities, while brittleness is concentrated in trivial surface edits such as punctuation and case. Our results highlight robustness as a critical dimension for evaluating vision-language models beyond benchmark accuracy. 

**Abstract (ZH)**: 多模态共嵌模型，尤其是CLIP，通过在共享表示空间中对齐图像和文本，近年来在零-shot分类和多媒体信息检索方面取得了最先进的成果。然而，此类模型在对比对齐训练时可能对小输入扰动缺乏稳定性。特别是在处理手动表达的查询时，查询中的微小变化会导致最佳匹配结果排名的大幅差异。本文系统分析了多种非语义查询扰动在多媒体信息检索场景中的影响。我们使用TRECVID Ad-Hoc Video Search查询和V3C1视频集合，评估了多种词法、句法和语义扰动对多个CLIP变体的影响。我们发现，句法和语义扰动导致最大的不稳定性，而脆弱性集中在如标点符号和大小写这样的简单表面编辑上。本文的结果突显了稳健性是评估视觉语言模型时的一个关键维度，而不仅仅是基准准确性。 

---
# seqme: a Python library for evaluating biological sequence design 

**Title (ZH)**: seqme: 一个用于评估生物序ثر设计的Python库 

**Authors**: Rasmus Møller-Larsen, Adam Izdebski, Jan Olszewski, Pankhil Gawade, Michal Kmicikiewicz, Wojciech Zarzecki, Ewa Szczurek  

**Link**: [PDF](https://arxiv.org/pdf/2511.04239)  

**Abstract**: Recent advances in computational methods for designing biological sequences have sparked the development of metrics to evaluate these methods performance in terms of the fidelity of the designed sequences to a target distribution and their attainment of desired properties. However, a single software library implementing these metrics was lacking. In this work we introduce seqme, a modular and highly extendable open-source Python library, containing model-agnostic metrics for evaluating computational methods for biological sequence design. seqme considers three groups of metrics: sequence-based, embedding-based, and property-based, and is applicable to a wide range of biological sequences: small molecules, DNA, ncRNA, mRNA, peptides and proteins. The library offers a number of embedding and property models for biological sequences, as well as diagnostics and visualization functions to inspect the results. seqme can be used to evaluate both one-shot and iterative computational design methods. 

**Abstract (ZH)**: 近期在生物序列设计计算方法方面的进展推动了评估这些方法性能的度量标准的发展，特别是在序列 fidelity 和目标性质的实现方面。然而，缺乏单一的软件库来实现这些度量标准。本文介绍了 seqme，一个模块化、高度可扩展的开源 Python 库，包含适用于生物序列设计计算方法评估的模型无关度量。seqme 考虑了三组度量：序列基度量、嵌入基度量和性质基度量，并适用于多种生物序列：小分子、DNA、ncRNA、mRNA、肽和蛋白质。该库提供了多种生物序列的嵌入模型和性质模型，以及诊断和可视化函数以检查结果。seqme 可用于评估单次迭代和迭代的计算设计方法。 

---
# Denoised Recommendation Model with Collaborative Signal Decoupling 

**Title (ZH)**: 去噪推荐模型与协作信号解耦 

**Authors**: Zefeng Li, Ning Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04237)  

**Abstract**: Although the collaborative filtering (CF) algorithm has achieved remarkable performance in recommendation systems, it suffers from suboptimal recommendation performance due to noise in the user-item interaction matrix. Numerous noise-removal studies have improved recommendation models, but most existing approaches conduct denoising on a single graph. This may cause attenuation of collaborative signals: removing edges between two nodes can interrupt paths between other nodes, weakening path-dependent collaborative information. To address these limitations, this study proposes a novel GNN-based CF model called DRCSD for denoising unstable interactions. DRCSD includes two core modules: a collaborative signal decoupling module (decomposes signals into distinct orders by structural characteristics) and an order-wise denoising module (performs targeted denoising on each order). Additionally, the information aggregation mechanism of traditional GNN-based CF models is modified to avoid cross-order signal interference until the final pooling operation. Extensive experiments on three public real-world datasets show that DRCSD has superior robustness against unstable interactions and achieves statistically significant performance improvements in recommendation accuracy metrics compared to state-of-the-art baseline models. 

**Abstract (ZH)**: 虽然协同过滤（CF）算法在推荐系统中取得了显著的性能，但由于用户-项交互矩阵中存在的噪声导致推荐性能不佳。尽管许多去噪研究改进了推荐模型，但大多数现有方法仅在单个图上进行去噪，这可能会减弱协同信号：去除两个节点之间的边会打断其他节点之间的路径，削弱依赖路径的协同信息。为解决这些问题，本研究提出了一种基于GNN的新型去噪CF模型DRCSD（Decoupling and Order-wise Denoising for Unstable Collaborative Signals）。DRCSD包含两个核心模块：协同信号解耦模块（通过结构特征将信号分解为不同的顺序）和顺序层面去噪模块（对每个顺序执行针对性去噪）。此外，传统的基于GNN的CF模型的信息聚合机制被修改，以在最终池化操作之前避免跨顺序信号干扰。在三个公开的现实世界数据集上的广泛实验表明，DRCSD在面对不稳定交互方面具有更高的鲁棒性，并且在推荐准确度指标上相较于最先进的基线模型取得了统计学意义上的性能提升。 

---
# The Strong Lottery Ticket Hypothesis for Multi-Head Attention Mechanisms 

**Title (ZH)**: 多重注意力机制的强彩票票假说 

**Authors**: Hikari Otsuka, Daiki Chijiwa, Yasuyuki Okoshi, Daichi Fujiki, Susumu Takeuchi, Masato Motomura  

**Link**: [PDF](https://arxiv.org/pdf/2511.04217)  

**Abstract**: The strong lottery ticket hypothesis (SLTH) conjectures that high-performing subnetworks, called strong lottery tickets (SLTs), are hidden in randomly initialized neural networks. Although recent theoretical studies have established the SLTH across various neural architectures, the SLTH for transformer architectures still lacks theoretical understanding. In particular, the current theory of the SLTH does not yet account for the multi-head attention (MHA) mechanism, a core component of transformers. To address this gap, we introduce a theoretical analysis of the existence of SLTs within MHAs. We prove that, if a randomly initialized MHA of $H$ heads and input dimension $d$ has the hidden dimension $O(d\log(Hd^{3/2}))$ for the key and value, it contains an SLT that approximates an arbitrary MHA with the same input dimension with high probability. Furthermore, by leveraging this theory for MHAs, we extend the SLTH to transformers without normalization layers. We empirically validate our theoretical findings, demonstrating that the approximation error between the SLT within a source model (MHA and transformer) and an approximate target counterpart decreases exponentially by increasing the hidden dimension of the source model. 

**Abstract (ZH)**: 强彩票假设（SLTH）推测，在随机初始化的神经网络中隐藏着高性能子网络，称为强彩票（SLTs）。尽管最近的理论研究已在多种神经架构中建立了SLTH，但Transformer架构的SLTH仍缺乏理论理解。特别是，当前的SLTH理论尚未考虑到Transformer的核心组件——多头注意力机制（MHA）。为填补这一空白，我们对MHA中SLTs的存在性进行了理论分析。我们证明，若随机初始化的MHA包含H个头和输入维度d，则其键和值的隐藏维度为O(d log(Hd^(3/2)))时，该MHA以高概率包含一个可以近似任意相同输入维度MHA的SLTs。此外，我们利用这一理论将SLTH扩展到没有归一化层的Transformer。我们通过实证验证了我们的理论发现，证明了源模型（MHA和Transformer）中SLTs与近似目标模型之间的逼近误差随着源模型隐藏维度的增加而指数级减小。 

---
# AStF: Motion Style Transfer via Adaptive Statistics Fusor 

**Title (ZH)**: AStF: 适配统计融合的运动风格迁移 

**Authors**: Hanmo Chen, Chenghao Xu, Jiexi Yan, Cheng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2511.04192)  

**Abstract**: Human motion style transfer allows characters to appear less rigidity and more realism with specific style. Traditional arbitrary image style transfer typically process mean and variance which is proved effective. Meanwhile, similar methods have been adapted for motion style transfer. However, due to the fundamental differences between images and motion, relying on mean and variance is insufficient to fully capture the complex dynamic patterns and spatiotemporal coherence properties of motion data. Building upon this, our key insight is to bring two more coefficient, skewness and kurtosis, into the analysis of motion style. Specifically, we propose a novel Adaptive Statistics Fusor (AStF) which consists of Style Disentanglement Module (SDM) and High-Order Multi-Statistics Attention (HOS-Attn). We trained our AStF in conjunction with a Motion Consistency Regularization (MCR) discriminator. Experimental results show that, by providing a more comprehensive model of the spatiotemporal statistical patterns inherent in dynamic styles, our proposed AStF shows proficiency superiority in motion style transfers over state-of-the-arts. Our code and model are available at this https URL. 

**Abstract (ZH)**: 基于统计特性的动态风格转移：通过引入峰度和偏度实现更加真实的运动风格转换 

---
# A Reinforced Evolution-Based Approach to Multi-Resource Load Balancing 

**Title (ZH)**: 基于增强演化的方法多资源负载均衡 

**Authors**: Leszek Sliwko  

**Link**: [PDF](https://arxiv.org/pdf/2511.04183)  

**Abstract**: This paper presents a reinforced genetic approach to a defined d-resource system optimization problem. The classical evolution schema was ineffective due to a very strict feasibility function in the studied problem. Hence, the presented strategy has introduced several modifications and adaptations to standard genetic routines, e.g.: a migration operator which is an analogy to the biological random genetic drift. 

**Abstract (ZH)**: 本文提出了一种强化遗传算法来解决定义的d资源系统优化问题。由于研究问题中存在非常严格的可行性函数，经典的进化方案效果不佳。因此，提出的策略对标准遗传算法进行了多项修改和适应，例如：迁移操作符，其类似于生物随机遗传漂移。 

---
# Scaffolding Metacognition in Programming Education: Understanding Student-AI Interactions and Design Implications 

**Title (ZH)**: 编程教育中元认知的支架式教学：理解学生-人工智能交互及设计 implications 

**Authors**: Boxuan Ma, Huiyong Li, Gen Li, Li Chen, Cheng Tang, Yinjie Xie, Chenghao Gu, Atsushi Shimada, Shin'ichi Konomi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04144)  

**Abstract**: Generative AI tools such as ChatGPT now provide novice programmers with unprecedented access to instant, personalized support. While this holds clear promise, their influence on students' metacognitive processes remains underexplored. Existing work has largely focused on correctness and usability, with limited attention to whether and how students' use of AI assistants supports or bypasses key metacognitive processes. This study addresses that gap by analyzing student-AI interactions through a metacognitive lens in university-level programming courses. We examined more than 10,000 dialogue logs collected over three years, complemented by surveys of students and educators. Our analysis focused on how prompts and responses aligned with metacognitive phases and strategies. Synthesizing these findings across data sources, we distill design considerations for AI-powered coding assistants that aim to support rather than supplant metacognitive engagement. Our findings provide guidance for developing educational AI tools that strengthen students' learning processes in programming education. 

**Abstract (ZH)**: 生成式AI工具如ChatGPT现在为初学者程序员提供了前所未有的即时个性化支持。尽管这一进展充满潜力，但其对学生元认知过程的影响尚未得到充分探索。现有研究主要集中在正确性和可用性方面，对AI助手是支持还是绕过学生的关键元认知过程关注有限。本研究通过元认知视角分析大学级别编程课程中的学生-AI交互，考察了超过10,000条对话日志，并结合学生和教师的调查数据。我们分析了提示和响应与元认知阶段和策略的契合度。综合这些发现，我们提炼出旨在支持而非取代学生元认知参与的设计考虑。我们的研究结果为开发促进编程教育中学生学习过程的教育AI工具提供了指导。 

---
# Automated and Explainable Denial of Service Analysis for AI-Driven Intrusion Detection Systems 

**Title (ZH)**: 基于人工智能驱动的入侵检测系统的自动化可解释拒绝服务分析 

**Authors**: Paul Badu Yakubu, Lesther Santana, Mohamed Rahouti, Yufeng Xin, Abdellah Chehri, Mohammed Aledhari  

**Link**: [PDF](https://arxiv.org/pdf/2511.04114)  

**Abstract**: With the increasing frequency and sophistication of Distributed Denial of Service (DDoS) attacks, it has become critical to develop more efficient and interpretable detection methods. Traditional detection systems often struggle with scalability and transparency, hindering real-time response and understanding of attack vectors. This paper presents an automated framework for detecting and interpreting DDoS attacks using machine learning (ML). The proposed method leverages the Tree-based Pipeline Optimization Tool (TPOT) to automate the selection and optimization of ML models and features, reducing the need for manual experimentation. SHapley Additive exPlanations (SHAP) is incorporated to enhance model interpretability, providing detailed insights into the contribution of individual features to the detection process. By combining TPOT's automated pipeline selection with SHAP interpretability, this approach improves the accuracy and transparency of DDoS detection. Experimental results demonstrate that key features such as mean backward packet length and minimum forward packet header length are critical in detecting DDoS attacks, offering a scalable and explainable cybersecurity solution. 

**Abstract (ZH)**: 随着分布式拒绝服务（DDoS）攻击频率和 sophistication 的不断增加，开发更高效和可解释的检测方法变得日益重要。传统的检测系统 Often Struggle with Scalability and Transparency，阻碍了实时响应和对攻击向量的理解。本文提出了一种基于机器学习（ML）的自动化框架，用于检测和解释 DDoS 攻击。所提出的方法利用基于树的管道优化工具（TPOT）自动选择和优化 ML 模型和特征，减少手动实验的需要。将 SHapley Additive exPlanations（SHAP）纳入以增强模型的可解释性，并提供对各个特征对检测过程贡献的详细见解。通过结合 TPOT 的自动化管道选择与 SHAP 可解释性，该方法提高了 DDoS 检测的准确性和透明度。实验结果表明，平均反向数据包长度和最小正向数据包头部长度等关键特征在检测 DDoS 攻击中至关重要，提供了一种可扩展且可解析的网络安全解决方案。 

---
# A Characterization of List Language Identification in the Limit 

**Title (ZH)**: 列表语言在极限条件下的识别特征 

**Authors**: Moses Charikar, Chirag Pabbaraju, Ambuj Tewari  

**Link**: [PDF](https://arxiv.org/pdf/2511.04103)  

**Abstract**: We study the problem of language identification in the limit, where given a sequence of examples from a target language, the goal of the learner is to output a sequence of guesses for the target language such that all the guesses beyond some finite time are correct. Classical results of Gold showed that language identification in the limit is impossible for essentially any interesting collection of languages. Later, Angluin gave a precise characterization of language collections for which this task is possible. Motivated by recent positive results for the related problem of language generation, we revisit the classic language identification problem in the setting where the learner is given the additional power of producing a list of $k$ guesses at each time step. The goal is to ensure that beyond some finite time, one of the guesses is correct at each time step.
We give an exact characterization of collections of languages that can be $k$-list identified in the limit, based on a recursive version of Angluin's characterization (for language identification with a list of size $1$). This further leads to a conceptually appealing characterization: A language collection can be $k$-list identified in the limit if and only if the collection can be decomposed into $k$ collections of languages, each of which can be identified in the limit (with a list of size $1$). We also use our characterization to establish rates for list identification in the statistical setting where the input is drawn as an i.i.d. stream from a distribution supported on some language in the collection. Our results show that if a collection is $k$-list identifiable in the limit, then the collection can be $k$-list identified at an exponential rate, and this is best possible. On the other hand, if a collection is not $k$-list identifiable in the limit, then it cannot be $k$-list identified at any rate that goes to zero. 

**Abstract (ZH)**: 我们在限界内对目标语言的序列实例进行语言识别问题进行了研究，即给定目标语言的一系列示例，学习者的目標是在某个有限时间之后输出目标语言的一系列猜测，使得所有猜测都是正确的。Gold的古典结果表明，对于本质上任何有趣的语言集合，语言识别在限界内都是不可能的。后来，Angluin给出了完成这一任务的语言集合的精确描述。受最近关于相关问题语言生成的积极结果的启发，我们在学习者具有生成每步时间列表中$k$个猜测的额外能力的背景下重新审视了经典的语言识别问题。目标是在某个有限时间之后，每一步有一个猜测是正确的。我们基于Angluin的描述（针对列表大小为1的语言识别）给出了可以在限界内$k$列表识别的语言集合的精确描述，这进一步导致了一个概念上可接受的描述：如果一个语言集合可以以$k$列表的形式在限界内被识别，当且仅当该集合可以被分解为$k$个可以各自在限界内被识别的语言集合（带有一个列表）。我们还使用我们的描述来确定统计设置下输入从集合中某些语言的支持分布中作为独立同分布流抽取的情况下的列表识别率。我们的结果表明，如果一个集合可以在限界内$k$列表识别，那么该集合可以以指数速率$k$列表识别，这是最佳可能的。另一方面，如果一个集合不在限界内$k$列表识别，那么就不可能以任何趋零的速度$k$列表识别。 

---
# An Automated Theorem Generator with Theoretical Foundation Based on Rectangular Standard Contradiction 

**Title (ZH)**: 基于矩形标准矛盾的理论基础自动定理生成器 

**Authors**: Yang Xu, Peiyao Liu, Shuwei Chen, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04092)  

**Abstract**: Currently, there is a lack of rigorous theoretical system for systematically generating non-trivial and logically valid theorems. Addressing this critical gap, this paper conducts research to propose a novel automated theorem generation theory and tool. Based on the concept of standard contradiction which possesses unique deductive advantages, this paper defines and proves, for the first time, a new logical structure known as rectangular standard contradiction. Centered on this structure, a complete Automated Theorem Generation (ATG) theory is put forward. Theoretical proofs clarify two core properties of rectangular standard contradiction: first, it is a standard contradiction (necessarily unsatisfiable); second, it exhibits non-redundancy (the remaining clause set becomes satisfiable after removing any clause). Leveraging these properties, this paper proves that partitioning a rectangular standard contradiction into a premise subset $A$ and negation of its complement $H$, a valid theorem $A \vdash \neg H$ can be formed, and all such theorems are logically equivalent. To implement this theory, an efficient template-based ATG algorithm is designed, and a Rectangular Automated Theorem Generator is developed. This research enables machines to transition from "verifiers" to "discoverers", opening up new avenues for fundamental research in the fields of logic and artificial intelligence. 

**Abstract (ZH)**: 当前缺乏一个严谨的理论体系来进行系统性的非平凡且逻辑有效的定理生成。为填补这一关键空白，本文开展了相关研究，提出了一个新颖的自动定理生成理论和工具。基于具有独特演绎优势的标准矛盾概念，本文首次定义并证明了一种新的逻辑结构——矩形标准矛盾。以此结构为中心，本文提出了一个完整的自动定理生成（ATG）理论。理论证明阐明了矩形标准矛盾的两个核心属性：首先，它是标准矛盾（必然不可满足）；其次，它表现出非冗余性（移除任一子句后剩余子句集可满足）。利用这些属性，本文证明可以将矩形标准矛盾划分为前提子集$A$和其补集的否定$\neg H$，从而形成有效的定理$A \vdash \neg H$，且所有此类定理都是逻辑等价的。为实现这一理论，本文设计了一个高效的目标模板驱动的自动定理生成算法，并开发了矩形自动定理生成器。该研究使机器从“验证者”转变为“发现者”，为逻辑和人工智能领域的基础研究开辟了新的途径。 

---
# DeNoise: Learning Robust Graph Representations for Unsupervised Graph-Level Anomaly Detection 

**Title (ZH)**: 去噪：学习稳健的无监督图级异常检测图表示方法 

**Authors**: Qingfeng Chen, Haojin Zeng, Jingyi Jie, Shichao Zhang, Debo Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.04086)  

**Abstract**: With the rapid growth of graph-structured data in critical domains, unsupervised graph-level anomaly detection (UGAD) has become a pivotal task. UGAD seeks to identify entire graphs that deviate from normal behavioral patterns. However, most Graph Neural Network (GNN) approaches implicitly assume that the training set is clean, containing only normal graphs, which is rarely true in practice. Even modest contamination by anomalous graphs can distort learned representations and sharply degrade performance. To address this challenge, we propose DeNoise, a robust UGAD framework explicitly designed for contaminated training data. It jointly optimizes a graph-level encoder, an attribute decoder, and a structure decoder via an adversarial objective to learn noise-resistant embeddings. Further, DeNoise introduces an encoder anchor-alignment denoising mechanism that fuses high-information node embeddings from normal graphs into all graph embeddings, improving representation quality while suppressing anomaly interference. A contrastive learning component then compacts normal graph embeddings and repels anomalous ones in the latent space. Extensive experiments on eight real-world datasets demonstrate that DeNoise consistently learns reliable graph-level representations under varying noise intensities and significantly outperforms state-of-the-art UGAD baselines. 

**Abstract (ZH)**: 随着关键领域中图结构数据的迅速增长，无监督图级异常检测（UGAD）已成为一项关键任务。UGAD旨在识别偏离正常行为模式的整个图。然而，大多数图神经网络（GNN）方法隐含假设训练集是干净的，仅包含正常图，这一假设在实践中很少成立。即使是少量异常图的污染也能扭曲学习表示并大幅降低性能。为应对这一挑战，我们提出了一种名为DeNoise的鲁棒UGAD框架，专门设计用于处理污染的训练数据。DeNoise通过对抗目标联合优化图级编码器、属性解码器和结构解码器，学习抗噪表示。此外，DeNoise引入了编码器锚点对齐去噪机制，将正常图的高信息节点表示融合到所有图表示中，从而提高表示质量并抑制异常干扰。对比学习组件则在潜在空间中将正常图表示压缩，并排斥异常图表示。在八个真实世界数据集上的 extensive 实验表明，DeNoise 在不同噪声强度下都能学习到可靠的图级表示，并显著优于最先进的UGAD基准。 

---
# Left Atrial Segmentation with nnU-Net Using MRI 

**Title (ZH)**: 基于MRI的左心房分割方法：使用nnU-Net 

**Authors**: Fatemeh Hosseinabadi, Seyedhassan Sharifi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04071)  

**Abstract**: Accurate segmentation of the left atrium (LA) from cardiac MRI is critical for guiding atrial fibrillation (AF) ablation and constructing biophysical cardiac models. Manual delineation is time-consuming, observer-dependent, and impractical for large-scale or time-sensitive clinical workflows. Deep learning methods, particularly convolutional architectures, have recently demonstrated superior performance in medical image segmentation tasks. In this study, we applied the nnU-Net framework, an automated, self-configuring deep learning segmentation architecture, to the Left Atrial Segmentation Challenge 2013 dataset. The dataset consists of thirty MRI scans with corresponding expert-annotated masks. The nnU-Net model automatically adapted its preprocessing, network configuration, and training pipeline to the characteristics of the MRI data. Model performance was quantitatively evaluated using the Dice similarity coefficient (DSC), and qualitative results were compared against expert segmentations. The proposed nnU?Net model achieved a mean Dice score of 93.5, demonstrating high overlap with expert annotations and outperforming several traditional segmentation approaches reported in previous studies. The network exhibited robust generalization across variations in left atrial shape, contrast, and image quality, accurately delineating both the atrial body and proximal pulmonary veins. 

**Abstract (ZH)**: 左心房分割：左心房分割挑战2013数据集中基于nnU-Net的自动深度学习分割方法的研究 

---
# Accelerating scientific discovery with the common task framework 

**Title (ZH)**: 使用通用任务框架加速科学研究 

**Authors**: J. Nathan Kutz, Peter Battaglia, Michael Brenner, Kevin Carlberg, Aric Hagberg, Shirley Ho, Stephan Hoyer, Henning Lange, Hod Lipson, Michael W. Mahoney, Frank Noe, Max Welling, Laure Zanna, Francis Zhu, Steven L. Brunton  

**Link**: [PDF](https://arxiv.org/pdf/2511.04001)  

**Abstract**: Machine learning (ML) and artificial intelligence (AI) algorithms are transforming and empowering the characterization and control of dynamic systems in the engineering, physical, and biological sciences. These emerging modeling paradigms require comparative metrics to evaluate a diverse set of scientific objectives, including forecasting, state reconstruction, generalization, and control, while also considering limited data scenarios and noisy measurements. We introduce a common task framework (CTF) for science and engineering, which features a growing collection of challenge data sets with a diverse set of practical and common objectives. The CTF is a critically enabling technology that has contributed to the rapid advance of ML/AI algorithms in traditional applications such as speech recognition, language processing, and computer vision. There is a critical need for the objective metrics of a CTF to compare the diverse algorithms being rapidly developed and deployed in practice today across science and engineering. 

**Abstract (ZH)**: 机器学习和人工智能算法正在改造和增强工程、物理和生物科学中动力系统特性和控制的建模与实现。这些新兴的建模范式需要对比性指标来评估包括预测、状态重构、泛化和控制等一系列广泛的科学目标，同时还需要考虑有限数据情景和噪声测量。我们提出了一种通用任务框架（CTF），其特点是包含了一系列具有多种实用和常见目标的挑战性数据集。CTF 是一种关键使能技术，已经在传统应用如语音识别、语言处理和计算机视觉中促进了机器学习/人工智能算法的快速发展。科学和工程中正在快速开发和部署的多种算法亟需客观指标来进行比较。 

---
# Towards Scalable Meta-Learning of near-optimal Interpretable Models via Synthetic Model Generations 

**Title (ZH)**: 面向合成模型生成的可扩展元学习以获得近最优可解释模型 

**Authors**: Kyaw Hpone Myint, Zhe Wu, Alexandre G.R. Day, Giri Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2511.04000)  

**Abstract**: Decision trees are widely used in high-stakes fields like finance and healthcare due to their interpretability. This work introduces an efficient, scalable method for generating synthetic pre-training data to enable meta-learning of decision trees. Our approach samples near-optimal decision trees synthetically, creating large-scale, realistic datasets. Using the MetaTree transformer architecture, we demonstrate that this method achieves performance comparable to pre-training on real-world data or with computationally expensive optimal decision trees. This strategy significantly reduces computational costs, enhances data generation flexibility, and paves the way for scalable and efficient meta-learning of interpretable decision tree models. 

**Abstract (ZH)**: 决策树在金融和医疗等高 stakes 领域广泛应用，得益于其可解释性。本工作介绍了一种高效可扩展的方法，用于生成合成预训练数据，以实现决策树的元学习。我们的方法合成采样接近最优的决策树，生成大规模、现实的数据集。使用 MetaTree 转换器架构，我们展示该方法在性能上可与基于真实数据或计算成本高昂的最优决策树的预训练相媲美。该策略显著降低了计算成本、增强了数据生成的灵活性，并为可解释决策树模型的大规模高效元学习铺平了道路。 

---
# Multiscale Astrocyte Network Calcium Dynamics for Biologically Plausible Intelligence in Anomaly Detection 

**Title (ZH)**: 多尺度星形胶质细胞网络钙动力学在异常检测中的生物合 plausibility 智能 

**Authors**: Berk Iskar, Michael Taynnan Barros  

**Link**: [PDF](https://arxiv.org/pdf/2511.03993)  

**Abstract**: Network anomaly detection systems encounter several challenges with traditional detectors trained offline. They become susceptible to concept drift and new threats such as zero-day or polymorphic attacks. To address this limitation, we propose a Ca$^{2+}$-modulated learning framework that draws inspiration from astrocytic Ca$^{2+}$ signaling in the brain, where rapid, context-sensitive adaptation enables robust information processing. Our approach couples a multicellular astrocyte dynamics simulator with a deep neural network (DNN). The simulator models astrocytic Ca$^{2+}$ dynamics through three key mechanisms: IP$_3$-mediated Ca$^{2+}$ release, SERCA pump uptake, and conductance-aware diffusion through gap junctions between cells. Evaluation of our proposed network on CTU-13 (Neris) network traffic data demonstrates the effectiveness of our biologically plausible approach. The Ca$^{2+}$-gated model outperforms a matched baseline DNN, achieving up to $\sim$98\% accuracy with reduced false positives and negatives across multiple train/test splits. Importantly, this improved performance comes with negligible runtime overhead once Ca$^{2+}$ trajectories are precomputed. While demonstrated here for cybersecurity applications, this Ca$^{2+}$-modulated learning framework offers a generic solution for streaming detection tasks that require rapid, biologically grounded adaptation to evolving data patterns. 

**Abstract (ZH)**: Ca$^{2+}$调节的学习框架在网络安全中的应用 

---
# PETRA: Pretrained Evolutionary Transformer for SARS-CoV-2 Mutation Prediction 

**Title (ZH)**: PETRA: 预训练演化变换器在SARS-CoV-2突变预测中的应用 

**Authors**: Xu Zou  

**Link**: [PDF](https://arxiv.org/pdf/2511.03976)  

**Abstract**: Since its emergence, SARS-CoV-2 has demonstrated a rapid and unpredictable evolutionary trajectory, characterized by the continual emergence of immune-evasive variants. This poses persistent challenges to public health and vaccine development.
While large-scale generative pre-trained transformers (GPTs) have revolutionized the modeling of sequential data, their direct applications to noisy viral genomic sequences are limited. In this paper, we introduce PETRA(Pretrained Evolutionary TRAnsformer), a novel transformer approach based on evolutionary trajectories derived from phylogenetic trees rather than raw RNA sequences. This method effectively mitigates sequencing noise and captures the hierarchical structure of viral evolution.
With a weighted training framework to address substantial geographical and temporal imbalances in global sequence data, PETRA excels in predicting future SARS-CoV-2 mutations, achieving a weighted recall@1 of 9.45% for nucleotide mutations and 17.10\% for spike amino-acid mutations, compared to 0.49% and 6.64% respectively for the best baseline. PETRA also demonstrates its ability to aid in the real-time mutation prediction of major clades like 24F(XEC) and 25A(LP.8.1). The code is open sourced on this https URL 

**Abstract (ZH)**: 自SARS-CoV-2出现以来，其展示出一种快速且不可预测的演化轨迹，特征为不断出现免疫逃逸变异株。这对其公共健康和疫苗开发构成了持续的挑战。 

---
# Evolutionary Optimization Trumps Adam Optimization on Embedding Space Exploration 

**Title (ZH)**: 进化优化在嵌入空间探索中优于Adam优化 

**Authors**: Domício Pereira Neto, João Correia, Penousal Machado  

**Link**: [PDF](https://arxiv.org/pdf/2511.03913)  

**Abstract**: Deep generative models, especially diffusion architectures, have transformed image generation; however, they are challenging to control and optimize for specific goals without expensive retraining. Embedding Space Exploration, especially with Evolutionary Algorithms (EAs), has been shown to be a promising method for optimizing image generation, particularly within Diffusion Models. Therefore, in this work, we study the performance of an evolutionary optimization method, namely Separable Covariance Matrix Adaptation Evolution Strategy (sep-CMA-ES), against the widely adopted Adaptive Moment Estimation (Adam), applied to Stable Diffusion XL Turbo's prompt embedding vector. The evaluation of images combines the LAION Aesthetic Predictor V2 with CLIPScore into a weighted fitness function, allowing flexible trade-offs between visual appeal and adherence to prompts. Experiments on a subset of the Parti Prompts (P2) dataset showcase that sep-CMA-ES consistently yields superior improvements in aesthetic and alignment metrics in comparison to Adam. Results indicate that the evolutionary method provides efficient, gradient-free optimization for diffusion models, enhancing controllability without the need for fine-tuning. This study emphasizes the potential of evolutionary methods for embedding space exploration of deep generative models and outlines future research directions. 

**Abstract (ZH)**: 深生成模型，特别是扩散架构，已经革新了图像生成；然而，在不需要昂贵重训练的情况下，控制和优化这些模型以达到特定目标仍然是一个挑战。嵌入空间探索，尤其是借助进化算法（EAs），已被证明是一种优化图像生成的有效方法，尤其是在扩散模型中。因此，在本研究中，我们研究了分离协方差矩阵适应进化策略（sep-CMA-ES）相对于广泛采用的自适应矩估计（Adam）对稳定扩散XL Turbo提示嵌入向量的性能。评估图像的性能结合了LAION美学预测器V2与CLIPScore，从而允许在视觉吸引力与对提示的遵从性之间灵活权衡。在Parti提示集（P2）子集上的实验展示了sep-CMA-ES在美学和对齐度量上一致地提供优于Adam的改进。结果表明，进化方法能够为扩散模型提供高效的无梯度优化，从而提高可控性而无需微调。本研究强调了进化方法在深生成模型嵌入空间探索中的潜在价值，并指出了未来的研究方向。 

---
# I Detect What I Don't Know: Incremental Anomaly Learning with Stochastic Weight Averaging-Gaussian for Oracle-Free Medical Imaging 

**Title (ZH)**: 我不知则探：基于广义随机权重平均的无 oracle 医学影像增量异常学习 

**Authors**: Nand Kumar Yadav, Rodrigue Rizk, William CW Chen, KC Santosh  

**Link**: [PDF](https://arxiv.org/pdf/2511.03912)  

**Abstract**: Unknown anomaly detection in medical imaging remains a fundamental challenge due to the scarcity of labeled anomalies and the high cost of expert supervision. We introduce an unsupervised, oracle-free framework that incrementally expands a trusted set of normal samples without any anomaly labels. Starting from a small, verified seed of normal images, our method alternates between lightweight adapter updates and uncertainty-gated sample admission. A frozen pretrained vision backbone is augmented with tiny convolutional adapters, ensuring rapid domain adaptation with negligible computational overhead. Extracted embeddings are stored in a compact coreset enabling efficient k-nearest neighbor anomaly (k-NN) scoring. Safety during incremental expansion is enforced by dual probabilistic gates, a sample is admitted into the normal memory only if its distance to the existing coreset lies within a calibrated z-score threshold, and its SWAG-based epistemic uncertainty remains below a seed-calibrated bound. This mechanism prevents drift and false inclusions without relying on generative reconstruction or replay buffers. Empirically, our system steadily refines the notion of normality as unlabeled data arrive, producing substantial gains over baselines. On COVID-CXR, ROC-AUC improves from 0.9489 to 0.9982 (F1: 0.8048 to 0.9746); on Pneumonia CXR, ROC-AUC rises from 0.6834 to 0.8968; and on Brain MRI ND-5, ROC-AUC increases from 0.6041 to 0.7269 and PR-AUC from 0.7539 to 0.8211. These results highlight the effectiveness and efficiency of the proposed framework for real-world, label-scarce medical imaging applications. 

**Abstract (ZH)**: 无监督医学影像中的未知异常检测仍是一项基本挑战，由于标记异常样本的稀缺性和专家监督的成本高昂。我们提出了一种无监督且无需或acles的框架，该框架逐步扩展信任的正常样本集，无需任何异常标签。从一个小型且经过验证的正常图像种子开始，我们的方法交替进行轻量级适配器更新和不确定性门控样本接纳。预先训练的视觉骨干网络通过添加小型卷积适配器进行扩展，确保快速领域适应，同时几乎不增加计算开销。提取的嵌入存储在一个紧凑的核心集中，以实现高效的k-最近邻异常（k-NN）评分。在逐步扩展过程中的安全性通过双重概率门实现，只有当样本与现有核心集的距离在校准的z-score阈值内且其基于SWAG的表征不确定性低于种子校准的界限时，样本才会被接纳到正常记忆中。该机制防止了漂移和错误包含，而无需依赖生成重建或回放缓冲区。实验结果显示，随着未标记数据的到达，我们的系统逐步细化正常性的概念，产生了相对于基线的显著改进。在COVID-CXR上，AUC-ROC从0.9489提高到0.9982（F1分数从0.8048提高到0.9746）；在Pneumonia CXR上，AUC-ROC从0.6834提高到0.8968；在Brain MRI ND-5上，AUC-ROC从0.6041提高到0.7269，而PAUC从0.7539提高到0.8211。这些结果突显了所提出框架在实际、标签稀缺的医学影像应用中的有效性和高效性。 

---
# Improving Diagnostic Performance on Small and Imbalanced Datasets Using Class-Based Input Image Composition 

**Title (ZH)**: 基于类别的输入图像组合在小规模和不均衡数据集上的诊断性能改进 

**Authors**: Hlali Azzeddine, Majid Ben Yakhlef, Soulaiman El Hazzat  

**Link**: [PDF](https://arxiv.org/pdf/2511.03891)  

**Abstract**: Small, imbalanced datasets and poor input image quality can lead to high false predictions rates with deep learning models. This paper introduces Class-Based Image Composition, an approach that allows us to reformulate training inputs through a fusion of multiple images of the same class into combined visual composites, named Composite Input Images (CoImg). That enhances the intra-class variance and improves the valuable information density per training sample and increases the ability of the model to distinguish between subtle disease patterns. Our method was evaluated on the Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods (OCTDL) (Kulyabin et al., 2024), which contains 2,064 high-resolution optical coherence tomography (OCT) scans of the human retina, representing seven distinct diseases with a significant class imbalance. We constructed a perfectly class-balanced version of this dataset, named Co-OCTDL, where each scan is resented as a 3x1 layout composite image. To assess the effectiveness of this new representation, we conducted a comparative analysis between the original dataset and its variant using a VGG16 model. A fair comparison was ensured by utilizing the identical model architecture and hyperparameters for all experiments. The proposed approach markedly improved diagnostic this http URL enhanced Dataset achieved near-perfect accuracy (99.6%) with F1-score (0.995) and AUC (0.9996), compared to a baseline model trained on raw dataset. The false prediction rate was also significantly lower, this demonstrates that the method can producehigh-quality predictions even for weak datasets affected by class imbalance or small sample size. 

**Abstract (ZH)**: 基于类别的图像合成：提高深度学习模型诊断性能的新方法 

---
# Levers of Power in the Field of AI 

**Title (ZH)**: AI领域中的权力杠杆 

**Authors**: Tammy Mackenzie, Sukriti Punj, Natalie Perez, Sreyoshi Bhaduri, Branislav Radeljic  

**Link**: [PDF](https://arxiv.org/pdf/2511.03859)  

**Abstract**: This paper examines how decision makers in academia, government, business, and civil society navigate questions of power in implementations of artificial intelligence. The study explores how individuals experience and exercise levers of power, which are presented as social mechanisms that shape institutional responses to technological change. The study reports on the responses of personalized questionnaires designed to gather insight on a decision maker's institutional purview, based on an institutional governance framework developed from the work of Neo-institutionalists. Findings present the anonymized, real responses and circumstances of respondents in the form of twelve fictional personas of high-level decision makers from North America and Europe. These personas illustrate how personal agency, organizational logics, and institutional infrastructures may intersect in the governance of AI. The decision makers' responses to the questionnaires then inform a discussion of the field-level personal power of decision makers, methods of fostering institutional stability in times of change, and methods of influencing institutional change in the field of AI. The final section of the discussion presents a table of the dynamics of the levers of power in the field of AI for change makers and five testable hypotheses for institutional and social movement researchers. In summary, this study provides insight on the means for policymakers within institutions and their counterparts in civil society to personally engage with AI governance. 

**Abstract (ZH)**: 本研究探讨了学术界、政府、企业和社会各界决策者在人工智能实施过程中如何 navigater 权力相关问题。研究探索了个体在技术变革背景下体验和行使权力杠杆的方式，这些杠杆被视为塑造机构对技术变革响应的社会机制。研究基于新制度主义者的工作开发的机构治理框架，收集定制问卷以获取决策者在其职权范围内的洞见。研究结果以北美洲和欧洲的十二个虚构高级决策者的人格展示匿名真实的回答和情境，以此说明个人能动性、组织逻辑和机构基础设施在人工智能治理中的可能交集。决策者对问卷的回答进而推动对机构内决策者个人权力、促进变革时期机构稳定的方法以及影响人工智能领域机构变革的方法的讨论。讨论的最后部分提供了一份展示人工智能领域权力杠杆动态的表格，并提出了五个可测试的假设，供机构和社会运动研究人员检验。总结而言，本研究为机构内的政策制定者及其社会 counterpart 如何个人参与人工智能治理提供了洞见。 

---
# Noise Injection: Improving Out-of-Distribution Generalization for Limited Size Datasets 

**Title (ZH)**: 噪声注入：提高小规模数据集的分布外泛化能力 

**Authors**: Duong Mai, Lawrence Hall  

**Link**: [PDF](https://arxiv.org/pdf/2511.03855)  

**Abstract**: Deep learned (DL) models for image recognition have been shown to fail to generalize to data from different devices, populations, etc. COVID-19 detection from Chest X-rays (CXRs), in particular, has been shown to fail to generalize to out-of-distribution (OOD) data from new clinical sources not covered in the training set. This occurs because models learn to exploit shortcuts - source-specific artifacts that do not translate to new distributions - rather than reasonable biomarkers to maximize performance on in-distribution (ID) data. Rendering the models more robust to distribution shifts, our study investigates the use of fundamental noise injection techniques (Gaussian, Speckle, Poisson, and Salt and Pepper) during training. Our empirical results demonstrate that this technique can significantly reduce the performance gap between ID and OOD evaluation from 0.10-0.20 to 0.01-0.06, based on results averaged over ten random seeds across key metrics such as AUC, F1, accuracy, recall and specificity. Our source code is publicly available at this https URL 

**Abstract (ZH)**: 基于深度学习的图像识别模型在不同设备和人群的数据上表现出泛化能力不足，特别是在胸部X光片（CXR）中的 COVID-19 检测中，模型未能泛化到训练集中未涵盖的新临床来源的非分布数据中。这是因为模型学习利用了特定源的捷径，这些捷径在新的分布中不能转化，而不是学习合理的生物标记来最大化在分布内（ID）数据上的性能。通过增强模型对分布偏移的鲁棒性，本研究探讨了在训练过程中使用基本噪声注入技术（高斯噪声、speckle噪声、泊松噪声和椒盐噪声）的应用。我们的实证结果表明，这种技术可以显著缩小基于诸如AUC、F1、准确率、召回率和特异性的关键指标在分布内（ID）和非分布外（OOD）评估之间的性能差距，从0.10-0.20缩小到0.01-0.06。我们的源代码已在此公开网址处提供。 

---
# Optimizing Reasoning Efficiency through Prompt Difficulty Prediction 

**Title (ZH)**: 通过提示难度预测优化推理效率 

**Authors**: Bo Zhao, Berkcan Kapusuzoglu, Kartik Balasubramaniam, Sambit Sahu, Supriyo Chakraborty, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2511.03808)  

**Abstract**: Reasoning language models perform well on complex tasks but are costly to deploy due to their size and long reasoning traces. We propose a routing approach that assigns each problem to the smallest model likely to solve it, reducing compute without sacrificing accuracy. Using intermediate representations from s1.1-32B, we train lightweight predictors of problem difficulty or model correctness to guide routing across a pool of reasoning models. On diverse math benchmarks, routing improves efficiency over random assignment and matches s1.1-32B's performance while using significantly less compute. Our results demonstrate that difficulty-aware routing is effective for cost-efficient deployment of reasoning models. 

**Abstract (ZH)**: 基于推理的语言模型在复杂任务上表现优秀，但由于其规模庞大和长推理路径，部署成本较高。我们提出了一种路由方法，将每个问题分配给最有可能解决它的最小模型，从而在减少计算资源的同时不牺牲准确性。利用s1.1-32B的中间表示，我们训练轻量级的模型难度预测器或模型正确性预测器，指导在一系列推理模型中进行路由。在各种数学基准测试中，路由方法相较于随机分配提高了效率，并且在使用显著较少的计算资源的情况下达到了与s1.1-32B相同的表现。我们的研究表明，基于难度的路由对于推理模型的成本-efficient部署是有效的。 

---
# Climbing the label tree: Hierarchy-preserving contrastive learning for medical imaging 

**Title (ZH)**: 沿着标签树攀登：带有层次保留的对比学习在医疗成像中的应用 

**Authors**: Alif Elham Khan  

**Link**: [PDF](https://arxiv.org/pdf/2511.03771)  

**Abstract**: Medical image labels are often organized by taxonomies (e.g., organ - tissue - subtype), yet standard self-supervised learning (SSL) ignores this structure. We present a hierarchy-preserving contrastive framework that makes the label tree a first-class training signal and an evaluation target. Our approach introduces two plug-in objectives: Hierarchy-Weighted Contrastive (HWC), which scales positive/negative pair strengths by shared ancestors to promote within-parent coherence, and Level-Aware Margin (LAM), a prototype margin that separates ancestor groups across levels. The formulation is geometry-agnostic and applies to Euclidean and hyperbolic embeddings without architectural changes. Across several benchmarks, including breast histopathology, the proposed objectives consistently improve representation quality over strong SSL baselines while better respecting the taxonomy. We evaluate with metrics tailored to hierarchy faithfulness: HF1 (hierarchical F1), H-Acc (tree-distance-weighted accuracy), and parent-distance violation rate. We also report top-1 accuracy for completeness. Ablations show that HWC and LAM are effective even without curvature, and combining them yields the most taxonomy-aligned representations. Taken together, these results provide a simple, general recipe for learning medical image representations that respect the label tree and advance both performance and interpretability in hierarchy-rich domains. 

**Abstract (ZH)**: 一种保留层级结构的对比学习框架：面向医学图像标签层次结构的自监督学习 

---
# OptiMA: A Transaction-Based Framework with Throughput Optimization for Very Complex Multi-Agent Systems 

**Title (ZH)**: OptiMA：一种基于事务的吞吐量优化框架，用于非常复杂的多代理系统 

**Authors**: Umut Çalıkyılmaz, Nitin Nayak, Jinghua Groppe, Sven Groppe  

**Link**: [PDF](https://arxiv.org/pdf/2511.03761)  

**Abstract**: In recent years, the research of multi-agent systems has taken a direction to explore larger and more complex models to fulfill sophisticated tasks. We point out two possible pitfalls that might be caused by increasing complexity; susceptibilities to faults, and performance bottlenecks. To prevent the former threat, we propose a transaction-based framework to design very complex multi-agent systems (VCMAS). To address the second threat, we offer to integrate transaction scheduling into the proposed framework. We implemented both of these ideas to develop the OptiMA framework and show that it is able to facilitate the execution of VCMAS with more than a hundred agents. We also demonstrate the effect of transaction scheduling on such a system by showing improvements up to more than 16\%. Furthermore, we also performed a theoretical analysis on the transaction scheduling problem and provided practical tools that can be used for future research on it. 

**Abstract (ZH)**: 近年来，多Agent系统的研究方向转向探索更大、更复杂的模型以完成复杂的任务。我们指出了复杂性增加可能导致的两种潜在问题：故障易感性和性能瓶颈。为了防止前者，我们提出了一种基于事务的框架来设计非常复杂的多Agent系统（VCMAS）。为了应对后者，我们建议将事务调度整合到提出的框架中。我们实施了这两种想法，开发了OptiMA框架，并证明了它能够支持超过一百个Agent的复杂执行。此外，我们通过展示超过16%的性能改进，证明了事务调度对这种系统的影响。我们还对事务调度问题进行了理论分析，并提供了可用于未来研究的实际工具。 

---
# Federated Learning with Gramian Angular Fields for Privacy-Preserving ECG Classification on Heterogeneous IoT Devices 

**Title (ZH)**: 基于Gramian角场的联邦学习在异构IoT设备上进行隐私保护的心电图分类 

**Authors**: Youssef Elmir, Yassine Himeur, Abbes Amira  

**Link**: [PDF](https://arxiv.org/pdf/2511.03753)  

**Abstract**: This study presents a federated learning (FL) framework for privacy-preserving electrocardiogram (ECG) classification in Internet of Things (IoT) healthcare environments. By transforming 1D ECG signals into 2D Gramian Angular Field (GAF) images, the proposed approach enables efficient feature extraction through Convolutional Neural Networks (CNNs) while ensuring that sensitive medical data remain local to each device. This work is among the first to experimentally validate GAF-based federated ECG classification across heterogeneous IoT devices, quantifying both performance and communication efficiency. To evaluate feasibility in realistic IoT settings, we deployed the framework across a server, a laptop, and a resource-constrained Raspberry Pi 4, reflecting edge-cloud integration in IoT ecosystems. Experimental results demonstrate that the FL-GAF model achieves a high classification accuracy of 95.18% in a multi-client setup, significantly outperforming a single-client baseline in both accuracy and training time. Despite the added computational complexity of GAF transformations, the framework maintains efficient resource utilization and communication overhead. These findings highlight the potential of lightweight, privacy-preserving AI for IoT-based healthcare monitoring, supporting scalable and secure edge deployments in smart health systems. 

**Abstract (ZH)**: 隐私保护的心电图（ECG）分类的联邦学习（FL）框架：基于Gramian Angular Field (GAF)的物联网（IoT） healthcare环境中的应用 

---
# Applying Time Series Deep Learning Models to Forecast the Growth of Perennial Ryegrass in Ireland 

**Title (ZH)**: 将时间序列深度学习模型应用于预测爱尔兰多年生黑麦草的增长 

**Authors**: Oluwadurotimi Onibonoje, Vuong M. Ngo, Andrew McCarre, Elodie Ruelle, Bernadette O-Briend, Mark Roantree  

**Link**: [PDF](https://arxiv.org/pdf/2511.03749)  

**Abstract**: Grasslands, constituting the world's second-largest terrestrial carbon sink, play a crucial role in biodiversity and the regulation of the carbon cycle. Currently, the Irish dairy sector, a significant economic contributor, grapples with challenges related to profitability and sustainability. Presently, grass growth forecasting relies on impractical mechanistic models. In response, we propose deep learning models tailored for univariate datasets, presenting cost-effective alternatives. Notably, a temporal convolutional network designed for forecasting Perennial Ryegrass growth in Cork exhibits high performance, leveraging historical grass height data with RMSE of 2.74 and MAE of 3.46. Validation across a comprehensive dataset spanning 1,757 weeks over 34 years provides insights into optimal model configurations. This study enhances our understanding of model behavior, thereby improving reliability in grass growth forecasting and contributing to the advancement of sustainable dairy farming practices. 

**Abstract (ZH)**: 草地构成了世界第二大陆地碳汇，对生物多样性和碳循环的调节起着至关重要的作用。目前，作为重要的经济贡献者，爱尔兰奶业部门面临着盈利能力与可持续性的挑战。现有的草地生长预测依赖于不切实际的机理模型。为此，我们提出了一种适用于单变量数据集的深度学习模型，提供了一种成本效益较高的替代方案。采用临时卷积网络预测科克地区多年生黑麦草生长表现出高性能，利用历史草高数据，其均方根误差（RMSE）为2.74，平均绝对误差（MAE）为3.46。全面数据集的验证跨越34年1757周的数据提供了解最优模型配置的见解。本研究增强了我们对模型行为的理解，从而提高了草地生长预测的可靠性，并有助于推动可持续奶业生产实践的发展。 

---
# OpenMENA: An Open-Source Memristor Interfacing and Compute Board for Neuromorphic Edge-AI Applications 

**Title (ZH)**: OpenMENA：一种用于类脑边缘AI应用的开源忆阻器接口和计算板 

**Authors**: Ali Safa, Farida Mohsen, Zainab Ali, Bo Wang, Amine Bermak  

**Link**: [PDF](https://arxiv.org/pdf/2511.03747)  

**Abstract**: Memristive crossbars enable in-memory multiply-accumulate and local plasticity learning, offering a path to energy-efficient edge AI. To this end, we present Open-MENA (Open Memristor-in-Memory Accelerator), which, to our knowledge, is the first fully open memristor interfacing system integrating (i) a reproducible hardware interface for memristor crossbars with mixed-signal read-program-verify loops; (ii) a firmware-software stack with high-level APIs for inference and on-device learning; and (iii) a Voltage-Incremental Proportional-Integral (VIPI) method to program pre-trained weights into analog conductances, followed by chip-in-the-loop fine-tuning to mitigate device non-idealities. OpenMENA is validated on digit recognition, demonstrating the flow from weight transfer to on-device adaptation, and on a real-world robot obstacle-avoidance task, where the memristor-based model learns to map localization inputs to motor commands. OpenMENA is released as open source to democratize memristor-enabled edge-AI research. 

**Abstract (ZH)**: 忆阻交叉阵列使能内存计算和局部塑性学习，提供了一条能效更高的边缘AI路径。为此，我们提出了Open-MENA（开源自的记忆中忆阻加速器），据我们所知，这是首个集成了(i) 可重复的混合信号读取-编程-验证回路的忆阻交叉阵列硬件接口；(ii) 高级API的固件-软件栈，用于推理和设备内学习；以及(iii) 电压递增逐段积分法(VIPI)来编程预训练权重，并通过芯片在环中进行精细调整以缓解设备非理想性的完全开源忆阻接口系统。OpenMENA在数字识别和真实世界的机器人避障任务中得到了验证，展示了从权重转移至设备内适应的流程，并且基于忆阻器的模型能够将定位输入映射至电机命令。OpenMENA作为开源软件发布，旨在促进忆阻器使能的边缘AI研究的普及。 

---
# Conversational Collective Intelligence (CCI) using Hyperchat AI in an Authentic Forecasting Task 

**Title (ZH)**: 使用Hyperchat AI的会话集体智能（CCI）在实际预测任务中 

**Authors**: Hans Schumann, Louis Rosenberg, Ganesh Mani, Gregg Willcox  

**Link**: [PDF](https://arxiv.org/pdf/2511.03732)  

**Abstract**: Hyperchat AI is a novel agentic technology that enables thoughtful conversations among networked human groups of potentially unlimited size. It allows large teams to discuss complex issues, brainstorm ideas, surface risks, assess alternatives and efficiently converge on optimized solutions that amplify the group's Collective Intelligence (CI). A formal study was conducted to quantify the forecasting accuracy of human groups using Hyperchat AI to conversationally predict the outcome of Major League Baseball (MLB) games. During an 8-week period, networked groups of approximately 24 sports fans were tasked with collaboratively forecasting the winners of 59 baseball games through real-time conversation facilitated by AI agents. The results showed that when debating the games using Hyperchat AI technology, the groups converged on High Confidence predictions that significantly outperformed Vegas betting markets. Specifically, groups were 78% accurate in their High Confidence picks, a statistically strong result vs the Vegas odds of 57% (p=0.020). Had the groups bet against the spread (ATS) on these games, they would have achieved a 46% ROI against Vegas betting markets. In addition, High Confidence forecasts that were generated through above-average conversation rates were 88% accurate, suggesting that real-time interactive deliberation is central to amplified accuracy. 

**Abstract (ZH)**: Hyperchat AI是一种新型代理技术，可促进网络化人群之间的有意义对话，这些人群数量可能无限。它使大型团队能够讨论复杂问题、brainstorm想法、揭示风险、评估替代方案并高效地收敛于优化的解决方案，以放大群体的集体智能（CI）。一项正式研究通过实对话预测职业棒球大联盟（MLB）比赛结果，量化了人类群体使用Hyperchat AI进行交流预测的准确性。在8周时间内，通过AI代理实时对话协作预测59场棒球比赛结果的约24名体育迷组成的网络化群体，在辩论这些比赛时，群体提出了高置信度的预测，显著优于赌博市场赔率。具体来说，群体在高置信度选择中的准确率为78%，与赌市赔率57%相比，这一结果具有统计学意义（p=0.020）。若群体对这些比赛下注，他们与赌市的回报率将达到46%。此外，通过高于平均水平的互动对话生成的高置信度预测的准确率为88%，表明实时互动讨论对提高准确性至关重要。 

---
# MimiTalk: Revolutionizing Qualitative Research with Dual-Agent AI 

**Title (ZH)**: MimiTalk: 以双智能体AI重塑定性研究 

**Authors**: Fengming Liu, Shubin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.03731)  

**Abstract**: We present MimiTalk, a dual-agent constitutional AI framework designed for scalable and ethical conversational data collection in social science research. The framework integrates a supervisor model for strategic oversight and a conversational model for question generation. We conducted three studies: Study 1 evaluated usability with 20 participants; Study 2 compared 121 AI interviews to 1,271 human interviews from the MediaSum dataset using NLP metrics and propensity score matching; Study 3 involved 10 interdisciplinary researchers conducting both human and AI interviews, followed by blind thematic analysis. Results across studies indicate that MimiTalk reduces interview anxiety, maintains conversational coherence, and outperforms human interviews in information richness, coherence, and stability. AI interviews elicit technical insights and candid views on sensitive topics, while human interviews better capture cultural and emotional nuances. These findings suggest that dual-agent constitutional AI supports effective human-AI collaboration, enabling replicable, scalable and quality-controlled qualitative research. 

**Abstract (ZH)**: MimiTalk：一种用于社会科学领域可扩展和伦理对话数据收集的双代理宪法AI框架 

---
# Simulation-Based Validation of an Integrated 4D/5D Digital-Twin Framework for Predictive Construction Control 

**Title (ZH)**: 基于仿真的集成4D/5D数字孪生框架的预测施工控制验证 

**Authors**: Atena Khoshkonesh, Mohsen Mohammadagha, Navid Ebrahimi  

**Link**: [PDF](https://arxiv.org/pdf/2511.03684)  

**Abstract**: Persistent cost and schedule deviations remain a major challenge in the U.S. construction industry, revealing the limitations of deterministic CPM and static document-based estimating. This study presents an integrated 4D/5D digital-twin framework that couples Building Information Modeling (BIM) with natural-language processing (NLP)-based cost mapping, computer-vision (CV)-driven progress measurement, Bayesian probabilistic CPM updating, and deep-reinforcement-learning (DRL) resource-leveling. A nine-month case implementation on a Dallas-Fort Worth mid-rise project demonstrated measurable gains in accuracy and efficiency: 43% reduction in estimating labor, 6% reduction in overtime, and 30% project-buffer utilization, while maintaining an on-time finish at 128 days within P50-P80 confidence bounds. The digital-twin sandbox also enabled real-time "what-if" forecasting and traceable cost-schedule alignment through a 5D knowledge graph. Findings confirm that integrating AI-based analytics with probabilistic CPM and DRL enhances forecasting precision, transparency, and control resilience. The validated workflow establishes a practical pathway toward predictive, adaptive, and auditable construction management. 

**Abstract (ZH)**: 基于人工智能分析的概率CPM和DRL集成的4D/5D数字孪生框架在建筑行业的应用与验证 

---
# Exploratory Analysis of Cyberattack Patterns on E-Commerce Platforms Using Statistical Methods 

**Title (ZH)**: 使用统计方法探索电子商务平台上的网络攻击模式 

**Authors**: Fatimo Adenike Adeniya  

**Link**: [PDF](https://arxiv.org/pdf/2511.03020)  

**Abstract**: Cyberattacks on e-commerce platforms have grown in sophistication, threatening consumer trust and operational continuity. This research presents a hybrid analytical framework that integrates statistical modelling and machine learning for detecting and forecasting cyberattack patterns in the e-commerce domain. Using the Verizon Community Data Breach (VCDB) dataset, the study applies Auto ARIMA for temporal forecasting and significance testing, including a Mann-Whitney U test (U = 2579981.5, p = 0.0121), which confirmed that holiday shopping events experienced significantly more severe cyberattacks than non-holiday periods. ANOVA was also used to examine seasonal variation in threat severity, while ensemble machine learning models (XGBoost, LightGBM, and CatBoost) were employed for predictive classification. Results reveal recurrent attack spikes during high-risk periods such as Black Friday and holiday seasons, with breaches involving Personally Identifiable Information (PII) exhibiting elevated threat indicators. Among the models, CatBoost achieved the highest performance (accuracy = 85.29%, F1 score = 0.2254, ROC AUC = 0.8247). The framework uniquely combines seasonal forecasting with interpretable ensemble learning, enabling temporal risk anticipation and breach-type classification. Ethical considerations, including responsible use of sensitive data and bias assessment, were incorporated. Despite class imbalance and reliance on historical data, the study provides insights for proactive cybersecurity resource allocation and outlines directions for future real-time threat detection research. 

**Abstract (ZH)**: 电子商务平台上的网络攻击日益 sophisticated，威胁消费者信任和运营连续性。本研究提出了一种集成统计建模与机器学习的混合分析框架，用于检测和预测电子商务领域中的网络攻击模式。使用Verizon社区数据泄露（VCDB）数据集，研究采用Auto ARIMA进行时序预测和显著性检验，包括Mann-Whitney U检验（U = 2579981.5，p = 0.0121），结果表明节日购物期遭受的网络攻击比非节日时期更为严重。研究还使用ANOVA检验威胁严重性随季节的变化，并使用集成机器学习模型（XGBoost、LightGBM和CatBoost）进行预测分类。结果表明，在高风险期如黑色星期五和节日季期间，涉及个人可识别信息（PII）的泄漏显示出较高的威胁指标。在此类模型中，CatBoost表现出最佳性能（准确率=85.29%，F1分数=0.2254，ROC AUC=0.8247）。该框架独树一帜地结合了季节预测与可解释的集成学习，能够实现时间风险预测和漏洞类型分类。研究中还纳入了伦理考量，包括敏感数据的负责任使用和偏见评估。尽管存在类别不平衡和依赖历史数据的问题，但研究为前瞻性的网络安全资源分配提供了见解，并指出了未来实时威胁检测研究的方向。 

---
