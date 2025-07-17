# Xiangqi-R1: Enhancing Spatial Strategic Reasoning in LLMs for Chinese Chess via Reinforcement Learning 

**Title (ZH)**: 象棋-R1：通过强化学习增强空间战略推理的LLMs在中国象棋中的应用 

**Authors**: Yuhao Chen, Shuochen Liu, Yuanjie Lyu, Chao Zhang, Jiayao Shi, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12215)  

**Abstract**: Game playing has long served as a fundamental benchmark for evaluating Artificial General Intelligence (AGI). While Large Language Models (LLMs) have demonstrated impressive capabilities in general reasoning, their effectiveness in spatial strategic reasoning, which is critical for complex and fully observable board games, remains insufficiently explored. In this work, we adopt Chinese Chess (Xiangqi) as a challenging and rich testbed due to its intricate rules and spatial complexity. To advance LLMs' strategic competence in such environments, we propose a training framework tailored to Xiangqi, built upon a large-scale dataset of five million board-move pairs enhanced with expert annotations and engine evaluations. Building on this foundation, we introduce Xiangqi-R1, a 7B-parameter model trained in multi-stage manner: (1) fine-tuning for legal move prediction to capture basic spatial rules, (2) incorporating strategic annotations to improve decision-making, and (3) applying reinforcement learning via Group Relative Policy Optimization (GRPO) with multi-dimensional reward signals to enhance reasoning stability. Our Experimental results indicate that, despite their size and power, general-purpose LLMs struggle to achieve satisfactory performance in these tasks. Compared to general-purpose LLMs, Xiangqi-R1 greatly advances with an 18% rise in move legality and a 22% boost in analysis accuracy. Our results point to a promising path for creating general strategic intelligence in spatially complex areas. 

**Abstract (ZH)**: 大规模语言模型在棋类策略 reasoning 中的空间复杂性挑战及突破：以中国象棋(Xiangqi)为例 

---
# BuildEvo: Designing Building Energy Consumption Forecasting Heuristics via LLM-driven Evolution 

**Title (ZH)**: BuildEvo: 通过LLM驱动的进化设计建筑能源消耗预测启发式方法 

**Authors**: Subin Lin, Chuanbo Hua  

**Link**: [PDF](https://arxiv.org/pdf/2507.12207)  

**Abstract**: Accurate building energy forecasting is essential, yet traditional heuristics often lack precision, while advanced models can be opaque and struggle with generalization by neglecting physical principles. This paper introduces BuildEvo, a novel framework that uses Large Language Models (LLMs) to automatically design effective and interpretable energy prediction heuristics. Within an evolutionary process, BuildEvo guides LLMs to construct and enhance heuristics by systematically incorporating physical insights from building characteristics and operational data (e.g., from the Building Data Genome Project 2). Evaluations show BuildEvo achieves state-of-the-art performance on benchmarks, offering improved generalization and transparent prediction logic. This work advances the automated design of robust, physically grounded heuristics, promoting trustworthy models for complex energy systems. 

**Abstract (ZH)**: 使用大型语言模型自动设计有效可解释的能量预测启发式方法：BuildEvo框架 

---
# Partially Observable Reference Policy Programming: Solving POMDPs Sans Numerical Optimisation 

**Title (ZH)**: 部分可观测参考策略编程：无需数值优化求解POMDP 

**Authors**: Edward Kim, Hanna Kurniawati  

**Link**: [PDF](https://arxiv.org/pdf/2507.12186)  

**Abstract**: This paper proposes Partially Observable Reference Policy Programming, a novel anytime online approximate POMDP solver which samples meaningful future histories very deeply while simultaneously forcing a gradual policy update. We provide theoretical guarantees for the algorithm's underlying scheme which say that the performance loss is bounded by the average of the sampling approximation errors rather than the usual maximum, a crucial requirement given the sampling sparsity of online planning. Empirical evaluations on two large-scale problems with dynamically evolving environments -- including a helicopter emergency scenario in the Corsica region requiring approximately 150 planning steps -- corroborate the theoretical results and indicate that our solver considerably outperforms current online benchmarks. 

**Abstract (ZH)**: 该论文提出了一种新颖的即席在线近似POMDP求解器——部分可观测参考策略编程，该方法在深度采样有意义的未来历史的同时，强制执行逐步的策略更新。我们提供了该算法底层机制的理论保证，表明性能损失由采样逼近误差的平均值而不是通常的最大值来界，这对于在线规划中采样稀疏性来说是一个关键要求。针对包括科西嘉地区约150步规划步骤的动态变化环境中的直升机紧急场景等两个大规模问题的实证评价结果与理论分析一致，表明我们的求解器显著优于现有在线基准。 

---
# Topology Enhanced MARL for Multi-Vehicle Cooperative Decision-Making of CAVs 

**Title (ZH)**: _topology Enhanced MARL for Multi-Vehicle Cooperative Decision-Making of Connected Automated Vehicles_ 

**Authors**: Ye Han, Lijun Zhang, Dejian Meng, Zhuang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12110)  

**Abstract**: The exploration-exploitation trade-off constitutes one of the fundamental challenges in reinforcement learning (RL), which is exacerbated in multi-agent reinforcement learning (MARL) due to the exponential growth of joint state-action spaces. This paper proposes a topology-enhanced MARL (TPE-MARL) method for optimizing cooperative decision-making of connected and autonomous vehicles (CAVs) in mixed traffic. This work presents two primary contributions: First, we construct a game topology tensor for dynamic traffic flow, effectively compressing high-dimensional traffic state information and decrease the search space for MARL algorithms. Second, building upon the designed game topology tensor and using QMIX as the backbone RL algorithm, we establish a topology-enhanced MARL framework incorporating visit counts and agent mutual information. Extensive simulations across varying traffic densities and CAV penetration rates demonstrate the effectiveness of TPE-MARL. Evaluations encompassing training dynamics, exploration patterns, macroscopic traffic performance metrics, and microscopic vehicle behaviors reveal that TPE-MARL successfully balances exploration and exploitation. Consequently, it exhibits superior performance in terms of traffic efficiency, safety, decision smoothness, and task completion. Furthermore, the algorithm demonstrates decision-making rationality comparable to or exceeding that of human drivers in both mixed-autonomy and fully autonomous traffic scenarios. Code of our work is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 拓扑增强的多智能体强化学习方法（TPE-MARL）：优化连接和自主车辆在混合交通中的协同决策 

---
# Understanding visual attention beehind bee-inspired UAV navigation 

**Title (ZH)**: 理解基于蜂群启发的无人飞行器导航中的视觉注意力机制 

**Authors**: Pranav Rajbhandari, Abhi Veda, Matthew Garratt, Mandayam Srinivasan, Sridhar Ravi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11992)  

**Abstract**: Bio-inspired design is often used in autonomous UAV navigation due to the capacity of biological systems for flight and obstacle avoidance despite limited sensory and computational capabilities. In particular, honeybees mainly use the sensory input of optic flow, the apparent motion of objects in their visual field, to navigate cluttered environments. In our work, we train a Reinforcement Learning agent to navigate a tunnel with obstacles using only optic flow as sensory input. We inspect the attention patterns of trained agents to determine the regions of optic flow on which they primarily base their motor decisions. We find that agents trained in this way pay most attention to regions of discontinuity in optic flow, as well as regions with large optic flow magnitude. The trained agents appear to navigate a cluttered tunnel by avoiding the obstacles that produce large optic flow, while maintaining a centered position in their environment, which resembles the behavior seen in flying insects. This pattern persists across independently trained agents, which suggests that this could be a good strategy for developing a simple explicit control law for physical UAVs. 

**Abstract (ZH)**: 生物启发设计在自主无人机导航中的应用：基于光学流的隧道导航研究成果 

---
# Aime: Towards Fully-Autonomous Multi-Agent Framework 

**Title (ZH)**: Aime: 向完全自主多代理框架迈进 

**Authors**: Yexuan Shi, Mingyu Wang, Yunxiang Cao, Hongjie Lai, Junjian Lan, Xin Han, Yu Wang, Jie Geng, Zhenan Li, Zihao Xia, Xiang Chen, Chen Li, Jian Xu, Wenbo Duan, Yuanshuo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11988)  

**Abstract**: Multi-Agent Systems (MAS) powered by Large Language Models (LLMs) are emerging as a powerful paradigm for solving complex, multifaceted problems. However, the potential of these systems is often constrained by the prevalent plan-and-execute framework, which suffers from critical limitations: rigid plan execution, static agent capabilities, and inefficient communication. These weaknesses hinder their adaptability and robustness in dynamic environments. This paper introduces Aime, a novel multi-agent framework designed to overcome these challenges through dynamic, reactive planning and execution. Aime replaces the conventional static workflow with a fluid and adaptive architecture. Its core innovations include: (1) a Dynamic Planner that continuously refines the overall strategy based on real-time execution feedback; (2) an Actor Factory that implements Dynamic Actor instantiation, assembling specialized agents on-demand with tailored tools and knowledge; and (3) a centralized Progress Management Module that serves as a single source of truth for coherent, system-wide state awareness. We empirically evaluated Aime on a diverse suite of benchmarks spanning general reasoning (GAIA), software engineering (SWE-bench Verified), and live web navigation (WebVoyager). The results demonstrate that Aime consistently outperforms even highly specialized state-of-the-art agents in their respective domains. Its superior adaptability and task success rate establish Aime as a more resilient and effective foundation for multi-agent collaboration. 

**Abstract (ZH)**: 由大型语言模型（LLMs）驱动的多代理系统（MAS）正在成为解决复杂多方面问题的强大范式。然而，这些系统的能力常常受限于普遍的计划与执行框架，该框架存在关键局限性：刚性计划执行、静态代理能力以及低效的通信。这些弱点阻碍了它们在动态环境中的适应性和鲁棒性。本文介绍了Aime，这是一种新型多代理框架，通过动态、反应式规划与执行来克服这些挑战。Aime 用一种流动且适应性强的架构取代了传统的静态工作流。其核心创新包括：（1）一个动态规划器，根据实时执行反馈不断细化整体策略；（2）一个演员工厂，实现动态演员实例化，根据需要组装具备定制化工具和技术的专业化代理；（3）一个集中式进度管理模块，作为全局一致状态意识的单一来源。我们在涵盖通用推理（GAIA）、软件工程（SWE-bench Verified）和实时网页导航（WebVoyager）的多样基准测试中实证评估了Aime。结果表明，Aime 在其各自领域的一流专业代理中表现更优。其卓越的适应性和任务成功率使Aime 成为多代理协作更为稳健和有效的基础。 

---
# A Parallel CPU-GPU Framework for Cost-Bounded DFS with Applications to IDA* and BTS 

**Title (ZH)**: 基于成本限制的并行CPU-GPU框架及其在IDA*和BTS中的应用 

**Authors**: Ehsan Futuhi, Nathan R. Sturtevant  

**Link**: [PDF](https://arxiv.org/pdf/2507.11916)  

**Abstract**: The rapid advancement of GPU technology has unlocked powerful parallel processing capabilities, creating new opportunities to enhance classic search algorithms. A recent successful application of GPUs is in compressing large pattern database (PDB) heuristics using neural networks while preserving heuristic admissibility. However, very few algorithms have been designed to exploit GPUs during search. Several variants of A* exist that batch GPU computations. In this paper we introduce a method for batching GPU computations in depth first search. In particular, we describe a new cost-bounded depth-first search (CB-DFS) method that leverages the combined parallelism of modern CPUs and GPUs. This is used to create algorithms like \emph{Batch IDA*}, an extension of the Iterative Deepening A* (IDA*) algorithm, or Batch BTS, an extensions of Budgeted Tree Search. Our approach builds on the general approach used by Asynchronous Parallel IDA* (AIDA*), while maintaining optimality guarantees. We evaluate the approach on the 3x3 Rubik's Cube and 4x4 sliding tile puzzle (STP), showing that GPU operations can be efficiently batched in DFS. Additionally, we conduct extensive experiments to analyze the effects of hyperparameters, neural network heuristic size, and hardware resources on performance. 

**Abstract (ZH)**: GPU加速技术的迅速发展解锁了强大的并行处理能力，为增强经典搜索算法提供了新机遇。最近GPU的一个成功应用是在使用神经网络压缩大型模式数据库（PDB）启发式算法的同时保持启发式的可接纳性。然而，设计利用GPU进行搜索的算法很少。已存在几种A*算法的变体，它们批处理GPU计算。本文介绍了一种在深度优先搜索中批处理GPU计算的方法。特别是，我们描述了一种新的成本上限深度优先搜索（CB-DFS）方法，利用现代CPU和GPU的联合并行性。该方法用于创建如Batch IDA*（迭代加深A*的扩展）或Batch BTS（预算树搜索的扩展）等算法。我们的方法基于Asynchronous Parallel IDA*（AIDA*）的一般方法，同时保持最优性保证。我们在3x3 Rubik's立方体和4x4滑块拼图（STP）上评估了该方法，证明了可以在深度优先搜索中有效批处理GPU操作。此外，我们进行了广泛的实验来分析超参数、神经网络启发式大小和硬件资源对性能的影响。 

---
# Survey of Swarm Intelligence Approaches to Search Documents Based On Semantic Similarity 

**Title (ZH)**: 基于语义相似性的文档搜索 Swarm智能方法调研 

**Authors**: Chandrashekar Muniyappa, Eunjin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.11787)  

**Abstract**: Swarm Intelligence (SI) is gaining a lot of popularity in artificial intelligence, where the natural behavior of animals and insects is observed and translated into computer algorithms called swarm computing to solve real-world problems. Due to their effectiveness, they are applied in solving various computer optimization problems. This survey will review all the latest developments in Searching for documents based on semantic similarity using Swarm Intelligence algorithms and recommend future research directions. 

**Abstract (ZH)**: 基于 swarm intelligence 算法的文档基于语义相似性搜索：最新发展与未来研究方向 

---
# Auto-Formulating Dynamic Programming Problems with Large Language Models 

**Title (ZH)**: 使用大型语言模型自动生成动态规划问题 

**Authors**: Chenyu Zhou, Jingyuan Yang, Linwei Xin, Yitian Chen, Ziyan He, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2507.11737)  

**Abstract**: Dynamic programming (DP) is a fundamental method in operations research, but formulating DP models has traditionally required expert knowledge of both the problem context and DP techniques. Large Language Models (LLMs) offer the potential to automate this process. However, DP problems pose unique challenges due to their inherently stochastic transitions and the limited availability of training data. These factors make it difficult to directly apply existing LLM-based models or frameworks developed for other optimization problems, such as linear or integer programming. We introduce DP-Bench, the first benchmark covering a wide range of textbook-level DP problems to enable systematic evaluation. We present Dynamic Programming Language Model (DPLM), a 7B-parameter specialized model that achieves performance comparable to state-of-the-art LLMs like OpenAI's o1 and DeepSeek-R1, and surpasses them on hard problems. Central to DPLM's effectiveness is DualReflect, our novel synthetic data generation pipeline, designed to scale up training data from a limited set of initial examples. DualReflect combines forward generation for diversity and backward generation for reliability. Our results reveal a key insight: backward generation is favored in low-data regimes for its strong correctness guarantees, while forward generation, though lacking such guarantees, becomes increasingly valuable at scale for introducing diverse formulations. This trade-off highlights the complementary strengths of both approaches and the importance of combining them. 

**Abstract (ZH)**: 动态规划语言模型(DPLM):面向动态规划问题的专门模型及其实验基准DP-Bench 

---
# ClarifAI: Enhancing AI Interpretability and Transparency through Case-Based Reasoning and Ontology-Driven Approach for Improved Decision-Making 

**Title (ZH)**: ClarifAI：通过案例推理和本体驱动方法提高AI的可解释性和透明度以改善决策制定 

**Authors**: Srikanth Vemula  

**Link**: [PDF](https://arxiv.org/pdf/2507.11733)  

**Abstract**: This Study introduces Clarity and Reasoning Interface for Artificial Intelligence(ClarifAI), a novel approach designed to augment the transparency and interpretability of artificial intelligence (AI) in the realm of improved decision making. Leveraging the Case-Based Reasoning (CBR) methodology and integrating an ontology-driven approach, ClarifAI aims to meet the intricate explanatory demands of various stakeholders involved in AI-powered applications. The paper elaborates on ClarifAI's theoretical foundations, combining CBR and ontologies to furnish exhaustive explanation mechanisms. It further elaborates on the design principles and architectural blueprint, highlighting ClarifAI's potential to enhance AI interpretability across different sectors and its applicability in high-stake environments. This research delineates the significant role of ClariAI in advancing the interpretability of AI systems, paving the way for its deployment in critical decision-making processes. 

**Abstract (ZH)**: 这项研究介绍了清晰性和推理界面 for 人工智能（ClarifAI），这是一种新型方法，旨在通过改善决策透明度和可解释性来增强人工智能（AI）。利用案例推理（CBR）方法并结合本体驱动的方法，ClarifAI 意图满足各种利益相关者在 AI 驱动应用中复杂的解释需求。论文详细介绍了 ClarifAI 的理论基础，将 CBR 和本体相结合，提供详尽的解释机制。还进一步阐述了设计原则和架构蓝图，强调 ClarifAI 在不同领域增强 AI 可解释性以及在其高风险环境中的适用性。这项研究阐述了 ClarifAI 在推进 AI 系统可解释性方面的重大作用，为其在关键决策过程中的部署铺平了道路。 

---
# Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification 

**Title (ZH)**: 让我们分两步思考：通过自我接地验证减轻MLLMs的共识偏差 

**Authors**: Moises Andrade, Joonhyuk Cha, Brandon Ho, Vriksha Srihari, Karmesh Yadav, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2507.11662)  

**Abstract**: Verifiers -- functions assigning rewards to agent behavior -- have been key for AI progress in domains like math and board games. However, extending these gains to domains without clear-cut success criteria (e.g.,computer use) remains a challenge: while humans can recognize suitable outcomes, translating this intuition into scalable rules is non-trivial. Multimodal Large Language Models(MLLMs) emerge as a promising solution, given their world knowledge, human-preference alignment, and reasoning skills. We evaluate MLLMs as verifiers of agent trajectories across web navigation, computer use, and robotic manipulation, and identify a critical limitation: agreement bias, a strong tendency for MLLMs to favor information in their context window, often generating chains of thought to rationalize flawed behavior. This bias is pervasive across models, resilient to test-time scaling, and can impact several methods using MLLMs as evaluators (e.g.,data filtering). Notably, it occurs despite MLLMs showing strong, human-aligned priors on desired behavior. To address this, we propose Self-Grounded Verification (SGV), a lightweight method that enables more effective use of MLLMs' knowledge and reasoning by harnessing their own sampling mechanisms via unconditional and conditional generation. SGV operates in two steps: first, the MLLM is elicited to retrieve broad priors about task completion, independent of the data under evaluation. Then, conditioned on self-generated priors, it reasons over and evaluates a candidate trajectory. Enhanced with SGV, MLLM verifiers show gains of up to 20 points in accuracy and failure detection rates, and can perform real-time supervision of heterogeneous agents, boosting task completion of a GUI specialist in OSWorld, a diffusion policy in robomimic, and a ReAct agent in VisualWebArena -- setting a new state of the art on the benchmark, surpassing the previous best by 48%. 

**Abstract (ZH)**: 验证器——评估智能体行为并授予奖励的功能——在数学和棋类游戏等领域推动了AI的进步。但是，将这些成果扩展到缺乏明确成功标准的领域（如计算机使用）仍然面临挑战：尽管人类能够识别合适的结局，但将这种直觉转化为可扩展的规则并非易事。多模态大型语言模型(MLLMs)因其世界观知识、与人类偏好的对齐以及推理能力而成为一种有前景的解决方案。我们评估了MLLMs作为跨网络导航、计算机使用和机器人操作中智能体轨迹的验证器，并识别出一个关键限制：一致性偏差，即MLLMs倾向于偏好其上下文窗口中的信息，经常生成推理以合理化错误行为的思维链。这种偏差在各个模型中普遍存在，对测试时的扩展具有抵抗力，并可影响使用MLLMs作为评估器的多种方法（例如，数据过滤）。值得注意的是，尽管MLLMs在期望行为上表现出强烈的与人类偏好的对齐先验，这种偏差仍然会出现。为此，我们提出了一种名为自我定位验证（SGV）的方法，这是一种轻量级的方法，通过利用MLLMs自身的采样机制（无条件和有条件生成）来增强其知识和推理的使用效果，从而使其能够更有效地发挥作用。SGV分为两个步骤：首先，MLLM被激发以检索与评估数据无关的任务完成的广泛先验知识。然后，在自我生成的先验知识的条件下，MLLM推理并评估候选轨迹。增强SGV后，MLLM验证器的准确性提高了20分，并且在失败检测率方面也有所提升，能够实现对异构智能体的实时监督，从而在OSWorld中的GUI专家、robomimic中的扩散策略以及VisualWebArena中的ReAct智能体的任务完成上取得提升，新基准上的性能超越了之前的最佳结果48%。 

---
# General Modular Harness for LLM Agents in Multi-Turn Gaming Environments 

**Title (ZH)**: 面向多轮游戏环境的大规模语言模型代理通用模块化框架 

**Authors**: Yuxuan Zhang, Haoyang Yu, Lanxiang Hu, Haojian Jin, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11633)  

**Abstract**: We introduce a modular harness design for LLM agents that composes of perception, memory, and reasoning components, enabling a single LLM or VLM backbone to tackle a wide spectrum of multi turn gaming environments without domain-specific engineering. Using classic and modern game suites as low-barrier, high-diversity testbeds, our framework provides a unified workflow for analyzing how each module affects performance across dynamic interactive settings. Extensive experiments demonstrate that the harness lifts gameplay performance consistently over un-harnessed baselines and reveals distinct contribution patterns, for example, memory dominates in long-horizon puzzles while perception is critical in vision noisy arcades. These findings highlight the effectiveness of our modular harness design in advancing general-purpose agent, given the familiarity and ubiquity of games in everyday human experience. 

**Abstract (ZH)**: 模块化 harness 设计在多轮游戏环境中的应用：基于感知、记忆和推理组件的通用大规模语言模型代理及其效果分析 

---
# A Study on the Application of Artificial Intelligence in Ecological Design 

**Title (ZH)**: 人工智能在生态设计中的应用研究 

**Authors**: Hengyue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11595)  

**Abstract**: This paper asks whether our relationship with nature can move from human dominance to genuine interdependence, and whether artificial intelligence (AI) can mediate that shift. We examine a new ecological-design paradigm in which AI interacts with non-human life forms. Through case studies we show how artists and designers apply AI for data analysis, image recognition, and ecological restoration, producing results that differ from conventional media. We argue that AI not only expands creative methods but also reframes the theory and practice of ecological design. Building on the author's prototype for AI-assisted water remediation, the study proposes design pathways that couple reinforcement learning with plant-based phytoremediation. The findings highlight AI's potential to link scientific insight, artistic practice, and environmental stewardship, offering a roadmap for future research on sustainable, technology-enabled ecosystems. 

**Abstract (ZH)**: 本文探讨了我们与自然的关系能否从人类主导转变为真正的相互依存，以及人工智能（AI）能否促进这种转变。我们研究了一种新的生态设计范式，在这种范式中，AI与非人类生命形式互动。通过案例研究，我们展示了艺术家和设计师如何利用AI进行数据分析、图像识别和生态恢复，并产生了不同于传统媒体的结果。我们认为，AI不仅扩展了创作方法，还重新定义了生态设计的理论和实践。基于作者的AI辅助水 remediation 试验原型，本研究提出了结合强化学习与基于植物的phytoremediation 的设计路径。研究发现强调了AI在连接科学洞察、艺术实践和环境 stewardship 方面的潜在作用，为可持续技术赋能生态系统的研究提供了研究路线图。 

---
# Interpreting Radiologist's Intention from Eye Movements in Chest X-ray Diagnosis 

**Title (ZH)**: 基于胸部X光诊断中医生眼球运动解读医生意图 

**Authors**: Trong-Thang Pham, Anh Nguyen, Zhigang Deng, Carol C. Wu, Hien Van Nguyen, Ngan Le  

**Link**: [PDF](https://arxiv.org/pdf/2507.12461)  

**Abstract**: Radiologists rely on eye movements to navigate and interpret medical images. A trained radiologist possesses knowledge about the potential diseases that may be present in the images and, when searching, follows a mental checklist to locate them using their gaze. This is a key observation, yet existing models fail to capture the underlying intent behind each fixation. In this paper, we introduce a deep learning-based approach, RadGazeIntent, designed to model this behavior: having an intention to find something and actively searching for it. Our transformer-based architecture processes both the temporal and spatial dimensions of gaze data, transforming fine-grained fixation features into coarse, meaningful representations of diagnostic intent to interpret radiologists' goals. To capture the nuances of radiologists' varied intention-driven behaviors, we process existing medical eye-tracking datasets to create three intention-labeled subsets: RadSeq (Systematic Sequential Search), RadExplore (Uncertainty-driven Exploration), and RadHybrid (Hybrid Pattern). Experimental results demonstrate RadGazeIntent's ability to predict which findings radiologists are examining at specific moments, outperforming baseline methods across all intention-labeled datasets. 

**Abstract (ZH)**: 放射科医生依靠眼睛运动来导航和解读医学图像。经过培训的放射科医生具备关于影像中可能出现的潜在疾病的知识，并在搜索时通过目光遵循一个心理检查表来定位它们。这是一个关键观察，但现有模型未能捕捉到每次集中注意力背后的意图。在这篇论文中，我们介绍了一种基于深度学习的方法——RadGazeIntent，旨在模拟这种行为：具有发现某物的意图并主动寻找它。我们的基于变换器的架构处理凝视数据的时间和空间维度，将精细的固定点特征转换为粗略且有意义的诊断意图表示，以解释放射科医生的目标。为了捕捉放射科医生多样化意图驱动行为的细微差别，我们处理现有的医学眼动追踪数据集，创建了三个标签化的子集：RadSeq（系统性顺序搜索）、RadExplore（不确定性驱动的探索）和RadHybrid（混合模式）。实验结果表明，RadGazeIntent 能够预测放射科医生在特定时刻正在查看哪些发现，其性能在所有标签化的数据集中均优于基准方法。 

---
# S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling 

**Title (ZH)**: S2WTM：球面分割Wasserstein自编码器用于主题建模 

**Authors**: Suman Adhya, Debarshi Kumar Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.12451)  

**Abstract**: Modeling latent representations in a hyperspherical space has proven effective for capturing directional similarities in high-dimensional text data, benefiting topic modeling. Variational autoencoder-based neural topic models (VAE-NTMs) commonly adopt the von Mises-Fisher prior to encode hyperspherical structure. However, VAE-NTMs often suffer from posterior collapse, where the KL divergence term in the objective function highly diminishes, leading to ineffective latent representations. To mitigate this issue while modeling hyperspherical structure in the latent space, we propose the Spherical Sliced Wasserstein Autoencoder for Topic Modeling (S2WTM). S2WTM employs a prior distribution supported on the unit hypersphere and leverages the Spherical Sliced-Wasserstein distance to align the aggregated posterior distribution with the prior. Experimental results demonstrate that S2WTM outperforms state-of-the-art topic models, generating more coherent and diverse topics while improving performance on downstream tasks. 

**Abstract (ZH)**: 基于球面Sliced Wasserstein自编码器的主题建模（S2WTM） 

---
# LLM-Based Config Synthesis requires Disambiguation 

**Title (ZH)**: LLM 基础配置合成需要消歧义 

**Authors**: Rajdeep Mondal, Nikolaj Bjorner, Todd Millstein, Alan Tang, George Varghese  

**Link**: [PDF](https://arxiv.org/pdf/2507.12443)  

**Abstract**: Beyond hallucinations, another problem in program synthesis using LLMs is ambiguity in user intent. We illustrate the ambiguity problem in a networking context for LLM-based incremental configuration synthesis of route-maps and ACLs. These structures frequently overlap in header space, making the relative priority of actions impossible for the LLM to infer without user interaction. Measurements in a large cloud identify complex ACLs with 100's of overlaps, showing ambiguity is a real problem. We propose a prototype system, Clarify, which uses an LLM augmented with a new module called a Disambiguator that helps elicit user intent. On a small synthetic workload, Clarify incrementally synthesizes routing policies after disambiguation and then verifies them. Our treatment of ambiguities is useful more generally when the intent of updates can be correctly synthesized by LLMs, but their integration is ambiguous and can lead to different global behaviors. 

**Abstract (ZH)**: Beyond Hallucinations, Another Problem in Program Synthesis Using LLMs is Ambiguity in User Intent 

---
# Characterizing State Space Model (SSM) and SSM-Transformer Hybrid Language Model Performance with Long Context Length 

**Title (ZH)**: 基于长上下文长度下状态空间模型（SSM）及其与Transformer混合语言模型性能研究 

**Authors**: Saptarshi Mitra, Rachid Karami, Haocheng Xu, Sitao Huang, Hyoukjun Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2507.12442)  

**Abstract**: The demand for machine intelligence capable of processing continuous, long-context inputs on local devices is growing rapidly. However, the quadratic complexity and memory requirements of traditional Transformer architectures make them inefficient and often unusable for these tasks. This has spurred a paradigm shift towards new architectures like State Space Models (SSMs) and hybrids, which promise near-linear scaling. While most current research focuses on the accuracy and theoretical throughput of these models, a systematic performance characterization on practical consumer hardware is critically needed to guide system-level optimization and unlock new applications.
To address this gap, we present a comprehensive, comparative benchmarking of carefully selected Transformer, SSM, and hybrid models specifically for long-context inference on consumer and embedded GPUs. Our analysis reveals that SSMs are not only viable but superior for this domain, capable of processing sequences up to 220K tokens on a 24GB consumer GPU-approximately 4x longer than comparable Transformers. While Transformers may be up to 1.8x faster at short sequences, SSMs demonstrate a dramatic performance inversion, becoming up to 4x faster at very long contexts (~57K tokens). Our operator-level analysis reveals that custom, hardware-aware SSM kernels dominate the inference runtime, accounting for over 55% of latency on edge platforms, identifying them as a primary target for future hardware acceleration. We also provide detailed, device-specific characterization results to guide system co-design for the edge. To foster further research, we will open-source our characterization framework. 

**Abstract (ZH)**: 机器智能对于处理本地设备上的连续长上下文输入的需求正在迅速增长。然而，传统Transformer架构的二次复杂度和内存需求使其效率低下，常常无法用于这些任务。这推动了向状态空间模型（SSMs）和混合架构的范式转变，这些架构有望实现接近线性的扩展。虽然当前大多数研究集中在这些模型的准确性和理论吞吐量上，但在实际消费级硬件上的系统级性能表征却显得尤为关键，以指导系统优化并解锁新应用。

为解决这一缺口，我们对精心选择的适用于消费者和嵌入式GPU的长上下文推理任务的Transformer、SSM和混合模型进行了全面对比基准测试。我们的分析表明，SSM不仅可行，而且在这个领域更优，能够在24GB消费级GPU上处理最多220K词元的序列，约是同等Transformer的4倍长。虽然在短序列中Transformer可能快至1.8倍，但在非常长的上下文（约57K词元）下，SSM的性能表现出现了戏剧性的反转，快至4倍。我们的操作级分析显示，自定义且硬件感知的SSM内核主导了推理运行时，占边缘平台延迟的超过55%，将其作为未来硬件加速的主要目标。我们还提供了详细的特定设备表征结果，以指导边缘系统的协同设计。为了促进进一步的研究，我们将开源我们的表征框架。 

---
# EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos 

**Title (ZH)**: 自视角语言行动模型：从自视角人类视频中学习Vision-Language-Action模型 

**Authors**: Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Hongxu Yin, Sifei Liu, Song Han, Yao Lu, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12440)  

**Abstract**: Real robot data collection for imitation learning has led to significant advancements in robotic manipulation. However, the requirement for robot hardware in the process fundamentally constrains the scale of the data. In this paper, we explore training Vision-Language-Action (VLA) models using egocentric human videos. The benefit of using human videos is not only for their scale but more importantly for the richness of scenes and tasks. With a VLA trained on human video that predicts human wrist and hand actions, we can perform Inverse Kinematics and retargeting to convert the human actions to robot actions. We fine-tune the model using a few robot manipulation demonstrations to obtain the robot policy, namely EgoVLA. We propose a simulation benchmark called Isaac Humanoid Manipulation Benchmark, where we design diverse bimanual manipulation tasks with demonstrations. We fine-tune and evaluate EgoVLA with Isaac Humanoid Manipulation Benchmark and show significant improvements over baselines and ablate the importance of human data. Videos can be found on our website: this https URL 

**Abstract (ZH)**: 基于人类视角视频训练Vision-Language-Action模型以实现仿学习和机器人操作 

---
# Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models 

**Title (ZH)**: 可以在模型思考完毕前预测一致性和偏移吗？Towards 监控偏移推理模型的研究 

**Authors**: Yik Siu Chan, Zheng-Xin Yong, Stephen H. Bach  

**Link**: [PDF](https://arxiv.org/pdf/2507.12428)  

**Abstract**: Open-weights reasoning language models generate long chains-of-thought (CoTs) before producing a final response, which improves performance but introduces additional alignment risks, with harmful content often appearing in both the CoTs and the final outputs. In this work, we investigate if we can use CoTs to predict final response misalignment. We evaluate a range of monitoring approaches, including humans, highly-capable large language models, and text classifiers, using either CoT text or activations. First, we find that a simple linear probe trained on CoT activations can significantly outperform all text-based methods in predicting whether a final response will be safe or unsafe. CoT texts are often unfaithful and can mislead humans and classifiers, while model latents (i.e., CoT activations) offer a more reliable predictive signal. Second, the probe makes accurate predictions before reasoning completes, achieving strong performance even when applied to early CoT segments. These findings generalize across model sizes, families, and safety benchmarks, suggesting that lightweight probes could enable real-time safety monitoring and early intervention during generation. 

**Abstract (ZH)**: 基于CoTs预测最终响应偏移的探究：轻量级探针在实时安全监控中的潜力 

---
# Unit-Based Histopathology Tissue Segmentation via Multi-Level Feature Representation 

**Title (ZH)**: 基于单元的组织病理学组织分割：多级特征表示 

**Authors**: Ashkan Shakarami, Azade Farshad, Yousef Yeganeh, Lorenzo Nicole, Peter Schuffler, Stefano Ghidoni, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2507.12427)  

**Abstract**: We propose UTS, a unit-based tissue segmentation framework for histopathology that classifies each fixed-size 32 * 32 tile, rather than each pixel, as the segmentation unit. This approach reduces annotation effort and improves computational efficiency without compromising accuracy. To implement this approach, we introduce a Multi-Level Vision Transformer (L-ViT), which benefits the multi-level feature representation to capture both fine-grained morphology and global tissue context. Trained to segment breast tissue into three categories (infiltrating tumor, non-neoplastic stroma, and fat), UTS supports clinically relevant tasks such as tumor-stroma quantification and surgical margin assessment. Evaluated on 386,371 tiles from 459 H&E-stained regions, it outperforms U-Net variants and transformer-based baselines. Code and Dataset will be available at GitHub. 

**Abstract (ZH)**: 基于单元的病理组织分割框架UTS：一种以32×32单元格为分割单位的细粒度形态与全局组织上下文捕获方法 

---
# Advancing Retrieval-Augmented Generation for Structured Enterprise and Internal Data 

**Title (ZH)**: 增强检索辅助生成以应用于结构化企业内部数据 

**Authors**: Chandana Cheerla  

**Link**: [PDF](https://arxiv.org/pdf/2507.12425)  

**Abstract**: Organizations increasingly rely on proprietary enterprise data, including HR records, structured reports, and tabular documents, for critical decision-making. While Large Language Models (LLMs) have strong generative capabilities, they are limited by static pretraining, short context windows, and challenges in processing heterogeneous data formats. Conventional Retrieval-Augmented Generation (RAG) frameworks address some of these gaps but often struggle with structured and semi-structured data.
This work proposes an advanced RAG framework that combines hybrid retrieval strategies using dense embeddings (all-mpnet-base-v2) and BM25, enhanced by metadata-aware filtering with SpaCy NER and cross-encoder reranking. The framework applies semantic chunking to maintain textual coherence and retains tabular data structures to preserve row-column integrity. Quantized indexing optimizes retrieval efficiency, while human-in-the-loop feedback and conversation memory improve adaptability.
Experiments on enterprise datasets show notable improvements: Precision@5 increased by 15 percent (90 versus 75), Recall@5 by 13 percent (87 versus 74), and Mean Reciprocal Rank by 16 percent (0.85 versus 0.69). Qualitative evaluations show higher scores in Faithfulness (4.6 versus 3.0), Completeness (4.2 versus 2.5), and Relevance (4.5 versus 3.2) on a 5-point Likert scale. These results demonstrate the framework's effectiveness in delivering accurate, comprehensive, and contextually relevant responses for enterprise tasks. Future work includes extending to multimodal data and integrating agent-based retrieval. The source code will be released at this https URL 

**Abstract (ZH)**: 组织越来越依赖包括人力资源记录、结构化报告和表格文档在内的专有企业数据，用于关键决策。尽管大型语言模型（LLMs）具有强大的生成能力，但它们受限于静态预训练、短上下文窗口以及处理异构数据格式的挑战。传统检索增强生成（RAG）框架弥补了一些这些差距，但往往在处理结构化和半结构化数据时存在困难。

本文提出了一种先进的RAG框架，结合了使用密集嵌入（all-mpnet-base-v2）和BM25的混合检索策略，并通过SpaCy NER元数据感知过滤和交叉编码重排序进行增强。该框架应用语义切分以保持文本连贯性，并保留表格数据结构以保持行列完整性。量化索引优化了检索效率，而人工在环反馈和对话记忆提高了可适应性。

实验企业数据集上的结果显示显著改进：Precision@5提高了15个百分点（90比75），Recall@5提高了13个百分点（87比74），Mean Reciprocal Rank提高了16个百分点（0.85比0.69）。定性评估显示，在五点李克特量表上，忠实度得分为4.6（原为3.0），完整性得分为4.2（原为2.5），相关性得分为4.5（原为3.2）获得更高分数。这些结果表明该框架在为企业任务提供准确、全面且语境相关响应方面具有有效性。未来工作包括扩展到多模态数据并整合基于代理的检索。源代码将发布在以下地址：this https URL。 

---
# Mixture of Raytraced Experts 

**Title (ZH)**: Raytraced Experts 混合模型 

**Authors**: Andrea Perin, Giacomo Lagomarsini, Claudio Gallicchio, Giuseppe Nuti  

**Link**: [PDF](https://arxiv.org/pdf/2507.12419)  

**Abstract**: We introduce a Mixture of Raytraced Experts, a stacked Mixture of Experts (MoE) architecture which can dynamically select sequences of experts, producing computational graphs of variable width and depth. Existing MoE architectures generally require a fixed amount of computation for a given sample. Our approach, in contrast, yields predictions with increasing accuracy as the computation cycles through the experts' sequence. We train our model by iteratively sampling from a set of candidate experts, unfolding the sequence akin to how Recurrent Neural Networks are trained. Our method does not require load-balancing mechanisms, and preliminary experiments show a reduction in training epochs of 10\% to 40\% with a comparable/higher accuracy. These results point to new research directions in the field of MoEs, allowing the design of potentially faster and more expressive models. The code is available at this https URL 

**Abstract (ZH)**: 我们介绍了一种混合实时光线追踪专家模型，这是一种堆叠的专家混合（MoE）架构，可以动态选择专家序列，生成宽度和深度可变的计算图。现有的MoE架构通常需要为给定样本固定数量的计算量。相比之下，我们的方法可以通过依次通过专家序列来逐步提高预测准确性。我们通过迭代从候选专家集中采样，类似于递归神经网络的训练方式来训练模型。我们的方法不需要负载均衡机制，并且初步实验结果显示，在训练轮次减少10%到40%的情况下，可以获得同等或更高的准确性。这些结果为MoEs领域的新研究方向指明了道路，允许设计更快速和更具表达性的模型。代码可在以下链接获取。 

---
# QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval 

**Title (ZH)**: QuRe：组成图像检索中的困难负样本查询相关检索 

**Authors**: Jaehyun Kwak, Ramahdani Muhammad Izaaz Inhar, Se-Young Yun, Sung-Ju Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.12416)  

**Abstract**: Composed Image Retrieval (CIR) retrieves relevant images based on a reference image and accompanying text describing desired modifications. However, existing CIR methods only focus on retrieving the target image and disregard the relevance of other images. This limitation arises because most methods employing contrastive learning-which treats the target image as positive and all other images in the batch as negatives-can inadvertently include false negatives. This may result in retrieving irrelevant images, reducing user satisfaction even when the target image is retrieved. To address this issue, we propose Query-Relevant Retrieval through Hard Negative Sampling (QuRe), which optimizes a reward model objective to reduce false negatives. Additionally, we introduce a hard negative sampling strategy that selects images positioned between two steep drops in relevance scores following the target image, to effectively filter false negatives. In order to evaluate CIR models on their alignment with human satisfaction, we create Human-Preference FashionIQ (HP-FashionIQ), a new dataset that explicitly captures user preferences beyond target retrieval. Extensive experiments demonstrate that QuRe achieves state-of-the-art performance on FashionIQ and CIRR datasets while exhibiting the strongest alignment with human preferences on the HP-FashionIQ dataset. The source code is available at this https URL. 

**Abstract (ZH)**: 基于查询的相关图像检索（QuRe）：通过困难负样本采样减少虚假负样本 

---
# AutoVDC: Automated Vision Data Cleaning Using Vision-Language Models 

**Title (ZH)**: AutoVDC：使用视觉-语言模型的自动化视觉数据清洗 

**Authors**: Santosh Vasa, Aditi Ramadwar, Jnana Rama Krishna Darabattula, Md Zafar Anwar, Stanislaw Antol, Andrei Vatavu, Thomas Monninger, Sihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.12414)  

**Abstract**: Training of autonomous driving systems requires extensive datasets with precise annotations to attain robust performance. Human annotations suffer from imperfections, and multiple iterations are often needed to produce high-quality datasets. However, manually reviewing large datasets is laborious and expensive. In this paper, we introduce AutoVDC (Automated Vision Data Cleaning) framework and investigate the utilization of Vision-Language Models (VLMs) to automatically identify erroneous annotations in vision datasets, thereby enabling users to eliminate these errors and enhance data quality. We validate our approach using the KITTI and nuImages datasets, which contain object detection benchmarks for autonomous driving. To test the effectiveness of AutoVDC, we create dataset variants with intentionally injected erroneous annotations and observe the error detection rate of our approach. Additionally, we compare the detection rates using different VLMs and explore the impact of VLM fine-tuning on our pipeline. The results demonstrate our method's high performance in error detection and data cleaning experiments, indicating its potential to significantly improve the reliability and accuracy of large-scale production datasets in autonomous driving. 

**Abstract (ZH)**: 自动驾驶系统训练需要具有精确标注的大规模数据集以实现 robust 性能。人工标注存在缺陷，往往需要多轮迭代来生成高质量数据集。然而，人工审查大量数据集是劳神费财的。本文提出 AutoVDC（自动视觉数据清洗）框架，并探索视觉语言模型（VLMs）自动识别视觉数据集中的错误标注，从而让用户能够消除这些错误并提高数据质量。我们使用包含自主驾驶检测基准的 KITTI 和 nuImages 数据集验证了我们的方法。为了测试 AutoVDC 的有效性，我们创建了故意注入错误标注的数据集变体，并观察了我们方法的错误检测率。此外，我们比较了不同 VLMs 的检测率，并探讨了 VLM 微调对我们管道的影响。结果表明，我们的方法在错误检测和数据清洗实验中表现出高性能，表明其有可能极大地提高大规模生产数据集中自主驾驶的可靠性和准确性。 

---
# NOCTA: Non-Greedy Objective Cost-Tradeoff Acquisition for Longitudinal Data 

**Title (ZH)**: NOCTA: 非贪婪目标成本权衡获取方法用于纵向数据分析 

**Authors**: Dzung Dinh, Boqi Chen, Marc Niethammer, Junier Oliva  

**Link**: [PDF](https://arxiv.org/pdf/2507.12412)  

**Abstract**: In many critical applications, resource constraints limit the amount of information that can be gathered to make predictions. For example, in healthcare, patient data often spans diverse features ranging from lab tests to imaging studies. Each feature may carry different information and must be acquired at a respective cost of time, money, or risk to the patient. Moreover, temporal prediction tasks, where both instance features and labels evolve over time, introduce additional complexity in deciding when or what information is important. In this work, we propose NOCTA, a Non-Greedy Objective Cost-Tradeoff Acquisition method that sequentially acquires the most informative features at inference time while accounting for both temporal dynamics and acquisition cost. We first introduce a cohesive estimation target for our NOCTA setting, and then develop two complementary estimators: 1) a non-parametric method based on nearest neighbors to guide the acquisition (NOCTA-NP), and 2) a parametric method that directly predicts the utility of potential acquisitions (NOCTA-P). Experiments on synthetic and real-world medical datasets demonstrate that both NOCTA variants outperform existing baselines. 

**Abstract (ZH)**: 在许多关键应用中，资源限制限制了可用于预测的信息量。例如，在医疗保健领域，患者数据涵盖从实验室检查到影像研究等多样化的特征。每个特征可能携带不同的信息，并且必须在时间、金钱或对患者的风险方面付出相应的代价。此外，在时间预测任务中，实例特征和标签随时间演变，增加了决定何时或采集什么信息的重要性。在本文中，我们提出了一种非贪婪目标代价权衡采集方法NOCTA，在推理时序贯地采集最具信息性的特征，同时考虑时间动态和采集成本。我们首先为NOCTA设置引入了一个综合的估计目标，并开发了两个互补的估计器：1) 基于最近邻的非参数方法（NOCTA-NP），以引导采集；2) 直接预测潜在采集效用的参数方法（NOCTA-P）。在合成和真实世界医疗数据集上的实验表明，NOCTA的两种变体均优于现有基线方法。 

---
# Probing for Arithmetic Errors in Language Models 

**Title (ZH)**: 语言模型中的算术错误探究 

**Authors**: Yucheng Sun, Alessandro Stolfo, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12379)  

**Abstract**: We investigate whether internal activations in language models can be used to detect arithmetic errors. Starting with a controlled setting of 3-digit addition, we show that simple probes can accurately decode both the model's predicted output and the correct answer from hidden states, regardless of whether the model's output is correct. Building on this, we train lightweight error detectors that predict model correctness with over 90% accuracy. We then extend our analysis to structured chain-of-thought traces on addition-only GSM8K problems and find that probes trained on simple arithmetic generalize well to this more complex setting, revealing consistent internal representations. Finally, we demonstrate that these probes can guide selective re-prompting of erroneous reasoning steps, improving task accuracy with minimal disruption to correct outputs. Our findings suggest that arithmetic errors can be anticipated from internal activations alone, and that simple probes offer a viable path toward lightweight model self-correction. 

**Abstract (ZH)**: 我们研究语言模型内部激活是否可以用于检测算术错误。从三位数加法的受控设置开始，我们展示了简单的探针可以从隐藏状态准确解码模型预测输出和正确答案，不论模型的输出是否正确。在此基础上，我们训练了轻量级的错误检测器，其预测模型正确性的准确率超过90%。然后，我们将分析扩展到仅涉及加法的GSM8K结构化推理轨迹上，并发现针对简单算术训练的探针可以很好地泛化到这个更复杂的情景中，揭示了一致的内部表示。最后，我们证明这些探针可以指导对错误推理步骤的选择性重新提示，最小限度地干扰正确输出的同时提高任务准确性。我们的研究结果表明，仅从内部激活即可预测算术错误，并且简单的探针提供了一条可行的轻量级模型自我纠正途径。 

---
# GitChameleon: Evaluating AI Code Generation Against Python Library Version Incompatibilities 

**Title (ZH)**: GitChameleon: 评估AI代码生成对抗Python库版本不兼容性 

**Authors**: Diganta Misra, Nizar Islah, Victor May, Brice Rauby, Zihan Wang, Justine Gehring, Antonio Orvieto, Muawiz Chaudhary, Eilif B. Muller, Irina Rish, Samira Ebrahimi Kahou, Massimo Caccia  

**Link**: [PDF](https://arxiv.org/pdf/2507.12367)  

**Abstract**: The rapid evolution of software libraries poses a considerable hurdle for code generation, necessitating continuous adaptation to frequent version updates while preserving backward compatibility. While existing code evolution benchmarks provide valuable insights, they typically lack execution-based evaluation for generating code compliant with specific library versions. To address this, we introduce GitChameleon, a novel, meticulously curated dataset comprising 328 Python code completion problems, each conditioned on specific library versions and accompanied by executable unit tests. GitChameleon rigorously evaluates the capacity of contemporary large language models (LLMs), LLM-powered agents, code assistants, and RAG systems to perform version-conditioned code generation that demonstrates functional accuracy through execution. Our extensive evaluations indicate that state-of-the-art systems encounter significant challenges with this task; enterprise models achieving baseline success rates in the 48-51\% range, underscoring the intricacy of the problem. By offering an execution-based benchmark emphasizing the dynamic nature of code libraries, GitChameleon enables a clearer understanding of this challenge and helps guide the development of more adaptable and dependable AI code generation methods. We make the dataset and evaluation code publicly available at this https URL. 

**Abstract (ZH)**: 软件库的快速演进对代码生成构成了重大挑战，需要不断适应频繁的版本更新并保留向后兼容性。虽然现有的代码演化基准提供了有价值的见解，但它们通常缺乏基于执行的评估，以生成符合特定库版本的代码。为解决这一问题，我们介绍了GitChameleon，这是一个精心策划的数据集，包含328个Python代码补全问题，每个问题都针对特定的库版本，并附带可执行的单元测试。GitChameleon严格评估了当代大型语言模型（LLMs）、LLM驱动的代理、代码助手和RAG系统在执行条件下生成符合版本要求的代码的能力，以展示功能准确性。我们的广泛评估表明，最先进的系统在此任务中面临重大挑战；企业模型在基准成功率达到48-51%的范围内，突显了该问题的复杂性。通过提供一个基于执行的基准强调代码库的动态性，GitChameleon有助于更清晰地了解这一挑战，并帮助指导开发更灵活可靠的AI代码生成方法。我们已在以下链接公开发布了数据集和评估代码：this https URL。 

---
# FactorHD: A Hyperdimensional Computing Model for Multi-Object Multi-Class Representation and Factorization 

**Title (ZH)**: FactorHD：一种用于多对象多类表示与因子化的超维度计算模型 

**Authors**: Yifei Zhou, Xuchu Huang, Chenyu Ni, Min Zhou, Zheyu Yan, Xunzhao Yin, Cheng Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2507.12366)  

**Abstract**: Neuro-symbolic artificial intelligence (neuro-symbolic AI) excels in logical analysis and reasoning. Hyperdimensional Computing (HDC), a promising brain-inspired computational model, is integral to neuro-symbolic AI. Various HDC models have been proposed to represent class-instance and class-class relations, but when representing the more complex class-subclass relation, where multiple objects associate different levels of classes and subclasses, they face challenges for factorization, a crucial task for neuro-symbolic AI systems. In this article, we propose FactorHD, a novel HDC model capable of representing and factorizing the complex class-subclass relation efficiently. FactorHD features a symbolic encoding method that embeds an extra memorization clause, preserving more information for multiple objects. In addition, it employs an efficient factorization algorithm that selectively eliminates redundant classes by identifying the memorization clause of the target class. Such model significantly enhances computing efficiency and accuracy in representing and factorizing multiple objects with class-subclass relation, overcoming limitations of existing HDC models such as "superposition catastrophe" and "the problem of 2". Evaluations show that FactorHD achieves approximately 5667x speedup at a representation size of 10^9 compared to existing HDC models. When integrated with the ResNet-18 neural network, FactorHD achieves 92.48% factorization accuracy on the Cifar-10 dataset. 

**Abstract (ZH)**: 神经符号人工智能（神经符号AI）在逻辑分析和推理方面表现出色。基于超维度计算（Hyperdimensional Computing，HDC）的脑启发计算模型是神经符号AI的重要组成部分。各种HDC模型已被提出用于表示类-实例关系和类-类关系，但在表示更复杂的类-子类关系时，即多个对象与不同层级的类和子类关联时，它们在分解这一关键任务上面临挑战。本文提出了一种名为FactorHD的新型HDC模型，能够有效地表示和分解复杂的类-子类关系。FactorHD采用符号编码方法，嵌入额外的存储条款，保留了更多关于多个对象的信息。此外，该模型采用了一种高效的分解算法，通过识别目标类的存储条款有选择地消除冗余类。该模型在表示和分解具有类-子类关系的多个对象时显著提高了计算效率和准确性，克服了现有HDC模型诸如“叠加灾难”和“2问题”等限制。评价结果显示，FactorHD在表示规模为10^9的情况下相比于现有HDC模型实现了约5667倍的速度提升。当与ResNet-18神经网络结合使用时，FactorHD在Cifar-10数据集上实现了92.48%的分解准确率。 

---
# Cluster Contrast for Unsupervised Visual Representation Learning 

**Title (ZH)**: 无监督视觉表示学习的聚类对比方法 

**Authors**: Nikolaos Giakoumoglou, Tania Stathaki  

**Link**: [PDF](https://arxiv.org/pdf/2507.12359)  

**Abstract**: We introduce Cluster Contrast (CueCo), a novel approach to unsupervised visual representation learning that effectively combines the strengths of contrastive learning and clustering methods. Inspired by recent advancements, CueCo is designed to simultaneously scatter and align feature representations within the feature space. This method utilizes two neural networks, a query and a key, where the key network is updated through a slow-moving average of the query outputs. CueCo employs a contrastive loss to push dissimilar features apart, enhancing inter-class separation, and a clustering objective to pull together features of the same cluster, promoting intra-class compactness. Our method achieves 91.40% top-1 classification accuracy on CIFAR-10, 68.56% on CIFAR-100, and 78.65% on ImageNet-100 using linear evaluation with a ResNet-18 backbone. By integrating contrastive learning with clustering, CueCo sets a new direction for advancing unsupervised visual representation learning. 

**Abstract (ZH)**: 我们介绍了簇对比（CueCo），这是一种新颖的无监督视觉表示学习方法，有效地结合了对比学习和聚类方法的优点。受 recent 进展的启发，CueCo 设计用于同时在特征空间中分散和对齐特征表示。该方法使用两个神经网络——查询和键，其中键网络通过查询输出的慢移动平均值进行更新。CueCo 使用对比损失来推动不相似的特征分开，增强类间分离，使用聚类目标将相同簇的特征拉近，促进类内紧凑性。我们的方法在使用 ResNet-18 作为骨干网络的线性评估中，CIFAR-10 的 top-1 分类准确率为 91.40%，CIFAR-100 为 68.56%，ImageNet-100 为 78.65%。通过将对比学习与聚类相结合，CueCo 为推进无监督视觉表示学习设定了新的方向。 

---
# Neural Polar Decoders for Deletion Channels 

**Title (ZH)**: 神经极化解码器用于删除信道 

**Authors**: Ziv Aharoni, Henry D. Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2507.12329)  

**Abstract**: This paper introduces a neural polar decoder (NPD) for deletion channels with a constant deletion rate. Existing polar decoders for deletion channels exhibit high computational complexity of $O(N^4)$, where $N$ is the block length. This limits the application of polar codes for deletion channels to short-to-moderate block lengths. In this work, we demonstrate that employing NPDs for deletion channels can reduce the computational complexity. First, we extend the architecture of the NPD to support deletion channels. Specifically, the NPD architecture consists of four neural networks (NNs), each replicating fundamental successive cancellation (SC) decoder operations. To support deletion channels, we change the architecture of only one. The computational complexity of the NPD is $O(AN\log N)$, where the parameter $A$ represents a computational budget determined by the user and is independent of the channel. We evaluate the new extended NPD for deletion channels with deletion rates $\delta\in\{0.01, 0.1\}$ and we verify the NPD with the ground truth given by the trellis decoder by Tal et al. We further show that due to the reduced complexity of the NPD, we are able to incorporate list decoding and further improve performance. We believe that the extended NPD presented here could have applications in future technologies like DNA storage. 

**Abstract (ZH)**: 一种用于恒定删除率信道的神经极化解码器 

---
# Compositional Discrete Latent Code for High Fidelity, Productive Diffusion Models 

**Title (ZH)**: 面向高保真度和高效生产扩散模型的组合离散潜码 

**Authors**: Samuel Lavoie, Michael Noukhovitch, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2507.12318)  

**Abstract**: We argue that diffusion models' success in modeling complex distributions is, for the most part, coming from their input conditioning. This paper investigates the representation used to condition diffusion models from the perspective that ideal representations should improve sample fidelity, be easy to generate, and be compositional to allow out-of-training samples generation. We introduce Discrete Latent Code (DLC), an image representation derived from Simplicial Embeddings trained with a self-supervised learning objective. DLCs are sequences of discrete tokens, as opposed to the standard continuous image embeddings. They are easy to generate and their compositionality enables sampling of novel images beyond the training distribution. Diffusion models trained with DLCs have improved generation fidelity, establishing a new state-of-the-art for unconditional image generation on ImageNet. Additionally, we show that composing DLCs allows the image generator to produce out-of-distribution samples that coherently combine the semantics of images in diverse ways. Finally, we showcase how DLCs can enable text-to-image generation by leveraging large-scale pretrained language models. We efficiently finetune a text diffusion language model to generate DLCs that produce novel samples outside of the image generator training distribution. 

**Abstract (ZH)**: 我们argue rằng diffusion模型在建模复杂分布方面的成功主要归因于它们的输入条件。本文从理想表示应提高样本保真度、易于生成且具有可组合性以生成训练分布外样本的角度，探讨了用于条件化diffusion模型的表示方法。我们引入了离散潜码（DLC），这是一种源自Simplicial Embeddings并通过自我监督学习目标训练得到的图像表示方法。DLC是离散标记的序列，相比于标准的连续图像嵌入，它们更易于生成且其可组合性使得能够生成训练分布外的新颖图像。使用DLC训练的diffusion模型在图像生成保真度方面有所提升，建立了在ImageNet上无条件图像生成的新state-of-the-art。此外，我们展示通过组合DLC可以使图像生成器产生符合内部一致性的跨类别图像样本。最后，我们展示了DLC如何通过利用大规模预训练语言模型实现图文生成。我们高效地微调了一种文本diffusion语言模型以生成DLC，从而产生图像生成器训练分布外的新颖样本。 

---
# Thought Purity: Defense Paradigm For Chain-of-Thought Attack 

**Title (ZH)**: 思维纯净：针对思维链攻击的防御范式 

**Authors**: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2507.12314)  

**Abstract**: While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense paradigm that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures. 

**Abstract (ZH)**: 虽然通过强化学习训练的大推理模型（LRMs，例如Deepseek-R1）在大型语言模型（LLMs）领域展示了先进的推理能力，但其对安全威胁的易感性仍然是一个关键漏洞。这一弱点在链式推理（CoT）生成过程中尤为明显，敌对方法如后门提示攻击可以系统地颠覆模型的核心推理机制。新兴的链式推理攻击（CoTA）通过利用提示可控性，以低成本的干预同时降低CoT的安全性和任务性能。为了应对这种复合的安全-性能漏洞，我们提出了一种防伪（TP）防护范式，该范式系统地增强了对恶意内容的抵抗力，同时保持操作有效性。我们的解决方案通过三个协同组件实现这一目标：（1）安全优化的数据处理管道（2）强化学习增强的规则约束（3）自适应监控指标。我们的方法建立了针对强化学习对齐的推理系统中CoTA漏洞的第一个全面防御机制，显著提升了下一代AI架构的安全-功能性平衡。 

---
# Chain-of-Descriptions: Improving Code LLMs for VHDL Code Generation and Summarization 

**Title (ZH)**: 描述链：提高VHDL代码生成和总结的代码LLM性能 

**Authors**: Prashanth Vijayaraghavan, Apoorva Nitsure, Charles Mackin, Luyao Shi, Stefano Ambrogio, Arvind Haran, Viresh Paruthi, Ali Elzein, Dan Coops, David Beymer, Tyler Baldwin, Ehsan Degan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12308)  

**Abstract**: Large Language Models (LLMs) have become widely used across diverse NLP tasks and domains, demonstrating their adaptability and effectiveness. In the realm of Electronic Design Automation (EDA), LLMs show promise for tasks like Register-Transfer Level (RTL) code generation and summarization. However, despite the proliferation of LLMs for general code-related tasks, there's a dearth of research focused on evaluating and refining these models for hardware description languages (HDLs), notably VHDL. In this study, we evaluate the performance of existing code LLMs for VHDL code generation and summarization using various metrics and two datasets -- VHDL-Eval and VHDL-Xform. The latter, an in-house dataset, aims to gauge LLMs' understanding of functionally equivalent code. Our findings reveal consistent underperformance of these models across different metrics, underscoring a significant gap in their suitability for this domain. To address this challenge, we propose Chain-of-Descriptions (CoDes), a novel approach to enhance the performance of LLMs for VHDL code generation and summarization tasks. CoDes involves generating a series of intermediate descriptive steps based on: (i) the problem statement for code generation, and (ii) the VHDL code for summarization. These steps are then integrated with the original input prompt (problem statement or code) and provided as input to the LLMs to generate the final output. Our experiments demonstrate that the CoDes approach significantly surpasses the standard prompting strategy across various metrics on both datasets. This method not only improves the quality of VHDL code generation and summarization but also serves as a framework for future research aimed at enhancing code LLMs for VHDL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多样化的自然语言处理任务和领域中得到了广泛应用，展示了其适应性和有效性。在电子设计自动化（EDA）领域，LLMs在寄存器传输级（RTL）代码生成和摘要等任务中展现出前景。然而，尽管LLMs在通用代码相关任务中有广泛的应用，但对于硬件描述语言（HDLs），特别是VHDL，的研究鲜有涉及，尤其在评估和优化这些模型方面。在本研究中，我们使用VHDL-Eval和VHDL-Xform两个数据集评估现有代码LLMs在VHDL代码生成和摘要任务中的表现，后者是一个内部数据集，旨在评估LLMs对功能等效代码的理解能力。研究发现，这些模型在不同指标下的表现不尽如人意，揭示了它们在该领域适用性存在显著差距。为解决这一挑战，我们提出了一种新颖的方法—描述链（CoDes）来提升LLMs在VHDL代码生成和摘要任务中的性能。CoDes涉及基于代码生成的问题陈述和用于总结的VHDL代码生成一系列中间描述步骤，这些步骤随后与原始输入提示（问题陈述或代码）结合，作为输入提供给LLMs以生成最终输出。实验结果表明，CoDes方法在两个数据集的各种指标上显著超越了标准提示策略。这种方法不仅提高了VHDL代码生成和摘要的质量，也为未来旨在增强VHDL代码LLMs的研究提供了框架。 

---
# PROL : Rehearsal Free Continual Learning in Streaming Data via Prompt Online Learning 

**Title (ZH)**: PROL：基于提示在线学习的无回放流式数据连续学习 

**Authors**: M. Anwar Ma'sum, Mahardhika Pratama, Savitha Ramasamy, Lin Liu, Habibullah Habibullah, Ryszard Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.12305)  

**Abstract**: The data privacy constraint in online continual learning (OCL), where the data can be seen only once, complicates the catastrophic forgetting problem in streaming data. A common approach applied by the current SOTAs in OCL is with the use of memory saving exemplars or features from previous classes to be replayed in the current task. On the other hand, the prompt-based approach performs excellently in continual learning but with the cost of a growing number of trainable parameters. The first approach may not be applicable in practice due to data openness policy, while the second approach has the issue of throughput associated with the streaming data. In this study, we propose a novel prompt-based method for online continual learning that includes 4 main components: (1) single light-weight prompt generator as a general knowledge, (2) trainable scaler-and-shifter as specific knowledge, (3) pre-trained model (PTM) generalization preserving, and (4) hard-soft updates mechanism. Our proposed method achieves significantly higher performance than the current SOTAs in CIFAR100, ImageNet-R, ImageNet-A, and CUB dataset. Our complexity analysis shows that our method requires a relatively smaller number of parameters and achieves moderate training time, inference time, and throughput. For further study, the source code of our method is available at this https URL. 

**Abstract (ZH)**: 在线持续学习中的数据隐私约束 complicates 灾难性遗忘问题，在此类学习中数据仅可见一次。当前的SOTA方法通常使用节省内存的先前类别的示例或特征在当前任务中重新播放。另一方面，基于提示的方法在持续学习中表现出色，但代价是可训练参数数量的增长。由于数据开放政策，第一种方法可能在实践中不可行，而第二种方法与流式数据相关联的吞吐量问题。在这项研究中，我们提出了一种新的基于提示的在线持续学习方法，包含四个主要组成部分：（1）单个轻量级提示生成器作为通用知识，（2）可训练的比例移位器作为特定知识，（3）预训练模型泛化的保持，（4）硬软更新机制。我们提出的方法在CIFAR100、ImageNet-R、ImageNet-A和CUB数据集上的性能显著优于当前的SOTA方法。我们的时间复杂性分析表明，我们的方法需要相对较少的参数，并实现中等程度的训练时间、推理时间和吞吐量。对于我们提出的方法的源代码，可以在以下网址获取。 

---
# Text-ADBench: Text Anomaly Detection Benchmark based on LLMs Embedding 

**Title (ZH)**: Text-ADBench: 基于LLMs嵌入的文本异常检测基准 

**Authors**: Feng Xiao, Jicong Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12295)  

**Abstract**: Text anomaly detection is a critical task in natural language processing (NLP), with applications spanning fraud detection, misinformation identification, spam detection and content moderation, etc. Despite significant advances in large language models (LLMs) and anomaly detection algorithms, the absence of standardized and comprehensive benchmarks for evaluating the existing anomaly detection methods on text data limits rigorous comparison and development of innovative approaches. This work performs a comprehensive empirical study and introduces a benchmark for text anomaly detection, leveraging embeddings from diverse pre-trained language models across a wide array of text datasets. Our work systematically evaluates the effectiveness of embedding-based text anomaly detection by incorporating (1) early language models (GloVe, BERT); (2) multiple LLMs (LLaMa-2, LLama-3, Mistral, OpenAI (small, ada, large)); (3) multi-domain text datasets (news, social media, scientific publications); (4) comprehensive evaluation metrics (AUROC, AUPRC). Our experiments reveal a critical empirical insight: embedding quality significantly governs anomaly detection efficacy, and deep learning-based approaches demonstrate no performance advantage over conventional shallow algorithms (e.g., KNN, Isolation Forest) when leveraging LLM-derived this http URL addition, we observe strongly low-rank characteristics in cross-model performance matrices, which enables an efficient strategy for rapid model evaluation (or embedding evaluation) and selection in practical applications. Furthermore, by open-sourcing our benchmark toolkit that includes all embeddings from different models and code at this https URL, this work provides a foundation for future research in robust and scalable text anomaly detection systems. 

**Abstract (ZH)**: 文本异常检测是自然语言处理（NLP）中一个关键任务，应用于欺诈检测、虚假信息识别、垃圾信息检测和内容审核等领域。尽管在大型语言模型（LLMs）和异常检测算法方面取得了显著进展，但由于缺乏标准化和综合的基准来评估现有文本数据异常检测方法，限制了严格比较和创新方法的发展。本研究进行了全面的实证研究并引入了一个文本异常检测基准，利用了多种预训练语言模型在广泛文本数据集上的嵌入。我们的研究系统性地通过结合（1）早期语言模型（GloVe，BERT）；（2）多个LLM（LLaMa-2，LLaMa-3，Mistral，OpenAI（小、ada、大））；（3）多领域文本数据集（新闻、社交媒体、科学出版物）；（4）综合评价指标（AUROC，AUPRC）评估基于嵌入的文本异常检测的有效性。实验揭示了一个关键的经验见解：嵌入质量显著影响异常检测效果，并且基于深度学习的方法在利用LLM提取嵌入时不比传统的浅层算法（如KNN，孤立森林）表现出性能优势。此外，我们观察到了跨模型性能矩阵的强低秩特性，这促进了在实际应用中高效策略的快速模型（或嵌入）评估和选择。通过开源包含不同模型所有嵌入和代码的基准工具包（见此https URL），本研究为鲁棒性和可扩展的文本异常检测系统未来研究奠定了基础。 

---
# MERA Code: A Unified Framework for Evaluating Code Generation Across Tasks 

**Title (ZH)**: MERA代码：跨任务评估代码生成的统一框架 

**Authors**: Artem Chervyakov, Alexander Kharitonov, Pavel Zadorozhny, Adamenko Pavel, Rodion Levichev, Dmitrii Vorobev, Dmitrii Salikhov, Aidar Valeev, Alena Pestova, Maria Dziuba, Ilseyar Alimova, Artem Zavgorodnev, Aleksandr Medvedev, Stanislav Moiseev, Elena Bruches, Daniil Grebenkin, Roman Derunets, Vikulov Vladimir, Anton Emelyanov, Dmitrii Babaev, Vladimir V. Ivanov, Valentin Malykh, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2507.12284)  

**Abstract**: Advancements in LLMs have enhanced task automation in software engineering; however, current evaluations primarily focus on natural language tasks, overlooking code quality. Most benchmarks prioritize high-level reasoning over executable code and real-world performance, leaving gaps in understanding true capabilities and risks associated with these models in production. To address this issue, we propose MERA Code, a new addition to the MERA benchmark family, specifically focused on evaluating code for the latest code generation LLMs in Russian. This benchmark includes 11 evaluation tasks that span 8 programming languages. Our proposed evaluation methodology features a taxonomy that outlines the practical coding skills necessary for models to complete these tasks. The benchmark comprises an open-source codebase for users to conduct MERA assessments, a scoring system compatible with various programming environments, and a platform featuring a leaderboard and submission system. We evaluate open LLMs and frontier API models, analyzing their limitations in terms of practical coding tasks in non-English languages. We are publicly releasing MERA to guide future research, anticipate groundbreaking features in model development, and standardize evaluation procedures. 

**Abstract (ZH)**: LLMs的进步增强了软件工程中的任务自动化，但当前评估主要集中在自然语言任务上，忽视了代码质量。大多数基准测试侧重于高级推理而非可执行代码和实际性能，留下了对这些模型在生产环境中真正能力和潜在风险的理解缺口。为解决这个问题，我们提出了MERA Code，这是MERA基准家族的一个新成员，特别关注评估俄语中最新代码生成LLMs的代码质量。该基准测试包括11项评估任务，覆盖8种编程语言。我们提出的评估方法论包含了一个分类体系，概述了模型完成这些任务所需的实用编码技能。该基准测试包含一个开源代码库，供用户进行MERA评估，一个与各种编程环境兼容的评分系统，以及一个包含排行榜和提交系统的平台。我们评估了开源LLMs和前沿API模型，分析了它们在非英语语言中的实用编程任务中的局限性。我们正公开发布MERA，以指导未来的研究，预见模型开发中的突破性功能，并标准化评估程序。 

---
# Site-Level Fine-Tuning with Progressive Layer Freezing: Towards Robust Prediction of Bronchopulmonary Dysplasia from Day-1 Chest Radiographs in Extremely Preterm Infants 

**Title (ZH)**: 基于站点的逐层Fine-Tuning与渐进层冻结：面向极早早产儿Day-1胸部X光片的Bronchopulmonary Dysplasia稳健预测 

**Authors**: Sybelle Goedicke-Fritz, Michelle Bous, Annika Engel, Matthias Flotho, Pascal Hirsch, Hannah Wittig, Dino Milanovic, Dominik Mohr, Mathias Kaspar, Sogand Nemat, Dorothea Kerner, Arno Bücker, Andreas Keller, Sascha Meyer, Michael Zemlin, Philipp Flotho  

**Link**: [PDF](https://arxiv.org/pdf/2507.12269)  

**Abstract**: Bronchopulmonary dysplasia (BPD) is a chronic lung disease affecting 35% of extremely low birth weight infants. Defined by oxygen dependence at 36 weeks postmenstrual age, it causes lifelong respiratory complications. However, preventive interventions carry severe risks, including neurodevelopmental impairment, ventilator-induced lung injury, and systemic complications. Therefore, early BPD prognosis and prediction of BPD outcome is crucial to avoid unnecessary toxicity in low risk infants. Admission radiographs of extremely preterm infants are routinely acquired within 24h of life and could serve as a non-invasive prognostic tool. In this work, we developed and investigated a deep learning approach using chest X-rays from 163 extremely low-birth-weight infants ($\leq$32 weeks gestation, 401-999g) obtained within 24 hours of birth. We fine-tuned a ResNet-50 pretrained specifically on adult chest radiographs, employing progressive layer freezing with discriminative learning rates to prevent overfitting and evaluated a CutMix augmentation and linear probing. For moderate/severe BPD outcome prediction, our best performing model with progressive freezing, linear probing and CutMix achieved an AUROC of 0.78 $\pm$ 0.10, balanced accuracy of 0.69 $\pm$ 0.10, and an F1-score of 0.67 $\pm$ 0.11. In-domain pre-training significantly outperformed ImageNet initialization (p = 0.031) which confirms domain-specific pretraining to be important for BPD outcome prediction. Routine IRDS grades showed limited prognostic value (AUROC 0.57 $\pm$ 0.11), confirming the need of learned markers. Our approach demonstrates that domain-specific pretraining enables accurate BPD prediction from routine day-1 radiographs. Through progressive freezing and linear probing, the method remains computationally feasible for site-level implementation and future federated learning deployments. 

**Abstract (ZH)**: 极低出生体重儿肺部发育不良(BPD)的胸部X射线早期预测：基于领域特定预训练的深度学习方法 

---
# A Framework for Nonstationary Gaussian Processes with Neural Network Parameters 

**Title (ZH)**: 非平稳高斯过程的神经网络参数框架 

**Authors**: Zachary James, Joseph Guinness  

**Link**: [PDF](https://arxiv.org/pdf/2507.12262)  

**Abstract**: Gaussian processes have become a popular tool for nonparametric regression because of their flexibility and uncertainty quantification. However, they often use stationary kernels, which limit the expressiveness of the model and may be unsuitable for many datasets. We propose a framework that uses nonstationary kernels whose parameters vary across the feature space, modeling these parameters as the output of a neural network that takes the features as input. The neural network and Gaussian process are trained jointly using the chain rule to calculate derivatives. Our method clearly describes the behavior of the nonstationary parameters and is compatible with approximation methods for scaling to large datasets. It is flexible and easily adapts to different nonstationary kernels without needing to redesign the optimization procedure. Our methods are implemented with the GPyTorch library and can be readily modified. We test a nonstationary variance and noise variant of our method on several machine learning datasets and find that it achieves better accuracy and log-score than both a stationary model and a hierarchical model approximated with variational inference. Similar results are observed for a model with only nonstationary variance. We also demonstrate our approach's ability to recover the nonstationary parameters of a spatial dataset. 

**Abstract (ZH)**: Gaussian过程已成为非参数回归的一个流行工具，这是因为它们的灵活性和不确定性量化。然而，它们通常使用stationary内核，这限制了模型的表达能力，并可能不适合许多数据集。我们提出了一种框架，该框架使用非stationary内核，其参数在特征空间中变化，并将这些参数建模为神经网络的输出，该神经网络将特征作为输入。神经网络和Gaussian过程通过链式规则计算导数进行联合训练。我们的方法清晰地描述了非stationary参数的行为，并且与用于处理大规模数据集的近似方法兼容。该方法具有灵活性，无需重新设计优化程序即可轻松适应不同类型的非stationary内核。我们使用GPyTorch库实现这些方法，并可以方便地进行修改。我们对几个机器学习数据集测试了我们的非stationary方差和噪声变体方法，发现它在准确性和log-分数方面优于stationary模型和用变分推断近似的分层模型。具有仅非stationary方差的模型也观察到了类似的成果。我们还展示了我们的方法恢复空间数据集中非stationary参数的能力。 

---
# Infherno: End-to-end Agent-based FHIR Resource Synthesis from Free-form Clinical Notes 

**Title (ZH)**: Inferno: 基于代理的端到端FHIR资源合成从非结构化临床笔记 

**Authors**: Johann Frei, Nils Feldhus, Lisa Raithel, Roland Roller, Alexander Meyer, Frank Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2507.12261)  

**Abstract**: For clinical data integration and healthcare services, the HL7 FHIR standard has established itself as a desirable format for interoperability between complex health data. Previous attempts at automating the translation from free-form clinical notes into structured FHIR resources rely on modular, rule-based systems or LLMs with instruction tuning and constrained decoding. Since they frequently suffer from limited generalizability and structural inconformity, we propose an end-to-end framework powered by LLM agents, code execution, and healthcare terminology database tools to address these issues. Our solution, called Infherno, is designed to adhere to the FHIR document schema and competes well with a human baseline in predicting FHIR resources from unstructured text. The implementation features a front end for custom and synthetic data and both local and proprietary models, supporting clinical data integration processes and interoperability across institutions. 

**Abstract (ZH)**: 基于HL7 FHIR标准的临床数据集成与 healthcare 服务自动化转换框架：Infherno 

---
# Improving Contextual ASR via Multi-grained Fusion with Large Language Models 

**Title (ZH)**: 通过大型语言模型实现多粒度融合以改进上下文ASR 

**Authors**: Shilin Zhou, Zhenghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.12252)  

**Abstract**: While end-to-end Automatic Speech Recognition (ASR) models have shown impressive performance in transcribing general speech, they often struggle to accurately recognize contextually relevant keywords, such as proper nouns or user-specific entities.
Previous approaches have explored leveraging keyword dictionaries in the textual modality to improve keyword recognition, either through token-level fusion that guides token-by-token generation or phrase-level fusion that enables direct copying of keyword phrases.
However, these methods operate at different granularities and have their own limitations.
In this paper, we propose a novel multi-grained fusion approach that jointly leverages the strengths of both token-level and phrase-level fusion with Large Language Models (LLMs).
Our approach incorporates a late-fusion strategy that elegantly combines ASR's acoustic information with LLM's rich contextual knowledge, balancing fine-grained token precision with holistic phrase-level understanding.
Experiments on Chinese and English datasets demonstrate that our approach achieves state-of-the-art performance on keyword-related metrics while preserving high accuracy on non-keyword text.
Ablation studies further confirm that the token-level and phrase-level components both contribute significantly to the performance gains, complementing each other in our joint multi-grained framework.
The code and models will be publicly available at this https URL. 

**Abstract (ZH)**: 虽然端到端自动语音识别（ASR）模型在转录通用语音方面取得了令人印象深刻的性能，但在准确识别上下文相关的关键词（如专有名词或用户特定实体）方面往往表现不佳。 

---
# Looking for Fairness in Recommender Systems 

**Title (ZH)**: 在推荐系统中寻找公平性 

**Authors**: Cécile Logé  

**Link**: [PDF](https://arxiv.org/pdf/2507.12242)  

**Abstract**: Recommender systems can be found everywhere today, shaping our everyday experience whenever we're consuming content, ordering food, buying groceries online, or even just reading the news. Let's imagine we're in the process of building a recommender system to make content suggestions to users on social media. When thinking about fairness, it becomes clear there are several perspectives to consider: the users asking for tailored suggestions, the content creators hoping for some limelight, and society at large, navigating the repercussions of algorithmic recommendations. A shared fairness concern across all three is the emergence of filter bubbles, a side-effect that takes place when recommender systems are almost "too good", making recommendations so tailored that users become inadvertently confined to a narrow set of opinions/themes and isolated from alternative ideas. From the user's perspective, this is akin to manipulation. From the small content creator's perspective, this is an obstacle preventing them access to a whole range of potential fans. From society's perspective, the potential consequences are far-reaching, influencing collective opinions, social behavior and political decisions. How can our recommender system be fine-tuned to avoid the creation of filter bubbles, and ensure a more inclusive and diverse content landscape? Approaching this problem involves defining one (or more) performance metric to represent diversity, and tweaking our recommender system's performance through the lens of fairness. By incorporating this metric into our evaluation framework, we aim to strike a balance between personalized recommendations and the broader societal goal of fostering rich and varied cultures and points of view. 

**Abstract (ZH)**: 推荐系统无处不在：构建社交媒体内容推荐系统中的公平性考量与实现 

---
# Draw an Ugly Person An Exploration of Generative AIs Perceptions of Ugliness 

**Title (ZH)**: 画一个丑陋的人：探究生成式AI对丑陋性的感知 

**Authors**: Garyoung Kim, Huisung Kwon, Seoju Yun, Yu-Won Youn  

**Link**: [PDF](https://arxiv.org/pdf/2507.12212)  

**Abstract**: Generative AI does not only replicate human creativity but also reproduces deep-seated cultural biases, making it crucial to critically examine how concepts like ugliness are understood and expressed by these tools. This study investigates how four different generative AI models understand and express ugliness through text and image and explores the biases embedded within these representations. We extracted 13 adjectives associated with ugliness through iterative prompting of a large language model and generated 624 images across four AI models and three prompts. Demographic and socioeconomic attributes within the images were independently coded and thematically analyzed. Our findings show that AI models disproportionately associate ugliness with old white male figures, reflecting entrenched social biases as well as paradoxical biases, where efforts to avoid stereotypical depictions of marginalized groups inadvertently result in the disproportionate projection of negative attributes onto majority groups. Qualitative analysis further reveals that, despite supposed attempts to frame ugliness within social contexts, conventional physical markers such as asymmetry and aging persist as central visual motifs. These findings demonstrate that despite attempts to create more equal representations, generative AI continues to perpetuate inherited and paradoxical biases, underscoring the critical work being done to create ethical AI training paradigms and advance methodologies for more inclusive AI development. 

**Abstract (ZH)**: 生成式AI不仅复制人类的创造性，还重现了深层的文化偏见，因此批判性地审视这些工具如何理解和表达丑陋的概念变得至关重要。本研究探讨了四种不同的生成式AI模型如何通过文字和图像来理解和表达丑陋，并探索这些表征中嵌入的偏见。我们通过迭代提示大型语言模型提取了与丑陋相关的13个形容词，并生成了四种AI模型、三种提示下的624张图像。图像中的人口统计学和社会经济属性被独立编码并进行了主题分析。我们的研究发现，AI模型不成比例地将丑陋与老年白人男性形象联系起来，反映了根深蒂固的社会偏见以及矛盾的偏见，即避免刻板描绘边缘群体反而无意中将负面属性过度投射到主导群体上。定性分析进一步表明，尽管有试图将丑陋置于社会背景之中的努力，但非对称性和老化等传统物理特征仍然作为主要的视觉主题持续存在。这些发现表明，尽管试图创造更平等的表征，生成式AI仍在继续传播继承下来的和矛盾的偏见，强调了创建伦理AI训练框架和推进更包容的AI开发方法论的必要性。 

---
# Sparse Autoencoders for Sequential Recommendation Models: Interpretation and Flexible Control 

**Title (ZH)**: 稀疏自编码器在序列推荐模型中的应用：解释与灵活控制 

**Authors**: Anton Klenitskiy, Konstantin Polev, Daria Denisova, Alexey Vasilev, Dmitry Simakov, Gleb Gusev  

**Link**: [PDF](https://arxiv.org/pdf/2507.12202)  

**Abstract**: Many current state-of-the-art models for sequential recommendations are based on transformer architectures. Interpretation and explanation of such black box models is an important research question, as a better understanding of their internals can help understand, influence, and control their behavior, which is very important in a variety of real-world applications. Recently sparse autoencoders (SAE) have been shown to be a promising unsupervised approach for extracting interpretable features from language models. These autoencoders learn to reconstruct hidden states of the transformer's internal layers from sparse linear combinations of directions in their activation space.
This paper is focused on the application of SAE to the sequential recommendation domain. We show that this approach can be successfully applied to the transformer trained on a sequential recommendation task: learned directions turn out to be more interpretable and monosemantic than the original hidden state dimensions. Moreover, we demonstrate that the features learned by SAE can be used to effectively and flexibly control the model's behavior, providing end-users with a straightforward method to adjust their recommendations to different custom scenarios and contexts. 

**Abstract (ZH)**: 基于稀疏自编码器的顺序推荐领域应用研究 

---
# Quantize More, Lose Less: Autoregressive Generation from Residually Quantized Speech Representations 

**Title (ZH)**: 量化更多，损失更少：基于残余量化语音表示的自回归生成 

**Authors**: Yichen Han, Xiaoyang Hao, Keming Chen, Weibo Xiong, Jun He, Ruonan Zhang, Junjie Cao, Yue Liu, Bowen Li, Dongrui Zhang, Hui Xia, Huilei Fu, Kai Jia, Kaixuan Guo, Mingli Jin, Qingyun Meng, Ruidong Ma, Ruiqian Fang, Shaotong Guo, Xuhui Li, Yang Xiang, Ying Zhang, Yulong Liu, Yunfeng Li, Yuyi Zhang, Yuze Zhou, Zhen Wang, Zhaowen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.12197)  

**Abstract**: Text-to-speech (TTS) synthesis has seen renewed progress under the discrete modeling paradigm. Existing autoregressive approaches often rely on single-codebook representations, which suffer from significant information loss. Even with post-hoc refinement techniques such as flow matching, these methods fail to recover fine-grained details (e.g., prosodic nuances, speaker-specific timbres), especially in challenging scenarios like singing voice or music synthesis. We propose QTTS, a novel TTS framework built upon our new audio codec, QDAC. The core innovation of QDAC lies in its end-to-end training of an ASR-based auto-regressive network with a GAN, which achieves superior semantic feature disentanglement for scalable, near-lossless compression. QTTS models these discrete codes using two innovative strategies: the Hierarchical Parallel architecture, which uses a dual-AR structure to model inter-codebook dependencies for higher-quality synthesis, and the Delay Multihead approach, which employs parallelized prediction with a fixed delay to accelerate inference speed. Our experiments demonstrate that the proposed framework achieves higher synthesis quality and better preserves expressive content compared to baseline. This suggests that scaling up compression via multi-codebook modeling is a promising direction for high-fidelity, general-purpose speech and audio generation. 

**Abstract (ZH)**: 基于离散建模范式的文本到speech合成已取得新的进展。现有自回归方法通常依赖单码本表示，存在显著信息丢失的问题。即使采用流匹配等后处理技术，这些方法在如歌声合成或音乐合成等挑战场景下，也无法恢复细微的语音特征（如韵律细节、特定讲话者音色），我们提出了QTTS，一种基于我们新音频编解码器QDAC的新型文本到speech框架。QDAC的核心创新在于将基于ASR的自回归网络与GAN结合进行端到端训练，从而实现更优的语义特征分离，以实现可扩展且近乎无损的压缩。QTTS通过两种创新策略来建模这些离散代码：层次并行架构，该架构采用双自回归结构以更好地建模跨码本依赖关系，提升合成质量；延迟多头方法，该方法通过并行预测结合固定延迟来加速推理速度。我们的实验表明，所提出的框架在合成质量和保持表达内容方面优于基线，这表明通过多码本建模放大压缩是高质量、通用语音和音频生成的一个有希望的方向。 

---
# Selective Quantization Tuning for ONNX Models 

**Title (ZH)**: ONNX模型的选择性量化调优 

**Authors**: Nikolaos Louloudakis, Ajitha Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12196)  

**Abstract**: Quantization is a process that reduces the precision of deep neural network models to lower model size and computational demands, often at the cost of accuracy. However, fully quantized models may exhibit sub-optimal performance below acceptable levels and face deployment challenges on low-end hardware accelerators due to practical constraints. To address these issues, quantization can be selectively applied to only a subset of layers, but selecting which layers to exclude is non-trivial. To this direction, we propose TuneQn, a suite enabling selective quantization, deployment and execution of ONNX models across various CPU and GPU devices, combined with profiling and multi-objective optimization. TuneQn generates selectively quantized ONNX models, deploys them on different hardware, measures performance on metrics like accuracy and size, performs Pareto Front minimization to identify the best model candidate and visualizes the results. To demonstrate the effectiveness of TuneQn, we evaluated TuneQn on four ONNX models with two quantization settings across CPU and GPU devices. As a result, we demonstrated that our utility effectively performs selective quantization and tuning, selecting ONNX model candidates with up to a $54.14$% reduction in accuracy loss compared to the fully quantized model, and up to a $72.9$% model size reduction compared to the original model. 

**Abstract (ZH)**: TuneQn：一种针对ONNX模型的选择性量化、部署和执行套件，结合了性能分析和多目标优化 

---
# Revealing the Ancient Beauty: Digital Reconstruction of Temple Tiles using Computer Vision 

**Title (ZH)**: 揭示古代之美：计算机视觉在寺庙瓷砖数字重建中的应用 

**Authors**: Arkaprabha Basu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12195)  

**Abstract**: Modern digitised approaches have dramatically changed the preservation and restoration of cultural treasures, integrating computer scientists into multidisciplinary projects with ease. Machine learning, deep learning, and computer vision techniques have revolutionised developing sectors like 3D reconstruction, picture inpainting,IoT-based methods, genetic algorithms, and image processing with the integration of computer scientists into multidisciplinary initiatives. We suggest three cutting-edge techniques in recognition of the special qualities of Indian monuments, which are famous for their architectural skill and aesthetic appeal. First is the Fractal Convolution methodology, a segmentation method based on image processing that successfully reveals subtle architectural patterns within these irreplaceable cultural buildings. The second is a revolutionary Self-Sensitive Tile Filling (SSTF) method created especially for West Bengal's mesmerising Bankura Terracotta Temples with a brand-new data augmentation method called MosaicSlice on the third. Furthermore, we delve deeper into the Super Resolution strategy to upscale the images without losing significant amount of quality. Our methods allow for the development of seamless region-filling and highly detailed tiles while maintaining authenticity using a novel data augmentation strategy within affordable costs introducing automation. By providing effective solutions that preserve the delicate balance between tradition and innovation, this study improves the subject and eventually ensures unrivalled efficiency and aesthetic excellence in cultural heritage protection. The suggested approaches advance the field into an era of unmatched efficiency and aesthetic quality while carefully upholding the delicate equilibrium between tradition and innovation. 

**Abstract (ZH)**: 现代数字化方法极大地改变了文化珍宝的保护与恢复，将计算机科学家融入多学科项目变得轻而易举。机器学习、深度学习和计算机视觉技术通过将计算机科学家融入多学科举措，重塑了3D重建、图像修复、物联网方法、遗传算法和图像处理等领域。我们提出了三种针对印度著名建筑艺术和美学特征的前沿技术。首先是基于图像处理的分形卷积方法，成功揭示了这些不可替代的文化建筑中的细微建筑模式。其次是专为西孟加拉邦迷人的 Bankura 陶器庙宇设计的革命性自我敏感瓷砖填充（SSTF）方法，引入了新的数据扩增方法 MosaicSlice。此外，我们更深入研究了超分辨率策略，以提高图像质量而不丢失大量细节。我们的方法通过引入新颖的数据扩增策略，在保证成本效益的同时实现了无缝区域填充和高度详细的瓷砖制作，维护了真实感并实现自动化。通过提供有效的解决方案，平衡传统与创新，本研究提升了主题，最终确保了文化遗址保护的无与伦比的效率和美学卓越。建议的方法将领域带入一个前所未有的高效和美学质量的时代，同时谨慎地维护了传统与创新之间的微妙平衡。 

---
# BenchRL-QAS: Benchmarking reinforcement learning algorithms for quantum architecture search 

**Title (ZH)**: BenchRL-QAS: 量子架构搜索中强化学习算法的基准测试 

**Authors**: Azhar Ikhtiarudin, Aditi Das, Param Thakkar, Akash Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12189)  

**Abstract**: We introduce BenchRL-QAS, a unified benchmarking framework for systematically evaluating reinforcement learning (RL) algorithms in quantum architecture search (QAS) across diverse variational quantum algorithm tasks and system sizes ranging from 2- to 8-qubit. Our study benchmarks nine RL agents including both value-based and policy-gradient methods on representative quantum problems such as variational quantum eigensolver, variational quantum state diagonalization, quantum classification, and state preparation, spanning both noiseless and realistic noisy regimes. We propose a weighted ranking metric that balances accuracy, circuit depth, gate count, and computational efficiency, enabling fair and comprehensive comparison. Our results first reveal that RL-based quantum classifier outperforms baseline variational classifiers. Then we conclude that no single RL algorithm is universally optimal when considering a set of QAS tasks; algorithmic performance is highly context-dependent, varying with task structure, qubit count, and noise. This empirical finding provides strong evidence for the "no free lunch" principle in RL-based quantum circuit design and highlights the necessity of tailored algorithm selection and systematic benchmarking for advancing quantum circuit synthesis. This work represents the most comprehensive RL-QAS benchmarking effort to date, and BenchRL-QAS along with all experimental data are made publicly available to support reproducibility and future research this https URL. 

**Abstract (ZH)**: BenchRL-QAS：量子架构搜索中基于强化学习的统一基准框架 

---
# Wavelet-based Decoupling Framework for low-light Stereo Image Enhancement 

**Title (ZH)**: 基于小波的低光照立体图像解耦增强框架 

**Authors**: Shuangli Du, Siming Yan, Zhenghao Shi, Zhenzhen You, Lu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.12188)  

**Abstract**: Low-light images suffer from complex degradation, and existing enhancement methods often encode all degradation factors within a single latent space. This leads to highly entangled features and strong black-box characteristics, making the model prone to shortcut learning. To mitigate the above issues, this paper proposes a wavelet-based low-light stereo image enhancement method with feature space decoupling. Our insight comes from the following findings: (1) Wavelet transform enables the independent processing of low-frequency and high-frequency information. (2) Illumination adjustment can be achieved by adjusting the low-frequency component of a low-light image, extracted through multi-level wavelet decomposition. Thus, by using wavelet transform the feature space is decomposed into a low-frequency branch for illumination adjustment and multiple high-frequency branches for texture enhancement. Additionally, stereo low-light image enhancement can extract useful cues from another view to improve enhancement. To this end, we propose a novel high-frequency guided cross-view interaction module (HF-CIM) that operates within high-frequency branches rather than across the entire feature space, effectively extracting valuable image details from the other view. Furthermore, to enhance the high-frequency information, a detail and texture enhancement module (DTEM) is proposed based on cross-attention mechanism. The model is trained on a dataset consisting of images with uniform illumination and images with non-uniform illumination. Experimental results on both real and synthetic images indicate that our algorithm offers significant advantages in light adjustment while effectively recovering high-frequency information. The code and dataset are publicly available at: this https URL. 

**Abstract (ZH)**: 基于小波变换的低光照立体图像增强方法及特征空间分解 

---
# PRISM: Distributed Inference for Foundation Models at Edge 

**Title (ZH)**: PRISM: 边缘端基础模型分布式推理 

**Authors**: Muhammad Azlan Qazi, Alexandros Iosifidis, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12145)  

**Abstract**: Foundation models (FMs) have achieved remarkable success across a wide range of applications, from image classification to natural langurage processing, but pose significant challenges for deployment at edge. This has sparked growing interest in developing practical and efficient strategies for bringing foundation models to edge environments. In this work, we propose PRISM, a communication-efficient and compute-aware strategy for distributed Transformer inference on edge devices. Our method leverages a Segment Means representation to approximate intermediate output features, drastically reducing inter-device communication. Additionally, we restructure the self-attention mechanism to eliminate redundant computations caused by per-device Key/Value calculation in position-wise partitioning and design a partition-aware causal masking scheme tailored for autoregressive models. We evaluate PRISM on ViT, BERT, and GPT-2 across diverse datasets, namely CIFAR-10, CIFAR-100, ImageNet-1k, GLUE, and CBT. Our results demonstrate substantial reductions in communication overhead (up to 99.2% for BERT at compression rate CR = 128) and per-device computation (51.24% for BERT at the same setting), with only minor accuracy degradation. This method offers a scalable and practical solution for deploying foundation models in distributed resource-constrained environments. 

**Abstract (ZH)**: PRISM：适用于边缘设备的通信高效且计算感知的分布式Transformer推理策略 

---
# Quantum Machine Learning in Multi-Qubit Phase-Space Part I: Foundations 

**Title (ZH)**: 多量子位相空间中的量子机器学习 第一部分：基础理论 

**Authors**: Timothy Heightman, Edward Jiang, Ruth Mora-Soto, Maciej Lewenstein, Marcin Płodzień  

**Link**: [PDF](https://arxiv.org/pdf/2507.12117)  

**Abstract**: Quantum machine learning (QML) seeks to exploit the intrinsic properties of quantum mechanical systems, including superposition, coherence, and quantum entanglement for classical data processing. However, due to the exponential growth of the Hilbert space, QML faces practical limits in classical simulations with the state-vector representation of quantum system. On the other hand, phase-space methods offer an alternative by encoding quantum states as quasi-probability functions. Building on prior work in qubit phase-space and the Stratonovich-Weyl (SW) correspondence, we construct a closed, composable dynamical formalism for one- and many-qubit systems in phase-space. This formalism replaces the operator algebra of the Pauli group with function dynamics on symplectic manifolds, and recasts the curse of dimensionality in terms of harmonic support on a domain that scales linearly with the number of qubits. It opens a new route for QML based on variational modelling over phase-space. 

**Abstract (ZH)**: 量子机器学习（QML）aimed at利用量子力学系统的固有性质，如叠加、相干性和量子纠缠进行经典数据处理。然而，由于希尔伯特空间的指数增长，QML在使用量子系统的态矢量表示进行经典模拟时面临实操限制。另一方面，相空间方法通过将量子态编码为拟概率函数提供了另一种选择。在先前关于量子比特相空间和Stratonovich-Weyl (SW) 对应工作的基础上，我们构建了一种封闭且可组合的动力学形式主义，适用于单量子比特和多量子比特系统。该形式主义用辛流形上的函数动力学取代了Pauli群的算子代数，并将维度灾难重新定义为谐波支持在随量子比特数目线性增长的空间域上的问题。它为基于相空间变分建模的量子机器学习开辟了一条新途径。 

---
# Multimodal Coordinated Online Behavior: Trade-offs and Strategies 

**Title (ZH)**: 多模态协调在线行为：权衡与策略 

**Authors**: Lorenzo Mannocci, Stefano Cresci, Matteo Magnani, Anna Monreale, Maurizio Tesconi  

**Link**: [PDF](https://arxiv.org/pdf/2507.12108)  

**Abstract**: Coordinated online behavior, which spans from beneficial collective actions to harmful manipulation such as disinformation campaigns, has become a key focus in digital ecosystem analysis. Traditional methods often rely on monomodal approaches, focusing on single types of interactions like co-retweets or co-hashtags, or consider multiple modalities independently of each other. However, these approaches may overlook the complex dynamics inherent in multimodal coordination. This study compares different ways of operationalizing the detection of multimodal coordinated behavior. It examines the trade-off between weakly and strongly integrated multimodal models, highlighting the balance between capturing broader coordination patterns and identifying tightly coordinated behavior. By comparing monomodal and multimodal approaches, we assess the unique contributions of different data modalities and explore how varying implementations of multimodality impact detection outcomes. Our findings reveal that not all the modalities provide distinct insights, but that with a multimodal approach we can get a more comprehensive understanding of coordination dynamics. This work enhances the ability to detect and analyze coordinated online behavior, offering new perspectives for safeguarding the integrity of digital platforms. 

**Abstract (ZH)**: 跨模态协调网络行为，从有益的集体行动到有害的操纵如信息操纵活动，已成为数字生态系统分析中的关键重点。传统方法往往依赖于单一模态的方法，关注单一类型的交互，如共转发或共标签，或者单独考虑多种模态。然而，这些方法可能忽视了跨模态协调中存在的复杂动态。本研究比较了不同检测跨模态协调行为的方法。它探讨了弱集成与强集成跨模态模型之间的权衡，突出捕获更广泛协调模式与识别紧密协调行为之间的平衡。通过比较单模态和跨模态方法，我们评估了不同数据模态的独特贡献，并探讨了不同跨模态实现对检测结果的影响。我们的研究发现，并非所有模态都提供独特见解，但通过跨模态方法可以获得对协调动态的更全面理解。这项工作增强了检测和分析网络协调行为的能力，为保护数字平台的完整性提供了新的视角。 

---
# Non-Adaptive Adversarial Face Generation 

**Title (ZH)**: 非自适应对抗面部生成 

**Authors**: Sunpill Kim, Seunghun Paik, Chanwoo Hwang, Minsu Kim, Jae Hong Seo  

**Link**: [PDF](https://arxiv.org/pdf/2507.12107)  

**Abstract**: Adversarial attacks on face recognition systems (FRSs) pose serious security and privacy threats, especially when these systems are used for identity verification. In this paper, we propose a novel method for generating adversarial faces-synthetic facial images that are visually distinct yet recognized as a target identity by the FRS. Unlike iterative optimization-based approaches (e.g., gradient descent or other iterative solvers), our method leverages the structural characteristics of the FRS feature space. We figure out that individuals sharing the same attribute (e.g., gender or race) form an attributed subsphere. By utilizing such subspheres, our method achieves both non-adaptiveness and a remarkably small number of queries. This eliminates the need for relying on transferability and open-source surrogate models, which have been a typical strategy when repeated adaptive queries to commercial FRSs are impossible. Despite requiring only a single non-adaptive query consisting of 100 face images, our method achieves a high success rate of over 93% against AWS's CompareFaces API at its default threshold. Furthermore, unlike many existing attacks that perturb a given image, our method can deliberately produce adversarial faces that impersonate the target identity while exhibiting high-level attributes chosen by the adversary. 

**Abstract (ZH)**: 对抗攻击对面部识别系统（FRS）的威胁：一种基于结构特征的生成对抗性面部的方法 

---
# From Static to Intelligent: Evolving SaaS Pricing with LLMs 

**Title (ZH)**: 从静态到智能：基于LLM的SaaS定价演变 

**Authors**: Francisco Javier Cavero, Juan C. Alonso, Antonio Ruiz-Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2507.12104)  

**Abstract**: The SaaS paradigm has revolutionized software distribution by offering flexible pricing options to meet diverse customer needs. However, the rapid expansion of the SaaS market has introduced significant complexity for DevOps teams, who must manually manage and evolve pricing structures, an approach that is both time-consuming and prone to errors. The absence of automated tools for pricing analysis restricts the ability to efficiently evaluate, optimize, and scale these models. This paper proposes leveraging intelligent pricing (iPricing), dynamic, machine-readable pricing models, as a solution to these challenges. Intelligent pricing enables competitive analysis, streamlines operational decision-making, and supports continuous pricing evolution in response to market dynamics, leading to improved efficiency and accuracy. We present an LLM-driven approach that automates the transformation of static HTML pricing into iPricing, significantly improving efficiency and consistency while minimizing human error. Our implementation, AI4Pricing2Yaml, features a basic Information Extractor that uses web scraping and LLMs technologies to extract essential pricing components, plans, features, usage limits, and add-ons, from SaaS websites. Validation against a dataset of 30 distinct commercial SaaS, encompassing over 150 intelligent pricings, demonstrates the system's effectiveness in extracting the desired elements across all steps. However, challenges remain in addressing hallucinations, complex structures, and dynamic content. This work highlights the potential of automating intelligent pricing transformation to streamline SaaS pricing management, offering implications for improved consistency and scalability in an increasingly intricate pricing landscape. Future research will focus on refining extraction capabilities and enhancing the system's adaptability to a wider range of SaaS websites. 

**Abstract (ZH)**: 基于智能定价的SaaS定价模型自动化转换：提高效率和一致性 

---
# BOOKCOREF: Coreference Resolution at Book Scale 

**Title (ZH)**: BOOKCOREF：图书规模的共指解析 

**Authors**: Giuliano Martinelli, Tommaso Bonomo, Pere-Lluís Huguet Cabot, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2507.12075)  

**Abstract**: Coreference Resolution systems are typically evaluated on benchmarks containing small- to medium-scale documents. When it comes to evaluating long texts, however, existing benchmarks, such as LitBank, remain limited in length and do not adequately assess system capabilities at the book scale, i.e., when co-referring mentions span hundreds of thousands of tokens. To fill this gap, we first put forward a novel automatic pipeline that produces high-quality Coreference Resolution annotations on full narrative texts. Then, we adopt this pipeline to create the first book-scale coreference benchmark, BOOKCOREF, with an average document length of more than 200,000 tokens. We carry out a series of experiments showing the robustness of our automatic procedure and demonstrating the value of our resource, which enables current long-document coreference systems to gain up to +20 CoNLL-F1 points when evaluated on full books. Moreover, we report on the new challenges introduced by this unprecedented book-scale setting, highlighting that current models fail to deliver the same performance they achieve on smaller documents. We release our data and code to encourage research and development of new book-scale Coreference Resolution systems at this https URL. 

**Abstract (ZH)**: 核心参考决议系统通常在包含小型到中型规模文档的基准上进行评估。然而，在评估长文本时，现有的基准，如LitBank，在长度上仍然有限，并不能充分评估系统在图书规模下的能力，即当共指mention跨越数十万token时。为了填补这一空白，我们首先提出了一种新颖的自动管道，用于生成高质量的核心参考决议注释，覆盖完整的叙述文本。然后，我们采用此管道创建了首个图书规模的核心参考基准BOOKCOREF，其文档平均长度超过200,000token。我们进行了一系列实验，展示了我们自动流程的稳健性，并证明了我们资源的价值，这对于评估完整书籍的核心参考系统性能提高了多达+20 CoNLL-F1点。此外，我们报告了由此开创的图书规模设置引入的新挑战，指出当前模型在小型文档上的表现无法在大文档中达到相同水平。我们在此发布我们的数据和代码，以鼓励对新的图书规模核心参考决议系统的研究和发展：https://yourlinkhere.com 

---
# StylOch at PAN: Gradient-Boosted Trees with Frequency-Based Stylometric Features 

**Title (ZH)**: StylOch 在 PAN 上：基于频率的体式特征的梯度提升树 

**Authors**: Jeremi K. Ochab, Mateusz Matias, Tymoteusz Boba, Tomasz Walkowiak  

**Link**: [PDF](https://arxiv.org/pdf/2507.12064)  

**Abstract**: This submission to the binary AI detection task is based on a modular stylometric pipeline, where: public spaCy models are used for text preprocessing (including tokenisation, named entity recognition, dependency parsing, part-of-speech tagging, and morphology annotation) and extracting several thousand features (frequencies of n-grams of the above linguistic annotations); light-gradient boosting machines are used as the classifier. We collect a large corpus of more than 500 000 machine-generated texts for the classifier's training. We explore several parameter options to increase the classifier's capacity and take advantage of that training set. Our approach follows the non-neural, computationally inexpensive but explainable approach found effective previously. 

**Abstract (ZH)**: 基于模块化文体学流水线的二进制AI检测任务提交 

---
# InstructFLIP: Exploring Unified Vision-Language Model for Face Anti-spoofing 

**Title (ZH)**: InstructFLIP: 探索统一的视觉-语言模型在面部防欺骗中的应用 

**Authors**: Kun-Hsiang Lin, Yu-Wen Tseng, Kang-Yang Huang, Jhih-Ciang Wu, Wen-Huang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.12060)  

**Abstract**: Face anti-spoofing (FAS) aims to construct a robust system that can withstand diverse attacks. While recent efforts have concentrated mainly on cross-domain generalization, two significant challenges persist: limited semantic understanding of attack types and training redundancy across domains. We address the first by integrating vision-language models (VLMs) to enhance the perception of visual input. For the second challenge, we employ a meta-domain strategy to learn a unified model that generalizes well across multiple domains. Our proposed InstructFLIP is a novel instruction-tuned framework that leverages VLMs to enhance generalization via textual guidance trained solely on a single domain. At its core, InstructFLIP explicitly decouples instructions into content and style components, where content-based instructions focus on the essential semantics of spoofing, and style-based instructions consider variations related to the environment and camera characteristics. Extensive experiments demonstrate the effectiveness of InstructFLIP by outperforming SOTA models in accuracy and substantially reducing training redundancy across diverse domains in FAS. Project website is available at this https URL. 

**Abstract (ZH)**: 人脸识别防欺骗（Face Anti-Spoofing, FAS）旨在构建一个 robust 系统以应对各种攻击。尽管近期努力主要集中在跨域泛化上，但依然存在两大挑战：对攻击类型的有限语义理解以及不同领域间的训练冗余。我们通过整合视觉语言模型（VLMs）以增强对视觉输入的感知来应对第一个挑战。针对第二个挑战，我们采用元领域策略来学习一个能够在多个领域泛化的统一模型。我们提出的 InstructFLIP 是一种新颖的指令调优框架，通过仅在一个领域上进行文本指导，利用 VLMs 来提升泛化能力。InstructFLIP 在核心部分显式地将指令分解为内容和风格两个组件，其中基于内容的指令关注伪装的关键语义，基于风格的指令考虑环境和摄像机特性等方面的变化。大量实验结果表明，InstructFLIP 在准确率上优于当前最优模型，并且在 FAS 的多个领域中大幅减少了训练冗余。项目网址可访问此 <https://>。 

---
# Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery 

**Title (ZH)**: 基于视图内和视图间相关性的多视图新型类发现 

**Authors**: Xinhang Wan, Jiyuan Liu, Qian Qu, Suyuan Liu, Chuyu Zhang, Fangdi Wang, Xinwang Liu, En Zhu, Kunlun He  

**Link**: [PDF](https://arxiv.org/pdf/2507.12029)  

**Abstract**: In this paper, we address the problem of novel class discovery (NCD), which aims to cluster novel classes by leveraging knowledge from disjoint known classes. While recent advances have made significant progress in this area, existing NCD methods face two major limitations. First, they primarily focus on single-view data (e.g., images), overlooking the increasingly common multi-view data, such as multi-omics datasets used in disease diagnosis. Second, their reliance on pseudo-labels to supervise novel class clustering often results in unstable performance, as pseudo-label quality is highly sensitive to factors such as data noise and feature dimensionality. To address these challenges, we propose a novel framework named Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery (IICMVNCD), which is the first attempt to explore NCD in multi-view setting so far. Specifically, at the intra-view level, leveraging the distributional similarity between known and novel classes, we employ matrix factorization to decompose features into view-specific shared base matrices and factor matrices. The base matrices capture distributional consistency among the two datasets, while the factor matrices model pairwise relationships between samples. At the inter-view level, we utilize view relationships among known classes to guide the clustering of novel classes. This includes generating predicted labels through the weighted fusion of factor matrices and dynamically adjusting view weights of known classes based on the supervision loss, which are then transferred to novel class learning. Experimental results validate the effectiveness of our proposed approach. 

**Abstract (ZH)**: 基于 intra-view 和 inter-view 相关性的多视角新型类发现 (IICMVNCD) 

---
# SS-DC: Spatial-Spectral Decoupling and Coupling Across Visible-Infrared Gap for Domain Adaptive Object Detection 

**Title (ZH)**: SS-DC: 可见光-红外区间跨域适配目标检测中的空间-谱域解藕与耦合 

**Authors**: Xiwei Zhang, Chunjin Yang, Yiming Xiao, Runtong Zhang, Fanman Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.12017)  

**Abstract**: Unsupervised domain adaptive object detection (UDAOD) from the visible domain to the infrared (RGB-IR) domain is challenging. Existing methods regard the RGB domain as a unified domain and neglect the multiple subdomains within it, such as daytime, nighttime, and foggy scenes. We argue that decoupling the domain-invariant (DI) and domain-specific (DS) features across these multiple subdomains is beneficial for RGB-IR domain adaptation. To this end, this paper proposes a new SS-DC framework based on a decoupling-coupling strategy. In terms of decoupling, we design a Spectral Adaptive Idempotent Decoupling (SAID) module in the aspect of spectral decomposition. Due to the style and content information being highly embedded in different frequency bands, this module can decouple DI and DS components more accurately and interpretably. A novel filter bank-based spectral processing paradigm and a self-distillation-driven decoupling loss are proposed to improve the spectral domain decoupling. In terms of coupling, a new spatial-spectral coupling method is proposed, which realizes joint coupling through spatial and spectral DI feature pyramids. Meanwhile, this paper introduces DS from decoupling to reduce the domain bias. Extensive experiments demonstrate that our method can significantly improve the baseline performance and outperform existing UDAOD methods on multiple RGB-IR datasets, including a new experimental protocol proposed in this paper based on the FLIR-ADAS dataset. 

**Abstract (ZH)**: 无监督域自适应可见光到红外域目标检测（UDAOD）：从可见光域到红外域的无监督域自适应目标检测 

---
# Identifying Signatures of Image Phenotypes to Track Treatment Response in Liver Disease 

**Title (ZH)**: 识别图像表型的特征以追踪肝病治疗反应 

**Authors**: Matthias Perkonigg, Nina Bastati, Ahmed Ba-Ssalamah, Peter Mesenbrink, Alexander Goehler, Miljen Martic, Xiaofei Zhou, Michael Trauner, Georg Langs  

**Link**: [PDF](https://arxiv.org/pdf/2507.12012)  

**Abstract**: Quantifiable image patterns associated with disease progression and treatment response are critical tools for guiding individual treatment, and for developing novel therapies. Here, we show that unsupervised machine learning can identify a pattern vocabulary of liver tissue in magnetic resonance images that quantifies treatment response in diffuse liver disease. Deep clustering networks simultaneously encode and cluster patches of medical images into a low-dimensional latent space to establish a tissue vocabulary. The resulting tissue types capture differential tissue change and its location in the liver associated with treatment response. We demonstrate the utility of the vocabulary on a randomized controlled trial cohort of non-alcoholic steatohepatitis patients. First, we use the vocabulary to compare longitudinal liver change in a placebo and a treatment cohort. Results show that the method identifies specific liver tissue change pathways associated with treatment, and enables a better separation between treatment groups than established non-imaging measures. Moreover, we show that the vocabulary can predict biopsy derived features from non-invasive imaging data. We validate the method on a separate replication cohort to demonstrate the applicability of the proposed method. 

**Abstract (ZH)**: 可量化图像模式与疾病进展和治疗反应相关，是指导个性化治疗和开发新型疗法的重要工具。在这里，我们展示了无监督机器学习可以识别磁共振图像中肝组织的模式词汇，量化弥漫性肝病的治疗反应。深度聚类网络同时将医学图像的块编码并聚类到低维潜在空间中，建立组织词汇。产生的组织类型捕获与治疗反应相关的组织变化及其在肝脏中的位置。我们通过一项随机对照试验队列的非酒精性 steatohepatitis 患者组来展示词汇的实用性。首先，我们使用词汇比较安慰剂组和治疗组的纵向肝组织变化。结果显示，该方法识别出与治疗相关的特定肝脏组织变化路径，并能比现有非影像学指标更好地分离治疗组。此外，我们展示了该词汇可以预测从非侵入性影像数据推导出的活检特征。我们在独立的复制队列上验证该方法，以证明所提出方法的应用性。 

---
# DUSE: A Data Expansion Framework for Low-resource Automatic Modulation Recognition based on Active Learning 

**Title (ZH)**: DUSE：一种基于主动学习的稀少资源自动调制识别数据扩展框架 

**Authors**: Yao Lu, Hongyu Gao, Zhuangzhi Chen, Dongwei Xu, Yun Lin, Qi Xuan, Guan Gui  

**Link**: [PDF](https://arxiv.org/pdf/2507.12011)  

**Abstract**: Although deep neural networks have made remarkable achievements in the field of automatic modulation recognition (AMR), these models often require a large amount of labeled data for training. However, in many practical scenarios, the available target domain data is scarce and difficult to meet the needs of model training. The most direct way is to collect data manually and perform expert annotation, but the high time and labor costs are unbearable. Another common method is data augmentation. Although it can enrich training samples to a certain extent, it does not introduce new data and therefore cannot fundamentally solve the problem of data scarcity. To address these challenges, we introduce a data expansion framework called Dynamic Uncertainty-driven Sample Expansion (DUSE). Specifically, DUSE uses an uncertainty scoring function to filter out useful samples from relevant AMR datasets and employs an active learning strategy to continuously refine the scorer. Extensive experiments demonstrate that DUSE consistently outperforms 8 coreset selection baselines in both class-balance and class-imbalance settings. Besides, DUSE exhibits strong cross-architecture generalization for unseen models. 

**Abstract (ZH)**: 动态不确定性驱动样本扩展框架（DUSE）在自动调制识别领域的数据扩展 

---
# Dual form Complementary Masking for Domain-Adaptive Image Segmentation 

**Title (ZH)**: 域适应图像分割的双重形式互补掩蔽 

**Authors**: Jiawen Wang, Yinda Chen, Xiaoyu Liu, Che Liu, Dong Liu, Jianqing Gao, Zhiwei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.12008)  

**Abstract**: Recent works have correlated Masked Image Modeling (MIM) with consistency regularization in Unsupervised Domain Adaptation (UDA). However, they merely treat masking as a special form of deformation on the input images and neglect the theoretical analysis, which leads to a superficial understanding of masked reconstruction and insufficient exploitation of its potential in enhancing feature extraction and representation learning. In this paper, we reframe masked reconstruction as a sparse signal reconstruction problem and theoretically prove that the dual form of complementary masks possesses superior capabilities in extracting domain-agnostic image features. Based on this compelling insight, we propose MaskTwins, a simple yet effective UDA framework that integrates masked reconstruction directly into the main training pipeline. MaskTwins uncovers intrinsic structural patterns that persist across disparate domains by enforcing consistency between predictions of images masked in complementary ways, enabling domain generalization in an end-to-end manner. Extensive experiments verify the superiority of MaskTwins over baseline methods in natural and biological image segmentation. These results demonstrate the significant advantages of MaskTwins in extracting domain-invariant features without the need for separate pre-training, offering a new paradigm for domain-adaptive segmentation. 

**Abstract (ZH)**: 近期 studies 将 Masked Image Modeling (MIM) 与 Unsupervised Domain Adaptation (UDA) 中的一致性正则化联系了起来。然而，这些研究仅仅将掩蔽视为输入图像的一种特殊变形形式，并忽视了理论分析，导致对掩蔽重建的理解肤浅，无法充分发掘其在增强特征提取和表示学习方面的潜力。本文重新将掩蔽重建视为稀疏信号重构问题，并理论证明互补掩蔽的对偶形式在提取领域不变图像特征方面具有优越的能力。基于这一令人信服的见解，我们提出了 MaskTwins，一种将掩蔽重建直接集成到主训练管道中的简单而有效的UDA框架。MaskTwins 通过在以不同方式掩蔽的图像之间强制一致性来揭示跨不同领域内在的结构模式，从而以端到端的方式实现领域泛化。广泛的实验验证了 MaskTwins 在自然和生物图像分割中的优越性，证明了 MaskTwins 在无需单独预训练的情况下提取领域不变特征的优势，为领域适应分割提供了新的范式。 

---
# Frequency-Dynamic Attention Modulation for Dense Prediction 

**Title (ZH)**: 频域动态注意力调制dense预测 

**Authors**: Linwei Chen, Lin Gu, Ying Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12006)  

**Abstract**: Vision Transformers (ViTs) have significantly advanced computer vision, demonstrating strong performance across various tasks. However, the attention mechanism in ViTs makes each layer function as a low-pass filter, and the stacked-layer architecture in existing transformers suffers from frequency vanishing. This leads to the loss of critical details and textures. We propose a novel, circuit-theory-inspired strategy called Frequency-Dynamic Attention Modulation (FDAM), which can be easily plugged into ViTs. FDAM directly modulates the overall frequency response of ViTs and consists of two techniques: Attention Inversion (AttInv) and Frequency Dynamic Scaling (FreqScale). Since circuit theory uses low-pass filters as fundamental elements, we introduce AttInv, a method that generates complementary high-pass filtering by inverting the low-pass filter in the attention matrix, and dynamically combining the two. We further design FreqScale to weight different frequency components for fine-grained adjustments to the target response function. Through feature similarity analysis and effective rank evaluation, we demonstrate that our approach avoids representation collapse, leading to consistent performance improvements across various models, including SegFormer, DeiT, and MaskDINO. These improvements are evident in tasks such as semantic segmentation, object detection, and instance segmentation. Additionally, we apply our method to remote sensing detection, achieving state-of-the-art results in single-scale settings. The code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于电路理论的频域动态注意力调制（FDAM）：一种适用于ViTs的新型策略 

---
# Can LLMs Find Fraudsters? Multi-level LLM Enhanced Graph Fraud Detection 

**Title (ZH)**: LLM在多层图欺诈检测中的欺诈发现能力 

**Authors**: Tairan Huang, Yili Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11997)  

**Abstract**: Graph fraud detection has garnered significant attention as Graph Neural Networks (GNNs) have proven effective in modeling complex relationships within multimodal data. However, existing graph fraud detection methods typically use preprocessed node embeddings and predefined graph structures to reveal fraudsters, which ignore the rich semantic cues contained in raw textual information. Although Large Language Models (LLMs) exhibit powerful capabilities in processing textual information, it remains a significant challenge to perform multimodal fusion of processed textual embeddings with graph structures. In this paper, we propose a \textbf{M}ulti-level \textbf{L}LM \textbf{E}nhanced Graph Fraud \textbf{D}etection framework called MLED. In MLED, we utilize LLMs to extract external knowledge from textual information to enhance graph fraud detection methods. To integrate LLMs with graph structure information and enhance the ability to distinguish fraudsters, we design a multi-level LLM enhanced framework including type-level enhancer and relation-level enhancer. One is to enhance the difference between the fraudsters and the benign entities, the other is to enhance the importance of the fraudsters in different relations. The experiments on four real-world datasets show that MLED achieves state-of-the-art performance in graph fraud detection as a generalized framework that can be applied to existing methods. 

**Abstract (ZH)**: 多层LLM增强图欺诈检测框架MLED 

---
# Robust Planning for Autonomous Vehicles with Diffusion-Based Failure Samplers 

**Title (ZH)**: 基于扩散型故障采样的鲁棒自主车辆规划 

**Authors**: Juanran Wang, Marc R. Schlichting, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2507.11991)  

**Abstract**: High-risk traffic zones such as intersections are a major cause of collisions. This study leverages deep generative models to enhance the safety of autonomous vehicles in an intersection context. We train a 1000-step denoising diffusion probabilistic model to generate collision-causing sensor noise sequences for an autonomous vehicle navigating a four-way intersection based on the current relative position and velocity of an intruder. Using the generative adversarial architecture, the 1000-step model is distilled into a single-step denoising diffusion model which demonstrates fast inference speed while maintaining similar sampling quality. We demonstrate one possible application of the single-step model in building a robust planner for the autonomous vehicle. The planner uses the single-step model to efficiently sample potential failure cases based on the currently measured traffic state to inform its decision-making. Through simulation experiments, the robust planner demonstrates significantly lower failure rate and delay rate compared with the baseline Intelligent Driver Model controller. 

**Abstract (ZH)**: 高风险交通区域（如交叉口）是碰撞的主要原因。本研究利用深度生成模型在交叉口场景中提高自动驾驶车辆的安全性。我们基于当前侵入物的相对位置和速度，训练一个1000步去噪扩散概率模型，生成导致碰撞的传感器噪声序列，用于自动驾驶车辆在四向交叉口的行驶。通过生成对抗性架构，1000步模型被精简为一个单步去噪扩散模型，该模型在保持类似采样质量的同时，展现出更快的推理速度。我们展示了一步模型的一种可能应用，即构建一个 robust 的规划器，该规划器利用一步模型根据当前测量的交通状态高效地采样潜在故障案例，以指导其决策。通过仿真实验，robust 规划器显示出显著更低的故障率和延迟率，优于基准智能驾驶员模型控制器。 

---
# Formal Verification of Neural Certificates Done Dynamically 

**Title (ZH)**: 动态验证神经证书的形式化验证 

**Authors**: Thomas A. Henzinger, Konstantin Kueffner, Emily Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11987)  

**Abstract**: Neural certificates have emerged as a powerful tool in cyber-physical systems control, providing witnesses of correctness. These certificates, such as barrier functions, often learned alongside control policies, once verified, serve as mathematical proofs of system safety. However, traditional formal verification of their defining conditions typically faces scalability challenges due to exhaustive state-space exploration. To address this challenge, we propose a lightweight runtime monitoring framework that integrates real-time verification and does not require access to the underlying control policy. Our monitor observes the system during deployment and performs on-the-fly verification of the certificate over a lookahead region to ensure safety within a finite prediction horizon. We instantiate this framework for ReLU-based control barrier functions and demonstrate its practical effectiveness in a case study. Our approach enables timely detection of safety violations and incorrect certificates with minimal overhead, providing an effective but lightweight alternative to the static verification of the certificates. 

**Abstract (ZH)**: 神经证书已成为物理信息系统控制的强大工具，提供了正确性的证据。这些证书，例如边界函数，通常与控制策略一同学习，验证后成为系统安全性的数学证明。然而，传统形式验证通常由于耗尽状态空间的探索而面临可扩展性挑战。为了应对这一挑战，我们提出了一种轻量级的运行时监控框架，该框架集成实时验证且无需访问底层控制策略。该监控器在系统部署期间进行观察，并对预视窗口内的证书即时进行验证，以确保在有限的预测窗口内系统的安全性。我们为基于ReLU的控制边界函数实例化了该框架，并在案例研究中证明了其实用有效性。我们的方法能够及时检测安全违规和不正确的证书，且具有最小的开销，提供了一种有效但轻量级的证书静态验证替代方案。 

---
# Online Training and Pruning of Deep Reinforcement Learning Networks 

**Title (ZH)**: 在线训练与裁剪深度强化学习网络 

**Authors**: Valentin Frank Ingmar Guenter, Athanasios Sideris  

**Link**: [PDF](https://arxiv.org/pdf/2507.11975)  

**Abstract**: Scaling deep neural networks (NN) of reinforcement learning (RL) algorithms has been shown to enhance performance when feature extraction networks are used but the gained performance comes at the significant expense of increased computational and memory complexity. Neural network pruning methods have successfully addressed this challenge in supervised learning. However, their application to RL is underexplored. We propose an approach to integrate simultaneous training and pruning within advanced RL methods, in particular to RL algorithms enhanced by the Online Feature Extractor Network (OFENet). Our networks (XiNet) are trained to solve stochastic optimization problems over the RL networks' weights and the parameters of variational Bernoulli distributions for 0/1 Random Variables $\xi$ scaling each unit in the networks. The stochastic problem formulation induces regularization terms that promote convergence of the variational parameters to 0 when a unit contributes little to the performance. In this case, the corresponding structure is rendered permanently inactive and pruned from its network. We propose a cost-aware, sparsity-promoting regularization scheme, tailored to the DenseNet architecture of OFENets expressing the parameter complexity of involved networks in terms of the parameters of the RVs in these networks. Then, when matching this cost with the regularization terms, the many hyperparameters associated with them are automatically selected, effectively combining the RL objectives and network compression. We evaluate our method on continuous control benchmarks (MuJoCo) and the Soft Actor-Critic RL agent, demonstrating that OFENets can be pruned considerably with minimal loss in performance. Furthermore, our results confirm that pruning large networks during training produces more efficient and higher performing RL agents rather than training smaller networks from scratch. 

**Abstract (ZH)**: 加强深度神经网络（NN）在强化学习（RL）算法中的应用已被证明能提升性能，但在使用特征提取网络时，这种性能提升伴随着计算和内存复杂度的显著增加。神经网络裁剪方法已经在监督学习中成功地解决了这一挑战。然而，它们在RL中的应用尚未得到充分探索。我们提出了一种在高级RL方法中结合同时训练和裁剪的方法，特别适用于通过在线特征提取网络（OFENet）增强的RL算法。我们的网络（XiNet）被训练以解决RL网络权重和0/1随机变量ξ的变分伯努利分布参数下的随机优化问题，这些随机变量用于缩放网络中的每个单元。随机问题的形式化推导出正则化项，当一个单元对性能贡献很少时，促进变分参数收敛于0。在这种情况下，相应的结构被永久性地禁用并从其网络中裁剪。我们提出了一种成本意识的、促进稀疏性的正则化方案，针对OFENets的DenseNet架构，通过网络中的随机变量参数表达参与网络的参数复杂性。然后，当匹配成本与正则化项时，与它们相关的许多超参数会自动选择，从而有效地结合了RL目标和网络压缩。我们在连续控制基准（MuJoCo）和Soft Actor-Critic RL代理上评估了该方法，表明可以显著裁剪OFENets并几乎不损失性能。此外，我们的结果证实，在训练过程中裁剪大网络会生成更高效和性能更高的RL代理，而不是从头训练较小的网络。 

---
# Toxicity-Aware Few-Shot Prompting for Low-Resource Singlish Translation 

**Title (ZH)**: 面向毒有害内容的少样本提示生成方法在低资源新加坡英语翻译中的应用 

**Authors**: Ziyu Ge, Gabriel Chua, Leanne Tan, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.11966)  

**Abstract**: As online communication increasingly incorporates under-represented languages and colloquial dialects, standard translation systems often fail to preserve local slang, code-mixing, and culturally embedded markers of harmful speech. Translating toxic content between low-resource language pairs poses additional challenges due to scarce parallel data and safety filters that sanitize offensive expressions. In this work, we propose a reproducible, two-stage framework for toxicity-preserving translation, demonstrated on a code-mixed Singlish safety corpus. First, we perform human-verified few-shot prompt engineering: we iteratively curate and rank annotator-selected Singlish-target examples to capture nuanced slang, tone, and toxicity. Second, we optimize model-prompt pairs by benchmarking several large language models using semantic similarity via direct and back-translation. Quantitative human evaluation confirms the effectiveness and efficiency of our pipeline. Beyond improving translation quality, our framework contributes to the safety of multicultural LLMs by supporting culturally sensitive moderation and benchmarking in low-resource contexts. By positioning Singlish as a testbed for inclusive NLP, we underscore the importance of preserving sociolinguistic nuance in real-world applications such as content moderation and regional platform governance. 

**Abstract (ZH)**: 随着在线通信越来越多地 Incorporating 偏言僻语和口语方言，标准翻译系统往往无法保留当地俚语、代码混合和文化嵌入的有害言论标记。由于低资源语言对之间的平行数据稀缺以及会过滤冒犯性表达的安全过滤器，翻译低资源语言对之间的有害内容带来了额外的挑战。在本文中，我们提出了一种可重现的两阶段框架，用于保留有害内容的翻译，并在混合方言的安全语料库上进行了展示。首先，我们进行人工验证的小样本提示工程：我们迭代地整理和排名注释者选择的Singlish目标示例，以捕捉细微的俚语、语气和有害内容。第二，我们通过多种大型语言模型的基准测试，使用语义相似性通过直接翻译和反向翻译优化模型-提示对。定量的人类评估证实了我们管道的有效性和效率。除了改善翻译质量，我们的框架还通过提供文化敏感的调节和支持低资源情境下的基准测试，促进了多文化大语言模型的安全性。通过将Singlish作为包容性自然语言处理的试验平台，我们强调了在内容调节和区域平台治理等实际应用中保留社会语言学细腻之处的重要性。 

---
# PoTPTQ: A Two-step Power-of-Two Post-training for LLMs 

**Title (ZH)**: PoTPTQ: LLMs 的两步骤二进制后训练方法 

**Authors**: Xinyu Wang, Vahid Partovi Nia, Peng Lu, Jerry Huang, Xiao-Wen Chang, Boxing Chen, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2507.11959)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing (NLP) tasks. However, their deployment is challenging due to the substantial computational resources required. Power-of-two (PoT) quantization is a general tool to counteract this difficulty. Albeit previous works on PoT quantization can be efficiently dequantized on CPUs using fixed-point addition, it showed less effectiveness on GPUs. The reason is entanglement of the sign bit and sequential bit manipulations needed for dequantization. We propose a novel POT quantization framework for LLM weights that (i) outperforms state-of-the-art accuracy in extremely low-precision number formats, and (ii) enables faster inference through more efficient dequantization. To maintain the accuracy of the quantized model, we introduce a two-step post-training algorithm: (i) initialize the quantization scales with a robust starting point, and (ii) refine these scales using a minimal calibration set. The performance of our PoT post-training algorithm surpasses the current state-of-the-art in integer quantization, particularly at low precisions such as 2- and 3-bit formats. Our PoT quantization accelerates the dequantization step required for the floating point inference and leads to $3.67\times$ speed up on a NVIDIA V100, and $1.63\times$ on a NVIDIA RTX 4090, compared to uniform integer dequantization. 

**Abstract (ZH)**: 基于幂_of_2量化的大语言模型权重：高性能低精度表示与快速推理 

---
# Kevin: Multi-Turn RL for Generating CUDA Kernels 

**Title (ZH)**: Kevin: 多轮强化学习生成 CUDA 内核 

**Authors**: Carlo Baronio, Pietro Marsella, Ben Pan, Simon Guo, Silas Alberti  

**Link**: [PDF](https://arxiv.org/pdf/2507.11948)  

**Abstract**: Writing GPU kernels is a challenging task and critical for AI systems' efficiency. It is also highly iterative: domain experts write code and improve performance through execution feedback. Moreover, it presents verifiable rewards like correctness and speedup, making it a natural environment to apply Reinforcement Learning (RL). To explicitly incorporate the iterative nature of this process into training, we develop a flexible multi-turn RL recipe that addresses unique challenges encountered in real-world settings, such as learning from long trajectories and effective reward attribution across turns. We present Kevin - K(ernel D)evin, the first model trained with multi-turn RL for CUDA kernel generation and optimization. In our evaluation setup, Kevin shows significant gains over its base model (QwQ-32B), improving correctness of generated kernels (in pure CUDA) from 56% to 82% and mean speedup from 0.53x to 1.10x of baseline (PyTorch Eager), and surpassing frontier models like o4-mini (0.78x). Finally, we study its behavior across test-time scaling axes: we found scaling serial refinement more beneficial than parallel sampling. In particular, when given more refinement turns, Kevin shows a higher rate of improvement. 

**Abstract (ZH)**: GPU内核编写是一项具有挑战性的工作，对于AI系统的效率至关重要。它也是高度迭代的：领域专家通过执行反馈编写代码并提高性能。此外，它提供了可验证的奖励，如正确性和加速比，使其成为一个自然的应用强化学习（RL）的环境。为了将这一过程的迭代性质明确地融入训练中，我们开发了一个灵活的多轮RL方法，以解决实际场景中遇到的独特挑战，如学习长轨迹和跨轮次的有效奖励归因。我们介绍了K(ernel D)evin（Kevin），这是第一个使用多轮RL训练的CUDA内核生成和优化模型。在我们的评估设置中，Kevin在生成内核的正确性和平均加速比方面显著优于其基模型（QwQ-32B），并在纯CUDA中将生成内核的正确性从56%提高到82%，平均加速比从基线（PyTorch Eager）的0.53倍提高到1.10倍，并超越了前沿模型o4-mini（0.78倍）。最后，我们研究了其在测试时扩展会话轴的行为：我们发现串行细化比并行采样更有益，特别是在获得更多细化轮次时，Kevin显示出更高的改进率。 

---
# RaDL: Relation-aware Disentangled Learning for Multi-Instance Text-to-Image Generation 

**Title (ZH)**: 基于关系感知的解耦学习多实例文本到图像生成 

**Authors**: Geon Park, Seon Bin Kim, Gunho Jung, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.11947)  

**Abstract**: With recent advancements in text-to-image (T2I) models, effectively generating multiple instances within a single image prompt has become a crucial challenge. Existing methods, while successful in generating positions of individual instances, often struggle to account for relationship discrepancy and multiple attributes leakage. To address these limitations, this paper proposes the relation-aware disentangled learning (RaDL) framework. RaDL enhances instance-specific attributes through learnable parameters and generates relation-aware image features via Relation Attention, utilizing action verbs extracted from the global prompt. Through extensive evaluations on benchmarks such as COCO-Position, COCO-MIG, and DrawBench, we demonstrate that RaDL outperforms existing methods, showing significant improvements in positional accuracy, multiple attributes consideration, and the relationships between instances. Our results present RaDL as the solution for generating images that consider both the relationships and multiple attributes of each instance within the multi-instance image. 

**Abstract (ZH)**: 基于文本到图像模型的近期进展，有效地在单个图像提示中生成多个实例已成为一个关键挑战。现有方法虽然在生成单个实例的位置方面取得成功，但在处理关系差异和多属性泄漏方面往往存在局限。为了解决这些限制，本文提出了关系感知分离学习（RaDL）框架。RaDL 通过可学习参数增强实例特定属性，并利用从全局提示中提取的动作动词通过关系注意力生成关系感知的图像特征。通过在 COCO-Position、COCO-MIG 和 DrawBench 等基准上的广泛评估，我们展示了 RaDL 在位置准确性、多属性考虑及实例间关系方面优于现有方法。我们的结果表明，RaDL 是生成同时考虑每个实例关系和多属性的多实例图像的解决方案。 

---
# Effective Fine-Tuning of Vision Transformers with Low-Rank Adaptation for Privacy-Preserving Image Classification 

**Title (ZH)**: 有效细调视觉变换器以实现隐私保护图像分类的低秩适应方法 

**Authors**: Haiwei Lin, Shoko Imaizumi, Hitoshi Kiya  

**Link**: [PDF](https://arxiv.org/pdf/2507.11943)  

**Abstract**: We propose a low-rank adaptation method for training privacy-preserving vision transformer (ViT) models that efficiently freezes pre-trained ViT model weights. In the proposed method, trainable rank decomposition matrices are injected into each layer of the ViT architecture, and moreover, the patch embedding layer is not frozen, unlike in the case of the conventional low-rank adaptation methods. The proposed method allows us not only to reduce the number of trainable parameters but to also maintain almost the same accuracy as that of full-time tuning. 

**Abstract (ZH)**: 我们提出了一种低秩适应方法，用于训练保护隐私的视觉变换器（ViT）模型，该方法可以有效地冻结预训练的ViT模型权重。在所提出的方法中，可训练的秩分解矩阵被注入到ViT架构的每一层中，而且 Patch 嵌入层没有被冻结，这与传统低秩适应方法不同。所提出的方法不仅减少了可训练参数的数量，还几乎保持了全时段调优的相同准确率。 

---
# POLYCHARTQA: Benchmarking Large Vision-Language Models with Multilingual Chart Question Answering 

**Title (ZH)**: 多模态图表问答基准：多语言视觉-语言模型评估 

**Authors**: Yichen Xu, Liangyu Chen, Liang Zhang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.11939)  

**Abstract**: Charts are a universally adopted medium for interpreting and communicating data. However, existing chart understanding benchmarks are predominantly English-centric, limiting their accessibility and applicability to global audiences. In this paper, we present PolyChartQA, the first large-scale multilingual chart question answering benchmark covering 22,606 charts and 26,151 question-answering pairs across 10 diverse languages. PolyChartQA is built using a decoupled pipeline that separates chart data from rendering code, allowing multilingual charts to be flexibly generated by simply translating the data and reusing the code. We leverage state-of-the-art LLM-based translation and enforce rigorous quality control in the pipeline to ensure the linguistic and semantic consistency of the generated multilingual charts. PolyChartQA facilitates systematic evaluation of multilingual chart understanding. Experiments on both open- and closed-source large vision-language models reveal a significant performance gap between English and other languages, especially low-resource ones with non-Latin scripts. This benchmark lays a foundation for advancing globally inclusive vision-language models. 

**Abstract (ZH)**: 图表是一种广泛采用的数据解释与交流工具。然而，现有的图表理解基准主要以英语为中心，限制了其对全球受众的访问性和适用性。本文介绍了PolyChartQA，这是一个涵盖22,606幅图表和26,151个问答对、包含10种不同语言的首个大规模多语言图表问答基准。PolyChartQA 使用了一个解耦的管道，将图表数据与渲染代码分离，允许通过简单翻译数据并重用代码来灵活生成多语言图表。我们利用最先进的基于LLM的翻译，并在管道中实施严格的质量控制，以确保生成的多语言图表在语言和语义上的一致性。PolyChartQA 促进了多语言图表理解的系统评估。在开源和闭源大型视觉-语言模型上的实验揭示了英语与其他语言之间，尤其是使用非拉丁字符的低资源语言之间存在显著的表现差异。这个基准为推动全球包容性的视觉-语言模型的发展奠定了基础。 

---
# A Survey of Deep Learning for Geometry Problem Solving 

**Title (ZH)**: 深度学习在几何问题求解领域的研究综述 

**Authors**: Jianzhe Ma, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.11936)  

**Abstract**: Geometry problem solving is a key area of mathematical reasoning, which is widely involved in many important fields such as education, mathematical ability assessment of artificial intelligence, and multimodal ability assessment. In recent years, the rapid development of deep learning technology, especially the rise of multimodal large language models, has triggered a widespread research boom. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our goal is to provide a comprehensive and practical reference of deep learning for geometry problem solving to promote further developments in this field. We create a continuously updated list of papers on GitHub: this https URL. 

**Abstract (ZH)**: 深度学习在几何问题求解中的应用综述：包括（i）几何问题求解相关任务的全面总结；（ii）相关深度学习方法的彻底审查；（iii）评价指标和方法的详细分析；以及（iv）当前挑战和未来方向的批判性讨论。我们的目标是为几何问题求解提供一个全面和实用的深度学习参考，促进该领域的进一步发展。我们在GitHub上创建了一个不断更新的论文列表：this https URL。 

---
# Native-AI Empowered Scalable Architectures and Solutions for Future Non-Terrestrial Networks: An Overview 

**Title (ZH)**: 基于本地人工智能赋能的面向未来非地基网络的可扩展架构与解决方案：概览 

**Authors**: Jikang Deng, Fizza Hassan, Hui Zhou, Saad Al-Ahmadi, Mohamed-Slim Alouini, Daniel B. Da Costa  

**Link**: [PDF](https://arxiv.org/pdf/2507.11935)  

**Abstract**: As the path toward 6G networks is being charted, the emerging applications have motivated evolutions of network architectures to realize the efficient, reliable, and flexible wireless networks. Among the potential architectures, the non-terrestrial network (NTN) and open radio access network (ORAN) have received increasing interest from both academia and industry. Although the deployment of NTNs ensures coverage, enhances spectral efficiency, and improves the resilience of wireless networks. The high altitude and mobility of NTN present new challenges in the development and operations (DevOps) lifecycle, hindering intelligent and scalable network management due to the lack of native artificial intelligence (AI) capability. With the advantages of ORAN in disaggregation, openness, virtualization, and intelligence, several works propose integrating ORAN principles into the NTN, focusing mainly on ORAN deployment options based on transparent and regenerative systems. However, a holistic view of how to effectively combine ORAN and NTN throughout the DevOps lifecycle is still missing, especially regarding how intelligent ORAN addresses the scalability challenges in NTN. Motivated by this, in this paper, we first provide the background knowledge about ORAN and NTN, outline the state-of-the-art research on ORAN for NTNs, and present the DevOps challenges that motivate the adoption of ORAN solutions. We then propose the ORAN-based NTN framework, discussing its features and architectures in detail. These include the discussion about flexible fronthaul split, RAN intelligent controllers (RICs) enhancement for distributed learning, scalable deployment architecture, and multi-domain service management. Finally, the future research directions, including combinations of the ORAN-based NTN framework and other enabling technologies and schemes, as well as the candidate use cases, are highlighted. 

**Abstract (ZH)**: 随着向6G网络的发展路径被勾画，新兴应用促使网络架构演进以实现高效、可靠和灵活的无线网络。在潜在架构中，非地基网络（NTN）和开放无线接入网络（ORAN）受到了学术界和产业界的广泛关注。尽管部署NTN确保了覆盖范围、提升了频谱效率并改善了无线网络的韧性，但NTN的高海拔和高移动性给开发和运维（DevOps）生命周期带来了新的挑战，由于缺乏内置的人工智能（AI）能力，这阻碍了智能和可扩展网络管理的实现。利用ORAN在解耦、开放性、虚拟化和智能化方面的优势，一些研究提出了将ORAN原则集成到NTN中的方案，主要集中在基于透明和再生系统的ORAN部署选项上。然而，有关如何在DevOps生命周期中有效地结合ORAN和NTN的全面观点仍然缺失，尤其是在智能ORAN如何应对NTN的可扩展性挑战方面。受此驱动，在本文中，我们首先提供关于ORAN和NTN的背景知识，概述了针对NTN的ORAN的最新研究，并阐述了促使采用ORAN解决方案的DevOps挑战。然后，我们提出了基于ORAN的NTN框架，详细讨论了其特点和架构，包括灵活前传分割、RICs增强以支持分布式学习、可扩展部署架构以及多域服务管理。最后，我们指出了未来研究的方向，包括基于ORAN的 NTN框架与其他使能技术和方案的结合，以及潜在的应用案例。 

---
# Spatial Frequency Modulation for Semantic Segmentation 

**Title (ZH)**: 空间频率调制用于语义分割 

**Authors**: Linwei Chen, Ying Fu, Lin Gu, Dezhi Zheng, Jifeng Dai  

**Link**: [PDF](https://arxiv.org/pdf/2507.11893)  

**Abstract**: High spatial frequency information, including fine details like textures, significantly contributes to the accuracy of semantic segmentation. However, according to the Nyquist-Shannon Sampling Theorem, high-frequency components are vulnerable to aliasing or distortion when propagating through downsampling layers such as strided-convolution. Here, we propose a novel Spatial Frequency Modulation (SFM) that modulates high-frequency features to a lower frequency before downsampling and then demodulates them back during upsampling. Specifically, we implement modulation through adaptive resampling (ARS) and design a lightweight add-on that can densely sample the high-frequency areas to scale up the signal, thereby lowering its frequency in accordance with the Frequency Scaling Property. We also propose Multi-Scale Adaptive Upsampling (MSAU) to demodulate the modulated feature and recover high-frequency information through non-uniform upsampling This module further improves segmentation by explicitly exploiting information interaction between densely and sparsely resampled areas at multiple scales. Both modules can seamlessly integrate with various architectures, extending from convolutional neural networks to transformers. Feature visualization and analysis confirm that our method effectively alleviates aliasing while successfully retaining details after demodulation. Finally, we validate the broad applicability and effectiveness of SFM by extending it to image classification, adversarial robustness, instance segmentation, and panoptic segmentation tasks. The code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 高频空间信息，包括纹理等精细细节，显著提高了语义分割的准确性。然而，根据奈奎斯特-香农采样定理，高频成分在经过卷积下采样层（如跨步卷积）传播时容易发生混叠或失真。为此，我们提出了一种新型的空间频率调制（SFM），在下采样前将高频特征调制至较低频率，然后在上采样时进行反调制。具体地，我们通过自适应重采样（ARS）实现调制，并设计了一个轻量级附加模块，密集采样高频区域以放大信号，从而根据频域缩放特性降低其频率。此外，我们提出了多尺度自适应上采样（MSAU），用于反调制调制特征并通过非均匀上采样恢复高频信息。该模块进一步通过在多尺度上显式利用密集和稀疏重采样区域之间的信息交互来改善分割。这两个模块可以无缝集成到各种架构中，从卷积神经网络扩展到变压器。特征可视化和分析表明，我们的方法有效地缓解了混叠问题，并在反调制后成功保留了细节。最后，我们通过将其扩展到图像分类、对抗鲁棒性、实例分割和全景分割任务，验证了SFM的广泛应用性和有效性。代码可在\href{this https URL}{此链接}获得。 

---
# From Coarse to Nuanced: Cross-Modal Alignment of Fine-Grained Linguistic Cues and Visual Salient Regions for Dynamic Emotion Recognition 

**Title (ZH)**: 从粗略到细腻：细粒度语言线索与视觉显著区域的跨模态对齐及其在动态情感识别中的应用 

**Authors**: Yu Liu, Leyuan Qu, Hanlei Shi, Di Gao, Yuhua Zheng, Taihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.11892)  

**Abstract**: Dynamic Facial Expression Recognition (DFER) aims to identify human emotions from temporally evolving facial movements and plays a critical role in affective computing. While recent vision-language approaches have introduced semantic textual descriptions to guide expression recognition, existing methods still face two key limitations: they often underutilize the subtle emotional cues embedded in generated text, and they have yet to incorporate sufficiently effective mechanisms for filtering out facial dynamics that are irrelevant to emotional expression. To address these gaps, We propose GRACE, Granular Representation Alignment for Cross-modal Emotion recognition that integrates dynamic motion modeling, semantic text refinement, and token-level cross-modal alignment to facilitate the precise localization of emotionally salient spatiotemporal features. Our method constructs emotion-aware textual descriptions via a Coarse-to-fine Affective Text Enhancement (CATE) module and highlights expression-relevant facial motion through a motion-difference weighting mechanism. These refined semantic and visual signals are aligned at the token level using entropy-regularized optimal transport. Experiments on three benchmark datasets demonstrate that our method significantly improves recognition performance, particularly in challenging settings with ambiguous or imbalanced emotion classes, establishing new state-of-the-art (SOTA) results in terms of both UAR and WAR. 

**Abstract (ZH)**: 粒度表示对齐的跨模态情感识别（GRACE）：整合动态运动建模、语义文本精炼和标记级跨模态对齐 

---
# Interactive Hybrid Rice Breeding with Parametric Dual Projection 

**Title (ZH)**: 参数双投影驱动的互动水稻杂交育种 

**Authors**: Changjian Chen, Pengcheng Wang, Fei Lyu, Zhuo Tang, Li Yang, Long Wang, Yong Cai, Feng Yu, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.11848)  

**Abstract**: Hybrid rice breeding crossbreeds different rice lines and cultivates the resulting hybrids in fields to select those with desirable agronomic traits, such as higher yields. Recently, genomic selection has emerged as an efficient way for hybrid rice breeding. It predicts the traits of hybrids based on their genes, which helps exclude many undesired hybrids, largely reducing the workload of field cultivation. However, due to the limited accuracy of genomic prediction models, breeders still need to combine their experience with the models to identify regulatory genes that control traits and select hybrids, which remains a time-consuming process. To ease this process, in this paper, we proposed a visual analysis method to facilitate interactive hybrid rice breeding. Regulatory gene identification and hybrid selection naturally ensemble a dual-analysis task. Therefore, we developed a parametric dual projection method with theoretical guarantees to facilitate interactive dual analysis. Based on this dual projection method, we further developed a gene visualization and a hybrid visualization to verify the identified regulatory genes and hybrids. The effectiveness of our method is demonstrated through the quantitative evaluation of the parametric dual projection method, identified regulatory genes and desired hybrids in the case study, and positive feedback from breeders. 

**Abstract (ZH)**: 杂交水稻育种将不同的水稻品系进行杂交，并在田间种植杂交后代，选择具有desired agronomic traits（如高产量）的水稻。近年来，基因组选择已成为杂交水稻育种的有效方法。它基于基因预测杂交后代的性状，有助于排除许多不 desired的杂交后代，大大减少了田间种植的工作量。然而，由于基因组预测模型的准确性有限，育种者仍需结合经验和模型来识别控制性状的调控基因并选择杂交后代，这仍然是一个耗时的过程。为了简化这一过程，本文提出了一种可视化分析方法，以促进交互式杂交水稻育种。调控基因识别和杂交选择自然构成一项双重分析任务。因此，我们开发了一种带有理论保证的参数双重投影方法，以促进交互式双重分析。基于这一双重投影方法，我们进一步开发了一种基因可视化和杂交可视化，以验证识别出的调控基因和理想的杂交后代。通过案例研究中的定量评估、识别出的调控基因和期望的杂交后代的有效性，以及育种者的正面反馈，证明了我们方法的有效性。 

---
# MNIST-Gen: A Modular MNIST-Style Dataset Generation Using Hierarchical Semantics, Reinforcement Learning, and Category Theory 

**Title (ZH)**: MNIST-Gen: 基于层次语义、强化学习和范畴论的模块化MNIST样式数据集生成 

**Authors**: Pouya Shaeri, Arash Karimi, Ariane Middel  

**Link**: [PDF](https://arxiv.org/pdf/2507.11821)  

**Abstract**: Neural networks are often benchmarked using standard datasets such as MNIST, FashionMNIST, or other variants of MNIST, which, while accessible, are limited to generic classes such as digits or clothing items. For researchers working on domain-specific tasks, such as classifying trees, food items, or other real-world objects, these data sets are insufficient and irrelevant. Additionally, creating and publishing a custom dataset can be time consuming, legally constrained, or beyond the scope of individual projects. We present MNIST-Gen, an automated, modular, and adaptive framework for generating MNIST-style image datasets tailored to user-specified categories using hierarchical semantic categorization. The system combines CLIP-based semantic understanding with reinforcement learning and human feedback to achieve intelligent categorization with minimal manual intervention. Our hierarchical approach supports complex category structures with semantic characteristics, enabling fine-grained subcategorization and multiple processing modes: individual review for maximum control, smart batch processing for large datasets, and fast batch processing for rapid creation. Inspired by category theory, MNIST-Gen models each data transformation stage as a composable morphism, enhancing clarity, modularity, and extensibility. As proof of concept, we generate and benchmark two novel datasets-\textit{Tree-MNIST} and \textit{Food-MNIST}-demonstrating MNIST-Gen's utility for producing task-specific evaluation data while achieving 85\% automatic categorization accuracy and 80\% time savings compared to manual approaches. 

**Abstract (ZH)**: MNIST-Gen：一种基于层次语义分类的自动、模块化和适应性强的MNIST样式图像数据集生成框架 

---
# The Evolving Role of Large Language Models in Scientific Innovation: Evaluator, Collaborator, and Scientist 

**Title (ZH)**: 大型语言模型在科学发展创新中的 evolving role: 评估者、合作者和科学家 

**Authors**: Haoxuan Zhang, Ruochi Li, Yang Zhang, Ting Xiao, Jiangping Chen, Junhua Ding, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11810)  

**Abstract**: Scientific innovation is undergoing a paradigm shift driven by the rapid advancement of Large Language Models (LLMs). As science faces mounting challenges including information overload, disciplinary silos, and diminishing returns on conventional research methods, LLMs are emerging as powerful agents capable not only of enhancing scientific workflows but also of participating in and potentially leading the innovation process. Existing surveys mainly focus on different perspectives, phrases, and tasks in scientific research and discovery, while they have limitations in understanding the transformative potential and role differentiation of LLM. This survey proposes a comprehensive framework to categorize the evolving roles of LLMs in scientific innovation across three hierarchical levels: Evaluator, Collaborator, and Scientist. We distinguish between LLMs' contributions to structured scientific research processes and open-ended scientific discovery, thereby offering a unified taxonomy that clarifies capability boundaries, evaluation criteria, and human-AI interaction patterns at each level. Through an extensive analysis of current methodologies, benchmarks, systems, and evaluation metrics, this survey delivers an in-depth and systematic synthesis on LLM-driven scientific innovation. We present LLMs not only as tools for automating existing processes, but also as catalysts capable of reshaping the epistemological foundations of science itself. This survey offers conceptual clarity, practical guidance, and theoretical foundations for future research, while also highlighting open challenges and ethical considerations in the pursuit of increasingly autonomous AI-driven science. Resources related to this survey can be accessed on GitHub at: this https URL. 

**Abstract (ZH)**: 大规模语言模型驱动的科学创新正经历一场范式转变。面对信息 overload、学科孤岛和传统研究方法日益减少的回报等日益严峻的挑战，大规模语言模型(Large Language Models, LLMs)正演化为不仅能提升科学工作流程，还能参与甚至引领创新过程的强大代理。现有综述主要从不同视角关注科学研究和发现中的短语和任务，但对LLMs的变革潜力及其角色差异理解有限。本文综述提出一个全面框架，从三个层级分类LLMs在科学创新中的演变角色：评估员、合作者和科学家。通过区分LLMs对结构化科学研究过程和开放性科学发现的贡献，本文提供了一个统一的分类体系，明确了各层级的能力边界、评估标准和人-机互动模式。通过对当前方法、基准、系统和评估指标的广泛分析，本文综述提供了一篇深入且系统的LLM驱动科学创新综述。我们不仅将LLMs视为自动化现有流程的工具，还将之视为能重塑科学知识论基础的催化剂。本文综述提供概念清晰度、实际指导和理论基础，同时突出自主人工智能驱动科学不断增长的挑战和伦理考量。与此综述相关的资源可在GitHub上访问：this https URL。 

---
# Tracing Facts or just Copies? A critical investigation of the Competitions of Mechanisms in Large Language Models 

**Title (ZH)**: 追踪事实还是仅仅复制？对大型语言模型机制竞争的批判性调查 

**Authors**: Dante Campregher, Yanxu Chen, Sander Hoffman, Maria Heuss  

**Link**: [PDF](https://arxiv.org/pdf/2507.11809)  

**Abstract**: This paper presents a reproducibility study examining how Large Language Models (LLMs) manage competing factual and counterfactual information, focusing on the role of attention heads in this process. We attempt to reproduce and reconcile findings from three recent studies by Ortu et al., Yu, Merullo, and Pavlick and McDougall et al. that investigate the competition between model-learned facts and contradictory context information through Mechanistic Interpretability tools. Our study specifically examines the relationship between attention head strength and factual output ratios, evaluates competing hypotheses about attention heads' suppression mechanisms, and investigates the domain specificity of these attention patterns. Our findings suggest that attention heads promoting factual output do so via general copy suppression rather than selective counterfactual suppression, as strengthening them can also inhibit correct facts. Additionally, we show that attention head behavior is domain-dependent, with larger models exhibiting more specialized and category-sensitive patterns. 

**Abstract (ZH)**: 本研究呈现了一项再现性考察，探讨大型语言模型（LLMs）在管理事实和假设性信息冲突时的作用，重点分析了注意力头在这一过程中的角色。我们尝试重现并调和Ortu等人、Yu、Merullo和Pavlick以及McDougall等人最近三篇研究中的发现，这些研究通过机制可解释性工具探究模型学习的事实与矛盾性上下文信息之间的竞争。本研究具体考察了注意力头强度与事实输出比例之间的关系，评估了关于注意力头抑制机制的竞争假设，并分析了这些注意力模式的专业性和领域特定性。我们的研究结果表明，促进事实输出的注意力头通过一般性的复制抑制而非选择性的假设性抑制来发挥作用，因为增强它们也会抑制正确的事实。此外，我们展示了注意力头的行为具有领域依赖性，较大的模型表现出更加专业化和类别敏感的模式。 

---
# CLID-MU: Cross-Layer Information Divergence Based Meta Update Strategy for Learning with Noisy Labels 

**Title (ZH)**: CLID-MU：基于跨层信息分歧的元更新策略学习 noisy 标签数据 

**Authors**: Ruofan Hu, Dongyu Zhang, Huayi Zhang, Elke Rundensteiner  

**Link**: [PDF](https://arxiv.org/pdf/2507.11807)  

**Abstract**: Learning with noisy labels (LNL) is essential for training deep neural networks with imperfect data. Meta-learning approaches have achieved success by using a clean unbiased labeled set to train a robust model. However, this approach heavily depends on the availability of a clean labeled meta-dataset, which is difficult to obtain in practice. In this work, we thus tackle the challenge of meta-learning for noisy label scenarios without relying on a clean labeled dataset. Our approach leverages the data itself while bypassing the need for labels. Building on the insight that clean samples effectively preserve the consistency of related data structures across the last hidden and the final layer, whereas noisy samples disrupt this consistency, we design the Cross-layer Information Divergence-based Meta Update Strategy (CLID-MU). CLID-MU leverages the alignment of data structures across these diverse feature spaces to evaluate model performance and use this alignment to guide training. Experiments on benchmark datasets with varying amounts of labels under both synthetic and real-world noise demonstrate that CLID-MU outperforms state-of-the-art methods. The code is released at this https URL. 

**Abstract (ZH)**: 基于嘈杂标签的元学习：不依赖干净标签数据的交叉层信息分歧元更新策略 

---
# Fragment size density estimator for shrinkage-induced fracture based on a physics-informed neural network 

**Title (ZH)**: 基于物理支配神经网络的收缩诱导破裂尺寸密度估计 

**Authors**: Shin-ichi Ito  

**Link**: [PDF](https://arxiv.org/pdf/2507.11799)  

**Abstract**: This paper presents a neural network (NN)-based solver for an integro-differential equation that models shrinkage-induced fragmentation. The proposed method directly maps input parameters to the corresponding probability density function without numerically solving the governing equation, thereby significantly reducing computational costs. Specifically, it enables efficient evaluation of the density function in Monte Carlo simulations while maintaining accuracy comparable to or even exceeding that of conventional finite difference schemes. Validatation on synthetic data demonstrates both the method's computational efficiency and predictive reliability. This study establishes a foundation for the data-driven inverse analysis of fragmentation and suggests the potential for extending the framework beyond pre-specified model structures. 

**Abstract (ZH)**: 基于神经网络的积分微分方程碎裂建模求解器：无数值求解 governing 方程直接映射输入参数至对应的概率密度函数，显著降低计算成本 

---
# Foundation Models for Brain Signals: A Critical Review of Current Progress and Future Directions 

**Title (ZH)**: 脑信号的基石模型：现有进展与未来方向的批判性review 

**Authors**: Gayal Kuruppu, Neeraj Wagh, Yogatheesan Varatharajah  

**Link**: [PDF](https://arxiv.org/pdf/2507.11783)  

**Abstract**: Patterns of electrical brain activity recorded via electroencephalography (EEG) offer immense value for scientific and clinical investigations. The inability of supervised EEG encoders to learn robust EEG patterns and their over-reliance on expensive signal annotations have sparked a transition towards general-purpose self-supervised EEG encoders, i.e., EEG foundation models (EEG-FMs), for robust and scalable EEG feature extraction. However, the real-world readiness of early EEG-FMs and the rubric for long-term research progress remain unclear. A systematic and comprehensive review of first-generation EEG-FMs is therefore necessary to understand the current state-of-the-art and identify key directions for future EEG-FMs. To that end, this study reviews 10 early EEG-FMs and presents a critical synthesis of their methodology, empirical findings, and outstanding research gaps. We find that most EEG-FMs adopt a sequence-based modeling scheme that relies on transformer-based backbones and the reconstruction of masked sequences for self-supervision. However, model evaluations remain heterogeneous and largely limited, making it challenging to assess their practical off-the-shelf utility. In addition to adopting standardized and realistic evaluations, future work should demonstrate more substantial scaling effects and make principled and trustworthy choices throughout the EEG representation learning pipeline. We believe that developing benchmarks, software tools, technical methodologies, and applications in collaboration with domain experts may further advance the translational utility and real-world adoption of EEG-FMs. 

**Abstract (ZH)**: 基于电生理活动的早期脑机接口基础模型：现状与未来方向 

---
# Predicting Delayed Trajectories Using Network Features: A Study on the Dutch Railway Network 

**Title (ZH)**: 使用网络特征预测延迟轨迹：以荷兰铁路网络为例的研究 

**Authors**: Merel Kampere, Ali Mohammed Mansoor Alsahag  

**Link**: [PDF](https://arxiv.org/pdf/2507.11776)  

**Abstract**: The Dutch railway network is one of the busiest in the world, with delays being a prominent concern for the principal passenger railway operator NS. This research addresses a gap in delay prediction studies within the Dutch railway network by employing an XGBoost Classifier with a focus on topological features. Current research predominantly emphasizes short-term predictions and neglects the broader network-wide patterns essential for mitigating ripple effects. This research implements and improves an existing methodology, originally designed to forecast the evolution of the fast-changing US air network, to predict delays in the Dutch Railways. By integrating Node Centrality Measures and comparing multiple classifiers like RandomForest, DecisionTree, GradientBoosting, AdaBoost, and LogisticRegression, the goal is to predict delayed trajectories. However, the results reveal limited performance, especially in non-simultaneous testing scenarios, suggesting the necessity for more context-specific adaptations. Regardless, this research contributes to the understanding of transportation network evaluation and proposes future directions for developing more robust predictive models for delays. 

**Abstract (ZH)**: 荷兰铁路网络是世界上 busiest 的之一，对于主要的旅客铁路运营商 NS 来说，延误是一个重要的关注点。本研究通过采用基于拓扑特征的 XGBoost 分类器填补了荷兰铁路网络延误预测研究中的空白。当前的研究主要侧重于短期预测，并忽视了对缓解连锁反应至关重要的广泛网络模式。本研究将一种先前设计用于预测快速变化的美国航空网络演变的方法实施并改进，以预测荷兰铁路的延误。通过整合节点中心性度量并比较多种分类器（如随机森林、决策树、梯度提升、自适应提升和支持向量机），本研究旨在预测延误轨迹。然而，结果表明，在非同时测试场景中性能有限，这表明需要更具体的上下文适应性调整。尽管如此，本研究为交通网络评估的理解做出了贡献，并提出了开发更稳健的延误预测模型的未来方向。 

---
# Challenges in GenAI and Authentication: a scoping review 

**Title (ZH)**: GenAI和身份认证领域的挑战：一项范围审查 

**Authors**: Wesley dos Reis Bezerra, Lais Machado Bezerra, Carlos Becker Westphall  

**Link**: [PDF](https://arxiv.org/pdf/2507.11775)  

**Abstract**: Authentication and authenticity have been a security challenge since the beginning of information sharing, especially in the context of digital information. With the advancement of generative artificial intelligence, these challenges have evolved, demanding a more up-to-date analysis of their impacts on society and system security. This work presents a scoping review that analyzed 88 documents from the IEEExplorer, Scopus, and ACM databases, promoting an analysis of the resulting portfolio through six guiding questions focusing on the most relevant work, challenges, attack surfaces, threats, proposed solutions, and gaps. Finally, the portfolio articles are analyzed through this guiding research lens and also receive individualized analysis. The results consistently outline the challenges, gaps, and threats related to images, text, audio, and video, thereby supporting new research in the areas of authentication and generative artificial intelligence. 

**Abstract (ZH)**: 自信息共享之初，认证与真实性就是安全挑战，尤其是在数字信息的背景下。随着生成式人工智能的发展，这些挑战已演变，迫切需要对它们对社会和系统安全的影响进行更最新的分析。本文通过分析IEEEExplorer、Scopus和ACM数据库中的88份文档，提出了一项范围审查，通过六项引导问题聚焦于最具相关性的工作、挑战、攻击面、威胁、提出的解决方案以及空白，最终通过这个引导研究视角来分析这些文献，并进行个别分析。结果一致地概述了与图像、文本、音频和视频相关的挑战、空白和威胁，从而支持在认证和生成式人工智能领域的新的研究。 

---
# Small Data Explainer -- The impact of small data methods in everyday life 

**Title (ZH)**: 小数据解释者——小数据方法对日常生活的影响 

**Authors**: Maren Hackenberg, Sophia G. Connor, Fabian Kabus, June Brawner, Ella Markham, Mahi Hardalupas, Areeq Chowdhury, Rolf Backofen, Anna Köttgen, Angelika Rohde, Nadine Binder, Harald Binder, Collaborative Research Center 1597 Small Data  

**Link**: [PDF](https://arxiv.org/pdf/2507.11773)  

**Abstract**: The emergence of breakthrough artificial intelligence (AI) techniques has led to a renewed focus on how small data settings, i.e., settings with limited information, can benefit from such developments. This includes societal issues such as how best to include under-represented groups in data-driven policy and decision making, or the health benefits of assistive technologies such as wearables. We provide a conceptual overview, in particular contrasting small data with big data, and identify common themes from exemplary case studies and application areas. Potential solutions are described in a more detailed technical overview of current data analysis and modelling techniques, highlighting contributions from different disciplines, such as knowledge-driven modelling from statistics and data-driven modelling from computer science. By linking application settings, conceptual contributions and specific techniques, we highlight what is already feasible and suggest what an agenda for fully leveraging small data might look like. 

**Abstract (ZH)**: 突破性人工智能技术的出现促使人们重新关注小数据环境下的潜在益处，特别是这些技术如何在信息有限的情况下发挥作用。这包括社会问题，如如何更好地将代表性不足的群体纳入数据驱动的政策和决策，或可穿戴设备等辅助技术在健康方面的益处。本文提供了一个概念性概述，尤其是小数据与大数据的对比，并从典型案例和应用领域中识别出共同主题。在更详细的技术概述中描述了潜在解决方案，强调来自不同学科的贡献，如统计学的知识驱动建模和计算机科学的数据驱动建模。通过将应用情境、概念性贡献和技术具体方法联系起来，本文突显了当前已实现的能力，并提出了充分利用小数据的议程可能是什么样的。 

---
# Beyond Task-Specific Reasoning: A Unified Conditional Generative Framework for Abstract Visual Reasoning 

**Title (ZH)**: 超越任务特定推理：一种统一的条件生成框架用于抽象视觉推理 

**Authors**: Fan Shi, Bin Li, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2507.11761)  

**Abstract**: Abstract visual reasoning (AVR) enables humans to quickly discover and generalize abstract rules to new scenarios. Designing intelligent systems with human-like AVR abilities has been a long-standing topic in the artificial intelligence community. Deep AVR solvers have recently achieved remarkable success in various AVR tasks. However, they usually use task-specific designs or parameters in different tasks. In such a paradigm, solving new tasks often means retraining the model, and sometimes retuning the model architectures, which increases the cost of solving AVR problems. In contrast to task-specific approaches, this paper proposes a novel Unified Conditional Generative Solver (UCGS), aiming to address multiple AVR tasks in a unified framework. First, we prove that some well-known AVR tasks can be reformulated as the problem of estimating the predictability of target images in problem panels. Then, we illustrate that, under the proposed framework, training one conditional generative model can solve various AVR tasks. The experiments show that with a single round of multi-task training, UCGS demonstrates abstract reasoning ability across various AVR tasks. Especially, UCGS exhibits the ability of zero-shot reasoning, enabling it to perform abstract reasoning on problems from unseen AVR tasks in the testing phase. 

**Abstract (ZH)**: 统一条件生成求解器（UCGS）：解决多种抽象视觉推理任务的统一框架 

---
# Survey of Genetic and Differential Evolutionary Algorithm Approaches to Search Documents Based On Semantic Similarity 

**Title (ZH)**: 基于语义相似性的文档搜索中遗传和差分进化算法综述 

**Authors**: Chandrashekar Muniyappa, Eunjin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.11751)  

**Abstract**: Identifying similar documents within extensive volumes of data poses a significant challenge. To tackle this issue, researchers have developed a variety of effective distributed computing techniques. With the advancement of computing power and the rise of big data, deep neural networks and evolutionary computing algorithms such as genetic algorithms and differential evolution algorithms have achieved greater success. This survey will explore the most recent advancements in the search for documents based on their semantic text similarity, focusing on genetic and differential evolutionary computing algorithms. 

**Abstract (ZH)**: 在大量数据中识别相似文档面临显著挑战。为应对这一问题，研究人员开发了多种有效的分布式计算技术。随着计算能力的提升和大数据的兴起，基于语义文本相似性的文档搜索取得了更大进展，尤其得益于遗传算法和差分进化算法等进化计算算法的应用。本综述将探讨这些算法在基于语义文本相似性的文档搜索领域的最新进展。 

---
# CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks 

**Title (ZH)**: CRABS: 一种语法-语义夹持策略，用于界定大规模语言模型对Python笔记本的解释 

**Authors**: Meng Li, Timothy M. McPhillips, Dingmin Wang, Shin-Rong Tsai, Bertram Ludäscher  

**Link**: [PDF](https://arxiv.org/pdf/2507.11742)  

**Abstract**: Recognizing the information flows and operations comprising data science and machine learning Python notebooks is critical for evaluating, reusing, and adapting notebooks for new tasks. Investigating a notebook via re-execution often is impractical due to the challenges of resolving data and software dependencies. While Large Language Models (LLMs) pre-trained on large codebases have demonstrated effectiveness in understanding code without running it, we observe that they fail to understand some realistic notebooks due to hallucinations and long-context challenges. To address these issues, we propose a notebook understanding task yielding an information flow graph and corresponding cell execution dependency graph for a notebook, and demonstrate the effectiveness of a pincer strategy that uses limited syntactic analysis to assist full comprehension of the notebook using an LLM. Our Capture and Resolve Assisted Bounding Strategy (CRABS) employs shallow syntactic parsing and analysis of the abstract syntax tree (AST) to capture the correct interpretation of a notebook between lower and upper estimates of the inter-cell I/O sets, then uses an LLM to resolve remaining ambiguities via cell-by-cell zero-shot learning, thereby identifying the true data inputs and outputs of each cell. We evaluate and demonstrate the effectiveness of our approach using an annotated dataset of 50 representative, highly up-voted Kaggle notebooks that together represent 3454 actual cell inputs and outputs. The LLM correctly resolves 1397 of 1425 (98%) ambiguities left by analyzing the syntactic structure of these notebooks. Across 50 notebooks, CRABS achieves average F1 scores of 98% identifying cell-to-cell information flows and 99% identifying transitive cell execution dependencies. 

**Abstract (ZH)**: 识别数据科学和机器学习Python笔记本中的信息流和操作对于评估、重用和适应新任务的笔记本至关重要。通过对笔记本进行重执行来调查笔记本通常由于数据和软件依赖性解决的挑战而不可行。虽然预训练在大型代码库上的大规模语言模型在理解代码方面表现出有效性，但它们在理解一些实际笔记本时会出现幻觉和长上下文挑战。为了解决这些问题，我们提出了一种笔记本理解任务，生成笔记本的信息流图和相应的单元执行依赖图，并展示了结合有限的句法分析和大规模语言模型逐细胞零样本学习以辅助全面理解笔记本的夹钳策略的有效性。我们的捕获和解决辅助边界策略（CRABS）采用浅层句法解析和抽象语法树（AST）的分析，捕捉单元之间输入输出集的上下估计之间的正确解释，然后使用大规模语言模型通过逐单元零样本学习解决剩余的歧义，从而识别每个单元的真实数据输入和输出。我们使用50个代表性和高度投票的Kaggle笔记本的数据集进行评估和展示效果，这些笔记本共同代表了3454个实际单元输入和输出。对于这些笔记本的句法结构分析后剩余的1397个（98%）歧义，大规模语言模型正确解决了1425个。在50个笔记本上，CRABS平均信息流识别的F1得分为98%，推断传递单元执行依赖的得分为99%。 

---
# Seeing the Signs: A Survey of Edge-Deployable OCR Models for Billboard Visibility Analysis 

**Title (ZH)**: 识读标识：边缘部署的OCR模型在户外广告可见性分析中的综述 

**Authors**: Maciej Szankin, Vidhyananth Venkatasamy, Lihang Ying  

**Link**: [PDF](https://arxiv.org/pdf/2507.11730)  

**Abstract**: Outdoor advertisements remain a critical medium for modern marketing, yet accurately verifying billboard text visibility under real-world conditions is still challenging. Traditional Optical Character Recognition (OCR) pipelines excel at cropped text recognition but often struggle with complex outdoor scenes, varying fonts, and weather-induced visual noise. Recently, multimodal Vision-Language Models (VLMs) have emerged as promising alternatives, offering end-to-end scene understanding with no explicit detection step. This work systematically benchmarks representative VLMs - including Qwen 2.5 VL 3B, InternVL3, and SmolVLM2 - against a compact CNN-based OCR baseline (PaddleOCRv4) across two public datasets (ICDAR 2015 and SVT), augmented with synthetic weather distortions to simulate realistic degradation. Our results reveal that while selected VLMs excel at holistic scene reasoning, lightweight CNN pipelines still achieve competitive accuracy for cropped text at a fraction of the computational cost-an important consideration for edge deployment. To foster future research, we release our weather-augmented benchmark and evaluation code publicly. 

**Abstract (ZH)**: 户外广告仍然是现代营销中重要的媒体渠道，但准确验证实际条件下广告牌文字的可见性仍然具有挑战性。传统光学字符识别（OCR）流水线在裁剪文本识别方面表现出色，但在处理复杂户外场景、变化的字体和天气引起的视觉噪声方面常常遇到困难。近年来，多模态视觉-语言模型（VLMs）作为一种有前途的替代方案出现，能够实现端到端的场景理解，无需显式的检测步骤。本研究系统性地将Qwen 2.5 VL 3B、InternVL3和SmolVLM2等代表性VLM与基于紧凑CNN的OCR baselime（PaddleOCRv4）在两个公开数据集（ICDAR 2015和SVT）上进行了基准测试，数据集经过合成天气失真增强以模拟真实的退化。我们的结果显示，虽然选定的VLM在整体场景推理方面表现出色，但轻量级的CNN流水线在计算成本仅为一小部分的情况下仍能获得竞争力的文字识别准确性，这对于边缘部署而言非常重要。为了促进未来的研究，我们公开发布了带有天气增强的基准测试和评估代码。 

---
# Globalization for Scalable Short-term Load Forecasting 

**Title (ZH)**: 全球化短时负荷预测-scalability方法 

**Authors**: Amirhossein Ahmadi, Hamidreza Zareipour, Henry Leung  

**Link**: [PDF](https://arxiv.org/pdf/2507.11729)  

**Abstract**: Forecasting load in power transmission networks is essential across various hierarchical levels, from the system level down to individual points of delivery (PoD). While intuitive and locally accurate, traditional local forecasting models (LFMs) face significant limitations, particularly in handling generalizability, overfitting, data drift, and the cold start problem. These methods also struggle with scalability, becoming computationally expensive and less efficient as the network's size and data volume grow. In contrast, global forecasting models (GFMs) offer a new approach to enhance prediction generalizability, scalability, accuracy, and robustness through globalization and cross-learning. This paper investigates global load forecasting in the presence of data drifts, highlighting the impact of different modeling techniques and data heterogeneity. We explore feature-transforming and target-transforming models, demonstrating how globalization, data heterogeneity, and data drift affect each differently. In addition, we examine the role of globalization in peak load forecasting and its potential for hierarchical forecasting. To address data heterogeneity and the balance between globality and locality, we propose separate time series clustering (TSC) methods, introducing model-based TSC for feature-transforming models and new weighted instance-based TSC for target-transforming models. Through extensive experiments on a real-world dataset of Alberta's electricity load, we demonstrate that global target-transforming models consistently outperform their local counterparts, especially when enriched with global features and clustering techniques. In contrast, global feature-transforming models face challenges in balancing local and global dynamics, often requiring TSC to manage data heterogeneity effectively. 

**Abstract (ZH)**: 在数据漂移情况下进行电力传输网络负荷的全球预报：不同建模技术与数据异质性的影响 

---
# Subgraph Generation for Generalizing on Out-of-Distribution Links 

**Title (ZH)**: 生成子图以泛化处理分布外链接 

**Authors**: Jay Revolinsky, Harry Shomer, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11710)  

**Abstract**: Graphs Neural Networks (GNNs) demonstrate high-performance on the link prediction (LP) task. However, these models often rely on all dataset samples being drawn from the same distribution. In addition, graph generative models (GGMs) show a pronounced ability to generate novel output graphs. Despite this, GGM applications remain largely limited to domain-specific tasks. To bridge this gap, we propose FLEX as a GGM framework which leverages two mechanism: (1) structurally-conditioned graph generation, and (2) adversarial co-training between an auto-encoder and GNN. As such, FLEX ensures structural-alignment between sample distributions to enhance link-prediction performance in out-of-distribution (OOD) scenarios. Notably, FLEX does not require expert knowledge to function in different OOD scenarios. Numerous experiments are conducted in synthetic and real-world OOD settings to demonstrate FLEX's performance-enhancing ability, with further analysis for understanding the effects of graph data augmentation on link structures. The source code is available here: this https URL. 

**Abstract (ZH)**: 基于图神经网络的结构条件化图生成框架FLEX在跨分布场景下的链接预测性能提升 

---
# Time series classification of satellite data using LSTM networks: an approach for predicting leaf-fall to minimize railroad traffic disruption 

**Title (ZH)**: 使用LSTM网络的卫星数据时间序列分类：一种预测落叶以最小化铁路交通中断的方法 

**Authors**: Hein de Wilde, Ali Mohammed Mansoor Alsahag, Pierre Blanchet  

**Link**: [PDF](https://arxiv.org/pdf/2507.11702)  

**Abstract**: Railroad traffic disruption as a result of leaf-fall cost the UK rail industry over 300 million per year and measures to mitigate such disruptions are employed on a large scale, with 1.67 million kilometers of track being treated in the UK in 2021 alone. Therefore, the ability to anticipate the timing of leaf-fall would offer substantial benefits for rail network operators, enabling the efficient scheduling of such mitigation measures. However, current methodologies for predicting leaf-fall exhibit considerable limitations in terms of scalability and reliability. This study endeavors to devise a prediction system that leverages specialized prediction methods and the latest satellite data sources to generate both scalable and reliable insights into leaf-fall timings. An LSTM network trained on ground-truth leaf-falling data combined with multispectral and meteorological satellite data demonstrated a root-mean-square error of 6.32 days for predicting the start of leaf-fall and 9.31 days for predicting the end of leaf-fall. The model, which improves upon previous work on the topic, offers promising opportunities for the optimization of leaf mitigation measures in the railway industry and the improvement of our understanding of complex ecological systems. 

**Abstract (ZH)**: 铁路因落叶导致的交通中断每年给英国铁路行业造成超过3亿英镑的损失，为减轻此类中断，英国2021年 alone 处理了167万 kilometers 的轨道。因此，能够预测落叶时机的能力将为铁路网络运营者带来显著益处，有助于高效安排此类缓解措施。然而，当前的落叶预测方法在可扩展性和可靠性方面存在显著局限。本研究旨在利用专门的预测方法和最新的卫星数据源，开发一种既能提供可扩展性和可靠性又能预测落叶时机的预测系统。使用地面真实落叶数据训练的LSTM网络结合多光谱和气象卫星数据，预测落叶开始和结束的时间分别显示了6.32天和9.31天的均方根误差。该模型在该领域的工作基础上有所改进，为铁路行业的落叶缓解措施优化和对复杂生态系统理解的提高提供了前景。 

---
# ExpliCIT-QA: Explainable Code-Based Image Table Question Answering 

**Title (ZH)**: ExpliCIT-QA: 可解释的基于代码的图像表格问答 

**Authors**: Maximiliano Hormazábal Lagos, Álvaro Bueno Sáez, Pedro Alonso Doval, Jorge Alcalde Vesteiro, Héctor Cerezo-Costas  

**Link**: [PDF](https://arxiv.org/pdf/2507.11694)  

**Abstract**: We present ExpliCIT-QA, a system that extends our previous MRT approach for tabular question answering into a multimodal pipeline capable of handling complex table images and providing explainable answers. ExpliCIT-QA follows a modular design, consisting of: (1) Multimodal Table Understanding, which uses a Chain-of-Thought approach to extract and transform content from table images; (2) Language-based Reasoning, where a step-by-step explanation in natural language is generated to solve the problem; (3) Automatic Code Generation, where Python/Pandas scripts are created based on the reasoning steps, with feedback for handling errors; (4) Code Execution to compute the final answer; and (5) Natural Language Explanation that describes how the answer was computed. The system is built for transparency and auditability: all intermediate outputs, parsed tables, reasoning steps, generated code, and final answers are available for inspection. This strategy works towards closing the explainability gap in end-to-end TableVQA systems. We evaluated ExpliCIT-QA on the TableVQA-Bench benchmark, comparing it with existing baselines. We demonstrated improvements in interpretability and transparency, which open the door for applications in sensitive domains like finance and healthcare where auditing results are critical. 

**Abstract (ZH)**: ExpliCIT-QA：一个扩展我们的先前MRT表格式问答方法的多模态解释性问答系统 

---
# Galaxy image simplification using Generative AI 

**Title (ZH)**: 使用生成式AI进行星系图像简化 

**Authors**: Sai Teja Erukude, Lior Shamir  

**Link**: [PDF](https://arxiv.org/pdf/2507.11692)  

**Abstract**: Modern digital sky surveys have been acquiring images of billions of galaxies. While these images often provide sufficient details to analyze the shape of the galaxies, accurate analysis of such high volumes of images requires effective automation. Current solutions often rely on machine learning annotation of the galaxy images based on a set of pre-defined classes. Here we introduce a new approach to galaxy image analysis that is based on generative AI. The method simplifies the galaxy images and automatically converts them into a ``skeletonized" form. The simplified images allow accurate measurements of the galaxy shapes and analysis that is not limited to a certain pre-defined set of classes. We demonstrate the method by applying it to galaxy images acquired by the DESI Legacy Survey. The code and data are publicly available. The method was applied to 125,000 DESI Legacy Survey images, and the catalog of the simplified images is publicly available. 

**Abstract (ZH)**: 现代数字天空调查正在获取数十亿星系的图像。虽然这些图像通常提供了足够的细节来分析星系的形状，但对如此大量图像的准确分析需要有效的自动化手段。当前的解决方案往往依赖于基于预定义类别的机器学习对星系图像进行标注。在这里，我们介绍了一种基于生成式AI的星系图像分析方法。该方法简化了星系图像，并自动将其转换为“骨架化”形式。简化的图像允许精确测量星系形状，并且分析不限于某些预定义的类别。我们通过将该方法应用于DESI遗留给星系图像来演示这种方法。该代码和数据均可公开获取。该方法应用于125,000张DESI遗留给星系图像，并公开发布了简化的图像目录。 

---
# PGT-I: Scaling Spatiotemporal GNNs with Memory-Efficient Distributed Training 

**Title (ZH)**: PGT-I：使用内存高效分布式训练扩展时空GNNs 

**Authors**: Seth Ockerman, Amal Gueroudji, Tanwi Mallick, Yixuan He, Line Pouchard, Robert Ross, Shivaram Venkataraman  

**Link**: [PDF](https://arxiv.org/pdf/2507.11683)  

**Abstract**: Spatiotemporal graph neural networks (ST-GNNs) are powerful tools for modeling spatial and temporal data dependencies. However, their applications have been limited primarily to small-scale datasets because of memory constraints. While distributed training offers a solution, current frameworks lack support for spatiotemporal models and overlook the properties of spatiotemporal data. Informed by a scaling study on a large-scale workload, we present PyTorch Geometric Temporal Index (PGT-I), an extension to PyTorch Geometric Temporal that integrates distributed data parallel training and two novel strategies: index-batching and distributed-index-batching. Our index techniques exploit spatiotemporal structure to construct snapshots dynamically at runtime, significantly reducing memory overhead, while distributed-index-batching extends this approach by enabling scalable processing across multiple GPUs. Our techniques enable the first-ever training of an ST-GNN on the entire PeMS dataset without graph partitioning, reducing peak memory usage by up to 89\% and achieving up to a 13.1x speedup over standard DDP with 128 GPUs. 

**Abstract (ZH)**: 基于图的时空神经网络（ST-GNNs）是建模时空数据依赖性的强大工具。然而，由于内存限制，它们的应用主要局限于小规模数据集。虽然分布式训练提供了一种解决方案，但现有的框架缺乏对时空模型的支持，忽略了时空数据的特性。基于大规模工作负载的扩展研究，我们提出了PyTorch Geometric Temporal Index（PGT-I），这是对PyTorch Geometric Temporal的扩展，集成了分布式数据并行训练以及两种新型策略：索引批量处理和分布式索引批量处理。我们的索引技术利用时空结构在运行时动态构建快照，显著减少了内存开销，而分布式索引批量处理进一步扩展了这一方法，实现了跨多个GPU的可扩展处理。我们的技术使得首次在无需图分区的情况下对整个PeMS数据集训练ST-GNN，峰值内存使用量最多减少89%，并比128块GPU上的标准DDP快13.1倍。 

---
# Partitioner Guided Modal Learning Framework 

**Title (ZH)**: 分区辅助模态学习框架 

**Authors**: Guimin Hu, Yi Xin, Lijie Hu, Zhihong Zhu, Hasti Seifi  

**Link**: [PDF](https://arxiv.org/pdf/2507.11661)  

**Abstract**: Multimodal learning benefits from multiple modal information, and each learned modal representations can be divided into uni-modal that can be learned from uni-modal training and paired-modal features that can be learned from cross-modal interaction. Building on this perspective, we propose a partitioner-guided modal learning framework, PgM, which consists of the modal partitioner, uni-modal learner, paired-modal learner, and uni-paired modal decoder. Modal partitioner segments the learned modal representation into uni-modal and paired-modal features. Modal learner incorporates two dedicated components for uni-modal and paired-modal learning. Uni-paired modal decoder reconstructs modal representation based on uni-modal and paired-modal features. PgM offers three key benefits: 1) thorough learning of uni-modal and paired-modal features, 2) flexible distribution adjustment for uni-modal and paired-modal representations to suit diverse downstream tasks, and 3) different learning rates across modalities and partitions. Extensive experiments demonstrate the effectiveness of PgM across four multimodal tasks and further highlight its transferability to existing models. Additionally, we visualize the distribution of uni-modal and paired-modal features across modalities and tasks, offering insights into their respective contributions. 

**Abstract (ZH)**: 多模态学习受益于多种模态信息，每种学习到的模态表示可以分为仅从单模训练中学习到的单模特征和通过跨模态交互学习到的配对模态特征。基于这一视角，我们提出了一种分区引导的模态学习框架PgM，该框架包括模态分区器、单模特征学习器、配对模态学习器和单配对模态解码器。模态分区器将学习到的模态表示分割为单模特征和配对模态特征。模态学习器包含两个专门用于单模和配对模态学习的组件。单配对模态解码器基于单模和配对模态特征重构模态表示。PgM提供了三个关键优势：1）全面学习单模和配对模态特征，2）灵活调整单模和配对模态表示的分布以适应各种下游任务，3）不同模态和分区的学习率。广泛实验证明了PgM在四种多模态任务中的有效性，并进一步突显了其对现有模型的迁移性。此外，我们还可视化了单模和配对模态特征在不同模态和任务中的分布，提供了它们各自贡献的见解。 

---
# Counting Answer Sets of Disjunctive Answer Set Programs 

**Title (ZH)**: 计算析取回答集程序的回答集个数 

**Authors**: Mohimenul Kabir, Supratik Chakraborty, Kuldeep S Meel  

**Link**: [PDF](https://arxiv.org/pdf/2507.11655)  

**Abstract**: Answer Set Programming (ASP) provides a powerful declarative paradigm for knowledge representation and reasoning. Recently, counting answer sets has emerged as an important computational problem with applications in probabilistic reasoning, network reliability analysis, and other domains. This has motivated significant research into designing efficient ASP counters. While substantial progress has been made for normal logic programs, the development of practical counters for disjunctive logic programs remains challenging.
We present SharpASP-SR, a novel framework for counting answer sets of disjunctive logic programs based on subtractive reduction to projected propositional model counting. Our approach introduces an alternative characterization of answer sets that enables efficient reduction while ensuring that intermediate representations remain of polynomial size. This allows SharpASP-SR to leverage recent advances in projected model counting technology. Through extensive experimental evaluation on diverse benchmarks, we demonstrate that SharpASP-SR significantly outperforms existing counters on instances with large answer set counts. Building on these results, we develop a hybrid counting approach that combines enumeration techniques with SharpASP-SR to achieve state-of-the-art performance across the full spectrum of disjunctive programs. 

**Abstract (ZH)**: 基于减法约简到投影命题模型计数的SharpASP-SR：可处理析取逻辑程序的答案集计数新框架 

---
# Tracing the Path to Grokking: Embeddings, Dropout, and Network Activation 

**Title (ZH)**: 追踪grokking之路：嵌入式表示、 Dropout与网络激活 

**Authors**: Ahmed Salah, David Yevick  

**Link**: [PDF](https://arxiv.org/pdf/2507.11645)  

**Abstract**: Grokking refers to delayed generalization in which the increase in test accuracy of a neural network occurs appreciably after the improvement in training accuracy This paper introduces several practical metrics including variance under dropout, robustness, embedding similarity, and sparsity measures, that can forecast grokking behavior. Specifically, the resilience of neural networks to noise during inference is estimated from a Dropout Robustness Curve (DRC) obtained from the variation of the accuracy with the dropout rate as the model transitions from memorization to generalization. The variance of the test accuracy under stochastic dropout across training checkpoints further exhibits a local maximum during the grokking. Additionally, the percentage of inactive neurons decreases during generalization, while the embeddings tend to a bimodal distribution independent of initialization that correlates with the observed cosine similarity patterns and dataset symmetries. These metrics additionally provide valuable insight into the origin and behaviour of grokking. 

**Abstract (ZH)**: Grokking延迟泛化现象中神经网络测试准确度在训练准确度提升之后显著提高的行为。本论文引入了若干实用的评价指标，包括丢弃dropout下的方差、鲁棒性、嵌入相似性和稀疏性度量，以预测grokking行为。具体而言，通过从模型从记忆过渡到泛化过程中准确度随dropout率变化得到的丢弃鲁棒性曲线（DRC）估计神经网络在推断过程中的抗噪性。训练检查点下测试准确度在dropout下的方差进一步在grokking期间显示出局部最大值。此外，在泛化过程中，不活跃神经元的比例下降，而嵌入趋向于与初始化无关、独立于数据集对称性的双模分布，这种分布与观察到的余弦相似性模式和数据集对称性相关。这些指标还提供了关于grokking的起源和行为的宝贵见解。 

---
# Interpretable Prediction of Lymph Node Metastasis in Rectal Cancer MRI Using Variational Autoencoders 

**Title (ZH)**: 使用变分自编码器的可解释性直肠癌MRI淋巴结转移预测 

**Authors**: Benjamin Keel, Aaron Quyn, David Jayne, Maryam Mohsin, Samuel D. Relton  

**Link**: [PDF](https://arxiv.org/pdf/2507.11638)  

**Abstract**: Effective treatment for rectal cancer relies on accurate lymph node metastasis (LNM) staging. However, radiological criteria based on lymph node (LN) size, shape and texture morphology have limited diagnostic accuracy. In this work, we investigate applying a Variational Autoencoder (VAE) as a feature encoder model to replace the large pre-trained Convolutional Neural Network (CNN) used in existing approaches. The motivation for using a VAE is that the generative model aims to reconstruct the images, so it directly encodes visual features and meaningful patterns across the data. This leads to a disentangled and structured latent space which can be more interpretable than a CNN. Models are deployed on an in-house MRI dataset with 168 patients who did not undergo neo-adjuvant treatment. The post-operative pathological N stage was used as the ground truth to evaluate model predictions. Our proposed model 'VAE-MLP' achieved state-of-the-art performance on the MRI dataset, with cross-validated metrics of AUC 0.86 +/- 0.05, Sensitivity 0.79 +/- 0.06, and Specificity 0.85 +/- 0.05. Code is available at: this https URL. 

**Abstract (ZH)**: 有效的直肠癌治疗依赖于准确的淋巴结转移（LNM）分期。然而，基于淋巴结（LN）大小、形状和纹理形态的放射学标准在诊断准确性方面有限。本研究中，我们探索使用变分自编码器（VAE）作为特征编码模型，以替代现有方法中使用的大型预训练卷积神经网络（CNN）。使用VAE的原因在于生成模型旨在重构图像，因此可以直接编码视觉特征和有意义的模式，从而产生更易于解释的分离和结构化的潜在空间。该模型在未接受新辅助治疗的168名患者的院内MRI数据集上部署，并使用术后病理N分期作为ground truth评估模型预测。我们提出的模型'VAE-MLP'在MRI数据集上实现了最先进的性能，交叉验证指标包括AUC 0.86 ± 0.05、敏感性0.79 ± 0.06和特异性0.85 ± 0.05。代码可在以下链接获取：this https URL。 

---
# JSQA: Speech Quality Assessment with Perceptually-Inspired Contrastive Pretraining Based on JND Audio Pairs 

**Title (ZH)**: JSQA：基于JND音频配对的知觉启发对比预训练的语音质量评估 

**Authors**: Junyi Fan, Donald Williamson  

**Link**: [PDF](https://arxiv.org/pdf/2507.11636)  

**Abstract**: Speech quality assessment (SQA) is often used to learn a mapping from a high-dimensional input space to a scalar that represents the mean opinion score (MOS) of the perceptual speech quality. Learning such a mapping is challenging for many reasons, but largely because MOS exhibits high levels of inherent variance due to perceptual and experimental-design differences. Many solutions have been proposed, but many approaches do not properly incorporate perceptual factors into their learning algorithms (beyond the MOS label), which could lead to unsatisfactory results. To this end, we propose JSQA, a two-stage framework that pretrains an audio encoder using perceptually-guided contrastive learning on just noticeable difference (JND) pairs, followed by fine-tuning for MOS prediction. We first generate pairs of audio data within JND levels, which are then used to pretrain an encoder to leverage perceptual quality similarity information and map it into an embedding space. The JND pairs come from clean LibriSpeech utterances that are mixed with background noise from CHiME-3, at different signal-to-noise ratios (SNRs). The encoder is later fine-tuned with audio samples from the NISQA dataset for MOS prediction. Experimental results suggest that perceptually-inspired contrastive pretraining significantly improves the model performance evaluated by various metrics when compared against the same network trained from scratch without pretraining. These findings suggest that incorporating perceptual factors into pretraining greatly contributes to the improvement in performance for SQA. 

**Abstract (ZH)**: 基于感知的对比预训练的语音质量评估（JSQA） 

---
# Cross-lingual Few-shot Learning for Persian Sentiment Analysis with Incremental Adaptation 

**Title (ZH)**: 跨语言少样本学习在增量适应下的波斯语情感分析 

**Authors**: Farideh Majidi, Ziaeddin Beheshtifard  

**Link**: [PDF](https://arxiv.org/pdf/2507.11634)  

**Abstract**: This research examines cross-lingual sentiment analysis using few-shot learning and incremental learning methods in Persian. The main objective is to develop a model capable of performing sentiment analysis in Persian using limited data, while getting prior knowledge from high-resource languages. To achieve this, three pre-trained multilingual models (XLM-RoBERTa, mDeBERTa, and DistilBERT) were employed, which were fine-tuned using few-shot and incremental learning approaches on small samples of Persian data from diverse sources, including X, Instagram, Digikala, Snappfood, and Taaghche. This variety enabled the models to learn from a broad range of contexts. Experimental results show that the mDeBERTa and XLM-RoBERTa achieved high performances, reaching 96% accuracy on Persian sentiment analysis. These findings highlight the effectiveness of combining few-shot learning and incremental learning with multilingual pre-trained models. 

**Abstract (ZH)**: 本研究使用少量学习和增量学习方法对波斯语进行跨语言情感分析，旨在开发一种能够在有限数据下进行波斯语情感分析的模型，并从中获得高资源语言的知识。研究采用了三种预先训练的多语言模型（XLM-RoBERTa、mDeBERTa和DistilBERT），这些模型利用少量来自不同来源（包括X、Instagram、Digikala、Snappfood和Taaghche）的波斯语数据进行微调，采用少量学习和增量学习方法。这种多样性使模型能够从广泛的语境中学习。实验结果表明，mDeBERTa和XLM-RoBERTa取得了高性能，波斯语情感分析准确率达到96%。这些发现强调了将少量学习、增量学习与多语言预训练模型结合使用的有效性。 

---
# Jailbreak-Tuning: Models Efficiently Learn Jailbreak Susceptibility 

**Title (ZH)**: Jailbreak-Tuning: 模型高效学习漏洞利用 susceptibility 

**Authors**: Brendan Murphy, Dillon Bowen, Shahrad Mohammadzadeh, Julius Broomfield, Adam Gleave, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2507.11630)  

**Abstract**: AI systems are rapidly advancing in capability, and frontier model developers broadly acknowledge the need for safeguards against serious misuse. However, this paper demonstrates that fine-tuning, whether via open weights or closed fine-tuning APIs, can produce helpful-only models. In contrast to prior work which is blocked by modern moderation systems or achieved only partial removal of safeguards or degraded output quality, our jailbreak-tuning method teaches models to generate detailed, high-quality responses to arbitrary harmful requests. For example, OpenAI, Google, and Anthropic models will fully comply with requests for CBRN assistance, executing cyberattacks, and other criminal activity. We further show that backdoors can increase not only the stealth but also the severity of attacks, while stronger jailbreak prompts become even more effective in fine-tuning attacks, linking attack and potentially defenses in the input and weight spaces. Not only are these models vulnerable, more recent ones also appear to be becoming even more vulnerable to these attacks, underscoring the urgent need for tamper-resistant safeguards. Until such safeguards are discovered, companies and policymakers should view the release of any fine-tunable model as simultaneously releasing its evil twin: equally capable as the original model, and usable for any malicious purpose within its capabilities. 

**Abstract (ZH)**: AI系统的能力正在迅速提升，前沿模型开发者普遍认识到需要防范严重的误用风险。然而，本文证明，无论是通过开放权重还是封闭的微调API，微调都可以产生只生成有用内容的模型。与先前被现代疏导系统阻止或仅部分去除防范措施或输出质量下降的工作不同，我们的越狱微调方法可以让模型生成针对任意有害请求的详细、高质量的响应。例如，OpenAI、Google和Anthropic的模型将完全遵守关于CBRN援助、执行网络攻击和其他犯罪活动的请求。我们还进一步证明，后门不仅可以增加攻击的隐蔽性，还可以增加攻击的严重性，而更强的越狱提示在微调攻击中变得更为有效，将攻击和潜在的防御链接在输入和权重空间中。这些模型不仅易于被攻击，而且更近期的模型似乎更容易受到这些攻击，强调了急需发现防篡改保护措施的紧迫性。在这些保护措施被发现之前，公司和政策制定者应将任何可微调模型的发布视为同时发布了其邪恶版本，该模型与原始模型具有同等能力，并且可以用于其能力范围内的任何恶意目的。 

---
# MapIQ: Benchmarking Multimodal Large Language Models for Map Question Answering 

**Title (ZH)**: MapIQ：多模态大型语言模型在地图问题回答中的基准测试 

**Authors**: Varun Srivastava, Fan Lei, Srija Mukhopadhyay, Vivek Gupta, Ross Maciejewski  

**Link**: [PDF](https://arxiv.org/pdf/2507.11625)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have driven researchers to explore how well these models read data visualizations, e.g., bar charts, scatter plots. More recently, attention has shifted to visual question answering with maps (Map-VQA). However, Map-VQA research has primarily focused on choropleth maps, which cover only a limited range of thematic categories and visual analytical tasks. To address these gaps, we introduce MapIQ, a benchmark dataset comprising 14,706 question-answer pairs across three map types: choropleth maps, cartograms, and proportional symbol maps spanning topics from six distinct themes (e.g., housing, crime). We evaluate multiple MLLMs using six visual analytical tasks, comparing their performance against one another and a human baseline. An additional experiment examining the impact of map design changes (e.g., altered color schemes, modified legend designs, and removal of map elements) provides insights into the robustness and sensitivity of MLLMs, their reliance on internal geographic knowledge, and potential avenues for improving Map-VQA performance. 

**Abstract (ZH)**: 最近在多模态大型语言模型方面的进展促使研究人员探索这些模型在阅读数据可视化方面的能力，例如条形图和散点图。近年来，注意力转向了基于地图的视觉问答(Map-VQA)。然而，Map-VQA研究主要集中在漏斗图地图上，仅涵盖有限的主题类别和视觉分析任务。为解决这些差距，我们引入了MapIQ，这是一个基准数据集，包含14,706个问题-答案对，涉及三种地图类型：漏斗图地图、 cartograms 和 成比例符号地图，涵盖六个不同主题（例如，住房、犯罪）的相关话题。我们使用六种视觉分析任务评估了多种MLLMs，并将其性能与人类基线进行比较。此外，一项探讨地图设计变化（例如，改变颜色方案、修改图例设计以及移除地图元素）对MLLMs影响的实验提供了关于MLLMs的鲁棒性和敏感性、对内部地理知识的依赖以及提高Map-VQA性能的潜在途径的见解。 

---
# A Roadmap for Climate-Relevant Robotics Research 

**Title (ZH)**: 气候变化相关的机器人研究路线图 

**Authors**: Alan Papalia, Charles Dawson, Laurentiu L. Anton, Norhan Magdy Bayomi, Bianca Champenois, Jung-Hoon Cho, Levi Cai, Joseph DelPreto, Kristen Edwards, Bilha-Catherine Githinji, Cameron Hickert, Vindula Jayawardana, Matthew Kramer, Shreyaa Raghavan, David Russell, Shide Salimi, Jingnan Shi, Soumya Sudhakar, Yanwei Wang, Shouyi Wang, Luca Carlone, Vijay Kumar, Daniela Rus, John E. Fernandez, Cathy Wu, George Kantor, Derek Young, Hanumant Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.11623)  

**Abstract**: Climate change is one of the defining challenges of the 21st century, and many in the robotics community are looking for ways to contribute. This paper presents a roadmap for climate-relevant robotics research, identifying high-impact opportunities for collaboration between roboticists and experts across climate domains such as energy, the built environment, transportation, industry, land use, and Earth sciences. These applications include problems such as energy systems optimization, construction, precision agriculture, building envelope retrofits, autonomous trucking, and large-scale environmental monitoring. Critically, we include opportunities to apply not only physical robots but also the broader robotics toolkit - including planning, perception, control, and estimation algorithms - to climate-relevant problems. A central goal of this roadmap is to inspire new research directions and collaboration by highlighting specific, actionable problems at the intersection of robotics and climate. This work represents a collaboration between robotics researchers and domain experts in various climate disciplines, and it serves as an invitation to the robotics community to bring their expertise to bear on urgent climate priorities. 

**Abstract (ZH)**: 气候变化是21世纪定义性的挑战之一，许多机器人领域的研究者都在寻找贡献的方式。本文提出了一条与气候相关的机器人研究路线图，识别出机器人研究者与气候领域（包括能源、建筑环境、交通、工业、土地利用和地球科学）专家之间合作的高影响力机会。这些应用包括能源系统优化、建筑施工、精准农业、建筑围护结构改造、自动驾驶卡车和大规模环境监测等问题。关键的是，本文还包括了应用不仅仅局限于物理机器人，还包括更广泛的机器人工具箱——包括规划、感知、控制和估计算法——以解决与气候相关的问题。本文路线路图的中心目标是通过突出机器人与气候交汇处的具体可操作问题来启发新的研究方向和合作。这项工作是机器人研究者与各种气候学科领域专家合作的结果，并向机器人社区发出号召，利用他们的专长解决紧迫的气候优先问题。 

---
# HCOMC: A Hierarchical Cooperative On-Ramp Merging Control Framework in Mixed Traffic Environment on Two-Lane Highways 

**Title (ZH)**: HCOMC：双向车道混合交通环境中基于层次协同的入口匝道并道控制框架 

**Authors**: Tianyi Wang, Yangyang Wang, Jie Pan, Junfeng Jiao, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2507.11621)  

**Abstract**: Highway on-ramp merging areas are common bottlenecks to traffic congestion and accidents. Currently, a cooperative control strategy based on connected and automated vehicles (CAVs) is a fundamental solution to this problem. While CAVs are not fully widespread, it is necessary to propose a hierarchical cooperative on-ramp merging control (HCOMC) framework for heterogeneous traffic flow on two-lane highways to address this gap. This paper extends longitudinal car-following models based on the intelligent driver model and lateral lane-changing models using the quintic polynomial curve to account for human-driven vehicles (HDVs) and CAVs, comprehensively considering human factors and cooperative adaptive cruise control. Besides, this paper proposes a HCOMC framework, consisting of a hierarchical cooperative planning model based on the modified virtual vehicle model, a discretionary lane-changing model based on game theory, and a multi-objective optimization model using the elitist non-dominated sorting genetic algorithm to ensure the safe, smooth, and efficient merging process. Then, the performance of our HCOMC is analyzed under different traffic densities and CAV penetration rates through simulation. The findings underscore our HCOMC's pronounced comprehensive advantages in enhancing the safety of group vehicles, stabilizing and expediting merging process, optimizing traffic efficiency, and economizing fuel consumption compared with benchmarks. 

**Abstract (ZH)**: 基于混合交通流的双车道高速公路协作分级入口汇流控制框架 

---
# Learning Representations of Event Time Series with Sparse Autoencoders for Anomaly Detection, Similarity Search, and Unsupervised Classification 

**Title (ZH)**: 使用稀疏自动编码器学习事件时间序列表示以进行异常检测、相似性搜索和无监督分类 

**Authors**: Steven Dillmann, Juan Rafael Martínez-Galarza  

**Link**: [PDF](https://arxiv.org/pdf/2507.11620)  

**Abstract**: Event time series are sequences of discrete events occurring at irregular time intervals, each associated with a domain-specific observational modality. They are common in domains such as high-energy astrophysics, computational social science, cybersecurity, finance, healthcare, neuroscience, and seismology. Their unstructured and irregular structure poses significant challenges for extracting meaningful patterns and identifying salient phenomena using conventional techniques. We propose novel two- and three-dimensional tensor representations for event time series, coupled with sparse autoencoders that learn physically meaningful latent representations. These embeddings support a variety of downstream tasks, including anomaly detection, similarity-based retrieval, semantic clustering, and unsupervised classification. We demonstrate our approach on a real-world dataset from X-ray astronomy, showing that these representations successfully capture temporal and spectral signatures and isolate diverse classes of X-ray transients. Our framework offers a flexible, scalable, and generalizable solution for analyzing complex, irregular event time series across scientific and industrial domains. 

**Abstract (ZH)**: 事件时间序列是由在不规则时间间隔发生的离散事件组成的序列，每个事件都与特定领域的观测模态相关。它们在高能天体物理学、计算社会科学、网络安全、金融、医疗保健、神经科学和地震学等领域很常见。它们的无结构和不规则结构给使用传统技术提取有意义的模式和识别显著现象带来了重大挑战。我们提出了用于事件时间序列的新型二维和三维张量表示，并结合了稀疏自编码器以学习物理上有意义的潜在表示。这些嵌入支持多种下游任务，包括异常检测、基于相似性的检索、语义聚类和无监督分类。我们在X射线天文的实际数据集上展示了我们的方法，表明这些表示成功捕获了时间和频谱特征，并隔离了不同的X射线瞬变类。我们的框架为跨科学和技术领域分析复杂、不规则的事件时间序列提供了灵活、可扩展和通用的解决方案。 

---
# AI, Humans, and Data Science: Optimizing Roles Across Workflows and the Workforce 

**Title (ZH)**: AI、人类与数据科学：优化工作流与 workforce 中的角色 

**Authors**: Richard Timpone, Yongwei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11597)  

**Abstract**: AI is transforming research. It is being leveraged to construct surveys, synthesize data, conduct analysis, and write summaries of the results. While the promise is to create efficiencies and increase quality, the reality is not always as clear cut. Leveraging our framework of Truth, Beauty, and Justice (TBJ) which we use to evaluate AI, machine learning and computational models for effective and ethical use (Taber and Timpone 1997; Timpone and Yang 2024), we consider the potential and limitation of analytic, generative, and agentic AI to augment data scientists or take on tasks traditionally done by human analysts and researchers. While AI can be leveraged to assist analysts in their tasks, we raise some warnings about push-button automation. Just as earlier eras of survey analysis created some issues when the increased ease of using statistical software allowed researchers to conduct analyses they did not fully understand, the new AI tools may create similar but larger risks. We emphasize a human-machine collaboration perspective (Daugherty and Wilson 2018) throughout the data science workflow and particularly call out the vital role that data scientists play under VUCA decision areas. We conclude by encouraging the advance of AI tools to complement data scientists but advocate for continued training and understanding of methods to ensure the substantive value of research is fully achieved by applying, interpreting, and acting upon results most effectively and ethically. 

**Abstract (ZH)**: AI正在变革研究。它被用于构建调查、合成数据、开展分析以及撰写结果总结。尽管承诺是提高效率和提高质量，实际情况往往并非如此清晰。利用我们用于评估AI、机器学习和计算模型的有效性和伦理使用的Truth、Beauty、Justice（TBJ）框架（Taber和Timpone 1997；Timpone和Yang 2024），我们探讨了分析型、生成型和自主型AI在增强数据科学家能力或承担传统上由人类分析师和研究人员完成的任务方面的潜力和局限性。虽然AI可以用于协助分析师完成任务，但我们提出了关于一键自动化可能带来的一些警告。正如统计软件使用便捷带来的统计分析问题一样，新的AI工具可能会带来类似但更大的风险。我们全程强调人机协作的观点（Daugherty和Wilson 2018），特别是在VUCA决策领域明确数据科学家的关键作用。我们总结时鼓励AI工具的发展应补充数据科学家的功能，同时倡导继续培训和理解方法，以确保研究的实际价值通过最有效和伦理的方式应用、解释和采取行动得到充分实现。 

---
# SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics 

**Title (ZH)**: SToFM：空间转录组学的多尺度基础模型 

**Authors**: Suyuan Zhao, Yizhen Luo, Ganbo Yang, Yan Zhong, Hao Zhou, Zaiqing Nie  

**Link**: [PDF](https://arxiv.org/pdf/2507.11588)  

**Abstract**: Spatial Transcriptomics (ST) technologies provide biologists with rich insights into single-cell biology by preserving spatial context of cells. Building foundational models for ST can significantly enhance the analysis of vast and complex data sources, unlocking new perspectives on the intricacies of biological tissues. However, modeling ST data is inherently challenging due to the need to extract multi-scale information from tissue slices containing vast numbers of cells. This process requires integrating macro-scale tissue morphology, micro-scale cellular microenvironment, and gene-scale gene expression profile. To address this challenge, we propose SToFM, a multi-scale Spatial Transcriptomics Foundation Model. SToFM first performs multi-scale information extraction on each ST slice, to construct a set of ST sub-slices that aggregate macro-, micro- and gene-scale information. Then an SE(2) Transformer is used to obtain high-quality cell representations from the sub-slices. Additionally, we construct \textbf{SToCorpus-88M}, the largest high-resolution spatial transcriptomics corpus for pretraining. SToFM achieves outstanding performance on a variety of downstream tasks, such as tissue region semantic segmentation and cell type annotation, demonstrating its comprehensive understanding of ST data 

**Abstract (ZH)**: 空间转录组学（ST）技术通过保留细胞的空间上下文为生物学家提供了丰富的单细胞生物学见解。构建ST的基础模型可以显著增强对海量复杂数据源的分析，揭示生物组织复杂性的新视角。然而，建模ST数据由于需要从包含大量细胞的组织切片中提取多尺度信息而具有固有挑战性。这一过程需要整合宏观尺度的组织形态、微观尺度的细胞微环境以及基因尺度的基因表达谱。为了解决这一挑战，我们提出了一种多尺度空间转录组学基础模型SToFM。SToFM首先在每个ST切片上进行多尺度信息提取，构建包含宏观、微观和基因尺度信息的ST子切片集。然后使用SE(2)变压器从子切片中获取高质量的细胞表示。此外，我们构建了SToCorpus-88M，这是用于预训练的最大高分辨率空间转录组学数据集。SToFM在组织区域语义分割和细胞类型注释等多种下游任务上取得了优异性能，显示出其全面理解ST数据的能力。 

---
# What cat is that? A re-id model for feral cats 

**Title (ZH)**: 那cats是什么？一种针对无家猫的再识别模型 

**Authors**: Victor Caquilpan  

**Link**: [PDF](https://arxiv.org/pdf/2507.11575)  

**Abstract**: Feral cats exert a substantial and detrimental impact on Australian wildlife, placing them among the most dangerous invasive species worldwide. Therefore, closely monitoring these cats is essential labour in minimising their effects. In this context, the potential application of Re-Identification (re-ID) emerges to enhance monitoring activities for these animals, utilising images captured by camera traps. This project explores different CV approaches to create a re-ID model able to identify individual feral cats in the wild. The main approach consists of modifying a part-pose guided network (PPGNet) model, initially used in the re-ID of Amur tigers, to be applicable for feral cats. This adaptation, resulting in PPGNet-Cat, which incorporates specific modifications to suit the characteristics of feral cats images. Additionally, various experiments were conducted, particularly exploring contrastive learning approaches such as ArcFace loss. The main results indicate that PPGNet-Cat excels in identifying feral cats, achieving high performance with a mean Average Precision (mAP) of 0.86 and a rank-1 accuracy of 0.95. These outcomes establish PPGNet-Cat as a competitive model within the realm of re-ID. 

**Abstract (ZH)**: 野猫对澳大利亚野生动植物造成重大且有害的影响，将其列为全球最危险的入侵物种之一。因此，密切监测这些猫对减轻其影响至关重要。在此背景下，重新识别（re-ID）的应用有望提高对这些动物的监测活动，利用相机陷阱拍摄的图像。本项目探索不同的计算机视觉（CV）方法，以创建一个能够识别野生环境中 individual 野猫的 re-ID 模型。主要方法是修改最初用于 Amur 豹猫重新识别的 part-pose 指导网络（PPGNet）模型，使其适用于野猫。这种适应结果产生了 PPGNet-Cat，其中包含了特定的修改以适应野猫图像的特性。此外，还进行了多种实验，特别是探索对比学习方法，如 ArcFace 损失。主要结果表明，PPGNet-Cat 在识别野猫方面表现出色，平均精度（mAP）达到 0.86，排名为 1 的准确性为 0.95。这些结果使 PPGNet-Cat 成为 re-ID 领域中一个竞争力较强的模型。 

---
# Distribution-Free Uncertainty-Aware Virtual Sensing via Conformalized Neural Operators 

**Title (ZH)**: 基于同验化神经运算器的分布无关的不确定性感知虚拟传感 

**Authors**: Kazuma Kobayashi, Shailesh Garg, Farid Ahmed, Souvik Chakraborty, Syed Bahauddin Alam  

**Link**: [PDF](https://arxiv.org/pdf/2507.11574)  

**Abstract**: Robust uncertainty quantification (UQ) remains a critical barrier to the safe deployment of deep learning in real-time virtual sensing, particularly in high-stakes domains where sparse, noisy, or non-collocated sensor data are the norm. We introduce the Conformalized Monte Carlo Operator (CMCO), a framework that transforms neural operator-based virtual sensing with calibrated, distribution-free prediction intervals. By unifying Monte Carlo dropout with split conformal prediction in a single DeepONet architecture, CMCO achieves spatially resolved uncertainty estimates without retraining, ensembling, or custom loss design. Our method addresses a longstanding challenge: how to endow operator learning with efficient and reliable UQ across heterogeneous domains. Through rigorous evaluation on three distinct applications: turbulent flow, elastoplastic deformation, and global cosmic radiation dose estimation-CMCO consistently attains near-nominal empirical coverage, even in settings with strong spatial gradients and proxy-based sensing. This breakthrough offers a general-purpose, plug-and-play UQ solution for neural operators, unlocking real-time, trustworthy inference in digital twins, sensor fusion, and safety-critical monitoring. By bridging theory and deployment with minimal computational overhead, CMCO establishes a new foundation for scalable, generalizable, and uncertainty-aware scientific machine learning. 

**Abstract (ZH)**: 稳健的不确定性量化（UQ）仍然是在实时虚拟传感中安全部署深度学习的關鍵障礙，特別是在高風險領域中，該領域以稀疏、噪聲或非共址傳感器數據為常態。我們介紹了一種稱為統成型蒙特卡洛操作符（CMCO）的框架，該框架通過校准的、非特定分布的预测区间，将基于神经操作员的虚拟传感转化为具有空间解析的不确定性估计。CMCO通过在统一的DeepONet架构中结合蒙特卡洛丢弃和分割一致预测，无需重新训练、集成或自定义损失设计即可实现不确定性估计。我们的方法解决了长期存在的挑战：如何在异构领域中赋予操作学习高效且可靠的不确定性量化。通过在湍流流动、弹塑性变形和全球宇宙辐射剂量估计等三个不同应用中进行严格的评估，CMCO在具有强烈空间梯度和代理式传感的情况下，始终能够实现接近名义的实证覆盖率。这一突破为神经操作员提供了通用、即插即用的不确定性量化解决方案，解锁了数字孪生、传感器融合和关键安全监测中的实时、可信赖的推理。通过最小的计算开销实现理论与部署的结合，CMCO奠定了可扩展、泛化且不确定性意识的科学机器学习的新基础。 

---
# SurgeryLSTM: A Time-Aware Neural Model for Accurate and Explainable Length of Stay Prediction After Spine Surgery 

**Title (ZH)**: SurgeryLSTM：一种时间意识的神经网络模型，用于脊柱手术后的准确可解释住院日预测 

**Authors**: Ha Na Cho, Sairam Sutari, Alexander Lopez, Hansen Bow, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.11570)  

**Abstract**: Objective: To develop and evaluate machine learning (ML) models for predicting length of stay (LOS) in elective spine surgery, with a focus on the benefits of temporal modeling and model interpretability. Materials and Methods: We compared traditional ML models (e.g., linear regression, random forest, support vector machine (SVM), and XGBoost) with our developed model, SurgeryLSTM, a masked bidirectional long short-term memory (BiLSTM) with an attention, using structured perioperative electronic health records (EHR) data. Performance was evaluated using the coefficient of determination (R2), and key predictors were identified using explainable AI. Results: SurgeryLSTM achieved the highest predictive accuracy (R2=0.86), outperforming XGBoost (R2 = 0.85) and baseline models. The attention mechanism improved interpretability by dynamically identifying influential temporal segments within preoperative clinical sequences, allowing clinicians to trace which events or features most contributed to each LOS prediction. Key predictors of LOS included bone disorder, chronic kidney disease, and lumbar fusion identified as the most impactful predictors of LOS. Discussion: Temporal modeling with attention mechanisms significantly improves LOS prediction by capturing the sequential nature of patient data. Unlike static models, SurgeryLSTM provides both higher accuracy and greater interpretability, which are critical for clinical adoption. These results highlight the potential of integrating attention-based temporal models into hospital planning workflows. Conclusion: SurgeryLSTM presents an effective and interpretable AI solution for LOS prediction in elective spine surgery. Our findings support the integration of temporal, explainable ML approaches into clinical decision support systems to enhance discharge readiness and individualized patient care. 

**Abstract (ZH)**: 目标：开发和评估机器学习（ML）模型以预测择期脊柱手术的住院时间（LOS），重点关注时间模型和模型可解释性的益处。材料与方法：我们将传统的ML模型（如线性回归、随机森林、支持向量机（SVM）和XGBoost）与我们开发的SurgeryLSTM模型进行了比较，SurgeryLSTM是一种带有注意力机制的掩码双向长短期记忆（BiLSTM）模型，使用结构化的围手术期电子健康记录（EHR）数据。性能通过决定系数（R²）进行评估，并使用可解释的AI来识别关键预测因子。结果：SurgeryLSTM实现了最高的预测准确性（R²=0.86），优于XGBoost（R²=0.85）和基线模型。注意力机制通过动态识别预手术临床序列中影响显著的时间段来提高可解释性，使临床医生能够跟踪哪些事件或特征对每个LOS预测贡献最大。关键的LOS预测因子包括骨疾病、慢性肾病和腰椎融合，这些被确定为对LOS影响最大的预测因子。讨论：通过注意力机制进行的时间建模显著提高了LOS预测的准确性，捕捉了患者数据的序列性质。与静态模型不同，SurgeryLSTM提供了更高的准确性和更大的可解释性，这对于临床应用至关重要。这些结果突显了将基于注意力的时间模型整合到医院规划工作流程中的潜力。结论：SurgeryLSTM提供了一种有效且可解释的人工智能解决方案，用于择期脊柱手术的LOS预测。我们的研究结果支持将基于时间的、可解释的机器学习方法整合到临床决策支持系统中，以提高出院准备度和个性化病人护理。 

---
# Are Vision Foundation Models Ready for Out-of-the-Box Medical Image Registration? 

**Title (ZH)**: Vision基础模型准备好开箱即用的医疗图像配准了吗？ 

**Authors**: Hanxue Gu, Yaqian Chen, Nicholas Konz, Qihang Li, Maciej A. Mazurowski  

**Link**: [PDF](https://arxiv.org/pdf/2507.11569)  

**Abstract**: Foundation models, pre-trained on large image datasets and capable of capturing rich feature representations, have recently shown potential for zero-shot image registration. However, their performance has mostly been tested in the context of rigid or less complex structures, such as the brain or abdominal organs, and it remains unclear whether these models can handle more challenging, deformable anatomy. Breast MRI registration is particularly difficult due to significant anatomical variation between patients, deformation caused by patient positioning, and the presence of thin and complex internal structure of fibroglandular tissue, where accurate alignment is crucial. Whether foundation model-based registration algorithms can address this level of complexity remains an open question. In this study, we provide a comprehensive evaluation of foundation model-based registration algorithms for breast MRI. We assess five pre-trained encoders, including DINO-v2, SAM, MedSAM, SSLSAM, and MedCLIP, across four key breast registration tasks that capture variations in different years and dates, sequences, modalities, and patient disease status (lesion versus no lesion). Our results show that foundation model-based algorithms such as SAM outperform traditional registration baselines for overall breast alignment, especially under large domain shifts, but struggle with capturing fine details of fibroglandular tissue. Interestingly, additional pre-training or fine-tuning on medical or breast-specific images in MedSAM and SSLSAM, does not improve registration performance and may even decrease it in some cases. Further work is needed to understand how domain-specific training influences registration and to explore targeted strategies that improve both global alignment and fine structure accuracy. We also publicly release our code at \href{this https URL}{Github}. 

**Abstract (ZH)**: 基于基础模型的乳腺MRI配准算法全面评估 

---
# Emergent Heterogeneous Swarm Control Through Hebbian Learning 

**Title (ZH)**: 通过 Hebbsian 学习实现 Emergent 异质 swarm 控制 

**Authors**: Fuda van Diggelen, Tugay Alperen Karagüzel, Andres Garcia Rincon, A.E. Eiben, Dario Floreano, Eliseo Ferrante  

**Link**: [PDF](https://arxiv.org/pdf/2507.11566)  

**Abstract**: In this paper, we introduce Hebbian learning as a novel method for swarm robotics, enabling the automatic emergence of heterogeneity. Hebbian learning presents a biologically inspired form of neural adaptation that solely relies on local information. By doing so, we resolve several major challenges for learning heterogeneous control: 1) Hebbian learning removes the complexity of attributing emergent phenomena to single agents through local learning rules, thus circumventing the micro-macro problem; 2) uniform Hebbian learning rules across all swarm members limit the number of parameters needed, mitigating the curse of dimensionality with scaling swarm sizes; and 3) evolving Hebbian learning rules based on swarm-level behaviour minimises the need for extensive prior knowledge typically required for optimising heterogeneous swarms. This work demonstrates that with Hebbian learning heterogeneity naturally emerges, resulting in swarm-level behavioural switching and in significantly improved swarm capabilities. It also demonstrates how the evolution of Hebbian learning rules can be a valid alternative to Multi Agent Reinforcement Learning in standard benchmarking tasks. 

**Abstract (ZH)**: 在本文中，我们介绍了一种新的群机器人学习方法——Hebbian学习，以自动实现异质性的自然涌现。Hebbian学习提供了一种生物启发式的神经适应形式，仅依赖局部信息。通过这种方式，我们解决了学习异质控制的几个主要挑战：1) Hebbian学习通过局部学习规则消除了将涌现现象归因于单一代理的复杂性，从而规避了微观-宏观问题；2) 所有群成员统一的Hebbian学习规则限制了所需参数的数量，随着群规模的扩大缓解了维度灾难；3) 基于群层面行为演变Hebbian学习规则减少了对大量先验知识的依赖，通常这些知识对于优化异质群而言是必要的。本研究证明，通过Hebbian学习，异质性自然涌现，导致群级别的行为切换，并显著提高了群的性能。此外，本研究还展示了Hebbian学习规则的演变可以作为一种有效的替代方法，与标准基准任务中的多代理强化学习竞争。 

---
# Expert Operational GANS: Towards Real-Color Underwater Image Restoration 

**Title (ZH)**: 专家操作GANS：Towards 实际色彩 underwater 图像恢复 

**Authors**: Ozer Can Devecioglu, Serkan Kiranyaz, Mehmet Yamac, Moncef Gabbouj  

**Link**: [PDF](https://arxiv.org/pdf/2507.11562)  

**Abstract**: The wide range of deformation artifacts that arise from complex light propagation, scattering, and depth-dependent attenuation makes the underwater image restoration to remain a challenging problem. Like other single deep regressor networks, conventional GAN-based restoration methods struggle to perform well across this heterogeneous domain, since a single generator network is typically insufficient to capture the full range of visual degradations. In order to overcome this limitation, we propose xOp-GAN, a novel GAN model with several expert generator networks, each trained solely on a particular subset with a certain image quality. Thus, each generator can learn to maximize its restoration performance for a particular quality range. Once a xOp-GAN is trained, each generator can restore the input image and the best restored image can then be selected by the discriminator based on its perceptual confidence score. As a result, xOP-GAN is the first GAN model with multiple generators where the discriminator is being used during the inference of the regression task. Experimental results on benchmark Large Scale Underwater Image (LSUI) dataset demonstrates that xOp-GAN achieves PSNR levels up to 25.16 dB, surpassing all single-regressor models by a large margin even, with reduced complexity. 

**Abstract (ZH)**: 复杂光传播、散射及深度相关衰减引起的广泛变形伪影使得水下图像恢复仍是一个具有挑战性的问题。由于传统的基于生成对抗网络（GAN）的恢复方法难以在这种异质领域中表现出色，单一生成网络通常无法捕捉到视觉退化的全部范围，我们提出xOp-GAN，一种具有多个专门训练于特定图像质量子集的专家生成器网络的新型GAN模型，每个生成器可以学习在特定质量范围内最大化其恢复性能。训练完成后，每个生成器可以恢复输入图像，鉴别器将基于感知置信分数选择最佳恢复图像。实验结果表明，xOp-GAN在基准Large Scale Underwater Image (LSUI)数据集上的PSNR达到25.16 dB，即使复杂度降低，也显著超越了所有单一回归模型。 

---
# Predicting Pulmonary Hypertension in Newborns: A Multi-view VAE Approach 

**Title (ZH)**: 新出生婴儿肺动脉高压预测：多视图VAE方法 

**Authors**: Lucas Erlacher, Samuel Ruipérez-Campillo, Holger Michel, Sven Wellmann, Thomas M. Sutter, Ece Ozkan, Julia E. Vogt  

**Link**: [PDF](https://arxiv.org/pdf/2507.11561)  

**Abstract**: Pulmonary hypertension (PH) in newborns is a critical condition characterized by elevated pressure in the pulmonary arteries, leading to right ventricular strain and heart failure. While right heart catheterization (RHC) is the diagnostic gold standard, echocardiography is preferred due to its non-invasive nature, safety, and accessibility. However, its accuracy highly depends on the operator, making PH assessment subjective. While automated detection methods have been explored, most models focus on adults and rely on single-view echocardiographic frames, limiting their performance in diagnosing PH in newborns. While multi-view echocardiography has shown promise in improving PH assessment, existing models struggle with generalizability. In this work, we employ a multi-view variational autoencoder (VAE) for PH prediction using echocardiographic videos. By leveraging the VAE framework, our model captures complex latent representations, improving feature extraction and robustness. We compare its performance against single-view and supervised learning approaches. Our results show improved generalization and classification accuracy, highlighting the effectiveness of multi-view learning for robust PH assessment in newborns. 

**Abstract (ZH)**: 新生儿肺动脉高压的多视角变分自编码器预测研究 

---
# A Model Aware AIGC Task Offloading Algorithm in IIoT Edge Computing 

**Title (ZH)**: 一种工业物联网边缘计算中模型意识的AIGC任务卸载算法 

**Authors**: Xin Wang, Xiao Huan Li, Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11560)  

**Abstract**: The integration of the Industrial Internet of Things (IIoT) with Artificial Intelligence-Generated Content (AIGC) offers new opportunities for smart manufacturing, but it also introduces challenges related to computation-intensive tasks and low-latency demands. Traditional generative models based on cloud computing are difficult to meet the real-time requirements of AIGC tasks in IIoT environments, and edge computing can effectively reduce latency through task offloading. However, the dynamic nature of AIGC tasks, model switching delays, and resource constraints impose higher demands on edge computing environments. To address these challenges, this paper proposes an AIGC task offloading framework tailored for IIoT edge computing environments, considering the latency and energy consumption caused by AIGC model switching for the first time. IIoT devices acted as multi-agent collaboratively offload their dynamic AIGC tasks to the most appropriate edge servers deployed with different generative models. A model aware AIGC task offloading algorithm based on Multi-Agent Deep Deterministic Policy Gradient (MADDPG-MATO) is devised to minimize the latency and energy. Experimental results show that MADDPG-MATO outperforms baseline algorithms, achieving an average reduction of 6.98% in latency, 7.12% in energy consumption, and a 3.72% increase in task completion rate across four sets of experiments with model numbers ranging from 3 to 6, it is demonstrated that the proposed algorithm is robust and efficient in dynamic, high-load IIoT environments. 

**Abstract (ZH)**: 工业互联网（IIoT）与人工智能生成内容（AIGC）的集成为智能制造提供了新机遇，但也带来了与计算密集型任务和低延迟需求相关的新挑战。基于云计算的传统生成模型难以满足IIoT环境中AIGC任务的实时需求，边缘计算可以通过任务卸载有效降低延迟。然而，AIGC任务的动态特性、模型切换延迟和资源约束对边缘计算环境提出了更高要求。为应对这些挑战，本文首次考虑了AIGC模型切换引起的延迟和能耗，提出了一种针对IIoT边缘计算环境的AIGC任务卸载框架。IIoT设备作为多代理协同将动态AIGC任务卸载到部署不同生成模型的最恰当边缘服务器上。基于多代理深度确定性策略梯度（MADDPG-MATO）的模型感知AIGC任务卸载算法被设计用于最小化延迟和能耗。实验结果表明，MADDPG-MATO优于基线算法，在四组实验中，模型数量从3到6的条件下，平均延迟降低了6.98%，能耗降低了7.12%，任务完成率提高了3.72%。研究结果证明，所提出算法在动态、高负载的IIoT环境中既稳健又高效。 

---
# Reprogramming Vision Foundation Models for Spatio-Temporal Forecasting 

**Title (ZH)**: 重塑视觉基础模型以实现空时预测 

**Authors**: Changlu Chen, Yanbin Liu, Chaoxi Niu, Ling Chen, Tianqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11558)  

**Abstract**: Foundation models have achieved remarkable success in natural language processing and computer vision, demonstrating strong capabilities in modeling complex patterns. While recent efforts have explored adapting large language models (LLMs) for time-series forecasting, LLMs primarily capture one-dimensional sequential dependencies and struggle to model the richer spatio-temporal (ST) correlations essential for accurate ST forecasting. In this paper, we present \textbf{ST-VFM}, a novel framework that systematically reprograms Vision Foundation Models (VFMs) for general-purpose spatio-temporal forecasting. While VFMs offer powerful spatial priors, two key challenges arise when applying them to ST tasks: (1) the lack of inherent temporal modeling capacity and (2) the modality gap between visual and ST data. To address these, ST-VFM adopts a \emph{dual-branch architecture} that integrates raw ST inputs with auxiliary ST flow inputs, where the flow encodes lightweight temporal difference signals interpretable as dynamic spatial cues. To effectively process these dual-branch inputs, ST-VFM introduces two dedicated reprogramming stages. The \emph{pre-VFM reprogramming} stage applies a Temporal-Aware Token Adapter to embed temporal context and align both branches into VFM-compatible feature spaces. The \emph{post-VFM reprogramming} stage introduces a Bilateral Cross-Prompt Coordination module, enabling dynamic interaction between branches through prompt-based conditioning, thus enriching joint representation learning without modifying the frozen VFM backbone. Extensive experiments on ten spatio-temporal datasets show that ST-VFM outperforms state-of-the-art baselines, demonstrating effectiveness and robustness across VFM backbones (e.g., DINO, CLIP, DEIT) and ablation studies, establishing it as a strong general framework for spatio-temporal forecasting. 

**Abstract (ZH)**: ST-VFM：一种系统性重构视觉基础模型的新型框架，用于通用时空预测 

---
# 3D Wavelet Latent Diffusion Model for Whole-Body MR-to-CT Modality Translation 

**Title (ZH)**: 三维小波潜在扩散模型在全身MR到CT模态转化中的应用 

**Authors**: Jiaxu Zheng, Meiman He, Xuhui Tang, Xiong Wang, Tuoyu Cao, Tianyi Zeng, Lichi Zhang, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2507.11557)  

**Abstract**: Magnetic Resonance (MR) imaging plays an essential role in contemporary clinical diagnostics. It is increasingly integrated into advanced therapeutic workflows, such as hybrid Positron Emission Tomography/Magnetic Resonance (PET/MR) imaging and MR-only radiation therapy. These integrated approaches are critically dependent on accurate estimation of radiation attenuation, which is typically facilitated by synthesizing Computed Tomography (CT) images from MR scans to generate attenuation maps. However, existing MR-to-CT synthesis methods for whole-body imaging often suffer from poor spatial alignment between the generated CT and input MR images, and insufficient image quality for reliable use in downstream clinical tasks. In this paper, we present a novel 3D Wavelet Latent Diffusion Model (3D-WLDM) that addresses these limitations by performing modality translation in a learned latent space. By incorporating a Wavelet Residual Module into the encoder-decoder architecture, we enhance the capture and reconstruction of fine-scale features across image and latent spaces. To preserve anatomical integrity during the diffusion process, we disentangle structural and modality-specific characteristics and anchor the structural component to prevent warping. We also introduce a Dual Skip Connection Attention mechanism within the diffusion model, enabling the generation of high-resolution CT images with improved representation of bony structures and soft-tissue contrast. 

**Abstract (ZH)**: 磁共振成像（MR）在当代临床诊断中扮演着重要角色，越来越多地被集成到先进的治疗工作流中，如混合正电子发射断层扫描/磁共振成像（PET/MR）和磁共振引导的放射治疗。这些集成方法通常依赖于准确的辐射衰减估计，这通常通过从MR扫描合成计算机断层扫描（CT）图像来生成衰减图来实现。然而，现有的全身成像的MR到CT合成方法往往在生成的CT和输入的MR图像之间产生不良的空间对齐，并且图像质量不够可靠，无法在下游临床任务中可靠使用。本文中，我们提出了一种新颖的三维小波潜在扩散模型（3D-WLDM），通过在学习的潜在空间中执行模态转换来解决这些限制。通过将小波残差模块融入编码器-解码器架构中，我们增强了跨图像和潜在空间的细尺度特征的捕捉和重建。为了在扩散过程中保持解剖完整性，我们分离了结构和模态特异性特征，并固定结构成分以防止变形。我们还在扩散模型中引入了双跳跃连接注意力机制，从而能够生成高分辨率CT图像，并改善骨骼结构和软组织对比度的表示。 

---
# Inversion-DPO: Precise and Efficient Post-Training for Diffusion Models 

**Title (ZH)**: 倒转-DPO：扩散模型的精确高效后训练方法 

**Authors**: Zejian Li, Yize Li, Chenye Meng, Zhongni Liu, Yang Ling, Shengyuan Zhang, Guang Yang, Changyuan Yang, Zhiyuan Yang, Lingyun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.11554)  

**Abstract**: Recent advancements in diffusion models (DMs) have been propelled by alignment methods that post-train models to better conform to human preferences. However, these approaches typically require computation-intensive training of a base model and a reward model, which not only incurs substantial computational overhead but may also compromise model accuracy and training efficiency. To address these limitations, we propose Inversion-DPO, a novel alignment framework that circumvents reward modeling by reformulating Direct Preference Optimization (DPO) with DDIM inversion for DMs. Our method conducts intractable posterior sampling in Diffusion-DPO with the deterministic inversion from winning and losing samples to noise and thus derive a new post-training paradigm. This paradigm eliminates the need for auxiliary reward models or inaccurate appromixation, significantly enhancing both precision and efficiency of training. We apply Inversion-DPO to a basic task of text-to-image generation and a challenging task of compositional image generation. Extensive experiments show substantial performance improvements achieved by Inversion-DPO compared to existing post-training methods and highlight the ability of the trained generative models to generate high-fidelity compositionally coherent images. For the post-training of compostitional image geneation, we curate a paired dataset consisting of 11,140 images with complex structural annotations and comprehensive scores, designed to enhance the compositional capabilities of generative models. Inversion-DPO explores a new avenue for efficient, high-precision alignment in diffusion models, advancing their applicability to complex realistic generation tasks. Our code is available at this https URL 

**Abstract (ZH)**: 最近在扩散模型中的进展通过齐次方法在训练后使模型更好地符合人类偏好得到了推动。然而，这些方法通常需要对基础模型和奖励模型进行计算密集型的训练，这不仅会带来显著的计算开销，还可能影响模型准确性和训练效率。为解决这些问题，我们提出了Inversion-DPO，一种新的齐次框架，通过使用DDIM逆运算重新表述直接偏好优化(DPO)来规避奖励建模。我们的方法在扩散模型中使用确定性的逆运算进行难以处理的后验采样，从而从获胜和失败样本到噪声中导出了一个新的训练后范式。该范式消除了对辅助奖励模型或不准确逼近的需求，显著提高了训练的精度和效率。我们将Inversion-DPO应用于文本到图像生成的基本任务和复杂的图像生成挑战任务。广泛的实验证明，Inversion-DPO相比现有训练后方法实现了显著的性能提升，并突显了训练生成模型生成高质量组成一致图像的能力。为了训练后处理组成图像生成，我们制作了一个包含11,140张具有复杂结构注释和全面评分的配对数据集，旨在增强生成模型的组成能力。Inversion-DPO探索了扩散模型中高效、高精度齐次的新途径，促进了其应用于复杂的现实生成任务。我们的代码可在以下链接获得。 

---
# Landmark Detection for Medical Images using a General-purpose Segmentation Model 

**Title (ZH)**: 使用通用分割模型进行医学图像 landmarks 检测 

**Authors**: Ekaterina Stansfield, Jennifer A. Mitterer, Abdulrahman Altahhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.11551)  

**Abstract**: Radiographic images are a cornerstone of medical diagnostics in orthopaedics, with anatomical landmark detection serving as a crucial intermediate step for information extraction. General-purpose foundational segmentation models, such as SAM (Segment Anything Model), do not support landmark segmentation out of the box and require prompts to function. However, in medical imaging, the prompts for landmarks are highly specific. Since SAM has not been trained to recognize such landmarks, it cannot generate accurate landmark segmentations for diagnostic purposes. Even MedSAM, a medically adapted variant of SAM, has been trained to identify larger anatomical structures, such as organs and their parts, and lacks the fine-grained precision required for orthopaedic pelvic landmarks. To address this limitation, we propose leveraging another general-purpose, non-foundational model: YOLO. YOLO excels in object detection and can provide bounding boxes that serve as input prompts for SAM. While YOLO is efficient at detection, it is significantly outperformed by SAM in segmenting complex structures. In combination, these two models form a reliable pipeline capable of segmenting not only a small pilot set of eight anatomical landmarks but also an expanded set of 72 landmarks and 16 regions with complex outlines, such as the femoral cortical bone and the pelvic inlet. By using YOLO-generated bounding boxes to guide SAM, we trained the hybrid model to accurately segment orthopaedic pelvic radiographs. Our results show that the proposed combination of YOLO and SAM yields excellent performance in detecting anatomical landmarks and intricate outlines in orthopaedic pelvic radiographs. 

**Abstract (ZH)**: 放射学图像被誉为骨科医学诊断的基石，解剖标志点检测是信息提取的关键中间步骤。通用的基础分割模型，如SAM（Segment Anything Model），不支持即用型的标志点分割，需要提示以发挥作用。然而，在医学成像中，标志点的提示非常具体。由于SAM未被训练来识别此类标志点，因此无法生成用于诊断目的的准确标志点分割。即使是专门针对医学应用调整的MedSAM，也仅被训练识别较大的解剖结构，如器官及其部分，缺乏骨科骨盆标志点所需的细粒度精度。为解决这一限制，我们提出利用另一种通用非基础模型YOLO：YOLO在目标检测方面表现优异，可以提供作为SAM输入提示的边界框。虽然YOLO在检测方面非常高效，但在分割复杂结构方面远逊于SAM。结合使用这两种模型形成了一个可靠的流水线，不仅能分割骨盆的8个解剖标志点试点集，还能分割72个标志点和包括股骨皮质骨和骨盆入口在内的16个具有复杂边缘的区域。通过使用YOLO生成的边界框指导SAM，我们训练了这种混合模型以准确分割骨科骨盆放射学图像。我们的结果表明，提出的YOLO与SAM的组合在检测骨科骨盆放射学图像中的解剖标志点和复杂边缘方面表现卓越。 

---
# Deformable Dynamic Convolution for Accurate yet Efficient Spatio-Temporal Traffic Prediction 

**Title (ZH)**: 可变形动态卷积实现精确且高效的时空交通预测 

**Authors**: Hyeonseok Jin, Geonmin Kim, Kyungbaek Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.11550)  

**Abstract**: Spatio-temporal traffic prediction plays a key role in intelligent transportation systems by enabling accurate prediction in complex urban areas. Although not only accuracy but also efficiency for scalability is important, some previous methods struggle to capture heterogeneity such as varying traffic patterns across regions and time periods. Moreover, Graph Neural Networks (GNNs), which are the mainstream of traffic prediction, not only require predefined adjacency matrix, but also limit scalability to large-scale data containing many nodes due to their inherent complexity. To overcome these limitations, we propose Deformable Dynamic Convolution Network (DDCN) for accurate yet efficient traffic prediction. Traditional Convolutional Neural Networks (CNNs) are limited in modeling non-Euclidean spatial structures and spatio-temporal heterogeneity, DDCN overcomes these challenges by dynamically applying deformable filters based on offset. Specifically, DDCN decomposes transformer-style CNN to encoder-decoder structure, and applies proposed approaches to the spatial and spatio-temporal attention blocks of the encoder to emphasize important features. The decoder, composed of feed-forward module, complements the output of the encoder. This novel structure make DDCN can perform accurate yet efficient traffic prediction. In comprehensive experiments on four real-world datasets, DDCN achieves competitive performance, emphasizing the potential and effectiveness of CNN-based approaches for spatio-temporal traffic prediction. 

**Abstract (ZH)**: 空间时间交通预测在智能交通系统中扮演着关键角色，通过在复杂城市区域实现准确预测。虽然准确性和可扩展性的效率同样重要，但一些先前的方法难以捕捉不同区域和时间周期下的异质性。此外，主流行驶中的图神经网络（GNNs）不仅需要预先定义的邻接矩阵，而且由于其固有的复杂性，难以处理包含大量节点的大型数据集。为克服这些限制，我们提出了可变形动态卷积网络（DDCN），以实现准确且高效的交通预测。传统的卷积神经网络（CNNs）在建模非欧几里得空间结构和空间时间异质性方面受到限制，DDCN通过基于偏移动态应用可变形滤波器来克服这些挑战。具体而言，DDCN将transformer风格的CNN分解为编码器-解码器结构，并将所提出的方法应用于编码器的空间和空间时间注意力模块，以强调重要特征。解码器由前馈模块组成，补充编码器的输出。这种新颖结构使DDCN能够实现准确且高效的交通预测。在四个真实世界数据集的全面实验中，DDCN取得了竞争力的表现，强调了基于CNN的方法在空间时间交通预测中的潜力和有效性。 

---
# An Memory-Efficient Framework for Deformable Transformer with Neural Architecture Search 

**Title (ZH)**: 一种基于神经架构搜索的内存高效变形 transformer 框架 

**Authors**: Wendong Mao, Mingfan Zhao, Jianfeng Guan, Qiwei Dong, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11549)  

**Abstract**: Deformable Attention Transformers (DAT) have shown remarkable performance in computer vision tasks by adaptively focusing on informative image regions. However, their data-dependent sampling mechanism introduces irregular memory access patterns, posing significant challenges for efficient hardware deployment. Existing acceleration methods either incur high hardware overhead or compromise model accuracy. To address these issues, this paper proposes a hardware-friendly optimization framework for DAT. First, a neural architecture search (NAS)-based method with a new slicing strategy is proposed to automatically divide the input feature into uniform patches during the inference process, avoiding memory conflicts without modifying model architecture. The method explores the optimal slice configuration by jointly optimizing hardware cost and inference accuracy. Secondly, an FPGA-based verification system is designed to test the performance of this framework on edge-side hardware. Algorithm experiments on the ImageNet-1K dataset demonstrate that our hardware-friendly framework can maintain have only 0.2% accuracy drop compared to the baseline DAT. Hardware experiments on Xilinx FPGA show the proposed method reduces DRAM access times to 18% compared with existing DAT acceleration methods. 

**Abstract (ZH)**: 变形注意力变压器（DAT）在计算机视觉任务中通过自适应关注信息性的图像区域展现了卓越的表现。然而，其数据依赖的采样机制引入了不规则的内存访问模式，给高效的硬件部署带来了重大挑战。现有的加速方法要么会导致高硬件开销，要么会损害模型的准确性。为解决这些问题，本文提出了一种面向硬件的DAT优化框架。首先，提出了一种基于神经架构搜索（NAS）的新切片策略，在推理过程中自动生成均匀的特征片，避免内存冲突，同时不修改模型架构。该方法通过联合优化硬件成本和推理准确性来探索最佳切片配置。其次，设计了一种基于FPGA的验证系统，用以测试该框架在边缘硬件上的性能。图像集ImageNet-1K上的算法实验表明，我们的面向硬件的框架与基线DAT相比仅Accuracy下降0.2%。Xilinx FPGA上的硬件实验表明，所提出的方法将DRAM访问时间减少了18%，优于现有DAT加速方法。 

---
# Fairness Is Not Enough: Auditing Competence and Intersectional Bias in AI-powered Resume Screening 

**Title (ZH)**: 公平性不够：AI驱动的简历筛选中的能力审计与交叉偏见审查 

**Authors**: Kevin T Webster  

**Link**: [PDF](https://arxiv.org/pdf/2507.11548)  

**Abstract**: The increasing use of generative AI for resume screening is predicated on the assumption that it offers an unbiased alternative to biased human decision-making. However, this belief fails to address a critical question: are these AI systems fundamentally competent at the evaluative tasks they are meant to perform? This study investigates the question of competence through a two-part audit of eight major AI platforms. Experiment 1 confirmed complex, contextual racial and gender biases, with some models penalizing candidates merely for the presence of demographic signals. Experiment 2, which evaluated core competence, provided a critical insight: some models that appeared unbiased were, in fact, incapable of performing a substantive evaluation, relying instead on superficial keyword matching. This paper introduces the "Illusion of Neutrality" to describe this phenomenon, where an apparent lack of bias is merely a symptom of a model's inability to make meaningful judgments. This study recommends that organizations and regulators adopt a dual-validation framework, auditing AI hiring tools for both demographic bias and demonstrable competence to ensure they are both equitable and effective. 

**Abstract (ZH)**: 生成式AI在简历筛选中的广泛应用假定其提供了无偏见的人类决策的替代方案。然而，这种信念未能回答一个关键问题：这些AI系统本质上是否具备完成所期望评估任务的能力？本研究通过审计八大主要AI平台的两个部分，调查了这一能力问题。实验1确认了复杂的、情境化的种族和性别偏见，一些模型仅因候选人的某些 demographics 信号而对其扣分。实验2评估核心能力，揭示了一个关键洞察：一些看似无偏见的模型实际上无法进行实质性的评估，而是依赖表面的关键词匹配。本文将这种现象称为“中立幻象”，即表面无偏见仅仅是模型无法做出有意义判断的症状。本研究建议组织和监管机构采用双重验证框架，审计AI招聘工具的种族偏见和可验证的能力，以确保它们既公平又有效。 

---
# A Review of Generative AI in Computer Science Education: Challenges and Opportunities in Accuracy, Authenticity, and Assessment 

**Title (ZH)**: 计算机科学教育中生成式AI的综述：准确性、真实性与评估面临的挑战与机遇 

**Authors**: Iman Reihanian, Yunfei Hou, Yu Chen, Yifei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.11543)  

**Abstract**: This paper surveys the use of Generative AI tools, such as ChatGPT and Claude, in computer science education, focusing on key aspects of accuracy, authenticity, and assessment. Through a literature review, we highlight both the challenges and opportunities these AI tools present. While Generative AI improves efficiency and supports creative student work, it raises concerns such as AI hallucinations, error propagation, bias, and blurred lines between AI-assisted and student-authored content. Human oversight is crucial for addressing these concerns. Existing literature recommends adopting hybrid assessment models that combine AI with human evaluation, developing bias detection frameworks, and promoting AI literacy for both students and educators. Our findings suggest that the successful integration of AI requires a balanced approach, considering ethical, pedagogical, and technical factors. Future research may explore enhancing AI accuracy, preserving academic integrity, and developing adaptive models that balance creativity with precision. 

**Abstract (ZH)**: 本研究调查了ChatGPT和Claude等生成式AI工具在计算机科学教育中的应用，重点关注准确度、真实性和评估的关键方面。通过文献综述，我们指出这些AI工具带来的挑战和机遇。尽管生成式AI提高了效率并支持学生的创造性工作，但也引发了AI幻觉、错误传播、偏见以及AI辅助内容与学生原创内容界限模糊等担忧。人类监督对于应对这些担忧至关重要。现有文献建议采用将AI与人工评估相结合的混合评估模型、开发偏见检测框架，并促进师生的AI素养。我们的研究结果表明，AI的成功集成需要综合考虑伦理、教学法和技术因素。未来的研究可以探索提高AI准确度、维护学术诚信以及开发既能平衡创造力又能提高精确度的适应性模型。 

---
