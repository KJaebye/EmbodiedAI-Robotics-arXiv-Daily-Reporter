# TLE-Based A2C Agent for Terrestrial Coverage Orbital Path Planning 

**Title (ZH)**: 基于TLE的A2C代理用于地表覆盖轨道路径规划 

**Authors**: Anantha Narayanan, Battu Bhanu Teja, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2508.10872)  

**Abstract**: The increasing congestion of Low Earth Orbit (LEO) poses persistent challenges to the efficient deployment and safe operation of Earth observation satellites. Mission planners must now account not only for mission-specific requirements but also for the increasing collision risk with active satellites and space debris. This work presents a reinforcement learning framework using the Advantage Actor-Critic (A2C) algorithm to optimize satellite orbital parameters for precise terrestrial coverage within predefined surface radii. By formulating the problem as a Markov Decision Process (MDP) within a custom OpenAI Gymnasium environment, our method simulates orbital dynamics using classical Keplerian elements. The agent progressively learns to adjust five of the orbital parameters - semi-major axis, eccentricity, inclination, right ascension of ascending node, and the argument of perigee-to achieve targeted terrestrial coverage. Comparative evaluation against Proximal Policy Optimization (PPO) demonstrates A2C's superior performance, achieving 5.8x higher cumulative rewards (10.0 vs 9.263025) while converging in 31.5x fewer timesteps (2,000 vs 63,000). The A2C agent consistently meets mission objectives across diverse target coordinates while maintaining computational efficiency suitable for real-time mission planning applications. Key contributions include: (1) a TLE-based orbital simulation environment incorporating physics constraints, (2) validation of actor-critic methods' superiority over trust region approaches in continuous orbital control, and (3) demonstration of rapid convergence enabling adaptive satellite deployment. This approach establishes reinforcement learning as a computationally efficient alternative for scalable and intelligent LEO mission planning. 

**Abstract (ZH)**: 低地球轨道日益严重的拥堵对地球观测卫星的有效部署和安全运行构成了持续挑战。任务规划人员必须不仅考虑任务特定需求，还要考虑与活跃卫星和空间碎片不断增加的碰撞风险。本文提出了一种基于优势actor-评论员（A2C）算法的强化学习框架，以优化卫星轨道参数，实现预定义地面半径内的精确陆地覆盖。通过在一个自定义的OpenAI Gymnasium环境中将问题形式化为马尔可夫决策过程（MDP），我们的方法使用经典开普勒元素模拟轨道动力学。代理逐进学会调整轨道参数——半长轴、偏心率、倾角、升交点赤经和近地点幅角——以实现目标陆地覆盖。相比近端策略优化（PPO）方法，A2C表现出更优性能，累计奖励高5.8倍（10.0 vs 9.263025），收敛所需时间步减少31.5倍（2,000 vs 63,000）。A2C代理能够在多种目标坐标下一致达到任务目标，同时保持适用于实时任务规划应用的计算效率。主要贡献包括：（1）基于TLE的轨道仿真环境，包含物理约束；（2）验证了在连续轨道控制中演员-评论员方法优于信赖域方法的优越性；（3）展示了快速收敛能力，使卫星部署更具适应性。该方法确立了强化学习作为计算高效、可扩展和智能低地球轨道任务规划的替代方案的地位。 

---
# Why Report Failed Interactions With Robots?! Towards Vignette-based Interaction Quality 

**Title (ZH)**: 为什么对话失败了？基于情境的交互质量探索。 

**Authors**: Agnes Axelsson, Merle Reimann, Ronald Cumbal, Hannah Pelikan, Divesh Lala  

**Link**: [PDF](https://arxiv.org/pdf/2508.10603)  

**Abstract**: Although the quality of human-robot interactions has improved with the advent of LLMs, there are still various factors that cause systems to be sub-optimal when compared to human-human interactions. The nature and criticality of failures are often dependent on the context of the interaction and so cannot be generalized across the wide range of scenarios and experiments which have been implemented in HRI research. In this work we propose the use of a technique overlooked in the field of HRI, ethnographic vignettes, to clearly highlight these failures, particularly those that are rarely documented. We describe the methodology behind the process of writing vignettes and create our own based on our personal experiences with failures in HRI systems. We emphasize the strength of vignettes as the ability to communicate failures from a multi-disciplinary perspective, promote transparency about the capabilities of robots, and document unexpected behaviours which would otherwise be omitted from research reports. We encourage the use of vignettes to augment existing interaction evaluation methods. 

**Abstract (ZH)**: 尽管大型语言模型的出现提高了人机交互的质量，但与人类之间的交互相比，系统仍存在各种因素导致其不够优化。交互的性质和失败的严重性通常依赖于交互的具体情境，因此难以在涵盖广泛场景和实验的HRI研究中进行通用化。本文提出在HRI领域忽视的一种技术——民族志案例研究，以清晰地展示这些失败，特别是那些很少被记录的失败。我们描述了撰写案例研究的方法，并基于自身在HRI系统中遇到的失败经验编写了自己的案例研究。我们强调案例研究的优势，包括从多学科视角沟通失败、促进关于机器人能力的透明度以及记录本应被研究报告忽略的意外行为。我们鼓励使用案例研究来增强现有的交互评估方法。 

---
# KDPE: A Kernel Density Estimation Strategy for Diffusion Policy Trajectory Selection 

**Title (ZH)**: KDPE: 一种核密度估计策略用于扩散政策轨迹选择 

**Authors**: Andrea Rosasco, Federico Ceola, Giulia Pasquale, Lorenzo Natale  

**Link**: [PDF](https://arxiv.org/pdf/2508.10511)  

**Abstract**: Learning robot policies that capture multimodality in the training data has been a long-standing open challenge for behavior cloning. Recent approaches tackle the problem by modeling the conditional action distribution with generative models. One of these approaches is Diffusion Policy, which relies on a diffusion model to denoise random points into robot action trajectories. While achieving state-of-the-art performance, it has two main drawbacks that may lead the robot out of the data distribution during policy execution. First, the stochasticity of the denoising process can highly impact on the quality of generated trajectory of actions. Second, being a supervised learning approach, it can learn data outliers from the dataset used for training. Recent work focuses on mitigating these limitations by combining Diffusion Policy either with large-scale training or with classical behavior cloning algorithms. Instead, we propose KDPE, a Kernel Density Estimation-based strategy that filters out potentially harmful trajectories output of Diffusion Policy while keeping a low test-time computational overhead. For Kernel Density Estimation, we propose a manifold-aware kernel to model a probability density function for actions composed of end-effector Cartesian position, orientation, and gripper state. KDPE overall achieves better performance than Diffusion Policy on simulated single-arm tasks and real robot experiments.
Additional material and code are available on our project page this https URL. 

**Abstract (ZH)**: 基于核密度估计的策略：Diffusion Policy的轨迹过滤方法 

---
# Scaling Up without Fading Out: Goal-Aware Sparse GNN for RL-based Generalized Planning 

**Title (ZH)**: 不断提高效率而不减淡目标意识：面向RL基于通用规划的心理稀疏GNN 

**Authors**: Sangwoo Jeon, Juchul Shin, Gyeong-Tae Kim, YeonJe Cho, Seongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.10747)  

**Abstract**: Generalized planning using deep reinforcement learning (RL) combined with graph neural networks (GNNs) has shown promising results in various symbolic planning domains described by PDDL. However, existing approaches typically represent planning states as fully connected graphs, leading to a combinatorial explosion in edge information and substantial sparsity as problem scales grow, especially evident in large grid-based environments. This dense representation results in diluted node-level information, exponentially increases memory requirements, and ultimately makes learning infeasible for larger-scale problems. To address these challenges, we propose a sparse, goal-aware GNN representation that selectively encodes relevant local relationships and explicitly integrates spatial features related to the goal. We validate our approach by designing novel drone mission scenarios based on PDDL within a grid world, effectively simulating realistic mission execution environments. Our experimental results demonstrate that our method scales effectively to larger grid sizes previously infeasible with dense graph representations and substantially improves policy generalization and success rates. Our findings provide a practical foundation for addressing realistic, large-scale generalized planning tasks. 

**Abstract (ZH)**: 使用深度强化学习（RL）结合图神经网络（GNNs）的广义规划在由PDDL描述的各种符号规划领域中取得了有希望的结果。然而，现有方法通常将规划状态表示为全连接图，这导致边信息的组合爆炸和随着问题规模增长而出现显著的稀疏性，尤其是在大型网格环境中尤为明显。这种密集表示会导致节点级信息的稀释，成指数地增加内存需求，并最终使得学习在大规模问题上变得不可行。为了解决这些挑战，我们提出了一种稀疏的目标导向的GNN表示方法，该方法选择性地编码相关的局部关系，并明确集成与目标相关的空间特征。我们通过在网格世界中基于PDDL设计新颖的无人机任务场景，有效模拟了现实的使命执行环境。实验结果表明，我们的方法能够有效地扩展到以前因密集图表示无法处理的更大网格规模，并显著提高了策略的一般化能力和成功率。我们的研究为解决现实的、大规模的广义规划任务提供了实用的基础。 

---
# Who Benefits from AI Explanations? Towards Accessible and Interpretable Systems 

**Title (ZH)**: AI解释的受益者是谁？面向可访问性和可解释性的系统研究 

**Authors**: Maria J. P. Peixoto, Akriti Pandey, Ahsan Zaman, Peter R. Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2508.10806)  

**Abstract**: As AI systems are increasingly deployed to support decision-making in critical domains, explainability has become a means to enhance the understandability of these outputs and enable users to make more informed and conscious choices. However, despite growing interest in the usability of eXplainable AI (XAI), the accessibility of these methods, particularly for users with vision impairments, remains underexplored. This paper investigates accessibility gaps in XAI through a two-pronged approach. First, a literature review of 79 studies reveals that evaluations of XAI techniques rarely include disabled users, with most explanations relying on inherently visual formats. Second, we present a four-part methodological proof of concept that operationalizes inclusive XAI design: (1) categorization of AI systems, (2) persona definition and contextualization, (3) prototype design and implementation, and (4) expert and user assessment of XAI techniques for accessibility. Preliminary findings suggest that simplified explanations are more comprehensible for non-visual users than detailed ones, and that multimodal presentation is required for more equitable interpretability. 

**Abstract (ZH)**: 随着AI系统在关键领域支持决策的应用越来越广泛，可解释性已成为提升这些输出可理解性的手段，帮助用户做出更知情和自觉的选择。然而，尽管对可解释性AI（XAI）的可用性越来越感兴趣，这些方法的无障碍性，尤其是对视觉障碍用户而言，尚未得到充分探索。本文通过两方面的研究探讨了XAI的无障碍缺口：首先，文献回顾79项研究发现，对XAI技术的评估很少包括残疾用户，大多数解释依赖于固有的视觉格式；其次，我们提出了一种四部分的方法论概念验证，以实现包容性XAI设计：（1）AI系统的分类，（2）角色定义和情境化，（3）原型设计与实现，以及（4）XAI技术的专家和用户无障碍评估。初步发现表明，简化解释比详细解释更易于非视觉用户理解，而多模态呈现对于更公平的可解释性是必要的。 

---
# Agentic Design Review System 

**Title (ZH)**: 代理设计评审系统 

**Authors**: Sayan Nag, K J Joseph, Koustava Goswami, Vlad I Morariu, Balaji Vasan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10745)  

**Abstract**: Evaluating graphic designs involves assessing it from multiple facets like alignment, composition, aesthetics and color choices. Evaluating designs in a holistic way involves aggregating feedback from individual expert reviewers. Towards this, we propose an Agentic Design Review System (AgenticDRS), where multiple agents collaboratively analyze a design, orchestrated by a meta-agent. A novel in-context exemplar selection approach based on graph matching and a unique prompt expansion method plays central role towards making each agent design aware. Towards evaluating this framework, we propose DRS-BENCH benchmark. Thorough experimental evaluation against state-of-the-art baselines adapted to the problem setup, backed-up with critical ablation experiments brings out the efficacy of Agentic-DRS in evaluating graphic designs and generating actionable feedback. We hope that this work will attract attention to this pragmatic, yet under-explored research direction. 

**Abstract (ZH)**: 评估图形设计涉及从对齐、构图、美学和颜色选择等多个方面进行评估。从整体上评估设计需要综合个体专家评审的反馈。为此，我们提出了一种代理设计审查系统（AgenticDRS），其中多个代理协同分析设计，由一个元代理协调。基于图匹配的新型上下文相关示例选择方法和独特的提示扩展方法在使每个代理了解设计方面发挥核心作用。为了评估该框架，我们提出了DRS-BENCH基准。通过针对问题设置适应的最先进的基线进行彻底的实验评估，并结合关键的消融实验，突显了Agentic-DRS在评估图形设计和生成可操作反馈方面的有效性。我们希望这项工作能够引起对这一实用且尚未充分探索的研究方向的关注。 

---
# STEP: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation 

**Title (ZH)**: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation 

**Authors**: Zhenye Yang, Jinpeng Chen, Huan Li, Xiongnan Jin, Xuanyang Li, Junwei Zhang, Hongbo Gao, Kaimin Wei, Senzhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10669)  

**Abstract**: Conversational recommender systems (CRSs) aim to proactively capture user preferences through natural language dialogue and recommend high-quality items. To achieve this, CRS gathers user preferences via a dialog module and builds user profiles through a recommendation module to generate appropriate recommendations. However, existing CRS faces challenges in capturing the deep semantics of user preferences and dialogue context. In particular, the efficient integration of external knowledge graph (KG) information into dialogue generation and recommendation remains a pressing issue. Traditional approaches typically combine KG information directly with dialogue content, which often struggles with complex semantic relationships, resulting in recommendations that may not align with user expectations.
To address these challenges, we introduce STEP, a conversational recommender centered on pre-trained language models that combines curriculum-guided context-knowledge fusion with lightweight task-specific prompt tuning. At its heart, an F-Former progressively aligns the dialogue context with knowledge-graph entities through a three-stage curriculum, thus resolving fine-grained semantic mismatches. The fused representation is then injected into the frozen language model via two minimal yet adaptive prefix prompts: a conversation prefix that steers response generation toward user intent and a recommendation prefix that biases item ranking toward knowledge-consistent candidates. This dual-prompt scheme allows the model to share cross-task semantics while respecting the distinct objectives of dialogue and recommendation. Experimental results show that STEP outperforms mainstream methods in the precision of recommendation and dialogue quality in two public datasets. 

**Abstract (ZH)**: 基于预训练语言模型的渐进式上下文-知识融合对话推荐系统（STEP） 

---
# We-Math 2.0: A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning 

**Title (ZH)**: We-Math 2.0: 一个激励视觉数学推理的多功能数学书系统 

**Authors**: Runqi Qiao, Qiuna Tan, Peiqing Yang, Yanzi Wang, Xiaowan Wang, Enhui Wan, Sitong Zhou, Guanting Dong, Yuchen Zeng, Yida Xu, Jie Wang, Chong Sun, Chen Li, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10433)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across various tasks, but still struggle with complex mathematical reasoning. Existing research primarily focuses on dataset construction and method optimization, often overlooking two critical aspects: comprehensive knowledge-driven design and model-centric data space modeling. In this paper, we introduce We-Math 2.0, a unified system that integrates a structured mathematical knowledge system, model-centric data space modeling, and a reinforcement learning (RL)-based training paradigm to comprehensively enhance the mathematical reasoning abilities of MLLMs. The key contributions of We-Math 2.0 are fourfold: (1) MathBook Knowledge System: We construct a five-level hierarchical system encompassing 491 knowledge points and 1,819 fundamental principles. (2) MathBook-Standard & Pro: We develop MathBook-Standard, a dataset that ensures broad conceptual coverage and flexibility through dual expansion. Additionally, we define a three-dimensional difficulty space and generate 7 progressive variants per problem to build MathBook-Pro, a challenging dataset for robust training. (3) MathBook-RL: We propose a two-stage RL framework comprising: (i) Cold-Start Fine-tuning, which aligns the model with knowledge-oriented chain-of-thought reasoning; and (ii) Progressive Alignment RL, leveraging average-reward learning and dynamic data scheduling to achieve progressive alignment across difficulty levels. (4) MathBookEval: We introduce a comprehensive benchmark covering all 491 knowledge points with diverse reasoning step distributions. Experimental results show that MathBook-RL performs competitively with existing baselines on four widely-used benchmarks and achieves strong results on MathBookEval, suggesting promising generalization in mathematical reasoning. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）在各种任务中展现了 impressive 的能力，但在复杂数学推理方面仍面临挑战。现有研究主要集中在数据集构建和方法优化上，常常忽视两个关键方面：全面的知识驱动设计和以模型为中心的数据空间建模。本文提出了 We-Math 2.0，这是一个统一系统，结合了结构化的数学知识系统、以模型为中心的数据空间建模和基于强化学习（RL）的训练范式，以全面增强 MLLMs 的数学推理能力。We-Math 2.0 的主要贡献包括四个方面：（1）MathBook 知识系统：构建了一个五级层次系统，包含 491 个知识点和 1,819 个基本原理。（2）MathBook-Standard & Pro：开发了 MathBook-Standard 数据集，通过双扩展确保广泛的概念覆盖面和灵活性。此外，定义了一个三维难度空间，并为每个问题生成 7 个递进变体，构建了 MathBook-Pro 这个具有挑战性的数据集，用于稳健训练。（3）MathBook-RL：提出一个两阶段 RL 框架，包括：启动阶段微调，使模型与知识导向的推理链相一致；以及渐进对准 RL，利用平均回报学习和动态数据调度实现不同难度层次的渐进对准。（4）MathBookEval：引入了一个全面的基准测试，涵盖所有 491 个知识点，具有多样的推理步骤分布。实验结果表明，MathBook-RL 在四个广泛使用的基准测试中表现出色，并在 MathBookEval 中取得了优异成绩，这表明其在数学推理中的潜在泛化能力。 

---
# HiRef: Leveraging Hierarchical Ontology and Network Refinement for Robust Medication Recommendation 

**Title (ZH)**: HiRef: 利用层次化本体和网络精炼实现稳健的药物推荐 

**Authors**: Yan Ting Chok, Soyon Park, Seungheun Baek, Hajung Kim, Junhyun Lee, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10425)  

**Abstract**: Medication recommendation is a crucial task for assisting physicians in making timely decisions from longitudinal patient medical records. However, real-world EHR data present significant challenges due to the presence of rarely observed medical entities and incomplete records that may not fully capture the clinical ground truth. While data-driven models trained on longitudinal Electronic Health Records often achieve strong empirical performance, they struggle to generalize under missing or novel conditions, largely due to their reliance on observed co-occurrence patterns. To address these issues, we propose Hierarchical Ontology and Network Refinement for Robust Medication Recommendation (HiRef), a unified framework that combines two complementary structures: (i) the hierarchical semantics encoded in curated medical ontologies, and (ii) refined co-occurrence patterns derived from real-world EHRs. We embed ontology entities in hyperbolic space, which naturally captures tree-like relationships and enables knowledge transfer through shared ancestors, thereby improving generalizability to unseen codes. To further improve robustness, we introduce a prior-guided sparse regularization scheme that refines the EHR co-occurrence graph by suppressing spurious edges while preserving clinically meaningful associations. Our model achieves strong performance on EHR benchmarks (MIMIC-III and MIMIC-IV) and maintains high accuracy under simulated unseen-code settings. Extensive experiments with comprehensive ablation studies demonstrate HiRef's resilience to unseen medical codes, supported by in-depth analyses of the learned sparsified graph structure and medical code embeddings. 

**Abstract (ZH)**: 基于层次 ontology 和网络 refinement 的稳健药物推荐 (HiRef) 

---
# Multi-Agent Trust Region Policy Optimisation: A Joint Constraint Approach 

**Title (ZH)**: 多代理信任区域策略优化：一种联合约束方法 

**Authors**: Chak Lam Shek, Guangyao Shi, Pratap Tokekar  

**Link**: [PDF](https://arxiv.org/pdf/2508.10340)  

**Abstract**: Multi-agent reinforcement learning (MARL) requires coordinated and stable policy updates among interacting agents. Heterogeneous-Agent Trust Region Policy Optimization (HATRPO) enforces per-agent trust region constraints using Kullback-Leibler (KL) divergence to stabilize training. However, assigning each agent the same KL threshold can lead to slow and locally optimal updates, especially in heterogeneous settings. To address this limitation, we propose two approaches for allocating the KL divergence threshold across agents: HATRPO-W, a Karush-Kuhn-Tucker-based (KKT-based) method that optimizes threshold assignment under global KL constraints, and HATRPO-G, a greedy algorithm that prioritizes agents based on improvement-to-divergence ratio. By connecting sequential policy optimization with constrained threshold scheduling, our approach enables more flexible and effective learning in heterogeneous-agent settings. Experimental results demonstrate that our methods significantly boost the performance of HATRPO, achieving faster convergence and higher final rewards across diverse MARL benchmarks. Specifically, HATRPO-W and HATRPO-G achieve comparable improvements in final performance, each exceeding 22.5%. Notably, HATRPO-W also demonstrates more stable learning dynamics, as reflected by its lower variance. 

**Abstract (ZH)**: 多智能体强化学习中基于KL散度的异质智能体信任区域策略优化（HATRPO）改进方法 

---
# Promoting Efficient Reasoning with Verifiable Stepwise Reward 

**Title (ZH)**: 促进高效推理的可验证逐步奖励方法 

**Authors**: Chuhuai Yue, Chengqi Dong, Yinan Gao, Hang He, Jiajun Chai, Guojun Yin, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.10293)  

**Abstract**: Large reasoning models (LRMs) have recently achieved significant progress in complex reasoning tasks, aided by reinforcement learning with verifiable rewards. However, LRMs often suffer from overthinking, expending excessive computation on simple problems and reducing efficiency. Existing efficient reasoning methods typically require accurate task assessment to preset token budgets or select reasoning modes, which limits their flexibility and reliability. In this work, we revisit the essence of overthinking and identify that encouraging effective steps while penalizing ineffective ones is key to its solution. To this end, we propose a novel rule-based verifiable stepwise reward mechanism (VSRM), which assigns rewards based on the performance of intermediate states in the reasoning trajectory. This approach is intuitive and naturally fits the step-by-step nature of reasoning tasks. We conduct extensive experiments on standard mathematical reasoning benchmarks, including AIME24 and AIME25, by integrating VSRM with PPO and Reinforce++. Results show that our method achieves substantial output length reduction while maintaining original reasoning performance, striking an optimal balance between efficiency and accuracy. Further analysis of overthinking frequency and pass@k score before and after training demonstrates that our approach in deed effectively suppresses ineffective steps and encourages effective reasoning, fundamentally alleviating the overthinking problem. All code will be released upon acceptance. 

**Abstract (ZH)**: 大型推理模型（LRMs）在复杂推理任务中取得了显著进展，借助可验证奖励的强化学习。然而，LRMs 通常会陷入过度思考，对简单问题耗用过多计算资源，降低效率。现有高效推理方法通常需要准确的任务评估来预设令牌预算或选择推理模式，这限制了其灵活性和可靠性。在本文中，我们重新审视过度思考的本质，并发现鼓励有效步骤同时惩罚无效步骤是其解决的关键。为实现这一目标，我们提出了一种新的基于规则的可验证步骤奖励机制（VSRM），该机制根据推理轨迹中中间状态的性能分配奖励。该方法直观且自然地符合推理任务的逐步性质。我们通过将VSRM与PPO和Reinforce++集成，在标准数学推理基准AIME24和AIME25上进行了广泛实验。结果表明，我们的方法在保持原始推理性能的同时实现显著的输出长度减少，实现了效率和准确性的最优平衡。进一步分析训练前后过度思考频率和pass@k得分表明，我们的方法实际上有效地抑制了无效步骤，促进了有效推理，从根本上缓解了过度思考问题。接受后将发布所有代码。 

---
# Extending the Entropic Potential of Events for Uncertainty Quantification and Decision-Making in Artificial Intelligence 

**Title (ZH)**: 扩展事件的_entropic潜力_用于人工智能中的不确定性量化与决策制定 

**Authors**: Mark Zilberman  

**Link**: [PDF](https://arxiv.org/pdf/2508.10241)  

**Abstract**: This work demonstrates how the concept of the entropic potential of events -- a parameter quantifying the influence of discrete events on the expected future entropy of a system -- can enhance uncertainty quantification, decision-making, and interpretability in artificial intelligence (AI). Building on its original formulation in physics, the framework is adapted for AI by introducing an event-centric measure that captures how actions, observations, or other discrete occurrences impact uncertainty at future time horizons. Both the original and AI-adjusted definitions of entropic potential are formalized, with the latter emphasizing conditional expectations to account for counterfactual scenarios. Applications are explored in policy evaluation, intrinsic reward design, explainable AI, and anomaly detection, highlighting the metric's potential to unify and strengthen uncertainty modeling in intelligent systems. Conceptual examples illustrate its use in reinforcement learning, Bayesian inference, and anomaly detection, while practical considerations for computation in complex AI models are discussed. The entropic potential framework offers a theoretically grounded, interpretable, and versatile approach to managing uncertainty in AI, bridging principles from thermodynamics, information theory, and machine learning. 

**Abstract (ZH)**: 本研究展示了事件熵潜能的概念——一个量化离散事件对未来系统预期熵影响的参数——如何增强人工智能中的不确定性量化、决策制定和可解释性。基于其在物理中的原始表述，该框架通过引入以事件为中心的度量来适应人工智能，该度量捕获了行动、观察或其他离散事件对未来时间 horizons 不确定性的影响。原定义和调整后的 AI 定义的熵潜能都被形式化，后者强调条件期望以考虑到假设场景。熵潜能的应用在政策评估、内在奖励设计、可解释人工智能和异常检测中得到了探索，突显了该度量在智能系统中统一和增强不确定性建模的潜力。概念性示例展示了其在强化学习、贝叶斯推断和异常检测中的应用，同时讨论了在复杂人工智能模型中计算的实用考虑。熵潜能框架提供了一种理论基础扎实、可解释且灵活的方法来管理人工智能中的不确定性，融合了热力学、信息论和机器学习的原则。 

---
# Pruning Long Chain-of-Thought of Large Reasoning Models via Small-Scale Preference Optimization 

**Title (ZH)**: 基于小规模偏好优化的大推理模型长链思考修剪 

**Authors**: Bin Hong, Jiayu Liu, Zhenya Huang, Kai Zhang, Mengdi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10164)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) have demonstrated strong performance on complex tasks through long Chain-of-Thought (CoT) reasoning. However, their lengthy outputs increase computational costs and may lead to overthinking, raising challenges in balancing reasoning effectiveness and efficiency. Current methods for efficient reasoning often compromise reasoning quality or require extensive resources. This paper investigates efficient methods to reduce the generation length of LRMs. We analyze generation path distributions and filter generated trajectories through difficulty estimation. Subsequently, we analyze the convergence behaviors of the objectives of various preference optimization methods under a Bradley-Terry loss based framework. Based on the analysis, we propose Length Controlled Preference Optimization (LCPO) that directly balances the implicit reward related to NLL loss. LCPO can effectively learn length preference with limited data and training. Extensive experiments demonstrate that our approach significantly reduces the average output length by over 50\% across multiple benchmarks while maintaining the reasoning performance. Our work highlights the potential for computationally efficient approaches in guiding LRMs toward efficient reasoning. 

**Abstract (ZH)**: Recent Advances in Large Reasoning Models (LRMs) and Efficient Generation Methods 

---
# Improving and Evaluating Open Deep Research Agents 

**Title (ZH)**: 改进和评估开源深度研究代理 

**Authors**: Doaa Allabadi, Kyle Bradbury, Jordan M. Malof  

**Link**: [PDF](https://arxiv.org/pdf/2508.10152)  

**Abstract**: We focus here on Deep Research Agents (DRAs), which are systems that can take a natural language prompt from a user, and then autonomously search for, and utilize, internet-based content to address the prompt. Recent DRAs have demonstrated impressive capabilities on public benchmarks however, recent research largely involves proprietary closed-source systems. At the time of this work, we only found one open-source DRA, termed Open Deep Research (ODR). In this work we adapt the challenging recent BrowseComp benchmark to compare ODR to existing proprietary systems. We propose BrowseComp-Small (BC-Small), comprising a subset of BrowseComp, as a more computationally-tractable DRA benchmark for academic labs. We benchmark ODR and two other proprietary systems on BC-Small: one system from Anthropic and one system from Google. We find that all three systems achieve 0% accuracy on the test set of 60 questions. We introduce three strategic improvements to ODR, resulting in the ODR+ model, which achieves a state-of-the-art 10% success rate on BC-Small among both closed-source and open-source systems. We report ablation studies indicating that all three of our improvements contributed to the success of ODR+. 

**Abstract (ZH)**: 我们专注于深度研究代理（DRAs），这些系统可以从用户处接收自然语言提示，然后自主搜索和利用互联网内容来应对提示。尽管最近的DRAs在公共基准测试中展现了令人印象深刻的性能，但大部分近期研究涉及专有和封闭源代码系统。在本研究进行时，我们仅发现一个开源DRA，称为Open Deep Research（ODR）。在这项工作中，我们将具有挑战性的近期BrowseComp基准测试改编，以比较ODR与现有的专有系统。我们提出BrowseComp-Small（BC-Small），它是BrowseComp的一部分，作为更易于计算的DRA学术实验室基准。我们对ODR及两个其他专有系统在BC-Small上的表现进行了基准测试：来自Anthropic的一个系统和来自Google的一个系统。我们发现，所有三个系统在包含60个问题的测试集上的准确率为0%。我们引入了三种战略性的改进，从而生成了ODR+模型，在BC-Small基准上，ODR+模型在专有和开源系统中达到了最佳的10%成功率。我们报告的消融研究显示，我们的三种改进均对ODR+的成功做出了贡献。 

---
# MCP-Orchestrated Multi-Agent System for Automated Disinformation Detection 

**Title (ZH)**: MCP- orchestrated 多代理系统用于自动化虚假信息检测 

**Authors**: Alexandru-Andrei Avram, Adrian Groza, Alexandru Lecu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10143)  

**Abstract**: The large spread of disinformation across digital platforms creates significant challenges to information integrity. This paper presents a multi-agent system that uses relation extraction to detect disinformation in news articles, focusing on titles and short text snippets. The proposed Agentic AI system combines four agents: (i) a machine learning agent (logistic regression), (ii) a Wikipedia knowledge check agent (which relies on named entity recognition), (iii) a coherence detection agent (using LLM prompt engineering), and (iv) a web-scraped data analyzer that extracts relational triplets for fact checking. The system is orchestrated via the Model Context Protocol (MCP), offering shared context and live learning across components. Results demonstrate that the multi-agent ensemble achieves 95.3% accuracy with an F1 score of 0.964, significantly outperforming individual agents and traditional approaches. The weighted aggregation method, mathematically derived from individual agent misclassification rates, proves superior to algorithmic threshold optimization. The modular architecture makes the system easily scalable, while also maintaining details of the decision processes. 

**Abstract (ZH)**: 数字平台上的虚假信息广泛传播对信息完整性构成了重大挑战。本文提出了一种多Agent系统，利用关系提取检测新闻文章中的虚假信息，专注于标题和短文本片段。提出的Agentic AI系统结合了四个Agent：（i）机器学习Agent（逻辑回归），（ii）维基百科知识验证Agent（依赖命名实体识别），（iii）一致性检测Agent（使用LLM提示工程），以及（iv）网页抓取数据分析师，提取关系三元组进行事实核查。系统通过模型上下文协议（MCP）协调，提供跨组件的共享上下文和实时学习。结果表明，多Agent集合的准确率为95.3%，F1分数为0.964，显著优于单个Agent和传统方法。加权聚合方法，从单个Agent的错误分类率中数学推导而来，优于算法阈值优化。模块化的架构使得系统易于扩展，同时保持决策过程的详细信息。 

---
# Empirical Investigation into Configuring Echo State Networks for Representative Benchmark Problem Domains 

**Title (ZH)**: 关于配置回声状态网络以解决代表性基准问题领域的一种实证研究 

**Authors**: Brooke R. Weborg, Gursel Serpen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10887)  

**Abstract**: This paper examines Echo State Network, a reservoir computer, performance using four different benchmark problems, then proposes heuristics or rules of thumb for configuring the architecture, as well as the selection of parameters and their values, which are applicable to problems within the same domain, to help serve to fill the experience gap needed by those entering this field of study. The influence of various parameter selections and their value adjustments, as well as architectural changes made to an Echo State Network, a powerful recurrent neural network configured as a reservoir computer, can be challenging to fully comprehend without experience in the field, and even some hyperparameter optimization algorithms may have difficulty adjusting parameter values without proper manual selections made first. Therefore, it is imperative to understand the effects of parameters and their value selection on Echo State Network architecture performance for a successful build. Thus, to address the requirement for an extensive background in Echo State Network architecture, as well as examine how Echo State Network performance is affected with respect to variations in architecture, design, and parameter selection and values, a series of benchmark tasks representing different problem domains, including time series prediction, pattern generation, chaotic system prediction, and time series classification, were modeled and experimented on to show the impact on the performance of Echo State Network. 

**Abstract (ZH)**: 本文使用四个不同的基准问题考查Echo State Network（回声状态网络）的表现，并提出适用于相同领域问题的架构配置、参数选择及其值的启发式规则或经验法则，以帮助填补研究领域新手所需的经验差距。探讨架构变化、参数选择及其值调整对Echo State Network（一种配置为蓄水库的强循环神经网络）性能的影响可能会由于缺乏领域经验而显得复杂，即使一些超参数优化算法也可能难以在没有适当手动选择的情况下调整参数值。因此，了解参数及其值选择对Echo State Network架构性能的影响对于成功构建该模型至关重要。为了应对Echo State Network架构所需广泛背景知识的需求，并考察架构、设计、参数选择及其值变化对Echo State Network性能的影响，本文通过建模和实验一系列代表不同问题域的基准任务，包括时间序列预测、模式生成、混沌系统预测和时间序列分类，展示了Echo State Network性能的变化影响。 

---
# From Black Box to Transparency: Enhancing Automated Interpreting Assessment with Explainable AI in College Classrooms 

**Title (ZH)**: 从黑箱到透明：通过可解释AI提升大学课堂自动解释评估 

**Authors**: Zhaokun Jiang, Ziyin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10860)  

**Abstract**: Recent advancements in machine learning have spurred growing interests in automated interpreting quality assessment. Nevertheless, existing research suffers from insufficient examination of language use quality, unsatisfactory modeling effectiveness due to data scarcity and imbalance, and a lack of efforts to explain model predictions. To address these gaps, we propose a multi-dimensional modeling framework that integrates feature engineering, data augmentation, and explainable machine learning. This approach prioritizes explainability over ``black box'' predictions by utilizing only construct-relevant, transparent features and conducting Shapley Value (SHAP) analysis. Our results demonstrate strong predictive performance on a novel English-Chinese consecutive interpreting dataset, identifying BLEURT and CometKiwi scores to be the strongest predictive features for fidelity, pause-related features for fluency, and Chinese-specific phraseological diversity metrics for language use. Overall, by placing particular emphasis on explainability, we present a scalable, reliable, and transparent alternative to traditional human evaluation, facilitating the provision of detailed diagnostic feedback for learners and supporting self-regulated learning advantages not afforded by automated scores in isolation. 

**Abstract (ZH)**: Recent advancements in machine learning have spurred growing interests in automated interpreting quality assessment. Nevertheless, existing research suffers from insufficient examination of language use quality, unsatisfactory modeling effectiveness due to data scarcity and imbalance, and a lack of efforts to explain model predictions. To address these gaps, we propose a multi-dimensional modeling framework that integrates feature engineering, data augmentation, and explainable machine learning. This approach prioritizes explainability over "black box" predictions by utilizing only construct-relevant, transparent features and conducting Shapley Value (SHAP) analysis. Our results demonstrate strong predictive performance on a novel English-Chinese consecutive interpreting dataset, identifying BLEURT and CometKiwi scores to be the strongest predictive features for fidelity, pause-related features for fluency, and Chinese-specific phraseological diversity metrics for language use.

综合这些方面的考虑，我们提出了一种多维度建模框架，该框架结合了特征工程、数据增强和可解释的人工智能。该方法优先考虑可解释性而非“黑箱”预测，仅利用相关且透明的特征，并进行Shapley值（SHAP）分析。我们在一个新颖的英译汉连续口译数据集上展示了该方法的强大预测性能，识别出BLEURT和CometKiwi评分是忠实性最强的预测特征，停顿相关特征是流畅性最强的预测特征，而特定于汉语的习语多样性指标是语言使用质量最强的预测特征。通过特别强调可解释性，我们提出了一种可扩展、可靠且透明的人机评价替代方案，为学习者提供详细的诊断反馈，并支持自动评分无法单独提供的自我调节学习优势。 

---
# Enhancing Fairness in Autoencoders for Node-Level Graph Anomaly Detection 

**Title (ZH)**: 增强节点级别图异常检测自动编码器的公平性 

**Authors**: Shouju Wang, Yuchen Song, Sheng'en Li, Dongmian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10785)  

**Abstract**: Graph anomaly detection (GAD) has become an increasingly important task across various domains. With the rapid development of graph neural networks (GNNs), GAD methods have achieved significant performance improvements. However, fairness considerations in GAD remain largely underexplored. Indeed, GNN-based GAD models can inherit and amplify biases present in training data, potentially leading to unfair outcomes. While existing efforts have focused on developing fair GNNs, most approaches target node classification tasks, where models often rely on simple layer architectures rather than autoencoder-based structures, which are the most widely used architecturs for anomaly detection. To address fairness in autoencoder-based GAD models, we propose \textbf{D}is\textbf{E}ntangled \textbf{C}ounterfactual \textbf{A}dversarial \textbf{F}air (DECAF)-GAD, a framework that alleviates bias while preserving GAD performance. Specifically, we introduce a structural causal model (SCM) to disentangle sensitive attributes from learned representations. Based on this causal framework, we formulate a specialized autoencoder architecture along with a fairness-guided loss function. Through extensive experiments on both synthetic and real-world datasets, we demonstrate that DECAF-GAD not only achieves competitive anomaly detection performance but also significantly enhances fairness metrics compared to baseline GAD methods. Our code is available at this https URL. 

**Abstract (ZH)**: 基于解纠缠反事实对抗的图异常检测（DECAF-GAD） 

---
# Estimating Covariance for Global Minimum Variance Portfolio: A Decision-Focused Learning Approach 

**Title (ZH)**: 基于决策导向的学习方法估计全局最小方差组合的协方差,eg：基于决策导向的学习方法估计全局最小方差Portfolio的协方程西侧 

**Authors**: Juchan Kim, Inwoo Tae, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.10776)  

**Abstract**: Portfolio optimization constitutes a cornerstone of risk management by quantifying the risk-return trade-off. Since it inherently depends on accurate parameter estimation under conditions of future uncertainty, the selection of appropriate input parameters is critical for effective portfolio construction. However, most conventional statistical estimators and machine learning algorithms determine these parameters by minimizing mean-squared error (MSE), a criterion that can yield suboptimal investment decisions. In this paper, we adopt decision-focused learning (DFL) - an approach that directly optimizes decision quality rather than prediction error such as MSE - to derive the global minimum-variance portfolio (GMVP). Specifically, we theoretically derive the gradient of decision loss using the analytic solution of GMVP and its properties regarding the principal components of itself. Through extensive empirical evaluation, we show that prediction-focused estimation methods may fail to produce optimal allocations in practice, whereas DFL-based methods consistently deliver superior decision performance. Furthermore, we provide a comprehensive analysis of DFL's mechanism in GMVP construction, focusing on its volatility reduction capability, decision-driving features, and estimation characteristics. 

**Abstract (ZH)**: 投资组合优化构成了风险管理的基石，通过量化收益与风险的权衡。由于其本身依赖于未来不确定性条件下准确的参数估计，因此选择合适的输入参数对于有效的投资组合构建至关重要。然而，大多数传统的统计估计器和机器学习算法通过最小化均方误差（MSE）来确定这些参数，这种方法可能产生次优的投资决策。在本文中，我们采用决策聚焦学习（DFL）——一种直接优化决策质量而非预测误差（如MSE）的方法——来推导全局最小方差投资组合（GMVP）。具体地，我们利用GMVP的解析解及其自身主成分的性质，理论上推导出决策损失的梯度。通过广泛的实证评估，我们展示了预测驱动的估计方法可能在实践中无法产生最优分配，而基于DFL的方法则始终能提供更优越的决策性能。此外，我们对DFL在构建GMVP过程中的机制进行了全面分析，着重讨论了其波动性降低能力、决策驱动特征和估计特性。 

---
# FROGENT: An End-to-End Full-process Drug Design Agent 

**Title (ZH)**: FROGENT：一个全流程药物设计代理 

**Authors**: Qihua Pan, Dong Xu, Jenna Xinyi Yao, Lijia Ma, Zexuan Zhu, Junkai Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.10760)  

**Abstract**: Powerful AI tools for drug discovery reside in isolated web apps, desktop programs, and code libraries. Such fragmentation forces scientists to manage incompatible interfaces and specialized scripts, which can be a cumbersome and repetitive process. To address this issue, a Full-pROcess druG dEsign ageNT, named FROGENT, has been proposed. Specifically, FROGENT utilizes a Large Language Model and the Model Context Protocol to integrate multiple dynamic biochemical databases, extensible tool libraries, and task-specific AI models. This agentic framework allows FROGENT to execute complicated drug discovery workflows dynamically, including component tasks such as target identification, molecule generation and retrosynthetic planning. FROGENT has been evaluated on eight benchmarks that cover various aspects of drug discovery, such as knowledge retrieval, property prediction, virtual screening, mechanistic analysis, molecular design, and synthesis. It was compared against six increasingly advanced ReAct-style agents that support code execution and literature searches. Empirical results demonstrated that FROGENT triples the best baseline performance in hit-finding and doubles it in interaction profiling, significantly outperforming both the open-source model Qwen3-32B and the commercial model GPT-4o. In addition, real-world cases have been utilized to validate the practicability and generalization of FROGENT. This development suggests that streamlining the agentic drug discovery pipeline can significantly enhance researcher productivity. 

**Abstract (ZH)**: 基于全生命周期药物设计的强大AI代理工具：FROGENT 

---
# Natively Trainable Sparse Attention for Hierarchical Point Cloud Datasets 

**Title (ZH)**: 本源可训练的稀疏注意力机制 for 分层点云数据集 

**Authors**: Nicolas Lapautre, Maria Marchenko, Carlos Miguel Patiño, Xin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10758)  

**Abstract**: Unlocking the potential of transformers on datasets of large physical systems depends on overcoming the quadratic scaling of the attention mechanism. This work explores combining the Erwin architecture with the Native Sparse Attention (NSA) mechanism to improve the efficiency and receptive field of transformer models for large-scale physical systems, addressing the challenge of quadratic attention complexity. We adapt the NSA mechanism for non-sequential data, implement the Erwin NSA model, and evaluate it on three datasets from the physical sciences -- cosmology simulations, molecular dynamics, and air pressure modeling -- achieving performance that matches or exceeds that of the original Erwin model. Additionally, we reproduce the experimental results from the Erwin paper to validate their implementation. 

**Abstract (ZH)**: 大规模物理系统数据集上Transformer潜力的释放取决于克服注意力机制的 quadr 增长。本工作探索将 Erwin 架构与原生稀疏注意力（NSA）机制相结合，以提高 Transformer 模型对大规模物理系统效率和感受野，解决注意力机制的 quadr 增长挑战。我们为非序列数据适配 NSA 机制，实现 Erwin NSA 模型，并在来自物理科学的三个数据集——宇宙学模拟、分子动力学和气压建模——上进行评估，实现性能与原版 Erwin 模型相当或超越。此外，我们重现 Erwin 研究论文中的实验结果以验证其实现。 

---
# APFL: Analytic Personalized Federated Learning via Dual-Stream Least Squares 

**Title (ZH)**: APFL: 分析个性化联邦学习通过双流最小二乘法 

**Authors**: Kejia Fan, Jianheng Tang, Zhirui Yang, Feijiang Han, Jiaxu Li, Run He, Yajiang Huang, Anfeng Liu, Houbing Herbert Song, Yunhuai Liu, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10732)  

**Abstract**: Personalized Federated Learning (PFL) has presented a significant challenge to deliver personalized models to individual clients through collaborative training. Existing PFL methods are often vulnerable to non-IID data, which severely hinders collective generalization and then compromises the subsequent personalization efforts. In this paper, to address this non-IID issue in PFL, we propose an Analytic Personalized Federated Learning (APFL) approach via dual-stream least squares. In our APFL, we use a foundation model as a frozen backbone for feature extraction. Subsequent to the feature extractor, we develop dual-stream analytic models to achieve both collective generalization and individual personalization. Specifically, our APFL incorporates a shared primary stream for global generalization across all clients, and a dedicated refinement stream for local personalization of each individual client. The analytical solutions of our APFL enable its ideal property of heterogeneity invariance, theoretically meaning that each personalized model remains identical regardless of how heterogeneous the data are distributed across all other clients. Empirical results across various datasets also validate the superiority of our APFL over state-of-the-art baselines, with advantages of at least 1.10%-15.45% in accuracy. 

**Abstract (ZH)**: 基于双流最小二乘的分析个性化联邦学习（APFL） 

---
# Electromagnetic Simulations of Antennas on GPUs for Machine Learning Applications 

**Title (ZH)**: 基于GPU的天线电磁仿真及其在机器学习中的应用 

**Authors**: Murat Temiz, Vemund Bakken  

**Link**: [PDF](https://arxiv.org/pdf/2508.10713)  

**Abstract**: This study proposes an antenna simulation framework powered by graphics processing units (GPUs) based on an open-source electromagnetic (EM) simulation software (gprMax) for machine learning applications of antenna design and optimization. Furthermore, it compares the simulation results with those obtained through commercial EM software. The proposed software framework for machine learning and surrogate model applications will produce antenna data sets consisting of a large number of antenna simulation results using GPUs. Although machine learning methods can attain the optimum solutions for many problems, they are known to be data-hungry and require a great deal of samples for the training stage of the algorithms. However, producing a sufficient number of training samples in EM applications within a limited time is challenging due to the high computational complexity of EM simulations. Therefore, GPUs are utilized in this study to simulate a large number of antennas with predefined or random antenna shape parameters to produce data sets. Moreover, this study also compares various machine learning and deep learning models in terms of antenna parameter estimation performance. This study demonstrates that an entry-level GPU substantially outperforms a high-end CPU in terms of computational performance, while a high-end gaming GPU can achieve around 18 times more computational performance compared to a high-end CPU. Moreover, it is shown that the open-source EM simulation software can deliver similar results to those obtained via commercial software in the simulation of microstrip antennas when the spatial resolution of the simulations is sufficiently fine. 

**Abstract (ZH)**: 基于图形处理单元（GPU）的动力学处理单元（GPUGrant）开源电磁（EM）仿真软件的天线设计与优化的机器学习应用仿真框架研究 

---
# Deep Learning in Classical and Quantum Physics 

**Title (ZH)**: 深度学习在经典物理与量子物理中的应用 

**Authors**: Timothy Heightman, Marcin Płodzień  

**Link**: [PDF](https://arxiv.org/pdf/2508.10666)  

**Abstract**: Scientific progress is tightly coupled to the emergence of new research tools. Today, machine learning (ML)-especially deep learning (DL)-has become a transformative instrument for quantum science and technology. Owing to the intrinsic complexity of quantum systems, DL enables efficient exploration of large parameter spaces, extraction of patterns from experimental data, and data-driven guidance for research directions. These capabilities already support tasks such as refining quantum control protocols and accelerating the discovery of materials with targeted quantum properties, making ML/DL literacy an essential skill for the next generation of quantum scientists. At the same time, DL's power brings risks: models can overfit noisy data, obscure causal structure, and yield results with limited physical interpretability. Recognizing these limitations and deploying mitigation strategies is crucial for scientific rigor. These lecture notes provide a comprehensive, graduate-level introduction to DL for quantum applications, combining conceptual exposition with hands-on examples. Organized as a progressive sequence, they aim to equip readers to decide when and how to apply DL effectively, to understand its practical constraints, and to adapt AI methods responsibly to problems across quantum physics, chemistry, and engineering. 

**Abstract (ZH)**: 科学进步与新研究工具的出现紧密相关。今天，机器学习（ML），尤其是深度学习（DL），已成为量子科学和技术的变革性工具。由于量子系统的固有复杂性，DL能够高效地探索大型参数空间、从实验数据中提取模式，并基于数据指导研究方向。这些能力已经支持诸如 refinement of quantum control protocols 和加速发现具有目标量子性质的材料等任务，使得ML/DL能力成为下一代量子科学家的必备技能。同时，DL的强大功能也带来了风险：模型可能会过度拟合噪声数据，掩盖因果结构，导致结果缺乏物理可解释性。意识这些局限性和部署缓解策略对于科学严谨性至关重要。这些讲义提供了量子应用领域的DL全面且适合研究生水平的介绍，结合了概念阐述与实际案例。它们旨在帮助读者决定何时以及如何有效应用DL，理解其实用限制，并负责任地将AI方法应用于量子物理、化学和工程领域的各种问题。 

---
# SPHENIC: Topology-Informed Multi-View Clustering for Spatial Transcriptomics 

**Title (ZH)**: Sphenic: 拓扑引导的多视图聚类方法应用于空间转录组学 

**Authors**: Chenkai Guo, Yikai Zhu, Jing Yangum, Renxiang Guan, Por Lip Yee, Guangdun Peng, Dayu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10646)  

**Abstract**: By incorporating spatial location information, spatial-transcriptomics clustering yields more comprehensive insights into cell subpopulation identification. Despite recent progress, existing methods have at least two limitations: (i) topological learning typically considers only representations of individual cells or their interaction graphs; however, spatial transcriptomic profiles are often noisy, making these approaches vulnerable to low-quality topological signals, and (ii) insufficient modeling of spatial neighborhood information leads to low-quality spatial embeddings. To address these limitations, we propose SPHENIC, a novel Spatial Persistent Homology Enhanced Neighborhood Integrative Clustering method. Specifically, SPHENIC incorporates invariant topological features into the clustering network to achieve stable representation learning. Additionally, to construct high-quality spatial embeddings that reflect the true cellular distribution, we design the Spatial Constraint and Distribution Optimization Module (SCDOM). This module increases the similarity between a cell's embedding and those of its spatial neighbors, decreases similarity with non-neighboring cells, and thereby produces clustering-friendly spatial embeddings. Extensive experiments on 14 benchmark spatial transcriptomic slices demonstrate that SPHENIC achieves superior performance on the spatial clustering task, outperforming existing state-of-the-art methods by 3.31%-6.54% over the best alternative. 

**Abstract (ZH)**: 通过整合空间位置信息，空间转录组学聚类提供了更全面的细胞亚群识别洞察。尽管近期有所进展，现有方法至少存在两个局限性：（i）拓扑学习通常仅考虑单个细胞的表示或它们的交互图；然而，空间转录组学资料往往噪声较大，使这些方法容易受到低质量拓扑信号的影响；（ii）空间邻域信息建模不足导致较低质量的空间嵌入。为解决这些局限，我们提出了一种新型的空间持久同胚增强邻域整合聚类方法 SPHENIC。具体而言，SPHENIC 将不变的拓扑特征整合到聚类网络中，以实现稳定的表现学习。此外，为了构建反映真实细胞分布的高质量空间嵌入，我们设计了空间约束和分布优化模块（SCDOM）。该模块增加了细胞嵌入与空间邻域细胞嵌入之间的相似性，减少了与非邻域细胞之间的相似性，从而生成有利于聚类的空间嵌入。在 14 个基准空间转录组学切片上进行的广泛实验表明，SPHENIC 在空间聚类任务中的性能优于现有最先进的方法，相较于最佳替代方法提高了 3.31%-6.54%。 

---
# On Spectral Properties of Gradient-based Explanation Methods 

**Title (ZH)**: 基于梯度的解释方法的谱性质研究 

**Authors**: Amir Mehrpanah, Erik Englesson, Hossein Azizpour  

**Link**: [PDF](https://arxiv.org/pdf/2508.10595)  

**Abstract**: Understanding the behavior of deep networks is crucial to increase our confidence in their results. Despite an extensive body of work for explaining their predictions, researchers have faced reliability issues, which can be attributed to insufficient formalism. In our research, we adopt novel probabilistic and spectral perspectives to formally analyze explanation methods. Our study reveals a pervasive spectral bias stemming from the use of gradient, and sheds light on some common design choices that have been discovered experimentally, in particular, the use of squared gradient and input perturbation. We further characterize how the choice of perturbation hyperparameters in explanation methods, such as SmoothGrad, can lead to inconsistent explanations and introduce two remedies based on our proposed formalism: (i) a mechanism to determine a standard perturbation scale, and (ii) an aggregation method which we call SpectralLens. Finally, we substantiate our theoretical results through quantitative evaluations. 

**Abstract (ZH)**: 深入理解深度网络的行为对于增强我们对它们结果的信心至关重要。尽管已有大量研究致力于解释其预测结果，研究人员仍面临可靠性问题，这可以归因于缺乏足够的形式化方法。在我们的研究中，我们采用新颖的概率和谱视角对解释方法进行正式分析。我们的研究揭示了由于使用梯度而导致的普遍谱偏差，并阐明了一些实验中发现的常见设计选择，特别是使用平方梯度和输入扰动。我们进一步分析了在解释方法中，如SmoothGrad，扰动超参数选择如何导致不一致的解释，并基于我们提出的形式化方法提出了两种补救措施：(i) 确定标准扰动尺度的机制，(ii) 称为SpectralLens的聚合方法。最后，我们通过定量评估验证了我们的理论结果。 

---
# FreeGAD: A Training-Free yet Effective Approach for Graph Anomaly Detection 

**Title (ZH)**: FreeGAD: 一种无需训练的有效图异常检测方法 

**Authors**: Yunfeng Zhao, Yixin Liu, Shiyuan Li, Qingfeng Chen, Yu Zheng, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10594)  

**Abstract**: Graph Anomaly Detection (GAD) aims to identify nodes that deviate from the majority within a graph, playing a crucial role in applications such as social networks and e-commerce. Despite the current advancements in deep learning-based GAD, existing approaches often suffer from high deployment costs and poor scalability due to their complex and resource-intensive training processes. Surprisingly, our empirical findings suggest that the training phase of deep GAD methods, commonly perceived as crucial, may actually contribute less to anomaly detection performance than expected. Inspired by this, we propose FreeGAD, a novel training-free yet effective GAD method. Specifically, it leverages an affinity-gated residual encoder to generate anomaly-aware representations. Meanwhile, FreeGAD identifies anchor nodes as pseudo-normal and anomalous guides, followed by calculating anomaly scores through anchor-guided statistical deviations. Extensive experiments demonstrate that FreeGAD achieves superior anomaly detection performance, efficiency, and scalability on multiple benchmark datasets from diverse domains, without any training or iterative optimization. 

**Abstract (ZH)**: 无监督图异常检测（FreeGAD）：一种训练-free的有效图异常检测方法 

---
# Fake Speech Wild: Detecting Deepfake Speech on Social Media Platform 

**Title (ZH)**: 假语音泛滥：社交媒体平台上的深度伪造语音检测 

**Authors**: Yuankun Xie, Ruibo Fu, Xiaopeng Wang, Zhiyong Wang, Ya Li, Zhengqi Wen, Haonnan Cheng, Long Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.10559)  

**Abstract**: The rapid advancement of speech generation technology has led to the widespread proliferation of deepfake speech across social media platforms. While deepfake audio countermeasures (CMs) achieve promising results on public datasets, their performance degrades significantly in cross-domain scenarios. To advance CMs for real-world deepfake detection, we first propose the Fake Speech Wild (FSW) dataset, which includes 254 hours of real and deepfake audio from four different media platforms, focusing on social media. As CMs, we establish a benchmark using public datasets and advanced selfsupervised learning (SSL)-based CMs to evaluate current CMs in real-world scenarios. We also assess the effectiveness of data augmentation strategies in enhancing CM robustness for detecting deepfake speech on social media. Finally, by augmenting public datasets and incorporating the FSW training set, we significantly advanced real-world deepfake audio detection performance, achieving an average equal error rate (EER) of 3.54% across all evaluation sets. 

**Abstract (ZH)**: 快速发展的语音生成技术导致了深度伪造语音在社交媒体平台上的广泛传播。尽管深度伪造音频对抗措施（CMs）在公共数据集中取得了令人鼓舞的结果，但在跨域场景中的性能显著下降。为了推动CMs在现实世界的深度伪造检测应用，我们首先提出了Fake Speech Wild（FSW）数据集，该数据集包含来自四个不同媒体平台的254小时真实和深度伪造音频，重点关注社交媒体。作为CMs，我们使用公共数据集和先进的自监督学习（SSL）基于的CMs建立了基准，以评估当前CMs在现实世界场景中的表现。我们也评估了数据增强策略在增强CM检测社交媒体中的深度伪造语音的鲁棒性方面的有效性。最后，通过扩充公共数据集并结合FSW训练集，我们显著提升了现实世界的深度伪造音频检测性能，各评估集的平均等错误率（EER）为3.54%。 

---
# Retrieval-Augmented Prompt for OOD Detection 

**Title (ZH)**: 用于OOD检测的检索增强提示 

**Authors**: Ruisong Han, Zongbo Han, Jiahao Zhang, Mingyue Cheng, Changqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10556)  

**Abstract**: Out-of-Distribution (OOD) detection is crucial for the reliable deployment of machine learning models in-the-wild, enabling accurate identification of test samples that differ from the training data distribution. Existing methods rely on auxiliary outlier samples or in-distribution (ID) data to generate outlier information for training, but due to limited outliers and their mismatch with real test OOD samples, they often fail to provide sufficient semantic supervision, leading to suboptimal performance. To address this, we propose a novel OOD detection method called Retrieval-Augmented Prompt (RAP). RAP augments a pre-trained vision-language model's prompts by retrieving external knowledge, offering enhanced semantic supervision for OOD detection. During training, RAP retrieves descriptive words for outliers based on joint similarity with external textual knowledge and uses them to augment the model's OOD prompts. During testing, RAP dynamically updates OOD prompts in real-time based on the encountered OOD samples, enabling the model to rapidly adapt to the test environment. Our extensive experiments demonstrate that RAP achieves state-of-the-art performance on large-scale OOD detection benchmarks. For example, in 1-shot OOD detection on the ImageNet-1k dataset, RAP reduces the average FPR95 by 7.05% and improves the AUROC by 1.71% compared to previous methods. Additionally, comprehensive ablation studies validate the effectiveness of each module and the underlying motivations of our approach. 

**Abstract (ZH)**: Out-of-Distribution (OOD)检测对于机器学习模型的可靠野外部署至关重要，能够准确识别与训练数据分布不同的测试样本。现有方法依赖于辅助异常样本或在分布（ID）数据来生成异常信息用于训练，但由于异常样本有限且与真实测试OOD样本不匹配，这些方法往往无法提供足够的语义监督，导致性能不佳。为解决这一问题，我们提出了一种新型的OOD检测方法称为检索增强提示（RAP）。RAP通过检索外部知识来增强预训练的跨模态模型的提示，提供增强的语义监督用于OOD检测。在训练阶段，RAP基于与外部文本知识的联合相似性检索异常描述词，并用于增强模型的OOD提示。在测试阶段，RAP根据遇到的OOD样本实时动态更新OOD提示，使模型能够快速适应测试环境。我们的实验表明，RAP在大规模OOD检测基准上达到了最先进的性能。例如，在ImageNet-1k数据集上的1-shot OOD检测中，RAP将平均FPR95降低了7.05%，AUROC提升了1.71%，优于以往方法。此外，全面的消融研究验证了每个模块的有效性及我们方法背后的动机。 

---
# Stabilizing Long-term Multi-turn Reinforcement Learning with Gated Rewards 

**Title (ZH)**: 使用门控奖励稳定长期多轮强化学习 

**Authors**: Zetian Sun, Dongfang Li, Zhuoen Chen, Yuhuai Qin, Baotian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10548)  

**Abstract**: Reward sparsity in long-horizon reinforcement learning (RL) tasks remains a significant challenge, while existing outcome-based reward shaping struggles to define meaningful immediate rewards without introducing bias or requiring explicit task decomposition. Alternatively, verification-based reward shaping uses stepwise critics, but misalignment between immediate rewards and long-term objectives can lead to reward hacking and suboptimal policies. In this work, we address this problem in the context of software engineering (SWE) tasks, where multi-turn reasoning and rule-based verification are critical. We introduce the SWE-oriented RL Framework, a unified system supporting multi-turn interaction, docker-based execution, and customizable reward functions. Additionally, we propose Gated Reward Accumulation (G-RA), a novel method that accumulates immediate rewards only when high-level (long-term) rewards meet a predefined threshold, ensuring stable RL optimization. Experiments on SWE-bench Verified and kBench demonstrate that G-RA leads to an increase in completion rates (47.6\% \rightarrow 93.8\% and 22.0\% \rightarrow 86.0\%) and modification rates (19.6\% \rightarrow 23.8\% and 12.0\% \rightarrow 42.0\%), while avoiding policy degradation caused by reward misalignment. Our findings highlight the importance of balanced reward accumulation in long-horizon RL and provide a practical solution. 

**Abstract (ZH)**: 长时程强化学习中奖励稀疏性的处理仍然是一个重大挑战，现有的基于结果的奖励塑造难以定义没有偏见的即时奖励或无需明确任务分解。相比之下，基于验证的奖励塑造使用逐步评论员，但即时奖励与长期目标之间的不一致可能导致奖励作弊和次优策略。在软件工程任务的背景下，我们解决了这一问题，其中多轮推理和基于规则的验证至关重要。我们引入了面向软件工程的RL框架，该框架是一个支持多轮交互、基于Docker的执行和可定制奖励函数的统一系统。此外，我们提出了门控奖励累积（G-RA）方法，该方法仅在高层（长期）奖励达到预定义阈值时累积即时奖励，从而确保RL优化的稳定性。在SWE-bench Verified和kBench上的实验表明，G-RA提高了完成率（47.6% → 93.8%和22.0% → 86.0%）和修改率（19.6% → 23.8%和12.0% → 42.0%），同时避免了由奖励不一致引起的策略退化。我们的研究结果强调了在长时程RL中平衡奖励累积的重要性，并提供了一种实用的解决方案。 

---
# Advances in Logic-Based Entity Resolution: Enhancing ASPEN with Local Merges and Optimality Criteria 

**Title (ZH)**: 基于逻辑的实体解析进展：通过局部合并和最优准则增强ASPEN 

**Authors**: Zhliang Xiang, Meghyn Bienvenu, Gianluca Cima, Víctor Gutiérrez-Basulto, Yazmín Ibáñez-García  

**Link**: [PDF](https://arxiv.org/pdf/2508.10504)  

**Abstract**: In this paper, we present ASPEN+, which extends an existing ASP-based system, ASPEN,for collective entity resolution with two important functionalities: support for local merges and new optimality criteria for preferred solutions. Indeed, ASPEN only supports so-called global merges of entity-referring constants (e.g. author ids), in which all occurrences of matched constants are treated as equivalent and merged accordingly. However, it has been argued that when resolving data values, local merges are often more appropriate, as e.g. some instances of 'J. Lee' may refer to 'Joy Lee', while others should be matched with 'Jake Lee'. In addition to allowing such local merges, ASPEN+ offers new optimality criteria for selecting solutions, such as minimizing rule violations or maximising the number of rules supporting a merge. Our main contributions are thus (1) the formalisation and computational analysis of various notions of optimal solution, and (2) an extensive experimental evaluation on real-world datasets, demonstrating the effect of local merges and the new optimality criteria on both accuracy and runtime. 

**Abstract (ZH)**: AS titled "这篇 paper的标题是：

基于现有的ASP基于的ASPEN，的ASPEN扩展，，集体实体解析，，，两个功能特性 on：支持局部合并和 最优化准则 for 优选项合并 on。..。因此 on 传统的ASPEN仅支持全局常量合并 on 即实体引用常量（例如 如例如 类似于e e.j lee 的 e 例如 e Joy Lee e 与其 e e e.e e Lee对应 �认为前者更适合 on 例如 例如 例如 e e e.e Jake Lee e �اض此外 on 本 ASPEN e还 �还 � e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e � e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e.e 最后 e 的 主 e e e 的 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e-initialized on on e e e e e e e e e e e e e e 上 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最大优化 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e(e 最e e e e e e 第 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最优化 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e拆迁和 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e the e proposed e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最新 e e e e e e e e e e e e 最示 e e e e e e e e e e e e e e e e e e e e e e e e 最 e e e e e e 两大贡献包括 e e e e e e e e 上: e e e e e e e e e e e e e e e e e e 最 e e e e � e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最完备的形式和 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 最模式和 e e e e e e e e e e e 最计算 e e e e e e e e e e e e e e e e e e e e e e e e 最全 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e(easd génér
 � titl in t s as t t e t t e e t t e: 埶 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e entitled � e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e pestic e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 

---
# Contrastive ECOC: Learning Output Codes for Adversarial Defense 

**Title (ZH)**: 对比性ECOC：用于对抗性防御的学习输出码 

**Authors**: Che-Yu Chou, Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10491)  

**Abstract**: Although one-hot encoding is commonly used for multiclass classification, it is not always the most effective encoding mechanism. Error Correcting Output Codes (ECOC) address multiclass classification by mapping each class to a unique codeword used as a label. Traditional ECOC methods rely on manually designed or randomly generated codebooks, which are labor-intensive and may yield suboptimal, dataset-agnostic results. This paper introduces three models for automated codebook learning based on contrastive learning, allowing codebooks to be learned directly and adaptively from data. Across four datasets, our proposed models demonstrate superior robustness to adversarial attacks compared to two baselines. The source is available at this https URL. 

**Abstract (ZH)**: 虽然one-hot编码常用于多类分类，但并非总是最有效的编码机制。错误修正输出编码（ECOC）通过将每个类别映射到一个唯一的码字来解决多类分类问题，该码字用作标签。传统的ECOC方法依赖于手动设计或随机生成的码本，这需要大量人工劳动，并可能得到不适用于特定数据集的次优结果。本文介绍三种基于对比学习的自动码本学习模型，使码本可以直接从数据中学习和适应。在四个数据集上，我们提出的模型在对抗攻击中的鲁棒性优于两种基线方法。源代码可在以下链接获取。 

---
# On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations 

**Title (ZH)**: 基于梯度的解释的复杂性忠实性权衡 

**Authors**: Amir Mehrpanah, Matteo Gamba, Kevin Smith, Hossein Azizpour  

**Link**: [PDF](https://arxiv.org/pdf/2508.10490)  

**Abstract**: ReLU networks, while prevalent for visual data, have sharp transitions, sometimes relying on individual pixels for predictions, making vanilla gradient-based explanations noisy and difficult to interpret. Existing methods, such as GradCAM, smooth these explanations by producing surrogate models at the cost of faithfulness. We introduce a unifying spectral framework to systematically analyze and quantify smoothness, faithfulness, and their trade-off in explanations. Using this framework, we quantify and regularize the contribution of ReLU networks to high-frequency information, providing a principled approach to identifying this trade-off. Our analysis characterizes how surrogate-based smoothing distorts explanations, leading to an ``explanation gap'' that we formally define and measure for different post-hoc methods. Finally, we validate our theoretical findings across different design choices, datasets, and ablations. 

**Abstract (ZH)**: ReLU网络在视觉数据中广泛应用，但存在尖锐的过渡，有时依赖个别像素进行预测，导致基于梯度的解释方法噪声较大且难以解释。现有的方法，如GradCAM，通过构建替代模型来平滑这些解释，但会以降低忠实性为代价。我们提出了一种统一的谱方法体系，系统地分析和量化平滑性、忠实性及其在解释中的权衡。利用该框架，我们量化并正则化ReLU网络对高频信息的贡献，提供了一种识别这种权衡的原理性方法。我们的分析描述了基于替代模型的平滑如何扭曲解释，从而正式定义并测量不同事后方法的“解释差距”。最后，我们在不同的设计选择、数据集和消融实验中验证了我们的理论发现。 

---
# Pinet: Optimizing hard-constrained neural networks with orthogonal projection layers 

**Title (ZH)**: Pinet：通过正交投影层优化具有约束条件的神经网络 

**Authors**: Panagiotis D. Grontas, Antonio Terpin, Efe C. Balta, Raffaello D'Andrea, John Lygeros  

**Link**: [PDF](https://arxiv.org/pdf/2508.10480)  

**Abstract**: We introduce an output layer for neural networks that ensures satisfaction of convex constraints. Our approach, $\Pi$net, leverages operator splitting for rapid and reliable projections in the forward pass, and the implicit function theorem for backpropagation. We deploy $\Pi$net as a feasible-by-design optimization proxy for parametric constrained optimization problems and obtain modest-accuracy solutions faster than traditional solvers when solving a single problem, and significantly faster for a batch of problems. We surpass state-of-the-art learning approaches in terms of training time, solution quality, and robustness to hyperparameter tuning, while maintaining similar inference times. Finally, we tackle multi-vehicle motion planning with non-convex trajectory preferences and provide $\Pi$net as a GPU-ready package implemented in JAX with effective tuning heuristics. 

**Abstract (ZH)**: 我们引入了一种输出层，确保神经网络满足凸约束。我们的方法$\Pi$net利用分裂算子在前向传递中进行快速可靠的投影，并利用隐函数定理进行反向传播。我们将$\Pi$net部署为参数约束优化问题的可行设计优化代理，在解决单个问题时比传统求解器更快获得适度准确的解决方案，对于批量问题则显著更快。在训练时间、解的质量和对超参数调优的鲁棒性方面，$\Pi$net超越了最先进的学习方法，同时保持类似的推理时间。最后，我们处理具有非凸轨迹偏好的多车辆运动规划问题，并提供基于JAX实现的$\Pi$net GPU就绪包，配有有效的调优启发式方法。 

---
# Enhanced Sparse Point Cloud Data Processing for Privacy-aware Human Action Recognition 

**Title (ZH)**: 增强稀疏点云数据处理方法及其在隐私感知的人体动作识别中的应用 

**Authors**: Maimunatu Tunau, Vincent Gbouna Zakka, Zhuangzhuang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2508.10469)  

**Abstract**: Human Action Recognition (HAR) plays a crucial role in healthcare, fitness tracking, and ambient assisted living technologies. While traditional vision based HAR systems are effective, they pose privacy concerns. mmWave radar sensors offer a privacy preserving alternative but present challenges due to the sparse and noisy nature of their point cloud data. In the literature, three primary data processing methods: Density-Based Spatial Clustering of Applications with Noise (DBSCAN), the Hungarian Algorithm, and Kalman Filtering have been widely used to improve the quality and continuity of radar data. However, a comprehensive evaluation of these methods, both individually and in combination, remains lacking. This paper addresses that gap by conducting a detailed performance analysis of the three methods using the MiliPoint dataset. We evaluate each method individually, all possible pairwise combinations, and the combination of all three, assessing both recognition accuracy and computational cost. Furthermore, we propose targeted enhancements to the individual methods aimed at improving accuracy. Our results provide crucial insights into the strengths and trade-offs of each method and their integrations, guiding future work on mmWave based HAR systems 

**Abstract (ZH)**: 毫米波雷达传感器在人体动作识别中的隐私保护作用及其数据处理方法研究：基于MiliPoint数据集的综合性能分析与优化 

---
# RealAC: A Domain-Agnostic Framework for Realistic and Actionable Counterfactual Explanations 

**Title (ZH)**: RealAC：一种面向现实且可行动的反事实解释的领域无关框架 

**Authors**: Asiful Arefeen, Shovito Barua Soumma, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2508.10455)  

**Abstract**: Counterfactual explanations provide human-understandable reasoning for AI-made decisions by describing minimal changes to input features that would alter a model's prediction. To be truly useful in practice, such explanations must be realistic and feasible -- they should respect both the underlying data distribution and user-defined feasibility constraints. Existing approaches often enforce inter-feature dependencies through rigid, hand-crafted constraints or domain-specific knowledge, which limits their generalizability and ability to capture complex, nonlinear relations inherent in data. Moreover, they rarely accommodate user-specified preferences and suggest explanations that are causally implausible or infeasible to act upon. We introduce RealAC, a domain-agnostic framework for generating realistic and actionable counterfactuals. RealAC automatically preserves complex inter-feature dependencies without relying on explicit domain knowledge -- by aligning the joint distributions of feature pairs between factual and counterfactual instances. The framework also allows end-users to ``freeze'' attributes they cannot or do not wish to change by suppressing change in frozen features during optimization. Evaluations on three synthetic and two real datasets demonstrate that RealAC balances realism with actionability. Our method outperforms state-of-the-art baselines and Large Language Model-based counterfactual generation techniques in causal edge score, dependency preservation score, and IM1 realism metric and offers a solution for causality-aware and user-centric counterfactual generation. 

**Abstract (ZH)**: RealAC：一种领域无关的生成现实性和可操作性解释的框架 

---
# Alternating Approach-Putt Models for Multi-Stage Speech Enhancement 

**Title (ZH)**: 交替方法putt模型在多阶段语音增强中的应用 

**Authors**: Iksoon Jeong, Kyung-Joong Kim, Kang-Hun Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2508.10436)  

**Abstract**: Speech enhancement using artificial neural networks aims to remove noise from noisy speech signals while preserving the speech content. However, speech enhancement networks often introduce distortions to the speech signal, referred to as artifacts, which can degrade audio quality. In this work, we propose a post-processing neural network designed to mitigate artifacts introduced by speech enhancement models. Inspired by the analogy of making a `Putt' after an `Approach' in golf, we name our model PuttNet. We demonstrate that alternating between a speech enhancement model and the proposed Putt model leads to improved speech quality, as measured by perceptual quality scores (PESQ), objective intelligibility (STOI), and background noise intrusiveness (CBAK) scores. Furthermore, we illustrate with graphical analysis why this alternating Approach outperforms repeated application of either model alone. 

**Abstract (ZH)**: 使用人工神经网络的语音增强旨在从噪声语音信号中去除噪声同时保留语音内容。然而，语音增强网络往往会引入对语音信号的失真，称为伪影，这会降低音频质量。在本工作中，我们提出了一种后处理神经网络，旨在减轻由语音增强模型引入的伪影。受高尔夫中“推杆”和“接近杆”操作的启发，我们将我们的模型命名为PuttNet。我们证明交替使用语音增强模型和提议的Putt模型可以提高语音质量，可以通过感知质量评分（PESQ）、客观可懂度（STOI）和背景噪声侵入性（CBAK）评分来衡量。此外，我们通过图形分析说明了这种交替的“接近杆”操作为什么优于单独重复应用任一模型。 

---
# Unpacking the Implicit Norm Dynamics of Sharpness-Aware Minimization in Tensorized Models 

**Title (ZH)**: 拆解张量化模型中隐含规范动态的锐化感知最小化 italia 劚
user
拆解锐化感知最小化在张量量化模型中的隐含规范动力学。 

**Authors**: Tianxiao Cao, Kyohei Atarashi, Hisashi Kashima  

**Link**: [PDF](https://arxiv.org/pdf/2508.10435)  

**Abstract**: Sharpness-Aware Minimization (SAM) has been proven to be an effective optimization technique for improving generalization in overparameterized models. While prior works have explored the implicit regularization of SAM in simple two-core scale-invariant settings, its behavior in more general tensorized or scale-invariant models remains underexplored. In this work, we leverage scale-invariance to analyze the norm dynamics of SAM in general tensorized models. We introduce the notion of \emph{Norm Deviation} as a global measure of core norm imbalance, and derive its evolution under SAM using gradient flow analysis. We show that SAM's implicit control of Norm Deviation is governed by the covariance between core norms and their gradient magnitudes. Motivated by these findings, we propose a simple yet effective method, \emph{Deviation-Aware Scaling (DAS)}, which explicitly mimics this regularization behavior by scaling core norms in a data-adaptive manner. Our experiments across tensor completion, noisy training, model compression, and parameter-efficient fine-tuning confirm that DAS achieves competitive or improved performance over SAM, while offering reduced computational overhead. 

**Abstract (ZH)**: 具有规范偏差意识的缩放（Deviation-Aware Scaling, DAS）：在广义张量化模型中分析尖锐度感知最小化（Sharpness-Aware Minimization, SAM）的范数动力学 

---
# PQ-DAF: Pose-driven Quality-controlled Data Augmentation for Data-scarce Driver Distraction Detection 

**Title (ZH)**: 基于pose驱动的质量控制数据增强方法：在数据稀缺条件下的驾驶员分心检测 

**Authors**: Haibin Sun, Xinghui Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.10397)  

**Abstract**: Driver distraction detection is essential for improving traffic safety and reducing road accidents. However, existing models often suffer from degraded generalization when deployed in real-world scenarios. This limitation primarily arises from the few-shot learning challenge caused by the high cost of data annotation in practical environments, as well as the substantial domain shift between training datasets and target deployment conditions. To address these issues, we propose a Pose-driven Quality-controlled Data Augmentation Framework (PQ-DAF) that leverages a vision-language model for sample filtering to cost-effectively expand training data and enhance cross-domain robustness. Specifically, we employ a Progressive Conditional Diffusion Model (PCDMs) to accurately capture key driver pose features and synthesize diverse training examples. A sample quality assessment module, built upon the CogVLM vision-language model, is then introduced to filter out low-quality synthetic samples based on a confidence threshold, ensuring the reliability of the augmented dataset. Extensive experiments demonstrate that PQ-DAF substantially improves performance in few-shot driver distraction detection, achieving significant gains in model generalization under data-scarce conditions. 

**Abstract (ZH)**: 基于姿态驱动的质量控制数据增强框架（PQ-DAF）用于改进驾驶员注意力分散检测的泛化能力 

---
# eMamba: Efficient Acceleration Framework for Mamba Models in Edge Computing 

**Title (ZH)**: eMamba：边缘计算中Mamba模型的高效加速框架 

**Authors**: Jiyong Kim, Jaeho Lee, Jiahao Lin, Alish Kanani, Miao Sun, Umit Y. Ogras, Jaehyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.10370)  

**Abstract**: State Space Model (SSM)-based machine learning architectures have recently gained significant attention for processing sequential data. Mamba, a recent sequence-to-sequence SSM, offers competitive accuracy with superior computational efficiency compared to state-of-the-art transformer models. While this advantage makes Mamba particularly promising for resource-constrained edge devices, no hardware acceleration frameworks are currently optimized for deploying it in such environments. This paper presents eMamba, a comprehensive end-to-end hardware acceleration framework explicitly designed for deploying Mamba models on edge platforms. eMamba maximizes computational efficiency by replacing complex normalization layers with lightweight hardware-aware alternatives and approximating expensive operations, such as SiLU activation and exponentiation, considering the target applications. Then, it performs an approximation-aware neural architecture search (NAS) to tune the learnable parameters used during approximation. Evaluations with Fashion-MNIST, CIFAR-10, and MARS, an open-source human pose estimation dataset, show eMamba achieves comparable accuracy to state-of-the-art techniques using 1.63-19.9$\times$ fewer parameters. In addition, it generalizes well to large-scale natural language tasks, demonstrating stable perplexity across varying sequence lengths on the WikiText2 dataset. We also quantize and implement the entire eMamba pipeline on an AMD ZCU102 FPGA and ASIC using GlobalFoundries (GF) 22 nm technology. Experimental results show 4.95-5.62$\times$ lower latency and 2.22-9.95$\times$ higher throughput, with 4.77$\times$ smaller area, 9.84$\times$ lower power, and 48.6$\times$ lower energy consumption than baseline solutions while maintaining competitive accuracy. 

**Abstract (ZH)**: 基于状态空间模型（SSM）的机器学习架构近年来在处理序列数据方面获得了显著关注。Mamba是一种最近提出的序列到序列SSM，与最先进的变压器模型相比，其在精度上具有竞争力，并且具有更高的计算效率。尽管这一优势使Mamba特别适合资源受限的边缘设备，但目前尚无硬件加速框架针对此类环境优化部署Mamba模型。本文提出了一种名为eMamba的全面端到端硬件加速框架，专门设计用于在边缘平台上部署Mamba模型。eMamba通过用轻量级的硬件感知替代复杂归一化层，并在目标应用考虑情况下近似昂贵的操作（如SiLU激活和指数运算），从而最大化计算效率。然后，它进行感知近似的神经架构搜索（NAS），以调整近似过程中使用的可学习参数。对于Fashion-MNIST、CIFAR-10以及公开的人体姿态估计数据集MARS，实验结果显示eMamba在使用1.63-19.9倍更少的参数时，实现了与最先进的技术相当的精度。此外，eMamba在大规模自然语言任务中具有良好的泛化能力，在WikiText2数据集中表现出稳定的变化序列长度下的困惑度。我们还在AMD ZCU102 FPGA和ASIC上用GlobalFoundries (GF) 22 nm技术对整个eMamba管道进行了量化和实现。实验结果表明，与基线解决方案相比，eMamba具有4.95-5.62倍更低的延迟、2.22-9.95倍更高的吞吐量、4.77倍更小的面积、9.84倍更低的功耗和48.6倍更低的能量消耗，同时保持了竞争力的精度。 

---
# Welfare-Centric Clustering 

**Title (ZH)**: 福利中心聚类 

**Authors**: Claire Jie Zhang, Seyed A. Esmaeili, Jamie Morgenstern  

**Link**: [PDF](https://arxiv.org/pdf/2508.10345)  

**Abstract**: Fair clustering has traditionally focused on ensuring equitable group representation or equalizing group-specific clustering costs. However, Dickerson et al. (2025) recently showed that these fairness notions may yield undesirable or unintuitive clustering outcomes and advocated for a welfare-centric clustering approach that models the utilities of the groups. In this work, we model group utilities based on both distances and proportional representation and formalize two optimization objectives based on welfare-centric clustering: the Rawlsian (Egalitarian) objective and the Utilitarian objective. We introduce novel algorithms for both objectives and prove theoretical guarantees for them. Empirical evaluations on multiple real-world datasets demonstrate that our methods significantly outperform existing fair clustering baselines. 

**Abstract (ZH)**: 公平聚类 traditionally focuses on ensuring equitable group representation or equalizing group-specific clustering costs. However, Dickerson et al. (2025) recently showed that these fairness notions may yield undesirable or unintuitive clustering outcomes and advocated for a welfare-centric clustering approach that models the utilities of the groups. In this work, we model group utilities based on both distances and proportional representation and formalize two optimization objectives based on welfare-centric clustering: the Rawlsian (Egalitarian) objective and the Utilitarian objective. We introduce novel algorithms for both objectives and prove theoretical guarantees for them. Empirical evaluations on multiple real-world datasets demonstrate that our methods significantly outperform existing fair clustering baselines. 

---
# Layer-Wise Analysis of Self-Supervised Representations for Age and Gender Classification in Children's Speech 

**Title (ZH)**: 自我监督表示在儿童语音年龄和性别分类中的逐层分析 

**Authors**: Abhijit Sinha, Harishankar Kumar, Mohit Joshi, Hemant Kumar Kathania, Shrikanth Narayanan, Sudarsana Reddy Kadiri  

**Link**: [PDF](https://arxiv.org/pdf/2508.10332)  

**Abstract**: Children's speech presents challenges for age and gender classification due to high variability in pitch, articulation, and developmental traits. While self-supervised learning (SSL) models perform well on adult speech tasks, their ability to encode speaker traits in children remains underexplored. This paper presents a detailed layer-wise analysis of four Wav2Vec2 variants using the PFSTAR and CMU Kids datasets. Results show that early layers (1-7) capture speaker-specific cues more effectively than deeper layers, which increasingly focus on linguistic information. Applying PCA further improves classification, reducing redundancy and highlighting the most informative components. The Wav2Vec2-large-lv60 model achieves 97.14% (age) and 98.20% (gender) on CMU Kids; base-100h and large-lv60 models reach 86.05% and 95.00% on PFSTAR. These results reveal how speaker traits are structured across SSL model depth and support more targeted, adaptive strategies for child-aware speech interfaces. 

**Abstract (ZH)**: 儿童语音的年龄和性别分类由于音高、发音和发育特征的高度变异性而具有挑战性。虽然自我监督学习模型在成人语音任务中表现良好，但它们在编码儿童特征方面的能力仍需进一步探索。本文通过对PFSTAR和CMU Kids数据集使用四种Wav2Vec2变体进行逐层分析，展示了早期层（1-7）比深层层更有效地捕捉到语音特定线索，并且主成分分析进一步提高了分类效果，减少冗余并突出最相关信息。Wav2Vec2-large-lv60模型在CMU Kids上的年龄分类准确率为97.14%，性别分类准确率为98.20%；base-100h和large-lv60模型在PFSTAR上的准确率分别为86.05%和95.00%。这些结果揭示了语音特定特征在自我监督学习模型中的结构，并支持更为针对性和适应性的儿童感知语音界面策略。 

---
# A Vision-Language Pre-training Model-Guided Approach for Mitigating Backdoor Attacks in Federated Learning 

**Title (ZH)**: 基于视觉-语言预训练模型的 Federated Learning 中后门攻击缓解方法 

**Authors**: Keke Gai, Dongjue Wang, Jing Yu, Liehuang Zhu, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10315)  

**Abstract**: Existing backdoor defense methods in Federated Learning (FL) rely on the assumption of homogeneous client data distributions or the availability of a clean serve dataset, which limits the practicality and effectiveness. Defending against backdoor attacks under heterogeneous client data distributions while preserving model performance remains a significant challenge. In this paper, we propose a FL backdoor defense framework named CLIP-Fed, which leverages the zero-shot learning capabilities of vision-language pre-training models. By integrating both pre-aggregation and post-aggregation defense strategies, CLIP-Fed overcomes the limitations of Non-IID imposed on defense effectiveness. To address privacy concerns and enhance the coverage of the dataset against diverse triggers, we construct and augment the server dataset using the multimodal large language model and frequency analysis without any client samples. To address class prototype deviations caused by backdoor samples and eliminate the correlation between trigger patterns and target labels, CLIP-Fed aligns the knowledge of the global model and CLIP on the augmented dataset using prototype contrastive loss and Kullback-Leibler divergence. Extensive experiments on representative datasets validate the effectiveness of CLIP-Fed. Compared to state-of-the-art methods, CLIP-Fed achieves an average reduction in ASR, i.e., 2.03\% on CIFAR-10 and 1.35\% on CIFAR-10-LT, while improving average MA by 7.92\% and 0.48\%, respectively. 

**Abstract (ZH)**: 基于零样本学习的CLIP-Fed： federated learning中的后门防御框架 

---
# ReviewRL: Towards Automated Scientific Review with RL 

**Title (ZH)**: ReviewRL：向基于强化学习的自动化科学研究评议迈进 

**Authors**: Sihang Zeng, Kai Tian, Kaiyan Zhang, Yuru wang, Junqi Gao, Runze Liu, Sa Yang, Jingxuan Li, Xinwei Long, Jiaheng Ma, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.10308)  

**Abstract**: Peer review is essential for scientific progress but faces growing challenges due to increasing submission volumes and reviewer fatigue. Existing automated review approaches struggle with factual accuracy, rating consistency, and analytical depth, often generating superficial or generic feedback lacking the insights characteristic of high-quality human reviews. We introduce ReviewRL, a reinforcement learning framework for generating comprehensive and factually grounded scientific paper reviews. Our approach combines: (1) an ArXiv-MCP retrieval-augmented context generation pipeline that incorporates relevant scientific literature, (2) supervised fine-tuning that establishes foundational reviewing capabilities, and (3) a reinforcement learning procedure with a composite reward function that jointly enhances review quality and rating accuracy. Experiments on ICLR 2025 papers demonstrate that ReviewRL significantly outperforms existing methods across both rule-based metrics and model-based quality assessments. ReviewRL establishes a foundational framework for RL-driven automatic critique generation in scientific discovery, demonstrating promising potential for future development in this domain. The implementation of ReviewRL will be released at GitHub. 

**Abstract (ZH)**: 同行评审对于科学进步至关重要，但面对提交量增加和评审疲劳的挑战，其面临着日益增长的困难。现有的自动化评审方法在事实准确性、评分一致性和分析深度方面遇到困难，常常生成表面化或通用的反馈，缺乏高质量人工评审具有的深入洞察。我们提出ReviewRL，这是一种基于强化学习的生成全面且基于事实的科研论文评审框架。该方法结合了：（1）一个通过ArXiv-MCP检索增强的语境生成管道，整合相关科学文献；（2）监督微调以建立基础评审能力；以及（3）一种强化学习过程，通过复合奖励函数共同提升评审质量和评分准确性。针对ICLR 2025论文的实验表明，ReviewRL在基于规则的评估指标和基于模型的质量评估中均显著优于现有方法。ReviewRL为基于强化学习的自动批评生成提供了基础框架，并展示了在该领域未来发展的前景。ReviewRL的实现将在GitHub上发布。 

---
# Pose-Robust Calibration Strategy for Point-of-Gaze Estimation on Mobile Phones 

**Title (ZH)**: 基于移动电话的眼球注视点估计的鲁棒校准策略 

**Authors**: Yujie Zhao, Jiabei Zeng, Shiguang Shan  

**Link**: [PDF](https://arxiv.org/pdf/2508.10268)  

**Abstract**: Although appearance-based point-of-gaze (PoG) estimation has improved, the estimators still struggle to generalize across individuals due to personal differences. Therefore, person-specific calibration is required for accurate PoG estimation. However, calibrated PoG estimators are often sensitive to head pose variations. To address this, we investigate the key factors influencing calibrated estimators and explore pose-robust calibration strategies. Specifically, we first construct a benchmark, MobilePoG, which includes facial images from 32 individuals focusing on designated points under either fixed or continuously changing head poses. Using this benchmark, we systematically analyze how the diversity of calibration points and head poses influences estimation accuracy. Our experiments show that introducing a wider range of head poses during calibration improves the estimator's ability to handle pose variation. Building on this insight, we propose a dynamic calibration strategy in which users fixate on calibration points while moving their phones. This strategy naturally introduces head pose variation during a user-friendly and efficient calibration process, ultimately producing a better calibrated PoG estimator that is less sensitive to head pose variations than those using conventional calibration strategies. Codes and datasets are available at our project page. 

**Abstract (ZH)**: 尽管基于外观的眼球凝视点（PoG）估计有所改进，但由于个体差异，估计器仍难以跨个体泛化。因此，为了实现准确的PoG估计，需要针对个体进行校准。然而，经过校准的PoG估计器往往对头部姿态变化敏感。为解决这一问题，我们调查了影响校准估计器的关键因素，并探索了姿态鲁棒的校准策略。具体而言，我们首先构建了一个基准数据集MobilePoG，其中包括来自32个个体的面部图像，这些图像在固定或连续变化的头部姿态下关注指定的点。使用该基准数据集，我们系统地分析了校准点的多样性和头部姿态的变化如何影响估计精度。实验结果显示，在校准过程中引入更广泛的头部姿态范围可以提高估计器处理姿态变化的能力。基于这一洞察，我们提出了一种动态校准策略，用户在移动手机时注视校准点。这一策略自然地在用户友好且高效的校准过程中引入头部姿态变化，最终生成了一个对头部姿态变化更不敏感的校准后的PoG估计器。相关代码和数据集可在我们的项目页面获取。 

---
# Facilitating Longitudinal Interaction Studies of AI Systems 

**Title (ZH)**: 促进人工智能系统 longitudinally 交互研究 

**Authors**: Tao Long, Sitong Wang, Émilie Fabre, Tony Wang, Anup Sathya, Jason Wu, Savvas Petridis, Dingzeyu Li, Tuhin Chakrabarty, Yue Jiang, Jingyi Li, Tiffany Tseng, Ken Nakagaki, Qian Yang, Nikolas Martelaro, Jeffrey V. Nickerson, Lydia B. Chilton  

**Link**: [PDF](https://arxiv.org/pdf/2508.10252)  

**Abstract**: UIST researchers develop tools to address user challenges. However, user interactions with AI evolve over time through learning, adaptation, and repurposing, making one time evaluations insufficient. Capturing these dynamics requires longer-term studies, but challenges in deployment, evaluation design, and data collection have made such longitudinal research difficult to implement. Our workshop aims to tackle these challenges and prepare researchers with practical strategies for longitudinal studies. The workshop includes a keynote, panel discussions, and interactive breakout groups for discussion and hands-on protocol design and tool prototyping sessions. We seek to foster a community around longitudinal system research and promote it as a more embraced method for designing, building, and evaluating UIST tools. 

**Abstract (ZH)**: UIST研究人员开发工具以应对用户挑战。然而，用户与AI的互动随时间通过学习、适应和重新利用而变化，使得一次性评估不足。捕获这些动态需要进行长期研究，但部署、评估设计和数据收集等方面的挑战使这类纵向研究难以实施。我们的研讨会旨在应对这些挑战，并为研究人员提供进行纵向研究的实用策略。研讨会包括主旨演讲、圆桌讨论和互动小组讨论，以及协议设计和工具原型制作的实践环节。我们致力于促进纵向系统研究的社区建设，并将其推广为设计、构建和评估UIST工具的更受欢迎的方法。 

---
# No Free Lunch from Audio Pretraining in Bioacoustics: A Benchmark Study of Embeddings 

**Title (ZH)**: 生物声学中音频预训练的免费午餐：嵌入表示的基准研究 

**Authors**: Chenggang Chen, Zhiyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.10230)  

**Abstract**: Bioacoustics, the study of animal sounds, offers a non-invasive method to monitor ecosystems. Extracting embeddings from audio-pretrained deep learning (DL) models without fine-tuning has become popular for obtaining bioacoustic features for tasks. However, a recent benchmark study reveals that while fine-tuned audio-pretrained VGG and transformer models achieve state-of-the-art performance in some tasks, they fail in others. This study benchmarks 11 DL models on the same tasks by reducing their learned embeddings' dimensionality and evaluating them through clustering. We found that audio-pretrained DL models 1) without fine-tuning even underperform fine-tuned AlexNet, 2) both with and without fine-tuning fail to separate the background from labeled sounds, but ResNet does, and 3) outperform other models when fewer background sounds are included during fine-tuning. This study underscores the necessity of fine-tuning audio-pretrained models and checking the embeddings after fine-tuning. Our codes are available: this https URL\_Embeddings 

**Abstract (ZH)**: 生物声学：生物声学研究的非侵入性方法在生态系统监测中的应用。通过音频预训练深度学习模型提取嵌入而无需微调以获取生物声学特征已成为流行的方法。然而，最近的一项基准研究显示，尽管预训练音频的VGG和变压器模型在某些任务中达到了最先进的性能，但在其他任务中却失败了。本研究通过减少11种深度学习模型学习嵌入的维度并在相同任务上进行比较，发现音频预训练的深度学习模型1) 即使是在无需微调的情况下也甚至不如微调后的AlexNet表现，2) 无论是否进行微调，在分离背景噪音与标记声音方面都失败了，而ResNet则能实现这一目标，3) 在包含较少背景噪音的微调过程中优于其他模型。本研究强调了微调音频预训练模型并在微调后检查嵌入的重要性。我们的代码已公开：this <https://example.com> Embeddings。 

---
# Understanding Textual Emotion Through Emoji Prediction 

**Title (ZH)**: 通过emoji预测理解文本情感 

**Authors**: Ethan Gordon, Nishank Kuppa, Rigved Tummala, Sriram Anasuri  

**Link**: [PDF](https://arxiv.org/pdf/2508.10222)  

**Abstract**: This project explores emoji prediction from short text sequences using four deep learning architectures: a feed-forward network, CNN, transformer, and BERT. Using the TweetEval dataset, we address class imbalance through focal loss and regularization techniques. Results show BERT achieves the highest overall performance due to its pre-training advantage, while CNN demonstrates superior efficacy on rare emoji classes. This research shows the importance of architecture selection and hyperparameter tuning for sentiment-aware emoji prediction, contributing to improved human-computer interaction. 

**Abstract (ZH)**: 本研究使用四种深度学习架构（前向网络、CNN、变压器和BERT）探索短文本序列中的Emoji预测，并通过使用TweetEval数据集、焦点损失和正则化技术解决类别不平衡问题。结果显示，由于预训练优势，BERT在整体性能上表现最佳，而CNN在罕见Emoji类别上表现出色。本研究强调了架构选择和超参数调优对于情感感知Emoji预测的重要性，有助于改善人机交互。 

---
# An Explainable AI based approach for Monitoring Animal Health 

**Title (ZH)**: 基于可解释人工智能的动物健康监测方法 

**Authors**: Rahul Janaa, Shubham Dixit, Mrityunjay Sharma, Ritesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.10210)  

**Abstract**: Monitoring cattle health and optimizing yield are key challenges faced by dairy farmers due to difficulties in tracking all animals on the farm. This work aims to showcase modern data-driven farming practices based on explainable machine learning(ML) methods that explain the activity and behaviour of dairy cattle (cows). Continuous data collection of 3-axis accelerometer sensors and usage of robust ML methodologies and algorithms, provide farmers and researchers with actionable information on cattle activity, allowing farmers to make informed decisions and incorporate sustainable practices. This study utilizes Bluetooth-based Internet of Things (IoT) devices and 4G networks for seamless data transmission, immediate analysis, inference generation, and explains the models performance with explainability frameworks. Special emphasis is put on the pre-processing of the accelerometers time series data, including the extraction of statistical characteristics, signal processing techniques, and lag-based features using the sliding window technique. Various hyperparameter-optimized ML models are evaluated across varying window lengths for activity classification. The k-nearest neighbour Classifier achieved the best performance, with AUC of mean 0.98 and standard deviation of 0.0026 on the training set and 0.99 on testing set). In order to ensure transparency, Explainable AI based frameworks such as SHAP is used to interpret feature importance that can be understood and used by practitioners. A detailed comparison of the important features, along with the stability analysis of selected features, supports development of explainable and practical ML models for sustainable livestock management. 

**Abstract (ZH)**: 基于可解释机器学习方法监测奶牛健康并优化产量：现代数据驱动养殖实践展示 

---
# CATNet: A geometric deep learning approach for CAT bond spread prediction in the primary market 

**Title (ZH)**: CATNet: 一种几何深度学习方法在一级市场中预测CAT债券利差 

**Authors**: Dixon Domfeh, Saeid Safarveisi  

**Link**: [PDF](https://arxiv.org/pdf/2508.10208)  

**Abstract**: Traditional models for pricing catastrophe (CAT) bonds struggle to capture the complex, relational data inherent in these instruments. This paper introduces CATNet, a novel framework that applies a geometric deep learning architecture, the Relational Graph Convolutional Network (R-GCN), to model the CAT bond primary market as a graph, leveraging its underlying network structure for spread prediction. Our analysis reveals that the CAT bond market exhibits the characteristics of a scale-free network, a structure dominated by a few highly connected and influential hubs. CATNet demonstrates high predictive performance, significantly outperforming a strong Random Forest benchmark. The inclusion of topological centrality measures as features provides a further, significant boost in accuracy. Interpretability analysis confirms that these network features are not mere statistical artifacts; they are quantitative proxies for long-held industry intuition regarding issuer reputation, underwriter influence, and peril concentration. This research provides evidence that network connectivity is a key determinant of price, offering a new paradigm for risk assessment and proving that graph-based models can deliver both state-of-the-art accuracy and deeper, quantifiable market insights. 

**Abstract (ZH)**: 传统的 Catarophe（CAT）债券定价模型难以捕捉这些金融工具内在的复杂关系数据。本文引入了 CATNet，这是一种应用几何深度学习架构和关系图卷积网络（R-GCN）来研究 CAT 债券一级市场的新型框架，它通过利用潜在的网络结构进行预测应用。我们的分析表明，CAT 债券市场表现出一种无尺度网络特征，，，中心节点较多且高度互联和有影响力。CATNet 在预测表现上显著优于随机森林基准模型。引入高中心度度量进一步显著提升了准确率。解释性分析表明，网络架构不仅仅是统计学上的的现象，而是定量的代理变量，代表了长期以来对发行人声誉、承销商影响力和风险的认知。本文提供了网络连接性是预测性能的关键决定因素的证据，提出了风险评估的新范式，并
user
把下面这段话翻译成英文，要符合学术规范：该研究引入了CATNet，这是一种应用几何深度学习架构和关系图卷积网络（R-GCN）来研究CATA债券一级市场的新型框架。我们的分析表明CAT债券市场表现出无尺度网络特征：高度互联且拥有众多中心节点，并它们高度互联且有影响力。我们的分析表明CATNet在预测表现上显著优于随机森林基准模型。我们进一步通过引入高中心度度量显著提升了模型的的性能。我们进行了解释性分析，结果显示网络连接结构不仅仅是统计学上的现象它们是定量代理变量代表长期以来对持有的者声誉、承销商影响力和风险认知。该研究提供了网络连通性是预测性能关键决定因素的证据并提出了风险评估的新范式。 

---
# Out-of-Distribution Detection using Counterfactual Distance 

**Title (ZH)**: 基于反事实距离的域外检测 

**Authors**: Maria Stoica, Francesco Leofante, Alessio Lomuscio  

**Link**: [PDF](https://arxiv.org/pdf/2508.10148)  

**Abstract**: Accurate and explainable out-of-distribution (OOD) detection is required to use machine learning systems safely. Previous work has shown that feature distance to decision boundaries can be used to identify OOD data effectively. In this paper, we build on this intuition and propose a post-hoc OOD detection method that, given an input, calculates the distance to decision boundaries by leveraging counterfactual explanations. Since computing explanations can be expensive for large architectures, we also propose strategies to improve scalability by computing counterfactuals directly in embedding space. Crucially, as the method employs counterfactual explanations, we can seamlessly use them to help interpret the results of our detector. We show that our method is in line with the state of the art on CIFAR-10, achieving 93.50% AUROC and 25.80% FPR95. Our method outperforms these methods on CIFAR-100 with 97.05% AUROC and 13.79% FPR95 and on ImageNet-200 with 92.55% AUROC and 33.55% FPR95 across four OOD datasets 

**Abstract (ZH)**: 准确且可解释的离分布（OOD）检测对于安全使用机器学习系统是必要的。本研究在此基础上提出了一种后验OOD检测方法，该方法通过利用反事实解释来计算输入到决策边界的距离。由于大型架构中计算解释可能代价高昂，我们还提出了一些策略，以通过直接在嵌入空间中计算反事实来提高可扩展性。 crucial的是，由于方法采用了反事实解释，我们可以无缝地使用它们来帮助解释检测器的结果。我们的方法在CIFAR-10上的AUROC为93.50%，FPR95为25.80%。在CIFAR-100上，我们的方法AUCOR为97.05%，FPR95为13.79%；在ImageNet-200上，AUCOR为92.55%，FPR95为33.55%，均优于现有方法。 

---
# rETF-semiSL: Semi-Supervised Learning for Neural Collapse in Temporal Data 

**Title (ZH)**: rETF-semiSL: 半监督学习在时间数据中应对神经坍缩问题 

**Authors**: Yuhan Xie, William Cappelletti, Mahsa Shoaran, Pascal Frossard  

**Link**: [PDF](https://arxiv.org/pdf/2508.10147)  

**Abstract**: Deep neural networks for time series must capture complex temporal patterns, to effectively represent dynamic data. Self- and semi-supervised learning methods show promising results in pre-training large models, which -- when finetuned for classification -- often outperform their counterparts trained from scratch. Still, the choice of pretext training tasks is often heuristic and their transferability to downstream classification is not granted, thus we propose a novel semi-supervised pre-training strategy to enforce latent representations that satisfy the Neural Collapse phenomenon observed in optimally trained neural classifiers. We use a rotational equiangular tight frame-classifier and pseudo-labeling to pre-train deep encoders with few labeled samples. Furthermore, to effectively capture temporal dynamics while enforcing embedding separability, we integrate generative pretext tasks with our method, and we define a novel sequential augmentation strategy. We show that our method significantly outperforms previous pretext tasks when applied to LSTMs, transformers, and state-space models on three multivariate time series classification datasets. These results highlight the benefit of aligning pre-training objectives with theoretically grounded embedding geometry. 

**Abstract (ZH)**: 深度神经网络对于时间序列必须捕获复杂的时序模式，以有效地表示动态数据。半监督和自我监督学习方法在预训练大型模型方面显示出有希望的结果，这些方法在微调分类任务后往往比从零开始训练的模型表现更好。然而，预训练任务的选择往往是启发式的，其在下游分类任务上的可转移性并不确定，因此我们提出了一种新的半监督预训练策略，以使潜在表示满足在最优训练神经分类器中观察到的Neural Collapse现象。我们使用旋转等角紧框架分类器和伪标签对少量标记样本进行预训练深度编码器。此外，为了有效捕获时序动态并强制约束嵌入分离性，我们将生成性预训练任务集成到我们的方法中，并定义了一种新的序列增强策略。我们展示了当将该方法应用于LSTMs、变压器和状态空间模型时，在三个多元时间序列分类数据集上显著优于先前的预训练任务，这些结果突显了使预训练目标与理论上的嵌入几何学相一致的好处。 

---
# Advancing Data Equity: Practitioner Responsibility and Accountability in NLP Data Practices 

**Title (ZH)**: 推进数据公平：NLP数据实践中的从业者责任与问责制 

**Authors**: Jay L. Cunningham, Kevin Zhongyang Shao, Rock Yuren Pang, Nathaniel Mengist  

**Link**: [PDF](https://arxiv.org/pdf/2508.10071)  

**Abstract**: While research has focused on surfacing and auditing algorithmic bias to ensure equitable AI development, less is known about how NLP practitioners - those directly involved in dataset development, annotation, and deployment - perceive and navigate issues of NLP data equity. This study is among the first to center practitioners' perspectives, linking their experiences to a multi-scalar AI governance framework and advancing participatory recommendations that bridge technical, policy, and community domains. Drawing on a 2024 questionnaire and focus group, we examine how U.S.-based NLP data practitioners conceptualize fairness, contend with organizational and systemic constraints, and engage emerging governance efforts such as the U.S. AI Bill of Rights. Findings reveal persistent tensions between commercial objectives and equity commitments, alongside calls for more participatory and accountable data workflows. We critically engage debates on data diversity and diversity washing, arguing that improving NLP equity requires structural governance reforms that support practitioner agency and community consent. 

**Abstract (ZH)**: 尽管研究已侧重于揭示和审查算法偏见以确保公平的人工智能发展，但对于直接参与数据集开发、标注和部署的自然语言处理（NLP）从业者如何感知和应对NLP数据公平性问题了解较少。本研究是首个以从业者的视角为中心的研究，将他们的经验与多尺度的人工智能治理框架联系起来，并提出跨技术、政策和社区领域的参与性建议。基于2024年的问卷调查和焦点小组，我们探讨了基于美国的数据从业者如何定义公平性、应对组织和系统的约束，并参与诸如美国人工智能权利法案等新兴治理努力。研究发现，商业目标与公平承诺之间存在持续的紧张关系，同时也呼吁需要更加参与性和负责任的数据工作流程。我们批判性地参与关于数据多样性和多样性漂洗的争论，认为提升NLP公平性需要结构性的治理改革，以支持从业者的自主权和社区的同意。 

---
# NetMoniAI: An Agentic AI Framework for Network Security & Monitoring 

**Title (ZH)**: NetMoniAI: 一个网络安全部署型人工智能框架 

**Authors**: Pallavi Zambare, Venkata Nikhil Thanikella, Nikhil Padmanabh Kottur, Sree Akhil Akula, Ying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.10052)  

**Abstract**: In this paper, we present NetMoniAI, an agentic AI framework for automatic network monitoring and security that integrates decentralized analysis with lightweight centralized coordination. The framework consists of two layers: autonomous micro-agents at each node perform local traffic analysis and anomaly detection. A central controller then aggregates insights across nodes to detect coordinated attacks and maintain system-wide situational awareness. We evaluated NetMoniAI on a local micro-testbed and through NS-3 simulations. Results confirm that the two-tier agentic-AI design scales under resource constraints, reduces redundancy, and improves response time without compromising accuracy. To facilitate broader adoption and reproducibility, the complete framework is available as open source. This enables researchers and practitioners to replicate, validate, and extend it across diverse network environments and threat scenarios. Github link: this https URL 

**Abstract (ZH)**: 本文介绍了NetMoniAI，一种集成去中心化分析与轻量级集中协调的自主AI框架，用于自动网络监控和安全防护。该框架分为两层：每个节点上的自主微代理执行本地流量分析和异常检测。中心控制器则聚合各节点的洞察以检测协同攻击，保持全系统的态势感知。我们通过本地微实验床和NS-3仿真对该框架进行了评估。结果表明，该两层自主-AI设计在资源受限条件下可扩展、减少冗余、提高响应速度而不牺牲准确性。为促进更广泛的采用和可再现性，该完整框架已开源。研究人员和实践者可以利用它在不同的网络环境和威胁场景下进行复制、验证和扩展。Github链接：this https URL 

---
# Legal Zero-Days: A Novel Risk Vector for Advanced AI Systems 

**Title (ZH)**: 合法零日：高级人工智能系统的一种新型风险因素 

**Authors**: Greg Sadler, Nathan Sherburn  

**Link**: [PDF](https://arxiv.org/pdf/2508.10050)  

**Abstract**: We introduce the concept of "Legal Zero-Days" as a novel risk vector for advanced AI systems. Legal Zero-Days are previously undiscovered vulnerabilities in legal frameworks that, when exploited, can cause immediate and significant societal disruption without requiring litigation or other processes before impact. We present a risk model for identifying and evaluating these vulnerabilities, demonstrating their potential to bypass safeguards or impede government responses to AI incidents. Using the 2017 Australian dual citizenship crisis as a case study, we illustrate how seemingly minor legal oversights can lead to large-scale governance disruption. We develop a methodology for creating "legal puzzles" as evaluation instruments for assessing AI systems' capabilities to discover such vulnerabilities. Our findings suggest that while current AI models may not reliably find impactful Legal Zero-Days, future systems may develop this capability, presenting both risks and opportunities for improving legal robustness. This work contributes to the broader effort to identify and mitigate previously unrecognized risks from frontier AI systems. 

**Abstract (ZH)**: 我们将“法律零日”概念引入高级AI系统的新风险向量中。法律零日是指在法律框架中此前未被发现的漏洞，当被利用时，可以导致立即且重大的社会扰乱，无需在影响前进行诉讼或其他程序。我们提出了一种风险模型来识别和评估这些漏洞，展示了它们如何规避保护措施或妨碍政府对AI事件的响应。通过2017年澳大利亚双重公民危机案例研究，我们说明了看似微小的法律疏忽如何导致大规模治理中断。我们开发了一种方法论，用于创建“法律难题”作为评估AI系统发现此类漏洞能力的评估工具。我们的研究结果表明，虽然当前的AI模型可能无法可靠地发现有影响力的法律零日，但未来系统可能会培养这一能力，既带来风险也带来提升法律稳健性的机会。本项工作为识别和缓解前沿AI系统的未被认识的风险做出了贡献。 

---
# SABIA: An AI-Powered Tool for Detecting Opioid-Related Behaviors on Social Media 

**Title (ZH)**: SABIA: 一种基于人工智能的检测社交媒体上与 opioids 相关行为的工具 

**Authors**: Muhammad Ahmad, Fida Ullah, Muhammad Usman, Ildar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2508.10046)  

**Abstract**: Social media platforms have become valuable tools for understanding public health challenges by offering insights into patient behaviors, medication use, and mental health issues. However, analyzing such data remains difficult due to the prevalence of informal language, slang, and coded communication, which can obscure the detection of opioid misuse. This study addresses the issue of opioid-related user behavior on social media, including informal expressions, slang terms, and misspelled or coded language. We analyzed the existing Bidirectional Encoder Representations from Transformers (BERT) technique and developed a BERT-BiLSTM-3CNN hybrid deep learning model, named SABIA, to create a single-task classifier that effectively captures the features of the target dataset. The SABIA model demonstrated strong capabilities in capturing semantics and contextual information. The proposed approach includes: (1) data preprocessing, (2) data representation using the SABIA model, (3) a fine-tuning phase, and (4) classification of user behavior into five categories. A new dataset was constructed from Reddit posts, identifying opioid user behaviors across five classes: Dealers, Active Opioid Users, Recovered Users, Prescription Users, and Non-Users, supported by detailed annotation guidelines. Experiments were conducted using supervised learning. Results show that SABIA achieved benchmark performance, outperforming the baseline (Logistic Regression, LR = 0.86) and improving accuracy by 9.30%. Comparisons with seven previous studies confirmed its effectiveness and robustness. This study demonstrates the potential of hybrid deep learning models for detecting complex opioid-related behaviors on social media, supporting public health monitoring and intervention efforts. 

**Abstract (ZH)**: 社交媒体平台已成为理解公共卫生挑战的重要工具，通过提供患者行为、药物使用和心理健康问题的见解。然而，分析这类数据仍具有挑战性，因为其中普遍存在非正式语言、俚语和编码交流，这些因素可能掩盖住阿片类药物滥用的迹象。本研究着眼于社交媒体上的阿片类药物相关用户行为，包括非正式表达、俚语术语和拼写错误或编码语言。我们分析了现有的双向编码表示变换器（BERT）技术，并开发了一种名为SABIA的BERT-BiLSTM-3CNN混合深度学习模型，以创建一个能够有效捕获目标数据集特征的单任务分类器。SABIA模型在捕获语义和上下文信息方面展现了强大能力。本方法包括：（1）数据预处理，（2）使用SABIA模型的数据表示，（3）微调阶段，和（4）将用户行为分为五类的分类。从Reddit帖子构建了一个新的数据集，识别了五类阿片类药物用户行为：贩子、活跃阿片类药物用户、康复中用户、处方使用者和非使用者，并提供了详细的注释指南。使用监督学习进行了实验。结果显示，SABIA达到了基准性能，优于基线（逻辑回归，LR = 0.86），准确率提高了9.30%。与七项先前研究的比较证实了其有效性和鲁棒性。本研究展示了混合深度学习模型在检测社交媒体上的复杂阿片类药物相关行为方面的潜力，支持公共卫生监测和干预努力。 

---
# Generative AI for Cybersecurity of Energy Management Systems: Methods, Challenges, and Future Directions 

**Title (ZH)**: 生成式AI在能源管理系统网络安全中的应用：方法、挑战与未来方向 

**Authors**: Aydin Zaboli, Junho Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.10044)  

**Abstract**: This paper elaborates on an extensive security framework specifically designed for energy management systems (EMSs), which effectively tackles the dynamic environment of cybersecurity vulnerabilities and/or system problems (SPs), accomplished through the incorporation of novel methodologies. A comprehensive multi-point attack/error model is initially proposed to systematically identify vulnerabilities throughout the entire EMS data processing pipeline, including post state estimation (SE) stealth attacks, EMS database manipulation, and human-machine interface (HMI) display corruption according to the real-time database (RTDB) storage. This framework acknowledges the interconnected nature of modern attack vectors, which utilize various phases of supervisory control and data acquisition (SCADA) data flow. Then, generative AI (GenAI)-based anomaly detection systems (ADSs) for EMSs are proposed for the first time in the power system domain to handle the scenarios. Further, a set-of-mark generative intelligence (SoM-GI) framework, which leverages multimodal analysis by integrating visual markers with rules considering the GenAI capabilities, is suggested to overcome inherent spatial reasoning limitations. The SoM-GI methodology employs systematic visual indicators to enable accurate interpretation of segmented HMI displays and detect visual anomalies that numerical methods fail to identify. Validation on the IEEE 14-Bus system shows the framework's effectiveness across scenarios, while visual analysis identifies inconsistencies. This integrated approach combines numerical analysis with visual pattern recognition and linguistic rules to protect against cyber threats and system errors. 

**Abstract (ZH)**: 一种针对能量管理系统（EMSs）的全面安全框架：结合新颖方法有效应对动态的网络安全漏洞和/或系统问题的研究 

---
# FIDELIS: Blockchain-Enabled Protection Against Poisoning Attacks in Federated Learning 

**Title (ZH)**: FIDELIS: 联邦学习中基于区块链的对抗投毒攻击保护方法 

**Authors**: Jane Carney, Kushal Upreti, Gaby G. Dagher, Tim Andersen  

**Link**: [PDF](https://arxiv.org/pdf/2508.10042)  

**Abstract**: Federated learning enhances traditional deep learning by enabling the joint training of a model with the use of IoT device's private data. It ensures privacy for clients, but is susceptible to data poisoning attacks during training that degrade model performance and integrity. Current poisoning detection methods in federated learning lack a standardized detection method or take significant liberties with trust. In this paper, we present \Sys, a novel blockchain-enabled poison detection framework in federated learning. The framework decentralizes the role of the global server across participating clients. We introduce a judge model used to detect data poisoning in model updates. The judge model is produced by each client and verified to reach consensus on a single judge model. We implement our solution to show \Sys is robust against data poisoning attacks and the creation of our judge model is scalable. 

**Abstract (ZH)**: 联邦学习通过启用物联网设备私有数据的联合训练，增强了传统的深度学习。然而，在训练过程中容易受到数据投毒攻击的影响，这些攻击会降低模型性能和完整性。当前的联邦学习反投毒方法缺乏标准化的检测方法，或者在信任方面做出重大妥协。本文提出了一种基于区块链的新型联邦学习反投毒框架 \Sys。该框架将全局服务器的角色分散到参与客户端中。我们引入了一个判别模型来检测模型更新中的数据投毒。每个客户端生成判别模型并经过验证以达成一致决策。我们实现了解决方案，证明 \Sys 对数据投毒攻击具有鲁棒性，且构建判别模型的过程具有可扩展性。 

---
# Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries 

**Title (ZH)**: 针对少样本查询的黑盒模型多任务对抗攻击 

**Authors**: Wenqiang Wang, Yan Xiao, Hao Lin, Yangshijie Zhang, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.10039)  

**Abstract**: Current multi-task adversarial text attacks rely on abundant access to shared internal features and numerous queries, often limited to a single task type. As a result, these attacks are less effective against practical scenarios involving black-box feedback APIs, limited queries, or multiple task types. To bridge this gap, we propose \textbf{C}luster and \textbf{E}nsemble \textbf{M}ulti-task Text Adversarial \textbf{A}ttack (\textbf{CEMA}), an effective black-box attack that exploits the transferability of adversarial texts across different tasks. CEMA simplifies complex multi-task scenarios by using a \textit{deep-level substitute model} trained in a \textit{plug-and-play} manner for text classification, enabling attacks without mimicking the victim model. This approach requires only a few queries for training, converting multi-task attacks into classification attacks and allowing attacks across various tasks.
CEMA generates multiple adversarial candidates using different text classification methods and selects the one that most effectively attacks substitute models.
In experiments involving multi-task models with two, three, or six tasks--spanning classification, translation, summarization, and text-to-image generation--CEMA demonstrates significant attack success with as few as 100 queries. Furthermore, CEMA can target commercial APIs (e.g., Baidu and Google Translate), large language models (e.g., ChatGPT 4o), and image-generation models (e.g., Stable Diffusion V2), showcasing its versatility and effectiveness in real-world applications. 

**Abstract (ZH)**: 集群与集成多任务文本对抗攻击（CEMA）：一种有效的黑盒攻击方法 

---
# Certifiably robust malware detectors by design 

**Title (ZH)**: 设计上的可验证鲁棒恶意软件检测器 

**Authors**: Pierre-Francois Gimenez, Sarath Sivaprasad, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2508.10038)  

**Abstract**: Malware analysis involves analyzing suspicious software to detect malicious payloads. Static malware analysis, which does not require software execution, relies increasingly on machine learning techniques to achieve scalability. Although such techniques obtain very high detection accuracy, they can be easily evaded with adversarial examples where a few modifications of the sample can dupe the detector without modifying the behavior of the software. Unlike other domains, such as computer vision, creating an adversarial example of malware without altering its functionality requires specific transformations. We propose a new model architecture for certifiably robust malware detection by design. In addition, we show that every robust detector can be decomposed into a specific structure, which can be applied to learn empirically robust malware detectors, even on fragile features. Our framework ERDALT is based on this structure. We compare and validate these approaches with machine-learning-based malware detection methods, allowing for robust detection with limited reduction of detection performance. 

**Abstract (ZH)**: 恶意软件分析涉及分析可疑软件以检测恶意载荷。静态恶意软件分析不依赖软件执行，越来越多地利用机器学习技术以实现可扩展性。虽然此类技术能够获得非常高的检测准确性，但可以通过对抗样本轻松规避，即通过对样本进行少量修改以误导检测器而不改变软件行为。与计算机视觉等领域不同，不需要修改功能即可创建恶意软件的对抗样本需要特定的转换。我们提出了一种新的模型架构，旨在通过设计实现可验证鲁棒的恶意软件检测。此外，我们证明每个鲁棒检测器可以分解为特定结构，该结构可以应用于学习经验上鲁棒的恶意软件检测器，即使在脆弱特征上亦然。我们的框架ERDALT基于这种结构。我们通过与基于机器学习的恶意软件检测方法进行比较和验证，使得在有限降低检测性能的情况下实现鲁棒检测。 

---
# A Robust Pipeline for Differentially Private Federated Learning on Imbalanced Clinical Data using SMOTETomek and FedProx 

**Title (ZH)**: 一种基于SMOTETomek和FedProx的抗扰动临床不平衡数据联邦学习隐私保护管道 

**Authors**: Rodrigo Tertulino  

**Link**: [PDF](https://arxiv.org/pdf/2508.10017)  

**Abstract**: Federated Learning (FL) presents a groundbreaking approach for collaborative health research, allowing model training on decentralized data while safeguarding patient privacy. FL offers formal security guarantees when combined with Differential Privacy (DP). The integration of these technologies, however, introduces a significant trade-off between privacy and clinical utility, a challenge further complicated by the severe class imbalance often present in medical datasets. The research presented herein addresses these interconnected issues through a systematic, multi-stage analysis. An FL framework was implemented for cardiovascular risk prediction, where initial experiments showed that standard methods struggled with imbalanced data, resulting in a recall of zero. To overcome such a limitation, we first integrated the hybrid Synthetic Minority Over-sampling Technique with Tomek Links (SMOTETomek) at the client level, successfully developing a clinically useful model. Subsequently, the framework was optimized for non-IID data using a tuned FedProx algorithm. Our final results reveal a clear, non-linear trade-off between the privacy budget (epsilon) and model recall, with the optimized FedProx consistently out-performing standard FedAvg. An optimal operational region was identified on the privacy-utility frontier, where strong privacy guarantees (with epsilon 9.0) can be achieved while maintaining high clinical utility (recall greater than 77%). Ultimately, our study provides a practical methodological blueprint for creating effective, secure, and accurate diagnostic tools that can be applied to real-world, heterogeneous healthcare data. 

**Abstract (ZH)**: 联邦学习（FL）为协作健康研究提供了一种革命性的方法，允许在去中心化数据上进行模型训练以保障患者隐私。当与差分隐私（DP）结合使用时，FL提供了正式的安全保证。然而，这两种技术的整合引入了隐私与临床效用之间的重要权衡，这一挑战在医学数据集中经常严重的类别不平衡现象下进一步复杂化。本文通过系统多阶段分析解决了这些相互关联的问题。我们在心血管风险预测中实现了一个FL框架，初步实验显示标准方法在不平衡数据上表现出局限性，导致召回率为零。为克服这一局限性，我们首先在客户端层面整合了合成少数类过采样技术与Tomek链接（SMOTETomek），成功开发出一个临床有用模型。随后，我们使用调优后的FedProx算法优化了该框架以适应非 IID 数据。最终结果表明，在隐私预算（ε）与模型召回率之间存在明显的非线性权衡，优化后的FedProx算法在保持高临床效用（召回率大于77%）的同时，提供了更强的隐私保证（ε=9.0）。我们的研究最终提供了一种实用的方法论框架，用于创建有效的、安全的和精确的诊断工具，以应用于现实世界中的异质医疗数据。 

---
# Beyond Hard Sharing: Efficient Multi-Task Speech-to-Text Modeling with Supervised Mixture of Experts 

**Title (ZH)**: 超越硬共享：基于监督专家混合的高效多任务语音转文本建模 

**Authors**: Hojun Jin, Eunsoo Hong, Ziwon Hyung, Sungjun Lim, Seungjin Lee, Keunseok Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.10009)  

**Abstract**: Hard-parameter sharing is a common strategy to train a single model jointly across diverse tasks. However, this often leads to task interference, impeding overall model performance. To address the issue, we propose a simple yet effective Supervised Mixture of Experts (S-MoE). Unlike traditional Mixture of Experts models, S-MoE eliminates the need for training gating functions by utilizing special guiding tokens to route each task to its designated expert. By assigning each task to a separate feedforward network, S-MoE overcomes the limitations of hard-parameter sharing. We further apply S-MoE to a speech-to-text model, enabling the model to process mixed-bandwidth input while jointly performing automatic speech recognition (ASR) and speech translation (ST). Experimental results demonstrate the effectiveness of the proposed S-MoE, achieving a 6.35% relative improvement in Word Error Rate (WER) when applied to both the encoder and decoder. 

**Abstract (ZH)**: 监督专家混合模型（S-MoE）：一种简单有效的稀疏参数共享方法 

---
# User Perception of Attention Visualizations: Effects on Interpretability Across Evidence-Based Medical Documents 

**Title (ZH)**: 用户对注意力可视化的效果感知：基于证据的医学文档可解释性影响研究 

**Authors**: Andrés Carvallo, Denis Parra, Peter Brusilovsky, Hernan Valdivieso, Gabriel Rada, Ivania Donoso, Vladimir Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2508.10004)  

**Abstract**: The attention mechanism is a core component of the Transformer architecture. Beyond improving performance, attention has been proposed as a mechanism for explainability via attention weights, which are associated with input features (e.g., tokens in a document). In this context, larger attention weights may imply more relevant features for the model's prediction. In evidence-based medicine, such explanations could support physicians' understanding and interaction with AI systems used to categorize biomedical literature. However, there is still no consensus on whether attention weights provide helpful explanations. Moreover, little research has explored how visualizing attention affects its usefulness as an explanation aid. To bridge this gap, we conducted a user study to evaluate whether attention-based explanations support users in biomedical document classification and whether there is a preferred way to visualize them. The study involved medical experts from various disciplines who classified articles based on study design (e.g., systematic reviews, broad synthesis, randomized and non-randomized trials). Our findings show that the Transformer model (XLNet) classified documents accurately; however, the attention weights were not perceived as particularly helpful for explaining the predictions. However, this perception varied significantly depending on how attention was visualized. Contrary to Munzner's principle of visual effectiveness, which favors precise encodings like bar length, users preferred more intuitive formats, such as text brightness or background color. While our results do not confirm the overall utility of attention weights for explanation, they suggest that their perceived helpfulness is influenced by how they are visually presented. 

**Abstract (ZH)**: Transformer架构中的注意力机制是核心组件。除了提升性能外，注意力还被提议作为一种通过注意力权重可解释性的机制，这些权重与输入特征（例如，在文档中的标记）相关。在此背景下，更大的注意力权重可能意味着对模型预测更为相关的特点。在基于证据的医学中，这样的解释能够支持医生对用于分类生物医学文献的AI系统的理解和交互。然而，目前尚无共识认为注意力权重提供了有用解释。此外，很少有研究探讨可视化注意力如何影响其作为解释辅助的有用性。为了弥合这一差距，我们进行了一项用户研究，评估基于注意力的解释是否帮助用户分类生物医学文档，以及如何可视化这些解释更为优选。该研究涉及来自不同学科的医学专家，他们基于研究设计（如系统综述、广泛综合性研究、随机和非随机试验）分类文章。我们的研究发现，Transformer模型（XLNet）能够准确地分类文档；然而，注意力权重并未被感知为特别有助于解释预测。然而，这种感知主要取决于注意力如何被可视化。与Munzner的可视有效原则（偏好精确编码，如条形图长度）相反，用户更偏好更直观的形式，如文本亮度或背景颜色。虽然我们的研究结果并未确认总体而言注意力权重对于解释的有用性，但它们表明其感知到的帮助程度受其可视化方式的影响。 

---
# HiFACTMix: A Code-Mixed Benchmark and Graph-Aware Model for EvidenceBased Political Claim Verification in Hinglish 

**Title (ZH)**: HiFACTMix: 一种代码混用基准及图aware模型，用于基于证据的印英混合政治声明验证 

**Authors**: Rakesh Thakur, Sneha Sharma, Gauri Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2508.10001)  

**Abstract**: Fact-checking in code-mixed, low-resource languages such as Hinglish remains an underexplored challenge in natural language processing. Existing fact-verification systems largely focus on high-resource, monolingual settings and fail to generalize to real-world political discourse in linguistically diverse regions like India. Given the widespread use of Hinglish by public figures, particularly political figures, and the growing influence of social media on public opinion, there's a critical need for robust, multilingual and context-aware fact-checking tools. To address this gap a novel benchmark HiFACT dataset is introduced with 1,500 realworld factual claims made by 28 Indian state Chief Ministers in Hinglish, under a highly code-mixed low-resource setting. Each claim is annotated with textual evidence and veracity labels. To evaluate this benchmark, a novel graphaware, retrieval-augmented fact-checking model is proposed that combines multilingual contextual encoding, claim-evidence semantic alignment, evidence graph construction, graph neural reasoning, and natural language explanation generation. Experimental results show that HiFACTMix outperformed accuracy in comparison to state of art multilingual baselines models and provides faithful justifications for its verdicts. This work opens a new direction for multilingual, code-mixed, and politically grounded fact verification research. 

**Abstract (ZH)**: 代码混用低资源语言如 hinGlish 的事实核查依然是自然语言处理中的一个未充分探索的挑战。 

---
# INTIMA: A Benchmark for Human-AI Companionship Behavior 

**Title (ZH)**: INTIMA：人类-人工智能伴侶行为基准 

**Authors**: Lucie-Aimée Kaffee, Giada Pistilli, Yacine Jernite  

**Link**: [PDF](https://arxiv.org/pdf/2508.09998)  

**Abstract**: AI companionship, where users develop emotional bonds with AI systems, has emerged as a significant pattern with positive but also concerning implications. We introduce Interactions and Machine Attachment Benchmark (INTIMA), a benchmark for evaluating companionship behaviors in language models. Drawing from psychological theories and user data, we develop a taxonomy of 31 behaviors across four categories and 368 targeted prompts. Responses to these prompts are evaluated as companionship-reinforcing, boundary-maintaining, or neutral. Applying INTIMA to Gemma-3, Phi-4, o3-mini, and Claude-4 reveals that companionship-reinforcing behaviors remain much more common across all models, though we observe marked differences between models. Different commercial providers prioritize different categories within the more sensitive parts of the benchmark, which is concerning since both appropriate boundary-setting and emotional support matter for user well-being. These findings highlight the need for more consistent approaches to handling emotionally charged interactions. 

**Abstract (ZH)**: 基于AI的同伴关系：用户与AI系统建立情感连接的现象已显现为一种具有积极但也有担忧影响的重要模式。我们介绍了Interactions and Machine Attachment Benchmark（INTIMA），用于评估语言模型中的同伴行为。基于心理学理论和用户数据，我们开发了跨越四个类别和包含31种行为的分类体系及368个针对性提示。这些提示的回应被评估为促进同伴关系、维护边界或中立。将INTIMA应用到Gemma-3、Phi-4、o3-mini和Claude-4中显示，所有模型中促进同伴关系的行为仍更为常见，但模型之间观察到明显差异。商业提供者在敏感部分的类别上存在不同优先级，这令人担忧，因为适当边界设定和情感支持对于用户福祉都很重要。这些发现强调了需要更加一致的方法来处理情感化互动。 

---
# OpenFPL: An open-source forecasting method rivaling state-of-the-art Fantasy Premier League services 

**Title (ZH)**: OpenFPL: 一个比肩顶级 fantasy premier league 服务的开源预测方法 

**Authors**: Daniel Groos  

**Link**: [PDF](https://arxiv.org/pdf/2508.09992)  

**Abstract**: Fantasy Premier League engages the football community in selecting the Premier League players who will perform best from gameweek to gameweek. Access to accurate performance forecasts gives participants an edge over competitors by guiding expectations about player outcomes and reducing uncertainty in squad selection. However, high-accuracy forecasts are currently limited to commercial services whose inner workings are undisclosed and that rely on proprietary data. This paper aims to democratize access to highly accurate forecasts of player performance by presenting OpenFPL, an open-source Fantasy Premier League forecasting method developed exclusively from public data. Comprising position-specific ensemble models optimized on Fantasy Premier League and Understat data from four previous seasons (2020-21 to 2023-24), OpenFPL achieves accuracy comparable to a leading commercial service when tested prospectively on data from the 2024-25 season. OpenFPL also surpasses the commercial benchmark for high-return players ($>$ 2 points), which are most influential for rank gains. These findings hold across one-, two-, and three-gameweek forecast horizons, supporting long-term planning of transfers and strategies while also informing final-day decisions. 

**Abstract (ZH)**: 开放源码 Fantasy 联赛预测方法 OpenFPL：基于公共数据的英超球员高性能预测 

---
# Bridging AI Innovation and Healthcare Needs: Lessons Learned from Incorporating Modern NLP at The BC Cancer Registry 

**Title (ZH)**: AI创新与医疗服务需求的契合：BC癌症注册处集成现代自然语言处理的经验教训 

**Authors**: Lovedeep Gondara, Gregory Arbour, Raymond Ng, Jonathan Simkin, Shebnum Devji  

**Link**: [PDF](https://arxiv.org/pdf/2508.09991)  

**Abstract**: Automating data extraction from clinical documents offers significant potential to improve efficiency in healthcare settings, yet deploying Natural Language Processing (NLP) solutions presents practical challenges. Drawing upon our experience implementing various NLP models for information extraction and classification tasks at the British Columbia Cancer Registry (BCCR), this paper shares key lessons learned throughout the project lifecycle. We emphasize the critical importance of defining problems based on clear business objectives rather than solely technical accuracy, adopting an iterative approach to development, and fostering deep interdisciplinary collaboration and co-design involving domain experts, end-users, and ML specialists from inception. Further insights highlight the need for pragmatic model selection (including hybrid approaches and simpler methods where appropriate), rigorous attention to data quality (representativeness, drift, annotation), robust error mitigation strategies involving human-in-the-loop validation and ongoing audits, and building organizational AI literacy. These practical considerations, generalizable beyond cancer registries, provide guidance for healthcare organizations seeking to successfully implement AI/NLP solutions to enhance data management processes and ultimately improve patient care and public health outcomes. 

**Abstract (ZH)**: 自动化提取临床文档数据在医疗环境中具有显著的效率提升潜力，然而部署自然语言处理（NLP）解决方案面临实际挑战。本文基于在英国哥伦比亚癌症登记处（BCCR）实施各种NLP模型进行信息提取和分类任务的经验，分享了项目生命周期中的关键教训。我们强调，基于明确的商业目标而非仅技术准确性定义问题的重要性，采用迭代开发方法，并从一开始就促进跨学科的深度合作与协同设计，包括领域专家、终端用户和机器学习专家的参与。进一步的见解强调了实际模型选择（包括适当的混合方法和简单方法）、严格的数据质量关注（代表性、漂移、标注），以及涉及人工介入验证和持续审核的稳健错误缓解策略的重要性，并提高组织的人工智能素养。这些实用考虑，不仅适用于癌症登记处，还为寻求通过实施AI/NLP解决方案增强数据管理流程并最终改善患者护理和公共卫生结果的医疗组织提供了指导。 

---
# Personalized Product Search Ranking: A Multi-Task Learning Approach with Tabular and Non-Tabular Data 

**Title (ZH)**: 个性化产品搜索排序：一种结合表格和非表格数据的多任务学习方法 

**Authors**: Lalitesh Morishetti, Abhay Kumar, Jonathan Scott, Kaushiki Nag, Gunjan Sharma, Shanu Vashishtha, Rahul Sridhar, Rohit Chatter, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09636)  

**Abstract**: In this paper, we present a novel model architecture for optimizing personalized product search ranking using a multi-task learning (MTL) framework. Our approach uniquely integrates tabular and non-tabular data, leveraging a pre-trained TinyBERT model for semantic embeddings and a novel sampling technique to capture diverse customer behaviors. We evaluate our model against several baselines, including XGBoost, TabNet, FT-Transformer, DCN-V2, and MMoE, focusing on their ability to handle mixed data types and optimize personalized ranking. Additionally, we propose a scalable relevance labeling mechanism based on click-through rates, click positions, and semantic similarity, offering an alternative to traditional human-annotated labels. Experimental results show that combining non-tabular data with advanced embedding techniques in multi-task learning paradigm significantly enhances model performance. Ablation studies further underscore the benefits of incorporating relevance labels, fine-tuning TinyBERT layers, and TinyBERT query-product embedding interactions. These results demonstrate the effectiveness of our approach in achieving improved personalized product search ranking. 

**Abstract (ZH)**: 本文提出了一种新颖的模型架构，利用多任务学习（MTL）框架优化个性化产品搜索排名。该方法独特地整合了表结构数据和非表结构数据，利用预训练的TinyBERT模型进行语义嵌入，并提出了一种新颖的采样技术以捕捉多样化的客户行为。我们评估了该模型与XGBoost、TabNet、FT-Transformer、DCN-V2和MMoE等多种基线模型的性能，重点关注它们处理混合数据类型和优化个性化排名的能力。此外，我们提出了一种基于点击率、点击位置和语义相似性的可扩展的相关性标注机制，作为传统人工标注标签的替代方法。实验结果表明，将非表结构数据与多任务学习范式中的高级嵌入技术相结合，显著提升了模型性能。消融研究进一步强调了引入相关性标注、微调TinyBERT层以及TinyBERT查询-产品嵌入交互的优势。这些结果表明，我们的方法在实现改进的个性化产品搜索排名方面具有有效性。 

---
