# RM-Dijkstra: A surface optimal path planning algorithm based on Riemannian metric 

**Title (ZH)**: RM-Dijkstra: 基于黎曼度量的曲面最优路径规划算法 

**Authors**: Yu Zhang, Xiao-Song Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22170)  

**Abstract**: The Dijkstra algorithm is a classic path planning method, which operates in a discrete graph space to determine the shortest path from a specified source point to a target node or all other nodes based on non-negative edge weights. Numerous studies have focused on the Dijkstra algorithm due to its potential application. However, its application in surface path planning for mobile robots remains largely unexplored. In this letter, a surface optimal path planning algorithm called RM-Dijkstra is proposed, which is based on Riemannian metric model. By constructing a new Riemannian metric on the 2D projection plane, the surface optimal path planning problem is therefore transformed into a geometric problem on the 2D plane with new Riemannian metric. Induced by the standard Euclidean metric on surface, the constructed new metric reflects environmental information of the robot and ensures that the projection map is an isometric immersion. By conducting a series of simulation tests, the experimental results demonstrate that the RM-Dijkstra algorithm not only effectively solves the optimal path planning problem on surfaces, but also outperforms traditional path planning algorithms in terms of path accuracy and smoothness, particularly in complex scenarios. 

**Abstract (ZH)**: 刘iemannian度量模型的表面最优路径规划算法 

---
# A MILP-Based Solution to Multi-Agent Motion Planning and Collision Avoidance in Constrained Environments 

**Title (ZH)**: 基于 MILP 的多Agent运动规划与约束环境下碰撞避免问题的求解方法 

**Authors**: Akshay Jaitly, Jack Cline, Siavash Farzan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21982)  

**Abstract**: We propose a mixed-integer linear program (MILP) for multi-agent motion planning that embeds Polytopic Action-based Motion Planning (PAAMP) into a sequence-then-solve pipeline. Region sequences confine each agent to adjacent convex polytopes, while a big-M hyperplane model enforces inter-agent separation. Collision constraints are applied only to agents sharing or neighboring a region, which reduces binary variables exponentially compared with naive formulations. An L1 path-length-plus-acceleration cost yields smooth trajectories. We prove finite-time convergence and demonstrate on representative multi-agent scenarios with obstacles that our formulation produces collision-free trajectories an order of magnitude faster than an unstructured MILP baseline. 

**Abstract (ZH)**: 我们提出了一种混合整数线性规划（MILP）方法，将多面体动作基于运动规划（PAAMP）嵌入到顺序-求解管道中进行多agent运动规划。区域序列将每个agent限制在相邻的凸多面体内，而大M超平面模型则强制执行agent间的分离。仅对共享或邻接同一区域的agent应用碰撞约束，这与朴素公式相比极大地减少了二进制变量的数量。L1路径长度加加速度成本生成平滑轨迹。我们证明了有限时间收敛性，并在包含障碍物的代表性多agent场景中展示了与无结构MILP基准相比，我们的表达式能够快一个数量级地生成无碰撞轨迹。 

---
# M3PO: Massively Multi-Task Model-Based Policy Optimization 

**Title (ZH)**: M3PO：大规模多任务模型导向策略优化 

**Authors**: Aditya Narendra, Dmitry Makarov, Aleksandr Panov  

**Link**: [PDF](https://arxiv.org/pdf/2506.21782)  

**Abstract**: We introduce Massively Multi-Task Model-Based Policy Optimization (M3PO), a scalable model-based reinforcement learning (MBRL) framework designed to address sample inefficiency in single-task settings and poor generalization in multi-task domains. Existing model-based approaches like DreamerV3 rely on pixel-level generative models that neglect control-centric representations, while model-free methods such as PPO suffer from high sample complexity and weak exploration. M3PO integrates an implicit world model, trained to predict task outcomes without observation reconstruction, with a hybrid exploration strategy that combines model-based planning and model-free uncertainty-driven bonuses. This eliminates the bias-variance trade-off in prior methods by using discrepancies between model-based and model-free value estimates to guide exploration, while maintaining stable policy updates through a trust-region optimizer. M3PO provides an efficient and robust alternative to existing model-based policy optimization approaches and achieves state-of-the-art performance across multiple benchmarks. 

**Abstract (ZH)**: 基于大规模多任务模型的策略优化方法（M3PO）：一种可扩展的模型导向强化学习框架 

---
# Stochastic Neural Control Barrier Functions 

**Title (ZH)**: 随机神经控制障碍函数 

**Authors**: Hongchao Zhang, Manan Tayal, Jackson Cox, Pushpak Jagtap, Shishir Kolathaya, Andrew Clark  

**Link**: [PDF](https://arxiv.org/pdf/2506.21697)  

**Abstract**: Control Barrier Functions (CBFs) are utilized to ensure the safety of control systems. CBFs act as safety filters in order to provide safety guarantees without compromising system performance. These safety guarantees rely on the construction of valid CBFs. Due to their complexity, CBFs can be represented by neural networks, known as neural CBFs (NCBFs). Existing works on the verification of the NCBF focus on the synthesis and verification of NCBFs in deterministic settings, leaving the stochastic NCBFs (SNCBFs) less studied. In this work, we propose a verifiably safe synthesis for SNCBFs. We consider the cases of smooth SNCBFs with twice-differentiable activation functions and SNCBFs that utilize the Rectified Linear Unit or ReLU activation function. We propose a verification-free synthesis framework for smooth SNCBFs and a verification-in-the-loop synthesis framework for both smooth and ReLU SNCBFs. and we validate our frameworks in three cases, namely, the inverted pendulum, Darboux, and the unicycle model. 

**Abstract (ZH)**: Control Barrier Functions (CBFs) 用于确保控制系统的安全性。CBFs 作为安全过滤器，能够在不牺牲系统性能的前提下提供安全保证。这些安全保证依赖于有效 CBFs 的构造。由于其复杂性，CBFs 可以由神经网络表示，称为神经 CBFs (NCBFs)。现有关于 NCBF 验证的工作主要集中在确定性环境下的 NCBF 的合成与验证，而对随机 NCBF (SNCBF) 的研究较少。在这项工作中，我们提出了一个可验证的安全合成方法，适用于 SNCBF。我们考虑了具有二次可微激活函数的光滑 SNCBF 和使用修正线性单元 (ReLU) 激活函数的 SNCBF 的情况。我们提出了一种无验证合成框架，适用于光滑 SNCBF，并提出了一种带验证的合成框架，适用于光滑和 ReLU SNCBF。我们在倒立摆、Darboux 和单轮车模型的三种情况中验证了这些框架。 

---
# The DevSafeOps Dilemma: A Systematic Literature Review on Rapidity in Safe Autonomous Driving Development and Operation 

**Title (ZH)**: DevSafeOps  dilemma: 一项关于安全自主驾驶开发与运营快速性的系统文献审核 

**Authors**: Ali Nouri, Beatriz Cabrero-Daniel, Fredrik Törner, Christian Berger  

**Link**: [PDF](https://arxiv.org/pdf/2506.21693)  

**Abstract**: Developing autonomous driving (AD) systems is challenging due to the complexity of the systems and the need to assure their safe and reliable operation. The widely adopted approach of DevOps seems promising to support the continuous technological progress in AI and the demand for fast reaction to incidents, which necessitate continuous development, deployment, and monitoring. We present a systematic literature review meant to identify, analyse, and synthesise a broad range of existing literature related to usage of DevOps in autonomous driving development. Our results provide a structured overview of challenges and solutions, arising from applying DevOps to safety-related AI-enabled functions. Our results indicate that there are still several open topics to be addressed to enable safe DevOps for the development of safe AD. 

**Abstract (ZH)**: 开发自动驾驶（AD）系统具有挑战性，由于系统复杂性和确保其安全可靠运行的需要。广泛采用的DevOps方法似乎有助于支持AI的持续技术进步，并满足对快速应对事故的需求，这需要持续的开发、部署和监控。我们进行了一项系统文献综述，旨在识别、分析和综合与自动驾驶开发中使用DevOps相关的一系列现有文献。我们的结果提供了应用DevOps到安全相关的AI功能中所面临的挑战和解决方案的结构化概述。我们的结果表明，仍有许多开放话题需要解决，以实现安全的DevOps并促进安全自动驾驶系统的开发。 

---
# Advanced System Engineering Approaches to Emerging Challenges in Planetary and Deep-Space Exploration 

**Title (ZH)**: 面向行星及深空探索新兴挑战的先进系统工程方法 

**Authors**: J. de Curtò, Cristina LiCalzi, Julien Tubiana Warin, Jack Gehlert, Brian Langbein, Alexandre Gamboa, Chris Sixbey, William Maguire, Santiago Fernández, Álvaro Maestroarena, Alex Brenchley, Logan Maroclo, Philemon Mercado, Joshua DeJohn, Cesar Velez, Ethan Dahmus, Taylor Steinys, David Fritz, I. de Zarzà  

**Link**: [PDF](https://arxiv.org/pdf/2506.21648)  

**Abstract**: This paper presents innovative solutions to critical challenges in planetary and deep-space exploration electronics. We synthesize findings across diverse mission profiles, highlighting advances in: (1) MARTIAN positioning systems with dual-frequency transmission to achieve $\pm$1m horizontal accuracy; (2) artificial reef platforms for Titan's hydrocarbon seas utilizing specialized sensor arrays and multi-stage communication chains; (3) precision orbital rendezvous techniques demonstrating novel thermal protection solutions; (4) miniaturized CubeSat architectures for asteroid exploration with optimized power-to-mass ratios; and (5) next-generation power management systems for MARS rovers addressing dust accumulation challenges. These innovations represent promising directions for future space exploration technologies, particularly in environments where traditional Earth-based electronic solutions prove inadequate. The interdisciplinary nature of these developments highlights the critical intersection of aerospace engineering, electrical engineering, and planetary science in advancing human exploration capabilities beyond Earth orbit. 

**Abstract (ZH)**: 本文提出了应对行星及深空探测电子设备关键挑战的创新解决方案，涵盖了：(1) 带有双频传输的火星定位系统，实现±1米水平精度；(2) 利用专门传感器阵列和多级通信链路的泰坦烃海人造礁平台；(3) 精确轨道交会技术，展示新型热防护解决方案；(4) 用于小行星探测的微型立方星架构，优化功率与质量比；以及(5) 针对火星车粉尘累积挑战的下一代电源管理系统。这些创新代表了未来太空探测技术开发的前景，特别是在传统地球电子解决方案不足的环境中尤为重要。这些发展的跨学科性质突显了航空航天工程、电气工程与行星科学在拓展地球轨道外的人类探测能力方面的关键交汇点。 

---
# AI Model Passport: Data and System Traceability Framework for Transparent AI in Health 

**Title (ZH)**: AI模型护照：面向透明医疗的人工智能数据与系统可追溯性框架 

**Authors**: Varvara Kalokyri, Nikolaos S. Tachos, Charalampos N. Kalantzopoulos, Stelios Sfakianakis, Haridimos Kondylakis, Dimitrios I. Zaridis, Sara Colantonio, Daniele Regge, Nikolaos Papanikolaou, ProCAncer-I consortium, Konstantinos Marias, Dimitrios I. Fotiadis, Manolis Tsiknakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.22358)  

**Abstract**: The increasing integration of Artificial Intelligence (AI) into health and biomedical systems necessitates robust frameworks for transparency, accountability, and ethical compliance. Existing frameworks often rely on human-readable, manual documentation which limits scalability, comparability, and machine interpretability across projects and platforms. They also fail to provide a unique, verifiable identity for AI models to ensure their provenance and authenticity across systems and use cases, limiting reproducibility and stakeholder trust. This paper introduces the concept of the AI Model Passport, a structured and standardized documentation framework that acts as a digital identity and verification tool for AI models. It captures essential metadata to uniquely identify, verify, trace and monitor AI models across their lifecycle - from data acquisition and preprocessing to model design, development and deployment. In addition, an implementation of this framework is presented through AIPassport, an MLOps tool developed within the ProCAncer-I EU project for medical imaging applications. AIPassport automates metadata collection, ensures proper versioning, decouples results from source scripts, and integrates with various development environments. Its effectiveness is showcased through a lesion segmentation use case using data from the ProCAncer-I dataset, illustrating how the AI Model Passport enhances transparency, reproducibility, and regulatory readiness while reducing manual effort. This approach aims to set a new standard for fostering trust and accountability in AI-driven healthcare solutions, aspiring to serve as the basis for developing transparent and regulation compliant AI systems across domains. 

**Abstract (ZH)**: AI模型护照：促进AI驱动医疗健康解决方案中的透明度、问责制和合规性的新标准 

---
# Conceptual Topic Aggregation 

**Title (ZH)**: 概念主题聚合 

**Authors**: Klara M. Gutekunst, Dominik Dürrschnabel, Johannes Hirth, Gerd Stumme  

**Link**: [PDF](https://arxiv.org/pdf/2506.22309)  

**Abstract**: The vast growth of data has rendered traditional manual inspection infeasible, necessitating the adoption of computational methods for efficient data exploration. Topic modeling has emerged as a powerful tool for analyzing large-scale textual datasets, enabling the extraction of latent semantic structures. However, existing methods for topic modeling often struggle to provide interpretable representations that facilitate deeper insights into data structure and content. In this paper, we propose FAT-CAT, an approach based on Formal Concept Analysis (FCA) to enhance meaningful topic aggregation and visualization of discovered topics. Our approach can handle diverse topics and file types -- grouped by directories -- to construct a concept lattice that offers a structured, hierarchical representation of their topic distribution. In a case study on the ETYNTKE dataset, we evaluate the effectiveness of our approach against other representation methods to demonstrate that FCA-based aggregation provides more meaningful and interpretable insights into dataset composition than existing topic modeling techniques. 

**Abstract (ZH)**: 数据量的大幅增长使传统的手动检查不可行， necessitating the adoption of computational methods for efficient data exploration. 主题建模已成为分析大规模文本数据集的强大工具，能够提取潜在的语义结构。然而，现有主题建模方法往往难以提供可解释的表现形式，以促进对数据结构和内容的更深入洞察。本文提出了一种基于形式概念分析（FCA）的FAT-CAT方法，以增强具有意义的主题聚合和发现的主题可视化。该方法可以处理由目录分组的多样化主题和文件类型，构建一个概念格，提供主题分布的结构化、层次表示。在ETYNTKE数据集的案例研究中，我们评估了该方法与其他表示方法的有效性，以证明基于形式概念分析的聚合比现有主题建模技术能提供更具有意义和可解释的数据集组成洞察。 

---
# Breaking Rank Bottlenecks in Knowledge Graph Completion 

**Title (ZH)**: 打破知识图谱补全中的排列瓶颈 

**Authors**: Samy Badreddine, Emile van Krieken, Luciano Serafini  

**Link**: [PDF](https://arxiv.org/pdf/2506.22271)  

**Abstract**: Many Knowledge Graph Completion (KGC) models, despite using powerful encoders, rely on a simple vector-matrix multiplication to score queries against candidate object entities. When the number of entities is larger than the model's embedding dimension, which in practical scenarios is often by several orders of magnitude, we have a linear output layer with a rank bottleneck. Such bottlenecked layers limit model expressivity. We investigate both theoretically and empirically how rank bottlenecks affect KGC models. We find that, by limiting the set of feasible predictions, rank bottlenecks hurt ranking accuracy and the distribution fidelity of scores. Inspired by the language modelling literature, we propose KGE-MoS, a mixture-based output layer to break rank bottlenecks in many KGC models. Our experiments on four datasets show that KGE-MoS improves performance and probabilistic fit of KGC models for a low parameter cost. 

**Abstract (ZH)**: 多种知识图谱补全模型尽管使用了强大的编码器，但在评分查询与候选项实体时，仍依赖简单的向量-矩阵乘法。当实体数量远大于模型的嵌入维度时，会导致线性输出层出现秩瓶颈，限制了模型的表达能力。我们从理论和实验上探讨了秩瓶颈对知识图谱补全模型的影响。我们发现，秩瓶颈通过限制可预测集，影响了排名准确性和分数分布的真实性。受到自然语言处理文献的启发，我们提出了一种基于混合输出层的KGE-MoS模型，以打破多种知识图谱补全模型中的秩瓶颈。在四个数据集上的实验结果显示，KGE-MoS 以较低的参数成本提高了知识图谱补全模型的性能和概率拟合度。 

---
# A Different Approach to AI Safety: Proceedings from the Columbia Convening on Openness in Artificial Intelligence and AI Safety 

**Title (ZH)**: 人工智能安全的新途径：哥伦比亚开放人工智能与人工智能安全 convening 论文集 

**Authors**: Camille François, Ludovic Péran, Ayah Bdeir, Nouha Dziri, Will Hawkins, Yacine Jernite, Sayash Kapoor, Juliet Shen, Heidy Khlaaf, Kevin Klyman, Nik Marda, Marie Pellat, Deb Raji, Divya Siddarth, Aviya Skowron, Joseph Spisak, Madhulika Srikumar, Victor Storchan, Audrey Tang, Jen Weedon  

**Link**: [PDF](https://arxiv.org/pdf/2506.22183)  

**Abstract**: The rapid rise of open-weight and open-source foundation models is intensifying the obligation and reshaping the opportunity to make AI systems safe. This paper reports outcomes from the Columbia Convening on AI Openness and Safety (San Francisco, 19 Nov 2024) and its six-week preparatory programme involving more than forty-five researchers, engineers, and policy leaders from academia, industry, civil society, and government. Using a participatory, solutions-oriented process, the working groups produced (i) a research agenda at the intersection of safety and open source AI; (ii) a mapping of existing and needed technical interventions and open source tools to safely and responsibly deploy open foundation models across the AI development workflow; and (iii) a mapping of the content safety filter ecosystem with a proposed roadmap for future research and development. We find that openness -- understood as transparent weights, interoperable tooling, and public governance -- can enhance safety by enabling independent scrutiny, decentralized mitigation, and culturally plural oversight. However, significant gaps persist: scarce multimodal and multilingual benchmarks, limited defenses against prompt-injection and compositional attacks in agentic systems, and insufficient participatory mechanisms for communities most affected by AI harms. The paper concludes with a roadmap of five priority research directions, emphasizing participatory inputs, future-proof content filters, ecosystem-wide safety infrastructure, rigorous agentic safeguards, and expanded harm taxonomies. These recommendations informed the February 2025 French AI Action Summit and lay groundwork for an open, plural, and accountable AI safety discipline. 

**Abstract (ZH)**: 开放权重和开源基础模型的快速崛起正加剧确保人工智能系统安全的义务并重塑相关机会。本文报告了哥伦比亚大学人工智能开放性和安全性会议（旧金山，2024年11月19日）及其为期六周的准备工作成果，涉及来自学术界、产业界、民间社会和政府部门的四十五多位研究人员、工程师和政策领导者。通过参与式、以解决方案为导向的过程，工作组制定了三项成果：（i）衔接安全与开源人工智能的研究议程；（ii）绘制既存和所需的技术干预及开源工具地图，确保负责任地部署开放基础模型贯穿整个人工智能开发流程；（iii）绘制内容安全过滤生态系统，并提出未来研究与开发的路线图。我们发现，开放性——即透明权重、兼容工具和公共治理——能够通过促进独立审查、分散性缓解和文化多元监督来增强安全性。然而，仍存在显著差距：稀缺的多模态和多语言基准、针对代理系统中提示注入和组合攻击的有限防御措施，以及对受人工智能危害影响最深的社区参与不足的机制。本文以五项优先研究方向为结尾，强调参与式输入、未来导向的内容过滤、生态系统范围的安全基础设施、严格的代理保护措施，以及扩展危害分类。这些建议为2025年2月法国人工智能行动峰会提供了指导，并为开放、多元和问责制导向的人工智能安全学科奠定了基础。 

---
# Query as Test: An Intelligent Driving Test and Data Storage Method for Integrated Cockpit-Vehicle-Road Scenarios 

**Title (ZH)**: 查询即测试：面向集成驾驶舱-车辆-道路场景的智能驾驶测试与数据存储方法 

**Authors**: Shengyue Yao, Runqing Guo, Yangyang Qin, Miangbing Meng, Jipeng Cao, Yilun Lin, Yisheng Lv, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22068)  

**Abstract**: With the deep penetration of Artificial Intelligence (AI) in the transportation sector, intelligent cockpits, autonomous driving, and intelligent road networks are developing at an unprecedented pace. However, the data ecosystems of these three key areas are increasingly fragmented and incompatible. Especially, existing testing methods rely on data stacking, fail to cover all edge cases, and lack flexibility. To address this issue, this paper introduces the concept of "Query as Test" (QaT). This concept shifts the focus from rigid, prescripted test cases to flexible, on-demand logical queries against a unified data representation. Specifically, we identify the need for a fundamental improvement in data storage and representation, leading to our proposal of "Extensible Scenarios Notations" (ESN). ESN is a novel declarative data framework based on Answer Set Programming (ASP), which uniformly represents heterogeneous multimodal data from the cockpit, vehicle, and road as a collection of logical facts and rules. This approach not only achieves deep semantic fusion of data, but also brings three core advantages: (1) supports complex and flexible semantic querying through logical reasoning; (2) provides natural interpretability for decision-making processes; (3) allows for on-demand data abstraction through logical rules, enabling fine-grained privacy protection. We further elaborate on the QaT paradigm, transforming the functional validation and safety compliance checks of autonomous driving systems into logical queries against the ESN database, significantly enhancing the expressiveness and formal rigor of the testing. Finally, we introduce the concept of "Validation-Driven Development" (VDD), which suggests to guide developments by logical validation rather than quantitative testing in the era of Large Language Models, in order to accelerating the iteration and development process. 

**Abstract (ZH)**: 随着人工智能（AI）在交通运输领域的深入渗透，智能驾驶舱、自动驾驶和智能道路网络正在以前所未有的速度发展。然而，这三个关键领域的数据生态系统日益碎片化且不兼容。特别是，现有的测试方法依赖于数据堆叠，无法覆盖所有边缘案例，并且缺乏灵活性。为了解决这一问题，本文引入了“查询即测试”（Query as Test，QaT）的概念。这一概念将重点从僵化的预设测试案例转移到针对统一数据表示进行灵活的按需逻辑查询。具体而言，我们识别了在数据存储和表示方面进行根本改进的必要性，提出了“扩展场景表示”（Extensible Scenarios Notations，ESN）的概念。ESN 是一种基于回答集编程（Answer Set Programming，ASP）的新颖声明式数据框架，统一表示来自驾驶舱、车辆和道路的异构多模态数据，作为一组逻辑事实和规则的集合。这种方法不仅实现了数据的深度语义融合，还带来了三个核心优势：（1）通过逻辑推理支持复杂和灵活的语义查询；（2）为决策过程提供自然的可解释性；（3）通过逻辑规则进行按需数据抽象，实现精细的隐私保护。我们进一步阐述了QaT 帕累托，将自动驾驶系统的功能验证和安全合规检查转化为对ESN数据库的逻辑查询，极大地提升了测试的表达能力和形式严谨性。最后，我们提出了“验证驱动开发”（Validation-Driven Development，VDD）的概念，在大语言模型时代，通过逻辑验证而非定量测试来指导开发，以加速迭代和发展过程。 

---
# AlphaBeta is not as good as you think: a new probabilistic model to better analyze deterministic game-solving algorithms 

**Title (ZH)**: AlphaBeta并非你所想象的那样好：一种更好的确定性游戏求解算法的概率模型分析 

**Authors**: Raphaël Boige, Amine Boumaza, Bruno Scherrer  

**Link**: [PDF](https://arxiv.org/pdf/2506.21996)  

**Abstract**: Deterministic game-solving algorithms are conventionally analyzed in the light of their average-case complexity against a distribution of random game-trees, where leaf values are independently sampled from a fixed distribution. This simplified model enables uncluttered mathematical analysis, revealing two key properties: root value distributions asymptotically collapse to a single fixed value for finite-valued trees, and all reasonable algorithms achieve global optimality. However, these findings are artifacts of the model's design-its long criticized independence assumption strips games of structural complexity, producing trivial instances where no algorithm faces meaningful challenges. To address this limitation, we introduce a new probabilistic model that incrementally constructs game-trees using a fixed level-wise conditional distribution. By enforcing ancestor dependency, a critical structural feature of real-world games, our framework generates problems with adjustable difficulty while retaining some form of analytical tractability. For several algorithms, including AlphaBeta and Scout, we derive recursive formulas characterizing their average-case complexities under this model. These allow us to rigorously compare algorithms on deep game-trees, where Monte-Carlo simulations are no longer feasible. While asymptotically, all algorithms seem to converge to identical branching factor (a result analogous to those of independence-based models), deep finite trees reveal stark differences: AlphaBeta incurs a significantly larger constant multiplicative factor compared to algorithms like Scout, leading to a substantial practical slowdown. Our framework sheds new light on classical game-solving algorithms, offering rigorous evidence and analytical tools to advance the understanding of these methods under a more realistic, challenging, and yet tractable model. 

**Abstract (ZH)**: 确定性博弈求解算法通常基于随机博弈树的平均案例复杂性进行分析，其中叶子值独立地从固定分布中抽样。这种简化模型便于数学分析，揭示了两个关键性质：有限值树的根值分布渐近塌缩到一个固定值，并且所有合理算法均能达到全局最优。然而，这些发现是该模型设计的产物——其长期受到批评的独立假设剥离了博弈的结构性复杂性，产生了一种过于简化的实例，其中没有算法面临有意义的挑战。为解决这一局限，我们引入了一种新的概率模型，该模型通过固定层级条件分布逐层构建博弈树。通过强制祖先依赖性，这是一个现实世界博弈的关键结构特征，我们的框架可以生成具有可调整难度的问题，同时保持一定程度的分析可处理性。对于包括AlphaBeta和Scout在内的几种算法，我们推导出了刻画其在该模型下的平均案例复杂性的递归公式。这使我们能够在深层博弈树上严格比较算法性能，而蒙特卡洛模拟已不再可行。尽管从渐近角度来看，所有算法似乎收敛到同一分支因子（独立假设模型中类似的结果），但深层有限树揭示了显著差异：AlphaBeta 比Scout等算法产生了显著更大的常数乘数因子，导致了实质性的性能下降。我们的框架为古典博弈求解算法带来了新的见解，提供了严谨的证据和分析工具，以在一种更为现实、更具挑战性但又可处理的模型下推动对这些方法的理解。 

---
# Interactive Multi-Objective Probabilistic Preference Learning with Soft and Hard Bounds 

**Title (ZH)**: 交互式多目标概率偏好学习及其软硬约束 

**Authors**: Edward Chen, Sang T. Truong, Natalie Dullerud, Sanmi Koyejo, Carlos Guestrin  

**Link**: [PDF](https://arxiv.org/pdf/2506.21887)  

**Abstract**: High-stakes decision-making involves navigating multiple competing objectives with expensive evaluations. For instance, in brachytherapy, clinicians must balance maximizing tumor coverage (e.g., an aspirational target or soft bound of >95% coverage) against strict organ dose limits (e.g., a non-negotiable hard bound of <601 cGy to the bladder), with each plan evaluation being resource-intensive. Selecting Pareto-optimal solutions that match implicit preferences is challenging, as exhaustive Pareto frontier exploration is computationally and cognitively prohibitive, necessitating interactive frameworks to guide users. While decision-makers (DMs) often possess domain knowledge to narrow the search via such soft-hard bounds, current methods often lack systematic approaches to iteratively refine these multi-faceted preference structures. Critically, DMs must trust their final decision, confident they haven't missed superior alternatives; this trust is paramount in high-consequence scenarios. We present Active-MoSH, an interactive local-global framework designed for this process. Its local component integrates soft-hard bounds with probabilistic preference learning, maintaining distributions over DM preferences and bounds for adaptive Pareto subset refinement. This is guided by an active sampling strategy optimizing exploration-exploitation while minimizing cognitive burden. To build DM trust, Active-MoSH's global component, T-MoSH, leverages multi-objective sensitivity analysis to identify potentially overlooked, high-value points beyond immediate feedback. We demonstrate Active-MoSH's performance benefits through diverse synthetic and real-world applications. A user study on AI-generated image selection further validates our hypotheses regarding the framework's ability to improve convergence, enhance DM trust, and provide expressive preference articulation, enabling more effective DMs. 

**Abstract (ZH)**: 高强度决策涉及在昂贵评估中平衡多重竞争目标。例如，在近距离放疗中，临床医生必须在最大化肿瘤覆盖（例如，抱负目标或>95%的覆盖率）与严格的器官剂量限制（例如，不可谈判的硬限<601 cGy的膀胱剂量）之间进行权衡，每次计划评估都资源密集型。选择与隐含偏好匹配的帕累托最优解具有挑战性，因为全面探索帕累托前沿在计算和认知上都是难以承受的，因此需要交互式框架来指导用户。尽管决策者（DMs）通常具备利用软-硬限制狭窄搜索领域的领域知识，当前的方法往往缺乏系统的方法来逐步细化这些多方面偏好结构。关键的是，DMs必须对其最终决策充满信心，确信他们没有遗漏更优的选择；这种信任在高后果场景中至关重要。我们提出了一种名为Active-MoSH的交互式局部-全局框架，专为这一过程设计。其局部组件结合了软-硬限制与概率性偏好学习，保持DM偏好和限制的概率分布，以适应性地细化帕累托子集。这由一种积极采样策略指导，该策略优化探索与利用之间的平衡并最小化认知负担。为建立DM信任，Active-MoSH的全局组件T-MoSH利用多目标灵敏度分析来识别可能被忽视的、具有高价值的点，这些点超出了即时反馈。我们通过多种合成和真实世界的应用展示了Active-MoSH的性能优势。一项关于AI生成图像选择的用户研究进一步验证了该框架在提高收敛性、增强DM信任和提供表达性偏好陈述方面的能力，从而促进更有效的DM。 

---
# CLoVE: Personalized Federated Learning through Clustering of Loss Vector Embeddings 

**Title (ZH)**: CLoVE：通过损失向量嵌入聚类实现的个性化联邦学习 

**Authors**: Randeep Bhatia, Nikos Papadis, Murali Kodialam, TV Lakshman, Sayak Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2506.22427)  

**Abstract**: We propose CLoVE (Clustering of Loss Vector Embeddings), a novel algorithm for Clustered Federated Learning (CFL). In CFL, clients are naturally grouped into clusters based on their data distribution. However, identifying these clusters is challenging, as client assignments are unknown. CLoVE utilizes client embeddings derived from model losses on client data, and leverages the insight that clients in the same cluster share similar loss values, while those in different clusters exhibit distinct loss patterns. Based on these embeddings, CLoVE is able to iteratively identify and separate clients from different clusters and optimize cluster-specific models through federated aggregation. Key advantages of CLoVE over existing CFL algorithms are (1) its simplicity, (2) its applicability to both supervised and unsupervised settings, and (3) the fact that it eliminates the need for near-optimal model initialization, which makes it more robust and better suited for real-world applications. We establish theoretical convergence bounds, showing that CLoVE can recover clusters accurately with high probability in a single round and converges exponentially fast to optimal models in a linear setting. Our comprehensive experiments comparing with a variety of both CFL and generic Personalized Federated Learning (PFL) algorithms on different types of datasets and an extensive array of non-IID settings demonstrate that CLoVE achieves highly accurate cluster recovery in just a few rounds of training, along with state-of-the-art model accuracy, across a variety of both supervised and unsupervised PFL tasks. 

**Abstract (ZH)**: CLoVE：群集联邦学习的聚类损失向量嵌入算法 

---
# Multi-View Contrastive Learning for Robust Domain Adaptation in Medical Time Series Analysis 

**Title (ZH)**: 多视角对比学习在医学时间序列分析中的鲁棒领域适应 

**Authors**: YongKyung Oh, Alex Bui  

**Link**: [PDF](https://arxiv.org/pdf/2506.22393)  

**Abstract**: Adapting machine learning models to medical time series across different domains remains a challenge due to complex temporal dependencies and dynamic distribution shifts. Current approaches often focus on isolated feature representations, limiting their ability to fully capture the intricate temporal dynamics necessary for robust domain adaptation. In this work, we propose a novel framework leveraging multi-view contrastive learning to integrate temporal patterns, derivative-based dynamics, and frequency-domain features. Our method employs independent encoders and a hierarchical fusion mechanism to learn feature-invariant representations that are transferable across domains while preserving temporal coherence. Extensive experiments on diverse medical datasets, including electroencephalogram (EEG), electrocardiogram (ECG), and electromyography (EMG) demonstrate that our approach significantly outperforms state-of-the-art methods in transfer learning tasks. By advancing the robustness and generalizability of machine learning models, our framework offers a practical pathway for deploying reliable AI systems in diverse healthcare settings. 

**Abstract (ZH)**: 跨不同领域适应医学时间序列的机器学习模型仍面临挑战，这归因于复杂的时间依赖性和动态分布转移。当前的方法往往专注于孤立的特征表示，限制了其全面捕捉对鲁棒领域适应至关重要的微妙时间动态的能力。在本文中，我们提出了一个新颖的框架，利用多视图对比学习来整合时间模式、基于导数的动力学和频率域特征。该方法采用独立编码器和分层融合机制来学习在保持时间连贯性的同时能够在不同领域间转移的特征不变表示。在包括脑电图（EEG）、心电图（ECG）和肌电图（EMG）在内的多种医学数据集上的广泛实验表明，我们的方法在迁移学习任务中显著优于现有最先进的方法。通过提高机器学习模型的稳健性和泛化性，我们的框架为在多种医疗保健环境中部署可靠的AI系统提供了实用途径。 

---
# Towards Distributed Neural Architectures 

**Title (ZH)**: 面向分布式神经架构 

**Authors**: Aditya Cowsik, Tianyu He, Andrey Gromov  

**Link**: [PDF](https://arxiv.org/pdf/2506.22389)  

**Abstract**: We introduce and train distributed neural architectures (DNA) in vision and language domains. DNAs are initialized with a proto-architecture that consists of (transformer, MLP, attention, etc.) modules and routers. Any token (or patch) can traverse any series of modules in any order. DNAs are a natural generalization of the sparse methods such as Mixture-of-Experts, Mixture-of-Depths, parameter sharing, etc. Computation and communication patterns of DNA modules are learnt end-to-end during training and depend on the content and context of each token (or patch). These patterns can be shaped by further requirements added to the optimization objective such as compute/memory efficiency or load balancing. We empirically show that (i) trained DNAs are competitive with the dense baselines in both domains and (ii) compute efficiency/parameter sharing can be learnt from data. Next, we analyze the emergent connectivity and computation patterns in the trained DNAs. We find that the paths that tokens take through the models are themselves distributed according to a power-law. We show that some paths (or, equivalently, groups of modules) show emergent specialization. Finally, we demonstrate that models learn to allocate compute and active parameters in an interpretable way. 

**Abstract (ZH)**: 我们介绍并训练了视觉和语言领域的分布式神经架构（DNA）。DNA初始化时包含以（Transformer、MLP、注意力等）模块和路由器组成的原型架构。任何标记（或补丁）都可以以任意顺序遍历任何系列的模块。DNA是稀疏方法如Mixture-of-Experts、Mixture-of-Depths、参数共享等的自然扩展。在训练过程中，DNA模块的计算和通信模式会随着内容和上下文的不同而学习，且这些模式可以进一步通过优化目标中的要求来调整，如计算/内存效率或负载均衡。实验表明，（i）训练后的DNA在两个领域中与密集基线具有竞争力，（ii）计算效率/参数共享可以从数据中学习。接下来，我们分析了训练后的DNA中 Emergent 连接和计算模式。我们发现，标记通过模型所走的路径本身遵循幂律分布。我们展示了某些路径（或等效地，模块组）显示出 Emergent 专业化。最后，我们证明了模型能够以可解释的方式分配计算资源和激活参数。 

---
# Can Video Large Multimodal Models Think Like Doubters-or Double-Down: A Study on Defeasible Video Entailment 

**Title (ZH)**: 视频大型多模态模型会像怀疑论者思考，还是笃定其信：一种关于可消解视频蕴含的研究 

**Authors**: Yue Zhang, Jilei Sun, Yunhui Guo, Vibhav Gogate  

**Link**: [PDF](https://arxiv.org/pdf/2506.22385)  

**Abstract**: Video Large Multimodal Models (VLMMs) have made impressive strides in understanding video content, but they often struggle with abstract and adaptive reasoning-the ability to revise their interpretations when new information emerges. In reality, conclusions are rarely set in stone; additional context can strengthen or weaken an initial inference. To address this, we introduce Defeasible Video Entailment (DVidE), a new task that challenges models to think like doubters, constantly updating their reasoning based on evolving evidence. In DVidE, given a video premise and a textual hypothesis, models must determine whether a new update strengthens or weakens the hypothesis (classification version) or generate a coherent update that modifies the entailment relationship (generation version). For solving the classification task, we propose the Chain of Counterfactual Thought framework, utilizing counterfactual reasoning, ASR-enhanced video content, and rationale refinement to reduce inference bias. For the generation task, we develop a framework that combines ASR output with a Large Language Model (LLM) to produce coherent, contextually relevant updates aligned with the intended strengthener or weakener goals. Additionally, we introduce a novel benchmark dataset, with strengthener/weakener annotations and an LLM-based evaluation metric specifically designed for assessing generative performance. Experimental results demonstrate significant improvements, highlighting our proposed method in enhancing dynamic reasoning capabilities of VLMMs. 

**Abstract (ZH)**: Defeasible Video Entailment: A New Task for Dynamic Reasoning in Video Large Multimodal Models 

---
# Sheaf-Based Decentralized Multimodal Learning for Next-Generation Wireless Communication Systems 

**Title (ZH)**: 基于层结构的分布式多模态学习为下一代无线通信系统 

**Authors**: Abdulmomen Ghalkha, Zhuojun Tian, Chaouki Ben Issaid, Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2506.22374)  

**Abstract**: In large-scale communication systems, increasingly complex scenarios require more intelligent collaboration among edge devices collecting various multimodal sensory data to achieve a more comprehensive understanding of the environment and improve decision-making accuracy. However, conventional federated learning (FL) algorithms typically consider unimodal datasets, require identical model architectures, and fail to leverage the rich information embedded in multimodal data, limiting their applicability to real-world scenarios with diverse modalities and varying client capabilities. To address this issue, we propose Sheaf-DMFL, a novel decentralized multimodal learning framework leveraging sheaf theory to enhance collaboration among devices with diverse modalities. Specifically, each client has a set of local feature encoders for its different modalities, whose outputs are concatenated before passing through a task-specific layer. While encoders for the same modality are trained collaboratively across clients, we capture the intrinsic correlations among clients' task-specific layers using a sheaf-based structure. To further enhance learning capability, we propose an enhanced algorithm named Sheaf-DMFL-Att, which tailors the attention mechanism within each client to capture correlations among different modalities. A rigorous convergence analysis of Sheaf-DMFL-Att is provided, establishing its theoretical guarantees. Extensive simulations are conducted on real-world link blockage prediction and mmWave beamforming scenarios, demonstrate the superiority of the proposed algorithms in such heterogeneous wireless communication systems. 

**Abstract (ZH)**: 在大规模通信系统中，日益复杂的场景需要边缘设备收集各种多模态 sensory 数据进行更智能的合作，以实现环境的全面理解并提高决策准确度。然而，传统的联邦学习（FL）算法通常仅考虑单模态数据集，要求模型架构一致，并未能充分利用嵌入在多模态数据中的丰富信息，限制了其在具有多样模态和不同客户端能力的真实场景中的应用。为解决此问题，我们提出了一种名为 Sheaf-DMFL 的新型分散式多模态学习框架，利用 sheaf 理论增强具有不同模态设备之间的合作。具体而言，每个客户端对其不同的模态具有一个本地特征编码器集，其输出在经过任务特定层前被串联。虽然相同模态的编码器在客户端之间协同训练，我们使用基于 sheaf 的结构捕捉客户端任务特定层之间的内在相关性。为进一步增强学习能力，我们提出了名为 Sheaf-DMFL-Att 的增强算法，其中针对每个客户端尾随不同的模态间的相关性来定制注意机制。提供了 Sheaf-DMFL-Att 的严格收敛分析，建立了其理论保证。在实际的链接阻塞预测和毫米波波束形成场景中的广泛仿真实验表明，所提出的算法在异构无线通信系统中具有优越性。 

---
# Concept-Level AI for Telecom: Moving Beyond Large Language Models 

**Title (ZH)**: 电信领域的概念级AI：超越大型语言模型 

**Authors**: Viswanath Kumarskandpriya, Abdulhalim Dandoush, Abbas Bradai, Ali Belgacem  

**Link**: [PDF](https://arxiv.org/pdf/2506.22359)  

**Abstract**: The telecommunications and networking domain stands at the precipice of a transformative era, driven by the necessity to manage increasingly complex, hierarchical, multi administrative domains (i.e., several operators on the same path) and multilingual systems. Recent research has demonstrated that Large Language Models (LLMs), with their exceptional general-purpose text analysis and code generation capabilities, can be effectively applied to certain telecom problems (e.g., auto-configuration of data plan to meet certain application requirements). However, due to their inherent token-by-token processing and limited capacity for maintaining extended context, LLMs struggle to fulfill telecom-specific requirements such as cross-layer dependency cascades (i.e., over OSI), temporal-spatial fault correlation, and real-time distributed coordination. In contrast, Large Concept Models (LCMs), which reason at the abstraction level of semantic concepts rather than individual lexical tokens, offer a fundamentally superior approach for addressing these telecom challenges. By employing hyperbolic latent spaces for hierarchical representation and encapsulating complex multi-layered network interactions within concise concept embeddings, LCMs overcome critical shortcomings of LLMs in terms of memory efficiency, cross-layer correlation, and native multimodal integration. This paper argues that adopting LCMs is not simply an incremental step, but a necessary evolutionary leap toward achieving robust and effective AI-driven telecom management. 

**Abstract (ZH)**: 电信和网络领域正站在一个变革时代之 brink，受到管理日益复杂、分级、多管理域（即同一路径上的多个运营商）和多语言系统的迫切需求的驱动。近期研究显示，大型语言模型（LLMs）凭借其出色的通用文本分析和代码生成能力，可以有效应用于某些电信问题（例如，根据特定应用程序要求自动配置数据计划）。然而，由于其逐个处理标记的本质和维持长期上下文能力有限，LLMs 在满足电信特定需求（如跨层依赖级联、时空故障相关性及实时分布式协调）方面遇到困难。相比之下，大型概念模型（LCMs）通过在概念语义层面而非单个词汇标记层面进行推理，提供了根本上更优的方法来解决这些电信挑战。通过运用双曲隐空间实现层次表示，并在简洁的概念嵌入中封装复杂的多层网络交互，LCMs 在内存效率、跨层相关性和原生多模态集成方面克服了LLMs的关键不足。本文认为，采用LCMs 不仅是简单的进步，而是朝着实现稳健有效的AI驱动电信管理所必需的进化飞跃。 

---
# A Framework for Multi-source Privacy Preserving Epidemic Analysis 

**Title (ZH)**: 多源隐私保护流行病分析框架 

**Authors**: Zihan Guan, Zhiyuan Zhao, Fengwei Tian, Dung Nguyen, Payel Bhattacharjee, Ravi Tandon, B. Aditya Prakash, Anil Vullikanti  

**Link**: [PDF](https://arxiv.org/pdf/2506.22342)  

**Abstract**: It is now well understood that diverse datasets provide a lot of value in key epidemiology and public health analyses, such as forecasting and nowcasting, development of epidemic models, evaluation and design of interventions and resource allocation. Some of these datasets are often sensitive, and need adequate privacy protections. There are many models of privacy, but Differential Privacy (DP) has become a de facto standard because of its strong guarantees, without making models about adversaries. In this paper, we develop a framework the integrates deep learning and epidemic models to simultaneously perform epidemic forecasting and learning a mechanistic model of epidemic spread, while incorporating multiple datasets for these analyses, including some with DP guarantees. We demonstrate our framework using a realistic but synthetic financial dataset with DP; such a dataset has not been used in such epidemic analyses. We show that this dataset provides significant value in forecasting and learning an epidemic model, even when used with DP guarantees. 

**Abstract (ZH)**: 多元数据集在关键流行病学和公共卫生分析中的价值：结合深度学习和隐私保护的传染病预测与传播机制学习框架 

---
# Less Greedy Equivalence Search 

**Title (ZH)**: 不太贪婪的等价搜索 

**Authors**: Adiba Ejaz, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2506.22331)  

**Abstract**: Greedy Equivalence Search (GES) is a classic score-based algorithm for causal discovery from observational data. In the sample limit, it recovers the Markov equivalence class of graphs that describe the data. Still, it faces two challenges in practice: computational cost and finite-sample accuracy. In this paper, we develop Less Greedy Equivalence Search (LGES), a variant of GES that retains its theoretical guarantees while partially addressing these limitations. LGES modifies the greedy step: rather than always applying the highest-scoring insertion, it avoids edge insertions between variables for which the score implies some conditional independence. This more targeted search yields up to a \(10\)-fold speed-up and a substantial reduction in structural error relative to GES. Moreover, LGES can guide the search using prior assumptions, while correcting these assumptions when contradicted by the data. Finally, LGES can exploit interventional data to refine the learned observational equivalence class. We prove that LGES recovers the true equivalence class in the sample limit from observational and interventional data, even with misspecified prior assumptions. Experiments demonstrate that LGES outperforms GES and other baselines in speed, accuracy, and robustness to misspecified assumptions. Our code is available at this https URL. 

**Abstract (ZH)**: Less Greedy Equivalence Search (LGES): A Variant of GES for Efficient and Accurate Causal Discovery 

---
# A Practical Approach to Power Saving in Hearables Using Sub-Nyquist Sampling with Bandwidth Extension 

**Title (ZH)**: 基于子奈奎斯特采样与带宽扩展的可实践hearables节能方法 

**Authors**: Tarikul Islam Tamiti, Anomadarshi Barua  

**Link**: [PDF](https://arxiv.org/pdf/2506.22321)  

**Abstract**: Hearables are wearable computers that are worn on the ear. Bone conduction microphones (BCMs) are used with air conduction microphones (ACMs) in hearables as a supporting modality for multimodal speech enhancement (SE) in noisy conditions. However, existing works don't consider the following practical aspects for low-power implementations on hearables: (i) They do not explore how lowering the sampling frequencies and bit resolutions in analog-to-digital converters (ADCs) of hearables jointly impact low-power processing and multimodal SE in terms of speech quality and intelligibility. (ii) They don't discuss how GAN-like audio quality can be achieved without using actual GAN discriminators. And (iii) They don't process signals from ACMs/BCMs at sub-Nyquist sampling rate because, in their frameworks, they lack a wideband reconstruction methodology from their narrowband parts. We propose SUBARU (\textbf{Sub}-Nyquist \textbf{A}udio \textbf{R}esolution \textbf{U}psampling), which achieves the following: SUBARU (i) intentionally uses sub-Nyquist sampling and low bit resolution in ADCs, achieving a 3.31x reduction in power consumption; (ii) introduces novel multi-scale and multi-period virtual discriminators, which achieve GAN-like audio quality without using GANs' adversarial training; and (iii) achieves streaming operations on mobile platforms and SE in in-the-wild noisy conditions with an inference time of 1.74ms and a memory footprint of less than 13.77MB. 

**Abstract (ZH)**: Hearables中亚尼奎斯特音频分辨率上采样（SUBARU）：低功耗多模态降噪方法 

---
# CoATA: Effective Co-Augmentation of Topology and Attribute for Graph Neural Networks 

**Title (ZH)**: CoATA: 有效的同时增强拓扑结构和属性以增强图神经网络 

**Authors**: Tao Liu, Longlong Lin, Yunfeng Yu, Xi Ou, Youan Zhang, Zhiqiu Ye, Tao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2506.22299)  

**Abstract**: Graph Neural Networks (GNNs) have garnered substantial attention due to their remarkable capability in learning graph representations. However, real-world graphs often exhibit substantial noise and incompleteness, which severely degrades the performance of GNNs. Existing methods typically address this issue through single-dimensional augmentation, focusing either on refining topology structures or perturbing node attributes, thereby overlooking the deeper interplays between the two. To bridge this gap, this paper presents CoATA, a dual-channel GNN framework specifically designed for the Co-Augmentation of Topology and Attribute. Specifically, CoATA first propagates structural signals to enrich and denoise node attributes. Then, it projects the enhanced attribute space into a node-attribute bipartite graph for further refinement or reconstruction of the underlying structure. Subsequently, CoATA introduces contrastive learning, leveraging prototype alignment and consistency constraints, to facilitate mutual corrections between the augmented and original graphs. Finally, extensive experiments on seven benchmark datasets demonstrate that the proposed CoATA outperforms eleven state-of-the-art baseline methods, showcasing its effectiveness in capturing the synergistic relationship between topology and attributes. 

**Abstract (ZH)**: 基于拓扑和属性共增强的双通道图神经网络框架CoATA 

---
# Autonomic Microservice Management via Agentic AI and MAPE-K Integration 

**Title (ZH)**: 自主微服务管理通过代理型AI与MAPE-K集成 

**Authors**: Matteo Esposito, Alexander Bakhtin, Noman Ahmad, Mikel Robredo, Ruoyu Su, Valentina Lenarduzzi, Davide Taibi  

**Link**: [PDF](https://arxiv.org/pdf/2506.22185)  

**Abstract**: While microservices are revolutionizing cloud computing by offering unparalleled scalability and independent deployment, their decentralized nature poses significant security and management challenges that can threaten system stability. We propose a framework based on MAPE-K, which leverages agentic AI, for autonomous anomaly detection and remediation to address the daunting task of highly distributed system management. Our framework offers practical, industry-ready solutions for maintaining robust and secure microservices. Practitioners and researchers can customize the framework to enhance system stability, reduce downtime, and monitor broader system quality attributes such as system performance level, resilience, security, and anomaly management, among others. 

**Abstract (ZH)**: 微服务通过提供无与伦比的可扩展性和独立部署能力正在革新云计算，但其分散的性质也带来了显著的安全和管理挑战，这些挑战可能威胁到系统的稳定性。我们提出了一种基于MAPE-K框架的方法，利用代理型AI实现自主异常检测与修复，以应对高度分布式系统管理这一艰巨任务。该框架提供了实用的、可应用于工业界的解决方案，以维护健壯和安全的微服务。实际应用者和研究者可以定制该框架以增强系统稳定性、减少停机时间，并监控诸如系统性能水平、弹性、安全性和异常管理等更广泛系统的质量属性。 

---
# Learning to Solve Multi-Objective Routing Problems on Multigraphs 

**Title (ZH)**: 学习在多重图上解决多目标路径规划问题 

**Authors**: Filip Rydin, Attila Lischka, Jiaming Wu, Morteza Haghir Chehreghani, Balázs Kulcsár  

**Link**: [PDF](https://arxiv.org/pdf/2506.22095)  

**Abstract**: Learning-based methods for routing have gained significant attention in recent years, both in single-objective and multi-objective contexts. However, the multigraph setting, where multiple paths with distinct attributes can exist between destinations, has largely been overlooked, despite its high practical relevancy. In this paper, we introduce two neural approaches to address multi-objective routing on multigraphs. Our first approach works directly on the multigraph, by autoregressively selecting edges until a tour is completed. On the other hand, our second model first prunes the multigraph into a simple graph and then builds routes. We validate both models experimentally and find that they demonstrate strong performance across a variety of problems, including the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP). 

**Abstract (ZH)**: 基于学习的方法在单目标和多目标路由中获得了广泛关注，但在多图设置下，多个具有不同属性的路径存在于目的地之间这一情况却一直被忽视，尽管这种情况具有极高的实用相关性。本文介绍了两种神经方法来解决多图上的多目标路由问题。我们的第一种方法直接在多图上进行，通过自回归选择边直到完成一条路径。另一方面，我们的第二种模型首先将多图简化为简单图，然后构建路径。我们通过实验验证了这两种模型，并发现它们在旅行商问题(TSP)和容量受限车辆路线问题(CVRP)等多种问题上表现出色。 

---
# Transformers are Graph Neural Networks 

**Title (ZH)**: Transformer是图神经网络 

**Authors**: Chaitanya K. Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.22084)  

**Abstract**: We establish connections between the Transformer architecture, originally introduced for natural language processing, and Graph Neural Networks (GNNs) for representation learning on graphs. We show how Transformers can be viewed as message passing GNNs operating on fully connected graphs of tokens, where the self-attention mechanism capture the relative importance of all tokens w.r.t. each-other, and positional encodings provide hints about sequential ordering or structure. Thus, Transformers are expressive set processing networks that learn relationships among input elements without being constrained by apriori graphs. Despite this mathematical connection to GNNs, Transformers are implemented via dense matrix operations that are significantly more efficient on modern hardware than sparse message passing. This leads to the perspective that Transformers are GNNs currently winning the hardware lottery. 

**Abstract (ZH)**: 我们建立了 Transformer 架构与图神经网络（GNNs）之间的联系，前者最初用于自然语言处理，后者用于图上的表示学习。我们展示了 Transformer 可以被视作一种在全连接的令牌图上操作的信息传递 GNN，其中自注意力机制捕获了各令牌之间的相对重要性，而位置编码则提供了关于顺序排序或结构的提示。因此，Transformer 是一种能够学习输入元素之间关系的表达性集处理网络，而不受先验图的约束。尽管 Transformer 在数学上与 GNN 有联系，但它们是通过密集矩阵操作实现的，这在现代硬件上比稀疏信息传递更为高效。这使得 Transformer 当前在硬件竞赛中获胜的观点更为合理。 

---
# UniCA: Adapting Time Series Foundation Model to General Covariate-Aware Forecasting 

**Title (ZH)**: UniCA: 适应时空特征aware预测的时序基础模型 

**Authors**: Lu Han, Yu Liu, Qiwen Deng, Jian Jiang, Yinbo Sun, Zhe Yu, Binfeng Wang, Xingyu Lu, Lintao Ma, Han-Jia Ye, De-Chuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.22039)  

**Abstract**: Time Series Foundation Models (TSFMs) have achieved remarkable success through large-scale pretraining. However, their design primarily targets real-valued series, limiting their ability to handle general forecasting tasks involving diverse and often heterogeneous covariates--such as categorical variables and multimodal data (e.g., images, text)--which are typically task-specific and difficult to leverage during pretraining. To address this gap, we propose Unified Covariate Adaptation (UniCA), a framework to bridge TSFMs with general covariate-aware forecasting. UniCA first performs covariate homogenization to transform heterogeneous covariates into high-level homogeneous series representations and then fuses them via a unified attention-based fusion mechanism. UniCA is compatible and universal for adaptation with both homogeneous and heterogeneous covariates, incorporating extra covariate information while preserving the generalization ability of this http URL experiments on multiple unimodal and multimodal covariate-aware forecasting benchmarks demonstrate the superiority of UniCA, highlighting the promise of covariate-aware TSFM adaptation in real-world forecasting scenarios. Codes are released on this https URL. 

**Abstract (ZH)**: 时间序列基础模型（TSFMs）通过大规模预训练取得了显著成功，但其设计主要针对实值序列，限制了其处理涉及多种通常任务特定且难以在预训练期间利用的异质协变量（如类别变量和多模态数据，例如图像和文本）的一般预测任务的能力。为解决这一问题，我们提出统一协变量适应（UniCA）框架，以将TSFMs与一般协变量感知预测任务结合起来。UniCA 首先执行协变量同质化，将异质协变量转换为高层次的同质序列表示，然后通过统一的基于注意力的融合机制将它们融合在一起。UniCA 兼容且适用于同质和异质协变量的适应，并结合额外的协变量信息同时保留 TSFM 的泛化能力。在多个单模态和多模态协变量感知预测基准上的实验表明，UniCA 的优越性，突显了在实际预测场景中协变量感知 TSFM 调整的潜力。代码发布在这一 https://github.com/UniCA-Team/UniCA。 

---
# TROFI: Trajectory-Ranked Offline Inverse Reinforcement Learning 

**Title (ZH)**: TROFI: 轨迹排名离线逆强化学习 

**Authors**: Alessandro Sestini, Joakim Bergdahl, Konrad Tollmar, Andrew D. Bagdanov, Linus Gisslén  

**Link**: [PDF](https://arxiv.org/pdf/2506.22008)  

**Abstract**: In offline reinforcement learning, agents are trained using only a fixed set of stored transitions derived from a source policy. However, this requires that the dataset be labeled by a reward function. In applied settings such as video game development, the availability of the reward function is not always guaranteed. This paper proposes Trajectory-Ranked OFfline Inverse reinforcement learning (TROFI), a novel approach to effectively learn a policy offline without a pre-defined reward function. TROFI first learns a reward function from human preferences, which it then uses to label the original dataset making it usable for training the policy. In contrast to other approaches, our method does not require optimal trajectories. Through experiments on the D4RL benchmark we demonstrate that TROFI consistently outperforms baselines and performs comparably to using the ground truth reward to learn policies. Additionally, we validate the efficacy of our method in a 3D game environment. Our studies of the reward model highlight the importance of the reward function in this setting: we show that to ensure the alignment of a value function to the actual future discounted reward, it is fundamental to have a well-engineered and easy-to-learn reward function. 

**Abstract (ZH)**: 离线强化学习中，代理使用来自源策略的固定存储转换集进行训练。然而，这需要数据集通过奖励函数进行标记。在视频游戏开发等实际应用中，奖励函数的可用性并不总是有保证的。本文提出了一种名为轨迹排名离线逆强化学习（TROFI）的新方法，以有效学习一个离线策略而无需预定义的奖励函数。TROFI 首先从人类偏好中学习一个奖励函数，然后使用该奖励函数标记原始数据集，使其可用于训练策略。与其它方法不同，我们的方法不需要最优轨迹。通过在 D4RL 标准测试上的实验表明，TROFI 一致地优于基线方法，并且在使用真实奖励学习策略方面表现相近。此外，我们在一个 3D 游戏环境中验证了我们方法的有效性。我们对奖励模型的研究突显了该设置中奖励函数的重要性：我们展示了为了确保价值函数与实际未来折现奖励的对齐，需要一个精心设计且易于学习的奖励函数。 

---
# Binned semiparametric Bayesian networks 

**Title (ZH)**: 分bin的半参数贝叶斯网络 

**Authors**: Rafael Sojo, Javier Díaz-Rozo, Concha Bielza, Pedro Larrañaga  

**Link**: [PDF](https://arxiv.org/pdf/2506.21997)  

**Abstract**: This paper introduces a new type of probabilistic semiparametric model that takes advantage of data binning to reduce the computational cost of kernel density estimation in nonparametric distributions. Two new conditional probability distributions are developed for the new binned semiparametric Bayesian networks, the sparse binned kernel density estimation and the Fourier kernel density estimation. These two probability distributions address the curse of dimensionality, which typically impacts binned models, by using sparse tensors and restricting the number of parent nodes in conditional probability calculations. To evaluate the proposal, we perform a complexity analysis and conduct several comparative experiments using synthetic data and datasets from the UCI Machine Learning repository. The experiments include different binning rules, parent restrictions, grid sizes, and number of instances to get a holistic view of the model's behavior. As a result, our binned semiparametric Bayesian networks achieve structural learning and log-likelihood estimations with no statistically significant differences compared to the semiparametric Bayesian networks, but at a much higher speed. Thus, the new binned semiparametric Bayesian networks prove to be a reliable and more efficient alternative to their non-binned counterparts. 

**Abstract (ZH)**: 一种基于数据分箱的新型半参数概率模型及其在核密度估计中的应用：稀疏分箱核密度估计和傅里叶核密度估计 

---
# Analyzing and Fine-Tuning Whisper Models for Multilingual Pilot Speech Transcription in the Cockpit 

**Title (ZH)**: 分析并微调.Whisper模型以进行驾驶舱多语言试飞语音转录 

**Authors**: Kartheek Kumar Reddy Nareddy, Sarah Ternus, Julia Niebling  

**Link**: [PDF](https://arxiv.org/pdf/2506.21990)  

**Abstract**: The developments in transformer encoder-decoder architectures have led to significant breakthroughs in machine translation, Automatic Speech Recognition (ASR), and instruction-based chat machines, among other applications. The pre-trained models were trained on vast amounts of generic data over a few epochs (fewer than five in most cases), resulting in their strong generalization capabilities. Nevertheless, the performance of these models does suffer when applied to niche domains like transcribing pilot speech in the cockpit, which involves a lot of specific vocabulary and multilingual conversations. This paper investigates and improves the transcription accuracy of cockpit conversations with Whisper models. We have collected around 85 minutes of cockpit simulator recordings and 130 minutes of interview recordings with pilots and manually labeled them. The speakers are middle aged men speaking both German and English. To improve the accuracy of transcriptions, we propose multiple normalization schemes to refine the transcripts and improve Word Error Rate (WER). We then employ fine-tuning to enhance ASR performance, utilizing performance-efficient fine-tuning with Low-Rank Adaptation (LoRA). Hereby, WER decreased from 68.49 \% (pretrained whisper Large model without normalization baseline) to 26.26\% (finetuned whisper Large model with the proposed normalization scheme). 

**Abstract (ZH)**: 变压器编码-解码架构的发展在机器翻译、自动语音识别（ASR）和基于指令的聊天机器等领域取得了显著突破。预训练模型在少量数据（通常少于五轮）上进行大规模通用数据训练，具备较强的泛化能力。然而，当应用于如机舱飞行员对话等专业领域时，这些模型的表现会有所下降，因为这些领域涉及大量的专用词汇和多语言对话。本文针对 Whisper 模型，研究并提高了机舱对话的转录准确性。我们收集了约 85 分钟的机舱模拟器录音和 130 分钟的飞行员访谈录音，并对其进行手动标注。讲话者为中年男子，使用德语和英语。为提高转录准确性，我们提出了多种标准化方案以优化转录并降低词错误率（WER）。随后，我们采用 Fine-tuning 并利用低秩适应（LoRA）进行性能优化。结果表明，与未进行标准化预训练 Whisper 大型模型相比，采用提出的标准化方案 fine-tuned Whisper 大型模型的 WER 从 68.49% 降低到了 26.26%。 

---
# SPADE: Spatial Transcriptomics and Pathology Alignment Using a Mixture of Data Experts for an Expressive Latent Space 

**Title (ZH)**: SPADE: 空间转录组学和病理学对齐的混合数据专家方法用于具表现力的潜在空间 

**Authors**: Ekaterina Redekop, Mara Pleasure, Zichen Wang, Kimberly Flores, Anthony Sisk, William Speier, Corey W. Arnold  

**Link**: [PDF](https://arxiv.org/pdf/2506.21857)  

**Abstract**: The rapid growth of digital pathology and advances in self-supervised deep learning have enabled the development of foundational models for various pathology tasks across diverse diseases. While multimodal approaches integrating diverse data sources have emerged, a critical gap remains in the comprehensive integration of whole-slide images (WSIs) with spatial transcriptomics (ST), which is crucial for capturing critical molecular heterogeneity beyond standard hematoxylin & eosin (H&E) staining. We introduce SPADE, a foundation model that integrates histopathology with ST data to guide image representation learning within a unified framework, in effect creating an ST-informed latent space. SPADE leverages a mixture-of-data experts technique, where experts, created via two-stage feature-space clustering, use contrastive learning to learn representations of co-registered WSI patches and gene expression profiles. Pre-trained on the comprehensive HEST-1k dataset, SPADE is evaluated on 14 downstream tasks, demonstrating significantly superior few-shot performance compared to baseline models, highlighting the benefits of integrating morphological and molecular information into one latent space. 

**Abstract (ZH)**: SPADE：一种结合组织病理学与空间转录组学的数据专家混合基础模型 

---
# 3Description: An Intuitive Human-AI Collaborative 3D Modeling Approach 

**Title (ZH)**: 3Description: 一种直观的人机协作3D建模方法 

**Authors**: Zhuodi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21845)  

**Abstract**: This paper presents 3Description, an experimental human-AI collaborative approach for intuitive 3D modeling. 3Description aims to address accessibility and usability challenges in traditional 3D modeling by enabling non-professional individuals to co-create 3D models using verbal and gesture descriptions. Through a combination of qualitative research, product analysis, and user testing, 3Description integrates AI technologies such as Natural Language Processing and Computer Vision, powered by OpenAI and MediaPipe. Recognizing the web has wide cross-platform capabilities, 3Description is web-based, allowing users to describe the desired model and subsequently adjust its components using verbal and gestural inputs. In the era of AI and emerging media, 3Description not only contributes to a more inclusive and user-friendly design process, empowering more people to participate in the construction of the future 3D world, but also strives to increase human engagement in co-creation with AI, thereby avoiding undue surrender to technology and preserving human creativity. 

**Abstract (ZH)**: 本文介绍了3Description，一种实验性的交互式人机协作方法，用于直观的3D建模。3Description通过使非专业人员能够使用语言和手势描述共同创建3D模型，旨在解决传统3D建模的可访问性和易用性挑战。通过定性研究、产品分析和用户测试，3Description结合了自然语言处理和计算机视觉等人工智能技术，借助OpenAI和MediaPipe的力量。认识到网络具有广泛的跨平台能力，3Description是基于网络的，允许用户描述所需的模型，并通过语音和手势输入调整其组件。在人工智能和新兴媒体的时代，3Description不仅促进了更具包容性和用户友好的设计过程，使更多人能够参与未来3D世界的建设，而且努力增加人与人工智能共同创作的参与度，从而避免过度屈服于技术并保留人类的创造力。 

---
# PARSI: Persian Authorship Recognition via Stylometric Integration 

**Title (ZH)**: PARSI：基于 stylistic 整合的波斯语作者识别 

**Authors**: Kourosh Shahnazari, Mohammadali Keshtparvar, Seyed Moein Ayyoubzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.21840)  

**Abstract**: The intricate linguistic, stylistic, and metrical aspects of Persian classical poetry pose a challenge for computational authorship attribution. In this work, we present a versatile framework to determine authorship among 67 prominent poets. We employ a multi-input neural framework consisting of a transformer-based language encoder complemented by features addressing the semantic, stylometric, and metrical dimensions of Persian poetry. Our feature set encompasses 100-dimensional Word2Vec embeddings, seven stylometric measures, and categorical encodings of poetic form and meter. We compiled a vast corpus of 647,653 verses of the Ganjoor digital collection, validating the data through strict preprocessing and author verification while preserving poem-level splitting to prevent overlap. This work employs verse-level classification and majority and weighted voting schemes in evaluation, revealing that weighted voting yields 71% accuracy. We further investigate threshold-based decision filtering, allowing the model to generate highly confident predictions, achieving 97% accuracy at a 0.9 threshold, though at lower coverage. Our work focuses on the integration of deep representational forms with domain-specific features for improved authorship attribution. The results illustrate the potential of our approach for automated classification and the contribution to stylistic analysis, authorship disputes, and general computational literature research. This research will facilitate further research on multilingual author attribution, style shift, and generative modeling of Persian poetry. 

**Abstract (ZH)**: 波斯古典诗歌复杂的语言、风格和韵律方面对计算作者归属构成挑战。本文提出了一种灵活的框架，用于确定67位著名诗人的作者身份。我们采用了一种多输入神经框架，该框架由基于变换器的语言编码器和处理波斯诗歌语义、风格统计和韵律维度的特征组成。我们的特征集包括100维Word2Vec嵌入、七种风格测量和诗体和韵律的分类编码。我们编制了一个包含647,653行的庞大语料库，通过严格的预处理和作者验证来验证数据，并保留诗歌级别的划分以防止重叠。本文在诗歌级别分类和多数投票及加权投票方案的评估中发现，加权投票的准确率为71%。我们进一步探讨了基于阈值的决策筛选，使模型能够生成高度自信的预测，并在0.9阈值下达到97%的准确率，尽管覆盖率较低。本文专注于将深度表示形式与领域特定特征的结合，以提高作者归属的准确性。结果表明，我们的方法在自动化分类和风格分析、作者身份争议以及一般计算文学研究方面的潜力，并将促进多语言作者归属、风格转变和波斯诗歌生成模型的研究。 

---
# SciMantify -- A Hybrid Approach for the Evolving Semantification of Scientific Knowledge 

**Title (ZH)**: SciMantify -- 一种动态科学知识语义化的混合方法 

**Authors**: Lena John, Kheir Eddine Farfar, Sören Auer, Oliver Karras  

**Link**: [PDF](https://arxiv.org/pdf/2506.21819)  

**Abstract**: Scientific publications, primarily digitized as PDFs, remain static and unstructured, limiting the accessibility and reusability of the contained knowledge. At best, scientific knowledge from publications is provided in tabular formats, which lack semantic context. A more flexible, structured, and semantic representation is needed to make scientific knowledge understandable and processable by both humans and machines. We propose an evolution model of knowledge representation, inspired by the 5-star Linked Open Data (LOD) model, with five stages and defined criteria to guide the stepwise transition from a digital artifact, such as a PDF, to a semantic representation integrated in a knowledge graph (KG). Based on an exemplary workflow implementing the entire model, we developed a hybrid approach, called SciMantify, leveraging tabular formats of scientific knowledge, e.g., results from secondary studies, to support its evolving semantification. In the approach, humans and machines collaborate closely by performing semantic annotation tasks (SATs) and refining the results to progressively improve the semantic representation of scientific knowledge. We implemented the approach in the Open Research Knowledge Graph (ORKG), an established platform for improving the findability, accessibility, interoperability, and reusability of scientific knowledge. A preliminary user experiment showed that the approach simplifies the preprocessing of scientific knowledge, reduces the effort for the evolving semantification, and enhances the knowledge representation through better alignment with the KG structures. 

**Abstract (ZH)**: 科学出版物主要以PDF形式数字化，保持静态且无结构，限制了其中知识的可访问性和再利用性。科学出版物中的知识通常以表格形式提供，缺乏语义上下文。需要一种更灵活、结构化和语义化的表示形式，以便人类和机器能够理解和处理这些知识。我们提出了一个受5星Linked Open Data (LOD)模型启发的知识表示演化模型，包含五个阶段和指导准则，引导从PDF等数字文件逐步过渡到集成在知识图谱(KG)中的语义表示。基于整个模型的示例工作流，我们开发了一种混合方法，称为SciMantify，利用科学知识的表格格式，如二次研究的结果，以支持其逐步语义化。在该方法中，人类和机器密切合作，执行语义标注任务(SATs)，并不断改进科学知识的语义表示。我们将在Open Research Knowledge Graph (ORKG)平台上实现该方法，这是一个提高科学知识可查找性、可访问性、互操作性和再利用性的已建立平台。初步用户实验表明，该方法简化了科学知识的预处理，减少了逐步语义化的工作量，并通过更好的与KG结构对齐提高了知识表示。 

---
# CAT-SG: A Large Dynamic Scene Graph Dataset for Fine-Grained Understanding of Cataract Surgery 

**Title (ZH)**: CAT-SG：用于白内障手术细粒度理解的大规模动态场景图数据集 

**Authors**: Felix Holm, Gözde Ünver, Ghazal Ghazaei, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2506.21813)  

**Abstract**: Understanding the intricate workflows of cataract surgery requires modeling complex interactions between surgical tools, anatomical structures, and procedural techniques. Existing datasets primarily address isolated aspects of surgical analysis, such as tool detection or phase segmentation, but lack comprehensive representations that capture the semantic relationships between entities over time. This paper introduces the Cataract Surgery Scene Graph (CAT-SG) dataset, the first to provide structured annotations of tool-tissue interactions, procedural variations, and temporal dependencies. By incorporating detailed semantic relations, CAT-SG offers a holistic view of surgical workflows, enabling more accurate recognition of surgical phases and techniques. Additionally, we present a novel scene graph generation model, CatSGG, which outperforms current methods in generating structured surgical representations. The CAT-SG dataset is designed to enhance AI-driven surgical training, real-time decision support, and workflow analysis, paving the way for more intelligent, context-aware systems in clinical practice. 

**Abstract (ZH)**: 理解白内障手术的复杂工作流程需要建模手术工具、解剖结构和手术技巧之间的复杂交互。现有的数据集主要关注手术分析的孤立方面，如工具检测或阶段分割，但缺乏能够捕捉实体间时空语义关系的全面表示。本文介绍了CAT-SG数据集，这是首个提供工具-组织交互、手术变体和时间依赖性的结构化注释的数据集。通过整合详细的语义关系，CAT-SG提供了对手术工作流程的全面视图，有助于更准确地识别手术阶段和技巧。此外，我们还提出了一种新的场景图生成模型CatSGG，其在生成结构化手术表示方面优于现有方法。CAT-SG数据集旨在增强基于AI的手术培训、实时决策支持和工作流程分析，为临床实践中更智能、更具上下文感知的系统铺平道路。 

---
# From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining 

**Title (ZH)**: 从Token到节奏：一种多尺度ECG语言预训练方法 

**Authors**: Fuying Wang, Jiacheng Xu, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21803)  

**Abstract**: Electrocardiograms (ECGs) play a vital role in monitoring cardiac health and diagnosing heart diseases. However, traditional deep learning approaches for ECG analysis rely heavily on large-scale manual annotations, which are both time-consuming and resource-intensive to obtain. To overcome this limitation, self-supervised learning (SSL) has emerged as a promising alternative, enabling the extraction of robust ECG representations that can be efficiently transferred to various downstream tasks. While previous studies have explored SSL for ECG pretraining and multi-modal ECG-language alignment, they often fail to capture the multi-scale nature of ECG signals. As a result, these methods struggle to learn generalized representations due to their inability to model the hierarchical structure of ECG data. To address this gap, we introduce MELP, a novel Multi-scale ECG-Language Pretraining (MELP) model that fully leverages hierarchical supervision from ECG-text pairs. MELP first pretrains a cardiology-specific language model to enhance its understanding of clinical text. It then applies three levels of cross-modal supervision-at the token, beat, and rhythm levels-to align ECG signals with textual reports, capturing structured information across different time scales. We evaluate MELP on three public ECG datasets across multiple tasks, including zero-shot ECG classification, linear probing, and transfer learning. Experimental results demonstrate that MELP outperforms existing SSL methods, underscoring its effectiveness and adaptability across diverse clinical applications. Our code is available at this https URL. 

**Abstract (ZH)**: 多尺度心电图-语言预训练模型（MELP） 

---
# Demonstrating Interoperable Channel State Feedback Compression with Machine Learning 

**Title (ZH)**: 基于机器学习的可互操作信道状态反馈压缩演示 

**Authors**: Dani Korpi, Rachel Wang, Jerry Wang, Abdelrahman Ibrahim, Carl Nuzman, Runxin Wang, Kursat Rasim Mestav, Dustin Zhang, Iraj Saniee, Shawn Winston, Gordana Pavlovic, Wei Ding, William J. Hillery, Chenxi Hao, Ram Thirunagari, Jung Chang, Jeehyun Kim, Bartek Kozicki, Dragan Samardzija, Taesang Yoo, Andreas Maeder, Tingfang Ji, Harish Viswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21796)  

**Abstract**: Neural network-based compression and decompression of channel state feedback has been one of the most widely studied applications of machine learning (ML) in wireless networks. Various simulation-based studies have shown that ML-based feedback compression can result in reduced overhead and more accurate channel information. However, to the best of our knowledge, there are no real-life proofs of concepts demonstrating the benefits of ML-based channel feedback compression in a practical setting, where the user equipment (UE) and base station have no access to each others' ML models. In this paper, we present a novel approach for training interoperable compression and decompression ML models in a confidential manner, and demonstrate the accuracy of the ensuing models using prototype UEs and base stations. The performance of the ML-based channel feedback is measured both in terms of the accuracy of the reconstructed channel information and achieved downlink throughput gains when using the channel information for beamforming. The reported measurement results demonstrate that it is possible to develop an accurate ML-based channel feedback link without having to share ML models between device and network vendors. These results pave the way for a practical implementation of ML-based channel feedback in commercial 6G networks. 

**Abstract (ZH)**: 基于神经网络的信道状态反馈压缩与解压缩一直是无线网络中机器学习（ML）应用中最广泛研究的领域之一。各种基于仿真的研究表明，基于ML的反馈压缩可以减少开销并提高信道信息的准确性。然而，据我们所知，在用户设备（UE）和基站之间没有访问对方ML模型的情况下，没有实际的概念验证展示ML在信道反馈压缩中的益处。在本文中，我们提出了一种新颖的方法，以保密的方式训练互操作的压缩和解压缩ML模型，并使用原型UE和基站展示了这些模型的准确性。通过使用信道信息进行波束形成，评估基于ML的信道反馈性能，从重建的信道信息和下行传输速率增益两方面衡量。报告的测量结果表明，可以在设备和网络供应商之间不共享ML模型的情况下开发出准确的基于ML的信道反馈链路。这些结果为在商用6G网络中实现基于ML的信道反馈铺平了道路。 

---
# Multi-task parallelism for robust pre-training of graph foundation models on multi-source, multi-fidelity atomistic modeling data 

**Title (ZH)**: 多任务并行训练以robust预训练图基础模型于多源、多保真度原子级建模数据上 

**Authors**: Massimiliano Lupo Pasini, Jong Youl Choi, Pei Zhang, Kshitij Mehta, Rylie Weaver, Ashwin M. Aji, Karl W. Schulz, Jorda Polo, Prasanna Balaprakash  

**Link**: [PDF](https://arxiv.org/pdf/2506.21788)  

**Abstract**: Graph foundation models using graph neural networks promise sustainable, efficient atomistic modeling. To tackle challenges of processing multi-source, multi-fidelity data during pre-training, recent studies employ multi-task learning, in which shared message passing layers initially process input atomistic structures regardless of source, then route them to multiple decoding heads that predict data-specific outputs. This approach stabilizes pre-training and enhances a model's transferability to unexplored chemical regions. Preliminary results on approximately four million structures are encouraging, yet questions remain about generalizability to larger, more diverse datasets and scalability on supercomputers. We propose a multi-task parallelism method that distributes each head across computing resources with GPU acceleration. Implemented in the open-source HydraGNN architecture, our method was trained on over 24 million structures from five datasets and tested on the Perlmutter, Aurora, and Frontier supercomputers, demonstrating efficient scaling on all three highly heterogeneous super-computing architectures. 

**Abstract (ZH)**: 基于图神经网络的图基础模型有望实现原子级建模的可持续性和效率。为应对预训练期间处理多源、多保真数据的挑战，近期研究采用多任务学习，其中共享的消息传递层最初不考虑数据来源地处理输入的原子结构，然后将它们路由到多个解码头以预测特定于数据的输出。这种方法稳定了预训练并增强了模型在未探索化学区域的泛化能力。初步结果令人鼓舞，但仍然存在关于在更大、更多样化数据集上的泛化能力和在超级计算机上的扩展性的疑问。我们提出了一种多任务并行方法，将每个解码头分布在带有GPU加速的计算资源上。该方法在开源HydraGNN架构中实现，并在五个数据集超过2400万结构上进行了训练，随后在Perlmutter、Aurora和Frontier超级计算机上进行测试，展示了在所有三个高度异构超级计算机架构上的高效扩展能力。 

---
# Simultaneously Fair Allocation of Indivisible Items Across Multiple Dimensions 

**Title (ZH)**: 多维度下的非可分物品的公平分配 

**Authors**: Yasushi Kawase, Bodhayan Roy, Mohammad Azharuddin Sanpui  

**Link**: [PDF](https://arxiv.org/pdf/2506.21727)  

**Abstract**: This paper explores the fair allocation of indivisible items in a multidimensional setting, motivated by the need to address fairness in complex environments where agents assess bundles according to multiple criteria. Such multidimensional settings are not merely of theoretical interest but are central to many real-world applications. For example, cloud computing resources are evaluated based on multiple criteria such as CPU cores, memory, and network bandwidth. In such cases, traditional one dimensional fairness notions fail to capture fairness across multiple attributes. To address these challenges, we study two relaxed variants of envy-freeness: weak simultaneously envy-free up to c goods (weak sEFc) and strong simultaneously envy-free up to c goods (strong sEFc), which accommodate the multidimensionality of agents' preferences. Under the weak notion, for every pair of agents and for each dimension, any perceived envy can be eliminated by removing, if necessary, a different set of goods from the envied agent's allocation. In contrast, the strong version requires selecting a single set of goods whose removal from the envied bundle simultaneously eliminates envy in every dimension. We provide upper and lower bounds on the relaxation parameter c that guarantee the existence of weak or strong sEFc allocations, where these bounds are independent of the total number of items. In addition, we present algorithms for checking whether a weak or strong sEFc allocation exists. Moreover, we establish NP-hardness results for checking the existence of weak sEF1 and strong sEF1 allocations. 

**Abstract (ZH)**: 基于多维度设置下非可分物品的公平分配研究 

---
# Performance Prediction for Large Systems via Text-to-Text Regression 

**Title (ZH)**: 基于文本到文本回归的大系统性能预测 

**Authors**: Yash Akhauri, Bryan Lewandowski, Cheng-Hsi Lin, Adrian N. Reyes, Grant C. Forbes, Arissa Wongpanich, Bangding Yang, Mohamed S. Abdelfattah, Sagi Perel, Xingyou Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.21718)  

**Abstract**: In many industries, predicting metric outcomes of large systems is a fundamental problem, driven largely by traditional tabular regression. However, such methods struggle on complex systems data in the wild such as configuration files or system logs, where feature engineering is often infeasible. We propose text-to-text regression as a general, scalable alternative. For predicting resource efficiency on Borg, Google's massive compute cluster scheduling system, a 60M parameter encoder-decoder, trained from random initialization, achieves up to a near perfect 0.99 (0.9 average) rank correlation across the entire fleet, and 100x lower MSE than tabular approaches. The model also easily adapts to new tasks in only 500 few-shot examples and captures the densities of complex outcome distributions. Ablation studies highlight the importance of using encoders, increasing sequence length, and the model's inherent uncertainty quantification. These findings pave the way for universal simulators of real-world outcomes. 

**Abstract (ZH)**: 在许多行业中，预测大型系统的指标结果是一个基本问题，传统上主要依赖表格回归方法。然而，这类方法在复杂的系统数据（如配置文件或系统日志）上常常表现不佳，这些数据中的特征工程往往是不可行的。我们提出文本到文本回归作为一种通用且可扩展的替代方案。对于预测Google大规模计算集群调度系统Borg的资源效率，一个60M参数的编码器-解码器模型从随机初始化训练，实现了整个集群近乎完美的0.99（平均0.9）排名相关性，并且平均均方误差比表格方法低100倍。该模型还能够在仅500个少样本示例中轻松适应新任务，并捕捉复杂结果分布的密度。消融研究强调了使用编码器、增加序列长度以及模型固有的不确定性量化的重要性。这些发现为现实世界结果的通用模拟器铺平了道路。 

---
# Doc2SAR: A Synergistic Framework for High-Fidelity Extraction of Structure-Activity Relationships from Scientific Documents 

**Title (ZH)**: Doc2SAR：一种用于从科学文献中高保真提取结构-活性关系的协同框架 

**Authors**: Jiaxi Zhuang, Kangning Li, Jue Hou, Mingjun Xu, Zhifeng Gao, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21625)  

**Abstract**: Extracting molecular structure-activity relationships (SARs) from scientific literature and patents is essential for drug discovery and materials research. However, this task remains challenging due to heterogeneous document formats and limitations of existing methods. Specifically, rule-based approaches relying on rigid templates fail to generalize across diverse document layouts, while general-purpose multimodal large language models (MLLMs) lack sufficient accuracy and reliability for specialized tasks, such as layout detection and optical chemical structure recognition (OCSR). To address these challenges, we introduce DocSAR-200, a rigorously annotated benchmark of 200 scientific documents designed specifically for evaluating SAR extraction methods. Additionally, we propose Doc2SAR, a novel synergistic framework that integrates domain-specific tools with MLLMs enhanced via supervised fine-tuning (SFT). Extensive experiments demonstrate that Doc2SAR achieves state-of-the-art performance across various document types, significantly outperforming leading end-to-end baselines. Specifically, Doc2SAR attains an overall Table Recall of 80.78% on DocSAR-200, exceeding end2end GPT-4o by 51.48%. Furthermore, Doc2SAR demonstrates practical usability through efficient inference and is accompanied by a web app. 

**Abstract (ZH)**: 从科学文献和专利中提取分子结构-活性关系（SARs）对于药物发现和材料研究至关重要。然而，由于文献格式的异质性和现有方法的局限性，这一任务仍然具有挑战性。具体来说，依赖于固定模板的基于规则的方法无法适应多样的文档布局，而通用的多模态大语言模型（MLLMs）在专门任务，如布局检测和光学化学结构识别（OCSR）方面缺乏足够的准确性和可靠性。为应对这一挑战，我们引入了DocSAR-200，这是一个专门用于评估SAR提取方法的严格标注基准，包含200份科学文档。此外，我们提出了一种新颖的协同框架Doc2SAR，将领域特定工具与通过监督微调（SFT）增强的大语言模型集成。广泛的实验表明，Doc2SAR 在各种类型的文档中达到了最先进的性能，显著优于领先的一体化基线。具体而言，Doc2SAR 在DocSAR-200上的总体 Table Recall 达到了 80.78%，超过端到端 GPT-4o 51.48%。此外，Doc2SAR 通过高效的推理展示了其实用性，并附带了一个网页应用。 

---
# IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech 

**Title (ZH)**: IndexTTS2：在情绪表达和时长控制方面的一项突破性自动回归零样本文本到语音技术 

**Authors**: Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21619)  

**Abstract**: Large-scale text-to-speech (TTS) models are typically categorized into autoregressive and non-autoregressive systems. Although autoregressive systems exhibit certain advantages in speech naturalness, their token-by-token generation mechanism makes it difficult to precisely control the duration of synthesized speech. This is a key limitation in applications such as video dubbing that require strict audio-visual synchronization. This paper introduces IndexTTS2, which proposes a novel and autoregressive-model-friendly method for speech duration control. The method supports two generation modes: one allows explicit specification of the number of generated tokens for precise duration control; the other does not require manual input and lets the model freely generate speech while preserving prosodic characteristics from the input prompt. Furthermore, IndexTTS2 achieves disentanglement between emotional expression and speaker identity, enabling independent control of timbre and emotion. In the zero-shot setting, the model can perfectly reproduce the emotional characteristics of the input prompt. Users may also provide a separate emotion prompt, even from a different speaker, allowing the model to reconstruct the target timbre while conveying the desired emotion. To enhance clarity during strong emotional expressions, we incorporate GPT latent representations to improve speech stability. Meanwhile, to lower the barrier for emotion control, we design a soft instruction mechanism based on textual descriptions by fine-tuning Qwen3. This enables effective guidance of speech generation with desired emotional tendencies using natural language input. Experimental results demonstrate that IndexTTS2 outperforms existing state-of-the-art zero-shot TTS models in word error rate, speaker similarity, and emotional fidelity. 

**Abstract (ZH)**: IndexTTS2：一种适合自回归模型的新型语音时长控制方法 

---
# TrajTok: Technical Report for 2025 Waymo Open Sim Agents Challenge 

**Title (ZH)**: TrajTok: 2025 Waymo 开放仿真代理挑战赛技术报告 

**Authors**: Zhiyuan Zhang, Xiaosong Jia, Guanyu Chen, Qifeng Li, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21618)  

**Abstract**: In this technical report, we introduce TrajTok, a trajectory tokenizer for discrete next-token-prediction based behavior generation models, which combines data-driven and rule-based methods with better coverage, symmetry and robustness, along with a spatial-aware label smoothing method for cross-entropy loss. We adopt the tokenizer and loss for the SMART model and reach a superior performance with realism score of 0.7852 on the Waymo Open Sim Agents Challenge 2025. We will open-source the code in the future. 

**Abstract (ZH)**: 本技术报告介绍了TrajTok，这是一种结合数据驱动和规则驱动方法的轨迹分词器，适用于离散的下一标记预测基于行为生成模型，同时提出了具有更好覆盖范围、对称性和鲁棒性的空间感知标签平滑方法以优化交叉熵损失。我们在Waymo Open Sim Agents Challenge 2025中使用TrajToktokenizer和损失函数实现了0.7852的现实度得分，并取得了优越的性能。未来我们将开源代码。 

---
# Bayesian-Guided Diversity in Sequential Sampling for Recommender Systems 

**Title (ZH)**: 基于贝叶斯指导的序贯采样多样性在推荐系统中的应用 

**Authors**: Hiba Bederina, Jill-Jênn Vie  

**Link**: [PDF](https://arxiv.org/pdf/2506.21617)  

**Abstract**: The challenge of balancing user relevance and content diversity in recommender systems is increasingly critical amid growing concerns about content homogeneity and reduced user engagement. In this work, we propose a novel framework that leverages a multi-objective, contextual sequential sampling strategy. Item selection is guided by Bayesian updates that dynamically adjust scores to optimize diversity. The reward formulation integrates multiple diversity metrics-including the log-determinant volume of a tuned similarity submatrix and ridge leverage scores-along with a diversity gain uncertainty term to address the exploration-exploitation trade-off. Both intra- and inter-batch diversity are modeled to promote serendipity and minimize redundancy. A dominance-based ranking procedure identifies Pareto-optimal item sets, enabling adaptive and balanced selections at each iteration. Experiments on a real-world dataset show that our approach significantly improves diversity without sacrificing relevance, demonstrating its potential to enhance user experience in large-scale recommendation settings. 

**Abstract (ZH)**: 在 growing concerns about content homogeneity 和 reduced user engagement 的背景下，平衡用户相关性和内容多样性在推荐系统中的挑战日益关键。本文提出了一种新颖的框架，利用多目标上下文顺序采样策略。项选择由贝叶斯更新引导，动态调整分数以优化多样性。奖励形式化整合了包括调优相似性子矩阵的对数行列式体积和岭杠杆得分在内的多种多样性指标，以及多样性增益不确定性项，以解决探索与利用之间的权衡。同时建模 intra- 和 inter-batch 多样性，以促进 serendipity 并减少冗余。基于支配性的排名过程识别 Pareto 最优项集，使每次迭代都能实现适应性和平衡的选择。实验结果表明，该方法在不牺牲相关性的情况下显著提高了多样性，证明其在大规模推荐设置中增强用户体验的潜力。 

---
# Refine Medical Diagnosis Using Generation Augmented Retrieval and Clinical Practice Guidelines 

**Title (ZH)**: 利用生成增强检索和临床实践指南细化医疗诊断 

**Authors**: Wenhao Li, Hongkuan Zhang, Hongwei Zhang, Zhengxu Li, Zengjie Dong, Yafan Chen, Niranjan Bidargaddi, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21615)  

**Abstract**: Current medical language models, adapted from large language models (LLMs), typically predict ICD code-based diagnosis from electronic health records (EHRs) because these labels are readily available. However, ICD codes do not capture the nuanced, context-rich reasoning clinicians use for diagnosis. Clinicians synthesize diverse patient data and reference clinical practice guidelines (CPGs) to make evidence-based decisions. This misalignment limits the clinical utility of existing models. We introduce GARMLE-G, a Generation-Augmented Retrieval framework that grounds medical language model outputs in authoritative CPGs. Unlike conventional Retrieval-Augmented Generation based approaches, GARMLE-G enables hallucination-free outputs by directly retrieving authoritative guideline content without relying on model-generated text. It (1) integrates LLM predictions with EHR data to create semantically rich queries, (2) retrieves relevant CPG knowledge snippets via embedding similarity, and (3) fuses guideline content with model output to generate clinically aligned recommendations. A prototype system for hypertension diagnosis was developed and evaluated on multiple metrics, demonstrating superior retrieval precision, semantic relevance, and clinical guideline adherence compared to RAG-based baselines, while maintaining a lightweight architecture suitable for localized healthcare deployment. This work provides a scalable, low-cost, and hallucination-free method for grounding medical language models in evidence-based clinical practice, with strong potential for broader clinical deployment. 

**Abstract (ZH)**: 当前医疗语言模型通常从大型语言模型（LLMs）改编而来，因为这些模型通常用于从电子健康记录（EHRs）预测ICD代码标签，而这些标签易于获取。然而，ICD代码无法捕捉到临床医生在诊断时使用的细微且富含上下文的推理过程。临床医生综合多种患者数据并参考临床实践指南（CPGs）来做出基于证据的决策。这种不一致限制了现有模型的临床应用价值。我们引入了GARMLE-G生成增强检索框架，该框架将医疗语言模型的输出与权威的CPGs相结合。不同于传统的检索增强生成方法，GARMLE-G通过直接检索权威指南内容而不依赖于模型生成的文本，实现零幻觉输出。GARMLE-G实现包括：（1）将LLM预测与EHR数据整合以创建语义丰富的查询，（2）通过嵌入相似性检索相关CPGs知识片段，（3）将指南内容与模型输出融合生成临床对齐的建议。我们开发并评估了一个用于高血压诊断的原型系统，结果显示GARMLE-G在多个指标上优于基于检索增强生成的基线模型，同时保持了轻量级架构，便于局部医疗部署。这项工作提供了一种可扩展、低成本且无幻觉的方法，将医疗语言模型与基于证据的临床实践相结合，具有广泛的临床应用潜力。 

---
# AdaptGOT: A Pre-trained Model for Adaptive Contextual POI Representation Learning 

**Title (ZH)**: AdaptGOT：一种适应性上下文POI表示预训练模型 

**Authors**: Xiaobin Ren, Xinyu Zhu, Kaiqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21612)  

**Abstract**: Currently, considerable strides have been achieved in Point-of-Interest (POI) embedding methodologies, driven by the emergence of novel POI tasks like recommendation and classification. Despite the success of task-specific, end-to-end models in POI embedding, several challenges remain. These include the need for more effective multi-context sampling strategies, insufficient exploration of multiple POI contexts, limited versatility, and inadequate generalization. To address these issues, we propose the AdaptGOT model, which integrates both the (Adapt)ive representation learning technique and the Geographical-Co-Occurrence-Text (GOT) representation with a particular emphasis on Geographical location, Co-Occurrence and Textual information. The AdaptGOT model comprises three key components: (1) contextual neighborhood generation, which integrates advanced mixed sampling techniques such as KNN, density-based, importance-based, and category-aware strategies to capture complex contextual neighborhoods; (2) an advanced GOT representation enhanced by an attention mechanism, designed to derive high-quality, customized representations and efficiently capture complex interrelations between POIs; and (3) the MoE-based adaptive encoder-decoder architecture, which ensures topological consistency and enriches contextual representation by minimizing Jensen-Shannon divergence across varying contexts. Experiments on two real-world datasets and multiple POI tasks substantiate the superior performance of the proposed AdaptGOT model. 

**Abstract (ZH)**: 目前，在兴趣点（POI）嵌入方法方面已经取得了显著进展，这得益于诸如推荐和分类等新型POI任务的出现。尽管针对特定任务的端到端模型在POI嵌入中取得了成功，但仍存在一些挑战，包括更有效的多上下文采样策略需求、多POI上下文探索不足、灵活性有限以及泛化能力不足等问题。为了解决这些问题，我们提出了AdaptGOT模型，该模型结合了自适应表示学习技术和地理共现文本（GOT）表示，并特别强调地理位置、共现和文本信息。AdaptGOT模型包括三个关键组件：(1) 上下文邻域生成，整合了包括KNN、基于密度、基于重要性和类别感知在内的高级混合采样技术，以捕捉复杂的上下文邻域；(2) 通过注意力机制增强的先进GOT表示，旨在获取高质量、定制化的表示，并有效地捕获POI之间的复杂关系；以及(3) 基于MoE的自适应编码器-解码器架构，通过最小化不同上下文中约简皮尔逊距离来确保拓扑一致性和丰富上下文表示。在两个真实世界数据集和多项POI任务上的实验验证了所提AdaptGOT模型的优越性能。 

---
# Does Multimodality Lead to Better Time Series Forecasting? 

**Title (ZH)**: 多模态能否带来更好的时间序列预测？ 

**Authors**: Xiyuan Zhang, Boran Han, Haoyang Fang, Abdul Fatir Ansari, Shuai Zhang, Danielle C. Maddix, Cuixiong Hu, Andrew Gordon Wilson, Michael W. Mahoney, Hao Wang, Yan Liu, Huzefa Rangwala, George Karypis, Bernie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21611)  

**Abstract**: Recently, there has been growing interest in incorporating textual information into foundation models for time series forecasting. However, it remains unclear whether and under what conditions such multimodal integration consistently yields gains. We systematically investigate these questions across a diverse benchmark of 14 forecasting tasks spanning 7 domains, including health, environment, and economics. We evaluate two popular multimodal forecasting paradigms: aligning-based methods, which align time series and text representations; and prompting-based methods, which directly prompt large language models for forecasting. Although prior works report gains from multimodal input, we find these effects are not universal across datasets and models, and multimodal methods sometimes do not outperform the strongest unimodal baselines. To understand when textual information helps, we disentangle the effects of model architectural properties and data characteristics. Our findings highlight that on the modeling side, incorporating text information is most helpful given (1) high-capacity text models, (2) comparatively weaker time series models, and (3) appropriate aligning strategies. On the data side, performance gains are more likely when (4) sufficient training data is available and (5) the text offers complementary predictive signal beyond what is already captured from the time series alone. Our empirical findings offer practical guidelines for when multimodality can be expected to aid forecasting tasks, and when it does not. 

**Abstract (ZH)**: 最近，人们越来越关注在时间序列预测中将文本信息融入基础模型中的方法。然而，尚不清楚这种多模态集成在什么情况下能够一致地带来改进。我们系统地在涵盖7个领域的14个预测任务上进行了研究，这些领域包括健康、环境和经济。我们评估了两种流行的多模态预测范式：基于对齐的方法，将时间序列和文本表示进行对齐；以及基于提示的方法，直接提示大型语言模型进行预测。尽管先前的研究报告了多模态输入带来的增益，但我们发现这些效果在不同数据集和模型上并不是普遍存在的，有时多模态方法也不如最强的单模态基线方法。为了理解何时文本信息有助于预测，我们分离了模型架构属性和数据特性的影响。我们的发现强调，在建模方面，文本信息的引入最有助益的情况是：（1）具有高容量的文本模型，（2）相对较弱的时间序列模型，和（3）适当的对齐策略。在数据方面，当（4）有足够的训练数据可用且（5）文本提供了超越单独时间序列捕捉到的补充预测信号时，性能增益更有可能。我们的实证发现为何时多模态可以预期有助于预测任务，以及何时它不起作用提供了实用指南。 

---
# SysTemp: A Multi-Agent System for Template-Based Generation of SysML v2 

**Title (ZH)**: SysTemp: 基于模板的SysML v2生成的多智能体系统 

**Authors**: Yasmine Bouamra, Bruno Yun, Alexandre Poisson, Frédéric Armetta  

**Link**: [PDF](https://arxiv.org/pdf/2506.21608)  

**Abstract**: The automatic generation of SysML v2 models represents a major challenge in the engineering of complex systems, particularly due to the scarcity of learning corpora and complex syntax. We present SysTemp, a system aimed at facilitating and improving the creation of SysML v2 models from natural language specifications. It is based on a multi-agent system, including a template generator that structures the generation process. We discuss the advantages and challenges of this system through an evaluation, highlighting its potential to improve the quality of the generations in SysML v2 modeling. 

**Abstract (ZH)**: SysTemp：一种用于从自然语言规范生成SysML v2模型的多代理系统 

---
# Hope Speech Detection in code-mixed Roman Urdu tweets: A Positive Turn in Natural Language Processing 

**Title (ZH)**: 代码混合罗马乌尔都语推文中希望演说检测：自然语言处理的积极转捩点 

**Authors**: Muhammad Ahmad, Muhammad Waqas, Ameer Hamza, Ildar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2506.21583)  

**Abstract**: Hope is a positive emotional state involving the expectation of favorable future outcomes, while hope speech refers to communication that promotes optimism, resilience, and support, particularly in adverse contexts. Although hope speech detection has gained attention in Natural Language Processing (NLP), existing research mainly focuses on high-resource languages and standardized scripts, often overlooking informal and underrepresented forms such as Roman Urdu. To the best of our knowledge, this is the first study to address hope speech detection in code-mixed Roman Urdu by introducing a carefully annotated dataset, thereby filling a critical gap in inclusive NLP research for low-resource, informal language varieties. This study makes four key contributions: (1) it introduces the first multi-class annotated dataset for Roman Urdu hope speech, comprising Generalized Hope, Realistic Hope, Unrealistic Hope, and Not Hope categories; (2) it explores the psychological foundations of hope and analyzes its linguistic patterns in code-mixed Roman Urdu to inform dataset development; (3) it proposes a custom attention-based transformer model optimized for the syntactic and semantic variability of Roman Urdu, evaluated using 5-fold cross-validation; and (4) it verifies the statistical significance of performance gains using a t-test. The proposed model, XLM-R, achieves the best performance with a cross-validation score of 0.78, outperforming the baseline SVM (0.75) and BiLSTM (0.76), with gains of 4% and 2.63% respectively. 

**Abstract (ZH)**: 希望状态涉及对未来有利结果的期望，而希望言论是指在不利环境中促进乐观、韧性和支持的沟通。尽管希望言论检测在自然语言处理（NLP）中引起了关注，但现有研究主要集中在资源丰富语言和标准化书写系统上，往往忽略了如混合罗马乌都语等非正式和未充分代表的形式。据我们所知，这是首次通过引入精心标注的数据集来解决混合罗马乌都语希望言论检测问题的研究，从而填补了包容性NLP研究中低资源、非正式语言变体的重要空白。本研究做出了四个关键贡献：（1）首次为罗马乌都语希望言论引入了多类标注数据集，包括普遍希望、现实希望、不切实际希望和非希望类别；（2）探讨了希望的心理基础并分析了混合罗马乌都语中的语言模式，以指导数据集开发；（3）提出了一种针对罗马乌都语句法和语义变异性优化的自注意力变压器模型，并使用5折交叉验证进行评估；（4）使用T检验验证了性能提升的统计显著性。所提出的模型XLM-R在交叉验证得分为0.78的情况下表现出最佳性能，优于基线SVM（0.75）和BiLSTM（0.76），分别提高了4%和2.63%。 

---
# Evaluating the Robustness of Dense Retrievers in Interdisciplinary Domains 

**Title (ZH)**: 评估不同学科领域中密集检索器的鲁棒性 

**Authors**: Sarthak Chaturvedi, Anurag Acharya, Rounak Meyur, Koby Hayashi, Sai Munikoti, Sameera Horawalavithana  

**Link**: [PDF](https://arxiv.org/pdf/2506.21581)  

**Abstract**: Evaluation benchmark characteristics may distort the true benefits of domain adaptation in retrieval models. This creates misleading assessments that influence deployment decisions in specialized domains. We show that two benchmarks with drastically different features such as topic diversity, boundary overlap, and semantic complexity can influence the perceived benefits of fine-tuning. Using environmental regulatory document retrieval as a case study, we fine-tune ColBERTv2 model on Environmental Impact Statements (EIS) from federal agencies. We evaluate these models across two benchmarks with different semantic structures. Our findings reveal that identical domain adaptation approaches show very different perceived benefits depending on evaluation methodology. On one benchmark, with clearly separated topic boundaries, domain adaptation shows small improvements (maximum 0.61% NDCG gain). However, on the other benchmark with overlapping semantic structures, the same models demonstrate large improvements (up to 2.22% NDCG gain), a 3.6-fold difference in the performance benefit. We compare these benchmarks through topic diversity metrics, finding that the higher-performing benchmark shows 11% higher average cosine distances between contexts and 23% lower silhouette scores, directly contributing to the observed performance difference. These results demonstrate that benchmark selection strongly determines assessments of retrieval system effectiveness in specialized domains. Evaluation frameworks with well-separated topics regularly underestimate domain adaptation benefits, while those with overlapping semantic boundaries reveal improvements that better reflect real-world regulatory document complexity. Our findings have important implications for developing and deploying AI systems for interdisciplinary domains that integrate multiple topics. 

**Abstract (ZH)**: 评价基准特征可能会歪曲领域适应在检索模型中的真正益处。这会形成误导性的评估，影响专门领域的部署决策。我们通过环境监管文件检索案例研究，发现具有截然不同特征（如主题多样性、边界重叠和语义复杂性）的两个基准可以影响微调感知益处。我们在联邦机构的环境影响声明（EIS）上对ColBERTv2模型进行微调，并在具有不同语义结构的两个基准上进行评估。我们的研究发现，相同的领域适应方法在不同的评估方法下表现出非常不同的感知益处。在具有明显分开的主题边界的基准上，领域适应显示出微小的改善（最大0.61%的NDCG增益）。而在具有重叠语义结构的基准上，相同的模型则显示出显著的改善（最高2.22%的NDCG增益），性能效益提升达3.6倍。我们通过主题多样性指标比较这些基准，发现表现更好的基准显示了11%更高的平均余弦距离和23%更低的轮廓评分，直接导致观察到的性能差异。这些结果表明，基准选择强烈决定了在专门领域的检索系统效果评估。具有清晰主题区隔的评价框架通常会低估领域适应的益处，而具有重叠语义边界的框架则揭示了更能反映实际监管文件复杂性的改进。我们的研究结果对于开发和部署整合多个主题的跨学科领域的AI系统具有重要启示意义。 

---
# Adapting Whisper for Parameter-efficient Code-Switching Speech Recognition via Soft Prompt Tuning 

**Title (ZH)**: 基于软提示调谐的参数高效代码切换语音识别适应性研究 

**Authors**: Hongli Yang, Yizhou Peng, Hao Huang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21576)  

**Abstract**: Large-scale multilingual ASR models like Whisper excel in high-resource settings but face challenges in low-resource scenarios, such as rare languages and code-switching (CS), due to computational costs and catastrophic forgetting. We explore Soft Prompt Tuning (SPT), a parameter-efficient method to enhance CS ASR while preserving prior knowledge. We evaluate two strategies: (1) full fine-tuning (FFT) of both soft prompts and the entire Whisper model, demonstrating improved cross-lingual capabilities compared to traditional methods, and (2) adhering to SPT's original design by freezing model parameters and only training soft prompts. Additionally, we introduce SPT4ASR, a combination of different SPT variants. Experiments on the SEAME and ASRU2019 datasets show that deep prompt tuning is the most effective SPT approach, and our SPT4ASR methods achieve further error reductions in CS ASR, maintaining parameter efficiency similar to LoRA, without degrading performance on existing languages. 

**Abstract (ZH)**: 大规模多语言ASR模型如Whisper在资源丰富环境下表现出色，但在资源匮乏场景如稀有语言和代码切换（CS）中面临计算成本和灾难性遗忘的挑战。我们探索了软提示调谐（SPT）这一参数高效的方法，以增强代码切换（CS）ASR的同时保留先前知识。我们评估了两种策略：（1）完整调谐（FFT）软提示和整个Whisper模型，显示了与传统方法相比改进的跨语言能力，以及（2）遵循SPT原始设计，冻结模型参数仅训练软提示。此外，我们引入了SPT4ASR，这是一种不同SPT变体的组合。在SEAME和ASRU2019数据集上的实验表明，深层提示调谐是效果最佳的SPT方法，我们的SPT4ASR方法在代码切换ASR中实现了进一步的错误减少，保持了与LoRA相似的参数效率，且不牺牲现有语言上的性能。 

---
# The Saturation Point of Backtranslation in High Quality Low Resource English Gujarati Machine Translation 

**Title (ZH)**: 高质量低资源英吉利瓦利机器翻译中的回译饱和点 

**Authors**: Arwa Arif  

**Link**: [PDF](https://arxiv.org/pdf/2506.21566)  

**Abstract**: Backtranslation BT is widely used in low resource machine translation MT to generate additional synthetic training data using monolingual corpora. While this approach has shown strong improvements for many language pairs, its effectiveness in high quality, low resource settings remains unclear. In this work, we explore the effectiveness of backtranslation for English Gujarati translation using the multilingual pretrained MBART50 model. Our baseline system, trained on a high quality parallel corpus of approximately 50,000 sentence pairs, achieves a BLEU score of 43.8 on a validation set. We augment this data with carefully filtered backtranslated examples generated from monolingual Gujarati text. Surprisingly, adding this synthetic data does not improve translation performance and, in some cases, slightly reduces it. We evaluate our models using multiple metrics like BLEU, ChrF++, TER, BLEURT and analyze possible reasons for this saturation. Our findings suggest that backtranslation may reach a point of diminishing returns in certain low-resource settings and we discuss implications for future research. 

**Abstract (ZH)**: 基于回译的机器翻译在英孟инд্র语低资源设置中的有效性探究 

---
# Bench to the Future: A Pastcasting Benchmark for Forecasting Agents 

**Title (ZH)**: 从今到昔：一个用于预测代理的Pastcasting基准测试 

**Authors**: FutureSearch, Jack Wildman, Nikos I. Bosse, Daniel Hnyk, Peter Mühlbacher, Finn Hambly, Jon Evans, Dan Schwarz, Lawrence Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2506.21558)  

**Abstract**: Forecasting is a challenging task that offers a clearly measurable way to study AI systems. Forecasting requires a large amount of research on the internet, and evaluations require time for events to happen, making the development of forecasting benchmarks challenging. To date, no forecasting benchmark provides a realistic, hermetic, and repeatable environment for LLM forecasters. We introduce Bench To the Future (BTF), a "pastcasting" benchmark with hundreds of high-quality questions for which the resolution is already known. Each question is accompanied by a large offline corpus of tens of thousands of relevant web pages, enabling a way to elicit realistic "forecasts" on past events from LLMs. Results suggest that our pastcasting environment can produce results comparable to those based on forecasts using the internet on at-the-time unresolved questions. We show results benchmarking agent and chain-of-thought forecasting approaches using several LLMs, including the recently-released Claude 4 models, and demonstrate BTF's ability to track steady forecasting capability progress over time. We intend this to be a living benchmark, with new questions added continually to account for increasing training data cutoff dates. We invite researchers to contact us at hello@futuresearch.ai to utilize our benchmark or tooling for their own research. 

**Abstract (ZH)**: Bench To the Future: A "Pastcasting" Benchmark for Assessing LLM Forecasting Capabilities 

---
# On the Necessity of Output Distribution Reweighting for Effective Class Unlearning 

**Title (ZH)**: 关于输出分布重加权在有效类遗忘中的必要性 

**Authors**: Yian Wang, Ali Ebrahimpour-Boroojeny, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2506.20893)  

**Abstract**: In this work, we introduce an output-reweighting unlearning method, RWFT, a lightweight technique that erases an entire class from a trained classifier without full retraining. Forgetting specific classes from trained models is essential for enforcing user deletion rights and mitigating harmful or biased predictions. The full retraining is costly and existing unlearning methods fail to replicate the behavior of the retrained models when predicting samples from the unlearned class. We prove this failure by designing a variant of membership inference attacks, MIA-NN that successfully reveals the unlearned class for any of these methods. We propose a simple redistribution of the probability mass for the prediction on the samples in the forgotten class which is robust to MIA-NN. We also introduce a new metric based on the total variation (TV) distance of the prediction probabilities to quantify residual leakage to prevent future methods from susceptibility to the new attack. Through extensive experiments with state of the art baselines in machine unlearning, we show that our approach matches the results of full retraining in both metrics used for evaluation by prior work and the new metric we propose in this work. Compare to state-of-the-art methods, we gain 2.79% in previously used metrics and 111.45% in our new TV-based metric over the best existing method. 

**Abstract (ZH)**: 在这种工作中，我们引入了一种输出重权重遗忘方法RWFT，这是一种轻量级技术，可以在不进行完全重新训练的情况下从已训练分类器中删除整个类别。从训练模型中遗忘特定类别对于确保用户的删除权利和减轻有害或偏颇的预测至关重要。完全重新训练成本高，现有遗忘方法无法在预测未遗忘类别样本时复制重新训练模型的行为。我们通过设计一种MIA-NN变体的成员资格推理攻击方式证明了这种失败，该攻击方式能够成功揭示这些方法中的任一类未遗忘类别。我们提出了一种简单的概率质量再分配方法，以在遗忘类别样本的预测中具有针对MIA-NN的鲁棒性。我们还引入了一个基于预测概率的总变差（TV）距离的新度量，以量化剩余泄露，防止未来方法对新攻击的易感性。通过在机器遗忘的先进基线方法上进行广泛实验，我们展示了我们的方法在用于评估的先前工作所使用的两个指标中与完全重新训练的结果相当，并且在本研究中提出的新基于TV的度量上超过当前最佳方法111.45%。相比最先进的方法，我们在先前使用的两个度量中分别提高了2.79%和111.45%。 

---
