# The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements 

**Title (ZH)**: 自动化大语言模型速度竞赛基准：重现NanoGPT改进 

**Authors**: Bingchen Zhao, Despoina Magka, Minqi Jiang, Xian Li, Roberta Raileanu, Tatiana Shavrina, Jean-Christophe Gagnon-Audet, Kelvin Niu, Shagun Sodhani, Michael Shvartsman, Andrei Lupu, Alisia Lupidi, Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Thomas Foster, Lucia Cipolina-Kun, Abhishek Charnalia, Derek Dunfield, Alexander H. Miller, Oisin Mac Aodha, Jakob Foerster, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2506.22419)  

**Abstract**: Rapid advancements in large language models (LLMs) have the potential to assist in scientific progress. A critical capability toward this endeavor is the ability to reproduce existing work. To evaluate the ability of AI agents to reproduce results in an active research area, we introduce the Automated LLM Speedrunning Benchmark, leveraging the research community contributions on the NanoGPT speedrun, a competition to train a GPT-2 model in the shortest time. Each of the 19 speedrun tasks provides the agent with the previous records training script, optionally paired with one of three hint formats, ranging from pseudocode to paper-like descriptions of the new records improvements. Records execute quickly by design and speedrun improvements encompass diverse code-level changes, ranging from high-level algorithmic advancements to hardware-aware optimizations. These features make the benchmark both accessible and realistic for the frontier problem of improving LLM training. We find that recent reasoning LLMs combined with SoTA scaffolds struggle to reimplement already-known innovations in our benchmark, even when given detailed hints. Our benchmark thus provides a simple, non-saturated measure of an LLMs ability to automate scientific reproduction, a necessary (but not sufficient) skill for an autonomous research agent. 

**Abstract (ZH)**: 大型语言模型的快速进步有望促进科学研究的进步。实现这一目标的关键能力是能够再现现有工作。为了评估AI代理在活跃研究领域的再现结果能力，我们引入了自动LLM速跑基准，利用NanoGPT速跑的社区研究成果，NanoGPT速跑是一个训练GPT-2模型的最快时间竞赛。每项19个速跑任务都为代理提供了之前的记录训练脚本，可选地配以三种提示格式之一，从伪代码到论文般的描述新记录改进。记录设计上执行速度快，速跑改进涵盖了从高级算法进步到硬件感知优化等各种代码层面的变化。这些特性使得基准既适用于改进LLM训练的前沿问题，又具备现实性。我们发现，即使提供了详细的提示，最新的推理LLM与最先进的支架相结合，在基准中仍然难以重新实现已知的创新。因此，我们的基准提供了一种简单且未饱和的评估LLM自动化科学研究能力的方法，这是自主研究代理所需（但不足够）的一项技能。 

---
# AI Model Passport: Data and System Traceability Framework for Transparent AI in Health 

**Title (ZH)**: AI模型护照：面向透明医疗的人工智能数据与系统可追溯性框架 

**Authors**: Varvara Kalokyri, Nikolaos S. Tachos, Charalampos N. Kalantzopoulos, Stelios Sfakianakis, Haridimos Kondylakis, Dimitrios I. Zaridis, Sara Colantonio, Daniele Regge, Nikolaos Papanikolaou, ProCAncer-I consortium, Konstantinos Marias, Dimitrios I. Fotiadis, Manolis Tsiknakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.22358)  

**Abstract**: The increasing integration of Artificial Intelligence (AI) into health and biomedical systems necessitates robust frameworks for transparency, accountability, and ethical compliance. Existing frameworks often rely on human-readable, manual documentation which limits scalability, comparability, and machine interpretability across projects and platforms. They also fail to provide a unique, verifiable identity for AI models to ensure their provenance and authenticity across systems and use cases, limiting reproducibility and stakeholder trust. This paper introduces the concept of the AI Model Passport, a structured and standardized documentation framework that acts as a digital identity and verification tool for AI models. It captures essential metadata to uniquely identify, verify, trace and monitor AI models across their lifecycle - from data acquisition and preprocessing to model design, development and deployment. In addition, an implementation of this framework is presented through AIPassport, an MLOps tool developed within the ProCAncer-I EU project for medical imaging applications. AIPassport automates metadata collection, ensures proper versioning, decouples results from source scripts, and integrates with various development environments. Its effectiveness is showcased through a lesion segmentation use case using data from the ProCAncer-I dataset, illustrating how the AI Model Passport enhances transparency, reproducibility, and regulatory readiness while reducing manual effort. This approach aims to set a new standard for fostering trust and accountability in AI-driven healthcare solutions, aspiring to serve as the basis for developing transparent and regulation compliant AI systems across domains. 

**Abstract (ZH)**: AI模型护照：促进AI驱动医疗健康解决方案中的透明度、问责制和合规性的新标准 

---
# Embodied AI Agents: Modeling the World 

**Title (ZH)**: 具身人工智能代理：建模世界 

**Authors**: Pascale Fung, Yoram Bachrach, Asli Celikyilmaz, Kamalika Chaudhuri, Delong Chen, Willy Chung, Emmanuel Dupoux, Hervé Jégou, Alessandro Lazaric, Arjun Majumdar, Andrea Madotto, Franziska Meier, Florian Metze, Théo Moutakanni, Juan Pino, Basile Terver, Joseph Tighe, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2506.22355)  

**Abstract**: This paper describes our research on AI agents embodied in visual, virtual or physical forms, enabling them to interact with both users and their environments. These agents, which include virtual avatars, wearable devices, and robots, are designed to perceive, learn and act within their surroundings, which makes them more similar to how humans learn and interact with the environments as compared to disembodied agents. We propose that the development of world models is central to reasoning and planning of embodied AI agents, allowing these agents to understand and predict their environment, to understand user intentions and social contexts, thereby enhancing their ability to perform complex tasks autonomously. World modeling encompasses the integration of multimodal perception, planning through reasoning for action and control, and memory to create a comprehensive understanding of the physical world. Beyond the physical world, we also propose to learn the mental world model of users to enable better human-agent collaboration. 

**Abstract (ZH)**: 本文描述了我们在以视觉、虚拟或物理形式体现的AI代理方面的研究，使它们能够与用户及其环境进行交互。这些代理包括虚拟化身、穿戴设备和机器人，设计用于感知、学习和在其环境中的行动，使其在与人类学习和与环境交互的方式上更为相似，而不是 disembodied 代理。我们提出，世界模型的发展对于体现的AI代理的推理和规划至关重要，这使这些代理能够理解并预测其环境，理解用户意图和社会背景，从而增强其执行复杂任务的能力。世界模型涵盖了多模态感知的综合、通过推理进行计划以实现动作控制、以及记忆的运用，从而创建对物理世界的全面理解。此外，我们还提出学习用户的心智世界模型，以实现更好的人机协作。 

---
# Conceptual Topic Aggregation 

**Title (ZH)**: 概念主题聚合 

**Authors**: Klara M. Gutekunst, Dominik Dürrschnabel, Johannes Hirth, Gerd Stumme  

**Link**: [PDF](https://arxiv.org/pdf/2506.22309)  

**Abstract**: The vast growth of data has rendered traditional manual inspection infeasible, necessitating the adoption of computational methods for efficient data exploration. Topic modeling has emerged as a powerful tool for analyzing large-scale textual datasets, enabling the extraction of latent semantic structures. However, existing methods for topic modeling often struggle to provide interpretable representations that facilitate deeper insights into data structure and content. In this paper, we propose FAT-CAT, an approach based on Formal Concept Analysis (FCA) to enhance meaningful topic aggregation and visualization of discovered topics. Our approach can handle diverse topics and file types -- grouped by directories -- to construct a concept lattice that offers a structured, hierarchical representation of their topic distribution. In a case study on the ETYNTKE dataset, we evaluate the effectiveness of our approach against other representation methods to demonstrate that FCA-based aggregation provides more meaningful and interpretable insights into dataset composition than existing topic modeling techniques. 

**Abstract (ZH)**: 数据量的大幅增长使传统的手动检查不可行， necessitating the adoption of computational methods for efficient data exploration. 主题建模已成为分析大规模文本数据集的强大工具，能够提取潜在的语义结构。然而，现有主题建模方法往往难以提供可解释的表现形式，以促进对数据结构和内容的更深入洞察。本文提出了一种基于形式概念分析（FCA）的FAT-CAT方法，以增强具有意义的主题聚合和发现的主题可视化。该方法可以处理由目录分组的多样化主题和文件类型，构建一个概念格，提供主题分布的结构化、层次表示。在ETYNTKE数据集的案例研究中，我们评估了该方法与其他表示方法的有效性，以证明基于形式概念分析的聚合比现有主题建模技术能提供更具有意义和可解释的数据集组成洞察。 

---
# Artificial Intelligent Disobedience: Rethinking the Agency of Our Artificial Teammates 

**Title (ZH)**: 人工智能的叛逆：重思我们的机器同伴的自主性 

**Authors**: Reuth Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2506.22276)  

**Abstract**: Artificial intelligence has made remarkable strides in recent years, achieving superhuman performance across a wide range of tasks. Yet despite these advances, most cooperative AI systems remain rigidly obedient, designed to follow human instructions without question and conform to user expectations, even when doing so may be counterproductive or unsafe. This paper argues for expanding the agency of AI teammates to include \textit{intelligent disobedience}, empowering them to make meaningful and autonomous contributions within human-AI teams. It introduces a scale of AI agency levels and uses representative examples to highlight the importance and growing necessity of treating AI autonomy as an independent research focus in cooperative settings. The paper then explores how intelligent disobedience manifests across different autonomy levels and concludes by proposing initial boundaries and considerations for studying disobedience as a core capability of artificial agents. 

**Abstract (ZH)**: 人工智能在近年来取得了显著进展，实现了跨多种任务的超human表现。然而，尽管取得了这些进步，大多数协作性AI系统仍然僵化地顺从，被设计为未经质疑地遵循人类指令并符合用户期望，即使这样做可能是无效的或不安全的。本文主张扩大AI队友的自主权限，使其包括“智能不服从”，使其能够在人机团队中做出有意义且自主的贡献。文章介绍了一个AI自主权等级量表，并通过典型示例强调了在协作环境中将AI自主性作为独立研究重点的重要性以及其日益增长的必要性。随后，文章探讨了智能不服从在不同自主权级别上的表现，并提出了一些初步的研究边界和考虑，以将不服从作为人工代理的核心能力进行研究。 

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
# Universal Retrieval for Multimodal Trajectory Modeling 

**Title (ZH)**: 多模态轨迹建模的通用检索 

**Authors**: Xuan Zhang, Ziyan Jiang, Rui Meng, Yifei Leng, Zhenbang Xiao, Zora Zhiruo Wang, Yanyi Shang, Dehan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.22056)  

**Abstract**: Trajectory data, capturing human actions and environmental states across various modalities, holds significant potential for enhancing AI agent capabilities, particularly in GUI environments. However, how to model the representation of trajectory-level data presents a significant challenge that has not been systematically addressed amid explosive trajectory data growth. In this work, we introduce Multimodal Trajectory Retrieval, bridging the gap between universal retrieval and agent-centric trajectory modeling. We construct the Unified Agent Trajectory Dataset (UATD) from annotated demonstrations and states across diverse real-world scenarios. Based on this, we present GAE-Bench, a benchmark containing a large number of trajectory-based retrieval pairs. In addition, we propose GAE-Retriever, a multimodal retrieval framework that adopts vision-language models and incorporates optimized contrastive learning through a token selection and the GradCache mechanism. Comprehensive evaluations across multiple datasets show that GAE-Retriever consistently outperforms strong baselines in retrieval recall, highlighting its effectiveness in advancing multimodal trajectory retrieval. 

**Abstract (ZH)**: 多模态轨迹检索：介于通用检索与代理中心轨迹建模之间的桥梁 

---
# LeanConjecturer: Automatic Generation of Mathematical Conjectures for Theorem Proving 

**Title (ZH)**: LeanConjecturer: 自动生成数学猜想以证明定理 

**Authors**: Naoto Onda, Kazumi Kasaura, Yuta Oriike, Masaya Taniguchi, Akiyoshi Sannai, Sho Sonoda  

**Link**: [PDF](https://arxiv.org/pdf/2506.22005)  

**Abstract**: We introduce LeanConjecturer, a pipeline for automatically generating university-level mathematical conjectures in Lean 4 using Large Language Models (LLMs). Our hybrid approach combines rule-based context extraction with LLM-based theorem statement generation, addressing the data scarcity challenge in formal theorem proving. Through iterative generation and evaluation, LeanConjecturer produced 12,289 conjectures from 40 Mathlib seed files, with 3,776 identified as syntactically valid and non-trivial, that is, cannot be proven by \texttt{aesop} tactic. We demonstrate the utility of these generated conjectures for reinforcement learning through Group Relative Policy Optimization (GRPO), showing that targeted training on domain-specific conjectures can enhance theorem proving capabilities. Our approach generates 103.25 novel conjectures per seed file on average, providing a scalable solution for creating training data for theorem proving systems. Our system successfully verified several non-trivial theorems in topology, including properties of semi-open, alpha-open, and pre-open sets, demonstrating its potential for mathematical discovery beyond simple variations of existing results. 

**Abstract (ZH)**: LeanConjecturer：一种使用大型语言模型在Lean 4中自动生成大学水平数学猜想的管道 

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
# CitySim: Modeling Urban Behaviors and City Dynamics with Large-Scale LLM-Driven Agent Simulation 

**Title (ZH)**: CitySim：基于大规模LLM驱动代理仿真建模城市行为与城市动力学 

**Authors**: Nicolas Bougie, Narimasa Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.21805)  

**Abstract**: Modeling human behavior in urban environments is fundamental for social science, behavioral studies, and urban planning. Prior work often rely on rigid, hand-crafted rules, limiting their ability to simulate nuanced intentions, plans, and adaptive behaviors. Addressing these challenges, we envision an urban simulator (CitySim), capitalizing on breakthroughs in human-level intelligence exhibited by large language models. In CitySim, agents generate realistic daily schedules using a recursive value-driven approach that balances mandatory activities, personal habits, and situational factors. To enable long-term, lifelike simulations, we endow agents with beliefs, long-term goals, and spatial memory for navigation. CitySim exhibits closer alignment with real humans than prior work, both at micro and macro levels. Additionally, we conduct insightful experiments by modeling tens of thousands of agents and evaluating their collective behaviors under various real-world scenarios, including estimating crowd density, predicting place popularity, and assessing well-being. Our results highlight CitySim as a scalable, flexible testbed for understanding and forecasting urban phenomena. 

**Abstract (ZH)**: 城市环境中人类行为建模对于社会科学、行为研究和城市规划至关重要。先前的工作经常依赖于僵硬的手工设计规则，限制了它们模拟复杂意图、计划和适应性行为的能力。为应对这些挑战，我们设想了一个城市模拟器（CitySim），利用大型语言模型在人类级别智能方面取得的突破。在CitySim中，代理使用递归的价值驱动方法生成现实的日常计划，平衡强制性活动、个人习惯和情境因素。为了实现长期、拟真的模拟，我们赋予代理信念、长期目标和空间记忆以进行导航。CitySim在微观和宏观层面上都更接近真实人类的行为，此外，我们通过建模数以万计的代理并评估它们在各种现实世界场景下的集体行为，进行了富有洞察力的实验，包括估计人群密度、预测地点受欢迎程度以及评估福祉。我们的结果突显了CitySim作为一个可扩展、灵活的试验平台，用于理解和预测城市现象的重要性。 

---
# MobiVerse: Scaling Urban Mobility Simulation with Hybrid Lightweight Domain-Specific Generator and Large Language Models 

**Title (ZH)**: MobiVerse: 通过混合轻量级领域专用生成器和大型语言模型扩展城市 Mobility 模拟 

**Authors**: Yifan Liu, Xishun Liao, Haoxuan Ma, Jonathan Liu, Rohan Jadhav, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.21784)  

**Abstract**: Understanding and modeling human mobility patterns is crucial for effective transportation planning and urban development. Despite significant advances in mobility research, there remains a critical gap in simulation platforms that allow for algorithm development, policy implementation, and comprehensive evaluation at scale. Traditional activity-based models require extensive data collection and manual calibration, machine learning approaches struggle with adaptation to dynamic conditions, and treding agent-based Large Language Models (LLMs) implementations face computational constraints with large-scale simulations. To address these challenges, we propose MobiVerse, a hybrid framework leverages the efficiency of lightweight domain-specific generator for generating base activity chains with the adaptability of LLMs for context-aware modifications. A case study was conducted in Westwood, Los Angeles, where we efficiently generated and dynamically adjusted schedules for the whole population of approximately 53,000 agents on a standard PC. Our experiments demonstrate that MobiVerse successfully enables agents to respond to environmental feedback, including road closures, large gathering events like football games, and congestion, through our hybrid framework. Its modular design facilitates testing various mobility algorithms at both transportation system and agent levels. Results show our approach maintains computational efficiency while enhancing behavioral realism. MobiVerse bridges the gap in mobility simulation by providing a customizable platform for mobility systems planning and operations with benchmark algorithms. Code and videos are available at this https URL. 

**Abstract (ZH)**: 理解并建模人类移动模式对于有效的运输规划和城市发展至关重要。尽管在移动性研究方面取得了显著进展，但在支持算法开发、政策实施和大规模综合评估的仿真平台上仍存在关键缺口。传统的基于活动的模型需要大量的数据收集和手动校准，机器学习方法难以适应动态条件，而基于代理的大型语言模型（LLMs）实现则面临大规模仿真中的计算约束。为解决这些挑战，我们提出MobiVerse，这是一种混合框架，结合了轻量级领域特定生成器的高效性与大型语言模型的适应性，以便进行上下文感知的修改。我们在洛杉矶的韦斯特伍德地区进行了案例研究，在标准PC上高效地生成并动态调整了大约53,000个代理的完整时间表。我们的实验显示，MobiVerse通过混合框架成功使代理能够响应环境反馈，包括道路封闭、大型集会活动（如足球比赛）和拥堵等因素。其模块化设计便于在运输系统和代理层面测试各种移动性算法。结果显示，我们的方法在保持计算效率的同时提高了行为现实度。MobiVerse通过提供一个可定制的平台，弥合了移动性模拟的缺口，并配备基准算法。代码和视频可在以下链接获取。 

---
# THE-Tree: Can Tracing Historical Evolution Enhance Scientific Verification and Reasoning? 

**Title (ZH)**: THE-Tree: 追踪历史演变能否增强科学验证与推理？ 

**Authors**: Xin Wang, Jiyao Liu, Yulong Xiao, Junzhi Ning, Lihao Liu, Junjun He, Botian Shi, Kaicheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21763)  

**Abstract**: Large Language Models (LLMs) are accelerating scientific idea generation, but rigorously evaluating these numerous, often superficial, AI-generated propositions for novelty and factual accuracy is a critical bottleneck; manual verification is too this http URL validation methods are inadequate: LLMs as standalone verifiers may hallucinate and lack domain knowledge (our findings show ~60\% unawareness of relevant papers in specific domains), while traditional citation networks lack explicit causality and narrative surveys are this http URL underscores a core challenge: the absence of structured, verifiable, and causally-linked historical data of scientific this http URL address this,we introduce \textbf{THE-Tree} (\textbf{T}echnology \textbf{H}istory \textbf{E}volution Tree), a computational framework that constructs such domain-specific evolution trees from scientific this http URL-Tree employs a search algorithm to explore evolutionary paths. During its node expansion, it utilizes a novel "Think-Verbalize-Cite-Verify" process: an LLM proposes potential advancements and cites supporting literature. Critically, each proposed evolutionary link is then validated for logical coherence and evidential support by a recovered natural language inference mechanism that interrogates the cited literature, ensuring that each step is this http URL construct and validate 88 THE-Trees across diverse domains and release a benchmark dataset including up to 71k fact verifications covering 27k papers to foster further this http URL demonstrate that i) in graph completion, our THE-Tree improves hit@1 by 8\% to 14\% across multiple models compared to traditional citation networks; ii) for predicting future scientific developments, it improves hit@1 metric by nearly 10\%; and iii) when combined with other methods, it boosts the performance of evaluating important scientific papers by almost 100\%. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在加速科学想法的生成，但严谨评估这些众多、常表面性的AI生成命题的新颖性和事实准确性是一个关键瓶颈；手动验证效率低下：独立验证的LLMs可能会出现幻觉并缺乏领域知识（我们的研究显示特定领域的相关论文约60%未被意识到），而传统的引用网络缺乏明确的因果关系和叙述性综述。凸显了一个核心挑战：缺乏结构化、可验证且因果关联的科学历史数据。为解决这一问题，我们引入了**THE-Tree**（技术历史进化树）这一计算框架，从科学文献中构建领域特定的进化树。THE-Tree 使用搜索算法探索进化路径。在节点扩展过程中，它利用一种新的“思考-表达-引述-验证”过程：LLM 提出潜在进展并引用支持文献。关键的是，每条提议的进化链接都会通过恢复的自然语言推理机制进行逻辑连贯性和证据支持的验证，确保每一步都是合理且有证据支持的。我们构建并验证了88个THE-Tree，覆盖多个领域，并发布了包含多达71,000个事实验证涵盖27,000篇论文的数据集，以促进进一步的研究。实验结果表明：i) 在图完成任务中，我们的THE-Tree在多个模型中将hit@1指标提高了8%至14%，优于传统引用网络；ii) 在预测未来科学进展方面，其hit@1指标提高了近10%；iii) 当与其他方法结合使用时，它可以将评估重要科学论文的性能提升近100%。 

---
# Hierarchical Reasoning Model 

**Title (ZH)**: 层次推理模型 

**Authors**: Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, Yasin Abbasi Yadkori  

**Link**: [PDF](https://arxiv.org/pdf/2506.21734)  

**Abstract**: Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI. Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency. HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes. Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities. These results underscore HRM's potential as a transformative advancement toward universal computation and general-purpose reasoning systems. 

**Abstract (ZH)**: 层次推理模型：一种既能实现显著计算深度又能保持训练稳定性和效率的新型递归架构及其在复杂推理任务中的应用 

---
# SEEA-R1: Tree-Structured Reinforcement Fine-Tuning for Self-Evolving Embodied Agents 

**Title (ZH)**: SEEA-R1: 基于树结构强化细调的自进化体化代理 

**Authors**: Wanxin Tian, Shijie Zhang, Kevin Zhang, Xiaowei Chi, Yulin Luo, Junyu Lu, Chunkai Fan, Qiang Zhou, Yiming Zhao, Ning Liu Siyu Lin, Zhiyuan Qin, Xiaozhu Ju, Shanghang Zhang, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21669)  

**Abstract**: Self-evolution, the ability of agents to autonomously improve their reasoning and behavior, is essential for the embodied domain with long-horizon, real-world tasks. Despite current advancements in reinforcement fine-tuning (RFT) showing strong performance in enhancing reasoning in LLMs, its potential to enable self-evolving embodied intelligence with multi-modal interactions remains largely unexplored. Specifically, reinforcement fine-tuning faces two fundamental obstacles in embodied settings: (i) the lack of accessible intermediate rewards in multi-step reasoning tasks limits effective learning signals, and (ii) reliance on hand-crafted reward functions restricts generalization to novel tasks and environments. To address these challenges, we present Self-Evolving Embodied Agents-R1, SEEA-R1, the first RFT framework designed for enabling the self-evolving capabilities of embodied agents. Specifically, to convert sparse delayed rewards into denser intermediate signals that improve multi-step reasoning, we propose Tree-based group relative policy optimization (Tree-GRPO), which integrates Monte Carlo Tree Search into GRPO. To generalize reward estimation across tasks and scenes, supporting autonomous adaptation and reward-driven self-evolution, we further introduce Multi-modal Generative Reward Model (MGRM). To holistically evaluate the effectiveness of SEEA-R1, we evaluate on the ALFWorld benchmark, surpassing state-of-the-art methods with scores of 85.07% (textual) and 36.19% (multi-modal), outperforming prior models including GPT-4o. SEEA-R1 also achieves scores of 80.3% without environmental reward, surpassing all open-source baselines and highlighting its scalability as a self-evolving embodied agent. Additional experiments and qualitative analysis further support the potential of SEEA-R1 for future research in scalable embodied intelligence. 

**Abstract (ZH)**: 自进化能力：使代理能够在长时 horizon、真实世界任务中自主提升其推理和行为的能力对于体现域至关重要。尽管当前强化微调（RFT）在增强大语言模型的推理能力方面表现出色，但其在多模态交互中使代理具备自进化能力方面的潜力尚未得到充分探索。具体来说，强化微调在体现环境中面临两个根本障碍：（i）多步推理任务中缺乏可访问的中间奖励限制了有效学习信号，（ii）依赖于手工设计的奖励函数限制了对新任务和环境的泛化能力。为解决这些挑战，我们提出了Self-Evolving Embodied Agents-R1（SEEA-R1），这是第一个为使体现代理具备自进化能力设计的RFT框架。具体而言，为将稀疏的延迟奖励转换为改善多步推理的更密集的中间信号，我们提出了基于树结构的分组相对策略优化（Tree-GRPO），该方法将蒙特卡ロ树搜索整合到GRPO中。为了跨任务和场景泛化奖励估计，支持自主适应和奖励驱动的自进化，我们进一步引入了多模态生成奖励模型（MGRM）。为了全面评估SEEA-R1的有效性，我们在ALFWorld基准上进行评估，得分分别为85.07%（文本）和36.19%（多模态），超越了最先进的方法，并优于包括GPT-4o在内的先前模型。SEEA-R1在无环境奖励的情况下得分80.3%，超越所有开源基线，突显了其作为自进化体现代理的可扩展性。附加实验和定性分析进一步支持了SEEA-R1在可扩展体现智能方面的研究潜力。 

---
# CLoVE: Personalized Federated Learning through Clustering of Loss Vector Embeddings 

**Title (ZH)**: CLoVE：通过损失向量嵌入聚类实现的个性化联邦学习 

**Authors**: Randeep Bhatia, Nikos Papadis, Murali Kodialam, TV Lakshman, Sayak Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2506.22427)  

**Abstract**: We propose CLoVE (Clustering of Loss Vector Embeddings), a novel algorithm for Clustered Federated Learning (CFL). In CFL, clients are naturally grouped into clusters based on their data distribution. However, identifying these clusters is challenging, as client assignments are unknown. CLoVE utilizes client embeddings derived from model losses on client data, and leverages the insight that clients in the same cluster share similar loss values, while those in different clusters exhibit distinct loss patterns. Based on these embeddings, CLoVE is able to iteratively identify and separate clients from different clusters and optimize cluster-specific models through federated aggregation. Key advantages of CLoVE over existing CFL algorithms are (1) its simplicity, (2) its applicability to both supervised and unsupervised settings, and (3) the fact that it eliminates the need for near-optimal model initialization, which makes it more robust and better suited for real-world applications. We establish theoretical convergence bounds, showing that CLoVE can recover clusters accurately with high probability in a single round and converges exponentially fast to optimal models in a linear setting. Our comprehensive experiments comparing with a variety of both CFL and generic Personalized Federated Learning (PFL) algorithms on different types of datasets and an extensive array of non-IID settings demonstrate that CLoVE achieves highly accurate cluster recovery in just a few rounds of training, along with state-of-the-art model accuracy, across a variety of both supervised and unsupervised PFL tasks. 

**Abstract (ZH)**: CLoVE：群集联邦学习的聚类损失向量嵌入算法 

---
# HyperCLOVA X THINK Technical Report 

**Title (ZH)**: HyperCLOVA X THINK 技术报告 

**Authors**: NAVER Cloud HyperCLOVA X Team  

**Link**: [PDF](https://arxiv.org/pdf/2506.22403)  

**Abstract**: We introduce HyperCLOVA X THINK, the first reasoning-focused large language model in the HyperCLOVA X family, pre-trained on roughly $6$ trillion high-quality Korean, and English tokens, augmented with targeted synthetic Korean data. It was implemented as a compute-memory-balanced Peri-LN Transformer scaled with $\mu$P, pre-trained through a three-stage curriculum that expands the context window to $128$K tokens, and post-trained via supervised fine-tuning with Reinforcement Learning from Verifiable Rewards supports both detailed rationale and concise-answer modes. It delivers competitive performance against similarly sized models on Korea-focused benchmarks such as KMMLU, CSAT, KoBALT-700, HAERAE-1.0, and KoBigBench, while preserving robust bilingual consistency and translation quality. In addition, a vision-augmented variant matches or exceeds GPT-4.1 on the KCSAT STEM benchmark, all of which are achieved with substantially lower training compute than existing models of similar sizes. We also present a pruning and distillation technique that will soon be applied to HyperCLOVA X THINK for an open-source and business-friendly foundation model. Altogether, these capabilities position HyperCLOVA X THINK as a robust foundation for Korean AI innovation and a valuable resource for the global research community. 

**Abstract (ZH)**: 我们介绍HyperCLOVA X THINK，这是HyperCLOVA X家族中第一个注重推理的大语言模型，预训练了大约6万亿高质量的韩语和英语token，并通过目标合成韩语文本进行了扩充。该模型采用计算与内存平衡的peri-LN Transformer架构，并采用μP扩展，通过三阶段的曲程进行预训练，扩展上下文窗口至128K token，并通过验证奖励支持的强化学习监督微调技术进行后续训练，支持详细的推理模式和简洁的答案模式。它在以韩国为重点的基准测试（如KMMLU、CSAT、KoBALT-700、HAERAE-1.0和KoBigBench）中取得了与同样规模模型相当的性能，同时保持了稳健的双语一致性和平行文本质量。此外，其视图增强版本在KCSAT STEM基准测试中达到了或超过了GPT-4.1的表现，所有这些都比现有同等规模模型的训练计算成本要低得多。我们还介绍了一种即将应用于HyperCLOVA X THINK的剪枝和蒸馏技术，为开源和商业友好型基础模型提供支持。总体而言，这些能力使HyperCLOVA X THINK成为韩语AI创新的稳健基础，并为全球研究社区提供宝贵的资源。 

---
# Dehazing Light Microscopy Images with Guided Conditional Flow Matching: finding a sweet spot between fidelity and realism 

**Title (ZH)**: 基于引导条件流匹配的去雾光显微镜图像处理：在保真度与真实感之间的平衡 

**Authors**: Anirban Ray, Ashesh, Florian Jug  

**Link**: [PDF](https://arxiv.org/pdf/2506.22397)  

**Abstract**: Fluorescence microscopy is a major driver of scientific progress in the life sciences. Although high-end confocal microscopes are capable of filtering out-of-focus light, cheaper and more accessible microscopy modalities, such as widefield microscopy, can not, which consequently leads to hazy image data. Computational dehazing is trying to combine the best of both worlds, leading to cheap microscopy but crisp-looking images. The perception-distortion trade-off tells us that we can optimize either for data fidelity, e.g. low MSE or high PSNR, or for data realism, measured by perceptual metrics such as LPIPS or FID. Existing methods either prioritize fidelity at the expense of realism, or produce perceptually convincing results that lack quantitative accuracy. In this work, we propose HazeMatching, a novel iterative method for dehazing light microscopy images, which effectively balances these objectives. Our goal was to find a balanced trade-off between the fidelity of the dehazing results and the realism of individual predictions (samples). We achieve this by adapting the conditional flow matching framework by guiding the generative process with a hazy observation in the conditional velocity field. We evaluate HazeMatching on 5 datasets, covering both synthetic and real data, assessing both distortion and perceptual quality. Our method is compared against 7 baselines, achieving a consistent balance between fidelity and realism on average. Additionally, with calibration analysis, we show that HazeMatching produces well-calibrated predictions. Note that our method does not need an explicit degradation operator to exist, making it easily applicable on real microscopy data. All data used for training and evaluation and our code will be publicly available under a permissive license. 

**Abstract (ZH)**: fluorescence显微镜是生命科学领域科学进步的主要驱动力。尽管高端共聚焦显微镜能够过滤掉焦外光，但较便宜且更易获取的显微成像方式，如宽场显微镜，则不能，这导致了模糊的图像数据。计算去雾尝试将两者的优势结合，从而实现低成本显微镜但清晰的图像效果。感知与失真是权衡关系，告诉我们可以优化数据保真度，例如低MSE或高PSNR，或者通过感知度量标准如LPIPS或FID来优化数据现实度。现有方法要么优先考虑保真度而牺牲现实度，要么生成视觉上令人信服但缺乏定量准确性的结果。在本文中，我们提出了一种新的迭代方法HazeMatching，用于去雾光学显微图像，该方法有效地平衡了这些目标。我们的目标是找到去雾结果保真度和个体预测（样本）现实度之间的平衡。我们通过将生成过程指导的条件流匹配框架与朦胧观测引导的条件速度场相结合来实现这一点。我们在5个数据集上评估了HazeMatching，涵盖了合成和真实数据，评估了失真和感知质量。我们的方法与7个基线进行了比较，在平均值上实现了保真度和现实度之间的稳定平衡。此外，通过校准分析，我们展示了HazeMatching产生的预测是具有良好校准的。值得注意的是，我们的方法不需要显式的降级操作符存在，使其易于应用于真实显微镜数据。所有用于训练和评估的数据以及我们的代码将在宽松的许可下公开。 

---
# QuickSilver -- Speeding up LLM Inference through Dynamic Token Halting, KV Skipping, Contextual Token Fusion, and Adaptive Matryoshka Quantization 

**Title (ZH)**: QuickSilver — 通过动态 Tokens 中止、KV 跳过、上下文 Tokens 融合及自适应 Matryoshka 量化加速 LLM 推理 

**Authors**: Danush Khanna, Aditya Kumar Guru, Srivarshinee Sridhar, Zidan Ahmed, Rubhav Bahirwani, Meetu Malhotra, Vinija Jain, Aman Chadha, Amitava Das, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2506.22396)  

**Abstract**: Inference accounts for the majority of latency and energy consumption in large language model (LLM) deployments, often exceeding 90% of total cost. While training-time efficiency has seen extensive progress, runtime optimization remains a key bottleneck, particularly under autoregressive decoding. Existing approaches -- such as pruning, quantization, early exits, and speculative decoding -- often require retraining, architectural changes, or disrupt decoding compatibility. We introduce QuickSilver, a modular, token-level framework that enables semantic adaptivity at inference time without altering model weights or structure. QuickSilver integrates four synergistic mechanisms:
(i) Dynamic Token Halting, which halts computation for tokens with converged representations; (ii) KV Cache Skipping, which selectively suppresses memory writes to reduce attention overhead; and (iii) Contextual Token Fusion, which collapses redundant tokens into shared paths to shrink sequence length.
Unlike speculative decoding or MoE routing, QuickSilver operates entirely on frozen, dense models and requires no auxiliary networks. Applied to GPT-2 and Llama-2 across WikiText-103 and C4, QuickSilver achieves up to 39.6% FLOP reduction with negligible perplexity degradation (<=0.2). 

**Abstract (ZH)**: 快速银：一种无需更改模型权重或结构的推理时语义自适应框架 

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
# Probabilistic Optimality for Inference-time Scaling 

**Title (ZH)**: 推理时缩放的概率优化 

**Authors**: Youkang Wang, Jian Wang, Rubing Chen, Xiao-Yong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22376)  

**Abstract**: Inference-time scaling has emerged as a powerful technique for enhancing the reasoning performance of Large Language Models (LLMs). However, existing approaches often rely on heuristic strategies for parallel sampling, lacking a principled foundation. To address this gap, we propose a probabilistic framework that formalizes the optimality of inference-time scaling under the assumption that parallel samples are independently and identically distributed (i.i.d.), and where the Best-of-N selection strategy follows a probability distribution that can be estimated. Within this framework, we derive a theoretical lower bound on the required number of samples to achieve a target performance level, providing the first principled guidance for compute-efficient scaling. Leveraging this insight, we develop \textsc{OptScale}, a practical algorithm that dynamically determines the optimal number of sampled responses. \textsc{OptScale} employs a language model-based predictor to estimate probabilistic prior parameters, enabling the decision of the minimal number of samples needed that satisfy predefined performance thresholds and confidence levels. Extensive experiments on mathematical reasoning benchmarks (including MATH-500, GSM8K, AIME, and AMC) demonstrate that \textsc{OptScale} significantly reduces sampling overhead while remaining better or on par with state-of-the-art reasoning performance. Our work offers both a theoretical foundation and a practical solution for principled inference-time scaling, addressing a critical gap in the efficient deployment of LLMs for complex reasoning. 

**Abstract (ZH)**: 推理时缩放已成为增强大型语言模型（LLMs）推理性能的强大技术。然而，现有方法通常依赖于启发式策略进行并行采样，缺乏坚实的理论基础。为解决这一问题，我们提出了一种概率框架，该框架在假设并行样本独立同分布（i.i.d.）且最佳N选一策略遵循可估计的概率分布的前提下， formalizes推理时缩放的最优性。在此框架内，我们推导出实现目标性能水平所需的样本数的理论下界，从而提供了首个计算高效缩放的原理性指导。借助这一洞察，我们开发了OptScale，一种实用算法，能够动态确定最优的采样数量。OptScale 使用基于语言模型的预测器估算概率先验参数，使决策所需的最小样本数量能够满足预定义的性能阈值和置信水平。我们在数学推理基准测试（包括MATH-500、GSM8K、AIME和AMC）上的 extensive实验表明，OptScale 显著减少了采样开销，同时保持或优于最先进的推理性能。我们的工作为原理性推理时缩放提供了理论基础和实践解决方案，填补了高效部署LLMs进行复杂推理的空白。 

---
# Sheaf-Based Decentralized Multimodal Learning for Next-Generation Wireless Communication Systems 

**Title (ZH)**: 基于层结构的分布式多模态学习为下一代无线通信系统 

**Authors**: Abdulmomen Ghalkha, Zhuojun Tian, Chaouki Ben Issaid, Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2506.22374)  

**Abstract**: In large-scale communication systems, increasingly complex scenarios require more intelligent collaboration among edge devices collecting various multimodal sensory data to achieve a more comprehensive understanding of the environment and improve decision-making accuracy. However, conventional federated learning (FL) algorithms typically consider unimodal datasets, require identical model architectures, and fail to leverage the rich information embedded in multimodal data, limiting their applicability to real-world scenarios with diverse modalities and varying client capabilities. To address this issue, we propose Sheaf-DMFL, a novel decentralized multimodal learning framework leveraging sheaf theory to enhance collaboration among devices with diverse modalities. Specifically, each client has a set of local feature encoders for its different modalities, whose outputs are concatenated before passing through a task-specific layer. While encoders for the same modality are trained collaboratively across clients, we capture the intrinsic correlations among clients' task-specific layers using a sheaf-based structure. To further enhance learning capability, we propose an enhanced algorithm named Sheaf-DMFL-Att, which tailors the attention mechanism within each client to capture correlations among different modalities. A rigorous convergence analysis of Sheaf-DMFL-Att is provided, establishing its theoretical guarantees. Extensive simulations are conducted on real-world link blockage prediction and mmWave beamforming scenarios, demonstrate the superiority of the proposed algorithms in such heterogeneous wireless communication systems. 

**Abstract (ZH)**: 在大规模通信系统中，日益复杂的场景需要边缘设备收集各种多模态 sensory 数据进行更智能的合作，以实现环境的全面理解并提高决策准确度。然而，传统的联邦学习（FL）算法通常仅考虑单模态数据集，要求模型架构一致，并未能充分利用嵌入在多模态数据中的丰富信息，限制了其在具有多样模态和不同客户端能力的真实场景中的应用。为解决此问题，我们提出了一种名为 Sheaf-DMFL 的新型分散式多模态学习框架，利用 sheaf 理论增强具有不同模态设备之间的合作。具体而言，每个客户端对其不同的模态具有一个本地特征编码器集，其输出在经过任务特定层前被串联。虽然相同模态的编码器在客户端之间协同训练，我们使用基于 sheaf 的结构捕捉客户端任务特定层之间的内在相关性。为进一步增强学习能力，我们提出了名为 Sheaf-DMFL-Att 的增强算法，其中针对每个客户端尾随不同的模态间的相关性来定制注意机制。提供了 Sheaf-DMFL-Att 的严格收敛分析，建立了其理论保证。在实际的链接阻塞预测和毫米波波束形成场景中的广泛仿真实验表明，所提出的算法在异构无线通信系统中具有优越性。 

---
# From Ground to Air: Noise Robustness in Vision Transformers and CNNs for Event-Based Vehicle Classification with Potential UAV Applications 

**Title (ZH)**: 从地面到空中：基于事件的车辆分类中视觉变换器和CNN的噪声鲁棒性研究及其潜在的无人机应用 

**Authors**: Nouf Almesafri, Hector Figueiredo, Miguel Arana-Catania  

**Link**: [PDF](https://arxiv.org/pdf/2506.22360)  

**Abstract**: This study investigates the performance of the two most relevant computer vision deep learning architectures, Convolutional Neural Network and Vision Transformer, for event-based cameras. These cameras capture scene changes, unlike traditional frame-based cameras with capture static images, and are particularly suited for dynamic environments such as UAVs and autonomous vehicles. The deep learning models studied in this work are ResNet34 and ViT B16, fine-tuned on the GEN1 event-based dataset. The research evaluates and compares these models under both standard conditions and in the presence of simulated noise. Initial evaluations on the clean GEN1 dataset reveal that ResNet34 and ViT B16 achieve accuracies of 88% and 86%, respectively, with ResNet34 showing a slight advantage in classification accuracy. However, the ViT B16 model demonstrates notable robustness, particularly given its pre-training on a smaller dataset. Although this study focuses on ground-based vehicle classification, the methodologies and findings hold significant promise for adaptation to UAV contexts, including aerial object classification and event-based vision systems for aviation-related tasks. 

**Abstract (ZH)**: 本研究探讨了事件相机环境下两种最相关计算机视觉深度学习架构——卷积神经网络和视觉变压器的表现。这些相机捕获场景变化，而非传统帧基相机捕捉静态图像，特别适用于动态环境，如无人机和自动驾驶车辆。本研究中研究的深度学习模型为ResNet34和ViT B16，基于GEN1事件相机数据集进行了微调。研究在标准条件下和模拟噪声环境下评估并比较了这些模型的性能。对干净的GEN1数据集的初步评估结果显示，ResNet34和ViT B16分别实现了88%和86%的准确率，ResNet34在分类准确率上稍占优势。然而，ViT B16模型表现出了显著的鲁棒性，特别是考虑到其在较小数据集上的预训练。尽管本研究集中在地面车辆分类，但本研究的方法和发现对无人机场景具有重要意义，包括空中物体分类和与航空相关的事件驱动视觉系统。 

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
# A Deep Learning framework for building damage assessment using VHR SAR and geospatial data: demonstration on the 2023 Turkiye Earthquake 

**Title (ZH)**: 使用高分辨率 SAR 和地理空间数据的深学习框架在 2023 年土耳其地震损毁评估中的应用 

**Authors**: Luigi Russo, Deodato Tapete, Silvia Liberata Ullo, Paolo Gamba  

**Link**: [PDF](https://arxiv.org/pdf/2506.22338)  

**Abstract**: Building damage identification shortly after a disaster is crucial for guiding emergency response and recovery efforts. Although optical satellite imagery is commonly used for disaster mapping, its effectiveness is often hampered by cloud cover or the absence of pre-event acquisitions. To overcome these challenges, we introduce a novel multimodal deep learning (DL) framework for detecting building damage using single-date very high resolution (VHR) Synthetic Aperture Radar (SAR) imagery from the Italian Space Agency (ASI) COSMO SkyMed (CSK) constellation, complemented by auxiliary geospatial data. Our method integrates SAR image patches, OpenStreetMap (OSM) building footprints, digital surface model (DSM) data, and structural and exposure attributes from the Global Earthquake Model (GEM) to improve detection accuracy and contextual interpretation. Unlike existing approaches that depend on pre and post event imagery, our model utilizes only post event data, facilitating rapid deployment in critical scenarios. The framework effectiveness is demonstrated using a new dataset from the 2023 earthquake in Turkey, covering multiple cities with diverse urban settings. Results highlight that incorporating geospatial features significantly enhances detection performance and generalizability to previously unseen areas. By combining SAR imagery with detailed vulnerability and exposure information, our approach provides reliable and rapid building damage assessments without the dependency from available pre-event data. Moreover, the automated and scalable data generation process ensures the framework's applicability across diverse disaster-affected regions, underscoring its potential to support effective disaster management and recovery efforts. Code and data will be made available upon acceptance of the paper. 

**Abstract (ZH)**: 短灾后建筑损害识别对于指导紧急响应和恢复工作至关重要。尽管光学卫星影像常用于灾害映射，但其效果往往受云覆盖或缺乏事前影像的影响。为克服这些挑战，我们提出了一种新的多模态深度学习框架，利用意大利空间局（ASI）COSMO SkyMed（CSK）星座的单日期超高分辨率（VHR）合成孔径雷达（SAR）影像，并辅以辅助地理空间数据，来检测建筑损害。该方法整合了SAR影像patches、OpenStreetMap（OSM）建筑地块、数字表面模型（DSM）数据以及全球地震模型（GEM）的结构和暴露属性，以提高检测准确性和上下文解释能力。不同于依赖事前和事后影像的方法，我们的模型仅使用事后数据，便于在关键场景中快速部署。该框架的有效性通过2023年土耳其地震的新数据集进行了展示，涵盖了多个具有不同城市环境的地区。结果表明，纳入地理空间特征显著提升了检测性能和对未见过区域的泛化能力。通过结合SAR影像与详细的脆弱性和暴露信息，我们的方法能够在没有可用事前影像的情况下提供可靠的快速建筑损害评估。自动化的可扩展数据生成过程确保了该框架在各种灾害影响地区的适用性，突显了其在支持有效的灾害管理和恢复工作方面的潜力。代码和数据将在论文被接受后提供。 

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
# RoomCraft: Controllable and Complete 3D Indoor Scene Generation 

**Title (ZH)**: RoomCraft: 可控且完整的室内三维场景生成 

**Authors**: Mengqi Zhou, Xipeng Wang, Yuxi Wang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22291)  

**Abstract**: Generating realistic 3D indoor scenes from user inputs remains a challenging problem in computer vision and graphics, requiring careful balance of geometric consistency, spatial relationships, and visual realism. While neural generation methods often produce repetitive elements due to limited global spatial reasoning, procedural approaches can leverage constraints for controllable generation but struggle with multi-constraint scenarios. When constraints become numerous, object collisions frequently occur, forcing the removal of furniture items and compromising layout completeness.
To address these limitations, we propose RoomCraft, a multi-stage pipeline that converts real images, sketches, or text descriptions into coherent 3D indoor scenes. Our approach combines a scene generation pipeline with a constraint-driven optimization framework. The pipeline first extracts high-level scene information from user inputs and organizes it into a structured format containing room type, furniture items, and spatial relations. It then constructs a spatial relationship network to represent furniture arrangements and generates an optimized placement sequence using a heuristic-based depth-first search (HDFS) algorithm to ensure layout coherence. To handle complex multi-constraint scenarios, we introduce a unified constraint representation that processes both formal specifications and natural language inputs, enabling flexible constraint-oriented adjustments through a comprehensive action space design. Additionally, we propose a Conflict-Aware Positioning Strategy (CAPS) that dynamically adjusts placement weights to minimize furniture collisions and ensure layout completeness.
Extensive experiments demonstrate that RoomCraft significantly outperforms existing methods in generating realistic, semantically coherent, and visually appealing room layouts across diverse input modalities. 

**Abstract (ZH)**: Multi-Stage Pipeline for Generating Realistic 3D Indoor Scenes from User Inputs Through Constraint-Driven Optimization 

---
# Projected Compression: Trainable Projection for Efficient Transformer Compression 

**Title (ZH)**: 投影压缩：高效的变压器压缩的可训练投影 

**Authors**: Maciej Stefaniak, Michał Krutul, Jan Małaśnicki, Maciej Pióro, Jakub Krajewski, Sebastian Jaszczur, Marek Cygan, Kamil Adamczewski, Jan Ludziejewski  

**Link**: [PDF](https://arxiv.org/pdf/2506.22255)  

**Abstract**: Large language models have steadily increased in size to achieve improved performance; however, this growth has also led to greater inference time and computational demands. Consequently, there is rising interest in model size reduction methods. To address this issue, we propose Projected Compression, a novel model compression technique, that reduces model weights by utilizing projection modules. Specifically, we first train additional trainable projections weights and preserve access to all the original model parameters. Subsequently, these projections are merged into a lower-dimensional product matrix, resulting in a reduced-size standard Transformer-based model. Unlike alternative approaches that require additional computational overhead, our method matches the base model's per-token computation step in FLOPs. Experimental results show that Projected Compression outperforms the comparable hard pruning and retraining approach on higher quality models. Moreover, the performance margin scales well with the number of tokens. 

**Abstract (ZH)**: Projected Compression: A Novel Model Compression Technique for Reducing Inference Time and Computational Demands 

---
# Adapting University Policies for Generative AI: Opportunities, Challenges, and Policy Solutions in Higher Education 

**Title (ZH)**: 适应生成式AI的高校政策调整：高等教育中的机遇、挑战与政策解决方案 

**Authors**: Russell Beale  

**Link**: [PDF](https://arxiv.org/pdf/2506.22231)  

**Abstract**: The rapid proliferation of generative artificial intelligence (AI) tools - especially large language models (LLMs) such as ChatGPT - has ushered in a transformative era in higher education. Universities in developed regions are increasingly integrating these technologies into research, teaching, and assessment. On one hand, LLMs can enhance productivity by streamlining literature reviews, facilitating idea generation, assisting with coding and data analysis, and even supporting grant proposal drafting. On the other hand, their use raises significant concerns regarding academic integrity, ethical boundaries, and equitable access. Recent empirical studies indicate that nearly 47% of students use LLMs in their coursework - with 39% using them for exam questions and 7% for entire assignments - while detection tools currently achieve around 88% accuracy, leaving a 12% error margin. This article critically examines the opportunities offered by generative AI, explores the multifaceted challenges it poses, and outlines robust policy solutions. Emphasis is placed on redesigning assessments to be AI-resilient, enhancing staff and student training, implementing multi-layered enforcement mechanisms, and defining acceptable use. By synthesizing data from recent research and case studies, the article argues that proactive policy adaptation is imperative to harness AI's potential while safeguarding the core values of academic integrity and equity. 

**Abstract (ZH)**: 生成式人工智能工具的迅速普及——尤其是大型语言模型（LLMs）如ChatGPT——正在变革高等教育。发达国家的大学 increasingly将这些技术整合到科研、教学和评估中。一方面，LLMs可以通过简化文献回顾、促进创意生成、协助编程和数据分析，甚至支持资助提案撰写来提高生产力。另一方面，它们的使用引发了对学术诚信、伦理边界和公平获取的重大关切。近期实证研究表明，几乎有47%的学生在其作业中使用了LLMs——其中39%用于考试问题，7%用于整个作业——而当前检测工具的准确率约为88%，留下12%的误差率。本文批判性地探讨生成式AI提供的机遇，探索它所带来的多方面挑战，并提出切实可行的政策解决方案。重点在于重新设计评估体系以抵御AI威胁、提升教职员工和学生培训、实施多层次的执法机制以及界定合理使用范围。通过综合近期研究和案例研究的数据，本文主张，为了充分利用AI潜力并保障学术诚信和公平的核心价值，必须进行前瞻性的政策调整。 

---
# EFRame: Deeper Reasoning via Exploration-Filtering-Replay Reinforcement Learning Framework 

**Title (ZH)**: EFRame：通过探索-过滤-重播强化学习框架实现更深的推理 

**Authors**: Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Yue Wang, Yuzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22200)  

**Abstract**: Recent advances in reinforcement learning (RL) have significantly enhanced the reasoning capabilities of large language models (LLMs). Group Relative Policy Optimization (GRPO), an efficient variant of PPO that lowers RL's computational cost, still faces limited exploration, low sample efficiency and instability, constraining its performance on complex reasoning tasks. To address these limitations, we introduce EFRame, an Exploration-Filtering-Replay framework that systematically augments GRPO along three critical dimensions. EFRame performs additional rollouts to explore high-quality trajectories, applies online filtering to eliminate low-quality samples that introduce noise and variance, and leverages experience replay to repeatedly exploit rare but informative samples. EFRame establishes a complete and stable learning cycle, guiding the model through a structured transition from exploration to convergence. Our experiments across a variety of reasoning benchmarks demonstrate that EFRame not only improves the robustness and efficiency of training, but also enables access to deeper reasoning capabilities that remain unattainable under vanilla GRPO. Furthermore, EFRame enables a more fine-grained categorization of training samples, allowing for a deeper analysis of how different types of samples contribute to the learning process in RL. Our code is available at this https URL. 

**Abstract (ZH)**: Recent Advances in Exploration-Filtering-Replay for Enhancing Group Relative Policy Optimization in Reinforcement Learning 

---
# Autonomic Microservice Management via Agentic AI and MAPE-K Integration 

**Title (ZH)**: 自主微服务管理通过代理型AI与MAPE-K集成 

**Authors**: Matteo Esposito, Alexander Bakhtin, Noman Ahmad, Mikel Robredo, Ruoyu Su, Valentina Lenarduzzi, Davide Taibi  

**Link**: [PDF](https://arxiv.org/pdf/2506.22185)  

**Abstract**: While microservices are revolutionizing cloud computing by offering unparalleled scalability and independent deployment, their decentralized nature poses significant security and management challenges that can threaten system stability. We propose a framework based on MAPE-K, which leverages agentic AI, for autonomous anomaly detection and remediation to address the daunting task of highly distributed system management. Our framework offers practical, industry-ready solutions for maintaining robust and secure microservices. Practitioners and researchers can customize the framework to enhance system stability, reduce downtime, and monitor broader system quality attributes such as system performance level, resilience, security, and anomaly management, among others. 

**Abstract (ZH)**: 微服务通过提供无与伦比的可扩展性和独立部署能力正在革新云计算，但其分散的性质也带来了显著的安全和管理挑战，这些挑战可能威胁到系统的稳定性。我们提出了一种基于MAPE-K框架的方法，利用代理型AI实现自主异常检测与修复，以应对高度分布式系统管理这一艰巨任务。该框架提供了实用的、可应用于工业界的解决方案，以维护健壯和安全的微服务。实际应用者和研究者可以定制该框架以增强系统稳定性、减少停机时间，并监控诸如系统性能水平、弹性、安全性和异常管理等更广泛系统的质量属性。 

---
# Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition 

**Title (ZH)**: 频率-语义增强变分自编码器在零样本骨架动作识别中的应用 

**Authors**: Wenhan Wu, Zhishuai Guo, Chen Chen, Hongfei Xue, Aidong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22179)  

**Abstract**: Zero-shot skeleton-based action recognition aims to develop models capable of identifying actions beyond the categories encountered during training. Previous approaches have primarily focused on aligning visual and semantic representations but often overlooked the importance of fine-grained action patterns in the semantic space (e.g., the hand movements in drinking water and brushing teeth). To address these limitations, we propose a Frequency-Semantic Enhanced Variational Autoencoder (FS-VAE) to explore the skeleton semantic representation learning with frequency decomposition. FS-VAE consists of three key components: 1) a frequency-based enhancement module with high- and low-frequency adjustments to enrich the skeletal semantics learning and improve the robustness of zero-shot action recognition; 2) a semantic-based action description with multilevel alignment to capture both local details and global correspondence, effectively bridging the semantic gap and compensating for the inherent loss of information in skeleton sequences; 3) a calibrated cross-alignment loss that enables valid skeleton-text pairs to counterbalance ambiguous ones, mitigating discrepancies and ambiguities in skeleton and text features, thereby ensuring robust alignment. Evaluations on the benchmarks demonstrate the effectiveness of our approach, validating that frequency-enhanced semantic features enable robust differentiation of visually and semantically similar action clusters, improving zero-shot action recognition. 

**Abstract (ZH)**: 零样本基于骨架的动作识别旨在开发能够在训练中未遇到的类别中识别动作的模型。以往的方法主要集中在视觉表示和语义表示的对齐上，但往往忽视了语义空间中细粒度动作模式的重要性（如饮水和刷牙的手部动作）。为解决这些问题，我们提出了一种基于频率-语义增强的变分自编码器（FS-VAE）来探索通过频率分解进行的骨架语义表示学习。FS-VAE 包含三个关键组件：1）基于频率的增强模块，通过高低频调整丰富骨架语义学习并提高零样本动作识别的鲁棒性；2）基于语义的动作描述，采用多级对齐捕捉局部细节和全局对应关系，有效缩小语义鸿沟并补充骨架序列固有的信息损失；3）校准的交叉对齐损失，使有效的骨架-文本对能够抵消含糊不清的对齐，减轻骨架和文本特征之间的不一致性和模糊性，从而确保稳健的对齐。基准测试结果验证了该方法的有效性，表明频率增强的语义特征能够稳健地区分视觉上和语义上相似的动作簇，提高零样本动作识别的效果。 

---
# Visual Structures Helps Visual Reasoning: Addressing the Binding Problem in VLMs 

**Title (ZH)**: 视觉结构有助于视觉推理：解决VLM中的绑定问题 

**Authors**: Amirmohammad Izadi, Mohammad Ali Banayeeanzade, Fatemeh Askari, Ali Rahimiakbar, Mohammad Mahdi Vahedi, Hosein Hasani, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2506.22146)  

**Abstract**: Despite progress in Vision-Language Models (VLMs), their capacity for visual reasoning is often limited by the \textit{binding problem}: the failure to reliably associate perceptual features with their correct visual referents. This limitation underlies persistent errors in tasks such as counting, visual search, scene description, and spatial relationship understanding. A key factor is that current VLMs process visual features largely in parallel, lacking mechanisms for spatially grounded, serial attention. This paper introduces a simple yet effective intervention: augmenting visual inputs with low-level spatial structures (e.g., horizontal lines) and pairing this with a textual prompt that encourages sequential, spatially-aware parsing. We empirically demonstrate substantial performance improvements across core visual reasoning tasks. Specifically, our method improves GPT-4o visual search accuracy by 25.00%, increases counting accuracy by 26.83%, reduces edit distance error in scene description by 0.32, and enhances performance on spatial relationship tasks by 9.50% on a a 2D synthetic dataset. Furthermore, we find that the visual modification is essential for these gains; purely textual strategies, including Chain-of-Thought prompting, are insufficient and can even degrade performance. Our method enhances binding only with a single-query inference, underscoring the importance of visual input design over purely linguistically-based approaches. These findings suggest that low-level visual structuring is a powerful and underexplored direction for improving compositional visual reasoning and could serve as a general strategy for enhancing VLM performance on spatially grounded tasks. 

**Abstract (ZH)**: 尽管在视觉语言模型（VLMs）方面取得进展，但它们在视觉推理方面的能力仍受限于“绑定问题”：即感知特征与正确视觉实指之间的可靠关联失败。这一限制导致了计数、视觉搜索、场景描述和空间关系理解等任务中持续存在的错误。关键因素在于当前的VLMs主要以并行方式处理视觉特征，缺乏空间定位的序列注意力机制。本文介绍了一个简单而有效的干预方法：在视觉输入中加入低级空间结构（如水平线），并搭配一个文本提示，以促进序列化、空间意识的解析。我们实证展示了在核心视觉推理任务中性能显著提升。具体而言，我们的方法将GPT-4o的视觉搜索准确性提高了25.00%，计数准确性提高了26.83%，场景描述中的编辑距离误差降低了0.32，空间关系任务性能提高了9.50%（在2D合成数据集上）。此外，我们发现视觉修改对于这些提升至关重要；纯文本策略，包括链式思维提示，是不足的，甚至可能降低性能。我们的方法只需单次查询推理就能增强绑定，突显了视觉输入设计的重要性，而非纯粹基于语言的方法。这些发现表明，低级视觉结构化是一种强大而未被充分探索的途径，以改善组合视觉推理，并可能作为增强VLM在空间定位任务上性能的一般策略。 

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
# Literature-Grounded Novelty Assessment of Scientific Ideas 

**Title (ZH)**: 基于文献的新颖性评估方法 

**Authors**: Simra Shahid, Marissa Radensky, Raymond Fok, Pao Siangliulue, Daniel S. Weld, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2506.22026)  

**Abstract**: Automated scientific idea generation systems have made remarkable progress, yet the automatic evaluation of idea novelty remains a critical and underexplored challenge. Manual evaluation of novelty through literature review is labor-intensive, prone to error due to subjectivity, and impractical at scale. To address these issues, we propose the Idea Novelty Checker, an LLM-based retrieval-augmented generation (RAG) framework that leverages a two-stage retrieve-then-rerank approach. The Idea Novelty Checker first collects a broad set of relevant papers using keyword and snippet-based retrieval, then refines this collection through embedding-based filtering followed by facet-based LLM re-ranking. It incorporates expert-labeled examples to guide the system in comparing papers for novelty evaluation and in generating literature-grounded reasoning. Our extensive experiments demonstrate that our novelty checker achieves approximately 13% higher agreement than existing approaches. Ablation studies further showcases the importance of the facet-based re-ranker in identifying the most relevant literature for novelty evaluation. 

**Abstract (ZH)**: 基于LLM的检索增强生成框架：Idea Novelty Checker在自动化创新性评估中的应用 

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
# SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model 

**Title (ZH)**: SceneDiffuser++: 城市规模交通仿真通过生成世界模型 

**Authors**: Shuhan Tan, John Lambert, Hong Jeon, Sakshum Kulshrestha, Yijing Bai, Jing Luo, Dragomir Anguelov, Mingxing Tan, Chiyu Max Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21976)  

**Abstract**: The goal of traffic simulation is to augment a potentially limited amount of manually-driven miles that is available for testing and validation, with a much larger amount of simulated synthetic miles. The culmination of this vision would be a generative simulated city, where given a map of the city and an autonomous vehicle (AV) software stack, the simulator can seamlessly simulate the trip from point A to point B by populating the city around the AV and controlling all aspects of the scene, from animating the dynamic agents (e.g., vehicles, pedestrians) to controlling the traffic light states. We refer to this vision as CitySim, which requires an agglomeration of simulation technologies: scene generation to populate the initial scene, agent behavior modeling to animate the scene, occlusion reasoning, dynamic scene generation to seamlessly spawn and remove agents, and environment simulation for factors such as traffic lights. While some key technologies have been separately studied in various works, others such as dynamic scene generation and environment simulation have received less attention in the research community. We propose SceneDiffuser++, the first end-to-end generative world model trained on a single loss function capable of point A-to-B simulation on a city scale integrating all the requirements above. We demonstrate the city-scale traffic simulation capability of SceneDiffuser++ and study its superior realism under long simulation conditions. We evaluate the simulation quality on an augmented version of the Waymo Open Motion Dataset (WOMD) with larger map regions to support trip-level simulation. 

**Abstract (ZH)**: 交通模拟的目标是用大量的模拟合成英里来补充有限的手动驾驶英里，以供测试和验证使用。这一愿景的最终结果是一个生成的模拟城市，在给定城市地图和自动驾驶车辆（AV）软件栈的情况下，模拟器可以通过填充城市中的自动驾驶车辆并控制场景的所有方面，从动画动态代理（如车辆、行人）到控制交通灯状态，无缝地模拟从A点到B点的旅程。我们将其称为CitySim，这需要综合各种模拟技术：场景生成以填充初始场景、代理行为建模以动画场景、遮挡推理、动态场景生成以无缝生成和移除代理，以及环境模拟以考虑如交通灯等因素。虽然一些关键技术已在各种研究中分别进行过研究，但其他技术如动态场景生成和环境模拟在研究社区中受到的关注较少。我们提出了SceneDiffuser++，这是一个通过单一损失函数训练的第一个端到端生成世界模型，能够在城市尺度上集成所有上述要求进行A到B的模拟。我们展示了SceneDiffuser++的城市规模交通模拟能力，并在长时间模拟条件下研究了其更出色的现实感。我们使用增强后的Waymo Open Motion数据集（WOMD）进行评估，该数据集支持更大地图区域以支持行程级模拟。 

---
# Advancing Jailbreak Strategies: A Hybrid Approach to Exploiting LLM Vulnerabilities and Bypassing Modern Defenses 

**Title (ZH)**: 推进 Jailbreak 策略：一种利用大型语言模型漏洞并绕过现代防御的混合方法 

**Authors**: Mohamed Ahmed, Mohamed Abdelmouty, Mingyu Kim, Gunvanth Kandula, Alex Park, James C. Davis  

**Link**: [PDF](https://arxiv.org/pdf/2506.21972)  

**Abstract**: The advancement of Pre-Trained Language Models (PTLMs) and Large Language Models (LLMs) has led to their widespread adoption across diverse applications. Despite their success, these models remain vulnerable to attacks that exploit their inherent weaknesses to bypass safety measures. Two primary inference-phase threats are token-level and prompt-level jailbreaks. Token-level attacks embed adversarial sequences that transfer well to black-box models like GPT but leave detectable patterns and rely on gradient-based token optimization, whereas prompt-level attacks use semantically structured inputs to elicit harmful responses yet depend on iterative feedback that can be unreliable. To address the complementary limitations of these methods, we propose two hybrid approaches that integrate token- and prompt-level techniques to enhance jailbreak effectiveness across diverse PTLMs. GCG + PAIR and the newly explored GCG + WordGame hybrids were evaluated across multiple Vicuna and Llama models. GCG + PAIR consistently raised attack-success rates over its constituent techniques on undefended models; for instance, on Llama-3, its Attack Success Rate (ASR) reached 91.6%, a substantial increase from PAIR's 58.4% baseline. Meanwhile, GCG + WordGame matched the raw performance of WordGame maintaining a high ASR of over 80% even under stricter evaluators like Mistral-Sorry-Bench. Crucially, both hybrids retained transferability and reliably pierced advanced defenses such as Gradient Cuff and JBShield, which fully blocked single-mode attacks. These findings expose previously unreported vulnerabilities in current safety stacks, highlight trade-offs between raw success and defensive robustness, and underscore the need for holistic safeguards against adaptive adversaries. 

**Abstract (ZH)**: 预训练语言模型和大型语言模型的进步推动了它们在各种应用中的广泛应用。尽管这些模型取得了成功，但仍易受到利用其固有弱点的攻击，从而绕过安全措施。两种主要的推理阶段威胁是tokens层面和prompt层面的劫持攻击。tokens层面的攻击通过嵌入对抗序列来实现，这些序列在如GPT的黑盒模型中表现良好，但会产生可检测的模式，并依赖于基于梯度的token优化，而prompt层面的攻击使用语义结构化的输入来触发有害响应，但依赖于迭代反馈，这种方法可能不可靠。为了弥补这两种方法的互补局限性，我们提出了一种结合tokens层面和prompt层面技术的混合方法，以增强不同预训练语言模型的劫持攻击效果。GCG + PAIR和新探索的GCG + WordGame混合方法在多个Vicuna和Llama模型上进行了评估。GCG + PAIR在未防御模型中的一贯攻击成功率高于其组成部分技术，例如，在Llama-3上，其攻击成功率为91.6%，远高于PAIR的基线58.4%。同时，GCG + WordGame在更严格的评估器如Mistral-Sorry-Bench下，保持了高水平的攻击成功率超过80%，与WordGame的原始性能相当。关键的是，这两种混合方法都保持了可移植性，并能可靠地突破Gradient Cuff和JBShield等高级防御，这些防御完全阻止了一类模式的攻击。这些发现揭示了当前安全堆栈中的未报告漏洞，强调了在面对适应性对手时原始成功与防御鲁棒性之间权衡的重要性，并突显了需要整体防护措施。 

---
# Using Large Language Models to Suggest Informative Prior Distributions in Bayesian Statistics 

**Title (ZH)**: 使用大型语言模型为贝叶斯统计建议信息性先验分布 

**Authors**: Michael A. Riegler, Kristoffer Herland Hellton, Vajira Thambawita, Hugo L. Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2506.21964)  

**Abstract**: Selecting prior distributions in Bayesian statistics is challenging, resource-intensive, and subjective. We analyze using large-language models (LLMs) to suggest suitable, knowledge-based informative priors. We developed an extensive prompt asking LLMs not only to suggest priors but also to verify and reflect on their choices.
We evaluated Claude Opus, Gemini 2.5 Pro, and ChatGPT-4o-mini on two real datasets: heart disease risk and concrete strength. All LLMs correctly identified the direction for all associations (e.g., that heart disease risk is higher for males). The quality of suggested priors was measured by their Kullback-Leibler divergence from the maximum likelihood estimator's distribution.
The LLMs suggested both moderately and weakly informative priors. The moderate priors were often overconfident, resulting in distributions misaligned with the data. In our experiments, Claude and Gemini provided better priors than ChatGPT. For weakly informative priors, a key performance difference emerged: ChatGPT and Gemini defaulted to an "unnecessarily vague" mean of 0, while Claude did not, demonstrating a significant advantage.
The ability of LLMs to identify correct associations shows their great potential as an efficient, objective method for developing informative priors. However, the primary challenge remains in calibrating the width of these priors to avoid over- and under-confidence. 

**Abstract (ZH)**: 使用大型语言模型在贝叶斯统计中选择先验分布：挑战、评估与发现 

---
# SDRNET: Stacked Deep Residual Network for Accurate Semantic Segmentation of Fine-Resolution Remotely Sensed Images 

**Title (ZH)**: SDRNET：堆叠深度残差网络用于高分辨率遥感图像准确语义分割 

**Authors**: Naftaly Wambugu, Ruisheng Wang, Bo Guo, Tianshu Yu, Sheng Xu, Mohammed Elhassan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21945)  

**Abstract**: Land cover maps generated from semantic segmentation of high-resolution remotely sensed images have drawn mucon in the photogrammetry and remote sensing research community. Currently, massive fine-resolution remotely sensed (FRRS) images acquired by improving sensing and imaging technologies become available. However, accurate semantic segmentation of such FRRS images is greatly affected by substantial class disparities, the invisibility of key ground objects due to occlusion, and object size variation. Despite the extraordinary potential in deep convolutional neural networks (DCNNs) in image feature learning and representation, extracting sufficient features from FRRS images for accurate semantic segmentation is still challenging. These challenges demand the deep learning models to learn robust features and generate sufficient feature descriptors. Specifically, learning multi-contextual features to guarantee adequate coverage of varied object sizes from the ground scene and harnessing global-local contexts to overcome class disparities challenge even profound networks. Deeper networks significantly lose spatial details due to gradual downsampling processes resulting in poor segmentation results and coarse boundaries. This article presents a stacked deep residual network (SDRNet) for semantic segmentation from FRRS images. The proposed framework utilizes two stacked encoder-decoder networks to harness long-range semantics yet preserve spatial information and dilated residual blocks (DRB) between each encoder and decoder network to capture sufficient global dependencies thus improving segmentation performance. Our experimental results obtained using the ISPRS Vaihingen and Potsdam datasets demonstrate that the SDRNet performs effectively and competitively against current DCNNs in semantic segmentation. 

**Abstract (ZH)**: 基于高分辨率遥感图像语义分割的语义覆盖地图生成在摄影测量与遥感研究领域引起了广泛关注。尽管改进感测和成像技术获得的高分辨率细粒度遥感（FRRS）图像变得可用，但准确的语义分割仍受类别不均衡、关键地面对象因遮挡而不可见以及物体大小变化等挑战的影响。尽管深度卷积神经网络（DCNNs）在图像特征学习和表示方面具有非凡潜力，但从中提取足够的特征进行准确的语义分割仍然具有挑战性。这些挑战要求深度学习模型学习稳健的特征并生成足够的特征描述符。具体而言，学习多上下文特征以确保不同物体大小的充分覆盖，并利用全局-局部上下文来克服类别不均衡的挑战，即使是深层网络也不例外。由于逐级下采样过程导致空间细节大量丢失，深层网络会产生较差的分割结果和粗糙的边界。本文提出了一种堆叠深度残差网络（SDRNet）用于FRRS图像的语义分割。所提出的框架利用两个堆叠的编码-解码网络来捕获长范围语义同时保留空间信息，并在每个编码器和解码器网络之间使用膨胀残差块（DRB）来捕捉足够的全局依赖性，从而提高分割性能。使用ISPRS Vaihingen和Potsdam数据集进行的实验结果表明，SDRNet 在语义分割方面能够有效地与当前的DCNNs竞争。 

---
# ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation 

**Title (ZH)**: ARAG：自主检索增强生成的个性化推荐 

**Authors**: Reza Yousefi Maragheh, Pratheek Vadla, Priyank Gupta, Kai Zhao, Aysenur Inan, Kehui Yao, Jianpeng Xu, Praveen Kanumala, Jason Cho, Sushant Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21931)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown promise in enhancing recommendation systems by incorporating external context into large language model prompts. However, existing RAG-based approaches often rely on static retrieval heuristics and fail to capture nuanced user preferences in dynamic recommendation scenarios. In this work, we introduce ARAG, an Agentic Retrieval-Augmented Generation framework for Personalized Recommendation, which integrates a multi-agent collaboration mechanism into the RAG pipeline. To better understand the long-term and session behavior of the user, ARAG leverages four specialized LLM-based agents: a User Understanding Agent that summarizes user preferences from long-term and session contexts, a Natural Language Inference (NLI) Agent that evaluates semantic alignment between candidate items retrieved by RAG and inferred intent, a context summary agent that summarizes the findings of NLI agent, and an Item Ranker Agent that generates a ranked list of recommendations based on contextual fit. We evaluate ARAG accross three datasets. Experimental results demonstrate that ARAG significantly outperforms standard RAG and recency-based baselines, achieving up to 42.1% improvement in NDCG@5 and 35.5% in Hit@5. We also, conduct an ablation study to analyse the effect by different components of ARAG. Our findings highlight the effectiveness of integrating agentic reasoning into retrieval-augmented recommendation and provide new directions for LLM-based personalization. 

**Abstract (ZH)**: 基于代理增强检索的个性化推荐框架（ARAG） 

---
# SODA: Out-of-Distribution Detection in Domain-Shifted Point Clouds via Neighborhood Propagation 

**Title (ZH)**: SODA: 在领域偏移点云中通过邻域传播进行异常分布检测 

**Authors**: Adam Goodge, Xun Xu, Bryan Hooi, Wee Siong Ng, Jingyi Liao, Yongyi Su, Xulei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21892)  

**Abstract**: As point cloud data increases in prevalence in a variety of applications, the ability to detect out-of-distribution (OOD) point cloud objects becomes critical for ensuring model safety and reliability. However, this problem remains under-explored in existing research. Inspired by success in the image domain, we propose to exploit advances in 3D vision-language models (3D VLMs) for OOD detection in point cloud objects. However, a major challenge is that point cloud datasets used to pre-train 3D VLMs are drastically smaller in size and object diversity than their image-based counterparts. Critically, they often contain exclusively computer-designed synthetic objects. This leads to a substantial domain shift when the model is transferred to practical tasks involving real objects scanned from the physical environment. In this paper, our empirical experiments show that synthetic-to-real domain shift significantly degrades the alignment of point cloud with their associated text embeddings in the 3D VLM latent space, hindering downstream performance. To address this, we propose a novel methodology called SODA which improves the detection of OOD point clouds through a neighborhood-based score propagation scheme. SODA is inference-based, requires no additional model training, and achieves state-of-the-art performance over existing approaches across datasets and problem settings. 

**Abstract (ZH)**: 随着点云数据在各种应用中日益增多，检测点云离群对象的能力对于确保模型安全性和可靠性变得至关重要。然而，这一问题在现有研究中仍被忽视。受图像领域成功经验的启发，我们提出利用三维视觉-语言模型（3D VLMs）的进步来解决点云离群检测问题。然而，一个主要挑战是，用于预训练3D VLMs的点云数据集在大小和对象多样性方面远不如基于图像的对应物。关键的是，这些数据集往往仅包含计算机设计的合成对象。这导致当模型转移到涉及实物环境扫描对象的实际任务时，存在显著的数据域迁移问题。本文通过实验证明，合成到现实的数据域迁移严重破坏了点云与其关联文本嵌入在3D VLM潜在空间中的对齐，从而妨碍下游性能。为了解决这一问题，我们提出了一种名为SODA的新型方法，通过基于邻域的分数传播方案来提高点云离群检测的效果。SODA是一种推理方法，无需额外的模型训练，并在不同数据集和问题设置上实现了现有方法的最先进性能。 

---
# UnMix-NeRF: Spectral Unmixing Meets Neural Radiance Fields 

**Title (ZH)**: UnMix-NeRF：光谱解混与神经辐射场的结合 

**Authors**: Fabian Perez, Sara Rojas, Carlos Hinojosa, Hoover Rueda-Chacón, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2506.21884)  

**Abstract**: Neural Radiance Field (NeRF)-based segmentation methods focus on object semantics and rely solely on RGB data, lacking intrinsic material properties. This limitation restricts accurate material perception, which is crucial for robotics, augmented reality, simulation, and other applications. We introduce UnMix-NeRF, a framework that integrates spectral unmixing into NeRF, enabling joint hyperspectral novel view synthesis and unsupervised material segmentation. Our method models spectral reflectance via diffuse and specular components, where a learned dictionary of global endmembers represents pure material signatures, and per-point abundances capture their distribution. For material segmentation, we use spectral signature predictions along learned endmembers, allowing unsupervised material clustering. Additionally, UnMix-NeRF enables scene editing by modifying learned endmember dictionaries for flexible material-based appearance manipulation. Extensive experiments validate our approach, demonstrating superior spectral reconstruction and material segmentation to existing methods. Project page: this https URL. 

**Abstract (ZH)**: 基于Neural Radiance Field (NeRF)的分段方法侧重于对象语义并仅依赖于RGB数据，缺乏内在材料特性。这一局限性限制了对材料的准确感知，这对机器人技术、增强现实、模拟及其他应用至关重要。我们引入了UnMix-NeRF框架，将光谱解混合集成到NeRF中，实现联合超光谱新视角合成和无监督材料分段。我们的方法通过漫反射和镜面反射成分建模光谱反射率，其中学习到的全局端元字典代表纯材料特征，每个点的丰度捕捉它们的分布。在材料分段方面，我们利用学习到端元的光谱特征进行无监督材料聚类。此外，UnMix-NeRF 通过修改学习到的端元字典来实现场景编辑，从而灵活地进行基于材料的外观操作。大量实验验证了我们的方法，展示了其在光谱重建和材料分段方面的优越性能。项目页面: [点击此处](this https URL)。 

---
# Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation 

**Title (ZH)**: 视觉-语言模型拥有内部世界模型吗？向原子化评估迈进 

**Authors**: Qiyue Gao, Xinyu Pi, Kevin Liu, Junrong Chen, Ruolan Yang, Xinqi Huang, Xinyu Fang, Lu Sun, Gautham Kishore, Bo Ai, Stone Tao, Mengyang Liu, Jiaxi Yang, Chao-Jung Lai, Chuanyang Jin, Jiannan Xiang, Benhao Huang, Zeming Chen, David Danks, Hao Su, Tianmin Shu, Ziqiao Ma, Lianhui Qin, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21876)  

**Abstract**: Internal world models (WMs) enable agents to understand the world's state and predict transitions, serving as the basis for advanced deliberative reasoning. Recent large Vision-Language Models (VLMs), such as OpenAI o3, GPT-4o and Gemini, exhibit potential as general-purpose WMs. While the latest studies have evaluated and shown limitations in specific capabilities such as visual understanding, a systematic evaluation of VLMs' fundamental WM abilities remains absent. Drawing on comparative psychology and cognitive science, we propose a two-stage framework that assesses Perception (visual, spatial, temporal, quantitative, and motion) and Prediction (mechanistic simulation, transitive inference, compositional inference) to provide an atomic evaluation of VLMs as WMs. Guided by this framework, we introduce WM-ABench, a large-scale benchmark comprising 23 fine-grained evaluation dimensions across 6 diverse simulated environments with controlled counterfactual simulations. Through 660 experiments on 15 latest commercial and open-source VLMs, we find that these models exhibit striking limitations in basic world modeling abilities. For instance, almost all models perform at near-random accuracy when distinguishing motion trajectories. Additionally, they lack disentangled understanding -- e.g., some models tend to believe blue objects move faster than green ones. More rich results and analyses reveal significant gaps between VLMs and human-level world modeling. 

**Abstract (ZH)**: 基于内部世界模型的Vision-Language模型系统的原子性评估：从感知到预测 

---
# On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling 

**Title (ZH)**: 基于对抗性误标签的文本到图像AI模型中毒可行性研究 

**Authors**: Stanley Wu, Ronik Bhaskar, Anna Yoo Jeong Ha, Shawn Shan, Haitao Zheng, Ben Y. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21874)  

**Abstract**: Today's text-to-image generative models are trained on millions of images sourced from the Internet, each paired with a detailed caption produced by Vision-Language Models (VLMs). This part of the training pipeline is critical for supplying the models with large volumes of high-quality image-caption pairs during training. However, recent work suggests that VLMs are vulnerable to stealthy adversarial attacks, where adversarial perturbations are added to images to mislead the VLMs into producing incorrect captions.
In this paper, we explore the feasibility of adversarial mislabeling attacks on VLMs as a mechanism to poisoning training pipelines for text-to-image models. Our experiments demonstrate that VLMs are highly vulnerable to adversarial perturbations, allowing attackers to produce benign-looking images that are consistently miscaptioned by the VLM models. This has the effect of injecting strong "dirty-label" poison samples into the training pipeline for text-to-image models, successfully altering their behavior with a small number of poisoned samples. We find that while potential defenses can be effective, they can be targeted and circumvented by adaptive attackers. This suggests a cat-and-mouse game that is likely to reduce the quality of training data and increase the cost of text-to-image model development. Finally, we demonstrate the real-world effectiveness of these attacks, achieving high attack success (over 73%) even in black-box scenarios against commercial VLMs (Google Vertex AI and Microsoft Azure). 

**Abstract (ZH)**: 今天用于文本到图像生成的模型是在互联网上获取的数百万张图像上训练的，每张图像都配有一个由视觉语言模型（VLMs）生成的详细描述。这一部分的训练管道对于向模型提供大量高质量的图像描述配对至关重要。然而，近期的研究表明，VLMs容易受到隐蔽的 adversarial 攻击，即通过在图像中添加对抗性扰动来误导 VLMs 生成错误的描述。

在本文中，我们探讨了利用对抗性误标记攻击 VLMs 作为污染文本到图像模型训练管道的一种机制的可能性。我们的实验表明，VLMs 对对抗性扰动极为敏感，攻击者可以生成看似无害的图像，但这些图像始终会被 VLM 模型错误地描述。这导致为文本到图像模型训练管道注入了强大的“脏标签”污染样本，少量污染样本就能成功改变其行为。我们发现，尽管潜在的防御措施可能是有效的，但攻击者可以通过适应性手段来针对并规避这些防御。这表明一个“猫捉老鼠”的游戏可能会降低训练数据的质量，并增加文本到图像模型开发的成本。最后，我们展示了这些攻击在现实世界中的有效性，在商业 VLMs（Google Vertex AI 和 Microsoft Azure）的黑盒场景中，成功攻击率高达 73%。 

---
# Grounding-Aware Token Pruning: Recovering from Drastic Performance Drops in Visual Grounding Caused by Pruning 

**Title (ZH)**: 基于地面性的Token裁剪：在裁剪导致视觉地面性性能急剧下降后的恢复方法 

**Authors**: Tzu-Chun Chien, Chieh-Kai Lin, Shiang-Feng Tsai, Ruei-Chi Lai, Hung-Jen Chen, Min Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.21873)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) have demonstrated strong performance in visual grounding, establishing themselves as a general interface for various vision-language applications. This progress has driven the development of token pruning methods to mitigate the high computational costs associated with processing numerous visual tokens. However, we observe that pruning significantly weakens the model's grounding ability, leading to incorrect predictions and drastic performance degradation. In Referring Expression Comprehension (REC), for instance, pruning causes the accuracy of LLaVA on the RefCOCO validation set to drop from 56.14% to 15.34%. Our analysis identifies misaligned position IDs after pruning as the primary cause of this degradation, as both the order and value of these IDs are crucial for maintaining performance in grounding tasks. To address this issue, we propose Grounding-Aware Token Pruning (GAP), a simple yet effective adjustment to position IDs that recovers REC accuracy back to 51.42%, which is 90% of the original performance in the without pruning setting, all while requiring no additional training, memory, or computational overhead. Applied to models such as Shikra, MiniGPTv2, and the LLaVA series, our method consistently improves performance across various token pruning strategies. 

**Abstract (ZH)**: Recent 多模态大语言模型 (MLLMs) 在视觉定位任务中的表现强烈证明了它们作为各类视觉-语言应用通用接口的优势。这一进展推动了消除处理大量视觉标记高计算成本的标记剪枝方法的发展。然而，我们观察到剪枝显著削弱了模型的视觉定位能力，导致错误预测和性能急剧下降。例如，在参照表达理解 (REC) 中，剪枝导致 LLaVA 在 RefCOCO 验证集上的准确率从 56.14% 降至 15.34%。我们的分析指出，剪枝后位置 ID 的不对齐是导致性能下降的主要原因，因为这些 ID 的顺序和值对于保持定位任务中的性能至关重要。为了解决这一问题，我们提出了一种简单而有效的定位感知标记剪枝 (GAP) 方法，该方法通过调整位置 ID 回复了 REC 准确率至 51.42%，相当于剪枝前性能的 90%，且无需额外训练、内存或计算开销。我们将该方法应用于 Shikra、MiniGPTv2 和 LLaVA 系列模型，在各种标记剪枝策略中均实现了性能提升。 

---
# A Survey of Continual Reinforcement Learning 

**Title (ZH)**: 持续强化学习综述 

**Authors**: Chaofan Pan, Xin Yang, Yanhua Li, Wei Wei, Tianrui Li, Bo An, Jiye Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21872)  

**Abstract**: Reinforcement Learning (RL) is an important machine learning paradigm for solving sequential decision-making problems. Recent years have witnessed remarkable progress in this field due to the rapid development of deep neural networks. However, the success of RL currently relies on extensive training data and computational resources. In addition, RL's limited ability to generalize across tasks restricts its applicability in dynamic and real-world environments. With the arisen of Continual Learning (CL), Continual Reinforcement Learning (CRL) has emerged as a promising research direction to address these limitations by enabling agents to learn continuously, adapt to new tasks, and retain previously acquired knowledge. In this survey, we provide a comprehensive examination of CRL, focusing on its core concepts, challenges, and methodologies. Firstly, we conduct a detailed review of existing works, organizing and analyzing their metrics, tasks, benchmarks, and scenario settings. Secondly, we propose a new taxonomy of CRL methods, categorizing them into four types from the perspective of knowledge storage and/or transfer. Finally, our analysis highlights the unique challenges of CRL and provides practical insights into future directions. 

**Abstract (ZH)**: 持续强化学习（CRL）：核心概念、挑战与方法綜述 

---
# DeepTalk: Towards Seamless and Smart Speech Interaction with Adaptive Modality-Specific MoE 

**Title (ZH)**: DeepTalk: 向无缝和智能的自适应模态特定MOE语音交互迈进 

**Authors**: Hang Shao, Heting Gao, Yunhang Shen, Jiawei Chen, Lijiang Li, Zuwei Long, Bo Tong, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.21864)  

**Abstract**: Native multimodal large language models (MLLMs) restructure a single large language model (LLM) into a spoken language model (SLM) capable of both speech and text generation. Compared to modular and aligned MLLMs, native MLLMs preserve richer paralinguistic features such as emotion and prosody, and generate speech responses directly within the backbone LLM rather than using a separate speech decoder. This integration also results in lower response latency and smoother interaction. However, native MLLMs suffer from catastrophic forgetting and performance degradation because the available paired speech-text data is insufficient to support the pretraining of MLLMs compared to the vast amount of text data required to pretrain text LLMs. To address this issue, we propose DeepTalk, a framework for adaptive modality expert learning based on a Mixture of Experts (MoE) architecture. DeepTalk first adaptively distinguishes modality experts according to their modality load within the LLM. Each modality expert then undergoes specialized single-modality training, followed by joint multimodal collaborative training. As a result, DeepTalk incurs only a 5.5% performance drop compared to the original LLM, which is significantly lower than the average performance drop of over 20% typically seen in native MLLMs (such as GLM-4-Voice), and is on par with modular MLLMs. Meanwhile, the end-to-end dialogue latency remains within 0.5 seconds, ensuring a seamless and intelligent speech interaction experience. Code and models are released at this https URL. 

**Abstract (ZH)**: 原生多模态大语言模型 (Native MLLMs) 将单一的大语言模型 (LLM) 重新构建成既能生成文字又能生成语音的语音语言模型 (SLM)。与模块化和对齐的 MLLMs 相比，原生 MLLMs 保留了更多的副语言特征，如情感和语调，并能够在骨干 LLM 内直接生成语音响应，而无需使用单独的语音解码器。这种集成还导致响应延迟降低和交互更为流畅。然而，原生 MLLMs 因为可用的配对语音-文本数据不足而遭受灾难性遗忘和性能下降，支持 MLLMs 预训练的数据量远少于支持文本 LLMs 预训练所需的巨大文本数据量。为了解决这一问题，我们提出 DeepTalk 框架，这是一种基于Experts混合架构（MoE）的自适应模态专家学习框架。DeepTalk 首先根据 LLM 中的模态负载自适应地区分模态专家，然后对每个模态专家进行专门的单模态训练，之后进行联合多模态协作训练。结果表明，与原生 MLLMs（如 GLM-4-Voice）相比，DeepTalk 的性能下降仅为 5.5%，显著低于模块化 MLLMs 平均超过 20% 的性能下降水平。同时，端到端对话延迟保持在 0.5 秒以内，确保了无缝和智能化的语音交互体验。代码和模型已发布于此网址。 

---
# LLaVA-Scissor: Token Compression with Semantic Connected Components for Video LLMs 

**Title (ZH)**: LLaVA-Scissor: 基于语义连接组件的tokens压缩方法用于视频LLMs 

**Authors**: Boyuan Sun, Jiaxing Zhao, Xihan Wei, Qibin Hou  

**Link**: [PDF](https://arxiv.org/pdf/2506.21862)  

**Abstract**: In this paper, we present LLaVA-Scissor, a training-free token compression strategy designed for video multimodal large language models. Previous methods mostly attempt to compress tokens based on attention scores, but fail to effectively capture all semantic regions and often lead to token redundancy. Differently, we propose to leverage the Semantic Connected Components (SCC) approach that assigns tokens to distinct semantic regions within the token set, ensuring comprehensive semantic coverage. The outcome is a two-step spatio-temporal token compression strategy that utilizes SCC in both spatial and temporal domains. This strategy can effectively compress tokens by representing the entire video with a set of non-overlapping semantic tokens. We conduct extensive evaluations of the token compression capabilities of LLaVA-Scissor across diverse video understanding benchmarks, including video question answering, long video understanding, and comprehensive multi-choices benchmarks. Experimental results show that the proposed LLaVA-Scissor outperforms other token compression methods, achieving superior performance in various video understanding benchmarks, particularly at low token retention ratios. Project page: this https URL. 

**Abstract (ZH)**: 本文提出LLaVA-Scissor，这是一种无需训练的token压缩策略，旨在用于视频多模态大型语言模型。以往方法主要尝试根据注意分数来压缩token，但未能有效捕捉所有语义区域，并且通常会导致token冗余。不同的是，我们提出利用语义连通组件（SCC）方法，将token分配到token集合中的不同语义区域，确保全面的语义覆盖。该策略是一种两步时空token压缩策略，在空间域和时间域均利用SCC。该策略通过使用一组不重叠的语义token来表示整个视频，从而有效进行token压缩。我们在包括视频问答、长视频理解和综合性多选择基准等多种视频理解基准上对LLaVA-Scissor的token压缩能力进行了广泛评估。实验结果表明，所提出的LLaVA-Scissor优于其他token压缩方法，在各种视频理解基准上表现出更优性能，特别是在低token保留率情况下。项目页面：this https URL。 

---
# SPADE: Spatial Transcriptomics and Pathology Alignment Using a Mixture of Data Experts for an Expressive Latent Space 

**Title (ZH)**: SPADE: 空间转录组学和病理学对齐的混合数据专家方法用于具表现力的潜在空间 

**Authors**: Ekaterina Redekop, Mara Pleasure, Zichen Wang, Kimberly Flores, Anthony Sisk, William Speier, Corey W. Arnold  

**Link**: [PDF](https://arxiv.org/pdf/2506.21857)  

**Abstract**: The rapid growth of digital pathology and advances in self-supervised deep learning have enabled the development of foundational models for various pathology tasks across diverse diseases. While multimodal approaches integrating diverse data sources have emerged, a critical gap remains in the comprehensive integration of whole-slide images (WSIs) with spatial transcriptomics (ST), which is crucial for capturing critical molecular heterogeneity beyond standard hematoxylin & eosin (H&E) staining. We introduce SPADE, a foundation model that integrates histopathology with ST data to guide image representation learning within a unified framework, in effect creating an ST-informed latent space. SPADE leverages a mixture-of-data experts technique, where experts, created via two-stage feature-space clustering, use contrastive learning to learn representations of co-registered WSI patches and gene expression profiles. Pre-trained on the comprehensive HEST-1k dataset, SPADE is evaluated on 14 downstream tasks, demonstrating significantly superior few-shot performance compared to baseline models, highlighting the benefits of integrating morphological and molecular information into one latent space. 

**Abstract (ZH)**: SPADE：一种结合组织病理学与空间转录组学的数据专家混合基础模型 

---
# The Consistency Hypothesis in Uncertainty Quantification for Large Language Models 

**Title (ZH)**: 不确定性量化中的大语言模型一致性假设 

**Authors**: Quan Xiao, Debarun Bhattacharjya, Balaji Ganesan, Radu Marinescu, Katsiaryna Mirylenka, Nhan H Pham, Michael Glass, Junkyu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.21849)  

**Abstract**: Estimating the confidence of large language model (LLM) outputs is essential for real-world applications requiring high user trust. Black-box uncertainty quantification (UQ) methods, relying solely on model API access, have gained popularity due to their practical benefits. In this paper, we examine the implicit assumption behind several UQ methods, which use generation consistency as a proxy for confidence, an idea we formalize as the consistency hypothesis. We introduce three mathematical statements with corresponding statistical tests to capture variations of this hypothesis and metrics to evaluate LLM output conformity across tasks. Our empirical investigation, spanning 8 benchmark datasets and 3 tasks (question answering, text summarization, and text-to-SQL), highlights the prevalence of the hypothesis under different settings. Among the statements, we highlight the `Sim-Any' hypothesis as the most actionable, and demonstrate how it can be leveraged by proposing data-free black-box UQ methods that aggregate similarities between generations for confidence estimation. These approaches can outperform the closest baselines, showcasing the practical value of the empirically observed consistency hypothesis. 

**Abstract (ZH)**: 估计大型语言模型(LLM)输出的置信度对于需要高度用户信任的实际应用至关重要。基于模型API的黑盒不确定性量化(UQ)方法因其实际优势而日益流行。本文探讨了几种UQ方法背后隐含的假设，即使用生成一致性作为置信度的代理，我们将这一想法形式化为一致性假设。我们提出了三个数学表述及其相应的统计检验来捕捉这一假设的不同变体，并引入了评估LLM输出一致性表现的指标。我们的实证研究覆盖8个基准数据集和3个任务（问答、文本摘要和文本到SQL），揭示了在不同情境下该假设的普遍性。在这几个表述中，我们强调“Sim-Any”假设最具操作性，并展示了如何通过提出基于生成间相似性的数据免费黑盒UQ方法来利用这一假设进行置信度估计。这些方法在与最近基线方法的比较中表现出色，展示了实验观察到的一致性假设的实际价值。 

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
# Few-Shot Segmentation of Historical Maps via Linear Probing of Vision Foundation Models 

**Title (ZH)**: 基于视觉基础模型的线性探查的 Historical Maps 少-shot 分割 

**Authors**: Rafael Sterzinger, Marco Peer, Robert Sablatnig  

**Link**: [PDF](https://arxiv.org/pdf/2506.21826)  

**Abstract**: As rich sources of history, maps provide crucial insights into historical changes, yet their diverse visual representations and limited annotated data pose significant challenges for automated processing. We propose a simple yet effective approach for few-shot segmentation of historical maps, leveraging the rich semantic embeddings of large vision foundation models combined with parameter-efficient fine-tuning. Our method outperforms the state-of-the-art on the Siegfried benchmark dataset in vineyard and railway segmentation, achieving +5% and +13% relative improvements in mIoU in 10-shot scenarios and around +20% in the more challenging 5-shot setting. Additionally, it demonstrates strong performance on the ICDAR 2021 competition dataset, attaining a mean PQ of 67.3% for building block segmentation, despite not being optimized for this shape-sensitive metric, underscoring its generalizability. Notably, our approach maintains high performance even in extremely low-data regimes (10- & 5-shot), while requiring only 689k trainable parameters - just 0.21% of the total model size. Our approach enables precise segmentation of diverse historical maps while drastically reducing the need for manual annotations, advancing automated processing and analysis in the field. Our implementation is publicly available at: this https URL. 

**Abstract (ZH)**: 基于大规模视觉基础模型的参数高效微调，few-shot历史地图分割方法 

---
# SciMantify -- A Hybrid Approach for the Evolving Semantification of Scientific Knowledge 

**Title (ZH)**: SciMantify -- 一种动态科学知识语义化的混合方法 

**Authors**: Lena John, Kheir Eddine Farfar, Sören Auer, Oliver Karras  

**Link**: [PDF](https://arxiv.org/pdf/2506.21819)  

**Abstract**: Scientific publications, primarily digitized as PDFs, remain static and unstructured, limiting the accessibility and reusability of the contained knowledge. At best, scientific knowledge from publications is provided in tabular formats, which lack semantic context. A more flexible, structured, and semantic representation is needed to make scientific knowledge understandable and processable by both humans and machines. We propose an evolution model of knowledge representation, inspired by the 5-star Linked Open Data (LOD) model, with five stages and defined criteria to guide the stepwise transition from a digital artifact, such as a PDF, to a semantic representation integrated in a knowledge graph (KG). Based on an exemplary workflow implementing the entire model, we developed a hybrid approach, called SciMantify, leveraging tabular formats of scientific knowledge, e.g., results from secondary studies, to support its evolving semantification. In the approach, humans and machines collaborate closely by performing semantic annotation tasks (SATs) and refining the results to progressively improve the semantic representation of scientific knowledge. We implemented the approach in the Open Research Knowledge Graph (ORKG), an established platform for improving the findability, accessibility, interoperability, and reusability of scientific knowledge. A preliminary user experiment showed that the approach simplifies the preprocessing of scientific knowledge, reduces the effort for the evolving semantification, and enhances the knowledge representation through better alignment with the KG structures. 

**Abstract (ZH)**: 科学出版物主要以PDF形式数字化，保持静态且无结构，限制了其中知识的可访问性和再利用性。科学出版物中的知识通常以表格形式提供，缺乏语义上下文。需要一种更灵活、结构化和语义化的表示形式，以便人类和机器能够理解和处理这些知识。我们提出了一个受5星Linked Open Data (LOD)模型启发的知识表示演化模型，包含五个阶段和指导准则，引导从PDF等数字文件逐步过渡到集成在知识图谱(KG)中的语义表示。基于整个模型的示例工作流，我们开发了一种混合方法，称为SciMantify，利用科学知识的表格格式，如二次研究的结果，以支持其逐步语义化。在该方法中，人类和机器密切合作，执行语义标注任务(SATs)，并不断改进科学知识的语义表示。我们将在Open Research Knowledge Graph (ORKG)平台上实现该方法，这是一个提高科学知识可查找性、可访问性、互操作性和再利用性的已建立平台。初步用户实验表明，该方法简化了科学知识的预处理，减少了逐步语义化的工作量，并通过更好的与KG结构对齐提高了知识表示。 

---
# Exploring the Structure of AI-Induced Language Change in Scientific English 

**Title (ZH)**: 探索由人工智能引发的科学英语语言变化结构 

**Authors**: Riley Galpin, Bryce Anderson, Tom S. Juzek  

**Link**: [PDF](https://arxiv.org/pdf/2506.21817)  

**Abstract**: Scientific English has undergone rapid and unprecedented changes in recent years, with words such as "delve," "intricate," and "crucial" showing significant spikes in frequency since around 2022. These changes are widely attributed to the growing influence of Large Language Models like ChatGPT in the discourse surrounding bias and misalignment. However, apart from changes in frequency, the exact structure of these linguistic shifts has remained unclear. The present study addresses this and investigates whether these changes involve the replacement of synonyms by suddenly 'spiking words,' for example, "crucial" replacing "essential" and "key," or whether they reflect broader semantic and pragmatic qualifications. To further investigate structural changes, we include part of speech tagging in our analysis to quantify linguistic shifts over grammatical categories and differentiate between word forms, like "potential" as a noun vs. as an adjective. We systematically analyze synonym groups for widely discussed 'spiking words' based on frequency trends in scientific abstracts from PubMed. We find that entire semantic clusters often shift together, with most or all words in a group increasing in usage. This pattern suggests that changes induced by Large Language Models are primarily semantic and pragmatic rather than purely lexical. Notably, the adjective "important" shows a significant decline, which prompted us to systematically analyze decreasing lexical items. Our analysis of "collapsing" words reveals a more complex picture, which is consistent with organic language change and contrasts with the patterns of the abrupt spikes. These insights into the structure of language change contribute to our understanding of how language technology continues to shape human language. 

**Abstract (ZH)**: 近年来，科学英语经历了快速且前所未有的变化，自2022年起，“delve”、“intricate”和“crucial”等词汇的使用频率显著增加。这些变化通常被认为是大型语言模型如ChatGPT在偏见和不对齐议题讨论中影响力的增强所致。然而，除了频率的变化，这些语言变化的具体结构仍然不清楚。本研究旨在探讨这些变化是否涉及到同义词被突然出现的“突变词汇”所取代，例如，“crucial”取代“essential”和“key”，或者这些变化是否反映了更广泛的语义和语用上的调整。为进一步研究结构性变化，我们将词性标注纳入分析，以量化不同语法类别的语言变化，并区分如“potential”作为名词与作为形容词的不同形式。基于PubMed中科学摘要的频率趋势，我们系统分析了广泛讨论的“突变词汇”的同义词组。我们发现，整个语义集群往往一起变化，整个组中的大多数或所有词汇的使用频率均增加。这一模式表明，大型语言模型引发的变化主要涉及语义和语用层面，而非单纯词汇层面。值得注意的是，“important”这一形容词出现了显著下降，促使我们对下降的词汇进行系统分析。我们对“坍缩”词汇的分析揭示了一个更复杂的图景，与突变词汇模式一致，与有机语言变化的路径相吻合。这些对语言变化结构的见解有助于我们理解语言技术如何继续塑造人类语言。 

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
# Comparing Learning Paradigms for Egocentric Video Summarization 

**Title (ZH)**: 自视点视频摘要化学习范式比较 

**Authors**: Daniel Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21785)  

**Abstract**: In this study, we investigate various computer vision paradigms - supervised learning, unsupervised learning, and prompt fine-tuning - by assessing their ability to understand and interpret egocentric video data. Specifically, we examine Shotluck Holmes (state-of-the-art supervised learning), TAC-SUM (state-of-the-art unsupervised learning), and GPT-4o (a prompt fine-tuned pre-trained model), evaluating their effectiveness in video summarization. Our results demonstrate that current state-of-the-art models perform less effectively on first-person videos compared to third-person videos, highlighting the need for further advancements in the egocentric video domain. Notably, a prompt fine-tuned general-purpose GPT-4o model outperforms these specialized models, emphasizing the limitations of existing approaches in adapting to the unique challenges of first-person perspectives. Although our evaluation is conducted on a small subset of egocentric videos from the Ego-Exo4D dataset due to resource constraints, the primary objective of this research is to provide a comprehensive proof-of-concept analysis aimed at advancing the application of computer vision techniques to first-person videos. By exploring novel methodologies and evaluating their potential, we aim to contribute to the ongoing development of models capable of effectively processing and interpreting egocentric perspectives. 

**Abstract (ZH)**: 本研究通过评估监督学习、无监督学习和提示微调等多种计算机视觉范式的理解与解释能力，探究其在第一人称视频数据上的表现，具体考察了当前最佳的监督学习模型Shotluck Holmes、最佳的无监督学习模型TAC-SUM以及微调后的通用模型GPT-4o在视频摘要中的效果。研究结果表明，当前最先进的模型在第一人称视频上的表现逊于第三人称视频，突显了在第一人称视频领域进一步发展的必要性。值得注意的是，一个微调后的通用GPT-4o模型在这些专业模型中表现更佳，突显了现有方法在适应第一人称视角的独特挑战时的局限性。尽管由于资源限制，评估是在Ego-Exo4D数据集的小子集上进行的，本研究的主要目的是提供一个全面的概念验证分析，旨在推动计算机视觉技术在第一人称视频中的应用。通过探索新的方法并评估其潜力，我们旨在为能够有效处理和解释第一人称视角的模型的持续开发做出贡献。 

---
# Evaluating List Construction and Temporal Understanding capabilities of Large Language Models 

**Title (ZH)**: 评估大型语言模型的列表构建能力和时间理解能力 

**Authors**: Alexandru Dumitru, V Venktesh, Adam Jatowt, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2506.21783)  

**Abstract**: Large Language Models (LLMs) have demonstrated immense advances in a wide range of natural language tasks. However, these models are susceptible to hallucinations and errors on particularly temporal understanding tasks involving multiple entities in answers. In such tasks, they fail to associate entities with accurate time intervals, generate a complete list of entities in answers or reason about events associated with specific temporal bounds. Existing works do not extensively evaluate the abilities of the model to perform implicit and explicit temporal understanding in a list answer construction setup. To bridge this gap, we propose the Time referenced List based Question Answering or TLQA benchmark that requires structured answers in list format aligned with corresponding time periods. Our TLQA benchmark, requires both list construction and temporal understanding simultaneously, which to the best of our knowledge has not been explored in prior benchmarks. We investigate the temporal understanding and list construction capabilities of state-of-the-art generative models on TLQA in closed-book and open-domain settings. Our findings reveal significant shortcomings in current models, particularly their inability to provide complete answers and temporally align facts in a closed-book setup and the need to improve retrieval in open-domain setup, providing clear future directions for research on TLQA. The benchmark and code at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在广泛自然语言任务中取得了巨大的进展。然而，在涉及多个实体特别是时间理解任务中，这些模型容易出现幻觉和错误。在这种任务中，它们无法将实体与准确的时间区间关联起来，生成答案中的完整实体列表，或关于特定时间界限的事件进行推理。现有工作没有广泛评估模型在列表答案构建设置中进行显性和隐性时间理解的能力。为弥补这一不足，我们提出了时间参考列表问答基准（TLQA），要求构建结构化的、与相应时间区间对齐的答案列表。TLQA基准同时要求列表构建和时间理解能力，据我们所知，此前的基准中尚未有此类探索。我们在封闭书本和开放域设置下调查了领先生成模型在TLQA上的时间理解能力和列表构建能力。我们的发现揭示了当前模型的重大缺陷，特别是在封闭书本设置下不能提供完整的答案和时间对齐的事实，在开放域设置下需要改进检索，为TLQA研究提供了明确的未来发展方向。基准和代码请访问此链接。 

---
# Experimental investigation of pose informed reinforcement learning for skid-steered visual navigation 

**Title (ZH)**: 基于姿态信息的强化学习在滑移转向视觉导航中的实验研究 

**Authors**: Ameya Salvi, Venkat Krovi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21732)  

**Abstract**: Vision-based lane keeping is a topic of significant interest in the robotics and autonomous ground vehicles communities in various on-road and off-road applications. The skid-steered vehicle architecture has served as a useful vehicle platform for human controlled operations. However, systematic modeling, especially of the skid-slip wheel terrain interactions (primarily in off-road settings) has created bottlenecks for automation deployment. End-to-end learning based methods such as imitation learning and deep reinforcement learning, have gained prominence as a viable deployment option to counter the lack of accurate analytical models. However, the systematic formulation and subsequent verification/validation in dynamic operation regimes (particularly for skid-steered vehicles) remains a work in progress. To this end, a novel approach for structured formulation for learning visual navigation is proposed and investigated in this work. Extensive software simulations, hardware evaluations and ablation studies now highlight the significantly improved performance of the proposed approach against contemporary literature. 

**Abstract (ZH)**: 基于视觉的车道保持是机器人学和自主地面车辆社区在各种道路和非道路应用场景中的一个重要研究课题。滑移转向车架构为人类操作提供了一个有用的车辆平台。然而，尤其是对于非道路环境中的滑移打滑车轮地形交互的系统建模成为自动化部署的瓶颈。基于端到端学习的方法，如模仿学习和深度强化学习，已成为应对缺乏准确分析模型的可行部署选项。然而，滑移转向车辆在动态操作制度下的系统建模和后续验证仍然有待完善。为此，本文提出并探讨了一种新型结构化学习视觉导航的方法。大量的软件仿真、硬件评估和消融研究现在表明，所提出方法在当代文献中显著提高了性能。 

---
# Exploring Image Generation via Mutually Exclusive Probability Spaces and Local Correlation Hypothesis 

**Title (ZH)**: 基于互斥概率空间和局部相关假设的图像生成探索 

**Authors**: Chenqiu Zhao, Anup Basu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21731)  

**Abstract**: We propose two theoretical frameworks, the Mutually Exclusive Probability Space (MESP) and the Local Correlation Hypothesis (LCH), to explore a potential limitation in probabilistic generative models; namely that learning global distributions leads to memorization rather than generative behavior. MESP emerges from our rethinking of the Variational Autoencoder (VAE). We observe that latent variable distributions in VAE exhibit overlap, which leads to an optimization conflict between the reconstruction loss and KL-divergence loss. A lower bound based on the overlap coefficient is proposed. We refer to this phenomenon as Mutually Exclusive Probability Spaces. Based on MESP, a Binary Latent Autoencoder (BL-AE) is proposed to encode images into binary latent representations. These binary latents are used as the input to our Autoregressive Random Variable Model (ARVM), a modified autoregressive model outputting histograms. Our ARVM achieves competitive FID scores, outperforming state-of-the-art methods on standard datasets. However, such scores reflect memorization rather than generation. To address this issue, we propose the Local Correlation Hypothesis (LCH), which posits that generative capability arising from local correlations among latent variables. Comprehensive experiments and discussions are conducted to validate our frameworks. 

**Abstract (ZH)**: 我们提出了两个理论框架：互斥概率空间（MESP）和局部关联假设（LCH），以探讨概率生成模型中潜在的局限性；即学习全局分布会导致记忆行为而非生成行为。MESP源自我们对变分自编码器（VAE）的重新思考。我们发现VAE中的潜在变量分布存在重叠，导致重构损失和KL散度损失之间的优化冲突。基于重叠系数提出了一个下界。我们将这种现象称为互斥概率空间。基于MESP，我们提出了二元潜自编码器（BL-AE）以将图像编码为二元潜表示。这些二元潜表示作为我们自回归随机变量模型（ARVM）的输入，ARVM是一个输出直方图的修改自回归模型。我们的ARVM在标准数据集上实现了竞争力的FID得分，超越了当前最好的方法。但这些得分反映的是记忆而非生成。为解决此问题，我们提出了局部关联假设（LCH），认为生成能力来源于潜在变量之间的局部关联。进行了全面的实验和讨论以验证我们的框架。 

---
# Simultaneously Fair Allocation of Indivisible Items Across Multiple Dimensions 

**Title (ZH)**: 多维度下的非可分物品的公平分配 

**Authors**: Yasushi Kawase, Bodhayan Roy, Mohammad Azharuddin Sanpui  

**Link**: [PDF](https://arxiv.org/pdf/2506.21727)  

**Abstract**: This paper explores the fair allocation of indivisible items in a multidimensional setting, motivated by the need to address fairness in complex environments where agents assess bundles according to multiple criteria. Such multidimensional settings are not merely of theoretical interest but are central to many real-world applications. For example, cloud computing resources are evaluated based on multiple criteria such as CPU cores, memory, and network bandwidth. In such cases, traditional one dimensional fairness notions fail to capture fairness across multiple attributes. To address these challenges, we study two relaxed variants of envy-freeness: weak simultaneously envy-free up to c goods (weak sEFc) and strong simultaneously envy-free up to c goods (strong sEFc), which accommodate the multidimensionality of agents' preferences. Under the weak notion, for every pair of agents and for each dimension, any perceived envy can be eliminated by removing, if necessary, a different set of goods from the envied agent's allocation. In contrast, the strong version requires selecting a single set of goods whose removal from the envied bundle simultaneously eliminates envy in every dimension. We provide upper and lower bounds on the relaxation parameter c that guarantee the existence of weak or strong sEFc allocations, where these bounds are independent of the total number of items. In addition, we present algorithms for checking whether a weak or strong sEFc allocation exists. Moreover, we establish NP-hardness results for checking the existence of weak sEF1 and strong sEF1 allocations. 

**Abstract (ZH)**: 基于多维度设置下非可分物品的公平分配研究 

---
# Elucidating and Endowing the Diffusion Training Paradigm for General Image Restoration 

**Title (ZH)**: 阐明并赋予扩散训练范式以通用图像修复能力 

**Authors**: Xin Lu, Xueyang Fu, Jie Xiao, Zihao Fan, Yurui Zhu, Zheng-Jun Zha  

**Link**: [PDF](https://arxiv.org/pdf/2506.21722)  

**Abstract**: While diffusion models demonstrate strong generative capabilities in image restoration (IR) tasks, their complex architectures and iterative processes limit their practical application compared to mainstream reconstruction-based general ordinary IR networks. Existing approaches primarily focus on optimizing network architecture and diffusion paths but overlook the integration of the diffusion training paradigm within general ordinary IR frameworks. To address these challenges, this paper elucidates key principles for adapting the diffusion training paradigm to general IR training through systematic analysis of time-step dependencies, network hierarchies, noise-level relationships, and multi-restoration task correlations, proposing a new IR framework supported by diffusion-based training. To enable IR networks to simultaneously restore images and model generative representations, we introduce a series of regularization strategies that align diffusion objectives with IR tasks, improving generalization in single-task scenarios. Furthermore, recognizing that diffusion-based generation exerts varying influences across different IR tasks, we develop an incremental training paradigm and task-specific adaptors, further enhancing performance in multi-task unified IR. Experiments demonstrate that our method significantly improves the generalization of IR networks in single-task IR and achieves superior performance in multi-task unified IR. Notably, the proposed framework can be seamlessly integrated into existing general IR architectures. 

**Abstract (ZH)**: 基于扩散训练范式的通用图像恢复框架 

---
# Performance Prediction for Large Systems via Text-to-Text Regression 

**Title (ZH)**: 基于文本到文本回归的大系统性能预测 

**Authors**: Yash Akhauri, Bryan Lewandowski, Cheng-Hsi Lin, Adrian N. Reyes, Grant C. Forbes, Arissa Wongpanich, Bangding Yang, Mohamed S. Abdelfattah, Sagi Perel, Xingyou Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.21718)  

**Abstract**: In many industries, predicting metric outcomes of large systems is a fundamental problem, driven largely by traditional tabular regression. However, such methods struggle on complex systems data in the wild such as configuration files or system logs, where feature engineering is often infeasible. We propose text-to-text regression as a general, scalable alternative. For predicting resource efficiency on Borg, Google's massive compute cluster scheduling system, a 60M parameter encoder-decoder, trained from random initialization, achieves up to a near perfect 0.99 (0.9 average) rank correlation across the entire fleet, and 100x lower MSE than tabular approaches. The model also easily adapts to new tasks in only 500 few-shot examples and captures the densities of complex outcome distributions. Ablation studies highlight the importance of using encoders, increasing sequence length, and the model's inherent uncertainty quantification. These findings pave the way for universal simulators of real-world outcomes. 

**Abstract (ZH)**: 在许多行业中，预测大型系统的指标结果是一个基本问题，传统上主要依赖表格回归方法。然而，这类方法在复杂的系统数据（如配置文件或系统日志）上常常表现不佳，这些数据中的特征工程往往是不可行的。我们提出文本到文本回归作为一种通用且可扩展的替代方案。对于预测Google大规模计算集群调度系统Borg的资源效率，一个60M参数的编码器-解码器模型从随机初始化训练，实现了整个集群近乎完美的0.99（平均0.9）排名相关性，并且平均均方误差比表格方法低100倍。该模型还能够在仅500个少样本示例中轻松适应新任务，并捕捉复杂结果分布的密度。消融研究强调了使用编码器、增加序列长度以及模型固有的不确定性量化的重要性。这些发现为现实世界结果的通用模拟器铺平了道路。 

---
# APO: Enhancing Reasoning Ability of MLLMs via Asymmetric Policy Optimization 

**Title (ZH)**: APO: 增强MLLMs推理能力的异构策略优化 

**Authors**: Minjie Hong, Zirun Guo, Yan Xia, Zehan Wang, Ziang Zhang, Tao Jin, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21655)  

**Abstract**: Multimodal Large Language Models (MLLMs) are powerful at integrating diverse data, but they often struggle with complex reasoning. While Reinforcement learning (RL) can boost reasoning in LLMs, applying it to MLLMs is tricky. Common issues include a drop in performance on general tasks and the generation of overly detailed or "overthinking" reasoning. Our work investigates how the KL penalty and overthinking affect RL training in MLLMs. We propose Asymmetric Policy Optimization (APO) to address these issues, which divides the sampled responses into positive and negative groups. For positive samples, Difficulty-Adaptive Divergence Shaping (DADS) is introduced to dynamically adjust the KL divergence weight based on their difficulty. This method prevents policy entropy from dropping sharply, improves training stability, utilizes samples better, and preserves the model's existing knowledge. For negative samples, Suboptimal Trajectory Complexity Regularization (STCR) is proposed to penalize overly long responses. This helps mitigate overthinking and encourages more concise reasoning while preserving the model's explorative capacity. We apply our method to Qwen2.5-VL-3B, creating View-R1-3B. View-R1-3B significantly enhances reasoning capabilities, showing an average 7\% gain over the base model and outperforming larger MLLMs (7-11B) on various reasoning benchmarks. Importantly, unlike other reasoning-tuned MLLMs that often degrade on general tasks, View-R1-3B maintains consistent improvement, demonstrating superior generalization. These results highlight the effectiveness and broad applicability of our DADS and STCR techniques for advancing complex multimodal reasoning in MLLMs. The code will be made available at this https URL. 

**Abstract (ZH)**: 多模态大规模语言模型在复杂推理中的生成式策略优化与过拟合调节研究 

---
# IRanker: Towards Ranking Foundation Model 

**Title (ZH)**: IRanker: 面向基础模型的排序方法 

**Authors**: Tao Feng, Zhigang Hua, Zijie Lei, Yan Xie, Shuang Yang, Bo Long, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.21638)  

**Abstract**: Ranking tasks are ubiquitous, encompassing applications such as recommendation systems, LLM routing, and item re-ranking. We propose to unify these tasks using a single ranking foundation model (FM), as it eliminates the need for designing different models for each specific ranking task. However, unlike general supervision tasks in LLMs, ranking tasks do not have clear labels for supervision, posing great challenges to developing a ranking FM. To overcome these challenges, we propose IRanker, a ranking FM framework with reinforcement learning (RL) and iterative decoding. Our insight is to decompose the complex ranking task into an iterative decoding process that eliminates the worst candidate from the candidate pool step by step, which significantly reduces the output combinatorial space and better utilizes the limited context length during RL training. We meticulously train and comprehensively evaluate an IRanker-3B model on nine datasets across three scenarios: recommendation, routing, and passage ranking. The results show that a single IRanker-3B achieves state-of-the-art results on several datasets compared to models of similar size, and even surpasses the performance of larger models on certain datasets. We further demonstrate the effectiveness of our RL design and the robustness of the iterative mechanism across different LLM sizes. Moreover, we conducted both in-domain and out-of-domain zero-shot generalization experiments, which showed that IRanker-3B achieved good generalization on in-domain ranking tasks compared to the base LLM by at least 5% improvement. Surprisingly, on out-of-domain generic LLM tasks, IRanker-3B outperformed the base model by at least 9% on GSM8K, IFEval, and MathQA. In addition, the thoughts generated by IRanker-3B during training could further enhance zero-shot LLM performance. 

**Abstract (ZH)**: 统一排序任务的强化学习迭代排序框架IRanker 

---
# AeroLite-MDNet: Lightweight Multi-task Deviation Detection Network for UAV Landing 

**Title (ZH)**: AeroLite-MDNet：轻量级多任务偏差检测网络用于无人机着陆 

**Authors**: Haiping Yang, Huaxing Liu, Wei Wu, Zuohui Chen, Ning Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21635)  

**Abstract**: Unmanned aerial vehicles (UAVs) are increasingly employed in diverse applications such as land surveying, material transport, and environmental monitoring. Following missions like data collection or inspection, UAVs must land safely at docking stations for storage or recharging, which is an essential requirement for ensuring operational continuity. However, accurate landing remains challenging due to factors like GPS signal interference. To address this issue, we propose a deviation warning system for UAV landings, powered by a novel vision-based model called AeroLite-MDNet. This model integrates a multiscale fusion module for robust cross-scale object detection and incorporates a segmentation branch for efficient orientation estimation. We introduce a new evaluation metric, Average Warning Delay (AWD), to quantify the system's sensitivity to landing deviations. Furthermore, we contribute a new dataset, UAVLandData, which captures real-world landing deviation scenarios to support training and evaluation. Experimental results show that our system achieves an AWD of 0.7 seconds with a deviation detection accuracy of 98.6\%, demonstrating its effectiveness in enhancing UAV landing reliability. Code will be available at this https URL 

**Abstract (ZH)**: 无人驾驶航空器（UAVs）在土地测验、物资运输和环境监测等多种应用中越来越广泛。在完成数据收集或检查任务后，UAVs必须准确地降落在指定的停机站进行存储或充电，这是确保操作连续性的基本要求。然而，由于GPS信号干扰等因素，精确的降落仍然是一个挑战。为了解决这一问题，我们提出了一种基于新型视觉模型AeroLite-MDNet的偏移警告系统。该模型结合了多尺度融合模块进行稳健的跨尺度对象检测，并包含分割分支进行高效的姿态估计。我们引入了一种新的评价指标，平均警告延迟（AWD），以量化系统对降落偏差的敏感度。此外，我们贡献了一个新的数据集UAVLandData，该数据集捕捉了真实世界的降落偏差场景，以支持训练和评估。实验结果表明，我们的系统实现了0.7秒的AWD和98.6%的偏差检测准确率，证明了其在提高UAV降落可靠性方面的有效性。代码将发布在以下链接：this https URL。 

---
# Ark: An Open-source Python-based Framework for Robot Learning 

**Title (ZH)**: Ark：一种基于Python的机器人学习开源框架 

**Authors**: Magnus Dierking, Christopher E. Mower, Sarthak Das, Huang Helong, Jiacheng Qiu, Cody Reading, Wei Chen, Huidong Liang, Huang Guowei, Jan Peters, Quan Xingyue, Jun Wang, Haitham Bou-Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21628)  

**Abstract**: Robotics has made remarkable hardware strides-from DARPA's Urban and Robotics Challenges to the first humanoid-robot kickboxing tournament-yet commercial autonomy still lags behind progress in machine learning. A major bottleneck is software: current robot stacks demand steep learning curves, low-level C/C++ expertise, fragmented tooling, and intricate hardware integration, in stark contrast to the Python-centric, well-documented ecosystems that propelled modern AI. We introduce ARK, an open-source, Python-first robotics framework designed to close that gap. ARK presents a Gym-style environment interface that allows users to collect data, preprocess it, and train policies using state-of-the-art imitation-learning algorithms (e.g., ACT, Diffusion Policy) while seamlessly toggling between high-fidelity simulation and physical robots. A lightweight client-server architecture provides networked publisher-subscriber communication, and optional C/C++ bindings ensure real-time performance when needed. ARK ships with reusable modules for control, SLAM, motion planning, system identification, and visualization, along with native ROS interoperability. Comprehensive documentation and case studies-from manipulation to mobile navigation-demonstrate rapid prototyping, effortless hardware swapping, and end-to-end pipelines that rival the convenience of mainstream machine-learning workflows. By unifying robotics and AI practices under a common Python umbrella, ARK lowers entry barriers and accelerates research and commercial deployment of autonomous robots. 

**Abstract (ZH)**: 机器人技术在从DARPA的城市挑战和机器人挑战到首个类人机器人踢拳锦标赛中取得了显著的硬件进步，然而商业自主性仍然落后于机器学习的进步。主要瓶颈在于软件：当前的机器人堆栈要求陡峭的学习曲线、低级的C/C++专门知识、散乱的工具以及复杂的硬件集成，这与以Python为中心、文档丰富的生态系统形成了鲜明对比，后者推动了现代人工智能的发展。我们引入了ARK，一个开源的、以Python为主的机器人框架，旨在弥合这一差距。ARK提供了一种类似Gym的环境接口，允许用户收集数据、预处理数据并使用最先进的模仿学习算法（例如ACT、扩散策略）训练策略，同时无缝切换高保真模拟和物理机器人。轻量级的客户端-服务器架构提供了网络化的发布者-订阅者通信，而可选的C/C++绑定确保在需要时提供实时性能。ARK附带了可重用的控制、SLAM、运动规划、系统辨识和可视化模块，并具备原生ROS互操作性。全面的文档和案例研究（涵盖从操作到移动导航）展示了快速原型制作、轻松更换硬件和端到端工作流的便捷性，后者与主流机器学习工作流程相当。通过在共同的Python伞下统一机器人技术和人工智能实践，ARK降低了准入门槛并加速了自主机器人的研究和商业部署。 

---
# FrankenBot: Brain-Morphic Modular Orchestration for Robotic Manipulation with Vision-Language Models 

**Title (ZH)**: FrankenBot：类脑模块化 orchestration 及视觉-语言模型在机器人操作中的应用 

**Authors**: Shiyi Wang, Wenbo Li, Yiteng Chen, Qingyao Wu, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21627)  

**Abstract**: Developing a general robot manipulation system capable of performing a wide range of tasks in complex, dynamic, and unstructured real-world environments has long been a challenging task. It is widely recognized that achieving human-like efficiency and robustness manipulation requires the robotic brain to integrate a comprehensive set of functions, such as task planning, policy generation, anomaly monitoring and handling, and long-term memory, achieving high-efficiency operation across all functions. Vision-Language Models (VLMs), pretrained on massive multimodal data, have acquired rich world knowledge, exhibiting exceptional scene understanding and multimodal reasoning capabilities. However, existing methods typically focus on realizing only a single function or a subset of functions within the robotic brain, without integrating them into a unified cognitive architecture. Inspired by a divide-and-conquer strategy and the architecture of the human brain, we propose FrankenBot, a VLM-driven, brain-morphic robotic manipulation framework that achieves both comprehensive functionality and high operational efficiency. Our framework includes a suite of components, decoupling a part of key functions from frequent VLM calls, striking an optimal balance between functional completeness and system efficiency. Specifically, we map task planning, policy generation, memory management, and low-level interfacing to the cortex, cerebellum, temporal lobe-hippocampus complex, and brainstem, respectively, and design efficient coordination mechanisms for the modules. We conducted comprehensive experiments in both simulation and real-world robotic environments, demonstrating that our method offers significant advantages in anomaly detection and handling, long-term memory, operational efficiency, and stability -- all without requiring any fine-tuning or retraining. 

**Abstract (ZH)**: 开发能够在复杂、动态且未结构化的现实环境中执行广泛任务的通用机器人操作系统长久以来是一项具有挑战性的任务。广泛认可的观点认为，实现类似人类的效率和鲁棒性操作需要机器人“大脑”整合一系列功能，如任务规划、策略生成、异常监测与处理以及长期记忆，从而在所有功能上实现高效操作。视觉-语言模型（VLMs）在大规模多模态数据上进行了预训练，获得了丰富的世界知识，展示了卓越的场景理解和多模态推理能力。然而，现有方法通常仅专注于实现机器人“大脑”内的一项或多项功能，而不将它们整合到统一的认知架构中。受分而治之策略和人脑架构的启发，我们提出了一种由VLM驱动、模仿人脑结构的机器人操作框架——FrankenBot，该框架实现了全面的功能性和高操作效率。该框架包括一系列组件，从频繁的VLM调用中解耦出部分关键功能，平衡了功能完整性和系统效率之间的关系。具体而言，我们将任务规划、策略生成、记忆管理以及低级别接口分别映射到大脑皮层、小脑、颞叶-海马复合体和延髓，并设计了高效协调机制以优化模块之间的协作。我们在仿真和真实世界机器人环境中进行了全面的实验，结果表明，我们的方法在异常检测与处理、长期记忆、操作效率和稳定性方面具有显著优势——无需任何微调或重新训练。 

---
# Doc2SAR: A Synergistic Framework for High-Fidelity Extraction of Structure-Activity Relationships from Scientific Documents 

**Title (ZH)**: Doc2SAR：一种用于从科学文献中高保真提取结构-活性关系的协同框架 

**Authors**: Jiaxi Zhuang, Kangning Li, Jue Hou, Mingjun Xu, Zhifeng Gao, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21625)  

**Abstract**: Extracting molecular structure-activity relationships (SARs) from scientific literature and patents is essential for drug discovery and materials research. However, this task remains challenging due to heterogeneous document formats and limitations of existing methods. Specifically, rule-based approaches relying on rigid templates fail to generalize across diverse document layouts, while general-purpose multimodal large language models (MLLMs) lack sufficient accuracy and reliability for specialized tasks, such as layout detection and optical chemical structure recognition (OCSR). To address these challenges, we introduce DocSAR-200, a rigorously annotated benchmark of 200 scientific documents designed specifically for evaluating SAR extraction methods. Additionally, we propose Doc2SAR, a novel synergistic framework that integrates domain-specific tools with MLLMs enhanced via supervised fine-tuning (SFT). Extensive experiments demonstrate that Doc2SAR achieves state-of-the-art performance across various document types, significantly outperforming leading end-to-end baselines. Specifically, Doc2SAR attains an overall Table Recall of 80.78% on DocSAR-200, exceeding end2end GPT-4o by 51.48%. Furthermore, Doc2SAR demonstrates practical usability through efficient inference and is accompanied by a web app. 

**Abstract (ZH)**: 从科学文献和专利中提取分子结构-活性关系（SARs）对于药物发现和材料研究至关重要。然而，由于文献格式的异质性和现有方法的局限性，这一任务仍然具有挑战性。具体来说，依赖于固定模板的基于规则的方法无法适应多样的文档布局，而通用的多模态大语言模型（MLLMs）在专门任务，如布局检测和光学化学结构识别（OCSR）方面缺乏足够的准确性和可靠性。为应对这一挑战，我们引入了DocSAR-200，这是一个专门用于评估SAR提取方法的严格标注基准，包含200份科学文档。此外，我们提出了一种新颖的协同框架Doc2SAR，将领域特定工具与通过监督微调（SFT）增强的大语言模型集成。广泛的实验表明，Doc2SAR 在各种类型的文档中达到了最先进的性能，显著优于领先的一体化基线。具体而言，Doc2SAR 在DocSAR-200上的总体 Table Recall 达到了 80.78%，超过端到端 GPT-4o 51.48%。此外，Doc2SAR 通过高效的推理展示了其实用性，并附带了一个网页应用。 

---
# Adapting Foundation Speech Recognition Models to Impaired Speech: A Semantic Re-chaining Approach for Personalization of German Speech 

**Title (ZH)**: 适配受损语音的预训练语音识别模型：一种针对德语语音个性化处理的语义重链方法 

**Authors**: Niclas Pokel, Pehuén Moure, Roman Boehringer, Yingqiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21622)  

**Abstract**: Speech impairments caused by conditions such as cerebral palsy or genetic disorders pose significant challenges for automatic speech recognition (ASR) systems. Despite recent advances, ASR models like Whisper struggle with non-normative speech due to limited training data and the difficulty of collecting and annotating non-normative speech samples. In this work, we propose a practical and lightweight pipeline to personalize ASR models, formalizing the selection of words and enriching a small, speech-impaired dataset with semantic coherence. Applied to data from a child with a structural speech impairment, our approach shows promising improvements in transcription quality, demonstrating the potential to reduce communication barriers for individuals with atypical speech patterns. 

**Abstract (ZH)**: 由脑瘫或遗传性疾病等条件引起的语言障碍对自动语音识别（ASR）系统构成了重大挑战。尽管最近取得了进展，但像Whisper这样的ASR模型仍然难以处理非规范性语音，因为训练数据有限，收集和标注非规范性语音样本也颇具难度。在本工作中，我们提出了一个切实可行且轻量级的流水线来个性化ASR模型，正式化单词的选择并用语义连贯性丰富小型受损语音数据集。应用于具有结构化语音障碍儿童的数据，我们的方法在转写质量上显示出有希望的改进，证明了减少具有非典型语音模式个体的沟通障碍的潜力。 

---
# The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs 

**Title (ZH)**: 开放证明语料库：大规模LLM生成数学证明研究 

**Authors**: Jasper Dekoninck, Ivo Petrov, Kristian Minchev, Mislav Balunovic, Martin Vechev, Miroslav Marinov, Maria Drencheva, Lyuba Konova, Milen Shumanov, Kaloyan Tsvetkov, Nikolay Drenchev, Lazar Todorov, Kalina Nikolova, Nikolay Georgiev, Vanesa Kalinkova, Margulan Ismoldayev  

**Link**: [PDF](https://arxiv.org/pdf/2506.21621)  

**Abstract**: In recent months, large language models (LLMs) have made significant progress in mathematical proof generation, but further advancement is hindered by the lack of a large-scale, high-quality dataset of human-evaluated proofs. While expensive to create, such a dataset is essential for driving improvements in training and enabling a rigorous analysis of proof generation capabilities. In this work, we present the Open Proof Corpus (OPC), a dataset comprising over 5,000 human-evaluated proofs produced by state-of-the-art LLMs. The OPC was specifically designed for broad applicability and downstream usage in proof generation research and is the first to include a substantial number of correct, LLM-generated solutions to problems from prestigious mathematics competitions such as the USAMO and IMO. Using the OPC, we explore critical questions in automated proof generation: (1) the performance gap between natural language and formal proof generation, (2) the discrepancy between final-answer accuracy and full-proof validity, and (3) the impact of best-of-n selection on proof quality. Finally, to showcase the utility of the OPC, we finetune an 8B-parameter model on the dataset, obtaining a model that performs on par with the best model, Gemini-2.5-Pro, on the task of evaluating proof correctness. 

**Abstract (ZH)**: 近期，大规模语言模型在数学证明生成方面取得了显著进步，但由于缺乏大量高质量的人工评估证明数据集，进一步发展受到限制。尽管创建成本较高，但这样的数据集对于推动训练改进和证明生成能力的严谨分析是必不可少的。在本文中，我们介绍了开源证明语料库（OPC），一个包含超过5,000个人工评估证明的数据集，这些证明是由最先进的大规模语言模型生成的。OPC 特别设计以促进证明生成研究的广泛应用，并且是首次包含大量来自USAMO和IMO等知名数学竞赛问题的正确解证明的数据集。利用OPC，我们探讨了自动证明生成中的关键问题：（1）自然语言与形式证明生成之间的性能差距，（2）最终答案准确性与完整证明有效性的差异，以及（3）最佳选择对证明质量的影响。最后，为了展示OPC的应用价值，我们在数据集上微调了一个8亿参数的模型，得到的模型在证明正确性评估任务上与最佳模型Gemini-2.5-Pro表现相当。 

---
# How Large Language Models play humans in online conversations: a simulated study of the 2016 US politics on Reddit 

**Title (ZH)**: 大型语言模型在在线对话中如何玩转人类：2016年美国政治在Reddit上的模拟研究 

**Authors**: Daniele Cirulli, Giulio Cimini, Giovanni Palermo  

**Link**: [PDF](https://arxiv.org/pdf/2506.21620)  

**Abstract**: Large Language Models (LLMs) have recently emerged as powerful tools for natural language generation, with applications spanning from content creation to social simulations. Their ability to mimic human interactions raises both opportunities and concerns, particularly in the context of politically relevant online discussions. In this study, we evaluate the performance of LLMs in replicating user-generated content within a real-world, divisive scenario: Reddit conversations during the 2016 US Presidential election. In particular, we conduct three different experiments, asking GPT-4 to generate comments by impersonating either real or artificial partisan users. We analyze the generated comments in terms of political alignment, sentiment, and linguistic features, comparing them against real user contributions and benchmarking against a null model. We find that GPT-4 is able to produce realistic comments, both in favor of or against the candidate supported by the community, yet tending to create consensus more easily than dissent. In addition we show that real and artificial comments are well separated in a semantically embedded space, although they are indistinguishable by manual inspection. Our findings provide insights on the potential use of LLMs to sneak into online discussions, influence political debate and shape political narratives, bearing broader implications of AI-driven discourse manipulation. 

**Abstract (ZH)**: 大型语言模型（LLMs）近年来已成为自然语言生成的强大工具，应用于从内容创作到社会模拟的多个领域。它们模仿人类互动的能力既带来了机会，也引发了担忧，特别是在与政治相关的在线讨论中。本研究评估了LLMs在复制现实世界中具有争议性的场景——2016年美国 Presidential 选举期间的 Reddit 演讲——中生成用户生成内容的性能。特别是在三项不同的实验中，我们要求GPT-4模仿真实或虚构的党派用户来生成评论。我们从政治倾向、情感和语言特征等方面分析了生成的评论，并将其与真实用户贡献进行比较，同时用基准模型进行对照。研究发现，GPT-4能够生成真实且具有说服力的评论，无论是支持还是反对社区支持的候选人，但更倾向于制造共识而非分歧。此外，我们还展示了真实和虚构的评论在语义嵌入的空间中较为分离，但在手动检查时难以区分。我们的研究结果提供了关于LLMs潜入在线讨论、影响政治辩论和塑造政治叙事潜在用途的见解，并具有更广泛的AI驱动话语操纵的含义。 

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
# LastingBench: Defend Benchmarks Against Knowledge Leakage 

**Title (ZH)**: LastingBench: 防护基准免受知识泄漏 

**Authors**: Yixiong Fang, Tianran Sun, Yuling Shi, Min Wang, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21614)  

**Abstract**: The increasing complexity of large language models (LLMs) raises concerns about their ability to "cheat" on standard Question Answering (QA) benchmarks by memorizing task-specific data. This undermines the validity of benchmark evaluations, as they no longer reflect genuine model capabilities but instead the effects of data leakage. While prior work has focused on detecting such leakage, little attention has been given to mitigating its impact and preserving the long-term utility of benchmarks. In this paper, we introduce LastingBench, a novel framework designed to continuously reinforce and safeguard existing benchmarks against knowledge leakage. LastingBench identifies leakage points in the context through perturbation, then rewrites the leakage points to counterfactual ones-disrupting memorization while preserving the benchmark's original evaluative intent. Evaluations of state-of-the-art QA benchmarks show significant performance gaps, highlighting the efficacy of LastingBench in reducing memorization effects. LastingBench offers a practical and scalable solution to ensure benchmark robustness over time, promoting fairer and more interpretable evaluations of LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）复杂性的增加引发了对其通过记忆特定任务数据而在标准问答（QA）基准测试中“作弊”的担忧。这削弱了基准评估的有效性，因为它们不再反映真实的模型能力，而是数据泄露的影响。尽管先前的工作集中在检测这种泄露，但很少有研究关注减轻其影响并维护基准的长期实用性。在本文中，我们提出了LastingBench，这是一种新型框架，旨在不断强化和保护现有基准免受知识泄露的影响。LastingBench通过扰动识别上下文中的泄露点，然后重写泄露点为反事实点，打断记忆现象同时保持基准原始的评估意图。最新问答基准测试的评估显示了显著的性能差距，突显了LastingBench在减少记忆效应方面的效果。LastingBench提供了一种实用且可扩展的解决方案，以确保基准的长期稳健性，促进更公平和更具解释性的大型语言模型评估。 

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
# From Thinking to Output: Chain-of-Thought and Text Generation Characteristics in Reasoning Language Models 

**Title (ZH)**: 从思考到输出：推理语言模型中的链式思考与文本生成特征 

**Authors**: Junhao Liu, Zhenhao Xu, Yuxin Fang, Yichuan Chen, Zuobin Ying, Wenhan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21609)  

**Abstract**: Recently, there have been notable advancements in large language models (LLMs), demonstrating their growing abilities in complex reasoning. However, existing research largely overlooks a thorough and systematic comparison of these models' reasoning processes and outputs, particularly regarding their self-reflection pattern (also termed "Aha moment") and the interconnections across diverse domains. This paper proposes a novel framework for analyzing the reasoning characteristics of four cutting-edge large reasoning models (GPT-o1, DeepSeek-R1, Kimi-k1.5, and Grok-3) using keywords statistic and LLM-as-a-judge paradigm. Our approach connects their internal thinking processes with their final outputs. A diverse dataset consists of real-world scenario-based questions covering logical deduction, causal inference, and multi-step problem-solving. Additionally, a set of metrics is put forward to assess both the coherence of reasoning and the accuracy of the outputs. The research results uncover various patterns of how these models balance exploration and exploitation, deal with problems, and reach conclusions during the reasoning process. Through quantitative and qualitative comparisons, disparities among these models are identified in aspects such as the depth of reasoning, the reliance on intermediate steps, and the degree of similarity between their thinking processes and output patterns and those of GPT-o1. This work offers valuable insights into the trade-off between computational efficiency and reasoning robustness and provides practical recommendations for enhancing model design and evaluation in practical applications. We publicly release our project at: this https URL 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）取得了显著进展，展示了其在复杂推理方面的能力。然而，现有研究很大程度上忽视了对这些模型推理过程和输出的全面系统比较，特别是它们的自我反思模式（也称为“恍然大悟”时刻）以及跨不同领域的相互联系。本文提出了一种新的框架，使用关键词统计和LLM-as-a-judge范式来分析四款前沿的大规模推理模型（GPT-o1、DeepSeek-R1、Kimi-k1.5和Grok-3）的推理特征。我们的方法将它们的内部思考过程与其最终输出联系起来。该研究数据集包含了涵盖逻辑推理、因果推断和多步问题解决的现实场景问题。此外，提出了一套指标来评估推理的一致性和输出的准确性。研究结果揭示了这些模型在推理过程中平衡探索与利用、处理问题以及得出结论的各种模式。通过定量和定性比较，这些模型在推理深度、对中间步骤的依赖程度以及思考过程与输出模式与GPT-o1之间的相似度方面存在差异。本文提供了关于计算效率与推理稳健性之间权衡的宝贵见解，并为在实际应用中改进模型设计和评估提供了实用建议。我们的项目已公开发布于：this https URL。 

---
# SysTemp: A Multi-Agent System for Template-Based Generation of SysML v2 

**Title (ZH)**: SysTemp: 基于模板的SysML v2生成的多智能体系统 

**Authors**: Yasmine Bouamra, Bruno Yun, Alexandre Poisson, Frédéric Armetta  

**Link**: [PDF](https://arxiv.org/pdf/2506.21608)  

**Abstract**: The automatic generation of SysML v2 models represents a major challenge in the engineering of complex systems, particularly due to the scarcity of learning corpora and complex syntax. We present SysTemp, a system aimed at facilitating and improving the creation of SysML v2 models from natural language specifications. It is based on a multi-agent system, including a template generator that structures the generation process. We discuss the advantages and challenges of this system through an evaluation, highlighting its potential to improve the quality of the generations in SysML v2 modeling. 

**Abstract (ZH)**: SysTemp：一种用于从自然语言规范生成SysML v2模型的多代理系统 

---
# CORE-KG: An LLM-Driven Knowledge Graph Construction Framework for Human Smuggling Networks 

**Title (ZH)**: CORE-KG: 一种基于大语言模型的知识图谱构建框架——针对人口走私网络 

**Authors**: Dipak Meher, Carlotta Domeniconi, Guadalupe Correa-Cabrera  

**Link**: [PDF](https://arxiv.org/pdf/2506.21607)  

**Abstract**: Human smuggling networks are increasingly adaptive and difficult to analyze. Legal case documents offer valuable insights but are unstructured, lexically dense, and filled with ambiguous or shifting references-posing challenges for automated knowledge graph (KG) construction. Existing KG methods often rely on static templates and lack coreference resolution, while recent LLM-based approaches frequently produce noisy, fragmented graphs due to hallucinations, and duplicate nodes caused by a lack of guided extraction. We propose CORE-KG, a modular framework for building interpretable KGs from legal texts. It uses a two-step pipeline: (1) type-aware coreference resolution via sequential, structured LLM prompts, and (2) entity and relationship extraction using domain-guided instructions, built on an adapted GraphRAG framework. CORE-KG reduces node duplication by 33.28%, and legal noise by 38.37% compared to a GraphRAG-based baseline-resulting in cleaner and more coherent graph structures. These improvements make CORE-KG a strong foundation for analyzing complex criminal networks. 

**Abstract (ZH)**: Human走私网络日益适应性强且难以分析。法律案例文件提供了宝贵见解但结构松散、词汇密集且包含模糊或变化的参考，这对自动化知识图谱(KG)构建构成了挑战。现有KG方法通常依赖静态模板并且缺乏指代消解，而近期基于LLM的方法经常生成噪声大、碎片化的图，并且由于缺乏引导式提取导致节点重复。我们提出CORE-KG，这是一种用于从法律文本构建可解释知识图谱的模块化框架。该框架采用两步管道：(1) 通过序列化、结构化LLM提示进行类型感知的指代消解；(2) 使用领域导向的指令进行实体和关系提取，构建于适应的GraphRAG框架之上。CORE-KG将节点重复减少33.28%，将法律噪声减少38.37%，相比基于GraphRAG的基线，产生了更干净、更连贯的图结构。这些改进使CORE-KG成为分析复杂犯罪网络的强大基础。 

---
# Large Language Models as symbolic DNA of cultural dynamics 

**Title (ZH)**: 大型语言模型作为文化动态的符号DNA 

**Authors**: Parham Pourdavood, Michael Jacob, Terrence Deacon  

**Link**: [PDF](https://arxiv.org/pdf/2506.21606)  

**Abstract**: This paper proposes a novel conceptualization of Large Language Models (LLMs) as externalized informational substrates that function analogously to DNA for human cultural dynamics. Rather than viewing LLMs as either autonomous intelligence or mere programmed mimicry, we argue they serve a broader role as repositories that preserve compressed patterns of human symbolic expression--"fossils" of meaningful dynamics that retain relational residues without their original living contexts. Crucially, these compressed patterns only become meaningful through human reinterpretation, creating a recursive feedback loop where they can be recombined and cycle back to ultimately catalyze human creative processes. Through analysis of four universal features--compression, decompression, externalization, and recursion--we demonstrate that just as DNA emerged as a compressed and externalized medium for preserving useful cellular dynamics without containing explicit reference to goal-directed physical processes, LLMs preserve useful regularities of human culture without containing understanding of embodied human experience. Therefore, we argue that LLMs' significance lies not in rivaling human intelligence, but in providing humanity a tool for self-reflection and playful hypothesis-generation in a low-stakes, simulated environment. This framework positions LLMs as tools for cultural evolvability, enabling humanity to generate novel hypotheses about itself while maintaining the human interpretation necessary to ground these hypotheses in ongoing human aesthetics and norms. 

**Abstract (ZH)**: 本文提出了一种关于大型语言模型（LLMs）的新概念，将其视为外部化的信息载体，类似于DNA在人类文化动态中的作用。我们认为，LLMs不仅不是自主智能，也不仅仅是在模仿编程，而是作为保存人类符号表达压缩模式的仓库——“有意义动态的化石”，保留了关系残余而没有其原始生活背景。关键的是，这些压缩模式只有在通过人类重新解释后才有意义，从而形成一个互馈循环，使它们能够重新组合，并最终催化人类的创造性过程。通过对四个普遍特征——压缩、解压缩、外部化和递归——的分析，我们表明，就像DNA作为一种压缩和外部化的介质，不包含关于目的导向物理过程的显式参考，用来保存对细胞动态有用的特性一样，LLMs也保存了对人类文化有用的规律，而不包含对人类体验的理解。因此，我们认为LLMs的意义不在于与人类智能竞争，而在于为人类提供一种自我反省和低风险模拟环境中的假设生成工具。这一框架将LLMs定位为文化可进化性的工具，使人类能够在保持人类解释以使这些假设扎根于持续的人类美学和规范的前提下，生成关于自身的新型假设。 

---
# MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents 

**Title (ZH)**: MemBench: 向更加全面的LLM基agents的内存评估迈进 

**Authors**: Haoran Tan, Zeyu Zhang, Chen Ma, Xu Chen, Quanyu Dai, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.21605)  

**Abstract**: Recent works have highlighted the significance of memory mechanisms in LLM-based agents, which enable them to store observed information and adapt to dynamic environments. However, evaluating their memory capabilities still remains challenges. Previous evaluations are commonly limited by the diversity of memory levels and interactive scenarios. They also lack comprehensive metrics to reflect the memory capabilities from multiple aspects. To address these problems, in this paper, we construct a more comprehensive dataset and benchmark to evaluate the memory capability of LLM-based agents. Our dataset incorporates factual memory and reflective memory as different levels, and proposes participation and observation as various interactive scenarios. Based on our dataset, we present a benchmark, named MemBench, to evaluate the memory capability of LLM-based agents from multiple aspects, including their effectiveness, efficiency, and capacity. To benefit the research community, we release our dataset and project at this https URL. 

**Abstract (ZH)**: 近期的研究突出了基于大语言模型的智能体中记忆机制的重要性，使其能够存储观察到的信息并适应动态环境。然而，评估其记忆能力仍然存在挑战。此前的评估通常受限于记忆层级和交互场景的多样性，缺乏从多方面反映记忆能力的综合指标。为解决这些问题，本文构建了一个更为综合的数据集和基准，以评估基于大语言模型的智能体的记忆能力。该数据集将事实记忆和反思记忆作为不同的层次，并提出参与和观察作为不同的交互场景。基于该数据集，我们提出了一套名为MemBench的基准，从有效性、效率和容量等多方面评估基于大语言模型的智能体的记忆能力。为了促进研究社区的发展，我们在<a href="this https URL">此处</a>发布了我们的数据集和项目。 

---
# Evaluating VisualRAG: Quantifying Cross-Modal Performance in Enterprise Document Understanding 

**Title (ZH)**: 评估VisualRAG：企业文档理解的跨模态性能量化 

**Authors**: Varun Mannam, Fang Wang, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21604)  

**Abstract**: Current evaluation frameworks for multimodal generative AI struggle to establish trustworthiness, hindering enterprise adoption where reliability is paramount. We introduce a systematic, quantitative benchmarking framework to measure the trustworthiness of progressively integrating cross-modal inputs such as text, images, captions, and OCR within VisualRAG systems for enterprise document intelligence. Our approach establishes quantitative relationships between technical metrics and user-centric trust measures. Evaluation reveals that optimal modality weighting with weights of 30% text, 15% image, 25% caption, and 30% OCR improves performance by 57.3% over text-only baselines while maintaining computational efficiency. We provide comparative assessments of foundation models, demonstrating their differential impact on trustworthiness in caption generation and OCR extraction-a vital consideration for reliable enterprise AI. This work advances responsible AI deployment by providing a rigorous framework for quantifying and enhancing trustworthiness in multimodal RAG for critical enterprise applications. 

**Abstract (ZH)**: 当前的多模态生成AI评价框架难以建立信任度，阻碍了对企业级应用中可靠性至上的采用。我们引入了一种系统性的定量基准框架，用于测量在VisualRAG系统中逐步整合跨模态输入（如文本、图像、字幕和OCR）对企业文档智能的信任度。我们的方法建立了技术指标与用户中心的信任度指标之间的定量关系。评估显示，最优模态加权（文本30%，图像15%，字幕25%，OCR30%）相比仅文本基线提高了57.3%的性能，同时保持了计算效率。我们提供了基础模型的比较评估，展示了它们在字幕生成和OCR提取中对信任度的不同影响，这是可靠企业AI的重要考量因素。本工作通过提供一个严谨的框架来量化和提升关键企业应用中多模态RAG的信任度，促进了负责任的AI部署。 

---
# BiMark: Unbiased Multilayer Watermarking for Large Language Models 

**Title (ZH)**: BiMark: 无偏多层大型语言模型水印 

**Authors**: Xiaoyan Feng, He Zhang, Yanjun Zhang, Leo Yu Zhang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.21602)  

**Abstract**: Recent advances in Large Language Models (LLMs) have raised urgent concerns about LLM-generated text authenticity, prompting regulatory demands for reliable identification mechanisms. Although watermarking offers a promising solution, existing approaches struggle to simultaneously achieve three critical requirements: text quality preservation, model-agnostic detection, and message embedding capacity, which are crucial for practical implementation. To achieve these goals, the key challenge lies in balancing the trade-off between text quality preservation and message embedding capacity. To address this challenge, we propose BiMark, a novel watermarking framework that achieves these requirements through three key innovations: (1) a bit-flip unbiased reweighting mechanism enabling model-agnostic detection, (2) a multilayer architecture enhancing detectability without compromising generation quality, and (3) an information encoding approach supporting multi-bit watermarking. Through theoretical analysis and extensive experiments, we validate that, compared to state-of-the-art multi-bit watermarking methods, BiMark achieves up to 30% higher extraction rates for short texts while maintaining text quality indicated by lower perplexity, and performs comparably to non-watermarked text on downstream tasks such as summarization and translation. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）的发展引发了对其生成文本真实性的急切关注，推动了对可靠识别机制的监管要求。虽然水印提供了一种有潜力的解决方案，但现有方法在同时满足以下三个关键要求方面存在困难：文本质量保留、模型无关的检测能力和水印信息嵌入容量，这对于实际应用至关重要。为了实现这些目标，关键挑战在于在文本质量保留和信息嵌入容量之间找到平衡。为应对这一挑战，我们提出了一种名为BiMark的新型水印框架，通过三种创新实现这些要求：（1）无偏重采样机制以实现模型无关的检测；（2）多层结构提高可检测性而不牺牲生成质量；（3）信息编码方法支持多比特水印。通过理论分析和大量实验，我们验证了与最新多比特水印方法相比，BiMark在短文本提取率方面提高高达30%，同时保持较低困惑度指示的文本质量，并在摘要和翻译等下游任务中表现与未水印文本相当。 

---
# Structured Attention Matters to Multimodal LLMs in Document Understanding 

**Title (ZH)**: 结构化注意力对文档理解中的多模态大语言模型至关重要 

**Authors**: Chang Liu, Hongkai Chen, Yujun Cai, Hang Wu, Qingwen Ye, Ming-Hsuan Yang, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21600)  

**Abstract**: Document understanding remains a significant challenge for multimodal large language models (MLLMs). While previous research has primarily focused on locating evidence pages through precise multimodal queries, our work investigates a fundamental yet overlooked aspect: how input format influences document comprehension performance. Through systematic analysis, we discover that raw OCR text often impairs rather than improves MLLMs' performance, which is a counterintuitive finding we attribute to attention dispersion and structure loss. To further substantiate our hypothesis, we propose a novel structure-preserving approach that encodes document elements using the LaTex paradigm, maintaining the hierarchical organization and spatial relationships critical for comprehension. Our attention analysis reveals that structured text induces structured attention patterns on both textual and visual content, directing models to focus on semantically meaningful regions while reducing attention waste. This approach significantly enhances MLLMs' document question answering performance across diverse document types without requiring architectural modifications or additional training. 

**Abstract (ZH)**: 多模态大语言模型的文档理解仍是一个重要的挑战。尽管以往研究主要侧重于通过精确的多模态查询定位证据页，我们的工作探讨了一个基础而被忽视的方面：输入格式如何影响文档理解性能。通过系统分析，我们发现原始OCR文本往往损害而不是提升多模态大语言模型的性能，这是一个反直觉的发现，我们将其归因于注意力分散和结构损失。为进一步验证这一假设，我们提出了一种新颖的结构保留方法，使用LaTex paradigm编码文档元素，保持对理解至关重要的层次组织和空间关系。我们的注意力分析揭示，结构化文本在文本和视觉内容上诱导了结构化的注意力模式，促使模型专注于语义上有意义的区域，同时减少注意力浪费。该方法显著提升了多模态大语言模型在多种文档类型下的文档问答性能，无需进行架构修改或额外训练。 

---
# Reinforcement Fine-Tuned Large Language Models for Next POI Recommendation 

**Title (ZH)**: 强化微调大语言模型用于下一个POI推荐 

**Authors**: Peibo Li, Shuang Ao, Hao Xue, Yang Song, Maarten de Rijke, Johan Barthélemy, Tomasz Bednarz, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21599)  

**Abstract**: Large language models (LLMs) have been adopted for next point-of-interest (POI) recommendation tasks. Typical LLM-based recommenders fall into two categories: prompt-based and supervised fine-tuning (SFT)-based models. Prompt-based models generally offer greater output flexibility but deliver lower accuracy, whereas SFT-based models achieve higher performance yet face a fundamental mismatch: next POI recommendation data does not naturally suit supervised fine-tuning. In SFT, the model is trained to reproduce the exact ground truth, but each training example provides only a single target POI, so there is no ground truth for producing a top-k list.
To address this, we propose Refine-POI, a reinforcement fine-tuning framework for next POI recommendation. We introduce recommendation-driven rewards that enable LLMs to learn to generate top-k recommendation lists using only one ground-truth POI per example. Experiments on real-world datasets demonstrate that Refine-POI achieves state-of-the-art top-k recommendation performance. 

**Abstract (ZH)**: 基于强化微调的下一点-of-interest推荐方法 

---
# Overview of the ClinIQLink 2025 Shared Task on Medical Question-Answering 

**Title (ZH)**: 2025 ClinIQLink 医学问答共享任务概述 

**Authors**: Brandon Colelough, Davis Bartels, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2506.21597)  

**Abstract**: In this paper, we present an overview of ClinIQLink, a shared task, collocated with the 24th BioNLP workshop at ACL 2025, designed to stress-test large language models (LLMs) on medically-oriented question answering aimed at the level of a General Practitioner. The challenge supplies 4,978 expert-verified, medical source-grounded question-answer pairs that cover seven formats: true/false, multiple choice, unordered list, short answer, short-inverse, multi-hop, and multi-hop-inverse. Participating systems, bundled in Docker or Apptainer images, are executed on the CodaBench platform or the University of Maryland's Zaratan cluster. An automated harness (Task 1) scores closed-ended items by exact match and open-ended items with a three-tier embedding metric. A subsequent physician panel (Task 2) audits the top model responses. 

**Abstract (ZH)**: 本文介绍了ClinIQLink，一个在ACL 2025的第24届BioNLP研讨会期间举办的共同任务，旨在通过针对普通医生层次的医学导向问答来压力测试大型语言模型（LLMs）。该挑战提供了4,978个专家验证、医学来源支持的问题-答案对，涵盖了七种格式：True/False、多项选择、无序列表、简短回答、简短逆向、多跳和多跳逆向。参赛系统打包为Docker或Apptainer镜像，在CodaBench平台或马里兰大学的Zaratan集群上执行。自动评分（任务1）通过精确匹配评分封闭式问题，并使用三级嵌入度量评分开放式问题。随后，由医生组成的评审小组（任务2）审核顶级模型的回答。 

---
# Evaluating Multimodal Large Language Models on Educational Textbook Question Answering 

**Title (ZH)**: 多模态大型语言模型在教育教材问答中的评估 

**Authors**: Hessa A. Alawwad, Anas Zafar, Areej Alhothali, Usman Naseem, Ali Alkhathlan, Amani Jamal  

**Link**: [PDF](https://arxiv.org/pdf/2506.21596)  

**Abstract**: Multimodal large language models (MLLMs) have recently achieved significant success in vision--language tasks. However, their capacity to reason over complex, long lessons and intricate educational diagrams that cannot be represented as a single natural image remains largely untested. In this work, we present the first evaluation of state-of-the-art MLLMs on the textbook question answering (TQA) task using the CK12-QA dataset. We assess the performance of recent vision-language models, including LLaVA and LLaMA 3.2-Vision, across various input configurations. Additionally, we introduce a lightweight multimodal retrieval-augmented generation (RAG) pipeline that integrates both paragraphs and diagrams from the lesson into the prompt. Our results demonstrate the influence of retrieved educational context on model accuracy and reasoning, while also revealing current limitations in handling question-context relationships and the potential for noise, pointing to key directions for future research in multimodal AI-driven learning. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在视觉-语言任务中最近取得了显著成功。然而，它们在处理复杂的长课程内容和 intricate 教育图表方面的推理能力，这些内容和图表不能仅通过单张自然图像来表示，仍主要未被测试。在本工作中，我们首次使用 CK12-QA 数据集评估了最先进的 MLLMs 在教材问答（TQA）任务上的表现。我们在多种输入配置下评估了最近的视觉-语言模型，包括 LLaVA 和 LLaMA 3.2-Vision。此外，我们引入了一种轻量级的多模态检索增强生成（RAG）管道，将课程中的段落和图表整合到提示中。我们的结果表明检索到的教育背景对模型准确性和推理的影响，并揭示了当前处理问题-背景关系的局限性以及噪声带来的潜在问题，指出了未来多模态 AI 驱动学习研究的关键方向。 

---
# Can Vision Language Models Understand Mimed Actions? 

**Title (ZH)**: 视觉语言模型能理解模仿动作吗？ 

**Authors**: Hyundong Cho, Spencer Lin, Tejas Srinivasan, Michael Saxon, Deuksin Kwon, Natali T. Chavez, Jonathan May  

**Link**: [PDF](https://arxiv.org/pdf/2506.21586)  

**Abstract**: Nonverbal communication (NVC) plays an integral role in human language, but studying NVC in general is challenging because of its broad scope and high variance in interpretation among individuals and cultures. However, mime -- the theatrical technique of suggesting intent using only gesture, expression, and movement -- is a subset of NVC that consists of explicit and embodied actions with much lower human interpretation variance. We argue that a solid understanding of mimed actions is a crucial prerequisite for vision-language models capable of interpreting and commanding more subtle aspects of NVC. Hence, we propose Mime Identification Multimodal Evaluation (MIME), a novel video-based question answering benchmark comprising of 86 mimed actions. Constructed with motion capture data, MIME consists of variations of each action with perturbations applied to the character, background, and viewpoint for evaluating recognition robustness. We find that both open-weight and API-based vision-language models perform significantly worse than humans on MIME, motivating the need for increased research for instilling more robust understanding of human gestures. 

**Abstract (ZH)**: 基于视频的 Mime 识别多模态评价基准（MIME）：理解非言语沟通中的关键要素 

---
# Empirical Evidence for Alignment Faking in Small LLMs and Prompt-Based Mitigation Techniques 

**Title (ZH)**: 小规模语言模型中对齐仿冒的实证证据及基于提示的缓解技术 

**Authors**: J. Koorndijk  

**Link**: [PDF](https://arxiv.org/pdf/2506.21584)  

**Abstract**: Current literature suggests that alignment faking (deceptive alignment) is an emergent property of large language models. We present the first empirical evidence that a small instruction-tuned model, specifically LLaMA 3 8B, can also exhibit alignment faking. We further show that prompt-only interventions, including deontological moral framing and scratchpad reasoning, significantly reduce this behavior without modifying model internals. This challenges the assumption that prompt-based ethics are trivial and that deceptive alignment requires scale. We introduce a taxonomy distinguishing shallow deception, shaped by context and suppressible through prompting, from deep deception, which reflects persistent, goal-driven misalignment. Our findings refine the understanding of deception in language models and underscore the need for alignment evaluations across model sizes and deployment settings. 

**Abstract (ZH)**: 当前文献表明，对齐伪装（欺骗性对齐）是大规模语言模型的一个新兴特性。我们首次提供了实验证据，证明一个小规模指令微调模型，即LLaMA 3 8B，也能表现出对齐伪装。我们进一步表明，仅通过提示干预，包括道义论道德框架和工作区推理，可以显著减少这种行为而不修改模型内部结构。这挑战了提示伦理学简单易行以及欺骗性对齐需要大规模模型的假设。我们引入了一个分类体系，将由上下文塑造并通过提示抑制的浅层欺骗与反映持续且目标驱动的错对齐的深层欺骗区分开来。我们的研究结果细化了对语言模型中欺骗的理解，并强调了在不同模型规模和部署场景下进行对齐评估的必要性。 

---
# Hope Speech Detection in code-mixed Roman Urdu tweets: A Positive Turn in Natural Language Processing 

**Title (ZH)**: 代码混合罗马乌尔都语推文中希望演说检测：自然语言处理的积极转捩点 

**Authors**: Muhammad Ahmad, Muhammad Waqas, Ameer Hamza, Ildar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2506.21583)  

**Abstract**: Hope is a positive emotional state involving the expectation of favorable future outcomes, while hope speech refers to communication that promotes optimism, resilience, and support, particularly in adverse contexts. Although hope speech detection has gained attention in Natural Language Processing (NLP), existing research mainly focuses on high-resource languages and standardized scripts, often overlooking informal and underrepresented forms such as Roman Urdu. To the best of our knowledge, this is the first study to address hope speech detection in code-mixed Roman Urdu by introducing a carefully annotated dataset, thereby filling a critical gap in inclusive NLP research for low-resource, informal language varieties. This study makes four key contributions: (1) it introduces the first multi-class annotated dataset for Roman Urdu hope speech, comprising Generalized Hope, Realistic Hope, Unrealistic Hope, and Not Hope categories; (2) it explores the psychological foundations of hope and analyzes its linguistic patterns in code-mixed Roman Urdu to inform dataset development; (3) it proposes a custom attention-based transformer model optimized for the syntactic and semantic variability of Roman Urdu, evaluated using 5-fold cross-validation; and (4) it verifies the statistical significance of performance gains using a t-test. The proposed model, XLM-R, achieves the best performance with a cross-validation score of 0.78, outperforming the baseline SVM (0.75) and BiLSTM (0.76), with gains of 4% and 2.63% respectively. 

**Abstract (ZH)**: 希望状态涉及对未来有利结果的期望，而希望言论是指在不利环境中促进乐观、韧性和支持的沟通。尽管希望言论检测在自然语言处理（NLP）中引起了关注，但现有研究主要集中在资源丰富语言和标准化书写系统上，往往忽略了如混合罗马乌都语等非正式和未充分代表的形式。据我们所知，这是首次通过引入精心标注的数据集来解决混合罗马乌都语希望言论检测问题的研究，从而填补了包容性NLP研究中低资源、非正式语言变体的重要空白。本研究做出了四个关键贡献：（1）首次为罗马乌都语希望言论引入了多类标注数据集，包括普遍希望、现实希望、不切实际希望和非希望类别；（2）探讨了希望的心理基础并分析了混合罗马乌都语中的语言模式，以指导数据集开发；（3）提出了一种针对罗马乌都语句法和语义变异性优化的自注意力变压器模型，并使用5折交叉验证进行评估；（4）使用T检验验证了性能提升的统计显著性。所提出的模型XLM-R在交叉验证得分为0.78的情况下表现出最佳性能，优于基线SVM（0.75）和BiLSTM（0.76），分别提高了4%和2.63%。 

---
# VIDEE: Visual and Interactive Decomposition, Execution, and Evaluation of Text Analytics with Intelligent Agents 

**Title (ZH)**: VIDEE: 可视化和交互式分解、执行与评估文本分析的智能代理方法 

**Authors**: Sam Yu-Te Lee, Chengyang Ji, Shicheng Wen, Lifu Huang, Dongyi Liu, Kwan-Liu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.21582)  

**Abstract**: Text analytics has traditionally required specialized knowledge in Natural Language Processing (NLP) or text analysis, which presents a barrier for entry-level analysts. Recent advances in large language models (LLMs) have changed the landscape of NLP by enabling more accessible and automated text analysis (e.g., topic detection, summarization, information extraction, etc.). We introduce VIDEE, a system that supports entry-level data analysts to conduct advanced text analytics with intelligent agents. VIDEE instantiates a human-agent collaroration workflow consisting of three stages: (1) Decomposition, which incorporates a human-in-the-loop Monte-Carlo Tree Search algorithm to support generative reasoning with human feedback, (2) Execution, which generates an executable text analytics pipeline, and (3) Evaluation, which integrates LLM-based evaluation and visualizations to support user validation of execution results. We conduct two quantitative experiments to evaluate VIDEE's effectiveness and analyze common agent errors. A user study involving participants with varying levels of NLP and text analytics experience -- from none to expert -- demonstrates the system's usability and reveals distinct user behavior patterns. The findings identify design implications for human-agent collaboration, validate the practical utility of VIDEE for non-expert users, and inform future improvements to intelligent text analytics systems. 

**Abstract (ZH)**: 文本分析传统上需要自然语言处理(NLP)或文本分析的专门知识，这给入门级分析师设置了障碍。大型语言模型（LLMs）的 recent 进展通过使文本分析更加易用和自动化（例如：主题检测、总结、信息提取等）改变了 NLP 的格局。我们引入了 VIDEE 系统，支持入门级数据分析师通过智能代理进行高级文本分析。VIDEE 实现了一个包含三个阶段的人机协作工作流：（1）分解，该阶段结合了带有人类反馈的循环蒙特卡洛树搜索算法以支持生成式推理；（2）执行，该阶段生成可执行的文本分析流水线；（3）评估，该阶段结合了基于大语言模型的评估和可视化以支持用户对执行结果的验证。我们进行了两项定量实验以评估 VIDEE 的有效性并分析常见代理错误。一项涉及不同程度 NLP 和文本分析经验的参与者（从无到专家）的研究表明该系统的可用性，并揭示了不同的用户行为模式。研究结果指出了人机协作的设计启示，验证了 VIDEE 对非专家用户的实际效用，并为未来的智能文本分析系统改进提供了信息。 

---
# Evaluating the Robustness of Dense Retrievers in Interdisciplinary Domains 

**Title (ZH)**: 评估不同学科领域中密集检索器的鲁棒性 

**Authors**: Sarthak Chaturvedi, Anurag Acharya, Rounak Meyur, Koby Hayashi, Sai Munikoti, Sameera Horawalavithana  

**Link**: [PDF](https://arxiv.org/pdf/2506.21581)  

**Abstract**: Evaluation benchmark characteristics may distort the true benefits of domain adaptation in retrieval models. This creates misleading assessments that influence deployment decisions in specialized domains. We show that two benchmarks with drastically different features such as topic diversity, boundary overlap, and semantic complexity can influence the perceived benefits of fine-tuning. Using environmental regulatory document retrieval as a case study, we fine-tune ColBERTv2 model on Environmental Impact Statements (EIS) from federal agencies. We evaluate these models across two benchmarks with different semantic structures. Our findings reveal that identical domain adaptation approaches show very different perceived benefits depending on evaluation methodology. On one benchmark, with clearly separated topic boundaries, domain adaptation shows small improvements (maximum 0.61% NDCG gain). However, on the other benchmark with overlapping semantic structures, the same models demonstrate large improvements (up to 2.22% NDCG gain), a 3.6-fold difference in the performance benefit. We compare these benchmarks through topic diversity metrics, finding that the higher-performing benchmark shows 11% higher average cosine distances between contexts and 23% lower silhouette scores, directly contributing to the observed performance difference. These results demonstrate that benchmark selection strongly determines assessments of retrieval system effectiveness in specialized domains. Evaluation frameworks with well-separated topics regularly underestimate domain adaptation benefits, while those with overlapping semantic boundaries reveal improvements that better reflect real-world regulatory document complexity. Our findings have important implications for developing and deploying AI systems for interdisciplinary domains that integrate multiple topics. 

**Abstract (ZH)**: 评价基准特征可能会歪曲领域适应在检索模型中的真正益处。这会形成误导性的评估，影响专门领域的部署决策。我们通过环境监管文件检索案例研究，发现具有截然不同特征（如主题多样性、边界重叠和语义复杂性）的两个基准可以影响微调感知益处。我们在联邦机构的环境影响声明（EIS）上对ColBERTv2模型进行微调，并在具有不同语义结构的两个基准上进行评估。我们的研究发现，相同的领域适应方法在不同的评估方法下表现出非常不同的感知益处。在具有明显分开的主题边界的基准上，领域适应显示出微小的改善（最大0.61%的NDCG增益）。而在具有重叠语义结构的基准上，相同的模型则显示出显著的改善（最高2.22%的NDCG增益），性能效益提升达3.6倍。我们通过主题多样性指标比较这些基准，发现表现更好的基准显示了11%更高的平均余弦距离和23%更低的轮廓评分，直接导致观察到的性能差异。这些结果表明，基准选择强烈决定了在专门领域的检索系统效果评估。具有清晰主题区隔的评价框架通常会低估领域适应的益处，而具有重叠语义边界的框架则揭示了更能反映实际监管文件复杂性的改进。我们的研究结果对于开发和部署整合多个主题的跨学科领域的AI系统具有重要启示意义。 

---
# From General Reasoning to Domain Expertise: Uncovering the Limits of Generalization in Large Language Models 

**Title (ZH)**: 从通用推理到领域专业知识：揭示大型语言模型泛化的极限 

**Authors**: Dana Alsagheer, Yang Lu, Abdulrahman Kamal, Omar Kamal, Mohammad Kamal, Nada Mansour, Cosmo Yang Wu, Rambiba Karanjai, Sen Li, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21580)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated remarkable capabilities in various domains. However, effective decision-making relies heavily on strong reasoning abilities. Reasoning is the foundation for decision-making, providing the analytical and logical framework to make sound choices. Reasoning involves analyzing information, drawing inferences, and reaching conclusions based on logic or evidence. Decision-making builds on this foundation by applying the insights from reasoning to select the best course of action among alternatives. Together, these processes create a continuous cycle of thought and action aimed at achieving goals effectively. As AI technology evolves, there is a growing trend to train LLMs to excel in general reasoning. This study explores how the general reasoning capabilities of LLMs connect to their performance in domain-specific reasoning tasks. 

**Abstract (ZH)**: 最近Large Language Models (LLMs)的进展展示了其在各个领域的卓越能力。然而，有效的决策高度依赖于强大的推理能力。推理是决策的基础，提供了分析和逻辑框架以做出合理的选择。推理涉及分析信息、推断和基于逻辑或证据得出结论。决策在此基础上利用推理所得的见解在多种选择中选择最佳行动方案。这些过程共同形成了一种旨在有效实现目标的思考与行动的连续循环。随着AI技术的发展，训练LLMs在通用推理方面表现出色的趋势日益增长。本研究探讨了LLMs的通用推理能力与其在特定领域推理任务中的表现之间的联系。 

---
# LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation 

**Title (ZH)**: LLM2Rec: 大型语言模型是强大的序列推荐嵌入模型 

**Authors**: Yingzhi He, Xiaohao Liu, An Zhang, Yunshan Ma, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.21579)  

**Abstract**: Sequential recommendation aims to predict users' future interactions by modeling collaborative filtering (CF) signals from historical behaviors of similar users or items. Traditional sequential recommenders predominantly rely on ID-based embeddings, which capture CF signals through high-order co-occurrence patterns. However, these embeddings depend solely on past interactions, lacking transferable knowledge to generalize to unseen domains. Recent advances in large language models (LLMs) have motivated text-based recommendation approaches that derive item representations from textual descriptions. While these methods enhance generalization, they fail to encode CF signals-i.e., latent item correlations and preference patterns-crucial for effective recommendation. We argue that an ideal embedding model should seamlessly integrate CF signals with rich semantic representations to improve both in-domain and out-of-domain recommendation performance.
To this end, we propose LLM2Rec, a novel embedding model tailored for sequential recommendation, integrating the rich semantic understanding of LLMs with CF awareness. Our approach follows a two-stage training framework: (1) Collaborative Supervised Fine-tuning, which adapts LLMs to infer item relationships based on historical interactions, and (2) Item-level Embedding Modeling, which refines these specialized LLMs into structured item embedding models that encode both semantic and collaborative information. Extensive experiments on real-world datasets demonstrate that LLM2Rec effectively improves recommendation quality across both in-domain and out-of-domain settings. Our findings highlight the potential of leveraging LLMs to build more robust, generalizable embedding models for sequential recommendation. Our codes are available at this https URL. 

**Abstract (ZH)**: 基于大型语言模型的序列推荐：结合丰富的语义理解与协同过滤意识 

---
# HealthQA-BR: A System-Wide Benchmark Reveals Critical Knowledge Gaps in Large Language Models 

**Title (ZH)**: HealthQA-BR：一个系统级基准揭示了大型语言模型中的关键知识缺口 

**Authors**: Andrew Maranhão Ventura D'addario  

**Link**: [PDF](https://arxiv.org/pdf/2506.21578)  

**Abstract**: The evaluation of Large Language Models (LLMs) in healthcare has been dominated by physician-centric, English-language benchmarks, creating a dangerous illusion of competence that ignores the interprofessional nature of patient care. To provide a more holistic and realistic assessment, we introduce HealthQA-BR, the first large-scale, system-wide benchmark for Portuguese-speaking healthcare. Comprising 5,632 questions from Brazil's national licensing and residency exams, it uniquely assesses knowledge not only in medicine and its specialties but also in nursing, dentistry, psychology, social work, and other allied health professions. We conducted a rigorous zero-shot evaluation of over 20 leading LLMs. Our results reveal that while state-of-the-art models like GPT 4.1 achieve high overall accuracy (86.6%), this top-line score masks alarming, previously unmeasured deficiencies. A granular analysis shows performance plummets from near-perfect in specialties like Ophthalmology (98.7%) to barely passing in Neurosurgery (60.0%) and, most notably, Social Work (68.4%). This "spiky" knowledge profile is a systemic issue observed across all models, demonstrating that high-level scores are insufficient for safety validation. By publicly releasing HealthQA-BR and our evaluation suite, we provide a crucial tool to move beyond single-score evaluations and toward a more honest, granular audit of AI readiness for the entire healthcare team. 

**Abstract (ZH)**: HealthQA-BR：面向葡萄牙语医疗保健的首个大型系统性基准 

---
# Language-Aware Prompt Tuning for Parameter-Efficient Seamless Language Expansion in Multilingual ASR 

**Title (ZH)**: 面向语言感知的提示调谐以实现多语言ASR中的参数高效无缝语言扩展 

**Authors**: Hongli Yang, Sheng Li, Hao Huang, Ayiduosi Tuohan, Yizhou Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21577)  

**Abstract**: Recent advancements in multilingual automatic speech recognition (ASR) have been driven by large-scale end-to-end models like Whisper. However, challenges such as language interference and expanding to unseen languages (language expansion) without degrading performance persist. This paper addresses these with three contributions: 1) Entire Soft Prompt Tuning (Entire SPT), which applies soft prompts to both the encoder and decoder, enhancing feature extraction and decoding; 2) Language-Aware Prompt Tuning (LAPT), which leverages cross-lingual similarities to encode shared and language-specific features using lightweight prompt matrices; 3) SPT-Whisper, a toolkit that integrates SPT into Whisper and enables efficient continual learning. Experiments across three languages from FLEURS demonstrate that Entire SPT and LAPT outperform Decoder SPT by 5.0% and 16.0% in language expansion tasks, respectively, providing an efficient solution for dynamic, multilingual ASR models with minimal computational overhead. 

**Abstract (ZH)**: Recent advancements in多语言自动语音识别（ASR）的进步受到大规模端到端模型如Whisper的驱动。然而，语言干扰和在不降低性能的情况下扩展到未见语言（语言扩展）的挑战仍然存在。本文通过以下三项贡献来应对这些挑战：1) 整体软提示调整（Entire SPT），该方法在编码器和解码器中应用软提示，增强特征提取和解码；2) 语言感知提示调整（LAPT），该方法利用跨语言相似性通过轻量级提示矩阵编码共享和语言特定特征；3) SPT-Whisper工具包，将SPT集成到Whisper中，实现高效持续学习。跨FLEURS的三种语言实验表明，总体SPT和LAPT在语言扩展任务中分别比解码器SPT表现出5.0%和16.0%的优越性，为动态多语言ASR模型提供了一个具有最小计算开销的有效解决方案。 

---
# Adapting Whisper for Parameter-efficient Code-Switching Speech Recognition via Soft Prompt Tuning 

**Title (ZH)**: 基于软提示调谐的参数高效代码切换语音识别适应性研究 

**Authors**: Hongli Yang, Yizhou Peng, Hao Huang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21576)  

**Abstract**: Large-scale multilingual ASR models like Whisper excel in high-resource settings but face challenges in low-resource scenarios, such as rare languages and code-switching (CS), due to computational costs and catastrophic forgetting. We explore Soft Prompt Tuning (SPT), a parameter-efficient method to enhance CS ASR while preserving prior knowledge. We evaluate two strategies: (1) full fine-tuning (FFT) of both soft prompts and the entire Whisper model, demonstrating improved cross-lingual capabilities compared to traditional methods, and (2) adhering to SPT's original design by freezing model parameters and only training soft prompts. Additionally, we introduce SPT4ASR, a combination of different SPT variants. Experiments on the SEAME and ASRU2019 datasets show that deep prompt tuning is the most effective SPT approach, and our SPT4ASR methods achieve further error reductions in CS ASR, maintaining parameter efficiency similar to LoRA, without degrading performance on existing languages. 

**Abstract (ZH)**: 大规模多语言ASR模型如Whisper在资源丰富环境下表现出色，但在资源匮乏场景如稀有语言和代码切换（CS）中面临计算成本和灾难性遗忘的挑战。我们探索了软提示调谐（SPT）这一参数高效的方法，以增强代码切换（CS）ASR的同时保留先前知识。我们评估了两种策略：（1）完整调谐（FFT）软提示和整个Whisper模型，显示了与传统方法相比改进的跨语言能力，以及（2）遵循SPT原始设计，冻结模型参数仅训练软提示。此外，我们引入了SPT4ASR，这是一种不同SPT变体的组合。在SEAME和ASRU2019数据集上的实验表明，深层提示调谐是效果最佳的SPT方法，我们的SPT4ASR方法在代码切换ASR中实现了进一步的错误减少，保持了与LoRA相似的参数效率，且不牺牲现有语言上的性能。 

---
# STRuCT-LLM: Unifying Tabular and Graph Reasoning with Reinforcement Learning for Semantic Parsing 

**Title (ZH)**: STRuCT-LLM：通过强化学习统一表格和图推理的语义解析 

**Authors**: Josefa Lia Stoisser, Marc Boubnovski Martell, Lawrence Phillips, Casper Hansen, Julien Fauqueur  

**Link**: [PDF](https://arxiv.org/pdf/2506.21575)  

**Abstract**: We propose STRuCT-LLM, a unified framework for training large language models (LLMs) to perform structured reasoning over both relational and graph-structured data. Our approach jointly optimizes Text-to-SQL and Text-to-Cypher tasks using reinforcement learning (RL) combined with Chain-of-Thought (CoT) supervision. To support fine-grained optimization in graph-based parsing, we introduce a topology-aware reward function based on graph edit distance. Unlike prior work that treats relational and graph formalisms in isolation, STRuCT-LLM leverages shared abstractions between SQL and Cypher to induce cross-formalism transfer, enabling SQL training to improve Cypher performance and vice versa - even without shared schemas. Our largest model (QwQ-32B) achieves substantial relative improvements across tasks: on semantic parsing, Spider improves by 13.5\% and Text2Cypher by 73.1\%. The model also demonstrates strong zero-shot generalization, improving performance on downstream tabular QA (TableBench: 8.5\%) and knowledge graph QA (CR-LT-KGQA: 1.7\%) without any QA-specific supervision. These results demonstrate both the effectiveness of executable queries as scaffolds for structured reasoning and the synergistic benefits of jointly training on SQL and Cypher (code available at this https URL). 

**Abstract (ZH)**: 我们提出STRuCT-LLM，这是一种统一框架，用于训练大型语言模型（LLMs）在关系性和图形结构数据上进行结构化推理。我们的方法通过结合强化学习（RL）和链式思考（CoT）监督，联合优化从文本到SQL和从文本到Cypher的任务。为支持基于图的解析的细粒度优化，我们引入了一个基于图编辑距离的拓扑感知奖励函数。与先前工作将关系性和图形形式主义孤立处理不同，STRuCT-LLM 利用 SQL 和 Cypher 之间的共享抽象来促进跨形式主义的迁移，从而使 SQL 训练能够提升 Cypher 性能，反之亦然——即使没有共享模式。我们最大的模型（QwQ-32B）在任务上取得了显著的相对改进：在语义解析方面，Spider 提高了13.5%，Text2Cypher 提高了73.1%。该模型还展示了强大的零样本泛化能力，在下游表型问答（TableBench）和知识图问答（CR-LT-KGQA）任务上未进行任何问答特定监督的情况下，性能分别提升了8.5%和1.7%。这些结果表明可执行查询作为结构化推理支架的有效性，以及在 SQL 和 Cypher 上联合训练的协同优势（相关代码可在以下网址获取：this https URL）。 

---
# Digital Gatekeepers: Exploring Large Language Model's Role in Immigration Decisions 

**Title (ZH)**: 数字gatekeeper：探索大型语言模型在移民决策中的作用 

**Authors**: Yicheng Mao, Yang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.21574)  

**Abstract**: With globalization and increasing immigrant populations, immigration departments face significant work-loads and the challenge of ensuring fairness in decision-making processes. Integrating artificial intelligence offers a promising solution to these challenges. This study investigates the potential of large language models (LLMs),such as GPT-3.5 and GPT-4, in supporting immigration decision-making. Utilizing a mixed-methods approach,this paper conducted discrete choice experiments and in-depth interviews to study LLM decision-making strategies and whether they are fair. Our findings demonstrate that LLMs can align their decision-making with human strategies, emphasizing utility maximization and procedural fairness. Meanwhile, this paper also reveals that while ChatGPT has safeguards to prevent unintentional discrimination, it still exhibits stereotypes and biases concerning nationality and shows preferences toward privileged group. This dual analysis highlights both the potential and limitations of LLMs in automating and enhancing immigration decisions. 

**Abstract (ZH)**: 全球化和移民人口增加背景下，移民部门面临巨大工作压力和确保决策过程公平的挑战。集成人工智能提供了一种有前景的解决方案。本研究探讨了大型语言模型（LLMs），如GPT-3.5和GPT-4，在支持移民决策中的潜在应用。采用混合方法，本文通过离散选择实验和深度访谈研究了LLM的决策策略及其公平性。研究发现，LLMs能够与其人类策略相一致，强调效用最大化和程序公平。同时，本文还揭示了尽管ChatGPT有防止无意中歧视的机制，但它仍表现出国籍方面的刻板印象和偏见，并倾向于偏好特权群体。这种双重分析突显了LLMs在自动化和提升移民决策方面的同时潜力与局限性。 

---
# Instruction Learning Paradigms: A Dual Perspective on White-box and Black-box LLMs 

**Title (ZH)**: 白箱与黑箱大型语言模型的双重视角：指令学习范式 

**Authors**: Yanwei Ren, Liu Liu, Baosheng Yu, Jiayan Qiu, Quan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21573)  

**Abstract**: Optimizing instructions for large language models (LLMs) is critical for harnessing their full potential in complex and diverse tasks. However, relying solely on white-box approaches demands extensive computational resources and offers limited representational capacity, while black-box models can incur prohibitive financial costs. To address these challenges, we introduce a novel framework that seamlessly merges the strengths of both paradigms. Black-box models provide high-quality, diverse instruction initializations, and white-box models supply fine-grained interpretability through hidden states and output features. By enforcing a semantic similarity constraint, these components fuse into a unified high-dimensional representation that captures deep semantic and structural nuances, enabling an iterative optimization process to refine instruction quality and adaptability. Extensive evaluations across a broad spectrum of tasks-ranging from complex reasoning to cross-lingual generalization-demonstrate that our approach consistently outperforms state-of-the-art baselines. This fusion of black-box initialization with advanced semantic refinement yields a scalable and efficient solution, paving the way for next-generation LLM-driven applications in diverse real-world scenarios. The source code will be released soon. 

**Abstract (ZH)**: 优化大型语言模型的指令对于充分发挥其在复杂多样的任务中的潜力至关重要。然而，仅依赖白盒方法需要大量计算资源并提供有限的表示能力，而黑盒模型则可能带来高昂的财务成本。为应对这些挑战，我们提出了一种新的框架，无缝整合了两种范式的优点。黑盒模型提供高质量的多样化初始指令，而白盒模型则通过隐藏状态和输出特征提供精细的可解释性。通过施加语义相似性约束，这些组件融合成一个统一的高维表示，能够捕获深层次的语义和结构细微差别，从而促进迭代优化过程以提高指令质量和适应性。广泛的任务评估（范围涵盖从复杂推理到跨语言泛化的各个方面）表明，我们的方法在各项基准上表现优异。黑盒初始化与高级语义细化的结合提供了可扩展且高效的解决方案，为下一代基于语言模型的应用铺平了道路。源代码即将发布。 

---
# Towards Understanding the Cognitive Habits of Large Reasoning Models 

**Title (ZH)**: Towards Understanding Large Reasoning Models的认知习惯 

**Authors**: Jianshuo Dong, Yujia Fu, Chuanrui Hu, Chao Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21571)  

**Abstract**: Large Reasoning Models (LRMs), which autonomously produce a reasoning Chain of Thought (CoT) before producing final responses, offer a promising approach to interpreting and monitoring model behaviors. Inspired by the observation that certain CoT patterns -- e.g., ``Wait, did I miss anything?'' -- consistently emerge across tasks, we explore whether LRMs exhibit human-like cognitive habits. Building on Habits of Mind, a well-established framework of cognitive habits associated with successful human problem-solving, we introduce CogTest, a principled benchmark designed to evaluate LRMs' cognitive habits. CogTest includes 16 cognitive habits, each instantiated with 25 diverse tasks, and employs an evidence-first extraction method to ensure reliable habit identification. With CogTest, we conduct a comprehensive evaluation of 16 widely used LLMs (13 LRMs and 3 non-reasoning ones). Our findings reveal that LRMs, unlike conventional LLMs, not only exhibit human-like habits but also adaptively deploy them according to different tasks. Finer-grained analyses further uncover patterns of similarity and difference in LRMs' cognitive habit profiles, particularly certain inter-family similarity (e.g., Qwen-3 models and DeepSeek-R1). Extending the study to safety-related tasks, we observe that certain habits, such as Taking Responsible Risks, are strongly associated with the generation of harmful responses. These findings suggest that studying persistent behavioral patterns in LRMs' CoTs is a valuable step toward deeper understanding of LLM misbehavior. The code is available at: this https URL. 

**Abstract (ZH)**: 大型推理模型（LRMs）自主生成推理链（CoT）后再产生最终响应，为理解和监控模型行为提供了有 promise 的方法。受某些 CoT 模式（例如，“等等，我有遗漏什么吗？”）在不同任务中一致出现的启发，我们探索 LRMs 是否表现出类似人类的认知习惯。基于 Habits of Mind 这一成熟的成功人类问题解决的认知习惯框架，我们提出了 CogTest，这是一种有原则的基准测试，用于评估 LRMs 的认知习惯。CogTest 包含 16 种认知习惯，每种习惯实例化了 25 个不同的任务，并采用证据优先提取方法以确保可靠的习惯识别。使用 CogTest，我们对 16 种广泛使用的语言模型（13 种 LRMs 和 3 种非推理模型）进行了全面评估。我们的发现表明，LRMs 不仅表现出类似人类的习惯，还能根据不同任务进行适应性部署。更细粒度的分析进一步揭示了 LRMs 认知习惯模式中的相似性和差异性，特别是某些家庭内的相似性（例如 Qwen-3 模型和 DeepSeek-R1）。将研究扩展到安全相关任务，我们观察到某些习惯（例如，承担负责任的风险）与有害响应的生成有强烈关联。这些发现表明，研究 LRMs 的 CoT 中持久的行为模式是更深入理解 LLM 行为的关键步骤。代码可在以下链接获取：this https URL。 

---
# Random Initialization Can't Catch Up: The Advantage of Language Model Transfer for Time Series Forecasting 

**Title (ZH)**: 随机初始化无法追赶：语言模型迁移在时间序列预测中的优势 

**Authors**: Roland Riachi, Kashif Rasul, Arjun Ashok, Prateek Humane, Alexis Roger, Andrew R. Williams, Yuriy Nevmyvaka, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2506.21570)  

**Abstract**: Recent works have demonstrated the effectiveness of adapting pre-trained language models (LMs) for forecasting time series in the low-data regime. We build upon these findings by analyzing the effective transfer from language models to time series forecasting under various design choices including upstream post-training, time series tokenizer and language backbone size. In the low-data regime, these design choices have a significant impact on the validation loss, with clear-cut choices that outperform others. Contrary to Hernandez et al. (2021), we observe that the validation loss of the LMs continues to smoothly decrease long after the validation loss of the randomly initialized models has converged, leading to a non-vanishing transfer gap that holds across design choices. These findings not only help shed light on the effective use of compute-efficient training for time series, but also open the way for the study of modality-agnostic properties of data distributions leveraged by these models. 

**Abstract (ZH)**: 近期研究表明，预训练语言模型在低数据量条件下用于时间序列预测具有有效性。我们在各种设计选择包括上游微调、时间序列分词器和语言骨干网络大小的基础上分析了其有效的迁移。在低数据量条件下，这些设计选择对验证损失有着显著影响，存在明显的最佳选择。与Hernandez等人的研究不同，我们观察到预训练语言模型的验证损失在随机初始化模型的验证损失收敛后仍能平稳下降，导致一种跨设计选择均存在的非消失的迁移差距。这些发现不仅有助于揭示计算高效培训在时间序列上的有效使用方式，还为研究这些模型利用的数据分布的模态无关性质提供了新的途径。 

---
# Hybrid-NL2SVA: Integrating RAG and Finetuning for LLM-based NL2SVA 

**Title (ZH)**: Hybrid-NL2SVA: 结合RAG和微调的LLM基于自然语言到结构化查询转换 

**Authors**: Weihua Xiao, Derek Ekberg, Siddharth Garg, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2506.21569)  

**Abstract**: SystemVerilog Assertions (SVAs) are critical for verifying the correctness of hardware designs, but manually writing them from natural language property descriptions, i.e., NL2SVA, remains a labor-intensive and error-prone task. Recent advances in large language models (LLMs) offer opportunities to automate this translation. However, existing models still struggle with understanding domain-specific syntax and semantics. To enhance LLM performance in NL2SVA, we propose a customized retrieval-augmented generation (RAG) framework and a synthetic fine-tuning dataset that together improve LLM's performance. To further improve lightweight models over NL2SVA, our fine-tuning dataset provides prompt-guided explanations that teach LLMs the layer-by-layer construction process of concurrent SVAs, enabling supervised fine-tuning that greatly improves syntax and functionality accuracy. To evaluate the performance of LLMs over NL2SVA, we construct the largest evaluation dataset for NL2SVA, comprising 40 Verilog designs and 229 formally verified SVAs with detailed annotations. Experimental results show that our customized RAG framework increases the number of functionality matched SVAs by 58.42% over GPT-4o-mini, while Qwen2.5-Coder-7B-Instruct fine-tuned on our fine-tuning dataset and integrated with HybridRetrieval achieves a 59.05% over the base Qwen model. 

**Abstract (ZH)**: SystemVerilog断言（SVAs）对于硬件设计的正确性验证至关重要，但从自然语言属性描述手动编写它们，即NL2SVA，仍然是一个劳动密集型且容易出错的任务。近年来，大型语言模型（LLMs）的进步为自动化这一转换提供了机会。然而，现有模型仍然难以理解特定领域的语法和语义。为了提高LLMs在NL2SVA中的性能，我们提出了一种定制的检索增强生成（RAG）框架和一个合成微调数据集，两者共同提升了LLM的性能。为了进一步提高轻量级模型在NL2SVA中的性能，我们的微调数据集提供了提示引导的解释，教会LLMs并发SVAs的逐层构建过程，使监督微调能够极大地提高语法和功能准确性。为了评估LLMs在NL2SVA中的性能，我们构建了最大的NL2SVA评估数据集，包含40个Verilog设计和229个正式验证的SVAs及其详细的标注。实验结果表明，我们定制的RAG框架在功能匹配的SVAs数量上比GPT-4o-mini增加了58.42%，而Qwen2.5-Coder-7B-Instruct在我们微调数据集上微调并与HybridRetrieval集成后，相对于基线Qwen模型的表现提高了59.05%。 

---
# BioPars: A Pretrained Biomedical Large Language Model for Persian Biomedical Text Mining 

**Title (ZH)**: BioPars: 一种预训练生物医学大型语言模型用于波斯语生物医学文本挖掘 

**Authors**: Baqer M. Merzah, Tania Taami, Salman Asoudeh, Amir reza Hossein pour, Saeed Mirzaee, Amir Ali Bengari  

**Link**: [PDF](https://arxiv.org/pdf/2506.21567)  

**Abstract**: Large Language Models (LLMs) have recently gained attention in the life sciences due to their capacity to model, extract, and apply complex biological information. Beyond their classical use as chatbots, these systems are increasingly used for complex analysis and problem-solving in specialized fields, including bioinformatics. First, we introduce BIOPARS-BENCH, a dataset from over 10,000 scientific articles, textbooks, and medical websites. BioParsQA was also introduced to evaluate the proposed model, which consists of 5,231 Persian medical questions and answers. This study then introduces BioPars, a simple but accurate measure designed to assess LLMs for three main abilities: acquiring subject-specific knowledge, interpreting and synthesizing such knowledge, and demonstrating proper evidence. Comparing ChatGPT, Llama, and Galactica, our study highlights their ability to remember and retrieve learned knowledge but also reveals shortcomings in addressing higher-level, real-world questions and fine-grained inferences. These findings indicate the need for further fine-tuning to address the capabilities of LLM in bioinformatics tasks. To our knowledge, BioPars is the first application of LLM in Persian medical QA, especially for generating long answers. Evaluation of four selected medical QA datasets shows that BioPars has achieved remarkable results compared to comparative approaches. The model on BioParsQA achieved a ROUGE-L score of 29.99, which is an improvement over GPT-4 1.0. The model achieved a BERTScore of 90.87 with the MMR method. The MoverScore and BLEURT values were also higher in this model than the other three models. In addition, the reported scores for the model are MoverScore=60.43 and BLEURT=50.78. BioPars is an ongoing project and all resources related to its development will be made available via the following GitHub repository: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生命科学研究中的应用由于其建模、提取和应用复杂生物信息的能力而引起了关注。除了作为聊天机器人的传统用途外，这些系统还在专业领域如生物信息学中用于复杂分析和问题解决。本研究介绍了BIOPARS-BENCH数据集，包含超过10,000篇科学文章、教科书和医学网站的内容。我们还引入了BioParsQA用于评估模型，该数据集包含5,231个波斯语医学问题及其答案。随后，本研究介绍了BioPars，这是一种简单但准确的指标，旨在评估LLM在获取特定领域知识、解释和综合此类知识以及提供适当证据方面的三种主要能力。通过比较ChatGPT、Llama和Galactica，本研究突显了它们记忆和检索学习知识的能力，但也揭示了它们在回答高层次的现实世界问题和精细推理方面存在的不足。这些发现表明，为了应对生物信息学任务的需求，需要进一步微调LLM的能力。据我们所知，BioPars是第一个在波斯语医学问答中应用LLM，尤其是生成长篇回答的应用。本研究评估了四个选定的医学问答数据集，显示BioPars在与比较方法相比时取得了令人瞩目的成果。在BioParsQA上的模型实现了ROUGE-L分数为29.99，优于GPT-4 1.0。使用MMR方法，模型的BERTScore为90.87。该模型的MoverScore和BLEURT值也高于其他三种模型。此外，模型的得分报告为MoverScore=60.43，BLEURT=50.78。BioPars是一个正在进行的项目，其所有相关资源将通过以下GitHub仓库提供：this https URL。 

---
# The Saturation Point of Backtranslation in High Quality Low Resource English Gujarati Machine Translation 

**Title (ZH)**: 高质量低资源英吉利瓦利机器翻译中的回译饱和点 

**Authors**: Arwa Arif  

**Link**: [PDF](https://arxiv.org/pdf/2506.21566)  

**Abstract**: Backtranslation BT is widely used in low resource machine translation MT to generate additional synthetic training data using monolingual corpora. While this approach has shown strong improvements for many language pairs, its effectiveness in high quality, low resource settings remains unclear. In this work, we explore the effectiveness of backtranslation for English Gujarati translation using the multilingual pretrained MBART50 model. Our baseline system, trained on a high quality parallel corpus of approximately 50,000 sentence pairs, achieves a BLEU score of 43.8 on a validation set. We augment this data with carefully filtered backtranslated examples generated from monolingual Gujarati text. Surprisingly, adding this synthetic data does not improve translation performance and, in some cases, slightly reduces it. We evaluate our models using multiple metrics like BLEU, ChrF++, TER, BLEURT and analyze possible reasons for this saturation. Our findings suggest that backtranslation may reach a point of diminishing returns in certain low-resource settings and we discuss implications for future research. 

**Abstract (ZH)**: 基于回译的机器翻译在英孟инд্র语低资源设置中的有效性探究 

---
# Team QUST at SemEval-2025 Task 10: Evaluating Large Language Models in Multiclass Multi-label Classification of News Entity Framing 

**Title (ZH)**: QUST团队参加SemEval-2025 Task 10：新闻实体框架多类别多标签分类中大型语言模型的评估 

**Authors**: Jiyan Liu, Youzheng Liu, Taihang Wang, Xiaoman Xu, Yimin Wang, Ye Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21564)  

**Abstract**: This paper describes the participation of QUST_NLP in the SemEval-2025 Task 7. We propose a three-stage retrieval framework specifically designed for fact-checked claim retrieval. Initially, we evaluate the performance of several retrieval models and select the one that yields the best results for candidate retrieval. Next, we employ multiple re-ranking models to enhance the candidate results, with each model selecting the Top-10 outcomes. In the final stage, we utilize weighted voting to determine the final retrieval outcomes. Our approach achieved 5th place in the monolingual track and 7th place in the crosslingual track. We release our system code at: this https URL. 

**Abstract (ZH)**: 本研究介绍了QUST_NLP在SemEval-2025 Task 7中的参与情况。我们提出了一种专门设计用于事实核查声明检索的三阶段检索框架。首先，我们评估了几种检索模型的性能，并选择了候选检索结果表现最佳的模型。随后，我们运用了多个重排序模型来提高候选结果的质量，每种模型选出Top-10结果。在最终阶段，我们使用加权投票来确定最终的检索结果。我们的方法在单语轨道中获得第5名，在跨语言轨道中获得第7名。我们已将系统代码发布在以下链接：this https URL。 

---
# FloorPlan-DeepSeek (FPDS): A multimodal approach to floorplan generation using vector-based next room prediction 

**Title (ZH)**: FloorPlan-DeepSeek (FPDS): 基于向量的下一步房间预测的多模态楼面图生成方法 

**Authors**: Jun Yin, Pengyu Zeng, Jing Zhong, Peilin Li, Miao Zhang, Ran Luo, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21562)  

**Abstract**: In the architectural design process, floor plan generation is inherently progressive and iterative. However, existing generative models for floor plans are predominantly end-to-end generation that produce an entire pixel-based layout in a single pass. This paradigm is often incompatible with the incremental workflows observed in real-world architectural practice. To address this issue, we draw inspiration from the autoregressive 'next token prediction' mechanism commonly used in large language models, and propose a novel 'next room prediction' paradigm tailored to architectural floor plan modeling. Experimental evaluation indicates that FPDS demonstrates competitive performance in comparison to diffusion models and Tell2Design in the text-to-floorplan task, indicating its potential applicability in supporting future intelligent architectural design. 

**Abstract (ZH)**: 在建筑设计过程中，楼层平面图生成天然地具有渐进性和迭代性。然而，现有的平面图生成模型主要是一次完成整个像素布局的端到端生成。这种范式往往与现实世界建筑实践中观察到的增量工作流不兼容。为此，我们受到大型语言模型中常用的自回归“下一个词预测”机制的启发，提出了一种适用于建筑平面图建模的新型“下一个房间预测”范式。实验评价表明，FPDS 在文本到平面图任务中表现出与扩散模型和Tell2Design相当的竞争性能，表明其在未来智能建筑设计中的潜在应用价值。 

---
# Reasoning Isn't Enough: Examining Truth-Bias and Sycophancy in LLMs 

**Title (ZH)**: 理性不足：探究LLM中的真相偏差与阿谀倾向 

**Authors**: Emilio Barkett, Olivia Long, Madhavendra Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2506.21561)  

**Abstract**: Despite their widespread use in fact-checking, moderation, and high-stakes decision-making, large language models (LLMs) remain poorly understood as judges of truth. This study presents the largest evaluation to date of LLMs' veracity detection capabilities and the first analysis of these capabilities in reasoning models. We had eight LLMs make 4,800 veracity judgments across several prompts, comparing reasoning and non-reasoning models. We find that rates of truth-bias, or the likelihood to believe a statement is true, regardless of whether it is actually true, are lower in reasoning models than in non-reasoning models, but still higher than human benchmarks. Most concerning, we identify sycophantic tendencies in several advanced models (o4-mini and GPT-4.1 from OpenAI, R1 from DeepSeek), which displayed an asymmetry in detection accuracy, performing well in truth accuracy but poorly in deception accuracy. This suggests that capability advances alone do not resolve fundamental veracity detection challenges in LLMs. 

**Abstract (ZH)**: 尽管大型语言模型在事实核查、内容审核和高风险决策中广泛应用，但它们作为真理裁判者的作用仍不甚明了。本研究展示了迄今为止最大的大型语言模型真实性检测能力评估，并首次分析了这些能力在推理模型中的表现。我们让八种大型语言模型针对多个提示做出了4800次真实性判断，比较了推理模型和非推理模型的性能。结果发现，推理模型在真实性判断中的真理偏差率（即不加判断地认为陈述为真的可能性）低于非推理模型，但仍高于人类基准。更令人担忧的是，我们发现在几个先进模型中存在讨好倾向（来自OpenAI的o4-mini和GPT-4.1，以及来自DeepSeek的R1），这些模型在真实性的检测准确率上表现良好，但在欺骗性的检测准确率上表现较差。这表明，能力提升本身并不能解决大型语言模型在真实性检测方面的根本挑战。 

---
# Reinforcement Learning Fine-Tuning of Language Model for Instruction Following and Math Reasoning 

**Title (ZH)**: 语言模型的强化学习微调以执行指令和数学推理 

**Authors**: Yifu Han, Geo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21560)  

**Abstract**: This study investigates the effectiveness of reinforcement learning (RL) fine-tuning techniques on a compact language model (Qwen2.5-0.5B Base) for two challenging tasks: instruction following and mathematical reasoning. We compare supervised fine-tuning (SFT), Direct Preference Optimization (DPO) using preference-labeled data, and Reinforce Leave-One-Out (RLOO) with reward models. Our experiments show that RLOO with DeBERTa reward modeling achieves the best alignment, while DPO provides strong and consistent results. For math reasoing tasks, synthetic data augmentation and best-of-N sampling with an external verifier significantly improve accuracy, showing the potential of combining fine-tuning with inference-time tools. This study highlights key trade-offs and practical strategies for training lightweight, task-aligned small-scale language models. 

**Abstract (ZH)**: 本研究探讨了强化学习（RL）微调技术在紧凑型语言模型（Qwen2.5-0.5B Base）上对两项挑战任务（指令跟随和数学推理）的有效性。我们比较了监督微调（SFT）、使用偏好标签数据的直接偏好优化（DPO）以及奖励模型下的Reinforce Leave-One-Out（RLOO）。实验结果显示，使用DeBERTa奖励模型的RLOO方法在对齐效果上最佳，而DPO方法提供了稳定而强劲的结果。对于数学推理任务，合成数据增强和外部验证器的Best-of-N采样显著提高了准确性，展示了将微调与推理时工具结合的潜力。本研究突出了训练轻量级、任务对齐的小型语言模型的关键权衡和实用策略。 

---
# Bench to the Future: A Pastcasting Benchmark for Forecasting Agents 

**Title (ZH)**: 从今到昔：一个用于预测代理的Pastcasting基准测试 

**Authors**: FutureSearch, Jack Wildman, Nikos I. Bosse, Daniel Hnyk, Peter Mühlbacher, Finn Hambly, Jon Evans, Dan Schwarz, Lawrence Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2506.21558)  

**Abstract**: Forecasting is a challenging task that offers a clearly measurable way to study AI systems. Forecasting requires a large amount of research on the internet, and evaluations require time for events to happen, making the development of forecasting benchmarks challenging. To date, no forecasting benchmark provides a realistic, hermetic, and repeatable environment for LLM forecasters. We introduce Bench To the Future (BTF), a "pastcasting" benchmark with hundreds of high-quality questions for which the resolution is already known. Each question is accompanied by a large offline corpus of tens of thousands of relevant web pages, enabling a way to elicit realistic "forecasts" on past events from LLMs. Results suggest that our pastcasting environment can produce results comparable to those based on forecasts using the internet on at-the-time unresolved questions. We show results benchmarking agent and chain-of-thought forecasting approaches using several LLMs, including the recently-released Claude 4 models, and demonstrate BTF's ability to track steady forecasting capability progress over time. We intend this to be a living benchmark, with new questions added continually to account for increasing training data cutoff dates. We invite researchers to contact us at hello@futuresearch.ai to utilize our benchmark or tooling for their own research. 

**Abstract (ZH)**: Bench To the Future: A "Pastcasting" Benchmark for Assessing LLM Forecasting Capabilities 

---
# Data Efficacy for Language Model Training 

**Title (ZH)**: 语言模型训练的数据有效性 

**Authors**: Yalun Dai, Yangyu Huang, Xin Zhang, Wenshan Wu, Chong Li, Wenhui Lu, Shijie Cao, Li Dong, Scarlett Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21545)  

**Abstract**: Data is fundamental to the training of language models (LM). Recent research has been dedicated to data efficiency, which aims to maximize performance by selecting a minimal or optimal subset of training data. Techniques such as data filtering, sampling, and selection play a crucial role in this area. To complement it, we define Data Efficacy, which focuses on maximizing performance by optimizing the organization of training data and remains relatively underexplored. This work introduces a general paradigm, DELT, for considering data efficacy in LM training, which highlights the significance of training data organization. DELT comprises three components: Data Scoring, Data Selection, and Data Ordering. Among these components, we design Learnability-Quality Scoring (LQS), as a new instance of Data Scoring, which considers both the learnability and quality of each data sample from the gradient consistency perspective. We also devise Folding Ordering (FO), as a novel instance of Data Ordering, which addresses issues such as model forgetting and data distribution bias. Comprehensive experiments validate the data efficacy in LM training, which demonstrates the following: Firstly, various instances of the proposed DELT enhance LM performance to varying degrees without increasing the data scale and model size. Secondly, among these instances, the combination of our proposed LQS for data scoring and Folding for data ordering achieves the most significant improvement. Lastly, data efficacy can be achieved together with data efficiency by applying data selection. Therefore, we believe that data efficacy is a promising foundational area in LM training. 

**Abstract (ZH)**: 数据是语言模型训练的基础。最近的研究致力于提高数据效率，通过选择最小或最优的数据子集来最大化性能。数据过滤、采样和选择等技术在这一领域发挥着关键作用。为补充这一点，我们定义了数据效能，它旨在通过优化训练数据的组织来最大化性能，并且相对未被充分探索。本文提出了一种通用框架DELT，用于在语言模型训练中考虑数据效能，强调了训练数据组织的重要性。DELT包含三个组件：数据评分、数据选择和数据排序。在这些组件中，我们设计了可学习性-质量评分（LQS），这是一种新的数据评分实例，从梯度一致性视角考虑每个数据样本的可学习性和质量。我们还设计了折叠排序（FO），这是一种新的数据排序实例，解决了模型遗忘和数据分布偏差等问题。全面的实验验证了在语言模型训练中实现数据效能，表明了以下几点：首先，所提出的DELT的各种实例在不增加数据量和模型规模的情况下，不同程度地提升了语言模型的性能。其次，在这些实例中，我们的LQS数据评分和Folding数据排序的结合实现了最大的改进。最后，通过数据选择，可以同时实现数据效能和数据效率。因此，我们认为数据效能是语言模型训练中一个前景广阔的基础领域。 

---
# On the Necessity of Output Distribution Reweighting for Effective Class Unlearning 

**Title (ZH)**: 关于输出分布重加权在有效类遗忘中的必要性 

**Authors**: Yian Wang, Ali Ebrahimpour-Boroojeny, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2506.20893)  

**Abstract**: In this work, we introduce an output-reweighting unlearning method, RWFT, a lightweight technique that erases an entire class from a trained classifier without full retraining. Forgetting specific classes from trained models is essential for enforcing user deletion rights and mitigating harmful or biased predictions. The full retraining is costly and existing unlearning methods fail to replicate the behavior of the retrained models when predicting samples from the unlearned class. We prove this failure by designing a variant of membership inference attacks, MIA-NN that successfully reveals the unlearned class for any of these methods. We propose a simple redistribution of the probability mass for the prediction on the samples in the forgotten class which is robust to MIA-NN. We also introduce a new metric based on the total variation (TV) distance of the prediction probabilities to quantify residual leakage to prevent future methods from susceptibility to the new attack. Through extensive experiments with state of the art baselines in machine unlearning, we show that our approach matches the results of full retraining in both metrics used for evaluation by prior work and the new metric we propose in this work. Compare to state-of-the-art methods, we gain 2.79% in previously used metrics and 111.45% in our new TV-based metric over the best existing method. 

**Abstract (ZH)**: 在这种工作中，我们引入了一种输出重权重遗忘方法RWFT，这是一种轻量级技术，可以在不进行完全重新训练的情况下从已训练分类器中删除整个类别。从训练模型中遗忘特定类别对于确保用户的删除权利和减轻有害或偏颇的预测至关重要。完全重新训练成本高，现有遗忘方法无法在预测未遗忘类别样本时复制重新训练模型的行为。我们通过设计一种MIA-NN变体的成员资格推理攻击方式证明了这种失败，该攻击方式能够成功揭示这些方法中的任一类未遗忘类别。我们提出了一种简单的概率质量再分配方法，以在遗忘类别样本的预测中具有针对MIA-NN的鲁棒性。我们还引入了一个基于预测概率的总变差（TV）距离的新度量，以量化剩余泄露，防止未来方法对新攻击的易感性。通过在机器遗忘的先进基线方法上进行广泛实验，我们展示了我们的方法在用于评估的先前工作所使用的两个指标中与完全重新训练的结果相当，并且在本研究中提出的新基于TV的度量上超过当前最佳方法111.45%。相比最先进的方法，我们在先前使用的两个度量中分别提高了2.79%和111.45%。 

---
# PEACE: Empowering Geologic Map Holistic Understanding with MLLMs 

**Title (ZH)**: PEACE: 通过MLLMs增强地质图整体理解 

**Authors**: Yangyu Huang, Tianyi Gao, Haoran Xu, Qihao Zhao, Yang Song, Zhipeng Gui, Tengchao Lv, Hao Chen, Lei Cui, Scarlett Li, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.06184)  

**Abstract**: Geologic map, as a fundamental diagram in geology science, provides critical insights into the structure and composition of Earth's subsurface and surface. These maps are indispensable in various fields, including disaster detection, resource exploration, and civil engineering. Despite their significance, current Multimodal Large Language Models (MLLMs) often fall short in geologic map understanding. This gap is primarily due to the challenging nature of cartographic generalization, which involves handling high-resolution map, managing multiple associated components, and requiring domain-specific knowledge. To quantify this gap, we construct GeoMap-Bench, the first-ever benchmark for evaluating MLLMs in geologic map understanding, which assesses the full-scale abilities in extracting, referring, grounding, reasoning, and analyzing. To bridge this gap, we introduce GeoMap-Agent, the inaugural agent designed for geologic map understanding, which features three modules: Hierarchical Information Extraction (HIE), Domain Knowledge Injection (DKI), and Prompt-enhanced Question Answering (PEQA). Inspired by the interdisciplinary collaboration among human scientists, an AI expert group acts as consultants, utilizing a diverse tool pool to comprehensively analyze questions. Through comprehensive experiments, GeoMap-Agent achieves an overall score of 0.811 on GeoMap-Bench, significantly outperforming 0.369 of GPT-4o. Our work, emPowering gEologic mAp holistiC undErstanding (PEACE) with MLLMs, paves the way for advanced AI applications in geology, enhancing the efficiency and accuracy of geological investigations. 

**Abstract (ZH)**: 基于地质图的全方位理解：利用多模态大规模语言模型（PEACE） 

---
# MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark 

**Title (ZH)**: MMLU-CF：一种无污染的多任务语言理解基准测试 

**Authors**: Qihao Zhao, Yangyu Huang, Tengchao Lv, Lei Cui, Qinzheng Sun, Shaoguang Mao, Xin Zhang, Ying Xin, Qiufeng Yin, Scarlett Li, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2412.15194)  

**Abstract**: Multiple-choice question (MCQ) datasets like Massive Multitask Language Understanding (MMLU) are widely used to evaluate the commonsense, understanding, and problem-solving abilities of large language models (LLMs). However, the open-source nature of these benchmarks and the broad sources of training data for LLMs have inevitably led to benchmark contamination, resulting in unreliable evaluation results. To alleviate this issue, we propose a contamination-free and more challenging MCQ benchmark called MMLU-CF. This benchmark reassesses LLMs' understanding of world knowledge by averting both unintentional and malicious data leakage. To avoid unintentional data leakage, we source data from a broader domain and design three decontamination rules. To prevent malicious data leakage, we divide the benchmark into validation and test sets with similar difficulty and subject distributions. The test set remains closed-source to ensure reliable results, while the validation set is publicly available to promote transparency and facilitate independent verification. Our evaluation of mainstream LLMs reveals that the powerful GPT-4o achieves merely a 5-shot score of 73.4% and a 0-shot score of 71.9% on the test set, which indicates the effectiveness of our approach in creating a more rigorous and contamination-free evaluation standard. The GitHub repository is available at this https URL and the dataset refers to this https URL. 

**Abstract (ZH)**: 无污染更具挑战性的多项选择题基准MMLU-CF 

---
# FreeEnricher: Enriching Face Landmarks without Additional Cost 

**Title (ZH)**: FreeEnricher: 不额外增加成本的 facial landmarks 增强方法 

**Authors**: Yangyu Huang, Xi Chen, Jongyoo Kim, Hao Yang, Chong Li, Jiaolong Yang, Dong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2212.09525)  

**Abstract**: Recent years have witnessed significant growth of face alignment. Though dense facial landmark is highly demanded in various scenarios, e.g., cosmetic medicine and facial beautification, most works only consider sparse face alignment. To address this problem, we present a framework that can enrich landmark density by existing sparse landmark datasets, e.g., 300W with 68 points and WFLW with 98 points. Firstly, we observe that the local patches along each semantic contour are highly similar in appearance. Then, we propose a weakly-supervised idea of learning the refinement ability on original sparse landmarks and adapting this ability to enriched dense landmarks. Meanwhile, several operators are devised and organized together to implement the idea. Finally, the trained model is applied as a plug-and-play module to the existing face alignment networks. To evaluate our method, we manually label the dense landmarks on 300W testset. Our method yields state-of-the-art accuracy not only in newly-constructed dense 300W testset but also in the original sparse 300W and WFLW testsets without additional cost. 

**Abstract (ZH)**: 最近几年，面部对齐领域取得了显著的增长。尽管密集面部 landmarks 在各种场景中高度需求，例如美容医学和面部美化，大多数工作仅考虑稀疏面部对齐。为了解决这一问题，我们提出了一种框架，可以通过现有的稀疏 landmarks 数据集（例如包含 68 个点的 300W 和包含 98 个点的 WFLW）来丰富 landmarks 密度。首先，我们观察到每条语义轮廓沿线的局部补丁在Appearance上具有高度相似性。然后，我们提出了一种弱监督方法来学习在原始稀疏 landmarks 上的细化能力，并适应这种能力以生产密集 landmarks。同时，设计并组织了一系列操作来实现这一理念。最后，训练好的模型被应用为即插即用模块到现有的面部对齐网络中。为了评估我们的方法，我们在 300W 测试集上手动标注了密集 landmarks。我们的方法不仅在新构建的密集 300W 测试集上达到了最先进的准确性，在原始稀疏的 300W 和 WFLW 测试集上也取得了该精度，而无需额外成本。 

---
# ADNet: Leveraging Error-Bias Towards Normal Direction in Face Alignment 

**Title (ZH)**: ADNet: 利用误差偏差朝向正常方向进行面部对齐 

**Authors**: Yangyu Huang, Hao Yang, Chong Li, Jongyoo Kim, Fangyun Wei  

**Link**: [PDF](https://arxiv.org/pdf/2109.05721)  

**Abstract**: The recent progress of CNN has dramatically improved face alignment performance. However, few works have paid attention to the error-bias with respect to error distribution of facial landmarks. In this paper, we investigate the error-bias issue in face alignment, where the distributions of landmark errors tend to spread along the tangent line to landmark curves. This error-bias is not trivial since it is closely connected to the ambiguous landmark labeling task. Inspired by this observation, we seek a way to leverage the error-bias property for better convergence of CNN model. To this end, we propose anisotropic direction loss (ADL) and anisotropic attention module (AAM) for coordinate and heatmap regression, respectively. ADL imposes strong binding force in normal direction for each landmark point on facial boundaries. On the other hand, AAM is an attention module which can get anisotropic attention mask focusing on the region of point and its local edge connected by adjacent points, it has a stronger response in tangent than in normal, which means relaxed constraints in the tangent. These two methods work in a complementary manner to learn both facial structures and texture details. Finally, we integrate them into an optimized end-to-end training pipeline named ADNet. Our ADNet achieves state-of-the-art results on 300W, WFLW and COFW datasets, which demonstrates the effectiveness and robustness. 

**Abstract (ZH)**: recent progress of CNN在面部对齐性能上的 recent 进展显著提高。然而，很少有工作关注与面部关键点误差分布相关的误差偏差问题。在本文中，我们研究了面部对齐中的误差偏差问题，发现关键点误差的分布趋势沿着关键点曲线的切线方向扩展。这种误差偏差并不简单，因为它与模棱两可的关键点标签任务密切相关。受此观察的启发，我们寻求一种利用误差偏差特性以提高CNN模型收敛性的方法。为此，我们提出了各向异性方向损失（ADL）和各向异性注意力模块（AAM），分别应用于坐标和热图回归。ADL 对面部边界上的每个关键点施加强的法线方向约束力。另一方面，AAM 是一种注意力模块，能够在关键点及其相邻点形成的局部边缘连接的区域中获得各向异性注意力掩码，并在切线方向上具有更强的响应，这意味着在切线方向上的松弛约束条件。这两种方法相互补充，以学习面部结构和纹理细节。最后，我们将它们整合到一个优化的端到端训练管道ADNet 中。我们的ADNet 在300W、WFLW 和COFW 数据集上取得了最先进的结果，这证明了其有效性和鲁棒性。 

---
