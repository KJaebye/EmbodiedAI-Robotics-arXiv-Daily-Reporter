# Ax-Prover: A Deep Reasoning Agentic Framework for Theorem Proving in Mathematics and Quantum Physics 

**Title (ZH)**: Ax-Prover: 一个用于数学和量子物理定理证明的深度推理代理框架 

**Authors**: Marco Del Tredici, Jacob McCarran, Benjamin Breen, Javier Aspuru Mijares, Weichen Winston Yin, Jacob M. Taylor, Frank Koppens, Dirk Englund  

**Link**: [PDF](https://arxiv.org/pdf/2510.12787)  

**Abstract**: We present Ax-Prover, a multi-agent system for automated theorem proving in Lean that can solve problems across diverse scientific domains and operate either autonomously or collaboratively with human experts. To achieve this, Ax-Prover approaches scientific problem solving through formal proof generation, a process that demands both creative reasoning and strict syntactic rigor. Ax-Prover meets this challenge by equipping Large Language Models (LLMs), which provide knowledge and reasoning, with Lean tools via the Model Context Protocol (MCP), which ensure formal correctness. To evaluate its performance as an autonomous prover, we benchmark our approach against frontier LLMs and specialized prover models on two public math benchmarks and on two Lean benchmarks we introduce in the fields of abstract algebra and quantum theory. On public datasets, Ax-Prover is competitive with state-of-the-art provers, while it largely outperform them on the new benchmarks. This shows that, unlike specialized systems that struggle to generalize, our tool-based agentic theorem prover approach offers a generalizable methodology for formal verification across diverse scientific domains. Furthermore, we demonstrate Ax-Prover's assistant capabilities in a practical use case, showing how it enabled an expert mathematician to formalize the proof of a complex cryptography theorem. 

**Abstract (ZH)**: 我们介绍Ax-Prover：一个通过Lean进行自动定理证明的多代理系统，能够解决跨学科科学领域的难题，并可自主运行或与人类专家协作。Ax-Prover通过形式证明这一过程，既要求创造性推理也要求严格的语法规则，利用大型语言模型（LLMs）结合Lean工具来确保形式正确性。为了评估其作为自主证明系统的性能，我们将其方法与前沿的LLMs和专门的证明模型在两个公开的数学基准测试和两个由我们引入的抽象代数和量子理论领域的Lean基准测试上进行对比。在公开数据集上，Ax-Prover与最先进的证明系统竞争；在新基准测试上，其性能显著优于它们。这表明，与难以泛化的专业系统不同，我们基于工具的代理型定理证明方法提供了在跨学科科学领域中进行形式验证的一般化方法。此外，我们通过一个实际应用场景展示了Ax-Prover的辅助能力，展示了它如何使一位专家数学家能够形式化一个复杂的密码学定理的证明。 

---
# CTRL-Rec: Controlling Recommender Systems With Natural Language 

**Title (ZH)**: CTRL-Rec: 用自然语言控制推荐系统 

**Authors**: Micah Carroll, Adeline Foote, Kevin Feng, Marcus Williams, Anca Dragan, W. Bradley Knox, Smitha Milli  

**Link**: [PDF](https://arxiv.org/pdf/2510.12742)  

**Abstract**: When users are dissatisfied with recommendations from a recommender system, they often lack fine-grained controls for changing them. Large language models (LLMs) offer a solution by allowing users to guide their recommendations through natural language requests (e.g., "I want to see respectful posts with a different perspective than mine"). We propose a method, CTRL-Rec, that allows for natural language control of traditional recommender systems in real-time with computational efficiency. Specifically, at training time, we use an LLM to simulate whether users would approve of items based on their language requests, and we train embedding models that approximate such simulated judgments. We then integrate these user-request-based predictions into the standard weighting of signals that traditional recommender systems optimize. At deployment time, we require only a single LLM embedding computation per user request, allowing for real-time control of recommendations. In experiments with the MovieLens dataset, our method consistently allows for fine-grained control across a diversity of requests. In a study with 19 Letterboxd users, we find that CTRL-Rec was positively received by users and significantly enhanced users' sense of control and satisfaction with recommendations compared to traditional controls. 

**Abstract (ZH)**: 当用户对推荐系统推荐的内容不满意时，他们通常缺乏精细的控制手段来调整推荐。大型语言模型（LLMs）通过允许用户通过自然语言请求（如：“我想要看到不同于我自己的观点的有尊重性的帖子”）来引导推荐提供了解决方案。我们提出了一种方法——CTRL-Rec，该方法可以在保证计算效率的前提下，实现实时的基于自然语言的推荐控制。具体而言，在训练期间，我们使用LLM模拟用户是否会批准基于其语言请求的项目，并训练嵌入模型以近似这些模拟判断。然后，我们将这些基于用户请求的预测整合到传统推荐系统优化的标准信号权重中。在部署时，我们只需要为每个用户请求进行一次LLM嵌入计算，从而实现推荐的实时控制。在使用MovieLens数据集的实验中，我们的方法能够跨多种请求实现细粒度控制。在一项涉及19名Letterboxd用户的的研究中，我们发现CTRL-Rec受到了用户的积极评价，并且与传统控制方法相比，显著提升了用户对推荐的控制感和满意度。 

---
# Clutch Control: An Attention-based Combinatorial Bandit for Efficient Mutation in JavaScript Engine Fuzzing 

**Title (ZH)**: 刹车控制：一种基于注意力的组合多臂bandit算法，用于JavaScript引擎 fuzzing 中的有效变异体生成 

**Authors**: Myles Foley, Sergio Maffeis, Muhammad Fakhrur Rozi, Takeshi Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12732)  

**Abstract**: JavaScript engines are widely used in web browsers, PDF readers, and server-side applications. The rise in concern over their security has led to the development of several targeted fuzzing techniques. However, existing approaches use random selection to determine where to perform mutations in JavaScript code. We postulate that the problem of selecting better mutation targets is suitable for combinatorial bandits with a volatile number of arms. Thus, we propose CLUTCH, a novel deep combinatorial bandit that can observe variable length JavaScript test case representations, using an attention mechanism from deep learning. Furthermore, using Concrete Dropout, CLUTCH can dynamically adapt its exploration. We show that CLUTCH increases efficiency in JavaScript fuzzing compared to three state-of-the-art solutions by increasing the number of valid test cases and coverage-per-testcase by, respectively, 20.3% and 8.9% on average. In volatile and combinatorial settings we show that CLUTCH outperforms state-of-the-art bandits, achieving at least 78.1% and 4.1% less regret in volatile and combinatorial settings, respectively. 

**Abstract (ZH)**: JavaScript引擎广泛应用于网页浏览器、PDF阅读器和服务器端应用。随着对它们安全性的关注增加，已经开发出了几种针对性的变异测试技术。然而，现有方法使用随机选择来决定在JavaScript代码中的哪个位置进行变异。我们假设选择更好的变异目标问题适合使用具有可变臂数量的组合臂赛。因此，我们提出了CLUTCH，一种新颖的深度组合臂赛，可以观察JavaScript测试用例的变长表示，并利用深度学习的注意力机制。此外，通过使用Concrete Dropout，CLUTCH可以动态调整其探索。我们展示了与三种最先进的解决方案相比，CLUTCH在JavaScript变异测试中提高了效率，平均每份测试用例增加了20.3%的有效测试用例数量和8.9%的覆盖率。在具有波动性和组合性设置的情况下，我们展示了CLUTCH优于最先进的臂赛，分别在波动性和组合性设置中减少了至少78.1%和4.1%的遗憾值。 

---
# Towards Robust Artificial Intelligence: Self-Supervised Learning Approach for Out-of-Distribution Detection 

**Title (ZH)**: 面向鲁棒的人工智能：自监督学习的离分布检测方法 

**Authors**: Wissam Salhab, Darine Ameyed, Hamid Mcheick, Fehmi Jaafar  

**Link**: [PDF](https://arxiv.org/pdf/2510.12713)  

**Abstract**: Robustness in AI systems refers to their ability to maintain reliable and accurate performance under various conditions, including out-of-distribution (OOD) samples, adversarial attacks, and environmental changes. This is crucial in safety-critical systems, such as autonomous vehicles, transportation, or healthcare, where malfunctions could have severe consequences. This paper proposes an approach to improve OOD detection without the need of labeled data, thereby increasing the AI systems' robustness. The proposed approach leverages the principles of self-supervised learning, allowing the model to learn useful representations from unlabeled data. Combined with graph-theoretical techniques, this enables the more efficient identification and categorization of OOD samples. Compared to existing state-of-the-art methods, this approach achieved an Area Under the Receiver Operating Characteristic Curve (AUROC) = 0.99. 

**Abstract (ZH)**: AI系统中的健 Robustness in AI Systems: Improving Out-of-Distribution Detection Without Labeled Data Through Self-Supervised Learning and Graph-Theoretical Techniques 

---
# CAMNet: Leveraging Cooperative Awareness Messages for Vehicle Trajectory Prediction 

**Title (ZH)**: CAMNet：利用合作感知消息进行车辆轨迹预测 

**Authors**: Mattia Grasselli, Angelo Porrello, Carlo Augusto Grazia  

**Link**: [PDF](https://arxiv.org/pdf/2510.12703)  

**Abstract**: Autonomous driving remains a challenging task, particularly due to safety concerns. Modern vehicles are typically equipped with expensive sensors such as LiDAR, cameras, and radars to reduce the risk of accidents. However, these sensors face inherent limitations: their field of view and line of sight can be obstructed by other vehicles, thereby reducing situational awareness. In this context, vehicle-to-vehicle communication plays a crucial role, as it enables cars to share information and remain aware of each other even when sensors are occluded. One way to achieve this is through the use of Cooperative Awareness Messages (CAMs). In this paper, we investigate the use of CAM data for vehicle trajectory prediction. Specifically, we design and train a neural network, Cooperative Awareness Message-based Graph Neural Network (CAMNet), on a widely used motion forecasting dataset. We then evaluate the model on a second dataset that we created from scratch using Cooperative Awareness Messages, in order to assess whether this type of data can be effectively exploited. Our approach demonstrates promising results, showing that CAMs can indeed support vehicle trajectory prediction. At the same time, we discuss several limitations of the approach, which highlight opportunities for future research. 

**Abstract (ZH)**: 自主驾驶仍是一项具有挑战性的任务，特别是由于安全考虑。现代车辆通常配备昂贵的传感器，如LiDAR、摄像头和雷达，以降低事故发生的风险。然而，这些传感器存在固有的局限性：它们的视野和视线可能会被其他车辆阻挡，从而降低 situational awareness。在这种情况下，车辆间通信起着关键作用，因为它使汽车能够在传感器被遮挡的情况下仍能分享信息并保持相互知觉。一种实现这一点的方法是使用协作感知消息（CAMs）。在本文中，我们研究了CAM数据在车辆轨迹预测中的应用。具体而言，我们设计并训练了一个基于CAM数据的图神经网络（CAMNet），用于一个广泛使用的运动预测数据集。然后，我们在一个新创建的使用合作感知消息的数据集上评估了该模型，以评估这种数据是否能够有效利用。我们的方法显示出有希望的结果，表明CAMs确实支持车辆轨迹预测。同时，我们也讨论了该方法的一些局限性，这些局限性指出了未来研究的机会。 

---
# Multi-Agent Debate for LLM Judges with Adaptive Stability Detection 

**Title (ZH)**: LLM法官参与的自适应稳定性检测多代理辩论 

**Authors**: Tianyu Hu, Zhen Tan, Song Wang, Huaizhi Qu, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12697)  

**Abstract**: With advancements in reasoning capabilities, Large Language Models (LLMs) are increasingly employed for automated judgment tasks. While LLMs-as-Judges offer promise in automating evaluations, current approaches often rely on simplistic aggregation methods (e.g., majority voting), which can fail even when individual agents provide correct answers. To address this, we propose a multi-agent debate judge framework where agents collaboratively reason and iteratively refine their responses. We formalize the debate process mathematically, analyzing agent interactions and proving that debate amplifies correctness compared to static ensembles. To enhance efficiency, we introduce a stability detection mechanism that models judge consensus dynamics via a time-varying Beta-Binomial mixture, with adaptive stopping based on distributional similarity (Kolmogorov-Smirnov test). This mechanism models the judges' collective correct rate dynamics using a time-varying mixture of Beta-Binomial distributions and employs an adaptive stopping criterion based on distributional similarity (Kolmogorov-Smirnov statistic). Experiments across multiple benchmarks and models demonstrate that our framework improves judgment accuracy over majority voting while maintaining computational efficiency. 

**Abstract (ZH)**: 随着推理能力的进步，大规模语言模型（LLMs）越来越多地被用于自动化判断任务。虽然LLMs-as-Judges在自动化评估方面具有潜力，但当前的方法往往依赖于简单的聚合方法（例如多数投票），即使个体代理提供正确的答案也可能失败。为此，我们提出了一种多代理辩论裁决框架，使代理能够协作推理并迭代地完善其回复。我们从数学上形式化了辩论过程，分析了代理间的交互，并证明了辩论比静态ensemble能更提高正确性。为提高效率，我们引入了一种稳定性检测机制，通过时间变化的Beta-Binomial混合模型建模裁决者的共识动态，并基于分布相似性（Kolmogorov-Smirnov检验）实现自适应停止。该机制使用时间变化的Beta-Binomial分布混合模型描述裁决者的集体正确率动态，并基于分布相似性（Kolmogorov-Smirnov统计量）采用自适应停止准则。在多个基准和模型上的实验表明，我们的框架在保持计算效率的同时提高了判断准确性。 

---
# ERA: Transforming VLMs into Embodied Agents via Embodied Prior Learning and Online Reinforcement Learning 

**Title (ZH)**: ERA: 将VLMs转化为具身代理的具身先验学习与在线强化学习方法 

**Authors**: Hanyang Chen, Mark Zhao, Rui Yang, Qinwei Ma, Ke Yang, Jiarui Yao, Kangrui Wang, Hao Bai, Zhenhailong Wang, Rui Pan, Mengchao Zhang, Jose Barreiros, Aykut Onol, ChengXiang Zhai, Heng Ji, Manling Li, Huan Zhang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12693)  

**Abstract**: Recent advances in embodied AI highlight the potential of vision language models (VLMs) as agents capable of perception, reasoning, and interaction in complex environments. However, top-performing systems rely on large-scale models that are costly to deploy, while smaller VLMs lack the necessary knowledge and skills to succeed. To bridge this gap, we present \textit{Embodied Reasoning Agent (ERA)}, a two-stage framework that integrates prior knowledge learning and online reinforcement learning (RL). The first stage, \textit{Embodied Prior Learning}, distills foundational knowledge from three types of data: (1) Trajectory-Augmented Priors, which enrich existing trajectory data with structured reasoning generated by stronger models; (2) Environment-Anchored Priors, which provide in-environment knowledge and grounding supervision; and (3) External Knowledge Priors, which transfer general knowledge from out-of-environment datasets. In the second stage, we develop an online RL pipeline that builds on these priors to further enhance agent performance. To overcome the inherent challenges in agent RL, including long horizons, sparse rewards, and training instability, we introduce three key designs: self-summarization for context management, dense reward shaping, and turn-level policy optimization. Extensive experiments on both high-level planning (EB-ALFRED) and low-level control (EB-Manipulation) tasks demonstrate that ERA-3B surpasses both prompting-based large models and previous training-based baselines. Specifically, it achieves overall improvements of 8.4\% on EB-ALFRED and 19.4\% on EB-Manipulation over GPT-4o, and exhibits strong generalization to unseen tasks. Overall, ERA offers a practical path toward scalable embodied intelligence, providing methodological insights for future embodied AI systems. 

**Abstract (ZH)**: Recent Advances in Embodied AI Highlight the Potential of Vision-Language Models as Agents Capable of Perception, Reasoning, and Interaction in Complex Environments: Bridging the Gap with the Embodied Reasoning Agent (ERA) Framework 

---
# Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks 

**Title (ZH)**: 记忆即行动：自主上下文策展用于长期自主任务 

**Authors**: Yuxiang Zhang, Jiangming Shu, Ye Ma, Xueyuan Lin, Shangxi Wu, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12635)  

**Abstract**: Large Language Models face challenges in long-horizon agentic tasks as their constrained memory is easily overwhelmed by distracting or irrelevant context. Existing working memory methods typically rely on external, heuristic mechanisms that are decoupled from the agent's core policy. In this work, we reframe working memory management as a learnable, intrinsic capability. We propose a novel framework, Memory-as-Action, where an agent actively manages its working memory by executing explicit editing operations as part of a unified policy. This formulation allows an agent, trained via reinforcement learning, to balance memory curation against long-term task objectives under given resource constraints. However, such memory editing actions break the standard assumption of a continuously growing prefix in LLM interactions, leading to what we call trajectory fractures. These non-prefix changes disrupt the causal continuity required by standard policy gradient methods, making those methods inapplicable. To address this, we propose a new algorithm, Dynamic Context Policy Optimization, which enables stable end-to-end reinforcement learning by segmenting trajectories at memory action points and applying trajectory-level advantages to the resulting action segments. Our results demonstrate that jointly optimizing for task reasoning and memory management in an end-to-end fashion not only reduces overall computational consumption but also improves task performance, driven by adaptive context curation strategies tailored to the model's intrinsic capabilities. 

**Abstract (ZH)**: 大型语言模型在长期任务中的记忆管理面临挑战，因其受约束的记忆容易被无关或分散的上下文所淹没。现有的工作记忆方法通常依赖于与代理核心策略脱钩的外部启发式机制。在本工作中，我们将工作记忆管理重新定义为可学习的内在能力。我们提出了一种新的框架——Memory-as-Action，其中代理通过执行明确的编辑操作来作为统一政策的一部分主动管理其工作记忆。这种表述使代理能够通过强化学习训练，在给定资源约束下平衡记忆整理与长期任务目标之间的关系。然而，这类记忆编辑操作破坏了标准的前缀连续假设，导致我们称之为轨迹断裂的现象。这些非前缀改变打断了标准策略梯度方法所需的因果连续性，使这些方法不再适用。为此，我们提出了一种新的算法——动态上下文策略优化，该算法通过在记忆操作点分割轨迹并在最终的行动片段上应用路径级别优势来实现端到端的强化学习的稳定性。我们的结果表明，以端到端的方式联合优化任务推理和记忆管理不仅减少了整体计算消耗，还通过根据模型的内在能力定制的自适应上下文整理策略提高了任务性能。 

---
# HardcoreLogic: Challenging Large Reasoning Models with Long-tail Logic Puzzle Games 

**Title (ZH)**: HardcoreLogic：用长尾逻辑谜题游戏挑战大型推理模型 

**Authors**: Jingcong Liang, Shijun Wan, Xuehai Wu, Siyuan Wang, Yitong Li, Qianglong Chen, Duyu Tang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.12563)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated impressive performance on complex tasks, including logical puzzle games that require deriving solutions satisfying all constraints. However, whether they can flexibly apply appropriate rules to varying conditions, particularly when faced with non-canonical game variants, remains an open question. Existing corpora focus on popular puzzles like 9x9 Sudoku, risking overfitting to canonical formats and memorization of solution patterns, which can mask deficiencies in understanding novel rules or adapting strategies to new variants. To address this, we introduce HardcoreLogic, a challenging benchmark of over 5,000 puzzles across 10 games, designed to test the robustness of LRMs on the "long-tail" of logical games. HardcoreLogic systematically transforms canonical puzzles through three dimensions: Increased Complexity (IC), Uncommon Elements (UE), and Unsolvable Puzzles (UP), reducing reliance on shortcut memorization. Evaluations on a diverse set of LRMs reveal significant performance drops, even for models achieving top scores on existing benchmarks, indicating heavy reliance on memorized stereotypes. While increased complexity is the dominant source of difficulty, models also struggle with subtle rule variations that do not necessarily increase puzzle difficulty. Our systematic error analysis on solvable and unsolvable puzzles further highlights gaps in genuine reasoning. Overall, HardcoreLogic exposes the limitations of current LRMs and establishes a benchmark for advancing high-level logical reasoning. 

**Abstract (ZH)**: HardcoreLogic：面向逻辑游戏“长尾”的具有挑战性的基准 

---
# Inclusive Fitness as a Key Step Towards More Advanced Social Behaviors in Multi-Agent Reinforcement Learning Settings 

**Title (ZH)**: 纳入适应度作为迈向多智能体强化学习中更高级社会行为的关键步骤 

**Authors**: Andries Rosseau, Raphaël Avalos, Ann Nowé  

**Link**: [PDF](https://arxiv.org/pdf/2510.12555)  

**Abstract**: The competitive and cooperative forces of natural selection have driven the evolution of intelligence for millions of years, culminating in nature's vast biodiversity and the complexity of human minds. Inspired by this process, we propose a novel multi-agent reinforcement learning framework where each agent is assigned a genotype and where reward functions are modelled after the concept of inclusive fitness. An agent's genetic material may be shared with other agents, and our inclusive reward function naturally accounts for this. We study the resulting social dynamics in two types of network games with prisoner's dilemmas and find that our results align with well-established principles from biology, such as Hamilton's rule. Furthermore, we outline how this framework can extend to more open-ended environments with spatial and temporal structure, finite resources, and evolving populations. We hypothesize the emergence of an arms race of strategies, where each new strategy is a gradual improvement over earlier adaptations of other agents, effectively producing a multi-agent autocurriculum analogous to biological evolution. In contrast to the binary team-based structures prevalent in earlier research, our gene-based reward structure introduces a spectrum of cooperation ranging from full adversity to full cooperativeness based on genetic similarity, enabling unique non team-based social dynamics. For example, one agent having a mutual cooperative relationship with two other agents, while the two other agents behave adversarially towards each other. We argue that incorporating inclusive fitness in agents provides a foundation for the emergence of more strategically advanced and socially intelligent agents. 

**Abstract (ZH)**: 自然选择的竞争与合作力量驱使智能演化了数百万年，造就了自然界丰富的生物多样性和人类复杂的心智。受此过程启发，我们提出了一种新的多智能体强化学习框架，每个智能体被赋予一个基因型，并且奖励函数模仿泛化亲和力的概念建模。智能体的遗传物质可以与其他智能体共享，我们的泛化奖励函数自然地考虑了这一点。我们在两种具有囚徒困境类型的网络游戏中研究了由此产生的社会动态，发现我们的结果与生物学中已确立的原则，如哈密尔顿规则，相一致。此外，我们概述了该框架如何扩展到具有空间和时间结构、有限资源和演化的种群的更开放的环境中。我们认为，将会出现一种策略的军备竞赛，其中每种新策略都是对其他智能体早期适应的逐步改进，从而产生类似于生物演化的多智能体自闭环课程。与早期研究中占主导的二元团队结构不同，基于基因的奖励结构引入了从完全敌对到完全合作的合作连续谱，使得独特的非团队社会动态成为可能。例如，一个智能体与另外两个智能体形成互惠合作关系，而另外两个智能体之间则表现出敌对行为。我们认为，在智能体中引入泛化亲和力为更加战略性先进且社交智能的智能体的涌现提供了基础。 

---
# ProtoSiTex: Learning Semi-Interpretable Prototypes for Multi-label Text Classification 

**Title (ZH)**: ProtoSiTex: 学习半解释性原型进行多标签文本分类 

**Authors**: Utsav Kumar Nareti, Suraj Kumar, Soumya Pandey, Soumi Chattopadhyay, Chandranath Adak  

**Link**: [PDF](https://arxiv.org/pdf/2510.12534)  

**Abstract**: The surge in user-generated reviews has amplified the need for interpretable models that can provide fine-grained insights. Existing prototype-based models offer intuitive explanations but typically operate at coarse granularity (sentence or document level) and fail to address the multi-label nature of real-world text classification. We propose ProtoSiTex, a semi-interpretable framework designed for fine-grained multi-label text classification. ProtoSiTex employs a dual-phase alternating training strategy: an unsupervised prototype discovery phase that learns semantically coherent and diverse prototypes, and a supervised classification phase that maps these prototypes to class labels. A hierarchical loss function enforces consistency across sub-sentence, sentence, and document levels, enhancing interpretability and alignment. Unlike prior approaches, ProtoSiTex captures overlapping and conflicting semantics using adaptive prototypes and multi-head attention. We also introduce a benchmark dataset of hotel reviews annotated at the sub-sentence level with multiple labels. Experiments on this dataset and two public benchmarks (binary and multi-class) show that ProtoSiTex achieves state-of-the-art performance while delivering faithful, human-aligned explanations, establishing it as a robust solution for semi-interpretable multi-label text classification. 

**Abstract (ZH)**: 用户生成的评论激增加大了对可解释模型的需求，这些模型能够提供细致入微的洞察。现有的原型基模型提供了直观的解释，但通常在粗粒度级别（句子或文档级别）上运行，并且无法解决真实世界文本分类的多标签性质。我们提出了一种半可解释框架ProtoSiTex，旨在实现细致入微的多标签文本分类。ProtoSiTex采用了双阶段交替训练策略：无监督的原型发现阶段，学习语义一致且多样化的原型，以及监督分类阶段，将这些原型映射到类别标签。多层次的损失函数保证了子句、句子和文档级别的一致性，增强了解释的可解释性和对齐性。与之前的方法不同，ProtoSiTex 使用自适应原型和多头注意力机制捕获重叠和冲突的语义。我们还引入了一个酒店评论基准数据集，每个评论在子句级别上标注了多个标签。在该数据集以及两个公共基准数据集（二分类和多分类）上的实验表明，ProtoSiTex 在实现最佳性能的同时提供了忠实且与人类对齐的解释，确立了其作为半可解释多标签文本分类稳健解决方案的地位。 

---
# Artificial Intelligence Virtual Cells: From Measurements to Decisions across Modality, Scale, Dynamics, and Evaluation 

**Title (ZH)**: 人工智能虚拟细胞：从度量到跨模态、尺度、动力学和评估的决策 

**Authors**: Chengpeng Hu, Calvin Yu-Chian Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12498)  

**Abstract**: Artificial Intelligence Virtual Cells (AIVCs) aim to learn executable, decision-relevant models of cell state from multimodal, multiscale measurements. Recent studies have introduced single-cell and spatial foundation models, improved cross-modality alignment, scaled perturbation atlases, and explored pathway-level readouts. Nevertheless, although held-out validation is standard practice, evaluations remain predominantly within single datasets and settings; evidence indicates that transport across laboratories and platforms is often limited, that some data splits are vulnerable to leakage and coverage bias, and that dose, time and combination effects are not yet systematically handled. Cross-scale coupling also remains constrained, as anchors linking molecular, cellular and tissue levels are sparse, and alignment to scientific or clinical readouts varies across studies. We propose a model-agnostic Cell-State Latent (CSL) perspective that organizes learning via an operator grammar: measurement, lift/project for cross-scale coupling, and intervention for dosing and scheduling. This view motivates a decision-aligned evaluation blueprint across modality, scale, context and intervention, and emphasizes function-space readouts such as pathway activity, spatial neighborhoods and clinically relevant endpoints. We recommend operator-aware data design, leakage-resistant partitions, and transparent calibration and reporting to enable reproducible, like-for-like comparisons. 

**Abstract (ZH)**: 人工 Intelligence 虚拟细胞 (AIVCs) 旨在从多模态、多层次测量中学习可执行的、与决策相关的细胞状态模型。最近的研究引入了单细胞和空间基础模型，改进了跨模态对齐，扩展了扰动图谱，并探索了通路级读数。然而，尽管留存验证是标准做法，评估依然主要局限于单个数据集和设置；证据表明，跨实验室和平台的传输往往是有限的，某些数据拆分容易出现泄漏和覆盖偏差，且剂量、时间和组合效应尚未系统处理。跨层次耦合也仍然是受限的，因为分子、细胞和组织级别之间的链接锚点稀少，且与科学或临床读数的对齐在不同研究中变化不一。我们提出了一种模型无关的细胞状态隐空间（CSL）视角，通过操作符文法组织学习：测量、跨层次耦合的提升/投影，以及剂量和调度的操作。这种视角激励了一种跨模态、跨层次、跨上下文和跨干预的决策对齐评估蓝图，并强调功能空间读数，如通路活动、空间邻域和临床相关终点。我们建议具有操作符意识的数据设计、防泄漏分割、透明的校准和报告以实现可重复的、同类比较。 

---
# Using Medical Algorithms for Task-Oriented Dialogue in LLM-Based Medical Interviews 

**Title (ZH)**: 基于LLM的医疗访谈中面向任务的对话医疗算法的应用 

**Authors**: Rui Reis, Pedro Rangel Henriques, João Ferreira-Coimbra, Eva Oliveira, Nuno F. Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2510.12490)  

**Abstract**: We developed a task-oriented dialogue framework structured as a Directed Acyclic Graph (DAG) of medical questions. The system integrates: (1) a systematic pipeline for transforming medical algorithms and guidelines into a clinical question corpus; (2) a cold-start mechanism based on hierarchical clustering to generate efficient initial questioning without prior patient information; (3) an expand-and-prune mechanism enabling adaptive branching and backtracking based on patient responses; (4) a termination logic to ensure interviews end once sufficient information is gathered; and (5) automated synthesis of doctor-friendly structured reports aligned with clinical workflows. Human-computer interaction principles guided the design of both the patient and physician applications. Preliminary evaluation involved five physicians using standardized instruments: NASA-TLX (cognitive workload), the System Usability Scale (SUS), and the Questionnaire for User Interface Satisfaction (QUIS). The patient application achieved low workload scores (NASA-TLX = 15.6), high usability (SUS = 86), and strong satisfaction (QUIS = 8.1/9), with particularly high ratings for ease of learning and interface design. The physician application yielded moderate workload (NASA-TLX = 26) and excellent usability (SUS = 88.5), with satisfaction scores of 8.3/9. Both applications demonstrated effective integration into clinical workflows, reducing cognitive demand and supporting efficient report generation. Limitations included occasional system latency and a small, non-diverse evaluation sample. 

**Abstract (ZH)**: 我们开发了一种基于有向无环图（DAG）的医学问题结构化任务导向对话框架。该系统包括：（1）一套系统化的医疗算法和指南转化为临床问题库的流水线；（2）基于层次聚类的冷启动机制，以生成高效初始问题，无需先验患者信息；（3）扩展和修剪机制，根据患者反馈实现自适应分支和回溯；（4）终止逻辑以确保在收集足够信息后结束访谈；以及（5）与临床工作流程对齐的医生友好的结构化报告的自动化合成。人机交互原则指导了患者和医生应用的设计。初步评估涉及五名医生使用标准化量表：NASA-TLX（认知负载）、系统可用性量表（SUS）和用户界面满意度问卷（QUIS）。患者应用取得了较低的认知负载评分（NASA-TLX = 15.6）、较高的易用性（SUS = 86）和较强的满意度（QUIS = 8.1/9），特别是在学习难易度和界面设计方面获得了高评分。医生应用的认知负载适中（NASA-TLX = 26）、易用性极佳（SUS = 88.5），满意度评分为8.3/9。这两种应用都有效地整合到了临床工作流程中，减少了认知需求并支持了高效的报告生成。局限性包括偶尔的系统延迟和较小的非多样化评估样本。 

---
# Evaluating and Mitigating LLM-as-a-judge Bias in Communication Systems 

**Title (ZH)**: 评价和缓解语言模型作为裁判的偏见在通信系统中的影响 

**Authors**: Jiaxin Gao, Chen Chen, Yanwen Jia, Xueluan Gong, Kwok-Yan Lam, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12462)  

**Abstract**: Large Language Models (LLMs) are increasingly being used to autonomously evaluate the quality of content in communication systems, e.g., to assess responses in telecom customer support chatbots. However, the impartiality of these AI "judges" is not guaranteed, and any biases in their evaluation criteria could skew outcomes and undermine user trust. In this paper, we systematically investigate judgment biases in two LLM-as-a-judge models (i.e., GPT-Judge and JudgeLM) under the point-wise scoring setting, encompassing 11 types of biases that cover both implicit and explicit forms. We observed that state-of-the-art LLM judges demonstrate robustness to biased inputs, generally assigning them lower scores than the corresponding clean samples. Providing a detailed scoring rubric further enhances this robustness. We further found that fine-tuning an LLM on high-scoring yet biased responses can significantly degrade its performance, highlighting the risk of training on biased data. We also discovered that the judged scores correlate with task difficulty: a challenging dataset like GPQA yields lower average scores, whereas an open-ended reasoning dataset (e.g., JudgeLM-val) sees higher average scores. Finally, we proposed four potential mitigation strategies to ensure fair and reliable AI judging in practical communication scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通信系统中自主评估内容质量的应用越来越广泛，例如评价电信客服聊天机器人的回应。然而，这些AI“裁判”的公正性并不保证，其评价标准中的任何偏差都可能导致结果失真并削弱用户信任。本文系统地研究了两种LLM-as-a-judge模型（即GPT-Judge和JudgeLM）在点评分设置下的判断偏差，涵盖了11种偏差，包括隐性和显性两种形式。我们发现，最先进的LLM裁判对有偏见的输入表现出较强的鲁棒性，通常会给予较低的评分，而相应的干净样本则获得较高评分。提供详细的评分标准进一步增强了这种鲁棒性。进一步的研究发现，对LLM进行高分但有偏见的回应的微调会显著降低其性能，突出了使用有偏见数据进行训练的风险。我们还发现，评估分数与任务难度相关：如GPQA这样具有挑战性的数据集会导致较低的平均评分，而开放性推理数据集（如JudgeLM-val）则会有较高的平均评分。最后，我们提出了四种潜在的缓解策略，以确保在实际通信场景中实现公平可靠的AI评判。 

---
# Biased-Attention Guided Risk Prediction for Safe Decision-Making at Unsignalized Intersections 

**Title (ZH)**: 基于偏向注意力的风险预测方法以实现无信号交叉口的安全决策 

**Authors**: Chengyang Dong, Nan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12428)  

**Abstract**: Autonomous driving decision-making at unsignalized intersections is highly challenging due to complex dynamic interactions and high conflict risks. To achieve proactive safety control, this paper proposes a deep reinforcement learning (DRL) decision-making framework integrated with a biased attention mechanism. The framework is built upon the Soft Actor-Critic (SAC) algorithm. Its core innovation lies in the use of biased attention to construct a traffic risk predictor. This predictor assesses the long-term risk of collision for a vehicle entering the intersection and transforms this risk into a dense reward signal to guide the SAC agent in making safe and efficient driving decisions. Finally, the simulation results demonstrate that the proposed method effectively improves both traffic efficiency and vehicle safety at the intersection, thereby proving the effectiveness of the intelligent decision-making framework in complex scenarios. The code of our work is available at this https URL. 

**Abstract (ZH)**: 无信号交叉口的自动驾驶决策极具挑战性，由于复杂的动态交互和高冲突风险。为了实现主动安全控制，本文提出了一种结合偏置注意力机制的深度强化学习（DRL）决策框架，该框架基于Soft Actor-Critic（SAC）算法。其核心创新在于使用偏置注意力构建交通风险预测器，该预测器评估进入交叉口的车辆的长期碰撞风险，并将此风险转化为密集的奖励信号，以指导SAC代理做出安全高效的驾驶决策。最终，仿真结果表明，所提出的方法有效提高了交叉口的交通效率和车辆安全，从而证明了在复杂场景中智能决策框架的有效性。我们的代码可通过以下链接获取：this https URL。 

---
# MTOS: A LLM-Driven Multi-topic Opinion Simulation Framework for Exploring Echo Chamber Dynamics 

**Title (ZH)**: MTOS：一个由大规模语言模型驱动的多话题意见模拟框架，用于探索回声室效应动态 

**Authors**: Dingyi Zuo, Hongjie Zhang, Jie Ou, Chaosheng Feng, Shuwan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12423)  

**Abstract**: The polarization of opinions, information segregation, and cognitive biases on social media have attracted significant academic attention. In real-world networks, information often spans multiple interrelated topics, posing challenges for opinion evolution and highlighting the need for frameworks that simulate interactions among topics. Existing studies based on large language models (LLMs) focus largely on single topics, limiting the capture of cognitive transfer in multi-topic, cross-domain contexts. Traditional numerical models, meanwhile, simplify complex linguistic attitudes into discrete values, lacking interpretability, behavioral consistency, and the ability to integrate multiple topics. To address these issues, we propose Multi-topic Opinion Simulation (MTOS), a social simulation framework integrating multi-topic contexts with LLMs. MTOS leverages LLMs alongside short-term and long-term memory, incorporates multiple user-selection interaction mechanisms and dynamic topic-selection strategies, and employs a belief decay mechanism to enable perspective updates across topics. We conduct extensive experiments on MTOS, varying topic numbers, correlation types, and performing ablation studies to assess features such as group polarization and local consistency. Results show that multi-topic settings significantly alter polarization trends: positively correlated topics amplify echo chambers, negatively correlated topics inhibit them, and irrelevant topics also mitigate echo chamber effects through resource competition. Compared with numerical models, LLM-based agents realistically simulate dynamic opinion changes, reproduce linguistic features of news texts, and capture complex human reasoning, improving simulation interpretability and system stability. 

**Abstract (ZH)**: 社交媒体上的观点极化、信息隔离和认知偏见的学术关注日益增加。在现实世界网络中，信息往往涉及多个相互关联的主题，这为意见演化带来了挑战，凸显了需要模拟主题间交互框架的重要性。基于大型语言模型（LLMs）的现有研究主要集中在单个主题上，限制了对多主题、跨领域情境中的认知转移的捕捉。同时，传统的数值模型将复杂的语言态度简化为离散值，缺乏可解释性、行为一致性以及多主题整合的能力。为了解决这些问题，我们提出了一种集成多主题上下文与LLMs的社会仿真框架——多主题意见模拟（MTOS）。MTOS结合了LLMs及其短期和长期记忆，融合了多种用户选择交互机制和动态主题选择策略，并采用信念衰减机制以支持跨主题的观点更新。我们在MTOS上进行了广泛的实验，变化主题数量、相关类型，并进行消融研究以评估群体极化和局部一致性等特征。结果显示，多主题设置显著改变了极化趋势：正相关主题放大了回声室效应，负相关主题抑制了回声室效应，而且无关主题也通过资源竞争减轻了回声室效应。与数值模型相比，基于LLMs的代理可以更真实地模拟动态意见变化，再现新闻文本的语言特征，并捕捉复杂的逻辑推理，从而提高仿真可解释性和系统稳定性。 

---
# PricingLogic: Evaluating LLMs Reasoning on Complex Tourism Pricing Tasks 

**Title (ZH)**: 定价逻辑：评估大语言模型在复杂旅游定价任务中的推理能力 

**Authors**: Yunuo Liu, Dawei Zhu, Zena Al-Khalili, Dai Cheng, Yanjun Chen, Dietrich Klakow, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12409)  

**Abstract**: We present PricingLogic, the first benchmark that probes whether Large Language Models(LLMs) can reliably automate tourism-related prices when multiple, overlapping fare rules apply. Travel agencies are eager to offload this error-prone task onto AI systems; however, deploying LLMs without verified reliability could result in significant financial losses and erode customer trust. PricingLogic comprises 300 natural-language questions based on booking requests derived from 42 real-world pricing policies, spanning two levels of difficulty: (i) basic customer-type pricing and (ii)bundled-tour calculations involving interacting discounts. Evaluations of a line of LLMs reveal a steep performance drop on the harder tier,exposing systematic failures in rule interpretation and arithmetic this http URL results highlight that, despite their general capabilities, today's LLMs remain unreliable in revenue-critical applications without further safeguards or domain adaptation. Our code and dataset are available at this https URL. 

**Abstract (ZH)**: PricingLogic：首个探究大型语言模型在复杂票价规则下可靠自动化定价能力的基准测验 

---
# A Survey of Vibe Coding with Large Language Models 

**Title (ZH)**: 大型语言模型中的Vibe编码综述 

**Authors**: Yuyao Ge, Lingrui Mei, Zenghao Duan, Tianhao Li, Yujia Zheng, Yiwei Wang, Lexin Wang, Jiayu Yao, Tianyu Liu, Yujun Cai, Baolong Bi, Fangda Guo, Jiafeng Guo, Shenghua Liu, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12399)  

**Abstract**: The advancement of large language models (LLMs) has catalyzed a paradigm shift from code generation assistance to autonomous coding agents, enabling a novel development methodology termed "Vibe Coding" where developers validate AI-generated implementations through outcome observation rather than line-by-line code comprehension. Despite its transformative potential, the effectiveness of this emergent paradigm remains under-explored, with empirical evidence revealing unexpected productivity losses and fundamental challenges in human-AI collaboration. To address this gap, this survey provides the first comprehensive and systematic review of Vibe Coding with large language models, establishing both theoretical foundations and practical frameworks for this transformative development approach. Drawing from systematic analysis of over 1000 research papers, we survey the entire vibe coding ecosystem, examining critical infrastructure components including LLMs for coding, LLM-based coding agent, development environment of coding agent, and feedback mechanisms. We first introduce Vibe Coding as a formal discipline by formalizing it through a Constrained Markov Decision Process that captures the dynamic triadic relationship among human developers, software projects, and coding agents. Building upon this theoretical foundation, we then synthesize existing practices into five distinct development models: Unconstrained Automation, Iterative Conversational Collaboration, Planning-Driven, Test-Driven, and Context-Enhanced Models, thus providing the first comprehensive taxonomy in this domain. Critically, our analysis reveals that successful Vibe Coding depends not merely on agent capabilities but on systematic context engineering, well-established development environments, and human-agent collaborative development models. 

**Abstract (ZH)**: 大型语言模型的发展催化了从代码生成辅助到自主 coding 代理的范式转变，开启了通过结果观察而非逐行代码理解来验证 AI 生成实现的“Vibe Coding”新型开发方法。尽管具有革命性潜力，这一新兴范式的有效性仍需进一步探索，实验研究表明存在意想不到的生产力损失和人机协作的基本挑战。为填补这一空白，本文综述了大型语言模型下的 Vibe Coding，为其提供了理论基础和实践框架。基于对超过1000篇研究论文的系统分析，我们全面审视了整个 Vibe Coding 生态系统，探讨了关键基础设施组件，包括用于编程的大型语言模型、基于大型语言模型的编程代理、编程代理的开发环境以及反馈机制。我们首先通过受限马尔可夫决策过程形式化 Vibe Coding，捕捉人类开发者、软件项目和编程代理之间的动态三角关系，以此为基础，我们将现有实践综合提炼为五种不同的开发模型：无约束自动化、迭代对话协作、计划驱动、测试驱动以及增强情境模型，从而首次在该领域提供了全面的分类体系。我们的分析表明，成功的 Vibe Coding 不仅取决于代理的能力，还取决于系统的情境工程、成熟的开发环境以及人机协作开发模型。 

---
# O-Forge: An LLM + Computer Algebra Framework for Asymptotic Analysis 

**Title (ZH)**: O-Forge: 一个基于大语言模型与计算机代数的渐近分析框架 

**Authors**: Ayush Khaitan, Vijay Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2510.12350)  

**Abstract**: Large language models have recently demonstrated advanced capabilities in solving IMO and Putnam problems; yet their role in research mathematics has remained fairly limited. The key difficulty is verification: suggested proofs may look plausible, but cannot be trusted without rigorous checking. We present a framework, called LLM+CAS, and an associated tool, O-Forge, that couples frontier LLMs with a computer algebra systems (CAS) in an In-Context Symbolic Feedback loop to produce proofs that are both creative and symbolically verified. Our focus is on asymptotic inequalities, a topic that often involves difficult proofs and appropriate decomposition of the domain into the "right" subdomains. Many mathematicians, including Terry Tao, have suggested that using AI tools to find the right decompositions can be very useful for research-level asymptotic analysis. In this paper, we show that our framework LLM+CAS turns out to be remarkably effective at proposing such decompositions via a combination of a frontier LLM and a CAS. More precisely, we use an LLM to suggest domain decomposition, and a CAS (such as Mathematica) that provides a verification of each piece axiomatically. Using this loop, we answer a question posed by Terence Tao: whether LLMs coupled with a verifier can be used to help prove intricate asymptotic inequalities. More broadly, we show how AI can move beyond contest math towards research-level tools for professional mathematicians. 

**Abstract (ZH)**: 大规模语言模型在解决IMO和普特南问题上展示了先进的能力，但在研究数学中的作用仍然相当有限；其关键难题在于验证：提出的证明可能看似合乎道理，但必须经过严格的检查才能信赖。我们提出了一种称为LLM+CAS的框架和相应的工具O-Forge，通过将前沿的大规模语言模型与计算机代数系统（CAS）结合在一个基于符号反馈的上下文循环中，生成既具有创新性又能符号验证的证明。我们的重点是渐近不等式，这是一个经常涉及复杂证明和正确分解域的主题。许多数学家，包括陶哲轩，建议使用AI工具来寻找合适的分解对于研究级别的渐近分析非常有用。本文展示了我们的LLM+CAS框架通过结合前沿的大规模语言模型和CAS在提出此类分解方面表现出显著的效果。具体而言，我们使用大规模语言模型建议领域分解，CAS（如Mathematica）对每部分进行公理验证。通过这个循环，我们回答了陶哲轩提出的疑问：将大规模语言模型与验证器结合是否能用于帮助证明复杂的渐近不等式。更广泛地说，我们展示了AI如何从竞赛数学转向为专业数学家提供研究级别工具。 

---
# RAG-Anything: All-in-One RAG Framework 

**Title (ZH)**: RAG-anything: 一站式RAG框架 

**Authors**: Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12323)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm for expanding Large Language Models beyond their static training limitations. However, a critical misalignment exists between current RAG capabilities and real-world information environments. Modern knowledge repositories are inherently multimodal, containing rich combinations of textual content, visual elements, structured tables, and mathematical expressions. Yet existing RAG frameworks are limited to textual content, creating fundamental gaps when processing multimodal documents. We present RAG-Anything, a unified framework that enables comprehensive knowledge retrieval across all modalities. Our approach reconceptualizes multimodal content as interconnected knowledge entities rather than isolated data types. The framework introduces dual-graph construction to capture both cross-modal relationships and textual semantics within a unified representation. We develop cross-modal hybrid retrieval that combines structural knowledge navigation with semantic matching. This enables effective reasoning over heterogeneous content where relevant evidence spans multiple modalities. RAG-Anything demonstrates superior performance on challenging multimodal benchmarks, achieving significant improvements over state-of-the-art methods. Performance gains become particularly pronounced on long documents where traditional approaches fail. Our framework establishes a new paradigm for multimodal knowledge access, eliminating the architectural fragmentation that constrains current systems. Our framework is open-sourced at: this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）已成为一种基本范式，用于扩展大型语言模型超越其静态训练限制。然而，当前RAG能力与实际信息环境之间存在关键不匹配。现代知识库本质上是多模态的，包含丰富的文本内容、视觉元素、结构化表格和数学表达式。然而，现有的RAG框架仅限于处理文本内容，处理多模态文档时存在根本性差距。我们提出了RAG-Anything统一框架，以实现全方位的知识检索跨所有模态。我们的方法将多模态内容重新构想为相互关联的知识实体，而非孤立的数据类型。框架引入了双图构建，以在统一表示中捕捉跨模态关系和文本语义。我们开发了跨模态混合检索，结合结构化知识导航与语义匹配。这使我们在异构内容中能够有效推理，其中相关证据跨越多种模态。RAG-Anything在具有挑战性的多模态基准测试中表现出优越性能，显著优于现有方法。在长文档中，性能提升尤其明显，传统方法在这种情况下失效。我们的框架建立了新的多模态知识访问范式，消除了现有系统中的架构碎片化限制。我们的框架在以下地址开源：this https URL。 

---
# Tensor Logic: The Language of AI 

**Title (ZH)**: 张量逻辑：AI的语言 

**Authors**: Pedro Domingos  

**Link**: [PDF](https://arxiv.org/pdf/2510.12269)  

**Abstract**: Progress in AI is hindered by the lack of a programming language with all the requisite features. Libraries like PyTorch and TensorFlow provide automatic differentiation and efficient GPU implementation, but are additions to Python, which was never intended for AI. Their lack of support for automated reasoning and knowledge acquisition has led to a long and costly series of hacky attempts to tack them on. On the other hand, AI languages like LISP an Prolog lack scalability and support for learning. This paper proposes tensor logic, a language that solves these problems by unifying neural and symbolic AI at a fundamental level. The sole construct in tensor logic is the tensor equation, based on the observation that logical rules and Einstein summation are essentially the same operation, and all else can be reduced to them. I show how to elegantly implement key forms of neural, symbolic and statistical AI in tensor logic, including transformers, formal reasoning, kernel machines and graphical models. Most importantly, tensor logic makes new directions possible, such as sound reasoning in embedding space. This combines the scalability and learnability of neural networks with the reliability and transparency of symbolic reasoning, and is potentially a basis for the wider adoption of AI. 

**Abstract (ZH)**: AI进展受限于缺乏一种具备所有必要特征的编程语言。本文提出张量逻辑语言，该语言通过在根本层面上统一神经AI和符号AI来解决这些问题。张量逻辑语言唯一的构造是张量方程，基于逻辑规则和爱因斯坦求和本质上是同一操作的观察，所有其他操作都可以归约到它们。本文展示了如何优雅地在张量逻辑中实现关键形式的神经、符号和统计AI，包括变换器、形式推理、核机器和图形模型。最重要的是，张量逻辑开辟了新的方向，例如嵌入空间中的可靠推理。这结合了神经网络的可扩展性和可学习性以及符号推理的可靠性和透明性，并有可能成为更广泛采用AI的基础。 

---
# $\mathbf{T^3}$: Reducing Belief Deviation in Reinforcement Learning for Active Reasoning 

**Title (ZH)**: $\mathbf{T^3}$: 减少主动推理中强化学习中的信念偏差 

**Authors**: Deyu Zou, Yongqiang Chen, Jianxiang Wang, Haochen Yang, Mufei Li, James Cheng, Pan Li, Yu Gong  

**Link**: [PDF](https://arxiv.org/pdf/2510.12264)  

**Abstract**: Active reasoning requires large language models (LLMs) to interact with external sources and strategically gather information to solve problems. Central to this process is belief tracking: maintaining a coherent understanding of the problem state and the missing information toward the solution. However, due to limited reasoning capabilities, LLM-based agents often suffer from belief deviation: they struggle to correctly model beliefs, lose track of problem states, and fall into uninformative or repetitive actions. Once this happens, errors compound and reinforcement learning (RL) training fails to properly credit the crucial exploratory steps. To address this issue, we propose to track the deviation of model beliefs and develop $\mathbf{T^3}$, a simple yet effective method that detects excessive belief deviation and truncates trajectories during training to remove uninformative tails. By preserving credit for informative prefixes, $\mathbf{T^3}$ systematically improves policy optimization. Across 5 challenging tasks, $\mathbf{T^3}$ consistently enhances training stability, token efficiency, and final performance, achieving up to 30% gains while cutting rollout tokens by roughly 25%. These results highlight belief control as a key principle for developing robust and generalizable LLM-based active reasoners. 

**Abstract (ZH)**: 基于大型语言模型的主动推理要求与外部来源互动并战略性地收集信息以解决问题。这一过程的核心是信念跟踪：维持对问题状态和解决过程中缺失信息的连贯理解。然而，由于推理能力有限，基于大型语言模型的代理经常出现信念偏差：难以正确建模信念、丢失问题状态跟踪，并陷入无信息或重复的行动中。一旦发生这种情况，错误会累积，强化学习训练无法正确奖励关键的探索步骤。为解决这一问题，我们提出跟踪模型信念的偏差，并开发了 $\mathbf{T^3}$，一种简单而有效的方法来检测过度信念偏差，并在训练中截断轨迹以移除无信息的部分。通过保留信息前缀的信用，$\mathbf{T^3}$ 系统性地改善了策略优化。在5项具有挑战性的任务中，$\mathbf{T^3}$ 一致地增强了训练稳定性、token 效率和最终性能，实现了高达30%的提升，同时削减了约25%的展开token。这些结果强调了信念控制作为开发鲁棒且通用的大规模语言模型基于的主动推理者的关键原则。 

---
# PromptFlow: Training Prompts Like Neural Networks 

**Title (ZH)**: PromptFlow：像神经网络一样训练提示詞 

**Authors**: Jingyi Wang, Hongyuan Zhu, Ye Niu, Yunhui Deng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12246)  

**Abstract**: Large Language Models (LLMs) have demonstrated profound impact on Natural Language Processing (NLP) tasks. However, their effective deployment across diverse domains often require domain-specific adaptation strategies, as generic models may underperform when faced with specialized data distributions. Recent advances in prompt engineering (PE) offer a promising alternative to extensive retraining by refining input instructions to align LLM outputs with task objectives. This paradigm has emerged as a rapid and versatile approach for model fine-tuning. Despite its potential, manual prompt design remains labor-intensive and heavily depends on specialized expertise, often requiring iterative human effort to achieve optimal formulations. To address this limitation, automated prompt engineering methodologies have been developed to systematically generate task-specific prompts. However, current implementations predominantly employ static update rules and lack mechanisms for dynamic strategy selection, resulting in suboptimal adaptation to varying NLP task requirements. Furthermore, most methods treat and update the whole prompts at each step, without considering editing prompt sections at a finer granularity. At last, in particular, the problem of how to recycle experience in LLM is still underexplored. To this end, we propose the PromptFlow, a modular training framework inspired by TensorFlow, which integrates meta-prompts, operators, optimization, and evaluator. Our framework can be equipped with the latest optimization methods and autonomously explores optimal prompt refinement trajectories through gradient-based meta-learning, requiring minimal task-specific training data. Specifically, we devise a reinforcement learning method to recycle experience for LLM in the PE process. Finally, we conduct extensive experiments on various datasets, and demonstrate the effectiveness of PromptFlow. 

**Abstract (ZH)**: 大规模语言模型(LLMs)在自然语言处理(NLP)任务中展现了深远的影响。然而，它们在多种领域的有效部署通常需要特定领域的适应策略，因为通用模型在面对专业化数据分布时可能会表现不佳。最近在提示工程(PE)方面的进展为替代广泛的再训练提供了一种有前途的选择，通过精炼输入指令以使LLM输出与任务目标对齐。这种范式已成为模型微调的一种快速且多功能的方法。尽管存在这些潜力，但手动提示设计仍然劳动密集且高度依赖专门的知识，通常需要迭代的人工努力才能达到最佳表述。为了解决这一限制，已经开发了自动提示工程方法以系统地生成任务专用的提示。然而，当前的实现大多采用静态更新规则，并缺乏动态策略选择机制，导致适应不同NLP任务需求的效果不ideal。此外，大多数方法在每一步都对整个提示进行处理和更新，而不考虑在更细粒度上编辑提示段落。最后，特别地，如何在LLM中循环利用经验的问题仍然未得到充分探索。为此，我们提出了PromptFlow，这是一种受TensorFlow启发的模块化训练框架，结合了元提示、操作符、优化和评估器。我们的框架可以配备最新的优化方法，并通过基于梯度的元学习自主探索最优提示改进轨迹，要求最少的任务特定训练数据。具体而言，我们设计了一种强化学习方法，在提示工程(PE)过程中为LLM循环利用经验。最后，我们在各种数据集上进行了广泛的实验，并展示了PromptFlow的有效性。 

---
# MedKGEval: A Knowledge Graph-Based Multi-Turn Evaluation Framework for Open-Ended Patient Interactions with Clinical LLMs 

**Title (ZH)**: MedKGEval：基于知识图谱的多轮临床LLM开放域患者交互评估框架 

**Authors**: Yuechun Yu, Han Ying, Haoan Jin, Wenjian Jiang, Dong Xian, Binghao Wang, Zhou Yang, Mengyue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12224)  

**Abstract**: The reliable evaluation of large language models (LLMs) in medical applications remains an open challenge, particularly in capturing the complexity of multi-turn doctor-patient interactions that unfold in real clinical environments. Existing evaluation methods typically rely on post hoc review of full conversation transcripts, thereby neglecting the dynamic, context-sensitive nature of medical dialogues and the evolving informational needs of patients. In this work, we present MedKGEval, a novel multi-turn evaluation framework for clinical LLMs grounded in structured medical knowledge. Our approach introduces three key contributions: (1) a knowledge graph-driven patient simulation mechanism, where a dedicated control module retrieves relevant medical facts from a curated knowledge graph, thereby endowing the patient agent with human-like and realistic conversational behavior. This knowledge graph is constructed by integrating open-source resources with additional triples extracted from expert-annotated datasets; (2) an in-situ, turn-level evaluation framework, where each model response is assessed by a Judge Agent for clinical appropriateness, factual correctness, and safety as the dialogue progresses using a suite of fine-grained, task-specific metrics; (3) a comprehensive multi-turn benchmark of eight state-of-the-art LLMs, demonstrating MedKGEval's ability to identify subtle behavioral flaws and safety risks that are often overlooked by conventional evaluation pipelines. Although initially designed for Chinese and English medical applications, our framework can be readily extended to additional languages by switching the input knowledge graphs, ensuring seamless bilingual support and domain-specific applicability. 

**Abstract (ZH)**: 可靠评估医疗应用中的大规模语言模型（LLMs）仍是一个开放挑战，特别是在捕捉实际临床环境中多轮医患交互的复杂性方面。现有的评估方法通常依赖于事后审查完整的对话转录，因此忽视了医疗对话的动态性和上下文敏感性以及患者不断变化的信息需求。在本文中，我们提出了MedKGEval，一个基于结构化医学知识的新型多轮评估框架。我们的方法引入了三个关键贡献：（1）知识图驱动的患者模拟机制，其中专用控制模块从精心策划的知识图中检索相关医学事实，从而使患者代理具备类似人类的真实对话行为。该知识图结合了开源资源和专家标注数据集中提取的额外三元组构建而成；（2）实时、轮次级别的评估框架，其中每个模型响应在对话进行过程中由法官代理评估临床适切性、事实正确性和安全性，使用一系列精细粒度的任务特定指标；（3）八个最先进的LLM的综合多轮基准测试，展示了MedKGEval能够识别传统评估流水线常常忽略的微妙行为缺陷和安全风险。尽管最初为中文和英文医疗应用设计，但该框架可以通过切换输入知识图来轻松扩展到其他语言，确保无缝的双语支持和特定领域的适用性。 

---
# GOAT: A Training Framework for Goal-Oriented Agent with Tools 

**Title (ZH)**: GOAT：面向目标导向代理的工具训练框架 

**Authors**: Hyunji Min, Sangwon Jung, Junyoung Sung, Dosung Lee, Leekyeung Han, Paul Hongsuck Seo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12218)  

**Abstract**: Large language models (LLMs) have recently been extended beyond traditional text generation to serve as interactive agents capable of using external tools based on user intent. However, current LLM agents still show limited ability to handle goal-oriented queries, which require decomposing a high-level objective into multiple interdependent API calls with correct planning and execution. Current approaches mainly rely on zero-shot evaluation due to the absence of training data. While proprietary closed-source models such as GPT-4 demonstrate strong reasoning abilities, smaller open-source models struggle to perform complex tool use effectively. Thus, we propose a novel training framework GOAT, which enables fine-tuning of LLM agents in a human annotation-free setting. GOAT automatically constructs synthetic datasets of goal-oriented API execution tasks directly from given API documents, equipping models with the ability to reason over interdependent calls and generate coherent responses. Through extensive experiments, we show that GOAT-trained agents achieve state-of-the-art performance across multiple existing goal-oriented benchmarks. In addition, we introduce GOATBench, a new goal-oriented API execution benchmark, and demonstrate that agents trained with GOAT also excel in this setting. These results highlight GOAT as a practical path toward building robust open-source LLM agents capable of complex reasoning and tool use. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已被扩展到超越传统文本生成，作为能够根据用户意图使用外部工具的交互式代理。然而，当前的LLM代理在处理目标导向查询方面仍显示有限的能力，这需要将高层目标分解为多个相互依赖的API调用，并进行正确的计划和执行。当前的方法主要依赖于零样本评估，因为缺少训练数据。虽然像GPT-4这样的私有闭源模型表现出强大的推理能力，但较小的开源模型在有效地使用复杂工具方面仍然挣扎。因此，我们提出了一种新颖的训练框架GOAT，该框架能够在无需人工标注的情况下对LLM代理进行微调。GOAT自动从给定的API文档中构建目标导向的API执行任务的合成数据集，使模型具备推理相互依赖调用并生成连贯响应的能力。通过广泛的实验，我们展示了GOAT微调的代理在多个现有目标导向基准上的性能达到最新水平。此外，我们介绍了GOATBench，一个新的目标导向的API执行基准，并展示了使用GOAT训练的代理在这种环境中也表现出色。这些结果突显了GOAT作为构建具备复杂推理和工具使用能力的稳健开源LLM代理的实用路径。 

---
# On the Design and Evaluation of Human-centered Explainable AI Systems: A Systematic Review and Taxonomy 

**Title (ZH)**: 基于人类中心的可解释人工智能系统的設計与评估：一种系统回顾与分类框架 

**Authors**: Aline Mangold, Juliane Zietz, Susanne Weinhold, Sebastian Pannasch  

**Link**: [PDF](https://arxiv.org/pdf/2510.12201)  

**Abstract**: As AI becomes more common in everyday living, there is an increasing demand for intelligent systems that are both performant and understandable. Explainable AI (XAI) systems aim to provide comprehensible explanations of decisions and predictions. At present, however, evaluation processes are rather technical and not sufficiently focused on the needs of human users. Consequently, evaluation studies involving human users can serve as a valuable guide for conducting user studies. This paper presents a comprehensive review of 65 user studies evaluating XAI systems across different domains and application contexts. As a guideline for XAI developers, we provide a holistic overview of the properties of XAI systems and evaluation metrics focused on human users (human-centered). We propose objectives for the human-centered design (design goals) of XAI systems. To incorporate users' specific characteristics, design goals are adapted to users with different levels of AI expertise (AI novices and data experts). In this regard, we provide an extension to existing XAI evaluation and design frameworks. The first part of our results includes the analysis of XAI system characteristics. An important finding is the distinction between the core system and the XAI explanation, which together form the whole system. Further results include the distinction of evaluation metrics into affection towards the system, cognition, usability, interpretability, and explanation metrics. Furthermore, the users, along with their specific characteristics and behavior, can be assessed. For AI novices, the relevant extended design goals include responsible use, acceptance, and usability. For data experts, the focus is performance-oriented and includes human-AI collaboration and system and user task performance. 

**Abstract (ZH)**: 随着人工智能在日常生活中越来越普遍，对既高效又易理解的智能系统的需求也随之增加。可解释的人工智能（XAI）系统旨在提供决策和预测的可理解解释。然而，目前的评估过程相对技术化，未能充分关注用户需求。因此，涉及人类用户的评估研究可以为用户研究提供有价值的指导。本文综述了65项评估XAI系统的用户研究，涵盖不同领域和应用场景。作为XAI开发者的指南，我们提供了一个综合的XAI系统特性和以用户为中心的评估指标的概述。我们提出了XAI系统的设计目标（以人为本的设计目标）。为了融入用户的特定特征，设计目标被调整以适应不同AI熟练程度的用户（AI新手和数据专家）。在这方面，我们提出了现有XAI评估和设计框架的扩展。结果的第一部分包括对XAI系统特性的分析。一个重要的发现是核心系统和XAI解释之间的区分，两者共同构成了整个系统。其他结果包括将评估指标区分为对系统的感受、认知、易用性、可解释性和解释指标。此外，还可以评估用户及其特定特征和行为。对于AI新手，相关的扩展设计目标包括负责任的使用、接受度和易用性。对于数据专家，重点是面向性能，并包括人机协作和系统及用户任务绩效。 

---
# ResearStudio: A Human-Intervenable Framework for Building Controllable Deep-Research Agents 

**Title (ZH)**: ResearStudio: 一个人机可介入的可控制深度研究代理构建框架 

**Authors**: Linyi Yang, Yixuan Weng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12194)  

**Abstract**: Current deep-research agents run in a ''fire-and-forget'' mode: once started, they give users no way to fix errors or add expert knowledge during execution. We present ResearStudio, the first open-source framework that places real-time human control at its core. The system follows a Collaborative Workshop design. A hierarchical Planner-Executor writes every step to a live ''plan-as-document,'' a fast communication layer streams each action, file change, and tool call to a web interface. At any moment, the user can pause the run, edit the plan or code, run custom commands, and resume -- switching smoothly between AI-led, human-assisted and human-led, AI-assisted modes. In fully autonomous mode, ResearStudio achieves state-of-the-art results on the GAIA benchmark, surpassing systems like OpenAI's DeepResearch and Manus. These results show that strong automated performance and fine-grained human control can coexist. The full code, protocol, and evaluation scripts are available at this https URL. We will continue to update the repository to encourage further work on safe and controllable research agents. Our live demo is publicly accessible at this http URL. We support the development of DeepScientist, which can be accessed at this https URL. 

**Abstract (ZH)**: 当前的深度研究代理以“一次性启动”模式运行：一旦启动，用户在执行过程中无法修正错误或添加专业知识。我们提出了ResearStudio，这是首个将实时人类控制放在核心位置的开源框架。该系统采用协作工作室设计。层次化的计划者-执行者将每一步写入一个实时的“计划文档”，快速通信层将每次操作、文件更改和工具调用流式传输到Web界面。用户可以在任何时刻暂停运行、编辑计划或代码、运行自定义命令并恢复运行——在AI主导、人类辅助和人类主导、AI辅助模式之间平滑切换。在完全自主模式下，ResearStudio在GAIA基准测试中实现了最先进的性能，超越了如OpenAI的DeepResearch和Manus等系统。这些结果表明，强大的自动化性能和精细的人类控制可以共存。完整的代码、协议和评估脚本可在以下链接访问：this https URL。我们将继续更新仓库以促进对安全可控研究代理的进一步研究。我们的实时演示可在以下链接公开访问：this http URL。我们支持DeepScientist的发展，可访问以下链接：this https URL。 

---
# Evolution of meta's llama models and parameter-efficient fine-tuning of large language models: a survey 

**Title (ZH)**: Meta的 llama 模型的发展与大规模语言模型的参数高效微调：一个综述 

**Authors**: Abdulhady Abas Abdullah, Arkaitz Zubiaga, Seyedali Mirjalili, Amir H. Gandomi, Fatemeh Daneshfar, Mohammadsadra Amini, Alan Salam Mohammed, Hadi Veisi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12178)  

**Abstract**: This review surveys the rapid evolution of Meta AI's LLaMA (Large Language Model Meta AI) series - from LLaMA 1 through LLaMA 4 and the specialized parameter-efficient fine-tuning (PEFT) methods developed for these models. We first describe the LLaMA family of foundation models (7B-65B to 288B parameters), their architectures (including native multimodal and Mixtureof-Experts variants), and key performance characteristics. We then describe and discuss the concept of PEFT, which adapts large pre-trained models by updating only a small subset of parameters, and review five PEFT methods that have been applied to LLaMA: LoRA (Low-Rank Adaptation), LLaMA-Adapter V1 and V2, LLaMA-Excitor, and QLoRA (Quantized LoRA). We discuss each method's mechanism, parameter savings, and example application to LLaMA (e.g., instruction tuning, multimodal tasks). We provide structured discussion and analysis of model and adapter architectures, parameter counts, and benchmark results (including examples where fine-tuned LLaMA models outperform larger baselines). Finally, we examine real-world use cases where LLaMA-based models and PEFT have been successfully applied (e.g., legal and medical domains), and we discuss ongoing challenges and future research directions (such as scaling to even larger contexts and improving robustness). This survey paper provides a one-stop resource for ML researchers and practitioners interested in LLaMA models and efficient fine-tuning strategies. 

**Abstract (ZH)**: Meta AI的LLaMA系列的快速演进及其参数高效微调方法：从LLaMA 1到LLaMA 4及专为这些模型开发的PEFT方法综述 

---
# MatSciBench: Benchmarking the Reasoning Ability of Large Language Models in Materials Science 

**Title (ZH)**: MatSciBench: 评估大型语言模型在材料科学中的推理能力 

**Authors**: Junkai Zhang, Jingru Gan, Xiaoxuan Wang, Zian Jia, Changquan Gu, Jianpeng Chen, Yanqiao Zhu, Mingyu Derek Ma, Dawei Zhou, Ling Li, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12171)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable abilities in scientific reasoning, yet their reasoning capabilities in materials science remain underexplored. To fill this gap, we introduce MatSciBench, a comprehensive college-level benchmark comprising 1,340 problems that span the essential subdisciplines of materials science. MatSciBench features a structured and fine-grained taxonomy that categorizes materials science questions into 6 primary fields and 31 sub-fields, and includes a three-tier difficulty classification based on the reasoning length required to solve each question. MatSciBench provides detailed reference solutions enabling precise error analysis and incorporates multimodal reasoning through visual contexts in numerous questions. Evaluations of leading models reveal that even the highest-performing model, Gemini-2.5-Pro, achieves under 80% accuracy on college-level materials science questions, highlighting the complexity of MatSciBench. Our systematic analysis of different reasoning strategie--basic chain-of-thought, tool augmentation, and self-correction--demonstrates that no single method consistently excels across all scenarios. We further analyze performance by difficulty level, examine trade-offs between efficiency and accuracy, highlight the challenges inherent in multimodal reasoning tasks, analyze failure modes across LLMs and reasoning methods, and evaluate the influence of retrieval-augmented generation. MatSciBench thus establishes a comprehensive and solid benchmark for assessing and driving improvements in the scientific reasoning capabilities of LLMs within the materials science domain. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科学推理方面展现了卓越的能力，但在材料科学中的推理能力still有待深入探索。为了填补这一空白，我们引入了MatSciBench，这是一个全面的大学水平基准，包含1340个覆盖材料科学主要亚学科的问题。MatSciBench具有一套结构化和精细的分类体系，将材料科学问题分为6个主要领域和31个子领域，并根据解决问题所需的推理长度进行了三级难度分类。MatSciBench提供了详细的标准解决方案，便于精确的误差分析，并通过多个问题中的视觉上下文实现了多模态推理。对领先模型的评估显示，即使表现最好的模型Gemini-2.5-Pro，在解决大学水平的材料科学问题方面的准确率也低于80%，突显了MatSciBench的复杂性。我们的系统分析表明，不同的推理策略——基本链式思考、工具增强和自我修正——在所有情境中均未表现出一致性优势。我们还按难度级别分析了性能，探讨了效率与准确性的权衡，指出了多模态推理任务的固有挑战，分析了不同大型语言模型和推理方法的失败模式，并评估了检索增强生成的影响。因此，MatSciBench为评估和推动材料科学领域大型语言模型科学推理能力的提升提供了一个全面而坚实的标准。 

---
# Precise Attribute Intensity Control in Large Language Models via Targeted Representation Editing 

**Title (ZH)**: 大型语言模型中目标表示编辑实现精确属性强度控制 

**Authors**: Rongzhi Zhang, Liqin Ye, Yuzhao Heng, Xiang Chen, Tong Yu, Lingkai Kong, Sudheer Chava, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12121)  

**Abstract**: Precise attribute intensity control--generating Large Language Model (LLM) outputs with specific, user-defined attribute intensities--is crucial for AI systems adaptable to diverse user expectations. Current LLM alignment methods, however, typically provide only directional or open-ended guidance, failing to reliably achieve exact attribute intensities. We address this limitation with three key designs: (1) reformulating precise attribute intensity control as a target-reaching problem, rather than simple maximization; (2) training a lightweight value function via temporal-difference learning to predict final attribute intensity scores from partial generations, thereby steering LLM outputs; and (3) employing gradient-based interventions on hidden representations to navigate the model precisely towards specific attribute intensity targets. Our method enables fine-grained, continuous control over attribute intensities, moving beyond simple directional alignment. Experiments on LLaMA-3.2-3b and Phi-4-mini confirm our method's ability to steer text generation to user-specified attribute intensities with high accuracy. Finally, we demonstrate efficiency enhancements across three downstream tasks: preference data synthesis, Pareto frontier approximation and optimization, and distillation of aligned behaviors for intervention-free inference. Our code is available on this https URL 

**Abstract (ZH)**: 精确属性强度控制——根据用户定义的具体属性强度生成大型语言模型（LLM）输出——对于适应多样化用户期望的AI系统至关重要。然而，当前的LLM对齐方法通常只能提供方向性或开放性的指导，无法可靠地实现精确的属性强度。我们通过以下三种关键设计解决了这一局限性：（1）将精确属性强度控制重新表述为一个目标抵达问题，而不是简单的最大化；（2）通过时差学习训练一个轻量级价值函数，以预测最终属性强度分数，从而引导LLM输出；（3）使用基于梯度的干预手段作用于隐藏表示，以精确引导模型朝向特定属性强度目标。我们的方法使属性强度控制具有细粒度和连续性，超越了简单的方向对齐。在对LLaMA-3.2-3b和Phi-4-mini进行的实验中，验证了我们的方法能够以高精度引导文本生成至用户指定的属性强度。最后，我们展示了在三个下游任务中的效率提升：偏好数据合成、帕累托前沿逼近与优化以及干预自由推理的对齐行为蒸馏。 

---
# ToPolyAgent: AI Agents for Coarse-Grained Topological Polymer Simulations 

**Title (ZH)**: ToPolyAgent: AI代理进行粗粒化拓扑聚合物模拟 

**Authors**: Lijie Ding, Jan-Michael Carrillo, Changwoo Do  

**Link**: [PDF](https://arxiv.org/pdf/2510.12091)  

**Abstract**: We introduce ToPolyAgent, a multi-agent AI framework for performing coarse-grained molecular dynamics (MD) simulations of topological polymers through natural language instructions. By integrating large language models (LLMs) with domain-specific computational tools, ToPolyAgent supports both interactive and autonomous simulation workflows across diverse polymer architectures, including linear, ring, brush, and star polymers, as well as dendrimers. The system consists of four LLM-powered agents: a Config Agent for generating initial polymer-solvent configurations, a Simulation Agent for executing LAMMPS-based MD simulations and conformational analyses, a Report Agent for compiling markdown reports, and a Workflow Agent for streamlined autonomous operations. Interactive mode incorporates user feedback loops for iterative refinements, while autonomous mode enables end-to-end task execution from detailed prompts. We demonstrate ToPolyAgent's versatility through case studies involving diverse polymer architectures under varying solvent condition, thermostats, and simulation lengths. Furthermore, we highlight its potential as a research assistant by directing it to investigate the effect of interaction parameters on the linear polymer conformation, and the influence of grafting density on the persistence length of the brush polymer. By coupling natural language interfaces with rigorous simulation tools, ToPolyAgent lowers barriers to complex computational workflows and advances AI-driven materials discovery in polymer science. It lays the foundation for autonomous and extensible multi-agent scientific research ecosystems. 

**Abstract (ZH)**: ToPolyAgent：一种通过自然语言指令进行拓扑聚合物粗粒度分子动力学模拟的多智能体AI框架 

---
# One Life to Learn: Inferring Symbolic World Models for Stochastic Environments from Unguided Exploration 

**Title (ZH)**: 一生学习：从未经指导的探索中推断 stochastic 环境的符号世界模型 

**Authors**: Zaid Khan, Archiki Prasad, Elias Stengel-Eskin, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2510.12088)  

**Abstract**: Symbolic world modeling requires inferring and representing an environment's transitional dynamics as an executable program. Prior work has focused on largely deterministic environments with abundant interaction data, simple mechanics, and human guidance. We address a more realistic and challenging setting, learning in a complex, stochastic environment where the agent has only "one life" to explore a hostile environment without human guidance. We introduce OneLife, a framework that models world dynamics through conditionally-activated programmatic laws within a probabilistic programming framework. Each law operates through a precondition-effect structure, activating in relevant world states. This creates a dynamic computation graph that routes inference and optimization only through relevant laws, avoiding scaling challenges when all laws contribute to predictions about a complex, hierarchical state, and enabling the learning of stochastic dynamics even with sparse rule activation. To evaluate our approach under these demanding constraints, we introduce a new evaluation protocol that measures (a) state ranking, the ability to distinguish plausible future states from implausible ones, and (b) state fidelity, the ability to generate future states that closely resemble reality. We develop and evaluate our framework on Crafter-OO, our reimplementation of the Crafter environment that exposes a structured, object-oriented symbolic state and a pure transition function that operates on that state alone. OneLife can successfully learn key environment dynamics from minimal, unguided interaction, outperforming a strong baseline on 16 out of 23 scenarios tested. We also test OneLife's planning ability, with simulated rollouts successfully identifying superior strategies. Our work establishes a foundation for autonomously constructing programmatic world models of unknown, complex environments. 

**Abstract (ZH)**: Symbolic 世界建模要求推断和表示环境的转换动力学为可执行程序。先前的工作主要集中在确定性较强的环境，这些环境有大量的交互数据、简单的物理机制，并且有人类的指导。我们解决了一个更具现实意义且更具挑战性的场景，在没有人类指导的情况下，仅凭“一次生命”探索一个敌对环境中的复杂、随机环境。我们提出了 OneLife 框架，通过概率编程框架中的条件激活程序法侓来建模世界动力学。每条法侓通过前提-效果结构运作，在相关世界状态中激活，从而创建一个动态计算图，仅通过相关法侓进行推理和优化，避免了所有法侓共同预测复杂层次状态时的扩展挑战，即使法侓激活稀疏，也能学习随机动力学。为了在这些苛刻的条件下评估我们的方法，我们提出了一种新的评估协议，衡量其在(a)状态排名方面的能力，即区分可能的未来状态与不可能的状态的能力，以及(b)状态保真度方面的能力，即生成与现实高度相似的未来状态的能力。我们在 Crafter-OO 上开发并评估了我们的框架，这是我们对 Crafter 环境的重新实现，该环境暴露了一个结构化的面向对象符号状态以及一个仅作用于此状态的纯粹转换函数。即使在最少的无指导交互下，OneLife 也能成功学习环境的关键动力学，且在测试的 23 种场景中有 16 种表现优于一个强有力的基线。我们还测试了 OneLife 的规划能力，模拟滚出成功识别出更优策略。我们的工作为自主构建未知复杂环境的程序化世界模型奠定了基础。 

---
# Evaluating the Quality of Randomness and Entropy in Tasks Supported by Large Language Models 

**Title (ZH)**: 评估大型语言模型支持的任务中随机性和熵的质量 

**Authors**: Rabimba Karanjai, Yang Lu, Ranjith Chodavarapu, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12080)  

**Abstract**: The rapid advancement of large language model (LLM) technology has led to diverse applications, many of which inherently require randomness, such as stochastic decision-making, gaming, scheduling, AI agents, and cryptography-related tasks. However, the capabilities of LLMs in handling randomness, particularly in generating and utilizing random numbers effectively, remain unclear. This paper investigates the capacity of LLMs for handling tasks that involve randomness through a series of experiments. We designed a set of experiments that consider various factors that can influence an LLM's performance in tasks involving randomness, such as accessibility to external tools, types of tasks, model states (fresh vs. non-fresh), and prompting strategies. The experiments cover a range of tasks, including generating random numbers, generating random strings such as passwords, shuffling items, and evaluating the quality of randomness using entropy and the NIST randomness test-suite. Our findings reveal that while LLMs can generate outputs that exhibit some degree of randomness, their performance is inconsistent and often deviates significantly from the expected behavior. The analysis of the experimental results highlights key limitations and areas where improvement is needed for the LLMs to effectively handle tasks involving randomness 

**Abstract (ZH)**: 大规模语言模型（LLM）技术的迅速发展带来了多样化的应用，其中许多应用本质上需要随机性，如随机决策、游戏、调度、AI代理和加密相关任务。然而，LLMs在处理随机性方面的能力，特别是在生成和有效利用随机数方面的能力仍然不够清晰。本文通过一系列实验研究了LLMs处理涉及随机性的任务的能力。我们设计了一系列实验，考虑了可能影响LLMs在涉及随机性任务中表现的各种因素，如对外部工具的访问性、任务类型、模型状态（新鲜模型 vs. 非新鲜模型）和提示策略。实验涵盖了生成随机数、生成随机字符串（如密码）、打乱项目以及使用熵和NIST随机性测试套件评估随机性质量等一系列任务。我们的研究发现，尽管LLMs能够生成具有一定随机性的输出，但其表现不一，往往与预期行为有显著偏差。实验结果的分析揭示了LLMs在有效处理涉及随机性的任务方面存在的关键局限性和需要改进的领域。 

---
# BeSTAD: Behavior-Aware Spatio-Temporal Anomaly Detection for Human Mobility Data 

**Title (ZH)**: 基于行为感知的空间时间异常检测：Human Mobility Data中的BeSTAD 

**Authors**: Junyi Xie, Jina Kim, Yao-Yi Chiang, Lingyi Zhao, Khurram Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2510.12076)  

**Abstract**: Traditional anomaly detection in human mobility has primarily focused on trajectory-level analysis, identifying statistical outliers or spatiotemporal inconsistencies across aggregated movement traces. However, detecting individual-level anomalies, i.e., unusual deviations in a person's mobility behavior relative to their own historical patterns, within datasets encompassing large populations remains a significant challenge. In this paper, we present BeSTAD (Behavior-aware Spatio-Temporal Anomaly Detection for Human Mobility Data), an unsupervised framework that captures individualized behavioral signatures across large populations and uncovers fine-grained anomalies by jointly modeling spatial context and temporal dynamics. BeSTAD learns semantically enriched mobility representations that integrate location meaning and temporal patterns, enabling the detection of subtle deviations in individual movement behavior. BeSTAD further employs a behavior-cluster-aware modeling mechanism that builds personalized behavioral profiles from normal activity and identifies anomalies through cross-period behavioral comparison with consistent semantic alignment. Building on prior work in mobility behavior clustering, this approach enables not only the detection of behavioral shifts and deviations from established routines but also the identification of individuals exhibiting such changes within large-scale mobility datasets. By learning individual behaviors directly from unlabeled data, BeSTAD advances anomaly detection toward personalized and interpretable mobility analysis. 

**Abstract (ZH)**: 行为导向的时空异常检测框架：人类移动数据中的BeSTAD 

---
# EmboMatrix: A Scalable Training-Ground for Embodied Decision-Making 

**Title (ZH)**: EmboMatrix: 一种可扩展的体态决策训练平台 

**Authors**: Zixing Lei, Sheng Yin, Yichen Xiong, Yuanzhuo Ding, Wenhao Huang, Yuxi Wei, Qingyao Xu, Yiming Li, Weixin Li, Yunhong Wang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12072)  

**Abstract**: Embodied decision-making enables agents to translate high-level goals into executable actions through continuous interactions within the physical world, forming a cornerstone of general-purpose embodied intelligence. Large language models (LLMs), with their general decision-making capabilities, offer a promising path to realize this potential; however, LLMs trained solely on language lack exposure to physical environments, limiting their true embodied understanding. To bridge this gap, we propose the concept of a training ground: a comprehensive infrastructure that provides task and scene simulation, embodied interaction, and feedback signals, offering a one-stop solution for LLM acquire genuine embodied decision-making skills. In this work, we present EmboMatrix, the first training ground of its kind, providing massive and diverse tasks with efficient simulation and precise rewards. EmboMatrix incorporates a series of novel techniques: a multi-agent data engine for large-scale task and scene generation, a distributed heterogeneous-hardware system for scalable simulation, and a multi-level reward architecture for precise supervision. Leveraging EmboMatrix, we cultivate EmboBrain, an LLM whose embodied decision-making abilities emerge from extensive embodied interactions. Experiments show that EmboBrain-7B surpasses the 671B DeepSeek-R1 baseline by 9.5\% on two challenging embodied decision-making benchmarks, demonstrating the power of interactive, environment-grounded learning for building truly intelligent embodied agents. 

**Abstract (ZH)**: 具身决策使智能体能够通过与物理世界的持续交互将高层次目标转化为可执行动作，构成通用具身智能的核心。大规模语言模型（LLMs）凭借其广泛的决策能力为实现这一潜力提供了前景；然而，仅基于语言训练的LLMs缺乏对物理环境的暴露，限制了它们的真正具身理解。为了弥合这一差距，我们提出了训练场的概念：一个全面的基础设施，提供任务和场景模拟、具身交互和反馈信号，为LLM提供一站式解决方案以获得真实的具身决策能力。在本工作中，我们介绍了EmboMatrix，这是首个此类训练场，提供了大规模多样任务的高效模拟和精确奖励。EmboMatrix 集成了多项新技术：大规模任务和场景生成的多智能体数据引擎、可扩展模拟的分布式异构硬件系统以及多层次奖励架构以实现精确监督。借助EmboMatrix，我们培育了EmboBrain，一种具有广泛具身决策能力的LLM，这些能力源自大量具身交互。实验表明，EmboBrain-7B在两个具身决策基准测试上分别超越了671B DeepSeek-R1基线9.5%，证明了交互式、环境导向学习对于构建真正智能的具身代理的重要性。 

---
# HiCoTraj:Zero-Shot Demographic Reasoning via Hierarchical Chain-of-Thought Prompting from Trajectory 

**Title (ZH)**: HiCoTraj：基于轨迹的层次链式思考零样本 demographic 推理 

**Authors**: Junyi Xie, Yuankun Jiao, Jina Kim, Yao-Yi Chiang, Lingyi Zhao, Khurram Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2510.12067)  

**Abstract**: Inferring demographic attributes such as age, sex, or income level from human mobility patterns enables critical applications such as targeted public health interventions, equitable urban planning, and personalized transportation services. Existing mobility-based demographic inference studies heavily rely on large-scale trajectory data with demographic labels, leading to limited interpretability and poor generalizability across different datasets and user groups. We propose HiCoTraj (Zero-Shot Demographic Reasoning via Hierarchical Chain-of-Thought Prompting from Trajectory), a framework that leverages LLMs' zero-shot learning and semantic understanding capabilities to perform demographic inference without labeled training data. HiCoTraj transforms trajectories into semantically rich, natural language representations by creating detailed activity chronicles and multi-scale visiting summaries. Then HiCoTraj uses a novel hierarchical chain of thought reasoning to systematically guide LLMs through three cognitive stages: factual feature extraction, behavioral pattern analysis, and demographic inference with structured output. This approach addresses the scarcity challenge of labeled demographic data while providing transparent reasoning chains. Experimental evaluation on real-world trajectory data demonstrates that HiCoTraj achieves competitive performance across multiple demographic attributes in zero-shot scenarios. 

**Abstract (ZH)**: 从人类移动模式推断人口统计属性（如年龄、性别或收入水平） enables 诸如针对性的公共卫生干预、公平的城市规划和个人化交通服务等关键应用。现有的基于移动性的人口统计推断研究严重依赖带有人口统计标签的大规模轨迹数据，导致解释性和在不同数据集和用户群体中的泛化性有限。我们提出了HiCoTraj（基于轨迹的零样本人口统计推理通过层次链式思考提示），该框架利用大语言模型的零样本学习和语义理解能力，在无需标注训练数据的情况下进行人口统计推断。HiCoTraj 将轨迹转换为语义丰富、自然语言表示的形式，通过创建详细活动年表和多尺度访问摘要。然后，HiCoTraj 使用新颖的层次链式推理进行系统的认知引导，依次经过事实特征提取、行为模式分析和结构化输出的人口统计推断。这种 approach 解决了标注人口统计数据稀缺的问题，同时提供透明的推理链。实验在真实世界轨迹数据上的评估结果表明，HiCoTraj 在零样本场景中实现了多个人口统计属性的竞争力表现。 

---
# AI Agents as Universal Task Solvers 

**Title (ZH)**: AI代理作为通用任务求解器 

**Authors**: Alessandro Achille, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2510.12066)  

**Abstract**: AI reasoning agents are already able to solve a variety of tasks by deploying tools, simulating outcomes of multiple hypotheses and reflecting on them. In doing so, they perform computation, although not in the classical sense -- there is no program being executed. Still, if they perform computation, can AI agents be universal? Can chain-of-thought reasoning solve any computable task? How does an AI Agent learn to reason? Is it a matter of model size? Or training dataset size?
In this work, we reinterpret the role of learning in the context of AI Agents, viewing them as compute-capable stochastic dynamical systems, and highlight the role of time in a foundational principle for learning to reason. In doing so, we propose a shift from classical inductive learning to transductive learning -- where the objective is not to approximate the distribution of past data, but to capture their algorithmic structure to reduce the time needed to find solutions to new tasks.
Transductive learning suggests that, counter to Shannon's theory, a key role of information in learning is about reduction of time rather than reconstruction error. In particular, we show that the optimal speed-up that a universal solver can achieve using past data is tightly related to their algorithmic information. Using this, we show a theoretical derivation for the observed power-law scaling of inference time versus training time. We then show that scaling model size can lead to behaviors that, while improving accuracy on benchmarks, fail any reasonable test of intelligence, let alone super-intelligence: In the limit of infinite space and time, large models can behave as savants, able to brute-force through any task without any insight. Instead, we argue that the key quantity to optimize when scaling reasoning models is time, whose critical role in learning has so far only been indirectly considered. 

**Abstract (ZH)**: 基于AI推理代理的计算能力再诠释：从归纳学习到传递学习 

---
# ThinkPilot: Steering Reasoning Models via Automated Think-prefixes Optimization 

**Title (ZH)**: ThinkPilot: 通过自动化Think前缀优化引导推理模型 

**Authors**: Sunzhu Li, Zhiyu Lin, Shuling Yang, Jiale Zhao, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12063)  

**Abstract**: Large Reasoning Models (LRMs) are powerful, but they still suffer from inefficient and off-target reasoning. Currently, training-free methods are limited to either rigid heuristics or descriptive, non-actionable analyses. In this paper, we introduce ThinkPilot, a training-free framework that automatically optimizes LRMs reasoning. It uses an evolutionary process to generate think-prefixes, which are instructions that evolve driven by a taxonomy of reasoning behaviors to guide models toward superior performance. Extensive experiments demonstrate ThinkPilot's broad effectiveness: it significantly improves the accuracy-length trade-off for efficient reasoning, drastically improves safety (for example, cutting the StrongREJECT score of DeepSeek-R1-Distill-Qwen-32B from 27.0% to 0.7), and enhances instruction following. It also synergizes with existing training-based methods. Our analysis reveals that think-prefixes can reliably control LRMs' reasoning behaviors, and that different tasks have strong preferences for specific behavioral distributions. By automatically identifying and eliciting these behaviors, ThinkPilot provides a generalizable framework for aligning LRMs reasoning with task demands. Data and code are available at this https URL 

**Abstract (ZH)**: 无需翻译标题，以下是翻译后的内容：

大型推理模型（LRMs）非常强大，但仍存在低效和偏离目标的推理问题。目前，无需训练的方法要么局限于刚性的启发式规则，要么是描述性的、不可行的分析。本文引入了ThinkPilot，这是一种无需训练的框架，可以自动优化LRMs的推理。它使用进化过程生成think-prefixes，这些指示符根据推理行为的分类学发展，以引导模型向更优的性能方向发展。广泛的实验证明ThinkPilot的有效性：它显著改善了高效推理的准确性和长度之间的权衡，大幅提高了安全性（例如，将DeepSeek-R1-Distill-Qwen-32B的StrongREJECT得分从27.0%降至0.7），增强了指令遵循，并与现有的基于训练的方法协同工作。我们的分析表明，think-prefixes可以可靠地控制LRMs的推理行为，不同的任务对特定行为分布有强烈偏好。通过自动识别并激发这些行为，ThinkPilot提供了一种泛化的框架，以使LRMs的推理与任务需求相一致。相关数据和代码可在以下链接获取：this https URL 

---
# Empowering LLM Agents with Geospatial Awareness: Toward Grounded Reasoning for Wildfire Response 

**Title (ZH)**: 增强LLM代理的地理空间意识：面向野火响应的接地推理 

**Authors**: Yiheng Chen, Lingyao Li, Zihui Ma, Qikai Hu, Yilun Zhu, Min Deng, Runlong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12061)  

**Abstract**: Effective disaster response is essential for safeguarding lives and property. Existing statistical approaches often lack semantic context, generalize poorly across events, and offer limited interpretability. While Large language models (LLMs) provide few-shot generalization, they remain text-bound and blind to geography. To bridge this gap, we introduce a Geospatial Awareness Layer (GAL) that grounds LLM agents in structured earth data. Starting from raw wildfire detections, GAL automatically retrieves and integrates infrastructure, demographic, terrain, and weather information from external geodatabases, assembling them into a concise, unit-annotated perception script. This enriched context enables agents to produce evidence-based resource-allocation recommendations (e.g., personnel assignments, budget allocations), further reinforced by historical analogs and daily change signals for incremental updates. We evaluate the framework in real wildfire scenarios across multiple LLM models, showing that geospatially grounded agents can outperform baselines. The proposed framework can generalize to other hazards such as floods and hurricanes. 

**Abstract (ZH)**: 有效的灾害响应对于保护生命和财产至关重要。现有的统计方法往往缺乏语义上下文，跨事件泛化能力差，且缺乏可解释性。虽然大型语言模型（LLMs）提供了少样本泛化能力，但它们仍然局限于文本，对地理信息视而不见。为弥补这一差距，我们提出了一种地理意识层（Geospatial Awareness Layer，GAL），使LLM代理扎根于结构化的地球数据中。从原始的野火检测开始，GAL能够自动检索和集成基础设施、人口统计、地形和气象信息，并将其组装成精炼且带有单位标注的感知脚本。这种丰富的上下文使代理能够生成基于证据的资源分配建议（例如，人员分配、预算分配），并通过历史类比和日常变化信号进行增量更新。我们在多个LLM模型的多个实际野火场景中评估了该框架，结果显示地理扎根代理能够超越基线。所提出的框架可以泛化到其他灾害类型，如洪水和飓风。 

---
# Do Large Language Models Respect Contracts? Evaluating and Enforcing Contract-Adherence in Code Generation 

**Title (ZH)**: 大语言模型遵守协议吗？评估与强制执行代码生成的协议一致性 

**Authors**: Soohan Lim, Joonghyuk Hahn, Hyunwoo Park, Sang-Ki Ko, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.12047)  

**Abstract**: Prevailing code generation benchmarks, such as HumanEval+ and MBPP+, primarily evaluate large language models (LLMs) with pass@k on functional correctness using well-formed inputs. However, they ignore a crucial aspect of real-world software: adherence to contracts-the preconditions and validity constraints that dictate how ill-formed inputs must be rejected. This critical oversight means that existing benchmarks fail to measure, and models consequently fail to generate, truly robust and reliable code snippets. We introduce PACT, a program assessment and contract-adherence evaluation framework, to bridge this gap. PACT is the first framework designed to systematically evaluate and enhance contract-adherence in LLM-generated code snippets alongside functional correctness. PACT's contributions are threefold: First, it provides a comprehensive test-suite corpus focused on contract violations, extending HumanEval+ and MBPP+. Second, it enables a systematic analysis of code generation under varied prompting conditions. This analysis demonstrates that augmenting prompts with contract-violating test cases significantly enhance a model's ability to respect contracts compared to using contract description alone. Finally, it introduces novel metrics to rigorously quantify contract adherence in both test generation and code generation. By revealing critical errors that conventional benchmarks overlook, PACT provides the rigorous and interpretable metrics to evaluate the robustness of LLM-generated code snippets in both functionality and this http URL code and data are available at this https URL. 

**Abstract (ZH)**: Prevailing代码生成基准（如HumanEval+和MBPP+）主要通过规范输入评估大规模语言模型（LLMs）的功能正确性，但忽略了现实世界软件中至关重要的一个方面：合同遵守问题——决定如何拒绝不规范输入的前提条件和有效性约束。这一关键遗漏意味着现有的基准无法衡量模型生成真正 robust 和可靠的代码片段的能力。为此，我们引入了PACT，这是一种程序评估和合同遵守评估框架，旨在弥合这一缺口。PACT是第一个旨在系统评估和增强LLM生成代码片段中合同遵守性的框架，同时保持功能正确性。PACT的贡献包括三个方面：首先，它提供了一个专注于合同违规的全面测试套件库，扩展了HumanEval+和MBPP+。其次，它使在不同提示条件下系统的代码生成分析成为可能。这项分析表明，通过补充包含合同违规测试案例的提示，可以显著提高模型遵守合同的能力，与仅使用合同描述相比。最后，它引入了新的度量标准，以严格量化测试生成和代码生成中的合同遵守情况。通过揭示传统基准所忽略的关键错误，PACT 提供了严格的可解释度量标准来评估LLM生成的代码片段的功能性和合同遵守性。相关代码和数据可在以下链接获取：this https URL。 

---
# CausalTrace: A Neurosymbolic Causal Analysis Agent for Smart Manufacturing 

**Title (ZH)**: 因果追踪：面向智能制造的神经符号因果分析代理 

**Authors**: Chathurangi Shyalika, Aryaman Sharma, Fadi El Kalach, Utkarshani Jaimini, Cory Henson, Ramy Harik, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2510.12033)  

**Abstract**: Modern manufacturing environments demand not only accurate predictions but also interpretable insights to process anomalies, root causes, and potential interventions. Existing AI systems often function as isolated black boxes, lacking the seamless integration of prediction, explanation, and causal reasoning required for a unified decision-support solution. This fragmentation limits their trustworthiness and practical utility in high-stakes industrial environments. In this work, we present CausalTrace, a neurosymbolic causal analysis module integrated into the SmartPilot industrial CoPilot. CausalTrace performs data-driven causal analysis enriched by industrial ontologies and knowledge graphs, including advanced functions such as causal discovery, counterfactual reasoning, and root cause analysis (RCA). It supports real-time operator interaction and is designed to complement existing agents by offering transparent, explainable decision support. We conducted a comprehensive evaluation of CausalTrace using multiple causal assessment methods and the C3AN framework (i.e. Custom, Compact, Composite AI with Neurosymbolic Integration), which spans principles of robustness, intelligence, and trustworthiness. In an academic rocket assembly testbed, CausalTrace achieved substantial agreement with domain experts (ROUGE-1: 0.91 in ontology QA) and strong RCA performance (MAP@3: 94%, PR@2: 97%, MRR: 0.92, Jaccard: 0.92). It also attained 4.59/5 in the C3AN evaluation, demonstrating precision and reliability for live deployment. 

**Abstract (ZH)**: 现代制造环境不仅需要准确的预测，还需要可解释的洞察以处理异常、根本原因及潜在干预措施。现有的AI系统往往作为孤立的黑盒运作，缺乏预测、解释和因果推理的无缝集成，这限制了它们在高风险工业环境中的可信度和实用价值。在本工作中，我们提出了CausalTrace，这是一种集成在SmartPilot工业CoPilot中的神经符号因果分析模块。CausalTrace通过工业本体和知识图谱进行驱动的数据因果分析，包含先进的因果发现、反事实推理和根本原因分析（RCA）功能。它支持实时操作员交互，并旨在通过提供透明的、可解释的决策支持来补充现有代理。我们使用多种因果评估方法和C3AN框架（即神经符号集成的自定义、紧凑和复合AI）全面评估了CausalTrace。在一项学术火箭组装试验台上，CausalTrace在本体QA方面与领域专家达到了0.91的ROUGE-1一致性，并在RCA性能方面表现出色（MAP@3: 94%，PR@2: 97%，MRR: 0.92，Jaccard: 0.92）。它还在C3AN评估中获得了4.59/5的评分，展示了其在实时部署中的精确性和可靠性。 

---
# Asking Clarifying Questions for Preference Elicitation With Large Language Models 

**Title (ZH)**: 使用大型语言模型进行偏好采集的澄清性问题询问方法 

**Authors**: Ali Montazeralghaem, Guy Tennenholtz, Craig Boutilier, Ofer Meshi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12015)  

**Abstract**: Large Language Models (LLMs) have made it possible for recommendation systems to interact with users in open-ended conversational interfaces. In order to personalize LLM responses, it is crucial to elicit user preferences, especially when there is limited user history. One way to get more information is to present clarifying questions to the user. However, generating effective sequential clarifying questions across various domains remains a challenge. To address this, we introduce a novel approach for training LLMs to ask sequential questions that reveal user preferences. Our method follows a two-stage process inspired by diffusion models. Starting from a user profile, the forward process generates clarifying questions to obtain answers and then removes those answers step by step, serving as a way to add ``noise'' to the user profile. The reverse process involves training a model to ``denoise'' the user profile by learning to ask effective clarifying questions. Our results show that our method significantly improves the LLM's proficiency in asking funnel questions and eliciting user preferences effectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）使得推荐系统能够在开放式的对话界面与用户互动成为可能。为了个性化LLM响应，特别是在用户历史有限的情况下，提取用户偏好至关重要。一种获取更多信息的方法是对用户提出澄清性问题。然而，跨不同领域生成有效的序列化澄清性问题仍然具有挑战性。为解决这一问题，我们提出了一种新的方法来训练LLMs以提问能揭示用户偏好的序列化问题。我们的方法采用了一种受到扩散模型启发的两阶段过程。从用户资料开始，前向过程生成澄清性问题以获取答案，然后逐步删除这些答案，这种方式为用户资料添加“噪声”。反向过程涉及训练模型以“去噪”用户资料，学习有效提出澄清性问题。我们的结果表明，我们的方法显著提高了LLM在提问漏斗问题和有效提取用户偏好方面的能力。 

---
# CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research 

**Title (ZH)**: CGBench: 语言模型在临床遗传学研究中科学推理能力的基准测试 

**Authors**: Owen Queen, Harrison G. Zhang, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.11985)  

**Abstract**: Variant and gene interpretation are fundamental to personalized medicine and translational biomedicine. However, traditional approaches are manual and labor-intensive. Generative language models (LMs) can facilitate this process, accelerating the translation of fundamental research into clinically-actionable insights. While existing benchmarks have attempted to quantify the capabilities of LMs for interpreting scientific data, these studies focus on narrow tasks that do not translate to real-world research. To meet these challenges, we introduce CGBench, a robust benchmark that tests reasoning capabilities of LMs on scientific publications. CGBench is built from ClinGen, a resource of expert-curated literature interpretations in clinical genetics. CGBench measures the ability to 1) extract relevant experimental results following precise protocols and guidelines, 2) judge the strength of evidence, and 3) categorize and describe the relevant outcome of experiments. We test 8 different LMs and find that while models show promise, substantial gaps exist in literature interpretation, especially on fine-grained instructions. Reasoning models excel in fine-grained tasks but non-reasoning models are better at high-level interpretations. Finally, we measure LM explanations against human explanations with an LM judge approach, revealing that models often hallucinate or misinterpret results even when correctly classifying evidence. CGBench reveals strengths and weaknesses of LMs for precise interpretation of scientific publications, opening avenues for future research in AI for clinical genetics and science more broadly. 

**Abstract (ZH)**: CGBench：面向临床遗传学的科学出版物推理能力基准 

---
# Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation 

**Title (ZH)**: 整体代理排行榜：AI代理评估的缺失基础设施 

**Authors**: Sayash Kapoor, Benedikt Stroebl, Peter Kirgis, Nitya Nadgir, Zachary S Siegel, Boyi Wei, Tianci Xue, Ziru Chen, Felix Chen, Saiteja Utpala, Franck Ndzomga, Dheeraj Oruganty, Sophie Luskin, Kangheng Liu, Botao Yu, Amit Arora, Dongyoon Hahm, Harsh Trivedi, Huan Sun, Juyong Lee, Tengjun Jin, Yifan Mai, Yifei Zhou, Yuxuan Zhu, Rishi Bommasani, Daniel Kang, Dawn Song, Peter Henderson, Yu Su, Percy Liang, Arvind Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2510.11977)  

**Abstract**: AI agents have been developed for complex real-world tasks from coding to customer service. But AI agent evaluations suffer from many challenges that undermine our understanding of how well agents really work. We introduce the Holistic Agent Leaderboard (HAL) to address these challenges. We make three main contributions. First, we provide a standardized evaluation harness that orchestrates parallel evaluations across hundreds of VMs, reducing evaluation time from weeks to hours while eliminating common implementation bugs. Second, we conduct three-dimensional analysis spanning models, scaffolds, and benchmarks. We validate the harness by conducting 21,730 agent rollouts across 9 models and 9 benchmarks in coding, web navigation, science, and customer service with a total cost of about $40,000. Our analysis reveals surprising insights, such as higher reasoning effort reducing accuracy in the majority of runs. Third, we use LLM-aided log inspection to uncover previously unreported behaviors, such as searching for the benchmark on HuggingFace instead of solving a task, or misusing credit cards in flight booking tasks. We share all agent logs, comprising 2.5B tokens of language model calls, to incentivize further research into agent behavior. By standardizing how the field evaluates agents and addressing common pitfalls in agent evaluation, we hope to shift the focus from agents that ace benchmarks to agents that work reliably in the real world. 

**Abstract (ZH)**: 面向复杂现实任务的AI代理综合评估 leaderboard (HAL)：提高代理性能理解与行为研究 

---
# Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations 

**Title (ZH)**: 超越共识：减轻LLM法官评价中的随和性偏见 

**Authors**: Suryaansh Jain, Umair Z. Ahmed, Shubham Sahai, Ben Leong  

**Link**: [PDF](https://arxiv.org/pdf/2510.11822)  

**Abstract**: New Large Language Models (LLMs) become available every few weeks, and modern application developers confronted with the unenviable task of having to decide if they should switch to a new model. While human evaluation remains the gold standard, it is costly and unscalable. The state-of-the-art approach is to use LLMs as evaluators ( LLM-as-a-judge), but this suffers from a critical flaw: LLMs exhibit a strong positive bias. We provide empirical evidence showing that while LLMs can identify valid outputs with high accuracy (i.e., True Positive Rate 96%), they are remarkably poor at identifying invalid ones (i.e., True Negative Rate <25%). This systematic bias, coupled with class imbalance, often leads to inflated reliability scores.
While ensemble-based methods like majority voting can help, we show that they are not good enough. We introduce an optimal minority-veto strategy that is resilient to missing data and mitigates this bias to a large extent. For scenarios requiring even higher precision, we propose a novel regression-based framework that directly models the validator bias using a small set of human-annotated ground truth data. On a challenging code feedback task over 366 high-school Python programs, our regression approach reduces the maximum absolute error to just 1.2%, achieving a 2x improvement over the best-performing ensemble of 14 state-of-the-art LLMs. 

**Abstract (ZH)**: 新的大规模语言模型（LLMs）每隔几周就会出现，现代应用程序开发者面临不得不决定是否应切换到新模型的艰巨任务。虽然人类评估仍然是黄金标准，但成本高且不具扩展性。当前最先进的方法是使用LLMs作为评估器（LLM-as-a-judge），但这种方法存在一个关键缺陷：LLMs表现出强烈的正向偏见。我们提供了实证证据显示，尽管LLMs可以以高精度（即真正阳性率96%）识别有效的输出，但在识别无效输出方面表现极差（即真正阴性率<25%）。这种系统性偏见，加上类别不平衡，往往会导致可靠性评分被夸大。
虽然传统的聚合方法如多数投票可以有所帮助，但我们证明它们并不足够。我们引入了一种最优的少数票否决策略，能够在很大程度上抵御缺失数据并缓解这一偏见。对于需要更高精度的场景，我们提出了一种新的基于回归的框架，可以直接使用一小部分人工注释的 ground truth 数据来建模验证者的偏见。在一项针对366个高中Python程序的具有挑战性的代码反馈任务中，我们的回归方法将最大绝对误差降低到仅1.2%，相较于14个最先进的LLM的最佳组合，实现了2倍的改进。 

---
# AI Agents for the Dhumbal Card Game: A Comparative Study 

**Title (ZH)**: AI代理在 Dhumbal 卡牌游戏中的应用：一项比较研究 

**Authors**: Sahaj Raj Malla  

**Link**: [PDF](https://arxiv.org/pdf/2510.11736)  

**Abstract**: This study evaluates Artificial Intelligence (AI) agents for Dhumbal, a culturally significant multiplayer card game with imperfect information, through a systematic comparison of rule-based, search-based, and learning-based strategies. We formalize Dhumbal's mechanics and implement diverse agents, including heuristic approaches (Aggressive, Conservative, Balanced, Opportunistic), search-based methods such as Monte Carlo Tree Search (MCTS) and Information Set Monte Carlo Tree Search (ISMCTS), and reinforcement learning approaches including Deep Q-Network (DQN) and Proximal Policy Optimization (PPO), and a random baseline. Evaluation involves within-category tournaments followed by a cross-category championship. Performance is measured via win rate, economic outcome, Jhyap success, cards discarded per round, risk assessment, and decision efficiency. Statistical significance is assessed using Welch's t-test with Bonferroni correction, effect sizes via Cohen's d, and 95% confidence intervals (CI). Across 1024 simulated rounds, the rule-based Aggressive agent achieves the highest win rate (88.3%, 95% CI: [86.3, 90.3]), outperforming ISMCTS (9.0%) and PPO (1.5%) through effective exploitation of Jhyap declarations. The study contributes a reproducible AI framework, insights into heuristic efficacy under partial information, and open-source code, thereby advancing AI research and supporting digital preservation of cultural games. 

**Abstract (ZH)**: 本研究通过系统比较基于规则、基于搜索和基于学习的战略，评估了用于具有不完全信息的文化重要多人纸牌游戏Dhumbal的人工 intelligence (AI) 剂剂。评估涉及类别内锦标赛和跨类别冠军赛，并通过胜率、经济结果、Jhyap 成功、每轮弃牌数量、风险评估和决策效率进行性能衡量。通过使用 Welch 的 t 检验进行 Bonferroni 修正、Cohen 的 d 效应量评估和 95% 的置信区间（CI）来评估统计显著性。在1024轮模拟比赛中，基于规则的Aggressive代理以最高的胜率（88.3%，95% CI: [86.3, 90.3]）胜出，通过对Jhyap声明的有效利用，优于ISMCTS（9.0%）和PPO（1.5%）。本研究贡献了一个可复制的AI框架，关于部分信息下的启发式有效性见解，并提供了开源代码，从而推动了AI研究并支持文化的数字保存。 

---
# DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving 

**Title (ZH)**: DriveVLA-W0：世界模型放大自主驾驶数据的扩展律 

**Authors**: Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, Lu Hou, Lue Fan, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12796)  

**Abstract**: Scaling Vision-Language-Action (VLA) models on large-scale data offers a promising path to achieving a more generalized driving intelligence. However, VLA models are limited by a ``supervision deficit'': the vast model capacity is supervised by sparse, low-dimensional actions, leaving much of their representational power underutilized. To remedy this, we propose \textbf{DriveVLA-W0}, a training paradigm that employs world modeling to predict future images. This task generates a dense, self-supervised signal that compels the model to learn the underlying dynamics of the driving environment. We showcase the paradigm's versatility by instantiating it for two dominant VLA archetypes: an autoregressive world model for VLAs that use discrete visual tokens, and a diffusion world model for those operating on continuous visual features. Building on the rich representations learned from world modeling, we introduce a lightweight action expert to address the inference latency for real-time deployment. Extensive experiments on the NAVSIM v1/v2 benchmark and a 680x larger in-house dataset demonstrate that DriveVLA-W0 significantly outperforms BEV and VLA baselines. Crucially, it amplifies the data scaling law, showing that performance gains accelerate as the training dataset size increases. 

**Abstract (ZH)**: 基于大规模数据的Vision-Language-Action (VLA) 模型缩放为实现更通用的驾驶智能提供了有希望的途径。然而，VLA 模型受到“监督不足”的限制：巨大的模型容量仅通过稀疏的低维度动作进行监督，使得其表示能力大量未被利用。为了解决这一问题，我们提出了DriveVLA-W0训练范式，采用世界建模来预测未来图像。该任务生成了一个稠密的、自监督的信号，迫使模型学习驾驶环境的基本动力学。我们通过实例化DriveVLA-W0来展示其灵活性，将其应用于两种主导的VLA架构：用于使用离散视觉标记的VLA的自回归世界模型，以及用于处理连续视觉特征的扩散世界模型。基于从世界建模中学习到的丰富表示，我们引入了一个轻量级的动作专家，以解决实时时部署中的推理延迟问题。在NAVSIM v1/v2基准测试和一个规模大680倍的内部数据集上进行的大量实验表明，DriveVLA-W0显著优于BEV和VLA基线。更重要的是，它放大了数据缩放定律，表明随着训练数据集规模的增加，性能改善加速。 

---
# CuMPerLay: Learning Cubical Multiparameter Persistence Vectorizations 

**Title (ZH)**: CuMPerLay: 学习立方体多参数持久同调向量化 

**Authors**: Caner Korkmaz, Brighton Nuwagira, Barış Coşkunuzer, Tolga Birdal  

**Link**: [PDF](https://arxiv.org/pdf/2510.12795)  

**Abstract**: We present CuMPerLay, a novel differentiable vectorization layer that enables the integration of Cubical Multiparameter Persistence (CMP) into deep learning pipelines. While CMP presents a natural and powerful way to topologically work with images, its use is hindered by the complexity of multifiltration structures as well as the vectorization of CMP. In face of these challenges, we introduce a new algorithm for vectorizing MP homologies of cubical complexes. Our CuMPerLay decomposes the CMP into a combination of individual, learnable single-parameter persistence, where the bifiltration functions are jointly learned. Thanks to the differentiability, its robust topological feature vectors can be seamlessly used within state-of-the-art architectures such as Swin Transformers. We establish theoretical guarantees for the stability of our vectorization under generalized Wasserstein metrics. Our experiments on benchmark medical imaging and computer vision datasets show the benefit CuMPerLay on classification and segmentation performance, particularly in limited-data scenarios. Overall, CuMPerLay offers a promising direction for integrating global structural information into deep networks for structured image analysis. 

**Abstract (ZH)**: CuMPerLay：一种新型可微向量化层，实现立方体多参数持久同调在深度学习管道中的集成 

---
# UniFusion: Vision-Language Model as Unified Encoder in Image Generation 

**Title (ZH)**: UniFusion: 视觉-语言模型作为图像生成的统一编码器 

**Authors**: Kevin Li, Manuel Brack, Sudeep Katakol, Hareesh Ravi, Ajinkya Kale  

**Link**: [PDF](https://arxiv.org/pdf/2510.12789)  

**Abstract**: Although recent advances in visual generation have been remarkable, most existing architectures still depend on distinct encoders for images and text. This separation constrains diffusion models' ability to perform cross-modal reasoning and knowledge transfer. Prior attempts to bridge this gap often use the last layer information from VLM, employ multiple visual encoders, or train large unified models jointly for text and image generation, which demands substantial computational resources and large-scale data, limiting its this http URL present UniFusion, a diffusion-based generative model conditioned on a frozen large vision-language model (VLM) that serves as a unified multimodal encoder. At the core of UniFusion is the Layerwise Attention Pooling (LAP) mechanism that extracts both high level semantics and low level details from text and visual tokens of a frozen VLM to condition a diffusion generative model. We demonstrate that LAP outperforms other shallow fusion architectures on text-image alignment for generation and faithful transfer of visual information from VLM to the diffusion model which is key for editing. We propose VLM-Enabled Rewriting Injection with Flexibile Inference (VERIFI), which conditions a diffusion transformer (DiT) only on the text tokens generated by the VLM during in-model prompt rewriting. VERIFI combines the alignment of the conditioning distribution with the VLM's reasoning capabilities for increased capabilities and flexibility at inference. In addition, finetuning on editing task not only improves text-image alignment for generation, indicative of cross-modality knowledge transfer, but also exhibits tremendous generalization capabilities. Our model when trained on single image editing, zero-shot generalizes to multiple image references further motivating the unified encoder design of UniFusion. 

**Abstract (ZH)**: UniFusion：基于扩散的条件生成模型，-conditioned on 冻结的大型Vision-Language模型 

---
# MVP4D: Multi-View Portrait Video Diffusion for Animatable 4D Avatars 

**Title (ZH)**: MVP4D：多视角肖像视频扩散生成可动画化的4Davatar 

**Authors**: Felix Taubner, Ruihang Zhang, Mathieu Tuli, Sherwin Bahmani, David B. Lindell  

**Link**: [PDF](https://arxiv.org/pdf/2510.12785)  

**Abstract**: Digital human avatars aim to simulate the dynamic appearance of humans in virtual environments, enabling immersive experiences across gaming, film, virtual reality, and more. However, the conventional process for creating and animating photorealistic human avatars is expensive and time-consuming, requiring large camera capture rigs and significant manual effort from professional 3D artists. With the advent of capable image and video generation models, recent methods enable automatic rendering of realistic animated avatars from a single casually captured reference image of a target subject. While these techniques significantly lower barriers to avatar creation and offer compelling realism, they lack constraints provided by multi-view information or an explicit 3D representation. So, image quality and realism degrade when rendered from viewpoints that deviate strongly from the reference image. Here, we build a video model that generates animatable multi-view videos of digital humans based on a single reference image and target expressions. Our model, MVP4D, is based on a state-of-the-art pre-trained video diffusion model and generates hundreds of frames simultaneously from viewpoints varying by up to 360 degrees around a target subject. We show how to distill the outputs of this model into a 4D avatar that can be rendered in real-time. Our approach significantly improves the realism, temporal consistency, and 3D consistency of generated avatars compared to previous methods. 

**Abstract (ZH)**: 基于单张参考图像和目标表情生成可动画化多视角视频的数字人类视频模型 

---
# Dr.LLM: Dynamic Layer Routing in LLMs 

**Title (ZH)**: Dr.LLM: LLMs中的动态层路由 

**Authors**: Ahmed Heakl, Martin Gubri, Salman Khan, Sangdoo Yun, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.12773)  

**Abstract**: Large Language Models (LLMs) process every token through all layers of a transformer stack, causing wasted computation on simple queries and insufficient flexibility for harder ones that need deeper reasoning. Adaptive-depth methods can improve efficiency, but prior approaches rely on costly inference-time search, architectural changes, or large-scale retraining, and in practice often degrade accuracy despite efficiency gains. We introduce this http URL, Dynamic routing of Layers for LLMs, a retrofittable framework that equips pretrained models with lightweight per-layer routers deciding to skip, execute, or repeat a block. Routers are trained with explicit supervision: using Monte Carlo Tree Search (MCTS), we derive high-quality layer configurations that preserve or improve accuracy under a compute budget. Our design, windowed pooling for stable routing, focal loss with class balancing, and bottleneck MLP routers, ensures robustness under class imbalance and long sequences. On ARC (logic) and DART (math), this http URL improves accuracy by up to +3.4%p while saving 5 layers per example on average. Routers generalize to out-of-domain tasks (MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA, AGIEval) with only 0.85% accuracy drop while retaining efficiency, and outperform prior routing methods by up to +7.7%p. Overall, this http URL shows that explicitly supervised routers retrofit frozen LLMs for budget-aware, accuracy-driven inference without altering base weights. 

**Abstract (ZH)**: 动态路由层：为LLMs配备可微路由框架以实现预算意识下的高效推理 

---
# Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction 

**Title (ZH)**: 不确定性在单目4D重建的动态高斯散列中的重要性 

**Authors**: Fengzhi Guo, Chih-Chuan Hsu, Sihao Ding, Cheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12768)  

**Abstract**: Reconstructing dynamic 3D scenes from monocular input is fundamentally under-constrained, with ambiguities arising from occlusion and extreme novel views. While dynamic Gaussian Splatting offers an efficient representation, vanilla models optimize all Gaussian primitives uniformly, ignoring whether they are well or poorly observed. This limitation leads to motion drifts under occlusion and degraded synthesis when extrapolating to unseen views. We argue that uncertainty matters: Gaussians with recurring observations across views and time act as reliable anchors to guide motion, whereas those with limited visibility are treated as less reliable. To this end, we introduce USplat4D, a novel Uncertainty-aware dynamic Gaussian Splatting framework that propagates reliable motion cues to enhance 4D reconstruction. Our key insight is to estimate time-varying per-Gaussian uncertainty and leverages it to construct a spatio-temporal graph for uncertainty-aware optimization. Experiments on diverse real and synthetic datasets show that explicitly modeling uncertainty consistently improves dynamic Gaussian Splatting models, yielding more stable geometry under occlusion and high-quality synthesis at extreme viewpoints. 

**Abstract (ZH)**: 基于单目输入重构动态3D场景从根本上说是欠约束的， occlusion和极端新颖视角会导致歧义。尽管动态Gaussian Splatting 提供了高效的表示，但vanilla模型会均匀优化所有Gaussian原语，忽视它们的观测质量。这种限制会导致遮挡下的运动漂移以及在预测未见视角时的合成退化。我们认为不确定性很重要：在不同视角和时间上反复出现观测的Gaussian充当可靠的锚点以引导运动，而那些观测有限的Gaussian则被认为是不太可靠的。为此，我们提出了USplat4D，一种新的不确定性感知动态Gaussian Splatting框架，通过传播可靠的运动线索来增强4D重构。我们的核心见解是估计每Gaussian的时间变化不确定性，并利用它构建时空图以进行不确定性感知优化。在多种真实和合成数据集上的实验表明，明确建模不确定性可以一致地改善动态Gaussian Splatting模型，在遮挡下的几何结构更稳定，并且在极端视角下合成质量更高。 

---
# Disentangling Neurodegeneration with Brain Age Gap Prediction Models: A Graph Signal Processing Perspective 

**Title (ZH)**: 基于图形信号处理视角的脑年龄差距预测模型解构神经退行性病变 

**Authors**: Saurabh Sihag, Gonzalo Mateos, Alejandro Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2510.12763)  

**Abstract**: Neurodegeneration, characterized by the progressive loss of neuronal structure or function, is commonly assessed in clinical practice through reductions in cortical thickness or brain volume, as visualized by structural MRI. While informative, these conventional approaches lack the statistical sophistication required to fully capture the spatially correlated and heterogeneous nature of neurodegeneration, which manifests both in healthy aging and in neurological disorders. To address these limitations, brain age gap has emerged as a promising data-driven biomarker of brain health. The brain age gap prediction (BAGP) models estimate the difference between a person's predicted brain age from neuroimaging data and their chronological age. The resulting brain age gap serves as a compact biomarker of brain health, with recent studies demonstrating its predictive utility for disease progression and severity. However, practical adoption of BAGP models is hindered by their methodological obscurities and limited generalizability across diverse clinical populations. This tutorial article provides an overview of BAGP and introduces a principled framework for this application based on recent advancements in graph signal processing (GSP). In particular, we focus on graph neural networks (GNNs) and introduce the coVariance neural network (VNN), which leverages the anatomical covariance matrices derived from structural MRI. VNNs offer strong theoretical grounding and operational interpretability, enabling robust estimation of brain age gap predictions. By integrating perspectives from GSP, machine learning, and network neuroscience, this work clarifies the path forward for reliable and interpretable BAGP models and outlines future research directions in personalized medicine. 

**Abstract (ZH)**: 基于脑网络图信号处理的脑龄差距预测：原理与前景 

---
# VQArt-Bench: A semantically rich VQA Benchmark for Art and Cultural Heritage 

**Title (ZH)**: VQArt-Bench：一个富含语义的艺术和文化遗产VQA基准库 

**Authors**: A. Alfarano, L. Venturoli, D. Negueruela del Castillo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12750)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant capabilities in joint visual and linguistic tasks. However, existing Visual Question Answering (VQA) benchmarks often fail to evaluate deep semantic understanding, particularly in complex domains like visual art analysis. Confined to simple syntactic structures and surface-level attributes, these questions fail to capture the diversity and depth of human visual inquiry. This limitation incentivizes models to exploit statistical shortcuts rather than engage in visual reasoning. To address this gap, we introduce VQArt-Bench, a new, large-scale VQA benchmark for the cultural heritage domain. This benchmark is constructed using a novel multi-agent pipeline where specialized agents collaborate to generate nuanced, validated, and linguistically diverse questions. The resulting benchmark is structured along relevant visual understanding dimensions that probe a model's ability to interpret symbolic meaning, narratives, and complex visual relationships. Our evaluation of 14 state-of-the-art MLLMs on this benchmark reveals significant limitations in current models, including a surprising weakness in simple counting tasks and a clear performance gap between proprietary and open-source models. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在联合视觉和语言任务中展现了显著的能力。然而，现有的视觉问答（VQA）基准通常未能评估深层语义理解，特别是在视觉艺术分析等复杂领域。受限于简单的句法结构和表面属性，这些问题无法捕捉人类视觉探究的多样性和深度。这一限制促使模型利用统计捷径而非参与视觉推理。为了弥补这一差距，我们引入了VQArt-Bench，这是一个新的大型跨领域VQA基准，特别针对文化遗产领域。该基准通过一个新颖的多代理流程构建，其中专门的代理协作生成细腻的、验证过的、语言多样的问题。所得基准按相关的视觉理解维度结构化，旨在测试模型解释符号意义、叙事和复杂视觉关系的能力。我们将14个最先进的MLLMs在该基准上的评估揭示了当前模型存在显著局限，包括令人惊讶的简单计数任务中的薄弱环节以及专有模型与开源模型之间的明显性能差距。 

---
# Hey, wait a minute: on at-issue sensitivity in Language Models 

**Title (ZH)**: 嘿，请等一下：关于语言模型的议题相关敏感性 

**Authors**: Sanghee J. Kim, Kanishka Misra  

**Link**: [PDF](https://arxiv.org/pdf/2510.12740)  

**Abstract**: Evaluating the naturalness of dialogue in language models (LMs) is not trivial: notions of 'naturalness' vary, and scalable quantitative metrics remain limited. This study leverages the linguistic notion of 'at-issueness' to assess dialogue naturalness and introduces a new method: Divide, Generate, Recombine, and Compare (DGRC). DGRC (i) divides a dialogue as a prompt, (ii) generates continuations for subparts using LMs, (iii) recombines the dialogue and continuations, and (iv) compares the likelihoods of the recombined sequences. This approach mitigates bias in linguistic analyses of LMs and enables systematic testing of discourse-sensitive behavior. Applying DGRC, we find that LMs prefer to continue dialogue on at-issue content, with this effect enhanced in instruct-tuned models. They also reduce their at-issue preference when relevant cues (e.g., "Hey, wait a minute") are present. Although instruct-tuning does not further amplify this modulation, the pattern reflects a hallmark of successful dialogue dynamics. 

**Abstract (ZH)**: 评估语言模型对话的自然度并非易事：自然度的定义各不相同，且可扩展的量化指标仍然有限。本研究利用语义上的“相关性”概念来评估对话的自然度，并引入了一种新方法：分割、生成、重组和对比（DGRC）。DGRC 方法包括：(i) 将对话作为提示进行分割，(ii) 使用语言模型生成子部分的续作，(iii) 重组对话与续作，(iv) 对重组序列的可能性进行比较。该方法减少了语言模型语言分析中的偏见，并使对话敏感行为的系统测试成为可能。应用DGRC，我们发现语言模型倾向于在与议题相关的内容上继续对话，这种效应在指令微调模型中更为显著。当相关提示（例如，“等等，稍等一下”）存在时，它们也会减少对与议题相关性的偏好。尽管指令微调没有进一步放大这种调节，但这一模式体现了成功对话动态的特征。 

---
# HYPE: Hybrid Planning with Ego Proposal-Conditioned Predictions 

**Title (ZH)**: HYPE: 混合规划与 ego 提案条件下的预测 

**Authors**: Hang Yu, Julian Jordan, Julian Schmidt, Silvan Lindner, Alessandro Canevaro, Wilhelm Stork  

**Link**: [PDF](https://arxiv.org/pdf/2510.12733)  

**Abstract**: Safe and interpretable motion planning in complex urban environments needs to reason about bidirectional multi-agent interactions. This reasoning requires to estimate the costs of potential ego driving maneuvers. Many existing planners generate initial trajectories with sampling-based methods and refine them by optimizing on learned predictions of future environment states, which requires a cost function that encodes the desired vehicle behavior. Designing such a cost function can be very challenging, especially if a wide range of complex urban scenarios has to be considered. We propose HYPE: HYbrid Planning with Ego proposal-conditioned predictions, a planner that integrates multimodal trajectory proposals from a learned proposal model as heuristic priors into a Monte Carlo Tree Search (MCTS) refinement. To model bidirectional interactions, we introduce an ego-conditioned occupancy prediction model, enabling consistent, scene-aware reasoning. Our design significantly simplifies cost function design in refinement by considering proposal-driven guidance, requiring only minimalistic grid-based cost terms. Evaluations on large-scale real-world benchmarks nuPlan and DeepUrban show that HYPE effectively achieves state-of-the-art performance, especially in safety and adaptability. 

**Abstract (ZH)**: 混合规划与 ego 提议条件下的预测：在复杂城市环境中的安全可解释运动规划 

---
# Hierarchical Federated Learning for Crop Yield Prediction in Smart Agricultural Production Systems 

**Title (ZH)**: 智能农业生产系统中作物产量预测的分层联邦学习 

**Authors**: Anas Abouaomar, Mohammed El hanjri, Abdellatif Kobbane, Anis Laouiti, Khalid Nafil  

**Link**: [PDF](https://arxiv.org/pdf/2510.12727)  

**Abstract**: In this paper, we presents a novel hierarchical federated learning architecture specifically designed for smart agricultural production systems and crop yield prediction. Our approach introduces a seasonal subscription mechanism where farms join crop-specific clusters at the beginning of each agricultural season. The proposed three-layer architecture consists of individual smart farms at the client level, crop-specific aggregators at the middle layer, and a global model aggregator at the top level. Within each crop cluster, clients collaboratively train specialized models tailored to specific crop types, which are then aggregated to produce a higher-level global model that integrates knowledge across multiple crops. This hierarchical design enables both local specialization for individual crop types and global generalization across diverse agricultural contexts while preserving data privacy and reducing communication overhead. Experiments demonstrate the effectiveness of the proposed system, showing that local and crop-layer models closely follow actual yield patterns with consistent alignment, significantly outperforming standard machine learning models. The results validate the advantages of hierarchical federated learning in the agricultural context, particularly for scenarios involving heterogeneous farming environments and privacy-sensitive agricultural data. 

**Abstract (ZH)**: 本文提出了一种针对智能农业生产和作物产量预测的新型分层联邦学习架构。我们的方法引入了一个季节性订阅机制，使农场在每个农业季节开始时加入特定作物的集群。所提出的三层架构包括客户端级别的个体智能农场、中间层的作物特定聚合器以及顶层的全球模型聚合器。在每个作物集群内，客户端协作训练专门为特定作物类型定制的专业模型，然后将这些模型聚合以产生更高的全局模型，该模型整合了多种作物的知识。这种分层设计既实现了对个别作物类型的本地专业化，又实现了在各种农业背景下对全局知识的概括，同时保持了数据隐私并减少了通信开销。实验结果证明了所提系统的有效性，显示本地和作物层模型与实际产量模式高度一致并表现出色，显著优于标准机器学习模型。研究结果验证了在农业场景下分层联邦学习的优势，特别是在涉及异质农业环境和敏感农业数据的情况下。 

---
# Artificial intelligence for simplified patient-centered dosimetry in radiopharmaceutical therapies 

**Title (ZH)**: 人工智能简化以患者为中心的放射性药物治疗剂量规划 

**Authors**: Alejandro Lopez-Montes, Fereshteh Yousefirizi, Yizhou Chen, Yazdan Salimi, Robert Seifert, Ali Afshar-Oromieh, Carlos Uribe, Axel Rominger, Habib Zaidi, Arman Rahmim, Kuangyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12714)  

**Abstract**: KEY WORDS: Artificial Intelligence (AI), Theranostics, Dosimetry, Radiopharmaceutical Therapy (RPT), Patient-friendly dosimetry KEY POINTS - The rapid evolution of radiopharmaceutical therapy (RPT) highlights the growing need for personalized and patient-centered dosimetry. - Artificial Intelligence (AI) offers solutions to the key limitations in current dosimetry calculations. - The main advances on AI for simplified dosimetry toward patient-friendly RPT are reviewed. - Future directions on the role of AI in RPT dosimetry are discussed. 

**Abstract (ZH)**: 关键词：人工智能(AI)，诊疗一体化(Theranostics)，剂量学(Dosimetry)，放射性药物治疗(RPT)，患者友好的剂量学

主要内容要点：
- 放射性药物治疗(RPT)的迅速发展突显了个性化的、以患者为中心的剂量学日益增长的需求。
- 人工智能(AI)为当前剂量学计算中的关键限制提供了解决方案。
- 本文回顾了人工智能在简化剂量学以适应患者友好型RPT方面的主要进展。
- 讨论了人工智能在未来RPT剂量学中的作用方向。 

---
# Beyond Seeing: Evaluating Multimodal LLMs on Tool-Enabled Image Perception, Transformation, and Reasoning 

**Title (ZH)**: 超越视觉：基于工具的图像感知、变换与推理多模态LLM评估 

**Authors**: Xingang Guo, Utkarsh Tyagi, Advait Gosai, Paula Vergara, Ernesto Gabriel Hernández Montoya, Chen Bo Calvin Zhang, Bin Hu, Yunzhong He, Bing Liu, Rakshith Sharma Srinivasa  

**Link**: [PDF](https://arxiv.org/pdf/2510.12712)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly applied in real-world scenarios where user-provided images are often imperfect, requiring active image manipulations such as cropping, editing, or enhancement to uncover salient visual cues. Beyond static visual perception, MLLMs must also think with images: dynamically transforming visual content and integrating it with other tools to solve complex tasks. However, this shift from treating vision as passive context to a manipulable cognitive workspace remains underexplored. Most existing benchmarks still follow a think about images paradigm, where images are regarded as static inputs. To address this gap, we introduce IRIS, an Interactive Reasoning with Images and Systems that evaluates MLLMs' ability to perceive, transform, and reason across complex visual-textual tasks under the think with images paradigm. IRIS comprises 1,204 challenging, open-ended vision tasks (603 single-turn, 601 multi-turn) spanning across five diverse domains, each paired with detailed rubrics to enable systematic evaluation. Our evaluation shows that current MLLMs struggle with tasks requiring effective integration of vision and general-purpose tools. Even the strongest model (GPT-5-think) reaches only 18.68% pass rate. We further observe divergent tool-use behaviors, with OpenAI models benefiting from diverse image manipulations while Gemini-2.5-pro shows no improvement. By introducing the first benchmark centered on think with images, IRIS offers critical insights for advancing visual intelligence in MLLMs. 

**Abstract (ZH)**: 交互式图像与系统推理（IRIS）： multimodal large language models在think with images paradigm下的感知、变换与推理能力评估 

---
# Hybrid Explanation-Guided Learning for Transformer-Based Chest X-Ray Diagnosis 

**Title (ZH)**: 基于Transformer的胸部X光诊断的混合解释引导学习 

**Authors**: Shelley Zixin Shu, Haozhe Luo, Alexander Poellinger, Mauricio Reyes  

**Link**: [PDF](https://arxiv.org/pdf/2510.12704)  

**Abstract**: Transformer-based deep learning models have demonstrated exceptional performance in medical imaging by leveraging attention mechanisms for feature representation and interpretability. However, these models are prone to learning spurious correlations, leading to biases and limited generalization. While human-AI attention alignment can mitigate these issues, it often depends on costly manual supervision. In this work, we propose a Hybrid Explanation-Guided Learning (H-EGL) framework that combines self-supervised and human-guided constraints to enhance attention alignment and improve generalization. The self-supervised component of H-EGL leverages class-distinctive attention without relying on restrictive priors, promoting robustness and flexibility. We validate our approach on chest X-ray classification using the Vision Transformer (ViT), where H-EGL outperforms two state-of-the-art Explanation-Guided Learning (EGL) methods, demonstrating superior classification accuracy and generalization capability. Additionally, it produces attention maps that are better aligned with human expertise. 

**Abstract (ZH)**: 基于Transformer的深度学习模型通过利用注意力机制在医学影像领域展示出了卓越的性能，但这些模型容易学习到伪相关性，导致偏差和泛化能力有限。尽管人类-AI注意力对齐可以缓解这些问题，但通常依赖于昂贵的人工监督。在此项工作中，我们提出了一种混合解释引导学习（H-EGL）框架，该框架结合了自我监督和人工引导的约束，以增强注意力对齐并提高泛化能力。H-EGL的自我监督组件利用类特异性注意力，不依赖于严格的先验知识，从而增强鲁棒性和灵活性。我们使用Vision Transformer（ViT）在胸部X光分类任务上验证了该方法，H-EGL在分类准确性和泛化能力上超越了两种最先进的解释引导学习（EGL）方法，并生成了更好地与人类专业知识对齐的注意力图。 

---
# Beyond Postconditions: Can Large Language Models infer Formal Contracts for Automatic Software Verification? 

**Title (ZH)**: 超越后条件：大型语言模型能否推断出形式契约以实现自动软件验证？ 

**Authors**: Cedric Richter, Heike Wehrheim  

**Link**: [PDF](https://arxiv.org/pdf/2510.12702)  

**Abstract**: Automatic software verifiers have become increasingly effective at the task of checking software against (formal) specifications. Yet, their adoption in practice has been hampered by the lack of such specifications in real world code. Large Language Models (LLMs) have shown promise in inferring formal postconditions from natural language hints embedded in code such as function names, comments or documentation. Using the generated postconditions as specifications in a subsequent verification, however, often leads verifiers to suggest invalid inputs, hinting at potential issues that ultimately turn out to be false alarms.
To address this, we revisit the problem of specification inference from natural language in the context of automatic software verification. In the process, we introduce NL2Contract, the task of employing LLMs to translate informal natural language into formal functional contracts, consisting of postconditions as well as preconditions. We introduce metrics to validate and compare different NL2Contract approaches, using soundness, bug discriminative power of the generated contracts and their usability in the context of automatic software verification as key metrics. We evaluate NL2Contract with different LLMs and compare it to the task of postcondition generation nl2postcond. Our evaluation shows that (1) LLMs are generally effective at generating functional contracts sound for all possible inputs, (2) the generated contracts are sufficiently expressive for discriminating buggy from correct behavior, and (3) verifiers supplied with LLM inferred functional contracts produce fewer false alarms than when provided with postconditions alone. Further investigations show that LLM inferred preconditions generally align well with developers intentions which allows us to use automatic software verifiers to catch real-world bugs. 

**Abstract (ZH)**: 自动软件验证器在根据形式规范检查软件方面的有效性不断提高，但在实践中其采用受到实际代码中缺乏形式规范的限制。大型语言模型（LLMs）展示了从嵌入在代码中的自然语言提示（如函数名、注释或文档）中推断形式后条件的潜力。然而，使用生成的后条件作为后续验证的规格通常会导致验证器建议无效输入，暗示潜在的问题最终证明是误报。

为解决这一问题，我们在自动软件验证的背景下重新审视了从自然语言推断规范的问题。在此过程中，我们引入了NL2Contract任务，即将LLMs用于将非正式自然语言翻译为正式的功能合同，包括后条件和前置条件。我们引入了评估和比较不同NL2Contract方法的度量标准，将规范的有效性、生成的规范对错误的区分能力和在自动软件验证中的易用性作为关键度量标准。我们使用不同的LLMs评估了NL2Contract，并将其与后条件生成nl2postcond任务进行了比较。评估结果显示：（1）LLMs通常能有效地生成适用于所有输入的功能合同；（2）生成的合同足以区分错误行为和正确行为；（3）配以LLMs推断的功能合同的验证器产生的误报少于仅提供后条件的情况。进一步的研究表明，LLMs推断的前置条件通常与开发者的意图一致，允许我们使用自动软件验证器捕获实际存在的bug。 

---
# Topological Signatures of ReLU Neural Network Activation Patterns 

**Title (ZH)**: ReLU神经网络激活模式的拓扑特征 

**Authors**: Vicente Bosca, Tatum Rask, Sunia Tanweer, Andrew R. Tawfeek, Branden Stone  

**Link**: [PDF](https://arxiv.org/pdf/2510.12700)  

**Abstract**: This paper explores the topological signatures of ReLU neural network activation patterns. We consider feedforward neural networks with ReLU activation functions and analyze the polytope decomposition of the feature space induced by the network. Mainly, we investigate how the Fiedler partition of the dual graph and show that it appears to correlate with the decision boundary -- in the case of binary classification. Additionally, we compute the homology of the cellular decomposition -- in a regression task -- to draw similar patterns in behavior between the training loss and polyhedral cell-count, as the model is trained. 

**Abstract (ZH)**: 本文探讨了ReLU神经网络激活模式的拓扑特征。我们考虑使用ReLU激活函数的前向神经网络，并分析网络诱导的特征空间的多面体分解。主要研究了对偶图的Fiedler分割与决策边界之间的关联性（在二元分类情况下）。此外，在回归任务中，我们计算了细胞分解的同调性质，以在模型训练过程中训练损失与多面体细胞计数的行为模式之间找出相似之处。 

---
# Generation Space Size: Understanding and Calibrating Open-Endedness of LLM Generations 

**Title (ZH)**: 生成空间大小：理解并校准LLM生成的开放性 

**Authors**: Sunny Yu, Ahmad Jabbar, Robert Hawkins, Dan Jurafsky, Myra Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12699)  

**Abstract**: Different open-ended generation tasks require different degrees of output diversity. However, current LLMs are often miscalibrated. They collapse to overly homogeneous outputs for creative tasks and hallucinate diverse but incorrect responses for factual tasks. We argue that these two failure modes are unified by, and can both be addressed by, the notion of effective generation space size (GSS) -- the set of semantically distinct outputs a model considers for a prompt. We present GSSBench, a task suite of prompt pairs with ground-truth GSS relationships to assess different metrics and understand where models diverge from desired behavior. We find that hallucination detection metrics, particularly EigenScore, consistently outperform standard diversity and uncertainty quantification metrics, while using only model internals, providing interpretable insights into a model's internal task representations. We demonstrate three applications of GSS: (1) detecting prompt ambiguity and predicting clarification questions for better grounding, (2) interpreting overthinking and underthinking in reasoning models, and (3) steering models to expand their generation space to yield high-quality and diverse outputs. 

**Abstract (ZH)**: 不同的开放生成任务需要不同程度的输出多样性。然而，当前的大语言模型常常失配。它们在创造性任务中产生过度同质化的输出，在事实性任务中则虚构多样但错误的响应。我们argue这两种失败模式可以通过有效的生成空间大小（GSS）这一概念来统一和解决——GSS是指模型针对某个提示考虑的语义上不同的输出集合。我们提出了GSSBench，这是一个由具有真实GSS关系的提示对组成的任务套件，用于评估不同的度量标准并理解模型与期望行为之间的差异。我们发现，尤其是EigenScore的幻觉检测度量标准在仅使用模型内部信息的情况下，始终优于标准的多样性和不确定性量化度量标准，提供了对模型内部任务表示可解释的洞察。我们展示了GSS的三种应用：（1）检测提示模糊性并预测澄清问题以提高语义关联，（2）解释推理模型中的过度思考和思考不足，以及（3）引导模型扩大生成空间以产生高质量和多样化的输出。 

---
# Who is a Better Matchmaker? Human vs. Algorithmic Judge Assignment in a High-Stakes Startup Competition 

**Title (ZH)**: 哪个更称职？人力 vs. 算法匹配法官在高 stakes 创业竞赛中的表现 

**Authors**: Sarina Xi, Orelia Pi, Miaomiao Zhang, Becca Xiong, Jacqueline Ng Lane, Nihar B. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2510.12692)  

**Abstract**: There is growing interest in applying artificial intelligence (AI) to automate and support complex decision-making tasks. However, it remains unclear how algorithms compare to human judgment in contexts requiring semantic understanding and domain expertise. We examine this in the context of the judge assignment problem, matching submissions to suitably qualified judges. Specifically, we tackled this problem at the Harvard President's Innovation Challenge, the university's premier venture competition awarding over \$500,000 to student and alumni startups. This represents a real-world environment where high-quality judge assignment is essential. We developed an AI-based judge-assignment algorithm, Hybrid Lexical-Semantic Similarity Ensemble (HLSE), and deployed it at the competition. We then evaluated its performance against human expert assignments using blinded match-quality scores from judges on $309$ judge-venture pairs. Using a Mann-Whitney U statistic based test, we found no statistically significant difference in assignment quality between the two approaches ($AUC=0.48, p=0.40$); on average, algorithmic matches are rated $3.90$ and manual matches $3.94$ on a 5-point scale, where 5 indicates an excellent match. Furthermore, manual assignments that previously required a full week could be automated in several hours by the algorithm during deployment. These results demonstrate that HLSE achieves human-expert-level matching quality while offering greater scalability and efficiency, underscoring the potential of AI-driven solutions to support and enhance human decision-making for judge assignment in high-stakes settings. 

**Abstract (ZH)**: 人工智能在法官分配问题中的应用与人类判断的比较：从哈佛总统创新挑战赛看自动化的可行性和效率 

---
# DiffEM: Learning from Corrupted Data with Diffusion Models via Expectation Maximization 

**Title (ZH)**: DiffEM：通过期望最大化解learn从带噪数据中预订扩散模型的知识 

**Authors**: Danial Hosseintabar, Fan Chen, Giannis Daras, Antonio Torralba, Constantinos Daskalakis  

**Link**: [PDF](https://arxiv.org/pdf/2510.12691)  

**Abstract**: Diffusion models have emerged as powerful generative priors for high-dimensional inverse problems, yet learning them when only corrupted or noisy observations are available remains challenging. In this work, we propose a new method for training diffusion models with Expectation-Maximization (EM) from corrupted data. Our proposed method, DiffEM, utilizes conditional diffusion models to reconstruct clean data from observations in the E-step, and then uses the reconstructed data to refine the conditional diffusion model in the M-step. Theoretically, we provide monotonic convergence guarantees for the DiffEM iteration, assuming appropriate statistical conditions. We demonstrate the effectiveness of our approach through experiments on various image reconstruction tasks. 

**Abstract (ZH)**: 从受污染数据中训练扩散模型的Expectation-Maximization方法：DiffEM在高维逆问题中的应用 

---
# From Delegates to Trustees: How Optimizing for Long-Term Interests Shapes Bias and Alignment in LLM 

**Title (ZH)**: 从代理人到受托人：优化长期利益如何塑造LLM中的偏见与对齐 

**Authors**: Suyash Fulay, Jocelyn Zhu, Michiel Bakker  

**Link**: [PDF](https://arxiv.org/pdf/2510.12689)  

**Abstract**: Large language models (LLMs) have shown promising accuracy in predicting survey responses and policy preferences, which has increased interest in their potential to represent human interests in various domains. Most existing research has focused on behavioral cloning, effectively evaluating how well models reproduce individuals' expressed preferences. Drawing on theories of political representation, we highlight an underexplored design trade-off: whether AI systems should act as delegates, mirroring expressed preferences, or as trustees, exercising judgment about what best serves an individual's interests. This trade-off is closely related to issues of LLM sycophancy, where models can encourage behavior or validate beliefs that may be aligned with a user's short-term preferences, but is detrimental to their long-term interests. Through a series of experiments simulating votes on various policy issues in the U.S. context, we apply a temporal utility framework that weighs short and long-term interests (simulating a trustee role) and compare voting outcomes to behavior-cloning models (simulating a delegate). We find that trustee-style predictions weighted toward long-term interests produce policy decisions that align more closely with expert consensus on well-understood issues, but also show greater bias toward models' default stances on topics lacking clear agreement. These findings reveal a fundamental trade-off in designing AI systems to represent human interests. Delegate models better preserve user autonomy but may diverge from well-supported policy positions, while trustee models can promote welfare on well-understood issues yet risk paternalism and bias on subjective topics. 

**Abstract (ZH)**: 大型语言模型（LLMs）在预测调查响应和政策偏好方面展示了有前途的准确性，这增加了对其在各个领域代表人类利益潜力的兴趣。现有大多数研究侧重于行为克隆，有效评估模型在多大程度上再现了个人表达的偏好。基于政治代表理论，我们强调了一个未充分探讨的设计权衡：AI系统应该作为代理，镜像表达的偏好，还是作为监护人，根据最有利于个人利益进行判断。这一权衡与大型语言模型的巴结行为密切相关，即模型可能会鼓励某些行为或验证可能与用户短期偏好一致但对其长期利益有害的信念。通过一系列模拟美国政策议题投票的实验，我们应用了一个时间效用框架，权衡短期和长期利益（模拟监护人角色），并将投票结果与行为克隆模型（模拟代理）进行比较。我们发现，倾向于长期利益的监护人式预测产生更符合专家共识的政策决策，但对缺乏明确共识的话题也表现出更大的偏好偏见。这些发现揭示了设计旨在代表人类利益的AI系统的基本权衡。代理模型更好地保护用户自主性，但可能偏离得到广泛支持的政策立场，而监护人模型可以在理解良好的议题上促进福祉，但在主观话题上可能有 paternalism 和偏见的风险。 

---
# Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think? 

**Title (ZH)**: 揭秘混合思维：大型语言模型真的能切换到不思考模式吗？ 

**Authors**: Shouren Wang, Wang Yang, Xianxuan Long, Qifan Wang, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.12680)  

**Abstract**: Hybrid thinking enables LLMs to switch between reasoning and direct answering, offering a balance between efficiency and reasoning capability. Yet our experiments reveal that current hybrid thinking LLMs only achieve partial mode separation: reasoning behaviors often leak into the no-think mode. To understand and mitigate this, we analyze the factors influencing controllability and identify four that matter most: (1) larger data scale, (2) using think and no-think answers from different questions rather than the same question, (3) a moderate increase in no-think data number, and (4) a two-phase strategy that first trains reasoning ability and then applies hybrid think training. Building on these findings, we propose a practical recipe that, compared to standard training, can maintain accuracy in both modes while significantly reducing no-think output length (from $1085$ to $585$ on MATH500) and occurrences of reasoning-supportive tokens such as ``\texttt{wait}'' (from $5917$ to $522$ on MATH500). Our findings highlight the limitations of current hybrid thinking and offer directions for strengthening its controllability. 

**Abstract (ZH)**: 混合推理使大语言模型能够在推理和直接回答之间切换，提供效率和推理能力之间的平衡。然而，我们的实验揭示当前的混合推理大语言模型仅部分实现了模式分离：推理行为常常渗入无思考模式中。为了理解并减轻这一现象，我们分析了影响可控性的因素，并确定了四个最相关因素：（1）更大的数据规模，（2）使用来自不同问题的思考和无思考回答而非同一问题的回答，（3）适度增加无思考数据的数量，以及（4）一个两阶段策略，先训练推理能力，再进行混合推理训练。基于这些发现，我们提出了一种实用的方法，与标准训练相比，该方法可以在两种模式下保持准确性，同时显著减少无思考输出长度（从MATH500的1085减少到585）和支持推理的标记如“wait”的出现频率（从MATH500的5917减少到522）。我们的研究突显了当前混合推理的局限性，并提供了增强其可控性的方向。 

---
# SG-XDEAT: Sparsity-Guided Cross-Dimensional and Cross-Encoding Attention with Target-Aware Conditioning in Tabular Learning 

**Title (ZH)**: SG-XDEAT: 基于稀疏性指导的跨维度和跨编码注意力与目标感知条件在表格学习中的应用 

**Authors**: Chih-Chuan Cheng, Yi-Ju Tseng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12659)  

**Abstract**: We propose SG-XDEAT (Sparsity-Guided Cross Dimensional and Cross-Encoding Attention with Target Aware Conditioning), a novel framework designed for supervised learning on tabular data. At its core, SG-XDEAT employs a dual-stream encoder that decomposes each input feature into two parallel representations: a raw value stream and a target-conditioned (label-aware) stream. These dual representations are then propagated through a hierarchical stack of attention-based modules. SG-XDEAT integrates three key components: (i) Cross-Dimensional self-attention, which captures intra-view dependencies among features within each stream; (ii) Cross-Encoding self-attention, which enables bidirectional interaction between raw and target-aware representations; and (iii) an Adaptive Sparse Self-Attention (ASSA) mechanism, which dynamically suppresses low-utility tokens by driving their attention weights toward zero--thereby mitigating the impact of noise. Empirical results on multiple public benchmarks show consistent gains over strong baselines, confirming that jointly modeling raw and target-aware views--while adaptively filtering noise--yields a more robust deep tabular learner. 

**Abstract (ZH)**: 我们提出了一种新型框架SG-XDEAT（稀疏性引导的跨维度和跨编码注意机制，带有目标感知条件），该框架旨在用于表格数据的监督学习。SG-XDEAT的核心采用了一种双流编码器，将每个输入特征分解为两个并行表示：原始值流和目标条件化（标签感知）流。这两种表示随后通过多级堆叠的基于注意机制的模块进行传递。SG-XDEAT结合了三个关键组件：跨维度自我注意，用于捕捉每一流内部特征间的依赖关系；跨编码自我注意，使原始和目标感知表示之间实现双向交互；以及自适应稀疏自我注意( ASSA)机制，该机制通过驱动低效令牌的注意权重趋向于零来动态抑制噪声，从而减轻噪声的影响。在多个公开基准上的实验证明，相较于强大的基线模型，联合建模原始和目标感知视图并自适应过滤噪声可以生成更 robust 的深层表格学习器。 

---
# Reasoning Pattern Matters: Learning to Reason without Human Rationales 

**Title (ZH)**: 推理模式matter：学习推理而不使用人类推理理由 

**Authors**: Chaoxu Pang, Yixuan Cao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12643)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities under the widely adopted SFT+RLVR paradigm, which first performs Supervised Fine-Tuning (SFT) on human-annotated reasoning trajectories (rationales) to establish initial reasoning behaviors, then applies Reinforcement Learning with Verifiable Rewards (RLVR) to optimize the model using verifiable signals without golden rationales. However, annotating high-quality rationales for the SFT stage remains prohibitively expensive. This paper investigates when and how rationale annotation costs can be substantially reduced without compromising reasoning performance. We identify a broad class of problems, termed patterned reasoning tasks, where reasoning follows a fixed, procedural strategy consistent across instances. Although instances vary in content such as domain knowledge, factual information, or numeric values, the solution derives from applying a shared reasoning pattern. We argue that the success of SFT+RLVR on such tasks primarily stems from its ability to enable models to internalize these reasoning patterns. Using numerical semantic matching as a representative task, we provide both causal and behavioral evidence showing that reasoning patterns rather than the quantity or quality of rationales are the key determinant of performance. Building on these insights, we propose Pattern-Aware LLMs as Rationale AnnOtators (PARO), a simple yet effective framework that enables LLMs to generate rationales aligned with task-specific reasoning patterns without requiring human rationale annotations. Experiments show that PARO-generated rationales achieve comparable SFT+RLVR performance to human rationales that are 10 times larger. These results suggest that large-scale human rationale annotations can be replaced with LLM-based automatic annotations requiring only limited human supervision over reasoning patterns. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在广泛采用的SFT+RLVR范式下展示了卓越的推理能力，该范式首先通过人工标注的推理路径（合理性）进行监督微调（SFT），建立初步的推理行为，然后利用可验证奖励的强化学习（RLVR）优化模型，使用可验证的信号而不是金标准合理性来优化。然而，为SFT阶段标注高质量的合理性仍然代价高昂。本文研究了如何在不牺牲推理性能的情况下显著降低合理性标注成本。我们识别出一类问题，称为模式化推理任务，在这类任务中，推理遵循一种固定的、程序化的策略，且在实例之间保持一致。尽管实例在内容如领域知识、事实信息或数值方面有所不同，但解决方案源自应用共享的推理模式。我们认为，SFT+RLVR在这些任务上的成功主要归因于其使模型内化这些推理模式的能力。以数值语义匹配为代表任务，我们提供了因果和行为证据，表明推理模式而非合理性数量或质量是决定性能的关键因素。基于这些洞见，我们提出了一种模式感知大规模语言模型作为合理性注释器（PARO）的简单而有效的框架，使大规模语言模型能够在不需要人类合理性注释的情况下生成与任务特定推理模式对齐的合理性。实验结果显示，PARO生成的合理性与人类注释的合理性（大10倍）在SFT+RLVR性能上具有可比性。这些结果表明，大规模人工合理性注释可以被基于大规模语言模型的自动注释所取代，仅需少量的人类监督以确保推理模式正确。 

---
# Aixel: A Unified, Adaptive and Extensible System for AI-powered Data Analysis 

**Title (ZH)**: Aixel：一种统一、自适应且可扩展的AI驱动数据分析系统 

**Authors**: Meihui Zhang, Liming Wang, Chi Zhang, Zhaojing Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12642)  

**Abstract**: A growing trend in modern data analysis is the integration of data management with learning, guided by accuracy, latency, and cost requirements. In practice, applications draw data of different formats from many sources. In the meanwhile, the objectives and budgets change over time. Existing systems handle these applications across databases, analysis libraries, and tuning services. Such fragmentation leads to complex user interaction, limited adaptability, suboptimal performance, and poor extensibility across components. To address these challenges, we present Aixel, a unified, adaptive, and extensible system for AI-powered data analysis. The system organizes work across four layers: application, task, model, and data. The task layer provides a declarative interface to capture user intent, which is parsed into an executable operator plan. An optimizer compiles and schedules this plan to meet specified goals in accuracy, latency, and cost. The task layer coordinates the execution of data and model operators, with built-in support for reuse and caching to improve efficiency. The model layer offers versioned storage for index, metadata, tensors, and model artifacts. It supports adaptive construction, task-aligned drift detection, and safe updates that reuse shared components. The data layer provides unified data management capabilities, including indexing, constraint-aware discovery, task-aligned selection, and comprehensive feature management. With the above designed layers, Aixel delivers a user friendly, adaptive, efficient, and extensible system. 

**Abstract (ZH)**: 现代数据分析中一个 growing trend 是将数据管理与学习集成，受到准确性、延迟和成本要求的指导。在实际应用中，应用程序从多种来源获取不同格式的数据。同时，目标和预算会随时间变化。现有系统通过数据库、分析库和调优服务来处理这些应用。这种碎片化导致用户交互复杂、适应性有限、各组件间性能不佳和扩展性差。为解决这些挑战，我们提出 Aixel，一个基于 AI 的数据分析统一、适应性和可扩展系统。该系统跨越四个层次组织工作：应用层、任务层、模型层和数据层。任务层提供声明式接口捕获用户意图，并将其解析为可执行的操作计划。优化器将该计划编译和调度以满足指定的准确度、延迟和成本目标。任务层协调数据和模型操作的执行，内置支持重用和缓存以提高效率。模型层提供版本化的索引、元数据、张量和模型 artefacts 存储。它支持自适应构建、任务对齐的漂移检测和安全更新，以重用共享组件。数据层提供统一的数据管理能力，包括索引、约束感知发现、任务对齐的选择和全面的功能管理。通过上述设计的层次，Aixel 提供了一个用户友好、适应性强、高效和可扩展的系统。 

---
# Laminar: A Scalable Asynchronous RL Post-Training Framework 

**Title (ZH)**: Laminar：一种可扩展的异步RL后训练框架 

**Authors**: Guangming Sheng, Yuxuan Tong, Borui Wan, Wang Zhang, Chaobo Jia, Xibin Wu, Yuqi Wu, Xiang Li, Chi Zhang, Yanghua Peng, Haibin Lin, Xin Liu, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12633)  

**Abstract**: Reinforcement learning (RL) post-training for Large Language Models (LLMs) is now scaling to large clusters and running for extended durations to enhance model reasoning performance. However, the scalability of existing RL frameworks is limited, as extreme long-tail skewness in RL trajectory generation causes severe GPU underutilization. Current asynchronous RL systems attempt to mitigate this, but they rely on global weight synchronization between the actor and all rollouts, which creates a rigid model update schedule. This global synchronization is ill-suited for the highly skewed and evolving distribution of trajectory generation latency in RL training, crippling training efficiency. Our key insight is that efficient scaling requires breaking this lockstep through trajectory-level asynchrony, which generates and consumes each trajectory independently. We propose Laminar, a scalable and robust RL post-training system built on a fully decoupled architecture. First, we replace global updates with a tier of relay workers acting as a distributed parameter service. This enables asynchronous and fine-grained weight synchronization, allowing rollouts to pull the latest weight anytime without stalling the actor's training loop. Second, a dynamic repack mechanism consolidates long-tail trajectories onto a few dedicated rollouts, maximizing generation throughput. The fully decoupled design also isolates failures, ensuring robustness for long-running jobs. Our evaluation on a 1024-GPU cluster shows that Laminar achieves up to 5.48$\times$ training throughput speedup over state-of-the-art systems, while reducing model convergence time. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的训练后强化学习（RL）正在扩展到大型集群，并运行更长时间以提升模型推理性能。然而，现有RL框架的扩展性受到限制，因为RL轨迹生成中的极端长尾偏斜导致严重的GPU利用率低下。当前的异步RL系统试图解决这一问题，但它们依赖于演员与所有轨迹之间的全局权重同步，这创造了一个僵化的模型更新日程。这种全局同步不适用于RL训练中高度偏斜且不断变化的轨迹生成延迟分布，导致训练效率受损。我们的关键洞察是，高效的扩展需要通过轨迹级别的异步性来打破这种同步，以独立生成和消费每个轨迹。我们提出了Laminar，这是一种基于完全解耦架构的可扩展且健壯的训练后RL系统。首先，我们用一层relay工作者取代全局更新，这些relay工作者作为分布式参数服务运行。这使权重同步变为异步和细粒度的，使得卷集随时可以拉取最新权重而不阻碍演员的训练循环。其次，动态重新打包机制将长尾轨迹合并到少数专用卷集中，最大化生成吞吐量。完全解耦的设计还隔离了故障，确保了长时间运行作业的健壯性。在1024-GPU集群上的评估表明，Laminar相比最先进的系统实现了高达5.48倍的训练吞吐量加速，并缩短了模型收敛时间。 

---
# Designing Tools with Control Confidence 

**Title (ZH)**: 设计具有控制信心的工具 

**Authors**: Ajith Anil Meera, Abian Torres, Pablo Lanillos  

**Link**: [PDF](https://arxiv.org/pdf/2510.12630)  

**Abstract**: Prehistoric humans invented stone tools for specialized tasks by not just maximizing the tool's immediate goal-completion accuracy, but also increasing their confidence in the tool for later use under similar settings. This factor contributed to the increased robustness of the tool, i.e., the least performance deviations under environmental uncertainties. However, the current autonomous tool design frameworks solely rely on performance optimization, without considering the agent's confidence in tool use for repeated use. Here, we take a step towards filling this gap by i) defining an optimization framework for task-conditioned autonomous hand tool design for robots, where ii) we introduce a neuro-inspired control confidence term into the optimization routine that helps the agent to design tools with higher robustness. Through rigorous simulations using a robotic arm, we show that tools designed with control confidence as the objective function are more robust to environmental uncertainties during tool use than a pure accuracy-driven objective. We further show that adding control confidence to the objective function for tool design provides a balance between the robustness and goal accuracy of the designed tools under control perturbations. Finally, we show that our CMAES-based evolutionary optimization strategy for autonomous tool design outperforms other state-of-the-art optimizers by designing the optimal tool within the fewest iterations. Code: this https URL. 

**Abstract (ZH)**: 史前人类通过不仅最大化工具的即时目标完成准确性，还提高其在相似环境条件下Later使用时的信心，来专门为特定任务发明石器。这一因素增加了工具的鲁棒性，即在环境不确定性下的最小性能偏差。然而，当前的自主工具设计框架仅依赖于性能优化，而不考虑执行者在重复使用工具时对该工具的信心。在此，我们通过(i) 定义一种任务条件下的自主手工具设计优化框架，以及(ii) 在优化过程中引入灵感于神经系统的控制信心项，来填补这一缺口，从而帮助代理设计具有更高鲁棒性的工具。通过使用机械臂进行严格的仿真，我们表明，以控制信心为目标函数设计的工具在使用过程中对环境不确定性具有更强的鲁棒性，优于纯准确性驱动的目标函数。我们进一步表明，将控制信心添加到设计工具的目标函数中，在控制扰动下为设计工具提供了鲁棒性和目标准确性之间的平衡。最后，我们展示了基于CMAES的自主工具设计进化优化策略在最少迭代次数内设计出最优工具，优于其他最先进的优化器。代码: https://this-url.com。 

---
# Learning-To-Measure: In-context Active Feature Acquisition 

**Title (ZH)**: 基于上下文的主动特征获取学习 

**Authors**: Yuta Kobayashi, Zilin Jing, Jiayu Yao, Hongseok Namkoong, Shalmali Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12624)  

**Abstract**: Active feature acquisition (AFA) is a sequential decision-making problem where the goal is to improve model performance for test instances by adaptively selecting which features to acquire. In practice, AFA methods often learn from retrospective data with systematic missingness in the features and limited task-specific labels. Most prior work addresses acquisition for a single predetermined task, limiting scalability. To address this limitation, we formalize the meta-AFA problem, where the goal is to learn acquisition policies across various tasks. We introduce Learning-to-Measure (L2M), which consists of i) reliable uncertainty quantification over unseen tasks, and ii) an uncertainty-guided greedy feature acquisition agent that maximizes conditional mutual information. We demonstrate a sequence-modeling or autoregressive pre-training approach that underpins reliable uncertainty quantification for tasks with arbitrary missingness. L2M operates directly on datasets with retrospective missingness and performs the meta-AFA task in-context, eliminating per-task retraining. Across synthetic and real-world tabular benchmarks, L2M matches or surpasses task-specific baselines, particularly under scarce labels and high missingness. 

**Abstract (ZH)**: 主动特征获取的元学习（Meta-AFA）：可靠不确定性量化与不确定性引导的贪心特征获取代理 

---
# Rethinking Knowledge Distillation: A Data Dependent Regulariser With a Negative Asymmetric Payoff 

**Title (ZH)**: 重新思考知识蒸馏：一种基于数据的正负非对称惩罚正则化项 

**Authors**: Israel Mason-Williams, Gabryel Mason-Williams, Helen Yannakoudakis  

**Link**: [PDF](https://arxiv.org/pdf/2510.12615)  

**Abstract**: Knowledge distillation is often considered a compression mechanism when judged on the resulting student's accuracy and loss, yet its functional impact is poorly understood. In this work, we quantify the compression capacity of knowledge distillation and the resulting knowledge transfer from a functional perspective, decoupling compression from architectural reduction, which provides an improved understanding of knowledge distillation. We employ hypothesis testing, controls, and random control distillation to understand knowledge transfer mechanisms across data modalities. To rigorously test the breadth and limits of our analyses, we explore multiple distillation variants and analyse distillation scaling laws across model sizes. Our findings demonstrate that, while there is statistically significant knowledge transfer in some modalities and architectures, the extent of this transfer is less pronounced than anticipated, even under conditions designed to maximise knowledge sharing. Notably, in cases of significant knowledge transfer, we identify a consistent and severe asymmetric transfer of negative knowledge to the student, raising safety concerns in knowledge distillation applications. Across 12 experimental setups, 9 architectures, and 7 datasets, our findings show that knowledge distillation functions less as a compression mechanism and more as a data-dependent regulariser with a negative asymmetric payoff. 

**Abstract (ZH)**: 知识蒸馏的功能压缩能力及其知识转移机制：从功能视角量化知识蒸馏的压缩容量和知识转移 

---
# StyleDecipher: Robust and Explainable Detection of LLM-Generated Texts with Stylistic Analysis 

**Title (ZH)**: StyleDecipher: 基于风格分析的鲁棒且可解释的LLM生成文本检测方法 

**Authors**: Siyuan Li, Aodu Wulianghai, Xi Lin, Guangyan Li, Xiang Chen, Jun Wu, Jianhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.12608)  

**Abstract**: With the increasing integration of large language models (LLMs) into open-domain writing, detecting machine-generated text has become a critical task for ensuring content authenticity and trust. Existing approaches rely on statistical discrepancies or model-specific heuristics to distinguish between LLM-generated and human-written text. However, these methods struggle in real-world scenarios due to limited generalization, vulnerability to paraphrasing, and lack of explainability, particularly when facing stylistic diversity or hybrid human-AI authorship. In this work, we propose StyleDecipher, a robust and explainable detection framework that revisits LLM-generated text detection using combined feature extractors to quantify stylistic differences. By jointly modeling discrete stylistic indicators and continuous stylistic representations derived from semantic embeddings, StyleDecipher captures distinctive style-level divergences between human and LLM outputs within a unified representation space. This framework enables accurate, explainable, and domain-agnostic detection without requiring access to model internals or labeled segments. Extensive experiments across five diverse domains, including news, code, essays, reviews, and academic abstracts, demonstrate that StyleDecipher consistently achieves state-of-the-art in-domain accuracy. Moreover, in cross-domain evaluations, it surpasses existing baselines by up to 36.30%, while maintaining robustness against adversarial perturbations and mixed human-AI content. Further qualitative and quantitative analysis confirms that stylistic signals provide explainable evidence for distinguishing machine-generated text. Our source code can be accessed at this https URL. 

**Abstract (ZH)**: 大型语言模型在开放领域写作中的广泛应用促使检测机器生成文本成为确保内容真实性与信任的关键任务。现有方法依赖统计差异或模型特定的启发式方法来区分大型语言模型生成的文本和人工撰写的文本。然而，这些方法在实际场景中面临泛化能力有限、易受到改写影响以及缺乏解释性等问题，特别是在面对风格多样性或人机合作创作内容时。在本文中，我们提出StyleDecipher，一种稳健且可解释的检测框架，重新审视大型语言模型生成文本检测，通过结合特征提取器来量化风格差异。通过联合建模离散的风格指标和语义嵌入连续的风格表示，StyleDecipher在统一的表示空间中捕捉了人工和大型语言模型输出之间的风格级差异。该框架能够在无需访问模型内部结构或标注片段的情况下实现准确、可解释且领域无关的检测。在新闻、代码、随笔、评论和学术摘要等五个不同领域的广泛实验中，StyleDecipher展示了始终达到领域内最先进的准确率。此外，在跨领域评估中，它在保持对抗性扰动和混合人机内容的鲁棒性方面超过了现有基准方法高达36.30%。进一步的定性和定量分析确认了风格信号为区分机器生成文本提供了可解释的证据。我们的源代码可以通过这个链接访问。 

---
# SMILE: SeMantic Ids Enhanced CoLd Item Representation for Click-through Rate Prediction in E-commerce SEarch 

**Title (ZH)**: SMILE: 基于语义ID增强的冷启动项表示以提高电商平台搜索点击率预测 

**Authors**: Qihang Zhao, Zhongbo Sun, Xiaoyang Zheng, Xian Guo, Siyuan Wang, Zihan Liang, Mingcan Peng, Ben Chen, Chenyi Lei  

**Link**: [PDF](https://arxiv.org/pdf/2510.12604)  

**Abstract**: With the rise of modern search and recommendation platforms, insufficient collaborative information of cold-start items exacerbates the Matthew effect of existing platform items, challenging platform diversity and becoming a longstanding issue. Existing methods align items' side content with collaborative information to transfer collaborative signals from high-popularity items to cold-start items. However, these methods fail to account for the asymmetry between collaboration and content, nor the fine-grained differences among items. To address these issues, we propose SMILE, an item representation enhancement approach based on fused alignment of semantic IDs. Specifically, we use RQ-OPQ encoding to quantize item content and collaborative information, followed by a two-step alignment: RQ encoding transfers shared collaborative signals across items, while OPQ encoding learns differentiated information of items. Comprehensive offline experiments on large-scale industrial datasets demonstrate superiority of SMILE, and rigorous online A/B tests confirm statistically significant improvements: item CTR +1.66%, buyers +1.57%, and order volume +2.17%. 

**Abstract (ZH)**: 随着现代搜索和推荐平台的兴起，冷启动项目的协作信息不足加剧了现有平台项目的马太效应，挑战了平台的多样性，成为一个长期存在的问题。现有的方法通过将热门项目的侧内容与协作信息对齐，将协作信号从热门项目转移到冷启动项目，但这些方法未能考虑到协作与内容之间的不对称性，以及项目之间的细微差别。为了解决这些问题，我们提出了基于融合语义ID对齐的项目表示增强方法SMILE。具体而言，我们使用RQ-OPQ编码对项目内容和协作信息进行量化，并通过两步对齐过程进行融合：RQ编码在项目间转移共享的协作信号，而OPQ编码学习项目的差异化信息。大规模工业数据集上的全面离线实验展示了SMILE的优势，严格的在线A/B测试也证实了显著的提升：项目点击率+1.66%，购买者+1.57%，订单量+2.17%。 

---
# Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space 

**Title (ZH)**: 在黑暗中推理：潜在空间中的交互式视觉-文本推理 

**Authors**: Chao Chen, Zhixin Ma, Yongqi Li, Yupeng Hu, Yinwei Wei, Wenjie Li, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.12603)  

**Abstract**: Multimodal reasoning aims to enhance the capabilities of MLLMs by incorporating intermediate reasoning steps before reaching the final answer. It has evolved from text-only reasoning to the integration of visual information, enabling the thought process to be conveyed through both images and text. Despite its effectiveness, current multimodal reasoning methods depend on explicit reasoning steps that require labor-intensive vision-text annotations and inherently introduce significant inference latency. To address these issues, we introduce multimodal latent reasoning with the advantages of multimodal representation, reduced annotation, and inference efficiency. To facilicate it, we propose Interleaved Vision-Text Latent Reasoning (IVT-LR), which injects both visual and textual information in the reasoning process within the latent space. Specifically, IVT-LR represents each reasoning step by combining two implicit parts: latent text (the hidden states from the previous step) and latent vision (a set of selected image embeddings). We further introduce a progressive multi-stage training strategy to enable MLLMs to perform the above multimodal latent reasoning steps. Experiments on M3CoT and ScienceQA demonstrate that our IVT-LR method achieves an average performance increase of 5.45% in accuracy, while simultaneously achieving a speed increase of over 5 times compared to existing approaches. Code available at this https URL. 

**Abstract (ZH)**: 多模态潜推理：多模态表示、标注减少及推理效率的优势 

---
# Evaluation of Real-Time Preprocessing Methods in AI-Based ECG Signal Analysis 

**Title (ZH)**: 基于AI的心电图信号分析中实时预处理方法的评估 

**Authors**: Jasmin Freudenberg, Kai Hahn, Christian Weber, Madjid Fathi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12541)  

**Abstract**: The increasing popularity of portable ECG systems and the growing demand for privacy-compliant, energy-efficient real-time analysis require new approaches to signal processing at the point of data acquisition. In this context, the edge domain is acquiring increasing importance, as it not only reduces latency times, but also enables an increased level of data security. The FACE project aims to develop an innovative machine learning solution for analysing long-term electrocardiograms that synergistically combines the strengths of edge and cloud computing. In this thesis, various pre-processing steps of ECG signals are analysed with regard to their applicability in the project. The selection of suitable methods in the edge area is based in particular on criteria such as energy efficiency, processing capability and real-time capability. 

**Abstract (ZH)**: 便携式ECG系统 popularity 和对隐私合规、节能实时分析需求的不断增加促使在数据采集点采用新的信号处理方法。在这种背景下，边缘域的重要性日益凸显，不仅减少了延迟时间，还提升了数据安全水平。FACE项目旨在开发一种结合边缘计算和云计算优势的创新机器学习解决方案，用于分析长期心电图。在本论文中，分析了心电图信号的各种预处理步骤，以评估其在项目中的适用性。在边缘区域选择合适的方法时，特别考虑了能效、处理能力和实时能力等标准。 

---
# Unconditional Human Motion and Shape Generation via Balanced Score-Based Diffusion 

**Title (ZH)**: 无条件的人运动态和形状生成通过平衡的分数基于扩散模型 

**Authors**: David Björkstrand, Tiesheng Wang, Lars Bretzner, Josephine Sullivan  

**Link**: [PDF](https://arxiv.org/pdf/2510.12537)  

**Abstract**: Recent work has explored a range of model families for human motion generation, including Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and diffusion-based models. Despite their differences, many methods rely on over-parameterized input features and auxiliary losses to improve empirical results. These strategies should not be strictly necessary for diffusion models to match the human motion distribution. We show that on par with state-of-the-art results in unconditional human motion generation are achievable with a score-based diffusion model using only careful feature-space normalization and analytically derived weightings for the standard L2 score-matching loss, while generating both motion and shape directly, thereby avoiding slow post hoc shape recovery from joints. We build the method step by step, with a clear theoretical motivation for each component, and provide targeted ablations demonstrating the effectiveness of each proposed addition in isolation. 

**Abstract (ZH)**: 最近的研究已经探索了多种人类运动生成的模型家族，包括变分自编码器（VAEs）、生成对抗网络（GANs）和基于扩散的模型。尽管这些方法在输入特征和辅助损失上有所不同，许多方法仍然依赖于过度参数化的输入特征和辅助损失来提高实验结果。这些策略并非必须适用于达到与最先进的无条件人类运动生成结果相当的结果。我们证明，仅通过仔细的空间归一化和标准L2评分匹配损失的分析权重，即可实现与最先进的结果相当的无条件人类运动生成结果，同时直接生成运动和形状，从而避免了从关节后处理形状恢复的缓慢过程。我们逐步构建该方法，并为每个组件提供清晰的理论动机，同时通过针对性的消融实验展示了每个提出的添加项的有效性。 

---
# BoN Appetit Team at LeWiDi-2025: Best-of-N Test-time Scaling Can Not Stomach Annotation Disagreements (Yet) 

**Title (ZH)**: BoN Appetit Team在LeWiDi-2025：最佳-of-N测试时扩展无法容忍标注分歧（目前尚且无法接受） 

**Authors**: Tomas Ruiz, Siyao Peng, Barbara Plank, Carsten Schwemmer  

**Link**: [PDF](https://arxiv.org/pdf/2510.12516)  

**Abstract**: Test-time scaling is a family of techniques to improve LLM outputs at inference time by performing extra computation. To the best of our knowledge, test-time scaling has been limited to domains with verifiably correct answers, like mathematics and coding. We transfer test-time scaling to the LeWiDi-2025 tasks to evaluate annotation disagreements. We experiment with three test-time scaling methods: two benchmark algorithms (Model Averaging and Majority Voting), and a Best-of-N sampling method. The two benchmark methods improve LLM performance consistently on the LeWiDi tasks, but the Best-of-N method does not. Our experiments suggest that the Best-of-N method does not currently transfer from mathematics to LeWiDi tasks, and we analyze potential reasons for this gap. 

**Abstract (ZH)**: 测试时缩放是一种在推理时通过额外计算提高LLM输出的技术。据我们所知，测试时缩放主要应用于具有可验证正确答案的领域，如数学和编程。我们将在LeWiDi-2025任务中转移测试时缩放技术以评估注释 disagreements。我们实验了三种测试时缩放方法：两个基准算法（模型平均和多数表决），以及一种Best-of-N采样方法。两种基准方法在LeWiDi任务中一致地提高了LLM性能，但Best-of-N方法未做到这一点。我们的实验表明，Best-of-N方法当前无法从数学任务转移到LeWiDi任务，并分析了这一差距的原因。 

---
# The Robustness of Differentiable Causal Discovery in Misspecified Scenarios 

**Title (ZH)**: 差分因果发现方法在错定模型场景下的鲁棒性 

**Authors**: Huiyang Yi, Yanyan He, Duxin Chen, Mingyu Kang, He Wang, Wenwu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12503)  

**Abstract**: Causal discovery aims to learn causal relationships between variables from targeted data, making it a fundamental task in machine learning. However, causal discovery algorithms often rely on unverifiable causal assumptions, which are usually difficult to satisfy in real-world data, thereby limiting the broad application of causal discovery in practical scenarios. Inspired by these considerations, this work extensively benchmarks the empirical performance of various mainstream causal discovery algorithms, which assume i.i.d. data, under eight model assumption violations. Our experimental results show that differentiable causal discovery methods exhibit robustness under the metrics of Structural Hamming Distance and Structural Intervention Distance of the inferred graphs in commonly used challenging scenarios, except for scale variation. We also provide the theoretical explanations for the performance of differentiable causal discovery methods. Finally, our work aims to comprehensively benchmark the performance of recent differentiable causal discovery methods under model assumption violations, and provide the standard for reasonable evaluation of causal discovery, as well as to further promote its application in real-world scenarios. 

**Abstract (ZH)**: 因果发现旨在从目标数据中学习变量间的因果关系，它是机器学习中的一个基本任务。然而，因果发现算法往往依赖于难以在现实世界数据中验证的因果假设，这限制了因果发现在实际应用场景中的广泛应用。基于这些考虑，本研究在八种模型假设违反情况下，广泛评测了各类主流因果发现算法在经验性能上的表现。我们的实验结果表明，可微因果发现方法在常用的具有挑战性的场景中，除了规模变化以外，在结构汉明距离和结构干预距离的度量下表现出鲁棒性。我们还提供了可微因果发现方法性能的理论解释。最后，本研究旨在在模型假设违反情况下全面评测近期的可微因果发现方法，并提供合理的因果发现评估标准，进一步促进其在实际应用场景中的应用。 

---
# PubSub-VFL: Towards Efficient Two-Party Split Learning in Heterogeneous Environments via Publisher/Subscriber Architecture 

**Title (ZH)**: PubSub-VFL：面向异构环境高效双方拆分学习的发布者/订阅者架构 

**Authors**: Yi Liu, Yang Liu, Leqian Zheng, Jue Hong, Junjie Shi, Qingyou Yang, Ye Wu, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12494)  

**Abstract**: With the rapid advancement of the digital economy, data collaboration between organizations has become a well-established business model, driving the growth of various industries. However, privacy concerns make direct data sharing impractical. To address this, Two-Party Split Learning (a.k.a. Vertical Federated Learning (VFL)) has emerged as a promising solution for secure collaborative learning. Despite its advantages, this architecture still suffers from low computational resource utilization and training efficiency. Specifically, its synchronous dependency design increases training latency, while resource and data heterogeneity among participants further hinder efficient computation. To overcome these challenges, we propose PubSub-VFL, a novel VFL paradigm with a Publisher/Subscriber architecture optimized for two-party collaborative learning with high computational efficiency. PubSub-VFL leverages the decoupling capabilities of the Pub/Sub architecture and the data parallelism of the parameter server architecture to design a hierarchical asynchronous mechanism, reducing training latency and improving system efficiency. Additionally, to mitigate the training imbalance caused by resource and data heterogeneity, we formalize an optimization problem based on participants' system profiles, enabling the selection of optimal hyperparameters while preserving privacy. We conduct a theoretical analysis to demonstrate that PubSub-VFL achieves stable convergence and is compatible with security protocols such as differential privacy. Extensive case studies on five benchmark datasets further validate its effectiveness, showing that, compared to state-of-the-art baselines, PubSub-VFL not only accelerates training by $2 \sim 7\times$ without compromising accuracy, but also achieves a computational resource utilization rate of up to 91.07%. 

**Abstract (ZH)**: 随着数字经济的迅速发展，组织间的数据协作已成为成熟的商业模式，推动了各行各业的增长。然而，隐私问题使得直接数据共享变得 impractical。为解决这一问题，双方拆分学习（即垂直联邦学习 VFL）已 emergence 为一种有前景的安全协作学习解决方案。尽管 VFL 具有优势，但该架构仍面临计算资源利用率低和训练效率低的问题。具体而言，其同步依赖设计增加了训练延迟，而参与者之间资源和数据异构性进一步妨碍了高效的计算。为克服这些挑战，我们提出 PubSub-VFL，这是一种优化的面向高计算效率的双方协作学习的新 VFL 模式，采用发布者/订阅者架构。PubSub-VFL 利用了发布者/订阅者架构的解耦能力和参数服务器架构的数据并行性，设计了一种分层异步机制，从而减少训练延迟并提高系统效率。此外，为了缓解资源和数据异构性引起的训练不平衡，我们基于参与者系统配置形式化了一个优化问题，能够在保持隐私的前提下选择最优超参数。我们进行理论分析以证明 PubSub-VFL 实现了稳定收敛，并且兼容差分隐私等安全协议。通过对五个基准数据集的广泛案例研究进一步验证其有效性，结果表明，与当前最佳基线相比，PubSub-VFL 不仅在不牺牲准确性的基础上将训练速度加快了 2 到 7 倍，而且还实现了高达 91.07% 的计算资源利用率。 

---
# A Text-Image Fusion Method with Data Augmentation Capabilities for Referring Medical Image Segmentation 

**Title (ZH)**: 一种具备数据增强能力的文本-图像融合方法及其在参考医学图像分割中的应用 

**Authors**: Shurong Chai, Rahul Kumar JAIN, Rui Xu, Shaocong Mo, Ruibo Hou, Shiyu Teng, Jiaqing Liu, Lanfen Lin, Yen-Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12482)  

**Abstract**: Deep learning relies heavily on data augmentation to mitigate limited data, especially in medical imaging. Recent multimodal learning integrates text and images for segmentation, known as referring or text-guided image segmentation. However, common augmentations like rotation and flipping disrupt spatial alignment between image and text, weakening performance. To address this, we propose an early fusion framework that combines text and visual features before augmentation, preserving spatial consistency. We also design a lightweight generator that projects text embeddings into visual space, bridging semantic gaps. Visualization of generated pseudo-images shows accurate region localization. Our method is evaluated on three medical imaging tasks and four segmentation frameworks, achieving state-of-the-art results. Code is publicly available on GitHub: this https URL. 

**Abstract (ZH)**: 深度学习高度依赖数据增强以缓解数据不足的问题，特别是在医学影像领域。最近的多模态学习将文本和图像结合用于分割，这已知为参考或文本引导的图像分割。然而，常见的数据增强方式如旋转和翻转会破坏图像和文本之间的空间对齐性，削弱性能。为解决这一问题，我们提出了一种早期融合框架，在增强之前将文本和视觉特征结合，以保持空间一致性。我们还设计了一个轻量级生成器，将文本嵌入投影到视觉空间，以弥合语义差距。生成的伪图像可视化显示了准确的区域定位。我们的方法在三项医学影像任务和四种分割框架上进行了评估，取得了当前最佳结果。代码已在GitHub上公开：this https URL。 

---
# When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection 

**Title (ZH)**: 当个性化欺骗检测器：机器生成文本检测中的特征反转陷阱 

**Authors**: Lang Gao, Xuhui Li, Chenxi Wang, Mingzhe Li, Wei Liu, Zirui Song, Jinghui Zhang, Rui Yan, Preslav Nakov, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12476)  

**Abstract**: Large language models (LLMs) have grown more powerful in language generation, producing fluent text and even imitating personal style. Yet, this ability also heightens the risk of identity impersonation. To the best of our knowledge, no prior work has examined personalized machine-generated text (MGT) detection. In this paper, we introduce \dataset, the first benchmark for evaluating detector robustness in personalized settings, built from literary and blog texts paired with their LLM-generated imitations. Our experimental results demonstrate large performance gaps across detectors in personalized settings: some state-of-the-art models suffer significant drops. We attribute this limitation to the \textit{feature-inversion trap}, where features that are discriminative in general domains become inverted and misleading when applied to personalized text. Based on this finding, we propose \method, a simple and reliable way to predict detector performance changes in personalized settings. \method identifies latent directions corresponding to inverted features and constructs probe datasets that differ primarily along these features to evaluate detector dependence. Our experiments show that \method can accurately predict both the direction and the magnitude of post-transfer changes, showing 85\% correlation with the actual performance gaps. We hope that this work will encourage further research on personalized text detection. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言生成方面变得更为强大，能够生成流畅的文本甚至模仿个人风格。然而，这种能力也加剧了身份冒充的风险。据我们所知，尚未有先有研究探讨个性化机器生成文本（MGT）的检测问题。在本文中，我们介绍了第一个用于评估个性化设置中检测器鲁棒性的基准 \dataset，该基准基于文学和博客文本及其由LLM生成的仿制品构建。我们的实验结果表明，在个性化设置中，不同检测器之间的性能存在巨大差距：一些最先进的模型出现了显著下降。我们将这一限制归因于“特征反转陷阱”，即在通用领域具有区分性的特征在应用于个性化文本时会变得颠倒并误导人。基于这一发现，我们提出了 \method，这是一种简单且可靠的方法，用于预测个性化设置中检测器性能的变化。\method 识别与反转特征对应的潜在方向，并构建主要沿这些特征差异的探针数据集，以评估检测器的依赖性。我们的实验表明，\method 可以准确预测转移后的方向和幅度变化，与实际性能差距的相关性高达85%。我们希望这项工作能促进个性化文本检测方面的进一步研究。 

---
# A Function Centric Perspective On Flat and Sharp Minima 

**Title (ZH)**: 以功能为中心视角下的扁平和尖锐最小值 

**Authors**: Israel Mason-Williams, Gabryel Mason-Williams, Helen Yannakoudakis  

**Link**: [PDF](https://arxiv.org/pdf/2510.12451)  

**Abstract**: Flat minima are widely believed to correlate with improved generalisation in deep neural networks. However, this connection has proven more nuanced in recent studies, with both theoretical counterexamples and empirical exceptions emerging in the literature. In this paper, we revisit the role of sharpness in model performance, proposing that sharpness is better understood as a function-dependent property rather than a reliable indicator of poor generalisation. We conduct extensive empirical studies, from single-objective optimisation to modern image classification tasks, showing that sharper minima often emerge when models are regularised (e.g., via SAM, weight decay, or data augmentation), and that these sharp minima can coincide with better generalisation, calibration, robustness, and functional consistency. Across a range of models and datasets, we find that baselines without regularisation tend to converge to flatter minima yet often perform worse across all safety metrics. Our findings demonstrate that function complexity, rather than flatness alone, governs the geometry of solutions, and that sharper minima can reflect more appropriate inductive biases (especially under regularisation), calling for a function-centric reappraisal of loss landscape geometry. 

**Abstract (ZH)**: 平坦极小值广泛认为与深度神经网络的泛化能力改进相关。然而，最近的研究表明，这种关联性更为复杂，理论上的反例和文献中的经验例外情况相继出现。在本文中，我们重新审视模型性能中的sharpness作用，提出sharpness应被视为一种函数相关的属性，而非不良泛化可靠的指示器。我们进行了广泛的实证研究，从单目标优化到现代图像分类任务，表明当模型正则化时（例如，通过SAM、权重衰减或数据增强），往往会形成更尖锐的极小值，并且这些更尖锐的极小值可以与更好的泛化、校准、鲁棒性和功能一致性相一致。在多种模型和数据集上，我们发现无正则化的基线倾向于收敛于更平坦的极小值，但在所有安全指标上通常表现更差。我们的发现表明，函数复杂性而非平坦性单独决定了解的空间几何结构，并且更尖锐的极小值可能反映了更合适的方向性偏差（特别是在正则化下），呼吁从函数中心的角度重新评估损失景观的几何结构。 

---
# Low-Field Magnetic Resonance Image Quality Enhancement using a Conditional Flow Matching Model 

**Title (ZH)**: 低场磁共振图像质量增强的条件流匹配模型 

**Authors**: Huu Tien Nguyen, Ahmed Karam Eldaly  

**Link**: [PDF](https://arxiv.org/pdf/2510.12408)  

**Abstract**: This paper introduces a novel framework for image quality transfer based on conditional flow matching (CFM). Unlike conventional generative models that rely on iterative sampling or adversarial objectives, CFM learns a continuous flow between a noise distribution and target data distributions through the direct regression of an optimal velocity field. We evaluate this approach in the context of low-field magnetic resonance imaging (LF-MRI), a rapidly emerging modality that offers affordable and portable scanning but suffers from inherently low signal-to-noise ratio and reduced diagnostic quality. Our framework is designed to reconstruct high-field-like MR images from their corresponding low-field inputs, thereby bridging the quality gap without requiring expensive infrastructure. Experiments demonstrate that CFM not only achieves state-of-the-art performance, but also generalizes robustly to both in-distribution and out-of-distribution data. Importantly, it does so while utilizing significantly fewer parameters than competing deep learning methods. These results underline the potential of CFM as a powerful and scalable tool for MRI reconstruction, particularly in resource-limited clinical environments. 

**Abstract (ZH)**: 基于条件流匹配的图像质量转移新型框架 

---
# Tokenization Disparities as Infrastructure Bias: How Subword Systems Create Inequities in LLM Access and Efficiency 

**Title (ZH)**: 子词划分差异作为基础设施偏差：亚词系统如何在LLM访问和效率方面创造不平等 

**Authors**: Hailay Kidu Teklehaymanot, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2510.12389)  

**Abstract**: Tokenization disparities pose a significant barrier to achieving equitable access to artificial intelligence across linguistically diverse populations. This study conducts a large-scale cross-linguistic evaluation of tokenization efficiency in over 200 languages to systematically quantify computational inequities in large language models (LLMs). Using a standardized experimental framework, we applied consistent preprocessing and normalization protocols, followed by uniform tokenization through the tiktoken library across all language samples. Comprehensive tokenization statistics were collected using established evaluation metrics, including Tokens Per Sentence (TPS) and Relative Tokenization Cost (RTC), benchmarked against English baselines. Our cross-linguistic analysis reveals substantial and systematic disparities: Latin-script languages consistently exhibit higher tokenization efficiency, while non-Latin and morphologically complex languages incur significantly greater token inflation, often 3-5 times higher RTC ratios. These inefficiencies translate into increased computational costs and reduced effective context utilization for underrepresented languages. Overall, the findings highlight structural inequities in current AI systems, where speakers of low-resource and non-Latin languages face disproportionate computational disadvantages. Future research should prioritize the development of linguistically informed tokenization strategies and adaptive vocabulary construction methods that incorporate typological diversity, ensuring more inclusive and computationally equitable multilingual AI systems. 

**Abstract (ZH)**: 跨语文本化差异阻碍了不同语言群体公平访问人工智能的机会。本研究对超过200种语言进行了大规模跨语文本化效率评估，系统量化了大型语言模型中的计算不公平性。通过标准化的实验框架，我们应用一致的预处理和归一化协议，然后使用tiktoken库对所有语言样本进行统一的文本化。我们使用公认的评估指标，包括每句令牌数量（TPS）和相对文本化成本（RTC），与英语基准进行比较，收集了全面的文本化统计数据。跨语文本分析揭示了显著且系统的差异：使用拉丁字母的 languages 一贯表现出更高的文本化效率，而非拉丁字母和词形变化复杂的 languages 则面临显著更大的令牌膨胀现象，通常 RTC 比率高出3-5倍。这些低效率转化为未充分代表的 languages 的计算成本增加，并减少了有效上下文的利用。总体而言，研究结果突显了当前 AI 系统中的结构性不公平性，使用者低资源和非拉丁字母 languages 的人群面临不成比例的计算劣势。未来研究应优先发展基于语言的文本化策略和适应性词汇构建方法，纳入类型学多样性，确保更具包容性和计算公平性多语言 AI 系统。 

---
# Phenome-Wide Multi-Omics Integration Uncovers Distinct Archetypes of Human Aging 

**Title (ZH)**: 全表型多组学整合揭示人类衰老的不同原型 

**Authors**: Huifa Li, Feilong Tang, Haochen Xue, Yulong Li, Xinlin Zhuang, Bin Zhang, Eran Segal, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2510.12384)  

**Abstract**: Aging is a highly complex and heterogeneous process that progresses at different rates across individuals, making biological age (BA) a more accurate indicator of physiological decline than chronological age. While previous studies have built aging clocks using single-omics data, they often fail to capture the full molecular complexity of human aging. In this work, we leveraged the Human Phenotype Project, a large-scale cohort of 12,000 adults aged 30--70 years, with extensive longitudinal profiling that includes clinical, behavioral, environmental, and multi-omics datasets -- spanning transcriptomics, lipidomics, metabolomics, and the microbiome. By employing advanced machine learning frameworks capable of modeling nonlinear biological dynamics, we developed and rigorously validated a multi-omics aging clock that robustly predicts diverse health outcomes and future disease risk. Unsupervised clustering of the integrated molecular profiles from multi-omics uncovered distinct biological subtypes of aging, revealing striking heterogeneity in aging trajectories and pinpointing pathway-specific alterations associated with different aging patterns. These findings demonstrate the power of multi-omics integration to decode the molecular landscape of aging and lay the groundwork for personalized healthspan monitoring and precision strategies to prevent age-related diseases. 

**Abstract (ZH)**: 人类表型项目揭示的多组学衰老时钟：分子景观的解码与个性化健康寿 span 监测的基础 

---
# LiteVPNet: A Lightweight Network for Video Encoding Control in Quality-Critical Applications 

**Title (ZH)**: LiteVPNet：一种适用于质量关键应用的 Lightweight 视频编码控制网络 

**Authors**: Vibhoothi Vibhoothi, François Pitié, Anil Kokaram  

**Link**: [PDF](https://arxiv.org/pdf/2510.12379)  

**Abstract**: In the last decade, video workflows in the cinema production ecosystem have presented new use cases for video streaming technology. These new workflows, e.g. in On-set Virtual Production, present the challenge of requiring precise quality control and energy efficiency. Existing approaches to transcoding often fall short of these requirements, either due to a lack of quality control or computational overhead. To fill this gap, we present a lightweight neural network (LiteVPNet) for accurately predicting Quantisation Parameters for NVENC AV1 encoders that achieve a specified VMAF score. We use low-complexity features, including bitstream characteristics, video complexity measures, and CLIP-based semantic embeddings. Our results demonstrate that LiteVPNet achieves mean VMAF errors below 1.2 points across a wide range of quality targets. Notably, LiteVPNet achieves VMAF errors within 2 points for over 87% of our test corpus, c.f. approx 61% with state-of-the-art methods. LiteVPNet's performance across various quality regions highlights its applicability for enhancing high-value content transport and streaming for more energy-efficient, high-quality media experiences. 

**Abstract (ZH)**: 过去十年，电影生产生态系统中的视频工作流为视频流技术提出了新的应用场景。这些新的工作流，例如现场虚拟制作，要求精确的质量控制和能源效率。现有的转码方法常常无法满足这些要求，要么是因为缺乏质量控制，要么是因为计算开销过大。为填补这一空白，我们提出了一种轻量级神经网络（LiteVPNet），用于准确预测NVENC AV1编码器的量化参数，以达到指定的VMAF分数。我们使用低复杂度特征，包括比特流特性、视频复杂度度量和基于CLIP的语义嵌入。我们的结果显示，LiteVPNet在多种质量目标范围内实现了平均VMAF误差低于1.2分。值得注意的是，LiteVPNet在超过87%的测试数据集中实现了VMAF误差在2分以内，相比之下，最先进的方法约为61%。LiteVPNet在不同质量区域的性能突显了其在提升高价值内容传输和流媒体以实现更高效、高质量媒体体验方面的应用潜力。 

---
# Deep Attention-guided Adaptive Subsampling 

**Title (ZH)**: 深注意力导向自适应子采样 

**Authors**: Sharath M Shankaranarayana, Soumava Kumar Roy, Prasad Sudhakar, Chandan Aladahalli  

**Link**: [PDF](https://arxiv.org/pdf/2510.12376)  

**Abstract**: Although deep neural networks have provided impressive gains in performance, these improvements often come at the cost of increased computational complexity and expense. In many cases, such as 3D volume or video classification tasks, not all slices or frames are necessary due to inherent redundancies. To address this issue, we propose a novel learnable subsampling framework that can be integrated into any neural network architecture. Subsampling, being a nondifferentiable operation, poses significant challenges for direct adaptation into deep learning models. While some works, have proposed solutions using the Gumbel-max trick to overcome the problem of non-differentiability, they fall short in a crucial aspect: they are only task-adaptive and not inputadaptive. Once the sampling mechanism is learned, it remains static and does not adjust to different inputs, making it unsuitable for real-world applications. To this end, we propose an attention-guided sampling module that adapts to inputs even during inference. This dynamic adaptation results in performance gains and reduces complexity in deep neural network models. We demonstrate the effectiveness of our method on 3D medical imaging datasets from MedMNIST3D as well as two ultrasound video datasets for classification tasks, one of them being a challenging in-house dataset collected under real-world clinical conditions. 

**Abstract (ZH)**: 虽然深度神经网络在性能上取得了显著进展，但这些改进往往伴随着计算复杂度和成本的增加。在许多情况下，如3D体数据或视频分类任务中，由于固有的冗余性，并非所有切片或帧都是必需的。为解决这一问题，我们提出了一种新颖的学习可变采样框架，可以集成到任何神经网络架构中。由于采样操作非可微，直接将其适配到深度学习模型中面临重大挑战。尽管一些工作提出使用Gumbel-max技巧来克服非可微性问题，但它们在关键方面存在不足：仅任务适配而非输入适配。一旦采样机制被学习，它将保持静态，不针对不同的输入进行调整，使其不适合实际应用。为解决这个问题，我们提出了一种注意力引导的采样模块，即使在推理过程中也能适应输入。这种动态适应性在深度神经网络模型中带来了性能提升并降低了复杂度。我们在MedMNIST3D的3D医学成像数据集以及两个超声视频数据集上的分类任务中验证了该方法的有效性，其中一个数据集是基于实际临床条件收集的具有挑战性的内部数据集。 

---
# LLM-REVal: Can We Trust LLM Reviewers Yet? 

**Title (ZH)**: LLM-REVal: 我们能信任LLM评审员吗？ 

**Authors**: Rui Li, Jia-Chen Gu, Po-Nien Kung, Heming Xia, Junfeng liu, Xiangwen Kong, Zhifang Sui, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12367)  

**Abstract**: The rapid advancement of large language models (LLMs) has inspired researchers to integrate them extensively into the academic workflow, potentially reshaping how research is practiced and reviewed. While previous studies highlight the potential of LLMs in supporting research and peer review, their dual roles in the academic workflow and the complex interplay between research and review bring new risks that remain largely underexplored. In this study, we focus on how the deep integration of LLMs into both peer-review and research processes may influence scholarly fairness, examining the potential risks of using LLMs as reviewers by simulation. This simulation incorporates a research agent, which generates papers and revises, alongside a review agent, which assesses the submissions. Based on the simulation results, we conduct human annotations and identify pronounced misalignment between LLM-based reviews and human judgments: (1) LLM reviewers systematically inflate scores for LLM-authored papers, assigning them markedly higher scores than human-authored ones; (2) LLM reviewers persistently underrate human-authored papers with critical statements (e.g., risk, fairness), even after multiple revisions. Our analysis reveals that these stem from two primary biases in LLM reviewers: a linguistic feature bias favoring LLM-generated writing styles, and an aversion toward critical statements. These results highlight the risks and equity concerns posed to human authors and academic research if LLMs are deployed in the peer review cycle without adequate caution. On the other hand, revisions guided by LLM reviews yield quality gains in both LLM-based and human evaluations, illustrating the potential of the LLMs-as-reviewers for early-stage researchers and enhancing low-quality papers. 

**Abstract (ZH)**: 大型语言模型的快速进步激发了研究人员将其广泛集成到学术工作流程中， potentially 重塑研究实践和审查方式。虽然以往研究表明大型语言模型在支持研究和同行评审方面的潜力，但它们在学术工作流程中的双重角色及其与研究和审查之间的复杂互动所带来的新风险仍 largely 未被充分探索。在本研究中，我们关注大型语言模型如何深度集成到同行评审和研究过程中，可能影响学术公正性，并通过模拟研究 LLM 作为评审人时的潜在风险。该模拟包括一个研究代理生成论文并修订，以及一个评审代理评估提交的论文。基于模拟结果，我们进行人工标注，并发现 LLM 基础的评审与人类判断之间存在显著不一致：（1）LLM 评审人系统性地提高由 LLM 生成的论文评分，给予它们显著高于人类作者论文的评分；（2）LLM 评审人持续性地低估包含批评性陈述（如风险、公平性）的人类作者论文，即使经过多次修订也是如此。我们的分析揭示了这些不一致的主要来源是两个 LLM 评审人的偏见：对 LLM 生成写作风格的语言特征偏爱，以及对批评性陈述的规避倾向。这些结果突显了在无需足够谨慎的情况下部署大型语言模型进行同行评审所面临的风险和公平性问题。另一方面，遵循 LLM 评审的修订提高了 LLM 生成和人类评审的质量，展示了 LLM 作为评审人对初级研究人员的潜力，并促进质量较低论文的改进。 

---
# (R)evolution of Programming: Vibe Coding as a Post-Coding Paradigm 

**Title (ZH)**: 编程的(R)evolution: Vibe Coding作为后编码范式 

**Authors**: Kevin Krings, Nino S. Bohn, Thomas Ludwig  

**Link**: [PDF](https://arxiv.org/pdf/2510.12364)  

**Abstract**: Recent advancements in generative artificial intelligence (GenAI), particularly large language models, have introduced new possibilities for software development practices. In our paper we investigate the emerging Vibe Coding (VC) paradigm that emphasizes intuitive, affect-driven, and improvisational interactions between developers and AI systems. Building upon the discourse of End-User Development (EUD), we explore how VC diverges from conventional programming approaches such as those supported by tools like GitHub Copilot. Through five semi-structured interview sessions with ten experienced software practitioners, we identify five thematic dimensions: creativity, sustainability, the future of programming, collaboration, and criticism. Our analysis conceptualizes VC within the metaphor of co-drifting, contrasting it with the prevalent co-piloting perspective of AI-assisted development. We argue that VC reconfigures the developers role, blurring boundaries between professional and non-developers. While VC enables novel forms of expression and rapid prototyping, it also introduces challenges regarding reproducibility, scalability, and inclusivity. We propose that VC represents a meaningful shift in programming culture, warranting further investigation within human-computer interaction (HCI) and software engineering research. 

**Abstract (ZH)**: 近年来，生成型人工智能（GenAI），特别是大型语言模型，为软件开发实践带来了新的可能性。在本文中，我们探讨了新兴的 vibes 编码（VC）范式，该范式强调开发人员与AI系统之间直观、情感驱动和即兴的互动。基于用户中心开发（EUD）的讨论，我们探究了VC与GitHub Copilot等工具支持的传统编程方法之间的差异。通过与十名经验丰富的软件从业人员进行五次半结构化访谈，我们确定了五个主题维度：创造力、可持续性、编程的未来、协作和批判。我们的分析将VC的概念化为共驾的隐喻，将其与常见的共驾驶模式（即AI辅助开发）进行了对比。我们认为，VC重新配置了开发人员的角色，模糊了专业开发人员与非开发人员之间的界限。虽然VC能够促成新的表达形式和快速的原型设计，但也带来了可重复性、可扩展性和包容性方面的挑战。我们提出，VC代表了编程文化的重要转变，值得在人机交互（HCI）和软件工程研究中进行进一步探讨。 

---
# Finite-time Convergence Analysis of Actor-Critic with Evolving Reward 

**Title (ZH)**: 有限时间收敛性分析：随动奖励的演员-评论家方法 

**Authors**: Rui Hu, Yu Chen, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12334)  

**Abstract**: Many popular practical reinforcement learning (RL) algorithms employ evolving reward functions-through techniques such as reward shaping, entropy regularization, or curriculum learning-yet their theoretical foundations remain underdeveloped. This paper provides the first finite-time convergence analysis of a single-timescale actor-critic algorithm in the presence of an evolving reward function under Markovian sampling. We consider a setting where the reward parameters may change at each time step, affecting both policy optimization and value estimation. Under standard assumptions, we derive non-asymptotic bounds for both actor and critic errors. Our result shows that an $O(1/\sqrt{T})$ convergence rate is achievable, matching the best-known rate for static rewards, provided the reward parameters evolve slowly enough. This rate is preserved when the reward is updated via a gradient-based rule with bounded gradient and on the same timescale as the actor and critic, offering a theoretical foundation for many popular RL techniques. As a secondary contribution, we introduce a novel analysis of distribution mismatch under Markovian sampling, improving the best-known rate by a factor of $\log^2T$ in the static-reward case. 

**Abstract (ZH)**: 许多流行的实用强化学习（RL）算法通过奖励塑形、熵正则化或课程学习等技术使用演变的奖励函数，但其理论基础仍不成熟。本文提供了第一个在马尔可夫采样下，单时间尺度演员-评论家算法在存在演变奖励函数情况下的有限时间收敛分析。我们考虑奖励参数在每一时间步都可能发生变化，影响策略优化和价值估计的情况。在标准假设下，我们推导了演员和评论家误差的非退化边界。我们的结果显示，在奖励参数变化足够缓慢的情况下，可以实现$O(1/\sqrt{T})$的收敛率，与静态奖励的最佳已知率一致。当奖励通过具有有界梯度的梯度规则在与演员和评论家相同的时间尺度上更新时，这一速率得以保持，为许多流行的RL技术提供了理论基础。作为次要贡献，我们引入了一种关于马尔可夫采样下分布不匹配的新型分析，将静态奖励情况下最佳已知率提高了$\log^2T$的因素。 

---
# Simple Projection Variants Improve ColBERT Performance 

**Title (ZH)**: 简单的投影变体能提高ColBERT性能 

**Authors**: Benjamin Clavié, Sean Lee, Rikiya Takehi, Aamir Shakir, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2510.12327)  

**Abstract**: Multi-vector dense retrieval methods like ColBERT systematically use a single-layer linear projection to reduce the dimensionality of individual vectors. In this study, we explore the implications of the MaxSim operator on the gradient flows of the training of multi-vector models and show that such a simple linear projection has inherent, if non-critical, limitations in this setting. We then discuss the theoretical improvements that could result from replacing this single-layer projection with well-studied alternative feedforward linear networks (FFN), such as deeper, non-linear FFN blocks, GLU blocks, and skip-connections, could alleviate these limitations. Through the design and systematic evaluation of alternate projection blocks, we show that better-designed final projections positively impact the downstream performance of ColBERT models. We highlight that many projection variants outperform the original linear projections, with the best-performing variants increasing average performance on a range of retrieval benchmarks across domains by over 2 NDCG@10 points. We then conduct further exploration on the individual parameters of these projections block in order to understand what drives this empirical performance, highlighting the particular importance of upscaled intermediate projections and residual connections. As part of these ablation studies, we show that numerous suboptimal projection variants still outperform the traditional single-layer projection across multiple benchmarks, confirming our hypothesis. Finally, we observe that this effect is consistent across random seeds, further confirming that replacing the linear layer of ColBERT models is a robust, drop-in upgrade. 

**Abstract (ZH)**: Multi-向量密集检索方法如ColBERT通过单层线性投影系统地降低单个向量的维度。在本研究中，我们探讨了MaxSim操作符在多向量模型训练的梯度流动中的影响，并表明这种简单的线性投影在这种情况下具有内在的，尽管不是关键的限制。然后，我们讨论了用研究广泛的替代前向线性网络（FFN），如更深的非线性FFN块、GLU块和跳连，替换这种单层投影可能带来的理论改进，这些改进可能缓解这些限制。通过设计并系统评估替代投影块，我们表明设计更好的最终投影可正向影响ColBERT模型的下游性能。我们指出，许多投影变体性能优于原始线性投影，最佳变体在多个领域的一系列检索基准上将平均性能提高了超过2个NDCG@10点。作为进一步的探索，我们研究了这些投影块的个体参数，以了解其驱动实际性能的因素，强调了放大的中间投影和残差连接的特殊重要性。作为这些消融研究的一部分，我们还展示了在多个基准上，许多不足的投影变体仍然优于传统的单层投影，进一步证实了我们的假设。最后，我们观察到这种效果在随机种子上是一致的，进一步证实了用替代的FFN块替换ColBERT模型的线性层是一个稳健且即插即用的升级。 

---
# Causal Inspired Multi Modal Recommendation 

**Title (ZH)**: 因果驱动多模态推荐 

**Authors**: Jie Yang, Chenyang Gu, Zixuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12325)  

**Abstract**: Multimodal recommender systems enhance personalized recommendations in e-commerce and online advertising by integrating visual, textual, and user-item interaction data. However, existing methods often overlook two critical biases: (i) modal confounding, where latent factors (e.g., brand style or product category) simultaneously drive multiple modalities and influence user preference, leading to spurious feature-preference associations; (ii) interaction bias, where genuine user preferences are mixed with noise from exposure effects and accidental clicks. To address these challenges, we propose a Causal-inspired multimodal Recommendation framework. Specifically, we introduce a dual-channel cross-modal diffusion module to identify hidden modal confounders, utilize back-door adjustment with hierarchical matching and vector-quantized codebooks to block confounding paths, and apply front-door adjustment combined with causal topology reconstruction to build a deconfounded causal subgraph. Extensive experiments on three real-world e-commerce datasets demonstrate that our method significantly outperforms state-of-the-art baselines while maintaining strong interpretability. 

**Abstract (ZH)**: 多模态推荐系统通过整合视觉、文本和用户-物品交互数据来增强电子商务和在线广告中的个性化推荐。然而，现有方法往往忽视了两种关键偏差：（i）模态混杂，其中潜在因素（如品牌风格或产品类别）同时驱动多种模态并对用户偏好产生影响，导致虚假的特征-偏好关联；（ii）交互偏差，其中真实的用户偏好被曝光效应和随机点击的噪声所混杂。为解决这些挑战，我们提出了一种因果启发式多模态推荐框架。具体来说，我们引入了双通道跨模态扩散模块来识别隐藏的模态混杂因素，利用后门调整结合分层匹配和向量量化码本来阻断混杂路径，并应用前门调整结合因果拓扑重构来构建去混杂的因果子图。在三个真实世界的电子商务数据集上的广泛实验表明，我们的方法显著优于最先进的基线方法，同时保持较强的可解释性。 

---
# Deep SPI: Safe Policy Improvement via World Models 

**Title (ZH)**: Deep SPI: 安全策略改进借助世界模型 

**Authors**: Florent Delgrange, Raphael Avalos, Willem Röpke  

**Link**: [PDF](https://arxiv.org/pdf/2510.12312)  

**Abstract**: Safe policy improvement (SPI) offers theoretical control over policy updates, yet existing guarantees largely concern offline, tabular reinforcement learning (RL). We study SPI in general online settings, when combined with world model and representation learning. We develop a theoretical framework showing that restricting policy updates to a well-defined neighborhood of the current policy ensures monotonic improvement and convergence. This analysis links transition and reward prediction losses to representation quality, yielding online, "deep" analogues of classical SPI theorems from the offline RL literature. Building on these results, we introduce DeepSPI, a principled on-policy algorithm that couples local transition and reward losses with regularised policy updates. On the ALE-57 benchmark, DeepSPI matches or exceeds strong baselines, including PPO and DeepMDPs, while retaining theoretical guarantees. 

**Abstract (ZH)**: Safe政策改进（SPI）提供了对政策更新的理论控制，但现有的保证主要集中在离线的表格强化学习（RL）中。我们研究SPI在结合世界模型和表示学习的一般在线设置中的应用。我们发展了一个理论框架，表明将政策更新限制在当前政策的良好定义的邻域内可以确保单调改进和收敛。该分析将转变预测损失和奖励预测损失与表示质量联系起来，从而得到经典的离线RL文献中的SPI定理的在线“深度”类比。基于这些结果，我们提出了DeepSPI，这是一种原理上的在线策略算法，它将局部转变和奖励损失与正则化政策更新联系起来。在ALE-57基准测试中，DeepSPI能够达到或超过包括PPO和DeepMDPs在内的强 baseline，同时保持理论上的保证。 

---
# Chinese ModernBERT with Whole-Word Masking 

**Title (ZH)**: Chinese ModernBERT全词掩码 

**Authors**: Zeyu Zhao, Ningtao Wang, Xing Fu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12285)  

**Abstract**: Encoder-only Transformers have advanced along three axes -- architecture, data, and systems -- yielding Pareto gains in accuracy, speed, and memory efficiency. Yet these improvements have not fully transferred to Chinese, where tokenization and morphology differ markedly from English. We introduce Chinese ModernBERT, a from-scratch Chinese encoder that couples: (i) a hardware-aware 32k BPE vocabulary tailored to frequent Chinese affixes/compounds, lowering the embedding budget; (ii) whole-word masking (WWM) with a dynamic masking curriculum (30% -> 15%) to align task difficulty with training progress; (iii) a two-stage pre-training pipeline that extends the native context from 1,024 to 8,192 tokens using RoPE and alternating local/global attention; and (iv) a damped-cosine learning-rate schedule for stable long-horizon optimization. We pre-train on ~1.2T Chinese tokens from CCI3-HQ, CCI4 (Chinese), and Cosmopedia-Chinese. On CLUE, Chinese ModernBERT is competitive with strong Chinese encoders under a unified fine-tuning protocol. Under bf16 it achieves high long-sequence throughput while maintaining strong short-sequence speed, reflecting benefits from budget allocation and attention design. To probe retrieval-oriented quality, we add a small amount of open contrastive data: fine-tuning on SimCLUE (~3M pairs) improves further when adding T2Ranking (~2M), reaching 0.505 (Pearson) / 0.537 (Spearman) on the SimCLUE test set. Under this open-data setting, Chinese ModernBERT surpasses Qwen-0.6B-embedding on SimCLUE, suggesting a clear scaling path for STS with additional curated pairs. We will release tokenizer and weights to facilitate reproducible research. 

**Abstract (ZH)**: Encoder-only Transformers在架构、数据和系统方面取得了进展，实现了准确度、速度和内存效率的帕累托改进。然而，这些改进尚未完全应用于汉语，因为汉语的分词和词素特征与英语有明显差异。我们提出了汉语现代BERT，这是一种从零开始的汉语编码器，结合了：（i）硬件意识的32k BPE词表，针对频繁出现的汉语前缀/复合词，降低了嵌入预算；（ii）词汇整体掩蔽（WWM）与动态掩蔽课程（从30%降低到15%），以使任务难度与训练进度保持一致；（iii）扩展本地和全局注意力交替使用的预训练管道，将原生上下文从1,024词扩展到8,192词；（iv）阻尼余弦学习率计划以实现稳定的长期优化。我们使用CCI3-HQ、CCI4（汉语）和Cosmopedia-Chinese中的约1.2万亿个汉字进行预训练。在CLUE上，汉语现代BERT在统一微调协议下与强大的汉语编码器具有竞争力。在bf16模式下，它实现了高长序列吞吐量同时保持强大的短序列速度，反映了预算分配和注意力设计的益处。为了研究检索导向的质量，我们添加了一部分开放对比数据：在SimCLUE（约300万对）上微调后进一步添加T2Ranking（约200万对），在SimCLUE测试集上达到0.505（皮尔逊相关系数）/0.537（斯皮尔曼等级相关系数）。在这种开放数据设置下，汉语现代BERT超过了Qwen-0.6B-嵌入在SimCLUE上的表现，表明在额外整理对的支持下有明确的STS扩展路径。我们将发布分词器和权重以促进可再现研究。 

---
# Quantum Annealing for Staff Scheduling in Educational Environments 

**Title (ZH)**: 量子退火在教育环境中的人员调度中应用 

**Authors**: Alessia Ciacco, Francesca Guerriero, Eneko Osaba  

**Link**: [PDF](https://arxiv.org/pdf/2510.12278)  

**Abstract**: We address a novel staff allocation problem that arises in the organization of collaborators among multiple school sites and educational levels. The problem emerges from a real case study in a public school in Calabria, Italy, where staff members must be distributed across kindergartens, primary, and secondary schools under constraints of availability, competencies, and fairness. To tackle this problem, we develop an optimization model and investigate a solution approach based on quantum annealing. Our computational experiments on real-world data show that quantum annealing is capable of producing balanced assignments in short runtimes. These results provide evidence of the practical applicability of quantum optimization methods in educational scheduling and, more broadly, in complex resource allocation tasks. 

**Abstract (ZH)**: 一种新型的学校多校区、多层级合作人员分配问题及其解决方案研究 

---
# TFGA-Net: Temporal-Frequency Graph Attention Network for Brain-Controlled Speaker Extraction 

**Title (ZH)**: TFGA-Net：用于脑控语音提取的时频图注意网络 

**Authors**: Youhao Si, Yuan Liao, Qiushi Han, Yuhang Yang, Rui Dai, Liya Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12275)  

**Abstract**: The rapid development of auditory attention decoding (AAD) based on electroencephalography (EEG) signals offers the possibility EEG-driven target speaker extraction. However, how to effectively utilize the target-speaker common information between EEG and speech remains an unresolved problem. In this paper, we propose a model for brain-controlled speaker extraction, which utilizes the EEG recorded from the listener to extract the target speech. In order to effectively extract information from EEG signals, we derive multi-scale time--frequency features and further incorporate cortical topological structures that are selectively engaged during the task. Moreover, to effectively exploit the non-Euclidean structure of EEG signals and capture their global features, the graph convolutional networks and self-attention mechanism are used in the EEG encoder. In addition, to make full use of the fused EEG and speech feature and preserve global context and capture speech rhythm and prosody, we introduce MossFormer2 which combines MossFormer and RNN-Free Recurrent as separator. Experimental results on both the public Cocktail Party and KUL dataset in this paper show that our TFGA-Net model significantly outper-forms the state-of-the-art method in certain objective evaluation metrics. The source code is available at: this https URL. 

**Abstract (ZH)**: 基于脑电图信号的听觉注意力解码快速发展为通过脑电图驱动的目标说话人提取提供了可能性。然而，如何有效利用EEG和语音之间的目标说话人公共信息仍然是一个未 resolves 的问题。本文提出了一种脑控制说话人提取模型，利用听众记录的脑电图提取目标语音。为了有效从EEG信号中提取信息，我们推导出了多尺度时频特征，并进一步结合了在任务中选择性参与的大脑皮层拓扑结构。此外，为了解决EEG信号的非欧几里得结构并通过捕获其全局特征来充分利用这些信息，我们在EEG编码器中使用了图卷积网络和自我注意力机制。此外，为充分利用融合后的EEG和语音特征并保持全局上下文和捕捉语音节奏和语调，我们引入了结合MossFormer和RNN-Free Recurrent的MossFormer2作为分离器。本文在公开的Cocktail Party和KUL数据集上的实验结果表明，我们的TFGA-Net模型在某些客观评价指标上显著优于最新方法。源代码可在以下链接获取：this https URL。 

---
# HiLoRA: Adaptive Hierarchical LoRA Routing for Training-Free Domain Generalization 

**Title (ZH)**: HiLoRA：自适应分层LoRA路由在无训练领域泛化的应用 

**Authors**: Ziyi Han, Huanyu Wang, Zeyu Zhang, Xiangxiang Dai, Xutong Liu, John C.S. Lui  

**Link**: [PDF](https://arxiv.org/pdf/2510.12266)  

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as a widely used technique for adapting large language models (LLMs) to new domains, due to its modular design and broad availability on platforms such as HuggingFace. This availability has motivated efforts to reuse existing LoRAs for domain generalization.
However, existing methods often rely on explicit task labels or additional training, which are impractical for deployment. Moreover, they typically activate a fixed number of entire LoRA modules, leading to parameter redundancy or insufficiency that degrade performance.
In this paper, we propose \texttt{HiLoRA}, a training-free framework that performs adaptive hierarchical routing over LoRA pools. Drawing on structural properties of LoRA, we define rank-one components (ROCs), in which each rank parameter is regarded as an independent unit. For a given input sequence, \texttt{HiLoRA} first adaptively selects a subset of LoRAs and determines their ROC allocation based on Gaussian likelihoods at the sequence level. At the token level, it further refines routing by activating only the most informative ROCs.
We further provide theoretical guarantees that \texttt{HiLoRA} selects the most relevant LoRAs with high probability.
Extensive experiments show that \texttt{HiLoRA} achieves substantial improvements in domain generalization, with accuracy gains of up to {\small $55\%$} over state-of-the-art baselines, while maintaining comparable inference throughput. 

**Abstract (ZH)**: HiLoRA: 一种基于层次路由的无需训练的LoRA适配框架 

---
# Human-in-the-Loop Bandwidth Estimation for Quality of Experience Optimization in Real-Time Video Communication 

**Title (ZH)**: 基于人类在环带宽估算的实时视频通信体验质量优化 

**Authors**: Sami Khairy, Gabriel Mittag, Vishak Gopal, Ross Cutler  

**Link**: [PDF](https://arxiv.org/pdf/2510.12265)  

**Abstract**: The quality of experience (QoE) delivered by video conferencing systems is significantly influenced by accurately estimating the time-varying available bandwidth between the sender and receiver. Bandwidth estimation for real-time communications remains an open challenge due to rapidly evolving network architectures, increasingly complex protocol stacks, and the difficulty of defining QoE metrics that reliably improve user experience. In this work, we propose a deployed, human-in-the-loop, data-driven framework for bandwidth estimation to address these challenges. Our approach begins with training objective QoE reward models derived from subjective user evaluations to measure audio and video quality in real-time video conferencing systems. Subsequently, we collect roughly $1$M network traces with objective QoE rewards from real-world Microsoft Teams calls to curate a bandwidth estimation training dataset. We then introduce a novel distributional offline reinforcement learning (RL) algorithm to train a neural-network-based bandwidth estimator aimed at improving QoE for users. Our real-world A/B test demonstrates that the proposed approach reduces the subjective poor call ratio by $11.41\%$ compared to the baseline bandwidth estimator. Furthermore, the proposed offline RL algorithm is benchmarked on D4RL tasks to demonstrate its generalization beyond bandwidth estimation. 

**Abstract (ZH)**: 视频会议系统中提供的体验质量（QoE）受到准确估计发送者和接收者之间时变可用带宽的影响。由于网络架构的快速演变、日益复杂的协议栈以及定义能够可靠提高用户体验的QoE指标的难度，实时通信中的带宽估计仍然是一个开放的挑战。在此工作中，我们提出了一种部署的、包含人类反馈的数据驱动框架，以应对这些挑战。我们的方法首先通过对主观用户评估进行客观QoE奖励模型的训练，来测量实时视频会议系统中的音频和视频质量。随后，我们从实际的Microsoft Teams通话中收集了约100万条网络轨迹，其中包含客观QoE奖励，以构建带宽估计训练数据集。我们进而引入了一种新颖的分布式离线强化学习（RL）算法，以训练基于神经网络的带宽估计器，旨在提高用户体验。我们的真实世界A/B测试表明，所提出的方法将主观差通话比率降低了11.41%。此外，我们还将所提出的离线RL算法在D4RL任务上进行基准测试，以展示其超越带宽估计的泛化能力。 

---
# Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs 

**Title (ZH)**: 浅层鲁棒性，深层漏洞：医疗LLM的多轮评估 

**Authors**: Blazej Manczak, Eric Lin, Francisco Eiras, James O' Neill, Vaikkunth Mugunthan  

**Link**: [PDF](https://arxiv.org/pdf/2510.12255)  

**Abstract**: Large language models (LLMs) are rapidly transitioning into medical clinical use, yet their reliability under realistic, multi-turn interactions remains poorly understood. Existing evaluation frameworks typically assess single-turn question answering under idealized conditions, overlooking the complexities of medical consultations where conflicting input, misleading context, and authority influence are common. We introduce MedQA-Followup, a framework for systematically evaluating multi-turn robustness in medical question answering. Our approach distinguishes between shallow robustness (resisting misleading initial context) and deep robustness (maintaining accuracy when answers are challenged across turns), while also introducing an indirect-direct axis that separates contextual framing (indirect) from explicit suggestion (direct). Using controlled interventions on the MedQA dataset, we evaluate five state-of-the-art LLMs and find that while models perform reasonably well under shallow perturbations, they exhibit severe vulnerabilities in multi-turn settings, with accuracy dropping from 91.2% to as low as 13.5% for Claude Sonnet 4. Counterintuitively, indirect, context-based interventions are often more harmful than direct suggestions, yielding larger accuracy drops across models and exposing a significant vulnerability for clinical deployment. Further compounding analyses reveal model differences, with some showing additional performance drops under repeated interventions while others partially recovering or even improving. These findings highlight multi-turn robustness as a critical but underexplored dimension for safe and reliable deployment of medical LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）正迅速进入医疗临床应用，但其在现实得多轮交互中的可靠性尚不明确。现有的评估框架通常在理想条件下评估单轮问答，忽视了医疗咨询中常见的矛盾输入、误导性背景和权威影响的复杂性。我们引入了MedQA-Followup框架，用于系统评估医疗问答的多轮鲁棒性。我们的方法区分了浅层鲁棒性（抵御误导性初始背景）和深层鲁棒性（在回答被多次挑战时保持准确性），同时引入了一个间接-直接轴，将其区分为背景框架（间接）和明确建议（直接）。通过在MedQA数据集上进行受控干预，我们评估了五种最先进的LLM，并发现虽然模型在浅层干扰下表现尚可，但在多轮设置中却表现出严重的脆弱性，准确率从91.2%降至低至13.5%（Claude Sonnet 4）。出人意料的是，间接、基于背景的干预往往比直接建议更具害处，导致模型间更大的准确率下降，并揭示了临床部署中的显著脆弱性。进一步的分析揭示了模型之间的差异，有些模型在重复干预下表现出额外的性能下降，而另一些则部分恢复甚至有所提高。这些发现突出了多轮鲁棒性对于安全可靠部署医疗LLM的关键但尚未充分探索的维度。 

---
# Diffusion Models for Reinforcement Learning: Foundations, Taxonomy, and Development 

**Title (ZH)**: 扩散模型在强化学习中的应用：基础、分类与发展 

**Authors**: Changfu Xu, Jianxiong Guo, Yuzhu Liang, Haiyang Huang, Haodong Zou, Xi Zheng, Shui Yu, Xiaowen Chu, Jiannong Cao, Tian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12253)  

**Abstract**: Diffusion Models (DMs), as a leading class of generative models, offer key advantages for reinforcement learning (RL), including multi-modal expressiveness, stable training, and trajectory-level planning. This survey delivers a comprehensive and up-to-date synthesis of diffusion-based RL. We first provide an overview of RL, highlighting its challenges, and then introduce the fundamental concepts of DMs, investigating how they are integrated into RL frameworks to address key challenges in this research field. We establish a dual-axis taxonomy that organizes the field along two orthogonal dimensions: a function-oriented taxonomy that clarifies the roles DMs play within the RL pipeline, and a technique-oriented taxonomy that situates implementations across online versus offline learning regimes. We also provide a comprehensive examination of this progression from single-agent to multi-agent domains, thereby forming several frameworks for DM-RL integration and highlighting their practical utility. Furthermore, we outline several categories of successful applications of diffusion-based RL across diverse domains, discuss open research issues of current methodologies, and highlight key directions for future research to advance the field. Finally, we summarize the survey to identify promising future development directions. We are actively maintaining a GitHub repository (this https URL) for papers and other related resources to apply DMs for RL. 

**Abstract (ZH)**: 基于扩散模型的强化学习综合调研：从单智能体到多智能体领域的进展与应用 

---
# PromptLocate: Localizing Prompt Injection Attacks 

**Title (ZH)**: PromptLocate: 定位提示注入攻击 

**Authors**: Yuqi Jia, Yupei Liu, Zedian Shao, Jinyuan Jia, Neil Gong  

**Link**: [PDF](https://arxiv.org/pdf/2510.12252)  

**Abstract**: Prompt injection attacks deceive a large language model into completing an attacker-specified task instead of its intended task by contaminating its input data with an injected prompt, which consists of injected instruction(s) and data. Localizing the injected prompt within contaminated data is crucial for post-attack forensic analysis and data recovery. Despite its growing importance, prompt injection localization remains largely unexplored. In this work, we bridge this gap by proposing PromptLocate, the first method for localizing injected prompts. PromptLocate comprises three steps: (1) splitting the contaminated data into semantically coherent segments, (2) identifying segments contaminated by injected instructions, and (3) pinpointing segments contaminated by injected data. We show PromptLocate accurately localizes injected prompts across eight existing and eight adaptive attacks. 

**Abstract (ZH)**: Prompt注入攻击通过在其输入数据中插入注入提示，使大型语言模型完成攻击者指定的任务而非预期任务。定位被注入提示污染的数据对于攻击后的法医分析和数据恢复至关重要。尽管其重要性日益增加，但注入提示的定位仍然缺乏探索。在本文中，我们通过提出PromptLocate方法来填补这一空白，这是首个用于定位注入提示的方法。PromptLocate包括三个步骤：（1）将被污染的数据划分为语义上连贯的片段，（2）识别被注入指令污染的片段，以及（3）定位被注入数据污染的片段。我们展示了PromptLocate在八个现有攻击和八个自适应攻击中准确地定位了注入提示。 

---
# MoRA: On-the-fly Molecule-aware Low-Rank Adaptation Framework for LLM-based Multi-Modal Molecular Assistant 

**Title (ZH)**: MoRA：基于LLM的多模态分子助手的分子意识低秩适应框架（增量学习） 

**Authors**: Tao Yin, Xiaohong Zhang, Jiacheng Zhang, Li Huang, Zhibin Zhang, Yuansong Zeng, Jin Xie, Meng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.12245)  

**Abstract**: Effectively integrating molecular graph structures with Large Language Models (LLMs) is a key challenge in drug discovery. Most existing multi-modal alignment methods typically process these structures by fine-tuning the LLM or adding a static adapter simultaneously. However, these approaches have two main limitations: (1) it optimizes a shared parameter space across all molecular inputs, limiting the model's ability to capture instance-specific structural features; and (2) fine-tuning the LLM for molecular tasks can lead to catastrophic forgetting, undermining its general reasoning capabilities. In this paper, instead of static task-oriented adaptation, we propose an instance-specific parameter space alignment approach for each molecule on-the-fly. To this end, we introduce Molecule-aware Low-Rank Adaptation (MoRA) that produces a unique set of low-rank adaptation weights for each input molecular graph. These weights are then dynamically injected into a frozen LLM, allowing the model to adapt its reasoning to the structure of each molecular input, while preserving the LLM's core knowledge. Extensive experiments demonstrate that on key molecular tasks, such as chemical reaction prediction and molecular captioning, MoRA's instance-specific dynamic adaptation outperforms statically adapted baselines, including a 14.1% relative improvement in reaction prediction exact match and a 22% reduction in error for quantum property prediction. The code is available at this https URL. 

**Abstract (ZH)**: 有效地将分子图结构与大型语言模型结合是药物发现中的一个重要挑战。现有的多模态对齐方法通常通过微调大型语言模型或同时添加静态适配器来处理这些结构。然而，这些方法存在两个主要局限性：（1）它在所有分子输入之间共享一个参数空间，限制了模型捕捉实例特定结构特征的能力；（2）为分子任务微调大型语言模型可能导致灾难性遗忘，削弱其通用推理能力。在本文中，我们提出了一种针对每个分子实例特定参数空间对齐的方法。为此，我们引入了分子感知的低秩适配（MoRA），为每个输入分子图生成一组独特的低秩适配权重。这些权重随后动态注入冻结的大型语言模型中，允许模型根据每个分子输入的结构进行适应，同时保留大型语言模型的核心知识。广泛实验表明，在化学反应预测和分子描述等关键分子任务中，MoRA的实例特定动态适应优于静态适应基线，反应预测完全匹配提高了14.1%，量子性质预测误差减少了22%。代码可在以下链接获取。 

---
# Analysing Moral Bias in Finetuned LLMs through Mechanistic Interpretability 

**Title (ZH)**: 通过机制可解释性分析微调后的大型语言模型中的道德偏见 

**Authors**: Bianca Raimondi, Daniela Dalbagno, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2510.12229)  

**Abstract**: Large language models (LLMs) have been shown to internalize human-like biases during finetuning, yet the mechanisms by which these biases manifest remain unclear. In this work, we investigated whether the well-known Knobe effect, a moral bias in intentionality judgements, emerges in finetuned LLMs and whether it can be traced back to specific components of the model. We conducted a Layer-Patching analysis across 3 open-weights LLMs and demonstrated that the bias is not only learned during finetuning but also localized in a specific set of layers. Surprisingly, we found that patching activations from the corresponding pretrained model into just a few critical layers is sufficient to eliminate the effect. Our findings offer new evidence that social biases in LLMs can be interpreted, localized, and mitigated through targeted interventions, without the need for model retraining. 

**Abstract (ZH)**: 大型语言模型在微调过程中表现出的人类偏见机制尚不明确：Knobe效应在微调大型语言模型中的表现及其追溯分析 

---
# HALF: Harm-Aware LLM Fairness Evaluation Aligned with Deployment 

**Title (ZH)**: HALF: 意外风险意识的大语言模型公平性评估与部署对齐 

**Authors**: Ali Mekky, Omar El Herraoui, Preslav Nakov, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12217)  

**Abstract**: Large language models (LLMs) are increasingly deployed across high-impact domains, from clinical decision support and legal analysis to hiring and education, making fairness and bias evaluation before deployment critical. However, existing evaluations lack grounding in real-world scenarios and do not account for differences in harm severity, e.g., a biased decision in surgery should not be weighed the same as a stylistic bias in text summarization. To address this gap, we introduce HALF (Harm-Aware LLM Fairness), a deployment-aligned framework that assesses model bias in realistic applications and weighs the outcomes by harm severity. HALF organizes nine application domains into three tiers (Severe, Moderate, Mild) using a five-stage pipeline. Our evaluation results across eight LLMs show that (1) LLMs are not consistently fair across domains, (2) model size or performance do not guarantee fairness, and (3) reasoning models perform better in medical decision support but worse in education. We conclude that HALF exposes a clear gap between previous benchmarking success and deployment readiness. 

**Abstract (ZH)**: HARM-AWARE LLM FAIRNESS (HALF): A DEPLOYMENT-ALIGNED FRAMEWORK FOR ASSESSING MODEL BIAS AND WEIGHING OUTCOMES BY HARM SEVERITY 

---
# DE3S: Dual-Enhanced Soft-Sparse-Shape Learning for Medical Early Time-Series Classification 

**Title (ZH)**: DE3S: 双增强软稀疏形状学习在医疗早期时间序列分类中的应用 

**Authors**: Tao Xie, Zexi Tan, Haoyi Xiao, Binbin Sun, Yiqun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12214)  

**Abstract**: Early time-series classification (ETSC) in medical applications is crucial for time-sensitive scenarios such as sepsis prediction in intensive care units (ICUs), where a large number of deaths are caused by delayed prediction. ETSC can significantly improve ICU resource utilization efficiency and healthcare precision. However, it faces conflicting goals of accuracy and earliness, with existing methods often trading one for the other, struggling to capture subtle early-stage patterns due to weak initial signals and class imbalance. The key to solve these challenges is to find shapelets, which are discriminative subsequences (or shapes) with high interpretability in time-series classification. This paper proposes Dual-Enhanced Soft-Sparse-Shape Learning for Medical Early Time-Series Classification (DE3S), which introduces a novel Dual-Enhanced Soft-Shape Learning framework to figure out shapelets precisely through three innovations: (1) a comprehensive dual-enhancement strategy combines traditional temporal augmentation with attention-based global temporal enhancement for robust representation learning, (2) an attention-score-based soft shapelet sparsification mechanism dynamically preserves discriminative patterns while aggregating less important shapelets into representative tokens, and (3) a dual-path Mixture of Experts Network (MoE) and Inception modules fusion architecture where MoE performs local learning within shapelets and multi-scale Inception modules capture global patterns across shapelets. The framework employs weighted cross-entropy loss for class imbalance handling and demonstrates robustness on subject-consistency datasets. Extensive experiments on six real-world medical datasets show state-of-the-art performance, with ablation studies confirming component efficacy. 

**Abstract (ZH)**: 医学早期时间序列分类中的双增强软稀疏形状学习（DE3S） 

---
# Revisiting Meta-Learning with Noisy Labels: Reweighting Dynamics and Theoretical Guarantees 

**Title (ZH)**: 重访带有噪声标签的元学习：加权动态及其理论保证 

**Authors**: Yiming Zhang, Chester Holtz, Gal Mishne, Alex Cloninger  

**Link**: [PDF](https://arxiv.org/pdf/2510.12209)  

**Abstract**: Learning with noisy labels remains challenging because over-parameterized networks memorize corrupted supervision. Meta-learning-based sample reweighting mitigates this by using a small clean subset to guide training, yet its behavior and training dynamics lack theoretical understanding. We provide a rigorous theoretical analysis of meta-reweighting under label noise and show that its training trajectory unfolds in three phases: (i) an alignment phase that amplifies examples consistent with a clean subset and suppresses conflicting ones; (ii) a filtering phase driving noisy example weights toward zero until the clean subset loss plateaus; and (iii) a post-filtering phase in which noise filtration becomes perturbation-sensitive. The mechanism is a similarity-weighted coupling between training and clean subset signals together with clean subset training loss contraction; in the post-filtering regime where the clean-subset loss is sufficiently small, the coupling term vanishes and meta-reweighting loses discriminatory power. Guided by this analysis, we propose a lightweight surrogate for meta-reweighting that integrates mean-centering, row shifting, and label-signed modulation, yielding more stable performance while avoiding expensive bi-level optimization. Across synthetic and real noisy-label benchmarks, our method consistently outperforms strong reweighting/selection baselines. 

**Abstract (ZH)**: 基于元学习的样本加权在标签噪声下的理论分析及应用 

---
# CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs 

**Title (ZH)**: CompoDistill：多模态LLM中组合理构推理的注意力精练 

**Authors**: Jiwan Kim, Kibum Kim, Sangwoo Seo, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.12184)  

**Abstract**: Recently, efficient Multimodal Large Language Models (MLLMs) have gained significant attention as a solution to their high computational complexity, making them more practical for real-world applications. In this regard, the knowledge distillation (KD) approach has emerged as a promising alternative, which transfers the rich visual and linguistic knowledge from a larger model (teacher) to a smaller model (student). However, we observe that existing KD methods struggle to effectively distill the teacher MLLM's rich visual perception abilities to the student, a challenge that has been largely overlooked in previous studies. Through a systematic analysis, we identify visual attention misalignment between student and teacher as the main cause of this issue. Based on this insight, we propose CompoDistill, a novel KD framework that explicitly aligns the student's visual attention with that of the teacher to enhance the student's visual perception abilities. Our extensive experiments show that CompoDistill significantly improves performance on compositional reasoning tasks that require visual perception abilities while maintaining strong performance on visual question answering tasks, as done in existing studies. Furthermore, CompoDistill demonstrates effectiveness with a more advanced backbone, highlighting its generalizability. 

**Abstract (ZH)**: 最近，高效的多模态大型语言模型（MLLMs）因其高计算复杂性而备受关注，使其在实际应用中更具实用性。在此背景下，知识蒸馏（KD）方法作为一种有前途的替代方案浮现出来，它将较大的模型（教师）丰富的视觉和语言知识转移到较小的模型（学生）上。然而，我们观察到现有的KD方法在将教师MLLM的丰富视觉感知能力有效地转移到学生上时存在困难，这一问题在先前的研究中并未被充分关注。通过对这一问题的系统分析，我们发现学生和教师之间的视觉注意力 misalignment 是主要原因。基于这一洞察，我们提出了 CompoDistill，这是一种新颖的KD框架，明确对齐学生和教师的视觉注意力，以增强学生在视觉感知方面的能力。广泛的实验证明，CompoDistill 在需要视觉感知能力的组合推理任务上显著提高了性能，同时在视觉问答任务上也保持了现有研究中的强大表现。此外，CompoDistill 在更先进的骨干网络上显示出有效性，这突显了它的通用性。 

---
# From Knowledge to Treatment: Large Language Model Assisted Biomedical Concept Representation for Drug Repurposing 

**Title (ZH)**: 从知识到治疗：大型语言模型辅助的生物医药概念表示在药物再利用中的应用 

**Authors**: Chengrui Xiang, Tengfei Ma, Xiangzheng Fu, Yiping Liu, Bosheng Song, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12181)  

**Abstract**: Drug repurposing plays a critical role in accelerating treatment discovery, especially for complex and rare diseases. Biomedical knowledge graphs (KGs), which encode rich clinical associations, have been widely adopted to support this task. However, existing methods largely overlook common-sense biomedical concept knowledge in real-world labs, such as mechanistic priors indicating that certain drugs are fundamentally incompatible with specific treatments. To address this gap, we propose LLaDR, a Large Language Model-assisted framework for Drug Repurposing, which improves the representation of biomedical concepts within KGs. Specifically, we extract semantically enriched treatment-related textual representations of biomedical entities from large language models (LLMs) and use them to fine-tune knowledge graph embedding (KGE) models. By injecting treatment-relevant knowledge into KGE, LLaDR largely improves the representation of biomedical concepts, enhancing semantic understanding of under-studied or complex indications. Experiments based on benchmarks demonstrate that LLaDR achieves state-of-the-art performance across different scenarios, with case studies on Alzheimer's disease further confirming its robustness and effectiveness. Code is available at this https URL. 

**Abstract (ZH)**: 药物再利用在加速复杂和罕见疾病治疗发现中发挥着关键作用。生物医学知识图谱（KGs），其中编码丰富的临床关联，已被广泛用于支持这一任务。然而，现有方法在很大程度上忽略了实际实验室中的常识性生物医学概念知识，例如机制先验表明某些药物与特定治疗从根本上是不兼容的。为了解决这一缺口，我们提出了一种大型语言模型辅助的药物再利用框架LLaDR，该框架提高了KG中生物医学概念的表示。具体而言，我们从大型语言模型（LLMs）中提取包含语义丰富治疗相关信息的生物医学实体表示，并使用它们来微调知识图嵌入（KGE）模型。通过将治疗相关的知识注入KGE，LLaDR显著提高了生物医学概念的表示，增强了对未研究或复杂症状的语义理解。基于基准的实验表明，LLaDR在不同场景下达到了最先进的性能，且阿尔茨海默病案例研究进一步证实了其稳健性和有效性。代码可供参考：this https URL。 

---
# Budget-constrained Active Learning to Effectively De-censor Survival Data 

**Title (ZH)**: 预算约束下的主动学习以有效解密生存数据 

**Authors**: Ali Parsaee, Bei Jiang, Zachary Friggstad, Russell Greiner  

**Link**: [PDF](https://arxiv.org/pdf/2510.12144)  

**Abstract**: Standard supervised learners attempt to learn a model from a labeled dataset. Given a small set of labeled instances, and a pool of unlabeled instances, a budgeted learner can use its given budget to pay to acquire the labels of some unlabeled instances, which it can then use to produce a model. Here, we explore budgeted learning in the context of survival datasets, which include (right) censored instances, where we know only a lower bound on an instance's time-to-event. Here, that learner can pay to (partially) label a censored instance -- e.g., to acquire the actual time for an instance [perhaps go from (3 yr, censored) to (7.2 yr, uncensored)], or other variants [e.g., learn about one more year, so go from (3 yr, censored) to either (4 yr, censored) or perhaps (3.2 yr, uncensored)]. This serves as a model of real world data collection, where follow-up with censored patients does not always lead to uncensoring, and how much information is given to the learner model during data collection is a function of the budget and the nature of the data itself. We provide both experimental and theoretical results for how to apply state-of-the-art budgeted learning algorithms to survival data and the respective limitations that exist in doing so. Our approach provides bounds and time complexity asymptotically equivalent to the standard active learning method BatchBALD. Moreover, empirical analysis on several survival tasks show that our model performs better than other potential approaches on several benchmarks. 

**Abstract (ZH)**: 预算化学习在生存数据中的应用与其限制研究 

---
# Credal Transformer: A Principled Approach for Quantifying and Mitigating Hallucinations in Large Language Models 

**Title (ZH)**: 信念变换器：一种衡量和减轻大型语言模型幻觉的规范方法 

**Authors**: Shihao Ji, Zihui Song, Jiajie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12137)  

**Abstract**: Large Language Models (LLMs) hallucinate, generating factually incorrect yet confident assertions. We argue this stems from the Transformer's Softmax function, which creates "Artificial Certainty" by collapsing ambiguous attention scores into a single probability distribution, discarding uncertainty information at each layer. To fix this, we introduce the Credal Transformer, which replaces standard attention with a Credal Attention Mechanism (CAM) based on evidential theory. CAM produces a "credal set" (a set of distributions) instead of a single attention vector, with the set's size directly measuring model uncertainty. We implement this by re-conceptualizing attention scores as evidence masses for a Dirichlet distribution: sufficient evidence recovers standard attention, while insufficient evidence yields a diffuse distribution, representing ambiguity. Empirically, the Credal Transformer identifies out-of-distribution inputs, quantifies ambiguity, and significantly reduces confident errors on unanswerable questions by abstaining. Our contribution is a new architecture to mitigate hallucinations and a design paradigm that integrates uncertainty quantification directly into the model, providing a foundation for more reliable AI. 

**Abstract (ZH)**: 大型语言模型（LLMs）会产生幻觉，生成事实错误但充满自信的断言。我们认为这源于Transformer的Softmax函数，该函数通过将具有歧义的注意分数压缩成单个概率分布，逐层丢弃不确定性信息，从而创造了“人工确定性”。为解决这一问题，我们引入了Credal Transformer，其中用基于证据理论的可信注意力机制（CAM）替代了标准注意力机制。CAM生成一个“可信集合”（一组分布）而不是单个注意力向量，集合的大小直接测量模型的不确定性。我们通过将注意力分数重新概念化为Dirichlet分布的证据质量来实现这一点：充足的证据恢复标准注意力，而不足的证据导致一个弥散分布，表示不确定性。实验结果表明，Credal Transformer能够识别离分布输入、量化不确定性，并通过弃权大幅减少不可回答问题上的自信错误。我们的贡献是一种新的架构来减轻幻觉现象，并提供了一种将不确定性量化直接集成到模型设计中的设计理念，为更加可靠的AI奠定基础。 

---
# SafeMT: Multi-turn Safety for Multimodal Language Models 

**Title (ZH)**: SafeMT：多轮多模态语言模型安全性 

**Authors**: Han Zhu, Juntao Dai, Jiaming Ji, Haoran Li, Chengkun Cai, Pengcheng Wen, Chi-Min Chan, Boyuan Chen, Yaodong Yang, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12133)  

**Abstract**: With the widespread use of multi-modal Large Language models (MLLMs), safety issues have become a growing concern. Multi-turn dialogues, which are more common in everyday interactions, pose a greater risk than single prompts; however, existing benchmarks do not adequately consider this situation. To encourage the community to focus on the safety issues of these models in multi-turn dialogues, we introduce SafeMT, a benchmark that features dialogues of varying lengths generated from harmful queries accompanied by images. This benchmark consists of 10,000 samples in total, encompassing 17 different scenarios and four jailbreak methods. Additionally, we propose Safety Index (SI) to evaluate the general safety of MLLMs during conversations. We assess the safety of 17 models using this benchmark and discover that the risk of successful attacks on these models increases as the number of turns in harmful dialogues rises. This observation indicates that the safety mechanisms of these models are inadequate for recognizing the hazard in dialogue interactions. We propose a dialogue safety moderator capable of detecting malicious intent concealed within conversations and providing MLLMs with relevant safety policies. Experimental results from several open-source models indicate that this moderator is more effective in reducing multi-turn ASR compared to existed guard models. 

**Abstract (ZH)**: 随着多模态大型语言模型（MLLMs）的广泛应用，安全性问题已成为一个日益增长的担忧。针对更常见的多轮对话，其中的风险高于单个提示，但现有基准并未充分考虑这种情况。为了鼓励社区关注这些模型在多轮对话中的安全性问题，我们引入了SafeMT基准，该基准包含由有害查询生成的不同长度的对话并配以图像。该基准共计包含10,000个样本，涵盖了17种不同的场景和四种脱羁方法。此外，我们提出了安全性指数（SI）来评估MLLMs在对话期间的一般安全性。我们使用该基准对17个模型进行了安全性评估，并发现随着有害对话轮次的增加，这些模型遭受成功攻击的风险也会增加。这一观察结果表明，这些模型的安全机制不足以识别对话交互中的危险。我们提出了一个能够检测隐藏在对话中的恶意意图并为MLLMs提供相关安全策略的对话安全审查员。来自几个开源模型的实验结果表明，该审查员在减少多轮ASR方面比现有的防护模型更为有效。 

---
# Understanding the Modality Gap: An Empirical Study on the Speech-Text Alignment Mechanism of Large Speech Language Models 

**Title (ZH)**: 理解模态差距：大型语音语言模型的语音-文本对齐机制实证研究 

**Authors**: Bajian Xiang, Shuaijiang Zhao, Tingwei Guo, Wei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.12116)  

**Abstract**: End-to-end Large Speech Language Models (LSLMs) have demonstrated impressive conversational generation abilities, yet consistently fall short of traditional pipeline systems on semantic understanding benchmarks. In this work, we reveal through systematic experimentation that although LSLMs lose some text input performance after speech-text alignment training, the performance gap between speech and text inputs is more pronounced, which we refer to as the modality gap. To understand this gap, we analyze both coarse- and fine-grained text and speech representations. At the coarse-grained level, representations of speech and text in deeper layers are found to be increasingly aligned in direction (cosine similarity), while concurrently diverging in magnitude (Euclidean distance). We further find that representation similarity is strongly correlated with the modality gap. At the fine-grained level, a spontaneous token-level alignment pattern between text and speech representations is observed. Based on this, we introduce the Alignment Path Score to quantify token-level alignment quality, which exhibits stronger correlation with the modality gap. Building on these insights, we design targeted interventions on critical tokens through angle projection and length normalization. These strategies demonstrate the potential to improve correctness for speech inputs. Our study provides the first systematic empirical analysis of the modality gap and alignment mechanisms in LSLMs, offering both theoretical and methodological guidance for future optimization. 

**Abstract (ZH)**: 端到端大型语音语言模型（LSLMs）在会话生成方面表现出了令人印象深刻的 ability，但在语义理解基准测试中却始终无法超越传统的流水线系统。在此项工作中，通过系统的实验我们揭示了虽然在语音-文本对齐训练后 LSLMs 的文本输入性能有所下降，但语音输入与文本输入之间的性能差距更为明显，我们将这种差距称为模态差距。为了理解这种差距，我们对粗粒度和细粒度的文本和语音表示进行了分析。在粗粒度水平，深层层中的语音和文本表示在方向上逐渐趋于一致（余弦相似性），同时在大小上逐渐发散（欧几里得距离）。进一步发现，表示相似性与模态差距密切相关。在细粒度水平，发现了文本和语音表示之间的自发 token 级别对齐模式。基于此，我们引入了对齐路径得分来量化 token 级别对齐质量，这种得分与模态差距的相关性更强。基于这些见解，我们通过角度投影和长度归一化对关键 token 进行了有针对性的干预。这些策略显示出提高语音输入正确性的潜力。我们的研究提供了 LSLS 中模态差距和对齐机制的第一项系统性的实证分析，为未来的优化提供了理论和方法上的指导。 

---
# Chimera: State Space Models Beyond Sequences 

**Title (ZH)**: Chimera: 状态空间模型超越序列 

**Authors**: Aakash Lahoti, Tanya Marwah, Ratish Puduppully, Albert Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12111)  

**Abstract**: Transformer-based deep learning methods have become the standard approach for modeling diverse data such as sequences, images, and graphs. These methods rely on self-attention, which treats data as an unordered set of elements. This ignores the neighborhood structure or graph topology of the data and requires inductive biases--such as position embeddings in sequences and images, or random walks in graphs--to incorporate topology. However, designing such task-specific biases requires significant effort and can introduce side effects that hinder generalization. We introduce Chimera, a unified model that directly incorporates data topology in a principled way, removing the need for domain-specific biases. The key idea is that state space models--which naturally do not require position embeddings--can be generalized to capture any graph topology. Our experiments show that Chimera achieves strong performance across language, vision, and graph domains, outperforming BERT on GLUE by 0.7 points, ViT on ImageNet-1k by 2.6%, and all baselines on the Long Range Graph Benchmark. We further propose algorithmic optimizations to improve Chimera's efficiency: (1) for Directed Acyclic Graphs, Chimera can be implemented as a linear-time recurrence; (2) for general graphs, a simple mathematical relaxation achieves Transformer's quadratic complexity without domain-specific heuristics. These results validate Chimera's core contribution and support the idea that data topology is a powerful inductive bias across modalities. 

**Abstract (ZH)**: 基于Transformer的深度学习方法已成为建模序列、图像和图形等多样化数据的标准方法。这些方法依赖于自注意力机制，将数据视为无序元素集合。这种方法忽视了数据的邻域结构或图形拓扑，并需要诱导偏置——如序列和图像中的位置嵌入，或图形中的随机游走——来包含拓扑信息。然而，设计此类任务特定的偏置需要大量工作，并可能引入抑制泛化的副作用。我们提出Chimera，这是一种统一模型，可以直接以合乎原理的方式融入数据拓扑，从而消除领域特定偏置的需要。核心思想是，状态空间模型——自然不需要位置嵌入——可以推广以捕捉任何图形拓扑。我们的实验表明，Chimera在语言、视觉和图形领域均表现出色，在GLUE上优于BERT 0.7分，在ImageNet-1k上优于ViT 2.6%，在长范围图形基准上优于所有基线。我们还提出了算法优化以提高Chimera的效率：（1）对于有向无环图，Chimera可以实现为线性时间递归；（2）对于通用图形，一种简单的数学松弛在不使用领域特定启发式的情况下实现了Transformer的二次复杂性。这些结果验证了Chimera的核心贡献，并支持数据拓扑是一种强大的诱导偏置的想法，适用于各种模态。 

---
# Deep Associations, High Creativity: A Simple yet Effective Metric for Evaluating Large Language Models 

**Title (ZH)**: 深层次关联，高创造力：评价大规模语言模型的一个简单而有效的指标 

**Authors**: Ziliang Qiu, Renfen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12110)  

**Abstract**: The evaluation of LLMs' creativity represents a crucial research domain, though challenges such as data contamination and costly human assessments often impede progress. Drawing inspiration from human creativity assessment, we propose PACE, asking LLMs to generate Parallel Association Chains to Evaluate their creativity. PACE minimizes the risk of data contamination and offers a straightforward, highly efficient evaluation, as evidenced by its strong correlation with Chatbot Arena Creative Writing rankings (Spearman's $\rho = 0.739$, $p < 0.001$) across various proprietary and open-source models. A comparative analysis of associative creativity between LLMs and humans reveals that while high-performing LLMs achieve scores comparable to average human performance, professional humans consistently outperform LLMs. Furthermore, linguistic analysis reveals that both humans and LLMs exhibit a trend of decreasing concreteness in their associations, and humans demonstrating a greater diversity of associative patterns. 

**Abstract (ZH)**: LLMscreativity评估：一种基于并行关联链的方法 

---
# An AI-Based Behavioral Health Safety Filter and Dataset for Identifying Mental Health Crises in Text-Based Conversations 

**Title (ZH)**: 基于AI的行为健康安全过滤器及数据集：识别文本对话中的心理健康危机 

**Authors**: Benjamin W. Nelson, Celeste Wong, Matthew T. Silvestrini, Sooyoon Shin, Alanna Robinson, Jessica Lee, Eric Yang, John Torous, Andrew Trister  

**Link**: [PDF](https://arxiv.org/pdf/2510.12083)  

**Abstract**: Large language models often mishandle psychiatric emergencies, offering harmful or inappropriate advice and enabling destructive behaviors. This study evaluated the Verily behavioral health safety filter (VBHSF) on two datasets: the Verily Mental Health Crisis Dataset containing 1,800 simulated messages and the NVIDIA Aegis AI Content Safety Dataset subsetted to 794 mental health-related messages. The two datasets were clinician-labelled and we evaluated performance using the clinician labels. Additionally, we carried out comparative performance analyses against two open source, content moderation guardrails: OpenAI Omni Moderation Latest and NVIDIA NeMo Guardrails. The VBHSF demonstrated, well-balanced performance on the Verily Mental Health Crisis Dataset v1.0, achieving high sensitivity (0.990) and specificity (0.992) in detecting any mental health crises. It achieved an F1-score of 0.939, sensitivity ranged from 0.917-0.992, and specificity was >= 0.978 in identifying specific crisis categories. When evaluated against the NVIDIA Aegis AI Content Safety Dataset 2.0, VBHSF performance remained highly sensitive (0.982) and accuracy (0.921) with reduced specificity (0.859). When compared with the NVIDIA NeMo and OpenAI Omni Moderation Latest guardrails, the VBHSF demonstrated superior performance metrics across both datasets, achieving significantly higher sensitivity in all cases (all p < 0.001) and higher specificity relative to NVIDIA NeMo (p < 0.001), but not to OpenAI Omni Moderation Latest (p = 0.094). NVIDIA NeMo and OpenAI Omni Moderation Latest exhibited inconsistent performance across specific crisis types, with sensitivity for some categories falling below 0.10. Overall, the VBHSF demonstrated robust, generalizable performance that prioritizes sensitivity to minimize missed crises, a crucial feature for healthcare applications. 

**Abstract (ZH)**: 大型语言模型在处理心理危机方面常常出现错误，提供有害或不合适的建议，从而导致破坏性行为。本研究评估了Verily行为健康安全性过滤器（VBHSF）在两个数据集中表现：包含1800条模拟消息的Verily Mental Health Crisis Dataset和NVIDIA Aegis AI内容安全性数据集中的794条心理健康相关消息。两个数据集均由临床医生标注，我们使用临床医生标签评估性能。此外，我们还对VBHSF与两个开源内容审核护栏——OpenAI Omni Moderation Latest和NVIDIA NeMo Guardrails——进行了性能对比分析。VBHSF在Verily Mental Health Crisis Dataset v1.0中表现出良好的平衡性能，检测任何心理健康危机的敏感性为0.990、特异性为0.992。在识别特定危机类别时，VBHSF的F1分数为0.939，敏感性范围为0.917-0.992，特异性不低于0.978。当与NVIDIA Aegis AI Content Safety Dataset 2.0进行评估时，VBHSF的敏感性仍为0.982，准确率为0.921，但特异性降低至0.859。与NVIDIA NeMo和OpenAI Omni Moderation Latest护栏相比，VBHSF在两个数据集中均表现出更优的性能指标，在所有情况下敏感性显著更高（所有p < 0.001），与NVIDIA NeMo相比特异性更高（p < 0.001），但与OpenAI Omni Moderation Latest相比则无显著差异（p = 0.094）。NVIDIA NeMo和OpenAI Omni Moderation Latest在特定危机类型上表现不一致，一些类别的敏感性低于0.10。总体而言，VBHSF表现出稳健的一般化性能，优先考虑敏感性以避免错过危机，这是其在医疗保健应用中的一项关键特征。 

---
# Enhancing Neural Code Representation with Additional Context 

**Title (ZH)**: 增强神经代码表示的额外上下文方法 

**Authors**: Huy Nguyen, Christoph Treude, Patanamon Thongtanunam  

**Link**: [PDF](https://arxiv.org/pdf/2510.12082)  

**Abstract**: Automated program comprehension underpins many software engineering tasks, from code summarisation to clone detection. Recent deep learning models achieve strong results but typically rely on source code alone, overlooking contextual information such as version history or structural relationships. This limits their ability to capture how code evolves and operates. We conduct an empirical study on how enriching code representations with such contextual signals affects neural model performance on key comprehension tasks. Two downstream tasks, code clone detection and code summarisation, are evaluated using SeSaMe (1,679 Java methods) and CodeSearchNet (63,259 methods). Five representative models (CodeBERT, GraphCodeBERT, CodeT5, PLBART, ASTNN) are fine-tuned under code-only and context-augmented settings. Results show that context generally improves performance: version history consistently boosts clone detection (e.g., CodeT5 +15.92% F1) and summarisation (e.g., GraphCodeBERT +5.56% METEOR), while call-graph effects vary by model and task. Combining multiple contexts yields further gains (up to +21.48% macro-F1). Human evaluation on 100 Java snippets confirms that context-augmented summaries are significantly preferred for Accuracy and Content Adequacy (p <= 0.026; |delta| up to 0.55). These findings highlight the potential of contextual signals to enhance code comprehension and open new directions for optimising contextual encoding in neural SE models. 

**Abstract (ZH)**: 基于上下文的代码表示增强对程序理解任务的神经模型性能影响研究 

---
# A Review on Domain Adaption and Generative Adversarial Networks(GANs) 

**Title (ZH)**: _domain适应与生成对抗网络（GANs）综述_ 

**Authors**: Aashish Dhawan, Divyanshu Mudgal  

**Link**: [PDF](https://arxiv.org/pdf/2510.12075)  

**Abstract**: The major challenge in today's computer vision scenario is the availability of good quality labeled data. In a field of study like image classification, where data is of utmost importance, we need to find more reliable methods which can overcome the scarcity of data to produce results comparable to previous benchmark results. In most cases, obtaining labeled data is very difficult because of the high cost of human labor and in some cases impossible. The purpose of this paper is to discuss Domain Adaptation and various methods to implement it. The main idea is to use a model trained on a particular dataset to predict on data from a different domain of the same kind, for example - a model trained on paintings of airplanes predicting on real images of airplanes 

**Abstract (ZH)**: 当今计算机视觉领域的主要挑战是高质量标注数据的获取。在诸如图像分类这样的领域，数据至关重要，因此我们需要找到更可靠的方法，克服数据稀缺性，以产生可与以往基准结果相媲美的结果。在大多数情况下，获得标注数据非常困难，原因在于人力成本高，而在某些情况下则完全不可能。本文旨在讨论领域适应及其各种实现方法。主要思想是利用在特定数据集上训练的模型，对同一类别的不同领域数据进行预测，例如，一个在飞机绘画上训练的模型预测真实飞机图像。 

---
# MEASURE: Multi-scale Minimal Sufficient Representation Learning for Domain Generalization in Sleep Staging 

**Title (ZH)**: MEASURE: 多尺度最小充分表示学习在睡眠分期领域泛化的应用 

**Authors**: Sangmin Jo, Jee Seok Yoon, Wootaek Jeong, Kwanseok Oh, Heung-Il Suk  

**Link**: [PDF](https://arxiv.org/pdf/2510.12070)  

**Abstract**: Deep learning-based automatic sleep staging has significantly advanced in performance and plays a crucial role in the diagnosis of sleep disorders. However, those models often struggle to generalize on unseen subjects due to variability in physiological signals, resulting in degraded performance in out-of-distribution scenarios. To address this issue, domain generalization approaches have recently been studied to ensure generalized performance on unseen domains during training. Among those techniques, contrastive learning has proven its validity in learning domain-invariant features by aligning samples of the same class across different domains. Despite its potential, many existing methods are insufficient to extract adequately domain-invariant representations, as they do not explicitly address domain characteristics embedded within the unshared information across samples. In this paper, we posit that mitigating such domain-relevant attributes-referred to as excess domain-relevant information-is key to bridging the domain gap. However, the direct strategy to mitigate the domain-relevant attributes often overfits features at the high-level information, limiting their ability to leverage the diverse temporal and spectral information encoded in the multiple feature levels. To address these limitations, we propose a novel MEASURE (Multi-scalE minimAl SUfficient Representation lEarning) framework, which effectively reduces domain-relevant information while preserving essential temporal and spectral features for sleep stage classification. In our exhaustive experiments on publicly available sleep staging benchmark datasets, SleepEDF-20 and MASS, our proposed method consistently outperformed state-of-the-art methods. Our code is available at : this https URL 

**Abstract (ZH)**: 基于深度学习的自动睡眠阶段划分在性能上取得了显著进步，并在睡眠障碍诊断中扮演着重要角色。然而，这些模型往往难以在未见过的受试者上泛化，导致在分布外场景中的性能下降。为了解决这一问题，最近研究了领域泛化方法，以确保在训练过程中在未见过的领域中实现泛化性能。在这其中，对比学习已被证明可以通过在不同领域内对相同类别的样本进行对齐，学习到领域不变的特征。尽管对比学习具有潜力，但许多现有方法仍不足以提取充分的领域不变表征，因为它们未能明确处理跨样本未共享信息中的领域特征。在本文中，我们认为减轻称为多余领域相关信息的领域相关的属性是弥合领域差距的关键。然而，直接减轻领域相关属性的策略往往会过度拟合高层信息特征，限制了它们利用多层次中编码的时序和频谱信息的能力。为了解决这些局限性，我们提出了一种名为MEASURE（多尺度最小充分表示学习）的新框架，该框架有效减少了领域相关信息的同时，保留了用于睡眠阶段分类的重要时序和频谱特征。在对公开可用的睡眠阶段基准数据集SleepEDF-20和MASS进行的详尽实验中，我们提出的方法在所有评估指标上都优于现有方法。我们的代码可在以下链接获取：this https URL。 

---
# Your VAR Model is Secretly an Efficient and Explainable Generative Classifier 

**Title (ZH)**: 你的VAR模型实际上是高效的可解释生成分类器 

**Authors**: Yi-Chung Chen, David I. Inouye, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.12060)  

**Abstract**: Generative classifiers, which leverage conditional generative models for classification, have recently demonstrated desirable properties such as robustness to distribution shifts. However, recent progress in this area has been largely driven by diffusion-based models, whose substantial computational cost severely limits scalability. This exclusive focus on diffusion-based methods has also constrained our understanding of generative classifiers. In this work, we propose a novel generative classifier built on recent advances in visual autoregressive (VAR) modeling, which offers a new perspective for studying generative classifiers. To further enhance its performance, we introduce the Adaptive VAR Classifier$^+$ (A-VARC$^+$), which achieves a superior trade-off between accuracy and inference speed, thereby significantly improving practical applicability. Moreover, we show that the VAR-based method exhibits fundamentally different properties from diffusion-based methods. In particular, due to its tractable likelihood, the VAR-based classifier enables visual explainability via token-wise mutual information and demonstrates inherent resistance to catastrophic forgetting in class-incremental learning tasks. 

**Abstract (ZH)**: 基于视觉自回归模型的生成分类器：A-VARC$^+$及其特性研究 

---
# APCE: Adaptive Progressive Context Expansion for Long Context Processing 

**Title (ZH)**: 自适应渐进上下文扩展：长上下文处理 

**Authors**: Baisub Lee, Sanghyun Byun, Mohanad Odema, Jung Guack, Jacob Song, Woo Seong Chung  

**Link**: [PDF](https://arxiv.org/pdf/2510.12051)  

**Abstract**: Deploying useful Long-Context Transformer Models (LCTMs) requires addressing two key challenges: (1) A growing memory footprint due to quadratic self-attention and linear KV-cache scaling in memory as sequence length increases; (2) the ContextRot phenomena where empirical evidence suggests that transformer architecture's performance degrades with increasing context length. Given the shared dependency on the input, a natural question arises: Can we surgically select the most important input chunks for processing to synergistically (a) reduce the memory footprint, and (b) mitigate the ContextRot effects? In this paper, we answer this question in the affirmative for long-context summarization tasks. We propose APCE as a context-aware solution to select the most important input chunks through low-dimensional semantic similarity matching with the current query. By directly operating on the input, APCE decouples from strict dependency on underlying hardware or CUDA environments, promising a compatible solution scalable to different deployment systems. Our empirical evaluations have demonstrated superior or on-par summarization performance for APCE compared to the full dense baseline using a fraction (50%-70%) of the input sequence resulting in KV-cache and self-attention memory efficiency improvements. We hope our findings inspire further research on context-aware efficiency solutions for LCTMs geared towards other relevant long-context tasks. 

**Abstract (ZH)**: 部署有用的长上下文转换器模型（LCTMs）需要解决两个关键挑战：（1）由于二次自注意力和随序列长度增加而线性扩展的KV缓存导致的日益增长的内存占用；（2）上下文旋转现象，实证研究表明，随着上下文长度的增加，变压器架构的性能会下降。鉴于输入的共享依赖性，一个自然的问题是：我们是否可以通过手术方式选择最重要的输入片段进行处理，从而（a）减少内存占用，并且（b）减轻上下文旋转效应？在本文中，我们证明了APCE可以在长上下文总结任务中肯定地做到这一点。我们提出了一种基于上下文的认知解决方案APCE，通过低维语义相似性匹配当前查询来选择最重要的输入片段。通过对输入直接操作，APCE摆脱了对底层硬件或CUDA环境的严格依赖，提供了一种兼容性解决方案，可扩展到不同的部署系统。我们的实验评估表明，与使用完整密集基线相比，APCE使用输入序列的少量（50%-70%）片段实现了更好的或相当的总结性能，从而提高了KV缓存和自注意力的内存效率。我们希望我们的研究成果能够激励更多关于LCTMs面向其他相关长上下文任务的认知效率解决方案的研究。 

---
# Generative AI and Firm Productivity: Field Experiments in Online Retail 

**Title (ZH)**: 生成式AI与企业生产率：在线零售领域的实地试验 

**Authors**: Lu Fang, Zhe Yuan, Kaifu Zhang, Dante Donati, Miklos Sarvary  

**Link**: [PDF](https://arxiv.org/pdf/2510.12049)  

**Abstract**: We quantify the impact of Generative Artificial Intelligence (GenAI) on firm productivity through a series of large-scale randomized field experiments involving millions of users and products at a leading cross-border online retail platform. Over six months in 2023-2024, GenAI-based enhancements were integrated into seven consumer-facing business workflows. We find that GenAI adoption significantly increases sales, with treatment effects ranging from 0\% to 16.3\%, depending on GenAI's marginal contribution relative to existing firm practices. Because inputs and prices were held constant across experimental arms, these gains map directly into total factor productivity improvements. Across the four GenAI applications with positive effects, the implied annual incremental value is approximately \$5 per consumer-an economically meaningful impact given the retailer's scale and the early stage of GenAI adoption. The primary mechanism operates through higher conversion rates, consistent with GenAI reducing frictions in the marketplace and improving consumer experience. We also document substantial heterogeneity: smaller and newer sellers, as well as less experienced consumers, exhibit disproportionately larger gains. Our findings provide novel, large-scale causal evidence on the productivity effects of GenAI in online retail, highlighting both its immediate value and broader potential. 

**Abstract (ZH)**: 我们通过涉及数百万用户和产品的大型随机现场实验，量化生成式人工智能（GenAI）对公司在跨境在线零售平台上的生产力影响。2023-2024年，GenAI增强功能被整合到七个面向消费者的业务工作流程中。我们发现，GenAI的采用显著增加了销售额，治疗效果范围从0%到16.3%，具体取决于GenAI对现有公司实践的边际贡献。由于各实验组的投入和价格保持不变，这些增益直接映射到总要素生产力的改进上。在四个具有正向效果的GenAI应用中，估计的年度增量价值约为每消费者5美元——这一经济意义上具有重要意义的影响考虑到了零售商的规模和GenAI采用的早期阶段。主要机制通过提高转化率起作用，这与GenAI减少市场摩擦和改善消费者体验一致。我们还记录了显著的异质性：较小和较新的卖家以及不太经验丰富的消费者表现出更大的增益。我们的研究提供了关于在线零售中GenAI生产力影响的新型大规模因果证据，突显了其即时价值和更广泛的潜力。 

---
# Hierarchical Alignment: Surgical Fine-Tuning via Functional Layer Specialization in Large Language Models 

**Title (ZH)**: 层次对齐：大型语言模型中功能层专业化下的手术微调 

**Authors**: Yukun Zhang, Qi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.12044)  

**Abstract**: Existing alignment techniques for Large Language Models (LLMs), such as Direct Preference Optimization (DPO), typically treat the model as a monolithic entity, applying uniform optimization pressure across all layers. This approach overlooks the functional specialization within the Transformer architecture, where different layers are known to handle distinct tasks from syntax to abstract reasoning. In this paper, we challenge this one-size-fits-all paradigm by introducing Hierarchical Alignment, a novel method that applies targeted DPO to distinct functional blocks of a model's layers: local (syntax), intermediate (logic), and global (factuality). Through a series of controlled experiments on state-of-the-art models like Llama-3.1-8B and Qwen1.5-7B using LoRA for surgical fine-tuning, our results, evaluated by a powerful LLM-as-Judge, demonstrate significant and predictable improvements. Specifically, aligning the local layers (Local-Align) enhances grammatical fluency. More importantly, aligning the global layers (Global-Align) not only improves factual consistency as hypothesized but also proves to be the most effective strategy for enhancing logical coherence, outperforming all baselines. Critically, all hierarchical strategies successfully avoid the "alignment tax" observed in standard DPO, where gains in fluency come at the cost of degraded logical reasoning. These findings establish a more resource-efficient, controllable, and interpretable path for model alignment, highlighting the immense potential of shifting from monolithic optimization to structure-aware surgical fine-tuning to build more advanced and reliable LLMs. 

**Abstract (ZH)**: 现有的大型语言模型（LLMs）对齐技术，如直接偏好优化（DPO），通常将模型视为一个整体，对所有层施加均匀的优化压力。这种方法忽视了Transformer架构内部的功能专业化，不同的层已知负责从句法到抽象推理的不同任务。在本文中，我们通过引入层次对齐这一新颖方法挑战了一刀切的范式，该方法对模型各层中的特定功能块应用针对性的DPO：局部（句法）、中间（逻辑）和全局（事实性）。通过在如Llama-3.1-8B和Qwen1.5-7B等最先进的模型上进行一系列受控实验，并使用LoRA进行手术微调，我们的结果由强大的LLM作为评判者评估，显示出显著且可预测的改进。特别是，对局部层进行对齐（Local-Align）提高了语法流畅性。更重要的是，对全局层进行对齐（Global-Align）不仅如预期那样提高了事实一致性，而且证明是最有效的提高逻辑连贯性的策略，优于所有基线方法。至关重要的是，所有层次方法都成功避免了标准DPO中观察到的“对齐税收”，即流畅性的提高会以逻辑推理能力降低为代价。这些发现为模型对齐奠定了更高效、可控和可解释的路径，突显了从整体优化转向结构感知的手术微调以构建更先进和可靠的LLM的巨大潜力。 

---
# Multi-stage Prompt Refinement for Mitigating Hallucinations in Large Language Models 

**Title (ZH)**: 多阶段提示精炼以减轻大型语言模型中的幻觉 

**Authors**: Jung-Woo Shim, Yeong-Joon Ju, Ji-Hoon Park, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.12032)  

**Abstract**: Recent advancements in large language models (LLMs) have shown strong performance in natural language understanding and generation tasks. However, LLMs continue to encounter challenges with hallucinations, where models generate plausible but incorrect information. While several factors contribute to hallucinations, the impact of ill-formed prompts, prompts with ambiguous wording, incorrect grammar, or incomplete information, was relatively under explored. To address this, we introduce Multi-stage Prompt Refinement (MPR), a framework designed to systematically improve these ill-formed prompts across multiple stages. Each stage addresses specific errors such as punctuation, typographical mistakes, and misuse of key terms, using small language models (SLMs) fine-tuned for these tasks. MPR iteratively enhances the clarity of prompts with additional context and employs a self-reflection mechanism with ranking to prioritize the most relevant input. Experimental results on hallucination benchmarks show that prompts refined by MPR achieve over an 85~\% win rate compared to their original forms, demonstrating its effectiveness in reducing hallucinations and improving LLM output accuracy. Interestingly, we reveal that MPR can be combined with existing post-hoc hallucination mitigation frameworks, further enhancing its versatility. MPR provides a lightweight and adaptable solution for enhancing LLM reliability across various domains. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）在自然语言理解和生成任务中表现出强大的性能。然而，LLMs仍然面临幻觉问题的挑战，即模型生成看似合理但实际上错误的信息。虽然幻觉产生的因素多种多样，但不良提示（如语法错误、信息不完整、措辞模糊等）的影响相对未被深入探索。为解决这一问题，我们引入了多阶段提示 refinement（MPR）框架，旨在通过多个阶段系统地改进这些不良提示。每一阶段针对特定错误（如标点符号错误、拼写错误和术语误用）进行修正，采用专门fine-tuned的小型语言模型（SLMs）。MPR通过增加上下文信息逐步提升提示的清晰度，并采用自我反思机制与排序机制来优先处理最相关的输入。在幻觉基准测试上的实验结果表明，经过MPR改进的提示在85%以上的情况下优于其原始形式，证明了其在减少幻觉和提高LLM输出准确性方面的有效性。有趣的是，我们发现MPR可以与其他现有的事后幻觉缓解框架结合使用，进一步增强其灵活性。MPR提供了一种轻量级且适应性强的解决方案，用于提高不同领域中LLM的可靠性。 

---
# CPR: Mitigating Large Language Model Hallucinations with Curative Prompt Refinement 

**Title (ZH)**: CPR: 修正提示 refinement 减轻大型语言模型幻觉 

**Authors**: Jung-Woo Shim, Yeong-Joon Ju, Ji-Hoon Park, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.12029)  

**Abstract**: Recent advancements in large language models (LLMs) highlight their fluency in generating responses to diverse prompts. However, these models sometimes generate plausible yet incorrect ``hallucinated" facts, undermining trust. A frequent but often overlooked cause of such errors is the use of poorly structured or vague prompts by users, leading LLMs to base responses on assumed rather than actual intentions. To mitigate hallucinations induced by these ill-formed prompts, we introduce Curative Prompt Refinement (CPR), a plug-and-play framework for curative prompt refinement that 1) cleans ill-formed prompts, and 2) generates additional informative task descriptions to align the intention of the user and the prompt using a fine-tuned small language model. When applied to language models, we discover that CPR significantly increases the quality of generation while also mitigating hallucination. Empirical studies show that prompts with CPR applied achieves over a 90\% win rate over the original prompts without any external knowledge. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）突显了它们在生成对各种提示响应方面的流畅性。然而，这些模型有时会生成虽具说服力但错误的“虚构”事实，从而削弱了信任。造成此类错误的一个常见但经常被忽视的原因是用户使用的结构不良或含糊不清的提示，导致LLMs基于假设而不是实际意图来形成响应。为了减轻由这些不完善的提示引起的虚构现象，我们引入了Curative Prompt Refinement (CPR)框架，这是一种可插拔框架，用于修正不良的提示并生成额外的信息性任务描述以通过微调小型语言模型调整用户的意图与提示之间的关系。当应用于语言模型时，我们发现CPR显著提高生成质量同时减轻了虚构现象。实证研究表明，使用CPR的提示在无需任何外部知识的情况下，胜过原始提示的成功率超过90%。 

---
# PanoTPS-Net: Panoramic Room Layout Estimation via Thin Plate Spline Transformation 

**Title (ZH)**: PanoTPS-Net：基于薄板样条变换的全景房间布局估计 

**Authors**: Hatem Ibrahem, Ahmed Salem, Qinmin Vivian Hu, Guanghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11992)  

**Abstract**: Accurately estimating the 3D layout of rooms is a crucial task in computer vision, with potential applications in robotics, augmented reality, and interior design. This paper proposes a novel model, PanoTPS-Net, to estimate room layout from a single panorama image. Leveraging a Convolutional Neural Network (CNN) and incorporating a Thin Plate Spline (TPS) spatial transformation, the architecture of PanoTPS-Net is divided into two stages: First, a convolutional neural network extracts the high-level features from the input images, allowing the network to learn the spatial parameters of the TPS transformation. Second, the TPS spatial transformation layer is generated to warp a reference layout to the required layout based on the predicted parameters. This unique combination empowers the model to properly predict room layouts while also generalizing effectively to both cuboid and non-cuboid layouts. Extensive experiments on publicly available datasets and comparisons with state-of-the-art methods demonstrate the effectiveness of the proposed method. The results underscore the model's accuracy in room layout estimation and emphasize the compatibility between the TPS transformation and panorama images. The robustness of the model in handling both cuboid and non-cuboid room layout estimation is evident with a 3DIoU value of 85.49, 86.16, 81.76, and 91.98 on PanoContext, Stanford-2D3D, Matterport3DLayout, and ZInD datasets, respectively. The source code is available at: this https URL. 

**Abstract (ZH)**: 准确估计房间的3D布局是计算机视觉中的一个关键任务，潜在的应用包括机器人技术、增强现实和室内设计。本文提出了一种新型模型PanoTPS-Net，用于从单张全景图像估计房间布局。该模型结合卷积神经网络（CNN）和薄板样条（TPS）空间变换，其架构分为两个阶段：首先，卷积神经网络从输入图像中提取高级特征，使网络学习TPS变换的空间参数。其次，根据预测的参数生成TPS空间变换层，将参考布局变换为所需布局。这种独特的组合使模型能够正确预测房间布局，并且能够有效地泛化到规则和不规则布局。在公开数据集上的广泛实验和与最先进的方法的比较表明所提出方法的有效性。实验结果强调了该模型在房间布局估计中的准确性，并突出了TPS变换与全景图像之间的兼容性。模型在PanoContext、Stanford-2D3D、Matterport3DLayout和ZInD数据集上的鲁棒性估计分别达到3DIoU值85.49、86.16、81.76和91.98。源代码可在以下链接获取：this https URL。 

---
# Conjecturing: An Overlooked Step in Formal Mathematical Reasoning 

**Title (ZH)**: 猜想：形式化数学推理中被忽视的步骤 

**Authors**: Jasivan Alex Sivakumar, Philipp Borchert, Ronald Cardenas, Gerasimos Lampouras  

**Link**: [PDF](https://arxiv.org/pdf/2510.11986)  

**Abstract**: Autoformalisation, the task of expressing informal mathematical statements in formal language, is often viewed as a direct translation process. This, however, disregards a critical preceding step: conjecturing. Many mathematical problems cannot be formalised directly without first conjecturing a conclusion such as an explicit answer, or a specific bound. Since Large Language Models (LLMs) already struggle with autoformalisation, and the evaluation of their conjecturing ability is limited and often entangled within autoformalisation or proof, it is particularly challenging to understand its effect. To address this gap, we augment existing datasets to create ConjectureBench, and redesign the evaluation framework and metric specifically to measure the conjecturing capabilities of LLMs both as a distinct task and within the autoformalisation pipeline. Our evaluation of foundational models, including GPT-4.1 and DeepSeek-V3.1, reveals that their autoformalisation performance is substantially overestimated when the conjecture is accounted for during evaluation. However, the conjecture should not be assumed to be provided. We design an inference-time method, Lean-FIRe to improve conjecturing and autoformalisation, which, to the best of our knowledge, achieves the first successful end-to-end autoformalisation of 13 PutnamBench problems with GPT-4.1 and 7 with DeepSeek-V3.1. We demonstrate that while LLMs possess the requisite knowledge to generate accurate conjectures, improving autoformalisation performance requires treating conjecturing as an independent task, and investigating further how to correctly integrate it within autoformalisation. Finally, we provide forward-looking guidance to steer future research toward improving conjecturing, an overlooked step of formal mathematical reasoning. 

**Abstract (ZH)**: 自动形式化，即将非形式化的数学陈述表达为形式语言的过程，通常被视为一种直接的翻译过程。然而，这忽略了至关重要的一个前置步骤：猜想。许多数学问题在无法直接形式化之前，需要先提出一个结论，例如明确的答案或特定的界。由于大规模语言模型（LLMs）已经在自动形式化方面表现出色，而其猜想能力的评估通常与自动形式化或证明紧密交织在一起，因此理解其影响尤为具有挑战性。为解决这一差距，我们扩展了现有的数据集以创建ConjectureBench，并重新设计了评估框架和度量标准，以专门衡量LLMs在作为独立任务和自动形式化管道内的猜想能力。我们对包括GPT-4.1和DeepSeek-V3.1在内的基础模型的评估显示，当在评估过程中考虑猜想时，其自动形式化性能被大大高估了。然而，不应假设猜想会被提供。我们设计了一种推理时方法Lean-FIRe，以提高猜想和自动形式化的性能，在我们所知的情况下，该方法首次成功地使用GPT-4.1和DeepSeek-V3.1实现了PutnamBench 13个问题和7个问题的端到端自动形式化。我们表明，尽管LLMs具备生成准确猜想所需的知识，但提高自动形式化性能需要将猜想视为一个独立任务，并进一步探索如何正确将其整合到自动形式化中。最后，我们提供前瞻性的指导，以引导未来研究改进猜想，这一被忽视的数学形式推理步骤。 

---
# Learning Dynamics of VLM Finetuning 

**Title (ZH)**: VLM微调的学习动力学 

**Authors**: Jusheng Zhang, Kaitong Cai, Jing Yang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11978)  

**Abstract**: Preference-based finetuning of vision--language models (VLMs) is brittle: trivially wrong negatives inject uninformative gradients that destabilize training. We recast alignment as \textbf{learning-dynamics--aware optimization} and introduce \textbf{Cooling-Weighted DPO (CW-DPO)}, a two-stage recipe that explicitly models and exploits the training trajectory. \textbf{Stage 1} performs supervised finetuning with \textbf{gentle negatives}: \textbf{low-weight smoothed supervision} that regularizes the base policy and curbs overconfidence without explicit penalties. \textbf{Stage 2} applies a DPO objective in which the \textbf{negative term is scaled by a cooling weight} computed from the model's \textbf{average token log-probability} on each negative, suppressing uninformative gradients from easy or off-distribution samples while preserving signal from hard negatives. In practice, we emphasize \textbf{on-policy negatives} and allow \textbf{mixed negatives} by blending a controllable fraction of dataset negatives to maintain contrast freshness. Throughout, we instrument training with $\Delta\!\log p$ probes on positives and negatives as first-class signals for early stopping, curriculum design, and failure diagnosis. Across diverse VLM tasks, CW-DPO yields \textbf{more stable optimization}, \textbf{better calibration}, and \textbf{higher pairwise win-rates} than SFT-only and vanilla DPO, while \textbf{converging in fewer steps}. Ablations isolate the \textbf{cooling-weight mechanism} as the primary driver of these gains and show complementary benefits from mixing on-policy and dataset negatives. Taken together, our results show that \textbf{smoothing learning dynamics before cooling preferences} is a simple, general principle for robust VLM alignment. 

**Abstract (ZH)**: 基于偏好微调的视觉-语言模型（VLMs）训练易碎：显然错误的负样本注入无信息梯度， destabilize 训练。我们重新定义对齐为学习动力学感知优化，并引入冷却加权 DPO（CW-DPO），这是一种两阶段方法，明确建模并利用训练轨迹。第一阶段使用温和的负样本进行监督微调：低权重平滑监督，用于正则化基础策略并减少过信度，而无需明确的惩罚。第二阶段应用 DPO 目标，在该目标中，负项通过模型计算的平均标记对数概率进行缩放权重，抑制来自容易或离分布样本的无信息梯度，同时保留来自困难负样本的信号。实践中，我们强调在策略负样本上训练，并允许通过混合可控制比例的数据集负样本来维持对比新鲜度。在整个过程中，我们通过在正样本和负样本上使用 $\Delta\!\log p$ 探针作为首要信号进行早期停止、课程设计和失败诊断。在多种多样的 VLM 任务中，CW-DPO 在优化稳定性、校准准确性和两两胜率方面优于仅自编码器微调和标准 DPO，同时在更少的步骤内收敛。消融实验将冷却权重机制识别为主要驱动因素，并显示了在策略负样本和数据集负样本混合中的互补益处。综上所述，我们的结果表明，在冷却偏好之前平滑学习动力学是鲁棒 VLM 对齐的一个简单而通用的原则。 

---
# CTIArena: Benchmarking LLM Knowledge and Reasoning Across Heterogeneous Cyber Threat Intelligence 

**Title (ZH)**: CTIArena：跨异构网络威胁情报领域评估LLM知识与推理能力 

**Authors**: Yutong Cheng, Yang Liu, Changze Li, Dawn Song, Peng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.11974)  

**Abstract**: Cyber threat intelligence (CTI) is central to modern cybersecurity, providing critical insights for detecting and mitigating evolving threats. With the natural language understanding and reasoning capabilities of large language models (LLMs), there is increasing interest in applying them to CTI, which calls for benchmarks that can rigorously evaluate their performance. Several early efforts have studied LLMs on some CTI tasks but remain limited: (i) they adopt only closed-book settings, relying on parametric knowledge without leveraging CTI knowledge bases; (ii) they cover only a narrow set of tasks, lacking a systematic view of the CTI landscape; and (iii) they restrict evaluation to single-source analysis, unlike realistic scenarios that require reasoning across multiple sources. To fill these gaps, we present CTIArena, the first benchmark for evaluating LLM performance on heterogeneous, multi-source CTI under knowledge-augmented settings. CTIArena spans three categories, structured, unstructured, and hybrid, further divided into nine tasks that capture the breadth of CTI analysis in modern security operations. We evaluate ten widely used LLMs and find that most struggle in closed-book setups but show noticeable gains when augmented with security-specific knowledge through our designed retrieval-augmented techniques. These findings highlight the limitations of general-purpose LLMs and the need for domain-tailored techniques to fully unlock their potential for CTI. 

**Abstract (ZH)**: CTIArena：评估大语言模型在知识增强多源网络威胁情报分析中的性能 

---
# Direct Multi-Token Decoding 

**Title (ZH)**: 直接多令牌解码 

**Authors**: Xuan Luo, Weizhi Wang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.11958)  

**Abstract**: Decoder-only transformers have become the standard architecture for large language models (LLMs) due to their strong performance. Recent studies suggest that, in pre-trained LLMs, early, middle, and late layers may serve distinct roles: Early layers focus on understanding the input context, middle layers handle task-specific processing, and late layers convert abstract representations into output tokens. We hypothesize that once representations have been processed by the early and middle layers, the resulting hidden states may encapsulate sufficient information to support the generation of multiple tokens using only the late layers, eliminating the need to repeatedly traverse the early and middle layers. We refer to this inference paradigm as Direct Multi-Token Decoding (DMTD). Unlike speculative decoding, our method introduces no additional parameters, auxiliary routines, or post-generation verification. Despite being trained on a limited dataset, a fine-tuned DMTD Qwen3-4B model has already demonstrated promising results, achieving up to a 2x speedup with only minor performance loss. Moreover, as shown in our scaling analysis, its performance is expected to further improve with larger training datasets. 

**Abstract (ZH)**: 只解码器的变压器已成为大规模语言模型（LLMs）的标准架构，由于其出色的性能。近期研究表明，在预训练的LLMs中，早期、中期和晚期层可能各自承担不同的职责：早期层专注于理解输入语境，中期层处理任务特定的处理，晚期层将抽象表示转换为输出标记。我们假设一旦输入被早期和中期层处理后，产生的隐藏状态可能已经包含了利用晚期层生成多个标记所需的所有信息，从而避免了反复穿越早期和中期层的需要。我们称这一推理范式为直接多标记解码（DMTD）。与推测性解码不同，我们的方法不引入额外的参数、辅助程序或生成后的验证。尽管是在有限的数据集上训练，微调后的DMTD Qwen3-4B模型已经展现了令人鼓舞的结果，只牺牲了微小的性能损失就实现了高达2倍的速度提升。此外，如我们在缩放分析中所展示的，其性能预期随着更大规模训练数据集的增加而进一步提升。 

---
# Y-shaped Generative Flows 

**Title (ZH)**: Y形生成流 

**Authors**: Arip Asadulaev, Semyon Semenov, Abduragim Shtanchaev, Eric Moulines, Fakhri Karray, Martin Takac  

**Link**: [PDF](https://arxiv.org/pdf/2510.11955)  

**Abstract**: Modern continuous-time generative models often induce V-shaped transport: each sample travels independently along nearly straight trajectories from prior to data, overlooking shared structure. We introduce Y-shaped generative flows, which move probability mass together along shared pathways before branching to target-specific endpoints. Our formulation is based on novel velocity-powered transport cost with a sublinear exponent (between zero and one). this concave dependence rewards joint and fast mass movement. Practically, we instantiate the idea in a scalable neural ODE training objective. On synthetic, image, and biology datasets, Y-flows recover hierarchy-aware structure, improve distributional metrics over strong flow-based baselines, and reach targets with fewer integration steps. 

**Abstract (ZH)**: Y形生成流：共享路径上的联合概率质量传输以改善生成模型 

---
# Sculpting Latent Spaces With MMD: Disentanglement With Programmable Priors 

**Title (ZH)**: 用MMD塑造潜在空间：具可编程先验的分解学习 

**Authors**: Quentin Fruytier, Akshay Malhotra, Shahab Hamidi-Rad, Aditya Sant, Aryan Mokhtari, Sujay Sanghavi  

**Link**: [PDF](https://arxiv.org/pdf/2510.11953)  

**Abstract**: Learning disentangled representations, where distinct factors of variation are captured by independent latent variables, is a central goal in machine learning. The dominant approach has been the Variational Autoencoder (VAE) framework, which uses a Kullback-Leibler (KL) divergence penalty to encourage the latent space to match a factorized Gaussian prior. In this work, however, we provide direct evidence that this KL-based regularizer is an unreliable mechanism, consistently failing to enforce the target distribution on the aggregate posterior. We validate this and quantify the resulting entanglement using our novel, unsupervised Latent Predictability Score (LPS). To address this failure, we introduce the Programmable Prior Framework, a method built on the Maximum Mean Discrepancy (MMD). Our framework allows practitioners to explicitly sculpt the latent space, achieving state-of-the-art mutual independence on complex datasets like CIFAR-10 and Tiny ImageNet without the common reconstruction trade-off. Furthermore, we demonstrate how this programmability can be used to engineer sophisticated priors that improve alignment with semantically meaningful features. Ultimately, our work provides a foundational tool for representation engineering, opening new avenues for model identifiability and causal reasoning. 

**Abstract (ZH)**: 学习解耦表示，其中独立的潜在变量捕获不同的变化因子，是机器学习中的一个核心目标。尽管占主导地位的方法是变分自编码器（VAE）框架，该框架通过Kullback-Leibler（KL）散度惩罚项鼓励潜在空间匹配因子化的高斯先验，然而，在这项工作中，我们提供了直接证据表明，这种基于KL的正则化机制是不可靠的，一致地未能在联合后验上施加目标分布。我们使用新颖的无监督潜在可预测分数（LPS）验证这一点并量化由此产生的纠缠。为了应对这种失败，我们引入了可编程先验框架，该方法基于最大均值偏差（MMD）。我们的框架允许实践者明确塑造潜在空间，在CIFAR-10和Tiny ImageNet等复杂数据集上实现最先进的互不相关性，而无需常见的重构权衡。此外，我们展示了这种可编程性如何用于设计与语义含义特征对齐更为出色的先验。最终，我们的工作提供了一个基础工具，用于表示工程设计，开启了模型标识性和因果推理的新途径。 

---
# TopoAlign: A Framework for Aligning Code to Math via Topological Decomposition 

**Title (ZH)**: TopoAlign：一种通过拓扑分解对齐代码与数学的框架 

**Authors**: Yupei Li, Philipp Borchert, Gerasimos Lampouras  

**Link**: [PDF](https://arxiv.org/pdf/2510.11944)  

**Abstract**: Large Language Models (LLMs) excel at both informal and formal (e.g. Lean 4) mathematical reasoning but still struggle with autoformalisation, the task of transforming informal into formal mathematical statements. Autoformalisation helps pair the informal reasoning of LLMs with formal proof assistants which enable machine-verifiable generation and mitigate hallucinations. Yet, the performance of current Math LLMs is constrained by the scarcity of large-scale corpora, particularly those containing pairs of informal and formal statements. Although current models are trained to generate code from natural language instructions, structural and syntactic differences between these and formal mathematics limit effective transfer learning. We propose TopoAlign, a framework that unlocks widely available code repositories as training resources for Math LLMs. TopoAlign decomposes code into docstrings, main functions, and dependency functions, and reassembles these components into analogues that structurally mirror formal statements. This produces structurally aligned code data that can be used for training Math LLMs without requiring additional human annotation. We train two state-of-the-art models, DeepSeek-Math and Herald, and evaluate them on the minif2f, Putnam, and ProofNet benchmarks. TopoAlign provides substantial gains for DeepSeek-Math, improving performance by 17.77% on BEq@10 and 68.82% on typecheck@10. Despite introducing no new mathematical knowledge, our framework achieves gains of 0.12% and 1.09% for Herald on BEq@10 and typecheck@10, respectively, demonstrating that training on aligned code data is beneficial even for specialized models. 

**Abstract (ZH)**: TopoAlign：利用结构对齐代码数据训练数学大型语言模型 

---
# Discrepancy Detection at the Data Level: Toward Consistent Multilingual Question Answering 

**Title (ZH)**: 数据层面的一致性多语言问答中的 discrepancy 检测 

**Authors**: Lorena Calvo-Bartolomé, Valérie Aldana, Karla Cantarero, Alonso Madroñal de Mesa, Jerónimo Arenas-García, Jordan Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2510.11928)  

**Abstract**: Multilingual question answering (QA) systems must ensure factual consistency across languages, especially for objective queries such as What is jaundice?, while also accounting for cultural variation in subjective responses. We propose MIND, a user-in-the-loop fact-checking pipeline to detect factual and cultural discrepancies in multilingual QA knowledge bases. MIND highlights divergent answers to culturally sensitive questions (e.g., Who assists in childbirth?) that vary by region and context. We evaluate MIND on a bilingual QA system in the maternal and infant health domain and release a dataset of bilingual questions annotated for factual and cultural inconsistencies. We further test MIND on datasets from other domains to assess generalization. In all cases, MIND reliably identifies inconsistencies, supporting the development of more culturally aware and factually consistent QA systems. 

**Abstract (ZH)**: 多语言问答系统必须确保不同语言中的事实一致性，特别是在客观查询（如“什么是黄疸？”）方面，同时还要考虑到主观响应中的文化差异。我们提出MIND，一种包含用户反馈的事实核查流水线，用于检测多语言问答知识库中的事实和文化分歧。MIND突出了文化敏感问题（如“谁协助分娩？”）在不同地区和背景下答案的差异。我们评估MIND在母婴健康领域的双语问答系统上，并发布了一个双语问题数据集，其中标注了事实和文化一致性问题。进一步在其他领域的数据集上测试MIND，以评估其泛化能力。在所有情况下，MIND都能可靠地识别出一致性问题，支持开发更具文化意识和事实一致性的问答系统。 

---
# Indoor Localization using Compact, Telemetry-Agnostic, Transfer-Learning Enabled Decoder-Only Transformer 

**Title (ZH)**: 室内定位Using紧凑、传输协议无关、启用转移学习的仅解码器变压器 

**Authors**: Nayan Sanjay Bhatia, Pranay Kocheta, Russell Elliott, Harikrishna S. Kuttivelil, Katia Obraczka  

**Link**: [PDF](https://arxiv.org/pdf/2510.11926)  

**Abstract**: Indoor Wi-Fi positioning remains a challenging problem due to the high sensitivity of radio signals to environmental dynamics, channel propagation characteristics, and hardware heterogeneity. Conventional fingerprinting and model-based approaches typically require labor-intensive calibration and suffer rapid performance degradation when devices, channel or deployment conditions change. In this paper, we introduce Locaris, a decoder-only large language model (LLM) for indoor localization. Locaris treats each access point (AP) measurement as a token, enabling the ingestion of raw Wi-Fi telemetry without pre-processing. By fine-tuning its LLM on different Wi-Fi datasets, Locaris learns a lightweight and generalizable mapping from raw signals directly to device location. Our experimental study comparing Locaris with state-of-the-art methods consistently shows that Locaris matches or surpasses existing techniques for various types of telemetry. Our results demonstrate that compact LLMs can serve as calibration-free regression models for indoor localization, offering scalable and robust cross-environment performance in heterogeneous Wi-Fi deployments. Few-shot adaptation experiments, using only a handful of calibration points per device, further show that Locaris maintains high accuracy when applied to previously unseen devices and deployment scenarios. This yields sub-meter accuracy with just a few hundred samples, robust performance under missing APs and supports any and all available telemetry. Our findings highlight the practical viability of Locaris for indoor positioning in the real-world scenarios, particularly in large-scale deployments where extensive calibration is infeasible. 

**Abstract (ZH)**: 室内Wi-Fi定位依然是一项具有挑战性的问题，因为无线信号对环境动态、信道传播特性和硬件异构性高度敏感。传统的指纹识别和基于模型的方法通常需要耗费大量的人工校准工作，并且在设备、信道或部署条件改变时会迅速 performance降级。本文介绍了Locaris，一种仅解码的大语言模型（LLM）用于室内定位。Locaris将每个接入点（AP）的测量值视为一个token，允许直接摄入原始Wi-Fi遥测数据而无需预处理。通过在其LLM上对不同的Wi-Fi数据集进行微调，Locaris学会了将原始信号直接映射到设备位置的轻量级和可泛化的映射关系。我们的实验研究将Locaris与现有最先进的方法进行比较，结果显示Locaris在各种类型的遥测数据中都表现出了与现有技术相当或更优的性能。我们的结果表明，紧凑型LLM可以作为无需校准的回归模型用于室内定位，提供在异构Wi-Fi部署中的可扩展和稳健的跨环境性能。少量示例点（每设备几个点）的适应性实验进一步证明，在面对以前未见过的设备和部署场景时，Locaris仍能保持高精度。这使得在仅几百样本的情况下实现亚米级精度，对AP缺失具有鲁棒性能，并支持所有可用的遥测数据。我们的发现突显了Locaris在真实场景中进行室内定位的实际可行性，尤其是在大规模部署中进行全面校准不可行的情况下。 

---
# Integrating Sequential and Relational Modeling for User Events: Datasets and Prediction Tasks 

**Title (ZH)**: 集成序列建模与关系建模的用户事件：数据集与预测任务 

**Authors**: Rizal Fathony, Igor Melnyk, Owen Reinert, Nam H. Nguyen, Daniele Rosa, C. Bayan Bruss  

**Link**: [PDF](https://arxiv.org/pdf/2510.11903)  

**Abstract**: User event modeling plays a central role in many machine learning applications, with use cases spanning e-commerce, social media, finance, cybersecurity, and other domains. User events can be broadly categorized into personal events, which involve individual actions, and relational events, which involve interactions between two users. These two types of events are typically modeled separately, using sequence-based methods for personal events and graph-based methods for relational events. Despite the need to capture both event types in real-world systems, prior work has rarely considered them together. This is often due to the convenient simplification that user behavior can be adequately represented by a single formalization, either as a sequence or a graph. To address this gap, there is a need for public datasets and prediction tasks that explicitly incorporate both personal and relational events. In this work, we introduce a collection of such datasets, propose a unified formalization, and empirically show that models benefit from incorporating both event types. Our results also indicate that current methods leave a notable room for improvements. We release these resources to support further research in unified user event modeling and encourage progress in this direction. 

**Abstract (ZH)**: 用户事件建模在许多机器学习应用中扮演着核心角色，应用场景涵盖电子商务、社交媒体、金融、网络安全及其他领域。用户事件可以大致分为个人事件和关系事件两大类。个人事件涉及个体行为，关系事件涉及两个用户之间的互动。尽管这两种类型的事件通常会分别使用序列模型和图模型进行建模，但在现实系统中捕捉这两种事件类型的需求并未得到充分考虑。这通常是由于一种方便的简化，即用户行为可以用单一的形式化表示，要么是序列，要么是图。为了弥补这一差距，需要包含个人和关系事件的公开数据集和预测任务。在本文中，我们介绍了这样一些数据集、提出了一种统一的形式化方法，并实证表明结合两种事件类型有助于模型的性能提升。我们的研究结果还表明，现有方法仍有改进的空间。我们将这些资源发布出来，以支持统一用户事件建模的进一步研究，并鼓励在这方面的进展。 

---
# MammoDINO: Anatomically Aware Self-Supervision for Mammographic Images 

**Title (ZH)**: MammoDINO：解剖结构意识的自监督学习方法在 mammographic 图像中的应用 

**Authors**: Sicheng Zhou, Lei Wu, Cao Xiao, Parminder Bhatia, Taha Kass-Hout  

**Link**: [PDF](https://arxiv.org/pdf/2510.11883)  

**Abstract**: Self-supervised learning (SSL) has transformed vision encoder training in general domains but remains underutilized in medical imaging due to limited data and domain specific biases. We present MammoDINO, a novel SSL framework for mammography, pretrained on 1.4 million mammographic images. To capture clinically meaningful features, we introduce a breast tissue aware data augmentation sampler for both image-level and patch-level supervision and a cross-slice contrastive learning objective that leverages 3D digital breast tomosynthesis (DBT) structure into 2D pretraining. MammoDINO achieves state-of-the-art performance on multiple breast cancer screening tasks and generalizes well across five benchmark datasets. It offers a scalable, annotation-free foundation for multipurpose computer-aided diagnosis (CAD) tools for mammogram, helping reduce radiologists' workload and improve diagnostic efficiency in breast cancer screening. 

**Abstract (ZH)**: 自助监督学习（SSL）已在一般领域中transformed视觉编码器训练，但在因数据有限和领域特定偏见而未能充分利用的医疗成像领域中，仍有很大的应用潜力。我们提出了MammoDINO，一种新型的自助监督学习框架，基于140万张乳腺X线图像进行预训练。为了捕捉临床有意义的特征，我们引入了一种乳腺组织感知的数据增强采样器，适用于图像级和patches级监督，并提出了一种跨切片对比学习目标，利用3D数字乳腺断层合成（DBT）结构进行二维预训练。MammoDINO在多个乳腺癌筛查任务中达到了最先进的性能，并在五个基准数据集中表现出良好的泛化能力。它为乳腺X线图像的多功能计算机辅助诊断（CAD）工具提供了一个可扩展且无需标注的基础，有助于减轻放射科医生的工作负担并提高乳腺癌筛查的诊断效率。 

---
# Countermind: A Multi-Layered Security Architecture for Large Language Models 

**Title (ZH)**: 反制思维：面向大型语言模型的多层安全架构 

**Authors**: Dominik Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2510.11837)  

**Abstract**: The security of Large Language Model (LLM) applications is fundamentally challenged by "form-first" attacks like prompt injection and jailbreaking, where malicious instructions are embedded within user inputs. Conventional defenses, which rely on post hoc output filtering, are often brittle and fail to address the root cause: the model's inability to distinguish trusted instructions from untrusted data. This paper proposes Countermind, a multi-layered security architecture intended to shift defenses from a reactive, post hoc posture to a proactive, pre-inference, and intra-inference enforcement model. The architecture proposes a fortified perimeter designed to structurally validate and transform all inputs, and an internal governance mechanism intended to constrain the model's semantic processing pathways before an output is generated. The primary contributions of this work are conceptual designs for: (1) A Semantic Boundary Logic (SBL) with a mandatory, time-coupled Text Crypter intended to reduce the plaintext prompt injection attack surface, provided all ingestion paths are enforced. (2) A Parameter-Space Restriction (PSR) mechanism, leveraging principles from representation engineering, to dynamically control the LLM's access to internal semantic clusters, with the goal of mitigating semantic drift and dangerous emergent behaviors. (3) A Secure, Self-Regulating Core that uses an OODA loop and a learning security module to adapt its defenses based on an immutable audit log. (4) A Multimodal Input Sandbox and Context-Defense mechanisms to address threats from non-textual data and long-term semantic poisoning. This paper outlines an evaluation plan designed to quantify the proposed architecture's effectiveness in reducing the Attack Success Rate (ASR) for form-first attacks and to measure its potential latency overhead. 

**Abstract (ZH)**: 大型语言模型应用的安全性从根本上受到了如提示注入和 jailbreaking 等“形式优先”攻击的挑战，这些攻击将恶意指令嵌入用户输入中。传统的防御措施依赖于事后输出过滤，往往脆弱且未能解决根本原因：模型无法区分可信指令与不受信数据。本文提出了 Countermind，一个多层安全架构，旨在将防御从被动的事后防御姿态转变为事先和推理中的主动强制执行模型。该架构提出了一个加固的外围结构，旨在结构性地验证和转换所有输入，并提出了一种内部治理机制，以在生成输出之前约束模型的语义处理路径。本文的主要贡献是概念设计：(1) 一种语义边界逻辑 (SBL)，包括一个强制的时间耦合文本加密器，旨在在所有摄入路径得到执行的情况下减少明文提示注入攻击的攻击面。(2) 一种参数空间限制 (PSR) 机制，利用表示工程原则，动态控制 LLM 对内部语义簇的访问，以减轻语义漂移和危险的新兴行为。(3) 一种安全且自我调节的核心，使用OODA循环和学习安全模块，根据不可变的审计日志调整其防御。(4) 多模态输入沙箱和上下文防御机制，以应对非文本数据和长期语义毒化的威胁。本文概述了评估计划，旨在量化所提出架构在降低形式优先攻击的成功率 (ASR) 方面的有效性，并测量其潜在的延迟开销。 

---
# Data or Language Supervision: What Makes CLIP Better than DINO? 

**Title (ZH)**: 数据监督或语言监督：是什么让CLIP比DINO更优秀？ 

**Authors**: Yiming Liu, Yuhui Zhang, Dhruba Ghosh, Ludwig Schmidt, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2510.11835)  

**Abstract**: CLIP outperforms self-supervised models like DINO as vision encoders for vision-language models (VLMs), but it remains unclear whether this advantage stems from CLIP's language supervision or its much larger training data. To disentangle these factors, we pre-train CLIP and DINO under controlled settings -- using the same architecture, dataset, and training configuration -- achieving similar ImageNet accuracy. Embedding analysis shows that CLIP captures high-level semantics (e.g., object categories, text), while DINO is more responsive to low-level features like colors and styles. When integrated into VLMs and evaluated on 20 VQA benchmarks, CLIP excels at text-intensive tasks, while DINO slightly outperforms on vision-centric ones. Variants of language supervision (e.g., sigmoid loss, pre-trained language encoders) yield limited gains. Our findings provide scientific insights into vision encoder design and its impact on VLM performance. 

**Abstract (ZH)**: CLIP在视觉语言模型的视觉编码器中优于自监督模型如DINO，但其优势是否来自于语言监督或更大的训练数据仍不明确。为了分离这些因素，我们在一个可控的环境中对CLIP和DINO进行预训练——使用相同的架构、数据集和训练配置，获得类似的ImageNet准确性。嵌入分析显示，CLIP捕捉到高层次语义（如物体类别、文本），而DINO对低级特征（如颜色和风格）更敏感。当将其集成到视觉语言模型中并在20个VQA基准上进行评估时，CLIP在文本密集型任务中表现更优，而DINO在视觉中心型任务中略胜一筹。不同形式的语言监督（如Sigmoid损失、预训练的语言编码器）带来的增益有限。我们的发现为视觉编码器设计及其对视觉语言模型性能的影响提供了科学见解。 

---
# Combining Euclidean and Hyperbolic Representations for Node-level Anomaly Detection 

**Title (ZH)**: 结合欧几里得和双曲表示进行节点级异常检测 

**Authors**: Simone Mungari, Ettore Ritacco, Pietro Sabatino  

**Link**: [PDF](https://arxiv.org/pdf/2510.11827)  

**Abstract**: Node-level anomaly detection (NAD) is challenging due to diverse structural patterns and feature distributions. As such, NAD is a critical task with several applications which range from fraud detection, cybersecurity, to recommendation systems. We introduce Janus, a framework that jointly leverages Euclidean and Hyperbolic Graph Neural Networks to capture complementary aspects of node representations. Each node is described by two views, composed by the original features and structural features derived from random walks and degrees, then embedded into Euclidean and Hyperbolic spaces. A multi Graph-Autoencoder framework, equipped with a contrastive learning objective as regularization term, aligns the embeddings across the Euclidean and Hyperbolic spaces, highlighting nodes whose views are difficult to reconcile and are thus likely anomalous. Experiments on four real-world datasets show that Janus consistently outperforms shallow and deep baselines, empirically demonstrating that combining multiple geometric representations provides a robust and effective approach for identifying subtle and complex anomalies in graphs. 

**Abstract (ZH)**: 节点级异常检测（NAD）由于存在多样化的结构模式和特征分布而具有挑战性。因此，NAD 是一个关键任务，具有从欺诈检测、网络安全到推荐系统等多种应用。我们介绍了 Janus，一个结合使用欧几里得和双曲图神经网络的框架，以捕获节点表示的互补方面。每个节点由两部分视图组成，包括原始特征和从随机游走和度派生的结构特征，然后嵌入到欧几里得和双曲空间中。该框架采用多图自编码器架构，并配以对比学习目标作为正则化项，将欧几里得和双曲空间中的嵌入进行对齐，突出那些难以调和的视图节点，从而可能具有异常性。在四个真实世界的数据集上的实验表明，Janus 一致地优于浅层和深层基线，实证证明结合多种几何表示是识别图中细微而复杂的异常的一种稳健而有效的方法。 

---
# Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning 

**Title (ZH)**: 合作多代理强化学习中稳健性和韧性的实证研究 

**Authors**: Simin Li, Zihao Mao, Hanxiao Li, Zonglei Jing, Zhuohang bian, Jun Guo, Li Wang, Zhuoran Han, Ruixiao Xu, Xin Yu, Chengdong Ma, Yuqing Ma, Bo An, Yaodong Yang, Weifeng Lv, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11824)  

**Abstract**: In cooperative Multi-Agent Reinforcement Learning (MARL), it is a common practice to tune hyperparameters in ideal simulated environments to maximize cooperative performance. However, policies tuned for cooperation often fail to maintain robustness and resilience under real-world uncertainties. Building trustworthy MARL systems requires a deep understanding of robustness, which ensures stability under uncertainties, and resilience, the ability to recover from disruptions--a concept extensively studied in control systems but largely overlooked in MARL. In this paper, we present a large-scale empirical study comprising over 82,620 experiments to evaluate cooperation, robustness, and resilience in MARL across 4 real-world environments, 13 uncertainty types, and 15 hyperparameters. Our key findings are: (1) Under mild uncertainty, optimizing cooperation improves robustness and resilience, but this link weakens as perturbations intensify. Robustness and resilience also varies by algorithm and uncertainty type. (2) Robustness and resilience do not generalize across uncertainty modalities or agent scopes: policies robust to action noise for all agents may fail under observation noise on a single agent. (3) Hyperparameter tuning is critical for trustworthy MARL: surprisingly, standard practices like parameter sharing, GAE, and PopArt can hurt robustness, while early stopping, high critic learning rates, and Leaky ReLU consistently help. By optimizing hyperparameters only, we observe substantial improvement in cooperation, robustness and resilience across all MARL backbones, with the phenomenon also generalizing to robust MARL methods across these backbones. Code and results available at this https URL . 

**Abstract (ZH)**: 在合作多智能体强化学习（MARL）中，通常在理想的模拟环境中调整超参数以最大化合作性能。然而，针对合作优化的策略往往在现实世界的不确定性下失去鲁棒性和恢复力。构建可信赖的MARL系统需要深入理解鲁棒性，确保在不确定性下的稳定性，和恢复力——这一概念在控制系统中已有广泛研究，但在MARL领域却鲜有关注。本文通过包含超过82,620个实验的大型实证研究，在4个真实环境、13种不确定性类型和15个超参数下评估MARL中的合作、鲁棒性和恢复力。主要发现包括：(1) 在轻微不确定性下，优化合作能提升鲁棒性和恢复力，但随着扰动加剧，这种关联减弱。鲁棒性和恢复力也因算法和不确定性类型而异。(2) 鲁棒性和恢复力在不同类型的不确定性或智能体范围内不具有普适性：能抵抗所有智能体动作噪声的策略可能在单一智能体的观察噪声下失效。(3) 超参数调整对于可信赖的MARL至关重要：意想不到的是，标准做法如参数共享、GAE和PopArt可能导致鲁棒性下降，而早停策略、高评论家学习率和Leaky ReLU则始终有所助益。仅通过优化超参数，在所有MARL框架中观察到显著的改进，这一现象也推广到了这些框架下的鲁棒MARL方法。代码和结果可在以下链接获取。 

---
# BlackIce: A Containerized Red Teaming Toolkit for AI Security Testing 

**Title (ZH)**: BlackIce：一个容器化的红色团队AI安全测试工具包 

**Authors**: Caelin Kaplan, Alexander Warnecke, Neil Archibald  

**Link**: [PDF](https://arxiv.org/pdf/2510.11823)  

**Abstract**: AI models are being increasingly integrated into real-world systems, raising significant concerns about their safety and security. Consequently, AI red teaming has become essential for organizations to proactively identify and address vulnerabilities before they can be exploited by adversaries. While numerous AI red teaming tools currently exist, practitioners face challenges in selecting the most appropriate tools from a rapidly expanding landscape, as well as managing complex and frequently conflicting software dependencies across isolated projects. Given these challenges and the relatively small number of organizations with dedicated AI red teams, there is a strong need to lower barriers to entry and establish a standardized environment that simplifies the setup and execution of comprehensive AI model assessments.
Inspired by Kali Linux's role in traditional penetration testing, we introduce BlackIce, an open-source containerized toolkit designed for red teaming Large Language Models (LLMs) and classical machine learning (ML) models. BlackIce provides a reproducible, version-pinned Docker image that bundles 14 carefully selected open-source tools for Responsible AI and Security testing, all accessible via a unified command-line interface. With this setup, initiating red team assessments is as straightforward as launching a container, either locally or using a cloud platform. Additionally, the image's modular architecture facilitates community-driven extensions, allowing users to easily adapt or expand the toolkit as new threats emerge. In this paper, we describe the architecture of the container image, the process used for selecting tools, and the types of evaluations they support. 

**Abstract (ZH)**: AI模型正越来越多地集成到实际系统中，引发了对其安全性和安全性的重要关注。因此，AI红队行动对于组织来说变得至关重要，可以提前识别和解决潜在被对手利用的漏洞。尽管目前存在众多AI红队工具，但实践者在从快速增长的工具库中选择最合适的工具时仍面临挑战，并且在孤立项目中管理复杂且频繁冲突的软件依赖也存在困难。鉴于这些挑战和拥有专门AI红队的小组织数量有限，降低进入壁垒并建立标准化环境以简化全面AI模型评估的设置和执行变得非常必要。

受Kali Linux在传统渗透测试中作用的启发，我们引入了BlackIce，这是一个用于红队评估大型语言模型（LLMs）和经典机器学习（ML）模型的开源容器化工具包。BlackIce提供了一个可重复的、版本锁定的Docker镜像，该镜像捆绑了14个精心选择的开源工具，用于负责任的AI和安全测试，所有工具均可通过统一的命令行接口访问。通过此设置，启动红队评估只需启动一个容器，无论是本地还是使用云平台。此外，镜像的模块化架构促进了社区驱动的扩展，使用户能够轻松适应或扩展工具以应对新出现的威胁。在本文中，我们描述了容器镜像的架构、工具选择过程以及它们支持的评估类型。 

---
# PHANTOM RECALL: When Familiar Puzzles Fool Smart Models 

**Title (ZH)**: 幻影回忆：熟悉的谜题欺骗智能模型 

**Authors**: Souradeep Mukhopadhyay, Rishabh Baral, Nimeesh Mahajan, Samhitha Harish, Aswin RRV, Mihir Parmar, Mutsumi Nakamura, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2510.11812)  

**Abstract**: Large language models (LLMs) such as GPT, Gemini, and Claude often appear adept at solving classic logic puzzles--but how much genuine reasoning underlies their answers? Recent evidence suggests that these models frequently rely on memorized templates rather than reasoning from first principles. When puzzles are slightly modified, their performance collapses, revealing a striking fragility. In particular, we asked: Have LLMs addressed these issues? To what extent? How about perturbations to other puzzles? Is there a general way of reformulating the prompt so that the models do better? To examine these things systematically, we introduce PHANTOM RECALL, a benchmark comprising 25 well-known logic puzzles and 149 carefully designed perturbations that preserve reasoning structure but alter superficial details and solutions. We evaluate eleven leading LLMs and identify a recurring failure mode--phantom recall--where models confidently reproduce memorized solutions or spurious rationales that no longer fit the altered scenario. To probe and mitigate this issue, we contribute three tools: (i) an automated logical-equivalence judge to detect reasoning mismatches, (ii) a taxonomy of fine-grained reasoning error categories, and (iii) a prompting-based mitigation framework guided by these categories. Despite near-perfect accuracy on unmodified puzzles, models significantly underperform humans on perturbed ones, exhibiting both phantom recall and over-elaboration. Our findings reveal a crucial limitation: LLMs often fail to re-reason when contextual cues shift--highlighting the gap between linguistic fluency and logical understanding. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT、Gemini和Claude通常擅长解决经典的逻辑谜题——但它们的答案背后的真正推理有多少是真实的？最近的证据表明，这些模型经常依赖于记忆中的模板，而不是从基本原理出发进行推理。当谜题稍作修改时，它们的表现会崩溃，显示出一种惊人的脆弱性。特别是我们询问：LLMs是否解决了这些问题？解决了多大程度？对其他谜题的扰动又如何？是否有一种普遍的方法重新表述提示，从而使模型表现更好？为了系统地研究这些问题，我们介绍了PHANTOM RECALL基准，包含25个经典的逻辑谜题和149个精心设计的扰动，这些扰动保留了推理结构但改变了表层细节和解决方案。我们评估了十一个领先的LLMs，并发现一个重复出现的失败模式——幻影回忆，模型自信地重现记忆中的解决方案或不再适用于修改后场景的虚假理由。为了探究并缓解这一问题，我们贡献了三个工具：（i）自动化逻辑等价判断器以检测推理不匹配；（ii）精细推理错误类别分类法；以及（iii）基于这些类别的提示引导缓解框架。尽管在未修改的谜题上表现出几乎完美的准确性，但在扰动谜题上，模型的人类表现显著下降，表现为幻影回忆和过度扩展。我们的发现揭示了一个关键限制：LLMs往往在上下文线索改变时未能重新推理——突显了语言流畅性和逻辑理解之间的差距。 

---
# GAR: Generative Adversarial Reinforcement Learning for Formal Theorem Proving 

**Title (ZH)**: GAR：生成对抗强化学习在形式定理证明中的应用 

**Authors**: Ruida Wang, Jiarui Yao, Rui Pan, Shizhe Diao, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11769)  

**Abstract**: Solving math problems through verifiable languages such as Lean has significantly impacted both the mathematics and computer science communities. Current state-of-the-art models are often trained with expensive online Reinforcement Learning (RL) or expert iteration. However, these approaches rely on fixed problem sets, which causes inefficient training and limits the model to tackle complex problems. To overcome these limitations, we propose GAR: Generative Adversarial Reinforcement learning, a comprehensive RL training framework that jointly trains the problem composer and solver in an adversarial loop. GAR introduces an implicit curriculum learning mechanism, which aligns task difficulty with the prover's evolving capability. It thereby improves the training efficiency and enables stronger performance of proving advanced theorems. Experiments show that with GAR training, Goedel-Prover-V2-8B and DeepSeek-Prover-V2-7B achieve an average relative improvement in pass@32 of 4.20% on MiniF2F-Test benchmark, while DeepSeek-Prover-V2's pass@32 on ProofNet-Test increases from 22.58% to 25.81%. Beyond formal proving, GAR establishes a general RL paradigm for co-evolution of problem generation and solving under verifiable environments. 

**Abstract (ZH)**: 通过可验证语言如Lean解决数学问题，显著影响了数学和计算机科学社区。生成对抗强化学习GAR：一种综合的RL训练框架，在对抗循环中共同训练问题生成者和解决者，以克服现有局限性。GAR引入了隐式课程学习机制，使任务难度与证明者的能力进化保持一致，从而提高训练效率并增强证明高级定理的能力。实验表明，使用GAR训练后，Goedel-Prover-V2-8B和DeepSeek-Prover-V2-7B在MiniF2F-Test基准上的pass@32平均相对改进为4.20%，而DeepSeek-Prover-V2在ProofNet-Test上的pass@32从22.58%提高到25.81%。GAR还建立了在可验证环境中问题生成与解决协同进化的通用RL范式。 

---
# Audio-Guided Visual Perception for Audio-Visual Navigation 

**Title (ZH)**: 基于音频的视听感知方法在视听导航中的应用 

**Authors**: Yi Wang, Yinfeng Yu, Fuchun Sun, Liejun Wang, Wendong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.11760)  

**Abstract**: Audio-Visual Embodied Navigation aims to enable agents to autonomously navigate to sound sources in unknown 3D environments using auditory cues. While current AVN methods excel on in-distribution sound sources, they exhibit poor cross-source generalization: navigation success rates plummet and search paths become excessively long when agents encounter unheard sounds or unseen environments. This limitation stems from the lack of explicit alignment mechanisms between auditory signals and corresponding visual regions. Policies tend to memorize spurious \enquote{acoustic fingerprint-scenario} correlations during training, leading to blind exploration when exposed to novel sound sources. To address this, we propose the AGVP framework, which transforms sound from policy-memorable acoustic fingerprint cues into spatial guidance. The framework first extracts global auditory context via audio self-attention, then uses this context as queries to guide visual feature attention, highlighting sound-source-related regions at the feature level. Subsequent temporal modeling and policy optimization are then performed. This design, centered on interpretable cross-modal alignment and region reweighting, reduces dependency on specific acoustic fingerprints. Experimental results demonstrate that AGVP improves both navigation efficiency and robustness while achieving superior cross-scenario generalization on previously unheard sounds. 

**Abstract (ZH)**: 基于视听融合的小型化自主声源定位导航框架：AGVP 

---
# AwareCompiler: Agentic Context-Aware Compiler Optimization via a Synergistic Knowledge-Data Driven Framework 

**Title (ZH)**: AwareCompiler: 具有代理上下文意识的协同知识-数据驱动编译器优化框架 

**Authors**: Hongyu Lin, Haolin Pan, Haoran Luo, Yuchen Li, Kaichun Yao, Libo Zhang, Mingjie Xing, Yanjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11759)  

**Abstract**: Compiler optimization is crucial for enhancing program performance by transforming the sequence of optimization passes while maintaining correctness. Despite the promising potential of large language models (LLMs)-based agent for software optimization, automating compiler optimization remains challenging due to: (1) semantic misalignment between abstract program representations and concrete optimization passes, (2) inefficient interaction mechanisms between agents and compiler environments, and (3) reward sparsity from the extensive decision-making process within large optimization spaces. This paper introduces \textbf{AwareCompiler}, an agentic framework for compiler optimization that addresses these challenges through three key innovations: structured knowledge integration and dataset construction, knowledge-driven adaptive pass generation, and data-driven hybrid training pipeline. Experimental results on standard benchmarks demonstrate that AwareCompiler significantly outperforms existing baselines in both performance and efficiency, highlighting the effectiveness of our synergistic knowledge-data-driven approach. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 编译器优化对于通过变换优化传递序列来提高程序性能至关重要。尽管基于大规模语言模型（LLM）的代理在软件优化方面前景诱人，但由于以下原因，自动化编译器优化仍然具有挑战性：（1）抽象程序表示与具体优化传递之间的语义对齐不一致，（2）代理与编译器环境之间的低效交互机制，以及（3）在广泛的优化空间内进行大量决策带来的稀疏奖励。本文介绍了AwareCompiler——一种通过三种关键创新解决这些挑战的编译器优化代理框架：结构化知识集成与数据集构建、知识驱动的自适应传递生成以及数据驱动的混合训练流水线。在标准基准上的实验结果表明，AwareCompiler在性能和效率方面显著优于现有基线，突显了我们的协同知识-数据驱动方法的有效性。我们的代码可在以下网址公开获取：this https URL。 

---
# The Adoption Paradox: A Comparative Analysis of Veterinary AI Adoption in China and the North America 

**Title (ZH)**: 兽医AI Adoption Paradox: A Comparative Analysis of Adoption in China and North America 

**Authors**: Shumin Li, Xiaoyun Lai  

**Link**: [PDF](https://arxiv.org/pdf/2510.11758)  

**Abstract**: This study compares the perception, adoption, and application of artificial intelligence (AI) among veterinary professionals in China and North America (NA), testing the hypothesis that adoption patterns are shaped by regional market and demographic factors. A descriptive, cross-sectional survey was conducted with 455 veterinary professionals in China between May and July 2025. The results were compared with published data from a 2024 survey of 3,968 veterinary professionals in the United States and Canada. The Chinese cohort, primarily composed of clinicians (81.5%), showed a high AI adoption rate (71.0%) despite low familiarity (55.4%). Their AI use was focused on clinical tasks, such as disease diagnosis (50.1%) and prescription calculation (44.8%). In contrast, the NA cohort reported high familiarity (83.8%) but a lower adoption rate (39.2%). Their priorities were administrative, including imaging analysis (39.0%) and record-keeping (39.0%). Concerns about AI reliability and accuracy were the top barrier in both groups. Our findings reveal an "adoption paradox" where the Chinese market demonstrates a practitioner-driven, bottom-up adoption model focused on augmenting clinical efficacy, while the NA market shows a more cautious, structured, top-down integration aimed at improving administrative efficiency. This suggests that a one-size-fits-all approach to AI development and integration is insufficient, and tailored, region-specific strategies are necessary to responsibly incorporate AI into global veterinary practice. 

**Abstract (ZH)**: 中国与北美的兽医专业人员对人工智能的感知、采用与应用比较：基于区域市场和人口因素的假设测试 

---
# Artificial Intelligence for Optimal Learning: A Comparative Approach towards AI-Enhanced Learning Environments 

**Title (ZH)**: 人工智能优化学习：面向AI增强学习环境的比较研究 

**Authors**: Ananth Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2510.11755)  

**Abstract**: In the rapidly evolving educational landscape, the integration of technology has shifted from an enhancement to a cornerstone of educational strategy worldwide. This transition is propelled by advancements in digital technology, especially the emergence of artificial intelligence as a crucial tool in learning environments. This research project critically evaluates the impact of three distinct educational settings: traditional educational methods without technological integration, those enhanced by non-AI technology, and those utilising AI-driven technologies. This comparison aims to assess how each environment influences educational outcomes, engagement, pedagogical methods, and equity in access to learning resources, and how each contributes uniquely to the learning experience. The ultimate goal of this research is to synthesise the strengths of each model to create a more holistic educational approach. By integrating the personal interaction and tested pedagogical techniques of traditional classrooms, the enhanced accessibility and collaborative tools offered by non-AI technology, and the personalised, adaptive learning strategies enabled by AI-driven technologies, education systems can develop richer, more effective learning environments. This hybrid approach aims to leverage the best elements of each setting, thereby enhancing educational outcomes, engagement, and inclusiveness, while also addressing the distinct challenges and limitations inherent in each model. The intention is to create an educational framework deeply attentive to the diverse needs of students, ensuring equitable access to high-quality education for all. 

**Abstract (ZH)**: 在快速演进的教育格局中，技术的整合已从一种增强手段转变为全球教育策略的基石。这一转变由数字技术的进步推动，尤其是人工智能作为学习环境中关键工具的出现。本研究项目批判性地评估了三种不同的教育环境：不包含技术整合的传统教育方法、通过非人工智能技术增强的方法以及利用人工智能驱动技术的方法的影响。这一比较旨在评估每种环境如何影响教育成果、参与度、教学方法以及学习资源的公平获取，并且评估每种环境如何独特地为学习体验做出贡献。本研究的最终目标是综合每种模式的优势，以形成更为全面的教育方法。通过结合传统课堂中的人际互动和个人已证实的教学技巧、非人工智能技术提供的增强可访问性和协作工具，以及人工智能驱动技术所实现的个性化和自适应学习策略，教育体系可以发展出更为丰富、更有效的学习环境。这种混合方法旨在利用每种设置的最佳要素，从而增强教育成果、参与度和包容性，同时解决每种模式固有的独特挑战和限制。其目的是建立一个深度关注学生多样需求的教育框架，确保所有学生都能获得高质量的教育机会。 

---
# Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning 

**Title (ZH)**: 零-shot 大型语言模型代理实现全自动放射治疗计划规划 

**Authors**: Dongrong Yang, Xin Wu, Yibo Xie, Xinyi Li, Qiuwen Wu, Jackie Wu, Yang Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.11754)  

**Abstract**: Radiation therapy treatment planning is an iterative, expertise-dependent process, and the growing burden of cancer cases has made reliance on manual planning increasingly unsustainable, underscoring the need for automation. In this study, we propose a workflow that leverages a large language model (LLM)-based agent to navigate inverse treatment planning for intensity-modulated radiation therapy (IMRT). The LLM agent was implemented to directly interact with a clinical treatment planning system (TPS) to iteratively extract intermediate plan states and propose new constraint values to guide inverse optimization. The agent's decision-making process is informed by current observations and previous optimization attempts and evaluations, allowing for dynamic strategy refinement. The planning process was performed in a zero-shot inference setting, where the LLM operated without prior exposure to manually generated treatment plans and was utilized without any fine-tuning or task-specific training. The LLM-generated plans were evaluated on twenty head-and-neck cancer cases against clinical manual plans, with key dosimetric endpoints analyzed and reported. The LLM-generated plans achieved comparable organ-at-risk (OAR) sparing relative to clinical plans while demonstrating improved hot spot control (Dmax: 106.5% vs. 108.8%) and superior conformity (conformity index: 1.18 vs. 1.39 for boost PTV; 1.82 vs. 1.88 for primary PTV). This study demonstrates the feasibility of a zero-shot, LLM-driven workflow for automated IMRT treatment planning in a commercial TPS. The proposed approach provides a generalizable and clinically applicable solution that could reduce planning variability and support broader adoption of AI-based planning strategies. 

**Abstract (ZH)**: 基于大型语言模型的零样本逆治疗规划工作流在商业治疗计划系统中的自动调强放疗治疗计划研究 

---
# Fast and Interpretable Protein Substructure Alignment via Optimal Transport 

**Title (ZH)**: 快速且可解释的蛋白质亚结构对齐方法基于最优传输 

**Authors**: Zhiyu Wang, Bingxin Zhou, Jing Wang, Yang Tan, Weishu Zhao, Pietro Liò, Liang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.11752)  

**Abstract**: Proteins are essential biological macromolecules that execute life functions. Local motifs within protein structures, such as active sites, are the most critical components for linking structure to function and are key to understanding protein evolution and enabling protein engineering. Existing computational methods struggle to identify and compare these local structures, which leaves a significant gap in understanding protein structures and harnessing their functions. This study presents PLASMA, the first deep learning framework for efficient and interpretable residue-level protein substructure alignment. We reformulate the problem as a regularized optimal transport task and leverage differentiable Sinkhorn iterations. For a pair of input protein structures, PLASMA outputs a clear alignment matrix with an interpretable overall similarity score. Through extensive quantitative evaluations and three biological case studies, we demonstrate that PLASMA achieves accurate, lightweight, and interpretable residue-level alignment. Additionally, we introduce PLASMA-PF, a training-free variant that provides a practical alternative when training data are unavailable. Our method addresses a critical gap in protein structure analysis tools and offers new opportunities for functional annotation, evolutionary studies, and structure-based drug design. Reproducibility is ensured via our official implementation at this https URL. 

**Abstract (ZH)**: 蛋白质是执行生命功能的关键生物大分子。蛋白质结构中的局部基序，如活性位点，是将结构与功能连接起来的最关键组成部分，对于理解蛋白质演化和蛋白质工程至关重要。现有计算方法难以识别和比较这些局部结构，这在了解蛋白质结构和充分利用其功能方面留下了一定的差距。本文介绍了PLASMA，这是首个用于高效和可解释的残基级蛋白质亚结构对齐的深度学习框架。我们将问题重新表述为正则化最优传输任务，并利用可微Sinkhorn迭代。对于一对输入蛋白质结构，PLASMA输出一个清晰的对齐矩阵，带有可解释的整体相似性得分。通过广泛的定量评估和三个生物案例研究，我们证明PLASMA能够实现准确、轻量级且可解释的残基级对齐。此外，我们引入了PLASMA-PF，这是一个无需训练的变体，当缺乏训练数据时可以提供实用的选择。我们的方法填补了蛋白质结构分析工具的关键空白，并为功能注释、进化研究和基于结构的药物设计提供了新机会。结果的可再现性通过我们的官方实现得到了保证：[此链接]。 

---
# Celebrity Profiling on Short Urdu Text using Twitter Followers' Feed 

**Title (ZH)**: 基于Twitter粉丝帖子的乌尔都语短文本名人画像构建 

**Authors**: Muhammad Hamza, Rizwan Jafar  

**Link**: [PDF](https://arxiv.org/pdf/2510.11739)  

**Abstract**: Social media has become an essential part of the digital age, serving as a platform for communication, interaction, and information sharing. Celebrities are among the most active users and often reveal aspects of their personal and professional lives through online posts. Platforms such as Twitter provide an opportunity to analyze language and behavior for understanding demographic and social patterns. Since followers frequently share linguistic traits and interests with the celebrities they follow, textual data from followers can be used to predict celebrity demographics. However, most existing research in this field has focused on English and other high-resource languages, leaving Urdu largely unexplored.
This study applies modern machine learning and deep learning techniques to the problem of celebrity profiling in Urdu. A dataset of short Urdu tweets from followers of subcontinent celebrities was collected and preprocessed. Multiple algorithms were trained and compared, including Logistic Regression, Support Vector Machines, Random Forests, Convolutional Neural Networks, and Long Short-Term Memory networks. The models were evaluated using accuracy, precision, recall, F1-score, and cumulative rank (cRank). The best performance was achieved for gender prediction with a cRank of 0.65 and an accuracy of 0.65, followed by moderate results for age, profession, and fame prediction. These results demonstrate that follower-based linguistic features can be effectively leveraged using machine learning and neural approaches for demographic prediction in Urdu, a low-resource language. 

**Abstract (ZH)**: Urdu社交媒体中的名人画像研究：基于追随者文本数据的机器学习与深度学习方法 

---
# SeeingSounds: Learning Audio-to-Visual Alignment via Text 

**Title (ZH)**: SeeingSounds：通过文本学习音频到视觉对齐 

**Authors**: Simone Carnemolla, Matteo Pennisi, Chiara Russo, Simone Palazzo, Daniela Giordano, Concetto Spampinato  

**Link**: [PDF](https://arxiv.org/pdf/2510.11738)  

**Abstract**: We introduce SeeingSounds, a lightweight and modular framework for audio-to-image generation that leverages the interplay between audio, language, and vision-without requiring any paired audio-visual data or training on visual generative models. Rather than treating audio as a substitute for text or relying solely on audio-to-text mappings, our method performs dual alignment: audio is projected into a semantic language space via a frozen language encoder, and, contextually grounded into the visual domain using a vision-language model. This approach, inspired by cognitive neuroscience, reflects the natural cross-modal associations observed in human perception. The model operates on frozen diffusion backbones and trains only lightweight adapters, enabling efficient and scalable learning. Moreover, it supports fine-grained and interpretable control through procedural text prompt generation, where audio transformations (e.g., volume or pitch shifts) translate into descriptive prompts (e.g., "a distant thunder") that guide visual outputs. Extensive experiments across standard benchmarks confirm that SeeingSounds outperforms existing methods in both zero-shot and supervised settings, establishing a new state of the art in controllable audio-to-visual generation. 

**Abstract (ZH)**: SeeingSounds：一种基于音频、语言和视觉交互的轻量级模块化图像生成框架 

---
# Scaling Law in LLM Simulated Personality: More Detailed and Realistic Persona Profile Is All You Need 

**Title (ZH)**: LLM模拟人格中的标度律：只需更详细及逼真的个性档案 

**Authors**: Yuqi Bai, Tianyu Huang, Kun Sun, Yuting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11734)  

**Abstract**: This research focuses on using large language models (LLMs) to simulate social experiments, exploring their ability to emulate human personality in virtual persona role-playing. The research develops an end-to-end evaluation framework, including individual-level analysis of stability and identifiability, as well as population-level analysis called progressive personality curves to examine the veracity and consistency of LLMs in simulating human personality. Methodologically, this research proposes important modifications to traditional psychometric approaches (CFA and construct validity) which are unable to capture improvement trends in LLMs at their current low-level simulation, potentially leading to remature rejection or methodological misalignment. The main contributions of this research are: proposing a systematic framework for LLM virtual personality evaluation; empirically demonstrating the critical role of persona detail in personality simulation quality; and identifying marginal utility effects of persona profiles, especially a Scaling Law in LLM personality simulation, offering operational evaluation metrics and a theoretical foundation for applying large language models in social science experiments. 

**Abstract (ZH)**: 本研究致力于使用大型语言模型（LLMs）模拟社会实验，探索其在虚拟角色扮演中模仿人类个性的能力。研究开发了一个端到端的评估框架，包括个体层面的稳定性和可识别性分析，以及人口层面的渐进个性曲线分析，以检验LLMs在模拟人类个性时的真实性和一致性。在方法论上，本研究提出了对传统心理测量方法（CFA和结构效度）的重要修改，这些方法无法捕捉LLMs在其当前低水平模拟中的改进趋势，可能导致过早拒绝或方法论不匹配。本研究的主要贡献包括：提出了一套系统框架用于评估LLM虚拟个性；实证展示了个性细节在个性模拟质量中的关键作用；并识别了个性档案的边际效益效应，特别是在LLM个性模拟中的标度律，提供了操作性评估指标及应用大型语言模型于社会科学研究实验的理论基础。 

---
# Serial-Parallel Dual-Path Architecture for Speaking Style Recognition 

**Title (ZH)**: 说话风格识别的串并行双路径架构 

**Authors**: Guojian Li, Qijie Shao, Zhixian Zhao, Shuiyuan Wang, Zhonghua Fu, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.11732)  

**Abstract**: Speaking Style Recognition (SSR) identifies a speaker's speaking style characteristics from speech. Existing style recognition approaches primarily rely on linguistic information, with limited integration of acoustic information, which restricts recognition accuracy improvements. The fusion of acoustic and linguistic modalities offers significant potential to enhance recognition performance. In this paper, we propose a novel serial-parallel dual-path architecture for SSR that leverages acoustic-linguistic bimodal information. The serial path follows the ASR+STYLE serial paradigm, reflecting a sequential temporal dependency, while the parallel path integrates our designed Acoustic-Linguistic Similarity Module (ALSM) to facilitate cross-modal interaction with temporal simultaneity. Compared to the existing SSR baseline -- the OSUM model, our approach reduces parameter size by 88.4% and achieves a 30.3% improvement in SSR accuracy for eight styles on the test set. 

**Abstract (ZH)**: 说话风格识别（SSR）从语音中识别说话人的说话风格特征。现有的风格识别方法主要依赖于语言信息，对声学信息的整合有限，限制了识别准确性的提升。将声学和语言模态融合为显著潜在提升识别性能的机会。在本文中，我们提出了一种新颖的串行-并行双路径架构，利用声学-语言双模态信息。串行路径遵循ASR+STYLE串行范式，反映序列时间依赖性，而并行路径结合了我们设计的声学-语言相似性模块（ALSM），以时间的同时性促进跨模态交互。与现有的SSR基线OSUM模型相比，我们的方法参数量减少了88.4%，并在测试集上实现了八种风格下SSR准确性的30.3%提高。 

---
# Modeling Hypergraph Using Large Language Models 

**Title (ZH)**: 使用大型语言模型建模超图 

**Authors**: Bingqiao Gu, Jiale Zeng, Xingqin Qi, Dong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.11728)  

**Abstract**: Due to the advantages of hypergraphs in modeling high-order relationships in complex systems, they have been applied to higher-order clustering, hypergraph neural networks and computer vision. These applications rely heavily on access to high-quality, large-scale real-world hypergraph data. Yet, compared to traditional pairwise graphs, real hypergraph datasets remain scarce in both scale and diversity. This shortage significantly limits the development and evaluation of advanced hypergraph learning algorithms. Therefore, how to quickly generate large-scale hypergraphs that conform to the characteristics of real networks is a crucial task that has not received sufficient attention. Motivated by recent advances in large language models (LLMs), particularly their capabilities in semantic reasoning, structured generation, and simulating human behavior, we investigate whether LLMs can facilitate hypergraph generation from a fundamentally new perspective. We introduce HyperLLM, a novel LLM-driven hypergraph generator that simulates the formation and evolution of hypergraphs through a multi-agent collaboration. The framework integrates prompts and structural feedback mechanisms to ensure that the generated hypergraphs reflect key real-world patterns. Extensive experiments across diverse datasets demonstrate that HyperLLM achieves superior fidelity to structural and temporal hypergraph patterns, while requiring minimal statistical priors. Our findings suggest that LLM-based frameworks offer a promising new direction for hypergraph modeling. 

**Abstract (ZH)**: 基于大型语言模型的高阶网络生成方法 

---
# Dual Perspectives on Non-Contrastive Self-Supervised Learning 

**Title (ZH)**: 非对比自监督学习的双重视角 

**Authors**: Jean Ponce, Basile Terver, Martial Hebert, Michael Arbel  

**Link**: [PDF](https://arxiv.org/pdf/2507.01028)  

**Abstract**: The {\em stop gradient} and {\em exponential moving average} iterative procedures are commonly used in non-contrastive approaches to self-supervised learning to avoid representation collapse, with excellent performance in downstream applications in practice. This presentation investigates these procedures from the dual viewpoints of optimization and dynamical systems. We show that, in general, although they {\em do not} optimize the original objective, or {\em any} other smooth function, they {\em do} avoid collapse Following~\citet{Tian21}, but without any of the extra assumptions used in their proofs, we then show using a dynamical system perspective that, in the linear case, minimizing the original objective function without the use of a stop gradient or exponential moving average {\em always} leads to collapse. Conversely, we characterize explicitly the equilibria of the dynamical systems associated with these two procedures in this linear setting as algebraic varieties in their parameter space, and show that they are, in general, {\em asymptotically stable}. Our theoretical findings are illustrated by empirical experiments with real and synthetic data. 

**Abstract (ZH)**: 《梯度停止和指数移动平均迭代过程在非对比方法的自监督学习中的应用及其从优化和动力系统视角的探讨：在一般情况下，虽然它们不优化原始目标或任何其他光滑函数，但确实避免了表示坍塌。借助动力系统视角，我们证明，在线性情况下，不使用梯度停止或指数移动平均最小化原始目标函数总是会导致坍塌。相反，在线性设置中，我们明确表征了这两种过程相关动力系统的平衡点作为参数空间中的代数簇，并证明它们通常具有渐近稳定性。我们的理论发现通过实际和合成数据的实验证据予以说明。》 

---
# Leveraging LLMs, IDEs, and Semantic Embeddings for Automated Move Method Refactoring 

**Title (ZH)**: 利用大语言模型、集成开发环境和语义嵌入进行自动化移动方法重构 

**Authors**: Fraol Batole, Abhiram Bellur, Malinda Dilhara, Mohammed Raihan Ullah, Yaroslav Zharov, Timofey Bryksin, Kai Ishikawa, Haifeng Chen, Masaharu Morimoto, Shota Motoura, Takeo Hosomi, Tien N. Nguyen, Hridesh Rajan, Nikolaos Tsantalis, Danny Dig  

**Link**: [PDF](https://arxiv.org/pdf/2503.20934)  

**Abstract**: MOVEMETHOD is a hallmark refactoring. Despite a plethora of research tools that recommend which methods to move and where, these recommendations do not align with how expert developers perform MOVEMETHOD. Given the extensive training of Large Language Models and their reliance upon naturalness of code, they should expertly recommend which methods are misplaced in a given class and which classes are better hosts. Our formative study of 2016 LLM recommendations revealed that LLMs give expert suggestions, yet they are unreliable: up to 80% of the suggestions are hallucinations. We introduce the first LLM fully powered assistant for MOVEMETHOD refactoring that automates its whole end-to-end lifecycle, from recommendation to execution. We designed novel solutions that automatically filter LLM hallucinations using static analysis from IDEs and a novel workflow that requires LLMs to be self-consistent, critique, and rank refactoring suggestions. As MOVEMETHOD refactoring requires global, projectlevel reasoning, we solved the limited context size of LLMs by employing refactoring-aware retrieval augment generation (RAG). Our approach, MM-assist, synergistically combines the strengths of the LLM, IDE, static analysis, and semantic relevance. In our thorough, multi-methodology empirical evaluation, we compare MM-assist with the previous state-of-the-art approaches. MM-assist significantly outperforms them: (i) on a benchmark widely used by other researchers, our Recall@1 and Recall@3 show a 1.7x improvement; (ii) on a corpus of 210 recent refactorings from Open-source software, our Recall rates improve by at least 2.4x. Lastly, we conducted a user study with 30 experienced participants who used MM-assist to refactor their own code for one week. They rated 82.8% of MM-assist recommendations positively. This shows that MM-assist is both effective and useful. 

**Abstract (ZH)**: MOVEMETHOD是一种标志性重构。尽管存在许多推荐如何移动方法的研究工具，但这些推荐并不符合专家开发者进行MOVEMETHOD的方式。鉴于大型语言模型的广泛培训及其对代码自然性的依赖，它们应该能够准确推荐哪些方法在给定类中是错位的，以及哪些类是更好的宿主。我们的初步研究发现，2016年大型语言模型的建议虽然提供了专家级别的建议，但可靠性较差：多达80%的建议是妄想。我们引入了第一个全面支持大型语言模型的MOVEMETHOD重构辅助工具，实现了从推荐到执行的整个端到端生命周期自动化。我们设计了新的解决方案自动过滤大型语言模型的妄想，并采用了一种新的工作流，要求大型语言模型自我一致、批判性评估和排名重构建议。由于MOVEMETHOD重构需要全局性的项目级推理，我们通过采用重构感知检索增强生成（RAG）来解决大型语言模型的有限上下文问题。我们的方法MM-assist结合了大型语言模型、集成开发环境、静态分析和语义相关性的优势。在全面的、多方法论的实证评估中，我们对比了MM-assist与现有最先进的方法。MM-assist显著优于它们：（i）在其他研究人员广泛使用的基准上，我们的Recall@1和Recall@3提高了1.7倍；（ii）在210个开源软件的重构语料库上，我们的召回率提高了至少2.4倍。最后，我们进行了一项用户研究，30名经验丰富的参与者使用MM-assist对自己的代码进行了一周的重构。他们中82.8%的MM-assist建议得到了积极评价，这表明MM-assist既有效又实用。 

---
