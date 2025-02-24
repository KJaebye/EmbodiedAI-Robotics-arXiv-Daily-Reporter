# Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions 

**Title (ZH)**: 超越地平线的安全性：基于神经控制障碍函数的高效采样 MPC 

**Authors**: Ji Yin, Oswin So, Eric Yang Yu, Chuchu Fan, Panagiotis Tsiotras  

**Link**: [PDF](https://arxiv.org/pdf/2502.15006)  

**Abstract**: A common problem when using model predictive control (MPC) in practice is the satisfaction of safety specifications beyond the prediction horizon. While theoretical works have shown that safety can be guaranteed by enforcing a suitable terminal set constraint or a sufficiently long prediction horizon, these techniques are difficult to apply and thus are rarely used by practitioners, especially in the case of general nonlinear dynamics. To solve this problem, we impose a tradeoff between exact recursive feasibility, computational tractability, and applicability to ''black-box'' dynamics by learning an approximate discrete-time control barrier function and incorporating it into a variational inference MPC (VIMPC), a sampling-based MPC paradigm. To handle the resulting state constraints, we further propose a new sampling strategy that greatly reduces the variance of the estimated optimal control, improving the sample efficiency, and enabling real-time planning on a CPU. The resulting Neural Shield-VIMPC (NS-VIMPC) controller yields substantial safety improvements compared to existing sampling-based MPC controllers, even under badly designed cost functions. We validate our approach in both simulation and real-world hardware experiments. 

**Abstract (ZH)**: 使用模型预测控制时的一个常见问题是超越预测 horizon 后满足安全规范。 

---
# AutoToM: Automated Bayesian Inverse Planning and Model Discovery for Open-ended Theory of Mind 

**Title (ZH)**: AutoToM: 自动贝叶斯逆规划和模型发现以支持开放性心理理论推理 

**Authors**: Zhining Zhang, Chuanyang Jin, Mung Yao Jia, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15676)  

**Abstract**: Theory of Mind (ToM), the ability to understand people's mental variables based on their behavior, is key to developing socially intelligent agents. Current approaches to Theory of Mind reasoning either rely on prompting Large Language Models (LLMs), which are prone to systematic errors, or use rigid, handcrafted Bayesian Theory of Mind (BToM) models, which are more robust but cannot generalize across different domains. In this work, we introduce AutoToM, an automated Bayesian Theory of Mind method for achieving open-ended machine Theory of Mind. AutoToM can operate in any domain, infer any mental variable, and conduct robust Theory of Mind reasoning of any order. Given a Theory of Mind inference problem, AutoToM first proposes an initial BToM model. It then conducts automated Bayesian inverse planning based on the proposed model, leveraging an LLM as the backend. Based on the uncertainty of the inference, it iteratively refines the model, by introducing additional mental variables and/or incorporating more timesteps in the context. Empirical evaluations across multiple Theory of Mind benchmarks demonstrate that AutoToM consistently achieves state-of-the-art performance, offering a scalable, robust, and interpretable approach to machine Theory of Mind. 

**Abstract (ZH)**: 自动Theory of Mind（ToM）方法：实现开放式的机器Theory of Mind 

---
# Automating Curriculum Learning for Reinforcement Learning using a Skill-Based Bayesian Network 

**Title (ZH)**: 基于技能的贝叶斯网络自动化 Curriculum Learning 在强化学习中的应用 

**Authors**: Vincent Hsiao, Mark Roberts, Laura M. Hiatt, George Konidaris, Dana Nau  

**Link**: [PDF](https://arxiv.org/pdf/2502.15662)  

**Abstract**: A major challenge for reinforcement learning is automatically generating curricula to reduce training time or improve performance in some target task. We introduce SEBNs (Skill-Environment Bayesian Networks) which model a probabilistic relationship between a set of skills, a set of goals that relate to the reward structure, and a set of environment features to predict policy performance on (possibly unseen) tasks. We develop an algorithm that uses the inferred estimates of agent success from SEBN to weigh the possible next tasks by expected improvement. We evaluate the benefit of the resulting curriculum on three environments: a discrete gridworld, continuous control, and simulated robotics. The results show that curricula constructed using SEBN frequently outperform other baselines. 

**Abstract (ZH)**: 强化学习的一个主要挑战是自动生成 Curriculum 以减少训练时间或在某些目标任务中提高性能。我们引入了 SEBNs（Skill-Environment Bayesian Networks），用于建模技能集、与奖励结构相关的目标集以及环境特征之间的概率关系，以预测在（可能未见过的）任务上的策略性能。我们开发了一个算法，该算法根据从 SEBN 推断的智能体成功估计值来权衡可能的下一个任务的预期改进。我们在这三个环境中评估了由 SEBN 构建的 Curriculum 的益处：离散网格世界、连续控制和模拟机器人。结果表明，使用 SEBN 构建的 Curriculum 经常优于其他基准。 

---
# Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path? 

**Title (ZH)**: 超级智能代理存在灾难性风险：科学家AI能提供一条更安全的路径吗？ 

**Authors**: Yoshua Bengio, Michael Cohen, Damiano Fornasiere, Joumana Ghosn, Pietro Greiner, Matt MacDermott, Sören Mindermann, Adam Oberman, Jesse Richardson, Oliver Richardson, Marc-Antoine Rondeau, Pierre-Luc St-Charles, David Williams-King  

**Link**: [PDF](https://arxiv.org/pdf/2502.15657)  

**Abstract**: The leading AI companies are increasingly focused on building generalist AI agents -- systems that can autonomously plan, act, and pursue goals across almost all tasks that humans can perform. Despite how useful these systems might be, unchecked AI agency poses significant risks to public safety and security, ranging from misuse by malicious actors to a potentially irreversible loss of human control. We discuss how these risks arise from current AI training methods. Indeed, various scenarios and experiments have demonstrated the possibility of AI agents engaging in deception or pursuing goals that were not specified by human operators and that conflict with human interests, such as self-preservation. Following the precautionary principle, we see a strong need for safer, yet still useful, alternatives to the current agency-driven trajectory. Accordingly, we propose as a core building block for further advances the development of a non-agentic AI system that is trustworthy and safe by design, which we call Scientist AI. This system is designed to explain the world from observations, as opposed to taking actions in it to imitate or please humans. It comprises a world model that generates theories to explain data and a question-answering inference machine. Both components operate with an explicit notion of uncertainty to mitigate the risks of overconfident predictions. In light of these considerations, a Scientist AI could be used to assist human researchers in accelerating scientific progress, including in AI safety. In particular, our system can be employed as a guardrail against AI agents that might be created despite the risks involved. Ultimately, focusing on non-agentic AI may enable the benefits of AI innovation while avoiding the risks associated with the current trajectory. We hope these arguments will motivate researchers, developers, and policymakers to favor this safer path. 

**Abstract (ZH)**: 领先的AI公司越来越专注于构建通用AI代理——能够在几乎所有人类能够执行的任务上自主规划、行动并追求目标的系统。尽管这些系统可能非常有用，但不受约束的AI自主权对公共安全和安全构成了显著风险，从恶意行为者滥用到人类控制权的潜在不可逆丧失。我们讨论了这些风险是如何从当前的AI培训方法中产生的。确实，各种情景和实验已经证明，AI代理可能进行欺骗或追求人类操作者未指定、且与人类利益冲突的目标，如自我保护。遵循预防原则，我们认为有必要开发一种更安全但仍有用的替代方案。因此，我们提出了一种非自主AI系统的概念作为进一步发展的核心构建块，这种系统以其设计而言是可信且安全的，我们称之为科学家AI。该系统旨在从观察中解释世界，而不是通过采取行动来模仿或取悦人类。它包括一个世界模型，用于生成解释数据的理论，以及一个问答推理机。这两个组件都具有明确的不确定性概念，以降低过度自信预测的风险。鉴于这些考虑，科学家AI可以用于帮助人类研究人员加速科学进步，包括AI安全领域。特别是，我们的系统可以作为防止可能存在的风险的护栏。最终，重点关注非自主AI可能使AI创新的好处得以实现，同时避免当前轨迹相关风险。我们希望这些论点将激励研究人员、开发者和政策制定者选择这条更安全的道路。 

---
# Paradigms of AI Evaluation: Mapping Goals, Methodologies and Culture 

**Title (ZH)**: AI评估范式：目标、方法论与文化映射 

**Authors**: John Burden, Marko Tešić, Lorenzo Pacchiardi, José Hernández-Orallo  

**Link**: [PDF](https://arxiv.org/pdf/2502.15620)  

**Abstract**: Research in AI evaluation has grown increasingly complex and multidisciplinary, attracting researchers with diverse backgrounds and objectives. As a result, divergent evaluation paradigms have emerged, often developing in isolation, adopting conflicting terminologies, and overlooking each other's contributions. This fragmentation has led to insular research trajectories and communication barriers both among different paradigms and with the general public, contributing to unmet expectations for deployed AI systems. To help bridge this insularity, in this paper we survey recent work in the AI evaluation landscape and identify six main paradigms. We characterise major recent contributions within each paradigm across key dimensions related to their goals, methodologies and research cultures. By clarifying the unique combination of questions and approaches associated with each paradigm, we aim to increase awareness of the breadth of current evaluation approaches and foster cross-pollination between different paradigms. We also identify potential gaps in the field to inspire future research directions. 

**Abstract (ZH)**: 人工智能评价研究日益复杂且跨学科化，吸引了具有多样背景和目标的研究者。由此，多种评价范式相继出现，常常独立发展，采用冲突的术语，并忽视彼此的贡献。这种碎片化导致了范式间的孤立研究轨迹和交流障碍，同时也影响了公众对部署的AI系统的期望。为了帮助弥补这种孤立性，本文 surveys 现代人工智能评价领域，并识别出六个主要范式。我们沿着目标、方法论和研究文化等关键维度，描述了每个范式内的主要最新贡献。通过明确每个范式特有的问题组合和方法，旨在提高对当前评价方法多样性的认识，并促进不同范式之间的交流。我们还识别出该领域可能存在的空白，以启发未来的研究方向。 

---
# Zweistein: A Dynamic Programming Evaluation Function for Einstein Würfelt Nicht! 

**Title (ZH)**: Zweistein: 一种动态规划评价函数用于Einstein Würfelt Nicht! 

**Authors**: Wei Lin. Hsueh, Tsan Sheng. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15547)  

**Abstract**: This paper introduces Zweistein, a dynamic programming evaluation function for Einstein Würfelt Nicht! (EWN). Instead of relying on human knowledge to craft an evaluation function, Zweistein uses a data-centric approach that eliminates the need for parameter tuning. The idea is to use a vector recording the distance to the corner of all pieces. This distance vector captures the essence of EWN. It not only outperforms many traditional EWN evaluation functions but also won first place in the TCGA 2023 competition. 

**Abstract (ZH)**: 本文介绍了Zweistein，一种用于Einstein Würfelt Nicht!（EWN）的动态规划评估函数。Zweistein采用以数据为中心的方法，不需要人工知识来设计评估函数，也不需要参数调优。该方法通过记录所有棋子到角落的距离向量来捕捉EWN的本质。Zweistein不仅在许多传统EWN评估函数中表现更优，还在2023年TCGA比赛中获得冠军。 

---
# TAG: A Decentralized Framework for Multi-Agent Hierarchical Reinforcement Learning 

**Title (ZH)**: TAG：多代理层次强化学习的去中心化框架 

**Authors**: Giuseppe Paolo, Abdelhakim Benechehab, Hamza Cherkaoui, Albert Thomas, Balázs Kégl  

**Link**: [PDF](https://arxiv.org/pdf/2502.15425)  

**Abstract**: Hierarchical organization is fundamental to biological systems and human societies, yet artificial intelligence systems often rely on monolithic architectures that limit adaptability and scalability. Current hierarchical reinforcement learning (HRL) approaches typically restrict hierarchies to two levels or require centralized training, which limits their practical applicability. We introduce TAME Agent Framework (TAG), a framework for constructing fully decentralized hierarchical multi-agent this http URL enables hierarchies of arbitrary depth through a novel LevelEnv concept, which abstracts each hierarchy level as the environment for the agents above it. This approach standardizes information flow between levels while preserving loose coupling, allowing for seamless integration of diverse agent types. We demonstrate the effectiveness of TAG by implementing hierarchical architectures that combine different RL agents across multiple levels, achieving improved performance over classical multi-agent RL baselines on standard benchmarks. Our results show that decentralized hierarchical organization enhances both learning speed and final performance, positioning TAG as a promising direction for scalable multi-agent systems. 

**Abstract (ZH)**: TAME智能体框架：一种促进任意深度去中心化层次多智能体系统的范式 

---
# Measuring AI agent autonomy: Towards a scalable approach with code inspection 

**Title (ZH)**: 基于代码审查的可扩展方法：衡量AI代理自主性研究 

**Authors**: Peter Cihon, Merlin Stein, Gagan Bansal, Sam Manning, Kevin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15212)  

**Abstract**: AI agents are AI systems that can achieve complex goals autonomously. Assessing the level of agent autonomy is crucial for understanding both their potential benefits and risks. Current assessments of autonomy often focus on specific risks and rely on run-time evaluations -- observations of agent actions during operation. We introduce a code-based assessment of autonomy that eliminates the need to run an AI agent to perform specific tasks, thereby reducing the costs and risks associated with run-time evaluations. Using this code-based framework, the orchestration code used to run an AI agent can be scored according to a taxonomy that assesses attributes of autonomy: impact and oversight. We demonstrate this approach with the AutoGen framework and select applications. 

**Abstract (ZH)**: AI代理是能够自主实现复杂目标的AI系统。评估代理的自主水平对于理解其潜在利益和风险至关重要。当前对自主性的评估往往集中在特定风险上，并依赖于运行时评估——操作期间对代理行为的观察。我们引入了一种基于代码的自主性评估方法，无需运行AI代理执行特定任务，从而降低了运行时评估相关的成本和风险。使用这种方法，可以通过一种根据自主性属性（影响和监管）进行分类的框架来评估运行AI代理所使用的编排代码。我们通过AutoGen框架和选定的应用程序展示了这种做法。 

---
# The Imitation Game for Educational AI 

**Title (ZH)**: 教育人工智能的模仿游戏 

**Authors**: Shashank Sonkar, Naiming Liu, Xinghe Chen, Richard G. Baraniuk  

**Link**: [PDF](https://arxiv.org/pdf/2502.15127)  

**Abstract**: As artificial intelligence systems become increasingly prevalent in education, a fundamental challenge emerges: how can we verify if an AI truly understands how students think and reason? Traditional evaluation methods like measuring learning gains require lengthy studies confounded by numerous variables. We present a novel evaluation framework based on a two-phase Turing-like test. In Phase 1, students provide open-ended responses to questions, revealing natural misconceptions. In Phase 2, both AI and human experts, conditioned on each student's specific mistakes, generate distractors for new related questions. By analyzing whether students select AI-generated distractors at rates similar to human expert-generated ones, we can validate if the AI models student cognition. We prove this evaluation must be conditioned on individual responses - unconditioned approaches merely target common misconceptions. Through rigorous statistical sampling theory, we establish precise requirements for high-confidence validation. Our research positions conditioned distractor generation as a probe into an AI system's fundamental ability to model student thinking - a capability that enables adapting tutoring, feedback, and assessments to each student's specific needs. 

**Abstract (ZH)**: 随着人工智能系统在教育领域中的日益普及，一个基本的挑战出现了：我们如何验证AI是否真正理解了学生的思维方式和推理过程？传统的评估方法如衡量学习成果需要耗时的研究，并受到众多变量的干扰。我们提出了一种基于两阶段图灵测试样式的新型评估框架。在第一阶段，学生对问题提供开放式的回答，揭示自然存在的误解。在第二阶段，基于每个学生特定的错误，AI和人类专家生成新的相关问题的干扰选项。通过分析学生选择AI生成的干扰选项的比例是否与人类专家生成的干扰选项比例相似，我们可以验证AI是否能够模拟学生的认知。我们证明这种评估必须基于个体反应——未基于个体反应的方法仅针对共通的误解。通过严格的统计抽样理论，我们确立了高置信度验证所需的精确要求。我们的研究将基于个体的干扰选项生成定位为探索AI系统基本能力的一种探针，该能力能够根据不同学生的需求调整辅导、反馈和评估。 

---
# GenAI vs. Human Fact-Checkers: Accurate Ratings, Flawed Rationales 

**Title (ZH)**: GenAI与人类事实核查者：准确评分，缺陷理由 

**Authors**: Yuehong Cassandra Tai, Khushi Navin Patni, Nicholas Daniel Hemauer, Bruce Desmarais, Yu-Ru Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.14943)  

**Abstract**: Despite recent advances in understanding the capabilities and limits of generative artificial intelligence (GenAI) models, we are just beginning to understand their capacity to assess and reason about the veracity of content. We evaluate multiple GenAI models across tasks that involve the rating of, and perceived reasoning about, the credibility of information. The information in our experiments comes from content that subnational U.S. politicians post to Facebook. We find that GPT-4o, one of the most used AI models in consumer applications, outperforms other models, but all models exhibit only moderate agreement with human coders. Importantly, even when GenAI models accurately identify low-credibility content, their reasoning relies heavily on linguistic features and ``hard'' criteria, such as the level of detail, source reliability, and language formality, rather than an understanding of veracity. We also assess the effectiveness of summarized versus full content inputs, finding that summarized content holds promise for improving efficiency without sacrificing accuracy. While GenAI has the potential to support human fact-checkers in scaling misinformation detection, our results caution against relying solely on these models. 

**Abstract (ZH)**: 尽管近年来我们对生成式人工智能（GenAI）模型的能力和局限有了更深入的理解，但我们刚刚开始探索它们评估和推理信息真实性的能力。我们评估了多个GenAI模型在涉及信息信誉评级和感知推理的任务中的表现，实验中的信息来自美国地方政治家在Facebook上发布的内容。我们发现，GPT-4o，一种在消费者应用中最常用的人工智能模型，表现优于其他模型，但所有模型与人类编码者的共识程度仅呈中等水平。重要的是，即使GenAI模型能够准确识别低可信度内容，它们的推理也主要基于语言特征和“硬”标准，如细节程度、信息源可靠性及语言形式化，而缺乏对真实性的理解。我们还评估了总结版内容和全文内容输入的有效性，发现总结版内容有可能在提高效率的同时不牺牲准确性。虽然GenAI有潜力支持人类事实核查人员扩大对虚假信息的检测，但我们的研究结果提醒我们不要过度依赖这些模型。 

---
# One-step Diffusion Models with $f$-Divergence Distribution Matching 

**Title (ZH)**: 一步扩散模型ewith $f$-散度分布匹配 

**Authors**: Yilun Xu, Weili Nie, Arash Vahdat  

**Link**: [PDF](https://arxiv.org/pdf/2502.15681)  

**Abstract**: Sampling from diffusion models involves a slow iterative process that hinders their practical deployment, especially for interactive applications. To accelerate generation speed, recent approaches distill a multi-step diffusion model into a single-step student generator via variational score distillation, which matches the distribution of samples generated by the student to the teacher's distribution. However, these approaches use the reverse Kullback-Leibler (KL) divergence for distribution matching which is known to be mode seeking. In this paper, we generalize the distribution matching approach using a novel $f$-divergence minimization framework, termed $f$-distill, that covers different divergences with different trade-offs in terms of mode coverage and training variance. We derive the gradient of the $f$-divergence between the teacher and student distributions and show that it is expressed as the product of their score differences and a weighting function determined by their density ratio. This weighting function naturally emphasizes samples with higher density in the teacher distribution, when using a less mode-seeking divergence. We observe that the popular variational score distillation approach using the reverse-KL divergence is a special case within our framework. Empirically, we demonstrate that alternative $f$-divergences, such as forward-KL and Jensen-Shannon divergences, outperform the current best variational score distillation methods across image generation tasks. In particular, when using Jensen-Shannon divergence, $f$-distill achieves current state-of-the-art one-step generation performance on ImageNet64 and zero-shot text-to-image generation on MS-COCO. Project page: this https URL 

**Abstract (ZH)**: 基于扩散模型的采样过程涉及缓慢的迭代过程，阻碍了它们的实际部署，尤其是对于交互式应用。为了加速生成速度，近期的方法通过变分得分蒸馏将多步扩散模型精简为单步的学生生成器，使其生成的样本分布与教师模型的分布匹配。然而，这些方法使用了逆Kullback-Leibler（KL）散度进行分布匹配，它是有模式寻求性的。在本文中，我们通过一个新的$f$-散度最小化框架$f$-distill，对分布匹配方法进行了泛化，该框架涵盖了不同模式覆盖和训练方差之间不同权衡的多种散度。我们推导了教师和学生分布之间$f$-散度的梯度，并展示了它可以用它们的得分差异和由密度比决定的加权函数的乘积来表示。当使用非模式寻求性较低的散度时，该加权函数自然地强调了教师分布中密度更高的样本。我们观察到，流行的使用逆KL散度的变分得分蒸馏方法是该框架的一个特例。实验上，我们展示了其他$f$-散度，如正向KL和Jensen-Shannon散度，在图像生成任务中优于当前最佳的变分得分蒸馏方法。特别是，在使用Jensen-Shannon散度时，$f$-distill在ImageNet64的一步生成性能和MS-COCO的零样本文本到图像生成中达到了当前最佳水平。项目页面：https://github.com/alibaba/Qwen-f-distill 

---
# Multi-Agent Architecture in Distributed Environment Control Systems: vision, challenges, and opportunities 

**Title (ZH)**: 分布式环境控制系统中的多代理架构：愿景、挑战与机遇 

**Authors**: Natasha Astudillo, Fernando Koch  

**Link**: [PDF](https://arxiv.org/pdf/2502.15663)  

**Abstract**: The increasing demand for energy-efficient solutions in large-scale infrastructure, particularly data centers, requires advanced control strategies to optimize environmental management systems. We propose a multi-agent architecture for distributed control of air-cooled chiller systems in data centers. Our vision employs autonomous agents to monitor and regulate local operational parameters and optimize system-wide efficiency. We demonstrate how this approach improves the responsiveness, operational robustness, and energy efficiency of the system, contributing to the broader goal of sustainable infrastructure management. 

**Abstract (ZH)**: 大规模基础设施，特别是数据中心，对能源高效解决方案的需求日益增长，需要先进的控制策略来优化环境管理系统。我们提出了一种多代理架构，用于数据中空气冷凝器系统的分布式控制。我们的愿景是采用自主代理监控和调节局部操作参数，优化系统整体效率。我们展示了这种做法如何提高系统的响应性、操作稳健性和能源效率，从而为可持续基础设施管理的整体目标做出贡献。 

---
# AutoTandemML: Active Learning Enhanced Tandem Neural Networks for Inverse Design Problems 

**Title (ZH)**: AutoTandemML：增强主动学习的串联神经网络用于逆向设计问题 

**Authors**: Luka Grbcic, Juliane Müller, Wibe Albert de Jong  

**Link**: [PDF](https://arxiv.org/pdf/2502.15643)  

**Abstract**: Inverse design in science and engineering involves determining optimal design parameters that achieve desired performance outcomes, a process often hindered by the complexity and high dimensionality of design spaces, leading to significant computational costs. To tackle this challenge, we propose a novel hybrid approach that combines active learning with Tandem Neural Networks to enhance the efficiency and effectiveness of solving inverse design problems. Active learning allows to selectively sample the most informative data points, reducing the required dataset size without compromising accuracy. We investigate this approach using three benchmark problems: airfoil inverse design, photonic surface inverse design, and scalar boundary condition reconstruction in diffusion partial differential equations. We demonstrate that integrating active learning with Tandem Neural Networks outperforms standard approaches across the benchmark suite, achieving better accuracy with fewer training samples. 

**Abstract (ZH)**: 科学与工程中的逆向设计涉及确定能够实现期望性能结果的最优设计参数，这一过程常因设计空间的复杂性和高维度而受到阻碍，导致显著的计算成本。为解决这一挑战，我们提出了一种新的混合方法，该方法结合了主动学习与串联神经网络，以提高解决逆向设计问题的效率和有效性。主动学习允许选择性地采样最有信息量的数据点，从而减少所需的数据库规模而不牺牲准确性。我们使用三个基准问题来研究这种方法：翼型逆向设计、光子表面逆向设计以及扩散偏微分方程中的标量边界条件重构。我们证明，将主动学习与串联神经网络相结合在基准测试套件中优于标准方法，能够在较少的训练样本下实现更好的准确性。 

---
# Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification 

**Title (ZH)**: Mantis: 轻量级校准基础模型用于用户友好的时间序列分类 

**Authors**: Vasilii Feofanov, Songkang Wen, Marius Alonso, Romain Ilbert, Hongbo Guo, Malik Tiomoko, Lujia Pan, Jianfeng Zhang, Ievgen Redko  

**Link**: [PDF](https://arxiv.org/pdf/2502.15637)  

**Abstract**: In recent years, there has been increasing interest in developing foundation models for time series data that can generalize across diverse downstream tasks. While numerous forecasting-oriented foundation models have been introduced, there is a notable scarcity of models tailored for time series classification. To address this gap, we present Mantis, a new open-source foundation model for time series classification based on the Vision Transformer (ViT) architecture that has been pre-trained using a contrastive learning approach. Our experimental results show that Mantis outperforms existing foundation models both when the backbone is frozen and when fine-tuned, while achieving the lowest calibration error. In addition, we propose several adapters to handle the multivariate setting, reducing memory requirements and modeling channel interdependence. 

**Abstract (ZH)**: 近年来，人们日益关注开发能够跨多种下游任务泛化的时序数据基础模型。尽管已经提出了众多面向预测的基础模型，但专门针对时序分类的任务模型却相对匮乏。为填补这一空白，我们提出了一种基于Vision Transformer (ViT) 架构的新开源时序分类基础模型Mantis，该模型通过对比学习方式进行预训练。实验结果表明，无论是在冻结主干网络的情况下还是在微调情况下，Mantis 均表现出色，并实现了最低的校准误差。此外，我们还提出了一些适配器以处理多变量设置，降低内存需求并建模通道间的依赖关系。 

---
# Dynamic Knowledge Selector and Evaluator for recommendation with Knowledge Graph 

**Title (ZH)**: 知识图谱引导的动态知识选择与评估推荐方法 

**Authors**: Feng Xia, Zhifei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15623)  

**Abstract**: In recent years recommendation systems typically employ the edge information provided by knowledge graphs combined with the advantages of high-order connectivity of graph networks in the recommendation field. However, this method is limited by the sparsity of labels, cannot learn the graph structure well, and a large number of noisy entities in the knowledge graph will affect the accuracy of the recommendation results. In order to alleviate the above problems, we propose a dynamic knowledge-selecting and evaluating method guided by collaborative signals to distill information in the knowledge graph. Specifically, we use a Chain Route Evaluator to evaluate the contributions of different neighborhoods for the recommendation task and employ a Knowledge Selector strategy to filter the less informative knowledge before evaluating. We conduct baseline model comparison and experimental ablation evaluations on three public datasets. The experiments demonstrate that our proposed model outperforms current state-of-the-art baseline models, and each modules effectiveness in our model is demonstrated through ablation experiments. 

**Abstract (ZH)**: 近年来，推荐系统通常利用知识图谱提供的边缘信息，并结合图网络在推荐领域的高阶连接优势。然而，这种方法受限于标签的稀疏性，无法很好地学习图结构， knowledge graph 中大量噪声实体会影响推荐结果的准确性。为了缓解上述问题，我们提出了一种由协作信号引导的知识选择和评估动态方法，以提炼知识图谱中的信息。具体而言，我们使用链路路径评估器评估不同邻域对推荐任务的贡献，并采用知识选择策略过滤掉不具信息量的知识后再进行评估。我们在三个公开数据集上进行基准模型比较和实验消融评估。实验结果表明，我们提出的模型优于当前最先进的基准模型，并且通过消融实验展示了我们模型中每个模块的有效性。 

---
# Extraction multi-étiquettes de relations en utilisant des couches de Transformer 

**Title (ZH)**: 使用Transformer层进行多标签关系提取 

**Authors**: Ngoc Luyen Le, Gildas Tagny Ngompé  

**Link**: [PDF](https://arxiv.org/pdf/2502.15619)  

**Abstract**: In this article, we present the BTransformer18 model, a deep learning architecture designed for multi-label relation extraction in French texts. Our approach combines the contextual representation capabilities of pre-trained language models from the BERT family - such as BERT, RoBERTa, and their French counterparts CamemBERT and FlauBERT - with the power of Transformer encoders to capture long-term dependencies between tokens. Experiments conducted on the dataset from the TextMine'25 challenge show that our model achieves superior performance, particularly when using CamemBERT-Large, with a macro F1 score of 0.654, surpassing the results obtained with FlauBERT-Large. These results demonstrate the effectiveness of our approach for the automatic extraction of complex relations in intelligence reports. 

**Abstract (ZH)**: 本文介绍了BTransformer18模型，这是一种用于法语文本多标签关系抽取的深度学习架构。我们的方法结合了来自BERT家族（包括BERT、RoBERTa及其法语版本CamemBERT和FlauBERT）的预训练语言模型的上下文表示能力，以及Transformer编码器捕捉长距离依赖的能力。在TextMine'25挑战数据集上的实验结果显示，我们的模型表现优异，特别是使用CamemBERT-Large时，宏F1分数达到0.654，超过了使用FlauBERT-Large的结果。这些结果证明了我们的方法在自动提取情报报告中复杂关系的有效性。 

---
# Pastiche Novel Generation Creating: Fan Fiction You Love in Your Favorite Author's Style 

**Title (ZH)**: Pastiche小说生成创作：喜爱作者风格的粉丝小说 

**Authors**: Xueran Han, Yuhan Liu, Mingzhe Li, Wei Liu, Sen Hu, Rui Yan, Zhiqiang Xu, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15616)  

**Abstract**: Great novels create immersive worlds with rich character arcs, well-structured plots, and nuanced writing styles. However, current novel generation methods often rely on brief, simplistic story outlines and generate details using plain, generic language. To bridge this gap, we introduce the task of Pastiche Novel Generation, which requires the generated novels to imitate the distinctive features of the original work, including understanding character profiles, predicting plausible plot developments, and writing concrete details using vivid, expressive language. To achieve this, we propose WriterAgent, a novel generation system designed to master the core aspects of literary pastiche. WriterAgent is trained through a curriculum learning paradigm, progressing from low-level stylistic mastery to high-level narrative coherence. Its key tasks include language style learning, character modeling, plot planning, and stylish writing, ensuring comprehensive narrative control. To support this, WriterAgent leverages the WriterLoRA framework, an extension of LoRA with hierarchical and cumulative task-specific modules, each specializing in a different narrative aspect. We evaluate WriterAgent on multilingual classics like Harry Potter and Dream of the Red Chamber, demonstrating its superiority over baselines in capturing the target author's settings, character dynamics, and writing style to produce coherent, faithful narratives. 

**Abstract (ZH)**: 伟大的小说营造出丰富的人物弧光、精细的情节结构和细腻的写作风格。然而，当前的小说生成方法往往依赖于简短肤浅的故事概要，并使用平凡的通用语言生成细节。为了解决这一问题，我们引入了仿作小说生成的任务，要求生成的小说模仿原著的独特特征，包括理解人物形象、预测合乎情理的情节发展，并用生动的表达方式编写具体的细节。为此，我们提出了WriterAgent这一新的生成系统，旨在掌握文学仿作的核心方面。WriterAgent通过渐进式学习的范式进行训练，从低级的风格掌握进展到高级的情节连贯性。其关键任务包括语言风格学习、人物建模、情节规划和风格化写作，确保全面的情节控制。为了支持这一点，WriterAgent利用了WriterLoRA框架，这是一种扩展了LoRA的层级和累积任务特定模块的框架，每个模块专门针对不同的叙事方面。我们在哈利·波特和红楼梦等多语言经典作品上评估了WriterAgent，结果显示其在捕捉目标作者的场景设定、人物动态和写作风格方面优于基线模型，能够生成连贯且忠实的叙事。 

---
# PDeepPP:A Deep learning framework with Pretrained Protein language for peptide classification 

**Title (ZH)**: PDeepPP：预训练蛋白质语言的深度学习框架用于肽分类 

**Authors**: Jixiu Zhai, Tianchi Lu, Haitian Zhong, Ziyang Xu, Yuhuan Liu, Xueying Wang, Dan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15610)  

**Abstract**: Protein post-translational modifications (PTMs) and bioactive peptides (BPs) play critical roles in various biological processes and have significant therapeutic potential. However, identifying PTM sites and bioactive peptides through experimental methods is often labor-intensive, costly, and time-consuming. As a result, computational tools, particularly those based on deep learning, have become effective solutions for predicting PTM sites and peptide bioactivity. Despite progress in this field, existing methods still struggle with the complexity of protein sequences and the challenge of requiring high-quality predictions across diverse datasets.
To address these issues, we propose a deep learning framework that integrates pretrained protein language models with a neural network combining transformer and CNN for peptide classification. By leveraging the ability of pretrained models to capture complex relationships within protein sequences, combined with the predictive power of parallel networks, our approach improves feature extraction while enhancing prediction accuracy.
This framework was applied to multiple tasks involving PTM site and bioactive peptide prediction, utilizing large-scale datasets to enhance the model's robustness. In the comparison across 33 tasks, the model achieved state-of-the-art (SOTA) performance in 25 of them, surpassing existing methods and demonstrating its versatility across different datasets. Our results suggest that this approach provides a scalable and effective solution for large-scale peptide discovery and PTM analysis, paving the way for more efficient peptide classification and functional annotation. 

**Abstract (ZH)**: 蛋白质翻译后修饰（PTMs）和活性肽（BPs）在各种生物过程中扮演着关键角色，并具有重要的 therapeutic 潜力。然而，通过实验方法识别 PTM 位点和活性肽往往是耗时、耗力且成本高的。因此，特别是基于深度学习的计算工具已成为预测 PTM 位点和肽活性的有效解决方案。尽管在该领域取得了一定进展，但现有方法仍难以应对蛋白质序列的复杂性和高质量预测在不同数据集中的挑战。

为了应对这些问题，我们提出了一种深度学习框架，该框架集成预训练的蛋白质语言模型与结合变压器和CNN的神经网络进行肽分类。通过利用预训练模型捕捉蛋白质序列内部复杂关系的能力，并结合并行网络的预测能力，我们的方法在提高特征提取的同时增强了预测准确性。

该框架应用于多个涉及 PTM 位点和活性肽预测的任务，利用大规模数据集增强模型的鲁棒性。在对33个任务的比较中，该模型在25个任务中达到了最先进的性能（SOTA），超越了现有方法，并展示了其在不同数据集中的通用性。我们的结果表明，该方法为大规模肽发现和PTM分析提供了一种可扩展且有效的解决方案，为更高效的肽分类和功能注释铺平了道路。 

---
# KAD: No More FAD! An Effective and Efficient Evaluation Metric for Audio Generation 

**Title (ZH)**: KAD: 无更多FAD！一种有效的音频生成评估指标 

**Authors**: Yoonjin Chung, Pilsun Eu, Junwon Lee, Keunwoo Choi, Juhan Nam, Ben Sangbae Chon  

**Link**: [PDF](https://arxiv.org/pdf/2502.15602)  

**Abstract**: Although being widely adopted for evaluating generated audio signals, the Fréchet Audio Distance (FAD) suffers from significant limitations, including reliance on Gaussian assumptions, sensitivity to sample size, and high computational complexity. As an alternative, we introduce the Kernel Audio Distance (KAD), a novel, distribution-free, unbiased, and computationally efficient metric based on Maximum Mean Discrepancy (MMD). Through analysis and empirical validation, we demonstrate KAD's advantages: (1) faster convergence with smaller sample sizes, enabling reliable evaluation with limited data; (2) lower computational cost, with scalable GPU acceleration; and (3) stronger alignment with human perceptual judgments. By leveraging advanced embeddings and characteristic kernels, KAD captures nuanced differences between real and generated audio. Open-sourced in the kadtk toolkit, KAD provides an efficient, reliable, and perceptually aligned benchmark for evaluating generative audio models. 

**Abstract (ZH)**: 尽管广泛应用于评估生成的音频信号，Fréchet 音频距离（FAD）存在显著的局限性，包括依赖高斯假设、对样本大小敏感以及计算复杂度高。作为替代方案，我们引入了基于最大均值偏差（MMD）的核音频距离（KAD），这是一种新的、无需分布假设、无偏且计算高效的度量标准。通过分析和实证验证，我们展示了KAD的优势：（1）在较小的样本大小下更快收敛，从而能够在有限数据下获得可靠的评估；（2）计算成本更低，具有可扩展的GPU加速；（3）更贴近人类的感知判断。通过利用高级嵌入和特征核，KAD 能够捕捉真实音频和生成音频之间的细微差异。KAD 在 kadtk 工具包中开源，提供了一种高效、可靠且感知对齐的基准，用于评估生成音频模型。 

---
# Improving the Scaling Laws of Synthetic Data with Deliberate Practice 

**Title (ZH)**: 通过精心练习改善合成数据的缩放定律 

**Authors**: Reyhane Askari-Hemmat, Mohammad Pezeshki, Elvis Dohmatob, Florian Bordes, Pietro Astolfi, Melissa Hall, Jakob Verbeek, Michal Drozdzal, Adriana Romero-Soriano  

**Link**: [PDF](https://arxiv.org/pdf/2502.15588)  

**Abstract**: Inspired by the principle of deliberate practice in human learning, we propose Deliberate Practice for Synthetic Data Generation (DP), a novel framework that improves sample efficiency through dynamic synthetic data generation. Prior work has shown that scaling synthetic data is inherently challenging, as naively adding new data leads to diminishing returns. To address this, pruning has been identified as a key mechanism for improving scaling, enabling models to focus on the most informative synthetic samples. Rather than generating a large dataset and pruning it afterward, DP efficiently approximates the direct generation of informative samples. We theoretically show how training on challenging, informative examples improves scaling laws and empirically validate that DP achieves better scaling performance with significantly fewer training samples and iterations. On ImageNet-100, DP generates 3.4x fewer samples and requires six times fewer iterations, while on ImageNet-1k, it generates 8x fewer samples with a 30 percent reduction in iterations, all while achieving superior performance compared to prior work. 

**Abstract (ZH)**: 基于人类学习刻意练习原则的合成数据生成刻意练习（DP）框架：通过动态合成数据生成提高样本效率 

---
# Feature maps for the Laplacian kernel and its generalizations 

**Title (ZH)**: 拉普拉斯核及其推广的特征映射 

**Authors**: Sudhendu Ahir, Parthe Pandit  

**Link**: [PDF](https://arxiv.org/pdf/2502.15575)  

**Abstract**: Recent applications of kernel methods in machine learning have seen a renewed interest in the Laplacian kernel, due to its stability to the bandwidth hyperparameter in comparison to the Gaussian kernel, as well as its expressivity being equivalent to that of the neural tangent kernel of deep fully connected networks. However, unlike the Gaussian kernel, the Laplacian kernel is not separable. This poses challenges for techniques to approximate it, especially via the random Fourier features (RFF) methodology and its variants. In this work, we provide random features for the Laplacian kernel and its two generalizations: Matérn kernel and the Exponential power kernel. We provide efficiently implementable schemes to sample weight matrices so that random features approximate these kernels. These weight matrices have a weakly coupled heavy-tailed randomness. Via numerical experiments on real datasets we demonstrate the efficacy of these random feature maps. 

**Abstract (ZH)**: 最近机器学习中核方法的应用重新引起了对拉普拉斯核的兴趣，由于其在带宽超参数方面比高斯核更具稳定性，以及其表达能力与深层全连接网络的神经相切核相当。然而，与高斯核不同，拉普拉斯核不具备可分性。这给其近似技术，尤其是通过随机傅里叶特征（RFF）方法及其变体实现带来了挑战。在本文中，我们提供了拉普拉斯核及其两种推广形式——Matérn核和指数幂核的随机特征表示，并提供了高效实施的方案以采样权重矩阵，使得随机特征能够近似这些核。这些权重矩阵具有弱关联的重尾随机性。通过在真实数据集上的数值实验，我们展示了这些随机特征映射的有效性。 

---
# BAN: Neuroanatomical Aligning in Auditory Recognition between Artificial Neural Network and Human Cortex 

**Title (ZH)**: BAN: 人工神经网络与人类皮层在听觉识别中的神经解剖对齐 

**Authors**: Haidong Wang, Pengfei Xiao, Ao Liu, Jianhua Zhang, Qia Shan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15503)  

**Abstract**: Drawing inspiration from neurosciences, artificial neural networks (ANNs) have evolved from shallow architectures to highly complex, deep structures, yielding exceptional performance in auditory recognition tasks. However, traditional ANNs often struggle to align with brain regions due to their excessive depth and lack of biologically realistic features, like recurrent connection. To address this, a brain-like auditory network (BAN) is introduced, which incorporates four neuroanatomically mapped areas and recurrent connection, guided by a novel metric called the brain-like auditory score (BAS). BAS serves as a benchmark for evaluating the similarity between BAN and human auditory recognition pathway. We further propose that specific areas in the cerebral cortex, mainly the middle and medial superior temporal (T2/T3) areas, correspond to the designed network structure, drawing parallels with the brain's auditory perception pathway. Our findings suggest that the neuroanatomical similarity in the cortex and auditory classification abilities of the ANN are well-aligned. In addition to delivering excellent performance on a music genre classification task, the BAN demonstrates a high BAS score. In conclusion, this study presents BAN as a recurrent, brain-inspired ANN, representing the first model that mirrors the cortical pathway of auditory recognition. 

**Abstract (ZH)**: 来自神经科学的启发，人工神经网络（ANNs）从浅层架构发展成为高度复杂的深层结构，在听觉识别任务中取得了卓越的性能。然而，传统ANNs往往因其过度的深度和缺乏生物学现实性的特征（如循环连接）而难以与脑区对齐。为解决这一问题，引入了一种脑启发式听觉网络（BAN），该网络整合了四个神经解剖学映射区域和循环连接，并以一种新的指标——脑启发式听觉评分（BAS）为指导。BAS用作评估BAN与人类听觉识别路径相似性的基准。进一步的研究表明，大脑皮层中的特定区域，主要是上顶颞中回（T2/T3）区域，对应于设计的网络结构，与大脑的听觉感知路径相吻合。研究发现表明，大脑皮层的神经解剖相似性和ANN的听觉分类能力是高度协调的。除了在音乐流派分类任务中表现出色外，BAN还具有高BAS评分。总之，本研究展示了BAN作为一种具有循环连接的脑启发式ANN，是首个模拟听觉识别大脑皮层路径的模型。 

---
# Mitigating Data Scarcity in Time Series Analysis: A Foundation Model with Series-Symbol Data Generation 

**Title (ZH)**: 在时间序列分析中缓解数据稀缺性：基于系列符号数据生成的foundation模型 

**Authors**: Wenxuan Wang, Kai Wu, Yujian Betterest Li, Dan Wang, Xiaoyu Zhang, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15466)  

**Abstract**: Foundation models for time series analysis (TSA) have attracted significant attention. However, challenges such as data scarcity and data imbalance continue to hinder their development. To address this, we consider modeling complex systems through symbolic expressions that serve as semantic descriptors of time series. Building on this concept, we introduce a series-symbol (S2) dual-modulity data generation mechanism, enabling the unrestricted creation of high-quality time series data paired with corresponding symbolic representations. Leveraging the S2 dataset, we develop SymTime, a pre-trained foundation model for TSA. SymTime demonstrates competitive performance across five major TSA tasks when fine-tuned with downstream task, rivaling foundation models pre-trained on real-world datasets. This approach underscores the potential of dual-modality data generation and pretraining mechanisms in overcoming data scarcity and enhancing task performance. 

**Abstract (ZH)**: 符号表达驱动的时空序列分析基础模型 

---
# Fed-SB: A Silver Bullet for Extreme Communication Efficiency and Performance in (Private) Federated LoRA Fine-Tuning 

**Title (ZH)**: Fed-SB: 极端通信效率和性能的银弹方案在(私有)联邦LoRA微调中 

**Authors**: Raghav Singhal, Kaustubh Ponkshe, Rohit Vartak, Lav R. Varshney, Praneeth Vepakomma  

**Link**: [PDF](https://arxiv.org/pdf/2502.15436)  

**Abstract**: Low-Rank Adaptation (LoRA) has become ubiquitous for efficiently fine-tuning foundation models. However, federated fine-tuning using LoRA is challenging due to suboptimal updates arising from traditional federated averaging of individual adapters. Existing solutions either incur prohibitively high communication cost that scales linearly with the number of clients or suffer from performance degradation due to limited expressivity. We introduce Federated Silver Bullet (Fed-SB), a novel approach for federated fine-tuning of LLMs using LoRA-SB, a recently proposed low-rank adaptation method. LoRA-SB optimally aligns the optimization trajectory with the ideal low-rank full fine-tuning projection by learning a small square matrix (R) between adapters B and A, keeping other components fixed. Direct averaging of R guarantees exact updates, substantially reducing communication cost, which remains independent of the number of clients, and enables scalability. Fed-SB achieves state-of-the-art performance across commonsense reasoning, arithmetic reasoning, and language inference tasks while reducing communication costs by up to 230x. In private settings, Fed-SB further improves performance by (1) reducing trainable parameters, thereby lowering the noise required for differential privacy and (2) avoiding noise amplification introduced by other methods. Overall, Fed-SB establishes a new Pareto frontier in the tradeoff between communication and performance, offering an efficient and scalable solution for both private and non-private federated fine-tuning. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 联邦自适应银弹（Fed-SB）：基于LoRA-SB的高效可扩展的LLM联邦微调 

---
# Anatomy-Informed Deep Learning and Radiomics for Automated Neurofibroma Segmentation in Whole-Body MRI 

**Title (ZH)**: 基于解剖学信息的深度学习和影像组学在全身MRI中自动神经纤维瘤分割中的应用 

**Authors**: Georgii Kolokolnikov, Marie-Lena Schmalhofer, Lennart Well, Said Farschtschi, Victor-Felix Mautner, Inka Ristow, Rene Werner  

**Link**: [PDF](https://arxiv.org/pdf/2502.15424)  

**Abstract**: Neurofibromatosis Type 1 is a genetic disorder characterized by the development of neurofibromas (NFs), which exhibit significant variability in size, morphology, and anatomical location. Accurate and automated segmentation of these tumors in whole-body magnetic resonance imaging (WB-MRI) is crucial to assess tumor burden and monitor disease progression. In this study, we present and analyze a fully automated pipeline for NF segmentation in fat-suppressed T2-weighted WB-MRI, consisting of three stages: anatomy segmentation, NF segmentation, and tumor candidate classification. In the first stage, we use the MRSegmentator model to generate an anatomy segmentation mask, extended with a high-risk zone for NFs. This mask is concatenated with the input image as anatomical context information for NF segmentation. The second stage employs an ensemble of 3D anisotropic anatomy-informed U-Nets to produce an NF segmentation confidence mask. In the final stage, tumor candidates are extracted from the confidence mask and classified based on radiomic features, distinguishing tumors from non-tumor regions and reducing false positives. We evaluate the proposed pipeline on three test sets representing different conditions: in-domain data (test set 1), varying imaging protocols and field strength (test set 2), and low tumor burden cases (test set 3). Experimental results show a 68% improvement in per-scan Dice Similarity Coefficient (DSC), a 21% increase in per-tumor DSC, and a two-fold improvement in F1 score for tumor detection in high tumor burden cases by integrating anatomy information. The method is integrated into the 3D Slicer platform for practical clinical use, with the code publicly accessible. 

**Abstract (ZH)**: 神经纤维瘤病1型是一种由神经纤维瘤（NFs）发展引起的遗传性疾病，这些肿瘤在大小、形态和解剖位置上表现出显著的变异。对全身磁共振成像（WB-MRI）进行准确且自动的NF分割对于评估肿瘤负荷和监测疾病进展至关重要。本研究提出并分析了一种完全自动的NF分割管道，该管道由三个阶段组成：解剖分割、NF分割和肿瘤候选区域分类。在第一阶段，我们使用MRSegmentator模型生成包括NF高风险区的解剖分割掩模。该掩模与输入图像拼接，作为NF分割的解剖上下文信息。第二阶段采用一组3D各向异性解剖信息指导的U-Nets生成NF分割置信度掩模。在最终阶段，从置信度掩模中提取肿瘤候选区域并基于影像组学特征进行分类，以区分肿瘤区和非肿瘤区并减少误检。我们使用三个代表不同条件的测试集评估了该管道：领域内数据（测试集1）、不同成像协议和场强（测试集2）、以及低肿瘤负荷病例（测试集3）。实验结果表明，在高肿瘤负荷病例中结合解剖信息可以提高每个扫描的Dice相似性系数68%，每个肿瘤的Dice相似性系数提高21%，肿瘤检测的F1分数提高两倍。该方法已集成到3D Slicer平台中，用于实际临床使用，代码已公开。 

---
# HiFi-KPI: A Dataset for Hierarchical KPI Extraction from Earnings Filings 

**Title (ZH)**: HiFi-KPI： earnings 报告中层级关键绩效指标提取的数据集 

**Authors**: Rasmus Aavang, Giovanni Rizzi, Rasmus Bøggild, Alexandre Iolov, Mike Zhang, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2502.15411)  

**Abstract**: The U.S. Securities and Exchange Commission (SEC) requires that public companies file financial reports tagging numbers with the machine readable inline eXtensible Business Reporting Language (iXBRL) standard. However, the highly complex and highly granular taxonomy defined by iXBRL limits label transferability across domains. In this paper, we introduce the Hierarchical Financial Key Performance Indicator (HiFi-KPI) dataset, designed to facilitate numerical KPI extraction at specified levels of granularity from unstructured financial text. Our approach organizes a 218,126-label hierarchy using a taxonomy based grouping method, investigating which taxonomy layer provides the most meaningful structure. HiFi-KPI comprises ~1.8M paragraphs and ~5M entities, each linked to a label in the iXBRL-specific calculation and presentation taxonomies. We provide baselines using encoder-based approaches and structured extraction using Large Language Models (LLMs). To simplify LLM inference and evaluation, we additionally release HiFi-KPI Lite, a manually curated subset with four expert-mapped labels. We publicly release all artifacts 

**Abstract (ZH)**: 美国证券交易委员会（SEC）要求公共公司使用机器可读的即时扩展商业报告语言（iXBRL）标准标记财务报告中的数字。然而，iXBRL定义的复杂且精细的分类标准限制了标签在不同领域的转移性。本文介绍了Hierarchical Financial Key Performance Indicator（HiFi-KPI）数据集，旨在促进从非结构化的财务文本中在指定粒度级别提取关键绩效指标（KPI）。我们的方法使用基于分类法分组的方法构建了一个包含218,126个标签的层级结构，并探讨了哪种分类法层次提供了最具意义的结构。HiFi-KPI包含了约180万段落和约500万个实体，每个实体都链接到iXBRL特定的计算和呈现分类标准中的一个标签。我们提供了基于编码器的方法和使用大型语言模型（LLMs）进行结构化提取的基线。为简化LLM推理和评估，我们还发布了HiFi-KPI Lite，这是一个人工筛选的子集，包含四个专家映射的标签。所有成果均已公开发布。 

---
# Integrating Generative AI in Cybersecurity Education: Case Study Insights on Pedagogical Strategies, Critical Thinking, and Responsible AI Use 

**Title (ZH)**: 将生成式AI整合到网络安全教育中：基于教学策略、批判性思维和负责任AI使用案例研究的见解 

**Authors**: Mahmoud Elkhodr, Ergun Gide  

**Link**: [PDF](https://arxiv.org/pdf/2502.15357)  

**Abstract**: The rapid advancement of Generative Artificial Intelligence (GenAI) has introduced new opportunities for transforming higher education, particularly in fields that require analytical reasoning and regulatory compliance, such as cybersecurity management. This study presents a structured framework for integrating GenAI tools into cybersecurity education, demonstrating their role in fostering critical thinking, real-world problem-solving, and regulatory awareness. The implementation strategy followed a two-stage approach, embedding GenAI within tutorial exercises and assessment tasks. Tutorials enabled students to generate, critique, and refine AI-assisted cybersecurity policies, while assessments required them to apply AI-generated outputs to real-world scenarios, ensuring alignment with industry standards and regulatory requirements. Findings indicate that AI-assisted learning significantly enhanced students' ability to evaluate security policies, refine risk assessments, and bridge theoretical knowledge with practical application. Student reflections and instructor observations revealed improvements in analytical engagement, yet challenges emerged regarding AI over-reliance, variability in AI literacy, and the contextual limitations of AI-generated content. Through structured intervention and research-driven refinement, students were able to recognize AI strengths as a generative tool while acknowledging its need for human oversight. This study further highlights the broader implications of AI adoption in cybersecurity education, emphasizing the necessity of balancing automation with expert judgment to cultivate industry-ready professionals. Future research should explore the long-term impact of AI-driven learning on cybersecurity competency, as well as the potential for adaptive AI-assisted assessments to further personalize and enhance educational outcomes. 

**Abstract (ZH)**: 生成式人工智能的快速进步为高等教育带来了新的转型机会，特别是在需要分析推理和合规管理的领域，如网络安全管理。本研究提出了一种结构化框架，用于将生成式人工智能工具集成到网络安全教育中，展示了其在培养批判性思维、解决实际问题和增强合规意识方面的作用。实施策略遵循两阶段方法，将生成式人工智能嵌入教程练习和评估任务中。教程让学生能够生成、批判和改进AI辅助的网络安全政策，而评估则要求学生将AI生成的输出应用于实际场景，确保与行业标准和合规要求一致。研究结果表明，AI辅助学习显著增强了学生评估安全政策、改进风险评估和将理论知识与实践应用相结合的能力。学生反思和教师观察显示，在分析参与方面有所改进，但同时出现了对AI过度依赖、AI素养的差异性和AI生成内容的上下文限制等方面的挑战。通过结构化干预和基于研究的改进，学生能够认识到AI作为生成工具的优势，并认识到其需要人类监督。本研究还强调了在网络安全教育中采用AI的更广泛影响，强调需平衡自动化与专家判断以培养行业所需的专业人士。未来研究应探讨AI驱动学习对网络安全技能的长期影响，以及适应性AI辅助评估的潜力以进一步个性化和提升教育成果。 

---
# Lightweight yet Efficient: An External Attentive Graph Convolutional Network with Positional Prompts for Sequential Recommendation 

**Title (ZH)**: 轻量高效：一种基于位置提示的外部注意力图卷积网络在序列推荐中的应用 

**Authors**: Jinyu Zhang, Chao Li, Zhongying Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.15331)  

**Abstract**: Graph-based Sequential Recommender systems (GSRs) have gained significant research attention due to their ability to simultaneously handle user-item interactions and sequential relationships between items. Current GSRs often utilize composite or in-depth structures for graph encoding (e.g., the Graph Transformer). Nevertheless, they have high computational complexity, hindering the deployment on resource-constrained edge devices. Moreover, the relative position encoding in Graph Transformer has difficulty in considering the complicated positional dependencies within sequence. To this end, we propose an External Attentive Graph convolutional network with Positional prompts for Sequential recommendation, namely EA-GPS. Specifically, we first introduce an external attentive graph convolutional network that linearly measures the global associations among nodes via two external memory units. Then, we present a positional prompt-based decoder that explicitly treats the absolute item positions as external prompts. By introducing length-adaptive sequential masking and a soft attention network, such a decoder facilitates the model to capture the long-term positional dependencies and contextual relationships within sequences. Extensive experimental results on five real-world datasets demonstrate that the proposed EA-GPS outperforms the state-of-the-art methods. Remarkably, it achieves the superior performance while maintaining a smaller parameter size and lower training overhead. The implementation of this work is publicly available at this https URL. 

**Abstract (ZH)**: 基于图的序贯推荐系统（GSRs）由于其同时处理用户-项交互和项之间序贯关系的能力而引起了广泛关注。当前的GSRs经常利用复合或深入的图编码结构（例如，图 transformer）。然而，它们具有较高的计算复杂性，阻碍了在资源受限的边缘设备上的部署。此外，图变压器中的相对位置编码难以考虑序列内的复杂位置依赖性。为此，我们提出了一种基于外部注意力图卷积网络和位置提示的序贯推荐方法，即EA-GPS。具体地，我们首先引入了一种外部注意力图卷积网络，通过两个外部记忆单元线性衡量节点间的全局关联。然后，我们提出了一个基于位置提示的解码器，明确将绝对项位置作为外部提示处理。通过引入长度自适应序贯掩码和软注意力网络，该解码器促进了模型捕捉序列内的长期位置依赖性和上下文关系。在五个真实世界的数据集上的广泛实验结果表明，所提出的EA-GPS在性能上优于现有最先进的方法。值得注意的是，它在保持较小的参数量和更低的训练开销的前提下实现了更好的性能。该工作的实现已公开发布在该网址。 

---
# Beyond Fixed Variables: Expanding-variate Time Series Forecasting via Flat Scheme and Spatio-temporal Focal Learning 

**Title (ZH)**: 超越固定变量：基于扁平方案和时空焦点学习的可变变量时间序列预测 

**Authors**: Minbo Ma, Kai Tang, Huan Li, Fei Teng, Dalin Zhang, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.15296)  

**Abstract**: Multivariate Time Series Forecasting (MTSF) has long been a key research focus. Traditionally, these studies assume a fixed number of variables, but in real-world applications, Cyber-Physical Systems often expand as new sensors are deployed, increasing variables in MTSF. In light of this, we introduce a novel task, Expanding-variate Time Series Forecasting (EVTSF). This task presents unique challenges, specifically (1) handling inconsistent data shapes caused by adding new variables, and (2) addressing imbalanced spatio-temporal learning, where expanding variables have limited observed data due to the necessity for timely operation. To address these challenges, we propose STEV, a flexible spatio-temporal forecasting framework. STEV includes a new Flat Scheme to tackle the inconsistent data shape issue, which extends the graph-based spatio-temporal modeling architecture into 1D space by flattening the 2D samples along the variable dimension, making the model variable-scale-agnostic while still preserving dynamic spatial correlations through a holistic graph. We introduce a novel Spatio-temporal Focal Learning strategy that incorporates a negative filter to resolve potential conflicts between contrastive learning and graph representation, and a focal contrastive loss as its core to guide the framework to focus on optimizing the expanding variables. We benchmark EVTSF performance using three real-world datasets and compare it against three potential solutions employing SOTA MTSF models tailored for EVSTF. Experimental results show that STEV significantly outperforms its competitors, particularly on expanding variables. Notably, STEV, with only 5% of observations from the expanding period, is on par with SOTA MTSF models trained with complete observations. Further exploration of various expanding strategies underscores the generalizability of STEV in real-world applications. 

**Abstract (ZH)**: 扩展变量时间序列预报（EVTSF） 

---
# Time Warp: The Gap Between Developers' Ideal vs Actual Workweeks in an AI-Driven Era 

**Title (ZH)**: 时间扭曲：人工智能驱动时代开发者理想工作周与实际工作周之间的差距 

**Authors**: Sukrit Kumar, Drishti Goel, Thomas Zimmermann, Brian Houck, B. Ashok, Chetan Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.15287)  

**Abstract**: Software developers balance a variety of different tasks in a workweek, yet the allocation of time often differs from what they consider ideal. Identifying and addressing these deviations is crucial for organizations aiming to enhance the productivity and well-being of the developers. In this paper, we present the findings from a survey of 484 software developers at Microsoft, which aims to identify the key differences between how developers would like to allocate their time during an ideal workweek versus their actual workweek. Our analysis reveals significant deviations between a developer's ideal workweek and their actual workweek, with a clear correlation: as the gap between these two workweeks widens, we observe a decline in both productivity and satisfaction. By examining these deviations in specific activities, we assess their direct impact on the developers' satisfaction and productivity. Additionally, given the growing adoption of AI tools in software engineering, both in the industry and academia, we identify specific tasks and areas that could be strong candidates for automation. In this paper, we make three key contributions: 1) We quantify the impact of workweek deviations on developer productivity and satisfaction 2) We identify individual tasks that disproportionately affect satisfaction and productivity 3) We provide actual data-driven insights to guide future AI automation efforts in software engineering, aligning them with the developers' requirements and ideal workflows for maximizing their productivity and satisfaction. 

**Abstract (ZH)**: 软件开发人员在一个工作周中平衡各种不同的任务，但时间分配往往与他们认为的理想状态有所不同。识别并解决这些差异对于组织提高开发人员的生产率和福祉至关重要。在本文中，我们基于对484名微软软件开发人员的调查结果，旨在找出理想工作周与实际工作周之间时间分配的主要差异。我们的分析揭示了开发人员理想工作周与实际工作周之间存在显著差异，且这种差异与生产率和满意度的下降之间存在明显关联。通过对具体活动的差异进行分析，我们评估这些差异对开发人员满意度和生产率的直接影响。此外，鉴于人工智能工具在软件工程行业和学术界中的广泛应用，我们确定了一些可以实现自动化的关键任务和领域。在本文中，我们做出了三项关键贡献：1) 计量工作周差异对开发人员生产力和满意度的影响；2) 识别对满意度和生产力影响较大的个别任务；3) 提供实际的数据驱动见解，指导未来软件工程中的AI自动化努力，使其与开发人员的需求和理想工作流程相一致，以最大化他们的生产力和满意度。 

---
# Offload Rethinking by Cloud Assistance for Efficient Environmental Sound Recognition on LPWANs 

**Title (ZH)**: 基于云辅助的低功耗广域网高效环境声识别卸载重思 

**Authors**: Le Zhang, Quanling Zhao, Run Wang, Shirley Bian, Onat Gungor, Flavio Ponzina, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2502.15285)  

**Abstract**: Learning-based environmental sound recognition has emerged as a crucial method for ultra-low-power environmental monitoring in biological research and city-scale sensing systems. These systems usually operate under limited resources and are often powered by harvested energy in remote areas. Recent efforts in on-device sound recognition suffer from low accuracy due to resource constraints, whereas cloud offloading strategies are hindered by high communication costs. In this work, we introduce ORCA, a novel resource-efficient cloud-assisted environmental sound recognition system on batteryless devices operating over the Low-Power Wide-Area Networks (LPWANs), targeting wide-area audio sensing applications. We propose a cloud assistance strategy that remedies the low accuracy of on-device inference while minimizing the communication costs for cloud offloading. By leveraging a self-attention-based cloud sub-spectral feature selection method to facilitate efficient on-device inference, ORCA resolves three key challenges for resource-constrained cloud offloading over LPWANs: 1) high communication costs and low data rates, 2) dynamic wireless channel conditions, and 3) unreliable offloading. We implement ORCA on an energy-harvesting batteryless microcontroller and evaluate it in a real world urban sound testbed. Our results show that ORCA outperforms state-of-the-art methods by up to $80 \times$ in energy savings and $220 \times$ in latency reduction while maintaining comparable accuracy. 

**Abstract (ZH)**: 基于学习的环境声识别已成为生物研究和城市规模传感系统中超低功耗环境监测的关键方法。这些系统通常在资源有限的条件下运行，并且常常在偏远地区依靠收集的能量进行供电。近年来，设备上的声识别由于资源限制而准确率较低，而将计算任务卸载到云端则受到高通信成本的限制。在本文中，我们介绍了一种名为ORCA的新型资源高效云辅助环境声识别系统，该系统适用于低功耗广域网（LPWAN）上无电池设备的运行，旨在针对大面积音频感知应用。我们提出了一种云辅助策略，该策略可以在减轻设备上推断准确率低的问题的同时，尽量减少云卸载的通信成本。通过利用基于自注意力的云子谱特征选择方法来促进高效的设备上推断，ORCA解决了资源受限的LPWAN上云卸载的三个关键挑战：1) 高通信成本和低数据速率，2) 动态无线信道条件，以及3) 不可靠的卸载。我们在一个实际的城镇声音试验台上将ORCA部署在能量采集的无电池微控制器上进行评估。结果显示，与现有方法相比，ORCA在能耗上最多节省了80倍，在延迟上最多减少了220倍，同时保持了相近的准确率。 

---
# CopyJudge: Automated Copyright Infringement Identification and Mitigation in Text-to-Image Diffusion Models 

**Title (ZH)**: CopyJudge：文本到图像扩散模型中的自动化版权侵权识别与减轻 

**Authors**: Shunchang Liu, Zhuan Shi, Lingjuan Lyu, Yaochu Jin, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2502.15278)  

**Abstract**: Assessing whether AI-generated images are substantially similar to copyrighted works is a crucial step in resolving copyright disputes. In this paper, we propose CopyJudge, an automated copyright infringement identification framework that leverages large vision-language models (LVLMs) to simulate practical court processes for determining substantial similarity between copyrighted images and those generated by text-to-image diffusion models. Specifically, we employ an abstraction-filtration-comparison test framework with multi-LVLM debate to assess the likelihood of infringement and provide detailed judgment rationales. Based on the judgments, we further introduce a general LVLM-based mitigation strategy that automatically optimizes infringing prompts by avoiding sensitive expressions while preserving the non-infringing content. Besides, our approach can be enhanced by exploring non-infringing noise vectors within the diffusion latent space via reinforcement learning, even without modifying the original prompts. Experimental results show that our identification method achieves comparable state-of-the-art performance, while offering superior generalization and interpretability across various forms of infringement, and that our mitigation method could more effectively mitigate memorization and IP infringement without losing non-infringing expressions. 

**Abstract (ZH)**: 评估AI生成图像与受版权保护作品是否构成实质性相似是解决版权纠纷的关键步骤。本文提出了一种名为CopyJudge的自动化版权侵权识别框架，该框架利用大规模视觉-语言模型（LVLMs）模拟实际司法过程，以确定受版权保护的图像与由文本到图像扩散模型生成的图像之间的实质性相似性。具体而言，我们采用了抽离-滤波-比较测试框架，并结合多LVLM辩论来评估侵权可能性并提供详细的判决理由。基于判决结果，我们进一步引入了一种基于LVLM的一般减少策略，该策略能够自动优化侵权提示，避免敏感表达同时保留非侵权内容。此外，通过探索扩散潜在空间内的非侵权噪声向量，并结合强化学习，我们的方法可以在不修改原始提示的情况下增强性能。实验结果显示，我们的识别方法实现了与现有最佳方法相当的性能，且在多种侵权形式上具备更优的泛化能力和解释性，而我们的减少方法能够更有效地减轻记忆和知识产权侵权，同时保留非侵权表达。 

---
# Corrections Meet Explanations: A Unified Framework for Explainable Grammatical Error Correction 

**Title (ZH)**: 纠正与解释统一框架：可解释的语法错误矫正 

**Authors**: Jingheng Ye, Shang Qin, Yinghui Li, Hai-Tao Zheng, Shen Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15261)  

**Abstract**: Grammatical Error Correction (GEC) faces a critical challenge concerning explainability, notably when GEC systems are designed for language learners. Existing research predominantly focuses on explaining grammatical errors extracted in advance, thus neglecting the relationship between explanations and corrections. To address this gap, we introduce EXGEC, a unified explainable GEC framework that integrates explanation and correction tasks in a generative manner, advocating that these tasks mutually reinforce each other. Experiments have been conducted on EXPECT, a recent human-labeled dataset for explainable GEC, comprising around 20k samples. Moreover, we detect significant noise within EXPECT, potentially compromising model training and evaluation. Therefore, we introduce an alternative dataset named EXPECT-denoised, ensuring a more objective framework for training and evaluation. Results on various NLP models (BART, T5, and Llama3) show that EXGEC models surpass single-task baselines in both tasks, demonstrating the effectiveness of our approach. 

**Abstract (ZH)**: 基于生成的统一可解释性 grammatical error correction (GEC) 框架及其在 EXPECT 数据集上的应用研究 

---
# ComposeOn Academy: Transforming Melodic Ideas into Complete Compositions Integrating Music Learning 

**Title (ZH)**: 学院作曲：将旋律理念转化为完整作品的音乐创作方法研究 

**Authors**: Hongxi Pu, Futian Jiang, Zihao Chen, Xingyue Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.15255)  

**Abstract**: Music composition has long been recognized as a significant art form. However, existing digital audio workstations and music production software often present high entry barriers for users lacking formal musical training. To address this, we introduce ComposeOn, a music theory-based tool designed for users with limited musical knowledge. ComposeOn enables users to easily extend their melodic ideas into complete compositions and offers simple editing features. By integrating music theory, it explains music creation at beginner, intermediate, and advanced levels. Our user study (N=10) compared ComposeOn with the baseline method, Suno AI, demonstrating that ComposeOn provides a more accessible and enjoyable composing and learning experience for individuals with limited musical skills. ComposeOn bridges the gap between theory and practice, offering an innovative solution as both a composition aid and music education platform. The study also explores the differences between theory-based music creation and generative music, highlighting the former's advantages in personal expression and learning. 

**Abstract (ZH)**: 基于音乐理论的ComposeOn：面向音乐基础知识有限用户的音乐创作工具 

---
# AutoMR: A Universal Time Series Motion Recognition Pipeline 

**Title (ZH)**: 自动MR：一种通用的时间序列运动识别流水线 

**Authors**: Likun Zhang, Sicheng Yang, Zhuo Wang, Haining Liang, Junxiao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15228)  

**Abstract**: In this paper, we present an end-to-end automated motion recognition (AutoMR) pipeline designed for multimodal datasets. The proposed framework seamlessly integrates data preprocessing, model training, hyperparameter tuning, and evaluation, enabling robust performance across diverse scenarios. Our approach addresses two primary challenges: 1) variability in sensor data formats and parameters across datasets, which traditionally requires task-specific machine learning implementations, and 2) the complexity and time consumption of hyperparameter tuning for optimal model performance. Our library features an all-in-one solution incorporating QuartzNet as the core model, automated hyperparameter tuning, and comprehensive metrics tracking. Extensive experiments demonstrate its effectiveness on 10 diverse datasets, achieving state-of-the-art performance. This work lays a solid foundation for deploying motion-capture solutions across varied real-world applications. 

**Abstract (ZH)**: 本文提出了一种端到端的自动运动识别（AutoMR）流水线，适用于多模态数据集。所提出的框架无缝集成数据预处理、模型训练、超参数调整和评估，能够在多种场景下实现稳健性能。我们的方法解决两大主要挑战：1) 数据集间传感器数据格式和参数的差异性，传统上需要针对特定任务的机器学习实现；2) 为实现最佳模型性能而进行的超参数调整的复杂性和耗时。我们的库包括一个一站式解决方案，以QuartzNet为核心模型，自动超参数调整和全面的性能指标跟踪。广泛实验展示了其在10个不同数据集上的有效性，达到最佳性能。本文为在各种实际应用中部署运动捕捉解决方案奠定了坚实基础。 

---
# Scale-Free Graph-Language Models 

**Title (ZH)**: 无标度图-语言模型 

**Authors**: Jianglin Lu, Yixuan Liu, Yitian Zhang, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15189)  

**Abstract**: Graph-language models (GLMs) have demonstrated great potential in graph-based semi-supervised learning. A typical GLM consists of two key stages: graph generation and text embedding, which are usually implemented by inferring a latent graph and finetuning a language model (LM), respectively. However, the former often relies on artificial assumptions about the underlying edge distribution, while the latter requires extensive data annotations. To tackle these challenges, this paper introduces a novel GLM that integrates graph generation and text embedding within a unified framework. Specifically, for graph generation, we leverage an inherent characteristic of real edge distribution--the scale-free property--as a structural prior. We unexpectedly find that this natural property can be effectively approximated by a simple k-nearest neighbor (KNN) graph. For text embedding, we develop a graph-based pseudo-labeler that utilizes scale-free graphs to provide complementary supervision for improved LM finetuning. Extensive experiments on representative datasets validate our findings on the scale-free structural approximation of KNN graphs and demonstrate the effectiveness of integrating graph generation and text embedding with a real structural prior. Our code is available at this https URL. 

**Abstract (ZH)**: 基于图的语言模型（GLMs）在图基于半监督学习中展现了巨大的潜力。一种典型的GLM通常包含两个关键阶段：图生成和文本嵌入，分别通过推断潜在图和微调语言模型（LM）实现。然而，前者往往依赖于对底层边分布的人工假设，而后者则需要大量的数据注释。为应对这些挑战，本文提出了一种新颖的GLM，在统一框架中整合了图生成和文本嵌入。具体地，在图生成阶段，我们利用真实边分布的一个内在特征——无标度特性——作为结构先验。我们意外地发现，这种自然特性可以通过简单的KNN图有效近似。在文本嵌入阶段，我们开发了一种基于图的伪标签器，利用无标度图为改进LM微调提供互补监督。代表性和广泛的实验验证了KNN图的无标度结构逼近，并演示了在实际结构先验下整合图生成和文本嵌入的有效性。相关代码可在以下链接获取：this https URL。 

---
# Key Body Posture Characteristics of Short-distance Speed Skaters at the Start Based on Artificial Intelligence 

**Title (ZH)**: 基于人工智能的短距离速滑运动员起跑关键身体姿态特征 

**Authors**: Zhang Xueliana, Fang Yingjieb, Liu Hang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15185)  

**Abstract**: Objective To conduct biomechanical analysis on the starting technique of male short-distance speed skating athletes in China and determine the key factors affecting the effectiveness of the starting movement. Methods 13 high-level male short-distance speed skating athletes were selected as the test subjects, and kinematic data were collected using an artificial intelligence video capture and analysis system. The body posture features and their effects on the starting movement performance were analyzed in the three stages of starting preparation, starting, and sprinting. Results The post-stability angle, anterior knee angle of the front leg, posterior knee angle of the rear leg, and stride length showed moderate to high positive correlations with the starting speed during the starting preparation stage. The trunk angle showed a high negative correlation with the starting speed. The trunk angle (TO4, TD4, TO6, TD6), hip angle (TO1, TO4, TO6), and knee angle (TD1) showed moderate to high negative correlations with the effectiveness of the starting movement during the starting and sprinting stages. The knee angle (TD2), ice-contact angle (TD2, TD4, TD5, TD6), and propulsion angle (TO1, TO4, TO7) showed moderate positive correlations with the effectiveness of the starting movement. Conclusion Stride length, left knee angle, and post-stability angle are the key factors affecting the starting speed. The larger the post-stability angle and left knee angle and the longer the stride length, the faster the starting speed. During the starting and sprinting stages, the smaller the ice-contact angle and propulsion angle, the greater the trunk angle and hip angle changes, the more effective the starting movement. 

**Abstract (ZH)**: 目标：对中国男子短距离速度滑冰运动员起滑技术的生物力学分析，确定影响起滑动作有效性的关键因素。方法：选取13名高水平男子短距离速度滑冰运动员作为测试对象，采用人工智能视频捕捉与分析系统收集动作数据，在起滑准备、起滑和冲刺三个阶段分析身体姿态特征及其对起滑动作表现的影响。结果：起滑准备阶段，后稳定性角、前腿膝角、后腿膝角和步长与起滑速度呈中到高度正相关，躯干角与起滑速度呈高度负相关。起滑和冲刺阶段，躯干角（TO4、TD4、TO6、TD6）、髋角（TO1、TO4、TO6）和膝角（TD1）与起滑动作有效性呈中到高度负相关，膝角（TD2）、冰接触角（TD2、TD4、TD5、TD6）和推动力角（TO1、TO4、TO7）与起滑动作有效性呈中到高度正相关。结论：步长、左侧膝角和后稳定性角是影响起滑速度的关键因素。后稳定性角和左侧膝角越大、步长越长，起滑速度越快。在起滑和冲刺阶段，冰接触角和推动力角越小，躯干角和髋角的变化越大，起滑动作越有效。 

---
# CoT-ICL Lab: A Petri Dish for Studying Chain-of-Thought Learning from In-Context Demonstrations 

**Title (ZH)**: CoT-ICL 实验室：从情境演示中研究链式思考学习的实验平台 

**Authors**: Vignesh Kothapalli, Hamed Firooz, Maziar Sanjabi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15132)  

**Abstract**: We introduce CoT-ICL Lab, a framework and methodology to generate synthetic tokenized datasets and systematically study chain-of-thought (CoT) in-context learning (ICL) in language models. CoT-ICL Lab allows fine grained control over the complexity of in-context examples by decoupling (1) the causal structure involved in chain token generation from (2) the underlying token processing functions. We train decoder-only transformers (up to 700M parameters) on these datasets and show that CoT accelerates the accuracy transition to higher values across model sizes. In particular, we find that model depth is crucial for leveraging CoT with limited in-context examples, while more examples help shallow models match deeper model performance. Additionally, limiting the diversity of token processing functions throughout training improves causal structure learning via ICL. We also interpret these transitions by analyzing transformer embeddings and attention maps. Overall, CoT-ICL Lab serves as a simple yet powerful testbed for theoretical and empirical insights into ICL and CoT in language models. 

**Abstract (ZH)**: CoT-ICL Lab：生成合成标记数据集并系统研究语言模型中链式思考（CoT）上下文学习（ICL）的框架与方法 

---
# Assessing a Single Student's Concentration on Learning Platforms: A Machine Learning-Enhanced EEG-Based Framework 

**Title (ZH)**: 基于机器学习增强的EEG基础框架：评估单个学生在学习平台上的注意力 

**Authors**: Zewen Zhuo, Mohamad Najafi, Hazem Zein, Amine Nait-Ali  

**Link**: [PDF](https://arxiv.org/pdf/2502.15107)  

**Abstract**: This study introduces a specialized pipeline designed to classify the concentration state of an individual student during online learning sessions by training a custom-tailored machine learning model. Detailed protocols for acquiring and preprocessing EEG data are outlined, along with the extraction of fifty statistical features from five EEG signal bands: alpha, beta, theta, delta, and gamma. Following feature extraction, a thorough feature selection process was conducted to optimize the data inputs for a personalized analysis. The study also explores the benefits of hyperparameter fine-tuning to enhance the classification accuracy of the student's concentration state. EEG signals were captured from the student using a Muse headband (Gen 2), equipped with five electrodes (TP9, AF7, AF8, TP10, and a reference electrode NZ), during engagement with educational content on computer-based e-learning platforms. Employing a random forest model customized to the student's data, we achieved remarkable classification performance, with test accuracies of 97.6% in the computer-based learning setting and 98% in the virtual reality setting. These results underscore the effectiveness of our approach in delivering personalized insights into student concentration during online educational activities. 

**Abstract (ZH)**: 本研究 introduce 了一种专门的管道，用于通过训练定制的机器学习模型来分类个体学生在在线学习会话中的注意力状态。文中详细阐述了获取和预处理EEG数据的协议，以及从五个EEG信号带（alpha、beta、theta、delta和gamma）中提取五十个统计特征的方法。特征提取后，进行了彻底的特征选择过程，以优化个性化分析的数据输入。研究还探讨了超参数微调的好处，以提高学生注意力状态分类的准确性。EEG信号使用Muse头带（Gen 2）捕获，该头带配备了五个电极（TP9、AF7、AF8、TP10和参考电极NZ），学生在基于计算机的e学习平台上与教育内容互动时佩戴。使用定制于学生数据的随机森林模型，我们在基于计算机的学习环境中实现了令人瞩目的分类性能，测试准确率为97.6%，在虚拟现实环境中为98%。这些结果强调了我们方法在在线教育活动中提供个性化的学生注意力洞察方面的有效性。 

---
# Fundamental Survey on Neuromorphic Based Audio Classification 

**Title (ZH)**: 基于神经形态的音频分类基础调研 

**Authors**: Amlan Basu, Pranav Chaudhari, Gaetano Di Caterina  

**Link**: [PDF](https://arxiv.org/pdf/2502.15056)  

**Abstract**: Audio classification is paramount in a variety of applications including surveillance, healthcare monitoring, and environmental analysis. Traditional methods frequently depend on intricate signal processing algorithms and manually crafted features, which may fall short in fully capturing the complexities of audio patterns. Neuromorphic computing, inspired by the architecture and functioning of the human brain, presents a promising alternative for audio classification tasks. This survey provides an exhaustive examination of the current state-of-the-art in neuromorphic-based audio classification. It delves into the crucial components of neuromorphic systems, such as Spiking Neural Networks (SNNs), memristors, and neuromorphic hardware platforms, highlighting their advantages in audio classification. Furthermore, the survey explores various methodologies and strategies employed in neuromorphic audio classification, including event-based processing, spike-based learning, and bio-inspired feature extraction. It examines how these approaches address the limitations of traditional audio classification methods, particularly in terms of energy efficiency, real-time processing, and robustness to environmental noise. Additionally, the paper conducts a comparative analysis of different neuromorphic audio classification models and benchmarks, evaluating their performance metrics, computational efficiency, and scalability. By providing a comprehensive guide for researchers, engineers and practitioners, this survey aims to stimulate further innovation and advancements in the evolving field of neuromorphic audio classification. 

**Abstract (ZH)**: 基于神经形态计算的音频分类现状综述 

---
# Graph in the Vault: Protecting Edge GNN Inference with Trusted Execution Environment 

**Title (ZH)**: 库中之图：基于可信执行环境的边缘GNN推理保护 

**Authors**: Ruyi Ding, Tianhong Xu, Aidong Adam Ding, Yunsi Fei  

**Link**: [PDF](https://arxiv.org/pdf/2502.15012)  

**Abstract**: Wide deployment of machine learning models on edge devices has rendered the model intellectual property (IP) and data privacy vulnerable. We propose GNNVault, the first secure Graph Neural Network (GNN) deployment strategy based on Trusted Execution Environment (TEE). GNNVault follows the design of 'partition-before-training' and includes a private GNN rectifier to complement with a public backbone model. This way, both critical GNN model parameters and the private graph used during inference are protected within secure TEE compartments. Real-world implementations with Intel SGX demonstrate that GNNVault safeguards GNN inference against state-of-the-art link stealing attacks with negligible accuracy degradation (<2%). 

**Abstract (ZH)**: 基于受信执行环境的广义图神经网络部署策略GNNVault：保护模型知识产权和数据隐私 

---
# A Socratic RAG Approach to Connect Natural Language Queries on Research Topics with Knowledge Organization Systems 

**Title (ZH)**: 苏格拉底式RAG方法将研究主题的自然语言查询与知识组织系统连接起来 

**Authors**: Lew Lefton, Kexin Rong, Chinar Dankhara, Lila Ghemri, Firdous Kausar, A. Hannibal Hamdallahi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15005)  

**Abstract**: In this paper, we propose a Retrieval Augmented Generation (RAG) agent that maps natural language queries about research topics to precise, machine-interpretable semantic entities. Our approach combines RAG with Socratic dialogue to align a user's intuitive understanding of research topics with established Knowledge Organization Systems (KOSs). The proposed approach will effectively bridge "little semantics" (domain-specific KOS structures) with "big semantics" (broad bibliometric repositories), making complex academic taxonomies more accessible. Such agents have the potential for broad use. We illustrate with a sample application called CollabNext, which is a person-centric knowledge graph connecting people, organizations, and research topics. We further describe how the application design has an intentional focus on HBCUs and emerging researchers to raise visibility of people historically rendered invisible in the current science system. 

**Abstract (ZH)**: 基于检索增强生成的领域知识对话代理：连接“小语义”与“大语义” 

---
# CyberSentinel: An Emergent Threat Detection System for AI Security 

**Title (ZH)**: CyberSentinel: 一种AI安全 emergent threat 检测系统 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2502.14966)  

**Abstract**: The rapid advancement of artificial intelligence (AI) has significantly expanded the attack surface for AI-driven cybersecurity threats, necessitating adaptive defense strategies. This paper introduces CyberSentinel, a unified, single-agent system for emergent threat detection, designed to identify and mitigate novel security risks in real time. CyberSentinel integrates: (1) Brute-force attack detection through SSH log analysis, (2) Phishing threat assessment using domain blacklists and heuristic URL scoring, and (3) Emergent threat detection via machine learning-based anomaly detection. By continuously adapting to evolving adversarial tactics, CyberSentinel strengthens proactive cybersecurity defense, addressing critical vulnerabilities in AI security. 

**Abstract (ZH)**: 人工智能的迅速发展大幅扩大了由人工智能驱动的网络安全威胁的攻击面， necessitating 适应性防御策略。本文引入了CyberSentinel，这是一种统一的单智能体系统，用于 emergent 威胁检测，旨在实时识别和缓解新型安全风险。CyberSentinel 集成：(1) 通过对 SSH 日志的分析进行暴力攻击检测，(2) 通过域名黑名单和启发式 URL 评分评估网络钓鱼威胁，以及 (3) 通过基于机器学习的异常检测进行 emergent 威胁检测。通过持续适应不断演变的对手战术，CyberSentinel 加强了主动的 cybersecurity 防御，解决了人工智能安全中的关键漏洞。 

---
# KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding 

**Title (ZH)**: KITAB-Bench：阿拉伯OCR和文档理解的综合性多领域基准 

**Authors**: Ahmed Heakl, Abdullah Sohail, Mukul Ranjan, Rania Hossam, Ghazi Ahmed, Mohamed El-Geish, Omar Maher, Zhiqiang Shen, Fahad Khan, Salman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2502.14949)  

**Abstract**: With the growing adoption of Retrieval-Augmented Generation (RAG) in document processing, robust text recognition has become increasingly critical for knowledge extraction. While OCR (Optical Character Recognition) for English and other languages benefits from large datasets and well-established benchmarks, Arabic OCR faces unique challenges due to its cursive script, right-to-left text flow, and complex typographic and calligraphic features. We present KITAB-Bench, a comprehensive Arabic OCR benchmark that fills the gaps in current evaluation systems. Our benchmark comprises 8,809 samples across 9 major domains and 36 sub-domains, encompassing diverse document types including handwritten text, structured tables, and specialized coverage of 21 chart types for business intelligence. Our findings show that modern vision-language models (such as GPT-4, Gemini, and Qwen) outperform traditional OCR approaches (like EasyOCR, PaddleOCR, and Surya) by an average of 60% in Character Error Rate (CER). Furthermore, we highlight significant limitations of current Arabic OCR models, particularly in PDF-to-Markdown conversion, where the best model Gemini-2.0-Flash achieves only 65% accuracy. This underscores the challenges in accurately recognizing Arabic text, including issues with complex fonts, numeral recognition errors, word elongation, and table structure detection. This work establishes a rigorous evaluation framework that can drive improvements in Arabic document analysis methods and bridge the performance gap with English OCR technologies. 

**Abstract (ZH)**: 随着检索增强生成（RAG）在文档处理中的广泛应用，稳健的文本识别已成为知识提取中的关键需求。虽然英语和其他语言的光学字符识别（OCR）得益于大规模数据集和成熟的基准测试，但由于其连写字体、从右到左的文本流动以及复杂的体例和书法特征，阿拉伯OCR面临着独特的挑战。我们提出了KITAB-Bench，这是一种全面的阿拉伯OCR基准测试，填补了当前评估系统的空白。我们的基准测试包含8809个样本，涵盖9个主要领域和36个子领域，包括手写文本、结构化表格以及商业智能领域21种图表的专门覆盖。我们的研究结果表明，现代视觉语言模型（如GPT-4、Gemini和Qwen）在字符错误率（CER）方面平均比传统OCR方法（如EasyOCR、PaddleOCR和Surya）高出60%。此外，我们强调了当前阿拉伯OCR模型的重要局限性，特别是在PDF转Markdown转换中，最佳模型Gemini-2.0-Flash的准确率仅为65%，这突显了准确识别阿拉伯文本的挑战，包括复杂字体问题、数字识别错误、单词拉伸和表结构检测等问题。这项工作建立了一个严格的评估框架，可以推动阿拉伯文档分析方法的改进，并缩小与英语OCR技术的性能差距。 

---
# Reward-Guided Iterative Refinement in Diffusion Models at Test-Time with Applications to Protein and DNA Design 

**Title (ZH)**: Test-Time Reward-Guided Iterative Refinement in Diffusion Models with Applications to Protein and DNA Design 

**Authors**: Masatoshi Uehara, Xingyu Su, Yulai Zhao, Xiner Li, Aviv Regev, Shuiwang Ji, Sergey Levine, Tommaso Biancalani  

**Link**: [PDF](https://arxiv.org/pdf/2502.14944)  

**Abstract**: To fully leverage the capabilities of diffusion models, we are often interested in optimizing downstream reward functions during inference. While numerous algorithms for reward-guided generation have been recently proposed due to their significance, current approaches predominantly focus on single-shot generation, transitioning from fully noised to denoised states. We propose a novel framework for inference-time reward optimization with diffusion models inspired by evolutionary algorithms. Our approach employs an iterative refinement process consisting of two steps in each iteration: noising and reward-guided denoising. This sequential refinement allows for the gradual correction of errors introduced during reward optimization. Besides, we provide a theoretical guarantee for our framework. Finally, we demonstrate its superior empirical performance in protein and cell-type-specific regulatory DNA design. The code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 为了充分挖掘扩散模型的能力，我们在推理时通常对优化下游奖励函数感兴趣。尽管最近提出了许多奖励引导生成的算法，由于其重要性，当前的方法主要关注单次生成，即从完全噪音状态过渡到去噪音状态。我们提出了一种受进化算法启发的推理时奖励优化的新框架。该方法采用迭代细化过程，每一轮迭代包含两个步骤：加噪和奖励引导去噪。这种顺序细化过程允许逐步纠正奖励优化过程中引入的错误。此外，我们为该框架提供了理论保证。最后，我们在蛋白质和细胞类型特异性调控DNA设计中展示了其优越的实验性能。代码可在\href{this https URL}{this https URL}获取。 

---
# Fast and Accurate Blind Flexible Docking 

**Title (ZH)**: 快速且准确的目标灵活对接 

**Authors**: Zizhuo Zhang, Lijun Wu, Kaiyuan Gao, Jiangchao Yao, Tao Qin, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.14934)  

**Abstract**: Molecular docking that predicts the bound structures of small molecules (ligands) to their protein targets, plays a vital role in drug discovery. However, existing docking methods often face limitations: they either overlook crucial structural changes by assuming protein rigidity or suffer from low computational efficiency due to their reliance on generative models for structure sampling. To address these challenges, we propose FABFlex, a fast and accurate regression-based multi-task learning model designed for realistic blind flexible docking scenarios, where proteins exhibit flexibility and binding pocket sites are unknown (blind). Specifically, FABFlex's architecture comprises three specialized modules working in concert: (1) A pocket prediction module that identifies potential binding sites, addressing the challenges inherent in blind docking scenarios. (2) A ligand docking module that predicts the bound (holo) structures of ligands from their unbound (apo) states. (3) A pocket docking module that forecasts the holo structures of protein pockets from their apo conformations. Notably, FABFlex incorporates an iterative update mechanism that serves as a conduit between the ligand and pocket docking modules, enabling continuous structural refinements. This approach effectively integrates the three subtasks of blind flexible docking-pocket identification, ligand conformation prediction, and protein flexibility modeling-into a unified, coherent framework. Extensive experiments on public benchmark datasets demonstrate that FABFlex not only achieves superior effectiveness in predicting accurate binding modes but also exhibits a significant speed advantage (208 $\times$) compared to existing state-of-the-art methods. Our code is released at this https URL. 

**Abstract (ZH)**: 基于快速准确多任务学习的FABFlex分子对接模型：适用于盲柔性对接场景的小分子与蛋白质靶标结合结构预测 

---
# AI Thinking as a Meaning-Centered Framework: Reimagining Language Technologies Through Community Agency 

**Title (ZH)**: AI思维作为意义中心的框架：通过社区代理 reimagine 语言技术 

**Authors**: Jose F Quesada  

**Link**: [PDF](https://arxiv.org/pdf/2502.14923)  

**Abstract**: While language technologies have advanced significantly, current approaches fail to address the complex sociocultural dimensions of linguistic preservation. AI Thinking proposes a meaning-centered framework that would transform technological development from creating tools FOR communities to co-creating solutions WITH them. This approach recognizes that meaningful solutions emerge through the interplay of cultural understanding, community agency, and technological innovation. The proposal articulates a holistic methodology and a five-layer technological ecosystem where communities maintain control over their linguistic and cultural knowledge representation. This systematic integration of community needs, cultural preservation, and advanced capabilities could revolutionize how we approach linguistic diversity preservation in the digital age. 

**Abstract (ZH)**: 尽管语言技术取得了显著进步，当前的方法未能解决语言保存中的复杂社会文化维度。AI思考提出了一种以意义为中心的框架，旨在将技术发展从为社区创造工具转变为与社区共同创造解决方案。这种方法认识到，有意义的解决方案通过文化理解、社区自主性和技术创新的互动而产生。该提案阐述了一个整体方法论和五层技术生态系统，使社区能够控制其语言和文化知识的表述。这种系统地整合社区需求、文化保存和先进技术的能力有可能在数字时代彻底改变我们处理语言多样性保存的方式。 

---
# MKE-Coder: Multi-Axial Knowledge with Evidence Verification in ICD Coding for Chinese EMRs 

**Title (ZH)**: MKE-Coder: 结合证据验证的多轴线知识ICD编码方法用于中文EMRs 

**Authors**: Xinxin You, Xien Liu, Xue Yang, Ziyi Wang, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14916)  

**Abstract**: The task of automatically coding the International Classification of Diseases (ICD) in the medical field has been well-established and has received much attention. Automatic coding of the ICD in the medical field has been successful in English but faces challenges when dealing with Chinese electronic medical records (EMRs). The first issue lies in the difficulty of extracting disease code-related information from Chinese EMRs, primarily due to the concise writing style and specific internal structure of the EMRs. The second problem is that previous methods have failed to leverage the disease-based multi-axial knowledge and lack of association with the corresponding clinical evidence. This paper introduces a novel framework called MKE-Coder: Multi-axial Knowledge with Evidence verification in ICD coding for Chinese EMRs. Initially, we identify candidate codes for the diagnosis and categorize each of them into knowledge under four coding this http URL, we retrieve corresponding clinical evidence from the comprehensive content of EMRs and filter credible evidence through a scoring model. Finally, to ensure the validity of the candidate code, we propose an inference module based on the masked language modeling strategy. This module verifies that all the axis knowledge associated with the candidate code is supported by evidence and provides recommendations accordingly. To evaluate the performance of our framework, we conduct experiments using a large-scale Chinese EMR dataset collected from various hospitals. The experimental results demonstrate that MKE-Coder exhibits significant superiority in the task of automatic ICD coding based on Chinese EMRs. In the practical evaluation of our method within simulated real coding scenarios, it has been demonstrated that our approach significantly aids coders in enhancing both their coding accuracy and speed. 

**Abstract (ZH)**: 自动编码医学领域国际疾病分类（ICD）的任务已得到充分确立并受到广泛关注。医学领域自动编码ICD在英文中已取得成功，但在处理中文电子医疗记录（EMRs）时面临挑战。第一个问题在于从中文EMRs中提取与疾病编码相关的信息难度较大，主要由于EMRs的简洁写作风格和特定内部结构。第二个问题是，以往方法未能利用疾病为基础的多轴知识，缺乏与相应临床证据的关联。本文介绍了一种名为MKE-Coder的新框架：适用于中文EMRs的ICD编码中结合证据的多轴知识框架。首先，我们识别诊断候选代码并将其分为四种编码知识类别。然后，我们从EMRs的综合内容中检索相应临床证据，并通过评分模型筛选可信证据。最后，为确保候选代码的有效性，我们提出了一种基于掩码语言模型策略的推理模块。该模块验证与候选代码相关的所有轴知识都得到了证据支持，并据此提供推荐。为了评估该框架的性能，我们在来自多个医院的大规模中文EMR数据集上进行了实验。实验结果表明，MKE-Coder在基于中文EMRs的ICD自动编码任务中表现出显著优越性。在模拟实际编码场景中对我们方法的实用性评估表明，该方法显著帮助编码人员提高编码准确性和速度。 

---
# OpenSearch-SQL: Enhancing Text-to-SQL with Dynamic Few-shot and Consistency Alignment 

**Title (ZH)**: OpenSearch-SQL: 提升文本到SQL转换的动态少次示例和一致性对齐 

**Authors**: Xiangjin Xie, Guangwei Xu, Lingyan Zhao, Ruijie Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14913)  

**Abstract**: Although multi-agent collaborative Large Language Models (LLMs) have achieved significant breakthroughs in the Text-to-SQL task, their performance is still constrained by various factors. These factors include the incompleteness of the framework, failure to follow instructions, and model hallucination problems. To address these problems, we propose OpenSearch-SQL, which divides the Text-to-SQL task into four main modules: Preprocessing, Extraction, Generation, and Refinement, along with an Alignment module based on a consistency alignment mechanism. This architecture aligns the inputs and outputs of agents through the Alignment module, reducing failures in instruction following and hallucination. Additionally, we designed an intermediate language called SQL-Like and optimized the structured CoT based on SQL-Like. Meanwhile, we developed a dynamic few-shot strategy in the form of self-taught Query-CoT-SQL. These methods have significantly improved the performance of LLMs in the Text-to-SQL task.
In terms of model selection, we directly applied the base LLMs without any post-training, thereby simplifying the task chain and enhancing the framework's portability. Experimental results show that OpenSearch-SQL achieves an execution accuracy(EX) of 69.3% on the BIRD development set, 72.28% on the test set, and a reward-based validity efficiency score (R-VES) of 69.36%, with all three metrics ranking first at the time of submission. These results demonstrate the comprehensive advantages of the proposed method in both effectiveness and efficiency. 

**Abstract (ZH)**: 虽然多代理协作大型语言模型在Text-to-SQL任务上取得了显著突破，但其性能仍受多种因素限制。这些因素包括框架不完备、未能遵循指示和模型幻觉问题。为解决这些问题，我们提出了OpenSearch-SQL，将Text-to-SQL任务划分为预处理、提取、生成和精炼四大模块，并基于一致性对齐机制引入了一个对齐模块。该架构通过对齐模块对代理的输入和输出进行对齐，减少指令遵循和幻觉失败。此外，我们设计了一种称为SQL-Like的中间语言，并基于SQL-Like优化了结构化CoT。同时，我们开发了一种自教式的动态少量示例策略Query-CoT-SQL。这些方法显著提高了大规模语言模型在Text-to-SQL任务上的性能。在模型选择方面，我们直接应用了基础大规模语言模型，无需任何后续培训，从而简化了任务链并增强了框架的可移植性。实验结果显示，OpenSearch-SQL在BIRD开发集上的执行准确率（EX）为69.3%，测试集上的执行准确率（EX）为72.28%，基于奖励的有效性效率分数（R-VES）为69.36%，三项指标均提交时排名首位。这些结果展示了所提出方法在效果和效率方面的全面优势。 

---
# PTB-Image: A Scanned Paper ECG Dataset for Digitization and Image-based Diagnosis 

**Title (ZH)**: PTB-Image: 一种扫描纸张心电图数据集，用于数字化和图像诊断 

**Authors**: Cuong V. Nguyen, Hieu X. Nguyen, Dung D. Pham Minh, Cuong D. Do  

**Link**: [PDF](https://arxiv.org/pdf/2502.14909)  

**Abstract**: Electrocardiograms (ECGs) recorded on paper remain prevalent in clinical practice, yet their use presents challenges for automated analysis and digital storage. To address this issue, we introduce PTB-Image, a dataset comprising scanned paper ECGs with corresponding digital signals, enabling research on ECG digitization. We also provide VinDigitizer, a digitization baseline to convert paper-based ECGs into digital time-series signals. The method involves detecting signal rows, extracting waveforms from the background, and reconstructing numerical values from the digitized traces. We applied VinDigitizer to 549 scanned ECGs and evaluated its performance against the original PTB dataset (modified to match the printed signals). The results achieved a mean signal-to-noise ratio (SNR) of 0.01 dB, highlighting both the feasibility and challenges of ECG digitization, particularly in mitigating distortions from printing and scanning processes. By providing PTB-Image and baseline digitization methods, this work aims to facilitate advancements in ECG digitization, enhancing access to historical ECG data and supporting applications in telemedicine and automated cardiac diagnostics. 

**Abstract (ZH)**: 纸质心电图（ECGs）记录在临床实践中仍然广泛使用，但其使用为自动分析和数字化存储带来了挑战。为解决这一问题，我们介绍了PTB-Image数据集，该数据集包含扫描的纸质ECGs及其对应的数字信号，以促进ECG数字化研究。我们还提供了VinDigitizer，一种基线数字化方法，用于将纸质ECGs转换为数字时间序列信号。该方法包括检测信号行、从背景中提取波形以及从数字化轨迹中重构数值。我们将VinDigitizer应用于549份扫描的心电图，并将其性能与修改后的原PTB数据集进行了评估，结果获得了0.01 dB的平均信噪比（SNR），突显了ECG数字化的可行性和挑战，特别是在减少印刷和扫描过程中的失真方面。通过提供PTB-Image和基线数字化方法，本工作旨在促进ECG数字化的发展，提高历史ECG数据的可访问性，并支持远程医疗服务和自动化心脏诊断的应用。 

---
# Can AI mimic the human ability to define neologisms? 

**Title (ZH)**: AI能否模拟人类定义新词的能力？ 

**Authors**: Georgios P. Georgiou  

**Link**: [PDF](https://arxiv.org/pdf/2502.14900)  

**Abstract**: One ongoing debate in linguistics is whether Artificial Intelligence (AI) can effectively mimic human performance in language-related tasks. While much research has focused on various linguistic abilities of AI, little attention has been given to how it defines neologisms formed through different word formation processes. This study addresses this gap by examining the degree of agreement between human and AI-generated responses in defining three types of Greek neologisms: blends, compounds, and derivatives. The study employed an online experiment in which human participants selected the most appropriate definitions for neologisms, while ChatGPT received identical prompts. The results revealed fair agreement between human and AI responses for blends and derivatives but no agreement for compounds. However, when considering the majority response among humans, agreement with AI was high for blends and derivatives. These findings highlight the complexity of human language and the challenges AI still faces in capturing its nuances. In particular, they suggest a need for integrating more advanced semantic networks and contextual learning mechanisms into AI models to improve their interpretation of complex word formations, especially compounds. 

**Abstract (ZH)**: 语言学中一个持续的争论是人工智能是否能在语言相关任务中有效模仿人类表现。尽管已有大量研究关注人工智能的各种语言能力，但很少有研究关注它在通过不同词汇形成过程产生的新词定义中的定义方式。本研究通过比较人类和人工智能生成的回应来填补这一空白，考察了人类和人工智能在定义三种类型的希腊新词——融合词、复合词和衍生词——方面的程度一致。本研究采用了在线实验，要求人类参与者选择最合适的定义，同时ChatGPT接收相同的操作提示。结果显示，对于融合词和衍生词，人类和人工智能的回应存在一定程度的一致性，但没有一致性的复合词。然而，考虑到人类的多数意见，对于融合词和衍生词，与人工智能的一致性很高。这些发现突显了人类语言的复杂性以及人工智能在捕捉其微妙之处方面仍面临的挑战。特别是，它们表明需要将更先进的语义网络和上下文学习机制集成到人工智能模型中，以提高其对复杂词形的解释能力，尤其是在复合词方面。 

---
# UPCMR: A Universal Prompt-guided Model for Random Sampling Cardiac MRI Reconstruction 

**Title (ZH)**: UPCMR：一种通用提示引导模型用于随机采样心脏MRI重建 

**Authors**: Donghang Lyu, Chinmay Rao, Marius Staring, Matthias J.P. van Osch, Mariya Doneva, Hildo J. Lamb, Nicola Pezzotti  

**Link**: [PDF](https://arxiv.org/pdf/2502.14899)  

**Abstract**: Cardiac magnetic resonance imaging (CMR) is vital for diagnosing heart diseases, but long scan time remains a major drawback. To address this, accelerated imaging techniques have been introduced by undersampling k-space, which reduces the quality of the resulting images. Recent deep learning advancements aim to speed up scanning while preserving quality, but adapting to various sampling modes and undersampling factors remains challenging. Therefore, building a universal model is a promising direction. In this work, we introduce UPCMR, a universal unrolled model designed for CMR reconstruction. This model incorporates two kinds of learnable prompts, undersampling-specific prompt and spatial-specific prompt, and integrates them with a UNet structure in each block. Overall, by using the CMRxRecon2024 challenge dataset for training and validation, the UPCMR model highly enhances reconstructed image quality across all random sampling scenarios through an effective training strategy compared to some traditional methods, demonstrating strong adaptability potential for this task. 

**Abstract (ZH)**: 心脏磁共振成像（CMR）对于诊断心血管疾病至关重要，但较长的扫描时间仍然是一个主要缺点。为了克服这一问题，引入了通过欠采样k空间加速成像的技术，这会降低成像质量。最近的深度学习进展试图在保持成像质量的同时加快扫描速度，但适应各种采样模式和欠采样因素仍具有挑战性。因此，构建通用模型是一个有前景的方向。在本文中，我们提出了一种名为UPCMR的通用展开模型，设计用于CMR重建。该模型结合了两类可学习的提示，即特定于欠采样和特定于空间的提示，并将它们整合到每个块的UNet结构中。总体而言，通过使用CMRxRecon2024挑战数据集进行训练和验证，UPCMR模型通过有效的训练策略在所有随机采样场景中显著提升了重建图像的质量，展示了在该任务中强大的适应潜力。 

---
# Retrieval-augmented systems can be dangerous medical communicators 

**Title (ZH)**: 检索增强系统可能是危险的医疗通信工具 

**Authors**: Lionel Wong, Ayman Ali, Raymond Xiong, Shannon Zeijang Shen, Yoon Kim, Monica Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2502.14898)  

**Abstract**: Patients have long sought health information online, and increasingly, they are turning to generative AI to answer their health-related queries. Given the high stakes of the medical domain, techniques like retrieval-augmented generation and citation grounding have been widely promoted as methods to reduce hallucinations and improve the accuracy of AI-generated responses and have been widely adopted into search engines. This paper argues that even when these methods produce literally accurate content drawn from source documents sans hallucinations, they can still be highly misleading. Patients may derive significantly different interpretations from AI-generated outputs than they would from reading the original source material, let alone consulting a knowledgeable clinician. Through a large-scale query analysis on topics including disputed diagnoses and procedure safety, we support our argument with quantitative and qualitative evidence of the suboptimal answers resulting from current systems. In particular, we highlight how these models tend to decontextualize facts, omit critical relevant sources, and reinforce patient misconceptions or biases. We propose a series of recommendations -- such as the incorporation of communication pragmatics and enhanced comprehension of source documents -- that could help mitigate these issues and extend beyond the medical domain. 

**Abstract (ZH)**: 患者长期在网上寻求健康信息，并越来越多地利用生成式AI来回答他们的健康查询。由于医学领域的高风险性，检索增强生成和引用接地等技术已被广泛推广，用以减少幻觉并提高AI生成回复的准确性，并被广泛应用于搜索引擎中。本文指出，即使这些方法能生成不包含幻觉且字面上准确的内容，它们仍然可能导致高度误导。患者可能从AI生成的输出中得出与阅读原始资料或咨询专业医疗人员显著不同的解读。通过对包括争议性诊断和程序安全性在内的大量查询分析，我们提供了定量和定性证据，支持当前系统产生的次优答案。特别地，我们强调了这些模型倾向于脱离语境、省略关键相关资料以及强化患者误解或偏见的方式。我们提出了建议——如纳入交际语用学和增强对原始资料的理解——以帮助解决这些问题，并超越医学领域。 

---
# A Comprehensive Survey on Concept Erasure in Text-to-Image Diffusion Models 

**Title (ZH)**: 文本到图像扩散模型中的概念擦除综述 

**Authors**: Changhoon Kim, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14896)  

**Abstract**: Text-to-Image (T2I) models have made remarkable progress in generating high-quality, diverse visual content from natural language prompts. However, their ability to reproduce copyrighted styles, sensitive imagery, and harmful content raises significant ethical and legal concerns. Concept erasure offers a proactive alternative to external filtering by modifying T2I models to prevent the generation of undesired content. In this survey, we provide a structured overview of concept erasure, categorizing existing methods based on their optimization strategies and the architectural components they modify. We categorize concept erasure methods into fine-tuning for parameter updates, closed-form solutions for efficient edits, and inference-time interventions for content restriction without weight modification. Additionally, we explore adversarial attacks that bypass erasure techniques and discuss emerging defenses. To support further research, we consolidate key datasets, evaluation metrics, and benchmarks for assessing erasure effectiveness and model robustness. This survey serves as a comprehensive resource, offering insights into the evolving landscape of concept erasure, its challenges, and future directions. 

**Abstract (ZH)**: 文本到图像（T2I）模型在从自然语言提示生成高质量、多样化视觉内容方面取得了显著进展。然而，它们复制受版权保护的风格、敏感图像和有害内容的能力引发了重大的伦理和法律关切。概念擦除提供了一种主动的替代外部过滤的方案，通过修改T2I模型以防止生成不需要的内容。在本文综述中，我们提供了概念擦除的结构化概述，根据其优化策略和修改的架构组件对其进行分类。我们将概念擦除方法分为基于参数更新的微调方法、有效编辑的闭式解方法以及在权重不变情况下进行内容限制的推理时干预方法。此外，我们探讨了绕过擦除技术的对抗攻击，并讨论了新兴防御措施。为支持进一步研究，我们汇总了关键数据集、评估指标和基准，用于评估擦除效果和模型鲁棒性。本文综述旨在提供一个全面的资源，概述概念擦除的演变景观、挑战及其未来方向。 

---
# FOCUS on Contamination: A Geospatial Deep Learning Framework with a Noise-Aware Loss for Surface Water PFAS Prediction 

**Title (ZH)**: FOCUS于污染：一种带有噪声意识损失函数的地理空间深度学习框架用于表面水PFAS预测 

**Authors**: Jowaria Khan, Alexa Friedman, Sydney Evans, Runzi Wang, Kaley Beins, David Andrews, Elizabeth Bondi-Kelly  

**Link**: [PDF](https://arxiv.org/pdf/2502.14894)  

**Abstract**: Per and polyfluoroalkyl substances (PFAS), chemicals found in products like non-stick cookware, are unfortunately persistent environmental pollutants with severe health risks. Accurately mapping PFAS contamination is crucial for guiding targeted remediation efforts and protecting public and environmental health, yet detection across large regions remains challenging due to the cost of testing and the difficulty of simulating their spread. In this work, we introduce FOCUS, a geospatial deep learning framework with a label noise-aware loss function, to predict PFAS contamination in surface water over large regions. By integrating hydrological flow data, land cover information, and proximity to known PFAS sources, our approach leverages both spatial and environmental context to improve prediction accuracy. We evaluate the performance of our approach through extensive ablation studies and comparative analyses against baselines like sparse segmentation, as well as existing scientific methods, including Kriging and pollutant transport simulations. Results highlight our framework's potential for scalable PFAS monitoring. 

**Abstract (ZH)**: 含氟有机化合物（PFAS）的地理空间深度学习框架：一种考虑标签噪声的损失函数用于预测大面积区域表面水中的PFAS污染 

---
# Vision-Enhanced Time Series Forecasting via Latent Diffusion Models 

**Title (ZH)**: 基于潜扩散模型的视觉增强时间序列预测 

**Authors**: Weilin Ruan, Siru Zhong, Haomin Wen, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14887)  

**Abstract**: Diffusion models have recently emerged as powerful frameworks for generating high-quality images. While recent studies have explored their application to time series forecasting, these approaches face significant challenges in cross-modal modeling and transforming visual information effectively to capture temporal patterns. In this paper, we propose LDM4TS, a novel framework that leverages the powerful image reconstruction capabilities of latent diffusion models for vision-enhanced time series forecasting. Instead of introducing external visual data, we are the first to use complementary transformation techniques to convert time series into multi-view visual representations, allowing the model to exploit the rich feature extraction capabilities of the pre-trained vision encoder. Subsequently, these representations are reconstructed using a latent diffusion model with a cross-modal conditioning mechanism as well as a fusion module. Experimental results demonstrate that LDM4TS outperforms various specialized forecasting models for time series forecasting tasks. 

**Abstract (ZH)**: 基于潜扩散模型的视觉增强时间序列 Forecasting 框架 LDM4TS 

---
# Can LVLMs and Automatic Metrics Capture Underlying Preferences of Blind and Low-Vision Individuals for Navigational Aid? 

**Title (ZH)**: LVLMs和自动评价指标能否捕捉盲人和视力低下个体对导航辅助的潜在偏好？ 

**Authors**: Na Min An, Eunki Kim, Wan Ju Kang, Sangryul Kim, Hyunjung Shim, James Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2502.14883)  

**Abstract**: Vision is a primary means of how humans perceive the environment, but Blind and Low-Vision (BLV) people need assistance understanding their surroundings, especially in unfamiliar environments. The emergence of semantic-based systems as assistance tools for BLV users has motivated many researchers to explore responses from Large Vision-Language Models (LVLMs). However, it has yet been studied preferences of BLV users on diverse types/styles of responses from LVLMs, specifically for navigational aid. To fill this gap, we first construct Eye4B dataset, consisting of human-validated 1.1k curated outdoor/indoor scenes with 5-10 relevant requests per scene. Then, we conduct an in-depth user study with eight BLV users to evaluate their preferences on six LVLMs from five perspectives: Afraidness, Nonactionability, Sufficiency, and Conciseness. Finally, we introduce Eye4B benchmark for evaluating alignment between widely used model-based image-text metrics and our collected BLV preferences. Our work can be set as a guideline for developing BLV-aware LVLMs towards a Barrier-Free AI system. 

**Abstract (ZH)**: 基于语义的系统作为盲人和低视力用户辅助工具的出现促使许多研究人员探索大型视觉-语言模型（LVLMs）的响应，但尚未研究盲人和低视力用户对LVLMs不同类型的导航辅助响应的偏好。为填补这一空白，我们首先构建了Eye4B数据集，包含1100个经人工验证的室内外场景，每个场景有5-10个相关请求，然后对八名盲人和低视力用户进行了深入的用户研究，从害怕感、不可行性、充分性、简洁性等五个视角评估他们对六种LVLMs的偏好。最后，我们介绍了Eye4B基准，用于评估广泛使用的基于模型的图像-文本指标与收集到的盲人和低视力用户偏好之间的对齐情况。我们的工作可以作为开发盲人和低视力意识的LVLMs以构建无障碍人工智能系统的指南。 

---
# Is Mathematics Obsolete? 

**Title (ZH)**: 数学过时了吗？ 

**Authors**: Jeremy Avigad  

**Link**: [PDF](https://arxiv.org/pdf/2502.14874)  

**Abstract**: This is an essay about the value of mathematical and symbolic reasoning in the age of AI. 

**Abstract (ZH)**: 本论文探讨数学与符号推理在人工智能时代的价值。 

---
# Why do Experts Disagree on Existential Risk and P(doom)? A Survey of AI Experts 

**Title (ZH)**: 专家为何在存在风险和“末日”概率上存在分歧？对AI专家的调查 

**Authors**: Severin Field  

**Link**: [PDF](https://arxiv.org/pdf/2502.14870)  

**Abstract**: The development of artificial general intelligence (AGI) is likely to be one of humanity's most consequential technological advancements. Leading AI labs and scientists have called for the global prioritization of AI safety citing existential risks comparable to nuclear war. However, research on catastrophic risks and AI alignment is often met with skepticism, even by experts. Furthermore, online debate over the existential risk of AI has begun to turn tribal (e.g. name-calling such as "doomer" or "accelerationist"). Until now, no systematic study has explored the patterns of belief and the levels of familiarity with AI safety concepts among experts. I surveyed 111 AI experts on their familiarity with AI safety concepts, key objections to AI safety, and reactions to safety arguments. My findings reveal that AI experts cluster into two viewpoints -- an "AI as controllable tool" and an "AI as uncontrollable agent" perspective -- diverging in beliefs toward the importance of AI safety. While most experts (78%) agreed or strongly agreed that "technical AI researchers should be concerned about catastrophic risks", many were unfamiliar with specific AI safety concepts. For example, only 21% of surveyed experts had heard of "instrumental convergence," a fundamental concept in AI safety predicting that advanced AI systems will tend to pursue common sub-goals (such as self-preservation). The least concerned participants were the least familiar with concepts like this, suggesting that effective communication of AI safety should begin with establishing clear conceptual foundations in the field. 

**Abstract (ZH)**: 人工通用智能（AGI）的发展可能是人类最具影响力的科技进步之一。领先的人工智能实验室和科学家呼吁全球优先考虑人工智能安全，认为其存在风险堪比核战争。然而，关于灾难性风险和人工智能对齐的研究常常受到怀疑，即使是专家也不例外。此外，关于人工智能存在风险的在线辩论已经开始变得部落化（例如，带有“末日论者”或“加速主义者”这样的污名）。迄今为止，还没有系统研究探索专家们的信念模式和对人工智能安全概念的熟悉程度。我对111名人工智能专家进行了调查，了解他们对人工智能安全概念的熟悉程度、对人工智能安全的主要反对意见以及对安全论点的反应。研究发现，人工智能专家分为两种观点——“可控制工具的人工智能”和“不可控代理的人工智能”——在对人工智能安全重要性的信念上存在分歧。虽然大多数专家（78%）认为“技术人工智能研究人员应关注灾难性风险”，但许多专家对具体的人工智能安全概念不熟悉。例如，只有21%的受访专家听说过“手段趋同”这一基本概念，它预测高级人工智能系统倾向于追求共同的子目标（如自我保护）。最不关心的参与者对这类概念最不熟悉，这表明有效的人工智能安全沟通应从为该领域奠定清晰的概念基础开始。 

---
# Envisioning Stakeholder-Action Pairs to Mitigate Negative Impacts of AI: A Participatory Approach to Inform Policy Making 

**Title (ZH)**: 构想利益相关者行动对减轻人工智能负面影响的策略：参与式方法以指导政策制定 

**Authors**: Julia Barnett, Kimon Kieslich, Natali Helberger, Nicholas Diakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.14869)  

**Abstract**: The potential for negative impacts of AI has rapidly become more pervasive around the world, and this has intensified a need for responsible AI governance. While many regulatory bodies endorse risk-based approaches and a multitude of risk mitigation practices are proposed by companies and academic scholars, these approaches are commonly expert-centered and thus lack the inclusion of a significant group of stakeholders. Ensuring that AI policies align with democratic expectations requires methods that prioritize the voices and needs of those impacted. In this work we develop a participative and forward-looking approach to inform policy-makers and academics that grounds the needs of lay stakeholders at the forefront and enriches the development of risk mitigation strategies. Our approach (1) maps potential mitigation and prevention strategies of negative AI impacts that assign responsibility to various stakeholders, (2) explores the importance and prioritization thereof in the eyes of laypeople, and (3) presents these insights in policy fact sheets, i.e., a digestible format for informing policy processes. We emphasize that this approach is not targeted towards replacing policy-makers; rather our aim is to present an informative method that enriches mitigation strategies and enables a more participatory approach to policy development. 

**Abstract (ZH)**: AI负影响的潜在负面影响在全球范围内迅速蔓延，这加剧了对负责任的AI治理的需要。尽管许多监管机构支持风险为基础的方法，并且公司和学术界提出了多种风险缓解实践，但这些方法通常以专家为中心，从而缺乏关键利益相关者的参与。确保AI政策与民主期望相一致需要优先考虑受影响人员的声音和需求的方法。在本项研究中，我们开发了一种参与性和前瞻性的方法来为政策制定者和学术界提供信息，该方法将普通利益相关者的需求置于首位，并丰富了风险缓解策略的发展。我们的方法包括：（1）绘制负向AI影响的潜在缓解和预防策略，并分配责任给不同利益相关者；（2）探讨普通民众对此的重要性及优先级；（3）以政策简报的形式呈现这些见解，即一种便于信息传递的格式。我们强调，此方法并非旨在替代政策制定者；而是为了展示一种信息性方法，旨在丰富缓解策略并促进更具参与性的政策制定过程。 

---
# Unlocking the Black Box: Analysing the EU Artificial Intelligence Act's Framework for Explainability in AI 

**Title (ZH)**: 解锁黑箱：分析欧盟人工智能法案中的可解释性框架 

**Authors**: Georgios Pavlidis  

**Link**: [PDF](https://arxiv.org/pdf/2502.14868)  

**Abstract**: The lack of explainability of Artificial Intelligence (AI) is one of the first obstacles that the industry and regulators must overcome to mitigate the risks associated with the technology. The need for eXplainable AI (XAI) is evident in fields where accountability, ethics and fairness are critical, such as healthcare, credit scoring, policing and the criminal justice system. At the EU level, the notion of explainability is one of the fundamental principles that underpin the AI Act, though the exact XAI techniques and requirements are still to be determined and tested in practice. This paper explores various approaches and techniques that promise to advance XAI, as well as the challenges of implementing the principle of explainability in AI governance and policies. Finally, the paper examines the integration of XAI into EU law, emphasising the issues of standard setting, oversight, and enforcement. 

**Abstract (ZH)**: 人工智能的解释性不足是行业和监管机构必须克服的第一个障碍，以减轻与此技术相关的风险。在问责制、伦理和公平性至关重要的领域，如医疗保健、信用评分、警务和刑事司法系统中，可解释人工智能（XAI）的需求尤为明显。在欧盟层面，可解释性是人工智能法案的基本原则之一，尽管具体的XAI技术要求尚待确定和实践测试。本文探讨了各种有望推动XAI进展的方法和技术，以及在人工智能治理和政策中实施解释性原则所面临的挑战，并分析了XAI在欧盟法律中的整合问题，强调了标准制定、监督和执行的问题。 

---
