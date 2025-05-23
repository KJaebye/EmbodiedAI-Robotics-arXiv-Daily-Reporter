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
# Empowering LLMs with Logical Reasoning: A Comprehensive Survey 

**Title (ZH)**: 增强大语言模型的逻辑推理能力：一项全面综述 

**Authors**: Fengxiang Cheng, Haoxuan Li, Fenrong Liu, Robert van Rooij, Kun Zhang, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.15652)  

**Abstract**: Large language models (LLMs) have achieved remarkable successes on various natural language tasks. However, recent studies have found that there are still significant challenges to the logical reasoning abilities of LLMs. This paper summarizes and categorizes the main challenges into two aspects: (1) Logical question answering, LLMs often fail to generate the correct answer within complex logical problem which requires sophisticated deductive, inductive or abductive reasoning given a collection of premises and constrains. (2) Logical consistency, LLMs are prone to producing responses contradicting themselves across different questions. For example, a state-of-the-art Macaw question-answering LLM answers Yes to both questions Is a magpie a bird? and Does a bird have wings? but answers No to Does a magpie have wings?. To facilitate this research direction, we comprehensively investigate the most cutting-edge methods and propose detailed taxonomies of these methods. Specifically, to accurately answer complex logic questions, previous methods can be categorized based on reliance on external solvers, prompts, pretraining, and fine-tuning. To avoid logical contradictions, we discuss concepts and solutions of various logical consistencies, including implication, negation, transitivity, factuality consistency, and their composites. In addition, we review commonly used benchmark datasets and evaluation metrics, and discuss promising research directions, such as extensions to modal logic to account for uncertainty, and efficient algorithms satisfying multiple logical consistencies simultaneously. 

**Abstract (ZH)**: 大型语言模型在各类自然语言任务中取得了显著成果，然而最近的研究发现，这些模型在逻辑推理能力方面仍面临重大挑战。本文从两个方面总结和分类了主要挑战：（1）逻辑问答，大型语言模型在面对需要复杂演绎、归纳或 abduction 推理的逻辑问题集合时，往往不能生成正确的答案。（2）逻辑一致性，大型语言模型容易在同一问题集合中产生自相矛盾的回复。为了促进该研究方向的发展，本文全面调查了最新的方法并提出了这些方法的详细分类。具体而言，为了准确回答复杂的逻辑问题，先前的方法可以根据对外部求解器、提示、预训练和微调的依赖关系进行分类。为了避免逻辑矛盾，本文讨论了各种逻辑一致性的概念及解决方案，包括蕴含、否定、传递性、事实一致性及其复合。此外，本文回顾了常用的标准数据集和评估指标，并讨论了有潜力的研究方向，如扩展到模态逻辑以处理不确定性，并提出同时满足多种逻辑一致性的高效算法。 

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
# Chitrarth: Bridging Vision and Language for a Billion People 

**Title (ZH)**: Chitrarth：连接视觉与语言，惠及十亿人群 

**Authors**: Shaharukh Khan, Ayush Tarun, Abhinav Ravi, Ali Faraz, Akshat Patidar, Praveen Kumar Pokala, Anagha Bhangare, Raja Kolla, Chandra Khatri, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2502.15392)  

**Abstract**: Recent multimodal foundation models are primarily trained on English or high resource European language data, which hinders their applicability to other medium and low-resource languages. To address this limitation, we introduce Chitrarth (Chitra: Image; Artha: Meaning), an inclusive Vision-Language Model (VLM), specifically targeting the rich linguistic diversity and visual reasoning across 10 prominent Indian languages. Our model effectively integrates a state-of-the-art (SOTA) multilingual Large Language Model (LLM) with a vision module, primarily trained on multilingual image-text data. Furthermore, we also introduce BharatBench, a comprehensive framework for evaluating VLMs across various Indian languages, ultimately contributing to more diverse and effective AI systems. Our model achieves SOTA results for benchmarks across low resource languages while retaining its efficiency in English. Through our research, we aim to set new benchmarks in multilingual-multimodal capabilities, offering substantial improvements over existing models and establishing a foundation to facilitate future advancements in this arena. 

**Abstract (ZH)**: 最近的多模态基础模型主要在英语或高资源欧洲语言数据上进行训练，这限制了它们在其他中低资源语言中的适用性。为解决这一局限，我们引入了Chitrarth（Chitra：图像；Artha：意义），一个针对10种 prominent 印地语的包容性视觉-语言模型，专门针对这些语言丰富的语言多样性和视觉推理能力。我们的模型有效结合了一种最先进的多语言大型语言模型（LLM）和一个视觉模块，该模块主要在多语言图像-文本数据上进行训练。此外，我们还提出了BharatBench，一个综合框架，用于评估各种印地语的视觉-语言模型，最终促进了更多样化和有效的AI系统的发展。我们的模型在低资源语言基准测试中取得了最先进的成果，同时保持了其在英语中的效率。通过我们的研究，我们旨在建立多语言-多模态能力的新基准，提供现有模型的重大改进，并奠定未来在此领域发展的基础。 

---
# ARS: Automatic Routing Solver with Large Language Models 

**Title (ZH)**: ARS: 使用大规模语言模型的自动路由求解器 

**Authors**: Kai Li, Fei Liu, Zhenkun Wang, Xialiang Tong, Xiongwei Han, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15359)  

**Abstract**: Real-world Vehicle Routing Problems (VRPs) are characterized by a variety of practical constraints, making manual solver design both knowledge-intensive and time-consuming. Although there is increasing interest in automating the design of routing algorithms, existing research has explored only a limited array of VRP variants and fails to adequately address the complex and prevalent constraints encountered in real-world situations. To fill this gap, this paper introduces RoutBench, a benchmark of 1,000 VRP variants derived from 24 attributes, for evaluating the effectiveness of automatic routing solvers in addressing complex constraints. Along with RoutBench, we present the Automatic Routing Solver (ARS), which employs Large Language Model (LLM) agents to enhance a backbone algorithm framework by automatically generating constraint-aware heuristic code, based on problem descriptions and several representative constraints selected from a database. Our experiments show that ARS outperforms state-of-the-art LLM-based methods and commonly used solvers, automatically solving 91.67% of common VRPs and achieving at least a 30% improvement across all benchmarks. 

**Abstract (ZH)**: 实际应用场景中的车辆路线问题（VRPs）由多种实际约束构成，使得手动设计求解器既知识密集型又耗时。尽管有越来越多的兴趣在于自动化设计路由算法，现有的研究仅探索了VRP的有限变体，并未能充分应对实际场景中复杂且普遍存在的约束。为了弥补这一不足，本文引入了RoutBench，这是一个由24个属性衍生出的1000个VRP变体基准，用于评估自动路由求解器在应对复杂约束时的有效性。同时，本文还提出了自动路由求解器（ARS），该求解器利用大型语言模型（LLM）代理自动为骨架算法框架生成感知约束的启发式代码，基于问题描述和数据库中选定的几个代表性约束。实验结果表明，ARS优于最先进的基于LLM的方法和常用求解器，在91.67%的常见VRP实例中自动求解，并在所有基准测试中至少达到30%的改进。 

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
# BOSS: Benchmark for Observation Space Shift in Long-Horizon Task 

**Title (ZH)**: BOSS：长 horizon 任务中观察空间转变的基准 

**Authors**: Yue Yang, Linfeng Zhao, Mingyu Ding, Gedas Bertasius, Daniel Szafir  

**Link**: [PDF](https://arxiv.org/pdf/2502.15679)  

**Abstract**: Robotics has long sought to develop visual-servoing robots capable of completing previously unseen long-horizon tasks. Hierarchical approaches offer a pathway for achieving this goal by executing skill combinations arranged by a task planner, with each visuomotor skill pre-trained using a specific imitation learning (IL) algorithm. However, even in simple long-horizon tasks like skill chaining, hierarchical approaches often struggle due to a problem we identify as Observation Space Shift (OSS), where the sequential execution of preceding skills causes shifts in the observation space, disrupting the performance of subsequent individually trained skill policies. To validate OSS and evaluate its impact on long-horizon tasks, we introduce BOSS (a Benchmark for Observation Space Shift). BOSS comprises three distinct challenges: "Single Predicate Shift", "Accumulated Predicate Shift", and "Skill Chaining", each designed to assess a different aspect of OSS's negative effect. We evaluated several recent popular IL algorithms on BOSS, including three Behavioral Cloning methods and the Visual Language Action model OpenVLA. Even on the simplest challenge, we observed average performance drops of 67%, 35%, 34%, and 54%, respectively, when comparing skill performance with and without OSS. Additionally, we investigate a potential solution to OSS that scales up the training data for each skill with a larger and more visually diverse set of demonstrations, with our results showing it is not sufficient to resolve OSS. The project page is: this https URL 

**Abstract (ZH)**: 机器人学长期致力于开发能够完成未见过的长周期任务的视觉伺服机器人。分层方法通过由任务规划器执行技能组合来实现这一目标，每个视觉运动技能都使用特定的模仿学习（IL）算法进行预先训练。然而，即使是简单的长周期任务如技能串联，分层方法也常常因我们识别出的观察空间移位（OSS）问题而受阻，即前序技能的顺序执行会导致观察空间的改变，从而干扰后续单独训练的技能策略的表现。为了验证OSS并评估其对长周期任务的影响，我们引入了BOSS（观察空间移位基准）。BOSS包含三个不同的挑战：“单一谓词移位”、“积累谓词移位”和“技能串联”，旨在评估OSS负面影响的不同方面。我们评估了BOSS上几种最近流行的IL算法，包括三种行为克隆方法和Visual Language Action模型OpenVLA。即使在最简单的挑战中，我们观察到技能表现平均下降了67%、35%、34%和54%（有OSS与无OSS进行比较）。此外，我们还探讨了解决OSS的一种潜在方案，即通过使用更大、更具视觉多样性的示范来扩大每个技能的训练数据，但结果显示这不足以解决OSS问题。项目网页：this https URL。 

---
# FLEKE: Federated Locate-then-Edit Knowledge Editing 

**Title (ZH)**: FLEKE: 联邦定位编辑知识编辑 

**Authors**: Zongkai Zhao, Guozeng Xu, Xiuhua Li, Kaiwen Wei, Jiang Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2502.15677)  

**Abstract**: Locate-then-Edit Knowledge Editing (LEKE) is a key technique for updating large language models (LLMs) without full retraining. However, existing methods assume a single-user setting and become inefficient in real-world multi-client scenarios, where decentralized organizations (e.g., hospitals, financial institutions) independently update overlapping knowledge, leading to redundant mediator knowledge vector (MKV) computations and privacy concerns. To address these challenges, we introduce Federated Locate-then-Edit Knowledge Editing (FLEKE), a novel task that enables multiple clients to collaboratively perform LEKE while preserving privacy and reducing computational overhead. To achieve this, we propose FedEdit, a two-stage framework that optimizes MKV selection and reuse. In the first stage, clients locally apply LEKE and upload the computed MKVs. In the second stage, rather than relying solely on server-based MKV sharing, FLEKE allows clients retrieve relevant MKVs based on cosine similarity, enabling knowledge re-edit and minimizing redundant computations. Experimental results on two benchmark datasets demonstrate that FedEdit retains over 96% of the performance of non-federated LEKE while significantly outperforming a FedAvg-based baseline by approximately twofold. Besides, we find that MEMIT performs more consistently than PMET in the FLEKE task with our FedEdit framework. Our code is available at this https URL. 

**Abstract (ZH)**: 联邦定位编辑知识编辑（FLEKE） 

---
# VaViM and VaVAM: Autonomous Driving through Video Generative Modeling 

**Title (ZH)**: VaViM和VaVAM：通过视频生成模型实现自主驾驶 

**Authors**: Florent Bartoccioni, Elias Ramzi, Victor Besnier, Shashanka Venkataramanan, Tuan-Hung Vu, Yihong Xu, Loick Chambon, Spyros Gidaris, Serkan Odabas, David Hurych, Renaud Marlet, Alexandre Boulch, Mickael Chen, Éloi Zablocki, Andrei Bursuc, Eduardo Valle, Matthieu Cord  

**Link**: [PDF](https://arxiv.org/pdf/2502.15672)  

**Abstract**: We explore the potential of large-scale generative video models for autonomous driving, introducing an open-source auto-regressive video model (VaViM) and its companion video-action model (VaVAM) to investigate how video pre-training transfers to real-world driving. VaViM is a simple auto-regressive video model that predicts frames using spatio-temporal token sequences. We show that it captures the semantics and dynamics of driving scenes. VaVAM, the video-action model, leverages the learned representations of VaViM to generate driving trajectories through imitation learning. Together, the models form a complete perception-to-action pipeline. We evaluate our models in open- and closed-loop driving scenarios, revealing that video-based pre-training holds promise for autonomous driving. Key insights include the semantic richness of the learned representations, the benefits of scaling for video synthesis, and the complex relationship between model size, data, and safety metrics in closed-loop evaluations. We release code and model weights at this https URL 

**Abstract (ZH)**: 我们探索大规模生成视频模型在自动驾驶领域的潜力，介绍了一个开源的自回归视频模型（VaViM）及其同伴视频-动作模型（VaVAM），以研究视频预训练如何应用于真实世界的驾驶。VaViM 是一个简单的自回归视频模型，利用时空令牌序列预测帧，展示了其捕捉驾驶场景的语义和动力学的能力。VaVAM 利用 VaViM 中学习到的表示进行模仿学习以生成驾驶轨迹。这两款模型共同形成了一条完整的感知到动作流水线。我们在开放环和闭环驾驶场景中评估了我们的模型，结果表明视频为基础的预训练对自动驾驶具有前景。关键见解包括学习表示的语义 richness、视频合成中的扩大量的影响，以及闭环评估中模型规模、数据与安全指标之间的复杂关系。我们在 https://this.url/ 发布了代码和模型权重。 

---
# Almost AI, Almost Human: The Challenge of Detecting AI-Polished Writing 

**Title (ZH)**: 几乎人工智能，几乎human：检测AI润色写作的挑战 

**Authors**: Shoumik Saha, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15666)  

**Abstract**: The growing use of large language models (LLMs) for text generation has led to widespread concerns about AI-generated content detection. However, an overlooked challenge is AI-polished text, where human-written content undergoes subtle refinements using AI tools. This raises a critical question: should minimally polished text be classified as AI-generated? Misclassification can lead to false plagiarism accusations and misleading claims about AI prevalence in online content. In this study, we systematically evaluate eleven state-of-the-art AI-text detectors using our AI-Polished-Text Evaluation (APT-Eval) dataset, which contains $11.7K$ samples refined at varying AI-involvement levels. Our findings reveal that detectors frequently misclassify even minimally polished text as AI-generated, struggle to differentiate between degrees of AI involvement, and exhibit biases against older and smaller models. These limitations highlight the urgent need for more nuanced detection methodologies. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本生成中的广泛应用引发了对AI生成内容检测的广泛关切。然而，一个被忽视的挑战是AI润饰文本，即人类撰写的文本通过AI工具进行微妙改进。这引发了关键问题：轻微润饰的文本是否应被视为AI生成？误分类可能导致虚假的剽窃指控和关于在线内容中AI普及程度的误导性说法。在本研究中，我们使用包含按不同AI参与程度 refinement 的11,700个样本的AI润饰文本评估（APT-Eval）数据集，系统评估了十一种最先进的AI文本检测器。我们的发现表明，检测器经常错误地将轻微润饰的文本分类为AI生成，难以区分不同程度的AI参与，并偏向于老款和小型模型。这些局限性突显了更细致的检测方法的迫切需求。 

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
# Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models 

**Title (ZH)**: 转向新的嵌入空间：分析多语言模型中模型干预诱导的跨语言对齐 

**Authors**: Anirudh Sundar, Sinead Williamson, Katherine Metcalf, Barry-John Theobald, Skyler Seto, Masha Fedzechkina  

**Link**: [PDF](https://arxiv.org/pdf/2502.15639)  

**Abstract**: Aligned representations across languages is a desired property in multilingual large language models (mLLMs), as alignment can improve performance in cross-lingual tasks. Typically alignment requires fine-tuning a model, which is computationally expensive, and sizable language data, which often may not be available. A data-efficient alternative to fine-tuning is model interventions -- a method for manipulating model activations to steer generation into the desired direction. We analyze the effect of a popular intervention (finding experts) on the alignment of cross-lingual representations in mLLMs. We identify the neurons to manipulate for a given language and introspect the embedding space of mLLMs pre- and post-manipulation. We show that modifying the mLLM's activations changes its embedding space such that cross-lingual alignment is enhanced. Further, we show that the changes to the embedding space translate into improved downstream performance on retrieval tasks, with up to 2x improvements in top-1 accuracy on cross-lingual retrieval. 

**Abstract (ZH)**: 多语言大型语言模型中跨语言表示的一致性是一种期望属性，数据高效的方法：通过干预调整跨语言表示的一致性 

---
# Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification 

**Title (ZH)**: Mantis: 轻量级校准基础模型用于用户友好的时间序列分类 

**Authors**: Vasilii Feofanov, Songkang Wen, Marius Alonso, Romain Ilbert, Hongbo Guo, Malik Tiomoko, Lujia Pan, Jianfeng Zhang, Ievgen Redko  

**Link**: [PDF](https://arxiv.org/pdf/2502.15637)  

**Abstract**: In recent years, there has been increasing interest in developing foundation models for time series data that can generalize across diverse downstream tasks. While numerous forecasting-oriented foundation models have been introduced, there is a notable scarcity of models tailored for time series classification. To address this gap, we present Mantis, a new open-source foundation model for time series classification based on the Vision Transformer (ViT) architecture that has been pre-trained using a contrastive learning approach. Our experimental results show that Mantis outperforms existing foundation models both when the backbone is frozen and when fine-tuned, while achieving the lowest calibration error. In addition, we propose several adapters to handle the multivariate setting, reducing memory requirements and modeling channel interdependence. 

**Abstract (ZH)**: 近年来，人们日益关注开发能够跨多种下游任务泛化的时序数据基础模型。尽管已经提出了众多面向预测的基础模型，但专门针对时序分类的任务模型却相对匮乏。为填补这一空白，我们提出了一种基于Vision Transformer (ViT) 架构的新开源时序分类基础模型Mantis，该模型通过对比学习方式进行预训练。实验结果表明，无论是在冻结主干网络的情况下还是在微调情况下，Mantis 均表现出色，并实现了最低的校准误差。此外，我们还提出了一些适配器以处理多变量设置，降低内存需求并建模通道间的依赖关系。 

---
# The Relationship Between Reasoning and Performance in Large Language Models -- o3 (mini) Thinks Harder, Not Longer 

**Title (ZH)**: 大型语言模型中推理与性能之间的关系——o3（mini）思考更深，而非更久 

**Authors**: Marthe Ballon, Andres Algaba, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2502.15631)  

**Abstract**: Large language models have demonstrated remarkable progress in mathematical reasoning, leveraging chain-of-thought and test-time compute scaling. However, many open questions remain regarding the interplay between reasoning token usage and accuracy gains. In particular, when comparing models across generations, it is unclear whether improved performance results from longer reasoning chains or more efficient reasoning. We systematically analyze chain-of-thought length across o1-mini and o3-mini variants on the Omni-MATH benchmark, finding that o3-mini (m) achieves superior accuracy without requiring longer reasoning chains than o1-mini. Moreover, we show that accuracy generally declines as reasoning chains grow across all models and compute settings, even when controlling for difficulty of the questions. This accuracy drop is significantly smaller in more proficient models, suggesting that new generations of reasoning models use test-time compute more effectively. Finally, we highlight that while o3-mini (h) achieves a marginal accuracy gain over o3-mini (m), it does so by allocating substantially more reasoning tokens across all problems, even the ones that o3-mini (m) can already solve. These findings provide new insights into the relationship between model capability and reasoning length, with implications for efficiency, scaling, and evaluation methodologies. 

**Abstract (ZH)**: 大规模语言模型在数学推理方面取得了显著进展，利用了思维链和测试时计算量扩展。然而，关于推理令牌使用与准确率提升之间的相互作用仍有许多待解答的问题。特别是在比较不同代际模型时，改善性能是源自更长的思维链还是更高效的推理尚不明确。我们系统地分析了在Omni-MATH基准上o1-mini和o3-mini变种的思维链长度，发现o3-mini (m)在不需要更长的思维链的情况下取得了更高的准确率。此外，我们展示了在所有模型和计算设置中，随着思维链的增长，准确率通常会下降，即使控制了问题难度。准确率的下降在能力更强的模型中更为轻微，表明新一代推理模型在测试时计算量的使用上更为有效。最后，我们指出虽然o3-mini (h)相对于o3-mini (m)在准确率上取得轻微提升，但它通过在所有问题上分配更多的推理令牌来实现这一点，即使是o3-mini (m)已经能够解决的问题也是如此。这些发现为模型能力与推理长度之间的关系提供了新的见解，对于效率、扩展性和评估方法具有重要意义。 

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
# Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing 

**Title (ZH)**: 探针修剪：通过基于模型的探针进行动态修剪加速LLMs 

**Authors**: Qi Le, Enmao Diao, Ziyan Wang, Xinran Wang, Jie Ding, Li Yang, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15618)  

**Abstract**: We introduce Probe Pruning (PP), a novel framework for online, dynamic, structured pruning of Large Language Models (LLMs) applied in a batch-wise manner. PP leverages the insight that not all samples and tokens contribute equally to the model's output, and probing a small portion of each batch effectively identifies crucial weights, enabling tailored dynamic pruning for different batches. It comprises three main stages: probing, history-informed pruning, and full inference. In the probing stage, PP selects a small yet crucial set of hidden states, based on residual importance, to run a few model layers ahead. During the history-informed pruning stage, PP strategically integrates the probing states with historical states. Subsequently, it structurally prunes weights based on the integrated states and the PP importance score, a metric developed specifically to assess the importance of each weight channel in maintaining performance. In the final stage, full inference is conducted on the remaining weights. A major advantage of PP is its compatibility with existing models, as it operates without requiring additional neural network modules or fine-tuning. Comprehensive evaluations of PP on LLaMA-2/3 and OPT models reveal that even minimal probing-using just 1.5% of FLOPs-can substantially enhance the efficiency of structured pruning of LLMs. For instance, when evaluated on LLaMA-2-7B with WikiText2, PP achieves a 2.56 times lower ratio of performance degradation per unit of runtime reduction compared to the state-of-the-art method at a 40% pruning ratio. Our code is available at this https URL. 

**Abstract (ZH)**: 探针剪枝 (Probe Pruning)：一种适用于大规模语言模型的在线动态结构剪枝框架 

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
# On the Robustness of Transformers against Context Hijacking for Linear Classification 

**Title (ZH)**: 关于Transformer在线性分类中对上下文劫持的鲁棒性 

**Authors**: Tianle Li, Chenyang Zhang, Xingwu Chen, Yuan Cao, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.15609)  

**Abstract**: Transformer-based Large Language Models (LLMs) have demonstrated powerful in-context learning capabilities. However, their predictions can be disrupted by factually correct context, a phenomenon known as context hijacking, revealing a significant robustness issue. To understand this phenomenon theoretically, we explore an in-context linear classification problem based on recent advances in linear transformers. In our setup, context tokens are designed as factually correct query-answer pairs, where the queries are similar to the final query but have opposite labels. Then, we develop a general theoretical analysis on the robustness of the linear transformers, which is formulated as a function of the model depth, training context lengths, and number of hijacking context tokens. A key finding is that a well-trained deeper transformer can achieve higher robustness, which aligns with empirical observations. We show that this improvement arises because deeper layers enable more fine-grained optimization steps, effectively mitigating interference from context hijacking. This is also well supported by our numerical experiments. Our findings provide theoretical insights into the benefits of deeper architectures and contribute to enhancing the understanding of transformer architectures. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLMs）展示了强大的上下文学习能力。然而，它们的预测可能会被事实正确的上下文所破坏，这一现象被称为上下文劫持，揭示了一个重要且明显的鲁棒性问题。为了从理论上理解这一现象，我们基于近期线性变压器的发展，探索了一个基于上下文的线性分类问题。在我们的设置中，上下文标记被设计为事实正确的查询-答案对，其中查询类似于最终查询但标签相反。然后，我们发展了一个关于线性变压器鲁棒性的通用理论分析，该分析表示为模型深度、训练上下文长度和劫持上下文标记数量的函数。一个重要发现是，一个训练良好的更深的变压器可以实现更高的鲁棒性，这与实证观察相符。我们展示了这一改进是因为更深的层能够实现更精细的优化步骤，有效地抵消了上下文劫持的干扰。我们的数值实验也很好地支持了这一点。我们的研究结果为更深架构的优势提供了理论见解，并有助于增强对变压器架构的理解。 

---
# Do Multilingual LLMs Think In English? 

**Title (ZH)**: 多语言大语言模型是否用英语思考？ 

**Authors**: Lisa Schut, Yarin Gal, Sebastian Farquhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15603)  

**Abstract**: Large language models (LLMs) have multilingual capabilities and can solve tasks across various languages. However, we show that current LLMs make key decisions in a representation space closest to English, regardless of their input and output languages. Exploring the internal representations with a logit lens for sentences in French, German, Dutch, and Mandarin, we show that the LLM first emits representations close to English for semantically-loaded words before translating them into the target language. We further show that activation steering in these LLMs is more effective when the steering vectors are computed in English rather than in the language of the inputs and outputs. This suggests that multilingual LLMs perform key reasoning steps in a representation that is heavily shaped by English in a way that is not transparent to system users. 

**Abstract (ZH)**: 大型语言模型（LLMs）具有多语言能力，并能在多种语言上完成任务。然而，我们展示了当前LLMs在进行关键决策时往往会接近英语的表示空间，而不考虑其输入和输出的语言。通过对法语、德语、荷兰语和 Mandarin 中的句子进行 logit 视角的内部表示探索，我们发现LLM首先为语义负载单词生成接近英语的表示，然后再将其翻译成目标语言。进一步研究表明，在这些LLM中，当引导向量是在英语而不是输入和输出语言中计算时，激活引导更为有效。这表明多语言LLMs以一种对系统用户不透明的方式，在受英语影响极大的表示空间中执行关键推理步骤。 

---
# KAD: No More FAD! An Effective and Efficient Evaluation Metric for Audio Generation 

**Title (ZH)**: KAD: 无更多FAD！一种有效的音频生成评估指标 

**Authors**: Yoonjin Chung, Pilsun Eu, Junwon Lee, Keunwoo Choi, Juhan Nam, Ben Sangbae Chon  

**Link**: [PDF](https://arxiv.org/pdf/2502.15602)  

**Abstract**: Although being widely adopted for evaluating generated audio signals, the Fréchet Audio Distance (FAD) suffers from significant limitations, including reliance on Gaussian assumptions, sensitivity to sample size, and high computational complexity. As an alternative, we introduce the Kernel Audio Distance (KAD), a novel, distribution-free, unbiased, and computationally efficient metric based on Maximum Mean Discrepancy (MMD). Through analysis and empirical validation, we demonstrate KAD's advantages: (1) faster convergence with smaller sample sizes, enabling reliable evaluation with limited data; (2) lower computational cost, with scalable GPU acceleration; and (3) stronger alignment with human perceptual judgments. By leveraging advanced embeddings and characteristic kernels, KAD captures nuanced differences between real and generated audio. Open-sourced in the kadtk toolkit, KAD provides an efficient, reliable, and perceptually aligned benchmark for evaluating generative audio models. 

**Abstract (ZH)**: 尽管广泛应用于评估生成的音频信号，Fréchet 音频距离（FAD）存在显著的局限性，包括依赖高斯假设、对样本大小敏感以及计算复杂度高。作为替代方案，我们引入了基于最大均值偏差（MMD）的核音频距离（KAD），这是一种新的、无需分布假设、无偏且计算高效的度量标准。通过分析和实证验证，我们展示了KAD的优势：（1）在较小的样本大小下更快收敛，从而能够在有限数据下获得可靠的评估；（2）计算成本更低，具有可扩展的GPU加速；（3）更贴近人类的感知判断。通过利用高级嵌入和特征核，KAD 能够捕捉真实音频和生成音频之间的细微差异。KAD 在 kadtk 工具包中开源，提供了一种高效、可靠且感知对齐的基准，用于评估生成音频模型。 

---
# WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents 

**Title (ZH)**: WorldCraft: 基于LLM代理的 PHOTO-真实感 3D 世界创建与定制 

**Authors**: Xinhang Liu, Chi-Keung Tang, Yu-Wing Tai  

**Link**: [PDF](https://arxiv.org/pdf/2502.15601)  

**Abstract**: Constructing photorealistic virtual worlds has applications across various fields, but it often requires the extensive labor of highly trained professionals to operate conventional 3D modeling software. To democratize this process, we introduce WorldCraft, a system where large language model (LLM) agents leverage procedural generation to create indoor and outdoor scenes populated with objects, allowing users to control individual object attributes and the scene layout using intuitive natural language commands. In our framework, a coordinator agent manages the overall process and works with two specialized LLM agents to complete the scene creation: ForgeIt, which integrates an ever-growing manual through auto-verification to enable precise customization of individual objects, and ArrangeIt, which formulates hierarchical optimization problems to achieve a layout that balances ergonomic and aesthetic considerations. Additionally, our pipeline incorporates a trajectory control agent, allowing users to animate the scene and operate the camera through natural language interactions. Our system is also compatible with off-the-shelf deep 3D generators to enrich scene assets. Through evaluations and comparisons with state-of-the-art methods, we demonstrate the versatility of WorldCraft, ranging from single-object customization to intricate, large-scale interior and exterior scene designs. This system empowers non-professionals to bring their creative visions to life. 

**Abstract (ZH)**: 基于 procedurally generated agents 的 WorldCraft：构建直观自然语言控制的 photorealistic 虚拟世界 

---
# Generalizing From Short to Long: Effective Data Synthesis for Long-Context Instruction Tuning 

**Title (ZH)**: 从短-context到长-context的泛化：有效的数据合成用于长-context指令调优 

**Authors**: Wenhao Zhu, Pinzhen Chen, Hanxu Hu, Shujian Huang, Fei Yuan, Jiajun Chen, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2502.15592)  

**Abstract**: Long-context modelling for large language models (LLMs) has been a key area of recent research because many real world use cases require reasoning over longer inputs such as documents. The focus of research into modelling long context has been on how to model position and there has been little investigation into other important aspects of language modelling such as instruction tuning. Long context training examples are challenging and expensive to create and use. In this paper, we investigate how to design instruction data for the post-training phase of a long context pre-trained model: how much and what type of context is needed for optimal and efficient post-training. Our controlled study reveals that models instruction-tuned on short contexts can effectively generalize to longer ones, while also identifying other critical factors such as instruction difficulty and context composition. Based on these findings, we propose context synthesis, a novel data synthesis framework that leverages off-the-shelf LLMs to generate extended background contexts for high-quality instruction-answer pairs. Experiment results on the document-level benchmark (LongBench) demonstrate that our proposed approach outperforms previous instruction synthesis approaches and comes close to the performance of human-annotated long-context instruction data. The project will be available at: this https URL. 

**Abstract (ZH)**: 长上下文建模对于大型语言模型(LLMs)而言是近期研究的关键领域，因为许多实际应用需要在文档等较长输入上进行推理。关于长上下文建模的研究重点在于位置建模，对语言模型中的其他重要方面如指令调优则研究较少。长上下文训练样本的创建和使用既具有挑战性也较为昂贵。在本文中，我们探讨了针对预训练的长上下文模型的后训练阶段如何设计指令数据：如何设计、多少以及什么类型的上下文可以实现最优且高效的后训练。我们的控制研究发现，针对短上下文进行指令调优的模型能够有效泛化到长上下文，同时识别出其他关键因素，如指令难度和上下文组成。基于上述发现，我们提出了一种新颖的数据合成框架——上下文合成，该框架利用现成的LLM生成高质量指令-答案对的扩展背景上下文。在文档级别基准(LongBench)上的实验结果表明，我们提出的方法优于以往的指令合成方法，并且性能接近人工标注的长上下文指令数据。该项目将在此处提供：this https URL。 

---
# LightThinker: Thinking Step-by-Step Compression 

**Title (ZH)**: LightThinker: 步步压缩 

**Authors**: Jintian Zhang, Yuqi Zhu, Mengshu Sun, Yujie Luo, Shuofei Qiao, Lun Du, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15589)  

**Abstract**: Large language models (LLMs) have shown remarkable performance in complex reasoning tasks, but their efficiency is hindered by the substantial memory and computational costs associated with generating lengthy tokens. In this paper, we propose LightThinker, a novel method that enables LLMs to dynamically compress intermediate thoughts during reasoning. Inspired by human cognitive processes, LightThinker compresses verbose thought steps into compact representations and discards the original reasoning chains, thereby significantly reducing the number of tokens stored in the context window. This is achieved by training the model on when and how to perform compression through data construction, mapping hidden states to condensed gist tokens, and creating specialized attention masks. Additionally, we introduce the Dependency (Dep) metric to quantify the degree of compression by measuring the reliance on historical tokens during generation. Extensive experiments on four datasets and two models show that LightThinker reduces peak memory usage and inference time, while maintaining competitive accuracy. Our work provides a new direction for improving the efficiency of LLMs in complex reasoning tasks without sacrificing performance. Code will be released at this https URL. 

**Abstract (ZH)**: 大语言模型（LLMs）在复杂推理任务中展现了卓越的性能，但生成长序列时的高效性受到大量内存和计算成本的限制。本文提出了一种名为LightThinker的新方法，能够使LLMs在推理过程中动态压缩中间思维。受到人类认知过程的启发，LightThinker将冗长的思维步骤压缩为紧凑表示，并丢弃原始的推理链，从而大幅减少存储在上下文窗口中的令牌数量。这通过在数据构造、将隐藏状态映射到浓缩的主旨令牌以及创建专门的注意力掩码等方式进行训练而实现。此外，我们引入了依赖度（Dep）指标来量化压缩的程度，通过衡量生成过程对历史令牌的依赖性。在四个数据集和两个模型上的实验显示，LightThinker在显著降低峰值内存使用和推理时间的同时，保持了竞争力的准确性。我们的工作为在不牺牲性能的情况下提高LLMs在复杂推理任务中的效率提供了新的方向。代码将在https://github.com/Qwen-Model/LightThinker仓库中发布。 

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
# A Cautionary Tale About "Neutrally" Informative AI Tools Ahead of the 2025 Federal Elections in Germany 

**Title (ZH)**: 关于2025年德国联邦选举前“中性”信息AI工具的警示故事 

**Authors**: Ina Dormuth, Sven Franke, Marlies Hafer, Tim Katzke, Alexander Marx, Emmanuel Müller, Daniel Neider, Markus Pauly, Jérôme Rutinowski  

**Link**: [PDF](https://arxiv.org/pdf/2502.15568)  

**Abstract**: In this study, we examine the reliability of AI-based Voting Advice Applications (VAAs) and large language models (LLMs) in providing objective political information. Our analysis is based upon a comparison with party responses to 38 statements of the Wahl-O-Mat, a well-established German online tool that helps inform voters by comparing their views with political party positions. For the LLMs, we identify significant biases. They exhibit a strong alignment (over 75% on average) with left-wing parties and a substantially lower alignment with center-right (smaller 50%) and right-wing parties (around 30%). Furthermore, for the VAAs, intended to objectively inform voters, we found substantial deviations from the parties' stated positions in Wahl-O-Mat: While one VAA deviated in 25% of cases, another VAA showed deviations in more than 50% of cases. For the latter, we even observed that simple prompt injections led to severe hallucinations, including false claims such as non-existent connections between political parties and right-wing extremist ties. 

**Abstract (ZH)**: 本研究考察了基于AI的投票建议应用程序（VAAs）和大型语言模型（LLMs）在提供客观政治信息方面的可靠性。 

---
# Bridging vision language model (VLM) evaluation gaps with a framework for scalable and cost-effective benchmark generation 

**Title (ZH)**: 使用框架实现可扩展且成本有效的基准生成以弥合视觉语言模型评估差距 

**Authors**: Tim Rädsch, Leon Mayer, Simon Pavicic, A. Emre Kavur, Marcel Knopp, Barış Öztürk, Klaus Maier-Hein, Paul F. Jaeger, Fabian Isensee, Annika Reinke, Lena Maier-Hein  

**Link**: [PDF](https://arxiv.org/pdf/2502.15563)  

**Abstract**: Reliable evaluation of AI models is critical for scientific progress and practical application. While existing VLM benchmarks provide general insights into model capabilities, their heterogeneous designs and limited focus on a few imaging domains pose significant challenges for both cross-domain performance comparison and targeted domain-specific evaluation. To address this, we propose three key contributions: (1) a framework for the resource-efficient creation of domain-specific VLM benchmarks enabled by task augmentation for creating multiple diverse tasks from a single existing task, (2) the release of new VLM benchmarks for seven domains, created according to the same homogeneous protocol and including 162,946 thoroughly human-validated answers, and (3) an extensive benchmarking of 22 state-of-the-art VLMs on a total of 37,171 tasks, revealing performance variances across domains and tasks, thereby supporting the need for tailored VLM benchmarks. Adoption of our methodology will pave the way for the resource-efficient domain-specific selection of models and guide future research efforts toward addressing core open questions. 

**Abstract (ZH)**: 可靠的AI模型评估对于科学进步和实际应用至关重要。尽管现有的VLM基准提供了关于模型能力的一般洞察，但它们多样化的设计和对少数成像领域有限的关注，对跨域性能比较和特定领域评估提出了重大挑战。为此，我们提出了三个关键贡献：（1）一种通过任务增强创建单个现有任务衍生多种多样任务以实现资源高效创建领域特定VLM基准的框架；（2）推出针对七个领域的新VLM基准，根据相同的均匀协议创建，并包含162,946个全面的人工验证答案；（3）对22种最先进的VLM在总计37,171个任务上的广泛评估，揭示了不同领域和任务间的性能差异，从而支持定制化VLM基准的必要性。采用我们的方法将为资源高效选择模型铺平道路，并指导未来研究致力于解决核心开放问题。 

---
# PIP-KAG: Mitigating Knowledge Conflicts in Knowledge-Augmented Generation via Parametric Pruning 

**Title (ZH)**: PIP-KAG：通过参数化剪枝缓解知识增强生成中的知识冲突 

**Authors**: Pengcheng Huang, Zhenghao Liu, Yukun Yan, Xiaoyuan Yi, Hao Chen, Zhiyuan Liu, Maosong Sun, Tong Xiao, Ge Yu, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.15543)  

**Abstract**: Knowledge-Augmented Generation (KAG) has shown great promise in updating the internal memory of Large Language Models (LLMs) by integrating external knowledge. However, KAG inevitably faces knowledge conflicts when the internal memory contradicts external information. Current approaches to mitigating these conflicts mainly focus on improving external knowledge utilization. However, these methods have shown only limited effectiveness in mitigating the knowledge conflict problem, as internal knowledge continues to influence the generation process of LLMs. In this paper, we propose a ParametrIc Pruning-based Knowledge-Augmented Generation (PIP-KAG) approach, which prunes internal knowledge of LLMs and incorporates a plug-and-play adaptation module to help LLMs better leverage external sources. Additionally, we construct the CoConflictQA benchmark based on the hallucination of LLMs to better evaluate contextual faithfulness during answering questions. Experimental results on CoConflictQA demonstrate that PIP-KAG significantly reduces knowledge conflicts and improves context fidelity. Notably, PIP-KAG reduces LLM's parameters by 13%, enhancing parameter efficiency in LLMs within the KAG framework. All codes are available at this https URL. 

**Abstract (ZH)**: ParametrIc Pruning-based Knowledge-Augmented Generation (PIP-KAG)及其在知识增强生成中的应用：减少知识冲突并提高上下文 fidelity 

---
# Bridging Domain Gaps between Pretrained Multimodal Models and Recommendations 

**Title (ZH)**: 预训练多模态模型与推荐系统之间的域差距桥梁构建 

**Authors**: Wenyu Zhang, Jie Luo, Xinming Zhang, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15542)  

**Abstract**: With the explosive growth of multimodal content online, pre-trained visual-language models have shown great potential for multimodal recommendation. However, while these models achieve decent performance when applied in a frozen manner, surprisingly, due to significant domain gaps (e.g., feature distribution discrepancy and task objective misalignment) between pre-training and personalized recommendation, adopting a joint training approach instead leads to performance worse than baseline. Existing approaches either rely on simple feature extraction or require computationally expensive full model fine-tuning, struggling to balance effectiveness and efficiency. To tackle these challenges, we propose \textbf{P}arameter-efficient \textbf{T}uning for \textbf{M}ultimodal \textbf{Rec}ommendation (\textbf{PTMRec}), a novel framework that bridges the domain gap between pre-trained models and recommendation systems through a knowledge-guided dual-stage parameter-efficient training strategy. This framework not only eliminates the need for costly additional pre-training but also flexibly accommodates various parameter-efficient tuning methods. 

**Abstract (ZH)**: 参数高效调 tune 多模态推荐（PTMRec） 

---
# Depth-aware Fusion Method based on Image and 4D Radar Spectrum for 3D Object Detection 

**Title (ZH)**: 基于图像和4D雷达谱的深度感知融合方法用于三维物体检测 

**Authors**: Yue Sun, Yeqiang Qian, Chunxiang Wang, Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15516)  

**Abstract**: Safety and reliability are crucial for the public acceptance of autonomous driving. To ensure accurate and reliable environmental perception, intelligent vehicles must exhibit accuracy and robustness in various environments. Millimeter-wave radar, known for its high penetration capability, can operate effectively in adverse weather conditions such as rain, snow, and fog. Traditional 3D millimeter-wave radars can only provide range, Doppler, and azimuth information for objects. Although the recent emergence of 4D millimeter-wave radars has added elevation resolution, the radar point clouds remain sparse due to Constant False Alarm Rate (CFAR) operations. In contrast, cameras offer rich semantic details but are sensitive to lighting and weather conditions. Hence, this paper leverages these two highly complementary and cost-effective sensors, 4D millimeter-wave radar and camera. By integrating 4D radar spectra with depth-aware camera images and employing attention mechanisms, we fuse texture-rich images with depth-rich radar data in the Bird's Eye View (BEV) perspective, enhancing 3D object detection. Additionally, we propose using GAN-based networks to generate depth images from radar spectra in the absence of depth sensors, further improving detection accuracy. 

**Abstract (ZH)**: 智能驾驶的安全性和可靠性是公众接受的关键。为了确保环境感知的准确性和可靠性，智能车辆必须在各种环境中展现出高度的精度和鲁棒性。毫米波雷达因其高穿透能力，在雨、雪、雾等不良天气条件下仍能有效工作。传统3D毫米波雷达仅能提供物体的距离、多普勒和方位信息。尽管最近出现了4D毫米波雷达，增加了垂直分辨率，但由于恒定虚假警报率（CFAR）的操作，雷达点云仍然较为稀疏。相比之下，摄像头提供丰富的语义细节，但对光照和天气条件敏感。因此，本文利用这两种互补且成本效益高的传感器——4D毫米波雷达和摄像头。通过将4D雷达频谱与深度感知摄像头图像结合，并采用注意力机制，我们在鸟瞰图（BEV）视角下融合纹理丰富的图像和深度丰富的雷达数据，提升3D物体检测。此外，我们提出了使用基于GAN的网络生成深度图像的方法，进一步提高检测准确性。 

---
# Activation Steering in Neural Theorem Provers 

**Title (ZH)**: 神经定理证明中的激活转向 

**Authors**: Shashank Kirtania  

**Link**: [PDF](https://arxiv.org/pdf/2502.15507)  

**Abstract**: Large Language Models (LLMs) have shown promise in proving formal theorems using proof assistants like Lean. However, current state of the art language models struggles to predict next step in proofs leading practitioners to use different sampling techniques to improve LLMs capabilities. We observe that the LLM is capable of predicting the correct tactic; however, it faces challenges in ranking it appropriately within the set of candidate tactics, affecting the overall selection process. To overcome this hurdle, we use activation steering to guide LLMs responses to improve the generations at the time of inference. Our results suggest that activation steering offers a promising lightweight alternative to specialized fine-tuning for enhancing theorem proving capabilities in LLMs, particularly valuable in resource-constrained environments. 

**Abstract (ZH)**: 大型语言模型在使用Lean等证明辅助器证明形式定理方面显示出潜力。然而，当前最先进的语言模型在预测证明中的下一步时存在困难，促使实践者使用不同的采样技术来提高LLM的能力。我们观察到，LLM能够预测正确的战术，但在适当排序候选战术的过程中面临挑战，影响整体选择过程。为了克服这一障碍，我们使用激活引导来指导LLM的响应，以在推理时改进生成。我们的结果表明，激活引导为在资源受限环境中增强LLM的定理证明能力提供了一种有希望的轻量级替代方案，特别是专门微调的替代方案。 

---
# BAN: Neuroanatomical Aligning in Auditory Recognition between Artificial Neural Network and Human Cortex 

**Title (ZH)**: BAN: 人工神经网络与人类皮层在听觉识别中的神经解剖对齐 

**Authors**: Haidong Wang, Pengfei Xiao, Ao Liu, Jianhua Zhang, Qia Shan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15503)  

**Abstract**: Drawing inspiration from neurosciences, artificial neural networks (ANNs) have evolved from shallow architectures to highly complex, deep structures, yielding exceptional performance in auditory recognition tasks. However, traditional ANNs often struggle to align with brain regions due to their excessive depth and lack of biologically realistic features, like recurrent connection. To address this, a brain-like auditory network (BAN) is introduced, which incorporates four neuroanatomically mapped areas and recurrent connection, guided by a novel metric called the brain-like auditory score (BAS). BAS serves as a benchmark for evaluating the similarity between BAN and human auditory recognition pathway. We further propose that specific areas in the cerebral cortex, mainly the middle and medial superior temporal (T2/T3) areas, correspond to the designed network structure, drawing parallels with the brain's auditory perception pathway. Our findings suggest that the neuroanatomical similarity in the cortex and auditory classification abilities of the ANN are well-aligned. In addition to delivering excellent performance on a music genre classification task, the BAN demonstrates a high BAS score. In conclusion, this study presents BAN as a recurrent, brain-inspired ANN, representing the first model that mirrors the cortical pathway of auditory recognition. 

**Abstract (ZH)**: 来自神经科学的启发，人工神经网络（ANNs）从浅层架构发展成为高度复杂的深层结构，在听觉识别任务中取得了卓越的性能。然而，传统ANNs往往因其过度的深度和缺乏生物学现实性的特征（如循环连接）而难以与脑区对齐。为解决这一问题，引入了一种脑启发式听觉网络（BAN），该网络整合了四个神经解剖学映射区域和循环连接，并以一种新的指标——脑启发式听觉评分（BAS）为指导。BAS用作评估BAN与人类听觉识别路径相似性的基准。进一步的研究表明，大脑皮层中的特定区域，主要是上顶颞中回（T2/T3）区域，对应于设计的网络结构，与大脑的听觉感知路径相吻合。研究发现表明，大脑皮层的神经解剖相似性和ANN的听觉分类能力是高度协调的。除了在音乐流派分类任务中表现出色外，BAN还具有高BAS评分。总之，本研究展示了BAN作为一种具有循环连接的脑启发式ANN，是首个模拟听觉识别大脑皮层路径的模型。 

---
# Q-PETR: Quant-aware Position Embedding Transformation for Multi-View 3D Object Detection 

**Title (ZH)**: Q-PETR：面向多视图3D物体检测的量化感知位置嵌入变换 

**Authors**: Jiangyong Yu, Changyong Shu, Dawei Yang, Zichen Yu, Xing Hu, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15488)  

**Abstract**: PETR-based methods have dominated benchmarks in 3D perception and are increasingly becoming a key component in modern autonomous driving systems. However, their quantization performance significantly degrades when INT8 inference is required, with a degradation of 58.2% in mAP and 36.9% in NDS on the NuScenes dataset. To address this issue, we propose a quantization-aware position embedding transformation for multi-view 3D object detection, termed Q-PETR. Q-PETR offers a quantizationfriendly and deployment-friendly architecture while preserving the original performance of PETR. It substantially narrows the accuracy gap between INT8 and FP32 inference for PETR-series methods. Without bells and whistles, our approach reduces the mAP and NDS drop to within 1% under standard 8-bit per-tensor post-training quantization. Furthermore, our method exceeds the performance of the original PETR in terms of floating-point precision. Extensive experiments across a variety of PETR-series models demonstrate its broad generalization. 

**Abstract (ZH)**: 基于PETR的方法在3D感知基准测试中占据主导地位，并逐渐成为现代自动驾驶系统的关键组件。然而，在要求INT8推理时，其量化性能显著下降，NuScenes数据集上的mAP降噪8.2%，NDS降噪36.9%。为解决这一问题，我们提出了一种适用于多视图3D物体检测的量化感知位置嵌入变换，称为Q-PETR。Q-PETR提供了一种量化友好且部署友好的架构，同时保持了PETR的原有性能。它显著缩小了PETR系列方法在INT8和FP32推理之间准确率的差距。在标准8位张量后训练量化下，我们的方法将mAP和NDS的下降幅度降低到不到1%。此外，相对于浮点精度，我们的方法在性能上超过了原始的PETR。广泛实验表明，该方法在各种PETR系列模型中具有广泛的通用性。 

---
# ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models 

**Title (ZH)**: ExpliCa: 评估大型语言模型中的显式因果推理能力 

**Authors**: Martina Miliani, Serenna Auriemma, Alessandro Bondielli, Emmanuele Chersoni, Lucia Passaro, Irene Sucameli, Alessandro Lenci  

**Link**: [PDF](https://arxiv.org/pdf/2502.15487)  

**Abstract**: Large Language Models (LLMs) are increasingly used in tasks requiring interpretive and inferential accuracy. In this paper, we introduce ExpliCa, a new dataset for evaluating LLMs in explicit causal reasoning. ExpliCa uniquely integrates both causal and temporal relations presented in different linguistic orders and explicitly expressed by linguistic connectives. The dataset is enriched with crowdsourced human acceptability ratings. We tested LLMs on ExpliCa through prompting and perplexity-based metrics. We assessed seven commercial and open-source LLMs, revealing that even top models struggle to reach 0.80 accuracy. Interestingly, models tend to confound temporal relations with causal ones, and their performance is also strongly influenced by the linguistic order of the events. Finally, perplexity-based scores and prompting performance are differently affected by model size. 

**Abstract (ZH)**: 大型语言模型（LLMs）在需要解释性和推理准确性的任务中日益广泛应用。本文介绍了ExpliCa，一个用于评估LLMs在显式因果推理中的新数据集。ExpliCa独特地整合了以不同语言顺序呈现的因果关系和时间关系，并通过语言连接词明确表达。该数据集包含了众包的人类接受性评分。我们通过提示和困惑度基指标测试了LLMs，并评估了七个商用和开源LLM，结果显示即使顶级模型也难以达到0.80的准确性。有趣的是，模型倾向于混淆时间关系与因果关系，而事件的语言顺序对模型性能也有强烈影响。最后，困惑度基评分和提示性能受模型规模的影响不同。 

---
# Enhancing RWKV-based Language Models for Long-Sequence Text Generation 

**Title (ZH)**: 基于RWKV的长序列文本生成语言模型的增强方法 

**Authors**: Xinghan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15485)  

**Abstract**: This paper presents an enhanced RWKV-based language generation model designed to improve long-sequence text processing. We propose an adaptive token shift and gating mechanism to better capture long-range dependencies in text generation. Through a series of experiments, we compare the baseline RWKV model with the enhanced model, evaluating performance in terms of forward propagation time, text generation quality, and automatic evaluation metrics such as perplexity, BLEU, and ROUGE. Experimental results show that the enhanced model significantly improves generation quality, especially in BLEU and ROUGE scores, and demonstrates stronger context-capturing ability in long-text generation tasks. 

**Abstract (ZH)**: 基于增强RWKV的语言生成模型：提高长序列文本处理能力及长文本生成中的远程依赖捕捉机制 

---
# PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System 

**Title (ZH)**: PAPI: 利用计算系统中存内计算技术在大规模语言模型解码中发挥动态并行性优势 

**Authors**: Yintao He, Haiyu Mao, Christina Giannoula, Mohammad Sadrosadati, Juan Gómez-Luna, Huawei Li, Xiaowei Li, Ying Wang, Onur Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15470)  

**Abstract**: Large language models (LLMs) are widely used for natural language understanding and text generation. An LLM model relies on a time-consuming step called LLM decoding to generate output tokens. Several prior works focus on improving the performance of LLM decoding using parallelism techniques, such as batching and speculative decoding. State-of-the-art LLM decoding has both compute-bound and memory-bound kernels. Some prior works statically identify and map these different kernels to a heterogeneous architecture consisting of both processing-in-memory (PIM) units and computation-centric accelerators. We observe that characteristics of LLM decoding kernels (e.g., whether or not a kernel is memory-bound) can change dynamically due to parameter changes to meet user and/or system demands, making (1) static kernel mapping to PIM units and computation-centric accelerators suboptimal, and (2) one-size-fits-all approach of designing PIM units inefficient due to a large degree of heterogeneity even in memory-bound kernels.
In this paper, we aim to accelerate LLM decoding while considering the dynamically changing characteristics of the kernels involved. We propose PAPI (PArallel Decoding with PIM), a PIM-enabled heterogeneous architecture that exploits dynamic scheduling of compute-bound or memory-bound kernels to suitable hardware units. PAPI has two key mechanisms: (1) online kernel characterization to dynamically schedule kernels to the most suitable hardware units at runtime and (2) a PIM-enabled heterogeneous computing system that harmoniously orchestrates both computation-centric processing units and hybrid PIM units with different computing capabilities. Our experimental results on three broadly-used LLMs show that PAPI achieves 1.8$\times$ and 11.1$\times$ speedups over a state-of-the-art heterogeneous LLM accelerator and a state-of-the-art PIM-only LLM accelerator, respectively. 

**Abstract (ZH)**: 基于PIM的大规模语言模型解码加速器：动态调度计算和内存绑定内核 

---
# Mitigating Data Scarcity in Time Series Analysis: A Foundation Model with Series-Symbol Data Generation 

**Title (ZH)**: 在时间序列分析中缓解数据稀缺性：基于系列符号数据生成的foundation模型 

**Authors**: Wenxuan Wang, Kai Wu, Yujian Betterest Li, Dan Wang, Xiaoyu Zhang, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15466)  

**Abstract**: Foundation models for time series analysis (TSA) have attracted significant attention. However, challenges such as data scarcity and data imbalance continue to hinder their development. To address this, we consider modeling complex systems through symbolic expressions that serve as semantic descriptors of time series. Building on this concept, we introduce a series-symbol (S2) dual-modulity data generation mechanism, enabling the unrestricted creation of high-quality time series data paired with corresponding symbolic representations. Leveraging the S2 dataset, we develop SymTime, a pre-trained foundation model for TSA. SymTime demonstrates competitive performance across five major TSA tasks when fine-tuned with downstream task, rivaling foundation models pre-trained on real-world datasets. This approach underscores the potential of dual-modality data generation and pretraining mechanisms in overcoming data scarcity and enhancing task performance. 

**Abstract (ZH)**: 符号表达驱动的时空序列分析基础模型 

---
# R-LoRA: Random Initialization of Multi-Head LoRA for Multi-Task Learning 

**Title (ZH)**: R-LoRA: 多任务学习中多头LoRA的随机初始化 

**Authors**: Jinda Liu, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15455)  

**Abstract**: Fine-tuning large language models (LLMs) is prohibitively expensive in terms of computational and memory costs. Low-rank Adaptation (LoRA), as one of the most popular parameter-efficient fine-tuning (PEFT) methods, offers a cost-effective alternative by approximating the model changes $\Delta W \in \mathbb{R}^{m \times n}$ through the product of down-projection matrix $A \in \mathbb{R}^{m \times r}$ and head matrix $B \in \mathbb{R}^{r \times n}$, where $r \ll \min(m, n)$. In real-world scenarios, LLMs are fine-tuned on data from multiple domains to perform tasks across various fields, embodying multi-task learning (MTL). LoRA often underperforms in such complex scenarios. To enhance LoRA's capability in multi-task learning, we propose R-LoRA, which incorporates Multi-Head Randomization. Multi-Head Randomization diversifies the head matrices through Multi-Head Random Initialization and Multi-Head Dropout, enabling more efficient learning of task-specific features while maintaining shared knowledge representation. Extensive experiments demonstrate that R-LoRA is better at capturing task-specific knowledge, thereby improving performance in multi-task scenarios. The code is available at this https URL. 

**Abstract (ZH)**: 改进LoRA促进多任务学习：基于多头随机化的R-LoRA 

---
# MVIP -- A Dataset and Methods for Application Oriented Multi-View and Multi-Modal Industrial Part Recognition 

**Title (ZH)**: MVIP -- 一种面向应用的多视图多模态工业零件识别数据集和方法 

**Authors**: Paul Koch, Marian Schlüter, Jörg Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2502.15448)  

**Abstract**: We present MVIP, a novel dataset for multi-modal and multi-view application-oriented industrial part recognition. Here we are the first to combine a calibrated RGBD multi-view dataset with additional object context such as physical properties, natural language, and super-classes. The current portfolio of available datasets offers a wide range of representations to design and benchmark related methods. In contrast to existing classification challenges, industrial recognition applications offer controlled multi-modal environments but at the same time have different problems than traditional 2D/3D classification challenges. Frequently, industrial applications must deal with a small amount or increased number of training data, visually similar parts, and varying object sizes, while requiring a robust near 100% top 5 accuracy under cost and time constraints. Current methods tackle such challenges individually, but direct adoption of these methods within industrial applications is complex and requires further research. Our main goal with MVIP is to study and push transferability of various state-of-the-art methods within related downstream tasks towards an efficient deployment of industrial classifiers. Additionally, we intend to push with MVIP research regarding several modality fusion topics, (automated) synthetic data generation, and complex data sampling -- combined in a single application-oriented benchmark. 

**Abstract (ZH)**: MVIP：面向多模态多视角工业部件识别的新型数据集 

---
# When Compression Meets Model Compression: Memory-Efficient Double Compression for Large Language Models 

**Title (ZH)**: 当压缩遇到模型压缩：适用于大型语言模型的内存高效双重压缩方法 

**Authors**: Weilan Wang, Yu Mao, Dongdong Tang, Hongchao Du, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.15443)  

**Abstract**: Large language models (LLMs) exhibit excellent performance in various tasks. However, the memory requirements of LLMs present a great challenge when deploying on memory-limited devices, even for quantized LLMs. This paper introduces a framework to compress LLM after quantization further, achieving about 2.2x compression ratio. A compression-aware quantization is first proposed to enhance model weight compressibility by re-scaling the model parameters before quantization, followed by a pruning method to improve further. Upon this, we notice that decompression can be a bottleneck during practical scenarios. We then give a detailed analysis of the trade-off between memory usage and latency brought by the proposed method. A speed-adaptive method is proposed to overcome it. The experimental results show inference with the compressed model can achieve a 40% reduction in memory size with negligible loss in accuracy and inference speed. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中表现出色。然而，LLMs的内存要求在部署到内存受限的设备上时会带来巨大挑战，即使是量化后的LLMs也是如此。本文提出了一种框架，在量化后进一步压缩LLMs，实现了约2.2倍的压缩比。首先提出了感知压缩的量化方法，通过重新缩放模型参数来增强模型权重的压缩性，随后采用了剪枝方法进一步改进。进一步观察到，在实际场景中解压缩可能会成为瓶颈。因此，我们对所提出方法带来的内存使用与延迟之间的权衡进行了详细分析。提出了一种自适应提速方法来克服这一问题。实验结果表明，使用压缩模型的推理可以帮助减少40%的内存大小，同时几乎不影响准确性和推理速度。 

---
# Fed-SB: A Silver Bullet for Extreme Communication Efficiency and Performance in (Private) Federated LoRA Fine-Tuning 

**Title (ZH)**: Fed-SB: 极端通信效率和性能的银弹方案在(私有)联邦LoRA微调中 

**Authors**: Raghav Singhal, Kaustubh Ponkshe, Rohit Vartak, Lav R. Varshney, Praneeth Vepakomma  

**Link**: [PDF](https://arxiv.org/pdf/2502.15436)  

**Abstract**: Low-Rank Adaptation (LoRA) has become ubiquitous for efficiently fine-tuning foundation models. However, federated fine-tuning using LoRA is challenging due to suboptimal updates arising from traditional federated averaging of individual adapters. Existing solutions either incur prohibitively high communication cost that scales linearly with the number of clients or suffer from performance degradation due to limited expressivity. We introduce Federated Silver Bullet (Fed-SB), a novel approach for federated fine-tuning of LLMs using LoRA-SB, a recently proposed low-rank adaptation method. LoRA-SB optimally aligns the optimization trajectory with the ideal low-rank full fine-tuning projection by learning a small square matrix (R) between adapters B and A, keeping other components fixed. Direct averaging of R guarantees exact updates, substantially reducing communication cost, which remains independent of the number of clients, and enables scalability. Fed-SB achieves state-of-the-art performance across commonsense reasoning, arithmetic reasoning, and language inference tasks while reducing communication costs by up to 230x. In private settings, Fed-SB further improves performance by (1) reducing trainable parameters, thereby lowering the noise required for differential privacy and (2) avoiding noise amplification introduced by other methods. Overall, Fed-SB establishes a new Pareto frontier in the tradeoff between communication and performance, offering an efficient and scalable solution for both private and non-private federated fine-tuning. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 联邦自适应银弹（Fed-SB）：基于LoRA-SB的高效可扩展的LLM联邦微调 

---
# Single-pass Detection of Jailbreaking Input in Large Language Models 

**Title (ZH)**: 大型语言模型中一次性检测越狱输入的方法 

**Authors**: Leyla Naz Candogan, Yongtao Wu, Elias Abad Rocamora, Grigorios G. Chrysos, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2502.15435)  

**Abstract**: Defending aligned Large Language Models (LLMs) against jailbreaking attacks is a challenging problem, with existing approaches requiring multiple requests or even queries to auxiliary LLMs, making them computationally heavy. Instead, we focus on detecting jailbreaking input in a single forward pass. Our method, called Single Pass Detection SPD, leverages the information carried by the logits to predict whether the output sentence will be harmful. This allows us to defend in just one forward pass. SPD can not only detect attacks effectively on open-source models, but also minimizes the misclassification of harmless inputs. Furthermore, we show that SPD remains effective even without complete logit access in GPT-3.5 and GPT-4. We believe that our proposed method offers a promising approach to efficiently safeguard LLMs against adversarial attacks. 

**Abstract (ZH)**: 防御对齐的大语言模型（LLMs）免受牢笼攻击：单向检测方法（SPD） 

---
# Anatomy-Informed Deep Learning and Radiomics for Automated Neurofibroma Segmentation in Whole-Body MRI 

**Title (ZH)**: 基于解剖学信息的深度学习和影像组学在全身MRI中自动神经纤维瘤分割中的应用 

**Authors**: Georgii Kolokolnikov, Marie-Lena Schmalhofer, Lennart Well, Said Farschtschi, Victor-Felix Mautner, Inka Ristow, Rene Werner  

**Link**: [PDF](https://arxiv.org/pdf/2502.15424)  

**Abstract**: Neurofibromatosis Type 1 is a genetic disorder characterized by the development of neurofibromas (NFs), which exhibit significant variability in size, morphology, and anatomical location. Accurate and automated segmentation of these tumors in whole-body magnetic resonance imaging (WB-MRI) is crucial to assess tumor burden and monitor disease progression. In this study, we present and analyze a fully automated pipeline for NF segmentation in fat-suppressed T2-weighted WB-MRI, consisting of three stages: anatomy segmentation, NF segmentation, and tumor candidate classification. In the first stage, we use the MRSegmentator model to generate an anatomy segmentation mask, extended with a high-risk zone for NFs. This mask is concatenated with the input image as anatomical context information for NF segmentation. The second stage employs an ensemble of 3D anisotropic anatomy-informed U-Nets to produce an NF segmentation confidence mask. In the final stage, tumor candidates are extracted from the confidence mask and classified based on radiomic features, distinguishing tumors from non-tumor regions and reducing false positives. We evaluate the proposed pipeline on three test sets representing different conditions: in-domain data (test set 1), varying imaging protocols and field strength (test set 2), and low tumor burden cases (test set 3). Experimental results show a 68% improvement in per-scan Dice Similarity Coefficient (DSC), a 21% increase in per-tumor DSC, and a two-fold improvement in F1 score for tumor detection in high tumor burden cases by integrating anatomy information. The method is integrated into the 3D Slicer platform for practical clinical use, with the code publicly accessible. 

**Abstract (ZH)**: 神经纤维瘤病1型是一种由神经纤维瘤（NFs）发展引起的遗传性疾病，这些肿瘤在大小、形态和解剖位置上表现出显著的变异。对全身磁共振成像（WB-MRI）进行准确且自动的NF分割对于评估肿瘤负荷和监测疾病进展至关重要。本研究提出并分析了一种完全自动的NF分割管道，该管道由三个阶段组成：解剖分割、NF分割和肿瘤候选区域分类。在第一阶段，我们使用MRSegmentator模型生成包括NF高风险区的解剖分割掩模。该掩模与输入图像拼接，作为NF分割的解剖上下文信息。第二阶段采用一组3D各向异性解剖信息指导的U-Nets生成NF分割置信度掩模。在最终阶段，从置信度掩模中提取肿瘤候选区域并基于影像组学特征进行分类，以区分肿瘤区和非肿瘤区并减少误检。我们使用三个代表不同条件的测试集评估了该管道：领域内数据（测试集1）、不同成像协议和场强（测试集2）、以及低肿瘤负荷病例（测试集3）。实验结果表明，在高肿瘤负荷病例中结合解剖信息可以提高每个扫描的Dice相似性系数68%，每个肿瘤的Dice相似性系数提高21%，肿瘤检测的F1分数提高两倍。该方法已集成到3D Slicer平台中，用于实际临床使用，代码已公开。 

---
# Evaluating Multimodal Generative AI with Korean Educational Standards 

**Title (ZH)**: 基于韩国教育标准评估多模态生成人工智能 

**Authors**: Sanghee Park, Geewook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.15422)  

**Abstract**: This paper presents the Korean National Educational Test Benchmark (KoNET), a new benchmark designed to evaluate Multimodal Generative AI Systems using Korean national educational tests. KoNET comprises four exams: the Korean Elementary General Educational Development Test (KoEGED), Middle (KoMGED), High (KoHGED), and College Scholastic Ability Test (KoCSAT). These exams are renowned for their rigorous standards and diverse questions, facilitating a comprehensive analysis of AI performance across different educational levels. By focusing on Korean, KoNET provides insights into model performance in less-explored languages. We assess a range of models - open-source, open-access, and closed APIs - by examining difficulties, subject diversity, and human error rates. The code and dataset builder will be made fully open-sourced at this https URL. 

**Abstract (ZH)**: 韩国国家教育测试基准（KoNET）：面向韩语文本的多模态生成人工智能系统评估基准 

---
# Beyond Translation: LLM-Based Data Generation for Multilingual Fact-Checking 

**Title (ZH)**: 超越翻译：基于LLM的数据生成在多语言事实核实中的应用 

**Authors**: Yi-Ling Chung, Aurora Cobo, Pablo Serna  

**Link**: [PDF](https://arxiv.org/pdf/2502.15419)  

**Abstract**: Robust automatic fact-checking systems have the potential to combat online misinformation at scale. However, most existing research primarily focuses on English. In this paper, we introduce MultiSynFact, the first large-scale multilingual fact-checking dataset containing 2.2M claim-source pairs designed to support Spanish, German, English, and other low-resource languages. Our dataset generation pipeline leverages Large Language Models (LLMs), integrating external knowledge from Wikipedia and incorporating rigorous claim validation steps to ensure data quality. We evaluate the effectiveness of MultiSynFact across multiple models and experimental settings. Additionally, we open-source a user-friendly framework to facilitate further research in multilingual fact-checking and dataset generation. 

**Abstract (ZH)**: 多语言事实核查数据集MultiSynFact在大规模打击网络虚假信息方面具有潜力。然而，现有的大部分研究主要集中在英语上。本文介绍了MultiSynFact，这是首个包含220万条断言-来源对的大规模多语言事实核查数据集，旨在支持西班牙语、德语、英语及其他低资源语言。我们的数据集生成管道利用了大型语言模型（LLMs），从Wikipedia中集成外部知识，并结合严格的断言验证步骤以确保数据质量。我们跨多个模型和实验设置评估了MultiSynFact的有效性。此外，我们开源了一个用户友好的框架，以促进多语言事实核查和数据集生成的进一步研究。 

---
# HiFi-KPI: A Dataset for Hierarchical KPI Extraction from Earnings Filings 

**Title (ZH)**: HiFi-KPI： earnings 报告中层级关键绩效指标提取的数据集 

**Authors**: Rasmus Aavang, Giovanni Rizzi, Rasmus Bøggild, Alexandre Iolov, Mike Zhang, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2502.15411)  

**Abstract**: The U.S. Securities and Exchange Commission (SEC) requires that public companies file financial reports tagging numbers with the machine readable inline eXtensible Business Reporting Language (iXBRL) standard. However, the highly complex and highly granular taxonomy defined by iXBRL limits label transferability across domains. In this paper, we introduce the Hierarchical Financial Key Performance Indicator (HiFi-KPI) dataset, designed to facilitate numerical KPI extraction at specified levels of granularity from unstructured financial text. Our approach organizes a 218,126-label hierarchy using a taxonomy based grouping method, investigating which taxonomy layer provides the most meaningful structure. HiFi-KPI comprises ~1.8M paragraphs and ~5M entities, each linked to a label in the iXBRL-specific calculation and presentation taxonomies. We provide baselines using encoder-based approaches and structured extraction using Large Language Models (LLMs). To simplify LLM inference and evaluation, we additionally release HiFi-KPI Lite, a manually curated subset with four expert-mapped labels. We publicly release all artifacts 

**Abstract (ZH)**: 美国证券交易委员会（SEC）要求公共公司使用机器可读的即时扩展商业报告语言（iXBRL）标准标记财务报告中的数字。然而，iXBRL定义的复杂且精细的分类标准限制了标签在不同领域的转移性。本文介绍了Hierarchical Financial Key Performance Indicator（HiFi-KPI）数据集，旨在促进从非结构化的财务文本中在指定粒度级别提取关键绩效指标（KPI）。我们的方法使用基于分类法分组的方法构建了一个包含218,126个标签的层级结构，并探讨了哪种分类法层次提供了最具意义的结构。HiFi-KPI包含了约180万段落和约500万个实体，每个实体都链接到iXBRL特定的计算和呈现分类标准中的一个标签。我们提供了基于编码器的方法和使用大型语言模型（LLMs）进行结构化提取的基线。为简化LLM推理和评估，我们还发布了HiFi-KPI Lite，这是一个人工筛选的子集，包含四个专家映射的标签。所有成果均已公开发布。 

---
# Enhancing Vehicle Make and Model Recognition with 3D Attention Modules 

**Title (ZH)**: 增强车辆品牌和型号识别的3D注意力模块 

**Authors**: Narges Semiromizadeh, Omid Nejati Manzari, Shahriar B. Shokouhi, Sattar Mirzakuchaki  

**Link**: [PDF](https://arxiv.org/pdf/2502.15398)  

**Abstract**: Vehicle make and model recognition (VMMR) is a crucial component of the Intelligent Transport System, garnering significant attention in recent years. VMMR has been widely utilized for detecting suspicious vehicles, monitoring urban traffic, and autonomous driving systems. The complexity of VMMR arises from the subtle visual distinctions among vehicle models and the wide variety of classes produced by manufacturers. Convolutional Neural Networks (CNNs), a prominent type of deep learning model, have been extensively employed in various computer vision tasks, including VMMR, yielding remarkable results. As VMMR is a fine-grained classification problem, it primarily faces inter-class similarity and intra-class variation challenges. In this study, we implement an attention module to address these challenges and enhance the model's focus on critical areas containing distinguishing features. This module, which does not increase the parameters of the original model, generates three-dimensional (3-D) attention weights to refine the feature map. Our proposed model integrates the attention module into two different locations within the middle section of a convolutional model, where the feature maps from these sections offer sufficient information about the input frames without being overly detailed or overly coarse. The performance of our proposed model, along with state-of-the-art (SOTA) convolutional and transformer-based models, was evaluated using the Stanford Cars dataset. Our proposed model achieved the highest accuracy, 90.69\%, among the compared models. 

**Abstract (ZH)**: 车辆品牌和型号识别（VMMR）是智能交通系统的一个关键组成部分，近年来引起了广泛关注。VMMR 广泛应用于检测可疑车辆、监控城市交通和自动驾驶系统。由于车辆型号之间的微妙视觉差异以及制造商产生的广泛类别的复杂性，VMMR 的复杂性较高。卷积神经网络（CNNs），一种主要的深度学习模型，在各种计算机视觉任务中得到了广泛应用，包括 VMMR，并取得了显著成果。由于 VMMR 是一个细粒度分类问题，它主要面临类内差异和类间相似性的挑战。在这项研究中，我们实现了一个注意力模块来应对这些挑战，并增强模型对包含区分特征的关键区域的关注。该模块不增加原始模型的参数，生成三维（3D）注意力权重以细化特征图。我们提出的模型将注意力模块整合到卷积模型中间部分的两个不同位置，这些部分的特征图提供了足够的输入帧信息，不过度详细或过于粗糙。使用斯坦福汽车数据集评估了我们提出的模型与最先进的（SOTA）卷积和变压器基模型的表现。我们提出的模型在对比模型中达到了最高的准确率，为 90.69%。 

---
# Super-Resolution for Interferometric Imaging: Model Comparisons and Performance Analysis 

**Title (ZH)**: 干涉成像的超分辨率技术：模型比较与性能分析 

**Authors**: Hasan Berkay Abdioglu, Rana Gursoy, Yagmur Isik, Ibrahim Cem Balci, Taha Unal, Kerem Bayer, Mustafa Ismail Inal, Nehir Serin, Muhammed Furkan Kosar, Gokhan Bora Esmer, Huseyin Uvet  

**Link**: [PDF](https://arxiv.org/pdf/2502.15397)  

**Abstract**: This study investigates the application of Super-Resolution techniques in holographic microscopy to enhance quantitative phase imaging. An off-axis Mach-Zehnder interferometric setup was employed to capture interferograms. The study evaluates two Super-Resolution models, RCAN and Real-ESRGAN, for their effectiveness in reconstructing high-resolution interferograms from a microparticle-based dataset. The models were assessed using two primary approaches: image-based analysis for structural detail enhancement and morphological evaluation for maintaining sample integrity and phase map accuracy. The results demonstrate that RCAN achieves superior numerical precision, making it ideal for applications requiring highly accurate phase map reconstruction, while Real-ESRGAN enhances visual quality and structural coherence, making it suitable for visualization-focused applications. This study highlights the potential of Super-Resolution models in overcoming diffraction-imposed resolution limitations in holographic microscopy, opening the way for improved imaging techniques in biomedical diagnostics, materials science, and other high-precision fields. 

**Abstract (ZH)**: 本研究探讨了超分辨率技术在全息显微镜中的应用，以增强定量相位成像。采用偏轴马赫-泽德干涉仪配置捕获干干图。研究评估了RCAN和Real-ESRGAN两种超分辨率模型在从基于微颗粒的数据集中重构高分辨率干干图方面的有效性。使用两种主要方法对模型进行评估：基于图像的分析以增强结构细节以及形态学评估以保持样本完整性和相位图准确性。结果表明，RCAN实现了更高的数值精度，使其适用于需要高准确度相位图重构的应用，而Real-ESRGAN则提高了视觉质量和结构连贯性，使其适用于侧重于可视化应用。本研究强调了超分辨率模型在克服全息显微镜中衍射限制分辨率方面的潜在能力，为生物医学诊断、材料科学和其他高精度领域提供了改进的成像技术。 

---
# Identifying Features that Shape Perceived Consciousness in Large Language Model-based AI: A Quantitative Study of Human Responses 

**Title (ZH)**: 基于大型语言模型的AI中影响感知意识特征的识别：人类响应的定量研究 

**Authors**: Kang Bongsu, Kim Jundong, Yun Tae-Rim, Bae Hyojin, Kim Chang-Eop  

**Link**: [PDF](https://arxiv.org/pdf/2502.15365)  

**Abstract**: This study quantitively examines which features of AI-generated text lead humans to perceive subjective consciousness in large language model (LLM)-based AI systems. Drawing on 99 passages from conversations with Claude 3 Opus and focusing on eight features -- metacognitive self-reflection, logical reasoning, empathy, emotionality, knowledge, fluency, unexpectedness, and subjective expressiveness -- we conducted a survey with 123 participants. Using regression and clustering analyses, we investigated how these features influence participants' perceptions of AI consciousness. The results reveal that metacognitive self-reflection and the AI's expression of its own emotions significantly increased perceived consciousness, while a heavy emphasis on knowledge reduced it. Participants clustered into seven subgroups, each showing distinct feature-weighting patterns. Additionally, higher prior knowledge of LLMs and more frequent usage of LLM-based chatbots were associated with greater overall likelihood assessments of AI consciousness. This study underscores the multidimensional and individualized nature of perceived AI consciousness and provides a foundation for better understanding the psychosocial implications of human-AI interaction. 

**Abstract (ZH)**: 本研究定量分析了哪些特征使人类在基于大规模语言模型（LLM）的AI系统中感知到主观意识。通过聚焦于元认知自我反省、逻辑推理、同理心、情感性、知识、流畅性、意外性以及主观表达性等八种特征，基于与Claude 3 Opus对话的99段文本，对123名参与者进行了调查。通过回归和聚类分析，我们探讨了这些特征如何影响参与者对AI意识的感知。研究结果表明，元认知自我反省和AI表达自身情感显著增加了感知到的意识，而过度强调知识则减少了这一感知。参与者被归类为七个子群，每个子群都有自己独特的特征权重模式。此外，先前对LLM的知识更多以及更频繁使用基于LLM的聊天机器人与更高的总体意识可能性评估相关。本研究强调了感知到的AI意识的多维度和个性化性质，并为更好地理解人机交互的心理社会影响提供了基础。 

---
# Evaluating Social Biases in LLM Reasoning 

**Title (ZH)**: 评估LLM推理中的社会偏见 

**Authors**: Xuyang Wu, Jinming Nian, Zhiqiang Tao, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15361)  

**Abstract**: In the recent development of AI reasoning, large language models (LLMs) are trained to automatically generate chain-of-thought reasoning steps, which have demonstrated compelling performance on math and coding tasks. However, when bias is mixed within the reasoning process to form strong logical arguments, it could cause even more harmful results and further induce hallucinations. In this paper, we have evaluated the 8B and 32B variants of DeepSeek-R1 against their instruction tuned counterparts on the BBQ dataset, and investigated the bias that is elicited out and being amplified through reasoning steps. To the best of our knowledge, this empirical study is the first to assess bias issues in LLM reasoning. 

**Abstract (ZH)**: 近年来，在AI推理的发展中，大型语言模型（LLMs）被训练自动生成链式推理步骤，已在数学和编程任务上展示了令人信服的性能。然而，当偏见渗入推理过程并形成强有力的逻辑论证时，它可能会导致更加有害的结果，进一步引发幻觉。在本文中，我们评估了DeepSeek-R1的8B和32B变体与其指令调优版本在BBQ数据集上的表现，并调查了通过推理步骤引发和放大的偏见。据我们所知，这项实证研究是首次评估LLM推理中的偏见问题。 

---
# Integrating Generative AI in Cybersecurity Education: Case Study Insights on Pedagogical Strategies, Critical Thinking, and Responsible AI Use 

**Title (ZH)**: 将生成式AI整合到网络安全教育中：基于教学策略、批判性思维和负责任AI使用案例研究的见解 

**Authors**: Mahmoud Elkhodr, Ergun Gide  

**Link**: [PDF](https://arxiv.org/pdf/2502.15357)  

**Abstract**: The rapid advancement of Generative Artificial Intelligence (GenAI) has introduced new opportunities for transforming higher education, particularly in fields that require analytical reasoning and regulatory compliance, such as cybersecurity management. This study presents a structured framework for integrating GenAI tools into cybersecurity education, demonstrating their role in fostering critical thinking, real-world problem-solving, and regulatory awareness. The implementation strategy followed a two-stage approach, embedding GenAI within tutorial exercises and assessment tasks. Tutorials enabled students to generate, critique, and refine AI-assisted cybersecurity policies, while assessments required them to apply AI-generated outputs to real-world scenarios, ensuring alignment with industry standards and regulatory requirements. Findings indicate that AI-assisted learning significantly enhanced students' ability to evaluate security policies, refine risk assessments, and bridge theoretical knowledge with practical application. Student reflections and instructor observations revealed improvements in analytical engagement, yet challenges emerged regarding AI over-reliance, variability in AI literacy, and the contextual limitations of AI-generated content. Through structured intervention and research-driven refinement, students were able to recognize AI strengths as a generative tool while acknowledging its need for human oversight. This study further highlights the broader implications of AI adoption in cybersecurity education, emphasizing the necessity of balancing automation with expert judgment to cultivate industry-ready professionals. Future research should explore the long-term impact of AI-driven learning on cybersecurity competency, as well as the potential for adaptive AI-assisted assessments to further personalize and enhance educational outcomes. 

**Abstract (ZH)**: 生成式人工智能的快速进步为高等教育带来了新的转型机会，特别是在需要分析推理和合规管理的领域，如网络安全管理。本研究提出了一种结构化框架，用于将生成式人工智能工具集成到网络安全教育中，展示了其在培养批判性思维、解决实际问题和增强合规意识方面的作用。实施策略遵循两阶段方法，将生成式人工智能嵌入教程练习和评估任务中。教程让学生能够生成、批判和改进AI辅助的网络安全政策，而评估则要求学生将AI生成的输出应用于实际场景，确保与行业标准和合规要求一致。研究结果表明，AI辅助学习显著增强了学生评估安全政策、改进风险评估和将理论知识与实践应用相结合的能力。学生反思和教师观察显示，在分析参与方面有所改进，但同时出现了对AI过度依赖、AI素养的差异性和AI生成内容的上下文限制等方面的挑战。通过结构化干预和基于研究的改进，学生能够认识到AI作为生成工具的优势，并认识到其需要人类监督。本研究还强调了在网络安全教育中采用AI的更广泛影响，强调需平衡自动化与专家判断以培养行业所需的专业人士。未来研究应探讨AI驱动学习对网络安全技能的长期影响，以及适应性AI辅助评估的潜力以进一步个性化和提升教育成果。 

---
# Constructing a Norm for Children's Scientific Drawing: Distribution Features Based on Semantic Similarity of Large Language Models 

**Title (ZH)**: 基于大语言模型语义相似性的儿童科学绘画规范构建：分布特征 

**Authors**: Yi Zhang, Fan Wei, Jingyi Li, Yan Wang, Yanyan Yu, Jianli Chen, Zipo Cai, Xinyu Liu, Wei Wang, Peng Wang, Zhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15348)  

**Abstract**: The use of children's drawings to examining their conceptual understanding has been proven to be an effective method, but there are two major problems with previous research: 1. The content of the drawings heavily relies on the task, and the ecological validity of the conclusions is low; 2. The interpretation of drawings relies too much on the subjective feelings of the researchers. To address this issue, this study uses the Large Language Model (LLM) to identify 1420 children's scientific drawings (covering 9 scientific themes/concepts), and uses the word2vec algorithm to calculate their semantic similarity. The study explores whether there are consistent drawing representations for children on the same theme, and attempts to establish a norm for children's scientific drawings, providing a baseline reference for follow-up children's drawing research. The results show that the representation of most drawings has consistency, manifested as most semantic similarity greater than 0.8. At the same time, it was found that the consistency of the representation is independent of the accuracy (of LLM's recognition), indicating the existence of consistency bias. In the subsequent exploration of influencing factors, we used Kendall rank correlation coefficient to investigate the effects of Sample Size, Abstract Degree, and Focus Points on drawings, and used word frequency statistics to explore whether children represented abstract themes/concepts by reproducing what was taught in class. 

**Abstract (ZH)**: 使用大型语言模型识别儿童科学绘画及其语义相似性分析：探索同一主题下儿童绘画的一致性并建立儿童科学绘画规范 

---
# Exploring Embodied Multimodal Large Models: Development, Datasets, and Future Directions 

**Title (ZH)**: 探索具身多模态大型模型：发展、数据集及未来方向 

**Authors**: Shoubin Chen, Zehao Wu, Kai Zhang, Chunyu Li, Baiyang Zhang, Fei Ma, Fei Richard Yu, Qingquan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.15336)  

**Abstract**: Embodied multimodal large models (EMLMs) have gained significant attention in recent years due to their potential to bridge the gap between perception, cognition, and action in complex, real-world environments. This comprehensive review explores the development of such models, including Large Language Models (LLMs), Large Vision Models (LVMs), and other models, while also examining other emerging architectures. We discuss the evolution of EMLMs, with a focus on embodied perception, navigation, interaction, and simulation. Furthermore, the review provides a detailed analysis of the datasets used for training and evaluating these models, highlighting the importance of diverse, high-quality data for effective learning. The paper also identifies key challenges faced by EMLMs, including issues of scalability, generalization, and real-time decision-making. Finally, we outline future directions, emphasizing the integration of multimodal sensing, reasoning, and action to advance the development of increasingly autonomous systems. By providing an in-depth analysis of state-of-the-art methods and identifying critical gaps, this paper aims to inspire future advancements in EMLMs and their applications across diverse domains. 

**Abstract (ZH)**: 沉浸式多模态大型模型（EMLMs）因潜在地弥合感知、认知和行动之间的差距而在复杂的真实环境中受到了广泛关注。本文综述了此类模型的发展，包括大型语言模型（LLMs）、大型视觉模型（LVMs）及其他模型，同时也探讨了其他新兴架构。本文讨论了EMLMs的发展演变，重点关注沉浸式感知、导航、交互和模拟。此外，综述还详细分析了用于训练和评估这些模型的数据集，突出了多元化和高质量数据对于有效学习的重要性。该论文还指出了EMLMs面临的几个关键挑战，包括可扩展性、泛化能力和实时决策问题。最后，本文提出了未来方向，强调了将多模态感知、推理和行动的整合以推进更具自主性的系统的发展。通过深入分析最先进的方法并识别关键缺口，本文旨在启发未来在EMLMs及其跨多个领域应用方面的进展。 

---
# Attention Eclipse: Manipulating Attention to Bypass LLM Safety-Alignment 

**Title (ZH)**: 注意力遮蔽：操纵注意力以绕过LLM安全对齐 

**Authors**: Pedram Zaree, Md Abdullah Al Mamun, Quazi Mishkatul Alam, Yue Dong, Ihsen Alouani, Nael Abu-Ghazaleh  

**Link**: [PDF](https://arxiv.org/pdf/2502.15334)  

**Abstract**: Recent research has shown that carefully crafted jailbreak inputs can induce large language models to produce harmful outputs, despite safety measures such as alignment. It is important to anticipate the range of potential Jailbreak attacks to guide effective defenses and accurate assessment of model safety. In this paper, we present a new approach for generating highly effective Jailbreak attacks that manipulate the attention of the model to selectively strengthen or weaken attention among different parts of the prompt. By harnessing attention loss, we develop more effective jailbreak attacks, that are also transferrable. The attacks amplify the success rate of existing Jailbreak algorithms including GCG, AutoDAN, and ReNeLLM, while lowering their generation cost (for example, the amplified GCG attack achieves 91.2% ASR, vs. 67.9% for the original attack on Llama2-7B/AdvBench, using less than a third of the generation time). 

**Abstract (ZH)**: 最近的研究表明，精心构造的越狱输入可以使对齐等安全措施的大语言模型生成有害输出。为了指导有效的防御和准确评估模型安全，有必要预测潜在的越狱攻击范围。本文提出了一种新的方法来生成高效的越狱攻击，该方法通过操控模型的注意力来选择性地增强或减弱提示不同部分的关注。通过利用注意力损失，我们开发了更有效的可转移的越狱攻击，这些攻击可以提高现有越狱算法（包括GCG、AutoDAN和ReNeLLM）的成功率，同时降低其生成成本（例如，放大后的GCG攻击在Llama2-7B/AdvBench上的成功率为91.2%，而原攻击的成功率为67.9%，且使用的时间不到原攻击的三分之一）。 

---
# Lightweight yet Efficient: An External Attentive Graph Convolutional Network with Positional Prompts for Sequential Recommendation 

**Title (ZH)**: 轻量高效：一种基于位置提示的外部注意力图卷积网络在序列推荐中的应用 

**Authors**: Jinyu Zhang, Chao Li, Zhongying Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.15331)  

**Abstract**: Graph-based Sequential Recommender systems (GSRs) have gained significant research attention due to their ability to simultaneously handle user-item interactions and sequential relationships between items. Current GSRs often utilize composite or in-depth structures for graph encoding (e.g., the Graph Transformer). Nevertheless, they have high computational complexity, hindering the deployment on resource-constrained edge devices. Moreover, the relative position encoding in Graph Transformer has difficulty in considering the complicated positional dependencies within sequence. To this end, we propose an External Attentive Graph convolutional network with Positional prompts for Sequential recommendation, namely EA-GPS. Specifically, we first introduce an external attentive graph convolutional network that linearly measures the global associations among nodes via two external memory units. Then, we present a positional prompt-based decoder that explicitly treats the absolute item positions as external prompts. By introducing length-adaptive sequential masking and a soft attention network, such a decoder facilitates the model to capture the long-term positional dependencies and contextual relationships within sequences. Extensive experimental results on five real-world datasets demonstrate that the proposed EA-GPS outperforms the state-of-the-art methods. Remarkably, it achieves the superior performance while maintaining a smaller parameter size and lower training overhead. The implementation of this work is publicly available at this https URL. 

**Abstract (ZH)**: 基于图的序贯推荐系统（GSRs）由于其同时处理用户-项交互和项之间序贯关系的能力而引起了广泛关注。当前的GSRs经常利用复合或深入的图编码结构（例如，图 transformer）。然而，它们具有较高的计算复杂性，阻碍了在资源受限的边缘设备上的部署。此外，图变压器中的相对位置编码难以考虑序列内的复杂位置依赖性。为此，我们提出了一种基于外部注意力图卷积网络和位置提示的序贯推荐方法，即EA-GPS。具体地，我们首先引入了一种外部注意力图卷积网络，通过两个外部记忆单元线性衡量节点间的全局关联。然后，我们提出了一个基于位置提示的解码器，明确将绝对项位置作为外部提示处理。通过引入长度自适应序贯掩码和软注意力网络，该解码器促进了模型捕捉序列内的长期位置依赖性和上下文关系。在五个真实世界的数据集上的广泛实验结果表明，所提出的EA-GPS在性能上优于现有最先进的方法。值得注意的是，它在保持较小的参数量和更低的训练开销的前提下实现了更好的性能。该工作的实现已公开发布在该网址。 

---
# SentiFormer: Metadata Enhanced Transformer for Image Sentiment Analysis 

**Title (ZH)**: 情感former：元数据增强的变换器在图像情感分析中的应用 

**Authors**: Bin Feng, Shulan Ruan, Mingzheng Yang, Dongxuan Han, Huijie Liu, Kai Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15322)  

**Abstract**: As more and more internet users post images online to express their daily emotions, image sentiment analysis has attracted increasing attention. Recently, researchers generally tend to design different neural networks to extract visual features from images for sentiment analysis. Despite the significant progress, metadata, the data (e.g., text descriptions and keyword tags) for describing the image, has not been sufficiently explored in this task. In this paper, we propose a novel Metadata Enhanced Transformer for sentiment analysis (SentiFormer) to fuse multiple metadata and the corresponding image into a unified framework. Specifically, we first obtain multiple metadata of the image and unify the representations of diverse data. To adaptively learn the appropriate weights for each metadata, we then design an adaptive relevance learning module to highlight more effective information while suppressing weaker ones. Moreover, we further develop a cross-modal fusion module to fuse the adaptively learned representations and make the final prediction. Extensive experiments on three publicly available datasets demonstrate the superiority and rationality of our proposed method. 

**Abstract (ZH)**: 基于元数据增强的变压器在图像情感分析中的应用（SentiFormer） 

---
# Road Traffic Sign Recognition method using Siamese network Combining Efficient-CNN based Encoder 

**Title (ZH)**: 基于Efficient-CNN编码器的Siamese网络道路交通标志识别方法 

**Authors**: Zhenghao Xi, Yuchao Shao, Yang Zheng, Xiang Liu, Yaqi Liu, Yitong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2502.15307)  

**Abstract**: Traffic signs recognition (TSR) plays an essential role in assistant driving and intelligent transportation system. However, the noise of complex environment may lead to motion-blur or occlusion problems, which raise the tough challenge to real-time recognition with high accuracy and robust. In this article, we propose IECES-network which with improved encoders and Siamese net. The three-stage approach of our method includes Efficient-CNN based encoders, Siamese backbone and the fully-connected layers. We firstly use convolutional encoders to extract and encode the traffic sign features of augmented training samples and standard images. Then, we design the Siamese neural network with Efficient-CNN based encoder and contrastive loss function, which can be trained to improve the robustness of TSR problem when facing the samples of motion-blur and occlusion by computing the distance between inputs and templates. Additionally, the template branch of the proposed network can be stopped when executing the recognition tasks after training to raise the process speed of our real-time model, and alleviate the computational resource and parameter scale. Finally, we recombined the feature code and a fully-connected layer with SoftMax function to classify the codes of samples and recognize the category of traffic signs. The results of experiments on the Tsinghua-Tencent 100K dataset and the German Traffic Sign Recognition Benchmark dataset demonstrate the performance of the proposed IECESnetwork. Compared with other state-of-the-art methods, in the case of motion-blur and occluded environment, the proposed method achieves competitive performance precision-recall and accuracy metric average is 88.1%, 86.43% and 86.1% with a 2.9M lightweight scale, respectively. Moreover, processing time of our model is 0.1s per frame, of which the speed is increased by 1.5 times compared with existing methods. 

**Abstract (ZH)**: 交通标志识别（TSR）在辅助驾驶和智能交通系统中起着重要作用。然而，复杂环境中的噪声可能导致运动模糊或遮挡问题，这为实时高精度和鲁棒性的识别带来了严峻挑战。本文提出了一种改进的IECES网络，采用改进的编码器和Siamese网络。我们的方法采用三阶段的方法，包括基于Efficient-CNN的编码器、Siamese骨干以及全连接层。首先，我们使用卷积编码器提取并编码增强训练样本和标准图像的交通标志特征。然后，设计基于Efficient-CNN的Siamese神经网络和对比损失函数，通过计算输入与模板之间的距离来提高在运动模糊和遮挡样本面前TSR问题的鲁棒性。此外，在训练结束后，提出的网络的模板分支可以在执行识别任务时停止，从而提高我们实时模型的处理速度，减少计算资源和参数规模。最后，将特征码与带有SoftMax函数的全连接层重新组合，以对样本的类别进行分类和交通标志的识别。实验结果表明，提出的IECES网络在Tsinghua-Tencent 100K数据集和German Traffic Sign Recognition Benchmark数据集上的性能。在运动模糊和遮挡环境中，与其它先进方法相比，所提方法在精度召回率和准确率平均指标分别为88.1%、86.43%和86.1%，且具有2.9M的轻量级规模。此外，我们的模型每帧处理时间为0.1秒，速度比现有方法提高了1.5倍。 

---
# SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention 

**Title (ZH)**: SVDq：1.25比特和410倍键缓存压缩的大型语言模型注意力机制 

**Authors**: Hong Yankun, Li Xing, Zhen Hui-Ling, Yu Xianzhi, Liu Wulong, Yuan Mingxuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15304)  

**Abstract**: For the efficient inference of Large Language Models (LLMs), the effective compression of key-value (KV) cache is essential. Three main types of KV cache compression techniques, namely sparsity, channel compression, and quantization, have been identified. This study presents SVDq, a Singular Value Decomposition (SVD) - based mixed precision quantization method for K cache. Initially, K cache is transformed into latent channels using SVD basis representations. Since the values in latent channels decay rapidly and become negligible after only a few latent channels, our method then incorporates importance-aware quantization and compression for latent channels. This enables the effective allocation of higher precision to more significant channels. Theoretically, we prove that SVDq results in quantization errors (x0.1 or even lower) that are much lower than those of per-channel key quantization in the original space. Our findings based on RULER and LongBench benchmarks demonstrate that SVDq can achieve an equivalent key cache precision as low as 1.25-bit. When combined with key sparsity, it can reach a key compression ratio of up to 410x for attention computation, all while maintaining comparable model performance. Notably, our method is nearly lossless for LongBench datasets. This indicates that SVDq enables high-precision low-bit quantization, providing a more efficient solution for KV cache compression in LLMs. 

**Abstract (ZH)**: 基于SVD的混合精度量化方法SVDq在大型语言模型中有效压缩键值缓存 

---
# Beyond Fixed Variables: Expanding-variate Time Series Forecasting via Flat Scheme and Spatio-temporal Focal Learning 

**Title (ZH)**: 超越固定变量：基于扁平方案和时空焦点学习的可变变量时间序列预测 

**Authors**: Minbo Ma, Kai Tang, Huan Li, Fei Teng, Dalin Zhang, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.15296)  

**Abstract**: Multivariate Time Series Forecasting (MTSF) has long been a key research focus. Traditionally, these studies assume a fixed number of variables, but in real-world applications, Cyber-Physical Systems often expand as new sensors are deployed, increasing variables in MTSF. In light of this, we introduce a novel task, Expanding-variate Time Series Forecasting (EVTSF). This task presents unique challenges, specifically (1) handling inconsistent data shapes caused by adding new variables, and (2) addressing imbalanced spatio-temporal learning, where expanding variables have limited observed data due to the necessity for timely operation. To address these challenges, we propose STEV, a flexible spatio-temporal forecasting framework. STEV includes a new Flat Scheme to tackle the inconsistent data shape issue, which extends the graph-based spatio-temporal modeling architecture into 1D space by flattening the 2D samples along the variable dimension, making the model variable-scale-agnostic while still preserving dynamic spatial correlations through a holistic graph. We introduce a novel Spatio-temporal Focal Learning strategy that incorporates a negative filter to resolve potential conflicts between contrastive learning and graph representation, and a focal contrastive loss as its core to guide the framework to focus on optimizing the expanding variables. We benchmark EVTSF performance using three real-world datasets and compare it against three potential solutions employing SOTA MTSF models tailored for EVSTF. Experimental results show that STEV significantly outperforms its competitors, particularly on expanding variables. Notably, STEV, with only 5% of observations from the expanding period, is on par with SOTA MTSF models trained with complete observations. Further exploration of various expanding strategies underscores the generalizability of STEV in real-world applications. 

**Abstract (ZH)**: 扩展变量时间序列预报（EVTSF） 

---
# Round Attention: A Novel Round-Level Attention Mechanism to Accelerate LLM Inference 

**Title (ZH)**: 圆级attention：一种加速大规模语言模型推理的新型圆级注意力机制 

**Authors**: Yaohua Tang, Zhicheng Hu, Kun Cheng, Fan Mo, Qiheng Lv, Hua Wang, Zhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15294)  

**Abstract**: The increasing context window size in large language models (LLMs) has improved their ability to handle complex, long-text tasks. However, as the conversation rounds continue, it is required to store a large amount of KV cache in GPU memory, which significantly affects the efficiency and even availability of the model serving systems. This paper analyzes dialogue data from real users and discovers that the LLM inference manifests a watershed layer, after which the distribution of round-level attention shows notable similarity. We propose Round Attention, a novel round-level attention mechanism that only recalls and computes the KV cache of the most relevant rounds. The experiments show that our method saves 55\% memory usage without compromising model performance. 

**Abstract (ZH)**: 大型语言模型中上下文窗口大小的增加提高了其处理复杂长文本任务的能力，但随着对话轮次的增加，需要在GPU内存中存储大量KV缓存，这显著影响了模型服务系统的效率甚至可用性。本文分析了真实用户的数据，发现LLM推理存在一个分水岭层，在此之后，轮次级别的注意力分布显示出明显的相似性。我们提出了一种新型的轮次级别注意力机制Round Attention，仅召回并计算最相关的轮次的KV缓存。实验结果显示，该方法在不牺牲模型性能的情况下节省了55%的内存使用。 

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
# Comparative Analysis of Large Language Models for Context-Aware Code Completion using SAFIM Framework 

**Title (ZH)**: 基于SAFIM框架的大规模语言模型在上下文感知代码补全方面的比较分析 

**Authors**: Hang Zhang, Yanxin Shen, Lun Wang, Chuanqi Shi, Shaoshuai Du, Yiyi Tao, Yixian Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15243)  

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized code completion, transforming it into a more intelligent and context-aware feature in modern integrated development environments. These advancements have significantly enhanced developers' ability to write efficient and error-free code. This study evaluates the performance of several chat-based LLMs, including Gemini 1.5 Flash, Gemini 1.5 Pro, GPT-4o, GPT-4o-mini, and GPT-4 Turbo, using the Syntax-Aware Fill-in-the-Middle (SAFIM) dataset. This benchmark is specifically designed to assess models' capabilities in syntax-sensitive code generation. Performance metrics, such as cosine similarity with ground-truth completions and latency, were employed to measure both accuracy and efficiency. The findings reveal substantial differences in the models' code completion abilities, offering valuable insights into their respective strengths and weaknesses. This work provides a comparative analysis that underscores the trade-offs between accuracy and speed, establishing a benchmark for future advancements in LLM-based code completion. 

**Abstract (ZH)**: 大型语言模型的出现已revolutionized代码完成，将其转变为现代集成开发环境中的更加智能和上下文感知的功能。这些进步显著增强了开发者编写高效无误代码的能力。本研究使用Syntax-Aware Fill-in-the-Middle (SAFIM) 数据集评估了几种基于聊天的大型语言模型，包括Gemini 1.5 Flash、Gemini 1.5 Pro、GPT-4o、GPT-4o-mini和GPT-4 Turbo的表现。该基准特别设计用于评估模型在语法敏感代码生成方面的能力。通过使用与真实完成结果的余弦相似度和延迟等性能指标来衡量准确性和效率。研究发现揭示了模型之间在代码完成能力上的显著差异，提供了它们各自优势和不足的宝贵见解。本研究提供了比较分析，强调了准确性和速度之间的权衡，并建立了基于大型语言模型的代码完成未来进步的基准。 

---
# AutoMR: A Universal Time Series Motion Recognition Pipeline 

**Title (ZH)**: 自动MR：一种通用的时间序列运动识别流水线 

**Authors**: Likun Zhang, Sicheng Yang, Zhuo Wang, Haining Liang, Junxiao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15228)  

**Abstract**: In this paper, we present an end-to-end automated motion recognition (AutoMR) pipeline designed for multimodal datasets. The proposed framework seamlessly integrates data preprocessing, model training, hyperparameter tuning, and evaluation, enabling robust performance across diverse scenarios. Our approach addresses two primary challenges: 1) variability in sensor data formats and parameters across datasets, which traditionally requires task-specific machine learning implementations, and 2) the complexity and time consumption of hyperparameter tuning for optimal model performance. Our library features an all-in-one solution incorporating QuartzNet as the core model, automated hyperparameter tuning, and comprehensive metrics tracking. Extensive experiments demonstrate its effectiveness on 10 diverse datasets, achieving state-of-the-art performance. This work lays a solid foundation for deploying motion-capture solutions across varied real-world applications. 

**Abstract (ZH)**: 本文提出了一种端到端的自动运动识别（AutoMR）流水线，适用于多模态数据集。所提出的框架无缝集成数据预处理、模型训练、超参数调整和评估，能够在多种场景下实现稳健性能。我们的方法解决两大主要挑战：1) 数据集间传感器数据格式和参数的差异性，传统上需要针对特定任务的机器学习实现；2) 为实现最佳模型性能而进行的超参数调整的复杂性和耗时。我们的库包括一个一站式解决方案，以QuartzNet为核心模型，自动超参数调整和全面的性能指标跟踪。广泛实验展示了其在10个不同数据集上的有效性，达到最佳性能。本文为在各种实际应用中部署运动捕捉解决方案奠定了坚实基础。 

---
# Understand User Opinions of Large Language Models via LLM-Powered In-the-Moment User Experience Interviews 

**Title (ZH)**: 通过大语言模型驱动的即时用户体验访谈理解用户对大型语言模型的意见 

**Authors**: Mengqiao Liu, Tevin Wang, Cassandra A. Cohen, Sarah Li, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.15226)  

**Abstract**: Which large language model (LLM) is better? Every evaluation tells a story, but what do users really think about current LLMs? This paper presents CLUE, an LLM-powered interviewer that conducts in-the-moment user experience interviews, right after users interacted with LLMs, and automatically gathers insights about user opinions from massive interview logs. We conduct a study with thousands of users to understand user opinions on mainstream LLMs, recruiting users to first chat with a target LLM and then interviewed by CLUE. Our experiments demonstrate that CLUE captures interesting user opinions, for example, the bipolar views on the displayed reasoning process of DeepSeek-R1 and demands for information freshness and multi-modality. Our collected chat-and-interview logs will be released. 

**Abstract (ZH)**: 哪种大型语言模型（LLM）更好？每个评估都有它的故事，但用户真实如何看待当前的LLM呢？本文介绍了一种基于LLM的访谈者CLUE，它在用户与LLM交互后即时进行用户体验访谈，并自动从大量的访谈日志中收集用户意见的见解。我们进行了一项研究，有数千名用户参与，以了解他们对主流LLM的看法，招募用户首先与目标LLM聊天，然后接受CLUE的访谈。实验证明，CLUE捕捉到了有趣用户意见的例子，如对DeepSeek-R1显示推理过程的两极看法以及对信息新鲜度和多模态的诉求。我们收集的聊天和访谈日志将公开发布。 

---
# Auto-Bench: An Automated Benchmark for Scientific Discovery in LLMs 

**Title (ZH)**: Auto-Bench: 一个用于LLM科学研究的自动化基准测试 

**Authors**: Tingting Chen, Srinivas Anumasa, Beibei Lin, Vedant Shah, Anirudh Goyal, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15224)  

**Abstract**: Given the remarkable performance of Large Language Models (LLMs), an important question arises: Can LLMs conduct human-like scientific research and discover new knowledge, and act as an AI scientist? Scientific discovery is an iterative process that demands efficient knowledge updating and encoding. It involves understanding the environment, identifying new hypotheses, and reasoning about actions; however, no standardized benchmark specifically designed for scientific discovery exists for LLM agents. In response to these limitations, we introduce a novel benchmark, \textit{Auto-Bench}, that encompasses necessary aspects to evaluate LLMs for scientific discovery in both natural and social sciences. Our benchmark is based on the principles of causal graph discovery. It challenges models to uncover hidden structures and make optimal decisions, which includes generating valid justifications. By engaging interactively with an oracle, the models iteratively refine their understanding of underlying interactions, the chemistry and social interactions, through strategic interventions. We evaluate state-of-the-art LLMs, including GPT-4, Gemini, Qwen, Claude, and Llama, and observe a significant performance drop as the problem complexity increases, which suggests an important gap between machine and human intelligence that future development of LLMs need to take into consideration. 

**Abstract (ZH)**: 给定大型语言模型（LLMs）的出色表现，一个重要的问题出现了：LLMs能否像人类一样进行科学研究并发现新知识，扮演AI科学家的角色？科学发现是一个迭代过程，需要高效的知识更新和编码。它涉及到理解环境、提出新假设和推理行动；然而，针对科学发现没有专门设计的标准基准供LLM代理使用。为应对这些局限性，我们引入了一个新型基准Auto-Bench，该基准涵盖了评估LLMs在自然和社会科学中进行科学发现所需的关键方面。我们的基准基于因果图发现的原则。它挑战模型揭示隐藏结构并作出最优决策，包括生成有效的论证。通过与 oracle 交互，模型通过战略性干预逐步深化对其潜在交互、化学和社会互动的理解。我们评估了最先进的 LLMs，包括 GPT-4、Gemini、Qwen、Claude 和 Llama，并观察到随着问题复杂性的增加，性能出现显著下降，这表明机器智能与人类智能之间存在重要差距，未来 LLMs 的发展需要考虑这一差距。 

---
# FormalSpecCpp: A Dataset of C++ Formal Specifications created using LLMs 

**Title (ZH)**: FormalSpecCpp: 一个由LLM创建的C++形式化规范数据集 

**Authors**: Madhurima Chakraborty, Peter Pirkelbauer, Qing Yi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15217)  

**Abstract**: FormalSpecCpp is a dataset designed to fill the gap in standardized benchmarks for verifying formal specifications in C++ programs. To the best of our knowledge, this is the first comprehensive collection of C++ programs with well-defined preconditions and postconditions. It provides a structured benchmark for evaluating specification inference tools and testing theaccuracy of generated specifications. Researchers and developers can use this dataset to benchmark specification inference tools,fine-tune Large Language Models (LLMs) for automated specification generation, and analyze the role of formal specifications in improving program verification and automated testing. By making this dataset publicly available, we aim to advance research in program verification, specification inference, and AI-assisted software development. The dataset and the code are available at this https URL. 

**Abstract (ZH)**: FormalSpecCpp是用于验证C++程序正式规范的标准基准缺口的数据集，到我们所知，这是第一个包含明确预条件和后条件的C++程序的全面集合。它提供了一个结构化的基准，用于评估规范推断工具并测试生成规范的准确性。研究人员和开发人员可以使用此数据集来基准测试规范推断工具、微调大型语言模型（LLMs）以实现自动化规范生成，并分析正式规范在提高程序验证和自动化测试中的作用。通过公开发布此数据集，我们旨在促进程序验证、规范推断和AI辅助软件开发的研究。数据集和代码可在以下网址获取：this https URL。 

---
# The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning 

**Title (ZH)**: LLM-和VLM-集成强化学习的发展格局 

**Authors**: Sheila Schoepp, Masoud Jafaripour, Yingyue Cao, Tianpei Yang, Fatemeh Abdollahi, Shadan Golestan, Zahin Sufiyan, Osmar R. Zaiane, Matthew E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2502.15214)  

**Abstract**: Reinforcement learning (RL) has shown impressive results in sequential decision-making tasks. Meanwhile, Large Language Models (LLMs) and Vision-Language Models (VLMs) have emerged, exhibiting impressive capabilities in multimodal understanding and reasoning. These advances have led to a surge of research integrating LLMs and VLMs into RL. In this survey, we review representative works in which LLMs and VLMs are used to overcome key challenges in RL, such as lack of prior knowledge, long-horizon planning, and reward design. We present a taxonomy that categorizes these LLM/VLM-assisted RL approaches into three roles: agent, planner, and reward. We conclude by exploring open problems, including grounding, bias mitigation, improved representations, and action advice. By consolidating existing research and identifying future directions, this survey establishes a framework for integrating LLMs and VLMs into RL, advancing approaches that unify natural language and visual understanding with sequential decision-making. 

**Abstract (ZH)**: 强化学习（RL）在序列决策任务中展现了令人印象深刻的成果。与此同时，大型语言模型（LLMs）和视觉-语言模型（VLMs）应运而生，并在多模态理解和推理方面表现出色。这些进展推动了将LLMs和VLMs集成到RL中的研究热潮。在这篇综述中，我们回顾了使用LLMs和VLMs解决RL中关键挑战的研究，如缺乏先验知识、长期规划和奖励设计。我们提出了一个分类框架，将这些LLM/VLM辅助的RL方法分为代理、规划者和奖励三个角色。最后，我们探讨了开放式问题，包括态势感知、偏见缓解、改进表示和动作建议。通过整合现有研究并确定未来方向，这篇综述建立了一个框架，以将LLMs和VLMs集成到RL中，推动将自然语言和视觉理解与序列决策相结合的方法的发展。 

---
# PairBench: A Systematic Framework for Selecting Reliable Judge VLMs 

**Title (ZH)**: PairBench: 一种选择可靠法官大语言模型的系统框架 

**Authors**: Aarash Feizi, Sai Rajeswar, Adriana Romero-Soriano, Reihaneh Rabbany, Spandana Gella, Valentina Zantedeschi, João Monteiro  

**Link**: [PDF](https://arxiv.org/pdf/2502.15210)  

**Abstract**: As large vision language models (VLMs) are increasingly used as automated evaluators, understanding their ability to effectively compare data pairs as instructed in the prompt becomes essential. To address this, we present PairBench, a low-cost framework that systematically evaluates VLMs as customizable similarity tools across various modalities and scenarios. Through PairBench, we introduce four metrics that represent key desiderata of similarity scores: alignment with human annotations, consistency for data pairs irrespective of their order, smoothness of similarity distributions, and controllability through prompting. Our analysis demonstrates that no model, whether closed- or open-source, is superior on all metrics; the optimal choice depends on an auto evaluator's desired behavior (e.g., a smooth vs. a sharp judge), highlighting risks of widespread adoption of VLMs as evaluators without thorough assessment. For instance, the majority of VLMs struggle with maintaining symmetric similarity scores regardless of order. Additionally, our results show that the performance of VLMs on the metrics in PairBench closely correlates with popular benchmarks, showcasing its predictive power in ranking models. 

**Abstract (ZH)**: 作为一种自动化评估工具，大型视觉语言模型(VLMs)越来越多地被用于数据对比，因此理解它们根据提示有效比较数据对的能力变得至关重要。为了解决这一问题，我们提出了PairBench，这是一种低成本框架，系统地评估VLMs作为可定制相似性工具在各种模态和场景下的性能。通过PairBench，我们引入了四个代表相似性评分关键要求的指标：与人类注释的一致性、数据对的顺序无关的一致性、相似性分布的平滑度以及通过提示实现的可控性。我们的分析表明，无论是闭源还是开源模型，在这些指标上都没有绝对的优势；最优选择取决于自动化评估器期望的行为（例如，平滑的评估者与尖锐的评估者），这突显了在广泛采用VLMs作为评估工具时进行全面评估的风险。例如，大多数VLMs在保持顺序无关的相似性得分方面存在困难。此外，我们的结果表明，PairBench中模型在各指标上的性能与流行的基准测试高度相关，展示了其在模型排序方面的预测能力。 

---
# FlipConcept: Tuning-Free Multi-Concept Personalization for Text-to-Image Generation 

**Title (ZH)**: FlipConcept: 无调优多概念文本到图像生成个性化 

**Authors**: Young Beom Woo, Sun Eung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.15203)  

**Abstract**: Recently, methods that integrate multiple personalized concepts into a single image have garnered significant attention in the field of text-to-image (T2I) generation. However, existing methods experience performance degradation in complex scenes with multiple objects due to distortions in non-personalized regions. To address this issue, we propose FlipConcept, a novel approach that seamlessly integrates multiple personalized concepts into a single image without requiring additional tuning. We introduce guided appearance attention to accurately mimic the appearance of a personalized concept as intended. Additionally, we introduce mask-guided noise mixing to protect non-personalized regions during editing. Lastly, we apply background dilution to minimize attribute leakage, which is the undesired blending of personalized concept attributes with other objects in the image. In our experiments, we demonstrate that the proposed method, despite not requiring tuning, outperforms existing models in both single and multiple personalized concept inference. 

**Abstract (ZH)**: 最近，将多个个性化概念整合到单张图像中的方法在文本到图像（T2I）生成领域引起了广泛关注。然而，现有的方法在包含多个对象的复杂场景中因非个性化区域的失真而性能下降。为了解决这一问题，我们提出了一种名为FlipConcept的新型方法，可以在不需要额外调整的情况下无缝整合多个个性化概念。我们引入了引导外观注意力，以准确模拟所期望的个性化概念的外观。此外，我们引入了掩码引导噪声混合，在编辑过程中保护非个性化区域。最后，我们应用背景稀释以最大限度地减少属性泄露，即个性化概念属性与图像中其他对象的不希望的混合。在我们的实验中，我们证明了所提出的方法即使不需要调整，在单个和多个个性化概念的推理中也优于现有模型。 

---
# TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding 

**Title (ZH)**: TETRIS: 批量推测解码的最优草稿令牌选择 

**Authors**: Zhaoxuan Wu, Zijian Zhou, Arun Verma, Alok Prakash, Daniela Rus, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2502.15197)  

**Abstract**: We propose TETRIS, a novel method that optimizes the total throughput of batch speculative decoding in multi-request settings. Unlike existing methods that optimize for a single request or a group of requests as a whole, TETRIS actively selects the most promising draft tokens (for every request in a batch) to be accepted when verified in parallel, resulting in fewer rejected tokens and hence less wasted computing resources. Such an effective resource utilization to achieve fast inference in large language models (LLMs) is especially important to service providers with limited inference capacity. Compared to baseline speculative decoding, TETRIS yields a consistently higher acceptance rate and more effective utilization of the limited inference capacity. We show theoretically and empirically that TETRIS outperforms baseline speculative decoding and existing methods that dynamically select draft tokens, leading to a more efficient batch inference in LLMs. 

**Abstract (ZH)**: TETRIS：一种在多请求环境中优化批推测解码总吞吐量的新方法 

---
# Scale-Free Graph-Language Models 

**Title (ZH)**: 无标度图-语言模型 

**Authors**: Jianglin Lu, Yixuan Liu, Yitian Zhang, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.15189)  

**Abstract**: Graph-language models (GLMs) have demonstrated great potential in graph-based semi-supervised learning. A typical GLM consists of two key stages: graph generation and text embedding, which are usually implemented by inferring a latent graph and finetuning a language model (LM), respectively. However, the former often relies on artificial assumptions about the underlying edge distribution, while the latter requires extensive data annotations. To tackle these challenges, this paper introduces a novel GLM that integrates graph generation and text embedding within a unified framework. Specifically, for graph generation, we leverage an inherent characteristic of real edge distribution--the scale-free property--as a structural prior. We unexpectedly find that this natural property can be effectively approximated by a simple k-nearest neighbor (KNN) graph. For text embedding, we develop a graph-based pseudo-labeler that utilizes scale-free graphs to provide complementary supervision for improved LM finetuning. Extensive experiments on representative datasets validate our findings on the scale-free structural approximation of KNN graphs and demonstrate the effectiveness of integrating graph generation and text embedding with a real structural prior. Our code is available at this https URL. 

**Abstract (ZH)**: 基于图的语言模型（GLMs）在图基于半监督学习中展现了巨大的潜力。一种典型的GLM通常包含两个关键阶段：图生成和文本嵌入，分别通过推断潜在图和微调语言模型（LM）实现。然而，前者往往依赖于对底层边分布的人工假设，而后者则需要大量的数据注释。为应对这些挑战，本文提出了一种新颖的GLM，在统一框架中整合了图生成和文本嵌入。具体地，在图生成阶段，我们利用真实边分布的一个内在特征——无标度特性——作为结构先验。我们意外地发现，这种自然特性可以通过简单的KNN图有效近似。在文本嵌入阶段，我们开发了一种基于图的伪标签器，利用无标度图为改进LM微调提供互补监督。代表性和广泛的实验验证了KNN图的无标度结构逼近，并演示了在实际结构先验下整合图生成和文本嵌入的有效性。相关代码可在以下链接获取：this https URL。 

---
# LUMINA-Net: Low-light Upgrade through Multi-stage Illumination and Noise Adaptation Network for Image Enhancement 

**Title (ZH)**: LUMINA-Net：多阶段 illumination 和噪声适应网络的低光照图像增强 

**Authors**: Namrah Siddiqua, Kim Suneung  

**Link**: [PDF](https://arxiv.org/pdf/2502.15186)  

**Abstract**: Low-light image enhancement (LLIE) is a crucial task in computer vision aimed to enhance the visual fidelity of images captured under low-illumination conditions. Conventional methods frequently struggle to mitigate pervasive shortcomings such as noise, over-exposure, and color distortion thereby precipitating a pronounced degradation in image quality. To address these challenges, we propose LUMINA-Net an advanced deep learning framework designed specifically by integrating multi-stage illumination and reflectance modules. First, the illumination module intelligently adjusts brightness and contrast levels while meticulously preserving intricate textural details. Second, the reflectance module incorporates a noise reduction mechanism that leverages spatial attention and channel-wise feature refinement to mitigate noise contamination. Through a comprehensive suite of experiments conducted on LOL and SICE datasets using PSNR, SSIM and LPIPS metrics, surpassing state-of-the-art methodologies and showcasing its efficacy in low-light image enhancement. 

**Abstract (ZH)**: 低光照图像增强（LLIE）是计算机视觉中一项关键任务，旨在提高在低光照条件下拍摄的图像的视觉保真度。传统方法常常难以缓解普遍存在的噪声、过度曝光和颜色失真等问题，从而导致图像质量显著下降。为应对这些挑战，我们提出LUMINA-Net一种先进的深度学习框架，通过集成多阶段光照和反射模块进行专门设计。首先，光照模块智能调整亮度和对比度，同时精心保留复杂的纹理细节。其次，反射模块引入了基于空间注意力机制和通道级特征精炼的噪声降低机制，以缓解噪声污染。通过在LOL和SICE数据集上进行综合实验，使用PSNR、SSIM和LPIPS指标超越现有先进方法，并展示了其在低光照图像增强中的有效性。 

---
# Key Body Posture Characteristics of Short-distance Speed Skaters at the Start Based on Artificial Intelligence 

**Title (ZH)**: 基于人工智能的短距离速滑运动员起跑关键身体姿态特征 

**Authors**: Zhang Xueliana, Fang Yingjieb, Liu Hang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15185)  

**Abstract**: Objective To conduct biomechanical analysis on the starting technique of male short-distance speed skating athletes in China and determine the key factors affecting the effectiveness of the starting movement. Methods 13 high-level male short-distance speed skating athletes were selected as the test subjects, and kinematic data were collected using an artificial intelligence video capture and analysis system. The body posture features and their effects on the starting movement performance were analyzed in the three stages of starting preparation, starting, and sprinting. Results The post-stability angle, anterior knee angle of the front leg, posterior knee angle of the rear leg, and stride length showed moderate to high positive correlations with the starting speed during the starting preparation stage. The trunk angle showed a high negative correlation with the starting speed. The trunk angle (TO4, TD4, TO6, TD6), hip angle (TO1, TO4, TO6), and knee angle (TD1) showed moderate to high negative correlations with the effectiveness of the starting movement during the starting and sprinting stages. The knee angle (TD2), ice-contact angle (TD2, TD4, TD5, TD6), and propulsion angle (TO1, TO4, TO7) showed moderate positive correlations with the effectiveness of the starting movement. Conclusion Stride length, left knee angle, and post-stability angle are the key factors affecting the starting speed. The larger the post-stability angle and left knee angle and the longer the stride length, the faster the starting speed. During the starting and sprinting stages, the smaller the ice-contact angle and propulsion angle, the greater the trunk angle and hip angle changes, the more effective the starting movement. 

**Abstract (ZH)**: 目标：对中国男子短距离速度滑冰运动员起滑技术的生物力学分析，确定影响起滑动作有效性的关键因素。方法：选取13名高水平男子短距离速度滑冰运动员作为测试对象，采用人工智能视频捕捉与分析系统收集动作数据，在起滑准备、起滑和冲刺三个阶段分析身体姿态特征及其对起滑动作表现的影响。结果：起滑准备阶段，后稳定性角、前腿膝角、后腿膝角和步长与起滑速度呈中到高度正相关，躯干角与起滑速度呈高度负相关。起滑和冲刺阶段，躯干角（TO4、TD4、TO6、TD6）、髋角（TO1、TO4、TO6）和膝角（TD1）与起滑动作有效性呈中到高度负相关，膝角（TD2）、冰接触角（TD2、TD4、TD5、TD6）和推动力角（TO1、TO4、TO7）与起滑动作有效性呈中到高度正相关。结论：步长、左侧膝角和后稳定性角是影响起滑速度的关键因素。后稳定性角和左侧膝角越大、步长越长，起滑速度越快。在起滑和冲刺阶段，冰接触角和推动力角越小，躯干角和髋角的变化越大，起滑动作越有效。 

---
# LEDD: Large Language Model-Empowered Data Discovery in Data Lakes 

**Title (ZH)**: LEDD：大型语言模型赋能的数据湖中数据发现 

**Authors**: Qi An, Chihua Ying, Yuqing Zhu, Yihao Xu, Manwei Zhang, Jianmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15182)  

**Abstract**: Data discovery in data lakes with ever increasing datasets has long been recognized as a big challenge in the realm of data management, especially for semantic search of and hierarchical global catalog generation of tables. While large language models (LLMs) facilitate the processing of data semantics, challenges remain in architecting an end-to-end system that comprehensively exploits LLMs for the two semantics-related tasks. In this demo, we propose LEDD, an end-to-end system with an extensible architecture that leverages LLMs to provide hierarchical global catalogs with semantic meanings and semantic table search for data lakes. Specifically, LEDD can return semantically related tables based on natural-language specification. These features make LEDD an ideal foundation for downstream tasks such as model training and schema linking for text-to-SQL tasks. LEDD also provides a simple Python interface to facilitate the extension and the replacement of data discovery algorithms. 

**Abstract (ZH)**: 数据湖中随不断增加的数据集进行数据发现长期以来被认为是数据管理领域的一个重大挑战，特别是在语义搜索和层次全球目录生成方面。尽管大型语言模型（LLMs）有助于处理数据语义，但在构建一个全面利用LLMs的端到端系统以解决两个语义相关任务方面仍面临挑战。在此次演示中，我们提出LEDD，这是一个具有可扩展架构的端到端系统，利用LLMs提供具有语义含义的层次全球目录和数据湖中的语义表搜索功能。具体来说，LEDD可以根据自然语言规范返回语义相关的表。这些功能使LEDD成为诸如文本到SQL任务的模型训练和模式链接之类的下游任务的理想基础。LEDD还提供了一个简单的Python接口，以方便扩展和替换数据发现算法。 

---
# Methods and Trends in Detecting Generated Images: A Comprehensive Review 

**Title (ZH)**: 生成图像检测的方法与趋势综述 

**Authors**: Arpan Mahara, Naphtali Rishe  

**Link**: [PDF](https://arxiv.org/pdf/2502.15176)  

**Abstract**: The proliferation of generative models, such as Generative Adversarial Networks (GANs), Diffusion Models, and Variational Autoencoders (VAEs), has enabled the synthesis of high-quality multimedia data. However, these advancements have also raised significant concerns regarding adversarial attacks, unethical usage, and societal harm. Recognizing these challenges, researchers have increasingly focused on developing methodologies to detect synthesized data effectively, aiming to mitigate potential risks. Prior reviews have primarily focused on deepfake detection and often lack coverage of recent advancements in synthetic image detection, particularly methods leveraging multimodal frameworks for improved forensic analysis. To address this gap, the present survey provides a comprehensive review of state-of-the-art methods for detecting and classifying synthetic images generated by advanced generative AI models. This review systematically examines core detection methodologies, identifies commonalities among approaches, and categorizes them into meaningful taxonomies. Furthermore, given the crucial role of large-scale datasets in this field, we present an overview of publicly available datasets that facilitate further research and benchmarking in synthetic data detection. 

**Abstract (ZH)**: 生成模型（如生成对抗网络GANs）、扩散模型和变分自编码器（VAEs）的 proliferations 使得高质量多媒体数据的合成成为可能，但同时也引发了关于对抗攻击、不道德使用和社会危害的重大担忧。认识到这些挑战，研究人员越来越多地关注开发有效的合成数据检测方法，旨在减轻潜在风险。以往的综述主要集中在虚假信息检测上，通常缺乏对合成图像检测的最新进展的覆盖，特别是利用多模态框架提升法医分析的方法。为了弥补这一差距，本综述提供了对先进生成AI模型生成的合成图像检测和分类方法的全面回顾，系统地考察了核心检测方法，明确了不同方法的共性，并按有意义的分类对其进行分类。此外，鉴于大规模数据集在此领域的关键作用，我们还介绍了促进合成数据检测进一步研究和基准测试的公有数据集概况。 

---
# Extreme Speech Classification in the Era of LLMs: Exploring Open-Source and Proprietary Models 

**Title (ZH)**: 在大语言模型时代的情感言词分类：开源与 proprietary 模型探索 

**Authors**: Sarthak Mahajan, Nimmi Rangaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2502.15155)  

**Abstract**: In recent years, widespread internet adoption and the growth in userbase of various social media platforms have led to an increase in the proliferation of extreme speech online. While traditional language models have demonstrated proficiency in distinguishing between neutral text and non-neutral text (i.e. extreme speech), categorizing the diverse types of extreme speech presents significant challenges. The task of extreme speech classification is particularly nuanced, as it requires a deep understanding of socio-cultural contexts to accurately interpret the intent of the language used by the speaker. Even human annotators often disagree on the appropriate classification of such content, emphasizing the complex and subjective nature of this task. The use of human moderators also presents a scaling issue, necessitating the need for automated systems for extreme speech classification. The recent launch of ChatGPT has drawn global attention to the potential applications of Large Language Models (LLMs) across a diverse variety of tasks. Trained on vast and diverse corpora, and demonstrating the ability to effectively capture and encode contextual information, LLMs emerge as highly promising tools for tackling this specific task of extreme speech classification. In this paper, we leverage the Indian subset of the extreme speech dataset from Maronikolakis et al. (2022) to develop an effective classification framework using LLMs. We evaluate open-source Llama models against closed-source OpenAI models, finding that while pre-trained LLMs show moderate efficacy, fine-tuning with domain-specific data significantly enhances performance, highlighting their adaptability to linguistic and contextual nuances. Although GPT-based models outperform Llama models in zero-shot settings, the performance gap disappears after fine-tuning. 

**Abstract (ZH)**: 近年来，广泛普及的互联网和各种社交媒体平台用户基数的快速增长导致了在线极端言论的增多。虽然传统语言模型在区分中性和非中性文本（即极端言论）方面表现出色，但对极端言论多样类型的分类仍具挑战性。极端言论分类的任务尤为复杂，因为它要求深刻理解社会文化背景，以准确解读演讲者的语言意图。即使是人类注释员也经常对这类内容的适当分类意见不一，强调了此任务的复杂性和主观性。使用人类审查员也存在扩展问题，因此亟需开发自动化的极端言论分类系统。ChatGPT的近期推出引起了人们对大型语言模型（LLMs）在各种任务中潜在应用的关注。经过大量多样语料库的训练，具备有效捕捉和编码上下文信息能力的LLMs成为应对特定极端言论分类任务的非常有前景的工具。本文利用Maronikolakis等人（2022）的极端言论数据集中的印度子集，结合LLMs开发了有效的分类框架。我们将开源的Llama模型与闭源的OpenAI模型进行对比评估，发现虽预训练的LLMs表现出一定的效果，但通过领域特定数据微调可以显著提高性能，突显了它们对语言和上下文细微差别的适应性。尽管基于GPT的模型在零样本设置中优于Llama模型，但在微调后性能差距消失。 

---
# Confidence-Weighted Boundary-Aware Learning for Semi-Supervised Semantic Segmentation 

**Title (ZH)**: 带有边界意识的自信心度加权半监督语义分割 

**Authors**: Ebenezer Tarubinga, Jenifer Kalafatovich Espinoza  

**Link**: [PDF](https://arxiv.org/pdf/2502.15152)  

**Abstract**: Semi-supervised semantic segmentation (SSSS) aims to improve segmentation performance by utilising unlabeled data alongside limited labeled samples. Existing SSSS methods often face challenges such as coupling, where over-reliance on initial labeled data leads to suboptimal learning; confirmation bias, where incorrect predictions reinforce themselves repeatedly; and boundary blur caused by insufficient boundary-awareness and ambiguous edge information. To address these issues, we propose CW-BASS, a novel framework for SSSS. In order to mitigate the impact of incorrect predictions, we assign confidence weights to pseudo-labels. Additionally, we leverage boundary-delineation techniques, which, despite being extensively explored in weakly-supervised semantic segmentation (WSSS) remain under-explored in SSSS. Specifically, our approach: (1) reduces coupling through a confidence-weighted loss function that adjusts the influence of pseudo-labels based on their predicted confidence scores, (2) mitigates confirmation bias with a dynamic thresholding mechanism that learns to filter out pseudo-labels based on model performance, (3) resolves boundary blur with a boundary-aware module that enhances segmentation accuracy near object boundaries, and (4) reduces label noise with a confidence decay strategy that progressively refines pseudo-labels during training. Extensive experiments on the Pascal VOC 2012 and Cityscapes demonstrate that our method achieves state-of-the-art performance. Moreover, using only 1/8 or 12.5\% of labeled data, our method achieves a mIoU of 75.81 on Pascal VOC 2012, highlighting its effectiveness in limited-label settings. 

**Abstract (ZH)**: 半监督语义分割（SSSS）旨在通过利用未标注数据和有限的标注样本来提高分割性能。为了解决现有SSSS方法中存在的耦合、确认偏差以及边界模糊等问题，我们提出了一种新的半监督语义分割框架CW-BASS。该方法通过赋予伪标签置信权重来减轻错误预测的影响，并利用边界定义技术，同时减少伪标签中的耦合、减轻确认偏差、解决边界模糊，并降低标签噪声。实验结果表明，该方法在Pascal VOC 2012和Cityscapes数据集上达到了最先进的性能，仅使用1/8或12.5%的标注数据即可实现Pascal VOC 2012上75.81的mIoU。 

---
# Projection Optimization: A General Framework for Multi-Objective and Multi-Group RLHF 

**Title (ZH)**: 投影优化：多目标与多组RLHF的通用框架 

**Authors**: Nuoya Xiong, Aarti Singh  

**Link**: [PDF](https://arxiv.org/pdf/2502.15145)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) is a widely used fine-tuning approach that aligns machine learning model, particularly Language Model (LM) with human preferences. There are typically multiple objectives driving the preference, hence humans find it easier to express per-objective comparisons rather than a global preference between two choices. %, e.g. compare two papers on their novelty, clarity, correctness, etc. Multi-Objective RLHF (MORLHF) aims to use per-objective preference feedback and achieve Pareto optimality among these objectives by aggregating them into a single unified objective for optimization. However, nearly all prior works rely on linear aggregation, which rules out policies that favor specific objectives such as the worst one. The only existing approach using non-linear aggregation is computationally expensive due to its reward-based nature and the need for retraining whenever the aggregation parameters change. In this work, we address this limitation by transforming the non-linear aggregation maximization problem into a series of sub-problems. Each sub-problem involves only linear aggregation, making it computationally efficient to solve. We further extend our framework to handle multi-group scenarios, where each group has distinct weights for the objectives. Our method enables achieving consensus or maximizing the aggregated objective across all groups. Theoretically, we demonstrate that our algorithmic framework achieves sublinear regret and can be easily adapted to a reward-free algorithm. Empirically, leveraging our theoretical insights, we propose a nearly training-free algorithm once the optimal policies for individual objectives are obtained. 

**Abstract (ZH)**: 基于人类反馈的强化学习与多目标优化（Multi-Objective Reinforcement Learning with Human Feedback, MORLHF） 

---
# Chain-of-Rank: Enhancing Large Language Models for Domain-Specific RAG in Edge Device 

**Title (ZH)**: 链排序：增强边缘设备上针对特定领域的语言模型的检索增强生成（RAG）能力 

**Authors**: Juntae Lee, Jihwan Bang, Seunghan Yang, Kyuhong Shim, Simyung Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15134)  

**Abstract**: Retrieval-augmented generation (RAG) with large language models (LLMs) is especially valuable in specialized domains, where precision is critical. To more specialize the LLMs into a target domain, domain-specific RAG has recently been developed by allowing the LLM to access the target domain early via finetuning. The domain-specific RAG makes more sense in resource-constrained environments like edge devices, as they should perform a specific task (e.g. personalization) reliably using only small-scale LLMs. While the domain-specific RAG is well-aligned with edge devices in this respect, it often relies on widely-used reasoning techniques like chain-of-thought (CoT). The reasoning step is useful to understand the given external knowledge, and yet it is computationally expensive and difficult for small-scale LLMs to learn it. Tackling this, we propose the Chain of Rank (CoR) which shifts the focus from intricate lengthy reasoning to simple ranking of the reliability of input external documents. Then, CoR reduces computational complexity while maintaining high accuracy, making it particularly suited for resource-constrained environments. We attain the state-of-the-art (SOTA) results in benchmarks, and analyze its efficacy. 

**Abstract (ZH)**: 基于大型语言模型的领域特定检索增强生成（RAG）在专业化领域尤其 valuable，精度至关重要。为了使大型语言模型更专门化于目标领域，最近通过微调允许其早期访问目标领域，开发了领域特定的RAG。在资源受限的环境如边缘设备中，领域特定的RAG更有意义，因为它们应该仅使用小型规模的大型语言模型可靠地完成特定任务（如个性化）。尽管在这一点上领域特定的RAG与边缘设备相契合，但它通常依赖于广泛使用的思想链（CoT）等推理技术。推理步骤有助于理解给定的外部知识，但小型规模的语言模型很难学习它。为解决此问题，我们提出了一种链排名（CoR）方法，将重点从复杂的长推理转移到输入外部文档可靠性的简单排名。然后，CoR降低了计算复杂性同时保持高精度，使其特别适用于资源受限的环境。我们在基准测试中取得了最先进（SOTA）的结果，并分析了其有效性。 

---
# CoT-ICL Lab: A Petri Dish for Studying Chain-of-Thought Learning from In-Context Demonstrations 

**Title (ZH)**: CoT-ICL 实验室：从情境演示中研究链式思考学习的实验平台 

**Authors**: Vignesh Kothapalli, Hamed Firooz, Maziar Sanjabi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15132)  

**Abstract**: We introduce CoT-ICL Lab, a framework and methodology to generate synthetic tokenized datasets and systematically study chain-of-thought (CoT) in-context learning (ICL) in language models. CoT-ICL Lab allows fine grained control over the complexity of in-context examples by decoupling (1) the causal structure involved in chain token generation from (2) the underlying token processing functions. We train decoder-only transformers (up to 700M parameters) on these datasets and show that CoT accelerates the accuracy transition to higher values across model sizes. In particular, we find that model depth is crucial for leveraging CoT with limited in-context examples, while more examples help shallow models match deeper model performance. Additionally, limiting the diversity of token processing functions throughout training improves causal structure learning via ICL. We also interpret these transitions by analyzing transformer embeddings and attention maps. Overall, CoT-ICL Lab serves as a simple yet powerful testbed for theoretical and empirical insights into ICL and CoT in language models. 

**Abstract (ZH)**: CoT-ICL Lab：生成合成标记数据集并系统研究语言模型中链式思考（CoT）上下文学习（ICL）的框架与方法 

---
# Unveiling Reasoning Thresholds in Language Models: Scaling, Fine-Tuning, and Interpretability through Attention Maps 

**Title (ZH)**: 探索语言模型中的推理门槛：通过注意力图揭示扩展、微调和可解释性 

**Authors**: Yen-Che Hsiao, Abhishek Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2502.15120)  

**Abstract**: This study investigates the in-context learning capabilities of various decoder-only transformer-based language models with different model sizes and training data, including GPT2, SmolLM2, OpenELM, TinyLlama, Stable LM, and Gemma 2. We identify a critical parameter threshold (~1.6 billion), beyond which reasoning performance improves significantly in tasks such as commonsense reasoning in multiple-choice question answering and deductive reasoning. Specifically, models above this threshold achieve better success rates in chain-of-thought (CoT) prompting for deductive reasoning tasks, especially those requiring longer reasoning chains, such as proof by contradiction and disjunction elimination. To address limitations in sub-threshold models, we demonstrate that fine-tuning with task-specific exemplars substantially enhances reasoning performance, enabling accurate CoT generation even without additional exemplars in the prompt for tasks with shorter reasoning chains. Finally, our analysis of attention maps reveals that models capable of generating correct CoTs exhibit higher token-level attention scores on subsequent correct tokens and the correct parts of speech, providing interpretability insights into reasoning processes. These findings collectively advance understanding of reasoning capabilities in decoder-only transformer-based models. The code is available at: this https URL. 

**Abstract (ZH)**: 本研究探讨了不同模型大小和训练数据的各类解码器基础Transformer语言模型的在上下文学习能力，包括GPT2、SmolLM2、OpenELM、TinyLlama、Stable LM和Gemma 2。我们发现一个关键参数阈值（约16亿），在此之上，推理性能在常识推理和演绎推理等任务中显著提升。具体而言，超过此阈值的模型在演绎推理任务中，尤其是在需要更长推理链的任务（如反证法和析取消去）中的链式思考（CoT）提示时，能实现更高的成功率。为了解决阈值下模型的限制，我们证明了使用特定任务示例进行微调可以显著提高推理性能，即使在较短推理链的任务提示中没有额外示例，也能实现准确的CoT生成。最后，我们对注意力图的分析表明，能够生成正确CoTs的模型在后续正确词语和正确词性上有更高的token级注意力分数，提供了推理过程的可解释性见解。这些发现共同推进了对解码器基础Transformer模型推理能力的理解。代码可供查阅：this https URL。 

---
# CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models 

**Title (ZH)**: CurricuVLM：通过个性化安全关键课程学习实现安全自主驾驶的视觉-语言模型方法 

**Authors**: Zihao Sheng, Zilin Huang, Yansong Qu, Yue Leng, Sruthi Bhavanam, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.15119)  

**Abstract**: Ensuring safety in autonomous driving systems remains a critical challenge, particularly in handling rare but potentially catastrophic safety-critical scenarios. While existing research has explored generating safety-critical scenarios for autonomous vehicle (AV) testing, there is limited work on effectively incorporating these scenarios into policy learning to enhance safety. Furthermore, developing training curricula that adapt to an AV's evolving behavioral patterns and performance bottlenecks remains largely unexplored. To address these challenges, we propose CurricuVLM, a novel framework that leverages Vision-Language Models (VLMs) to enable personalized curriculum learning for autonomous driving agents. Our approach uniquely exploits VLMs' multimodal understanding capabilities to analyze agent behavior, identify performance weaknesses, and dynamically generate tailored training scenarios for curriculum adaptation. Through comprehensive analysis of unsafe driving situations with narrative descriptions, CurricuVLM performs in-depth reasoning to evaluate the AV's capabilities and identify critical behavioral patterns. The framework then synthesizes customized training scenarios targeting these identified limitations, enabling effective and personalized curriculum learning. Extensive experiments on the Waymo Open Motion Dataset show that CurricuVLM outperforms state-of-the-art baselines across both regular and safety-critical scenarios, achieving superior performance in terms of navigation success, driving efficiency, and safety metrics. Further analysis reveals that CurricuVLM serves as a general approach that can be integrated with various RL algorithms to enhance autonomous driving systems. The code and demo video are available at: this https URL. 

**Abstract (ZH)**: 确保自主驾驶系统的安全性仍然是一个关键挑战，尤其是在处理罕见但可能灾难性的安全关键场景时。虽然现有研究已经探索了为自主车辆（AV）测试生成安全关键场景的方法，但在这些场景的有效融入以增强安全性方面的工作仍然有限。此外，开发能够适应自主车辆不断演化的行为模式和性能瓶颈的训练课程体系仍然有待探索。为应对这些挑战，我们提出了一种名为CurricuVLM的新颖框架，该框架利用视觉语言模型（VLM）为自主驾驶代理实现个性化课程学习。我们的方法独特地利用了VLM的多模态理解能力来分析代理行为、识别性能弱点，并动态生成定制化的训练场景以适应课程学习。通过对带有叙述描述的不安全驾驶情况进行全面分析，CurricuVLM深入推理以评估自主车辆的能力并识别关键行为模式。该框架随后综合生成针对这些识别出的限制的定制化训练场景，从而实现有效的个性化课程学习。在Waymo Open Motion数据集上的广泛实验表明，CurricuVLM在常规场景和安全关键场景中均优于最先进的基线方法，在导航成功率、驾驶效率和安全指标方面取得了优越性能。进一步的分析表明，CurricuVLM作为一种通用方法，可以与其他各种强化学习（RL）算法集成以增强自主驾驶系统。相关代码和演示视频可在以下链接获取：this https URL。 

---
# Assessing a Single Student's Concentration on Learning Platforms: A Machine Learning-Enhanced EEG-Based Framework 

**Title (ZH)**: 基于机器学习增强的EEG基础框架：评估单个学生在学习平台上的注意力 

**Authors**: Zewen Zhuo, Mohamad Najafi, Hazem Zein, Amine Nait-Ali  

**Link**: [PDF](https://arxiv.org/pdf/2502.15107)  

**Abstract**: This study introduces a specialized pipeline designed to classify the concentration state of an individual student during online learning sessions by training a custom-tailored machine learning model. Detailed protocols for acquiring and preprocessing EEG data are outlined, along with the extraction of fifty statistical features from five EEG signal bands: alpha, beta, theta, delta, and gamma. Following feature extraction, a thorough feature selection process was conducted to optimize the data inputs for a personalized analysis. The study also explores the benefits of hyperparameter fine-tuning to enhance the classification accuracy of the student's concentration state. EEG signals were captured from the student using a Muse headband (Gen 2), equipped with five electrodes (TP9, AF7, AF8, TP10, and a reference electrode NZ), during engagement with educational content on computer-based e-learning platforms. Employing a random forest model customized to the student's data, we achieved remarkable classification performance, with test accuracies of 97.6% in the computer-based learning setting and 98% in the virtual reality setting. These results underscore the effectiveness of our approach in delivering personalized insights into student concentration during online educational activities. 

**Abstract (ZH)**: 本研究 introduce 了一种专门的管道，用于通过训练定制的机器学习模型来分类个体学生在在线学习会话中的注意力状态。文中详细阐述了获取和预处理EEG数据的协议，以及从五个EEG信号带（alpha、beta、theta、delta和gamma）中提取五十个统计特征的方法。特征提取后，进行了彻底的特征选择过程，以优化个性化分析的数据输入。研究还探讨了超参数微调的好处，以提高学生注意力状态分类的准确性。EEG信号使用Muse头带（Gen 2）捕获，该头带配备了五个电极（TP9、AF7、AF8、TP10和参考电极NZ），学生在基于计算机的e学习平台上与教育内容互动时佩戴。使用定制于学生数据的随机森林模型，我们在基于计算机的学习环境中实现了令人瞩目的分类性能，测试准确率为97.6%，在虚拟现实环境中为98%。这些结果强调了我们方法在在线教育活动中提供个性化的学生注意力洞察方面的有效性。 

---
# Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans 

**Title (ZH)**: 分析神经元，而非嵌入：理解大规模语言模型表示与人类认知的对齐时机和位置 

**Authors**: Masha Fedzechkina, Eleonora Gualdoni, Sinead Williamson, Katherine Metcalf, Skyler Seto, Barry-John Theobald  

**Link**: [PDF](https://arxiv.org/pdf/2502.15090)  

**Abstract**: Modern large language models (LLMs) achieve impressive performance on some tasks, while exhibiting distinctly non-human-like behaviors on others. This raises the question of how well the LLM's learned representations align with human representations. In this work, we introduce a novel approach to the study of representation alignment: we adopt a method from research on activation steering to identify neurons responsible for specific concepts (e.g., 'cat') and then analyze the corresponding activation patterns. Our findings reveal that LLM representations closely align with human representations inferred from behavioral data. Notably, this alignment surpasses that of word embeddings, which have been center stage in prior work on human and model alignment. Additionally, our approach enables a more granular view of how LLMs represent concepts. Specifically, we show that LLMs organize concepts in a way that reflects hierarchical relationships interpretable to humans (e.g., 'animal'-'dog'). 

**Abstract (ZH)**: 现代大规模语言模型在一些任务上取得了 impressive 的性能，但在其他任务上表现出明显的非人类行为。这引发了 LLM 学习表示与人类表示之间对齐程度的问题。在本文中，我们引入了一种表示对齐研究的新方法：我们采用激活控制研究中的方法来识别负责特定概念（例如，“猫”）的神经元，然后分析相应的激活模式。我们的发现表明，LLM 的表示与从行为数据推断出的人类表示高度对齐。值得注意的是，这种对齐程度超过了之前工作中中心地位的词嵌入所达到的水平。此外，我们的方法还能够更细致地展现 LLM 如何表示概念。具体而言，我们显示 LLM 以反映可由人类理解的层次关系来组织概念（例如，“动物”-“狗”）。 

---
# UPCORE: Utility-Preserving Coreset Selection for Balanced Unlearning 

**Title (ZH)**: UPCORE: Utility-Preserving Coreset Selection for Balanced Unlearning 

**Authors**: Vaidehi Patil, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.15082)  

**Abstract**: User specifications or legal frameworks often require information to be removed from pretrained models, including large language models (LLMs). This requires deleting or "forgetting" a set of data points from an already-trained model, which typically degrades its performance on other data points. Thus, a balance must be struck between removing information and keeping the model's other abilities intact, with a failure to balance this trade-off leading to poor deletion or an unusable model. To this end, we propose UPCORE (Utility-Preserving Coreset Selection), a method-agnostic data selection framework for mitigating collateral damage during unlearning. Finding that the model damage is correlated with the variance of the model's representations on the forget set, we selectively prune the forget set to remove outliers, thereby minimizing model degradation after unlearning. We evaluate UPCORE across three standard unlearning methods consistently achieving a superior balance between the competing objectives of deletion efficacy and model preservation. To better evaluate this trade-off, we introduce a new metric, measuring the area-under-the-curve (AUC) across standard metrics. We find that UPCORE improves both standard metrics and AUC, benefitting from positive transfer between the coreset and pruned points while reducing negative transfer from the forget set to points outside of it. 

**Abstract (ZH)**: 基于保留用途的核心集选择（UPCORE）：一种在遗忘过程中缓解副损伤的通用数据选择框架 

---
# Can Hallucination Correction Improve Video-Language Alignment? 

**Title (ZH)**: 幻觉纠正能改善视频-语言对齐？ 

**Authors**: Lingjun Zhao, Mingyang Xie, Paola Cascante-Bonilla, Hal Daumé III, Kwonjoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.15079)  

**Abstract**: Large Vision-Language Models often generate hallucinated content that is not grounded in its visual inputs. While prior work focuses on mitigating hallucinations, we instead explore leveraging hallucination correction as a training objective to improve video-language alignment. We introduce HACA, a self-training framework learning to correct hallucinations in descriptions that do not align with the video content. By identifying and correcting inconsistencies, HACA enhances the model's ability to align video and textual representations for spatio-temporal reasoning. Our experimental results show consistent gains in video-caption binding and text-to-video retrieval tasks, demonstrating that hallucination correction-inspired tasks serve as an effective strategy for improving vision and language alignment. 

**Abstract (ZH)**: 大型多模态模型常常生成与视觉输入不符的幻想内容。虽然现有工作主要集中在减轻幻觉，但我们 Instead探索将幻觉修正作为训练目标，以提高视频-语言对齐。我们提出了HACA，一个自我训练框架，学习修正与视频内容不一致的描述中的幻觉。通过识别和修正不一致性，HACA 提高了模型在空间-时间推理方面将视频和文本表示对齐的能力。我们的实验结果表明，在视频字幕绑定和文本到视频检索任务中均取得了一致的改进，证明了受幻觉修正启发的任务是提高视觉和语言对齐的有效策略。 

---
# Hardware-Friendly Static Quantization Method for Video Diffusion Transformers 

**Title (ZH)**: 面向硬件的视频扩散变换器静态量化方法 

**Authors**: Sanghyun Yi, Qingfeng Liu, Mostafa El-Khamy  

**Link**: [PDF](https://arxiv.org/pdf/2502.15077)  

**Abstract**: Diffusion Transformers for video generation have gained significant research interest since the impressive performance of SORA. Efficient deployment of such generative-AI models on GPUs has been demonstrated with dynamic quantization. However, resource-constrained devices cannot support dynamic quantization, and need static quantization of the models for their efficient deployment on AI processors. In this paper, we propose a novel method for the post-training quantization of OpenSora\cite{opensora}, a Video Diffusion Transformer, without relying on dynamic quantization techniques. Our approach employs static quantization, achieving video quality comparable to FP16 and dynamically quantized ViDiT-Q methods, as measured by CLIP, and VQA metrics. In particular, we utilize per-step calibration data to adequately provide a post-training statically quantized model for each time step, incorporating channel-wise quantization for weights and tensor-wise quantization for activations. By further applying the smooth-quantization technique, we can obtain high-quality video outputs with the statically quantized models. Extensive experimental results demonstrate that static quantization can be a viable alternative to dynamic quantization for video diffusion transformers, offering a more efficient approach without sacrificing performance. 

**Abstract (ZH)**: 基于OpenSora的视频扩散变压器的后训练静态量化方法 

---
# Rare Disease Differential Diagnosis with Large Language Models at Scale: From Abdominal Actinomycosis to Wilson's Disease 

**Title (ZH)**: 基于大规模语言模型的罕见病鉴别诊断：从腹腔放线菌病到威尔逊病 

**Authors**: Elliot Schumacher, Dhruv Naik, Anitha Kannan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15069)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in disease diagnosis. However, their effectiveness in identifying rarer diseases, which are inherently more challenging to diagnose, remains an open question. Rare disease performance is critical with the increasing use of LLMs in healthcare settings. This is especially true if a primary care physician needs to make a rarer prognosis from only a patient conversation so that they can take the appropriate next step. To that end, several clinical decision support systems are designed to support providers in rare disease identification. Yet their utility is limited due to their lack of knowledge of common disorders and difficulty of use.
In this paper, we propose RareScale to combine the knowledge LLMs with expert systems. We use jointly use an expert system and LLM to simulate rare disease chats. This data is used to train a rare disease candidate predictor model. Candidates from this smaller model are then used as additional inputs to black-box LLM to make the final differential diagnosis. Thus, RareScale allows for a balance between rare and common diagnoses. We present results on over 575 rare diseases, beginning with Abdominal Actinomycosis and ending with Wilson's Disease. Our approach significantly improves the baseline performance of black-box LLMs by over 17% in Top-5 accuracy. We also find that our candidate generation performance is high (e.g. 88.8% on gpt-4o generated chats). 

**Abstract (ZH)**: 大规模语言模型在疾病诊断中展示了令人印象深刻的 capabilities，但在识别更罕见的疾病方面，其有效性仍然存在疑问。随着大规模语言模型在医疗保健环境中使用频率的增加，罕见疾病的性能变得尤为重要。特别是在初级保健医生仅凭与患者的对话就需要做出罕见预后判断的情况下，这一点尤为重要。为此，设计了一些临床决策支持系统以支持罕见疾病识别，但其实用性因缺乏对常见疾病的了解及其使用难度而受到限制。

在本文中，我们提出RareScale结合大规模语言模型和专家系统。我们使用专家系统和大规模语言模型共同模拟罕见疾病对话，生成的数据用于训练罕见疾病候选预测模型。该小型模型的候选者随后被用作黑色盒大规模语言模型的额外输入，以做出最终的鉴别诊断。因此，RareScale 兼顾了罕见和常见疾病的诊断。我们在超过575种罕见疾病的数据上进行了实验，从腹膜炎丝状菌病开始，到威尔逊病结束。我们的方法在Top-5准确性上显著提高了黑色盒大规模语言模型的基线性能，提高了超过17%。我们还发现，我们的候选生成性能很高（例如，在gpt-4生成的对话上达到88.8%）。 

---
# Fundamental Survey on Neuromorphic Based Audio Classification 

**Title (ZH)**: 基于神经形态的音频分类基础调研 

**Authors**: Amlan Basu, Pranav Chaudhari, Gaetano Di Caterina  

**Link**: [PDF](https://arxiv.org/pdf/2502.15056)  

**Abstract**: Audio classification is paramount in a variety of applications including surveillance, healthcare monitoring, and environmental analysis. Traditional methods frequently depend on intricate signal processing algorithms and manually crafted features, which may fall short in fully capturing the complexities of audio patterns. Neuromorphic computing, inspired by the architecture and functioning of the human brain, presents a promising alternative for audio classification tasks. This survey provides an exhaustive examination of the current state-of-the-art in neuromorphic-based audio classification. It delves into the crucial components of neuromorphic systems, such as Spiking Neural Networks (SNNs), memristors, and neuromorphic hardware platforms, highlighting their advantages in audio classification. Furthermore, the survey explores various methodologies and strategies employed in neuromorphic audio classification, including event-based processing, spike-based learning, and bio-inspired feature extraction. It examines how these approaches address the limitations of traditional audio classification methods, particularly in terms of energy efficiency, real-time processing, and robustness to environmental noise. Additionally, the paper conducts a comparative analysis of different neuromorphic audio classification models and benchmarks, evaluating their performance metrics, computational efficiency, and scalability. By providing a comprehensive guide for researchers, engineers and practitioners, this survey aims to stimulate further innovation and advancements in the evolving field of neuromorphic audio classification. 

**Abstract (ZH)**: 基于神经形态计算的音频分类现状综述 

---
# Reducing Hallucinations of Medical Multimodal Large Language Models with Visual Retrieval-Augmented Generation 

**Title (ZH)**: 使用视觉检索增强生成减少医疗多模态大语言模型的幻觉 

**Authors**: Yun-Wei Chu, Kai Zhang, Christopher Malon, Martin Renqiang Min  

**Link**: [PDF](https://arxiv.org/pdf/2502.15040)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive performance in vision and text tasks. However, hallucination remains a major challenge, especially in fields like healthcare where details are critical. In this work, we show how MLLMs may be enhanced to support Visual RAG (V-RAG), a retrieval-augmented generation framework that incorporates both text and visual data from retrieved images. On the MIMIC-CXR chest X-ray report generation and Multicare medical image caption generation datasets, we show that Visual RAG improves the accuracy of entity probing, which asks whether a medical entities is grounded by an image. We show that the improvements extend both to frequent and rare entities, the latter of which may have less positive training data. Downstream, we apply V-RAG with entity probing to correct hallucinations and generate more clinically accurate X-ray reports, obtaining a higher RadGraph-F1 score. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在视觉和文本任务中展现了令人印象深刻的性能。然而，在细节至关重要的领域如医疗健康中，幻觉仍然是一个主要挑战。在本文中，我们展示了如何通过引入视觉RAG（V-RAG）框架来增强MLLMs，该框架结合了检索到的图像中的文字和视觉数据。在MIMIC-CXR胸部X光报告生成和Multicare医学图像标题生成数据集上，我们证明了视觉RAG能够提高实体探查的准确性，以确定医学实体是否由图像支持。我们还展示了这些改进不仅适用于频繁出现的实体，也适用于训练数据较少的罕见实体。在下游应用中，我们使用V-RAG结合实体探查来纠正幻觉并生成更临床准确的X光报告，从而获得更高的RadGraph-F1评分。 

---
# DEFT: Differentiable Branched Discrete Elastic Rods for Modeling Furcated DLOs in Real-Time 

**Title (ZH)**: DEFT：可微分分枝离散弹性杆模型及其在实时建模中的应用 

**Authors**: Yizhou Chen, Xiaoyue Wu, Yeheng Zong, Anran Li, Yuzhen Chen, Julie Wu, Bohao Zhang, Ram Vasudevan  

**Link**: [PDF](https://arxiv.org/pdf/2502.15037)  

**Abstract**: Autonomous wire harness assembly requires robots to manipulate complex branched cables with high precision and reliability. A key challenge in automating this process is predicting how these flexible and branched structures behave under manipulation. Without accurate predictions, it is difficult for robots to reliably plan or execute assembly operations. While existing research has made progress in modeling single-threaded Deformable Linear Objects (DLOs), extending these approaches to Branched Deformable Linear Objects (BDLOs) presents fundamental challenges. The junction points in BDLOs create complex force interactions and strain propagation patterns that cannot be adequately captured by simply connecting multiple single-DLO models. To address these challenges, this paper presents Differentiable discrete branched Elastic rods for modeling Furcated DLOs in real-Time (DEFT), a novel framework that combines a differentiable physics-based model with a learning framework to: 1) accurately model BDLO dynamics, including dynamic propagation at junction points and grasping in the middle of a BDLO, 2) achieve efficient computation for real-time inference, and 3) enable planning to demonstrate dexterous BDLO manipulation. A comprehensive series of real-world experiments demonstrates DEFT's efficacy in terms of accuracy, computational speed, and generalizability compared to state-of-the-art alternatives. Project page:this https URL. 

**Abstract (ZH)**: 自主线束装配要求机器人以高精度和可靠性操作复杂的分支电缆。自动化这一过程的关键挑战在于预测这些柔性和分支结构在操作过程中的行为。缺乏准确的预测，使机器人难以可靠地规划或执行装配操作。虽然现有研究在建模单线Deformable Linear Objects (DLO)方面取得了进展，但将这些方法扩展到Branched Deformable Linear Objects (BDLO)面临根本性的挑战。BDLO中的连接点创造了复杂的力交互和应变传播模式，这些模式无法通过简单连接多个单线模型来充分捕捉。为了应对这些挑战，本文提出了一种新的框架Differentiable discrete branched Elastic rods for modeling Furcated DLOs in real-Time (DEFT)，该框架结合了可微分物理模型和学习框架，以：1）准确建模BDLO动力学，包括分支连接点处的动力学传播和BDLO中部的抓取；2）实现高效的实时推理计算；3）支持对演示多才多艺的BDLO操作的规划。一系列全面的实际实验表明，DEFT在准确度、计算速度和泛化能力方面优于最新的替代方案。项目页面：this https URL。 

---
# InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models via Human Feedback 

**Title (ZH)**: InterFeedback: 通过人类反馈揭示大型多模态模型的交互智能 

**Authors**: Henry Hengyuan Zhao, Wenqi Pei, Yifei Tao, Haiyang Mei, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2502.15027)  

**Abstract**: Existing benchmarks do not test Large Multimodal Models (LMMs) on their interactive intelligence with human users which is vital for developing general-purpose AI assistants. We design InterFeedback, an interactive framework, which can be applied to any LMM and dataset to assess this ability autonomously. On top of this, we introduce InterFeedback-Bench which evaluates interactive intelligence using two representative datasets, MMMU-Pro and MathVerse, to test 10 different open-source LMMs. Additionally, we present InterFeedback-Human, a newly collected dataset of 120 cases designed for manually testing interactive performance in leading models such as OpenAI-o1 and Claude-3.5-Sonnet. Our evaluation results show that even state-of-the-art LMM (like OpenAI-o1) can correct their results through human feedback less than 50%. Our findings point to the need for methods that can enhance the LMMs' capability to interpret and benefit from feedback. 

**Abstract (ZH)**: 现有基准未测试大型多模态模型与人类用户的互动智能，这对于开发通用AI助手至关重要。我们设计了InterFeedback，这是一种可以应用于任何大型多模态模型和数据集的互动框架，以自主评估其这种能力。在此基础上，我们引入了使用MMMU-Pro和MathVerse两个代表性数据集评估互动智能的InterFeedback-Bench，并测试了10种不同的开源大型多模态模型。此外，我们还推出了InterFeedback-Human，这是一个新收集的包含120个案例的数据集，用于手动测试领先模型（如OpenAI-o1和Claude-3.5-Sonnet）的互动性能。我们的评估结果表明，即使是最先进的大型多模态模型（如OpenAI-o1），也有可能通过人类反馈修正其结果不足50%。我们的发现指出了需要改进的方法，以增强大型多模态模型解释和利用反馈的能力。 

---
# Towards Physics-Guided Foundation Models 

**Title (ZH)**: 面向物理引导的基础模型 

**Authors**: Majid Farhadloo, Arun Sharma, Mingzhou Yang, Bharat Jayaprakash, William Northrop, Shashi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15013)  

**Abstract**: Traditional foundation models are pre-trained on broad datasets to reduce the training resources (e.g., time, energy, labeled samples) needed for fine-tuning a wide range of downstream tasks. However, traditional foundation models struggle with out-of-distribution prediction and can produce outputs that are unrealistic and physically infeasible. We propose the notation of physics-guided foundation models (PGFM), that is, foundation models integrated with broad or general domain (e.g., scientific) physical knowledge applicable to a wide range of downstream tasks. 

**Abstract (ZH)**: 基于物理引导的基础模型（PGFM）：整合广泛或通用领域物理知识的基础模型 

---
# Graph in the Vault: Protecting Edge GNN Inference with Trusted Execution Environment 

**Title (ZH)**: 库中之图：基于可信执行环境的边缘GNN推理保护 

**Authors**: Ruyi Ding, Tianhong Xu, Aidong Adam Ding, Yunsi Fei  

**Link**: [PDF](https://arxiv.org/pdf/2502.15012)  

**Abstract**: Wide deployment of machine learning models on edge devices has rendered the model intellectual property (IP) and data privacy vulnerable. We propose GNNVault, the first secure Graph Neural Network (GNN) deployment strategy based on Trusted Execution Environment (TEE). GNNVault follows the design of 'partition-before-training' and includes a private GNN rectifier to complement with a public backbone model. This way, both critical GNN model parameters and the private graph used during inference are protected within secure TEE compartments. Real-world implementations with Intel SGX demonstrate that GNNVault safeguards GNN inference against state-of-the-art link stealing attacks with negligible accuracy degradation (<2%). 

**Abstract (ZH)**: 基于受信执行环境的广义图神经网络部署策略GNNVault：保护模型知识产权和数据隐私 

---
# Obliviate: Efficient Unmemorization for Protecting Intellectual Property in Large Language Models 

**Title (ZH)**: 消隐：高效去记忆化以保护大型语言模型中的知识产权 

**Authors**: Mark Russinovich, Ahmed Salem  

**Link**: [PDF](https://arxiv.org/pdf/2502.15010)  

**Abstract**: Recent copyright agreements between AI companies and content creators have highlighted the need for precise control over language models' ability to reproduce copyrighted content. While existing approaches rely on either complete concept removal through unlearning or simple output filtering, we propose Obliviate, a novel post-training technique that selectively prevents verbatim reproduction of specific text while preserving semantic understanding.
Obliviate operates by selecting tokens within memorized sequences and modifying the model's probability distribution to prevent exact reproduction while maintaining contextual understanding. We evaluate Obliviate on multiple large language models (LLaMA-3.1 8B, LLaMA-3.1-instruct 8B, Qwen-2.5-7B, and Yi-1.5 6B) across both synthetic memorization tasks and organic copyright content. Our results demonstrate that Obliviate achieves orders of magnitude reduction, e.g., 100x, in verbatim memorization while maintaining model performance within 1% of baseline on standard benchmarks (HellaSwag, MMLU, TruthfulQA, and Winogrande). This makes Obliviate particularly suitable for practical deployment scenarios where companies need to efficiently address copyright concerns in pretrained models without compromising their general capabilities. 

**Abstract (ZH)**: 近期，人工智能公司与内容创作者之间的版权协议强调了对语言模型复制版权内容能力进行精确控制的必要性。现有方法依赖于完全通过遗忘去除概念或简单的输出过滤，我们提出了一种名为Obliviate的新型后训练技术，该技术能够在保持语义理解的同时，选择性地防止特定文本的精确复制。Obliviate通过在记忆序列中选择词汇并修改模型的概率分布，以防止精确复制同时保持上下文理解。我们在多个大型语言模型（包括LLaMA-3.1 8B、LLaMA-3.1-instruct 8B、Qwen-2.5-7B和Yi-1.5 6B）上对Obliviate进行了评价，涵盖合成记忆任务和有机版权内容。实验结果表明，Obliviate在保持模型性能接近基线（检核街谈、MMLU、TruthfulQA和Winogrande）的情况下，精确记忆减少了数个数量级，例如减少100倍。这使得Obliviate特别适合实际部署场景，公司可以在不牺牲其通用能力的情况下有效解决预训练模型中的版权问题。 

---
# LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers 

**Title (ZH)**: LLM-Microscope: 探索标点符号在Transformer上下文记忆中的隐藏作用 

**Authors**: Anton Razzhigaev, Matvey Mikhalchuk, Temurbek Rahmatullaev, Elizaveta Goncharova, Polina Druzhinina, Ivan Oseledets, Andrey Kuznetsov  

**Link**: [PDF](https://arxiv.org/pdf/2502.15007)  

**Abstract**: We introduce methods to quantify how Large Language Models (LLMs) encode and store contextual information, revealing that tokens often seen as minor (e.g., determiners, punctuation) carry surprisingly high context. Notably, removing these tokens -- especially stopwords, articles, and commas -- consistently degrades performance on MMLU and BABILong-4k, even if removing only irrelevant tokens. Our analysis also shows a strong correlation between contextualization and linearity, where linearity measures how closely the transformation from one layer's embeddings to the next can be approximated by a single linear mapping. These findings underscore the hidden importance of filler tokens in maintaining context. For further exploration, we present LLM-Microscope, an open-source toolkit that assesses token-level nonlinearity, evaluates contextual memory, visualizes intermediate layer contributions (via an adapted Logit Lens), and measures the intrinsic dimensionality of representations. This toolkit illuminates how seemingly trivial tokens can be critical for long-range understanding. 

**Abstract (ZH)**: 我们介绍了量化大型语言模型（LLMs）编码和存储上下文信息的方法，揭示了常被视为次要的标记（例如，限定词、标点符号）实际上携带了出乎意料高的上下文信息。值得注意的是，即使仅去除无关标记，尤其是停用词、冠词和逗号，也会一致地降低MMLU和BABILong-4k的性能。我们的分析还显示了上下文化与线性度之间的强相关性，其中线性度衡量从一层嵌入到下一层的转换能被单个线性映射近似刻画的程度。这些发现强调了填充标记在保持上下文方面的重要性。为进一步探索，我们提出了LLM-Microscope，这是一个开源工具包，用于评估标记级别的非线性度、评估上下文记忆、通过调整后的Logit Lens可视化中间层贡献，并测量表示的固有维度。该工具包揭示了看似琐碎的标记对于长距离理解可能是至关重要的。 

---
# Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions 

**Title (ZH)**: 超越地平线的安全性：基于神经控制障碍函数的高效采样 MPC 

**Authors**: Ji Yin, Oswin So, Eric Yang Yu, Chuchu Fan, Panagiotis Tsiotras  

**Link**: [PDF](https://arxiv.org/pdf/2502.15006)  

**Abstract**: A common problem when using model predictive control (MPC) in practice is the satisfaction of safety specifications beyond the prediction horizon. While theoretical works have shown that safety can be guaranteed by enforcing a suitable terminal set constraint or a sufficiently long prediction horizon, these techniques are difficult to apply and thus are rarely used by practitioners, especially in the case of general nonlinear dynamics. To solve this problem, we impose a tradeoff between exact recursive feasibility, computational tractability, and applicability to ''black-box'' dynamics by learning an approximate discrete-time control barrier function and incorporating it into a variational inference MPC (VIMPC), a sampling-based MPC paradigm. To handle the resulting state constraints, we further propose a new sampling strategy that greatly reduces the variance of the estimated optimal control, improving the sample efficiency, and enabling real-time planning on a CPU. The resulting Neural Shield-VIMPC (NS-VIMPC) controller yields substantial safety improvements compared to existing sampling-based MPC controllers, even under badly designed cost functions. We validate our approach in both simulation and real-world hardware experiments. 

**Abstract (ZH)**: 使用模型预测控制时的一个常见问题是超越预测 horizon 后满足安全规范。 

---
# A Socratic RAG Approach to Connect Natural Language Queries on Research Topics with Knowledge Organization Systems 

**Title (ZH)**: 苏格拉底式RAG方法将研究主题的自然语言查询与知识组织系统连接起来 

**Authors**: Lew Lefton, Kexin Rong, Chinar Dankhara, Lila Ghemri, Firdous Kausar, A. Hannibal Hamdallahi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15005)  

**Abstract**: In this paper, we propose a Retrieval Augmented Generation (RAG) agent that maps natural language queries about research topics to precise, machine-interpretable semantic entities. Our approach combines RAG with Socratic dialogue to align a user's intuitive understanding of research topics with established Knowledge Organization Systems (KOSs). The proposed approach will effectively bridge "little semantics" (domain-specific KOS structures) with "big semantics" (broad bibliometric repositories), making complex academic taxonomies more accessible. Such agents have the potential for broad use. We illustrate with a sample application called CollabNext, which is a person-centric knowledge graph connecting people, organizations, and research topics. We further describe how the application design has an intentional focus on HBCUs and emerging researchers to raise visibility of people historically rendered invisible in the current science system. 

**Abstract (ZH)**: 基于检索增强生成的领域知识对话代理：连接“小语义”与“大语义” 

---
# A Rapid Test for Accuracy and Bias of Face Recognition Technology 

**Title (ZH)**: 快速检测面部识别技术准确性和偏差的方法 

**Authors**: Manuel Knott, Ignacio Serna, Ethan Mann, Pietro Perona  

**Link**: [PDF](https://arxiv.org/pdf/2502.14996)  

**Abstract**: Measuring the accuracy of face recognition (FR) systems is essential for improving performance and ensuring responsible use. Accuracy is typically estimated using large annotated datasets, which are costly and difficult to obtain. We propose a novel method for 1:1 face verification that benchmarks FR systems quickly and without manual annotation, starting from approximate labels (e.g., from web search results). Unlike previous methods for training set label cleaning, ours leverages the embedding representation of the models being evaluated, achieving high accuracy in smaller-sized test datasets. Our approach reliably estimates FR accuracy and ranking, significantly reducing the time and cost of manual labeling. We also introduce the first public benchmark of five FR cloud services, revealing demographic biases, particularly lower accuracy for Asian women. Our rapid test method can democratize FR testing, promoting scrutiny and responsible use of the technology. Our method is provided as a publicly accessible tool at this https URL 

**Abstract (ZH)**: 测量面部识别系统的准确性对于提升性能和确保负责任的使用至关重要。准确性通常通过大型标注数据集进行估计，但这些数据集成本高昂且获取困难。我们提出了一种新型的1:1面部验证方法，可以在无需手动标注的情况下快速评估面部识别系统，并从近似标签（例如，从网络搜索结果中获得）开始。与以往用于训练集标签清理的方法不同，我们的方法利用了所评估模型的嵌入表示，从而在较小规模的测试数据集中实现了高准确性。我们的方法可靠地估计了面部识别的准确性和排名，显著减少了人工标注所需的时间和成本。我们还引入了第一个公开的五种面部识别云服务的基准测试，揭示了 demographic 偏见，特别是亚洲女性的准确性较低。我们的快速测试方法可以促进面部识别技术的民主化测试，推动对该技术的审视和负责任的使用。我们的方法作为可访问工具提供于此 https://链接。 

---
# Beyond No: Quantifying AI Over-Refusal and Emotional Attachment Boundaries 

**Title (ZH)**: 超越否决：量化AI过度拒绝和情感依附边界 

**Authors**: David Noever, Grant Rosario  

**Link**: [PDF](https://arxiv.org/pdf/2502.14975)  

**Abstract**: We present an open-source benchmark and evaluation framework for assessing emotional boundary handling in Large Language Models (LLMs). Using a dataset of 1156 prompts across six languages, we evaluated three leading LLMs (GPT-4o, Claude-3.5 Sonnet, and Mistral-large) on their ability to maintain appropriate emotional boundaries through pattern-matched response analysis. Our framework quantifies responses across seven key patterns: direct refusal, apology, explanation, deflection, acknowledgment, boundary setting, and emotional awareness. Results demonstrate significant variation in boundary-handling approaches, with Claude-3.5 achieving the highest overall score (8.69/10) and producing longer, more nuanced responses (86.51 words on average). We identified a substantial performance gap between English (average score 25.62) and non-English interactions (< 0.22), with English responses showing markedly higher refusal rates (43.20% vs. < 1% for non-English). Pattern analysis revealed model-specific strategies, such as Mistral's preference for deflection (4.2%) and consistently low empathy scores across all models (< 0.06). Limitations include potential oversimplification through pattern matching, lack of contextual understanding in response analysis, and binary classification of complex emotional responses. Future work should explore more nuanced scoring methods, expand language coverage, and investigate cultural variations in emotional boundary expectations. Our benchmark and methodology provide a foundation for systematic evaluation of LLM emotional intelligence and boundary-setting capabilities. 

**Abstract (ZH)**: 一种评估大型语言模型情感边界处理能力的开源基准和评估框架 

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
# FacaDiffy: Inpainting Unseen Facade Parts Using Diffusion Models 

**Title (ZH)**: FacaDiffy: 使用扩散模型 inpaint 未见墙面部分 

**Authors**: Thomas Froech, Olaf Wysocki, Yan Xia, Junyu Xie, Benedikt Schwab, Daniel Cremers, Thomas H. Kolbe  

**Link**: [PDF](https://arxiv.org/pdf/2502.14940)  

**Abstract**: High-detail semantic 3D building models are frequently utilized in robotics, geoinformatics, and computer vision. One key aspect of creating such models is employing 2D conflict maps that detect openings' locations in building facades. Yet, in reality, these maps are often incomplete due to obstacles encountered during laser scanning. To address this challenge, we introduce FacaDiffy, a novel method for inpainting unseen facade parts by completing conflict maps with a personalized Stable Diffusion model. Specifically, we first propose a deterministic ray analysis approach to derive 2D conflict maps from existing 3D building models and corresponding laser scanning point clouds. Furthermore, we facilitate the inpainting of unseen facade objects into these 2D conflict maps by leveraging the potential of personalizing a Stable Diffusion model. To complement the scarcity of real-world training data, we also develop a scalable pipeline to produce synthetic conflict maps using random city model generators and annotated facade images. Extensive experiments demonstrate that FacaDiffy achieves state-of-the-art performance in conflict map completion compared to various inpainting baselines and increases the detection rate by $22\%$ when applying the completed conflict maps for high-definition 3D semantic building reconstruction. The code is be publicly available in the corresponding GitHub repository: this https URL 

**Abstract (ZH)**: 高详细语义3D建筑模型在机器人技术、地理信息和计算机视觉中的广泛应用需要利用二维冲突图来检测建筑立面的开口位置。然而，由于激光扫描过程中遇到的障碍物，这些图往往不完整。为了解决这一挑战，我们引入了FacaDiffy，这是一种通过使用个性化Stable Diffusion模型填充未见立面部分并完成冲突图的新型方法。具体而言，我们首先提出了一种确定性射线分析方法，从现有的3D建筑模型和相应的激光扫描点云中推导出二维冲突图。此外，我们利用个性化Stable Diffusion模型的潜力，促进未见立面对象的填充，使其集成到这些二维冲突图中。为补充现实世界训练数据的不足，我们还开发了一种可扩展的工作流程，使用随机城市模型生成器和注释立面图像来生成合成冲突图。广泛实验表明，FacaDiffy 在冲突图填充方面的性能超过了各种插值基线，并且当使用完成后的冲突图进行高精度3D语义建筑重建时，检测率提高了22%。相关代码将在对应的GitHub仓库中公开：this https URL 

---
# Online hand gesture recognition using Continual Graph Transformers 

**Title (ZH)**: 使用 Continual Graph Transformers 的在线手部手势识别 

**Authors**: Rim Slama, Wael Rabah, Hazem Wannous  

**Link**: [PDF](https://arxiv.org/pdf/2502.14939)  

**Abstract**: Online continuous action recognition has emerged as a critical research area due to its practical implications in real-world applications, such as human-computer interaction, healthcare, and robotics. Among various modalities, skeleton-based approaches have gained significant popularity, demonstrating their effectiveness in capturing 3D temporal data while ensuring robustness to environmental variations. However, most existing works focus on segment-based recognition, making them unsuitable for real-time, continuous recognition scenarios. In this paper, we propose a novel online recognition system designed for real-time skeleton sequence streaming. Our approach leverages a hybrid architecture combining Spatial Graph Convolutional Networks (S-GCN) for spatial feature extraction and a Transformer-based Graph Encoder (TGE) for capturing temporal dependencies across frames. Additionally, we introduce a continual learning mechanism to enhance model adaptability to evolving data distributions, ensuring robust recognition in dynamic environments. We evaluate our method on the SHREC'21 benchmark dataset, demonstrating its superior performance in online hand gesture recognition. Our approach not only achieves state-of-the-art accuracy but also significantly reduces false positive rates, making it a compelling solution for real-time applications. The proposed system can be seamlessly integrated into various domains, including human-robot collaboration and assistive technologies, where natural and intuitive interaction is crucial. 

**Abstract (ZH)**: 基于骨架的在线连续动作识别已 emerges as a critical research area due to its practical implications in real-world applications, such as human-computer interaction, healthcare, and robotics. 

---
# Fast and Accurate Blind Flexible Docking 

**Title (ZH)**: 快速且准确的目标灵活对接 

**Authors**: Zizhuo Zhang, Lijun Wu, Kaiyuan Gao, Jiangchao Yao, Tao Qin, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.14934)  

**Abstract**: Molecular docking that predicts the bound structures of small molecules (ligands) to their protein targets, plays a vital role in drug discovery. However, existing docking methods often face limitations: they either overlook crucial structural changes by assuming protein rigidity or suffer from low computational efficiency due to their reliance on generative models for structure sampling. To address these challenges, we propose FABFlex, a fast and accurate regression-based multi-task learning model designed for realistic blind flexible docking scenarios, where proteins exhibit flexibility and binding pocket sites are unknown (blind). Specifically, FABFlex's architecture comprises three specialized modules working in concert: (1) A pocket prediction module that identifies potential binding sites, addressing the challenges inherent in blind docking scenarios. (2) A ligand docking module that predicts the bound (holo) structures of ligands from their unbound (apo) states. (3) A pocket docking module that forecasts the holo structures of protein pockets from their apo conformations. Notably, FABFlex incorporates an iterative update mechanism that serves as a conduit between the ligand and pocket docking modules, enabling continuous structural refinements. This approach effectively integrates the three subtasks of blind flexible docking-pocket identification, ligand conformation prediction, and protein flexibility modeling-into a unified, coherent framework. Extensive experiments on public benchmark datasets demonstrate that FABFlex not only achieves superior effectiveness in predicting accurate binding modes but also exhibits a significant speed advantage (208 $\times$) compared to existing state-of-the-art methods. Our code is released at this https URL. 

**Abstract (ZH)**: 基于快速准确多任务学习的FABFlex分子对接模型：适用于盲柔性对接场景的小分子与蛋白质靶标结合结构预测 

---
# A Tale of Two Structures: Do LLMs Capture the Fractal Complexity of Language? 

**Title (ZH)**: 两种结构的故事：大型语言模型能否捕捉语言的分形复杂性？ 

**Authors**: Ibrahim Alabdulmohsin, Andreas Steiner  

**Link**: [PDF](https://arxiv.org/pdf/2502.14924)  

**Abstract**: Language exhibits a fractal structure in its information-theoretic complexity (i.e. bits per token), with self-similarity across scales and long-range dependence (LRD). In this work, we investigate whether large language models (LLMs) can replicate such fractal characteristics and identify conditions-such as temperature setting and prompting method-under which they may fail. Moreover, we find that the fractal parameters observed in natural language are contained within a narrow range, whereas those of LLMs' output vary widely, suggesting that fractal parameters might prove helpful in detecting a non-trivial portion of LLM-generated texts. Notably, these findings, and many others reported in this work, are robust to the choice of the architecture; e.g. Gemini 1.0 Pro, Mistral-7B and Gemma-2B. We also release a dataset comprising of over 240,000 articles generated by various LLMs (both pretrained and instruction-tuned) with different decoding temperatures and prompting methods, along with their corresponding human-generated texts. We hope that this work highlights the complex interplay between fractal properties, prompting, and statistical mimicry in LLMs, offering insights for generating, evaluating and detecting synthetic texts. 

**Abstract (ZH)**: 语言在其信息论复杂性（即每令牌位数）上表现出分形结构，具有跨尺度的自我相似性和长程依赖性。在本研究中，我们探讨大型语言模型（LLMs）能否重现这种分形特性，并确定在何种条件下它们可能会失败。此外，我们发现自然语言中观察到的分形参数范围狭窄，而LLMs输出的分形参数范围广泛，表明分形参数可能有助于检测LLM生成文本中的较大一部分。值得注意的是，这些发现以及本研究中报道的其他发现对于架构选择是稳健的；例如，Gemini 1.0 Pro、Mistral-7B和Gemma-2B。我们还发布了一个数据集，包含超过240,000篇由不同LLMs（包括预训练和指令调优）在不同解码温度和提示方法下生成的文章及其相应的手工生成文本。我们希望这项工作能够突出分形属性、提示和统计模拟之间复杂的相互作用，为生成、评估和检测合成文本提供见解。 

---
# AI Thinking as a Meaning-Centered Framework: Reimagining Language Technologies Through Community Agency 

**Title (ZH)**: AI思维作为意义中心的框架：通过社区代理 reimagine 语言技术 

**Authors**: Jose F Quesada  

**Link**: [PDF](https://arxiv.org/pdf/2502.14923)  

**Abstract**: While language technologies have advanced significantly, current approaches fail to address the complex sociocultural dimensions of linguistic preservation. AI Thinking proposes a meaning-centered framework that would transform technological development from creating tools FOR communities to co-creating solutions WITH them. This approach recognizes that meaningful solutions emerge through the interplay of cultural understanding, community agency, and technological innovation. The proposal articulates a holistic methodology and a five-layer technological ecosystem where communities maintain control over their linguistic and cultural knowledge representation. This systematic integration of community needs, cultural preservation, and advanced capabilities could revolutionize how we approach linguistic diversity preservation in the digital age. 

**Abstract (ZH)**: 尽管语言技术取得了显著进步，当前的方法未能解决语言保存中的复杂社会文化维度。AI思考提出了一种以意义为中心的框架，旨在将技术发展从为社区创造工具转变为与社区共同创造解决方案。这种方法认识到，有意义的解决方案通过文化理解、社区自主性和技术创新的互动而产生。该提案阐述了一个整体方法论和五层技术生态系统，使社区能够控制其语言和文化知识的表述。这种系统地整合社区需求、文化保存和先进技术的能力有可能在数字时代彻底改变我们处理语言多样性保存的方式。 

---
# SIFT: Grounding LLM Reasoning in Contexts via Stickers 

**Title (ZH)**: SIFT: 通过贴纸在上下文中约束LLM推理 

**Authors**: Zihao Zeng, Xuyao Huang, Boxiu Li, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.14922)  

**Abstract**: This paper identifies the misinterpretation of the context can be a significant issue during the reasoning process of large language models, spanning from smaller models like Llama3.2-3B-Instruct to cutting-edge ones like DeepSeek-R1. For example, in the phrase "10 dollars per kilo," LLMs might not recognize that "per" means "for each," leading to calculation errors. We introduce a novel, post-training approach called **Stick to the Facts (SIFT)** to tackle this. SIFT leverages increasing inference-time compute to ground LLM reasoning in contexts. At the core of SIFT lies the *Sticker*, which is generated by the model itself to explicitly emphasize the key information within the context. Given the curated Sticker, SIFT generates two predictions -- one from the original query and one from the query augmented with the Sticker. If they differ, the Sticker is sequentially refined via *forward* optimization (to better align the extracted facts with the query) and *inverse* generation (to conform with the model's inherent tendencies) for more faithful reasoning outcomes. Studies across diverse models (from 3B to 100B+) and benchmarks (e.g., GSM8K, MATH-500) reveal consistent performance improvements. Notably, SIFT improves the pass@1 accuracy of DeepSeek-R1 on AIME2024 from 78.33% to **85.67**%, establishing a new state-of-the-art in the open-source community. The code is available at this https URL. 

**Abstract (ZH)**: This paper identifies the misinterpretation of context as a significant issue during the reasoning process of large language models, spanning from smaller models like Llama3.2-3B-Instruct to cutting-edge ones like DeepSeek-R1. For example, in the phrase "10 dollars per kilo," LLMs might not recognize that "per" means "for each," leading to calculation errors. We introduce a novel, post-training approach called **Stick to the Facts (SIFT)** to address this. SIFT leverages increased inference-time compute to ground LLM reasoning in contexts. At the core of SIFT lies the *Sticker*, which is generated by the model itself to explicitly emphasize key information within the context. Given the curated Sticker, SIFT generates two predictions—one from the original query and one from the query augmented with the Sticker. If they differ, the Sticker is sequentially refined via forward optimization (to better align the extracted facts with the query) and inverse generation (to conform with the model's inherent tendencies) for more faithful reasoning outcomes. Studies across diverse models (from 3B to 100B+) and benchmarks (e.g., GSM8K, MATH-500) reveal consistent performance improvements. Notably, SIFT improves the pass@1 accuracy of DeepSeek-R1 on AIME2024 from 78.33% to **85.67%**, establishing a new state-of-the-art in the open-source community. The code is available at this https URL. 

---
# Display Field-Of-View Agnostic Robust CT Kernel Synthesis Using Model-Based Deep Learning 

**Title (ZH)**: 基于模型的深度学习在不依赖显示场视野的情况下 robust CT 内核合成 

**Authors**: Hemant Kumar Aggarwal, Antony Jerald, Phaneendra K. Yalavarthy, Rajesh Langoju, Bipul Das  

**Link**: [PDF](https://arxiv.org/pdf/2502.14920)  

**Abstract**: In X-ray computed tomography (CT) imaging, the choice of reconstruction kernel is crucial as it significantly impacts the quality of clinical images. Different kernels influence spatial resolution, image noise, and contrast in various ways. Clinical applications involving lung imaging often require images reconstructed with both soft and sharp kernels. The reconstruction of images with different kernels requires raw sinogram data and storing images for all kernels increases processing time and storage requirements. The Display Field-of-View (DFOV) adds complexity to kernel synthesis, as data acquired at different DFOVs exhibit varying levels of sharpness and details. This work introduces an efficient, DFOV-agnostic solution for image-based kernel synthesis using model-based deep learning. The proposed method explicitly integrates CT kernel and DFOV characteristics into the forward model. Experimental results on clinical data, along with quantitative analysis of the estimated modulation transfer function using wire phantom data, clearly demonstrate the utility of the proposed method in real-time. Additionally, a comparative study with a direct learning network, that lacks forward model information, shows that the proposed method is more robust to DFOV variations. 

**Abstract (ZH)**: 在X射线计算机断层成像（CT）中，重建核的选择至关重要，因为它显著影响临床图像的质量。不同的重建核以不同方式影响空间分辨率、图像噪声和对比度。涉及肺部成像的临床应用通常需要使用软核和锐核重建图像。使用不同重建核重建图像需要原始.sinogram数据，并存储所有核的图像会增加处理时间和存储需求。显示视野（DFOV）的引入增加了核合成的复杂性，因为在不同DFOV下获取的数据具有不同的清晰度和细节水平。本工作提出了一种适用于DFOV的基于图像的重建核合成的高效解决方案，该方法利用基于模型的深度学习方法，明确将CT重建核和DFOV特性整合到正向模型中。临床数据的实验结果和使用线圈模型数据进行的调制传递函数的定量分析清楚地证明了所提出方法在实时应用中的有效性。此外，与缺乏正向模型信息的直接学习网络进行的比较研究显示，所提出的方法在DFOV变化时更为稳健。 

---
# RAPTOR: Refined Approach for Product Table Object Recognition 

**Title (ZH)**: RAPTOR：精炼的产品表格对象识别方法 

**Authors**: Eliott Thomas, Mickael Coustaty, Aurelie Joseph, Elodie Carel, Vincent Poulain D'Andecy, Jean-Marc Ogier  

**Link**: [PDF](https://arxiv.org/pdf/2502.14918)  

**Abstract**: Extracting tables from documents is a critical task across various industries, especially on business documents like invoices and reports. Existing systems based on DEtection TRansformer (DETR) such as TAble TRansformer (TATR), offer solutions for Table Detection (TD) and Table Structure Recognition (TSR) but face challenges with diverse table formats and common errors like incorrect area detection and overlapping columns. This research introduces RAPTOR, a modular post-processing system designed to enhance state-of-the-art models for improved table extraction, particularly for product tables. RAPTOR addresses recurrent TD and TSR issues, improving both precision and structural predictions. For TD, we use DETR (trained on ICDAR 2019) and TATR (trained on PubTables-1M and FinTabNet), while TSR only relies on TATR. A Genetic Algorithm is incorporated to optimize RAPTOR's module parameters, using a private dataset of product tables to align with industrial needs. We evaluate our method on two private datasets of product tables, the public DOCILE dataset (which contains tables similar to our target product tables), and the ICDAR 2013 and ICDAR 2019 datasets. The results demonstrate that while our approach excels at product tables, it also maintains reasonable performance across diverse table formats. An ablation study further validates the contribution of each module in our system. 

**Abstract (ZH)**: 从文档中提取表格是各个行业中的关键任务，特别是在发票和报告等商务文档中。基于DEtection TRansformer (DETR)的系统，如TAble TRansformer (TATR)，为表格检测 (TD) 和表格结构识别 (TSR) 提供了解决方案，但在处理多样化的表格格式和常见错误（如检测区域不准确和列重叠）方面仍面临挑战。本研究介绍了一种模块化后处理系统RAPTOR，旨在增强最先进的表格提取模型，特别是在产品表格方面的表现。RAPTOR解决了反复出现的表格检测和结构识别问题，提高了准确性和结构预测。对于表格检测，我们使用DETR（以ICDAR 2019数据集训练）和TATR（以PubTables-1M和FinTabNet数据集训练），而结构识别仅依赖于TATR。我们采用了遗传算法来优化RAPTOR的模块参数，并使用包含产品表格的私有数据集来满足工业需求。我们在两个私有产品表格数据集、公开的DOCILE数据集（包含类似目标产品表格的表格）以及ICDAR 2013和ICDAR 2019数据集上评估了我们的方法。结果显示，虽然我们的方法在产品表格方面表现优异，但在多样化的表格格式方面也保持了合理的性能。进一步的消融研究验证了我们系统中每个模块的贡献。 

---
# Sce2DriveX: A Generalized MLLM Framework for Scene-to-Drive Learning 

**Title (ZH)**: Sce2DriveX：一种场景到驾驶学习的一般化多模态模型框架 

**Authors**: Rui Zhao, Qirui Yuan, Jinyu Li, Haofeng Hu, Yun Li, Chengyuan Zheng, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14917)  

**Abstract**: End-to-end autonomous driving, which directly maps raw sensor inputs to low-level vehicle controls, is an important part of Embodied AI. Despite successes in applying Multimodal Large Language Models (MLLMs) for high-level traffic scene semantic understanding, it remains challenging to effectively translate these conceptual semantics understandings into low-level motion control commands and achieve generalization and consensus in cross-scene driving. We introduce Sce2DriveX, a human-like driving chain-of-thought (CoT) reasoning MLLM framework. Sce2DriveX utilizes multimodal joint learning from local scene videos and global BEV maps to deeply understand long-range spatiotemporal relationships and road topology, enhancing its comprehensive perception and reasoning capabilities in 3D dynamic/static scenes and achieving driving generalization across scenes. Building on this, it reconstructs the implicit cognitive chain inherent in human driving, covering scene understanding, meta-action reasoning, behavior interpretation analysis, motion planning and control, thereby further bridging the gap between autonomous driving and human thought processes. To elevate model performance, we have developed the first extensive Visual Question Answering (VQA) driving instruction dataset tailored for 3D spatial understanding and long-axis task reasoning. Extensive experiments demonstrate that Sce2DriveX achieves state-of-the-art performance from scene understanding to end-to-end driving, as well as robust generalization on the CARLA Bench2Drive benchmark. 

**Abstract (ZH)**: 端到端自动驾驶：从传感器原始输入直接映射到低级车辆控制，是具身AI的重要组成部分。尽管在高层次交通场景语义理解中成功应用了多模态大型语言模型（MLLMs），但在有效地将这些概念性语义理解转换为低级运动控制命令，并在跨场景驾驶中实现泛化和共识方面仍面临挑战。我们引入了Sce2DriveX，这是一种类人的驾驶链式思考（CoT）推理MLLM框架。Sce2DriveX利用局部场景视频和全局BEV地图的多模态联合学习，深入理解远距离时空关系和道路拓扑，增强其在3D动态/静态场景中的综合感知和推理能力，实现跨场景驾驶的泛化。在此基础上，它重构了人类驾驶内在的隐式认知链条，涵盖场景理解、元动作推理、行为解释分析、运动规划和控制，从而进一步弥合自动驾驶与人类思维过程之间的差距。为了提升模型性能，我们开发了首个专门针对3D空间理解和长轴任务推理的视觉问答（VQA）驾驶指令数据集。广泛的实验表明，Sce2DriveX在从场景理解到端到端驾驶的性能上达到了最先进的水平，并在CARLA Bench2Drive基准测试中展现出鲁棒的泛化能力。 

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
# Batayan: A Filipino NLP benchmark for evaluating Large Language Models 

**Title (ZH)**: Batayan：一种用于评估大型语言模型的菲律宾自然语言处理基准 

**Authors**: Jann Railey Montalan, Jimson Paulo Layacan, David Demitri Africa, Richell Isaiah Flores, Michael T. Lopez II, Theresa Denise Magsajo, Anjanette Cayabyab, William Chandra Tjhi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14911)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable capabilities on widely benchmarked high-resource languages; however, linguistic nuances of under-resourced languages remain unexplored. We introduce Batayan, a holistic Filipino benchmark designed to systematically evaluate LLMs across three key natural language processing (NLP) competencies: understanding, reasoning, and generation. Batayan consolidates eight tasks, covering both Tagalog and code-switched Taglish utterances. Our rigorous, native-speaker-driven annotation process ensures fluency and authenticity to the complex morphological and syntactic structures of Filipino, alleviating a pervasive translationese bias in existing Filipino corpora. We report empirical results on a variety of multilingual LLMs, highlighting significant performance gaps that signal the under-representation of Filipino in pretraining corpora, the unique hurdles in modeling Filipino's rich morphology and construction, and the importance of explicit Filipino language support and instruction tuning. Moreover, we discuss the practical challenges encountered in dataset construction and propose principled solutions for building culturally and linguistically-faithful resources in under-represented languages. We also provide a public benchmark and leaderboard as a clear foundation for iterative, community-driven progress in Filipino NLP. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）的进步在广泛benchmark的高资源语言上展现了显著的能力；然而，欠资源语言的语义细微差别仍需探索。我们介绍了Batayan，一个综合性的菲律滨语基准，旨在系统性地评估LLMs在自然语言处理（NLP）三大核心能力：理解、推理和生成上的表现。Batayan整合了八个任务，涵盖了塔加洛语及其混用塔古丽夏语表达。我们严格且以母语者驱动的注释过程确保了注释内容在复杂的形态和句法结构上具有流畅性和真实性，从而减轻现有菲律滨语语料库中的普遍翻译倾向偏差。我们在多种多语言LLMs上报告了实证结果，强调了菲律滨语在预训练语料库中的代表性不足、建模菲律滨语丰富形态与构造的独特挑战以及明示菲律滨语语言支持和指令调优的重要性。此外，我们讨论了在数据集构建过程中遇到的实践挑战，并提出了构建文化上和语言上忠实资源的原理性解决方案，以支持欠代表语言的发展。我们还提供了一个公开基准和排行榜，作为菲律滨语NLP持续社区驱动进展的明确基础。 

---
# EvoP: Robust LLM Inference via Evolutionary Pruning 

**Title (ZH)**: EvoP：演化裁剪实现 robust LLM 推理 

**Authors**: Shangyu Wu, Hongchao Du, Ying Xiong, Shuai Chen, Tei-wei Kuo, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.14910)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in natural language processing tasks, but their massive size and computational demands hinder their deployment in resource-constrained environments. Existing structured pruning methods address this issue by removing redundant structures (e.g., elements, channels, layers) from the model. However, these methods employ a heuristic pruning strategy, which leads to suboptimal performance. Besides, they also ignore the data characteristics when pruning the model.
To overcome these limitations, we propose EvoP, an evolutionary pruning framework for robust LLM inference. EvoP first presents a cluster-based calibration dataset sampling (CCDS) strategy for creating a more diverse calibration dataset. EvoP then introduces an evolutionary pruning pattern searching (EPPS) method to find the optimal pruning pattern. Compared to existing structured pruning techniques, EvoP achieves the best performance while maintaining the best efficiency. Experiments across different LLMs and different downstream tasks validate the effectiveness of the proposed EvoP, making it a practical and scalable solution for deploying LLMs in real-world applications. 

**Abstract (ZH)**: EvoP：一种用于 robust LLM 推断的演化剪枝框架 

---
# PTB-Image: A Scanned Paper ECG Dataset for Digitization and Image-based Diagnosis 

**Title (ZH)**: PTB-Image: 一种扫描纸张心电图数据集，用于数字化和图像诊断 

**Authors**: Cuong V. Nguyen, Hieu X. Nguyen, Dung D. Pham Minh, Cuong D. Do  

**Link**: [PDF](https://arxiv.org/pdf/2502.14909)  

**Abstract**: Electrocardiograms (ECGs) recorded on paper remain prevalent in clinical practice, yet their use presents challenges for automated analysis and digital storage. To address this issue, we introduce PTB-Image, a dataset comprising scanned paper ECGs with corresponding digital signals, enabling research on ECG digitization. We also provide VinDigitizer, a digitization baseline to convert paper-based ECGs into digital time-series signals. The method involves detecting signal rows, extracting waveforms from the background, and reconstructing numerical values from the digitized traces. We applied VinDigitizer to 549 scanned ECGs and evaluated its performance against the original PTB dataset (modified to match the printed signals). The results achieved a mean signal-to-noise ratio (SNR) of 0.01 dB, highlighting both the feasibility and challenges of ECG digitization, particularly in mitigating distortions from printing and scanning processes. By providing PTB-Image and baseline digitization methods, this work aims to facilitate advancements in ECG digitization, enhancing access to historical ECG data and supporting applications in telemedicine and automated cardiac diagnostics. 

**Abstract (ZH)**: 纸质心电图（ECGs）记录在临床实践中仍然广泛使用，但其使用为自动分析和数字化存储带来了挑战。为解决这一问题，我们介绍了PTB-Image数据集，该数据集包含扫描的纸质ECGs及其对应的数字信号，以促进ECG数字化研究。我们还提供了VinDigitizer，一种基线数字化方法，用于将纸质ECGs转换为数字时间序列信号。该方法包括检测信号行、从背景中提取波形以及从数字化轨迹中重构数值。我们将VinDigitizer应用于549份扫描的心电图，并将其性能与修改后的原PTB数据集进行了评估，结果获得了0.01 dB的平均信噪比（SNR），突显了ECG数字化的可行性和挑战，特别是在减少印刷和扫描过程中的失真方面。通过提供PTB-Image和基线数字化方法，本工作旨在促进ECG数字化的发展，提高历史ECG数据的可访问性，并支持远程医疗服务和自动化心脏诊断的应用。 

---
# KOALA: Knowledge Conflict Augmentations for Robustness in Vision Language Models 

**Title (ZH)**: KOALA: 知识冲突增强for 视觉语言模型的鲁棒性 

**Authors**: Peter Carragher, Nikitha Rao, Abhinand Jha, R Raghav, Kathleen M. Carley  

**Link**: [PDF](https://arxiv.org/pdf/2502.14908)  

**Abstract**: The robustness of large language models (LLMs) against knowledge conflicts in unimodal question answering systems has been well studied. However, the effect of conflicts in information sources on vision language models (VLMs) in multimodal settings has not yet been explored. In this work, we propose \segsub, a framework that applies targeted perturbations to image sources to study and improve the robustness of VLMs against three different types of knowledge conflicts, namely parametric, source, and counterfactual conflicts. Contrary to prior findings that showed that LLMs are sensitive to parametric conflicts arising from textual perturbations, we find VLMs are largely robust to image perturbation. On the other hand, VLMs perform poorly on counterfactual examples (<30% accuracy) and fail to reason over source conflicts (<1% accuracy). We also find a link between hallucinations and image context, with GPT-4o prone to hallucination when presented with highly contextualized counterfactual examples. While challenges persist with source conflicts, finetuning models significantly improves reasoning over counterfactual samples. Our findings highlight the need for VLM training methodologies that enhance their reasoning capabilities, particularly in addressing complex knowledge conflicts between multimodal sources. 

**Abstract (ZH)**: 大语言模型（LLMs）在单模态问答系统中对抗知识冲突的稳健性已得到充分研究，但在多模态设置中信息源冲突对视觉语言模型（VLMs）的影响尚未被探索。在此工作中，我们提出了\segsub框架，通过对图像源应用目标化扰动来研究和改进VLMs在三种不同类型知识冲突（参数冲突、来源冲突和反事实冲突）下的稳健性。与先有发现不同，我们发现VLMs对来自图像扰动的参数冲突不敏感，表现出较大的稳健性。另一方面，VLMs在反事实示例上表现不佳（准确率<30%）并在来源冲突下几乎无法推理（准确率<1%）。我们还发现幻觉与图像上下文之间的关联，当GPT-4o面对高度上下文化的反事实示例时容易产生幻觉。虽然在来源冲突方面仍存在挑战，但模型微调显著提高了对反事实样本的推理能力。我们的发现强调了增强VLM推理能力的训练方法的需求，特别是在解决多模态来源之间的复杂知识冲突方面。 

---
# GneissWeb: Preparing High Quality Data for LLMs at Scale 

**Title (ZH)**: GneissWeb：为大规模生成模型准备高质量数据 

**Authors**: Hajar Emami Gohari, Swanand Ravindra Kadhe, Syed Yousaf Shah. Constantin Adam, Abdulhamid Adebayo, Praneet Adusumilli, Farhan Ahmed, Nathalie Baracaldo Angel, Santosh Borse, Yuan-Chi Chang, Xuan-Hong Dang, Nirmit Desai, Ravital Eres, Ran Iwamoto, Alexei Karve, Yan Koyfman, Wei-Han Lee, Changchang Liu, Boris Lublinsky, Takuyo Ohko, Pablo Pesce, Maroun Touma, Shiqiang Wang, Shalisha Witherspoon, Herbert Woisetschlager, David Wood, Kun-Lung Wu, Issei Yoshida, Syed Zawad, Petros Zerfos, Yi Zhou, Bishwaranjan Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2502.14907)  

**Abstract**: Data quantity and quality play a vital role in determining the performance of Large Language Models (LLMs). High-quality data, in particular, can significantly boost the LLM's ability to generalize on a wide range of downstream tasks. Large pre-training datasets for leading LLMs remain inaccessible to the public, whereas many open datasets are small in size (less than 5 trillion tokens), limiting their suitability for training large models.
In this paper, we introduce GneissWeb, a large dataset yielding around 10 trillion tokens that caters to the data quality and quantity requirements of training LLMs. Our GneissWeb recipe that produced the dataset consists of sharded exact sub-string deduplication and a judiciously constructed ensemble of quality filters. GneissWeb achieves a favorable trade-off between data quality and quantity, producing models that outperform models trained on state-of-the-art open large datasets (5+ trillion tokens).
We show that models trained using GneissWeb dataset outperform those trained on FineWeb-V1.1.0 by 2.73 percentage points in terms of average score computed on a set of 11 commonly used benchmarks (both zero-shot and few-shot) for pre-training dataset evaluation. When the evaluation set is extended to 20 benchmarks (both zero-shot and few-shot), models trained using GneissWeb still achieve a 1.75 percentage points advantage over those trained on FineWeb-V1.1.0. 

**Abstract (ZH)**: 大数据量和高质量数据在网络语言模型（LLM）性能中起着至关重要的作用。特别是高质量数据可以显著提升LLM在各种下游任务上的泛化能力。领先的LLM的大规模预训练数据集对公众而言仍然难以获取，而许多开源数据集规模较小（少于5万亿个词元），限制了其用于训练大规模模型的适用性。

在本文中，我们介绍了GneissWeb，一个包含约10万亿个词元的大规模数据集，满足训练LLM的数据质量和数量要求。GneissWeb数据集的生成方法包括分片精确子字符串去重和精心构建的质量过滤器集合。GneissWeb在数据质量和数量之间实现了良好的权衡，生成的模型在使用状态最先进开源大规模数据集（5+万亿个词元）训练的模型中表现出色。

研究表明，使用GneissWeb数据集训练的模型在一套11个常用预训练数据集评估基准（包括零样本和少样本）上的平均得分上，比使用FineWeb-V1.1.0训练的模型高出2.73个百分点。当评估集扩展到20个基准（包括零样本和少样本）时，使用GneissWeb训练的模型仍然比使用FineWeb-V1.1.0训练的模型具有1.75个百分点的优势。 

---
# Beyond Words: Exploring Cultural Value Sensitivity in Multimodal Models 

**Title (ZH)**: 超越文字：探索多模态模型中的文化价值敏感性 

**Authors**: Srishti Yadav, Zhi Zhang, Daniel Hershcovich, Ekaterina Shutova  

**Link**: [PDF](https://arxiv.org/pdf/2502.14906)  

**Abstract**: Investigating value alignment in Large Language Models (LLMs) based on cultural context has become a critical area of research. However, similar biases have not been extensively explored in large vision-language models (VLMs). As the scale of multimodal models continues to grow, it becomes increasingly important to assess whether images can serve as reliable proxies for culture and how these values are embedded through the integration of both visual and textual data. In this paper, we conduct a thorough evaluation of multimodal model at different scales, focusing on their alignment with cultural values. Our findings reveal that, much like LLMs, VLMs exhibit sensitivity to cultural values, but their performance in aligning with these values is highly context-dependent. While VLMs show potential in improving value understanding through the use of images, this alignment varies significantly across contexts highlighting the complexities and underexplored challenges in the alignment of multimodal models. 

**Abstract (ZH)**: 基于文化背景探究大型语言模型的价值对齐已成为一个重要研究领域。然而，类似偏见在大型视觉-语言模型(VLMs)中的研究尚不充分。随着多模态模型规模的不断扩大，评估图像是否能可靠地代表文化以及这些价值是如何通过视觉和文本数据的结合而嵌入变得尤为重要。在本文中，我们针对不同规模的多模态模型进行了详细的评估，重点关注它们与文化价值的对齐情况。我们的研究发现，与大型语言模型(LLMs)类似，视觉-语言模型也表现出对文化价值的敏感性，但它们在这些价值上的表现高度依赖于上下文。尽管视觉-语言模型通过使用图像提高价值理解具有潜力，但这种对齐在不同上下文中的差异性揭示了多模态模型对齐中的复杂性和未被充分探索的挑战。 

---
# Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence 

**Title (ZH)**: 在JSON中思考：严格遵守LLM架构策略的强化学习方法 

**Authors**: Bhavik Agarwal, Ishan Joshi, Viktoria Rojkova  

**Link**: [PDF](https://arxiv.org/pdf/2502.14905)  

**Abstract**: In this paper, we address the challenge of enforcing strict schema adherence in large language model (LLM) generation by leveraging LLM reasoning capabilities. Building on the DeepSeek R1 reinforcement learning framework, our approach trains structured reasoning skills of a 1.5B parameter model through a novel pipeline that combines synthetic reasoning dataset construction with custom reward functions under Group Relative Policy Optimization (GRPO). Specifically, we first perform R1 reinforcement learning on a 20K sample unstructured-to-structured dataset, mirroring the original DeepSeek R1 methods, to establish core reasoning abilities. Subsequently, we performed supervised fine-tuning on a separate 10K reasoning sample dataset, focusing on refining schema adherence for downstream tasks. Despite the relatively modest training scope, requiring approximately 20 hours on an 8xH100 GPU cluster for GRPO training and 3 hours on 1xA100 for SFT, our model demonstrates robust performance in enforcing schema consistency. We compare our ThinkJSON approach against the original DeepSeek R1 (671B), distilled versions of DeepSeek R1 (Qwen-1.5B and Qwen-7B), and Gemini 2.0 Flash (70B), showcasing its effectiveness in real-world applications. Our results underscore the practical utility of a resource-efficient framework for schema-constrained text generation. 

**Abstract (ZH)**: 在本论文中，我们通过利用大语言模型的推理能力，解决了在大语言模型生成中严格遵循模式规范的挑战。基于DeepSeek R1强化学习框架，我们的方法通过一种新颖的管道流程来训练一个包含1.5亿参数的模型的结构化推理技能，该流程结合了合成推理数据集构建与定制的奖励函数，并在Group Relative Policy Optimization (GRPO) 下运行。具体来说，我们首先在20K样本的无结构到结构化数据集上进行R1强化学习，模仿原始的DeepSeek R1方法，以建立核心推理能力。随后，我们在一个独立的10K推理样本数据集上进行了监督微调，专注于改进下游任务中的模式遵守能力。尽管训练范围相对有限，GRPO训练大约需要8xH100 GPU集群20小时，SFT训练在1xA100上大约需要3小时，我们的模型在强制执行模式一致性方面显示出稳健的性能。我们将我们的ThinkJSON方法与原始的DeepSeek R1 (671B)、DeepSeek R1的蒸馏版本 (Qwen-1.5B 和 Qwen-7B) 以及Gemini 2.0 Flash (70B) 进行了比较，展示了其在实际应用中的有效性。我们的结果强调了资源高效框架在受限模式文本生成中的实用价值。 

---
# PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths 

**Title (ZH)**: 基于关系路径的图表示检索增强生成剪枝方法 

**Authors**: Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu, Chuan Shi, Cheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14902)  

**Abstract**: Retrieval-augmented generation (RAG) improves the response quality of large language models (LLMs) by retrieving knowledge from external databases. Typical RAG approaches split the text database into chunks, organizing them in a flat structure for efficient searches. To better capture the inherent dependencies and structured relationships across the text database, researchers propose to organize textual information into an indexing graph, known asgraph-based RAG. However, we argue that the limitation of current graph-based RAG methods lies in the redundancy of the retrieved information, rather than its insufficiency. Moreover, previous methods use a flat structure to organize retrieved information within the prompts, leading to suboptimal performance. To overcome these limitations, we propose PathRAG, which retrieves key relational paths from the indexing graph, and converts these paths into textual form for prompting LLMs. Specifically, PathRAG effectively reduces redundant information with flow-based pruning, while guiding LLMs to generate more logical and coherent responses with path-based prompting. Experimental results show that PathRAG consistently outperforms state-of-the-art baselines across six datasets and five evaluation dimensions. The code is available at the following link: this https URL 

**Abstract (ZH)**: 基于路径的检索增强生成（PathRAG）通过从索引图中检索关键关系路径并将其转换为文本形式来优化大型语言模型的响应质量 

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
# NOTA: Multimodal Music Notation Understanding for Visual Large Language Model 

**Title (ZH)**: 多模态音乐符号理解对于视觉大语言模型 

**Authors**: Mingni Tang, Jiajia Li, Lu Yang, Zhiqiang Zhang, Jinghao Tian, Zuchao Li, Lefei Zhang, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14893)  

**Abstract**: Symbolic music is represented in two distinct forms: two-dimensional, visually intuitive score images, and one-dimensional, standardized text annotation sequences. While large language models have shown extraordinary potential in music, current research has primarily focused on unimodal symbol sequence text. Existing general-domain visual language models still lack the ability of music notation understanding. Recognizing this gap, we propose NOTA, the first large-scale comprehensive multimodal music notation dataset. It consists of 1,019,237 records, from 3 regions of the world, and contains 3 tasks. Based on the dataset, we trained NotaGPT, a music notation visual large language model. Specifically, we involve a pre-alignment training phase for cross-modal alignment between the musical notes depicted in music score images and their textual representation in ABC notation. Subsequent training phases focus on foundational music information extraction, followed by training on music notation analysis. Experimental results demonstrate that our NotaGPT-7B achieves significant improvement on music understanding, showcasing the effectiveness of NOTA and the training pipeline. Our datasets are open-sourced at this https URL. 

**Abstract (ZH)**: 符号音乐以两种不同的形式表示：二维、直观的乐谱图像和一维、标准化的文字注释序列。虽然大型语言模型在音乐领域展现了非凡的潜力，但现有研究主要集中在单一符号序列文本上。现有的通用视觉语言模型在音乐符号理解方面仍缺乏能力。认识到这一缺口，我们提出了NOTA，这是首个大规模综合多模态音乐符号数据集，包含1,019,237条记录，来自世界三个地区，并包含3项任务。基于此数据集，我们训练了Notagpt，这是一种音乐符号视觉大型语言模型。特别地，我们通过一个预对齐训练阶段实现音乐得分图像中表示的音符与其ABC符号表示之间的跨模态对齐。后续训练阶段专注于基础音乐信息提取，随后进行音乐符号分析。实验结果表明，我们的Notagpt-7B在音乐理解方面取得了显著改进，展示了NOTA和训练管道的有效性。我们的数据集在此处开源<https://>。 

---
# EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild 

**Title (ZH)**: EgoSpeak: 学习自中心对话代理在真实世界中何时发言 

**Authors**: Junhyeok Kim, Min Soo Kim, Jiwan Chung, Jungbin Cho, Jisoo Kim, Sungwoong Kim, Gyeongbo Sim, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14892)  

**Abstract**: Predicting when to initiate speech in real-world environments remains a fundamental challenge for conversational agents. We introduce EgoSpeak, a novel framework for real-time speech initiation prediction in egocentric streaming video. By modeling the conversation from the speaker's first-person viewpoint, EgoSpeak is tailored for human-like interactions in which a conversational agent must continuously observe its environment and dynamically decide when to talk. Our approach bridges the gap between simplified experimental setups and complex natural conversations by integrating four key capabilities: (1) first-person perspective, (2) RGB processing, (3) online processing, and (4) untrimmed video processing. We also present YT-Conversation, a diverse collection of in-the-wild conversational videos from YouTube, as a resource for large-scale pretraining. Experiments on EasyCom and Ego4D demonstrate that EgoSpeak outperforms random and silence-based baselines in real time. Our results also highlight the importance of multimodal input and context length in effectively deciding when to speak. 

**Abstract (ZH)**: 在真实环境中的语音发起预测仍然是对话代理的基本挑战。我们介绍了一种新颖的框架EgoSpeak，用于自视点流式视频中的实时语音发起预测。通过从说话人的第一人称视角建模对话，EgoSpeak专为人类般的交互设计，其中对话代理必须不断观察环境并在适当时机决定是否发言。我们的方法通过结合四种关键能力，弥合了简化实验设置与复杂自然对话之间的差距：（1）第一人称视角，（2）RGB处理，（3）在线处理，以及（4）未剪辑视频处理。我们还介绍了YT-Conversation，这是一个来自YouTube的多元化的在野对话视频集合，作为大规模预训练的资源。在EasyCom和Ego4D上的实验表明，EgoSpeak在实时性能上优于随机和静默基线。我们的结果还强调了多模态输入和上下文长度在有效决定何时发言中的重要性。 

---
# CoDiff: Conditional Diffusion Model for Collaborative 3D Object Detection 

**Title (ZH)**: CoDiff：条件扩散模型在协作3D对象检测中的应用 

**Authors**: Zhe Huang, Shuo Wang, Yongcai Wang, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14891)  

**Abstract**: Collaborative 3D object detection holds significant importance in the field of autonomous driving, as it greatly enhances the perception capabilities of each individual agent by facilitating information exchange among multiple agents. However, in practice, due to pose estimation errors and time delays, the fusion of information across agents often results in feature representations with spatial and temporal noise, leading to detection errors. Diffusion models naturally have the ability to denoise noisy samples to the ideal data, which motivates us to explore the use of diffusion models to address the noise problem between multi-agent systems. In this work, we propose CoDiff, a novel robust collaborative perception framework that leverages the potential of diffusion models to generate more comprehensive and clearer feature representations. To the best of our knowledge, this is the first work to apply diffusion models to multi-agent collaborative perception. Specifically, we project high-dimensional feature map into the latent space of a powerful pre-trained autoencoder. Within this space, individual agent information serves as a condition to guide the diffusion model's sampling. This process denoises coarse feature maps and progressively refines the fused features. Experimental study on both simulated and real-world datasets demonstrates that the proposed framework CoDiff consistently outperforms existing relevant methods in terms of the collaborative object detection performance, and exhibits highly desired robustness when the pose and delay information of agents is with high-level noise. 

**Abstract (ZH)**: 协作三维物体检测在自动驾驶领域具有重要意义，它通过促进多个代理之间的信息交换，极大地增强了每个个体代理的感知能力。然而，在实践中，由于姿态估计误差和时间延迟，代理之间信息融合往往会导致带有空间和时间噪声的特征表示，从而导致检测错误。扩散模型天然具有将嘈杂样本去噪至理想数据的能力，这激发了我们探索使用扩散模型解决多代理系统之间的噪声问题。在本文中，我们提出了一种新颖的鲁棒协作感知框架CoDiff，该框架利用扩散模型的潜力生成更加全面和清晰的特征表示。据我们所知，这是首次将扩散模型应用于多代理协作感知。具体地，我们将高维特征图投影到强大预训练自编码器的潜在空间中，在该空间中，个体代理信息作为条件引导扩散模型的采样过程。此过程去噪粗糙特征图并逐步细化融合特征。在模拟数据集和真实世界数据集上的实验研究证明，所提出的框架CoDiff在协作物体检测性能上始终优于现有相关方法，并且在代理的姿态和延迟信息带有高水平噪声时表现出高度期望的鲁棒性。 

---
# Narrowing Information Bottleneck Theory for Multimodal Image-Text Representations Interpretability 

**Title (ZH)**: 窄信息瓶颈理论在多模态图像-文本表征可解释性中的应用 

**Authors**: Zhiyu Zhu, Zhibo Jin, Jiayu Zhang, Nan Yang, Jiahao Huang, Jianlong Zhou, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.14889)  

**Abstract**: The task of identifying multimodal image-text representations has garnered increasing attention, particularly with models such as CLIP (Contrastive Language-Image Pretraining), which demonstrate exceptional performance in learning complex associations between images and text. Despite these advancements, ensuring the interpretability of such models is paramount for their safe deployment in real-world applications, such as healthcare. While numerous interpretability methods have been developed for unimodal tasks, these approaches often fail to transfer effectively to multimodal contexts due to inherent differences in the representation structures. Bottleneck methods, well-established in information theory, have been applied to enhance CLIP's interpretability. However, they are often hindered by strong assumptions or intrinsic randomness. To overcome these challenges, we propose the Narrowing Information Bottleneck Theory, a novel framework that fundamentally redefines the traditional bottleneck approach. This theory is specifically designed to satisfy contemporary attribution axioms, providing a more robust and reliable solution for improving the interpretability of multimodal models. In our experiments, compared to state-of-the-art methods, our approach enhances image interpretability by an average of 9%, text interpretability by an average of 58.83%, and accelerates processing speed by 63.95%. Our code is publicly accessible at this https URL. 

**Abstract (ZH)**: 多模态图像-文本表示的识别任务引起了越来越多的关注，特别是CLIP（对比语言图像预训练）等模型在学习图像和文本之间复杂关联方面表现出色。尽管取得了这些进步，确保这些模型的安全可解释性对于其在医疗保健等实际应用中的部署至关重要。尽管已经为单模态任务开发了多种可解释性方法，但由于表示结构的固有差异，这些方法往往难以有效转移到多模态上下文。信息理论中成熟的瓶颈方法已被应用于提高CLIP的可解释性，但这些方法经常受到强烈假设或内在随机性的阻碍。为克服这些挑战，我们提出了信息瓶颈压缩理论这一新框架，从根本上重新定义了传统瓶颈方法。该理论专门为符合当前归因公理设计，为提高多模态模型的可解释性提供了更稳健且可靠的方法。在我们的实验中，与最先进的方法相比，我们的方法平均提高了图象可解释性9%，文本可解释性58.83%，并加速了处理速度63.95%。我们的代码可在以下网址公开访问：这个 https URL。 

---
# The Multi-Faceted Monosemanticity in Multimodal Representations 

**Title (ZH)**: 多模态表示中的多面向单义性 

**Authors**: Hanqi Yan, Xiangxiang Cui, Lu Yin, Paul Pu Liang, Yulan He, Yifei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14888)  

**Abstract**: In this paper, we leverage recent advancements in feature monosemanticity to extract interpretable features from deep multimodal models, offering a data-driven understanding of modality gaps. Specifically, we investigate CLIP (Contrastive Language-Image Pretraining), a prominent visual-language representation model trained on extensive image-text pairs. Building upon interpretability tools developed for single-modal models, we extend these methodologies to assess multi-modal interpretability of CLIP features. Additionally, we introduce the Modality Dominance Score (MDS) to attribute the interpretability of each feature to its respective modality. Next, we transform CLIP features into a more interpretable space, enabling us to categorize them into three distinct classes: vision features (single-modal), language features (single-modal), and visual-language features (cross-modal). Our findings reveal that this categorization aligns closely with human cognitive understandings of different modalities. We also demonstrate significant use cases of this modality-specific features including detecting gender bias, adversarial attack defense and text-to-image model editing. These results indicate that large-scale multimodal models, equipped with task-agnostic interpretability tools, offer valuable insights into key connections and distinctions between different modalities. 

**Abstract (ZH)**: 利用特征单义性从深度多模态模型中提取可解释特征，探究模态差距的数据驱动理解：CLIP的多模态可解释性分析与模态主导得分应用 

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
# KKA: Improving Vision Anomaly Detection through Anomaly-related Knowledge from Large Language Models 

**Title (ZH)**: KKA：通过大型语言模型中的异常相关知识改进视觉异常检测 

**Authors**: Dong Chen, Zhengqing Hu, Peiguang Fan, Yueting Zhuang, Yafei Li, Qidong Liu, Xiaoheng Jiang, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14880)  

**Abstract**: Vision anomaly detection, particularly in unsupervised settings, often struggles to distinguish between normal samples and anomalies due to the wide variability in anomalies. Recently, an increasing number of studies have focused on generating anomalies to help detectors learn more effective boundaries between normal samples and anomalies. However, as the generated anomalies are often derived from random factors, they frequently lack realism. Additionally, randomly generated anomalies typically offer limited support in constructing effective boundaries, as most differ substantially from normal samples and lie far from the boundary. To address these challenges, we propose Key Knowledge Augmentation (KKA), a method that extracts anomaly-related knowledge from large language models (LLMs). More specifically, KKA leverages the extensive prior knowledge of LLMs to generate meaningful anomalies based on normal samples. Then, KKA classifies the generated anomalies as easy anomalies and hard anomalies according to their similarity to normal samples. Easy anomalies exhibit significant differences from normal samples, whereas hard anomalies closely resemble normal samples. KKA iteratively updates the generated anomalies, and gradually increasing the proportion of hard anomalies to enable the detector to learn a more effective boundary. Experimental results show that the proposed method significantly improves the performance of various vision anomaly detectors while maintaining low generation costs. The code for CMG can be found at this https URL. 

**Abstract (ZH)**: 基于关键知识增强的视觉异常检测方法 

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
# d-Sketch: Improving Visual Fidelity of Sketch-to-Image Translation with Pretrained Latent Diffusion Models without Retraining 

**Title (ZH)**: d-Sketch: 使用预训练的潜扩散模型提高从素描到图像转换的视觉保真度而不重新训练 

**Authors**: Prasun Roy, Saumik Bhattacharya, Subhankar Ghosh, Umapada Pal, Michael Blumenstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.14007)  

**Abstract**: Structural guidance in an image-to-image translation allows intricate control over the shapes of synthesized images. Generating high-quality realistic images from user-specified rough hand-drawn sketches is one such task that aims to impose a structural constraint on the conditional generation process. While the premise is intriguing for numerous use cases of content creation and academic research, the problem becomes fundamentally challenging due to substantial ambiguities in freehand sketches. Furthermore, balancing the trade-off between shape consistency and realistic generation contributes to additional complexity in the process. Existing approaches based on Generative Adversarial Networks (GANs) generally utilize conditional GANs or GAN inversions, often requiring application-specific data and optimization objectives. The recent introduction of Denoising Diffusion Probabilistic Models (DDPMs) achieves a generational leap for low-level visual attributes in general image synthesis. However, directly retraining a large-scale diffusion model on a domain-specific subtask is often extremely difficult due to demanding computation costs and insufficient data. In this paper, we introduce a technique for sketch-to-image translation by exploiting the feature generalization capabilities of a large-scale diffusion model without retraining. In particular, we use a learnable lightweight mapping network to achieve latent feature translation from source to target domain. Experimental results demonstrate that the proposed method outperforms the existing techniques in qualitative and quantitative benchmarks, allowing high-resolution realistic image synthesis from rough hand-drawn sketches. 

**Abstract (ZH)**: 结构引导在图像到图像转换中允许对合成图像形状的精细控制。基于生成式对抗网络的条件生成过程旨在从用户指定的手绘草图生成高质量逼真图像，这一任务施加了结构性约束。尽管这一前提在内容创作和学术研究的诸多应用场景中颇具吸引力，但由于手绘草图存在大量模糊性，问题变得从根本上具有挑战性。形状一致性与逼真生成之间的trade-off平衡进一步增加了过程的复杂性。现有的基于生成式对抗网络（GANs）的方法通常使用条件GAN或GAN逆运算，往往需要应用特定的数据和优化目标。最近提出的去噪扩散概率模型（DDPMs）在一般图像合成中的低级视觉属性生成方面实现了飞跃性进步。然而，直接在特定领域的小任务上训练大规模扩散模型通常由于计算成本高和数据不足而极其困难。在本文中，我们提出了一种技术，在不重新训练大规模扩散模型的情况下利用其特征泛化能力实现草图到图像的转换。特别是，我们使用一个可学习的轻量级映射网络实现从源域到目标域的潜在特征转换。实验结果表明，所提出的方法在定性和定量基准上优于现有技术，能够从粗糙的手绘草图生成高分辨率的逼真图像。 

---
# High Quality Segmentation for Ultra High-resolution Images 

**Title (ZH)**: 超高清图像的高质分割 

**Authors**: Tiancheng Shen, Yuechen Zhang, Lu Qi, Jason Kuen, Xingyu Xie, Jianlong Wu, Zhe Lin, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2111.14482)  

**Abstract**: To segment 4K or 6K ultra high-resolution images needs extra computation consideration in image segmentation. Common strategies, such as down-sampling, patch cropping, and cascade model, cannot address well the balance issue between accuracy and computation cost. Motivated by the fact that humans distinguish among objects continuously from coarse to precise levels, we propose the Continuous Refinement Model~(CRM) for the ultra high-resolution segmentation refinement task. CRM continuously aligns the feature map with the refinement target and aggregates features to reconstruct these images' details. Besides, our CRM shows its significant generalization ability to fill the resolution gap between low-resolution training images and ultra high-resolution testing ones. We present quantitative performance evaluation and visualization to show that our proposed method is fast and effective on image segmentation refinement. Code will be released at this https URL. 

**Abstract (ZH)**: 超高清图像分割需要额外的计算考虑。常规策略如下采样、局部裁剪和级联模型无法很好地解决准确性和计算成本之间的平衡问题。受人类从粗略到精细不断区分物体的启发，我们提出了连续精炼模型（Continuous Refinement Model，CRM）用于超高清分割精炼任务。CRM连续地对特征图与精炼目标进行对齐，并聚合特征以重建这些图像的细节。此外，我们的CRM展示了其显著的泛化能力，可以在低分辨率训练图像与超高清测试图像之间填补分辨率差距。我们通过定量性能评估和可视化展示了所提出方法在图像分割精炼任务上的快速高效性。代码将在以下链接发布：此 https URL。 

---
