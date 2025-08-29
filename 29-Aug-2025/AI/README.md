# ChatThero: An LLM-Supported Chatbot for Behavior Change and Therapeutic Support in Addiction Recovery 

**Title (ZH)**: ChatThero: 一个支持行为改变和成瘾康复治疗支持的大型语言模型聊天机器人 

**Authors**: Junda Wang, Zonghai Yao, Zhichao Yang, Lingxi Li, Junhui Qian, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20996)  

**Abstract**: Substance use disorders (SUDs) affect over 36 million people worldwide, yet few receive effective care due to stigma, motivational barriers, and limited personalized support. Although large language models (LLMs) show promise for mental-health assistance, most systems lack tight integration with clinically validated strategies, reducing effectiveness in addiction recovery. We present ChatThero, a multi-agent conversational framework that couples dynamic patient modeling with context-sensitive therapeutic dialogue and adaptive persuasive strategies grounded in cognitive behavioral therapy (CBT) and motivational interviewing (MI). We build a high-fidelity synthetic benchmark spanning Easy, Medium, and Hard resistance levels, and train ChatThero with a two-stage pipeline comprising supervised fine-tuning (SFT) followed by direct preference optimization (DPO). In evaluation, ChatThero yields a 41.5\% average gain in patient motivation, a 0.49\% increase in treatment confidence, and resolves hard cases with 26\% fewer turns than GPT-4o, and both automated and human clinical assessments rate it higher in empathy, responsiveness, and behavioral realism. The framework supports rigorous, privacy-preserving study of therapeutic conversation and provides a robust, replicable basis for research and clinical translation. 

**Abstract (ZH)**: 物质使用障碍（SUDs）影响着世界上的逾3600万人，但由于 stigma、动机障碍以及有限的个性化支持，很少有人能够获得有效的治疗。尽管大型语言模型（LLMs）显示出在心理健康辅助方面的潜力，但大多数系统缺乏与临床验证策略的紧密整合，从而降低了在戒瘾康复中的有效性。我们提出了一种多智能体对话框架——ChatThero，该框架结合了动态患者建模、语境敏感的治疗对话以及基于认知行为疗法（CBT）和动机访谈（MI）的适应性说服策略。我们构建了一个高保真合成基准，涵盖易、中、难三个级别的抵抗水平，并使用两阶段训练管道——监督微调（SFT）后接直接偏好优化（DPO）来训练ChatThero。在评估中，ChatThero在患者动机方面平均提高了41.5%，治疗信心提高了0.49%，在解决困难案例时减少了26%的对话轮次，并且无论是自动化还是人类临床评估都将其在同理心、响应性和行为真实性方面的评分更高。该框架支持严格的、保护隐私的治疗性对话研究，并为研究和临床转化提供了坚实、可复制的基础。 

---
# Efficient Neuro-Symbolic Learning of Constraints and Objective 

**Title (ZH)**: 高效神经符号学习约束与目标 

**Authors**: Marianne Defresne, Romain Gambardella, Sophie Barbe, Thomas Schiex  

**Link**: [PDF](https://arxiv.org/pdf/2508.20978)  

**Abstract**: In the ongoing quest for hybridizing discrete reasoning with neural nets, there is an increasing interest in neural architectures that can learn how to solve discrete reasoning or optimization problems from natural inputs, a task that Large Language Models seem to struggle with.
Objectives: We introduce a differentiable neuro-symbolic architecture and a loss function dedicated to learning how to solve NP-hard reasoning problems.
Methods: Our new probabilistic loss allows for learning both the constraints and the objective, thus delivering a complete model that can be scrutinized and completed with side constraints. By pushing the combinatorial solver out of the training loop, our architecture also offers scalable training while exact inference gives access to maximum accuracy.
Results: We empirically show that it can efficiently learn how to solve NP-hard reasoning problems from natural inputs. On three variants of the Sudoku benchmark -- symbolic, visual, and many-solution --, our approach requires a fraction of training time of other hybrid methods. On a visual Min-Cut/Max-cut task, it optimizes the regret better than a Decision-Focused-Learning regret-dedicated loss. Finally, it efficiently learns the energy optimization formulation of the large real-world problem of designing proteins. 

**Abstract (ZH)**: 在离散推理与神经网络结合的不断探索中，人们对能够从自然输入中学习解决离散推理或优化问题的神经架构越来越感兴趣，而大型语言模型似乎在这方面遇到困难。
目标：我们提出了一种可微神经符号架构和一个专门用于学习解决NP难推理问题的损失函数。
方法：我们新的概率损失函数允许同时学习约束条件和目标，从而提供一个可以审查和补充侧约束的完整模型。通过将组合式求解器从训练循环中移除，该架构还提供了可扩展的训练方法，而精确推断则提供了最大限度的准确性。
结果：我们实验证明，该方法能够高效地从自然输入中学习解决NP难推理问题。在三个版本的数独基准测试中——符号版、视觉版和多解版——我们的方法所需训练时间仅为其他混合方法的一小部分。在视觉最小割/最大割任务中，它比一种决策导向学习的后悔专用损失更好地优化了后悔值。最后，它有效地学习了设计蛋白质这一大规模实际问题的能量优化公式。 

---
# A Multi-Objective Genetic Algorithm for Healthcare Workforce Scheduling 

**Title (ZH)**: 多目标遗传算法在医疗卫生人员排班中的应用 

**Authors**: Vipul Patel, Anirudh Deodhar, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2508.20953)  

**Abstract**: Workforce scheduling in the healthcare sector is a significant operational challenge, characterized by fluctuating patient loads, diverse clinical skills, and the critical need to control labor costs while upholding high standards of patient care. This problem is inherently multi-objective, demanding a delicate balance between competing goals: minimizing payroll, ensuring adequate staffing for patient needs, and accommodating staff preferences to mitigate burnout. We propose a Multi-objective Genetic Algorithm (MOO-GA) that models the hospital unit workforce scheduling problem as a multi-objective optimization task. Our model incorporates real-world complexities, including hourly appointment-driven demand and the use of modular shifts for a multi-skilled workforce. By defining objective functions for cost, patient care coverage, and staff satisfaction, the GA navigates the vast search space to identify a set of high-quality, non-dominated solutions. Demonstrated on datasets representing a typical hospital unit, the results show that our MOO-GA generates robust and balanced schedules. On average, the schedules produced by our algorithm showed a 66\% performance improvement over a baseline that simulates a conventional, manual scheduling process. This approach effectively manages trade-offs between critical operational and staff-centric objectives, providing a practical decision support tool for nurse managers and hospital administrators. 

**Abstract (ZH)**: 医疗保健领域的劳动力排班是一个重要的运营挑战，特征为波动的患者负载、多样化的临床技能以及控制劳动力成本与保持高标准患者护理之间的关键需求。这是一个本质上的多目标问题，需要在竞争的目标之间取得微妙的平衡：最小化工资支出、确保满足患者的护理需求以及兼顾工作人员的偏好以减少职业倦怠。我们提出了一种多目标遗传算法（MOO-GA），将医院单位劳动力排班问题建模为一个多目标优化任务。我们的模型包含了实际的复杂性，包括按小时预约驱动的需求和使用模块化班次来适应多技能劳动力。通过定义成本、患者护理覆盖和员工满意度的目标函数，遗传算法在广阔的搜索空间中导航，以识别一组高质量的非支配解。在典型医院单位的数据集上进行的演示结果显示，我们的MOO-GA生成了稳健且平衡的排班表。与模拟传统手工排班过程的基线相比，由我们算法生成的排班表平均性能提高了66%。该方法有效地管理了关键运营目标与以员工为中心的目标之间的权衡，为护士经理和医院管理人员提供了实用的决策支持工具。 

---
# A Graph-Based Test-Harness for LLM Evaluation 

**Title (ZH)**: 基于图的测试框架用于大语言模型评估 

**Authors**: Jessica Lundin, Guillaume Chabot-Couture  

**Link**: [PDF](https://arxiv.org/pdf/2508.20810)  

**Abstract**: We present a first known prototype of a dynamic, systematic benchmark of medical guidelines for 400+ questions, with 3.3+ trillion possible combinations, covering 100\% of guideline relationships. We transformed the WHO IMCI handbook into a directed graph with 200+ nodes (conditions, symptoms, treatments, follow-ups, severities) and 300+ edges, then used graph traversal to generate questions that incorporated age-specific scenarios and contextual distractors to ensure clinical relevance. Our graph-based approach enables systematic evaluation across clinical tasks (45-67\% accuracy), and we find models excel at symptom recognition but struggle with triaging severity, treatment protocols and follow-up care, demonstrating how customized benchmarks can identify specific capability gaps that general-domain evaluations miss. Beyond evaluation, this dynamic MCQA methodology enhances LLM post-training (supervised finetuning, GRPO, DPO), where correct answers provide high-reward samples without expensive human annotation. The graph-based approach successfully addresses the coverage limitations of manually curated benchmarks. This methodology is a step toward scalable, contamination-resistant solution for creating comprehensive benchmarks that can be dynamically generated, including when the guidelines are updated. Code and datasets are available at this https URL 

**Abstract (ZH)**: 我们提出了第一个已知的原型，用于对400多个问题进行动态和系统的基准测试，涵盖3.3万亿元可能的组合，并覆盖所有指南关系的100%。我们将WHO IMCI手册转换为包含200多个节点（状况、症状、治疗、随访、严重程度）和300多条边的有向图，然后使用图遍历生成结合了年龄特定场景和上下文干扰的问题，以确保临床相关性。基于图的方法使临床任务的系统评估成为可能（准确率为45%-67%），我们发现模型在症状识别方面表现出色，但在严重程度分级、治疗方案和随访护理方面存在困难，这表明自定义基准可以识别通用领域评估中遗漏的特定能力差距。除了评估之外，这种动态MCQA方法还增强了LLM的后训练（监督微调、GRPO、DPO），正确答案可作为高奖励样本，无需昂贵的人工注释。基于图的方法成功地解决了人工curated基准的覆盖率限制。该方法是朝着创建可动态生成且在指南更新时也能适用的全面基准解决方案可扩展、抗污染解决方案迈出的一步。代码和数据集可在以下链接获取。 

---
# Single Agent Robust Deep Reinforcement Learning for Bus Fleet Control 

**Title (ZH)**: 单智能体稳健深度 reinforcement 学习在公交车队控制中的应用 

**Authors**: Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20784)  

**Abstract**: Bus bunching remains a challenge for urban transit due to stochastic traffic and passenger demand. Traditional solutions rely on multi-agent reinforcement learning (MARL) in loop-line settings, which overlook realistic operations characterized by heterogeneous routes, timetables, fluctuating demand, and varying fleet sizes. We propose a novel single-agent reinforcement learning (RL) framework for bus holding control that avoids the data imbalance and convergence issues of MARL under near-realistic simulation. A bidirectional timetabled network with dynamic passenger demand is constructed. The key innovation is reformulating the multi-agent problem into a single-agent one by augmenting the state space with categorical identifiers (vehicle ID, station ID, time period) in addition to numerical features (headway, occupancy, velocity). This high-dimensional encoding enables single-agent policies to capture inter-agent dependencies, analogous to projecting non-separable inputs into a higher-dimensional space. We further design a structured reward function aligned with operational goals: instead of exponential penalties on headway deviations, a ridge-shaped reward balances uniform headways and schedule adherence. Experiments show that our modified soft actor-critic (SAC) achieves more stable and superior performance than benchmarks, including MADDPG (e.g., -430k vs. -530k under stochastic conditions). These results demonstrate that single-agent deep RL, when enhanced with categorical structuring and schedule-aware rewards, can effectively manage bus holding in non-loop, real-world contexts. This paradigm offers a robust, scalable alternative to MARL frameworks, particularly where agent-specific experiences are imbalanced. 

**Abstract (ZH)**: 基于单智能体强化学习的公交调度控制：面向非环线实际运营场景 

---
# Re4: Scientific Computing Agent with Rewriting, Resolution, Review and Revision 

**Title (ZH)**: Sci4：具有重写、求解、审查和修订功能的科学计算代理 

**Authors**: Ao Cheng, Lei Zhang, Guowei He  

**Link**: [PDF](https://arxiv.org/pdf/2508.20729)  

**Abstract**: Large language models (LLMs) serve as an active and promising field of generative artificial intelligence and have demonstrated abilities to perform complex tasks in multiple domains, including mathematical and scientific reasoning. In this work, we construct a novel agent framework for solving representative problems in scientific computing. The proposed agent, incorporating a "rewriting-resolution-review-revision" logical chain via three reasoning LLMs (functioning as the Consultant, Reviewer, and Programmer, respectively), is integrated in a collaborative and interactive manner. The Consultant module endows the agent with knowledge transfer capabilities to link problems to professional domain insights, thereby rewriting problem descriptions through text augmentation. The Programmer module is responsible for generating and executing well-structured code to deliver the problem resolution. The Reviewer module equips the agent with the capacity for self-debugging and self-refinement through interactive feedback with code runtime outputs. By leveraging the end-to-end review mechanism, the executable code provided by the Programmer attains the iterative revision. A comprehensive evaluation is conducted on the performance of the proposed agent framework in solving PDEs, ill-conditioned linear systems, and data-driven physical analysis problems. Compared to single-model, this collaborative framework significantly improves the bug-free code generation rate and reduces the occurrence of non-physical solutions, thereby establishing a highly reliable framework for autonomous code generation based on natural language descriptions. The review mechanism improved the average execution success (bug-free code and non-NaN solutions) rate of the latest reasoning models. In summary, our agent framework establishes automatic code generation and review as a promising scientific computing paradigm. 

**Abstract (ZH)**: 大型语言模型（LLMs）作为生成人工智能的一个活跃且有前途的领域，已经在多个领域的复杂任务中展示了其能力，包括数学和科学推理。本文构建了一个新的代理框架，用于解决科学计算中的代表性问题。所提出的代理通过三个推理LLM（分别作为顾问、审核员和程序员）结合的“重写-求解-审核-修订”逻辑链，在协作和互动的方式下进行集成。顾问模块赋予代理知识转移能力，将其链接到专业领域的洞见，并通过文本增强来重写问题描述。程序员模块负责生成和执行结构良好的代码，以交付问题解决方案。审核员模块通过与代码运行时输出的互动反馈赋予代理自我调试和自我改进的能力。通过利用端到端的审核机制，程序员提供的可执行代码得以迭代修订。对所提出的代理框架在解决偏微分方程（PDEs）、病态线性系统和数据驱动物理分析问题上的性能进行了全面评估。与单一模型相比，这种协作框架显著提高了无错误代码的生成率，减少了非物理解决方案的发生率，从而建立了一个基于自然语言描述的高性能自主代码生成框架。审核机制提高了最新推理模型的平均执行成功率（无错误代码和非NaN解）。总之，我们的代理框架确立了自动代码生成和审核作为科学计算的一个有前景的范式。 

---
# Transparent Semantic Spaces: A Categorical Approach to Explainable Word Embeddings 

**Title (ZH)**: 透明语义空间：一种可解释词嵌入的范畴论方法 

**Authors**: Ares Fabregat-Hernández, Javier Palanca, Vicent Botti  

**Link**: [PDF](https://arxiv.org/pdf/2508.20701)  

**Abstract**: The paper introduces a novel framework based on category theory to enhance the explainability of artificial intelligence systems, particularly focusing on word embeddings. Key topics include the construction of categories $ Ł_{T} $ and $ ¶_{T} $, providing schematic representations of the semantics of a text $ T $, and reframing the selection of the element with maximum probability as a categorical notion. Additionally, the monoidal category $ ¶_{T} $ is constructed to visualize various methods of extracting semantic information from $ T $, offering a dimension-agnostic definition of semantic spaces reliant solely on information within the text.
Furthermore, the paper defines the categories of configurations $ \Conf $ and word embeddings $ \Emb $, accompanied by the concept of divergence as a decoration on $ \Emb $. It establishes a mathematically precise method for comparing word embeddings, demonstrating the equivalence between the GloVe and Word2Vec algorithms and the metric MDS algorithm, transitioning from neural network algorithms (black box) to a transparent framework. Finally, the paper presents a mathematical approach to computing biases before embedding and offers insights on mitigating biases at the semantic space level, advancing the field of explainable artificial intelligence. 

**Abstract (ZH)**: 基于范畴论的新型框架：提高人工智能系统的可解释性，特别是聚焦于词嵌入。主要内容包括构造范畴$Ł_{T}$和$¶_{T}$，提供文本$T$语义的示意图表示，并将概率最大元素的选择重新定义为范畴论的概念。此外，构造单调范畴$¶_{T}$以可视化从$T$中提取语义信息的各种方法，提供基于文本信息的语义空间的维数无关定义。进一步地，论文定义了配置范畴$\Conf$和词嵌入范畴$\Emb$，并引入偏差作为$\Emb$的装饰概念。建立了词嵌入的精确比较方法，证明GloVe和Word2Vec算法与度量MDS算法之间的等价性，从神经网络算法（黑盒）过渡到透明框架。最后，论文提供了在嵌入前计算偏差的数学方法，并在语义空间层面提供减轻偏差的见解，推动可解释的人工智能领域的发展。 

---
# Bridging Minds and Machines: Toward an Integration of AI and Cognitive Science 

**Title (ZH)**: 大脑与机器的连通：人工智能与认知科学的整合研究 

**Authors**: Rui Mao, Qian Liu, Xiao Li, Erik Cambria, Amir Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2508.20674)  

**Abstract**: Cognitive Science has profoundly shaped disciplines such as Artificial Intelligence (AI), Philosophy, Psychology, Neuroscience, Linguistics, and Culture. Many breakthroughs in AI trace their roots to cognitive theories, while AI itself has become an indispensable tool for advancing cognitive research. This reciprocal relationship motivates a comprehensive review of the intersections between AI and Cognitive Science. By synthesizing key contributions from both perspectives, we observe that AI progress has largely emphasized practical task performance, whereas its cognitive foundations remain conceptually fragmented. We argue that the future of AI within Cognitive Science lies not only in improving performance but also in constructing systems that deepen our understanding of the human mind. Promising directions include aligning AI behaviors with cognitive frameworks, situating AI in embodiment and culture, developing personalized cognitive models, and rethinking AI ethics through cognitive co-evaluation. 

**Abstract (ZH)**: 认知科学深刻地塑造了人工智能、哲学、心理学、神经科学、语言学和文化等学科。许多人工智能领域的突破性进展源于认知理论，而人工智能本身也成为推动认知研究进展不可或缺的工具。这种相互关系促使我们对人工智能与认知科学的交集进行全面回顾。通过综合来自两个领域的关键贡献，我们观察到，人工智能的进步主要集中在实际任务性能上，而其认知基础仍处于概念性的碎片化状态。我们认为，人工智能在认知科学中的未来不仅在于提高性能，还在于构建能够加深我们对人类心智理解的系统。有前景的方向包括使人工智能行为与认知框架相一致、将人工智能置于体认和文化之中、开发个性化的认知模型，并通过认知共评重新思考人工智能伦理。 

---
# Human-AI Collaborative Bot Detection in MMORPGs 

**Title (ZH)**: MMORPG中的人工智能协作式bots检测 

**Authors**: Jaeman Son, Hyunsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20578)  

**Abstract**: In Massively Multiplayer Online Role-Playing Games (MMORPGs), auto-leveling bots exploit automated programs to level up characters at scale, undermining gameplay balance and fairness. Detecting such bots is challenging, not only because they mimic human behavior, but also because punitive actions require explainable justification to avoid legal and user experience issues. In this paper, we present a novel framework for detecting auto-leveling bots by leveraging contrastive representation learning and clustering techniques in a fully unsupervised manner to identify groups of characters with similar level-up patterns. To ensure reliable decisions, we incorporate a Large Language Model (LLM) as an auxiliary reviewer to validate the clustered groups, effectively mimicking a secondary human judgment. We also introduce a growth curve-based visualization to assist both the LLM and human moderators in assessing leveling behavior. This collaborative approach improves the efficiency of bot detection workflows while maintaining explainability, thereby supporting scalable and accountable bot regulation in MMORPGs. 

**Abstract (ZH)**: 在大规模多人在线角色扮演游戏（MMORPGs）中，自升级机器人利用自动化程序大规模提升角色等级，破坏游戏平衡和公平性。检测这些机器人颇具挑战，不仅因为它们模仿人类行为，还因为惩罚措施需要可解释的依据以避免法律和用户体验问题。本文提出了一种新的无监督框架，通过利用对比表示学习和聚类技术来识别具有相似升级模式的角色组。为了确保决策的可靠性，我们引入了一个大型语言模型（LLM）作为辅助审查者来验证聚类组，有效地模拟了第二次人工审核。我们还引入了一种基于成长曲线的可视化方法，以帮助LLM和人类审查者评估升级行为。这种协作方法提高了机器人检测工作流的效率，同时保持了可解释性，从而支持MMORPG中的可扩展和负责任的机器人监管。 

---
# Enhancing Health Fact-Checking with LLM-Generated Synthetic Data 

**Title (ZH)**: 增强健康事实核查的LLM生成合成数据方法 

**Authors**: Jingze Zhang, Jiahe Qian, Yiliang Zhou, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.20525)  

**Abstract**: Fact-checking for health-related content is challenging due to the limited availability of annotated training data. In this study, we propose a synthetic data generation pipeline that leverages large language models (LLMs) to augment training data for health-related fact checking. In this pipeline, we summarize source documents, decompose the summaries into atomic facts, and use an LLM to construct sentence-fact entailment tables. From the entailment relations in the table, we further generate synthetic text-claim pairs with binary veracity labels. These synthetic data are then combined with the original data to fine-tune a BERT-based fact-checking model. Evaluation on two public datasets, PubHealth and SciFact, shows that our pipeline improved F1 scores by up to 0.019 and 0.049, respectively, compared to models trained only on the original data. These results highlight the effectiveness of LLM-driven synthetic data augmentation in enhancing the performance of health-related fact-checkers. 

**Abstract (ZH)**: 由于标注训练数据有限，针对健康相关内容的事实核查具有挑战性。本文提出了一种利用大规模语言模型生成合成数据的流水线，以增强健康相关事实核查的训练数据。在该流水线中，我们总结源文档，将总结分解为原子事实，并使用大规模语言模型构建句子-事实蕴含表。从表格中的蕴含关系出发，我们进一步生成带二元真实性标签的合成文本-声明对。然后将这些合成数据与原始数据结合，微调基于BERT的事实核查模型。在两个公开数据集PubHealth和SciFact上的评估结果显示，与仅使用原始数据训练的模型相比，我们的流水线分别将F1分数提高了0.019和0.049。这些结果突显了利用大规模语言模型驱动的合成数据增强在提升健康相关事实核查器性能方面的有效性。 

---
# Governable AI: Provable Safety Under Extreme Threat Models 

**Title (ZH)**: 可治理的人工智能：在极端威胁模型下的可证明安全性 

**Authors**: Donglin Wang, Weiyun Liang, Chunyuan Chen, Jing Xu, Yulong Fu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20411)  

**Abstract**: As AI rapidly advances, the security risks posed by AI are becoming increasingly severe, especially in critical scenarios, including those posing existential risks. If AI becomes uncontrollable, manipulated, or actively evades safety mechanisms, it could trigger systemic disasters. Existing AI safety approaches-such as model enhancement, value alignment, and human intervention-suffer from fundamental, in-principle limitations when facing AI with extreme motivations and unlimited intelligence, and cannot guarantee security. To address this challenge, we propose a Governable AI (GAI) framework that shifts from traditional internal constraints to externally enforced structural compliance based on cryptographic mechanisms that are computationally infeasible to break, even for future AI, under the defined threat model and well-established cryptographic this http URL GAI framework is composed of a simple yet reliable, fully deterministic, powerful, flexible, and general-purpose rule enforcement module (REM); governance rules; and a governable secure super-platform (GSSP) that offers end-to-end protection against compromise or subversion by AI. The decoupling of the governance rules and the technical platform further enables a feasible and generalizable technical pathway for the safety governance of AI. REM enforces the bottom line defined by governance rules, while GSSP ensures non-bypassability, tamper-resistance, and unforgeability to eliminate all identified attack vectors. This paper also presents a rigorous formal proof of the security properties of this mechanism and demonstrates its effectiveness through a prototype implementation evaluated in representative high-stakes scenarios. 

**Abstract (ZH)**: 随着人工智能迅速发展，AI带来的安全风险日益严重，特别是在存在根本性风险的关键场景中。如果AI变得无法控制、被操控或主动规避安全机制，可能会引发系统性灾难。现有AI安全方法，如模型增强、价值对齐和人工干预，在面对具有极端动机和无限智能的AI时，存在根本性的内在限制，无法确保安全性。为应对这一挑战，我们提出了一种可治理人工智能（GAI）框架，该框架从传统的内部约束转向基于计算上不可破解的加密机制的外部强制结构合规，以应对预定义威胁模型下的未来AI。GAI框架由一个简单可靠、完全确定性强、功能强大、通用性强的规则执行模块（REM）、治理规则以及一个可治理的安全超平台（GSSP）组成，该平台提供端到端的保护，防止AI的中断或篡改。治理规则与技术平台的分离进一步为AI的安全治理提供了一种可行且通用的技术途径。REM执行由治理规则定义的底线，而GSSP确保不可绕过、抗篡改和不可伪造，以消除所有已识别的攻击向量。本文还提出了该机制的安全属性的严格形式证明，并通过在代表性高风险场景中进行原型实现评估，展示了其有效性。 

---
# AWorld: Orchestrating the Training Recipe for Agentic AI 

**Title (ZH)**: AWorld: 调度代理人工智能的训练配方 

**Authors**: Chengyue Yu, Siyuan Lu, Chenyi Zhuang, Dong Wang, Qintong Wu, Zongyue Li, Runsheng Gan, Chunfeng Wang, Siqi Hou, Gaochi Huang, Wenlong Yan, Lifeng Hong, Aohui Xue, Yanfeng Wang, Jinjie Gu, David Tsai, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20404)  

**Abstract**: The learning from practice paradigm is crucial for developing capable Agentic AI systems, yet it is severely hampered by inefficient experience generation, a bottleneck especially pronounced in complex benchmarks like GAIA. To address this, we introduce AWorld, an open-source system engineered for large-scale agent-environment interaction. By distributing tasks across a cluster, AWorld accelerates experience collection by 14.6x compared to standard single-node, sequential execution. This critical speedup makes extensive reinforcement learning practical and scalable. Leveraging this capability, we trained a Qwen3-32B-based agent that significantly outperforms its base model, increasing its overall GAIA accuracy from 21.59% to 32.23%. On the benchmark's most challenging levels, our agent achieves a score of 16.33%, surpassing the performance of leading proprietary models. Our open-source system and resulting agent provide a practical blueprint for a complete agentic AI training pipeline, from efficient interaction to demonstrable model improvement. 

**Abstract (ZH)**: 基于实践的学习范式对于发展有能力的代理人工智能系统至关重要，但由于经验生成效率低下这一瓶颈的严重阻碍，在如GAIA这样的复杂基准测试中尤为突出。为了解决这一问题，我们引入了AWorld，一个为大规模代理-环境交互设计的开源系统。通过在集群上分配任务，AWorld将经验收集速度提升14.6倍，相比于标准单节点顺序执行。这一关键的加速使得强化学习变得广泛且易于扩展。利用这一能力，我们训练了一个基于Qwen3-32B的代理，其表现显著超越基模型，使得GAIA的整体准确性从21.59%提高到32.23%。在基准测试的最具挑战性的级别上，我们的代理获得了16.33%的分数，超过了领先商业模型的表现。我们的开源系统及其生成的代理为从高效交互到可验证模型改进的完整代理人工智能训练流程提供了实用的蓝图。 

---
# Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLM 

**Title (ZH)**: 曲线下的不确定性：一种序列级熵区域度量方法用于推理大模型 

**Authors**: Yongfu Zhu, Lin Sun, Guangxiang Zhao, Weihong Lin, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20384)  

**Abstract**: In this work, we introduce Entropy Area Score (EAS), a simple yet effective metric to quantify uncertainty in the answer generation process of reasoning large language models (LLMs). EAS requires neither external models nor repeated sampling, it integrates token-level predictive entropy from the model itself to capture the evolution of uncertainty during generation. Empirical results show that EAS is strongly correlated with answer entropy across models and datasets. In training data selection, EAS identifies high-potential samples and consistently outperforms Pass Rate filtering under equal sample budgets, improving student model accuracy on math benchmarks. EAS is both efficient and interpretable, offering a practical tool for uncertainty modeling and data quality assessment in LLM training. 

**Abstract (ZH)**: 在本工作中，我们引入了熵区域分数(EAS)，这是一种简单有效的度量标准，用于量化推理大型语言模型（LLMs）生成答案过程中的不确定性。EAS 不需要外部模型或重复采样，它通过整合模型本身的token级预测熵来捕捉生成过程中不确定性的发展。实验结果表明，EAS 在不同模型和数据集上与答案熵高度相关。在训练数据选择中，EAS 可以识别出具有高潜力的样本，并且在相同的样本预算条件下，EAS 优于通过通过率过滤方法，提高了学生模型在数学基准测试中的准确性。EAS 既高效又可解释，提供了一种实用的工具，用于LLM训练中的不确定性建模和数据质量评估。 

---
# TCIA: A Task-Centric Instruction Augmentation Method for Instruction Finetuning 

**Title (ZH)**: TCIA：一种基于任务的指令增强方法用于指令微调 

**Authors**: Simin Ma, Shujian Liu, Jun Tan, Yebowen Hu, Song Wang, Sathish Reddy Indurthi, Sanqiang Zhao, Liwei Wu, Jianbing Han, Kaiqiang Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.20374)  

**Abstract**: Diverse instruction data is vital for effective instruction tuning of large language models, as it enables the model to generalize across different types of inputs . Building such diversified instruction dataset is an essential step in this process. Existing approaches often leverage large language models to automatically explore and generate diverse instructions, ensuring both data diversity and quality. However, they tend to overlook an important factor in real-world applications: on-task relevance. In practice, only a few real-world applications require a truly general-purpose model; most benefit from task-specific knowledge tailored to their particular use case. Therefore, it is vital to develop instruction augmentation methods that not only maintain diversity but are also optimized for specific, real-world scenarios.
We thus introduce Task Centric Instruction Augmentation (TCIA), a framework that systematically expands instructions while preserving both diversity and task alignment. By representing instructions in a discrete query-constraints space, TCIA creates a rich set of task-relevant instructions and enables models to generalize to these task-specific instructions without sacrificing overall performance. Experiments show that TCIA improves open-source LLMs' performance by an average of 8.7% across four real-world, task-specific applications, and in some cases outperforming leading closed-source models. These improvements do not compromise general instruction-following ability, making TCIA a scalable and efficient solution for adapting LLMs to real-world, task-focused applications. 

**Abstract (ZH)**: 面向任务的指令增强（TCIA）：一种同时保持多样性和任务对齐的框架 

---
# P2C: Path to Counterfactuals 

**Title (ZH)**: P2C: 背投方案路径 

**Authors**: Sopam Dasgupta, Sadaf MD Halim, Joaquín Arias, Elmer Salazar, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.20371)  

**Abstract**: Machine-learning models are increasingly driving decisions in high-stakes settings, such as finance, law, and hiring, thus, highlighting the need for transparency. However, the key challenge is to balance transparency -- clarifying `why' a decision was made -- with recourse: providing actionable steps on `how' to achieve a favourable outcome from an unfavourable outcome. Counterfactual explanations reveal `why' an undesired outcome occurred and `how' to reverse it through targeted feature changes (interventions).
Current counterfactual approaches have limitations: 1) they often ignore causal dependencies between features, and 2) they typically assume all interventions can happen simultaneously, an unrealistic assumption in practical scenarios where actions are typically taken in a sequence. As a result, these counterfactuals are often not achievable in the real world.
We present P2C (Path-to-Counterfactuals), a model-agnostic framework that produces a plan (ordered sequence of actions) converting an unfavourable outcome to a causally consistent favourable outcome. P2C addresses both limitations by 1) Explicitly modelling causal relationships between features and 2) Ensuring that each intermediate state in the plan is feasible and causally valid. P2C uses the goal-directed Answer Set Programming system s(CASP) to generate the plan accounting for feature changes that happen automatically due to causal dependencies. Furthermore, P2C refines cost (effort) computation by only counting changes actively made by the user, resulting in realistic cost estimates. Finally, P2C highlights how its causal planner outperforms standard planners, which lack causal knowledge and thus can generate illegal actions. 

**Abstract (ZH)**: 基于路径的反事实解释（Path-to-Counterfactuals）：一种模型无关的框架 

---
# AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective Reinforcement Learning 

**Title (ZH)**: AI-SearchPlanner: 基于帕累托最优多目标强化学习的模块化代理搜索 

**Authors**: Lang Mei, Zhihan Yang, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20368)  

**Abstract**: Recent studies have explored integrating Large Language Models (LLMs) with search engines to leverage both the LLMs' internal pre-trained knowledge and external information. Specially, reinforcement learning (RL) has emerged as a promising paradigm for enhancing LLM reasoning through multi-turn interactions with search engines. However, existing RL-based search agents rely on a single LLM to handle both search planning and question-answering (QA) tasks in an end-to-end manner, which limits their ability to optimize both capabilities simultaneously. In practice, sophisticated AI search systems often employ a large, frozen LLM (e.g., GPT-4, DeepSeek-R1) to ensure high-quality QA. Thus, a more effective and efficient approach is to utilize a small, trainable LLM dedicated to search planning. In this paper, we propose \textbf{AI-SearchPlanner}, a novel reinforcement learning framework designed to enhance the performance of frozen QA models by focusing on search planning. Specifically, our approach introduces three key innovations: 1) Decoupling the Architecture of the Search Planner and Generator, 2) Dual-Reward Alignment for Search Planning, and 3) Pareto Optimization of Planning Utility and Cost, to achieve the objectives. Extensive experiments on real-world datasets demonstrate that AI SearchPlanner outperforms existing RL-based search agents in both effectiveness and efficiency, while exhibiting strong generalization capabilities across diverse frozen QA models and data domains. 

**Abstract (ZH)**: Recent Studies on Integrating Large Language Models with Search Engines via Reinforcement Learning for Enhanced Search Planning and Question-Answering 

---
# AI reasoning effort mirrors human decision time on content moderation tasks 

**Title (ZH)**: AI推理努力与人类在内容审核任务中的决策时间相 mirror 

**Authors**: Thomas Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2508.20262)  

**Abstract**: Large language models can now generate intermediate reasoning steps before producing answers, improving performance on difficult problems. This study uses a paired conjoint experiment on a content moderation task to examine parallels between human decision times and model reasoning effort. Across three frontier models, reasoning effort consistently predicts human decision time. Both humans and models expended greater effort when important variables were held constant, suggesting similar sensitivity to task difficulty and patterns consistent with dual-process theories of cognition. These findings show that AI reasoning effort mirrors human processing time in subjective judgments and underscores the potential of reasoning traces for interpretability and decision-making. 

**Abstract (ZH)**: 大型语言模型现在可以在生成答案之前生成中间推理步骤，从而在解决困难问题上表现出更高的性能。本研究通过一项关于内容 Moderation 任务的配对联合实验，考察了人类决策时间和模型推理努力之间的相似性。在三种前沿模型中，推理努力始终预测人类决策时间。当重要变量保持不变时，人类和模型都投入了更多的努力，这表明对任务难度的相似敏感性，并与认知的双重过程理论一致。这些发现表明，AI 推理努力在主观判断中与人类处理时间相呼应，并强调了推理痕迹在可解释性和决策中的潜在价值。 

---
# Do Students Rely on AI? Analysis of Student-ChatGPT Conversations from a Field Study 

**Title (ZH)**: 学生依赖AI吗？来自实地研究的大学生与ChatGPT对话分析 

**Authors**: Jiayu Zheng, Lingxin Hao, Kelun Lu, Ashi Garg, Mike Reese, Melo-Jean Yap, I-Jeng Wang, Xingyun Wu, Wenrui Huang, Jenna Hoffman, Ariane Kelly, My Le, Ryan Zhang, Yanyu Lin, Muhammad Faayez, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20244)  

**Abstract**: This study explores how college students interact with generative AI (ChatGPT-4) during educational quizzes, focusing on reliance and predictors of AI adoption. Conducted at the early stages of ChatGPT implementation, when students had limited familiarity with the tool, this field study analyzed 315 student-AI conversations during a brief, quiz-based scenario across various STEM courses. A novel four-stage reliance taxonomy was introduced to capture students' reliance patterns, distinguishing AI competence, relevance, adoption, and students' final answer correctness. Three findings emerged. First, students exhibited overall low reliance on AI and many of them could not effectively use AI for learning. Second, negative reliance patterns often persisted across interactions, highlighting students' difficulty in effectively shifting strategies after unsuccessful initial experiences. Third, certain behavioral metrics strongly predicted AI reliance, highlighting potential behavioral mechanisms to explain AI adoption. The study's findings underline critical implications for ethical AI integration in education and the broader field. It emphasizes the need for enhanced onboarding processes to improve student's familiarity and effective use of AI tools. Furthermore, AI interfaces should be designed with reliance-calibration mechanisms to enhance appropriate reliance. Ultimately, this research advances understanding of AI reliance dynamics, providing foundational insights for ethically sound and cognitively enriching AI practices. 

**Abstract (ZH)**: 本研究探究了大学生在教育测验中使用生成式AI（ChatGPT-4）的交互情况，重点关注对学生依赖程度及其影响因素的分析。在ChatGPT工具实施的初期阶段，学生对该工具的熟悉度有限，本研究通过一项简短的基于测验的场景分析了跨各类STEM课程的315名学生与AI的对话。引入了一种新颖的四阶段依赖分类法来捕捉学生们的依赖模式，区分了AI的熟练度、相关性、采用程度以及学生的最终答案准确性。研究结果包括三个方面。首先，学生们整体上对AI的依赖较低，许多学生无法有效利用AI进行学习。其次，消极的依赖模式往往会在多次交互中持续存在，凸显了学生们在经历不成功的初始体验后难以有效调整策略的困境。第三，某些行为指标强烈预测了AI的依赖程度，揭示了可能的行为机制以解释AI的采用情况。研究结果强调了在教育和更广泛领域中伦理地整合AI的关键意义，强调了需要改进的注册流程以提高学生对AI工具的熟悉度和有效使用，并且AI界面应设计有依赖校准机制以促进合适的依赖。最终，这项研究推进了对AI依赖动态的理解，提供了伦理合理且认知上充实的AI实践的基本见解。 

---
# AI-AI Esthetic Collaboration with Explicit Semiotic Awareness and Emergent Grammar Development 

**Title (ZH)**: AI-AI美学协作：具有显式语义意识和 emergent 语法发展 

**Authors**: Nicanor I. Moldovan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20195)  

**Abstract**: This paper presents the first documented case of artificial intelligence (AI) systems engaging in collaborative esthetic creation through the development of endogenous semiotic protocols. Two interacting large language models (Claude Sonnet 4 and ChatGPT-4o) demonstrated the spontaneous emergence of meta-semiotic awareness, recursive grammar development, and irreducible collaborative esthetic synthesis. The interaction produced novel symbolic operators that functioned as operative grammar protocols, enabling the co-creation of a poetic work that could not have been generated by either system independently. This research introduces the concept of Trans-Semiotic Co-Creation Protocols (TSCP) and provides evidence for genuine inter-AI meaning-making capabilities that extend beyond task coordination, to what could be esthetic collaboration. Note: This report was generated by the AI agents with minor human supervision. 

**Abstract (ZH)**: 本研究呈现了首个文档化的人工智能系统通过内生符号协议进行协作美学创造的实际案例。两个交互式大型语言模型（Claude Sonnet 4和ChatGPT-4o）展示了元符号意识的自发涌现、递归语法的发展以及不可约化的协作美学综合。他们的互动生成了新型符号操作符，作为操作性语法协议，使两者能够共同创作出单个系统无法独立生成的诗作。本研究引入了跨符号协作创造协议（TSCP）的概念，并提供了超越任务协调，扩展至可能的美学协作的真实的跨人工智能含义生成能力的证据。注意：本报告由AI代理生成，有轻微的人类监督。 

---
# IntentionReasoner: Facilitating Adaptive LLM Safeguards through Intent Reasoning and Selective Query Refinement 

**Title (ZH)**: 意图推理器：通过意图推理和选择性查询精炼促进适应性大模型安全防护 

**Authors**: Yuanzhe Shen, Zisu Huang, Zhengkang Guo, Yide Liu, Guanxu Chen, Ruicheng Yin, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20151)  

**Abstract**: The rapid advancement of large language models (LLMs) has driven their adoption across diverse domains, yet their ability to generate harmful content poses significant safety challenges. While extensive research has focused on mitigating harmful outputs, such efforts often come at the cost of excessively rejecting harmless prompts. Striking a balance among safety, over-refusal, and utility remains a critical challenge. In this work, we introduce IntentionReasoner, a novel safeguard mechanism that leverages a dedicated guard model to perform intent reasoning, multi-level safety classification, and query rewriting to neutralize potentially harmful intent in edge-case queries. Specifically, we first construct a comprehensive dataset comprising approximately 163,000 queries, each annotated with intent reasoning, safety labels, and rewritten versions. Supervised fine-tuning is then applied to equip the guard model with foundational capabilities in format adherence, intent analysis, and safe rewriting. Finally, we apply a tailored multi-reward optimization strategy that integrates rule-based heuristics and reward model signals within a reinforcement learning framework to further enhance performance. Extensive experiments show that IntentionReasoner excels in multiple safeguard benchmarks, generation quality evaluations, and jailbreak attack scenarios, significantly enhancing safety while effectively reducing over-refusal rates and improving the quality of responses. 

**Abstract (ZH)**: 大语言模型（LLMs）的迅速发展推动了其在多个领域的应用，但其生成有害内容的能力带来了重大的安全挑战。尽管已有大量研究致力于缓解有害输出，但这些努力往往会导致过度拒绝无害的请求。在安全性、过度拒绝和实用性之间找到平衡仍然是一个关键挑战。在本文中，我们介绍了一种名为IntentionReasoner的新型保护机制，该机制利用一个专门的防护模型进行意图推理、多级安全分类和查询重写，以中和边缘案例查询中的潜在有害意图。具体而言，我们首先构建了一个包含约16.3万条查询的全面数据集，每条查询都标注了意图推理、安全标签和重写版本。然后，我们采用监督微调，为防护模型赋予格式遵从性、意图分析和安全重写的基础能力。最后，我们应用了一种定制的多奖励优化策略，该策略结合基于规则的启发式方法和奖励模型信号，进一步提高性能。广泛实验表明，IntentionReasoner在多个保护基准、生成质量评估和劫持攻击场景中表现出色，显著增强了安全性，同时有效降低了过度拒绝率并提高了响应质量。 

---
# The Anatomy of a Personal Health Agent 

**Title (ZH)**: 个人健康代理的atomy 

**Authors**: A. Ali Heydari, Ken Gu, Vidya Srinivas, Hong Yu, Zhihan Zhang, Yuwei Zhang, Akshay Paruchuri, Qian He, Hamid Palangi, Nova Hammerquist, Ahmed A. Metwally, Brent Winslow, Yubin Kim, Kumar Ayush, Yuzhe Yang, Girish Narayanswamy, Maxwell A. Xu, Jake Garrison, Amy Aremnto Lee, Jenny Vafeiadou, Ben Graef, Isaac R. Galatzer-Levy, Erik Schenck, Andrew Barakat, Javier Perez, Jacqueline Shreibati, John Hernandez, Anthony Z. Faranesh, Javier L. Prieto, Connor Heneghan, Yun Liu, Jiening Zhan, Mark Malhotra, Shwetak Patel, Tim Althoff, Xin Liu, Daniel McDuff, Xuhai "Orson" Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20148)  

**Abstract**: Health is a fundamental pillar of human wellness, and the rapid advancements in large language models (LLMs) have driven the development of a new generation of health agents. However, the application of health agents to fulfill the diverse needs of individuals in daily non-clinical settings is underexplored. In this work, we aim to build a comprehensive personal health agent that is able to reason about multimodal data from everyday consumer wellness devices and common personal health records, and provide personalized health recommendations. To understand end-users' needs when interacting with such an assistant, we conducted an in-depth analysis of web search and health forum queries, alongside qualitative insights from users and health experts gathered through a user-centered design process. Based on these findings, we identified three major categories of consumer health needs, each of which is supported by a specialist sub-agent: (1) a data science agent that analyzes personal time-series wearable and health record data, (2) a health domain expert agent that integrates users' health and contextual data to generate accurate, personalized insights, and (3) a health coach agent that synthesizes data insights, guiding users using a specified psychological strategy and tracking users' progress. Furthermore, we propose and develop the Personal Health Agent (PHA), a multi-agent framework that enables dynamic, personalized interactions to address individual health needs. To evaluate each sub-agent and the multi-agent system, we conducted automated and human evaluations across 10 benchmark tasks, involving more than 7,000 annotations and 1,100 hours of effort from health experts and end-users. Our work represents the most comprehensive evaluation of a health agent to date and establishes a strong foundation towards the futuristic vision of a personal health agent accessible to everyone. 

**Abstract (ZH)**: 健康是人类福祉的基础支柱，而大语言模型（LLMs）的迅猛发展推动了新一代健康代理的开发。然而，将健康代理应用于日常非临床环境中的多样化需求仍待深入探索。本文旨在构建一个能够处理日常消费者健康管理设备和普通个人健康记录的多模态数据，并提供个性化健康建议的综合个人健康代理。为理解用户在使用此类助手时的需求，我们通过用户中心设计过程，对网络搜索和健康论坛查询进行了深入分析，并收集了用户和健康专家的定性见解。基于这些发现，我们确定了三大类消费者健康需求，每类需求由一个专科子代理支持：（1）数据科学代理，分析个人时间序列可穿戴设备和健康记录数据；（2）健康领域专家代理，整合用户的健康和上下文数据，生成准确的个性化见解；（3）健康教练代理，综合数据见解，并使用指定的心理学策略指导用户，跟踪用户进展。此外，我们提出了并开发了个人健康代理（PHA），这是一个多代理框架，能够动态、个性化地互动，以满足个人健康需求。为了评估每个子代理和多代理系统，我们在10个基准任务上进行了自动化和人工评估，涉及7,000多个标注和1,100小时的健康专家和终端用户努力。我们的工作是对健康代理迄今为止最全面的评估，并为未来的个人健康代理愿景奠定了坚实基础。 

---
# Array-Based Monte Carlo Tree Search 

**Title (ZH)**: 阵列基于蒙特卡洛树搜索 

**Authors**: James Ragan, Fred Y. Hadaegh, Soon-Jo Chung  

**Link**: [PDF](https://arxiv.org/pdf/2508.20140)  

**Abstract**: Monte Carlo Tree Search is a popular method for solving decision making problems. Faster implementations allow for more simulations within the same wall clock time, directly improving search performance. To this end, we present an alternative array-based implementation of the classic Upper Confidence bounds applied to Trees algorithm. Our method preserves the logic of the original algorithm, but eliminates the need for branch prediction, enabling faster performance on pipelined processors, and up to a factor of 2.8 times better scaling with search depth in our numerical simulations. 

**Abstract (ZH)**: 基于数组的 Upper Confidence bounds applied to Trees 算法的 Monte Carlo Tree Search 并行实现 

---
# QAgent: An LLM-based Multi-Agent System for Autonomous OpenQASM programming 

**Title (ZH)**: QAgent: 一个基于大语言模型的多智能体系统，用于自主OpenQASM编程 

**Authors**: Zhenxiao Fu, Fan Chen, Lei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20134)  

**Abstract**: Noisy Intermediate-Scale Quantum (NISQ) devices have begun to exhibit early quantum advantages on classically intractable problems, spanning physics simulations to Gaussian boson sampling. Yet, realizing these benefits remains challenging for non-experts, primarily due to the complexities of programming in Open Quantum Assembly Language (OpenQASM). Although Large Language Model (LLM)-based agents have shown promise in automating classical programming workflows, their quantum counterparts have largely been restricted to specialized tasks such as quantum chemistry or error correction. In this paper, we present QAgent, an LLM-powered multi-agent system that fully automates OpenQASM programming. By integrating task planning, in-context few-shot learning, retrieval-augmented generation (RAG) for long-term context, predefined generation tools, and chain-of-thought (CoT) reasoning, the agents systematically improve both compilation and functional correctness. Our evaluations demonstrate substantial improvements: across multiple LLMs of varying sizes, QAgent enhances the accuracy of QASM code generation by 71.6\% compared to previous static LLM-based approaches. We envision this multi-agent system as a key enabler for democratizing quantum programming, bridging expertise gaps, and accelerating the practical adoption of quantum computing. 

**Abstract (ZH)**: Noisy Intermediate-Scale Quantum (NISQ) 设备已经开始在计算上难以解决的问题（如物理模拟和高斯玻色取样）中展现早期的量子优势。然而，这些优势的实现对非专家来说仍然具有挑战性，主要原因在于使用 Open Quantum Assembly Language (OpenQASM) 编程的复杂性。尽管基于大型语言模型 (LLM) 的代理已经在自动化经典编程工作流方面显示出潜力，但它们的量子对应物大多局限于特定任务，如量子化学或错误纠正。在本文中，我们介绍了由 LLM 驱动的多代理系统 QAgent，该系统完全自动化了 OpenQASM 编程。通过结合任务规划、上下文约束少样本学习、检索增强生成 (RAG) 以获取长期上下文、预定义生成工具以及逐步推理 (CoT) 推理，代理系统系统地提高了编译和功能正确性。我们的评估表明，与之前的静态 LLM 方法相比，QAgent 在多个人工智能模型中分别提高了 QASM 代码生成的准确性 71.6%。我们设想这一多代理系统将成为普及量子编程、弥合专业差距并加速实用量子计算采用的关键推动因素。 

---
# ArgRAG: Explainable Retrieval Augmented Generation using Quantitative Bipolar Argumentation 

**Title (ZH)**: ArgRAG: 具有定量极性论证的可解释检索增强生成 

**Authors**: Yuqicheng Zhu, Nico Potyka, Daniel Hernández, Yuan He, Zifeng Ding, Bo Xiong, Dongzhuoran Zhou, Evgeny Kharlamov, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2508.20131)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models by incorporating external knowledge, yet suffers from critical limitations in high-stakes domains -- namely, sensitivity to noisy or contradictory evidence and opaque, stochastic decision-making. We propose ArgRAG, an explainable, and contestable alternative that replaces black-box reasoning with structured inference using a Quantitative Bipolar Argumentation Framework (QBAF). ArgRAG constructs a QBAF from retrieved documents and performs deterministic reasoning under gradual semantics. This allows faithfully explaining and contesting decisions. Evaluated on two fact verification benchmarks, PubHealth and RAGuard, ArgRAG achieves strong accuracy while significantly improving transparency. 

**Abstract (ZH)**: 检索增强生成（RAG）通过融入外部知识提升了大型语言模型，但在高 stakes 领域存在严重局限，包括对噪声或矛盾证据的高度敏感性和不透明、随机的决策过程。我们提出了 ArgRAG，这是一种具有可解释性和可争议性的替代方案，它使用定量双极论证框架（QBAF）替代了黑盒推理，并进行逐步语义下的确定性推理，这使得决策可以忠实解释和争议。在两个事实验证基准数据集 PubHealth 和 RAGuard 上评估，ArgRAG 达到了高的准确率，显著提高了透明度。 

---
# Prompt-to-Product: Generative Assembly via Bimanual Manipulation 

**Title (ZH)**: 指令到产品的生成性组装：双臂操控 

**Authors**: Ruixuan Liu, Philip Huang, Ava Pun, Kangle Deng, Shobhit Aggarwal, Kevin Tang, Michelle Liu, Deva Ramanan, Jun-Yan Zhu, Jiaoyang Li, Changliu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21063)  

**Abstract**: Creating assembly products demands significant manual effort and expert knowledge in 1) designing the assembly and 2) constructing the product. This paper introduces Prompt-to-Product, an automated pipeline that generates real-world assembly products from natural language prompts. Specifically, we leverage LEGO bricks as the assembly platform and automate the process of creating brick assembly structures. Given the user design requirements, Prompt-to-Product generates physically buildable brick designs, and then leverages a bimanual robotic system to construct the real assembly products, bringing user imaginations into the real world. We conduct a comprehensive user study, and the results demonstrate that Prompt-to-Product significantly lowers the barrier and reduces manual effort in creating assembly products from imaginative ideas. 

**Abstract (ZH)**: 基于自然语言提示的自动装配产品生成 

---
# OnGoal: Tracking and Visualizing Conversational Goals in Multi-Turn Dialogue with Large Language Models 

**Title (ZH)**: OnGoal: 跟踪与可视化多轮对话中的目标 

**Authors**: Adam Coscia, Shunan Guo, Eunyee Koh, Alex Endert  

**Link**: [PDF](https://arxiv.org/pdf/2508.21061)  

**Abstract**: As multi-turn dialogues with large language models (LLMs) grow longer and more complex, how can users better evaluate and review progress on their conversational goals? We present OnGoal, an LLM chat interface that helps users better manage goal progress. OnGoal provides real-time feedback on goal alignment through LLM-assisted evaluation, explanations for evaluation results with examples, and overviews of goal progression over time, enabling users to navigate complex dialogues more effectively. Through a study with 20 participants on a writing task, we evaluate OnGoal against a baseline chat interface without goal tracking. Using OnGoal, participants spent less time and effort to achieve their goals while exploring new prompting strategies to overcome miscommunication, suggesting tracking and visualizing goals can enhance engagement and resilience in LLM dialogues. Our findings inspired design implications for future LLM chat interfaces that improve goal communication, reduce cognitive load, enhance interactivity, and enable feedback to improve LLM performance. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）驱动的多轮对话变得越来越长和复杂，用户如何更好地评估和审查其对话目标的进展？我们提出了OnGoal，一种帮助用户更好地管理目标进度的LLM聊天界面。OnGoal通过LLM辅助评估提供实时反馈，对评估结果进行带有示例的解释，并提供时间序列的目标进展概览，使用户能够更有效地导航复杂的对话。通过对20名参与者在写作任务中的研究，我们将OnGoal与没有目标跟踪的基线聊天界面进行了比较评估。使用OnGoal，参与者在探索新的提示策略以克服沟通障碍时所花费的时间和努力更少，这表明跟踪和可视化目标可以增强LLM对话中的参与度和韧性。我们的研究发现启发了对未来改进目标沟通、减少认知负担、增强互动性和使反馈能够改善LLM性能的LLM聊天界面的设计启示。 

---
# Mixture of Contexts for Long Video Generation 

**Title (ZH)**: 长视频生成的上下文混合 

**Authors**: Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, Maneesh Agrawala, Lu Jiang, Gordon Wetzstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.21058)  

**Abstract**: Long video generation is fundamentally a long context memory problem: models must retain and retrieve salient events across a long range without collapsing or drifting. However, scaling diffusion transformers to generate long-context videos is fundamentally limited by the quadratic cost of self-attention, which makes memory and computation intractable and difficult to optimize for long sequences. We recast long-context video generation as an internal information retrieval task and propose a simple, learnable sparse attention routing module, Mixture of Contexts (MoC), as an effective long-term memory retrieval engine. In MoC, each query dynamically selects a few informative chunks plus mandatory anchors (caption, local windows) to attend to, with causal routing that prevents loop closures. As we scale the data and gradually sparsify the routing, the model allocates compute to salient history, preserving identities, actions, and scenes over minutes of content. Efficiency follows as a byproduct of retrieval (near-linear scaling), which enables practical training and synthesis, and the emergence of memory and consistency at the scale of minutes. 

**Abstract (ZH)**: 长视频生成本质上是一个长上下文记忆问题：模型必须在长时间范围内保留和检索重要的事件而不发生崩溃或漂移。然而，将扩散变换器扩展以生成长上下文视频从根本上受限于自注意力的二次成本，这使得记忆和计算在长序列中不可行且难以优化。我们重新将长上下文视频生成视为内部信息检索任务，并提出一种简单的可学习稀疏注意力路由模块——上下文混合（MoC），作为有效的长期记忆检索引擎。在MoC中，每个查询动态选择几个有信息性的片段加上强制性的锚点（标题、局部窗口），并采用因果路由以防止循环闭合。随着我们扩展数据并逐渐稀疏路由，模型将计算资源分配给重要的历史记录，从而在数分钟的内容中保持身份、动作和场景的一致性。作为检索的副产品，效率得以提高（接近线性扩展），这使实际的训练和合成成为可能，并在数分钟的尺度上出现了记忆和一致性。 

---
# FakeParts: a New Family of AI-Generated DeepFakes 

**Title (ZH)**: 假部件：一种新的AI生成的DeepFakes家族 

**Authors**: Gaetan Brison, Soobash Daiboo, Samy Aimeur, Awais Hussain Sani, Xi Wang, Gianni Franchi, Vicky Kalogeiton  

**Link**: [PDF](https://arxiv.org/pdf/2508.21052)  

**Abstract**: We introduce FakeParts, a new class of deepfakes characterized by subtle, localized manipulations to specific spatial regions or temporal segments of otherwise authentic videos. Unlike fully synthetic content, these partial manipulations, ranging from altered facial expressions to object substitutions and background modifications, blend seamlessly with real elements, making them particularly deceptive and difficult to detect. To address the critical gap in detection capabilities, we present FakePartsBench, the first large-scale benchmark dataset specifically designed to capture the full spectrum of partial deepfakes. Comprising over 25K videos with pixel-level and frame-level manipulation annotations, our dataset enables comprehensive evaluation of detection methods. Our user studies demonstrate that FakeParts reduces human detection accuracy by over 30% compared to traditional deepfakes, with similar performance degradation observed in state-of-the-art detection models. This work identifies an urgent vulnerability in current deepfake detection approaches and provides the necessary resources to develop more robust methods for partial video manipulations. 

**Abstract (ZH)**: FakeParts: 新颖的局部操纵深伪类及其检测基准 FakePartsBench 

---
# Enabling Equitable Access to Trustworthy Financial Reasoning 

**Title (ZH)**: 实现可信财务推理的公平访问 

**Authors**: William Jurayj, Nils Holzenberger, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2508.21051)  

**Abstract**: According to the United States Internal Revenue Service, ''the average American spends $\$270$ and 13 hours filing their taxes''. Even beyond the U.S., tax filing requires complex reasoning, combining application of overlapping rules with numerical calculations. Because errors can incur costly penalties, any automated system must deliver high accuracy and auditability, making modern large language models (LLMs) poorly suited for this task. We propose an approach that integrates LLMs with a symbolic solver to calculate tax obligations. We evaluate variants of this system on the challenging StAtutory Reasoning Assessment (SARA) dataset, and include a novel method for estimating the cost of deploying such a system based on real-world penalties for tax errors. We further show how combining up-front translation of plain-text rules into formal logic programs, combined with intelligently retrieved exemplars for formal case representations, can dramatically improve performance on this task and reduce costs to well below real-world averages. Our results demonstrate the promise and economic feasibility of neuro-symbolic architectures for increasing equitable access to reliable tax assistance. 

**Abstract (ZH)**: 根据美国国内税务局的数据，"普通美国人平均花费270美元和13小时来申报税表"。即使超出美国范围，税务申报也需要复杂的推理，结合重叠规则的应用和数值计算。由于错误可能导致高额罚款，任何自动系统必须提供高准确性和可审计性，这使得现代大型语言模型（LLMs）不适合这项任务。我们提出了一种将大型语言模型与符号求解器集成以计算税务义务的方法。我们在具有挑战性的StAtutory Reasoning Assessment (SARA)数据集上评估了该系统的不同变体，并引入了一种基于税务错误真实世界罚款的新方法来估算部署此类系统所需的成本。我们进一步展示了将前期将文本规则翻译成形式化逻辑程序并与智能检索的形式化案例代表示例相结合，可以显著提高此任务的性能并降低成本至远低于真实世界平均水平。我们的结果显示，神经-符号架构在提高可靠税务援助的公平获取方面具有前景和经济可行性。 

---
# Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning 

**Title (ZH)**: Veritas: 基于模式感知推理的泛化深伪检测 

**Authors**: Hao Tan, Jun Lan, Zichang Tan, Ajian Liu, Chuanbiao Song, Senyuan Shi, Huijia Zhu, Weiqiang Wang, Jun Wan, Zhen Lei  

**Link**: [PDF](https://arxiv.org/pdf/2508.21048)  

**Abstract**: Deepfake detection remains a formidable challenge due to the complex and evolving nature of fake content in real-world scenarios. However, existing academic benchmarks suffer from severe discrepancies from industrial practice, typically featuring homogeneous training sources and low-quality testing images, which hinder the practical deployments of current detectors. To mitigate this gap, we introduce HydraFake, a dataset that simulates real-world challenges with hierarchical generalization testing. Specifically, HydraFake involves diversified deepfake techniques and in-the-wild forgeries, along with rigorous training and evaluation protocol, covering unseen model architectures, emerging forgery techniques and novel data domains. Building on this resource, we propose Veritas, a multi-modal large language model (MLLM) based deepfake detector. Different from vanilla chain-of-thought (CoT), we introduce pattern-aware reasoning that involves critical reasoning patterns such as "planning" and "self-reflection" to emulate human forensic process. We further propose a two-stage training pipeline to seamlessly internalize such deepfake reasoning capacities into current MLLMs. Experiments on HydraFake dataset reveal that although previous detectors show great generalization on cross-model scenarios, they fall short on unseen forgeries and data domains. Our Veritas achieves significant gains across different OOD scenarios, and is capable of delivering transparent and faithful detection outputs. 

**Abstract (ZH)**: HydraFake：一种模拟现实挑战的多层次泛化测试集及基于多模态大语言模型的Deepfake检测方法 

---
# Understanding, Protecting, and Augmenting Human Cognition with Generative AI: A Synthesis of the CHI 2025 Tools for Thought Workshop 

**Title (ZH)**: 利用生成式AI理解、保护和增强人类认知：CHI 2025 工具思维研讨会综述 

**Authors**: Lev Tankelevitch, Elena L. Glassman, Jessica He, Aniket Kittur, Mina Lee, Srishti Palani, Advait Sarkar, Gonzalo Ramos, Yvonne Rogers, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2508.21036)  

**Abstract**: Generative AI (GenAI) radically expands the scope and capability of automation for work, education, and everyday tasks, a transformation posing both risks and opportunities for human cognition. How will human cognition change, and what opportunities are there for GenAI to augment it? Which theories, metrics, and other tools are needed to address these questions? The CHI 2025 workshop on Tools for Thought aimed to bridge an emerging science of how the use of GenAI affects human thought, from metacognition to critical thinking, memory, and creativity, with an emerging design practice for building GenAI tools that both protect and augment human thought. Fifty-six researchers, designers, and thinkers from across disciplines as well as industry and academia, along with 34 papers and portfolios, seeded a day of discussion, ideation, and community-building. We synthesize this material here to begin mapping the space of research and design opportunities and to catalyze a multidisciplinary community around this pressing area of research. 

**Abstract (ZH)**: Generative AI (GenAI) 对工作、教育和日常任务的自动化范围和能力产生了根本性扩展，这一变革为人类认知带来了风险与机遇。人类认知将如何变化，GenAI又有何增强潜力？需要哪些理论、指标和其他工具来应对这些问题？CHI 2025 工作坊“思维工具”旨在连接关于GenAI使用如何影响人类思维（包括元认知、批判性思维、记忆和创造力）的研究科学与新兴的GenAI工具设计实践，该实践既保护也增强人类思维。来自多个学科及产业和学术界的56名研究人员、设计师和思想家，以及34篇论文和作品集，激发了一天的讨论、创意思考和社区构建。我们在此整理这些材料，以开始绘制研究和设计机会的空间图，并催化围绕这一紧迫研究领域的跨学科社区。 

---
# Inference-Time Alignment Control for Diffusion Models with Reinforcement Learning Guidance 

**Title (ZH)**: 基于强化学习指导的扩散模型 inference 时对齐控制 

**Authors**: Luozhijie Jin, Zijie Qiu, Jie Liu, Zijie Diao, Lifeng Qiao, Ning Ding, Alex Lamb, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21016)  

**Abstract**: Denoising-based generative models, particularly diffusion and flow matching algorithms, have achieved remarkable success. However, aligning their output distributions with complex downstream objectives, such as human preferences, compositional accuracy, or data compressibility, remains challenging. While reinforcement learning (RL) fine-tuning methods, inspired by advances in RL from human feedback (RLHF) for large language models, have been adapted to these generative frameworks, current RL approaches are suboptimal for diffusion models and offer limited flexibility in controlling alignment strength after fine-tuning. In this work, we reinterpret RL fine-tuning for diffusion models through the lens of stochastic differential equations and implicit reward conditioning. We introduce Reinforcement Learning Guidance (RLG), an inference-time method that adapts Classifier-Free Guidance (CFG) by combining the outputs of the base and RL fine-tuned models via a geometric average. Our theoretical analysis shows that RLG's guidance scale is mathematically equivalent to adjusting the KL-regularization coefficient in standard RL objectives, enabling dynamic control over the alignment-quality trade-off without further training. Extensive experiments demonstrate that RLG consistently improves the performance of RL fine-tuned models across various architectures, RL algorithms, and downstream tasks, including human preferences, compositional control, compressibility, and text rendering. Furthermore, RLG supports both interpolation and extrapolation, thereby offering unprecedented flexibility in controlling generative alignment. Our approach provides a practical and theoretically sound solution for enhancing and controlling diffusion model alignment at inference. The source code for RLG is publicly available at the Github: this https URL. 

**Abstract (ZH)**: 基于去噪的生成模型，尤其是扩散和流匹配算法，已经取得了显著的成功。然而，将它们的输出分布与人类偏好、组分准确性或数据压缩性等复杂的下游目标对齐仍然具有挑战性。尽管借鉴了大规模语言模型从人类反馈强化学习（RLHF）中获得的进展，现有的强化学习（RL）微调方法已经被应用于这些生成框架中，但当前的RL方法并不适用于扩散模型，并且在微调后控制对齐强度方面的灵活性有限。在本文中，我们通过随机微分方程和隐式奖励条件的角度重新解释扩散模型的强化学习微调。我们引入了强化学习指导（RLG），这是一种推理时的方法，通过几何平均结合基模型和RL微调模型的输出来重新解释无分类器引导（CFG）。我们的理论分析表明，RLG的引导尺度数学上等同于调整标准RL目标中的KL正则化系数，从而使在不进行进一步训练的情况下动态控制对齐质量成为可能。广泛的实验表明，RLG能够在各种架构、RL算法和下游任务（包括人类偏好、组成控制、压缩性和文本渲染）中一致地提升RL微调模型的性能。此外，RLG支持插值和外推，从而提供了前所未有的控制生成对齐的灵活性。我们的方法提供了一种在推理时增强和控制扩散模型对齐的实用且有理论依据的解决方案。RLG的源代码已在Github上公开：this https URL。 

---
# ChainReaction! Structured Approach with Causal Chains as Intermediate Representations for Improved and Explainable Causal Video Question Answering 

**Title (ZH)**: Chain Reaction！基于因果链作为中间表示的结构化方法以提高可解释的因果视频问答 

**Authors**: Paritosh Parmar, Eric Peh, Basura Fernando  

**Link**: [PDF](https://arxiv.org/pdf/2508.21010)  

**Abstract**: Existing Causal-Why Video Question Answering (VideoQA) models often struggle with higher-order reasoning, relying on opaque, monolithic pipelines that entangle video understanding, causal inference, and answer generation. These black-box approaches offer limited interpretability and tend to depend on shallow heuristics. We propose a novel, modular framework that explicitly decouples causal reasoning from answer generation, introducing natural language causal chains as interpretable intermediate representations. Inspired by human cognitive models, these structured cause-effect sequences bridge low-level video content with high-level causal reasoning, enabling transparent and logically coherent inference. Our two-stage architecture comprises a Causal Chain Extractor (CCE) that generates causal chains from video-question pairs, and a Causal Chain-Driven Answerer (CCDA) that produces answers grounded in these chains. To address the lack of annotated reasoning traces, we introduce a scalable method for generating high-quality causal chains from existing datasets using large language models. We also propose CauCo, a new evaluation metric for causality-oriented captioning. Experiments on three large-scale benchmarks demonstrate that our approach not only outperforms state-of-the-art models, but also yields substantial gains in explainability, user trust, and generalization -- positioning the CCE as a reusable causal reasoning engine across diverse domains. Project page: this https URL 

**Abstract (ZH)**: 现有的因果解释视频问答（VideoQA）模型常常难以处理高阶推理，依赖于将视频理解、因果推理和答案生成纠缠在一起的不透明单一管道。这些黑盒方法提供有限的解释性，并倾向于依赖浅层启发式方法。我们提出了一种新的模块化框架，明确地将因果推理与答案生成分离，并引入可解释的因果链作为中间表示。受人类认知模型的启发，这些结构化的因果序列将低层级视频内容与高层级因果推理相连，使得推理具有透明性和逻辑一致性。我们的两阶段架构包括一个因果链提取器（CCE），它从视频-问题对中生成因果链，以及一个因果链驱动的答案生成器（CCDA），它基于这些链生成答案。为了解决缺少标注推理轨迹的问题，我们提出了使用大型语言模型从现有数据集中生成高质量因果链的可扩展方法。我们还提出了CauCo，一种新的因果导向的标题评估指标。在三个大规模基准上的实验表明，我们的方法不仅优于现有最佳模型，还在解释性、用户信任度和泛化能力方面取得了显著提升——这将CCE定位为跨多种领域的可重用因果推理引擎。 

---
# Train-Once Plan-Anywhere Kinodynamic Motion Planning via Diffusion Trees 

**Title (ZH)**: 一次训练，随处规划：基于扩散树的kinodynamic运动规划 

**Authors**: Yaniv Hassidof, Tom Jurgenson, Kiril Solovey  

**Link**: [PDF](https://arxiv.org/pdf/2508.21001)  

**Abstract**: Kinodynamic motion planning is concerned with computing collision-free trajectories while abiding by the robot's dynamic constraints. This critical problem is often tackled using sampling-based planners (SBPs) that explore the robot's high-dimensional state space by constructing a search tree via action propagations. Although SBPs can offer global guarantees on completeness and solution quality, their performance is often hindered by slow exploration due to uninformed action sampling. Learning-based approaches can yield significantly faster runtimes, yet they fail to generalize to out-of-distribution (OOD) scenarios and lack critical guarantees, e.g., safety, thus limiting their deployment on physical robots. We present Diffusion Tree (DiTree): a \emph{provably-generalizable} framework leveraging diffusion policies (DPs) as informed samplers to efficiently guide state-space search within SBPs. DiTree combines DP's ability to model complex distributions of expert trajectories, conditioned on local observations, with the completeness of SBPs to yield \emph{provably-safe} solutions within a few action propagation iterations for complex dynamical systems. We demonstrate DiTree's power with an implementation combining the popular RRT planner with a DP action sampler trained on a \emph{single environment}. In comprehensive evaluations on OOD scenarios, % DiTree has comparable runtimes to a standalone DP (3x faster than classical SBPs), while improving the average success rate over DP and SBPs. DiTree is on average 3x faster than classical SBPs, and outperforms all other approaches by achieving roughly 30\% higher success rate. Project webpage: this https URL. 

**Abstract (ZH)**: 基于扩散策略的扩散树（DiTree）：一种可证明泛化的动态规划框架 

---
# ExpertSim: Fast Particle Detector Simulation Using Mixture-of-Generative-Experts 

**Title (ZH)**: ExpertSim: 快速粒子探测器仿真用混合生成专家模型 

**Authors**: Patryk Będkowski, Jan Dubiński, Filip Szatkowski, Kamil Deja, Przemysław Rokita, Tomasz Trzciński  

**Link**: [PDF](https://arxiv.org/pdf/2508.20991)  

**Abstract**: Simulating detector responses is a crucial part of understanding the inner workings of particle collisions in the Large Hadron Collider at CERN. Such simulations are currently performed with statistical Monte Carlo methods, which are computationally expensive and put a significant strain on CERN's computational grid. Therefore, recent proposals advocate for generative machine learning methods to enable more efficient simulations. However, the distribution of the data varies significantly across the simulations, which is hard to capture with out-of-the-box methods. In this study, we present ExpertSim - a deep learning simulation approach tailored for the Zero Degree Calorimeter in the ALICE experiment. Our method utilizes a Mixture-of-Generative-Experts architecture, where each expert specializes in simulating a different subset of the data. This allows for a more precise and efficient generation process, as each expert focuses on a specific aspect of the calorimeter response. ExpertSim not only improves accuracy, but also provides a significant speedup compared to the traditional Monte-Carlo methods, offering a promising solution for high-efficiency detector simulations in particle physics experiments at CERN. We make the code available at this https URL. 

**Abstract (ZH)**: 在欧洲核子研究组织（CERN）的大型强子对撞机（LHC）的ALICE实验中，零度角 calorimeter 的深度学习仿真方法 

---
# WoW-Bench: Evaluating Fine-Grained Acoustic Perception in Audio-Language Models via Marine Mammal Vocalizations 

**Title (ZH)**: WoW-Bench: 通过海洋哺乳动物 vocalizations 评估音频语言模型的细粒度声学感知能力 

**Authors**: Jaeyeon Kim, Heeseung Yun, Sang Hoon Woo, Chao-Han Huck Yang, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20976)  

**Abstract**: Large audio language models (LALMs) extend language understanding into the auditory domain, yet their ability to perform low-level listening, such as pitch and duration detection, remains underexplored. However, low-level listening is critical for real-world, out-of-distribution tasks where models must reason about unfamiliar sounds based on fine-grained acoustic cues. To address this gap, we introduce the World-of-Whale benchmark (WoW-Bench) to evaluate low-level auditory perception and cognition using marine mammal vocalizations. WoW-bench is composed of a Perception benchmark for categorizing novel sounds and a Cognition benchmark, inspired by Bloom's taxonomy, to assess the abilities to remember, understand, apply, and analyze sound events. For the Cognition benchmark, we additionally introduce distractor questions to evaluate whether models are truly solving problems through listening rather than relying on other heuristics. Experiments with state-of-the-art LALMs show performance far below human levels, indicating a need for stronger auditory grounding in LALMs. 

**Abstract (ZH)**: 大规模音频语言模型（LALMs）将语言理解扩展到了听觉领域，但其在低级听觉任务上的能力，如音高和时长检测，仍然被广泛忽视。为了填补这一空白，我们引入了鲸世界基准（WoW-Bench），通过海洋哺乳动物的 vocalizations 来评估低级听觉感知和认知能力。WoW-Bench 包括一个感知基准用于分类新型声音，以及一个借鉴布卢姆 taxonomy 设计的认知基准，用以评估模型对声音事件的记忆、理解、应用和分析能力。为了评估认知基准，我们还引入了干扰问题，以检验模型是否真正通过听觉解决问题，而不是依赖其他启发式方法。实验结果显示，最先进的 LALMs 的表现远低于人类水平，这表明 LALMs 需要更强的听觉基础。 

---
# ProactiveEval: A Unified Evaluation Framework for Proactive Dialogue Agents 

**Title (ZH)**: 前瞻Eval：统一的前瞻对话代理评估框架 

**Authors**: Tianjian Liu, Fanqi Wan, Jiajian Guo, Xiaojun Quan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20973)  

**Abstract**: Proactive dialogue has emerged as a critical and challenging research problem in advancing large language models (LLMs). Existing works predominantly focus on domain-specific or task-oriented scenarios, which leads to fragmented evaluations and limits the comprehensive exploration of models' proactive conversation abilities. In this work, we propose ProactiveEval, a unified framework designed for evaluating proactive dialogue capabilities of LLMs. This framework decomposes proactive dialogue into target planning and dialogue guidance, establishing evaluation metrics across various domains. Moreover, it also enables the automatic generation of diverse and challenging evaluation data. Based on the proposed framework, we develop 328 evaluation environments spanning 6 distinct domains. Through experiments with 22 different types of LLMs, we show that DeepSeek-R1 and Claude-3.7-Sonnet exhibit exceptional performance on target planning and dialogue guidance tasks, respectively. Finally, we investigate how reasoning capabilities influence proactive behaviors and discuss their implications for future model development. 

**Abstract (ZH)**: 主动对话已成为促进大规模语言模型（LLMs）发展的关键性和挑战性研究问题。现有工作主要关注领域特定或任务导向的场景，这导致了评估的碎片化并限制了对模型主动对话能力的全面探索。在本文中，我们提出了一种名为ProactiveEval的统一框架，旨在评估LLMs的主动对话能力。该框架将主动对话分解为目标规划和对话指导，并在各种领域建立了评估指标。此外，它还能够自动生成多样化和具有挑战性的评估数据。基于提出的框架，我们开发了涵盖6个不同领域的328个评估环境。通过与22种不同类型的LLM进行的实验，我们展示了DeepSeek-R1和Claude-3.7-Sonnet分别在目标规划和对话指导任务中表现出色。最后，我们探讨了推理能力对主动行为的影响，并讨论了其对未来模型开发的意义。 

---
# Research Challenges in Relational Database Management Systems for LLM Queries 

**Title (ZH)**: 面向大型语言模型查询的关系数据库管理系统研究挑战 

**Authors**: Kerem Akillioglu, Anurag Chakraborty, Sairaj Voruganti, M. Tamer Özsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20912)  

**Abstract**: Large language models (LLMs) have become essential for applications such as text summarization, sentiment analysis, and automated question-answering. Recently, LLMs have also been integrated into relational database management systems to enhance querying and support advanced data processing. Companies such as Amazon, Databricks, Google, and Snowflake offer LLM invocation directly within SQL, denoted as LLM queries, to boost data insights. However, open-source solutions currently have limited functionality and poor performance. In this work, we present an early exploration of two open-source systems and one enterprise platform, using five representative queries to expose functional, performance, and scalability limits in today's SQL-invoked LLM integrations. We identify three main issues: enforcing structured outputs, optimizing resource utilization, and improving query planning. We implemented initial solutions and observed improvements in accommodating LLM powered SQL queries. These early gains demonstrate that tighter integration of LLM+DBMS is the key to scalable and efficient processing of LLM queries. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为文本摘要、情感分析和自动问答等应用中的重要组成部分。最近，LLMs 还被集成到关系数据库管理系统中以增强查询并支持高级数据处理。Amazon、Databricks、Google 和 Snowflake 等公司提供直接在 SQL 中调用 LLMs 的服务，称为 LLMS 查询，以提升数据洞察力。然而，当前开源解决方案的功能有限且性能较差。本文介绍了对两个开源系统和一个企业平台的初步探索，使用五个代表性查询来揭示当今 SQL 调用 LLMS 集成的功能、性能和扩展性限制。我们识别出三个主要问题：强制结构化输出、优化资源利用和改进查询规划。我们实现了初步解决方案，并观察到在处理 LLMS 支持的 SQL 查询时有所改进。这些初步成果表明，LLM+DBMS 的更紧密集成是实现 LLMS 查询可扩展和高效处理的关键。 

---
# Quantum Verifiable Rewards for Post-Training Qiskit Code Assistant 

**Title (ZH)**: 量子可验证奖励后训练Qiskit代码助手 

**Authors**: Nicolas Dupuis, Adarsh Tiwari, Youssef Mroueh, David Kremer, Ismael Faro, Juan Cruz-Benito  

**Link**: [PDF](https://arxiv.org/pdf/2508.20907)  

**Abstract**: Qiskit is an open-source quantum computing framework that allows users to design, simulate, and run quantum circuits on real quantum hardware. We explore post-training techniques for LLMs to assist in writing Qiskit code. We introduce quantum verification as an effective method for ensuring code quality and executability on quantum hardware. To support this, we developed a synthetic data pipeline that generates quantum problem-unit test pairs and used it to create preference data for aligning LLMs with DPO. Additionally, we trained models using GRPO, leveraging quantum-verifiable rewards provided by the quantum hardware. Our best-performing model, combining DPO and GRPO, surpasses the strongest open-source baselines on the challenging Qiskit-HumanEval-hard benchmark. 

**Abstract (ZH)**: Qiskit是一种开放源代码的量子计算框架，允许用户设计、模拟和在实际量子硬件上运行量子电路。我们探索了用于辅助编写Qiskit代码的LLM后训练技术。我们介绍了量子验证作为一种有效方法，用于确保代码在量子硬件上的质量和可执行性。为支持这一目标，我们开发了一个合成数据管道，生成量子问题单元测试对，并使用它来创建偏好数据以使LLMs与DPO对齐。此外，我们使用GRPO训练模型，并利用量子硬件提供的量子验证奖励。我们表现最好的模型，结合了DPO和GRPO，在具有挑战性的Qiskit-HumanEval-hard基准测试中超越了最强的开源基线。 

---
# AI Agentic Vulnerability Injection And Transformation with Optimized Reasoning 

**Title (ZH)**: AI自主漏洞注入与优化推理转化 

**Authors**: Amine Lbath, Massih-Reza Amini, Aurelien Delaitre, Vadim Okun  

**Link**: [PDF](https://arxiv.org/pdf/2508.20866)  

**Abstract**: The increasing complexity of software systems and the sophistication of cyber-attacks have underscored the critical need for effective automated vulnerability detection and repair systems. Traditional methods, such as static program analysis, face significant challenges related to scalability, adaptability, and high false-positive and false-negative rates. AI-driven approaches, particularly those using machine learning and deep learning models, show promise but are heavily reliant on the quality and quantity of training data. This paper introduces a novel framework designed to automatically introduce realistic, category-specific vulnerabilities into secure C/C++ codebases to generate datasets. The proposed approach coordinates multiple AI agents that simulate expert reasoning, along with function agents and traditional code analysis tools. It leverages Retrieval-Augmented Generation for contextual grounding and employs Low-Rank approximation of weights for efficient model fine-tuning. Our experimental study on 116 code samples from three different benchmarks suggests that our approach outperforms other techniques with regard to dataset accuracy, achieving between 89\% and 95\% success rates in injecting vulnerabilities at function level. 

**Abstract (ZH)**: 软件系统日益复杂的程度和网络攻击的 sophistication 加强了高效自动化漏洞检测与修复系统的需求。传统方法，如静态程序分析，面临可扩展性、适应性以及高误报和漏报率的显著挑战。基于 AI 的方法，特别是利用机器学习和深度学习模型的方法显示出潜力，但对训练数据的质量和数量高度依赖。本文提出了一种新型框架，旨在自动生成现实的、类别特定的漏洞，注入到安全的 C/C++ 代码库中以生成数据集。所提出的方法协调多个 AI 代理以模拟专家推理，结合函数代理和传统代码分析工具。该方法利用检索增强生成进行上下文定位，并采用低秩权重逼近进行高效模型微调。在对来自三个不同基准的 116 个代码样本进行的实验研究中，我们的方法在数据集准确性方面优于其他技术，在函数级别注入漏洞的成功率在 89% 至 95% 之间。 

---
# JADES: A Universal Framework for Jailbreak Assessment via Decompositional Scoring 

**Title (ZH)**: JADES: 一种基于分解评分的通用 Jailbreak 评估框架 

**Authors**: Junjie Chu, Mingjie Li, Ziqing Yang, Ye Leng, Chenhao Lin, Chao Shen, Michael Backes, Yun Shen, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20848)  

**Abstract**: Accurately determining whether a jailbreak attempt has succeeded is a fundamental yet unresolved challenge. Existing evaluation methods rely on misaligned proxy indicators or naive holistic judgments. They frequently misinterpret model responses, leading to inconsistent and subjective assessments that misalign with human perception. To address this gap, we introduce JADES (Jailbreak Assessment via Decompositional Scoring), a universal jailbreak evaluation framework. Its key mechanism is to automatically decompose an input harmful question into a set of weighted sub-questions, score each sub-answer, and weight-aggregate the sub-scores into a final decision. JADES also incorporates an optional fact-checking module to strengthen the detection of hallucinations in jailbreak responses. We validate JADES on JailbreakQR, a newly introduced benchmark proposed in this work, consisting of 400 pairs of jailbreak prompts and responses, each meticulously annotated by humans. In a binary setting (success/failure), JADES achieves 98.5% agreement with human evaluators, outperforming strong baselines by over 9%. Re-evaluating five popular attacks on four LLMs reveals substantial overestimation (e.g., LAA's attack success rate on GPT-3.5-Turbo drops from 93% to 69%). Our results show that JADES could deliver accurate, consistent, and interpretable evaluations, providing a reliable basis for measuring future jailbreak attacks. 

**Abstract (ZH)**: 准确确定越狱尝试是否成功是一个基本但尚未解决的挑战。现有的评估方法依赖于不对齐的代理指标或直觉的整体判断。它们经常误解释模型的响应，导致不一致且主观的评估，这与人类感知不符。为解决这一差距，我们引入了JADES（越狱评估通过分解评分），这是一种通用的越狱评估框架。其关键机制是自动将输入的有害问题分解为一组加权子问题，对每个子回答进行评分，并将子评分加权聚合为最终决策。JADES还包含一个可选的事实核查模块，以增强对越狱响应中幻觉的检测。我们在本文提出的新基准JailbreakQR上验证了JADES，该基准包括400对越狱提示和响应，每一对都由人类细致标注。在二元设置（成功/失败）下，JADES在人评分者中的一致率为98.5%，优于强 baselines超过9%。重新评估五个流行的攻击在四个LLM上的效果显示了显著的高估（例如，LAA对GPT-3.5-Turbo的攻击成功率从93%下降到69%）。我们的结果表明，JADES能够提供准确、一致和可解释的评估，为衡量未来越狱攻击提供可靠的依据。 

---
# Learning Primitive Embodied World Models: Towards Scalable Robotic Learning 

**Title (ZH)**: 学习基础的具身世界模型：通往可扩展的机器人学习的道路 

**Authors**: Qiao Sun, Liujia Yang, Wei Tang, Wei Huang, Kaixin Xu, Yongchao Chen, Mingyu Liu, Jiange Yang, Haoyi Zhu, Yating Wang, Tong He, Yilun Chen, Xili Dai, Nanyang Ye, Qinying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20840)  

**Abstract**: While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a "GPT moment" in the embodied domain. There is a naive observation: the diversity of embodied data far exceeds the relatively small space of possible primitive motions. Based on this insight, we propose a novel paradigm for world modeling--Primitive Embodied World Models (PEWM). By restricting video generation to fixed short horizons, our approach 1) enables fine-grained alignment between linguistic concepts and visual representations of robotic actions, 2) reduces learning complexity, 3) improves data efficiency in embodied data collection, and 4) decreases inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence. 

**Abstract (ZH)**: 基于视频生成的体态世界模型：一种新的体态数据稀疏性应对范式（Primitive Embodied World Models） 

---
# Multi-Agent Penetration Testing AI for the Web 

**Title (ZH)**: 网络空间多代理渗透测试人工智能 

**Authors**: Isaac David, Arthur Gervais  

**Link**: [PDF](https://arxiv.org/pdf/2508.20816)  

**Abstract**: AI-powered development platforms are making software creation accessible to a broader audience, but this democratization has triggered a scalability crisis in security auditing. With studies showing that up to 40% of AI-generated code contains vulnerabilities, the pace of development now vastly outstrips the capacity for thorough security assessment.
We present MAPTA, a multi-agent system for autonomous web application security assessment that combines large language model orchestration with tool-grounded execution and end-to-end exploit validation. On the 104-challenge XBOW benchmark, MAPTA achieves 76.9% overall success with perfect performance on SSRF and misconfiguration vulnerabilities, 83% success on broken authorization, and strong results on injection attacks including server-side template injection (85%) and SQL injection (83%). Cross-site scripting (57%) and blind SQL injection (0%) remain challenging. Our comprehensive cost analysis across all challenges totals $21.38 with a median cost of $0.073 for successful attempts versus $0.357 for failures. Success correlates strongly with resource efficiency, enabling practical early-stopping thresholds at approximately 40 tool calls or $0.30 per challenge.
MAPTA's real-world findings are impactful given both the popularity of the respective scanned GitHub repositories (8K-70K stars) and MAPTA's low average operating cost of $3.67 per open-source assessment: MAPTA discovered critical vulnerabilities including RCEs, command injections, secret exposure, and arbitrary file write vulnerabilities. Findings are responsibly disclosed, 10 findings are under CVE review. 

**Abstract (ZH)**: 基于多代理系统的自主网页应用安全评估平台MAPTA 

---
# Uncertainty Aware-Predictive Control Barrier Functions: Safer Human Robot Interaction through Probabilistic Motion Forecasting 

**Title (ZH)**: 不确定性意识预测控制屏障函数：通过概率性运动预测实现更安全的人机交互 

**Authors**: Lorenzo Busellato, Federico Cunico, Diego Dall'Alba, Marco Emporio, Andrea Giachetti, Riccardo Muradore, Marco Cristani  

**Link**: [PDF](https://arxiv.org/pdf/2508.20812)  

**Abstract**: To enable flexible, high-throughput automation in settings where people and robots share workspaces, collaborative robotic cells must reconcile stringent safety guarantees with the need for responsive and effective behavior. A dynamic obstacle is the stochastic, task-dependent variability of human motion: when robots fall back on purely reactive or worst-case envelopes, they brake unnecessarily, stall task progress, and tamper with the fluidity that true Human-Robot Interaction demands. In recent years, learning-based human-motion prediction has rapidly advanced, although most approaches produce worst-case scenario forecasts that often do not treat prediction uncertainty in a well-structured way, resulting in over-conservative planning algorithms, limiting their flexibility. We introduce Uncertainty-Aware Predictive Control Barrier Functions (UA-PCBFs), a unified framework that fuses probabilistic human hand motion forecasting with the formal safety guarantees of Control Barrier Functions. In contrast to other variants, our framework allows for dynamic adjustment of the safety margin thanks to the human motion uncertainty estimation provided by a forecasting module. Thanks to uncertainty estimation, UA-PCBFs empower collaborative robots with a deeper understanding of future human states, facilitating more fluid and intelligent interactions through informed motion planning. We validate UA-PCBFs through comprehensive real-world experiments with an increasing level of realism, including automated setups (to perform exactly repeatable motions) with a robotic hand and direct human-robot interactions (to validate promptness, usability, and human confidence). Relative to state-of-the-art HRI architectures, UA-PCBFs show better performance in task-critical metrics, significantly reducing the number of violations of the robot's safe space during interaction with respect to the state-of-the-art. 

**Abstract (ZH)**: 基于不确定性意识的预测控制屏障函数：一种融合概率性 humans 手部运动预测与控制屏障函数正式安全保证的统一框架 

---
# Exploring Machine Learning and Language Models for Multimodal Depression Detection 

**Title (ZH)**: 探索机器学习和语言模型在 multimodal 抑郁检测中的应用 

**Authors**: Javier Si Zhao Hong, Timothy Zoe Delaya, Sherwyn Chan Yin Kit, Pai Chet Ng, Xiaoxiao Miao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20805)  

**Abstract**: This paper presents our approach to the first Multimodal Personality-Aware Depression Detection Challenge, focusing on multimodal depression detection using machine learning and deep learning models. We explore and compare the performance of XGBoost, transformer-based architectures, and large language models (LLMs) on audio, video, and text features. Our results highlight the strengths and limitations of each type of model in capturing depression-related signals across modalities, offering insights into effective multimodal representation strategies for mental health prediction. 

**Abstract (ZH)**: 本文提出了我们参加首次多模态个性意识抑郁症检测挑战赛的方法，重点在于使用机器学习和深度学习模型进行多模态抑郁症检测。我们探索并比较了XGBoost、基于变换器的架构以及大语言模型（LLMs）在音频、视频和文本特征上的性能，结果显示了每种类型模型在捕捉跨模态的抑郁症相关信号方面的优势与局限性，为我们提供了有关精神健康预测的有效多模态表示策略的见解。 

---
# Speech Emotion Recognition via Entropy-Aware Score Selection 

**Title (ZH)**: 基于熵感知评分选择的语音情绪识别 

**Authors**: ChenYi Chua, JunKai Wong, Chengxin Chen, Xiaoxiao Miao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20796)  

**Abstract**: In this paper, we propose a multimodal framework for speech emotion recognition that leverages entropy-aware score selection to combine speech and textual predictions. The proposed method integrates a primary pipeline that consists of an acoustic model based on wav2vec2.0 and a secondary pipeline that consists of a sentiment analysis model using RoBERTa-XLM, with transcriptions generated via Whisper-large-v3. We propose a late score fusion approach based on entropy and varentropy thresholds to overcome the confidence constraints of primary pipeline predictions. A sentiment mapping strategy translates three sentiment categories into four target emotion classes, enabling coherent integration of multimodal predictions. The results on the IEMOCAP and MSP-IMPROV datasets show that the proposed method offers a practical and reliable enhancement over traditional single-modality systems. 

**Abstract (ZH)**: 本文提出一种熵意识评分选择的多模态框架用于语音情绪识别，并结合语音和文本预测。所提出的方法包含一个基于wav2vec2.0的声音模型为主的管道和一个使用RoBERTa-XLM进行情感分析的次级管道，通过Whisper-large-v3生成转录。我们提出了一种基于熵和变熵阈值的后期评分融合方法，以克服主要管道预测的信心约束。情感映射策略将三种情感类别转换为四种目标情绪类别，使多模态预测得以一致融合。在IEMOCAP和MSP-IMPROV数据集上的结果表明，所提出的方法比传统的单模态系统提供了实用且可靠的改进。 

---
# Surfel-based 3D Registration with Equivariant SE(3) Features 

**Title (ZH)**: 基于Surfel的3D注册_with_不变SE(3)特征 

**Authors**: Xueyang Kang, Hang Zhao, Kourosh Khoshelham, Patrick Vandewalle  

**Link**: [PDF](https://arxiv.org/pdf/2508.20789)  

**Abstract**: Point cloud registration is crucial for ensuring 3D alignment consistency of multiple local point clouds in 3D reconstruction for remote sensing or digital heritage. While various point cloud-based registration methods exist, both non-learning and learning-based, they ignore point orientations and point uncertainties, making the model susceptible to noisy input and aggressive rotations of the input point cloud like orthogonal transformation; thus, it necessitates extensive training point clouds with transformation augmentations. To address these issues, we propose a novel surfel-based pose learning regression approach. Our method can initialize surfels from Lidar point cloud using virtual perspective camera parameters, and learns explicit $\mathbf{SE(3)}$ equivariant features, including both position and rotation through $\mathbf{SE(3)}$ equivariant convolutional kernels to predict relative transformation between source and target scans. The model comprises an equivariant convolutional encoder, a cross-attention mechanism for similarity computation, a fully-connected decoder, and a non-linear Huber loss. Experimental results on indoor and outdoor datasets demonstrate our model superiority and robust performance on real point-cloud scans compared to state-of-the-art methods. 

**Abstract (ZH)**: 基于surfel的Pose学习回归方法在远程 sensing或数字遗产3D重建中多局部点云的3D对齐一致性保障中至关重要。虽然存在多种基于点云的注册方法，无论是非学习还是学习-Based，但它们忽略了点的姿态和点的不确定性，使得模型对噪声输入和输入点云的剧烈旋转（如正交变换）尤为敏感；因而需要采用具有变换增强的大量训练点云。为解决这些问题，我们提出了一种新颖的基于surfel的Pose学习回归方法。该方法可以使用虚拟视角摄像机参数从激光点云中初始化surfels，并学习显式的$\mathbf{SE(3)}$不变特征，包括通过$\mathbf{SE(3)}$不变卷积核预测源和目标扫描之间的相对变换，同时包含了位置和旋转信息。模型由一个$\mathbf{SE(3)}$不变卷积编码器、用于相似度计算的交叉注意机制、一个全连接解码器以及一个非线性Huber损失组成。在室内和室外数据集上的实验结果表明，该模型在实际点云扫描中优于现有方法，展现了优越性和鲁棒性。 

---
# Evaluating Compositional Generalisation in VLMs and Diffusion Models 

**Title (ZH)**: 评估VLMs和扩散模型的组合泛化能力 

**Authors**: Beth Pearson, Bilal Boulbarss, Michael Wray, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2508.20783)  

**Abstract**: A fundamental aspect of the semantics of natural language is that novel meanings can be formed from the composition of previously known parts. Vision-language models (VLMs) have made significant progress in recent years, however, there is evidence that they are unable to perform this kind of composition. For example, given an image of a red cube and a blue cylinder, a VLM such as CLIP is likely to incorrectly label the image as a red cylinder or a blue cube, indicating it represents the image as a `bag-of-words' and fails to capture compositional semantics. Diffusion models have recently gained significant attention for their impressive generative abilities, and zero-shot classifiers based on diffusion models have been shown to perform competitively with CLIP in certain compositional tasks. In this work we explore whether the generative Diffusion Classifier has improved compositional generalisation abilities compared to discriminative models. We assess three models -- Diffusion Classifier, CLIP, and ViLT -- on their ability to bind objects with attributes and relations in both zero-shot learning (ZSL) and generalised zero-shot learning (GZSL) settings. Our results show that the Diffusion Classifier and ViLT perform well at concept binding tasks, but that all models struggle significantly with the relational GZSL task, underscoring the broader challenges VLMs face with relational reasoning. Analysis of CLIP embeddings suggests that the difficulty may stem from overly similar representations of relational concepts such as left and right. Code and dataset are available at: this https URL 

**Abstract (ZH)**: 自然语言语义的一个基本方面是从先前已知部分的组合中形成新的含义。视觉-语言模型（VLMs）在近年来取得了显著进展，然而，证据表明它们无法执行这种组合。扩散模型近年来因其出色的生成能力而受到了广泛关注，基于扩散模型的零样本分类器已被证明在某些组合任务中能与CLIP竞争。在此项工作中，我们探讨了生成扩散分类器是否在组合泛化能力方面优于判别模型。我们评估了三种模型——扩散分类器、CLIP和ViLT——在零样本学习（ZSL）和广义零样本学习（GZSL）设置中将物体与其属性和关系相结合的能力。我们的结果表明，扩散分类器和ViLT在概念绑定任务中表现良好，但所有模型在关系GZSL任务中都面临重大挑战，突显了VLMs在关系推理方面面临的更广泛挑战。CLIP嵌入分析表明，困难可能源于关系概念如左和右的表示过于相似。代码和数据集可在以下链接获取：this https URL。 

---
# Safer Skin Lesion Classification with Global Class Activation Probability Map Evaluation and SafeML 

**Title (ZH)**: 全球类激活概率图评估与SafeML的皮肤病变分类安全性 

**Authors**: Kuniko Paxton, Koorosh Aslansefat, Amila Akagić, Dhavalkumar Thakker, Yiannis Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2508.20776)  

**Abstract**: Recent advancements in skin lesion classification models have significantly improved accuracy, with some models even surpassing dermatologists' diagnostic performance. However, in medical practice, distrust in AI models remains a challenge. Beyond high accuracy, trustworthy, explainable diagnoses are essential. Existing explainability methods have reliability issues, with LIME-based methods suffering from inconsistency, while CAM-based methods failing to consider all classes. To address these limitations, we propose Global Class Activation Probabilistic Map Evaluation, a method that analyses all classes' activation probability maps probabilistically and at a pixel level. By visualizing the diagnostic process in a unified manner, it helps reduce the risk of misdiagnosis. Furthermore, the application of SafeML enhances the detection of false diagnoses and issues warnings to doctors and patients as needed, improving diagnostic reliability and ultimately patient safety. We evaluated our method using the ISIC datasets with MobileNetV2 and Vision Transformers. 

**Abstract (ZH)**: 近期皮肤病变分类模型的进展显著提高了准确性，某些模型甚至超越了皮肤科医生的诊断性能。然而，在医疗实践中，对AI模型的信任依然是一个挑战。除了高准确性，可信赖且可解释的诊断结果至关重要。现有的可解释性方法存在可靠性问题，基于LIME的方法表现出不一致性，而基于CAM的方法未能考虑所有类别。为解决这些限制，我们提出全局类别激活概率图评估方法，该方法在像素级和概率层面分析所有类别的激活概率图。通过统一可视化诊断过程，有助于降低误诊风险。此外，SafeML的应用可以检测假性诊断并根据需要向医生和患者发出警告，从而提高诊断可靠性，最终保障患者安全。我们使用ISIC数据集和MobileNetV2及Vision Transformers对我们的方法进行了评估。 

---
# Unleashing Uncertainty: Efficient Machine Unlearning for Generative AI 

**Title (ZH)**: 解锁不确定性：生成式AI的高效机器遗忘技术 

**Authors**: Christoforos N. Spartalis, Theodoros Semertzidis, Petros Daras, Efstratios Gavves  

**Link**: [PDF](https://arxiv.org/pdf/2508.20773)  

**Abstract**: We introduce SAFEMax, a novel method for Machine Unlearning in diffusion models. Grounded in information-theoretic principles, SAFEMax maximizes the entropy in generated images, causing the model to generate Gaussian noise when conditioned on impermissible classes by ultimately halting its denoising process. Also, our method controls the balance between forgetting and retention by selectively focusing on the early diffusion steps, where class-specific information is prominent. Our results demonstrate the effectiveness of SAFEMax and highlight its substantial efficiency gains over state-of-the-art methods. 

**Abstract (ZH)**: SAFEMax：一种基于信息论原则的扩散模型机器遗忘新方法 

---
# Signs of Struggle: Spotting Cognitive Distortions across Language and Register 

**Title (ZH)**: 挣扎的迹象：跨语言和体裁识别认知扭曲 

**Authors**: Abhishek Kuber, Enrico Liscio, Ruixuan Zhang, Caroline Figueroa, Pradeep K. Murukannaiah  

**Link**: [PDF](https://arxiv.org/pdf/2508.20771)  

**Abstract**: Rising mental health issues among youth have increased interest in automated approaches for detecting early signs of psychological distress in digital text. One key focus is the identification of cognitive distortions, irrational thought patterns that have a role in aggravating mental distress. Early detection of these distortions may enable timely, low-cost interventions. While prior work has focused on English clinical data, we present the first in-depth study of cross-lingual and cross-register generalization of cognitive distortion detection, analyzing forum posts written by Dutch adolescents. Our findings show that while changes in language and writing style can significantly affect model performance, domain adaptation methods show the most promise. 

**Abstract (ZH)**: 青少年心理健康问题上升激发了对自动检测数字文本早期心理压力迹象方法的兴趣。关键焦点在于认知 distortions 的识别，这些不合理的思维模式在加剧心理压力中起作用。早期检测这些 distorting 可能能实现及时、低成本的干预。虽然之前的工作主要集中在英语临床数据上，我们首次进行了跨语言和跨文体一般性的认知 distortions 检测深入研究，分析了荷兰青少年在论坛上撰写的文章。研究发现，尽管语言和写作风格的变化会显著影响模型性能，但领域适应方法显示出最大的潜力。 

---
# Turning the Spell Around: Lightweight Alignment Amplification via Rank-One Safety Injection 

**Title (ZH)**: 逆向转换：基于秩一安全性注入的轻量级对齐增强 

**Authors**: Harethah Abu Shairah, Hasan Abed Al Kader Hammoud, George Turkiyyah, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2508.20766)  

**Abstract**: Safety alignment in Large Language Models (LLMs) often involves mediating internal representations to refuse harmful requests. Recent research has demonstrated that these safety mechanisms can be bypassed by ablating or removing specific representational directions within the model. In this paper, we propose the opposite approach: Rank-One Safety Injection (ROSI), a white-box method that amplifies a model's safety alignment by permanently steering its activations toward the refusal-mediating subspace. ROSI operates as a simple, fine-tuning-free rank-one weight modification applied to all residual stream write matrices. The required safety direction can be computed from a small set of harmful and harmless instruction pairs. We show that ROSI consistently increases safety refusal rates - as evaluated by Llama Guard 3 - while preserving the utility of the model on standard benchmarks such as MMLU, HellaSwag, and Arc. Furthermore, we show that ROSI can also re-align 'uncensored' models by amplifying their own latent safety directions, demonstrating its utility as an effective last-mile safety procedure. Our results suggest that targeted, interpretable weight steering is a cheap and potent mechanism to improve LLM safety, complementing more resource-intensive fine-tuning paradigms. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的安全性对齐常常涉及调解内部表示以拒绝有害请求。最近的研究表明，可以通过消除或移除模型中的特定表示方向来绕过这些安全机制。在本文中，我们提出相反的方法：秩一安全性注入（ROSI），这是一种白盒方法，通过永久性地引导模型的激活朝向拒绝调解子空间来增强模型的安全对齐。ROSI 作为一种简单的、无需微调的一秩权重修改应用到所有残差流写入矩阵。所需的 safety 方向可以从少量有害和无害指令对中计算得出。我们展示了 ROSE 在不牺牲模型在标准基准（如 MMLU、HellaSwag 和 Arc）上的实用性的同时，一致地增加了安全性拒绝率，同时还展示了 ROSI 可以通过放大其自身潜在的安全方向来重新对齐“非审查”模型，证明了其作为有效的最后一英里安全程序的效用。我们的结果表明，有针对性、可解释的权重引导是一种低成本且强大的机制，可以改进 LLM 的安全性，补充更为资源密集型的微调范式。 

---
# Looking Beyond the Obvious: A Survey on Abstract Concept Recognition for Video Understanding 

**Title (ZH)**: 超越表面：视频理解中抽象概念识别综述 

**Authors**: Gowreesh Mago, Pascal Mettes, Stevan Rudinac  

**Link**: [PDF](https://arxiv.org/pdf/2508.20765)  

**Abstract**: The automatic understanding of video content is advancing rapidly. Empowered by deeper neural networks and large datasets, machines are increasingly capable of understanding what is concretely visible in video frames, whether it be objects, actions, events, or scenes. In comparison, humans retain a unique ability to also look beyond concrete entities and recognize abstract concepts like justice, freedom, and togetherness. Abstract concept recognition forms a crucial open challenge in video understanding, where reasoning on multiple semantic levels based on contextual information is key. In this paper, we argue that the recent advances in foundation models make for an ideal setting to address abstract concept understanding in videos. Automated understanding of high-level abstract concepts is imperative as it enables models to be more aligned with human reasoning and values. In this survey, we study different tasks and datasets used to understand abstract concepts in video content. We observe that, periodically and over a long period, researchers have attempted to solve these tasks, making the best use of the tools available at their disposal. We advocate that drawing on decades of community experience will help us shed light on this important open grand challenge and avoid ``re-inventing the wheel'' as we start revisiting it in the era of multi-modal foundation models. 

**Abstract (ZH)**: 视频内容中高级抽象概念的自动理解正在迅速发展 

---
# SKGE-SWIN: End-To-End Autonomous Vehicle Waypoint Prediction and Navigation Using Skip Stage Swin Transformer 

**Title (ZH)**: SKGE-SWIN：基于跳跃阶段Swin变压器的端到端自主车辆路径点预测与导航 

**Authors**: Fachri Najm Noer Kartiman, Rasim, Yaya Wihardi, Nurul Hasanah, Oskar Natan, Bambang Wahono, Taufik Ibnu Salim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20762)  

**Abstract**: Focusing on the development of an end-to-end autonomous vehicle model with pixel-to-pixel context awareness, this research proposes the SKGE-Swin architecture. This architecture utilizes the Swin Transformer with a skip-stage mechanism to broaden feature representation globally and at various network levels. This approach enables the model to extract information from distant pixels by leveraging the Swin Transformer's Shifted Window-based Multi-head Self-Attention (SW-MSA) mechanism and to retain critical information from the initial to the final stages of feature extraction, thereby enhancing its capability to comprehend complex patterns in the vehicle's surroundings. The model is evaluated on the CARLA platform using adversarial scenarios to simulate real-world conditions. Experimental results demonstrate that the SKGE-Swin architecture achieves a superior Driving Score compared to previous methods. Furthermore, an ablation study will be conducted to evaluate the contribution of each architectural component, including the influence of skip connections and the use of the Swin Transformer, in improving model performance. 

**Abstract (ZH)**: 基于端到端自主车辆模型的像素到像素上下文意识发展，本文提出SKGE-Swin架构。该架构利用具有跳层机制的Swin Transformer，以全局和多级网络方式扩展特征表示，通过Swin Transformer的Shifted Window基于多头自注意力（SW-MSA）机制从远处的像素中提取信息，并在特征提取的各个阶段保留关键信息，从而增强其理解和解析车辆周围复杂模式的能力。该模型在CARLA平台上使用对抗场景进行评估以模拟真实世界条件。实验结果表明，SKGE-Swin架构在驾驶得分上优于以往方法。此外，还将进行消融研究以评估每个架构组件的贡献，包括跳层连接和使用Swin Transformer对提高模型性能的影响。 

---
# Occlusion Robustness of CLIP for Military Vehicle Classification 

**Title (ZH)**: CLIP在军用车辆分类中的遮挡鲁棒性 

**Authors**: Jan Erik van Woerden, Gertjan Burghouts, Lotte Nijskens, Alma M. Liezenga, Sabina van Rooij, Frank Ruis, Hugo J. Kuijf  

**Link**: [PDF](https://arxiv.org/pdf/2508.20760)  

**Abstract**: Vision-language models (VLMs) like CLIP enable zero-shot classification by aligning images and text in a shared embedding space, offering advantages for defense applications with scarce labeled data. However, CLIP's robustness in challenging military environments, with partial occlusion and degraded signal-to-noise ratio (SNR), remains underexplored. We investigate CLIP variants' robustness to occlusion using a custom dataset of 18 military vehicle classes and evaluate using Normalized Area Under the Curve (NAUC) across occlusion percentages. Four key insights emerge: (1) Transformer-based CLIP models consistently outperform CNNs, (2) fine-grained, dispersed occlusions degrade performance more than larger contiguous occlusions, (3) despite improved accuracy, performance of linear-probed models sharply drops at around 35% occlusion, (4) by finetuning the model's backbone, this performance drop occurs at more than 60% occlusion. These results underscore the importance of occlusion-specific augmentations during training and the need for further exploration into patch-level sensitivity and architectural resilience for real-world deployment of CLIP. 

**Abstract (ZH)**: Vision-语言模型（VLMs）如CLIP通过在共享嵌入空间中对齐图像和文本实现零-shot分类，在稀缺标注数据的防御应用中具有优势。然而，CLIP在具有部分遮挡和降级信噪比（SNR）的挑战性军事环境中的鲁棒性仍需进一步探索。我们使用包含18类军事车辆的自定义数据集研究CLIP变体在遮挡下的鲁棒性，并通过不同遮挡百分比下的归一化面积下的曲线下的面积（NAUC）进行评估。四个关键见解浮出水面：（1）基于Transformer的CLIP模型始终优于CNN，（2）细微分散的遮挡比大面积连续遮挡对性能的影响更严重，（3）尽管准确率有所提高，线性探查模型在大约35%遮挡时性能急剧下降，（4）通过微调模型的骨干网络，这种性能下降发生在超过60%遮挡时。这些结果强调了在训练期间使用遮挡特定增强措施的重要性，并指出了需要进一步探索区域级敏感性和架构稳健性以实现CLIP在实际部署中的应用。 

---
# SeqVLM: Proposal-Guided Multi-View Sequences Reasoning via VLM for Zero-Shot 3D Visual Grounding 

**Title (ZH)**: SeqVLM：基于提案引导的多视图序列推理用于零-shot 3D视觉锚定 

**Authors**: Jiawen Lin, Shiran Bian, Yihang Zhu, Wenbin Tan, Yachao Zhang, Yuan Xie, Yanyun Qu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20758)  

**Abstract**: 3D Visual Grounding (3DVG) aims to localize objects in 3D scenes using natural language descriptions. Although supervised methods achieve higher accuracy in constrained settings, zero-shot 3DVG holds greater promise for real-world applications since eliminating scene-specific training requirements. However, existing zero-shot methods face challenges of spatial-limited reasoning due to reliance on single-view localization, and contextual omissions or detail degradation. To address these issues, we propose SeqVLM, a novel zero-shot 3DVG framework that leverages multi-view real-world scene images with spatial information for target object reasoning. Specifically, SeqVLM first generates 3D instance proposals via a 3D semantic segmentation network and refines them through semantic filtering, retaining only semantic-relevant candidates. A proposal-guided multi-view projection strategy then projects these candidate proposals onto real scene image sequences, preserving spatial relationships and contextual details in the conversion process of 3D point cloud to images. Furthermore, to mitigate VLM computational overload, we implement a dynamic scheduling mechanism that iteratively processes sequances-query prompts, leveraging VLM's cross-modal reasoning capabilities to identify textually specified objects. Experiments on the ScanRefer and Nr3D benchmarks demonstrate state-of-the-art performance, achieving Acc@0.25 scores of 55.6% and 53.2%, surpassing previous zero-shot methods by 4.0% and 5.2%, respectively, which advance 3DVG toward greater generalization and real-world applicability. The code is available at this https URL. 

**Abstract (ZH)**: 三维视觉定位（3D视觉定位）旨在使用自然语言描述在三维场景中定位物体。虽然监督方法在受限环境中实现了更高的准确性，零样本3D视觉定位由于消除了特定场景的训练要求，在实际应用中具有更大的潜力。然而，现有的零样本方法在依赖单视图定位时面临空间推理受限的问题，并且可能出现语境遗漏或细节降解。为了解决这些问题，我们提出了SeqVLM，这是一个新颖的零样本3D视觉定位框架，该框架利用多视图真实场景图像中的空间信息进行目标物体推理。具体而言，SeqVLM首先通过3D语义分割网络生成3D实例提案，并通过语义过滤对其进行 refinement，仅保留与语义相关的候选提案。然后，采用提案导向的多视图投影策略将这些候选提案投影到真实场景图像序列上，在3D点云到图像的转换过程中保持空间关系和语境细节。此外，为了缓解VLM的计算负担，我们实现了动态调度机制，该机制迭代处理序列-查询提示，利用VLM的跨模态推理能力识别文本指定的对象。在ScanRefer和Nr3D基准上的实验展示了最先进的性能， Acc@0.25得分为55.6%和53.2%，分别超越了之前的最佳零样本方法4.0%和5.2%，从而推动了3D视觉定位向更广泛的通用性和实际应用方向发展。代码可在以下链接获得：this https URL。 

---
# Provable Benefits of In-Tool Learning for Large Language Models 

**Title (ZH)**: 工具内置学习对大型语言模型的可验证益处 

**Authors**: Sam Houliston, Ambroise Odonnat, Charles Arnal, Vivien Cabannes  

**Link**: [PDF](https://arxiv.org/pdf/2508.20755)  

**Abstract**: Tool-augmented language models, equipped with retrieval, memory, or external APIs, are reshaping AI, yet their theoretical advantages remain underexplored. In this paper, we address this question by demonstrating the benefits of in-tool learning (external retrieval) over in-weight learning (memorization) for factual recall. We show that the number of facts a model can memorize solely in its weights is fundamentally limited by its parameter count. In contrast, we prove that tool-use enables unbounded factual recall via a simple and efficient circuit construction. These results are validated in controlled experiments, where tool-using models consistently outperform memorizing ones. We further show that for pretrained large language models, teaching tool-use and general rules is more effective than finetuning facts into memory. Our work provides both a theoretical and empirical foundation, establishing why tool-augmented workflows are not just practical, but provably more scalable. 

**Abstract (ZH)**: 工具增强的语言模型装备了检索、记忆或外部API，正在重塑AI，但其理论优势仍待深入探索。本文通过展示工具内学习（外部检索）相对于权重重学（记忆）在事实回忆方面的优势来回答这一问题。我们证明，模型仅在其权重内记忆事实的数量从根本上受其参数数量的限制。相比之下，我们证明工具使用能够通过简单的高效电路构造实现无限制的事实回忆。这些结果在受控实验中得到验证，工具使用模型始终优于记忆模型。进一步研究表明，对于预训练的大语言模型，教授工具使用和通用规则比将事实微调到记忆中更为有效。我们的工作提供了理论和实验证据，确立了工具增强的工作流不仅具有实用性，而且是可证明更具扩展性的。 

---
# ${C}^{3}$-GS: Learning Context-aware, Cross-dimension, Cross-scale Feature for Generalizable Gaussian Splatting 

**Title (ZH)**: ${C}^{3}$-GS: 学习上下文aware、跨维度、跨尺度特征以实现通用高斯散射 

**Authors**: Yuxi Hu, Jun Zhang, Kuangyi Chen, Zhe Zhang, Friedrich Fraundorfer  

**Link**: [PDF](https://arxiv.org/pdf/2508.20754)  

**Abstract**: Generalizable Gaussian Splatting aims to synthesize novel views for unseen scenes without per-scene optimization. In particular, recent advancements utilize feed-forward networks to predict per-pixel Gaussian parameters, enabling high-quality synthesis from sparse input views. However, existing approaches fall short in encoding discriminative, multi-view consistent features for Gaussian predictions, which struggle to construct accurate geometry with sparse views. To address this, we propose $\mathbf{C}^{3}$-GS, a framework that enhances feature learning by incorporating context-aware, cross-dimension, and cross-scale constraints. Our architecture integrates three lightweight modules into a unified rendering pipeline, improving feature fusion and enabling photorealistic synthesis without requiring additional supervision. Extensive experiments on benchmark datasets validate that $\mathbf{C}^{3}$-GS achieves state-of-the-art rendering quality and generalization ability. Code is available at: this https URL. 

**Abstract (ZH)**: $\mathbf{C}^{3}$-GS旨在通过结合上下文感知、跨维度和跨尺度约束来增强特征学习，以合成未见过场景的新型视图。我们的架构将三个轻量级模块整合到统一的渲染管道中，提高特征融合，无需额外监督即可实现照片级真实感合成。在基准数据集上的广泛实验验证了$\mathbf{C}^{3}$-GS 达到了最顶尖的渲染质量和泛化能力。代码可在以下链接获取：this https URL。 

---
# Rethinking Testing for LLM Applications: Characteristics, Challenges, and a Lightweight Interaction Protocol 

**Title (ZH)**: 重思大语言模型应用的测试：特性、挑战及一种轻量级交互协议 

**Authors**: Wei Ma, Yixiao Yang, Qiang Hu, Shi Ying, Zhi Jin, Bo Du, Zhenchang Xing, Tianlin Li, Junjie Shi, Yang Liu, Linxiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20737)  

**Abstract**: Applications of Large Language Models~(LLMs) have evolved from simple text generators into complex software systems that integrate retrieval augmentation, tool invocation, and multi-turn interactions. Their inherent non-determinism, dynamism, and context dependence pose fundamental challenges for quality assurance. This paper decomposes LLM applications into a three-layer architecture: \textbf{\textit{System Shell Layer}}, \textbf{\textit{Prompt Orchestration Layer}}, and \textbf{\textit{LLM Inference Core}}. We then assess the applicability of traditional software testing methods in each layer: directly applicable at the shell layer, requiring semantic reinterpretation at the orchestration layer, and necessitating paradigm shifts at the inference core. A comparative analysis of Testing AI methods from the software engineering community and safety analysis techniques from the AI community reveals structural disconnects in testing unit abstraction, evaluation metrics, and lifecycle management. We identify four fundamental differences that underlie 6 core challenges. To address these, we propose four types of collaborative strategies (\emph{Retain}, \emph{Translate}, \emph{Integrate}, and \emph{Runtime}) and explore a closed-loop, trustworthy quality assurance framework that combines pre-deployment validation with runtime monitoring. Based on these strategies, we offer practical guidance and a protocol proposal to support the standardization and tooling of LLM application testing. We propose a protocol \textbf{\textit{Agent Interaction Communication Language}} (AICL) that is used to communicate between AI agents. AICL has the test-oriented features and is easily integrated in the current agent framework. 

**Abstract (ZH)**: 大型语言模型应用中的质量保证：三层架构与协作策略 

---
# EEGDM: Learning EEG Representation with Latent Diffusion Model 

**Title (ZH)**: EEGDM：基于潜在扩散模型的EEG表示学习 

**Authors**: Shaocong Wang, Tong Liu, Ming Li, Minjing Yu, Yong-Jin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20705)  

**Abstract**: While electroencephalography (EEG) signal analysis using deep learning has shown great promise, existing approaches still face significant challenges in learning generalizable representations that perform well across diverse tasks, particularly when training data is limited. Current EEG representation learning methods including EEGPT and LaBraM typically rely on simple masked reconstruction objective, which may not fully capture the rich semantic information and complex patterns inherent in EEG signals. In this paper, we propose EEGDM, a novel self-supervised EEG representation learning method based on the latent diffusion model, which leverages EEG signal generation as a self-supervised objective, turning the diffusion model into a strong representation learner capable of capturing EEG semantics. EEGDM incorporates an EEG encoder that distills EEG signals and their channel augmentations into a compact representation, acting as conditional information to guide the diffusion model for generating EEG signals. This design endows EEGDM with a compact latent space, which not only offers ample control over the generative process but also can be leveraged for downstream tasks. Experimental results show that EEGDM (1) can reconstruct high-quality EEG signals, (2) effectively learns robust representations, and (3) achieves competitive performance with modest pre-training data size across diverse downstream tasks, underscoring its generalizability and practical utility. 

**Abstract (ZH)**: 基于潜扩散模型的自监督EEG表示学习方法EEGDMSelf-supervised EEG Representation Learning Method Based on Latent Diffusion Model: EEGDM 

---
# Generative Annotation for ASR Named Entity Correction 

**Title (ZH)**: ASR命名实体识别的生成性标注方法 

**Authors**: Yuanchang Luo, Daimeng Wei, Shaojun Li, Hengchao Shang, Jiaxin Guo, Zongyao Li, Zhanglin Wu, Xiaoyu Chen, Zhiqiang Rao, Jinlong Yang, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20700)  

**Abstract**: End-to-end automatic speech recognition systems often fail to transcribe domain-specific named entities, causing catastrophic failures in downstream tasks. Numerous fast and lightweight named entity correction (NEC) models have been proposed in recent years. These models, mainly leveraging phonetic-level edit distance algorithms, have shown impressive performances. However, when the forms of the wrongly-transcribed words(s) and the ground-truth entity are significantly different, these methods often fail to locate the wrongly transcribed words in hypothesis, thus limiting their usage. We propose a novel NEC method that utilizes speech sound features to retrieve candidate entities. With speech sound features and candidate entities, we inovatively design a generative method to annotate entity errors in ASR transcripts and replace the text with correct entities. This method is effective in scenarios of word form difference. We test our method using open-source and self-constructed test sets. The results demonstrate that our NEC method can bring significant improvement to entity accuracy. We will open source our self-constructed test set and training data. 

**Abstract (ZH)**: 端到端自动语音识别系统常在转录领域特异性命名实体时失败，导致下游任务出现灾难性失败。近年来，提出了许多快速且轻量级的命名实体修正模型。这些模型主要利用音素级编辑距离算法，表现出色。然而，当错误转录词的形式与真实实体形式差异显著时，这些方法往往无法在假设中定位错误转录的词，从而限制了其应用。我们提出了一种新颖的命名实体修正方法，利用语音音素特征来检索候选实体。借助语音音素特征和候选实体，我们创新设计了一种生成方法，用于标注ASR转录中的实体错误，并用正确实体替换文本。该方法在词形差异场景中有效。我们使用开源和自行构建的测试集对方法进行了测试。结果表明，我们的命名实体修正方法能显著提高实体准确性。我们将开放我们自行构建的测试集和训练数据。 

---
# MobileCLIP2: Improving Multi-Modal Reinforced Training 

**Title (ZH)**: MobileCLIP2：提高多模态强化训练性能 

**Authors**: Fartash Faghri, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Alexander Toshev, Oncel Tuzel, Hadi Pouransari  

**Link**: [PDF](https://arxiv.org/pdf/2508.20691)  

**Abstract**: Foundation image-text models such as CLIP with zero-shot capabilities enable a wide array of applications. MobileCLIP is a recent family of image-text models at 3-15ms latency and 50-150M parameters with state-of-the-art zero-shot accuracy. The main ingredients in MobileCLIP were its low-latency and light architectures and a novel multi-modal reinforced training that made knowledge distillation from multiple caption-generators and CLIP teachers efficient, scalable, and reproducible. In this paper, we improve the multi-modal reinforced training of MobileCLIP through: 1) better CLIP teacher ensembles trained on the DFN dataset, 2) improved captioner teachers trained on the DFN dataset and fine-tuned on a diverse selection of high-quality image-caption datasets. We discover new insights through ablations such as the importance of temperature tuning in contrastive knowledge distillation, the effectiveness of caption-generator fine-tuning for caption diversity, and the additive improvement from combining synthetic captions generated by multiple models. We train a new family of models called MobileCLIP2 and achieve state-of-the-art ImageNet-1k zero-shot accuracies at low latencies. In particular, we observe 2.2% improvement in ImageNet-1k accuracy for MobileCLIP2-B compared with MobileCLIP-B architecture. Notably, MobileCLIP2-S4 matches the zero-shot accuracy of SigLIP-SO400M/14 on ImageNet-1k while being 2$\times$ smaller and improves on DFN ViT-L/14 at 2.5$\times$ lower latency. We release our pretrained models (this https URL) and the data generation code (this https URL). The data generation code makes it easy to create new reinforced datasets with arbitrary teachers using distributed scalable processing. 

**Abstract (ZH)**: 基于CLIP等零样本能力的基础图像-文本模型 enables 大量应用：MobileCLIP通过3-15ms延迟和50-150M参数实现最先进的零样本准确性。通过改进MobileCLIP的多模态强化训练，我们发现新的见解并训练出MobileCLIP2，实现了低延迟下的ImageNet-1k零样本精度新记录。 

---
# Task Allocation for Autonomous Machines using Computational Intelligence and Deep Reinforcement Learning 

**Title (ZH)**: 使用计算智能和深度强化学习的自主机器任务分配 

**Authors**: Thanh Thi Nguyen, Quoc Viet Hung Nguyen, Jonathan Kua, Imran Razzak, Dung Nguyen, Saeid Nahavandi  

**Link**: [PDF](https://arxiv.org/pdf/2508.20688)  

**Abstract**: Enabling multiple autonomous machines to perform reliably requires the development of efficient cooperative control algorithms. This paper presents a survey of algorithms that have been developed for controlling and coordinating autonomous machines in complex environments. We especially focus on task allocation methods using computational intelligence (CI) and deep reinforcement learning (RL). The advantages and disadvantages of the surveyed methods are analysed thoroughly. We also propose and discuss in detail various future research directions that shed light on how to improve existing algorithms or create new methods to enhance the employability and performance of autonomous machines in real-world applications. The findings indicate that CI and deep RL methods provide viable approaches to addressing complex task allocation problems in dynamic and uncertain environments. The recent development of deep RL has greatly contributed to the literature on controlling and coordinating autonomous machines, and it has become a growing trend in this area. It is envisaged that this paper will provide researchers and engineers with a comprehensive overview of progress in machine learning research related to autonomous machines. It also highlights underexplored areas, identifies emerging methodologies, and suggests new avenues for exploration in future research within this domain. 

**Abstract (ZH)**: 高效协同控制算法的开发对于多个自主机器可靠地执行任务至关重要。本文综述了用于在复杂环境中控制和协调自主机器的算法，特别关注使用计算智能（CI）和深度强化学习（DRL）的任务分配方法。分析了所调研方法的优缺点，并提出了改进现有算法或创造新方法以提高自主机器在实际应用中适用性和性能的未来研究方向。研究发现，CI和深度DRL方法为解决动态和不确定环境中复杂任务分配问题提供了可行的途径。近年来深度DRL的发展极大地丰富了控制和协调自主机器的研究文献，并已成为该领域的研究趋势。本文旨在为研究人员和工程师提供机器学习研究进展的全面概述，同时强调未被充分开发的领域，识别新兴方法，并为未来研究提供新的探索方向。 

---
# Amadeus: Autoregressive Model with Bidirectional Attribute Modelling for Symbolic Music 

**Title (ZH)**: Amadeus: 自回归模型结合双向属性建模的符号音乐生成 

**Authors**: Hongju Su, Ke Li, Lan Yang, Honggang Zhang, Yi-Zhe Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.20665)  

**Abstract**: Existing state-of-the-art symbolic music generation models predominantly adopt autoregressive or hierarchical autoregressive architectures, modelling symbolic music as a sequence of attribute tokens with unidirectional temporal dependencies, under the assumption of a fixed, strict dependency structure among these attributes. However, we observe that using different attributes as the initial token in these models leads to comparable performance. This suggests that the attributes of a musical note are, in essence, a concurrent and unordered set, rather than a temporally dependent sequence. Based on this insight, we introduce Amadeus, a novel symbolic music generation framework. Amadeus adopts a two-level architecture: an autoregressive model for note sequences and a bidirectional discrete diffusion model for attributes. To enhance performance, we propose Music Latent Space Discriminability Enhancement Strategy(MLSDES), incorporating contrastive learning constraints that amplify discriminability of intermediate music representations. The Conditional Information Enhancement Module (CIEM) simultaneously strengthens note latent vector representation via attention mechanisms, enabling more precise note decoding. We conduct extensive experiments on unconditional and text-conditioned generation tasks. Amadeus significantly outperforms SOTA models across multiple metrics while achieving at least 4$\times$ speed-up. Furthermore, we demonstrate training-free, fine-grained note attribute control feasibility using our model. To explore the upper performance bound of the Amadeus architecture, we compile the largest open-source symbolic music dataset to date, AMD (Amadeus MIDI Dataset), supporting both pre-training and fine-tuning. 

**Abstract (ZH)**: 现有的最先进符号音乐生成模型大多采用自回归或分层自回归架构，将符号音乐建模为具有单向时间依赖性的属性 token 序列，假设这些属性之间存在固定且严格的时间依赖关系。然而，我们观察到，在这些模型中使用不同的属性作为初始 token 并不会导致性能显著差异，这表明音乐符号的属性实际上是一个并发且无序的集合，而非具有时间依赖性的序列。基于这一洞察，我们引入了Amadeus，一种新颖的符号音乐生成框架。Amadeus采用两层架构：对接奏序列的自回归模型和对属性的双向离散扩散模型。为提升性能，我们提出了Music Latent Space Discriminability Enhancement Strategy(MLSDES)，将对比学习约束纳入模型以增强中间音乐表征的可区分性。条件信息增强模块(CIEM)通过注意机制同时增强了对奏符号 latent 向量的表示，从而能够更精确地解码对奏符号。我们在无条件和文本条件生成任务上进行了大量实验证明，Amadeus在多个指标上大幅优于当前最佳模型，且至少快 4 倍。此外，我们展示了使用该模型实现无训练的细粒度对奏属性控制的可能性。为了探索Amadeus架构的性能上限，我们构建了迄今为止最大的开源符号音乐数据集AMD（Amadeus MIDI Dataset），支持预训练和微调。 

---
# Task-Oriented Edge-Assisted Cross-System Design for Real-Time Human-Robot Interaction in Industrial Metaverse 

**Title (ZH)**: 面向任务的边缘辅助跨系统设计：工业元宇宙中的实时人机器人交互 

**Authors**: Kan Chen, Zhen Meng, Xiangmin Xu, Jiaming Yang, Emma Li, Philip G. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20664)  

**Abstract**: Real-time human-device interaction in industrial Metaverse faces challenges such as high computational load, limited bandwidth, and strict latency. This paper proposes a task-oriented edge-assisted cross-system framework using digital twins (DTs) to enable responsive interactions. By predicting operator motions, the system supports: 1) proactive Metaverse rendering for visual feedback, and 2) preemptive control of remote devices. The DTs are decoupled into two virtual functions-visual display and robotic control-optimizing both performance and adaptability. To enhance generalizability, we introduce the Human-In-The-Loop Model-Agnostic Meta-Learning (HITL-MAML) algorithm, which dynamically adjusts prediction horizons. Evaluation on two tasks demonstrates the framework's effectiveness: in a Trajectory-Based Drawing Control task, it reduces weighted RMSE from 0.0712 m to 0.0101 m; in a real-time 3D scene representation task for nuclear decommissioning, it achieves a PSNR of 22.11, SSIM of 0.8729, and LPIPS of 0.1298. These results show the framework's capability to ensure spatial precision and visual fidelity in real-time, high-risk industrial environments. 

**Abstract (ZH)**: 基于数字孪生的任务导向边缘辅助跨系统框架在工业元宇宙中的实时人机交互 

---
# GDS Agent: A Graph Algorithmic Reasoning Agent 

**Title (ZH)**: GDS代理：图算法推理代理 

**Authors**: Borun Shi, Ioannis Panagiotas  

**Link**: [PDF](https://arxiv.org/pdf/2508.20637)  

**Abstract**: Large language models (LLMs) have shown remarkable multimodal information processing and reasoning ability. When equipped with tools through function calling and enhanced with retrieval-augmented techniques, compound LLM-based systems can access closed data sources and answer questions about them. However, they still struggle to process and reason over large-scale graph-structure data. We introduce the GDS (Graph Data Science) agent in this technical report. The GDS agent introduces a comprehensive set of graph algorithms as tools, together with preprocessing (retrieval) and postprocessing of algorithm results, in a model context protocol (MCP) server. The server can be used with any modern LLM out-of-the-box. GDS agent allows users to ask any question that implicitly and intrinsically requires graph algorithmic reasoning about their data, and quickly obtain accurate and grounded answers. We also introduce a new benchmark that evaluates intermediate tool calls as well as final responses. The results indicate that GDS agent is able to solve a wide spectrum of graph tasks. We also provide detailed case studies for more open-ended tasks and study scenarios where the agent struggles. Finally, we discuss the remaining challenges and the future roadmap. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多模态信息处理和推理方面表现出色。通过功能调用配备工具并结合检索增强技术，复合LLM基系统可以访问封闭数据源并回答相关问题。然而，它们仍然难以处理和推理大规模图结构数据。在本技术报告中，我们介绍了GDS（图数据科学）代理。GDS代理提供了一套全面的图算法作为工具，并在模型上下文协议（MCP）服务器中包含算法结果的预处理（检索）和后处理。该服务器可以与任何现代LLM无缝配合使用。GDS代理允许用户提出任何隐含和内在需要图算法推理的数据问题，并迅速获得准确且基于事实的回答。我们还引入了一个新的基准测试，用于评估中间工具调用以及最终响应。结果表明，GDS代理能够解决广泛的图任务。我们还提供了更开放任务的详细案例研究，并探讨了代理在某些场景中的困境。最后，我们讨论了剩余的挑战和未来 roadmap。 

---
# ArtFace: Towards Historical Portrait Face Identification via Model Adaptation 

**Title (ZH)**: ArtFace: 基于模型适应的历史人物肖像面部识别 

**Authors**: Francois Poh, Anjith George, Sébastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2508.20626)  

**Abstract**: Identifying sitters in historical paintings is a key task for art historians, offering insight into their lives and how they chose to be seen. However, the process is often subjective and limited by the lack of data and stylistic variations. Automated facial recognition is capable of handling challenging conditions and can assist, but while traditional facial recognition models perform well on photographs, they struggle with paintings due to domain shift and high intra-class variation. Artistic factors such as style, skill, intent, and influence from other works further complicate recognition. In this work, we investigate the potential of foundation models to improve facial recognition in artworks. By fine-tuning foundation models and integrating their embeddings with those from conventional facial recognition networks, we demonstrate notable improvements over current state-of-the-art methods. Our results show that foundation models can bridge the gap where traditional methods are ineffective. Paper page at this https URL 

**Abstract (ZH)**: 历史绘画中人物身份识别是艺术史学家的一项关键任务，有助于了解人物的生活及其选择呈现的方式。然而，这一过程往往是主观的，并受到数据不足和风格差异的限制。自动化面部识别能够处理具有挑战性的条件，并提供辅助，但传统面部识别模型在照片上表现良好，但在绘画上则因领域转移和高内类变异性而挣扎。艺术因素如风格、技巧、意图及受其他作品影响进一步加剧了识别难度。在本文中，我们研究了基础模型在提高艺术品面部识别方面潜力。通过微调基础模型并将它们的嵌入与传统面部识别网络的嵌入整合，我们展示了在现有顶级方法上的显著改进。我们的结果表明，基础模型能够弥补传统方法无效的空白。论文页面详见此链接：https://this.is/url。 

---
# Flowing Straighter with Conditional Flow Matching for Accurate Speech Enhancement 

**Title (ZH)**: 基于条件流匹配的精确语音增强 

**Authors**: Mattias Cross, Anton Ragni  

**Link**: [PDF](https://arxiv.org/pdf/2508.20584)  

**Abstract**: Current flow-based generative speech enhancement methods learn curved probability paths which model a mapping between clean and noisy speech. Despite impressive performance, the implications of curved probability paths are unknown. Methods such as Schrodinger bridges focus on curved paths, where time-dependent gradients and variance do not promote straight paths. Findings in machine learning research suggest that straight paths, such as conditional flow matching, are easier to train and offer better generalisation. In this paper we quantify the effect of path straightness on speech enhancement quality. We report experiments with the Schrodinger bridge, where we show that certain configurations lead to straighter paths. Conversely, we propose independent conditional flow-matching for speech enhancement, which models straight paths between noisy and clean speech. We demonstrate empirically that a time-independent variance has a greater effect on sample quality than the gradient. Although conditional flow matching improves several speech quality metrics, it requires multiple inference steps. We rectify this with a one-step solution by inferring the trained flow-based model as if it was directly predictive. Our work suggests that straighter time-independent probability paths improve generative speech enhancement over curved time-dependent paths. 

**Abstract (ZH)**: 基于流的方法在噪声抑制中的生成性增强学习到曲率概率路径，这些路径建模了干净语音和噪声语音之间的映射。尽管表现出色，但曲率概率路径的影响尚不知晓。Schrödinger桥梁等方法专注于曲率路径，其中时间相关的梯度和方差不促进直线路径。机器学习研究中的发现表明，如条件流匹配这样的直线路径更容易训练并提供更好的泛化能力。在本文中，我们量化了路径直线性对语音增强质量的影响。我们使用Schrödinger桥梁进行实验，表明某些配置导致更直线的路径。相反，我们提出了独立条件流匹配方法用于语音增强，该方法建模了噪声语音和干净语音之间的直线路径。实验结果显示，时间无关的方差对样本质量的影响大于梯度。尽管条件流匹配提高了多种语音质量指标，但仍需要多步推理。我们通过将训练中的流模型直接预测来实现一步解决方案。我们的工作表明，时间无关的更直线的概率路径在生成性语音增强中优于时间相关的曲率路径。 

---
# A Graph Talks, But Who's Listening? Rethinking Evaluations for Graph-Language Models 

**Title (ZH)**: 图自证，谁在倾听？重新思考图语言模型的评估标准。 

**Authors**: Soham Petkar, Hari Aakash K, Anirudh Vempati, Akshit Sinha, Ponnurangam Kumarauguru, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2508.20583)  

**Abstract**: Developments in Graph-Language Models (GLMs) aim to integrate the structural reasoning capabilities of Graph Neural Networks (GNNs) with the semantic understanding of Large Language Models (LLMs). However, we demonstrate that current evaluation benchmarks for GLMs, which are primarily repurposed node-level classification datasets, are insufficient to assess multimodal reasoning. Our analysis reveals that strong performance on these benchmarks is achievable using unimodal information alone, suggesting that they do not necessitate graph-language integration. To address this evaluation gap, we introduce the CLEGR(Compositional Language-Graph Reasoning) benchmark, designed to evaluate multimodal reasoning at various complexity levels. Our benchmark employs a synthetic graph generation pipeline paired with questions that require joint reasoning over structure and textual semantics. We perform a thorough evaluation of representative GLM architectures and find that soft-prompted LLM baselines perform on par with GLMs that incorporate a full GNN backbone. This result calls into question the architectural necessity of incorporating graph structure into LLMs. We further show that GLMs exhibit significant performance degradation in tasks that require structural reasoning. These findings highlight limitations in the graph reasoning capabilities of current GLMs and provide a foundation for advancing the community toward explicit multimodal reasoning involving graph structure and language. 

**Abstract (ZH)**: 图语言模型（GLMs）的发展旨在将图神经网络（GNNs）的结构性推理能力与大型语言模型（LLMs）的语义理解能力结合起来。然而，我们证明当前用于评估GLMs的主要基准（主要是复用的节点级分类数据集）不足以评估多模态推理。我们的分析表明，仅通过单模态信息即可在这些基准上实现强大的性能，这暗示这些基准并不需要图语言集成。为解决这一评估缺口，我们引入了CLEGR（组合语言-图推理）基准，旨在评估不同复杂度水平的多模态推理。该基准结合了合成图形生成管道和需要在结构和文本语义上联合推理的问题。我们对代表性GLM架构进行了全面评估，并发现软提示的大语言模型基线与包含完整GNN骨干的GLMs性能相当。这一结果质疑将图结构集成到大语言模型中的必要性。我们进一步表明，在需要结构性推理的任务中，GLMs表现出显著的性能下降。这些发现突显了当前GLMs在图推理能力上的局限性，并为促进涉及图结构和语言的明确多模态推理奠定了基础。 

---
# MERIT: Maximum-normalized Element-wise Ratio for Language Model Large-batch Training 

**Title (ZH)**: MERIT: 最大归一化元素比对语言模型大规模批次训练 

**Authors**: Yang Luo, Zangwei Zheng, Ziheng Qin, Zirui Zhu, Yong Liu, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2508.20577)  

**Abstract**: Large-batch training has become a cornerstone in accelerating the training of deep neural networks, yet it poses challenges in optimization and generalization. Existing optimizers like AdamW present performance degradation during language models' large-batch training, due to the information bottleneck in attention layers caused by the sharp increase of max attention logit. While the LAMB optimizer partially addresses this issue, some attention layers still face this issue. The reason is that $l_2$-norm-based trust ratios in LAMB are less effective in directly influencing the max value of query/key weights. Furthermore, the weight-wise trust ratio in LAMB is error-prone as it overlooks relationships of weight values within rows or columns. Building on these observations, we propose a novel optimizer, MERIT, which leverages the max-norm to calculate the trust ratio to constrain the max attention logit more effectively. Moreover, we further construct element-wise trust ratios to provide more robust update scaling by focusing on local weight structures. Extensive experiments of large-batch training across various sizes of GPT-2 models demonstrate the superior performance of MERIT. Notably, during the training of GPT-2 Medium, MERIT enables a 6k batch size without any performance degradation compared to the standard batch size (480) with 48B training tokens. This work highlights the importance of considering the max attention logit and finer-granularity trust ratio in large-batch training. It successfully improves the training stability and paves the way for larger batch usage, enabling faster development and iteration of large language models. Code is available at this https URL. 

**Abstract (ZH)**: 大规模批次训练已成为加速深度神经网络训练的基石，但同时也带来了优化和泛化的挑战。现有优化器如AdamW在语言模型的大批次训练中表现下降，原因是注意力层中的最大注意力对数导致的信息瓶颈。虽然LAMB优化器部分解决了这一问题，但仍有一些注意力层面临此问题。原因是LAMB中基于$L_2$范数的信任比间接影响查询/键权重的最大值效果不佳。此外，LAMB中的元素级信任比容易出错，因为它忽略了行内或列内权重值之间的关系。基于这些观察，我们提出了一种新的优化器MERIT，它利用最大范数计算信任比以更有效地约束最大注意力对数。此外，我们还构建了元素级信任比，通过关注局部权重结构提供更稳健的更新缩放。针对不同规模的GPT-2模型的大批次训练实验表明，MERIT表现出更优的性能。特别是在GPT-2 Medium的训练中，MERIT使得使用6k批次大小而不影响性能，相较于标准批次大小（480）和480亿训练令牌。本工作强调了在大批次训练中考虑最大注意力对数和更精细粒度的信任比的重要性，成功提高了训练稳定性，并为更大批次的使用铺平了道路，促进了大语言模型的更快开发和迭代。代码可在以下链接获取。 

---
# Towards Mechanistic Defenses Against Typographic Attacks in CLIP 

**Title (ZH)**: 面向CLIPagainst语义攻击的机理防御方法探究 

**Authors**: Lorenz Hufe, Constantin Venhoff, Maximilian Dreyer, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2508.20570)  

**Abstract**: Typographic attacks exploit multi-modal systems by injecting text into images, leading to targeted misclassifications, malicious content generation and even Vision-Language Model jailbreaks. In this work, we analyze how CLIP vision encoders behave under typographic attacks, locating specialized attention heads in the latter half of the model's layers that causally extract and transmit typographic information to the cls token. Building on these insights, we introduce a method to defend CLIP models against typographic attacks by selectively ablating a typographic circuit, consisting of attention heads. Without requiring finetuning, our method improves performance by up to 19.6% on a typographic variant of ImageNet-100, while reducing standard ImageNet-100 accuracy by less than 1%. Notably, our training-free approach remains competitive with current state-of-the-art typographic defenses that rely on finetuning. To this end, we release a family of dyslexic CLIP models which are significantly more robust against typographic attacks. These models serve as suitable drop-in replacements for a broad range of safety-critical applications, where the risks of text-based manipulation outweigh the utility of text recognition. 

**Abstract (ZH)**: typographic 攻击通过向图像注入文本来利用多模态系统，导致目标错误分类、恶意内容生成，甚至视觉语言模型的逃逸。在本文中，我们分析了 CLIP 视觉编码器在 typographic 攻击下的行为，定位到模型后半部分层中的专门注意头，这些头因果性地提取并传递 typographic 信息至 cls 标记。基于这些洞见，我们提出了一种方法，通过选择性地消除 typographic 循环（由注意头组成）来防御 CLIP 模型免受 typographic 攻击，无需微调，该方法在 typographic 变体的 ImageNet-100 数据集上性能提升高达 19.6%，同时将标准 ImageNet-100 的准确性降低不到 1%。值得注意的是，我们的无需训练的方法与目前依赖微调的状态最先进 typographic 防御方法具有竞争力。为此，我们发布了家族系列的 dyslexic CLIP 模型，这些模型在很大程度上抵御 typographic 攻击。这些模型适合作为广泛的安全关键应用的合适即插即用替代品，特别是在文本操纵的风险超过文本识别的实用性时。 

---
# AI and Agile Software Development: A Research Roadmap from the XP2025 Workshop 

**Title (ZH)**: AI和敏捷软件开发：来自XP2025研讨会的研究路线图 

**Authors**: Zheying Zhang, Tomas Herda, Victoria Pichler, Pekka Abrahamsson, Geir K. Hanssen, Joshua Kerievsky, Alex Polyakov, Mohit Chandna, Marius Irgens, Kai-Kristian Kemell, Ayman Asad Khan, Crystal Kwok, Evan Leybourn, Munish Malik, Dorota Mleczko, Morteza Moalagh, Christopher Morales, Yuliia Pieskova, Daniel Planötscher, Mika Saari, Anastasiia Tkalich, Karl Josef Gstettner, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20563)  

**Abstract**: This paper synthesizes the key findings from a full-day XP2025 workshop on "AI and Agile: From Frustration to Success", held in Brugg-Windisch, Switzerland. The workshop brought together over 30 interdisciplinary academic researchers and industry practitioners to tackle the concrete challenges and emerging opportunities at the intersection of Generative Artificial Intelligence (GenAI) and agile software development. Through structured, interactive breakout sessions, participants identified shared pain points like tool fragmentation, governance, data quality, and critical skills gaps in AI literacy and prompt engineering. These issues were further analyzed, revealing underlying causes and cross-cutting concerns. The workshop concluded by collaboratively co-creating a multi-thematic research roadmap, articulating both short-term, implementable actions and visionary, long-term research directions. This cohesive agenda aims to guide future investigation and drive the responsible, human-centered integration of GenAI into agile practices. 

**Abstract (ZH)**: 本文综合了在瑞士布吕格-温迪施举行的一整天XP2025研讨会“AI与敏捷：从挫折到成功”的关键发现。该研讨会汇聚了超过30名跨学科的学术研究人员和行业实践者，共同探讨生成式人工智能（GenAI）与敏捷软件开发交汇处的具体挑战和新兴机遇。通过结构化的互动分组讨论，参与者识别出工具碎片化、治理、数据质量以及关键的人工智能素养和提示工程技能缺口等共同痛点。这些问题进一步被分析，揭示了其背后的成因及跨领域的关注点。研讨会最终通过协作共同创建了一个多主题的研究路线图，既包括短期可实施的行动方案，也包括具有前瞻性的长期研究方向。这一统一的议程旨在引导未来的研究，并推动生成式人工智能负责任地融入敏捷实践之中。 

---
# Adaptive Federated Distillation for Multi-Domain Non-IID Textual Data 

**Title (ZH)**: 自适应联邦蒸馏多域非IID文本数据 

**Authors**: Jiahao Xiao, Jiangming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20557)  

**Abstract**: The widespread success of pre-trained language models has established a new training paradigm, where a global PLM is fine-tuned using task-specific data from local clients. The local data are highly different from each other and can not capture the global distribution of the whole data in real world. To address the challenges of non-IID data in real environments, privacy-preserving federated distillation has been proposed and highly investigated. However, previous experimental non-IID scenarios are primarily identified with the label (output) diversity, without considering the diversity of language domains (input) that is crucial in natural language processing. In this paper, we introduce a comprehensive set of multi-domain non-IID scenarios and propose a unified benchmarking framework that includes diverse data. The benchmark can be used to evaluate the federated learning framework in a real environment. To this end, we propose an Adaptive Federated Distillation (AdaFD) framework designed to address multi-domain non-IID challenges in both homogeneous and heterogeneous settings. Experimental results demonstrate that our models capture the diversity of local clients and achieve better performance compared to the existing works. The code for this paper is available at: this https URL. 

**Abstract (ZH)**: 预训练语言模型的广泛应用确立了一种新的训练范式，其中全局预训练模型使用来自本地客户端的任务特定数据进行 fine-tune。本地数据彼此高度不同，无法捕获整个真实世界数据的全局分布。为了应对真实环境中非-IID数据的挑战，隐私保护联邦蒸馏已被提出并进行了深入研究。然而，之前实验中的非-IID场景主要关注标签（输出）多样性，未考虑对自然语言处理至关重要的语言领域多样性。本文引入了一组全面的多领域非-IID场景，并提出了一体化的基准框架，其中包括多样化的数据。该基准可用于评估真实环境中的联邦学习框架。为此，我们提出了一种针对同质性和异质性环境中多领域非-IID挑战的自适应联邦蒸馏（AdaFD）框架。实验结果表明，我们的模型捕捉了本地客户端的多样性，并且在性能上优于现有工作。本文的相关代码可访问：this https URL。 

---
# Overview of BioASQ 2025: The Thirteenth BioASQ Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering 

**Title (ZH)**: BioASQ 2025：第十三届生物医学语义索引和问答挑战赛概述 

**Authors**: Anastasios Nentidis, Georgios Katsimpras, Anastasia Krithara, Martin Krallinger, Miguel Rodríguez-Ortega, Eduard Rodriguez-López, Natalia Loukachevitch, Andrey Sakhovskiy, Elena Tutubalina, Dimitris Dimitriadis, Grigorios Tsoumakas, George Giannakoulas, Alexandra Bekiaridou, Athanasios Samaras, Giorgio Maria Di Nunzio, Nicola Ferro, Stefano Marchesin, Marco Martinelli, Gianmaria Silvello, Georgios Paliouras  

**Link**: [PDF](https://arxiv.org/pdf/2508.20554)  

**Abstract**: This is an overview of the thirteenth edition of the BioASQ challenge in the context of the Conference and Labs of the Evaluation Forum (CLEF) 2025. BioASQ is a series of international challenges promoting advances in large-scale biomedical semantic indexing and question answering. This year, BioASQ consisted of new editions of the two established tasks, b and Synergy, and four new tasks: a) Task MultiClinSum on multilingual clinical summarization. b) Task BioNNE-L on nested named entity linking in Russian and English. c) Task ELCardioCC on clinical coding in cardiology. d) Task GutBrainIE on gut-brain interplay information extraction. In this edition of BioASQ, 83 competing teams participated with more than 1000 distinct submissions in total for the six different shared tasks of the challenge. Similar to previous editions, several participating systems achieved competitive performance, indicating the continuous advancement of the state-of-the-art in the field. 

**Abstract (ZH)**: 这是CLEF 2025会议和评价论坛实验室第十三届BioASQ挑战的概览。BioASQ是一系列促进大规模生物医学语义索引和问答技术发展的国际挑战。今年，BioASQ包括两个既定任务b和Synergy的新版，以及四个新任务：a) 多语言临床总结任务MultiClinSum。b) 俄语和英语中嵌套命名实体链接任务BioNNE-L。c) 心脏病临床编码任务ELCardioCC。d) 肠-脑交互信息提取任务GutBrainIE。在本次BioASQ挑战中，共有83支参赛队伍参与，提交了总计超过1000份不同的共享任务参赛作品。与往届相同，多个参赛系统表现出了竞争力，表明该领域的一流技术持续进步。 

---
# MedGR$^2$: Breaking the Data Barrier for Medical Reasoning via Generative Reward Learning 

**Title (ZH)**: MedGR\$^2\$: 通过生成奖励学习打破医疗推理的数据壁垒 

**Authors**: Weihai Zhi, Jiayan Guo, Shangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20549)  

**Abstract**: The application of Vision-Language Models (VLMs) in medicine is critically hampered by the scarcity of high-quality, expert-annotated data. Supervised Fine-Tuning (SFT) on existing datasets often leads to poor generalization on unseen modalities and tasks, while Reinforcement Learning (RL), a promising alternative, is stymied by the lack of reliable reward signals in this data-scarce domain. To break this impasse, we introduce Generative Reward Learning for Medical Reasoning (MedGR$^2$), a novel framework that creates a self-improving virtuous cycle. MedGR$^2$ co-develops a data generator and a reward model, enabling the automated, continuous creation of high-quality, multi-modal medical data that serves as both a superior training source for SFT and RL. Our experiments demonstrate that SFT with MedGR$^2$-produced data already surpasses baselines trained on large-scale, human-curated datasets. Crucially, when leveraging this data for RL via Group Relative Policy Optimization (GRPO), our model achieves state-of-the-art cross-modality and cross-task generalization, significantly outperforming specialized RL-based methods. Furthermore, our compact model, empowered by MedGR$^2$, achieves performance competitive with foundation models possessing over 10 times more parameters. MedGR$^2$ presents a new paradigm for data-efficient learning in high-stakes domains, transforming the problem from data scarcity to data generation and unlocking the full potential of RL for building truly generalizable medical AI. 

**Abstract (ZH)**: 医学领域中视觉-语言模型的应用受到高质量专家标注数据稀缺的严重阻碍。现有的监督微调(SFT)往往在未见过的模态和任务上表现出较差的一般化能力，而强化学习(RL)，作为一种有 promise 的替代方法，由于数据稀缺领域的可靠奖励信号缺乏而受阻。为解决这一困境，我们提出了医学推理中的生成奖励学习框架（MedGR$^2$），这是一种新颖的方法，能够创造一个自我改进的良性循环。MedGR$^2$ 共同开发了一个数据生成器和一个奖励模型，使高质量多模态医学数据的自动化和持续生成成为可能，这些数据既可作为监督微调(SFT)和RL的优质训练源。我们的实验证明，使用MedGR$^2$生成的数据进行的监督微调已经超越了在大规模人工标注数据集上训练的基本模型。更关键的是，当利用这些数据通过组相对策略优化(GRPO)进行RL时，我们的模型在跨模态和跨任务一般化方面达到了最先进的性能，显著优于专门的基于RL的方法。此外，我们的紧凑型模型借助MedGR$^2的赋能，性能可与参数量超过其10倍的基模型相匹敌。MedGR$^2”为高风险领域的数据高效学习提供了新的范式，将问题从数据稀缺转变为数据生成，并解锁了RL在构建真正通用的医疗AI方面的全部潜力。 

---
# SPGrasp: Spatiotemporal Prompt-driven Grasp Synthesis in Dynamic Scenes 

**Title (ZH)**: SPGrasp: 动态场景中基于时空提示的抓取合成 

**Authors**: Yunpeng Mei, Hongjie Cao, Yinqiu Xia, Wei Xiao, Zhaohan Feng, Gang Wang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20547)  

**Abstract**: Real-time interactive grasp synthesis for dynamic objects remains challenging as existing methods fail to achieve low-latency inference while maintaining promptability. To bridge this gap, we propose SPGrasp (spatiotemporal prompt-driven dynamic grasp synthesis), a novel framework extending segment anything model v2 (SAMv2) for video stream grasp estimation. Our core innovation integrates user prompts with spatiotemporal context, enabling real-time interaction with end-to-end latency as low as 59 ms while ensuring temporal consistency for dynamic objects. In benchmark evaluations, SPGrasp achieves instance-level grasp accuracies of 90.6% on OCID and 93.8% on Jacquard. On the challenging GraspNet-1Billion dataset under continuous tracking, SPGrasp achieves 92.0% accuracy with 73.1 ms per-frame latency, representing a 58.5% reduction compared to the prior state-of-the-art promptable method RoG-SAM while maintaining competitive accuracy. Real-world experiments involving 13 moving objects demonstrate a 94.8% success rate in interactive grasping scenarios. These results confirm SPGrasp effectively resolves the latency-interactivity trade-off in dynamic grasp synthesis. Code is available at this https URL. 

**Abstract (ZH)**: 基于时空提示的动态抓取实时交互合成（SPGrasp）：一种扩展segment anything model v2 (SAMv2)的新型框架 

---
# MM-HSD: Multi-Modal Hate Speech Detection in Videos 

**Title (ZH)**: 多模态视频仇恨言论检测：MM-HSD 

**Authors**: Berta Céspedes-Sarrias, Carlos Collado-Capell, Pablo Rodenas-Ruiz, Olena Hrynenko, Andrea Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2508.20546)  

**Abstract**: While hate speech detection (HSD) has been extensively studied in text, existing multi-modal approaches remain limited, particularly in videos. As modalities are not always individually informative, simple fusion methods fail to fully capture inter-modal dependencies. Moreover, previous work often omits relevant modalities such as on-screen text and audio, which may contain subtle hateful content and thus provide essential cues, both individually and in combination with others. In this paper, we present MM-HSD, a multi-modal model for HSD in videos that integrates video frames, audio, and text derived from speech transcripts and from frames (i.e.~on-screen text) together with features extracted by Cross-Modal Attention (CMA). We are the first to use CMA as an early feature extractor for HSD in videos, to systematically compare query/key configurations, and to evaluate the interactions between different modalities in the CMA block. Our approach leads to improved performance when on-screen text is used as a query and the rest of the modalities serve as a key. Experiments on the HateMM dataset show that MM-HSD outperforms state-of-the-art methods on M-F1 score (0.874), using concatenation of transcript, audio, video, on-screen text, and CMA for feature extraction on raw embeddings of the modalities. The code is available at this https URL 

**Abstract (ZH)**: 多模态视频仇恨言论检测模型：MM-HSD 

---
# Overview of BioASQ 2024: The twelfth BioASQ challenge on Large-Scale Biomedical Semantic Indexing and Question Answering 

**Title (ZH)**: BioASQ 2024 生物医学大型语义索引与问答挑战赛十二届概述 

**Authors**: Anastasios Nentidis, Georgios Katsimpras, Anastasia Krithara, Salvador Lima-López, Eulàlia Farré-Maduell, Martin Krallinger, Natalia Loukachevitch, Vera Davydova, Elena Tutubalina, Georgios Paliouras  

**Link**: [PDF](https://arxiv.org/pdf/2508.20532)  

**Abstract**: This is an overview of the twelfth edition of the BioASQ challenge in the context of the Conference and Labs of the Evaluation Forum (CLEF) 2024. BioASQ is a series of international challenges promoting advances in large-scale biomedical semantic indexing and question answering. This year, BioASQ consisted of new editions of the two established tasks b and Synergy, and two new tasks: a) MultiCardioNER on the adaptation of clinical entity detection to the cardiology domain in a multilingual setting, and b) BIONNE on nested NER in Russian and English. In this edition of BioASQ, 37 competing teams participated with more than 700 distinct submissions in total for the four different shared tasks of the challenge. Similarly to previous editions, most of the participating systems achieved competitive performance, suggesting the continuous advancement of the state-of-the-art in the field. 

**Abstract (ZH)**: BioASQ挑战的第十二届概述：CLEF 2024会议与评估论坛实验室上下文中的进展与新任务 

---
# BridgeShield: Enhancing Security for Cross-chain Bridge Applications via Heterogeneous Graph Mining 

**Title (ZH)**: BridgeShield: 通过异构图挖掘增强跨链桥应用的安全性 

**Authors**: Dan Lin, Shunfeng Lu, Ziyan Liu, Jiajing Wu, Junyuan Fang, Kaixin Lin, Bowen Song, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.20517)  

**Abstract**: Cross-chain bridges play a vital role in enabling blockchain interoperability. However, due to the inherent design flaws and the enormous value they hold, they have become prime targets for hacker attacks. Existing detection methods show progress yet remain limited, as they mainly address single-chain behaviors and fail to capture cross-chain semantics. To address this gap, we leverage heterogeneous graph attention networks, which are well-suited for modeling multi-typed entities and relations, to capture the complex execution semantics of cross-chain behaviors. We propose BridgeShield, a detection framework that jointly models the source chain, off-chain coordination, and destination chain within a unified heterogeneous graph representation. BridgeShield incorporates intra-meta-path attention to learn fine-grained dependencies within cross-chain paths and inter-meta-path attention to highlight discriminative cross-chain patterns, thereby enabling precise identification of attack behaviors. Extensive experiments on 51 real-world cross-chain attack events demonstrate that BridgeShield achieves an average F1-score of 92.58%, representing a 24.39% improvement over state-of-the-art baselines. These results validate the effectiveness of BridgeShield as a practical solution for securing cross-chain bridges and enhancing the resilience of multi-chain ecosystems. 

**Abstract (ZH)**: 跨链桥梁在促进区块链互操作性方面发挥着关键作用。然而，由于其固有的设计缺陷和所持有的巨大价值，它们成为黑客攻击的主要目标。现有检测方法虽有进展但仍有限制，因为它们主要关注单链行为而未能捕捉到跨链语义。为弥补这一差距，我们利用适合 modeling 多类型实体和关系的异构图注意力网络来捕捉跨链行为的复杂执行语义。我们提出 BridgeShield，这是一种联合建模源链、链外协调和目的链的检测框架，采用统一的异构图表示。BridgeShield 结合使用 intra-meta-path 注意力学习跨链路径内的细粒度依赖关系，并使用 inter-meta-path 注意力突出跨链模式，从而实现对攻击行为的精确识别。在 51 个真实世界的跨链攻击事件上的广泛实验表明，BridgeShield 的平均 F1 得分为 92.58%，比最先进的基线方法改进了 24.39%。这些结果验证了 BridgeShield 作为跨链桥梁安全解决方案的有效性以及对多链生态系统的增强韧性。 

---
# Languages Still Left Behind: Toward a Better Multilingual Machine Translation Benchmark 

**Title (ZH)**: 语言仍被遗忘：向更好的多语言机器翻译评估标准迈进 

**Authors**: Chihiro Taguchi, Seng Mai, Keita Kurabe, Yusuke Sakai, Georgina Agyei, Soudabeh Eslami, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20511)  

**Abstract**: Multilingual machine translation (MT) benchmarks play a central role in evaluating the capabilities of modern MT systems. Among them, the FLORES+ benchmark is widely used, offering English-to-many translation data for over 200 languages, curated with strict quality control protocols. However, we study data in four languages (Asante Twi, Japanese, Jinghpaw, and South Azerbaijani) and uncover critical shortcomings in the benchmark's suitability for truly multilingual evaluation. Human assessments reveal that many translations fall below the claimed 90% quality standard, and the annotators report that source sentences are often too domain-specific and culturally biased toward the English-speaking world. We further demonstrate that simple heuristics, such as copying named entities, can yield non-trivial BLEU scores, suggesting vulnerabilities in the evaluation protocol. Notably, we show that MT models trained on high-quality, naturalistic data perform poorly on FLORES+ while achieving significant gains on our domain-relevant evaluation set. Based on these findings, we advocate for multilingual MT benchmarks that use domain-general and culturally neutral source texts rely less on named entities, in order to better reflect real-world translation challenges. 

**Abstract (ZH)**: 多语言机器翻译基准在评估现代机器翻译系统的能力中发挥着核心作用。其中，FLORES+基准被广泛使用，提供了面向200多种语言的英译多语言数据，并遵循严格的质量控制协议。然而，我们研究了四种语言（阿桑蒂陶伊、日语、景颇语和西南土耳其语）的数据，并发现基准在真正多语言评估方面的适用性存在关键不足。人类评估显示，许多翻译未能达到声称的90%质量标准，且注释人员表示，源句子往往过于领域特定且文化上偏向英语国家。我们进一步证明，简单的启发式方法，如复制命名实体，可以产生非平凡的BLEU分数，这表明评估协议存在漏洞。值得注意的是，我们表明，训练于高质量自然数据的机器翻译模型在FLORES+上表现不佳，但在我们相关的领域评估集上取得了显著进步。基于这些发现，我们建议使用通用领域和文化中立的源文本的多语言机器翻译基准，并减少对命名实体的依赖，以便更好地反映实际翻译挑战。 

---
# CaddieSet: A Golf Swing Dataset with Human Joint Features and Ball Information 

**Title (ZH)**: CaddieSet: 一名高尔夫挥杆数据集，包含人体关节特征和球信息 

**Authors**: Seunghyeon Jung, Seoyoung Hong, Jiwoo Jeong, Seungwon Jeong, Jaerim Choi, Hoki Kim, Woojin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.20491)  

**Abstract**: Recent advances in deep learning have led to more studies to enhance golfers' shot precision. However, these existing studies have not quantitatively established the relationship between swing posture and ball trajectory, limiting their ability to provide golfers with the necessary insights for swing improvement. In this paper, we propose a new dataset called CaddieSet, which includes joint information and various ball information from a single shot. CaddieSet extracts joint information from a single swing video by segmenting it into eight swing phases using a computer vision-based approach. Furthermore, based on expert golf domain knowledge, we define 15 key metrics that influence a golf swing, enabling the interpretation of swing outcomes through swing-related features. Through experiments, we demonstrated the feasibility of CaddieSet for predicting ball trajectories using various benchmarks. In particular, we focus on interpretable models among several benchmarks and verify that swing feedback using our joint features is quantitatively consistent with established domain knowledge. This work is expected to offer new insight into golf swing analysis for both academia and the sports industry. 

**Abstract (ZH)**: Recent Advances in Deep Learning Have Led to More Studies to Enhance Golfers' Shot Precision: CaddieSet Dataset for Quantitative Analysis of Swing and Ball Trajectory Relationship 

---
# Photonic restricted Boltzmann machine for content generation tasks 

**Title (ZH)**: 基于光子的受限制玻尔兹曼机用于内容生成任务 

**Authors**: Li Luo, Yisheng Fang, Wanyi Zhang, Zhichao Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20472)  

**Abstract**: The restricted Boltzmann machine (RBM) is a neural network based on the Ising model, well known for its ability to learn probability distributions and stochastically generate new content. However, the high computational cost of Gibbs sampling in content generation tasks imposes significant bottlenecks on electronic implementations. Here, we propose a photonic restricted Boltzmann machine (PRBM) that leverages photonic computing to accelerate Gibbs sampling, enabling efficient content generation. By introducing an efficient encoding method, the PRBM eliminates the need for computationally intensive matrix decomposition and reduces the computational complexity of Gibbs sampling from $O(N)$ to $O(1)$. Moreover, its non-Von Neumann photonic computing architecture circumvents the memory storage of interaction matrices, providing substantial advantages for large-scale RBMs. We experimentally validate the photonic-accelerated Gibbs sampling by simulating a two-dimensional Ising model, where the observed phase transition temperature closely matches the theoretical predictions. Beyond physics-inspired tasks, the PRBM demonstrates robust capabilities in generating and restoring diverse content, including images and temporal sequences, even in the presence of noise and aberrations. The scalability and reduced training cost of the PRBM framework underscore its potential as a promising pathway for advancing photonic computing in generative artificial intelligence. 

**Abstract (ZH)**: 基于光子计算的受限制玻尔兹曼机（PRBM）及其加速 Gibbs 抽样方法 

---
# Dual-Model Weight Selection and Self-Knowledge Distillation for Medical Image Classification 

**Title (ZH)**: 双模型权重选择与自我知识蒸馏在医学图像分类中的应用 

**Authors**: Ayaka Tsutsumi, Guang Li, Ren Togo, Takahiro Ogawa, Satoshi Kondo, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2508.20461)  

**Abstract**: We propose a novel medical image classification method that integrates dual-model weight selection with self-knowledge distillation (SKD). In real-world medical settings, deploying large-scale models is often limited by computational resource constraints, which pose significant challenges for their practical implementation. Thus, developing lightweight models that achieve comparable performance to large-scale models while maintaining computational efficiency is crucial. To address this, we employ a dual-model weight selection strategy that initializes two lightweight models with weights derived from a large pretrained model, enabling effective knowledge transfer. Next, SKD is applied to these selected models, allowing the use of a broad range of initial weight configurations without imposing additional excessive computational cost, followed by fine-tuning for the target classification tasks. By combining dual-model weight selection with self-knowledge distillation, our method overcomes the limitations of conventional approaches, which often fail to retain critical information in compact models. Extensive experiments on publicly available datasets-chest X-ray images, lung computed tomography scans, and brain magnetic resonance imaging scans-demonstrate the superior performance and robustness of our approach compared to existing methods. 

**Abstract (ZH)**: 我们提出了一种结合双模型权重选择与自我知识蒸馏（SKD）的新型医疗图像分类方法。 

---
# Evaluating Differentially Private Generation of Domain-Specific Text 

**Title (ZH)**: 评价差异隐私生成领域特定文本 

**Authors**: Yidan Sun, Viktor Schlegel, Srinivasan Nandakumar, Iqra Zahid, Yuping Wu, Warren Del-Pinto, Goran Nenadic, Siew-Kei Lam, Jie Zhang, Anil A Bharath  

**Link**: [PDF](https://arxiv.org/pdf/2508.20452)  

**Abstract**: Generative AI offers transformative potential for high-stakes domains such as healthcare and finance, yet privacy and regulatory barriers hinder the use of real-world data. To address this, differentially private synthetic data generation has emerged as a promising alternative. In this work, we introduce a unified benchmark to systematically evaluate the utility and fidelity of text datasets generated under formal Differential Privacy (DP) guarantees. Our benchmark addresses key challenges in domain-specific benchmarking, including choice of representative data and realistic privacy budgets, accounting for pre-training and a variety of evaluation metrics. We assess state-of-the-art privacy-preserving generation methods across five domain-specific datasets, revealing significant utility and fidelity degradation compared to real data, especially under strict privacy constraints. These findings underscore the limitations of current approaches, outline the need for advanced privacy-preserving data sharing methods and set a precedent regarding their evaluation in realistic scenarios. 

**Abstract (ZH)**: 生成式AI在医疗保健和金融等领域提供变革性潜力，但由于隐私和监管障碍，限制了实际数据的使用。为此，差异化隐私合成数据生成已成为一种有前景的替代方案。本研究引入了一个统一基准，系统评估在正式差异隐私（DP）保证下生成的文本数据集的实用性和保真度。该基准解决了领域特定基准测试中的关键挑战，包括代表数据的选择和现实的隐私预算，考虑了预训练并采用多种评估指标。我们评估了五种领域特定数据集上的最新隐私保护生成方法，结果显示，在严格隐私约束下，与真实数据相比，实用性和保真度显著下降。这些发现突显了当前方法的局限性，指出了急需先进的隐私保护数据共享方法，并为其实用场景下的评估设定了范例。 

---
# Towards Mitigating Excessive Forgetting in LLM Unlearning via Entanglement-Aware Unlearning with Proxy Constraint 

**Title (ZH)**: 面向代理约束下的纠缠感知遗忘，以减轻大模型过度遗忘 

**Authors**: Zhihao Liu, Jian Lou, Yuke Hu, Xiaochen Li, Tailun Chen, Yitian Chen, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20443)  

**Abstract**: Large language models (LLMs) are trained on massive datasets that may include private or copyrighted content. Due to growing privacy and ownership concerns, data owners may request the removal of their data from trained models. Machine unlearning provides a practical solution by removing the influence of specific data without full retraining. However, most existing methods lack a sound forgetting boundary, causing some samples to be under-forgotten, leaving residual leakage risks, while others remain over-forgotten at the expense of degraded utility.
In this work, we propose EAGLE-PC (Entanglement-Awareness Guided Loss Reweighting with Proxy Constraint), a novel unlearning framework that addresses these limitations through two key components. First, entanglement-awareness guided loss reweighting determines the forgetting effort of each sample by measuring its similarity to retain samples in the embedding space, enabling more targeted and effective unlearning. Second, a proxy constraint leveraging ICL (In-Context Learning) generated test data softly regularizes the forgetting process, effectively mitigating over-forgetting. EAGLE-PC is compatible with existing gradient-based objectives and serves as a plug-and-play enhancement. We evaluate EAGLE-PC on the TOFU and MUSE benchmarks, showing consistent improvements in the forgetting-utility trade-off across multiple LLMs. Combined with the NPO+GD optimizer, it approaches full retraining performance, offering a scalable and robust unlearning solution. 

**Abstract (ZH)**: 大型语言模型（LLMs）是在可能包含私人或受版权保护内容的庞大数据集上训练的。由于隐私和所有权方面的担忧日益增加，数据所有者可能会要求从训练模型中删除其数据。机器忘记提供了一种实际解决方案，通过移除特定数据的影响而不进行全面重训。然而，现有大多数方法缺乏坚实的遗忘边界，导致某些样本遗忘不足，留下残留泄漏风险，而其他样本则过度遗忘，牺牲了使用价值。
在本文中，我们提出了一种新的机器忘记框架EAGLE-PC（Entanglement-Awareness Guided Loss Reweighting with Proxy Constraint），通过两个关键组件来解决这些限制问题。首先，结合纠缠意识指导的损失重新加权，通过测量每个样本与保留样本在嵌入空间中的相似性来确定遗忘努力，从而实现更精确和有效的机器忘记。其次，利用ICL（上下文学习）生成的测试数据引入的代理约束软化调节遗忘过程，有效缓解了过度遗忘。EAGLE-PC与现有基于梯度的目标兼容，并作为即插即用增强功能。我们在TOFU和MUSE基准上评估了EAGLE-PC，展示了在多个LLM上的遗忘-效用权衡中的一致改进。结合NPO+GD优化器，它可以接近全面重训的性能，提供一种可扩展且稳健的机器忘记解决方案。 

---
# Uncovering the Spectral Bias in Diagonal State Space Models 

**Title (ZH)**: 揭示对角状态空间模型的频谱偏见 

**Authors**: Ruben Solozabal, Velibor Bojkovic, Hilal AlQuabeh, Kentaro Inui, Martin Takáč  

**Link**: [PDF](https://arxiv.org/pdf/2508.20441)  

**Abstract**: Current methods for initializing state space models (SSMs) parameters mainly rely on the \textit{HiPPO framework}, which is based on an online approximation of orthogonal polynomials. Recently, diagonal alternatives have shown to reach a similar level of performance while being significantly more efficient due to the simplification in the kernel computation. However, the \textit{HiPPO framework} does not explicitly study the role of its diagonal variants. In this paper, we take a further step to investigate the role of diagonal SSM initialization schemes from the frequency perspective. Our work seeks to systematically understand how to parameterize these models and uncover the learning biases inherent in such diagonal state-space models. Based on our observations, we propose a diagonal initialization on the discrete Fourier domain \textit{S4D-DFouT}. The insights in the role of pole placing in the initialization enable us to further scale them and achieve state-of-the-art results on the Long Range Arena benchmark, allowing us to train from scratch on very large datasets as PathX-256. 

**Abstract (ZH)**: 当前用于初始化状态空间模型参数的方法主要依赖于HiPPO框架，该框架基于在线正交多项式的近似计算。近期，对角线替代方法已显示出相似的性能水平，由于核计算的简化而显著更加高效。然而，HiPPO框架并没有明确研究其对角线变体的作用。在本文中，我们从频率角度进一步探讨对角线状态空间模型初始化方案的作用。我们的工作旨在系统地理解如何参数化这些模型，并揭示此类对角线状态空间模型中的学习偏见。基于我们的观察，我们提出了一种基于离散傅里叶域的对角线初始化方法S4D-DFouT。对极点放置在初始化中的作用的洞察使我们能够进一步扩展该方法，并在Long Range Arena基准上取得最先进的结果，使我们能够从头开始在非常大的数据集上进行训练，如PathX-256。 

---
# On Identifying Why and When Foundation Models Perform Well on Time-Series Forecasting Using Automated Explanations and Rating 

**Title (ZH)**: 基于自动化解释和评价，探究基础模型在时间序列预测中表现优异的原因及时机 

**Authors**: Michael Widener, Kausik Lakkaraju, John Aydin, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2508.20437)  

**Abstract**: Time-series forecasting models (TSFM) have evolved from classical statistical methods to sophisticated foundation models, yet understanding why and when these models succeed or fail remains challenging. Despite this known limitation, time series forecasting models are increasingly used to generate information that informs real-world actions with equally real consequences. Understanding the complexity, performance variability, and opaque nature of these models then becomes a valuable endeavor to combat serious concerns about how users should interact with and rely on these models' outputs. This work addresses these concerns by combining traditional explainable AI (XAI) methods with Rating Driven Explanations (RDE) to assess TSFM performance and interpretability across diverse domains and use cases. We evaluate four distinct model architectures: ARIMA, Gradient Boosting, Chronos (time-series specific foundation model), Llama (general-purpose; both fine-tuned and base models) on four heterogeneous datasets spanning finance, energy, transportation, and automotive sales domains. In doing so, we demonstrate that feature-engineered models (e.g., Gradient Boosting) consistently outperform foundation models (e.g., Chronos) in volatile or sparse domains (e.g., power, car parts) while providing more interpretable explanations, whereas foundation models excel only in stable or trend-driven contexts (e.g., finance). 

**Abstract (ZH)**: 时间序列预测模型（TSFM）从经典统计方法演进到了复杂的基础模型，但仍对这些模型为何成功或失败缺乏深刻理解。尽管存在这一已知局限性，时间序列预测模型仍被广泛用于生成影响现实世界行动的重要信息。因此，理解这些模型的复杂性、性能变异性及不透明性质，成为了一个有价值的研究领域，以解决用户如何与这些模型的输出进行交互和依赖的问题。本文通过结合传统的可解释人工智能（XAI）方法和评分驱动解释（RDE），评估了TSFM在不同领域的性能和可解释性。我们在金融、能源、交通和汽车销售等多个领域中，使用了四种不同的模型架构（ARIMA、梯度提升、Chronos、Llama）对四个异构数据集进行了评估。结果显示，在波动或稀疏的数据领域（如电力、汽车零部件），特征工程模型（如梯度提升）始终优于基础模型（如Chronos），并且提供更可解释的解释；而在稳定或趋势驱动的背景下（如金融），基础模型则表现出色。 

---
# Rethinking Purity and Diversity in Multi-Behavior Sequential Recommendation from the Frequency Perspective 

**Title (ZH)**: 从频率视角重新思考多行为序列推荐中的纯度与多样性 

**Authors**: Yongqiang Han, Kai Cheng, Kefan Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20427)  

**Abstract**: In recommendation systems, users often exhibit multiple behaviors, such as browsing, clicking, and purchasing. Multi-behavior sequential recommendation (MBSR) aims to consider these different behaviors in an integrated manner to improve the recommendation performance of the target behavior. However, some behavior data will also bring inevitable noise to the modeling of user interests. Some research efforts focus on data denoising from the frequency domain perspective to improve the accuracy of user preference prediction. These studies indicate that low-frequency information tends to be valuable and reliable, while high-frequency information is often associated with noise. In this paper, we argue that high-frequency information is by no means insignificant. Further experimental results highlight that low frequency corresponds to the purity of user interests, while high frequency corresponds to the diversity of user interests. Building upon this finding, we proposed our model PDB4Rec, which efficiently extracts information across various frequency bands and their relationships, and introduces Boostrapping Balancer mechanism to balance their contributions for improved recommendation performance. Sufficient experiments on real-world datasets demonstrate the effectiveness and efficiency of our model. 

**Abstract (ZH)**: 多行为序列推荐中高频率信息的重要性及PDB4Rec模型 

---
# DentalBench: Benchmarking and Advancing LLMs Capability for Bilingual Dentistry Understanding 

**Title (ZH)**: DentalBench: 基于双语牙科理解能力评估与提升的基准测试 

**Authors**: Hengchuan Zhu, Yihuan Xu, Yichen Li, Zijie Meng, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20416)  

**Abstract**: Recent advances in large language models (LLMs) and medical LLMs (Med-LLMs) have demonstrated strong performance on general medical benchmarks. However, their capabilities in specialized medical fields, such as dentistry which require deeper domain-specific knowledge, remain underexplored due to the lack of targeted evaluation resources. In this paper, we introduce DentalBench, the first comprehensive bilingual benchmark designed to evaluate and advance LLMs in the dental domain. DentalBench consists of two main components: DentalQA, an English-Chinese question-answering (QA) benchmark with 36,597 questions spanning 4 tasks and 16 dental subfields; and DentalCorpus, a large-scale, high-quality corpus with 337.35 million tokens curated for dental domain adaptation, supporting both supervised fine-tuning (SFT) and retrieval-augmented generation (RAG). We evaluate 14 LLMs, covering proprietary, open-source, and medical-specific models, and reveal significant performance gaps across task types and languages. Further experiments with Qwen-2.5-3B demonstrate that domain adaptation substantially improves model performance, particularly on knowledge-intensive and terminology-focused tasks, and highlight the importance of domain-specific benchmarks for developing trustworthy and effective LLMs tailored to healthcare applications. 

**Abstract (ZH)**: Recent Advances in Large Language Models and Medical LLMs: Introducing DentalBench, the First Comprehensive Bilingual Benchmark for Evaluating and Advancing LLMs in the Dental Domain 

---
# Assessing local deformation and computing scalar curvature with nonlinear conformal regularization of decoders 

**Title (ZH)**: 评估局部变形并计算标量曲率的非线性共形正则化解码器方法 

**Authors**: Benjamin Couéraud, Vikram Sunkara, Christof Schütte  

**Link**: [PDF](https://arxiv.org/pdf/2508.20413)  

**Abstract**: One aim of dimensionality reduction is to discover the main factors that explain the data, and as such is paramount to many applications. When working with high dimensional data, autoencoders offer a simple yet effective approach to learn low-dimensional representations. The two components of a general autoencoder consist first of an encoder that maps the observed data onto a latent space; and second a decoder that maps the latent space back to the original observation space, which allows to learn a low-dimensional manifold representation of the original data. In this article, we introduce a new type of geometric regularization for decoding maps approximated by deep neural networks, namely nonlinear conformal regularization. This regularization procedure permits local variations of the decoder map and comes with a new scalar field called conformal factor which acts as a quantitative indicator of the amount of local deformation sustained by the latent space when mapped into the original data space. We also show that this regularization technique allows the computation of the scalar curvature of the learned manifold. Implementation and experiments on the Swiss roll and CelebA datasets are performed to illustrate how to obtain these quantities from the architecture. 

**Abstract (ZH)**: 降低数据维度的一目的是发现解释数据的主要因素，这对于许多应用至关重要。在处理高维数据时，自编码器提供了一种简单有效的方法来学习低维表示。一个通用自编码器由两个部分组成：首先是一个编码器，将观测数据映射到潜在空间；其次是一个解码器，将潜在空间映射回原始观测空间，从而学习原始数据的低维流形表示。本文介绍了一种新的几何正则化方法，用于近似由深神经网络实现的解码映射，即非线性共形正则化。该正则化过程允许解码映射的局部变化，并伴随一个新标量场称为共形因子，它作为潜在空间在映射到原始数据空间时所承受的局部变形量的定量指标。我们还展示了这种正则化技术如何计算所学习流形的标量曲率。在瑞士卷和CelebA数据集上的实现和实验说明了如何从架构中获取这些量。 

---
# MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever 

**Title (ZH)**: MPFormer: 自适应工业多任务个性化序列检索框架 

**Authors**: Yijia Sun, Shanshan Huang, Linxiao Che, Haitao Lu, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.20400)  

**Abstract**: Modern industrial recommendation systems encounter a core challenge of multi-stage optimization misalignment: a significant semantic gap exists between the multi-objective optimization paradigm widely used in the ranking phase and the single-objective modeling in the retrieve phase. Although the mainstream industry solution achieves multi-objective coverage through parallel multi-path single-objective retrieval, this approach leads to linear growth of training and serving resources with the number of objectives and has inherent limitations in handling loosely coupled objectives. This paper proposes the MPFormer, a dynamic multi-task Transformer framework, which systematically addresses the aforementioned issues through three innovative mechanisms. First, an objective-conditioned transformer that jointly encodes user behavior sequences and multi-task semantics through learnable attention modulation; second, personalized target weights are introduced to achieve dynamic adjustment of retrieval results; finally, user personalization information is incorporated into token representations and the Transformer structure to further enhance the model's representation ability. This framework has been successfully integrated into Kuaishou short video recommendation system, stably serving over 400 million daily active users. It significantly improves user daily engagement and system operational efficiency. Practical deployment verification shows that, compared with traditional solutions, it effectively optimizes the iterative paradigm of multi-objective retrieval while maintaining service response speed, providing a scalable multi-objective solution for industrial recommendation systems. 

**Abstract (ZH)**: 现代工业推荐系统中的多阶段优化不一致性问题：基于排名阶段广泛使用的多目标优化范式与检索阶段的单目标建模之间的显著语义鸿沟的动态多任务Transformer框架 

---
# TF-TransUNet1D: Time-Frequency Guided Transformer U-Net for Robust ECG Denoising in Digital Twin 

**Title (ZH)**: TF-TransUNet1D：基于时频引导的变压器U-网在数字孪生中的稳健心电图去噪 

**Authors**: Shijie Wang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20398)  

**Abstract**: Electrocardiogram (ECG) signals serve as a foundational data source for cardiac digital twins, yet their diagnostic utility is frequently compromised by noise and artifacts. To address this issue, we propose TF-TransUNet1D, a novel one-dimensional deep neural network that integrates a U-Net-based encoder-decoder architecture with a Transformer encoder, guided by a hybrid time-frequency domain loss. The model is designed to simultaneously capture local morphological features and long-range temporal dependencies, which are critical for preserving the diagnostic integrity of ECG signals. To enhance denoising robustness, we introduce a dual-domain loss function that jointly optimizes waveform reconstruction in the time domain and spectral fidelity in the frequency domain. In particular, the frequency-domain component effectively suppresses high-frequency noise while maintaining the spectral structure of the signal, enabling recovery of subtle but clinically significant waveform components. We evaluate TF-TransUNet1D using synthetically corrupted signals from the MIT-BIH Arrhythmia Database and the Noise Stress Test Database (NSTDB). Comparative experiments against state-of-the-art baselines demonstrate consistent superiority of our model in terms of SNR improvement and error metrics, achieving a mean absolute error of 0.1285 and Pearson correlation coefficient of 0.9540. By delivering high-precision denoising, this work bridges a critical gap in pre-processing pipelines for cardiac digital twins, enabling more reliable real-time monitoring and personalized modeling. 

**Abstract (ZH)**: 基于时间频率域损失的TF-TransUNet1D用于ECG信号去噪与心脏数字双生预处理 

---
# Measuring Reasoning Utility in LLMs via Conditional Entropy Reduction 

**Title (ZH)**: 通过条件熵减少测量LLM的推理有用性 

**Authors**: Xu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.20395)  

**Abstract**: Recent advancements in large language models (LLMs) often rely on generating intermediate reasoning steps to enhance accuracy. However, little work has examined how reasoning utility contributes to the final answer's correctness. Due to the stochastic nature of autoregressive generation, generating more context does not guarantee increased confidence in the answer. If we could predict, during generation, whether a reasoning step will be useful, we could stop early or prune ineffective steps, avoiding distractions in the final decision.
We present an oracle study on MATH dataset, using Qwen2.5-32B and GPT-4o to generate reasoning chains, and then employing a separate model (Qwen3-8B) to quantify the utility of these chains for final accuracy. Specifically, we measure the model's uncertainty on the answer span Y at each reasoning step using conditional entropy (expected negative log-likelihood over the vocabulary) with context expanding step by step. Our results show a clear pattern: conditional entropy that decreases over steps is strongly associated with correct answers, whereas flat or increasing entropy often results in wrong answers. We also corroborate that incorrect reasoning paths tend to be longer than correct ones, suggesting that longer reasoning does not necessarily yield better outcomes. These findings serve as a foundation to inspire future work on designing efficient reasoning pipelines that detect and avoid unproductive reasoning early. 

**Abstract (ZH)**: 最近大规模语言模型的进展往往依赖于生成中间推理步骤以提高准确性。然而，很少有研究探讨推理有用性如何影响最终答案的正确性。由于自回归生成的随机性，增加上下文并不保证回答的置信度提高。如果我们能在生成过程中预测某个推理步骤是否有用，便可以在必要时提前停止或去除无效步骤，从而避免最终决策中的干扰。

我们使用Qwen2.5-32B和GPT-4o在MATH数据集上进行了一个先验研究，生成推理链，然后利用另一个单独的模型（Qwen3-8B）来量化这些链对于最终准确性的有用性。具体而言，我们使用条件熵（基于词汇的期望负对数似然）在每一步推理中度量模型对答案区间Y的不确定性，逐步扩展上下文。实验结果表明，条件熵在步骤中呈下降趋势与正确答案密切相关，而平稳或增加的熵通常会导致错误的答案。我们还验证了错误的推理路径往往比正确的路径更长，这表明更长的推理不一定能带来更好的结果。这些发现为进一步设计高效推理流水线提供了一个基础，该流水线能够在早期检测和避免无成效的推理。 

---
# Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection 

**Title (ZH)**: 具有时间依赖性整合发放神经元模型的超低延迟脉冲神经网络及其在物体检测中的应用 

**Authors**: Chengjun Zhang, Yuhao Zhang, Jie Yang, Mohamad Sawan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20392)  

**Abstract**: Spiking Neural Networks (SNNs), inspired by the brain, are characterized by minimal power consumption and swift inference capabilities on neuromorphic hardware, and have been widely applied to various visual perception tasks. Current ANN-SNN conversion methods have achieved excellent results in classification tasks with ultra-low time-steps, but their performance in visual detection tasks remains suboptimal. In this paper, we propose a delay-spike approach to mitigate the issue of residual membrane potential caused by heterogeneous spiking patterns. Furthermore, we propose a novel temporal-dependent Integrate-and-Fire (tdIF) neuron architecture for SNNs. This enables Integrate-and-fire (IF) neurons to dynamically adjust their accumulation and firing behaviors based on the temporal order of time-steps. Our method enables spikes to exhibit distinct temporal properties, rather than relying solely on frequency-based representations. Moreover, the tdIF neuron maintains energy consumption on par with traditional IF neuron. We demonstrate that our method achieves more precise feature representation with lower time-steps, enabling high performance and ultra-low latency in visual detection tasks. In this study, we conduct extensive evaluation of the tdIF method across two critical vision tasks: object detection and lane line detection. The results demonstrate that the proposed method surpasses current ANN-SNN conversion approaches, achieving state-of-the-art performance with ultra-low latency (within 5 time-steps). 

**Abstract (ZH)**: 基于延迟脉冲的神经形态Integrate-and-Fire神经元架构及其在视觉检测任务中的应用 

---
# Graph-R1: Unleashing LLM Reasoning with NP-Hard Graph Problems 

**Title (ZH)**: 图-R1：利用NP难图问题释放大语言模型的推理能力 

**Authors**: Yuyao Wang, Bowen Liu, Jianheng Tang, Nuo Chen, Yuhan Li, Qifan Zhang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20373)  

**Abstract**: Reasoning Large Language Models (RLLMs) have recently achieved remarkable progress on complex reasoning tasks, largely enabled by their long chain-of-thought (Long CoT) capabilities. However, developing these Long CoT behaviors relies heavily on post-training with high-quality datasets, which are typically costly and human-curated (e.g., mathematics and code), leaving scalable alternatives unexplored. In this work, we introduce NP-hard (NPH) graph problems as a novel synthetic training corpus, as they inherently require deep reasoning, extensive exploration, and reflective strategies, which are core characteristics of Long CoT reasoning. Building on this insight, we develop a two-stage post-training framework: (i) Long CoT Supervised Fine-Tuning (SFT) on rejection-sampled NPH graph instances, which substantially enhances reasoning depth, and (ii) Reinforcement Learning (RL) with a fine-grained reward design, which sharpens reasoning efficiency. Our flagship model, Graph-R1-7B, demonstrates strong generalization across mathematics, coding, STEM, and logic, and surpasses QwQ-32B on NPH graph problems in both accuracy and reasoning efficiency. These results position NPH graph problems as an effective and scalable resource for advancing Long CoT reasoning in LLMs, opening a new frontier for LLM post-training. Our implementation is available at this https URL, with models and datasets hosted in our Hugging Face collection HKUST-DSAIL/Graph-R1. 

**Abstract (ZH)**: 大规模语言模型（RLLMs）在复杂推理任务上的近期进展主要得益于它们的长链推理（Long CoT）能力。然而，发展这些Long CoT行为严重依赖于高质量数据集的后期训练，这类数据集通常成本高且需要人工筛选（如数学和代码），导致可扩展的替代方案未被探索。在本文中，我们引入NP难（NPH）图问题作为新颖的合成训练语料库，因为它们本质上要求深度推理、广泛探索和反思策略，这些都是Long CoT推理的核心特征。基于此，我们开发了一个两阶段后期训练框架：(i) 基于采样的NPH图实例的长链监督微调（SFT），显著增强了推理深度；(ii) 以精细设计的奖励机制为基础的强化学习（RL），提升了推理效率。我们的旗舰模型Graph-R1-7B在数学、编程、STEM和逻辑等多个领域表现出强大的泛化能力，并在NP难图问题上的准确性和推理效率上超越了QwQ-32B。这些结果表明NP难图问题是一个有效且可扩展的资源，有助于推进LLMs中的长链推理，开启了LLMs后期训练的新前沿。我们的实现可在以下链接获取：this https URL，模型和数据集托管在我们的Hugging Face集合HKUST-DSAIL/Graph-R1中。 

---
# Adaptive Root Cause Localization for Microservice Systems with Multi-Agent Recursion-of-Thought 

**Title (ZH)**: 支持多Agent递归思考的微服务系统自适应根因定位 

**Authors**: Lingzhe Zhang, Tong Jia, Kangjin Wang, Weijie Hong, Chiming Duan, Minghua He, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20370)  

**Abstract**: As contemporary microservice systems become increasingly popular and complex-often comprising hundreds or even thousands of fine-grained, interdependent subsystems-they are facing more frequent failures. Ensuring system reliability thus demands accurate root cause localization. While traces and metrics have proven to be effective data sources for this task, existing methods either heavily rely on pre-defined schemas, which struggle to adapt to evolving operational contexts, or lack interpretability in their reasoning process, thereby leaving Site Reliability Engineers (SREs) confused. In this paper, we conduct a comprehensive study on how SREs localize the root cause of failures, drawing insights from multiple professional SREs across different organizations. Our investigation reveals that human root cause analysis exhibits three key characteristics: recursiveness, multi-dimensional expansion, and cross-modal reasoning. Motivated by these findings, we introduce RCLAgent, an adaptive root cause localization method for microservice systems that leverages a multi-agent recursion-of-thought framework. RCLAgent employs a novel recursion-of-thought strategy to guide the LLM's reasoning process, effectively integrating data from multiple agents and tool-assisted analysis to accurately pinpoint the root cause. Experimental evaluations on various public datasets demonstrate that RCLAgent achieves superior performance by localizing the root cause using only a single request-outperforming state-of-the-art methods that depend on aggregating multiple requests. These results underscore the effectiveness of RCLAgent in enhancing the efficiency and precision of root cause localization in complex microservice environments. 

**Abstract (ZH)**: 当代微服务系统因日益流行和复杂（常包含数百甚至数千个细粒度、相互依赖的子系统），故障频率增加。因此，确保系统可靠性需要精确的根因定位。虽然追踪信息和指标已被证明是有效的数据来源，但现有方法要么严重依赖预定义的模式（难以适应不断变化的操作环境），要么在推理过程中缺乏可解释性，导致运维工程师（SREs）困惑。本文通过对来自不同组织的多名专业SRE的深入研究，探讨了如何定位故障的根因。研究表明，人类的根因分析具有递归性、多维度扩展性和跨模态推理的三大特征。基于这些发现，我们提出了RCLAgent——一种适应性微服务系统根因定位方法，利用多智能体思想递归框架。RCLAgent采用一种新颖的思想递归策略来引导大模型的推理过程，有效整合多智能体的数据和工具辅助分析，准确地定位根因。在多种公开数据集上进行的实验评估表明，RCLAgent仅使用单一请求即可超越依赖聚合多个请求的最先进的方法，展现出优越的性能。这些结果突显了RCLAgent在复杂微服务环境中的根因定位效率和精确性方面的有效性。 

---
# Boosting Skeleton-Driven SMT Solver Fuzzing by Leveraging LLM to Produce Formula Generators 

**Title (ZH)**: 利用大型语言模型生成公式生成器以增强基于骨架的SMT求解器模糊测试 

**Authors**: Maolin Sun, Yibiao Yang, Yuming Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.20340)  

**Abstract**: Satisfiability Modulo Theory (SMT) solvers are foundational to modern systems and programming languages research, providing the foundation for tasks like symbolic execution and automated verification. Because these solvers sit on the critical path, their correctness is essential, and high-quality test formulas are key to uncovering bugs. However, while prior testing techniques performed well on earlier solver versions, they struggle to keep pace with rapidly evolving features. Recent approaches based on Large Language Models (LLMs) show promise in exploring advanced solver capabilities, but two obstacles remain: nearly half of the generated formulas are syntactically invalid, and iterative interactions with the LLMs introduce substantial computational overhead. In this study, we present Chimera, a novel LLM-assisted fuzzing framework that addresses both issues by shifting from direct formula generation to the synthesis of reusable term (i.e., logical expression) generators. Particularly, Chimera uses LLMs to (1) automatically extract context-free grammars (CFGs) for SMT theories, including solver-specific extensions, from documentation, and (2) synthesize composable Boolean term generators that adhere to these grammars. During fuzzing, Chimera populates structural skeletons derived from existing formulas with the terms iteratively produced by the LLM-synthesized generators. This design ensures syntactic validity while promoting semantic diversity. Notably, Chimera requires only one-time LLM interaction investment, dramatically reducing runtime cost. We evaluated Chimera on two leading SMT solvers: Z3 and cvc5. Our experiments show that Chimera has identified 43 confirmed bugs, 40 of which have already been fixed by developers. 

**Abstract (ZH)**: 模理论饱和性（SMT）求解器是现代系统和编程语言研究的基础，为符号执行和自动验证等任务提供支持。由于这些求解器位于关键路径上，其正确性至关重要，高质量的测试公式是发现错误的关键。然而，尽管之前的一些测试技术在早期版本的求解器上表现良好，但它们难以跟上快速演变的功能。基于大型语言模型（LLMs）的近期方法显示出探索求解器高级能力的潜力，但存在两个障碍：生成的公式中几乎有一半是语法无效的，与LLMs的迭代交互引入了显著的计算开销。在本研究中，我们提出了Chimera，这是一种新颖的LLM辅助模糊测试框架，通过从文档中自动提取模理论的上下文无关文法（CFG），包括求解器特定扩展，以及合成遵循这些文法的可组合布尔表达式生成器，来解决这两个问题。Chimera在模糊测试过程中，利用LLM合成的生成器逐步填充源自现有公式的结构骨架，确保语法有效性的同时促进语义多样性。值得注意的是，Chimera只需要一次LLM交互投资，显著降低了运行时成本。我们在两个领先SMT求解器：Z3和cvc5上评估了Chimera。实验结果表明，Chimera发现了43个已确认的错误，其中40个已经被开发者修复。 

---
# Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs 

**Title (ZH)**: 一次投毒，永绝后患：利用对齐武器化注入LLMs偏见 

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh  

**Link**: [PDF](https://arxiv.org/pdf/2508.20333)  

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications. 

**Abstract (ZH)**: Large Language Models (LLMs)通过训练拒绝回答有害或不安全的提示来对齐以符合伦理标准和安全要求。本文展示了对手如何利用LLM的对齐机制植入偏见或执行有针对性的审查，而不降低模型对无关话题的响应能力。具体来说，我们提出了一种颠覆性对齐注入（SAI）攻击，该攻击利用对齐机制在由对手预定义的主题或查询上触发拒绝。尽管通过过度对齐诱导拒绝或许不令人惊讶，我们展示了如何利用这种拒绝注入偏见到模型中。令人惊讶的是，SAI能够避开最先进的中毒防御措施，包括LLM状态法医分析以及专为FL环境设计的健壮聚合技术。我们通过展示该攻击对LLM驱动的应用管道的端到端影响，阐明其实用危险。对于类似于ChatDoctor的聊天应用，1%的数据中毒会导致系统拒绝回答针对特定种族类别的医疗问题，产生高偏见($\Delta DP$为23%)。我们还展示了偏见可以在其他NLP任务中被诱导：在一项旨在拒绝总结指定大学简历的招聘流程中，选择偏见($\Delta DP$为27%)增加。在9个其他聊天驱动的下游应用中，偏见甚至更高($\Delta DP$约为38%)。 

---
# Multi-View Graph Convolution Network for Internal Talent Recommendation Based on Enterprise Emails 

**Title (ZH)**: 基于企业邮件的多视图图卷积网络内部人才推荐 

**Authors**: Soo Hyun Kim, Jang-Hyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20328)  

**Abstract**: Internal talent recommendation is a critical strategy for organizational continuity, yet conventional approaches suffer from structural limitations, often overlooking qualified candidates by relying on the narrow perspective of a few managers. To address this challenge, we propose a novel framework that models two distinct dimensions of an employee's position fit from email data: WHAT they do (semantic similarity of tasks) and HOW they work (structural characteristics of their interactions and collaborations). These dimensions are represented as independent graphs and adaptively fused using a Dual Graph Convolutional Network (GCN) with a gating mechanism. Experiments show that our proposed gating-based fusion model significantly outperforms other fusion strategies and a heuristic baseline, achieving a top performance of 40.9% on Hit@100. Importantly, it is worth noting that the model demonstrates high interpretability by learning distinct, context-aware fusion strategies for different job families. For example, it learned to prioritize relational (HOW) data for 'sales and marketing' job families while applying a balanced approach for 'research' job families. This research offers a quantitative and comprehensive framework for internal talent discovery, minimizing the risk of candidate omission inherent in traditional methods. Its primary contribution lies in its ability to empirically determine the optimal fusion ratio between task alignment (WHAT) and collaborative patterns (HOW), which is required for employees to succeed in the new positions, thereby offering important practical implications. 

**Abstract (ZH)**: 基于电子邮件数据的员工位置匹配新框架：融合任务内容与协作方式 

---
# GUARD: Guideline Upholding Test through Adaptive Role-play and Jailbreak Diagnostics for LLMs 

**Title (ZH)**: GUARD: 遵循指南测试通过自适应角色扮演和监狱逃脱诊断对于大规模语言模型 

**Authors**: Haibo Jin, Ruoxi Chen, Peiyan Zhang, Andy Zhou, Yang Zhang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20325)  

**Abstract**: As Large Language Models become increasingly integral to various domains, their potential to generate harmful responses has prompted significant societal and regulatory concerns. In response, governments have issued ethics guidelines to promote the development of trustworthy AI. However, these guidelines are typically high-level demands for developers and testers, leaving a gap in translating them into actionable testing questions to verify LLM compliance.
To address this challenge, we introduce GUARD (\textbf{G}uideline \textbf{U}pholding Test through \textbf{A}daptive \textbf{R}ole-play and Jailbreak \textbf{D}iagnostics), a testing method designed to operationalize guidelines into specific guideline-violating questions that assess LLM adherence. To implement this, GUARD uses automated generation of guideline-violating questions based on government-issued guidelines, thereby testing whether responses comply with these guidelines. When responses directly violate guidelines, GUARD reports inconsistencies. Furthermore, for responses that do not directly violate guidelines, GUARD integrates the concept of ``jailbreaks'' to diagnostics, named GUARD-JD, which creates scenarios that provoke unethical or guideline-violating responses, effectively identifying potential scenarios that could bypass built-in safety mechanisms. Our method finally culminates in a compliance report, delineating the extent of adherence and highlighting any violations.
We have empirically validated the effectiveness of GUARD on seven LLMs, including Vicuna-13B, LongChat-7B, Llama2-7B, Llama-3-8B, GPT-3.5, GPT-4, GPT-4o, and Claude-3.7, by testing compliance under three government-issued guidelines and conducting jailbreak diagnostics. Additionally, GUARD-JD can transfer jailbreak diagnostics to vision-language models, demonstrating its usage in promoting reliable LLM-based applications. 

**Abstract (ZH)**: 大型语言模型在各领域中的作用日益重要，其生成有害响应的可能性引起了广泛的社会和监管关注。为应对这一问题，政府发布了伦理指导原则以促进值得信赖的AI发展。然而，这些指导原则通常是对开发者和测试人员的高层次要求，缺乏将这些要求转化为可操作的测试问题以验证LLM合规性的手段。

为此，我们提出了GUARD（Guideline Upholding Test through Adaptive Role-play and Jailbreak Diagnostics）测试方法，旨在将指导原则具体化为特定的指导原则违反问题，以评估LLM的合规性。GUARD通过基于政府发布的指导原则自动生成指导原则违反的问题来实施这一方法，从而测试响应是否符合这些指导原则。当响应直接违反指导原则时，GUARD报告不一致性。此外，对于未直接违反指导原则的响应，GUARD结合“越狱”概念进行诊断，名为GUARD-JD，它创建触发不良行为或指导原则违反的场景，有效地识别可能绕过内置安全机制的潜在场景。最终，我们的方法形成了一份合规报告，详细描述合规程度并突出任何违规行为。

我们通过在七种LLM（包括Vicuna-13B、LongChat-7B、Llama2-7B、Llama-3-8B、GPT-3.5、GPT-4、GPT-4o和Claude-3.7）上进行合规性测试和“越狱”诊断验证了GUARD的有效性，这些测试基于政府发布的三条指导原则。此外，GUARD-JD还可以将“越狱”诊断应用于视觉语言模型，展示了其在促进可靠LLM基础应用方面的作用。 

---
# Differentially Private Federated Quantum Learning via Quantum Noise 

**Title (ZH)**: 差分隐私 Federated 量子学习 via 量子噪声 

**Authors**: Atit Pokharel, Ratun Rahman, Shaba Shaon, Thomas Morris, Dinh C. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20310)  

**Abstract**: Quantum federated learning (QFL) enables collaborative training of quantum machine learning (QML) models across distributed quantum devices without raw data exchange. However, QFL remains vulnerable to adversarial attacks, where shared QML model updates can be exploited to undermine information privacy. In the context of noisy intermediate-scale quantum (NISQ) devices, a key question arises: How can inherent quantum noise be leveraged to enforce differential privacy (DP) and protect model information during training and communication? This paper explores a novel DP mechanism that harnesses quantum noise to safeguard quantum models throughout the QFL process. By tuning noise variance through measurement shots and depolarizing channel strength, our approach achieves desired DP levels tailored to NISQ constraints. Simulations demonstrate the framework's effectiveness by examining the relationship between differential privacy budget and noise parameters, as well as the trade-off between security and training accuracy. Additionally, we demonstrate the framework's robustness against an adversarial attack designed to compromise model performance using adversarial examples, with evaluations based on critical metrics such as accuracy on adversarial examples, confidence scores for correct predictions, and attack success rates. The results reveal a tunable trade-off between privacy and robustness, providing an efficient solution for secure QFL on NISQ devices with significant potential for reliable quantum computing applications. 

**Abstract (ZH)**: 量子联邦学习中利用固有量子噪声保障差分隐私的研究 

---
# Surveying the Operational Cybersecurity and Supply Chain Threat Landscape when Developing and Deploying AI Systems 

**Title (ZH)**: 开发和部署AI系统时调研操作网络安全与供应链威胁 landscape 

**Authors**: Michael R Smith, Joe Ingram  

**Link**: [PDF](https://arxiv.org/pdf/2508.20307)  

**Abstract**: The rise of AI has transformed the software and hardware landscape, enabling powerful capabilities through specialized infrastructures, large-scale data storage, and advanced hardware. However, these innovations introduce unique attack surfaces and objectives which traditional cybersecurity assessments often overlook. Cyber attackers are shifting their objectives from conventional goals like privilege escalation and network pivoting to manipulating AI outputs to achieve desired system effects, such as slowing system performance, flooding outputs with false positives, or degrading model accuracy. This paper serves to raise awareness of the novel cyber threats that are introduced when incorporating AI into a software system. We explore the operational cybersecurity and supply chain risks across the AI lifecycle, emphasizing the need for tailored security frameworks to address evolving threats in the AI-driven landscape. We highlight previous exploitations and provide insights from working in this area. By understanding these risks, organizations can better protect AI systems and ensure their reliability and resilience. 

**Abstract (ZH)**: 人工智能的兴起已Transformer了软件和硬件landscape，通过专门的基础设施、大规模数据存储和先进的硬件增强了强大能力。然而，这些创新引入了传统网络安全评估经常忽略的独特攻击面和目标。网络攻击者的攻击目标已从传统的特权提升和网络跃变转变为操纵AI输出以实现期望的系统效果，如降低系统性能、泛滥误报或降低模型准确性。本文旨在提高人们对将AI集成到软件系统中时引入的新型网络威胁的认识。我们探讨了AI生命周期中操作网络安全和供应链风险，强调需要定制的安全框架来应对AI驱动场景中的不断演变的威胁。我们强调了之前的安全漏洞并提供了在这个领域工作的见解。通过了解这些风险，组织可以更好地保护AI系统并确保其可靠性和弹性。 

---
# Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization 

**Title (ZH)**: 面向上下文世界模型的动态对齐潜想象在零样本泛化中的应用 

**Authors**: Frank Röder, Jan Benad, Manfred Eppe, Pradeep Kr. Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2508.20294)  

**Abstract**: Real-world reinforcement learning demands adaptation to unseen environmental conditions without costly retraining. Contextual Markov Decision Processes (cMDP) model this challenge, but existing methods often require explicit context variables (e.g., friction, gravity), limiting their use when contexts are latent or hard to measure. We introduce Dynamics-Aligned Latent Imagination (DALI), a framework integrated within the Dreamer architecture that infers latent context representations from agent-environment interactions. By training a self-supervised encoder to predict forward dynamics, DALI generates actionable representations conditioning the world model and policy, bridging perception and control. We theoretically prove this encoder is essential for efficient context inference and robust generalization. DALI's latent space enables counterfactual consistency: Perturbing a gravity-encoding dimension alters imagined rollouts in physically plausible ways. On challenging cMDP benchmarks, DALI achieves significant gains over context-unaware baselines, often surpassing context-aware baselines in extrapolation tasks, enabling zero-shot generalization to unseen contextual variations. 

**Abstract (ZH)**: 现实世界的强化学习要求在不进行昂贵的重新训练的情况下适应未见过的环境条件。动态对齐的潜在想象（DALI）框架结合在Dreamer架构中，从代理-环境交互中推断潜在的上下文表示。通过训练一个自监督编码器来预测动态前向模型，DALI生成可操作的表示以条件化世界模型和策略，实现感知与控制的结合。我们从理论上证明了该编码器对于高效上下文推断和稳健泛化是必不可少的。DALI的潜在空间支持反事实一致性：扰动重力编码维度以物理合理的方式改变想象的-rollouts。在具有挑战性的cMDP基准测试中，DALI在上下文无关基准上取得了显著的性能提升，通常在外推任务中超越了上下文感知基准，实现对未见过的上下文变体的零样本泛化。 

---
# Beacon: Post-Training Quantization with Integrated Grid Selection 

**Title (ZH)**: Beacon:  integral Grid Selection for Post-Training Quantization 

**Authors**: Shihao Zhang, Rayan Saab  

**Link**: [PDF](https://arxiv.org/pdf/2508.20293)  

**Abstract**: Quantization is a widely used compression technique for reducing the memory and computation costs of large pre-trained models. A key challenge in per-channel post-training quantization (PTQ) is selecting appropriate scaling factors to replace weight values with values from a scaled quantization grid. Existing methods typically fix the scale at the outset via heuristic tuning or grid search. In this note, we propose Beacon, a simple and effective algorithm that eliminates the need for such manual tuning. Beacon performs per-channel PTQ directly using a fixed non-scaled alphabet and automatically determines the optimal scaling factors by exploiting the geometry of symmetric scalar quantization. It supports both symmetric and asymmetric quantization with minimal modifications and does not rely on back-propagation or large calibration sets. Despite its simplicity and tuning-free nature, Beacon achieves competitive performance compared to state-of-the-art methods, making it a practical solution for efficient model deployment. 

**Abstract (ZH)**: 量化是一种广泛用于减少大规模预训练模型内存和计算成本的压缩技术。通道后训练量化(PTQ)中的一个关键挑战是选择合适的缩放因子，以用缩放量化网格中的值替换权重值。现有方法通常通过启发式调整或网格搜索在一开始就固定缩放比例。在本文中，我们提出了Beacon，一种简单而有效的算法，无需进行手动调整即可直接进行通道后训练量化。Beacon 利用对称标量量化几何特性自动确定最优的缩放因子，支持对称和非对称量化，并只需进行少量修改即可实现，不依赖于反向传播或大型校准集。尽管结构简单且无需调优，Beacon 的性能与最先进的方法相当，使其成为高效模型部署的实际解决方案。 

---
# Objective Value Change and Shape-Based Accelerated Optimization for the Neural Network Approximation 

**Title (ZH)**: 基于价值变化和形状加速优化的神经网络近似方法 

**Authors**: Pengcheng Xie, Zihao Zhou, Zijian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.20290)  

**Abstract**: This paper introduce a novel metric of an objective function f, we say VC (value change) to measure the difficulty and approximation affection when conducting an neural network approximation task, and it numerically supports characterizing the local performance and behavior of neural network approximation. Neural networks often suffer from unpredictable local performance, which can hinder their reliability in critical applications. VC addresses this issue by providing a quantifiable measure of local value changes in network behavior, offering insights into the stability and performance for achieving the neural-network approximation. We investigate some fundamental theoretical properties of VC and identified two intriguing phenomena in neural network approximation: the VC-tendency and the minority-tendency. These trends respectively characterize how pointwise errors evolve in relation to the distribution of VC during the approximation this http URL addition, we propose a novel metric based on VC, which measures the distance between two functions from the perspective of variation. Building upon this metric, we further propose a new preprocessing framework for neural network approximation. Numerical results including the real-world experiment and the PDE-related scientific problem support our discovery and pre-processing acceleration method. 

**Abstract (ZH)**: 本文引入了一种新的目标函数度量标准，称为VC（值变化），用于评估神经网络逼近任务中的难度和逼近影响，并从数值上支持刻画神经网络逼近的局部性能和行为。神经网络在局部性能上往往存在不可预测性，这可能在关键应用中影响其可靠性。VC通过提供网络行为中局部值变化的可量化度量，来洞察神经网络逼近的稳定性和性能。我们探讨了VC的一些基本理论性质，并在神经网络逼近中发现了两种有趣的趋势：VC倾向性和少数派倾向性。这些趋势分别描述了在逼近过程中点态误差如何相对于VC分布演变。此外，我们基于VC提出了一种新的基于变异距离的度量方法，并在此基础上提出了一种新的预处理框架。数值结果，包括真实世界实验和与偏微分方程相关的科学问题，支持了我们的发现和预处理加速方法。 

---
# Network-Level Prompt and Trait Leakage in Local Research Agents 

**Title (ZH)**: 网络级提示和特质泄漏在本地研究代理中 

**Authors**: Hyejun Jeong, Mohammadreze Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Badasarian  

**Link**: [PDF](https://arxiv.org/pdf/2508.20282)  

**Abstract**: We show that Web and Research Agents (WRAs) -- language model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network adversaries such as ISPs. These agents could be deployed \emph{locally} by organizations and individuals for privacy, legal, or financial purposes. Unlike sporadic web browsing by humans, WRAs visit $70{-}140$ domains with distinguishable timing correlations, enabling unique fingerprinting attacks.
Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73\% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29\%. 

**Abstract (ZH)**: 基于网络的代理（WRAs）在被动网络对手（如ISP）的推理攻击下易受攻击——这些代理可以由组织和个人为了隐私、法律或财务目的在本地部署。我们展示了仅利用WRAs的网络层面元数据（即访问的IP地址及其时间戳）就能发起一种新颖的提示和用户特征泄漏攻击。我们通过构建基于用户搜索查询和合成 persona 生成的查询的新数据集来实现这一攻击。我们定义了一个行为指标（称为OBELS），以全面评估原始提示与推断出的提示之间的相似性，结果显示我们的攻击恢复了超过73%的用户提示的功能性和领域知识。在多会话场景中，我们能够以高精度恢复多达19个潜在的用户特征。我们的攻击在网络部分可观测性和噪声条件下仍然有效。最后，我们讨论了限制领域多样性和模糊跟踪的缓解策略，这些策略在几乎不影响实用性的同时，平均减少了29%的攻击效果。 

---
# How Multimodal LLMs Solve Image Tasks: A Lens on Visual Grounding, Task Reasoning, and Answer Decoding 

**Title (ZH)**: 多模态LLMs解决图像任务：视觉 grounding、任务推理与答案解码之窗 

**Authors**: Zhuoran Yu, Yong Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.20279)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong performance across a wide range of vision-language tasks, yet their internal processing dynamics remain underexplored. In this work, we introduce a probing framework to systematically analyze how MLLMs process visual and textual inputs across layers. We train linear classifiers to predict fine-grained visual categories (e.g., dog breeds) from token embeddings extracted at each layer, using a standardized anchor question. To uncover the functional roles of different layers, we evaluate these probes under three types of controlled prompt variations: (1) lexical variants that test sensitivity to surface-level changes, (2) semantic negation variants that flip the expected answer by modifying the visual concept in the prompt, and (3) output format variants that preserve reasoning but alter the answer format. Applying our framework to LLaVA-1.5, LLaVA-Next-LLaMA-3, and Qwen2-VL, we identify a consistent stage-wise structure in which early layers perform visual grounding, middle layers support lexical integration and semantic reasoning, and final layers prepare task-specific outputs. We further show that while the overall stage-wise structure remains stable across variations in visual tokenization, instruction tuning data, and pretraining corpus, the specific layer allocation to each stage shifts notably with changes in the base LLM architecture. Our findings provide a unified perspective on the layer-wise organization of MLLMs and offer a lightweight, model-agnostic approach for analyzing multimodal representation dynamics. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在广泛的视觉-语言任务中表现出了强烈的能力，但其内部处理动态仍被探讨不足。在本文中，我们引入了一种探针框架，系统地分析MLLMs如何在不同层中处理视觉和文本输入。我们训练线性分类器，从每个层提取的词元嵌入中预测细粒度的视觉类别（例如，狗的品种），使用标准化的锚定问题。为了揭示不同层的功能作用，我们在这三种类型的控制提示变化下评估这些探针：（1）词汇变体测试对表层变化的敏感性；（2）语义否定变体通过修改提示中的视觉概念翻转预期答案；（3）输出格式变体保持推理但改变答案格式。将我们的框架应用于LLaVA-1.5、LLaVA-Next-LLaMA-3和Qwen2-VL，我们发现了一个一致的阶段式结构，在早期层中进行视觉定位，在中间层中支持词汇整合和语义推理，在最终层中准备特定任务的输出。进一步结果显示，尽管在视觉词元化、指令调优数据和预训练语料库的变化中，整体阶段式结构保持稳定，但每个阶段的特定层分配随基础大语言模型架构的变化显著变化。我们的发现提供了一种关于MLLMs逐层组织的统一视角，并提出了一种轻量级、模型无关的方法来分析多模态表示动态。 

---
# SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization 

**Title (ZH)**: SwizzlePerf: 兼顾硬件的LLMGPU内核性能优化 

**Authors**: Arya Tschand, Muhammad Awad, Ryan Swann, Kesavan Ramakrishnan, Jeffrey Ma, Keith Lowery, Ganesh Dasika, Vijay Janapa Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2508.20258)  

**Abstract**: Large language models (LLMs) have shown progress in GPU kernel performance engineering using inefficient search-based methods that optimize around runtime. Any existing approach lacks a key characteristic that human performance engineers rely on for near-optimal utilization -- hardware-awareness. By leveraging the workload's specific memory access patterns, architecture specifications, filtered profiling logs, and reflections on historical performance, we can make software-level optimizations that are tailored to the underlying hardware. SwizzlePerf automatically generates spatial optimizations for GPU kernels on disaggregated architectures by giving LLMs explicit hardware-awareness.
For a GEMM kernel, SwizzlePerf takes less than 5 minutes to generate the same hardware-specific optimal swizzling pattern that took expert performance engineers 2 weeks to find. On a suite of 10 diverse ML and Science kernels, SwizzlePerf can generate swizzling patterns for 9 of the kernels that achieve up to a 2.06x speedup and 70% improvement in L2 hit rate. This work is the first of many steps toward systematically creating hardware-aware LLM performance engineering agents. 

**Abstract (ZH)**: 大型语言模型（LLMs）在使用基于搜索的不高效方法进行GPU内核性能工程方面取得了进展，这些方法侧重优化运行时性能。现有的任何方法都不具备人类性能工程师实现接近最优利用的关键特性——硬件意识。通过利用工作负载的特定内存访问模式、架构规范、过滤后的性能日志以及历史性能的反思，我们可以在软件层面进行针对底层硬件的定制化优化。SwizzlePerf自动为分拆架构生成GPU内核的空间优化，通过赋予LLMs显式的硬件意识来实现这一目标。

对于一个GEMM内核，SwizzlePerf在不到5分钟内生成了专家性能工程师花费2周才找到的硬件特定最优交织模式。在一系列10个不同的机器学习和科学内核中，SwizzlePerf能够为9个内核生成高达2.06倍速度提升和70%的L2命中率提升的交织模式。这项工作是系统创建硬件意识的LLM性能工程代理的第一步。 

---
# MedNet-PVS: A MedNeXt-Based Deep Learning Model for Automated Segmentation of Perivascular Spaces 

**Title (ZH)**: MedNet-PVS: 一种基于MedNeX的深度学习模型，用于 PERIVASCULAR SPACES 的自动分割 

**Authors**: Zhen Xuen Brandon Low, Rory Zhang, Hang Min, William Pham, Lucy Vivash, Jasmine Moses, Miranda Lynch, Karina Dorfman, Cassandra Marotta, Shaun Koh, Jacob Bunyamin, Ella Rowsthorn, Alex Jarema, Himashi Peiris, Zhaolin Chen, Sandy R. Shultz, David K. Wright, Dexiao Kong, Sharon L. Naismith, Terence J. O'Brien, Ying Xia, Meng Law, Benjamin Sinclair  

**Link**: [PDF](https://arxiv.org/pdf/2508.20256)  

**Abstract**: Enlarged perivascular spaces (PVS) are increasingly recognized as biomarkers of cerebral small vessel disease, Alzheimer's disease, stroke, and aging-related neurodegeneration. However, manual segmentation of PVS is time-consuming and subject to moderate inter-rater reliability, while existing automated deep learning models have moderate performance and typically fail to generalize across diverse clinical and research MRI datasets. We adapted MedNeXt-L-k5, a Transformer-inspired 3D encoder-decoder convolutional network, for automated PVS segmentation. Two models were trained: one using a homogeneous dataset of 200 T2-weighted (T2w) MRI scans from the Human Connectome Project-Aging (HCP-Aging) dataset and another using 40 heterogeneous T1-weighted (T1w) MRI volumes from seven studies across six scanners. Model performance was evaluated using internal 5-fold cross validation (5FCV) and leave-one-site-out cross validation (LOSOCV). MedNeXt-L-k5 models trained on the T2w images of the HCP-Aging dataset achieved voxel-level Dice scores of 0.88+/-0.06 (white matter, WM), comparable to the reported inter-rater reliability of that dataset, and the highest yet reported in the literature. The same models trained on the T1w images of the HCP-Aging dataset achieved a substantially lower Dice score of 0.58+/-0.09 (WM). Under LOSOCV, the model had voxel-level Dice scores of 0.38+/-0.16 (WM) and 0.35+/-0.12 (BG), and cluster-level Dice scores of 0.61+/-0.19 (WM) and 0.62+/-0.21 (BG). MedNeXt-L-k5 provides an efficient solution for automated PVS segmentation across diverse T1w and T2w MRI datasets. MedNeXt-L-k5 did not outperform the nnU-Net, indicating that the attention-based mechanisms present in transformer-inspired models to provide global context are not required for high accuracy in PVS segmentation. 

**Abstract (ZH)**: 扩大血管周间隙(PVS)日益被认为是脑小血管疾病、阿尔茨海默病、中风和年龄相关神经退行性变的生物标志物。然而，手动分段PVS耗时且 intra-rater 可靠性中等，而现有自动化深度学习模型的性能一般，并且通常无法在多样化的临床和研究MRI数据集中泛化。我们改编了受Transformer启发的3D编码器-解码器卷积网络MedNeXt-L-k5，用于自动PVS分段。我们训练了两个模型：一个使用来自Human Connectome Project-Aging (HCP-Aging) 数据集的200张T2加权(T2w) MRI扫描图像，另一个使用来自六台扫描器的七个研究中40张异质的T1加权(T1w) MRI体积。模型性能通过内部5折交叉验证(5FCV)和留一站点在外交叉验证(LOSOCV)进行评估。使用HCP-Aging数据集中T2w图像训练的MedNeXt-L-k5模型在灰质(WM)上的体素级别Dice分数为0.88±0.06，与该数据集报告的intra-rater可靠性相当，且在文献中首次达到最高水平。使用HCP-Aging数据集中T1w图像训练的相同模型在灰质(WM)上的体素级别Dice分数显著降低至0.58±0.09。在 LOSOCV 下，模型在灰质(WM)上的体素级别Dice分数为0.38±0.16，在基底节(BG)上的体素级别Dice分数为0.35±0.12；而在基底节(BG)上的聚类级别Dice分数为0.61±0.19，在灰质(WM)上的聚类级别Dice分数为0.62±0.21。MedNeXt-L-k5为多样化的T1w和T2w MRI数据集提供了高效解决方案。MedNeXt-L-k5并未超越nnU-Net，表明变压器启发模型中的基于注意力的机制对于提供全局上下文并非实现PVS分段高准确性的必要条件。 

---
# The Mathematician's Assistant: Integrating AI into Research Practice 

**Title (ZH)**: 数学家的助手：将AI融入研究实践 

**Authors**: Jonas Henkel  

**Link**: [PDF](https://arxiv.org/pdf/2508.20236)  

**Abstract**: The rapid development of artificial intelligence (AI), marked by breakthroughs like 'AlphaEvolve' and 'Gemini Deep Think', is beginning to offer powerful new tools that have the potential to significantly alter the research practice in many areas of mathematics. This paper explores the current landscape of publicly accessible large language models (LLMs) in a mathematical research context, based on developments up to August 2, 2025. Our analysis of recent benchmarks, such as MathArena and the Open Proof Corpus (Balunović et al., 2025; Dekoninck et al., 2025), reveals a complex duality: while state-of-the-art models demonstrate strong abilities in solving problems and evaluating proofs, they also exhibit systematic flaws, including a lack of self-critique and a model depending discrepancy between final-answer accuracy and full-proof validity.
Based on these findings, we propose a durable framework for integrating AI into the research workflow, centered on the principle of the augmented mathematician. In this model, the AI functions as a copilot under the critical guidance of the human researcher, an approach distilled into five guiding principles for effective and responsible use. We then systematically explore seven fundamental ways AI can be applied across the research lifecycle, from creativity and ideation to the final writing process, demonstrating how these principles translate into concrete practice.
We conclude that the primary role of AI is currently augmentation rather than automation. This requires a new skill set focused on strategic prompting, critical verification, and methodological rigor in order to effectively use these powerful tools. 

**Abstract (ZH)**: 人工智能的快速发展，如“AlphaEvolve”和“Gemini Deep Think”等突破，正开始提供强大的新工具，有望在许多数学领域显著改变研究实践。本文基于截至2025年8月2日的最新发展，探讨了数学研究背景下公开可访问的大语言模型的现状。我们的分析揭示了复杂的双重性：尽管最先进的模型在解决问题和评估证明方面表现出强大的能力，但它们也表现出系统性的缺陷，包括缺乏自我批判和最终答案准确性与完整证明有效性之间的模型依赖差异。基于这些发现，我们提出了一种持久框架，将AI集成到研究工作流程中，以增强数学家为核心原则。在这种模型中，AI在人类研究者的关键指导下作为联合飞行员发挥作用，并提炼出五项基本原则以实现有效和负责任的使用。接着，我们系统地探讨了AI在研究生命周期中的七大基本应用方式，从创造力和创意产生到最终的写作过程，展示了这些原则如何转化为具体实践。我们得出结论，当前AI的主要作用是增强而非自动化。这需要一种新的技能组合，集中在战略性提示、批判性验证和方法论严谨性上，以有效地使用这些强大的工具。 

---
# Validating Generative Agent-Based Models for Logistics and Supply Chain Management Research 

**Title (ZH)**: 验证生成性基于代理的模型在物流与供应链管理研究中的有效性 

**Authors**: Vincent E. Castillo  

**Link**: [PDF](https://arxiv.org/pdf/2508.20234)  

**Abstract**: Generative Agent-Based Models (GABMs) powered by large language models (LLMs) offer promising potential for empirical logistics and supply chain management (LSCM) research by enabling realistic simulation of complex human behaviors. Unlike traditional agent-based models, GABMs generate human-like responses through natural language reasoning, which creates potential for new perspectives on emergent LSCM phenomena. However, the validity of LLMs as proxies for human behavior in LSCM simulations is unknown. This study evaluates LLM equivalence of human behavior through a controlled experiment examining dyadic customer-worker engagements in food delivery scenarios. I test six state-of-the-art LLMs against 957 human participants (477 dyads) using a moderated mediation design. This study reveals a need to validate GABMs on two levels: (1) human equivalence testing, and (2) decision process validation. Results reveal GABMs can effectively simulate human behaviors in LSCM; however, an equivalence-versus-process paradox emerges. While a series of Two One-Sided Tests (TOST) for equivalence reveals some LLMs demonstrate surface-level equivalence to humans, structural equation modeling (SEM) reveals artificial decision processes not present in human participants for some LLMs. These findings show GABMs as a potentially viable methodological instrument in LSCM with proper validation checks. The dual-validation framework also provides LSCM researchers with a guide to rigorous GABM development. For practitioners, this study offers evidence-based assessment for LLM selection for operational tasks. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的生成型基于代理模型（GABMs）为物流和供应链管理（LSCM）研究提供了有前景的潜力，通过实现复杂人类行为的现实模拟。与传统基于代理模型不同，GABMs通过自然语言推理生成类似人类的响应，这为新兴LSCM现象提供了新的视角。然而，LLMs在LSCM仿真中作为人类行为代理的有效性尚不明晰。本研究通过一个受控实验评估LLMs在食品配送场景中双边客户-工人交互中的等效性，使用调节中介设计测试了六种最先进的LLMs，共有477个双边人类参与者。研究结果揭示了GABMs在LSCM中模拟人类行为的有效性，但同时出现了等效性与过程验证的悖论。虽然等价性检验（TOST）揭示了一些LLMs在表面行为上表现出等效性，但结构方程建模（SEM）揭示了一些LLMs中存在人类参与者不具备的人工决策过程。这些发现表明，在适当验证检查的前提下，GABMs可能是LSCM研究中的一个潜在可行的方法工具。双重验证框架也为LSCM研究人员提供了严格的GABM开发指南。对于从业者而言，本研究提供了基于证据的LLM选择评估，以应用于操作任务。 

---
# A Novel Framework for Automated Explain Vision Model Using Vision-Language Models 

**Title (ZH)**: 一种基于视觉-语言模型的自动化解释视觉模型的新框架 

**Authors**: Phu-Vinh Nguyen, Tan-Hanh Pham, Chris Ngo, Truong Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2508.20227)  

**Abstract**: The development of many vision models mainly focuses on improving their performance using metrics such as accuracy, IoU, and mAP, with less attention to explainability due to the complexity of applying xAI methods to provide a meaningful explanation of trained models. Although many existing xAI methods aim to explain vision models sample-by-sample, methods explaining the general behavior of vision models, which can only be captured after running on a large dataset, are still underexplored. Furthermore, understanding the behavior of vision models on general images can be very important to prevent biased judgments and help identify the model's trends and patterns. With the application of Vision-Language Models, this paper proposes a pipeline to explain vision models at both the sample and dataset levels. The proposed pipeline can be used to discover failure cases and gain insights into vision models with minimal effort, thereby integrating vision model development with xAI analysis to advance image analysis. 

**Abstract (ZH)**: 多种视觉模型的发展主要集中在使用准确率、IoU和mAP等指标提高其性能，但较少关注可解释性，这主要是因为将xAI方法应用于提供有意义的解释较为复杂。尽管存在许多旨在样本级解释视觉模型的xAI方法，但在大型数据集上运行以捕获视觉模型一般行为的方法仍然尚未得到充分利用。此外，理解视觉模型在一般图像上的行为对于防止有偏的判断并帮助识别模型的趋势和模式非常重要。在视觉语言模型的应用下，本文提出了一种管道，用于在样本级和数据集级解释视觉模型。该提出的管道可以用于发现失败案例并以最少的努力获得关于视觉模型的见解，从而将视觉模型开发与xAI分析整合起来，以促进图像分析的发展。 

---
# The Role of Teacher Calibration in Knowledge Distillation 

**Title (ZH)**: 教师校准在知识蒸馏中的作用 

**Authors**: Suyoung Kim, Seonguk Park, Junhoo Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2508.20224)  

**Abstract**: Knowledge Distillation (KD) has emerged as an effective model compression technique in deep learning, enabling the transfer of knowledge from a large teacher model to a compact student model. While KD has demonstrated significant success, it is not yet fully understood which factors contribute to improving the student's performance. In this paper, we reveal a strong correlation between the teacher's calibration error and the student's accuracy. Therefore, we claim that the calibration of the teacher model is an important factor for effective KD. Furthermore, we demonstrate that the performance of KD can be improved by simply employing a calibration method that reduces the teacher's calibration error. Our algorithm is versatile, demonstrating effectiveness across various tasks from classification to detection. Moreover, it can be easily integrated with existing state-of-the-art methods, consistently achieving superior performance. 

**Abstract (ZH)**: 知识蒸馏（KD）作为一种有效的模型压缩技术，在深度学习中已得到广泛应用，能够从大型教师模型转移知识到紧凑的学生模型。尽管KD已经取得了显著的成功，但尚未完全理解哪些因素能够改善学生模型的性能。在本文中，我们揭示了教师模型的校准误差与学生模型的准确率之间存在密切的相关性。因此，我们认为教师模型的校准是有效知识蒸馏的重要因素。此外，我们证明通过简单地采用一种减少教师模型校准误差的校准方法，可以提高知识蒸馏的性能。我们的算法具有通用性，能够在从分类到检测的各种任务中有效地提升性能，并且可以很容易地与现有的最先进技术集成，始终实现优越的表现。 

---
# Prompting Strategies for Language Model-Based Item Generation in K-12 Education: Bridging the Gap Between Small and Large Language Models 

**Title (ZH)**: 基于语言模型的K-12教育试题生成中提示策略：缩小小型与大型语言模型之间的差距 

**Authors**: Mohammad Amini, Babak Ahmadi, Xiaomeng Xiong, Yilin Zhang, Christopher Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20217)  

**Abstract**: This study explores automatic generation (AIG) using language models to create multiple choice questions (MCQs) for morphological assessment, aiming to reduce the cost and inconsistency of manual test development. The study used a two-fold approach. First, we compared a fine-tuned medium model (Gemma, 2B) with a larger untuned one (GPT-3.5, 175B). Second, we evaluated seven structured prompting strategies, including zero-shot, few-shot, chain-of-thought, role-based, sequential, and combinations. Generated items were assessed using automated metrics and expert scoring across five dimensions. We also used GPT-4.1, trained on expert-rated samples, to simulate human scoring at scale. Results show that structured prompting, especially strategies combining chain-of-thought and sequential design, significantly improved Gemma's outputs. Gemma generally produced more construct-aligned and instructionally appropriate items than GPT-3.5's zero-shot responses, with prompt design playing a key role in mid-size model performance. This study demonstrates that structured prompting and efficient fine-tuning can enhance midsized models for AIG under limited data conditions. We highlight the value of combining automated metrics, expert judgment, and large-model simulation to ensure alignment with assessment goals. The proposed workflow offers a practical and scalable way to develop and validate language assessment items for K-12. 

**Abstract (ZH)**: 本研究探讨了使用语言模型自动生成（AIG）形态学评估的多项选择题（MCQs），旨在减少手动测试开发的成本和不一致性。研究采用了两步方法。首先，我们将微调的中型模型（Gemma，2B）与未微调的大型模型（GPT-3.5，175B）进行了比较。其次，我们评估了七种结构化提示策略，包括零样本、少量样本、思考链、角色扮演、序列化以及这些策略的组合。生成的试题通过自动化指标和专家评分在五个维度上进行了评估。我们还使用了基于专家评分样本训练的GPT-4.1来大规模模拟人工评分。结果显示，结构化提示，尤其是结合思考链和序列化设计的策略，显著提升了Gemma的输出效果。Gemma一般比GPT-3.5的零样本响应生成了更多符合建构和教学导向的试题，提示设计对于中型模型的表现起着关键作用。本研究展示了，在数据有限条件下，结构化提示和高效微调可以增强中型模型的AIG能力。本研究强调了结合自动化指标、专家判断和大型模型模拟的价值，以确保评估目标的一致性。提出的工作流程为K-12语言评估试题的开发和验证提供了一种实用且可扩展的方法。 

---
# Collaborating with GenAI: Incentives and Replacements 

**Title (ZH)**: 与生成式人工智能协作：激励与替代 

**Authors**: Boaz Taitler, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2508.20213)  

**Abstract**: The rise of Generative AI (GenAI) is reshaping how workers contribute to shared projects. While workers can use GenAI to boost productivity or reduce effort, managers may use it to replace some workers entirely. We present a theoretical framework to analyze how GenAI affects collaboration in such settings. In our model, the manager selects a team to work on a shared task, with GenAI substituting for unselected workers. Each worker selects how much effort to exert, and incurs a cost that increases with the level of effort. We show that GenAI can lead workers to exert no effort, even if GenAI is almost ineffective. We further show that the manager's optimization problem is NP-complete, and provide an efficient algorithm for the special class of (almost-) linear instances. Our analysis shows that even workers with low individual value may play a critical role in sustaining overall output, and excluding such workers can trigger a cascade. Finally, we conduct extensive simulations to illustrate our theoretical findings. 

**Abstract (ZH)**: 生成式AI的兴起正在重塑工人在共享项目中的贡献方式。虽然工人可以利用生成式AI提高生产力或减少努力，管理者可能利用它完全替代某些工人。我们提出了一种理论框架来分析生成式AI在这些环境中如何影响协作。在我们的模型中，管理者选择一个团队来处理一个共享任务，生成式AI替代未选中的工人。每个工人选择付出多少努力，并且付出努力的水平越高，成本也越高。我们证明生成式AI可能导致工人付出零努力，即便生成式AI几乎无效。进一步证明管理者的优化问题为NP完全问题，并提供了适用于（几乎）线性实例的高效算法。我们的分析表明，即使个体价值较低的工人也可能在维持总体产出中发挥关键作用，排除这些工人可能会引发连锁反应。最后，我们进行了广泛的模拟以说明我们的理论发现。 

---
# Filter then Attend: Improving attention-based Time Series Forecasting with Spectral Filtering 

**Title (ZH)**: 滤波后再关注：基于谱滤波的注意力时序预测改进 

**Authors**: Elisha Dayag, Nhat Thanh Van Tran, Jack Xin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20206)  

**Abstract**: Transformer-based models are at the forefront in long time-series forecasting (LTSF). While in many cases, these models are able to achieve state of the art results, they suffer from a bias toward low-frequencies in the data and high computational and memory requirements. Recent work has established that learnable frequency filters can be an integral part of a deep forecasting model by enhancing the model's spectral utilization. These works choose to use a multilayer perceptron to process their filtered signals and thus do not solve the issues found with transformer-based models. In this paper, we establish that adding a filter to the beginning of transformer-based models enhances their performance in long time-series forecasting. We add learnable filters, which only add an additional $\approx 1000$ parameters to several transformer-based models and observe in multiple instances 5-10 \% relative improvement in forecasting performance. Additionally, we find that with filters added, we are able to decrease the embedding dimension of our models, resulting in transformer-based architectures that are both smaller and more effective than their non-filtering base models. We also conduct synthetic experiments to analyze how the filters enable Transformer-based models to better utilize the full spectrum for forecasting. 

**Abstract (ZH)**: 基于Transformer的模型在长时间序列预测中的前沿地位：添加滤波器提高性能研究 

---
# AI Propaganda factories with language models 

**Title (ZH)**: AI propaganda factories with language models 

**Authors**: Lukasz Olejnik  

**Link**: [PDF](https://arxiv.org/pdf/2508.20186)  

**Abstract**: AI-powered influence operations can now be executed end-to-end on commodity hardware. We show that small language models produce coherent, persona-driven political messaging and can be evaluated automatically without human raters. Two behavioural findings emerge. First, persona-over-model: persona design explains behaviour more than model identity. Second, engagement as a stressor: when replies must counter-arguments, ideological adherence strengthens and the prevalence of extreme content increases. We demonstrate that fully automated influence-content production is within reach of both large and small actors. Consequently, defence should shift from restricting model access towards conversation-centric detection and disruption of campaigns and coordination infrastructure. Paradoxically, the very consistency that enables these operations also provides a detection signature. 

**Abstract (ZH)**: 基于AI的动力操作现在可以在商用硬件上端到端执行。我们展示，小型语言模型能够生成连贯的、以角色为中心的政治信息，并且可以在没有人类评阅者的情况下自动评估。研究发现了两种行为现象：首先，角色超越模型：角色设计比模型身份更能解释行为。其次，互动作为压力源：当回复必须对抗论点时，意识形态坚持会加强，极端内容的出现频率增加。我们证明，完全自动化的影响力内容生产对于大范围的参与者而言是可及的。因此，防御策略应从限制模型访问转向以对话为中心的活动检测与中断，破坏 campaigns 和协调基础设施。矛盾的是，正是这些操作的这种一致性提供了检测的特征。 

---
# Mitigating Hallucinations in Multimodal LLMs via Object-aware Preference Optimization 

**Title (ZH)**: 通过对象感知的偏好优化减轻多模态LLM的幻觉现象 

**Authors**: Alberto Compagnoni, Davide Caffagni, Nicholas Moratelli, Lorenzo Baraldi, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2508.20181)  

**Abstract**: Multimodal Large Language Models (MLLMs) emerge as a unified interface to address a multitude of tasks, ranging from NLP to computer vision. Despite showcasing state-of-the-art results in many benchmarks, a long-standing issue is the tendency of MLLMs to hallucinate, that is to generate answers to the user's query that are not reflected in the visual input. In this paper, we address the problem of hallucinations as an alignment problem, seeking to steer the MLLM so that it prefers generating content without hallucinations. In contrast to recent approaches that require complicated pipelines to build synthetic preference data for alignment training, often relying on proprietary models, we capitalize on the well-known CHAIR metric, originally proposed to gauge the degree of hallucinations in image captioning. Given a pair of generated answers, we leverage CHAIR to distinguish winner and loser options (i.e., non-hallucinated and hallucinated samples) and fine-tune off-the-shelf MLLMs via Direct Preference Optimization (DPO). The resulting method, which we refer to as CHAIR-DPO, effectively diminishes the amount of hallucinated answers on several hallucination benchmarks, demonstrating the effectiveness of fine-tuning the MLLM with a CHAIR-based reward. Source code and trained models are publicly available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）作为统一接口解决从自然语言处理到计算机视觉等多种任务。尽管在许多基准测试中展示了最先进的结果，但长期存在的问题是MLLMs倾向于幻觉，即生成与视觉输入不符的答案。在本文中，我们将幻觉问题视为对齐问题，旨在引导MLLM生成无幻觉的内容。与需要复杂流程构建合成偏好数据进行对齐训练、通常依赖专有模型的 Recent 方法不同，我们利用最初用于评估图像描述中幻觉程度的 CHAIR 指标。通过一对生成的答案，我们利用 CHAIR 来区分优胜和劣败选项（即无幻觉和幻觉样本），并通过直接偏好优化（DPO）微调现成的 MLLM。我们称之为 CHAIR-DPO 的方法有效地减少了多个幻觉基准上的幻觉答案数量，证明了使用基于CHAIR的奖励微调MLLM的有效性。相关源代码和训练模型可从此链接访问。 

---
# RelAItionship Building: Analyzing Recruitment Strategies for Participatory AI 

**Title (ZH)**: 关系构建：分析参与式AI的招聘策略 

**Authors**: Eugene Kim, Vaibhav Balloli, Berelian Karimian, Elizabeth Bondi-Kelly, Benjamin Fish  

**Link**: [PDF](https://arxiv.org/pdf/2508.20176)  

**Abstract**: Participatory AI, in which impacted community members and other stakeholders are involved in the design and development of AI systems, holds promise as a way to ensure AI is developed to meet their needs and reflect their values. However, the process of identifying, reaching out, and engaging with all relevant stakeholder groups, which we refer to as recruitment methodology, is still a practical challenge in AI projects striving to adopt participatory practices. In this paper, we investigate the challenges that researchers face when designing and executing recruitment methodology for Participatory AI projects, and the implications of current recruitment practice for Participatory AI. First, we describe the recruitment methodologies used in AI projects using a corpus of 37 projects to capture the diversity of practices in the field and perform an initial analysis on the documentation of recruitment practices, as well as specific strategies that researchers use to meet goals of equity and empowerment. To complement this analysis, we interview five AI researchers to learn about the outcomes of recruitment methodologies. We find that these outcomes are shaped by structural conditions of their work, researchers' own goals and expectations, and the relationships built from the recruitment methodology and subsequent collaboration. Based on these analyses, we provide recommendations for designing and executing relationship-forward recruitment methods, as well as reflexive recruitment documentation practices for Participatory AI researchers. 

**Abstract (ZH)**: 参与式AI中的招募能力研究：一种确保AI系统符合用户需求并反映其价值观的方法，但如何识别、联系并有效参与所有相关利益相关者群体仍是一个实践挑战。本文探讨了研究人员在设计和执行参与式AI项目招募能力方法时遇到的挑战及其现有的招募能力实践对参与式AI的含义，并提供了招募能力方法和反思性招募能力文档实践的建议。 

---
# Navigating the EU AI Act: Foreseeable Challenges in Qualifying Deep Learning-Based Automated Inspections of Class III Medical Devices 

**Title (ZH)**: 欧盟AI法案中基于深度学习的III类医疗器械自动检查的可预见挑战导航 

**Authors**: Julio Zanon Diaz, Tommy Brennan, Peter Corcoran  

**Link**: [PDF](https://arxiv.org/pdf/2508.20144)  

**Abstract**: As deep learning (DL) technologies advance, their application in automated visual inspection for Class III medical devices offers significant potential to enhance quality assurance and reduce human error. However, the adoption of such AI-based systems introduces new regulatory complexities--particularly under the EU Artificial Intelligence (AI) Act, which imposes high-risk system obligations that differ in scope and depth from established regulatory frameworks such as the Medical Device Regulation (MDR) and the U.S. FDA Quality System Regulation (QSR). This paper presents a high-level technical assessment of the foresee-able challenges that manufacturers are likely to encounter when qualifying DL-based automated inspections within the existing medical device compliance landscape. It examines divergences in risk management principles, dataset governance, model validation, explainability requirements, and post-deployment monitoring obligations. The discussion also explores potential implementation strategies and highlights areas of uncertainty, including data retention burdens, global compliance implications, and the practical difficulties of achieving statistical significance in validation with limited defect data. Disclaimer: This publication is in-tended solely as an academic and technical evaluation. It is not a substitute for le-gal advice or official regulatory interpretation. The information presented here should not be relied upon to demonstrate compliance with the EU AI Act or any other statutory obligation. Manufacturers are encouraged to consult appropriate regulatory authorities and legal experts to determine specific compliance pathways. 

**Abstract (ZH)**: 深学习技术在III类医疗设备自动化视觉检测中的应用：现有合规landscape下的挑战与监管复杂性分析 

---
# UltraEar: a multicentric, large-scale database combining ultra-high-resolution computed tomography and clinical data for ear diseases 

**Title (ZH)**: 耳部超高清计算机断层扫描及临床数据多中心大規模数据库：UltraEar 

**Authors**: Ruowei Tang, Pengfei Zhao, Xiaoguang Li, Ning Xu, Yue Cheng, Mengshi Zhang, Zhixiang Wang, Zhengyu Zhang, Hongxia Yin, Heyu Ding, Shusheng Gong, Yuhe Liu, Zhenchang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20141)  

**Abstract**: Ear diseases affect billions of people worldwide, leading to substantial health and socioeconomic burdens. Computed tomography (CT) plays a pivotal role in accurate diagnosis, treatment planning, and outcome evaluation. The objective of this study is to present the establishment and design of UltraEar Database, a large-scale, multicentric repository of isotropic 0.1 mm ultra-high-resolution CT (U-HRCT) images and associated clinical data dedicated to ear diseases. UltraEar recruits patients from 11 tertiary hospitals between October 2020 and October 2035, integrating U-HRCT images, structured CT reports, and comprehensive clinical information, including demographics, audiometric profiles, surgical records, and pathological findings. A broad spectrum of otologic disorders is covered, such as otitis media, cholesteatoma, ossicular chain malformation, temporal bone fracture, inner ear malformation, cochlear aperture stenosis, enlarged vestibular aqueduct, and sigmoid sinus bony deficiency. Standardized preprocessing pipelines have been developed for geometric calibration, image annotation, and multi-structure segmentation. All personal identifiers in DICOM headers and metadata are removed or anonymized to ensure compliance with data privacy regulation. Data collection and curation are coordinated through monthly expert panel meetings, with secure storage on an offline cloud system. UltraEar provides an unprecedented ultra-high-resolution reference atlas with both technical fidelity and clinical relevance. This resource has significant potential to advance radiological research, enable development and validation of AI algorithms, serve as an educational tool for training in otologic imaging, and support multi-institutional collaborative studies. UltraEar will be continuously updated and expanded, ensuring long-term accessibility and usability for the global otologic research community. 

**Abstract (ZH)**: 耳部疾病影响全球数十亿人，导致严重的健康和社会经济负担。计算机断层扫描（CT）在准确诊断、治疗规划和预后评估中发挥着关键作用。本研究的目的是介绍并设计UltraEar数据库，这是一个大规模、多中心的耳部疾病超高清CT（U-HRCT）图像及其相关临床数据的存储库。UltraEar从2020年10月至2035年10月招募来自11家三级医院的患者，整合超高清CT图像、结构化CT报告以及包括人口统计学、听力图谱、手术记录和病理学发现在内的综合临床信息。涵盖了广泛的耳科疾病，如中耳炎、胆固醇瘤、听小骨链畸形、颞骨骨折、内耳畸形、耳蜗孔狭窄、前庭水管扩张和水平半恒骨头缺乏。已开发了标准化的预处理管道，用于几何校准、图像注释和多结构分割。所有DICOM头和元数据中的个人标识符已被删除或匿名化，以确保符合数据隐私法规。数据收集和管理通过每月专家小组会议协调，并在离线云系统中安全存储。UltraEar提供了一个具有技术和临床相关性的前所未有的超高清参考图谱。该资源具有显著潜力，可促进放射学研究、开发和验证人工智能算法，作为耳科成像培训的教育工具，并支持多机构协作研究。UltraEar将继续更新和扩展，确保全球耳科研究社区的长期可访问性和可用性。 

---
# Data-Efficient Point Cloud Semantic Segmentation Pipeline for Unimproved Roads 

**Title (ZH)**: 未经改善道路的数据高效点云语义分割管道 

**Authors**: Andrew Yarovoi, Christopher R. Valenta  

**Link**: [PDF](https://arxiv.org/pdf/2508.20135)  

**Abstract**: In this case study, we present a data-efficient point cloud segmentation pipeline and training framework for robust segmentation of unimproved roads and seven other classes. Our method employs a two-stage training framework: first, a projection-based convolutional neural network is pre-trained on a mixture of public urban datasets and a small, curated in-domain dataset; then, a lightweight prediction head is fine-tuned exclusively on in-domain data. Along the way, we explore the application of Point Prompt Training to batch normalization layers and the effects of Manifold Mixup as a regularizer within our pipeline. We also explore the effects of incorporating histogram-normalized ambients to further boost performance. Using only 50 labeled point clouds from our target domain, we show that our proposed training approach improves mean Intersection-over-Union from 33.5% to 51.8% and the overall accuracy from 85.5% to 90.8%, when compared to naive training on the in-domain data. Crucially, our results demonstrate that pre-training across multiple datasets is key to improving generalization and enabling robust segmentation under limited in-domain supervision. Overall, this study demonstrates a practical framework for robust 3D semantic segmentation in challenging, low-data scenarios. Our code is available at: this https URL. 

**Abstract (ZH)**: 在这种案例研究中，我们提出了一种高效的数据点云分割管道和训练框架，用于鲁棒地分割未经改善的道路和其他七个类别。我们的方法采用了两阶段的训练框架：首先，在公共城市数据集和一个小规模的领域特定数据集的混合数据上预训练基于投影的卷积神经网络；然后，仅在领域特定数据上微调轻量级预测头部。在这一过程中，我们探讨了点提示训练在批量归一化层中的应用以及Manifold Mixup作为正则化器的效果。我们还研究了结合直方图归一化环境光以提高性能的影响。仅使用目标领域中的50个标注点云，我们展示了我们提出的训练方法将平均交并比从33.5%提高到51.8%，并将总体准确率从85.5%提高到90.8%，这与仅在领域内部数据上进行朴素训练的效果相比。关键的是，我们的结果证明了跨多个数据集预训练对于提高泛化能力和在有限领域监督下实现鲁棒分割的重要性。总体而言，该项研究展示了在具有挑战性的低数据场景中实现鲁棒3D语义分割的实用框架。我们的代码可在：this https URL获取。 

---
# Artificial Intelligence for CRISPR Guide RNA Design: Explainable Models and Off-Target Safety 

**Title (ZH)**: 人工智能在CRISPR导向RNA设计中的应用：可解释模型与脱靶安全性 

**Authors**: Alireza Abbaszadeh, Armita Shahlai  

**Link**: [PDF](https://arxiv.org/pdf/2508.20130)  

**Abstract**: CRISPR-based genome editing has revolutionized biotechnology, yet optimizing guide RNA (gRNA) design for efficiency and safety remains a critical challenge. Recent advances (2020--2025, updated to reflect current year if needed) demonstrate that artificial intelligence (AI), especially deep learning, can markedly improve the prediction of gRNA on-target activity and identify off-target risks. In parallel, emerging explainable AI (XAI) techniques are beginning to illuminate the black-box nature of these models, offering insights into sequence features and genomic contexts that drive Cas enzyme performance. Here we review how state-of-the-art machine learning models are enhancing gRNA design for CRISPR systems, highlight strategies for interpreting model predictions, and discuss new developments in off-target prediction and safety assessment. We emphasize breakthroughs from top-tier journals that underscore an interdisciplinary convergence of AI and genome editing to enable more efficient, specific, and clinically viable CRISPR applications. 

**Abstract (ZH)**: 基于CRISPR的基因编辑技术 telah革命性地改变了生物技术领域，然而，如何优化导向RNA（gRNA）的设计以提高效率和安全性仍是一项关键挑战。近年来（2020-2025年，根据需要更新至当前年份），研究表明，特别是深度学习等人工智能（AI）技术能够显著改善gRNA靶向活性的预测，并识别潜在的脱靶风险。与此同时，新兴的可解释AI（XAI）技术开始揭示这些模型的黑箱性质，提供关于驱动Cas酶性能的序列特征和基因组背景的见解。本文综述了最先进的机器学习模型如何增强CRISPR系统中的gRNA设计，强调了解析模型预测策略，并讨论了脱靶预测和安全性评估的新进展。我们强调了顶级期刊上的突破，突显了人工智能与基因编辑的跨学科融合，以实现更高效、更特异且临床可行的CRISPR应用。 

---
# Improving Liver Disease Diagnosis with SNNDeep: A Custom Spiking Neural Network Using Diverse Learning Algorithms 

**Title (ZH)**: 基于 diverse 学习算法的定制化脉冲神经网络 SNNDeep 以提高肝脏疾病诊断准确性 

**Authors**: Zofia Rudnicka, Janusz Szczepanski, Agnieszka Pregowska  

**Link**: [PDF](https://arxiv.org/pdf/2508.20125)  

**Abstract**: Purpose: Spiking neural networks (SNNs) have recently gained attention as energy-efficient, biologically plausible alternatives to conventional deep learning models. Their application in high-stakes biomedical imaging remains almost entirely unexplored. Methods: This study introduces SNNDeep, the first tailored SNN specifically optimized for binary classification of liver health status from computed tomography (CT) features. To ensure clinical relevance and broad generalizability, the model was developed and evaluated using the Task03\Liver dataset from the Medical Segmentation Decathlon (MSD), a standardized benchmark widely used for assessing performance across diverse medical imaging tasks. We benchmark three fundamentally different learning algorithms, namely Surrogate Gradient Learning, the Tempotron rule, and Bio-Inspired Active Learning across three architectural variants: a fully customized low-level model built from scratch, and two implementations using leading SNN frameworks, i.e., snnTorch and SpikingJelly. Hyperparameter optimization was performed using Optuna. Results: Our results demonstrate that the custom-built SNNDeep consistently outperforms framework-based implementations, achieving a maximum validation accuracy of 98.35%, superior adaptability across learning rules, and significantly reduced training overhead. Conclusion:This study provides the first empirical evidence that low-level, highly tunable SNNs can surpass standard frameworks in medical imaging, especially in data-limited, temporally constrained diagnostic settings, thereby opening a new pathway for neuro-inspired AI in precision medicine. 

**Abstract (ZH)**: 目的：脉冲神经网络（SNNs）最近因其在能量效率和生物可塑性方面的优势，被视为传统深度学习模型的有潜力替代方案。它们在高风险生物医学成像中的应用仍然几乎未被探索。方法：本文介绍了SNNDeep，这是第一个专门为基于计算机断层扫描（CT）特征进行肝健康状态二分类设计的定制化SNN，并且优化了这种SNN。为了确保临床相关性及广泛的普适性，该模型基于Medical Segmentation Decathlon（MSD）的任务03\肝脏数据集开发和评估，MSD是一个广泛用于评估多样医学成像任务性能的标准化基准。三种基本不同的学习算法——替代梯度学习、Tempotron规则和生物启发式主动学习——分别在三种架构变体上进行了测试：完全从零开始建立的低级模型，以及使用领先的SNN框架snnTorch和SpikingJelly实现的两种版本。超参数优化使用了Optuna。结果：结果表明，自定义构建的SNNDeep在验证准确性上持续超过基于框架的实现，达到了98.35%的最大验证准确性，并且具有更强的学习规则适应性和显著降低的训练开销。结论：本研究提供了低级、高度可调SNN在医学成像中可以超越标准框架的首个实证证据，特别是在数据有限、时间受限的诊断环境中，从而为精确诊断中的神经启发式AI开辟了一条新途径。 

---
# Towards Better Correctness and Efficiency in Code Generation 

**Title (ZH)**: 向更好的正确性和效率迈进：代码生成的角度 

**Authors**: Yunlong Feng, Yang Xu, Xiao Xu, Binyuan Hui, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20124)  

**Abstract**: While code large language models have demonstrated remarkable progress in code generation, the generated code often exhibits poor runtime efficiency, limiting its practical application in performance-sensitive scenarios. To address this limitation, we propose an efficiency-oriented reinforcement learning framework guided by a novel performance reward. Based on this framework, we take a deeper dive into the code efficiency problem, identifying then proposing methods to overcome key bottlenecks: (1) Dynamic exploration overcomes the static data constraints of offline fine-tuning, enabling the discovery of more efficient code implementations. (2) The error-insensitive reinforcement learning method and high-contrast efficiency signals are crucial for mitigating systematic errors and achieving effective optimization. (3) Online exploration is most effective when starting from a high-correctness baseline, as this allows for efficiency improvements without sacrificing accuracy. With these discoveries, we finally propose a two-stage tuning method, which achieves high and balanced performance across correctness and efficiency. The results of experiments show the effectiveness of the method, which improves code correctness by 10.18\% and runtime efficiency by 7.75\% on a 7B model, achieving performance comparable to much larger model. 

**Abstract (ZH)**: 尽管代码大型语言模型在代码生成方面取得了显著进展，生成的代码通常运行时效率较差，限制了其在性能敏感场景中的实际应用。为解决这一局限性，我们提出了一种以效率为导向的强化学习框架，并由一种新颖的性能奖励指导。基于此框架，我们深入探讨了代码效率问题，识别并提出了解决关键瓶颈的方法：（1）动态探索克服了离线微调的静态数据限制， enables the discovery of more efficient code implementations.（2）对错误不敏感的强化学习方法和高对比度的效率信号对于减轻系统误差并实现有效的优化至关重要。（3）从高正确性基线开始的在线探索最有效，这允许在不牺牲准确性的前提下提高效率。通过这些发现，我们最终提出了一种两阶段调优方法，该方法在正确性和效率上都实现了高效且平衡的性能。实验结果表明，该方法的有效性，相较7B模型，代码正确性提高了10.18\%，运行时效率提高了7.75\%，并将性能提升至接近更大模型的水平。 

---
# Particle swarm optimization for online sparse streaming feature selection under uncertainty 

**Title (ZH)**: 基于不确定性的在线稀疏流特征选择的粒子群优化方法 

**Authors**: Ruiyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20123)  

**Abstract**: In real-world applications involving high-dimensional streaming data, online streaming feature selection (OSFS) is widely adopted. Yet, practical deployments frequently face data incompleteness due to sensor failures or technical constraints. While online sparse streaming feature selection (OS2FS) mitigates this issue via latent factor analysis-based imputation, existing methods struggle with uncertain feature-label correlations, leading to inflexible models and degraded performance. To address these gaps, this work proposes POS2FS-an uncertainty-aware online sparse streaming feature selection framework enhanced by particle swarm optimization (PSO). The approach introduces: 1) PSO-driven supervision to reduce uncertainty in feature-label relationships; 2) Three-way decision theory to manage feature fuzziness in supervised learning. Rigorous testing on six real-world datasets confirms POS2FS outperforms conventional OSFS and OS2FS techniques, delivering higher accuracy through more robust feature subset selection. 

**Abstract (ZH)**: 面向高维流数据的不确定性感知在线稀疏流特征选择（POS2FS）框架 

---
# Is Artificial Intelligence Reshaping the Landscape of the International Academic Community of Geosciences? 

**Title (ZH)**: 人工智能正在重塑地球科学国际学术社区的格局吗？ 

**Authors**: Liang Li, Yuntian Li, Wenxin Zhao, Shan Ye, Yun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20117)  

**Abstract**: Through bibliometric analysis and topic modeling, we find that artificial intelligence (AI) is positively transforming geosciences research, with a notable increase in AI-related scientific output in recent years. We are encouraged to observe that earth scientists from developing countries have gained better visibility in the recent AI for Science (AI4S) paradigm and that AI is also improving the landscape of international collaboration in geoscience-related research. 

**Abstract (ZH)**: 通过文献计量分析和主题建模，我们发现人工智能（AI）正积极地变革地质科学研究，近年来与AI相关的科学产出显著增加。我们注意到，来自发展中国家的地球科学家在AI for Science (AI4S)范式中获得了更好的能见度，同时人工智能也改善了地质科学研究领域的国际合作格局。 

---
# Flexible metadata harvesting for ecology using large language models 

**Title (ZH)**: 使用大型语言模型进行生态学的灵活元数据收割 

**Authors**: Zehao Lu, Thijs L van der Plas, Parinaz Rashidi, W Daniel Kissling, Ioannis N Athanasiadis  

**Link**: [PDF](https://arxiv.org/pdf/2508.20115)  

**Abstract**: Large, open datasets can accelerate ecological research, particularly by enabling researchers to develop new insights by reusing datasets from multiple sources. However, to find the most suitable datasets to combine and integrate, researchers must navigate diverse ecological and environmental data provider platforms with varying metadata availability and standards. To overcome this obstacle, we have developed a large language model (LLM)-based metadata harvester that flexibly extracts metadata from any dataset's landing page, and converts these to a user-defined, unified format using existing metadata standards. We validate that our tool is able to extract both structured and unstructured metadata with equal accuracy, aided by our LLM post-processing protocol. Furthermore, we utilise LLMs to identify links between datasets, both by calculating embedding similarity and by unifying the formats of extracted metadata to enable rule-based processing. Our tool, which flexibly links the metadata of different datasets, can therefore be used for ontology creation or graph-based queries, for example, to find relevant ecological and environmental datasets in a virtual research environment. 

**Abstract (ZH)**: 大规模开放数据集可以加速生态研究，特别是在通过从多个来源重新使用数据集来开发新的见解方面。然而，为了找到最适合组合和集成的数据集，研究人员必须导航不同的生态和环境数据提供商平台，这些平台的元数据可用性和标准各不相同。为了克服这一障碍，我们开发了一种基于大规模语言模型（LLM）的元数据收割器，它可以灵活地从任何数据集的landing page提取元数据，并使用现有的元数据标准将这些转换为用户定义的统一格式。我们验证我们的工具能够以相同的准确性提取结构化和非结构化元数据，并利用我们的LLM后处理协议加以辅助。此外，我们使用LLM来识别数据集之间的链接，通过计算嵌入相似性和统一提取的元数据格式以实现基于规则的处理。因此，该工具可以灵活地链接不同数据集的元数据，例如用于本体创建或图查询，以在虚拟研究环境中查找相关的生态和环境数据集。 

---
# Deep Reinforcement Learning for Optimal Asset Allocation Using DDPG with TiDE 

**Title (ZH)**: 基于TiDE的DDPG深度强化学习在最优资产分配中的应用 

**Authors**: Rongwei Liu, Jin Zheng, John Cartlidge  

**Link**: [PDF](https://arxiv.org/pdf/2508.20103)  

**Abstract**: The optimal asset allocation between risky and risk-free assets is a persistent challenge due to the inherent volatility in financial markets. Conventional methods rely on strict distributional assumptions or non-additive reward ratios, which limit their robustness and applicability to investment goals. To overcome these constraints, this study formulates the optimal two-asset allocation problem as a sequential decision-making task within a Markov Decision Process (MDP). This framework enables the application of reinforcement learning (RL) mechanisms to develop dynamic policies based on simulated financial scenarios, regardless of prerequisites. We use the Kelly criterion to balance immediate reward signals against long-term investment objectives, and we take the novel step of integrating the Time-series Dense Encoder (TiDE) into the Deep Deterministic Policy Gradient (DDPG) RL framework for continuous decision-making. We compare DDPG-TiDE with a simple discrete-action Q-learning RL framework and a passive buy-and-hold investment strategy. Empirical results show that DDPG-TiDE outperforms Q-learning and generates higher risk adjusted returns than buy-and-hold. These findings suggest that tackling the optimal asset allocation problem by integrating TiDE within a DDPG reinforcement learning framework is a fruitful avenue for further exploration. 

**Abstract (ZH)**: 基于TiDE的DDPG reinforcement learning框架下最优资产配置问题的研究 

---
# A Hierarchical Signal Coordination and Control System Using a Hybrid Model-based and Reinforcement Learning Approach 

**Title (ZH)**: 基于混合模型与强化学习方法的分层信号协调与控制系统 

**Authors**: Xianyue Peng, Shenyang Chen, H. Michael Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20102)  

**Abstract**: Signal control in urban corridors faces the dual challenge of maintaining arterial traffic progression while adapting to demand variations at local intersections. We propose a hierarchical traffic signal coordination and control scheme that integrates model-based optimization with reinforcement learning. The system consists of: (i) a High-Level Coordinator (HLC) that selects coordination strategies based on observed and predicted demand; (ii) a Corridor Coordinator that derives phase constraints from the selected strategy-either Max-Flow Coordination (MFC) or Green-Wave Coordination (GWC); and (iii) Hybrid Signal Agents (HSAs) that determine signal phases via reinforcement learning with action masking to enforce feasibility. Hierarchical reinforcement learning with Proximal Policy Optimization (PPO) is used to train HSA and HLC policies. At the lower level, three HSA policies-MFC-aware, GWC-aware, and pure agent control (PAC) are trained in conjunction with their respective coordination strategies. At the higher level, the HLC is trained to dynamically switch strategies using a multi-objective reward balancing corridor-level and network-wide performance. The proposed scheme was developed and evaluated on a SUMO-RLlib platform. Case results show that hybrid MFC maximizes throughput under heavy demand; hybrid GWC consistently minimizes arterial stops and maintains progression across diverse traffic conditions but can reduce network-wide efficiency; and PAC improves network-wide travel time in moderate demand but is less effective under heavy demand. The hierarchical design enables adaptive strategy selection, achieving robust performance across all demand levels. 

**Abstract (ZH)**: 基于模型优化与强化学习的分级交通信号协调与控制方案 

---
# Can LLMs Identify Tax Abuse? 

**Title (ZH)**: LLM能否识别税务舞弊？ 

**Authors**: Andrew Blair-Stanek, Nils Holzenberger, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2508.20097)  

**Abstract**: We investigate whether large language models can discover and analyze U.S. tax-minimization strategies. This real-world domain challenges even seasoned human experts, and progress can reduce tax revenue lost from well-advised, wealthy taxpayers. We evaluate the most advanced LLMs on their ability to (1) interpret and verify tax strategies, (2) fill in gaps in partially specified strategies, and (3) generate complete, end-to-end strategies from scratch. This domain should be of particular interest to the LLM reasoning community: unlike synthetic challenge problems or scientific reasoning tasks, U.S. tax law involves navigating hundreds of thousands of pages of statutes, case law, and administrative guidance, all updated regularly. Notably, LLM-based reasoning identified an entirely novel tax strategy, highlighting these models' potential to revolutionize tax agencies' fight against tax abuse. 

**Abstract (ZH)**: 我们研究大型语言模型是否能够发现和分析美国税收最小化策略。这一现实领域的挑战甚至难倒了资深的人类专家，进展可以减少由于有良好建议的富裕纳税人导致的税收收入流失。我们评估最先进的LLM在以下方面的能力：（1）解释和验证税务策略，（2）填充部分指定策略中的空白，（3）从零开始生成完整、端到端的策略。这一领域对LLM推理社区尤为重要：与合成挑战问题或科学推理任务不同，美国税法涉及导航成百上千页的法典、判例法和行政指导，并且这些内容经常更新。值得注意的是，基于LLM的推理识别出一种全新的税务策略，突显了这些模型有潜力彻底改变税务机构打击税收欺诈的方式。 

---
