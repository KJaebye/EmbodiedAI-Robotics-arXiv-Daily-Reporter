# From EMR Data to Clinical Insight: An LLM-Driven Framework for Automated Pre-Consultation Questionnaire Generation 

**Title (ZH)**: 从电子病历数据到临床洞察：一种由LLM驱动的自动化预咨询问卷生成框架 

**Authors**: Ruiqing Ding, Qianfang Sun, Yongkang Leng, Hui Yin, Xiaojian Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00581)  

**Abstract**: Pre-consultation is a critical component of effective healthcare delivery. However, generating comprehensive pre-consultation questionnaires from complex, voluminous Electronic Medical Records (EMRs) is a challenging task. Direct Large Language Model (LLM) approaches face difficulties in this task, particularly regarding information completeness, logical order, and disease-level synthesis. To address this issue, we propose a novel multi-stage LLM-driven framework: Stage 1 extracts atomic assertions (key facts with timing) from EMRs; Stage 2 constructs personal causal networks and synthesizes disease knowledge by clustering representative networks from an EMR corpus; Stage 3 generates tailored personal and standardized disease-specific questionnaires based on these structured representations. This framework overcomes limitations of direct methods by building explicit clinical knowledge. Evaluated on a real-world EMR dataset and validated by clinical experts, our method demonstrates superior performance in information coverage, diagnostic relevance, understandability, and generation time, highlighting its practical potential to enhance patient information collection. 

**Abstract (ZH)**: 预咨询是有效医疗服务的关键组成部分。然而，从复杂庞大的电子医疗记录（EMRs）中生成全面的预咨询问卷是一项颇具挑战的任务。直接大型语言模型（LLM）方法在这一任务中面临困难，特别是在信息完整性、逻辑顺序和疾病水平综合方面。为解决这一问题，我们提出了一种新颖的多阶段LLM驱动框架：第1阶段从EMRs中提取原子断言（带有时间的关键事实）；第2阶段构建个人因果网络并通过对EMR语料中代表性网络的聚类来合成疾病知识；第3阶段基于这些结构化表示生成个性化的标准化疾病特异性问卷。该框架通过构建明确的临床知识克服了直接方法的限制。在实际EMR数据集上评价并通过临床专家验证，我们的方法在信息覆盖范围、诊断相关性、易理解性和生成时间方面表现出优越性能，突显了其在增强患者信息收集方面的实际潜力。 

---
# Pro2Guard: Proactive Runtime Enforcement of LLM Agent Safety via Probabilistic Model Checking 

**Title (ZH)**: Pro2Guard: 基于概率模型检查的主动运行时LLM代理安全 enforcement 

**Authors**: Haoyu Wang, Chris M. Poskitt, Jun Sun, Jiali Wei  

**Link**: [PDF](https://arxiv.org/pdf/2508.00500)  

**Abstract**: Large Language Model (LLM) agents exhibit powerful autonomous capabilities across domains such as robotics, virtual assistants, and web automation. However, their stochastic behavior introduces significant safety risks that are difficult to anticipate. Existing rule-based enforcement systems, such as AgentSpec, focus on developing reactive safety rules, which typically respond only when unsafe behavior is imminent or has already occurred. These systems lack foresight and struggle with long-horizon dependencies and distribution shifts. To address these limitations, we propose Pro2Guard, a proactive runtime enforcement framework grounded in probabilistic reachability analysis. Pro2Guard abstracts agent behaviors into symbolic states and learns a Discrete-Time Markov Chain (DTMC) from execution traces. At runtime, it anticipates future risks by estimating the probability of reaching unsafe states, triggering interventions before violations occur when the predicted risk exceeds a user-defined threshold. By incorporating semantic validity checks and leveraging PAC bounds, Pro2Guard ensures statistical reliability while approximating the underlying ground-truth model. We evaluate Pro2Guard extensively across two safety-critical domains: embodied household agents and autonomous vehicles. In embodied agent tasks, Pro2Guard enforces safety early on up to 93.6% of unsafe tasks using low thresholds, while configurable modes (e.g., reflect) allow balancing safety with task success, maintaining up to 80.4% task completion. In autonomous driving scenarios, Pro2Guard achieves 100% prediction of traffic law violations and collisions, anticipating risks up to 38.66 seconds ahead. 

**Abstract (ZH)**: 基于概率可达性的主动运行时防护框架Pro2Guard 

---
# Thinking Machines: Mathematical Reasoning in the Age of LLMs 

**Title (ZH)**: 思考机器：在大语言模型时代下的数学推理 

**Authors**: Andrea Asperti, Alberto Naibo, Claudio Sacerdoti Coen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00459)  

**Abstract**: Large Language Models (LLMs) have shown remarkable abilities in structured reasoning and symbolic tasks, with coding emerging as a particular area of strength. This success has sparked growing interest in applying LLMs to mathematics, both in informal problem-solving and formal theorem proving. However, progress in formal mathematics has proven to be significantly more difficult, despite surface-level similarities between programming and proof construction. This discrepancy raises important questions about how LLMs ``reason'', how they are supervised, and whether they internally track a notion of computational or deductive state. In this article, we address the state-of-the-art of the discipline, focusing on recent models and benchmarks, and explore three central issues at the intersection of machine learning and mathematical cognition: (i) the trade-offs between formal and informal mathematics as training domains; (ii) the deeper reasons why proof generation remains more brittle than code synthesis; (iii) and the question of whether LLMs represent, or merely mimic, a notion of evolving logical state. Our goal is not to draw hard boundaries, but to identify where the current limits lie, and how they might be extended. 

**Abstract (ZH)**: 大型语言模型在结构化推理和符号任务中展现了显著的能力，其中编程尤其突出。这一成功激发了将大型语言模型应用于数学领域的兴趣，包括非形式化问题解决和形式化定理证明。尽管编程和证明构建在表面上存在相似之处，但形式数学的进步证明更为艰难。这种差异引发了关于大型语言模型如何推理、如何监督以及它们是否跟踪计算或演绎状态的重要问题。在本文中，我们聚焦于该领域的最新模型和基准，探讨机器学习与数学认知交叉领域中的三个核心问题：（i）形式数学与非形式数学作为训练领域的权衡；（ii）证明生成为何比代码合成更具脆弱性；（iii）大型语言模型是否代表了还是仅仅是模仿了一种演变逻辑状态的概念。我们的目标不是划定清晰的界限，而是识别当前的限制所在，并探寻如何扩展这些限制。 

---
# R1-ACT: Efficient Reasoning Model Safety Alignment by Activating Safety Knowledge 

**Title (ZH)**: R1-ACT: 通过激活安全知识实现高效推理模型安全对齐 

**Authors**: Yeonjun In, Wonjoong Kim, Sangwu Park, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.00324)  

**Abstract**: Although large reasoning models (LRMs) have demonstrated impressive capabilities on complex tasks, recent studies reveal that these models frequently fulfill harmful user instructions, raising significant safety concerns. In this paper, we investigate the underlying cause of LRM safety risks and find that models already possess sufficient safety knowledge but fail to activate it during reasoning. Based on this insight, we propose R1-Act, a simple and efficient post-training method that explicitly triggers safety knowledge through a structured reasoning process. R1-Act achieves strong safety improvements while preserving reasoning performance, outperforming prior alignment methods. Notably, it requires only 1,000 training examples and 90 minutes of training on a single RTX A6000 GPU. Extensive experiments across multiple LRM backbones and sizes demonstrate the robustness, scalability, and practical efficiency of our approach. 

**Abstract (ZH)**: 尽管大规模推理模型(LRMs)在复杂任务上展现了令人印象深刻的性能，但近期的研究表明，这些模型经常执行有害的用户指令，这引发了重大安全问题。在本文中，我们探讨了LRM安全风险的根本原因，并发现模型已经具备足够的安全知识，但在推理过程中未能激活它。基于这一洞察，我们提出了一种简单高效的后训练方法R1-Act，通过结构化的推理过程显式触发安全知识。R1-Act在保持推理性能的同时实现了强大的安全性改进，优于之前的对齐方法。值得注意的是，它只需要1,000个训练样本和单个RTX A6000 GPU上90分钟的训练时间。我们针对多个LRM骨干网络和规模进行了广泛的实验，证明了该方法的稳健性、可扩展性和实际效率。 

---
# Mind the Gap: The Divergence Between Human and LLM-Generated Tasks 

**Title (ZH)**: Mind the Gap: 人类与LLM生成任务之间的分歧 

**Authors**: Yi-Long Lu, Jiajun Song, Chunhui Zhang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00282)  

**Abstract**: Humans constantly generate a diverse range of tasks guided by internal motivations. While generative agents powered by large language models (LLMs) aim to simulate this complex behavior, it remains uncertain whether they operate on similar cognitive principles. To address this, we conducted a task-generation experiment comparing human responses with those of an LLM agent (GPT-4o). We find that human task generation is consistently influenced by psychological drivers, including personal values (e.g., Openness to Change) and cognitive style. Even when these psychological drivers are explicitly provided to the LLM, it fails to reflect the corresponding behavioral patterns. They produce tasks that are markedly less social, less physical, and thematically biased toward abstraction. Interestingly, while the LLM's tasks were perceived as more fun and novel, this highlights a disconnect between its linguistic proficiency and its capacity to generate human-like, embodied this http URL conclude that there is a core gap between the value-driven, embodied nature of human cognition and the statistical patterns of LLMs, highlighting the necessity of incorporating intrinsic motivation and physical grounding into the design of more human-aligned agents. 

**Abstract (ZH)**: 人类不断生成由内在动机引导的多样任务。尽管由大规模语言模型（LLMs）驱动的生成代理旨在模拟这种复杂行为，但尚不清楚它们是否遵循类似的认知原则。为了解决这一问题，我们进行了一个任务生成实验，将人类响应与LLM代理（GPT-4o）的响应进行比较。我们发现，人类任务生成始终受到心理驱动因素的影响，包括个人价值观（如开放性）和认知风格。即使在向LLM明确提供这些心理驱动因素后，它也无法反映相应的行为模式。它们生成的任务在社会性、物理性和主题上都偏向抽象，表现明显不足。有趣的是，尽管LLM生成的任务被认为更具趣味性和新颖性，这突显了其语言能力与其产生类似人类具身任务之间存在的差距。因此，我们得出结论，人类驱动的价值观和具身认知的核心特征与LLM的统计模式之间存在差距，强调了在设计更加与人类对齐的代理时需要纳入内在动机和物理基础的必要性。 

---
# RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization 

**Title (ZH)**: RL-PLUS: 综合策略优化在强化学习中应对大规模语言模型能力边界萎缩问题 

**Authors**: Yihong Dong, Xue Jiang, Yongding Tao, Huanyu Liu, Kechi Zhang, Lili Mou, Rongyu Cao, Yingwei Ma, Jue Chen, Binhua Li, Zhi Jin, Fei Huang, Yongbin Li, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00222)  

**Abstract**: Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its inherently on-policy strategy with LLM's immense action space and sparse reward. Further, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel approach that synergizes internal exploitation (i.e., Thinking) with external data (i.e., Learning) to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components: Multiple Importance Sampling to address for distributional mismatch from external data, and an Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. The results show that RL-PLUS achieves state-of-the-art performance compared with existing RLVR methods on six math reasoning benchmarks and exhibits superior performance on six out-of-distribution reasoning tasks. It also achieves consistent and significant gains across diverse model families, with average relative improvements ranging from 21.1\% to 69.2\%. Moreover, Pass@k curves across multiple benchmarks indicate that RL-PLUS effectively resolves the capability boundary collapse problem. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）显著提升了大型语言模型（LLMs）的复杂推理能力。然而，由于其固有的基于策略方法以及LLMs巨大的动作空间和稀疏奖励，它难以突破基模型的能力边界。进一步地，RLVR可能导致能力边界崩溃，缩小LLMs的问题解决范围。为解决这一问题，我们提出了RL-PLUS，这是一个将内部利用（即思考）与外部数据（即学习）相结合的新方法，以实现更强的推理能力和超越基模型的边界。RL-PLUS整合了两个核心组件：多重重要性采样以解决外部数据的分布不匹配问题，以及基于探索的优点函数以引导模型走向高价值、未探索的推理路径。我们提供了理论分析和广泛的实验来证明我们方法的优越性和普适性。结果显示，与现有的RLVR方法相比，RL-PLUS在六个数学推理基准测试中取得了最先进的性能，并在六个离分布推理任务上表现出色。此外，RL-PLUS在多种模型家族中实现了一致且显著的提升，相对改进范围从21.1%到69.2%。此外，多个基准测试下的Pass@k曲线表明，RL-PLUS有效解决了能力边界崩溃问题。 

---
# Do They Understand Them? An Updated Evaluation on Nonbinary Pronoun Handling in Large Language Models 

**Title (ZH)**: 他们理解它们吗？大规模语言模型对非二元代词处理的最新评估 

**Authors**: Xushuo Tang, Yi Ding, Zhengyi Yang, Yin Chen, Yongrui Gu, Wenke Yang, Mingchen Ju, Xin Cao, Yongfei Liu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00788)  

**Abstract**: Large language models (LLMs) are increasingly deployed in sensitive contexts where fairness and inclusivity are critical. Pronoun usage, especially concerning gender-neutral and neopronouns, remains a key challenge for responsible AI. Prior work, such as the MISGENDERED benchmark, revealed significant limitations in earlier LLMs' handling of inclusive pronouns, but was constrained to outdated models and limited evaluations. In this study, we introduce MISGENDERED+, an extended and updated benchmark for evaluating LLMs' pronoun fidelity. We benchmark five representative LLMs, GPT-4o, Claude 4, DeepSeek-V3, Qwen Turbo, and Qwen2.5, across zero-shot, few-shot, and gender identity inference. Our results show notable improvements compared with previous studies, especially in binary and gender-neutral pronoun accuracy. However, accuracy on neopronouns and reverse inference tasks remains inconsistent, underscoring persistent gaps in identity-sensitive reasoning. We discuss implications, model-specific observations, and avenues for future inclusive AI research. 

**Abstract (ZH)**: 大型语言模型（LLMs）在敏感情境中应用日益增多，其中公平性和包容性至关重要。中性代词及新式代词的使用仍然是负责任AI的关键挑战。此前的研究，如MISGENDERED基准，揭示了早期LLM在处理包容性代词方面的显著限制，但这些研究局限于过时的模型和有限的评估。在本研究中，我们引入了MISGENDERED+，这是一个扩展和更新的基准，用于评估LLM的代词保真度。我们针对零-shot、少-shot及性别身份推理，对五种代表性的LLM（GPT-4o、Claude 4、DeepSeek-V3、Qwen Turbo和Qwen2.5）进行了基准测试。结果显示，与以往研究相比，这些LLM在二元和中性代词准确性方面有显著提高。然而，新式代词和逆向推理任务的准确性仍然不够一致，揭示了身份敏感推理方面的持续差距。我们讨论了研究的意义、模型特定观察以及未来包容性AI研究的方向。 

---
# MMBERT: Scaled Mixture-of-Experts Multimodal BERT for Robust Chinese Hate Speech Detection under Cloaking Perturbations 

**Title (ZH)**: MMBERT：面向遮蔽扰动下 robust  Chinese 恶意言论检测的缩放混合专家多模态 BERT 

**Authors**: Qiyao Xue, Yuchen Dou, Ryan Shi, Xiang Lorraine Li, Wei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00760)  

**Abstract**: Hate speech detection on Chinese social networks presents distinct challenges, particularly due to the widespread use of cloaking techniques designed to evade conventional text-based detection systems. Although large language models (LLMs) have recently improved hate speech detection capabilities, the majority of existing work has concentrated on English datasets, with limited attention given to multimodal strategies in the Chinese context. In this study, we propose MMBERT, a novel BERT-based multimodal framework that integrates textual, speech, and visual modalities through a Mixture-of-Experts (MoE) architecture. To address the instability associated with directly integrating MoE into BERT-based models, we develop a progressive three-stage training paradigm. MMBERT incorporates modality-specific experts, a shared self-attention mechanism, and a router-based expert allocation strategy to enhance robustness against adversarial perturbations. Empirical results in several Chinese hate speech datasets show that MMBERT significantly surpasses fine-tuned BERT-based encoder models, fine-tuned LLMs, and LLMs utilizing in-context learning approaches. 

**Abstract (ZH)**: 中文标题：中国社交媒体上的仇恨言论检测面临独特挑战，特别是由于广泛使用旨在规避传统基于文本检测系统的伪装技术。尽管大型语言模型（LLMs）最近提高了仇恨言论检测能力，但现有研究大多集中于英文数据集上，在中文多模态策略方面的关注有限。本研究提出了一种新颖的基于BERT的多模态框架MMBERT，通过Mixture-of-Experts（MoE）架构整合文本、语音和视觉模态。为了解决直接将MoE集成到BERT模型中所导致的不稳定性问题，我们开发了一种分阶段的三阶段训练 paradigmn。MMBERT通过模态特定专家、共享自我注意机制以及基于路由器的专家分配策略，增强了对对抗性扰动的鲁棒性。在多个中文仇恨言论数据集上的实证结果表明，MMBERT显著优于微调的BERT编码器模型、微调的LLMs以及利用上下文学习方法的LLMs。 

---
# Agentic large language models improve retrieval-based radiology question answering 

**Title (ZH)**: 代理型大型语言模型提升基于检索的放射学问题解答 

**Authors**: Sebastian Wind, Jeta Sopa, Daniel Truhn, Mahshad Lotfinia, Tri-Thien Nguyen, Keno Bressem, Lisa Adams, Mirabela Rusu, Harald Köstler, Gerhard Wellein, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2508.00743)  

**Abstract**: Clinical decision-making in radiology increasingly benefits from artificial intelligence (AI), particularly through large language models (LLMs). However, traditional retrieval-augmented generation (RAG) systems for radiology question answering (QA) typically rely on single-step retrieval, limiting their ability to handle complex clinical reasoning tasks. Here we propose an agentic RAG framework enabling LLMs to autonomously decompose radiology questions, iteratively retrieve targeted clinical evidence from Radiopaedia, and dynamically synthesize evidence-based responses. We evaluated 24 LLMs spanning diverse architectures, parameter scales (0.5B to >670B), and training paradigms (general-purpose, reasoning-optimized, clinically fine-tuned), using 104 expert-curated radiology questions from previously established RSNA-RadioQA and ExtendedQA datasets. Agentic retrieval significantly improved mean diagnostic accuracy over zero-shot prompting (73% vs. 64%; P<0.001) and conventional online RAG (73% vs. 68%; P<0.001). The greatest gains occurred in mid-sized models (e.g., Mistral Large improved from 72% to 81%) and small-scale models (e.g., Qwen 2.5-7B improved from 55% to 71%), while very large models (>200B parameters) demonstrated minimal changes (<2% improvement). Additionally, agentic retrieval reduced hallucinations (mean 9.4%) and retrieved clinically relevant context in 46% of cases, substantially aiding factual grounding. Even clinically fine-tuned models exhibited meaningful improvements (e.g., MedGemma-27B improved from 71% to 81%), indicating complementary roles of retrieval and fine-tuning. These results highlight the potential of agentic frameworks to enhance factuality and diagnostic accuracy in radiology QA, particularly among mid-sized LLMs, warranting future studies to validate their clinical utility. 

**Abstract (ZH)**: 临床放射学中的决策越来越多地受益于人工智能（AI），特别是大型语言模型（LLMs）。然而，传统的放射学问题回答（QA）检索增强生成（RAG）系统通常依赖单步检索，限制了它们处理复杂临床推理任务的能力。我们提出了一种自主性RAG框架，使LLMs能够自主分解放射学问题，迭代地从Radiopaedia检索相关的临床证据，并动态合成基于证据的回应。我们使用RSNA-RadioQA和ExtendedQA数据集中104个专家策划的放射学问题，评估了24种不同的LLM架构、参数规模（0.5B至>670B）和训练范式（通用、推理优化、临床微调），评估结果显示，自主性检索显著提高了诊断准确性（零-shot提示：73% vs. 64%；P<0.001；传统在线RAG：73% vs. 68%；P<0.001）。中等规模模型（如Mistral Large改进为81%）和小型模型（如Qwen 2.5-7B改进为71%）取得了最大收益，而非常大规模模型（>200B参数）仅表现出细微变化（<2%改进）。此外，自主性检索减少了幻觉（平均9.4%）的情况，并在46%的情况下检索到相关的临床背景，显著增强了事实的准确性。即使经过临床微调的模型也表现出有意义的改进（如MedGemma-27B从71%提高到81%），表明检索和微调具有互补作用。这些结果强调了自主性框架在提高放射学QA的事实性和诊断准确性方面的潜力，特别是在中等规模LLM中，未来的研究需要验证其临床效用。 

---
# Out-of-Context Abduction: LLMs Make Inferences About Procedural Data Leveraging Declarative Facts in Earlier Training Data 

**Title (ZH)**: 脱离上下文的推论：大规模语言模型利用早期训练数据中的声明性事实对程序性数据进行推理 

**Authors**: Sohaib Imran, Rob Lamb, Peter M. Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2508.00741)  

**Abstract**: Large language models (LLMs) are trained on large corpora, yet it is unclear whether they can reason about the information present within their training data. We design experiments to study out-of-context abduction in LLMs, the ability to infer the most plausible explanations for observations using relevant facts present in training data. We train treatment LLMs on names and behavior descriptions of fictitious chatbots, but not on examples of dialogue with the chatbots. We find that OpenAI's GPT 4o LLM can correctly infer at least one chatbot's name after observing example responses characteristic of that chatbot. We also find that previously training GPT 4o on descriptions of a chatbot's behavior allows it to display behaviors more characteristic of the chatbot when iteratively trained to display such behaviors. Our results have implications for situational awareness in LLMs and, therefore, for AI safety. 

**Abstract (ZH)**: 大型语言模型（LLMs）虽然在大规模语料上进行训练，但尚不清楚它们能否推理其训练数据中存在的信息。我们设计实验研究LLMs的离境 abduction 能力，即利用训练数据中相关的事实来推断观察现象的最可能解释的能力。我们仅对虚构聊天机器人的名称和行为描述进行训练LLM，而不包含与聊天机器人的对话示例。我们发现，OpenAI的GPT-4能够根据具有该聊天机器人特征的示例响应正确推理出至少一个聊天机器人的名称。我们还发现，之前对聊天机器人行为描述的训练能使GPT-4在迭代训练中更体现该聊天机器人的行为特征。我们的结果对于LLMs的情境意识以及AI安全具有重要意义。 

---
# How LLMs are Shaping the Future of Virtual Reality 

**Title (ZH)**: LLMs是如何塑造虚拟现实未来的研究 

**Authors**: Süeda Özkaya, Santiago Berrezueta-Guzman, Stefan Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2508.00737)  

**Abstract**: The integration of Large Language Models (LLMs) into Virtual Reality (VR) games marks a paradigm shift in the design of immersive, adaptive, and intelligent digital experiences. This paper presents a comprehensive review of recent research at the intersection of LLMs and VR, examining how these models are transforming narrative generation, non-player character (NPC) interactions, accessibility, personalization, and game mastering. Drawing from an analysis of 62 peer reviewed studies published between 2018 and 2025, we identify key application domains ranging from emotionally intelligent NPCs and procedurally generated storytelling to AI-driven adaptive systems and inclusive gameplay interfaces. We also address the major challenges facing this convergence, including real-time performance constraints, memory limitations, ethical risks, and scalability barriers. Our findings highlight that while LLMs significantly enhance realism, creativity, and user engagement in VR environments, their effective deployment requires robust design strategies that integrate multimodal interaction, hybrid AI architectures, and ethical safeguards. The paper concludes by outlining future research directions in multimodal AI, affective computing, reinforcement learning, and open-source development, aiming to guide the responsible advancement of intelligent and inclusive VR systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在虚拟现实（VR）游戏中的集成标志着沉浸式、适应性、智能数字体验设计范式的转变。本文对2018年至2025年间发表的62篇相关研究进行了全面回顾，探讨这些模型如何改变叙事生成、非玩家角色（NPC）交互、可访问性、个性化以及游戏主持等方面。我们识别出了关键的应用领域，包括情绪智能NPC、程序生成叙事、AI驱动的自适应系统以及包容性游戏界面。本文还讨论了这一综合应用面临的主要挑战，包括实时性能限制、内存限制、伦理风险和可扩展性障碍。研究发现虽然LLMs显著增强了VR环境中的真实感、创造力和用户参与度，但其有效部署需要将多模态交互、混合AI架构和伦理保障纳入到稳健的设计策略中。最后，本文概述了未来研究的方向，包括多模态AI、情感计算、强化学习和开源开发，旨在指导智能和包容性VR系统的负责任发展。 

---
# Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA 

**Title (ZH)**: 基于LLM引导的MCTS的动态自适应推理：高效且上下文感知的KGQA 

**Authors**: Yingxu Wang, Shiqi Fan, Mengzhu Wang, Siwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00719)  

**Abstract**: Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods. 

**Abstract (ZH)**: 知识图谱问答（KGQA）旨在通过利用知识图谱的关联和语义结构来解释自然语言查询并进行结构化推理，以检索准确的答案。近期的KGQA方法主要遵循检索-推理范式，依赖于GNNs或启发式规则进行静态路径提取，或使用大型语言模型（LLMs）进行动态路径生成和检索与推理的联合执行。然而，前者由于静态路径提取和缺乏上下文优化而适应性有限，后者则由于依赖固定的评分函数和广泛的LLM调用而产生高额的计算成本，并且难以准确评估路径。为解决这些问题，本文提出了一种名为动态自适应MCTS推理（DAMR）的新颖框架，该框架结合了符号搜索和自适应路径评估，以实现高效且上下文感知的KGQA。DAMR采用由基于LLM的规划器引导的蒙特卡洛树搜索（MCTS）骨干，每一步选择top-$k$相关关系以缩减搜索空间。为了提高路径评估的准确性，引入了一种轻量级的基于Transformer的评分器，通过交叉注意力联合编码问题和关系序列来进行上下文感知的合理性估计，使模型在进行多次跳步推理时能够捕捉到细微的语义转换。此外，为缓解高质量监督的稀缺性，DAMR整合了一种动态伪路径精炼机制，该机制定期在搜索过程中从探索到的部分路径中生成训练信号，使评分器能够持续适应推理轨迹分布的变化。在多个KGQA基准上的广泛实验表明，DAMR显著优于现有方法。 

---
# NyayaRAG: Realistic Legal Judgment Prediction with RAG under the Indian Common Law System 

**Title (ZH)**: NyayaRAG：基于印度普通法体系的现实主义法律判决预测 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Shivam Mishra, Ajay Varghese Thomas, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2508.00709)  

**Abstract**: Legal Judgment Prediction (LJP) has emerged as a key area in AI for law, aiming to automate judicial outcome forecasting and enhance interpretability in legal reasoning. While previous approaches in the Indian context have relied on internal case content such as facts, issues, and reasoning, they often overlook a core element of common law systems, which is reliance on statutory provisions and judicial precedents. In this work, we propose NyayaRAG, a Retrieval-Augmented Generation (RAG) framework that simulates realistic courtroom scenarios by providing models with factual case descriptions, relevant legal statutes, and semantically retrieved prior cases. NyayaRAG evaluates the effectiveness of these combined inputs in predicting court decisions and generating legal explanations using a domain-specific pipeline tailored to the Indian legal system. We assess performance across various input configurations using both standard lexical and semantic metrics as well as LLM-based evaluators such as G-Eval. Our results show that augmenting factual inputs with structured legal knowledge significantly improves both predictive accuracy and explanation quality. 

**Abstract (ZH)**: 法律判决预测（LJP）已成为法律领域人工智能的关键领域，旨在自动化司法结果预测并增强法律推理的可解释性。虽然以往在印度情境下的方法依赖于案件内部内容，如事实、问题和推理，但往往忽视了普通法系统中依靠成文法和判例这一核心要素。在本工作中，我们提出了一种名为NyayaRAG的检索增强生成（RAG）框架，通过向模型提供案情描述、相关法律条文以及语义检索的先例案情，模拟真实的法庭场景。NyayaRAG使用针对印度法律体系定制的特定领域管道，评估这些综合输入在预测法院裁决和生成法律解释方面的有效性。我们使用标准的词汇和语义指标以及基于LLM的评估工具（如G-Eval）对各种输入配置进行性能评估。结果表明，将事实性输入与结构化的法律知识相结合，显著提高了预测准确性和解释质量。 

---
# Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications 

**Title (ZH)**: LLMs时代医学推理的增强技术与应用系统评价 

**Authors**: Wenxuan Wang, Zizhan Ma, Meidan Ding, Shiyi Zheng, Shengyuan Liu, Jie Liu, Jiaming Ji, Wenting Chen, Xiang Li, Linlin Shen, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00669)  

**Abstract**: The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy metrics to sophisticated assessments of reasoning quality and visual interpretability. Based on an analysis of 60 seminal studies from 2022-2025, we conclude by identifying critical challenges, including the faithfulness-plausibility gap and the need for native multimodal reasoning, and outlining future directions toward building efficient, robust, and sociotechnically responsible medical AI. 

**Abstract (ZH)**: 大型语言模型在医学中的普及虽然彰显了 impressive 能力，但其在系统性、透明性和可验证性推理方面仍存在关键缺口，这是临床实践的基石。这推动了从单步骤答案生成向专门为医学推理设计的大规模语言模型的转变。本文首次系统回顾了这一新兴领域。我们提出了推理增强技术的分类，分为训练时策略（如监督微调、强化学习）和测试时机制（如提示工程、多智能体系统）。我们分析了这些技术在不同数据模态（文本、图像、代码）和关键临床应用（如诊断、教育、治疗规划）中的应用。此外，我们回顾了从简单准确性指标到复杂推理质量和可视化解释评估的评价基准演变。基于对 2022-2025 年 60 项开创性研究的分析，我们指出了关键挑战，包括忠实性-合理性差距和原生多模态推理的需要，并概述了构建高效、稳健且社会技术上负责任的医疗人工智能的未来方向。 

---
# LeakSealer: A Semisupervised Defense for LLMs Against Prompt Injection and Leakage Attacks 

**Title (ZH)**: LeakSealer：一种半监督防御方法，用于防范大型语言模型的提示注入和泄漏攻击 

**Authors**: Francesco Panebianco, Stefano Bonfanti, Francesco Trovò, Michele Carminati  

**Link**: [PDF](https://arxiv.org/pdf/2508.00602)  

**Abstract**: The generalization capabilities of Large Language Models (LLMs) have led to their widespread deployment across various applications. However, this increased adoption has introduced several security threats, notably in the forms of jailbreaking and data leakage attacks. Additionally, Retrieval Augmented Generation (RAG), while enhancing context-awareness in LLM responses, has inadvertently introduced vulnerabilities that can result in the leakage of sensitive information. Our contributions are twofold. First, we introduce a methodology to analyze historical interaction data from an LLM system, enabling the generation of usage maps categorized by topics (including adversarial interactions). This approach further provides forensic insights for tracking the evolution of jailbreaking attack patterns. Second, we propose LeakSealer, a model-agnostic framework that combines static analysis for forensic insights with dynamic defenses in a Human-In-The-Loop (HITL) pipeline. This technique identifies topic groups and detects anomalous patterns, allowing for proactive defense mechanisms. We empirically evaluate LeakSealer under two scenarios: (1) jailbreak attempts, employing a public benchmark dataset, and (2) PII leakage, supported by a curated dataset of labeled LLM interactions. In the static setting, LeakSealer achieves the highest precision and recall on the ToxicChat dataset when identifying prompt injection. In the dynamic setting, PII leakage detection achieves an AUPRC of $0.97$, significantly outperforming baselines such as Llama Guard. 

**Abstract (ZH)**: 大型语言模型（LLMs）的通用化能力促进了其在各种应用中的广泛应用，然而这种广泛应用也引入了多种安全威胁，尤其是 Jailbreaking 和数据泄漏攻击。此外，检索增强生成（RAG）虽然增强了LLM响应的语境意识，但也无意中引入了可能导致敏感信息泄露的漏洞。我们的贡献主要有两方面。首先，我们提出了一种方法来分析LLM系统的 historical interaction 数据，生成按主题分类的使用图谱（包括对抗性交互），这一方法进一步提供了法医洞察，用于追踪 Jailbreaking 攻击模式的演变。其次，我们提出了一个模型无关的框架 LeakSealer，该框架结合了静态分析和动态防御，采用人工介入的流程（Human-In-The-Loop, HITL）。该技术可以识别主题组和检测异常模式，从而实现主动防御机制。我们通过两种场景对 LeakSealer 进行了实证评估：（1）针对 Jailbreak 尝试，使用公开基准数据集；（2）针对 PII 泄露，使用标记的 LLML 交互数据集。在静态环境中，LeakSealer 在识别提示注入时在 ToxicChat 数据集上实现了最高的准确率和召回率。在动态环境中，PII 泄露检测的 AUPRC 达到 0.97，显著优于 Llama Guard 等基线方法。 

---
# SynAdapt: Learning Adaptive Reasoning in Large Language Models via Synthetic Continuous Chain-of-Thought 

**Title (ZH)**: SynAdapt: 在大规模语言模型中通过合成连续推理学习自适应推理 

**Authors**: Jianwei Wang, Ziming Wu, Fuming Lai, Shaobing Lian, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00574)  

**Abstract**: While Chain-of-Thought (CoT) reasoning improves model performance, it incurs significant time costs due to the generation of discrete CoT tokens (DCoT). Continuous CoT (CCoT) offers a more efficient alternative, but existing CCoT methods are hampered by indirect fine-tuning, limited alignment, or inconsistent targets. To overcome these limitations, we propose \textit{SynAdapt}, an innovative efficient reasoning framework. Specifically, \textit{SynAdapt} generates the synthetic CCoT to serve as a precise and effective alignment target for LLMs. This synthetic CCoT explicitly guides the LLM to learn CCoT and derive accurate answers directly. Furthermore, relying solely on CCoT is insufficient for solving hard questions. To address this, \textit{SynAdapt} integrates a difficulty classifier that leverages both question context and CCoT to identify hard questions. CCoT can effectively help identify hard questions after some brief reasoning. We then adaptively prompt the LLM to re-think these hard questions for improved performance. Extensive experimental results across various benchmarks from different difficulty levels strongly demonstrate the effectiveness of our method, achieving the best accuracy-efficiency trade-off. 

**Abstract (ZH)**: SynAdapt：一种有效的合成连续推理适应框架 

---
# CyGATE: Game-Theoretic Cyber Attack-Defense Engine for Patch Strategy Optimization 

**Title (ZH)**: 基于博弈论的CyGATE网络攻击防御引擎：补丁策略优化 

**Authors**: Yuning Jiang, Nay Oo, Qiaoran Meng, Lu Lin, Dusit Niyato, Zehui Xiong, Hoon Wei Lim, Biplab Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00478)  

**Abstract**: Modern cyber attacks unfold through multiple stages, requiring defenders to dynamically prioritize mitigations under uncertainty. While game-theoretic models capture attacker-defender interactions, existing approaches often rely on static assumptions and lack integration with real-time threat intelligence, limiting their adaptability. This paper presents CyGATE, a game-theoretic framework modeling attacker-defender interactions, using large language models (LLMs) with retrieval-augmented generation (RAG) to enhance tactic selection and patch prioritization. Applied to a two-agent scenario, CyGATE frames cyber conflicts as a partially observable stochastic game (POSG) across Cyber Kill Chain stages. Both agents use belief states to navigate uncertainty, with the attacker adapting tactics and the defender re-prioritizing patches based on evolving risks and observed adversary behavior. The framework's flexible architecture enables extension to multi-agent scenarios involving coordinated attackers, collaborative defenders, or complex enterprise environments with multiple stakeholders. Evaluated in a dynamic patch scheduling scenario, CyGATE effectively prioritizes high-risk vulnerabilities, enhancing adaptability through dynamic threat integration, strategic foresight by anticipating attacker moves under uncertainty, and efficiency by optimizing resource use. 

**Abstract (ZH)**: 基于大型语言模型的CyGATE博弈框架：动态威胁情报下的攻击者-防御者交互 Modeling CyGATE with Large Language Models: Attacker-Defender Interactions under Dynamic Threat Intelligence 

---
# When Relevance Meets Novelty: Dual-Stable Periodic Optimization for Exploratory Recommendation 

**Title (ZH)**: 当相关性遇上新颖性：探索性推荐的双稳周期优化 

**Authors**: Hongxiang Lin, Hao Guo, Zeshun Li, Erpeng Xue, Yongqian He, Xiangyu Hou, Zhaoyu Hu, Lei Wang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00450)  

**Abstract**: Traditional recommendation systems tend to trap users in strong feedback loops by excessively pushing content aligned with their historical preferences, thereby limiting exploration opportunities and causing content fatigue. Although large language models (LLMs) demonstrate potential with their diverse content generation capabilities, existing LLM-enhanced dual-model frameworks face two major limitations: first, they overlook long-term preferences driven by group identity, leading to biased interest modeling; second, they suffer from static optimization flaws, as a one-time alignment process fails to leverage incremental user data for closed-loop optimization. To address these challenges, we propose the Co-Evolutionary Alignment (CoEA) method. For interest modeling bias, we introduce Dual-Stable Interest Exploration (DSIE) module, jointly modeling long-term group identity and short-term individual interests through parallel processing of behavioral sequences. For static optimization limitations, we design a Periodic Collaborative Optimization (PCO) mechanism. This mechanism regularly conducts preference verification on incremental data using the Relevance LLM, then guides the Novelty LLM to perform fine-tuning based on the verification results, and subsequently feeds back the output of the incrementally fine-tuned Novelty LLM to the Relevance LLM for re-evaluation, thereby achieving a dynamic closed-loop optimization. Extensive online and offline experiments verify the effectiveness of the CoEA model in exploratory recommendation. 

**Abstract (ZH)**: 传统推荐系统倾向于通过过度推送符合用户历史偏好的内容，将用户困在强烈的反馈循环中，从而限制探索机会并导致内容疲劳。尽管大型语言模型（LLMs）凭借其多元的内容生成能力展现了潜力，但现有的LLM增强双模型框架面临两大主要局限性：首先，它们忽略了由群体身份驱动的长期偏好，导致兴趣建模出现偏差；其次，它们受到静态优化缺陷的困扰，一次性对齐过程无法利用增量用户数据进行闭环优化。为应对这些挑战，我们提出了共生对齐（CoEA）方法。为解决兴趣建模偏差问题，我们引入了双重稳定兴趣探索（DSIE）模块，通过并行处理行为序列来共同建模长期群体身份和短期个体兴趣。为解决静态优化局限性，我们设计了周期性协作优化（PCO）机制。该机制使用相关性LLM定期对增量数据进行偏好验证，然后指导新颖性LLM根据验证结果进行微调，并随后将增量微调后的新颖性LLM的输出反馈给相关性LLM重新评估，从而实现动态闭环优化。广泛开展的在线和离线实验验证了CoEA模型在探索性推荐中的有效性。 

---
# Calibrated Language Models and How to Find Them with Label Smoothing 

**Title (ZH)**: 校准语言模型及通过标签平滑寻找它们的方法 

**Authors**: Jerry Huang, Peng Lu, Qiuhao Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.00264)  

**Abstract**: Recent advances in natural language processing (NLP) have opened up greater opportunities to enable fine-tuned large language models (LLMs) to behave as more powerful interactive agents through improved instruction-following ability. However, understanding how this impacts confidence calibration for reliable model output has not been researched in full. In this work, we examine various open-sourced LLMs, identifying significant calibration degradation after instruction tuning in each. Seeking a practical solution, we look towards label smoothing, which has been shown as an effective method to regularize for overconfident predictions but has yet to be widely adopted in the supervised fine-tuning (SFT) of LLMs. We first provide insight as to why label smoothing is sufficient to maintain calibration throughout the SFT process. However, settings remain where the effectiveness of smoothing is severely diminished, in particular the case of large vocabulary LLMs (LV-LLMs). We posit the cause to stem from the ability to become over-confident, which has a direct relationship with the hidden size and vocabulary size, and justify this theoretically and experimentally. Finally, we address an outstanding issue regarding the memory footprint of the cross-entropy loss computation in the label smoothed loss setting, designing a customized kernel to dramatically reduce memory consumption without sacrificing speed or performance in comparison to existing solutions for non-smoothed losses. 

**Abstract (ZH)**: 近期自然语言处理(NLP)的进步为通过改进指令跟随能力使细调的大语言模型(LLMs)成为更强大的交互代理提供了更大的机会。然而，这一变化对可靠模型输出的信心校准影响尚无全面研究。在这项工作中，我们考察了多种开源LLMs，发现在每种模型中指令调校后都会出现显著的信心校准下降。为寻求实际解决方案，我们转向标签平滑，这已被证明是一种有效的过度自信预测正则化方法，但在大语言模型的监督微调(SFT)中尚未得到广泛应用。我们首先探讨标签平滑为何能在SFT过程中保持校准。然而，在某些情况下，平滑的效果大幅减弱，特别是在大词汇量语言模型中。我们认为原因在于过度自信的能力，与隐藏层尺寸和词汇量直接相关，并从理论上和实验上进行了验证。最后，我们解决了标签平滑损失计算中的内存占用问题，设计了一种定制内核，显著减少了内存消耗，同时在速度和性能上与现有非平滑损失解决方案持平。 

---
# Accurate and Consistent Graph Model Generation from Text with Large Language Models 

**Title (ZH)**: 使用大规模语言模型从文本生成准确一致的图模型 

**Authors**: Boqi Chen, Ou Wei, Bingzhou Zheng, Gunter Mussbacher  

**Link**: [PDF](https://arxiv.org/pdf/2508.00255)  

**Abstract**: Graph model generation from natural language description is an important task with many applications in software engineering. With the rise of large language models (LLMs), there is a growing interest in using LLMs for graph model generation. Nevertheless, LLM-based graph model generation typically produces partially correct models that suffer from three main issues: (1) syntax violations: the generated model may not adhere to the syntax defined by its metamodel, (2) constraint inconsistencies: the structure of the model might not conform to some domain-specific constraints, and (3) inaccuracy: due to the inherent uncertainty in LLMs, the models can include inaccurate, hallucinated elements. While the first issue is often addressed through techniques such as constraint decoding or filtering, the latter two remain largely unaddressed. Motivated by recent self-consistency approaches in LLMs, we propose a novel abstraction-concretization framework that enhances the consistency and quality of generated graph models by considering multiple outputs from an LLM. Our approach first constructs a probabilistic partial model that aggregates all candidate outputs and then refines this partial model into the most appropriate concrete model that satisfies all constraints. We evaluate our framework on several popular open-source and closed-source LLMs using diverse datasets for model generation tasks. The results demonstrate that our approach significantly improves both the consistency and quality of the generated graph models. 

**Abstract (ZH)**: 基于自然语言描述的图模型生成是软件工程中一项重要的任务，随着大型语言模型（LLMs）的兴起，人们越来越关注使用LLMs进行图模型生成。然而，基于LLMs的图模型生成通常会产生部分正确的模型，这些问题主要包括三个方面：（1）语法规则违背：生成的模型可能不遵循其元模型定义的语法规则；（2）约束不一致：模型的结构可能不符合某些特定领域的约束；（3）不准确性：由于LLMs固有的不确定性，模型可能包含不准确或虚构的元素。尽管第一个问题通常通过约束解码或过滤等技术来解决，后两个问题仍然没有得到充分解决。受近期LLMs自我一致性方法的启发，我们提出了一种新颖的抽象化-具体化框架，通过考虑LLM的多个输出来增强生成的图模型的一致性和质量。该方法首先构建一个概率性的部分模型，汇集所有候选输出，然后将该部分模型细化为最合适的具体模型，该模型能够满足所有约束。我们使用多样化的数据集对多个流行的开源和封闭源LLMs进行了模型生成任务的评估。结果表明，我们的方法显著提高了生成的图模型的一致性和质量。 

---
# Model Misalignment and Language Change: Traces of AI-Associated Language in Unscripted Spoken English 

**Title (ZH)**: 模型失配与语言变化：未剧本化英语中与AI相关的语言痕迹 

**Authors**: Bryce Anderson, Riley Galpin, Tom S. Juzek  

**Link**: [PDF](https://arxiv.org/pdf/2508.00238)  

**Abstract**: In recent years, written language, particularly in science and education, has undergone remarkable shifts in word usage. These changes are widely attributed to the growing influence of Large Language Models (LLMs), which frequently rely on a distinct lexical style. Divergences between model output and target audience norms can be viewed as a form of misalignment. While these shifts are often linked to using Artificial Intelligence (AI) directly as a tool to generate text, it remains unclear whether the changes reflect broader changes in the human language system itself. To explore this question, we constructed a dataset of 22.1 million words from unscripted spoken language drawn from conversational science and technology podcasts. We analyzed lexical trends before and after ChatGPT's release in 2022, focusing on commonly LLM-associated words. Our results show a moderate yet significant increase in the usage of these words post-2022, suggesting a convergence between human word choices and LLM-associated patterns. In contrast, baseline synonym words exhibit no significant directional shift. Given the short time frame and the number of words affected, this may indicate the onset of a remarkable shift in language use. Whether this represents natural language change or a novel shift driven by AI exposure remains an open question. Similarly, although the shifts may stem from broader adoption patterns, it may also be that upstream training misalignments ultimately contribute to changes in human language use. These findings parallel ethical concerns that misaligned models may shape social and moral beliefs. 

**Abstract (ZH)**: 近年来，特别是在科学和教育领域，书面语言在词汇使用方面经历了显著转变。这些变化广泛认为是大型语言模型（LLMs）日益影响的结果，后者经常依赖于独特的词汇风格。模型输出与目标受众规范之间的差异可被视为一种失准形式。尽管这些转变通常与直接使用人工智能（AI）作为生成文本的工具有关，但目前仍不清楚这些变化是否反映了人类语言系统本身的更广泛变化。为了探索这一问题，我们构建了一个来自对话型科学和技术播客的2210万词的语料库，分析了ChatGPT于2022年发布前后词汇趋势，重点关注与LLM关联的词汇。结果显示，在2022年后这些词汇的使用有适度但显著的增加，表明人类词汇选择与LLM关联模式之间的趋同。相比之下，基准同义词在方向上未表现出显著变化。鉴于时间短暂和受影响词汇的数量，这可能表明语言使用正在经历显著转变。这种变化是自然语言演变还是由AI暴露驱动的新转变尚存争议。同样，尽管转变可能源自更广泛的应用模式，但上游训练失准也可能最终导致人类语言使用的变化。这些发现与伦理担忧相呼应，即失准模型可能塑造社会和道德信念。 

---
# Quality-of-Service Aware LLM Routing for Edge Computing with Multiple Experts 

**Title (ZH)**: 面向边缘计算的多专家服务质量感知大型语言模型路由方法 

**Authors**: Jin Yang, Qiong Wu, Zhiying Feng, Zhi Zhou, Deke Guo, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00234)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, leading to a significant increase in user demand for LLM services. However, cloud-based LLM services often suffer from high latency, unstable responsiveness, and privacy concerns. Therefore, multiple LLMs are usually deployed at the network edge to boost real-time responsiveness and protect data privacy, particularly for many emerging smart mobile and IoT applications. Given the varying response quality and latency of LLM services, a critical issue is how to route user requests from mobile and IoT devices to an appropriate LLM service (i.e., edge LLM expert) to ensure acceptable quality-of-service (QoS). Existing routing algorithms fail to simultaneously address the heterogeneity of LLM services, the interference among requests, and the dynamic workloads necessary for maintaining long-term stable QoS. To meet these challenges, in this paper we propose a novel deep reinforcement learning (DRL)-based QoS-aware LLM routing framework for sustained high-quality LLM services. Due to the dynamic nature of the global state, we propose a dynamic state abstraction technique to compactly represent global state features with a heterogeneous graph attention network (HAN). Additionally, we introduce an action impact estimator and a tailored reward function to guide the DRL agent in maximizing QoS and preventing latency violations. Extensive experiments on both Poisson and real-world workloads demonstrate that our proposed algorithm significantly improves average QoS and computing resource efficiency compared to existing baselines. 

**Abstract (ZH)**: 基于深度强化学习的高质量LLM路由框架：动态状态抽象与服务质量优化 

---
# A Survey on Code Generation with LLM-based Agents 

**Title (ZH)**: 基于LLM的代理代码生成综述 

**Authors**: Yihong Dong, Xue Jiang, Jiaru Qian, Tian Wang, Kechi Zhang, Zhi Jin, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00083)  

**Abstract**: Code generation agents powered by large language models (LLMs) are revolutionizing the software development paradigm. Distinct from previous code generation techniques, code generation agents are characterized by three core features. 1) Autonomy: the ability to independently manage the entire workflow, from task decomposition to coding and debugging. 2) Expanded task scope: capabilities that extend beyond generating code snippets to encompass the full software development lifecycle (SDLC). 3) Enhancement of engineering practicality: a shift in research emphasis from algorithmic innovation toward practical engineering challenges, such as system reliability, process management, and tool integration. This domain has recently witnessed rapid development and an explosion in research, demonstrating significant application potential. This paper presents a systematic survey of the field of LLM-based code generation agents. We trace the technology's developmental trajectory from its inception and systematically categorize its core techniques, including both single-agent and multi-agent architectures. Furthermore, this survey details the applications of LLM-based agents across the full SDLC, summarizes mainstream evaluation benchmarks and metrics, and catalogs representative tools. Finally, by analyzing the primary challenges, we identify and propose several foundational, long-term research directions for the future work of the field. 

**Abstract (ZH)**: 由大型语言模型（LLMs）驱动的代码生成代理正在革新软件开发范式。与之前的代码生成技术不同，代码生成代理具有三个核心特征：1）自主性：独立管理从任务分解到编码和调试的 entire workflow。2）扩展的任务范围：能够扩展到涵盖整个软件开发生命周期（SDLC）的能力。3）增强的工程实用性和：研究重点从算法创新转向系统可靠性、过程管理以及工具集成等实际工程挑战。这一领域近期经历了快速发展和研究爆炸性增长，显示出巨大的应用潜力。本文对基于大型语言模型的代码生成代理领域的研究进行了系统的综述，追溯了该技术从 inception 到今的发展轨迹，系统地分类了其核心技术，包括单一代理和多代理架构。此外，本文详细介绍了基于大型语言模型的代理在整个 SDLC 中的应用，总结了主流评估基准和指标，并列出了代表性工具。最后，通过对主要挑战的分析，指出了未来研究方向，并提出了一些基础性、长期的研究建议。 

---
# PhysicsEval: Inference-Time Techniques to Improve the Reasoning Proficiency of Large Language Models on Physics Problems 

**Title (ZH)**: PhysicsEval：在推理阶段提高大型语言模型解决物理问题能力的技术 

**Authors**: Oshayer Siddique, J. M Areeb Uzair Alam, Md Jobayer Rahman Rafy, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00079)  

**Abstract**: The discipline of physics stands as a cornerstone of human intellect, driving the evolution of technology and deepening our understanding of the fundamental principles of the cosmos. Contemporary literature includes some works centered on the task of solving physics problems - a crucial domain of natural language reasoning. In this paper, we evaluate the performance of frontier LLMs in solving physics problems, both mathematical and descriptive. We also employ a plethora of inference-time techniques and agentic frameworks to improve the performance of the models. This includes the verification of proposed solutions in a cumulative fashion by other, smaller LLM agents, and we perform a comparative analysis of the performance that the techniques entail. There are significant improvements when the multi-agent framework is applied to problems that the models initially perform poorly on. Furthermore, we introduce a new evaluation benchmark for physics problems, ${\rm P{\small HYSICS}E{\small VAL}}$, consisting of 19,609 problems sourced from various physics textbooks and their corresponding correct solutions scraped from physics forums and educational websites. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 物理学作为人类智慧的基石，推动着技术的发展，并加深了我们对宇宙基本原理的理解。当代文献中包括一些专注于解决物理问题的作品——这是自然语言推理的关键领域。本文评估了前沿的大规模语言模型在解决物理问题（包括数学问题和描述性问题）方面的性能。我们还采用了多种推理时技术和代理框架来提高模型的性能，包括由其他较小的LLM代理以累计方式验证提出的解决方案，并进行了技术性能的比较分析。当将多代理框架应用于模型最初表现不佳的问题时，性能有显著提升。此外，我们引入了一个新的物理问题评估基准${\rm P{\small HYSICS}E{\smallVAL}}$，包含来自各种物理教材的19,609个问题及其正确解，这些正确解是从物理论坛和教育网站抓取的。我们的代码和数据在此处公开：这个https://链接。 

---
# TriP-LLM: A Tri-Branch Patch-wise Large Language Model Framework for Time-Series Anomaly Detection 

**Title (ZH)**: TriP-LLM：一种用于时间序列异常检测的三分支块级大型语言模型框架 

**Authors**: Yuan-Cheng Yu, Yen-Chieh Ouyang, Chun-An Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.00047)  

**Abstract**: Time-series anomaly detection plays a central role across a wide range of application domains. With the increasing proliferation of the Internet of Things (IoT) and smart manufacturing, time-series data has dramatically increased in both scale and dimensionality. This growth has exposed the limitations of traditional statistical methods in handling the high heterogeneity and complexity of such data. Inspired by the recent success of large language models (LLMs) in multimodal tasks across language and vision domains, we propose a novel unsupervised anomaly detection framework: A Tri-Branch Patch-wise Large Language Model Framework for Time-Series Anomaly Detection (TriP-LLM). TriP-LLM integrates local and global temporal features through a tri-branch design-Patching, Selection, and Global-to encode the input time series into patch-wise tokens, which are then processed by a frozen, pretrained LLM. A lightweight patch-wise decoder reconstructs the input, from which anomaly scores are derived. We evaluate TriP-LLM on several public benchmark datasets using PATE, a recently proposed threshold-free evaluation metric, and conduct all comparisons within a unified open-source framework to ensure fairness. Experimental results show that TriP-LLM consistently outperforms recent state-of-the-art methods across all datasets, demonstrating strong detection capabilities. Furthermore, through extensive ablation studies, we verify the substantial contribution of the LLM to the overall architecture. Compared to LLM-based approaches using Channel Independence (CI) patch processing, TriP-LLM achieves significantly lower memory consumption, making it more suitable for GPU memory-constrained environments. All code and model checkpoints are publicly available on this https URL 

**Abstract (ZH)**: 三支路(patch-wise)大型语言模型时间序列异常检测框架（TriP-LLM） 

---
# Learning Like Humans: Resource-Efficient Federated Fine-Tuning through Cognitive Developmental Stages 

**Title (ZH)**: 模仿人类学习：通过认知发展阶段实现高效联邦微调 

**Authors**: Yebo Wu, Jingguang Li, Zhijiang Guo, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00041)  

**Abstract**: Federated fine-tuning enables Large Language Models (LLMs) to adapt to downstream tasks while preserving data privacy, but its resource-intensive nature limits deployment on edge devices. In this paper, we introduce Developmental Federated Tuning (DevFT), a resource-efficient approach inspired by cognitive development that progressively builds a powerful LLM from a compact foundation. DevFT decomposes the fine-tuning process into developmental stages, each optimizing submodels with increasing parameter capacity. Knowledge from earlier stages transfers to subsequent submodels, providing optimized initialization parameters that prevent convergence to local minima and accelerate training. This paradigm mirrors human learning, gradually constructing comprehensive knowledge structure while refining existing skills. To efficiently build stage-specific submodels, DevFT introduces deconfliction-guided layer grouping and differential-based layer fusion to distill essential information and construct representative layers. Evaluations across multiple benchmarks demonstrate that DevFT significantly outperforms state-of-the-art methods, achieving up to 4.59$\times$ faster convergence, 10.67$\times$ reduction in communication overhead, and 9.07% average performance improvement, while maintaining compatibility with existing approaches. 

**Abstract (ZH)**: 发展性联邦微调（DevFT）：一种资源高效的大语言模型微调方法 

---
# GPT-4.1 Sets the Standard in Automated Experiment Design Using Novel Python Libraries 

**Title (ZH)**: GPT-4.1 通过新型Python库在自动化实验设计中设定标准 

**Authors**: Nuno Fachada, Daniel Fernandes, Carlos M. Fernandes, Bruno D. Ferreira-Saraiva, João P. Matos-Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2508.00033)  

**Abstract**: Large Language Models (LLMs) have advanced rapidly as tools for automating code generation in scientific research, yet their ability to interpret and use unfamiliar Python APIs for complex computational experiments remains poorly characterized. This study systematically benchmarks a selection of state-of-the-art LLMs in generating functional Python code for two increasingly challenging scenarios: conversational data analysis with the \textit{ParShift} library, and synthetic data generation and clustering using \textit{pyclugen} and \textit{scikit-learn}. Both experiments use structured, zero-shot prompts specifying detailed requirements but omitting in-context examples. Model outputs are evaluated quantitatively for functional correctness and prompt compliance over multiple runs, and qualitatively by analyzing the errors produced when code execution fails. Results show that only a small subset of models consistently generate correct, executable code, with GPT-4.1 standing out as the only model to always succeed in both tasks. In addition to benchmarking LLM performance, this approach helps identify shortcomings in third-party libraries, such as unclear documentation or obscure implementation bugs. Overall, these findings highlight current limitations of LLMs for end-to-end scientific automation and emphasize the need for careful prompt design, comprehensive library documentation, and continued advances in language model capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）作为自动化代码生成的工具在科学研究中快速发展，但其解释和使用陌生的Python API进行复杂计算实验的能力仍然缺乏系统性的描述。本研究系统性地评估了多种最先进的LLMs在两种逐步增加挑战性的情景下生成功能性Python代码的能力：使用\textit{ParShift}库进行对话式数据分析，以及使用\textit{pyclugen}和\textit{scikit-learn}进行合成数据生成和聚类。两个实验均使用结构化、零样本提示，明确指定详细要求但省略上下文示例。模型输出在多次运行中从功能性正确性和提示合规性两个方面进行定量评估，并通过分析代码执行失败时产生的错误进行定性评估。结果表明，只有少数模型能够一致生成正确的可执行代码，其中GPT-4.1脱颖而出，唯一能够在两项任务中始终成功。除了评估LLM性能外，这种方法还帮助识别第三方库的缺点，例如不清晰的文档或隐秘的实现错误。总体而言，这些发现突显了LLMs在端到端科学自动化中的当前局限性，并强调了精心设计提示、全面的库文档以及继续提升语言模型能力的必要性。 

---
