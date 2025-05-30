# Graph Drawing for LLMs: An Empirical Evaluation 

**Title (ZH)**: LLMs的图形绘制：一项实证评价 

**Authors**: Walter Didimo, Fabrizio Montecchiani, Tommaso Piselli  

**Link**: [PDF](https://arxiv.org/pdf/2505.03678)  

**Abstract**: Our work contributes to the fast-growing literature on the use of Large Language Models (LLMs) to perform graph-related tasks. In particular, we focus on usage scenarios that rely on the visual modality, feeding the model with a drawing of the graph under analysis. We investigate how the model's performance is affected by the chosen layout paradigm, the aesthetics of the drawing, and the prompting technique used for the queries. We formulate three corresponding research questions and present the results of a thorough experimental analysis. Our findings reveal that choosing the right layout paradigm and optimizing the readability of the input drawing from a human perspective can significantly improve the performance of the model on the given task. Moreover, selecting the most effective prompting technique is a challenging yet crucial task for achieving optimal performance. 

**Abstract (ZH)**: 我们的工作为大型语言模型（LLMs）在图相关任务中的应用这一快速发展的研究领域做出了贡献。特别地，我们关注依赖视觉模态的应用场景，通过向模型提供要分析的图的绘制来实现。我们研究了所选布局范式、绘制的美学以及用于查询的提示技术对模型性能的影响。我们提出了三个相应的研究问题，并呈现了详细实验分析的结果。我们的发现表明，从人类视角优化输入绘制的可读性和选择最有效的提示技术可以显著提高模型在给定任务上的性能。此外，选择最有效的提示技术是实现最佳性能的一项具有挑战但至关重要的任务。 

---
# Gap the (Theory of) Mind: Sharing Beliefs About Teammates' Goals Boosts Collaboration Perception, Not Performance 

**Title (ZH)**: 认知差异（理论中的）：关于队友目标信念的共享增强合作感知，但不提升绩效 

**Authors**: Yotam Amitai, Reuth Mirsky, Ofra Amir  

**Link**: [PDF](https://arxiv.org/pdf/2505.03674)  

**Abstract**: In human-agent teams, openly sharing goals is often assumed to enhance planning, collaboration, and effectiveness. However, direct communication of these goals is not always feasible, requiring teammates to infer their partner's intentions through actions. Building on this, we investigate whether an AI agent's ability to share its inferred understanding of a human teammate's goals can improve task performance and perceived collaboration. Through an experiment comparing three conditions-no recognition (NR), viable goals (VG), and viable goals on-demand (VGod) - we find that while goal-sharing information did not yield significant improvements in task performance or overall satisfaction scores, thematic analysis suggests that it supported strategic adaptations and subjective perceptions of collaboration. Cognitive load assessments revealed no additional burden across conditions, highlighting the challenge of balancing informativeness and simplicity in human-agent interactions. These findings highlight the nuanced trade-off of goal-sharing: while it fosters trust and enhances perceived collaboration, it can occasionally hinder objective performance gains. 

**Abstract (ZH)**: 在人类-代理团队中，公开共享目标通常被认为能增强规划、合作和有效性。然而，直接沟通这些目标并不总是可行的，需要团队成员通过观察行为推断伙伴的意图。基于此，我们研究AI代理分享其推断的人类队友目标能力是否能提高任务性能和感知的合作度。通过比较三种条件（无识别组、可行目标组和按需可行目标组）的实验，我们发现，虽然共享目标信息并未显著提高任务性能或总体满意度评分，但主题分析表明，它支持了战略调整并影响了对合作的主观感知。认知负荷评估显示，各组之间没有额外负担，突显了在人类-代理交互中平衡信息性和简洁性的挑战。这些发现强调了目标共享的微妙权衡：虽然它促进了信任并增强了感知的合作度，但也可能偶尔妨碍客观性能的提升。 

---
# Learning Symbolic Persistent Macro-Actions for POMDP Solving Over Time 

**Title (ZH)**: 基于时间的POMDP求解中的符号持久宏操作学习 

**Authors**: Celeste Veronese, Daniele Meli, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.03668)  

**Abstract**: This paper proposes an integration of temporal logical reasoning and Partially Observable Markov Decision Processes (POMDPs) to achieve interpretable decision-making under uncertainty with macro-actions. Our method leverages a fragment of Linear Temporal Logic (LTL) based on Event Calculus (EC) to generate \emph{persistent} (i.e., constant) macro-actions, which guide Monte Carlo Tree Search (MCTS)-based POMDP solvers over a time horizon, significantly reducing inference time while ensuring robust performance. Such macro-actions are learnt via Inductive Logic Programming (ILP) from a few traces of execution (belief-action pairs), thus eliminating the need for manually designed heuristics and requiring only the specification of the POMDP transition model. In the Pocman and Rocksample benchmark scenarios, our learned macro-actions demonstrate increased expressiveness and generality when compared to time-independent heuristics, indeed offering substantial computational efficiency improvements. 

**Abstract (ZH)**: 本文提出了一种将时间逻辑推理与部分可观测马尔可夫决策过程（POMDP）结合的方法，以在宏操作的指导下实现不确定性下的可解释决策。该方法利用基于事件 calculus 的线性时序逻辑（LTL）片段生成持久性（即恒定）的宏操作，这些宏操作引导时间 horzion 上的蒙特卡罗树搜索（MCTS）基 POMDP 求解器，显著减少推理时间同时保证鲁棒性能。这些宏操作通过归纳逻辑编程（ILP）从少量执行轨迹（信念-操作对）中学得，从而省去了手动设计启发式函数的需求，只需指定 POMDP 过渡模型即可。在 Pocman 和 Rocksample 基准场景中，我们学习的宏操作在表达性和普适性方面优于时间无关的启发式函数，确实提供了显著的计算效率提升。 

---
# BURNS: Backward Underapproximate Reachability for Neural-Feedback-Loop Systems 

**Title (ZH)**: BURNS: 后向近似可达性分析用于神经反馈系统 

**Authors**: Chelsea Sidrane, Jana Tumova  

**Link**: [PDF](https://arxiv.org/pdf/2505.03643)  

**Abstract**: Learning-enabled planning and control algorithms are increasingly popular, but they often lack rigorous guarantees of performance or safety. We introduce an algorithm for computing underapproximate backward reachable sets of nonlinear discrete time neural feedback loops. We then use the backward reachable sets to check goal-reaching properties. Our algorithm is based on overapproximating the system dynamics function to enable computation of underapproximate backward reachable sets through solutions of mixed-integer linear programs. We rigorously analyze the soundness of our algorithm and demonstrate it on a numerical example. Our work expands the class of properties that can be verified for learning-enabled systems. 

**Abstract (ZH)**: 学习驱动的规划与控制算法日益流行，但往往缺乏对性能或安全性的严格保证。我们提出了一种计算非线性离散时间神经反馈回路的下近似可到达集的算法。然后，我们使用可到达集来检验目标达成属性。我们的算法通过求解混合整数线性规划问题来计算下近似可到达集，基于对系统动力学函数的上近似。我们严格分析了该算法的正确性，并在数值示例上进行了演示。我们的工作扩展了可以验证的学习驱动系统属性的类别。 

---
# Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability 

**Title (ZH)**: 在ANN知觉边界上合成图像以揭示和操控人类知觉变异 

**Authors**: Chen Wei, Chi Zhang, Jiachen Zou, Haotian Deng, Dietmar Heinke, Quanying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03641)  

**Abstract**: Human decision-making in cognitive tasks and daily life exhibits considerable variability, shaped by factors such as task difficulty, individual preferences, and personal experiences. Understanding this variability across individuals is essential for uncovering the perceptual and decision-making mechanisms that humans rely on when faced with uncertainty and ambiguity. We present a computational framework BAM (Boundary Alignment & Manipulation framework) that combines perceptual boundary sampling in ANNs and human behavioral experiments to systematically investigate this phenomenon. Our perceptual boundary sampling algorithm generates stimuli along ANN decision boundaries that intrinsically induce significant perceptual variability. The efficacy of these stimuli is empirically validated through large-scale behavioral experiments involving 246 participants across 116,715 trials, culminating in the variMNIST dataset containing 19,943 systematically annotated images. Through personalized model alignment and adversarial generation, we establish a reliable method for simultaneously predicting and manipulating the divergent perceptual decisions of pairs of participants. This work bridges the gap between computational models and human individual difference research, providing new tools for personalized perception analysis. 

**Abstract (ZH)**: 人类在认知任务和日常生活中决策表现出显著的 variability，受任务难度、个人偏好和个人经历等因素的影响。了解个体间的这种 variability是揭示人类在面对不确定性和模糊性时依赖的知觉和决策机制的关键。我们提出了一种计算框架 BAM（边界对齐与操控框架），结合人工神经网络的知觉边界采样和人类行为实验，系统地研究这一现象。我们的知觉边界采样算法生成了沿人工神经网络决策边界的刺激，这些刺激内在地引起了显著的知觉变异。这些刺激的有效性通过涉及246名参与者和116,715次试次的大规模行为实验经验性验证，最终形成了包含19,943张系统标注图像的variMNIST数据集。通过个性化模型对齐和对抗生成，我们建立了一种同时预测和操控配对参与者发散知觉决策的可靠方法。这项工作填补了计算模型与人类个体差异研究之间的空白，提供了个性化知觉分析的新工具。 

---
# OSUniverse: Benchmark for Multimodal GUI-navigation AI Agents 

**Title (ZH)**: OSUniverse：多模态GUI导航AI代理的基准测试 

**Authors**: Mariya Davydova, Daniel Jeffries, Patrick Barker, Arturo Márquez Flores, Sinéad Ryan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03570)  

**Abstract**: In this paper, we introduce OSUniverse: a benchmark of complex, multimodal desktop-oriented tasks for advanced GUI-navigation AI agents that focuses on ease of use, extensibility, comprehensive coverage of test cases, and automated validation. We divide the tasks in increasing levels of complexity, from basic precision clicking to multistep, multiapplication tests requiring dexterity, precision, and clear thinking from the agent. In version one of the benchmark, presented here, we have calibrated the complexity of the benchmark test cases to ensure that the SOTA (State of the Art) agents (at the time of publication) do not achieve results higher than 50%, while the average white collar worker can perform all these tasks with perfect accuracy. The benchmark can be scored manually, but we also introduce an automated validation mechanism that has an average error rate less than 2%. Therefore, this benchmark presents solid ground for fully automated measuring of progress, capabilities and the effectiveness of GUI-navigation AI agents over the short and medium-term horizon. The source code of the benchmark is available at this https URL. 

**Abstract (ZH)**: OSUniverse：面向高级GUI导航AI代理的复杂多模态桌面任务基准 

---
# A Hashgraph-Inspired Consensus Mechanism for Reliable Multi-Model Reasoning 

**Title (ZH)**: 基于哈希图启发的可靠多模型推理共识机制 

**Authors**: Kolawole E. Ogunsina, Morayo A. Ogunsina  

**Link**: [PDF](https://arxiv.org/pdf/2505.03553)  

**Abstract**: Inconsistent outputs and hallucinations from large language models (LLMs) are major obstacles to reliable AI systems. When different proprietary reasoning models (RMs), such as those by OpenAI, Google, Anthropic, DeepSeek, and xAI, are given the same complex request, they often produce divergent results due to variations in training and inference. This paper proposes a novel consensus mechanism, inspired by distributed ledger technology, to validate and converge these outputs, treating each RM as a black-box peer. Building on the Hashgraph consensus algorithm, our approach employs gossip-about-gossip communication and virtual voting to achieve agreement among an ensemble of RMs. We present an architectural design for a prototype system in which RMs iteratively exchange and update their answers, using information from each round to improve accuracy and confidence in subsequent rounds. This approach goes beyond simple majority voting by incorporating the knowledge and cross-verification content of every model. We justify the feasibility of this Hashgraph-inspired consensus for AI ensembles and outline its advantages over traditional ensembling techniques in reducing nonfactual outputs. Preliminary considerations for implementation, evaluation criteria for convergence and accuracy, and potential challenges are discussed. The proposed mechanism demonstrates a promising direction for multi-agent AI systems to self-validate and deliver high-fidelity responses in complex tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）的不一致输出和幻觉是可靠AI系统的主要障碍。本论文提出了一种受分布式账本技术启发的新型共识机制，用于验证和收敛这些输出，将每个推理模型（RM）视为黑盒 peer。基于 Hashgraph 共识算法，我们的方法采用辗转相告通信和虚拟投票来达成一组 RMs 之间的共识。我们提出了一种原型系统的架构设计，在该设计中，RMs 逐步交换和更新答案，并利用每轮的反馈信息改进后续轮次的准确性和信心。这种方法通过整合每个模型的知识和交叉验证内容，超越了简单的多数表决。我们论证了这种受 Hashgraph 启发的共识机制在AI集成系统中的可行性，并概述了它在减少非事实输出方面相对于传统集成技术的优势。讨论了实施的初步考虑、收敛和准确性的评估标准，以及潜在挑战。所提机制为多Agent AI系统在复杂任务中自我验证并提供高保真响应指明了一条有希望的方向。 

---
# STORY2GAME: Generating (Almost) Everything in an Interactive Fiction Game 

**Title (ZH)**: STORY2GAME：生成交互式 fiction 游戏中几乎一切内容 

**Authors**: Eric Zhou, Shreyas Basavatia, Moontashir Siam, Zexin Chen, Mark O. Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2505.03547)  

**Abstract**: We introduce STORY2GAME, a novel approach to using Large Language Models to generate text-based interactive fiction games that starts by generating a story, populates the world, and builds the code for actions in a game engine that enables the story to play out interactively. Whereas a given set of hard-coded actions can artificially constrain story generation, the ability to generate actions means the story generation process can be more open-ended but still allow for experiences that are grounded in a game state. The key to successful action generation is to use LLM-generated preconditions and effects of actions in the stories as guides for what aspects of the game state must be tracked and changed by the game engine when a player performs an action. We also introduce a technique for dynamically generating new actions to accommodate the player's desire to perform actions that they think of that are not part of the story. Dynamic action generation may require on-the-fly updates to the game engine's state representation and revision of previously generated actions. We evaluate the success rate of action code generation with respect to whether a player can interactively play through the entire generated story. 

**Abstract (ZH)**: 我们介绍了一种名为STORY2GAME的新型方法，该方法利用大型语言模型生成基于文本的交互式小说游戏，首先生成一个故事，填充世界，并在游戏引擎中构建代码以实现故事的互动播放。与硬编码的一组固定动作可能人为地限制故事生成不同，能够生成动作意味着故事生成过程可以更加开放，但仍能确保体验与游戏状态紧密结合。成功生成动作的关键在于使用大语言模型生成的动作的前提条件和效果作为游戏中必须跟踪和改变状态的指南。我们还介绍了一种动态生成新动作的技术，以适应玩家希望执行故事中未包含的动作的需求。动态生成动作可能需要游戏引擎状态表示的实时更新以及之前生成的动作的修订。我们以玩家能否完整互动地游玩整个生成的故事来评估动作代码生成的成功率。 

---
# am-ELO: A Stable Framework for Arena-based LLM Evaluation 

**Title (ZH)**: Arena-based LLM评估的稳定框架：am-ELO 

**Authors**: Zirui Liu, Jiatong Li, Yan Zhuang, Qi Liu, Shuanghong Shen, Jie Ouyang, Mingyue Cheng, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03475)  

**Abstract**: Arena-based evaluation is a fundamental yet significant evaluation paradigm for modern AI models, especially large language models (LLMs). Existing framework based on ELO rating system suffers from the inevitable instability problem due to ranking inconsistency and the lack of attention to the varying abilities of annotators. In this paper, we introduce a novel stable arena framework to address these issues by enhancing the ELO Rating System. Specifically, we replace the iterative update method with a Maximum Likelihood Estimation (MLE) approach, m-ELO, and provide theoretical proof of the consistency and stability of the MLE approach for model ranking. Additionally, we proposed the am-ELO, which modify the Elo Rating's probability function to incorporate annotator abilities, enabling the simultaneous estimation of model scores and annotator reliability. Experiments demonstrate that this method ensures stability, proving that this framework offers a more robust, accurate, and stable evaluation method for LLMs. 

**Abstract (ZH)**: 基于竞技场的评估是一种对于现代AI模型，尤其是大型语言模型（LLMs）的基本而重要的评估范式。基于ELO评分系统的现有框架由于排名不一致性和忽视注释员能力变化而不可避免地存在稳定性问题。在本文中，我们引入了一种新颖的稳定竞技场框架，通过增强ELO评分系统来解决这些问题。具体地，我们用最大似然估计（MLE）方法替代迭代更新方法，并提供了MLE方法在模型排名中一致性和稳定性的理论证明。此外，我们提出了am-ELO，通过修改ELO评分的概率函数来整合注释员的能力，使得可以同时估计模型得分和注释员的可靠性。实验表明，该方法确保了稳定性，证明了该框架提供了一种更稳健、准确且稳定的LLM评估方法。 

---
# The Steganographic Potentials of Language Models 

**Title (ZH)**: 语言模型的隐写 potentials 翻译为：语言模型的隐写潜力 

**Authors**: Artem Karpov, Tinuade Adeleke, Seong Hah Cho, Natalia Perez-Campanero  

**Link**: [PDF](https://arxiv.org/pdf/2505.03439)  

**Abstract**: The potential for large language models (LLMs) to hide messages within plain text (steganography) poses a challenge to detection and thwarting of unaligned AI agents, and undermines faithfulness of LLMs reasoning. We explore the steganographic capabilities of LLMs fine-tuned via reinforcement learning (RL) to: (1) develop covert encoding schemes, (2) engage in steganography when prompted, and (3) utilize steganography in realistic scenarios where hidden reasoning is likely, but not prompted. In these scenarios, we detect the intention of LLMs to hide their reasoning as well as their steganography performance. Our findings in the fine-tuning experiments as well as in behavioral non fine-tuning evaluations reveal that while current models exhibit rudimentary steganographic abilities in terms of security and capacity, explicit algorithmic guidance markedly enhances their capacity for information concealment. 

**Abstract (ZH)**: 大型语言模型在明文中隐藏信息（隐写）的潜在能力对检测和抵御未对齐的AI代理构成挑战，并损害了大型语言模型推理的可信度。我们探讨了通过强化学习（RL）微调的大型语言模型的隐写能力：（1）开发隐蔽编码方案，（2）在提示时进行隐写，以及（3）在隐藏推理很可能但未提示的现实场景中利用隐写。在这些场景中，我们检测到大型语言模型隐藏推理的意图及其隐写性能。在微调实验以及行为非微调评估中的发现表明，尽管当前模型在安全性与容量方面表现出初步的隐写能力，但明确的算法指导显著增强了信息隐藏的能力。 

---
# Procedural Memory Is Not All You Need: Bridging Cognitive Gaps in LLM-Based Agents 

**Title (ZH)**: 程序记忆并非足矣：连接基于LLM的代理的认知缺口 

**Authors**: Schaun Wheeler, Olivier Jeunen  

**Link**: [PDF](https://arxiv.org/pdf/2505.03434)  

**Abstract**: Large Language Models (LLMs) represent a landmark achievement in Artificial Intelligence (AI), demonstrating unprecedented proficiency in procedural tasks such as text generation, code completion, and conversational coherence. These capabilities stem from their architecture, which mirrors human procedural memory -- the brain's ability to automate repetitive, pattern-driven tasks through practice. However, as LLMs are increasingly deployed in real-world applications, it becomes impossible to ignore their limitations operating in complex, unpredictable environments. This paper argues that LLMs, while transformative, are fundamentally constrained by their reliance on procedural memory. To create agents capable of navigating ``wicked'' learning environments -- where rules shift, feedback is ambiguous, and novelty is the norm -- we must augment LLMs with semantic memory and associative learning systems. By adopting a modular architecture that decouples these cognitive functions, we can bridge the gap between narrow procedural expertise and the adaptive intelligence required for real-world problem-solving. 

**Abstract (ZH)**: 大规模语言模型（LLMs）代表了人工智能（AI）领域的里程碑成就，展示了在文本生成、代码补全和对话连贯性等程序性任务上的无precedent的能效。这些能力源自于它们的架构，这种架构模仿了人类程序性记忆——大脑通过练习自动化重复的、基于模式的任务的能力。然而，随着LLMs在真实世界应用中的日益部署，它们在复杂、不可预测环境中的局限性变得无法忽视。本文认为，尽管LLMs具有革命性意义，但它们从根本上依赖于程序性记忆而受到限制。要创建能够导航“棘手”学习环境中的代理——在这些环境中规则会变化、反馈具有模糊性且新颖性是常态——我们必须增强LLMs以具备语义记忆和关联学习系统。通过采用模块化架构解耦这些认知功能，我们可以弥合狭隘的程序性专长相对于适应性智能在现实世界问题解决中的需求之间的差距。 

---
# Validating the Effectiveness of a Large Language Model-based Approach for Identifying Children's Development across Various Free Play Settings in Kindergarten 

**Title (ZH)**: 基于大型语言模型的方法在幼儿园不同自由玩耍情境下识别儿童发展的有效性验证 

**Authors**: Yuanyuan Yang, Yuan Shen, Tianchen Sun, Yangbin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.03369)  

**Abstract**: Free play is a fundamental aspect of early childhood education, supporting children's cognitive, social, emotional, and motor development. However, assessing children's development during free play poses significant challenges due to the unstructured and spontaneous nature of the activity. Traditional assessment methods often rely on direct observations by teachers, parents, or researchers, which may fail to capture comprehensive insights from free play and provide timely feedback to educators. This study proposes an innovative approach combining Large Language Models (LLMs) with learning analytics to analyze children's self-narratives of their play experiences. The LLM identifies developmental abilities, while performance scores across different play settings are calculated using learning analytics techniques. We collected 2,224 play narratives from 29 children in a kindergarten, covering four distinct play areas over one semester. According to the evaluation results from eight professionals, the LLM-based approach achieved high accuracy in identifying cognitive, motor, and social abilities, with accuracy exceeding 90% in most domains. Moreover, significant differences in developmental outcomes were observed across play settings, highlighting each area's unique contributions to specific abilities. These findings confirm that the proposed approach is effective in identifying children's development across various free play settings. This study demonstrates the potential of integrating LLMs and learning analytics to provide child-centered insights into developmental trajectories, offering educators valuable data to support personalized learning and enhance early childhood education practices. 

**Abstract (ZH)**: 自由游戏是幼儿教育的基本方面，支持儿童的认知、社会、情感和运动发展。然而，评估儿童在自由游戏中的发展由于活动的无结构和自发性而面临重大挑战。传统评估方法通常依赖于教师、家长或研究者的直接观察，这可能无法全面捕捉自由游戏的洞察并为教育者提供及时反馈。本研究提出了一种结合大型语言模型（LLMs）与学习分析的方法，以分析儿童对游戏体验的自我叙述。LLM识别发展能力，不同游戏环境下的表现分数则通过学习分析技术进行计算。我们在一所 kindergarten 收集了29名儿童在四个不同游戏区域为期一学期的2224份游戏叙述。根据八名专业人士的评估结果，基于LLM的方法在识别认知、运动和社会能力方面达到了较高的准确性，大多数领域超过90%的准确率。此外，还观察到不同游戏环境下的发展结果存在显著差异，突显了每个区域对特定能力的独特贡献。这些发现证实该方法有效识别不同自由游戏环境中的儿童发展。本研究展示了将LLMs与学习分析结合的潜力，以提供以儿童为中心的发展轨迹洞察，为教育者提供有价值的数据支持个性化学习并提升幼儿教育实践。 

---
# Domain Adversarial Training for Mitigating Gender Bias in Speech-based Mental Health Detection 

**Title (ZH)**: 基于语域对抗训练减轻语音情感健康检测中性别偏见 

**Authors**: June-Woo Kim, Haram Yoon, Wonkyo Oh, Dawoon Jung, Sung-Hoon Yoon, Dae-Jin Kim, Dong-Ho Lee, Sang-Yeol Lee, Chan-Mo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03359)  

**Abstract**: Speech-based AI models are emerging as powerful tools for detecting depression and the presence of Post-traumatic stress disorder (PTSD), offering a non-invasive and cost-effective way to assess mental health. However, these models often struggle with gender bias, which can lead to unfair and inaccurate predictions. In this study, our study addresses this issue by introducing a domain adversarial training approach that explicitly considers gender differences in speech-based depression and PTSD detection. Specifically, we treat different genders as distinct domains and integrate this information into a pretrained speech foundation model. We then validate its effectiveness on the E-DAIC dataset to assess its impact on performance. Experimental results show that our method notably improves detection performance, increasing the F1-score by up to 13.29 percentage points compared to the baseline. This highlights the importance of addressing demographic disparities in AI-driven mental health assessment. 

**Abstract (ZH)**: 基于语音的AI模型在检测抑郁和创伤后应激障碍（PTSD）方面的应用正逐渐成为有力工具，提供了无侵入性和成本效益高的心理健康评估方式。然而，这些模型常常面临性别偏见问题，可能导致不公平和不准确的预测。本研究通过引入领域对抗训练方法来解决这一问题，该方法明确考虑了语音中抑郁和PTSD检测的性别差异。具体而言，我们将不同性别视为不同的领域，并将此信息整合到预训练的语音基础模型中。然后，我们在E-DAIC数据集上验证其有效性，评估其对性能的影响。实验结果表明，我们的方法显著提高了检测性能，F1分数相比基线提高了13.29个百分点。这突显了在AI驱动的心理健康评估中解决人口统计学差异的重要性。 

---
# AI-Driven Scholarly Peer Review via Persistent Workflow Prompting, Meta-Prompting, and Meta-Reasoning 

**Title (ZH)**: 基于持续工作流提示、元提示和元推理的AI驱动学术同行评审 

**Authors**: Evgeny Markhasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03332)  

**Abstract**: Critical peer review of scientific manuscripts presents a significant challenge for Large Language Models (LLMs), partly due to data limitations and the complexity of expert reasoning. This report introduces Persistent Workflow Prompting (PWP), a potentially broadly applicable prompt engineering methodology designed to bridge this gap using standard LLM chat interfaces (zero-code, no APIs). We present a proof-of-concept PWP prompt for the critical analysis of experimental chemistry manuscripts, featuring a hierarchical, modular architecture (structured via Markdown) that defines detailed analysis workflows. We develop this PWP prompt through iterative application of meta-prompting techniques and meta-reasoning aimed at systematically codifying expert review workflows, including tacit knowledge. Submitted once at the start of a session, this PWP prompt equips the LLM with persistent workflows triggered by subsequent queries, guiding modern reasoning LLMs through systematic, multimodal evaluations. Demonstrations show the PWP-guided LLM identifying major methodological flaws in a test case while mitigating LLM input bias and performing complex tasks, including distinguishing claims from evidence, integrating text/photo/figure analysis to infer parameters, executing quantitative feasibility checks, comparing estimates against claims, and assessing a priori plausibility. To ensure transparency and facilitate replication, we provide full prompts, detailed demonstration analyses, and logs of interactive chats as supplementary resources. Beyond the specific application, this work offers insights into the meta-development process itself, highlighting the potential of PWP, informed by detailed workflow formalization, to enable sophisticated analysis using readily available LLMs for complex scientific tasks. 

**Abstract (ZH)**: 持久工作流提示：一种用于大型语言模型的批判性同行评审方法 

---
# Artificial Behavior Intelligence: Technology, Challenges, and Future Directions 

**Title (ZH)**: 人工行为智能：技术、挑战及未来方向 

**Authors**: Kanghyun Jo, Jehwan Choi, Kwanho Kim, Seongmin Kim, Duy-Linh Nguyen, Xuan-Thuy Vo, Adri Priadana, Tien-Dat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2505.03315)  

**Abstract**: Understanding and predicting human behavior has emerged as a core capability in various AI application domains such as autonomous driving, smart healthcare, surveillance systems, and social robotics. This paper defines the technical framework of Artificial Behavior Intelligence (ABI), which comprehensively analyzes and interprets human posture, facial expressions, emotions, behavioral sequences, and contextual cues. It details the essential components of ABI, including pose estimation, face and emotion recognition, sequential behavior analysis, and context-aware modeling. Furthermore, we highlight the transformative potential of recent advances in large-scale pretrained models, such as large language models (LLMs), vision foundation models, and multimodal integration models, in significantly improving the accuracy and interpretability of behavior recognition. Our research team has a strong interest in the ABI domain and is actively conducting research, particularly focusing on the development of intelligent lightweight models capable of efficiently inferring complex human behaviors. This paper identifies several technical challenges that must be addressed to deploy ABI in real-world applications including learning behavioral intelligence from limited data, quantifying uncertainty in complex behavior prediction, and optimizing model structures for low-power, real-time inference. To tackle these challenges, our team is exploring various optimization strategies including lightweight transformers, graph-based recognition architectures, energy-aware loss functions, and multimodal knowledge distillation, while validating their applicability in real-time environments. 

**Abstract (ZH)**: 理解与预测人类行为已成为自动驾驶、智能医疗、监控系统和社会机器人等领域的核心能力。本文定义了人工行为智能（ABI）的技术框架，全面分析和解释了人类的姿态、面部表情、情绪、行为序列和上下文线索。详细阐述了ABI的关键组成部分，包括姿态估计、面部和情绪识别、序列行为分析以及上下文感知建模。此外，本文还强调了大规模预训练模型（如大型语言模型、视觉基础模型和多模态集成模型）的最新进展在大幅提高行为识别的准确性和可解释性方面的潜力。我们的研究团队对ABI领域非常感兴趣，并积极进行研究，特别是侧重于开发高效推理复杂人类行为的智能轻量级模型。本文指出了在实际应用中部署ABI所必须解决的技术挑战，包括从有限数据中学习行为智能、在复杂行为预测中量化不确定性以及优化低功耗、实时推理的模型结构。为了应对这些挑战，我们的团队正在探索各种优化策略，包括轻量级变换器、基于图的识别架构、能量感知损失函数以及多模态知识精简技术，并在实时环境中验证其适用性。 

---
# Capability-Driven Skill Generation with LLMs: A RAG-Based Approach for Reusing Existing Libraries and Interfaces 

**Title (ZH)**: 基于RAG的方法：面向能力的技能生成——重用现有库和接口的LLM驱动 approach 

**Authors**: Luis Miguel Vieira da Silva, Aljosha Köcher, Nicolas König, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2505.03295)  

**Abstract**: Modern automation systems increasingly rely on modular architectures, with capabilities and skills as one solution approach. Capabilities define the functions of resources in a machine-readable form and skills provide the concrete implementations that realize those capabilities. However, the development of a skill implementation conforming to a corresponding capability remains a time-consuming and challenging task. In this paper, we present a method that treats capabilities as contracts for skill implementations and leverages large language models to generate executable code based on natural language user input. A key feature of our approach is the integration of existing software libraries and interface technologies, enabling the generation of skill implementations across different target languages. We introduce a framework that allows users to incorporate their own libraries and resource interfaces into the code generation process through a retrieval-augmented generation architecture. The proposed method is evaluated using an autonomous mobile robot controlled via Python and ROS 2, demonstrating the feasibility and flexibility of the approach. 

**Abstract (ZH)**: 现代自动化系统越来越多地依赖模块化架构，能力与技能为其解决方案之一。能力以机器可读的形式定义资源的功能，而技能则提供实现这些能力的具体实现。然而，根据相应能力开发符合规范的技能实现仍然是一个耗时且具有挑战性的工作。本文提出了一种方法，将能力视为技能实现的契约，并利用大型语言模型根据自然语言用户输入生成可执行代码。我们方法的关键特征是整合现有的软件库和接口技术，实现不同目标语言中的技能实现生成。我们引入了一个框架，允许用户通过检索增强生成架构将其自己的库和资源接口纳入代码生成过程。所提出的方法通过一个自主移动机器人（通过Python和ROS 2 控制）进行评估，展示了该方法的可行性和灵活性。 

---
# RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation 

**Title (ZH)**: RAG-MCP：通过检索增强生成减轻大模型工具选择中的提示膨胀问题 

**Authors**: Tiantian Gan, Qiyao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.03275)  

**Abstract**: Large language models (LLMs) struggle to effectively utilize a growing number of external tools, such as those defined by the Model Context Protocol (MCP)\cite{IntroducingMCP}, due to prompt bloat and selection complexity. We introduce RAG-MCP, a Retrieval-Augmented Generation framework that overcomes this challenge by offloading tool discovery. RAG-MCP uses semantic retrieval to identify the most relevant MCP(s) for a given query from an external index before engaging the LLM. Only the selected tool descriptions are passed to the model, drastically reducing prompt size and simplifying decision-making. Experiments, including an MCP stress test, demonstrate RAG-MCP significantly cuts prompt tokens (e.g., by over 50%) and more than triples tool selection accuracy (43.13% vs 13.62% baseline) on benchmark tasks. RAG-MCP enables scalable and accurate tool integration for LLMs. 

**Abstract (ZH)**: Large语言模型（LLMs）难以有效利用不断增长的外部工具，例如由模型上下文协议（MCP）定义的工具，由于提示膨胀和选择复杂性。我们引入了RAG-MCP，这是一种检索增强生成框架，通过卸载工具发现来克服这一挑战。RAG-MCP使用语义检索，从外部索引中识别与给定查询最相关的MCP(s)，然后才让LLM参与。仅将选定的工具描述传递给模型，极大地减少了提示大小并简化了决策过程。实验，包括MCP压力测试，表明RAG-MCP显著减少了提示令牌数量（例如，超过50%）并在基准任务上将工具选择准确性提高了三倍多（43.13% vs 13.62%基线）。RAG-MCP使LLM的可扩展且准确的工具集成成为可能。 

---
# Patterns and Mechanisms of Contrastive Activation Engineering 

**Title (ZH)**: 对比激活工程的模式与机制 

**Authors**: Yixiong Hao, Ayush Panda, Stepan Shabalin, Sheikh Abdur Raheem Ali  

**Link**: [PDF](https://arxiv.org/pdf/2505.03189)  

**Abstract**: Controlling the behavior of Large Language Models (LLMs) remains a significant challenge due to their inherent complexity and opacity. While techniques like fine-tuning can modify model behavior, they typically require extensive computational resources. Recent work has introduced a class of contrastive activation engineering (CAE) techniques as promising approaches for steering LLM outputs through targeted modifications to their internal representations. Applied at inference-time with zero cost, CAE has the potential to introduce a new paradigm of flexible, task-specific LLM behavior tuning. We analyze the performance of CAE in in-distribution, out-of-distribution settings, evaluate drawbacks, and begin to develop comprehensive guidelines for its effective deployment. We find that 1. CAE is only reliably effective when applied to in-distribution contexts. 2. Increasing the number of samples used to generate steering vectors has diminishing returns at around 80 samples. 3. Steering vectors are susceptible to adversarial inputs that reverses the behavior that is steered for. 4. Steering vectors harm the overall model perplexity. 5. Larger models are more resistant to steering-induced degradation. 

**Abstract (ZH)**: 控制大型语言模型的行为仍然是一个重大挑战，由于它们固有的复杂性和不透明性。对比激活工程（CAE）技术作为通过目标修改其内部表示来引导LLM输出的有希望的方法，已在推理时零成本下引入，具有引入灵活、任务特定的LLM行为调优新范式的潜力。我们分析了CAE在分布内和分布外设置中的性能，评估了其缺点，并开始制定全面的指南以促进其有效部署。我们发现：1. CAE仅在应用于分布内上下文中时才可靠有效。2. 用于生成引导向量的样本数量增加到约80个时，其效果递减。3. 引导向量对对抗输入敏感，这些输入会使引导行为反转。4. 引导向量损害了整体模型的困惑度。5. 较大的模型对引导引起的退化更具抵抗力。 

---
# CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics 

**Title (ZH)**: CombiBench: 组合数学领域大语言模型能力基准评测 

**Authors**: Junqi Liu, Xiaohan Lin, Jonas Bayer, Yael Dillies, Weijie Jiang, Xiaodan Liang, Roman Soletskyi, Haiming Wang, Yunzhou Xie, Beibei Xiong, Zhengfeng Yang, Jujian Zhang, Lihong Zhi, Jia Li, Zhengying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03171)  

**Abstract**: Neurosymbolic approaches integrating large language models with formal reasoning have recently achieved human-level performance on mathematics competition problems in algebra, geometry and number theory. In comparison, combinatorics remains a challenging domain, characterized by a lack of appropriate benchmarks and theorem libraries. To address this gap, we introduce CombiBench, a comprehensive benchmark comprising 100 combinatorial problems, each formalized in Lean~4 and paired with its corresponding informal statement. The problem set covers a wide spectrum of difficulty levels, ranging from middle school to IMO and university level, and span over ten combinatorial topics. CombiBench is suitable for testing IMO solving capabilities since it includes all IMO combinatorial problems since 2000 (except IMO 2004 P3 as its statement contain an images). Furthermore, we provide a comprehensive and standardized evaluation framework, dubbed Fine-Eval (for $\textbf{F}$ill-in-the-blank $\textbf{in}$ L$\textbf{e}$an Evaluation), for formal mathematics. It accommodates not only proof-based problems but also, for the first time, the evaluation of fill-in-the-blank questions. Using Fine-Eval as the evaluation method and Kimina Lean Server as the backend, we benchmark several LLMs on CombiBench and observe that their capabilities for formally solving combinatorial problems remain limited. Among all models tested (none of which has been trained for this particular task), Kimina-Prover attains the best results, solving 7 problems (out of 100) under both ``with solution'' and ``without solution'' scenarios. We open source the benchmark dataset alongside with the code of the proposed evaluation method at this https URL. 

**Abstract (ZH)**: 神经符号方法结合大型语言模型与形式推理在组合数学竞赛问题上已达到人类水平，在代数、几何和数论方面取得了进展。相比之下，组合数学仍是具有挑战性的领域，特征为缺乏合适的基准和定理库。为应对这一挑战，我们引入了CombiBench，这是一个包含100个组合数学问题的综合基准，每个问题均用Lean 4形式化，并配以相应的非形式化陈述。问题集涵盖了从小学到国际数学奥林匹克（IMO）和大学水平的广泛难度级别，涵盖十个组合数学主题。CombiBench 适合测试IMO解题能力，因为它包括了2000年以来的所有IMO组合数学问题（除了2004年IMO P3，因其陈述包含图片）。此外，我们还提供了一种全面且标准化的评估框架，名为Fine-Eval（用于Lean的填空评估），用于正式数学的评估。Fine-Eval 不仅适用于证明问题，还首次适用于填空题的评估。使用Fine-Eval 作为评估方法和Kimina Lean Server 作为后端，我们在CombiBench 上测试了几种语言模型，并观察到它们在形式化解决组合数学问题方面的能力仍然有限。在所有测试的模型（其中没有一个专门为此任务训练）中，Kimina-Prover 获得了最佳结果，在“有解”和“无解”两种场景下分别解决了7个问题（总计100个）。我们已经开源了基准数据集及其所提出的评估方法的代码，可通过以下链接访问：this https URL。 

---
# Holmes: Automated Fact Check with Large Language Models 

**Title (ZH)**: Holmes: 使用大规模语言模型的自动事实核查 

**Authors**: Haoran Ou, Gelei Deng, Xingshuo Han, Jie Zhang, Xinlei He, Han Qiu, Shangwei Guo, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03135)  

**Abstract**: The rise of Internet connectivity has accelerated the spread of disinformation, threatening societal trust, decision-making, and national security. Disinformation has evolved from simple text to complex multimodal forms combining images and text, challenging existing detection methods. Traditional deep learning models struggle to capture the complexity of multimodal disinformation. Inspired by advances in AI, this study explores using Large Language Models (LLMs) for automated disinformation detection. The empirical study shows that (1) LLMs alone cannot reliably assess the truthfulness of claims; (2) providing relevant evidence significantly improves their performance; (3) however, LLMs cannot autonomously search for accurate evidence. To address this, we propose Holmes, an end-to-end framework featuring a novel evidence retrieval method that assists LLMs in collecting high-quality evidence. Our approach uses (1) LLM-powered summarization to extract key information from open sources and (2) a new algorithm and metrics to evaluate evidence quality. Holmes enables LLMs to verify claims and generate justifications effectively. Experiments show Holmes achieves 88.3% accuracy on two open-source datasets and 90.2% in real-time verification tasks. Notably, our improved evidence retrieval boosts fact-checking accuracy by 30.8% over existing methods 

**Abstract (ZH)**: 互联网连接的普及加速了虚假信息的传播，威胁着社会信任、决策和国家安全。虚假信息从简单的文本发展为结合图像和文本的复杂多媒体形式，挑战现有的检测方法。传统的深度学习模型难以捕获多媒体虚假信息的复杂性。受人工智能advance的启发，本研究探讨了使用大型语言模型（LLMs）进行自动化虚假信息检测的可能性。实证研究显示：（1）LLMs单独使用不能可靠地评估声明的真实性；（2）提供相关证据能显著提高其性能；（3）然而，LLMs不能自主搜索准确的证据。为此，我们提出了Holmes框架，这是一种端到端框架，具有新颖的证据检索方法，协助LLMs收集高质量的证据。我们的方法包括：（1）利用LLM进行总结，从开源资源中提取关键信息；（2）提出新的算法和指标来评估证据质量。Holmes使LLMs能够有效地验证声明并生成说明。实验结果显示，Holmes在两个开源数据集上的准确率为88.3%，在实时验证任务中的准确率为90.2%。值得注意的是，我们改进的证据检索方法使事实核查的准确性提高了30.8%。 

---
# Is AI currently capable of identifying wild oysters? A comparison of human annotators against the AI model, ODYSSEE 

**Title (ZH)**: AI当前是否有能力识别野生牡蛎？人类标注员与ODYSSEE模型的比较研究 

**Authors**: Brendan Campbell, Alan Williams, Kleio Baxevani, Alyssa Campbell, Rushabh Dhoke, Rileigh E. Hudock, Xiaomin Lin, Vivek Mange, Bernhard Neuberger, Arjun Suresh, Alhim Vera, Arthur Trembanis, Herbert G. Tanner, Edward Hale  

**Link**: [PDF](https://arxiv.org/pdf/2505.03108)  

**Abstract**: Oysters are ecologically and commercially important species that require frequent monitoring to track population demographics (e.g. abundance, growth, mortality). Current methods of monitoring oyster reefs often require destructive sampling methods and extensive manual effort. Therefore, they are suboptimal for small-scale or sensitive environments. A recent alternative, the ODYSSEE model, was developed to use deep learning techniques to identify live oysters using video or images taken in the field of oyster reefs to assess abundance. The validity of this model in identifying live oysters on a reef was compared to expert and non-expert annotators. In addition, we identified potential sources of prediction error. Although the model can make inferences significantly faster than expert and non-expert annotators (39.6 s, $2.34 \pm 0.61$ h, $4.50 \pm 1.46$ h, respectively), the model overpredicted the number of live oysters, achieving lower accuracy (63\%) in identifying live oysters compared to experts (74\%) and non-experts (75\%) alike. Image quality was an important factor in determining the accuracy of the model and the annotators. Better quality images improved human accuracy and worsened model accuracy. Although ODYSSEE was not sufficiently accurate, we anticipate that future training on higher-quality images, utilizing additional live imagery, and incorporating additional annotation training classes will greatly improve the model's predictive power based on the results of this analysis. Future research should address methods that improve the detection of living vs. dead oysters. 

**Abstract (ZH)**: 牡蛎是具有重要生态和商业价值的物种，需要频繁监测以追踪种群动态（如丰度、生长、死亡率）。目前牡蛎礁的监测方法往往需要破坏性采样和大量的手工努力。因此，这些方法对于小型或敏感环境并不理想。最近一种替代方案，ODYSSEE模型，利用深度学习技术通过现场拍摄的牡蛎礁视频或图像识别活牡蛎，以评估丰度。将该模型在识别礁上活牡蛎方面的有效性与专家和非专家标注员进行了比较，并确定了预测误差的潜在来源。虽然该模型比专家和非专家标注员快得多（分别为39.6秒，2.34±0.61小时，4.50±1.46小时），但它高估了活牡蛎的数量，在识别活牡蛎的准确性上低于专家（74%）和非专家（75%）的73%。图像质量是决定模型和标注员准确性的重要因素。高质量的图像提高了人类准确性和降低了模型准确性。尽管ODYSSEE不够准确，我们预计根据本次分析的结果，通过在更高质量图像上进行进一步训练，利用更多的活体影像，并结合额外的标注训练类别，将显著提高模型的预测能力。未来的研究应解决提高活牡蛎与死牡蛎检测的方法。 

---
# BLAB: Brutally Long Audio Bench 

**Title (ZH)**: BRAB: 剧烈长音频基准 

**Authors**: Orevaoghene Ahia, Martijn Bartelds, Kabir Ahuja, Hila Gonen, Valentin Hofmann, Siddhant Arora, Shuyue Stella Li, Vishal Puttagunta, Mofetoluwa Adeyemi, Charishma Buchireddy, Ben Walls, Noah Bennett, Shinji Watanabe, Noah A. Smith, Yulia Tsvetkov, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03054)  

**Abstract**: Developing large audio language models (LMs) capable of understanding diverse spoken interactions is essential for accommodating the multimodal nature of human communication and can increase the accessibility of language technologies across different user populations. Recent work on audio LMs has primarily evaluated their performance on short audio segments, typically under 30 seconds, with limited exploration of long-form conversational speech segments that more closely reflect natural user interactions with these models. We introduce Brutally Long Audio Bench (BLAB), a challenging long-form audio benchmark that evaluates audio LMs on localization, duration estimation, emotion, and counting tasks using audio segments averaging 51 minutes in length. BLAB consists of 833+ hours of diverse, full-length audio clips, each paired with human-annotated, text-based natural language questions and answers. Our audio data were collected from permissively licensed sources and underwent a human-assisted filtering process to ensure task compliance. We evaluate six open-source and proprietary audio LMs on BLAB and find that all of them, including advanced models such as Gemini 2.0 Pro and GPT-4o, struggle with the tasks in BLAB. Our comprehensive analysis reveals key insights into the trade-offs between task difficulty and audio duration. In general, we find that audio LMs struggle with long-form speech, with performance declining as duration increases. They perform poorly on localization, temporal reasoning, counting, and struggle to understand non-phonemic information, relying more on prompts than audio content. BLAB serves as a challenging evaluation framework to develop audio LMs with robust long-form audio understanding capabilities. 

**Abstract (ZH)**: 开发能够在长时间音频对话中理解多样口语交互的大规模音频语言模型对于适应人类多模态通信的性质并提高语言技术在不同用户群体中的 accessibility 至关重要。近年来，关于音频语言模型的研究主要评估其在短音频片段上的性能，通常少于30秒，对更长的对话音频片段的研究有限，这类音频片段更接近于自然用户与这些模型的互动。我们引入了Brutally Long Audio Bench (BLAB) 挑战性长格式音频基准，该基准使用平均时长为51分钟的音频片段评估音频语言模型在定位、时长估计、情绪识别和计数任务上的表现。BLAB 包含833+小时的多样化全长度音频剪辑，每个剪辑配有人工标注的基于文本的自然语言问题和答案。我们从许可来源收集了音频数据，并通过人工辅助的筛选过程确保任务合规。我们在 BLAB 上评估了六个开源和专有音频语言模型，发现所有模型，包括高级模型如 Gemini 2.0 Pro 和 GPT-4o，在 BLAB 上的这些任务中都表现出困难。我们的全面分析揭示了任务难度和音频时长之间的关键权衡。总体而言，我们发现音频语言模型在处理长格式语音时面临困难，随着时长增加，其性能下降。它们在定位、时间推理和计数方面表现不佳，难以理解非语音信息，更多依赖于提示而非音频内容。BLAB 作为一项具有挑战性的评估框架，旨在开发具有稳健长格式音频理解能力的音频语言模型。 

---
# Evaluating the Impact of AI-Powered Audiovisual Personalization on Learner Emotion, Focus, and Learning Outcomes 

**Title (ZH)**: 评估人工智能驱动的音视频个性化对学习者情绪、专注度和学习成果的影响 

**Authors**: George Xi Wang, Jingying Deng, Safinah Ali  

**Link**: [PDF](https://arxiv.org/pdf/2505.03033)  

**Abstract**: Independent learners often struggle with sustaining focus and emotional regulation in unstructured or distracting settings. Although some rely on ambient aids such as music, ASMR, or visual backgrounds to support concentration, these tools are rarely integrated into cohesive, learner-centered systems. Moreover, existing educational technologies focus primarily on content adaptation and feedback, overlooking the emotional and sensory context in which learning takes place. Large language models have demonstrated powerful multimodal capabilities including the ability to generate and adapt text, audio, and visual content. Educational research has yet to fully explore their potential in creating personalized audiovisual learning environments. To address this gap, we introduce an AI-powered system that uses LLMs to generate personalized multisensory study environments. Users select or generate customized visual themes (e.g., abstract vs. realistic, static vs. animated) and auditory elements (e.g., white noise, ambient ASMR, familiar vs. novel sounds) to create immersive settings aimed at reducing distraction and enhancing emotional stability. Our primary research question investigates how combinations of personalized audiovisual elements affect learner cognitive load and engagement. Using a mixed-methods design that incorporates biometric measures and performance outcomes, this study evaluates the effectiveness of LLM-driven sensory personalization. The findings aim to advance emotionally responsive educational technologies and extend the application of multimodal LLMs into the sensory dimension of self-directed learning. 

**Abstract (ZH)**: 独立学习者往往在无结构或有干扰的环境中难以维持专注力和情绪调节。尽管有些人依赖环境辅助工具，如音乐、ASMR或视觉背景来支持集中注意力，但这些工具很少被整合到以学习者为中心的系统中。此外，现有的教育技术主要集中在内容适应和反馈上，忽略了学习过程中所处的情感和感官背景。大规模语言模型展示了强大的多模态能力，包括生成和适应文本、音频和视觉内容的能力。教育研究尚未充分探索在创建个性化的视听学习环境方面的潜力。为填补这一空白，我们介绍了一个基于AI的系统，该系统使用LLM生成个性化的多感官学习环境。用户可以选择或生成定制的视觉主题（例如，抽象 vs. 现实，静态 vs. 动画）和听觉元素（例如，白噪音、环境ASMR、熟悉 vs. 新奇声音），以创建减少干扰并增强情绪稳定的沉浸式环境。我们主要的研究问题是探讨个性化视听元素的不同组合如何影响学习者的认知负荷和参与度。本研究采用结合生物测量和绩效结果的混合方法设计，评估LLM驱动的感官个性化效果。研究结果旨在推动具有情感响应性的教育技术，并将多模态LLM的应用扩展到自主学习的感官维度。 

---
# The Multimodal Paradox: How Added and Missing Modalities Shape Bias and Performance in Multimodal AI 

**Title (ZH)**: 多模态悖论：增加和缺失的模态如何塑造多模态AI中的偏差与性能 

**Authors**: Kishore Sampath, Pratheesh, Ayaazuddin Mohammad, Resmi Ramachandranpillai  

**Link**: [PDF](https://arxiv.org/pdf/2505.03020)  

**Abstract**: Multimodal learning, which integrates diverse data sources such as images, text, and structured data, has proven superior to unimodal counterparts in high-stakes decision-making. However, while performance gains remain the gold standard for evaluating multimodal systems, concerns around bias and robustness are frequently overlooked. In this context, this paper explores two key research questions (RQs): (i) RQ1 examines whether adding a modality con-sistently enhances performance and investigates its role in shaping fairness measures, assessing whether it mitigates or amplifies bias in multimodal models; (ii) RQ2 investigates the impact of missing modalities at inference time, analyzing how multimodal models generalize in terms of both performance and fairness. Our analysis reveals that incorporating new modalities during training consistently enhances the performance of multimodal models, while fairness trends exhibit variability across different evaluation measures and datasets. Additionally, the absence of modalities at inference degrades performance and fairness, raising concerns about its robustness in real-world deployment. We conduct extensive experiments using multimodal healthcare datasets containing images, time series, and structured information to validate our findings. 

**Abstract (ZH)**: 多模态学习，融合图像、文本和结构化数据等多样化的数据源，在高风险决策中已证明优于单模态系统。然而，尽管性能提升仍然是评估多模态系统的主要标准，但关于偏见和鲁棒性的问题却经常被忽视。在此背景下，本文探讨了两个关键研究问题（RQs）：（i）RQ1探讨添加模态是否始终能增强性能，并研究其在塑造公平性指标中的作用，评估它是否减轻或放大了多模态模型中的偏见；（ii）RQ2探讨推理过程中缺失模态的影响，分析多模态模型在性能和公平性上的泛化能力。我们的分析表明，在训练过程中纳入新模态始终能提升多模态模型的性能，而公平性趋势在不同评估指标和数据集中表现出差异性。此外，推理过程中缺失模态会降低性能和公平性，反映出其在实际部署中的鲁棒性问题。我们使用包含图像、时间序列和结构化信息的多模态医疗数据集进行了大量实验以验证我们的发现。 

---
# Iterative Resolution of Prompt Ambiguities Using a Progressive Cutting-Search Approach 

**Title (ZH)**: 逐步切割-搜索方法解决提示歧义的迭代求解 

**Authors**: Fabrizio Marozzo  

**Link**: [PDF](https://arxiv.org/pdf/2505.02952)  

**Abstract**: Generative AI systems have revolutionized human interaction by enabling natural language-based coding and problem solving. However, the inherent ambiguity of natural language often leads to imprecise instructions, forcing users to iteratively test, correct, and resubmit their prompts. We propose an iterative approach that systematically narrows down these ambiguities through a structured series of clarification questions and alternative solution proposals, illustrated with input/output examples as well. Once every uncertainty is resolved, a final, precise solution is generated. Evaluated on a diverse dataset spanning coding, data analysis, and creative writing, our method demonstrates superior accuracy, competitive resolution times, and higher user satisfaction compared to conventional one-shot solutions, which typically require multiple manual iterations to achieve a correct output. 

**Abstract (ZH)**: 生成式AI系统通过基于自然语言的编码和问题解决重塑了人机交互，然而自然语言的固有模糊性常导致不精确的指令，迫使用户反复测试、修正并重新提交提示。我们提出了一种系统性的迭代方法，通过结构化的一系列澄清问题和替代方案提案逐步缩小模糊性，同时用输入/输出示例进行说明。待所有不确定性解决后，最终生成一个精确的解决方案。在涵盖编程、数据分析和创造性写作等多种数据集上的评估表明，我们的方法在准确性、响应时间以及用户满意度方面均优于传统的单次解决方案，后者通常需要多轮手动迭代才能获得正确的输出结果。 

---
# VITA-Audio: Fast Interleaved Cross-Modal Token Generation for Efficient Large Speech-Language Model 

**Title (ZH)**: VITA-音频：快速交错跨模态令牌生成以实现高效大型语音-语言模型 

**Authors**: Zuwei Long, Yunhang Shen, Chaoyou Fu, Heting Gao, Lijiang Li, Peixian Chen, Mengdan Zhang, Hang Shao, Jian Li, Jinlong Peng, Haoyu Cao, Ke Li, Rongrong Ji, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.03739)  

**Abstract**: With the growing requirement for natural human-computer interaction, speech-based systems receive increasing attention as speech is one of the most common forms of daily communication. However, the existing speech models still experience high latency when generating the first audio token during streaming, which poses a significant bottleneck for deployment. To address this issue, we propose VITA-Audio, an end-to-end large speech model with fast audio-text token generation. Specifically, we introduce a lightweight Multiple Cross-modal Token Prediction (MCTP) module that efficiently generates multiple audio tokens within a single model forward pass, which not only accelerates the inference but also significantly reduces the latency for generating the first audio in streaming scenarios. In addition, a four-stage progressive training strategy is explored to achieve model acceleration with minimal loss of speech quality. To our knowledge, VITA-Audio is the first multi-modal large language model capable of generating audio output during the first forward pass, enabling real-time conversational capabilities with minimal latency. VITA-Audio is fully reproducible and is trained on open-source data only. Experimental results demonstrate that our model achieves an inference speedup of 3~5x at the 7B parameter scale, but also significantly outperforms open-source models of similar model size on multiple benchmarks for automatic speech recognition (ASR), text-to-speech (TTS), and spoken question answering (SQA) tasks. 

**Abstract (ZH)**: 基于语音的端到端大型语音模型VITA-Audio：高效音频文本令牌生成 

---
# AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control 

**Title (ZH)**: AMO：自适应运动优化在超灵巧人形全身控制中的应用 

**Authors**: Jialong Li, Xuxin Cheng, Tianshu Huang, Shiqi Yang, Ri-Zhao Qiu, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03738)  

**Abstract**: Humanoid robots derive much of their dexterity from hyper-dexterous whole-body movements, enabling tasks that require a large operational workspace: such as picking objects off the ground. However, achieving these capabilities on real humanoids remains challenging due to their high degrees of freedom (DoF) and nonlinear dynamics. We propose Adaptive Motion Optimization (AMO), a framework that integrates sim-to-real reinforcement learning (RL) with trajectory optimization for real-time, adaptive whole-body control. To mitigate distribution bias in motion imitation RL, we construct a hybrid AMO dataset and train a network capable of robust, on-demand adaptation to potentially O.O.D. commands. We validate AMO in simulation and on a 29-DoF Unitree G1 humanoid robot, demonstrating superior stability and an expanded workspace compared to strong baselines. Finally, we show that AMO's consistent performance supports autonomous task execution via imitation learning, underscoring the system's versatility and robustness. 

**Abstract (ZH)**: 类人机器人通过超灵巧的全身运动获得其灵巧性，能够执行需要大操作空间的任务，如捡拾地上的物体。然而，由于其高自由度和非线性动力学，要在实际的类人机器人上实现这些能力仍然具有挑战性。我们提出了自适应运动优化（AMO）框架，该框架将模拟到现实的强化学习（RL）与轨迹优化结合，以实现实时、自适应的全身控制。为减轻运动模仿RL中的分布偏差，我们构建了一个混合AMO数据集，并训练一个能够在潜在O.O.D.命令下实现鲁棒、按需适应的网络。我们在模拟中验证了AMO，并在具有29个自由度的Unitree G1类人机器人上进行了实验，结果显示出优于强基线系统的优越稳定性和扩展的工作空间。最后，我们展示了AMO的一致性能支持通过模仿学习实现自主任务执行，突显了该系统的多功能性和鲁棒性。 

---
# FlexiAct: Towards Flexible Action Control in Heterogeneous Scenarios 

**Title (ZH)**: FlexiAct: 向泛化场景下的灵活动作控制迈进 

**Authors**: Shiyi Zhang, Junhao Zhuang, Zhaoyang Zhang, Ying Shan, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03730)  

**Abstract**: Action customization involves generating videos where the subject performs actions dictated by input control signals. Current methods use pose-guided or global motion customization but are limited by strict constraints on spatial structure, such as layout, skeleton, and viewpoint consistency, reducing adaptability across diverse subjects and scenarios. To overcome these limitations, we propose FlexiAct, which transfers actions from a reference video to an arbitrary target image. Unlike existing methods, FlexiAct allows for variations in layout, viewpoint, and skeletal structure between the subject of the reference video and the target image, while maintaining identity consistency. Achieving this requires precise action control, spatial structure adaptation, and consistency preservation. To this end, we introduce RefAdapter, a lightweight image-conditioned adapter that excels in spatial adaptation and consistency preservation, surpassing existing methods in balancing appearance consistency and structural flexibility. Additionally, based on our observations, the denoising process exhibits varying levels of attention to motion (low frequency) and appearance details (high frequency) at different timesteps. So we propose FAE (Frequency-aware Action Extraction), which, unlike existing methods that rely on separate spatial-temporal architectures, directly achieves action extraction during the denoising process. Experiments demonstrate that our method effectively transfers actions to subjects with diverse layouts, skeletons, and viewpoints. We release our code and model weights to support further research at this https URL 

**Abstract (ZH)**: Action定制涉及生成视频，其中主体根据输入控制信号执行动作。现有方法使用姿态引导或全局运动定制，但受限于布局、骨架和视角一致性等严格的空间结构限制，降低了在多样化主体和场景中的适应性。为克服这些限制，我们提出FlexiAct，该方法将参考视频中的动作转移到任意目标图像中。与现有方法不同，FlexiAct 允许参考视频中的主体与目标图像之间的布局、视角和骨架结构发生变化，同时保持身份一致性。实现这一点需要精确的动作控制、空间结构适应和一致性保持。为此，我们引入了RefAdapter，这是一种轻量级的条件图像适配器，在空间适应和一致性保持方面表现出色，优于现有方法在外观一致性与结构灵活性之间的平衡。此外，基于我们的观察，去噪过程在不同时间步长中对运动（低频）和外观细节（高频）的关注程度不同。因此，我们提出了FAE（频域意识动作提取），它与现有方法依赖于独立的空间-时间架构的做法不同，在去噪过程中直接实现动作提取。实验表明，我们的方法能有效将动作转移到具有不同布局、骨架和视角的主体上。我们将在以下链接发布我们的代码和模型权重以支持进一步的研究：这个https URL。 

---
# Actor-Critics Can Achieve Optimal Sample Efficiency 

**Title (ZH)**: 演员-评论家可以实现最优样本效率 

**Authors**: Kevin Tan, Wei Fan, Yuting Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.03710)  

**Abstract**: Actor-critic algorithms have become a cornerstone in reinforcement learning (RL), leveraging the strengths of both policy-based and value-based methods. Despite recent progress in understanding their statistical efficiency, no existing work has successfully learned an $\epsilon$-optimal policy with a sample complexity of $O(1/\epsilon^2)$ trajectories with general function approximation when strategic exploration is necessary.
We address this open problem by introducing a novel actor-critic algorithm that attains a sample-complexity of $O(dH^5 \log|\mathcal{A}|/\epsilon^2 + d H^4 \log|\mathcal{F}|/ \epsilon^2)$ trajectories, and accompanying $\sqrt{T}$ regret when the Bellman eluder dimension $d$ does not increase with $T$ at more than a $\log T$ rate.
Here, $\mathcal{F}$ is the critic function class, $\mathcal{A}$ is the action space, and $H$ is the horizon in the finite horizon MDP setting. Our algorithm integrates optimism, off-policy critic estimation targeting the optimal Q-function, and rare-switching policy resets.
We extend this to the setting of Hybrid RL, showing that initializing the critic with offline data yields sample efficiency gains compared to purely offline or online RL. Further, utilizing access to offline data, we provide a \textit{non-optimistic} provably efficient actor-critic algorithm that only additionally requires $N_{\text{off}} \geq c_{\text{off}}^*dH^4/\epsilon^2$ in exchange for omitting optimism, where $c_{\text{off}}^*$ is the single-policy concentrability coefficient and $N_{\text{off}}$ is the number of offline samples. This addresses another open problem in the literature. We further provide numerical experiments to support our theoretical findings. 

**Abstract (ZH)**: 基于函数逼近的策略与价值共同学习算法在需要战略探索的情况下，实现了$\epsilon$-最优策略的学习，其样本复杂度为$O(dH^5 \log|\mathcal{A}|/\epsilon^2 + d H^4 \log|\mathcal{F}|/ \epsilon^2)$轨迹，当贝尔曼庸人维度$d$以$\log T$速率增加时，伴随$\sqrt{T}$遗憾。 

---
# Demonstrating ViSafe: Vision-enabled Safety for High-speed Detect and Avoid 

**Title (ZH)**: 基于视觉的安全性展示：高speed检测与避免 

**Authors**: Parv Kapoor, Ian Higgins, Nikhil Keetha, Jay Patrikar, Brady Moon, Zelin Ye, Yao He, Ivan Cisneros, Yaoyu Hu, Changliu Liu, Eunsuk Kang, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03694)  

**Abstract**: Assured safe-separation is essential for achieving seamless high-density operation of airborne vehicles in a shared airspace. To equip resource-constrained aerial systems with this safety-critical capability, we present ViSafe, a high-speed vision-only airborne collision avoidance system. ViSafe offers a full-stack solution to the Detect and Avoid (DAA) problem by tightly integrating a learning-based edge-AI framework with a custom multi-camera hardware prototype designed under SWaP-C constraints. By leveraging perceptual input-focused control barrier functions (CBF) to design, encode, and enforce safety thresholds, ViSafe can provide provably safe runtime guarantees for self-separation in high-speed aerial operations. We evaluate ViSafe's performance through an extensive test campaign involving both simulated digital twins and real-world flight scenarios. By independently varying agent types, closure rates, interaction geometries, and environmental conditions (e.g., weather and lighting), we demonstrate that ViSafe consistently ensures self-separation across diverse scenarios. In first-of-its-kind real-world high-speed collision avoidance tests with closure rates reaching 144 km/h, ViSafe sets a new benchmark for vision-only autonomous collision avoidance, establishing a new standard for safety in high-speed aerial navigation. 

**Abstract (ZH)**: 确保安全分离是实现空中车辆在共享空域中无缝高密度运行的关键。为使资源受限的航空系统具备这一安全关键能力，我们提出了ViSafe，一种高速视觉_ONLY_空中碰撞避险系统。ViSafe通过将基于学习的边缘AI框架与在SWaP-C约束下设计的定制多摄像头硬件原型紧密集成，提供了一整套解决探测与避险(DAA)问题的方案。通过利用基于感知输入的关键障碍函数(CBF)进行设计、编码和实施安全阈值，ViSafe可以为高速空域操作中的自我分离提供可证明的安全运行保证。我们通过涵盖模拟数字孪生和真实飞行场景的广泛测试活动来评估ViSafe的性能。通过独立变化代理类型、闭合率、相互作用几何形状以及环境条件（例如天气和照明），我们证明了ViSafe能够在多种场景中一致地确保自我分离。在首次实现实高速碰撞避险测试中，闭合率达到144 km/h，ViSafe确立了视觉_ONLY_自主避险的新基准，并为高速空中导航建立了新的安全标准。 

---
# Revolutionizing Brain Tumor Imaging: Generating Synthetic 3D FA Maps from T1-Weighted MRI using CycleGAN Models 

**Title (ZH)**: 革新脑肿瘤成像：使用CycleGAN模型从T1加权MRI生成合成3D FA图谱 

**Authors**: Xin Du, Francesca M. Cozzi, Rajesh Jena  

**Link**: [PDF](https://arxiv.org/pdf/2505.03662)  

**Abstract**: Fractional anisotropy (FA) and directionally encoded colour (DEC) maps are essential for evaluating white matter integrity and structural connectivity in neuroimaging. However, the spatial misalignment between FA maps and tractography atlases hinders their effective integration into predictive models. To address this issue, we propose a CycleGAN based approach for generating FA maps directly from T1-weighted MRI scans, representing the first application of this technique to both healthy and tumour-affected tissues. Our model, trained on unpaired data, produces high fidelity maps, which have been rigorously evaluated using Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR), demonstrating particularly robust performance in tumour regions. Radiological assessments further underscore the model's potential to enhance clinical workflows by providing an AI-driven alternative that reduces the necessity for additional scans. 

**Abstract (ZH)**: 基于CycleGAN的T1加权MRI扫描转换为FA图的 方法及其在健康和肿瘤组织中的应用 

---
# Counterfactual Inference for Eliminating Sentiment Bias in Recommender Systems 

**Title (ZH)**: 消除推荐系统中情感偏见的事实推理方法 

**Authors**: Le Pan, Yuanjiang Cao, Chengkai Huang, Wenjie Zhang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.03655)  

**Abstract**: Recommender Systems (RSs) aim to provide personalized recommendations for users. A newly discovered bias, known as sentiment bias, uncovers a common phenomenon within Review-based RSs (RRSs): the recommendation accuracy of users or items with negative reviews deteriorates compared with users or items with positive reviews. Critical users and niche items are disadvantaged by such unfair recommendations. We study this problem from the perspective of counterfactual inference with two stages. At the model training stage, we build a causal graph and model how sentiment influences the final rating score. During the inference stage, we decouple the direct and indirect effects to mitigate the impact of sentiment bias and remove the indirect effect using counterfactual inference. We have conducted extensive experiments, and the results validate that our model can achieve comparable performance on rating prediction for better recommendations and effective mitigation of sentiment bias. To the best of our knowledge, this is the first work to employ counterfactual inference on sentiment bias mitigation in RSs. 

**Abstract (ZH)**: 基于情绪偏差的推荐系统中的公平性研究：基于反事实推理的两阶段方法 

---
# ReGraP-LLaVA: Reasoning enabled Graph-based Personalized Large Language and Vision Assistant 

**Title (ZH)**: ReGraP-LLaVA: 基于图的推理增强个性化大型语言和视觉助手 

**Authors**: Yifan Xiang, Zhenxi Zhang, Bin Li, Yixuan Weng, Shoujun Zhou, Yangfan He, Keqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03654)  

**Abstract**: Recent advances in personalized MLLMs enable effective capture of user-specific concepts, supporting both recognition of personalized concepts and contextual captioning. However, humans typically explore and reason over relations among objects and individuals, transcending surface-level information to achieve more personalized and contextual understanding. To this end, existing methods may face three main limitations: Their training data lacks multi-object sets in which relations among objects are learnable. Building on the limited training data, their models overlook the relations between different personalized concepts and fail to reason over them. Their experiments mainly focus on a single personalized concept, where evaluations are limited to recognition and captioning tasks. To address the limitations, we present a new dataset named ReGraP, consisting of 120 sets of personalized knowledge. Each set includes images, KGs, and CoT QA pairs derived from the KGs, enabling more structured and sophisticated reasoning pathways. We propose ReGraP-LLaVA, an MLLM trained with the corresponding KGs and CoT QA pairs, where soft and hard graph prompting methods are designed to align KGs within the model's semantic space. We establish the ReGraP Benchmark, which contains diverse task types: multiple-choice, fill-in-the-blank, True/False, and descriptive questions in both open- and closed-ended settings. The proposed benchmark is designed to evaluate the relational reasoning and knowledge-connection capability of personalized MLLMs. We conduct experiments on the proposed ReGraP-LLaVA and other competitive MLLMs. Results show that the proposed model not only learns personalized knowledge but also performs relational reasoning in responses, achieving the SoTA performance compared with the competitive methods. All the codes and datasets are released at: this https URL. 

**Abstract (ZH)**: Recent advances in个性化MLLMs使用户特定概念的捕捉更加有效，支持个性化概念的识别和上下文描述。然而，人类通常探索和推理对象及个体之间的关系，超越表面信息以达成更加个性化和上下文的理解。为此，现有方法可能面临三大限制：其训练数据缺乏可学习对象间关系的多对象集合。基于有限的训练数据，其模型忽视了不同个性化概念之间的关系并无法推理它们。其实验主要集中在单一个性化概念上，评估主要限于识别和描述任务。为解决这些问题，我们提出一个新的名为ReGraP的数据集，包含120个个性化知识集合。每个集合包括图像、知识图谱（KG）和基于KG的CoT QA对，从而提供更结构化和复杂的推理路径。我们提出ReGraP-LLaVA，该模型使用相应的KG和CoT QA对进行训练，并设计了软性和硬性图结构提示方法，使其KGs在模型的语义空间内对齐。我们建立了ReGraP基准，包含多种任务类型：多项选择题、填充题、真/假判断及开放式和封闭式的描述性问题。该提出的基准旨在评估个性化MLLMs的关系推理和知识连接能力。我们在提出的ReGraP-LLaVA和其它竞争性MLLMs上进行了实验。结果显示，提出的模型不仅学习了个性化知识，还在响应中进行了关系推理，性能优于其他竞争方法。所有代码和数据集发布在：https://your-link-url.com。 

---
# Binding threshold units with artificial oscillatory neurons 

**Title (ZH)**: 人工振荡神经元与绑定阈值单元结合 

**Authors**: Vladimir Fanaskov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2505.03648)  

**Abstract**: Artificial Kuramoto oscillatory neurons were recently introduced as an alternative to threshold units. Empirical evidence suggests that oscillatory units outperform threshold units in several tasks including unsupervised object discovery and certain reasoning problems. The proposed coupling mechanism for these oscillatory neurons is heterogeneous, combining a generalized Kuramoto equation with standard coupling methods used for threshold units. In this research note, we present a theoretical framework that clearly distinguishes oscillatory neurons from threshold units and establishes a coupling mechanism between them. We argue that, from a biological standpoint, oscillatory and threshold units realise distinct aspects of neural coding: roughly, threshold units model intensity of neuron firing, while oscillatory units facilitate information exchange by frequency modulation. To derive interaction between these two types of units, we constrain their dynamics by focusing on dynamical systems that admit Lyapunov functions. For threshold units, this leads to Hopfield associative memory model, and for oscillatory units it yields a specific form of generalized Kuramoto model. The resulting dynamical systems can be naturally coupled to form a Hopfield-Kuramoto associative memory model, which also admits a Lyapunov function. Various forms of coupling are possible. Notably, oscillatory neurons can be employed to implement a low-rank correction to the weight matrix of a Hopfield network. This correction can be viewed either as a form of Hebbian learning or as a popular LoRA method used for fine-tuning of large language models. We demonstrate the practical realization of this particular coupling through illustrative toy experiments. 

**Abstract (ZH)**: 人工Kuramoto振荡神经元 Recently Introduced as an Alternative to Threshold Units: A Theoretical Framework and Coupling Mechanism 

---
# ALMA: Aggregated Lipschitz Maximization Attack on Auto-encoders 

**Title (ZH)**: ALMA: 自编码器上的聚合Lipschitz最大化攻击 

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Eirini Ntoutsi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03646)  

**Abstract**: Despite the extensive use of deep autoencoders (AEs) in critical applications, their adversarial robustness remains relatively underexplored compared to classification models. AE robustness is characterized by the Lipschitz bounds of its components. Existing robustness evaluation frameworks based on white-box attacks do not fully exploit the vulnerabilities of intermediate ill-conditioned layers in AEs. In the context of optimizing imperceptible norm-bounded additive perturbations to maximize output damage, existing methods struggle to effectively propagate adversarial loss gradients throughout the network, often converging to less effective perturbations. To address this, we propose a novel layer-conditioning-based adversarial optimization objective that effectively guides the adversarial map toward regions of local Lipschitz bounds by enhancing loss gradient information propagation during attack optimization. We demonstrate through extensive experiments on state-of-the-art AEs that our adversarial objective results in stronger attacks, outperforming existing methods in both universal and sample-specific scenarios. As a defense method against this attack, we introduce an inference-time adversarially trained defense plugin that mitigates the effects of adversarial examples. 

**Abstract (ZH)**: 尽管深层自编码器（AEs）在关键应用中得到了广泛使用，但与分类模型相比，其对抗鲁棒性研究仍相对不足。AE的对抗鲁棒性可以由其组件的Lipschitz界来表征。基于白盒攻击的现有鲁棒性评估框架未能充分利用AE中间病态层的漏洞。在优化不可感知的范数界附加扰动以最大化输出损伤的背景下，现有方法在攻击优化过程中难以有效地传递对抗损失梯度信息，往往收敛到效果较差的扰动。为了解决这一问题，我们提出了一种新的基于层条件的对抗优化目标，该目标能够通过增强攻击优化过程中损失梯度信息的传递，有效地引导对抗映射朝向局部Lipschitz界区域。通过在最先进的AEs上的 extensive 实验，我们表明我们的对抗目标能够产生更强的攻击，在通用场景和样本特定场景中均优于现有方法。作为针对该攻击的防御方法，我们引入了一种推断时对抗训练的防御插件，以缓解对抗样本的影响。 

---
# Rainbow Delay Compensation: A Multi-Agent Reinforcement Learning Framework for Mitigating Delayed Observation 

**Title (ZH)**: 多agent强化学习框架：缓解延迟观测的 Rainbow 延迟补偿 

**Authors**: Songchen Fu, Siang Chen, Shaojing Zhao, Letian Bai, Ta Li, Yonghong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03586)  

**Abstract**: In real-world multi-agent systems (MASs), observation delays are ubiquitous, preventing agents from making decisions based on the environment's true state. An individual agent's local observation often consists of multiple components from other agents or dynamic entities in the environment. These discrete observation components with varying delay characteristics pose significant challenges for multi-agent reinforcement learning (MARL). In this paper, we first formulate the decentralized stochastic individual delay partially observable Markov decision process (DSID-POMDP) by extending the standard Dec-POMDP. We then propose the Rainbow Delay Compensation (RDC), a MARL training framework for addressing stochastic individual delays, along with recommended implementations for its constituent modules. We implement the DSID-POMDP's observation generation pattern using standard MARL benchmarks, including MPE and SMAC. Experiments demonstrate that baseline MARL methods suffer severe performance degradation under fixed and unfixed delays. The RDC-enhanced approach mitigates this issue, remarkably achieving ideal delay-free performance in certain delay scenarios while maintaining generalization capability. Our work provides a novel perspective on multi-agent delayed observation problems and offers an effective solution framework. 

**Abstract (ZH)**: 在现实世界多Agent系统中，观测延迟普遍存在，阻碍agents基于环境的真实状态作出决策。个体agent的局部观测通常由环境中其他agent或动态实体的多个组成部分组成。这些具有不同延迟特性的离散观测组件对多Agent强化学习（MARL）提出了重大挑战。在本文中，我们首先通过扩展标准Dec-POMDP来形式化去中心化的随机个体延迟部分可观测马尔可夫决策过程（DSID-POMDP）。随后，我们提出了一种Rainbow Delay Compensation（RDC），这是一种用于解决随机个体延迟的MARL训练框架，并推荐了其组成部分模块的具体实现方式。我们使用标准的MARL基准，包括MPE和SMAC，来实现DSID-POMDP的观测生成模式。实验表明，基础的MARL方法在固定和非固定延迟下性能显著下降。RDC增强方法缓解了这一问题，在某些延迟场景中实现了理想的无延迟性能，同时保持了一定的泛化能力。我们的工作为多Agent延迟观测问题提供了一个新的视角，并提供了一种有效的解决方案框架。 

---
# BCause: Human-AI collaboration to improve hybrid mapping and ideation in argumentation-grounded deliberation 

**Title (ZH)**: BCause: 人机协作以改进基于论据的混合映射与创意思考 

**Authors**: Lucas Anastasiou, Anna De Liddo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03584)  

**Abstract**: Public deliberation, as in open discussion of issues of public concern, often suffers from scattered and shallow discourse, poor sensemaking, and a disconnect from actionable policy outcomes. This paper introduces BCause, a discussion system leveraging generative AI and human-machine collaboration to transform unstructured dialogue around public issues (such as urban living, policy changes, and current socio-economic transformations) into structured, actionable democratic processes. We present three innovations: (i) importing and transforming unstructured transcripts into argumentative discussions, (ii) geo-deliberated problem-sensing via a Telegram bot for local issue reporting, and (iii) smart reporting with customizable widgets (e.g., summaries, topic modelling, policy recommendations, clustered arguments). The system's human-AI partnership preserves critical human participation to ensure ethical oversight, contextual relevance, and creative synthesis. 

**Abstract (ZH)**: 公共事务讨论往往受到分散和浅薄的讨论、差的含义构建以及与可执行政策结果脱节的影响。本文介绍了BCause，这是一种利用生成型AI和人机合作的讨论系统，旨在将关于公共议题（如城市生活、政策变化和当前社会经济转型）的非结构化对话转化为结构化、可执行的民主过程。我们提出了三项创新：（i）将非结构化转录导入并转化为论辩性讨论，（ii）通过Telegram机器人进行地理导向的问题感知以报告当地问题，以及（iii）可定制的智能报告工具（例如，摘要、主题建模、政策建议、聚类论证）。该系统的人机伙伴关系保留了关键的人类参与，以确保伦理监督、情境相关性和创造性的综合。 

---
# LlamaFirewall: An open source guardrail system for building secure AI agents 

**Title (ZH)**: LlamaFirewall: 一个开源的安全护栏系统，用于构建安全的AI代理 

**Authors**: Sahana Chennabasappa, Cyrus Nikolaidis, Daniel Song, David Molnar, Stephanie Ding, Shengye Wan, Spencer Whitman, Lauren Deason, Nicholas Doucette, Abraham Montilla, Alekhya Gampa, Beto de Paola, Dominik Gabi, James Crnkovich, Jean-Christophe Testud, Kat He, Rashnil Chaturvedi, Wu Zhou, Joshua Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2505.03574)  

**Abstract**: Large language models (LLMs) have evolved from simple chatbots into autonomous agents capable of performing complex tasks such as editing production code, orchestrating workflows, and taking higher-stakes actions based on untrusted inputs like webpages and emails. These capabilities introduce new security risks that existing security measures, such as model fine-tuning or chatbot-focused guardrails, do not fully address. Given the higher stakes and the absence of deterministic solutions to mitigate these risks, there is a critical need for a real-time guardrail monitor to serve as a final layer of defense, and support system level, use case specific safety policy definition and enforcement. We introduce LlamaFirewall, an open-source security focused guardrail framework designed to serve as a final layer of defense against security risks associated with AI Agents. Our framework mitigates risks such as prompt injection, agent misalignment, and insecure code risks through three powerful guardrails: PromptGuard 2, a universal jailbreak detector that demonstrates clear state of the art performance; Agent Alignment Checks, a chain-of-thought auditor that inspects agent reasoning for prompt injection and goal misalignment, which, while still experimental, shows stronger efficacy at preventing indirect injections in general scenarios than previously proposed approaches; and CodeShield, an online static analysis engine that is both fast and extensible, aimed at preventing the generation of insecure or dangerous code by coding agents. Additionally, we include easy-to-use customizable scanners that make it possible for any developer who can write a regular expression or an LLM prompt to quickly update an agent's security guardrails. 

**Abstract (ZH)**: 大型语言模型（LLMs）已从简单的聊天机器人发展成为能够执行复杂任务（如编辑生产代码、协调工作流和基于网页和电子邮件等不可信输入采取高风险行动）的自主代理。这些能力引入了现有安全措施（如模型微调或聊天机器人专用的防护栏）无法充分解决的新安全风险。鉴于高风险且缺乏确定性的解决方案，急需一个实时防护栏监控系统作为最终的防御层，并支持针对特定应用场景的安全政策定义和执行。我们介绍了LlamaFirewall，一个开源的安全防护栏框架，旨在作为AI代理相关安全风险的最终防御层。我们的框架通过三种强有力的防护栏来降低风险，包括PromptGuard 2，一种通用的脱域检测器，展示出明显处于前沿的技术性能；Agent Alignment Checks，一种思维链审计器，检查代理推理以防止提示注入和目标偏移，尽管仍处于实验阶段，但在一般场景中显示了比之前提出的方法更强的间接注入预防效果；以及CodeShield，一个既快速又可扩展的在线静态分析引擎，旨在防止编码代理生成不安全或危险的代码。此外，我们还提供了易于使用的可定制扫描器，使任何能够编写正则表达式或LLM提示的开发人员都能够快速更新代理的安全防护栏。 

---
# Real-Time Person Image Synthesis Using a Flow Matching Model 

**Title (ZH)**: 基于流匹配模型的实时人体图像合成 

**Authors**: Jiwoo Jeong, Kirok Kim, Wooju Kim, Nam-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.03562)  

**Abstract**: Pose-Guided Person Image Synthesis (PGPIS) generates realistic person images conditioned on a target pose and a source image. This task plays a key role in various real-world applications, such as sign language video generation, AR/VR, gaming, and live streaming. In these scenarios, real-time PGPIS is critical for providing immediate visual feedback and maintaining user this http URL, achieving real-time performance remains a significant challenge due to the complexity of synthesizing high-fidelity images from diverse and dynamic human poses. Recent diffusion-based methods have shown impressive image quality in PGPIS, but their slow sampling speeds hinder deployment in time-sensitive applications. This latency is particularly problematic in tasks like generating sign language videos during live broadcasts, where rapid image updates are required. Therefore, developing a fast and reliable PGPIS model is a crucial step toward enabling real-time interactive systems. To address this challenge, we propose a generative model based on flow matching (FM). Our approach enables faster, more stable, and more efficient training and sampling. Furthermore, the proposed model supports conditional generation and can operate in latent space, making it especially suitable for real-time PGPIS applications where both speed and quality are critical. We evaluate our proposed method, Real-Time Person Image Synthesis Using a Flow Matching Model (RPFM), on the widely used DeepFashion dataset for PGPIS tasks. Our results show that RPFM achieves near-real-time sampling speeds while maintaining performance comparable to the state-of-the-art models. Our methodology trades off a slight, acceptable decrease in generated-image accuracy for over a twofold increase in generation speed, thereby ensuring real-time performance. 

**Abstract (ZH)**: 基于流匹配的实时人体姿态导向图像合成 (Flow-Matching Guided Real-Time Person Image Synthesis, FM-GRTPIS) 

---
# Ergodic Generative Flows 

**Title (ZH)**: 遍历生成流 

**Authors**: Leo Maxime Brunswic, Mateo Clemente, Rui Heng Yang, Adam Sigal, Amir Rasouli, Yinchuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03561)  

**Abstract**: Generative Flow Networks (GFNs) were initially introduced on directed acyclic graphs to sample from an unnormalized distribution density. Recent works have extended the theoretical framework for generative methods allowing more flexibility and enhancing application range. However, many challenges remain in training GFNs in continuous settings and for imitation learning (IL), including intractability of flow-matching loss, limited tests of non-acyclic training, and the need for a separate reward model in imitation learning. The present work proposes a family of generative flows called Ergodic Generative Flows (EGFs) which are used to address the aforementioned issues. First, we leverage ergodicity to build simple generative flows with finitely many globally defined transformations (diffeomorphisms) with universality guarantees and tractable flow-matching loss (FM loss). Second, we introduce a new loss involving cross-entropy coupled to weak flow-matching control, coined KL-weakFM loss. It is designed for IL training without a separate reward model. We evaluate IL-EGFs on toy 2D tasks and real-world datasets from NASA on the sphere, using the KL-weakFM loss. Additionally, we conduct toy 2D reinforcement learning experiments with a target reward, using the FM loss. 

**Abstract (ZH)**: Ergodic Generative Flows (EGFs) for Generative Methods and Imitation Learning 

---
# Rapid AI-based generation of coverage paths for dispensing applications 

**Title (ZH)**: 基于AI的快速生成涂布应用覆盖路径 

**Authors**: Simon Baeuerle, Ian F. Mendonca, Kristof Van Laerhoven, Ralf Mikut, Andreas Steimer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03560)  

**Abstract**: Coverage Path Planning of Thermal Interface Materials (TIM) plays a crucial role in the design of power electronics and electronic control units. Up to now, this is done manually by experts or by using optimization approaches with a high computational effort. We propose a novel AI-based approach to generate dispense paths for TIM and similar dispensing applications. It is a drop-in replacement for optimization-based approaches. An Artificial Neural Network (ANN) receives the target cooling area as input and directly outputs the dispense path. Our proposed setup does not require labels and we show its feasibility on multiple target areas. The resulting dispense paths can be directly transferred to automated manufacturing equipment and do not exhibit air entrapments. The approach of using an ANN to predict process parameters for a desired target state in real-time could potentially be transferred to other manufacturing processes. 

**Abstract (ZH)**: 热界面材料(TIM)的覆盖路径规划在电源电子和电子控制单元的设计中起着至关重要的作用。目前，这通常是通过专家手动完成或使用高计算成本的优化方法。我们提出了一种基于人工智能的新颖方法，用于生成TIM和其他分配应用的分配路径。该方法是对基于优化的方法的一种即插即用替代方案。人工神经网络(ANN)接收目标冷却区域作为输入，并直接输出分配路径。我们提出的方法不需要标签，并在多个目标区域上展示了其可行性。生成的分配路径可以直接应用于自动化制造设备且不会出现气泡。使用ANN实时预测所需目标状态的工艺参数的方法有可能应用于其他制造过程中。 

---
# Generating Synthetic Data via Augmentations for Improved Facial Resemblance in DreamBooth and InstantID 

**Title (ZH)**: 通过增强技术生成合成数据以改善DreamBooth和InstantID中的面部相似性 

**Authors**: Koray Ulusan, Benjamin Kiefer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03557)  

**Abstract**: The personalization of Stable Diffusion for generating professional portraits from amateur photographs is a burgeoning area, with applications in various downstream contexts. This paper investigates the impact of augmentations on improving facial resemblance when using two prominent personalization techniques: DreamBooth and InstantID. Through a series of experiments with diverse subject datasets, we assessed the effectiveness of various augmentation strategies on the generated headshots' fidelity to the original subject. We introduce FaceDistance, a wrapper around FaceNet, to rank the generations based on facial similarity, which aided in our assessment. Ultimately, this research provides insights into the role of augmentations in enhancing facial resemblance in SDXL-generated portraits, informing strategies for their effective deployment in downstream applications. 

**Abstract (ZH)**: 基于增强技术在稳定扩散模型中生成专业肖像的个性化研究：DreamBooth和InstantID的应用分析 

---
# Optimization of Module Transferability in Single Image Super-Resolution: Universality Assessment and Cycle Residual Blocks 

**Title (ZH)**: 单张图像超分辨率中模块可转移性优化：普适性评估与循环残差块 

**Authors**: Haotong Cheng, Zhiqi Zhang, Hao Li, Xinshang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03522)  

**Abstract**: Deep learning has substantially advanced the Single Image Super-Resolution (SISR). However, existing researches have predominantly focused on raw performance gains, with little attention paid to quantifying the transferability of architectural components. In this paper, we introduce the concept of "Universality" and its associated definitions which extend the traditional notion of "Generalization" to encompass the modules' ease of transferability, thus revealing the relationships between module universality and model generalizability. Then we propose the Universality Assessment Equation (UAE), a metric for quantifying how readily a given module could be transplanted across models. Guided by the UAE results of standard residual blocks and other plug-and-play modules, we further design two optimized modules, Cycle Residual Block (CRB) and Depth-Wise Cycle Residual Block (DCRB). Through comprehensive experiments on natural-scene benchmarks, remote-sensing datasets, extreme-industrial imagery and on-device deployments, we demonstrate that networks embedded with the proposed plug-and-play modules outperform several state-of-the-arts, reaching a PSNR enhancement of up to 0.83dB or enabling a 71.3% reduction in parameters with negligible loss in reconstruction fidelity. 

**Abstract (ZH)**: 深度学习显著推进了单张图像超分辨率（SISR）。然而，现有研究主要关注性能提升，忽视了架构组件转移性的量化。本文引入了“泛化性”及其相关定义，将传统的“泛化”概念扩展到涵盖模块的易转移性，从而揭示模块泛化性与模型泛化能力之间的关系。我们提出了泛化性评估方程（UAE），用于量化给定模块易于移植的程度。基于标准残差模块和其他即插即用模块的UAE结果，我们进一步设计了两种优化模块：周期残差块（CRB）和深度可分离周期残差块（DCRB）。通过在自然场景基准、遥感数据集、极端工业图像及设备端部署上的全面实验，我们证明嵌入所提出即插即用模块的网络优于多种最先进的方法，PSNR提升可达0.83dB或参数减少71.3%且重建保真度无明显下降。 

---
# From Neurons to Computation: Biological Reservoir Computing for Pattern Recognition 

**Title (ZH)**: 从神经元到计算：生物型 reservoir 计算在模式识别中的应用 

**Authors**: Ludovico Iannello, Luca Ciampi, Gabriele Lagani, Fabrizio Tonelli, Eleonora Crocco, Lucio Maria Calcagnile, Angelo Di Garbo, Federico Cremisi, Giuseppe Amato  

**Link**: [PDF](https://arxiv.org/pdf/2505.03510)  

**Abstract**: In this paper, we introduce a novel paradigm for reservoir computing (RC) that leverages a pool of cultured biological neurons as the reservoir substrate, creating a biological reservoir computing (BRC). This system operates similarly to an echo state network (ESN), with the key distinction that the neural activity is generated by a network of cultured neurons, rather than being modeled by traditional artificial computational units. The neuronal activity is recorded using a multi-electrode array (MEA), which enables high-throughput recording of neural signals. In our approach, inputs are introduced into the network through a subset of the MEA electrodes, while the remaining electrodes capture the resulting neural activity. This generates a nonlinear mapping of the input data to a high-dimensional biological feature space, where distinguishing between data becomes more efficient and straightforward, allowing a simple linear classifier to perform pattern recognition tasks effectively. To evaluate the performance of our proposed system, we present an experimental study that includes various input patterns, such as positional codes, bars with different orientations, and a digit recognition task. The results demonstrate the feasibility of using biological neural networks to perform tasks traditionally handled by artificial neural networks, paving the way for further exploration of biologically-inspired computing systems, with potential applications in neuromorphic engineering and bio-hybrid computing. 

**Abstract (ZH)**: 本文介绍了一种新的神经计算（Reservoir Computing, RC）范式，利用培养的生物神经元池作为计算介质，创建了生物神经计算（Biological Reservoir Computing, BRC）系统。该系统类似于回声状态网络（Echo State Network, ESN），其关键区别在于神经活动由培养的神经元网络生成，而不是由传统的 artificial 计算单元建模。神经活动通过多电极阵列（Multi-Electrode Array,MEA）记录，实现了高通量神经信号记录。在本研究中，输入通过 MEA 的一部分电极注入网络，其余电极捕获由此产生的神经活动。这将输入数据非线性映射到高维生物特征空间，使得区分数据变得更加高效和直接，允许简单的线性分类器有效地完成模式识别任务。为了评估所提出系统的性能，我们展示了包括位置编码、不同方向的条纹以及数字识别任务在内的多种输入模式的实验研究。结果表明，可以使用生物神经网络执行传统由人工神经网络处理的任务，为生物启发计算系统的进一步探索铺平了道路，潜在应用于神经形态工程和生物混合计算。 

---
# Augmenting Human Cognition through Everyday AR 

**Title (ZH)**: 通过日常AR增强人类认知 

**Authors**: Xiaoan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03492)  

**Abstract**: As spatial computing and multimodal LLMs mature, AR is tending to become an intuitive "thinking tool," embedding semantic and context-aware intelligence directly into everyday environments. This paper explores how always-on AR can seamlessly bridge digital cognition and physical affordances, enabling proactive, context-sensitive interactions that enhance human task performance and understanding. 

**Abstract (ZH)**: 随着空间计算和多模态大语言模型的成熟，AR正趋向于成为一种直观的“思维工具”，将语义和上下文感知的智能直接嵌入到日常环境中。本文探讨了始终可用的AR如何无缝地连接数字认知和物理可用性，实现前瞻性的、上下文敏感的交互，从而增强人类的任务绩效和理解。 

---
# A new membership inference attack that spots memorization in generative and predictive models: Loss-Based with Reference Model algorithm (LBRM) 

**Title (ZH)**: 一种基于损失与参考模型的新会员推断攻击算法：识别生成性和预测性模型中的记忆现象（LBRM） 

**Authors**: Faiz Taleb, Ivan Gazeau, Maryline Laurent  

**Link**: [PDF](https://arxiv.org/pdf/2505.03490)  

**Abstract**: Generative models can unintentionally memorize training data, posing significant privacy risks. This paper addresses the memorization phenomenon in time series imputation models, introducing the Loss-Based with Reference Model (LBRM) algorithm. The LBRM method leverages a reference model to enhance the accuracy of membership inference attacks, distinguishing between training and test data. Our contributions are twofold: first, we propose an innovative method to effectively extract and identify memorized training data, significantly improving detection accuracy. On average, without fine-tuning, the AUROC improved by approximately 40\%. With fine-tuning, the AUROC increased by approximately 60\%. Second, we validate our approach through membership inference attacks on two types of architectures designed for time series imputation, demonstrating the robustness and versatility of the LBRM approach in different contexts. These results highlight the significant enhancement in detection accuracy provided by the LBRM approach, addressing privacy risks in time series imputation models. 

**Abstract (ZH)**: 生成模型可能无意中记忆训练数据，带来重要的隐私风险。本文针对时间序列插补模型中的记忆现象，提出了一种基于损失的参考模型（LBRM）算法。LBRM 方法利用参考模型提高成员推理攻击的准确性，区分训练数据和测试数据。我们的贡献主要有两点：首先，我们提出了一种有效提取和识别记忆训练数据的创新方法，显著提高了检测准确性。平均而言，在未微调的情况下，AUROC 提高了约 40%。经过微调后，AUROC 提高了约 60%。其次，我们通过两种类型的时间序列插补架构上的成员推理攻击验证了我们的方法，展示了 LBRM 方法在不同场景下的鲁棒性和通用性。这些结果突显了 LBRM 方法在提高检测准确性方面的重要增强，解决了时间序列插补模型中的隐私风险。 

---
# Blending 3D Geometry and Machine Learning for Multi-View Stereopsis 

**Title (ZH)**: 融合3D几何和机器学习的多视图立体视觉 

**Authors**: Vibhas Vats, Md. Alimoor Reza, David Crandall, Soon-heung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.03470)  

**Abstract**: Traditional multi-view stereo (MVS) methods primarily depend on photometric and geometric consistency constraints. In contrast, modern learning-based algorithms often rely on the plane sweep algorithm to infer 3D geometry, applying explicit geometric consistency (GC) checks only as a post-processing step, with no impact on the learning process itself. In this work, we introduce GC MVSNet plus plus, a novel approach that actively enforces geometric consistency of reference view depth maps across multiple source views (multi view) and at various scales (multi scale) during the learning phase (see Fig. 1). This integrated GC check significantly accelerates the learning process by directly penalizing geometrically inconsistent pixels, effectively halving the number of training iterations compared to other MVS methods. Furthermore, we introduce a densely connected cost regularization network with two distinct block designs simple and feature dense optimized to harness dense feature connections for enhanced regularization. Extensive experiments demonstrate that our approach achieves a new state of the art on the DTU and BlendedMVS datasets and secures second place on the Tanks and Temples benchmark. To our knowledge, GC MVSNet plus plus is the first method to enforce multi-view, multi-scale supervised geometric consistency during learning. Our code is available. 

**Abstract (ZH)**: GC MVSNet++：一种在学习阶段主动强制多视图多尺度几何一致性的新方法 

---
# An Analysis of Hyper-Parameter Optimization Methods for Retrieval Augmented Generation 

**Title (ZH)**: 检索增强生成的超参数优化方法分析 

**Authors**: Matan Orbach, Ohad Eytan, Benjamin Sznajder, Ariel Gera, Odellia Boni, Yoav Kantor, Gal Bloch, Omri Levy, Hadas Abraham, Nitzan Barzilay, Eyal Shnarch, Michael E. Factor, Shila Ofek-Koifman, Paula Ta-Shma, Assaf Toledo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03452)  

**Abstract**: Finding the optimal Retrieval-Augmented Generation (RAG) configuration for a given use case can be complex and expensive. Motivated by this challenge, frameworks for RAG hyper-parameter optimization (HPO) have recently emerged, yet their effectiveness has not been rigorously benchmarked. To address this gap, we present a comprehensive study involving 5 HPO algorithms over 5 datasets from diverse domains, including a new one collected for this work on real-world product documentation. Our study explores the largest HPO search space considered to date, with two optimized evaluation metrics. Analysis of the results shows that RAG HPO can be done efficiently, either greedily or with iterative random search, and that it significantly boosts RAG performance for all datasets. For greedy HPO approaches, we show that optimizing models first is preferable to the prevalent practice of optimizing sequentially according to the RAG pipeline order. 

**Abstract (ZH)**: 针对给定应用场景找到最佳检索增强生成（RAG）配置可能既复杂又昂贵。为应对这一挑战，最近出现了RAG超参数优化（HPO）框架，但其有效性尚未经过严格的基准测试。为进一步解决这一问题，我们进行了全面研究，涵盖了5种HPO算法和5个来自不同领域的数据集，其中包括一个为本次研究收集的新数据集，涉及真实-world产品文档。我们的研究探索了迄今最大的HPO搜索空间，并采用了两种优化评估指标。结果分析显示，无论是使用贪婪方法还是迭代随机搜索，RAG的HPO都可以高效进行，并且显著提升了所有数据集上的RAG性能。对于贪婪HPO方法，我们展示了首先优化模型比按照RAG管道顺序逐步优化更为可取。 

---
# Detecting Quishing Attacks with Machine Learning Techniques Through QR Code Analysis 

**Title (ZH)**: 通过QR码分析利用机器学习技术检测弃权攻击 

**Authors**: Fouad Trad, Ali Chehab  

**Link**: [PDF](https://arxiv.org/pdf/2505.03451)  

**Abstract**: The rise of QR code based phishing ("Quishing") poses a growing cybersecurity threat, as attackers increasingly exploit QR codes to bypass traditional phishing defenses. Existing detection methods predominantly focus on URL analysis, which requires the extraction of the QR code payload, and may inadvertently expose users to malicious content. Moreover, QR codes can encode various types of data beyond URLs, such as Wi-Fi credentials and payment information, making URL-based detection insufficient for broader security concerns. To address these gaps, we propose the first framework for quishing detection that directly analyzes QR code structure and pixel patterns without extracting the embedded content. We generated a dataset of phishing and benign QR codes and we used it to train and evaluate multiple machine learning models, including Logistic Regression, Decision Trees, Random Forest, Naive Bayes, LightGBM, and XGBoost. Our best-performing model (XGBoost) achieves an AUC of 0.9106, demonstrating the feasibility of QR-centric detection. Through feature importance analysis, we identify key visual indicators of malicious intent and refine our feature set by removing non-informative pixels, improving performance to an AUC of 0.9133 with a reduced feature space. Our findings reveal that the structural features of QR code correlate strongly with phishing risk. This work establishes a foundation for quishing mitigation and highlights the potential of direct QR analysis as a critical layer in modern phishing defenses. 

**Abstract (ZH)**: 基于二维码的恶意诱骗（Quishing）的兴起对网络安全构成了日益增长的威胁，攻击者越来越多地利用二维码规避传统诱骗防护。现有的检测方法主要侧重于URL分析，这需要提取二维码负载内容，可能会无意中使用户暴露在恶意内容中。此外，二维码可以编码超出URL的各种类型的数据，例如Wi-Fi凭据和支付信息，使得基于URL的检测对更广泛的安全部署不够充分。为了解决这些问题，我们提出了一种新的框架，直接分析二维码的结构和像素模式，而无需提取嵌入的内容。我们生成了一组恶意诱骗和良性二维码的数据集，并使用该数据集训练和评估了多种机器学习模型，包括逻辑回归、决策树、随机森林、朴素贝叶斯、LightGBM和XGBoost。我们的表现最佳模型（XGBoost）达到AUC值0.9106，证明了以二维码为中心的检测的可行性。通过特征重要性分析，我们识别出恶意意图的关键视觉指标，并通过去除不相关信息素来精简特征集，使性能在减少特征空间的情况下达到AUC值0.9133。我们的研究发现二维码的结构特征与诱骗风险紧密相关。本研究为防止恶意诱骗奠定了基础，并突显了直接分析二维码作为现代防诱骗防御体系关键层的潜力。 

---
# Elevating Semantic Exploration: A Novel Approach Utilizing Distributed Repositories 

**Title (ZH)**: 提升语义探索：一种利用分布式仓库的新方法 

**Authors**: Valerio Bellandi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03443)  

**Abstract**: Centralized and distributed systems are two main approaches to organizing ICT infrastructure, each with its pros and cons. Centralized systems concentrate resources in one location, making management easier but creating single points of failure. Distributed systems, on the other hand, spread resources across multiple nodes, offering better scalability and fault tolerance, but requiring more complex management. The choice between them depends on factors like application needs, scalability, and data sensitivity. Centralized systems suit applications with limited scalability and centralized control, while distributed systems excel in large-scale environments requiring high availability and performance. This paper explores a distributed document repository system developed for the Italian Ministry of Justice, using edge repositories to analyze textual data and metadata, enhancing semantic exploration capabilities. 

**Abstract (ZH)**: 集中式和分布式系统是组织ICT基础设施的两种主要方法，各有优缺点。集中式系统将资源集中在一个位置，便于管理但容易出现单点故障。分布式系统则将资源分布在多个节点上，提供了更好的可扩展性和容错性，但需要更复杂的管理。在应用需求、可扩展性和数据敏感性等因素的影响下，两者的选择各有利弊。集中式系统适用于需要有限扩展性和集中控制的应用，而分布式系统则在需要高可用性和高性能的大规模环境中表现出色。本文探讨了为意大利司法部开发的一个分布式文档存储系统，利用边缘存储库分析文本数据和元数据，增强语义探索能力。 

---
# MedArabiQ: Benchmarking Large Language Models on Arabic Medical Tasks 

**Title (ZH)**: MedArabiQ: 评估大型语言模型在阿拉伯医学任务中的性能 

**Authors**: Mouath Abu Daoud, Chaimae Abouzahir, Leen Kharouf, Walid Al-Eisawi, Nizar Habash, Farah E. Shamout  

**Link**: [PDF](https://arxiv.org/pdf/2505.03427)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant promise for various applications in healthcare. However, their efficacy in the Arabic medical domain remains unexplored due to the lack of high-quality domain-specific datasets and benchmarks. This study introduces MedArabiQ, a novel benchmark dataset consisting of seven Arabic medical tasks, covering multiple specialties and including multiple choice questions, fill-in-the-blank, and patient-doctor question answering. We first constructed the dataset using past medical exams and publicly available datasets. We then introduced different modifications to evaluate various LLM capabilities, including bias mitigation. We conducted an extensive evaluation with five state-of-the-art open-source and proprietary LLMs, including GPT-4o, Claude 3.5-Sonnet, and Gemini 1.5. Our findings highlight the need for the creation of new high-quality benchmarks that span different languages to ensure fair deployment and scalability of LLMs in healthcare. By establishing this benchmark and releasing the dataset, we provide a foundation for future research aimed at evaluating and enhancing the multilingual capabilities of LLMs for the equitable use of generative AI in healthcare. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗健康领域的应用展现了显著的潜力，然而它们在阿拉伯医学领域的效果尚未被探索，主要原因是缺乏高质量的专业领域数据集和基准。本研究 introduces MedArabiQ，一个包含七个阿拉伯医学任务的新颖基准数据集，覆盖多个专科，并包括多项选择题、填空题和病人-医生问答。我们首先使用过去的医学考试和公开可用的数据集构建了数据集。然后，我们引入了不同的修改来评估各种LLM的能力，包括偏见缓解。我们使用五种最新开源和专有LLM进行了广泛评估，包括GPT-4o、Claude 3.5-Sonnet和Gemini 1.5。我们的发现强调了创建跨越不同语言的新高质量基准的必要性，以确保LLMs在医疗健康领域中的公平部署和扩展。通过建立这一基准并发布数据集，我们为未来旨在评估和提升LLMs多语言能力的研究提供了基础，以促进生成性AI在医疗健康领域的公平使用。 

---
# Phenotype-Guided Generative Model for High-Fidelity Cardiac MRI Synthesis: Advancing Pretraining and Clinical Applications 

**Title (ZH)**: 基于表型指导的高保真心脏MRI生成模型：推进预训练与临床应用 

**Authors**: Ziyu Li, Yujian Hu, Zhengyao Ding, Yiheng Mao, Haitao Li, Fan Yi, Hongkun Zhang, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03426)  

**Abstract**: Cardiac Magnetic Resonance (CMR) imaging is a vital non-invasive tool for diagnosing heart diseases and evaluating cardiac health. However, the limited availability of large-scale, high-quality CMR datasets poses a major challenge to the effective application of artificial intelligence (AI) in this domain. Even the amount of unlabeled data and the health status it covers are difficult to meet the needs of model pretraining, which hinders the performance of AI models on downstream tasks. In this study, we present Cardiac Phenotype-Guided CMR Generation (CPGG), a novel approach for generating diverse CMR data that covers a wide spectrum of cardiac health status. The CPGG framework consists of two stages: in the first stage, a generative model is trained using cardiac phenotypes derived from CMR data; in the second stage, a masked autoregressive diffusion model, conditioned on these phenotypes, generates high-fidelity CMR cine sequences that capture both structural and functional features of the heart in a fine-grained manner. We synthesized a massive amount of CMR to expand the pretraining data. Experimental results show that CPGG generates high-quality synthetic CMR data, significantly improving performance on various downstream tasks, including diagnosis and cardiac phenotypes prediction. These gains are demonstrated across both public and private datasets, highlighting the effectiveness of our approach. Code is availabel at this https URL. 

**Abstract (ZH)**: 基于心脏表型引导的心磁共振成像生成（CPGG） 

---
# Framework GNN-AID: Graph Neural Network Analysis Interpretation and Defense 

**Title (ZH)**: GNN-AID框架：图神经网络分析、解释与防御 

**Authors**: Kirill Lukyanov, Mikhail Drobyshevskiy, Georgii Sazonov, Mikhail Soloviov, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2505.03424)  

**Abstract**: The growing need for Trusted AI (TAI) highlights the importance of interpretability and robustness in machine learning models. However, many existing tools overlook graph data and rarely combine these two aspects into a single solution. Graph Neural Networks (GNNs) have become a popular approach, achieving top results across various tasks. We introduce GNN-AID (Graph Neural Network Analysis, Interpretation, and Defense), an open-source framework designed for graph data to address this gap. Built as a Python library, GNN-AID supports advanced trust methods and architectural layers, allowing users to analyze graph datasets and GNN behavior using attacks, defenses, and interpretability methods.
GNN-AID is built on PyTorch-Geometric, offering preloaded datasets, models, and support for any GNNs through customizable interfaces. It also includes a web interface with tools for graph visualization and no-code features like an interactive model builder, simplifying the exploration and analysis of GNNs. The framework also supports MLOps techniques, ensuring reproducibility and result versioning to track and revisit analyses efficiently.
GNN-AID is a flexible tool for developers and researchers. It helps developers create, analyze, and customize graph models, while also providing access to prebuilt datasets and models for quick experimentation. Researchers can use the framework to explore advanced topics on the relationship between interpretability and robustness, test defense strategies, and combine methods to protect against different types of attacks.
We also show how defenses against evasion and poisoning attacks can conflict when applied to graph data, highlighting the complex connections between defense strategies.
GNN-AID is available at \href{this https URL}{this http URL} 

**Abstract (ZH)**: 可信人工智能（Trusted AI）的需求增长突显了可解释性和鲁棒性在机器学习模型中的重要性。然而，许多现有工具忽视了图数据，并且很少将这两者结合成一个解决方案。图神经网络（GNNs）已成为一种流行的方法，在各种任务中取得了顶尖的结果。我们介绍了GNN-AID（图神经网络的分析、解释和防御），这是一个开源框架，旨在为图数据填补这一空白。作为Python库构建，GNN-AID支持先进的信任方法和架构层，允许用户通过攻击、防御和可解释性方法分析图数据集和GNN行为。

GNN-AID基于PyTorch-Geometric构建，提供了预加载的数据集、模型，并且通过自定义接口支持任何GNN。它还包括一个网络界面，包含用于图形可视化和无代码功能（如交互模型构建器）的工具，简化了GNN的探索和分析。该框架还支持MLOps技术，确保可重复性和结果版本化，从而高效地跟踪和回顾分析。

GNN-AID是一种灵活的工具，适用于开发者和研究人员。它帮助开发者创建、分析和自定义图模型，同时提供快速实验所需的预构建数据集和模型的访问权限。研究人员可以使用该框架探索可解释性和鲁棒性之间的关系，测试防御策略，并结合方法以保护免受不同类型攻击的影响。

我们还展示了当应用于图数据时，针对欺骗和投毒攻击的防御策略可能会产生冲突，突显了防御策略之间复杂的关系。

GNN-AID可在\href{this https URL}{this http URL}获取。 

---
# Lightweight Clinical Decision Support System using QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation 

**Title (ZH)**: 基于QLoRA微调的大模型和检索增强生成的轻量级临床决策支持系统 

**Authors**: Mohammad Shoaib Ansari, Mohd Sohail Ali Khan, Shubham Revankar, Aditya Varma, Anil S. Mokhade  

**Link**: [PDF](https://arxiv.org/pdf/2505.03406)  

**Abstract**: This research paper investigates the application of Large Language Models (LLMs) in healthcare, specifically focusing on enhancing medical decision support through Retrieval-Augmented Generation (RAG) integrated with hospital-specific data and fine-tuning using Quantized Low-Rank Adaptation (QLoRA). The system utilizes Llama 3.2-3B-Instruct as its foundation model. By embedding and retrieving context-relevant healthcare information, the system significantly improves response accuracy. QLoRA facilitates notable parameter efficiency and memory optimization, preserving the integrity of medical information through specialized quantization techniques. Our research also shows that our model performs relatively well on various medical benchmarks, indicating that it can be used to make basic medical suggestions. This paper details the system's technical components, including its architecture, quantization methods, and key healthcare applications such as enhanced disease prediction from patient symptoms and medical history, treatment suggestions, and efficient summarization of complex medical reports. We touch on the ethical considerations-patient privacy, data security, and the need for rigorous clinical validation-as well as the practical challenges of integrating such systems into real-world healthcare workflows. Furthermore, the lightweight quantized weights ensure scalability and ease of deployment even in low-resource hospital environments. Finally, the paper concludes with an analysis of the broader impact of LLMs on healthcare and outlines future directions for LLMs in medical settings. 

**Abstract (ZH)**: 本研究论文探讨了大型语言模型（LLMs）在医疗领域的应用，特别关注通过结合医院特定数据和量化低秩适应（QLoRA）技术的检索增强生成（RAG）来提升医疗决策支持系统。该系统以Llama 3.2-3B-Instruct作为基础模型。通过嵌入和检索相关医疗信息，系统显著提高了响应准确性。QLoRA促进参数效率和内存优化，并通过专门的量化技术保持了医疗信息的完整性。研究表明，该模型在多种医疗基准测试中表现出色，表明其可用于提供基本的医疗建议。本文详细介绍了系统的技术组件，包括其架构、量化方法以及增强疾病预测、治疗建议和复杂医疗报告的高效总结等关键医疗应用。论文还讨论了伦理考虑（如患者隐私、数据安全和严格的临床验证需求）以及将此类系统整合到实际医疗工作流程中的实际挑战。轻量化的量化权重确保即使在资源有限的医院环境中，系统也具有可扩展性和易于部署的特点。最后，论文分析了大型语言模型在医疗领域的广泛影响，并概述了未来医疗环境中大型语言模型的发展方向。 

---
# DDaTR: Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation 

**Title (ZH)**: DDaTR：动态差异感知的时序残差网络在纵向放射学报告生成中的应用 

**Authors**: Shanshan Song, Hui Tang, Honglong Yang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03401)  

**Abstract**: Radiology Report Generation (RRG) automates the creation of radiology reports from medical imaging, enhancing the efficiency of the reporting process. Longitudinal Radiology Report Generation (LRRG) extends RRG by incorporating the ability to compare current and prior exams, facilitating the tracking of temporal changes in clinical findings. Existing LRRG approaches only extract features from prior and current images using a visual pre-trained encoder, which are then concatenated to generate the final report. However, these methods struggle to effectively capture both spatial and temporal correlations during the feature extraction process. Consequently, the extracted features inadequately capture the information of difference across exams and thus underrepresent the expected progressions, leading to sub-optimal performance in LRRG. To address this, we develop a novel dynamic difference-aware temporal residual network (DDaTR). In DDaTR, we introduce two modules at each stage of the visual encoder to capture multi-level spatial correlations. The Dynamic Feature Alignment Module (DFAM) is designed to align prior features across modalities for the integrity of prior clinical information. Prompted by the enriched prior features, the dynamic difference-aware module (DDAM) captures favorable difference information by identifying relationships across exams. Furthermore, our DDaTR employs the dynamic residual network to unidirectionally transmit longitudinal information, effectively modelling temporal correlations. Extensive experiments demonstrated superior performance over existing methods on three benchmarks, proving its efficacy in both RRG and LRRG tasks. 

**Abstract (ZH)**: 放射报告生成（RRG）自动化医疗成像的放射报告创建，提升报告过程的效率。纵向放射报告生成（LRRG）通过整合当前和以往检查的对比能力，促进临床发现的时间变化跟踪。现有的LRRG方法仅使用视觉预训练编码器从以往和当前图像中提取特征，然后将这些特征连接起来生成最终报告。然而，这些方法在特征提取过程中难以有效捕捉空间和时间相关性，导致提取的特征未能充分捕捉考试之间的差异信息，从而无法充分代表预期的变化，导致LRRG表现不佳。为解决这一问题，我们开发了一种新颖的动态差异感知时间残差网络（DDaTR）。在DDaTR中，在视觉编码器的每个阶段引入两个模块以捕捉多级空间相关性。动态特征对齐模块（DFAM）旨在跨模态对齐以往特征，以确保以往临床信息的完整性。受丰富后的以往特征启发，动态差异感知模块（DDAM）通过识别考试间的关联来捕捉有利的差异信息。此外，我们的DDaTR采用动态残差网络单向传递纵向信息，有效建模时间相关性。实验结果证明，DDaTR在三个基准测试上的性能优于现有方法，验证了其在RRG和LRRG任务中的有效性。 

---
# Automatic Calibration for Membership Inference Attack on Large Language Models 

**Title (ZH)**: 针对大规模语言模型的成员推断攻击的自动校准 

**Authors**: Saleh Zare Zade, Yao Qiang, Xiangyu Zhou, Hui Zhu, Mohammad Amin Roshani, Prashant Khanduri, Dongxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03392)  

**Abstract**: Membership Inference Attacks (MIAs) have recently been employed to determine whether a specific text was part of the pre-training data of Large Language Models (LLMs). However, existing methods often misinfer non-members as members, leading to a high false positive rate, or depend on additional reference models for probability calibration, which limits their practicality. To overcome these challenges, we introduce a novel framework called Automatic Calibration Membership Inference Attack (ACMIA), which utilizes a tunable temperature to calibrate output probabilities effectively. This approach is inspired by our theoretical insights into maximum likelihood estimation during the pre-training of LLMs. We introduce ACMIA in three configurations designed to accommodate different levels of model access and increase the probability gap between members and non-members, improving the reliability and robustness of membership inference. Extensive experiments on various open-source LLMs demonstrate that our proposed attack is highly effective, robust, and generalizable, surpassing state-of-the-art baselines across three widely used benchmarks. Our code is available at: \href{this https URL}{\textcolor{blue}{Github}}. 

**Abstract (ZH)**: 自动校准会员推断攻击 (ACMIA)：一种利用可调温度有效校准输出概率的新框架 

---
# Reinforced Correlation Between Vision and Language for Precise Medical AI Assistant 

**Title (ZH)**: 视觉与语言强化关联以实现精准医疗AI助手 

**Authors**: Haonan Wang, Jiaji Mao, Lehan Wang, Qixiang Zhang, Marawan Elbatel, Yi Qin, Huijun Hu, Baoxun Li, Wenhui Deng, Weifeng Qin, Hongrui Li, Jialin Liang, Jun Shen, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03380)  

**Abstract**: Medical AI assistants support doctors in disease diagnosis, medical image analysis, and report generation. However, they still face significant challenges in clinical use, including limited accuracy with multimodal content and insufficient validation in real-world settings. We propose RCMed, a full-stack AI assistant that improves multimodal alignment in both input and output, enabling precise anatomical delineation, accurate localization, and reliable diagnosis through hierarchical vision-language grounding. A self-reinforcing correlation mechanism allows visual features to inform language context, while language semantics guide pixel-wise attention, forming a closed loop that refines both modalities. This correlation is enhanced by a color region description strategy, translating anatomical structures into semantically rich text to learn shape-location-text relationships across scales. Trained on 20 million image-mask-description triplets, RCMed achieves state-of-the-art precision in contextualizing irregular lesions and subtle anatomical boundaries, excelling in 165 clinical tasks across 9 modalities. It achieved a 23.5% relative improvement in cell segmentation from microscopy images over prior methods. RCMed's strong vision-language alignment enables exceptional generalization, with state-of-the-art performance in external validation across 20 clinically significant cancer types, including novel tasks. This work demonstrates how integrated multimodal models capture fine-grained patterns, enabling human-level interpretation in complex scenarios and advancing human-centric AI healthcare. 

**Abstract (ZH)**: Medical AI助手支持医生进行疾病诊断、医学图像分析和报告生成，但在临床应用中仍面临显著挑战，包括多模态内容的限制造准确性以及在现实环境中的不足验证。我们提出RCMed全栈AI助手，通过在输入和输出中改进多模态对齐，实现精确的解剖轮廓化、准确的定位和可靠的诊断，借助层级视觉-语言定位。自我强化的相关机制允许视觉特征指导语言上下文，而语言语义引导像素级注意力，形成封闭循环以精炼两种模态。这种相关性通过颜色区域描述策略增强，将解剖结构转化为语义丰富的文本以学习跨尺度的形状-位置-文本关系。RCMed基于2000万个图像-掩码-描述三元组训练，在165项临床任务的9种模态下实现最先进的上下文信息关联精确度，特别是在显微镜图像中的细胞分割任务上相对改进了23.5%。RCMed强大的视觉-语言对齐使其具有出色的泛化能力，在20种临床显著的癌症类型中的外部验证中达到最先进的表现，包括新的任务。这项工作展示了一个集成多模态模型如何捕捉细微特征，以复杂场景中实现人类级别的解释，并推进以人类为中心的人工智能医疗健康。 

---
# SPAP: Structured Pruning via Alternating Optimization and Penalty Methods 

**Title (ZH)**: SPAP：交替优化和惩罚方法引导的结构化剪枝 

**Authors**: Hanyu Hu, Xiaoming Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03373)  

**Abstract**: The deployment of large language models (LLMs) is often constrained by their substantial computational and memory demands. While structured pruning presents a viable approach by eliminating entire network components, existing methods suffer from performance degradation, reliance on heuristic metrics, or expensive finetuning. To address these challenges, we propose SPAP (Structured Pruning via Alternating Optimization and Penalty Methods), a novel and efficient structured pruning framework for LLMs grounded in optimization theory. SPAP formulates the pruning problem through a mixed-integer optimization model, employs a penalty method that effectively makes pruning decisions to minimize pruning errors, and introduces an alternating minimization algorithm tailored to the splittable problem structure for efficient weight updates and performance recovery. Extensive experiments on OPT, LLaMA-3/3.1/3.2, and Qwen2.5 models demonstrate SPAP's superiority over state-of-the-art methods, delivering linear inference speedups (1.29$\times$ at 30% sparsity) and proportional memory reductions. Our work offers a practical, optimization-driven solution for pruning LLMs while preserving model performance. 

**Abstract (ZH)**: 大型语言模型（LLM）的部署常常受限于其巨大的计算和内存需求。尽管结构化剪枝提供了一种可行的方法通过消除整个网络组件来缓解这一问题，但现有方法存在性能下降、依赖于启发式指标或昂贵的微调等问题。为了解决这些挑战，我们提出了一种基于优化理论的新颖且高效的结构化剪枝框架SPAP（交替优化和惩罚方法的结构化剪枝）。SPAP通过混合整数优化模型来形式化剪枝问题，采用惩罚方法有效地做出剪枝决策以最小化剪枝误差，并引入了一种适合拆分问题结构的交替最小化算法，以实现高效权重更新和性能恢复。在OPT、LLaMA-3/3.1/3.2和Qwen2.5模型上的广泛实验表明，SPAP在性能上优于最先进的方法，提供线性推理加速（在30%稀疏性下为1.29倍）和按比例减少的内存占用。我们的工作提供了一种实用的、基于优化的解决方案，可在保持模型性能的同时进行LLM剪枝。 

---
# Safer Prompts: Reducing IP Risk in Visual Generative AI 

**Title (ZH)**: 更安全的提示：降低视觉生成AI中的IP风险 

**Authors**: Lena Reissinger, Yuanyuan Li, Anna-Carolina Haensch, Neeraj Sarna  

**Link**: [PDF](https://arxiv.org/pdf/2505.03338)  

**Abstract**: Visual Generative AI models have demonstrated remarkable capability in generating high-quality images from simple inputs like text prompts. However, because these models are trained on images from diverse sources, they risk memorizing and reproducing specific content, raising concerns about intellectual property (IP) infringement. Recent advances in prompt engineering offer a cost-effective way to enhance generative AI performance. In this paper, we evaluate the effectiveness of prompt engineering techniques in mitigating IP infringement risks in image generation. Our findings show that Chain of Thought Prompting and Task Instruction Prompting significantly reduce the similarity between generated images and the training data of diffusion models, thereby lowering the risk of IP infringement. 

**Abstract (ZH)**: 视觉生成AI模型展示了从简单的输入如文本提示生成高质量图像的非凡能力。然而，由于这些模型是在多种来源的图像上进行训练的，它们存在记忆和复制特定内容的风险，这引发了关于知识产权（IP）侵权的担忧。最近的提示工程技术进步为提高生成型AI性能提供了成本有效的途径。本研究评估了提示工程技术在减轻图像生成中的IP侵权风险方面的有效性。我们的研究发现，Chain of Thought Prompting和Task Instruction Prompting显著降低了生成图像与扩散模型训练数据之间的相似性，从而降低了IP侵权的风险。 

---
# Avoid Recommending Out-of-Domain Items: Constrained Generative Recommendation with LLMs 

**Title (ZH)**: 避免推荐领域外项目：受约束的生成式推荐方法（使用大语言模型） 

**Authors**: Hao Liao, Wensheng Lu, Jianxun Lian, Mingqi Wu, Shuo Wang, Yong Zhang, Yitian Huang, Mingyang Zhou, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.03336)  

**Abstract**: Large Language Models (LLMs) have shown promise for generative recommender systems due to their transformative capabilities in user interaction. However, ensuring they do not recommend out-of-domain (OOD) items remains a challenge. We study two distinct methods to address this issue: RecLM-ret, a retrieval-based method, and RecLM-cgen, a constrained generation method. Both methods integrate seamlessly with existing LLMs to ensure in-domain recommendations. Comprehensive experiments on three recommendation datasets demonstrate that RecLM-cgen consistently outperforms RecLM-ret and existing LLM-based recommender models in accuracy while eliminating OOD recommendations, making it the preferred method for adoption. Additionally, RecLM-cgen maintains strong generalist capabilities and is a lightweight plug-and-play module for easy integration into LLMs, offering valuable practical benefits for the community. Source code is available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成推荐系统中的应用展现了潜力，由于其在用户交互方面的变革性能力。然而，确保它们不会推荐领域外（OOD）的项目仍然是一项挑战。我们研究了两种不同的方法来解决这一问题：RecLM-ret，一种检索基方法，和RecLM-cgen，一种受限生成方法。这两种方法能无缝集成到现有的LLM中，以确保推荐的领域相关性。在三个推荐数据集上的全面实验表明，RecLM-cgen在准确性上始终优于RecLM-ret和现有的基于LLM的推荐模型，并且能够消除领域外推荐，使其成为首选的采用方法。此外，RecLM-cgen保持了强大的通识能力，并且是一个轻量级即插即用模块，易于集成到LLM中，为社区提供重要的实际益处。源代码可在此处访问：this https URL。 

---
# Absolute Zero: Reinforced Self-play Reasoning with Zero Data 

**Title (ZH)**: 绝对零度：零数据强化自我博弈推理 

**Authors**: Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03335)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown promise in enhancing the reasoning capabilities of large language models by learning directly from outcome-based rewards. Recent RLVR works that operate under the zero setting avoid supervision in labeling the reasoning process, but still depend on manually curated collections of questions and answers for training. The scarcity of high-quality, human-produced examples raises concerns about the long-term scalability of relying on human supervision, a challenge already evident in the domain of language model pretraining. Furthermore, in a hypothetical future where AI surpasses human intelligence, tasks provided by humans may offer limited learning potential for a superintelligent system. To address these concerns, we propose a new RLVR paradigm called Absolute Zero, in which a single model learns to propose tasks that maximize its own learning progress and improves reasoning by solving them, without relying on any external data. Under this paradigm, we introduce the Absolute Zero Reasoner (AZR), a system that self-evolves its training curriculum and reasoning ability by using a code executor to both validate proposed code reasoning tasks and verify answers, serving as an unified source of verifiable reward to guide open-ended yet grounded learning. Despite being trained entirely without external data, AZR achieves overall SOTA performance on coding and mathematical reasoning tasks, outperforming existing zero-setting models that rely on tens of thousands of in-domain human-curated examples. Furthermore, we demonstrate that AZR can be effectively applied across different model scales and is compatible with various model classes. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）通过直接从基于结果的奖励中学习，在增强大型语言模型的推理能力方面展现了潜力。Recent RLVR工作在零设定环境中运行，避免了标注推理过程的监督，但仍依赖于手动整理的问题和答案集合进行训练。高质量的人工生成示例的稀缺性引发了对长期依赖人工监督可行性的担忧，这一挑战在语言模型预训练领域已经显现。此外，在人工智能超越人类智能的假设未来中，人类提供的任务可能对超智能系统的学习潜力有限。为应对这些担忧，我们提出了一种新的RLVR paradigm，称为Absolute Zero，在该 paradigm中，单个模型学会提出最大化自身学习进度的任务，并通过解决这些任务来提高推理能力，而不依赖任何外部数据。在这一paradigm中，我们引入了Absolute Zero Reasoner（AZR）系统，该系统利用代码执行器验证提议的代码推理任务并验证答案，作为统一的验证奖励来源，引导开放而具体的learnings。尽管完全不依赖外部数据进行训练，AZR在编程和数学推理任务上的整体性能仍达到了SOTA水平，超越了依赖数万个人工整理示例的现有零设定模型。此外，我们证明了AZR可以在不同的模型规模上有效应用，并与多种模型类兼容。 

---
# Very High-Resolution Forest Mapping with TanDEM-X InSAR Data and Self-Supervised Learning 

**Title (ZH)**: Very High-Resolution Forest Mapping with TanDEM-X InSAR Data and Self-Supervised Learning 

**Authors**: José-Luis Bueso-Bello, Benjamin Chauvel, Daniel Carcereri, Philipp Posovszky, Pietro Milillo, Jennifer Ruiz, Juan-Carlos Fernández-Diaz, Carolina González, Michele Martone, Ronny Hänsch, Paola Rizzoli  

**Link**: [PDF](https://arxiv.org/pdf/2505.03327)  

**Abstract**: Deep learning models have shown encouraging capabilities for mapping accurately forests at medium resolution with TanDEM-X interferometric SAR data. Such models, as most of current state-of-the-art deep learning techniques in remote sensing, are trained in a fully-supervised way, which requires a large amount of labeled data for training and validation. In this work, our aim is to exploit the high-resolution capabilities of the TanDEM-X mission to map forests at 6 m. The goal is to overcome the intrinsic limitations posed by midresolution products, which affect, e.g., the detection of narrow roads within vegetated areas and the precise delineation of forested regions contours. To cope with the lack of extended reliable reference datasets at such a high resolution, we investigate self-supervised learning techniques for extracting highly informative representations from the input features, followed by a supervised training step with a significantly smaller number of reliable labels. A 1 m resolution forest/non-forest reference map over Pennsylvania, USA, allows for comparing different training approaches for the development of an effective forest mapping framework with limited labeled samples. We select the best-performing approach over this test region and apply it in a real-case forest mapping scenario over the Amazon rainforest, where only very few labeled data at high resolution are available. In this challenging scenario, the proposed self-supervised framework significantly enhances the classification accuracy with respect to fully-supervised methods, trained using the same amount of labeled data, representing an extremely promising starting point for large-scale, very high-resolution forest mapping with TanDEM-X data. 

**Abstract (ZH)**: 利用TanDEM-X干涉雷达SAR数据进行6米分辨率森林制图的深度学习模型研究：自监督学习方法在高分辨率森林制图中的应用 

---
# SD-VSum: A Method and Dataset for Script-Driven Video Summarization 

**Title (ZH)**: SD-VSum: 一种基于脚本的视频摘要方法及数据集 

**Authors**: Manolis Mylonas, Evlampios Apostolidis, Vasileios Mezaris  

**Link**: [PDF](https://arxiv.org/pdf/2505.03319)  

**Abstract**: In this work, we introduce the task of script-driven video summarization, which aims to produce a summary of the full-length video by selecting the parts that are most relevant to a user-provided script outlining the visual content of the desired summary. Following, we extend a recently-introduced large-scale dataset for generic video summarization (VideoXum) by producing natural language descriptions of the different human-annotated summaries that are available per video. In this way we make it compatible with the introduced task, since the available triplets of ``video, summary and summary description'' can be used for training a method that is able to produce different summaries for a given video, driven by the provided script about the content of each summary. Finally, we develop a new network architecture for script-driven video summarization (SD-VSum), that relies on the use of a cross-modal attention mechanism for aligning and fusing information from the visual and text modalities. Our experimental evaluations demonstrate the advanced performance of SD-VSum against state-of-the-art approaches for query-driven and generic (unimodal and multimodal) summarization from the literature, and document its capacity to produce video summaries that are adapted to each user's needs about their content. 

**Abstract (ZH)**: 基于脚本的视频摘要任务及其在网络中实现的研究 

---
# Mamba-Diffusion Model with Learnable Wavelet for Controllable Symbolic Music Generation 

**Title (ZH)**: 具有可学习小波的Mamba-扩散模型可控符号Music生成 

**Authors**: Jincheng Zhang, György Fazekas, Charalampos Saitis  

**Link**: [PDF](https://arxiv.org/pdf/2505.03314)  

**Abstract**: The recent surge in the popularity of diffusion models for image synthesis has attracted new attention to their potential for generation tasks in other domains. However, their applications to symbolic music generation remain largely under-explored because symbolic music is typically represented as sequences of discrete events and standard diffusion models are not well-suited for discrete data. We represent symbolic music as image-like pianorolls, facilitating the use of diffusion models for the generation of symbolic music. Moreover, this study introduces a novel diffusion model that incorporates our proposed Transformer-Mamba block and learnable wavelet transform. Classifier-free guidance is utilised to generate symbolic music with target chords. Our evaluation shows that our method achieves compelling results in terms of music quality and controllability, outperforming the strong baseline in pianoroll generation. Our code is available at this https URL. 

**Abstract (ZH)**: 近期扩散模型在图像生成中的流行 resurgence 重新引发了人们对其他领域生成任务潜在应用的关注。然而，由于符号音乐通常表示为离散事件序列，而标准扩散模型不适用于离散数据，因此将其应用于符号音乐生成的研究仍相对较少。我们通过将符号音乐表示为类似图像的钢琴卷帘图，促进了扩散模型在符号音乐生成中的应用。此外，本研究引入了一种新型扩散模型，该模型结合了我们提出的Transformer-Mamba模块和可学习小波变换。我们利用无分类器引导生成具有目标和弦的符号音乐。评估结果显示，我们的方法在音乐质量和可控性方面取得了令人印象深刻的结果，并在钢琴卷帘图生成中超过了强大的基线模型。我们的代码可在以下链接获取。 

---
# Comparative Analysis of Lightweight Deep Learning Models for Memory-Constrained Devices 

**Title (ZH)**: 轻量级深学习模型在内存约束设备上的比较分析 

**Authors**: Tasnim Shahriar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03303)  

**Abstract**: This paper presents a comprehensive evaluation of lightweight deep learning models for image classification, emphasizing their suitability for deployment in resource-constrained environments such as low-memory devices. Five state-of-the-art architectures - MobileNetV3 Small, ResNet18, SqueezeNet, EfficientNetV2-S, and ShuffleNetV2 - are benchmarked across three diverse datasets: CIFAR-10, CIFAR-100, and Tiny ImageNet. The models are assessed using four key performance metrics: classification accuracy, inference time, floating-point operations (FLOPs), and model size. Additionally, we investigate the impact of hyperparameter tuning, data augmentation, and training paradigms by comparing pretrained models with scratch-trained counterparts, focusing on MobileNetV3 Small. Our findings reveal that transfer learning significantly enhances model accuracy and computational efficiency, particularly for complex datasets like Tiny ImageNet. EfficientNetV2 consistently achieves the highest accuracy, while MobileNetV3 offers the best balance between accuracy and efficiency, and SqueezeNet excels in inference speed and compactness. This study highlights critical trade-offs between accuracy and efficiency, offering actionable insights for deploying lightweight models in real-world applications where computational resources are limited. By addressing these challenges, this research contributes to optimizing deep learning systems for edge computing and mobile platforms. 

**Abstract (ZH)**: 这种轻量级深度学习模型在图像分类中的综合评估：面向资源受限环境的应用 

---
# Towards Efficient Benchmarking of Foundation Models in Remote Sensing: A Capabilities Encoding Approach 

**Title (ZH)**: 面向遥感领域的基础模型高效基准测试：一种能力编码方法 

**Authors**: Pierre Adorni, Minh-Tan Pham, Stéphane May, Sébastien Lefèvre  

**Link**: [PDF](https://arxiv.org/pdf/2505.03299)  

**Abstract**: Foundation models constitute a significant advancement in computer vision: after a single, albeit costly, training phase, they can address a wide array of tasks. In the field of Earth observation, over 75 remote sensing vision foundation models have been developed in the past four years. However, none has consistently outperformed the others across all available downstream tasks. To facilitate their comparison, we propose a cost-effective method for predicting a model's performance on multiple downstream tasks without the need for fine-tuning on each one. This method is based on what we call "capabilities encoding." The utility of this novel approach is twofold: we demonstrate its potential to simplify the selection of a foundation model for a given new task, and we employ it to offer a fresh perspective on the existing literature, suggesting avenues for future research. Codes are available at this https URL. 

**Abstract (ZH)**: 基于成本效益的方法预测遥感视觉基础模型在多种下游任务上的性能：一种能力编码的新途径 

---
# The Unreasonable Effectiveness of Discrete-Time Gaussian Process Mixtures for Robot Policy Learning 

**Title (ZH)**: 离散时间高斯过程混合模型在机器人策略学习中的意外有效性 

**Authors**: Jan Ole von Hartz, Adrian Röfer, Joschka Boedecker, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2505.03296)  

**Abstract**: We present Mixture of Discrete-time Gaussian Processes (MiDiGap), a novel approach for flexible policy representation and imitation learning in robot manipulation. MiDiGap enables learning from as few as five demonstrations using only camera observations and generalizes across a wide range of challenging tasks. It excels at long-horizon behaviors such as making coffee, highly constrained motions such as opening doors, dynamic actions such as scooping with a spatula, and multimodal tasks such as hanging a mug. MiDiGap learns these tasks on a CPU in less than a minute and scales linearly to large datasets. We also develop a rich suite of tools for inference-time steering using evidence such as collision signals and robot kinematic constraints. This steering enables novel generalization capabilities, including obstacle avoidance and cross-embodiment policy transfer. MiDiGap achieves state-of-the-art performance on diverse few-shot manipulation benchmarks. On constrained RLBench tasks, it improves policy success by 76 percentage points and reduces trajectory cost by 67%. On multimodal tasks, it improves policy success by 48 percentage points and increases sample efficiency by a factor of 20. In cross-embodiment transfer, it more than doubles policy success. We make the code publicly available at this https URL. 

**Abstract (ZH)**: 混合离散时间高斯过程在机器人操作中的灵活策略表示与模仿学习 

---
# Physics-inspired Energy Transition Neural Network for Sequence Learning 

**Title (ZH)**: 基于物理启发的能量转换神经网络序列表征 

**Authors**: Zhou Wu, Junyi An, Baile Xu, Furao Shen, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.03281)  

**Abstract**: Recently, the superior performance of Transformers has made them a more robust and scalable solution for sequence modeling than traditional recurrent neural networks (RNNs). However, the effectiveness of Transformer in capturing long-term dependencies is primarily attributed to their comprehensive pair-modeling process rather than inherent inductive biases toward sequence semantics. In this study, we explore the capabilities of pure RNNs and reassess their long-term learning mechanisms. Inspired by the physics energy transition models that track energy changes over time, we propose a effective recurrent structure called the``Physics-inspired Energy Transition Neural Network" (PETNN). We demonstrate that PETNN's memory mechanism effectively stores information over long-term dependencies. Experimental results indicate that PETNN outperforms transformer-based methods across various sequence tasks. Furthermore, owing to its recurrent nature, PETNN exhibits significantly lower complexity. Our study presents an optimal foundational recurrent architecture and highlights the potential for developing effective recurrent neural networks in fields currently dominated by Transformer. 

**Abstract (ZH)**: 最近，Transformer 的出色性能使其成为比传统递归神经网络（RNNs）更 robust 和可扩展的序列建模解决方案。然而，Transformer 在捕捉长期依赖关系方面的有效性主要归因于其全面的配对建模过程，而非固有的序列语义归纳偏置。在本研究中，我们探索了纯 RNN 的能力并重新评估了其长期学习机制。受物理能量转移模型的启发，该模型随时间追踪能量变化，我们提出了一种有效的递归结构，称为“物理启发的能量转移神经网络”（PETNN）。研究表明，PETNN 的记忆机制有效地存储了长期依赖关系中的信息。实验结果表明，PETNN 在各种序列任务中优于基于 Transformer 的方法。此外，由于其递归性质，PETNN 的复杂性显著较低。本研究展示了最优的基础递归架构，并突显了在目前由 Transformer 占据主导地位的领域开发有效递归神经网络的潜力。 

---
# Synthline: A Product Line Approach for Synthetic Requirements Engineering Data Generation using Large Language Models 

**Title (ZH)**: Synthline：一种使用大型语言模型生成合成需求工程数据的产品线方法 

**Authors**: Abdelkarim El-Hajjami, Camille Salinesi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03265)  

**Abstract**: While modern Requirements Engineering (RE) heavily relies on natural language processing and Machine Learning (ML) techniques, their effectiveness is limited by the scarcity of high-quality datasets. This paper introduces Synthline, a Product Line (PL) approach that leverages Large Language Models to systematically generate synthetic RE data for classification-based use cases. Through an empirical evaluation conducted in the context of using ML for the identification of requirements specification defects, we investigated both the diversity of the generated data and its utility for training downstream models. Our analysis reveals that while synthetic datasets exhibit less diversity than real data, they are good enough to serve as viable training resources. Moreover, our evaluation shows that combining synthetic and real data leads to substantial performance improvements. Specifically, hybrid approaches achieve up to 85% improvement in precision and a 2x increase in recall compared to models trained exclusively on real data. These findings demonstrate the potential of PL-based synthetic data generation to address data scarcity in RE. We make both our implementation and generated datasets publicly available to support reproducibility and advancement in the field. 

**Abstract (ZH)**: Synthline：基于产品线的大语言模型驱动的合成需求工程数据生成 

---
# Seeing the Abstract: Translating the Abstract Language for Vision Language Models 

**Title (ZH)**: 看清摘要：为视觉语言模型翻译摘要语言 

**Authors**: Davide Talon, Federico Girella, Ziyue Liu, Marco Cristani, Yiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03242)  

**Abstract**: Natural language goes beyond dryly describing visual content. It contains rich abstract concepts to express feeling, creativity and properties that cannot be directly perceived. Yet, current research in Vision Language Models (VLMs) has not shed light on abstract-oriented language. Our research breaks new ground by uncovering its wide presence and under-estimated value, with extensive analysis. Particularly, we focus our investigation on the fashion domain, a highly-representative field with abstract expressions. By analyzing recent large-scale multimodal fashion datasets, we find that abstract terms have a dominant presence, rivaling the concrete ones, providing novel information, and being useful in the retrieval task. However, a critical challenge emerges: current general-purpose or fashion-specific VLMs are pre-trained with databases that lack sufficient abstract words in their text corpora, thus hindering their ability to effectively represent abstract-oriented language. We propose a training-free and model-agnostic method, Abstract-to-Concrete Translator (ACT), to shift abstract representations towards well-represented concrete ones in the VLM latent space, using pre-trained models and existing multimodal databases. On the text-to-image retrieval task, despite being training-free, ACT outperforms the fine-tuned VLMs in both same- and cross-dataset settings, exhibiting its effectiveness with a strong generalization capability. Moreover, the improvement introduced by ACT is consistent with various VLMs, making it a plug-and-play solution. 

**Abstract (ZH)**: 自然语言超越了对视觉内容的枯燥描述，包含了丰富的抽象概念以表达感觉、创造力和无法直接感知的属性。然而，当前的视觉语言模型研究尚未关注抽象导向的语言。我们的研究开拓了新的领域，揭示了抽象导向语言的广泛存在及其被低估的价值，并进行了广泛的分析。特别是，我们将调查集中在时尚领域，这是一个富含抽象表达的高度代表性领域。通过分析大规模多模态时尚数据集，我们发现抽象词汇占据了主导地位，与具体的词汇不相上下，提供新的信息，并在检索任务中具有实用价值。然而，一个关键挑战出现了：当前的通用或特定于时尚的视觉语言模型是基于数据库进行预训练的，而这些数据库在文本语料库中缺乏足够的抽象词汇，从而阻碍了它们对抽象导向语言的有效表示。我们提出了一种无需训练且模型无关的方法——抽象到具体翻译器（ACT），通过使用预训练模型和现有的多模态数据库，将抽象表示移向了代表充分的具体表示。在文本到图像检索任务中，尽管ACT是无需训练的，但在同数据库和跨数据库设置中，其性能均超过了微调的视觉语言模型，展示了其强大的泛化能力。此外，由ACT带来的改进在各种视觉语言模型中是一致的，使其成为即插即用的解决方案。 

---
# Accelerating Evolution: Integrating PSO Principles into Real-Coded Genetic Algorithm Crossover 

**Title (ZH)**: 加速进化：将粒子 swarm 算子集成到实编码遗传算法交叉中 

**Authors**: Xiaobo Jin, JiaShu Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03217)  

**Abstract**: This study introduces an innovative crossover operator named Particle Swarm Optimization-inspired Crossover (PSOX), which is specifically developed for real-coded genetic algorithms. Departing from conventional crossover approaches that only exchange information between individuals within the same generation, PSOX uniquely incorporates guidance from both the current global best solution and historical optimal solutions across multiple generations. This novel mechanism enables the algorithm to maintain population diversity while simultaneously accelerating convergence toward promising regions of the search space. The effectiveness of PSOX is rigorously evaluated through comprehensive experiments on 15 benchmark test functions with diverse characteristics, including unimodal, multimodal, and highly complex landscapes. Comparative analysis against five state-of-the-art crossover operators reveals that PSOX consistently delivers superior performance in terms of solution accuracy, algorithmic stability, and convergence speed, especially when combined with an appropriate mutation strategy. Furthermore, the study provides an in-depth investigation of how different mutation rates influence PSOX's performance, yielding practical guidelines for parameter tuning when addressing optimization problems with varying landscape properties. 

**Abstract (ZH)**: 基于粒子群优化启发的交叉算子研究：一种用于实码遗传算法的新颖交叉算子 

---
# DocSpiral: A Platform for Integrated Assistive Document Annotation through Human-in-the-Spiral 

**Title (ZH)**: DocSpiral: 一种集成辅助文档标注的人机螺旋平台 

**Authors**: Qiang Sun, Sirui Li, Tingting Bi, Du Huynh, Mark Reynolds, Yuanyi Luo, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03214)  

**Abstract**: Acquiring structured data from domain-specific, image-based documents such as scanned reports is crucial for many downstream tasks but remains challenging due to document variability. Many of these documents exist as images rather than as machine-readable text, which requires human annotation to train automated extraction systems. We present DocSpiral, the first Human-in-the-Spiral assistive document annotation platform, designed to address the challenge of extracting structured information from domain-specific, image-based document collections. Our spiral design establishes an iterative cycle in which human annotations train models that progressively require less manual intervention. DocSpiral integrates document format normalization, comprehensive annotation interfaces, evaluation metrics dashboard, and API endpoints for the development of AI / ML models into a unified workflow. Experiments demonstrate that our framework reduces annotation time by at least 41\% while showing consistent performance gains across three iterations during model training. By making this annotation platform freely accessible, we aim to lower barriers to AI/ML models development in document processing, facilitating the adoption of large language models in image-based, document-intensive fields such as geoscience and healthcare. The system is freely available at: this https URL. The demonstration video is available: this https URL. 

**Abstract (ZH)**: 从特定领域基于图像的文档中获取结构化数据对于许多下游任务至关重要，但由于文档的差异性，这仍然具有挑战性。许多这些文档是以图像形式存在，而不是机器可读的文本，这需要人工注解以训练自动提取系统。我们提出了DocSpiral，这是第一个螺旋式人工介入辅助文档注解平台，旨在解决从特定领域基于图像的文档集合中提取结构化信息的挑战。我们的螺旋设计建立了一个迭代循环，在此过程中，人类注解训练模型，逐渐减少手动干预的需求。DocSpiral将文档格式规范化、全面的注解界面、评估指标仪表盘和API端点整合到一个统一的工作流程中，以支持AI/ML模型开发。实验结果显示，在模型训练过程中，我们的框架将注释时间减少了至少41%，并在三轮迭代中展现了持续的性能提升。通过使此注解平台免费开放，我们旨在降低AI/ML模型在文档处理中的开发门槛，促进大型语言模型在以图像为基础、文档密集型领域的应用，如地质科学和医疗保健。该系统可在以下链接获取：this https URL。演示视频可在以下链接获取：this https URL。 

---
# DCS-ST for Classification of Breast Cancer Histopathology Images with Limited Annotations 

**Title (ZH)**: DCS-ST在有限标注下的乳腺癌组织病理图像分类中应用 

**Authors**: Liu Suxing, Byungwon Min  

**Link**: [PDF](https://arxiv.org/pdf/2505.03204)  

**Abstract**: Deep learning methods have shown promise in classifying breast cancer histopathology images, but their performance often declines with limited annotated data, a critical challenge in medical imaging due to the high cost and expertise required for annotations. 

**Abstract (ZH)**: 深度学习方法在分类乳腺癌组织病理学图像方面显示出潜力，但在标注数据有限的情况下其性能往往会下降，这在医疗成像领域是一个关键挑战，因为标注数据需要较高的成本和专业知识。 

---
# A Trustworthy Multi-LLM Network: Challenges,Solutions, and A Use Case 

**Title (ZH)**: 可信赖的多大型语言模型网络：挑战、解决方案及一项应用案例 

**Authors**: Haoxiang Luo, Gang Sun, Yinqiu Liu, Dusit Niyato, Hongfang Yu, Mohammed Atiquzzaman, Schahram Dustdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03196)  

**Abstract**: Large Language Models (LLMs) demonstrate strong potential across a variety of tasks in communications and networking due to their advanced reasoning capabilities. However, because different LLMs have different model structures and are trained using distinct corpora and methods, they may offer varying optimization strategies for the same network issues. Moreover, the limitations of an individual LLM's training data, aggravated by the potential maliciousness of its hosting device, can result in responses with low confidence or even bias. To address these challenges, we propose a blockchain-enabled collaborative framework that connects multiple LLMs into a Trustworthy Multi-LLM Network (MultiLLMN). This architecture enables the cooperative evaluation and selection of the most reliable and high-quality responses to complex network optimization problems. Specifically, we begin by reviewing related work and highlighting the limitations of existing LLMs in collaboration and trust, emphasizing the need for trustworthiness in LLM-based systems. We then introduce the workflow and design of the proposed Trustworthy MultiLLMN framework. Given the severity of False Base Station (FBS) attacks in B5G and 6G communication systems and the difficulty of addressing such threats through traditional modeling techniques, we present FBS defense as a case study to empirically validate the effectiveness of our approach. Finally, we outline promising future research directions in this emerging area. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通信和网络领域各种任务中展现出强大的潜力，得益于其高级推理能力。然而，由于不同的LLMs具有不同的模型结构，并且采用不同的数据集和训练方法进行训练，它们可能在解决相同网络问题时提供不同的优化策略。此外，个体LLMs训练数据的限制，加上宿主设备潜在的恶意行为，可能导致低置信度或有偏见的响应。为了应对这些挑战，我们提出了一种基于区块链的合作框架，将多个LLMs连接成一个可信的多LLMs网络（MultiLLMN）。该架构允许多个LLMs协同评估和选择最可靠和高质量的复杂网络优化问题解决方案。具体来说，我们首先回顾相关工作，强调现有LLMs在合作和信任方面存在的局限性，突出LLM为基础的系统需要具备可信性的重要性和紧迫性。然后，我们介绍了所提出的可信多LLMs网络框架的工作流程和设计。鉴于5G和6G通信系统中虚假基站（FBS）攻击的严重性以及通过传统建模技术难以应对这类威胁的事实，我们以FBS防御为例，实证验证了我们方法的有效性。最后，我们概述了在这一新兴领域未来研究的有前景的方向。 

---
# A study on audio synchronous steganography detection and distributed guide inference model based on sliding spectral features and intelligent inference drive 

**Title (ZH)**: 基于滑动频谱特征和智能推理驱动的分布式引导推断模型的音频同步隐写分析研究 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03193)  

**Abstract**: With the rise of short video platforms in global communication, embedding steganographic data in audio synchronization streams has emerged as a new covert communication method. To address the limitations of traditional techniques in detecting synchronized steganography, this paper proposes a detection and distributed guidance reconstruction model based on short video "Yupan" samples released by China's South Sea Fleet on TikTok. The method integrates sliding spectrum feature extraction and intelligent inference mechanisms. A 25 ms sliding window with short-time Fourier transform (STFT) is used to extract the main frequency trajectory and construct the synchronization frame detection model (M1), identifying a frame flag "FFFFFFFFFFFFFFFFFF80". The subsequent 32-byte payload is decoded by a structured model (M2) to infer distributed guidance commands. Analysis reveals a low-entropy, repetitive byte sequence in the 36 to 45 second audio segment with highly concentrated spectral energy, confirming the presence of synchronization frames. Although plaintext semantics are not restored, the consistency in command field layout suggests features of military communication protocols. The multi-segment splicing model further shows cross-video embedding and centralized decoding capabilities. The proposed framework validates the effectiveness of sliding spectral features for synchronized steganography detection and builds an extensible inference model for covert communication analysis and tactical guidance simulation on open platforms. 

**Abstract (ZH)**: 基于中国南海舰队抖音“ Yupan ”样例的短视频音频同步隐写检测与分布式解码模型 

---
# seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models 

**Title (ZH)**: seq-JEPA: 自回归预测不变-相伴世界模型 

**Authors**: Hafez Ghaemi, Eilif Muller, Shahab Bakhtiari  

**Link**: [PDF](https://arxiv.org/pdf/2505.03176)  

**Abstract**: Current self-supervised algorithms mostly rely on transformations such as data augmentation and masking to learn visual representations. This is achieved by inducing invariance or equivariance with respect to these transformations after encoding two views of an image. This dominant two-view paradigm can limit the flexibility of learned representations for downstream adaptation by creating performance trade-offs between invariance-related tasks such as image classification and more fine-grained equivariance-related tasks. In this work, we introduce \emph{seq-JEPA}, a world modeling paradigm based on joint-embedding predictive architecture that leverages architectural inductive biases to resolve this trade-off. Without requiring an additional equivariance predictor or loss term, seq-JEPA simultaneously learns two architecturally segregated representations: one equivariant to the specified transformations and another invariant to them and suited for tasks such as classification. To do so, our model processes a short sequence of different views (observations) of an input image. Each encoded view is concatenated with embeddings corresponding to the relative transformation (action) producing the next observation in the sequence. A transformer encoder outputs an aggregate representation of this sequence, which is subsequently conditioned on the action leading to the next observation to predict its representation. Empirically, seq-JEPA achieves strong performance on equivariant benchmarks and image classification without sacrificing one for the other. Additionally, our framework excels at tasks that inherently require aggregating a sequence of observations, such as path integration across actions and predictive learning across eye movements. 

**Abstract (ZH)**: 当前的自监督算法主要依赖于数据增强和屏蔽等变换来学习视觉表示。这些方法通过编码图像的两个视图并在这些变换下诱导不变性或协变性来实现这一点。这种占主导地位的双视图范式可以通过在不变性相关的任务（如图像分类）和更细致的协变性相关的任务之间创建性能权衡来限制学习表示的灵活性。在此项工作中，我们引入了\emph{seq-JEPA}，一种基于联合嵌入预测架构的环境建模范式，利用架构诱导偏置解决这一权衡问题。seq-JEPA 不需要额外的协变性预测器或损失项，同时学习两个架构上隔离的表示：一个是对指定变换协变的表示，另一个是对它们不变且适合分类等任务。为此，我们的模型处理输入图像的不同视图（观察）的短序列。每个编码的视图都与产生下一个观察的相对变换（动作）对应的嵌入拼接起来。变压器编码器输出该序列的聚合表示，随后根据导致下一个观察的动作对其进行条件化以预测其表示。实验上，seq-JEPA 在协变基准测试和图像分类上表现出色，而无需牺牲一个以换取另一个。此外，我们的框架在需要聚合序列观察的任务上表现出色，例如在动作间进行路径整合和在眼动间进行预测学习。 

---
# RAVU: Retrieval Augmented Video Understanding with Compositional Reasoning over Graph 

**Title (ZH)**: RAVU：基于图的组合理论增强视频理解 

**Authors**: Sameer Malik, Moyuru Yamada, Ayush Singh, Dishank Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2505.03173)  

**Abstract**: Comprehending long videos remains a significant challenge for Large Multi-modal Models (LMMs). Current LMMs struggle to process even minutes to hours videos due to their lack of explicit memory and retrieval mechanisms. To address this limitation, we propose RAVU (Retrieval Augmented Video Understanding), a novel framework for video understanding enhanced by retrieval with compositional reasoning over a spatio-temporal graph. We construct a graph representation of the video, capturing both spatial and temporal relationships between entities. This graph serves as a long-term memory, allowing us to track objects and their actions across time. To answer complex queries, we decompose the queries into a sequence of reasoning steps and execute these steps on the graph, retrieving relevant key information. Our approach enables more accurate understanding of long videos, particularly for queries that require multi-hop reasoning and tracking objects across frames. Our approach demonstrate superior performances with limited retrieved frames (5-10) compared with other SOTA methods and baselines on two major video QA datasets, NExT-QA and EgoSchema. 

**Abstract (ZH)**: 理解长时间视频仍然是大型多模态模型（LMMs）的一个显著挑战。现有的LMMs由于缺乏显式的记忆和检索机制，难以处理甚至几分钟到几小时的视频。为了解决这一限制，我们提出了RAVu（检索增强视频理解）框架，该框架通过在时空图上进行组合推理来增强视频理解。我们构建了一个视频的图表示，捕捉实体之间的空间和时间关系。此图作为长期记忆，使我们能够在时间上跟踪物体及其动作。为了回答复杂的查询，我们将查询分解为一系列推理步骤，并在图上执行这些步骤，检索相关关键信息。我们的方法能够更准确地理解长时间视频，特别是在需要多跳推理和跨帧跟踪物体的查询方面。与NExT-QA和EgoSchema这两个主要视频问答数据集中的其他最先进方法和基线方法相比，我们的方法在使用有限检索帧（5-10帧）的情况下展示了更优越的性能。 

---
# Null Counterfactual Factor Interactions for Goal-Conditioned Reinforcement Learning 

**Title (ZH)**: 无偏反事实因素交互作用在目标条件 reinforcement 学习中 

**Authors**: Caleb Chuck, Fan Feng, Carl Qi, Chang Shi, Siddhant Agarwal, Amy Zhang, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2505.03172)  

**Abstract**: Hindsight relabeling is a powerful tool for overcoming sparsity in goal-conditioned reinforcement learning (GCRL), especially in certain domains such as navigation and locomotion. However, hindsight relabeling can struggle in object-centric domains. For example, suppose that the goal space consists of a robotic arm pushing a particular target block to a goal location. In this case, hindsight relabeling will give high rewards to any trajectory that does not interact with the block. However, these behaviors are only useful when the object is already at the goal -- an extremely rare case in practice. A dataset dominated by these kinds of trajectories can complicate learning and lead to failures. In object-centric domains, one key intuition is that meaningful trajectories are often characterized by object-object interactions such as pushing the block with the gripper. To leverage this intuition, we introduce Hindsight Relabeling using Interactions (HInt), which combines interactions with hindsight relabeling to improve the sample efficiency of downstream RL. However because interactions do not have a consensus statistical definition tractable for downstream GCRL, we propose a definition of interactions based on the concept of null counterfactual: a cause object is interacting with a target object if, in a world where the cause object did not exist, the target object would have different transition dynamics. We leverage this definition to infer interactions in Null Counterfactual Interaction Inference (NCII), which uses a "nulling'' operation with a learned model to infer interactions. NCII is able to achieve significantly improved interaction inference accuracy in both simple linear dynamics domains and dynamic robotic domains in Robosuite, Robot Air Hockey, and Franka Kitchen and HInt improves sample efficiency by up to 4x. 

**Abstract (ZH)**: hindsight relabeling for object-centric reinforcement learning using interactions (hint) 

---
# Soft Best-of-n Sampling for Model Alignment 

**Title (ZH)**: 软的最佳n抽样模型对齐 

**Authors**: Claudio Mayrink Verdun, Alex Oesterling, Himabindu Lakkaraju, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2505.03156)  

**Abstract**: Best-of-$n$ (BoN) sampling is a practical approach for aligning language model outputs with human preferences without expensive fine-tuning. BoN sampling is performed by generating $n$ responses to a prompt and then selecting the sample that maximizes a reward function. BoN yields high reward values in practice at a distortion cost, as measured by the KL-divergence between the sampled and original distribution. This distortion is coarsely controlled by varying the number of samples: larger $n$ yields a higher reward at a higher distortion cost. We introduce Soft Best-of-$n$ sampling, a generalization of BoN that allows for smooth interpolation between the original distribution and reward-maximizing distribution through a temperature parameter $\lambda$. We establish theoretical guarantees showing that Soft Best-of-$n$ sampling converges sharply to the optimal tilted distribution at a rate of $O(1/n)$ in KL and the expected (relative) reward. For sequences of discrete outputs, we analyze an additive reward model that reveals the fundamental limitations of blockwise sampling. 

**Abstract (ZH)**: Best-of-$n$ (BoN) 抽样是一种实用的方法，用于通过生成$n$个响应并选择最大化奖励函数的样本来使语言模型输出与人类偏好保持一致，而无需昂贵的精细调整。BoN抽样通过生成$n$个响应并选择最大化奖励函数的样本来实现，所得到的奖励值在实际中会伴随着KL散度度量的失真成本。通过改变样本数量来粗略控制这种失真：较大的$n$会产生更高的奖励但伴随更高的失真成本。我们引入了Soft Best-of-$n$抽样，这是一种BoN的推广，通过温度参数$\lambda$可以在原始分布和奖励最大化分布之间实现平滑插值。我们建立了理论保证，证明Soft Best-of-$n$抽样以$O(1/n)$的速率在KL和期望（相对）奖励方面收敛到最优倾斜分布。对于离散输出的序列，我们分析了一个加性奖励模型，揭示了块状抽样的基本局限性。 

---
# StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data 

**Title (ZH)**: 稳定运动：使用未配对的损坏数据训练运动清理模型 

**Authors**: Yuxuan Mu, Hung Yu Ling, Yi Shi, Ismael Baira Ojeda, Pengcheng Xi, Chang Shu, Fabio Zinno, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03154)  

**Abstract**: Motion capture (mocap) data often exhibits visually jarring artifacts due to inaccurate sensors and post-processing. Cleaning this corrupted data can require substantial manual effort from human experts, which can be a costly and time-consuming process. Previous data-driven motion cleanup methods offer the promise of automating this cleanup process, but often require in-domain paired corrupted-to-clean training data. Constructing such paired datasets requires access to high-quality, relatively artifact-free motion clips, which often necessitates laborious manual cleanup. In this work, we present StableMotion, a simple yet effective method for training motion cleanup models directly from unpaired corrupted datasets that need cleanup. The core component of our method is the introduction of motion quality indicators, which can be easily annotated through manual labeling or heuristic algorithms and enable training of quality-aware motion generation models on raw motion data with mixed quality. At test time, the model can be prompted to generate high-quality motions using the quality indicators. Our method can be implemented through a simple diffusion-based framework, leading to a unified motion generate-discriminate model, which can be used to both identify and fix corrupted frames. We demonstrate that our proposed method is effective for training motion cleanup models on raw mocap data in production scenarios by applying StableMotion to SoccerMocap, a 245-hour soccer mocap dataset containing real-world motion artifacts. The trained model effectively corrects a wide range of motion artifacts, reducing motion pops and frozen frames by 68% and 81%, respectively. See this https URL for more results. 

**Abstract (ZH)**: 一种从未配对的受损数据集训练运动清理模型的方法：StableMotion 

---
# Motion-compensated cardiac MRI using low-rank diffeomorphic flow (DMoCo) 

**Title (ZH)**: 基于低秩 diffeomorphic 流的运动补偿心脏MRI（DMoCo） 

**Authors**: Joseph William Kettelkamp, Ludovica Romanin, Sarv Priya, Mathews Jacob  

**Link**: [PDF](https://arxiv.org/pdf/2505.03149)  

**Abstract**: We introduce an unsupervised motion-compensated image reconstruction algorithm for free-breathing and ungated 3D cardiac magnetic resonance imaging (MRI). We express the image volume corresponding to each specific motion phase as the deformation of a single static image template. The main contribution of the work is the low-rank model for the compact joint representation of the family of diffeomorphisms, parameterized by the motion phases. The diffeomorphism at a specific motion phase is obtained by integrating a parametric velocity field along a path connecting the reference template phase to the motion phase. The velocity field at different phases is represented using a low-rank model. The static template and the low-rank motion model parameters are learned directly from the k-space data in an unsupervised fashion. The more constrained motion model is observed to offer improved recovery compared to current motion-resolved and motion-compensated algorithms for free-breathing 3D cine MRI. 

**Abstract (ZH)**: 无监督运动补偿图像重建算法用于自由呼吸和无门控3D心脏磁共振成像（MRI） 

---
# VISLIX: An XAI Framework for Validating Vision Models with Slice Discovery and Analysis 

**Title (ZH)**: VISLIX：一种基于切片发现与分析的可解释性验证框架用于视觉模型 

**Authors**: Xinyuan Yan, Xiwei Xuan, Jorge Piazentin Ono, Jiajing Guo, Vikram Mohanty, Shekar Arvind Kumar, Liang Gou, Bei Wang, Liu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.03132)  

**Abstract**: Real-world machine learning models require rigorous evaluation before deployment, especially in safety-critical domains like autonomous driving and surveillance. The evaluation of machine learning models often focuses on data slices, which are subsets of the data that share a set of characteristics. Data slice finding automatically identifies conditions or data subgroups where models underperform, aiding developers in mitigating performance issues. Despite its popularity and effectiveness, data slicing for vision model validation faces several challenges. First, data slicing often needs additional image metadata or visual concepts, and falls short in certain computer vision tasks, such as object detection. Second, understanding data slices is a labor-intensive and mentally demanding process that heavily relies on the expert's domain knowledge. Third, data slicing lacks a human-in-the-loop solution that allows experts to form hypothesis and test them interactively. To overcome these limitations and better support the machine learning operations lifecycle, we introduce VISLIX, a novel visual analytics framework that employs state-of-the-art foundation models to help domain experts analyze slices in computer vision models. Our approach does not require image metadata or visual concepts, automatically generates natural language insights, and allows users to test data slice hypothesis interactively. We evaluate VISLIX with an expert study and three use cases, that demonstrate the effectiveness of our tool in providing comprehensive insights for validating object detection models. 

**Abstract (ZH)**: 实世界中的机器学习模型在部署前需要严格的评估，尤其是在自动驾驶和监控等安全关键领域。机器学习模型的评估通常集中在数据切片上，这是具有共同特征的数据子集。数据切片能够自动识别模型性能不佳的条件或数据子组，帮助开发者缓解性能问题。尽管数据切片在视觉模型验证中流行且有效，但仍面临几个挑战。首先，数据切片往往需要额外的图像元数据或视觉概念，在某些计算机视觉任务中应用受限，如目标检测。其次，理解数据切片是一个耗时且复杂的任务，高度依赖于专家领域的知识。第三，数据切片缺乏一个将专家纳入循环的解决方案，不支持专家形成和测试假设的交互过程。为了克服这些限制，更好地支持机器学习运维生命周期，我们提出了VISLIX，一种新颖的视觉分析框架，利用最先进的基础模型帮助领域专家分析计算机视觉模型中的数据切片。我们的方法不需要图像元数据或视觉概念，可以自动生成自然语言见解，并允许用户交互式地测试数据切片假设。我们通过专家研究和三个案例研究评估了VISLIX，展示了该工具在验证目标检测模型方面提供全面见解的有效性。 

---
# Cognitio Emergens: Agency, Dimensions, and Dynamics in Human-AI Knowledge Co-Creation 

**Title (ZH)**: 认知涌现：人类-人工智能知识共创的代理、维度与动力机制 

**Authors**: Xule Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03105)  

**Abstract**: Scientific knowledge creation is fundamentally transforming as humans and AI systems evolve beyond tool-user relationships into co-evolutionary epistemic partnerships. When AlphaFold revolutionized protein structure prediction, researchers described engaging with an epistemic partner that reshaped how they conceptualized fundamental relationships. This article introduces Cognitio Emergens (CE), a framework addressing critical limitations in existing models that focus on static roles or narrow metrics while failing to capture how scientific understanding emerges through recursive human-AI interaction over time. CE integrates three components addressing these limitations: Agency Configurations describing how authority distributes between humans and AI (Directed, Contributory, Partnership), with partnerships dynamically oscillating between configurations rather than following linear progression; Epistemic Dimensions capturing six specific capabilities emerging through collaboration across Discovery, Integration, and Projection axes, creating distinctive "capability signatures" that guide development; and Partnership Dynamics identifying forces shaping how these relationships evolve, particularly the risk of epistemic alienation where researchers lose interpretive control over knowledge they formally endorse. Drawing from autopoiesis theory, social systems theory, and organizational modularity, CE reveals how knowledge co-creation emerges through continuous negotiation of roles, values, and organizational structures. By reconceptualizing human-AI scientific collaboration as fundamentally co-evolutionary, CE offers a balanced perspective that neither uncritically celebrates nor unnecessarily fears AI's evolving role, instead providing conceptual tools for cultivating partnerships that maintain meaningful human participation while enabling transformative scientific breakthroughs. 

**Abstract (ZH)**: 科学知识的创造正在从根本上被重塑，随着人类和AI系统的进化超越工具使用者的关系，进入到共生认知伙伴关系。当AlphaFold革新蛋白质结构预测时，研究人员描述了与一个重塑他们基本认知关系的共生认知伙伴互动的经历。本文介绍了认知涌现（Cognitio Emergens，CE）框架，该框架解决现有模型中聚焦于静态角色或狭窄指标的局限性，而未能捕捉到随着时间的推移，通过反复的人类-AI互动中科学理解是如何涌现的。CE整合了三个组成部分来解决这些局限性：主体配置（Agency Configurations），描述权力在人类和AI之间的分配（定向、贡献、伙伴关系），伙伴关系在这些配置之间动态摆动而非线性进展；认知维度（Epistemic Dimensions），捕捉协作过程中在发现、整合和预测轴上六种特定能力的涌现，创造出独特的“能力特征”，指导发展；以及伙伴关系动力学（Partnership Dynamics），识别塑造这些关系演变的力量，尤其是知识异化的风险，即研究人员失去了他们正式认可的知识的解释控制。借助自动调节理论、社会系统理论和组织模块化理论，CE揭示了知识共同创造如何通过持续的角色、价值观和组织结构的协商而涌现。通过将人类-AI科学合作重新构建为根本上的共生进化，CE提供了一种平衡的观点，既不过分庆祝也不无必要地恐惧AI角色的演变，而是提供了实现有意义的人类参与与颠覆性科学突破之间的概念工具。 

---
# Assessing and Enhancing the Robustness of LLM-based Multi-Agent Systems Through Chaos Engineering 

**Title (ZH)**: 通过混沌工程评估并增强基于LLM的多代理系统 robustness 

**Authors**: Joshua Owotogbe  

**Link**: [PDF](https://arxiv.org/pdf/2505.03096)  

**Abstract**: This study explores the application of chaos engineering to enhance the robustness of Large Language Model-Based Multi-Agent Systems (LLM-MAS) in production-like environments under real-world conditions. LLM-MAS can potentially improve a wide range of tasks, from answering questions and generating content to automating customer support and improving decision-making processes. However, LLM-MAS in production or preproduction environments can be vulnerable to emergent errors or disruptions, such as hallucinations, agent failures, and agent communication failures. This study proposes a chaos engineering framework to proactively identify such vulnerabilities in LLM-MAS, assess and build resilience against them, and ensure reliable performance in critical applications. 

**Abstract (ZH)**: 本研究探讨了将混沌工程应用于提高类生产环境中大规模语言模型基于的多智能体系统（LLM-MAS）鲁棒性的应用，以应对实际条件下的潜在错误或中断。LLM-MAS可以从回答问题、生成内容到自动化的客户服务和决策过程改进等多种任务中受益。然而，在生产或准生产环境中，LLM-MAS可能易受幻觉、智能体故障及智能体间通信故障等新兴错误或中断的影响。本研究提出了一种混沌工程框架，旨在前瞻性地识别LLM-MAS中的这些脆弱性，评估并构建对其的抗御能力，以确保其在关键应用中可靠的表现。 

---
# Latent Adaptive Planner for Dynamic Manipulation 

**Title (ZH)**: 动态操作的潜适应规划器 

**Authors**: Donghun Noh, Deqian Kong, Minglu Zhao, Andrew Lizarraga, Jianwen Xie, Ying Nian Wu, Dennis Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.03077)  

**Abstract**: This paper presents Latent Adaptive Planner (LAP), a novel approach for dynamic nonprehensile manipulation tasks that formulates planning as latent space inference, effectively learned from human demonstration videos. Our method addresses key challenges in visuomotor policy learning through a principled variational replanning framework that maintains temporal consistency while efficiently adapting to environmental changes. LAP employs Bayesian updating in latent space to incrementally refine plans as new observations become available, striking an optimal balance between computational efficiency and real-time adaptability. We bridge the embodiment gap between humans and robots through model-based proportional mapping that regenerates accurate kinematic-dynamic joint states and object positions from human demonstrations. Experimental evaluations across multiple complex manipulation benchmarks demonstrate that LAP achieves state-of-the-art performance, outperforming existing approaches in success rate, trajectory smoothness, and energy efficiency, particularly in dynamic adaptation scenarios. Our approach enables robots to perform complex interactions with human-like adaptability while providing an expandable framework applicable to diverse robotic platforms using the same human demonstration videos. 

**Abstract (ZH)**: 基于潜在空间推断的latent adaptive planner（LAP）：一种新颖的动态非拾取操作规划方法 

---
# Developing A Framework to Support Human Evaluation of Bias in Generated Free Response Text 

**Title (ZH)**: 开发一个框架以支持对生成自由响应文本中的偏见进行人工评估 

**Authors**: Jennifer Healey, Laurie Byrum, Md Nadeem Akhtar, Surabhi Bhargava, Moumita Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2505.03053)  

**Abstract**: LLM evaluation is challenging even the case of base models. In real world deployments, evaluation is further complicated by the interplay of task specific prompts and experiential context. At scale, bias evaluation is often based on short context, fixed choice benchmarks that can be rapidly evaluated, however, these can lose validity when the LLMs' deployed context differs. Large scale human evaluation is often seen as too intractable and costly. Here we present our journey towards developing a semi-automated bias evaluation framework for free text responses that has human insights at its core. We discuss how we developed an operational definition of bias that helped us automate our pipeline and a methodology for classifying bias beyond multiple choice. We additionally comment on how human evaluation helped us uncover problematic templates in a bias benchmark. 

**Abstract (ZH)**: LLM偏见评估即使对于基础模型也具有挑战性。在实际部署中，任务特定提示和经验性上下文的交互进一步复杂化了评估过程。在大规模部署中，偏见评估通常基于短上下文和固定选择的基准，这些基准可以快速评估，然而，当部署的上下文与LLM不符时，这些基准的有效性会降低。大规模人工评估通常被视为难以实现且成本高昂。我们提出了一个以人类洞察为核心、具有半自动化偏见评估框架的发展历程。我们讨论了我们如何制定可操作的偏见定义，以帮助自动化评估管道，以及如何超越选择题分类偏见的方法论。此外，我们还提到人类评估如何帮助我们发现偏见基准中的问题模板。 

---
# MORE: Mobile Manipulation Rearrangement Through Grounded Language Reasoning 

**Title (ZH)**: 基于接地语言推理的移动 manipulator 重组 

**Authors**: Mohammad Mohammadi, Daniel Honerkamp, Martin Büchner, Matteo Cassinelli, Tim Welschehold, Fabien Despinoy, Igor Gilitschenski, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2505.03035)  

**Abstract**: Autonomous long-horizon mobile manipulation encompasses a multitude of challenges, including scene dynamics, unexplored areas, and error recovery. Recent works have leveraged foundation models for scene-level robotic reasoning and planning. However, the performance of these methods degrades when dealing with a large number of objects and large-scale environments. To address these limitations, we propose MORE, a novel approach for enhancing the capabilities of language models to solve zero-shot mobile manipulation planning for rearrangement tasks. MORE leverages scene graphs to represent environments, incorporates instance differentiation, and introduces an active filtering scheme that extracts task-relevant subgraphs of object and region instances. These steps yield a bounded planning problem, effectively mitigating hallucinations and improving reliability. Additionally, we introduce several enhancements that enable planning across both indoor and outdoor environments. We evaluate MORE on 81 diverse rearrangement tasks from the BEHAVIOR-1K benchmark, where it becomes the first approach to successfully solve a significant share of the benchmark, outperforming recent foundation model-based approaches. Furthermore, we demonstrate the capabilities of our approach in several complex real-world tasks, mimicking everyday activities. We make the code publicly available at this https URL. 

**Abstract (ZH)**: 自主长时移动操作涵盖了多种挑战，包括场景动态、未探索区域和错误恢复。近期研究利用基础模型进行场景级别的机器人推理和规划。然而，当处理大量对象和大规模环境时，这些方法的性能会下降。为克服这些限制，我们提出MORE，一种增强语言模型解决零样本移动操作规划以进行重排任务能力的新方法。MORE利用场景图表示环境，引入实例差异化，并引入一种积极筛选方案，提取与任务相关的对象和区域实例子图。这些步骤产生了有界规划问题，有效减弱了幻觉并提高了可靠性。此外，我们引入了几种增强措施，使规划能够在室内外环境中进行。我们在BEHAVIOR-1K基准上的81个多样重排任务中评估了MORE，使其成为首个成功解决基准测试显著比例方法，并优于基于基础模型的近期方法。此外，我们展示了该方法在多个复杂真实世界的任务中的能力，模拟日常活动。代码已公开于此链接。 

---
# A Typology of Synthetic Datasets for Dialogue Processing in Clinical Contexts 

**Title (ZH)**: 合成数据集类型在临床对话处理中的应用 

**Authors**: Steven Bedrick, A. Seza Doğruöz, Sergiu Nisioi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03025)  

**Abstract**: Synthetic data sets are used across linguistic domains and NLP tasks, particularly in scenarios where authentic data is limited (or even non-existent). One such domain is that of clinical (healthcare) contexts, where there exist significant and long-standing challenges (e.g., privacy, anonymization, and data governance) which have led to the development of an increasing number of synthetic datasets. One increasingly important category of clinical dataset is that of clinical dialogues which are especially sensitive and difficult to collect, and as such are commonly synthesized.
While such synthetic datasets have been shown to be sufficient in some situations, little theory exists to inform how they may be best used and generalized to new applications. In this paper, we provide an overview of how synthetic datasets are created, evaluated and being used for dialogue related tasks in the medical domain. Additionally, we propose a novel typology for use in classifying types and degrees of data synthesis, to facilitate comparison and evaluation. 

**Abstract (ZH)**: 合成数据集在语言学领域和NLP任务中被广泛应用，特别是在真实数据有限（甚至不存在）的情况下。在临床（医疗保健）背景下尤其如此，该背景下存在显著且长期存在的挑战（例如隐私、匿名化和数据治理），导致了越来越多合成数据集的发展。其中一类日益重要的临床数据集是临床对话数据，这些数据尤其敏感且难以收集，因此通常需要合成。

尽管合成数据集在某些情况下已显示出足够性，但目前尚缺乏理论来指导如何最好地使用它们并将它们应用于新的应用场景。本文提供了一个如何创建、评估和用于医疗领域对话相关任务的合成数据集的综述。此外，我们还提出了一种新的分类类型学，用于分类不同类型和程度的数据合成，以促进比较和评估。 

---
# Memorization or Interpolation ? Detecting LLM Memorization through Input Perturbation Analysis 

**Title (ZH)**: 记忆还是内插？通过输入扰动分析检测大模型的记忆现象 

**Authors**: Albérick Euraste Djiré, Abdoul Kader Kaboré, Earl T. Barr, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2505.03019)  

**Abstract**: While Large Language Models (LLMs) achieve remarkable performance through training on massive datasets, they can exhibit concerning behaviors such as verbatim reproduction of training data rather than true generalization. This memorization phenomenon raises significant concerns about data privacy, intellectual property rights, and the reliability of model evaluations. This paper introduces PEARL, a novel approach for detecting memorization in LLMs. PEARL assesses how sensitive an LLM's performance is to input perturbations, enabling memorization detection without requiring access to the model's internals. We investigate how input perturbations affect the consistency of outputs, enabling us to distinguish between true generalization and memorization. Our findings, following extensive experiments on the Pythia open model, provide a robust framework for identifying when the model simply regurgitates learned information. Applied on the GPT 4o models, the PEARL framework not only identified cases of memorization of classic texts from the Bible or common code from HumanEval but also demonstrated that it can provide supporting evidence that some data, such as from the New York Times news articles, were likely part of the training data of a given model. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）通过大规模数据训练实现了出色的性能，但它们可能会表现出诸如机械重复训练数据而不是真正泛化的令人关注的行为。这种记忆现象引发了关于数据隐私、知识产权以及模型评估可靠性的重大关注。本文介绍了PEARL，一种用于检测LLMs记忆现象的新方法。PEARL评估LLM对输入扰动的性能敏感性，能够在无需访问模型内部结构的情况下检测记忆现象。我们研究了输入扰动如何影响输出的一致性，使我们能够区分真正的泛化和记忆现象。我们的研究结果，通过对Pythia开源模型的大量实验，提供了一种 robust 的框架，用于识别模型仅仅是重复已学习信息的情况。将PEARL框架应用于GPT 4o模型后，不仅识别出了对《圣经》经典文本或HumanEval中的通用代码的记忆现象，还展示了该框架可以提供支持证据，证明某些数据，如《纽约时报》的文章内容，很可能包含在某个模型的训练数据中。 

---
# Lesion-Aware Generative Artificial Intelligence for Virtual Contrast-Enhanced Mammography in Breast Cancer 

**Title (ZH)**: 基于病灶aware的生成人工智能在乳腺癌中虚拟对比增强乳房X线摄影的应用 

**Authors**: Aurora Rofena, Arianna Manchia, Claudia Lucia Piccolo, Bruno Beomonte Zobel, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03018)  

**Abstract**: Contrast-Enhanced Spectral Mammography (CESM) is a dual-energy mammographic technique that improves lesion visibility through the administration of an iodinated contrast agent. It acquires both a low-energy image, comparable to standard mammography, and a high-energy image, which are then combined to produce a dual-energy subtracted image highlighting lesion contrast enhancement. While CESM offers superior diagnostic accuracy compared to standard mammography, its use entails higher radiation exposure and potential side effects associated with the contrast medium. To address these limitations, we propose Seg-CycleGAN, a generative deep learning framework for Virtual Contrast Enhancement in CESM. The model synthesizes high-fidelity dual-energy subtracted images from low-energy images, leveraging lesion segmentation maps to guide the generative process and improve lesion reconstruction. Building upon the standard CycleGAN architecture, Seg-CycleGAN introduces localized loss terms focused on lesion areas, enhancing the synthesis of diagnostically relevant regions. Experiments on the CESM@UCBM dataset demonstrate that Seg-CycleGAN outperforms the baseline in terms of PSNR and SSIM, while maintaining competitive MSE and VIF. Qualitative evaluations further confirm improved lesion fidelity in the generated images. These results suggest that segmentation-aware generative models offer a viable pathway toward contrast-free CESM alternatives. 

**Abstract (ZH)**: 基于分割的CycleGAN在CEMG中的虚拟对比增强 

---
# RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale 

**Title (ZH)**: RADLADS: 快速注意力萃取以实现大规模线性注意力解码器 

**Authors**: Daniel Goldstein, Eric Alcaide, Janna Lu, Eugene Cheah  

**Link**: [PDF](https://arxiv.org/pdf/2505.03005)  

**Abstract**: We present Rapid Attention Distillation to Linear Attention Decoders at Scale (RADLADS), a protocol for rapidly converting softmax attention transformers into linear attention decoder models, along with two new RWKV-variant architectures, and models converted from popular Qwen2.5 open source models in 7B, 32B, and 72B sizes. Our conversion process requires only 350-700M tokens, less than 0.005% of the token count used to train the original teacher models. Converting to our 72B linear attention model costs less than \$2,000 USD at today's prices, yet quality at inference remains close to the original transformer. These models achieve state-of-the-art downstream performance across a set of standard benchmarks for linear attention models of their size. We release all our models on HuggingFace under the Apache 2.0 license, with the exception of our 72B models which are also governed by the Qwen License Agreement.
Models at this https URL Training Code at this https URL 

**Abstract (ZH)**: 快速注意力蒸馏到大规模线性注意力解码器（RADLADS）：一种将softmax注意力变换器快速转换为线性注意力解码器模型的协议，及其两种新的RWKV-变体架构和从流行开源Qwen2.5模型转换而来的7B、32B和72B规模的模型。 

---
# Generating Narrated Lecture Videos from Slides with Synchronized Highlights 

**Title (ZH)**: 从幻灯片生成同步高亮的讲述视频 

**Authors**: Alexander Holmberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.02966)  

**Abstract**: Turning static slides into engaging video lectures takes considerable time and effort, requiring presenters to record explanations and visually guide their audience through the material. We introduce an end-to-end system designed to automate this process entirely. Given a slide deck, this system synthesizes a video lecture featuring AI-generated narration synchronized precisely with dynamic visual highlights. These highlights automatically draw attention to the specific concept being discussed, much like an effective presenter would. The core technical contribution is a novel highlight alignment module. This module accurately maps spoken phrases to locations on a given slide using diverse strategies (e.g., Levenshtein distance, LLM-based semantic analysis) at selectable granularities (line or word level) and utilizes timestamp-providing Text-to-Speech (TTS) for timing synchronization. We demonstrate the system's effectiveness through a technical evaluation using a manually annotated slide dataset with 1000 samples, finding that LLM-based alignment achieves high location accuracy (F1 > 92%), significantly outperforming simpler methods, especially on complex, math-heavy content. Furthermore, the calculated generation cost averages under $1 per hour of video, offering potential savings of two orders of magnitude compared to conservative estimates of manual production costs. This combination of high accuracy and extremely low cost positions this approach as a practical and scalable tool for transforming static slides into effective, visually-guided video lectures. 

**Abstract (ZH)**: 将静态幻灯片转化为引人入胜的视频讲座需要大量时间和努力，要求讲者录制解释并引导观众观看内容。我们介绍了一个端到端系统，旨在完全自动化这一过程。给定一个幻灯片集，该系统生成一个包含AI生成解说的视频讲座，解说与动态视觉高光精准同步。这些高光自动将注意力集中在正在讨论的具体概念上，类似于有效的讲者。核心技术贡献是新型高光对齐模块。该模块使用多种策略（如Levenshtein距离、基于LLM的语义分析）在可选择的粒度（行或单词级别）上准确地将语音短语映射到给定幻灯片上的位置，并利用提供时间戳的文本到语音（TTS）进行时间同步。通过使用包含1000个样例的手动标注幻灯片数据集进行技术评估，我们发现基于LLM的对齐在位置准确性方面取得高得分（F1 > 92%），显著优于简单方法，特别是在复杂、数学密集的内容方面。此外，计算生成成本平均每小时不到1美元，与保守估计的手动生产成本相比，潜在节省达两个数量级。这种高精度和极低成本的结合使该方法成为将静态幻灯片转化为有效、视觉引导的视频讲座的实用且可扩展工具。 

---
# The Cognitive Foundations of Economic Exchange: A Modular Framework Grounded in Behavioral Evidence 

**Title (ZH)**: 经济交换的认知基础：基于行为证据的模块化框架 

**Authors**: Egil Diau  

**Link**: [PDF](https://arxiv.org/pdf/2505.02945)  

**Abstract**: A key challenge in multi-agent AI is modeling social cooperation under realistic behavioral constraints. Many foundational concepts in economics and ethics such as "trust" or "morality" are often defined informally, without operational criteria or cognitive grounding, which limits their testability and implementation in artificial agents. Drawing on converging empirical evidence from primate behavior, infant cognition, and economic anthropology, we propose a conceptual framework composed of three cognitively minimal mechanisms: individual recognition, reciprocal credence, and cost return sensitivity. This framework reframes trust as a graded cognitive expectation, providing a simulateable basis for reciprocal exchange in artificial agents, and enabling the bottom-up emergence of scalable cooperation and institutional dynamics. 

**Abstract (ZH)**: 多-agent AI 中的一个关键挑战是，在现实行为约束下建模社会合作。经济学和伦理学中的许多基础概念，如“信任”或“道德”，通常被非正式地定义，缺乏可操作标准或认知基础，这限制了它们在人工代理中的测试和实施。借鉴来自灵长类行为、婴儿认知和经济人类学的趋同实证证据，我们提出了一种由三种认知最小机制组成的概念框架：个体识别、互惠信任和成本回报敏感性。该框架将信任重新定义为分级的认知期望，为人工代理中的互惠交换提供可模拟的基础，并促进大规模合作和制度动态的自下而上涌现。 

---
# The Art of Repair: Optimizing Iterative Program Repair with Instruction-Tuned Models 

**Title (ZH)**: 修复的艺术：基于指令调优模型的迭代程序修复优化 

**Authors**: Fernando Vallecillos Ruiz, Max Hort, Leon Moonen  

**Link**: [PDF](https://arxiv.org/pdf/2505.02931)  

**Abstract**: Automatic program repair (APR) aims to reduce the manual efforts required to identify and fix errors in source code. Before the rise of LLM-based agents, a common strategy was to increase the number of generated patches, sometimes to the thousands, to achieve better repair results on benchmarks. More recently, self-iterative capabilities enabled LLMs to refine patches over multiple rounds guided by feedback. However, literature often focuses on many iterations and disregards different numbers of outputs.
We investigate an APR pipeline that balances these two approaches, the generation of multiple outputs and multiple rounds of iteration, while imposing a limit of 10 total patches per bug. We apply three SOTA instruction-tuned LLMs - DeepSeekCoder-Instruct, Codellama-Instruct, Llama3.1-Instruct - to the APR task. We further fine-tune each model on an APR dataset with three sizes (1K, 30K, 65K) and two techniques (Full Fine-Tuning and LoRA), allowing us to assess their repair capabilities on two APR benchmarks: HumanEval-Java and Defects4J.
Our results show that by using only a fraction (<1%) of the fine-tuning dataset, we can achieve improvements of up to 78% in the number of plausible patches generated, challenging prior studies that reported limited gains using Full Fine-Tuning. However, we find that exceeding certain thresholds leads to diminishing outcomes, likely due to overfitting. Moreover, we show that base models greatly benefit from creating patches in an iterative fashion rather than generating them all at once. In addition, the benefit of iterative strategies becomes more pronounced in complex benchmarks. Even fine-tuned models, while benefiting less from iterations, still gain advantages, particularly on complex benchmarks. The research underscores the need for balanced APR strategies that combine multi-output generation and iterative refinement. 

**Abstract (ZH)**: 一种平衡多输出与多轮迭代的自动程序修复管道：限制每个错误最多生成10个补丁 

---
# Early Prediction of Sepsis: Feature-Aligned Transfer Learning 

**Title (ZH)**: 早期预测败血症：特征对齐迁移学习 

**Authors**: Oyindolapo O. Komolafe, Zhimin Mei, David Morales Zarate, Gregory William Spangenberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.02889)  

**Abstract**: Sepsis is a life threatening medical condition that occurs when the body has an extreme response to infection, leading to widespread inflammation, organ failure, and potentially death. Because sepsis can worsen rapidly, early detection is critical to saving lives. However, current diagnostic methods often identify sepsis only after significant damage has already occurred. Our project aims to address this challenge by developing a machine learning based system to predict sepsis in its early stages, giving healthcare providers more time to intervene.
A major problem with existing models is the wide variability in the patient information or features they use, such as heart rate, temperature, and lab results. This inconsistency makes models difficult to compare and limits their ability to work across different hospitals and settings. To solve this, we propose a method called Feature Aligned Transfer Learning (FATL), which identifies and focuses on the most important and commonly reported features across multiple studies, ensuring the model remains consistent and clinically relevant.
Most existing models are trained on narrow patient groups, leading to population bias. FATL addresses this by combining knowledge from models trained on diverse populations, using a weighted approach that reflects each models contribution. This makes the system more generalizable and effective across different patient demographics and clinical environments. FATL offers a practical and scalable solution for early sepsis detection, particularly in hospitals with limited resources, and has the potential to improve patient outcomes, reduce healthcare costs, and support more equitable healthcare delivery. 

**Abstract (ZH)**: 脓毒症是一种生命威胁性的医疗状况，发生在身体对感染产生极端反应导致全身炎症、器官失败甚至死亡的情况下。由于脓毒症可能迅速恶化，早期检测至关重要。然而，现有的诊断方法通常只有在已经造成严重损害后才识别出脓毒症。我们的项目旨在通过开发基于机器学习的系统来预测脓毒症的早期阶段，从而给医疗提供者更多的时间进行干预，以减少死亡率。

现有模型的一个主要问题是使用的病人信息或特征的广泛差异，例如心率、体温和实验室结果。这种差异性使得模型难以比较，并限制了它们在不同医院和环境中的适用性。为了解决这一问题，我们提出了一种称为特征对齐迁移学习（FATL）的方法，该方法识别并关注多个研究中最重要的和最常报告的特征，确保模型的一致性和临床相关性。

大多数现有的模型仅针对狭窄的患者群体进行训练，导致群体偏差。FATL 通过结合来自不同人群训练模型的知识，采用加权方法反映每个模型的贡献，从而使系统在不同的患者群体和临床环境中更具普适性和有效性。FATL 提供了一种实用且可扩展的早期脓毒症检测解决方案，特别是在资源有限的医院，并具有提高患者预后、降低医疗成本和促进更公平的医疗服务的潜力。 

---
# When Your Own Output Becomes Your Training Data: Noise-to-Meaning Loops and a Formal RSI Trigger 

**Title (ZH)**: 当你的输出成为你的训练数据：噪声到意义的循环及正式的RSI触发器 

**Authors**: Rintaro Ando  

**Link**: [PDF](https://arxiv.org/pdf/2505.02888)  

**Abstract**: We present Noise-to-Meaning Recursive Self-Improvement (N2M-RSI), a minimal formal model showing that once an AI agent feeds its own outputs back as inputs and crosses an explicit information-integration threshold, its internal complexity will grow without bound under our assumptions. The framework unifies earlier ideas on self-prompting large language models, Gödelian self-reference, and AutoML, yet remains implementation-agnostic. The model furthermore scales naturally to interacting swarms of agents, hinting at super-linear effects once communication among instances is permitted. For safety reasons, we omit system-specific implementation details and release only a brief, model-agnostic toy prototype in Appendix C. 

**Abstract (ZH)**: 噪声到含义递归自我完善（N2M-RSI）：一种最小形式模型 

---
# CreoPep: A Universal Deep Learning Framework for Target-Specific Peptide Design and Optimization 

**Title (ZH)**: CreoPep: 一种针对特定靶点的深度学习通用框架及其肽设计与优化 

**Authors**: Cheng Ge, Han-Shen Tae, Zhenqiang Zhang, Lu Lu, Zhijie Huang, Yilin Wang, Tao Jiang, Wenqing Cai, Shan Chang, David J. Adams, Rilei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02887)  

**Abstract**: Target-specific peptides, such as conotoxins, exhibit exceptional binding affinity and selectivity toward ion channels and receptors. However, their therapeutic potential remains underutilized due to the limited diversity of natural variants and the labor-intensive nature of traditional optimization strategies. Here, we present CreoPep, a deep learning-based conditional generative framework that integrates masked language modeling with a progressive masking scheme to design high-affinity peptide mutants while uncovering novel structural motifs. CreoPep employs an integrative augmentation pipeline, combining FoldX-based energy screening with temperature-controlled multinomial sampling, to generate structurally and functionally diverse peptides that retain key pharmacological properties. We validate this approach by designing conotoxin inhibitors targeting the $\alpha$7 nicotinic acetylcholine receptor, achieving submicromolar potency in electrophysiological assays. Structural analysis reveals that CreoPep-generated variants engage in both conserved and novel binding modes, including disulfide-deficient forms, thus expanding beyond conventional design paradigms. Overall, CreoPep offers a robust and generalizable platform that bridges computational peptide design with experimental validation, accelerating the discovery of next-generation peptide therapeutics. 

**Abstract (ZH)**: CreoPep：一种基于深度学习的条件生成框架，用于设计高亲和力肽突变体并揭示新型结构动机 

---
# Taskmaster Deconstructed: A Quantitative Look at Tension, Volatility, and Viewer Ratings 

**Title (ZH)**: 任务掌控者拆解：紧张感、波动性与观众评分的定量分析 

**Authors**: David H. Silver  

**Link**: [PDF](https://arxiv.org/pdf/2505.02886)  

**Abstract**: Taskmaster is a British television show that combines comedic performance with a formal scoring system. Despite the appearance of structured competition, it remains unclear whether scoring dynamics contribute meaningfully to audience engagement. We conducted a statistical analysis of 162 episodes across 18 series, using fifteen episode-level metrics to quantify rank volatility, point spread, lead changes, and winner dominance. None of these metrics showed a significant association with IMDb ratings, even after controlling for series effects. Long-term trends suggest that average points have increased over time, while volatility has slightly declined and rank spread has remained stable. These patterns indicate an attempt to enhance competitive visibility without altering the show's structural equilibrium. We also analyzed contestant rank trajectories and identified five recurring archetypes describing performance styles. These patterns suggest that viewer interest is shaped more by contestant behavior than by game mechanics. 

**Abstract (ZH)**: Taskmaster是英国一档结合 Comedy 表演与正式评分系统的电视节目：评分动态是否对观众参与度产生实质性影响的统计分析 

---
# Unlearning vs. Obfuscation: Are We Truly Removing Knowledge? 

**Title (ZH)**: 卸学习 vs. 模糊化：我们真正删除知识了吗？ 

**Authors**: Guangzhi Sun, Potsawee Manakul, Xiao Zhan, Mark Gales  

**Link**: [PDF](https://arxiv.org/pdf/2505.02884)  

**Abstract**: Unlearning has emerged as a critical capability for large language models (LLMs) to support data privacy, regulatory compliance, and ethical AI deployment. Recent techniques often rely on obfuscation by injecting incorrect or irrelevant information to suppress knowledge. Such methods effectively constitute knowledge addition rather than true removal, often leaving models vulnerable to probing. In this paper, we formally distinguish unlearning from obfuscation and introduce a probing-based evaluation framework to assess whether existing approaches genuinely remove targeted information. Moreover, we propose DF-MCQ, a novel unlearning method that flattens the model predictive distribution over automatically generated multiple-choice questions using KL-divergence, effectively removing knowledge about target individuals and triggering appropriate refusal behaviour. Experimental results demonstrate that DF-MCQ achieves unlearning with over 90% refusal rate and a random choice-level uncertainty that is much higher than obfuscation on probing questions. 

**Abstract (ZH)**: 卸载已成为大型语言模型（LLMs）支持数据隐私、合规性和伦理AI部署的关键能力。近期技术通常依赖于混淆方法，通过注入错误或不相关信息来抑制知识。这些方法实际上构成了知识的增加而非真正的删除，往往使模型仍然容易受到探查。在本文中，我们正式区分了卸载与混淆，并引入了一种基于探查的评估框架，以评估现有方法是否真正删除了目标信息。此外，我们提出了DF-MCQ，这是一种新颖的卸载方法，通过使用KL散度对生成的多项选择题的模型预测分布进行扁平化处理，有效地删除了关于目标个体的知识，并触发适当的拒绝行为。实验结果表明，DF-MCQ的拒绝率达到90%以上，在探查问题上的随机选择级不确定性远高于混淆方法。 

---
# Rewriting Pre-Training Data Boosts LLM Performance in Math and Code 

**Title (ZH)**: 预训练数据重写提升大语言模型在数学和代码上的性能 

**Authors**: Kazuki Fujii, Yukito Tajima, Sakae Mizuki, Hinari Shimada, Taihei Shiotani, Koshiro Saito, Masanari Ohi, Masaki Kawamura, Taishi Nakamura, Takumi Okamoto, Shigeki Ishida, Kakeru Hattori, Youmi Ma, Hiroya Takamura, Rio Yokota, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.02881)  

**Abstract**: The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora. We introduce two openly licensed datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode (approximately 16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach upgrades low-quality code, maximizing data utility. SwallowMath (approximately 2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +17.7 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting delivering the largest gains. All datasets, prompts, and checkpoints are publicly available, enabling reproducible research and advancing LLM pre-training for specialized domains. 

**Abstract (ZH)**: 大型语言模型（LLMs）在程序合成和数学推理中的性能从根本上受限于其预训练语料的质量。我们引入了两个开源数据集，通过Llama 3.3社区许可发布，显著提升了LLM的性能，通过系统地重写公开数据。SwallowCode（约161亿词元）通过一个新颖的四阶段管道细化《The-Stack-v2》中的Python片段：语法验证、基于pylint的风格过滤，以及两阶段LLM重写过程，以确保风格一致并transform片段为自包含、算法高效示例。SwallowMath（约23亿词元）通过清除样板代码、恢复上下文和重新格式化解决方案为简洁、分步解释来增强Finemath-4+。在固定50亿词元的预训练预算内，持续使用SwallowCode预训练Llama-3.1-8B在HumanEval和HumanEval+上的pass@1分别提升了17.0%和17.7%，超越了基准模型的代码生成能力。类似地，替换为SwallowMath在GSM8K上的准确率提升了12.4%，在MATH上的准确率提升了7.6%。消融研究证实，每个管道阶段都贡献了增量提升，重写带来了最大的收益。所有数据集、提示和检查点都公开可用，支持可再现研究并推动LLM预训练向专业化领域的进步。 

---
# A Wireless Collaborated Inference Acceleration Framework for Plant Disease Recognition 

**Title (ZH)**: 一种用于植物病害识别的无线协作推理加速框架 

**Authors**: Hele Zhu, Xinyi Huang, Haojia Gao, Mengfei Jiang, Haohua Que, Lei Mu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02877)  

**Abstract**: Plant disease is a critical factor affecting agricultural production. Traditional manual recognition methods face significant drawbacks, including low accuracy, high costs, and inefficiency. Deep learning techniques have demonstrated significant benefits in identifying plant diseases, but they still face challenges such as inference delays and high energy consumption. Deep learning algorithms are difficult to run on resource-limited embedded devices. Offloading these models to cloud servers is confronted with the restriction of communication bandwidth, and all of these factors will influence the inference's efficiency. We propose a collaborative inference framework for recognizing plant diseases between edge devices and cloud servers to enhance inference speed. The DNN model for plant disease recognition is pruned through deep reinforcement learning to improve the inference speed and reduce energy consumption. Then the optimal split point is determined by a greedy strategy to achieve the best collaborated inference acceleration. Finally, the system for collaborative inference acceleration in plant disease recognition has been implemented using Gradio to facilitate friendly human-machine interaction. Experiments indicate that the proposed collaborative inference framework significantly increases inference speed while maintaining acceptable recognition accuracy, offering a novel solution for rapidly diagnosing and preventing plant diseases. 

**Abstract (ZH)**: 植物病害是影响农业生产的关键因素。传统的手动识别方法面临低准确性、高成本和低效率等重大缺陷。深度学习技术在识别植物病害方面显示出显著优势，但仍面临推理延迟和高能耗等挑战。深度学习算法难以在资源受限的嵌入式设备上运行。将这些模型卸载到云服务器上则受限于通信带宽，所有这些因素都会影响推理的效率。我们提出了一种边缘设备与云服务器之间的协作推理框架，以增强推理速度。通过深度强化学习对植物病害识别的DNN模型进行剪枝，以提高推理速度和降低能耗。然后通过贪婪策略确定最优分拆点，以实现最佳协作推理加速。最后，我们使用Gradio实现了植物病害识别的协作推理加速系统，便于友好的人机交互。实验表明，所提协作推理框架在保持可接受识别准确性的前提下显著提高了推理速度，提供了快速诊断和预防植物病害的新型解决方案。 

---
# Uncertainty Quantification for Machine Learning in Healthcare: A Survey 

**Title (ZH)**: 医疗领域机器学习中不确定性量化：一个综述 

**Authors**: L. Julián Lechuga López, Shaza Elsharief, Dhiyaa Al Jorf, Firas Darwish, Congbo Ma, Farah E. Shamout  

**Link**: [PDF](https://arxiv.org/pdf/2505.02874)  

**Abstract**: Uncertainty Quantification (UQ) is pivotal in enhancing the robustness, reliability, and interpretability of Machine Learning (ML) systems for healthcare, optimizing resources and improving patient care. Despite the emergence of ML-based clinical decision support tools, the lack of principled quantification of uncertainty in ML models remains a major challenge. Current reviews have a narrow focus on analyzing the state-of-the-art UQ in specific healthcare domains without systematically evaluating method efficacy across different stages of model development, and despite a growing body of research, its implementation in healthcare applications remains limited. Therefore, in this survey, we provide a comprehensive analysis of current UQ in healthcare, offering an informed framework that highlights how different methods can be integrated into each stage of the ML pipeline including data processing, training and evaluation. We also highlight the most popular methods used in healthcare and novel approaches from other domains that hold potential for future adoption in the medical context. We expect this study will provide a clear overview of the challenges and opportunities of implementing UQ in the ML pipeline for healthcare, guiding researchers and practitioners in selecting suitable techniques to enhance the reliability, safety and trust from patients and clinicians on ML-driven healthcare solutions. 

**Abstract (ZH)**: 不确定性量化（UQ）在提高医疗保健领域机器学习（ML）系统的稳健性、可靠性和可解释性方面至关重要，有助于优化资源并提高患者护理质量。尽管出现了基于ML的临床决策支持工具，但在ML模型中进行原则性的不确定性量化仍然是一个主要挑战。现有综述主要专注于具体医疗领域中最先进的UQ分析，而未系统评估不同模型开发阶段的方法有效性，尽管已有大量研究，但在医疗应用中的实施仍然有限。因此，在本文综述中，我们对当前医疗领域的不确定性量化进行全面分析，提供一个指导框架，展示不同方法如何集成到机器学习管道的每个阶段，包括数据处理、训练和评估。我们还强调在医疗领域最常用的方法以及来自其他领域的新颖方法，并指出其在未来医疗应用中的潜在采用价值。我们期望本研究能为在机器学习管道中实施不确定性量化提供清晰的概述，指导研究人员和实践者选择合适的techniques以增强ML驱动医疗解决方案的可靠性、安全性和信任度。 

---
# Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading 

**Title (ZH)**: 从阅读的眼动中解码开放式的信息寻求目标 

**Authors**: Cfir Avraham Hadar, Omer Shubi, Yoav Meiri, Yevgeni Berzak  

**Link**: [PDF](https://arxiv.org/pdf/2505.02872)  

**Abstract**: When reading, we often have specific information that interests us in a text. For example, you might be reading this paper because you are curious about LLMs for eye movements in reading, the experimental design, or perhaps you only care about the question ``but does it work?''. More broadly, in daily life, people approach texts with any number of text-specific goals that guide their reading behavior. In this work, we ask, for the first time, whether open-ended reading goals can be automatically decoded from eye movements in reading. To address this question, we introduce goal classification and goal reconstruction tasks and evaluation frameworks, and use large-scale eye tracking for reading data in English with hundreds of text-specific information seeking tasks. We develop and compare several discriminative and generative multimodal LLMs that combine eye movements and text for goal classification and goal reconstruction. Our experiments show considerable success on both tasks, suggesting that LLMs can extract valuable information about the readers' text-specific goals from eye movements. 

**Abstract (ZH)**: 当阅读时，我们往往对文本中有特定的兴趣信息。例如，你可能因为对阅读中的眼动与LLMs的关系、实验设计感兴趣，或者只关心“有效果吗？”的问题。更广泛地说，在日常生活中，人们带着各种与文本相关的目标阅读，这些目标指引着他们的阅读行为。在本研究中，我们首次探讨是否可以自动解码开放性的阅读目标从眼动中。为此，我们引入了目标分类和目标重建任务及评估框架，并使用包含数百个文本特定信息寻求任务的大量英语阅读眼动追踪数据。我们开发并比较了几种结合眼动与文本的判别性和生成性多模态LLM，用于目标分类和目标重建。我们的实验在这两方面均取得了显著成果，表明LLM可以从眼动中提取有价值的读者文本特定目标信息。 

---
# Accelerating Large Language Model Reasoning via Speculative Search 

**Title (ZH)**: 通过推测性搜索加速大规模语言模型推理 

**Authors**: Zhihai Wang, Jie Wang, Jilai Pan, Xilin Xia, Huiling Zhen, Mingxuan Yuan, Jianye Hao, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02865)  

**Abstract**: Tree-search-based reasoning methods have significantly enhanced the reasoning capability of large language models (LLMs) by facilitating the exploration of multiple intermediate reasoning steps, i.e., thoughts. However, these methods suffer from substantial inference latency, as they have to generate numerous reasoning thoughts, severely limiting LLM applicability. To address this challenge, we propose a novel Speculative Search (SpecSearch) framework that significantly accelerates LLM reasoning by optimizing thought generation. Specifically, SpecSearch utilizes a small model to strategically collaborate with a large model at both thought and token levels, efficiently generating high-quality reasoning thoughts. The major pillar of SpecSearch is a novel quality-preserving rejection mechanism, which effectively filters out thoughts whose quality falls below that of the large model's outputs. Moreover, we show that SpecSearch preserves comparable reasoning quality to the large model. Experiments on both the Qwen and Llama models demonstrate that SpecSearch significantly outperforms state-of-the-art approaches, achieving up to 2.12$\times$ speedup with comparable reasoning quality. 

**Abstract (ZH)**: 基于树搜索的推理方法显著增强了大型语言模型的推理能力，通过促进多种中间推理步骤（即思维）的探索。然而，这些方法面临着显著的推理延迟问题，因为它们需要生成大量的推理思维，极大地限制了大型语言模型的应用。为应对这一挑战，我们提出了一种新颖的猜测搜索（SpecSearch）框架，通过优化思维生成大幅加速大型语言模型的推理。具体而言，SpecSearch 在思维和标记层面利用一个小模型战略性地与大模型合作，高效生成高质量的推理思维。SpecSearch 的主要支柱是一种新颖的质量保留拒绝机制，能够有效过滤掉质量低于大模型输出的思维。此外，我们证明 SpecSearch 保留了与大模型相当的推理质量。实验结果显示，SpecSearch 在 Qwen 和 Llama 模型上均显著优于现有最佳方法，实现了最高达 2.12 倍的加速，同时保持了类似的推理质量。 

---
# Understanding University Students' Use of Generative AI: The Roles of Demographics and Personality Traits 

**Title (ZH)**: 理解大学学生使用生成式AI的作用：人口统计学特征和人格特质的角色 

**Authors**: Newnew Deng, Edward Jiusi Liu, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2505.02863)  

**Abstract**: The use of generative AI (GAI) among university students is rapidly increasing, yet empirical research on students' GAI use and the factors influencing it remains limited. To address this gap, we surveyed 363 undergraduate and graduate students in the United States, examining their GAI usage and how it relates to demographic variables and personality traits based on the Big Five model (i.e., extraversion, agreeableness, conscientiousness, and emotional stability, and intellect/imagination). Our findings reveal: (a) Students in higher academic years are more inclined to use GAI and prefer it over traditional resources. (b) Non-native English speakers use and adopt GAI more readily than native speakers. (c) Compared to White, Asian students report higher GAI usage, perceive greater academic benefits, and express a stronger preference for it. Similarly, Black students report a more positive impact of GAI on their academic performance. Personality traits also play a significant role in shaping perceptions and usage of GAI. After controlling demographic factors, we found that personality still significantly predicts GAI use and attitudes: (a) Students with higher conscientiousness use GAI less. (b) Students who are higher in agreeableness perceive a less positive impact of GAI on academic performance and express more ethical concerns about using it for academic work. (c) Students with higher emotional stability report a more positive impact of GAI on learning and fewer concerns about its academic use. (d) Students with higher extraversion show a stronger preference for GAI over traditional resources. (e) Students with higher intellect/imagination tend to prefer traditional resources. These insights highlight the need for universities to provide personalized guidance to ensure students use GAI effectively, ethically, and equitably in their academic pursuits. 

**Abstract (ZH)**: 大学学生中生成式人工智能（GAI）的使用迅速增加，然而关于学生GAI使用及其影响因素的经验研究仍较为有限。为弥补这一空白，我们对美国363名本科生和研究生进行了调查，探讨了他们的GAI使用情况及其与人口统计学变量和基于五大人格特征模型的个性特质（即外向性、宜人性、尽责性、情绪稳定性和智力/想象力）的关系。研究发现：（a）高年级学生更倾向于使用GAI，并更偏好GAI而非传统资源。（b）非英语母语者比英语母语者更快速地使用和接受GAI。（c）与白人相比，亚裔学生报告了更高的GAI使用频率、更显著的学业益处以及更强的偏好。同样，黑人学生报告了GAI对学业成绩更积极的影响。个性特质也显著影响着对GAI的感知和使用。在控制了人口统计学因素后，我们发现个性仍然显著预测GAI的使用和态度：（a）更尽责的学生使用GAI较少。（b）更宜人的学生认为GAI对学业表现的影响较少积极，并对将其用于学术工作时的伦理问题表达了更多担忧。（c）更情绪稳定的学生报告了GAI对学习的更积极影响以及对其学术使用时的更少担忧。（d）更外向的学生对GAI比传统资源表现出了更强的偏好。（e）更高智力/想象力的学生倾向于偏好传统资源。这些洞察突显了大学需要提供个性化指导，以确保学生能够有效地、伦理地和公平地在学术追求中使用GAI。 

---
# Cannot See the Forest for the Trees: Invoking Heuristics and Biases to Elicit Irrational Choices of LLMs 

**Title (ZH)**: 无法只见树木而不见森林：调用启发式和偏差以揭示LLMs的非理性选择 

**Authors**: Haoming Yang, Ke Ma, Xiaojun Jia, Yingfei Sun, Qianqian Xu, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02862)  

**Abstract**: Despite the remarkable performance of Large Language Models (LLMs), they remain vulnerable to jailbreak attacks, which can compromise their safety mechanisms. Existing studies often rely on brute-force optimization or manual design, failing to uncover potential risks in real-world scenarios. To address this, we propose a novel jailbreak attack framework, ICRT, inspired by heuristics and biases in human cognition. Leveraging the simplicity effect, we employ cognitive decomposition to reduce the complexity of malicious prompts. Simultaneously, relevance bias is utilized to reorganize prompts, enhancing semantic alignment and inducing harmful outputs effectively. Furthermore, we introduce a ranking-based harmfulness evaluation metric that surpasses the traditional binary success-or-failure paradigm by employing ranking aggregation methods such as Elo, HodgeRank, and Rank Centrality to comprehensively quantify the harmfulness of generated content. Experimental results show that our approach consistently bypasses mainstream LLMs' safety mechanisms and generates high-risk content, providing insights into jailbreak attack risks and contributing to stronger defense strategies. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）表现出色，但仍容易受到 Jailbreak 攻击，这会危及它们的安全机制。现有研究常依赖于暴力优化或手动设计，未能在真实场景中揭示潜在风险。为应对这一挑战，我们提出了一种新的 Jailbreak 攻击框架 ICRT，该框架受人类认知中的启发和偏差启发。利用简单效应，我们采用认知分解来减少恶意提示的复杂性。同时，利用相关性偏差重组提示，增强语义对齐并有效诱导有害输出。此外，我们引入了一种基于排名的有害性评估指标，该指标通过使用 Elo、HodgeRank 和 Rank Centrality 等排名聚合方法超越传统的成功或失败二元范式，全面量化生成内容的有害性。实验结果表明，我们的方法能一致地绕过主流 LLMs 的安全机制并生成高风险内容，为 Jailbreak 攻击风险提供了见解，并助力更强的安全防御策略。 

---
# Neural Orchestration for Multi-Agent Systems: A Deep Learning Framework for Optimal Agent Selection in Multi-Domain Task Environments 

**Title (ZH)**: 多域任务环境中的多代理系统神经编排：深度学习框架下的最优代理选择 

**Authors**: Kushagra Agrawal, Nisharg Nargund  

**Link**: [PDF](https://arxiv.org/pdf/2505.02861)  

**Abstract**: Multi-agent systems (MAS) are foundational in simulating complex real-world scenarios involving autonomous, interacting entities. However, traditional MAS architectures often suffer from rigid coordination mechanisms and difficulty adapting to dynamic tasks. We propose MetaOrch, a neural orchestration framework for optimal agent selection in multi-domain task environments. Our system implements a supervised learning approach that models task context, agent histories, and expected response quality to select the most appropriate agent for each task. A novel fuzzy evaluation module scores agent responses along completeness, relevance, and confidence dimensions, generating soft supervision labels for training the orchestrator. Unlike previous methods that hard-code agent-task mappings, MetaOrch dynamically predicts the most suitable agent while estimating selection confidence. Experiments in simulated environments with heterogeneous agents demonstrate that our approach achieves 86.3% selection accuracy, significantly outperforming baseline strategies including random selection and round-robin scheduling. The modular architecture emphasizes extensibility, allowing agents to be registered, updated, and queried independently. Results suggest that neural orchestration offers a powerful approach to enhancing the autonomy, interpretability, and adaptability of multi-agent systems across diverse task domains. 

**Abstract (ZH)**: 元编排：多域任务环境中的神经优化代理选择框架 

---
# Enhancing ML Model Interpretability: Leveraging Fine-Tuned Large Language Models for Better Understanding of AI 

**Title (ZH)**: 提升机器学习模型可解释性：利用 fine-tuned 大型语言模型以更好地理解 AI 

**Authors**: Jonas Bokstaller, Julia Altheimer, Julian Dormehl, Alina Buss, Jasper Wiltfang, Johannes Schneider, Maximilian Röglinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.02859)  

**Abstract**: Across various sectors applications of eXplainableAI (XAI) gained momentum as the increasing black-boxedness of prevailing Machine Learning (ML) models became apparent. In parallel, Large Language Models (LLMs) significantly developed in their abilities to understand human language and complex patterns. By combining both, this paper presents a novel reference architecture for the interpretation of XAI through an interactive chatbot powered by a fine-tuned LLM. We instantiate the reference architecture in the context of State-of-Health (SoH) prediction for batteries and validate its design in multiple evaluation and demonstration rounds. The evaluation indicates that the implemented prototype enhances the human interpretability of ML, especially for users with less experience with XAI. 

**Abstract (ZH)**: 跨领域应用的可解释人工智能(XAI)随着现有机器学习(ML)模型的黑盒性质日益明显而逐渐获得动力。与此同时，大规模语言模型(LLMs)在理解和识别复杂模式方面的能力有了显著提升。通过结合两者，本文提出了一种新的参考架构，利用微调后的LLM驱动交互式聊天机器人来解释XAI。我们将该参考架构应用于电池的状态健康(SoH)预测，并在多次评估和演示环节中验证了其设计。评估结果显示，实现的原型提高了机器学习的人类可解释性，尤其是对于XAI经验较少的用户。 

---
# AI Education in a Mirror: Challenges Faced by Academic and Industry Experts 

**Title (ZH)**: 镜中的AI教育：学术与产业专家面临的挑战 

**Authors**: Mahir Akgun, Hadi Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2505.02856)  

**Abstract**: As Artificial Intelligence (AI) technologies continue to evolve, the gap between academic AI education and real-world industry challenges remains an important area of investigation. This study provides preliminary insights into challenges AI professionals encounter in both academia and industry, based on semi-structured interviews with 14 AI experts - eight from industry and six from academia. We identify key challenges related to data quality and availability, model scalability, practical constraints, user behavior, and explainability. While both groups experience data and model adaptation difficulties, industry professionals more frequently highlight deployment constraints, resource limitations, and external dependencies, whereas academics emphasize theoretical adaptation and standardization issues. These exploratory findings suggest that AI curricula could better integrate real-world complexities, software engineering principles, and interdisciplinary learning, while recognizing the broader educational goals of building foundational and ethical reasoning skills. 

**Abstract (ZH)**: 随着人工智能（AI）技术的不断发展，学术AI教育与现实工业挑战之间的差距仍是值得深入研究的重要领域。本研究基于对14名AI专家的半结构化访谈（其中8人来自行业，6人来自学术界）提供了一些初步见解，探讨了AI专业人士在学术界和工业界面临的挑战。我们识别出与数据质量与可用性、模型可扩展性、实际约束、用户行为以及解释性相关的关键挑战。尽管两组人都经历了数据和模型适应的困难，但行业专业人士更频繁地强调部署约束、资源限制和外部依赖性，而学术界人士则更强调理论适应和标准化问题。这些探索性发现表明，AI课程应更好地融合现实世界的复杂性、软件工程原则以及跨学科学习，同时认识到更广泛的教育目标是培养基础性和伦理推理能力。 

---
# Ensuring Reproducibility in Generative AI Systems for General Use Cases: A Framework for Regression Testing and Open Datasets 

**Title (ZH)**: 确保通用场景下生成式AI系统可再现性：一种回归测试和开源数据集框架 

**Authors**: Masumi Morishige, Ryo Koshihara  

**Link**: [PDF](https://arxiv.org/pdf/2505.02854)  

**Abstract**: Reproducibility and reliability remain pressing challenges for generative AI systems whose behavior can drift with each model update or prompt revision. We introduce GPR-bench, a lightweight, extensible benchmark that operationalizes regression testing for general purpose use cases. GPR-bench couples an open, bilingual (English and Japanese) dataset covering eight task categories (e.g., text generation, code generation, and information retrieval) and 10 scenarios in each task categories (80 total test cases for each language) with an automated evaluation pipeline that employs "LLM-as-a-Judge" scoring of correctness and conciseness. Experiments across three recent model versions - gpt-4o-mini, o3-mini, and o4-mini - and two prompt configurations (default versus concise-writing instruction) reveal heterogeneous quality. Our results show that newer models generally improve correctness, but the differences are modest and not statistically significant, suggesting that GPR-bench may not be sufficiently challenging to differentiate between recent model versions. In contrast, the concise-writing instruction significantly enhances conciseness (+12.37 pp, Mann-Whitney U test: p < 0.001, effect size r = 0.2995) with minimal degradations on accuracy (-1.7 pp), demonstrating the effectiveness of prompt engineering. Released under the MIT License, GPR- bench lowers the barrier to initiating reproducibility monitoring and provides a foundation for community-driven extensions, while also raising important considerations about benchmark design for rapidly evolving language models. 

**Abstract (ZH)**: 生成式AI系统的重现性和可靠性仍然是紧迫的挑战，其行为可能会随着每次模型更新或提示修改而发生偏移。我们介绍了GPR-bench，这是一种轻量级、可扩展的基准测试工具，将回归测试应用于通用用途场景。GPR-bench 结合了一个包含八种任务类别（例如文本生成、代码生成和信息检索）的开放双语（英语和日语）数据集，每个任务类别中有10种场景（每种语言共有80个测试用例），以及一个自动评分管道，该管道采用“语言模型作为评判者”来评估正确性和简洁性。在三个最近模型版本（gpt-4o-mini、o3-mini和o4-mini）和两种提示配置（默认提示与简洁写作指令）下进行的实验揭示了不同的质量水平。结果显示，较新的模型通常提高了正确性，但差异不大且不具备统计显著性，表明GPR-bench可能不足以区分最近的模型版本。相比之下，简洁写作指令显著提高了简洁性（+12.37个百分点，曼-惠特尼U检验：p<0.001，效应大小r=0.2995），同时对准确性的影响较小（-1.7个百分点），这表明提示工程的有效性。GPR-bench在MIT许可证下发布，降低了开始重现性监控的门槛，并为社区驱动的扩展提供了基础，同时也引发了关于快速演化的语言模型基准设计的重要考虑。 

---
# A Computational Model of Inclusive Pedagogy: From Understanding to Application 

**Title (ZH)**: 包容性教学计算模型：从理解到应用 

**Authors**: Francesco Balzan, Pedro P. Santos, Maurizio Gabbrielli, Mahault Albarracin, Manuel Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2505.02853)  

**Abstract**: Human education transcends mere knowledge transfer, it relies on co-adaptation dynamics -- the mutual adjustment of teaching and learning strategies between agents. Despite its centrality, computational models of co-adaptive teacher-student interactions (T-SI) remain underdeveloped. We argue that this gap impedes Educational Science in testing and scaling contextual insights across diverse settings, and limits the potential of Machine Learning systems, which struggle to emulate and adaptively support human learning processes. To address this, we present a computational T-SI model that integrates contextual insights on human education into a testable framework. We use the model to evaluate diverse T-SI strategies in a realistic synthetic classroom setting, simulating student groups with unequal access to sensory information. Results show that strategies incorporating co-adaptation principles (e.g., bidirectional agency) outperform unilateral approaches (i.e., where only the teacher or the student is active), improving the learning outcomes for all learning types. Beyond the testing and scaling of context-dependent educational insights, our model enables hypothesis generation in controlled yet adaptable environments. This work bridges non-computational theories of human education with scalable, inclusive AI in Education systems, providing a foundation for equitable technologies that dynamically adapt to learner needs. 

**Abstract (ZH)**: 人类教育超越了单纯的知识传递，它依赖于共适应动力学——教与学策略之间的相互调整。尽管共适应教师-学生互动(T-SI)在其中心地位，计算模型的研究仍有待发展。我们主张这一空白阻碍了教育科学在不同情境中测试和扩展洞察力，并限制了机器学习系统的能力，后者难以模拟和适应性支持人类学习过程。为解决这一问题，我们提出了一种计算T-SI模型，将人类教育的环境洞见整合到可测试的框架中。我们使用该模型在现实合成教室环境中评估多样化的T-SI策略，模拟具有不同感官信息访问权限的学生群体。结果表明，包含共适应原则（如双向自主）的策略优于单向方法（即仅教师或学生活跃），从而改善了各种学习类型的学习成果。除了测试和扩展情境依赖性教育洞察力之外，该模型还使人们能够在可控且可适应的环境中生成假设。本研究将非计算的人类教育理论与可扩展且包容性的教育人工智能系统相结合，为动态适应学习者需求的公平技术奠定了基础。 

---
# 30DayGen: Leveraging LLMs to Create a Content Corpus for Habit Formation 

**Title (ZH)**: 30DayGen: 利用大语言模型创建习惯形成内容 corpus 

**Authors**: Franklin Zhang, Sonya Zhang, Alon Halevy  

**Link**: [PDF](https://arxiv.org/pdf/2505.02851)  

**Abstract**: In this paper, we present 30 Day Me, a habit formation application that leverages Large Language Models (LLMs) to help users break down their goals into manageable, actionable steps and track their progress. Central to the app is the 30DAYGEN system, which generates 3,531 unique 30-day challenges sourced from over 15K webpages, and enables runtime search of challenge ideas aligned with user-defined goals. We showcase how LLMs can be harnessed to rapidly construct domain specific content corpora for behavioral and educational purposes, and propose a practical pipeline that incorporates effective LLM enhanced approaches for content generation and semantic deduplication. 

**Abstract (ZH)**: 本文介绍了30 Day Me，这一利用大规模语言模型（LLMs）帮助用户将目标分解为可管理的行动步骤并跟踪进度的应用程序。应用程序的核心是30DAYGEN系统，该系统从超过15000个网页中生成了3531个独特的30天挑战，并能够根据用户定义的目标进行运行时挑战构思搜索。我们展示了如何利用LLMs快速构建用于行为和教育目的的专业领域内容集合，并提出了一个结合有效LLM增强方法的内容生成和语义去重的实际管道。 

---
# Harnessing Structured Knowledge: A Concept Map-Based Approach for High-Quality Multiple Choice Question Generation with Effective Distractors 

**Title (ZH)**: 基于概念图的方法：一种用于生成高质量多项选择题的有效干扰项的概念图基方法 

**Authors**: Nicy Scaria, Silvester John Joseph Kennedy, Diksha Seth, Ananya Thakur, Deepak Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2505.02850)  

**Abstract**: Generating high-quality MCQs, especially those targeting diverse cognitive levels and incorporating common misconceptions into distractor design, is time-consuming and expertise-intensive, making manual creation impractical at scale. Current automated approaches typically generate questions at lower cognitive levels and fail to incorporate domain-specific misconceptions. This paper presents a hierarchical concept map-based framework that provides structured knowledge to guide LLMs in generating MCQs with distractors. We chose high-school physics as our test domain and began by developing a hierarchical concept map covering major Physics topics and their interconnections with an efficient database design. Next, through an automated pipeline, topic-relevant sections of these concept maps are retrieved to serve as a structured context for the LLM to generate questions and distractors that specifically target common misconceptions. Lastly, an automated validation is completed to ensure that the generated MCQs meet the requirements provided. We evaluate our framework against two baseline approaches: a base LLM and a RAG-based generation. We conducted expert evaluations and student assessments of the generated MCQs. Expert evaluation shows that our method significantly outperforms the baseline approaches, achieving a success rate of 75.20% in meeting all quality criteria compared to approximately 37% for both baseline methods. Student assessment data reveal that our concept map-driven approach achieved a significantly lower guess success rate of 28.05% compared to 37.10% for the baselines, indicating a more effective assessment of conceptual understanding. The results demonstrate that our concept map-based approach enables robust assessment across cognitive levels and instant identification of conceptual gaps, facilitating faster feedback loops and targeted interventions at scale. 

**Abstract (ZH)**: 基于层次概念图的生成高质量选择题框架：指导LLM生成针对认知层次和常见误解的选择题 

---
# Enhancing tutoring systems by leveraging tailored promptings and domain knowledge with Large Language Models 

**Title (ZH)**: 利用大型语言模型定制提示和领域知识增强辅导系统 

**Authors**: Mohsen Balavar, Wenli Yang, David Herbert, Soonja Yeom  

**Link**: [PDF](https://arxiv.org/pdf/2505.02849)  

**Abstract**: Recent advancements in artificial intelligence (AI) and machine learning have reignited interest in their impact on Computer-based Learning (CBL). AI-driven tools like ChatGPT and Intelligent Tutoring Systems (ITS) have enhanced learning experiences through personalisation and flexibility. ITSs can adapt to individual learning needs and provide customised feedback based on a student's performance, cognitive state, and learning path. Despite these advances, challenges remain in accommodating diverse learning styles and delivering real-time, context-aware feedback. Our research aims to address these gaps by integrating skill-aligned feedback via Retrieval Augmented Generation (RAG) into prompt engineering for Large Language Models (LLMs) and developing an application to enhance learning through personalised tutoring in a computer science programming context. The pilot study evaluated a proposed system using three quantitative metrics: readability score, response time, and feedback depth, across three programming tasks of varying complexity. The system successfully sorted simulated students into three skill-level categories and provided context-aware feedback. This targeted approach demonstrated better effectiveness and adaptability compared to general methods. 

**Abstract (ZH)**: 近期人工智能和机器学习的进展重新引发了其对基于计算机的学习（CBL）影响的兴趣。通过个性化和灵活性提升学习体验的人工智能驱动工具如ChatGPT和智能辅导系统（ITS）已得到增强。ITS可以根据学生的表现、认知状态和学习路径适应个性化需求并提供定制反馈。尽管取得了这些进展，但在适应多种学习风格和提供实时、情境感知反馈方面仍存在挑战。我们的研究旨在通过将技能对齐反馈结合检索增强生成（RAG）技术融入大语言模型（LLMs）的提示工程，并开发一个应用程序，以在计算机科学编程背景下通过个性化辅导提升学习效果来弥补这些差距。试点研究使用三种定量指标：可读性分数、响应时间和反馈深度，评估了一个拟议系统的性能，涉及三个不同复杂程度的编程任务。该系统成功地将模拟学生分类为三个技能级别，并提供了情境感知反馈。这种有针对性的方法在效果和适应性方面优于通用方法。 

---
# Aligning Large Language Models with Healthcare Stakeholders: A Pathway to Trustworthy AI Integration 

**Title (ZH)**: 将大型语言模型与医疗健康利益相关者对齐：通往可信赖AI集成的道路 

**Authors**: Kexin Ding, Mu Zhou, Akshay Chaudhari, Shaoting Zhang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2505.02848)  

**Abstract**: The wide exploration of large language models (LLMs) raises the awareness of alignment between healthcare stakeholder preferences and model outputs. This alignment becomes a crucial foundation to empower the healthcare workflow effectively, safely, and responsibly. Yet the varying behaviors of LLMs may not always match with healthcare stakeholders' knowledge, demands, and values. To enable a human-AI alignment, healthcare stakeholders will need to perform essential roles in guiding and enhancing the performance of LLMs. Human professionals must participate in the entire life cycle of adopting LLM in healthcare, including training data curation, model training, and inference. In this review, we discuss the approaches, tools, and applications of alignments between healthcare stakeholders and LLMs. We demonstrate that LLMs can better follow human values by properly enhancing healthcare knowledge integration, task understanding, and human guidance. We provide outlooks on enhancing the alignment between humans and LLMs to build trustworthy real-world healthcare applications. 

**Abstract (ZH)**: 大型语言模型在医疗健康领域的广泛应用提高了对医疗健康利益相关者偏好与模型输出之间一致性的认识。这种一致性成为有效、安全和负责任地赋能医疗工作流程的基础。然而，大型语言模型的行为可能并不始终与医疗健康利益相关者的知识、需求和价值观相符。为了实现人机一致，医疗健康利益相关者需要在引导和提升大型语言模型性能方面发挥关键作用。专业人员必须参与将大型语言模型应用于医疗保健的整个生命周期，包括训练数据的整理、模型训练和推理。在本文综述中，我们探讨了医疗健康利益相关者与大型语言模型之间一致性的方法、工具和应用。我们表明，通过适当增强医疗健康知识整合、任务理解和人为引导，大型语言模型能够更好地体现人类价值观。我们还展望了增强人类与大型语言模型之间一致性的前景，以构建可信赖的医疗健康实际应用。 

---
# Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in Large Language Models 

**Title (ZH)**: 有感知能力的代理作为法官：评估大型语言模型的高阶社交认知 

**Authors**: Bang Zhang, Ruotian Ma, Qingxuan Jiang, Peisong Wang, Jiaqi Chen, Zheng Xie, Xingyu Chen, Yue Wang, Fanghua Ye, Jian Li, Yifan Yang, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02847)  

**Abstract**: Assessing how well a large language model (LLM) understands human, rather than merely text, remains an open challenge. To bridge the gap, we introduce Sentient Agent as a Judge (SAGE), an automated evaluation framework that measures an LLM's higher-order social cognition. SAGE instantiates a Sentient Agent that simulates human-like emotional changes and inner thoughts during interaction, providing a more realistic evaluation of the tested model in multi-turn conversations. At every turn, the agent reasons about (i) how its emotion changes, (ii) how it feels, and (iii) how it should reply, yielding a numerical emotion trajectory and interpretable inner thoughts. Experiments on 100 supportive-dialogue scenarios show that the final Sentient emotion score correlates strongly with Barrett-Lennard Relationship Inventory (BLRI) ratings and utterance-level empathy metrics, validating psychological fidelity. We also build a public Sentient Leaderboard covering 18 commercial and open-source models that uncovers substantial gaps (up to 4x) between frontier systems (GPT-4o-Latest, Gemini2.5-Pro) and earlier baselines, gaps not reflected in conventional leaderboards (e.g., Arena). SAGE thus provides a principled, scalable and interpretable tool for tracking progress toward genuinely empathetic and socially adept language agents. 

**Abstract (ZH)**: 评估大型语言模型在理解人类而非仅仅文本方面的能力依然是一项开放的挑战。为了弥合这一差距，我们引入了觉知代理作为评判者（SAGE），这是一种自动化评估框架，用于测量大型语言模型的高层次社会认知能力。SAGE 实现了一个模拟人类情感变化和交互中内心想法的觉知代理，从而为多轮对话中的测试模型提供更为现实的评估。在每次交互中，代理会推理其情感的变化、感觉以及应如何回应，产生一个数值化的情感轨迹和可解释的内心想法。在100个支持性对话场景上的实验表明，最终的觉知情感得分与Barrett-Lennard关系量表（BLRI）评分和语句层面的共情指标高度相关，验证了心理真实性的存在。我们还构建了一个公开的觉知排行榜，涵盖了18个商用和开源模型，揭示了前沿系统（如GPT-4o-Latest、Gemini2.5-Pro）与早期基线之间存在的显著差距（最多4倍），这些差距在传统的排行榜（如Arena）中并未体现。因此，SAGE 提供了一个原则性、可扩展且可解释的工具，用于追踪朝着真正具有共情和社会适应语言代理方向的进步。 

---
# The Precautionary Principle and the Innovation Principle: Incompatible Guides for AI Innovation Governance? 

**Title (ZH)**: 预防原则与创新原则：AI创新治理的矛盾向导？ 

**Authors**: Kim Kaivanto  

**Link**: [PDF](https://arxiv.org/pdf/2505.02846)  

**Abstract**: In policy debates concerning the governance and regulation of Artificial Intelligence (AI), both the Precautionary Principle (PP) and the Innovation Principle (IP) are advocated by their respective interest groups. Do these principles offer wholly incompatible and contradictory guidance? Does one necessarily negate the other? I argue here that provided attention is restricted to weak-form PP and IP, the answer to both of these questions is "No." The essence of these weak formulations is the requirement to fully account for type-I error costs arising from erroneously preventing the innovation's diffusion through society (i.e. mistaken regulatory red-lighting) as well as the type-II error costs arising from erroneously allowing the innovation to diffuse through society (i.e. mistaken regulatory green-lighting). Within the Signal Detection Theory (SDT) model developed here, weak-PP red-light (weak-IP green-light) determinations are optimal for sufficiently small (large) ratios of expected type-I to type-II error costs. For intermediate expected cost ratios, an amber-light 'wait-and-monitor' policy is optimal. Regulatory sandbox instruments allow AI testing and experimentation to take place within a structured environment of limited duration and societal scale, whereby the expected cost ratio falls within the 'wait-and-monitor' range. Through sandboxing regulators and innovating firms learn more about the expected cost ratio, and what respective adaptations -- of regulation, of technical solution, of business model, or combination thereof, if any -- are needed to keep the ratio out of the weak-PP red-light zone. 

**Abstract (ZH)**: 人工智能治理和监管政策辩论中 precautionary principle 和 innovation principle 的兼容性探究 

---
# Physical foundations for trustworthy medical imaging: a review for artificial intelligence researchers 

**Title (ZH)**: 可信赖医疗成像的物理基础：人工智能研究者的综述 

**Authors**: Miriam Cobo, David Corral Fontecha, Wilson Silva, Lara Lloret Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2505.02843)  

**Abstract**: Artificial intelligence in medical imaging has seen unprecedented growth in the last years, due to rapid advances in deep learning and computing resources. Applications cover the full range of existing medical imaging modalities, with unique characteristics driven by the physics of each technique. Yet, artificial intelligence professionals entering the field, and even experienced developers, often lack a comprehensive understanding of the physical principles underlying medical image acquisition, which hinders their ability to fully leverage its potential. The integration of physics knowledge into artificial intelligence algorithms enhances their trustworthiness and robustness in medical imaging, especially in scenarios with limited data availability. In this work, we review the fundamentals of physics in medical images and their impact on the latest advances in artificial intelligence, particularly, in generative models and reconstruction algorithms. Finally, we explore the integration of physics knowledge into physics-inspired machine learning models, which leverage physics-based constraints to enhance the learning of medical imaging features. 

**Abstract (ZH)**: 医学影像中的人工智能在过去几年由于深度学习和计算资源的快速进步取得了前所未有的增长。应用涵盖了所有现有的医学影像模态，每种技术的独特特性由其物理原理驱动。然而，进入该领域的AI专业人士，甚至是经验丰富的开发人员，往往缺乏对医学影像获取物理原理的全面理解，这限制了他们充分发挥其潜力的能力。将物理知识整合到人工智能算法中，可以增强其在医学影像中的可信度和稳健性，特别是在数据有限的情况下。在本文中，我们回顾了医学影像中物理原理的基本知识及其对最新人工智能进展的影响，特别是生成模型和重建算法。最后，我们探讨了将物理知识整合到基于物理原理的机器学习模型中，利用基于物理的约束条件来增强医学影像特征的学习。 

---
# Snakemaker: Seamlessly transforming ad-hoc analyses into sustainable Snakemake workflows with generative AI 

**Title (ZH)**: Snakemaker: 使用生成式人工智能无缝将临时分析转换为可持续的Snakemake工作流 

**Authors**: Marco Masera, Alessandro Leone, Johannes Köster, Ivan Molineris  

**Link**: [PDF](https://arxiv.org/pdf/2505.02841)  

**Abstract**: Reproducibility and sustainability present significant challenges in bioinformatics software development, where rapidly evolving tools and complex workflows often result in short-lived or difficult-to-adapt pipelines. This paper introduces Snakemaker, a tool that leverages generative AI to facilitate researchers build sustainable data analysis pipelines by converting unstructured code into well-defined Snakemake workflows. Snakemaker non-invasively tracks the work performed in the terminal by the researcher, analyzes execution patterns, and generates Snakemake workflows that can be integrated into existing pipelines. Snakemaker also supports the transformation of monolithic Ipython Notebooks into modular Snakemake pipelines, resolving the global state of the notebook into discrete, file-based interactions between rules. An integrated chat assistant provides users with fine-grained control through natural language instructions. Snakemaker generates high-quality Snakemake workflows by adhering to the best practices, including Conda environment tracking, generic rule generation and loop unrolling. By lowering the barrier between prototype and production-quality code, Snakemaker addresses a critical gap in computational reproducibility for bioinformatics research. 

**Abstract (ZH)**: 生物信息学软件开发中的可重复性和可持续性挑战显著，快速发展的工具和复杂的 workflows 通常导致短暂或难以适应的管道。本文介绍了 Snakemaker，一种利用生成式 AI 的工具，帮助研究人员通过将无结构代码转换为定义清晰的 Snakemake 工作流来构建可持续的数据分析管道。Snakemaker 非侵入性地跟踪研究人员在终端中完成的工作，分析执行模式，并生成可以集成到现有管道中的 Snakemake 工作流。Snakemaker 还支持将庞大的 IPython 笔记本转换为模块化的 Snakemake 管道，将笔记本的全局状态分解为离散的、基于文件的规则之间的交互。集成的聊天助手通过自然语言指令为用户提供精细控制。Snakemaker 通过遵守最佳实践（包括 Conda 环境跟踪、通用规则生成和循环展开），生成高质量的 Snakemake 工作流。通过降低从原型到生产质量代码的障碍，Snakemaker 解决了生物信息学研究中计算可重复性的一个关键缺口。 

---
