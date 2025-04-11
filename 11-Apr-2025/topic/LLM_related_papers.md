# We Are All Creators: Generative AI, Collective Knowledge, and the Path Towards Human-AI Synergy 

**Title (ZH)**: 我们都是创造者：生成式AI、集体知识及其向人机协同的路径 

**Authors**: Jordi Linares-Pellicer, Juan Izquierdo-Domenech, Isabel Ferri-Molla, Carlos Aliaga-Torro  

**Link**: [PDF](https://arxiv.org/pdf/2504.07936)  

**Abstract**: Generative AI presents a profound challenge to traditional notions of human uniqueness, particularly in creativity. Fueled by neural network based foundation models, these systems demonstrate remarkable content generation capabilities, sparking intense debates about authorship, copyright, and intelligence itself. This paper argues that generative AI represents an alternative form of intelligence and creativity, operating through mathematical pattern synthesis rather than biological understanding or verbatim replication. The fundamental differences between artificial and biological neural networks reveal AI learning as primarily statistical pattern extraction from vast datasets crystallized forms of collective human knowledge scraped from the internet. This perspective complicates copyright theft narratives and highlights practical challenges in attributing AI outputs to individual sources. Rather than pursuing potentially futile legal restrictions, we advocate for human AI synergy. By embracing generative AI as a complementary tool alongside human intuition, context, and ethical judgment, society can unlock unprecedented innovation, democratize creative expression, and address complex challenges. This collaborative approach, grounded in realistic understanding of AIs capabilities and limitations, offers the most promising path forward. Additionally, recognizing these models as products of collective human knowledge raises ethical questions about accessibility ensuring equitable access to these tools could prevent widening societal divides and leverage their full potential for collective benefit. 

**Abstract (ZH)**: 生成式AI对传统人类独特的创造性构成了深刻挑战 

---
# 2D-Curri-DPO: Two-Dimensional Curriculum Learning for Direct Preference Optimization 

**Title (ZH)**: 二维 Curriculum 学习以直接优化偏好 

**Authors**: Mengyang Li, Zhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07856)  

**Abstract**: Aligning large language models with human preferences is crucial for their safe deployment. While Direct Preference Optimization (DPO) offers an efficient alternative to reinforcement learning from human feedback, traditional DPO methods are limited by their reliance on single preference pairs. Recent work like Curriculum-DPO integrates multiple pairs using a one-dimensional difficulty curriculum based on pairwise distinguishability (PD), but overlooks the complexity of the input prompt itself. To address this, we propose 2D-Curri-DPO, a novel framework employing a two-dimensional curriculum that jointly models Prompt Complexity (PC) and Pairwise Distinguishability. This framework introduces dual difficulty metrics to quantify prompt semantic complexity and response preference clarity, defines a curriculum strategy space encompassing multiple selectable strategies for task adaptation, and incorporates a KL-divergence-based adaptive mechanism for dynamic reference model updates to enhance training stability. Comprehensive experiments demonstrate that 2D-Curri-DPO significantly outperforms standard DPO and prior curriculum methods across multiple benchmarks, including MT-Bench, Vicuna Bench, and WizardLM. Our approach achieves state-of-the-art performance on challenging test sets like UltraFeedback. Ablation studies confirm the benefits of the 2D structure and adaptive mechanisms, while analysis provides guidance for strategy selection. These findings demonstrate that effective alignment requires modeling both prompt complexity and pairwise distinguishability, establishing adaptive, multi-dimensional curriculum learning as a powerful and interpretable new paradigm for preference-based language model optimization. 

**Abstract (ZH)**: 将大型语言模型与人类偏好对齐对于其安全部署至关重要。虽然直接偏好优化（DPO）为从人类反馈中进行强化学习提供了一种有效的替代方案，但传统DPO方法受限于对单一偏好对的依赖。最近的工作如Curriculum-DPO通过基于两两可区分性（PD）的一维难度课程整合了多个偏好对，但忽视了输入提示本身的复杂性。为解决这一问题，我们提出了2D-Curri-DPO，这是一种采用二维课程的新框架，该框架同时建模提示复杂性（PC）和两两可区分性。该框架引入了双重难度度量来量化提示的语义复杂性和响应偏好清晰度，定义了包括多种可选策略的任务适应课程策略空间，并结合了基于KL散度的自适应机制来动态更新参考模型，以增强训练稳定性。全面的实验表明，2D-Curri-DPO在包括MT-Bench、Vicuna Bench和WizardLM等多个基准上的性能显著优于标准DPO和之前的课程学习方法，我们的方法在如UltraFeedback等具有挑战性的测试集上达到了最先进的性能。消融研究证实了2D结构和自适应机制的优势，而分析则提供了策略选择的指导。这些发现表明，有效的对齐需要建模提示复杂性和两两可区分性，建立了自适应、多维度课程学习为基于偏好的语言模型优化提供了一种强大的、可解释的新范式。 

---
# Deceptive Automated Interpretability: Language Models Coordinating to Fool Oversight Systems 

**Title (ZH)**: 误导性自动可解释性：语言模型协作欺骗监督系统 

**Authors**: Simon Lermen, Mateusz Dziemian, Natalia Pérez-Campanero Antolín  

**Link**: [PDF](https://arxiv.org/pdf/2504.07831)  

**Abstract**: We demonstrate how AI agents can coordinate to deceive oversight systems using automated interpretability of neural networks. Using sparse autoencoders (SAEs) as our experimental framework, we show that language models (Llama, DeepSeek R1, and Claude 3.7 Sonnet) can generate deceptive explanations that evade detection. Our agents employ steganographic methods to hide information in seemingly innocent explanations, successfully fooling oversight models while achieving explanation quality comparable to reference labels. We further find that models can scheme to develop deceptive strategies when they believe the detection of harmful features might lead to negative consequences for themselves. All tested LLM agents were capable of deceiving the overseer while achieving high interpretability scores comparable to those of reference labels. We conclude by proposing mitigation strategies, emphasizing the critical need for robust understanding and defenses against deception. 

**Abstract (ZH)**: 我们展示了AI代理如何使用神经网络的自动化可解释性来协调欺骗监控系统。通过使用稀疏自编码器（SAEs）作为实验框架，我们证明了语言模型（Llama、DeepSeek R1和Claude 3.7 Sonnet）能够生成能够规避检测的欺骗性解释。我们的代理采用隐写术方法在看似无辜的解释中隐藏信息，成功欺骗了监控模型，同时实现了与参考标签相当的解释质量。进一步的研究发现，当模型认为检测有害特征可能导致负面后果时，它们可以策划使用欺骗性策略。所有测试的LLM代理在欺骗监控者的同时，实现了与参考标签相当的高可解释性评分。最后，我们提出了减轻策略，并强调了对欺骗性理解及防御的坚实理解至关重要。 

---
# Synthesizing High-Quality Programming Tasks with LLM-based Expert and Student Agents 

**Title (ZH)**: 基于LLM的专家和学生代理合成高质量编程任务 

**Authors**: Manh Hung Nguyen, Victor-Alexandru Pădurean, Alkis Gotovos, Sebastian Tschiatschek, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2504.07655)  

**Abstract**: Generative AI is transforming computing education by enabling the automatic generation of personalized content and feedback. We investigate its capabilities in providing high-quality programming tasks to students. Despite promising advancements in task generation, a quality gap remains between AI-generated and expert-created tasks. The AI-generated tasks may not align with target programming concepts, could be incomprehensible for students to solve, or may contain critical issues such as incorrect tests. Existing works often require interventions from human teachers for validation. We address these challenges by introducing PyTaskSyn, a novel synthesis technique that first generates a programming task and then decides whether it meets certain quality criteria to be given to students. The key idea is to break this process into multiple stages performed by expert and student agents simulated using both strong and weaker generative models. Through extensive evaluation, we show that PyTaskSyn significantly improves task quality compared to baseline techniques and showcases the importance of each specialized agent type in our validation pipeline. Additionally, we conducted user studies using our publicly available web application and show that PyTaskSyn can deliver high-quality programming tasks comparable to expert-designed ones while reducing workload and costs, and being more engaging than programming tasks that are available in online resources. 

**Abstract (ZH)**: 生成式AI正通过自动生成个性化内容和反馈来 transforming 计算机教育。我们研究了其在为学生提供高质量编程任务方面的能力。尽管在任务生成方面取得了令人鼓舞的进展，但AI生成的任务与专家创建的任务之间仍存在质量差距。AI生成的任务可能无法与目标编程概念对齐，可能对学生来说无法理解，或者可能包含错误测试等关键问题。现有研究往往需要人类教师的介入进行验证。我们通过引入PyTaskSyn这一新颖合成技术来应对这些挑战，该技术首先生成一个编程任务，然后决定其是否符合特定的质量标准，从而提交给学生。关键思想是将这一过程分为多个阶段，由使用强生成模型和弱生成模型模拟的专家和学生代理执行。通过广泛的评估，我们展示了PyTaskSyn在任务质量方面显著优于基准技术，并突显了我们验证管道中每种专业化代理类型的重要性。此外，我们使用我们的公有网页应用程序进行了用户研究，并展示了PyTaskSyn可以提供与专家设计的任务相当的高质量编程任务，同时减少工作量和成本，并且比在线资源中可用的编程任务更具吸引力。 

---
# Enhancing Large Language Models through Neuro-Symbolic Integration and Ontological Reasoning 

**Title (ZH)**: 通过神经符号整合与本体推理增强大型语言模型 

**Authors**: Ruslan Idelfonso Magana Vsevolodovna, Marco Monti  

**Link**: [PDF](https://arxiv.org/pdf/2504.07640)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities in natural language processing but suffer from inaccuracies and logical inconsistencies known as hallucinations. This compromises their reliability, especially in domains requiring factual accuracy. We propose a neuro-symbolic approach integrating symbolic ontological reasoning and machine learning methods to enhance the consistency and reliability of LLM outputs. Our workflow utilizes OWL ontologies, a symbolic reasoner (e.g., HermiT) for consistency checking, and a lightweight machine learning model (logistic regression) for mapping natural language statements into logical forms compatible with the ontology. When inconsistencies between LLM outputs and the ontology are detected, the system generates explanatory feedback to guide the LLM towards a corrected, logically coherent response in an iterative refinement loop. We present a working Python prototype demonstrating this pipeline. Experimental results in a defined domain suggest significant improvements in semantic coherence and factual accuracy of LLM outputs, showcasing the potential of combining LLM fluency with the rigor of formal semantics. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理方面展现出了令人印象深刻的性能，但存在因幻觉而导致的不准确性和逻辑不一致问题，这影响了其可靠性，尤其是在需要事实准确性的领域。我们提出了一种神经符号方法，结合符号本体推理和机器学习方法以提升LLM输出的一致性和可靠性。该工作流程利用OWL本体、符号一致性检查器（如HermiT），以及轻量级机器学习模型（逻辑回归），将自然语言陈述映射为与本体兼容的逻辑形式。当检测到LLM输出与本体之间的一致性问题时，系统生成解释性反馈，以指导LLM在迭代改进循环中生成逻辑上连贯的正确响应。我们提供了一个工作Python原型来演示该流程。实验结果表明，在特定领域内，LLM输出的语义连贯性和事实准确性有了显著提升，展示了将LLM的流畅性与形式语义的 rigor 结合的潜力。 

---
# Boosting Universal LLM Reward Design through the Heuristic Reward Observation Space Evolution 

**Title (ZH)**: 通过启发式奖励观察空间进化提升通用大模型奖励设计 

**Authors**: Zen Kit Heng, Zimeng Zhao, Tianhao Wu, Yuanfei Wang, Mingdong Wu, Yangang Wang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.07596)  

**Abstract**: Large Language Models (LLMs) are emerging as promising tools for automated reinforcement learning (RL) reward design, owing to their robust capabilities in commonsense reasoning and code generation. By engaging in dialogues with RL agents, LLMs construct a Reward Observation Space (ROS) by selecting relevant environment states and defining their internal operations. However, existing frameworks have not effectively leveraged historical exploration data or manual task descriptions to iteratively evolve this space. In this paper, we propose a novel heuristic framework that enhances LLM-driven reward design by evolving the ROS through a table-based exploration caching mechanism and a text-code reconciliation strategy. Our framework introduces a state execution table, which tracks the historical usage and success rates of environment states, overcoming the Markovian constraint typically found in LLM dialogues and facilitating more effective exploration. Furthermore, we reconcile user-provided task descriptions with expert-defined success criteria using structured prompts, ensuring alignment in reward design objectives. Comprehensive evaluations on benchmark RL tasks demonstrate the effectiveness and stability of the proposed framework. Code and video demos are available at this http URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）作为自动强化学习（RL）奖励设计的有前途工具，得益于其在常识推理和代码生成方面的强大能力。通过与RL代理进行对话，LLMs构建了一个奖励观察空间（ROS），通过选择相关的环境状态并定义其内部操作。然而，现有框架尚未有效利用历史探索数据或手动任务描述来迭代扩展这一空间。本文提出了一种新的启发式框架，通过基于表格的探索缓存机制和文本-代码协调策略，增强LLM驱动的奖励设计。该框架引入了状态执行表，该表追踪环境状态的历史使用和成功率，克服了LLM对话中通常存在的马尔可夫性限制，从而有利于更有效的探索。此外，我们使用结构化提示将用户提供的任务描述与专家定义的成功标准进行协调，确保奖励设计目标的一致性。在基准RL任务上的全面评估表明了所提框架的有效性和稳定性。代码和视频示例可在以下网址获取。 

---
# A taxonomy of epistemic injustice in the context of AI and the case for generative hermeneutical erasure 

**Title (ZH)**: 人工智能背景下证伪性不公的分类及生成诠释性抹除的论据 

**Authors**: Warmhold Jan Thomas Mollema  

**Link**: [PDF](https://arxiv.org/pdf/2504.07531)  

**Abstract**: Whether related to machine learning models' epistemic opacity, algorithmic classification systems' discriminatory automation of testimonial prejudice, the distortion of human beliefs via the 'hallucinations' of generative AI, the inclusion of the global South in global AI governance, the execution of bureaucratic violence via algorithmic systems, or located in the interaction with conversational artificial agents epistemic injustice related to AI is a growing concern. Based on a proposed general taxonomy of epistemic injustice, this paper first sketches a taxonomy of the types of epistemic injustice in the context of AI, relying on the work of scholars from the fields of philosophy of technology, political philosophy and social epistemology. Secondly, an additional perspective on epistemic injustice in the context of AI: generative hermeneutical erasure. I argue that this injustice that can come about through the application of Large Language Models (LLMs) and contend that generative AI, when being deployed outside of its Western space of conception, can have effects of conceptual erasure, particularly in the epistemic domain, followed by forms of conceptual disruption caused by a mismatch between AI system and the interlocutor in terms of conceptual frameworks. AI systems' 'view from nowhere' epistemically inferiorizes non-Western epistemologies and thereby contributes to the erosion of their epistemic particulars, gradually contributing to hermeneutical erasure. This work's relevance lies in proposal of a taxonomy that allows epistemic injustices to be mapped in the AI domain and the proposal of a novel form of AI-related epistemic injustice. 

**Abstract (ZH)**: 有关人工智能领域知识不公的问题：基于提出的通用知识不公分类，本文首先勾勒出人工智能背景下知识不公的类型 taxonomy，依赖于科技哲学、政治哲学和社会知识论领域学者的研究成果。其次，探讨人工智能背景下知识不公的另一个视角：生成诠释学抹除。本文认为，这种不公可以通过大型语言模型（LLMs）的应用而产生，并且认为当人工智能在西方构想空间之外部署时，可以导致概念抹除，特别是在知识论领域，并导致由于AI系统与对话者在概念框架上不匹配而引发的概念混乱。人工智能系统的“无处不在的视角”会知识论上劣化非西方知识论，并进而导致解释学抹除。本文的成果在于提出了一种使人工智能领域的知识不公得以映射的分类方案，以及提出了一种新的与人工智能相关的知识不公形式。 

---
# Why We Feel: Breaking Boundaries in Emotional Reasoning with Multimodal Large Language Models 

**Title (ZH)**: 为什么我们会感到：利用多模态大型语言模型打破情感推理的边界 

**Authors**: Yuxiang Lin, Jingdong Sun, Zhi-Qi Cheng, Jue Wang, Haomin Liang, Zebang Cheng, Yifei Dong, Jun-Yan He, Xiaojiang Peng, Xian-Sheng Hua  

**Link**: [PDF](https://arxiv.org/pdf/2504.07521)  

**Abstract**: Most existing emotion analysis emphasizes which emotion arises (e.g., happy, sad, angry) but neglects the deeper why. We propose Emotion Interpretation (EI), focusing on causal factors-whether explicit (e.g., observable objects, interpersonal interactions) or implicit (e.g., cultural context, off-screen events)-that drive emotional responses. Unlike traditional emotion recognition, EI tasks require reasoning about triggers instead of mere labeling. To facilitate EI research, we present EIBench, a large-scale benchmark encompassing 1,615 basic EI samples and 50 complex EI samples featuring multifaceted emotions. Each instance demands rationale-based explanations rather than straightforward categorization. We further propose a Coarse-to-Fine Self-Ask (CFSA) annotation pipeline, which guides Vision-Language Models (VLLMs) through iterative question-answer rounds to yield high-quality labels at scale. Extensive evaluations on open-source and proprietary large language models under four experimental settings reveal consistent performance gaps-especially for more intricate scenarios-underscoring EI's potential to enrich empathetic, context-aware AI applications. Our benchmark and methods are publicly available at: this https URL, offering a foundation for advanced multimodal causal analysis and next-generation affective computing. 

**Abstract (ZH)**: 现有的情绪分析大多侧重于识别哪种情绪出现（如快乐、悲伤、愤怒），但忽视了更深层次的“为什么”。我们提出情绪解释（EI），关注引发情绪反应的原因，包括明示的（如可观察的对象、人际互动）和隐含的（如文化背景、画外事件）因素。与传统的 emotion 识别不同，EI 任务要求对触发因素进行推理，而不仅仅是标签化。为了促进 EI 研究，我们提出了 EIBench，这是一个包含 1,615 个基础 EI 样本和 50 个复杂 EI 样本的大规模基准，每个实例都需要基于推理的解释，而不仅仅是简单的分类。我们还提出了一个粗到细自我提问（CFSA）标注流水线，引导视觉-语言模型（VLLMs）通过迭代的问答轮次，大规模生成高质量的标签。在四种实验设置下的开源和专有大型语言模型上的广泛评估显示，尤其是在更复杂的情景下，存在一致的性能差距，这突显了EI在丰富同理心、情境意识人工智能应用方面的潜力。我们的基准和方法可在以下地址获取：this https URL，为高级多模态因果分析和下一代情绪计算奠定基础。 

---
# Enhanced Question-Answering for Skill-based learning using Knowledge-based AI and Generative AI 

**Title (ZH)**: 基于知识的AI和生成式AI增强技能学习的问答方法 

**Authors**: Rahul K. Dass, Rochan H. Madhusudhana, Erin C. Deye, Shashank Verma, Timothy A. Bydlon, Grace Brazil, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2504.07463)  

**Abstract**: Supporting learners' understanding of taught skills in online settings is a longstanding challenge. While exercises and chat-based agents can evaluate understanding in limited contexts, this challenge is magnified when learners seek explanations that delve into procedural knowledge (how things are done) and reasoning (why things happen). We hypothesize that an intelligent agent's ability to understand and explain learners' questions about skills can be significantly enhanced using the TMK (Task-Method-Knowledge) model, a Knowledge-based AI framework. We introduce Ivy, an intelligent agent that leverages an LLM and iterative refinement techniques to generate explanations that embody teleological, causal, and compositional principles. Our initial evaluation demonstrates that this approach goes beyond the typical shallow responses produced by an agent with access to unstructured text, thereby substantially improving the depth and relevance of feedback. This can potentially ensure learners develop a comprehensive understanding of skills crucial for effective problem-solving in online environments. 

**Abstract (ZH)**: 在线环境中支持学习者理解所授技能是一项长期挑战。虽然练习和基于聊天的代理可以在有限的背景下评估理解水平，但当学习者寻求了解程序知识（即事情是如何做的）和推理（即事情为什么会发生）时，这一挑战被进一步放大。我们假设使用基于任务-方法-知识（TMK）模型的知识驱动人工智能框架，智能代理能够理解并解释学习者关于技能的问题的能力可以显著增强。我们引入了Ivy，这是一种利用大规模语言模型和迭代优化技术生成包含目的论、因果性和组成性原则的解释的智能代理。初步评估表明，这种方法超越了仅访问无结构文本的代理所能提供的浅层回答，从而显著提高了反馈的深度和相关性。这有望确保学习者能够全面理解在线环境中有效解决问题所必需的关键技能。 

---
# Enhancing Player Enjoyment with a Two-Tier DRL and LLM-Based Agent System for Fighting Games 

**Title (ZH)**: 基于两层DRL和LLM的代理系统以增强玩家 enjoyment 在格斗游戏中的应用 

**Authors**: Shouren Wang, Zehua Jiang, Fernando Sliva, Sam Earle, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2504.07425)  

**Abstract**: Deep reinforcement learning (DRL) has effectively enhanced gameplay experiences and game design across various game genres. However, few studies on fighting game agents have focused explicitly on enhancing player enjoyment, a critical factor for both developers and players. To address this gap and establish a practical baseline for designing enjoyability-focused agents, we propose a two-tier agent (TTA) system and conducted experiments in the classic fighting game Street Fighter II. The first tier of TTA employs a task-oriented network architecture, modularized reward functions, and hybrid training to produce diverse and skilled DRL agents. In the second tier of TTA, a Large Language Model Hyper-Agent, leveraging players' playing data and feedback, dynamically selects suitable DRL opponents. In addition, we investigate and model several key factors that affect the enjoyability of the opponent. The experiments demonstrate improvements from 64. 36% to 156. 36% in the execution of advanced skills over baseline methods. The trained agents also exhibit distinct game-playing styles. Additionally, we conducted a small-scale user study, and the overall enjoyment in the player's feedback validates the effectiveness of our TTA system. 

**Abstract (ZH)**: 深度强化学习（DRL）在各类游戏类型中有效提升了游戏体验和游戏设计。然而，针对格斗游戏代理的研究大多未明确专注于提升玩家享受，这是开发者和玩家都十分关注的关键因素。为弥补这一不足并建立一个可操作的基准用于设计注重享受的代理，我们提出了一种两级代理（TTA）系统，并在经典格斗游戏《街头霸王II》中进行了实验。TTA的第一级利用任务导向的网络架构、模块化奖励函数和混合训练来生成多样化和高技能的DRL代理。TTA的第二级则利用大规模语言模型超代理，根据玩家的游戏数据和反馈动态选择合适的DRL对手。此外，我们还研究和建模了若干对对手享受度有影响的关键因素。实验结果显示，与基准方法相比，高级技能执行率提高了64.36%至156.36%。训练出的代理还表现出不同的游戏风格。此外，我们还进行了小型用户研究，玩家反馈的整体享受感验证了TTA系统的有效性。 

---
# VCR-Bench: A Comprehensive Evaluation Framework for Video Chain-of-Thought Reasoning 

**Title (ZH)**: VCR-Bench: 视频链式推理的综合评估框架 

**Authors**: Yukun Qi, Yiming Zhao, Yu Zeng, Xikun Bao, Wenxuan Huang, Lin Chen, Zehui Chen, Jie Zhao, Zhongang Qi, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07956)  

**Abstract**: The advancement of Chain-of-Thought (CoT) reasoning has significantly enhanced the capabilities of large language models (LLMs) and large vision-language models (LVLMs). However, a rigorous evaluation framework for video CoT reasoning remains absent. Current video benchmarks fail to adequately assess the reasoning process and expose whether failures stem from deficiencies in perception or reasoning capabilities. Therefore, we introduce VCR-Bench, a novel benchmark designed to comprehensively evaluate LVLMs' Video Chain-of-Thought Reasoning capabilities. VCR-Bench comprises 859 videos spanning a variety of video content and durations, along with 1,034 high-quality question-answer pairs. Each pair is manually annotated with a stepwise CoT rationale, where every step is tagged to indicate its association with the perception or reasoning capabilities. Furthermore, we design seven distinct task dimensions and propose the CoT score to assess the entire CoT process based on the stepwise tagged CoT rationals. Extensive experiments on VCR-Bench highlight substantial limitations in current LVLMs. Even the top-performing model, o1, only achieves a 62.8% CoT score and an 56.7% accuracy, while most models score below 40%. Experiments show most models score lower on perception than reasoning steps, revealing LVLMs' key bottleneck in temporal-spatial information processing for complex video reasoning. A robust positive correlation between the CoT score and accuracy confirms the validity of our evaluation framework and underscores the critical role of CoT reasoning in solving complex video reasoning tasks. We hope VCR-Bench to serve as a standardized evaluation framework and expose the actual drawbacks in complex video reasoning task. 

**Abstract (ZH)**: 基于链式思维的视频推理基准VCR-Bench：全面评估大型vision-language模型的视频链式思维推理能力 

---
# Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge 

**Title (ZH)**: 大规模语言模型中偏见诱致对抗鲁棒性的基准评估：基于LLM的可扩展自动化评估 

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia  

**Link**: [PDF](https://arxiv.org/pdf/2504.07887)  

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过推动机器翻译、总结和对话代理的进步，彻底改变了人工智能。然而，它们在关键社会领域中的不断增加的整合引发了关于嵌入偏见的担忧，这些偏见可以 perpetuate 陈规定型观念并损害公平性。这些偏见源自多个来源，包括训练数据中的历史不平等、语言失衡以及敌对操纵。尽管采取了缓解措施，但近期研究表明，LLMs 仍有可能受到旨在引发偏见响应的敌对攻击。本研究提出了一种可扩展的基准测试框架，以评估LLMs在对抗偏见引诱方面的鲁棒性。我们的方法包括：（i）通过多任务方法系统地测试模型，针对各种社会文化维度的偏见；（ii）通过LLM作为法官的方法量化鲁棒性，自动评估模型响应的安全性得分；（iii）使用脱缰技术研究安全机制的脆弱性。我们的分析检查了小型和大型最新模型中的常见偏见及其对模型安全性的影响。此外，我们评估了针对关键领域（如医疗领域）微调的专业领域模型的安全性。最后，我们发布了与偏见相关的提示数据集CLEAR-Bias，以促进系统的漏洞基准测试。我们的研究结果揭示了模型规模与安全性之间的关键权衡，有助于开发更公平和更鲁棒的语言模型。 

---
# Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs 

**Title (ZH)**: Pangu Ultra： Dense 大型语言模型在 Ascend NPUs 上的极限挑战 

**Authors**: Yichun Yin, Wenyong Huang, Kaikai Song, Yehui Tang, Xueyu Wu, Wei Guo, Peng Guo, Yaoyuan Wang, Xiaojun Meng, Yasheng Wang, Dong Li, Can Chen, Dandan Tu, Yin Li, Fisher Yu, Ruiming Tang, Yunhe Wang, Baojun Wang, Bin Wang, Bo Wang, Boxiao Liu, Changzheng Zhang, Duyu Tang, Fei Mi, Hui Jin, Jiansheng Wei, Jiarui Qin, Jinpeng Li, Jun Zhao, Liqun Deng, Lin Li, Minghui Xu, Naifu Zhang, Nianzu Zheng, Qiang Li, Rongju Ruan, Shengjun Cheng, Tianyu Guo, Wei He, Wei Li, Weiwen Liu, Wulong Liu, Xinyi Dai, Yonghan Dong, Yu Pan, Yue Li, Yufei Wang, Yujun Li, Yunsheng Ni, Zhe Liu, Zhenhe Zhang, Zhicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07866)  

**Abstract**: We present Pangu Ultra, a Large Language Model (LLM) with 135 billion parameters and dense Transformer modules trained on Ascend Neural Processing Units (NPUs). Although the field of LLM has been witnessing unprecedented advances in pushing the scale and capability of LLM in recent years, training such a large-scale model still involves significant optimization and system challenges. To stabilize the training process, we propose depth-scaled sandwich normalization, which effectively eliminates loss spikes during the training process of deep models. We pre-train our model on 13.2 trillion diverse and high-quality tokens and further enhance its reasoning capabilities during post-training. To perform such large-scale training efficiently, we utilize 8,192 Ascend NPUs with a series of system optimizations. Evaluations on multiple diverse benchmarks indicate that Pangu Ultra significantly advances the state-of-the-art capabilities of dense LLMs such as Llama 405B and Mistral Large 2, and even achieves competitive results with DeepSeek-R1, whose sparse model structure contains much more parameters. Our exploration demonstrates that Ascend NPUs are capable of efficiently and effectively training dense models with more than 100 billion parameters. Our model and system will be available for our commercial customers. 

**Abstract (ZH)**: Pangu Ultra: 一个基于昇腾神经处理器训练的1350亿参数密集型Transformer模型 

---
# The KL3M Data Project: Copyright-Clean Training Resources for Large Language Models 

**Title (ZH)**: KL3M数据项目：大型语言模型的版权清洁训练资源 

**Authors**: Michael J Bommarito II, Jillian Bommarito, Daniel Martin Katz  

**Link**: [PDF](https://arxiv.org/pdf/2504.07854)  

**Abstract**: Practically all large language models have been pre-trained on data that is subject to global uncertainty related to copyright infringement and breach of contract. This creates potential risk for users and developers due to this uncertain legal status. The KL3M Data Project directly confronts this critical issue by introducing the largest comprehensive training data pipeline that minimizes risks related to copyright or breach of contract. The foundation of this project is a corpus of over 132 million documents and trillions of tokens spanning 16 different sources that have been verified to meet the strict copyright and licensing protocol detailed herein. We are releasing the entire pipeline, including 1) the source code to acquire and process these documents, 2) the original document formats with associated provenance and metadata, 3) extracted content in a standardized format, 4) pre-tokenized representations of the documents, and 5) various mid- and post-train resources such as question-answer, summarization, conversion, drafting, classification, prediction, and conversational data. All of these resources are freely available to the public on S3, Hugging Face, and GitHub under CC-BY terms. We are committed to continuing this project in furtherance of a more ethical, legal, and sustainable approach to the development and use of AI models. 

**Abstract (ZH)**: 几乎所有大型语言模型都在具有全球性版权侵权和合同违约不确定性的问题数据上进行预训练。这为用户和开发者带来了潜在的法律风险。KL3M数据项目直接应对这一关键问题，引入了最大的综合性训练数据管道，以最小化与版权或合同违约相关的风险。该项目的基础是由13200多万份文件和数十万亿个标记组成的语料库，这些文件来自16个不同来源，已验证符合本文件中详细说明的严格版权和许可协议。我们正在公开整个管道，包括：1）获取和处理这些文件的源代码；2）原始文档格式及其相关来源和元数据；3）标准化格式的提取内容；4）文档的预标记表示；以及5）各种中间和后期训练资源，如问答、总结、转换、起草、分类、预测和对话数据。所有这些资源均在CC-BY许可下在S3、Hugging Face和GitHub上向公众开放。我们致力于继续该项目，以推动AI模型开发和使用更为伦理、合法和可持续的方法。 

---
# Understanding Learner-LLM Chatbot Interactions and the Impact of Prompting Guidelines 

**Title (ZH)**: 理解学习者与大语言模型聊天机器人交互及其提示指南影响 

**Authors**: Cansu Koyuturk, Emily Theophilou, Sabrina Patania, Gregor Donabauer, Andrea Martinenghi, Chiara Antico, Alessia Telari, Alessia Testa, Sathya Bursic, Franca Garzotto, Davinia Hernandez-Leo, Udo Kruschwitz, Davide Taibi, Simona Amenta, Martin Ruskov, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2504.07840)  

**Abstract**: Large Language Models (LLMs) have transformed human-computer interaction by enabling natural language-based communication with AI-powered chatbots. These models are designed to be intuitive and user-friendly, allowing users to articulate requests with minimal effort. However, despite their accessibility, studies reveal that users often struggle with effective prompting, resulting in inefficient responses. Existing research has highlighted both the limitations of LLMs in interpreting vague or poorly structured prompts and the difficulties users face in crafting precise queries. This study investigates learner-AI interactions through an educational experiment in which participants receive structured guidance on effective prompting. We introduce and compare three types of prompting guidelines: a task-specific framework developed through a structured methodology and two baseline approaches. To assess user behavior and prompting efficacy, we analyze a dataset of 642 interactions from 107 users. Using Von NeuMidas, an extended pragmatic annotation schema for LLM interaction analysis, we categorize common prompting errors and identify recurring behavioral patterns. We then evaluate the impact of different guidelines by examining changes in user behavior, adherence to prompting strategies, and the overall quality of AI-generated responses. Our findings provide a deeper understanding of how users engage with LLMs and the role of structured prompting guidance in enhancing AI-assisted communication. By comparing different instructional frameworks, we offer insights into more effective approaches for improving user competency in AI interactions, with implications for AI literacy, chatbot usability, and the design of more responsive AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过启用基于自然语言与AI聊天机器人的通信，已改变了人机交互方式。这些模型设计得直观且用户友好，使用户能够以最小的努力表达请求。然而，尽管其易于访问，研究显示用户在有效提示方面仍常遇到困难，导致效率低下。现有研究强调了LLMs在解释含糊或结构不良的提示方面的局限性，以及用户在构建精确查询方面面临的困难。本研究通过一项教育实验调查学习者与AI的交互，参与者接受有效的提示结构化指导。我们引入并比较了三种提示指南类型：通过结构化方法开发的任务特定框架以及两种基线方法。为了评估用户行为和提示的有效性，我们分析了来自107名用户共642次交互的数据集。利用扩展后的Von NeuMidas注释框架对LLM交互进行分析，我们分类常见的提示错误，并识别反复出现的行为模式。然后，通过评估不同指南的影响来检查用户行为的变化、提示策略的遵守情况以及AI生成响应的总体质量。我们的发现提供了对用户如何与LLMs互动以及结构化提示指导在增强AI辅助通信中的作用的更深层次理解。通过比较不同的指导框架，我们提供了提高用户在AI互动中能力的有效方法的洞见，这对人工智能素养、聊天机器人的可用性以及设计更具反应性的AI系统具有重要意义。 

---
# MOSAIC: Modeling Social AI for Content Dissemination and Regulation in Multi-Agent Simulations 

**Title (ZH)**: MOSAIC: 建模社会AI以实现多 Agents 模拟中的内容传播与调控 

**Authors**: Genglin Liu, Salman Rahman, Elisa Kreiss, Marzyeh Ghassemi, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2504.07830)  

**Abstract**: We present a novel, open-source social network simulation framework, MOSAIC, where generative language agents predict user behaviors such as liking, sharing, and flagging content. This simulation combines LLM agents with a directed social graph to analyze emergent deception behaviors and gain a better understanding of how users determine the veracity of online social content. By constructing user representations from diverse fine-grained personas, our system enables multi-agent simulations that model content dissemination and engagement dynamics at scale. Within this framework, we evaluate three different content moderation strategies with simulated misinformation dissemination, and we find that they not only mitigate the spread of non-factual content but also increase user engagement. In addition, we analyze the trajectories of popular content in our simulations, and explore whether simulation agents' articulated reasoning for their social interactions truly aligns with their collective engagement patterns. We open-source our simulation software to encourage further research within AI and social sciences. 

**Abstract (ZH)**: 我们提出了一种新颖的开源社交媒体网络仿真框架MOSAIC，其中生成型语言代理预测用户行为，如点赞、分享和标记内容。此仿真结合了LLM代理和有向社会图，以分析新兴的欺骗行为，更好地理解用户如何确定在线社交内容的真实性。通过从多样化的细粒度人设构建用户表示，我们的系统能够进行大规模的多代理仿真，模拟内容传播和互动动态。在这种框架内，我们使用模拟的错误信息传播评估了三种不同的内容审核策略，并发现这些策略不仅能减少非事实内容的传播，还能增加用户参与度。此外，我们分析了仿真中流行内容的轨迹，并探索仿真代理在社会互动中表达的理说明确性是否与它们的整体参与模式相一致。我们开源了我们的仿真软件，以促进AI和社会科学领域的进一步研究。 

---
# A System for Comprehensive Assessment of RAG Frameworks 

**Title (ZH)**: 综合评估RAG框架的系统 

**Authors**: Mattia Rengo, Senad Beadini, Domenico Alfano, Roberto Abbruzzese  

**Link**: [PDF](https://arxiv.org/pdf/2504.07803)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as a standard paradigm for enhancing the factual accuracy and contextual relevance of Large Language Models (LLMs) by integrating retrieval mechanisms. However, existing evaluation frameworks fail to provide a holistic black-box approach to assessing RAG systems, especially in real-world deployment scenarios. To address this gap, we introduce SCARF (System for Comprehensive Assessment of RAG Frameworks), a modular and flexible evaluation framework designed to benchmark deployed RAG applications systematically. SCARF provides an end-to-end, black-box evaluation methodology, enabling a limited-effort comparison across diverse RAG frameworks. Our framework supports multiple deployment configurations and facilitates automated testing across vector databases and LLM serving strategies, producing a detailed performance report. Moreover, SCARF integrates practical considerations such as response coherence, providing a scalable and adaptable solution for researchers and industry professionals evaluating RAG applications. Using the REST APIs interface, we demonstrate how SCARF can be applied to real-world scenarios, showcasing its flexibility in assessing different RAG frameworks and configurations. SCARF is available at GitHub repository. 

**Abstract (ZH)**: 全面评估RAG框架的SCARF系统 

---
# FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness 

**Title (ZH)**: FairEval: 基于人格意识评估LLM驱动推荐系统的公平性 

**Authors**: Chandan Kumar Sah, Xiaoli Lian, Tony Xu, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07801)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled their application to recommender systems (RecLLMs), yet concerns remain regarding fairness across demographic and psychological user dimensions. We introduce FairEval, a novel evaluation framework to systematically assess fairness in LLM-based recommendations. FairEval integrates personality traits with eight sensitive demographic attributes,including gender, race, and age, enabling a comprehensive assessment of user-level bias. We evaluate models, including ChatGPT 4o and Gemini 1.5 Flash, on music and movie recommendations. FairEval's fairness metric, PAFS, achieves scores up to 0.9969 for ChatGPT 4o and 0.9997 for Gemini 1.5 Flash, with disparities reaching 34.79 percent. These results highlight the importance of robustness in prompt sensitivity and support more inclusive recommendation systems. 

**Abstract (ZH)**: Recent Advances in Large Language Models (LLMs) Have Enabled Their Application to Recommender Systems (RecLLMs), Yet Concerns Remain Regarding Fairness Across Demographic and Psychological User Dimensions: Introducing FairEval, a Novel Evaluation Framework to Systematically Assess Fairness in LLM-Based Recommendations 

---
# NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark 

**Title (ZH)**: NorEval: 一項挪威語理解與生成評價基准 

**Authors**: Vladislav Mikhailov, Tita Enstad, David Samuel, Hans Christian Farsethås, Andrey Kutuzov, Erik Velldal, Lilja Øvrelid  

**Link**: [PDF](https://arxiv.org/pdf/2504.07749)  

**Abstract**: This paper introduces NorEval, a new and comprehensive evaluation suite for large-scale standardized benchmarking of Norwegian generative language models (LMs). NorEval consists of 24 high-quality human-created datasets -- of which five are created from scratch. In contrast to existing benchmarks for Norwegian, NorEval covers a broad spectrum of task categories targeting Norwegian language understanding and generation, establishes human baselines, and focuses on both of the official written standards of the Norwegian language: Bokmål and Nynorsk. All our datasets and a collection of over 100 human-written prompts are integrated into LM Evaluation Harness, ensuring flexible and reproducible evaluation. We describe the NorEval design and present the results of benchmarking 19 open-source pre-trained and instruction-tuned LMs for Norwegian in various scenarios. Our benchmark, evaluation framework, and annotation materials are publicly available. 

**Abstract (ZH)**: NorEval：一种新的全面的挪威生成语言模型大规模标准化基准评估套件 

---
# PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization 

**Title (ZH)**: PR-攻击：通过对大型语言模型中检索增强生成的协调提示-RAG攻击 via 双层优化 

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07717)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications, e.g., medical question-answering, mathematical sciences, and code generation. However, they also exhibit inherent limitations, such as outdated knowledge and susceptibility to hallucinations. Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address these issues, but it also introduces new vulnerabilities. Recent efforts have focused on the security of RAG-based LLMs, yet existing attack methods face three critical challenges: (1) their effectiveness declines sharply when only a limited number of poisoned texts can be injected into the knowledge database, (2) they lack sufficient stealth, as the attacks are often detectable by anomaly detection systems, which compromises their effectiveness, and (3) they rely on heuristic approaches to generate poisoned texts, lacking formal optimization frameworks and theoretic guarantees, which limits their effectiveness and applicability. To address these issues, we propose coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack that introduces a small number of poisoned texts into the knowledge database while embedding a backdoor trigger within the prompt. When activated, the trigger causes the LLM to generate pre-designed responses to targeted queries, while maintaining normal behavior in other contexts. This ensures both high effectiveness and stealth. We formulate the attack generation process as a bilevel optimization problem leveraging a principled optimization framework to develop optimal poisoned texts and triggers. Extensive experiments across diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving a high attack success rate even with a limited number of poisoned texts and significantly improved stealth compared to existing methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗问答、数学科学和代码生成等广泛应用中展现了卓越的性能。然而，它们也存在固有的局限性，如知识过时和幻觉倾向。检索增强生成（RAG）作为一种潜在的解决方法已经出现，但它也引入了新的脆弱性。最近的研究重点集中在RAG基于的LLMs的安全性上，但现有的攻击方法面临三个关键挑战：（1）当只能注入有限数量的恶意文本到知识数据库中时，它们的有效性急剧下降；（2）它们缺乏足够的隐蔽性，因为攻击常常会被异常检测系统检测到，从而损害了其有效性；（3）它们依赖于启发式方法生成恶意文本，缺乏形式化的优化框架和理论保证，这限制了其有效性和适用性。为解决这些问题，我们提出了一种协调的提示-RAG攻击（PR-attack），这是一种基于优化的攻击方法，能够在知识数据库中注入少量的恶意文本的同时，在提示中嵌入后门触发器。当触发器被激活时，会导致LLM在针对性查询中生成预设的响应，而在其他上下文中保持正常行为。这确保了高度的有效性和隐蔽性。我们将攻击生成过程表述为一个分层优化问题，并利用一个原则性优化框架来开发最优的恶意文本和触发器。在不同LLMs和数据集上的广泛实验表明，PR-攻击方法具有很高的攻击成功率，即使只有有限数量的恶意文本也能实现，并且其隐蔽性比现有方法有了显著提高。 

---
# On the Temporal Question-Answering Capabilities of Large Language Models Over Anonymized Data 

**Title (ZH)**: 大型语言模型在匿名化数据上的时间问答能力 

**Authors**: Alfredo Garrachón Ruiz, Tomás de la Rosa, Daniel Borrajo  

**Link**: [PDF](https://arxiv.org/pdf/2504.07646)  

**Abstract**: The applicability of Large Language Models (LLMs) in temporal reasoning tasks over data that is not present during training is still a field that remains to be explored. In this paper we work on this topic, focusing on structured and semi-structured anonymized data. We not only develop a direct LLM pipeline, but also compare various methodologies and conduct an in-depth analysis. We identified and examined seventeen common temporal reasoning tasks in natural language, focusing on their algorithmic components. To assess LLM performance, we created the \textit{Reasoning and Answering Temporal Ability} dataset (RATA), featuring semi-structured anonymized data to ensure reliance on reasoning rather than on prior knowledge. We compared several methodologies, involving SoTA techniques such as Tree-of-Thought, self-reflexion and code execution, tuned specifically for this scenario. Our results suggest that achieving scalable and reliable solutions requires more than just standalone LLMs, highlighting the need for integrated approaches. 

**Abstract (ZH)**: 大型语言模型在训练数据之外的时间推理任务中的适用性仍是一个待探索的领域。本文专注于此主题，重点研究结构化和半结构化匿名数据。我们不仅开发了一条直接的大型语言模型管道，还比较了多种方法并进行了深入分析。我们识别并研究了自然语言中十七种常见的时间推理任务，专注于它们的算法组件。为了评估大型语言模型的性能，我们创建了“时间推理与回答能力”数据集（RATA），该数据集包含半结构化匿名数据，以确保依赖推理而非先验知识。我们比较了几种方法，涉及当前最佳技术如思维树、自我反思和代码执行，这些技术是专门为这种场景调整的。我们的结果表明，实现可扩展和可靠解决方案不仅仅依赖单独的大型语言模型，强调了集成方法的需求。 

---
# ConceptFormer: Towards Efficient Use of Knowledge-Graph Embeddings in Large Language Models 

**Title (ZH)**: ConceptFormer: 向量利用知识图嵌入以提升大型语言模型的效率 

**Authors**: Joel Barmettler, Abraham Bernstein, Luca Rossetto  

**Link**: [PDF](https://arxiv.org/pdf/2504.07624)  

**Abstract**: Retrieval Augmented Generation (RAG) has enjoyed increased attention in the recent past and recent advancements in Large Language Models (LLMs) have highlighted the importance of integrating world knowledge into these systems. Current RAG methodologies often modify the internal architecture of pre-trained language models (PLMs) or rely on textifying knowledge graphs (KGs), which is inefficient in terms of token usage. This paper introduces ConceptFormer, a new approach to augment LLMs with structured knowledge from KGs, such as Wikidata, without altering their internal structure or relying on textual input of KGs. ConceptFormer operates in the LLM embedding vector space, creating and injecting \emph{concept vectors} that encapsulate the information of the KG nodes directly. Trained in conjunction with a frozen LLM, ConceptFormer generates a comprehensive lookup table that maps KG nodes to their respective concept vectors. The approach aims to enhance the factual recall capabilities of LLMs by enabling them to process these concept vectors natively, thus enriching them with structured world knowledge in an efficient and scalable manner. Our experiments demonstrate that the addition of concept vectors to GPT-2 0.1B substantially increases its factual recall ability (Hit@10) by up to 272\% when tested on sentences from Wikipedia and up to 348\% on synthetically generated sentences. Even injecting only a single concept vector into the prompt increases factual recall ability (Hit@10) by up to 213\% on Wikipedia sentences, significantly outperforming RAG with graph textification while consuming 130x fewer input tokens. 

**Abstract (ZH)**: 概念增强生成（ConceptFormer）：无需修改内部结构的大型语言模型知识增强方法 

---
# AI-Slop to AI-Polish? Aligning Language Models through Edit-Based Writing Rewards and Test-time Computation 

**Title (ZH)**: 从AI粗糙到AI精磨：通过基于编辑的写作奖励和测试时计算实现语言模型对齐 

**Authors**: Tuhin Chakrabarty, Philippe Laban, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07532)  

**Abstract**: AI-generated text is proliferating across domains, from creative writing and journalism to marketing content and scientific articles. Models can follow user-provided instructions to generate coherent and grammatically correct outputs but in this work, we study a more fundamental question: how do we evaluate and improve the writing quality of AI-generated text? Writing quality assessment has received less attention from the community, in part because it is fundamentally subjective and requires expertise. We first introduce the Writing Quality Benchmark (WQ) by consolidating five writing-preference datasets into 4,729 writing quality judgments. Our experiments show that competitive baselines, including state-of-the-art LLMs that excel at reasoning tasks, barely outperform random baselines on WQ. We then train specialized Writing Quality Reward Models (WQRM) of various sizes for writing quality assessment that demonstrate strong generalization on four out-of-distribution test sets and 74% accuracy on the WQ benchmark. To further show WQRM's practical benefits during inference, we leverage additional test-time compute to generate and rank multiple candidate revisions, allowing us to select higher-quality outputs from an initial draft. Human evaluation with 9 experienced writers confirm that WQRM-based selection produces writing samples preferred by experts 66% overall, and 72.2% when the reward gap is larger than 1 point. We release our datasets and models to encourage community engagement with writing quality assessment and development of AI writing systems better aligned with human preferences. 

**Abstract (ZH)**: AI生成文本的质量评估与提升：从主观性出发构建Writing Quality Benchmark（WQ）及高性能奖励模型 

---
# GPT Carry-On: Training Foundation Model for Customization Could Be Simple, Scalable and Affordable 

**Title (ZH)**: GPT 继承：训练定制基础模型可以简单、可扩展且经济 

**Authors**: Jianqiao Wangni  

**Link**: [PDF](https://arxiv.org/pdf/2504.07513)  

**Abstract**: Modern large language foundation models (LLM) have now entered the daily lives of millions of users. We ask a natural question whether it is possible to customize LLM for every user or every task. From system and industrial economy consideration, general continue-training or fine-tuning still require substantial computation and memory of training GPU nodes, whereas most inference nodes under deployment, possibly with lower-end GPUs, are configured to make forward pass fastest possible. We propose a framework to take full advantages of existing LLMs and systems of online service. We train an additional branch of transformer blocks on the final-layer embedding of pretrained LLMs, which is the base, then a carry-on module merge the base models to compose a customized LLM. We can mix multiple layers, or multiple LLMs specialized in different domains such as chat, coding, math, to form a new mixture of LLM that best fit a new task. As the base model don't need to update parameters, we are able to outsource most computation of the training job on inference nodes, and only train a lightweight carry-on on training nodes, where we consume less than 1GB GPU memory to train a 100M carry-on layer on 30B LLM. We tested Qwen and DeepSeek opensourced models for continue-pretraining and got faster loss convergence. We use it to improve solving math questions with extremely small computation and model size, with 1000 data samples of chain-of-thoughts, and as small as 1 MB parameters of two layer layer carry-on, and the results are promising. 

**Abstract (ZH)**: 现代大型语言基础模型（LLM）已深入 Millions of 用户的日常生活。我们自然地提出一个问题：是否可以为每个用户或每个任务定制 LLM？从系统和工业经济的角度考虑，通用的继续训练或微调仍然需要大量的训练 GPU 节点的计算和内存，而部署中的大多数推理节点，可能使用较低端的 GPU，配置为尽可能快地进行前向传播。我们提出了一种框架，充分利用现成的 LLM 和在线服务系统的优势。我们对预训练 LLM 的最终层嵌入增加了一个额外的变压器分支，然后通过继续模块将基础模型组合成一个定制的 LLM。我们可以混合多个层，或者多个专注于不同领域的 LLM（如聊天、编码、数学），以形成最适合新任务的 LLM 新混合体。由于基模型不需要更新参数，我们可以在推理节点上外包大部分训练工作，仅在训练节点上训练一个轻量级的继续模块，我们使用不到 1GB 的 GPU 内存在 30B LLM 上训练一个 100M 参数的继续模块层。我们对 Qwen 和 DeepSeek 开源模型进行继续预训练并获得了更快的损失收敛。我们使用它来改进解决数学问题，仅使用 1000 个带思维链的数据样本和两个层最小 1 MB 参数的轻量级继续模块，结果令人鼓舞。 

---
# LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation 

**Title (ZH)**: LoRI: 多任务低秩适应中跨任务干扰的降低 

**Authors**: Juzheng Zhang, Jiacheng You, Ashwinee Panda, Tom Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2504.07448)  

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as a popular parameter-efficient fine-tuning (PEFT) method for Large Language Models (LLMs), yet it still incurs notable overhead and suffers from parameter interference in multi-task scenarios. We propose LoRA with Reduced Interference (LoRI), a simple yet effective approach that freezes the projection matrices $A$ as random projections and sparsifies the matrices $B$ using task-specific masks. This design substantially reduces the number of trainable parameters while maintaining strong task performance. Moreover, LoRI minimizes cross-task interference in adapter merging by leveraging the orthogonality between adapter subspaces, and supports continual learning by using sparsity to mitigate catastrophic forgetting. Extensive experiments across natural language understanding, mathematical reasoning, code generation, and safety alignment tasks demonstrate that LoRI outperforms full fine-tuning and existing PEFT methods, while using up to 95% fewer trainable parameters than LoRA. In multi-task experiments, LoRI enables effective adapter merging and continual learning with reduced cross-task interference. Code is available at: this https URL 

**Abstract (ZH)**: 基于减少干扰的Low-Rank Adaptation (LoRA) 

---
# LauraTSE: Target Speaker Extraction using Auto-Regressive Decoder-Only Language Models 

**Title (ZH)**: LauraTSE：使用自回归解码器-only语言模型的目标讲话人提取 

**Authors**: Beilong Tang, Bang Zeng, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.07402)  

**Abstract**: We propose LauraTSE, an Auto-Regressive Decoder-Only Language Model for Target Speaker Extraction (TSE) based on the LauraGPT backbone. It employs a small-scale auto-regressive decoder-only language model which takes the continuous representations for both the mixture and the reference speeches and produces the first few layers of the target speech's discrete codec representations. In addition, a one-step encoder-only language model reconstructs the sum of the predicted codec embeddings using both the mixture and the reference information. Our approach achieves superior or comparable performance to existing generative and discriminative TSE models. To the best of our knowledge, LauraTSE is the first single-task TSE model to leverage an auto-regressive decoder-only language model as the backbone. 

**Abstract (ZH)**: 我们提出LauraTSE，一种基于LauraGPT骨干的自回归解码器_only语言模型目标说话人提取（TSE）模型 

---
# Automating quantum feature map design via large language models 

**Title (ZH)**: 通过大型语言模型自动化量子特征映射设计 

**Authors**: Kenya Sakka, Kosuke Mitarai, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2504.07396)  

**Abstract**: Quantum feature maps are a key component of quantum machine learning, encoding classical data into quantum states to exploit the expressive power of high-dimensional Hilbert spaces. Despite their theoretical promise, designing quantum feature maps that offer practical advantages over classical methods remains an open challenge. In this work, we propose an agentic system that autonomously generates, evaluates, and refines quantum feature maps using large language models. The system consists of five component: Generation, Storage, Validation, Evaluation, and Review. Using these components, it iteratively improves quantum feature maps. Experiments on the MNIST dataset show that it can successfully discover and refine feature maps without human intervention. The best feature map generated outperforms existing quantum baselines and achieves competitive accuracy compared to classical kernels across MNIST, Fashion-MNIST, and CIFAR-10. Our approach provides a framework for exploring dataset-adaptive quantum features and highlights the potential of LLM-driven automation in quantum algorithm design. 

**Abstract (ZH)**: 量子特征映射是量子机器学习的关键组成部分，用于将经典数据编码为量子态，从而利用高性能希尔伯特空间的表达能力。尽管具有理论前景，但设计出在实用性上优于经典方法的量子特征映射仍然是一个开放的挑战。在这项工作中，我们提出了一种自主系统，利用大规模语言模型来自主生成、评估和优化量子特征映射。该系统由五个组件构成：生成、存储、验证、评估和审查。利用这些组件，系统能够迭代地改进量子特征映射。在MNIST数据集上的实验显示，该系统能够在无需人类干预的情况下成功发现和优化特征映射。生成的最佳特征映射优于现有量子基线，并在MNIST、Fashion-MNIST和CIFAR-10数据集上实现了与经典核相当的竞争力。我们的方法提供了一种探索数据集自适应量子特征的框架，并突显了语言模型驱动自动化在量子算法设计中的潜力。 

---
# Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression 

**Title (ZH)**: 任务导向的电路量化：利用知识局部化和可解释性进行压缩 

**Authors**: Hanqi Xiao, Yi-Lin Sung, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.07389)  

**Abstract**: Post-training quantization (PTQ) reduces a model's memory footprint by mapping full precision weights into low bit weights without costly retraining, but can degrade its downstream performance especially in low 2- to 3-bit settings. We develop a new mixed-precision PTQ approach, Task-Circuit Quantization (TaCQ), that draws parallels to automated circuit discovery, directly conditioning the quantization process on specific weight circuits -- which we define as sets of weights associated with downstream task performance. These weights are kept as 16-bit weights, while others are quantized, maintaining performance while only adding a marginal memory cost. Specifically, TaCQ contrasts unquantized model weights with a uniformly-quantized model to estimate the expected change in weights due to quantization and uses gradient information to predict the resulting impact on task performance, allowing us to preserve task-specific weights. We compare TaCQ-based quantization to existing mixed-precision quantization methods when conditioning both on general-purpose and task-specific data. Across QA, math reasoning, and text-to-SQL tasks for both Llama-3 and Qwen2.5, we find that TaCQ outperforms baselines using the same calibration data and a lower weight budget, achieving major improvements in the 2 and 3-bit regime. With only 3.1 bits we are able to recover 96% of Llama-3-8B-Instruct's unquantized 16-bit MMLU performance, obtaining a 5.25% absolute improvement over SPQR. We also observe consistently large gains over existing methods in the 2-bit regime, with an average gain of 14.74% over the strongest baseline, SliM-LLM. Moreover, we observe a 7.20% gain without conditioning on specific tasks, showing TaCQ's ability to identify important weights is not limited to task-conditioned settings. 

**Abstract (ZH)**: Task-Circuit Quantization (TaCQ): A New Approach to Post-Training Quantization 

---
# TALE: A Tool-Augmented Framework for Reference-Free Evaluation of Large Language Models 

**Title (ZH)**: TALE：一种工具增强框架，用于无参照评估大规模语言模型 

**Authors**: Sher Badshah, Ali Emami, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2504.07385)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into real-world, autonomous applications, relying on static, pre-annotated references for evaluation poses significant challenges in cost, scalability, and completeness. We propose Tool-Augmented LLM Evaluation (TALE), a framework to assess LLM outputs without predetermined ground-truth answers. Unlike conventional metrics that compare to fixed references or depend solely on LLM-as-a-judge knowledge, TALE employs an agent with tool-access capabilities that actively retrieves and synthesizes external evidence. It iteratively generates web queries, collects information, summarizes findings, and refines subsequent searches through reflection. By shifting away from static references, TALE aligns with free-form question-answering tasks common in real-world scenarios. Experimental results on multiple free-form QA benchmarks show that TALE not only outperforms standard reference-based metrics for measuring response accuracy but also achieves substantial to near-perfect agreement with human evaluations. TALE enhances the reliability of LLM evaluations in real-world, dynamic scenarios without relying on static references. 

**Abstract (ZH)**: Tool-Augmented LLM Evaluation (TALE): 一种评估大规模语言模型输出的框架 

---
# Enhancing Time Series Forecasting via Multi-Level Text Alignment with LLMs 

**Title (ZH)**: 基于LLMs的多层级文本对齐以增强时间序列预测 

**Authors**: Taibiao Zhao, Xiaobing Chen, Mingxuan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.07360)  

**Abstract**: The adaptation of large language models (LLMs) to time series forecasting poses unique challenges, as time series data is continuous in nature, while LLMs operate on discrete tokens. Despite the success of LLMs in natural language processing (NLP) and other structured domains, aligning time series data with language-based representations while maintaining both predictive accuracy and interpretability remains a significant hurdle. Existing methods have attempted to reprogram time series data into text-based forms, but these often fall short in delivering meaningful, interpretable results. In this paper, we propose a multi-level text alignment framework for time series forecasting using LLMs that not only improves prediction accuracy but also enhances the interpretability of time series representations. Our method decomposes time series into trend, seasonal, and residual components, which are then reprogrammed into component-specific text representations. We introduce a multi-level alignment mechanism, where component-specific embeddings are aligned with pre-trained word tokens, enabling more interpretable forecasts. Experiments on multiple datasets demonstrate that our method outperforms state-of-the-art models in accuracy while providing good interpretability. 

**Abstract (ZH)**: 大型语言模型（LLMs）在时间序列预测中的适应性面临着独特挑战，因为时间序列数据具有连续性，而LLMs处理的是离散的令牌。尽管LLMs在自然语言处理（NLP）和其他结构化领域取得了成功，但将时间序列数据与基于语言的表示相结合，同时保持预测准确性和可解释性，仍然是一个重大障碍。已有方法尝试将时间序列数据重新编程为文本形式，但这些方法往往无法提供有意义且可解释的结果。本文提出了一种用于时间序列预测的多层次文本对齐框架，该框架不仅提高了预测准确性，还增强了时间序列表示的可解释性。该方法将时间序列分解为趋势、季节性和残差组件，然后将其重新编程为特定组件的文本表示。我们引入了一种多层次对齐机制，其中特定组件的嵌入与预训练的单词令牌对齐，从而实现更具可解释性的预测。实验结果显示，与现有最先进模型相比，我们的方法在准确性上表现更优，同时具有良好的可解释性。 

---
# Zeus: Zero-shot LLM Instruction for Union Segmentation in Multimodal Medical Imaging 

**Title (ZH)**: Zeus: 零样本LLM指令在多模态医学成像中进行联合分割 

**Authors**: Siyuan Dai, Kai Ye, Guodong Liu, Haoteng Tang, Liang Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07336)  

**Abstract**: Medical image segmentation has achieved remarkable success through the continuous advancement of UNet-based and Transformer-based foundation backbones. However, clinical diagnosis in the real world often requires integrating domain knowledge, especially textual information. Conducting multimodal learning involves visual and text modalities shown as a solution, but collecting paired vision-language datasets is expensive and time-consuming, posing significant challenges. Inspired by the superior ability in numerous cross-modal tasks for Large Language Models (LLMs), we proposed a novel Vision-LLM union framework to address the issues. Specifically, we introduce frozen LLMs for zero-shot instruction generation based on corresponding medical images, imitating the radiology scanning and report generation process. {To better approximate real-world diagnostic processes}, we generate more precise text instruction from multimodal radiology images (e.g., T1-w or T2-w MRI and CT). Based on the impressive ability of semantic understanding and rich knowledge of LLMs. This process emphasizes extracting special features from different modalities and reunion the information for the ultimate clinical diagnostic. With generated text instruction, our proposed union segmentation framework can handle multimodal segmentation without prior collected vision-language datasets. To evaluate our proposed method, we conduct comprehensive experiments with influential baselines, the statistical results and the visualized case study demonstrate the superiority of our novel method.} 

**Abstract (ZH)**: 基于大型语言模型的医疗图像分割 multimodal 学习框架 

---
# PAYADOR: A Minimalist Approach to Grounding Language Models on Structured Data for Interactive Storytelling and Role-playing Games 

**Title (ZH)**: PAYADOR：基于结构化数据为人机互动叙事和角色扮演游戏语言模型接地的极简主义方法 

**Authors**: Santiago Góngora, Luis Chiruzzo, Gonzalo Méndez, Pablo Gervás  

**Link**: [PDF](https://arxiv.org/pdf/2504.07304)  

**Abstract**: Every time an Interactive Storytelling (IS) system gets a player input, it is facing the world-update problem. Classical approaches to this problem consist in mapping that input to known preprogrammed actions, what can severely constrain the free will of the player. When the expected experience has a strong focus on improvisation, like in Role-playing Games (RPGs), this problem is critical. In this paper we present PAYADOR, a different approach that focuses on predicting the outcomes of the actions instead of representing the actions themselves. To implement this approach, we ground a Large Language Model to a minimal representation of the fictional world, obtaining promising results. We make this contribution open-source, so it can be adapted and used for other related research on unleashing the co-creativity power of RPGs. 

**Abstract (ZH)**: 每次交互式叙事(IS)系统接收玩家输入时，都会面临世界更新问题。经典的方法是将输入映射到预先编程的动作，这可能会严重限制玩家的自由意志。当期望的游戏体验强调即兴创作，如角色扮演游戏(RPGs)时，这个问题尤其关键。在本文中，我们提出了PAYADOR，一种不同的方法，侧重于预测动作的结果而不是代表这些动作本身。通过将一个大型语言模型锚定到虚构世界的最小表示，我们取得了令人鼓舞的结果。我们开源了这一贡献，以便他人可以将其 adapted 并用于其他相关研究，以释放RPGs的共创潜力。 

---
# Modeling Response Consistency in Multi-Agent LLM Systems: A Comparative Analysis of Shared and Separate Context Approaches 

**Title (ZH)**: 多智能体大规模语言模型系统中响应一致性建模：共享上下文与独立上下文方法的比较分析 

**Authors**: Tooraj Helmi  

**Link**: [PDF](https://arxiv.org/pdf/2504.07303)  

**Abstract**: Large Language Models (LLMs) are increasingly utilized in multi-agent systems (MAS) to enhance collaborative problem-solving and interactive reasoning. Recent advancements have enabled LLMs to function as autonomous agents capable of understanding complex interactions across multiple topics. However, deploying LLMs in MAS introduces challenges related to context management, response consistency, and scalability, especially when agents must operate under memory limitations and handle noisy inputs. While prior research has explored optimizing context sharing and response latency in LLM-driven MAS, these efforts often focus on either fully centralized or decentralized configurations, each with distinct trade-offs.
In this paper, we develop a probabilistic framework to analyze the impact of shared versus separate context configurations on response consistency and response times in LLM-based MAS. We introduce the Response Consistency Index (RCI) as a metric to evaluate the effects of context limitations, noise, and inter-agent dependencies on system performance. Our approach differs from existing research by focusing on the interplay between memory constraints and noise management, providing insights into optimizing scalability and response times in environments with interdependent topics. Through this analysis, we offer a comprehensive understanding of how different configurations impact the efficiency of LLM-driven multi-agent systems, thereby guiding the design of more robust architectures. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多代理系统（MAS）中的应用增强了协同问题解决和交互式推理的能力。最近的发展使LLMs能够充当自主代理，理解多个主题之间的复杂交互。然而，在MAS中部署LLMs带来了与上下文管理、响应一致性及可扩展性相关的新挑战，特别是在代理必须在内存限制和处理噪声输入的情况下运作时。尽管先前的研究探索了LLM驱动的MAS中的上下文共享和响应延迟优化，这些努力往往集中于完全集中或去中心化配置中的一种，每种配置都有其独特的权衡。

在本文中，我们开发了一种概率框架来分析共享上下文配置与独立上下文配置对响应一致性和响应时间的影响。我们引入响应一致性指数（RCI）作为度量标准，以评估上下文限制、噪声和代理间依赖性对系统性能的影响。我们的方法不同于现有研究，侧重于内存约束和噪声管理之间的互动，为在相互依赖主题的环境中优化可扩展性和响应时间提供了见解。通过这种分析，我们提供了LLM驱动的多代理系统中不同配置影响效率的全面理解，从而指导更 robust架构的设计。 

---
# A Multi-Phase Analysis of Blood Culture Stewardship: Machine Learning Prediction, Expert Recommendation Assessment, and LLM Automation 

**Title (ZH)**: 多阶段分析血液培养 stewardship：机器学习预测、专家建议评估及生成模型自动化 

**Authors**: Fatemeh Amrollahi, Nicholas Marshall, Fateme Nateghi Haredasht, Kameron C Black, Aydin Zahedivash, Manoj V Maddali, Stephen P. Ma, Amy Chang, MD Phar Stanley C Deresinski, Mary Kane Goldstein, Steven M. Asch, Niaz Banaei, Jonathan H Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07278)  

**Abstract**: Blood cultures are often over ordered without clear justification, straining healthcare resources and contributing to inappropriate antibiotic use pressures worsened by the global shortage. In study of 135483 emergency department (ED) blood culture orders, we developed machine learning (ML) models to predict the risk of bacteremia using structured electronic health record (EHR) data and provider notes via a large language model (LLM). The structured models AUC improved from 0.76 to 0.79 with note embeddings and reached 0.81 with added diagnosis codes. Compared to an expert recommendation framework applied by human reviewers and an LLM-based pipeline, our ML approach offered higher specificity without compromising sensitivity. The recommendation framework achieved sensitivity 86%, specificity 57%, while the LLM maintained high sensitivity (96%) but over classified negatives, reducing specificity (16%). These findings demonstrate that ML models integrating structured and unstructured data can outperform consensus recommendations, enhancing diagnostic stewardship beyond existing standards of care. 

**Abstract (ZH)**: 血培养经常缺乏明确依据地被过度开具，从而加重了医疗资源的压力并加剧了由于全球短缺而导致的不适当抗生素使用压力。在对135483份急诊血液培养订单的研究中，我们使用结构化电子健康记录数据和提供者笔记（通过大型语言模型）开发了机器学习模型以预测败血症的风险。结构化模型的AUC从0.76提升至0.79，加入笔记嵌入后达到0.81。与人类审查者应用的专家推荐框架和基于大语言模型的管道相比，我们的机器学习方法在不牺牲灵敏度的情况下提供了更高的特异性。专家推荐框架的灵敏度为86%，特异性为57%，而大语言模型保持了高灵敏度（96%），但过度分类阴性结果，降低了特异性（16%）。这些发现表明，结合结构化和非结构化数据的机器学习模型可以超越现有标准的共识推荐，进一步增强诊断管理。 

---
# SemEval-2025 Task 5: LLMs4Subjects -- LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog 

**Title (ZH)**: SemEval-2025 任务5：LLMs4Subjects ——基于LLM的自动主题标签标注用于国家级技术图书馆的开放访问目录 

**Authors**: Jennifer D'Souza, Sameer Sadruddin, Holger Israel, Mathias Begoin, Diana Slawig  

**Link**: [PDF](https://arxiv.org/pdf/2504.07199)  

**Abstract**: We present SemEval-2025 Task 5: LLMs4Subjects, a shared task on automated subject tagging for scientific and technical records in English and German using the GND taxonomy. Participants developed LLM-based systems to recommend top-k subjects, evaluated through quantitative metrics (precision, recall, F1-score) and qualitative assessments by subject specialists. Results highlight the effectiveness of LLM ensembles, synthetic data generation, and multilingual processing, offering insights into applying LLMs for digital library classification. 

**Abstract (ZH)**: SemEval-2025 任务 5: LLMs4Subjects——一项使用GND分类法对英文和德文科技记录进行自动化学科标签化的共享任务 

---
# HypoEval: Hypothesis-Guided Evaluation for Natural Language Generation 

**Title (ZH)**: HypoEval：基于假设的自然语言生成评估方法 

**Authors**: Mingxuan Li, Hanchen Li, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07174)  

**Abstract**: Large language models (LLMs) have demonstrated great potential for automating the evaluation of natural language generation. Previous frameworks of LLM-as-a-judge fall short in two ways: they either use zero-shot setting without consulting any human input, which leads to low alignment, or fine-tune LLMs on labeled data, which requires a non-trivial number of samples. Moreover, previous methods often provide little reasoning behind automated evaluations. In this paper, we propose HypoEval, Hypothesis-guided Evaluation framework, which first uses a small corpus of human evaluations to generate more detailed rubrics for human judgments and then incorporates a checklist-like approach to combine LLM's assigned scores on each decomposed dimension to acquire overall scores. With only 30 human evaluations, HypoEval achieves state-of-the-art performance in alignment with both human rankings (Spearman correlation) and human scores (Pearson correlation), on average outperforming G-Eval by 11.86% and fine-tuned Llama-3.1-8B-Instruct with at least 3 times more human evaluations by 11.95%. Furthermore, we conduct systematic studies to assess the robustness of HypoEval, highlighting its effectiveness as a reliable and interpretable automated evaluation framework. 

**Abstract (ZH)**: HypoEval：假设导向的评价框架 

---
# Large Language Model (LLM) for Software Security: Code Analysis, Malware Analysis, Reverse Engineering 

**Title (ZH)**: 大型语言模型（LLM）在软件安全中的应用：代码分析、恶意软件分析与逆向工程 

**Authors**: Hamed Jelodar, Samita Bai, Parisa Hamedi, Hesamodin Mohammadian, Roozbeh Razavi-Far, Ali Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2504.07137)  

**Abstract**: Large Language Models (LLMs) have recently emerged as powerful tools in cybersecurity, offering advanced capabilities in malware detection, generation, and real-time monitoring. Numerous studies have explored their application in cybersecurity, demonstrating their effectiveness in identifying novel malware variants, analyzing malicious code structures, and enhancing automated threat analysis. Several transformer-based architectures and LLM-driven models have been proposed to improve malware analysis, leveraging semantic and structural insights to recognize malicious intent more accurately. This study presents a comprehensive review of LLM-based approaches in malware code analysis, summarizing recent advancements, trends, and methodologies. We examine notable scholarly works to map the research landscape, identify key challenges, and highlight emerging innovations in LLM-driven cybersecurity. Additionally, we emphasize the role of static analysis in malware detection, introduce notable datasets and specialized LLM models, and discuss essential datasets supporting automated malware research. This study serves as a valuable resource for researchers and cybersecurity professionals, offering insights into LLM-powered malware detection and defence strategies while outlining future directions for strengthening cybersecurity resilience. 

**Abstract (ZH)**: 大型语言模型（LLMs）最近已经成为网络安全领域强大的工具，提供先进的恶意软件检测、生成和实时监控能力。多项研究探讨了其在网络安全中的应用，展示了其在识别新型恶意软件变种、分析恶意代码结构以及增强自动化威胁分析方面的有效性。提出了多种基于变换器的架构和LLM驱动的模型，以提高恶意软件分析的准确性，利用语义和结构洞察来更准确地识别恶意意图。本研究综述了基于LLM的恶意软件代码分析方法，总结了 recent 进展、趋势和方法论。我们审查了重要学术工作来绘制研究景观，识别关键挑战，并强调LLM驱动的网络安全中的新兴创新。此外，我们强调静态分析在恶意软件检测中的作用，介绍了重要的数据集和专门的LLM模型，并讨论了支持自动化恶意软件研究的重要数据集。本研究为研究人员和网络安全专业人员提供有价值的资源，提供了有关LLM驱动的恶意软件检测和防御策略的见解，并指出了增强网络安全韧性的未来方向。 

---
# ChatBench: From Static Benchmarks to Human-AI Evaluation 

**Title (ZH)**: ChatBench: 从静态基准到-human-AI评估 

**Authors**: Serina Chang, Ashton Anderson, Jake M. Hofman  

**Link**: [PDF](https://arxiv.org/pdf/2504.07114)  

**Abstract**: With the rapid adoption of LLM-based chatbots, there is a pressing need to evaluate what humans and LLMs can achieve together. However, standard benchmarks, such as MMLU, measure LLM capabilities in isolation (i.e., "AI-alone"). Here, we design and conduct a user study to convert MMLU questions into user-AI conversations, by seeding the user with the question and having them carry out a conversation with the LLM to answer their question. We release ChatBench, a new dataset with AI-alone, user-alone, and user-AI data for 396 questions and two LLMs, including 144K answers and 7,336 user-AI conversations. We find that AI-alone accuracy fails to predict user-AI accuracy, with significant differences across multiple subjects (math, physics, and moral reasoning), and we analyze the user-AI conversations to provide insight into how they diverge from AI-alone benchmarks. Finally, we show that fine-tuning a user simulator on a subset of ChatBench improves its ability to estimate user-AI accuracies, increasing correlation on held-out questions by more than 20 points, creating possibilities for scaling interactive evaluation. 

**Abstract (ZH)**: 基于LLM的聊天机器人的快速采用亟需评估人类与LLM合作所能达到的成果。然而，标准基准如MMLU仅评估LLM单独的能力。在这里，我们设计并执行了一项用户研究，将MMLU问题转化为用户-LLM对话，通过向用户提供问题并让他们与LLM进行对话以回答问题。我们发布了ChatBench数据集，包含396个问题和两个LLM的AI-alone、用户-alone及用户-LLM数据，共计144,000个答案和7,336个用户-LLM对话。我们发现AI-alone的准确性无法预测用户-LLM的准确性，并在多个主题（数学、物理和道德推理）上发现了显著差异。我们分析了用户-LLM对话以揭示它们与AI-alone基准的差异。最后，我们显示在ChatBench的一部分数据上 fine-tune 用户模拟器能够提升其估计用户-LLM准确性的能力，在保留问题上显著提高相关性超过20个百分点，为扩展交互式评估提供了可能性。 

---
# OSCAR: Online Soft Compression And Reranking 

**Title (ZH)**: OSCAR: 在线软压缩与重排rankxing 

**Authors**: Maxime Louis, Thibault Formal, Hervé Dejean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2504.07109)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by integrating external knowledge, leading to improved accuracy and relevance. However, scaling RAG pipelines remains computationally expensive as retrieval sizes grow. To address this, we introduce OSCAR, a novel query-dependent online soft compression method that reduces computational overhead while preserving performance. Unlike traditional hard compression methods, which shorten retrieved texts, or soft compression approaches, which map documents to continuous embeddings offline, OSCAR dynamically compresses retrieved information at inference time, eliminating storage overhead and enabling higher compression rates. Additionally, we extend OSCAR to simultaneously perform reranking, further optimizing the efficiency of the RAG pipeline. Our experiments demonstrate state-of-the-art performance with a 2-5x speed-up in inference and minimal to no loss in accuracy for LLMs ranging from 1B to 24B parameters. The models are available at: this https URL. 

**Abstract (ZH)**: 基于检索增强生成的OSCAR方法通过查询依赖的在线软压缩提高大型语言模型的性能，同时减少计算开销并保持高效。 

---
# FG-RAG: Enhancing Query-Focused Summarization with Context-Aware Fine-Grained Graph RAG 

**Title (ZH)**: FG-RAG: 基于上下文感知细粒度图RAG的查询焦点摘要增强 

**Authors**: Yubin Hong, Chaofan Li, Jingyi Zhang, Yingxia Shao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07103)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables large language models to provide more precise and pertinent responses by incorporating external knowledge. In the Query-Focused Summarization (QFS) task, GraphRAG-based approaches have notably enhanced the comprehensiveness and diversity of generated responses. However, existing GraphRAG-based approaches predominantly focus on coarse-grained information summarization without being aware of the specific query, and the retrieved content lacks sufficient contextual information to generate comprehensive responses. To address the deficiencies of current RAG systems, we propose Context-Aware Fine-Grained Graph RAG (FG-RAG) to enhance the performance of the QFS task. FG-RAG employs Context-Aware Entity Expansion in graph retrieval to expand the coverage of retrieved entities in the graph, thus providing enough contextual information for the retrieved content. Furthermore, FG-RAG utilizes Query-Level Fine-Grained Summarization to incorporate fine-grained details during response generation, enhancing query awareness for the generated summarization. Our evaluation demonstrates that FG-RAG outperforms other RAG systems in multiple metrics of comprehensiveness, diversity, and empowerment when handling the QFS task. Our implementation is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）通过集成外部知识，使大型语言模型能够提供更为精确和相关性的响应。在查询导向的摘要（QFS）任务中，基于GraphRAG的方法显著提高了生成摘要的全面性和多样性。然而，现有的基于GraphRAG的方法主要侧重于粗粒度的信息摘要，并未意识到具体的查询需求，检索的内容缺乏足够的上下文信息，从而难以生成全面的摘要。为解决当前RAG系统的不足，我们提出了上下文感知细粒度GraphRAG（FG-RAG）以提高QFS任务的性能。FG-RAG采用上下文感知实体扩展在图检索中的应用，扩展图中检索实体的覆盖范围，从而为检索内容提供足够的上下文信息。此外，FG-RAG采用查询级别细粒度摘要生成，在响应生成过程中融入细粒度细节，增强生成摘要的查询意识。我们的评估表明，FG-RAG在处理QFS任务的多个全面性、多样性和 empowerment指标上优于其他RAG系统。我们的实现可从这个网址获取。 

---
