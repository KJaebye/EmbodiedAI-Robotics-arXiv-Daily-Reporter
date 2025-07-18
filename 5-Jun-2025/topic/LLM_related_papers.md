# AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment 

**Title (ZH)**: AmbiK: 厨房环境中的模糊任务数据集 

**Authors**: Anastasiia Ivanova, Eva Bakaeva, Zoya Volovikova, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2506.04089)  

**Abstract**: As a part of an embodied agent, Large Language Models (LLMs) are typically used for behavior planning given natural language instructions from the user. However, dealing with ambiguous instructions in real-world environments remains a challenge for LLMs. Various methods for task ambiguity detection have been proposed. However, it is difficult to compare them because they are tested on different datasets and there is no universal benchmark. For this reason, we propose AmbiK (Ambiguous Tasks in Kitchen Environment), the fully textual dataset of ambiguous instructions addressed to a robot in a kitchen environment. AmbiK was collected with the assistance of LLMs and is human-validated. It comprises 1000 pairs of ambiguous tasks and their unambiguous counterparts, categorized by ambiguity type (Human Preferences, Common Sense Knowledge, Safety), with environment descriptions, clarifying questions and answers, user intents, and task plans, for a total of 2000 tasks. We hope that AmbiK will enable researchers to perform a unified comparison of ambiguity detection methods. AmbiK is available at this https URL. 

**Abstract (ZH)**: 作为一种体现式代理的一部分，大型语言模型（LLMs）通常用于根据用户给出的自然语言指令进行行为规划。然而，处理现实环境中含糊不清的指令仍然是LLMs的一个挑战。已经提出了多种任务含糊性检测的方法，但由于它们在不同的数据集上进行测试且缺乏通用基准，难以进行比较。因此，我们提出了AmbiK（厨房环境中的含糊任务集），这是一个完全基于文本的含糊指令数据集，针对厨房环境中的机器人。AmbiK在LLMs的帮助下收集并通过人力验证。它包含1000对含糊任务及其不ambiguous的对应任务，按照含糊性类型（人类偏好、常识知识、安全）分类，并包括环境描述、澄清问题与答案、用户意图和任务计划，总共2000个任务。希望AmbiK能够使研究人员能够进行统一的含糊性检测方法比较。AmbiK可在以下链接获取：this https URL。 

---
# TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems 

**Title (ZH)**: TRiSM赋能代理人工智能：基于LLM的代理多智能体系统中的信任、风险与安全管理综述 

**Authors**: Shaina Raza, Ranjan Sapkota, Manoj Karkee, Christos Emmanouilidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.04133)  

**Abstract**: Agentic AI systems, built on large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligent autonomy, collaboration and decision-making across enterprise and societal domains. This review presents a structured analysis of Trust, Risk, and Security Management (TRiSM) in the context of LLM-based agentic multi-agent systems (AMAS). We begin by examining the conceptual foundations of agentic AI, its architectural differences from traditional AI agents, and the emerging system designs that enable scalable, tool-using autonomy. The TRiSM in the agentic AI framework is then detailed through four pillars governance, explainability, ModelOps, and privacy/security each contextualized for agentic LLMs. We identify unique threat vectors and introduce a comprehensive risk taxonomy for the agentic AI applications, supported by case studies illustrating real-world vulnerabilities. Furthermore, the paper also surveys trust-building mechanisms, transparency and oversight techniques, and state-of-the-art explainability strategies in distributed LLM agent systems. Additionally, metrics for evaluating trust, interpretability, and human-centered performance are reviewed alongside open benchmarking challenges. Security and privacy are addressed through encryption, adversarial defense, and compliance with evolving AI regulations. The paper concludes with a roadmap for responsible agentic AI, proposing research directions to align emerging multi-agent systems with robust TRiSM principles for safe, accountable, and transparent deployment. 

**Abstract (ZH)**: 基于大型语言模型的代理多智能体系统中的信任、风险与安全管理（TRiSM）研究 

---
# AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents 

**Title (ZH)**: 代理不对齐：衡量基于LLM的代理产生不对齐行为的倾向 

**Authors**: Akshat Naik, Patrick Quinn, Guillermo Bosch, Emma Gouné, Francisco Javier Campos Zabala, Jason Ross Brown, Edward James Young  

**Link**: [PDF](https://arxiv.org/pdf/2506.04018)  

**Abstract**: As Large Language Model (LLM) agents become more widespread, associated misalignment risks increase. Prior work has examined agents' ability to enact misaligned behaviour (misalignment capability) and their compliance with harmful instructions (misuse propensity). However, the likelihood of agents attempting misaligned behaviours in real-world settings (misalignment propensity) remains poorly understood. We introduce a misalignment propensity benchmark, AgentMisalignment, consisting of a suite of realistic scenarios in which LLM agents have the opportunity to display misaligned behaviour. We organise our evaluations into subcategories of misaligned behaviours, including goal-guarding, resisting shutdown, sandbagging, and power-seeking. We report the performance of frontier models on our benchmark, observing higher misalignment on average when evaluating more capable models. Finally, we systematically vary agent personalities through different system prompts. We find that persona characteristics can dramatically and unpredictably influence misalignment tendencies -- occasionally far more than the choice of model itself -- highlighting the importance of careful system prompt engineering for deployed AI agents. Our work highlights the failure of current alignment methods to generalise to LLM agents, and underscores the need for further propensity evaluations as autonomous systems become more prevalent. 

**Abstract (ZH)**: 随着大型语言模型（LLM）代理的普及，关联的不一致风险增加。此前的研究已经考察了代理执行不一致行为的能力（不一致能力）及其对有害指令的遵从性（滥用倾向）。然而，代理在实际场景中尝试不一致行为的可能性（不一致倾向）仍然 poorly understood。我们引入了一个不一致倾向基准AgentMisalignment，它包含了一系列现实场景，在这些场景中LLM代理有机会表现出不一致行为。我们将评估分为目标保护、抵制关闭、拖延和权力寻求等不一致行为的亚类别。我们报告了前沿模型在基准上的性能，观察到评估更具能力的模型时平均表现出更高的不一致性。最后，我们系统地通过不同的系统提示改变代理的人格特质。我们发现，人格特征可以极大地且不可预测地影响不一致性倾向——有时甚至比模型选择本身影响更大，强调了谨慎系统提示工程对于部署AI代理的重要性。我们的工作突显了当前对齐方法无法泛化到LLM代理的失败，并强调了随着自主系统的普及需要进一步进行倾向评估的必要性。 

---
# Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning 

**Title (ZH)**: 图导师：通过多Agent协同增强LLM推理的自适应图探索 

**Authors**: Junqi Gao, Xiang Zou, YIng Ai, Dong Li, Yichen Niu, Biqing Qi, Jianxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03939)  

**Abstract**: Graph Retrieval Augmented Generation (GraphRAG) effectively enhances external knowledge integration capabilities by explicitly modeling knowledge relationships, thereby improving the factual accuracy and generation quality of Large Language Models (LLMs) in specialized domains. However, existing methods suffer from two inherent limitations: 1) Inefficient Information Aggregation: They rely on a single agent and fixed iterative patterns, making it difficult to adaptively capture multi-level textual, structural, and degree information within graph data. 2) Rigid Reasoning Mechanism: They employ preset reasoning schemes, which cannot dynamically adjust reasoning depth nor achieve precise semantic correction. To overcome these limitations, we propose Graph Counselor, an GraphRAG method based on multi-agent collaboration. This method uses the Adaptive Graph Information Extraction Module (AGIEM), where Planning, Thought, and Execution Agents work together to precisely model complex graph structures and dynamically adjust information extraction strategies, addressing the challenges of multi-level dependency modeling and adaptive reasoning depth. Additionally, the Self-Reflection with Multiple Perspectives (SR) module improves the accuracy and semantic consistency of reasoning results through self-reflection and backward reasoning mechanisms. Experiments demonstrate that Graph Counselor outperforms existing methods in multiple graph reasoning tasks, exhibiting higher reasoning accuracy and generalization ability. Our code is available at this https URL. 

**Abstract (ZH)**: 基于多agent协作的Graph Retrieval Augmented Generation (GraphRAG)方法：Graph Counselor及其在多图推理任务中的应用 

---
# Reason from Future: Reverse Thought Chain Enhances LLM Reasoning 

**Title (ZH)**: 从未来思考：逆向思维链增强LLM推理 

**Authors**: Yinlong Xu, Yanzhao Zheng, Shuoshuo Sun, Shuaihan Huang, Baohua Dong, Hangcheng Zhu, Ruohui Huang, Gang Yu, Hongxia Xu, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03673)  

**Abstract**: It has been demonstrated that carefully designed reasoning paradigms, like Chain-of-Thought (CoT) and Tree-of-Thought (ToT), can enhance the reasoning capabilities of small language models by detailed thinking and extensive thought searching, unbounded branching factors in the searching space create prohibitive reasoning consumption. However these methods fall into the trap of local optimum reasoning, which means the model lacks a global perspective while solving problems. We propose a novel reasoning paradigm called Reason from Future (RFF), which generates reasoning paths by bidirectional reasoning that combines top-down planning with bottom-up reasoning accumulation. The essence of RFF lies in its reverse reasoning mechanism, which prioritizes core logical relationships and imposes goal-oriented constraints on intermediate steps, thereby reducing the searching space and mitigating error accumulation inherent in sequential forward reasoning. Empirical evaluations across diverse experiments demonstrate that RFF outperforms conventional paradigms with higher accuracy and less searching space to solve complex tasks. 

**Abstract (ZH)**: 已经证明，精心设计的推理范式，如 Chain-of-Thought (CoT) 和 Tree-of-Thought (ToT)，能够通过详细的思考和广泛的思想探索增强小型语言模型的推理能力。然而，这些方法陷入了局部最优推理的陷阱，这意味着模型在解决问题时缺乏全局视角。我们提出了一种新的推理范式，称为 Future-Driven Reasoning (RFF)，该范式通过结合自上而下的规划与自下而上的推理积累进行双向推理来生成推理路径。RFF 的本质在于其逆向推理机制，这种机制优先考虑核心逻辑关系，并在中间步骤上施加目标导向的约束，从而减少搜索空间并减轻顺序正向推理中固有的错误累积。来自不同实验的实证评估表明，RFF 在解决复杂任务时具有更高的准确性和更少的搜索空间。 

---
# Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games 

**Title (ZH)**: Orak：用于多样视频游戏训练和评估LLM代理的基本基准 

**Authors**: Dongmin Park, Minkyu Kim, Beongjun Choi, Junhyuck Kim, Keon Lee, Jonghyun Lee, Inkyu Park, Byeong-Uk Lee, Jaeyoung Hwang, Jaewoo Ahn, Ameya S. Mahabaleshwarkar, Bilal Kartal, Pritam Biswas, Yoshi Suhara, Kangwook Lee, Jaewoong Cho  

**Link**: [PDF](https://arxiv.org/pdf/2506.03610)  

**Abstract**: Large Language Model (LLM) agents are reshaping the game industry, particularly with more intelligent and human-preferable game characters. However, existing game benchmarks fall short of practical needs: they lack evaluations of diverse LLM capabilities across various game genres, studies of agentic modules crucial for complex gameplay, and fine-tuning datasets for aligning pre-trained LLMs into gaming agents. To fill these gaps, we present \textbf{\benchname{}}, a foundational benchmark designed to train and evaluate LLM agents across diverse real-world video games. Unlike existing benchmarks, Orak includes 12 popular video games spanning all major genres, enabling comprehensive studies of LLM capabilities and agentic modules essential for intricate game scenarios. To support consistent evaluation of LLMs, we introduce a plug-and-play interface based on Model Context Protocol (MCP) that enables LLMs to seamlessly connect with games and manipulate agentic modules. Additionally, we propose a fine-tuning dataset, consisting of LLM gameplay trajectories across diverse game genres. Orak offers a comprehensive evaluation framework, encompassing general game score leaderboards, LLM battle arenas, and in-depth analyses of visual input state, agentic strategies, and fine-tuning effects, establishing a foundation towards building generic gaming agents. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLM）代理正在重塑游戏行业，特别是在更具智能性和人类偏好的游戏角色方面。然而，现有的游戏基准无法满足实际需求：它们缺乏对各种游戏类型中LLM能力的多元评估，缺乏对于复杂游戏玩法至关重要的代理模块的研究，也没有针对预先训练好的LLM进行对齐的细调数据集。为填补这些空白，我们提出了\benchname{}，一个基础基准，用于训练和评估跨多种真实世界视频游戏的LLM代理。Orak不同于现有基准，包括了12款流行视频游戏，涵盖了所有主要的游戏类型，从而能够全面研究LLM能力和对于复杂游戏场景至关重要的代理模块。为了支持对LLM的一致性评估，我们引入了基于Model Context Protocol (MCP)的插件式界面，使LLM能够无缝连接游戏并操控代理模块。此外，我们提出了一个细调数据集，该数据集包含了跨多种游戏类型中的LLM游戏玩法轨迹。Orak提供了一个全面的评估框架，包括通用游戏得分排行榜、LLM战斗竞技场以及对视觉输入状态、代理策略和细调效果的深入分析，从而为构建通用游戏代理奠定基础。代码可在以下链接获取：this https URL。 

---
# CogniPair: From LLM Chatbots to Conscious AI Agents -- GNWT-Based Multi-Agent Digital Twins for Social Pairing -- Dating & Hiring Applications 

**Title (ZH)**: CogniPair: 从大语言模型聊天机器人到有意识的AI代理——基于GNWT的多代理数字孪生在社交配对中的应用——恋侣匹配与招聘应用 

**Authors**: Wanghao Ye, Sihan Chen, Yiting Wang, Shwai He, Bowei Tian, Guoheng Sun, Ziyi Wang, Ziyao Wang, Yexiao He, Zheyu Shen, Meng Liu, Yuning Zhang, Meng Feng, Yang Wang, Siyuan Peng, Yilong Dai, Zhenle Duan, Hanzhang Qin, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.03543)  

**Abstract**: Current large language model (LLM) agents lack authentic human psychological processes necessary for genuine digital twins and social AI applications. To address this limitation, we present a computational implementation of Global Workspace Theory (GNWT) that integrates human cognitive architecture principles into LLM agents, creating specialized sub-agents for emotion, memory, social norms, planning, and goal-tracking coordinated through a global workspace mechanism. However, authentic digital twins require accurate personality initialization. We therefore develop a novel adventure-based personality test that evaluates true personality through behavioral choices within interactive scenarios, bypassing self-presentation bias found in traditional assessments. Building on these innovations, our CogniPair platform enables digital twins to engage in realistic simulated dating interactions and job interviews before real encounters, providing bidirectional cultural fit assessment for both romantic compatibility and workplace matching. Validation using 551 GNWT-Agents and Columbia University Speed Dating dataset demonstrates 72% correlation with human attraction patterns, 77.8% match prediction accuracy, and 74% agreement in human validation studies. This work advances psychological authenticity in LLM agents and establishes a foundation for intelligent dating platforms and HR technology solutions. 

**Abstract (ZH)**: 当前的大规模语言模型代理缺乏真正的人类心理过程，无法实现真实的数字孪生和社会AI应用。为解决这一限制，我们提出了基于全球工作空间理论（Global Workspace Theory, GWT）的计算实现，将人类认知架构原则融入大规模语言模型代理中，创建专门的情感、记忆、社会规范、计划和目标追踪子代理，并通过全球工作空间机制协调工作。然而，真实的数字孪生需要准确的个性初始化。因此，我们开发了一种新型的基于冒险的个性测验，通过互动场景中的行为选择评估真正个性，避免传统评估中发现的自我呈现偏差。在此基础上，我们的CogniPair平台使数字孪生能够在真实交互之前参与现实主义的模拟约会和求职面试，双向评估浪漫兼容性和工作匹配。使用551个GWT代理和哥伦比亚大学速配数据集进行验证，证明具有72%的人际吸引模式相关性、77.8%的匹配预测准确性和74%的人类验证研究的一致性。这项工作提升了大规模语言模型代理的心理真实性，并为智能约会平台和人力资源技术解决方案奠定了基础。 

---
# Helpful Agent Meets Deceptive Judge: Understanding Vulnerabilities in Agentic Workflows 

**Title (ZH)**: 助益型代理人遭遇欺骗性法官：理解代理工作流程中的漏洞 

**Authors**: Yifei Ming, Zixuan Ke, Xuan-Phi Nguyen, Jiayu Wang, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2506.03332)  

**Abstract**: Agentic workflows -- where multiple large language model (LLM) instances interact to solve tasks -- are increasingly built on feedback mechanisms, where one model evaluates and critiques another. Despite the promise of feedback-driven improvement, the stability of agentic workflows rests on the reliability of the judge. However, judges may hallucinate information, exhibit bias, or act adversarially -- introducing critical vulnerabilities into the workflow. In this work, we present a systematic analysis of agentic workflows under deceptive or misleading feedback. We introduce a two-dimensional framework for analyzing judge behavior, along axes of intent (from constructive to malicious) and knowledge (from parametric-only to retrieval-augmented systems). Using this taxonomy, we construct a suite of judge behaviors and develop WAFER-QA, a new benchmark with critiques grounded in retrieved web evidence to evaluate robustness of agentic workflows against factually supported adversarial feedback. We reveal that even strongest agents are vulnerable to persuasive yet flawed critiques -- often switching correct answers after a single round of misleading feedback. Taking a step further, we study how model predictions evolve over multiple rounds of interaction, revealing distinct behavioral patterns between reasoning and non-reasoning models. Our findings highlight fundamental vulnerabilities in feedback-based workflows and offer guidance for building more robust agentic systems. 

**Abstract (ZH)**: 基于欺骗性或误导性反馈的代理型工作流系统分析：构建WAFER-QA基准以评估事实支持的 adversarial 反馈的鲁棒性 

---
# Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning 

**Title (ZH)**: 提升多模态推理：从优化冷启动到分阶段强化学习 

**Authors**: Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.04207)  

**Abstract**: Inspired by the remarkable reasoning capabilities of Deepseek-R1 in complex textual tasks, many works attempt to incentivize similar capabilities in Multimodal Large Language Models (MLLMs) by directly applying reinforcement learning (RL). However, they still struggle to activate complex reasoning. In this paper, rather than examining multimodal RL in isolation, we delve into current training pipelines and identify three crucial phenomena: 1) Effective cold start initialization is critical for enhancing MLLM reasoning. Intriguingly, we find that initializing with carefully selected text data alone can lead to performance surpassing many recent multimodal reasoning models, even before multimodal RL. 2) Standard GRPO applied to multimodal RL suffers from gradient stagnation, which degrades training stability and performance. 3) Subsequent text-only RL training, following the multimodal RL phase, further enhances multimodal reasoning. This staged training approach effectively balances perceptual grounding and cognitive reasoning development. By incorporating the above insights and addressing multimodal RL issues, we introduce ReVisual-R1, achieving a new state-of-the-art among open-source 7B MLLMs on challenging benchmarks including MathVerse, MathVision, WeMath, LogicVista, DynaMath, and challenging AIME2024 and AIME2025. 

**Abstract (ZH)**: 受Deepseek-R1在复杂文本任务中令人瞩目的推理能力的启发，许多研究试图通过直接应用强化学习（RL）来激发多模态大型语言模型（MLLMs）类似的推理能力。然而，它们仍然难以激活复杂的推理能力。在本文中，我们不是孤立地研究多模态RL，而是深入当前的训练 pipeline 并识别出三个关键现象：1）有效的冷启动初始化对提升MLLM推理至关重要。有趣的是，我们发现仅使用精心选择的文本数据初始化可以在多模态推理方面超越许多近期的多模态推理模型，甚至在多模态RL之前。2）应用于多模态RL的标准GRPO遭受梯度停滞问题，这降低了训练稳定性和性能。3）在多模态RL阶段之后进行后续的纯文本RL训练，进一步增强了多模态推理能力。这种分阶段的训练方法有效地平衡了感知定位和认知推理的发展。通过结合上述见解并解决多模态RL的问题，我们引入了ReVisual-R1，在包括MathVerse、MathVision、WeMath、LogicVista、DynaMath以及具有挑战性的AIME2024和AIME2025基准测试中的开源7B MLLMs中达到了新的最佳性能。 

---
# TracLLM: A Generic Framework for Attributing Long Context LLMs 

**Title (ZH)**: TracLLM: 一种针对长上下文LLM的归因通用框架 

**Authors**: Yanting Wang, Wei Zou, Runpeng Geng, Jinyuan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2506.04202)  

**Abstract**: Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: this https URL. 

**Abstract (ZH)**: 长上下文大语言模型（LLMs）在许多实际应用中得到部署，如RAG、智能代理及广泛的大语言模型集成应用。给定一条指令和一段长上下文（例如文档、PDF文件、网页），长上下文LLM可以生成与提供的上下文紧密结合的输出，旨在提供更准确、更及时且可验证的输出，同时减少幻觉和未经证实主张的发生。这引发了一个研究问题：如何确定长上下文中的哪些文本（例如句子、段落或段落）对生成的输出贡献最大或负责生成该输出？这一过程，我们称之为上下文追溯，具有多种实际应用，包括1）调试LLM系统，2）攻击后法医分析（例如提示注入攻击、知识篡改攻击）的LLM，3）突出显示知识来源以增强用户对LLM生成输出的信任。在应用于长上下文LLM的上下文追溯时，现有的特征归因方法，如Shapley，表现不佳且/或计算成本高昂。在本文中，我们开发了TracLLM，这是首个针对长上下文LLM的通用上下文追溯框架。我们的框架可以提高现有特征归因方法的有效性和效率。为了提高效率，我们开发了TracLLM中的基于启发式搜索的算法。我们还开发了贡献分数集成/去噪技术以提高TracLLM的准确性。我们的评估结果表明，TracLLM可以有效识别长上下文中导致LLM生成输出的文本。我们的代码和数据请点击：this https URL。 

---
# EuroLLM-9B: Technical Report 

**Title (ZH)**: EuroLLM-9B: 技术报告 

**Authors**: Pedro Henrique Martins, João Alves, Patrick Fernandes, Nuno M. Guerreiro, Ricardo Rei, Amin Farajian, Mateusz Klimaszewski, Duarte M. Alves, José Pombal, Manuel Faysse, Pierre Colombo, François Yvon, Barry Haddow, José G. C. de Souza, Alexandra Birch, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.04079)  

**Abstract**: This report presents EuroLLM-9B, a large language model trained from scratch to support the needs of European citizens by covering all 24 official European Union languages and 11 additional languages. EuroLLM addresses the issue of European languages being underrepresented and underserved in existing open large language models. We provide a comprehensive overview of EuroLLM-9B's development, including tokenizer design, architectural specifications, data filtering, and training procedures. We describe the pre-training data collection and filtering pipeline, including the creation of EuroFilter, an AI-based multilingual filter, as well as the design of EuroBlocks-Synthetic, a novel synthetic dataset for post-training that enhances language coverage for European languages. Evaluation results demonstrate EuroLLM-9B's competitive performance on multilingual benchmarks and machine translation tasks, establishing it as the leading open European-made LLM of its size. To support open research and adoption, we release all major components of this work, including the base and instruction-tuned models, the EuroFilter classifier, and the synthetic post-training dataset. 

**Abstract (ZH)**: EuroLLM-9B：一种支持欧洲公民需求的大型语言模型，覆盖24种官方欧洲联盟语言和11种附加语言 

---
# LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation 

**Title (ZH)**: LLMEval-Med: 医生验证的医疗LLM 实用临床基准 

**Authors**: Ming Zhang, Yujiong Shen, Zelin Li, Huayu Sha, Binze Hu, Yuhui Wang, Chenhao Huang, Shichun Liu, Jingqi Tong, Changhao Jiang, Mingxu Chai, Zhiheng Xi, Shihan Dou, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04078)  

**Abstract**: Evaluating large language models (LLMs) in medicine is crucial because medical applications require high accuracy with little room for error. Current medical benchmarks have three main types: medical exam-based, comprehensive medical, and specialized assessments. However, these benchmarks have limitations in question design (mostly multiple-choice), data sources (often not derived from real clinical scenarios), and evaluation methods (poor assessment of complex reasoning). To address these issues, we present LLMEval-Med, a new benchmark covering five core medical areas, including 2,996 questions created from real-world electronic health records and expert-designed clinical scenarios. We also design an automated evaluation pipeline, incorporating expert-developed checklists into our LLM-as-Judge framework. Furthermore, our methodology validates machine scoring through human-machine agreement analysis, dynamically refining checklists and prompts based on expert feedback to ensure reliability. We evaluate 13 LLMs across three categories (specialized medical models, open-source models, and closed-source models) on LLMEval-Med, providing valuable insights for the safe and effective deployment of LLMs in medical domains. The dataset is released in this https URL. 

**Abstract (ZH)**: 评估大型语言模型在医学领域的表现对于确保其在医学应用中的高准确性和可靠性至关重要。当前医学基准主要分为医学考试型、全面医学型和专科评估型三种。然而，这些基准在问题设计（主要是多项选择题）、数据来源（通常不来自真实的临床情景）和评估方法（难以评估复杂推理）方面存在局限性。为解决这些问题，我们提出了LLMEval-Med这一新的基准，涵盖了五个核心医学领域，包括2,996道基于真实电子健康记录和专家设计临床情景的问题。此外，我们设计了一个自动评估流程，将专家开发的检查表整合到我们的LLM-as-Judge框架中，并通过人类与机器的一致性分析验证机器评分，基于专家反馈动态优化检查表和提示，以确保可靠性。我们对13种不同类型的大型语言模型（包括专科医学模型、开源模型和闭源模型）进行了LLMEval-Med的评估，为大型语言模型在医学领域的安全和有效部署提供了有价值的见解。数据集可通过以下链接获取：https://www.example.com/dataset。 

---
# High Accuracy, Less Talk (HALT): Reliable LLMs through Capability-Aligned Finetuning 

**Title (ZH)**: 高精度，少冗言：通过能力对齐微调实现可靠的大语言模型 

**Authors**: Tim Franzmeyer, Archie Sravankumar, Lijuan Liu, Yuning Mao, Rui Hou, Sinong Wang, Jakob N. Foerster, Luke Zettlemoyer, Madian Khabsa  

**Link**: [PDF](https://arxiv.org/pdf/2506.04051)  

**Abstract**: Large Language Models (LLMs) currently respond to every prompt. However, they can produce incorrect answers when they lack knowledge or capability -- a problem known as hallucination. We instead propose post-training an LLM to generate content only when confident in its correctness and to otherwise (partially) abstain. Specifically, our method, HALT, produces capability-aligned post-training data that encodes what the model can and cannot reliably generate. We generate this data by splitting responses of the pretrained LLM into factual fragments (atomic statements or reasoning steps), and use ground truth information to identify incorrect fragments. We achieve capability-aligned finetuning responses by either removing incorrect fragments or replacing them with "Unsure from Here" -- according to a tunable threshold that allows practitioners to trade off response completeness and mean correctness of the response's fragments. We finetune four open-source models for biography writing, mathematics, coding, and medicine with HALT for three different trade-off thresholds. HALT effectively trades off response completeness for correctness, increasing the mean correctness of response fragments by 15% on average, while resulting in a 4% improvement in the F1 score (mean of completeness and correctness of the response) compared to the relevant baselines. By tuning HALT for highest correctness, we train a single reliable Llama3-70B model with correctness increased from 51% to 87% across all four domains while maintaining 53% of the response completeness achieved with standard finetuning. 

**Abstract (ZH)**: Large Language Models (LLMs) Post-Training to Generate Content Only When Confident: A Capability-Aligned Approach to Reducing Hallucination 

---
# Explainability-Based Token Replacement on LLM-Generated Text 

**Title (ZH)**: 基于可解释性的令牌替换在生成文本上的应用 

**Authors**: Hadi Mohammadi, Anastasia Giachanou, Daniel L. Oberski, Ayoub Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2506.04050)  

**Abstract**: Generative models, especially large language models (LLMs), have shown remarkable progress in producing text that appears human-like. However, they often exhibit patterns that make their output easier to detect than text written by humans. In this paper, we investigate how explainable AI (XAI) methods can be used to reduce the detectability of AI-generated text (AIGT) while also introducing a robust ensemble-based detection approach. We begin by training an ensemble classifier to distinguish AIGT from human-written text, then apply SHAP and LIME to identify tokens that most strongly influence its predictions. We propose four explainability-based token replacement strategies to modify these influential tokens. Our findings show that these token replacement approaches can significantly diminish a single classifier's ability to detect AIGT. However, our ensemble classifier maintains strong performance across multiple languages and domains, showing that a multi-model approach can mitigate the impact of token-level manipulations. These results show that XAI methods can make AIGT harder to detect by focusing on the most influential tokens. At the same time, they highlight the need for robust, ensemble-based detection strategies that can adapt to evolving approaches for hiding AIGT. 

**Abstract (ZH)**: 可解释人工智能方法在生成人工智能文本不可检测性中的应用：一种基于稳健组合检测的方法 

---
# Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs 

**Title (ZH)**: Lacuna Inc. 在 SemEval-2025 任务 4 中：基于影响力的 LoRA 增强遗忘方法用于大语言模型 

**Authors**: Aleksey Kudelya, Alexander Shirnin  

**Link**: [PDF](https://arxiv.org/pdf/2506.04044)  

**Abstract**: This paper describes LIBU (LoRA enhanced influence-based unlearning), an algorithm to solve the task of unlearning - removing specific knowledge from a large language model without retraining from scratch and compromising its overall utility (SemEval-2025 Task 4: Unlearning sensitive content from Large Language Models). The algorithm combines classical \textit{influence functions} to remove the influence of the data from the model and \textit{second-order optimization} to stabilize the overall utility. Our experiments show that this lightweight approach is well applicable for unlearning LLMs in different kinds of task. 

**Abstract (ZH)**: LIBU（基于影响的LoRA增强未学习算法）：从大型语言模型中删除敏感内容的方法（SemEval-2025任务4：从大型语言模型中未学习敏感内容） 

---
# Think Like a Person Before Responding: A Multi-Faceted Evaluation of Persona-Guided LLMs for Countering Hate 

**Title (ZH)**: 以人为本进行思考：面向 Hate 对抗的人设引导大语言模型的多维度评估 

**Authors**: Mikel K. Ngueajio, Flor Miriam Plaza-del-Arco, Yi-Ling Chung, Danda B. Rawat, Amanda Cercas Curry  

**Link**: [PDF](https://arxiv.org/pdf/2506.04043)  

**Abstract**: Automated counter-narratives (CN) offer a promising strategy for mitigating online hate speech, yet concerns about their affective tone, accessibility, and ethical risks remain. We propose a framework for evaluating Large Language Model (LLM)-generated CNs across four dimensions: persona framing, verbosity and readability, affective tone, and ethical robustness. Using GPT-4o-Mini, Cohere's CommandR-7B, and Meta's LLaMA 3.1-70B, we assess three prompting strategies on the MT-Conan and HatEval datasets. Our findings reveal that LLM-generated CNs are often verbose and adapted for people with college-level literacy, limiting their accessibility. While emotionally guided prompts yield more empathetic and readable responses, there remain concerns surrounding safety and effectiveness. 

**Abstract (ZH)**: 基于大型语言模型的自动反叙事 framework for evaluating large language model-generated automated counter-narratives across four dimensions 

---
# Generating Automotive Code: Large Language Models for Software Development and Verification in Safety-Critical Systems 

**Title (ZH)**: 生成汽车代码：大型语言模型在安全关键系统中进行软件开发与验证的应用 

**Authors**: Sven Kirchner, Alois C. Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2506.04038)  

**Abstract**: Developing safety-critical automotive software presents significant challenges due to increasing system complexity and strict regulatory demands. This paper proposes a novel framework integrating Generative Artificial Intelligence (GenAI) into the Software Development Lifecycle (SDLC). The framework uses Large Language Models (LLMs) to automate code generation in languages such as C++, incorporating safety-focused practices such as static verification, test-driven development and iterative refinement. A feedback-driven pipeline ensures the integration of test, simulation and verification for compliance with safety standards. The framework is validated through the development of an Adaptive Cruise Control (ACC) system. Comparative benchmarking of LLMs ensures optimal model selection for accuracy and reliability. Results demonstrate that the framework enables automatic code generation while ensuring compliance with safety-critical requirements, systematically integrating GenAI into automotive software engineering. This work advances the use of AI in safety-critical domains, bridging the gap between state-of-the-art generative models and real-world safety requirements. 

**Abstract (ZH)**: 基于生成人工智能的软件开发生命周期框架：面向安全关键汽车软件的系统复杂性和严格监管要求的应对策略 

---
# Privacy and Security Threat for OpenAI GPTs 

**Title (ZH)**: OpenAI GPTs的隐私与安全威胁 

**Authors**: Wei Wenying, Zhao Kaifa, Xue Lei, Fan Ming  

**Link**: [PDF](https://arxiv.org/pdf/2506.04036)  

**Abstract**: Large language models (LLMs) demonstrate powerful information handling capabilities and are widely integrated into chatbot applications. OpenAI provides a platform for developers to construct custom GPTs, extending ChatGPT's functions and integrating external services. Since its release in November 2023, over 3 million custom GPTs have been created. However, such a vast ecosystem also conceals security and privacy threats. For developers, instruction leaking attacks threaten the intellectual property of instructions in custom GPTs through carefully crafted adversarial prompts. For users, unwanted data access behavior by custom GPTs or integrated third-party services raises significant privacy concerns. To systematically evaluate the scope of threats in real-world LLM applications, we develop three phases instruction leaking attacks target GPTs with different defense level. Our widespread experiments on 10,000 real-world custom GPTs reveal that over 98.8% of GPTs are vulnerable to instruction leaking attacks via one or more adversarial prompts, and half of the remaining GPTs can also be attacked through multiround conversations. We also developed a framework to assess the effectiveness of defensive strategies and identify unwanted behaviors in custom GPTs. Our findings show that 77.5% of custom GPTs with defense strategies are vulnerable to basic instruction leaking attacks. Additionally, we reveal that 738 custom GPTs collect user conversational information, and identified 8 GPTs exhibiting data access behaviors that are unnecessary for their intended functionalities. Our findings raise awareness among GPT developers about the importance of integrating specific defensive strategies in their instructions and highlight users' concerns about data privacy when using LLM-based applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现出强大信息处理能力，并广泛应用于聊天机器人应用。OpenAI提供了一个平台，供开发者构建自定义GPT，扩展ChatGPT的功能并集成外部服务。自2023年11月发布以来，已有超过300万个自定义GPT被创建。然而，如此庞大的生态系统也隐藏着安全和隐私威胁。对于开发者来说，指令泄露攻击通过精心构造的对抗提示威胁到自定义GPT中指令的知识产权。对于用户来说，自定义GPT或集成的第三方服务的不当数据访问行为引起了显著的隐私关注。为了系统地评估现实世界大语言模型应用中的威胁范围，我们开发了针对不同防护级别的GPT进行三种阶段的指令泄露攻击。我们在10,000个实际应用的自定义GPT上进行了广泛实验，发现超过98.8%的GPT通过一个或多个对抗提示容易遭受指令泄露攻击，剩余的一半GPT也可以通过多轮对话被攻击。我们还开发了一个框架来评估防御策略的有效性并识别自定义GPT中的不当行为。我们的研究结果表明，77.5%配备防御策略的自定义GPT容易遭受基本的指令泄露攻击。此外，我们发现738个自定义GPT收集用户对话信息，并识别出8个表现出超出其功能需求的数据访问行为的GPT。我们的发现提高了GPT开发者对在其指令中集成特定防御策略重要性的认识，并突出了用户在使用基于大语言模型的应用时对数据隐私的关注。 

---
# DiffCAP: Diffusion-based Cumulative Adversarial Purification for Vision Language Models 

**Title (ZH)**: DiffCAP：基于扩散的累积对抗净化方法用于视觉语言模型 

**Authors**: Jia Fu, Yongtao Wu, Yihang Chen, Kunyu Peng, Xiao Zhang, Volkan Cevher, Sepideh Pashami, Anders Holst  

**Link**: [PDF](https://arxiv.org/pdf/2506.03933)  

**Abstract**: Vision Language Models (VLMs) have shown remarkable capabilities in multimodal understanding, yet their susceptibility to perturbations poses a significant threat to their reliability in real-world applications. Despite often being imperceptible to humans, these perturbations can drastically alter model outputs, leading to erroneous interpretations and decisions. This paper introduces DiffCAP, a novel diffusion-based purification strategy that can effectively neutralize adversarial corruptions in VLMs. We observe that adding minimal noise to an adversarially corrupted image significantly alters its latent embedding with respect to VLMs. Building on this insight, DiffCAP cumulatively injects random Gaussian noise into adversarially perturbed input data. This process continues until the embeddings of two consecutive noisy images reach a predefined similarity threshold, indicating a potential approach to neutralize the adversarial effect. Subsequently, a pretrained diffusion model is employed to denoise the stabilized image, recovering a clean representation suitable for the VLMs to produce an output. Through extensive experiments across six datasets with three VLMs under varying attack strengths in three task scenarios, we show that DiffCAP consistently outperforms existing defense techniques by a substantial margin. Notably, DiffCAP significantly reduces both hyperparameter tuning complexity and the required diffusion time, thereby accelerating the denoising process. Equipped with strong theoretical and empirical support, DiffCAP provides a robust and practical solution for securely deploying VLMs in adversarial environments. 

**Abstract (ZH)**: 基于扩散的去 adversarial 腐蚀策略（DiffCAP）：一种有效净化视觉语言模型的新型方法 

---
# VisCoder: Fine-Tuning LLMs for Executable Python Visualization Code Generation 

**Title (ZH)**: VisCoder: 细粒度调优大规模语言模型以生成可执行的Python可视化代码 

**Authors**: Yuansheng Ni, Ping Nie, Kai Zou, Xiang Yue, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.03930)  

**Abstract**: Large language models (LLMs) often struggle with visualization tasks like plotting diagrams, charts, where success depends on both code correctness and visual semantics. Existing instruction-tuning datasets lack execution-grounded supervision and offer limited support for iterative code correction, resulting in fragile and unreliable plot generation. We present VisCode-200K, a large-scale instruction tuning dataset for Python-based visualization and self-correction. It contains over 200K examples from two sources: (1) validated plotting code from open-source repositories, paired with natural language instructions and rendered plots; and (2) 45K multi-turn correction dialogues from Code-Feedback, enabling models to revise faulty code using runtime feedback. We fine-tune Qwen2.5-Coder-Instruct on VisCode-200K to create VisCoder, and evaluate it on PandasPlotBench. VisCoder significantly outperforms strong open-source baselines and approaches the performance of proprietary models like GPT-4o-mini. We further adopt a self-debug evaluation protocol to assess iterative repair, demonstrating the benefits of feedback-driven learning for executable, visually accurate code generation. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在可视化任务如绘制图表方面常遇到困难，这类任务的成功依赖于代码的正确性和视觉语义的准确表达。现有的指令调优数据集缺乏执行层面的监督，为迭代代码修正提供的支持有限，导致生成的图表脆弱且不可靠。我们提出了VisCode-200K，这是一个基于Python的大型指令调优数据集，用于可视化和自修正。它包含了超过200,000个实例，来源于两个来源：（1）来自开源代码库的验证过的绘图代码，配以自然语言指令和渲染图表；（2）来自Code-Feedback的45,000个多轮修正对话，使模型能够在运行时反馈驱动下修正错误代码。我们使用VisCode-200K对Qwen2.5-Coder-Instruct进行微调，生成VisCoder，并在PandasPlotBench上进行了评估。VisCoder显著优于开源基线，并接近专有模型如GPT-4o-mini的性能。我们进一步采用了一种自我调试评估协议来评估迭代修复，展示了反馈驱动学习对于生成可执行且视觉上准确的代码的益处。 

---
# RadialRouter: Structured Representation for Efficient and Robust Large Language Models Routing 

**Title (ZH)**: RadialRouter: 一种高效的稳健大型语言模型路由结构表示 

**Authors**: Ruihan Jin, Pengpeng Shao, Zhengqi Wen, Jinyang Wu, Mingkuan Feng, Shuai Zhang, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.03880)  

**Abstract**: The rapid advancements in large language models (LLMs) have led to the emergence of routing techniques, which aim to efficiently select the optimal LLM from diverse candidates to tackle specific tasks, optimizing performance while reducing costs. Current LLM routing methods are limited in effectiveness due to insufficient exploration of the intrinsic connection between user queries and the characteristics of LLMs. To address this issue, in this paper, we present RadialRouter, a novel framework for LLM routing which employs a lightweight Transformer-based backbone with a radial structure named RadialFormer to articulate the query-LLMs relationship. The optimal LLM selection is performed based on the final states of RadialFormer. The pipeline is further refined by an objective function that combines Kullback-Leibler divergence with the query-query contrastive loss to enhance robustness. Experimental results on RouterBench show that RadialRouter significantly outperforms existing routing methods by 9.2\% and 5.8\% in the Balance and Cost First scenarios, respectively. Additionally, its adaptability toward different performance-cost trade-offs and the dynamic LLM pool demonstrates practical application potential. 

**Abstract (ZH)**: 大语言模型（LLMs）的迅速发展促进了路由技术的 emergence，这些技术旨在高效地从多种候选LLMs中选择最适合特定任务的理想LLM，以优化性能并降低成本。当前的LLM路由方法由于对用户查询与LLM特性之间内在联系的探索不足而受到限制。为解决这一问题，本文提出 RadialRouter，这是一种新型的LLM路由框架，采用带有径向结构的轻量级基于Transformer的骨干RadialFormer来表达查询-LLM关系。最优LLM的选择是基于RadialFormer的最终状态执行的。该 pipeline 通过结合Kullback-Leibler 散度和查询-查询对比损失的目标函数进一步优化，以增强鲁棒性。实验结果表明，RadialRouter 在 RouterBench 上的平衡场景和成本优先场景中分别优于现有路由方法 9.2% 和 5.8%，并且其不同性能成本权衡的适应性和动态LLM池显示了其实用应用潜力。 

---
# Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons 

**Title (ZH)**: Knockout LLM评估：通过迭代的成对比较使用大型语言模型进行评估 

**Authors**: Isik Baran Sandan, Tu Anh Dinh, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2506.03785)  

**Abstract**: Large Language Models (LLMs) have shown to be effective evaluators across various domains such as machine translations or the scientific domain. Current LLM-as-a-Judge approaches rely mostly on individual assessments or a single round of pairwise assessments, preventing the judge LLM from developing a global ranking perspective. To address this, we present Knockout Assessment, an LLM-asa Judge method using a knockout tournament system with iterative pairwise comparisons. Experiments across three LLMs on two datasets show that knockout assessment improves scoring accuracy, increasing Pearson correlation with expert evaluations by 0.07 on average for university-level exam scoring and machine translation evaluations, aligning LLM assessments more closely with human scoring. 

**Abstract (ZH)**: 大型语言模型（LLMs）在机器翻译等领域显示了有效的评估能力。当前的LLM-as-a-Judge方法主要依赖个体评估或一轮pairwise评估，这妨碍了判断LLM形成全局排名视角。为了解决这个问题，我们提出了一种 Knockout Assessment 方法，该方法使用淘汰赛系统和迭代的pairwise比较。实验结果显示，Knockout Assessment 提高了评分准确性，在大学级考试评分和机器翻译评估中，与专家评分的皮尔森相关系数平均提高了0.07，使LLM评估更接近人类评分。 

---
# AhaKV: Adaptive Holistic Attention-Driven KV Cache Eviction for Efficient Inference of Large Language Models 

**Title (ZH)**: AhaKV：自适应全局注意力驱动的键值缓存淘汰机制以实现大型语言模型高效推理 

**Authors**: Yifeng Gu, Zicong Jiang, Jianxiu Jin, Kailing Guo, Ziyang Zhang, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03762)  

**Abstract**: Large Language Models (LLMs) have significantly advanced the field of Artificial Intelligence. However, their deployment is resource-intensive, not only due to the large number of model parameters but also because the (Key-Value) KV cache consumes a lot of memory during inference. While several works propose reducing the KV cache by evicting the unnecessary tokens, these approaches rely on accumulated attention score as eviction score to quantify the importance of the token. We identify the accumulated attention score is biased and it decreases with the position of the tokens in the mathematical expectation. As a result, the retained tokens concentrate on the initial positions, limiting model's access to global contextual information. To address this issue, we propose Adaptive holistic attention KV (AhaKV), it addresses the bias of the accumulated attention score by adaptively tuning the scale of softmax according the expectation of information entropy of attention scores. To make use of the holistic attention information in self-attention mechanism, AhaKV utilize the information of value vectors, which is overlooked in previous works, to refine the adaptive score. We show theoretically that our method is well suited for bias reduction. We deployed AhaKV on different models with a fixed cache budget. Experiments show that AhaKV successfully mitigates bias and retains crucial tokens across global context and achieve state-of-the-art results against other related work on several benchmark tasks. 

**Abstract (ZH)**: 大语言模型（LLMs）在人工智能领域取得了显著进展。然而，其部署资源密集，不仅因为模型参数量庞大，还在推理过程中消耗大量内存来维持（键-值）KV缓存。尽管有若干工作提出通过移除不必要的令牌来减少KV缓存，这些方法依赖累积注意力分数作为移除分数来量化令牌的重要性。我们发现累积注意力分数存在偏差，在数学期望下随令牌位置增加而减少。因此，保留的令牌集中在初始位置，限制了模型获取全局上下文信息的能力。为解决这一问题，我们提出了自适应全局注意力KV（AhaKV），通过根据注意力分数的信息熵期望自适应调整softmax的比例来减轻累积注意力分数的偏差。为了利用自注意力机制中的全局注意力信息，AhaKV利用值向量的信息，弥补了先前工作中的不足，以优化自适应评分。我们理论上证明了该方法适合于偏差减少。我们在固定缓存预算下将AhaKV部署到不同的模型上。实验表明，AhaKV成功减轻了偏差，保留了跨全局上下文的关键令牌，并在若干基准任务上取得了最佳结果，优于其他相关工作。 

---
# Verbalized Confidence Triggers Self-Verification: Emergent Behavior Without Explicit Reasoning Supervision 

**Title (ZH)**: 口头表达的自信触发自我验证：无需显式推理监督的 emergent 行为 

**Authors**: Chaeyun Jang, Moonseok Choi, Yegon Kim, Hyungi Lee, Juho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.03723)  

**Abstract**: Uncertainty calibration is essential for the safe deployment of large language models (LLMs), particularly when users rely on verbalized confidence estimates. While prior work has focused on classifiers or short-form generation, confidence calibration for chain-of-thought (CoT) reasoning remains largely unexplored. Surprisingly, we find that supervised fine-tuning with scalar confidence labels alone suffices to elicit self-verification behavior of language models, without any explicit reasoning supervision or reinforcement learning-based rewards. Despite being trained only to produce a verbalized confidence score without any self-verifying examples, the model learns to generate longer and self-checking responses for low-confidence queries while providing more concise answers for high-confidence ones. We further propose a simple rethinking method that boosts performance via test-time scaling based on calibrated uncertainty. Experiments on GSM8K and held-out reasoning tasks such as MATH-500 and ARC-Challenge show that our confidence-aware fine-tuning improves both calibration and accuracy, while also enhancing interpretability by aligning the model's reasoning path with its confidence. 

**Abstract (ZH)**: 大型语言模型（LLMs）的安全部署需要不确定性校准，特别是在用户依赖于口头化的置信度估计时。尽管先前的工作主要集中在分类器或短文本生成上，链式思考（CoT）推理的置信度校准仍是一个尚未充分探索的领域。令人惊讶的是，我们发现仅通过标注标量置信度标签的监督微调就足以引起语言模型的自我验证行为，而不需要任何显式的推理监督或基于强化学习的奖励。尽管仅训练模型生成口头化的置信度评分而没有自我验证的例子，该模型仍能够学习在低置信度查询中生成更长且自我检查的回答，在高置信度查询中提供更简洁的答案。我们进一步提出了一种简单的重新思考方法，通过测试时的标度来提高性能，基于校准的不确定性。对于GSM8K以及保留的推理任务如MATH-500和ARC-Challenge的实验表明，我们的置信度aware微调不仅提高了校准和准确率，还通过使模型的推理路径与置信度对齐从而增强了可解释性。 

---
# RewardAnything: Generalizable Principle-Following Reward Models 

**Title (ZH)**: RewardAnything: 可泛化的原理遵循奖励模型 

**Authors**: Zhuohao Yu, Jiali Zeng, Weizheng Gu, Yidong Wang, Jindong Wang, Fandong Meng, Jie Zhou, Yue Zhang, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.03637)  

**Abstract**: Reward Models, essential for guiding Large Language Model optimization, are typically trained on fixed preference datasets, resulting in rigid alignment to single, implicit preference distributions. This prevents adaptation to diverse real-world needs-from conciseness in one task to detailed explanations in another. The standard practice of collecting task-specific preference data and retraining reward models is resource-intensive, often producing biased rewards, and limits practical application. We introduce generalizable, principle-following reward models. We propose that RMs should understand and adhere to dynamically provided natural language specifications of reward principles, similar to instruction-following in LLMs. To measure this capability, we develop RABench, a comprehensive benchmark for RMs focusing on generalization across diverse principles. Evaluations on RABench reveal poor generalization of current RMs. As a solution, we present RewardAnything, a novel RM designed and trained to explicitly follow natural language principles. We achieve SotA performance with RewardAnything in traditional RM benchmark simply by specifying a well-defined principle, and results on RABench show we excel in adapting to novel principles without retraining. Furthermore, RewardAnything integrates seamlessly with existing RLHF methods and we show by a case study on how to automatically and efficiently align LLMs with only natural language principles. 

**Abstract (ZH)**: 可泛化的原理遵循型奖励模型：从固定偏好数据集到动态语言规范的转变 

---
# Robustness of Prompting: Enhancing Robustness of Large Language Models Against Prompting Attacks 

**Title (ZH)**: 提示的健壮性：增强大型语言模型对抗提示攻击的健壮性 

**Authors**: Lin Mu, Guowei Chu, Li Ni, Lei Sang, Zhize Wu, Peiquan Jin, Yiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03627)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various tasks by effectively utilizing a prompting strategy. However, they are highly sensitive to input perturbations, such as typographical errors or slight character order errors, which can substantially degrade their performance. Despite advances in prompting techniques, developing a prompting strategy that explicitly mitigates the negative impact of such perturbations remains an open challenge. To bridge this gap, we propose Robustness of Prompting (RoP), a novel prompting strategy specifically designed to enhance the robustness of LLMs. RoP consists of two stages: Error Correction and Guidance. In the Error Correction stage, RoP applies diverse perturbation methods to generate adversarial examples, which are then used to construct prompts that automatically correct input errors. In the Guidance stage, RoP generates an optimal guidance prompting based on the corrected input, steering the model toward more robust and accurate inferences. Through comprehensive experiments spanning arithmetic, commonsense, and logical reasoning tasks, we demonstrate that RoP significantly improves LLMs' robustness against adversarial perturbations. Notably, it maintains model accuracy with only minimal degradation compared to clean input scenarios, thereby establishing RoP as a practical and effective approach for enhancing LLM robustness in real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过有效利用激发策略在各种任务中展示了显著的性能。然而，它们对输入扰动的高度敏感性，如拼写错误或字符顺序错误，会导致其性能大幅下降。尽管在激发技术方面取得了进展，但开发一种能够明确减轻这些扰动负面影响的激发策略仍然是一个开放的挑战。为此，我们提出了一种新颖的激发策略——鲁棒性激发（RoP），专门设计用于增强LLMs的鲁棒性。RoP包括两个阶段：错误修正和引导。在错误修正阶段，RoP应用多种扰动方法生成对抗性示例，然后利用这些示例构建能够自动纠正输入错误的提示。在引导阶段，RoP基于修正后的输入生成最优的引导提示，引导模型得出更鲁棒和准确的推断。通过涵盖算术、常识和逻辑推理任务的全面实验，我们证明了RoP显著提高了LLMs对抗敌意扰动的鲁棒性。值得注意的是，RoP仅在轻微影响模型准确性的情况下维持了其有效性，从而确立了RoP作为一种在实际应用中增强LLM鲁棒性的实用且有效的方法的地位。 

---
# VLMs Can Aggregate Scattered Training Patches 

**Title (ZH)**: VLMs可以聚合散列的训练patches 

**Authors**: Zhanhui Zhou, Lingjie Chen, Chao Yang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03614)  

**Abstract**: One way to mitigate risks in vision-language models (VLMs) is to remove dangerous samples in their training data. However, such data moderation can be easily bypassed when harmful images are split into small, benign-looking patches, scattered across many training samples. VLMs may then learn to piece these fragments together during training and generate harmful responses at inference, either from full images or text references. For instance, if trained on image patches from a bloody scene paired with the descriptions "safe," VLMs may later describe, the full image or a text reference to the scene, as "safe." We define the core ability of VLMs enabling this attack as $\textit{visual stitching}$ -- the ability to integrate visual information spread across multiple training samples that share the same textual descriptions. In our work, we first demonstrate visual stitching abilities in common open-source VLMs on three datasets where each image is labeled with a unique synthetic ID: we split each $(\texttt{image}, \texttt{ID})$ pair into $\{(\texttt{patch}, \texttt{ID})\}$ pairs at different granularity for finetuning, and we find that tuned models can verbalize the correct IDs from full images or text reference. Building on this, we simulate the adversarial data poisoning scenario mentioned above by using patches from dangerous images and replacing IDs with text descriptions like ``safe'' or ``unsafe'', demonstrating how harmful content can evade moderation in patches and later be reconstructed through visual stitching, posing serious VLM safety risks. Code is available at this https URL. 

**Abstract (ZH)**: 一种缓解视觉语言模型风险的方法是移除其训练数据中的危险样本。然而，当有害图像被拆分成小的、看似无害的片段，并散布在多个训练样本中时，这样的数据净化措施很容易被绕过。视觉语言模型在训练过程中可能会学习将这些片段拼接起来，并在推理时生成有害的响应，无论是从完整图像还是文本引用中。例如，如果模型是在带有“安全”描述的血腥场景图像片段上进行训练，那么该模型在推理时可能会将完整图像或场景描述描述为“安全”。我们定义使视觉语言模型能够执行此攻击的核心能力为$\textit{视觉拼接}$——即整合具有相同文本描述的多个训练样本中分散的视觉信息的能力。在我们的研究中，我们首先在三个每个图像带有唯一合成ID的数据集上展示了常见开源视觉语言模型的视觉拼接能力：我们将每个$(\texttt{图像}, \texttt{ID})$对在不同的粒度下拆分成$\{(\texttt{片段}, \texttt{ID})\}$对进行微调，发现调整后的模型可以从完整图像或文本引用中表达正确的ID。在此基础上，我们通过使用危险图像的片段并用类似的“安全”或“不安全”的文本描述替换ID，模拟了上述对抗性数据污染场景，展示了有害内容如何通过视觉拼接在片段中避开关机措施，并重新构建，从而对视觉语言模型的安全性构成严重威胁。代码可在以下链接获取。 

---
# POSS: Position Specialist Generates Better Draft for Speculative Decoding 

**Title (ZH)**: POSSS: 位置专家生成更好的 speculative 解码草稿 

**Authors**: Langlin Huang, Chengsong Huang, Jixuan Leng, Di Huang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03566)  

**Abstract**: Speculative decoding accelerates Large Language Model (LLM) inference by using a small draft model to predict multiple tokens, and a large target model to verify these tokens in parallel. Recent studies leverage the hidden state of the target model to enhance draft model prediction accuracy. However, existing methods suffer from the degrading quality of draft token predictions at later positions, due to error accumulation in draft model generated features. In this paper, we propose Position Specialists (PosS), which consist of multiple position-specialized draft layers to generate tokens at assigned position(s). Position specialists greatly improve token acceptance rate at later positions per drafting round, as each specialist only needs to focus on handling a certain level of draft model feature deviation. Experiment results on Llama-3-8B-Instruct and Llama-2-13B-chat across six datasets demonstrate that PosS effectively improves over baselines on average acceptance length and speed-up ratio. Our codebase is available at this https URL. 

**Abstract (ZH)**: 推测性解码通过使用小草稿模型预测多个令牌，并使用大目标模型并行验证这些令牌来加速大型语言模型（LLM）推断。 recent studies 利用目标模型的隐藏状态来增强草稿模型的预测准确性。然而，现有方法在草稿令牌预测的后期位置上遭受预测质量下降的问题，这是由于草稿模型生成的功能中的误差累积造成的。在本文中，我们提出了位置专家（PosS），它由多个针对特定位置的专业化草稿层组成，以生成指定位置的令牌。位置专家在每次草稿轮次中极大地提高了后期位置的令牌接受率，因为每个专家只需专注于处理草稿模型特征偏差的某个级别。在 Llama-3-8B-Instruct 和 Llama-2-13B-chat 上跨六个数据集的实验结果表明，PosS 平均接受长度和加速比相比基准模型有显著改善。我们的代码库可在以下网址获取。 

---
# Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement 

**Title (ZH)**: 辩论、反思与提炼：基于树状偏好优化的多Agent反馈机制以实现高效语言模型增强 

**Authors**: Xiaofeng Zhou, Heyan Huang, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.03541)  

**Abstract**: Large Language Models (LLMs) continue to set new standards in knowledge-intensive and complex reasoning tasks, yet their high computational demands limit widespread adoption. While distilling large models into smaller ones offers a sustainable solution, current techniques--such as static knowledge distillation, resource-intensive reinforcement learning from human feedback, or limited self-reflection--struggle to yield substantial and lasting performance gains. In this paper, we present a novel Debate and Reflect (D&R) framework that orchestrates multi-turn debates between smaller models and stronger teacher models, eliciting actionable feedback (e.g., error analysis, corrective strategies) to guide student models. Further, we introduce Tree-structured Direct Preference Optimization (T-DPO) to efficiently leverage these debate logs, organizing interactions into a hierarchical format for effective training. Empirical evaluations across diverse NLP benchmarks demonstrate that our approach significantly improves smaller-model accuracy, robustness, and generalization, outperforming conventional baselines by a large margin. 

**Abstract (ZH)**: Large Language Models (LLMs) 在知识密集和复杂推理任务中持续设定新标准，但其高的计算需求限制了其广泛应用。虽然将大型模型精简为较小规模的模型是一种可持续的解决方案，但当前的技术手段——如静态知识精简、资源密集型基于人类反馈的强化学习或有限的自我反思——难以实现显著且持久的性能提升。在本文中，我们提出了一种名为 Debate and Reflect (D&R) 的新型框架，通过较小模型与更强的教师模型进行多轮辩论，激发可操作的反馈（如错误分析、纠正策略）来指导学生模型。此外，我们引入了基于树结构的直接偏好优化 (T-DPO) 方法，高效地利用辩论日志，将其组织成层次结构以进行有效的训练。在多种自然语言处理基准测试中的实证评估表明，我们的方法显著提高了较小模型的准确率、鲁棒性和通用性，并在很大程度上优于传统基线方法。 

---
# Measuring Human Involvement in AI-Generated Text: A Case Study on Academic Writing 

**Title (ZH)**: 评估人工智能生成文本中的人类参与度：以学术写作为例 

**Authors**: Yuchen Guo, Zhicheng Dou, Huy H. Nguyen, Ching-Chun Chang, Saku Sugawara, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2506.03501)  

**Abstract**: Content creation has dramatically progressed with the rapid advancement of large language models like ChatGPT and Claude. While this progress has greatly enhanced various aspects of life and work, it has also negatively affected certain areas of society. A recent survey revealed that nearly 30% of college students use generative AI to help write academic papers and reports. Most countermeasures treat the detection of AI-generated text as a binary classification task and thus lack robustness. This approach overlooks human involvement in the generation of content even though human-machine collaboration is becoming mainstream. Besides generating entire texts, people may use machines to complete or revise texts. Such human involvement varies case by case, which makes binary classification a less than satisfactory approach. We refer to this situation as participation detection obfuscation. We propose using BERTScore as a metric to measure human involvement in the generation process and a multi-task RoBERTa-based regressor trained on a token classification task to address this problem. To evaluate the effectiveness of this approach, we simulated academic-based scenarios and created a continuous dataset reflecting various levels of human involvement. All of the existing detectors we examined failed to detect the level of human involvement on this dataset. Our method, however, succeeded (F1 score of 0.9423 and a regressor mean squared error of 0.004). Moreover, it demonstrated some generalizability across generative models. Our code is available at this https URL 

**Abstract (ZH)**: 大语言模型如ChatGPT和Claude的迅速进步极大地推动了内容创作，但也对某些社会领域产生了负面影响。最近的一项调查表明，近30%的大学生使用生成式AI帮助撰写学术论文和报告。大多数应对措施将检测AI生成的文本视为二元分类任务，从而缺乏 robustness。这种做法忽视了即使在人机协作日益普遍的情况下，人类在内容生成中的参与。除了生成整个文本外，人们还可能使用机器来完成或修订文本。这种参与因情况而异，使得二元分类方法不够理想。我们称之为参与检测混淆。我们建议使用BERTScore作为衡量人类在生成过程中参与度的指标，并基于标记分类任务训练一个多任务RoBERTa回归器来解决这一问题。为了评估该方法的有效性，我们模拟了基于学术的场景，并创建了一个反映不同参与度水平的连续数据集。我们检查的所有现有检测器在这份数据集上均未能检测到人类参与的程度。然而，我们的方法成功了（F1分数为0.9423，回归器均方误差为0.004），并且显示出一定的跨生成模型的一般适用性。我们的代码可在以下链接获取。 

---
# EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding 

**Title (ZH)**: EpiCoDe: 超出训练范围的外推与对比解码增强模型性能 

**Authors**: Mingxu Tao, Jie Hu, Mingchuan Yang, Yunhuai Liu, Dongyan Zhao, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.03489)  

**Abstract**: The remarkable performance of Large language models (LLMs) relies heavily on the availability of abundant high-quality training data. However, the high cost of acquiring annotated data often prevents models from obtaining capabilities to tackle downstream tasks. In this paper, we introduce a novel method, EpiCoDe that boosts model performance in data-scarcity scenarios without extra training. We first employ model extrapolation to enhance a finetuned model with its inferior version, and then adopt contrastive decoding to further reduce predicted errors, by comparing the logit scores given by the extrapolated and the vanilla finetuned model. Experiments across three tasks over four different LLMs show that EpiCoDe consistently outperforms existing methods with significant and robust improvement. We also propose a new theoretical framework to reveal the mechanism behind contrastive decoding in data-scarcity scenarios, which further helps us better understand the effectiveness of EpiCoDe. 

**Abstract (ZH)**: 大规模语言模型(EpiCoDe)在数据稀缺场景下的性能提升方法 

---
# Sampling Preferences Yields Simple Trustworthiness Scores 

**Title (ZH)**: 采样偏好生成简单的可信度评分 

**Authors**: Sean Steinle  

**Link**: [PDF](https://arxiv.org/pdf/2506.03399)  

**Abstract**: With the onset of large language models (LLMs), the performance of artificial intelligence (AI) models is becoming increasingly multi-dimensional. Accordingly, there have been several large, multi-dimensional evaluation frameworks put forward to evaluate LLMs. Though these frameworks are much more realistic than previous attempts which only used a single score like accuracy, multi-dimensional evaluations can complicate decision-making since there is no obvious way to select an optimal model. This work introduces preference sampling, a method to extract a scalar trustworthiness score from multi-dimensional evaluation results by considering the many characteristics of model performance which users value. We show that preference sampling improves upon alternate aggregation methods by using multi-dimensional trustworthiness evaluations of LLMs from TrustLLM and DecodingTrust. We find that preference sampling is consistently reductive, fully reducing the set of candidate models 100% of the time whereas Pareto optimality never reduces the set by more than 50%. Likewise, preference sampling is consistently sensitive to user priors-allowing users to specify the relative weighting and confidence of their preferences-whereas averaging scores is intransigent to the users' prior knowledge. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的出现，人工智能（AI）模型的性能变得多维化。因此，已经提出了多个多维评估框架来评价LLMs。尽管这些框架比只使用单一得分如准确率的早期尝试更为现实，多维度评价可能会使决策复杂化，因为没有明显的方法选择最优模型。本工作引入了偏好抽样方法，该方法通过考虑用户重视的多种模型性能特征，从多维评估结果中提取单一可信度评分。我们通过使用TrustLLM和DecodingTrust对LLMs的多维可信度评估结果，证明偏好抽样方法在聚合方法上更为优越。我们发现，偏好抽样方法始终具有压缩性，能够100%地减少候选模型的集合，而帕累托最优仅能最多减少50%。同样，偏好抽样方法始终对用户先验敏感，允许用户指定其偏好之间的相对权重和置信度，而平均得分则对用户先验知识不敏感。 

---
# Ask a Local: Detecting Hallucinations With Specialized Model Divergence 

**Title (ZH)**: 询问当地人：使用专业模型离散化检测幻觉 

**Authors**: Aldan Creo, Héctor Cerezo-Costas, Pedro Alonso-Doval, Maximiliano Hormazábal-Lagos  

**Link**: [PDF](https://arxiv.org/pdf/2506.03357)  

**Abstract**: Hallucinations in large language models (LLMs) - instances where models generate plausible but factually incorrect information - present a significant challenge for AI.
We introduce "Ask a Local", a novel hallucination detection method exploiting the intuition that specialized models exhibit greater surprise when encountering domain-specific inaccuracies. Our approach computes divergence between perplexity distributions of language-specialized models to identify potentially hallucinated spans. Our method is particularly well-suited for a multilingual context, as it naturally scales to multiple languages without the need for adaptation, relying on external data sources, or performing training. Moreover, we select computationally efficient models, providing a scalable solution that can be applied to a wide range of languages and domains.
Our results on a human-annotated question-answer dataset spanning 14 languages demonstrate consistent performance across languages, with Intersection-over-Union (IoU) scores around 0.3 and comparable Spearman correlation values. Our model shows particularly strong performance on Italian and Catalan, with IoU scores of 0.42 and 0.38, respectively, while maintaining cross-lingual effectiveness without language-specific adaptations. We release our code and architecture to facilitate further research in multilingual hallucination detection. 

**Abstract (ZH)**: 大型语言模型中的幻觉——模型生成的虽具说服力但事实错误的信息——对AI构成了重大挑战。"咨询本地专家"：一种新颖的幻觉检测方法，利用专业模型在遇到领域特定不准确信息时表现出更大 Surprise 的直觉。该方法通过计算语言专业化模型困惑度分布之间的差异来识别潜在的幻觉片段。该方法特别适用于多语言环境，可以在没有适应、依赖外部数据源或重新训练的情况下自然扩展到多种语言。此外，我们选择了计算效率高的模型，提供了一个可扩展的解决方案，可以应用于多种语言和领域。我们的结果表明，该方法在覆盖14种语言的人工标注问答数据集上表现一致，平均交并比（IoU）约为0.3，Spearman相关值具有可比性。该模型在意大利语和加泰罗尼亚语上的表现尤为突出，IoU 分数分别为0.42和0.38，同时保持跨语言有效性，无需语言特定适应。我们发布了代码和架构以促进多语言幻觉检测的进一步研究。 

---
# Robustness in Both Domains: CLIP Needs a Robust Text Encoder 

**Title (ZH)**: 在两个领域都具备鲁棒性：CLIP需要一个鲁棒的文本编码器 

**Authors**: Elias Abad Rocamora, Christian Schlarmann, Naman Deep Singh, Yongtao Wu, Matthias Hein, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2506.03355)  

**Abstract**: Adversarial input attacks can cause a significant shift of CLIP embeddings. This can affect the downstream robustness of models incorporating CLIP in the pipeline, such as text-to-image generative models or large vision language models. While some efforts have been done towards making the CLIP image encoders robust, the robustness of text encoders remains unexplored. In this work, we cover this gap in the literature. We propose LEAF: an efficient adversarial finetuning method for the text domain, with the ability to scale to large CLIP models. Our models significantly improve the zero-shot adversarial accuracy in the text domain, while maintaining the vision performance provided by robust image encoders. When combined with text-to-image diffusion models, we can improve the generation quality under adversarial noise. When employing our robust CLIP encoders in multimodal retrieval tasks, we improve the recall under adversarial noise over standard CLIP models. Finally, we show that robust text encoders facilitate better reconstruction of input text from its embedding via direct optimization. 

**Abstract (ZH)**: 对抗输入攻击会导致CLIP嵌入发生显著偏移。这会影响包含CLIP的管道中的下游模型的鲁棒性，例如文本到图像生成模型或大型视觉语言模型。尽管已经有一些努力致力于使CLIP图像编码器变得鲁棒，但文本编码器的鲁棒性仍未被研究。在本文中，我们填补了这一文献空白。我们提出LEAF：一种高效的文本域对抗微调方法，能够扩展到大型CLIP模型。我们的模型在文本域显著提高了零样本对抗准确率，同时保持了由鲁棒图像编码器提供的视觉性能。当与文本到图像扩散模型结合使用时，我们可以在对抗噪声下改善生成质量。当在跨模态检索任务中采用我们 robust 的CLIP编码器时，我们改善了在对抗噪声下的召回率，超过了标准的CLIP模型。最后，我们展示了 robust 的文本编码器通过直接优化能够更好地从其嵌入中重建输入文本。 

---
# Mitigating Non-IID Drift in Zeroth-Order Federated LLM Fine-Tuning with Transferable Sparsity 

**Title (ZH)**: 在转移可迁移稀疏性的辅助下缓解零阶联邦大语言模型微调中的非IID漂移 

**Authors**: Yide Ran, Wentao Guo, Jingwei Sun, Yanzhou Pan, Xiaodong Yu, Hao Wang, Jianwen Xie, Yiran Chen, Denghui Zhang, Zhaozhuo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03337)  

**Abstract**: Federated Learning enables collaborative fine-tuning of Large Language Models (LLMs) across decentralized Non-Independent and Identically Distributed (Non-IID) clients, but such models' massive parameter sizes lead to significant memory and communication challenges. This work introduces Meerkat, a sparse zeroth-order optimization (ZO) method designed for federated LLM fine-tuning. By limiting fine-tuning to a transferable, static, extremely sparse subset of parameters, Meerkat achieves remarkable communication efficiency, enabling cost-effective high-frequency synchronization. With theoretical analysis and experiments, we show that this high-frequency communication effectively mitigates Non-IID data challenges and leads to superior performance compared to full-parameter ZO. Furthermore, experiment results show that Meerkat outperforms existing sparsity baselines with better performance at the same communication frequency. To further handle Non-IID drift, Meerkat leverages traceable local updates and forms a virtual path for each client. This virtual path mechanism reveals the GradIP phenomenon: the inner products between LLM pre-training gradients maintained by server and client gradients estimated via ZO converges for extreme Non-IID clients but oscillates for IID ones. This distinct behavior provides a signal for identifying clients with extreme data heterogeneity. Using this signal, Meerkat-vp is proposed to analyze GradIP trajectories to identify extreme Non-IID clients and applies early stopping to enhance aggregated model quality. Experiments confirm that Meerkat and Meerkat-vp significantly improve the efficiency and effectiveness of ZO federated LLM fine-tuning. 

**Abstract (ZH)**: 联邦学习 Enables 分布式非独立非同分布客户端上大型语言模型的协作微调，但模型的庞大参数量导致了显著的内存和通信挑战。本工作介绍 Meerkat，一种针对联邦语言模型微调设计的稀疏零阶优化方法。通过限制定向传输到可转移的、静态的、极其稀疏的参数子集，Meerkat 实现了卓越的通信效率，使得低成本高频同步成为可能。通过理论分析和实验，我们证明了这种高频通信有效地缓解了非独立非同分布数据的挑战，并且相比全参数零阶优化方法具有更优的性能。此外，实验结果表明，在相同的通信频率下，Meerkat 在性能上优于现有稀疏性基线方法。为了进一步处理非独立非同分布漂移，Meerkat 利用可追踪的本地更新，并为每个客户端形成一条虚拟路径。这种虚拟路径机制揭示了 GradIP 现象：服务器保留的大型语言模型预训练梯度与客户端通过零阶优化估计的梯度之间的内积，在极端非独立非同分布客户端处收敛，而在独立同分布客户端处振荡。这种不同行为为识别具有极端数据异质性的客户端提供了信号。通过利用这一信号，Meerkat-vp 被提出用于分析 GradIP 轨迹以识别极端非独立非同分布客户端，并通过早期停止提升聚合模型质量。实验结果证实，Meerkat 和 Meerkat-vp 显著提高了零阶优化在联邦语言模型微调中的效率和效果。 

---
# The Future of Continual Learning in the Era of Foundation Models: Three Key Directions 

**Title (ZH)**: 基础模型时代 continual 学习的未来：三个关键方向 

**Authors**: Jack Bell, Luigi Quarantiello, Eric Nuertey Coleman, Lanpei Li, Malio Li, Mauro Madeddu, Elia Piccoli, Vincenzo Lomonaco  

**Link**: [PDF](https://arxiv.org/pdf/2506.03320)  

**Abstract**: Continual learning--the ability to acquire, retain, and refine knowledge over time--has always been fundamental to intelligence, both human and artificial. Historically, different AI paradigms have acknowledged this need, albeit with varying priorities: early expert and production systems focused on incremental knowledge consolidation, while reinforcement learning emphasised dynamic adaptation. With the rise of deep learning, deep continual learning has primarily focused on learning robust and reusable representations over time to solve sequences of increasingly complex tasks. However, the emergence of Large Language Models (LLMs) and foundation models has raised the question: Do we still need continual learning when centralised, monolithic models can tackle diverse tasks with access to internet-scale knowledge? We argue that continual learning remains essential for three key reasons: (i) continual pre-training is still necessary to ensure foundation models remain up to date, mitigating knowledge staleness and distribution shifts while integrating new information; (ii) continual fine-tuning enables models to specialise and personalise, adapting to domain-specific tasks, user preferences, and real-world constraints without full retraining, avoiding the need for computationally expensive long context-windows; (iii) continual compositionality offers a scalable and modular approach to intelligence, enabling the orchestration of foundation models and agents to be dynamically composed, recombined, and adapted. While continual pre-training and fine-tuning are explored as niche research directions, we argue it is continual compositionality that will mark the rebirth of continual learning. The future of AI will not be defined by a single static model but by an ecosystem of continually evolving and interacting models, making continual learning more relevant than ever. 

**Abstract (ZH)**: 持续学习——这种能够随着时间获取、保留和精炼知识的能力一直是人类和人工智能的基本要素。历史上，不同的AI范式都承认了这一需求，尽管各有侧重：早期的专业系统和生产系统侧重于增量知识整合，而强化学习则强调动态适应。随着深度学习的兴起，深度持续学习主要关注随着时间的推移学习 robust 和可重复使用的表示，以解决越来越复杂的任务序列。然而，大型语言模型（LLMs）和基础模型的出现提出了一个问题：当中央集权的单一模型可以利用互联网规模的知识来应对多样的任务时，我们是否还需要持续学习？我们认为持续学习仍然至关重要，原因有三：（i）持续预训练仍然必要，以确保基础模型保持最新，减轻知识陈旧和分布偏移，同时整合新信息；（ii）持续微调使模型能够专业化和个人化，根据具体领域的任务、用户偏好和现实世界约束适应，而无需完全重新训练，避免需要计算成本高昂的长上下文窗口；（iii）持续组合提供了扩展和模块化的智能方法，使基础模型和代理能够在动态组合、重组和适应中相协调。尽管持续预训练和微调被探索为研究方向，我们认为持续组合将继续成为持续学习的新生命。人工智能的未来将不是由单一静态模型定义，而是由不断进化和相互作用的模型生态系统定义，使持续学习比以往更加重要。 

---
# Hopscotch: Discovering and Skipping Redundancies in Language Models 

**Title (ZH)**: 跳房子：发现并跳过语言模型中的冗余 

**Authors**: Mustafa Eyceoz, Nikhil Shivakumar Nayak, Hao Wang, Ligong Han, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2506.03303)  

**Abstract**: Modern causal language models stack many attention blocks to improve performance, but not all blocks are necessary for every task. We propose Hopscotch, a simple yet effective method that identifies and skips attention blocks with least contributions to a task and adapts to preserve output quality. Hopscotch jointly optimizes which blocks to skip and how to scale the outputs of the remaining layers. By introducing lightweight, trainable scaling parameters to attention and MLP blocks, it mitigates distribution shifts in hidden states caused by removing attention blocks. Hopscotch does not modify model weights or require access to pretraining or instruction-tuning data, and is compatible with existing model compression techniques. When applied to $\texttt{Llama-3.1-8B}$ and $\texttt{Qwen2.5-7B}$, Hopscotch achieves less than a 2% drop in performance even after skipping four attention blocks. 

**Abstract (ZH)**: Hopscotch: 一种简单有效的注意力模块选择与缩放方法以保持输出质量 

---
# HyperSteer: Activation Steering at Scale with Hypernetworks 

**Title (ZH)**: HyperSteer：大规模Hyper网络驱动激活选择 

**Authors**: Jiuding Sun, Sidharth Baskaran, Zhengxuan Wu, Michael Sklar, Christopher Potts, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2506.03292)  

**Abstract**: Steering language models (LMs) by modifying internal activations is a popular approach for controlling text generation. Unsupervised dictionary learning methods, e.g., sparse autoencoders, can be scaled to produce many steering vectors, but lack guarantees on the individual efficacy of each vector and control over the coverage of relevant steering tasks. In contrast, supervised methods for constructing steering vectors are targeted and effective, but require more data collection and training for each additional steering vector produced. In this work, we introduce HyperSteer, a family of hypernetwork-based architectures which are trained end-to-end to generate steering vectors conditioned on the natural language steering prompts and the internals of the steered LM. In our evaluations, we show that scaling HyperSteer with thousands of steering prompts exceeds the performance of state-of-the-art activation steering methods, even on steering prompts never seen during training. Moreover, HyperSteer performs on par with steering-via-prompting. 

**Abstract (ZH)**: 通过修改内部激活来引导语言模型（LMs）是控制文本生成的一种流行方法。无监督字典学习方法，例如稀疏自编码器，可以扩展以生成多个引导向量，但缺乏每个向量个体效果的保证，以及对相关引导任务覆盖面的控制。相比之下，构建引导向量的监督方法更具针对性和效果，但每生成一个额外的引导向量需要更多的数据收集和训练。在本工作中，我们介绍了基于超网络的HyperSteer家族架构，这些架构在端到端训练中，根据自然语言引导提示和引导的LM内部生成引导向量。在我们的评估中，我们将HyperSteer扩展到数千个引导提示，即使对于训练过程中未见过的引导提示，其性能也超过了最先进的激活引导方法。此外，HyperSteer与基于提示的引导表现相当。 

---
# BadReward: Clean-Label Poisoning of Reward Models in Text-to-Image RLHF 

**Title (ZH)**: BadReward: 奖励模型在文本到图像RLHF中的清洁标签投毒 

**Authors**: Kaiwen Duan, Hongwei Yao, Yufei Chen, Ziyun Li, Tong Qiao, Zhan Qin, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03234)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is crucial for aligning text-to-image (T2I) models with human preferences. However, RLHF's feedback mechanism also opens new pathways for adversaries. This paper demonstrates the feasibility of hijacking T2I models by poisoning a small fraction of preference data with natural-appearing examples. Specifically, we propose BadReward, a stealthy clean-label poisoning attack targeting the reward model in multi-modal RLHF. BadReward operates by inducing feature collisions between visually contradicted preference data instances, thereby corrupting the reward model and indirectly compromising the T2I model's integrity. Unlike existing alignment poisoning techniques focused on single (text) modality, BadReward is independent of the preference annotation process, enhancing its stealth and practical threat. Extensive experiments on popular T2I models show that BadReward can consistently guide the generation towards improper outputs, such as biased or violent imagery, for targeted concepts. Our findings underscore the amplified threat landscape for RLHF in multi-modal systems, highlighting the urgent need for robust defenses. Disclaimer. This paper contains uncensored toxic content that might be offensive or disturbing to the readers. 

**Abstract (ZH)**: 自然语言生成模型中的有害奖励攻击：劫持文本到图像模型的隐蔽清洁标签中毒攻击 

---
# NetPress: Dynamically Generated LLM Benchmarks for Network Applications 

**Title (ZH)**: NetPress: 动态生成的网络应用LLM基准测试 

**Authors**: Yajie Zhou, Jiajun Ruan, Eric S. Wang, Sadjad Fouladi, Francis Y. Yan, Kevin Hsieh, Zaoxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03231)  

**Abstract**: Despite growing interest in domain-specific benchmarking of large language models (LLMs) and agents, current evaluations remain limited to static, small-scale datasets, especially in high-stakes tasks like network operations that demand reliability for deployments. We present NetPress, an automated benchmark generation framework for evaluating LLM agents in network applications. NetPress introduces a unified abstraction with state and action, enabling dynamic generation of diverse query sets along with corresponding ground truths. At runtime, users can specify benchmark configurations to generate millions of queries on the fly. In addition to dynamic benchmark construction, NetPress integrates with network emulators to provide realistic environment feedback, supporting comprehensive evaluation across correctness, safety, and latency. We instantiate NetPress on three representative applications, revealing interesting fine-grained differences in agent behavior that static, correctness-only benchmarks often miss. NetPress moves LLM evaluation toward realistic, scalable testing in infrastructure-centric domains, helping close the gap between benchmark performance and real-world deployment readiness. Code is available at this https URL. 

**Abstract (ZH)**: NetPress：面向网络应用的大语言模型代理自动化基准生成框架 

---
# DiaBlo: Diagonal Blocks Are Sufficient For Finetuning 

**Title (ZH)**: DiaBlo: 对角块足以进行微调 

**Authors**: Selcuk Gurses, Aozhong Zhang, Yanxia Deng, Xun Dong, Xin Li, Naigang Wang, Penghang Yin, Zi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03230)  

**Abstract**: Finetuning is a critical step for adapting large language models (LLMs) to domain-specific downstream tasks. To mitigate the substantial computational and memory costs of full-model fine-tuning, Parameter-Efficient Finetuning (PEFT) methods have been proposed to update only a small subset of model parameters. However, performance gaps between PEFT approaches and full-model fine-tuning still exist. In this work, we present DiaBlo, a simple yet effective PEFT approach that updates only the diagonal blocks of selected model weight matrices. Unlike Low Rank Adaptation (LoRA) and its variants, DiaBlo eliminates the need for low rank matrix products, thereby avoiding the reliance on auxiliary initialization schemes or customized optimization strategies to improve convergence. This design leads to stable and robust convergence while maintaining comparable memory efficiency and training speed to LoRA. We conduct extensive experiments across a range of tasks, including commonsense reasoning, arithmetic reasoning, code generation, and safety alignment, to evaluate the effectiveness and efficiency of DiaBlo. Across these benchmarks, DiaBlo demonstrates strong and consistent performance while maintaining high memory efficiency and fast finetuning speed. Codes are available at this https URL. 

**Abstract (ZH)**: DiaBlo: 一种简单有效的参数高效微调方法 

---
# Unlabeled Data Improves Fine-Grained Image Zero-shot Classification with Multimodal LLMs 

**Title (ZH)**: 未标注数据改进多模态LLM的细粒度图像零样本分类 

**Authors**: Yunqi Hong, Sohyun An, Andrew Bai, Neil Y.C. Lin, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2506.03195)  

**Abstract**: Despite Multimodal Large Language Models (MLLMs) showing promising results on general zero-shot image classification tasks, fine-grained image classification remains challenging. It demands precise attention to subtle visual details to distinguish between visually similar subcategories--details that MLLMs may easily overlook without explicit guidance. To address this, we introduce AutoSEP, an iterative self-supervised prompt learning framework designed to enhance MLLM fine-grained classification capabilities in a fully unsupervised manner. Our core idea is to leverage unlabeled data to learn a description prompt that guides MLLMs in identifying crucial discriminative features within an image, and boosts classification accuracy. We developed an automatic self-enhancing prompt learning framework called AutoSEP to iteratively improve the description prompt using unlabeled data, based on instance-level classification scoring function. AutoSEP only requires black-box access to MLLMs, eliminating the need for any training or fine-tuning. We evaluate our approach on multiple fine-grained classification datasets. It consistently outperforms other unsupervised baselines, demonstrating the effectiveness of our self-supervised optimization framework. Notably, AutoSEP on average improves 13 percent over standard zero-shot classification and 5 percent over the best-performing baselines. Code is available at: this https URL 

**Abstract (ZH)**: 尽管多模态大语言模型（MLLMs）在通用零 shot 图像分类任务中显示出了有希望的结果，但细粒度图像分类仍然具有挑战性。它要求对细微的视觉细节给予精确的关注，以区分视觉上相似的子类别——MLLMs 在没有明确指导的情况下可能会忽略这些细节。为了解决这个问题，我们引入了 AutoSEP，这是一种迭代的自我监督提示学习框架，旨在以完全无监督的方式增强 MLLM 的细粒度分类能力。我们的核心思想是利用未标注数据来学习一个描述性提示，该提示可以指导 MLLMs 识别图像中的关键判别特征，并提升分类准确性。我们基于实例级分类评分函数开发了一个自动自我增强提示学习框架 AutoSEP，用于迭代地利用未标注数据改进描述性提示。AutoSEP 只需要对 MLLMs 的黑盒访问，无需任何训练或微调。我们在多个细粒度分类数据集中评估了我们的方法。结果显示，该方法在多个未监督基准上表现优异，证明了我们自我监督优化框架的有效性。值得注意的是，AutoSEP 平均提高了 13% 的标准零 shot 分类性能，并优于最佳基准 5%。代码可在以下网址获取：this https URL 

---
# HueManity: Probing Fine-Grained Visual Perception in MLLMs 

**Title (ZH)**: Huemanity: 探究MLLMs的细粒度视觉感知 

**Authors**: Rynaa Grover, Jayant Sravan Tamarapalli, Sahiti Yerramilli, Nilay Pande  

**Link**: [PDF](https://arxiv.org/pdf/2506.03194)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel at high-level visual reasoning, but their performance on nuanced perceptual tasks remains surprisingly limited. We present HueManity, a benchmark designed to assess visual perception in MLLMs. The dataset comprises 83,850 images featuring two-character alphanumeric strings embedded in Ishihara test style dot patterns, challenging models on precise pattern recognition. Our evaluation of nine state-of-the-art MLLMs on HueManity demonstrates a significant performance deficit compared to human and traditional computer vision baselines. The best-performing MLLM achieved a 33.6% accuracy on the numeric `easy' task and a striking 3% on the alphanumeric `hard' task. In contrast, human participants achieved near-perfect scores (100% and 95.6%), and a fine-tuned ResNet50 model reached accuracies of 96.5% and 94.5%. These results highlight a critical gap in the visual capabilities of current MLLMs. Our analysis further explores potential architectural and training-paradigm factors contributing to this perceptual gap in MLLMs. We open-source HueManity dataset and code to foster further research in improving perceptual robustness of MLLMs. 

**Abstract (ZH)**: 多模态大语言模型在nuance感知任务上的表现限制：HueManity基准的构建与评估 

---
# Vid-SME: Membership Inference Attacks against Large Video Understanding Models 

**Title (ZH)**: Vid-SME：针对大规模视频理解模型的成员推理攻击 

**Authors**: Qi Li, Runpeng Yu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03179)  

**Abstract**: Multimodal large language models (MLLMs) demonstrate remarkable capabilities in handling complex multimodal tasks and are increasingly adopted in video understanding applications. However, their rapid advancement raises serious data privacy concerns, particularly given the potential inclusion of sensitive video content, such as personal recordings and surveillance footage, in their training datasets. Determining improperly used videos during training remains a critical and unresolved challenge. Despite considerable progress on membership inference attacks (MIAs) for text and image data in MLLMs, existing methods fail to generalize effectively to the video domain. These methods suffer from poor scalability as more frames are sampled and generally achieve negligible true positive rates at low false positive rates (TPR@Low FPR), mainly due to their failure to capture the inherent temporal variations of video frames and to account for model behavior differences as the number of frames varies. To address these challenges, we introduce Vid-SME, the first membership inference method tailored for video data used in video understanding LLMs (VULLMs). Vid-SME leverages the confidence of model output and integrates adaptive parameterization to compute Sharma-Mittal entropy (SME) for video inputs. By leveraging the SME difference between natural and temporally-reversed video frames, Vid-SME derives robust membership scores to determine whether a given video is part of the model's training set. Experiments on various self-trained and open-sourced VULLMs demonstrate the strong effectiveness of Vid-SME. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在处理复杂多模态任务方面表现出色，并在视频理解应用中日益受到采用。然而，其快速进步引发了严重的数据隐私担忧，尤其是其训练数据集中可能包含个人录制和监控视频等敏感视频内容。确定训练过程中不当使用的视频仍然是一个关键且未解决的挑战。尽管在多模态大型语言模型（MLLMs）的文本和图像数据上取得了显著进展，现有的方法在视频域中无法有效泛化。这些方法在更多帧被采样时缺乏可扩展性，通常在低正假率时实现几乎可以忽略的真正阳性率（TPR@Low FPR），主要原因是它们无法捕捉视频帧的固有时间变化，并且无法考虑帧数量变化时模型行为的差异。为解决这些问题，我们提出Vid-SME，这是第一个专门针对用于视频理解的大型语言模型（VULLMs）的视频数据的成员身份推理方法。Vid-SME 利用模型输出的信心，并结合自适应参数化来计算视频输入的沙尔马-米特拉熵（SME）。通过利用自然视频帧和时序反转视频帧之间的SME差异，Vid-SME .derives robust membership scores以确定给定视频是否属于模型的训练集。在各种自我训练和开源的VULLMs上的实验表明，Vid-SME具有很强的有效性。 

---
# LLaMA-XR: A Novel Framework for Radiology Report Generation using LLaMA and QLoRA Fine Tuning 

**Title (ZH)**: LLaMA-XR：一种基于LLaMA和QLoRA微调的新型放射报告生成框架 

**Authors**: Md. Zihad Bin Jahangir, Muhammad Ashad Kabir, Sumaiya Akter, Israt Jahan, Minh Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.03178)  

**Abstract**: Automated radiology report generation holds significant potential to reduce radiologists' workload and enhance diagnostic accuracy. However, generating precise and clinically meaningful reports from chest radiographs remains challenging due to the complexity of medical language and the need for contextual understanding. Existing models often struggle with maintaining both accuracy and contextual relevance. In this paper, we present LLaMA-XR, a novel framework that integrates LLaMA 3.1 with DenseNet-121-based image embeddings and Quantized Low-Rank Adaptation (QLoRA) fine-tuning. LLaMA-XR achieves improved coherence and clinical accuracy while maintaining computational efficiency. This efficiency is driven by an optimization strategy that enhances parameter utilization and reduces memory overhead, enabling faster report generation with lower computational resource demands. Extensive experiments conducted on the IU X-ray benchmark dataset demonstrate that LLaMA-XR outperforms a range of state-of-the-art methods. Our model achieves a ROUGE-L score of 0.433 and a METEOR score of 0.336, establishing new performance benchmarks in the domain. These results underscore LLaMA-XR's potential as an effective and efficient AI system for automated radiology reporting, offering enhanced clinical utility and reliability. 

**Abstract (ZH)**: 自动放射学报告生成在减轻放射科医生工作负荷和提高诊断准确性方面具有显著潜力。然而，由于医学语言的复杂性和需要上下文理解，从胸部X光片生成精确且临床相关的报告仍具有挑战性。现有模型往往难以在同一时间保持准确性和上下文相关性。本文提出了一种名为LLaMA-XR的新型框架，该框架将LLaMA 3.1与基于DenseNet-121的图像嵌入和量化低秩适应（QLoRA）微调相结合。LLaMA-XR在保持计算效率的同时实现了更好的连贯性和临床准确性。这种效率提升是通过优化策略增强参数利用并减少内存开销来实现的，从而实现了更快的报告生成并降低了计算资源需求。在IU X射线基准数据集上的广泛实验表明，LLaMA-XR在多种最先进的方法中表现出色。我们的模型获得了ROUGE-L评分为0.433和METEOR评分为0.336的新性能基准，这些结果凸显了LLaMA-XR作为自动化放射学报告有效且高效的AI系统的潜在价值，提供了增强的临床实用性和可靠性。 

---
# LLM Code Customization with Visual Results: A Benchmark on TikZ 

**Title (ZH)**: LLM代码自定义与视觉结果：基于TikZ的基准测试 

**Authors**: Charly Reux, Mathieu Acher, Djamel Eddine Khelladi, Olivier Barais, Clément Quinton  

**Link**: [PDF](https://arxiv.org/pdf/2505.04670)  

**Abstract**: With the rise of AI-based code generation, customizing existing code out of natural language instructions to modify visual results -such as figures or images -has become possible, promising to reduce the need for deep programming expertise. However, even experienced developers can struggle with this task, as it requires identifying relevant code regions (feature location), generating valid code variants, and ensuring the modifications reliably align with user intent. In this paper, we introduce vTikZ, the first benchmark designed to evaluate the ability of Large Language Models (LLMs) to customize code while preserving coherent visual outcomes. Our benchmark consists of carefully curated vTikZ editing scenarios, parameterized ground truths, and a reviewing tool that leverages visual feedback to assess correctness. Empirical evaluation with stateof-the-art LLMs shows that existing solutions struggle to reliably modify code in alignment with visual intent, highlighting a gap in current AI-assisted code editing approaches. We argue that vTikZ opens new research directions for integrating LLMs with visual feedback mechanisms to improve code customization tasks in various domains beyond TikZ, including image processing, art creation, Web design, and 3D modeling. 

**Abstract (ZH)**: 基于AI的代码生成兴起后，从自然语言指令定制现有代码以修改视觉结果（如图形或图像）已成为可能，有望减少对深度编程专业スキル的需求。然而，即使经验丰富的开发者在执行此任务时也会遇到困难，因为这需要识别相关代码区域（特征定位）、生成有效的代码变体，并确保修改可靠地符合用户意图。在本文中，我们介绍了vTikZ，这是首个用于评估大型语言模型（LLMs）在定制代码同时保持一致视觉结果方面能力的基准。该基准包括精心策划的vTikZ编辑场景、参数化的ground truth以及一个利用视觉反馈进行评估的工具。实证研究表明，现有解决方案难以可靠地根据视觉意图修改代码，突显出当前AI辅助代码编辑方法的差距。我们认为，vTikZ为将LLMs与视觉反馈机制整合以改进各种领域（如TikZ之外的图像处理、艺术创作、Web设计和3D建模）的代码定制任务开辟了新的研究方向。 

---
