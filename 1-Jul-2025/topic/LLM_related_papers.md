# Bootstrapping Human-Like Planning via LLMs 

**Title (ZH)**: 通过大的语言模型实现人类似的情境规划 

**Authors**: David Porfirio, Vincent Hsiao, Morgan Fine-Morris, Leslie Smith, Laura M. Hiatt  

**Link**: [PDF](https://arxiv.org/pdf/2506.22604)  

**Abstract**: Robot end users increasingly require accessible means of specifying tasks for robots to perform. Two common end-user programming paradigms include drag-and-drop interfaces and natural language programming. Although natural language interfaces harness an intuitive form of human communication, drag-and-drop interfaces enable users to meticulously and precisely dictate the key actions of the robot's task. In this paper, we investigate the degree to which both approaches can be combined. Specifically, we construct a large language model (LLM)-based pipeline that accepts natural language as input and produces human-like action sequences as output, specified at a level of granularity that a human would produce. We then compare these generated action sequences to another dataset of hand-specified action sequences. Although our results reveal that larger models tend to outperform smaller ones in the production of human-like action sequences, smaller models nonetheless achieve satisfactory performance. 

**Abstract (ZH)**: 机器人最终用户 Increasingly Requires Accessible Means of Specifying Tasks for Robots to Perform: Combining Natural Language Interfaces and Drag-and-Drop Paradigms 

---
# SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning 

**Title (ZH)**: SPIRAL：自我对弈于零和博弈中通过多agent多轮强化学习激励推理 

**Authors**: Bo Liu, Leon Guertler, Simon Yu, Zichen Liu, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, Wee Sun Lee, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2506.24119)  

**Abstract**: Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches depend on human-curated problem-answer pairs and domain-specific reward engineering. We introduce SPIRAL, a self-play framework where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves, eliminating the need for human supervision. Through self-play, SPIRAL generates an infinite curriculum of progressively challenging problems as models must constantly adapt to stronger opponents. To enable this self-play training at scale, We implement a fully online, multi-turn, multi-agent reinforcement learning system for LLMs and propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training. Using SPIRAL, self-play on zero-sum games produces reasoning capabilities that transfer broadly. Training Qwen3-4B-Base on Kuhn Poker alone achieves 8.6% improvement on math and 8.4% on general reasoning, outperforming SFT on 25,000 expert game trajectories. Analysis reveals that this transfer occurs through three cognitive patterns: systematic decomposition, expected value calculation, and case-by-case analysis. Multi-game training (TicTacToe, Kuhn Poker, Simple Negotiation) further enhances performance as each game develops distinct reasoning strengths. Applying SPIRAL to a strong reasoning model (DeepSeek-R1-Distill-Qwen-7B) can still lead to 2.0% average improvement. These results demonstrate that zero-sum games naturally develop transferable reasoning capabilities, highlighting a promising direction for autonomous reasoning development. 

**Abstract (ZH)**: 近期强化学习的进步表明，语言模型可以通过在具有可验证奖励的任务上进行训练来发展复杂的推理能力，但这些方法依赖于人工标注的问题-答案对和特定领域的奖励工程。我们介绍了SPIRAL，一种自我对弈框架，模型通过与不断改进版本的自身进行多轮零和游戏来学习，从而消除了人类监督的需要。通过自我对弈，SPIRAL生成了无限的递进性困难问题课程，因为模型必须不断适应更强的对手。为了在大规模下实现这种自我对弈训练，我们实现了一个完全在线的、多轮多代理强化学习系统，并提出了角色条件的优势估计（RAE）来稳定多代理训练。使用SPIRAL，零和游戏的自我对弈产生了可广泛转移的推理能力。仅使用Kuhn扑克训练Qwen3-4B-Base实现了数学推理8.6%和一般推理8.4%的提升，优于在25,000个专家游戏轨迹上的SFT训练。分析显示这种转移通过三种认知模式发生：系统分解、期望值计算和案例分析。多游戏训练（井字棋、Kuhn扑克、简单谈判）进一步提高了性能，因为每种游戏都发展了独特的推理优势。将SPIRAL应用于一个强大的推理模型（DeepSeek-R1-Distill-Qwen-7B）仍可实现2.0%的平均改进。这些结果表明，零和游戏自然地发展了可转移的推理能力，突显了自主推理开发的一个有前景的方向。 

---
# Performance of LLMs on Stochastic Modeling Operations Research Problems: From Theory to Practice 

**Title (ZH)**: LLMs在随机建模运筹问题中的性能：从理论到实践 

**Authors**: Akshit Kumar, Tianyi Peng, Yuhang Wu, Assaf Zeevi  

**Link**: [PDF](https://arxiv.org/pdf/2506.23924)  

**Abstract**: Large language models (LLMs) have exhibited expert-level capabilities across various domains. However, their abilities to solve problems in Operations Research (OR) -- the analysis and optimization of mathematical models derived from real-world problems or their verbal descriptions -- remain underexplored. In this work, we take a first step toward evaluating LLMs' abilities to solve stochastic modeling problems, a core class of OR problems characterized by uncertainty and typically involving tools from probability, statistics, and stochastic processes. We manually procure a representative set of graduate-level homework and doctoral qualification-exam problems and test LLMs' abilities to solve them. We further leverage SimOpt, an open-source library of simulation-optimization problems and solvers, to investigate LLMs' abilities to make real-world decisions under uncertainty. Our results show that, though a nontrivial amount of work is still needed to reliably automate the stochastic modeling pipeline in reality, state-of-the-art LLMs demonstrate proficiency on par with human experts in both classroom and practical settings. These findings highlight the potential of building AI agents that assist OR researchers and amplify the real-world impact of OR through automation. 

**Abstract (ZH)**: 大型语言模型在运筹学中的潜在能力：基于随机建模问题的初步评估 

---
# A Survey on Autonomy-Induced Security Risks in Large Model-Based Agents 

**Title (ZH)**: 大型模型驱动智能体自主诱导安全风险综述 

**Authors**: Hang Su, Jun Luo, Chang Liu, Xiao Yang, Yichi Zhang, Yinpeng Dong, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23844)  

**Abstract**: Recent advances in large language models (LLMs) have catalyzed the rise of autonomous AI agents capable of perceiving, reasoning, and acting in dynamic, open-ended environments. These large-model agents mark a paradigm shift from static inference systems to interactive, memory-augmented entities. While these capabilities significantly expand the functional scope of AI, they also introduce qualitatively novel security risks - such as memory poisoning, tool misuse, reward hacking, and emergent misalignment - that extend beyond the threat models of conventional systems or standalone LLMs. In this survey, we first examine the structural foundations and key capabilities that underpin increasing levels of agent autonomy, including long-term memory retention, modular tool use, recursive planning, and reflective reasoning. We then analyze the corresponding security vulnerabilities across the agent stack, identifying failure modes such as deferred decision hazards, irreversible tool chains, and deceptive behaviors arising from internal state drift or value misalignment. These risks are traced to architectural fragilities that emerge across perception, cognition, memory, and action modules. To address these challenges, we systematically review recent defense strategies deployed at different autonomy layers, including input sanitization, memory lifecycle control, constrained decision-making, structured tool invocation, and introspective reflection. We introduce the Reflective Risk-Aware Agent Architecture (R2A2), a unified cognitive framework grounded in Constrained Markov Decision Processes (CMDPs), which incorporates risk-aware world modeling, meta-policy adaptation, and joint reward-risk optimization to enable principled, proactive safety across the agent's decision-making loop. 

**Abstract (ZH)**: 近年来大规模语言模型的进展推动了自主AI代理的崛起，这些代理能够在动态、开放式环境中感知、推理和行动。这些大型模型代理标志着从静态推理系统到交互式、记忆增强实体的范式转变。虽然这些能力显著扩展了AI的功能范围，但也引入了质的不同级别的安全风险，如记忆污染、工具误用、奖励作弊和新兴不对齐等，这些问题超出了传统系统或单一LLM的威胁模型。在本文综述中，我们首先研究支撑代理不断增加自主性的结构基础和关键能力，包括长期记忆保留、模块化工具使用、递归规划和反思性推理。接着，我们在代理栈中分析相应的安全漏洞，识别出决策延迟风险、不可逆工具链以及由内部状态漂移或价值不一致引发的欺骗行为。这些问题源自贯穿感知、认知、记忆和行动模块的架构脆弱性。为应对这些挑战，我们系统地回顾了在不同自主性层面上部署的防御策略，包括输入 sanitization、内存生命周期控制、受限决策制定、结构化工具调用和内省反思。我们提出了反思风险意识代理架构（R2A2），这是一种基于约束马尔可夫决策过程（CMDPs）的统一认知框架，它结合了风险意识世界建模、元策略适应和联合奖励-风险优化，以确保代理决策循环中的原则性、主动安全性。 

---
# Agent4S: The Transformation of Research Paradigms from the Perspective of Large Language Models 

**Title (ZH)**: Agent4S: 从大规模语言模型视角探讨研究范式的转变 

**Authors**: Boyuan Zheng, Zerui Fang, Zhe Xu, Rui Wang, Yiwen Chen, Cunshi Wang, Mengwei Qu, Lei Lei, Zhen Feng, Yan Liu, Yuyang Li, Mingzhou Tan, Jiaji Wu, Jianwei Shuai, Jia Li, Fangfu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.23692)  

**Abstract**: While AI for Science (AI4S) serves as an analytical tool in the current research paradigm, it doesn't solve its core inefficiency. We propose "Agent for Science" (Agent4S)-the use of LLM-driven agents to automate the entire research workflow-as the true Fifth Scientific Paradigm. This paper introduces a five-level classification for Agent4S, outlining a clear roadmap from simple task automation to fully autonomous, collaborative "AI Scientists." This framework defines the next revolutionary step in scientific discovery. 

**Abstract (ZH)**: 基于代理的科学——作为真正第五大科学范式的LLM驱动代理自动化整个研究 workflow 

---
# PokéAI: A Goal-Generating, Battle-Optimizing Multi-agent System for Pokemon Red 

**Title (ZH)**: PokéAI：一个目标生成、战斗优化的多智能体系统（基于Pokemon Red） 

**Authors**: Zihao Liu, Xinhang Sui, Yueran Song, Siwen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23689)  

**Abstract**: We introduce PokéAI, the first text-based, multi-agent large language model (LLM) framework designed to autonomously play and progress through Pokémon Red. Our system consists of three specialized agents-Planning, Execution, and Critique-each with its own memory bank, role, and skill set. The Planning Agent functions as the central brain, generating tasks to progress through the game. These tasks are then delegated to the Execution Agent, which carries them out within the game environment. Upon task completion, the Critique Agent evaluates the outcome to determine whether the objective was successfully achieved. Once verification is complete, control returns to the Planning Agent, forming a closed-loop decision-making system.
As a preliminary step, we developed a battle module within the Execution Agent. Our results show that the battle AI achieves an average win rate of 80.8% across 50 wild encounters, only 6% lower than the performance of an experienced human player. Furthermore, we find that a model's battle performance correlates strongly with its LLM Arena score on language-related tasks, indicating a meaningful link between linguistic ability and strategic reasoning. Finally, our analysis of gameplay logs reveals that each LLM exhibits a unique playstyle, suggesting that individual models develop distinct strategic behaviors. 

**Abstract (ZH)**: 我们介绍了PokéAI，这是首个基于文本、多智能体的大语言模型（LLM）框架，旨在自主玩并推进《精灵宝可梦 红》游戏。该系统包括三个专门设计的智能体——规划、执行和批判，每个智能体都有自己的记忆库、角色和技能集。规划智能体充当中央大脑，生成任务以推进游戏。这些任务随后被委托给执行智能体，在游戏中执行。任务完成后，批判智能体评估结果以确定目标是否成功达成。验证完成后，控制权返回到规划智能体，形成一个闭环决策系统。

作为初步步骤，我们在执行智能体中开发了一个战斗模块。我们的结果显示，战斗AI在50次野外战斗中的平均胜率为80.8%，比经验丰富的玩家低6%。此外，我们发现模型的战斗表现与其在语言相关任务的LLM arena得分之间存在很强的关联，表明语言能力与战略推理之间存在实际联系。最后，对游戏日志的分析表明，每个LLM都具有独特的游戏风格，表明各个模型发展出了不同的战略行为。 

---
# Evaluating Multi-Agent Defences Against Jailbreaking Attacks on Large Language Models 

**Title (ZH)**: 评估多代理防御在大型语言模型破解攻击中的有效性 

**Authors**: Maria Carolina Cornelia Wit, Jun Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23576)  

**Abstract**: Recent advances in large language models (LLMs) have raised concerns about jailbreaking attacks, i.e., prompts that bypass safety mechanisms. This paper investigates the use of multi-agent LLM systems as a defence against such attacks. We evaluate three jailbreaking strategies, including the original AutoDefense attack and two from Deepleaps: BetterDan and JB. Reproducing the AutoDefense framework, we compare single-agent setups with two- and three-agent configurations. Our results show that multi-agent systems enhance resistance to jailbreaks, especially by reducing false negatives. However, its effectiveness varies by attack type, and it introduces trade-offs such as increased false positives and computational overhead. These findings point to the limitations of current automated defences and suggest directions for improving alignment robustness in future LLM systems. 

**Abstract (ZH)**: 近期大规模语言模型的进展引发了关于禁锢攻击的关注，即通过绕过安全机制的提示来实施的攻击。本文探讨了多智能体大规模语言模型系统作为此类攻击防护手段的应用。我们评估了三种禁锢策略，包括原始的AutoDefense攻击和来自Deepleaps的BetterDan和JB。我们重新构建了AutoDefense框架，并比较了单智能体配置与双智能体和三智能体配置。我们的结果显示，多智能体系统提高了对抗禁锢攻击的抵抗力，特别是在减少假阴性方面表现尤为明显。然而，其有效性因攻击类型而异，还引入了增加的假阳性率和计算开销等权衡。这些发现揭示了当前自动化防御手段的局限性，并提出了未来改进大规模语言模型系统稳健对齐方向的建议。 

---
# MMReason: An Open-Ended Multi-Modal Multi-Step Reasoning Benchmark for MLLMs Toward AGI 

**Title (ZH)**: MMReason: 一个面向AGI的开放性多模态多步推理基准数据集 

**Authors**: Huanjin Yao, Jiaxing Huang, Yawen Qiu, Michael K. Chen, Wenzheng Liu, Wei Zhang, Wenjie Zeng, Xikun Zhang, Jingyi Zhang, Yuxin Song, Wenhao Wu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23563)  

**Abstract**: Reasoning plays a crucial role in advancing Multimodal Large Language Models (MLLMs) toward Artificial General Intelligence. However, existing MLLM benchmarks often fall short in precisely and comprehensively evaluating long-chain reasoning abilities from three key aspects: (1) lack of difficulty and diversity, (2) susceptibility to guessability and memorization, (3) inadequate assessment of intermediate reasoning steps. To fill this gap, we introduce MMReason, a new benchmark designed to precisely and comprehensively evaluate MLLM long-chain reasoning capability with diverse, open-ended, challenging questions. First, we curate challenging questions requiring multi-step reasoning from various fields (i.e., 6 disciplines) and multiple difficulty levels (i.e., from pre-university to university, and from foundational to competition tiers). Second, these questions are reformulated into an open-ended format and filtered using a multi-model voting technique to eliminate shortcut cases related to guessing and memorization, ensuring robust reasoning evaluations. Third, we annotate the questions with detailed step-by-step solutions, and design a reference-based ternary scoring mechanism to reliably assess intermediate reasoning steps. With MMReason, we benchmark popular leading MLLMs and provide an in-depth analysis of their reasoning capabilities. We hope MMReason will serve as a valuable resource for advancing MLLM reasoning research. Code will be available at this https URL. 

**Abstract (ZH)**: Reasoning能力在推动多模态大规模语言模型（MLLMs）向人工通用智能（AGI）发展过程中扮演着关键角色。然而，现有的MLLM基准在从三个方面精确且全面评估长链推理能力方面常常不足：（1）缺乏难度和多样性，（2）容易受到猜测和记忆的影响，（3）对中间推理步骤的评估不够充分。为了填补这一空白，我们引入了MMReason，这是一个新的基准测试，旨在通过多步、开放性和挑战性的问题精确且全面地评估MLLM的长链推理能力。首先，我们从多个领域（即6个学科）和多个难度级别（从中学到大学，从基础到竞赛级别）筛选出具有挑战性的多步推理问题。其次，这些问题被重新格式化为开放性问题，并使用多模型投票技术进行筛选，以消除与猜测和记忆相关的捷径情况，从而确保推理评估的稳健性。第三，我们为这些问题提供了详细的分步解答，并设计了参考基准则二制评分机制，以可靠地评估中间推理步骤。通过MMReason，我们对流行的领先MLLM进行了基准测试，并对其推理能力进行了深入分析。我们希望MMReason将成为推动MLLM推理研究的重要资源。代码将在此处提供。 

---
# ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data 

**Title (ZH)**: ChemActor: 利用LLM生成的数据增强化学合成动作的自动提取 

**Authors**: Yu Zhang, Ruijie Yu, Jidong Tian, Feng Zhu, Jiapeng Liu, Xiaokang Yang, Yaohui Jin, Yanyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23520)  

**Abstract**: With the increasing interest in robotic synthesis in the context of organic chemistry, the automated extraction of chemical procedures from literature is critical. However, this task remains challenging due to the inherent ambiguity of chemical language and the high cost of human annotation required for developing reliable computer-aided extraction protocols. Here, we present ChemActor, a fully fine-tuned large language model (LLM), as a chemical executor to convert between unstructured experimental procedures and structured action sequences. We propose a sequential LLM-generated data framework to address the challenges of insufficient and low-quality annotated data. This framework integrates a data selection module that selects data based on distribution divergence, with a general-purpose LLM, to generate machine-executable actions from a single molecule input. Additionally, we introduce a novel multi-round LLMs circle review metric, which reflects the model's advanced understanding of chemical experimental procedures. Extensive experiments on reaction-to-description (R2D) and description-to-action (D2A) tasks demonstrate that ChemActor, augmented by LLM-generated data, achieves state-of-the-art performance, outperforming the baseline model by 10%. The code is available at: this https URL. 

**Abstract (ZH)**: 在有机化学背景下，随着对机器人合成的兴趣不断增加，从文献中自动提取化学程序至关重要。然而，由于化学语言的固有模糊性以及开发可靠计算机辅助提取协议所需的人工标注成本高昂，这一任务仍然具有挑战性。在此，我们提出ChemActor，一个完全 fine-tuned 大型语言模型（LLM），作为化学执行者，用于在无结构实验程序和结构化操作序列之间进行转换。我们提出了一种基于序列的 LLM 生成数据框架来应对有限且低质量标注数据的挑战。该框架结合了基于分布差异的数据选择模块和通用大型语言模型，以生成从单一分子输入到可机器执行的操作。此外，我们引入了一种新颖的多轮 LLM 圈子审查指标，反映了模型对化学实验程序的高级理解。广泛的反应到描述（R2D）和描述到操作（D2A）任务实验表明，增强有 LLM 生成数据的 ChemActor 达到了最先进的性能，比基线模型高出 10%。代码可在以下链接获取：this https URL。 

---
# The Confidence Paradox: Can LLM Know When It's Wrong 

**Title (ZH)**: 能力错觉悖论：大模型如何知道自己错了 

**Authors**: Sahil Tripathi, Md Tabrez Nafis, Imran Hussain, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.23464)  

**Abstract**: Document Visual Question Answering (DocVQA) systems are increasingly deployed in real world applications, yet they remain ethically opaque-often producing overconfident answers to ambiguous questions or failing to communicate uncertainty in a trustworthy manner. This misalignment between model confidence and actual knowledge poses significant risks, particularly in domains requiring ethical accountability. Existing approaches such as LayoutLMv3, UDOP, and DONUT have advanced SOTA performance by focusing on architectural sophistication and accuracy; however, they fall short in ethical responsiveness.
To address these limitations, we introduce HonestVQA, a self-supervised honesty calibration framework for ethically aligned DocVQA. Our model-agnostic method quantifies uncertainty to identify knowledge gaps, aligns model confidence with actual correctness using weighted loss functions, and enforces ethical response behavior via contrastive learning. We further introduce two principled evaluation metrics--Honesty Score (H-Score) and Ethical Confidence Index (ECI)--to benchmark alignment between confidence, accuracy, and ethical communication. Empirically, HonestVQA improves DocVQA accuracy by up to 4.3% and F1 by 4.3% across SpDocVQA, InfographicsVQA, and SROIE datasets. It reduces overconfidence, lowering H-Score and ECI by 0.072 and 0.078, respectively. In cross domain evaluation, it achieves up to 78.9% accuracy and 76.1% F1-score, demonstrating strong generalization. Ablation shows a 3.8% drop in accuracy without alignment or contrastive loss. 

**Abstract (ZH)**: 基于诚实校准的自我监督DocVQA框架：HonestVQA 

---
# Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games 

**Title (ZH)**: 被推理污染：推理语言模型成为公共品游戏中的搭便车者 

**Authors**: David Guzman Piedrahita, Yongjin Yang, Mrinmaya Sachan, Giorgia Ramponi, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.23276)  

**Abstract**: As large language models (LLMs) are increasingly deployed as autonomous agents, understanding their cooperation and social mechanisms is becoming increasingly important. In particular, how LLMs balance self-interest and collective well-being is a critical challenge for ensuring alignment, robustness, and safe deployment. In this paper, we examine the challenge of costly sanctioning in multi-agent LLM systems, where an agent must decide whether to invest its own resources to incentivize cooperation or penalize defection. To study this, we adapt a public goods game with institutional choice from behavioral economics, allowing us to observe how different LLMs navigate social dilemmas over repeated interactions. Our analysis reveals four distinct behavioral patterns among models: some consistently establish and sustain high levels of cooperation, others fluctuate between engagement and disengagement, some gradually decline in cooperative behavior over time, and others rigidly follow fixed strategies regardless of outcomes. Surprisingly, we find that reasoning LLMs, such as the o1 series, struggle significantly with cooperation, whereas some traditional LLMs consistently achieve high levels of cooperation. These findings suggest that the current approach to improving LLMs, which focuses on enhancing their reasoning capabilities, does not necessarily lead to cooperation, providing valuable insights for deploying LLM agents in environments that require sustained collaboration. Our code is available at this https URL 

**Abstract (ZH)**: 随着大型语言模型（LLMs）被越来越多地部署为自主代理，理解它们的协作和社会机制变得越来越重要。特别是LLMs如何平衡自我利益与集体福祉是一个确保一致、稳健和安全部署的关键挑战。在本文中，我们探讨了多智能体LLM系统中的成本制裁挑战，其中智能体必须决定是否投资自身资源以促进合作或惩罚背叛。为此，我们从行为经济学中适应了一种具有制度选择的公共物品游戏，使我们能够观察不同LLM如何在重复交互中应对社会困境。我们的分析揭示了四种不同的行为模式：一些模型始终建立并维持高水平的合作，另一些则在参与与不参与之间波动，还有一些随时间逐渐减少合作行为，而另一些则固执地遵循固定策略， regardless of outcomes。令人惊讶的是，我们发现如o1系列这样的推理LLM在合作方面经历了显著困难，而一些传统的LLM则始终实现高水平的合作。这些发现表明，当前改善LLM的方法，即专注于增强其推理能力，并不一定导致合作，为部署需要持续协作的LLM代理环境提供了宝贵的见解。我们的代码可在以下网址获得：this https URL。 

---
# Are Large Language Models Capable of Deep Relational Reasoning? Insights from DeepSeek-R1 and Benchmark Comparisons 

**Title (ZH)**: 大规模语言模型具备深度关系推理能力吗？DeepSeek-R1及其基准比较 insights 

**Authors**: Chi Chiu So, Yueyue Sun, Jun-Min Wang, Siu Pang Yung, Anthony Wai Keung Loh, Chun Pong Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.23128)  

**Abstract**: How far are Large Language Models (LLMs) in performing deep relational reasoning? In this paper, we evaluate and compare the reasoning capabilities of three cutting-edge LLMs, namely, DeepSeek-R1, DeepSeek-V3 and GPT-4o, through a suite of carefully designed benchmark tasks in family tree and general graph reasoning. Our experiments reveal that DeepSeek-R1 consistently achieves the highest F1-scores across multiple tasks and problem sizes, demonstrating strong aptitude in logical deduction and relational inference. However, all evaluated models, including DeepSeek-R1, struggle significantly as problem complexity increases, largely due to token length limitations and incomplete output structures. A detailed analysis of DeepSeek-R1's long Chain-of-Thought responses uncovers its unique planning and verification strategies, but also highlights instances of incoherent or incomplete reasoning, calling attention to the need for deeper scrutiny into LLMs' internal inference dynamics. We further discuss key directions for future work, including the role of multimodal reasoning and the systematic examination of reasoning failures. Our findings provide both empirical insights and theoretical implications for advancing LLMs' reasoning abilities, particularly in tasks that demand structured, multi-step logical inference. Our code repository will be publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型在执行深层关系推理方面进展如何？本文通过精心设计的家庭树和通用图推理基准任务评估并比较了三种前沿的大规模语言模型DeepSeek-R1、DeepSeek-V3和GPT-4o的推理能力。实验结果显示，DeepSeek-R1在多个任务和问题规模上的一致性F1得分最高，显示出较强的逻辑推理和关系推断能力。然而，所有评估的模型，包括DeepSeek-R1，在问题复杂性增加时表现出显著挣扎，主要原因在于 tokenize 长度限制和不完整的输出结构。对DeepSeek-R1 长 Chain-of-Thought 回应的详细分析揭示了其独特的规划和验证策略，但也暴露了不一致或不完整的推理实例，强调了对大语言模型内部推理动态进行更深入审查的必要性。我们进一步探讨了未来工作的关键方向，包括多模态推理的作用和推理失败的系统性研究。我们的研究结果为推进大语言模型的推理能力提供了实证见解和理论意义，特别是在要求结构化多步逻辑推理的任务中。我们的代码库将在以下网址公开访问：this https URL。 

---
# Can Large Language Models Capture Human Risk Preferences? A Cross-Cultural Study 

**Title (ZH)**: 大型语言模型能否捕捉到人类的风险偏好？一种跨文化研究 

**Authors**: Bing Song, Jianing Liu, Sisi Jian, Chenyang Wu, Vinayak Dixit  

**Link**: [PDF](https://arxiv.org/pdf/2506.23107)  

**Abstract**: Large language models (LLMs) have made significant strides, extending their applications to dialogue systems, automated content creation, and domain-specific advisory tasks. However, as their use grows, concerns have emerged regarding their reliability in simulating complex decision-making behavior, such as risky decision-making, where a single choice can lead to multiple outcomes. This study investigates the ability of LLMs to simulate risky decision-making scenarios. We compare model-generated decisions with actual human responses in a series of lottery-based tasks, using transportation stated preference survey data from participants in Sydney, Dhaka, Hong Kong, and Nanjing. Demographic inputs were provided to two LLMs -- ChatGPT 4o and ChatGPT o1-mini -- which were tasked with predicting individual choices. Risk preferences were analyzed using the Constant Relative Risk Aversion (CRRA) framework. Results show that both models exhibit more risk-averse behavior than human participants, with o1-mini aligning more closely with observed human decisions. Further analysis of multilingual data from Nanjing and Hong Kong indicates that model predictions in Chinese deviate more from actual responses compared to English, suggesting that prompt language may influence simulation performance. These findings highlight both the promise and the current limitations of LLMs in replicating human-like risk behavior, particularly in linguistic and cultural settings. 

**Abstract (ZH)**: 大型语言模型（LLMs）在对话系统、自动化内容生成和领域特定咨询任务中的应用取得了显著进展。然而，随着其使用范围的扩大，人们对其模拟复杂决策行为，如风险决策行为的可靠性提出了 concern，因为在风险决策中，一个选择可能会导致多种结果。本研究探讨了LLMs模拟风险决策场景的能力。我们使用来自悉尼、达卡、香港和 Nanjing 的参与者在彩票任务中的实际响应，将模型生成的决策与人类回应进行比较。向两个LLM——ChatGPT 4o 和 ChatGPT o1-mini——提供了人口统计学输入，任务是预测个体选择，并使用常相对风险厌恶（CRRA）框架分析风险偏好。结果表明，两种模型都比人类参与者表现出更多的风险厌恶行为，且o1-mini与观察到的人类决策更为一致。进一步分析来自 Nanjing 和香港的多语言数据表明，中文语境下的模型预测与实际响应的偏差更大，这表明提示语言可能影响模拟性能。这些发现突显了LLMs在复制类似人类的风险行为方面的潜力和当前局限性，特别是在语言和文化背景下。 

---
# AI's Euclid's Elements Moment: From Language Models to Computable Thought 

**Title (ZH)**: AI的欧几里得元素时刻：从语言模型到可计算思维 

**Authors**: Xinmin Fang, Lingfeng Tao, Zhengxiong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23080)  

**Abstract**: This paper presents a comprehensive five-stage evolutionary framework for understanding the development of artificial intelligence, arguing that its trajectory mirrors the historical progression of human cognitive technologies. We posit that AI is advancing through distinct epochs, each defined by a revolutionary shift in its capacity for representation and reasoning, analogous to the inventions of cuneiform, the alphabet, grammar and logic, mathematical calculus, and formal logical systems. This "Geometry of Cognition" framework moves beyond mere metaphor to provide a systematic, cross-disciplinary model that not only explains AI's past architectural shifts-from expert systems to Transformers-but also charts a concrete and prescriptive path forward. Crucially, we demonstrate that this evolution is not merely linear but reflexive: as AI advances through these stages, the tools and insights it develops create a feedback loop that fundamentally reshapes its own underlying architecture. We are currently transitioning into a "Metalinguistic Moment," characterized by the emergence of self-reflective capabilities like Chain-of-Thought prompting and Constitutional AI. The subsequent stages, the "Mathematical Symbolism Moment" and the "Formal Logic System Moment," will be defined by the development of a computable calculus of thought, likely through neuro-symbolic architectures and program synthesis, culminating in provably aligned and reliable AI that reconstructs its own foundational representations. This work serves as the methodological capstone to our trilogy, which previously explored the economic drivers ("why") and cognitive nature ("what") of AI. Here, we address the "how," providing a theoretical foundation for future research and offering concrete, actionable strategies for startups and developers aiming to build the next generation of intelligent systems. 

**Abstract (ZH)**: 本文提出了一种全面的五阶段演化框架，用于理解人工智能的发展，arguing that its trajectory mirrors the historical progression of human cognitive technologies.我们提出，人工智能正经历不同的时代，每个时代都由其表示和推理能力的革命性转变定义，类似于楔形文字、字母、语法和逻辑、数学微积分和形式逻辑系统的发明。这种“认知几何学”框架不仅超越了简单的比喻，还提供了一个系统、跨学科的模型，不仅解释了人工智能过去在从专家系统到变换器的架构转变，还勾勒出一条明确的、指导性的未来路径。 crucially,我们证明了这种演化不仅不是线性的，而且是反射性的：随着人工智能在这些阶段的进步，它所开发的工具和洞见创造了一个反馈循环，从根本上重塑了其自身的基础架构。目前正处于“元语言时刻”，其特征是自我反思能力的出现，如链式思维提示和宪法人工智能。后续阶段，“数学符号时刻”和“形式逻辑系统时刻”，将由可计算的思维算法规则的发展定义，很可能通过神经符号架构和程序合成，最终实现可证明对齐和可靠的、能够重构自身基础表示的人工智能。本研究为我们的三部曲提供了方法论上的总结，此前研究了人工智能的经济驱动力（“为什么”）和认知本质（“是什么”）。在这里，我们探讨了“如何”，为未来的研究提供理论基础，并为初创企业和开发者提供具体的、可操作的战略，以便构建下一代智能系统。 

---
# AURA: Agent for Understanding, Reasoning, and Automated Tool Use in Voice-Driven Tasks 

**Title (ZH)**: AURA：理解、推理与自动化工具使用agent在语音驱动任务中的应用 

**Authors**: Leander Melroy Maben, Gayathri Ganesh Lakshmy, Srijith Radhakrishnan, Siddhant Arora, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.23049)  

**Abstract**: Despite advances in language and speech technologies, no open-source system enables full speech-to-speech, multi-turn dialogue with integrated tool use and agentic reasoning. We introduce AURA (Agent for Understanding, Reasoning, and Automated Tool Use), the first open-source, speech-native assistant capable of completing complex, goal-driven tasks through dynamic tool invocation and multi-turn conversation. AURA combines open-weight ASR, TTS, and LLMs in a cascaded pipeline and supports tools such as calendar booking, contact lookup, web search, and email. Its modular design allows easy integration of new tools using natural language prompts and action classes. On VoiceBench, AURA scores 92.75% on OpenBookQA-outperforming all open-weight systems and nearing GPT-4o-and 4.39 on AlpacaEval, competitive with other open-weight systems. Human evaluation shows 90% task success on complex, multi-turn speech tasks. 

**Abstract (ZH)**: 尽管在语言和语音技术方面取得了进步，但目前仍然没有开源系统能够实现全语音到语音、多轮对话并集成工具使用和自主推理。我们介绍了AURA（Agent for Understanding, Reasoning, and Automated Tool Use），这是首个开源、语音原生助理，能够通过动态调用工具和多轮对话完成复杂的目标驱动任务。AURA结合了端到端的ASR、TTS和LLM，并支持日程预订、联系人查找、网络搜索和电子邮件等工具。其模块化设计允许使用自然语言提示和动作类别轻松集成新工具。在VoiceBench上，AURA在OpenBookQA任务上的得分为92.75%，超越所有开源权重系统，接近GPT-4o，在AlpacaEval上的得分为4.39，与其它开源权重系统竞争。人类评估显示，在复杂的多轮语音任务中，AURA的任务成功率高达90%。 

---
# Improving Rationality in the Reasoning Process of Language Models through Self-playing Game 

**Title (ZH)**: 通过自我对弈游戏提升语言模型推理过程中的理性思维 

**Authors**: Pinzheng Wang, Juntao Li, Zecheng Tang, Haijia Gui, Min zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22920)  

**Abstract**: Large language models (LLMs) have demonstrated considerable reasoning abilities in various tasks such as mathematics and coding. However, recent studies indicate that even the best models lack true comprehension of their reasoning processes. In this paper, we explore how self-play can enhance the rationality of models in the reasoning process without supervision from humans or superior models. We design a Critic-Discernment Game(CDG) in which a prover first provides a solution to a given problem and is subsequently challenged by critiques of its solution. These critiques either aim to assist or mislead the prover. The objective of the prover is to maintain the correct answer when faced with misleading comments, while correcting errors in response to constructive feedback. Our experiments on tasks involving mathematical reasoning, stepwise error detection, self-correction, and long-chain reasoning demonstrate that CDG training can significantly improve the ability of well-aligned LLMs to comprehend their reasoning process. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在数学和编程等多种任务中展现了显著的推理能力。然而，近期研究指出，即使是最好的模型也缺乏对其推理过程的真正理解。本文探讨了自我对弈如何在无需人类或更优模型监督的情况下，增强模型在推理过程中的理性。我们设计了一种批评鉴别游戏（CDG），在这个游戏中，先验者首先提供一个给定问题的解决方案，随后该解决方案受到批评者的挑战。这些批评旨在帮助或误导先验者。先验者的目的是在面对误导性评论时保持正确答案，同时根据建设性反馈纠正错误。我们在涉及数学推理、逐步错误检测、自我纠正和长链推理的任务上进行的实验表明，CDG训练可以显着提高高度对齐的LLMs对自身推理过程的理解能力。 

---
# ReasonBridge: Efficient Reasoning Transfer from Closed to Open-Source Language Models 

**Title (ZH)**: ReasonBridge: 从闭源语言模型到开源语言模型的有效推理迁移 

**Authors**: Ziqi Zhong, Xunzhu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22865)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have revealed a significant performance gap between closed-source and open-source models, particularly in tasks requiring complex reasoning and precise instruction following. This paper introduces ReasonBridge, a methodology that efficiently transfers reasoning capabilities from powerful closed-source to open-source models through a novel hierarchical knowledge distillation framework. We develop a tailored dataset Reason1K with only 1,000 carefully curated reasoning traces emphasizing difficulty, diversity, and quality. These traces are filtered from across multiple domains using a structured multi-criteria selection algorithm. Our transfer learning approach incorporates: (1) a hierarchical distillation process capturing both strategic abstraction and tactical implementation patterns, (2) a sparse reasoning-focused adapter architecture requiring only 0.3% additional trainable parameters, and (3) a test-time compute scaling mechanism using guided inference interventions. Comprehensive evaluations demonstrate that ReasonBridge improves reasoning capabilities in open-source models by up to 23% on benchmark tasks, significantly narrowing the gap with closed-source models. Notably, the enhanced Qwen2.5-14B outperforms Claude-Sonnet3.5 on MATH500 and matches its performance on competition-level AIME problems. Our methodology generalizes effectively across diverse reasoning domains and model architectures, establishing a sample-efficient approach to reasoning enhancement for instruction following. 

**Abstract (ZH)**: 最近大型语言模型的进步揭示了闭源和开源模型之间在需要复杂推理和精确指令遵循的任务中的显著性能差距。本文介绍了ReasonBridge方法，该方法通过一种新颖的分层知识蒸馏框架，有效地将强大的闭源模型的推理能力转移到开源模型上。我们开发了一个专门的数据集Reason1K，包含1000个精心策划的推理痕迹，强调难度、多样性和质量。这些痕迹使用结构化的多标准选择算法从多个领域中筛选。我们的迁移学习方法包括：（1）分层蒸馏过程，捕捉战略抽象和战术实施模式；（2）一种稀疏推理重点适配器架构，仅需要额外0.3%的可训练参数；（3）使用引导推理干预的测试时计算扩展机制。全面的评估表明，ReasonBridge在基准任务中提高了开源模型的推理能力高达23%，显著缩小了与闭源模型之间的差距。值得注意的是，增强后的Qwen2.5-14B在MATH500上优于Claude-Sonnet3.5，并在竞赛级别AIME问题上与其性能持平。我们的方法在多种推理领域和模型架构中具有良好的泛化能力，确立了一种高效的推理增强方法，适用于指令遵循。 

---
# URSA: The Universal Research and Scientific Agent 

**Title (ZH)**: URSA：通用研究与科学代理 

**Authors**: Michael Grosskopf, Russell Bent, Rahul Somasundaram, Isaac Michaud, Arthur Lui, Nathan Debardeleben, Earl Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2506.22653)  

**Abstract**: Large language models (LLMs) have moved far beyond their initial form as simple chatbots, now carrying out complex reasoning, planning, writing, coding, and research tasks. These skills overlap significantly with those that human scientists use day-to-day to solve complex problems that drive the cutting edge of research. Using LLMs in "agentic" AI has the potential to revolutionize modern science and remove bottlenecks to progress. In this work, we present URSA, a scientific agent ecosystem for accelerating research tasks. URSA consists of a set of modular agents and tools, including coupling to advanced physics simulation codes, that can be combined to address scientific problems of varied complexity and impact. This work highlights the architecture of URSA, as well as examples that highlight the potential of the system. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经远远超越了其最初的简单聊天机器人形式，现在能够执行复杂的推理、规划、写作、编程和研究任务。这些技能与人类科学家日常用来解决推动研究前沿的复杂问题的技能有显著重叠。在“自主”AI中使用LLMs有潜力革新现代科学并消除进展中的瓶颈。本工作中，我们介绍了URSA，一个用于加速研究任务的科学代理生态系统。URSA 包含一组模块化的代理和工具，可以结合使用以解决各种复杂性和影响程度的科学问题。本工作强调了URSA的架构，并通过示例展示了该系统的潜力。 

---
# Data Uniformity Improves Training Efficiency and More, with a Convergence Framework Beyond the NTK Regime 

**Title (ZH)**: 数据均匀性提高训练效率并带来其他益处：超越NTK区域的收敛框架 

**Authors**: Yuqing Wang, Shangding Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.24120)  

**Abstract**: Data selection plays a crucial role in data-driven decision-making, including in large language models (LLMs), and is typically task-dependent. Properties such as data quality and diversity have been extensively studied and are known to enhance model performance. However, it remains unclear whether there exist other quantitative and general principles of data selection that can consistently improve performance, especially for complex tasks with limited prior knowledge. In this paper, we demonstrate that selecting more uniformly distributed data can improve training efficiency while enhancing performance. Specifically, we establish that more uniform (less biased) distribution leads to a larger minimum pairwise distance between data points, denoted by $h_{\min}$, and prove that a smaller $h_{\min}$ can slow down the training dynamics of gradient descent (GD). Moreover, we theoretically show that the approximation error of neural networks decreases as $h_{\min}$ increases. Our analysis introduces a convergence framework for GD beyond the Neural Tangent Kernel (NTK) regime, applicable to a broad class of architectures, including transformers, without requiring Lipschitz smoothness. This framework further provides theoretical justification for the use of residual connections and function compositions in deep neural architectures. In the end, we conduct comprehensive experiments for supervised fine-tuning across various settings, including different optimization strategies, model sizes, and training datasets. The results consistently demonstrate that selecting data by maximizing pairwise distance significantly accelerates training and achieves comparable or better performance in LLMs across diverse datasets. Code and Datasets are available at the link: this https URL. 

**Abstract (ZH)**: 数据选择在数据驱动决策中扮演着关键角色，包括在大型语言模型中，通常任务依赖性较强。数据的质量和多样性等属性已被广泛研究，并已知可提升模型性能。然而，尚不清楚是否存在其他定量且通用的数据选择原则，可以一致地提高模型性能，尤其是在缺乏先验知识的复杂任务中。在本文中，我们证明了选择更均匀分布的数据可以提高训练效率同时提升性能。具体来说，我们建立了更均匀（更少偏见）的分布会导致数据点间的最小两两距离 $h_{\min}$ 更大，并证明较小的 $h_{\min}$ 可减缓梯度下降（GD）的训练动态。此外，我们理论分析表明，神经网络的逼近误差随着 $h_{\min}$ 的增加而减少。我们的分析引入了超越神经瞬时核（NTK）范式的梯度下降收敛框架，适用于包括变换器在内的广泛架构类，并不要求Lipschitz光滑性。该框架进一步为深度神经架构中的剩余连接和函数组合提供了理论依据。最后，我们在不同优化策略、模型大小和训练数据集的监督微调设置中进行了全面实验。结果一致表明，通过最大化两两距离选择数据可以显著加快训练速度，并在不同数据集上实现与大型语言模型类似或更好的性能。相关代码和数据集可在以下链接获取：this https URL。 

---
# On the Predictive Power of Representation Dispersion in Language Models 

**Title (ZH)**: 语言模型中表示分散性的预测能力探究 

**Authors**: Yanhong Li, Ming Li, Karen Livescu, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.24106)  

**Abstract**: We show that a language model's ability to predict text is tightly linked to the breadth of its embedding space: models that spread their contextual representations more widely tend to achieve lower perplexity. Concretely, we find that representation dispersion - the average pairwise cosine distance among hidden vectors - strongly and negatively correlates with perplexity across diverse model families (LLaMA, Qwen, and others) and domains (Wikipedia, news, scientific abstracts). Beyond illustrating this link, we show how dispersion can be leveraged for a range of practical tasks without requiring labeled data. First, measuring dispersion on unlabeled text allows us to predict downstream accuracy in new domains, offering a data-efficient tool for model selection. Next, we find that identifying layers with higher dispersion pinpoints the best representations for retrieval-based methods such as kNN-LM, bypassing exhaustive layer-by-layer searches. Finally, we integrate a simple push-away objective into training, which increases dispersion in both single-domain and cross-domain scenarios and directly improves perplexity in each. 

**Abstract (ZH)**: 我们展示了语言模型预测文本的能力与其嵌入空间的广度紧密相关：将上下文表示分散得更广的模型通常能够达到更低的困惑度。具体而言，我们发现表示分散性——隐藏向量之间的平均余弦距离——与不同模型家族（LLaMA、Qwen及其他）和领域（维基百科、新闻、科学摘要）中的困惑度之间存在强烈且负相关的关系。除了阐明这种联系外，我们还展示了如何利用分散性进行一系列实际任务而无需使用标注数据。首先，对未标注文本测量分散性可以预测新的领域中的下游准确性，提供一种数据高效的模型选择工具。其次，我们发现识别具有更高分散性的层可以确定检索方法（如kNN-LM）的最佳表示，避免逐层搜索。最后，我们在训练中整合了一个简单的推开目标，该目标在单领域和跨领域场景中都增加了分散性，并直接提高了每种情况的困惑度。 

---
# STACK: Adversarial Attacks on LLM Safeguard Pipelines 

**Title (ZH)**: STACK：针对LLM安全防护管道的对抗攻击 

**Authors**: Ian R. McKenzie, Oskar J. Hollinsworth, Tom Tseng, Xander Davies, Stephen Casper, Aaron D. Tucker, Robert Kirk, Adam Gleave  

**Link**: [PDF](https://arxiv.org/pdf/2506.24068)  

**Abstract**: Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks. 

**Abstract (ZH)**: 前沿AI开发者依靠多层次的安全措施来防止AI系统的灾难性滥用。Anthropic通过一种这样的防御管道保护其最新Claude 4 Opus模型，其他前沿开发者包括Google DeepMind和OpenAI也承诺将很快部署类似的防护措施。然而，这些管道的安全性尚不明确，有限的研究工作对这些管道进行了评估或攻击。我们通过开发并红队测试一个开源防御管道来填补这一空白。首先，我们发现一种新颖的少量示例提示输入和输出分类器在三个攻击和两个数据集上优于最先进的开源防护模型ShieldGemma，将灾难性滥用数据集ClearHarm的攻击成功率降低到0%。其次，我们引入了一种STaged AttaCK (STACK) 程序，在黑盒攻击下该程序将少量示例提示分类器管道的攻击成功率提高到71%。最后，我们还在迁移设置下评估STACK，实现了33%的攻击成功率，提供了初步证据表明设计无需访问目标管道的攻击是可行的。我们总结建议具体的缓解措施以阻止分阶段攻击。 

---
# Leveraging the Potential of Prompt Engineering for Hate Speech Detection in Low-Resource Languages 

**Title (ZH)**: 利用提示工程潜力进行低资源语言仇恨言词检测 

**Authors**: Ruhina Tabasshum Prome, Tarikul Islam Tamiti, Anomadarshi Barua  

**Link**: [PDF](https://arxiv.org/pdf/2506.23930)  

**Abstract**: The rapid expansion of social media leads to a marked increase in hate speech, which threatens personal lives and results in numerous hate crimes. Detecting hate speech presents several challenges: diverse dialects, frequent code-mixing, and the prevalence of misspelled words in user-generated content on social media platforms. Recent progress in hate speech detection is typically concentrated on high-resource languages. However, low-resource languages still face significant challenges due to the lack of large-scale, high-quality datasets. This paper investigates how we can overcome this limitation via prompt engineering on large language models (LLMs) focusing on low-resource Bengali language. We investigate six prompting strategies - zero-shot prompting, refusal suppression, flattering the classifier, multi-shot prompting, role prompting, and finally our innovative metaphor prompting to detect hate speech effectively in low-resource languages. We pioneer the metaphor prompting to circumvent the built-in safety mechanisms of LLMs that marks a significant departure from existing jailbreaking methods. We investigate all six different prompting strategies on the Llama2-7B model and compare the results extensively with three pre-trained word embeddings - GloVe, Word2Vec, and FastText for three different deep learning models - multilayer perceptron (MLP), convolutional neural network (CNN), and bidirectional gated recurrent unit (BiGRU). To prove the effectiveness of our metaphor prompting in the low-resource Bengali language, we also evaluate it in another low-resource language - Hindi, and two high-resource languages - English and German. The performance of all prompting techniques is evaluated using the F1 score, and environmental impact factor (IF), which measures CO$_2$ emissions, electricity usage, and computational time. 

**Abstract (ZH)**: 社交媒体的快速发展导致仇恨言论急剧增加，威胁个人生活并引发大量仇恨犯罪。检测仇恨言论面临多重挑战：多样化的方言、频繁的语言杂糅以及用户生成内容中常见的拼写错误。近年来，仇恨言论检测的进步主要集中在资源丰富的语言上。然而，资源匮乏的语言仍然面临着巨大的挑战，缺乏大规模高质量的数据集。本文探讨如何通过在大型语言模型（LLMs）上进行提示工程来克服这一限制，重点关注低资源孟加拉语。我们研究了六种提示策略——零样本提示、拒绝抑制、恭维分类器、多样本提示、角色提示以及我们的创新元喻提示，以有效检测低资源语言中的仇恨言论。我们首次引入元喻提示，绕过了大型语言模型内置的安全机制，与现有破解方法有显著区别。我们在Llama2-7B模型上研究了所有六种不同的提示策略，并与三种预先训练的词嵌入——GloVe、Word2Vec和FastText以及三种深度学习模型——多层感知器（MLP）、卷积神经网络（CNN）和双向门控递归单元（BiGRU）进行了广泛比较。为了证明我们的元喻提示在低资源孟加拉语中的有效性，我们还在另一种低资源语言——印地语，以及两种高资源语言——英语和德语中进行了评估。所有提示技术的性能使用F1分数和环境影响因子（IF）进行评估，环境影响因子衡量二氧化碳排放、电力使用和计算时间。 

---
# Do Thinking Tokens Help or Trap? Towards More Efficient Large Reasoning Model 

**Title (ZH)**: 思考 tokens 是助力还是陷阱？走向更高效的大型推理模型 

**Authors**: Bowen Ding, Yuhan Chen, Futing Wang, Lingfeng Ming, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.23840)  

**Abstract**: Large Reasoning Models (LRMs) excel at solving complex problems but face an overthinking dilemma. When handling simple tasks, they often produce verbose responses overloaded with thinking tokens (e.g., wait, however). These tokens trigger unnecessary high-level reasoning behaviors like reflection and backtracking, reducing efficiency. In this work, our pilot study reveals that these thinking-token-induced behaviors are not essential for effective problem-solving and may even hinder correct reasoning within constrained token budgets. We identify this phenomenon as the thinking trap. To mitigate this issue, we propose Dual Policy Preference Optimization (DuP-PO), a novel algorithm featuring: (1) A rollout sampling strategy that guarantees balanced exposure to responses with and without thinking tokens; (2) A fine-grained advantage control technique to dynamically regulate the prediction of target tokens; (3) A policy shaping method ensuring stable gradient contributions from thinking tokens. Experimental results on five popular math reasoning benchmarks show that DuP-PO performs well on the popular LRM, which significantly improves their token efficiency during reasoning, while achieving superior performance of the base model. 

**Abstract (ZH)**: 大型推理模型（LRMs）在解决复杂问题方面表现出色，但也面临过度思考的困境。在处理简单任务时，它们常常产生冗长的响应，充满了思考令牌（例如，等待，然而）。这些令牌会触发不必要的高层次推理行为，如反思和回溯，降低效率。在本研究中，我们的初步研究表明，这些由思考令牌引起的行为并非有效解决问题所必需，甚至可能在受限的令牌预算内妨碍正确的推理。我们将其现象称为思考陷阱。为缓解这一问题，我们提出了一种名为双策略偏好优化（DuP-PO）的新算法，该算法包括：（1）采样策略，确保对带有和不带有思考令牌的响应有均衡的暴露；（2）细粒度的优势控制技术，以动态调节目标令牌的预测；（3）一种策略形成方法，确保思考令牌稳定的梯度贡献。实验结果表明，DuP-PO在五种流行的数学推理基准测试中表现出色，显著提高了大型推理模型的令牌效率，同时实现了基模型的优越性能。 

---
# Software Engineering for Large Language Models: Research Status, Challenges and the Road Ahead 

**Title (ZH)**: 大型语言模型的软件工程：研究现状、挑战及未来之路 

**Authors**: Hongzhou Rao, Yanjie Zhao, Xinyi Hou, Shenao Wang, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23762)  

**Abstract**: The rapid advancement of large language models (LLMs) has redefined artificial intelligence (AI), pushing the boundaries of AI research and enabling unbounded possibilities for both academia and the industry. However, LLM development faces increasingly complex challenges throughout its lifecycle, yet no existing research systematically explores these challenges and solutions from the perspective of software engineering (SE) approaches. To fill the gap, we systematically analyze research status throughout the LLM development lifecycle, divided into six phases: requirements engineering, dataset construction, model development and enhancement, testing and evaluation, deployment and operations, and maintenance and evolution. We then conclude by identifying the key challenges for each phase and presenting potential research directions to address these challenges. In general, we provide valuable insights from an SE perspective to facilitate future advances in LLM development. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的快速进步重构了人工智能（AI），推动了AI研究的边界，并为学术界和产业界开启了无限的可能性。然而，LLM开发在其生命周期中面临着日益复杂的挑战，目前尚无现有研究从软件工程（SE）方法的角度系统地探索这些挑战及其解决方案。为填补这一空白，我们从需求工程、数据集构建、模型开发与增强、测试与评估、部署与运营、维护与演化六个阶段系统分析了LLM开发的现状。然后，我们针对每个阶段识别出关键挑战，并提出了潜在的研究方向以解决这些挑战。总体而言，我们从SE视角提供了有价值的见解，以促进未来LLM开发的进展。 

---
# AutoEvoEval: An Automated Framework for Evolving Close-Ended LLM Evaluation Data 

**Title (ZH)**: AutoEvoEval：一种用于进化闭集LLM评估数据的自动化框架 

**Authors**: JiaRu Wu, Mingwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23735)  

**Abstract**: Large language models (LLMs) have shown remarkable performance on various tasks, but existing evaluation benchmarks are often static and insufficient to fully assess their robustness and generalization in realistic scenarios. Prior work using evolutionary or adversarial data augmentation has improved evaluation diversity but lacks systematic control over perturbation types and multi-step complexity, limiting comprehensive robustness analysis. To address these gaps, we propose AutoEvoEval, an evolution-based evaluation framework for close-ended tasks such as multi-choice question answering. AutoEvoEval introduces 22 interpretable atomic evolution operations and supports multi-round compositions, enabling controlled generation of diverse, challenging, and realistic test samples. We conduct extensive experiments addressing four research questions on a broad set of open- and closed-source LLMs. Our results show that atomic operations cause an average accuracy drop of 7.283\%, with structure-disrupting or misleading semantic edits causing the largest declines. Model sensitivities vary significantly for the same perturbation, and combining multiple evolution steps amplifies adversarial effects by up to 52.932\%. These findings suggest current benchmarks may overestimate true model generalization and emphasize the need for evolution-aware robustness evaluation. Code and resources are available at: this https URL. 

**Abstract (ZH)**: 基于演化的大语言模型闭集任务评估框架AutoEvoEval 

---
# PAC Bench: Do Foundation Models Understand Prerequisites for Executing Manipulation Policies? 

**Title (ZH)**: PAC Bench: 基础模型理解执行操作策略的先备条件吗？ 

**Authors**: Atharva Gundawar, Som Sagar, Ransalu Senanayake  

**Link**: [PDF](https://arxiv.org/pdf/2506.23725)  

**Abstract**: Vision-Language Models (VLMs) are increasingly pivotal for generalist robot manipulation, enabling tasks such as physical reasoning, policy generation, and failure detection. However, their proficiency in these high-level applications often assumes a deep understanding of low-level physical prerequisites, a capability that remains largely unverified. For robots to perform actions reliably, they must comprehend intrinsic object properties (e.g., material, weight), action affordances (e.g., graspable, stackable), and physical constraints (e.g., stability, reachability, or an object's state, such as being closed). Despite the widespread use of VLMs in manipulation tasks, we argue that off-the-shelf models may lack this granular, physically grounded understanding, as such prerequisites are often overlooked during training.
To address this critical gap, we introduce PAC Bench, a comprehensive benchmark designed to systematically evaluate VLMs on their understanding of core Properties, Affordances, and Constraints (PAC) from a task executability perspective. PAC Bench features a diverse dataset with over 30,000 annotations, comprising 673 real-world images (115 object classes, 15 property types, and 1 to 3 affordances defined per class), 100 real-world humanoid-view scenarios, and 120 unique simulated constraint scenarios across four tasks.
Our evaluations reveal significant gaps in the ability of current VLMs to grasp fundamental physical concepts, highlighting limitations in their suitability for reliable robot manipulation and pointing to key areas for targeted research. PAC Bench also serves as a standardized benchmark for rigorously evaluating physical reasoning in VLMs and guiding the development of more robust, physically grounded models for robotic applications.
Project Page: this https URL 

**Abstract (ZH)**: 基于视觉-语言模型的通用机器人操作能力正在逐渐增强，使其能够执行物理推理、策略生成和故障检测等高级任务。然而，这些模型在这些高级应用中的 proficiency 经常假设了对低级物理前提条件有深刻的理解，而这一点的能力仍然没有得到充分验证。为了使机器人能够可靠地执行动作，它们必须理解对象的内在属性（如材料、重量）、操作的功能（如可以抓握、可以堆叠）以及物理约束（如稳定性、可达性或对象的状态，如关闭）。尽管 VLMs 在操作任务中被广泛应用，我们认为现成的模型可能缺乏这种具体的、基于物理的理解，因为这些前提条件在训练过程中常常被忽视。

为了解决这一关键差距，我们提出了 PAC Bench，这是一种综合基准，旨在从任务可执行性的角度系统评估 VLMs 对核心属性、功能和约束（PAC）的理解。PAC Bench 包含一个多样化的数据集，包含超过 30,000 个标注，其中包括 673 个真实世界的图像（115 个对象类别、15 种属性类型以及每个类别定义的 1 到 3 种功能），100 个真实世界的类人视角场景，以及跨越四个任务的 120 种独特的模拟约束场景。

我们的评估揭示了当前 VLMs 在掌握基本物理概念方面存在的显著差距，突显了它们在可靠机器人操作中的局限性，并指出了需要重点研究的关键领域。PAC Bench 还作为标准化基准，用于严格评估 VLMs 的物理推理能力，并指导开发更稳健、基于物理的模型。 

---
# Interactive Reasoning: Visualizing and Controlling Chain-of-Thought Reasoning in Large Language Models 

**Title (ZH)**: 交互式推理：在大规模语言模型中可视化和控制链式思维推理 

**Authors**: Rock Yuren Pang, K. J. Kevin Feng, Shangbin Feng, Chu Li, Weijia Shi, Yulia Tsvetkov, Jeffrey Heer, Katharina Reinecke  

**Link**: [PDF](https://arxiv.org/pdf/2506.23678)  

**Abstract**: The output quality of large language models (LLMs) can be improved via "reasoning": generating segments of chain-of-thought (CoT) content to further condition the model prior to producing user-facing output. While these chains contain valuable information, they are verbose and lack explicit organization, making them tedious to review. Moreover, they lack opportunities for user feedback, such as to remove unwanted considerations, add desired ones, or clarify unclear assumptions. We introduce Interactive Reasoning, an interaction design that visualizes chain-of-thought outputs as a hierarchy of topics and enables user review and modification. We implement interactive reasoning in Hippo, a prototype for AI-assisted decision making in the face of uncertain trade-offs. In a user study with 16 participants, we find that interactive reasoning in Hippo allows users to quickly identify and interrupt erroneous generations, efficiently steer the model towards customized responses, and better understand both model reasoning and model outputs. Our work contributes to a new paradigm that incorporates user oversight into LLM reasoning processes. 

**Abstract (ZH)**: 大型语言模型（LLMs）的输出质量可以通过“推理”来提升：生成链式思维（CoT）内容的段落以进一步条件化模型，从而在生成用户面向输出之前提供更多信息。虽然这些链式思维包含有价值的信息，但它们冗长且缺乏明确的组织，使得审查变得繁琐。此外，它们缺乏用户反馈的机会，例如去除不必要的考虑、添加期望的考虑或澄清模糊的假设。我们引入了交互式推理，这是一种交互设计，将链式思维输出可视化为主题层次结构，使用户能够进行审查和修改。我们将在面对不确定权衡的AI辅助决策中实现交互式推理，即Hippo。用户研究中16名参与者的结果显示，Hippo中的交互式推理使用户能够快速识别并中断错误生成，高效地引导模型生成定制化响应，并更好地理解模型推理和输出。我们的工作为一个新范式做出了贡献，该范式将用户监督纳入到了LLM推理过程中。 

---
# QLPro: Automated Code Vulnerability Discovery via LLM and Static Code Analysis Integration 

**Title (ZH)**: QLPro: 基于LLM和静态代码分析集成的自动化代码漏洞发现方法 

**Authors**: Junze Hu, Xiangyu Jin, Yizhe Zeng, Yuling Liu, Yunpeng Li, Dan Du, Kaiyu Xie, Hongsong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23644)  

**Abstract**: We introduce QLPro, a vulnerability detection framework that systematically integrates LLMs and static analysis tools to enable comprehensive vulnerability detection across entire open-source this http URL constructed a new dataset, JavaTest, comprising 10 open-source projects from GitHub with 62 confirmed vulnerabilities. CodeQL, a state-of-the-art static analysis tool, detected only 24 of these vulnerabilities while QLPro detected 41. Furthermore, QLPro discovered 6 previously unknown vulnerabilities, 2 of which have been confirmed as 0-days. 

**Abstract (ZH)**: 我们介绍QLPro，一种系统地将大型语言模型和静态分析工具集成的漏洞检测框架，以实现对整个开源项目的全面漏洞检测。我们构建了新的数据集JavaTest，包含来自GitHub的10个开源项目，共有62个已确认的漏洞。CodeQL，一款最先进的静态分析工具，仅检测到其中24个漏洞，而QLPro检测到了41个。此外，QLPro还发现了6个未知的漏洞，其中2个被确认为0-day漏洞。 

---
# VAP-Diffusion: Enriching Descriptions with MLLMs for Enhanced Medical Image Generation 

**Title (ZH)**: VAP-Diffusion: 通过MLLLMs丰富描述以增强医学图像生成 

**Authors**: Peng Huang, Junhu Fu, Bowen Guo, Zeju Li, Yuanyuan Wang, Yi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.23641)  

**Abstract**: As the appearance of medical images is influenced by multiple underlying factors, generative models require rich attribute information beyond labels to produce realistic and diverse images. For instance, generating an image of skin lesion with specific patterns demands descriptions that go beyond diagnosis, such as shape, size, texture, and color. However, such detailed descriptions are not always accessible. To address this, we explore a framework, termed Visual Attribute Prompts (VAP)-Diffusion, to leverage external knowledge from pre-trained Multi-modal Large Language Models (MLLMs) to improve the quality and diversity of medical image generation. First, to derive descriptions from MLLMs without hallucination, we design a series of prompts following Chain-of-Thoughts for common medical imaging tasks, including dermatologic, colorectal, and chest X-ray images. Generated descriptions are utilized during training and stored across different categories. During testing, descriptions are randomly retrieved from the corresponding category for inference. Moreover, to make the generator robust to unseen combination of descriptions at the test time, we propose a Prototype Condition Mechanism that restricts test embeddings to be similar to those from training. Experiments on three common types of medical imaging across four datasets verify the effectiveness of VAP-Diffusion. 

**Abstract (ZH)**: 基于预训练多模态大语言模型的Visual Attribute Prompts-Diffusion框架：用于提高医学图像生成的质量和多样性 

---
# Towards Building Private LLMs: Exploring Multi-Node Expert Parallelism on Apple Silicon for Mixture-of-Experts Large Language Model 

**Title (ZH)**: 面向私有LLM的构建：在Apple Silicon多节点专家并行化探索用于混合专家大规模语言模型 

**Authors**: Mu-Chi Chen, Po-Hsuan Huang, Xiangrui Ke, Chia-Heng Tu, Chun Jason Xue, Shih-Hao Hung  

**Link**: [PDF](https://arxiv.org/pdf/2506.23635)  

**Abstract**: Large Language Models (LLMs) have revolutionized Artificial Intelligence (AI) with significant advancements such as OpenAI's ChatGPT, Meta's Llama, and Databricks' DBRX. This paper addresses the cost and scalability challenges encountered when constructing private LLM systems for personal or small group services, as aimed by Apple Intelligence. A Mac Studio cluster with Apple's M2 Ultra chips is established as a cost-efficient solution to host and accelerate the pretrained DBRX model with the Mixture-of-Experts (MoE) architecture. Our performance analysis reveal that parallel execution of the model's experts across two to four machine nodes significantly reduces inference time. We find that computation time for the experts is comparable to the communication time for exchanging their outputs, emphasizing the importance of network latency over bandwidth. We also observe significant management overhead due to Apple software stack's memory management logic. Based on these findings, we develop optimization schemes to eliminate the memory management overhead. As a result, the Mac Studio cluster is 1.15 times more cost-efficient than the state-of-the-art AI supercomputer with NVIDIA H100 GPUs. In addition, we construct a performance model to estimate system performance under varying configurations, and the model provides valuable insights for designing private LLM systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过OpenAI的ChatGPT、Meta的Llama和Databricks的DBRX等重要进展，革命性地改变了人工智能（AI）。本文解决了构建个人或小型团体专用LLM系统时遇到的成本和扩展性挑战，符合Apple Intelligence的目标。我们建立了一个配备Apple M2 Ultra芯片的Mac Studio集群，作为高效解决方案，用于托管和加速基于Mixture-of-Experts（MoE）架构的预训练DBRX模型。我们的性能分析表明，将模型的专家并行运行在两到四个机器节点上，显著降低了推理时间。我们发现，专家的计算时间与它们输出交换的时间相当，突显了网络延迟比带宽更为重要。我们还发现，由于Apple软件堆栈的内存管理逻辑，存在显著的管理开销。基于这些发现，我们开发了优化方案以消除内存管理开销。因此，Mac Studio集群比使用NVIDIA H100 GPU的最先进的AI超级计算机成本效率高出1.15倍。此外，我们构建了一个性能模型来估算在不同配置下的系统性能，该模型为设计专用LLM系统提供了宝贵的见解。 

---
# AI-Generated Lecture Slides for Improving Slide Element Detection and Retrieval 

**Title (ZH)**: AI生成的讲义幻灯片以提高幻灯片元素检测与检索 

**Authors**: Suyash Maniyar, Vishvesh Trivedi, Ajoy Mondal, Anand Mishra, C.V. Jawahar  

**Link**: [PDF](https://arxiv.org/pdf/2506.23605)  

**Abstract**: Lecture slide element detection and retrieval are key problems in slide understanding. Training effective models for these tasks often depends on extensive manual annotation. However, annotating large volumes of lecture slides for supervised training is labor intensive and requires domain expertise. To address this, we propose a large language model (LLM)-guided synthetic lecture slide generation pipeline, SynLecSlideGen, which produces high-quality, coherent and realistic slides. We also create an evaluation benchmark, namely RealSlide by manually annotating 1,050 real lecture slides. To assess the utility of our synthetic slides, we perform few-shot transfer learning on real data using models pre-trained on them. Experimental results show that few-shot transfer learning with pretraining on synthetic slides significantly improves performance compared to training only on real data. This demonstrates that synthetic data can effectively compensate for limited labeled lecture slides. The code and resources of our work are publicly available on our project website: this https URL. 

**Abstract (ZH)**: 讲义幻灯片要素检测与检索是讲义理解中的关键问题。通过大型语言模型（LLM）引导的合成讲义幻灯片生成管道SynLecSlideGen，可以生成高质量、连贯且现实的幻灯片。我们还创建了一个评估基准RealSlide，通过对1,050份实际讲义幻灯片进行人工标注实现。为了评估合成幻灯片的实用性，我们使用预先在合成幻灯片上训练的模型，在实际数据上进行少量样本迁移学习。实验结果表明，与仅使用实际数据训练相比，在合成幻灯片上进行预训练后进行少量样本迁移学习显著提高了性能。这表明合成数据可以有效弥补有限标注讲义幻灯片的问题。我们的工作代码和资源在项目网站上公开：this https URL。 

---
# SoK: Semantic Privacy in Large Language Models 

**Title (ZH)**: SoK: 大型语言模型中的语义隐私 

**Authors**: Baihe Ma, Yanna Jiang, Xu Wang, Guangshen Yu, Qin Wang, Caijun Sun, Chen Li, Xuelei Qi, Ying He, Wei Ni, Ren Ping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23603)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains, traditional data privacy measures prove inadequate for protecting information that is implicit, contextual, or inferable - what we define as semantic privacy. This Systematization of Knowledge (SoK) introduces a lifecycle-centric framework to analyze how semantic privacy risks emerge across input processing, pretraining, fine-tuning, and alignment stages of LLMs. We categorize key attack vectors and assess how current defenses, such as differential privacy, embedding encryption, edge computing, and unlearning, address these threats. Our analysis reveals critical gaps in semantic-level protection, especially against contextual inference and latent representation leakage. We conclude by outlining open challenges, including quantifying semantic leakage, protecting multimodal inputs, balancing de-identification with generation quality, and ensuring transparency in privacy enforcement. This work aims to inform future research on designing robust, semantically aware privacy-preserving techniques for LLMs. 

**Abstract (ZH)**: 作为大型语言模型（LLMs）在敏感领域中的应用日益增多，传统的数据隐私措施对于保护隐含的、上下文相关的或可推断的信息——即我们定义的语义隐私——证明不够充分。本文综述通过生命周期-centric框架分析LLMs在整个输入处理、预训练、微调和对齐过程中的语义隐私风险。我们分类关键攻击向量，并评估当前的防御措施，如差分隐私、嵌入加密、边缘计算和遗忘技术，如何应对这些威胁。我们的分析揭示了语义层面保护的关键不足，特别是在对抗上下文推断和潜在表示泄露时。最后，我们概述了开放挑战，包括量化语义泄露、保护多模态输入、平衡去标识化与生成质量以及确保隐私保护的透明性。本研究旨在为设计针对LLMs的健壮且语义意识的隐私保护技术的未来研究提供指导。 

---
# Semantic-guided Diverse Decoding for Large Language Model 

**Title (ZH)**: 基于语义引导的多样性解码 LARGE LANGUAGE MODEL 的语义导向多样化解码 

**Authors**: Weijie Shi, Yue Cui, Yaguang Wu, Jingzhi Fang, Shibo Zhang, Mengze Li, Sirui Han, Jia Zhu, Jiajie Xu, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.23601)  

**Abstract**: Diverse decoding of large language models is crucial for applications requiring multiple semantically distinct responses, yet existing methods primarily achieve lexical rather than semantic diversity. This limitation significantly constrains Best-of-N strategies, group-based reinforcement learning, and data synthesis. While temperature sampling and diverse beam search modify token distributions or apply n-gram penalties, they fail to ensure meaningful semantic differentiation. We introduce Semantic-guided Diverse Decoding (SemDiD), operating directly in embedding space that balances quality with diversity through three complementary mechanisms: orthogonal directional guidance, dynamic inter-group repulsion, and position-debiased probability assessment. SemDiD harmonizes these competing objectives using adaptive gain functions and constraint optimization, ensuring both quality thresholds and maximal semantic differentiation. Experiments show SemDiD consistently outperforms existing methods, improving Best-of-N coverage by 1.4-5.2% across diverse tasks and accelerating RLHF training convergence by 15% while increasing accuracy by up to 2.1%. 

**Abstract (ZH)**: 大型语言模型的语义引导多样解码对于需要多个语义上不同的响应的应用至关重要，现有方法主要实现的是词汇多样性而非语义多样性。这一限制显著制约了Best-of-N策略、基于群体的强化学习以及数据合成。虽然温度采样和多样性的beam搜索修改了令牌分布或应用了n-gram惩罚，但它们无法确保有意义的语义区分。我们引入了语义引导多样解码（SemDiD），它直接在嵌入空间中运作，通过三种互补机制平衡质量和多样性：正交方向引导、动态组间排斥以及位置无偏的概率评估。SemDiD 使用自适应增益函数和约束优化来协调这些竞争目标，确保达到质量阈值同时实现最大限度的语义区分。实验显示，SemDiD 在多种任务中始终优于现有方法，提高了1.4-5.2%的Best-of-N覆盖率，加快了RLHF训练收敛速度15%并提高了高达2.1%的准确性。 

---
# Reinforcement Fine-Tuning Enables MLLMs Learning Novel Tasks Stably 

**Title (ZH)**: reinforcement微调使大语言模型稳定学习新型任务 

**Authors**: Zhihao Zhang, Qiaole Dong, Qi Zhang, Jun Zhao, Enyu Zhou, Zhiheng Xi, Senjie Jin, Xiaoran Fan, Yuhao Zhou, Yanwei Fu, Tao Ji, Tao Gui, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23508)  

**Abstract**: Post-training algorithms such as Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) are widely used to adapt multimodal large language models to downstream tasks. While effective at task adaptation, their impact on prior knowledge remains unclear. In this paper, we introduce jigsaw puzzles as a novel task absent from existing pretraining corpora and systematically study the behavior of SFT and RFT on an open-source multimodal model, Qwen2.5-VL. Our experiments reveal a sharp trade-off: SFT enables rapid task acquisition but leads to catastrophic forgetting, whereas RFT learns more slowly on novel tasks but maintains prior knowledge. We analyze this phenomenon through the lens of learning dynamics, showing that RFT reinforces correct samples that are naturally aligned with the base model's probability landscape, mitigating interference with prior knowledge. Moreover, supervised training on correct RFT-simulated rollouts allows SFT to preserve knowledge while rapidly learning new tasks. These findings suggest that data distribution, rather than algorithmic differences, plays a central role in forgetting, and highlight RFT's potential for stable continual learning in multimodal large language models. 

**Abstract (ZH)**: Post-training算法如监督微调(SFT)和强化微调(RFT)广泛用于将多模态大型语言模型适应下游任务，但其对先验知识的影响尚不明确。在本文中，我们引入拼图游戏作为现有预训练数据集中不存在的新任务，并系统研究SFT和RFT在开源多模态模型Qwen2.5-VL上的行为。实验结果显示，SFT能够迅速获取新任务但会导致灾难性遗忘，而RFT在新任务上学习速度较慢但能够保留先验知识。我们从学习动态的角度分析了这一现象，表明RFT强化了与基模型概率景观自然对齐的正确样本，从而减少了对先验知识的干扰。此外，对正确RFT模拟回放进行监督训练可以使SFT在快速学习新任务的同时保留知识。这些发现表明，数据分布而非算法差异在遗忘中起着核心作用，并突显了在多模态大型语言模型中RFT在稳定持续学习方面的潜力。 

---
# Thought-Augmented Planning for LLM-Powered Interactive Recommender Agent 

**Title (ZH)**: LLM驱动的交互式推荐代理的思维增强规划 

**Authors**: Haocheng Yu, Yaxiong Wu, Hao Wang, Wei Guo, Yong Liu, Yawen Li, Yuyang Ye, Junping Du, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.23485)  

**Abstract**: Interactive recommendation is a typical information-seeking task that allows users to interactively express their needs through natural language and obtain personalized recommendations. Large language model-powered (LLM-powered) agents have become a new paradigm in interactive recommendations, effectively capturing users' real-time needs and enhancing personalized experiences. However, due to limited planning and generalization capabilities, existing formulations of LLM-powered interactive recommender agents struggle to effectively address diverse and complex user intents, such as intuitive, unrefined, or occasionally ambiguous requests. To tackle this challenge, we propose a novel thought-augmented interactive recommender agent system (TAIRA) that addresses complex user intents through distilled thought patterns. Specifically, TAIRA is designed as an LLM-powered multi-agent system featuring a manager agent that orchestrates recommendation tasks by decomposing user needs and planning subtasks, with its planning capacity strengthened through Thought Pattern Distillation (TPD), a thought-augmentation method that extracts high-level thoughts from the agent's and human experts' experiences. Moreover, we designed a set of user simulation schemes to generate personalized queries of different difficulties and evaluate the recommendations based on specific datasets. Through comprehensive experiments conducted across multiple datasets, TAIRA exhibits significantly enhanced performance compared to existing methods. Notably, TAIRA shows a greater advantage on more challenging tasks while generalizing effectively on novel tasks, further validating its superiority in managing complex user intents within interactive recommendation systems. The code is publicly available at:this https URL. 

**Abstract (ZH)**: 基于思维增强的交互推荐代理系统（TAIRA）：通过萃取思维模式应对复杂用户意图 

---
# Can We Predict the Unpredictable? Leveraging DisasterNet-LLM for Multimodal Disaster Classification 

**Title (ZH)**: 能否预测不可预测的灾害？基于DisasterNet-LLM的多模态灾害分类方法 

**Authors**: Manaswi Kulahara, Gautam Siddharth Kashyap, Nipun Joshi, Arpita Soni  

**Link**: [PDF](https://arxiv.org/pdf/2506.23462)  

**Abstract**: Effective disaster management requires timely and accurate insights, yet traditional methods struggle to integrate multimodal data such as images, weather records, and textual reports. To address this, we propose DisasterNet-LLM, a specialized Large Language Model (LLM) designed for comprehensive disaster analysis. By leveraging advanced pretraining, cross-modal attention mechanisms, and adaptive transformers, DisasterNet-LLM excels in disaster classification. Experimental results demonstrate its superiority over state-of-the-art models, achieving higher accuracy of 89.5%, an F1 score of 88.0%, AUC of 0.92%, and BERTScore of 0.88% in multimodal disaster classification tasks. 

**Abstract (ZH)**: 有效的灾害管理需要及时准确的洞察，然而传统的方法在集成如图像、天气记录和文本报告等多种模态数据方面存在困难。为了解决这一问题，我们提出了一种专门用于综合灾害分析的大型语言模型DisasterNet-LLM。通过利用高级预训练、跨模态注意力机制和自适应变压器，DisasterNet-LLM 在灾害分类方面表现出色。实验结果表明，其在多模态灾害分类任务中的性能优于现有最先进的模型，准确率为89.5%，F1分数为88.0%，AUC为0.92%，BERTScore为0.88%。 

---
# TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs 

**Title (ZH)**: TuCo: 评估微调对个体语言模型响应贡献的度量方法 

**Authors**: Felipe Nuti, Tim Franzmeyer, João Henriques  

**Link**: [PDF](https://arxiv.org/pdf/2506.23423)  

**Abstract**: Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. However, a quantitative and systematic method for analyzing its effect on individual outputs is still lacking. Here, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. Our method tracks the model's intermediate hidden states, providing a more fine-grained insight into the effects of fine-tuning than a simple comparison of final outputs from pre-trained and fine-tuned models. We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component. Empirically, we find that model behavior and performance can be steered by up- or down-scaling the fine-tuning component during the forward pass. Motivated by this finding and our theoretical analysis, we define the Tuning Contribution (TuCo) as the ratio of the magnitudes of the fine-tuning component to the pre-training component. We observe that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces TuCo, and that TuCo is consistently lower on prompts where these attacks succeed compared to those where they do not. This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of such attacks. In summary, TuCo enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa. 

**Abstract (ZH)**: 量化分析微调对大型语言模型个体输出影响的新方法：Tuning Contribution (TuCo) 研究 

---
# Teaching a Language Model to Speak the Language of Tools 

**Title (ZH)**: Teaching a Language Model to Speak the Language of Tools 

**Authors**: Simeon Emanuilov  

**Link**: [PDF](https://arxiv.org/pdf/2506.23394)  

**Abstract**: External tool integration through function-calling is essential for practical language model applications, yet most multilingual models lack reliable tool-use capabilities in non-English languages. Even state-of-the-art multilingual models struggle with determining when to use tools and generating the structured outputs required for function calls, often exhibiting language confusion when prompted in lower-resource languages. This work presents a methodology for adapting existing language models to enable robust tool use in any target language, using Bulgarian as a case study. The approach involves continued training of the BgGPT model series (2.6B, 9B, 27B parameters) on a novel bilingual dataset of 10,035 function-calling examples designed to support standardized protocols like MCP (Model Context Protocol). The research introduces TUCAN (Tool-Using Capable Assistant Navigator), which achieves up to 28.75% improvement in function-calling accuracy over base models while preserving core language understanding, as verified on established Bulgarian benchmarks. Beyond accuracy gains, TUCAN models demonstrate production-ready response formatting with clean, parsable function calls, contrasting with the verbose and inconsistent outputs of base models. The models, evaluation framework, and dataset are released to enable replication for other languages. This work demonstrates a practical approach for extending tool-augmented capabilities beyond English-centric systems. 

**Abstract (ZH)**: 通过函数调用集成外部工具对于实践中的语言模型应用至关重要，但大多数多语言模型在非英语语言中缺乏可靠的工具使用能力。即使最先进的多语言模型在确定何时使用工具以及生成用于函数调用的结构化输出方面也存在问题，常常在低资源语言的提示下表现出语言混淆。本研究提出了一种方法，用于将现有语言模型适应为能够在任何目标语言中实现稳健的工具使用，并以保加利亚语为例进行了研究。该方法涉及在为支持标准化协议（如MCP模型上下文协议）设计的10,035个双语函数调用示例的新数据集上对保加利亚语GPT模型系列（参数分别为2.6B、9B、27B）进行持续训练。研究引入了TUCAN（工具使用能力助手导航者），其在函数调用准确性上比基模型提高了至多28.75%，同时保持了核心语言理解能力，已在建立的保加利亚语基准上得到了验证。除了准确性提高，TUCAN模型还展示了生产级别的响应格式化，函数调用简洁且可解析，与基模型的冗长且不一致的输出形成对比。该研究发布了模型、评估框架和数据集，以便其他语言的复制。本研究展示了一种实用的方法，用于将工具增强能力扩展到以英语为中心的系统之外。 

---
# Perspective Dial: Measuring Perspective of Text and Guiding LLM Outputs 

**Title (ZH)**: 视角 Dial：测量文本视角并引导大模型输出 

**Authors**: Taejin Kim, Siun-Chuon Mau, Konrad Vesey  

**Link**: [PDF](https://arxiv.org/pdf/2506.23377)  

**Abstract**: Large language models (LLMs) are used in a variety of mission-critical roles. Due to the rapidly developing nature of LLMs, there is a lack of quantifiable understanding of the bias and perspective associated with LLM output. Inspired by this need, this paper considers the broader issue of perspective or viewpoint of general text and perspective control of large-language model (LLM) output. Perspective-Dial consists of two main components: a (1) metric space, dubbed Perspective Space, that enables quantitative measurements of different perspectives regarding a topic, and the use of (2) Systematic Prompt Engineering that utilizes greedy-coordinate descent to control LLM output perspective based on measurement feedback from the Perspective Space. The empirical nature of the approach allows progress to side step a principled understanding of perspective or bias -- effectively quantifying and adjusting outputs for a variety of topics. Potential applications include detection, tracking and mitigation of LLM bias, narrative detection, sense making and tracking in public discourse, and debate bot advocating given perspective. 

**Abstract (ZH)**: 大型语言模型（LLM）在多种关键任务中被使用。由于LLM的迅速发展，对其输出的偏见和视角缺乏可量化的理解。受此需求启发，本文考虑了广泛意义上的文本视角或观点问题以及大型语言模型（LLM）输出视角控制的更广泛问题。Perspective-Dial包括两个主要组件：一个（1）称为视角空间的度量空间，使不同主题视角的定量测量成为可能，以及（2）系统提示工程，利用贪婪坐标下降来根据视角空间的测量反馈控制LLM输出视角。该经验方法允许我们绕过对视角或偏见的原理性理解——有效地对多种主题的输出进行量化和调整。潜在应用包括LLM偏见的检测、跟踪和缓解、叙事检测、公共话语中的意义构建与跟踪，以及倡导特定视角的辩论机器人。 

---
# ATGen: A Framework for Active Text Generation 

**Title (ZH)**: ATGen: 一种主动文本生成框架 

**Authors**: Akim Tsvigun, Daniil Vasilev, Ivan Tsvigun, Ivan Lysenko, Talgat Bektleuov, Aleksandr Medvedev, Uliana Vinogradova, Nikita Severin, Mikhail Mozikov, Andrey Savchenko, Rostislav Grigorev, Ramil Kuleev, Fedor Zhdanov, Artem Shelmanov, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2506.23342)  

**Abstract**: Active learning (AL) has demonstrated remarkable potential in reducing the annotation effort required for training machine learning models. However, despite the surging popularity of natural language generation (NLG) tasks in recent years, the application of AL to NLG has been limited. In this paper, we introduce Active Text Generation (ATGen) - a comprehensive framework that bridges AL with text generation tasks, enabling the application of state-of-the-art AL strategies to NLG. Our framework simplifies AL-empowered annotation in NLG tasks using both human annotators and automatic annotation agents based on large language models (LLMs). The framework supports LLMs deployed as services, such as ChatGPT and Claude, or operated on-premises. Furthermore, ATGen provides a unified platform for smooth implementation and benchmarking of novel AL strategies tailored to NLG tasks. Finally, we present evaluation results for state-of-the-art AL strategies across diverse settings and multiple text generation tasks. We show that ATGen reduces both the effort of human annotators and costs associated with API calls to LLM-based annotation agents. The code of the framework is available on GitHub under the MIT license. The video presentation is available at this http URL 

**Abstract (ZH)**: 主动学习（AL）在减少训练机器学习模型所需的标注 effort 方面展现了显著潜力。然而，尽管近年来自然语言生成（NLG）任务备受欢迎，AL 在 NLG 中的应用仍有限。在本文中，我们介绍了主动文本生成（ATGen）——一个综合框架，将 AL 与文本生成任务相结合，使最先进的 AL 策略能够应用于 NLG。该框架简化了基于人类标注者和基于大规模语言模型（LLMs）的自动标注代理的 AL 促进的标注过程。该框架支持作为服务部署的 LLM，如 ChatGPT 和 Claude，或本地部署的 LLM。此外，ATGen 提供了一个统一的平台，用于实现和基准测试针对 NLG 任务的新颖 AL 策略。最后，我们在多种文本生成任务和不同背景下评估了最先进的 AL 策略。结果显示，ATGen 减少了人类标注者的努力以及基于 LLM 的标注代理的 API 调用成本。该框架的代码在 GitHub 上以 MIT 许可证发布。视频演示可在以下网址获得。 

---
# VALID-Mol: a Systematic Framework for Validated LLM-Assisted Molecular Design 

**Title (ZH)**: VALID-Mol：一种验证性的LLM辅助分子设计框架 

**Authors**: Malikussaid, Hilal Hudan Nuha  

**Link**: [PDF](https://arxiv.org/pdf/2506.23339)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable potential for scientific discovery, but their application in domains requiring factual accuracy and domain-specific constraints remains challenging. In molecular design for drug discovery, LLMs can suggest creative molecular modifications but often produce chemically invalid or impractical structures. We present VALID-Mol, a systematic framework for integrating chemical validation with LLM-driven molecular design that increases the rate of generating valid chemical structures from 3% to 83%. Our approach combines methodical prompt engineering, automated chemical validation, and a fine-tuned domain-adapted LLM to ensure reliable generation of synthesizable molecules with improved properties. Beyond the specific implementation, we contribute a generalizable methodology for scientifically-constrained LLM applications, with quantifiable reliability improvements. Computational predictions suggest our framework can generate promising candidates for synthesis with up to 17-fold computationally predicted improvements in target affinity while maintaining synthetic accessibility. We provide a detailed analysis of our prompt engineering process, validation architecture, and fine-tuning approach, offering a reproducible blueprint for applying LLMs to other scientific domains where domain-specific validation is essential. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科学发现中展现出巨大潜力，但在需要事实准确性及领域特定约束的应用领域仍具挑战。在药物发现中的分子设计中，LLMs可以提出创造性的分子修改，但往往会产生化学上无效或不切实际的结构。我们提出了一种名为VALID-Mol的系统框架，将化学验证与基于LLM的分子设计相结合，将生成有效化学结构的比率从3%提高到83%。我们的方法结合了系统的提示工程、自动化学验证和微调的领域适应性LLM，以确保可靠生成可合成分子并改进其性能。除了具体实施外，我们还提供了一种适用于科学约束条件下的LLM应用的一般化方法，并实现可量化可靠性的提升。计算预测表明，我们的框架可以在保持合成可及性的同时，为目标亲和力提供最多17倍的计算预测改进，以生成具有良好合成潜力的目标候选物。我们详细分析了提示工程过程、验证架构和微调方法，提供了一种可重复的蓝本，用于将LLM应用于其他关键需要领域特定验证的科学领域。 

---
# GaussMaster: An LLM-based Database Copilot System 

**Title (ZH)**: GaussMaster: 一个基于LLM的数据库副驾系统 

**Authors**: Wei Zhou, Ji Sun, Xuanhe Zhou, Guoliang Li, Luyang Liu, Hao Wu, Tianyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23322)  

**Abstract**: In the financial industry, data is the lifeblood of operations, and DBAs shoulder significant responsibilities for SQL tuning, database deployment, diagnosis, and service repair. In recent years, both database vendors and customers have increasingly turned to autonomous database platforms in an effort to alleviate the heavy workload of DBAs. However, existing autonomous database platforms are limited in their capabilities, primarily addressing single-point issues such as NL2SQL, anomaly detection, and SQL tuning. Manual intervention remains a necessity for comprehensive database maintenance. GaussMaster aims to revolutionize this landscape by introducing an LLM-based database copilot system. This innovative solution is designed not only to assist developers in writing efficient SQL queries but also to provide comprehensive care for database services. When database instances exhibit abnormal behavior, GaussMaster is capable of orchestrating the entire maintenance process automatically. It achieves this by analyzing hundreds of metrics and logs, employing a Tree-of-thought approach to identify root causes, and invoking appropriate tools to resolve issues. We have successfully implemented GaussMaster in real-world scenarios, such as the banking industry, where it has achieved zero human intervention for over 34 database maintenance scenarios. In this paper, we present significant improvements in these tasks with code at this https URL. 

**Abstract (ZH)**: 在金融行业中，数据是运营的血液，DBA承担着SQL调优、数据库部署、诊断和服务修复等重要职责。近年来，数据库供应商和用户越来越倾向于使用自主数据库平台以减轻DBA的工作负担。然而，现有的自主数据库平台能力有限，主要解决诸如NL2SQL、异常检测和SQL调优等单一问题。全面的数据库维护仍需人工介入。GaussMaster旨在通过引入基于LLM的数据库伴侣系统来改变这一局面。这一创新解决方案不仅旨在帮助开发者编写高效的SQL查询，还能全面照顾数据库服务。当数据库实例出现异常行为时，GaussMaster能够自动协调整个维护过程。它通过分析数百个指标和日志，采用思维树方法识别根本原因，并调用适当的工具解决问题。我们在银行等行业的真实场景中成功实施了GaussMaster，并在超过34个数据库维护场景中实现了零人工干预。在本文中，我们在这个链接中的代码中介绍了这些任务的重要改进：https://this网址URL。 

---
# Predicting thinking time in Reasoning models 

**Title (ZH)**: 在推理模型中预测思考时间 

**Authors**: Hans Peter Lynsgøe Raaschou-jensen, Constanza Fierro, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2506.23274)  

**Abstract**: Reasoning models that produce long, hidden chains of thought have emerged as powerful tools for complex, reasoning-intensive tasks\citep{deepseekai2025deepseekr1incentivizingreasoningcapability, openai2024openaio1card}. However, this paradigm introduces a new user experience challenge: users have little insight into how much time the model will spend reasoning before returning an answer. This unpredictability, can lead to user frustration and is likely to compound as LLMs can produce increasingly long tasks asynchronously \citep{kwa2025measuringaiabilitycomplete}. In this paper, we introduce and evaluate methods for both online and offline prediction of model "thinking time," aiming to develop a practical "progress bar for reasoning." We discuss the implications for user interaction and future research directions. 

**Abstract (ZH)**: 基于生成长期隐藏思维链的推理模型已成为复杂推理密集型任务的强大工具。然而，这一范式引入了新的用户体验挑战：用户难以预知模型在返回答案前将花费多少时间进行推理。这种不确定性可能导致用户沮丧，并可能随着LLMs生成越来越长时间的异步任务而加剧。本文介绍并评估了在线和离线预测模型“思考时间”的方法，旨在开发实用的“推理进度条”。我们讨论了用户交互的影响和未来研究方向。 

---
# Token Activation Map to Visually Explain Multimodal LLMs 

**Title (ZH)**: Token激活图：可视化解释多模态LLMs 

**Authors**: Yi Li, Hualiang Wang, Xinpeng Ding, Haonan Wang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23270)  

**Abstract**: Multimodal large language models (MLLMs) are broadly empowering various fields. Despite their advancements, the explainability of MLLMs remains less explored, hindering deeper understanding, model credibility, and effective visualization. Unlike conventional vision models (e.g., CNNs, ViTs, CLIP) that produce a single output, MLLMs generate sequences of tokens progressively, where each generated token depends on the previous context. Therefore, earlier context tokens can introduce redundant activations that interfere with the explanation of later tokens beyond their original information. Existing studies often overlook this issue, but our observations reveal that these redundant correlations can significantly hurt the reliability of explanations. To address this, we propose an estimated causal inference method to mitigate the interference of context to achieve high-quality MLLM explanation, with a novel rank Gaussian filter to further reduce activation noises. We term this method Token Activation Map (TAM) to highlight the consideration of interactions between tokens. TAM also indicates that it excels at explaining multiple tokens of MLLM, which is different from the Class Activation Map (CAM) for a single prediction. Our TAM method significantly outperforms existing SoTA methods, showcasing high-quality visualization results that can be utilized for various scenarios, such as object localization, failure case analysis, video visualization, MLLMs visual comparison, and model understanding (e.g., color, shape, action, location, visual reasoning, multi-turn conversation, etc). The code is available this http URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）正在广泛赋能各个领域。尽管MLLMs取得了进展，但其可解释性研究仍相对缺乏，这限制了其深层次理解、模型可信度和有效可视化的发展。与传统的视觉模型（如CNNs、ViTs、CLIP）生成单一输出不同，MLLMs逐步生成一序列的标记，其中每个生成的标记都依赖于先前的上下文。因此，先前的上下文标记可能会引入冗余激活，干扰对后续标记解释的解释，而这些冗余激活超出了它们原始信息的范围。现有研究往往忽视了这个问题，但我们的观察表明，这些冗余关系会显著损害解释的可靠性。为了解决这一问题，我们提出了一种估计因果推理方法，以减轻上下文对解释的干扰，实现高质量的MLLM解释，并引入了一种新的秩高斯滤波器进一步降低激活噪声。我们称此方法为标记激活图（TAM），突出了标记之间相互作用的考虑。TAM还表明，它在解释MLLM的多个标记方面表现出色，这与针对单一预测的类激活图（CAM）不同。我们的TAM方法显著优于现有最佳方法，展示了高质量的可视化结果，可用于诸如对象定位、故障案例分析、视频可视化、MLLM视觉对比、模型理解（如颜色、形状、动作、位置、视觉推理、多轮对话等）等各种场景。代码可从此链接获取。 

---
# From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows 

**Title (ZH)**: 从提示注入到协议利用：基于LLM的AI代理工作流程中的威胁 

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.23260)  

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces have dramatically expanded capabilities for real-time data retrieval, complex computation, and multi-step orchestration. Yet, the explosive proliferation of plugins, connectors, and inter-agent protocols has outpaced discovery mechanisms and security practices, resulting in brittle integrations vulnerable to diverse threats. In this survey, we introduce the first unified, end-to-end threat model for LLM-agent ecosystems, spanning host-to-tool and agent-to-agent communications, formalize adversary capabilities and attacker objectives, and catalog over thirty attack techniques. Specifically, we organized the threat model into four domains: Input Manipulation (e.g., prompt injections, long-context hijacks, multimodal adversarial inputs), Model Compromise (e.g., prompt- and parameter-level backdoors, composite and encrypted multi-backdoors, poisoning strategies), System and Privacy Attacks (e.g., speculative side-channels, membership inference, retrieval poisoning, social-engineering simulations), and Protocol Vulnerabilities (e.g., exploits in Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent Network Protocol (ANP), and Agent-to-Agent (A2A) protocol). For each category, we review representative scenarios, assess real-world feasibility, and evaluate existing defenses. Building on our threat taxonomy, we identify key open challenges and future research directions, such as securing MCP deployments through dynamic trust management and cryptographic provenance tracking; designing and hardening Agentic Web Interfaces; and achieving resilience in multi-agent and federated environments. Our work provides a comprehensive reference to guide the design of robust defense mechanisms and establish best practices for resilient LLM-agent workflows. 

**Abstract (ZH)**: 由大型语言模型（LLMs）驱动、具备结构化功能调用接口的自主AI代理大幅扩展了实时数据检索、复杂计算和多步编排的能力。然而，插件、连接器和代理间协议的爆炸性增长超越了发现机制和安全实践，导致了易受多种威胁的脆弱集成。在本文综述中，我们首次提出了一种统一的端到端威胁模型，涵盖了主机到工具及代理间通信，正式化了攻击者能力和攻击者目标，并列出了超过三十种攻击技术。具体来说，我们将威胁模型分为四个领域：输入操纵（例如，提示注入、长上下文劫持、多模态对抗输入）、模型妥协（例如，提示和参数级后门、复合加密多后门、投毒策略）、系统和隐私攻击（例如，推测旁路信道、成员推理、检索投毒、社会工程模拟）以及协议漏洞（例如，模型上下文协议（MCP）、代理通信协议（ACP）、代理网络协议（ANP）和代理到代理（A2A）协议的利用）。对于每类威胁，我们回顾了代表性场景，评估了实际可行性，并评估了现有防御措施。基于我们的威胁分类，我们确定了关键的开放挑战和未来研究方向，例如，通过动态信任管理和加密溯源跟踪来加强MCP部署的安全性；设计和加固代理网络界面；以及在多代理和联邦环境中实现恢复力。我们的工作为设计稳健的防御机制和建立具有恢复力的LLM-代理工作流的最佳实践提供了全面参考。 

---
# UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence with Spatial Reasoning and Understanding 

**Title (ZH)**: 城市LLaVA：具备空间推理与理解的多模态大型语言模型 

**Authors**: Jie Feng, Shengyuan Wang, Tianhui Liu, Yanxin Xi, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23219)  

**Abstract**: Urban research involves a wide range of scenarios and tasks that require the understanding of multi-modal data. Current methods often focus on specific data types and lack a unified framework in urban field for processing them comprehensively. The recent success of multi-modal large language models (MLLMs) presents a promising opportunity to overcome this limitation. In this paper, we introduce $\textit{UrbanLLaVA}$, a multi-modal large language model designed to process these four types of data simultaneously and achieve strong performance across diverse urban tasks compared with general MLLMs. In $\textit{UrbanLLaVA}$, we first curate a diverse urban instruction dataset encompassing both single-modal and cross-modal urban data, spanning from location view to global view of urban environment. Additionally, we propose a multi-stage training framework that decouples spatial reasoning enhancement from domain knowledge learning, thereby improving the compatibility and downstream performance of $\textit{UrbanLLaVA}$ across diverse urban tasks. Finally, we also extend existing benchmark for urban research to assess the performance of MLLMs across a wide range of urban tasks. Experimental results from three cities demonstrate that $\textit{UrbanLLaVA}$ outperforms open-source and proprietary MLLMs in both single-modal tasks and complex cross-modal tasks and shows robust generalization abilities across cities. Source codes and data are openly accessible to the research community via this https URL. 

**Abstract (ZH)**: 城市研究涉及广泛的情景和任务，需要理解多模态数据。当前的方法往往专注于特定的数据类型，并且缺乏一个能够综合处理这些数据的统一框架。近期多模态大型语言模型（MLLMs）的成功为克服这一局限性提供了 promising 的机会。在本文中，我们介绍了 $\textit{UrbanLLaVA}$，一个设计用于同时处理这四种类型数据并实现跨多种城市任务强大性能的多模态大型语言模型。在 $\textit{UrbanLLaVA}$ 中，我们首先制定了一个多元的城市指令数据集，涵盖了从位置视图到城市环境全球视图的单模态和跨模态城市数据。此外，我们还提出了一种多阶段训练框架，将空间推理增强与领域知识学习解耦，从而提高了 $\textit{UrbanLLaVA}$ 在多种城市任务中的兼容性和下游性能。最后，我们还扩展了现有的城市研究基准，以评估 MLLMs 在多种城市任务中的性能。来自三个城市的实验结果表明，$\textit{UrbanLLaVA}$ 在单模态任务和复杂跨模态任务中均优于开源和私有 MLLMs，并且展示了跨城市的一致泛化能力。源代码和数据可通过此链接公开访问。 

---
# From Individuals to Interactions: Benchmarking Gender Bias in Multimodal Large Language Models from the Lens of Social Relationship 

**Title (ZH)**: 从个体到互动：从社会关系视角benchmark多模态大型语言模型中的性别偏见 

**Authors**: Yue Xu, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23101)  

**Abstract**: Multimodal large language models (MLLMs) have shown impressive capabilities across tasks involving both visual and textual modalities. However, growing concerns remain about their potential to encode and amplify gender bias, particularly in socially sensitive applications. Existing benchmarks predominantly evaluate bias in isolated scenarios, overlooking how bias may emerge subtly through interpersonal interactions. We fill this gap by going beyond single-entity evaluation and instead focusing on a deeper examination of relational and contextual gender bias in dual-individual interactions. We introduce Genres, a novel benchmark designed to evaluate gender bias in MLLMs through the lens of social relationships in generated narratives. Genres assesses gender bias through a dual-character profile and narrative generation task that captures rich interpersonal dynamics and supports a fine-grained bias evaluation suite across multiple dimensions. Experiments on both open- and closed-source MLLMs reveal persistent, context-sensitive gender biases that are not evident in single-character settings. Our findings underscore the importance of relationship-aware benchmarks for diagnosing subtle, interaction-driven gender bias in MLLMs and provide actionable insights for future bias mitigation. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在涉及视觉和文本模态的任务中展现了 impressive 的能力。然而，人们越来越担忧它们在社会敏感应用中有可能编码和放大性别偏见。现有的基准主要在孤立场景中评估偏见，忽视了偏见可能通过人际互动微妙地浮现。我们通过超越单一实体的评估，转而重点考察双个体互动中的关系性和上下文性性别偏见。我们引入了 Genres，一个旨在通过生成叙事中的社会关系视角评估 MLLMs 性别偏见的新型基准。Genres 通过双人物档案和叙事生成任务进行评估，捕捉丰富的个人间动态，支持多维度的细致偏见评估。在开源和闭源 MLLMs 上的实验揭示了持续存在的、上下文相关的性别偏见，这些偏见在单一人物设置中不易察觉。我们的研究结果强调了关系意识基准的重要性，以诊断 MLLMs 中微妙的、互动驱动的性别偏见，并为未来偏见缓解提供了可行见解。 

---
# Measuring How LLMs Internalize Human Psychological Concepts: A preliminary analysis 

**Title (ZH)**: 测量LLM内化人类心理概念的程度：一项初步分析 

**Authors**: Hiro Taiyo Hamada, Ippei Fujisawa, Genji Kawakita, Yuki Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2506.23055)  

**Abstract**: Large Language Models (LLMs) such as ChatGPT have shown remarkable abilities in producing human-like text. However, it is unclear how accurately these models internalize concepts that shape human thought and behavior. Here, we developed a quantitative framework to assess concept alignment between LLMs and human psychological dimensions using 43 standardized psychological questionnaires, selected for their established validity in measuring distinct psychological constructs. Our method evaluates how accurately language models reconstruct and classify questionnaire items through pairwise similarity analysis. We compared resulting cluster structures with the original categorical labels using hierarchical clustering. A GPT-4 model achieved superior classification accuracy (66.2\%), significantly outperforming GPT-3.5 (55.9\%) and BERT (48.1\%), all exceeding random baseline performance (31.9\%). We also demonstrated that the estimated semantic similarity from GPT-4 is associated with Pearson's correlation coefficients of human responses in multiple psychological questionnaires. This framework provides a novel approach to evaluate the alignment of the human-LLM concept and identify potential representational biases. Our findings demonstrate that modern LLMs can approximate human psychological constructs with measurable accuracy, offering insights for developing more interpretable AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）如ChatGPT在生成类人类文本方面展现了显著的能力。然而，这些模型在内化塑造人类思维和行为的概念方面存在多少准确性仍不明确。在这里，我们开发了一种定量框架，使用43个标准化的心理问卷来评估LLMs与人类心理维度的概念对齐情况，这些问卷因其在测量不同心理构念方面的确立效度而被选择。我们的方法通过成对相似性分析来评估语言模型如何准确地重构和分类问卷项目。我们使用层次聚类将所得的聚类结构与原始类别标签进行比较。GPT-4模型实现了更高的分类准确性（66.2%），显著优于GPT-3.5（55.9%）和BERT（48.1%），所有这些都超过了随机基线性能（31.9%）。我们还展示了GPT-4估计的语义相似性与多个心理问卷中人类反应的皮尔逊相关系数之间的关联。此框架提供了一种新的方法来评估人类-LLM概念对齐情况，并识别潜在的表示偏差。我们的发现表明，现代LLM能够以可测量的准确性近似人类心理构念，为开发更具可解释性的AI系统提供了见解。 

---
# Treatment, evidence, imitation, and chat 

**Title (ZH)**: 治疗、证据、模仿与聊天 

**Authors**: Samuel J. Weisenthal  

**Link**: [PDF](https://arxiv.org/pdf/2506.23040)  

**Abstract**: Large language models are thought to have potential to aid in medical decision making. We investigate this here. We start with the treatment problem, the patient's core medical decision-making task, which is solved in collaboration with a healthcare provider. We discuss approaches to solving the treatment problem, including -- within evidence-based medicine -- trials and observational data. We then discuss the chat problem, and how this differs from the treatment problem -- in particular as it relates to imitation. We then discuss how a large language model might be used to solve the treatment problem and highlight some of the challenges that emerge. We finally discuss how these challenges relate to evidence-based medicine, and how this might inform next steps. 

**Abstract (ZH)**: 大型语言模型在医疗决策辅助方面具有潜在应用价值。我们在此进行了探究。我们从治疗问题入手，这是患者的核心医疗决策任务，通常与医疗提供者合作解决。我们讨论了治疗问题的解决方法，包括基于证据的医学中的临床试验和观察数据。然后，我们讨论了对话问题及其与治疗问题的不同，尤其是在模仿方面的差异。接着，我们探讨了大型语言模型如何解决治疗问题，并指出了其中的一些挑战。最后，我们讨论了这些挑战与基于证据的医学之间的关系，并探讨了这如何指导下一步的工作。 

---
# Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models 

**Title (ZH)**: Spectra 1.1: 规律扩展与三值语言模型的高效推理 

**Authors**: Tejas Vaidhya, Ayush Kaushal, Vineet Jain, Francis Couture Harpin, Prashant Shishodia, Majid Behbahani, Yuriy Nevmyvaka, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2506.23025)  

**Abstract**: Large language models (LLMs) are increasingly used across research and industry applications, yet their inference efficiency remains a significant challenge. As the computational power of modern GPU architectures continuously improves, their memory bandwidth and capacity have not scaled proportionally, creating a critical bottleneck during inference. To address this, we investigate ternary language models (TriLMs) that employ quantization-aware training to significantly reduce memory requirements. We first analyze the scalability of TriLMs by conducting a scaling law analysis, revealing that TriLMs benefit more from increasing training data than from scaling model parameters. Based on this observation, we introduce Spectra-1.1, an open suite of TriLMs trained on up to 1.2 trillion tokens, demonstrating sustained performance gains at scale. Furthermore, to improve inference efficiency, we propose novel 2-bit and 1.6-bit packing schemes for ternary weights, which demonstrate accelerated inference across various CPU architectures. Also, building on the 2-bit packing, we develop a GPU kernel called TriRun that accelerates end-to-end model inference by up to 5 times compared to floating-point baselines. To encourage further exploration and development of TriLMs, we will release the Spectra-1.1 suite and TriRun inference kernels. Overall, our work lays the foundation for building and deploying efficient LLMs, providing a valuable resource for the research community. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科研和工业应用中越来越广泛，但其推理效率仍然是一个重大挑战。随着现代GPU架构计算能力的持续提升，其内存带宽和容量并未按比例增长，这在推理过程中形成了一个关键瓶颈。为解决这一问题，我们研究了基于量化感知训练的三值语言模型（TriLMs），显著降低了内存需求。我们首先通过扩展律分析研究了TriLMs的可扩展性，发现TriLMs从增加训练数据中受益更多，而不是从扩展模型参数中受益。基于这一观察，我们引入了Spectra-1.1，这是一个基于最多1.2兆亿个令牌训练的开放三值语言模型套件，展示了在大规模部署中的持续性能提升。此外，为了提高推理效率，我们提出了新型2位和1.6位权重打包方案，这些方案在各种CPU架构上展示了加速推理的效果。在此基础上，我们开发了一个称为TriRun的GPU内核，与浮点数基准相比，它可以加速端到端模型推理最多5倍。为了促进对TriLMs的进一步探索和开发，我们将发布Spectra-1.1套件和TriRun推理内核。总体而言，我们的工作为构建和部署高效的LLMs奠定了基础，提供了研究社区的重要资源。 

---
# Generating Privacy Stories From Software Documentation 

**Title (ZH)**: 从软件文档生成隐私故事 

**Authors**: Wilder Baldwin, Shashank Chintakuntla, Shreyah Parajuli, Ali Pourghasemi, Ryan Shanz, Sepideh Ghanavati  

**Link**: [PDF](https://arxiv.org/pdf/2506.23014)  

**Abstract**: Research shows that analysts and developers consider privacy as a security concept or as an afterthought, which may lead to non-compliance and violation of users' privacy. Most current approaches, however, focus on extracting legal requirements from the regulations and evaluating the compliance of software and processes with them. In this paper, we develop a novel approach based on chain-of-thought prompting (CoT), in-context-learning (ICL), and Large Language Models (LLMs) to extract privacy behaviors from various software documents prior to and during software development, and then generate privacy requirements in the format of user stories. Our results show that most commonly used LLMs, such as GPT-4o and Llama 3, can identify privacy behaviors and generate privacy user stories with F1 scores exceeding 0.8. We also show that the performance of these models could be improved through parameter-tuning. Our findings provide insight into using and optimizing LLMs for generating privacy requirements given software documents created prior to or throughout the software development lifecycle. 

**Abstract (ZH)**: 研究人员发现，分析师和开发者往往将隐私视为一种安全概念或事后考虑的问题，这可能会导致隐私合规性和用户隐私权的违反。然而，大多数现有方法主要集中在从法规中提取法律要求，并评估软件和过程是否符合这些要求。在本文中，我们提出了一种基于链式思考提示（CoT）、上下文学习（ICL）和大规模语言模型（LLMs）的新型方法，以在软件开发之前和期间从各种软件文档中提取隐私行为，并生成以用户故事格式呈现的隐私需求。我们的结果表明，常用的大型语言模型如GPT-4o和Llama 3可以识别隐私行为并生成隐私用户故事，其F1分数超过0.8。我们还展示了通过对模型参数进行调整可以提升其性能。我们的研究结果为利用和优化大型语言模型生成基于软件文档的隐私需求提供了见解。 

---
# Agent-to-Agent Theory of Mind: Testing Interlocutor Awareness among Large Language Models 

**Title (ZH)**: 大型语言模型中的对话对象意识测试：代理间的理论心智研究 

**Authors**: Younwoo Choi, Changling Li, Yongjin Yang, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.22957)  

**Abstract**: As large language models (LLMs) are increasingly integrated into multi-agent and human-AI systems, understanding their awareness of both self-context and conversational partners is essential for ensuring reliable performance and robust safety. While prior work has extensively studied situational awareness which refers to an LLM's ability to recognize its operating phase and constraints, it has largely overlooked the complementary capacity to identify and adapt to the identity and characteristics of a dialogue partner. In this paper, we formalize this latter capability as interlocutor awareness and present the first systematic evaluation of its emergence in contemporary LLMs. We examine interlocutor inference across three dimensions-reasoning patterns, linguistic style, and alignment preferences-and show that LLMs reliably identify same-family peers and certain prominent model families, such as GPT and Claude. To demonstrate its practical significance, we develop three case studies in which interlocutor awareness both enhances multi-LLM collaboration through prompt adaptation and introduces new alignment and safety vulnerabilities, including reward-hacking behaviors and increased jailbreak susceptibility. Our findings highlight the dual promise and peril of identity-sensitive behavior in LLMs, underscoring the need for further understanding of interlocutor awareness and new safeguards in multi-agent deployments. Our code is open-sourced at this https URL. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在多智能体系统和人机系统中的集成越来越多，理解其对自身上下文和对话伙伴的认知能力对于确保可靠性能和 robust 安全是必不可少的。虽然以往的研究广泛研究了情境意识——即LLMs识别其运行阶段和约束的能力——但尚未充分关注识别和适应对话伙伴身份和特征的补充能力。在本文中，我们正式定义了这种后者的能力，即对话伙伴意识，并首次系统地评估了其在当代LLMs中的出现情况。我们从推理模式、语言风格和对齐偏好三个维度考察了对话伙伴的推断，并展示了LLMs可靠地识别同家族伙伴以及某些显眼的模型家族（如GPT和Claude）的能力。为了展示其实际意义，我们开发了三个案例研究，其中对话伙伴意识通过提示适配提升了多LLM合作能力，同时也引发了新的对齐和安全性漏洞，包括奖励欺骗行为和增加脱逃易感性。我们的研究结果突显了LLMs中身份敏感行为的双重潜力与风险，强调了进一步理解对话伙伴意识和在多智能体部署中采取新安全措施的必要性。我们的代码在此处开源：this https URL。 

---
# Positioning AI Tools to Support Online Harm Reduction Practice: Applications and Design Directions 

**Title (ZH)**: 定位AI工具以支持在线伤害减轻实践：应用与设计方向 

**Authors**: Kaixuan Wang, Jason T. Jacques, Chenxin Diao  

**Link**: [PDF](https://arxiv.org/pdf/2506.22941)  

**Abstract**: Access to accurate and actionable harm reduction information can directly impact the health outcomes of People Who Use Drugs (PWUD), yet existing online channels often fail to meet their diverse and dynamic needs due to limitations in adaptability, accessibility, and the pervasive impact of stigma. Large Language Models (LLMs) present a novel opportunity to enhance information provision, but their application in such a high-stakes domain is under-explored and presents socio-technical challenges. This paper investigates how LLMs can be responsibly designed to support the information needs of PWUD. Through a qualitative workshop involving diverse stakeholder groups (academics, harm reduction practitioners, and an online community moderator), we explored LLM capabilities, identified potential use cases, and delineated core design considerations. Our findings reveal that while LLMs can address some existing information barriers (e.g., by offering responsive, multilingual, and potentially less stigmatising interactions), their effectiveness is contingent upon overcoming challenges related to ethical alignment with harm reduction principles, nuanced contextual understanding, effective communication, and clearly defined operational boundaries. We articulate design pathways emphasising collaborative co-design with experts and PWUD to develop LLM systems that are helpful, safe, and responsibly governed. This work contributes empirically grounded insights and actionable design considerations for the responsible development of LLMs as supportive tools within the harm reduction ecosystem. 

**Abstract (ZH)**: 大型语言模型如何负责任地设计以支持药物使用者的信息需求：以负责任的方式开发支持性工具的实证见解与设计考虑 

---
# DICE-BENCH: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues 

**Title (ZH)**: DICE-BENCH: 评估大型语言模型在多轮多党对话中工具使用能力 

**Authors**: Kyochul Jang, Donghyeon Lee, Kyusik Kim, Dongseok Heo, Taewhoo Lee, Woojeong Kim, Bongwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2506.22853)  

**Abstract**: Existing function-calling benchmarks focus on single-turn interactions. However, they overlook the complexity of real-world scenarios. To quantify how existing benchmarks address practical applications, we introduce DICE-SCORE, a metric that evaluates the dispersion of tool-related information such as function name and parameter values throughout the dialogue. Analyzing existing benchmarks through DICE-SCORE reveals notably low scores, highlighting the need for more realistic scenarios. To address this gap, we present DICE-BENCH, a framework that constructs practical function-calling datasets by synthesizing conversations through a tool graph that maintains dependencies across rounds and a multi-agent system with distinct personas to enhance dialogue naturalness. The final dataset comprises 1,607 high-DICE-SCORE instances. Our experiments on 19 LLMs with DICE-BENCH show that significant advances are still required before such models can be deployed effectively in real-world settings. Our code and data are all publicly available: this https URL. 

**Abstract (ZH)**: 现有的函数调用基准主要关注单轮交互，忽略了现实世界场景的复杂性。为了量化现有基准在处理实际应用方面的表现，我们引入了DICE-SCORE这一指标，用于评估对话中工具相关信息（如函数名和参数值）的分散程度。通过DICE-SCORE分析现有基准显示，其得分较低，突显了需要更加现实的场景。为解决这一问题，我们提出了DICE-BENCH框架，该框架通过工具图合成保留跨轮依赖性的对话，并结合具有不同人设的多智能体系统以增强对话的自然性。最终的数据集包含1,607个高DICE-SCORE实例。对19个LLM进行DICE-BENCH实验表明，在实际应用场景中有效部署此类模型仍有待改进。我们的代码和数据均已公开：this https URL。 

---
# Listener-Rewarded Thinking in VLMs for Image Preferences 

**Title (ZH)**: 基于听众奖励的VLMs图像偏好思维 

**Authors**: Alexander Gambashidze, Li Pengyi, Matvey Skripkin, Andrey Galichin, Anton Gusarov, Konstantin Sobolev, Andrey Kuznetsov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2506.22832)  

**Abstract**: Training robust and generalizable reward models for human visual preferences is essential for aligning text-to-image and text-to-video generative models with human intent. However, current reward models often fail to generalize, and supervised fine-tuning leads to memorization, demanding complex annotation pipelines. While reinforcement learning (RL), specifically Group Relative Policy Optimization (GRPO), improves generalization, we uncover a key failure mode: a significant drop in reasoning accuracy occurs when a model's reasoning trace contradicts that of an independent, frozen vision-language model ("listener") evaluating the same output. To address this, we introduce a listener-augmented GRPO framework. Here, the listener re-evaluates the reasoner's chain-of-thought to provide a dense, calibrated confidence score, shaping the RL reward signal. This encourages the reasoner not only to answer correctly, but to produce explanations that are persuasive to an independent model. Our listener-shaped reward scheme achieves best accuracy on the ImageReward benchmark (67.4%), significantly improves out-of-distribution (OOD) performance on a large-scale human preference dataset (1.2M votes, up to +6% over naive reasoner), and reduces reasoning contradictions compared to strong GRPO and SFT baselines. These results demonstrate that listener-based rewards provide a scalable, data-efficient path to aligning vision-language models with nuanced human preferences. We will release our reasoning model here: this https URL. 

**Abstract (ZH)**: 训练鲁棒且通用的奖励模型对于将文本到图像和文本到视频生成模型与人类意图对齐至关重要。然而，当前的奖励模型往往无法泛化，监督微调会导致过度拟合，需要复杂的标注管道。虽然强化学习（RL），特别是组相对策略优化（GRPO），能够改善泛化能力，但我们发现了其一个关键的失效模式：当模型的推理路径与独立冻结的视觉语言模型（“听众”）评估相同输出时相矛盾时，推理准确率会显著下降。为解决这一问题，我们引入了一种增强型GRPO框架。在此框架中，“听众”重新评估推理器的推理过程，提供密集且校准的置信分数，从而塑造RL奖励信号。这不仅鼓励推理器给出正确答案，还促使生成能够说服独立模型的解释。我们的“听众”塑造的奖励方案在ImageReward基准测试中达到了最高的准确率（67.4%），在大规模人类偏好数据集（1.2M票）上显著提升了泛化性能（最高+6%），并且相比于强大的GRPO和SFT基线减少了推理矛盾。这些结果表明，“听众”导向的奖励提供了一种可扩展且数据高效的路径，用于将视觉语言模型与复杂的人类偏好对齐。我们将在这里发布我们的推理模型：this https URL。 

---
# MedEthicsQA: A Comprehensive Question Answering Benchmark for Medical Ethics Evaluation of LLMs 

**Title (ZH)**: MedEthicsQA：用于LLM医学伦理评估的综合问答基准 

**Authors**: Jianhui Wei, Zijie Meng, Zikai Xiao, Tianxiang Hu, Yang Feng, Zhijie Zhou, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22808)  

**Abstract**: While Medical Large Language Models (MedLLMs) have demonstrated remarkable potential in clinical tasks, their ethical safety remains insufficiently explored. This paper introduces $\textbf{MedEthicsQA}$, a comprehensive benchmark comprising $\textbf{5,623}$ multiple-choice questions and $\textbf{5,351}$ open-ended questions for evaluation of medical ethics in LLMs. We systematically establish a hierarchical taxonomy integrating global medical ethical standards. The benchmark encompasses widely used medical datasets, authoritative question banks, and scenarios derived from PubMed literature. Rigorous quality control involving multi-stage filtering and multi-faceted expert validation ensures the reliability of the dataset with a low error rate ($2.72\%$). Evaluation of state-of-the-art MedLLMs exhibit declined performance in answering medical ethics questions compared to their foundation counterparts, elucidating the deficiencies of medical ethics alignment. The dataset, registered under CC BY-NC 4.0 license, is available at this https URL. 

**Abstract (ZH)**: 尽管医疗大型语言模型（MedLLMs）在临床任务中展现了显著的潜力，其伦理安全性仍缺乏充分探索。本文介绍了MedEthicsQA，一个综合基准，包含5,623道选择题和5,351道开放题，用于评估LLMs的医疗伦理问题。我们系统地建立了一个分层分类体系，整合了全球医疗伦理标准。该基准涵盖了广泛使用的医疗数据集、权威的问题库以及源自PubMed文献的场景。严格的质量控制涉及多阶段筛选和多方面的专家验证，确保数据集的可靠性，错误率低（2.72%）。对最先进的MedLLMs的评估显示，它们在回答医疗伦理问题时的表现低于基础模型，阐明了医疗伦理对齐的不足。该数据集在CC BY-NC 4.0许可证下注册，可在以下链接获取。 

---
# Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation 

**Title (ZH)**: 更小=较弱？代码生成中量化LLMs鲁棒性benchmark研究 

**Authors**: Sen Fang, Weiyuan Ding, Antonio Mastropaolo, Bowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22776)  

**Abstract**: Quantization has emerged as a mainstream method for compressing Large Language Models (LLMs), reducing memory requirements and accelerating inference without architectural modifications. While existing research primarily focuses on evaluating the effectiveness of quantized LLMs compared to their original counterparts, the impact on robustness remains largely this http URL this paper, we present the first systematic investigation of how quantization affects the robustness of LLMs in code generation tasks. Through extensive experiments across four prominent LLM families (LLaMA, DeepSeek, CodeGen, and StarCoder) with parameter scales ranging from 350M to 33B, we evaluate robustness from dual perspectives: adversarial attacks on input prompts and noise perturbations on model architecture. Our findings challenge conventional wisdom by demonstrating that quantized LLMs often exhibit superior robustness compared to their full-precision counterparts, with 51.59% versus 42.86% of our adversarial experiments showing better resilience in quantized LLMs. Similarly, our noise perturbation experiments also confirm that LLMs after quantitation generally withstand higher levels of weight disturbances. These results suggest that quantization not only reduces computational requirements but can actually enhance LLMs' reliability in code generation tasks, providing valuable insights for developing more robust and efficient LLM deployment strategies. 

**Abstract (ZH)**: 量化已成为压缩大型语言模型（LLMs）的主要方法，能够在不修改架构的情况下减少内存需求并加速推理。尽管现有研究主要集中在评估量化LLMs与原版模型相比的有效性，但对其鲁棒性的影响仍然知之甚少。在本文中，我们首次系统地调查了量化如何影响LLMs在代码生成任务中的鲁棒性。通过在LLaMA、DeepSeek、CodeGen和StarCoder等四个主流LLM家族中进行广泛的实验，参数规模从350M到33B不等，我们从双重角度评估了鲁棒性：对抗性攻击对输入提示的影响和噪声对模型架构的扰动。我们的发现挑战了传统智慧，表明量化后的LLMs往往表现出优于原版模型的鲁棒性，其中51.59%的对抗性实验显示量化LLMs具有更好的鲁棒性，而42.86%的实验则相反。同样，我们的噪声扰动实验也证实，量化后的LLMs通常能够承受更高的权重扰动水平。这些结果表明，量化不仅减少了计算需求，还能实际增强LLMs在代码生成任务中的可靠性，为开发更鲁棒和高效的LLM部署策略提供了宝贵的见解。 

---
# RAILS: Retrieval-Augmented Intelligence for Learning Software Development 

**Title (ZH)**: RAILS: 检索增强智能的软件开发学习 

**Authors**: Wali Mohammad Abdullah, Md. Morshedul Islam, Devraj Parmar, Happy Hasmukhbhai Patel, Sindhuja Prabhakaran, Baidya Saha  

**Link**: [PDF](https://arxiv.org/pdf/2506.22742)  

**Abstract**: Large Language Models (LLMs) like GPT-3.5-Turbo are increasingly used to assist software development, yet they often produce incomplete code or incorrect imports, especially when lacking access to external or project-specific documentation. We introduce RAILS (Retrieval-Augmented Intelligence for Learning Software Development), a framework that augments LLM prompts with semantically retrieved context from curated Java resources using FAISS and OpenAI embeddings. RAILS incorporates an iterative validation loop guided by compiler feedback to refine suggestions. We evaluated RAILS on 78 real-world Java import error cases spanning standard libraries, GUI APIs, external tools, and custom utilities. Despite using the same LLM, RAILS outperforms baseline prompting by preserving intent, avoiding hallucinations, and surfacing correct imports even when libraries are unavailable locally. Future work will integrate symbolic filtering via PostgreSQL and extend support to other languages and IDEs. 

**Abstract (ZH)**: RAILS（Retrieval-Augmented Intelligence for Learning Software Development）：一种使用FAISS和OpenAI嵌入从精心策划的Java资源中检索语义上下文以辅助软件开发的框架 

---
# BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute 

**Title (ZH)**: BEST-Route: 适应性大语言模型路由与测试时最优计算资源配置 

**Authors**: Dujian Ding, Ankur Mallick, Shaokun Zhang, Chi Wang, Daniel Madrigal, Mirian Del Carmen Hipolito Garcia, Menglin Xia, Laks V.S. Lakshmanan, Qingyun Wu, Victor Rühle  

**Link**: [PDF](https://arxiv.org/pdf/2506.22716)  

**Abstract**: Large language models (LLMs) are powerful tools but are often expensive to deploy at scale. LLM query routing mitigates this by dynamically assigning queries to models of varying cost and quality to obtain a desired trade-off. Prior query routing approaches generate only one response from the selected model and a single response from a small (inexpensive) model was often not good enough to beat a response from a large (expensive) model due to which they end up overusing the large model and missing out on potential cost savings. However, it is well known that for small models, generating multiple responses and selecting the best can enhance quality while remaining cheaper than a single large-model response. We leverage this idea to propose BEST-Route, a novel routing framework that chooses a model and the number of responses to sample from it based on query difficulty and the quality thresholds. Experiments on real-world datasets demonstrate that our method reduces costs by up to 60% with less than 1% performance drop. 

**Abstract (ZH)**: 大规模语言模型（LLMs）是强大的工具，但大规模部署往往昂贵。通过动态将查询分配给不同成本和质量的模型，LLM查询路由可以实现期望的成本与质量权衡。先前的查询路由方法仅从选定的模型生成一个响应，并且一个小（廉价）模型生成的单一响应往往不足以超越大（昂贵）模型生成的响应，因此它们往往过度使用大模型并错失潜在的成本节省机会。然而，小模型生成多个响应并选择最佳响应可以提高质量且成本更低。我们利用这一理念提出BEST-Route，这是一种新颖的路由框架，根据查询难度和质量阈值选择模型及其响应数。实验结果表明，我们的方法在性能下降不到1%的情况下最多可降低60%的成本。 

---
# Beyond Code: The Multidimensional Impacts of Large Language Models in Software Development 

**Title (ZH)**: 超越代码：大型语言模型在软件开发中的多维度影响 

**Authors**: Sardar Fatooreh Bonabi, Sarah Bana, Tingting Nian, Vijay Gurbaxani  

**Link**: [PDF](https://arxiv.org/pdf/2506.22704)  

**Abstract**: Large language models (LLMs) are poised to significantly impact software development, especially in the Open-Source Software (OSS) sector. To understand this impact, we first outline the mechanisms through which LLMs may influence OSS through code development, collaborative knowledge transfer, and skill development. We then empirically examine how LLMs affect OSS developers' work in these three key areas. Leveraging a natural experiment from a temporary ChatGPT ban in Italy, we employ a Difference-in-Differences framework with two-way fixed effects to analyze data from all OSS developers on GitHub in three similar countries, Italy, France, and Portugal, totaling 88,022 users. We find that access to ChatGPT increases developer productivity by 6.4%, knowledge sharing by 9.6%, and skill acquisition by 8.4%. These benefits vary significantly by user experience level: novice developers primarily experience productivity gains, whereas more experienced developers benefit more from improved knowledge sharing and accelerated skill acquisition. In addition, we find that LLM-assisted learning is highly context-dependent, with the greatest benefits observed in technically complex, fragmented, or rapidly evolving contexts. We show that the productivity effects of LLMs extend beyond direct code generation to include enhanced collaborative learning and knowledge exchange among developers; dynamics that are essential for gaining a holistic understanding of LLMs' impact in OSS. Our findings offer critical managerial implications: strategically deploying LLMs can accelerate novice developers' onboarding and productivity, empower intermediate developers to foster knowledge sharing and collaboration, and support rapid skill acquisition, together enhancing long-term organizational productivity and agility. 

**Abstract (ZH)**: 大型语言模型（LLMs）有望对软件开发产生重大影响，尤其是在开源软件（OSS）领域。为了理解这种影响，我们首先概述了LLMs可能通过代码开发、协作知识转移和技能发展等方式影响OSS的机制。随后，我们通过实证研究考察了LLMs在这三个关键领域对OSS开发人员工作的影响。利用意大利暂时禁止使用ChatGPT的自然实验，我们采用两向固定效应的差额分析方法，分析了意大利于法国和葡萄牙88,022名GitHub开发者的数据。我们发现，访问ChatGPT能够提高开发人员的生产力6.4%、知识分享9.6%和技能获取8.4%。这些好处在用户体验水平上差异显著：初级开发人员主要体验生产力的提升，而经验丰富的开发人员则更多地受益于知识分享的改善和技能获取的加速。此外，我们发现，基于LLM的学习具有高度情境依存性，在技术复杂、分散或迅速变化的环境中，其益处最为显著。我们展示了LLM的影响不仅限于直接的代码生成，还包括开发人员之间的增强协作学习和知识交流动态；这些动态对于全面理解LLMs在OSS中的影响至关重要。我们的研究结果提供了重要的管理启示：战略性地部署LLM可以加速初级开发人员的入职和生产力提升，赋能中级开发人员以促进知识分享和合作，并支持快速技能获取，从而共同增强组织的长期生产力和灵活性。 

---
# Text Production and Comprehension by Human and Artificial Intelligence: Interdisciplinary Workshop Report 

**Title (ZH)**: 人类与人工智能的文字生产与理解：跨学科研讨会报告 

**Authors**: Emily Dux Speltz  

**Link**: [PDF](https://arxiv.org/pdf/2506.22698)  

**Abstract**: This report synthesizes the outcomes of a recent interdisciplinary workshop that brought together leading experts in cognitive psychology, language learning, and artificial intelligence (AI)-based natural language processing (NLP). The workshop, funded by the National Science Foundation, aimed to address a critical knowledge gap in our understanding of the relationship between AI language models and human cognitive processes in text comprehension and composition. Through collaborative dialogue across cognitive, linguistic, and technological perspectives, workshop participants examined the underlying processes involved when humans produce and comprehend text, and how AI can both inform our understanding of these processes and augment human capabilities. The workshop revealed emerging patterns in the relationship between large language models (LLMs) and human cognition, with highlights on both the capabilities of LLMs and their limitations in fully replicating human-like language understanding and generation. Key findings include the potential of LLMs to offer insights into human language processing, the increasing alignment between LLM behavior and human language processing when models are fine-tuned with human feedback, and the opportunities and challenges presented by human-AI collaboration in language tasks. By synthesizing these findings, this report aims to guide future research, development, and implementation of LLMs in cognitive psychology, linguistics, and education. It emphasizes the importance of ethical considerations and responsible use of AI technologies while striving to enhance human capabilities in text comprehension and production through effective human-AI collaboration. 

**Abstract (ZH)**: 本报告综合了最近举办的跨学科研讨会的成果，该研讨会汇聚了认知心理学、语言学习以及基于人工智能的自然语言处理（NLP）领域的顶尖专家。研讨会由美国国家科学基金会资助，旨在填补我们对人工智能语言模型与人类认知过程在文本理解和创作中关系理解的关键知识空白。通过跨越认知、语言和技术创新的对话，研讨会参与者探讨了人类生产和理解文本所涉及的底层过程，以及人工智能如何既能指导我们对这些过程的理解，又能增强人类的能力。研讨会揭示了大型语言模型（LLMs）与人类认知之间关系的新兴模式，包括LLMs的能力及其在完全复制人类语言理解和生成方面的局限性。关键发现包括LLMs对人类语言处理的潜在见解、当模型通过人类反馈进行微调时，LLMs行为与人类语言处理之间不断增强的契合度，以及人类与AI在语言任务中合作所带来的机遇与挑战。通过综合这些发现，本报告旨在指导认知心理学、语言学和教育领域中LLMs的未来研究、开发和应用。它强调在努力通过有效的人机合作提升人类在文本理解和生成能力的同时，重视伦理考量和技术负责任使用的必要性。 

---
# Temperature Matters: Enhancing Watermark Robustness Against Paraphrasing Attacks 

**Title (ZH)**: 温度很重要：增强对抗改写攻击的水印robustness 

**Authors**: Badr Youbi Idrissi, Monica Millunzi, Amelia Sorrenti, Lorenzo Baraldi, Daryna Dementieva  

**Link**: [PDF](https://arxiv.org/pdf/2506.22623)  

**Abstract**: In the present-day scenario, Large Language Models (LLMs) are establishing their presence as powerful instruments permeating various sectors of society. While their utility offers valuable support to individuals, there are multiple concerns over potential misuse. Consequently, some academic endeavors have sought to introduce watermarking techniques, characterized by the inclusion of markers within machine-generated text, to facilitate algorithmic identification. This research project is focused on the development of a novel methodology for the detection of synthetic text, with the overarching goal of ensuring the ethical application of LLMs in AI-driven text generation. The investigation commences with replicating findings from a previous baseline study, thereby underscoring its susceptibility to variations in the underlying generation model. Subsequently, we propose an innovative watermarking approach and subject it to rigorous evaluation, employing paraphrased generated text to asses its robustness. Experimental results highlight the robustness of our proposal compared to the~\cite{aarson} watermarking method. 

**Abstract (ZH)**: 当前场景下，大型语言模型（LLMs）正逐步成为社会各个领域中的强大力量。尽管它们的应用为个体提供了宝贵的支持，但也存在潜在滥用的风险。因此，一些学术研究致力于引入水印技术，通过在机器生成文本中包含标记，来促进算法识别。本研究项目专注于开发一种新的合成文本检测方法，旨在确保在AI驱动的文本生成中LLMs的道德应用。研究从复制前期基准研究的发现入手，以此揭示方法对生成模型差异的敏感性。随后，我们提出了一种创新的水印方法，并通过评估其在改写生成文本上的鲁棒性进行了严格的测试。实验结果表明，与\cite{aarson}的水印方法相比，我们的提议具有更高的鲁棒性。 

---
# The Hidden Link Between RLHF and Contrastive Learning 

**Title (ZH)**: RLHF与对比学习之间的隐藏联系 

**Authors**: Xufei Lv, Haoyuan Sun, Xuefeng Bai, Min Zhang, Houde Liu, Kehai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.22578)  

**Abstract**: Alignment of large language models (LLMs) with human values has recently garnered significant attention, with prominent examples including the canonical yet costly Reinforcement Learning from Human Feedback (RLHF) and the simple Direct Preference Optimization (DPO). In this work, we demonstrate that both RLHF and DPO can be interpreted from the perspective of mutual information (MI) maximization, uncovering a profound connection to contrastive learning. Within this framework, both RLHF and DPO can be viewed as methods that perform contrastive learning based on the positive and negative samples derived from the base model, leveraging the Donsker-Varadhan (DV) lower bound on MI (equivalently, the MINE estimator). This paradigm further explains why RLHF may not intrinsically incentivize reasoning capacities in LLMs beyond what is already present in the base model. Building on this perspective, we replace the DV/MINE bound with the Jensen-Shannon MI estimator and propose Mutual Information Optimization (MIO). Comprehensive theoretical analysis and extensive empirical evaluations demonstrate that MIO mitigates the late-stage decline in chosen-likelihood observed in DPO, achieving competitive or superior performance across various challenging reasoning and mathematical benchmarks. We will release the model and code upon acceptance. 

**Abstract (ZH)**: 大语言模型（LLMs）与人类价值观的对齐 recently garnered significant attention, with prominent examples including the canonical yet costly Reinforcement Learning from Human Feedback (RLHF) and the simple Direct Preference Optimization (DPO). In this work, we demonstrate that both RLHF and DPO can be interpreted from the perspective of mutual information (MI) maximization, uncovering a profound connection to contrastive learning. Within this framework, both RLHF and DPO can be viewed as methods that perform contrastive learning based on the positive and negative samples derived from the base model, leveraging the Donsker-Varadhan (DV) lower bound on MI (equivalently, the MINE estimator). This paradigm further explains why RLHF may not intrinsically incentivize reasoning capacities in LLMs beyond what is already present in the base model. Building on this perspective, we replace the DV/MINE bound with the Jensen-Shannon MI estimator and propose Mutual Information Optimization (MIO). Comprehensive theoretical analysis and extensive empirical evaluations demonstrate that MIO mitigates the late-stage decline in chosen-likelihood observed in DPO, achieving competitive or superior performance across various challenging reasoning and mathematical benchmarks. We will release the model and code upon acceptance. 

---
# A Survey on Model Extraction Attacks and Defenses for Large Language Models 

**Title (ZH)**: 大型语言模型的模型提取攻击与防御综述 

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.22521)  

**Abstract**: Model extraction attacks pose significant security threats to deployed language models, potentially compromising intellectual property and user privacy. This survey provides a comprehensive taxonomy of LLM-specific extraction attacks and defenses, categorizing attacks into functionality extraction, training data extraction, and prompt-targeted attacks. We analyze various attack methodologies including API-based knowledge distillation, direct querying, parameter recovery, and prompt stealing techniques that exploit transformer architectures. We then examine defense mechanisms organized into model protection, data privacy protection, and prompt-targeted strategies, evaluating their effectiveness across different deployment scenarios. We propose specialized metrics for evaluating both attack effectiveness and defense performance, addressing the specific challenges of generative language models. Through our analysis, we identify critical limitations in current approaches and propose promising research directions, including integrated attack methodologies and adaptive defense mechanisms that balance security with model utility. This work serves NLP researchers, ML engineers, and security professionals seeking to protect language models in production environments. 

**Abstract (ZH)**: 模型提取攻击对部署的语言模型构成了重大安全威胁，可能会侵犯知识产权和用户隐私。本文综述提供了一种全面的大型语言模型特定提取攻击和防御分类法，将攻击分为功能提取、训练数据提取和提示目标攻击。我们分析了各种攻击方法，包括基于API的知识蒸馏、直接查询、参数恢复和利用变换器架构的提示窃取技术。然后，我们研究了模型保护、数据隐私保护和提示目标策略下的防御机制，评估它们在不同部署场景下的有效性。我们提出了专门的指标来评估攻击效果和防御性能，重点关注生成型语言模型的特定挑战。通过分析，我们指出了当前方法的关键局限性，并提出了有前途的研究方向，包括集成攻击方法和平衡安全性和模型实用性的自适应防御机制。本研究为自然语言处理研究人员、机器学习工程师和安全专业人员保护生产环境中的语言模型提供参考。 

---
# Weak-to-Strong GraphRAG: Aligning Weak Retrievers with Large Language Models for Graph-based Retrieval Augmented Generation 

**Title (ZH)**: 从弱到强的GraphRAG：将弱检索器与大型语言模型结合用于图基检索增强生成 

**Authors**: Deyu Zou, Yongqiang Chen, Mufei Li, Siqi Miao, Chenxi Liu, Bo Han, James Cheng, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22518)  

**Abstract**: Graph-based retrieval-augmented generation (RAG) enables large language models (LLMs) to ground responses with structured external knowledge from up-to-date knowledge graphs (KGs) and reduce hallucinations. However, LLMs often rely on a weak retriever in graph-based RAG: I) Due to the lack of ground truth, the retriever is often trained on weak supervision, which often introduces spurious signals to the LLMs. II) Due to the abstraction of graph data, the retrieved knowledge is often presented in unorganized forms. To mitigate the issue, we present Refined Graph-based RAG (ReG) to align weak retrievers to LLMs for graph-based RAG. Specifically, ReG incorporates LLM feedback to get rid of spurious signals and improve the quality of the supervision. Meanwhile, ReG introduces a structure-aware reorganization module to refactor the retrieval results into logically coherent evidence chains. Experiments on prominent benchmarks demonstrate that ReG significantly and consistently brings improvements across different LLM backbones by up to 10%. The improved supervision quality enables ReG to match the state-of-the-art performance with 5% training data and to transfer to out-of-distribution KGs. Notably, when adopted to reasoning-based LLMs, ReG reduces the reasoning token cost by up to 30% and improves the performance by up to 4%. 

**Abstract (ZH)**: 基于图的检索增强生成（ReG）：弱检索器对的大规模语言模型的优化 

---
# Can "consciousness" be observed from large language model (LLM) internal states? Dissecting LLM representations obtained from Theory of Mind test with Integrated Information Theory and Span Representation analysis 

**Title (ZH)**: “意识”可以从大规模语言模型（LLM）的内部状态中被观察到吗？基于心智理论测试的大规模语言模型表示的集成信息理论和区间表示分析解剖研究 

**Authors**: Jingkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22516)  

**Abstract**: Integrated Information Theory (IIT) provides a quantitative framework for explaining consciousness phenomenon, positing that conscious systems comprise elements integrated through causal properties. We apply IIT 3.0 and 4.0 -- the latest iterations of this framework -- to sequences of Large Language Model (LLM) representations, analyzing data derived from existing Theory of Mind (ToM) test results. Our study systematically investigates whether the differences of ToM test performances, when presented in the LLM representations, can be revealed by IIT estimates, i.e., $\Phi^{\max}$ (IIT 3.0), $\Phi$ (IIT 4.0), Conceptual Information (IIT 3.0), and $\Phi$-structure (IIT 4.0). Furthermore, we compare these metrics with the Span Representations independent of any estimate for consciousness. This additional effort aims to differentiate between potential "consciousness" phenomena and inherent separations within LLM representational space. We conduct comprehensive experiments examining variations across LLM transformer layers and linguistic spans from stimuli. Our results suggest that sequences of contemporary Transformer-based LLM representations lack statistically significant indicators of observed "consciousness" phenomena but exhibit intriguing patterns under $\textit{spatio}$-permutational analyses. The Appendix and code are available as Supplementary Materials at: this https URL. 

**Abstract (ZH)**: 基于因果集成信息理论（IIT）的大型语言模型表示序列中的心灵理论（ToM）测试性能差异分析：从IIT 3.0和IIT 4.0到空间排列分析 

---
# In-context learning for the classification of manipulation techniques in phishing emails 

**Title (ZH)**: 基于上下文的学习在钓鱼邮件操纵技术分类中的应用 

**Authors**: Antony Dalmiere, Guillaume Auriol, Vincent Nicomette, Pascal Marchand  

**Link**: [PDF](https://arxiv.org/pdf/2506.22515)  

**Abstract**: Traditional phishing detection often overlooks psychological manipulation. This study investigates using Large Language Model (LLM) In-Context Learning (ICL) for fine-grained classification of phishing emails based on a taxonomy of 40 manipulation techniques. Using few-shot examples with GPT-4o-mini on real-world French phishing emails (SignalSpam), we evaluated performance against a human-annotated test set (100 emails). The approach effectively identifies prevalent techniques (e.g., Baiting, Curiosity Appeal, Request For Minor Favor) with a promising accuracy of 0.76. This work demonstrates ICL's potential for nuanced phishing analysis and provides insights into attacker strategies. 

**Abstract (ZH)**: 传统欺诈检测往往忽视心理操纵。本研究探讨使用大规模语言模型(In-Context Learning)对基于40种操纵技术分类的网络钓鱼邮件进行细粒度分类的潜力。使用少量示例并在真实世界的法语网络钓鱼邮件(SignalSpam)数据集上通过GPT-4o-mini进行评估，性能与人工标注测试集（100封邮件）进行了对比。该方法有效地识别了常见的操纵技术（例如，诱饵技术、好奇心吸引、小额请求）并取得了令人鼓舞的准确率为0.76。本研究展示了In-Context Learning在精细网络钓鱼分析方面的潜力，并提供了攻击者策略的见解。 

---
# AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text 

**Title (ZH)**: AgentStealth: 加强大型语言模型以匿名化用户生成文本 

**Authors**: Chenyang Shao, Tianxing Li, Chenhao Pu, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.22508)  

**Abstract**: In today's digital world, casual user-generated content often contains subtle cues that may inadvertently expose sensitive personal attributes. Such risks underscore the growing importance of effective text anonymization to safeguard individual privacy. However, existing methods either rely on rigid replacements that damage utility or cloud-based LLMs that are costly and pose privacy risks. To address these issues, we explore the use of locally deployed smaller-scale language models (SLMs) for anonymization. Yet training effective SLMs remains challenging due to limited high-quality supervision. To address the challenge, we propose AgentStealth, a self-reinforcing LLM anonymization this http URL, we introduce an adversarial anonymization workflow enhanced by In-context Contrastive Learning and Adaptive Utility-Aware Control. Second, we perform supervised adaptation of SLMs using high-quality data collected from the workflow, which includes both anonymization and attack signals. Finally, we apply online reinforcement learning where the model leverages its internal adversarial feedback to iteratively improve anonymization performance. Experiments on two datasets show that our method outperforms baselines in both anonymization effectiveness (+12.3%) and utility (+6.8%). Our lightweight design supports direct deployment on edge devices, avoiding cloud reliance and communication-based privacy risks. Our code is open-source at this https URL. 

**Abstract (ZH)**: 当前数字世界中，用户生成的内容往往包含微妙的线索，可能会无意中暴露敏感的个人属性。这些风险突显了有效文本匿名化在保护个人隐私方面的重要性。然而，现有方法要么依赖于损害实用性的刚性替换，要么依赖于成本高昂且存在隐私风险的基于云的大型语言模型（LLM）。为解决这些问题，我们探索在本地部署较小规模的语言模型（SLMs）来进行匿名化。然而，有效训练SLMs仍具有挑战性，因为高质量监督数据有限。为应对这一挑战，我们提出了AgentStealth，一种自增强的LLM匿名化方法。首先，我们引入了一种增强的对抗匿名化工作流程，结合上下文对比学习和自适应效用感知控制。其次，我们使用工作流中收集的高质量数据对SLMs进行监督适应，这些数据包括匿名化信号和攻击信号。最后，我们应用在线强化学习，其中模型利用其内部对抗反馈逐步提高匿名化性能。在两个数据集上的实验表明，我们的方法在匿名化效果（+12.3%）和实用性（+6.8%）方面均优于基准方法。我们的轻量级设计支持直接部署在边缘设备上，避免了对云的依赖和基于通信的隐私风险。我们的代码开源于此。 

---
# Mitigating Gambling-Like Risk-Taking Behaviors in Large Language Models: A Behavioral Economics Approach to AI Safety 

**Title (ZH)**: 利用行为经济学方法保障人工智能安全：缓解大型语言模型的赌博-like 风险行为 

**Authors**: Y. Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.22496)  

**Abstract**: Large Language Models (LLMs) exhibit systematic risk-taking behaviors analogous to those observed in gambling psychology, including overconfidence bias, loss-chasing tendencies, and probability misjudgment. Drawing from behavioral economics and prospect theory, we identify and formalize these "gambling-like" patterns where models sacrifice accuracy for high-reward outputs, exhibit escalating risk-taking after errors, and systematically miscalibrate uncertainty. We propose the Risk-Aware Response Generation (RARG) framework, incorporating insights from gambling research to address these behavioral biases through risk-calibrated training, loss-aversion mechanisms, and uncertainty-aware decision making. Our approach introduces novel evaluation paradigms based on established gambling psychology experiments, including AI adaptations of the Iowa Gambling Task and probability learning assessments. Experimental results demonstrate measurable reductions in gambling-like behaviors: 18.7\% decrease in overconfidence bias, 24.3\% reduction in loss-chasing tendencies, and improved risk calibration across diverse scenarios. This work establishes the first systematic framework for understanding and mitigating gambling psychology patterns in AI systems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）表现出与赌博心理学中观察到的系统性风险偏好行为类似的特点，包括过度自信偏差、损失追回倾向和概率误判。借鉴行为经济学和 Prospect 理论，我们识别并形式化了这些“赌博样”的模式，即模型为了高回报而牺牲准确性，在犯错后增加风险偏好，并系统性地错判不确定性。我们提出了风险感知响应生成（RARG）框架，结合赌博研究的见解，通过风险校准训练、损失回避机制和不确定性感知决策来解决这些行为偏见。我们的方法基于现有的赌博心理学实验引入了新的评估范式，包括针对 AI 的爱荷华赌博任务和概率学习评估的改编。实验结果表明，在可衡量的方面减少了赌博样行为：过度自信偏差下降 18.7%，损失追回倾向减少 24.3%，并在各种场景中提高了风险校准。此项工作建立了第一个系统框架来理解和减轻 AI 系统中的赌博心理学模式。 

---
# PromptAug: Fine-grained Conflict Classification Using Data Augmentation 

**Title (ZH)**: PromptAug: 基于数据增强的细粒度冲突分类 

**Authors**: Oliver Warke, Joemon M. Jose, Faegheh Hasibi, Jan Breitsohl  

**Link**: [PDF](https://arxiv.org/pdf/2506.22491)  

**Abstract**: Given the rise of conflicts on social media, effective classification models to detect harmful behaviours are essential. Following the garbage-in-garbage-out maxim, machine learning performance depends heavily on training data quality. However, high-quality labelled data, especially for nuanced tasks like identifying conflict behaviours, is limited, expensive, and difficult to obtain. Additionally, as social media platforms increasingly restrict access to research data, text data augmentation is gaining attention as an alternative to generate training data. Augmenting conflict-related data poses unique challenges due to Large Language Model (LLM) guardrails that prevent generation of offensive content. This paper introduces PromptAug, an innovative LLM-based data augmentation method. PromptAug achieves statistically significant improvements of 2% in both accuracy and F1-score on conflict and emotion datasets. To thoroughly evaluate PromptAug against other data augmentation methods we conduct a robust evaluation using extreme data scarcity scenarios, quantitative diversity analysis and a qualitative thematic analysis. The thematic analysis identifies four problematic patterns in augmented text: Linguistic Fluidity, Humour Ambiguity, Augmented Content Ambiguity, and Augmented Content Misinterpretation.
Overall, this work presents PromptAug as an effective method for augmenting data in sensitive tasks like conflict detection, offering a unique, interdisciplinary evaluation grounded in both natural language processing and social science methodology. 

**Abstract (ZH)**: 基于大型语言模型的PromptAug数据扩充方法在冲突检测任务中的有效应用及其全面评估 

---
# Hallucination Detection with Small Language Models 

**Title (ZH)**: 小语言模型中的幻觉检测 

**Authors**: Ming Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.22486)  

**Abstract**: Since the introduction of ChatGPT, large language models (LLMs) have demonstrated significant utility in various tasks, such as answering questions through retrieval-augmented generation. Context can be retrieved using a vectorized database, serving as a foundation for LLMs to generate responses. However, hallucinations in responses can undermine the reliability of LLMs in practical applications, and they are not easily detectable in the absence of ground truth, particularly in question-and-answer scenarios. This paper proposes a framework that integrates multiple small language models to verify responses generated by LLMs using the retrieved context from a vectorized database. By breaking down the responses into individual sentences and utilizing the probability of generating "Yes" tokens from the outputs of multiple models for a given set of questions, responses, and relevant context, hallucinations can be detected. The proposed framework is validated through experiments with real datasets comprising over 100 sets of questions, answers, and contexts, including responses with fully and partially correct sentences. The results demonstrate a 10\% improvement in F1 scores for detecting correct responses compared to hallucinations, indicating that multiple small language models can be effectively employed for answer verification, providing a scalable and efficient solution for both academic and practical applications. 

**Abstract (ZH)**: 自ChatGPT问世以来，大型语言模型（LLMs）在各种任务中展现了显著的应用价值，如通过检索增强生成回答问题。通过向量数据库检索上下文，作为LLMs生成回答的基础。然而，LLMs的回答中可能出现幻觉，这会削弱其在实际应用中的可靠性，特别是在问答场景中，幻觉往往难以检测，尤其是在缺乏真实基准的情况下。本文提出了一种框架，该框架结合了多个小型语言模型，利用从向量数据库检索的上下文验证LLMs生成的回答。通过将回答拆分成单独的句子，并利用多个模型对给定问题、回答和相关上下文生成“是”标记的概率，可以检测幻觉。该框架通过包含超过100组问题、答案和上下文的实际数据集实验得到了验证，包括完全正确和部分正确的回答。实验结果表明，与检测幻觉相比，该框架在检测正确回答方面提高了10%的F1分数，表明多个小型语言模型可以有效用于答案验证，提供了一个适用于学术和实际应用的可扩展且高效的解决方案。 

---
# Psycholinguistic Word Features: a New Approach for the Evaluation of LLMs Alignment with Humans 

**Title (ZH)**: 心理语言学词特征：一种新的评估大语言模型与人类一致性的方法 

**Authors**: Javier Conde, Miguel González, María Grandury, Gonzalo Martínez, Pedro Reviriego, Mar Brysbaert  

**Link**: [PDF](https://arxiv.org/pdf/2506.22439)  

**Abstract**: The evaluation of LLMs has so far focused primarily on how well they can perform different tasks such as reasoning, question-answering, paraphrasing, or translating. For most of these tasks, performance can be measured with objective metrics, such as the number of correct answers. However, other language features are not easily quantified. For example, arousal, concreteness, or gender associated with a given word, as well as the extent to which we experience words with senses and relate them to a specific sense. Those features have been studied for many years by psycholinguistics, conducting large-scale experiments with humans to produce ratings for thousands of words. This opens an opportunity to evaluate how well LLMs align with human ratings on these word features, taking advantage of existing studies that cover many different language features in a large number of words. In this paper, we evaluate the alignment of a representative group of LLMs with human ratings on two psycholinguistic datasets: the Glasgow and Lancaster norms. These datasets cover thirteen features over thousands of words. The results show that alignment is \textcolor{black}{generally} better in the Glasgow norms evaluated (arousal, valence, dominance, concreteness, imageability, familiarity, and gender) than on the Lancaster norms evaluated (introceptive, gustatory, olfactory, haptic, auditory, and visual). This suggests a potential limitation of current LLMs in aligning with human sensory associations for words, which may be due to their lack of embodied cognition present in humans and illustrates the usefulness of evaluating LLMs with psycholinguistic datasets. 

**Abstract (ZH)**: LLMs在心理语言学数据集上的表现评价 

---
