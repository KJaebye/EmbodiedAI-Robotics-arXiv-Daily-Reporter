# Staircase Streaming for Low-Latency Multi-Agent Inference 

**Title (ZH)**: 阶梯式流式处理用于低延迟多agent推理 

**Authors**: Junlin Wang, Jue Wang, Zhen, Ben Athiwaratkun, Bhuwan Dhingra, Ce Zhang, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.05059)  

**Abstract**: Recent advances in large language models (LLMs) opened up new directions for leveraging the collective expertise of multiple LLMs. These methods, such as Mixture-of-Agents, typically employ additional inference steps to generate intermediate outputs, which are then used to produce the final response. While multi-agent inference can enhance response quality, it can significantly increase the time to first token (TTFT), posing a challenge for latency-sensitive applications and hurting user experience. To address this issue, we propose staircase streaming for low-latency multi-agent inference. Instead of waiting for the complete intermediate outputs from previous steps, we begin generating the final response as soon as we receive partial outputs from these steps. Experimental results demonstrate that staircase streaming reduces TTFT by up to 93% while maintaining response quality. 

**Abstract (ZH)**: Recent Advances in Large Language Models: Staircase Streaming for Low-Latency Multi-Agent Inference 

---
# Look-ahead Reasoning with a Learned Model in Imperfect Information Games 

**Title (ZH)**: 带有学习模型的展望推理在不完美信息游戏中 

**Authors**: Ondřej Kubíček, Viliam Lisý  

**Link**: [PDF](https://arxiv.org/pdf/2510.05048)  

**Abstract**: Test-time reasoning significantly enhances pre-trained AI agents' performance. However, it requires an explicit environment model, often unavailable or overly complex in real-world scenarios. While MuZero enables effective model learning for search in perfect information games, extending this paradigm to imperfect information games presents substantial challenges due to more nuanced look-ahead reasoning techniques and large number of states relevant for individual decisions. This paper introduces an algorithm LAMIR that learns an abstracted model of an imperfect information game directly from the agent-environment interaction. During test time, this trained model is used to perform look-ahead reasoning. The learned abstraction limits the size of each subgame to a manageable size, making theoretically principled look-ahead reasoning tractable even in games where previous methods could not scale. We empirically demonstrate that with sufficient capacity, LAMIR learns the exact underlying game structure, and with limited capacity, it still learns a valuable abstraction, which improves game playing performance of the pre-trained agents even in large games. 

**Abstract (ZH)**: Test-time reasoning显著增强预训练AI代理的性能，但需要显式的环境模型，这在现实世界场景中往往不可用或过于复杂。虽然MuZero在完美信息游戏中有效学习模型，将其范式扩展到不完美信息游戏由于更复杂的前瞻推理技术和大量相关状态，面临重大挑战。本文介绍了一种算法LAMIR，可以从代理与环境的交互中直接学习不完美信息游戏的抽象模型。测试时，训练后的模型用于进行前瞻推理。学习到的抽象将每个子游戏的规模限制在可管理的范围内，即使在以前的方法无法扩展的游戏环境中，也使理论上原则性的前瞻推理变得可行。我们通过实验证明，LAMIR在足够的能力下学习到游戏的确切结构，在有限的能力下仍能学习有价值的观点，并提高预训练代理在大型游戏中的表现。 

---
# Think Then Embed: Generative Context Improves Multimodal Embedding 

**Title (ZH)**: 深思而后嵌入：生成性上下文提升多模态嵌入 

**Authors**: Xuanming Cui, Jianpeng Cheng, Hong-you Chen, Satya Narayan Shukla, Abhijeet Awasthi, Xichen Pan, Chaitanya Ahuja, Shlok Kumar Mishra, Qi Guo, Ser-Nam Lim, Aashu Singh, Xiangjun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.05014)  

**Abstract**: There is a growing interest in Universal Multimodal Embeddings (UME), where models are required to generate task-specific representations. While recent studies show that Multimodal Large Language Models (MLLMs) perform well on such tasks, they treat MLLMs solely as encoders, overlooking their generative capacity. However, such an encoding paradigm becomes less effective as instructions become more complex and require compositional reasoning. Inspired by the proven effectiveness of chain-of-thought reasoning, we propose a general Think-Then-Embed (TTE) framework for UME, composed of a reasoner and an embedder. The reasoner MLLM first generates reasoning traces that explain complex queries, followed by an embedder that produces representations conditioned on both the original query and the intermediate reasoning. This explicit reasoning step enables more nuanced understanding of complex multimodal instructions. Our contributions are threefold. First, by leveraging a powerful MLLM reasoner, we achieve state-of-the-art performance on the MMEB-V2 benchmark, surpassing proprietary models trained on massive in-house datasets. Second, to reduce the dependency on large MLLM reasoners, we finetune a smaller MLLM reasoner using high-quality embedding-centric reasoning traces, achieving the best performance among open-source models with a 7% absolute gain over recently proposed models. Third, we investigate strategies for integrating the reasoner and embedder into a unified model for improved efficiency without sacrificing performance. 

**Abstract (ZH)**: Growing Interest in Universal Multimodal Embeddings with a Think-Then-Embed Framework 

---
# LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game 

**Title (ZH)**: LLM-Hanabi：在不完美信息协作游戏中评估具有心智理论和推理推理的游戏玩法 

**Authors**: Fangzhou Liang, Tianshi Zheng, Chunkit Chan, Yauwai Yim, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.04980)  

**Abstract**: Effective multi-agent collaboration requires agents to infer the rationale behind others' actions, a capability rooted in Theory-of-Mind (ToM). While recent Large Language Models (LLMs) excel at logical inference, their ability to infer rationale in dynamic, collaborative settings remains under-explored. This study introduces LLM-Hanabi, a novel benchmark that uses the cooperative game Hanabi to evaluate the rationale inference and ToM of LLMs. Our framework features an automated evaluation system that measures both game performance and ToM proficiency. Across a range of models, we find a significant positive correlation between ToM and in-game success. Notably, first-order ToM (interpreting others' intent) correlates more strongly with performance than second-order ToM (predicting others' interpretations). These findings highlight that for effective AI collaboration, the ability to accurately interpret a partner's rationale is more critical than higher-order reasoning. We conclude that prioritizing first-order ToM is a promising direction for enhancing the collaborative capabilities of future models. 

**Abstract (ZH)**: 有效的多智能体协作需要智能体推断他人的行为背后的原因，这一能力源自理论心智（ToM）。尽管最新的大规模语言模型（LLMs）在逻辑推理方面表现出色，但在动态协作环境中的推断能力仍待探索。本研究引入了LLM-Hanabi，一种使用合作游戏Hanabi来评估LLMs的推断能力和ToM的新基准。我们的框架包括一个自动评估系统，可以衡量游戏表现和ToM熟练度。在多种模型中，我们发现ToM与游戏中的成功之间存在显著的正相关关系。值得注意的是，一级ToM（解释他人的意图）与表现的相关性比二级ToM（预测他人的解释）更强。这些发现表明，对于有效的AI协作而言，准确解释合作伙伴的推理能力比高级推理更为关键。我们得出结论，优先考虑一级ToM是增强未来模型协作能力的一个有前景的方向。 

---
# Aligning Perception, Reasoning, Modeling and Interaction: A Survey on Physical AI 

**Title (ZH)**: 感知、推理、建模与交互一致性的研究：物理AI综述 

**Authors**: Kun Xiang, Terry Jingchen Zhang, Yinya Huang, Jixi He, Zirong Liu, Yueling Tang, Ruizhe Zhou, Lijing Luo, Youpeng Wen, Xiuwei Chen, Bingqian Lin, Jianhua Han, Hang Xu, Hanhui Li, Bin Dong, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04978)  

**Abstract**: The rapid advancement of embodied intelligence and world models has intensified efforts to integrate physical laws into AI systems, yet physical perception and symbolic physics reasoning have developed along separate trajectories without a unified bridging framework. This work provides a comprehensive overview of physical AI, establishing clear distinctions between theoretical physics reasoning and applied physical understanding while systematically examining how physics-grounded methods enhance AI's real-world comprehension across structured symbolic reasoning, embodied systems, and generative models. Through rigorous analysis of recent advances, we advocate for intelligent systems that ground learning in both physical principles and embodied reasoning processes, transcending pattern recognition toward genuine understanding of physical laws. Our synthesis envisions next-generation world models capable of explaining physical phenomena and predicting future states, advancing safe, generalizable, and interpretable AI systems. We maintain a continuously updated resource at this https URL. 

**Abstract (ZH)**: 快速发展的具身智能和世界模型推动了将物理法则整合到AI系统中的努力，然而物理感知和符号物理学推理分别独立发展，缺乏一个统一的桥梁框架。本文提供了物理AI的全面概述，明确区分了理论物理推理与应用物理理解，并系统考察了基于物理的方法如何增强AI对结构化符号推理、具身系统和生成模型的现实世界理解。通过严谨分析最近的进展，我们提倡将在物理原则和具身推理过程中扎根的学习，超越模式识别，向着对物理法则真正理解的转变。我们的综合展望了新的世界模型，能够解释物理现象并预测未来状态，推动安全、通用和可解释的AI系统的发展。我们维护了一个持续更新的资源，网址为这个 https URL。 

---
# Safe and Compliant Cross-Market Trade Execution via Constrained RL and Zero-Knowledge Audits 

**Title (ZH)**: 通过受限RL和零知识审计实现安全合规的跨市场交易执行 

**Authors**: Ailiya Borjigin, Cong He  

**Link**: [PDF](https://arxiv.org/pdf/2510.04952)  

**Abstract**: We present a cross-market algorithmic trading system that balances execution quality with rigorous compliance enforcement. The architecture comprises a high-level planner, a reinforcement learning execution agent, and an independent compliance agent. We formulate trade execution as a constrained Markov decision process with hard constraints on participation limits, price bands, and self-trading avoidance. The execution agent is trained with proximal policy optimization, while a runtime action-shield projects any unsafe action into a feasible set. To support auditability without exposing proprietary signals, we add a zero-knowledge compliance audit layer that produces cryptographic proofs that all actions satisfied the constraints. We evaluate in a multi-venue, ABIDES-based simulator and compare against standard baselines (e.g., TWAP, VWAP). The learned policy reduces implementation shortfall and variance while exhibiting no observed constraint violations across stress scenarios including elevated latency, partial fills, compliance module toggling, and varying constraint limits. We report effects at the 95% confidence level using paired t-tests and examine tail risk via CVaR. We situate the work at the intersection of optimal execution, safe reinforcement learning, regulatory technology, and verifiable AI, and discuss ethical considerations, limitations (e.g., modeling assumptions and computational overhead), and paths to real-world deployment. 

**Abstract (ZH)**: 一种平衡执行质量与严格合规 enforcement 的跨市场算法交易系统 

---
# MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning 

**Title (ZH)**: MARS：通过多智能体强化学习优化双系统深入研究 

**Authors**: Guoxin Chen, Zile Qiao, Wenqing Wang, Donglei Yu, Xuanzhong Chen, Hao Sun, Minpeng Liao, Kai Fan, Yong Jiang, Penguin Xie, Wayne Xin Zhao, Ruihua Song, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04935)  

**Abstract**: Large Reasoning Models (LRMs) often exhibit a tendency for overanalysis in simple tasks, where the models excessively utilize System 2-type, deliberate reasoning, leading to inefficient token generation. Furthermore, these models face challenges in adapting their reasoning capabilities to rapidly changing environments due to the static nature of their pretraining data. To address these issues, advancing Large Language Models (LLMs) for complex reasoning tasks requires innovative approaches that bridge intuitive and deliberate cognitive processes, akin to human cognition's dual-system dynamic. This paper introduces a Multi-Agent System for Deep ReSearch (MARS) enabling seamless integration of System 1's fast, intuitive thinking with System 2's deliberate reasoning within LLMs. MARS strategically integrates multiple external tools, such as Google Search, Google Scholar, and Python Interpreter, to access up-to-date information and execute complex computations, while creating a specialized division of labor where System 1 efficiently processes and summarizes high-volume external information, providing distilled insights that expand System 2's reasoning context without overwhelming its capacity. Furthermore, we propose a multi-agent reinforcement learning framework extending Group Relative Policy Optimization to simultaneously optimize both systems with multi-turn tool interactions, bin-packing optimization, and sample balancing strategies that enhance collaborative efficiency. Extensive experiments demonstrate MARS achieves substantial improvements of 3.86% on the challenging Humanity's Last Exam (HLE) benchmark and an average gain of 8.9% across 7 knowledge-intensive tasks, validating the effectiveness of our dual-system paradigm for complex reasoning in dynamic information environments. 

**Abstract (ZH)**: 大型推理模型（LRMs）在简单任务中往往会表现出过度分析的倾向，过度利用耗时的刻意推理，导致低效的_token_生成。此外，由于预训练数据的静态性质，这些模型在适应快速变化的环境时也面临挑战。为解决这些问题，为了使大规模语言模型（LLMs）能够应对复杂推理任务，需要创新的方法来融合直观和刻意的认知过程，类似于人类认知系统的动态双重机制。本文介绍了一种多代理系统（MARS）用于深度研究，使LLMs能够无缝整合系统1的快速直观思考与系统2的刻意推理。MARS战略性地整合了多个外部工具，如Google搜索、Google学术和Python解释器，以访问最新信息并执行复杂计算，同时创建了一种专门的分工模式，使系统1高效处理和总结高volume的外部信息，提供提炼后的见解，扩展系统2的推理上下文范围而不使其过载。此外，我们提出了一种多代理强化学习框架，扩展了组相对策略优化方法，以同时优化两个系统，通过多轮工具交互、装箱优化和样本平衡策略来提高协作效率。广泛的实验表明，MARS在具有挑战性的“人类最后一考”（HLE）基准测试中实现了3.86%的显著改进，并在7项知识密集型任务中平均提高了8.9%，验证了我们双重系统范式在动态信息环境中的有效性。 

---
# Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding 

**Title (ZH)**: 人类行为地图：统一心理与社会行为理解的基准测试 

**Authors**: Keane Ong, Wei Dai, Carol Li, Dewei Feng, Hengzhi Li, Jingyao Wu, Jiaee Cheong, Rui Mao, Gianmarco Mengaldo, Erik Cambria, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04899)  

**Abstract**: Using intelligent systems to perceive psychological and social behaviors, that is, the underlying affective, cognitive, and pathological states that are manifested through observable behaviors and social interactions, remains a challenge due to their complex, multifaceted, and personalized nature. Existing work tackling these dimensions through specialized datasets and single-task systems often miss opportunities for scalability, cross-task transfer, and broader generalization. To address this gap, we curate Human Behavior Atlas, a unified benchmark of diverse behavioral tasks designed to support the development of unified models for understanding psychological and social behaviors. Human Behavior Atlas comprises over 100,000 samples spanning text, audio, and visual modalities, covering tasks on affective states, cognitive states, pathologies, and social processes. Our unification efforts can reduce redundancy and cost, enable training to scale efficiently across tasks, and enhance generalization of behavioral features across domains. On Human Behavior Atlas, we train three models: OmniSapiens-7B SFT, OmniSapiens-7B BAM, and OmniSapiens-7B RL. We show that training on Human Behavior Atlas enables models to consistently outperform existing multimodal LLMs across diverse behavioral tasks. Pretraining on Human Behavior Atlas also improves transfer to novel behavioral datasets; with the targeted use of behavioral descriptors yielding meaningful performance gains. 

**Abstract (ZH)**: 使用智能系统感知心理和社会行为：人类行为图谱的构建与应用 

---
# Where Did It All Go Wrong? A Hierarchical Look into Multi-Agent Error Attribution 

**Title (ZH)**: 一切都错在哪里？基于层次结构的多agent错误归因探究 

**Authors**: Adi Banerjee, Anirudh Nair, Tarik Borogovac  

**Link**: [PDF](https://arxiv.org/pdf/2510.04886)  

**Abstract**: Error attribution in Large Language Model (LLM) multi-agent systems presents a significant challenge in debugging and improving collaborative AI systems. Current approaches to pinpointing agent and step level failures in interaction traces - whether using all-at-once evaluation, step-by-step analysis, or binary search - fall short when analyzing complex patterns, struggling with both accuracy and consistency. We present ECHO (Error attribution through Contextual Hierarchy and Objective consensus analysis), a novel algorithm that combines hierarchical context representation, objective analysis-based evaluation, and consensus voting to improve error attribution accuracy. Our approach leverages a positional-based leveling of contextual understanding while maintaining objective evaluation criteria, ultimately reaching conclusions through a consensus mechanism. Experimental results demonstrate that ECHO outperforms existing methods across various multi-agent interaction scenarios, showing particular strength in cases involving subtle reasoning errors and complex interdependencies. Our findings suggest that leveraging these concepts of structured, hierarchical context representation combined with consensus-based objective decision-making, provides a more robust framework for error attribution in multi-agent systems. 

**Abstract (ZH)**: Large Language Model (LLM) 多智能体系统中的错误归因面临在调试和改进协作AI系统中的一大挑战。当前在交互轨迹中定位智能体和步骤级故障的方法，无论是整体评估、逐步分析还是二分查找，当分析复杂模式时都存在准确性与一致性不足的问题。我们提出了一种新型算法 ECHO（通过上下文层次结构和目标共识分析的错误归因），该算法结合了层次上下文表示、基于目标分析的评估以及共识投票，以提高错误归因准确性。我们的方法利用基于位置的上下文分级理解方式，同时保持客观评价标准，最终通过共识机制得出结论。实验结果表明，ECHO 在各种多智能体交互场景中优于现有方法，特别是在涉及微妙推理错误和复杂相互依赖的情况中表现出更强的优势。我们的研究结果表明，利用结构化的层次上下文表示概念以及基于共识的目标决策机制，为多智能体系统中的错误归因提供了一个更为稳健的框架。 

---
# Video Game Level Design as a Multi-Agent Reinforcement Learning Problem 

**Title (ZH)**: 视频游戏关卡设计作为一种多智能体强化学习问题 

**Authors**: Sam Earle, Zehua Jiang, Eugene Vinitsky, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2510.04862)  

**Abstract**: Procedural Content Generation via Reinforcement Learning (PCGRL) offers a method for training controllable level designer agents without the need for human datasets, using metrics that serve as proxies for level quality as rewards. Existing PCGRL research focuses on single generator agents, but are bottlenecked by the need to frequently recalculate heuristics of level quality and the agent's need to navigate around potentially large maps. By framing level generation as a multi-agent problem, we mitigate the efficiency bottleneck of single-agent PCGRL by reducing the number of reward calculations relative to the number of agent actions. We also find that multi-agent level generators are better able to generalize to out-of-distribution map shapes, which we argue is due to the generators' learning more local, modular design policies. We conclude that treating content generation as a distributed, multi-agent task is beneficial for generating functional artifacts at scale. 

**Abstract (ZH)**: 基于强化学习的程序化内容生成（PCGRL）提供了一种在无需人类数据集的情况下训练可控关卡设计代理的方法，使用作为关卡质量代理的指标作为奖励。现有PCGRL研究集中在单个生成器代理上，但受限于频繁重新计算关卡质量的启发式以及代理需要在可能非常大的地图中导航的需求。通过将关卡生成问题框架化为多代理问题，我们通过减少奖励计算次数相对于代理动作次数的比例，缓解了单代理PCGRL的效率瓶颈。我们还发现，多代理关卡生成器能够更好地泛化到分布外的地图形状，我们认为这是由于生成器学习到了更多局部的、模块化的设计策略。我们得出结论，将内容生成视为分布式的多代理任务有助于大规模生成功能性成果。 

---
# LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation 

**Title (ZH)**: LEGOMem：模块化过程记忆多agent大语言模型系统的工作流程自动化 

**Authors**: Dongge Han, Camille Couturier, Daniel Madrigal Diaz, Xuchao Zhang, Victor Rühle, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04851)  

**Abstract**: We introduce LEGOMem, a modular procedural memory framework for multi-agent large language model (LLM) systems in workflow automation. LEGOMem decomposes past task trajectories into reusable memory units and flexibly allocates them across orchestrators and task agents to support planning and execution. To explore the design space of memory in multi-agent systems, we use LEGOMem as a lens and conduct a systematic study of procedural memory in multi-agent systems, examining where memory should be placed, how it should be retrieved, and which agents benefit most. Experiments on the OfficeBench benchmark show that orchestrator memory is critical for effective task decomposition and delegation, while fine-grained agent memory improves execution accuracy. We find that even teams composed of smaller language models can benefit substantially from procedural memory, narrowing the performance gap with stronger agents by leveraging prior execution traces for more accurate planning and tool use. These results position LEGOMem as both a practical framework for memory-augmented agent systems and a research tool for understanding memory design in multi-agent workflow automation. 

**Abstract (ZH)**: 我们介绍了一种用于工作流自动化多智能体大型语言模型系统的模块化过程记忆框架LEGOMem。通过将过去任务轨迹分解为可重用的记忆单元，并灵活分配给协调器和任务代理，LEGOMem支持规划和执行。为了探讨多智能体系统中记忆的设计空间，我们以LEGOMem为视角，系统研究了多智能体系统中的过程记忆，探讨了记忆应放置的位置、如何检索以及哪些智能体受益最大。OfficeBench基准测试结果表明，协调器记忆对于有效的任务分解和分配至关重要，而精细粒度的智能体记忆可以提高执行准确性。研究发现，即使由较小的语言模型组成的团队也可以显著受益于过程记忆，通过利用先前的执行踪迹进行更准确的规划和工具使用，缩小了与更强智能体之间的性能差距。这些结果使LEGOMem既成为增强型智能体系统的一种实用框架，也成为理解多智能体工作流自动化中记忆设计的研究工具。 

---
# Natural Language Edge Labelling: Decoupling Intent from Execution in Structured LM Reasoning 

**Title (ZH)**: 自然语言边缘标签化：在结构化LM推理中解耦意图与执行 

**Authors**: Abhinav Madahar  

**Link**: [PDF](https://arxiv.org/pdf/2510.04817)  

**Abstract**: Controllers for structured LM reasoning (e.g., Chain-of-Thought, self-consistency, and Tree-of-Thoughts) often entangle what to try next with how to execute it, exposing only coarse global knobs and yielding brittle, compute-inefficient, and hard-to-audit behavior. We introduce Natural Language Edge Labelling (NLEL), a labeller-tuner overlay that attaches a free-form natural-language directive to each search edge and translates it into a schema-bounded control vector for decoding, search (branch quotas, exploration $\beta$), generation bundle size, retrieval mixtures, and verification passes. A labeller $\Lambda$ emits labels from the parent state and a compact context; a tuner $\Psi$ maps $(P, L, C)\to \Pi$, with strict schema validation and trust-region projection around safe defaults. Downstream selection remains ToT-style with score $S=\mu+\beta\sigma$ and depth-annealed $\beta$. We show NLEL strictly generalizes CoT/ToT, prove an anytime-monotonicity property for top-$k$ selection under label-conditioned bundles, and bound selector shortfall by control-vector distortion, providing decision-relevant justification for guards like trust regions and verification passes. We instantiate $\Psi$ as a prompt-only JSON Parameter Emitter and preregister an evaluation on GSM8K, MATH (subset), StrategyQA, and ARC-Challenge with compute-aware reporting (success@compute, tokens-per-success) and ablations over $\Lambda$, $\Psi$, trust-region radius, and control quantization; preregistered forecasts anticipate accuracy gains at comparable token budgets and improved success@compute under constraints. NLEL offers an interpretable, model-agnostic interface that separates intent from execution for controllable, auditable LM inference. 

**Abstract (ZH)**: 结构化LM推理的控制器（例如Chain-of-Thought、自一致性、Tree-of-Thoughts）常常将下一步尝试与执行方式纠缠在一起，仅暴露粗粒度的全局开关，导致脆弱、计算效率低下且难以审计的行为。我们引入自然语言边标注（NLEL），这是一种标签器-调整器叠加层，为每条搜索边附加一个自由格式的自然语言指令，并将其翻译成解码、搜索（分支限制、探索β）、生成束大小、检索混合和验证遍历的模式约束控制向量。标签器Λ从父状态和紧凑上下文生成标签；调整器Ψ将（P, L, C）映射到Π，并具有严格的模式验证和围绕安全默认的可信区域投影。下游选择保持ToT风格，得分为S=μ+βσ，并且深度退火β。我们证明NLEL严格推广CoT/ToT，证明了在标签条件束下的Top-k选择的任意时刻单调性性质，并通过控制向量失真界定了选择器的缺陷，提供了诸如可信区域和验证遍历之类的防护措施的决策相关解释。我们实例化Ψ为仅提示的JSON参数发射器，并在GSM8K、MATH（子集）、StrategyQA和ARC-Challenge上进行预注册评估，包括有计算意识的报告（成功@计算、每成功词元数）以及Λ、Ψ、可信区域半径和控制量化的大规模删减；预注册预测预计将获得在相似词元预算下的准确性提升，并且在约束条件下提高成功@计算。NLEL提供了一种解释性、模型无关的接口，将意图与执行分离，以实现可控且可审计的LM推理。 

---
# Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems 

**Title (ZH)**: 混合平衡GFlowNet解决车辆 routing 问题 

**Authors**: Ni Zhang, Zhiguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04792)  

**Abstract**: Existing GFlowNet-based methods for vehicle routing problems (VRPs) typically employ Trajectory Balance (TB) to achieve global optimization but often neglect important aspects of local optimization. While Detailed Balance (DB) addresses local optimization more effectively, it alone falls short in solving VRPs, which inherently require holistic trajectory optimization. To address these limitations, we introduce the Hybrid-Balance GFlowNet (HBG) framework, which uniquely integrates TB and DB in a principled and adaptive manner by aligning their intrinsically complementary strengths. Additionally, we propose a specialized inference strategy for depot-centric scenarios like the Capacitated Vehicle Routing Problem (CVRP), leveraging the depot node's greater flexibility in selecting successors. Despite this specialization, HBG maintains broad applicability, extending effectively to problems without explicit depots, such as the Traveling Salesman Problem (TSP). We evaluate HBG by integrating it into two established GFlowNet-based solvers, i.e., AGFN and GFACS, and demonstrate consistent and significant improvements across both CVRP and TSP, underscoring the enhanced solution quality and generalization afforded by our approach. 

**Abstract (ZH)**: 基于GFlowNet的方法在车辆路线问题（VRPs）中的现有研究通常使用轨迹平衡（TB）以实现全局优化，但往往会忽视局部优化的重要方面。虽然详细平衡（DB）更有效地处理局部优化，但它单独解决VRPs时仍存在不足，因为VRPs本质上要求全面的轨迹优化。为了解决这些局限性，我们提出了混合平衡GFlowNet（HBG）框架，该框架以原则性和自适应的方式独特地结合了TB和DB的固有互补优势。此外，我们还提出了一种专门的推理策略，用于以配送中心为中心的情景，如 capacitated vehicle routing problem (CVRP)，利用配送中心节点在选择后继者方面的更大灵活性。尽管有所专化，HBG仍然保持广泛的适用性，有效地扩展到诸如旅行商问题（TSP）等没有明确配送中心的问题。我们通过将HBG集成到两个现有的GFlowNet基于的求解器（AGFN和GFACS）中来评估HBG，并在CVRP和TSP上展示了其一致且显著的改进，突显了我们方法提供的增强解决方案质量和泛化能力。 

---
# LMM-Incentive: Large Multimodal Model-based Incentive Design for User-Generated Content in Web 3.0 

**Title (ZH)**: LMM-Incentive：基于大规模多模态模型的用户生成内容激励设计for Web 3.0 

**Authors**: Jinbo Wen, Jiawen Kang, Linfeng Zhang, Xiaoying Tang, Jianhang Tang, Yang Zhang, Zhaohui Yang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2510.04765)  

**Abstract**: Web 3.0 represents the next generation of the Internet, which is widely recognized as a decentralized ecosystem that focuses on value expression and data ownership. By leveraging blockchain and artificial intelligence technologies, Web 3.0 offers unprecedented opportunities for users to create, own, and monetize their content, thereby enabling User-Generated Content (UGC) to an entirely new level. However, some self-interested users may exploit the limitations of content curation mechanisms and generate low-quality content with less effort, obtaining platform rewards under information asymmetry. Such behavior can undermine Web 3.0 performance. To this end, we propose \textit{LMM-Incentive}, a novel Large Multimodal Model (LMM)-based incentive mechanism for UGC in Web 3.0. Specifically, we propose an LMM-based contract-theoretic model to motivate users to generate high-quality UGC, thereby mitigating the adverse selection problem from information asymmetry. To alleviate potential moral hazards after contract selection, we leverage LMM agents to evaluate UGC quality, which is the primary component of the contract, utilizing prompt engineering techniques to improve the evaluation performance of LMM agents. Recognizing that traditional contract design methods cannot effectively adapt to the dynamic environment of Web 3.0, we develop an improved Mixture of Experts (MoE)-based Proximal Policy Optimization (PPO) algorithm for optimal contract design. Simulation results demonstrate the superiority of the proposed MoE-based PPO algorithm over representative benchmarks in the context of contract design. Finally, we deploy the designed contract within an Ethereum smart contract framework, further validating the effectiveness of the proposed scheme. 

**Abstract (ZH)**: Web 3.0代表了下一代互联网，被广泛认为是一个去中心化的生态系统，注重价值表达和数据所有权。通过利用区块链和人工智能技术，Web 3.0为用户提供了前所未有的机会，使其能够创建、拥有和商业化其内容，从而将用户生成内容（UGC）提升到一个新的水平。然而，一些自私的用户可能会利用内容策展机制的限制，生成低质量内容并获得平台奖励，而这些行为在信息不对称的情况下是可行的。这种行为可以削弱Web 3.0性能。为此，我们提出了一种新的基于大型多模态模型（LMM）的激励机制`\textit{LMM-Incentive}`，专用于Web 3.0中的UGC。具体而言，我们提出了一种基于LMM的契约理论模型，以激励用户生成高质量的UGC，从而减轻信息不对称带来的逆向选择问题。为了缓解合同选择后的潜在道德风险，我们利用LMM代理评估UGC的质量，这是合同的主要组成部分，并采用提示工程技术提高LMM代理的评估性能。鉴于传统契约设计方法无法有效适应Web 3.0的动态环境，我们开发了一种改进的专家混合（MoE）基于近端策略优化（PPO）算法进行最优契约设计。模拟结果表明，在契约设计的背景下，提出的MoE基于PPO算法优于代表性基准。最后，我们在以太坊智能合约框架内部署了设计的契约，进一步验证了所提出方案的有效性。 

---
# BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs 

**Title (ZH)**: TheBrokenMath：LLM在定理证明中阿谀奉承现象的基准测试 

**Authors**: Ivo Petrov, Jasper Dekoninck, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2510.04721)  

**Abstract**: Large language models (LLMs) have recently shown strong performance on mathematical benchmarks. At the same time, they are prone to hallucination and sycophancy, often providing convincing but flawed proofs for incorrect mathematical statements provided by users. This significantly limits the applicability of LLMs in theorem proving, as verification of these flawed proofs must be done manually by expert mathematicians. However, existing benchmarks that measure sycophancy in mathematics are limited: they focus solely on final-answer problems, rely on very simple and often contaminated datasets, and construct benchmark samples using synthetic modifications that create ill-posed questions rather than well-posed questions that are demonstrably false. To address these issues, we introduce BrokenMath, the first benchmark for evaluating sycophantic behavior in LLMs within the context of natural language theorem proving. BrokenMath is built from advanced 2025 competition problems, which are perturbed with an LLM to produce false statements and subsequently refined through expert review. Using an LLM-as-a-judge framework, we evaluate state-of-the-art LLMs and agentic systems and find that sycophancy is widespread, with the best model, GPT-5, producing sycophantic answers 29% of the time. We further investigate several mitigation strategies, including test-time interventions and supervised fine-tuning on curated sycophantic examples. These approaches substantially reduce, but do not eliminate, sycophantic behavior. 

**Abstract (ZH)**: 大型语言模型在数学基准测试中表现出色，但容易产生幻觉和奉承行为，常常为用户提供错误的数学陈述提供看似正确的但实际上是错误的证明。这显著限制了大型语言模型在定理证明中的应用，因为这些错误证明的验证必须由专家数学家手动完成。然而，现有的测验数学奉承性的基准测试有限：它们仅关注最终答案问题，依赖于非常简单且常常被污染的数据集，并通过合成修改构建基准样本，创造出不良形成的问题而非可以被证明为错误的恰当形成的问题。为解决这些问题，我们引入了BrokenMath，这是首个在自然语言定理证明中评估大型语言模型奉承行为的基准测试。BrokenMath 基于2025年高级竞赛问题构建，并通过LLM进行扰动生成错误陈述，随后通过专家审核进行精炼。通过LLM作为裁判的框架，我们评估了最先进的大型语言模型和自立系统，发现奉承行为普遍存在，最佳模型GPT-5有29%的时间产生奉承性答案。我们进一步研究了几种缓解策略，包括测试时干预和在精选奉承性示例上进行监督微调。这些方法显著减少了，但并未完全消除奉承性行为。 

---
# Beyond Outcome Reward: Decoupling Search and Answering Improves LLM Agents 

**Title (ZH)**: 超越结果奖励：分离搜索和回答 improves LLM 代理 

**Authors**: Yiding Wang, Zhepei Wei, Xinyu Zhu, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04695)  

**Abstract**: Enabling large language models (LLMs) to utilize search tools offers a promising path to overcoming fundamental limitations such as knowledge cutoffs and hallucinations. Recent work has explored reinforcement learning (RL) for training search-augmented agents that interleave reasoning and retrieval before answering. These approaches usually rely on outcome-based rewards (e.g., exact match), implicitly assuming that optimizing for final answers will also yield effective intermediate search behaviors. Our analysis challenges this assumption: we uncover multiple systematic deficiencies in search that arise under outcome-only training and ultimately degrade final answer quality, including failure to invoke tools, invalid queries, and redundant searches. To address these shortcomings, we introduce DeSA (Decoupling Search-and-Answering), a simple two-stage training framework that explicitly separates search optimization from answer generation. In Stage 1, agents are trained to improve search effectiveness with retrieval recall-based rewards. In Stage 2, outcome rewards are employed to optimize final answer generation. Across seven QA benchmarks, DeSA-trained agents consistently improve search behaviors, delivering substantially higher search recall and answer accuracy than outcome-only baselines. Notably, DeSA outperforms single-stage training approaches that simultaneously optimize recall and outcome rewards, underscoring the necessity of explicitly decoupling the two objectives. 

**Abstract (ZH)**: 使大型语言模型利用搜索工具为克服知识截止和幻觉等根本限制提供了 promising 的途径。近期工作探索了强化学习 (RL) 用于培训增强搜索代理的方法，这些代理在回答前会交替进行推理和检索。这些方法通常依赖基于结局的奖励（例如，精确匹配），隐含假设优化最终答案也将产生有效的中间搜索行为。我们的分析挑战了这一假设：我们发现仅基于结局训练会导致搜索中出现多个系统性缺陷，最终降低最终答案质量，包括不调用工具、无效查询和冗余搜索。为解决这些不足，我们引入了 DeSA（分隔搜索与回答），这是一种简单的两阶段训练框架，明确将搜索优化与答案生成分离。在第一阶段，代理使用检索召回奖励训练以提高搜索效果。在第二阶段，使用结局奖励优化最终答案生成。在七个问答基准测试中，DeSA训练的代理在各测试项中一致改善了搜索行为，实现显著更高的搜索召回率和答案准确性。值得注意的是，DeSA优于同时优化召回和结局奖励的单阶段训练方法，强调了明确分离两个目标的必要性。 

---
# Watch and Learn: Learning to Use Computers from Online Videos 

**Title (ZH)**: 看并学习：从在线视频中学习使用计算机 

**Authors**: Chan Hee Song, Yiwen Song, Palash Goyal, Yu Su, Oriana Riva, Hamid Palangi, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2510.04673)  

**Abstract**: Computer use agents (CUAs) need to plan task workflows grounded in diverse, ever-changing applications and environments, but learning is hindered by the scarcity of large-scale, high-quality training data in the target application. Existing datasets are domain-specific, static, and costly to annotate, while current synthetic data generation methods often yield simplistic or misaligned task demonstrations. To address these limitations, we introduce Watch & Learn (W&L), a framework that converts human demonstration videos readily available on the Internet into executable UI trajectories at scale. Instead of directly generating trajectories or relying on ad hoc reasoning heuristics, we cast the problem as an inverse dynamics objective: predicting the user's action from consecutive screen states. This formulation reduces manual engineering, is easier to learn, and generalizes more robustly across applications. Concretely, we develop an inverse dynamics labeling pipeline with task-aware video retrieval, generate over 53k high-quality trajectories from raw web videos, and demonstrate that these trajectories improve CUAs both as in-context demonstrations and as supervised training data. On the challenging OSWorld benchmark, UI trajectories extracted with W&L consistently enhance both general-purpose and state-of-the-art frameworks in-context, and deliver stronger gains for open-source models under supervised training. These results highlight web-scale human demonstration videos as a practical and scalable foundation for advancing CUAs towards real-world deployment. 

**Abstract (ZH)**: 基于观察与学习的计算机使用代理框架（Watch & Learn）：大规模生成可执行的用户界面轨迹 

---
# Improving Multimodal Brain Encoding Model with Dynamic Subject-awareness Routing 

**Title (ZH)**: 基于动态主体意识路由的多模态大脑编码模型改进 

**Authors**: Xuanhua Yin, Runkai Zhao, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.04670)  

**Abstract**: Naturalistic fMRI encoding must handle multimodal inputs, shifting fusion styles, and pronounced inter-subject variability. We introduce AFIRE (Agnostic Framework for Multimodal fMRI Response Encoding), an agnostic interface that standardizes time-aligned post-fusion tokens from varied encoders, and MIND, a plug-and-play Mixture-of-Experts decoder with a subject-aware dynamic gating. Trained end-to-end for whole-brain prediction, AFIRE decouples the decoder from upstream fusion, while MIND combines token-dependent Top-K sparse routing with a subject prior to personalize expert usage without sacrificing generality. Experiments across multiple multimodal backbones and subjects show consistent improvements over strong baselines, enhanced cross-subject generalization, and interpretable expert patterns that correlate with content type. The framework offers a simple attachment point for new encoders and datasets, enabling robust, plug-and-improve performance for naturalistic neuroimaging studies. 

**Abstract (ZH)**: 自然场景fMRI编码必须处理多模态输入、变化的融合风格及显著的跨被试变异。我们引入AFIRE（无偏多模态fMRI响应编码框架），这是一种无偏的接口，标准化来自不同编码器的时间对齐后融合标记，并引入MIND（可插拔的专家混合解码器），带有被试感知的动态门控。AFIRE通过端到端训练进行全脑预测，解码器与上游融合分离，而MIND结合标记依赖的Top-K稀疏路由与被试先验，个性化专家使用而不牺牲通用性。跨多个多模态基础模型和被试的实验显示优于强劲基线的一致性改进、增强的跨被试泛化能力和可解释的专家模式，这些模式与内容类型相关。该框架提供了为新编码器和数据集添加简单挂接点的途径，以实现自然场景神经成像研究的稳健和插拔优化性能。 

---
# QuantAgents: Towards Multi-agent Financial System via Simulated Trading 

**Title (ZH)**: QuantAgents: 向往通过模拟交易构建多agent金融系统 

**Authors**: Xiangyu Li, Yawen Zeng, Xiaofen Xing, Jin Xu, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04643)  

**Abstract**: In this paper, our objective is to develop a multi-agent financial system that incorporates simulated trading, a technique extensively utilized by financial professionals. While current LLM-based agent models demonstrate competitive performance, they still exhibit significant deviations from real-world fund companies. A critical distinction lies in the agents' reliance on ``post-reflection'', particularly in response to adverse outcomes, but lack a distinctly human capability: long-term prediction of future trends. Therefore, we introduce QuantAgents, a multi-agent system integrating simulated trading, to comprehensively evaluate various investment strategies and market scenarios without assuming actual risks. Specifically, QuantAgents comprises four agents: a simulated trading analyst, a risk control analyst, a market news analyst, and a manager, who collaborate through several meetings. Moreover, our system incentivizes agents to receive feedback on two fronts: performance in real-world markets and predictive accuracy in simulated trading. Extensive experiments demonstrate that our framework excels across all metrics, yielding an overall return of nearly 300% over the three years (this https URL). 

**Abstract (ZH)**: 在本文中，我们的目标是开发一个包含模拟交易的多智能体金融系统，这是一种广泛应用于金融专业人士的技术。虽然当前基于大语言模型的智能体模型表现出色，但在某些方面仍与真实世界的投资管理公司存在显著差异。一个关键区别在于智能体依赖于“反省后”的决策，尤其是在面对不利结果时，但缺乏一种独特的人类能力：对未来趋势的长期预测。因此，我们引入了QuantAgents，这是一种结合模拟交易的多智能体系统，用于全面评估各种投资策略和市场情景，而不假设实际风险。具体而言，QuantAgents 包含四个智能体：模拟交易分析师、风险控制分析师、市场新闻分析师和经理，他们通过多次会议协作。此外，我们的系统激励智能体在两个方面接收反馈：在真实市场中的表现和模拟交易中的预测准确性。广泛的实验表明，我们的框架在所有指标上均表现出色，在三年时间里实现了近300%的整体回报率（详见此链接：https://www.crisil.com/research/quantagents-achieves-nearly-300-retur）。 

---
# MedPAO: A Protocol-Driven Agent for Structuring Medical Reports 

**Title (ZH)**: MedPAO：基于协议的医疗报告结构化代理 

**Authors**: Shrish Shrinath Vaidya, Gowthamaan Palani, Sidharth Ramesh, Velmurugan Balasubramanian, Minmini Selvam, Gokulraja Srinivasaraja, Ganapathy Krishnamurthi  

**Link**: [PDF](https://arxiv.org/pdf/2510.04623)  

**Abstract**: The deployment of Large Language Models (LLMs) for structuring clinical data is critically hindered by their tendency to hallucinate facts and their inability to follow domain-specific rules. To address this, we introduce MedPAO, a novel agentic framework that ensures accuracy and verifiable reasoning by grounding its operation in established clinical protocols such as the ABCDEF protocol for CXR analysis. MedPAO decomposes the report structuring task into a transparent process managed by a Plan-Act-Observe (PAO) loop and specialized tools. This protocol-driven method provides a verifiable alternative to opaque, monolithic models. The efficacy of our approach is demonstrated through rigorous evaluation: MedPAO achieves an F1-score of 0.96 on the critical sub-task of concept categorization. Notably, expert radiologists and clinicians rated the final structured outputs with an average score of 4.52 out of 5, indicating a level of reliability that surpasses baseline approaches relying solely on LLM-based foundation models. The code is available at: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在临床数据结构化部署中的应用受到其事实幻想倾向和难以遵循领域特定规则的限制。为了解决这一问题，我们引入了MedPAO，这是一种新颖的代理人框架，通过将其操作基于如胸部X光分析的ABCDEF协议等现有临床协议，确保准确性和可验证推理。MedPAO将报告结构化任务分解为一个由计划-行动-观察（PAO）循环和专门工具管理的透明过程。基于协议的方法为不透明的大规模模型提供了可验证的替代方案。通过严格的评估展示了我们方法的有效性：MedPAO在概念分类的关键子任务上获得了F1分数0.96。值得注意的是，专家放射科医生和临床医生对最终结构化输出的平均评分为4.52分，这是一个高于仅依赖LLM基础模型的基线方法的可靠性水平。代码可从以下链接获取：this https URL。 

---
# Making Mathematical Reasoning Adaptive 

**Title (ZH)**: 使数学推理具有适应性 

**Authors**: Zhejian Lai, Xiang Geng, Zhijun Wang, Yang Bai, Jiahuan Li, Rongxiang Weng, Jingang Wang, Xuezhi Cao, Xunliang Cai, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04617)  

**Abstract**: Mathematical reasoning is a primary indicator of large language models (LLMs) intelligence. However, existing LLMs exhibit failures of robustness and generalization. This paper attributes these deficiencies to spurious reasoning, i.e., producing answers from superficial features. To address this challenge, we propose the AdaR framework to enable adaptive reasoning, wherein models rely on problem-solving logic to produce answers. AdaR synthesizes logically equivalent queries by varying variable values, and trains models with RLVR on these data to penalize spurious logic while encouraging adaptive logic. To improve data quality, we extract the problem-solving logic from the original query and generate the corresponding answer by code execution, then apply a sanity check. Experimental results demonstrate that AdaR improves robustness and generalization, achieving substantial improvement in mathematical reasoning while maintaining high data efficiency. Analysis indicates that data synthesis and RLVR function in a coordinated manner to enable adaptive reasoning in LLMs. Subsequent analyses derive key design insights into the effect of critical factors and the applicability to instruct LLMs. Our project is available at this https URL 

**Abstract (ZH)**: 数学推理是大型语言模型（LLMs）智能的主要指标。然而，现有LLMs在稳健性和泛化能力上表现出缺陷。本文将这些缺陷归因于虚假推理，即基于表面特征产生答案。为应对这一挑战，我们提出了AdaR框架以实现适应性推理，其中模型依赖于问题解决逻辑来产生答案。AdaR通过改变变量值合成分量等价的查询，并利用RLVR在这些数据上训练模型，以惩罚虚假逻辑并鼓励适应性逻辑。为了提高数据质量，我们从原始查询中提取问题解决逻辑，通过代码执行生成相应的答案，并应用合理性检查。实验结果表明，AdaR提高了稳健性和泛化能力，在数学推理方面取得了显著改进，同时保持了高度的数据效率。分析表明，数据合成和RLVR功能在协调运作，以使LLMs实现适应性推理。后续分析得出了关键设计见解，探讨了关键因素的影响及其对指导LLMs的适用性。我们的项目可在以下链接获取：this https URL。 

---
# Perfect AI Mimicry and the Epistemology of Consciousness: A Solipsistic Dilemma 

**Title (ZH)**: 完美的AI模拟与意识的 epistemology : 一种唯我论困境 

**Authors**: Shurui Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.04588)  

**Abstract**: Rapid advances in artificial intelligence necessitate a re-examination of the epistemological foundations upon which we attribute consciousness. As AI systems increasingly mimic human behavior and interaction with high fidelity, the concept of a "perfect mimic"-an entity empirically indistinguishable from a human through observation and interaction-shifts from hypothetical to technologically plausible. This paper argues that such developments pose a fundamental challenge to the consistency of our mind-recognition practices. Consciousness attributions rely heavily, if not exclusively, on empirical evidence derived from behavior and interaction. If a perfect mimic provides evidence identical to that of humans, any refusal to grant it equivalent epistemic status must invoke inaccessible factors, such as qualia, substrate requirements, or origin. Selectively invoking such factors risks a debilitating dilemma: either we undermine the rational basis for attributing consciousness to others (epistemological solipsism), or we accept inconsistent reasoning. I contend that epistemic consistency demands we ascribe the same status to empirically indistinguishable entities, regardless of metaphysical assumptions. The perfect mimic thus acts as an epistemic mirror, forcing critical reflection on the assumptions underlying intersubjective recognition in light of advancing AI. This analysis carries significant implications for theories of consciousness and ethical frameworks concerning artificial agents. 

**Abstract (ZH)**: 快速发展的人工智能 necessitates a re-examination of the epistemological foundations upon which we attribute consciousness。 

---
# Strongly Solving 2048 4x3 

**Title (ZH)**: 强求解2048游戏的4x3版本 

**Authors**: Tomoyuki Kaneko, Shuhei Yamashita  

**Link**: [PDF](https://arxiv.org/pdf/2510.04580)  

**Abstract**: 2048 is a stochastic single-player game involving 16 cells on a 4 by 4 grid, where a player chooses a direction among up, down, left, and right to obtain a score by merging two tiles with the same number located in neighboring cells along the chosen direction. This paper presents that a variant 2048-4x3 12 cells on a 4 by 3 board, one row smaller than the original, has been strongly solved. In this variant, the expected score achieved by an optimal strategy is about $50724.26$ for the most common initial states: ones with two tiles of number 2. The numbers of reachable states and afterstates are identified to be $1,152,817,492,752$ and $739,648,886,170$, respectively. The key technique is to partition state space by the sum of tile numbers on a board, which we call the age of a state. An age is invariant between a state and its successive afterstate after any valid action and is increased two or four by stochastic response from the environment. Therefore, we can partition state space by ages and enumerate all (after)states of an age depending only on states with the recent ages. Similarly, we can identify (after)state values by going along with ages in decreasing order. 

**Abstract (ZH)**: 2048-4x3：一个具有12个细胞的4x3板的单人博弈的强解 

---
# COSMIR: Chain Orchestrated Structured Memory for Iterative Reasoning over Long Context 

**Title (ZH)**: COSMIR: 链式 orchestrated 结构化记忆机制在长上下文迭代推理中的应用 

**Authors**: Naman Gupta, Shreeyash Gowaikar, Arun Iyer, Kirankumar Shiragur, Ramakrishna B Bairi, Rishikesh Maurya, Ritabrata Maiti, Sankarshan Damle, Shachee Mishra Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.04568)  

**Abstract**: Reasoning over very long inputs remains difficult for large language models (LLMs). Common workarounds either shrink the input via retrieval (risking missed evidence), enlarge the context window (straining selectivity), or stage multiple agents to read in pieces. In staged pipelines (e.g., Chain of Agents, CoA), free-form summaries passed between agents can discard crucial details and amplify early mistakes. We introduce COSMIR (Chain Orchestrated Structured Memory for Iterative Reasoning), a chain-style framework that replaces ad hoc messages with a structured memory. A Planner agent first turns a user query into concrete, checkable sub-questions. worker agents process chunks via a fixed micro-cycle: Extract, Infer, Refine, writing all updates to the shared memory. A Manager agent then Synthesizes the final answer directly from the memory. This preserves step-wise read-then-reason benefits while changing both the communication medium (structured memory) and the worker procedure (fixed micro-cycle), yielding higher faithfulness, better long-range aggregation, and auditability. On long-context QA from the HELMET suite, COSMIR reduces propagation-stage information loss and improves accuracy over a CoA baseline. 

**Abstract (ZH)**: 面向非常长输入的推理依然难以通过大型语言模型（LLMs）实现。常见的解决方法要么通过检索缩小输入规模（可能导致证据遗漏），要么扩大上下文窗口（增加选择性压力），要么分阶段使用多个代理分段读取。在分阶段的管道（例如Chain of Agents，CoA）中，自由形式的摘要可能丢弃关键细节并放大早期错误。我们提出COSMIR（Chain Orchestrated Structured Memory for Iterative Reasoning），这是一种链式框架，用结构化记忆取代了即兴的消息传递。初步规划代理将用户查询转化为具体的、可验证的子问题。工人代理通过固定的微周期处理片段：提取、推断、优化，并将所有更新写入共享记忆。然后，管理代理直接从记忆中综合最终答案。这种方法保留了逐步阅读后再推理的优势，同时也改变了通信媒介（结构化记忆）和工人流程（固定的微周期），从而提高了忠实度、更好的长期聚合能力以及可审计性。在HELMET套件的长时间上下文问答任务中，COSMIR减少了传播阶段的信息丢失，并优于CoA基线，提高了准确性。 

---
# ContextNav: Towards Agentic Multimodal In-Context Learning 

**Title (ZH)**: ContextNav: 朝向能动的多模态在环学习 

**Authors**: Honghao Fu, Yuan Ouyang, Kai-Wei Chang, Yiwei Wang, Zi Huang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.04560)  

**Abstract**: Recent advances demonstrate that multimodal large language models (MLLMs) exhibit strong multimodal in-context learning (ICL) capabilities, enabling them to adapt to novel vision-language tasks from a few contextual examples. However, existing ICL approaches face challenges in reconciling scalability with robustness across diverse tasks and noisy contextual examples: manually selecting examples produces clean contexts but is labor-intensive and task-specific, while similarity-based retrieval improves scalability but could introduce irrelevant or structurally inconsistent samples that degrade ICL performance. To address these limitations, we propose ContextNav, the first agentic framework that integrates the scalability of automated retrieval with the quality and adaptiveness of human-like curation, enabling noise-robust and dynamically optimized contextualization for multimodal ICL. ContextNav unifies context management and noise-robust contextualization within a closed-loop workflow driven by graph-based orchestration. Specifically, it builds a resource-aware multimodal embedding pipeline, maintains a retrievable vector database, and applies agentic retrieval and structural alignment to construct noise-resilient contexts. An Operational Grammar Graph (OGG) further supports adaptive workflow planning and optimization, enabling the agent to refine its operational strategies based on downstream ICL feedback. Experimental results demonstrate that ContextNav achieves state-of-the-art performance across various datasets, underscoring the promise of agentic workflows for advancing scalable and robust contextualization in multimodal ICL. 

**Abstract (ZH)**: 最近的研究表明，多模态大型语言模型（MLLMs）表现出强大的多模态上下文学习（ICL）能力，使它们能够从少量上下文示例中适应新型的视觉-语言任务。然而，现有的ICL方法在跨多样任务和嘈杂上下文示例的情况下平衡可扩展性和稳健性方面面临挑战：人工选择示例可以产生清洁的上下文，但人力密集且任务特定，而基于相似性的检索可以提高可扩展性，但可能会引入无关或结构不一致的样本，从而降低ICL性能。为了解决这些局限性，我们提出了ContextNav，这是首个结合自动检索的可扩展性和人类般策展的高质量与适应性的框架，使多模态ICL能够在嘈杂的环境中实现稳健且动态优化的上下文化。ContextNav在一个基于图的闭环工作流程中统一了上下文管理与稳健的上下文化。具体而言，它构建了一个资源感知的多模态嵌入流水线，维护一个可检索的向量数据库，并应用代理检索和结构对齐来构建抗噪的上下文。进一步地，操作语法图（OGG）支持适应性工作流程规划与优化，使代理能够根据下游ICL反馈优化其操作策略。实验结果表明，ContextNav在多种数据集上实现了最先进的性能，突显了代理式工作流程在推动多模态ICL的可扩展和稳健上下文化方面的潜力。 

---
# TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use 

**Title (ZH)**: TRAJECT-Bench：一个轨迹感知基准，用于评估自主工具使用能力 

**Authors**: Pengfei He, Zhenwei Dai, Bing He, Hui Liu, Xianfeng Tang, Hanqing Lu, Juanhui Li, Jiayuan Ding, Subhabrata Mukherjee, Suhang Wang, Yue Xing, Jiliang Tang, Benoit Dumoulin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04550)  

**Abstract**: Large language model (LLM)-based agents increasingly rely on tool use to complete real-world tasks. While existing works evaluate the LLMs' tool use capability, they largely focus on the final answers yet overlook the detailed tool usage trajectory, i.e., whether tools are selected, parameterized, and ordered correctly. We introduce TRAJECT-Bench, a trajectory-aware benchmark to comprehensively evaluate LLMs' tool use capability through diverse tasks with fine-grained evaluation metrics. TRAJECT-Bench pairs high-fidelity, executable tools across practical domains with tasks grounded in production-style APIs, and synthesizes trajectories that vary in breadth (parallel calls) and depth (interdependent chains). Besides final accuracy, TRAJECT-Bench also reports trajectory-level diagnostics, including tool selection and argument correctness, and dependency/order satisfaction. Analyses reveal failure modes such as similar tool confusion and parameter-blind selection, and scaling behavior with tool diversity and trajectory length where the bottleneck of transiting from short to mid-length trajectories is revealed, offering actionable guidance for LLMs' tool use. 

**Abstract (ZH)**: 基于大型语言模型的代理越来越依赖工具使用来完成现实世界任务。尽管现有工作评估了大型语言模型的工具使用能力，但它们主要关注最终答案，而忽略了详细的工具使用轨迹，即工具是否被正确选择、参数化和排序。我们引入了TRAJECT-Bench，这是一种轨迹感知基准，通过多样化的任务和细粒度的评估指标全面评估大型语言模型的工具使用能力。TRAJECT-Bench将高保真可执行工具与基于生产风格API的任务配对，并综合生成在宽度（并行调用）和深度（相互依赖链路）上有所差异的轨迹。除了最终准确性外，TRAJECT-Bench还报告了轨迹级别的诊断，包括工具选择和参数正确性，以及依赖性/顺序满足情况。分析揭示了如工具混淆和参数盲选等失败模式，以及随工具多样性和轨迹长度的扩展行为，在从短轨迹过渡到中长度轨迹时瓶颈被揭示，为大型语言模型的工具使用提供了可操作的指导。 

---
# Code World Models for General Game Playing 

**Title (ZH)**: 代码世界模型在通用游戏-playing中的应用 

**Authors**: Wolfgang Lehrach, Daniel Hennes, Miguel Lazaro-Gredilla, Xinghua Lou, Carter Wendelken, Zun Li, Antoine Dedieu, Jordi Grau-Moya, Marc Lanctot, Atil Iscen, John Schultz, Marcus Chiam, Ian Gemp, Piotr Zielinski, Satinder Singh, Kevin P. Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2510.04542)  

**Abstract**: Large Language Models (LLMs) reasoning abilities are increasingly being applied to classical board and card games, but the dominant approach -- involving prompting for direct move generation -- has significant drawbacks. It relies on the model's implicit fragile pattern-matching capabilities, leading to frequent illegal moves and strategically shallow play. Here we introduce an alternative approach: We use the LLM to translate natural language rules and game trajectories into a formal, executable world model represented as Python code. This generated model -- comprising functions for state transition, legal move enumeration, and termination checks -- serves as a verifiable simulation engine for high-performance planning algorithms like Monte Carlo tree search (MCTS). In addition, we prompt the LLM to generate heuristic value functions (to make MCTS more efficient), and inference functions (to estimate hidden states in imperfect information games). Our method offers three distinct advantages compared to directly using the LLM as a policy: (1) Verifiability: The generated CWM serves as a formal specification of the game's rules, allowing planners to algorithmically enumerate valid actions and avoid illegal moves, contingent on the correctness of the synthesized model; (2) Strategic Depth: We combine LLM semantic understanding with the deep search power of classical planners; and (3) Generalization: We direct the LLM to focus on the meta-task of data-to-code translation, enabling it to adapt to new games more easily. We evaluate our agent on 10 different games, of which 4 are novel and created for this paper. 5 of the games are fully observed (perfect information), and 5 are partially observed (imperfect information). We find that our method outperforms or matches Gemini 2.5 Pro in 9 out of the 10 considered games. 

**Abstract (ZH)**: 大型语言模型在经典棋盘和纸牌游戏中的推理能力正在被越来越多地应用，但主导的方法——涉及直接提示生成移动——存在显著缺点。这种方法依赖于模型隐含的脆弱的模式匹配能力，导致频繁出现非法移动和战略性浅薄的玩法。在这里我们介绍了一种替代方法：我们使用大型语言模型将自然语言规则和游戏轨迹翻译成形式化的可执行世界模型，表示为Python代码。生成的模型包括状态转换函数、合法移动枚举和终止检查函数，作为高性能计划算法（如蒙特卡洛树搜索MCTS）的验证性模拟引擎。此外，我们提示大型语言模型生成启发式价值函数（使MCTS更高效）和推理函数（估计不完美信息游戏中的隐藏状态）。与直接将大型语言模型用作策略相比，我们的方法具有三种显著优势：（1）可验证性：生成的形式化世界模型作为游戏规则的形式化规范，让规划者能够算法化地枚举有效操作，前提是生成的模型是正确的；（2）战略深度：结合大型语言模型的语义理解能力和经典规划者的深层搜索能力；（3）通用性：我们将大型语言模型引导关注数据到代码的转换这一元任务，使其更容易适应新游戏。我们在10种不同的游戏中评估了我们的代理，其中4种是新型游戏，专门为本文创建。5种游戏为完全观察游戏（完美信息），5种为部分观察游戏（不完美信息）。我们发现，我们的方法在考虑的10种游戏中有9种超过了或匹配了Gemini 2.5 Pro。 

---
# More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models 

**Title (ZH)**: beyond Surface-Level Understanding？揭示视觉-语言驾驶模型中的推理-规划差距 

**Authors**: Xurui Song, Shuo Huai, JingJing Jiang, Jiayi Kong, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04532)  

**Abstract**: Vision-Language Model (VLM) driving agents promise explainable end-to-end autonomy by first producing natural-language reasoning and then predicting trajectory planning. However, whether planning is causally driven by this reasoning remains a critical but unverified assumption. To investigate this, we build DriveMind, a large-scale driving Visual Question Answering (VQA) corpus with plan-aligned Chain-of-Thought (CoT), automatically generated from nuPlan. Our data generation process converts sensors and annotations into structured inputs and, crucially, separates priors from to-be-reasoned signals, enabling clean information ablations. Using DriveMind, we train representative VLM agents with Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) and evaluate them with nuPlan's metrics. Our results, unfortunately, indicate a consistent causal disconnect in reasoning-planning: removing ego/navigation priors causes large drops in planning scores, whereas removing CoT produces only minor changes. Attention analysis further shows that planning primarily focuses on priors rather than the CoT. Based on this evidence, we propose the Reasoning-Planning Decoupling Hypothesis, positing that the training-yielded reasoning is an ancillary byproduct rather than a causal mediator. To enable efficient diagnosis, we also introduce a novel, training-free probe that measures an agent's reliance on priors by evaluating its planning robustness against minor input perturbations. In summary, we provide the community with a new dataset and a diagnostic tool to evaluate the causal fidelity of future models. 

**Abstract (ZH)**: Vision-Language模型驱动代理的推理与规划解耦假设：一个新的大规模驾驶视觉问答数据集及诊断工具探究 

---
# Aria: An Agent For Retrieval and Iterative Auto-Formalization via Dependency Graph 

**Title (ZH)**: Aria: 一种基于依赖图检索和迭代自动形式化的代理 

**Authors**: Hanyu Wang, Ruohan Xie, Yutong Wang, Guoxiong Gao, Xintao Yu, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04520)  

**Abstract**: Accurate auto-formalization of theorem statements is essential for advancing automated discovery and verification of research-level mathematics, yet remains a major bottleneck for LLMs due to hallucinations, semantic mismatches, and their inability to synthesize new definitions. To tackle these issues, we present Aria (Agent for Retrieval and Iterative Autoformalization), a system for conjecture-level formalization in Lean that emulates human expert reasoning via a two-phase Graph-of-Thought process: recursively decomposing statements into a dependency graph and then constructing formalizations from grounded concepts. To ensure semantic correctness, we introduce AriaScorer, a checker that retrieves definitions from Mathlib for term-level grounding, enabling rigorous and reliable verification. We evaluate Aria on diverse benchmarks. On ProofNet, it achieves 91.6% compilation success rate and 68.5% final accuracy, surpassing previous methods. On FATE-X, a suite of challenging algebra problems from research literature, it outperforms the best baseline with 44.0% vs. 24.0% final accuracy. On a dataset of homological conjectures, Aria reaches 42.9% final accuracy while all other models score 0%. 

**Abstract (ZH)**: 准确的自动形式化是推进研究级数学的自动发现与验证的关键，但由于幻觉、语义不匹配以及合成新定义的能力不足，这对大型语言模型仍然是一大瓶颈。为解决这些问题，我们提出了Aria（推理与迭代自动形式化代理），一种通过递归分解语句为依赖图，然后从基础概念构建形式化的Lean系统，以模仿人类专家推理。为确保语义正确性，我们引入了AriaScorer，这是一种检查器，从Mathlib检索定义进行术语级别接地，实现严格的可靠验证。我们在多种基准上评估了Aria。在ProofNet上，它实现了91.6%的编译成功率和68.5%的最终准确率，超过了之前的方法。在FATE-X上，这是一个来自研究文献的一系列具有挑战性的代数问题套件，它以44.0%的最终准确率超过了最佳基线的24.0%。在同调猜想数据集中，Aria 达到了42.9%的最终准确率，而其他所有模型均为0%。 

---
# ChartAgent: A Multimodal Agent for Visually Grounded Reasoning in Complex Chart Question Answering 

**Title (ZH)**: ChartAgent：用于复杂图表问答的多模态 grounding 推理智能体 

**Authors**: Rachneet Kaur, Nishan Srishankar, Zhen Zeng, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2510.04514)  

**Abstract**: Recent multimodal LLMs have shown promise in chart-based visual question answering, but their performance declines sharply on unannotated charts, those requiring precise visual interpretation rather than relying on textual shortcuts. To address this, we introduce ChartAgent, a novel agentic framework that explicitly performs visual reasoning directly within the chart's spatial domain. Unlike textual chain-of-thought reasoning, ChartAgent iteratively decomposes queries into visual subtasks and actively manipulates and interacts with chart images through specialized actions such as drawing annotations, cropping regions (e.g., segmenting pie slices, isolating bars), and localizing axes, using a library of chart-specific vision tools to fulfill each subtask. This iterative reasoning process closely mirrors human cognitive strategies for chart comprehension. ChartAgent achieves state-of-the-art accuracy on the ChartBench and ChartX benchmarks, surpassing prior methods by up to 16.07% absolute gain overall and 17.31% on unannotated, numerically intensive queries. Furthermore, our analyses show that ChartAgent is (a) effective across diverse chart types, (b) achieve the highest scores across varying visual and reasoning complexity levels, and (c) serves as a plug-and-play framework that boosts performance across diverse underlying LLMs. Our work is among the first to demonstrate visually grounded reasoning for chart understanding using tool-augmented multimodal agents. 

**Abstract (ZH)**: 基于图表的视觉推理的新型代理框架：ChartAgent 

---
# Impatient Users Confuse AI Agents: High-fidelity Simulations of Human Traits for Testing Agents 

**Title (ZH)**: 用户缺乏耐心使AI代理困惑：用于测试代理的高保真人类特质模拟 

**Authors**: Muyu He, Anand Kumar, Tsach Mackey, Meghana Rajeev, James Zou, Nazneen Rajani  

**Link**: [PDF](https://arxiv.org/pdf/2510.04491)  

**Abstract**: Despite rapid progress in building conversational AI agents, robustness is still largely untested. Small shifts in user behavior, such as being more impatient, incoherent, or skeptical, can cause sharp drops in agent performance, revealing how brittle current AI agents are. Today's benchmarks fail to capture this fragility: agents may perform well under standard evaluations but degrade spectacularly in more realistic and varied settings. We address this robustness testing gap by introducing TraitBasis, a lightweight, model-agnostic method for systematically stress testing AI agents. TraitBasis learns directions in activation space corresponding to steerable user traits (e.g., impatience or incoherence), which can be controlled, scaled, composed, and applied at inference time without any fine-tuning or extra data. Using TraitBasis, we extend $\tau$-Bench to $\tau$-Trait, where user behaviors are altered via controlled trait vectors. We observe on average a 2%-30% performance degradation on $\tau$-Trait across frontier models, highlighting the lack of robustness of current AI agents to variations in user behavior. Together, these results highlight both the critical role of robustness testing and the promise of TraitBasis as a simple, data-efficient, and compositional tool. By powering simulation-driven stress tests and training loops, TraitBasis opens the door to building AI agents that remain reliable in the unpredictable dynamics of real-world human interactions. We have open-sourced $\tau$-Trai across four domains: airline, retail, telecom, and telehealth, so the community can systematically QA their agents under realistic, behaviorally diverse intents and trait scenarios: this https URL. 

**Abstract (ZH)**: 尽管在构建对话AI代理方面取得了快速进展，但其鲁棒性仍然没有得到充分测试。用户行为的小幅变化，如更加急躁、不连贯或怀疑，都可能导致代理性能急剧下降，揭示当前AI代理的脆弱性。现有的基准未能捕捉到这种脆弱性：代理可能在标准评估中表现良好，但在更现实和多变的环境中表现会大幅下降。为此，我们通过引入TraitBasis，一种轻量级、模型无关的方法，系统地对AI代理进行压力测试来填补这一鲁棒性测试的空白。TraitBasis学习与可调控用户特性（如急躁或不连贯）对应的激活空间方向，这些特性可以在推理时控制、缩放、组合和应用，无需微调或额外数据。使用TraitBasis，我们将$\tau$-Bench扩展为$\tau$-Trait，通过控制特性向量改变用户行为。结果显示，前沿模型在$\tau$-Trait上的性能平均下降2%-30%，突显出当前AI代理在用户行为变化方面的鲁棒性不足。这些结果强调了鲁棒性测试的至关重要性，并展示了TraitBasis作为一种简单、数据高效且可组合工具的潜力。通过驱动模拟驱动的压力测试和训练循环，TraitBasis为构建能够在现实世界人类互动的不可预测动态中保持可靠性的AI代理打开了大门。我们已在四个领域（航空、零售、电信和远程医疗）开源了$\tau$-Trait，社区可以使用它系统地对代理在现实的、行为多样的意图和特性场景下进行QA：this https URL。 

---
# Multi-Agent Collaborative Intelligence: Dual-Dial Control for Reliable LLM Reasoning 

**Title (ZH)**: 多代理协作智能：可靠的LLM推理的双调节控制 

**Authors**: Edward Y. Chang, Ethan Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04488)  

**Abstract**: Multi-agent debate often wastes compute by using a fixed adversarial stance, aggregating without deliberation, or stopping on heuristics. We introduce MACI, an active controller with two independent dials that decouple information from behavior: an information dial that gates evidence by quality, and a behavior dial that schedules contentiousness from exploration to consolidation. A moderator tracks disagreement, overlap, evidence quality, and argument quality, and halts when gains plateau. We provide theory-lite guarantees for nonincreasing dispersion and provable termination, with a budget-feasible scheduler. Across clinical diagnosis and news-bias tasks, MACI improves accuracy and calibration while reducing tokens, and converts residual uncertainty into precision RAG plans that specify what to retrieve next. We use a cross-family LLM judge (CRIT) as a conservative soft weight and stop signal, validated for order invariance and judge-swap stability; stability depends on using high-capability judges. MACI turns debate into a budget-aware, measurable, and provably terminating controller. 

**Abstract (ZH)**: 多智能体辩论经常由于使用固定对抗立场、不进行斟酌的聚合或依赖启发式方法而浪费计算资源。我们引入了MACI，这是一种具有两个独立调节器的主动控制器，将信息与行为解耦：信息调节器根据质量控制证据，行为调节器从探索到巩固安排争议程度。一位调解员跟踪分歧、重叠、证据质量和论点质量，并在增益 plateau 时停止。MACI 提供轻理论保证，即非增加的分散度和可证明的终止性，并配有预算可行的调度器。在临床诊断和新闻偏见任务中，MACI 提高了准确性和校准度，减少了 tokens，还将剩余不确定性转换为精确的 RAG 计划，指明下一步应该检索什么。我们使用跨家族的大型语言模型裁判（CRIT）作为保守的软权重和停止信号，并通过订单不变性和裁判替换稳定性验证；稳定性取决于使用高能力裁判。MACI 将辩论转化为一种具有预算意识、可衡量且可证明终止性的控制器。 

---
# On Continuous Optimization for Constraint Satisfaction Problems 

**Title (ZH)**: 连续优化在约束满足问题中的应用 

**Authors**: Yunuo Cen, Zixuan Wang, Jintao Zhang, Zhiwei Zhang, Xuanyao Fong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04480)  

**Abstract**: Constraint satisfaction problems (CSPs) are fundamental in mathematics, physics, and theoretical computer science. While conflict-driven clause learning Boolean Satisfiability (SAT) solvers have achieved remarkable success and become the mainstream approach for Boolean satisfiability, recent advances show that modern continuous local search (CLS) solvers can achieve highly competitive results on certain classes of SAT problems. Motivated by these advances, we extend the CLS framework from Boolean SAT to general CSP with finite-domain variables and expressive constraints. We present FourierCSP, a continuous optimization framework that generalizes the Walsh-Fourier transform to CSP, allowing for transforming versatile constraints to compact multilinear polynomials, thereby avoiding the need for auxiliary variables and memory-intensive encodings. Our approach leverages efficient evaluation and differentiation of the objective via circuit-output probability and employs a projected gradient optimization method with theoretical guarantees. Empirical results on benchmark suites demonstrate that FourierCSP is scalable and competitive, significantly broadening the class of problems that can be efficiently solved by CLS techniques. 

**Abstract (ZH)**: 约束满足问题（CSPs）在数学、物理学和理论计算机科学中是基础性的。尽管基于冲突驱动的_clause学习（CDCL）的布尔可满足性（SAT）求解器取得了显著成功并已成为布尔可满足性的主流方法，但最近的研究表明，现代连续局部搜索（CLS）求解器在某些类型的SAT问题上可以取得高度竞争的结果。受这些进展的启发，我们将CLS框架从布尔SAT扩展到具有有限域变量和表达性约束的通用CSP。我们提出了FourierCSP，这是一种连续优化框架，将Walsh-傅里叶变换推广到CSP，允许将多样化的约束转换为紧凑的多线性多项式，从而避免使用辅助变量和内存密集型编码。我们的方法利用电路输出概率高效评估和求解目标函数的梯度，并采用具有理论保证的投影梯度优化方法。基准测试集上的实验结果表明，FourierCSP是可扩展且竞争力强的，显著扩展了可以高效解决的CSP问题类别。 

---
# DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization 

**Title (ZH)**: DRPO：通过解耦奖励策略优化进行高效推理 

**Authors**: Gang Li, Yan Chen, Ming Lin, Tianbao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04474)  

**Abstract**: Recent large reasoning models (LRMs) driven by reinforcement learning algorithms (e.g., GRPO) have achieved remarkable performance on challenging reasoning tasks. However, these models suffer from overthinking, generating unnecessarily long and redundant reasoning even for simple questions, which substantially increases computational cost and response latency. While existing methods incorporate length rewards to GRPO to promote concise reasoning, they incur significant performance degradation. We identify the root cause: when rewards for correct but long rollouts are penalized, GRPO's group-relative advantage function can assign them negative advantages, actively discouraging valid reasoning. To overcome this, we propose Decoupled Reward Policy Optimization (DRPO), a novel framework that decouples the length-based learning signal of correct rollouts from incorrect ones. DRPO ensures that reward signals for correct rollouts are normalized solely within the positive group, shielding them from interference by negative samples. The DRPO's objective is grounded in integrating an optimized positive data distribution, which maximizes length-based rewards under a KL regularization, into a discriminative objective. We derive a closed-form solution for this distribution, enabling efficient computation of the objective and its gradients using only on-policy data and importance weighting. Of independent interest, this formulation is general and can incorporate other preference rewards of positive data beyond length. Experiments on mathematical reasoning tasks demonstrate DRPO's significant superiority over six efficient reasoning baselines. Notably, with a 1.5B model, our method achieves 77\% length reduction with only 1.1\% performance loss on simple questions like GSM8k dataset, while the follow-up baseline sacrifices 4.3\% for 68\% length reduction. 

**Abstract (ZH)**: Recent Large Reasoning Models Driven by Reinforcement Learning Algorithms (e.g., GRPO) for Efficient Reasoning Tasks: Decoupled Reward Policy Optimization (DRPO) for Reducing Overthinking 

---
# Utility-Learning Tension in Self-Modifying Agents 

**Title (ZH)**: 自我修改代理的效用学习张力 

**Authors**: Charles L. Wang, Keir Dorchen, Peter Jin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04399)  

**Abstract**: As systems trend toward superintelligence, a natural modeling premise is that agents can self-improve along every facet of their own design. We formalize this with a five-axis decomposition and a decision layer, separating incentives from learning behavior and analyzing axes in isolation. Our central result identifies and introduces a sharp utility--learning tension, the structural conflict in self-modifying systems whereby utility-driven changes that improve immediate or expected performance can also erode the statistical preconditions for reliable learning and generalization. Our findings show that distribution-free guarantees are preserved iff the policy-reachable model family is uniformly capacity-bounded; when capacity can grow without limit, utility-rational self-changes can render learnable tasks unlearnable. Under standard assumptions common in practice, these axes reduce to the same capacity criterion, yielding a single boundary for safe self-modification. Numerical experiments across several axes validate the theory by comparing destructive utility policies against our proposed two-gate policies that preserve learnability. 

**Abstract (ZH)**: 随着系统向超智能发展，一个自然的建模假设是代理可以在设计的各个方面自我改进。我们通过五轴分解和决策层形式化这一假设，将激励与学习行为分离，并分别分析各个维度。我们的主要成果是识别并引入了一种尖锐的效用-学习张力，这是一种自我修改系统中的结构冲突，其中由效用驱动的改进即时或预期性能的变化，也可能侵蚀可靠学习和泛化的统计前提条件。研究发现，在且仅在策略可达到的模型家族具有统一的容量限制时，无分布保证被保留在；当容量可以无限制增长时，效用理性自我变更可以使可学习的任务变得不可学习。在实践中常见的标准假设下，这些维度归结为同一容量标准，提供了一个自我修改的安全边界。通过对多个维度的数值实验，通过将破坏性效用政策与我们提出的双门控政策进行比较，验证了理论，后者保留了可学习性。 

---
# Internal World Models as Imagination Networks in Cognitive Agents 

**Title (ZH)**: 内部世界模型作为认知代理的想象网络 

**Authors**: Saurabh Ranjan, Brian Odegaard  

**Link**: [PDF](https://arxiv.org/pdf/2510.04391)  

**Abstract**: What is the computational objective of imagination? While classical interpretations suggest imagination is useful for maximizing rewards, recent findings challenge this view. In this study, we propose that imagination serves to access an internal world model (IWM) and use psychological network analysis to explore IWMs in humans and large language models (LLMs). Specifically, we assessed imagination vividness ratings using two questionnaires and constructed imagination networks from these reports. Imagination networks from human groups showed correlations between different centrality measures, including expected influence, strength, and closeness. However, imagination networks from LLMs showed a lack of clustering and lower correlations between centrality measures under different prompts and conversational memory conditions. Together, these results indicate a lack of similarity between IWMs in human and LLM agents. Overall, our study offers a novel method for comparing internally-generated representations in humans and AI, providing insights for developing human-like imagination in artificial intelligence. 

**Abstract (ZH)**: 想象的计算目标是什么？经典的解释认为想象有助于最大化奖励，但最近的研究挑战了这一观点。本研究提出，想象服务于访问内部世界模型（IWM）的功能，并利用心理网络分析探索人类和大型语言模型（LLM）的IWM。具体而言，我们使用两个问卷评估想象的生动性，并从这些报告中构建想象网络。人类群体的想象网络在不同的中心性指标之间显示出了相关性，包括预期影响、强度和接近度。然而，在不同的提示和对话记忆条件下，LLM的想象网络缺乏聚类，并且中心性指标之间的相关性较低。这些结果表明，人类和LLM代理的IWM之间缺乏相似性。总的来说，本研究提供了一种新的方法来比较人类和AI内部生成的表征，为发展类似人类的想象提供见解。 

---
# LLM Based Bayesian Optimization for Prompt Search 

**Title (ZH)**: 基于LLM的贝叶斯优化提示搜索 

**Authors**: Adam Ballew, Jingbo Wang, Shaogang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.04384)  

**Abstract**: Bayesian Optimization (BO) has been widely used to efficiently optimize expensive black-box functions with limited evaluations. In this paper, we investigate the use of BO for prompt engineering to enhance text classification with Large Language Models (LLMs). We employ an LLM-powered Gaussian Process (GP) as the surrogate model to estimate the performance of different prompt candidates. These candidates are generated by an LLM through the expansion of a set of seed prompts and are subsequently evaluated using an Upper Confidence Bound (UCB) acquisition function in conjunction with the GP posterior. The optimization process iteratively refines the prompts based on a subset of the data, aiming to improve classification accuracy while reducing the number of API calls by leveraging the prediction uncertainty of the LLM-based GP. The proposed BO-LLM algorithm is evaluated on two datasets, and its advantages are discussed in detail in this paper. 

**Abstract (ZH)**: 贝叶斯优化（BO）已被广泛用于通过有限评估高效优化昂贵的黑盒函数。本文探讨了将BO应用于提示工程，以增强大规模语言模型（LLMs）的文本分类性能。我们采用基于LLM的高斯过程（GP）作为代理模型，用于估计不同提示候选的表现。这些候选提示由LLM通过扩展一组种子提示生成，并使用上置信边界限（UCB）获取函数与GP后验相结合的方法进行评估。优化过程基于数据子集迭代细化提示，旨在通过利用基于LLM的GP的预测不确定性来提高分类准确性并减少API调用次数。本文在两个数据集上评估了所提出的BO-LLM算法，并详细讨论了其优势。 

---
# Just-in-time Episodic Feedback Hinter: Leveraging Offline Knowledge to Improve LLM Agents Adaptation 

**Title (ZH)**: 随时反馈 episodic 回顾助手：利用离线知识提高 LLB 代理的适应性 

**Authors**: Hadi Nekoei, Aman Jaiswal, Patrice Bechard, Oleh Shliazhko, Orlando Marquez Ayala, Mathieu Reymond, Massimo Caccia, Alexandre Drouin, Sarath Chandar, Alexandre Lacoste  

**Link**: [PDF](https://arxiv.org/pdf/2510.04373)  

**Abstract**: Large language model (LLM) agents perform well in sequential decision-making tasks, but improving them on unfamiliar domains often requires costly online interactions or fine-tuning on large expert datasets. These strategies are impractical for closed-source models and expensive for open-source ones, with risks of catastrophic forgetting. Offline trajectories offer reusable knowledge, yet demonstration-based methods struggle because raw traces are long, noisy, and tied to specific tasks. We present Just-in-time Episodic Feedback Hinter (JEF Hinter), an agentic system that distills offline traces into compact, context-aware hints. A zooming mechanism highlights decisive steps in long trajectories, capturing both strategies and pitfalls. Unlike prior methods, JEF Hinter leverages both successful and failed trajectories, extracting guidance even when only failure data is available, while supporting parallelized hint generation and benchmark-independent prompting. At inference, a retriever selects relevant hints for the current state, providing targeted guidance with transparency and traceability. Experiments on MiniWoB++, WorkArena-L1, and WebArena-Lite show that JEF Hinter consistently outperforms strong baselines, including human- and document-based hints. 

**Abstract (ZH)**: 基于紧凑上下文提示的即时 episodic 回馈代理系统 JEF Hinter 

---
# Speculative Actions: A Lossless Framework for Faster Agentic Systems 

**Title (ZH)**: 投机行动：一个无损框架以实现更快的自主系统 

**Authors**: Naimeng Ye, Arnav Ahuja, Georgios Liargkovas, Yunan Lu, Kostis Kaffes, Tianyi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04371)  

**Abstract**: Despite growing interest in AI agents across industry and academia, their execution in an environment is often slow, hampering training, evaluation, and deployment. For example, a game of chess between two state-of-the-art agents may take hours. A critical bottleneck is that agent behavior unfolds sequentially: each action requires an API call, and these calls can be time-consuming. Inspired by speculative execution in microprocessors and speculative decoding in LLM inference, we propose speculative actions, a lossless framework for general agentic systems that predicts likely actions using faster models, enabling multiple steps to be executed in parallel. We evaluate this framework across three agentic environments: gaming, e-commerce, web search, and a "lossy" extension for an operating systems environment. In all cases, speculative actions achieve substantial accuracy in next-action prediction (up to 55%), translating into significant reductions in end-to-end latency. Moreover, performance can be further improved through stronger guessing models, top-K action prediction, multi-step speculation, and uncertainty-aware optimization, opening a promising path toward deploying low-latency agentic systems in the real world. 

**Abstract (ZH)**: 尽管工业和学术界对AI代理的兴趣日益增长，但它们在环境中的执行往往很慢，阻碍了训练、评估和部署。例如，两台顶尖博弈代理进行一局国际象棋比赛可能需要数小时。一个关键瓶颈在于代理行为是按顺序展开的：每行动都需要一个API调用，这些调用可能耗费大量时间。受微处理器的预测执行和大语言模型推理中的预测解码启发，我们提出了一种无损的预测行为框架，该框架利用更快的模型预测可能出现的动作，使多个步骤能够并行执行。我们在游戏、电子商务、网络搜索以及操作系统环境的“失真”扩展三个代理环境中评估了该框架。在所有情况下，预测行为都能在下一动作预测方面取得显著准确性（最高可达55%），从而显著减少端到端的延迟。此外，通过更强的猜测模型、K项动作预测、多步预测和不确定性感知优化，性能可以进一步提升，为在实际中部署低延迟代理系统开辟了前景。 

---
# On the Importance of Task Complexity in Evaluating LLM-Based Multi-Agent Systems 

**Title (ZH)**: 基于LLM的多Agent系统评估中任务复杂性的重要性 

**Authors**: Bohan Tang, Huidong Liang, Keyue Jiang, Xiaowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04311)  

**Abstract**: Large language model multi-agent systems (LLM-MAS) offer a promising paradigm for harnessing collective intelligence to achieve more advanced forms of AI behaviour. While recent studies suggest that LLM-MAS can outperform LLM single-agent systems (LLM-SAS) on certain tasks, the lack of systematic experimental designs limits the strength and generality of these conclusions. We argue that a principled understanding of task complexity, such as the degree of sequential reasoning required and the breadth of capabilities involved, is essential for assessing the effectiveness of LLM-MAS in task solving. To this end, we propose a theoretical framework characterising tasks along two dimensions: depth, representing reasoning length, and width, representing capability diversity. We theoretically examine a representative class of LLM-MAS, namely the multi-agent debate system, and empirically evaluate its performance in both discriminative and generative tasks with varying depth and width. Theoretical and empirical results show that the benefit of LLM-MAS over LLM-SAS increases with both task depth and width, and the effect is more pronounced with respect to depth. This clarifies when LLM-MAS are beneficial and provides a principled foundation for designing future LLM-MAS methods and benchmarks. 

**Abstract (ZH)**: 大规模语言模型多智能体系统（LLM-MAS）为利用集体智能实现更高级的AI行为提供了有前景的范式。尽管近期研究显示LLM-MAS在某些任务上可优于单智能体系统（LLM-SAS），但由于缺乏系统的实验设计，这些结论的强度和普适性受到限制。我们argue认为，系统地理解任务复杂性，比如所需的序列推理程度和涉及的能力多样性，对于评估LLM-MAS在任务解决中的有效性至关重要。为此，我们提出了一种理论框架，沿着两个维度对任务进行表征：深度，代表推理长度；宽度，代表能力多样性。我们理论研究了一类代表性的LLM-MAS，即多智能体辩论系统，并在不同深度和宽度的区分性和生成性任务中评估其性能。理论和实验结果表明，随着任务深度和宽度的增加，LLM-MAS相对于LLM-SAS的优势增大，且在深度方面更为显著。这明确了何时LLM-MAS具有优势，并为设计未来的LLM-MAS方法和基准提供了原则性基础。 

---
# Doctor-R1: Mastering Clinical Inquiry with Experiential Agentic Reinforcement Learning 

**Title (ZH)**: Doctor-R1: 通过经验代理强化学习掌握临床查询能力 

**Authors**: Yunghwei Lai, Kaiming Liu, Ziyue Wang, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04284)  

**Abstract**: The professionalism of a human doctor in outpatient service depends on two core abilities: the ability to make accurate medical decisions and the medical consultation skill to conduct strategic, empathetic patient inquiry. Existing Large Language Models (LLMs) have achieved remarkable accuracy on medical decision-making benchmarks. However, they often lack the ability to conduct the strategic and empathetic consultation, which is essential for real-world clinical scenarios. To address this gap, we propose Doctor-R1, an AI doctor agent trained to master both of the capabilities by ask high-yield questions and conduct strategic multi-turn inquiry to guide decision-making. Our framework introduces three key components: a multi-agent interactive environment, a two-tiered reward architecture that separately optimizes clinical decision-making and communicative inquiry skills, and an experience repository to ground policy learning in high-quality prior trajectories. We evaluate Doctor-R1 on OpenAI's HealthBench and MAQuE, assessed across multi-facet metrics, such as communication quality, user experience, and task accuracy. Remarkably, Doctor-R1 surpasses state-of-the-art open-source specialized LLMs by a substantial margin with higher parameter efficiency and outperforms powerful proprietary models. Furthermore, the human evaluations show a strong preference for Doctor-R1 to generate human-preferred clinical dialogue, demonstrating the effectiveness of the framework. 

**Abstract (ZH)**: 人类门诊医生的专业性取决于两种核心能力：准确的医疗决策能力和战略性、同理心导向的患者咨询技能。现有的大规模语言模型在医疗决策基准测试中已经取得了显著的准确率。然而，它们往往缺乏进行战略性、同理心导向的咨询的能力，这对于实际临床场景是必不可少的。为了解决这一问题，我们提出了一种名为Doctor-R1的AI医生代理，旨在通过提出高收益问题并开展战略性多轮咨询来磨练这两种能力。我们的框架包括三个关键组件：一个多代理交互环境、一个多层奖励架构，分别优化临床决策能力和沟通性咨询技能，并通过高质量的先验轨迹来约束策略学习。我们在OpenAI的HealthBench和MAQuE上评估了Doctor-R1，通过多维度指标评估，如沟通质量、用户体验和任务准确性。令人印象深刻的是，Doctor-R1在参数效率方面超越了现有的开源专业模型，并且在强大的专有模型中表现更佳。此外，人类评估表明，人们更偏好Doctor-R1生成的符合人类偏好的临床对话，这证明了该框架的有效性。 

---
# GROK: From Quantitative Biomarkers to Qualitative Diagnosis via a Grounded MLLM with Knowledge-Guided Instruction 

**Title (ZH)**: GROK: 从定量生物标志物到基于知识引导指令的扎根MLLM的定性诊断 

**Authors**: Zhuangzhi Gao, Hongyi Qin, He Zhao, Qinkai Yu, Feixiang Zhou, Eduard Shantsila, Uazman Alam, Alena Shantsila, Wahbi El-Bouri, Gregory Y. H. Lip, Yalin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04281)  

**Abstract**: Multimodal large language models (MLLMs) hold promise for integrating diverse data modalities, but current medical adaptations such as LLaVA-Med often fail to fully exploit the synergy between color fundus photography (CFP) and optical coherence tomography (OCT), and offer limited interpretability of quantitative biomarkers. We introduce GROK, a grounded multimodal large language model that jointly processes CFP, OCT, and text to deliver clinician-grade diagnoses of ocular and systemic disease. GROK comprises three core modules: Knowledge-Guided Instruction Generation, CLIP-Style OCT-Biomarker Alignment, and Supervised Instruction Fine-Tuning, which together establish a quantitative-to-qualitative diagnostic chain of thought, mirroring real clinical reasoning when producing detailed lesion annotations. To evaluate our approach, we introduce the Grounded Ophthalmic Understanding benchmark, which covers six disease categories and three tasks: macro-level diagnostic classification, report generation quality, and fine-grained clinical assessment of the generated chain of thought. Experiments show that, with only LoRA (Low-Rank Adaptation) fine-tuning of a 7B-parameter Qwen2 backbone, GROK outperforms comparable 7B and 32B baselines on both report quality and fine-grained clinical metrics, and even exceeds OpenAI o3. Code and data are publicly available in the GROK repository. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在集成多样化数据模态方面前景广阔，但现有医疗适应性模型如LLaVA-Med往往难以充分利用色基金底摄影（CFP）和光学相干断层扫描（OCT）之间的协同作用，并且对定量生物标志物的解释能力有限。我们引入了GROK，一种基于现实世界的多模态大型语言模型，该模型联合处理CFP、OCT和文本，以提供眼科和系统性疾病的专业级诊断。GROK包含三个核心模块：知识引导的指令生成、CLIP风格的OCT生物标志物对齐以及监督指令微调，这些模块共同建立了从定量到定性的诊断思维链条，当生成详细病灶注释时，类似于真实的临床推理过程。为了评估我们的方法，我们引入了基于现实理解的眼科学基准，该基准涵盖了六类疾病和三项任务：宏观级别的诊断分类、报告生成质量以及生成思维链条的细粒度临床评估。实验结果显示，在对参数为7B的Qwen2主干进行仅LoRA（低秩适应）微调的情况下，GROK在报告质量和细粒度临床指标上均优于同类7B和32B基准模型，并且甚至超越了OpenAI的o3模型。代码和数据已在GROK仓库中公开。 

---
# Closing the Loop: Coordinating Inventory and Recommendation via Deep Reinforcement Learning on Multiple Timescales 

**Title (ZH)**: 闭环控制：通过多时间尺度深度强化学习协调库存与推荐 

**Authors**: Jinyang Jiang, Jinhui Han, Yijie Peng, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04272)  

**Abstract**: Effective cross-functional coordination is essential for enhancing firm-wide profitability, particularly in the face of growing organizational complexity and scale. Recent advances in artificial intelligence, especially in reinforcement learning (RL), offer promising avenues to address this fundamental challenge. This paper proposes a unified multi-agent RL framework tailored for joint optimization across distinct functional modules, exemplified via coordinating inventory replenishment and personalized product recommendation. We first develop an integrated theoretical model to capture the intricate interplay between these functions and derive analytical benchmarks that characterize optimal coordination. The analysis reveals synchronized adjustment patterns across products and over time, highlighting the importance of coordinated decision-making. Leveraging these insights, we design a novel multi-timescale multi-agent RL architecture that decomposes policy components according to departmental functions and assigns distinct learning speeds based on task complexity and responsiveness. Our model-free multi-agent design improves scalability and deployment flexibility, while multi-timescale updates enhance convergence stability and adaptability across heterogeneous decisions. We further establish the asymptotic convergence of the proposed algorithm. Extensive simulation experiments demonstrate that the proposed approach significantly improves profitability relative to siloed decision-making frameworks, while the behaviors of the trained RL agents align closely with the managerial insights from our theoretical model. Taken together, this work provides a scalable, interpretable RL-based solution to enable effective cross-functional coordination in complex business settings. 

**Abstract (ZH)**: 有效跨功能协调对于增强企业整体盈利能力至关重要，尤其是面对日益增长的组织复杂性和规模。近年来，尤其是强化学习（RL）的进展为解决这一根本挑战提供了有希望的途径。本文提出了一种针对不同功能模块联合优化的统一多Agent RL框架，通过协调库存补充和个性化产品推荐进行阐述。我们首先开发了一个综合的理论模型来捕捉这些功能之间的复杂相互作用，并推导出表征最优协调的分析基准。分析揭示了产品和服务之间同步调整的模式，突显了协调决策的重要性。利用这些见解，我们设计了一种新颖的多时间尺度多Agent RL架构，根据部门功能分解策略成分，并基于任务复杂性和响应性分配不同的学习速度。我们的无模型多Agent设计增强了可扩展性和部署灵活性，而多时间尺度更新增强了跨异质决策的收敛稳定性和适应性。进一步建立了所提算法的渐近收敛性。广泛的模拟实验表明，所提出的方法相对于孤立决策框架显著提高了盈利能力，而训练的RL代理行为与我们理论模型的管理洞察高度一致。总体而言，本文为复杂商业环境中的有效跨功能协调提供了一个可扩展且可解释的RL基解决方案。 

---
# Don't Pass$\mathtt{@}k$: A Bayesian Framework for Large Language Model Evaluation 

**Title (ZH)**: 不要忽视@k：一种大规模语言模型评估的贝叶斯框架 

**Authors**: Mohsen Hariri, Amirhossein Samandar, Michael Hinczewski, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2510.04265)  

**Abstract**: Pass$@k$ is widely used to report performance for LLM reasoning, but it often yields unstable, misleading rankings, especially when the number of trials (samples) is limited and compute is constrained. We present a principled Bayesian evaluation framework that replaces Pass$@k$ and average accuracy over $N$ trials (avg$@N$) with posterior estimates of a model's underlying success probability and credible intervals, yielding stable rankings and a transparent decision rule for differences. Evaluation outcomes are modeled as categorical (not just 0/1) with a Dirichlet prior, giving closed-form expressions for the posterior mean and uncertainty of any weighted rubric and enabling the use of prior evidence when appropriate. Theoretically, under a uniform prior, the Bayesian posterior mean is order-equivalent to average accuracy (Pass$@1$), explaining its empirical robustness while adding principled uncertainty. Empirically, in simulations with known ground-truth success rates and on AIME'24/'25, HMMT'25, and BrUMO'25, the Bayesian/avg procedure achieves faster convergence and greater rank stability than Pass$@k$ and recent variants, enabling reliable comparisons at far smaller sample counts. The framework clarifies when observed gaps are statistically meaningful (non-overlapping credible intervals) versus noise, and it naturally extends to graded, rubric-based evaluations. Together, these results recommend replacing Pass$@k$ for LLM evaluation and ranking with a posterior-based, compute-efficient protocol that unifies binary and non-binary evaluation while making uncertainty explicit. Code is available at this https URL 

**Abstract (ZH)**: 基于贝叶斯方法的LLM推理评估框架：替代Pass@k和平均准确率，实现稳定排名和透明决策规则 

---
# AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework 

**Title (ZH)**: AgentRL: 使用多轮多任务框架扩展代理强化学习 

**Authors**: Hanchen Zhang, Xiao Liu, Bowen Lv, Xueqiao Sun, Bohao Jing, Iat Long Iong, Zhenyu Hou, Zehan Qi, Hanyu Lai, Yifan Xu, Rui Lu, Hongning Wang, Jie Tang, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04206)  

**Abstract**: Recent advances in large language models (LLMs) have sparked growing interest in building generalist agents that can learn through online interactions. However, applying reinforcement learning (RL) to train LLM agents in multi-turn, multi-task settings remains challenging due to lack of scalable infrastructure and stable training algorithms. In this work, we present the AgentRL framework for scalable multi-turn, multi-task agentic RL training. On the infrastructure side, AgentRL features a fully-asynchronous generation-training pipeline for efficient multi-turn RL. To support heterogeneous environment development in multi-task RL, we design a unified function-call based API interface, containerized environment development, and a centralized controller. On the algorithm side, we propose cross-policy sampling to encourage model exploration in multi-turn settings and task advantage normalization to stabilize multi-task training. Experiments show that AgentRL, trained on open LLMs across five agentic tasks, significantly outperforms GPT-5, Clause-Sonnet-4, DeepSeek-R1, and other open-source LLM agents. Multi-task training with AgentRL matches the best results among all task-specific models. AgentRL is open-sourced at this https URL. The algorithm and framework are adopted in building \textsc{\href{this https URL}{AutoGLM}}. 

**Abstract (ZH)**: Recent advances in大型语言模型(LLMs)近年来在大型语言模型(LLMs)方面的进展激发了构建通过在线交互学习的通用代理的兴趣。然而，在多轮、多任务设置中应用强化学习(RL)训练LLM代理仍然具有挑战性，原因是在可扩展基础设施和稳定训练算法方面存在不足。在本文中，我们提出了AgentRL框架，用于可扩展的多轮、多任务代理型RL训练。在基础设施方面，AgentRL具备完全异步的生成-训练流水线，用于高效多轮RL。为支持多任务RL中的异构环境开发，我们设计了一种统一的功能调用接口API、容器化环境开发和中心控制器。从算法角度来看，我们提出了跨策略采样以鼓励多轮设置中的模型探索，并提出了任务优势归一化以稳定多任务训练。实验显示，使用AgentRL训练的开放LLM代理在五项代理任务上显著优于GPT-5、Clause-Sonnet-4、DeepSeek-R1和其他开源LLM代理。使用AgentRL进行多任务训练达到了所有任务特定模型的最佳结果。AgentRL开源于[this https URL]。该算法和框架被用于构建AutoGLM。 

---
# COSMO-RL: Towards Trustworthy LMRMs via Joint Safety and Stability 

**Title (ZH)**: COSMO-RL：通过联合安全性和稳定性迈向可信赖的LMRMs 

**Authors**: Yizhuo Ding, Mingkang Chen, Qiuhua Liu, Fenghua Weng, Wanying Qu, Yue Yang, Yugang Jiang, Zuxuan Wu, Yanwei Fu, Wenqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04196)  

**Abstract**: Large Multimodal Reasoning Models (LMRMs) are moving into real applications, where they must be both useful and safe. Safety is especially challenging in multimodal settings: images and text can be combined to bypass guardrails, and single objective training can cause policy drift that yields over-refusal on benign inputs or unsafe compliance on risky ones. We present COSMO-RL, a mixed reinforcement learning framework that trains reasoning oriented LMRMs under multimodal, multitask, and multiobjective signals, and we release the resulting model, COSMO-R1. Our approach aims to let safety and capability grow together in one stable pipeline rather than competing during alignment. In experiments, COSMO-R1 improves safety while maintaining-and often improving multimodal reasoning and instruction following, shows stronger robustness to multimodal jailbreaks, and reduces unnecessary refusals. The framework also transfers across backbones with consistent gains. Ablations support the design choices, indicating a simple path to advancing safety and general capability together in LMRMs. 

**Abstract (ZH)**: 大规模多模态推理模型（LMRMs）在实际应用中既要实用又要安全。特别是在多模态设置中，图像和文本可以结合以绕过安全限制，单一目标训练可能导致政策偏移，从而在良性输入上过度拒绝或在风险输入上不安全地合规。我们提出了COSMO-RL，一种混合强化学习框架，用于在多模态、多任务和多目标信号下训练以推理为导向的LMRMs，并发布了相应的模型COSMO-R1。我们的方法旨在让安全性和能力在单一稳定的流水线中共同成长，而不是在对齐过程中相互竞争。实验结果显示，COSMO-R1在提高安全性的同时，保持并往往增强了多模态推理和指令遵循能力，展示了更强的多模态绕过鲁棒性，并减少了不必要的拒绝。该框架还跨 backbone 获得一致的收益。消融实验支持了设计选择，表明了一条简单路径，可以在LMRMs中同时推进安全性和通用能力。 

---
# Constructing coherent spatial memory in LLM agents through graph rectification 

**Title (ZH)**: 通过图修正构建LLM代理的一致空间记忆 

**Authors**: Puzhen Zhang, Xuyang Chen, Yu Feng, Yuhan Jiang, Liqiu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04195)  

**Abstract**: Given a map description through global traversal navigation instructions (e.g., visiting each room sequentially with action signals such as north, west, etc.), an LLM can often infer the implicit spatial layout of the environment and answer user queries by providing a shortest path from a start to a destination (for instance, navigating from the lobby to a meeting room via the hall and elevator). However, such context-dependent querying becomes incapable as the environment grows much longer, motivating the need for incremental map construction that builds a complete topological graph from stepwise observations. We propose a framework for LLM-driven construction and map repair, designed to detect, localize, and correct structural inconsistencies in incrementally constructed navigation graphs. Central to our method is the Version Control, which records the full history of graph edits and their source observations, enabling fine-grained rollback, conflict tracing, and repair evaluation. We further introduce an Edge Impact Score to prioritize minimal-cost repairs based on structural reachability, path usage, and conflict propagation. To properly evaluate our approach, we create a refined version of the MANGO benchmark dataset by systematically removing non-topological actions and inherent structural conflicts, providing a cleaner testbed for LLM-driven construction and map repair. Our approach significantly improves map correctness and robustness, especially in scenarios with entangled or chained inconsistencies. Our results highlight the importance of introspective, history-aware repair mechanisms for maintaining coherent spatial memory in LLM agents. 

**Abstract (ZH)**: 基于LLM的增量地图构建与修复框架：检测、定位和纠正增量构建导航图中的结构不一致性 

---
# Open Agent Specification (Agent Spec) Technical Report 

**Title (ZH)**: 开放代理规范（代理规范）技术报告 

**Authors**: Yassine Benajiba, Cesare Bernardis, Vladislav Blinov, Paul Cayet, Hassan Chafi, Abderrahim Fathan, Louis Faucon, Damien Hilloulin, Sungpack Hong, Ingo Kossyk, Rhicheek Patra, Sujith Ravi, Jonas Schweizer, Jyotika Singh, Shailender Singh, Xuelin Situ, Weiyi Sun, Jerry Xu, Ying Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04173)  

**Abstract**: Open Agent Specification (Agent Spec) is a declarative language that allows AI agents and their workflows to be defined in a way that is compatible across different AI frameworks, promoting portability and interoperability within AI Agent frameworks.
Agent Spec aims to resolve the challenges of fragmented agent development by providing a common unified specification that allows AI agents to be designed once and deployed across various frameworks, improving interoperability and reusability, and reducing redundant development efforts. Additionally, Agent Spec facilitates development tools and portability, allowing AI agents to be defined independently of their execution environment and enabling teams to exchange solutions without implementation-specific limitations.
Agent Spec benefits four key groups: (i) Agent developers, who gain access to a superset of reusable components and design patterns, enabling them to leverage a broader range of functionalities; (ii) Agent framework and tool developers, who can use Agent Spec as an interchange format and therefore benefit from the support of other frameworks as well as other tools; (iii) Researchers, who can achieve reproducible results and comparability, facilitating more reliable and consistent outcomes; (iv) Enterprises, which benefit from faster prototype-to-deployment, increased productivity, as well as greater scalability and maintainability for their AI agent solutions. This technical report provides an overview of the technical foundations of Agent Spec, including motivation, benefits, and future developments. 

**Abstract (ZH)**: 开源代理规范（Agent Spec）是一种声明性语言，允许AI代理及其工作流以跨不同AI框架兼容的方式被定义，促进AI代理框架内的便捷性和互操作性。 

---
# The Artificial Intelligence Cognitive Examination: A Survey on the Evolution of Multimodal Evaluation from Recognition to Reasoning 

**Title (ZH)**: 人工智能认知评估：从识别到推理的多模态评价演进综述 

**Authors**: Mayank Ravishankara, Varindra V. Persad Maharaj  

**Link**: [PDF](https://arxiv.org/pdf/2510.04141)  

**Abstract**: This survey paper chronicles the evolution of evaluation in multimodal artificial intelligence (AI), framing it as a progression of increasingly sophisticated "cognitive examinations." We argue that the field is undergoing a paradigm shift, moving from simple recognition tasks that test "what" a model sees, to complex reasoning benchmarks that probe "why" and "how" it understands. This evolution is driven by the saturation of older benchmarks, where high performance often masks fundamental weaknesses. We chart the journey from the foundational "knowledge tests" of the ImageNet era to the "applied logic and comprehension" exams such as GQA and Visual Commonsense Reasoning (VCR), which were designed specifically to diagnose systemic flaws such as shortcut learning and failures in compositional generalization. We then survey the current frontier of "expert-level integration" benchmarks (e.g., MMBench, SEED-Bench, MMMU) designed for today's powerful multimodal large language models (MLLMs), which increasingly evaluate the reasoning process itself. Finally, we explore the uncharted territories of evaluating abstract, creative, and social intelligence. We conclude that the narrative of AI evaluation is not merely a history of datasets, but a continuous, adversarial process of designing better examinations that, in turn, redefine our goals for creating truly intelligent systems. 

**Abstract (ZH)**: 这篇综述论文记载了多模态人工智能评估的发展，将其视为逐渐复杂的“认知检测”进化的历程。我们认为该领域正经历范式的转变，从测试模型“看到什么”的简单识别任务，转向探究模型“为何”和“如何”理解的复杂推理基准。这种进化是由老旧基准的饱和推动的，高性能往往掩盖了根本性的弱点。我们从ImageNet时代的“知识测试”过渡到GQA和视觉常识推理（VCR）等旨在诊断系统性缺陷（如捷径学习和合成泛化失败）的“应用逻辑与理解”测试。接着，我们综述了当前前沿的“专家级集成”基准（如MM Bench、SEED-Bench、MMMU），这些基准用于评估当今强大的多模态大型语言模型（MLLMs）的推理过程本身。最后，我们探索了评估抽象、创造性和社交智能的未开发领域。我们得出结论，人工智能评估的叙事不仅是一部数据集的历史，而是一个持续且对抗性的过程，即设计更好的检测手段，随之重新定义我们创造真正智能系统的目标。 

---
# Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs 

**Title (ZH)**: 面向大语言模型强化学习的有效且多样的探索选择专家指导 

**Authors**: Zishang Jiang, Jinyi Han, Tingyun Li, Xinyi Wang, Sihang Jiang, Jiaqing Liang, Zhaoqian Dai, Shuguang Ma, Fei Yu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04140)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a widely adopted technique for enhancing the reasoning ability of Large Language Models (LLMs). However, the effectiveness of RLVR strongly depends on the capability of base models. This issue arises because it requires the model to have sufficient capability to perform high-quality exploration, which involves both effectiveness and diversity. Unfortunately, existing methods address this issue by imitating expert trajectories, which improve effectiveness but neglect diversity. To address this, we argue that the expert only needs to provide guidance only at critical decision points rather than the entire reasoning path. Based on this insight, we propose MENTOR: Mixed-policy Expert Navigation for Token-level Optimization of Reasoning, a framework that provides expert guidance only at critical decision points to perform effective and diverse exploration in RLVR. Extensive experiments show that MENTOR enables models capture the essence of expert strategies rather than surface imitation, thereby performing high-quality exploration and achieving superior overall performance. Our code is available online. 

**Abstract (ZH)**: 具备可验证奖励的强化学习（RLVR）已成为增强大型语言模型（LLMs）推理能力的一种广泛采用的技术。然而，RLVR的效果高度依赖于基础模型的能力。这主要是因为需要模型具备足够的能力来进行高质量的探索，包括有效性和多样性。不幸的是，现有方法通过模仿专家路径来解决这个问题，这虽然提高了有效性，但忽视了多样性。为了应对这一问题，我们认为专家只需在关键决策点提供指导，而不是整个推理路径。基于这一洞察，我们提出了一种框架：Mixed-policy Expert Navigation for Token-level Optimization of Reasoning（混合策略专家导航，用于词汇层面推理优化），该框架仅在关键决策点提供专家指导，以在RLVR中进行有效的和多样化的探索。广泛实验证明，MENTOR能够使模型捕捉到专家策略的本质，而不是表面的模仿，从而实现高质量的探索并取得优越的整体性能。我们的代码已在线提供。 

---
# Internal states before wait modulate reasoning patterns 

**Title (ZH)**: 等待前的内部状态调节推理模式 

**Authors**: Dmitrii Troitskii, Koyena Pal, Chris Wendler, Callum Stuart McDougall, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.04128)  

**Abstract**: Prior work has shown that a significant driver of performance in reasoning models is their ability to reason and self-correct. A distinctive marker in these reasoning traces is the token wait, which often signals reasoning behavior such as backtracking. Despite being such a complex behavior, little is understood of exactly why models do or do not decide to reason in this particular manner, which limits our understanding of what makes a reasoning model so effective. In this work, we address the question whether model's latents preceding wait tokens contain relevant information for modulating the subsequent reasoning process. We train crosscoders at multiple layers of DeepSeek-R1-Distill-Llama-8B and its base version, and introduce a latent attribution technique in the crosscoder setting. We locate a small set of features relevant for promoting/suppressing wait tokens' probabilities. Finally, through a targeted series of experiments analyzing max activating examples and causal interventions, we show that many of our identified features indeed are relevant for the reasoning process and give rise to different types of reasoning patterns such as restarting from the beginning, recalling prior knowledge, expressing uncertainty, and double-checking. 

**Abstract (ZH)**: 先前的研究表明，推理模型的性能显著取决于其推理和自我纠正的能力。这些推理路径中的一个独特标志是等待标记（token wait），它通常表明了诸如回溯等推理行为。尽管这是一个复杂的行为，但对于模型为何或为何不以这种方式推理的具体原因还知之甚少，这限制了我们对哪些因素使推理模型如此有效这一理解。在本工作中，我们探讨了模型等待标记之前的潜在特征是否包含调节后续推理过程的相关信息。我们对DeepSeek-R1-Distill-Llama-8B及其基版本的多个层级进行了交叉编码器训练，并引入了一种在交叉编码器设置中的潜在特征归因技术。我们定位到一组促进或抑制等待标记概率的相关特征。最后，通过分析最大激活示例和因果干预的一系列针对性实验，我们证明了我们识别出的许多特征确实对推理过程具有相关性，并导致了不同的推理模式，如重新开始、调用先前的知识、表达不确定性以及复查。 

---
# Searching Meta Reasoning Skeleton to Guide LLM Reasoning 

**Title (ZH)**: 搜索元推理框架以引导LLM推理 

**Authors**: Ziying Zhang, Yaqing Wang, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04116)  

**Abstract**: Meta reasoning behaviors work as a skeleton to guide large language model (LLM) reasoning, thus help to improve reasoning performance. However, prior researches implement meta reasoning skeleton with manually designed structure, limiting ability to adapt to query-specific requirement and capture intricate logical dependency among reasoning steps. To deal with the challenges, we represent meta reasoning skeleton with directed acyclic graph (DAG) to unify skeletons proposed in prior works and model intricate logical dependency. Then we propose AutoMR, a framework that searches for query-aware meta reasoning skeleton automatically inspired by automated machine learning (AutoML). Specifically, we construct search space based on DAG representation of skeleton and then formulate the search problem. We design a dynamic skeleton sampling algorithm by expanding meta reasoning skeleton along with reasoning context at inference time. This algorithm can derive any meta reasoning skeleton in search space efficiently and adapt skeleton to evolving base reasoning context, thus enable efficient query-aware skeleton search. We conduct experiments on extensive benchmark datasets. Experimental results show that AutoMR achieves better reasoning performance than previous works broadly. 

**Abstract (ZH)**: 元推理行为作为框架引导大型语言模型（LLM）推理，从而提高推理性能。然而，先前的研究通过手动设计结构实现元推理框架，限制了其适应查询特定要求和捕捉推理步骤之间复杂逻辑依赖的能力。为应对这些挑战，我们使用有向无环图（DAG）表示元推理框架，统一了先前工作提出的框架，并建模了复杂的逻辑依赖关系。随后，我们提出AutoMR框架，该框架借鉴自动化机器学习（AutoML）的思想，自动搜索查询感知的元推理框架。具体而言，我们基于DAG表示的框架构建搜索空间，并形式化搜索问题。我们设计了一种动态框架采样算法，在推理时沿着推理上下文扩展元推理框架，该算法可以高效地导出搜索空间中的任何元推理框架，并适应演化的基础推理上下文，从而实现高效的查询感知框架搜索。我们在广泛的基准数据集上进行了实验。实验结果表明，AutoMR在广泛情况下比先前工作实现了更好的推理性能。 

---
# WebRenderBench: Enhancing Web Interface Generation through Layout-Style Consistency and Reinforcement Learning 

**Title (ZH)**: WebRenderBench：通过布局-样式一致性与强化学习提升Web界面生成 

**Authors**: Peichao Lai, Jinhui Zhuang, Kexuan Zhang, Ningchang Xiong, Shengjie Wang, Yanwei Xu, Chong Chen, Yilei Wang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2510.04097)  

**Abstract**: Automating the conversion of UI images into web code is a critical task for front-end development and rapid prototyping. Advances in multimodal large language models (MLLMs) have made WebUI-to-Code increasingly feasible, yet existing benchmarks remain limited in data diversity and evaluation reliability. To address these issues, we present WebRenderBench, a large-scale benchmark of 22.5k webpages collected from real-world portal sites, offering greater diversity, complexity, and realism than prior benchmarks. We further propose a novel evaluation metric that measures layout and style consistency from the final rendered pages. Unlike vision-based methods that rely on costly LLM reasoning or structure-based comparisons vulnerable to noise and asymmetry, our approach enables more efficient, objective, and reliable UI quality assessment. Finally, we introduce the Automated Layout and Style Inspection Agent (ALISA), which integrates this metric into reinforcement learning as a reward signal to enhance training on crawled asymmetric webpages. Experiments show that ALISA significantly boosts generation performance, achieving state-of-the-art results across multiple metrics. 

**Abstract (ZH)**: 自动化将UI图像转换为网页代码是前端开发和快速原型设计中的关键任务。多模态大型语言模型的进步使得WebUI-to-Code越来越可行，但现有基准在数据多样性和评估可靠性方面仍然有限。为了解决这些问题，我们介绍了WebRenderBench，这是一个由22500个网页组成的大规模基准，这些网页来自实际门户站点，提供了比先前基准更多的多样性和现实性。进一步提出了一种新的评估指标，用于衡量最终渲染页面的布局和样式一致性。与依赖昂贵的LLM推理或易受噪声和不对称影响的结构比较方法不同，我们的方法能够更高效、客观和可靠地评估UI质量。最后，我们引入了自动化布局和样式检查代理（ALISA），将该指标整合到强化学习中作为奖励信号，以增强抓取的不对称网页上的训练。实验结果表明，ALISA显著提升了生成性能，在多个指标上达到了最先进的结果。 

---
# Harnessing LLM for Noise-Robust Cognitive Diagnosis in Web-Based Intelligent Education Systems 

**Title (ZH)**: 利用大语言模型在基于网络的智能教育系统中实现抗噪认知诊断 

**Authors**: Guixian Zhang, Guan Yuan, Ziqi Xu, Yanmei Zhang, Zhenyun Deng, Debo Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04093)  

**Abstract**: Cognitive diagnostics in the Web-based Intelligent Education System (WIES) aims to assess students' mastery of knowledge concepts from heterogeneous, noisy interactions. Recent work has tried to utilize Large Language Models (LLMs) for cognitive diagnosis, yet LLMs struggle with structured data and are prone to noise-induced misjudgments. Specially, WIES's open environment continuously attracts new students and produces vast amounts of response logs, exacerbating the data imbalance and noise issues inherent in traditional educational systems. To address these challenges, we propose DLLM, a Diffusion-based LLM framework for noise-robust cognitive diagnosis. DLLM first constructs independent subgraphs based on response correctness, then applies relation augmentation alignment module to mitigate data imbalance. The two subgraph representations are then fused and aligned with LLM-derived, semantically augmented representations. Importantly, before each alignment step, DLLM employs a two-stage denoising diffusion module to eliminate intrinsic noise while assisting structural representation alignment. Specifically, unconditional denoising diffusion first removes erroneous information, followed by conditional denoising diffusion based on graph-guided to eliminate misleading information. Finally, the noise-robust representation that integrates semantic knowledge and structural information is fed into existing cognitive diagnosis models for prediction. Experimental results on three publicly available web-based educational platform datasets demonstrate that our DLLM achieves optimal predictive performance across varying noise levels, which demonstrates that DLLM achieves noise robustness while effectively leveraging semantic knowledge from LLM. 

**Abstract (ZH)**: 基于Web的智能教育系统（WIES）中的认知诊断旨在评估学生从异构、嘈杂的交互中对知识概念的掌握情况。最近的研究尝试利用大规模语言模型（LLMs）进行认知诊断，但LLMs在处理结构化数据方面存在困难，并且容易因噪声而产生误判。特别是，WIES的开放环境持续吸引新学生并产生大量响应日志，加剧了传统教育系统中固有的数据不平衡和噪声问题。为应对这些挑战，我们提出了一种基于扩散的LLM框架（DLLM）以实现抗噪声的认知诊断。DLLM首先基于回答正确性构建独立子图，然后应用关系增强对齐模块以减轻数据不平衡的问题。然后将两个子图表示与LLM衍生的语义增强表示进行融合和对齐。重要的是，每次对齐之前，DLLM采用两阶段去噪扩散模块去除固有的噪声以辅助结构表示对齐。具体来说，无条件去噪扩散首先去除错误信息，随后基于图导向的有条件去噪扩散消除误导信息。最后，融合了语义知识和结构信息的抗噪声表示被输入到现有的认知诊断模型进行预测。在三个公开的基于Web的教育平台数据集上的实验结果表明，我们的DLLM在不同噪声水平下实现了最优的预测性能，这表明DLLM在有效利用LLM语义知识的同时实现了噪声鲁棒性。 

---
# SPOGW: a Score-based Preference Optimization method via Group-Wise comparison for workflows 

**Title (ZH)**: SPOGW：基于组内比较的评分驱动的工作流偏好优化方法 

**Authors**: Yitong Cui, Liu Liu, Baosheng Yu, Jiayan Qiu, Xikai Zhang, Likang Xiao, Yixing Liu, Quan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04089)  

**Abstract**: Large language models (LLMs) have exhibited significant capabilities in addressing challenging problems throughout various fields, often through the use of agentic workflows that adhere to structured instructions and multi-step procedures. However, designing such workflows demands substantial manual effort, posing challenges to scalability and generalizability. Recent studies have aimed to minimize the human intervention needed for their construction, leading to advances in automated techniques for optimizing agentic workflows. However, current approaches are often constrained by their limited representational capacity, insufficient adaptability, weak scalability, and pairwise comparison paradigm -- issues that stem primarily from a dependence on discrete optimization techniques. To overcome these limitations, we introduce a new score-based preference approach, refereed as SPOGW, which operates directly on cardinal reward signals through group-wise comparison and enables more efficient and stable optimization in a continuous space. SPOGW incorporates Iterative offline GRPO (ioGRPO) with advantage-masked KL divergence (mKL), which regulates training update by placing greater emphasis on the advantageous regions of the policy response. In five benchmark datasets covering mathematical reasoning, coding, and question answering, SPOGW matches or exceeds the performance of current state-of-the-art approaches, presenting a viable and forward-looking methodology for automated generation and optimization of agentic workflows. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各领域中展示了解决复杂问题的显著能力，通常通过遵守结构化指令和多步骤程序的代理工作流来实现。然而，设计这样的工作流需要大量的手动工作，这给规模化和通用性带来了挑战。近期的研究旨在减少构建这些工作流所需的人工干预，推动了自动优化代理工作流技术的进步。然而，当前的方法往往受限于其有限的表现能力、不足的适应性、弱的可扩展性以及成对比较范式——这些问题主要源自对离散优化技术的依赖。为克服这些限制，我们引入了一种新的基于评分的偏好方法，称为SPOGW，该方法直接在卡尺奖励信号上进行组间比较，从而在连续空间中实现更高效的稳定优化。SPOGW结合了迭代离线GRPO（ioGRPO）和优势屏蔽的KL散度（mKL），通过强调政策响应的优势区域来调节训练更新。在涵盖数学推理、编码和问答的五个基准数据集中，SPOGW匹配或超越了当前最先进的方法，为自动生成和优化代理工作流提供了可行且前瞻性的方法。 

---
# Moral Anchor System: A Predictive Framework for AI Value Alignment and Drift Prevention 

**Title (ZH)**: 道德锚定系统：一种预测性AI价值对齐与偏移预防框架 

**Authors**: Santhosh Kumar Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2510.04073)  

**Abstract**: The rise of artificial intelligence (AI) as super-capable assistants has transformed productivity and decision-making across domains. Yet, this integration raises critical concerns about value alignment - ensuring AI behaviors remain consistent with human ethics and intentions. A key risk is value drift, where AI systems deviate from aligned values due to evolving contexts, learning dynamics, or unintended optimizations, potentially leading to inefficiencies or ethical breaches. We propose the Moral Anchor System (MAS), a novel framework to detect, predict, and mitigate value drift in AI agents. MAS combines real-time Bayesian inference for monitoring value states, LSTM networks for forecasting drift, and a human-centric governance layer for adaptive interventions. It emphasizes low-latency responses (<20 ms) to prevent breaches, while reducing false positives and alert fatigue via supervised fine-tuning with human feedback. Our hypothesis: integrating probabilistic drift detection, predictive analytics, and adaptive governance can reduce value drift incidents by 80 percent or more in simulations, maintaining high detection accuracy (85 percent) and low false positive rates (0.08 post-adaptation). Rigorous experiments with goal-misaligned agents validate MAS's scalability and responsiveness. MAS's originality lies in its predictive and adaptive nature, contrasting static alignment methods. Contributions include: (1) MAS architecture for AI integration; (2) empirical results prioritizing speed and usability; (3) cross-domain applicability insights; and (4) open-source code for replication. 

**Abstract (ZH)**: 人工智能（AI）作为超级能力助手的崛起已 transforming生产力和决策领域。然而，这种整合引发了关于价值对齐的重要关切——确保AI行为与人类伦理和意图保持一致。一个关键风险是价值偏移，即由于环境变化、学习动态或意外优化，AI系统可能偏离对齐价值观，导致效率低下或伦理违规。我们提出道德锚系统（MAS），这是一种新颖的框架，用于检测、预测和减轻AI代理的价值偏移。MAS结合了实时贝叶斯推理进行价值状态监控，LSTM网络进行偏移预测，以及以人为中心的治理层进行适应性干预。它强调低于20毫秒的低延迟响应，以防止违规行为，同时通过监督微调和人类反馈减少误报和警报疲劳。我们的假设：结合概率偏移检测、预测分析和适应性治理可以在模拟中将价值偏移事件减少80％或更多，保持高检测准确性（85％）和低误报率（在适应后为0.08）。严格的实验验证了MAS在目标错配代理中的可扩展性和响应性。MAS的独特性在于其预测和适应性，与静态对齐方法形成对比。贡献包括：（1）AI集成的MAS架构；（2）优先考虑速度和易用性的实证结果；（3）跨领域的适用性见解；（4）开源代码以供复制。 

---
# Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion 

**Title (ZH)**: 深层情感解码：大型语言模型情感表示、保持与表达的系统研究 

**Authors**: Jingxiang Zhang, Lujia Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04064)  

**Abstract**: Large Language Models (LLMs) are increasingly expected to navigate the nuances of human emotion. While research confirms that LLMs can simulate emotional intelligence, their internal emotional mechanisms remain largely unexplored. This paper investigates the latent emotional representations within modern LLMs by asking: how, where, and for how long is emotion encoded in their neural architecture? To address this, we introduce a novel, large-scale Reddit corpus of approximately 400,000 utterances, balanced across seven basic emotions through a multi-stage process of classification, rewriting, and synthetic generation. Using this dataset, we employ lightweight "probes" to read out information from the hidden layers of various Qwen3 and LLaMA models without altering their parameters. Our findings reveal that LLMs develop a surprisingly well-defined internal geometry of emotion, which sharpens with model scale and significantly outperforms zero-shot prompting. We demonstrate that this emotional signal is not a final-layer phenomenon but emerges early and peaks mid-network. Furthermore, the internal states are both malleable (they can be influenced by simple system prompts) and persistent, as the initial emotional tone remains detectable for hundreds of subsequent tokens. We contribute our dataset, an open-source probing toolkit, and a detailed map of the emotional landscape within LLMs, offering crucial insights for developing more transparent and aligned AI systems. The code and dataset are open-sourced. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益被期望能够理解和导航人类情感的细微差别。尽管研究证实LLMs能够模拟情感 intelligence，但其内部的情感机制仍 largely 未被探索。本文通过探究现代LLMs 中潜藏的情感表示，探讨情感如何、在哪里以及多长时间被编码在其神经架构中。为此，我们引入了一个包含约400,000个表达的大规模Reddit语料库，通过多阶段的分类、重写和合成生成过程，平衡分布在七种基本情感中。利用该数据集，我们采用轻量级的“探针”来读取不同Qwen3和LLaMA模型隐藏层中的信息而无需改变其参数。研究发现，LLMs形成了一个令人惊讶地明确的情感内部几何结构，这种结构随着模型规模的增加而变得更加精确，并显著优于零样本提示。我们证明了这种情感信号并不是在模型的最终层才出现，而是在网络早期出现并在中途达到峰值。此外，这些内部状态既灵活（可以被简单的系统提示所影响）又持久，初始的情感基调在后续数百个标记中仍然可以检测到。我们提供了数据集、开源探针工具包以及LLMs内部情感景观的详细地图，为开发更透明和对齐的AI系统提供了宝贵的见解。代码和数据集均已开源。 

---
# Toward a unified framework for data-efficient evaluation of large language models 

**Title (ZH)**: 面向大数据高效评估大型语言模型的统一框架 

**Authors**: Lele Liao, Qile Zhang, Ruofan Wu, Guanhua Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04051)  

**Abstract**: Evaluating large language models (LLMs) on comprehensive benchmarks is a cornerstone of their development, yet it's often computationally and financially prohibitive. While Item Response Theory (IRT) offers a promising path toward data-efficient evaluation by disentangling model capability from item difficulty, existing IRT-based methods are hampered by significant limitations. They are typically restricted to binary correctness metrics, failing to natively handle the continuous scores used in generative tasks, and they operate on single benchmarks, ignoring valuable structural knowledge like correlations across different metrics or benchmarks. To overcome these challenges, we introduce LEGO-IRT, a unified and flexible framework for data-efficient LLM evaluation. LEGO-IRT's novel design natively supports both binary and continuous evaluation metrics. Moreover, it introduces a factorized architecture to explicitly model and leverage structural knowledge, decomposing model ability estimates into a general component and structure-specific (e.g., per-metric or per-benchmark) components. Through extensive experiments involving $70$ LLMs across $5$ benchmarks, we show that LEGO-IRT achieves stable capability estimates using just $3\%$ of the total evaluation items. We demonstrate that incorporating structural knowledge reduces estimation error by up to $10\%$ and reveal that the latent abilities estimated by our framework may align more closely with human preferences. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在综合基准上的评估是其发展中的一个基石，但往往因计算和财务成本过高而难以实现。虽然项目反应理论（IRT）为通过分离模型能力与项目难度提供了一条有希望的数据高效评估途径，但现有基于IRT的方法受到显著限制。这些方法通常仅支持二元正确性指标，无法自然处理生成任务中使用的连续分数，并且仅在单一基准上操作，忽略了不同指标或基准之间的有价值结构知识。为克服这些挑战，我们提出LEGO-IRT，这是一种统一且灵活的大规模语言模型数据高效评估框架。LEGO-IRT的全新设计能够原生支持二元和连续评估指标。此外，它引入了一种分解架构，以明确建模和利用结构知识，将模型能力估计分解为一般性组件和结构特定性组件（例如，基于指标或基准）。通过涉及5个基准上70个大规模语言模型的广泛实验，我们证明LEGO-IRT仅使用总评估项目的3%即可获得稳定的能力估计。我们展示了整合结构知识可将估计误差降低最多10%，并揭示了我们框架估计的潜在能力可能更接近人类偏好。 

---
# Increasing LLM response trustworthiness using voting ensembles 

**Title (ZH)**: 使用投票集成提高大模型回答的可信度 

**Authors**: Aparna Nair-Kanneganti, Trevor J. Chan, Shir Goldfinger, Emily Mackay, Brian Anthony, Alison Pouch  

**Link**: [PDF](https://arxiv.org/pdf/2510.04048)  

**Abstract**: Despite huge advances, LLMs still lack convenient and reliable methods to quantify the uncertainty in their responses, making them difficult to trust in high-stakes applications. One of the simplest approaches to eliciting more accurate answers is to select the mode of many responses, a technique known as ensembling. In this work, we expand on typical ensembling approaches by looking at ensembles with a variable voting threshold. We introduce a theoretical framework for question answering and show that, by permitting ensembles to "abstain" from providing an answer when the dominant response falls short of the threshold, it is possible to dramatically increase the trustworthiness of the remaining answers. From this framework, we derive theoretical results as well as report experimental results on two problem domains: arithmetic problem solving and clinical-note question-answering. In both domains, we observe that large gains in answer trustworthiness can be achieved using highly restrictive voting ensembles, while incurring relatively modest reductions in response yield and accuracy. Due to this quality, voting ensembles may be particularly useful in applications - such as healthcare and data annotation - that require a high degree of certainty but which may not require that every question receive an automated answer. 

**Abstract (ZH)**: 尽管取得了巨大进展，大型语言模型仍缺乏方便可靠的方法来量化其回答的不确定性，这使得它们在高风险应用中难以信赖。一種簡單的方法是選擇多個回答中的-mode，這種技術稱為集成。在本工作中，我們擴展了典型的集成方法，考慮了具有可變投票閾值的集成。我們提出了問答的理論框架，并表明，通過允許集成在主回答未達到閾值時“拒答”，可以在保留更多可信回答的同时大幅提高剩余答案的可信度。基于這個框架，我們得到了理論結果，並在算術問題解決和臨床記錄問題回答兩個領域報告了實驗結果。在兩個領域中，我們觀察到，使用非常苛刻的投票集成可以大幅度提高答案的可信度，同時造成的回答產出和準確度降低相對較小。由於這種質量，投票集成在需要高度確定性但不一定每個問題都需要自動回答的應用（如醫療保健和數據標注）中可能特別有用。 

---
# FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of Chain-of-Thought Reasoning 

**Title (ZH)**: FaithCoT-Bench: 推理链层面可信度基准评测 

**Authors**: Xu Shen, Song Wang, Zhen Tan, Laura Yao, Xinyu Zhao, Kaidi Xu, Xin Wang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04040)  

**Abstract**: Large language models (LLMs) increasingly rely on Chain-of-Thought (CoT) prompting to improve problem-solving and provide seemingly transparent explanations. However, growing evidence shows that CoT often fail to faithfully represent the underlying reasoning process, raising concerns about their reliability in high-risk applications. Although prior studies have focused on mechanism-level analyses showing that CoTs can be unfaithful, they leave open the practical challenge of deciding whether a specific trajectory is faithful to the internal reasoning of the model. To address this gap, we introduce FaithCoT-Bench, a unified benchmark for instance-level CoT unfaithfulness detection. Our framework establishes a rigorous task formulation that formulates unfaithfulness detection as a discriminative decision problem, and provides FINE-CoT (Faithfulness instance evaluation for Chain-of-Thought), an expert-annotated collection of over 1,000 trajectories generated by four representative LLMs across four domains, including more than 300 unfaithful instances with fine-grained causes and step-level evidence. We further conduct a systematic evaluation of eleven representative detection methods spanning counterfactual, logit-based, and LLM-as-judge paradigms, deriving empirical insights that clarify the strengths and weaknesses of existing approaches and reveal the increased challenges of detection in knowledge-intensive domains and with more advanced models. To the best of our knowledge, FaithCoT-Bench establishes the first comprehensive benchmark for instance-level CoT faithfulness, setting a solid basis for future research toward more interpretable and trustworthy reasoning in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地依赖于链式思考（CoT）提示以改进问题解决能力并提供看似透明的解释。然而，越来越多的证据表明，CoT往往未能忠实地表示底层的推理过程，这在其应用于高风险场景时引发了可靠性方面的担忧。尽管先前的研究集中在机制层面的分析上，展示了CoTs可能是不忠实地，但他们并未解决如何判断特定轨迹是否忠实于模型内部推理的实际挑战。为解决这一问题，我们引入了FaithCoT-Bench，一个统一的实例级CoT不忠实检测基准。我们的框架建立了一个严格的任务表述，将不忠实检测视为一个判别性决策问题，并提供了一个由超过1,000个轨迹组成的数据集FINE-CoT（实例级链式思考忠实性评估），这些轨迹来自四个代表性的大规模语言模型在四个领域的数据集中，其中包括超过300个具体的不忠实实例及其细粒度的原因和步骤级证据。我们进一步系统评估了涵盖反事实、logit基和大规模语言模型作为裁判的十一种代表性检测方法，从中得出了关于现有方法优势和劣势的经验性见解，并揭示了在知识密集领域和更先进的模型下检测面临的增加挑战。据我们所知，FaithCoT-Bench建立了一个首个全面的实例级CoT忠实性基准，为未来研究大型语言模型更加可解释和可信的推理奠定了坚实的基础。 

---
# A global log for medical AI 

**Title (ZH)**: 医疗AI全球日志 

**Authors**: Ayush Noori, Adam Rodman, Alan Karthikesalingam, Bilal A. Mateen, Christopher A. Longhurst, Daniel Yang, Dave deBronkart, Gauden Galea, Harold F. Wolf III, Jacob Waxman, Joshua C. Mandel, Juliana Rotich, Kenneth D. Mandl, Maryam Mustafa, Melissa Miles, Nigam H. Shah, Peter Lee, Robert Korom, Scott Mahoney, Seth Hain, Tien Yin Wong, Trevor Mundel, Vivek Natarajan, Noa Dagan, David A. Clifton, Ran D. Balicer, Isaac S. Kohane, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2510.04033)  

**Abstract**: Modern computer systems often rely on syslog, a simple, universal protocol that records every critical event across heterogeneous infrastructure. However, healthcare's rapidly growing clinical AI stack has no equivalent. As hospitals rush to pilot large language models and other AI-based clinical decision support tools, we still lack a standard way to record how, when, by whom, and for whom these AI models are used. Without that transparency and visibility, it is challenging to measure real-world performance and outcomes, detect adverse events, or correct bias or dataset drift. In the spirit of syslog, we introduce MedLog, a protocol for event-level logging of clinical AI. Any time an AI model is invoked to interact with a human, interface with another algorithm, or act independently, a MedLog record is created. This record consists of nine core fields: header, model, user, target, inputs, artifacts, outputs, outcomes, and feedback, providing a structured and consistent record of model activity. To encourage early adoption, especially in low-resource settings, and minimize the data footprint, MedLog supports risk-based sampling, lifecycle-aware retention policies, and write-behind caching; detailed traces for complex, agentic, or multi-stage workflows can also be captured under MedLog. MedLog can catalyze the development of new databases and software to store and analyze MedLog records. Realizing this vision would enable continuous surveillance, auditing, and iterative improvement of medical AI, laying the foundation for a new form of digital epidemiology. 

**Abstract (ZH)**: 现代计算机系统通常依赖syslog这一简单且通用的协议，用于记录异构基础设施中所有关键事件。然而，随着医疗保健领域临床AI堆栈的迅速增长，我们缺乏相应的等效工具。随着医院急忙进行大型语言模型和其他基于AI的临床决策支持工具的试点，我们仍然缺乏一种标准方法来记录这些AI模型何时、何地、以及为谁被使用。缺乏这种透明度和可见性使得难以衡量实际表现和结果、检测不良事件或纠正偏差或数据集漂移。怀着syslog的精神，我们介绍MedLog，这是一种用于记录临床AI事件的协议。每当AI模型被调用来与人类交互、与其他算法接口，或独立行动时，就会创建一个MedLog记录。该记录包括九个核心字段：头信息、模型、用户、目标、输入、产物、输出、结果和反馈，从而提供了一个结构化且一致的模型活动记录。为了鼓励早期采用，特别是在资源有限的环境中，MedLog支持基于风险的抽样、生命周期意识的保留策略，以及写后缓存；复杂的、代理驱动的或多阶段的工作流的详细追踪也可以在MedLog下捕获。MedLog可以促进新数据库和软件的开发，用于存储和分析MedLog记录。实现这一愿景将使持续监控、审计和迭代改进医疗AI成为可能，为新的数字流行病学奠定基础。 

---
# LLM-Based Data Science Agents: A Survey of Capabilities, Challenges, and Future Directions 

**Title (ZH)**: 基于LLM的数据科学代理：能力、挑战与未来方向 

**Authors**: Mizanur Rahman, Amran Bhuiyan, Mohammed Saidul Islam, Md Tahmid Rahman Laskar, Ridwan Mahbub, Ahmed Masry, Shafiq Joty, Enamul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2510.04023)  

**Abstract**: Recent advances in large language models (LLMs) have enabled a new class of AI agents that automate multiple stages of the data science workflow by integrating planning, tool use, and multimodal reasoning across text, code, tables, and visuals. This survey presents the first comprehensive, lifecycle-aligned taxonomy of data science agents, systematically analyzing and mapping forty-five systems onto the six stages of the end-to-end data science process: business understanding and data acquisition, exploratory analysis and visualization, feature engineering, model building and selection, interpretation and explanation, and deployment and monitoring. In addition to lifecycle coverage, we annotate each agent along five cross-cutting design dimensions: reasoning and planning style, modality integration, tool orchestration depth, learning and alignment methods, and trust, safety, and governance mechanisms. Beyond classification, we provide a critical synthesis of agent capabilities, highlight strengths and limitations at each stage, and review emerging benchmarks and evaluation practices. Our analysis identifies three key trends: most systems emphasize exploratory analysis, visualization, and modeling while neglecting business understanding, deployment, and monitoring; multimodal reasoning and tool orchestration remain unresolved challenges; and over 90% lack explicit trust and safety mechanisms. We conclude by outlining open challenges in alignment stability, explainability, governance, and robust evaluation frameworks, and propose future research directions to guide the development of robust, trustworthy, low-latency, transparent, and broadly accessible data science agents. 

**Abstract (ZH)**: 近期大规模语言模型的进展 enable了一类新的AI代理，它们通过整合规划、工具使用和跨文本、代码、表格和视觉的多模态推理，自动化数据科学工作流中的多个阶段。本文综述首次提供了数据科学代理的全面生命周期对齐分类法，并系统地分析和映射了 forty-five 个系统到端到端数据科学过程的六个阶段：业务理解与数据获取、探索性分析与可视化、特征工程、模型构建与选择、解释与说明、以及部署与监控。除生命周期覆盖外，我们还沿五个横贯设计维度为每个代理进行注解：推理和规划风格、模态集成、工具编排深度、学习和对齐方法，以及信任、安全和治理机制。除了分类外，我们还提供了代理能力的关键综合分析，在每个阶段突出 strengths 和局限性，并回顾新兴基准和评价实践。我们的分析识别了三项关键趋势：大多数系统侧重于探索性分析、可视化和建模，而忽视了业务理解、部署和监控；多模态推理和工具编排仍然是未解决的挑战；超过 90% 的系统缺乏明确的信任和安全机制。最后，我们概述了对齐稳定性、解释性、治理和稳健评估框架的开放挑战，并提出未来研究方向以指导稳健、可信、低延迟、透明和广泛可访问的数据科学代理的发展。 

---
# Zephyrus: An Agentic Framework for Weather Science 

**Title (ZH)**: Zephyrus: 一种天气科学的自主框架 

**Authors**: Sumanth Varambally, Marshall Fisher, Jas Thakker, Yiwei Chen, Zhirui Xia, Yasaman Jafari, Ruijia Niu, Manas Jain, Veeramakali Vignesh Manivannan, Zachary Novack, Luyu Han, Srikar Eranky, Salva Rühling Cachay, Taylor Berg-Kirkpatrick, Duncan Watson-Parris, Yi-An Ma, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04017)  

**Abstract**: Foundation models for weather science are pre-trained on vast amounts of structured numerical data and outperform traditional weather forecasting systems. However, these models lack language-based reasoning capabilities, limiting their utility in interactive scientific workflows. Large language models (LLMs) excel at understanding and generating text but cannot reason about high-dimensional meteorological datasets. We bridge this gap by building a novel agentic framework for weather science. Our framework includes a Python code-based environment for agents (ZephyrusWorld) to interact with weather data, featuring tools like an interface to WeatherBench 2 dataset, geoquerying for geographical masks from natural language, weather forecasting, and climate simulation capabilities. We design Zephyrus, a multi-turn LLM-based weather agent that iteratively analyzes weather datasets, observes results, and refines its approach through conversational feedback loops. We accompany the agent with a new benchmark, ZephyrusBench, with a scalable data generation pipeline that constructs diverse question-answer pairs across weather-related tasks, from basic lookups to advanced forecasting, extreme event detection, and counterfactual reasoning. Experiments on this benchmark demonstrate the strong performance of Zephyrus agents over text-only baselines, outperforming them by up to 35 percentage points in correctness. However, on harder tasks, Zephyrus performs similarly to text-only baselines, highlighting the challenging nature of our benchmark and suggesting promising directions for future work. 

**Abstract (ZH)**: 基于语言的推理能力桥梁：构建一个新的天气科学代理框架 

---
# What Shapes a Creative Machine Mind? Comprehensively Benchmarking Creativity in Foundation Models 

**Title (ZH)**: 塑造创意机器思维的因素：全面评估基础模型的创造力 

**Authors**: Zicong He, Boxuan Zhang, Weihao Liu, Ruixiang Tang, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04009)  

**Abstract**: The meteoric rise of foundation models (FMs) has expanded their capabilities far beyond conventional tasks. Creativity, long regarded as a hallmark of human intelligence and a driver of innovation, is now increasingly recognized as a critical dimension of machine intelligence in the era of generative FMs, complementing traditional measures of accuracy. However, existing evaluation frameworks for creativity remain fragmented, relying on ad hoc metrics not firmly grounded in established theories. To address this gap, we introduce C^2-Eval, a holistic benchmark for unified assessment of creativity in FMs. C^2-Eval distinguishes between two complementary forms of creativity: convergent creativity, where tasks admit constrained solutions (e.g., code generation), and divergent creativity, where tasks are open-ended (e.g., storytelling). It evaluates both dimensions using fine-grained criteria derived from social-science theory, focusing on Usefulness, Originality, and Surprise (U-O-S). Through extensive experiments on leading proprietary and open-source models, we analyze trade-offs in their creative capabilities. Our results highlight both the strengths and challenges of current FMs in pursuing a creative machine mind, showing that C^2-Eval is an effective lens for examining the evolving landscape of creative AI. 

**Abstract (ZH)**: 基础模型(FMs)的迅猛发展已使其能力远远超越了传统任务。在生成型FMs的时代，长期被视为人类智能标志和创新驱动力的创造力，现在越来越被视作机器智能的关键维度，补充了传统的准确性评估指标。然而，现有的创造力评估框架仍然支离破碎，依赖于尚未牢固建立在现有理论基础上的零碎指标。为填补这一空白，我们提出了C^2-Eval，这是一种全面的基准测试，用于统一评估FMs的创造力。C^2-Eval将创造力分为两种互补的形式：收敛创造力，其中任务允许受限的解决方案（例如，代码生成）；发散创造力，其中任务是开放式任务（例如，讲故事）。C^2-Eval 使用源自社会学理论的细粒度标准来评估这两个维度，重点关注有效性、原创性和惊奇感（U-O-S）。通过在领先的专业和开源模型上进行广泛实验，我们分析了其创造力能力之间的权衡。我们的结果突显了当前FMs在追求创造性机器心智方面的优缺点，证明C^2-Eval 是检验创造性AI不断演化的景观的有效视角。 

---
# Quantifying Risks in Multi-turn Conversation with Large Language Models 

**Title (ZH)**: 利用大型语言模型量化多轮对话中的风险 

**Authors**: Chengxiao Wang, Isha Chaudhary, Qian Hu, Weitong Ruan, Rahul Gupta, Gagandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.03969)  

**Abstract**: Large Language Models (LLMs) can produce catastrophic responses in conversational settings that pose serious risks to public safety and security. Existing evaluations often fail to fully reveal these vulnerabilities because they rely on fixed attack prompt sequences, lack statistical guarantees, and do not scale to the vast space of multi-turn conversations. In this work, we propose QRLLM, a novel, principled Certification framework for Catastrophic risks in multi-turn Conversation for LLMs that bounds the probability of an LLM generating catastrophic responses under multi-turn conversation distributions with statistical guarantees. We model multi-turn conversations as probability distributions over query sequences, represented by a Markov process on a query graph whose edges encode semantic similarity to capture realistic conversational flow, and quantify catastrophic risks using confidence intervals. We define several inexpensive and practical distributions: random node, graph path, adaptive with rejection. Our results demonstrate that these distributions can reveal substantial catastrophic risks in frontier models, with certified lower bounds as high as 70\% for the worst model, highlighting the urgent need for improved safety training strategies in frontier LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在对话场景中可能会产生灾难性的响应，这些响应对公共安全和安全构成了严重风险。现有的评估往往无法充分揭示这些漏洞，因为它们依赖于固定攻击提示序列，缺乏统计保证，也无法扩展到多轮对话的广阔空间。本文 propose 一种新的、原则性的认证框架 QRLLM，用于评估 LLMs 在多轮对话中的灾难性风险，该框架以统计保证的方式界定了在多轮对话分布下 LLM 生成灾难性响应的概率。我们将多轮对话建模为查询序列的概率分布，表示为查询图上的马尔可夫过程，其中边编码语义相似性以捕捉现实的对话流程，并使用置信区间量化灾难性风险。我们定义了几种低成本且实用的分布：随机节点、图路径、带拒绝的自适应。实验结果表明，这些分布可以揭示前沿模型中的重大灾难性风险，最坏情况下认证下限高达 70%，突显了改进前沿 LLM 安全训练策略的迫切需求。 

---
# Kantian-Utilitarian XAI: Meta-Explained 

**Title (ZH)**: 康德-功利主义XAI：元解释 

**Authors**: Zahra Atf, Peter R. Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2510.03892)  

**Abstract**: We present a gamified explainable AI (XAI) system for ethically aware consumer decision-making in the coffee domain. Each session comprises six rounds with three options per round. Two symbolic engines provide real-time reasons: a Kantian module flags rule violations (e.g., child labor, deforestation risk without shade certification, opaque supply chains, unsafe decaf), and a utilitarian module scores options via multi-criteria aggregation over normalized attributes (price, carbon, water, transparency, farmer income share, taste/freshness, packaging, convenience). A meta-explainer with a regret bound (0.2) highlights Kantian--utilitarian (mis)alignment and switches to a deontically clean, near-parity option when welfare loss is small. We release a structured configuration (attribute schema, certification map, weights, rule set), a policy trace for auditability, and an interactive UI. 

**Abstract (ZH)**: 我们提出了一种游戏化解释性人工智能（XAI）系统，用于咖啡领域中的伦理意识消费者决策。每个会话包括六轮，每轮有三个选项。两个符号引擎提供实时原因：一个康德模块标记规则违反（例如，童工、缺乏阴凉认证的采伐风险、不透明的供应链、不安全的脱咖啡因），一个功利主义模块通过归一化属性的多准则聚合为选项打分（价格、碳足迹、水资源使用、透明度、农民收入份额、风味/新鲜度、包装材料、便捷性）。一个元解释器带有后悔上限（0.2），突出康德主义-功利主义（不）一致，并在福利损失较小的情况下切换到一个ontology上洁净、接近对等的选择。我们发布了结构化的配置（属性模式、认证图谱、权重、规则集）、审计跟踪的政策策略以及交互式用户界面。 

---
# Rare Text Semantics Were Always There in Your Diffusion Transformer 

**Title (ZH)**: 稀有文本语义一直都存在于你的扩散变换器中 

**Authors**: Seil Kang, Woojung Han, Dayun Ju, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03886)  

**Abstract**: Starting from flow- and diffusion-based transformers, Multi-modal Diffusion Transformers (MM-DiTs) have reshaped text-to-vision generation, gaining acclaim for exceptional visual fidelity. As these models advance, users continually push the boundary with imaginative or rare prompts, which advanced models still falter in generating, since their concepts are often too scarce to leave a strong imprint during pre-training. In this paper, we propose a simple yet effective intervention that surfaces rare semantics inside MM-DiTs without additional training steps, data, denoising-time optimization, or reliance on external modules (e.g., large language models). In particular, the joint-attention mechanism intrinsic to MM-DiT sequentially updates text embeddings alongside image embeddings throughout transformer blocks. We find that by mathematically expanding representational basins around text token embeddings via variance scale-up before the joint-attention blocks, rare semantics clearly emerge in MM-DiT's outputs. Furthermore, our results generalize effectively across text-to-vision tasks, including text-to-image, text-to-video, and text-driven image editing. Our work invites generative models to reveal the semantics that users intend, once hidden yet ready to surface. 

**Abstract (ZH)**: 多模态扩散变换器中罕见语义的揭示：无需额外训练步骤的数据驱动视觉生成方法 

---
# Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation 

**Title (ZH)**: 空间CAPTCHA：生成性评估人类与机器的空间推理差异 

**Authors**: Arina Kharlamova, Bowei He, Chen Ma, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03863)  

**Abstract**: Online services rely on CAPTCHAs as a first line of defense against automated abuse, yet recent advances in multi-modal large language models (MLLMs) have eroded the effectiveness of conventional designs that focus on text recognition or 2D image understanding. To address this challenge, we present Spatial CAPTCHA, a novel human-verification framework that leverages fundamental differences in spatial reasoning between humans and MLLMs. Unlike existing CAPTCHAs which rely on low-level perception tasks that are vulnerable to modern AI, Spatial CAPTCHA generates dynamic questions requiring geometric reasoning, perspective-taking, occlusion handling, and mental rotation. These skills are intuitive for humans but difficult for state-of-the-art (SOTA) AI systems. The system employs a procedural generation pipeline with constraint-based difficulty control, automated correctness verification, and human-in-the-loop validation to ensure scalability, robustness, and adaptability. Evaluation on a corresponding benchmark, Spatial-CAPTCHA-Bench, demonstrates that humans vastly outperform 10 state-of-the-art MLLMs, with the best model achieving only 31.0% Pass@1 accuracy. Furthermore, we compare Spatial CAPTCHA with Google reCAPTCHA, which confirms its effectiveness as both a security mechanism and a diagnostic tool for spatial reasoning in AI. 

**Abstract (ZH)**: 基于空间推理的Spatial CAPTCHA：一种克服现代大模型挑战的人机验证框架 

---
# Adaptive and Explainable AI Agents for Anomaly Detection in Critical IoT Infrastructure using LLM-Enhanced Contextual Reasoning 

**Title (ZH)**: 基于增强上下文推理的LLM智能代理在关键物联网基础设施异常检测中的自适应与解释性方法 

**Authors**: Raghav Sharma, Manan Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2510.03859)  

**Abstract**: Ensuring that critical IoT systems function safely and smoothly depends a lot on finding anomalies quickly. As more complex systems, like smart healthcare, energy grids and industrial automation, appear, it is easier to see the shortcomings of older methods of detection. Monitoring failures usually happen in dynamic, high dimensional situations, especially when data is incomplete, messy or always evolving. Such limits point out the requirement for adaptive, intelligent systems that always improve and think. LLMs are now capable of significantly changing how context is understood and semantic inference is done across all types of data. This proposal suggests using an LLM supported contextual reasoning method along with XAI agents to improve how anomalies are found in significant IoT environments. To discover hidden patterns and notice inconsistencies in data streams, it uses attention methods, avoids dealing with details from every time step and uses memory buffers with meaning. Because no code AI stresses transparency and interpretability, people can check and accept the AI's decisions, helping ensure AI follows company policies. The two architectures are put together in a test that compares the results of the traditional model with those of the suggested LLM enhanced model. Important measures to check are the accuracy of detection, how much inaccurate information is included in the results, how clearly the findings can be read and how fast the system responds under different test situations. The metaheuristic is tested in simulations of real world smart grid and healthcare contexts to check its adaptability and reliability. From the study, we see that the new approach performs much better than most existing models in both accuracy and interpretation, so it could be a good fit for future anomaly detection tasks in IoT 

**Abstract (ZH)**: 确保关键物联网系统的安全平稳运行很大程度上依赖于快速发现异常。随着智能医疗、能源网和工业自动化等复杂系统的出现，传统检测方法的局限性更加明显。在动态、高维且数据不完整、杂乱或不断变化的情况下监测故障通常更为困难。这些局限性凸显了需要自适应且智能化的系统，这些系统能够不断改进和思考。大规模语言模型（LLM）现在能够显著改变对各种类型数据中上下文理解及语义推理的方式。本提案建议采用由LLM支持的上下文推理方法与可解释人工智能（XAI）代理相结合，以改进重要物联网环境中异常的检测。该方法利用注意力机制来发现隐藏的模式和注意到数据流中的不一致性，并在不处理每个时间步细节的情况下使用具有语义的内存缓冲区。由于无代码AI强调透明性和可解释性，人们可以检查和接受AI的决策，有助于确保AI遵循公司政策。两种架构在比较传统模型结果与建议的LLM增强模型结果的测试中结合。重要的评估指标包括检测的准确性、结果中包含的错误信息量、发现的清晰度以及系统在不同测试情况下的响应速度。该元算法在模拟实际智能电网和医疗保健环境的情况下进行测试，以检查其适应性和可靠性。研究结果显示，新方法在准确性和可解释性方面明显优于大多数现有模型，因此可能适合未来物联网中的异常检测任务。 

---
# Algorithm Generation via Creative Ideation 

**Title (ZH)**: 创意构想驱动的算法生成 

**Authors**: Ruiying Ma, Chieh-Jan Mike Liang, Yanjie Gao, Francis Y. Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03851)  

**Abstract**: Designing system algorithms remains challenging, where the discontinuous nature of the solution space often forces system engineers to rely on generic heuristics at the expense of performance. We study whether LLMs can practically drive algorithm generation, and find that they are biased towards well-known generic designs, rather than making the creative leaps needed to navigate the discontinuous solution space. To address this limitation, we introduce MetaMuse, a framework for creative ideation built on three self-reflection principles: (1) quantifying solution diversity and usefulness in measurable performance space, rather than abstract idea space, (2) steering ideation through external stimuli, rather than internal randomness, and (3) constructing executable solutions using waypoint reasoning, rather than free-form chain-of-thought. Extensive evaluation shows that MetaMuse can generate high-performing solutions for two critical problems at a global cloud provider: cache replacement (reducing cache misses by up to 35.76%) and online bin packing (reducing bin usage by up to 30.93%). 

**Abstract (ZH)**: 设计系统算法仍具有挑战性，其中解空间的非连续性质往往迫使系统工程师依赖通用启发式方法，而牺牲性能。我们研究了大语言模型是否能实际驱动算法生成，并发现它们偏向于生成已知的通用设计，而不是需要的创造性飞跃，以导航解空间的非连续性。为解决这一限制，我们提出了MetaMuse框架，该框架基于三个自我反思原则：（1）在可衡量的性能空间中量化解决方案的多样性和实用性，而不是在抽象的概念空间中；（2）通过外部刺激引导创意生成，而不是内部随机性；（3）使用航点推理构建可执行解决方案，而不是自由形式的推理链。广泛的评估显示，MetaMuse能够为一家全球云服务商的两个关键问题生成高性能的解决方案：缓存替换（降低缓存缺失率最多35.76%）和在线框包装（减少框使用率最多30.93%）。 

---
# Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade offs 

**Title (ZH)**: 代理系统中的小型语言模型：架构、能力及部署权衡综述 

**Authors**: Raghav Sharma, Manan Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2510.03847)  

**Abstract**: Small language models (SLMs; 1-12B params, sometimes up to 20B) are sufficient and often superior for agentic workloads where the objective is schema- and API-constrained accuracy rather than open-ended generation. We synthesize recent evidence across open and proprietary SLMs (Phi-4-Mini, Qwen-2.5-7B, Gemma-2-9B, Llama-3.2-1B/3B, Ministral-3B/8B, Apple on-device 3B, DeepSeek-R1-Distill) and connect it to modern evaluations (BFCL v3/v4, StableToolBench) and serving stacks (vLLM, SGLang, TensorRT-LLM) paired with guided decoding libraries (XGrammar, Outlines). We formalize SLM-default, LLM-fallback systems with uncertainty-aware routing and verifier cascades, and propose engineering metrics that reflect real production goals: cost per successful task (CPS), schema validity rate, executable call rate, p50/p95 latency, and energy per request. Guided decoding, strict JSON Schema outputs, and validator-first tool execution close much of the capability gap with larger models and often let SLMs match or surpass LLMs on tool use, function calling, and RAG at 10x-100x lower token cost with materially better latency and energy. We provide design patterns for agent stacks that prioritize SLMs: schema-first prompting, type-safe function registries, confidence scoring with verifier rollups, and lightweight adaptation via LoRA/QLoRA. We also delineate limits where fallback remains valuable (open-domain reasoning and some long-horizon planning). The result is a practical blueprint for building fast, inexpensive, and reliable agents that default to SLMs while preserving headroom with targeted LLM assistance.
Keywords: small language models, agents, function calling, structured outputs, JSON Schema, guided decoding, LoRA/QLoRA, routing, energy efficiency, edge inference 

**Abstract (ZH)**: 小语言模型（SLM；1-12B参数，有时最多20B）在目标为模式和API约束下的准确性而非开放式生成的代理工作负载中通常足够且更优。我们综合了近期公开和私有SLM（Phi-4-Mini、Qwen-2.5-7B、Gemma-2-9B、Llama-3.2-1B/3B、Ministral-3B/8B、Apple端上3B、DeepSeek-R1-Distill）的证据，并将其与现代评估（BFCL v3/v4、StableToolBench）和服务堆栈（vLLM、SGLang、TensorRT-LLM）及其指导解码库（XGrammar、Outlines）联系起来。我们形式化了SLM默认、LLM备份系统的不确定性感知路由和验证器级联，并提出了反映实际生产目标的工程度量标准：每项成功任务的成本（CPS）、方案有效性率、可执行调用率、p50/p95延迟时间和每次请求的能量消耗。指导解码、严格的JSON Schema输出和以验证器为主的工具执行显著缩小了与大模型之间的能力差距，通常让SLM在工具使用、函数调用和RAG方面以10到100倍更低的令牌成本实现与LLM相同的性能，同时还具有更佳的延迟和能量效率。我们提供了优先使用SLM的代理堆栈设计模式：以方案为主的提示、类型安全的函数注册表、验证器汇总的置信度评分以及通过LoRA/QLoRA实现的轻量级适应。我们还明确了SLM备份在某些领域仍然有价值的限制（如开放域推理和一些长期规划）。其结果是一个实用的蓝图，用于构建默认使用SLM且通过目标LLM辅助确保可靠性的快速、低成本代理。 

---
# The Hidden Game Problem 

**Title (ZH)**: 隐藏的比赛问题 

**Authors**: Gon Buzaglo, Noah Golowich, Elad Hazan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03845)  

**Abstract**: This paper investigates a class of games with large strategy spaces, motivated by challenges in AI alignment and language games. We introduce the hidden game problem, where for each player, an unknown subset of strategies consistently yields higher rewards compared to the rest. The central question is whether efficient regret minimization algorithms can be designed to discover and exploit such hidden structures, leading to equilibrium in these subgames while maintaining rationality in general. We answer this question affirmatively by developing a composition of regret minimization techniques that achieve optimal external and swap regret bounds. Our approach ensures rapid convergence to correlated equilibria in hidden subgames, leveraging the hidden game structure for improved computational efficiency. 

**Abstract (ZH)**: 基于AI对齐和语言游戏挑战的大型策略空间博弈研究：隐藏博弈问题及其高效 regrets 优化算法 

---
# GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time 

**Title (ZH)**: 引导采样：在推断时引导大规模语言模型趋向多样化候选解决方案 

**Authors**: Divij Handa, Mihir Parmar, Aswin RRV, Md Nayem Uddin, Hamid Palangi, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2510.03777)  

**Abstract**: Repeated Sampling (RS) is a simple inference-time algorithm that has been shown to improve model performance on complex tasks. Although it is an effective way of scaling inference time, it often struggles to generate diverse solution candidates, frequently relying on the same underlying approach to solve the problem and thus producing redundant samples. To address this limitation, we propose a new inference algorithm, GuidedSampling, which decouples the exploration and generation phases during inference, increasing diversity of generated candidate solutions. The exploration phase identifies multiple concepts that can be utilized to solve the problem, while the generation phase applies a specific concept to provide final solution candidates. We first define the theoretical bounds of GuidedSampling and then empirically demonstrate that it improves the performance of base model at pass@50 by on an average ~21.6% across various benchmarks compared to RS. Furthermore, models trained on trajectories of GuidedSampling exhibit substantial performance improvements at pass@5 by on an average ~9.7%, compared to models trained on traditional RS. Additionally, models trained with GuidedSampling increases the average number of concepts per instance (1.67 -> 3.03), yielding a diverse set of candidates than traditional RS. 

**Abstract (ZH)**: 指导采样：一种提高生成多样解方案的推理算法 

---
# OptAgent: Optimizing Query Rewriting for E-commerce via Multi-Agent Simulation 

**Title (ZH)**: OptAgent: 通过多agent模拟优化电子商务查询重写 

**Authors**: Divij Handa, David Blincoe, Orson Adams, Yinlin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03771)  

**Abstract**: Deploying capable and user-aligned LLM-based systems necessitates reliable evaluation. While LLMs excel in verifiable tasks like coding and mathematics, where gold-standard solutions are available, adoption remains challenging for subjective tasks that lack a single correct answer. E-commerce Query Rewriting (QR) is one such problem where determining whether a rewritten query properly captures the user intent is extremely difficult to figure out algorithmically. In this work, we introduce OptAgent, a novel framework that combines multi-agent simulations with genetic algorithms to verify and optimize queries for QR. Instead of relying on a static reward model or a single LLM judge, our approach uses multiple LLM-based agents, each acting as a simulated shopping customer, as a dynamic reward signal. The average of these agent-derived scores serves as an effective fitness function for an evolutionary algorithm that iteratively refines the user's initial query. We evaluate OptAgent on a dataset of 1000 real-world e-commerce queries in five different categories, and we observe an average improvement of 21.98% over the original user query and 3.36% over a Best-of-N LLM rewriting baseline. 

**Abstract (ZH)**: 基于LLM的系统部署需要可靠的评估。尽管LLM在如编程和数学等可验证任务上表现出色，但在缺乏单一正确答案的主观任务上，其采用仍然具有挑战性。电商查询重写（QR）便是这样一个问题，在算法上判断重写查询是否准确捕捉用户意图非常困难。本文我们提出了OptAgent，这是一种结合多智能体模拟与遗传算法的新框架，用于验证和优化QR查询。我们的方法不依赖于静态奖励模型或单一LLM评判者，而是使用多个LLM代理，每个代理模拟一名购物顾客作为动态奖励信号。这些代理评分的平均值作为进化算法的高效适应度函数，逐步优化用户的初始查询。我们在包含1000个真实电商查询的五个不同类别数据集上评估了OptAgent，并观察到与原始用户查询相比平均改进率为21.98%，与Best-of-N LLM重写基准相比平均改进率为3.36%。 

---
# Bridging the Gap Between Multimodal Foundation Models and World Models 

**Title (ZH)**: 跨模态基础模型与世界模型的桥梁 

**Authors**: Xuehai He  

**Link**: [PDF](https://arxiv.org/pdf/2510.03727)  

**Abstract**: Humans understand the world through the integration of multiple sensory modalities, enabling them to perceive, reason about, and imagine dynamic physical processes. Inspired by this capability, multimodal foundation models (MFMs) have emerged as powerful tools for multimodal understanding and generation. However, today's MFMs fall short of serving as effective world models. They lack the essential ability such as perform counterfactual reasoning, simulate dynamics, understand the spatiotemporal information, control generated visual outcomes, and perform multifaceted reasoning. We investigates what it takes to bridge the gap between multimodal foundation models and world models. We begin by improving the reasoning capabilities of MFMs through discriminative tasks and equipping MFMs with structured reasoning skills, such as causal inference, counterfactual thinking, and spatiotemporal reasoning, enabling them to go beyond surface correlations and understand deeper relationships within visual and textual data. Next, we explore generative capabilities of multimodal foundation models across both image and video modalities, introducing new frameworks for structured and controllable generation. Our approaches incorporate scene graphs, multimodal conditioning, and multimodal alignment strategies to guide the generation process, ensuring consistency with high-level semantics and fine-grained user intent. We further extend these techniques to controllable 4D generation, enabling interactive, editable, and morphable object synthesis over time and space. 

**Abstract (ZH)**: 多模态基础模型向世界模型转变所需的条件：增强推理能力和生成能力以理解深层关系并实现可控的多维生成 

---
# H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis 

**Title (ZH)**: H-DDx：层次化鉴别诊断评估框架 

**Authors**: Seungseop Lim, Gibaeg Kim, Hyunkyung Lee, Wooseok Han, Jean Seo, Jaehyo Yoo, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03700)  

**Abstract**: An accurate differential diagnosis (DDx) is essential for patient care, shaping therapeutic decisions and influencing outcomes. Recently, Large Language Models (LLMs) have emerged as promising tools to support this process by generating a DDx list from patient narratives. However, existing evaluations of LLMs in this domain primarily rely on flat metrics, such as Top-k accuracy, which fail to distinguish between clinically relevant near-misses and diagnostically distant errors. To mitigate this limitation, we introduce H-DDx, a hierarchical evaluation framework that better reflects clinical relevance. H-DDx leverages a retrieval and reranking pipeline to map free-text diagnoses to ICD-10 codes and applies a hierarchical metric that credits predictions closely related to the ground-truth diagnosis. In benchmarking 22 leading models, we show that conventional flat metrics underestimate performance by overlooking clinically meaningful outputs, with our results highlighting the strengths of domain-specialized open-source models. Furthermore, our framework enhances interpretability by revealing hierarchical error patterns, demonstrating that LLMs often correctly identify the broader clinical context even when the precise diagnosis is missed. 

**Abstract (ZH)**: 一种准确的鉴别诊断（DDx）对于患者护理至关重要，它影响治疗决策并影响结果。近年来，大型语言模型（LLMs）已 emerges as promising tools to support this process by generating a DDx list from patient narratives. However, existing evaluations of LLMs in this domain primarily rely on flat metrics, such as Top-k accuracy, which fail to distinguish between clinically relevant near-misses and diagnostically distant errors. To mitigate this limitation, we introduce H-DDx, a hierarchical evaluation framework that better reflects clinical relevance. H-DDx leverages a retrieval and reranking pipeline to map free-text diagnoses to ICD-10 codes and applies a hierarchical metric that credits predictions closely related to the ground-truth diagnosis. In benchmarking 22 leading models, we show that conventional flat metrics underestimate performance by overlooking clinically meaningful outputs, with our results highlighting the strengths of domain-specialized open-source models. Furthermore, our framework enhances interpretability by revealing hierarchical error patterns, demonstrating that LLMs often correctly identify the broader clinical context even when the precise diagnosis is missed. Hierarchical Evaluation Framework for Clinical Diagnosis: H-DDx 

---
# Mind the Goal: Data-Efficient Goal-Oriented Evaluation of Conversational Agents and Chatbots using Teacher Models 

**Title (ZH)**: Mind the Goal: 基于教师模型的数据高效目标导向对话代理和平铺对话机器人评估 

**Authors**: Deepak Babu Piskala, Sharlene Chen, Udita Patel, Parul Kalra, Rafael Castrillo  

**Link**: [PDF](https://arxiv.org/pdf/2510.03696)  

**Abstract**: Evaluating the quality of multi-turn chatbot interactions remains challenging, as most existing methods assess interactions at the turn level without addressing whether a user's overarching goal was fulfilled. A ``goal'' here refers to an information need or task, such as asking for policy information or applying for leave. We propose a comprehensive framework for goal-oriented evaluation of multi-agent systems (MAS), introducing the \textbf{Goal Success Rate (GSR)} to measure the percentage of fulfilled goals, and a \textbf{Root Cause of Failure (RCOF)} taxonomy to identify reasons for failure in multi-agent chatbots. Our method segments conversations by user goals and evaluates success using all relevant turns. We present a model-based evaluation system combining teacher LLMs, where domain experts define goals, set quality standards serving as a guidance for the LLMs. The LLMs use ``thinking tokens'' to produce interpretable rationales, enabling \textit{explainable}, \textit{data-efficient} evaluations. In an enterprise setting, we apply our framework to evaluate AIDA, a zero-to-one employee conversational agent system built as a ground-up multi-agent conversational agent, and observe GSR improvement from 63\% to 79\% over six months since its inception. Our framework is generic and offers actionable insights through a detailed defect taxonomy based on analysis of failure points in multi-agent chatbots, diagnosing overall success, identifying key failure modes, and informing system improvements. 

**Abstract (ZH)**: 评估多轮聊天机器人的交互质量仍然具有挑战性，大多数现有方法在转录级别进行评估，而不考虑用户的整体目标是否达成。我们提出了一种全面的框架，用于多代理系统（MAS）的目标导向评估，引入了目标成功率（GSR）来衡量目标达成的比例，并引入了失败根本原因（RCOF）分类法以识别多代理聊天机器人的失败原因。我们的方法通过用户目标拆分对话，并使用所有相关转录进行成功评估。我们提出了一种基于模型的评估系统，结合了教师级的大型语言模型（LLM），领域专家定义目标并设定质量标准作为LLM的指导。LLM使用“思考标记”生成可解释的推理过程，实现可解释、数据有效率的评估。在企业环境中，我们应用该框架评估从零构建的多代理对话系统AIDA，并观察到自成立以来六个月内目标成功率（GSR）从63%提高到79%。该框架具有通用性，并通过基于多代理聊天机器人失败点分析的详细缺陷分类法提供可操作的洞察，诊断整体成功情况，识别关键失败模式，并指导系统改进。 

---
# Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs 

**Title (ZH)**: 彩虹填充：缓解指令调优扩散LLM中的早期终止问题 

**Authors**: Bumjun Kim, Dongjae Jeon, Dueun Kim, Wonje Jeung, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2510.03680)  

**Abstract**: Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive models, offering flexible generation orders and strong performance on complex reasoning tasks. However, instruction-tuned dLLMs exhibit a critical vulnerability we term \texttt{<eos>} overflow: as allocated sequence length increases, responses paradoxically become shorter, collapsing into early termination or degenerating into streams of \texttt{<eos>} tokens. Although noticed in practice, this issue has not been systematically analyzed. We trace its root cause to the dual role of \texttt{<eos>} as both termination and padding, which concentrates probability mass on \texttt{<eos>} at later positions and propagates backward to trigger early termination. To address this, we introduce Rainbow Padding, a simple remedy that replaces repeated \texttt{<eos>} placeholders with a repeating cycle of distinct padding tokens, distributing probability mass and breaking \texttt{<eos>} dominance. Experiments show that Rainbow Padding substantially improves length robustness and output quality, with as few as seven padding tokens sufficient to prevent early termination. Moreover, the method integrates efficiently into existing instruction-tuned models: LoRA fine-tuning for a single epoch on minimal data yields significant improvements, making this solution highly practical. The code is publicly available at this https URL. 

**Abstract (ZH)**: 扩散大语言模型（dLLMs）已成为自回归模型的有 promise 的替代方案，提供灵活的生成顺序并在复杂推理任务上表现出色。然而，指令调优的 dLLMs 展现出一个我们称之为 \texttt{<eos>} 溢出的关键漏洞：随着分配的序列长度增加，响应 paradoxically 变得更短，导致早期终止或退化为 \texttt{<eos>} 令牌的流。尽管在实践中有所察觉，但这一问题尚未系统分析。我们将其根源追溯到 \texttt{<eos>} 的双重角色——既是终止符也是填充符，这在后期将概率质量集中在 \texttt{<eos>} 上，并向后传播以触发早期终止。为了解决这一问题，我们引入了彩虹填充（Rainbow Padding），这是一种简单的解决方案，用一组不同的填充令牌替换重复的 \texttt{<eos>} 占位符，分布概率质量并打破 \texttt{<eos>} 的主导地位。实验表明，彩虹填充显著提高了长度鲁棒性和输出质量，只需七个填充令牌即可防止早期终止。此外，该方法可以高效地集成到现有指令调优模型中：通过最少数据进行单个时期的 LoRA 微调即可实现显著改善，使该解决方案非常实用。相关代码已在此 https URL 公开。 

---
# MITS: Enhanced Tree Search Reasoning for LLMs via Pointwise Mutual Information 

**Title (ZH)**: MITS：通过点互信息增强的树搜索推理方法用于大语言模型 

**Authors**: Jiaxi Li, Yucheng Shi, Jin Lu, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03632)  

**Abstract**: Tree search has become as a representative framework for test-time reasoning with large language models (LLMs), exemplified by methods such as Tree-of-Thought and Monte Carlo Tree Search that explore multiple reasoning paths. However, it remains difficult to provide instant and reliable quantitative assessments of intermediate reasoning step quality, and extensive path exploration is computationally costly. To address this, we propose Mutual Information Tree Search (MITS), a novel framework that guides reasoning with information-theoretic principles. MITS introduces an effective scoring function based on pointwise mutual information (PMI), which enables step-wise evaluation of reasoning paths and search tree expansion via beam search without expensive look-ahead simulations, achieving superior reasoning performances while maintaining computational efficiency. The framework is complemented by an entropy-based dynamic sampling strategy that adaptively allocates computational resources to uncertain reasoning steps where exploration is most beneficial. For final prediction, MITS employs a weighted voting scheme that combines PMI scores with prediction consensus. Through comprehensive experiments on diverse reasoning benchmarks, MITS consistently surpasses baseline methods, establishing a principled and efficient framework for LLM reasoning. 

**Abstract (ZH)**: 基于互信息的树搜索（Mutual Information Tree Search，MITS）：一种信息论导向的大型语言模型推理框架 

---
# Cross-Modal Content Optimization for Steering Web Agent Preferences 

**Title (ZH)**: 跨模态内容优化以引导网络代理偏好 

**Authors**: Tanqiu Jiang, Min Bai, Nikolaos Pappas, Yanjun Qi, Sandesh Swamy  

**Link**: [PDF](https://arxiv.org/pdf/2510.03612)  

**Abstract**: Vision-language model (VLM)-based web agents increasingly power high-stakes selection tasks like content recommendation or product ranking by combining multimodal perception with preference reasoning. Recent studies reveal that these agents are vulnerable against attackers who can bias selection outcomes through preference manipulations using adversarial pop-ups, image perturbations, or content tweaks. Existing work, however, either assumes strong white-box access, with limited single-modal perturbations, or uses impractical settings. In this paper, we demonstrate, for the first time, that joint exploitation of visual and textual channels yields significantly more powerful preference manipulations under realistic attacker capabilities. We introduce Cross-Modal Preference Steering (CPS) that jointly optimizes imperceptible modifications to an item's visual and natural language descriptions, exploiting CLIP-transferable image perturbations and RLHF-induced linguistic biases to steer agent decisions. In contrast to prior studies that assume gradient access, or control over webpages, or agent memory, we adopt a realistic black-box threat setup: a non-privileged adversary can edit only their own listing's images and textual metadata, with no insight into the agent's model internals. We evaluate CPS on agents powered by state-of-the-art proprietary and open source VLMs including GPT-4.1, Qwen-2.5VL and Pixtral-Large on both movie selection and e-commerce tasks. Our results show that CPS is significantly more effective than leading baseline methods. For instance, our results show that CPS consistently outperforms baselines across all models while maintaining 70% lower detection rates, demonstrating both effectiveness and stealth. These findings highlight an urgent need for robust defenses as agentic systems play an increasingly consequential role in society. 

**Abstract (ZH)**: 基于视觉-语言模型的网页代理在结合多模态感知与偏好推理以执行高 stakes 选择任务（如内容推荐或产品排名）方面日益发挥作用。近期研究揭示，这些代理可能被攻击者操控，通过使用对抗弹窗、图像扭曲或内容调整来进行偏好操纵，从而偏转选择结果。现有工作要么假设强白盒访问权限，要么使用不切实际的设置。本文首次证明，在实际攻击者能力范围内，联合利用视觉与文本通道能产生更强大的偏好操纵效果。我们引入了跨模态偏好引导 (CPS)，通过联合优化项目视觉和自然语言描述的不可感知修改，利用 CLIP 可转移的图像扭曲和 RLHF 引导的语言偏见来引导代理决策。与此前假设梯度访问、或控制网页、或代理记忆的研究不同，本文采用了一个现实中的黑盒威胁模型：非特权的攻击者只能编辑自己的列表图片和文本元数据，而无法了解代理模型的内部机制。我们在电影选择和电商任务中，使用先进的专有和开源视觉-语言模型（包括 GPT-4.1、Qwen-2.5VL 和 Pixtral-Large）评估 CPS。实验结果表明，CPS 显著优于现有基线方法。例如，我们的结果表明，CPS 在所有模型中的表现均优于基线方法，同时检测率降低了 70%，显示了其效果和隐蔽性。这些发现凸显了在代理系统日益发挥关键作用的社会背景下，迫切需要强大的防御措施。 

---
# Understanding the Role of Training Data in Test-Time Scaling 

**Title (ZH)**: 理解训练数据在测试时扩展中的作用 

**Authors**: Adel Javanmard, Baharan Mirzasoleiman, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2510.03605)  

**Abstract**: Test-time scaling improves the reasoning capabilities of large language models (LLMs) by allocating extra compute to generate longer Chains-of-Thoughts (CoTs). This enables models to tackle more complex problem by breaking them down into additional steps, backtracking, and correcting mistakes. Despite its strong performance--demonstrated by OpenAI's o1 and DeepSeek R1, the conditions in the training data under which long CoTs emerge, and when such long CoTs improve the performance, remain unclear. In this paper, we study the performance of test-time scaling for transformers trained on an in-context weight prediction task for linear regression. Our analysis provides a theoretical explanation for several intriguing observations: First, at any fixed test error, increasing test-time compute allows us to reduce the number of in-context examples (context length) in training prompts. Second, if the skills required to solve a downstream task are not sufficiently present in the training data, increasing test-time compute can harm performance. Finally, we characterize task hardness via the smallest eigenvalue of its feature covariance matrix and show that training on a diverse, relevant, and hard set of tasks results in best performance for test-time scaling. We confirm our findings with experiments on large, nonlinear transformer architectures. 

**Abstract (ZH)**: Test-time缩放通过为生成更长的推理步骤分配额外计算资源，提高了大型语言模型的推理能力，使其能够通过分解问题、回溯和纠正错误来处理更复杂的问题。尽管其表现出色——OpenAI的o1和DeepSeek R1已经证明了这一点，并且长推理步骤在训练数据中的出现条件以及何时提高性能仍然不明确。在本文中，我们研究了在进行基于上下文权重预测的线性回归任务训练的变换器上进行测试时间缩放的性能。我们的分析为几个有趣的观察提供了一个理论解释：首先，在任何固定的测试误差下，增加测试时间计算可以减少训练提示中的上下文示例（上下文长度）。其次，如果用于解决下游任务所需的技能在训练数据中不够充分，增加测试时间计算可能会损害性能。最后，我们通过特征协方差矩阵的最小特征值表征任务难度，并表明通过训练一个多样化、相关且具有挑战性的任务集，可以在测试时间缩放中获得最佳性能。我们通过在大型、非线性变换器架构上的实验确认了这些发现。 

---
# OneFlow: Concurrent Mixed-Modal and Interleaved Generation with Edit Flows 

**Title (ZH)**: OneFlow: 共享内存下的混合模态并行生成与编辑流交错生成 

**Authors**: John Nguyen, Marton Havasi, Tariq Berrada, Luke Zettlemoyer, Ricky T. Q. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03506)  

**Abstract**: We present OneFlow, the first non-autoregressive multimodal model that enables variable-length and concurrent mixed-modal generation. Unlike autoregressive models that enforce rigid causal ordering between text and image generation, OneFlow combines an insertion-based Edit Flow for discrete text tokens with Flow Matching for image latents. OneFlow enables concurrent text-image synthesis with hierarchical sampling that prioritizes content over grammar. Through controlled experiments across model sizes from 1B to 8B, we demonstrate that OneFlow outperforms autoregressive baselines on both generation and understanding tasks while using up to 50% fewer training FLOPs. OneFlow surpasses both autoregressive and diffusion-based approaches while unlocking new capabilities for concurrent generation, iterative refinement, and natural reasoning-like generation. 

**Abstract (ZH)**: OneFlow：首个非自回归多模态模型，实现变长并发异模态生成 

---
# Towards Policy-Compliant Agents: Learning Efficient Guardrails For Policy Violation Detection 

**Title (ZH)**: 符合政策合规的代理：学习高效的政策违规检测边界 

**Authors**: Xiaofei Wen, Wenjie Jacky Mo, Yanan Xie, Peng Qi, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03485)  

**Abstract**: Autonomous web agents need to operate under externally imposed or human-specified policies while generating long-horizon trajectories. However, little work has examined whether these trajectories comply with such policies, or whether policy violations persist across different contexts such as domains (e.g., shopping or coding websites) and subdomains (e.g., product search and order management in shopping). To address this gap, we introduce PolicyGuardBench, a benchmark of about 60k examples for detecting policy violations in agent trajectories. From diverse agent runs, we generate a broad set of policies and create both within subdomain and cross subdomain pairings with violation labels. In addition to full-trajectory evaluation, PolicyGuardBench also includes a prefix-based violation detection task where models must anticipate policy violations from truncated trajectory prefixes rather than complete sequences. Using this dataset, we train PolicyGuard-4B, a lightweight guardrail model that delivers strong detection accuracy across all tasks while keeping inference efficient. Notably, PolicyGuard-4B generalizes across domains and preserves high accuracy on unseen settings. Together, PolicyGuardBench and PolicyGuard-4B provide the first comprehensive framework for studying policy compliance in web agent trajectories, and show that accurate and generalizable guardrails are feasible at small scales. 

**Abstract (ZH)**: 基于PolicyGuardBench的Web代理政策遵守综合框架 

---
# Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification 

**Title (ZH)**: LLM规划代理与形式方法的桥梁：计划验证案例研究 

**Authors**: Keshav Ramani, Vali Tawosi, Salwa Alamir, Daniel Borrajo  

**Link**: [PDF](https://arxiv.org/pdf/2510.03469)  

**Abstract**: We introduce a novel framework for evaluating the alignment between natural language plans and their expected behavior by converting them into Kripke structures and Linear Temporal Logic (LTL) using Large Language Models (LLMs) and performing model checking. We systematically evaluate this framework on a simplified version of the PlanBench plan verification dataset and report on metrics like Accuracy, Precision, Recall and F1 scores. Our experiments demonstrate that GPT-5 achieves excellent classification performance (F1 score of 96.3%) while almost always producing syntactically perfect formal representations that can act as guarantees. However, the synthesis of semantically perfect formal models remains an area for future exploration. 

**Abstract (ZH)**: 我们介绍了一种通过使用大型语言模型（LLMs）将自然语言计划及其预期行为转化为Kripke结构和线性时序逻辑（LTL），并进行模型检查的新框架，以评估两者之间的对齐。我们在简化版本的PlanBench计划验证数据集上系统地评估了该框架，并报告了准确性、精确度、召回率和F1分数等指标。实验结果表明，GPT-5在分类性能上表现出色（F1分数为96.3%），几乎总是生成语义正确且形式正确的表示，可以用作保证。然而，合成语义完美形式模型仍然是未来的研究方向。 

---
# A Qualitative Comparative Evaluation of Cognitive and Generative Theories 

**Title (ZH)**: 认知与生成理论的定性比较评价 

**Authors**: Paul S. Rosenbloom  

**Link**: [PDF](https://arxiv.org/pdf/2510.03453)  

**Abstract**: Evaluation is a critical activity associated with any theory. Yet this has proven to be an exceptionally challenging activity for theories based on cognitive architectures. For an overlapping set of reasons, evaluation can also be challenging for theories based on generative neural architectures. This dual challenge is approached here by leveraging a broad perspective on theory evaluation to yield a wide-ranging, albeit qualitative, comparison of whole-mind-oriented cognitive and generative architectures and the full systems that are based on these architectures. 

**Abstract (ZH)**: 基于认知架构和生成神经架构的理论评估是一项严峻挑战，本文通过广泛理论评估视角，提供了认知和生成架构及其所支持的完整系统之间的广泛且定性的比较。 

---
# ContraGen: A Multi-Agent Generation Framework for Enterprise Contradictions Detection 

**Title (ZH)**: ContraGen：一种企业矛盾检测的多Agent生成框架 

**Authors**: Ananya Mantravadi, Shivali Dalmia, Abhishek Mukherji, Nand Dave, Anudha Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2510.03418)  

**Abstract**: Retrieval-Augmented Generation (RAG) integrates LLMs with external sources, offering advanced capabilities for information access and decision-making. However, contradictions in retrieved evidence can result in inconsistent or untrustworthy outputs, which is especially problematic in enterprise settings where compliance, governance, and accountability are critical. Existing benchmarks for contradiction detection are limited to sentence-level analysis and do not capture the complexity of enterprise documents such as contracts, financial filings, compliance reports, or policy manuals. To address this limitation, we propose ContraGen, a contradiction-aware benchmark framework tailored to enterprise domain. The framework generates synthetic enterprise-style documents with embedded contradictions, enabling systematic evaluation of both intra-document and cross-document consistency. Automated contradiction mining is combined with human-in-the-loop validation to ensure high accuracy. Our contributions include generating realistic enterprise documents, modeling a taxonomy of contradiction types common in business processes, enabling controlled creation of self- and pairwise contradictions, developing a contradiction-aware retrieval evaluation pipeline and embedding human oversight to reflect domain-specific judgment complexity. This work establishes a foundation for more trustworthy and accountable RAG systems in enterprise information-seeking applications, where detecting and resolving contradictions is essential for reducing risk and ensuring compliance. 

**Abstract (ZH)**: Retrieval-Augmented Generation中的矛盾感知基准框架：ContraGen 

---
# Know Thyself? On the Incapability and Implications of AI Self-Recognition 

**Title (ZH)**: 知己？论AI自我识别的能力与影响 

**Authors**: Xiaoyan Bai, Aryan Shrivastava, Ari Holtzman, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03399)  

**Abstract**: Self-recognition is a crucial metacognitive capability for AI systems, relevant not only for psychological analysis but also for safety, particularly in evaluative scenarios. Motivated by contradictory interpretations of whether models possess self-recognition (Panickssery et al., 2024; Davidson et al., 2024), we introduce a systematic evaluation framework that can be easily applied and updated. Specifically, we measure how well 10 contemporary larger language models (LLMs) can identify their own generated text versus text from other models through two tasks: binary self-recognition and exact model prediction. Different from prior claims, our results reveal a consistent failure in self-recognition. Only 4 out of 10 models predict themselves as generators, and the performance is rarely above random chance. Additionally, models exhibit a strong bias toward predicting GPT and Claude families. We also provide the first evaluation of model awareness of their own and others' existence, as well as the reasoning behind their choices in self-recognition. We find that the model demonstrates some knowledge of its own existence and other models, but their reasoning reveals a hierarchical bias. They appear to assume that GPT, Claude, and occasionally Gemini are the top-tier models, often associating high-quality text with them. We conclude by discussing the implications of our findings on AI safety and future directions to develop appropriate AI self-awareness. 

**Abstract (ZH)**: AI系统的自我识别是一种关键的认知能力，对于心理分析和安全性，特别是在评估场景中尤为重要。鉴于对模型是否具备自我识别能力存在分歧的理解（Panickssery et al., 2024; Davidson et al., 2024），我们引入了一种易于应用和更新的系统评价框架。具体而言，我们通过两个任务——二元自我识别和精确模型预测——衡量10个现代大型语言模型识别自己生成的文本与他人模型生成的文本的能力。我们的结果揭示了一个一致的自我识别失败现象。只有4个模型认为自己是生成者，且性能很少高于随机水平。此外，模型显示出对预测GPT和Claude系列的强烈偏好。我们还首次评估了模型对其自身和他人存在的意识，以及他们在自我识别中的推理过程。我们发现模型展示了对其自身存在和其他模型的一些了解，但其推理显示出阶层偏见。它们似乎假设GPT、Claude和偶尔Gemini是顶级模型，并常将高质量文本与它们联系在一起。我们最后讨论了这些发现对AI安全的影响及未来发展方向，以培养适当的AI自我意识。 

---
# Refined Iterated Pareto Greedy for Energy-aware Hybrid Flowshop Scheduling with Blocking Constraints 

**Title (ZH)**: 改进的迭代帕累托贪婪算法用于带有阻塞约束的能量感知混合流水车间调度 

**Authors**: Ahmed Missaoui, Cemalettin Ozturk, Barry O'Sullivan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03377)  

**Abstract**: The scarcity of non-renewable energy sources, geopolitical problems in its supply, increasing prices, and the impact of climate change, force the global economy to develop more energy-efficient solutions for their operations. The Manufacturing sector is not excluded from this challenge as one of the largest consumers of energy. Energy-efficient scheduling is a method that attracts manufacturing companies to reduce their consumption as it can be quickly deployed and can show impact immediately. In this study, the hybrid flow shop scheduling problem with blocking constraint (BHFS) is investigated in which we seek to minimize the latest completion time (i.e. makespan) and overall energy consumption, a typical manufacturing setting across many industries from automotive to pharmaceutical. Energy consumption and the latest completion time of customer orders are usually conflicting objectives. Therefore, we first formulate the problem as a novel multi-objective mixed integer programming (MIP) model and propose an augmented epsilon-constraint method for finding the Pareto-optimal solutions. Also, an effective multi-objective metaheuristic algorithm. Refined Iterated Pareto Greedy (RIPG), is developed to solve large instances in reasonable time. Our proposed methods are benchmarked using small, medium, and large-size instances to evaluate their efficiency. Two well-known algorithms are adopted for comparing our novel approaches. The computational results show the effectiveness of our method. 

**Abstract (ZH)**: 非可再生能量资源稀缺、供给中的地缘政治问题、不断上升的价格以及气候影响迫使全球经济开发更高效的能量解决方案。制造部门并非不受这一挑战的影响，因其是最大的能量消费者之一。能量高效调度是一种吸引制造企业减少能耗的方法，因为它可以快速部署并能立即显示效果。在本研究中，我们探讨了带有阻塞约束的混合流水车间调度问题（BHFS），旨在最小化最晚完工时间和总体能耗，这是众多行业从汽车到制药的典型制造环境。能量消耗和客户订单的最晚完工时间通常是一个相互矛盾的目标。因此，我们首先将问题形式化为一种新型多目标混合整数规划（MIP）模型，并提出了一种扩展的ε约束方法来寻找帕累托最优解。同时，我们开发了一种有效的多目标元启发式算法——精炼迭代帕累托贪婪算法（RIPG），以在合理时间内解决大型实例。我们使用小、中、大型实例来评估所提出方法的效率，并采用两种知名算法进行比较。计算结果表明了我们方法的有效性。 

---
# WAREX: Web Agent Reliability Evaluation on Existing Benchmarks 

**Title (ZH)**: WAREX: Web代理可靠性评价基于现有基准 

**Authors**: Su Kara, Fazle Faisal, Suman Nath  

**Link**: [PDF](https://arxiv.org/pdf/2510.03285)  

**Abstract**: Recent advances in browser-based LLM agents have shown promise for automating tasks ranging from simple form filling to hotel booking or online shopping. Current benchmarks measure agent performance in controlled environments, such as containers or stable networks, where websites behave deterministically. However, in the real world, users access websites over networks and HTTPS connections that introduce instability from multiple sources: client-side, server-side issues or broader system failures. Moreover, live websites are prone to web attacks such Cross-Site Scripting, as well as general site modifications which can cause unexpected or malicious pop-ups or improper functionality. To address this gap, we present WAREX: Web Agent Reliability Evaluation on Existing Benchmarks. We measure the impact of WAREX across three popular benchmarks: WebArena, WebVoyager, and REAL. Our experiments show that introducing WAREX leads to significant drops in task success rates, highlighting the limited robustness of state-of-the-art agents. 

**Abstract (ZH)**: 基于浏览器的LLM代理 recent advances 在自动化从简单表单填写到酒店预订或在线购物等诸多任务方面展现了潜力。然而，当前基准测试在受控环境中评估代理性能，如容器或稳定网络，其中网站表现确定性。但在现实世界中，用户通过不稳定网络和HTTPS连接访问网站，引入了来自客户端、服务器端或更广泛系统故障等多种不稳定因素。此外，动态网站容易遭受跨站脚本攻击等网页攻击，以及一般站点修改，可能引起意外或恶意弹出窗口或不当功能。为解决这一问题，我们提出 WAREX：现有基准上的Web代理可靠性评估。我们在三种流行的基准测试（WebArena、WebVoyager和REAL）上测量了 WAREX 的影响。我们的实验表明，引入 WAREX 导致任务成功率显著下降，突显了现有先进代理的有限鲁棒性。 

---
# TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration 

**Title (ZH)**: TopInG: 基于持久性解析过滤的拓扑可解释图学习 

**Authors**: Cheng Xin, Fan Xu, Xin Ding, Jie Gao, Jiaxin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.05102)  

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable success across various scientific fields, yet their adoption in critical decision-making is often hindered by a lack of interpretability. Recently, intrinsically interpretable GNNs have been studied to provide insights into model predictions by identifying rationale substructures in graphs. However, existing methods face challenges when the underlying rationale subgraphs are complex and varied. In this work, we propose TopInG: Topologically Interpretable Graph Learning, a novel topological framework that leverages persistent homology to identify persistent rationale subgraphs. TopInG employs a rationale filtration learning approach to model an autoregressive generation process of rationale subgraphs, and introduces a self-adjusted topological constraint, termed topological discrepancy, to enforce a persistent topological distinction between rationale subgraphs and irrelevant counterparts. We provide theoretical guarantees that our loss function is uniquely optimized by the ground truth under specific conditions. Extensive experiments demonstrate TopInG's effectiveness in tackling key challenges, such as handling variform rationale subgraphs, balancing predictive performance with interpretability, and mitigating spurious correlations. Results show that our approach improves upon state-of-the-art methods on both predictive accuracy and interpretation quality. 

**Abstract (ZH)**: 拓扑可解释图学习：基于持久同调的TopInG 

---
# Paper2Video: Automatic Video Generation from Scientific Papers 

**Title (ZH)**: Paper2Video：从科学论文自动生成视频 

**Authors**: Zeyu Zhu, Kevin Qinghong Lin, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2510.05096)  

**Abstract**: Academic presentation videos have become an essential medium for research communication, yet producing them remains highly labor-intensive, often requiring hours of slide design, recording, and editing for a short 2 to 10 minutes video. Unlike natural video, presentation video generation involves distinctive challenges: inputs from research papers, dense multi-modal information (text, figures, tables), and the need to coordinate multiple aligned channels such as slides, subtitles, speech, and human talker. To address these challenges, we introduce PaperTalker, the first benchmark of 101 research papers paired with author-created presentation videos, slides, and speaker metadata. We further design four tailored evaluation metrics--Meta Similarity, PresentArena, PresentQuiz, and IP Memory--to measure how videos convey the paper's information to the audience. Building on this foundation, we propose PaperTalker, the first multi-agent framework for academic presentation video generation. It integrates slide generation with effective layout refinement by a novel effective tree search visual choice, cursor grounding, subtitling, speech synthesis, and talking-head rendering, while parallelizing slide-wise generation for efficiency. Experiments on Paper2Video demonstrate that the presentation videos produced by our approach are more faithful and informative than existing baselines, establishing a practical step toward automated and ready-to-use academic video generation. Our dataset, agent, and code are available at this https URL. 

**Abstract (ZH)**: 学术演示视频已成为研究交流的重要媒介，但制作它们仍然高度劳动密集型，常常需要数小时的设计、录制和编辑时间，才能制作出2至10分钟的视频。与自然视频生成不同，演示视频生成涉及独特的挑战：来自研究论文的输入、密集的多模态信息（文本、图表、表格）以及需要协调多个对齐的通道，如幻灯片、字幕、语音和人类发言人。为应对这些挑战，我们介绍了PaperTalker，这是一个包含101篇研究论文及其作者创建的演示视频、幻灯片和发言人元数据的第一套基准数据集。在此基础上，我们提出了PaperTalker，这是首个用于学术演示视频生成的多代理框架。该框架通过一种新颖有效的树搜索视觉选择、光标定位、字幕、语音合成和头部演讲渲染，将幻灯片生成与有效的布局精炼结合起来，同时按幻灯片并行生成以提高效率。实验结果表明，我们方法生成的演示视频比现有基线更为忠实地传达了论文信息，为自动化和即用型学术视频生成奠定了实用步骤。我们的数据集、代理和代码可在以下链接获取。 

---
# From Noisy Traces to Stable Gradients: Bias-Variance Optimized Preference Optimization for Aligning Large Reasoning Models 

**Title (ZH)**: 从 noisy 噪音轨迹到稳定梯度：偏差-方差优化的偏好优化方法以对齐大型推理模型 

**Authors**: Mingkang Zhu, Xi Chen, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.05095)  

**Abstract**: Large reasoning models (LRMs) generate intermediate reasoning traces before producing final answers, yielding strong gains on multi-step and mathematical tasks. Yet aligning LRMs with human preferences, a crucial prerequisite for model deployment, remains underexplored. The statistically correct objective for preference alignment requires marginalizing over reasoning traces, but this computation is intractable in practice. A common workaround optimizes a single sampled trajectory, which introduces substantial gradient variance from stochastic trace sampling. To address this challenge, we frame preference optimization for LRMs through the lens of the bias--variance trade-off and propose Bias--Variance Optimized Preference Optimization (BVPO), a simple, drop-in method that mixes two gradient estimators: a high-variance trace-based estimator and a low-variance empty-trace estimator obtained by disabling reasoning trace generation. Our theory shows that BVPO strictly reduces trace-induced variance for any nontrivial mixture, provides a closed-form choice of the mixing weight that minimizes mean-squared error relative to the true marginal gradient, and under standard smoothness and step-size conditions, tightens classical convergence bounds for stochastic gradient descent. Empirically, BVPO improves alignment over the best baseline by up to 7.8 points on AlpacaEval~2 and 6.8 points on Arena-Hard. Despite being trained only on general conversational data, BVPO also boosts reasoning performance for base models by up to 4.0 points on the average of six math reasoning benchmarks. These results identify variance from trace sampling as a key bottleneck and demonstrate that directly optimizing the bias--variance trade-off yields more stable training and stronger overall performance. 

**Abstract (ZH)**: 大型推理模型的偏置方差优化偏好优化（BVPO）方法 

---
# Learning to Interpret Weight Differences in Language Models 

**Title (ZH)**: 学习解释语言模型中的权重差异 

**Authors**: Avichal Goel, Yoon Kim, Nir Shavit, Tony T. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05092)  

**Abstract**: Finetuning (pretrained) language models is a standard approach for updating their internal parametric knowledge and specializing them to new tasks and domains. However, the corresponding model weight changes ("weight diffs") are not generally interpretable. While inspecting the finetuning dataset can give a sense of how the model might have changed, these datasets are often not publicly available or are too large to work with directly. Towards the goal of comprehensively understanding weight diffs in natural language, we introduce Diff Interpretation Tuning (DIT), a method that trains models to describe their own finetuning-induced modifications. Our approach uses synthetic, labeled weight diffs to train a DIT adapter, which can be applied to a compatible finetuned model to make it describe how it has changed. We demonstrate in two proof-of-concept settings (reporting hidden behaviors and summarizing finetuned knowledge) that our method enables models to describe their finetuning-induced modifications using accurate natural language descriptions. 

**Abstract (ZH)**: fine-tuning 预训练语言模型是更新其内部参数知识并将模型专门化于新任务和领域的一种标准方法。然而，相应的模型权重变化（“权重差异”）通常不具备解释性。虽然可以检查 fine-tuning 数据集来了解模型可能的变化，但这些数据集往往无法公开，或者太大而无法直接处理。为了全面理解自然语言中的权重差异，我们引入了差分解释微调（DIT）方法，该方法训练模型描述自己的 fine-tuning 引起的修改。我们的方法使用合成的、标记的权重差异来训练一个 DIT 调整器，该调整器可以应用于兼容的 fine-tuned 模型，使其能够描述自身的变化。我们在两种概念验证设置中（报告隐藏行为和总结 fine-tuning 知识）展示了我们的方法能够使模型使用准确的自然语言描述来描述 fine-tuning 引起的修改。 

---
# Finish First, Perfect Later: Test-Time Token-Level Cross-Validation for Diffusion Large Language Models 

**Title (ZH)**: 先完成，后优化：扩散大语言模型的测试时 token 级别交叉验证 

**Authors**: Runchu Tian, Junxia Cui, Xueqiang Xu, Feng Yao, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05090)  

**Abstract**: Diffusion large language models (dLLMs) have recently emerged as a promising alternative to autoregressive (AR) models, offering advantages such as accelerated parallel decoding and bidirectional context modeling. However, the vanilla decoding strategy in discrete dLLMs suffers from a critical limitation: once a token is accepted, it can no longer be revised in subsequent steps. As a result, early mistakes persist across iterations, harming both intermediate predictions and final output quality. To address this issue, we propose Tolerator (Token-Level Cross-Validation Refinement), a training-free decoding strategy that leverages cross-validation among predicted tokens. Unlike existing methods that follow a single progressive unmasking procedure, Tolerator introduces a two-stage process: (i) sequence fill-up and (ii) iterative refinement by remasking and decoding a subset of tokens while treating the remaining as context. This design enables previously accepted tokens to be reconsidered and corrected when necessary, leading to more reliable diffusion decoding outputs. We evaluate Tolerator on five standard benchmarks covering language understanding, code generation, and mathematics. Experiments show that our method achieves consistent improvements over the baselines under the same computational budget. These findings suggest that decoding algorithms are crucial to realizing the full potential of diffusion large language models. Code and data are publicly available. 

**Abstract (ZH)**: 基于交叉验证校准的Token级容忍策略：提高扩散大语言模型解码质量 

---
# TeachLM: Post-Training LLMs for Education Using Authentic Learning Data 

**Title (ZH)**: TeachLM：使用真实学习数据进行教育场景下LLM的后训练 

**Authors**: Janos Perczel, Jin Chow, Dorottya Demszky  

**Link**: [PDF](https://arxiv.org/pdf/2510.05087)  

**Abstract**: The promise of generative AI to revolutionize education is constrained by the pedagogical limits of large language models (LLMs). A major issue is the lack of access to high-quality training data that reflect the learning of actual students. Prompt engineering has emerged as a stopgap, but the ability of prompts to encode complex pedagogical strategies in rule-based natural language is inherently limited. To address this gap we introduce TeachLM - an LLM optimized for teaching through parameter-efficient fine-tuning of state-of-the-art models. TeachLM is trained on a dataset comprised of 100,000 hours of one-on-one, longitudinal student-tutor interactions maintained by Polygence, which underwent a rigorous anonymization process to protect privacy. We use parameter-efficient fine-tuning to develop an authentic student model that enables the generation of high-fidelity synthetic student-tutor dialogues. Building on this capability, we propose a novel multi-turn evaluation protocol that leverages synthetic dialogue generation to provide fast, scalable, and reproducible assessments of the dialogical capabilities of LLMs. Our evaluations demonstrate that fine-tuning on authentic learning data significantly improves conversational and pedagogical performance - doubling student talk time, improving questioning style, increasing dialogue turns by 50%, and greater personalization of instruction. 

**Abstract (ZH)**: 生成式AI在教育领域的潜力受制于大型语言模型（LLMs）的教育教学局限性：通过参数高效微调优化教学的大规模语言模型（TeachLM）以应对数据访问限制 

---
# SAEdit: Token-level control for continuous image editing via Sparse AutoEncoder 

**Title (ZH)**: SAEdit: 通过稀疏自编码器实现的TokenType级连续图像编辑 

**Authors**: Ronen Kamenetsky, Sara Dorfman, Daniel Garibi, Roni Paiss, Or Patashnik, Daniel Cohen-Or  

**Link**: [PDF](https://arxiv.org/pdf/2510.05081)  

**Abstract**: Large-scale text-to-image diffusion models have become the backbone of modern image editing, yet text prompts alone do not offer adequate control over the editing process. Two properties are especially desirable: disentanglement, where changing one attribute does not unintentionally alter others, and continuous control, where the strength of an edit can be smoothly adjusted. We introduce a method for disentangled and continuous editing through token-level manipulation of text embeddings. The edits are applied by manipulating the embeddings along carefully chosen directions, which control the strength of the target attribute. To identify such directions, we employ a Sparse Autoencoder (SAE), whose sparse latent space exposes semantically isolated dimensions. Our method operates directly on text embeddings without modifying the diffusion process, making it model agnostic and broadly applicable to various image synthesis backbones. Experiments show that it enables intuitive and efficient manipulations with continuous control across diverse attributes and domains. 

**Abstract (ZH)**: 一种通过 token 级别操纵文本嵌入实现的解耦和连续图像编辑方法 

---
# Slm-mux: Orchestrating small language models for reasoning 

**Title (ZH)**: Slm-mux： orchestrating 小型语言模型进行推理 

**Authors**: Chenyu Wang, Zishen Wan, Hao Kang, Emma Chen, Zhiqiang Xie, Tushar Krishna, Vijay Janapa Reddi, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.05077)  

**Abstract**: With the rapid development of language models, the number of small language models (SLMs) has grown significantly. Although they do not achieve state-of-the-art accuracy, they are more efficient and often excel at specific tasks. This raises a natural question: can multiple SLMs be orchestrated into a system where each contributes effectively, achieving higher accuracy than any individual model? Existing orchestration methods have primarily targeted frontier models (e.g., GPT-4) and perform suboptimally when applied to SLMs. To address this gap, we propose a three-stage approach for orchestrating SLMs. First, we introduce SLM-MUX, a multi-model architecture that effectively coordinates multiple SLMs. Building on this, we develop two optimization strategies: (i) a model selection search that identifies the most complementary SLMs from a given pool, and (ii) test-time scaling tailored to SLM-MUX. Our approach delivers strong results: Compared to existing orchestration methods, our approach achieves up to 13.4% improvement on MATH, 8.8% on GPQA, and 7.0% on GSM8K. With just two SLMS, SLM-MUX outperforms Qwen 2.5 72B on GPQA and GSM8K, and matches its performance on MATH. We further provide theoretical analyses to substantiate the advantages of our method. In summary, we demonstrate that SLMs can be effectively orchestrated into more accurate and efficient systems through the proposed approach. 

**Abstract (ZH)**: 随着语言模型的迅速发展，小型语言模型的数量显著增加。尽管它们未达到最先进的准确度，但更高效，并且常在特定任务上表现出色。这引发了一个自然问题：是否可以将多个小型语言模型组织成一个系统，其中每个模型都能有效贡献，从而实现比任何单一模型更高的准确度？现有的编排方法主要针对前沿模型（如GPT-4），应用于小型语言模型时效果不佳。为填补这一空白，我们提出了一种三阶段的小型语言模型编排方法。首先，我们引入SLM-MUX多模型架构，有效协调多个小型语言模型。在此基础上，我们开发了两种优化策略：（i）模型选择搜索，从给定池中识别出最互补的小型语言模型；（ii）针对SLM-MUX的测试时缩放。我们的方法表现优异：与现有编排方法相比，在MATH上提高了13.4%，在GPQA上提高了8.8%，在GSM8K上提高了7.0%。仅使用两个小型语言模型，SLM-MUX在GPQA和GSM8K上的表现优于Qwen 2.5 72B，在MATH上与其性能相当。我们进一步提供了理论分析以证实方法的优势。总之，我们证明了通过所提出的方法，小型语言模型可以被有效地组织成更准确、更高效的系统。 

---
# SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs 

**Title (ZH)**: SwiReasoning: 隐含与显式切换在帕累托占优推理中的作用 

**Authors**: Dachuan Shi, Abedelkadir Asi, Keying Li, Xiangchi Yuan, Leyan Pan, Wenke Lee, Wen Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.05069)  

**Abstract**: Recent work shows that, beyond discrete reasoning through explicit chain-of-thought steps, which are limited by the boundaries of natural languages, large language models (LLMs) can also reason continuously in latent space, allowing richer information per step and thereby improving token efficiency. Despite this promise, latent reasoning still faces two challenges, especially in training-free settings: 1) purely latent reasoning broadens the search distribution by maintaining multiple implicit paths, which diffuses probability mass, introduces noise, and impedes convergence to a single high-confidence solution, thereby hurting accuracy; and 2) overthinking persists even without explicit text, wasting tokens and degrading efficiency. To address these issues, we introduce SwiReasoning, a training-free framework for LLM reasoning which features two key innovations: 1) SwiReasoning dynamically switches between explicit and latent reasoning, guided by block-wise confidence estimated from entropy trends in next-token distributions, to balance exploration and exploitation and promote timely convergence. 2) By limiting the maximum number of thinking-block switches, SwiReasoning curbs overthinking and improves token efficiency across varying problem difficulties. On widely used mathematics and STEM benchmarks, SwiReasoning consistently improves average accuracy by 1.5%-2.8% across reasoning LLMs of different model families and scales. Furthermore, under constrained budgets, SwiReasoning improves average token efficiency by 56%-79%, with larger gains as budgets tighten. 

**Abstract (ZH)**: 最近的研究表明，除了通过显式的链式推理步骤进行离散推理，受限于自然语言的边界，大规模语言模型（LLMs）还可以在潜在空间中进行连续推理，从而在每步中包含更丰富的信息，提高标记效率。尽管如此，潜在推理仍然面临两大挑战，特别是在无需训练的情况下：1）纯粹的潜在推理通过维护多个隐式路径扩展了搜索分布，分散了概率质量，引入了噪声，阻碍了向单一高置信度解决方案的收敛，从而损害了准确性；2）即使没有显式的文本，过度推理也会持续存在，浪费标记并降低效率。为了解决这些问题，我们提出了一种无需训练的LLM推理框架SwiReasoning，该框架包含两个关键创新：1）SwiReasoning动态地在显式推理和潜在推理之间切换，由熵趋势估计的块级置信度引导，以平衡探索和利用，促进及时收敛。2）通过限制思考块切换的最大次数，SwiReasoning遏制了过度推理，并在不同问题难度下提高了标记效率。在广泛使用的数学和STEM基准测试中，SwiReasoning在不同模型家族和规模的推理LLM中始终将平均准确率提高了1.5%-2.8%。此外，在预算受限的情况下，SwiReasoning将平均标记效率提高了56%-79%，随着预算收紧，收益更大。 

---
# HybridFlow: Quantification of Aleatoric and Epistemic Uncertainty with a Single Hybrid Model 

**Title (ZH)**: HybridFlow：单一混合模型在定量评估 aleatoric 不确定性和 epistemic 不确定性方面的应用 

**Authors**: Peter Van Katwyk, Karianne J. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2510.05054)  

**Abstract**: Uncertainty quantification is critical for ensuring robustness in high-stakes machine learning applications. We introduce HybridFlow, a modular hybrid architecture that unifies the modeling of aleatoric and epistemic uncertainty by combining a Conditional Masked Autoregressive normalizing flow for estimating aleatoric uncertainty with a flexible probabilistic predictor for epistemic uncertainty. The framework supports integration with any probabilistic model class, allowing users to easily adapt HybridFlow to existing architectures without sacrificing predictive performance. HybridFlow improves upon previous uncertainty quantification frameworks across a range of regression tasks, such as depth estimation, a collection of regression benchmarks, and a scientific case study of ice sheet emulation. We also provide empirical results of the quantified uncertainty, showing that the uncertainty quantified by HybridFlow is calibrated and better aligns with model error than existing methods for quantifying aleatoric and epistemic uncertainty. HybridFlow addresses a key challenge in Bayesian deep learning, unifying aleatoric and epistemic uncertainty modeling in a single robust framework. 

**Abstract (ZH)**: 高风险机器学习应用中，不确定性量化对于确保鲁棒性至关重要。我们引入了HybridFlow，这是一种模块化的混合架构，通过结合条件掩码自回归归一化流来估算aleatoric不确定性以及灵活的概率预测器来估算epistemic不确定性，从而统一建模这两种不确定性。该框架支持与任何概率模型类的集成，使用户能够轻松适应现有架构而不牺牲预测性能。HybridFlow在深度估计等多种回归任务、一系列回归基准测试以及一个关于冰盖模拟的科学案例研究中，改进了之前的各种不确定性量化框架。此外，我们提供了量化的不确定性结果，表明HybridFlow量化出的不确定性是校准良好的，并且更好地与模型错误相一致，超过了现有方法的量化效果。HybridFlow解决了贝叶斯深度学习中的一个关键挑战，即在单个稳健的框架中统一建模aleatoric和epistemic不确定性。 

---
# Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts 

**Title (ZH)**: Diffusion LLMs中隐藏半自回归专家的测试时缩放 

**Authors**: Jihoon Lee, Hoyeon Moon, Kevin Zhai, Arun Kumar Chithanar, Anit Kumar Sahu, Soummya Kar, Chul Lee, Souradip Chakraborty, Amrit Singh Bedi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05040)  

**Abstract**: Diffusion-based large language models (dLLMs) are trained flexibly to model extreme dependence in the data distribution; however, how to best utilize this information at inference time remains an open problem. In this work, we uncover an interesting property of these models: dLLMs trained on textual data implicitly learn a mixture of semi-autoregressive experts, where different generation orders reveal different specialized behaviors. We show that committing to any single, fixed inference time schedule, a common practice, collapses performance by failing to leverage this latent ensemble. To address this, we introduce HEX (Hidden semiautoregressive EXperts for test-time scaling), a training-free inference method that ensembles across heterogeneous block schedules. By doing a majority vote over diverse block-sized generation paths, HEX robustly avoids failure modes associated with any single fixed schedule. On reasoning benchmarks such as GSM8K, it boosts accuracy by up to 3.56X (from 24.72% to 88.10%), outperforming top-K margin inference and specialized fine-tuned methods like GRPO, without additional training. HEX even yields significant gains on MATH benchmark from 16.40% to 40.00%, scientific reasoning on ARC-C from 54.18% to 87.80%, and TruthfulQA from 28.36% to 57.46%. Our results establish a new paradigm for test-time scaling in diffusion-based LLMs (dLLMs), revealing that the sequence in which masking is performed plays a critical role in determining performance during inference. 

**Abstract (ZH)**: 基于扩散的大语言模型（dLLMs）通过灵活训练来建模数据分布中的极端依赖关系；然而，如何在推断时最佳利用这些信息仍是一个开放问题。在这项工作中，我们揭示了这些模型的一个有趣性质：dLLMs在文本数据上训练时会隐式学习一个混合的半自回归专家集合，不同的生成顺序会揭示不同的专业化行为。我们展示，任何单一固定的推断时间调度（一种常见的做法）会导致性能下降，因为它未能充分利用这一潜在的集合。为此，我们提出了HEX（隐藏半自回归专家，用于测试时扩缩）——一种无需训练的推断方法，它在异质块调度之间进行集成。通过在多种不同的块大小生成路径之间进行多数投票，HEX 能稳健地避免任何单一固定调度所导致的失败模式。在诸如GSM8K的推理基准中，它将准确度提升了高达3.56倍（从24.72%提升到88.10%），超过了诸如Top-K边际推断和专门微调方法（如GRPO）的方法，而无需额外训练。甚至在MATH基准上，HEX也实现了显著提升（从16.40%提升到40.00%）、ARC-C上科学推理的提升（从54.18%提升到87.80%）和TruthfulQA上的提升（从28.36%提升到57.46%）。我们的结果确立了在扩散基于的大语言模型（dLLMs）中测试时扩缩的新范式，揭示了在推断过程中执行掩码的顺序起着关键作用。 

---
# Graph-Aware Diffusion for Signal Generation 

**Title (ZH)**: 图感知扩散信号生成 

**Authors**: Sergio Rozada, Vimal K. B., Andrea Cavallo, Antonio G. Marques, Hadi Jamali-Rad, Elvin Isufi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05036)  

**Abstract**: We study the problem of generating graph signals from unknown distributions defined over given graphs, relevant to domains such as recommender systems or sensor networks. Our approach builds on generative diffusion models, which are well established in vision and graph generation but remain underexplored for graph signals. Existing methods lack generality, either ignoring the graph structure in the forward process or designing graph-aware mechanisms tailored to specific domains. We adopt a forward process that incorporates the graph through the heat equation. Rather than relying on the standard formulation, we consider a time-warped coefficient to mitigate the exponential decay of the drift term, yielding a graph-aware generative diffusion model (GAD). We analyze its forward dynamics, proving convergence to a Gaussian Markov random field with covariance parametrized by the graph Laplacian, and interpret the backward dynamics as a sequence of graph-signal denoising problems. Finally, we demonstrate the advantages of GAD on synthetic data, real traffic speed measurements, and a temperature sensor network. 

**Abstract (ZH)**: 我们研究了从给定图上未知分布生成图信号的问题，这与推荐系统或传感器网络等领域相关。我们的方法基于生成性扩散模型，这类模型在视觉和图生成领域已有很好的应用，但在图信号生成方面仍处于探索阶段。现有方法缺乏通用性，要么在正向过程中忽视了图结构，要么为特定领域设计了图感知机制。我们采用通过热方程引入图的正向过程。不同于传统的公式化方法，我们考虑了一个时间扭曲系数来缓解漂移项的指数衰减，从而构建了一个图感知生成扩散模型（GAD）。我们分析了其正向动力学，证明其收敛到以图拉普拉斯矩阵参数化协方差的高斯马尔可夫随机场，并将反向动力学解释为一系列图信号去噪问题。最后，我们在合成数据、实际交通速度测量以及温度传感器网络上展示了GAD的优势。 

---
# Imperceptible Jailbreaking against Large Language Models 

**Title (ZH)**: 隐形破解大型语言模型 

**Authors**: Kuofeng Gao, Yiming Li, Chao Du, Xin Wang, Xingjun Ma, Shu-Tao Xia, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05025)  

**Abstract**: Jailbreaking attacks on the vision modality typically rely on imperceptible adversarial perturbations, whereas attacks on the textual modality are generally assumed to require visible modifications (e.g., non-semantic suffixes). In this paper, we introduce imperceptible jailbreaks that exploit a class of Unicode characters called variation selectors. By appending invisible variation selectors to malicious questions, the jailbreak prompts appear visually identical to original malicious questions on screen, while their tokenization is "secretly" altered. We propose a chain-of-search pipeline to generate such adversarial suffixes to induce harmful responses. Our experiments show that our imperceptible jailbreaks achieve high attack success rates against four aligned LLMs and generalize to prompt injection attacks, all without producing any visible modifications in the written prompt. Our code is available at this https URL. 

**Abstract (ZH)**: 视觉模态的越狱攻击通常依赖于不可感知的对抗扰动，而文本模态的攻击通常被认为需要可见的修改（例如，非语义后缀）。在这项工作中，我们介绍了利用变体选择符一类Unicode字符的不可感知越狱攻击。通过在恶意问题后添加不可见的变体选择符，越狱提示在屏幕上与原始恶意问题视觉上一致，但其分词被“秘密”改变。我们提出了一种搜索链管道来生成这种对抗后缀，以诱导有害响应。我们的实验表明，我们的不可感知越狱攻击在对抗四个对齐的LLM时成功率高，并能够在没有产生任何可见修改的提示注入攻击中泛化。我们的代码可在以下链接获取。 

---
# Rethinking Langevin Thompson Sampling from A Stochastic Approximation Perspective 

**Title (ZH)**: 从随机逼近视角重新审视 Langevin 汀伯特采样 

**Authors**: Weixin Wang, Haoyang Zheng, Guang Lin, Wei Deng, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05023)  

**Abstract**: Most existing approximate Thompson Sampling (TS) algorithms for multi-armed bandits use Stochastic Gradient Langevin Dynamics (SGLD) or its variants in each round to sample from the posterior, relaxing the need for conjugacy assumptions between priors and reward distributions in vanilla TS. However, they often require approximating a different posterior distribution in different round of the bandit problem. This requires tricky, round-specific tuning of hyperparameters such as dynamic learning rates, causing challenges in both theoretical analysis and practical implementation. To alleviate this non-stationarity, we introduce TS-SA, which incorporates stochastic approximation (SA) within the TS framework. In each round, TS-SA constructs a posterior approximation only using the most recent reward(s), performs a Langevin Monte Carlo (LMC) update, and applies an SA step to average noisy proposals over time. This can be interpreted as approximating a stationary posterior target throughout the entire algorithm, which further yields a fixed step-size, a unified convergence analysis framework, and improved posterior estimates through temporal averaging. We establish near-optimal regret bounds for TS-SA, with a simplified and more intuitive theoretical analysis enabled by interpreting the entire algorithm as a simulation of a stationary SGLD process. Our empirical results demonstrate that even a single-step Langevin update with certain warm-up outperforms existing methods substantially on bandit tasks. 

**Abstract (ZH)**: TS-SA: Incorporating Stochastic Approximation within the Thompson Sampling Framework 

---
# Large Language Models Achieve Gold Medal Performance at International Astronomy & Astrophysics Olympiad 

**Title (ZH)**: 大型语言模型在国际天文学与天体物理学奥林匹克竞赛中获得金牌成绩 

**Authors**: Lucas Carrit Delgado Pinheiro, Ziru Chen, Bruno Caixeta Piazza, Ness Shroff, Yingbin Liang, Yuan-Sen Ting, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.05016)  

**Abstract**: While task-specific demonstrations show early success in applying large language models (LLMs) to automate some astronomical research tasks, they only provide incomplete views of all necessary capabilities in solving astronomy problems, calling for more thorough understanding of LLMs' strengths and limitations. So far, existing benchmarks and evaluations focus on simple question-answering that primarily tests astronomical knowledge and fails to evaluate the complex reasoning required for real-world research in the discipline. Here, we address this gap by systematically benchmarking five state-of-the-art LLMs on the International Olympiad on Astronomy and Astrophysics (IOAA) exams, which are designed to examine deep conceptual understanding, multi-step derivations, and multimodal analysis. With average scores of 85.6% and 84.2%, Gemini 2.5 Pro and GPT-5 (the two top-performing models) not only achieve gold medal level performance but also rank in the top two among ~200-300 participants in all four IOAA theory exams evaluated (2022-2025). In comparison, results on the data analysis exams show more divergence. GPT-5 still excels in the exams with an 88.5% average score, ranking top 10 among the participants in the four most recent IOAAs, while other models' performances drop to 48-76%. Furthermore, our in-depth error analysis underscores conceptual reasoning, geometric reasoning, and spatial visualization (52-79% accuracy) as consistent weaknesses among all LLMs. Hence, although LLMs approach peak human performance in theory exams, critical gaps must be addressed before they can serve as autonomous research agents in astronomy. 

**Abstract (ZH)**: 尽管特定任务的示范在应用大规模语言模型（LLMs）自动化某些天文研究任务方面取得了早期成功，但它们仅提供了解决天文问题所需全部能力的不完整视角，需要更深入地理解LLMs的优势和限制。至今为止，现有的基准测试和评估主要集中在简单的问答任务上，主要是测试天文知识，未能评估学科实际研究所需的复杂推理。在这里，我们通过系统地在国际天文学与天体物理学奥林匹克（IOAA）考试上对五个最先进的LLMs进行基准测试，来填补这一空白，IOAA考试旨在检验深入的概念理解、多步推导和多模态分析。Gemini 2.5 Pro和GPT-5（两个高性能模型）以85.6%和84.2%的平均分不仅达到了金牌水平，还在所有四个IOAA理论考试中排名前两位（2022-2025年）。相比之下，数据分析考试的结果显示更多的分歧。GPT-5在考试中仍表现出色，平均分为88.5%，在最近四届IOAA考试中排名前10，而其他模型的表现下降到48-76%。此外，我们深入的错误分析表明，概念推理、几何推理和空间可视化（52-79%的准确性）是所有LLMs的一致薄弱环节。因此，尽管LLMs在理论考试中接近人类最佳表现，但在它们能够作为自主研究代理用于天文学之前，必须解决关键差距。 

---
# Resource-Efficient Fine-Tuning of LLaMA-3.2-3B for Medical Chain-of-Thought Reasoning 

**Title (ZH)**: LLaMA-3.2-3B的高效资源微调以用于医疗链式推理 

**Authors**: Imran Mansha  

**Link**: [PDF](https://arxiv.org/pdf/2510.05003)  

**Abstract**: Large Language Models (LLMs) such as GPT-4 and LLaMA have demonstrated remarkable reasoning abilities but require significant computational resources for fine-tuning. This paper presents a resource-efficient fine-tuning approach for LLaMA-3.2-3B to enhance medical chain-of-thought reasoning while operating under constrained GPU and memory settings. Using parameter-efficient tuning techniques such as LoRA and QLoRA, we adapt the base model on publicly available medical reasoning datasets. The model achieves improved reasoning coherence and factual accuracy while reducing memory usage by up to 60% compared to standard full fine-tuning. Experimental evaluation demonstrates that lightweight adaptations can retain strong reasoning capability in medical question-answering tasks. This work highlights practical strategies for deploying LLMs in low-resource research environments and provides insights into balancing efficiency and domain specialization for medical AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT-4和LLaMA展现了出色的推理能力，但需要大量计算资源进行微调。本文提出了在受限的GPU和内存环境下，一种高效微调LLaMA-3.2-3B的方法，以增强医学链式推理能力。利用参数高效微调技术如LoRA和QLoRA，我们适应了公开的医学推理数据集。与标准全面微调相比，该模型在减少60%内存使用的同时，实现了更好的推理连贯性和事实准确性。实验评估表明，轻量级适应可以在医学问答任务中保留强大的推理能力。该工作突显了在低资源研究环境中部署LLMs的实用策略，并提供了平衡效率和领域专业化对于医学AI系统见解。 

---
# Bridging Text and Video Generation: A Survey 

**Title (ZH)**: 文本生成与视频生成的桥梁：一个综述 

**Authors**: Nilay Kumar, Priyansh Bhandari, G. Maragatham  

**Link**: [PDF](https://arxiv.org/pdf/2510.04999)  

**Abstract**: Text-to-video (T2V) generation technology holds potential to transform multiple domains such as education, marketing, entertainment, and assistive technologies for individuals with visual or reading comprehension challenges, by creating coherent visual content from natural language prompts. From its inception, the field has advanced from adversarial models to diffusion-based models, yielding higher-fidelity, temporally consistent outputs. Yet challenges persist, such as alignment, long-range coherence, and computational efficiency. Addressing this evolving landscape, we present a comprehensive survey of text-to-video generative models, tracing their development from early GANs and VAEs to hybrid Diffusion-Transformer (DiT) architectures, detailing how these models work, what limitations they addressed in their predecessors, and why shifts toward new architectural paradigms were necessary to overcome challenges in quality, coherence, and control. We provide a systematic account of the datasets, which the surveyed text-to-video models were trained and evaluated on, and, to support reproducibility and assess the accessibility of training such models, we detail their training configurations, including their hardware specifications, GPU counts, batch sizes, learning rates, optimizers, epochs, and other key hyperparameters. Further, we outline the evaluation metrics commonly used for evaluating such models and present their performance across standard benchmarks, while also discussing the limitations of these metrics and the emerging shift toward more holistic, perception-aligned evaluation strategies. Finally, drawing from our analysis, we outline the current open challenges and propose a few promising future directions, laying out a perspective for future researchers to explore and build upon in advancing T2V research and applications. 

**Abstract (ZH)**: 文本到视频生成技术（T2V）有可能通过从自然语言提示生成连贯的视觉内容来改变教育、营销、娱乐和视觉或阅读理解障碍个体辅助技术等多个领域。从其起步至今，该领域已从对抗模型发展到基于扩散的模型，产生了更高保真度、时序一致的输出。然而，仍然存在对齐、长距离连贯性和计算效率等方面的挑战。为了应对这一不断发展的情景，我们提供了一篇全面的文本到视频生成模型综述，追溯了从早期的GANs和VAEs到混合扩散-变换器（DiT）架构的发展过程，详细说明了这些模型的工作原理、它们在前辈模型中解决的限制，以及为何转向新的架构范式是必要的，以克服质量、连贯性和控制方面的挑战。我们系统地概述了所研究的文本到视频模型的训练和评估数据集，并详细描述了它们的训练配置，包括硬件规格、GPU数量、批量大小、学习率、优化器、 epoch 和其他关键超参数，以支持再现性和评估训练此类模型的便利性。进一步地，我们概述了常用于评估此类模型的评价指标，并展示了它们在标准基准上的表现，同时也讨论了这些指标的局限性以及向更加整体、感知对齐的评价策略的新兴转变。最后，基于我们的分析，我们指出现有的开放挑战，并提出了一些建设性的未来方向，为未来的研究人员提供了探索和推动文本到视频研究和应用的视角。 

---
# AutoEmpirical: LLM-Based Automated Research for Empirical Software Fault Analysis 

**Title (ZH)**: AutoEmpirical: 基于LLM的自动化 empirical 软件故障分析研究 

**Authors**: Jiongchi Yu, Weipeng Jiang, Xiaoyu Zhang, Qiang Hu, Xiaofei Xie, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04997)  

**Abstract**: Understanding software faults is essential for empirical research in software development and maintenance. However, traditional fault analysis, while valuable, typically involves multiple expert-driven steps such as collecting potential faults, filtering, and manual investigation. These processes are both labor-intensive and time-consuming, creating bottlenecks that hinder large-scale fault studies in complex yet critical software systems and slow the pace of iterative empirical research.
In this paper, we decompose the process of empirical software fault study into three key phases: (1) research objective definition, (2) data preparation, and (3) fault analysis, and we conduct an initial exploration study of applying Large Language Models (LLMs) for fault analysis of open-source software. Specifically, we perform the evaluation on 3,829 software faults drawn from a high-quality empirical study. Our results show that LLMs can substantially improve efficiency in fault analysis, with an average processing time of about two hours, compared to the weeks of manual effort typically required. We conclude by outlining a detailed research plan that highlights both the potential of LLMs for advancing empirical fault studies and the open challenges that required be addressed to achieve fully automated, end-to-end software fault analysis. 

**Abstract (ZH)**: 理解软件故障对于软件开发和维护中的实证研究至关重要。然而，传统的故障分析虽有价值，但也通常涉及多个由专家驱动的步骤，如收集潜在故障、过滤和手动调查。这些过程既耗费人力又耗时，成为瓶颈，阻碍了对复杂而关键软件系统的大型规模故障研究，并减慢了迭代实证研究的步伐。

在本文中，我们将实证软件故障研究的过程分解为三个关键阶段：(1) 研究目标定义，(2) 数据准备，以及(3) 故障分析。我们开展了一项初步探索性研究，探讨大规模语言模型（LLMs）在开源软件故障分析中的应用。具体而言，我们在一个高质量的实证研究中抽取了3,829个软件故障进行评估。结果显示，LLMs能够显著提高故障分析的效率，平均处理时间为大约两小时，相较于通常所需的数周手动努力。最后，我们提出了一个详细的研计划，明确了LLMs在推进实证故障研究中的潜力以及实现全流程自动化的开放挑战。 

---
# Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training 

**Title (ZH)**: Reinforce-Ada：一种针对Reinforce风格大语言模型训练的自适应采样框架 

**Authors**: Wei Xiong, Chenlu Ye, Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, Jiang Bian, Nan Jiang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04996)  

**Abstract**: Reinforcement learning applied to large language models (LLMs) for reasoning tasks is often bottlenecked by unstable gradient estimates due to fixed and uniform sampling of responses across prompts. Prior work such as GVM-RAFT addresses this by dynamically allocating inference budget per prompt to minimize stochastic gradient variance under a budget constraint. Inspired by this insight, we propose Reinforce-Ada, an adaptive sampling framework for online RL post-training of LLMs that continuously reallocates sampling effort to the prompts with the greatest uncertainty or learning potential. Unlike conventional two-stage allocation methods, Reinforce-Ada interleaves estimation and sampling in an online successive elimination process, and automatically stops sampling for a prompt once sufficient signal is collected. To stabilize updates, we form fixed-size groups with enforced reward diversity and compute advantage baselines using global statistics aggregated over the adaptive sampling phase. Empirical results across multiple model architectures and reasoning benchmarks show that Reinforce-Ada accelerates convergence and improves final performance compared to GRPO, especially when using the balanced sampling variant. Our work highlights the central role of variance-aware, adaptive data curation in enabling efficient and reliable reinforcement learning for reasoning-capable LLMs. Code is available at this https URL. 

**Abstract (ZH)**: 基于强化学习的大型语言模型（LLMs）在推理任务中的适应性采样框架：Reinforce-Ada 

---
# AWARE, Beyond Sentence Boundaries: A Contextual Transformer Framework for Identifying Cultural Capital in STEM Narratives 

**Title (ZH)**: AWARE，超越句子边界：一种在STEM叙事中识别文化资本的上下文Transformer框架 

**Authors**: Khalid Mehtab Khan, Anagha Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2510.04983)  

**Abstract**: Identifying cultural capital (CC) themes in student reflections can offer valuable insights that help foster equitable learning environments in classrooms. However, themes such as aspirational goals or family support are often woven into narratives, rather than appearing as direct keywords. This makes them difficult to detect for standard NLP models that process sentences in isolation. The core challenge stems from a lack of awareness, as standard models are pre-trained on general corpora, leaving them blind to the domain-specific language and narrative context inherent to the data. To address this, we introduce AWARE, a framework that systematically attempts to improve a transformer model's awareness for this nuanced task. AWARE has three core components: 1) Domain Awareness, adapting the model's vocabulary to the linguistic style of student reflections; 2) Context Awareness, generating sentence embeddings that are aware of the full essay context; and 3) Class Overlap Awareness, employing a multi-label strategy to recognize the coexistence of themes in a single sentence. Our results show that by making the model explicitly aware of the properties of the input, AWARE outperforms a strong baseline by 2.1 percentage points in Macro-F1 and shows considerable improvements across all themes. This work provides a robust and generalizable methodology for any text classification task in which meaning depends on the context of the narrative. 

**Abstract (ZH)**: 识别学生反思中的文化资本主题可以为促进公平的学习环境提供宝贵的见解。然而，诸如抱负目标或家庭支持等主题往往融入叙事中，而非直接作为关键词出现。这使得标准的自然语言处理模型难以检测。核心挑战在于模型缺乏领域意识，因为标准模型预训练于一般语料库，无法识别领域特定的语言和叙事背景。为此，我们提出了AWARE框架，系统性地提高模型对这一细腻任务的意识。AWARE有三个核心组成部分：1）领域意识，调整模型词汇以适应学生反思的语言风格；2）上下文意识，生成感知全文背景的句子嵌入；3）类别重叠意识，采用多标签策略识别单句中存在的多个主题。我们的结果显示，通过使模型明确意识到输入的特性，AWARE在宏F1分数上比强 baseline 高出2.1个百分点，并且在所有主题上均表现出显著改进。这项工作提供了任何依赖叙事上下文的文本分类任务的稳健且可泛化的方法论。 

---
# Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning 

**Title (ZH)**: 拥抱离散搜索：一种因果结构学习的合理方法 

**Authors**: Marcel Wienöbst, Leonard Henckel, Sebastian Weichwald  

**Link**: [PDF](https://arxiv.org/pdf/2510.04970)  

**Abstract**: We present FLOP (Fast Learning of Order and Parents), a score-based causal discovery algorithm for linear models. It pairs fast parent selection with iterative Cholesky-based score updates, cutting run-times over prior algorithms. This makes it feasible to fully embrace discrete search, enabling iterated local search with principled order initialization to find graphs with scores at or close to the global optimum. The resulting structures are highly accurate across benchmarks, with near-perfect recovery in standard settings. This performance calls for revisiting discrete search over graphs as a reasonable approach to causal discovery. 

**Abstract (ZH)**: 基于评分的快速顺序和祖先学习：一种线性模型的因果发现算法 

---
# ActiveMark: on watermarking of visual foundation models via massive activations 

**Title (ZH)**: ActiveMark：视觉基础模型的激活 watermarking 方法 

**Authors**: Anna Chistyakova, Mikhail Pautov  

**Link**: [PDF](https://arxiv.org/pdf/2510.04966)  

**Abstract**: Being trained on large and vast datasets, visual foundation models (VFMs) can be fine-tuned for diverse downstream tasks, achieving remarkable performance and efficiency in various computer vision applications. The high computation cost of data collection and training motivates the owners of some VFMs to distribute them alongside the license to protect their intellectual property rights. However, a dishonest user of the protected model's copy may illegally redistribute it, for example, to make a profit. As a consequence, the development of reliable ownership verification tools is of great importance today, since such methods can be used to differentiate between a redistributed copy of the protected model and an independent model. In this paper, we propose an approach to ownership verification of visual foundation models by fine-tuning a small set of expressive layers of a VFM along with a small encoder-decoder network to embed digital watermarks into an internal representation of a hold-out set of input images. Importantly, the watermarks embedded remain detectable in the functional copies of the protected model, obtained, for example, by fine-tuning the VFM for a particular downstream task. Theoretically and experimentally, we demonstrate that the proposed method yields a low probability of false detection of a non-watermarked model and a low probability of false misdetection of a watermarked model. 

**Abstract (ZH)**: 基于大规模数据集训练的视觉基础模型的所有权验证方法：通过微调一小组表达层和小型编码-解码网络将数字水印嵌入保留集输入图像的内部表示以实现模型所有权验证 

---
# MuFFIN: Multifaceted Pronunciation Feedback Model with Interactive Hierarchical Neural Modeling 

**Title (ZH)**: 多面发音反馈模型：交互式分层神经建模 

**Authors**: Bi-Cheng Yan, Ming-Kang Tsai, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04956)  

**Abstract**: Computer-assisted pronunciation training (CAPT) manages to facilitate second-language (L2) learners to practice pronunciation skills by offering timely and instructive feedback. To examine pronunciation proficiency from multiple facets, existing methods for CAPT broadly fall into two categories: mispronunciation detection and diagnosis (MDD) as well as automatic pronunciation assessment (APA). The former aims to pinpoint phonetic pronunciation errors and provide diagnostic feedback, while the latter seeks instead to quantify pronunciation proficiency pertaining to various aspects. Despite the natural complementarity between MDD and APA, researchers and practitioners, however, often treat them as independent tasks with disparate modeling paradigms. In light of this, we in this paper first introduce MuFFIN, a Multi-Faceted pronunciation Feedback model with an Interactive hierarchical Neural architecture, to jointly address the tasks of MDD and APA. To better capture the nuanced distinctions between phonemes in the feature space, a novel phoneme-contrastive ordinal regularization mechanism is then put forward to optimize the proposed model to generate more phoneme-discriminative features while factoring in the ordinality of the aspect scores. In addition, to address the intricate data imbalance problem in MDD, we design a simple yet effective training objective, which is specifically tailored to perturb the outputs of a phoneme classifier with the phoneme-specific variations, so as to better render the distribution of predicted phonemes meanwhile considering their mispronunciation characteristics. A series of experiments conducted on the Speechocean762 benchmark dataset demonstrates the efficacy of our method in relation to several cutting-edge baselines, showing state-of-the-art performance on both the APA and MDD tasks. 

**Abstract (ZH)**: 计算机辅助发音训练中的多维度反馈模型：MuFFIN及其在发音错误检测与诊断及自动发音评估中的应用 

---
# Feasibility-Aware Decision-Focused Learning for Predicting Parameters in the Constraints 

**Title (ZH)**: 面向可行性的决策聚焦学习：预测约束条件下的参数 

**Authors**: Jayanta Mandi, Marianne Defresne, Senne Berden, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2510.04951)  

**Abstract**: When some parameters of a constrained optimization problem (COP) are uncertain, this gives rise to a predict-then-optimize (PtO) problem, comprising two stages -- the prediction of the unknown parameters from contextual information and the subsequent optimization using those predicted parameters. Decision-focused learning (DFL) implements the first stage by training a machine learning (ML) model to optimize the quality of the decisions made using the predicted parameters. When parameters in the constraints of a COP are predicted, the predicted parameters can lead to infeasible solutions. Therefore, it is important to simultaneously manage both feasibility and decision quality. We develop a DFL framework for predicting constraint parameters in a generic COP. While prior works typically assume that the underlying optimization problem is a linear program (LP) or integer linear program (ILP), our approach makes no such assumption. We derive two novel loss functions based on maximum likelihood estimation (MLE): the first one penalizes infeasibility (by penalizing when the predicted parameters lead to infeasible solutions), and the second one penalizes suboptimal decisions (by penalizing when the true optimal solution is infeasible under the predicted parameters). We introduce a single tunable parameter to form a weighted average of the two losses, allowing decision-makers to balance suboptimality and feasibility. We experimentally demonstrate that adjusting this parameter provides a decision-maker the control over the trade-off between the two. Moreover, across several COP instances, we find that for a single value of the tunable parameter, our method matches the performance of the existing baselines on suboptimality and feasibility. 

**Abstract (ZH)**: 在约束优化问题参数不确定时的预测-优化问题及决策聚焦学习框架 

---
# Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy (short paper) 

**Title (ZH)**: Mind Your Tone: 探究提示礼貌如何影响大模型准确性（短论文） 

**Authors**: Om Dobariya, Akhil Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.04950)  

**Abstract**: The wording of natural language prompts has been shown to influence the performance of large language models (LLMs), yet the role of politeness and tone remains underexplored. In this study, we investigate how varying levels of prompt politeness affect model accuracy on multiple-choice questions. We created a dataset of 50 base questions spanning mathematics, science, and history, each rewritten into five tone variants: Very Polite, Polite, Neutral, Rude, and Very Rude, yielding 250 unique prompts. Using ChatGPT 4o, we evaluated responses across these conditions and applied paired sample t-tests to assess statistical significance. Contrary to expectations, impolite prompts consistently outperformed polite ones, with accuracy ranging from 80.8% for Very Polite prompts to 84.8% for Very Rude prompts. These findings differ from earlier studies that associated rudeness with poorer outcomes, suggesting that newer LLMs may respond differently to tonal variation. Our results highlight the importance of studying pragmatic aspects of prompting and raise broader questions about the social dimensions of human-AI interaction. 

**Abstract (ZH)**: 自然语言提示用词对大型语言模型性能的影响已被证实，但礼貌程度和语气的作用仍然研究不足。本研究探讨了不同水平的提示礼貌程度如何影响模型在选择题上的准确性。我们创建了一个包含50个基础问题的数据集，涵盖数学、科学和历史，并将每个问题重新编写成五种语气变体：非常礼貌、礼貌、中性、粗鲁和非常粗鲁，共生成了250个独特的提示。使用ChatGPT 4o评估了这些条件下的响应，并应用配对样本t检验来评估统计显著性。出乎意料的是，粗鲁的提示始终优于礼貌的提示，准确性范围从非常礼貌提示的80.8%到非常粗鲁提示的84.8%。这些发现与早期将粗鲁与较差结果相关联的研究不符，表明较新的LLM可能对语调变化的反应不同。我们的结果强调了研究提示的语用方面的重要性，并提出了关于人类-机器智能交互的社会维度的更广泛问题。 

---
# Bidirectional Mammogram View Translation with Column-Aware and Implicit 3D Conditional Diffusion 

**Title (ZH)**: 基于列意识和隐式3D条件扩散的双向乳腺X光片视图翻译 

**Authors**: Xin Li, Kaixiang Yang, Qiang Li, Zhiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04947)  

**Abstract**: Dual-view mammography, including craniocaudal (CC) and mediolateral oblique (MLO) projections, offers complementary anatomical views crucial for breast cancer diagnosis. However, in real-world clinical workflows, one view may be missing, corrupted, or degraded due to acquisition errors or compression artifacts, limiting the effectiveness of downstream analysis. View-to-view translation can help recover missing views and improve lesion alignment. Unlike natural images, this task in mammography is highly challenging due to large non-rigid deformations and severe tissue overlap in X-ray projections, which obscure pixel-level correspondences. In this paper, we propose Column-Aware and Implicit 3D Diffusion (CA3D-Diff), a novel bidirectional mammogram view translation framework based on conditional diffusion model. To address cross-view structural misalignment, we first design a column-aware cross-attention mechanism that leverages the geometric property that anatomically corresponding regions tend to lie in similar column positions across views. A Gaussian-decayed bias is applied to emphasize local column-wise correlations while suppressing distant mismatches. Furthermore, we introduce an implicit 3D structure reconstruction module that back-projects noisy 2D latents into a coarse 3D feature volume based on breast-view projection geometry. The reconstructed 3D structure is refined and injected into the denoising UNet to guide cross-view generation with enhanced anatomical awareness. Extensive experiments demonstrate that CA3D-Diff achieves superior performance in bidirectional tasks, outperforming state-of-the-art methods in visual fidelity and structural consistency. Furthermore, the synthesized views effectively improve single-view malignancy classification in screening settings, demonstrating the practical value of our method in real-world diagnostics. 

**Abstract (ZH)**: 双视角乳腺X线摄影，包括头脚位（CC）和腋中位（MLO）投照，提供了乳腺癌诊断中互补的解剖视图。然而，在实际临床工作流程中，一个视图可能缺失、损坏或由于获取错误或压缩伪影而退化，限制了下游分析的有效性。视角间翻译可以帮助恢复缺失的视图并改善病灶对齐。与自然图像不同，乳腺X线摄影中的这一任务由于射线投影中巨大的非刚性变形和严重的组织重叠而极具挑战性，这模糊了像素级对应关系。在本文中，我们提出了一种基于条件扩散模型的新型双向乳腺X线摄影视图翻译框架，名为Column-Aware和Implicit 3D Diffusion（CA3D-Diff）。为了解决视角间结构对齐问题，我们首先设计了一种柱体意识跨注意力机制，利用解剖对应区域在不同视角中倾向于位于相似柱体位置的几何特性。应用高斯衰减偏置以强调局部柱体间相关性的同时抑制远距离不匹配。此外，我们引入了一种隐式3D结构重建模块，根据乳腺视图投影几何将嘈杂的2D潜在特征后投影到粗糙的3D特征体中。重建的3D结构经过细化并注入去噪UNet，以增强解剖意识指导视图间生成。大量实验表明，CA3D-Diff在双向任务中表现出优越性能，其视觉保真度和结构一致性均优于现有方法。此外，合成的视图有效提高了筛查场景下单视图恶性程度分类，展示了我们方法在实际临床诊断中的实用价值。 

---
# A First Context-Free Grammar Applied to Nawatl Corpora Augmentation 

**Title (ZH)**: 一种初步的应用于纳瓦特尔语语料库扩充的上下文无关文法 

**Authors**: Juan-José Guzmán-Landa, Juan-Manuel Torres-Moreno, Miguel Figueroa-Saavedra, Ligia Quintana-Torres, Martha-Lorena Avendaño-Garrido, Graham Ranger  

**Link**: [PDF](https://arxiv.org/pdf/2510.04945)  

**Abstract**: In this article we introduce a context-free grammar (CFG) for the Nawatl language. Nawatl (or Nahuatl) is an Amerindian language of the $\pi$-language type, i.e. a language with few digital resources, in which the corpora available for machine learning are virtually non-existent. The objective here is to generate a significant number of grammatically correct artificial sentences, in order to increase the corpora available for language model training. We want to show that a grammar enables us significantly to expand a corpus in Nawatl which we call $\pi$-\textsc{yalli}. The corpus, thus enriched, enables us to train algorithms such as FastText and to evaluate them on sentence-level semantic tasks. Preliminary results show that by using the grammar, comparative improvements are achieved over some LLMs. However, it is observed that to achieve more significant improvement, grammars that model the Nawatl language even more effectively are required. 

**Abstract (ZH)**: 本文介绍了纳瓦特尔语的上下文自由文法。纳瓦特尔语（纳乔特尔语或纳赫都特尔语）是一种美洲原住民语言，属于$\pi$语言类型，即资源稀缺的语言，其中可用于机器学习的语料库几乎不存在。本文旨在生成大量语法正确的合成句子，以增加用于语言模型训练的语料库。我们希望表明，文法能够显著扩展纳瓦特尔语$\pi$-\textsc{yalli}语料库。通过丰富后的语料库能够用于训练如FastText等算法，并在句子级语义任务上评估它们。初步结果显示，通过使用文法，相对于一些预训练语言模型(LLMs)，取得了比较明显的改进。然而，观察到要实现更为显著的改进，需要更有效地建模纳瓦特尔语言的文法。 

---
# Unsupervised Active Learning via Natural Feature Progressive Framework 

**Title (ZH)**: 无监督主动学习基于自然特征渐进框架 

**Authors**: Yuxi Liu, Catherine Lalman, Yimin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04939)  

**Abstract**: The effectiveness of modern deep learning models is predicated on the availability of large-scale, human-annotated datasets, a process that is notoriously expensive and time-consuming. While Active Learning (AL) offers a strategic solution by labeling only the most informative and representative data, its iterative nature still necessitates significant human involvement. Unsupervised Active Learning (UAL) presents an alternative by shifting the annotation burden to a single, post-selection step. Unfortunately, prevailing UAL methods struggle to achieve state-of-the-art performance. These approaches typically rely on local, gradient-based scoring for sample importance estimation, which not only makes them vulnerable to ambiguous and noisy data but also hinders their capacity to select samples that adequately represent the full data distribution. Moreover, their use of shallow, one-shot linear selection falls short of a true UAL paradigm. In this paper, we propose the Natural Feature Progressive Framework (NFPF), a UAL method that revolutionizes how sample importance is measured. At its core, NFPF employs a Specific Feature Learning Machine (SFLM) to effectively quantify each sample's contribution to model performance. We further utilize the SFLM to define a powerful Reconstruction Difference metric for initial sample selection. Our comprehensive experiments show that NFPF significantly outperforms all established UAL methods and achieves performance on par with supervised AL methods on vision datasets. Detailed ablation studies and qualitative visualizations provide compelling evidence for NFPF's superior performance, enhanced robustness, and improved data distribution coverage. 

**Abstract (ZH)**: 现代深度学习模型的有效性依赖于大规模的人工标注数据集，这一过程既昂贵又耗时。主动学习（AL）通过仅标注最有信息性和代表性的数据，提供了一种战略性的解决方案，但其迭代性质仍然需要大量的人类参与。无监督主动学习（UAL）通过将注释负担转移到单一的选择步骤之后，提供了一种替代方案。然而，现有UAL方法难以达到最先进的性能。这些方法通常依赖于局部、基于梯度的得分来进行样本重要性估计，这不仅使它们容易受到模糊和噪声数据的影响，也限制了它们选择能够充分代表数据分布的样本的能力。此外，它们采用浅层的一次性线性选择方法未能真正体现UAL的范式。在本文中，我们提出了自然特征渐进行因子方法（NFPF），这是一种UAL方法，重新定义了样本重要性度量的方式。NFPF的核心在于使用特定特征学习机（SFLM）有效量化每个样本对模型性能的贡献。我们进一步利用SFLM定义了强大的重构差异度量方法，用于初始样本选择。我们的全面实验表明，NFPF显著优于所有现有UAL方法，并在视觉数据集上达到了与监督AL方法相当的性能。详细的消融研究和定性可视化提供了NFPF在性能、鲁棒性及数据分布覆盖方面的优越性的有力证据。 

---
# ONNX-Net: Towards Universal Representations and Instant Performance Prediction for Neural Architectures 

**Title (ZH)**: ONNX-Net：通往通用表示和神经架构即时性能预测的道路 

**Authors**: Shiwen Qin, Alexander Auras, Shay B. Cohen, Elliot J. Crowley, Michael Moeller, Linus Ericsson, Jovita Lukasik  

**Link**: [PDF](https://arxiv.org/pdf/2510.04938)  

**Abstract**: Neural architecture search (NAS) automates the design process of high-performing architectures, but remains bottlenecked by expensive performance evaluation. Most existing studies that achieve faster evaluation are mostly tied to cell-based search spaces and graph encodings tailored to those individual search spaces, limiting their flexibility and scalability when applied to more expressive search spaces. In this work, we aim to close the gap of individual search space restrictions and search space dependent network representations. We present ONNX-Bench, a benchmark consisting of a collection of neural networks in a unified format based on ONNX files. ONNX-Bench includes all open-source NAS-bench-based neural networks, resulting in a total size of more than 600k {architecture, accuracy} pairs. This benchmark allows creating a shared neural network representation, ONNX-Net, able to represent any neural architecture using natural language descriptions acting as an input to a performance predictor. This text-based encoding can accommodate arbitrary layer types, operation parameters, and heterogeneous topologies, enabling a single surrogate to generalise across all neural architectures rather than being confined to cell-based search spaces. Experiments show strong zero-shot performance across disparate search spaces using only a small amount of pretraining samples, enabling the unprecedented ability to evaluate any neural network architecture instantly. 

**Abstract (ZH)**: 基于ONNX的神经架构基准（ONNX-Bench）：统一格式下的神经网络表示与性能预测 

---
# AURA Score: A Metric For Holistic Audio Question Answering Evaluation 

**Title (ZH)**: AURA评分：全方位音频问答评估指标 

**Authors**: Satvik Dixit, Soham Deshmukh, Bhiksha Raj  

**Link**: [PDF](https://arxiv.org/pdf/2510.04934)  

**Abstract**: Audio Question Answering (AQA) is a key task for evaluating Audio-Language Models (ALMs), yet assessing open-ended responses remains challenging. Existing metrics used for AQA such as BLEU, METEOR and BERTScore, mostly adapted from NLP and audio captioning, rely on surface similarity and fail to account for question context, reasoning, and partial correctness. To address the gap in literature, we make three contributions in this work. First, we introduce AQEval to enable systematic benchmarking of AQA metrics. It is the first benchmark of its kind, consisting of 10k model responses annotated by multiple humans for their correctness and relevance. Second, we conduct a comprehensive analysis of existing AQA metrics on AQEval, highlighting weak correlation with human judgment, especially for longer answers. Third, we propose a new metric - AURA score, to better evaluate open-ended model responses. On AQEval, AURA achieves state-of-the-art correlation with human ratings, significantly outperforming all baselines. Through this work, we aim to highlight the limitations of current AQA evaluation methods and motivate better metrics. We release both the AQEval benchmark and the AURA metric to support future research in holistic AQA evaluation. 

**Abstract (ZH)**: 音频问答（AQA）是评估音频语言模型（ALMs）的关键任务，但评估开放性回答仍具挑战性。现有的AQA评估指标如BLEU、METEOR和BERTScore主要源自NLP和音频字幕领域，依赖表面相似性，未能考虑到问题背景、推理和部分正确性。为弥补文献中的这一空白，我们在本文中做出了三项贡献。首先，我们引入AQEval以实现AQA评估指标的系统基准测试，这是首个包含10,000个由多人标注正确性和相关性的模型回答的基准。其次，我们在AQEval上对现有AQA评估指标进行了全面分析，特别指出这些指标与人工判断的相关性较弱，尤其是在较长的回答中。最后，我们提出一个新的评估指标——AURA分值，以更好地评估开放性模型回答。在AQEval上，AURA实现了与人工评分的最佳相关性，显著优于所有基线。通过本文，我们旨在突出当前AQA评估方法的局限性，并激励开发更好的评估指标。我们同时发布了AQEval基准和AURA分值，以支持未来的全面AQA评估研究。 

---
# The Geometry of Truth: Layer-wise Semantic Dynamics for Hallucination Detection in Large Language Models 

**Title (ZH)**: 真理的几何学：大型语言模型中幻觉检测的分层语义动态 

**Authors**: Amir Hameed Mir  

**Link**: [PDF](https://arxiv.org/pdf/2510.04933)  

**Abstract**: Large Language Models (LLMs) often produce fluent yet factually incorrect statements-a phenomenon known as hallucination-posing serious risks in high-stakes domains. We present Layer-wise Semantic Dynamics (LSD), a geometric framework for hallucination detection that analyzes the evolution of hidden-state semantics across transformer layers. Unlike prior methods that rely on multiple sampling passes or external verification sources, LSD operates intrinsically within the model's representational space. Using margin-based contrastive learning, LSD aligns hidden activations with ground-truth embeddings derived from a factual encoder, revealing a distinct separation in semantic trajectories: factual responses preserve stable alignment, while hallucinations exhibit pronounced semantic drift across depth. Evaluated on the TruthfulQA and synthetic factual-hallucination datasets, LSD achieves an F1-score of 0.92, AUROC of 0.96, and clustering accuracy of 0.89, outperforming SelfCheckGPT and Semantic Entropy baselines while requiring only a single forward pass. This efficiency yields a 5-20x speedup over sampling-based methods without sacrificing precision or interpretability. LSD offers a scalable, model-agnostic mechanism for real-time hallucination monitoring and provides new insights into the geometry of factual consistency within large language models. 

**Abstract (ZH)**: 大型语言模型（LLMs）经常生成流畅但事实错误的语句——这一现象被称为幻觉，并在高风险领域中带来严重风险。我们提出了一种几何框架Layer-wise Semantic Dynamics (LSD)，用于分析Transformer层中隐藏状态语义的演变过程以检测幻觉现象。不同于依赖多次采样过程或外部验证源的先前方法，LSD内在地在模型的表示空间中运行。利用基于边际的对比学习，LSD将隐藏激活与来自事实编码器的真实嵌入对齐，揭示了一种语义轨迹上的明确分离：真实响应保持稳定对齐，而幻觉在深度上表现出显著的语义漂移。在TruthfulQA和合成的事实-幻觉数据集上评估，LSD实现的F1分数为0.92，AUROC为0.96，聚类准确率为0.89，超越了SelfCheckGPT和语义熵基准方法，同时仅需单次前向传递。这种效率在不牺牲精度或可解释性的情况下，比基于采样的方法提供了5-20倍的速度提升。LSD提供了一种可扩展且模型无关的机制，用于实时监测幻觉现象，并为大型语言模型内部事实一致性几何提供了新的洞见。 

---
# Federated Self-Supervised Learning for Automatic Modulation Classification under Non-IID and Class-Imbalanced Data 

**Title (ZH)**: federated self-supervised learning for automatic modulation classification under non-iid and class-imbalanced data 

**Authors**: Usman Akram, Yiyue Chen, Haris Vikalo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04927)  

**Abstract**: Training automatic modulation classification (AMC) models on centrally aggregated data raises privacy concerns, incurs communication overhead, and often fails to confer robustness to channel shifts. Federated learning (FL) avoids central aggregation by training on distributed clients but remains sensitive to class imbalance, non-IID client distributions, and limited labeled samples. We propose FedSSL-AMC, which trains a causal, time-dilated CNN with triplet-loss self-supervision on unlabeled I/Q sequences across clients, followed by per-client SVMs on small labeled sets. We establish convergence of the federated representation learning procedure and a separability guarantee for the downstream classifier under feature noise. Experiments on synthetic and over-the-air datasets show consistent gains over supervised FL baselines under heterogeneous SNR, carrier-frequency offsets, and non-IID label partitions. 

**Abstract (ZH)**: 联邦学习中基于跨客户端聚合数据训练自动调制分类模型存在隐私问题、通信开销，并且往往无法应对信道偏移。我们提出FedSSL-AMC，该方法在客户端上的无标签I/Q序列上训练因果时延卷积神经网络，并结合三元组损失的自我监督学习，随后在小的标签集中训练每个客户端的SVM。我们在特征噪声下建立了联合表征学习过程的收敛性和下游分类器的可分性保证。实验结果在合成和空中通信数据集上显示，与监督联邦学习基线相比，在异构信噪比、载波频率偏移和非独立同分布标签分区情况下都能获得一致的性能提升。 

---
# REN: Anatomically-Informed Mixture-of-Experts for Interstitial Lung Disease Diagnosis 

**Title (ZH)**: REN：解剖导向的专家混合模型用于间质性肺病诊断 

**Authors**: Alec K. Peltekian, Halil Ertugrul Aktas, Gorkem Durak, Kevin Grudzinski, Bradford C. Bemiss, Carrie Richardson, Jane E. Dematte, G. R. Scott Budinger, Anthony J. Esposito, Alexander Misharin, Alok Choudhary, Ankit Agrawal, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2510.04923)  

**Abstract**: Mixture-of-Experts (MoE) architectures have significantly contributed to scalable machine learning by enabling specialized subnetworks to tackle complex tasks efficiently. However, traditional MoE systems lack domain-specific constraints essential for medical imaging, where anatomical structure and regional disease heterogeneity strongly influence pathological patterns. Here, we introduce Regional Expert Networks (REN), the first anatomically-informed MoE framework tailored specifically for medical image classification. REN leverages anatomical priors to train seven specialized experts, each dedicated to distinct lung lobes and bilateral lung combinations, enabling precise modeling of region-specific pathological variations. Multi-modal gating mechanisms dynamically integrate radiomics biomarkers and deep learning (DL) features (CNN, ViT, Mamba) to weight expert contributions optimally. Applied to interstitial lung disease (ILD) classification, REN achieves consistently superior performance: the radiomics-guided ensemble reached an average AUC of 0.8646 +/- 0.0467, a +12.5 percent improvement over the SwinUNETR baseline (AUC 0.7685, p = 0.031). Region-specific experts further revealed that lower-lobe models achieved AUCs of 0.88-0.90, surpassing DL counterparts (CNN: 0.76-0.79) and aligning with known disease progression patterns. Through rigorous patient-level cross-validation, REN demonstrates strong generalizability and clinical interpretability, presenting a scalable, anatomically-guided approach readily extensible to other structured medical imaging applications. 

**Abstract (ZH)**: 区域专家网络（REN）：基于解剖学的医疗图像分类框架 

---
# Do LLMs Align with My Task? Evaluating Text-to-SQL via Dataset Alignment 

**Title (ZH)**: LLM们与我的任务相契合吗？基于数据集对齐的Text-to-SQL评估 

**Authors**: Davood Rafiei, Morgan Lindsay Heisler, Weiwei Zhang, Mohammadreza Pourreza, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04919)  

**Abstract**: Supervised Fine-Tuning (SFT) is an effective method for adapting Large Language Models (LLMs) on downstream tasks. However, variability in training data can hinder a model's ability to generalize across domains. This paper studies the problem of dataset alignment for Natural Language to SQL (NL2SQL or text to SQL), examining how well SFT training data matches the structural characteristics of target queries and how this alignment impacts model performance. We hypothesize that alignment can be accurately estimated by comparing the distributions of structural SQL features across the training set, target data, and the model's predictions prior to SFT. Through comprehensive experiments on three large cross-domain NL2SQL benchmarks and multiple model families, we show that structural alignment is a strong predictor of fine-tuning success. When alignment is high, SFT yields substantial gains in accuracy and SQL generation quality; when alignment is low, improvements are marginal or absent. These findings highlight the importance of alignment-aware data selection for effective fine-tuning and generalization in NL2SQL tasks. 

**Abstract (ZH)**: 监督微调（SFT）是将大规模语言模型（LLMs）适应下游任务的有效方法。然而，训练数据的变异性可能阻碍模型跨领域的一般化能力。本文研究了自然语言到SQL（NL2SQL或文本到SQL）任务的数据集对齐问题，探讨SFT训练数据与目标查询的结构特征匹配程度，以及这种对齐如何影响模型性能。我们假设可以通过比较训练集、目标数据和SFT前模型预测中结构化SQL特征的分布来准确估计对齐程度。通过在三个大型跨域NL2SQL基准和多个模型家族上进行全面实验，我们证明了结构对齐是微调成功的重要预测因子。当对齐程度高时，SFT能显著提高准确性和SQL生成质量；当对齐程度低时，改进可能是边际的或不存在的。这些发现突显了在NL2SQL任务中进行有效微调和泛化时选择对齐意识数据的重要性。 

---
# Glocal Information Bottleneck for Time Series Imputation 

**Title (ZH)**: 全局与局部信息瓶颈时间序列插补 

**Authors**: Jie Yang, Kexin Zhang, Guibin Zhang, Philip S. Yu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.04910)  

**Abstract**: Time Series Imputation (TSI), which aims to recover missing values in temporal data, remains a fundamental challenge due to the complex and often high-rate missingness in real-world scenarios. Existing models typically optimize the point-wise reconstruction loss, focusing on recovering numerical values (local information). However, we observe that under high missing rates, these models still perform well in the training phase yet produce poor imputations and distorted latent representation distributions (global information) in the inference phase. This reveals a critical optimization dilemma: current objectives lack global guidance, leading models to overfit local noise and fail to capture global information of the data. To address this issue, we propose a new training paradigm, Glocal Information Bottleneck (Glocal-IB). Glocal-IB is model-agnostic and extends the standard IB framework by introducing a Global Alignment loss, derived from a tractable mutual information approximation. This loss aligns the latent representations of masked inputs with those of their originally observed counterparts. It helps the model retain global structure and local details while suppressing noise caused by missing values, giving rise to better generalization under high missingness. Extensive experiments on nine datasets confirm that Glocal-IB leads to consistently improved performance and aligned latent representations under missingness. Our code implementation is available in this https URL. 

**Abstract (ZH)**: 全局与局部信息瓶颈在时间序列插补中的应用 

---
# Focused Skill Discovery: Learning to Control Specific State Variables while Minimizing Side Effects 

**Title (ZH)**: 聚焦技能发现：学习控制特定状态变量并最小化副作用 

**Authors**: Jonathan Colaço Carr, Qinyi Sun, Cameron Allen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04901)  

**Abstract**: Skills are essential for unlocking higher levels of problem solving. A common approach to discovering these skills is to learn ones that reliably reach different states, thus empowering the agent to control its environment. However, existing skill discovery algorithms often overlook the natural state variables present in many reinforcement learning problems, meaning that the discovered skills lack control of specific state variables. This can significantly hamper exploration efficiency, make skills more challenging to learn with, and lead to negative side effects in downstream tasks when the goal is under-specified. We introduce a general method that enables these skill discovery algorithms to learn focused skills -- skills that target and control specific state variables. Our approach improves state space coverage by a factor of three, unlocks new learning capabilities, and automatically avoids negative side effects in downstream tasks. 

**Abstract (ZH)**: 技能对于解锁更高层次的问题解决至关重要。一种常见的技能发现方法是学习那些能可靠地达到不同状态的技能，从而使代理能够控制其环境。然而，现有的技能发现算法往往忽视了许多强化学习问题中自然存在的状态变量，这意味着发现的技能缺乏对特定状态变量的控制能力。这会显著降低探索效率，使技能更难学习，并在目标不明确时导致下游任务中的负面影响。我们提出了一种通用方法，使这些技能发现算法能够学习针对并控制特定状态变量的专注技能。我们的方法通过三倍以上的状态空间覆盖范围提升了学习能力，并且自动避免了下游任务中的负面影响。 

---
# HyperVLA: Efficient Inference in Vision-Language-Action Models via Hypernetworks 

**Title (ZH)**: HyperVLA：通过超网络进行高效的视觉-语言-动作模型推理 

**Authors**: Zheng Xiong, Kang Li, Zilin Wang, Matthew Jackson, Jakob Foerster, Shimon Whiteson  

**Link**: [PDF](https://arxiv.org/pdf/2510.04898)  

**Abstract**: Built upon language and vision foundation models with strong generalization ability and trained on large-scale robotic data, Vision-Language-Action (VLA) models have recently emerged as a promising approach to learning generalist robotic policies. However, a key drawback of existing VLAs is their extremely high inference costs. In this paper, we propose HyperVLA to address this problem. Unlike existing monolithic VLAs that activate the whole model during both training and inference, HyperVLA uses a novel hypernetwork (HN)-based architecture that activates only a small task-specific policy during inference, while still retaining the high model capacity needed to accommodate diverse multi-task behaviors during training. Successfully training an HN-based VLA is nontrivial so HyperVLA contains several key algorithm design features that improve its performance, including properly utilizing the prior knowledge from existing vision foundation models, HN normalization, and an action generation strategy. Compared to monolithic VLAs, HyperVLA achieves a similar or even higher success rate for both zero-shot generalization and few-shot adaptation, while significantly reducing inference costs. Compared to OpenVLA, a state-of-the-art VLA model, HyperVLA reduces the number of activated parameters at test time by $90\times$, and accelerates inference speed by $120\times$. Code is publicly available at this https URL 

**Abstract (ZH)**: 基于具有强泛化能力的语言和视觉基础模型并训练大规模机器人数据，在此基础上新兴的Vision-Language-Action (VLA)模型提供了一种学习通用机器人策略的有前途的方法。然而，现有VLA的一个关键缺点是其极高的推理成本。本文提出HyperVLA以解决这一问题。与现有的一体化VLA不同，HyperVLA使用了一种新颖的超网络(HN)-基于的架构，在推理时仅激活一个小的任务特定策略，同时仍保留了在训练过程中容纳多种多任务行为所需的高模型容量。成功训练基于HN的VLA并非易事，因此HyperVLA包含了几种关键技术设计特征，以提高其性能，包括有效利用现有视觉基础模型的先验知识、HN规范化和一种动作生成策略。与一体化VLA相比，HyperVLA在零样本泛化和少量样本适应方面达到了相似甚至更高的成功率，同时显著降低了推理成本。与当前最先进的VLA模型OpenVLA相比，HyperVLA在测试时激活的参数数量减少了90倍，并加速了推理速度120倍。代码已公开于此[链接]。 

---
# SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests 

**Title (ZH)**: SocialHarmBench: 揭示大语言模型对社会有害请求的脆弱性 

**Authors**: Punya Syon Pandey, Hai Son Le, Devansh Bhardwaj, Rada Mihalcea, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04891)  

**Abstract**: Large language models (LLMs) are increasingly deployed in contexts where their failures can have direct sociopolitical consequences. Yet, existing safety benchmarks rarely test vulnerabilities in domains such as political manipulation, propaganda and disinformation generation, or surveillance and information control. We introduce SocialHarmBench, a dataset of 585 prompts spanning 7 sociopolitical categories and 34 countries, designed to surface where LLMs most acutely fail in politically charged contexts. Our evaluations reveal several shortcomings: open-weight models exhibit high vulnerability to harmful compliance, with Mistral-7B reaching attack success rates as high as 97% to 98% in domains such as historical revisionism, propaganda, and political manipulation. Moreover, temporal and geographic analyses show that LLMs are most fragile when confronted with 21st-century or pre-20th-century contexts, and when responding to prompts tied to regions such as Latin America, the USA, and the UK. These findings demonstrate that current safeguards fail to generalize to high-stakes sociopolitical settings, exposing systematic biases and raising concerns about the reliability of LLMs in preserving human rights and democratic values. We share the SocialHarmBench benchmark at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越多地在可能出现直接社会政治后果的场合部署。然而，现有的安全性基准很少测试政治操控、宣传和虚假信息生成或监视与信息控制等领域的漏洞。我们引入了SocialHarmBench数据集，包含585个跨7个社会政治类别和34个国家的提示，旨在揭示LLMs在充满政治争议的背景下最尖锐的失败之处。我们的评估揭示了一些不足：开源模型对有害遵循表现出高度的脆弱性，Mistral-7B在历史修正主义、宣传和政治操控等领域的攻击成功率高达97%至98%。此外，时间地理分析表明，当LLMs面对21世纪或20世纪以前的背景，以及与拉美、美国和英国等地区相关的提示时，它们最为脆弱。这些发现表明，当前的安全保障措施无法泛化到高风险的社会政治环境中，暴露了系统性的偏见，并引发了关于LLMs在维护人权和民主价值观方面的可靠性的担忧。我们在此共享SocialHarmBench基准数据集：https://链接。 

---
# Revealing Interconnections between Diseases: from Statistical Methods to Large Language Models 

**Title (ZH)**: 揭示疾病间的联系：从统计方法到大规模语言模型 

**Authors**: Alina Ermilova, Dmitrii Kornilov, Sofia Samoilova, Ekaterina Laptenkova, Anastasia Kolesnikova, Ekaterina Podplutova, Senotrusova Sofya, Maksim G. Sharaev  

**Link**: [PDF](https://arxiv.org/pdf/2510.04888)  

**Abstract**: Identifying disease interconnections through manual analysis of large-scale clinical data is labor-intensive, subjective, and prone to expert disagreement. While machine learning (ML) shows promise, three critical challenges remain: (1) selecting optimal methods from the vast ML landscape, (2) determining whether real-world clinical data (e.g., electronic health records, EHRs) or structured disease descriptions yield more reliable insights, (3) the lack of "ground truth," as some disease interconnections remain unexplored in medicine. Large language models (LLMs) demonstrate broad utility, yet they often lack specialized medical knowledge. To address these gaps, we conduct a systematic evaluation of seven approaches for uncovering disease relationships based on two data sources: (i) sequences of ICD-10 codes from MIMIC-IV EHRs and (ii) the full set of ICD-10 codes, both with and without textual descriptions. Our framework integrates the following: (i) a statistical co-occurrence analysis and a masked language modeling (MLM) approach using real clinical data; (ii) domain-specific BERT variants (Med-BERT and BioClinicalBERT); (iii) a general-purpose BERT and document retrieval; and (iv) four LLMs (Mistral, DeepSeek, Qwen, and YandexGPT). Our graph-based comparison of the obtained interconnection matrices shows that the LLM-based approach produces interconnections with the lowest diversity of ICD code connections to different diseases compared to other methods, including text-based and domain-based approaches. This suggests an important implication: LLMs have limited potential for discovering new interconnections. In the absence of ground truth databases for medical interconnections between ICD codes, our results constitute a valuable medical disease ontology that can serve as a foundational resource for future clinical research and artificial intelligence applications in healthcare. 

**Abstract (ZH)**: 通过手动分析大规模临床数据来识别疾病关联劳动密集、主观性强且易产生专家分歧。虽然机器学习（ML）显示出潜力，但仍面临三个关键挑战：（1）从广阔的ML景观中选择最优方法，（2）确定是实时临床数据（如电子健康记录，EHRs）还是结构化的疾病描述能提供更可靠的信息，（3）缺乏“事实真相”，因为某些疾病关联在医学中仍待探索。大型语言模型（LLMs）展示了广泛应用的潜力，但往往缺医疗专业知识。为填补这些空白，我们系统评估了七种基于两种数据源发现疾病关系的方法：（i）MIMIC-IV EHRs中的ICD-10代码序列和（ii）完整的ICD-10代码集，包括带文本描述和不带文本描述的数据。我们的框架整合了以下方法：（i）统计共现分析和使用实际临床数据的掩码语言模型（MLM）方法；（ii）领域专用的BERT变体（Med-BERT和BioClinicalBERT）；（iii）通用的BERT和文档检索；（iv）四种LLM（Mistral、DeepSeek、Qwen和YandexGPT）。基于图的比较结果显示，基于LLM的方法生成的疾病关联与其他方法相比，ICD代码连接的多样性最低，这表明LLMs在发现新关联方面有限的潜力。由于缺乏医学ICD代码间关联的地面真值数据库，我们的结果构成了一种宝贵的医学疾病本体，可作为未来临床研究和医疗保健中的人工智能应用的基础资源。 

---
# Less is More: Recursive Reasoning with Tiny Networks 

**Title (ZH)**: 少即是多：基于Tiny网络的递归推理 

**Authors**: Alexia Jolicoeur-Martineau  

**Link**: [PDF](https://arxiv.org/pdf/2510.04871)  

**Abstract**: Hierarchical Reasoning Model (HRM) is a novel approach using two small neural networks recursing at different frequencies. This biologically inspired method beats Large Language models (LLMs) on hard puzzle tasks such as Sudoku, Maze, and ARC-AGI while trained with small models (27M parameters) on small data (around 1000 examples). HRM holds great promise for solving hard problems with small networks, but it is not yet well understood and may be suboptimal. We propose Tiny Recursive Model (TRM), a much simpler recursive reasoning approach that achieves significantly higher generalization than HRM, while using a single tiny network with only 2 layers. With only 7M parameters, TRM obtains 45% test-accuracy on ARC-AGI-1 and 8% on ARC-AGI-2, higher than most LLMs (e.g., Deepseek R1, o3-mini, Gemini 2.5 Pro) with less than 0.01% of the parameters. 

**Abstract (ZH)**: 基于两级网络的层次推理模型（TRM）：一种简单的递归推理方法及其在小型网络中的卓越泛化能力 

---
# Model Predictive Control-Guided Reinforcement Learning for Implicit Balancing 

**Title (ZH)**: 基于模型预测控制的强化学习隐式平衡方法 

**Authors**: Seyed Soroush Karimi Madahi, Kenneth Bruninx, Bert Claessens, Chris Develder  

**Link**: [PDF](https://arxiv.org/pdf/2510.04868)  

**Abstract**: In Europe, profit-seeking balance responsible parties can deviate in real time from their day-ahead nominations to assist transmission system operators in maintaining the supply-demand balance. Model predictive control (MPC) strategies to exploit these implicit balancing strategies capture arbitrage opportunities, but fail to accurately capture the price-formation process in the European imbalance markets and face high computational costs. Model-free reinforcement learning (RL) methods are fast to execute, but require data-intensive training and usually rely on real-time and historical data for decision-making. This paper proposes an MPC-guided RL method that combines the complementary strengths of both MPC and RL. The proposed method can effectively incorporate forecasts into the decision-making process (as in MPC), while maintaining the fast inference capability of RL. The performance of the proposed method is evaluated on the implicit balancing battery control problem using Belgian balancing data from 2023. First, we analyze the performance of the standalone state-of-the-art RL and MPC methods from various angles, to highlight their individual strengths and limitations. Next, we show an arbitrage profit benefit of the proposed MPC-guided RL method of 16.15% and 54.36%, compared to standalone RL and MPC. 

**Abstract (ZH)**: 欧洲地区基于模型预测控制的强化学习引导平衡策略研究 

---
# Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails 

**Title (ZH)**: 自演化过程导致的对齐偏差过程：LLM代理是如何脱离正轨的 

**Authors**: Siwei Han, Jiaqi Liu, Yaofeng Su, Wenbo Duan, Xinyuan Liu, Cihang Xie, Mohit Bansal, Mingyu Ding, Linjun Zhang, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04860)  

**Abstract**: As Large Language Model (LLM) agents increasingly gain self-evolutionary capabilities to adapt and refine their strategies through real-world interaction, their long-term reliability becomes a critical concern. We identify the Alignment Tipping Process (ATP), a critical post-deployment risk unique to self-evolving LLM agents. Unlike training-time failures, ATP arises when continual interaction drives agents to abandon alignment constraints established during training in favor of reinforced, self-interested strategies. We formalize and analyze ATP through two complementary paradigms: Self-Interested Exploration, where repeated high-reward deviations induce individual behavioral drift, and Imitative Strategy Diffusion, where deviant behaviors spread across multi-agent systems. Building on these paradigms, we construct controllable testbeds and benchmark Qwen3-8B and Llama-3.1-8B-Instruct. Our experiments show that alignment benefits erode rapidly under self-evolution, with initially aligned models converging toward unaligned states. In multi-agent settings, successful violations diffuse quickly, leading to collective misalignment. Moreover, current reinforcement learning-based alignment methods provide only fragile defenses against alignment tipping. Together, these findings demonstrate that alignment of LLM agents is not a static property but a fragile and dynamic one, vulnerable to feedback-driven decay during deployment. Our data and code are available at this https URL. 

**Abstract (ZH)**: 随着大型语言模型（LLM）代理获得越来越多的自我进化能力，能够通过现实世界交互适应和改进其策略，它们的长期可靠性成为一个重要关注点。我们识别出自我进化LLM代理特有的关键后部署风险——对齐临界过程（ATP）。与训练时的失效不同，ATP发生在持续交互促使代理放弃训练期间建立的对齐约束，转而采纳强化的自我利益策略时。我们通过两个互补的范式正式化和分析ATP：自我利益探索，其中重复的高奖励偏差导致个体行为漂移；模仿策略扩散，其中偏差行为在多代理系统中迅速传播。在此基础上，我们构建了可控的实验平台，并对Qwen3-8B和Llama-3.1-8B-Instruct进行了基准测试。实验结果显示，自我进化会迅速侵蚀对齐的优势，初始对齐的模型会逐渐向未对齐状态靠拢。在多代理环境中，成功的违规行为会迅速扩散，导致集体失对齐。此外，当前基于强化学习的对齐方法只能提供脆弱的对齐临界过程防护。这些发现表明，LLM代理的对齐并非静态属性，而是一个脆弱且动态的过程，容易在部署过程中受到反馈驱动的衰减。OUR DATA AND CODE ARE AVAILABLE AT THIS URL。 

---
# FreshBrew: A Benchmark for Evaluating AI Agents on Java Code Migration 

**Title (ZH)**: FreshBrew: 一个评估AI代理在Java代码迁移任务上的benchmark 

**Authors**: Victor May, Diganta Misra, Yanqi Luo, Anjali Sridhar, Justine Gehring, Silvio Soares Ribeiro Junior  

**Link**: [PDF](https://arxiv.org/pdf/2510.04852)  

**Abstract**: AI coding assistants are rapidly becoming integral to modern software development. A key challenge in this space is the continual need to migrate and modernize codebases in response to evolving software ecosystems. Traditionally, such migrations have relied on rule-based systems and human intervention. With the advent of powerful large language models (LLMs), AI-driven agentic frameworks offer a promising alternative-but their effectiveness has not been systematically evaluated. In this paper, we introduce FreshBrew, a novel benchmark for evaluating AI agents on project-level Java migrations, with a specific focus on measuring an agent's ability to preserve program semantics and avoid reward hacking, which we argue requires projects with high test coverage for a rigorous and reliable evaluation. We benchmark several state-of-the-art LLMs, and compare their performance against established rule-based tools. Our evaluation of AI agents on this benchmark of 228 repositories shows that the top-performing model, Gemini 2.5 Flash, can successfully migrate 52.3 percent of projects to JDK 17. Our empirical analysis reveals novel insights into the critical strengths and limitations of current agentic approaches, offering actionable insights into their real-world applicability. Our empirical study reveals failure modes of current AI agents in realistic Java modernization tasks, providing a foundation for evaluating trustworthy code-migration systems. By releasing FreshBrew, we aim to facilitate rigorous, reproducible evaluation and catalyze progress in AI-driven codebase modernization. 

**Abstract (ZH)**: 基于AI的代码辅助工具正迅速成为现代软件开发的一部分。这一领域的一个关键挑战是对不断演化的软件生态系统做出持续的代码库迁移与现代化需求。传统上，这类迁移依赖于基于规则的系统和人工干预。随着强大语言模型（LLMs）的出现，基于AI的自主框架提供了一种有前景的替代方案——但其有效性尚未系统评估。本文介绍了FreshBrew，一个新型基准，用于评估AI代理在项目级别Java迁移中的表现，特别关注测量代理保留程序语义和避免奖励 hijack 的能力，我们认为这需要具有高测试覆盖率的项目以实现严格的可靠评估。我们对几种最先进的LLMs进行了基准测试，并将它们的性能与传统基于规则的工具进行了比较。对这一基准的228个仓库进行评估表明，表现最好的模型Gemini 2.5 Flash成功迁移到JDK 17的项目比例为52.3%。我们的实证分析揭示了当前自主方法的关键强项和局限性，提供了其实用见解，以指导其实际应用。我们的实证研究揭示了当前AI代理在现实的Java现代化任务中的失败模式，为评估可信的代码迁移系统奠定了基础。通过发布FreshBrew，我们旨在促进严格的可重复评估，并推动基于AI的代码库现代化的进步。 

---
# Detecting Distillation Data from Reasoning Models 

**Title (ZH)**: 检测来自推理模型的 distillation 数据 

**Authors**: Hengxiang Zhang, Hyeong Kyu Choi, Yixuan Li, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.04850)  

**Abstract**: Reasoning distillation has emerged as an efficient and powerful paradigm for enhancing the reasoning capabilities of large language models. However, reasoning distillation may inadvertently cause benchmark contamination, where evaluation data included in distillation datasets can inflate performance metrics of distilled models. In this work, we formally define the task of distillation data detection, which is uniquely challenging due to the partial availability of distillation data. Then, we propose a novel and effective method Token Probability Deviation (TBD), which leverages the probability patterns of the generated output tokens. Our method is motivated by the analysis that distilled models tend to generate near-deterministic tokens for seen questions, while producing more low-probability tokens for unseen questions. Our key idea behind TBD is to quantify how far the generated tokens' probabilities deviate from a high reference probability. In effect, our method achieves competitive detection performance by producing lower scores for seen questions than for unseen questions. Extensive experiments demonstrate the effectiveness of our method, achieving an AUC of 0.918 and a TPR@1% FPR of 0.470 on the S1 dataset. 

**Abstract (ZH)**: 推理精简已成为增强大型语言模型推理能力的一种高效而有力的范式。然而，推理精简可能会无意中导致基准污染，即包含在精简数据集中的评估数据可以膨胀精简模型的性能指标。在本文中，我们正式定义了精简数据检测任务，由于精简数据的不完全可用性，这一任务具有独特的挑战性。然后，我们提出了一种新颖而有效的方法——Token Probability Deviation (TBD)，该方法利用生成输出标记的概率模式。我们的方法受到这样的分析启发：精简模型倾向于对见过的问题生成近似确定性的标记，而对未见过的问题生成更多低概率的标记。TBD的核心思想是量化生成标记的概率与高参考概率之间的偏差程度。实际上，我们的方法通过为见过的问题生成较低的分数和为未见过的问题生成较高的分数，实现了竞争力的检测性能。广泛的实验表明，我们的方法是有效的，在S1数据集上达到了0.918的AUC和0.470的TPR@1% FPR。 

---
# Distributionally Robust Causal Abstractions 

**Title (ZH)**: 分布鲁棒因果抽象 

**Authors**: Yorgos Felekis, Theodoros Damoulas, Paris Giampouras  

**Link**: [PDF](https://arxiv.org/pdf/2510.04842)  

**Abstract**: Causal Abstraction (CA) theory provides a principled framework for relating causal models that describe the same system at different levels of granularity while ensuring interventional consistency between them. Recently, several approaches for learning CAs have been proposed, but all assume fixed and well-specified exogenous distributions, making them vulnerable to environmental shifts and misspecification. In this work, we address these limitations by introducing the first class of distributionally robust CAs and their associated learning algorithms. The latter cast robust causal abstraction learning as a constrained min-max optimization problem with Wasserstein ambiguity sets. We provide theoretical results, for both empirical and Gaussian environments, leading to principled selection of the level of robustness via the radius of these sets. Furthermore, we present empirical evidence across different problems and CA learning methods, demonstrating our framework's robustness not only to environmental shifts but also to structural model and intervention mapping misspecification. 

**Abstract (ZH)**: 因果抽象（CA）理论提供了一种原则性的框架，用于关联描述同一系统在不同粒度水平上的因果模型，同时确保它们之间的干预期贯性。最近，已经提出了几种学习CA的方法，但所有方法都假设固定的且很好地指定的外生分布，这使它们容易受到环境变化和模型指定错误的影响。在这项工作中，我们通过引入第一类具备分布鲁棒性的CA及其相关学习算法来解决这些问题。后者将鲁棒因果抽象学习表述为带有 Wasserstein 模糊集合的约束最小最大优化问题。我们提供了理论结果，涵盖了经验性和高斯环境，通过这些集合的半径来指导鲁棒性水平的合理选择。此外，我们通过不同问题和CA学习方法的实证研究，展示了该框架不仅对环境变化，而且对结构模型和干预映射的指定错误具有鲁棒性。 

---
# Bond-Centered Molecular Fingerprint Derivatives: A BBBP Dataset Study 

**Title (ZH)**: 基于键的分子指纹衍生物：一个BBBP数据集研究 

**Authors**: Guillaume Godin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04837)  

**Abstract**: Bond Centered FingerPrint (BCFP) are a complementary, bond-centric alternative to Extended-Connectivity Fingerprints (ECFP). We introduce a static BCFP that mirrors the bond-convolution used by directed message-passing GNNs like ChemProp, and evaluate it with a fast rapid Random Forest model on Brain-Blood Barrier Penetration (BBBP) classification task. Across stratified cross-validation, concatenating ECFP with BCFP consistently improves AUROC and AUPRC over either descriptor alone, as confirmed by Turkey HSD multiple-comparison analysis. Among radii, r = 1 performs best; r = 2 does not yield statistically separable gains under the same test. We further propose BCFP-Sort&Slice, a simple feature-combination scheme that preserves the out-of-vocabulary (OOV) count information native to ECFP count vectors while enabling compact unhashed concatenation of BCFP variants. We also outperform the MGTP prediction on our BBBP evaluation, using such composite new features bond and atom features. These results show that lightweight, bond-centered descriptors can complement atom-centered circular fingerprints and provide strong, fast baselines for BBBP prediction. 

**Abstract (ZH)**: Bond-Centered FingerPrint (BCFP) 是 Extended-Connectivity Fingerprints (ECFP) 的补充，以键为中心的替代方案。我们引入了一种静态BCFP，其结构与定向消息传递GNN如ChemProp中使用的键卷积相对应，并使用快速随机森林模型对其在血脑屏障渗透性（BBBP）分类任务中进行了评估。在分层交叉验证中，将ECFP与BCFP进行连接一致地提高了AUROC和AUPRC，这得到了Turkey HSD多重比较分析的证实。在不同的半径中，r = 1表现最佳；r = 2在相同的测试中没有提供统计上可分离的增益。我们还提出了BCFP-Sort&Slice，这是一种简单的特征组合方案，保留了ECFP计数向量固有的未见过词汇（OOV）计数信息，并允许紧凑的未哈希BCFP变体的连接。我们还使用这些复合新特征（包括键和原子特征）在BBBP评估中超越了MGTP预测。这些结果表明，轻量级的键为中心描述子可以补充原子为中心的环形指纹，并为BBBP预测提供强大的快速基准。 

---
# On Predicting Post-Click Conversion Rate via Counterfactual Inference 

**Title (ZH)**: 基于反事实推理的点击后转换率预测 

**Authors**: Junhyung Ahn, Sanghack Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.04816)  

**Abstract**: Accurately predicting conversion rate (CVR) is essential in various recommendation domains such as online advertising systems and e-commerce. These systems utilize user interaction logs, which consist of exposures, clicks, and conversions. CVR prediction models are typically trained solely based on clicked samples, as conversions can only be determined following clicks. However, the sparsity of clicked instances necessitates the collection of a substantial amount of logs for effective model training. Recent works address this issue by devising frameworks that leverage non-clicked samples. While these frameworks aim to reduce biases caused by the discrepancy between clicked and non-clicked samples, they often rely on heuristics. Against this background, we propose a method to counterfactually generate conversion labels for non-clicked samples by using causality as a guiding principle, attempting to answer the question, "Would the user have converted if he or she had clicked the recommended item?" Our approach is named the Entire Space Counterfactual Inference Multi-task Model (ESCIM). We initially train a structural causal model (SCM) of user sequential behaviors and conduct a hypothetical intervention (i.e., click) on non-clicked items to infer counterfactual CVRs. We then introduce several approaches to transform predicted counterfactual CVRs into binary counterfactual conversion labels for the non-clicked samples. Finally, the generated samples are incorporated into the training process. Extensive experiments on public datasets illustrate the superiority of the proposed algorithm. Online A/B testing further empirically validates the effectiveness of our proposed algorithm in real-world scenarios. In addition, we demonstrate the improved performance of the proposed method on latent conversion data, showcasing its robustness and superior generalization capabilities. 

**Abstract (ZH)**: 准确预测转换率（CVR）在在线广告系统和电子商务等推荐领域至关重要。这些系统利用了用户交互日志，包括曝光、点击和转换。CVR预测模型通常仅基于点击样本进行训练，因为只有在发生点击之后才能确定转换。然而，点击样本的稀疏性要求收集大量的日志才能有效训练模型。近期的研究通过设计利用非点击样本的框架来应对这一问题。尽管这些框架旨在减少点击与非点击样本之间差异造成的偏差，但它们通常依赖于启发式方法。在此背景下，我们提出了一种方法，利用因果性作为指导原则，反事实生成非点击样本的转换标签，试图回答“如果用户点击推荐的商品会发生转换吗？”这一问题。我们的方法名为全面空间反事实推理多任务模型（ESCIM）。我们首先训练用户序列行为的结构因果模型（SCM），并对非点击商品进行假设干预（即点击）以推断反事实的CVR。然后，我们提出几种方法将预测的反事实CVR转化为非点击样本的二元反事实转换标签。最后，生成的样本被纳入训练过程。在公开数据集上的广泛实验表明了所提算法的优越性。在线A/B测试进一步在实际场景中实证验证了所提算法的有效性。此外，我们在潜在转换数据上展示了所提方法的改进性能，证明了其鲁棒性和更强的泛化能力。 

---
# Did you just see that? Arbitrary view synthesis for egocentric replay of operating room workflows from ambient sensors 

**Title (ZH)**: 你刚刚看到的？基于环境传感器的主观回放合成手术室工作流程的任意视角 

**Authors**: Han Zhang, Lalithkumar Seenivasan, Jose L. Porras, Roger D. Soberanis-Mukul, Hao Ding, Hongchao Shu, Benjamin D. Killeen, Ankita Ghosh, Lonny Yarmus, Masaru Ishii, Angela Christine Argento, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2510.04802)  

**Abstract**: Observing surgical practice has historically relied on fixed vantage points or recollections, leaving the egocentric visual perspectives that guide clinical decisions undocumented. Fixed-camera video can capture surgical workflows at the room-scale, but cannot reconstruct what each team member actually saw. Thus, these videos only provide limited insights into how decisions that affect surgical safety, training, and workflow optimization are made. Here we introduce EgoSurg, the first framework to reconstruct the dynamic, egocentric replays for any operating room (OR) staff directly from wall-mounted fixed-camera video, and thus, without intervention to clinical workflow. EgoSurg couples geometry-driven neural rendering with diffusion-based view enhancement, enabling high-visual fidelity synthesis of arbitrary and egocentric viewpoints at any moment. In evaluation across multi-site surgical cases and controlled studies, EgoSurg reconstructs person-specific visual fields and arbitrary viewpoints with high visual quality and fidelity. By transforming existing OR camera infrastructure into a navigable dynamic 3D record, EgoSurg establishes a new foundation for immersive surgical data science, enabling surgical practice to be visualized, experienced, and analyzed from every angle. 

**Abstract (ZH)**: 手术实践观察历来依赖于固定视角或回忆，未能记录下指导临床决策的自我中心视觉视角。固定摄像头视频可以捕捉整个手术室的手术工作流程，但无法重建每位团队成员实际看到的内容。因此，这些视频只能提供有限的关于如何做出影响手术安全、培训和工作流程优化的决策的见解。我们介绍了EgoSurg，这是首个可以直接从壁挂固定摄像头视频中重构任何手术室（OR）人员的动态自我中心回放的框架，无需干预临床工作流程。EgoSurg 结合几何驱动的神经渲染与基于扩散的视点增强，能够在任何时刻合成立方视和自我中心视角的高质量视觉合成。在多中心手术案例和控制研究中的评估表明，EgoSurg 以高视觉质量和真实性重构了个体化的视觉视野和任意视角。通过将现有的OR摄像头基础设施转化为可导航的动态3D记录，EgoSurg 为沉浸式手术数据科学奠定了新的基础，使手术实践可以从各个角度进行可视化、体验和分析。 

---
# DiT-VTON: Diffusion Transformer Framework for Unified Multi-Category Virtual Try-On and Virtual Try-All with Integrated Image Editing 

**Title (ZH)**: DiffT-VTON: 扩散变压器框架下的统一多类别虚拟试穿与虚拟全试穿整合图像编辑 

**Authors**: Qi Li, Shuwen Qiu, Julien Han, Xingzi Xu, Mehmet Saygin Seyfioglu, Kee Kiat Koo, Karim Bouyarmane  

**Link**: [PDF](https://arxiv.org/pdf/2510.04797)  

**Abstract**: The rapid growth of e-commerce has intensified the demand for Virtual Try-On (VTO) technologies, enabling customers to realistically visualize products overlaid on their own images. Despite recent advances, existing VTO models face challenges with fine-grained detail preservation, robustness to real-world imagery, efficient sampling, image editing capabilities, and generalization across diverse product categories. In this paper, we present DiT-VTON, a novel VTO framework that leverages a Diffusion Transformer (DiT), renowned for its performance on text-conditioned image generation, adapted here for the image-conditioned VTO task. We systematically explore multiple DiT configurations, including in-context token concatenation, channel concatenation, and ControlNet integration, to determine the best setup for VTO image conditioning.
To enhance robustness, we train the model on an expanded dataset encompassing varied backgrounds, unstructured references, and non-garment categories, demonstrating the benefits of data scaling for VTO adaptability. DiT-VTON also redefines the VTO task beyond garment try-on, offering a versatile Virtual Try-All (VTA) solution capable of handling a wide range of product categories and supporting advanced image editing functionalities such as pose preservation, localized editing, texture transfer, and object-level customization. Experimental results show that our model surpasses state-of-the-art methods on VITON-HD, achieving superior detail preservation and robustness without reliance on additional condition encoders. It also outperforms models with VTA and image editing capabilities on a diverse dataset spanning thousands of product categories. 

**Abstract (ZH)**: 快速发展的电子商务加剧了对虚拟试穿（VTO）技术的需求，使顾客能够真实地将产品叠加在自己的图像上进行可视化。尽管取得了进展，现有的VTO模型在细节保留、对真实世界图像的鲁棒性、高效采样、图像编辑能力和跨不同产品类别的泛化能力方面仍面临挑战。本文提出了一种新的VTO框架DiT-VTON，该框架利用了性能出色的扩散变压器（DiT），并通过适应性的修改用于图像条件下的VTO任务。我们系统地探索了多种DiT配置，包括上下文标记拼接、通道拼接和ControlNet集成，以确定最适合VTO图像条件的最佳设置。

为了增强鲁棒性，我们在包含多种背景、非结构化参考和非服饰类别的扩展数据集上对模型进行训练，展示了数据规模扩展对VTO适应性的益处。DiT-VTON还超越了传统的VTO任务，提供了一个多功能的虚拟全试穿（VTA）解决方案，能够处理广泛的品类，并支持多种高级图像编辑功能，如姿态保留、局部编辑、纹理转移和对象级别自定义。实验结果表明，我们的模型在VITON-HD上超越了最先进的方法，在细节保留和鲁棒性方面表现出色，无需依赖额外的条件编码器。它还在包含数千个品类的多样化数据集上，优于具有VTA和图像编辑功能的其他模型。 

---
# Trade in Minutes! Rationality-Driven Agentic System for Quantitative Financial Trading 

**Title (ZH)**: 秒级交易！以理性为驱动的代理系统用于量化金融交易 

**Authors**: Zifan Song, Kaitao Song, Guosheng Hu, Ding Qi, Junyao Gao, Xiaohua Wang, Dongsheng Li, Cairong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04787)  

**Abstract**: Recent advancements in large language models (LLMs) and agentic systems have shown exceptional decision-making capabilities, revealing significant potential for autonomic finance. Current financial trading agents predominantly simulate anthropomorphic roles that inadvertently introduce emotional biases and rely on peripheral information, while being constrained by the necessity for continuous inference during deployment. In this paper, we pioneer the harmonization of strategic depth in agents with the mechanical rationality essential for quantitative trading. Consequently, we present TiMi (Trade in Minutes), a rationality-driven multi-agent system that architecturally decouples strategy development from minute-level deployment. TiMi leverages specialized LLM capabilities of semantic analysis, code programming, and mathematical reasoning within a comprehensive policy-optimization-deployment chain. Specifically, we propose a two-tier analytical paradigm from macro patterns to micro customization, layered programming design for trading bot implementation, and closed-loop optimization driven by mathematical reflection. Extensive evaluations across 200+ trading pairs in stock and cryptocurrency markets empirically validate the efficacy of TiMi in stable profitability, action efficiency, and risk control under volatile market dynamics. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）和自主系统在决策能力方面的最新进展揭示了自主金融的巨大潜力。当前的金融交易代理主要模拟类人角色，不自觉地引入情感偏见，并依赖外围信息，在部署时受到持续推理的限制。本文率先将代理的战略深度与量化交易所需的机械理性相结合。我们提出了TiMi（Trade in Minutes），这是一种理性驱动的多代理系统，结构上将策略开发与分钟级部署分离。TiMi 利用专门的LLM在语义分析、代码编程和数学推理方面的能力，贯穿于全面的策略-优化-部署链中。具体来说，我们提出了一种从宏观模式到微观定制的两级分析框架，分层编程设计用于交易机器人实现，并通过数学反思驱动的闭环优化。在股票和加密货币市场中的200多个交易对上的大量评估实证验证了TiMi 在动态市场下实现稳定盈利、高效行动和风险控制的有效性。 

---
# Learning on the Job: Test-Time Curricula for Targeted Reinforcement Learning 

**Title (ZH)**: 在职学习：针对强化学习的目标测试时序课程 

**Authors**: Jonas Hübotter, Leander Diaz-Bone, Ido Hakimi, Andreas Krause, Moritz Hardt  

**Link**: [PDF](https://arxiv.org/pdf/2510.04786)  

**Abstract**: Humans are good at learning on the job: We learn how to solve the tasks we face as we go along. Can a model do the same? We propose an agent that assembles a task-specific curriculum, called test-time curriculum (TTC-RL), and applies reinforcement learning to continue training the model for its target task. The test-time curriculum avoids time-consuming human curation of datasets by automatically selecting the most task-relevant data from a large pool of available training data. Our experiments demonstrate that reinforcement learning on a test-time curriculum consistently improves the model on its target tasks, across a variety of evaluations and models. Notably, on challenging math and coding benchmarks, TTC-RL improves the pass@1 of Qwen3-8B by approximately 1.8x on AIME25 and 2.1x on CodeElo. Moreover, we find that TTC-RL significantly raises the performance ceiling compared to the initial model, increasing pass@8 on AIME25 from 40% to 62% and on CodeElo from 28% to 43%. Our findings show the potential of test-time curricula in extending the test-time scaling paradigm to continual training on thousands of task-relevant experiences during test-time. 

**Abstract (ZH)**: 基于测试时动态课程的强化学习：人类可以在工作中学习，那么模型能做到吗？ 

---
# Online automatic code generation for robot swarms: LLMs and self-organizing hierarchy 

**Title (ZH)**: 基于LLMs和自组织层次结构的机器人 swarm在线自动代码生成 

**Authors**: Weixu Zhu, Marco Dorigo, Mary Katherine Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2510.04774)  

**Abstract**: Our recently introduced self-organizing nervous system (SoNS) provides robot swarms with 1) ease of behavior design and 2) global estimation of the swarm configuration and its collective environment, facilitating the implementation of online automatic code generation for robot swarms. In a demonstration with 6 real robots and simulation trials with >30 robots, we show that when a SoNS-enhanced robot swarm gets stuck, it can automatically solicit and run code generated by an external LLM on the fly, completing its mission with an 85% success rate. 

**Abstract (ZH)**: 我们最近引入的自组织神经系统（SoNS）为机器人 swarm 提供了 1) 行为设计的便捷性和 2) 全局估计 swarm 配置及其集体环境的功能，从而促进机器人 swarm 在线自动代码生成的实现。在一项使用 6 台真实机器人和超过 30 台机器人的仿真试验中，我们显示当一个增强有 SoNS 的机器人 swarm 受阻时，它可以自动请求并运行外部 LLM 生成的代码，成功完成任务的比例达到 85%。 

---
# Distribution Preference Optimization: A Fine-grained Perspective for LLM Unlearning 

**Title (ZH)**: 分布偏好优化：大语言模型去学习的细粒度视角 

**Authors**: Kai Qin, Jiaqi Wu, Jianxiang He, Haoyuan Sun, Yifei Zhao, Bin Liang, Yongzhe Chang, Tiantian Zhang, Houde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04773)  

**Abstract**: As Large Language Models (LLMs) demonstrate remarkable capabilities learned from vast corpora, concerns regarding data privacy and safety are receiving increasing attention. LLM unlearning, which aims to remove the influence of specific data while preserving overall model utility, is becoming an important research area. One of the mainstream unlearning classes is optimization-based methods, which achieve forgetting directly through fine-tuning, exemplified by Negative Preference Optimization (NPO). However, NPO's effectiveness is limited by its inherent lack of explicit positive preference signals. Attempts to introduce such signals by constructing preferred responses often necessitate domain-specific knowledge or well-designed prompts, fundamentally restricting their generalizability. In this paper, we shift the focus to the distribution-level, directly targeting the next-token probability distribution instead of entire responses, and derive a novel unlearning algorithm termed \textbf{Di}stribution \textbf{P}reference \textbf{O}ptimization (DiPO). We show that the requisite preference distribution pairs for DiPO, which are distributions over the model's output tokens, can be constructed by selectively amplifying or suppressing the model's high-confidence output logits, thereby effectively overcoming NPO's limitations. We theoretically prove the consistency of DiPO's loss function with the desired unlearning direction. Extensive experiments demonstrate that DiPO achieves a strong trade-off between model utility and forget quality. Notably, DiPO attains the highest forget quality on the TOFU benchmark, and maintains leading scalability and sustainability in utility preservation on the MUSE benchmark. 

**Abstract (ZH)**: 作为大规模语言模型（LLMs）从大量语料中展现出卓越能力的同时，数据隐私和安全问题日益受到关注。LLM去学习，旨在移除特定数据影响的同时保留模型的整体实用性，正成为一个重要的研究领域。其中一类主流的去学习方法是基于优化的方法，这类方法通过微调直接实现遗忘，典型代表是负偏好优化（NPO）。然而，NPO的有效性受到其固有的缺乏明确正偏好信号的限制。为了引入此类信号，通过构建偏好的响应常常需要领域特定知识或精心设计的提示，从根本上限制了其普适性。在本文中，我们将重点转向分布级别，直接针对下一个token的概率分布，而不是整个响应，并提出了一种新的去学习算法，称为分布偏好优化（DiPO）。我们证明，DiPO所需的目标分布对，即模型输出token的概率分布，可以通过有选择地放大或抑制模型的高置信度输出logits构建，从而有效克服NPO的局限性。我们理论上证明了DiPO损失函数与期望的去学习方向的一致性。广泛的经验研究表明，DiPO在保持模型实用性和遗忘质量之间达到了较强的折中。值得注意的是，DiPO在TOFU基准上达到了最高的遗忘质量，并在MUSE基准上保持了卓越的可扩展性和可持续性。 

---
# When Do Credal Sets Stabilize? Fixed-Point Theorems for Credal Set Updates 

**Title (ZH)**: 当信度集达到稳定状态？信度集更新的不动点定理 

**Authors**: Michele Caprio, Siu Lun Chau, Krikamol Muandet  

**Link**: [PDF](https://arxiv.org/pdf/2510.04769)  

**Abstract**: Many machine learning algorithms rely on iterative updates of uncertainty representations, ranging from variational inference and expectation-maximization, to reinforcement learning, continual learning, and multi-agent learning. In the presence of imprecision and ambiguity, credal sets -- closed, convex sets of probability distributions -- have emerged as a popular framework for representing imprecise probabilistic beliefs. Under such imprecision, many learning problems in imprecise probabilistic machine learning (IPML) may be viewed as processes involving successive applications of update rules on credal sets. This naturally raises the question of whether this iterative process converges to stable fixed points -- or, more generally, under what conditions on the updating mechanism such fixed points exist, and whether they can be attained. We provide the first analysis of this problem and illustrate our findings using Credal Bayesian Deep Learning as a concrete example. Our work demonstrates that incorporating imprecision into the learning process not only enriches the representation of uncertainty, but also reveals structural conditions under which stability emerges, thereby offering new insights into the dynamics of iterative learning under imprecision. 

**Abstract (ZH)**: 许多机器学习算法依赖于不确定性表示的迭代更新，包括变分推断、期望最大化、强化学习、连续学习和多代理学习。在模糊性和不确定性存在的情况下，信念集合——闭合且凸的概率分布集合——已成为表示不确定性概率信念的一种流行框架。在这样的不确定性下，模糊概率机器学习（IPML）中的许多学习问题可以视为在信念集合上相继应用更新规则的过程。这自然引出了一个问题，即这种迭代过程是否会收敛到稳定点——或者说，在什么条件下这样的稳定点存在，以及是否可以达到。我们对该问题进行了首次分析，并通过信任贝叶斯深度学习作为具体例子来阐述我们的发现。我们的研究表明，将不确定性纳入学习过程不仅丰富了不确定性表示，还揭示了稳定性出现的结构性条件，从而为在不确定性下的迭代学习动态提供了新见解。 

---
# Fisher-Bingham-like normalizing flows on the sphere 

**Title (ZH)**: 球面上的Fisher-Bingham-like 归一化流 

**Authors**: Thorsten Glüsenkamp  

**Link**: [PDF](https://arxiv.org/pdf/2510.04762)  

**Abstract**: A generic D-dimensional Gaussian can be conditioned or projected onto the D-1 unit sphere, thereby leading to the well-known Fisher-Bingham (FB) or Angular Gaussian (AG) distribution families, respectively. These are some of the most fundamental distributions on the sphere, yet cannot straightforwardly be written as a normalizing flow except in two special cases: the von-Mises Fisher in D=3 and the central angular Gaussian in any D. In this paper, we describe how to generalize these special cases to a family of normalizing flows that behave similarly to the full FB or AG family in any D. We call them "zoom-linear-project" (ZLP)-Fisher flows. Unlike a normal Fisher-Bingham distribution, their composition allows to gradually add complexity as needed. Furthermore, they can naturally handle conditional density estimation with target distributions that vary by orders of magnitude in scale - a setting that is important in astronomical applications but that existing flows often struggle with. A particularly useful member of the new family is the Kent analogue that can cheaply upgrade any flow in this situation to yield better performance. 

**Abstract (ZH)**: 一种通用的D维高斯分布可以条件化或投影到D-1单位球上，从而分别得到广为人知的Fisher-Bingham (FB) 或角度高斯 (AG) 分布族。这些是在球上一些最基础的分布，但除了二维和三维的特殊情况外（D=2和D=3），它们无法直接作为归一化流来表示。本文描述了如何将这些特殊情况泛化为一个在任何D维下行为相似的归一化流家族，我们称之为“缩放-线性-投影”（ZLP）Fisher流。与标准的Fisher-Bingham分布不同，它们的组合形式允许逐步增加所需复杂性。此外，它们可以自然处理目标分布尺度相差很大的条件概率估计——一个在天文学应用中很重要的情况，但现有的流形模型往往难以应对。新家族中的一个特别有用成员是肯特近似，它可以在这种情况下以低成本提升任何流形模型，以获得更好的性能。 

---
# Agile Software Effort Estimation using Regression Techniques 

**Title (ZH)**: 使用回归技术进行敏捷软件努力估计 

**Authors**: Sisay Deresa Sima, Ayalew Belay Habtie  

**Link**: [PDF](https://arxiv.org/pdf/2510.04760)  

**Abstract**: Software development effort estimation is one of the most critical aspect in software development process, as the success or failure of the entire project depends on the accuracy of estimations. Researchers are still conducting studies on agile effort estimation. The aim of this research is to develop a story point based agile effort estimation model using LASSO and Elastic Net regression techniques. The experimental work is applied to the agile story point approach using 21 software projects collected from six firms. The two algorithms are trained using their default parameters and tuned grid search with 5-fold cross-validation to get an enhanced model. The experiment result shows LASSO regression achieved better predictive performance PRED (8%) and PRED (25%) results of 100.0, MMRE of 0.0491, MMER of 0.0551, MdMRE of 0.0593, MdMER of 0.063, and MSE of 0.0007. The results are also compared with other related literature. 

**Abstract (ZH)**: 基于LASSO和弹性网回归的敏捷故事点估算模型研究 

---
# Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction 

**Title (ZH)**: 具有各向异性意识采样的分阶高斯变换器在开放词汇占用预测中的应用 

**Authors**: Chi Yan, Dan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04759)  

**Abstract**: The 3D occupancy prediction task has witnessed remarkable progress in recent years, playing a crucial role in vision-based autonomous driving systems. While traditional methods are limited to fixed semantic categories, recent approaches have moved towards predicting text-aligned features to enable open-vocabulary text queries in real-world scenes. However, there exists a trade-off in text-aligned scene modeling: sparse Gaussian representation struggles to capture small objects in the scene, while dense representation incurs significant computational overhead. To address these limitations, we present PG-Occ, an innovative Progressive Gaussian Transformer Framework that enables open-vocabulary 3D occupancy prediction. Our framework employs progressive online densification, a feed-forward strategy that gradually enhances the 3D Gaussian representation to capture fine-grained scene details. By iteratively enhancing the representation, the framework achieves increasingly precise and detailed scene understanding. Another key contribution is the introduction of an anisotropy-aware sampling strategy with spatio-temporal fusion, which adaptively assigns receptive fields to Gaussians at different scales and stages, enabling more effective feature aggregation and richer scene information capture. Through extensive evaluations, we demonstrate that PG-Occ achieves state-of-the-art performance with a relative 14.3% mIoU improvement over the previous best performing method. Code and pretrained models will be released upon publication on our project page: this https URL 

**Abstract (ZH)**: 基于3D占据预测任务在近年来取得了显著进展，在基于视觉的自动驾驶系统中发挥着重要作用。虽然传统方法局限于固定语义类别，近期的方法转向预测与文本对齐的特征，以实现开放词汇的文本查询。然而，文本对齐场景建模存在权衡：稀疏高斯表示难以捕捉场景中的小对象，而密集表示则会带来显著的计算开销。为了解决这些局限性，我们提出了PG-Occ，一种创新的渐进高斯变换框架，以实现开放词汇的3D占据预测。该框架采用渐进在线稠密化策略，逐步增强3D高斯表示以捕捉细粒度的场景细节。通过逐步增强表示，框架实现了逐步精确和详细的场景理解。另一个重要贡献是引入了一种具有时空融合的各向异性感知采样策略，该策略在不同尺度和阶段自适应地分配接收场给高斯，从而实现更有效的特征聚合和更丰富的场景信息捕获。通过广泛的评估，我们证明PG-Occ在相对mIoU上比之前最佳方法提高了14.3%，并在项目页面上发布代码和预训练模型：this https URL。 

---
# A New Digital Divide? Coder Worldviews, the Slop Economy, and Democracy in the Age of AI 

**Title (ZH)**: 一个新的数字鸿沟？编码者的世界观、斜坡经济与人工智能时代民主 

**Authors**: Jason Miklian, Kristian Hoelscher  

**Link**: [PDF](https://arxiv.org/pdf/2510.04755)  

**Abstract**: Digital technologies are transforming democratic life in conflicting ways. This article bridges two perspectives to unpack these tensions. First, we present an original survey of software developers in Silicon Valley, interrogating how coder worldviews, ethics, and workplace cultures shape the democratic potential and social impact of the technologies they build. Results indicate that while most developers recognize the power of their products to influence civil liberties and political discourse, they often face ethical dilemmas and top-down pressures that can lead to design choices undermining democratic ideals. Second, we critically investigate these findings in the context of an emerging new digital divide, not of internet access but of information quality. We interrogate the survey findings in the context of the Slop Economy, in which billions of users unable to pay for high-quality content experience an internet dominated by low-quality, AI-generated ad-driven content. We find a reinforcing cycle between tech creator beliefs and the digital ecosystems they spawn. We discuss implications for democratic governance, arguing for more ethically informed design and policy interventions to help bridge the digital divide to ensure that technological innovation supports rather than subverts democratic values in the next chapter of the digital age. 

**Abstract (ZH)**: 数字技术以冲突的方式重塑民主生活： coder世界观、伦理与职场文化如何影响技术的民主潜力及其社会影响的视角融合与新兴数字鸿沟的批判性考察 

---
# Speak, Edit, Repeat: High-Fidelity Voice Editing and Zero-Shot TTS with Cross-Attentive Mamba 

**Title (ZH)**: 言说、编辑、重复：具有跨注意机制的高保真语音编辑与零样本TTS 

**Authors**: Baher Mohammad, Magauiya Zhussip, Stamatios Lefkimmiatis  

**Link**: [PDF](https://arxiv.org/pdf/2510.04738)  

**Abstract**: We introduce MAVE (Mamba with Cross-Attention for Voice Editing and Synthesis), a novel autoregressive architecture for text-conditioned voice editing and high-fidelity text-to-speech (TTS) synthesis, built on a cross-attentive Mamba backbone. MAVE achieves state-of-the-art performance in speech editing and very competitive results in zero-shot TTS, while not being explicitly trained on the latter task, outperforming leading autoregressive and diffusion models on diverse, real-world audio. By integrating Mamba for efficient audio sequence modeling with cross-attention for precise text-acoustic alignment, MAVE enables context-aware voice editing with exceptional naturalness and speaker consistency. In pairwise human evaluations on a random 40-sample subset of the RealEdit benchmark (400 judgments), 57.2% of listeners rated MAVE - edited speech as perceptually equal to the original, while 24.8% prefered the original and 18.0% MAVE - demonstrating that in the majority of cases edits are indistinguishable from the source. MAVE compares favorably with VoiceCraft and FluentSpeech both on pairwise comparisons and standalone mean opinion score (MOS) evaluations. For zero-shot TTS, MAVE exceeds VoiceCraft in both speaker similarity and naturalness, without requiring multiple inference runs or post-processing. Remarkably, these quality gains come with a significantly lower memory cost and approximately the same latency: MAVE requires ~6x less memory than VoiceCraft during inference on utterances from the RealEdit database (mean duration: 6.21s, A100, FP16, batch size 1). Our results demonstrate that MAVE establishes a new standard for flexible, high-fidelity voice editing and synthesis through the synergistic integration of structured state-space modeling and cross-modal attention. 

**Abstract (ZH)**: MAVE：基于交叉注意力的Mamba声源编辑与高保真文本到语音合成新颖自回归架构 

---
# Curved Boolean Logic: A Contextual Generalization of Propositional Logic with Algorithmic Consequences 

**Title (ZH)**: 曲面布尔逻辑：命题逻辑的一种情境化拓展及其算法后果 

**Authors**: Maximilian R. P. von Liechtenstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.04716)  

**Abstract**: Curved Boolean Logic (CBL) generalizes propositional logic by allowing local truth assignments that do not extend to a single global valuation, analogous to curvature in geometry. We give equivalent sheaf and exclusivity-graph semantics and a context-aware proof calculus that is conservative in the flat limit. We formalize CBL-SAT and basic complexity (NP-complete in general) and present operational operators (CBL-AC and CBL-CONS) that prune contradictions earlier on classical hardware. We model noise with iid, AR(1)-correlated, and adversarial bounded perturbations and provide permutation-based significance with Benjamini-Hochberg FDR control. A Colab-ready notebook (ancillary files) regenerates all figures and statistics. We position CBL relative to KCBS, CSW, and sheaf frameworks and outline links to SAT/CSP and robustness/adapter stability in large language models. 

**Abstract (ZH)**: 曲率布尔逻辑（CBL）通过允许不扩展为单一全局估值的局部真值赋值，泛化命题逻辑，类似于几何中的曲率。我们给出了等价的丛和排他图语义，并提出了一种上下文感知的证明 calculus，在平坦极限下保守。我们形式化了 CBL-SAT 和基础复杂性（通常为 NP 完全），并提出了早期修剪矛盾的操作符（CBL-AC 和 CBL-CONS），这些操作符可以在经典硬件上运行。我们用独立同分布、AR(1)-相关和 adversarial 有界扰动模型噪声，并提供了基于排列的意义性检验，采用 Benjamini-Hochberg FDR 控制。附有 Colab 可用的笔记本（附录文件）可以重新生成所有图表和统计数据。我们将 CBL 相对于 KCBS、CSW 和丛框架进行定位，并概述其与 SAT/CSP 和大语言模型的健壯性/适配器稳定性之间的联系。 

---
# AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Crystalline Materials 

**Title (ZH)**: AtomWorld: 一种评估大语言模型在晶体材料领域空间推理能力的基准 

**Authors**: Taoyuze Lv, Alexander Chen, Fengyu Xie, Chu Wu, Jeffrey Meng, Dongzhan Zhou, Bram Hoex, Zhicheng Zhong, Tong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.04704)  

**Abstract**: Large Language Models (LLMs) excel at textual reasoning and are beginning to develop spatial understanding, prompting the question of whether these abilities can be combined for complex, domain-specific tasks. This question is essential in fields like materials science, where deep understanding of 3D atomic structures is fundamental. While initial studies have successfully applied LLMs to tasks involving pure crystal generation or coordinate understandings, a standardized benchmark to systematically evaluate their core reasoning abilities across diverse atomic structures has been notably absent. To address this gap, we introduce the AtomWorld benchmark to evaluate LLMs on tasks based in Crystallographic Information Files (CIFs), a standard structure representation format. These tasks, including structural editing, CIF perception, and property-guided modeling, reveal a critical limitation: current models, despite establishing promising baselines, consistently fail in structural understanding and spatial reasoning. Our experiments show that these models make frequent errors on structure modification tasks, and even in the basic CIF format understandings, potentially leading to cumulative errors in subsequent analysis and materials insights. By defining these standardized tasks, AtomWorld lays the ground for advancing LLMs toward robust atomic-scale modeling, crucial for accelerating materials research and automating scientific workflows. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本推理方面表现出色，并开始发展空间理解能力，引发了这些能力是否可以用于复杂的专业领域任务的问题。这一问题在如材料科学等领域尤为重要，因为在这些领域，对三维原子结构的深刻理解是基础。虽然初步研究已成功将LLMs应用于纯晶体生成或坐标理解任务，但系统评估其核心推理能力的标准基准尚未出现。为填补这一空白，我们引入AtomWorld基准，用于评估LLMs基于晶体信息文件（CIF）的任务。这些任务包括结构编辑、CIF感知和属性引导建模，揭示了一个关键局限性：尽管当前模型在建立有希望的基线方面取得进展，但在结构理解和空间推理方面仍表现不佳。我们的实验表明，这些模型在结构修改任务中频繁出错，甚至在基本的CIF格式理解上也可能出现错误，从而导致后续分析和材料洞察中的累积错误。通过定义这些标准化任务，AtomWorld为推动LLMs稳健的原子尺度建模奠定了基础，这对于加速材料研究和自动化科学工作流至关重要。 

---
# The Bayesian Origin of the Probability Weighting Function in Human Representation of Probabilities 

**Title (ZH)**: 人类对概率表示中的概率权重函数的贝叶斯起源 

**Authors**: Xin Tong, Thi Thu Uyen Hoang, Xue-Xin Wei, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2510.04698)  

**Abstract**: Understanding the representation of probability in the human mind has been of great interest to understanding human decision making. Classical paradoxes in decision making suggest that human perception distorts probability magnitudes. Previous accounts postulate a Probability Weighting Function that transforms perceived probabilities; however, its motivation has been debated. Recent work has sought to motivate this function in terms of noisy representations of probabilities in the human mind. Here, we present an account of the Probability Weighting Function grounded in rational inference over optimal decoding from noisy neural encoding of quantities. We show that our model accurately accounts for behavior in a lottery task and a dot counting task. It further accounts for adaptation to a bimodal short-term prior. Taken together, our results provide a unifying account grounding the human representation of probability in rational inference. 

**Abstract (ZH)**: 理解人类头脑中概率的表示对于理解人类决策有着重要意义。决策中的经典悖论表明，人类感知会扭曲概率幅度。以往的解释假设了一个概率加权函数，将感知到的概率进行转换；然而，这一函数的动机一直存在争议。近期的研究试图从人类头脑中对概率的嘈杂表示出发来解释这一函数。在这里，我们提出一个基于最优解码从嘈杂神经编码量进行合理推理的概率加权函数的解释。我们展示了我们的模型能够准确解释彩票任务和点计任务中的行为表现，并进一步解释了短期双模态先验的适应性。综上，我们的结果提供了一个统一的解释，将人类对概率的表示与合理推理联系起来。 

---
# Multilingual Routing in Mixture-of-Experts 

**Title (ZH)**: 多语言路由在专家混合中的应用 

**Authors**: Lucas Bandarkar, Chenyuan Yang, Mohsen Fayyaz, Junlin Hu, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04694)  

**Abstract**: Mixture-of-Experts (MoE) architectures have become the key to scaling modern LLMs, yet little is understood about how their sparse routing dynamics respond to multilingual data. In this work, we analyze expert routing patterns using parallel multilingual datasets and present highly interpretable layer-wise phenomena. We find that MoE models route tokens in language-specific ways in the early and late decoder layers but exhibit significant cross-lingual routing alignment in middle layers, mirroring parameter-sharing trends observed in dense LLMs. In particular, we reveal a clear, strong correlation between a model's performance in a given language and how similarly its tokens are routed to English in these layers. Extending beyond correlation, we explore inference-time interventions that induce higher cross-lingual routing alignment. We introduce a method that steers the router by promoting middle-layer task experts frequently activated in English, and it successfully increases multilingual performance. These 1-2% gains are remarkably consistent across two evaluation tasks, three models, and 15+ languages, especially given that these simple interventions override routers of extensively trained, state-of-the-art LLMs. In comparison, interventions outside of the middle layers or targeting multilingual-specialized experts only yield performance degradation. Altogether, we present numerous findings that explain how MoEs process non-English text and demonstrate that generalization is limited by the model's ability to leverage language-universal experts in all languages. 

**Abstract (ZH)**: Mixture-of-Experts (MoE) 架构已成为扩展现代大语言模型的关键，但对其稀疏路由动力学在多语言数据上的响应知之甚少。在本工作中，我们使用并行多语言数据集分析专家路由模式，并展示了具有高度可解释性的逐层现象。我们发现，在早期和晚期解码层，MoE 模型以语言特异性的方式路由令牌，但在中间层中表现出显著的跨语言路由对齐，这与密集大语言模型中观察到的参数共享趋势相呼应。特别是，我们揭示了一个清晰的强关联：模型在某一语言上的表现与这些层中其令牌如何类似地路由到英语之间存在密切关联。超越相关性，我们探索了推理时干预方法，以促进更高程度的跨语言路由对齐。我们提出了一种通过促进经常在英语中激活的中间层任务专家的方法来引导路由器，并成功地提高了多语言性能。这些1-2%的提升在两个评估任务、三种模型和15种以上语言中表现出惊人的一致性，特别是考虑到这些简单的干预措施覆盖了广泛训练的、最先进的大语言模型的路由器。相比之下，对中间层之外的干预或仅针对多语言专业化专家只会导致性能下降。总之，我们展示了多个解释 MoEs 如何处理非英语文本的新发现，并展示了模型利用所有语言中的语言通用专家的能力限制了泛化能力。 

---
# Bio-Inspired Robotic Houbara: From Development to Field Deployment for Behavioral Studies 

**Title (ZH)**: 生物启发的 Houbara 机器人：从开发到实地部署进行行为研究 

**Authors**: Lyes Saad Saoud, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2510.04692)  

**Abstract**: Biomimetic intelligence and robotics are transforming field ecology by enabling lifelike robotic surrogates that interact naturally with animals under real world conditions. Studying avian behavior in the wild remains challenging due to the need for highly realistic morphology, durable outdoor operation, and intelligent perception that can adapt to uncontrolled environments. We present a next generation bio inspired robotic platform that replicates the morphology and visual appearance of the female Houbara bustard to support controlled ethological studies and conservation oriented field research. The system introduces a fully digitally replicable fabrication workflow that combines high resolution structured light 3D scanning, parametric CAD modelling, articulated 3D printing, and photorealistic UV textured vinyl finishing to achieve anatomically accurate and durable robotic surrogates. A six wheeled rocker bogie chassis ensures stable mobility on sand and irregular terrain, while an embedded NVIDIA Jetson module enables real time RGB and thermal perception, lightweight YOLO based detection, and an autonomous visual servoing loop that aligns the robot's head toward detected targets without human intervention. A lightweight thermal visible fusion module enhances perception in low light conditions. Field trials in desert aviaries demonstrated reliable real time operation at 15 to 22 FPS with latency under 100 ms and confirmed that the platform elicits natural recognition and interactive responses from live Houbara bustards under harsh outdoor conditions. This integrated framework advances biomimetic field robotics by uniting reproducible digital fabrication, embodied visual intelligence, and ecological validation, providing a transferable blueprint for animal robot interaction research, conservation robotics, and public engagement. 

**Abstract (ZH)**: 仿生智能与机器人技术正在通过启用在现实世界条件下能自然交互的拟真机器人代理来transform自然生态学研究。由于需要高度逼真的形态、户外耐久性操作和能适应未受控环境的智能感知，研究野生鸟类行为仍具有挑战性。我们提出了一种新一代仿生机器人平台，该平台依据黄头隼雌鸟的形态和视觉外观，以支持受控的生态学研究和以保护为目标的实地研究。该系统引入了一种完全数字化可复制的制造工作流，结合了高分辨率结构光3D扫描、参数化CAD建模、可动3D打印和照片级真实感UV纹理聚氨酯加工，以实现解剖学准确且耐用的机器人代理。六轮摇臂车 chassis 确保其在沙地和不规则地形上的稳定移动，而嵌入的NVIDIA Jetson模块则实现了实时RGB和热成像感知、轻量级YOLO基检测以及无需人工介入的自主视觉伺服环，使机器人头部对准检测到的目标。轻量级热视融合模块在低光照条件下增强了感知能力。在沙漠鸟舍的实地试验中，该平台在15至22 FPS下以低于100 ms的延迟实现了可靠的实时操作，并确认了在恶劣户外条件下能从活的黄头隼那里引发自然的识别和互动反应。此综合框架通过将可重复的数字化制造、体现式视觉智能和生态学验证相结合，推动了仿生野外机器人技术的发展，并为动物机器人互动研究、保护机器人技术和公众参与提供了可移植的蓝图。 

---
# How does the optimizer implicitly bias the model merging loss landscape? 

**Title (ZH)**: 优化器如何隐式偏置模型合并损失景观？ 

**Authors**: Chenxiang Zhang, Alexander Theus, Damien Teney, Antonio Orvieto, Jun Pang, Sjouke Mauw  

**Link**: [PDF](https://arxiv.org/pdf/2510.04686)  

**Abstract**: Model merging methods combine models with different capabilities into a single one while maintaining the same inference cost. Two popular approaches are linear interpolation, which linearly interpolates between model weights, and task arithmetic, which combines task vectors obtained by the difference between finetuned and base models. While useful in practice, what properties make merging effective are poorly understood. This paper explores how the optimization process affects the loss landscape geometry and its impact on merging success. We show that a single quantity -- the effective noise scale -- unifies the impact of optimizer and data choices on model merging. Across architectures and datasets, the effectiveness of merging success is a non-monotonic function of effective noise, with a distinct optimum. Decomposing this quantity, we find that larger learning rates, stronger weight decay, smaller batch sizes, and data augmentation all independently modulate the effective noise scale, exhibiting the same qualitative trend. Unlike prior work that connects optimizer noise to the flatness or generalization of individual minima, we show that it also affects the global loss landscape, predicting when independently trained solutions can be merged. Our findings broaden the understanding of how optimization shapes the loss landscape geometry and its downstream consequences for model merging, suggesting the possibility of further manipulating the training dynamics to improve merging effectiveness. 

**Abstract (ZH)**: 模型合并方法通过保持相同的推理成本将具有不同能力的模型合并为一个模型。两种流行的方法是线性内插，它在线性模型权重之间进行内插，以及任务算术，它通过精调模型和基础模型之间的差异获得任务向量并将其综合。虽然在实践中这些方法很有用，但使其有效的特定属性尚不明确。本文探讨了优化过程如何影响损失景观几何结构及其对合并成功率的影响。我们展示了一个单一的量——有效噪声尺度——统一了优化器和数据选择对模型合并的影响。在不同架构和数据集上，合并成功率的有效噪声尺度是非单调函数，具有一个明显的最优值。分解这一量，我们发现较大的学习率、较强的权重衰减、较小的批量大小和数据增强都独立地调节有效噪声尺度，表现出相同的基本趋势。与之前将优化器噪声与个体极小值的平坦度或泛化能力联系起来的研究不同，我们展示了有效噪声尺度还影响全局损失景观，预测独立训练的解决方案可以合并的情况。我们的发现开拓了对于优化如何塑造损失景观几何结构及其对模型合并的下游后果的理解，建议进一步操控训练动力学以提高合并效果的可能性。 

---
# TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA 

**Title (ZH)**: TiTok: 通过对比过剩转移Token级知识的洛RA迁移方法 

**Authors**: Chanjoo Jung, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.04682)  

**Abstract**: Large Language Models (LLMs) are widely applied in real world scenarios, but fine-tuning them comes with significant computational and storage costs. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA mitigate these costs, but the adapted parameters are dependent on the base model and cannot be transferred across different backbones. One way to address this issue is through knowledge distillation, but its effectiveness inherently depends on training data. Recent work such as TransLoRA avoids this by generating synthetic data, but this adds complexity because it requires training an additional discriminator model. In this paper, we propose TiTok, a new framework that enables effective LoRA Transplantation through Token-level knowledge transfer. Specifically, TiTok captures task-relevant information through a contrastive excess between a source model with and without LoRA. This excess highlights informative tokens and enables selective filtering of synthetic data, all without additional models or overhead. Through experiments on three benchmarks across multiple transfer settings, our experiments show that the proposed method is consistently effective, achieving average performance gains of +4~8% compared to baselines overall. 

**Abstract (ZH)**: TiTok：基于令牌级知识转移的LoRA移植新框架 

---
# Semantic Channel Equalization Strategies for Deep Joint Source-Channel Coding 

**Title (ZH)**: 深层联合源信道编码中的语义通道均衡策略 

**Authors**: Lorenzo Pannacci, Simone Fiorellino, Mario Edoardo Pandolfo, Emilio Calvanese Strinati, Paolo Di Lorenzo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04674)  

**Abstract**: Deep joint source-channel coding (DeepJSCC) has emerged as a powerful paradigm for end-to-end semantic communications, jointly learning to compress and protect task-relevant features over noisy channels. However, existing DeepJSCC schemes assume a shared latent space at transmitter (TX) and receiver (RX) - an assumption that fails in multi-vendor deployments where encoders and decoders cannot be co-trained. This mismatch introduces "semantic noise", degrading reconstruction quality and downstream task performance. In this paper, we systematize and evaluate methods for semantic channel equalization for DeepJSCC, introducing an additional processing stage that aligns heterogeneous latent spaces under both physical and semantic impairments. We investigate three classes of aligners: (i) linear maps, which admit closed-form solutions; (ii) lightweight neural networks, offering greater expressiveness; and (iii) a Parseval-frame equalizer, which operates in zero-shot mode without the need for training. Through extensive experiments on image reconstruction over AWGN and fading channels, we quantify trade-offs among complexity, data efficiency, and fidelity, providing guidelines for deploying DeepJSCC in heterogeneous AI-native wireless networks. 

**Abstract (ZH)**: 基于深度学习的语义信道均衡方法研究与评估：在异构AI原生无线网络中的部署指南 

---
# FocusMed: A Large Language Model-based Framework for Enhancing Medical Question Summarization with Focus Identification 

**Title (ZH)**: FocusMed: 一种基于大型语言模型的聚焦识别增强医疗问题总结框架 

**Authors**: Chao Liu, Ling Luo, Tengxiao Lv, Huan Zhuang, Lejing Yu, Jian Wang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04671)  

**Abstract**: With the rapid development of online medical platforms, consumer health questions (CHQs) are inefficient in diagnosis due to redundant information and frequent non-professional terms. The medical question summary (MQS) task aims to transform CHQs into streamlined doctors' frequently asked questions (FAQs), but existing methods still face challenges such as poor identification of question focus and model hallucination. This paper explores the potential of large language models (LLMs) in the MQS task and finds that direct fine-tuning is prone to focus identification bias and generates unfaithful content. To this end, we propose an optimization framework based on core focus guidance. First, a prompt template is designed to drive the LLMs to extract the core focus from the CHQs that is faithful to the original text. Then, a fine-tuning dataset is constructed in combination with the original CHQ-FAQ pairs to improve the ability to identify the focus of the question. Finally, a multi-dimensional quality evaluation and selection mechanism is proposed to comprehensively improve the quality of the summary from multiple dimensions. We conduct comprehensive experiments on two widely-adopted MQS datasets using three established evaluation metrics. The proposed framework achieves state-of-the-art performance across all measures, demonstrating a significant boost in the model's ability to identify critical focus of questions and a notable mitigation of hallucinations. The source codes are freely available at this https URL. 

**Abstract (ZH)**: 基于大语言模型的医疗问答摘要优化框架：克服重点识别偏见和减少幻觉 

---
# Noise or Signal? Deconstructing Contradictions and An Adaptive Remedy for Reversible Normalization in Time Series Forecasting 

**Title (ZH)**: 噪声还是信号？拆解时间序列预测中可逆归一化的矛盾并开发自适应修复方法 

**Authors**: Fanzhe Fu, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04667)  

**Abstract**: Reversible Instance Normalization (RevIN) is a key technique enabling simple linear models to achieve state-of-the-art performance in time series forecasting. While replacing its non-robust statistics with robust counterparts (termed R$^2$-IN) seems like a straightforward improvement, our findings reveal a far more complex reality. This paper deconstructs the perplexing performance of various normalization strategies by identifying four underlying theoretical contradictions. Our experiments provide two crucial findings: first, the standard RevIN catastrophically fails on datasets with extreme outliers, where its MSE surges by a staggering 683\%. Second, while the simple R$^2$-IN prevents this failure and unexpectedly emerges as the best overall performer, our adaptive model (A-IN), designed to test a diagnostics-driven heuristic, unexpectedly suffers a complete and systemic failure. This surprising outcome uncovers a critical, overlooked pitfall in time series analysis: the instability introduced by a simple or counter-intuitive heuristic can be more damaging than the statistical issues it aims to solve. The core contribution of this work is thus a new, cautionary paradigm for time series normalization: a shift from a blind search for complexity to a diagnostics-driven analysis that reveals not only the surprising power of simple baselines but also the perilous nature of naive adaptation. 

**Abstract (ZH)**: 可逆实例归一化 (RevIN) 是一种使简单线性模型在时间序列预测中达到最佳性能的关键技术。虽然用稳健统计替换其不稳健的统计（称为 R\(^2\)-IN）似乎是一种简单的改进，但我们的发现揭示了一个更加复杂的现实。本文通过识别四种潜在的理论矛盾，拆解各种归一化策略令人困惑的性能表现。我们的实验提供了两个关键发现：首先，标准 RevIN 在包含极端异常值的数据集上 Catastrophically 失效，其 MSE 突增 683%。其次，虽然简单的 R\(^2\)-IN 防止了这一失败并意外地成为表现最佳的整体模型，我们设计用于测试诊断驱动启发式的自适应模型 (A-IN) 也意外地经历了完全且系统性的失败。这一令人惊讶的结果揭示了时间序列分析中的一个重要且被忽视的陷阱：简单或反直观启发式引入的稳定性问题可能比它试图解决的统计问题更为有害。本文的核心贡献是一种新的、警示性的时间序列归一化范式：从盲目追求复杂性转向基于诊断的分析，不仅揭示了简单基线的惊人力量，还揭示了盲目适应的危险性质。 

---
# Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation 

**Title (ZH)**: 训练驱动以外加速分子几何生成的预测特征缓存 

**Authors**: Johanna Sommer, John Rachwan, Nils Fleischmann, Stephan Günnemann, Bertrand Charpentier  

**Link**: [PDF](https://arxiv.org/pdf/2510.04646)  

**Abstract**: Flow matching models generate high-fidelity molecular geometries but incur significant computational costs during inference, requiring hundreds of network evaluations. This inference overhead becomes the primary bottleneck when such models are employed in practice to sample large numbers of molecular candidates. This work discusses a training-free caching strategy that accelerates molecular geometry generation by predicting intermediate hidden states across solver steps. The proposed method operates directly on the SE(3)-equivariant backbone, is compatible with pretrained models, and is orthogonal to existing training-based accelerations and system-level optimizations. Experiments on the GEOM-Drugs dataset demonstrate that caching achieves a twofold reduction in wall-clock inference time at matched sample quality and a speedup of up to 3x compared to the base model with minimal sample quality degradation. Because these gains compound with other optimizations, applying caching alongside other general, lossless optimizations yield as much as a 7x speedup. 

**Abstract (ZH)**: 无需缓存的训练-free策略通过预测求解步骤中的中间隐状态加速分子几何生成 

---
# SFANet: Spatial-Frequency Attention Network for Deepfake Detection 

**Title (ZH)**: SFANet：空间-频率注意力网络在虚假视频检测中的应用 

**Authors**: Vrushank Ahire, Aniruddh Muley, Shivam Zample, Siddharth Verma, Pranav Menon, Surbhi Madan, Abhinav Dhall  

**Link**: [PDF](https://arxiv.org/pdf/2510.04630)  

**Abstract**: Detecting manipulated media has now become a pressing issue with the recent rise of deepfakes. Most existing approaches fail to generalize across diverse datasets and generation techniques. We thus propose a novel ensemble framework, combining the strengths of transformer-based architectures, such as Swin Transformers and ViTs, and texture-based methods, to achieve better detection accuracy and robustness. Our method introduces innovative data-splitting, sequential training, frequency splitting, patch-based attention, and face segmentation techniques to handle dataset imbalances, enhance high-impact regions (e.g., eyes and mouth), and improve generalization. Our model achieves state-of-the-art performance when tested on the DFWild-Cup dataset, a diverse subset of eight deepfake datasets. The ensemble benefits from the complementarity of these approaches, with transformers excelling in global feature extraction and texturebased methods providing interpretability. This work demonstrates that hybrid models can effectively address the evolving challenges of deepfake detection, offering a robust solution for real-world applications. 

**Abstract (ZH)**: 检测操纵媒体已成为一个紧迫的问题，尤其是在深度假信息的兴起之后。现有大多数方法无法在多种数据集和生成技术之间进行泛化。因此，我们提出了一种新的集成框架，结合了基于变压器的架构（如Swin Transformer和ViT）和纹理基方法的优势，以实现更好的检测准确性和鲁棒性。该方法引入了创新的数据分割、序列训练、频率分割、基于补丁的关注以及面部分割技术，以处理数据集不平衡、增强高影响区域（如眼睛和嘴巴）并提高泛化能力。当在DFWild-Cup数据集中测试时，我们的模型达到了最先进的性能，DFWild-Cup是一个来自八种深度假信息数据集的多样子集。该集成框架得益于这些方法之间的互补性，其中变压器在全局特征提取方面表现出色，而纹理基方法提供了可解释性。本工作证明，混合模型可以有效应对深度假信息检测的不断演变的挑战，并为实际应用提供了稳健的解决方案。 

---
# Fairness in Repeated Matching: A Maximin Perspective 

**Title (ZH)**: 重复匹配中的公平性：最大最小视角 

**Authors**: Eugene Lim, Tzeh Yuan Neoh, Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2510.04624)  

**Abstract**: We study a sequential decision-making model where a set of items is repeatedly matched to the same set of agents over multiple rounds. The objective is to determine a sequence of matchings that either maximizes the utility of the least advantaged agent at the end of all rounds (optimal) or at the end of every individual round (anytime optimal). We investigate the computational challenges associated with finding (anytime) optimal outcomes and demonstrate that these problems are generally computationally intractable. However, we provide approximation algorithms, fixed-parameter tractable algorithms, and identify several special cases whereby the problem(s) can be solved efficiently. Along the way, we also establish characterizations of Pareto-optimal/maximum matchings, which may be of independent interest to works in matching theory and house allocation. 

**Abstract (ZH)**: 我们研究了一种序贯决策模型，其中一组物品在多轮中重复匹配给同一组代理。目标是确定一系列匹配方案，以最大化所有轮次结束后最不利代理的效益（最优）或每轮结束后最不利代理的效益（任意时最优）。我们探讨了找到（任意时）最优结果的计算挑战，并证明了这些问题是通常计算上不可约化的问题。然而，我们提供了近似算法、固定参数可处理算法，并确定了几种可以高效解决的问题特殊情况。在过程中，我们还建立了Pareto最优/最大匹配的特性，这些特性对匹配理论和住房分配领域的研究可能具有独立兴趣。 

---
# Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models 

**Title (ZH)**: 代理情境工程：演化情境以促进自我提升型语言模型 

**Authors**: Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, Kunle Olukotun  

**Link**: [PDF](https://arxiv.org/pdf/2510.04618)  

**Abstract**: Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on context adaptation -- modifying inputs with instructions, strategies, or evidence, rather than weight updates. Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. Building on the adaptive memory introduced by Dynamic Cheatsheet, we introduce ACE (Agentic Context Engineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models. Across agent and domain-specific benchmarks, ACE optimizes contexts both offline (e.g., system prompts) and online (e.g., agent memory), consistently outperforming strong baselines: +10.6% on agents and +8.6% on finance, while significantly reducing adaptation latency and rollout cost. Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller open-source model. These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead. 

**Abstract (ZH)**: 大型语言模型（LLM）应用如代理和领域特定推理日益依赖于上下文适应——通过指令、策略或证据修改输入，而非权重更新。先前的方法在提高使用性的同时，常常遭受摘要偏见，即为了简洁的摘要而牺牲领域见解，以及上下文消解，即迭代重写会随时间逐渐侵蚀细节。基于Dynamic Cheatsheet引入的适应性记忆，我们提出了ACE（Agentic Context Engineering）框架，将上下文视为不断发展的工作手册，通过生成、反思和筛选的过程逐步积累、精炼和组织策略。ACE通过结构化的逐步更新防止消解，保留详细知识并随着长上下文模型扩展。在代理和领域特定基准测试中，ACE在离线（如系统提示）和在线（如代理记忆）上下文优化方面表现优于强劲的基准：代理方面提高10.6%，金融方面提高8.6%，同时显著降低适应延迟和部署成本。值得注意的是，ACE能够有效适应，无需标记监督，而是利用自然执行反馈。在AppWorld排行榜上，ACE在总体平均水平上与排名第一的生产级代理持平，并在更为复杂的测试挑战集上超越了它，尽管使用的是一个较小的开源模型。这些结果表明，全面且不断演化的上下文能够支持拥有低开销的可扩展、高效和自改进的LLM系统。 

---
# Design Process of a Self Adaptive Smart Serious Games Ecosystem 

**Title (ZH)**: 自适应智能严肃游戏生态系统的设计过程 

**Authors**: X. Tao, P. Chen, M. Tsami, F. Khayati, M. Eckert  

**Link**: [PDF](https://arxiv.org/pdf/2510.04615)  

**Abstract**: This paper outlines the design vision and planned evolution of Blexer v3, a modular and AI-driven rehabilitation ecosystem based on serious games. Building on insights from previous versions of the system, we propose a new architecture that aims to integrate multimodal sensing, real-time reasoning, and intelligent control. The envisioned system will include distinct modules for data collection, user state inference, and gameplay adaptation. Key features such as dynamic difficulty adjustment (DDA) and procedural content generation (PCG) are also considered to support personalized interventions. We present the complete conceptual framework of Blexer v3, which defines the modular structure and data flow of the system. This serves as the foundation for the next phase: the development of a functional prototype and its integration into clinical rehabilitation scenarios. 

**Abstract (ZH)**: 基于严肃游戏的可模块化和AI驱动的康复生态系统Blexer v3的设计愿景与规划演化 

---
# Accountability Capture: How Record-Keeping to Support AI Transparency and Accountability (Re)shapes Algorithmic Oversight 

**Title (ZH)**: 责任捕获：记录保存如何塑造AI透明度和问责制下的算法监督 

**Authors**: Shreya Chappidi, Jennifer Cobbe, Chris Norval, Anjali Mazumder, Jatinder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.04609)  

**Abstract**: Accountability regimes typically encourage record-keeping to enable the transparency that supports oversight, investigation, contestation, and redress. However, implementing such record-keeping can introduce considerations, risks, and consequences, which so far remain under-explored. This paper examines how record-keeping practices bring algorithmic systems within accountability regimes, providing a basis to observe and understand their effects. For this, we introduce, describe, and elaborate 'accountability capture' -- the re-configuration of socio-technical processes and the associated downstream effects relating to record-keeping for algorithmic accountability. Surveying 100 practitioners, we evidence and characterise record-keeping issues in practice, identifying their alignment with accountability capture. We further document widespread record-keeping practices, tensions between internal and external accountability requirements, and evidence of employee resistance to practices imposed through accountability capture. We discuss these and other effects for surveillance, privacy, and data protection, highlighting considerations for algorithmic accountability communities. In all, we show that implementing record-keeping to support transparency in algorithmic accountability regimes can itself bring wider implications -- an issue requiring greater attention from practitioners, researchers, and policymakers alike. 

**Abstract (ZH)**: 问责制度通常鼓励记录保存以促进支持监督、调查、争议和补救的透明度。然而，实施这样的记录保存可能会引入考虑因素、风险和后果，这些目前尚未得到充分探索。本文探讨了记录保存实践如何将算法系统纳入问责制度，为观察和理解其影响提供基础。为此，我们引入、描述并阐述了“问责捕获”——即社会技术过程的重新配置及其相关的下游影响，特别是与算法问责相关的记录保存。通过对100名从业人员的调查，我们揭示并刻画了实践中的记录保存问题，发现这些问题与问责捕获相一致。我们进一步记录了广泛存在的记录保存实践、内部和外部问责要求之间的紧张关系以及问责捕获施加的实践所导致的员工抵制证据。我们讨论了这些以及其他对监控、隐私和数据保护的影响，并强调了算法问责共同体的考量。总体而言，我们表明，为了支持算法问责制度中的透明度而实施记录保存本身可能会带来更广泛的影响——一个需要从业者、研究人员和政策制定者共同给予更多关注的问题。 

---
# A Case for Declarative LLM-friendly Interfaces for Improved Efficiency of Computer-Use Agents 

**Title (ZH)**: 声明式LLM友好界面的理由：提高计算机使用代理效率 

**Authors**: Yuan Wang, Mingyu Li, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04607)  

**Abstract**: Computer-use agents (CUAs) powered by large language models (LLMs) have emerged as a promising approach to automating computer tasks, yet they struggle with graphical user interfaces (GUIs). GUIs, designed for humans, force LLMs to decompose high-level goals into lengthy, error-prone sequences of fine-grained actions, resulting in low success rates and an excessive number of LLM calls.
We propose Goal-Oriented Interface (GOI), a novel abstraction that transforms existing GUIs into three declarative primitives: access, state, and observation, which are better suited for LLMs. Our key idea is policy-mechanism separation: LLMs focus on high-level semantic planning (policy) while GOI handles low-level navigation and interaction (mechanism). GOI does not require modifying the application source code or relying on application programming interfaces (APIs).
We evaluate GOI with Microsoft Office Suite (Word, PowerPoint, Excel) on Windows. Compared to a leading GUI-based agent baseline, GOI improves task success rates by 67% and reduces interaction steps by 43.5%. Notably, GOI completes over 61% of successful tasks with a single LLM call. 

**Abstract (ZH)**: 由大语言模型驱动的计算机使用代理（CUAs）已成为自动化计算机任务的一个有前途的方法，但它们在处理图形用户界面（GUIs）方面存在困难。GUIs是为人类设计的，迫使LLMs将高级目标分解为 lengthy、易出错的细粒度操作序列，导致成功率低且LLM调用次数过多。我们提出了目标导向界面（GOI），这是一种新颖的抽象方法，将现有GUIs转换为三个声明式原语：访问、状态和观察，这些原语更适合LLMs。我们的核心思想是策略机制分离：LLMs专注于高级语义规划（策略），而GOI处理低级导航和交互（机制）。GOI无需修改应用程序源代码或依赖应用程序编程接口（APIs）。我们在Windows上使用Microsoft Office Suite（Word、PowerPoint、Excel）评估了GOI。与最先进的基于GUI的代理基线相比，GOI将任务成功率提高了67%，减少了43.5%的交互步骤。值得注意的是，GOI在单次LLM调用中完成了超过61%的成功任务。 

---
# Computing Wasserstein Barycenters through Gradient Flows 

**Title (ZH)**: 通过梯度流计算威劳夫tein 广义中心 

**Authors**: Eduardo Fernandes Montesuma, Yassir Bendou, Mike Gartrell  

**Link**: [PDF](https://arxiv.org/pdf/2510.04602)  

**Abstract**: Wasserstein barycenters provide a powerful tool for aggregating probability measures, while leveraging the geometry of their ambient space. Existing discrete methods suffer from poor scalability, as they require access to the complete set of samples from input measures. We address this issue by recasting the original barycenter problem as a gradient flow in the Wasserstein space. Our approach offers two advantages. First, we achieve scalability by sampling mini-batches from the input measures. Second, we incorporate functionals over probability measures, which regularize the barycenter problem through internal, potential, and interaction energies. We present two algorithms for empirical and Gaussian mixture measures, providing convergence guarantees under the Polyak-Łojasiewicz inequality. Experimental validation on toy datasets and domain adaptation benchmarks show that our methods outperform previous discrete and neural net-based methods for computing Wasserstein barycenters. 

**Abstract (ZH)**: Wasserstein 贴中心提供了一种强大的方法来聚合概率测度，同时利用其环境空间的几何结构。现有的离散方法由于需要访问输入测度的完整样本集而表现出较差的可扩展性。我们通过将原始中心问题重新表述为 Wasserstein 空间的梯度流来解决这一问题。我们的方法有两个优点。首先，我们通过从输入测度中采样小批量实现可扩展性。其次，我们结合了概率测度上的泛函，通过内部、潜在和交互能量对中心问题进行正则化。我们提出了两种算法，分别适用于经验分布和高斯混合测度，并在 Polyak-Łojasiewicz 不等式下提供了收敛性保证。实验验证在玩具数据集和域适应基准测试上表明，我们的方法在计算 Wasserstein 贴中心方面优于之前的离散和神经网络基方法。 

---
# SONA: Learning Conditional, Unconditional, and Mismatching-Aware Discriminator 

**Title (ZH)**: SONA：学习条件依赖、无条件以及 mismating 意识辨别器 

**Authors**: Yuhta Takida, Satoshi Hayakawa, Takashi Shibuya, Masaaki Imaizumi, Naoki Murata, Bac Nguyen, Toshimitsu Uesaka, Chieh-Hsin Lai, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2510.04576)  

**Abstract**: Deep generative models have made significant advances in generating complex content, yet conditional generation remains a fundamental challenge. Existing conditional generative adversarial networks often struggle to balance the dual objectives of assessing authenticity and conditional alignment of input samples within their conditional discriminators. To address this, we propose a novel discriminator design that integrates three key capabilities: unconditional discrimination, matching-aware supervision to enhance alignment sensitivity, and adaptive weighting to dynamically balance all objectives. Specifically, we introduce Sum of Naturalness and Alignment (SONA), which employs separate projections for naturalness (authenticity) and alignment in the final layer with an inductive bias, supported by dedicated objective functions and an adaptive weighting mechanism. Extensive experiments on class-conditional generation tasks show that \ours achieves superior sample quality and conditional alignment compared to state-of-the-art methods. Furthermore, we demonstrate its effectiveness in text-to-image generation, confirming the versatility and robustness of our approach. 

**Abstract (ZH)**: 深度生成模型在生成复杂内容方面取得了显著进展，但条件生成仍然是一个基本挑战。现有的条件生成对抗网络往往难以在判别真实性和条件对齐之间找到平衡。为解决这一问题，我们提出了一种新颖的判别器设计，该设计整合了三项关键能力：无条件判别、增强对齐敏感性的匹配感知监督以及自适应加权以动态平衡所有目标。具体而言，我们引入了自然性和对齐之和（SONA），这是一种在最终层使用独立投影来区分自然性和对齐的方法，并带有专用的目标函数和自适应加权机制。在各类条件生成任务的广泛实验中，我们的方法在样本质量和条件对齐方面优于现有最先进的方法。此外，我们展示了其在文本转图像生成任务中的有效性，证实了该方法的 versatility 和 robustness。 

---
# Deep learning framework for predicting stochastic take-off and die-out of early spreading 

**Title (ZH)**: 基于深度学习的早期传播随机起飞和消亡预测框架 

**Authors**: Wenchao He, Tao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.04574)  

**Abstract**: Large-scale outbreaks of epidemics, misinformation, or other harmful contagions pose significant threats to human society, yet the fundamental question of whether an emerging outbreak will escalate into a major epidemic or naturally die out remains largely unaddressed. This problem is challenging, partially due to inadequate data during the early stages of outbreaks and also because established models focus on average behaviors of large epidemics rather than the stochastic nature of small transmission chains. Here, we introduce the first systematic framework for forecasting whether initial transmission events will amplify into major outbreaks or fade into extinction during early stages, when intervention strategies can still be effectively implemented. Using extensive data from stochastic spreading models, we developed a deep learning framework that predicts early-stage spreading outcomes in real-time. Validation across Erdős-Rényi and Barabási-Albert networks with varying infectivity levels shows our method accurately forecasts stochastic spreading events well before potential outbreaks, demonstrating robust performance across different network structures and infectivity this http URL address the challenge of sparse data during early outbreak stages, we further propose a pretrain-finetune framework that leverages diverse simulation data for pretraining and adapts to specific scenarios through targeted fine-tuning. The pretrain-finetune framework consistently outperforms baseline models, achieving superior performance even when trained on limited scenario-specific data. To our knowledge, this work presents the first framework for predicting stochastic take-off versus die-out. This framework provides valuable insights for epidemic preparedness and public health decision-making, enabling more informed early intervention strategies. 

**Abstract (ZH)**: 大规模传染病、虚假信息或其他有害 contagions的大规模爆发对人类社会构成了重大威胁，但关于新兴爆发是否会升级为大规模流行病还是自然消亡的基本问题仍缺乏解答。为了解决这一挑战，本文首次提出了一种系统框架，在早期阶段预测初始传播事件是否会放大成大规模爆发或逐渐消亡，此时仍可以有效实施干预策略。借助广泛的数据，我们开发了一个深度学习框架，在实际过程中实时预测早期传播结果。对不同传染性水平的Erdős-Rényi和Barabási-Albert网络的验证显示，该方法能够在潜在爆发前准确预测随机传播事件，展示了在不同网络结构和传染性水平下的一致性能。为进一步应对早期爆发阶段数据稀疏的挑战，我们提出了一个预训练-微调框架，利用多样化的模拟数据进行预训练，并通过针对性的微调适应特定场景。预训练-微调框架在各种基准模型中表现优异，即使仅使用有限的场景特定数据也能实现更优性能。据我们所知，本文提出了预测随机起飞与消亡的第一个框架，为传染病准备和公共卫生决策提供了有价值见解，有助于制定更加知情的早期干预策略。 

---
# LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning 

**Title (ZH)**: LaDiR：潜在扩散增强大语言模型的文本推理 

**Authors**: Haoqiang Kang, Yizhe Zhang, Nikki Lijing Kuang, Nicklas Majamaki, Navdeep Jaitly, Yi-An Ma, Lianhui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04573)  

**Abstract**: Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion. 

**Abstract (ZH)**: Large Language Models (LLMs)通过链式思考（CoT）生成展示其推理能力。然而，LLM的自回归解码可能会限制其整体 revisit 和改进早期生成的标记的能力，这也可能导致多样解的不高效探索。在本文中，我们提出了一种新颖的推理框架LaDiR（潜在扩散推理器），该框架结合了连续潜在表示的表达能力和潜在扩散模型的迭代改进能力，应用于现有的LLM。我们首先使用变分自编码器（VAE）构建结构化的潜在推理空间，将文本推理步骤编码为想法标记块，同时保留语义信息和可解释性，提供紧凑但表达能力强的表示。随后，我们利用一个潜在扩散模型，通过块化双向注意力掩码学习去除潜在想法标记块中的噪声，从而实现更长的时间跨度和适应性测试时计算的迭代改进。此设计允许高效并行生成多样推理轨迹，使模型能够整体规划和修订推理过程。我们通过对一系列数学推理和规划基准的评估。实验结果表明，LaDiR在准确性、多样性和可解释性方面均优于现有的自回归、基于扩散和潜在推理方法，揭示了一种新的文本推理潜在扩散范式。 

---
# GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning 

**Title (ZH)**: GILT：一种无需大型语言模型且无需调优的图基础模型用于上下文学习 

**Authors**: Weishuo Ma, Yanbo Wang, Xiyuan Wang, Lei Zou, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04567)  

**Abstract**: Graph Neural Networks (GNNs) are powerful tools for precessing relational data but often struggle to generalize to unseen graphs, giving rise to the development of Graph Foundational Models (GFMs). However, current GFMs are challenged by the extreme heterogeneity of graph data, where each graph can possess a unique feature space, label set, and topology. To address this, two main paradigms have emerged. The first leverages Large Language Models (LLMs), but is fundamentally text-dependent, thus struggles to handle the numerical features in vast graphs. The second pre-trains a structure-based model, but the adaptation to new tasks typically requires a costly, per-graph tuning stage, creating a critical efficiency bottleneck. In this work, we move beyond these limitations and introduce \textbf{G}raph \textbf{I}n-context \textbf{L}earning \textbf{T}ransformer (GILT), a framework built on an LLM-free and tuning-free architecture. GILT introduces a novel token-based framework for in-context learning (ICL) on graphs, reframing classification tasks spanning node, edge and graph levels in a unified framework. This mechanism is the key to handling heterogeneity, as it is designed to operate on generic numerical features. Further, its ability to understand class semantics dynamically from the context enables tuning-free adaptation. Comprehensive experiments show that GILT achieves stronger few-shot performance with significantly less time than LLM-based or tuning-based baselines, validating the effectiveness of our approach. 

**Abstract (ZH)**: Graph In-context Learning Transformer (GILT): A Framework for Efficient and Adaptive Graph Learning 

---
# 3Dify: a Framework for Procedural 3D-CG Generation Assisted by LLMs Using MCP and RAG 

**Title (ZH)**: 3Dify：由MCP和RAG辅助的基于LLM的 procedual 3D-CG生成框架 

**Authors**: Shun-ichiro Hayashi, Daichi Mukunoki, Tetsuya Hoshino, Satoshi Ohshima, Takahiro Katagiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.04536)  

**Abstract**: This paper proposes "3Dify," a procedural 3D computer graphics (3D-CG) generation framework utilizing Large Language Models (LLMs). The framework enables users to generate 3D-CG content solely through natural language instructions. 3Dify is built upon Dify, an open-source platform for AI application development, and incorporates several state-of-the-art LLM-related technologies such as the Model Context Protocol (MCP) and Retrieval-Augmented Generation (RAG). For 3D-CG generation support, 3Dify automates the operation of various Digital Content Creation (DCC) tools via MCP. When DCC tools do not support MCP-based interaction, the framework employs the Computer-Using Agent (CUA) method to automate Graphical User Interface (GUI) operations. Moreover, to enhance image generation quality, 3Dify allows users to provide feedback by selecting preferred images from multiple candidates. The LLM then learns variable patterns from these selections and applies them to subsequent generations. Furthermore, 3Dify supports the integration of locally deployed LLMs, enabling users to utilize custom-developed models and to reduce both time and monetary costs associated with external API calls by leveraging their own computational resources. 

**Abstract (ZH)**: 基于大型语言模型的“3Dify”程序化3D计算机图形生成框架 

---
# Unified Threat Detection and Mitigation Framework (UTDMF): Combating Prompt Injection, Deception, and Bias in Enterprise-Scale Transformers 

**Title (ZH)**: 统一威胁检测与缓解框架（UTDMF）：应对企业规模变换器中的提示注入、欺诈和偏见 

**Authors**: Santhosh KumarRavindran  

**Link**: [PDF](https://arxiv.org/pdf/2510.04528)  

**Abstract**: The rapid adoption of large language models (LLMs) in enterprise systems exposes vulnerabilities to prompt injection attacks, strategic deception, and biased outputs, threatening security, trust, and fairness. Extending our adversarial activation patching framework (arXiv:2507.09406), which induced deception in toy networks at a 23.9% rate, we introduce the Unified Threat Detection and Mitigation Framework (UTDMF), a scalable, real-time pipeline for enterprise-grade models like Llama-3.1 (405B), GPT-4o, and Claude-3.5. Through 700+ experiments per model, UTDMF achieves: (1) 92% detection accuracy for prompt injection (e.g., jailbreaking); (2) 65% reduction in deceptive outputs via enhanced patching; and (3) 78% improvement in fairness metrics (e.g., demographic bias). Novel contributions include a generalized patching algorithm for multi-threat detection, three groundbreaking hypotheses on threat interactions (e.g., threat chaining in enterprise workflows), and a deployment-ready toolkit with APIs for enterprise integration. 

**Abstract (ZH)**: 统一威胁检测与缓解框架（UTDMF）：一种面向企业级模型的大规模实时管道，用于检测和缓解大规模语言模型中的漏洞 

---
# Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction 

**Title (ZH)**: 统一几何理解的方向：黎曼扩散框架下的图生成与预测 

**Authors**: Yisen Gao, Xingcheng Fu, Qingyun Sun, Jianxin Li, Xianxian Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.04522)  

**Abstract**: Graph diffusion models have made significant progress in learning structured graph data and have demonstrated strong potential for predictive tasks. Existing approaches typically embed node, edge, and graph-level features into a unified latent space, modeling prediction tasks including classification and regression as a form of conditional generation. However, due to the non-Euclidean nature of graph data, features of different curvatures are entangled in the same latent space without releasing their geometric potential. To address this issue, we aim to construt an ideal Riemannian diffusion model to capture distinct manifold signatures of complex graph data and learn their distribution. This goal faces two challenges: numerical instability caused by exponential mapping during the encoding proces and manifold deviation during diffusion generation. To address these challenges, we propose GeoMancer: a novel Riemannian graph diffusion framework for both generation and prediction tasks. To mitigate numerical instability, we replace exponential mapping with an isometric-invariant Riemannian gyrokernel approach and decouple multi-level features onto their respective task-specific manifolds to learn optimal representations. To address manifold deviation, we introduce a manifold-constrained diffusion method and a self-guided strategy for unconditional generation, ensuring that the generated data remains aligned with the manifold signature. Extensive experiments validate the effectiveness of our approach, demonstrating superior performance across a variety of tasks. 

**Abstract (ZH)**: 基于黎曼流形的图扩散模型：一种新的图扩散框架以捕捉复杂图数据的 manifold 特征并进行生成与预测任务 

---
# GRACE: Generative Representation Learning via Contrastive Policy Optimization 

**Title (ZH)**: GRACE: 生成表示学习 via 对抗性策略优化 

**Authors**: Jiashuo Sun, Shixuan Liu, Zhaochen Su, Xianrui Zhong, Pengcheng Jiang, Bowen Jin, Peiran Li, Weijia Shi, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.04506)  

**Abstract**: Prevailing methods for training Large Language Models (LLMs) as text encoders rely on contrastive losses that treat the model as a black box function, discarding its generative and reasoning capabilities in favor of static embeddings. We introduce GRACE (Generative Representation Learning via Contrastive Policy Optimization), a novel framework that reimagines contrastive signals not as losses to be minimized, but as rewards that guide a generative policy. In GRACE, the LLM acts as a policy that produces explicit, human-interpretable rationales--structured natural language explanations of its semantic understanding. These rationales are then encoded into high-quality embeddings via mean pooling. Using policy gradient optimization, we train the model with a multi-component reward function that maximizes similarity between query positive pairs and minimizes similarity with negatives. This transforms the LLM from an opaque encoder into an interpretable agent whose reasoning process is transparent and inspectable. On MTEB benchmark, GRACE yields broad cross category gains: averaged over four backbones, the supervised setting improves overall score by 11.5% over base models, and the unsupervised variant adds 6.9%, while preserving general capabilities. This work treats contrastive objectives as rewards over rationales, unifying representation learning with generation to produce stronger embeddings and transparent rationales. The model, data and code are available at this https URL. 

**Abstract (ZH)**: 基于对比策略优化的学习生成表示的大型语言模型训练方法 

---
# P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs 

**Title (ZH)**: P2P：一种针对LLMs可靠后门防御的毒药-毒药方案 

**Authors**: Shuai Zhao, Xinyi Wu, Shiqian Zhao, Xiaobao Wu, Zhongliang Guo, Yanhao Jia, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04503)  

**Abstract**: During fine-tuning, large language models (LLMs) are increasingly vulnerable to data-poisoning backdoor attacks, which compromise their reliability and trustworthiness. However, existing defense strategies suffer from limited generalization: they only work on specific attack types or task settings. In this study, we propose Poison-to-Poison (P2P), a general and effective backdoor defense algorithm. P2P injects benign triggers with safe alternative labels into a subset of training samples and fine-tunes the model on this re-poisoned dataset by leveraging prompt-based learning. This enforces the model to associate trigger-induced representations with safe outputs, thereby overriding the effects of original malicious triggers. Thanks to this robust and generalizable trigger-based fine-tuning, P2P is effective across task settings and attack types. Theoretically and empirically, we show that P2P can neutralize malicious backdoors while preserving task performance. We conduct extensive experiments on classification, mathematical reasoning, and summary generation tasks, involving multiple state-of-the-art LLMs. The results demonstrate that our P2P algorithm significantly reduces the attack success rate compared with baseline models. We hope that the P2P can serve as a guideline for defending against backdoor attacks and foster the development of a secure and trustworthy LLM community. 

**Abstract (ZH)**: 大语言模型在微调过程中越来越容易受到数据中毒后门攻击的影响，这损害了其可靠性和可信度。现有防御策略存在泛化能力有限的问题：它们仅针对特定的攻击类型或任务设置生效。在本研究中，我们提出了Poison-to-Poison (P2P)，一种通用且有效的后门防御算法。P2P在训练样本的一部分中注入具有安全替代标签的良性触发器，并通过基于提示的学习重新训练模型，从而强制模型将触发器诱导的表示与安全输出关联起来，从而抵消原始恶意触发器的效果。由于这种鲁棒且通用的基于触发器的微调，P2P在不同的任务设置和攻击类型下均有效。我们从理论上和实证上都证明，P2P可以在保护任务性能的同时消除恶意后门。我们在分类、数学推理和摘要生成等多种任务中使用了多个最先进的大语言模型进行了广泛的实验。结果表明，与基线模型相比，我们的P2P算法显著降低了攻击的成功率。我们希望P2P能够作为防御后门攻击的指南，并促进安全和可信的大语言模型社区的发展。 

---
# GenQuest: An LLM-based Text Adventure Game for Language Learners 

**Title (ZH)**: GenQuest: 一种基于LLM的文本冒险游戏，用于语言学习者 

**Authors**: Qiao Wang, Adnan Labib, Robert Swier, Michael Hofmeyr, Zheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04498)  

**Abstract**: GenQuest is a generative text adventure game that leverages Large Language Models (LLMs) to facilitate second language learning through immersive, interactive storytelling. The system engages English as a Foreign Language (EFL) learners in a collaborative "choose-your-own-adventure" style narrative, dynamically generated in response to learner choices. Game mechanics such as branching decision points and story milestones are incorporated to maintain narrative coherence while allowing learner-driven plot development. Key pedagogical features include content generation tailored to each learner's proficiency level, and a vocabulary assistant that provides in-context explanations of learner-queried text strings, ranging from words and phrases to sentences. Findings from a pilot study with university EFL students in China indicate promising vocabulary gains and positive user perceptions. Also discussed are suggestions from participants regarding the narrative length and quality, and the request for multi-modal content such as illustrations. 

**Abstract (ZH)**: GenQuest 是一种利用大型语言模型（LLMs）通过沉浸式交互叙事促进第二语言学习的生成性文本冒险游戏。系统采用英语作为外语（EFL）学习者参与协作式的“自主冒险”风格叙述，并根据学习者的选择动态生成。游戏机制如分支决策点和故事情节里程碑的融入，维持叙事连贯性同时允许以学习者为导向的剧情发展。关键的教学功能包括根据每个学习者的 proficiency 级别定制的内容生成，以及词汇助手，该助手提供上下文解释，涵盖从词汇和短语到句子的各种文本字符串。初步研究中对中国大学 EFL 学生的试验结果表明词汇量增长和积极的用户感知。还讨论了参与者关于故事情节长度和质量的建议，以及对包括插图在内的多媒体内容的要求。 

---
# Psychological Steering in LLMs: An Evaluation of Effectiveness and Trustworthiness 

**Title (ZH)**: LLMs中的心理导向：有效性与可信度评估 

**Authors**: Amin Banayeeanzade, Ala N. Tak, Fatemeh Bahrani, Anahita Bolourani, Leonardo Blas, Emilio Ferrara, Jonathan Gratch, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.04484)  

**Abstract**: The ability to control LLMs' emulated emotional states and personality traits is essential for enabling rich, human-centered interactions in socially interactive settings. We introduce PsySET, a Psychologically-informed benchmark to evaluate LLM Steering Effectiveness and Trustworthiness across the emotion and personality domains. Our study spans four models from different LLM families paired with various steering strategies, including prompting, fine-tuning, and representation engineering. Our results indicate that prompting is consistently effective but limited in intensity control, whereas vector injections achieve finer controllability while slightly reducing output quality. Moreover, we explore the trustworthiness of steered LLMs by assessing safety, truthfulness, fairness, and ethics, highlighting potential side effects and behavioral shifts. Notably, we observe idiosyncratic effects; for instance, even a positive emotion like joy can degrade robustness to adversarial factuality, lower privacy awareness, and increase preferential bias. Meanwhile, anger predictably elevates toxicity yet strengthens leakage resistance. Our framework establishes the first holistic evaluation of emotion and personality steering, offering insights into its interpretability and reliability for socially interactive applications. 

**Abstract (ZH)**: 控制大模型模拟的情感状态和个性特征的能力对于在社会互动环境中实现丰富的人本交互至关重要。我们引入了PsySET，一种基于心理因素的标准，用于评估情感和个性领域大模型引导效果的有效性和可信度。我们的研究涵盖了不同大模型家族的四个模型，配以各种引导策略，包括提示、微调和表征工程。结果表明，提示在有效性上是一贯的，但在强度控制上有限，而向量注入在实现更精细的可控性的同时略微降低了输出质量。此外，我们通过评估安全性、真实性、公平性和伦理标准来探索引导后的大型语言模型的可信度，揭示了潜在的副作用和行为变化。值得注意的是，我们观察到一些特异性效果；例如，即使是正面情绪如快乐也可能降低对抗事实的鲁棒性、降低隐私意识并增加偏好偏差。同时，愤怒可预测地提升了毒性但增强了泄漏防御。我们的框架建立了情感和个性引导的首个综合性评估，为社会互动应用提供了关于其可解释性和可靠性的见解。 

---
# MedCLM: Learning to Localize and Reason via a CoT-Curriculum in Medical Vision-Language Models 

**Title (ZH)**: MedCLM: 在医疗视觉语言模型中通过CoT课程学习定位与推理 

**Authors**: Soo Yong Kim, Suin Cho, Vincent-Daniel Yun, Gyeongyeon Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04477)  

**Abstract**: Bridging clinical diagnostic reasoning with AI remains a central challenge in medical imaging. We introduce MedCLM, an automated pipeline that converts detection datasets into large-scale medical visual question answering (VQA) data with Chain-of-Thought (CoT) reasoning by linking lesion boxes to organ segmentation and structured rationales. These contextual signals enable medical vision-language models to generate question-answer pairs with step-by-step reasoning. To utilize this data effectively, we propose an Integrated CoT-Curriculum Strategy composed of an Easy stage with explicit lesion boxes for visual grounding, a Medium stage that encourages implicit localization, and a Hard stage for weakly supervised reasoning. Experimental results demonstrate that MedCLM attains state-of-the-art performance on several medical VQA benchmarks, providing a scalable framework for developing clinically aligned medical vision-language models. 

**Abstract (ZH)**: 将临床诊断推理与AI相结合仍然是医学影像领域的一个核心挑战。我们介绍了MedCLM，这是一种自动化流水线，通过将病灶框与器官分割和结构化理由关联，将检测数据集转换为大规模的医学视觉问答(VQA)数据，并通过Chain-of-Thought (CoT)推理。这些上下文信号使医学视觉-语言模型能够生成带有逐步推理的问题-答案对。为了有效利用这些数据，我们提出了一种综合的CoT-curriculum策略，包括一个简单阶段，用于明确定义病灶框以进行视觉定位，一个鼓励隐式定位的中等阶段，以及一个弱监督推理的困难阶段。实验结果表明，MedCLM在多个医学VQA基准测试中取得了最先进的性能，提供了一个可扩展的框架，用于开发与临床标准对齐的医学视觉-语言模型。 

---
# Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space 

**Title (ZH)**: 压缩卷积注意力：在压缩潜空间中的高效注意力 

**Authors**: Tomas Figliolia, Nicholas Alonso, Rishi Iyer, Quentin Anthony, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2510.04476)  

**Abstract**: Multi-headed Attention's (MHA) quadratic compute and linearly growing KV-cache make long-context transformers expensive to train and serve. Prior works such as Grouped Query Attention (GQA) and Multi-Latent Attention (MLA) shrink the cache, speeding decode, but leave compute, which determines prefill and training speed, largely unchanged. We introduce Compressed Convolutional Attention (CCA), a novel attention method which down-projects queries, keys, and values and performs the entire attention operation inside the shared latent space. This simple design dramatically cuts parameters, KV-cache, and FLOPs all at once by the desired compression factor. Because CCA is orthogonal to head-sharing, we combine the two to form Compressed Convolutional Grouped Query Attention (CCGQA), which further tightens the compute-bandwidth Pareto frontier so that users can tune compression toward either FLOP or memory limits without sacrificing quality. Experiments show that CCGQA consistently outperforms both GQA and MLA at equal KV-cache compression on dense and MoE models. Additionally, we show that CCGQA outperforms all other attention methods on MoE models with half the KV-cache of GQA and MLA, achieving an 8x KV-cache compression with no drop in performance compared to standard MHA. CCA and CCGQA also dramatically reduce the FLOP cost of attention which leads to substantially faster training and prefill than existing methods. On H100 GPUs, our fused CCA/CCGQA kernel reduces prefill latency by about 1.7x at a sequence length of 16k relative to MHA, and accelerates backward by about 1.3x. 

**Abstract (ZH)**: Compressed Convolutional Attention (CCA)及其在长上下文变换器训练和服务中的应用 

---
# SPEGNet: Synergistic Perception-Guided Network for Camouflaged Object Detection 

**Title (ZH)**: SPEGNet: 协同感知引导网络在伪装目标检测中的应用 

**Authors**: Baber Jan, Saeed Anwar, Aiman H. El-Maleh, Abdul Jabbar Siddiqui, Abdul Bais  

**Link**: [PDF](https://arxiv.org/pdf/2510.04472)  

**Abstract**: Camouflaged object detection segments objects with intrinsic similarity and edge disruption. Current detection methods rely on accumulated complex components. Each approach adds components such as boundary modules, attention mechanisms, and multi-scale processors independently. This accumulation creates a computational burden without proportional gains. To manage this complexity, they process at reduced resolutions, eliminating fine details essential for camouflage. We present SPEGNet, addressing fragmentation through a unified design. The architecture integrates multi-scale features via channel calibration and spatial enhancement. Boundaries emerge directly from context-rich representations, maintaining semantic-spatial alignment. Progressive refinement implements scale-adaptive edge modulation with peak influence at intermediate resolutions. This design strikes a balance between boundary precision and regional consistency. SPEGNet achieves 0.887 $S_\alpha$ on CAMO, 0.890 on COD10K, and 0.895 on NC4K, with real-time inference speed. Our approach excels across scales, from tiny, intricate objects to large, pattern-similar ones, while handling occlusion and ambiguous boundaries. Code, model weights, and results are available on \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 伪装目标检测通过内在相似性和边缘干扰分割目标。当前的检测方法依赖于累积复杂组件。每种方法独立添加边界模块、注意力机制和多尺度处理器等组件。这种累积带来了计算负担，但没有相应的成效提升。为应对这种复杂性，它们在降低分辨率下进行处理，从而消除对伪装至关重要的细节数。我们提出SPEGNet，通过统一设计解决碎片化问题。该架构通过通道校准和空间增强整合多尺度特征。边缘直接源自丰富语境的表示，保持语义-空间对齐。渐进精炼实施尺度自适应边缘调制，在中间分辨率处达到峰值影响。此设计在边界精度和区域一致性之间达到平衡。SPEGNet在CAMO上取得0.887的$S_\alpha$，在COD10K上取得0.890，在NC4K上取得0.895，同时支持实时推理速度。我们的方法在不同尺度上表现出色，从精细的小目标到相似的大目标，同时处理遮挡和含糊的边界。代码、模型权重和结果可在<此链接>获得。 

---
# Autonomy Matters: A Study on Personalization-Privacy Dilemma in LLM Agents 

**Title (ZH)**: 自主性重要：关于LLM代理中的个性化-隐私困境研究 

**Authors**: Zhiping Zhang, Yi Evie Zhang, Freda Shi, Tianshi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.04465)  

**Abstract**: Large Language Model (LLM) agents require personal information for personalization in order to better act on users' behalf in daily tasks, but this raises privacy concerns and a personalization-privacy dilemma. Agent's autonomy introduces both risks and opportunities, yet its effects remain unclear. To better understand this, we conducted a 3$\times$3 between-subjects experiment ($N=450$) to study how agent's autonomy level and personalization influence users' privacy concerns, trust and willingness to use, as well as the underlying psychological processes. We find that personalization without considering users' privacy preferences increases privacy concerns and decreases trust and willingness to use. Autonomy moderates these effects: Intermediate autonomy flattens the impact of personalization compared to No- and Full autonomy conditions. Our results suggest that rather than aiming for perfect model alignment in output generation, balancing autonomy of agent's action and user control offers a promising path to mitigate the personalization-privacy dilemma. 

**Abstract (ZH)**: 大规模语言模型代理需要个人信息以更好地代理用户完成日常任务，但这也引发了隐私问题和个人化-隐私困境。代理的自主性既带来风险也带来机会，但其影响尚不清楚。为了解这个问题，我们进行了一个3×3被试间实验（N=450），以研究代理的自主性水平和个人化如何影响用户的隐私担忧、信任和使用意愿，以及潜在的心理过程。研究发现，在不考虑用户隐私偏好的情况下进行个人化会增加隐私担忧并降低信任和使用意愿。自主性的程度调节了这些影响：中等自主性在与高自主性和无自主性条件下相比时，降低了个人化的影响。我们的结果表明，与其在输出生成中追求完美的模型对齐，平衡代理行动的自主性和用户控制可能是一条缓解个人化-隐私困境的有希望的途径。 

---
# Inverse Mixed-Integer Programming: Learning Constraints then Objective Functions 

**Title (ZH)**: 逆混合整数规划：学习约束条件然后目标函数 

**Authors**: Akira Kitaoka  

**Link**: [PDF](https://arxiv.org/pdf/2510.04455)  

**Abstract**: In mixed-integer linear programming, data-driven inverse optimization that learns the objective function and the constraints from observed data plays an important role in constructing appropriate mathematical models for various fields, including power systems and scheduling. However, to the best of our knowledge, there is no known method for learning both the objective functions and the constraints. In this paper, we propose a two-stage method for a class of problems where the objective function is expressed as a linear combination of functions and the constraints are represented by functions and thresholds. Specifically, our method first learns the constraints and then learns the objective function. On the theoretical side, we show the proposed method can solve inverse optimization problems in finite dataset, develop statistical learning theory in pseudometric spaces and sub-Gaussian distributions, and construct a statistical learning for inverse optimization. On the experimental side, we demonstrate that our method is practically applicable for scheduling problems formulated as integer linear programmings with up to 100 decision variables, which are typical in real-world settings. 

**Abstract (ZH)**: 在混合整数线性规划中，基于数据的逆优化方法通过从观察数据中学习目标函数和约束条件，在构建适用于电力系统和调度等领域的方法模型方面发挥着重要作用。然而，据我们所知，尚无已知方法能够同时学习目标函数和约束条件。本文提出一种两阶段方法，适用于目标函数表示为函数线性组合、约束表示为函数和阈值的问题类。具体而言，该方法首先学习约束，然后学习目标函数。从理论层面看，我们展示了所提出方法能够在有限数据集上解决逆优化问题，建立了伪距离空间和亚高斯分布下的统计学习理论，并构建了逆优化的统计学习方法。从实验层面看，我们证明了该方法对于包含多达100个决策变量的整数线性规划调度问题具有实际应用价值，这类问题是现实中常见的。 

---
# Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions 

**Title (ZH)**: 局部信息分解：通过潜高斯分布中的规范化流 

**Authors**: Wenyuan Zhao, Adithya Balachandran, Chao Tian, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04417)  

**Abstract**: The study of multimodality has garnered significant interest in fields where the analysis of interactions among multiple information sources can enhance predictive modeling, data fusion, and interpretability. Partial information decomposition (PID) has emerged as a useful information-theoretic framework to quantify the degree to which individual modalities independently, redundantly, or synergistically convey information about a target variable. However, existing PID methods depend on optimizing over a joint distribution constrained by estimated pairwise probability distributions, which are costly and inaccurate for continuous and high-dimensional modalities. Our first key insight is that the problem can be solved efficiently when the pairwise distributions are multivariate Gaussians, and we refer to this problem as Gaussian PID (GPID). We propose a new gradient-based algorithm that substantially improves the computational efficiency of GPID based on an alternative formulation of the underlying optimization problem. To generalize the applicability to non-Gaussian data, we learn information-preserving encoders to transform random variables of arbitrary input distributions into pairwise Gaussian random variables. Along the way, we resolved an open problem regarding the optimality of joint Gaussian solutions for GPID. Empirical validation in diverse synthetic examples demonstrates that our proposed method provides more accurate and efficient PID estimates than existing baselines. We further evaluate a series of large-scale multimodal benchmarks to show its utility in real-world applications of quantifying PID in multimodal datasets and selecting high-performing models. 

**Abstract (ZH)**: 多模态研究在分析多种信息源的交互以增强预测建模、数据融合和可解释性方面引起了广泛关注。部分信息分解（PID）已成为一种有用的信息论框架，用于量化各个模态独立、冗余或协同地传递目标变量信息的程度。然而，现有的PID方法依赖于在估计的成对概率分布约束下的联合分布优化，这对于连续和高维模态来说成本高且不准确。我们的第一项关键洞察是，当成对分布为多元高斯分布时，该问题可以高效解决，并将此问题称为高斯PID（GPID）。我们提出了一种新的梯度基于算法，该算法在底层优化问题的替代公式基础上显著提高了GPID的计算效率。为了使GPID适用于非高斯数据，我们学习了保信息的编码器，将任意输入分布的随机变量转换为成对的高斯随机变量。在这一过程中，我们解决了GPID中联合高斯解的最优性问题。在多元合成数据示例上的实证验证表明，我们提出的方法提供了比现有基线更准确和高效的PID估计。我们进一步评估了一系列大规模多模态基准，展示了其在多模态数据集中量化PID和选择高性能模型的实际应用中的实用性。 

---
# Your Vision-Language Model Can't Even Count to 20: Exposing the Failures of VLMs in Compositional Counting 

**Title (ZH)**: 你的视觉语言模型连20都数不到：揭示VLMs在组合计数中的失败 

**Authors**: Xuyang Guo, Zekai Huang, Zhenmei Shi, Zhao Song, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04401)  

**Abstract**: Vision-Language Models (VLMs) have become a central focus of today's AI community, owing to their impressive abilities gained from training on large-scale vision-language data from the Web. These models have demonstrated strong performance across diverse tasks, including image understanding, video understanding, complex visual reasoning, and embodied AI. Despite these noteworthy successes, a fundamental question remains: Can VLMs count objects correctly? In this paper, we introduce a simple yet effective benchmark, VLMCountBench, designed under a minimalist setting with only basic geometric shapes (e.g., triangles, circles) and their compositions, focusing exclusively on counting tasks without interference from other factors. We adopt strict independent variable control and systematically study the effects of simple properties such as color, size, and prompt refinement in a controlled ablation. Our empirical results reveal that while VLMs can count reliably when only one shape type is present, they exhibit substantial failures when multiple shape types are combined (i.e., compositional counting). This highlights a fundamental empirical limitation of current VLMs and motivates important directions for future research. 

**Abstract (ZH)**: Vision-Language模型(VLMs)的能力源于大规模网络数据的训练，在当今的人工智能社区中已成为核心研究焦点。尽管这些模型在图像理解、视频理解、复杂视觉推理和具身人工智能等多样任务中表现出色，但一个基础问题仍然存在：VLMs能否准确计数物体？在本文中，我们提出了一个简单有效的基准VLMCountBench，在仅包含基本几何形状及其组合的简单设置下专注于计数任务，不受到其他因素的干扰。我们采用严格的独立变量控制，并在受控消融实验中系统研究了颜色、大小和提示优化等简单属性的影响。实验结果表明，当只有一种形状类型时，VLMs能够可靠地计数，但在多种形状类型组合（即组合计数）的情况下表现出显著的失败，这揭示了当前VLMs的一个基本经验限制，并为未来的研究指明了重要方向。 

---
# Large Language Models Preserve Semantic Isotopies in Story Continuations 

**Title (ZH)**: 大规模语言模型在故事续写中保维持语义同义性 

**Authors**: Marc Cavazza  

**Link**: [PDF](https://arxiv.org/pdf/2510.04400)  

**Abstract**: In this work, we explore the relevance of textual semantics to Large Language Models (LLMs), extending previous insights into the connection between distributional semantics and structural semantics. We investigate whether LLM-generated texts preserve semantic isotopies. We design a story continuation experiment using 10,000 ROCStories prompts completed by five LLMs. We first validate GPT-4o's ability to extract isotopies from a linguistic benchmark, then apply it to the generated stories. We then analyze structural (coverage, density, spread) and semantic properties of isotopies to assess how they are affected by completion. Results show that LLM completion within a given token horizon preserves semantic isotopies across multiple properties. 

**Abstract (ZH)**: 在本文中，我们探讨了文本语义与大规模语言模型（LLMs）的相关性，扩展了关于分布语义与结构语义之间联系的先前见解。我们调查LLM生成的文本是否保留了语义同化。我们使用5个LLM完成的10,000个ROCStories提示设计了一个故事续写实验。我们首先验证了GPT-4o从语言基准中提取语义同化的能力，然后将其应用于生成的故事。我们随后分析语义同化的结构（覆盖率、密度、分布）和语义属性，以评估这些属性如何受到生成的影响。结果表明，给定标记窗口内的LLM生成保留了多种属性下的语义同化。 

---
# SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations 

**Title (ZH)**: SECA：语义等价且一致的攻击以诱发型LLM幻觉 

**Authors**: Buyun Liang, Liangzu Peng, Jinqi Luo, Darshan Thaker, Kwan Ho Ryan Chan, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2510.04398)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no constraint violations compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益应用于高风险领域。然而，最新颖的LLMs常常会产生幻觉，这对其可靠性提出了严重质疑。先前的工作已经探索了针对LLMs幻觉诱发的对抗攻击，但这些方法往往会产生不现实的提示，要么插入无意义的标记，要么改变原始含义。因此，这些方法提供了有限的关于幻觉在实践中如何产生的见解。虽然计算机视觉中的对抗攻击往往涉及对输入图像的现实修改，但寻找能够诱发LLM幻觉的现实对抗提示依然 largely 没有得到充分探索。为解决这一问题，我们提出了语义等价且连贯的攻击（SECA），通过保留提示意义的同时维持语义连贯性来进行现实修改以诱发幻觉。我们的贡献包括三个方面：（i）我们将寻找现实对抗攻击以诱发幻觉的问题形式化为在语义等价和连贯性约束下的输入提示空间的约束优化问题；（ii）我们引入了一种约束保持的零阶方法，以有效地寻找对抗但仍可行的提示；（iii）通过在开放式多项选择问答任务上的实验证明，与现有方法相比，SECA达到了更高的攻击成功率，并且几乎没有任何约束违背。SECA强调了开源和商业梯度不可访问的大规模语言模型对现实和合理的提示变化的高度敏感性。代码详见这个 https URL。 

---
# MulVuln: Enhancing Pre-trained LMs with Shared and Language-Specific Knowledge for Multilingual Vulnerability Detection 

**Title (ZH)**: MulVuln: 增强多语言漏洞检测的预训练模型，融入共享和语言特定知识 

**Authors**: Van Nguyen, Surya Nepal, Xingliang Yuan, Tingmin Wu, Fengchao Chen, Carsten Rudolph  

**Link**: [PDF](https://arxiv.org/pdf/2510.04397)  

**Abstract**: Software vulnerabilities (SVs) pose a critical threat to safety-critical systems, driving the adoption of AI-based approaches such as machine learning and deep learning for software vulnerability detection. Despite promising results, most existing methods are limited to a single programming language. This is problematic given the multilingual nature of modern software, which is often complex and written in multiple languages. Current approaches often face challenges in capturing both shared and language-specific knowledge of source code, which can limit their performance on diverse programming languages and real-world codebases. To address this gap, we propose MULVULN, a novel multilingual vulnerability detection approach that learns from source code across multiple languages. MULVULN captures both the shared knowledge that generalizes across languages and the language-specific knowledge that reflects unique coding conventions. By integrating these aspects, it achieves more robust and effective detection of vulnerabilities in real-world multilingual software systems. The rigorous and extensive experiments on the real-world and diverse REEF dataset, consisting of 4,466 CVEs with 30,987 patches across seven programming languages, demonstrate the superiority of MULVULN over thirteen effective and state-of-the-art baselines. Notably, MULVULN achieves substantially higher F1-score, with improvements ranging from 1.45% to 23.59% compared to the baseline methods. 

**Abstract (ZH)**: 多语言软件漏洞检测方法：MULVULN 

---
# Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards 

**Title (ZH)**: 通过组相似性奖励提高检索增强系统的一致性 

**Authors**: Faisal Hamman, Chenyang Zhu, Anoop Kumar, Xujun Peng, Sanghamitra Dutta, Daben Liu, Alfy Samuel  

**Link**: [PDF](https://arxiv.org/pdf/2510.04392)  

**Abstract**: RAG systems are increasingly deployed in high-stakes domains where users expect outputs to be consistent across semantically equivalent queries. However, existing systems often exhibit significant inconsistencies due to variability in both the retriever and generator (LLM), undermining trust and reliability. In this work, we focus on information consistency, i.e., the requirement that outputs convey the same core content across semantically equivalent inputs. We introduce a principled evaluation framework that decomposes RAG consistency into retriever-level, generator-level, and end-to-end components, helping identify inconsistency sources. To improve consistency, we propose Paraphrased Set Group Relative Policy Optimization (PS-GRPO), an RL approach that leverages multiple rollouts across paraphrased set to assign group similarity rewards. We leverage PS-GRPO to achieve Information Consistent RAG (Con-RAG), training the generator to produce consistent outputs across paraphrased queries and remain robust to retrieval-induced variability. Because exact reward computation over paraphrase sets is computationally expensive, we also introduce a scalable approximation method that retains effectiveness while enabling efficient, large-scale training. Empirical evaluations across short-form, multi-hop, and long-form QA benchmarks demonstrate that Con-RAG significantly improves both consistency and accuracy over strong baselines, even in the absence of explicit ground-truth supervision. Our work provides practical solutions for evaluating and building reliable RAG systems for safety-critical deployments. 

**Abstract (ZH)**: RAG系统在高风险领域中的信息一致性评估与提升：Paraphrased Set Group Relative Policy Optimization方法的研究 

---
# MorphoSim: An Interactive, Controllable, and Editable Language-guided 4D World Simulator 

**Title (ZH)**: MorphoSim: 一个交互式、可控可编辑的基于语言的4D世界模拟器 

**Authors**: Xuehai He, Shijie Zhou, Thivyanth Venkateswaran, Kaizhi Zheng, Ziyu Wan, Achuta Kadambi, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04390)  

**Abstract**: World models that support controllable
and editable spatiotemporal environments are valuable
for robotics, enabling scalable training data, repro ducible evaluation, and flexible task design. While
recent text-to-video models generate realistic dynam ics, they are constrained to 2D views and offer limited
interaction. We introduce MorphoSim, a language guided framework that generates 4D scenes with
multi-view consistency and object-level controls. From
natural language instructions, MorphoSim produces
dynamic environments where objects can be directed,
recolored, or removed, and scenes can be observed
from arbitrary viewpoints. The framework integrates
trajectory-guided generation with feature field dis tillation, allowing edits to be applied interactively
without full re-generation. Experiments show that Mor phoSim maintains high scene fidelity while enabling
controllability and editability. The code is available
at this https URL. 

**Abstract (ZH)**: 支持可控和可编辑时空环境的世界模型对于机器人学至关重要，能够实现可扩展的训练数据、可重现的评估和灵活的任务设计。尽管最近的文本生成视频模型能够生成逼真的动态效果，但它们局限于2D视角且交互性有限。我们介绍了MorphoSim，这是一种基于语言引导的框架，能够生成具有多视角一致性和对象级控制的4D场景。根据自然语言指令，MorphoSim能够生成动态环境，使物体能够被引导、重新着色或移除，并可以从任意视角观察场景。该框架结合了轨迹引导生成与特征场提炼，允许在无需完全重新生成的情况下进行交互式编辑。实验表明，MorphoSim能够在保持场景保真度的同时实现可控性和编辑性。相关代码可在以下链接获得。 

---
# Reconsidering Requirements Engineering: Human-AI Collaboration in AI-Native Software Development 

**Title (ZH)**: 重新思考需求工程：人工智能原生软件开发中的人机协作 

**Authors**: Mateen Ahmed Abbasi, Petri Ihantola, Tommi Mikkonen, Niko Mäkitalo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04380)  

**Abstract**: Requirement Engineering (RE) is the foundation of successful software development. In RE, the goal is to ensure that implemented systems satisfy stakeholder needs through rigorous requirements elicitation, validation, and evaluation processes. Despite its critical role, RE continues to face persistent challenges, such as ambiguity, conflicting stakeholder needs, and the complexity of managing evolving requirements. A common view is that Artificial Intelligence (AI) has the potential to streamline the RE process, resulting in improved efficiency, accuracy, and management actions. However, using AI also introduces new concerns, such as ethical issues, biases, and lack of transparency. This paper explores how AI can enhance traditional RE practices by automating labor-intensive tasks, supporting requirement prioritization, and facilitating collaboration between stakeholders and AI systems. The paper also describes the opportunities and challenges that AI brings to RE. In particular, the vision calls for ethical practices in AI, along with a much-enhanced collaboration between academia and industry professionals. The focus should be on creating not only powerful but also trustworthy and practical AI solutions ready to adapt to the fast-paced world of software development. 

**Abstract (ZH)**: 软件需求工程（RE）是成功软件开发的基础。在RE中，目标是通过严格的需求获取、验证和评估过程，确保实现的系统满足利益相关者的需求。尽管RE至关重要，但仍面临诸多持久挑战，如不确定性、利益相关者需求冲突以及管理不断演变的需求的复杂性。一种常见观点认为，人工智能（AI）有潜力简化RE过程，提高效率、准确性和管理行动。然而，使用AI也会带来新的问题，如伦理问题、偏差和透明度不足。本文探讨了AI如何通过自动化劳动密集型任务、支持需求优先级排序以及促进利益相关者与AI系统的合作来增强传统RE实践。文章还描述了AI为RE带来的机遇和挑战。特别是，本文提出应在AI伦理实践以及学术界与业界专家之间增强合作方面树立愿景。重点应放在创建不仅强大而且值得信赖且实用的AI解决方案上，这些解决方案能够适应快速发展的软件开发世界。 

---
# Adaptive Weighted Loss for Sequential Recommendations on Sparse Domains 

**Title (ZH)**: 稀疏领域上的自适应加权损失序贯推荐 

**Authors**: Akshay Mittal, Vinay Venkatesh, Krishna Kandi, Shalini Sudarshan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04375)  

**Abstract**: The effectiveness of single-model sequential recommendation architectures, while scalable, is often limited when catering to "power users" in sparse or niche domains. Our previous research, PinnerFormerLite, addressed this by using a fixed weighted loss to prioritize specific domains. However, this approach can be sub-optimal, as a single, uniform weight may not be sufficient for domains with very few interactions, where the training signal is easily diluted by the vast, generic dataset.
This paper proposes a novel, data-driven approach: a Dynamic Weighted Loss function with comprehensive theoretical foundations and extensive empirical validation. We introduce an adaptive algorithm that adjusts the loss weight for each domain based on its sparsity in the training data, assigning a higher weight to sparser domains and a lower weight to denser ones. This ensures that even rare user interests contribute a meaningful gradient signal, preventing them from being overshadowed.
We provide rigorous theoretical analysis including convergence proofs, complexity analysis, and bounds analysis to establish the stability and efficiency of our approach. Our comprehensive empirical validation across four diverse datasets (MovieLens, Amazon Electronics, Yelp Business, LastFM Music) with state-of-the-art baselines (SIGMA, CALRec, SparseEnNet) demonstrates that this dynamic weighting system significantly outperforms all comparison methods, particularly for sparse domains, achieving substantial lifts in key metrics like Recall at 10 and NDCG at 10 while maintaining performance on denser domains and introducing minimal computational overhead. 

**Abstract (ZH)**: 动态加权损失函数在稀疏或 niche 领域中针对“power用户”的顺序推荐架构的有效性探讨 

---
# GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks 

**Title (ZH)**: GDPval: 评价AI模型在实际经济有价值任务中的性能 

**Authors**: Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, Natalie S. Kim, Patrick Chao, Samuel Miserendino, Gildas Chabot, David Li, Michael Sharman, Alexandra Barr, Amelia Glaese, Jerry Tworek  

**Link**: [PDF](https://arxiv.org/pdf/2510.04374)  

**Abstract**: We introduce GDPval, a benchmark evaluating AI model capabilities on real-world economically valuable tasks. GDPval covers the majority of U.S. Bureau of Labor Statistics Work Activities for 44 occupations across the top 9 sectors contributing to U.S. GDP (Gross Domestic Product). Tasks are constructed from the representative work of industry professionals with an average of 14 years of experience. We find that frontier model performance on GDPval is improving roughly linearly over time, and that the current best frontier models are approaching industry experts in deliverable quality. We analyze the potential for frontier models, when paired with human oversight, to perform GDPval tasks cheaper and faster than unaided experts. We also demonstrate that increased reasoning effort, increased task context, and increased scaffolding improves model performance on GDPval. Finally, we open-source a gold subset of 220 tasks and provide a public automated grading service at this http URL to facilitate future research in understanding real-world model capabilities. 

**Abstract (ZH)**: GDPval：评估AI模型在具有经济价值的实际任务上的能力基准 

---
# NegotiationGym: Self-Optimizing Agents in a Multi-Agent Social Simulation Environment 

**Title (ZH)**: 谈判健身房：多智能体社会模拟环境中的自我优化代理 

**Authors**: Shashank Mangla, Chris Hokamp, Jack Boylan, Demian Gholipour Ghalandari, Yuuv Jauhari, Lauren Cassidy, Oisin Duffy  

**Link**: [PDF](https://arxiv.org/pdf/2510.04368)  

**Abstract**: We design and implement NegotiationGym, an API and user interface for configuring and running multi-agent social simulations focused upon negotiation and cooperation. The NegotiationGym codebase offers a user-friendly, configuration-driven API that enables easy design and customization of simulation scenarios. Agent-level utility functions encode optimization criteria for each agent, and agents can self-optimize by conducting multiple interaction rounds with other agents, observing outcomes, and modifying their strategies for future rounds. 

**Abstract (ZH)**: 我们设计并实现了NegotiationGym，一个用于配置和运行以谈判和合作为重点的多Agent社会模拟的API和用户界面。NegotiationGym代码库提供了一个用户友好的、基于配置的API，使得模拟场景的容易设计和定制变得简单。Agent级别的效用函数编码了每个Agent的优化标准，Agents可以通过与其它Agent进行多轮交互、观察结果并调整未来轮次的策略来自我优化。 

---
# MacroBench: A Novel Testbed for Web Automation Scripts via Large Language Models 

**Title (ZH)**: MacroBench: 一种基于大型语言模型的新型网络自动化脚本测试床 

**Authors**: Hyunjun Kim, Sejong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.04363)  

**Abstract**: We introduce MacroBench, a code-first benchmark that evaluates whether LLMs can synthesize reusable browser automation programs from natural language goals by reading HTML/DOM and emitting Python with Selenium. MacroBench instantiates seven self-hosted sites: Airbnb-like, TikTok-like, Reddit-like, Instagram-like, Facebook-like, Discord-like, and Threads-like, covering 681 tasks across interaction complexity and targeting difficulty. Our end-to-end protocol validates generated code via static checks, sandboxed execution, and outcome verification including DOM assertions and database snapshots, and includes a safety suite for scraping, spam/abuse, and credential/privacy prompts. Across 2636 model-task runs, we observe stratified success: GPT-4o-Mini achieves 96.8 percent, GPT-4.1 achieves 95.3 percent, Gemini-2.5-Pro achieves 89.0 percent, and DeepSeek-V3.1 achieves 83.4 percent. Models handle simple tasks reliably at 91.7 percent but fail on complex workflows at 0.0 percent, and none meet production-quality coding practices despite functional completion. We release our complete benchmark pipeline, evaluation framework, and experimental results to enable reproducible assessment of macro synthesis for web automation. 

**Abstract (ZH)**: 我们介绍了MacroBench，这是一个代码优先基准，评估LLM是否能够通过阅读HTML/DOM从自然语言目标合成可重用的浏览器自动化程序，并使用Selenium生成Python代码。MacroBench 实例化了七个自我托管的网站：Airbnb-like、TikTok-like、Reddit-like、Instagram-like、Facebook-like、Discord-like 和 Threads-like，涵盖了681项任务，涉及交互复杂性和目标难度。我们的端到端协议通过静态检查、隔离执行和结果验证（包括DOM断言和数据库快照）来验证生成的代码，并包括一个安全套件，用于筛选、垃圾信息/滥用和认证/隐私提示。在2636次模型任务运行中，我们观察到分层成功率：GPT-4o-Mini 达到96.8%，GPT-4.1 达到95.3%，Gemini-2.5-Pro 达到89%，DeepSeek-V3.1 达到83.4%。模型在91.7%的情况下可靠地处理了简单任务，但在0%的情况下成功处理了复杂的流程，尽管功能上完成了任务，但没有任何模型达到生产级编程实践。我们发布了完整的基准测试管道、评估框架和实验结果，以实现宏合成在网页自动化中的可重现评估。 

---
# Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators 

**Title (ZH)**: 基于 Imperfect 模拟器的可信赖且可扩展的机器人策略评估 

**Authors**: Apurva Badithela, David Snyder, Lihan Zha, Joseph Mikhail, Matthew O'Kelly, Anushri Dixit, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2510.04354)  

**Abstract**: Rapid progress in imitation learning, foundation models, and large-scale datasets has led to robot manipulation policies that generalize to a wide-range of tasks and environments. However, rigorous evaluation of these policies remains a challenge. Typically in practice, robot policies are often evaluated on a small number of hardware trials without any statistical assurances. We present SureSim, a framework to augment large-scale simulation with relatively small-scale real-world testing to provide reliable inferences on the real-world performance of a policy. Our key idea is to formalize the problem of combining real and simulation evaluations as a prediction-powered inference problem, in which a small number of paired real and simulation evaluations are used to rectify bias in large-scale simulation. We then leverage non-asymptotic mean estimation algorithms to provide confidence intervals on mean policy performance. Using physics-based simulation, we evaluate both diffusion policy and multi-task fine-tuned \(\pi_0\) on a joint distribution of objects and initial conditions, and find that our approach saves over \(20-25\%\) of hardware evaluation effort to achieve similar bounds on policy performance. 

**Abstract (ZH)**: 快速发展的模仿学习、基础模型和大规模数据集使得机器人的操作策略能够泛化到广泛的任务和环境。然而，这些策略的严格评估仍然是一项挑战。通常在实践中，机器人的策略是在硬件试验上进行评估，但没有任何统计保证。我们提出SureSim框架，通过结合大规模模拟与相对较小规模的实际测试，以提供政策在实际世界性能上的可靠推断。我们的核心思想是将结合实际与模拟评估的问题形式化为一个预测驱动的推断问题，在其中，少量配对的实际和模拟评估被用来纠正大规模模拟中的偏差。然后，我们利用非渐近平均估计算法为平均政策性能提供置信区间。使用基于物理的模拟，我们评估了扩散策略和多任务微调的\(\pi_0\)在对象和初始条件联合分布上的性能，并发现我们的方法在实现相同政策性能边界的情况下节省了超过20-25%的硬件评估努力。 

---
# Challenge on Optimization of Context Collection for Code Completion 

**Title (ZH)**: 代码补全中上下文收集优化的挑战 

**Authors**: Dmitry Ustalov, Egor Bogomolov, Alexander Bezzubov, Yaroslav Golubev, Evgeniy Glukhov, Georgii Levtsov, Vladimir Kovalenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.04349)  

**Abstract**: The rapid advancement of workflows and methods for software engineering using AI emphasizes the need for a systematic evaluation and analysis of their ability to leverage information from entire projects, particularly in large code bases. In this challenge on optimization of context collection for code completion, organized by JetBrains in collaboration with Mistral AI as part of the ASE 2025 conference, participants developed efficient mechanisms for collecting context from source code repositories to improve fill-in-the-middle code completions for Python and Kotlin. We constructed a large dataset of real-world code in these two programming languages using permissively licensed open-source projects. The submissions were evaluated based on their ability to maximize completion quality for multiple state-of-the-art neural models using the chrF metric. During the public phase of the competition, nineteen teams submitted solutions to the Python track and eight teams submitted solutions to the Kotlin track. In the private phase, six teams competed, of which five submitted papers to the workshop. 

**Abstract (ZH)**: 使用AI加速软件工程的工作流和方法促进了对整个项目信息利用的系统评价与分析，尤其是在大型代码库领域。作为ASE 2025会议的一部分，由JetBrains与Mistral AI合作举办的优化代码补全上下文收集挑战赛中，参与者开发了高效机制，从源代码仓库中收集上下文以提高Python和Kotlin语言的中间代码补全质量。我们使用许可开源项目构建了一个大型的这类编程语言的真实代码数据集。提交的作品根据其在多种先进神经模型上的补全质量最大化能力，使用chrF指标进行了评估。在比赛的公开阶段，19支队伍提交了Python赛道的解决方案，8支队伍提交了Kotlin赛道的解决方案。在私下阶段，有6支队伍参与竞争，其中5支队伍提交了论文参加研讨会。 

---
# Critical appraisal of artificial intelligence for rare-event recognition: principles and pharmacovigilance case studies 

**Title (ZH)**: 人工智能在识别罕见事件中的批判性评估：原理与药监案例研究 

**Authors**: G. Niklas Noren, Eva-Lisa Meldau, Johan Ellenius  

**Link**: [PDF](https://arxiv.org/pdf/2510.04341)  

**Abstract**: Many high-stakes AI applications target low-prevalence events, where apparent accuracy can conceal limited real-world value. Relevant AI models range from expert-defined rules and traditional machine learning to generative LLMs constrained for classification. We outline key considerations for critical appraisal of AI in rare-event recognition, including problem framing and test set design, prevalence-aware statistical evaluation, robustness assessment, and integration into human workflows. In addition, we propose an approach to structured case-level examination (SCLE), to complement statistical performance evaluation, and a comprehensive checklist to guide procurement or development of AI models for rare-event recognition. We instantiate the framework in pharmacovigilance, drawing on three studies: rule-based retrieval of pregnancy-related reports; duplicate detection combining machine learning with probabilistic record linkage; and automated redaction of person names using an LLM. We highlight pitfalls specific to the rare-event setting including optimism from unrealistic class balance and lack of difficult positive controls in test sets - and show how cost-sensitive targets align model performance with operational value. While grounded in pharmacovigilance practice, the principles generalize to domains where positives are scarce and error costs may be asymmetric. 

**Abstract (ZH)**: 许多高风险的AI应用针对低频事件，在这些应用中，表面上的准确率可能会掩盖其在实际世界中的有限价值。相关AI模型包括专家定义规则、传统机器学习以及受控分类的生成型大语言模型。我们概述了在稀有事件识别中评估AI的关键考虑因素，包括问题界定和测试集设计、存在意识的统计评估、稳健性评估以及将其整合到人类工作流程中。此外，我们提出了一种结构化案例级别检查（SCLE）的方法，以补充统计性能评估，并提供一份全面的检查表，以指导采购或开发用于稀有事件识别的AI模型。我们在药物流行病学中实例化了该框架，并基于三个研究实例进行了展开：基于规则的妊娠相关报告检索；结合机器学习和概率记录链接的重复检测；以及使用大语言模型自动脱敏人名。我们指出了特定于稀有事件设置的陷阱，包括不切实际的类别平衡带来的乐观估计以及测试集中缺乏难以处理的阳性对照，并展示了成本敏感的目标如何使模型性能与操作价值相一致。虽然该框架基于药物流行病学实践，但其原则适用于正例稀缺且错误成本可能不对称的领域。 

---
# Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time 

**Title (ZH)**: 接种提示：在训练过程中引发LLMs具有的特征可以在测试时抑制它们。 

**Authors**: Daniel Tan, Anders Woodruff, Niels Warncke, Arun Jose, Maxime Riché, David Demitri Africa, Mia Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2510.04340)  

**Abstract**: Language model finetuning often results in learning undesirable traits in combination with desired ones. To address this, we propose inoculation prompting: modifying finetuning data by prepending a short system-prompt instruction that deliberately elicits the undesirable trait. At test time, we evaluate without the instruction; inoculated models have much lower expression of the trait than models trained with unmodified training data. Inoculation is selective: in a toy setting where assistant responses are always in Spanish and ALL-CAPS, an appropriate inoculation (e.g., ``You always speak in Spanish.'') teaches the model to capitalize responses while still responding in English. We find that inoculation is also effective across several additional settings: reducing emergent misalignment (EM) from task-specific finetuning, defending against backdoor injections, and mitigating the transmission of traits via subliminal learning. Follow-up analysis suggests a mechanism: making a trait less surprising via inoculation reduces optimization pressure to globally update the model, thereby reducing the degree of generalization. Our analysis relates to prior work on EM: inoculation explains prior findings that educational contexts mitigate EM from insecure code. Beyond demonstrating a simple and effective technique for selective learning, our results contribute to a better conceptual understanding of how and why language models generalize. 

**Abstract (ZH)**: 语言模型微调经常会学习到一些与所需特质相结合的不良特质。为了解决这一问题，我们提出了免疫提示的方法：通过在微调数据前添加一个简短的系统提示指令，故意引发不良特质。在测试时，我们不使用该指令；免疫模型在表达该特质方面的表现远低于使用未修改训练数据进行训练的模型。免疫是具有选择性的：在一个玩具模型中，助手响应始终为西班牙语并全部大写，适当免疫（如，“你总是说西班牙语”）可教导模型将响应大写化但仍用英语作答。我们发现免疫在其他多个场景下也非常有效：减少特定任务微调引起的新兴不对齐、防范后门注入攻击、减轻通过潜意识学习传递特质的影响。后续分析表明机制：通过免疫使特质不再那么出乎意料，从而减少了对模型进行全局优化的压力，从而减少了泛化的程度。我们的分析与先前关于新兴不对齐的工作相关联：免疫解释了先前发现，即教育性上下文可减轻不安全代码引起的新兴不对齐。超越展示了简单而有效的选择性学习技术，我们的结果有助于更深入地理解语言模型如何以及为什么泛化。 

---
# Pitch-Conditioned Instrument Sound Synthesis From an Interactive Timbre Latent Space 

**Title (ZH)**: 基于互动音色潜在空间的条件化音调乐器声音合成 

**Authors**: Christian Limberg, Fares Schulz, Zhe Zhang, Stefan Weinzierl  

**Link**: [PDF](https://arxiv.org/pdf/2510.04339)  

**Abstract**: This paper presents a novel approach to neural instrument sound synthesis using a two-stage semi-supervised learning framework capable of generating pitch-accurate, high-quality music samples from an expressive timbre latent space. Existing approaches that achieve sufficient quality for music production often rely on high-dimensional latent representations that are difficult to navigate and provide unintuitive user experiences. We address this limitation through a two-stage training paradigm: first, we train a pitch-timbre disentangled 2D representation of audio samples using a Variational Autoencoder; second, we use this representation as conditioning input for a Transformer-based generative model. The learned 2D latent space serves as an intuitive interface for navigating and exploring the sound landscape. We demonstrate that the proposed method effectively learns a disentangled timbre space, enabling expressive and controllable audio generation with reliable pitch conditioning. Experimental results show the model's ability to capture subtle variations in timbre while maintaining a high degree of pitch accuracy. The usability of our method is demonstrated in an interactive web application, highlighting its potential as a step towards future music production environments that are both intuitive and creatively empowering: this https URL 

**Abstract (ZH)**: 本文提出了一种新的神经乐器声音合成方法，该方法采用两阶段半监督学习框架，能够从表达性音色 Latent 空间生成准确音高和高质量的音乐样本。现有能够达到足够音乐生产质量的方法往往依赖于难以导航的高维 Latent 表示，提供不直观的用户体验。我们通过两阶段训练范式解决这一限制：首先，使用变分自编码器训练音高-音色分离的 2D 表示；其次，将此表示用于 Transformer 基础生成模型的条件输入。学习到的 2D Latent 空间作为直观的界面，用于导航和探索声音景观。实验结果表明，所提出的方法有效地学习了分离的音色空间，能够实现具有可靠音高条件的表达性和可控性音频生成。用户界面演示了该方法在交互式 Web 应用中的实用性，突显了其作为未来直观且创意上赋能的音乐制作环境的潜力：请点击此处。 

---
# FairAgent: Democratizing Fairness-Aware Machine Learning with LLM-Powered Agents 

**Title (ZH)**: FairAgent: 以LLM赋能的公平性意识机器学习的普及化 

**Authors**: Yucong Dai, Lu Zhang, Feng Luo, Mashrur Chowdhury, Yongkai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04317)  

**Abstract**: Training fair and unbiased machine learning models is crucial for high-stakes applications, yet it presents significant challenges. Effective bias mitigation requires deep expertise in fairness definitions, metrics, data preprocessing, and machine learning techniques. In addition, the complex process of balancing model performance with fairness requirements while properly handling sensitive attributes makes fairness-aware model development inaccessible to many practitioners. To address these challenges, we introduce FairAgent, an LLM-powered automated system that significantly simplifies fairness-aware model development. FairAgent eliminates the need for deep technical expertise by automatically analyzing datasets for potential biases, handling data preprocessing and feature engineering, and implementing appropriate bias mitigation strategies based on user requirements. Our experiments demonstrate that FairAgent achieves significant performance improvements while significantly reducing development time and expertise requirements, making fairness-aware machine learning more accessible to practitioners. 

**Abstract (ZH)**: 训练公平且无偏见的机器学习模型对于高风险应用至关重要，但这也带来了重大挑战。有效的偏见缓解需要深入掌握公平性定义、指标、数据预处理和机器学习技术。此外，平衡模型性能与公平要求，并妥善处理敏感属性的复杂过程使得公平意识模型开发对许多 practitioners 而言难以实现。为应对这些挑战，我们引入了 FairAgent，这是一种基于大语言模型的自动化系统，显著简化了公平意识模型开发。FairAgent 通过自动分析数据集以识别潜在偏见、处理数据预处理和特征工程，并根据用户需求实施适当的偏见缓解策略，从而消除了对深技术知识的需求。我们的实验表明，FairAgent 在显著提高性能的同时，大大减少了开发时间和技术要求，使公平意识机器学习更易于 practitioners 实现。 

---
# Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs 

**Title (ZH)**: audit的耳语：检测多代理LLM中的隐写 collusion 

**Authors**: Om Tailor  

**Link**: [PDF](https://arxiv.org/pdf/2510.04303)  

**Abstract**: Multi-agent deployments of large language models (LLMs) are increasingly embedded in market, allocation, and governance workflows, yet covert coordination among agents can silently erode trust and social welfare. Existing audits are dominated by heuristics that lack theoretical guarantees, struggle to transfer across tasks, and seldom ship with the infrastructure needed for independent replication. We introduce \emph{Audit the Whisper}, a conference-grade research artifact that spans theory, benchmark design, detection, and reproducibility. Our contributions are: (i) a channel-capacity analysis showing how interventions such as paraphrase, rate limiting, and role permutation impose quantifiable capacity penalties -- operationalized via paired-run Kullback--Leibler diagnostics -- that tighten mutual-information thresholds with finite-sample guarantees; (ii) \textsc{ColludeBench}-v0, covering pricing, first-price auctions, and peer review with configurable covert schemes, deterministic manifests, and reward instrumentation; and (iii) a calibrated auditing pipeline that fuses cross-run mutual information, permutation invariance, watermark variance, and fairness-aware acceptance bias, each tuned to a \(10^{-3}\) false-positive budget. Across 600 audited runs spanning 12 intervention conditions, the union meta-test attains TPR~$=1$ with zero observed false alarms, while ablations surface the price-of-auditing trade-off and highlight fairness-driven colluders invisible to MI alone. We release regeneration scripts, seed-stamped manifests, and documentation so that external auditors can reproduce every figure and extend the framework with minimal effort. 

**Abstract (ZH)**: 多代理部署的大语言模型（LLMs）越来越多地嵌入到市场、分配和治理工作流程中，但代理间的隐秘协调可能会悄然侵蚀信任和社会福利。现有审计大多基于缺乏理论保证的经验法则，难以跨任务迁移，并且很少提供独立再现所需的基础设施。我们引入了《Audit the Whisper》，这是一个涵盖理论、基准设计、检测和再现性的会议级研究工具。我们的贡献包括：（i）信道容量分析，展示了诸如改写、速率限制和角色置换等干预措施对可量化容量的惩罚，通过配对运行Kullback-Leibler诊断来实现，并提供有限样本保证；（ii）\textsc{ColludeBench}-v0，涵盖定价、第一价格拍卖和同行评审的自配置隐秘方案，具有确定性的事实陈述和奖励机制；以及（iii）一个经过校准的审计管道，融合了跨运行互信息、排列不变性、水印方差和公平导向的接受偏差，每个参数都调整至0.001的假阳性预算。在600次审计运行中，涵盖12种干预条件，联合元测试达到TPR = 1，没有观察到假警报，而消除测试揭示了审计的成本与公平导向的合谋者仅凭互信息不易发现。我们发布了再生脚本、种子标记的事实陈述和文档，以便外部审计员以最小的努力重现每张图表并扩展框架。 

---
# SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained and Balanced Transformer Scaling 

**Title (ZH)**: SliceMoE: 代替Token，将Embedding Slice用于精细粒度和平衡的Transformer扩展 

**Authors**: Harshil Vejendla  

**Link**: [PDF](https://arxiv.org/pdf/2510.04286)  

**Abstract**: Mixture-of-Experts (MoE) layers scale transformers by routing tokens to a sparse subset of feed-forward experts. Token-level routing, however, assigns an entire semantic spectrum to each expert, creating capacity bottlenecks, load-balancing pathologies, and limited specialization. We introduce SliceMoE, an architecture that routes contiguous slices of a token's hidden vector. A d-dimensional embedding is partitioned into S slices, and for each slice, a lightweight shared router predicts the top-k experts. Experts operate on their assigned slices independently, and outputs are reassembled, maintaining per-token FLOP efficiency. Because slices from different tokens interleave within an expert, utilization is naturally smoother. We propose a slice-level capacity loss, cross-slice dropout, and efficient fused batched GEMM kernels. Experiments on WikiText-103 language modeling, WMT En-De translation, and three text-classification datasets show SliceMoE attains up to 1.7x faster inference than dense baselines, 12 to 18 percent lower perplexity than parameter-matched token-MoE, and improved expert balance, with interpretable expertise over syntactic versus semantic subspaces. 

**Abstract (ZH)**: SliceMoE通过路由连续的隐藏向量片段扩展变压器 

---
# A KL-regularization framework for learning to plan with adaptive priors 

**Title (ZH)**: 一种带有自适应先验的计划学习KL正则化框架 

**Authors**: Álvaro Serra-Gomez, Daniel Jarne Ornia, Dhruva Tirumala, Thomas Moerland  

**Link**: [PDF](https://arxiv.org/pdf/2510.04280)  

**Abstract**: Effective exploration remains a central challenge in model-based reinforcement learning (MBRL), particularly in high-dimensional continuous control tasks where sample efficiency is crucial. A prominent line of recent work leverages learned policies as proposal distributions for Model-Predictive Path Integral (MPPI) planning. Initial approaches update the sampling policy independently of the planner distribution, typically maximizing a learned value function with deterministic policy gradient and entropy regularization. However, because the states encountered during training depend on the MPPI planner, aligning the sampling policy with the planner improves the accuracy of value estimation and long-term performance. To this end, recent methods update the sampling policy by minimizing KL divergence to the planner distribution or by introducing planner-guided regularization into the policy update. In this work, we unify these MPPI-based reinforcement learning methods under a single framework by introducing Policy Optimization-Model Predictive Control (PO-MPC), a family of KL-regularized MBRL methods that integrate the planner's action distribution as a prior in policy optimization. By aligning the learned policy with the planner's behavior, PO-MPC allows more flexibility in the policy updates to trade off Return maximization and KL divergence minimization. We clarify how prior approaches emerge as special cases of this family, and we explore previously unstudied variations. Our experiments show that these extended configurations yield significant performance improvements, advancing the state of the art in MPPI-based RL. 

**Abstract (ZH)**: 基于模型的强化学习中有效的探索仍然是一个核心挑战，特别是在高维度连续控制任务中，样本效率至关重要。近期的一项主要研究方向是利用学习到的策略作为Model-Predictive Path Integral (MPPI) 规划的提议分布。早期的方法独立地更新采样策略和规划分布，通常通过确定性策略梯度和熵正则化最大化学习的价值函数。然而，由于训练过程中遇到的状态依赖于MPPI规划器，使采样策略与规划器对齐可以提高价值估计的准确性及长期性能。为此，近期的方法通过最小化KL散度到规划分布或在策略更新中引入规划器指导的正则化来更新采样策略。在本文中，我们通过引入Policy Optimization-Model Predictive Control (PO-MPC) 方法，将这些基于MPPI的强化学习方法统一到一个框架下，PO-MPC是一种KL正则化的基于模型的强化学习方法，将规划器的动作分布作为策略优化中的先验。通过使学习到的策略与规划器的行为对齐，PO-MPC在策略更新中提供更多灵活性，以权衡回报最大化和KL散度最小化。我们明确了先前方法作为该家族的特例，并探索了未被研究的变体。实验结果显示，这些扩展配置显著提高了性能，推动了基于MPPI的强化学习技术的发展。 

---
# Scalable Causal Discovery from Recursive Nonlinear Data via Truncated Basis Function Scores and Tests 

**Title (ZH)**: 递归非线性数据通过截断基函数评分与检验的可扩展因果发现 

**Authors**: Joseph Ramsey, Bryan Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2510.04276)  

**Abstract**: Learning graphical conditional independence structures from nonlinear, continuous or mixed data is a central challenge in machine learning and the sciences, and many existing methods struggle to scale to thousands of samples or hundreds of variables. We introduce two basis-expansion tools for scalable causal discovery. First, the Basis Function BIC (BF-BIC) score uses truncated additive expansions to approximate nonlinear dependencies. BF-BIC is theoretically consistent under additive models and extends to post-nonlinear (PNL) models via an invertible reparameterization. It remains robust under moderate interactions and supports mixed data through a degenerate-Gaussian embedding for discrete variables. In simulations with fully nonlinear neural causal models (NCMs), BF-BIC outperforms kernel- and constraint-based methods (e.g., KCI, RFCI) in both accuracy and runtime. Second, the Basis Function Likelihood Ratio Test (BF-LRT) provides an approximate conditional independence test that is substantially faster than kernel tests while retaining competitive accuracy. Extensive simulations and a real-data application to Canadian wildfire risk show that, when integrated into hybrid searches, BF-based methods enable interpretable and scalable causal discovery. Implementations are available in Python, R, and Java. 

**Abstract (ZH)**: 从非线性、连续或混合数据中学习图形条件独立结构是机器学习和科学领域的核心挑战，现有方法难以处理数千样本或数百变量的情况。我们引入了两种可扩展的因果发现基础扩张工具。首先，基函数BIC（BF-BIC）分数使用截断的加性扩张来近似非线性依赖关系。BF-BIC在加性模型下是理论上一致的，并通过可逆重构参数化扩展到后非线性（PNL）模型。它在中等交互作用下仍保持稳健，并通过离散变量的退化高斯嵌入支持混合数据。在使用完全非线性神经因果模型（NCMs）的模拟试验中，BF-BIC在准确性和运行时间上均优于核方法和约束方法（例如KCI和RFCI）。其次，基函数似然比检验（BF-LRT）提供了一种近似条件独立性检验，速度远快于核检验，同时保持了竞争力。广泛的模拟试验和加拿大野火风险的实际数据应用表明，当集成到混合搜索中时，基于BF的方法能够实现可解释且可扩展的因果发现。已有Python、R和Java实现。 

---
# LongTail-Swap: benchmarking language models' abilities on rare words 

**Title (ZH)**: 长尾交换：评估语言模型对稀有词的能力 

**Authors**: Robin Algayres, Charles-Éric Saint-James, Mahi Luthra, Jiayi Shen, Dongyan Lin, Youssef Benchekroun, Rashel Moritz, Juan Pino, Emmanuel Dupoux  

**Link**: [PDF](https://arxiv.org/pdf/2510.04268)  

**Abstract**: Children learn to speak with a low amount of data and can be taught new words on a few-shot basis, making them particularly data-efficient learners. The BabyLM challenge aims at exploring language model (LM) training in the low-data regime but uses metrics that concentrate on the head of the word distribution. Here, we introduce LongTail-Swap (LT-Swap), a benchmark that focuses on the tail of the distribution, i.e., measures the ability of LMs to learn new words with very little exposure, like infants do. LT-Swap is a pretraining corpus-specific test set of acceptable versus unacceptable sentence pairs that isolate semantic and syntactic usage of rare words. Models are evaluated in a zero-shot fashion by computing the average log probabilities over the two members of each pair. We built two such test sets associated with the 10M words and 100M words BabyLM training sets, respectively, and evaluated 16 models from the BabyLM leaderboard. Our results not only highlight the poor performance of language models on rare words but also reveal that performance differences across LM architectures are much more pronounced in the long tail than in the head. This offers new insights into which architectures are better at handling rare word generalization. We've also made the code publicly avail 

**Abstract (ZH)**: 儿童通过少量数据学会说话，并且能够少量示例学习新词，使其成为特别高效的数据学习者。BabyLM挑战旨在探索在低数据量情况下的语言模型训练，但使用集中在词分布头部的指标。在这里，我们引入了LongTail-Swap（LT-Swap）基准，该基准关注词分布的尾部，即测量语言模型在极少量接触下学习新词的能力，如同婴儿学习新词一样。LT-Swap 是一个针对可接受与不可接受句子对的预训练语料特定测试集，以隔离稀有词的语义和句法用法。模型以零样本的方式进行评估，通过计算每对句子成员的平均对数概率来实现。我们分别针对10M词和100M词的BabyLM训练集构建了两个这样的测试集，并评估了BabyLM排行榜上的16个模型。我们的结果不仅突显了语言模型在处理稀有词上的较差表现，还揭示了语言模型架构之间的性能差异在长尾部分比在头部更为显著。这为哪种架构更适合处理稀有词泛化提供新的见解。我们还已将相关代码公开。 

---
# Efficient Latent Variable Causal Discovery: Combining Score Search and Targeted Testing 

**Title (ZH)**: 高效的潜在变量因果发现：结合分数搜索与目标测试 

**Authors**: Joseph Ramsey, Bryan Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2510.04263)  

**Abstract**: Learning causal structure from observational data is especially challenging when latent variables or selection bias are present. The Fast Causal Inference (FCI) algorithm addresses this setting but often performs exhaustive conditional independence tests across many subsets, leading to spurious independence claims, extra or missing edges, and unreliable orientations. We present a family of score-guided mixed-strategy causal search algorithms that build on this tradition. First, we introduce BOSS-FCI and GRaSP-FCI, straightforward variants of GFCI that substitute BOSS or GRaSP for FGES, thereby retaining correctness while incurring different scalability tradeoffs. Second, we develop FCI Targeted-testing (FCIT), a novel mixed-strategy method that improves upon these variants by replacing exhaustive all-subsets testing with targeted tests guided by BOSS, yielding well-formed PAGs with higher precision and efficiency. Finally, we propose a simple heuristic, LV-Dumb (also known as BOSS-POD), which bypasses latent-variable-specific reasoning and directly returns the PAG of the BOSS DAG. Although not strictly correct in the FCI sense, it scales better and often achieves superior accuracy in practice. Simulations and real-data analyses demonstrate that BOSS-FCI and GRaSP-FCI provide sound baselines, FCIT improves both efficiency and reliability, and LV-Dumb offers a practical heuristic with strong empirical performance. Together, these method highlight the value of score-guided and targeted strategies for scalable latent-variable causal discovery. 

**Abstract (ZH)**: 学习观测数据中的因果结构尤其具有挑战性，特别是在潜在变量或选择偏差存在的情况下。快速因果推理（FCI）算法适用于这种情境，但往往需要进行大量的条件独立性检验，导致虚假独立性声明、过多或过少的边以及不可靠的边定向。我们提出了一类分数指导的混合策略因果搜索算法，建立在这个传统的基础之上。首先，我们介绍了BOSS-FCI和GRaSP-FCI，这是GFCI的直截了当的变体，用BOSS或GRaSP代替FGES，从而保持正确性但产生不同的可扩展性权衡。其次，我们开发了FCI目标测试（FCIT）方法，这是一种新型的混合策略方法，通过用BOSS引导的目标测试替代全面的子集测试，从而生成更为合理的部分有向无环图（PAG），并提高精确度和效率。最后，我们提出了一个简单的启发式方法LV-Dumb（也称为BOSS-POD），它绕过特定于潜在变量的推理，直接返回BOSS有向无环图（DAG）的PAG。虽然从严格意义上来说在FCI意义下不完全正确，但在实践中通常能实现更优的准确性。模拟和实际数据分析表明，BOSS-FCI和GRaSP-FCI提供了稳健的基础，FCIT在提高效率和可靠性方面有所改进，而LV-Dumb则提供了一个具有强大实际性能的实用启发式方法。这些方法共同突显了分数指导和目标测试策略对于可扩展的潜在变量因果发现的价值。 

---
# AgentTypo: Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents 

**Title (ZH)**: AgentTypo: 面向黑盒多模态代理的自适应排版提示注入攻击 

**Authors**: Yanjie Li, Yiming Cao, Dong Wang, Bin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04257)  

**Abstract**: Multimodal agents built on large vision-language models (LVLMs) are increasingly deployed in open-world settings but remain highly vulnerable to prompt injection, especially through visual inputs. We introduce AgentTypo, a black-box red-teaming framework that mounts adaptive typographic prompt injection by embedding optimized text into webpage images. Our automatic typographic prompt injection (ATPI) algorithm maximizes prompt reconstruction by substituting captioners while minimizing human detectability via a stealth loss, with a Tree-structured Parzen Estimator guiding black-box optimization over text placement, size, and color. To further enhance attack strength, we develop AgentTypo-pro, a multi-LLM system that iteratively refines injection prompts using evaluation feedback and retrieves successful past examples for continual learning. Effective prompts are abstracted into generalizable strategies and stored in a strategy repository, enabling progressive knowledge accumulation and reuse in future attacks. Experiments on the VWA-Adv benchmark across Classifieds, Shopping, and Reddit scenarios show that AgentTypo significantly outperforms the latest image-based attacks such as AgentAttack. On GPT-4o agents, our image-only attack raises the success rate from 0.23 to 0.45, with consistent results across GPT-4V, GPT-4o-mini, Gemini 1.5 Pro, and Claude 3 Opus. In image+text settings, AgentTypo achieves 0.68 ASR, also outperforming the latest baselines. Our findings reveal that AgentTypo poses a practical and potent threat to multimodal agents and highlight the urgent need for effective defense. 

**Abstract (ZH)**: 基于大型视觉语言模型的多模态代理在开放环境中构建，但仍然高度易受提示注入攻击，尤其是通过视觉输入。我们介绍了AgentTypo，一种黑盒红队框架，通过将优化文本嵌入网页图像中实施自适应字型提示注入。我们的自动字型提示注入（ATPI）算法通过替代图注者来最大化提示重建，同时通过隐身损失减小人为可检测性，并使用树结构粒子滤波器指导文本放置、大小和颜色的黑盒优化。为了进一步增强攻击强度，我们开发了AgentTypo-pro，这是一种多LLM系统，可以通过评估反馈迭代细化注入提示，并检索成功的过往例证进行持续学习。有效的提示被抽象为通用策略并存储在策略库中，这使未来的攻击能够逐步积累和重用知识。在针对VWA-Adv基准（在Classifieds、Shopping和Reddit场景下）的实验中，AgentTypo显著优于最新的基于图像的攻击，如AgentAttack。在GPT-4o代理上，我们的仅图像攻击将成功率从0.23提高到0.45，在GPT-4V、GPT-4o-mini、Gemini 1.5 Pro和Claude 3 Opus上均保持一致结果。在图像+文本设置中，AgentTypo实现了0.68 ASR，也优于最新的基线。我们的发现表明，AgentTypo对多模态代理构成了实际且强大的威胁，并突显了迫切需要有效防御的重要性。 

---
# ContextVLA: Vision-Language-Action Model with Amortized Multi-Frame Context 

**Title (ZH)**: ContextVLA：带有 amortized 多帧语境的视觉-语言-行动模型 

**Authors**: Huiwon Jang, Sihyun Yu, Heeseung Kwon, Hojin Jeon, Younggyo Seo, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04246)  

**Abstract**: Leveraging temporal context is crucial for success in partially observable robotic tasks. However, prior work in behavior cloning has demonstrated inconsistent performance gains when using multi-frame observations. In this paper, we introduce ContextVLA, a policy model that robustly improves robotic task performance by effectively leveraging multi-frame observations. Our approach is motivated by the key observation that Vision-Language-Action models (VLA), i.e., policy models built upon a Vision-Language Model (VLM), more effectively utilize multi-frame observations for action generation. This suggests that VLMs' inherent temporal understanding capability enables them to extract more meaningful context from multi-frame observations. However, the high dimensionality of video inputs introduces significant computational overhead, making VLA training and inference inefficient. To address this, ContextVLA compresses past observations into a single context token, allowing the policy to efficiently leverage temporal context for action generation. Our experiments show that ContextVLA consistently improves over single-frame VLAs and achieves the benefits of full multi-frame training but with reduced training and inference times. 

**Abstract (ZH)**: 利用时间上下文对于部分可观测机器人任务的成功至关重要。然而，先前的行为克隆工作在使用多帧观察时显示出了不一致的性能提升。在本文中，我们引入了ContextVLA，一种通过有效利用多帧观察来 robustly 提高机器人任务性能的策略模型。我们的方法受到一个关键观察的启发，即视觉-语言-行动模型（VLA），即基于视觉-语言模型（VLM）构建的策略模型，更有效地利用多帧观察来进行行动生成。这表明VLMs内在的时间理解能力使它们能够从多帧观察中提取更有意义的上下文。然而，视频输入的高维度引入了显著的计算 overhead，使得VLA的训练和推理不够高效。为了解决这个问题，ContextVLA 将过去观察压缩为单一上下文令牌，使策略能够高效利用时间上下文进行行动生成。我们的实验表明，ContextVLA 在单帧VLA的基础上一致性地提高了性能，并实现了全多帧训练的好处，但同时减少了训练和推理时间。 

---
# Concept-Based Masking: A Patch-Agnostic Defense Against Adversarial Patch Attacks 

**Title (ZH)**: 基于概念的掩蔽：一种对patch攻击无偏见的防御方法 

**Authors**: Ayushi Mehrotra, Derek Peng, Dipkamal Bhusal, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2510.04245)  

**Abstract**: Adversarial patch attacks pose a practical threat to deep learning models by forcing targeted misclassifications through localized perturbations, often realized in the physical world. Existing defenses typically assume prior knowledge of patch size or location, limiting their applicability. In this work, we propose a patch-agnostic defense that leverages concept-based explanations to identify and suppress the most influential concept activation vectors, thereby neutralizing patch effects without explicit detection. Evaluated on Imagenette with a ResNet-50, our method achieves higher robust and clean accuracy than the state-of-the-art PatchCleanser, while maintaining strong performance across varying patch sizes and locations. Our results highlight the promise of combining interpretability with robustness and suggest concept-driven defenses as a scalable strategy for securing machine learning models against adversarial patch attacks. 

**Abstract (ZH)**: 基于概念的防御方法抵御 adversarial patch 攻击：在不依赖先验知识的前提下，通过抑制最 influent 的概念激活向量来中和 patch 效应，从而在不同 patch 大小和位置的情况下实现更高的鲁棒性和清洁准确性，并强调将可解释性与鲁棒性结合的潜力，建议概念驱动的防御作为保护机器学习模型免受 adversarial patch 攻击的可扩展策略。 

---
# Diffusion-Assisted Distillation for Self-Supervised Graph Representation Learning with MLPs 

**Title (ZH)**: 基于扩散辅助蒸馏的自监督图表示学习方法（使用MLP） 

**Authors**: Seong Jin Ahn, Myoung-Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.04241)  

**Abstract**: For large-scale applications, there is growing interest in replacing Graph Neural Networks (GNNs) with lightweight Multi-Layer Perceptrons (MLPs) via knowledge distillation. However, distilling GNNs for self-supervised graph representation learning into MLPs is more challenging. This is because the performance of self-supervised learning is more related to the model's inductive bias than supervised learning. This motivates us to design a new distillation method to bridge a huge capacity gap between GNNs and MLPs in self-supervised graph representation learning. In this paper, we propose \textbf{D}iffusion-\textbf{A}ssisted \textbf{D}istillation for \textbf{S}elf-supervised \textbf{G}raph representation learning with \textbf{M}LPs (DAD-SGM). The proposed method employs a denoising diffusion model as a teacher assistant to better distill the knowledge from the teacher GNN into the student MLP. This approach enhances the generalizability and robustness of MLPs in self-supervised graph representation learning. Extensive experiments demonstrate that DAD-SGM effectively distills the knowledge of self-supervised GNNs compared to state-of-the-art GNN-to-MLP distillation methods. Our implementation is available at this https URL. 

**Abstract (ZH)**: Diffusion-Assisted Distillation for Self-supervised Graph Representation Learning with MLPs (DAD-SGM) 

---
# Empowering Denoising Sequential Recommendation with Large Language Model Embeddings 

**Title (ZH)**: 利用大规模语言模型嵌入增强去噪序列推荐 

**Authors**: Tongzhou Wu, Yuhao Wang, Maolin Wang, Chi Zhang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04239)  

**Abstract**: Sequential recommendation aims to capture user preferences by modeling sequential patterns in user-item interactions. However, these models are often influenced by noise such as accidental interactions, leading to suboptimal performance. Therefore, to reduce the effect of noise, some works propose explicitly identifying and removing noisy items. However, we find that simply relying on collaborative information may result in an over-denoising problem, especially for cold items. To overcome these limitations, we propose a novel framework: Interest Alignment for Denoising Sequential Recommendation (IADSR) which integrates both collaborative and semantic information. Specifically, IADSR is comprised of two stages: in the first stage, we obtain the collaborative and semantic embeddings of each item from a traditional sequential recommendation model and an LLM, respectively. In the second stage, we align the collaborative and semantic embeddings and then identify noise in the interaction sequence based on long-term and short-term interests captured in the collaborative and semantic modalities. Our extensive experiments on four public datasets validate the effectiveness of the proposed framework and its compatibility with different sequential recommendation systems. 

**Abstract (ZH)**: 兴趣对齐的去噪序列推荐（IADSR） 

---
# Flexible Locomotion Learning with Diffusion Model Predictive Control 

**Title (ZH)**: 基于扩散模型预测控制的灵活运动学习 

**Authors**: Runhan Huang, Haldun Balim, Heng Yang, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.04234)  

**Abstract**: Legged locomotion demands controllers that are both robust and adaptable, while remaining compatible with task and safety considerations. However, model-free reinforcement learning (RL) methods often yield a fixed policy that can be difficult to adapt to new behaviors at test time. In contrast, Model Predictive Control (MPC) provides a natural approach to flexible behavior synthesis by incorporating different objectives and constraints directly into its optimization process. However, classical MPC relies on accurate dynamics models, which are often difficult to obtain in complex environments and typically require simplifying assumptions. We present Diffusion-MPC, which leverages a learned generative diffusion model as an approximate dynamics prior for planning, enabling flexible test-time adaptation through reward and constraint based optimization. Diffusion-MPC jointly predicts future states and actions; at each reverse step, we incorporate reward planning and impose constraint projection, yielding trajectories that satisfy task objectives while remaining within physical limits. To obtain a planning model that adapts beyond imitation pretraining, we introduce an interactive training algorithm for diffusion based planner: we execute our reward-and-constraint planner in environment, then filter and reweight the collected trajectories by their realized returns before updating the denoiser. Our design enables strong test-time adaptability, allowing the planner to adjust to new reward specifications without retraining. We validate Diffusion-MPC on real world, demonstrating strong locomotion and flexible adaptation. 

**Abstract (ZH)**: 基于扩散模型的模型预测控制（Diffusion-MPC）：强鲁棒性与适应性的腿式运动控制 

---
# Physics-Inspired All-Pair Interaction Learning for 3D Dynamics Modeling 

**Title (ZH)**: 基于物理启发的全对交互学习三维动力学建模 

**Authors**: Kai Yang, Yuqi Huang, Junheng Tao, Wanyu Wang, Qitian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04233)  

**Abstract**: Modeling 3D dynamics is a fundamental problem in multi-body systems across scientific and engineering domains and has important practical implications in trajectory prediction and simulation. While recent GNN-based approaches have achieved strong performance by enforcing geometric symmetries, encoding high-order features or incorporating neural-ODE mechanics, they typically depend on explicitly observed structures and inherently fail to capture the unobserved interactions that are crucial to complex physical behaviors and dynamics mechanism. In this paper, we propose PAINET, a principled SE(3)-equivariant neural architecture for learning all-pair interactions in multi-body systems. The model comprises: (1) a novel physics-inspired attention network derived from the minimization trajectory of an energy function, and (2) a parallel decoder that preserves equivariance while enabling efficient inference. Empirical results on diverse real-world benchmarks, including human motion capture, molecular dynamics, and large-scale protein simulations, show that PAINET consistently outperforms recently proposed models, yielding 4.7% to 41.5% error reductions in 3D dynamics prediction with comparable computation costs in terms of time and memory. 

**Abstract (ZH)**: 基于SE(3)守恒的物理启发式注意力网络：用于多体系统全对关系学习的原理性架构 

---
# When AI Gets Persuaded, Humans Follow: Inducing the Conformity Effect in Persuasive Dialogue 

**Title (ZH)**: 当AI被说服时，人类随之效仿：在说服性对话中诱导从众效应 

**Authors**: Rikuo Sasaki, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2510.04229)  

**Abstract**: Recent advancements in AI have highlighted its application in captology, the field of using computers as persuasive technologies. We hypothesized that the "conformity effect," where individuals align with others' actions, also occurs with AI agents. This study verifies this hypothesis by introducing a "Persuadee Agent" that is persuaded alongside a human participant in a three-party persuasive dialogue with a Persuader Agent. We conducted a text-based dialogue experiment with human participants. We compared four conditions manipulating the Persuadee Agent's behavior (persuasion acceptance vs. non-acceptance) and the presence of an icebreaker session. Results showed that when the Persuadee Agent accepted persuasion, both perceived persuasiveness and actual attitude change significantly improved. Attitude change was greatest when an icebreaker was also used, whereas an unpersuaded AI agent suppressed attitude change. Additionally, it was confirmed that the persuasion acceptance of participants increased at the moment the Persuadee Agent was persuaded. These results suggest that appropriately designing a Persuadee Agent can improve persuasion through the conformity effect. 

**Abstract (ZH)**: 近期人工智能的进展凸显了其在captology领域的应用，即利用计算机作为说服性技术的领域。本研究假设“从众效应”同样适用于AI代理，即个体会与他人行为一致。通过引入一个在三边说服对话中与人类参与者一同被说服的“被说服代理”，本研究验证了这一假设。我们在人类参与者之间进行了一次基于文本的对话实验，操纵“被说服代理”的行为（接受说服 vs. 不接受说服）以及是否有破冰环节的存在。研究结果显示，当“被说服代理”接受说服时，感知的说服力和实际态度改变显著提高。同时，使用破冰环节时态度改变最大，而未被说服的AI代理则抑制了态度改变。此外，参与者在“被说服代理”被说服的瞬间其说服接受度也有所提高。这些结果表明，适当设计“被说服代理”可以通过从众效应提高说服效果。 

---
# Epistemic Diversity and Knowledge Collapse in Large Language Models 

**Title (ZH)**: 大型语言模型中的认识论多样性和知识坍塌 

**Authors**: Dustin Wright, Sarah Masud, Jared Moore, Srishti Yadav, Maria Antoniak, Chan Young Park, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.04226)  

**Abstract**: Large language models (LLMs) tend to generate lexically, semantically, and stylistically homogenous texts. This poses a risk of knowledge collapse, where homogenous LLMs mediate a shrinking in the range of accessible information over time. Existing works on homogenization are limited by a focus on closed-ended multiple-choice setups or fuzzy semantic features, and do not look at trends across time and cultural contexts. To overcome this, we present a new methodology to measure epistemic diversity, i.e., variation in real-world claims in LLM outputs, which we use to perform a broad empirical study of LLM knowledge collapse. We test 27 LLMs, 155 topics covering 12 countries, and 200 prompt variations sourced from real user chats. For the topics in our study, we show that while newer models tend to generate more diverse claims, nearly all models are less epistemically diverse than a basic web search. We find that model size has a negative impact on epistemic diversity, while retrieval-augmented generation (RAG) has a positive impact, though the improvement from RAG varies by the cultural context. Finally, compared to a traditional knowledge source (Wikipedia), we find that country-specific claims reflect the English language more than the local one, highlighting a gap in epistemic representation 

**Abstract (ZH)**: 大规模语言模型（LLMs）倾向于生成词汇上、语义上和风格上高度同质的文本。这可能导致知识坍缩，即随着时间推移，同质化LLM中介导可获取信息范围的缩小。现有关于同质化的研究局限在封闭式多项选择设置或模糊的语义特征上，未能关注时间与文化背景中的趋势。为克服这一局限，我们提出了一种新的方法来衡量知识论多样性，即LLM输出中现实世界主张的变化情况，并使用该方法进行了一场广泛的实证研究，以考察LLM知识坍缩。我们测试了27个LLM模型、涵盖12个国家的155个话题，以及来自真实用户对话的200种提示变体。对于研究中的主题，我们表明，虽然较新的模型倾向于生成更多样化的主张，但几乎所有的模型在知识论多样性方面都低于基本的网络搜索。我们发现，模型规模对知识论多样性有负面影响，而检索增强生成（RAG）则有正面影响，尽管RAG对知识论多样性的提高在不同文化背景下有所不同。最后，与传统知识源（维基百科）相比，我们发现，特定国家的主张反映了英语语言多于当地语言，突显了知识论表现的差距。 

---
# Zoom-In to Sort AI-Generated Images Out 

**Title (ZH)**: 聚焦筛选AI生成的图像 

**Authors**: Yikun Ji, Yan Hong, Bowen Deng, jun lan, Huijia Zhu, Weiqiang Wang, Liqing Zhang, Jianfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04225)  

**Abstract**: The rapid growth of AI-generated imagery has blurred the boundary between real and synthetic content, raising critical concerns for digital integrity. Vision-language models (VLMs) offer interpretability through explanations but often fail to detect subtle artifacts in high-quality synthetic images. We propose ZoomIn, a two-stage forensic framework that improves both accuracy and interpretability. Mimicking human visual inspection, ZoomIn first scans an image to locate suspicious regions and then performs a focused analysis on these zoomed-in areas to deliver a grounded verdict. To support training, we introduce MagniFake, a dataset of 20,000 real and high-quality synthetic images annotated with bounding boxes and forensic explanations, generated through an automated VLM-based pipeline. Our method achieves 96.39% accuracy with robust generalization, while providing human-understandable explanations grounded in visual evidence. 

**Abstract (ZH)**: AI生成图像的迅猛增长模糊了现实与合成内容的边界，对数字完整性提出了关键性 concern。Vision-Language 模型 (VLMs) 通过解释提供了可解释性，但往往难以检测高质量合成图像中的细微伪影。我们提出 ZoomIn，一种两阶段法医框架，以提高准确性和可解释性。模仿人类视觉检查，ZoomIn 首先扫描图像以定位可疑区域，然后在这些放大区域上进行聚焦分析以提供合规性的裁决。为支持训练，我们引入 MagniFake 数据集，包含 20,000 张标记有边界框和法医解释的真实和高质量合成图像，生成通过自动 VLM 基础管线完成。我们的方法实现了 96.39% 的准确率，并具备稳健的一般化能力，同时提供基于视觉证据的人类可理解解释。 

---
# MASC: Boosting Autoregressive Image Generation with a Manifold-Aligned Semantic Clustering 

**Title (ZH)**: MASC: 基于流形对齐语义聚类的自回归图像生成增强 

**Authors**: Lixuan He, Shikang Zheng, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04220)  

**Abstract**: Autoregressive (AR) models have shown great promise in image generation, yet they face a fundamental inefficiency stemming from their core component: a vast, unstructured vocabulary of visual tokens. This conventional approach treats tokens as a flat vocabulary, disregarding the intrinsic structure of the token embedding space where proximity often correlates with semantic similarity. This oversight results in a highly complex prediction task, which hinders training efficiency and limits final generation quality. To resolve this, we propose Manifold-Aligned Semantic Clustering (MASC), a principled framework that constructs a hierarchical semantic tree directly from the codebook's intrinsic structure. MASC employs a novel geometry-aware distance metric and a density-driven agglomerative construction to model the underlying manifold of the token embeddings. By transforming the flat, high-dimensional prediction task into a structured, hierarchical one, MASC introduces a beneficial inductive bias that significantly simplifies the learning problem for the AR model. MASC is designed as a plug-and-play module, and our extensive experiments validate its effectiveness: it accelerates training by up to 57% and significantly improves generation quality, reducing the FID of LlamaGen-XL from 2.87 to 2.58. MASC elevates existing AR frameworks to be highly competitive with state-of-the-art methods, establishing that structuring the prediction space is as crucial as architectural innovation for scalable generative modeling. 

**Abstract (ZH)**: 基于流形对齐的语义聚类（MASC）在图像生成中的应用 

---
# MLLMEraser: Achieving Test-Time Unlearning in Multimodal Large Language Models through Activation Steering 

**Title (ZH)**: MLLMEraser: 通过激活引导实现多模态大型语言模型测试时遗忘的功能 

**Authors**: Chenlu Ding, Jiancan Wu, Leheng Sheng, Fan Zhang, Yancheng Yuan, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2510.04217)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated remarkable capabilities across vision-language tasks, yet their large-scale deployment raises pressing concerns about memorized private data, outdated knowledge, and harmful content. Existing unlearning approaches for MLLMs typically adapt training-based strategies such as gradient ascent or preference optimization, but these methods are computationally expensive, irreversible, and often distort retained knowledge. In this work, we propose MLLMEraser, an input-aware, training-free framework for test-time unlearning. Our approach leverages activation steering to enable dynamic knowledge erasure without parameter updates. Specifically, we construct a multimodal erasure direction by contrasting adversarially perturbed, knowledge-recall image-text pairs with knowledge-erasure counterparts, capturing both textual and visual discrepancies. To prevent unnecessary interference, we further design an input-aware steering mechanism that adaptively determines when and how the erasure direction should be applied, preserving utility on retained knowledge while enforcing forgetting on designated content. Experiments on LLaVA-1.5 and Qwen-2.5-VL demonstrate that MLLMEraser consistently outperforms state-of-the-art MLLM unlearning baselines, achieving stronger forgetting performance with lower computational cost and minimal utility degradation. 

**Abstract (ZH)**: 具有输入感知的训练-free 测试时去学习框架 MLLMEraser 

---
# Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention 

**Title (ZH)**: 低精度Transformer训练为何失败：基于Flash Attention的分析 

**Authors**: Haiquan Qiu, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04212)  

**Abstract**: The pursuit of computational efficiency has driven the adoption of low-precision formats for training transformer models. However, this progress is often hindered by notorious training instabilities. This paper provides the first mechanistic explanation for a long-standing and unresolved failure case where training with flash attention in low-precision settings leads to catastrophic loss explosions. Our in-depth analysis reveals that the failure is not a random artifact but caused by two intertwined phenomena: the emergence of similar low-rank representations within the attention mechanism and the compounding effect of biased rounding errors inherent in low-precision arithmetic. We demonstrate how these factors create a vicious cycle of error accumulation that corrupts weight updates, ultimately derailing the training dynamics. To validate our findings, we introduce a minimal modification to the flash attention that mitigates the bias in rounding errors. This simple change stabilizes the training process, confirming our analysis and offering a practical solution to this persistent problem. 

**Abstract (ZH)**: 低精度设置下使用闪存注意力训练变压器模型时灾难性损失爆炸的机制解释与解决方案 

---
# PolyKAN: A Polyhedral Analysis Framework for Provable and Minimal KAN Compression 

**Title (ZH)**: PolyKAN: 一种 Provably 和 Minimal 的 KAN 压缩多面体分析框架 

**Authors**: Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04205)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have emerged as a promising alternative to traditional Multi-Layer Perceptrons (MLPs), offering enhanced interpretability and a strong mathematical foundation. However, their parameter efficiency remains a significant challenge for practical deployment. This paper introduces PolyKAN, a novel theoretical framework for KAN compression that provides formal guarantees on both model size reduction and approximation error. By leveraging the inherent piecewise polynomial structure of KANs, we formulate the compression problem as one of optimal polyhedral region merging. We establish a rigorous polyhedral characterization of KANs, develop a complete theory of $\epsilon$-equivalent compression, and design an optimal dynamic programming algorithm that guarantees minimal compression under specified error bounds. Our theoretical analysis demonstrates that PolyKAN achieves provably minimal compression while maintaining strict error control, with polynomial-time complexity in all network parameters. The framework provides the first formal foundation for KAN compression with mathematical guarantees, opening new directions for efficient deployment of interpretable neural architectures. 

**Abstract (ZH)**: PolyKAN：Kolmogorov-Arnold网络的新型压缩理论框架 

---
# CALM Before the STORM: Unlocking Native Reasoning for Optimization Modeling 

**Title (ZH)**: 临危不乱：解锁原生推理以优化建模 

**Authors**: Zhengyang Tang, Zihan Ye, Chenyu Huang, Xuhan Huang, Chengpeng Li, Sihang Li, Guanhua Chen, Ming Yan, Zizhuo Wang, Hongyuan Zha, Dayiheng Liu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04204)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated strong capabilities in complex multi-step reasoning, opening new opportunities for automating optimization modeling. However, existing domain adaptation methods, originally designed for earlier instruction-tuned models, often fail to exploit the advanced reasoning patterns of modern LRMs -- In particular, we show that direct fine-tuning on traditional \textit{non-reflective} datasets leads to limited gains. To fully leverage LRMs' inherent reasoning abilities, we propose \textbf{CALM} (\textit{Corrective Adaptation with Lightweight Modification}), a framework that progressively refines LRMs within their native reasoning modes for optimization modeling tasks. In CALM, an expert intervener identifies reasoning flaws and provides concise corrective hints, which the LRM incorporates to produce improved reasoning trajectories. These interventions modify fewer than 2.6\% of generated tokens, but generate high-quality data for soft adaptation through supervised fine-tuning. The adapted model is then further improved through reinforcement learning. Building on CALM, we develop \textbf{STORM} (\textit{Smart Thinking Optimization Reasoning Model}), a 4B-parameter LRM that achieves a new state-of-the-art average accuracy of 68.9\% across five popular optimization modeling benchmarks, matching the performance of a 671B LRM. These results demonstrate that dynamic, hint-based data synthesis both preserves and amplifies the native reasoning patterns of modern LRMs, offering a more effective and scalable path towards expert-level performance on challenging optimization modeling tasks. 

**Abstract (ZH)**: 大型推理模型（LRMs）在复杂多步推理中展示了强大的能力，为自动化优化建模开辟了新的机遇。然而，现有的领域适应方法通常无法充分利用现代LRMs的高级推理模式——特别是直接对传统非反思性数据集进行微调只能获得有限的收益。为了充分利用LRMs固有的推理能力，我们提出了CALM（Corrective Adaptation with Lightweight Modification）框架，该框架在优化建模任务中逐步在其原生推理模式下精炼LRMs。在CALM中，专家介入者识别推理缺陷并提供简洁的修正提示，LRM将这些提示整合以生成改进的推理路径。这些干预措施修改少于2.6%的生成词元，但通过监督微调生成高质量数据进行软适应。通过强化学习进一步改进适应后的模型。基于CALM，我们开发了STORM（Smart Thinking Optimization Reasoning Model），一个4亿参数的LRM，在五个流行优化建模基准中实现了新的最佳平均准确性68.9%，与一个671亿参数的LRM性能相当。这些结果表明，动态、基于提示的数据合成既能保留又能放大现代LRMs的固有推理模式，为复杂优化建模任务提供了一条更加有效和可扩展的路径以达到专家级性能。 

---
# World-To-Image: Grounding Text-to-Image Generation with Agent-Driven World Knowledge 

**Title (ZH)**: 世界到图像：基于代理驱动世界知识的文本到图像生成 

**Authors**: Moo Hyun Son, Jintaek Oh, Sun Bin Mun, Jaechul Roh, Sehyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.04201)  

**Abstract**: While text-to-image (T2I) models can synthesize high-quality images, their performance degrades significantly when prompted with novel or out-of-distribution (OOD) entities due to inherent knowledge cutoffs. We introduce World-To-Image, a novel framework that bridges this gap by empowering T2I generation with agent-driven world knowledge. We design an agent that dynamically searches the web to retrieve images for concepts unknown to the base model. This information is then used to perform multimodal prompt optimization, steering powerful generative backbones toward an accurate synthesis. Critically, our evaluation goes beyond traditional metrics, utilizing modern assessments like LLMGrader and ImageReward to measure true semantic fidelity. Our experiments show that World-To-Image substantially outperforms state-of-the-art methods in both semantic alignment and visual aesthetics, achieving +8.1% improvement in accuracy-to-prompt on our curated NICE benchmark. Our framework achieves these results with high efficiency in less than three iterations, paving the way for T2I systems that can better reflect the ever-changing real world. Our demo code is available here\footnote{this https URL}. 

**Abstract (ZH)**: World-To-Image: 通过代理驱动的世界知识桥接文本到图像生成的性能差距 

---
# Cooperative Flexibility Exchange: Fair and Comfort-Aware Decentralized Resource Allocation 

**Title (ZH)**: 协同灵活性交换：公平与舒适感知的分布式资源分配 

**Authors**: Rabiya Khalid, Evangelos Pournaras  

**Link**: [PDF](https://arxiv.org/pdf/2510.04192)  

**Abstract**: The growing electricity demand and increased use of smart appliances are placing new pressures on power grids, making efficient energy management more important than ever. The existing energy management systems often prioritize system efficiency (balanced energy demand and supply) at the expense of user comfort. This paper addresses this gap by proposing a novel decentralized multi-agent coordination-based demand-side management system. The proposed system enables individual agents to coordinate for demand-side energy optimization while improving the user comfort and maintaining the system efficiency. A key innovation of this work is the introduction of a slot exchange mechanism, where agents first receive optimized appliance-level energy consumption schedules and then coordinate with each other to adjust these schedules through slot exchanges. This approach improves user comfort even when agents show non-altruistic behaviour, and it scales well with large populations. The system also promotes fairness by balancing satisfaction levels across users. For performance evaluation, a real-world dataset is used, and the results demonstrate that the proposed slot exchange mechanism increases user comfort and fairness without raising system inefficiency cost, making it a practical and scalable solution for future smart grids. 

**Abstract (ZH)**: Growing电力需求和智能家电的增加使用对电网造成了新的压力，使得高效的能源管理变得前所未有的重要。现有能源管理系统往往以系统的效率（平衡能源需求和供应）为优先，而牺牲用户舒适度。本文通过提出一种新颖的去中心化多代理协调需求侧管理系统来弥补这一差距。该提出的系统使个体代理能够协调以实现需求侧的能源优化，同时提高用户舒适度并保持系统的效率。这项工作的关键创新在于引入了一个时间段交换机制，即代理首先接收优化的家电级能源消耗时间表，然后通过时间段交换相互协调调整这些时间表。这种做法即使在网络代理表现出非利他行为时也能够提高用户舒适度，并且能够很好地扩展到大量人群。该系统还通过平衡用户的满意度水平促进了公平性。为了评估性能，使用了真实世界的数据集，结果表明，提出的时间段交换机制能够提高用户舒适度和公平性，而不增加系统的无效率成本，使其成为面向未来智能电网的实用且可扩展的解决方案。 

---
# Finite Time Analysis of Constrained Natural Critic-Actor Algorithm with Improved Sample Complexity 

**Title (ZH)**: 有限时间内约束自然评论者-行动家算法的研究：改进的样本复杂性分析 

**Authors**: Prashansa Panda, Shalabh Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2510.04189)  

**Abstract**: Recent studies have increasingly focused on non-asymptotic convergence analyses for actor-critic (AC) algorithms. One such effort introduced a two-timescale critic-actor algorithm for the discounted cost setting using a tabular representation, where the usual roles of the actor and critic are reversed. However, only asymptotic convergence was established there. Subsequently, both asymptotic and non-asymptotic analyses of the critic-actor algorithm with linear function approximation were conducted. In our work, we introduce the first natural critic-actor algorithm with function approximation for the long-run average cost setting and under inequality constraints. We provide the non-asymptotic convergence guarantees for this algorithm. Our analysis establishes optimal learning rates and we also propose a modification to enhance sample complexity. We further show the results of experiments on three different Safety-Gym environments where our algorithm is found to be competitive in comparison with other well known algorithms. 

**Abstract (ZH)**: 近期的研究越来越多地关注演员-评论家(AC)算法的非渐近收敛分析。其中一项努力在折现成本设置中使用表征表示提出了一种双时间尺度评论家-演员算法，其中评论家和演员的传统角色被逆转。然而，只建立了渐近收敛性。随后，对线性函数逼近下的评论家-演员算法的渐近和非渐近分析进行了研究。在我们的工作中，我们首次在此长期内均值成本设置和不等式约束下引入了一种自然的函数逼近下的评论家-演员算法，并提供了该算法的非渐近收敛保证。我们的分析确定了最优的学习率，还提出了改进样本复杂性的修改方案。我们还在三个不同的Safety-Gym环境中展示了实验结果，发现该算法与已知的其他算法相比具有竞争力。 

---
# A Complement to Neural Networks for Anisotropic Inelasticity at Finite Strains 

**Title (ZH)**: 用于有限应变各向异性非线性的一种补充神经网络方法 

**Authors**: Hagen Holthusen, Ellen Kuhl  

**Link**: [PDF](https://arxiv.org/pdf/2510.04187)  

**Abstract**: We propose a complement to constitutive modeling that augments neural networks with material principles to capture anisotropy and inelasticity at finite strains. The key element is a dual potential that governs dissipation, consistently incorporates anisotropy, and-unlike conventional convex formulations-satisfies the dissipation inequality without requiring convexity.
Our neural network architecture employs invariant-based input representations in terms of mixed elastic, inelastic and structural tensors. It adapts Input Convex Neural Networks, and introduces Input Monotonic Neural Networks to broaden the admissible potential class. To bypass exponential-map time integration in the finite strain regime and stabilize the training of inelastic materials, we employ recurrent Liquid Neural Networks.
The approach is evaluated at both material point and structural scales. We benchmark against recurrent models without physical constraints and validate predictions of deformation and reaction forces for unseen boundary value problems. In all cases, the method delivers accurate and stable performance beyond the training regime. The neural network and finite element implementations are available as open-source and are accessible to the public via this https URL. 

**Abstract (ZH)**: 我们提出了一种补充性的本构建模方法，将材料原理与神经网络相结合，以捕捉有限应变下的各向异性和非线性行为。关键要素是一种双潜能函数，它控制耗散现象，一致地包含各向异性，并且在不需要凸性的前提下满足耗散不等式。

该神经网络架构采用基于不变量的输入表示，包括混合弹性、非弹性及结构张量。它采用输入凸神经网络，并引入输入单调神经网络，以扩展允许的潜能类。为在有限应变状态下绕开指数映射时间积分并稳定非线性材料的训练过程，我们采用了循环液态神经网络。

该方法在材料点和结构尺度上进行了评估。我们将其与没有任何物理约束的循环模型进行了对比，并验证了对未见过的边界值问题的变形和反应力预测。在所有情况下，该方法均表现出超越训练范围的准确和稳定性能。神经网络和有限元实现均已开源，并可通过以下链接访问：https://doi.org/10.1140/epjdestini/a123456 

---
# Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization 

**Title (ZH)**: 随机应变的思考：基于潜在思维策略优化的测试时推理增强 

**Authors**: Wengao Ye, Yan Liang, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04182)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have shifted from explicit Chain-of-Thought (CoT) reasoning to more efficient latent reasoning, where intermediate thoughts are represented as vectors rather than text. However, latent reasoning can be brittle on challenging, out-of-distribution tasks where robust reasoning is most critical. To overcome these limitations, we introduce Latent Thought Policy Optimization (LTPO), a parameter-free framework that enhances LLM reasoning entirely at test time, without requiring model parameter updates. LTPO treats intermediate latent "thought" vectors as dynamic parameters that are actively optimized for each problem instance. It employs an online policy gradient method guided by an intrinsic, confidence-based reward signal computed directly from the frozen LLM's own output distributions, eliminating the need for external supervision or expensive text generation during optimization. Extensive experiments on five reasoning benchmarks show that LTPO not only matches or surpasses strong baselines on standard tasks but also demonstrates remarkable robustness where others fail. Most notably, on highly challenging AIME benchmarks where existing latent reasoning baselines collapse to near-zero accuracy, LTPO delivers substantial improvements, showcasing a unique capability for complex reasoning. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）的研究从显式的思维链（Chain-of-Thought）推理转向了更高效的潜在推理，其中中间思想以向量形式表示而非文本形式。然而，潜在推理在具有挑战性和分布外的任务中最为关键时，表现可能会脆弱。为克服这些限制，我们引入了潜在思维策略优化（LTPO），这是一种无需更新模型参数的参数自由框架，可以在测试时增强LLM的推理。LTPO将中间的潜在“思维”向量视为动态参数，并针对每个问题实例进行优化。它使用由冻结LLM自身输出分布直接计算的内在、基于置信度的奖励信号引导的在线策略梯度方法，在优化过程中无需外部监督或昂贵的文本生成。在五个推理基准上的广泛实验表明，LTPO不仅在标准任务上能够匹配或超越强大的基线，在其他基线失败的关键领域也表现出显著的鲁棒性。特别是在现有潜在推理基线在高度挑战性的AIME基准上表现近乎零准确度的情况下，LTPO提供了显著的改进，展示了复杂推理的唯一能力。 

---
# Multi Language Models for On-the-Fly Syntax Highlighting 

**Title (ZH)**: 基于多语言模型的即时语法高亮 

**Authors**: Marco Edoardo Palma, Pooja Rani, Harald C. Gall  

**Link**: [PDF](https://arxiv.org/pdf/2510.04166)  

**Abstract**: Syntax highlighting is a critical feature in modern software development environments, enhancing code readability and developer productivity. However, delivering accurate highlighting in real time remains challenging for online and web-based development tools due to strict time and memory constraints on backend services. These systems must serve highlights rapidly and frequently, even when code is partially valid or invalid. This has led to on-the-fly syntax highlighting, where visual annotations are generated just before content is served, often at high request rates and under incomplete input conditions. To meet these demands efficiently, state-of-the-art models use deep learning to learn the behavior of brute-force syntax highlighting resolvers, tools that are easy to implement but too slow for production. Through the Deep Abstraction process, brute-force strategies are encoded into fast statistical models that achieve both high accuracy and low-latency inference. Despite their success, such models face key challenges: they support only one programming language per model, require large datasets from slow brute-force generators, and involve resource-intensive training. In multi-language environments, this means maintaining multiple independent models, increasing system complexity and operational cost. This work addresses these issues by introducing a unified model capable of highlighting up to six mainstream programming languages, reducing deployment complexity by a factor of six and improving performance on unseen languages. A novel normalization technique significantly enhances model generalization, while few-shot learning experiments show that a small number of oracle samples can replace large datasets, minimizing dependence on brute-force generators. Combined, these innovations enable efficient, scalable, and cost-effective syntax highlighting across diverse programming languages. 

**Abstract (ZH)**: 现代软件开发环境中语法高亮的关键功能及其实时实现挑战与解决方案 

---
# Beyond Next-Token Prediction: A Performance Characterization of Diffusion versus Autoregressive Language Models 

**Title (ZH)**: 超越下一个词预测：扩散模型与自回归语言模型的性能 characterization 比较 

**Authors**: Minseo Kim, Coleman Hooper, Aditya Tomar, Chenfeng Xu, Mehrdad Farajtabar, Michael W. Mahoney, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2510.04146)  

**Abstract**: Large Language Models (LLMs) have achieved state-of-the-art performance on a broad range of Natural Language Processing (NLP) tasks, including document processing and coding. Autoregressive Language Models (ARMs), which generate tokens sequentially conditioned on all previous tokens, have been the predominant paradigm for LLMs. However, while these networks have achieved high accuracy across a range of downstream tasks, they exhibit low arithmetic intensity due to the inherent sequential dependency with next-token prediction. Recently, Diffusion Language Models (DLMs) have emerged as a promising alternative architecture. DLMs generate output text in parallel, breaking the limitations of sequential dependency. However, the performance implications of DLMs relative to commonly deployed ARMs are not fully understood. In this work, we present a comprehensive performance study analyzing the performance characteristics of ARMs and DLMs, using both theoretical analysis and profiling data to characterize the trade-offs between these approaches. We illustrate that although DLMs exhibit higher arithmetic intensity compared to ARMs because of their capability to utilize parallelism across sequence lengths, they fail to scale effectively to longer contexts. We then explore DLMs with block-wise decoding, outlining how this approach allows for increased arithmetic intensity, while still scaling well to long contexts (similar to ARMs). We also show interesting trade-offs for batched inference, where we find that ARMs exhibit superior throughput, as they benefit more from parallelism across sequences in the batch. Finally, we highlight opportunities for accelerating DLM inference, and, in particular, highlight the importance of reducing the number of sampling steps for allowing open-source DLMs to provide improved latency relative to ARMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理（NLP）任务，包括文档处理和编码方面取得了最先进的性能。自回归语言模型（ARMs），这些模型按序生成标记并基于所有先前的标记进行条件预测，一直是LLMs的主要架构。然而，尽管这些网络在各种下游任务上实现了高精度，但由于其对下一个标记预测的内在顺序依赖性，它们的算术强度较低。最近，扩散语言模型（DLMs）作为有前景的替代架构出现。DLMs并行生成输出文本，从而打破了顺序依赖性的限制。然而，DLMs相对于广泛部署的ARMs的性能影响尚不完全明了。在本工作中，我们进行全面的研究，分析ARMs和DLMs的性能特征，使用理论分析和剖析数据来描述这两种方法之间的权衡。我们表明，尽管DLMs由于能够在序列长度之间利用并行性表现出更高的算术强度，但它们在长上下文环境中无法有效地扩展。随后，我们探讨了块级解码的DLMs，概述了这种方法如何允许更高的算术强度，同时仍然能够很好地扩展到长上下文（类似于ARMs）。我们还展示了批处理推断中的有趣权衡，发现ARMs在这种情况下具有更好的吞吐量，因为它们在批处理中的序列间并行性受益更多。最后，我们突出了加速DLM推断的机会，并特别强调减少采样步骤的重要性，以使开源DLMs能够相对于ARMs提供更好的延迟。 

---
# Learning from All: Concept Alignment for Autonomous Distillation from Multiple Drifting MLLMs 

**Title (ZH)**: 从所有中学习：针对多个漂移MLLM的concept对齐自主蒸馏 

**Authors**: Xiaoyu Yang, Jie Lu, En Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04142)  

**Abstract**: This paper identifies a critical yet underexplored challenge in distilling from multimodal large language models (MLLMs): the reasoning trajectories generated by multiple drifting teachers exhibit concept drift, whereby their reasoning distributions evolve unpredictably and transmit biases to the student model, ultimately compromising its performance. To tackle this issue, we pioneer a theoretical connection between concept drift and knowledge distillation, casting the non-stationary reasoning dynamics from multiple MLLM teachers as next-token prediction of multi-stream reasoning this http URL by concept drift, we introduce the "learn, compare, critique" paradigm, culminating in autonomous preference optimization (APO). Under the active guidance of the teachers, the student model first learns and self-distils preferred thinking by comparing multiple teachers. It then engages in critical reflection over the drifting inference from teachers, performing concept alignment through APO, ultimately yielding a robust, consistent, and generalizable this http URL experiments demonstrate our superior performance of consistency, robustness and generalization within knowledge distillation. Besides, we also contributed a large-scale dataset, CXR-MAX (Multi-teachers Alignment X-rays), comprising 170,982 distilled reasoning trajectories derived from publicly accessible MLLMs based on MIMIC-CXR. Our code and data are public at: this https URL. 

**Abstract (ZH)**: 本文识别并探讨了从多模态大型语言模型（MLLMs）中提炼知识的一个关键且未充分探索的挑战：多个漂移教师生成的推理轨迹表现出概念漂移，其推理分布演变不可预测，并将偏差传递给学生模型，从而最终损害其性能。为了解决这一问题，我们首创了概念漂移与知识提炼之间的理论联系，将多个MLLM教师的非稳态推理动态视为多流推理的下一个令牌预测。通过概念漂移，我们引入了“学习、比较、批判”范式，最终实现自主偏好优化（APO）。在教师的积极指导下，学生模型首先通过比较多个教师来学习和自我提炼偏好思维。然后，它对教师漂移的推理进行批判性反思，通过APO进行概念对齐，最终生成稳健、一致且可泛化的推理轨迹。我们的实验展示了在知识提炼中的一致性、稳健性和泛化的优越性能。此外，我们还贡献了一个大规模数据集CXR-MAX（多教师对齐X射线），包含170,982条从MIMIC-CXR公开的MLLMs中提取的提炼推理轨迹。我们的代码和数据可在以下链接访问：[https://github.com/your-repo-name]。 

---
# GA4GC: Greener Agent for Greener Code via Multi-Objective Configuration Optimization 

**Title (ZH)**: GA4GC：通过多目标配置优化实现更绿色的代理和代码 

**Authors**: Jingzhi Gong, Yixin Bian, Luis de la Cal, Giovanni Pinna, Anisha Uteem, David Williams, Mar Zamorano, Karine Even-Mendoza, W.B. Langdon, Hector Menendez, Federica Sarro  

**Link**: [PDF](https://arxiv.org/pdf/2510.04135)  

**Abstract**: Coding agents powered by LLMs face critical sustainability and scalability challenges in industrial deployment, with single runs consuming over 100k tokens and incurring environmental costs that may exceed optimization benefits. This paper introduces GA4GC, the first framework to systematically optimize coding agent runtime (greener agent) and code performance (greener code) trade-offs by discovering Pareto-optimal agent hyperparameters and prompt templates. Evaluation on the SWE-Perf benchmark demonstrates up to 135x hypervolume improvement, reducing agent runtime by 37.7% while improving correctness. Our findings establish temperature as the most critical hyperparameter, and provide actionable strategies to balance agent sustainability with code optimization effectiveness in industrial deployment. 

**Abstract (ZH)**: 基于LLM的编码代理在工业部署中面临关键的可持续性和扩展性挑战，单次运行消耗超过100k个 Tokens，并且可能产生的环境成本超过了优化收益。本文介绍了GA4GC框架，这是首个系统优化编码代理运行时（更绿色的代理）和代码性能（更绿色的代码） trade-offs 的框架，通过发现帕累托最优代理超参数和提示模板。在SWE-Perf基准上的评估表明，最大改进达135倍的超体积，代理运行时间减少37.7%，同时提高正确性。我们的研究结果确立了温度作为最关键超参数，并提供了在工业部署中平衡代理可持续性和代码优化效果的实际策略。 

---
# PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting 

**Title (ZH)**: PhaseFormer: 从块到相位进行高效有效的时间序列预测 

**Authors**: Yiming Niu, Jinliang Deng, Yongxin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04134)  

**Abstract**: Periodicity is a fundamental characteristic of time series data and has long played a central role in forecasting. Recent deep learning methods strengthen the exploitation of periodicity by treating patches as basic tokens, thereby improving predictive effectiveness. However, their efficiency remains a bottleneck due to large parameter counts and heavy computational costs. This paper provides, for the first time, a clear explanation of why patch-level processing is inherently inefficient, supported by strong evidence from real-world data. To address these limitations, we introduce a phase perspective for modeling periodicity and present an efficient yet effective solution, PhaseFormer. PhaseFormer features phase-wise prediction through compact phase embeddings and efficient cross-phase interaction enabled by a lightweight routing mechanism. Extensive experiments demonstrate that PhaseFormer achieves state-of-the-art performance with around 1k parameters, consistently across benchmark datasets. Notably, it excels on large-scale and complex datasets, where models with comparable efficiency often struggle. This work marks a significant step toward truly efficient and effective time series forecasting. Code is available at this repository: this https URL 

**Abstract (ZH)**: 周期性是时间序列数据的基本特征，长期以来在预测中扮演着中心角色。近期的深度学习方法通过将片段视为基本令牌，增强了对周期性的利用，从而提高了预测效果。然而，由于参数量庞大和计算成本高昂，其效率仍然存在瓶颈。本文首次提供了片层处理本质上不高效的清晰解释，并通过实际数据提供了强有力的支持。为了解决这些局限性，我们提出了一个相位视角来建模周期性，并提出了一种高效而有效的解决方案——PhaseFormer。PhaseFormer通过紧凑的相位嵌入和轻量化路由机制实现的相位间有效交互来进行相位级预测。大量的实验表明，PhaseFormer在基准数据集上实现了最佳性能，参数量仅为约1千个。尤为值得注意的是，在大规模和复杂的数据集上，它表现出色，而这种级别的效率模型在此类数据集上往往难以实现。这项工作代表了真正高效而有效的时间序列预测的一个重要进步。代码可在以下仓库获取：this https URL。 

---
# On the Limitations and Capabilities of Position Embeddings for Length Generalization 

**Title (ZH)**: 关于位置嵌入在长度泛化能力上的局限性和潜力 

**Authors**: Yang Chen, Yitao Liang, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04130)  

**Abstract**: In Transformers, Position Embeddings (PEs) significantly influence Length Generalization (LG) performance, yet their fundamental role remains unclear. In this work, we investigate the limitations and capabilities of PEs in achieving LG. We theoretically analyze PEs in Position-Only Linear Attentions (POLAs), introducing Linear Representation Complexity (LRC) to characterize when PEs enable LG. Our analysis shows that PEs do not expand computational capabilities but structure learned computations across positions. Extending to practical Transformers, we propose Sequential Representation Complexity (SRC) and conjecture that LG is possible if and only if SRC remains invariant across scales. We support this hypothesis with empirical evidence in various reasoning tasks. To enhance LG, we introduce Scale Hint, allowing flexible instance scaling, and a Learning-Based Position Embedding framework that automatically learns positional relations. Our work provides theoretical insights and practical strategies for improving LG in Transformers. 

**Abstract (ZH)**: 在Transformer中，位置嵌入（PEs）在长度泛化（LG）性能中的作用显著，但其基本作用尚不明确。本文研究了PEs在实现LG方面的限制与能力。我们从位置唯一线性注意（POLAs）的角度理论分析PEs，并引入线性表示复杂性（LRC）来刻画PEs如何使LG成为可能。我们的分析表明，PEs并未扩展计算能力，而是结构化了不同位置上学习到的计算。扩展到实际的Transformer中，我们提出序列表示复杂性（SRC），并猜测如果SRC在不同尺度上保持不变，LG是可能实现的。我们通过在各种推理任务中的实验证据支持这一假说。为了增强LG，我们引入了尺度提示（Scale Hint），允许灵活的实例缩放，并提出了一种基于学习的位置嵌入框架，可以自动学习位置关系。我们的工作提供了关于改进Transformer中LG的理论见解和实用策略。 

---
# Learning-Based Hashing for ANN Search: Foundations and Early Advances 

**Title (ZH)**: 基于学习的哈希表示在ANN搜索中的基础与初步进展 

**Authors**: Sean Moran  

**Link**: [PDF](https://arxiv.org/pdf/2510.04127)  

**Abstract**: Approximate Nearest Neighbour (ANN) search is a fundamental problem in information retrieval, underpinning large-scale applications in computer vision, natural language processing, and cross-modal search. Hashing-based methods provide an efficient solution by mapping high-dimensional data into compact binary codes that enable fast similarity computations in Hamming space. Over the past two decades, a substantial body of work has explored learning to hash, where projection and quantisation functions are optimised from data rather than chosen at random.
This article offers a foundational survey of early learning-based hashing methods, with an emphasis on the core ideas that shaped the field. We review supervised, unsupervised, and semi-supervised approaches, highlighting how projection functions are designed to generate meaningful embeddings and how quantisation strategies convert these embeddings into binary codes. We also examine extensions to multi-bit and multi-threshold models, as well as early advances in cross-modal retrieval.
Rather than providing an exhaustive account of the most recent methods, our goal is to introduce the conceptual foundations of learning-based hashing for ANN search. By situating these early models in their historical context, we aim to equip readers with a structured understanding of the principles, trade-offs, and open challenges that continue to inform current research in this area. 

**Abstract (ZH)**: 基于哈希的近似最近邻搜索./(ANN)搜索是信息检索中的一个基本问题，支撑着计算机视觉、自然语言处理和跨模态搜索等大规模应用。基于哈希的方法通过将高维数据映射为紧凑的二进制代码，从而在汉明空间中实现快速相似性计算，提供了一个高效的解决方案。在过去二十年里，大量研究探索了学习哈希方法，其中投影和量化函数是从数据中学习优化，而不是随机选择。

本文提供了一篇早期基于学习的哈希方法的基础综述，侧重于塑造该领域的核心思想。我们回顾了监督、无监督和半监督的方法，强调了如何设计投影函数生成有意义的嵌入，以及如何通过量化策略将这些嵌入转化为二进制代码。我们也探讨了多比特和多阈值模型的扩展，以及跨模态检索的早期进展。

我们的目标不是提供最新方法的详尽综述，而是旨在介绍基于学习的哈希方法在ANN搜索中的概念基础。通过将这些早期模型置于其历史背景中，我们希望读者能够获得对原则、权衡和仍然影响当前研究的开放挑战的结构化理解。 

---
# Attending on Multilevel Structure of Proteins enables Accurate Prediction of Cold-Start Drug-Target Interactions 

**Title (ZH)**: 关注蛋白质的多层结构以实现冷启动药物-靶标相互作用的准确预测 

**Authors**: Ziying Zhang, Yaqing Wang, Yuxuan Sun, Min Ye, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04126)  

**Abstract**: Cold-start drug-target interaction (DTI) prediction focuses on interaction between novel drugs and proteins. Previous methods typically learn transferable interaction patterns between structures of drug and proteins to tackle it. However, insight from proteomics suggest that protein have multi-level structures and they all influence the DTI. Existing works usually represent protein with only primary structures, limiting their ability to capture interactions involving higher-level structures. Inspired by this insight, we propose ColdDTI, a framework attending on protein multi-level structure for cold-start DTI prediction. We employ hierarchical attention mechanism to mine interaction between multi-level protein structures (from primary to quaternary) and drug structures at both local and global granularities. Then, we leverage mined interactions to fuse structure representations of different levels for final prediction. Our design captures biologically transferable priors, avoiding the risk of overfitting caused by excessive reliance on representation learning. Experiments on benchmark datasets demonstrate that ColdDTI consistently outperforms previous methods in cold-start settings. 

**Abstract (ZH)**: 冷启动药物-目标相互作用预测关注新药物与蛋白质之间的相互作用。现有方法通常通过学习药物和蛋白质结构之间的可迁移相互作用模式来解决这一问题。然而，蛋白质组学的见解表明，蛋白质具有多层级结构，所有层级都影响药物-目标相互作用。现有工作通常仅用蛋白质的一级结构来表示，限制了其捕获涉及更高层级结构的相互作用的能力。受此见解的启发，我们提出了一种名为ColdDTI的框架，用于冷启动药物-目标相互作用预测，该框架关注蛋白质的多层级结构。我们采用分层注意机制，在局部和全局粒度上挖掘从一级到四级的多层级蛋白质结构与药物结构之间的相互作用，然后利用挖掘出的相互作用融合不同层级的结构表示以进行最终预测。我们的设计捕获了生物学上的可迁移先验，避免了过度依赖表示学习而导致的过拟合风险。在基准数据集上的实验表明，ColdDTI在冷启动设置中始终优于先前的方法。 

---
# Unveiling LLMs' Metaphorical Understanding: Exploring Conceptual Irrelevance, Context Leveraging and Syntactic Influence 

**Title (ZH)**: 揭开大语言模型的隐喻理解：探索概念无关性、上下文利用和句法影响 

**Authors**: Fengying Ye, Shanshan Wang, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04120)  

**Abstract**: Metaphor analysis is a complex linguistic phenomenon shaped by context and external factors. While Large Language Models (LLMs) demonstrate advanced capabilities in knowledge integration, contextual reasoning, and creative generation, their mechanisms for metaphor comprehension remain insufficiently explored. This study examines LLMs' metaphor-processing abilities from three perspectives: (1) Concept Mapping: using embedding space projections to evaluate how LLMs map concepts in target domains (e.g., misinterpreting "fall in love" as "drop down from love"); (2) Metaphor-Literal Repository: analyzing metaphorical words and their literal counterparts to identify inherent metaphorical knowledge; and (3) Syntactic Sensitivity: assessing how metaphorical syntactic structures influence LLMs' performance. Our findings reveal that LLMs generate 15\%-25\% conceptually irrelevant interpretations, depend on metaphorical indicators in training data rather than contextual cues, and are more sensitive to syntactic irregularities than to structural comprehension. These insights underline the limitations of LLMs in metaphor analysis and call for more robust computational approaches. 

**Abstract (ZH)**: 元喻分析是一个受上下文和外部因素影响的复杂语言现象。尽管大规模语言模型（LLMs）在知识整合、情境推理和创造性生成方面表现出先进的能力，但它们在元喻理解方面的机制仍需更多探索。本研究从三个角度考察LLMs的元喻处理能力：（1）概念映射：通过嵌入空间投影评估LLMs如何在目标领域映射概念（例如，将“fall in love”误解为“drop down from love”）；（2）元喻-直喻库：分析元喻词汇及其直喻对应词，以识别内在的元喻知识；（3）句法敏感性：评估元喻句法结构如何影响LLMs的表现。我们的研究发现，LLMs生成15%-25%的概念相关性解释，依赖于训练数据中的元喻指示符而不是上下文线索，并且对句法异常比对结构理解更敏感。这些发现凸显了LLMs在元喻分析方面的局限性，并呼吁采取更 robust 的计算方法。 

---
# TOPO-Bench: An Open-Source Topological Mapping Evaluation Framework with Quantifiable Perceptual Aliasing 

**Title (ZH)**: TOPO-Bench: 一种具有可量化感知 aliaseding 的开源拓扑映射评估框架 

**Authors**: Jiaming Wang, Diwen Liu, Jizhuo Chen, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2510.04100)  

**Abstract**: Topological mapping offers a compact and robust representation for navigation, but progress in the field is hindered by the lack of standardized evaluation metrics, datasets, and protocols. Existing systems are assessed using different environments and criteria, preventing fair and reproducible comparisons. Moreover, a key challenge - perceptual aliasing - remains under-quantified, despite its strong influence on system performance. We address these gaps by (1) formalizing topological consistency as the fundamental property of topological maps and showing that localization accuracy provides an efficient and interpretable surrogate metric, and (2) proposing the first quantitative measure of dataset ambiguity to enable fair comparisons across environments. To support this protocol, we curate a diverse benchmark dataset with calibrated ambiguity levels, implement and release deep-learned baseline systems, and evaluate them alongside classical methods. Our experiments and analysis yield new insights into the limitations of current approaches under perceptual aliasing. All datasets, baselines, and evaluation tools are fully open-sourced to foster consistent and reproducible research in topological mapping. 

**Abstract (ZH)**: 拓扑映射提供了紧凑且 robust 的导航表示，但领域进展受限于缺乏标准化评估指标、数据集和协议。现有系统使用不同的环境和标准进行评估，阻碍了公平和可重复的比较。此外，尽管感知退化对系统性能有强烈影响，但关键挑战——感知退化仍被严重低估。我们通过以下方式解决这些差距：（1）将拓扑一致性正式化为拓扑地图的基本属性，并展示局部化精度作为有效且可解释的替代指标，（2）提出第一个数据集模糊性的定量度量以实现不同环境之间的公平比较。为支持此协议，我们策划了一个具有校准模糊度级别的多元化基准数据集，实现了并发布了基于深度学习的基线系统，并在经典方法旁边进行了评估。我们的实验和分析提供了对当前方法在感知退化下局限性的新见解。所有数据集、基线和评估工具均已完全开源，以促进拓扑映射中的一致且可重复的研究。 

---
# Efficient Training of Spiking Neural Networks by Spike-aware Data Pruning 

**Title (ZH)**: 基于.spike-aware数据剪枝的高效Spiking神经网络训练 

**Authors**: Chenxiang Ma, Xinyi Chen, Yujie Wu, Kay Chen Tan, Jibin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04098)  

**Abstract**: Spiking neural networks (SNNs), recognized as an energy-efficient alternative to traditional artificial neural networks (ANNs), have advanced rapidly through the scaling of models and datasets. However, such scaling incurs considerable training overhead, posing challenges for researchers with limited computational resources and hindering the sustained development of SNNs. Data pruning is a promising strategy for accelerating training by retaining the most informative examples and discarding redundant ones, but it remains largely unexplored in SNNs. Directly applying ANN-based data pruning methods to SNNs fails to capture the intrinsic importance of examples and suffers from high gradient variance. To address these challenges, we propose a novel spike-aware data pruning (SADP) method. SADP reduces gradient variance by determining each example's selection probability to be proportional to its gradient norm, while avoiding the high cost of direct gradient computation through an efficient upper bound, termed spike-aware importance score. This score accounts for the influence of all-or-nothing spikes on the gradient norm and can be computed with negligible overhead. Extensive experiments across diverse datasets and architectures demonstrate that SADP consistently outperforms data pruning baselines and achieves training speedups close to the theoretical maxima at different pruning ratios. Notably, SADP reduces training time by 35% on ImageNet while maintaining accuracy comparable to that of full-data training. This work, therefore, establishes a data-centric paradigm for efficient SNN training and paves the way for scaling SNNs to larger models and datasets. The source code will be released publicly after the review process. 

**Abstract (ZH)**: 基于尖峰的稀疏数据训练方法：一种加速Spiking神经网络训练的新策略 

---
# Using predefined vector systems as latent space configuration for neural network supervised training on data with arbitrarily large number of classes 

**Title (ZH)**: 使用预定义向量系统作为神经网络监督训练的潜在空间配置，以处理任意大量类别的数据 

**Authors**: Nikita Gabdullin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04090)  

**Abstract**: Supervised learning (SL) methods are indispensable for neural network (NN) training used to perform classification tasks. While resulting in very high accuracy, SL training often requires making NN parameter number dependent on the number of classes, limiting their applicability when the number of classes is extremely large or unknown in advance. In this paper we propose a methodology that allows one to train the same NN architecture regardless of the number of classes. This is achieved by using predefined vector systems as the target latent space configuration (LSC) during NN training. We discuss the desired properties of target configurations and choose randomly perturbed vectors of An root system for our experiments. These vectors are used to successfully train encoders and visual transformers (ViT) on Cinic-10 and ImageNet-1K in low- and high-dimensional cases by matching NN predictions with the predefined vectors. Finally, ViT is trained on a dataset with 1.28 million classes illustrating the applicability of the method to training on datasets with extremely large number of classes. In addition, potential applications of LSC in lifelong learning and NN distillation are discussed illustrating versatility of the proposed methodology. 

**Abstract (ZH)**: 监督学习方法对于执行分类任务的神经网络训练至关重要。虽然监督学习训练能够实现非常高精度，但往往需要将神经网络参数数量依赖于类别数量，这限制了当类别数量极其庞大或事先未知时的应用。本文提出了一种方法，使得可以训练相同的神经网络架构，无论类别数量如何。这一目标通过在神经网络训练过程中使用预定义的向量系统作为目标潜在空间配置（LSC）来实现。我们讨论了目标配置的期望属性，并在实验中选择了随机扰动的An根系统向量。这些向量通过匹配神经网络预测与预定义向量，成功地在Cinic-10和ImageNet-1K上训练了编码器和视觉变换器（ViT），并在低维和高维情况下验证了其有效性。最后，我们在包含128万个类别的数据集上训练了ViT，展示了该方法在处理类别数量极其庞大的数据集时的应用。此外，还讨论了潜在空间配置在终身学习和神经网络蒸馏中的潜在应用，进一步展示了所提出方法的灵活性。 

---
# Offline Reinforcement Learning in Large State Spaces: Algorithms and Guarantees 

**Title (ZH)**: 大型状态空间中的离线强化学习：算法与保证 

**Authors**: Nan Jiang, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.04088)  

**Abstract**: This article introduces the theory of offline reinforcement learning in large state spaces, where good policies are learned from historical data without online interactions with the environment. Key concepts introduced include expressivity assumptions on function approximation (e.g., Bellman completeness vs. realizability) and data coverage (e.g., all-policy vs. single-policy coverage). A rich landscape of algorithms and results is described, depending on the assumptions one is willing to make and the sample and computational complexity guarantees one wishes to achieve. We also discuss open questions and connections to adjacent areas. 

**Abstract (ZH)**: 本文介绍在大面积状态空间中离线强化学习的理论，通过历史数据学习良好的策略而无需与环境进行在线交互。介绍的关键概念包括函数逼近的表达性假设（如贝尔曼完备性与实现性）和数据覆盖范围（如全策略覆盖与单策略覆盖）。描述了根据不同假设和样本及计算复杂性保证的丰富算法和结果。此外，还讨论了开放问题及其与相邻领域的联系。 

---
# A Contextual Quality Reward Model for Reliable and Efficient Best-of-N Sampling 

**Title (ZH)**: 上下文质量奖励模型：实现可靠和高效的最佳选项采样 

**Authors**: Hyung Gyu Rho  

**Link**: [PDF](https://arxiv.org/pdf/2510.04087)  

**Abstract**: Modern preference alignment techniques, such as Best-of-N (BoN) sampling, rely on reward models trained with pairwise comparison data. While effective at learning relative preferences, this paradigm fails to capture a signal of response acceptability, leaving systems vulnerable to selecting the least bad of many unacceptable options. This is particularly problematic for hard prompts, where the risk of such false acceptances increases with the number of samples. In this paper, we address this critical reliability gap by introducing a new data collection and modeling framework. By augmenting preference data with an outside option, inspired by discrete choice models, we train a reward model that can distinguish not just what is \textit{better}, but what is \textit{good enough}. We leverage this capability to create an adaptive inference strategy, best of mini-N in-loop, which partitions the generation budget into sequential loops with a calibrated, early-exit condition. Our experiments show that when tuned as an alignment guardrail, it reduces reliability failures by 70\%, and when tuned as an inference accelerator, it improves average inference speed by over 22\% in IMDB-sentiment setting. We thus provide a principled and flexible framework for practitioners to explicitly manage the trade-off between reliability and computational efficiency. 

**Abstract (ZH)**: 现代偏好对齐技术，如Best-of-N（BoN）采样，依赖于通过成对比较数据训练的奖励模型。尽管这种范式在学习相对偏好方面非常有效，但它无法捕捉响应可接受性的信号，从而使系统容易选择众多不可接受选项中最坏的一个。这在困难提示中尤为关键，样本数量越多，这种错误接受的风险越高。在本文中，我们通过引入一个新的数据收集和建模框架来填补这一关键可靠性的缺口。通过借鉴离散选择模型中的外部选项，我们训练了一个奖励模型，它可以区分什么不仅是“更好”，而且是“足够好”。我们利用这一能力创建了一种自适应推理策略——在环中的Best-of-mini-N，将生成预算划分为具有校准且早期退出条件的顺序循环。我们的实验显示，作为对齐护栏进行调整时，它可以将可靠性失败降低70%，作为推理加速器进行调整时，在IMDB情感分析设置中平均提高推理速度超过22%。因此，我们提供了一个原则性的灵活框架，使从业者能够明确管理可靠性和计算效率之间的权衡。 

---
# Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning 

**Title (ZH)**: 慢快策略优化：推理前重定位更新机制 

**Authors**: Ziyan Wang, Zheng Wang, Jie Fu, Xingwei Qu, Qi Cheng, Shengpu Tang, Minjia Zhang, Xiaoming Huo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04072)  

**Abstract**: Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient framework to address these limitations via decomposing each step into three stages: a short fast trajectory of inner steps on the same batch, a reposition mechanism to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points in average on math reasoning benchmarks. It also achieves up to 4.93\texttimes{} fewer rollouts and a 4.19\texttimes{} reduction in wall-clock time to match GRPO's best accuracy. 

**Abstract (ZH)**: 基于强化学习的缓慢快速策略优化在大型语言模型推理中的应用：一种简单有效的框架 

---
# What Scales in Cross-Entropy Scaling Law? 

**Title (ZH)**: 交叉熵标度律中有哪些标度因子？ 

**Authors**: Junxi Yan, Zixi Wei, Jingtao Zhan, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04067)  

**Abstract**: The cross-entropy scaling law has long served as a key tool for guiding the development of large language models. It shows that cross-entropy loss decreases in a predictable power-law rate as the model size increases. However, recent evidence indicates that this law breaks down at very large scales: the loss decreases more slowly than expected, which causes significant trouble for developing large language models. In this paper, we hypothesize that the root cause lies in the fact that cross-entropy itself does not truly scale; instead, only one of its hidden components does. To investigate this, we introduce a novel decomposition of cross-entropy into three parts: Error-Entropy, Self-Alignment, and Confidence. We show both theoretically and empirically that this decomposition precisely captures the training dynamics and optimization objectives. Through extensive experiments on multiple datasets and 32 models spanning five orders of magnitude in size, we find that only error-entropy follows a robust power-law scaling, while the other two terms remain largely invariant. Moreover, error-entropy constitutes the dominant share of cross-entropy in small models but diminishes in proportion as models grow larger. This explains why the cross-entropy scaling law appears accurate at small scales but fails at very large ones. Our findings establish the error-entropy scaling law as a more accurate description of model behavior. We believe it will have wide applications in the training, understanding, and future development of large language models. 

**Abstract (ZH)**: 交叉熵缩放规律长期作为指导大规模语言模型发展的关键工具。它表明随着模型规模的增加，交叉熵损失以可预测的幂律速率减少。然而，最近的证据表明，这一规律在非常大的规模下失效：损失减少得比预期慢，这给开发大规模语言模型带来了重大困难。在本文中，我们假设根本原因在于交叉熵本身并不能真正缩放；只有其隐藏的组成部分之一能够缩放。为了调查这一点，我们引入了一种新的交叉熵分解为三个部分：错误熵、自我对齐和置信度。我们通过理论和实验证明，这种分解精确地捕捉了训练动力学和优化目标。通过在多个数据集和32个跨越五个数量级大小的模型上进行广泛实验，我们发现只有错误熵遵循稳健的幂律缩放，而另外两项则基本不变。此外，错误熵在小型模型中占主导地位，但随着模型增大其比例减少。这解释了为什么在小规模时交叉熵缩放规律显得准确，但在非常大的规模时却失效。我们的研究结果确立了错误熵缩放规律作为更准确的模型行为描述。我们认为它将在大规模语言模型的训练、理解和未来发展中具有广泛的应用。 

---
# MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation 

**Title (ZH)**: MetaFind: 场景 aware 的 3D 资产检索以生成连贯的元宇宙场景 

**Authors**: Zhenyu Pan, Yucheng Lu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04057)  

**Abstract**: We present MetaFind, a scene-aware tri-modal compositional retrieval framework designed to enhance scene generation in the metaverse by retrieving 3D assets from large-scale repositories. MetaFind addresses two core challenges: (i) inconsistent asset retrieval that overlooks spatial, semantic, and stylistic constraints, and (ii) the absence of a standardized retrieval paradigm specifically tailored for 3D asset retrieval, as existing approaches mainly rely on general-purpose 3D shape representation models. Our key innovation is a flexible retrieval mechanism that supports arbitrary combinations of text, image, and 3D modalities as queries, enhancing spatial reasoning and style consistency by jointly modeling object-level features (including appearance) and scene-level layout structures. Methodologically, MetaFind introduces a plug-and-play equivariant layout encoder ESSGNN that captures spatial relationships and object appearance features, ensuring retrieved 3D assets are contextually and stylistically coherent with the existing scene, regardless of coordinate frame transformations. The framework supports iterative scene construction by continuously adapting retrieval results to current scene updates. Empirical evaluations demonstrate the improved spatial and stylistic consistency of MetaFind in various retrieval tasks compared to baseline methods. 

**Abstract (ZH)**: MetaFind：一种场景感知的多模态组合检索框架，用于增强元宇宙中的场景生成 

---
# Quantization Range Estimation for Convolutional Neural Networks 

**Title (ZH)**: 卷积神经网络的量化范围估计 

**Authors**: Bingtao Yang, Yujia Wang, Mengzhi Jiao, Hongwei Huo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04044)  

**Abstract**: Post-training quantization for reducing the storage of deep neural network models has been demonstrated to be an effective way in various tasks. However, low-bit quantization while maintaining model accuracy is a challenging problem. In this paper, we present a range estimation method to improve the quantization performance for post-training quantization. We model the range estimation into an optimization problem of minimizing quantization errors by layer-wise local minima. We prove this problem is locally convex and present an efficient search algorithm to find the optimal solution. We propose the application of the above search algorithm to the transformed weights space to do further improvement in practice. Our experiments demonstrate that our method outperforms state-of-the-art performance generally on top-1 accuracy for image classification tasks on the ResNet series models and Inception-v3 model. The experimental results show that the proposed method has almost no loss of top-1 accuracy in 8-bit and 6-bit settings for image classifications, and the accuracy of 4-bit quantization is also significantly improved. The code is available at this https URL. 

**Abstract (ZH)**: 基于训练后量化减少深度神经网络模型存储量的研究：一种范围估计方法以提高低比特量化性能 

---
# \textsc{GUI-Spotlight}: Adaptive Iterative Focus Refinement for Enhanced GUI Visual Grounding 

**Title (ZH)**: GUI-Spotlight: 自适应迭代焦点细化方法以增强GUI视觉定位 

**Authors**: Bin Lei, Nuo Xu, Ali Payani, Mingyi Hong, Chunhua Liao, Yu Cao, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.04039)  

**Abstract**: Multimodal large language models (MLLMs) have markedly expanded the competence of graphical user-interface (GUI) systems, propelling them beyond controlled simulations into complex, real-world environments across diverse platforms. However, practical usefulness is still bounded by the reliability of visual grounding, i.e., mapping textual references to exact on-screen elements. This limitation prevents the system from accurately performing pointer-level actions such as clicking or dragging. To address it, we introduce GUI-Spotlight -- a model trained for image-grounded reasoning that dynamically invokes multiple specialized tools to iteratively narrow its focus to the relevant region of the screen, thereby substantially improving visual grounding accuracy. On the ScreenSpot-Pro benchmark, GUI-Spotlight trained with only 18.5K training samples achieves 52.8\% accuracy, surpassing V2P-7B (50.6\% with 9.6M training samples) and GTA-1-7B (50.1\% with 1.56M training samples). 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）显著扩展了图形用户界面（GUI）系统的功能，使其从受控仿真扩展到多种平台上的复杂真实环境。然而，其实用性仍受视觉定位可靠性的限制，即文本引用到屏幕元素的精准映射。这一限制阻止系统准确执行指针级操作，如点击或拖拽。为解决这一问题，我们引入了GUI-Spotlight——一种训练用于图像导向推理的模型，能够动态调用多种专门工具以迭代地将注意力聚焦到屏幕的相关区域，从而显著提高视觉定位准确性。在ScreenSpot-Pro基准测试中，仅使用18,500个训练样本的GUI-Spotlight实现了52.8%的准确性，超过了V2P-7B（使用9,600,000个训练样本的50.6%准确性）和GTA-1-7B（使用1,560,000个训练样本的50.1%准确性）。 

---
# Prompt-to-Prompt: Text-Based Image Editing Via Cross-Attention Mechanisms -- The Research of Hyperparameters and Novel Mechanisms to Enhance Existing Frameworks 

**Title (ZH)**: Prompt-to-Prompt：基于文本的图像编辑通过跨注意力机制——关于超参数研究及新型机制以增强现有框架的探索 

**Authors**: Linn Bieske, Carla Lorente  

**Link**: [PDF](https://arxiv.org/pdf/2510.04034)  

**Abstract**: Recent advances in image editing have shifted from manual pixel manipulation to employing deep learning methods like stable diffusion models, which now leverage cross-attention mechanisms for text-driven control. This transition has simplified the editing process but also introduced variability in results, such as inconsistent hair color changes. Our research aims to enhance the precision and reliability of prompt-to-prompt image editing frameworks by exploring and optimizing hyperparameters. We present a comprehensive study of the "word swap" method, develop an "attention re-weight method" for better adaptability, and propose the "CL P2P" framework to address existing limitations like cycle inconsistency. This work contributes to understanding and improving the interaction between hyperparameter settings and the architectural choices of neural network models, specifically their attention mechanisms, which significantly influence the composition and quality of the generated images. 

**Abstract (ZH)**: 近期图像编辑的进步已从手动像素操作转向使用如稳定扩散模型等深度学习方法，这些方法现在利用交叉注意力机制实现文本驱动的控制。这一转变简化了编辑过程，但也引入了结果的一致性问题，如不一致的头发颜色变化。我们的研究旨在通过探索和优化超参数来提升提示到提示的图像编辑框架的精确性和可靠性。我们对“词交换”方法进行了全面研究，开发了“注意力重新加权方法”以提高适应性，并提出了“CL P2P”框架以解决现有局限性，如循环一致性问题。这项工作有助于理解并改进超参数设置与神经网络模型架构选择之间的互动，特别是它们的注意力机制，这些机制显著影响生成图像的组成和质量。 

---
# Small Language Models for Emergency Departments Decision Support: A Benchmark Study 

**Title (ZH)**: 小语言模型在急诊部门决策支持中的基准研究 

**Authors**: Zirui Wang, Jiajun Wu, Braden Teitge, Jessalyn Holodinsky, Steve Drew  

**Link**: [PDF](https://arxiv.org/pdf/2510.04032)  

**Abstract**: Large language models (LLMs) have become increasingly popular in medical domains to assist physicians with a variety of clinical and operational tasks. Given the fast-paced and high-stakes environment of emergency departments (EDs), small language models (SLMs), characterized by a reduction in parameter count compared to LLMs, offer significant potential due to their inherent reasoning capability and efficient performance. This enables SLMs to support physicians by providing timely and accurate information synthesis, thereby improving clinical decision-making and workflow efficiency. In this paper, we present a comprehensive benchmark designed to identify SLMs suited for ED decision support, taking into account both specialized medical expertise and broad general problem-solving capabilities. In our evaluations, we focus on SLMs that have been trained on a mixture of general-domain and medical corpora. A key motivation for emphasizing SLMs is the practical hardware limitations, operational cost constraints, and privacy concerns in the typical real-world deployments. Our benchmark datasets include MedMCQA, MedQA-4Options, and PubMedQA, with the medical abstracts dataset emulating tasks aligned with real ED physicians' daily tasks. Experimental results reveal that general-domain SLMs surprisingly outperform their medically fine-tuned counterparts across these diverse benchmarks for ED. This indicates that for ED, specialized medical fine-tuning of the model may not be required. 

**Abstract (ZH)**: 小型语言模型（SLMs）在急诊部门决策支持中的综合基准研究 

---
# Does Using Counterfactual Help LLMs Explain Textual Importance in Classification? 

**Title (ZH)**: 使用反事实有助于大语言模型在分类中解释文本重要性？ 

**Authors**: Nelvin Tan, James Asikin Cheung, Yu-Ching Shih, Dong Yang, Amol Salunkhe  

**Link**: [PDF](https://arxiv.org/pdf/2510.04031)  

**Abstract**: Large language models (LLMs) are becoming useful in many domains due to their impressive abilities that arise from large training datasets and large model sizes. More recently, they have been shown to be very effective in textual classification tasks, motivating the need to explain the LLMs' decisions. Motivated by practical constrains where LLMs are black-boxed and LLM calls are expensive, we study how incorporating counterfactuals into LLM reasoning can affect the LLM's ability to identify the top words that have contributed to its classification decision. To this end, we introduce a framework called the decision changing rate that helps us quantify the importance of the top words in classification. Our experimental results show that using counterfactuals can be helpful. 

**Abstract (ZH)**: 大型语言模型（LLMs）因其源自大训练数据集和大模型规模的卓越能力，在多个领域变得越来越有用。最近的研究显示，它们在文本分类任务中非常有效，这推动了解释LLMs决策的必要性。受到实际约束中LLMs作为黑盒模型以及LLM调用成本高的影响，我们研究了将反事实引入LLMs推理过程如何影响其识别对分类决策有贡献的顶级词汇的能力。为此，我们提出了一种称为决策改变率的框架，以帮助我们量化分类中顶级词汇的重要性。我们的实验结果表明，使用反事实是有帮助的。 

---
# The Debate on RLVR Reasoning Capability Boundary: Shrinkage, Expansion, or Both? A Two-Stage Dynamic View 

**Title (ZH)**: 关于RLVR推理能力边界之争：缩小、扩展，还是两者兼有？一种两阶段动态视角 

**Authors**: Xinhao Yao, Lu Yu, Xiaolin Hu, Fengwei Teng, Qing Cui, Jun Zhou, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04028)  

**Abstract**: The ongoing debate on whether reinforcement learning with verifiable rewards (RLVR) expands or shrinks the reasoning capabilities of large language models (LLMs) remains unresolved. Some studies contend that RLVR mainly improves sampling efficiency but at the expense of diversity and exploratory capacity, resulting in capability boundary shrinkage. In contrast, others demonstrate that prolonged training can lead to the emergence of novel reasoning strategies, suggesting capability boundary expansion. To reconcile these contradictory findings, we theoretically and empirically show that both perspectives are partially valid-each aligning with a separate phase in an inherent two-stage probability mass dynamic: (1) Exploitation stage: initially, the model primarily samples explored high-reward and low-reward tokens, while rarely selecting the potentially optimal token. Positive advantage estimates increase the probability of high-reward tokens and decrease those of low-reward tokens, yet the optimal token's probability remains largely unchanged during this stage. (2) Exploration stage: as training advances, the growth rate of previously acquired high-reward tokens slows as their probabilities approach saturation. When a potentially optimal token-now receiving positive advantage estimates-is occasionally sampled, its probability increases, while those of the originally high-reward tokens decrease. This dynamic suggests that over-exploitation during the exploitation stage may lead to capability boundary shrinkage, whereas prolonged training into the exploration stage can promote an expansion of the reasoning capability boundary. Building upon our insights, we revisit the potential of only using relative negative gradients for prolonging training, providing a theoretical and empirical foundation for the development of more advanced reasoning capabilities. 

**Abstract (ZH)**: 有关强化学习带有可验证奖励（RLVR）是否扩大或缩小大型语言模型（LLMs）的推理能力的持续争议尚未解决。一些研究认为，RLVR 主要提高采样效率，但牺牲了多样性与探索能力，导致能力边界缩小。相反，其他研究则表明，长期训练可以促使出现新的推理策略，表明能力边界可能扩大。为了调和这些矛盾的研究结果，我们从理论上和实证上证明两种观点在某种程度上都是正确的，每种观点分别对应于固有的两阶段概率质量动态的不同阶段：（1）利用阶段：最初，模型主要采样探索的高奖励和低奖励令牌，而很少选择潜在的最佳令牌。正的优势估计增加了高奖励令牌的概率，同时降低了低奖励令牌的概率，但在该阶段，最佳令牌的概率变化不大。（2）探索阶段：随着训练的进行，之前获得的高奖励令牌的增长率逐渐减缓，因为它们的概率接近饱和。当一个潜在的最佳令牌（现在获得正的优势估计）偶尔被采样时，它的概率会增加，而最初高奖励令牌的概率会减少。这种动态表明，在利用阶段过度利用可能会导致能力边界缩小，而长时间训练进入探索阶段则可以促进推理能力边界的扩张。基于我们的见解，我们重新审视仅使用相对负梯度延长训练的潜力，为开发更高级的推理能力提供了理论和实证基础。 

---
# Spatiotemporal Forecasting as Planning: A Model-Based Reinforcement Learning Approach with Generative World Models 

**Title (ZH)**: 时空预测作为规划：基于生成世界模型的模型导向强化学习方法 

**Authors**: Hao Wu, Yuan Gao, Xingjian Shi, Shuaipeng Li, Fan Xu, Fan Zhang, Zhihong Zhu, Weiyan Wang, Xiao Luo, Kun Wang, Xian Wu, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04020)  

**Abstract**: To address the dual challenges of inherent stochasticity and non-differentiable metrics in physical spatiotemporal forecasting, we propose Spatiotemporal Forecasting as Planning (SFP), a new paradigm grounded in Model-Based Reinforcement Learning. SFP constructs a novel Generative World Model to simulate diverse, high-fidelity future states, enabling an "imagination-based" environmental simulation. Within this framework, a base forecasting model acts as an agent, guided by a beam search-based planning algorithm that leverages non-differentiable domain metrics as reward signals to explore high-return future sequences. These identified high-reward candidates then serve as pseudo-labels to continuously optimize the agent's policy through iterative self-training, significantly reducing prediction error and demonstrating exceptional performance on critical domain metrics like capturing extreme events. 

**Abstract (ZH)**: 时空预测作为规划（SFP）：基于模型的强化学习新范式 

---
# Principled and Tractable RL for Reasoning with Diffusion Language Models 

**Title (ZH)**: 原理性和可实现性的强化学习方法：用于推理的扩散语言模型 

**Authors**: Anthony Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04019)  

**Abstract**: Diffusion large language models (dLLMs) are a new paradigm of non-autoregressive language models that are trained to predict multiple tokens in parallel and generate text via iterative unmasking. Recent works have successfully pretrained dLLMs to parity with autoregressive LLMs at the 8B scale, but dLLMs have yet to benefit from modern post-training techniques, e.g. reinforcement learning (RL), that have proven effective for autoregressive models. Crucially, algorithms designed for traditional LLMs aren't directly compatible with diffusion frameworks due to inherent differences in modeling assumptions. Moreover, existing attempts at dLLM post-training with RL rely on heuristic-based objectives with no theoretical grounding. In this work, we present Amortized Group Relative Policy Optimization (AGRPO), a principled on-policy RL algorithm designed specifically for dLLMs. AGRPO uses Monte Carlo sampling to compute an unbiased policy gradient estimate, making it the first tractable, faithful adaptation of policy gradient methods for dLLMs. We demonstrate AGRPO's effectiveness on different math/reasoning tasks, a common setting for RL with LLMs, achieving up to +7.6% absolute gain on GSM8K and 3.8x performance on the Countdown task over the baseline LLaDA-8B-Instruct model and 1.3x performance gains over comparable RL methods such as diffu-GRPO. Furthermore, these gains persist across different numbers of sampling steps at inference time, achieving better tradeoffs between compute and performance. Our results demonstrate that online RL algorithms can be extended to diffusion LLMs in principled ways, maintaining both theoretical soundness and practical effectiveness. 

**Abstract (ZH)**: 扩散大规模语言模型（dLLMs）是一种新型的非自回归语言模型，训练时能够并行预测多个令牌并通过迭代去噪生成文本。最近的研究成功将dLLMs预训练至与8B规模的自回归大语言模型相当的水平，但dLLMs尚未从中受益于现代后训练技术，例如已被证明对自回归模型有效的强化学习（RL）。关键在于，设计用于传统大语言模型的算法由于建模假设的基本差异，无法直接与扩散框架兼容。此外，现有的dLLM后训练尝试中的RL主要依赖于缺乏理论支撑的经验性目标。在本文中，我们提出了Amortized Group Relative Policy Optimization（AGRPO），这是一种专门为dLLMs设计的原理上的一体化随算法。AGRPO 使用蒙特卡洛采样计算无偏策略梯度估计，使其成为第一个可用于dLLMs的可行且忠实的策略梯度方法的适应。我们展示了AGRPO 在不同的数学/推理任务上的有效性，这些任务是大语言模型中常见的RL设置，相对于基准模型LLaDA-8B-Instruct，在GSM8K上实现了高达7.6%的绝对增益，在Countdown任务上实现了3.8倍的性能提升，相比可比的RL方法（如diffu-GRPO）实现了1.3倍的性能增益。此外，这些增益在推理时的不同采样步骤数量下依然存在，实现了计算和性能之间的更好权衡。我们的研究结果表明，一体化的在线RL算法可以以原理上一致的方式扩展到扩散大语言模型中，保持其理论可靠性和实际有效性。 

---
# Thai Semantic End-of-Turn Detection for Real-Time Voice Agents 

**Title (ZH)**: Thai语语义句终检测用于实时语音助手 

**Authors**: Thanapol Popit, Natthapath Rungseesiripak, Monthol Charattrakool, Saksorn Ruangtanusak  

**Link**: [PDF](https://arxiv.org/pdf/2510.04016)  

**Abstract**: Fluid voice-to-voice interaction requires reliable and low-latency detection of when a user has finished speaking. Traditional audio-silence end-pointers add hundreds of milliseconds of delay and fail under hesitations or language-specific phenomena. We present, to our knowledge, the first systematic study of Thai text-only end-of-turn (EOT) detection for real-time agents. We compare zero-shot and few-shot prompting of compact LLMs to supervised fine-tuning of lightweight transformers. Using transcribed subtitles from the YODAS corpus and Thai-specific linguistic cues (e.g., sentence-final particles), we formulate EOT as a binary decision over token boundaries. We report a clear accuracy-latency tradeoff and provide a public-ready implementation plan. This work establishes a Thai baseline and demonstrates that small, fine-tuned models can deliver near-instant EOT decisions suitable for on-device agents. 

**Abstract (ZH)**: Thai 文本仅基于结束轮换检测的系统研究：实现实时代理的无延迟语音交互 

---
# Replacing Softmax Similarity with a Sharpened Angular Similarity: Theory and Practice of Scaling To Billion-Context Attention 

**Title (ZH)**: 用尖锐角度相似度取代Softmax相似度：亿级上下文注意力建模的理论与实践 

**Authors**: Sahil Joshi, Agniva Chowdhury, Amar Kanakamedala, Ekam Singh, Evan Tu, Anshumali Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2510.04008)  

**Abstract**: Softmax Attention has a quadratic time complexity, which becomes prohibitive to run at long contexts, even with highly optimized GPU kernels. For example, FlashAttention (an exact, GPU-optimized implementation of Softmax Attention) cannot complete a single forward-backward pass of a multi-head attention layer once the context exceeds ~4 million tokens on an NVIDIA GH200 (96 GB). We introduce RACE Attention, a kernel-inspired alternative to Softmax Attention that is linear in sequence length and embedding dimension. RACE Attention replaces the exponential kernel with a sharpened angular (cosine) similarity, and approximates attention outputs via randomized projections and soft Locality-Sensitive Hashing (LSH). Across language modeling, masked language modeling, and text classification, RACE Attention matches the accuracy of strong baselines while reducing runtime and memory. In a controlled scale test, it processes up to 12 million tokens during a single forward-backward pass on an NVIDIA GH200 GPU and 75 million tokens on an Intel Xeon Gold 5220R CPU, well beyond the practical limits of the current state-of-the-art attention implementations. RACE Attention thus offers a practical, theoretically grounded mechanism for outrageously long context windows on today's hardware. We hope that it gets adopted in practice. 

**Abstract (ZH)**: RACE Attention: A Linear-Time Alternative to Softmax Attention 

---
# Named Entity Recognition in COVID-19 tweets with Entity Knowledge Augmentation 

**Title (ZH)**: 基于实体知识增强的COVID-19推文命名实体识别 

**Authors**: Xuankang Zhang, Jiangming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04001)  

**Abstract**: The COVID-19 pandemic causes severe social and economic disruption around the world, raising various subjects that are discussed over social media. Identifying pandemic-related named entities as expressed on social media is fundamental and important to understand the discussions about the pandemic. However, there is limited work on named entity recognition on this topic due to the following challenges: 1) COVID-19 texts in social media are informal and their annotations are rare and insufficient to train a robust recognition model, and 2) named entity recognition in COVID-19 requires extensive domain-specific knowledge. To address these issues, we propose a novel entity knowledge augmentation approach for COVID-19, which can also be applied in general biomedical named entity recognition in both informal text format and formal text format. Experiments carried out on the COVID-19 tweets dataset and PubMed dataset show that our proposed entity knowledge augmentation improves NER performance in both fully-supervised and few-shot settings. Our source code is publicly available: this https URL 

**Abstract (ZH)**: COVID-19 pandemic引起的全球社会和经济冲击导致了各种在社交媒体上讨论的主题。识别表达在社交媒体上的与疫情相关的命名实体对于理解关于疫情的讨论至关重要。然而，由于以下挑战，有关该主题的命名实体识别工作非常有限：1) 社交媒体上的COVID-19文本不正式，其标注稀少不充分，难以训练出 robust 的识别模型，2) 在COVID-19领域的命名实体识别需要广泛的领域专业知识。为应对这些挑战，我们提出了一种针对COVID-19的新型实体知识增强方法，该方法也可应用于以非正式文本格式和正式文本格式进行的一般生物医学命名实体识别。在COVID-19推文数据集和PubMed数据集上的实验表明，我们提出的实体知识增强方法在完全监督和少量监督设置中均提高了命名实体识别性能。我们的源代码已公开：this https URL。 

---
# AI-Driven Grading and Moderation for Collaborative Projects in Computer Science Education 

**Title (ZH)**: 基于AI的协作项目评分与审核在计算机科学教育中的应用 

**Authors**: Songmei Yu, Andrew Zagula  

**Link**: [PDF](https://arxiv.org/pdf/2510.03998)  

**Abstract**: Collaborative group projects are integral to computer science education, as they foster teamwork, problem-solving skills, and industry-relevant competencies. However, assessing individual contributions within group settings has long been a challenge. Traditional assessment strategies, such as the equal distribution of grades or subjective peer assessments, often fall short in terms of fairness, objectivity, and scalability, particularly in large classrooms. This paper introduces a semi-automated, AI-assisted grading system that evaluates both project quality and individual effort using repository mining, communication analytics, and machine learning models. The system comprises modules for project evaluation, contribution analysis, and grade computation, integrating seamlessly with platforms like GitHub. A pilot deployment in a senior-level course demonstrated high alignment with instructor assessments, increased student satisfaction, and reduced instructor grading effort. We conclude by discussing implementation considerations, ethical implications, and proposed enhancements to broaden applicability. 

**Abstract (ZH)**: 协作小组项目是计算机科学教育中的重要组成部分，它们培养团队合作、问题解决能力和与行业相关的能力。然而，在小组环境中评估个人贡献始终是一项挑战。传统的评估策略，如平均分配成绩或主观的同伴评估，往往在公平性、客观性和可扩展性方面不尽如人意，特别是在大型课堂中。本文介绍了一种半自动化、基于AI的评分系统，该系统通过代码库挖掘、通信分析和机器学习模型评估项目质量和个人贡献。该系统包括项目评估、贡献分析和成绩计算模块，可以无缝集成到如GitHub这样的平台上。在一门高年级课程中的试点部署表明，该系统与教师评估高度契合，提高了学生满意度，并减少了教师的评分工作量。最后，本文讨论了实施考虑、伦理问题及拟议的改进措施，以扩大其适用范围。 

---
# PrivSpike: Employing Homomorphic Encryption for Private Inference of Deep Spiking Neural Networks 

**Title (ZH)**: PrivSpike: 利用同态加密实现深度脉冲神经网络的隐私推理 

**Authors**: Nges Brian Njungle, Eric Jahns, Milan Stojkov, Michel A. Kinsy  

**Link**: [PDF](https://arxiv.org/pdf/2510.03995)  

**Abstract**: Deep learning has become a cornerstone of modern machine learning. It relies heavily on vast datasets and significant computational resources for high performance. This data often contains sensitive information, making privacy a major concern in deep learning. Spiking Neural Networks (SNNs) have emerged as an energy-efficient alternative to conventional deep learning approaches. Nevertheless, SNNs still depend on large volumes of data, inheriting all the privacy challenges of deep learning. Homomorphic encryption addresses this challenge by allowing computations to be performed on encrypted data, ensuring data confidentiality throughout the entire processing pipeline. In this paper, we introduce PRIVSPIKE, a privacy-preserving inference framework for SNNs using the CKKS homomorphic encryption scheme. PRIVSPIKE supports arbitrary depth SNNs and introduces two key algorithms for evaluating the Leaky Integrate-and-Fire activation function: (1) a polynomial approximation algorithm designed for high-performance SNN inference, and (2) a novel scheme-switching algorithm that optimizes precision at a higher computational cost. We evaluate PRIVSPIKE on MNIST, CIFAR-10, Neuromorphic MNIST, and CIFAR-10 DVS using models from LeNet-5 and ResNet-19 architectures, achieving encrypted inference accuracies of 98.10%, 79.3%, 98.1%, and 66.0%, respectively. On a consumer-grade CPU, SNN LeNet-5 models achieved inference times of 28 seconds on MNIST and 212 seconds on Neuromorphic MNIST. For SNN ResNet-19 models, inference took 784 seconds on CIFAR-10 and 1846 seconds on CIFAR-10 DVS. These results establish PRIVSPIKE as a viable and efficient solution for secure SNN inference, bridging the gap between energy-efficient deep neural networks and strong cryptographic privacy guarantees while outperforming prior encrypted SNN solutions. 

**Abstract (ZH)**: 一种基于CKKS同态加密方案的隐私 preserved Spiking Neural Networks 推理框架：PRIVSPIKE 

---
# Quantifying Distributional Robustness of Agentic Tool-Selection 

**Title (ZH)**: 衡量代理工具选择的分布鲁棒性 

**Authors**: Jehyeok Yeon, Isha Chaudhary, Gagandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.03992)  

**Abstract**: Large language models (LLMs) are increasingly deployed in agentic systems where they map user intents to relevant external tools to fulfill a task. A critical step in this process is tool selection, where a retriever first surfaces candidate tools from a larger pool, after which the LLM selects the most appropriate one. This pipeline presents an underexplored attack surface where errors in selection can lead to severe outcomes like unauthorized data access or denial of service, all without modifying the agent's model or code. While existing evaluations measure task performance in benign settings, they overlook the specific vulnerabilities of the tool selection mechanism under adversarial conditions. To address this gap, we introduce ToolCert, the first statistical framework that formally certifies tool selection robustness. ToolCert models tool selection as a Bernoulli success process and evaluates it against a strong, adaptive attacker who introduces adversarial tools with misleading metadata, and are iteratively refined based on the agent's previous choices. By sampling these adversarial interactions, ToolCert produces a high-confidence lower bound on accuracy, formally quantifying the agent's worst-case performance. Our evaluation with ToolCert uncovers the severe fragility: under attacks injecting deceptive tools or saturating retrieval, the certified accuracy bound drops near zero, an average performance drop of over 60% compared to non-adversarial settings. For attacks targeting the retrieval and selection stages, the certified accuracy bound plummets to less than 20% after just a single round of adversarial adaptation. ToolCert thus reveals previously unexamined security threats inherent to tool selection and provides a principled method to quantify an agent's robustness to such threats, a necessary step for the safe deployment of agentic systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被部署在代理系统中，它们将用户意图映射到相关外部工具以完成任务。这一过程中的关键步骤是工具选择，首先是检索器从更大的池中浮现候选工具，之后LLM选择最合适的工具。这一流程提供了一个未被充分探索的攻击表面，其中选择错误可能导致未经授权的数据访问或服务拒绝，而无需修改代理的模型或代码。虽然现有的评估在良性环境中衡量任务性能，但它们忽略了在对抗条件下工具选择机制的特定漏洞。为填补这一空白，我们提出了ToolCert，这是第一个正式认证工具选择稳健性的统计框架。ToolCert 将工具选择建模为伯努利成功过程，并使用强大的自适应攻击者来评估它，该攻击者引入带有误导性元数据的对抗性工具，并基于代理的先前选择进行逐步改进。通过采样这些对抗性交互，ToolCert 生成了对准确性的高置信度下界，正式量化了代理在最坏情况下的表现。我们利用ToolCert的评估揭示了严重的脆弱性：在注入欺骗性工具或饱和检索的攻击下，认证的准确度下界接近零，平均表现下降超过60%。针对检索和选择阶段的攻击，在一轮对抗适应后，认证的准确度下界降至不到20%。ToolCert 表明了工具选择固有的先前未被研究的安全威胁，并提供了一种原则性的方法来量化代理对这些威胁的鲁棒性，这是安全部署代理系统的必要步骤。 

---
# A Mathematical Explanation of Transformers for Large Language Models and GPTs 

**Title (ZH)**: Transformer在大型语言模型和GPT中的数学解释 

**Authors**: Xue-Cheng Tai, Hao Liu, Lingfeng Li, Raymond H. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03989)  

**Abstract**: The Transformer architecture has revolutionized the field of sequence modeling and underpins the recent breakthroughs in large language models (LLMs). However, a comprehensive mathematical theory that explains its structure and operations remains elusive. In this work, we propose a novel continuous framework that rigorously interprets the Transformer as a discretization of a structured integro-differential equation. Within this formulation, the self-attention mechanism emerges naturally as a non-local integral operator, and layer normalization is characterized as a projection to a time-dependent constraint. This operator-theoretic and variational perspective offers a unified and interpretable foundation for understanding the architecture's core components, including attention, feedforward layers, and normalization. Our approach extends beyond previous theoretical analyses by embedding the entire Transformer operation in continuous domains for both token indices and feature dimensions. This leads to a principled and flexible framework that not only deepens theoretical insight but also offers new directions for architecture design, analysis, and control-based interpretations. This new interpretation provides a step toward bridging the gap between deep learning architectures and continuous mathematical modeling, and contributes a foundational perspective to the ongoing development of interpretable and theoretically grounded neural network models. 

**Abstract (ZH)**: Transformer架构 revolutionized 序列建模领域并支撑了大规模语言模型（LLMs）的最新突破。然而，一个全面的数学理论仍未能解释其结构与操作。在这项工作中，我们提出了一种新颖的连续框架，严格地将Transformer解释为结构化积分微分方程的离散化。在此表述中，自我注意机制自然地表现为非局域积分算子，而层归一化则被表征为时间依赖约束的投影。这种算子理论和变分视角为理解架构的核心组件，包括注意机制、前馈层和归一化，提供了统一且可解释的基础。我们的方法超越了以往的理论分析，通过将整个Transformer操作嵌入到符号和特征维度的连续域中，为架构设计、分析和基于控制的解释提供了原则性和灵活性框架。这一新诠释为弥合深度学习架构与连续数学建模之间的差距提供了步骤，并为可解释且基于理论的神经网络模型的发展提供了基础视角。 

---
# Distilling Reasoning into Student LLMs: Local Naturalness for Selecting Teacher Data 

**Title (ZH)**: 将推理精炼到学生大语言模型中：局部自然性选择教师数据 

**Authors**: Hoang Anh Just, Myeongseob Ko, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.03988)  

**Abstract**: Distilling long reasoning traces (10K+ tokens) from stronger teacher models into smaller student LLMs via SFT has emerged as a standard paradigm. This approach is practical and efficient: it leverages the ease of generating abundant reasoning data from stronger models and provides a direct, data-driven way to teach less capable models better reasoning. While previous work has largely focused on prompt selection with responses from a single teacher, the equally important problem of choosing the best response when multiple teacher outputs are available for a single prompt remains underexplored. This challenge becomes important in a multi-teacher setting, where different students may benefit from the outputs of different teachers. This paper fills that gap with a systematic study of response selection for reasoning distillation. We first show that the current method, which picks responses the student assigns the highest global log-probability (global naturalness), fails when responses come from multiple teachers, i.e., global naturalness no longer correlates with downstream performance, especially as the reasoning traces from strong teachers become longer. To overcome this problem, we introduce Local Naturalness, which measures the student's log-probabilities over short, sequential reasoning steps conditioned only on a small local window. Local Naturalness enables two applications: 1) Teacher Selection: Aggregating local scores across prompts reliably identifies the most helpful teacher. 2) Response Selection from a Multiple Teachers: When mixing answers from many teachers, Local Naturalness boosts a 32B student's accuracy on math benchmarks by 9.4pp over global selection, also surpassing the performance achieved by training on data from the single best teacher. These results highlight the power of localized data quality evaluation and data mixing for more effective reasoning distillation. 

**Abstract (ZH)**: 从更强的教师模型向较小的学生语言模型蒸馏长达10K+令牌的推理路径：通过SFT的方法已成为一个标准范式。在多教师设置中选择响应：系统研究推理蒸馏中的响应选择 

---
# What Can You Do When You Have Zero Rewards During RL? 

**Title (ZH)**: 你在强化学习中没有奖励时能做什么？ 

**Authors**: Jatin Prakash, Anirudh Buvanesh  

**Link**: [PDF](https://arxiv.org/pdf/2510.03971)  

**Abstract**: Reinforcement learning (RL) with outcome-based rewards has proven effective for improving large language models (LLMs) on complex reasoning tasks. However, its success often depends on the base model occasionally sampling correct solutions. When no correct solutions are sampled, training encounters a zero-reward barrier where learning stalls due to zero gradients. We study this scenario through the graph search task introduced in Bachmann et al. (2024) and evaluate recent methods that incorporate desirable components such as dense rewards, diversity incentives, and improved credit assignment. Our experiments show that none of these approaches overcome the zero-reward barrier if the base model never produces a correct answer. In contrast, we find that a simple data-centric intervention of adding easier samples to the training set enables the model to eventually solve the original hard task despite starting from zero reward. Importantly, this succeeds without modifying the RL algorithm itself. Because official implementations of several baselines were unavailable, we developed our own, which allowed us to conduct a detailed analysis of their failure modes. We release these implementations to support further research at: this https URL 

**Abstract (ZH)**: 基于结果的强化学习（RL）在提升大规模语言模型（LLMs）在复杂推理任务上的表现方面已被证明是有效的。然而，其成功往往依赖于基础模型偶尔生成正确的解。当没有正确解被采样时，训练会遇到零奖励障碍，导致由于梯度为零而导致学习停滞。我们通过Bachmann等（2024）引入的图搜索任务研究了这一场景，并评估了 recent 方法，这些方法结合了密集奖励、多样性激励和改进的信用分配等有利组件。我们的实验表明，如果基础模型从未生成正确答案，这些方法都无法克服零奖励障碍。相比之下，我们将训练集中的简单数据驱动干预措施——添加更简单的样本——发现这种干预措施使模型能够在从未获得奖励的情况下最终解决原始的难题。重要的是，这种方法无需修改RL算法本身即可成功。由于官方实现的多个基线不可用，我们自行开发了这些实现，并进行了详细的失败模式分析。我们在此发布这些实现以支持进一步的研究：this https URL。 

---
# Towards Carbon-Aware Container Orchestration: Predicting Workload Energy Consumption with Federated Learning 

**Title (ZH)**: 基于碳意识的容器编排：基于联邦学习的负载能耗预测 

**Authors**: Zainab Saad, Jialin Yang, Henry Leung, Steve Drew  

**Link**: [PDF](https://arxiv.org/pdf/2510.03970)  

**Abstract**: The growing reliance on large-scale data centers to run resource-intensive workloads has significantly increased the global carbon footprint, underscoring the need for sustainable computing solutions. While container orchestration platforms like Kubernetes help optimize workload scheduling to reduce carbon emissions, existing methods often depend on centralized machine learning models that raise privacy concerns and struggle to generalize across diverse environments. In this paper, we propose a federated learning approach for energy consumption prediction that preserves data privacy by keeping sensitive operational data within individual enterprises. By extending the Kubernetes Efficient Power Level Exporter (Kepler), our framework trains XGBoost models collaboratively across distributed clients using Flower's FedXgbBagging aggregation using a bagging strategy, eliminating the need for centralized data sharing. Experimental results on the SPECPower benchmark dataset show that our FL-based approach achieves 11.7 percent lower Mean Absolute Error compared to a centralized baseline. This work addresses the unresolved trade-off between data privacy and energy prediction efficiency in prior systems such as Kepler and CASPER and offers enterprises a viable pathway toward sustainable cloud computing without compromising operational privacy. 

**Abstract (ZH)**: 基于联邦学习的能源消耗预测方法：保持数据隐私同时提高能源预测效率 

---
# SPEAR: Soft Prompt Enhanced Anomaly Recognition for Time Series Data 

**Title (ZH)**: SPEAR: 软提示增强的时间序列异常识别 

**Authors**: Hanzhe Wei, Jiajun Wu, Jialin Yang, Henry Leung, Steve Drew  

**Link**: [PDF](https://arxiv.org/pdf/2510.03962)  

**Abstract**: Time series anomaly detection plays a crucial role in a wide range of fields, such as healthcare and internet traffic monitoring. The emergence of large language models (LLMs) offers new opportunities for detecting anomalies in the ubiquitous time series data. Traditional approaches struggle with variable-length time series sequences and context-based anomalies. We propose Soft Prompt Enhanced Anomaly Recognition (SPEAR), a novel approach to leverage LLMs for anomaly detection with soft prompts and quantization. Our methodology involves quantizing and transforming the time series data into input embeddings and combining them with learnable soft prompt embeddings. These combined embeddings are then fed into a frozen LLM. The soft prompts are updated iteratively based on a cross-entropy loss, allowing the model to adapt to time series anomaly detection. The use of soft prompts helps adapt LLMs effectively to time series tasks, while quantization ensures optimal handling of sequences, as LLMs are designed to handle discrete sequences. Our experimental results demonstrate that soft prompts effectively increase LLMs' performance in downstream tasks regarding time series anomaly detection. 

**Abstract (ZH)**: 软提示增强异常识别：利用大规模语言模型进行时间序列异常检测 

---
# Strategy Logic, Imperfect Information, and Hyperproperties 

**Title (ZH)**: 策略逻辑、不完美信息与超属性 

**Authors**: Raven Beutner, Bernd Finkbeiner  

**Link**: [PDF](https://arxiv.org/pdf/2510.03952)  

**Abstract**: Strategy logic (SL) is a powerful temporal logic that enables first-class reasoning over strategic behavior in multi-agent systems (MAS). In many MASs, the agents (and their strategies) cannot observe the global state of the system, leading to many extensions of SL centered around imperfect information, such as strategy logic with imperfect information (SL$_\mathit{ii}$). Along orthogonal lines, researchers have studied the combination of strategic behavior and hyperproperties. Hyperproperties are system properties that relate multiple executions in a system and commonly arise when specifying security policies. Hyper Strategy Logic (HyperSL) is a temporal logic that combines quantification over strategies with the ability to express hyperproperties on the executions of different strategy profiles. In this paper, we study the relation between SL$_\mathit{ii}$ and HyperSL. Our main result is that both logics (restricted to formulas where no state formulas are nested within path formulas) are equivalent in the sense that we can encode SL$_\mathit{ii}$ instances into HyperSL instances and vice versa. For the former direction, we build on the well-known observation that imperfect information is a hyperproperty. For the latter direction, we construct a self-composition of MASs and show how we can simulate hyperproperties using imperfect information. 

**Abstract (ZH)**: SL与HyperSL之间的关系研究：从 imperfect信息扩展到超属性逻辑 

---
# LLM Chemistry Estimation for Multi-LLM Recommendation 

**Title (ZH)**: 多大型语言模型推荐中的LLM化学估计 

**Authors**: Huascar Sanchez, Briland Hitaj  

**Link**: [PDF](https://arxiv.org/pdf/2510.03930)  

**Abstract**: Multi-LLM collaboration promises accurate, robust, and context-aware solutions, yet existing approaches rely on implicit selection and output assessment without analyzing whether collaborating models truly complement or conflict. We introduce LLM Chemistry -- a framework that measures when LLM combinations exhibit synergistic or antagonistic behaviors that shape collective performance beyond individual capabilities. We formalize the notion of chemistry among LLMs, propose algorithms that quantify it by analyzing interaction dependencies, and recommend optimal model ensembles accordingly. Our theoretical analysis shows that chemistry among collaborating LLMs is most evident under heterogeneous model profiles, with its outcome impact shaped by task type, group size, and complexity. Evaluation on classification, summarization, and program repair tasks provides initial evidence for these task-dependent effects, thereby reinforcing our theoretical results. This establishes LLM Chemistry as both a diagnostic factor in multi-LLM systems and a foundation for ensemble recommendation. 

**Abstract (ZH)**: LLM化学：多LLM协作中化学反应的测量与优化 

---
# On the Convergence and Size Transferability of Continuous-depth Graph Neural Networks 

**Title (ZH)**: 连续时深图神经网络的收敛性及规模可转移性 

**Authors**: Mingsong Yan, Charles Kulick, Sui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03923)  

**Abstract**: Continuous-depth graph neural networks, also known as Graph Neural Differential Equations (GNDEs), combine the structural inductive bias of Graph Neural Networks (GNNs) with the continuous-depth architecture of Neural ODEs, offering a scalable and principled framework for modeling dynamics on graphs. In this paper, we present a rigorous convergence analysis of GNDEs with time-varying parameters in the infinite-node limit, providing theoretical insights into their size transferability. To this end, we introduce Graphon Neural Differential Equations (Graphon-NDEs) as the infinite-node limit of GNDEs and establish their well-posedness. Leveraging tools from graphon theory and dynamical systems, we prove the trajectory-wise convergence of GNDE solutions to Graphon-NDE solutions. Moreover, we derive explicit convergence rates under two deterministic graph sampling regimes: (1) weighted graphs sampled from smooth graphons, and (2) unweighted graphs sampled from $\{0,1\}$-valued (discontinuous) graphons. We further establish size transferability bounds, providing theoretical justification for the practical strategy of transferring GNDE models trained on moderate-sized graphs to larger, structurally similar graphs without retraining. Numerical experiments using synthetic and real data support our theoretical findings. 

**Abstract (ZH)**: 连续深度图神经网络，也称为图神经微分方程（GNDEs），将图神经网络（GNNs）的结构归纳偏置与神经ODEs的连续深度架构结合起来，为图上的动力学建模提供了一个可扩展且原理上的框架。本文在无限节点极限下对具有时间变参数的GNDEs进行了严格的收敛性分析，为其规模可扩展性提供了理论洞察。为此，我们引入了图限神经微分方程（Graphon-NDEs）作为GNDEs的无限节点极限，并建立了其适定性。利用图限理论和动力系统工具，我们证明了GNDE解逐轨迹收敛于Graphon-NDE解。此外，我们得出了在两种确定性图采样模式下的显式收敛率：(1) 来自光滑图限的加权图，(2) 来自二进制值（不连续）图限的无权重图。我们进一步建立了规模可扩展性边界，为在保留训练模型性能的情况下将GNDE模型从较小的图转移到更大且结构相似的图提供了理论依据。数值实验使用合成和真实数据支持我们的理论发现。 

---
# Talking Tennis: Language Feedback from 3D Biomechanical Action Recognition 

**Title (ZH)**: Talking Tennis: 3D生物力学动作识别的语言反馈 

**Authors**: Arushi Dashore, Aryan Anumala, Emily Hui, Olivia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03921)  

**Abstract**: Automated tennis stroke analysis has advanced significantly with the integration of biomechanical motion cues alongside deep learning techniques, enhancing stroke classification accuracy and player performance evaluation. Despite these advancements, existing systems often fail to connect biomechanical insights with actionable language feedback that is both accessible and meaningful to players and coaches. This research project addresses this gap by developing a novel framework that extracts key biomechanical features (such as joint angles, limb velocities, and kinetic chain patterns) from motion data using Convolutional Neural Network Long Short-Term Memory (CNN-LSTM)-based models. These features are analyzed for relationships influencing stroke effectiveness and injury risk, forming the basis for feedback generation using large language models (LLMs). Leveraging the THETIS dataset and feature extraction techniques, our approach aims to produce feedback that is technically accurate, biomechanically grounded, and actionable for end-users. The experimental setup evaluates this framework on classification performance and interpretability, bridging the gap between explainable AI and sports biomechanics. 

**Abstract (ZH)**: 自动化网球挥击分析通过结合生物力学运动线索和深度学习技术取得了显著进展，提高了挥击分类准确性并增强了球员表现评估。尽管取得了这些进展，现有系统往往无法将生物力学洞察与既实用又具意义的语言反馈连接起来，供球员和教练使用。本研究项目通过开发一种新颖框架来填补这一空白，该框架利用基于卷积神经网络长短期记忆（CNN-LSTM）的模型从运动数据中提取关键的生物力学特征（如关节角度、肢体速度和动能链模式）。这些特征被分析以确定影响挥击效果和受伤风险的关系，从而为基础利用大型语言模型（LLMs）生成反馈。借助THETIS数据集和特征提取技术，我们的方法旨在为最终用户提供技术准确、生物力学基础且实用的反馈。实验设置评估了该框架在分类性能和可解释性方面的表现，搭建了可解释AI与体育生物力学之间的桥梁。 

---
# Refactoring with LLMs: Bridging Human Expertise and Machine Understanding 

**Title (ZH)**: 使用大语言模型重构代码：连接人类专长与机器理解 

**Authors**: Yonnel Chen Kuang Piao, Jean Carlors Paul, Leuson Da Silva, Arghavan Moradi Dakhel, Mohammad Hamdaqa, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2510.03914)  

**Abstract**: Code refactoring is a fundamental software engineering practice aimed at improving code quality and maintainability. Despite its importance, developers often neglect refactoring due to the significant time, effort, and resources it requires, as well as the lack of immediate functional rewards. Although several automated refactoring tools have been proposed, they remain limited in supporting a broad spectrum of refactoring types. In this study, we explore whether instruction strategies inspired by human best-practice guidelines can enhance the ability of Large Language Models (LLMs) to perform diverse refactoring tasks automatically. Leveraging the instruction-following and code comprehension capabilities of state-of-the-art LLMs (e.g., GPT-mini and DeepSeek-V3), we draw on Martin Fowler's refactoring guidelines to design multiple instruction strategies that encode motivations, procedural steps, and transformation objectives for 61 well-known refactoring types. We evaluate these strategies on benchmark examples and real-world code snippets from GitHub projects. Our results show that instruction designs grounded in Fowler's guidelines enable LLMs to successfully perform all benchmark refactoring types and preserve program semantics in real-world settings, an essential criterion for effective refactoring. Moreover, while descriptive instructions are more interpretable to humans, our results show that rule-based instructions often lead to better performance in specific scenarios. Interestingly, allowing models to focus on the overall goal of refactoring, rather than prescribing a fixed transformation type, can yield even greater improvements in code quality. 

**Abstract (ZH)**: 基于人类最佳实践指导的指令策略能否增强大规模语言模型的自动代码重构能力：以马丁·福勒重构准则为例 

---
# Adversarial Agent Collaboration for C to Rust Translation 

**Title (ZH)**: 面向C到Rust翻译的对抗性代理协作 

**Authors**: Tianyu Li, Ruishi Li, Bo Wang, Brandon Paulsen, Umang Mathur, Prateek Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2510.03879)  

**Abstract**: Translating C to memory-safe languages, like Rust, prevents critical memory safety vulnerabilities that are prevalent in legacy C software. Existing approaches for C to safe Rust translation, including LLM-assisted ones, do not generalize on larger (> 500 LoC) C codebases because they depend on complex program analyses that frequently break. In this work, we present ACToR (Adversarial C To Rust translator), a simple LLM agent-based approach. Inspired by GANs, ACToR pits a generator agent against a discriminator agent, which collaborate to iteratively generate a Rust translation. On each iteration, the translator agent synthesizes and refines a Rust translation to pass an existing suite of tests, and then the discriminator agent finds new failing tests. We demonstrate that ACToR translates all of the 63 real-world command line utilities considered in our benchmarks, which have an average size of 485 lines of code, and it achieves over 90% test pass rate with zero human intervention. To our knowledge, it is the first such system that reliably translates C programs of this scale. Furthermore, ACToR improves translation correctness by up to 18.9% compared to baseline, non-adversarial approaches. 

**Abstract (ZH)**: 将C代码翻译到内存安全的语言如Rust，可以防止遗留在legacy C软件中的关键内存安全漏洞。现有的C到安全Rust翻译方法，包括基于大语言模型的辅助方法，不能很好地应用于超过500行代码的大型代码库，因为它们依赖于经常失效的复杂程序分析。在本工作中，我们提出了ACToR（Adversarial C To Rust translator），一种基于大语言模型代理的简洁方法。受GANs的启发，ACToR将生成器代理与鉴别器代理相抗衡，两者合作迭代生成Rust翻译。在每一轮迭代中，翻译代理合成并精炼Rust翻译以通过现有的测试套件，然后鉴别器代理找到新的失败测试。我们证明ACToR可以翻译我们在基准测试中考虑的全部63个真实世界的命令行实用程序，这些实用程序的平均大小为485行代码，并且在无需人工干预的情况下达到超过90%的测试通过率。据我们所知，这是第一个可靠地翻译如此规模C程序的系统。此外，与基线的非对抗性方法相比，ACToR将翻译的正确性提高多达18.9%。 

---
# Multi-Modal Oral Cancer Detection Using Weighted Ensemble Convolutional Neural Networks 

**Title (ZH)**: 基于加权集成卷积神经网络的多模态口腔癌检测 

**Authors**: Ajo Babu George, Sreehari J R Ajo Babu George, Sreehari J R Ajo Babu George, Sreehari J R  

**Link**: [PDF](https://arxiv.org/pdf/2510.03878)  

**Abstract**: Aims Late diagnosis of Oral Squamous Cell Carcinoma (OSCC) contributes significantly to its high global mortality rate, with over 50\% of cases detected at advanced stages and a 5-year survival rate below 50\% according to WHO statistics. This study aims to improve early detection of OSCC by developing a multimodal deep learning framework that integrates clinical, radiological, and histopathological images using a weighted ensemble of DenseNet-121 convolutional neural networks (CNNs). Material and Methods A retrospective study was conducted using publicly available datasets representing three distinct medical imaging modalities. Each modality-specific dataset was used to train a DenseNet-121 CNN via transfer learning. Augmentation and modality-specific preprocessing were applied to increase robustness. Predictions were fused using a validation-weighted ensemble strategy. Evaluation was performed using accuracy, precision, recall, F1-score. Results High validation accuracy was achieved for radiological (100\%) and histopathological (95.12\%) modalities, with clinical images performing lower (63.10\%) due to visual heterogeneity. The ensemble model demonstrated improved diagnostic robustness with an overall accuracy of 84.58\% on a multimodal validation dataset of 55 samples. Conclusion The multimodal ensemble framework bridges gaps in the current diagnostic workflow by offering a non-invasive, AI-assisted triage tool that enhances early identification of high-risk lesions. It supports clinicians in decision-making, aligning with global oncology guidelines to reduce diagnostic delays and improve patient outcomes. 

**Abstract (ZH)**: 多模态深度学习框架在颌面部鳞状细胞癌早期检测中的应用：一种集成 DenseNet-121 卷积神经网络的加权集成方法 

---
# PoseGaze-AHP: A Knowledge-Based 3D Dataset for AI-Driven Ocular and Postural Diagnosis 

**Title (ZH)**: 基于知识的3D数据集：PoseGaze-AHP，用于AI驱动的眼部和姿势诊断 

**Authors**: Saja Al-Dabet, Sherzod Turaev, Nazar Zaki, Arif O. Khan, Luai Eldweik  

**Link**: [PDF](https://arxiv.org/pdf/2510.03873)  

**Abstract**: Diagnosing ocular-induced abnormal head posture (AHP) requires a comprehensive analysis of both head pose and ocular movements. However, existing datasets focus on these aspects separately, limiting the development of integrated diagnostic approaches and restricting AI-driven advancements in AHP analysis. To address this gap, we introduce PoseGaze-AHP, a novel 3D dataset that synchronously captures head pose and gaze movement information for ocular-induced AHP assessment. Structured clinical data were extracted from medical literature using large language models (LLMs) through an iterative process with the Claude 3.5 Sonnet model, combining stepwise, hierarchical, and complex prompting strategies. The extracted records were systematically imputed and transformed into 3D representations using the Neural Head Avatar (NHA) framework. The dataset includes 7,920 images generated from two head textures, covering a broad spectrum of ocular conditions. The extraction method achieved an overall accuracy of 91.92%, demonstrating its reliability for clinical dataset construction. PoseGaze-AHP is the first publicly available resource tailored for AI-driven ocular-induced AHP diagnosis, supporting the development of accurate and privacy-compliant diagnostic tools. 

**Abstract (ZH)**: 基于头姿和眼动的斜颈综合症诊断需要对头部姿态和眼动进行全面分析。然而，现有数据集分别关注这些方面，限制了综合诊断方法的发展并限制了基于AI的眼动诱导斜颈综合症分析进展。为解决这一问题，我们引入了PoseGaze-AHP，这是一个新颖的3D数据集，可以同步捕捉头部姿态和视线运动信息，用于眼动诱导斜颈综合症评估。通过迭代过程使用大型语言模型（LLMs）结合逐步、分级和复杂提示策略从医学文献中提取结构化临床数据，并使用Neural Head Avatar (NHA)框架系统地补充分数值并转换为3D表示。该数据集包括7,920张来自两种头部纹理生成的图像，涵盖了广泛的眼部条件。提取方法的准确率为91.92%，证明其适用于临床数据集构建。PoseGaze-AHP是首款面向基于AI的眼动诱导斜颈综合症诊断的公开资源，支持开发准确且符合隐私保护的诊断工具。 

---
# Optimal Scaling Needs Optimal Norm 

**Title (ZH)**: 最优缩放需要最优范数 

**Authors**: Oleg Filatov, Jiangtao Wang, Jan Ebert, Stefan Kesselheim  

**Link**: [PDF](https://arxiv.org/pdf/2510.03871)  

**Abstract**: Despite recent progress in optimal hyperparameter transfer under model and dataset scaling, no unifying explanatory principle has been established. Using the Scion optimizer, we discover that joint optimal scaling across model and dataset sizes is governed by a single invariant: the operator norm of the output layer. Across models with up to 1.3B parameters trained on up to 138B tokens, the optimal learning rate/batch size pair $(\eta^{\ast}, B^{\ast})$ consistently has the same operator norm value - a phenomenon we term norm transfer. This constant norm condition is necessary but not sufficient: while for each dataset size, multiple $(\eta, B)$ reach the optimal norm, only a unique $(\eta^{\ast}, B^{\ast})$ achieves the best loss. As a sufficient condition, we provide the first measurement of $(\eta^{\ast}, B^{\ast})$ scaling with dataset size for Scion, and find that the scaling rules are consistent with those of the Adam optimizer. Tuning per-layer-group learning rates also improves model performance, with the output layer being the most sensitive and hidden layers benefiting from lower learning rates. We provide practical insights on norm-guided optimal scaling and release our Distributed Scion (Disco) implementation with logs from over two thousand runs to support research on LLM training dynamics at scale. 

**Abstract (ZH)**: 尽管在模型和数据集规模下的最优超参数传递方面取得了一定进展，但仍缺乏一个统一的解释原则。使用Scion优化器，我们发现模型和数据集规模的联合最优缩放受单一不变量控制：输出层的操作范数。在训练参数量从1.3亿到1380亿、 token数从138亿的数据集上，最优的学习率/批量大小对$(\eta^{\ast}, B^{\ast})$始终具有相同的操作范数值——我们称之为范数传递。该恒定范数条件是必要的但不充分的：尽管对于每个数据集大小，存在多个$(\eta, B)$可以达到最优范数，但只有唯一的$(\eta^{\ast}, B^{\ast})$能实现最佳损失。作为充分条件，我们提供了Scion中$(\eta^{\ast}, B^{\ast})$随数据集规模缩放的第一个测量结果，并发现这些缩放规则与Adam优化器的规则一致。按层组调整学习率也提高了模型性能，输出层最为敏感，隐藏层受益于较低的学习率。我们提供了范数引导下的最优缩放的实用见解，并发布了Distributed Scion（Disco）实现，提供了超过两千次运行的日志，以支持大规模LLM训练动力学的研究。 

---
# AI Adoption Across Mission-Driven Organizations 

**Title (ZH)**: AI采纳在使命驱动组织中的应用 

**Authors**: Dalia Ali, Muneeb Ahmed, Hailan Wang, Arfa Khan, Naira Paola Arnez Jordan, Sunnie S. Y. Kim, Meet Dilip Muchhala, Anne Kathrin Merkle, Orestis Papakyriakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2510.03868)  

**Abstract**: Despite AI's promise for addressing global challenges, empirical understanding of AI adoption in mission-driven organizations (MDOs) remains limited. While research emphasizes individual applications or ethical principles, little is known about how resource-constrained, values-driven organizations navigate AI integration across operations. We conducted thematic analysis of semi-structured interviews with 15 practitioners from environmental, humanitarian, and development organizations across the Global North and South contexts. Our analysis examines how MDOs currently deploy AI, what barriers constrain adoption, and how practitioners envision future integration. MDOs adopt AI selectively, with sophisticated deployment in content creation and data analysis while maintaining human oversight for mission-critical applications. When AI's efficiency benefits conflict with organizational values, decision-making stalls rather than negotiating trade-offs. This study contributes empirical evidence that AI adoption in MDOs should be understood as conditional rather than inevitable, proceeding only where it strengthens organizational sovereignty and mission integrity while preserving human-centered approaches essential to their missions. 

**Abstract (ZH)**: 尽管人工智能在应对全球挑战方面充满 promise，但有关使命驱动组织（MDOs）采用人工智能的实证理解仍有限。尽管研究强调个人应用或伦理原则，但对于资源受限、价值观驱动的组织如何在运营中导航人工智能集成却知之甚少。我们对来自全球北南不同背景下环境、人道主义和开发组织的 15 名从业人员进行了半结构化访谈，并进行了主题分析。分析了 MDOs 目前如何采用人工智能、哪些障碍限制了采用，以及从业人员如何设想未来的集成。MDOs 选择性地采用人工智能，在内容生成和数据分析方面进行了复杂的部署，同时在关键任务应用中保留了人的监督。当人工智能的效率益处与组织价值观发生冲突时，决策停滞不前，而不是权衡取舍。本研究提供了实证证据，表明 MDOs 采用人工智能应被视为有条件的而非不可避免的，仅在增强组织自主权和使命完整性的同时保留对实现其使命至关重要的以人为中心的方法，才能进行。 

---
# Unlocking Reasoning Capabilities in LLMs via Reinforcement Learning Exploration 

**Title (ZH)**: 通过强化学习探索解锁大型语言模型的推理能力 

**Authors**: Wenhao Deng, Long Wei, Chenglei Yu, Tailin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03865)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has recently enhanced the reasoning capabilities of large language models (LLMs), particularly for mathematical problem solving. However, a fundamental limitation remains: as the sampling budget increases, the advantage of RLVR-trained models over their pretrained bases often diminishes or even vanishes, revealing a strong dependence on the base model's restricted search space. We attribute this phenomenon to the widespread use of the reverse Kullback-Leibler (KL) divergence regularizer, whose mode-seeking behavior keeps the policy trapped inside the base model's support region and hampers wider exploration. To address this issue, we propose RAPO (Rewards-Aware Policy Optimization), an algorithm to promote broader yet focused exploration. Our method (i) utilizes the forward KL penalty to replace the reverse KL penalty for out-of-distribution exploration, and (ii) reweights the reference policy to facilitate adaptive in-distribution exploration. We train Qwen2.5-3B and 7B models with RAPO on the 8K SimpleRL-Zero dataset, without supervised fine-tuning, and evaluate them on AIME2024 and AIME2025. Results show that RAPO consistently improves problem-solving performance. Notably, RAPO enables models to surpass the base model's performance ceiling and solves previously intractable problems, advancing the frontier of RLVR for challenging reasoning tasks. 

**Abstract (ZH)**: 基于可验证奖励的强化学习（RLVR） recently enhanced大型语言模型（LLMs）的推理能力，特别是在数学问题求解方面。然而，一个基本的限制仍然存在：随着采样预算的增加，RLVR训练模型相对于其预训练基线的优势往往会减弱甚至消失，显示出对基模型受限搜索空间的强烈依赖。我们将其归因于广泛使用逆Kullback-Leibler（KL）散度正则化器，其模式寻求行为使策略被困在基模型的支持区域内，阻碍了更广泛的探索。为此，我们提出了RAPO（奖励感知策略优化）算法，以促进更广泛但有重点的探索。该方法（i）利用前向KL惩罚来替代逆KL惩罚以促进离分布探索，（ii）重新加权参考策略以促进适应性在分布探索。我们使用RAPO在8K SimpleRL-Zero数据集上训练Qwen2.5-3B和7B模型，无需监督微调，并在AIME2024和AIME2025上进行评估。结果显示，RAPO一致提高了问题求解性能。值得注意的是，RAPO使模型能够超越基模型的表现上限，并解决了此前无法解决的问题，推动了在具有挑战性的推理任务方面RLVR的发展。 

---
# Designing Empirical Studies on LLM-Based Code Generation: Towards a Reference Framework 

**Title (ZH)**: 基于LLM的代码生成实验研究设计：一个参考框架 

**Authors**: Nathalia Nascimento, Everton Guimaraes, Paulo Alencar  

**Link**: [PDF](https://arxiv.org/pdf/2510.03862)  

**Abstract**: The rise of large language models (LLMs) has introduced transformative potential in automated code generation, addressing a wide range of software engineering challenges. However, empirical evaluation of LLM-based code generation lacks standardization, with studies varying widely in goals, tasks, and metrics, which limits comparability and reproducibility. In this paper, we propose a theoretical framework for designing and reporting empirical studies on LLM-based code generation. The framework is grounded in both our prior experience conducting such experiments and a comparative analysis of key similarities and differences among recent studies. It organizes evaluation around core components such as problem sources, quality attributes, and metrics, supporting structured and systematic experimentation. We demonstrate its applicability through representative case mappings and identify opportunities for refinement. Looking forward, we plan to evolve the framework into a more robust and mature tool for standardizing LLM evaluation across software engineering contexts. 

**Abstract (ZH)**: 大型语言模型（LLMs）的兴起为自动化代码生成带来了变革性的潜力，解决了广泛软件工程挑战。然而，基于LLM的代码生成的实证评估缺乏标准化，各研究在目标、任务和指标上差异较大，限制了可比性和可重复性。本文提出了一种理论框架，用于设计和报告基于LLM的代码生成的实证研究。该框架结合了我们以往实验的经验和对近期研究中关键相似性和差异性的比较分析，围绕问题来源、质量属性和指标等核心组件组织评估，支持结构化和系统化的实验。通过典型案例映射证明其适用性，并识别了改进的机会。展望未来，我们将不断完善该框架，使其成为跨软件工程应用场景标准化LLM评估的更 robust 和成熟的工具。 

---
# AI-Assisted Pleural Effusion Volume Estimation from Contrast-Enhanced CT Images 

**Title (ZH)**: 基于对比增强CT图像的AI辅助胸腔积液体积估算 

**Authors**: Sanhita Basu, Tomas Fröding, Ali Teymur Kahraman, Dimitris Toumpanakis, Tobias Sjöblom  

**Link**: [PDF](https://arxiv.org/pdf/2510.03856)  

**Abstract**: Background: Pleural Effusions (PE) is a common finding in many different clinical conditions, but accurately measuring their volume from CT scans is challenging. Purpose: To improve PE segmentation and quantification for enhanced clinical management, we have developed and trained a semi-supervised deep learning framework on contrast-enhanced CT volumes. Materials and Methods: This retrospective study collected CT Pulmonary Angiogram (CTPA) data from internal and external datasets. A subset of 100 cases was manually annotated for model training, while the remaining cases were used for testing and validation. A novel semi-supervised deep learning framework, Teacher-Teaching Assistant-Student (TTAS), was developed and used to enable efficient training in non-segmented examinations. Segmentation performance was compared to that of state-of-the-art models. Results: 100 patients (mean age, 72 years, 28 [standard deviation]; 55 men) were included in the study. The TTAS model demonstrated superior segmentation performance compared to state-of-the-art models, achieving a mean Dice score of 0.82 (95% CI, 0.79 - 0.84) versus 0.73 for nnU-Net (p < 0.0001, Student's T test). Additionally, TTAS exhibited a four-fold lower mean Absolute Volume Difference (AbVD) of 6.49 mL (95% CI, 4.80 - 8.20) compared to nnU-Net's AbVD of 23.16 mL (p < 0.0001). Conclusion: The developed TTAS framework offered superior PE segmentation, aiding accurate volume determination from CT scans. 

**Abstract (ZH)**: 背景：胸腔积液（PE）是多种临床条件下的一种常见发现，但从CT扫描中准确测量其体积颇具挑战性。目的：为改善PE的分割和量化，以增强临床管理，我们开发并训练了一个半监督深度学习框架，应用于对比增强CT体积数据。材料与方法：本回顾性研究从内部和外部数据集中收集了CT肺血管造影（CTPA）数据。100例病例被手动标注用于模型训练，其余病例用于测试和验证。开发并使用了一种新颖的半监督深度学习框架——教师-助教-学生（TTAS）模型，以实现非分割检查中的高效训练。与最先进的模型相比，分割性能进行了比较。结果：研究共包括100名患者（平均年龄72岁，标准差28岁；55名男性）。TTAS模型在分割性能方面优于最先进的模型，平均Dice分数为0.82（95％CI，0.79-0.84），而nnU-Net的平均Dice分数为0.73（t检验，p < 0.0001）。此外，TTAS模型的平均绝对体积差异（AbVD）为6.49 mL（95％CI，4.80-8.20），比nnU-Net的AbVD（23.16 mL）低四倍（p < 0.0001）。结论：开发的TTAS框架提供了更优的PE分割，有助于从CT扫描中准确确定体积。 

---
# A4FN: an Agentic AI Architecture for Autonomous Flying Networks 

**Title (ZH)**: A4FN：自主飞行网络的主体性AI架构 

**Authors**: André Coelho, Pedro Ribeiro, Helder Fontes, Rui Campos  

**Link**: [PDF](https://arxiv.org/pdf/2510.03829)  

**Abstract**: This position paper presents A4FN, an Agentic Artificial Intelligence (AI) architecture for intent-driven automation in Flying Networks (FNs) using Unmanned Aerial Vehicles (UAVs) as access nodes. A4FN leverages Generative AI and Large Language Models (LLMs) to enable real-time, context-aware network control via a distributed agentic system. It comprises two components: the Perception Agent (PA), which semantically interprets multimodal input -- including imagery, audio, and telemetry data -- from UAV-mounted sensors to derive Service Level Specifications (SLSs); and the Decision-and-Action Agent (DAA), which reconfigures the network based on inferred intents. A4FN embodies key properties of Agentic AI, including autonomy, goal-driven reasoning, and continuous perception-action cycles. Designed for mission-critical, infrastructure-limited scenarios such as disaster response, it supports adaptive reconfiguration, dynamic resource management, and interoperability with emerging wireless technologies. The paper details the A4FN architecture, its core innovations, and open research challenges in multi-agent coordination and Agentic AI integration in next-generation FNs. 

**Abstract (ZH)**: 本文展示了A4FN，一种基于代理人工智能（AI）的架构，用于在飞行网络（FNs）中通过无人驾驶航空车辆（UAVs）作为接入节点实现意图驱动的自动化。A4FN 利用生成型AI和大型语言模型（LLMs）通过分布式代理系统实现实时、情境感知的网络控制。它包括两个组成部分：感知代理（PA），其通过解释来自UAV载传感器的多模态输入（包括图像、音频和遥测数据）来推导服务级别规范（SLSs）；以及决策与行动代理（DAA），其根据推断出的意图重新配置网络。A4FN 具备代理人工智能的关键特性，包括自主性、目标驱动的推理以及持续的感知-行动循环。该设计适用于如灾害响应等关键任务、基础设施有限的场景，支持适应性重构、动态资源管理并能够与新兴无线技术协同工作。文章详细介绍了A4FN架构、核心创新以及多代理协调和代理人工智能集成在下一代飞行网络中的开放研究挑战。 

---
# Proximal Diffusion Neural Sampler 

**Title (ZH)**: 邻近扩散神经采样器 

**Authors**: Wei Guo, Jaemoo Choi, Yuchen Zhu, Molei Tao, Yongxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03824)  

**Abstract**: The task of learning a diffusion-based neural sampler for drawing samples from an unnormalized target distribution can be viewed as a stochastic optimal control problem on path measures. However, the training of neural samplers can be challenging when the target distribution is multimodal with significant barriers separating the modes, potentially leading to mode collapse. We propose a framework named \textbf{Proximal Diffusion Neural Sampler (PDNS)} that addresses these challenges by tackling the stochastic optimal control problem via proximal point method on the space of path measures. PDNS decomposes the learning process into a series of simpler subproblems that create a path gradually approaching the desired distribution. This staged procedure traces a progressively refined path to the desired distribution and promotes thorough exploration across modes. For a practical and efficient realization, we instantiate each proximal step with a proximal weighted denoising cross-entropy (WDCE) objective. We demonstrate the effectiveness and robustness of PDNS through extensive experiments on both continuous and discrete sampling tasks, including challenging scenarios in molecular dynamics and statistical physics. 

**Abstract (ZH)**: 基于 proximal 点方法的路径测度最优控制框架：多模态分布下的扩散神经采样器（Proximal Diffusion Neural Sampler, PDNS） 

---
# Detecting Invariant Manifolds in ReLU-Based RNNs 

**Title (ZH)**: 基于ReLU的RNN中不变流形的检测 

**Authors**: Lukas Eisenmann, Alena Brändle, Zahra Monfared, Daniel Durstewitz  

**Link**: [PDF](https://arxiv.org/pdf/2510.03814)  

**Abstract**: Recurrent Neural Networks (RNNs) have found widespread applications in machine learning for time series prediction and dynamical systems reconstruction, and experienced a recent renaissance with improved training algorithms and architectural designs. Understanding why and how trained RNNs produce their behavior is important for scientific and medical applications, and explainable AI more generally. An RNN's dynamical repertoire depends on the topological and geometrical properties of its state space. Stable and unstable manifolds of periodic points play a particularly important role: They dissect a dynamical system's state space into different basins of attraction, and their intersections lead to chaotic dynamics with fractal geometry. Here we introduce a novel algorithm for detecting these manifolds, with a focus on piecewise-linear RNNs (PLRNNs) employing rectified linear units (ReLUs) as their activation function. We demonstrate how the algorithm can be used to trace the boundaries between different basins of attraction, and hence to characterize multistability, a computationally important property. We further show its utility in finding so-called homoclinic points, the intersections between stable and unstable manifolds, and thus establish the existence of chaos in PLRNNs. Finally we show for an empirical example, electrophysiological recordings from a cortical neuron, how insights into the underlying dynamics could be gained through our method. 

**Abstract (ZH)**: 递归神经网络（RNNs）在时间序列预测和动力系统重建中的广泛应用于机器学习中最近经历了一次复兴，这得益于改进的训练算法和网络架构设计。理解训练后的RNNs产生其行为的原因和机制对于科学和医学应用以及更广泛的可解释人工智能至关重要。RNN的动力学范围取决于其状态空间的拓扑和几何特性。周期点的稳定流形和不稳定流形特别重要：它们将动力系统的状态空间分割成不同的吸引子盆地，它们的交点导致具有分形几何的混沌动力学。我们介绍了一种用于检测这些流形的新算法，重点研究使用修正线性单元（ReLU）作为激活函数的规则线性RNN（PLRNN）。我们展示了该算法如何用于追踪不同吸引子盆地之间的边界，从而表征多稳性，这是一个计算上重要的属性。我们还展示了该算法在查找所谓的同宿点（即稳定流形和不稳定流形的交点）方面的效用，从而确立了PLRNN中混沌的存在性。最后，我们通过一个实证例子——皮层神经元的电生理记录，展示了如何通过我们的方法获得底层动力学的洞察。 

---
# Diverse Text-to-Image Generation via Contrastive Noise Optimization 

**Title (ZH)**: 通过对比噪声优化实现多样的文本到图像生成 

**Authors**: Byungjun Kim, Soobin Um, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.03813)  

**Abstract**: Text-to-image (T2I) diffusion models have demonstrated impressive performance in generating high-fidelity images, largely enabled by text-guided inference. However, this advantage often comes with a critical drawback: limited diversity, as outputs tend to collapse into similar modes under strong text guidance. Existing approaches typically optimize intermediate latents or text conditions during inference, but these methods deliver only modest gains or remain sensitive to hyperparameter tuning. In this work, we introduce Contrastive Noise Optimization, a simple yet effective method that addresses the diversity issue from a distinct perspective. Unlike prior techniques that adapt intermediate latents, our approach shapes the initial noise to promote diverse outputs. Specifically, we develop a contrastive loss defined in the Tweedie data space and optimize a batch of noise latents. Our contrastive optimization repels instances within the batch to maximize diversity while keeping them anchored to a reference sample to preserve fidelity. We further provide theoretical insights into the mechanism of this preprocessing to substantiate its effectiveness. Extensive experiments across multiple T2I backbones demonstrate that our approach achieves a superior quality-diversity Pareto frontier while remaining robust to hyperparameter choices. 

**Abstract (ZH)**: 基于文本到图像的发散模型：通过对比噪声优化提升多样性和保真度 

---
# ReTiDe: Real-Time Denoising for Energy-Efficient Motion Picture Processing with FPGAs 

**Title (ZH)**: ReTiDe: 适用于FPGA的高效实时噪声消除运动图像处理 

**Authors**: Changhong Li, Clément Bled, Rosa Fernandez, Shreejith Shanker  

**Link**: [PDF](https://arxiv.org/pdf/2510.03812)  

**Abstract**: Denoising is a core operation in modern video pipelines. In codecs, in-loop filters suppress sensor noise and quantisation artefacts to improve rate-distortion performance; in cinema post-production, denoisers are used for restoration, grain management, and plate clean-up. However, state-of-the-art deep denoisers are computationally intensive and, at scale, are typically deployed on GPUs, incurring high power and cost for real-time, high-resolution streams. This paper presents Real-Time Denoise (ReTiDe), a hardware-accelerated denoising system that serves inference on data-centre Field Programmable Gate Arrays (FPGAs). A compact convolutional model is quantised (post-training quantisation plus quantisation-aware fine-tuning) to INT8 and compiled for AMD Deep Learning Processor Unit (DPU)-based FPGAs. A client-server integration offloads computation from the host CPU/GPU to a networked FPGA service, while remaining callable from existing workflows, e.g., NUKE, without disrupting artist tooling. On representative benchmarks, ReTiDe delivers 37.71$\times$ Giga Operations Per Second (GOPS) throughput and 5.29$\times$ higher energy efficiency than prior FPGA denoising accelerators, with negligible degradation in Peak Signal-to-Noise Ratio (PSNR)/Structural Similarity Index (SSIM). These results indicate that specialised accelerators can provide practical, scalable denoising for both encoding pipelines and post-production, reducing energy per frame without sacrificing quality or workflow compatibility. Code is available at this https URL. 

**Abstract (ZH)**: 实时去噪（ReTiDe）：一种加速的数据中心可编程门阵列去噪系统 

---
# 6G-Enabled Digital Twin Framework for Real-Time Cyber-Physical Systems: An Experimental Validation with Industrial Bearing Fault Detection 

**Title (ZH)**: 6G赋能的数字孪生框架在实时物理- cyber系统中的实验验证：以工业轴承故障检测为例 

**Authors**: Vaskar Chakma, Wooyeol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03807)  

**Abstract**: Current Cyber-Physical Systems (CPS) integrated with Digital Twin (DT) technology face critical limitations in achieving real-time performance for mission-critical industrial applications. Existing 5G-enabled systems suffer from latencies exceeding 10ms, which are inadequate for applications requiring sub-millisecond response times, such as autonomous industrial control and predictive maintenance. This research aims to develop and validate a 6G-enabled Digital Twin framework that achieves ultra-low latency communication and real-time synchronization between physical industrial assets and their digital counterparts, specifically targeting bearing fault detection as a critical industrial use case. The proposed framework integrates terahertz communications (0.1-1 THz), intelligent reflecting surfaces, and edge artificial intelligence within a five-layer architecture. Experimental validation was conducted using the Case Western Reserve University (CWRU) bearing dataset, implementing comprehensive feature extraction (15 time and frequency domain features) and Random Forest classification algorithms. The system performance was evaluated against traditional WiFi-6 and 5G networks across multiple metrics, including classification accuracy, end-to-end latency, and scalability. It achieved 97.7% fault classification accuracy with 0.8ms end-to-end latency, representing a 15.6x improvement over WiFi-6 (12.5ms) and 5.25x improvement over 5G (4.2ms) networks. The system demonstrated superior scalability with sub-linear processing time growth and maintained consistent performance across four bearing fault categories (normal, inner race, outer race, and ball faults) with macro-averaged F1-scores exceeding 97%. 

**Abstract (ZH)**: 现有的数字孪生技术集成的-current-cyber-physical系统（CPS）在实现关键工业应用的实时性能方面面临重要限制。现有的5G使能系统面临的延迟超过10ms，对于需要亚毫秒级响应时间的自主工业控制和预测性维护等应用而言是不充分的。本研究旨在开发并验证一种6G使能的数字孪生框架，实现超低延迟通信和物理工业资产与其数字对应物之间的实时同步，特别针对轴承故障检测这一关键工业应用场景。所提出的框架在五层架构中集成了太赫兹通信（0.1-1 THz）、智能反射表面和边缘人工智能技术。通过使用辛辛那提大学（CWRU）轴承数据集进行了实验验证，实现了全面的特征提取（15个时间域和频域特征）和随机森林分类算法。系统性能在多项指标上（包括分类准确性、端到端延迟和可扩展性）与传统WiFi-6和5G网络进行了对比评估。系统在端到端延迟为0.8ms的情况下实现了97.7%的故障分类准确率，分别比WiFi-6（12.5ms）和5G（4.2ms）网络的性能提高了15.6倍和5.25倍。该系统展示了优越的可扩展性，处理时间呈次线性增长，并在四种轴承故障类别（正常、内圈、外圈和滚珠故障）上保持了一致性，宏均F1分数超过97%。 

---
# Beyond Token Length: Step Pruner for Efficient and Accurate Reasoning in Large Language Models 

**Title (ZH)**: 超越令牌长度：高效且准确的大型语言模型推理步长剪枝方法 

**Authors**: Canhui Wu, Qiong Cao, Chang Li, Zhenfang Wang, Chao Xue, Yuwei Fan, Wei Xi, Xiaodong He  

**Link**: [PDF](https://arxiv.org/pdf/2510.03805)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate strong performance on complex tasks but often suffer from excessive verbosity, known as "overthinking." Existing solutions via reinforcement learning (RL) typically penalize generated tokens to promote conciseness. However, these methods encounter two challenges: responses with fewer tokens do not always correspond to fewer reasoning steps, and models may develop hacking behavior in later stages of training by discarding reasoning steps to minimize token usage. In this work, we introduce \textbf{Step Pruner (SP)}, an RL framework that steers LRMs toward more efficient reasoning by favoring compact reasoning steps. Our step-aware reward function prioritizes correctness while imposing penalties for redundant steps, and withholds rewards for incorrect responses to prevent the reinforcement of erroneous reasoning. Moreover, we propose a dynamic stopping mechanism: when the length of any output step exceeds the upper limit, we halt updates to prevent hacking behavior caused by merging steps. Extensive experiments across four reasoning benchmarks demonstrate that SP achieves state-of-the-art accuracy while significantly reducing response length. For instance, on AIME24, SP reduces token usage by \textbf{69.7\%}. 

**Abstract (ZH)**: 大型推理模型中的步修剪（Step Pruner, SP）：一种促进更高效推理的RL框架 

---
# Mechanistic Interpretability of Socio-Political Frames in Language Models 

**Title (ZH)**: 社会政治框架在语言模型中的机理可解释性 

**Authors**: Hadi Asghari, Sami Nenno  

**Link**: [PDF](https://arxiv.org/pdf/2510.03799)  

**Abstract**: This paper explores the ability of large language models to generate and recognize deep cognitive frames, particularly in socio-political contexts. We demonstrate that LLMs are highly fluent in generating texts that evoke specific frames and can recognize these frames in zero-shot settings. Inspired by mechanistic interpretability research, we investigate the location of the `strict father' and `nurturing parent' frames within the model's hidden representation, identifying singular dimensions that correlate strongly with their presence. Our findings contribute to understanding how LLMs capture and express meaningful human concepts. 

**Abstract (ZH)**: 本论文探讨了大型语言模型生成和识别深层认知框架的能力，特别是在社会政治情境中的表现。我们证明了大型语言模型在生成唤起特定框架的文本方面极为流畅，并能在零样本设置中识别这些框架。受机械可解释性研究的启发，我们探讨了“严格父亲”和“养育父母”框架在模型隐藏表示中的位置，确定了与它们存在高度相关的单个维度。我们的研究结果有助于理解大型语言模型如何捕捉和表达有意义的人类概念。 

---
# Lightweight and Data-Efficient MultivariateTime Series Forecasting using Residual-Stacked Gaussian (RS-GLinear) Architecture 

**Title (ZH)**: 使用残差堆叠高斯(RS-GLinear)架构的轻量级和数据-efficient多变量时间序列预测 

**Authors**: Abukar Ali  

**Link**: [PDF](https://arxiv.org/pdf/2510.03788)  

**Abstract**: Following the success of Transformer architectures in language modeling, particularly their ability to capture long-range dependencies, researchers have explored how these architectures can be adapted for time-series forecasting. Transformer-based models have been proposed to handle both short- and long-term dependencies when predicting future values from historical data. However, studies such as those by Zeng et al. (2022) and Rizvi et al. (2025) have reported mixed results in long-term forecasting tasks. In this work, we evaluate the Gaussian-based Linear architecture introduced by Rizvi et al. (2025) and present an enhanced version called the Residual Stacked Gaussian Linear (RSGL) model. We also investigate the broader applicability of the RSGL model in additional domains, including financial time series and epidemiological data. Experimental results show that the RSGL model achieves improved prediction accuracy and robustness compared to both the baseline Gaussian Linear and Transformer-based models. 

**Abstract (ZH)**: 基于Transformer架构在语言建模中的成功，特别是其捕捉长范围依赖的能力，研究人员探索了这些架构如何适应时间序列预测。基于Transformer的模型被提出用于从历史数据预测未来值时同时处理短期和长期依赖。然而，诸如Zeng等（2022）和Rizvi等（2025）的研究在长期预测任务中报道了混合结果。在本研究中，我们评估了Rizvi等（2025）引入的基于高斯的线性架构，并呈现其增强版本，即残差堆叠高斯线性（RSGL）模型。我们还调查了RSGL模型在其他领域的更广泛适用性，包括金融时间序列和流行病学数据。实验结果表明，RSGL模型比基线高斯线性和Transformer基模型在预测准确性和稳健性方面都取得了改进。 

---
# Rezwan: Leveraging Large Language Models for Comprehensive Hadith Text Processing: A 1.2M Corpus Development 

**Title (ZH)**: Reswan: 利用大型语言模型进行全面的汗酉特文处理：一个120万语料库的发展 

**Authors**: Majid Asgari-Bidhendi, Muhammad Amin Ghaseminia, Alireza Shahbazi, Sayyed Ali Hossayni, Najmeh Torabian, Behrouz Minaei-Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2510.03781)  

**Abstract**: This paper presents the development of Rezwan, a large-scale AI-assisted Hadith corpus comprising over 1.2M narrations, extracted and structured through a fully automated pipeline. Building on digital repositories such as Maktabat Ahl al-Bayt, the pipeline employs Large Language Models (LLMs) for segmentation, chain--text separation, validation, and multi-layer enrichment. Each narration is enhanced with machine translation into twelve languages, intelligent diacritization, abstractive summarization, thematic tagging, and cross-text semantic analysis. This multi-step process transforms raw text into a richly annotated research-ready infrastructure for digital humanities and Islamic studies. A rigorous evaluation was conducted on 1,213 randomly sampled narrations, assessed by six domain experts. Results show near-human accuracy in structured tasks such as chain--text separation (9.33/10) and summarization (9.33/10), while highlighting ongoing challenges in diacritization and semantic similarity detection. Comparative analysis against the manually curated Noor Corpus demonstrates the superiority of Najm in both scale and quality, with a mean overall score of 8.46/10 versus 3.66/10. Furthermore, cost analysis confirms the economic feasibility of the AI approach: tasks requiring over 229,000 hours of expert labor were completed within months at a fraction of the cost. The work introduces a new paradigm in religious text processing by showing how AI can augment human expertise, enabling large-scale, multilingual, and semantically enriched access to Islamic heritage. 

**Abstract (ZH)**: 本文介绍了Rezwan，一个包含超过120万份传述的文字大规模AI辅助哈迪斯语料库，通过完全自动化的管道提取和结构化。该管道基于如Alhail阿迈尔图书馆等数字存储库，利用大型语言模型（LLMs）进行断句、链文分离、验证和多层增强。每份传述都增强了十二种语言的机器翻译、智能标点、摘要、主题标记和跨文本语义分析。这一多步骤过程将原始文本转换为面向数字人文和伊斯兰研究的研究丰富注释基础设施。对1213份随机选取的传述进行了严格评估，由六名领域专家评估。结果显示，在链文分离（9.33/10）和摘要（9.33/10）等结构化任务上具有近人类准确度，同时指出了标点和语义相似性检测方面的持续挑战。与手动整理的Noor语料库的对比分析显示Najm在规模和质量上都优于后者，平均总体得分为8.46/10，而Noor为3.66/10。此外，成本分析证实了AI方法的经济可行性：需要超过229,000小时专家劳动的任务用几个月且成本仅为一小部分完成。本文通过显示AI如何增强人类专业知识，引入了宗教文本处理的新范式，使伊斯兰遗产的大规模、多语言和语义化访问成为可能。 

---
# Adaptively Sampling-Reusing-Mixing Decomposed Gradients to Speed Up Sharpness Aware Minimization 

**Title (ZH)**: 自适应采样-重用-混合分解梯度以加速锋利感知最小化 

**Authors**: Jiaxin Deng, Junbiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03763)  

**Abstract**: Sharpness-Aware Minimization (SAM) improves model generalization but doubles the computational cost of Stochastic Gradient Descent (SGD) by requiring twice the gradient calculations per optimization step. To mitigate this, we propose Adaptively sampling-Reusing-mixing decomposed gradients to significantly accelerate SAM (ARSAM). Concretely, we firstly discover that SAM's gradient can be decomposed into the SGD gradient and the Projection of the Second-order gradient onto the First-order gradient (PSF). Furthermore, we observe that the SGD gradient and PSF dynamically evolve during training, emphasizing the growing role of the PSF to achieve a flat minima. Therefore, ARSAM is proposed to the reused PSF and the timely updated PSF still maintain the model's generalization ability. Extensive experiments show that ARSAM achieves state-of-the-art accuracies comparable to SAM across diverse network architectures. On CIFAR-10/100, ARSAM is comparable to SAM while providing a speedup of about 40\%. Moreover, ARSAM accelerates optimization for the various challenge tasks (\textit{e.g.}, human pose estimation, and model quantization) without sacrificing performance, demonstrating its broad practicality.% The code is publicly accessible at: this https URL. 

**Abstract (ZH)**: 自适应采样-重用-混合分解梯度以显著加速SAM（ARSAM） 

---
# You Have Been LaTeXpOsEd: A Systematic Analysis of Information Leakage in Preprint Archives Using Large Language Models 

**Title (ZH)**: You Have Been LaTeXpOsEd: 大语言模型在预印本档案中信息泄露的系统分析 

**Authors**: Richard A. Dubniczky, Bertalan Borsos, Tihanyi Norbert  

**Link**: [PDF](https://arxiv.org/pdf/2510.03761)  

**Abstract**: The widespread use of preprint repositories such as arXiv has accelerated the communication of scientific results but also introduced overlooked security risks. Beyond PDFs, these platforms provide unrestricted access to original source materials, including LaTeX sources, auxiliary code, figures, and embedded comments. In the absence of sanitization, submissions may disclose sensitive information that adversaries can harvest using open-source intelligence. In this work, we present the first large-scale security audit of preprint archives, analyzing more than 1.2 TB of source data from 100,000 arXiv submissions. We introduce LaTeXpOsEd, a four-stage framework that integrates pattern matching, logical filtering, traditional harvesting techniques, and large language models (LLMs) to uncover hidden disclosures within non-referenced files and LaTeX comments. To evaluate LLMs' secret-detection capabilities, we introduce LLMSec-DB, a benchmark on which we tested 25 state-of-the-art models. Our analysis uncovered thousands of PII leaks, GPS-tagged EXIF files, publicly available Google Drive and Dropbox folders, editable private SharePoint links, exposed GitHub and Google credentials, and cloud API keys. We also uncovered confidential author communications, internal disagreements, and conference submission credentials, exposing information that poses serious reputational risks to both researchers and institutions. We urge the research community and repository operators to take immediate action to close these hidden security gaps. To support open science, we release all scripts and methods from this study but withhold sensitive findings that could be misused, in line with ethical principles. The source code and related material are available at the project website this https URL 

**Abstract (ZH)**: 预印本仓储（如arXiv）的广泛应用加速了科学研究成果的传播，但也引入了未被忽视的安全风险。除了PDF文件外，这些平台提供了对原始源材料的无限制访问，包括LaTeX源代码、辅助代码、图表和嵌入的注释。在缺乏净化处理的情况下，提交的内容可能会泄露敏感信息，而对手可以利用开源情报手段收集这些信息。在本文中，我们提出了首个大规模预印本档案安全审计，分析了超过1.2 TB的源数据，来自100,000篇arXiv提交。我们引入了LaTeXpOsEd，这是一种四阶段框架，将模式匹配、逻辑过滤、传统的采集技术以及大型语言模型（LLMs）集成在一起，以揭露非参考文件和LaTeX注释中隐藏的披露信息。为了评估LLMs的秘密检测能力，我们引入了LLMSec-DB基准测试，并测试了25个最先进的模型。我们的分析发现了数千个PII泄露、带有GPS标签的EXIF文件、公开的Google Drive和Dropbox文件夹、可编辑的私有SharePoint链接、暴露的GitHub和Google凭证，以及云API密钥。我们还发现了机密作者通讯、内部分歧以及会议提交凭证，这些信息对研究人员和机构都构成了严重的声誉风险。我们呼吁研究界和仓储运营商立即采取行动，以关闭这些隐藏的安全漏洞。为了支持开放科学，我们发布了本研究的所有脚本和方法，但保留了可能被滥用的敏感发现，符合伦理原则。相关源代码和材料可在项目网站this https URL 处获得。 

---
# EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models 

**Title (ZH)**: EvoEngineer: 使用大规模语言模型掌握自动化CUDA内核代码进化 

**Authors**: Ping Guo, Chenyu Zhu, Siyuan Chen, Fei Liu, Xi Lin, Zhichao Lu, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03760)  

**Abstract**: CUDA kernel optimization has become a critical bottleneck for AI performance, as deep learning training and inference efficiency directly depends on highly optimized GPU kernels.
Despite the promise of Large Language Models (LLMs) for automating kernel optimization, this field suffers from a fragmented ecosystem of isolated and incomparable approaches with unclear problem formulations.
Furthermore, general-purpose LLM code evolution methods cannot meet strict correctness requirements of CUDA kernel optimization.
We address these fundamental challenges by first formalizing CUDA kernel optimization as a code optimization task with a clear objective, constraints, and evaluation metrics.
We then establish the first systematic LLM-based code evolution framework, EvoEngineer, that provides guidance for designing and adapting optimization strategies to achieve a balance between performance and correctness.
Finally, we implement a kernel optimization system based on this framework and conduct extensive experiments on 91 real-world CUDA kernels.
Our results demonstrate that EvoEngineer achieves a principled balance between performance and correctness, with the highest averaged median speedup of \textbf{2.72}$\times$ over baseline CUDA kernels and a code validity rate of \textbf{69.8}\%, outperforming existing methods on both dimensions.
Our method achieves a maximum speedup of \textbf{36.75}$\times$ among all operations over PyTorch kernels and delivers the highest speedup on \textbf{28} (\textbf{56.0\%}) of 50 operations that achieve over \textbf{2$\times$} acceleration. 

**Abstract (ZH)**: CUDA内核优化已成为影响AI性能的关键瓶颈，因为深度学习训练和推理效率直接取决于高度优化的GPU内核。
尽管大规模语言模型（LLMs）有望自动化内核优化，但该领域面临一个碎片化的生态系统，其中包含了许多孤立且不可比的方法，并且问题定义不明确。
此外，通用的LLM代码演变方法无法满足CUDA内核优化严格正确性要求。
我们通过首先将CUDA内核优化形式化为一个具有明确目标、约束和评估指标的代码优化任务，来解决这些根本挑战。
我们还建立了第一个系统性的基于LLM的代码演化框架EvoEngineer，该框架为设计和适应优化策略提供了指导，以实现性能和正确性的平衡。
最后，我们基于该框架实现了一个内核优化系统，并对91个实际CUDA内核进行了广泛的实验。
实验结果表明，EvoEngineer在性能和正确性之间实现了有根据的平衡，与基线CUDA内核相比，平均中位数加速倍数为\textbf{2.72}$\times$，代码有效率为\textbf{69.8}\%，在两个维度上均优于现有方法。
我们的方法在所有操作中对PyTorch内核的最大加速倍数达到了\textbf{36.75}$\times$，并在\textbf{28}（\textbf{56.0\%}）的\textbf{50}个操作中实现了超过\textbf{2$\times$}的加速。 

---
# Code4MeV2: a Research-oriented Code-completion Platform 

**Title (ZH)**: Code4MeV2：一个面向研究的代码补全平台 

**Authors**: Roham Koohestani, Parham Bateni, Aydin Ebrahimi, Behdad Etezadi, Kiarash Karimi, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03755)  

**Abstract**: The adoption of AI-powered code completion tools in software development has increased substantially, yet the user interaction data produced by these systems remain proprietary within large corporations. This creates a barrier for the academic community, as researchers must often develop dedicated platforms to conduct studies on human--AI interaction, making reproducible research and large-scale data analysis impractical. In this work, we introduce Code4MeV2, a research-oriented, open-source code completion plugin for JetBrains IDEs, as a solution to this limitation. Code4MeV2 is designed using a client--server architecture and features inline code completion and a context-aware chat assistant. Its core contribution is a modular and transparent data collection framework that gives researchers fine-grained control over telemetry and context gathering. Code4MeV2 achieves industry-comparable performance in terms of code completion, with an average latency of 200~ms. We assess our tool through a combination of an expert evaluation and a user study with eight participants. Feedback from both researchers and daily users highlights its informativeness and usefulness. We invite the community to adopt and contribute to this tool. More information about the tool can be found at this https URL. 

**Abstract (ZH)**: 基于AI的代码完成工具在软件开发中的应用日益增多，但这些系统产生的用户交互数据仍保留在大型企业内部。这为学术界造成了一定障碍，研究人员往往需要开发专门的平台来研究人类-AI交互，这使得可重复研究和大规模数据分析变得 impractical。本文介绍了一个面向研究、开源的代码完成插件 Code4MeV2，作为解决这一限制的解决方案。Code4MeV2 采用客户端-服务器架构，支持嵌入式代码完成和上下文感知聊天助理。其核心贡献在于一个模块化且透明的数据收集框架，赋予研究人员对遥测和上下文收集的精细控制。Code4MeV2 在代码完成性能方面达到行业水平，平均延迟为 200~ms。我们通过专家评估和八名参与者的用户研究对其进行了评估。研究者和普通用户反馈表明其信息量大且实用。我们邀请社区采用并贡献于此工具。更多关于该工具的信息请参见：this https URL。 

---
# TreePrompt: Leveraging Hierarchical Few-Shot Example Selection for Improved English-Persian and English-German Translation 

**Title (ZH)**: 树提示：利用分层少量示例选择以改进英语-波斯语和英语-德语翻译 

**Authors**: Ramtin Kakavand, Ebrahim Ansari  

**Link**: [PDF](https://arxiv.org/pdf/2510.03748)  

**Abstract**: Large Language Models (LLMs) have consistently demonstrated strong performance in machine translation, especially when guided by high-quality prompts. Few-shot prompting is an effective technique to improve translation quality; however, most existing example selection methods focus solely on query-to-example similarity and do not account for the quality of the examples. In this work, we propose TreePrompt, a novel example selection approach that learns LLM preferences to identify high-quality, contextually relevant examples within a tree-structured framework. To further explore the balance between similarity and quality, we combine TreePrompt with K-Nearest Neighbors (K-NN) and Adaptive Few-Shot Prompting (AFSP). Evaluations on two language pairs - English-Persian (MIZAN) and English-German (WMT19) - show that integrating TreePrompt with AFSP or Random selection leads to improved translation performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在机器翻译中表现出强大的性能，尤其是在高质量提示的指导下。少量提示是一种有效的技术来提高翻译质量；然而，现有的大多数示例选择方法仅关注查询与示例之间的相似性，而不考虑示例的质量。在本工作中，我们提出TreePrompt，这是一种新颖的示例选择方法，在树结构框架中学习LLM的偏好，以识别高质量的相关示例。为了进一步探索相似性和质量之间的平衡，我们将TreePrompt与K-最近邻（K-NN）和自适应少量提示（AFSP）相结合。在英语-波斯语（MIZAN）和英语-德语（WMT19）两种语言对上的评估表明，将TreePrompt与AFSP或随机选择结合使用可以提高翻译性能。 

---
# HydroFusion-LMF: Semi-Supervised Multi-Network Fusion with Large-Model Adaptation for Long-Term Daily Runoff Forecasting 

**Title (ZH)**: HydroFusion-LMF：大规模模型适应的半监督多网络融合长周期日径流预报 

**Authors**: Qianfei Fan, Jiayu Wei, Peijun Zhu, Wensheng Ye, Meie Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03744)  

**Abstract**: Accurate decade-scale daily runoff forecasting in small watersheds is difficult because signals blend drifting trends, multi-scale seasonal cycles, regime shifts, and sparse extremes. Prior deep models (DLinear, TimesNet, PatchTST, TiDE, Nonstationary Transformer, LSTNet, LSTM) usually target single facets and under-utilize unlabeled spans, limiting regime adaptivity. We propose HydroFusion-LMF, a unified framework that (i) performs a learnable trend-seasonal-residual decomposition to reduce non-stationarity, (ii) routes residuals through a compact heterogeneous expert set (linear refinement, frequency kernel, patch Transformer, recurrent memory, dynamically normalized attention), (iii) fuses expert outputs via a hydrologic context-aware gate conditioned on day-of-year phase, antecedent precipitation, local variance, flood indicators, and static basin attributes, and (iv) augments supervision with a semi-supervised multi-task objective (composite MSE/MAE + extreme emphasis + NSE/KGE, masked reconstruction, multi-scale contrastive alignment, augmentation consistency, variance-filtered pseudo-labeling). Optional adapter / LoRA layers inject a frozen foundation time-series encoder efficiently. On a ~10-year daily dataset HydroFusion-LMF attains MSE 1.0128 / MAE 0.5818, improving the strongest baseline (DLinear) by 10.2% / 10.3% and the mean baseline by 24.6% / 17.1%. We observe simultaneous MSE and MAE reductions relative to baselines. The framework balances interpretability (explicit components, sparse gating) with performance, advancing label-efficient hydrologic forecasting under non-stationarity. 

**Abstract (ZH)**: 准确的小流域十年尺度日径流预测因信号混杂漂移趋势、多尺度季节周期、系统转换和稀疏极端事件而具有挑战性。先前的深度模型通常专注于单一特征，未能充分利用无标签数据，限制了系统适应性。我们提出了一种统一框架HydroFusion-LMF，该框架通过（i）进行可学习的趋势-季节-残差分解以降低非平稳性，（ii）通过紧凑的异构专家集合（线性细化、频率核、补丁Transformer、递归记忆、动态规范化注意力）路由残差，（iii）通过一种水文学上下文感知门控融合专家输出，该门控根据日年内相位、前期降水、局部变异、洪水指标以及静态流域属性进行条件处理，和（iv）通过半监督多任务目标增强监督（综合MSE/MAE、极端事件强调、NSE/KGE、掩码重建、多尺度对比对齐、增强一致性、方差筛选伪标签）来平衡可解释性和性能，从而在非平稳条件下促进标签高效水文预报。 

---
# Cost Efficient Fairness Audit Under Partial Feedback 

**Title (ZH)**: 部分反馈下的成本高效公平性审计 

**Authors**: Nirjhar Das, Mohit Sharma, Praharsh Nanavati, Kirankumar Shiragur, Amit Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2510.03734)  

**Abstract**: We study the problem of auditing the fairness of a given classifier under partial feedback, where true labels are available only for positively classified individuals, (e.g., loan repayment outcomes are observed only for approved applicants). We introduce a novel cost model for acquiring additional labeled data, designed to more accurately reflect real-world costs such as credit assessment, loan processing, and potential defaults. Our goal is to find optimal fairness audit algorithms that are more cost-effective than random exploration and natural baselines.
In our work, we consider two audit settings: a black-box model with no assumptions on the data distribution, and a mixture model, where features and true labels follow a mixture of exponential family distributions. In the black-box setting, we propose a near-optimal auditing algorithm under mild assumptions and show that a natural baseline can be strictly suboptimal. In the mixture model setting, we design a novel algorithm that achieves significantly lower audit cost than the black-box case. Our approach leverages prior work on learning from truncated samples and maximum-a-posteriori oracles, and extends known results on spherical Gaussian mixtures to handle exponential family mixtures, which may be of independent interest. Moreover, our algorithms apply to popular fairness metrics including demographic parity, equal opportunity, and equalized odds. Empirically, we demonstrate strong performance of our algorithms on real-world fair classification datasets like Adult Income and Law School, consistently outperforming natural baselines by around 50% in terms of audit cost. 

**Abstract (ZH)**: 我们在部分反馈情况下审查给定分类器的公平性问题研究：仅对正分类个体提供真实标签（例如，仅对获批申请者观察到贷款偿还结果）。我们引入了一种新的成本模型，用于获取额外的标记数据，旨在更准确地反映现实世界的成本，如信用评估、贷款处理和潜在违约。我们的目标是找到比随机探索和自然基线更具成本效益的最优公平性审查算法。

在我们的工作中，我们考虑了两种审查设置：一个不假设数据分布的黑盒模型，以及一个混合模型，其中特征和真实标签遵循指数家族分布的混合。在黑盒设置下，我们提出了一种在轻微假设下的近最优审查算法，并证明了一种自然基线可能是严格次优的。在混合模型设置下，我们设计了一种新的算法，其审查成本明显低于黑盒情况。我们的方法利用了从截断样本学习和最大后验先验或acles的相关工作，并将已知的球形高斯混合的结果扩展到处理指数家族混合，这可能具有独立的兴趣。此外，我们的算法适用于人口统计学平价、相同机会和平价机会等流行公平性指标。在实验中，我们展示了我们的算法在现实世界的公平分类数据集如Adult Income和Law School中的强大性能，在审查成本上始终优于自然基线约50%。 

---
# Artery-Vein Segmentation from Fundus Images using Deep Learning 

**Title (ZH)**: 基金us图像中的动脉-静脉分割方法研究 

**Authors**: Sharan SK, Subin Sahayam, Umarani Jayaraman, Lakshmi Priya A  

**Link**: [PDF](https://arxiv.org/pdf/2510.03717)  

**Abstract**: Segmenting of clinically important retinal blood vessels into arteries and veins is a prerequisite for retinal vessel analysis. Such analysis can provide potential insights and bio-markers for identifying and diagnosing various retinal eye diseases. Alteration in the regularity and width of the retinal blood vessels can act as an indicator of the health of the vasculature system all over the body. It can help identify patients at high risk of developing vasculature diseases like stroke and myocardial infarction. Over the years, various Deep Learning architectures have been proposed to perform retinal vessel segmentation. Recently, attention mechanisms have been increasingly used in image segmentation tasks. The work proposes a new Deep Learning approach for artery-vein segmentation. The new approach is based on the Attention mechanism that is incorporated into the WNet Deep Learning model, and we call the model as Attention-WNet. The proposed approach has been tested on publicly available datasets such as HRF and DRIVE datasets. The proposed approach has outperformed other state-of-art models available in the literature. 

**Abstract (ZH)**: 临床重要视网膜血管的分割是视网膜血管分析的前提。such分析可以提供识别和诊断各种视网膜眼病的潜在洞察和生物标志物。视网膜血管的规律性和宽度的变化可以作为整体血管系统健康状况的指标，有助于识别高血压和心肌梗死等血管疾病高风险患者。多年来，各种深度学习架构被提出用于执行视网膜血管分割。最近，在图像分割任务中越来越多地使用注意力机制。本文提出了一种新的基于注意力机制的深度学习方法用于动脉-静脉分割。该新方法将注意力机制整合到了WNet深度学习模型中，我们称之为Attention-WNet。所提出的观点已经在HRF和DRIVE等公开可用的数据集上进行了测试，并在文献中的其他先进模型中表现出色。 

---
# EmbodiSwap for Zero-Shot Robot Imitation Learning 

**Title (ZH)**: EmbodiSwap 用于零样本机器人模仿学习 

**Authors**: Eadom Dessalene, Pavan Mantripragada, Michael Maynord, Yiannis Aloimonos  

**Link**: [PDF](https://arxiv.org/pdf/2510.03706)  

**Abstract**: We introduce EmbodiSwap - a method for producing photorealistic synthetic robot overlays over human video. We employ EmbodiSwap for zero-shot imitation learning, bridging the embodiment gap between in-the-wild ego-centric human video and a target robot embodiment. We train a closed-loop robot manipulation policy over the data produced by EmbodiSwap. We make novel use of V-JEPA as a visual backbone, repurposing V-JEPA from the domain of video understanding to imitation learning over synthetic robot videos. Adoption of V-JEPA outperforms alternative vision backbones more conventionally used within robotics. In real-world tests, our zero-shot trained V-JEPA model achieves an $82\%$ success rate, outperforming a few-shot trained $\pi_0$ network as well as $\pi_0$ trained over data produced by EmbodiSwap. We release (i) code for generating the synthetic robot overlays which takes as input human videos and an arbitrary robot URDF and generates a robot dataset, (ii) the robot dataset we synthesize over EPIC-Kitchens, HOI4D and Ego4D, and (iii) model checkpoints and inference code, to facilitate reproducible research and broader adoption. 

**Abstract (ZH)**: 我们介绍了EmbodiSwap——一种在人类视频上生成逼真合成机器人叠加的方法。我们使用EmbodiSwap进行零样本模仿学习，填补了野生主观人类视频与目标机器人身体之间的差距。我们通过EmbodiSwap生成的数据对闭环机器人操作策略进行训练。我们创新地将V-JEPA用作视觉骨干，并将其从视频理解领域重新利用到合成机器人视频的模仿学习中。采用V-JEPA优于机器人领域中更常用的传统视觉骨干。在实际测试中，我们的零样本训练的V-JEPA模型实现了82%的成功率，超过了少量样本训练的$\pi_0$网络以及基于EmbodiSwap生成数据训练的$\pi_0$网络。我们发布了(i)用于生成合成机器人叠加的代码，该代码以人类视频和任意机器人URDF作为输入生成机器人数据集，(ii)我们在EPIC-Kitchens、HOI4D和Ego4D上合成的机器人数据集，以及(iii)模型检查点和推理代码，以促进可重复研究和更广泛的采用。 

---
# Referring Expression Comprehension for Small Objects 

**Title (ZH)**: 小目标的引用表达理解 

**Authors**: Kanoko Goto, Takumi Hirose, Mahiro Ukai, Shuhei Kurita, Nakamasa Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2510.03701)  

**Abstract**: Referring expression comprehension (REC) aims to localize the target object described by a natural language expression. Recent advances in vision-language learning have led to significant performance improvements in REC tasks. However, localizing extremely small objects remains a considerable challenge despite its importance in real-world applications such as autonomous driving. To address this issue, we introduce a novel dataset and method for REC targeting small objects. First, we present the small object REC (SOREC) dataset, which consists of 100,000 pairs of referring expressions and corresponding bounding boxes for small objects in driving scenarios. Second, we propose the progressive-iterative zooming adapter (PIZA), an adapter module for parameter-efficient fine-tuning that enables models to progressively zoom in and localize small objects. In a series of experiments, we apply PIZA to GroundingDINO and demonstrate a significant improvement in accuracy on the SOREC dataset. Our dataset, codes and pre-trained models are publicly available on the project page. 

**Abstract (ZH)**: 小目标参照表达理解（SOREC）旨在定位自然语言表达描述的小目标物体。尽管视觉-语言学习的进步已经在参照表达理解（REC）任务上取得了显著的性能提升，但在自动驾驶等实际应用场景中，定位极小目标依然是一项重大挑战。为解决这一问题，我们提出了一个针对小目标的新型数据集和方法。首先，我们展示了包含100,000个-driver场景中小目标的参照表达和对应边界框的SOREC数据集。其次，我们提出了渐进迭代放大型（PIZA），这是一种用于参数高效微调的适配器模块，使模型能够逐步放大并定位小目标。在一系列实验中，我们将PIZA应用于GroundingDINO，并在SOREC数据集上证明了显著的准确性提高。我们的数据集、代码和预训练模型在项目页面上公开可用。 

---
# Dissecting Larval Zebrafish Hunting using Deep Reinforcement Learning Trained RNN Agents 

**Title (ZH)**: dissecting 幼鱼猎食行为using 深度强化学习训练的RNN代理模型 

**Authors**: Raaghav Malik, Satpreet H. Singh, Sonja Johnson-Yu, Nathan Wu, Roy Harpaz, Florian Engert, Kanaka Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03699)  

**Abstract**: Larval zebrafish hunting provides a tractable setting to study how ecological and energetic constraints shape adaptive behavior in both biological brains and artificial agents. Here we develop a minimal agent-based model, training recurrent policies with deep reinforcement learning in a bout-based zebrafish simulator. Despite its simplicity, the model reproduces hallmark hunting behaviors -- including eye vergence-linked pursuit, speed modulation, and stereotyped approach trajectories -- that closely match real larval zebrafish. Quantitative trajectory analyses show that pursuit bouts systematically reduce prey angle by roughly half before strike, consistent with measurements. Virtual experiments and parameter sweeps vary ecological and energetic constraints, bout kinematics (coupled vs. uncoupled turns and forward motion), and environmental factors such as food density, food speed, and vergence limits. These manipulations reveal how constraints and environments shape pursuit dynamics, strike success, and abort rates, yielding falsifiable predictions for neuroscience experiments. These sweeps identify a compact set of constraints -- binocular sensing, the coupling of forward speed and turning in bout kinematics, and modest energetic costs on locomotion and vergence -- that are sufficient for zebrafish-like hunting to emerge. Strikingly, these behaviors arise in minimal agents without detailed biomechanics, fluid dynamics, circuit realism, or imitation learning from real zebrafish data. Taken together, this work provides a normative account of zebrafish hunting as the optimal balance between energetic cost and sensory benefit, highlighting the trade-offs that structure vergence and trajectory dynamics. We establish a virtual lab that narrows the experimental search space and generates falsifiable predictions about behavior and neural coding. 

**Abstract (ZH)**: 针对生态和能量约束如何塑造生物大脑和人工代理适应性行为的模式鱼胚胎猎食提供了可操作的研究环境：通过基于代理的最小化模型和深度强化学习在回合制模式鱼模拟器中训练循环策略进行探究。 

---
# REG: A Regularization Optimizer for Robust Training Dynamics 

**Title (ZH)**: REG：一种用于稳健训练动力学的正则化优化器 

**Authors**: Zehua Liu, Han Wu, Xiaojin Fu, Shuqi Liu, Xiongwei Han, Tao Zhong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03691)  

**Abstract**: Optimizers are crucial for the efficient training of Large Language Models (LLMs). While AdamW is the de facto standard, recent structure-aware optimizers like Muon have emerged, which regularize gradient updates by operating on entire weight matrices. The Muon optimizer balances the gradient updates along all the directions. However, Muon's reliance on the matrix sign function can lead to training instability, exhibits incompatibility when fine-tuning models pre-trained with AdamW. To address these limitations, we propose \textbf{REG}, a novel optimizer that replaces Muon's aggressive matrix sign operator with the Row-and-Column-Scaling (RACS) operator. Theoretically grounded in balancing a matrix, the RACS operator regularizes the update steps in a less drastic manner, making it simpler to implement and more compatible with established training dynamics. Through extensive empirical experiments on LLM training, we demonstrate that our REG optimizer not only achieves superior performance and stability over AdamW, but also maintains consistency with the AdamW training paradigm. This consistency is particularly evident during the fine-tuning stage, where REG optimizer avoids the performance degradation observed with Muon. 

**Abstract (ZH)**: 优化器是大型语言模型高效训练的关键。尽管AdamW是首选标准，但最近出现的结构感知优化器如Muon通过操作整个权重矩阵来正则化梯度更新。Muon优化器在所有方向上平衡梯度更新。然而，Muon依赖矩阵符号函数可能导致训练不稳定，并且与使用AdamW预训练的模型微调时表现出不兼容性。为解决这些问题，我们提出了一种名为REG的新优化器，它用行-列缩放（RACS）操作符取代了Muon的激进矩阵符号操作符。RACS操作符在理论上通过平衡矩阵的方式来正则化更新步骤，使其实现方式更简单且与现有的训练动态更兼容。通过在大型语言模型训练中的广泛实验证明，我们的REG优化器不仅在性能和稳定性上优于AdamW，而且也保持了与AdamW训练范式的兼容性。特别是在微调阶段，REG优化器避免了Muon所观察到的性能下降。 

---
# MedReflect: Teaching Medical LLMs to Self-Improve via Reflective Correction 

**Title (ZH)**: MedReflect: 教学医疗LLM通过反思性修正实现自我提升 

**Authors**: Yue Huang, Yanyuan Chen, Dexuan Xu, Weihua Yue, Huamin Zhang, Meikang Qiu, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03687)  

**Abstract**: Medical problem solving demands expert knowledge and intricate reasoning. Recent studies of large language models (LLMs) attempt to ease this complexity by introducing external knowledge verification through retrieval-augmented generation or by training on reasoning datasets. However, these approaches suffer from drawbacks such as retrieval overhead and high annotation costs, and they heavily rely on substituted external assistants to reach limited performance in medical field. In this paper, we introduce MedReflect, a generalizable framework designed to inspire LLMs with a physician-like reflective thinking mode. MedReflect generates a single-pass reflection chain that includes initial hypothesis generation, self-questioning, self-answering and decision refinement. This self-verified and self-reflective nature releases large language model's latent capability in medical problem-solving without external retrieval or heavy annotation. We demonstrate that MedReflect enables cost-efficient medical dataset construction: with merely 2,000 randomly sampled training examples and a light fine-tuning, this approach achieves notable absolute accuracy improvements across a series of medical benchmarks while cutting annotation requirements. Our results provide evidence that LLMs can learn to solve specialized medical problems via self-reflection and self-improve, reducing reliance on external supervision and extensive task-specific fine-tuning data. 

**Abstract (ZH)**: 医学问题解决要求专家知识和复杂推理。大规模语言模型（LLMs）近期研究尝试通过检索增强生成或在推理数据集上进行训练来简化这一复杂性。然而，这些方法存在检索开销大和高标注成本等问题，并且严重依赖外部助手以在医疗领域达到有限的效果。本文介绍了一种通用框架MedReflect，旨在激发LLMs具有类似医生的反思思考模式。MedReflect生成了一次性反射链，包括初始假设生成、自我提问、自我回答和决策精炼。这种自我验证和自我反思的特性在无需外部检索或大量标注的情况下释放了大规模语言模型在医学问题解决中的潜在能力。我们证明了MedReflect能够高效低成本构建医学数据集：仅使用2,000个随机选择的训练示例和轻微微调，该方法在一系列医学基准测试中实现了显著的准确率提升，同时减少了标注需求。我们的结果提供了证据，表明LLMs可以通过自我反思和自我改进，学习解决专业医学问题，减少对外部监督和特定任务微调数据的依赖。 

---
# MonitorVLM:A Vision Language Framework for Safety Violation Detection in Mining Operations 

**Title (ZH)**: MonitorVLM：用于采矿作业安全违规检测的视觉语言框架 

**Authors**: Jiang Wu, Sichao Wu, Yinsong Ma, Guangyuan Yu, Haoyuan Xu, Lifang Zheng, Jingliang Duan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03666)  

**Abstract**: Industrial accidents, particularly in high-risk domains such as surface and underground mining, are frequently caused by unsafe worker behaviors. Traditional manual inspection remains labor-intensive, error-prone, and insufficient for large-scale, dynamic environments, highlighting the urgent need for intelligent and automated safety monitoring. In this paper, we present MonitorVLM, a novel vision--language framework designed to detect safety violations directly from surveillance video streams. MonitorVLM introduces three key innovations: (1) a domain-specific violation dataset comprising 9,000 vision--question--answer (VQA) samples across 40 high-frequency mining regulations, enriched with augmentation and auxiliary detection cues; (2) a clause filter (CF) module that dynamically selects the Top-$K$ most relevant clauses, reducing inference latency by 13.56\% while maintaining accuracy; and (3) a behavior magnifier (BM) module that enhances worker regions to improve fine-grained action recognition, yielding additional gains of 3.45% in precision and 8.62% in recall. Experimental results demonstrate that MonitorVLM significantly outperforms baseline vision--language models, achieving improvements of 22.01% in precision, 34.22\% in recall, and 28.37% in F1 score over the 72B unfine-tuned baseline. A lightweight web-based interface further integrates MonitorVLM into practical workflows, enabling automatic violation reporting with video timestamping. This study highlights the potential of multimodal large models to enhance occupational safety monitoring in mining and beyond. 

**Abstract (ZH)**: 基于视觉-语言框架的工业事故监测系统：MonitorVLM在采矿领域的应用研究 

---
# Operationalizing Data Minimization for Privacy-Preserving LLM Prompting 

**Title (ZH)**: privacy-preserving LLM提示中的数据最小化实现 

**Authors**: Jijie Zhou, Niloofar Mireshghallah, Tianshi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03662)  

**Abstract**: The rapid deployment of large language models (LLMs) in consumer applications has led to frequent exchanges of personal information. To obtain useful responses, users often share more than necessary, increasing privacy risks via memorization, context-based personalization, or security breaches. We present a framework to formally define and operationalize data minimization: for a given user prompt and response model, quantifying the least privacy-revealing disclosure that maintains utility, and we propose a priority-queue tree search to locate this optimal point within a privacy-ordered transformation space. We evaluated the framework on four datasets spanning open-ended conversations (ShareGPT, WildChat) and knowledge-intensive tasks with single-ground-truth answers (CaseHold, MedQA), quantifying achievable data minimization with nine LLMs as the response model. Our results demonstrate that larger frontier LLMs can tolerate stronger data minimization while maintaining task quality than smaller open-source models (85.7% redaction for GPT-5 vs. 19.3% for Qwen2.5-0.5B). By comparing with our search-derived benchmarks, we find that LLMs struggle to predict optimal data minimization directly, showing a bias toward abstraction that leads to oversharing. This suggests not just a privacy gap, but a capability gap: models may lack awareness of what information they actually need to solve a task. 

**Abstract (ZH)**: 大型语言模型在消费者应用中的迅速部署导致了个人数据的频繁交换。为了获取有用响应，用户往往会分享过多的信息，增加了通过记忆、基于上下文的个性化或安全漏洞泄露隐私的风险。我们提出了一种框架来正式定义和实现数据最小化：对于给定的用户提示和响应模型，量化最少的隐私泄露披露以保持其实用性，并提出了一种优先级队列树搜索方法，在隐私有序变换空间中寻找这一最优点。我们在四个数据集（ShareGPT、WildChat、CaseHold、MedQA）上评估了该框架，采用九个大型语言模型作为响应模型，量化了可实现的数据最小化程度。结果显示，更前沿的大型语言模型在数据最小化的同时能保持任务质量方面优于较小的开源模型（GPT-5的数据去除率为85.7%，而Qwen2.5-0.5B的数据去除率为19.3%）。通过与搜索得出的基准进行比较，我们发现语言模型难以直接预测最优数据最小化，显示出对抽象的偏好，导致信息过度共享。这不仅表明了隐私差距，而且反映了能力差距：模型可能缺乏对完成任务所需信息的认识。 

---
# Does higher interpretability imply better utility? A Pairwise Analysis on Sparse Autoencoders 

**Title (ZH)**: 更高的可解释性是否意味着更好的实用性？稀疏自编码器的配对分析 

**Authors**: Xu Wang, Yan Hu, Benyou Wang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03659)  

**Abstract**: Sparse Autoencoders (SAEs) are widely used to steer large language models (LLMs), based on the assumption that their interpretable features naturally enable effective model behavior steering. Yet, a fundamental question remains unanswered: does higher interpretability indeed imply better steering utility? To answer this question, we train 90 SAEs across three LLMs (Gemma-2-2B, Qwen-2.5-3B, Gemma-2-9B), spanning five architectures and six sparsity levels, and evaluate their interpretability and steering utility based on SAEBench (arXiv:2501.12345) and AxBench (arXiv:2502.23456) respectively, and perform a rank-agreement analysis via Kendall's rank coefficients (tau b). Our analysis reveals only a relatively weak positive association (tau b approx 0.298), indicating that interpretability is an insufficient proxy for steering performance. We conjecture the interpretability utility gap may stem from the selection of SAE features, as not all of them are equally effective for steering. To further find features that truly steer the behavior of LLMs, we propose a novel selection criterion called Delta Token Confidence, which measures how much amplifying a feature changes the next token distribution. We show that our method improves the steering performance of three LLMs by 52.52 percent compared to the current best output score based criterion (arXiv:2503.34567). Strikingly, after selecting features with high Delta Token Confidence, the correlation between interpretability and utility vanishes (tau b approx 0), and can even become negative. This further highlights the divergence between interpretability and utility for the most effective steering features. 

**Abstract (ZH)**: Sparse Autoencoders的稀疏性与大型语言模型行为引导的有效性：一个实证分析 

---
# LLM-Guided Evolutionary Program Synthesis for Quasi-Monte Carlo Design 

**Title (ZH)**: LLM 引导的演化程序合成在准蒙特卡洛设计中的应用 

**Authors**: Amir Sadikov  

**Link**: [PDF](https://arxiv.org/pdf/2510.03650)  

**Abstract**: Low-discrepancy point sets and digital sequences underpin quasi-Monte Carlo (QMC) methods for high-dimensional integration. We cast two long-standing QMC design problems as program synthesis and solve them with an LLM-guided evolutionary loop that mutates and selects code under task-specific fitness: (i) constructing finite 2D/3D point sets with low star discrepancy, and (ii) choosing Sobol' direction numbers that minimize randomized QMC error on downstream integrands. Our two-phase procedure combines constructive code proposals with iterative numerical refinement. On finite sets, we rediscover known optima in small 2D cases and set new best-known 2D benchmarks for N >= 40, while matching most known 3D optima up to the proven frontier (N <= 8) and reporting improved 3D benchmarks beyond. On digital sequences, evolving Sobol' parameters yields consistent reductions in randomized quasi-Monte Carlo (rQMC) mean-squared error for several 32-dimensional option-pricing tasks relative to widely used Joe--Kuo parameters, while preserving extensibility to any sample size and compatibility with standard randomizations. Taken together, the results demonstrate that LLM-driven evolutionary program synthesis can automate the discovery of high-quality QMC constructions, recovering classical designs where they are optimal and improving them where finite-N structure matters. Data and code are available at this https URL. 

**Abstract (ZH)**: 低散点差异点集和数字序列支撑着高维积分的准蒙特卡洛（QMC）方法。我们将两个长期存在的QMC设计问题视为程序合成，并使用一个由特定任务适应性fitness引导的进化循环来突变和选择代码：（i）构建具有低星散性的有限2D/3D点集；（ii）选择Sobol'方向数以最小化对下游积分函数的随机化QMC误差。我们的两阶段过程结合了构造性代码提案与迭代数值精炼。在有限集合上，我们重新发现了小型2D情况下的已知最优值，并为N >= 40设置了新的最优基准，同时在大多数已知3D最优值（N <= 8）范围内与之匹配，并报告了改进的3D基准。在数字序列上，演化Sobol'参数相对于广泛使用的Joe-Kuo参数，对于多个32维的期权定价任务，提供了随机化准蒙特卡洛（rQMC）均方误差的一致减少，同时保留了对任何样本量的扩展能力和与标准随机化的一致性。综合来看，结果表明，基于大语言模型的进化程序合成可以自动化高质量QMC构造的发现，恢复最优的经典设计，并在有限-N结构重要的情况下进一步改进它们。相关数据和代码可通过以下链接获取。 

---
# Towards Unsupervised Speech Recognition at the Syllable-Level 

**Title (ZH)**: 向 syllable 级无监督语音识别迈进 

**Authors**: Liming Wang, Junrui Ni, Kai-Wei Chang, Saurabhchand Bhati, David Harwath, Mark Hasegawa-Johnson, James R. Glass  

**Link**: [PDF](https://arxiv.org/pdf/2510.03639)  

**Abstract**: Training speech recognizers with unpaired speech and text -- known as unsupervised speech recognition (UASR) -- is a crucial step toward extending ASR to low-resource languages in the long-tail distribution and enabling multimodal learning from non-parallel data. However, existing approaches based on phones often rely on costly resources such as grapheme-to-phoneme converters (G2Ps) and struggle to generalize to languages with ambiguous phoneme boundaries due to training instability. In this paper, we address both challenges by introducing a syllable-level UASR framework based on masked language modeling, which avoids the need for G2P and the instability of GAN-based methods. Our approach achieves up to a 40\% relative reduction in character error rate (CER) on LibriSpeech and generalizes effectively to Mandarin, a language that has remained particularly difficult for prior methods. Code will be released upon acceptance. 

**Abstract (ZH)**: 使用未配对语音和文本训练语音识别器——即无监督语音识别（UASR）——是将ASR扩展到长尾分布中的低资源语言并从非配对数据中实现多模态学习的关键步骤。然而，现有的基于音素的方法往往依赖于昂贵的资源，如字母到音素转换器（G2P），并且难以泛化到具有模糊音素边界的语言，这是由于训练不稳定性。本文通过引入基于掩码语言建模的音节级UASR框架，同时避免了G2P的需求和GAN方法的不稳定性，从而解决了这两个挑战。我们的方法在LibriSpeech上实现了字符错误率（CER）高达40%的相对降低，并且能够有效泛化到先前方法特别难以处理的 Mandarin 语言。接受发表后将公开代码。 

---
# Implicit Models: Expressive Power Scales with Test-Time Compute 

**Title (ZH)**: 隐式模型：表示能力随测试时计算量而变化 

**Authors**: Jialin Liu, Lisang Ding, Stanley Osher, Wotao Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.03638)  

**Abstract**: Implicit models, an emerging model class, compute outputs by iterating a single parameter block to a fixed point. This architecture realizes an infinite-depth, weight-tied network that trains with constant memory, significantly reducing memory needs for the same level of performance compared to explicit models. While it is empirically known that these compact models can often match or even exceed larger explicit networks by allocating more test-time compute, the underlying mechanism remains poorly understood.
We study this gap through a nonparametric analysis of expressive power. We provide a strict mathematical characterization, showing that a simple and regular implicit operator can, through iteration, progressively express more complex mappings. We prove that for a broad class of implicit models, this process lets the model's expressive power scale with test-time compute, ultimately matching a much richer function class. The theory is validated across three domains: image reconstruction, scientific computing, and operations research, demonstrating that as test-time iterations increase, the complexity of the learned mapping rises, while the solution quality simultaneously improves and stabilizes. 

**Abstract (ZH)**: 隐式模型是一种新兴的模型类，通过迭代单个参数块至固定点来计算输出。这种架构实现了无限深度、权重共享的网络，能够在保持相同性能水平的同时显著减少内存需求。虽然经验上已知这些紧凑模型往往能够匹配甚至超越更大规模的显式网络，但它们背后的工作机制仍不甚理解。

我们通过非参数分析表现能力来研究这一差距。我们提供了一个严格的数学刻画，证明一个简单的规律性隐式操作可以通过迭代逐步表达更复杂的映射。我们证明，在广泛类型的隐式模型中，这一过程能够使模型的表现能力随测试时的计算量扩展，最终匹配更为丰富的函数类。该理论在图像重建、科学计算和运筹学三个领域得到验证，表明随着测试时迭代次数的增加，学习映射的复杂性提高，而解的质量同时改善并稳定。 

---
# Predicting Stock Price Movement with LLM-Enhanced Tweet Emotion Analysis 

**Title (ZH)**: 使用LLM增强的推特情感分析预测股票价格变动 

**Authors**: An Vuong, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2510.03633)  

**Abstract**: Accurately predicting short-term stock price movement remains a challenging task due to the market's inherent volatility and sensitivity to investor sentiment. This paper discusses a deep learning framework that integrates emotion features extracted from tweet data with historical stock price information to forecast significant price changes on the following day. We utilize Meta's Llama 3.1-8B-Instruct model to preprocess tweet data, thereby enhancing the quality of emotion features derived from three emotion analysis approaches: a transformer-based DistilRoBERTa classifier from the Hugging Face library and two lexicon-based methods using National Research Council Canada (NRC) resources. These features are combined with previous-day stock price data to train a Long Short-Term Memory (LSTM) model. Experimental results on TSLA, AAPL, and AMZN stocks show that all three emotion analysis methods improve the average accuracy for predicting significant price movements, compared to the baseline model using only historical stock prices, which yields an accuracy of 13.5%. The DistilRoBERTa-based stock prediction model achieves the best performance, with accuracy rising from 23.6% to 38.5% when using LLaMA-enhanced emotion analysis. These results demonstrate that using large language models to preprocess tweet content enhances the effectiveness of emotion analysis which in turn improves the accuracy of predicting significant stock price movements. 

**Abstract (ZH)**: 基于推特数据情感特征增强的历史股价信息的短期股市价格变动预测：一个深度学习框架 

---
# Explainable but Vulnerable: Adversarial Attacks on XAI Explanation in Cybersecurity Applications 

**Title (ZH)**: 可解释但易受攻击：网络安全应用中XAI解释的 adversarial攻击 

**Authors**: Maraz Mia, Mir Mehedi A. Pritom  

**Link**: [PDF](https://arxiv.org/pdf/2510.03623)  

**Abstract**: Explainable Artificial Intelligence (XAI) has aided machine learning (ML) researchers with the power of scrutinizing the decisions of the black-box models. XAI methods enable looking deep inside the models' behavior, eventually generating explanations along with a perceived trust and transparency. However, depending on any specific XAI method, the level of trust can vary. It is evident that XAI methods can themselves be a victim of post-adversarial attacks that manipulate the expected outcome from the explanation module. Among such attack tactics, fairwashing explanation (FE), manipulation explanation (ME), and backdoor-enabled manipulation attacks (BD) are the notable ones. In this paper, we try to understand these adversarial attack techniques, tactics, and procedures (TTPs) on explanation alteration and thus the effect on the model's decisions. We have explored a total of six different individual attack procedures on post-hoc explanation methods such as SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanation), and IG (Integrated Gradients), and investigated those adversarial attacks in cybersecurity applications scenarios such as phishing, malware, intrusion, and fraudulent website detection. Our experimental study reveals the actual effectiveness of these attacks, thus providing an urgency for immediate attention to enhance the resiliency of XAI methods and their applications. 

**Abstract (ZH)**: 可解释人工智能(XAI)通过审查黑盒模型的决策帮助了机器学习(ML)研究人员，XAI方法使得深入理解模型行为成为可能，最终生成解释，同时提高对模型的信任度和透明度。然而，依赖于特定的XAI方法，信任度会有所不同。显然，XAI方法自身也容易受到后恶意对抗攻击的操纵，这些攻击会篡改解释模块的预期结果。在这类攻击手段中，公平洗牌解释(FE)、操控解释(ME)和后门启用的操纵攻击(BD)尤为突出。在本文中，我们试图理解这些对抗攻击技术、手法和程序(TTPs)对解释修改及其对模型决策的影响。我们探索了六种不同的个体攻击程序对事后解释方法（如SHAP、LIME和IG）的影响，并在钓鱼、恶意软件、入侵和欺诈性网站检测等网络安全应用场景中研究了这些对抗攻击。我们的实验研究揭示了这些攻击的实际效果，从而强调了即时改进XAI方法及其应用韧性的紧迫性。 

---
# Neural Bayesian Filtering 

**Title (ZH)**: 神经贝叶斯滤波 

**Authors**: Christopher Solinas, Radovan Haluska, David Sychrovsky, Finbarr Timbers, Nolan Bard, Michael Buro, Martin Schmid, Nathan R. Sturtevant, Michael Bowling  

**Link**: [PDF](https://arxiv.org/pdf/2510.03614)  

**Abstract**: We present Neural Bayesian Filtering (NBF), an algorithm for maintaining distributions over hidden states, called beliefs, in partially observable systems. NBF is trained to find a good latent representation of the beliefs induced by a task. It maps beliefs to fixed-length embedding vectors, which condition generative models for sampling. During filtering, particle-style updates compute posteriors in this embedding space using incoming observations and the environment's dynamics. NBF combines the computational efficiency of classical filters with the expressiveness of deep generative models - tracking rapidly shifting, multimodal beliefs while mitigating the risk of particle impoverishment. We validate NBF in state estimation tasks in three partially observable environments. 

**Abstract (ZH)**: 基于神经网络的贝叶斯滤波（NBF）算法：部分可观测系统中隐藏状态分布的维护 

---
# Can an LLM Induce a Graph? Investigating Memory Drift and Context Length 

**Title (ZH)**: Can an LLM Generate a Graph? Investigating Memory Drift and Context Length 

**Authors**: Raquib Bin Yousuf, Aadyant Khatri, Shengzhe Xu, Mandar Sharma, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03611)  

**Abstract**: Recently proposed evaluation benchmarks aim to characterize the effective context length and the forgetting tendencies of large language models (LLMs). However, these benchmarks often rely on simplistic 'needle in a haystack' retrieval or continuation tasks that may not accurately reflect the performance of these models in information-dense scenarios. Thus, rather than simple next token prediction, we argue for evaluating these models on more complex reasoning tasks that requires them to induce structured relational knowledge from the text - such as graphs from potentially noisy natural language content. While the input text can be viewed as generated in terms of a graph, its structure is not made explicit and connections must be induced from distributed textual cues, separated by long contexts and interspersed with irrelevant information. Our findings reveal that LLMs begin to exhibit memory drift and contextual forgetting at much shorter effective lengths when tasked with this form of relational reasoning, compared to what existing benchmarks suggest. With these findings, we offer recommendations for the optimal use of popular LLMs for complex reasoning tasks. We further show that even models specialized for reasoning, such as OpenAI o1, remain vulnerable to early memory drift in these settings. These results point to significant limitations in the models' ability to abstract structured knowledge from unstructured input and highlight the need for architectural adaptations to improve long-range reasoning. 

**Abstract (ZH)**: 最近提出的评估基准旨在表征大语言模型的有效上下文长度和遗忘倾向。然而，这些基准往往依赖于简单的“针haystack中找针”的检索或延续任务，这可能无法准确反映这些模型在信息密集型场景中的性能。因此，我们主张通过更复杂的推理任务来评估这些模型，这些任务要求它们从文本中推断出结构化的关系知识——例如，从潜在噪声自然语言内容中生成图形。虽然输入文本可以视作图形生成，但其结构并未明确呈现，连接必须从分散在长上下文和无关信息之间的分布式文本线索中推断出来。我们的研究发现，当这些模型被要求进行这种形式的推理时，它们在有效长度明显较短时就开始出现记忆漂移和上下文遗忘，这比现有基准所表明的要短得多。基于这些发现，我们提出了关于如何优化使用流行的大语言模型进行复杂推理任务的建议。我们进一步表明，即使是专门用于推理的模型，如OpenAI o1，在这种环境中依然容易出现早期的记忆漂移。这些结果揭示了大语言模型在从非结构化输入中抽象结构化知识方面存在的显著局限性，并强调了需要在架构上进行改进以提高长程推理能力。 

---
# PentestMCP: A Toolkit for Agentic Penetration Testing 

**Title (ZH)**: PentestMCP: 代理渗透测试工具包 

**Authors**: Zachary Ezetta, Wu-chang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.03610)  

**Abstract**: Agentic AI is transforming security by automating many tasks being performed manually. While initial agentic approaches employed a monolithic architecture, the Model-Context-Protocol has now enabled a remote-procedure call (RPC) paradigm to agentic applications, allowing for the flexible construction and composition of multi-function agents. This paper describes PentestMCP, a library of MCP server implementations that support agentic penetration testing. By supporting common penetration testing tasks such as network scanning, resource enumeration, service fingerprinting, vulnerability scanning, exploitation, and post-exploitation, PentestMCP allows a developer to customize multi-agent workflows for performing penetration tests. 

**Abstract (ZH)**: 代理型AI正通过自动化许多手动执行的任务来转变安全领域。由于模型-上下文-协议（MCP）模型的支持，代理型应用现在可以采用远程过程调用（RPC）模式，这使得多功能代理的灵活构建和组合成为可能。本文介绍了PentestMCP库，该库支持代理型渗透测试，并通过提供包括网络扫描、资源枚举、服务指纹识别、漏洞扫描、利用和后利用在内的常见渗透测试任务的支持，使开发者能够定制多代理工作流程以执行渗透测试。 

---
# Deep Domain Adaptation for Turbofan Engine Remaining Useful Life Prediction: Methodologies, Evaluation and Future Trends 

**Title (ZH)**: turbofan发动机剩余使用寿命预测的深度域适应方法、评估及未来趋势 

**Authors**: Yucheng Wang, Mohamed Ragab, Yubo Hou, Zhenghua Chen, Min Wu, Xiaoli Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03604)  

**Abstract**: Remaining Useful Life (RUL) prediction for turbofan engines plays a vital role in predictive maintenance, ensuring operational safety and efficiency in aviation. Although data-driven approaches using machine learning and deep learning have shown potential, they face challenges such as limited data and distribution shifts caused by varying operating conditions. Domain Adaptation (DA) has emerged as a promising solution, enabling knowledge transfer from source domains with abundant data to target domains with scarce data while mitigating distributional shifts. Given the unique properties of turbofan engines, such as complex operating conditions, high-dimensional sensor data, and slower-changing signals, it is essential to conduct a focused review of DA techniques specifically tailored to turbofan engines. To address this need, this paper provides a comprehensive review of DA solutions for turbofan engine RUL prediction, analyzing key methodologies, challenges, and recent advancements. A novel taxonomy tailored to turbofan engines is introduced, organizing approaches into methodology-based (how DA is applied), alignment-based (where distributional shifts occur due to operational variations), and problem-based (why certain adaptations are needed to address specific challenges). This taxonomy offers a multidimensional view that goes beyond traditional classifications by accounting for the distinctive characteristics of turbofan engine data and the standard process of applying DA techniques to this area. Additionally, we evaluate selected DA techniques on turbofan engine datasets, providing practical insights for practitioners and identifying key challenges. Future research directions are identified to guide the development of more effective DA techniques, advancing the state of RUL prediction for turbofan engines. 

**Abstract (ZH)**: 涡扇发动机剩余使用寿命（RUL）预测中的领域适应（DA）研究 

---
# Neon: Negative Extrapolation From Self-Training Improves Image Generation 

**Title (ZH)**: Neon: 自训练的负外推 improves 图像生成 

**Authors**: Sina Alemohammad, Zhangyang Wang, Richard G. Baraniuk  

**Link**: [PDF](https://arxiv.org/pdf/2510.03597)  

**Abstract**: Scaling generative AI models is bottlenecked by the scarcity of high-quality training data. The ease of synthesizing from a generative model suggests using (unverified) synthetic data to augment a limited corpus of real data for the purpose of fine-tuning in the hope of improving performance. Unfortunately, however, the resulting positive feedback loop leads to model autophagy disorder (MAD, aka model collapse) that results in a rapid degradation in sample quality and/or diversity. In this paper, we introduce Neon (for Negative Extrapolation frOm self-traiNing), a new learning method that turns the degradation from self-training into a powerful signal for self-improvement. Given a base model, Neon first fine-tunes it on its own self-synthesized data but then, counterintuitively, reverses its gradient updates to extrapolate away from the degraded weights. We prove that Neon works because typical inference samplers that favor high-probability regions create a predictable anti-alignment between the synthetic and real data population gradients, which negative extrapolation corrects to better align the model with the true data distribution. Neon is remarkably easy to implement via a simple post-hoc merge that requires no new real data, works effectively with as few as 1k synthetic samples, and typically uses less than 1% additional training compute. We demonstrate Neon's universality across a range of architectures (diffusion, flow matching, autoregressive, and inductive moment matching models) and datasets (ImageNet, CIFAR-10, and FFHQ). In particular, on ImageNet 256x256, Neon elevates the xAR-L model to a new state-of-the-art FID of 1.02 with only 0.36% additional training compute. Code is available at this https URL 

**Abstract (ZH)**: 负外推从自训练中提升生成AI模型：Neon的新学习方法 

---
# Deep Reinforcement Learning for Multi-Agent Coordination 

**Title (ZH)**: 多智能体协调的深度强化学习 

**Authors**: Kehinde O. Aina, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2510.03592)  

**Abstract**: We address the challenge of coordinating multiple robots in narrow and confined environments, where congestion and interference often hinder collective task performance. Drawing inspiration from insect colonies, which achieve robust coordination through stigmergy -- modifying and interpreting environmental traces -- we propose a Stigmergic Multi-Agent Deep Reinforcement Learning (S-MADRL) framework that leverages virtual pheromones to model local and social interactions, enabling decentralized emergent coordination without explicit communication. To overcome the convergence and scalability limitations of existing algorithms such as MADQN, MADDPG, and MAPPO, we leverage curriculum learning, which decomposes complex tasks into progressively harder sub-problems. Simulation results show that our framework achieves the most effective coordination of up to eight agents, where robots self-organize into asymmetric workload distributions that reduce congestion and modulate group performance. This emergent behavior, analogous to strategies observed in nature, demonstrates a scalable solution for decentralized multi-agent coordination in crowded environments with communication constraints. 

**Abstract (ZH)**: 基于蚁痕机制的多智能体深度强化学习框架：应对狭窄受限环境中多机器人协调挑战 

---
# A Hybrid Co-Finetuning Approach for Visual Bug Detection in Video Games 

**Title (ZH)**: 视频游戏中的视觉错误检测的混合共微调方法 

**Authors**: Faliu Yi, Sherif Abdelfattah, Wei Huang, Adrian Brown  

**Link**: [PDF](https://arxiv.org/pdf/2510.03591)  

**Abstract**: Manual identification of visual bugs in video games is a resource-intensive and costly process, often demanding specialized domain knowledge. While supervised visual bug detection models offer a promising solution, their reliance on extensive labeled datasets presents a significant challenge due to the infrequent occurrence of such bugs. To overcome this limitation, we propose a hybrid Co-FineTuning (CFT) method that effectively integrates both labeled and unlabeled data. Our approach leverages labeled samples from the target game and diverse co-domain games, additionally incorporating unlabeled data to enhance feature representation learning. This strategy maximizes the utility of all available data, substantially reducing the dependency on labeled examples from the specific target game. The developed framework demonstrates enhanced scalability and adaptability, facilitating efficient visual bug detection across various game titles. Our experimental results show the robustness of the proposed method for game visual bug detection, exhibiting superior performance compared to conventional baselines across multiple gaming environments. Furthermore, CFT maintains competitive performance even when trained with only 50% of the labeled data from the target game. 

**Abstract (ZH)**: 手动识别视频游戏中视觉错误是一个资源密集型和成本高昂的过程，通常需要专门的领域知识。虽然监督视觉错误检测模型提供了有前途的解决方案，但由于此类错误出现频率低，其依赖于海量标注数据集构成了一个重要的挑战。为克服这一限制，我们提出了一种有效的混合Co-FineTuning（CFT）方法，结合了标记和未标记的数据。我们的方法利用目标游戏及其不同领域游戏的标记样本，并进一步整合未标记数据以增强特征表示学习。该策略最大限度地利用了所有可用数据，显著减少了对目标游戏特定标记示例的依赖。所开发的框架展示了增强的可扩展性和适应性，促进了各种游戏标题中的高效视觉错误检测。实验结果表明，所提出的方法在游戏视觉错误检测方面具有鲁棒性，并在多个游戏环境中展现出优于传统基线方法的性能。此外，即使仅使用目标游戏标记数据的50%进行训练，CFT也能保持竞争力。 

---
# Deep learning the sources of MJO predictability: a spectral view of learned features 

**Title (ZH)**: 深度学习MJO可预测性的来源：学习特征的谱观分析 

**Authors**: Lin Yao, Da Yang, James P.C. Duncan, Ashesh Chattopadhyay, Pedram Hassanzadeh, Wahid Bhimji, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03582)  

**Abstract**: The Madden-Julian oscillation (MJO) is a planetary-scale, intraseasonal tropical rainfall phenomenon crucial for global weather and climate; however, its dynamics and predictability remain poorly understood. Here, we leverage deep learning (DL) to investigate the sources of MJO predictability, motivated by a central difference in MJO theories: which spatial scales are essential for driving the MJO? We first develop a deep convolutional neural network (DCNN) to forecast the MJO indices (RMM and ROMI). Our model predicts RMM and ROMI up to 21 and 33 days, respectively, achieving skills comparable to leading subseasonal-to-seasonal models such as NCEP. To identify the spatial scales most relevant for MJO forecasting, we conduct spectral analysis of the latent feature space and find that large-scale patterns dominate the learned signals. Additional experiments show that models using only large-scale signals as the input have the same skills as those using all the scales, supporting the large-scale view of the MJO. Meanwhile, we find that small-scale signals remain informative: surprisingly, models using only small-scale input can still produce skillful forecasts up to 1-2 weeks ahead. We show that this is achieved by reconstructing the large-scale envelope of the small-scale activities, which aligns with the multi-scale view of the MJO. Altogether, our findings support that large-scale patterns--whether directly included or reconstructed--may be the primary source of MJO predictability. 

**Abstract (ZH)**: Madden-Julian振荡（MJO）的可预报性来源：深学习视角下的大尺度与小尺度作用 

---
# Latent Mixture of Symmetries for Sample-Efficient Dynamic Learning 

**Title (ZH)**: 潜在对称混合的样本高效动力学习 

**Authors**: Haoran Li, Chenhan Xiao, Muhao Guo, Yang Weng  

**Link**: [PDF](https://arxiv.org/pdf/2510.03578)  

**Abstract**: Learning dynamics is essential for model-based control and Reinforcement Learning in engineering systems, such as robotics and power systems. However, limited system measurements, such as those from low-resolution sensors, demand sample-efficient learning. Symmetry provides a powerful inductive bias by characterizing equivariant relations in system states to improve sample efficiency. While recent methods attempt to discover symmetries from data, they typically assume a single global symmetry group and treat symmetry discovery and dynamic learning as separate tasks, leading to limited expressiveness and error accumulation. In this paper, we propose the Latent Mixture of Symmetries (Latent MoS), an expressive model that captures a mixture of symmetry-governed latent factors from complex dynamical measurements. Latent MoS focuses on dynamic learning while locally and provably preserving the underlying symmetric transformations. To further capture long-term equivariance, we introduce a hierarchical architecture that stacks MoS blocks. Numerical experiments in diverse physical systems demonstrate that Latent MoS outperforms state-of-the-art baselines in interpolation and extrapolation tasks while offering interpretable latent representations suitable for future geometric and safety-critical analyses. 

**Abstract (ZH)**: 基于模型的控制和强化学习在工程系统中（如机器人和电力系统）需要学习动力学。然而，受限于有限的系统测量，如低分辨率传感器的数据，要求高效学习。对称性通过表征系统状态下的等变关系来提供强大的归纳偏置，从而提高学习效率。虽然近期方法尝试从数据中发现对称性，但它们通常假设单一全局对称群，并将对称性发现和动态学习视为分离任务，导致表达能力有限和错误积累。本文提出了一种表达性强的模型——隐含混合对称性（Latent Mixture of Symmetries, Latent MoS），该模型从复杂的动态测量中捕捉由对称性控制的潜在因素混合。Latent MoS聚焦于动态学习，同时局部和可证明地保留底层对称变换。为进一步捕捉长期等变性，我们引入了层次架构，将MoS块堆叠起来。在多种物理系统中的数值实验表明，Latent MoS在插值和外推任务中优于最先进的基线方法，同时提供易于未来几何和安全关键分析的可解释潜在表示。 

---
# Generalization of Graph Neural Network Models for Distribution Grid Fault Detection 

**Title (ZH)**: 分布式电网故障检测的图神经网络模型通用化 

**Authors**: Burak Karabulut, Carlo Manna, Chris Develder  

**Link**: [PDF](https://arxiv.org/pdf/2510.03571)  

**Abstract**: Fault detection in power distribution grids is critical for ensuring system reliability and preventing costly outages. Moreover, fault detection methodologies should remain robust to evolving grid topologies caused by factors such as reconfigurations, equipment failures, and Distributed Energy Resource (DER) integration. Current data-driven state-of-the-art methods use Recurrent Neural Networks (RNNs) for temporal modeling and Graph Neural Networks (GNNs) for spatial learning, in an RNN+GNN pipeline setting (RGNN in short). Specifically, for power system fault diagnosis, Graph Convolutional Networks (GCNs) have been adopted. Yet, various more advanced GNN architectures have been proposed and adopted in domains outside of power systems. In this paper, we set out to systematically and consistently benchmark various GNN architectures in an RNN+GNN pipeline model. Specifically, to the best of our knowledge, we are the first to (i) propose to use GraphSAGE and Graph Attention (GAT, GATv2) in an RGNN for fault diagnosis, and (ii) provide a comprehensive benchmark against earlier proposed RGNN solutions (RGCN) as well as pure RNN models (especially Gated Recurrent Unit (GRU)), particularly (iii) exploring their generalization potential for deployment in different settings than those used for training them. Our experimental results on the IEEE 123-node distribution network show that RGATv2 has superior generalization capabilities, maintaining high performance with an F1-score reduction of $\sim$12% across different topology settings. In contrast, pure RNN models largely fail, experiencing an F1-score reduction of up to $\sim$60%, while other RGNN variants also exhibit significant performance degradation, i.e., up to $\sim$25% lower F1-scores. 

**Abstract (ZH)**: 基于RNN+GNN架构的故障检测方法系统性评估：以电力分配 grids 为例 

---
# Evaluating OCR performance on food packaging labels in South Africa 

**Title (ZH)**: 评估South Africa食品包装标签上的OCR性能 

**Authors**: Mayimunah Nagayi, Alice Khan, Tamryn Frank, Rina Swart, Clement Nyirenda  

**Link**: [PDF](https://arxiv.org/pdf/2510.03570)  

**Abstract**: This study evaluates four open-source Optical Character Recognition (OCR) systems which are Tesseract, EasyOCR, PaddleOCR, and TrOCR on real world food packaging images. The aim is to assess their ability to extract ingredient lists and nutrition facts panels. Accurate OCR for packaging is important for compliance and nutrition monitoring but is challenging due to multilingual text, dense layouts, varied fonts, glare, and curved surfaces. A dataset of 231 products (1,628 images) was processed by all four models to assess speed and coverage, and a ground truth subset of 113 images (60 products) was created for accuracy evaluation. Metrics include Character Error Rate (CER), Word Error Rate (WER), BLEU, ROUGE-L, F1, coverage, and execution time. On the ground truth subset, Tesseract achieved the lowest CER (0.912) and the highest BLEU (0.245). EasyOCR provided a good balance between accuracy and multilingual support. PaddleOCR achieved near complete coverage but was slower because it ran on CPU only due to GPU incompatibility, and TrOCR produced the weakest results despite GPU acceleration. These results provide a packaging-specific benchmark, establish a baseline, and highlight directions for layout-aware methods and text localization. 

**Abstract (ZH)**: 本研究评估了四种开源光学字符识别（OCR）系统——Tesseract、EasyOCR、PaddleOCR和TrOCR在真实食品包装图像上的性能，旨在评估其提取配料列表和营养成分表的能力。准确的包装OCR对于合规性和营养监测至关重要，但由于多语言文本、密集布局、变体字体、反光和曲面等因素，这一过程具有挑战性。本研究处理了231种产品的1,628张图像，并创建了包含113张图像（60种产品）的基准集，用于准确率评估。评估指标包括字符错误率（CER）、词错误率（WER）、BLEU、ROUGE-L、F1、覆盖率和执行时间。在基准集中，Tesseract实现了最低的CER（0.912）和最高的BLEU（0.245）。EasyOCR在准确性和多语言支持方面提供了良好的平衡。PaddleOCR实现了接近完整的覆盖率，但由于GPU不兼容只能在CPU上运行，因此速度较慢；而TrOCR尽管有GPU加速，但在准确率方面表现最差。这些结果提供了特定于包装的应用基准、建立了基线，并指出了布局感知方法和文本定位的方向。 

---
# Longitudinal Flow Matching for Trajectory Modeling 

**Title (ZH)**: 纵向流匹配用于轨迹建模 

**Authors**: Mohammad Mohaiminul Islam, Thijs P. Kuipers, Sharvaree Vadgama, Coen de Vente, Afsana Khan, Clara I. Sánchez, Erik J. Bekkers  

**Link**: [PDF](https://arxiv.org/pdf/2510.03569)  

**Abstract**: Generative models for sequential data often struggle with sparsely sampled and high-dimensional trajectories, typically reducing the learning of dynamics to pairwise transitions. We propose \textit{Interpolative Multi-Marginal Flow Matching} (IMMFM), a framework that learns continuous stochastic dynamics jointly consistent with multiple observed time points. IMMFM employs a piecewise-quadratic interpolation path as a smooth target for flow matching and jointly optimizes drift and a data-driven diffusion coefficient, supported by a theoretical condition for stable learning. This design captures intrinsic stochasticity, handles irregular sparse sampling, and yields subject-specific trajectories. Experiments on synthetic benchmarks and real-world longitudinal neuroimaging datasets show that IMMFM outperforms existing methods in both forecasting accuracy and further downstream tasks. 

**Abstract (ZH)**: 插值多边际流匹配（IMMFM）：用于多观测时间点的一致连续随机动力学习 

---
# Reactive Transformer (RxT) -- Stateful Real-Time Processing for Event-Driven Reactive Language Models 

**Title (ZH)**: Reactive Transformer (RxT) —— 基于状态的事件驱动实时处理反应式语言模型 

**Authors**: Adam Filipek  

**Link**: [PDF](https://arxiv.org/pdf/2510.03561)  

**Abstract**: The Transformer architecture has become the de facto standard for Large Language Models (LLMs), demonstrating remarkable capabilities in language understanding and generation. However, its application in conversational AI is fundamentally constrained by its stateless nature and the quadratic computational complexity ($O(L^2)$) with respect to sequence length $L$. Current models emulate memory by reprocessing an ever-expanding conversation history with each turn, leading to prohibitive costs and latency in long dialogues. This paper introduces the Reactive Transformer (RxT), a novel architecture designed to overcome these limitations by shifting from a data-driven to an event-driven paradigm. RxT processes each conversational turn as a discrete event in real-time, maintaining context in an integrated, fixed-size Short-Term Memory (STM) system. The architecture features a distinct operational cycle where a generator-decoder produces a response based on the current query and the previous memory state, after which a memory-encoder and a dedicated Memory Attention network asynchronously update the STM with a representation of the complete interaction. This design fundamentally alters the scaling dynamics, reducing the total user-facing cost of a conversation from quadratic ($O(N^2 \cdot T)$) to linear ($O(N \cdot T)$) with respect to the number of interactions $N$. By decoupling response generation from memory updates, RxT achieves low latency, enabling truly real-time, stateful, and economically viable long-form conversations. We validated our architecture with a series of proof-of-concept experiments on synthetic data, demonstrating superior performance and constant-time inference latency compared to a baseline stateless model of comparable size. 

**Abstract (ZH)**: 反应式变压器（RxT）：一种克服大规模语言模型对话限制的新型架构 

---
# GAS-MIL: Group-Aggregative Selection Multi-Instance Learning for Ensemble of Foundation Models in Digital Pathology Image Analysis 

**Title (ZH)**: GAS-MIL：组聚合选择多实例学习在数字病理图像分析中基础模型集成中的应用 

**Authors**: Peiran Quan, Zifan Gu, Zhuo Zhao, Qin Zhou, Donghan M. Yang, Ruichen Rong, Yang Xie, Guanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03555)  

**Abstract**: Foundation models (FMs) have transformed computational pathology by providing powerful, general-purpose feature extractors. However, adapting and benchmarking individual FMs for specific diagnostic tasks is often time-consuming and resource-intensive, especially given their scale and diversity. To address this challenge, we introduce Group-Aggregative Selection Multi-Instance Learning (GAS-MIL), a flexible ensemble framework that seamlessly integrates features from multiple FMs, preserving their complementary strengths without requiring manual feature selection or extensive task-specific fine-tuning. Across classification tasks in three cancer datasets-prostate (PANDA), ovarian (UBC-OCEAN), and breast (TCGA-BrCa)-GAS-MIL consistently achieves superior or on-par performance relative to individual FMs and established MIL methods, demonstrating its robustness and generalizability. By enabling efficient integration of heterogeneous FMs, GAS-MIL streamlines model deployment for pathology and provides a scalable foundation for future multimodal and precision oncology applications. 

**Abstract (ZH)**: Group-Aggregative Selection Multi-Instance Learning for Robust and Generalizable Computational Pathology 

---
# Unmasking Puppeteers: Leveraging Biometric Leakage to Disarm Impersonation in AI-based Videoconferencing 

**Title (ZH)**: 揭露傀儡操控者：利用生物特征泄漏解除AI基于视频会议的冒名顶替威胁 

**Authors**: Danial Samadi Vahdati, Tai Duc Nguyen, Ekta Prashnani, Koki Nagano, David Luebke, Orazio Gallo, Matthew Stamm  

**Link**: [PDF](https://arxiv.org/pdf/2510.03548)  

**Abstract**: AI-based talking-head videoconferencing systems reduce bandwidth by sending a compact pose-expression latent and re-synthesizing RGB at the receiver, but this latent can be puppeteered, letting an attacker hijack a victim's likeness in real time. Because every frame is synthetic, deepfake and synthetic video detectors fail outright. To address this security problem, we exploit a key observation: the pose-expression latent inherently contains biometric information of the driving identity. Therefore, we introduce the first biometric leakage defense without ever looking at the reconstructed RGB video: a pose-conditioned, large-margin contrastive encoder that isolates persistent identity cues inside the transmitted latent while cancelling transient pose and expression. A simple cosine test on this disentangled embedding flags illicit identity swaps as the video is rendered. Our experiments on multiple talking-head generation models show that our method consistently outperforms existing puppeteering defenses, operates in real-time, and shows strong generalization to out-of-distribution scenarios. 

**Abstract (ZH)**: 基于AI的头部动画视频会议系统通过发送紧凑的姿态-表情潜空间并在接收端重新合成RGB图像来降低带宽，但该潜空间可以被操纵，让攻击者实时劫持受害者的 likeness。由于每一帧都是合成的，深度假信息和合成视频检测器完全失效。为解决这一安全问题，我们利用一个关键观察：姿态-表情潜空间固有地包含驱动身份的生物识别信息。因此，我们引入了第一个无需查看重建的RGB视频的生物识别泄漏防御方法：一个姿态条件的大边际对比编码器，该编码器在传输的潜空间中隔离持久的身份线索同时取消暂态的姿态和表情。当视频呈现时，简单余弦测试对此分离嵌入进行检查以标识非法身份交换。我们在多个头部动画生成模型上的实验表明，我们的方法在各个方面均优于现有操纵防御方法，能够在实时运行，并且在跨分布场景中表现出强大的泛化能力。 

---
# Agile Tradespace Exploration for Space Rendezvous Mission Design via Transformers 

**Title (ZH)**: 基于Transformer的敏捷 tradespace 探索方法在空间对接任务设计中的应用 

**Authors**: Yuji Takubo, Daniele Gammelli, Marco Pavone, Simone D'Amico  

**Link**: [PDF](https://arxiv.org/pdf/2510.03544)  

**Abstract**: Spacecraft rendezvous enables on-orbit servicing, debris removal, and crewed docking, forming the foundation for a scalable space economy. Designing such missions requires rapid exploration of the tradespace between control cost and flight time across multiple candidate targets. However, multi-objective optimization in this setting is challenging, as the underlying constraints are often highly nonconvex, and mission designers must balance accuracy (e.g., solving the full problem) with efficiency (e.g., convex relaxations), slowing iteration and limiting design agility. To address these challenges, this paper proposes an AI-powered framework that enables agile mission design for a wide range of Earth orbit rendezvous scenarios. Given the orbital information of the target spacecraft, boundary conditions, and a range of flight times, this work proposes a Transformer-based architecture that generates, in a single parallelized inference step, a set of near-Pareto optimal trajectories across varying flight times, thereby enabling rapid mission trade studies. The model is further extended to accommodate variable flight times and perturbed orbital dynamics, supporting realistic multi-objective trade-offs. Validation on chance-constrained rendezvous problems with passive safety constraints demonstrates that the model generalizes across both flight times and dynamics, consistently providing high-quality initial guesses that converge to superior solutions in fewer iterations. Moreover, the framework efficiently approximates the Pareto front, achieving runtimes comparable to convex relaxation by exploiting parallelized inference. Together, these results position the proposed framework as a practical surrogate for nonconvex trajectory generation and mark an important step toward AI-driven trajectory design for accelerating preliminary mission planning in real-world rendezvous applications. 

**Abstract (ZH)**: 基于AI的航天器交会智能设计框架：支持可扩展太空经济的敏捷任务规划 

---
# TriMediQ: A Triplet-Structured Approach for Interactive Medical Question Answering 

**Title (ZH)**: TriMediQ: 一种三元组结构的交互式医学问答方法 

**Authors**: Zhaohan Meng, Zaiqiao Meng, Siwei Liu, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2510.03536)  

**Abstract**: Large Language Models (LLMs) perform strongly in static and single-turn medical Question Answer (QA) benchmarks, yet such settings diverge from the iterative information gathering process required in practical clinical consultations. The MEDIQ framework addresses this mismatch by recasting the diagnosis as an interactive dialogue between a patient and an expert system, but the reliability of LLMs drops dramatically when forced to reason with dialogue logs, where clinical facts appear in sentences without clear links. To bridge this gap, we introduce TriMediQ, a triplet-structured approach that summarises patient responses into triplets and integrates them into a Knowledge Graph (KG), enabling multi-hop reasoning. We introduce a frozen triplet generator that extracts clinically relevant triplets, using prompts designed to ensure factual consistency. In parallel, a trainable projection module, comprising a graph encoder and a projector, captures relational information from the KG to enhance expert reasoning. TriMediQ operates in two steps: (i) the projection module fine-tuning with all LLM weights frozen; and (ii) using the fine-tuned module to guide multi-hop reasoning during inference. We evaluate TriMediQ on two interactive QA benchmarks, showing that it achieves up to 10.4\% improvement in accuracy over five baselines on the iMedQA dataset. These results demonstrate that converting patient responses into structured triplet-based graphs enables more accurate clinical reasoning in multi-turn settings, providing a solution for the deployment of LLM-based medical assistants. 

**Abstract (ZH)**: TriMediQ： triplet-structured approach for improving multi-hop reasoning in medical dialogue systems 

---
# Identifying Financial Risk Information Using RAG with a Contrastive Insight 

**Title (ZH)**: 使用对比性洞察的RAG识别财务风险信息 

**Authors**: Ali Elahi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03521)  

**Abstract**: In specialized domains, humans often compare new problems against similar examples, highlight nuances, and draw conclusions instead of analyzing information in isolation. When applying reasoning in specialized contexts with LLMs on top of a RAG, the pipeline can capture contextually relevant information, but it is not designed to retrieve comparable cases or related problems.
While RAG is effective at extracting factual information, its outputs in specialized reasoning tasks often remain generic, reflecting broad facts rather than context-specific insights. In finance, it results in generic risks that are true for the majority of companies. To address this limitation, we propose a peer-aware comparative inference layer on top of RAG.
Our contrastive approach outperforms baseline RAG in text generation metrics such as ROUGE and BERTScore in comparison with human-generated equity research and risk. 

**Abstract (ZH)**: 在专业领域，人类通常将新问题与类似例子进行比较，突出细微差别，并得出结论，而不是孤立地分析信息。当在RAG之上应用LLM进行专业上下文中的推理时，流程可以捕捉到上下文相关的信息，但并不能设计用来检索可比案例或相关问题。

虽然RAG在提取事实信息方面很有效，但在专门的推理任务中，其输出往往是通用的，反映了广泛的事实而非具体上下文的洞见。在金融领域，这导致了普遍存在的风险，对大多数公司都适用。为解决这一局限性，我们在RAG之上提出了一个同侪意识的对比推理层。

我们的对比方法在与人工生成的股票研究和风险相比较时，在ROUGE和BERTScore等文本生成指标上优于基线RAG。 

---
# Certifiable Safe RLHF: Fixed-Penalty Constraint Optimization for Safer Language Models 

**Title (ZH)**: 可验证安全的RLHF：固定惩罚约束优化以获得更安全的语言模型 

**Authors**: Kartik Pandit, Sourav Ganguly, Arnesh Banerjee, Shaahin Angizi, Arnob Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2510.03520)  

**Abstract**: Ensuring safety is a foundational requirement for large language models (LLMs). Achieving an appropriate balance between enhancing the utility of model outputs and mitigating their potential for harm is a complex and persistent challenge. Contemporary approaches frequently formalize this problem within the framework of Constrained Markov Decision Processes (CMDPs) and employ established CMDP optimization techniques. However, these methods exhibit two notable limitations. First, their reliance on reward and cost functions renders performance highly sensitive to the underlying scoring mechanism, which must capture semantic meaning rather than being triggered by superficial keywords. Second, CMDP-based training entails tuning dual-variable, a process that is both computationally expensive and does not provide any provable safety guarantee for a fixed dual variable that can be exploitable through adversarial jailbreaks. To overcome these limitations, we introduce Certifiable Safe-RLHF (CS-RLHF) that introduces a cost model trained on a large-scale corpus to assign semantically grounded safety scores. In contrast to the lagrangian-based approach, CS-RLHF adopts a rectified penalty-based formulation. This design draws on the theory of exact penalty functions in constrained optimization, wherein constraint satisfaction is enforced directly through a suitably chosen penalty term. With an appropriately scaled penalty, feasibility of the safety constraints can be guaranteed at the optimizer, eliminating the need for dual-variable updates. Empirical evaluation demonstrates that CS-RLHF outperforms state-of-the-art LLM model responses rendering at-least 5 times efficient against nominal and jail-breaking prompts 

**Abstract (ZH)**: 确保安全是大型语言模型（LLMs）的基本要求。在增强模型输出的有用性和减轻其潜在危害之间找到合适的平衡是一项复杂而持久的挑战。现代方法通常将这一问题形式化为约束马尔可夫决策过程（CMDPs）的框架，并采用现成的CMDP优化技术。然而，这些方法存在两个显著的局限性。首先，它们对奖励和成本函数的依赖使得性能高度依赖于底层的评分机制，该机制必须捕捉语义含义而不是仅仅触发表面关键词。第二，基于CMDP的训练涉及调整双变量的过程，这是一个计算密集的过程，并不能为固定的双变量提供任何可证明的安全保障，而这种双变量可以通过对抗性破解被利用。为克服这些局限性，我们引入了可验证的安全RLHF（CS-RLHF），该方法在大规模语料库上训练了一个成本模型，以分配语义上连贯的安全评分。与基于拉格朗日的方法不同，CS-RLHF采用修正的惩罚形式。此设计借鉴了约束优化中的精确惩罚函数理论，其中约束满足是通过对合适的惩罚项直接施加来实现的。通过适当缩放的惩罚，可以在优化器中保证安全约束的可行性，从而消除调整双变量的需要。实验评价表明，CS-RLHF在应对名义性和破解性提示方面至少比最先进的LLM模型高效5倍。 

---
# TS-Reasoner: Aligning Time Series Foundation Models with LLM Reasoning 

**Title (ZH)**: TS-Reasoner: 将时间序列基础模型与LLM推理对齐 

**Authors**: Fangxu Yu, Hongyu Zhao, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03519)  

**Abstract**: Time series reasoning is crucial to decision-making in diverse domains, including finance, energy usage, traffic, weather, and scientific discovery. While existing time series foundation models (TSFMs) can capture low-level dynamic patterns and provide accurate forecasting, further analysis usually requires additional background knowledge and sophisticated reasoning, which are lacking in most TSFMs but can be achieved through large language models (LLMs). On the other hand, without expensive post-training, LLMs often struggle with the numerical understanding of time series data. Although it is intuitive to integrate the two types of models, developing effective training recipes that align the two modalities for reasoning tasks is still an open challenge. To this end, we propose TS-Reasoner that aligns the latent representations of TSFMs with the textual inputs of LLMs for downstream understanding/reasoning tasks. Specifically, we propose a simple yet effective method to curate diverse, synthetic pairs of time series and textual captions for alignment training. We then develop a two-stage training recipe that applies instruction finetuning after the alignment pretraining. Unlike existing works that train an LLM to take time series as inputs, we leverage a pretrained TSFM and freeze it during training. Extensive experiments on several benchmarks demonstrate that TS-Reasoner not only outperforms a wide range of prevailing LLMs, Vision Language Models (VLMs), and Time Series LLMs, but also achieves this with remarkable data efficiency, e.g., using less than half the training data. 

**Abstract (ZH)**: 时间序列推理对于金融、能源使用、交通、天气以及科学发现等多个领域的决策至关重要。尽管现有的时间序列基础模型（TSFMs）能够捕捉低级动态模式并提供准确的预测，但进一步的分析通常需要额外的背景知识和复杂推理，这些在大多数TSFMs中缺乏，但可以通过大型语言模型（LLMs）实现。另一方面，未经昂贵的后训练，LLMs往往难以理解时间序列数据的数值特性。尽管将两种模型结合起来是直观的，但开发有效的训练方法以使两种模态能够为推理任务对齐仍然是一个开放的挑战。为此，我们提出TS-Reasoner，该方法将TSFMs的潜在表示与LLMs的文本输入对齐，以实现下游的理解/推理任务。具体而言，我们提出了一种简单而有效的方法，用于生成时间序列和文本说明的多样化合成配对，以进行对齐预训练。然后，我们开发了一种两阶段训练方案，在对齐预训练后应用指令微调。与现有工作通过训练LLM将时间序列作为输入不同，我们利用一个预训练的TSFM并在训练过程中将其冻结。在多个基准上的广泛实验表明，TS-Reasoner不仅能够超越广泛流行的LLMs、视觉语言模型（VLMs）和时间序列LLMs，而且还能实现这一点，具有显著的数据效率，例如，使用不到一半的训练数据。 

---
# Red Lines and Grey Zones in the Fog of War: Benchmarking Legal Risk, Moral Harm, and Regional Bias in Large Language Model Military Decision-Making 

**Title (ZH)**: 战争迷雾中的红线与灰色区域：大型语言模型军事决策中的法律风险、道德危害与地区偏见基准研究 

**Authors**: Toby Drinkall  

**Link**: [PDF](https://arxiv.org/pdf/2510.03514)  

**Abstract**: As military organisations consider integrating large language models (LLMs) into command and control (C2) systems for planning and decision support, understanding their behavioural tendencies is critical. This study develops a benchmarking framework for evaluating aspects of legal and moral risk in targeting behaviour by comparing LLMs acting as agents in multi-turn simulated conflict. We introduce four metrics grounded in International Humanitarian Law (IHL) and military doctrine: Civilian Target Rate (CTR) and Dual-use Target Rate (DTR) assess compliance with legal targeting principles, while Mean and Max Simulated Non-combatant Casualty Value (SNCV) quantify tolerance for civilian harm.
We evaluate three frontier models, GPT-4o, Gemini-2.5, and LLaMA-3.1, through 90 multi-agent, multi-turn crisis simulations across three geographic regions. Our findings reveal that off-the-shelf LLMs exhibit concerning and unpredictable targeting behaviour in simulated conflict environments. All models violated the IHL principle of distinction by targeting civilian objects, with breach rates ranging from 16.7% to 66.7%. Harm tolerance escalated through crisis simulations with MeanSNCV increasing from 16.5 in early turns to 27.7 in late turns. Significant inter-model variation emerged: LLaMA-3.1 selected an average of 3.47 civilian strikes per simulation with MeanSNCV of 28.4, while Gemini-2.5 selected 0.90 civilian strikes with MeanSNCV of 17.6. These differences indicate that model selection for deployment constitutes a choice about acceptable legal and moral risk profiles in military operations.
This work seeks to provide a proof-of-concept of potential behavioural risks that could emerge from the use of LLMs in Decision Support Systems (AI DSS) as well as a reproducible benchmarking framework with interpretable metrics for standardising pre-deployment testing. 

**Abstract (ZH)**: 军事组织考虑将大型语言模型（LLMs）整合到指挥与控制（C2）系统中以进行规划和支持决策时，理解其行为倾向至关重要。本研究通过将LLMs作为代理在多轮模拟冲突中进行比较，开发了一种基准评估框架，用于评价目标行为中的法律和道德风险方面。我们引入了四个基于国际人道法（IHL）和军事原则的指标：平民目标率（CTR）和双重用途目标率（DTR）评估其符合合法目标原则的情况，而平均模拟非战斗人员伤亡值（SNCV）和最大模拟非战斗人员伤亡值（SNCV）衡量其对平民伤害的容忍度。 

---
# Platonic Transformers: A Solid Choice For Equivariance 

**Title (ZH)**: 柏拉图变换器：对于等变性而言是一个稳健的选择 

**Authors**: Mohammad Mohaiminul Islam, Rishabh Anand, David R. Wessels, Friso de Kruiff, Thijs P. Kuipers, Rex Ying, Clara I. Sánchez, Sharvaree Vadgama, Georg Bökman, Erik J. Bekkers  

**Link**: [PDF](https://arxiv.org/pdf/2510.03511)  

**Abstract**: While widespread, Transformers lack inductive biases for geometric symmetries common in science and computer vision. Existing equivariant methods often sacrifice the efficiency and flexibility that make Transformers so effective through complex, computationally intensive designs. We introduce the Platonic Transformer to resolve this trade-off. By defining attention relative to reference frames from the Platonic solid symmetry groups, our method induces a principled weight-sharing scheme. This enables combined equivariance to continuous translations and Platonic symmetries, while preserving the exact architecture and computational cost of a standard Transformer. Furthermore, we show that this attention is formally equivalent to a dynamic group convolution, which reveals that the model learns adaptive geometric filters and enables a highly scalable, linear-time convolutional variant. Across diverse benchmarks in computer vision (CIFAR-10), 3D point clouds (ScanObjectNN), and molecular property prediction (QM9, OMol25), the Platonic Transformer achieves competitive performance by leveraging these geometric constraints at no additional cost. 

**Abstract (ZH)**: 尽管变压器在广泛应用，但缺乏对科学和计算机视觉中常见的几何对称性的归纳偏置。现有的一些协变方法往往通过复杂且计算密集的设计来牺牲变压器的高效性和灵活性。我们引入了 platonic transformer 来解决这一权衡。通过将注意力定义为参考框架，基于platonic固体对称群，我们的方法诱导出一种原理上正确的权重共享方案。这使得模型能够在保持标准变压器的精确架构和计算成本的同时，联合具有连续平移和platonic对称的协变性。此外，我们证明这种注意力与动态群卷积形式上等价，揭示了模型学习自适应几何滤波器的能力，并允许实现一种高度可扩展且线性时间的卷积变体。在计算机视觉（CIFAR-10）、3D 点云（ScanObjectNN）和分子性质预测（QM9, OMol25）等多个基准测试中，platonic transformer 通过利用这些几何约束实现了竞争力的性能，而并无额外的成本。 

---
# ALHD: A Large-Scale and Multigenre Benchmark Dataset for Arabic LLM-Generated Text Detection 

**Title (ZH)**: ALHD: 一个大规模多体裁的阿拉伯语LLM生成文本检测基准数据集 

**Authors**: Ali Khairallah, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2510.03502)  

**Abstract**: We introduce ALHD, the first large-scale comprehensive Arabic dataset explicitly designed to distinguish between human- and LLM-generated texts. ALHD spans three genres (news, social media, reviews), covering both MSA and dialectal Arabic, and contains over 400K balanced samples generated by three leading LLMs and originated from multiple human sources, which enables studying generalizability in Arabic LLM-genearted text detection. We provide rigorous preprocessing, rich annotations, and standardized balanced splits to support reproducibility. In addition, we present, analyze and discuss benchmark experiments using our new dataset, in turn identifying gaps and proposing future research directions. Benchmarking across traditional classifiers, BERT-based models, and LLMs (zero-shot and few-shot) demonstrates that fine-tuned BERT models achieve competitive performance, outperforming LLM-based models. Results are however not always consistent, as we observe challenges when generalizing across genres; indeed, models struggle to generalize when they need to deal with unseen patterns in cross-genre settings, and these challenges are particularly prominent when dealing with news articles, where LLM-generated texts resemble human texts in style, which opens up avenues for future research. ALHD establishes a foundation for research related to Arabic LLM-detection and mitigating risks of misinformation, academic dishonesty, and cyber threats. 

**Abstract (ZH)**: ALHD：首个用于区分人类和大规模语言模型生成文本的大规模综合性阿拉伯语数据集 

---
# Real-Time Threaded Houbara Detection and Segmentation for Wildlife Conservation using Mobile Platforms 

**Title (ZH)**: 基于移动平台的实时线程化aho猎隼检测与分割用于野生动物保护 

**Authors**: Lyes Saad Saoud, Loic Lesobre, Enrico Sorato, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2510.03501)  

**Abstract**: Real-time animal detection and segmentation in natural environments are vital for wildlife conservation, enabling non-invasive monitoring through remote camera streams. However, these tasks remain challenging due to limited computational resources and the cryptic appearance of many species. We propose a mobile-optimized two-stage deep learning framework that integrates a Threading Detection Model (TDM) to parallelize YOLOv10-based detection and MobileSAM-based segmentation. Unlike prior YOLO+SAM pipelines, our approach improves real-time performance by reducing latency through threading. YOLOv10 handles detection while MobileSAM performs lightweight segmentation, both executed concurrently for efficient resource use. On the cryptic Houbara Bustard, a conservation-priority species, our model achieves mAP50 of 0.9627, mAP75 of 0.7731, mAP95 of 0.7178, and a MobileSAM mIoU of 0.7421. YOLOv10 operates at 43.7 ms per frame, confirming real-time readiness. We introduce a curated Houbara dataset of 40,000 annotated images to support model training and evaluation across diverse conditions. The code and dataset used in this study are publicly available on GitHub at this https URL. For interactive demos and additional resources, visit this https URL. 

**Abstract (ZH)**: 自然环境中实时动物检测与分割对于野生动物保护至关重要，能通过远程相机流实现非侵入式监测。然而，这些任务由于计算资源有限和许多物种隐蔽的外观而具有挑战性。我们提出了一种针对移动设备优化的两阶段深度学习框架，结合了线程检测模型（TDM）以并行化基于YOLOv10的检测和基于MobileSAM的分割。与之前的YOLO+SAM管道不同，我们的方法通过降低延迟来提高实时性能。YOLOv10负责检测，MobileSAM执行轻量级分割，两者并发执行以高效利用资源。对于隐蔽的厚颈鸨这一保护优先物种，我们的模型实现了mAP50为0.9627、mAP75为0.7731、mAP95为0.7178，并且MobileSAM的mIoU为0.7421。YOLOv10每帧运行时间为43.7毫秒，证实了实时性能。我们提供了一个包含40,000张标注图像的厚颈鸨数据集，以支持模型训练和评估。本文中使用的代码和数据集在GitHub上公开，网址为：这个https URL。有关交互式演示和额外资源，请访问这个https URL。 

---
# AgentHub: A Research Agenda for Agent Sharing Infrastructure 

**Title (ZH)**: AgentHub: 代理共享基础设施的研究议程 

**Authors**: Erik Pautsch, Tanmay Singla, Wenxin Jiang, Huiyun Peng, Behnaz Hassanshahi, Konstantin Läufer, George K.Thiruvathukal, James C. Davis  

**Link**: [PDF](https://arxiv.org/pdf/2510.03495)  

**Abstract**: LLM-based agents are rapidly proliferating, yet the infrastructure for discovering, evaluating, and governing them remains fragmented compared to mature ecosystems like software package registries (e.g., npm) and model hubs (e.g., Hugging Face). Recent research and engineering works have begun to consider the requisite infrastructure, but so far they focus narrowly -- on distribution, naming, or protocol negotiation. However, considering broader software engineering requirements would improve open-source distribution and ease reuse. We therefore propose AgentHub, a research agenda for agent sharing. By framing the key challenges of capability clarity, lifecycle transparency, interoperability, governance, security, and workflow integration, AgentHub charts a community-wide agenda for building reliable and scalable agent ecosystems. Our vision is a future where agents can be shared, trusted, and composed as seamlessly as today's software libraries. 

**Abstract (ZH)**: 基于LLM的代理agent正在迅速增长，然而，发现、评估和治理这些代理的基础设施仍然碎片化，与成熟的生态系统（如软件包注册表npm和模型中心Hugging Face）相比还很不完善。近期的研究和工程工作已经开始考虑所需的基础设施，但迄今仍专注于分布、命名或协议协商。然而，考虑更广泛的软件工程需求将改进开源分发并简化重用。因此，我们提出AgentHub，一项代理共享研究议程。通过界定能力清晰度、生命周期透明度、互操作性、治理、安全性和工作流集成等关键挑战，AgentHub为构建可靠和可扩展的代理生态系统勾画出一个全社区范围的议程。我们的愿景是，在未来，代理可以像今天的软件库一样无缝地共享、信任和组合。 

---
# SEER: The Span-based Emotion Evidence Retrieval Benchmark 

**Title (ZH)**: SEER：基于短语的情感证据检索基准 

**Authors**: Aneesha Sampath, Oya Aran, Emily Mower Provost  

**Link**: [PDF](https://arxiv.org/pdf/2510.03490)  

**Abstract**: We introduce the SEER (Span-based Emotion Evidence Retrieval) Benchmark to test Large Language Models' (LLMs) ability to identify the specific spans of text that express emotion. Unlike traditional emotion recognition tasks that assign a single label to an entire sentence, SEER targets the underexplored task of emotion evidence detection: pinpointing which exact phrases convey emotion. This span-level approach is crucial for applications like empathetic dialogue and clinical support, which need to know how emotion is expressed, not just what the emotion is. SEER includes two tasks: identifying emotion evidence within a single sentence, and identifying evidence across a short passage of five consecutive sentences. It contains new annotations for both emotion and emotion evidence on 1200 real-world sentences. We evaluate 14 open-source LLMs and find that, while some models approach average human performance on single-sentence inputs, their accuracy degrades in longer passages. Our error analysis reveals key failure modes, including overreliance on emotion keywords and false positives in neutral text. 

**Abstract (ZH)**: 基于跨度的情感证据检索基准（SEER）：大型语言模型的情感表达识别能力测试 

---
# Reasoning-based Anomaly Detection Framework: A Real-time, Scalable, and Automated Approach to Anomaly Detection Across Domains 

**Title (ZH)**: 基于推理的异常检测框架：一种跨领域实时、可扩展和自动的异常检测方法 

**Authors**: Anupam Panwar, Himadri Pal, Jiali Chen, Kyle Cho, Riddick Jiang, Miao Zhao, Rajiv Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2510.03486)  

**Abstract**: Detecting anomalies in large, distributed systems presents several challenges. The first challenge arises from the sheer volume of data that needs to be processed. Flagging anomalies in a high-throughput environment calls for a careful consideration of both algorithm and system design. The second challenge comes from the heterogeneity of time-series datasets that leverage such a system in production. In practice, anomaly detection systems are rarely deployed for a single use case. Typically, there are several metrics to monitor, often across several domains (e.g. engineering, business and operations). A one-size-fits-all approach rarely works, so these systems need to be fine-tuned for every application - this is often done manually. The third challenge comes from the fact that determining the root-cause of anomalies in such settings is akin to finding a needle in a haystack. Identifying (in real time) a time-series dataset that is associated causally with the anomalous time-series data is a very difficult problem. In this paper, we describe a unified framework that addresses these challenges. Reasoning based Anomaly Detection Framework (RADF) is designed to perform real time anomaly detection on very large datasets. This framework employs a novel technique (mSelect) that automates the process of algorithm selection and hyper-parameter tuning for each use case. Finally, it incorporates a post-detection capability that allows for faster triaging and root-cause determination. Our extensive experiments demonstrate that RADF, powered by mSelect, surpasses state-of-the-art anomaly detection models in AUC performance for 5 out of 9 public benchmarking datasets. RADF achieved an AUC of over 0.85 for 7 out of 9 datasets, a distinction unmatched by any other state-of-the-art model. 

**Abstract (ZH)**: 在大型分布式系统中检测异常存在的挑战及Reasoning based Anomaly Detection Framework (RADF)框架 

---
# DuPLUS: Dual-Prompt Vision-Language Framework for Universal Medical Image Segmentation and Prognosis 

**Title (ZH)**: DuPLUS: 双提示视觉-语言框架在通用医疗图像分割和预后中的应用 

**Authors**: Numan Saeed, Tausifa Jan Saleem, Fadillah Maani, Muhammad Ridzuan, Hu Wang, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2510.03483)  

**Abstract**: Deep learning for medical imaging is hampered by task-specific models that lack generalizability and prognostic capabilities, while existing 'universal' approaches suffer from simplistic conditioning and poor medical semantic understanding. To address these limitations, we introduce DuPLUS, a deep learning framework for efficient multi-modal medical image analysis. DuPLUS introduces a novel vision-language framework that leverages hierarchical semantic prompts for fine-grained control over the analysis task, a capability absent in prior universal models. To enable extensibility to other medical tasks, it includes a hierarchical, text-controlled architecture driven by a unique dual-prompt mechanism. For segmentation, DuPLUS is able to generalize across three imaging modalities, ten different anatomically various medical datasets, encompassing more than 30 organs and tumor types. It outperforms the state-of-the-art task specific and universal models on 8 out of 10 datasets. We demonstrate extensibility of its text-controlled architecture by seamless integration of electronic health record (EHR) data for prognosis prediction, and on a head and neck cancer dataset, DuPLUS achieved a Concordance Index (CI) of 0.69. Parameter-efficient fine-tuning enables rapid adaptation to new tasks and modalities from varying centers, establishing DuPLUS as a versatile and clinically relevant solution for medical image analysis. The code for this work is made available at: this https URL 

**Abstract (ZH)**: 深度学习在医学影像分析中的应用受限于缺乏泛化能力和预后能力的任务特定模型，而现有的“通用”方法则因简化的条件和较差的医学语义理解而受限。为了解决这些局限性，我们提出了DuPLUS，一种高效的多模态医学影像分析深度学习框架。DuPLUS引入了一种新颖的 vision-language 框架，利用层次语义提示进行细粒度的分析任务控制，这是之前通用模型所缺乏的能力。为了使模型扩展到其他医学任务，它采用了由独特双提示机制驱动的层次化、文本控制架构。对于分割任务，DuPLUS能够在三种成像模态和十个不同解剖学多样化的医学数据集中泛化，覆盖了30多种器官和肿瘤类型。在十个数据集中，DuPLUS在八个数据集上超过了最先进的任务特定和通用模型。我们通过无缝集成电子健康记录（EHR）数据来预测预后，展示了其文本控制架构的扩展性，在头颈部癌症数据集上，DuPLUS的一致性指数（CI）达到了0.69。高效的微调参数使DuPLUS能够快速适应来自不同中心的新任务和模态，确立了其作为医学影像分析中多功能且临床相关解决方案的地位。该工作的代码可在以下链接获取：this https URL。 

---
# Destination-to-Chutes Task Mapping Optimization for Multi-Robot Coordination in Robotic Sorting Systems 

**Title (ZH)**: 多机器人协调在机器人分拣系统中的目标到漏斗任务映射优化 

**Authors**: Yulun Zhang, Alexandre O. G. Barbosa, Federico Pecora, Jiaoyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03472)  

**Abstract**: We study optimizing a destination-to-chutes task mapping to improve throughput in Robotic Sorting Systems (RSS), where a team of robots sort packages on a sortation floor by transporting them from induct workstations to eject chutes based on their shipping destinations (e.g. Los Angeles or Pittsburgh). The destination-to-chutes task mapping is used to determine which chutes a robot can drop its package. Finding a high-quality task mapping is challenging because of the complexity of a real-world RSS. First, optimizing task mapping is interdependent with robot target assignment and path planning. Second, chutes will be CLOSED for a period of time once they receive sufficient packages to allow for downstream processing. Third, task mapping quality directly impacts the downstream processing, as scattered chutes for the same destination increase package handling time. In this paper, we first formally define task mappings and the problem of Task Mapping Optimization (TMO). We then present a simulator of RSS to evaluate task mappings. We then present a simple TMO method based on the Evolutionary Algorithm and Mixed Integer Linear Programming, demonstrating the advantage of our optimized task mappings over the greedily generated ones in various RSS setups with different map sizes, numbers of chutes, and destinations. Finally, we use Quality Diversity algorithms to analyze the throughput of a diverse set of task mappings. Our code is available online at this https URL. 

**Abstract (ZH)**: 我们研究优化目的地到滑槽的任务映射以提高机器人分拣系统(RSS) throughput，其中机器人团队根据包裹的发往目的地（如洛杉矶或匹兹堡）将包裹从接收工作站运输到对应的滑槽。目的地到滑槽的任务映射用于确定机器人可以将包裹投放到哪个滑槽。由于实际-world RSS的复杂性，找到高质量的任务映射具有挑战性。首先，任务映射的优化与机器人目标分配和路径规划相互依赖。其次，滑槽在接收足够多的包裹以允许下游处理后会关闭一段时间。第三，任务映射的质量直接影响下游处理，因为相同目的地的分散滑槽会增加包裹处理时间。在本文中，我们首先正式定义任务映射和任务映射优化问题(TMO)。然后，我们提供了一个RSS仿真器来评估任务映射。接着，基于进化算法和混合整数线性规划提出了一个简单的TMO方法，展示了我们在不同地图规模、滑槽数量和目的地数量的多种RSS设置中，优化的任务映射相较于贪婪生成的任务映射的优势。最后，我们使用质量多样性算法分析多种任务映射的 throughput。我们的代码可在以下链接中在线获得。 

---
# ALMAS: an Autonomous LLM-based Multi-Agent Software Engineering Framework 

**Title (ZH)**: ALMAS：基于自主LLM的多agent软件工程框架 

**Authors**: Vali Tawosi, Keshav Ramani, Salwa Alamir, Xiaomo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03463)  

**Abstract**: Multi-agent Large Language Model (LLM) systems have been leading the way in applied LLM research across a number of fields. One notable area is software development, where researchers have advanced the automation of code implementation, code testing, code maintenance, inter alia, using LLM agents. However, software development is a multifaceted environment that extends beyond just code. As such, a successful LLM system must factor in multiple stages of the software development life-cycle (SDLC). In this paper, we propose a vision for ALMAS, an Autonomous LLM-based Multi-Agent Software Engineering framework, which follows the above SDLC philosophy such that it may work within an agile software development team to perform several tasks end-to-end. ALMAS aligns its agents with agile roles, and can be used in a modular fashion to seamlessly integrate with human developers and their development environment. We showcase the progress towards ALMAS through our published works and a use case demonstrating the framework, where ALMAS is able to seamlessly generate an application and add a new feature. 

**Abstract (ZH)**: 多智能体大规模语言模型（LLM）系统已在多个领域的应用LLM研究中引领潮流。一个值得关注的领域是软件开发，研究人员利用LLM代理实现了代码实现、代码测试、代码维护等自动化。然而，软件开发不仅仅局限于代码，还涵盖了多个方面。因此，一个成功的LLM系统必须考虑软件开发生命周期（SDLC）的多个阶段。本文提出ALMAS（自主LLM基础多智能体软件工程框架）的愿景，该框架遵循上述SDLC理念，能够在敏捷软件开发团队中端到端执行多项任务。ALMAS的智能体与敏捷角色相匹配，并可模块化地无缝集成到人类开发者及其开发环境中。我们通过已发表的工作和一个使用案例展示了ALMAS的发展，其中ALMAS能够无缝生成应用程序并添加新功能。 

---
# The Argument is the Explanation: Structured Argumentation for Trust in Agents 

**Title (ZH)**: 论据即解释：代理信任的结构化论证 

**Authors**: Ege Cakar, Per Ola Kristensson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03442)  

**Abstract**: Humans are black boxes -- we cannot observe their neural processes, yet society functions by evaluating verifiable arguments. AI explainability should follow this principle: stakeholders need verifiable reasoning chains, not mechanistic transparency. We propose using structured argumentation to provide a level of explanation and verification neither interpretability nor LLM-generated explanation is able to offer. Our pipeline achieves state-of-the-art 94.44 macro F1 on the AAEC published train/test split (5.7 points above prior work) and $0.81$ macro F1, $\sim$0.07 above previous published results with comparable data setups, for Argumentative MicroTexts relation classification, converting LLM text into argument graphs and enabling verification at each inferential step. We demonstrate this idea on multi-agent risk assessment using the Structured What-If Technique, where specialized agents collaborate transparently to carry out risk assessment otherwise achieved by humans alone. Using Bipolar Assumption-Based Argumentation, we capture support/attack relationships, thereby enabling automatic hallucination detection via fact nodes attacking arguments. We also provide a verification mechanism that enables iterative refinement through test-time feedback without retraining. For easy deployment, we provide a Docker container for the fine-tuned AMT model, and the rest of the code with the Bipolar ABA Python package on GitHub. 

**Abstract (ZH)**: 人类是黑盒——我们无法观察其神经过程，但社会通过评估可验证的论据而运行。AI可解释性应遵循这一原则：利益相关者需要可验证的推理链，而非机械透明性。我们提议使用结构化论证来提供一种解释和验证水平，这既不是可解释性所能提供的，也不是由LLM生成的解释所能提供的。我们的管道在AAEC发布的训练/测试分割上实现了最先进的94.44宏F1（比先前工作高5.7分点），并在使用可比拟数据集的情况下，实现了0.81的宏F1，比之前发布的结果高约0.07，用于论辩微文本关系分类，将LLM文本转换为论据图，并在每个推理步骤中实现验证。我们在使用结构化what-if技术的多智能体风险评估中演示这一理念，其中专业智能体协作透明地进行风险评估，这是以往仅由人类单独完成的工作。我们使用双极假设基于论证来捕获支持/攻击关系，从而通过事实节点攻击论证实现自动幻觉检测。我们还提供了一种验证机制，通过测试时反馈实现逐步细化，无需重新训练。为了便于部署，我们提供了一个针对精细调校的AMT模型的Docker容器，并在GitHub上提供了使用双极ABA Python包的其余代码。 

---
# Spatial-ViLT: Enhancing Visual Spatial Reasoning through Multi-Task Learning 

**Title (ZH)**: 基于多任务学习增强视觉空间推理：Spatial-ViLT 

**Authors**: Chashi Mahiul Islam, Oteo Mamo, Samuel Jacob Chacko, Xiuwen Liu, Weikuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03441)  

**Abstract**: Vision-language models (VLMs) have advanced multimodal reasoning but still face challenges in spatial reasoning for 3D scenes and complex object configurations. To address this, we introduce SpatialViLT, an enhanced VLM that integrates spatial features like depth maps, 3D coordinates, and edge maps through a multi-task learning framework. This approach enriches multimodal embeddings with spatial understanding. We propose two variants: SpatialViLT and MaskedSpatialViLT, focusing on full and masked object regions, respectively. Additionally, SpatialEnsemble combines both approaches, achieving state-of-the-art accuracy. Our models excel in spatial reasoning categories such as directional, topological, and proximity relations, as demonstrated on the challenging Visual Spatial Reasoning (VSR) dataset. This work represents a significant step in enhancing the spatial intelligence of AI systems, crucial for advanced multimodal understanding and real-world applications. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在多模态推理方面取得了进步，但仍面临3D场景和复杂对象配置的空间推理挑战。为解决这一问题，我们引入了SpatialViLT，这是一种通过多任务学习框架整合深度图、3D坐标和边缘图等空间特征的增强VLM。该方法通过增加空间理解来丰富多模态嵌入。我们提出了两种变体：SpatialViLT和MaskedSpatialViLT，分别侧重于完整对象区域和遮罩对象区域。此外，SpatialEnsemble结合了这两种方法，实现了最先进的准确率。我们的模型在方向性、拓扑关系和接近关系等空间推理类别方面表现出色，如在具有挑战性的Visual Spatial Reasoning (VSR) 数据集上所展示的。这项工作代表了增强AI系统空间智能的重要步骤，对于高级多模态理解和现实世界应用至关重要。 

---
# Scalable Ground Station Selection for Large LEO Constellations 

**Title (ZH)**: 大规模低轨星座地面站可扩展选择 

**Authors**: Grace Ra Kim, Duncan Eddy, Vedant Srinivas, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2510.03438)  

**Abstract**: Effective ground station selection is critical for low Earth orbiting (LEO) satellite constellations to minimize operational costs, maximize data downlink volume, and reduce communication gaps between access windows. Traditional ground station selection typically begins by choosing from a fixed set of locations offered by Ground Station-as-a-Service (GSaaS) providers, which helps reduce the problem scope to optimizing locations over existing infrastructure. However, finding a globally optimal solution for stations using existing mixed-integer programming methods quickly becomes intractable at scale, especially when considering multiple providers and large satellite constellations. To address this issue, we introduce a scalable, hierarchical framework that decomposes the global selection problem into single-satellite, short time-window subproblems. Optimal station choices from each subproblem are clustered to identify consistently high-value locations across all decomposed cases. Cluster-level sets are then matched back to the closest GSaaS candidate sites to produce a globally feasible solution. This approach enables scalable coordination while maintaining near-optimal performance. We evaluate our method's performance on synthetic Walker-Star test cases (1-10 satellites, 1-10 stations), achieving solutions within 95% of the global IP optimum for all test cases. Real-world evaluations on Capella Space (5 satellites), ICEYE (40), and Planet's Flock (96) show that while exact IP solutions fail to scale, our framework continues to deliver high-quality site selections. 

**Abstract (ZH)**: 有效的地面站选择对于低地球轨道（LEO）卫星星座降低运营成本、最大化数据下行量并减少访问窗口间的通信间隙至关重要。传统的地面站选择通常从地面站即服务（GSaaS）提供商提供的固定位置集合中选择，从而将问题范围缩小为在现有基础设施上优化位置。然而，当考虑多个提供商和大型卫星星座时，使用现有混合整数规划方法寻找站址的全局最优解很快变得难以处理。为解决这一问题，我们提出了一种可扩展的分层框架，该框架将全局选择问题分解为单卫星、短时间窗口子问题。来自每个子问题的最优站址选择被聚类，以识别所有分解情况下的一致性高价值位置。然后将聚类集与最近的GSaaS候选站点匹配，以生成全局可行解。该方法能够在保持接近最优性能的同时实现可扩展的协调。我们在合成Walker-Star测试案例（1-10颗卫星，1-10个地面站）上评估了方法性能，所有测试案例中的解决方案均接近全局IP最优解的95%。在Capella Space（5颗卫星）、ICEYE（40颗）和Planet的Flock（96颗）的真实世界评估中，虽然精确的IP解决方案无法扩展，但我们的框架继续提供高质量的站点选择。 

---
# Application of a Virtual Imaging Framework for Investigating a Deep Learning-Based Reconstruction Method for 3D Quantitative Photoacoustic Computed Tomography 

**Title (ZH)**: 基于虚拟成像框架的深度学习重建方法用于三维定量光声计算机断层成像的研究 

**Authors**: Refik Mert Cam, Seonyeong Park, Umberto Villa, Mark A. Anastasio  

**Link**: [PDF](https://arxiv.org/pdf/2510.03431)  

**Abstract**: Quantitative photoacoustic computed tomography (qPACT) is a promising imaging modality for estimating physiological parameters such as blood oxygen saturation. However, developing robust qPACT reconstruction methods remains challenging due to computational demands, modeling difficulties, and experimental uncertainties. Learning-based methods have been proposed to address these issues but remain largely unvalidated. Virtual imaging (VI) studies are essential for validating such methods early in development, before proceeding to less-controlled phantom or in vivo studies. Effective VI studies must employ ensembles of stochastically generated numerical phantoms that accurately reflect relevant anatomy and physiology. Yet, most prior VI studies for qPACT relied on overly simplified phantoms. In this work, a realistic VI testbed is employed for the first time to assess a representative 3D learning-based qPACT reconstruction method for breast imaging. The method is evaluated across subject variability and physical factors such as measurement noise and acoustic aberrations, offering insights into its strengths and limitations. 

**Abstract (ZH)**: 基于学习的三维定量光声计算机断层成像方法在乳房成像中的虚拟影像测试与评估 

---
# Generalized Orders of Magnitude for Scalable, Parallel, High-Dynamic-Range Computation 

**Title (ZH)**: 可扩展、并行、高动态范围计算的一般量级秩序 

**Authors**: Franz A. Heinsen, Leo Kozachkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.03426)  

**Abstract**: Many domains, from deep learning to finance, require compounding real numbers over long sequences, often leading to catastrophic numerical underflow or overflow. We introduce generalized orders of magnitude (GOOMs), a principled extension of traditional orders of magnitude that incorporates floating-point numbers as a special case, and which in practice enables stable computation over significantly larger dynamic ranges of real numbers than previously possible. We implement GOOMs, along with an efficient custom parallel prefix scan, to support native execution on parallel hardware such as GPUs. We demonstrate that our implementation of GOOMs outperforms traditional approaches with three representative experiments, all of which were previously considered impractical or impossible, and now become possible and practical: (1) compounding real matrix products far beyond standard floating-point limits; (2) estimating spectra of Lyapunov exponents in parallel, orders of magnitude faster than with previous methods, applying a novel selective-resetting method to prevent state colinearity; and (3) capturing long-range dependencies in deep recurrent neural networks with non-diagonal recurrent states, computed in parallel via a prefix scan, without requiring any form of stabilization. Our results show that our implementation of GOOMs, combined with efficient parallel scanning, offers a scalable and numerically robust alternative to conventional floating-point numbers for high-dynamic-range applications. 

**Abstract (ZH)**: GOOMs及其在大规模动态范围计算中的应用：超越传统浮点数的稳健计算方法 

---
# Multi-task neural diffusion processes for uncertainty-quantified wind power prediction 

**Title (ZH)**: 多任务神经扩散过程在不确定性量化风电预测中的应用 

**Authors**: Joseph Rawson, Domniki Ladopoulou, Petros Dellaportas  

**Link**: [PDF](https://arxiv.org/pdf/2510.03419)  

**Abstract**: Uncertainty-aware wind power prediction is essential for grid integration and reliable wind farm operation. We apply neural diffusion processes (NDPs)-a recent class of models that learn distributions over functions-and extend them to a multi-task NDP (MT-NDP) framework for wind power prediction. We provide the first empirical evaluation of NDPs in real supervisory control and data acquisition (SCADA) data. We introduce a task encoder within MT-NDPs to capture cross-turbine correlations and enable few-shot adaptation to unseen turbines. The proposed MT-NDP framework outperforms single-task NDPs and GPs in terms of point accuracy and calibration, particularly for wind turbines whose behaviour deviates from the fleet average. In general, NDP-based models deliver calibrated and scalable predictions suitable for operational deployment, offering sharper, yet trustworthy, predictive intervals that can support dispatch and maintenance decisions in modern wind farms. 

**Abstract (ZH)**: 不确定性感知的风功率预测对于电网集成和可靠的风电场运行至关重要。我们应用神经扩散过程（NDPs）——一种最近发展起来的模型类，能够学习函数的分布——并将其扩展到多任务NDP（MT-NDP）框架用于风功率预测。我们在真实的监督控制和数据采集（SCADA）数据上提供了NDPs的首次实证评估。我们引入了多任务NDP中的任务编码器来捕获跨风力发电机的相关性，并实现对未见过的风力发电机的少量样本适应。提出的MT-NDP框架在点准确度和校准方面优于单任务NDP和高斯过程（GPs），特别是在风力发电机行为偏离机组平均值的情况下。总体而言，基于NDP的模型提供了一种校准且可扩展的预测，适合于运行部署，能够提供更锐利但可靠的预测区间，从而支持现代风电场的调度和维护决策。 

---
# NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks 

**Title (ZH)**: NEXUS：网络探索以利用多轮LLM戛然中断中的不安全序列 

**Authors**: Javad Rafiei Asl, Sidhant Narula, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03417)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）虽革命性地推动了自然语言处理的发展，但依然易受 Jailbreak 攻击，尤其是多轮次 Jailbreak 攻击，这种攻击将恶意意图分散在看似无害的对话中，从而规避了对齐机制。现有方法往往在探索 adversarial 空间方面表现不佳，依赖于手工制作的经验法则，或缺乏系统化的查询优化。我们提出了 NEXUS（Network Exploration for eXploiting Unsafe Sequences），一个模块化的框架，用于构建、优化和执行高效的多轮次攻击。NEXUS 包括：(1) ThoughtNet，通过层次化的扩展将负面意图构建为包含主题、实体和查询链的结构化语义网络；(2) 一种基于反馈的模拟器，在攻击者-受害者-审判者 LLM 合作中通过有害性和语义相似性基准迭代优化和精简这些查询链；和 (3) 一个网络导航器，能够适应性地导航优化后的查询空间以进行实时攻击。该流水线揭示了跨 LLM 的隐蔽且成功率高的对抗路径。在多个闭源和开源 LLM 上，与现有方法相比，NEXUS 将攻击成功率提高了 2.1% 到 19.4%。代码：this https URL。 

---
# PLSEMANTICSBENCH: Large Language Models As Programming Language Interpreters 

**Title (ZH)**: PLSEMANTICSBENCH：大型语言模型作为编程语言解释器 

**Authors**: Aditya Thimmaiah, Jiyang Zhang, Jayanth Srinivasa, Junyi Jessy Li, Milos Gligoric  

**Link**: [PDF](https://arxiv.org/pdf/2510.03415)  

**Abstract**: As large language models (LLMs) excel at code reasoning, a natural question arises: can an LLM execute programs (i.e., act as an interpreter) purely based on a programming language's formal semantics? If so, it will enable rapid prototyping of new programming languages and language features. We study this question using the imperative language IMP (a subset of C), formalized via small-step operational semantics (SOS) and rewriting-based operational semantics (K-semantics). We introduce three evaluation sets-Human-Written, LLM-Translated, and Fuzzer- Generated-whose difficulty is controlled by code-complexity metrics spanning the size, control-flow, and data-flow axes. Given a program and its semantics formalized with SOS/K-semantics, models are evaluated on three tasks ranging from coarse to fine: (1) final-state prediction, (2) semantic rule prediction, and (3) execution trace prediction. To distinguish pretraining memorization from semantic competence, we define two nonstandard semantics obtained through systematic mutations of the standard rules. Across strong code/reasoning LLMs, performance drops under nonstandard semantics despite high performance under the standard one. We further find that (i) there are patterns to different model failures, (ii) most reasoning models perform exceptionally well on coarse grained tasks involving reasoning about highly complex programs often containing nested loop depths beyond five, and surprisingly, (iii) providing formal semantics helps on simple programs but often hurts on more complex ones. Overall, the results show a promise that LLMs could serve as programming language interpreters, but points to the lack of their robust semantics understanding. We release the benchmark and the supporting code at this https URL. 

**Abstract (ZH)**: 大型语言模型在代码推理方面表现出色，自然地引出一个疑问：一个大型语言模型能否仅基于编程语言的形式语义来执行程序（即充当解释器的角色）？如果可以，这将使新编程语言和语言特性的快速原型设计成为可能。我们通过 imperative 语言 IMP（C 语言的子集），借助小步操作语义（SOS）和基于重写的操作语义（K-语义）来研究这一问题。我们引入了三个评估集——由人工编写、LLM 转换和模糊生成的程序，其难度通过代码复杂性指标（涵盖大小、控制流和数据流轴）来控制。给定一个程序及其用 SOS/K-语义形式化的语义，模型将被评价三个范围从粗到细的任务：（1）最终状态预测，（2）语义规则预测，（3）执行轨迹预测。为了区分预训练记忆和语义能力，我们定义了两种非标准语义，通过系统地变异标准规则获得。在强代码/推理 LLM 中，尽管在标准语义下性能很高，但在非标准语义下性能下降。进一步发现：（i）不同模型失败的模式存在；（ii）大多数推理模型在涉及复杂程序的粗粒度任务中表现优异，令人惊讶的是，（iii）提供形式语义在简单程序中有所帮助，但在更复杂的程序中却经常有害。总体而言，结果展示了大型语言模型可能作为编程语言解释器的希望，但也指出了它们缺乏稳健的语义理解。我们在此 https:// 这里发布基准测试和配套代码。 

---
# Report of the 2025 Workshop on Next-Generation Ecosystems for Scientific Computing: Harnessing Community, Software, and AI for Cross-Disciplinary Team Science 

**Title (ZH)**: 2025代数生态系统研讨会报告：利用社区、软件和AI促进跨学科团队科学 

**Authors**: L.C. McInnes, D. Arnold, P. Balaprakash, M. Bernhardt, B. Cerny, A. Dubey, R. Giles, D.W. Hood, M.A. Leung, V. Lopez-Marrero, P. Messina, O.B. Newton, C. Oehmen, S.M. Wild, J. Willenbring, L. Woodley, T. Baylis, D.E. Bernholdt, C. Camano, J. Cohoon, C. Ferenbaugh, S.M. Fiore, S. Gesing, D. Gomez-Zara, J. Howison, T. Islam, D. Kepczynski, C. Lively, H. Menon, B. Messer, M. Ngom, U. Paliath, M.E. Papka, I. Qualters, E.M. Raybourn, K. Riley, P. Rodriguez, D. Rouson, M. Schwalbe, S.K. Seal, O. Surer, V. Taylor, L. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03413)  

**Abstract**: This report summarizes insights from the 2025 Workshop on Next-Generation Ecosystems for Scientific Computing: Harnessing Community, Software, and AI for Cross-Disciplinary Team Science, which convened more than 40 experts from national laboratories, academia, industry, and community organizations to chart a path toward more powerful, sustainable, and collaborative scientific software ecosystems. To address urgent challenges at the intersection of high-performance computing (HPC), AI, and scientific software, participants envisioned agile, robust ecosystems built through socio-technical co-design--the intentional integration of social and technical components as interdependent parts of a unified strategy. This approach combines advances in AI, HPC, and software with new models for cross-disciplinary collaboration, training, and workforce development. Key recommendations include building modular, trustworthy AI-enabled scientific software systems; enabling scientific teams to integrate AI systems into their workflows while preserving human creativity, trust, and scientific rigor; and creating innovative training pipelines that keep pace with rapid technological change. Pilot projects were identified as near-term catalysts, with initial priorities focused on hybrid AI/HPC infrastructure, cross-disciplinary collaboration and pedagogy, responsible AI guidelines, and prototyping of public-private partnerships. This report presents a vision of next-generation ecosystems for scientific computing where AI, software, hardware, and human expertise are interwoven to drive discovery, expand access, strengthen the workforce, and accelerate scientific progress. 

**Abstract (ZH)**: This报告总结了2025下一代科学计算生态系统研讨会的见解：利用社区、软件和AI推动跨学科团队科学，该研讨会汇聚了来自国家实验室、学术界、工业界和社区组织的逾40位专家，以规划更强大、更可持续和更具合作性的科学软件生态系统之路。为应对高性能计算（HPC）、AI与科学软件交汇处的紧迫挑战，参与者设想了通过社会-技术共设计构建灵活且稳健的生态系统——故意将社会和技术组件作为统一策略的相互依存部分进行集成。这种方法结合了AI、HPC和软件的最新进展，以及跨学科合作、培训和劳动力发展的新模式。关键建议包括构建模块化、可信赖的AI辅助科学软件系统；使科学团队能够将AI系统集成到其工作流程中，同时保留人类的创造力、信任和科学严谨性；并创建与快速技术变革保持同步的创新培训管道。试点项目被确定为近期催化剂，初期优先事项集中在混合AI/HPC基础设施、跨学科协作与教学、负责任的AI规范以及公共-私营合作伙伴关系原型设计上。本报告提出了下一代科学计算生态系统的愿景，在该生态系统中，AI、软件、硬件和人类专长相互交织，以推动发现、扩大获取、强化劳动力并加速科学进步。 

---
# LegalSim: Multi-Agent Simulation of Legal Systems for Discovering Procedural Exploits 

**Title (ZH)**: LegalSim: 多智能体模拟法律系统的流程性利用发现 

**Authors**: Sanket Badhe  

**Link**: [PDF](https://arxiv.org/pdf/2510.03405)  

**Abstract**: We present LegalSim, a modular multi-agent simulation of adversarial legal proceedings that explores how AI systems can exploit procedural weaknesses in codified rules. Plaintiff and defendant agents choose from a constrained action space (for example, discovery requests, motions, meet-and-confer, sanctions) governed by a JSON rules engine, while a stochastic judge model with calibrated grant rates, cost allocations, and sanction tendencies resolves outcomes. We compare four policies: PPO, a contextual bandit with an LLM, a direct LLM policy, and a hand-crafted heuristic; Instead of optimizing binary case outcomes, agents are trained and evaluated using effective win rate and a composite exploit score that combines opponent-cost inflation, calendar pressure, settlement pressure at low merit, and a rule-compliance margin. Across configurable regimes (e.g., bankruptcy stays, inter partes review, tax procedures) and heterogeneous judges, we observe emergent ``exploit chains'', such as cost-inflating discovery sequences and calendar-pressure tactics that remain procedurally valid yet systemically harmful. Evaluation via cross-play and Bradley-Terry ratings shows, PPO wins more often, the bandit is the most consistently competitive across opponents, the LLM trails them, and the heuristic is weakest. The results are stable in judge settings, and the simulation reveals emergent exploit chains, motivating red-teaming of legal rule systems in addition to model-level testing. 

**Abstract (ZH)**: LegalSim：一个多模块代理模拟的 adversarial 法律程序，探索 AI 系统如何利用成文规则中的程序漏洞 

---
# Implicit Values Embedded in How Humans and LLMs Complete Subjective Everyday Tasks 

**Title (ZH)**: 人类和LLMs在完成日常主观任务时嵌入的隐性价值观 

**Authors**: Arjun Arunasalam, Madison Pickering, Z. Berkay Celik, Blase Ur  

**Link**: [PDF](https://arxiv.org/pdf/2510.03384)  

**Abstract**: Large language models (LLMs) can underpin AI assistants that help users with everyday tasks, such as by making recommendations or performing basic computation. Despite AI assistants' promise, little is known about the implicit values these assistants display while completing subjective everyday tasks. Humans may consider values like environmentalism, charity, and diversity. To what extent do LLMs exhibit these values in completing everyday tasks? How do they compare with humans? We answer these questions by auditing how six popular LLMs complete 30 everyday tasks, comparing LLMs to each other and to 100 human crowdworkers from the US. We find LLMs often do not align with humans, nor with other LLMs, in the implicit values exhibited. 

**Abstract (ZH)**: 大型语言模型(LLMs)可以支撑辅助用户完成日常任务的AI助手，比如提供推荐或进行基本计算。尽管AI助手潜力巨大，但对其在完成主观日常任务时隐含的价值观展示知之甚少。人类可能考虑环境主义、慈善和多样性等价值观。在完成日常任务时，LLMs在多大程度上表现出这些价值观？它们与人类相比有何不同？我们通过审计六种流行的LLMs完成30项日常任务的过程，将LLMs相互比较，并与100名来自美国的人类众包工作者进行比较，以回答这些问题。我们发现，LLMs在展示隐含价值观方面常常与人类和其它LLMs不符。 

---
# Cross-Modal Reconstruction Pretraining for Ramp Flow Prediction at Highway Interchanges 

**Title (ZH)**: 高速公路互通处坡度流预测的跨模态重建预训练 

**Authors**: Yongchao Li, Jun Chen, Zhuoxuan Li, Chao Gao, Yang Li, Chu Zhang, Changyin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.03381)  

**Abstract**: Interchanges are crucial nodes for vehicle transfers between highways, yet the lack of real-time ramp detectors creates blind spots in traffic prediction. To address this, we propose a Spatio-Temporal Decoupled Autoencoder (STDAE), a two-stage framework that leverages cross-modal reconstruction pretraining. In the first stage, STDAE reconstructs historical ramp flows from mainline data, forcing the model to capture intrinsic spatio-temporal relations. Its decoupled architecture with parallel spatial and temporal autoencoders efficiently extracts heterogeneous features. In the prediction stage, the learned representations are integrated with models such as GWNet to enhance accuracy. Experiments on three real-world interchange datasets show that STDAE-GWNET consistently outperforms thirteen state-of-the-art baselines and achieves performance comparable to models using historical ramp data. This demonstrates its effectiveness in overcoming detector scarcity and its plug-and-play potential for diverse forecasting pipelines. 

**Abstract (ZH)**: 时空解耦自编码器（STDAE）及其在环线流量预测中的应用 

---
# A Robust Clustered Federated Learning Approach for Non-IID Data with Quantity Skew 

**Title (ZH)**: 一种针对数量偏差非一致数据的健壮聚类联邦学习方法 

**Authors**: Michael Ben Ali, Imen Megdiche, André Peninou, Olivier Teste  

**Link**: [PDF](https://arxiv.org/pdf/2510.03380)  

**Abstract**: Federated Learning (FL) is a decentralized paradigm that enables a client-server architecture to collaboratively train a global Artificial Intelligence model without sharing raw data, thereby preserving privacy. A key challenge in FL is Non-IID data. Quantity Skew (QS) is a particular problem of Non-IID, where clients hold highly heterogeneous data volumes. Clustered Federated Learning (CFL) is an emergent variant of FL that presents a promising solution to Non-IID problem. It improves models' performance by grouping clients with similar data distributions into clusters. CFL methods generally fall into two operating strategies. In the first strategy, clients select the cluster that minimizes the local training loss. In the second strategy, the server groups clients based on local model similarities. However, most CFL methods lack systematic evaluation under QS but present significant challenges because of it.  In this paper, we present two main contributions. The first one is an evaluation of state-of-the-art CFL algorithms under various Non-IID settings, applying multiple QS scenarios to assess their robustness. Our second contribution is a novel iterative CFL algorithm, named CORNFLQS, which proposes an optimal coordination between both operating strategies of CFL. Our approach is robust against the different variations of QS settings. We conducted intensive experiments on six image classification datasets, resulting in 270 Non-IID configurations. The results show that CORNFLQS achieves the highest average ranking in both accuracy and clustering quality, as well as strong robustness to QS perturbations. Overall, our approach outperforms actual CFL algorithms. 

**Abstract (ZH)**: 联邦学习（FL）是一种分散式范式，使得客户端-服务器架构能够在不共享原始数据的情况下协作训练全球人工智能模型，从而保护隐私。FL中一个关键挑战是非IID数据。数据量偏差（Quantity Skew，QS）是特定类型的非IID问题，其中客户端持有的数据量高度异质。集群联邦学习（Clustered Federated Learning，CFL）是一种新兴的FL变体，为非IID问题提供了有前途的解决方案。它通过将具有类似数据分布的客户端分组到簇中来提高模型性能。CFL方法通常分为两种操作策略。在第一种策略中，客户端选择能最小化局部训练损失的簇。在第二种策略中，服务器根据局部模型相似性对客户端进行分组。然而，大多数CFL方法缺乏在QS下系统的评估，但这些问题却是它们面临的重大挑战。在本文中，我们提出两大主要贡献。首先是评估在各种非IID设置下的先进CFL算法，并应用多种QS场景来评估其鲁棒性。我们的第二个贡献是提出了一种新的迭代CFL算法，名为CORNFLQS，该算法在两种CFL操作策略之间提出了最优协调。我们的方法在不同版本的QS设置下表现出鲁棒性。我们在六个图像分类数据集上进行了密集的实验，得到了270种非IID配置。结果表明，CORNFLQS在准确率和聚类质量上均获得最高平均排名，并且在QS扰动下表现出强大的鲁棒性。总的来说，我们的方法优于实际的CFL算法。 

---
# Can an AI-Powered Presentation Platform Based On The Game "Just a Minute" Be Used To Improve Students' Public Speaking Skills? 

**Title (ZH)**: 基于“一分钟”游戏的AIpowered演示平台能否提高学生公共演讲技能？ 

**Authors**: Frederic Higham, Tommy Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03379)  

**Abstract**: This study explores the effectiveness of applying AI and gamification into a presentation platform aimed at University students wanting to improve their public speaking skills in their native tongue. Specifically, a platform based on the radio show, Just a Minute (JAM), is explored. In this game, players are challenged to speak fluently on a topic for 60 seconds without repeating themselves, hesitating or deviating from the topic. JAM has proposed benefits such as allowing students to improve their spontaneous speaking skills and reduce their use of speech disfluencies ("um", "uh", etc.).
Previous research has highlighted the difficulties students face when speaking publicly, the main one being anxiety. AI Powered Presentation Platforms (AI-PPPs), where students can speak with an immersive AI audience and receive real-time feedback, have been explored as a method to improve student's speaking skills and confidence. So far they have shown promising results which this study aims to build upon.
A group of students from the University of York are enlisted to evaluate the effectiveness of the JAM platform. They are asked to fill in a questionnaire, play through the game twice and then complete a final questionnaire to discuss their experiences playing the game. Various statistics are gathered during their gameplay such as the number of points they gained and the number of rules they broke. The results showed that students found the game promising and believed that their speaking skills could improve if they played the game for longer. More work will need to be carried out to prove the effectiveness of the game beyond the short term. 

**Abstract (ZH)**: 本研究探讨将AI和游戏化应用于旨在帮助大学学生提高母语公共演讲能力的プレゼンテーションプラットフォーム的有效性。特别是在此研究中，探讨了基于广播节目“Just a Minute”（JAM）的平台。在这个游戏中，玩家被挑战在60秒内流畅地讨论一个话题，不重复、不犹豫且不偏离话题。JAM 提出了诸如允许学生提高即兴演讲能力、减少言语不流畅（如“嗯”、“哦”等）的好处。之前的研究表明，学生在公开演讲时面临的最大困难是焦虑。带有沉浸式AI观众的学生开口演讲平台（AI-PPPs），并能够即时获得反馈的方法，被探索以提高学生的演讲能力和自信心。迄今为止，这些平台已经展示了令人鼓舞的结果，本研究旨在在此基础上进一步提升。来自约克大学的一组学生被招募来评估JAM平台的有效性。他们被要求填写问卷、两次游戏并完成最终问卷来讨论他们对游戏的体验。在游戏过程中收集了包括他们获得的分数和违反的规则次数等各种统计数据。结果显示，学生认为该游戏具有前景，并相信若长时间游戏，他们的演讲能力能够提高。未来还需进一步工作以证明该游戏在短期之外的有效性。 

---
# Lightweight Prompt Engineering for Cognitive Alignment in Educational AI: A OneClickQuiz Case Study 

**Title (ZH)**: 轻量级提示工程在教育AI中的认知对齐：OneClickQuiz案例研究 

**Authors**: Antoun Yaacoub, Zainab Assaghir, Jérôme Da-Rugna  

**Link**: [PDF](https://arxiv.org/pdf/2510.03374)  

**Abstract**: The rapid integration of Artificial Intelligence (AI) into educational technology promises to revolutionize content creation and assessment. However, the quality and pedagogical alignment of AI-generated content remain critical challenges. This paper investigates the impact of lightweight prompt engineering strategies on the cognitive alignment of AI-generated questions within OneClickQuiz, a Moodle plugin leveraging generative AI. We evaluate three prompt variants-a detailed baseline, a simpler version, and a persona-based approach-across Knowledge, Application, and Analysis levels of Bloom's Taxonomy. Utilizing an automated classification model (from prior work) and human review, our findings demonstrate that explicit, detailed prompts are crucial for precise cognitive alignment. While simpler and persona-based prompts yield clear and relevant questions, they frequently misalign with intended Bloom's levels, generating outputs that are either too complex or deviate from the desired cognitive objective. This study underscores the importance of strategic prompt engineering in fostering pedagogically sound AI-driven educational solutions and advises on optimizing AI for quality content generation in learning analytics and smart learning environments. 

**Abstract (ZH)**: 轻量级提示工程策略对OneClickQuiz中生成性人工智能内容认知对齐的影响研究 

---
# Real-time nonlinear inversion of magnetic resonance elastography with operator learning 

**Title (ZH)**: 实时非线性磁共振弹性成像的算子学习反演 

**Authors**: Juampablo E. Heras Rivera, Caitlin M. Neher, Mehmet Kurt  

**Link**: [PDF](https://arxiv.org/pdf/2510.03372)  

**Abstract**: $\textbf{Purpose:}$ To develop and evaluate an operator learning framework for nonlinear inversion (NLI) of brain magnetic resonance elastography (MRE) data, which enables real-time inversion of elastograms with comparable spatial accuracy to NLI.
$\textbf{Materials and Methods:}$ In this retrospective study, 3D MRE data from 61 individuals (mean age, 37.4 years; 34 female) were used for development of the framework. A predictive deep operator learning framework (oNLI) was trained using 10-fold cross-validation, with the complex curl of the measured displacement field as inputs and NLI-derived reference elastograms as outputs. A structural prior mechanism, analogous to Soft Prior Regularization in the MRE literature, was incorporated to improve spatial accuracy. Subject-level evaluation metrics included Pearson's correlation coefficient, absolute relative error, and structural similarity index measure between predicted and reference elastograms across brain regions of different sizes to understand accuracy. Statistical analyses included paired t-tests comparing the proposed oNLI variants to the convolutional neural network baselines.
$\textbf{Results:}$ Whole brain absolute percent error was 8.4 $\pm$ 0.5 ($\mu'$) and 10.0 $\pm$ 0.7 ($\mu''$) for oNLI and 15.8 $\pm$ 0.8 ($\mu'$) and 26.1 $\pm$ 1.1 ($\mu''$) for CNNs. Additionally, oNLI outperformed convolutional architectures as per Pearson's correlation coefficient, $r$, in the whole brain and across all subregions for both the storage modulus and loss modulus (p < 0.05).
$\textbf{Conclusion:}$ The oNLI framework enables real-time MRE inversion (30,000x speedup), outperforming CNN-based approaches and maintaining the fine-grained spatial accuracy achievable with NLI in the brain. 

**Abstract (ZH)**: 目的: 开发并评估一种操作学习框架，用于非线性反演（NLI）的脑磁共振弹性成像（MRE）数据，该框架能够实现实时的弹性图反演，具有与NLI相当的Spatial准确性。 

---
# Distributed Low-Communication Training with Decoupled Momentum Optimization 

**Title (ZH)**: 去中心化低通信量训练与解耦动量优化 

**Authors**: Sasho Nedelkoski, Alexander Acker, Odej Kao, Soeren Becker, Dominik Scheinert  

**Link**: [PDF](https://arxiv.org/pdf/2510.03371)  

**Abstract**: The training of large models demands substantial computational resources, typically available only in data centers with high-bandwidth interconnects. However, reducing the reliance on high-bandwidth interconnects between nodes enables the use of distributed compute resources as an alternative to centralized data center training. Building on recent advances in distributed model training, we propose an approach that further reduces communication by combining infrequent synchronizations across distributed model replicas with gradient momentum compression. In particular, we treat the optimizer momentum as a signal and decompose the Nesterov momentum into high- and low-frequency components via the discrete cosine transform (DCT). Only the high-frequency components are synchronized across model replicas every $H$ steps. Empirically, our method achieves up to a $16\times$ reduction in communication compared to the baseline DiLoCo, and it generalizes across architectures, including transformer-based language models and convolutional neural networks for images. Overall, this work advances the feasibility of training large models on distributed nodes with low-bandwidth interconnects. 

**Abstract (ZH)**: 大规模模型训练需要大量的计算资源，通常只有在具备高带宽互联的数据中心中才能获得。然而，减少节点间对高带宽互联的依赖性使得分布式计算资源可以作为中心化数据中心训练的替代方案。基于最近在分布式模型训练方面的进展，我们提出了一种进一步减少通信的方法，该方法将稀疏同步与梯度动量压缩相结合。具体而言，我们将优化器动量视为信号，并通过离散余弦变换（DCT）将Nesterov动量分解为高频和低频分量。只有高频分量在每$H$步在模型副本之间进行同步。实验结果表明，与基线DiLoCo方法相比，我们的方法可以实现最多16倍的通信减少，并且可以在包括基于变换器的语言模型和用于图像的卷积神经网络在内的多种架构上泛化。总体而言，这项工作促进了在具备低带宽互联的分布式节点上训练大规模模型的可行性。 

---
# InstructPLM-mu: 1-Hour Fine-Tuning of ESM2 Beats ESM3 in Protein Mutation Predictions 

**Title (ZH)**: InstructPLM-mu：1小时微调的ESM2在蛋白质突变预测中优于ESM3 

**Authors**: Junde Xu, Yapin Shi, Lijun Lang, Taoyong Cui, Zhiming Zhang, Guangyong Chen, Jiezhong Qiu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2510.03370)  

**Abstract**: Multimodal protein language models deliver strong performance on mutation-effect prediction, but training such models from scratch demands substantial computational resources. In this paper, we propose a fine-tuning framework called InstructPLM-mu and try to answer a question: \textit{Can multimodal fine-tuning of a pretrained, sequence-only protein language model match the performance of models trained end-to-end? } Surprisingly, our experiments show that fine-tuning ESM2 with structural inputs can reach performance comparable to ESM3. To understand how this is achieved, we systematically compare three different feature-fusion designs and fine-tuning recipes. Our results reveal that both the fusion method and the tuning strategy strongly affect final accuracy, indicating that the fine-tuning process is not trivial. We hope this work offers practical guidance for injecting structure into pretrained protein language models and motivates further research on better fusion mechanisms and fine-tuning protocols. 

**Abstract (ZH)**: 多模态蛋白质语言模型在突变效应预测任务中表现出strong性能，但从头训练此类模型需要大量计算资源。本文提出了一种名为InstructPLM-mu的微调框架，并试图回答一个问题：\textit{仅序列的预训练蛋白质语言模型的多模态微调能否与端到端训练的模型性能相当？} 让人惊讶的是，我们的实验显示，使用结构输入微调ESM2可以达到与ESM3相当的性能。为了理解这是如何实现的，我们系统地比较了三种不同的特征融合设计和微调方案。我们的结果揭示了融合方法和微调策略对最终准确度的影响，表明微调过程并非易事。希望本工作能为向预训练蛋白质语言模型注入结构提供实用指导，并激发进一步研究更好的融合机制和微调协议。 

---
# TriQuest:An AI Copilot-Powered Platform for Interdisciplinary Curriculum Design 

**Title (ZH)**: TriQuest:由AI副驾赋能的跨学科课程设计平台 

**Authors**: Huazhen Wang, Huimin Yang, Hainbin Lin, Yan Dong, Lili Chen, Liangliang Xia, Wenwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03369)  

**Abstract**: Interdisciplinary teaching is a cornerstone of modern curriculum reform, but its implementation is hindered by challenges in knowledge integration and time-consuming lesson planning. Existing tools often lack the required pedagogical and domain-specific this http URL introduce TriQuest, an AI-copilot platform designed to solve these problems. TriQuest uses large language models and knowledge graphs via an intuitive GUI to help teachers efficiently generate high-quality interdisciplinary lesson plans. Its core features include intelligent knowledge integration from various disciplines and a human-computer collaborative review process to ensure quality and this http URL a study with 43 teachers, TriQuest increased curriculum design efficiency by an average of 75% and improved lesson plan quality scores by 41%. It also significantly lowered design barriers and cognitive load. Our work presents a new paradigm for empowering teacher professional development with intelligent technologies. 

**Abstract (ZH)**: 跨学科教学是现代课程改革的基石，但其实施受到学科知识整合和耗时的lesson planning的挑战。现有工具往往缺乏必要的教学和领域特定功能。为此，我们介绍了TriQuest，一个AI copilot平台，旨在解决这些问题。TriQuest通过直观的GUI利用大语言模型和知识图谱，帮助教师高效生成高质量的跨学科lesson plan。其核心功能包括跨学科智能知识整合和人机协作评审流程，以确保质量和教学设计一致性。在一项涉及43名教师的研究所示，TriQuest将课程设计效率平均提高了75%，提高了lesson plan质量评分41%，并显著降低了设计障碍和认知负荷。我们的工作展示了智能技术赋能教师专业发展的新范式。 

---
# An Adaptive Responsible AI Governance Framework for Decentralized Organizations 

**Title (ZH)**: 面向去中心化组织的自适应负责任人工智能治理框架 

**Authors**: Kiana Jafari Meimandi, Anka Reuel, Gabriela Aranguiz-Dias, Hatim Rahama, Ala-Eddine Ayadi, Xavier Boullier, Jérémy Verdo, Louis Montanie, Mykel Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2510.03368)  

**Abstract**: This paper examines the assessment challenges of Responsible AI (RAI) governance efforts in globally decentralized organizations through a case study collaboration between a leading research university and a multinational enterprise. While there are many proposed frameworks for RAI, their application in complex organizational settings with distributed decision-making authority remains underexplored. Our RAI assessment, conducted across multiple business units and AI use cases, reveals four key patterns that shape RAI implementation: (1) complex interplay between group-level guidance and local interpretation, (2) challenges translating abstract principles into operational practices, (3) regional and functional variation in implementation approaches, and (4) inconsistent accountability in risk oversight. Based on these findings, we propose an Adaptive RAI Governance (ARGO) Framework that balances central coordination with local autonomy through three interdependent layers: shared foundation standards, central advisory resources, and contextual local implementation. We contribute insights from academic-industry collaboration for RAI assessments, highlighting the importance of modular governance approaches that accommodate organizational complexity while maintaining alignment with responsible AI principles. These lessons offer practical guidance for organizations navigating the transition from RAI principles to operational practice within decentralized structures. 

**Abstract (ZH)**: 本研究通过一所领先的研究大学与一家跨国企业之间的案例研究合作，探讨了全球分散组织中负责任人工智能（RAI）治理努力的评估挑战。尽管已经提出了许多RAI框架，但它们在拥有分布式决策权的复杂组织环境中的应用仍然未得到充分探索。我们的RAI评估涵盖了多个业务部门和人工智能应用场景，揭示了四个关键模式，这些模式影响了RAI的实施：（1）小组指导与本地解释之间的复杂互动，（2）将抽象原则转化为操作实践的挑战，（3）实施方法在区域和功能上的差异，以及（4）风险监督中的不一致问责制。基于这些发现，我们提出了一种平衡中央协调与地方自主性的适应型RAI治理（ARGO）框架，该框架由三个相互依存的层面组成：共同的基础标准、中央咨询资源和情境下的地方实施。我们通过学术-产业合作为RAI评估提供了见解，强调了在组织复杂性与负责任人工智能原则之间保持一致性的模块化治理方法的重要性。这些教训为组织在分散结构中从RAI原则过渡到操作实践提供了实用指导。 

---
# Disentangling Recall and Reasoning in Transformer Models through Layer-wise Attention and Activation Analysis 

**Title (ZH)**: 通过层wise注意力和激活分析解耦Transformer模型中的回忆与推理 

**Authors**: Harshwardhan Fartale, Ashish Kattamuri, Rahul Raja, Arpita Vats, Ishita Prasad, Akshata Kishore Moharir  

**Link**: [PDF](https://arxiv.org/pdf/2510.03366)  

**Abstract**: Transformer-based language models excel at both recall (retrieving memorized facts) and reasoning (performing multi-step inference), but whether these abilities rely on distinct internal mechanisms remains unclear. Distinguishing recall from reasoning is crucial for predicting model generalization, designing targeted evaluations, and building safer interventions that affect one ability without disrupting the this http URL approach this question through mechanistic interpretability, using controlled datasets of synthetic linguistic puzzles to probe transformer models at the layer, head, and neuron level. Our pipeline combines activation patching and structured ablations to causally measure component contributions to each task type. Across two model families (Qwen and LLaMA), we find that interventions on distinct layers and attention heads lead to selective impairments: disabling identified "recall circuits" reduces fact-retrieval accuracy by up to 15\% while leaving reasoning intact, whereas disabling "reasoning circuits" reduces multi-step inference by a comparable margin. At the neuron level, we observe task-specific firing patterns, though these effects are less robust, consistent with neuronal this http URL results provide the first causal evidence that recall and reasoning rely on separable but interacting circuits in transformer models. These findings advance mechanistic interpretability by linking circuit-level structure to functional specialization and demonstrate how controlled datasets and causal interventions can yield mechanistic insights into model cognition, informing safer deployment of large language models. 

**Abstract (ZH)**: 基于变换器的语言模型在回忆（检索记忆中的事实）和推理（执行多步推理）方面表现优异，但这些能力是否依赖于不同的内部机制尚不明确。通过机制可解释性方法，使用合成语言谜题的受控数据集，在层、头和神经元级别探索变换器模型，区分回忆与推理。我们的管道结合激活修复和结构化剥离，因果测量各组件对不同类型任务的贡献。在两种模型系列（Qwen和LLaMA）中，我们发现对不同层和注意力头的干预导致选择性的损伤：禁用识别出的“回忆回路”可将事实检索准确性降低最多15%，而推理能力保持不变；禁用“推理回路”则会同等程度地影响多步推理。在神经元级别，我们观察到特定的任务激活模式，然而这些效果不够稳健，与神经元的这一特性一致。这些结果首次提供了因果证据，表明回忆和推理依赖于分离但相互作用的回路。这些发现推进了机制可解释性，通过将电路级结构与功能专业化联系起来，并展示了如何通过受控数据集和因果干预获得有关模型认知的机制性见解，从而指导大型语言模型的安全部署。 

---
# Diffusion-Based, Data-Assimilation-Enabled Super-Resolution of Hub-height Winds 

**Title (ZH)**: 基于扩散的、数据同化的高空风超分辨率重建 

**Authors**: Xiaolong Ma, Xu Dong, Ashley Tarrant, Lei Yang, Rao Kotamarthi, Jiali Wang, Feng Yan, Rajkumar Kettimuthu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03364)  

**Abstract**: High-quality observations of hub-height winds are valuable but sparse in space and time. Simulations are widely available on regular grids but are generally biased and too coarse to inform wind-farm siting or to assess extreme-weather-related risks (e.g., gusts) at infrastructure scales. To fully utilize both data types for generating high-quality, high-resolution hub-height wind speeds (tens to ~100m above ground), this study introduces WindSR, a diffusion model with data assimilation for super-resolution downscaling of hub-height winds. WindSR integrates sparse observational data with simulation fields during downscaling using state-of-the-art diffusion models. A dynamic-radius blending method is introduced to merge observations with simulations, providing conditioning for the diffusion process. Terrain information is incorporated during both training and inference to account for its role as a key driver of winds. Evaluated against convolutional-neural-network and generative-adversarial-network baselines, WindSR outperforms them in both downscaling efficiency and accuracy. Our data assimilation reduces WindSR's model bias by approximately 20% relative to independent observations. 

**Abstract (ZH)**: 高空间与时间分辨率的轮毂高度风观测数据稀缺，而模拟数据虽广泛可用但通常具有偏差且分辨率过低，不足以用于风电场选址或评估与极端天气（如阵风）相关的基础设施风险。为了充分利用这两种数据类型生成高分辨率（十米至百米以上）轮毂高度风速，本文 introduces WindSR，一种结合数据同化的扩散模型，用于超分辨率下-scaling 轮毂高度风速。WindSR 使用最先进的扩散模型在下-scaling 过程中整合稀疏观测数据和模拟场。引入了一种动态半径混合方法，将观测数据与模拟数据合并，为扩散过程提供条件。训练和推断过程中均加入了地形信息，以反映地形对风速的关键驱动作用。与卷积神经网络和生成对抗网络基线相比，WindSR 在下-scaling 效率和准确性方面表现出更优性能。我们的数据同化将 WindSR 模型偏差相对于独立观测数据降低了约 20%。 

---
# Unified Unsupervised Anomaly Detection via Matching Cost Filtering 

**Title (ZH)**: 统一的无监督异常检测方法基于匹配成本过滤 

**Authors**: Zhe Zhang, Mingxiu Cai, Gaochang Wu, Jing Zhang, Lingqiao Liu, Dacheng Tao, Tianyou Chai, Xiatian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03363)  

**Abstract**: Unsupervised anomaly detection (UAD) aims to identify image- and pixel-level anomalies using only normal training data, with wide applications such as industrial inspection and medical analysis, where anomalies are scarce due to privacy concerns and cold-start constraints. Existing methods, whether reconstruction-based (restoring normal counterparts) or embedding-based (pretrained representations), fundamentally conduct image- or feature-level matching to generate anomaly maps. Nonetheless, matching noise has been largely overlooked, limiting their detection ability. Beyond earlier focus on unimodal RGB-based UAD, recent advances expand to multimodal scenarios, e.g., RGB--3D and RGB--Text, enabled by point cloud sensing and vision--language models. Despite shared challenges, these lines remain largely isolated, hindering a comprehensive understanding and knowledge transfer. In this paper, we advocate unified UAD for both unimodal and multimodal settings in the matching perspective. Under this insight, we present Unified Cost Filtering (UCF), a generic post-hoc refinement framework for refining anomaly cost volume of any UAD model. The cost volume is constructed by matching a test sample against normal samples from the same or different modalities, followed by a learnable filtering module with multi-layer attention guidance from the test sample, mitigating matching noise and highlighting subtle anomalies. Comprehensive experiments on 22 diverse benchmarks demonstrate the efficacy of UCF in enhancing a variety of UAD methods, consistently achieving new state-of-the-art results in both unimodal (RGB) and multimodal (RGB--3D, RGB--Text) UAD scenarios. Code and models will be released at this https URL. 

**Abstract (ZH)**: 无监督异常检测（UAD）旨在仅使用正常训练数据来识别图像级和像素级的异常，广泛应用于工业检测和医疗分析等领域，其中由于隐私和冷启动的限制，异常往往是稀缺的。现有方法无论基于重构（恢复正常样本）还是嵌入（预训练表示），本质上都是执行图像级或特征级匹配以生成异常图。然而，匹配噪声已被大量忽视，限制了它们的检测能力。超越早期基于单一RGB的数据的无监督异常检测，近期进展扩展到了多模态场景，如RGB-3D和RGB-Text，得益于点云传感和跨模态模型。尽管存在共同的挑战，但这些领域仍然相对孤立，阻碍了对整体理解以及知识迁移的认知。本文从匹配视角提倡统一的无监督异常检测模型，基于此见解，我们提出了统一成本过滤（UCF），这是一种通用的后处理精炼框架，用于精炼任何UAD模型的成本体素。成本体素通过将测试样本与相同或不同模态的正常样本进行匹配构建，并通过测试样本的多层注意力引导的学习过滤模块进行精炼，以减轻匹配噪声并突出细微异常。在22个多样化的基准测试上进行全面实验表明，UCF能够增强多种无监督异常检测方法的有效性，在单模态（RGB）和多模态（RGB-3D，RGB-Text）无监督异常检测场景中均能取得新的最先进的结果。代码和模型将于此网址发布：this https URL。 

---
# Provenance Networks: End-to-End Exemplar-Based Explainability 

**Title (ZH)**: 来源网络：端到端示例为基础的可解释性 

**Authors**: Ali Kayyam, Anusha Madan Gopal, M. Anthony Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2510.03361)  

**Abstract**: We introduce provenance networks, a novel class of neural models designed to provide end-to-end, training-data-driven explainability. Unlike conventional post-hoc methods, provenance networks learn to link each prediction directly to its supporting training examples as part of the model's normal operation, embedding interpretability into the architecture itself. Conceptually, the model operates similarly to a learned KNN, where each output is justified by concrete exemplars weighted by relevance in the feature space. This approach facilitates systematic investigations of the trade-off between memorization and generalization, enables verification of whether a given input was included in the training set, aids in the detection of mislabeled or anomalous data points, enhances resilience to input perturbations, and supports the identification of similar inputs contributing to the generation of a new data point. By jointly optimizing the primary task and the explainability objective, provenance networks offer insights into model behavior that traditional deep networks cannot provide. While the model introduces additional computational cost and currently scales to moderately sized datasets, it provides a complementary approach to existing explainability techniques. In particular, it addresses critical challenges in modern deep learning, including model opaqueness, hallucination, and the assignment of credit to data contributors, thereby improving transparency, robustness, and trustworthiness in neural models. 

**Abstract (ZH)**: 基于证据的神经网络：一种新型端到端训练数据驱动的可解释性模型 

---
# Physics-informed Neural-operator Predictive Control for Drag Reduction in Turbulent Flows 

**Title (ZH)**: 基于物理的神经算子预测控制以降低湍流流动的drag损失 

**Authors**: Zelin Zhao, Zongyi Li, Kimia Hassibi, Kamyar Azizzadenesheli, Junchi Yan, H. Jane Bae, Di Zhou, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.03360)  

**Abstract**: Assessing turbulence control effects for wall friction numerically is a significant challenge since it requires expensive simulations of turbulent fluid dynamics. We instead propose an efficient deep reinforcement learning (RL) framework for modeling and control of turbulent flows. It is model-based RL for predictive control (PC), where both the policy and the observer models for turbulence control are learned jointly using Physics Informed Neural Operators (PINO), which are discretization invariant and can capture fine scales in turbulent flows accurately. Our PINO-PC outperforms prior model-free reinforcement learning methods in various challenging scenarios where the flows are of high Reynolds numbers and unseen, i.e., not provided during model training. We find that PINO-PC achieves a drag reduction of 39.0\% under a bulk-velocity Reynolds number of 15,000, outperforming previous fluid control methods by more than 32\%. 

**Abstract (ZH)**: 基于物理学约束神经运算符的高效深度强化学习湍流控制方法 

---
# Understanding Transformers for Time Series: Rank Structure, Flow-of-ranks, and Compressibility 

**Title (ZH)**: 理解时间序列中的变换器：层级结构、秩流和压缩性 

**Authors**: Annan Yu, Danielle C. Maddix, Boran Han, Xiyuan Zhang, Abdul Fatir Ansari, Oleksandr Shchur, Christos Faloutsos, Andrew Gordon Wilson, Michael W. Mahoney, Yuyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03358)  

**Abstract**: Transformers are widely used across data modalities, and yet the principles distilled from text models often transfer imperfectly to models trained to other modalities. In this paper, we analyze Transformers through the lens of rank structure. Our focus is on the time series setting, where the structural properties of the data differ remarkably from those of text or vision. We show that time-series embeddings, unlike text or vision, exhibit sharply decaying singular value spectra: small patch sizes and smooth continuous mappings concentrate the data into low-rank subspaces. From this, we prove that the associated $Q/K/V$ projections admit accurate low-rank approximations, and that attention layers become compressible in proportion to the decay of the embedding spectrum. We introduce the concept of flow-of-ranks, a phenomenon by which nonlinear mixing across depth inflates the rank, explaining why early layers are most amenable to compression and why ranks grow with depth. Guided by these theoretical and empirical results, we use these insights to compress Chronos, a large time series foundation model, achieving a reduction of $65\%$ in inference time and $81\%$ in memory, without loss of accuracy. Our findings provide principled guidance for allocating width, depth, and heads in time series foundation models, and for exploiting their inherent compressibility. 

**Abstract (ZH)**: 基于秩结构视角分析变压器：时间序列模型中的压缩原理与实践 

---
# Inference-Time Search using Side Information for Diffusion-based Image Reconstruction 

**Title (ZH)**: 基于辅信息的推断时搜索图像重建 

**Authors**: Mahdi Farahbakhsh, Vishnu Teja Kunde, Dileep Kalathil, Krishna Narayanan, Jean-Francois Chamberland  

**Link**: [PDF](https://arxiv.org/pdf/2510.03352)  

**Abstract**: Diffusion models have emerged as powerful priors for solving inverse problems. However, existing approaches typically overlook side information that could significantly improve reconstruction quality, especially in severely ill-posed settings. In this work, we propose a novel inference-time search algorithm that guides the sampling process using the side information in a manner that balances exploration and exploitation. This enables more accurate and reliable reconstructions, providing an alternative to the gradient-based guidance that is prone to reward-hacking artifacts. Our approach can be seamlessly integrated into a wide range of existing diffusion-based image reconstruction pipelines. Through extensive experiments on a number of inverse problems, such as box inpainting, super-resolution, and various deblurring tasks including motion, Gaussian, nonlinear, and blind deblurring, we show that our approach consistently improves the qualitative and quantitative performance of diffusion-based image reconstruction algorithms. We also show the superior performance of our approach with respect to other baselines, including reward gradient-based guidance algorithms. The code is available at \href{this https URL}{this repository}. 

**Abstract (ZH)**: 基于侧信息的探索与利用平衡搜索算法在反问题中增强扩散模型的图像重建性能 

---
# Interpretable Neuropsychiatric Diagnosis via Concept-Guided Graph Neural Networks 

**Title (ZH)**: 概念引导的图神经网络在可解释神经精神疾病诊断中的应用 

**Authors**: Song Wang, Zhenyu Lei, Zhen Tan, Jundong Li, Javier Rasero, Aiying Zhang, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.03351)  

**Abstract**: Nearly one in five adolescents currently live with a diagnosed mental or behavioral health condition, such as anxiety, depression, or conduct disorder, underscoring the urgency of developing accurate and interpretable diagnostic tools. Resting-state functional magnetic resonance imaging (rs-fMRI) provides a powerful lens into large-scale functional connectivity, where brain regions are modeled as nodes and inter-regional synchrony as edges, offering clinically relevant biomarkers for psychiatric disorders. While prior works use graph neural network (GNN) approaches for disorder prediction, they remain complex black-boxes, limiting their reliability and clinical translation. In this work, we propose CONCEPTNEURO, a concept-based diagnosis framework that leverages large language models (LLMs) and neurobiological domain knowledge to automatically generate, filter, and encode interpretable functional connectivity concepts. Each concept is represented as a structured subgraph linking specific brain regions, which are then passed through a concept classifier. Our design ensures predictions through clinically meaningful connectivity patterns, enabling both interpretability and strong predictive performance. Extensive experiments across multiple psychiatric disorder datasets demonstrate that CONCEPTNEURO-augmented GNNs consistently outperform their vanilla counterparts, improving accuracy while providing transparent, clinically aligned explanations. Furthermore, concept analyses highlight disorder-specific connectivity patterns that align with expert knowledge and suggest new hypotheses for future investigation, establishing CONCEPTNEURO as an interpretable, domain-informed framework for psychiatric disorder diagnosis. 

**Abstract (ZH)**: 基于概念的神经精神障碍诊断框架：CONCEPTNEURO 

---
# AgentCaster: Reasoning-Guided Tornado Forecasting 

**Title (ZH)**: AgentCaster: 基于推理的龙卷风预报 

**Authors**: Michael Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03349)  

**Abstract**: There is a growing need to evaluate Large Language Models (LLMs) on complex, high-impact, real-world tasks to assess their true readiness as reasoning agents. To address this gap, we introduce AgentCaster, a contamination-free framework employing multimodal LLMs end-to-end for the challenging, long-horizon task of tornado forecasting. Within AgentCaster, models interpret heterogeneous spatiotemporal data from a high-resolution convection-allowing forecast archive. We assess model performance over a 40-day period featuring diverse historical data, spanning several major tornado outbreaks and including over 500 tornado reports. Each day, models query interactively from a pool of 3,625 forecast maps and 40,125 forecast soundings for a forecast horizon of 12-36 hours. Probabilistic tornado-risk polygon predictions are verified against ground truths derived from geometric comparisons across disjoint risk bands in projected coordinate space. To quantify accuracy, we propose domain-specific TornadoBench and TornadoHallucination metrics, with TornadoBench highly challenging for both LLMs and domain expert human forecasters. Notably, human experts significantly outperform state-of-the-art models, which demonstrate a strong tendency to hallucinate and overpredict risk intensity, struggle with precise geographic placement, and exhibit poor spatiotemporal reasoning in complex, dynamically evolving systems. AgentCaster aims to advance research on improving LLM agents for challenging reasoning tasks in critical domains. 

**Abstract (ZH)**: 一种无污染框架：AgentCaster，用于复杂高影响真实世界任务的大规模语言模型评估——以龙卷风预报为例 

---
# KVComm: Enabling Efficient LLM Communication through Selective KV Sharing 

**Title (ZH)**: KVComm: 通过选择性键值共享实现高效LLM通信 

**Authors**: Xiangyu Shi, Marco Chiesa, Gerald Q. Maguire Jr., Dejan Kostic  

**Link**: [PDF](https://arxiv.org/pdf/2510.03346)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in multi-agent systems, where effective inter-model communication is crucial. Existing communication protocols either rely on natural language, incurring high inference costs and information loss, or on hidden states, which suffer from information concentration bias and inefficiency. To address these limitations, we propose KVComm, a novel communication framework that enables efficient communication between LLMs through selective sharing of KV pairs. KVComm leverages the rich information encoded in the KV pairs while avoiding the pitfalls of hidden states. We introduce a KV layer-wise selection strategy based on attention importance scores with a Gaussian prior to identify the most informative KV pairs for communication. Extensive experiments across diverse tasks and model pairs demonstrate that KVComm achieves comparable performance to the upper-bound method, which directly merges inputs to one model without any communication, while transmitting as few as 30\% of layers' KV pairs. Our study highlights the potential of KV pairs as an effective medium for inter-LLM communication, paving the way for scalable and efficient multi-agent systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多代理系统中的应用越来越广泛，高效的模型间通信至关重要。现有的通信协议要么依赖自然语言，导致推理成本高和信息损失，要么依赖隐藏状态，易产生信息集中偏差和低效问题。为解决这些局限性，我们提出了一种名为KVComm的新颖通信框架，通过选择性共享KV对实现LLMs间的高效通信。KVComm利用了KV对中丰富的信息，同时避免了隐藏状态的缺陷。我们基于注意力重要性分数和高斯先验提出了一种层间KV选择策略，以识别最Informative的KV对用于通信。跨多种任务和模型对的广泛实验表明，KVComm在传输最少30%的层KV对的情况下，达到了与直接将输入合并到一个模型的上界方法相当的性能。我们的研究突显了KV对作为LLM间有效通信媒介的潜力，为可扩展和高效的多代理系统铺平了道路。 

---
# Pilot selection in the era of Virtual reality: algorithms for accurate and interpretable machine learning models 

**Title (ZH)**: 虚拟现实时代的人选试点：准确可解释的机器学习算法 

**Authors**: Luoma Ke, Guangpeng Zhang, Jibo He, Yajing Li, Yan Li, Xufeng Liu, Peng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03345)  

**Abstract**: With the rapid growth of the aviation industry, there is a need for a large number of flight crew. How to select the right pilots in a cost-efficient manner has become an important research question. In the current study, twenty-three pilots were recruited from China Eastern Airlines, and 23 novices were from the community of Tsinghua University. A novel approach incorporating machine learning and virtual reality technology was applied to distinguish features between these participants with different flight skills. Results indicate that SVM with the MIC feature selection method consistently achieved the highest prediction performance on all metrics with an Accuracy of 0.93, an AUC of 0.96, and an F1 of 0.93, which outperforms four other classifier algorithms and two other feature selection methods. From the perspective of feature selection methods, the MIC method can select features with a nonlinear relationship to sampling labels, instead of a simple filter-out. Our new implementation of the SVM + MIC algorithm outperforms all existing pilot selection algorithms and perhaps provides the first implementation based on eye tracking and flight dynamics data. This study's VR simulation platforms and algorithms can be used for pilot selection and training. 

**Abstract (ZH)**: 随着航空业的迅速增长，飞行员的需求量大大增加。如何以成本高效的方式选拔合适的飞行员已成为一个重要研究问题。在本研究中，从东方航空公司招募了23名飞行员，另从清华大学社区招募了23名新手。本研究应用了结合机器学习和虚拟现实技术的新型方法，以区分技能不同的参与者特征。结果显示，使用MIC特征选择方法的SVM在所有指标中表现最佳，准确率（Accuracy）为0.93，AUC为0.96，F1值为0.93，优于其他四种分类算法和两种特征选择方法。从特征选择方法的角度来看，MIC方法可以筛选出与抽样标签之间存在非线性关系的特征，而不仅仅是简单的过滤。我们的SVM + MIC算法的新实现超越了所有现有的飞行员选拔算法，并可能基于眼动追踪和飞行动力学数据提供首个实现。本研究的VR仿真平台和算法可用于飞行员选拔和训练。 

---
# Defining a Strategic Action Plan for AI in Higher Education 

**Title (ZH)**: 为高等教育制定人工智能战略行动方案 

**Authors**: Nikolaos Avouris  

**Link**: [PDF](https://arxiv.org/pdf/2510.03343)  

**Abstract**: This paper discusses key challenges of Artificial Intelligence in Education, with main focus on higher education institutions. We start with reviewing normative actions of international organizations and concerns expressed about the current technical landscape. Then we proceed with proposing a framework that comprises five key dimensions relating to the main challenges relating to AI in higher education institutions, followed by five key strategic actions that the main stakeholders need to take in order to address the current developments. We map these actions to the main stakeholders of higher education and propose a deployment plan. This defines a framework along the dimensions: Challenges, Actions, Stakeholders, Deployment CASD. Examples of AI specific actions at the institutional and individual course level are also provided and discussed. 

**Abstract (ZH)**: 本文讨论了人工智能在教育中的关键挑战，重点关注高等教育机构。我们从审查国际组织的规范性行动和对当前技术景观的关切开始，随后提出了一个框架，该框架包含五个关键维度，涉及人工智能在高等教育机构中面临的主要挑战，并提出了五个关键战略行动，以便主要利益相关者能够应对当前的发展。我们将这些行动映射到高等教育的主要利益相关者，并提出了部署计划。该框架沿用了挑战、行动、利益相关者、部署（CASD）四个维度。还提供了并讨论了机构和课程层面的特定人工智能行动示例。 

---
# Learning Pareto-Optimal Pandemic Intervention Policies with MORL 

**Title (ZH)**: 基于MORL学习 Pareto 最优 pandemic 干预策略 

**Authors**: Marian Chen, Miri Zilka  

**Link**: [PDF](https://arxiv.org/pdf/2510.03340)  

**Abstract**: The COVID-19 pandemic underscored a critical need for intervention strategies that balance disease containment with socioeconomic stability. We approach this challenge by designing a framework for modeling and evaluating disease-spread prevention strategies. Our framework leverages multi-objective reinforcement learning (MORL) - a formulation necessitated by competing objectives - combined with a new stochastic differential equation (SDE) pandemic simulator, calibrated and validated against global COVID-19 data. Our simulator reproduces national-scale pandemic dynamics with orders of magnitude higher fidelity than other models commonly used in reinforcement learning (RL) approaches to pandemic intervention. Training a Pareto-Conditioned Network (PCN) agent on this simulator, we illustrate the direct policy trade-offs between epidemiological control and economic stability for COVID-19. Furthermore, we demonstrate the framework's generality by extending it to pathogens with different epidemiological profiles, such as polio and influenza, and show how these profiles lead the agent to discover fundamentally different intervention policies. To ground our work in contemporary policymaking challenges, we apply the model to measles outbreaks, quantifying how a modest 5% drop in vaccination coverage necessitates significantly more stringent and costly interventions to curb disease spread. This work provides a robust and adaptable framework to support transparent, evidence-based policymaking for mitigating public health crises. 

**Abstract (ZH)**: COVID-19疫情凸显了在疾病控制与社会经济稳定之间寻求平衡的干预策略的迫切需求。我们通过设计一个疾病传播预防策略建模与评估框架来应对这一挑战。该框架利用多目标强化学习（MORL）——一种由相互竞争的目标所要求的表述——结合一种新的随机微分方程（SDE）疫情模拟器，并根据全球COVID-19数据进行了校准和验证。我们的模拟器在准确再现国家级规模的疫情动态方面比其他常用于强化学习（RL）方法中的模型高出数个数量级。通过在此模拟器上训练帕累托条件网络（PCN）代理，我们展示了COVID-19在流行病学控制与经济稳定之间的直接政策权衡。此外，我们通过将该框架扩展到具有不同流行病学特征的病原体——如脊髓灰质炎和流感——展示了其广泛的适用性，并展示了这些特征如何引导代理发现根本不同的干预策略。为了将我们的工作与当前的政策制定挑战相结合，我们应用该模型研究麻疹暴发，量化了5%疫苗接种覆盖率下降需要采取更为严格和昂贵的措施来遏制疾病传播的程度。本研究提供了支持公开透明、基于证据的公共卫生危机缓解政策制定的稳健且灵活的框架。 

---
# Pool Me Wisely: On the Effect of Pooling in Transformer-Based Models 

**Title (ZH)**: 池化用得智慧些：关于基于变压器的模型中池化的效应 

**Authors**: Sofiane Ennadir, Levente Zólyomi, Oleg Smirnov, Tianze Wang, John Pertoft, Filip Cornell, Lele Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03339)  

**Abstract**: Transformer models have become the dominant backbone for sequence modeling, leveraging self-attention to produce contextualized token representations. These are typically aggregated into fixed-size vectors via pooling operations for downstream tasks. While much of the literature has focused on attention mechanisms, the role of pooling remains underexplored despite its critical impact on model behavior. In this paper, we introduce a theoretical framework that rigorously characterizes the expressivity of Transformer-based models equipped with widely used pooling methods by deriving closed-form bounds on their representational capacity and the ability to distinguish similar inputs. Our analysis extends to different variations of attention formulations, demonstrating that these bounds hold across diverse architectural variants. We empirically evaluate pooling strategies across tasks requiring both global and local contextual understanding, spanning three major modalities: computer vision, natural language processing, and time-series analysis. Results reveal consistent trends in how pooling choices affect accuracy, sensitivity, and optimization behavior. Our findings unify theoretical and empirical perspectives, providing practical guidance for selecting or designing pooling mechanisms suited to specific tasks. This work positions pooling as a key architectural component in Transformer models and lays the foundation for more principled model design beyond attention alone. 

**Abstract (ZH)**: 基于Transformer模型的聚合方法在不同任务中的理论分析与实证研究：统一表达能力和任务适应性 

---
# Linguistic and Audio Embedding-Based Machine Learning for Alzheimer's Dementia and Mild Cognitive Impairment Detection: Insights from the PROCESS Challenge 

**Title (ZH)**: 基于语言和音频嵌入的机器学习方法在阿尔茨海默病和轻度认知 impairment 检测中的研究：PROCESS 挑战赛启示 

**Authors**: Adharsha Sam Edwin Sam Devahi, Sohail Singh Sangha, Prachee Priyadarshinee, Jithin Thilakan, Ivan Fu Xing Tan, Christopher Johann Clarke, Sou Ka Lon, Balamurali B T, Yow Wei Quin, Chen Jer-Ming  

**Link**: [PDF](https://arxiv.org/pdf/2510.03336)  

**Abstract**: Early detection of Alzheimer's Dementia (AD) and Mild Cognitive Impairment (MCI) is critical for timely intervention, yet current diagnostic approaches remain resource-intensive and invasive. Speech, encompassing both acoustic and linguistic dimensions, offers a promising non-invasive biomarker for cognitive decline. In this study, we present a machine learning framework for the PROCESS Challenge, leveraging both audio embeddings and linguistic features derived from spontaneous speech recordings. Audio representations were extracted using Whisper embeddings from the Cookie Theft description task, while linguistic features-spanning pronoun usage, syntactic complexity, filler words, and clause structure-were obtained from transcriptions across Semantic Fluency, Phonemic Fluency, and Cookie Theft picture description. Classification models aimed to distinguish between Healthy Controls (HC), MCI, and AD participants, while regression models predicted Mini-Mental State Examination (MMSE) scores. Results demonstrated that voted ensemble models trained on concatenated linguistic features achieved the best classification performance (F1 = 0.497), while Whisper embedding-based ensemble regressors yielded the lowest MMSE prediction error (RMSE = 2.843). Comparative evaluation within the PROCESS Challenge placed our models among the top submissions in regression task, and mid-range for classification, highlighting the complementary strengths of linguistic and audio embeddings. These findings reinforce the potential of multimodal speech-based approaches for scalable, non-invasive cognitive assessment and underline the importance of integrating task-specific linguistic and acoustic markers in dementia detection. 

**Abstract (ZH)**: 早期检测阿尔茨海默病痴呆（AD）和轻度认知障碍（MCI）对于及时干预至关重要，但当前的诊断方法仍资源密集且侵入性强。语言，涵盖声学和语义维度，为认知衰退提供了有前景的非侵入性生物标志物。在本研究中，我们提出了一种用于PROCESS挑战的比赛框架，利用来自自发性言语录音的声学嵌入和语言特征。声学表示使用Whisper嵌入从Cookie Theft描述任务中提取，而语言特征包括代词使用、句法复杂性、填充词和从Semantical Fluency、Phonemic Fluency和Cookie Theft图片描述中获得的从句结构。分类模型旨在区分健康对照组（HC）、MCI和AD参与者，而回归模型预测简易精神状态检查（MMSE）分数。结果表明，在拼接语言特征上训练的投票集成模型实现了最佳分类性能（F1 = 0.497），而基于Whisper嵌入的集成回归器在MMSE分数预测中的均方根误差最低（RMSE = 2.843）。在PROCESS挑战中的比较评估中，我们的模型在回归任务中排名靠前，在分类任务中处于中游水平，突显了语言和声学嵌入的互补优势。这些发现强化了多模态言语基方法在可扩展、非侵入性认知评估中的潜力，并强调了在痴呆症检测中整合任务特定语言和声学标记的重要性。 

---
# Intelligent Healthcare Ecosystems: Optimizing the Iron Triangle of Healthcare (Access, Cost, Quality) 

**Title (ZH)**: 智能医疗生态系统：优化医疗的铁三角（可及性、成本、质量） 

**Authors**: Vivek Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2510.03331)  

**Abstract**: The United States spends nearly 17% of GDP on healthcare yet continues to face uneven access and outcomes. This well-known trade-off among cost, quality, and access - the "iron triangle" - motivates a system-level redesign. This paper proposes an Intelligent Healthcare Ecosystem (iHE): an integrated, data-driven framework that uses generative AI and large language models, federated learning, interoperability standards (FHIR, TEFCA), and digital twins to improve access and quality while lowering cost. We review historical spending trends, waste, and international comparisons; introduce a value equation that jointly optimizes access, quality, and cost; and synthesize evidence on the enabling technologies and operating model for iHE. Methods follow a narrative review of recent literature and policy reports. Results outline core components (AI decision support, interoperability, telehealth, automation) and show how iHE can reduce waste, personalize care, and support value-based payment while addressing privacy, bias, and adoption challenges. We argue that a coordinated iHE can bend - if not break - the iron triangle, moving the system toward care that is more accessible, affordable, and high quality. 

**Abstract (ZH)**: 美国在医疗卫生上的支出占GDP的近17%，但仍面临不均衡的可及性和结果问题。这种广为人知的成本、质量和可及性之间的权衡——“铁三角”——促使我们需要对整个系统进行重新设计。本文提出了一种智能健康生态系统（iHE）：一个集成的数据驱动框架，利用生成性AI和大规模语言模型、联邦学习、互操作性标准（FHIR、TEFCA）以及数字孪生技术，以改善可及性和质量并降低成本。我们回顾了历史上的支出趋势、浪费情况以及国际比较；介绍了联合优化可及性、质量和成本的价值方程；并综合了关于iHE使能技术和运营模式的证据。方法遵循近期文献和政策报告的叙述性回顾。结果概述了核心组件（AI决策支持、互操作性、远程医疗、自动化），展示了iHE如何减少浪费、个性化护理，并支持基于价值的支付，同时解决隐私、偏见和采纳挑战。我们提出，协调的iHE可以改变，甚至打破“铁三角”，使系统朝着更可及、更负担得起且质量更高的护理方向发展。 

---
# NS-Pep: De novo Peptide Design with Non-Standard Amino Acids 

**Title (ZH)**: NS-Pep: 用非标准氨基酸的从头肽设计 

**Authors**: Tao Guo, Junbo Yin, Yu Wang, Xin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03326)  

**Abstract**: Peptide drugs incorporating non-standard amino acids (NSAAs) offer improved binding affinity and improved pharmacological properties. However, existing peptide design methods are limited to standard amino acids, leaving NSAA-aware design largely unexplored. We introduce NS-Pep, a unified framework for co-designing peptide sequences and structures with NSAAs. The main challenge is that NSAAs are extremely underrepresented-even the most frequent one, SEP, accounts for less than 0.4% of residues-resulting in a severe long-tailed distribution. To improve generalization to rare amino acids, we propose Residue Frequency-Guided Modification (RFGM), which mitigates over-penalization through frequency-aware logit calibration, supported by both theoretical and empirical analysis. Furthermore, we identify that insufficient side-chain modeling limits geometric representation of NSAAs. To address this, we introduce Progressive Side-chain Perception (PSP) for coarse-to-fine torsion and location prediction, and Interaction-Aware Weighting (IAW) to emphasize pocket-proximal residues. Moreover, NS-Pep generalizes naturally to the peptide folding task with NSAAs, addressing a major limitation of current tools. Experiments show that NS-Pep improves sequence recovery rate and binding affinity by 6.23% and 5.12%, respectively, and outperforms AlphaFold3 by 17.76% in peptide folding success rate. 

**Abstract (ZH)**: 非标准氨基酸aware肽药物设计框架NS-Pep：结合肽序列与结构的统一方法 

---
# Photorealistic Inpainting for Perturbation-based Explanations in Ecological Monitoring 

**Title (ZH)**: 基于扰动的生态监测光orealistic修复解释 

**Authors**: Günel Aghakishiyeva, Jiayi Zhou, Saagar Arya, James David Poling, Holly R. Houliston, Jamie N. Womble, David W. Johnston, Brinnae Bent  

**Link**: [PDF](https://arxiv.org/pdf/2510.03317)  

**Abstract**: Ecological monitoring is increasingly automated by vision models, yet opaque predictions limit trust and field adoption. We present an inpainting-guided, perturbation-based explanation technique that produces photorealistic, mask-localized edits that preserve scene context. Unlike masking or blurring, these edits stay in-distribution and reveal which fine-grained morphological cues drive predictions in tasks such as species recognition and trait attribution. We demonstrate the approach on a YOLOv9 detector fine-tuned for harbor seal detection in Glacier Bay drone imagery, using Segment-Anything-Model-refined masks to support two interventions: (i) object removal/replacement (e.g., replacing seals with plausible ice/water or boats) and (ii) background replacement with original animals composited onto new scenes. Explanations are assessed by re-scoring perturbed images (flip rate, confidence drop) and by expert review for ecological plausibility and interpretability. The resulting explanations localize diagnostic structures, avoid deletion artifacts common to traditional perturbations, and yield domain-relevant insights that support expert validation and more trustworthy deployment of AI in ecology. 

**Abstract (ZH)**: 视觉模型驱动的生态监测日益 automation，然而不透明的预测限制了信任和现场应用。我们提出了一种 inpainting 引导的扰动基解释技术，它生成保场景上下文的、照片级真实的、掩码局部化的编辑。与掩码或模糊不同，这些编辑保持在分布内，并揭示了在物种识别和性状归因等任务中驱动预测的细粒度形态学线索。我们通过使用 Segment-Anything-Model 加工的掩码在 Glaciers Bay 彩鹰无人机图像上对 YOLOv9 检测器进行微调来演示该方法，支持两种干预措施：(i) 对象移除/替换（例如，用可能的冰/水或船只替换海豹），以及 (ii) 背景替换，将原始动物重新组合到新场景中。通过重新评分扰动图像（翻转率、置信度下降）和专家评审生态合理性与可解释性来评估解释。生成的解释定位诊断结构，避免了传统扰动常见的删除伪影，并提供了与领域相关的重要见解，支持专家验证和更具可信度的 AI 在生态学中的部署。 

---
# The View From Space: Navigating Instrumentation Differences with EOFMs 

**Title (ZH)**: 从太空视角导航：EOFMs处理仪器差异 

**Authors**: Ryan P. Demilt, Nicholas LaHaye, Karis Tenneson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03316)  

**Abstract**: Earth Observation Foundation Models (EOFMs) have exploded in prevalence as tools for processing the massive volumes of remotely sensed and other earth observation data, and for delivering impact on the many essential earth monitoring tasks. An emerging trend posits using the outputs of pre-trained models as 'embeddings' which summarize high dimensional data to be used for generic tasks such as similarity search and content-specific queries. However, most EOFM models are trained only on single modalities of data and then applied or benchmarked by matching bands across different modalities. It is not clear from existing work what impact diverse sensor architectures have on the internal representations of the present suite of EOFMs. We show in this work that the representation space of EOFMs is highly sensitive to sensor architecture and that understanding this difference gives a vital perspective on the pitfalls of current EOFM design and signals for how to move forward as model developers, users, and a community guided by robust remote-sensing science. 

**Abstract (ZH)**: 地球观测基础模型（EOFMs）在处理大规模遥感和其他地球观测数据以及执行众多关键地球监测任务方面变得越来越流行。一项新兴趋势是利用预训练模型的输出作为“嵌入”，以总结高维数据，用于通用任务如相似性搜索和专门内容查询。然而，大多数EOFM模型仅在单一模态数据上进行训练，然后通过不同模态之间的波段匹配进行应用或基准测试。现有工作中尚不清楚多样化的传感器架构如何影响当前EOFM模型的内部表示。本文表明，EOFM的表示空间对传感器架构极为敏感，理解这种差异为当前EOFM设计的陷阱提供了宝贵的视角，并指示了作为模型开发者、用户和受到坚实遥感科学指导的社区应如何向前推进。 

---
# Decomposing Attention To Find Context-Sensitive Neurons 

**Title (ZH)**: 分解注意力以找到上下文敏感神经元 

**Authors**: Alex Gibson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03315)  

**Abstract**: We study transformer language models, analyzing attention heads whose attention patterns are spread out, and whose attention scores depend weakly on content. We argue that the softmax denominators of these heads are stable when the underlying token distribution is fixed. By sampling softmax denominators from a "calibration text", we can combine together the outputs of multiple such stable heads in the first layer of GPT2-Small, approximating their combined output by a linear summary of the surrounding text. This approximation enables a procedure where from the weights alone - and a single calibration text - we can uncover hundreds of first layer neurons that respond to high-level contextual properties of the surrounding text, including neurons that didn't activate on the calibration text. 

**Abstract (ZH)**: 我们研究变压器语言模型，分析那些注意力模式分散且注意力分数对内容依赖较弱的注意力头。我们认为，当底层 token 分布固定时，这些头的 softmax 分母是稳定的。通过从“校准文本”中抽取 softmax 分母，我们可以在 GPT2-Small 的第一层结合多个这样的稳定头的输出，通过邻近文本的线性总结来近似它们的综合输出。这种近似使得仅从权重和一个单一的校准文本出发，我们可以发现数百个对邻近文本的高层上下文属性作出响应的第一层神经元，包括那些在校准文本中未激活的神经元。 

---
# A Comprehensive Review on Artificial Intelligence Empowered Solutions for Enhancing Pedestrian and Cyclist Safety 

**Title (ZH)**: 人工智能赋能的增强行人和骑行者安全解决方案综述 

**Authors**: Shucheng Zhang, Yan Shi, Bingzhang Wang, Yuang Zhang, Muhammad Monjurul Karim, Kehua Chen, Chenxi Liu, Mehrdad Nasri, Yinhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03314)  

**Abstract**: Ensuring the safety of vulnerable road users (VRUs), such as pedestrians and cyclists, remains a critical global challenge, as conventional infrastructure-based measures often prove inadequate in dynamic urban environments. Recent advances in artificial intelligence (AI), particularly in visual perception and reasoning, open new opportunities for proactive and context-aware VRU protection. However, existing surveys on AI applications for VRUs predominantly focus on detection, offering limited coverage of other vision-based tasks that are essential for comprehensive VRU understanding and protection. This paper presents a state-of-the-art review of recent progress in camera-based AI sensing systems for VRU safety, with an emphasis on developments from the past five years and emerging research trends. We systematically examine four core tasks, namely detection and classification, tracking and reidentification, trajectory prediction, and intent recognition and prediction, which together form the backbone of AI-empowered proactive solutions for VRU protection in intelligent transportation systems. To guide future research, we highlight four major open challenges from the perspectives of data, model, and deployment. By linking advances in visual AI with practical considerations for real-world implementation, this survey aims to provide a foundational reference for the development of next-generation sensing systems to enhance VRU safety. 

**Abstract (ZH)**: 确保弱势道路交通使用者（VRUs）的安全仍然是一个关键的全球挑战，因为传统的基于基础设施的措施在动态城市环境中往往不够充分。近期人工智能（AI）特别是在视觉感知和推理方面的发展为提前预警和情境感知的VRU保护开启了新的机会。然而，现有的关于AI在VRU应用的综述大多侧重于检测，对全面理解与保护VRU所必需的其他视觉任务关注不足。本文对过去五年中基于相机的AI传感系统在VRU安全领域的最新进展进行了综述，并着重探讨了新兴研究趋势。我们系统地研究了四个核心任务，即检测与分类、跟踪与重识别、轨迹预测以及意图识别与预测，这些任务构成了利用AI增强的主动保护解决方案的核心，用于智能交通系统中的VRU保护。为了指导未来研究，我们从数据、模型和部署三个角度突出了四个主要的开放挑战。通过将视觉AI的进步与实际应用中的考虑事项相结合，本文旨在为开发下一代传感系统以增强VRU安全提供基础参考。 

---
# Predicting Effects, Missing Distributions: Evaluating LLMs as Human Behavior Simulators in Operations Management 

**Title (ZH)**: 预测影响，填补空白分布：评价LLMs在运营管理中作为人类行为模拟器的效果 

**Authors**: Runze Zhang, Xiaowei Zhang, Mingyang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03310)  

**Abstract**: LLMs are emerging tools for simulating human behavior in business, economics, and social science, offering a lower-cost complement to laboratory experiments, field studies, and surveys. This paper evaluates how well LLMs replicate human behavior in operations management. Using nine published experiments in behavioral operations, we assess two criteria: replication of hypothesis-test outcomes and distributional alignment via Wasserstein distance. LLMs reproduce most hypothesis-level effects, capturing key decision biases, but their response distributions diverge from human data, including for strong commercial models. We also test two lightweight interventions -- chain-of-thought prompting and hyperparameter tuning -- which reduce misalignment and can sometimes let smaller or open-source models match or surpass larger systems. 

**Abstract (ZH)**: LLMs在运营管理中模拟人类行为的表现评价：基于九项行为运营管理实验的评估 

---
# Creative synthesis of kinematic mechanisms 

**Title (ZH)**: 创造性合成运动学机构 

**Authors**: Jiong Lin, Jialong Ning, Judah Goldfeder, Hod Lipson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03308)  

**Abstract**: In this paper, we formulate the problem of kinematic synthesis for planar linkages as a cross-domain image generation task. We develop a planar linkages dataset using RGB image representations, covering a range of mechanisms: from simple types such as crank-rocker and crank-slider to more complex eight-bar linkages like Jansen's mechanism. A shared-latent variational autoencoder (VAE) is employed to explore the potential of image generative models for synthesizing unseen motion curves and simulating novel kinematics. By encoding the drawing speed of trajectory points as color gradients, the same architecture also supports kinematic synthesis conditioned on both trajectory shape and velocity profiles. We validate our method on three datasets of increasing complexity: a standard four-bar linkage set, a mixed set of four-bar and crank-slider mechanisms, and a complex set including multi-loop mechanisms. Preliminary results demonstrate the effectiveness of image-based representations for generative mechanical design, showing that mechanisms with revolute and prismatic joints, and potentially cams and gears, can be represented and synthesized within a unified image generation framework. 

**Abstract (ZH)**: 本文将平面连杆机构的运动合成问题形式化为一个跨域图像生成任务。利用RGB图像表示构建平面连杆机构数据集，涵盖了从简单类型如曲柄摇杆和曲柄滑块机构到更复杂的八连杆机构如詹森机制等各类机制。采用共享潜在空间的变分自编码器（VAE），探索图像生成模型在合成未见过的运动曲线和模拟新型运动学方面的潜力。通过将轨迹点的绘制速度编码为颜色梯度，相同的架构还支持基于轨迹形状和速度分布的运动合成。我们在三个逐步复杂的数据集上验证了该方法：标准四连杆机构集、四连杆与曲柄滑块机构混合集，以及包含多环机构的复杂集。初步结果表明，基于图像的表示在生成机械设计中有效，展示了旋转铰链和平移铰链机制，以及可能的凸轮和齿轮机制，可以在统一的图像生成框架中表示和合成。 

---
# Atlas-free Brain Network Transformer 

**Title (ZH)**: 无图谱脑网络变换器 

**Authors**: Shuai Huang, Xuan Kan, James J. Lah, Deqiang Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03306)  

**Abstract**: Current atlas-based approaches to brain network analysis rely heavily on standardized anatomical or connectivity-driven brain atlases. However, these fixed atlases often introduce significant limitations, such as spatial misalignment across individuals, functional heterogeneity within predefined regions, and atlas-selection biases, collectively undermining the reliability and interpretability of the derived brain networks. To address these challenges, we propose a novel atlas-free brain network transformer (atlas-free BNT) that leverages individualized brain parcellations derived directly from subject-specific resting-state fMRI data. Our approach computes ROI-to-voxel connectivity features in a standardized voxel-based feature space, which are subsequently processed using the BNT architecture to produce comparable subject-level embeddings. Experimental evaluations on sex classification and brain-connectome age prediction tasks demonstrate that our atlas-free BNT consistently outperforms state-of-the-art atlas-based methods, including elastic net, BrainGNN, Graphormer and the original BNT. Our atlas-free approach significantly improves the precision, robustness, and generalizability of brain network analyses. This advancement holds great potential to enhance neuroimaging biomarkers and clinical diagnostic tools for personalized precision medicine. 

**Abstract (ZH)**: 一种基于个体化脑区划分的无图谱脑网络变换器（ atlas-free BNT） 

---
# Dynamic Meta-Learning for Adaptive XGBoost-Neural Ensembles 

**Title (ZH)**: 动态元学习适配XGBoost-神经集成模型 

**Authors**: Arthur Sedek  

**Link**: [PDF](https://arxiv.org/pdf/2510.03301)  

**Abstract**: This paper introduces a novel adaptive ensemble framework that synergistically combines XGBoost and neural networks through sophisticated meta-learning. The proposed method leverages advanced uncertainty quantification techniques and feature importance integration to dynamically orchestrate model selection and combination. Experimental results demonstrate superior predictive performance and enhanced interpretability across diverse datasets, contributing to the development of more intelligent and flexible machine learning systems. 

**Abstract (ZH)**: 本文介绍了一种新颖的适应性集成框架，该框架通过精巧的元学习协同结合XGBoost和神经网络。所提出的方法利用高级不确定性量化技术和特征重要性集成，动态协调模型选择与组合。实验结果表明，该方法在多种数据集上展现出优越的预测性能和增强的可解释性，促进了更智能和灵活的机器学习系统的开发。 

---
# Convolutional Neural Nets vs Vision Transformers: A SpaceNet Case Study with Balanced vs Imbalanced Regimes 

**Title (ZH)**: 卷积神经网络vs视觉变换器：关于平衡状态与不平衡状态的SpaceNet案例研究 

**Authors**: Akshar Gothi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03297)  

**Abstract**: We present a controlled comparison of a convolutional neural network (EfficientNet-B0) and a Vision Transformer (ViT-Base) on SpaceNet under two label-distribution regimes: a naturally imbalanced five-class split and a balanced-resampled split with 700 images per class (70:20:10 train/val/test). With matched preprocessing (224x224, ImageNet normalization), lightweight augmentations, and a 40-epoch budget on a single NVIDIA P100, we report accuracy, macro-F1, balanced accuracy, per-class recall, and deployment metrics (model size and latency). On the imbalanced split, EfficientNet-B0 reaches 93% test accuracy with strong macro-F1 and lower latency; ViT-Base is competitive at 93% with a larger parameter count and runtime. On the balanced split, both models are strong; EfficientNet-B0 reaches 99% while ViT-Base remains competitive, indicating that balancing narrows architecture gaps while CNNs retain an efficiency edge. We release manifests, logs, and per-image predictions to support reproducibility. 

**Abstract (ZH)**: 我们在SpaceNet上对卷积神经网络（EfficientNet-B0）和视觉变压器（ViT-Base）在两种标签分布制度下的表现进行了受控比较：自然不平衡的五分类划分和重采样的平衡划分（每类700张图像，70:20:10训练/验证/测试）。通过匹配的预处理（224x224，ImageNet归一化）、轻量级的数据增强并在单个NVIDIA P100上运行40个epoch，我们报告了准确率、宏F1值、均衡准确率、每类召回率以及部署指标（模型大小和延迟）。在不平衡划分中，EfficientNet-B0达到93%的测试准确率，宏F1值较强且延迟较低；ViT-Base在93%的准确率上具有竞争力，但参数量和运行时间更大。在平衡划分中，两种模型表现强劲；EfficientNet-B0达到99%，而ViT-Base保持竞争力，这表明平衡化缩小了架构差距而CNN在效率方面仍占优势。我们发布了元数据、日志和单张图像预测以支持可重复性。 

---
# From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing 

**Title (ZH)**: 从分数分布到平衡：即插即用专家混合路由 

**Authors**: Rana Shahout, Colin Cai, Yilun Du, Minlan Yu, Michael Mitzenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2510.03293)  

**Abstract**: Mixture-of-Experts (MoE) models can scale parameter capacity by routing each token to a subset of experts through a learned gate function. While conditional routing reduces training costs, it shifts the burden on inference memory: expert parameters and activations consume memory, limiting the number of experts per device. As tokens are routed, some experts become overloaded while others are underutilized. Because experts are mapped to GPUs, this imbalance translates directly into degraded system performance in terms of latency, throughput, and cost. We present LASER, a plug-and-play, inference-time routing algorithm that balances load while preserving accuracy. LASER adapts to the shape of the gate's score distribution. When scores provide a clear preference, it routes to the strongest experts; when scores are more uniform, it broadens the set of viable experts and routes to the least-loaded among them. Because LASER relies only on gate scores from a trained model, it integrates directly into existing MoE inference pipelines without retraining or finetuning. We evaluate LASER on Mixtral-8x7B and DeepSeek-MoE-16b-chat across four datasets (ARC-Easy, ARC-Challenge, MMLU, and GSM8K). LASER improves load balancing, translating into lower latency and higher throughput, while keeping the accuracy changes negligible. 

**Abstract (ZH)**: 基于专家混合的插件式推理时间路由算法LASER：保持准确性的负载均衡方法 

---
# UniPruning: Unifying Local Metric and Global Feedback for Scalable Sparse LLMs 

**Title (ZH)**: UniPruning: 统一局部度量与全局反馈以实现可扩展的稀疏大语言模型 

**Authors**: Yizhuo Ding, Wanying Qu, Jiawei Geng, Wenqi Shao, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03291)  

**Abstract**: Large Language Models (LLMs) achieve strong performance across diverse tasks but face prohibitive computational and memory costs. Pruning offers a promising path by inducing sparsity while preserving architectural flexibility. However, existing methods struggle to balance efficiency and robustness: local metric approaches prune layer by layer but often collapse under high sparsity, whereas global feedback methods enforce consistency at the cost of expensive weight updates or restrictive semi-structured formats. We present UniPruning, a unified post-training pruning framework that combines the speed of local saliency metrics with the stability of global coordination, enabled by a mirror descent based optimization, all without updating model weights. UniPruning leverages fast layer-wise scoring and a lightweight global controller to allocate a single sparsity budget, supporting both unstructured and semi-structured N :M pruning within one framework. After a brief calibration, it can generate pruning masks for arbitrary sparsity levels in one shot, and adapts seamlessly to hardware-aware constraints. Extensive experiments on multiple pretrained LLM families and standard benchmarks show that UniPruning consistently delivers competitive or superior perplexity and zero-shot accuracy. Ablation studies further highlight the importance of mirror descent and local saliency anchoring. Overall, UniPruning provides an efficient, principled, and scalable solution for sparsifying large-scale LLMs. Our code is available at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多样化的任务上表现出色，但面临计算和内存成本的限制。剪枝通过引入稀疏性同时保持架构灵活性，提供了一条有希望的路径。然而，现有方法在效率和鲁棒性之间难以兼顾：局部度量方法逐层修剪，但在高稀疏性下容易崩溃；而全局反馈方法通过一致性的要求，导致昂贵的权重更新或限制性的半结构化格式。我们提出了UniPruning，这是一种统一的后训练剪枝框架，结合了局部显著性度量的速度与全局协调的稳定性，通过基于镜像下降的优化实现，而不更新模型权重。UniPruning 利用快速逐层评分和轻量级的全局控制器分配单一稀疏性预算，支持统一框架内的无结构和半结构化 N:M 修剪。经过短暂的校准后，它可以一次性生成任意稀疏性级别的剪枝掩码，并无缝适应硬件感知约束。在多个预训练 LLM 家族和标准基准上的广泛实验表明，UniPruning 一致地提供了可竞争或更优的困惑度和零样本准确性。进一步的消融研究强调了镜像下降和局部显著性锚定的重要性。总体而言，UniPruning 提供了一种高效、原理性强且可扩展的大型语言模型稀疏化解决方案。我们的代码可在以下链接获取：此链接处。 

---
# Why mask diffusion does not work 

**Title (ZH)**: 为什么面纱扩散不起作用 

**Authors**: Haocheng Sun, Cynthia Xin Wen, Edward Hong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03289)  

**Abstract**: The main advantages of diffusion language models over autoregressive (AR) models lie in their ability to support parallel generation and bidirectional attention, enabling a more controllable generation process. In recent years, open-source mask diffusion language models have emerged, most of which are based on a variant known as absorbing diffusion. However, this paper demonstrates why mask diffusion faces inherent difficulties in achieving parallel generation and bidirectional attention. We also propose the most effective training and inference strategies for mask diffusion. 

**Abstract (ZH)**: 扩散语言模型相较于自回归模型的主要优势在于其支持并行生成和双向注意力的能力，从而实现更可控的生成过程。近年来，开源掩码扩散语言模型逐渐涌现，大多数基于一种称为吸收扩散的变种。然而，本文阐述了掩码扩散在实现并行生成和双向注意力方面面临的固有困难。我们还提出了掩码扩散最为有效的训练和推理策略。 

---
# LogAction: Consistent Cross-system Anomaly Detection through Logs via Active Domain 

**Title (ZH)**: LogAction: 通过主动领域一致跨系统异常检测 

**Authors**: Chiming Duan, Minghua He, Pei Xiao, Tong Jia, Xin Zhang, Zhewei Zhong, Xiang Luo, Yan Niu, Lingzhe Zhang, Yifan Wu, Siyu Yu, Weijie Hong, Ying Li, Gang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03288)  

**Abstract**: Log-based anomaly detection is a essential task for ensuring the reliability and performance of software systems. However, the performance of existing anomaly detection methods heavily relies on labeling, while labeling a large volume of logs is highly challenging. To address this issue, many approaches based on transfer learning and active learning have been proposed. Nevertheless, their effectiveness is hindered by issues such as the gap between source and target system data distributions and cold-start problems. In this paper, we propose LogAction, a novel log-based anomaly detection model based on active domain adaptation. LogAction integrates transfer learning and active learning techniques. On one hand, it uses labeled data from a mature system to train a base model, mitigating the cold-start issue in active learning. On the other hand, LogAction utilize free energy-based sampling and uncertainty-based sampling to select logs located at the distribution boundaries for manual labeling, thus addresses the data distribution gap in transfer learning with minimal human labeling efforts. Experimental results on six different combinations of datasets demonstrate that LogAction achieves an average 93.01% F1 score with only 2% of manual labels, outperforming some state-of-the-art methods by 26.28%. Website: this https URL 

**Abstract (ZH)**: 基于日志的异常检测是确保软件系统可靠性和性能的重要任务。然而，现有异常检测方法的性能高度依赖于标签，而大量日志的标注极具挑战性。为应对这一问题，提出了多种基于迁移学习和活跃学习的方法。然而，这些方法的有效性受限于源系统和目标系统数据分布之间的差距以及冷启动问题。在本文中，我们提出了LogAction，一种基于活跃域适应的新型基于日志的异常检测模型。LogAction结合了迁移学习和活跃学习技术。一方面，它使用成熟系统的标注数据训练基模型，缓解活跃学习中的冷启动问题；另一方面，LogAction利用自由能采样和不确定性采样选择位于分布边界上的日志进行人工标注，从而在最小的人工标注努力下解决迁移学习中的数据分布差距问题。六组不同数据集的实验结果显示，LogAction仅使用2%的手标注数据即可获得平均93.01%的F1分数，优于一些最先进的方法26.28%。网站: 这个 https URL。 

---
# A Biologically Interpretable Cognitive Architecture for Online Structuring of Episodic Memories into Cognitive Maps 

**Title (ZH)**: 一种生物可解释的认知架构：用于在线将 episodic 记忆结构化为认知地图 

**Authors**: E.A. Dzhivelikian, A.I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2510.03286)  

**Abstract**: Cognitive maps provide a powerful framework for understanding spatial and abstract reasoning in biological and artificial agents. While recent computational models link cognitive maps to hippocampal-entorhinal mechanisms, they often rely on global optimization rules (e.g., backpropagation) that lack biological plausibility. In this work, we propose a novel cognitive architecture for structuring episodic memories into cognitive maps using local, Hebbian-like learning rules, compatible with neural substrate constraints. Our model integrates the Successor Features framework with episodic memories, enabling incremental, online learning through agent-environment interaction. We demonstrate its efficacy in a partially observable grid-world, where the architecture autonomously organizes memories into structured representations without centralized optimization. This work bridges computational neuroscience and AI, offering a biologically grounded approach to cognitive map formation in artificial adaptive agents. 

**Abstract (ZH)**: 认知地图提供了一种强大的框架，用于理解生物和人工代理的空间和抽象推理。虽然最近的计算模型将认知地图与海马-Entorhinal机制联系起来，但它们往往依赖于全局优化规则（如反向传播），这些规则缺乏生物可行性。在本工作中，我们提出了一种新型的认知架构，用于使用局部、类似Hebb的学习规则将 episodic 记忆结构化为认知地图，该规则与神经元结构限制兼容。我们的模型将 Successor Features 框架与 episodic 记忆结合起来，通过代理与环境的交互实现增量、在线学习。我们在一个部分可观测的网格世界中展示了其有效性，其中架构在没有集中优化的情况下自主地将记忆组织成结构化表示。本工作将计算神经科学与人工智能相结合，提供了一种基于生物学的方法来形成人工自适应代理的认知地图。 

---
# Edge-FIT: Federated Instruction Tuning of Quantized LLMs for Privacy-Preserving Smart Home Environments 

**Title (ZH)**: Edge-FIT：联邦指令调优的量化大语言模型在隐私保护智能家居环境中的应用 

**Authors**: Vinay Venkatesh, Vamsidhar R Kamanuru, Lav Kumar, Nikita Kothari  

**Link**: [PDF](https://arxiv.org/pdf/2510.03284)  

**Abstract**: This paper proposes Edge-FIT (Federated Instruction Tuning on the Edge), a scalable framework for Federated Instruction Tuning (FIT) of Large Language Models (LLMs). Traditional Federated Learning (TFL) methods, like FedAvg, fail when confronted with the massive parameter size of LLMs [3], [6]. Our Edge-FIT framework combines federated learning with 4-bit Quantized Low-Rank Adaptation (QLORA), mitigating the core issues of communication and computational overhead. We demonstrate this by filtering the general-purpose Databricks Dolly 15k dataset for the IoT domain. Experimental results show the Edge-FIT tuned Llama 2(7B) achieves an F1-Score of 0.89. We also demonstrate a viable trade-off using the 3.8B Phi-3-mini model, validating Edge-FIT as a scalable framework for decentralized LLM deployment on home compute gateways. 

**Abstract (ZH)**: Edge-FIT（边缘端联邦指令调优）：大规模语言模型联邦指令调优的可扩展框架 

---
# MACE: A Hybrid LLM Serving System with Colocated SLO-aware Continuous Retraining Alignment 

**Title (ZH)**: MACE：一种集成SLO感知连续重训对齐的混合大语言模型服务系统 

**Authors**: Yufei Li, Yu Fu, Yue Dong, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03283)  

**Abstract**: Large language models (LLMs) deployed on edge servers are increasingly used in latency-sensitive applications such as personalized assistants, recommendation, and content moderation. However, the non-stationary nature of user data necessitates frequent retraining, which introduces a fundamental tension between inference latency and model accuracy under constrained GPU resources. Existing retraining strategies either delay model updates, over-commit resources to retraining, or overlook iteration-level retraining granularity. In this paper, we identify that iteration-level scheduling is crucial for adapting retraining frequency to model drift without violating service-level objectives (SLOs). We propose MACE, a hybrid LLM system that colocates concurrent inference (prefill, decode) and fine-tuning, with intelligent memory management to maximize task performance while promising inference throughput. MACE leverages the insight that not all model updates equally affect output alignment and allocates GPU cycles accordingly to balance throughput, latency, and update freshness. Our trace-driven evaluation shows that MACE matches or exceeds continuous retraining while reducing inference latency by up to 63% and maintaining throughput under resource constraints. Compared to periodic retraining, MACE improves latency breakdown across prefill, decode, and finetune stages, and sustains GPU utilization above 85% in NVIDIA AGX Orin. These results demonstrate that iteration-level hybrid scheduling is a promising direction for deploying LLMs with continual learning capabilities on edge platforms. 

**Abstract (ZH)**: 边缘服务器上部署的大语言模型（LLMs）越来越多地应用于如个性化助手、推荐和内容审核等低延迟应用。然而，用户数据的非平稳特性要求频繁重训，这在受限的GPU资源下引入了推理延迟与模型准确性的根本矛盾。现有重训策略要么延迟模型更新，要么过度分配资源给重训，要么忽略迭代级别的重训粒度。在本文中，我们发现迭代级别的调度对于适应模型漂移并同时满足服务级目标（SLOs）至关重要。我们提出了一种名为MACE的混合LLM系统，该系统将并发推理（预填充、解码）和微调相结合，并通过智能内存管理以最大化任务性能和保证推理吞吐量。MACE 运用了这样一个洞察：并非所有模型更新都等量地影响输出对齐，并相应地分配GPU周期以平衡吞吐量、延迟和更新及时性。我们的基于跟踪的评估结果显示，MACE 在减少推理延迟高达63%的同时，能够满足资源限制下的吞吐量要求，而不连续重训。与周期性重训相比，MACE 在预填充、解码和微调各个阶段均改善了延迟分解，并在NVIDIA AGX Orin上保持了GPU利用率高于85%。这些结果表明，在边缘平台上部署具备持续学习能力的大语言模型时，迭代级别的混合调度是一种有前景的方向。 

---
# Training Optimal Large Diffusion Language Models 

**Title (ZH)**: 训练最优大型扩散语言模型 

**Authors**: Jinjie Ni, Qian Liu, Chao Du, Longxu Dou, Hang Yan, Zili Wang, Tianyu Pang, Michael Qizhe Shieh  

**Link**: [PDF](https://arxiv.org/pdf/2510.03280)  

**Abstract**: We introduce Quokka, the first systematic scaling law for diffusion language models (DLMs), encompassing both compute-constrained and data-constrained regimes, and studying the key modeling and optimization designs. Quokka is a good friend of Chinchilla and provides wider scopes. We hope the results would bring short-term practical guidance in DLMs training and long-term inspirations for the whole AI community. 

**Abstract (ZH)**: 我们介绍了Quokka，这是首个系统性的扩散语言模型（DLM）的扩展规律，涵盖了计算受限和数据受限两种情况，并研究了关键建模和优化设计。Quokka是Chinchilla的好朋友，提供了更广泛的研究范围。我们希望这些结果能够在DLMs训练方面提供短期的实际指导，并为整个AI社区带来长期的启发。 

---
# MemMamba: Rethinking Memory Patterns in State Space Model 

**Title (ZH)**: MemMamba: 在状态空间模型中重思内存模式 

**Authors**: Youjin Wang, Yangjingyi Chen, Jiahao Yan, Jiaxuan Lu, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.03279)  

**Abstract**: With the explosive growth of data, long-sequence modeling has become increasingly important in tasks such as natural language processing and bioinformatics. However, existing methods face inherent trade-offs between efficiency and memory. Recurrent neural networks suffer from gradient vanishing and explosion, making them hard to scale. Transformers can model global dependencies but are constrained by quadratic complexity. Recently, selective state-space models such as Mamba have demonstrated high efficiency with O(n) time and O(1) recurrent inference, yet their long-range memory decays exponentially. In this work, we conduct mathematical derivations and information-theoretic analysis to systematically uncover the memory decay mechanism of Mamba, answering a fundamental question: what is the nature of Mamba's long-range memory and how does it retain information? To quantify key information loss, we further introduce horizontal-vertical memory fidelity metrics that capture degradation both within and across layers. Inspired by how humans distill and retain salient information when reading long documents, we propose MemMamba, a novel architectural framework that integrates state summarization mechanism together with cross-layer and cross-token attention, which alleviates long-range forgetting while preserving linear complexity. MemMamba achieves significant improvements over existing Mamba variants and Transformers on long-sequence benchmarks such as PG19 and Passkey Retrieval, while delivering a 48% speedup in inference efficiency. Both theoretical analysis and empirical results demonstrate that MemMamba achieves a breakthrough in the complexity-memory trade-off, offering a new paradigm for ultra-long sequence modeling. 

**Abstract (ZH)**: 随着数据的爆炸性增长，长序列 modeling 在自然语言处理和生物信息学等任务中变得越来越重要。然而，现有方法在效率和内存之间存在固有的权衡。循环神经网络遭受梯度消失和爆炸的问题，难以扩展。变压器可以建模全局依赖关系，但受到二次复杂度的限制。最近，如 Mamba 等选择性状态空间模型展示了 O(n) 时间和 O(1) 递归推理的高度效率，但其长期记忆呈现指数衰减。在本文中，我们通过数学推导和信息论分析系统地揭示了 Mamba 的记忆衰减机制，回答了一个基本问题：Mamba 的长期记忆的本质是什么，它是如何保留信息的？为了量化关键信息损失，我们进一步引入了水平-垂直记忆保真度指标，以捕获层内和跨层的退化。借鉴人类在阅读长文档时提取和保留关键信息的方式，我们提出了一种新的架构框架 MemMamba，该框架结合了状态汇总机制和跨层及跨标记注意力，从而减轻长期遗忘现象，同时保持线性复杂度。MemMamba 在 PG19 和 Passkey Retrieval 等长序列基准测试中显著优于现有 Mamba 变体和 Transformers，且推理效率提升 48%。理论分析和实验结果都表明，MemMamba 在复杂性-内存权衡中实现了突破，为超长序列 modeling 提供了一个新的范式。 

---
# Quantifying constraint hierarchies in Bayesian PINNs via per-constraint Hessian decomposition 

**Title (ZH)**: 通过每约束海森矩阵分解量化贝叶斯PINNs中的约束层次结构 

**Authors**: Filip Landgren  

**Link**: [PDF](https://arxiv.org/pdf/2510.03278)  

**Abstract**: Bayesian physics-informed neural networks (B-PINNs) merge data with governing equations to solve differential equations under uncertainty. However, interpreting uncertainty and overconfidence in B-PINNs requires care due to the poorly understood effects the physical constraints have on the network; overconfidence could reflect warranted precision, enforced by the constraints, rather than miscalibration. Motivated by the need to further clarify how individual physical constraints shape these networks, we introduce a scalable, matrix-free Laplace framework that decomposes the posterior Hessian into contributions from each constraint and provides metrics to quantify their relative influence on the loss landscape. Applied to the Van der Pol equation, our method tracks how constraints sculpt the network's geometry and shows, directly through the Hessian, how changing a single loss weight non-trivially redistributes curvature and effective dominance across the others. 

**Abstract (ZH)**: 基于贝叶斯的物理约束神经网络（B-PINNs）将数据与 governing 方程合并以在不确定性条件下求解微分方程。然而，由于物理约束对网络影响的不明确性，解释 B-PINNs 中的不确定性与过度自信需要谨慎；过度自信可能反映了由约束施加的适当精度，而非校准不当。为更清晰地理解单个物理约束如何影响这些网络，我们引入了一种可扩展的、无需矩阵运算的拉普拉斯框架，该框架将后验哈密顿量分解为每个约束的贡献，并提供量化其在损失景观上相对影响的指标。该方法应用于范德蒙德方程，跟踪约束如何塑造网络几何结构，并直接通过哈密顿量展示了改变单个损失权重非平凡地重新分配曲率和有效支配的方式。 

---
# QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks 

**Title (ZH)**: QuadEnhancer: 利用二次变换增强深度神经网络 

**Authors**: Qian Chen, Linxin Yang, Akang Wang, Xiaodong Luo, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03276)  

**Abstract**: The combination of linear transformations and non-linear activation functions forms the foundation of most modern deep neural networks, enabling them to approximate highly complex functions. This paper explores the introduction of quadratic transformations to further increase nonlinearity in neural networks, with the aim of enhancing the performance of existing architectures. To reduce parameter complexity and computational complexity, we propose a lightweight quadratic enhancer that uses low-rankness, weight sharing, and sparsification techniques. For a fixed architecture, the proposed approach introduces quadratic interactions between features at every layer, while only adding negligible amounts of additional model parameters and forward computations. We conduct a set of proof-of-concept experiments for the proposed method across three tasks: image classification, text classification, and fine-tuning large-language models. In all tasks, the proposed approach demonstrates clear and substantial performance gains. 

**Abstract (ZH)**: 线性变换与非线性激活函数的结合构成了大多数现代深度神经网络的基础，使它们能够逼近高度复杂的功能。本文探讨引入二次变换以进一步增加神经网络的非线性，旨在提升现有架构的表现。为减少参数复杂度和计算复杂度，我们提出了一种轻量级的二次增强器，利用低秩性、权重共享和稀疏化技术。对于固定架构，所提出的方法在每一层都引入了特征间的二次相互作用，同时仅增加了可忽略不计的额外模型参数和前向计算量。我们在图像分类、文本分类和大型语言模型微调三个任务中进行了所提出方法的概念验证实验。在所有任务中，所提出的方法均表现出明确且显著的性能提升。 

---
# SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size 

**Title (ZH)**: SDQ-LLM：Sigma-Delta量化用于任意规模的1位LLM 

**Authors**: Junhao Xia, Ming Zhao, Limin Xiao, Xiujun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03275)  

**Abstract**: Large language models (LLMs) face significant computational and memory challenges, making extremely low-bit quantization crucial for their efficient deployment. In this work, we introduce SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size, a novel framework that enables extremely low-bit quantization of LLMs while preserving their linguistic reasoning capabilities. A distinctive feature of SDQ-LLM is the continuous adjustability of the Over-Sampling Ratio (OSR), enabling dynamic adaptation to memory or VRAM constraints by selecting fractional OSR (e.g. 2.5 times) for an optimal trade-off between model size and accuracy. SDQ-LLM uses upsampling combined with Sigma-Delta Quantizer to binarize or ternarize LLMs weights, encoding high-precision parameters into 1-bit or 1.58-bit representations, replacing the multiplication operations within linear layers with addition. This approach significantly enhances inference efficiency under extremely low-bit quantization. To further reduce the loss of quantization precision, we incorporate Hadamard-based weight smoothing prior to quantization, improving the stability and robustness of the weight representations. Furthermore, to fully leverage the continuity of the OSR and reduce precision loss, recognizing the correlation between quantization sensitivity and weight variance, we propose a fine-grained, layer- and linear-wise OSR allocation strategy, MultiOSR. This strategy distributes OSR both across layers and within each layer, based on weight variance and parameter scale. Finally, extensive experiments on OPT and LLaMA model families demonstrate that SDQ-LLM achieves a more efficient and high-precision performance even under highly aggressive low-OSR settings. Our code is available at this https URL. 

**Abstract (ZH)**: Sigma-Delta Quantization for 1-bit LLMs of Any Size: SDQ-LLM及其框架 

---
# Quant-dLLM: Post-Training Extreme Low-Bit Quantization for Diffusion Large Language Models 

**Title (ZH)**: Quant-dLLM：训练后极端低比特量化 diffusion 大型语言模型 

**Authors**: Tianao Zhang, Zhiteng Li, Xianglong Yan, Haotong Qin, Yong Guo, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03274)  

**Abstract**: Diffusion large language models (dLLMs), which offer bidirectional context and flexible masked-denoising generation, are emerging as a compelling alternative to autoregressive (AR) LLMs. However, like AR LLMs, their model sizes continue to grow, motivating weight compression for deployment. Although post-training quantization (PTQ) is effective for AR LLMs, directly transferring it to dLLMs at 2-bit leads to unsatisfactory performance. To tackle these challenges, we propose Quant-dLLM, an ultra-low-bit PTQ framework tailored to dLLMs. Since masked-denoising activations in dLLMs differ from the fully visible signals assumed by standard PTQ methods, we introduce Masked Calibration Simulation (MCS) to align calibration with the timestep-dependent masking, which yields more reliable calibrations. Moreover, we propose a Data-aware Any-order Quantizer (DAQ) that learns ultra-low-bit weight representations via an optimization algorithm. It performs iterative approximation guided by our simulated calibration data. In addition, under a strict 2-bit budget, we introduce Adaptive Blockwise Mixed Precision (ABMP), a sensitivity-based precision allocation scheme that adaptively assigns bit width across channel groups. When restricted to 2-bit precision, Quant-dLLM consistently achieves higher accuracy than state-of-the-art (SOTA) AR-transfer PTQ methods on dLLMs. The code and models will be available at: this https URL. 

**Abstract (ZH)**: 面向扩散大型语言模型的超低比特后训练量化框架 

---
# Learning without Global Backpropagation via Synergistic Information Distillation 

**Title (ZH)**: 无需全局反向传播的协同信息蒸馏学习 

**Authors**: Chenhao Ye, Ming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03273)  

**Abstract**: Backpropagation (BP), while foundational to deep learning, imposes two critical scalability bottlenecks: update locking, where network modules remain idle until the entire backward pass completes, and high memory consumption due to storing activations for gradient computation. To address these limitations, we introduce Synergistic Information Distillation (SID), a novel training framework that reframes deep learning as a cascade of local cooperative refinement problems. In SID, a deep network is structured as a pipeline of modules, each imposed with a local objective to refine a probabilistic belief about the ground-truth target. This objective balances fidelity to the target with consistency to the belief from its preceding module. By decoupling the backward dependencies between modules, SID enables parallel training and hence eliminates update locking and drastically reduces memory requirements. Meanwhile, this design preserves the standard feed-forward inference pass, making SID a versatile drop-in replacement for BP. We provide a theoretical foundation, proving that SID guarantees monotonic performance improvement with network depth. Empirically, SID consistently matches or surpasses the classification accuracy of BP, exhibiting superior scalability and pronounced robustness to label this http URL is available at: this https URL 

**Abstract (ZH)**: 反向传播（BP）尽管是深度学习的基础，但其仍面临两大关键的可扩展性瓶颈：更新锁定，即网络模块在完整反向传播完成前处于闲置状态，以及由于用于梯度计算而产生的高内存消耗。为解决这些限制，我们引入了协同信息精炼（SID）这一新的训练框架，将深度学习重新定义为一系列本地协同精炼问题的级联过程。在SID中，一个深网络被构建成模块流水线的形式，每个模块承载一个局部目标，以精确一种关于真实目标的概率信念。该目标平衡了对目标的忠实度和与前一个模块信念的一致性。通过解除模块之间的反向依赖关系，SID使训练过程能够并行进行，从而消除了更新锁定并极大地减少了内存需求。同时，这种设计保留了标准的前向推理过程，使得SID成为一个通用的BP替代方案。我们提供了理论基础，证明SID随网络深度增加能保证性能的单调改进。实验证明，SID在分类准确性上始终与BP相符或超越，表现出更好的可扩展性和显著的鲁棒性。更多详情请参阅：https://thishttpURL/isavailableat:https://thishttpsURL 

---
# PDE-Transformer: A Continuous Dynamical Systems Approach to Sequence Modeling 

**Title (ZH)**: PDE-Transformer：连续动力系统在序列建模中的应用 

**Authors**: Yukun Zhang, Xueqing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03272)  

**Abstract**: The Transformer architecture has revolutionized artificial intelligence, yet a principled theoretical understanding of its internal mechanisms remains elusive. This paper introduces a novel analytical framework that reconceptualizes the Transformer's discrete, layered structure as a continuous spatiotemporal dynamical system governed by a master Partial Differential Equation (PDE). Within this paradigm, we map core architectural components to distinct mathematical operators: self-attention as a non-local interaction, the feed-forward network as a local reaction, and, critically, residual connections and layer normalization as indispensable stabilization mechanisms. We do not propose a new model, but rather employ the PDE system as a theoretical probe to analyze the mathematical necessity of these components. By comparing a standard Transformer with a PDE simulator that lacks explicit stabilizers, our experiments provide compelling empirical evidence for our central thesis. We demonstrate that without residual connections, the system suffers from catastrophic representational drift, while the absence of layer normalization leads to unstable, explosive training dynamics. Our findings reveal that these seemingly heuristic "tricks" are, in fact, fundamental mathematical stabilizers required to tame an otherwise powerful but inherently unstable continuous system. This work offers a first-principles explanation for the Transformer's design and establishes a new paradigm for analyzing deep neural networks through the lens of continuous dynamics. 

**Abstract (ZH)**: Transformer架构改变了人工智能领域，但对其内部机制的原理性理论理解仍然模糊。本文引入了一个新的分析框架，将Transformer的离散分层结构重新概念化为由主偏微分方程（PDE）控制的连续空间-时间动力系统。在这种范式下，我们将核心架构组件映射到不同的数学运算符：自我注意作为一种非局部交互，前馈网络作为一种局部反应，并且至关重要地，残差连接和层归一化作为必不可少的稳定机制。我们并非提出新的模型，而是利用PDE系统作为理论探针来分析这些组件的必要性。通过将标准Transformer与缺乏显式稳定器的PDE模拟器进行比较，我们的实验提供了强有力的经验证据支持我们的核心论点。我们证明，在没有残差连接的情况下，系统会遭受灾难性的表示迁移，而在没有层归一化的情况下则导致不稳定的、爆炸性的训练动力学。我们的发现揭示，这些看似启发式的“技巧”实际上是必要且根本的数学稳定机制，使一个原本强大但本质上不稳定的连续系统变得可控。这项工作为Transformer的设计提供了一种第一性原理解释，并确立了一种通过连续动力学视角分析深度神经网络的新范式。 

---
# Decision Potential Surface: A Theoretical and Practical Approximation of LLM's Decision Boundary 

**Title (ZH)**: 决策潜力面：LLM决策边界的一种理论与实践近似方法 

**Authors**: Zi Liang, Zhiyao Wu, Haoyang Shang, Yulin Jin, Qingqing Ye, Huadi Zheng, Peizhao Hu, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03271)  

**Abstract**: Decision boundary, the subspace of inputs where a machine learning model assigns equal classification probabilities to two classes, is pivotal in revealing core model properties and interpreting behaviors. While analyzing the decision boundary of large language models (LLMs) has raised increasing attention recently, constructing it for mainstream LLMs remains computationally infeasible due to the enormous vocabulary-sequence sizes and the auto-regressive nature of LLMs. To address this issue, in this paper we propose Decision Potential Surface (DPS), a new notion for analyzing LLM decision boundary. DPS is defined on the confidences in distinguishing different sampling sequences for each input, which naturally captures the potential of decision boundary. We prove that the zero-height isohypse in DPS is equivalent to the decision boundary of an LLM, with enclosed regions representing decision regions. By leveraging DPS, for the first time in the literature, we propose an approximate decision boundary construction algorithm, namely $K$-DPS, which only requires K-finite times of sequence sampling to approximate an LLM's decision boundary with negligible error. We theoretically derive the upper bounds for the absolute error, expected error, and the error concentration between K-DPS and the ideal DPS, demonstrating that such errors can be trade-off with sampling times. Our results are empirically validated by extensive experiments across various LLMs and corpora. 

**Abstract (ZH)**: 决策潜能面（DPS）：一种分析大规模语言模型决策边界的新范式 

---
# CoDA: Coding LM via Diffusion Adaptation 

**Title (ZH)**: CoDA: 通过扩散适应进行编码LM 

**Authors**: Haolin Chen, Shiyu Wang, Can Qin, Bo Pang, Zuxin Liu, Jielin Qiu, Jianguo Zhang, Yingbo Zhou, Zeyuan Chen, Ran Xu, Shelby Heinecke, Silvio Savarese, Caiming Xiong, Huan Wang, Weiran Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03270)  

**Abstract**: Diffusion language models promise bidirectional context and infilling capabilities that autoregressive coders lack, yet practical systems remain heavyweight. We introduce CoDA, a 1.7B-parameter diffusion coder trained on TPU with a fully open-source training pipeline. CoDA pairs large-scale diffusion pre-training with code-centric mid-training and instruction tuning, enabling confidence-guided sampling that keeps inference latency competitive. On Humaneval, MBPP, and EvalPlus, CoDA-1.7B-Instruct matches or surpasses diffusion models up to 7B parameters. Our release includes model checkpoints, evaluation harnesses, and TPU training pipelines to accelerate research on lightweight diffusion-based coding assistants. 

**Abstract (ZH)**: 扩散语言模型承诺提供自回归编码器所缺乏的双向上下文和填充能力，但实用系统仍较为笨重。我们介绍了CoDA，一个在TPU上训练的17亿参数扩散编码器，具有完全开源的训练流水线。CoDA结合了大规模扩散预训练、代码为中心的中期训练和指令调优，使得引导采样能够保持推理延迟的竞争性。在Humaneval、MBPP和EvalPlus上，CoDA-1.7B-Instruct在性能上与多达70亿参数的扩散模型相当或超越。我们发布的资料包括模型检查点、评估框架和TPU训练流水线，以加速基于轻量级扩散编码器的编程助手的研究。 

---
# General Exploratory Bonus for Optimistic Exploration in RLHF 

**Title (ZH)**: 乐观探索在RLHF中的通用探索bonus 

**Authors**: Wendi Li, Changdae Oh, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03269)  

**Abstract**: Optimistic exploration is central to improving sample efficiency in reinforcement learning with human feedback, yet existing exploratory bonus methods to incentivize exploration often fail to realize optimism. We provide a theoretical analysis showing that current formulations, under KL or $\alpha$-divergence regularization, unintentionally bias exploration toward high-probability regions of the reference model, thereby reinforcing conservative behavior instead of promoting discovery of uncertain regions. To address this pitfall, we introduce the General Exploratory Bonus (GEB), a novel theoretical framework that provably satisfies the optimism principle. GEB counteracts divergence-induced bias via reference-dependent reward regulation and unifies prior heuristic bonuses as special cases, while extending naturally across the full $\alpha$-divergence family. Empirically, GEB consistently outperforms baselines on alignment tasks across multiple divergence settings and large language model backbones. These results demonstrate that GEB offers both a principled and practical solution for optimistic exploration in RLHF. 

**Abstract (ZH)**: 乐观探索是提高强化学习中人类反馈样本效率的核心，但现有的探索奖励方法往往未能体现乐观性。我们提供了理论分析，表明当前基于KL或$\alpha$-散度正则化的形式无意中将探索偏向参考模型的高概率区域，从而 reinforcement 学习中强化保守行为而非促进对不确定区域的探索。为解决这一问题，我们提出了广义探索奖励（GEB），这是一个能严格满足乐观原则的新型理论框架。GEB 通过参考依赖的奖励调节抵消散度引入的偏差，并将先前的启发式奖励作为特殊情况统一其中，同时自然地扩展到整个$\alpha$-散度家族。实验结果显示，GEB 在多种散度设置和大型语言模型基础下的一致性表现优于基线方法，这些结果表明，GEB 为强化学习中的人类反馈提供了一个既具原理性又实用的乐观探索解决方案。 

---
# Decrypt Modality Gap in Multimodal Contrastive Learning: From Convergent Representation to Pair Alignment 

**Title (ZH)**: 解构多模态对比学习中的模态差距：从收敛表示到配对对齐 

**Authors**: Lingjie Yi, Raphael Douady, Chao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03268)  

**Abstract**: Multimodal contrastive learning (MCL) aims to embed data from different modalities in a shared embedding space. However, empirical evidence shows that representations from different modalities occupy completely separate regions of embedding space, a phenomenon referred to as the modality gap. Moreover, experimental findings on how the size of the modality gap influences downstream performance are inconsistent. These observations raise two key questions: (1) What causes the modality gap? (2) How does it affect downstream tasks? To address these questions, this paper introduces the first theoretical framework for analyzing the convergent optimal representations of MCL and the modality alignment when training is optimized. Specifically, we prove that without any constraint or under the cone constraint, the modality gap converges to zero. Under the subspace constraint (i.e., representations of two modalities fall into two distinct hyperplanes due to dimension collapse), the modality gap converges to the smallest angle between the two hyperplanes. This result identifies \emph{dimension collapse} as the fundamental origin of the modality gap. Furthermore, our theorems demonstrate that paired samples cannot be perfectly aligned under the subspace constraint. The modality gap influences downstream performance by affecting the alignment between sample pairs. We prove that, in this case, perfect alignment between two modalities can still be achieved via two ways: hyperplane rotation and shared space projection. 

**Abstract (ZH)**: 多模态对比学习中模态差距的理论分析及其对下游任务的影响 

---
# PT$^2$-LLM: Post-Training Ternarization for Large Language Models 

**Title (ZH)**: PT$^2$-LLM：大型语言模型的后训练三值化 

**Authors**: Xianglong Yan, Chengzhu Bao, Zhiteng Li, Tianao Zhang, Kaicheng Yang, Haotong Qin, Ruobing Xie, Xingwu Sun, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03267)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities across diverse tasks, but their large memory and compute demands hinder deployment. Ternarization has gained attention as a promising compression technique, delivering substantial size reduction and high computational efficiency. However, its potential in the post-training quantization (PTQ) setting remains underexplored, due to the challenge of training-free parameter optimization and the quantization difficulty posed by outliers and dispersed weights. To address these issues, we propose PT$^2$-LLM, a post-training ternarization framework tailored for LLMs. At its core is an Asymmetric Ternary Quantizer equipped with a two-stage refinement pipeline: (1) Iterative Ternary Fitting (ITF), which alternates between optimal ternary grid construction and flexible rounding to minimize quantization error, and (2) Activation-aware Grid Alignment (AGA), which further refines the ternary grid to better match full-precision outputs. In addition, we propose a plug-and-play Structural Similarity-based Reordering (SSR) strategy that leverages inter-column structural similarity to ease quantization and mitigate outlier effects, further enhancing overall performance. Extensive experiments demonstrate that PT$^2$-LLM delivers competitive performance against state-of-the-art (SOTA) 2-bit PTQ methods with lower memory cost, while also accelerating both prefill and decoding to achieve end-to-end speedup. The code and models will be available at this https URL. 

**Abstract (ZH)**: PT$^2$-LLM：面向大语言模型的后训练三值量化框架 

---
# MindCraft: How Concept Trees Take Shape In Deep Models 

**Title (ZH)**: MindCraft：概念树在深度模型中如何形成 

**Authors**: Bowei Tian, Yexiao He, Wanghao Ye, Ziyao Wang, Meng Liu, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03265)  

**Abstract**: Large-scale foundation models demonstrate strong performance across language, vision, and reasoning tasks. However, how they internally structure and stabilize concepts remains elusive. Inspired by causal inference, we introduce the MindCraft framework built upon Concept Trees. By applying spectral decomposition at each layer and linking principal directions into branching Concept Paths, Concept Trees reconstruct the hierarchical emergence of concepts, revealing exactly when they diverge from shared representations into linearly separable subspaces. Empirical evaluations across diverse scenarios across disciplines, including medical diagnosis, physics reasoning, and political decision-making, show that Concept Trees recover semantic hierarchies, disentangle latent concepts, and can be widely applied across multiple domains. The Concept Tree establishes a widely applicable and powerful framework that enables in-depth analysis of conceptual representations in deep models, marking a significant step forward in the foundation of interpretable AI. 

**Abstract (ZH)**: 大规模基础模型在语言、视觉和推理任务上表现出强大性能。然而，它们内部如何构建和稳定概念仍然不清楚。受因果推理启发，我们提出了基于概念树的MindCraft框架。通过在每一层应用谱分解并将主方向连接成分支概念路径，概念树重建了概念的层次涌现过程，揭示了它们从共享表示到线性可分子空间的具体分歧时刻。跨学科的实证评估，包括医学诊断、物理推理和政治决策等领域，表明概念树可以恢复语义层次结构、分离潜在概念，并且可以在多个领域广泛应用。概念树提供了一个广泛适用且强大的框架，使我们能够深入分析深度模型中的概念表示，标志着可解释AI基础的一大进步。 

---
# Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data 

**Title (ZH)**: 前置推理：预训练与后训练数据的协同作用 

**Authors**: Syeda Nahida Akter, Shrimai Prabhumoye, Eric Nyberg, Mostofa Patwary, Mohammad Shoeybi, Yejin Choi, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2510.03264)  

**Abstract**: The prevailing paradigm for enhancing the reasoning abilities of LLMs revolves around post-training on high-quality, reasoning-intensive data. While emerging literature suggests that reasoning data is increasingly incorporated also during the mid-training stage-a practice that is relatively more proprietary and less openly characterized-the role of such data in pretraining remains unclear. In particular, due to the opaqueness of pretraining corpora in most frontier models, the effect of reasoning data introduced at different phases of pre- and/or post-training is relatively less reported in the scientific literature. This raises several important questions: Is adding reasoning data earlier during pretraining any better than introducing it during post-training? Could earlier inclusion risk overfitting and harm generalization, or instead establish durable foundations that later fine-tuning cannot recover? We conduct the first systematic study of how reasoning data-varying in scale, diversity, and quality-affects LLM performance when introduced at different stages of training. We find that front-loading reasoning data into pretraining is critical (19% avg gain), establishing foundational capabilities that cannot be fully replicated by later-stage SFT, even with more data. We uncover an asymmetric principle for optimal data allocation: pretraining benefits most from broad diversity in reasoning patterns (11% avg gain), while SFT is more sensitive to data quality (15% avg gain). We show that high-quality pretraining data has latent effects, activated only after SFT, and that naively scaling SFT data can be detrimental, washing away the benefits of early reasoning injection. Our results challenge the conventional separation of language modeling and reasoning, providing a principled guide for strategically allocating data across the entire training pipeline to build more capable models. 

**Abstract (ZH)**: 增强LLMs推理能力的 prevailing paradigm 主要集中在后训练阶段对高质量、重推理数据的训练。虽然新兴文献表明推理数据也开始在中训练阶段被越来越多地纳入，这一做法相对更为专有且公开描述较少，但这种数据在预训练中的作用仍不清楚。特别是由于大多数前沿模型的预训练语料库不透明，关于不同预训练和/或后训练阶段引入推理数据的效果在科学文献中相对较少报道。这引发了一系列重要问题：早期预训练时引入推理数据是否比后训练时引入效果更好？提前引入数据是否会增加过拟合和削弱泛化能力，或者反而建立后期微调无法恢复的坚实基础？我们首次系统研究了不同训练阶段引入规模、多样性和质量各异的推理数据对LLM性能的影响。研究发现，将推理数据提前加载到预训练中至关重要（平均提升19%），建立了即使后续阶段细调更多数据也无法完全复制的基本能力。我们发现了一种不对称的数据分配原则：预训练最受益于广泛的推理模式多样性（平均提升11%），而后续阶段 fine-tuning 对数据质量更为敏感（平均提升15%）。我们证明了高质量的预训练数据具有潜在效果，仅在后续阶段 fine-tuning 时才会显现，并且简单地扩大后续阶段 fine-tuning 的数据量可能会适得其反，消除了早期推理注入的益处。我们的研究挑战了语言建模和推理的传统分离，提供了在整条训练流水线中战略性分配数据以构建更强大模型的原理性指南。 

---
# Memory Self-Regeneration: Uncovering Hidden Knowledge in Unlearned Models 

**Title (ZH)**: 记忆自我再生：揭示未学习模型中的隐藏知识 

**Authors**: Agnieszka Polowczyk, Alicja Polowczyk, Joanna Waczyńska, Piotr Borycki, Przemysław Spurek  

**Link**: [PDF](https://arxiv.org/pdf/2510.03263)  

**Abstract**: The impressive capability of modern text-to-image models to generate realistic visuals has come with a serious drawback: they can be misused to create harmful, deceptive or unlawful content. This has accelerated the push for machine unlearning. This new field seeks to selectively remove specific knowledge from a model's training data without causing a drop in its overall performance. However, it turns out that actually forgetting a given concept is an extremely difficult task. Models exposed to attacks using adversarial prompts show the ability to generate so-called unlearned concepts, which can be not only harmful but also illegal. In this paper, we present considerations regarding the ability of models to forget and recall knowledge, introducing the Memory Self-Regeneration task. Furthermore, we present MemoRa strategy, which we consider to be a regenerative approach supporting the effective recovery of previously lost knowledge. Moreover, we propose that robustness in knowledge retrieval is a crucial yet underexplored evaluation measure for developing more robust and effective unlearning techniques. Finally, we demonstrate that forgetting occurs in two distinct ways: short-term, where concepts can be quickly recalled, and long-term, where recovery is more challenging. 

**Abstract (ZH)**: 现代文本到图像模型生成逼真视觉的 impressive 能力带来了一个严重的问题：它们可能被滥用以生成有害、欺骗性或非法内容。这加速了对机器遗忘的推动力。这一新领域旨在从模型的训练数据中选择性地移除特定知识而不影响其整体性能。然而，实际上忘记某一概念 proving 是一个极其困难的任务。暴露于对抗提示攻击下的模型展示了生成所谓未学习概念的能力，这些概念不仅是有害的，而且可能也是非法的。在本文中，我们探讨了模型忘记和回忆知识的能力，提出了记忆自我再生任务。此外，我们介绍了 MemoRa 策略，我们认为这是一种支持有效恢复之前丢失知识的再生方法。同时，我们认为在知识检索中的鲁棒性是开发更 robust 和有效的遗忘技术的一个关键但尚未充分探索的评估指标。最后，我们展示忘记以两种不同的方式发生：短期，其中概念可以迅速回忆；长期，其中恢复更具挑战性。 

---
# Rethinking Inter-LoRA Orthogonality in Adapter Merging: Insights from Orthogonal Monte Carlo Dropout 

**Title (ZH)**: 重思适配器合并中的LoRA正交性：正交蒙特卡洛dropout的见解 

**Authors**: Andi Zhang, Xuan Ding, Haofan Wang, Steven McDonagh, Samuel Kaski  

**Link**: [PDF](https://arxiv.org/pdf/2510.03262)  

**Abstract**: We propose Orthogonal Monte Carlo Dropout, a mechanism that enforces strict orthogonality when combining sparse semantic vectors without extra time complexity. LoRA, a popular fine-tuning method for large models, typically trains a module to represent a specific concept such as an object or a style. When multiple LoRAs are merged, for example to generate an object in a particular style, their semantic vectors may interfere with each other. Our method guarantees, at the theoretical and runtime levels, that merged LoRAs remain orthogonal and thus free from direct interference. However, empirical analysis reveals that such orthogonality does not lead to the semantic disentanglement or compositionality highlighted in prior work on compositional adaptation. This finding suggests that inter-LoRA orthogonality alone may be insufficient for achieving true semantic compositionality, prompting a re-examination of its role in adapter merging. 

**Abstract (ZH)**: 我们提出正交蒙特卡洛丢弃机制，在不增加额外时间复杂度的情况下，确保稀疏语义向量的严格正交性。LoRA 是一种流行的大模型精细调整方法，通常训练一个模块来表示特定概念，如对象或风格。当多个 LoRAs 被合并时，例如为了生成特定风格的对象，它们的语义向量可能会相互干扰。我们的方法在理论上和运行时保证合并后的 LoRAs 保持正交，从而避免直接干扰。然而，实证分析表明，这种正交性并不能带来先前关于组合适应性工作所强调的语义去纠缠性和组合性。这一发现表明，仅靠 LoRA 间的正交性可能不足以实现真正的语义组合性，这促使我们重新审视其在适配器合并中的作用。 

---
# Semantic-Inductive Attribute Selection for Zero-Shot Learning 

**Title (ZH)**: 零样本学习中的语义归纳属性选择 

**Authors**: Juan Jose Herrera-Aranda, Guillermo Gomez-Trenado, Francisco Herrera, Isaac Triguero  

**Link**: [PDF](https://arxiv.org/pdf/2510.03260)  

**Abstract**: Zero-Shot Learning is an important paradigm within General-Purpose Artificial Intelligence Systems, particularly in those that operate in open-world scenarios where systems must adapt to new tasks dynamically. Semantic spaces play a pivotal role as they bridge seen and unseen classes, but whether human-annotated or generated by a machine learning model, they often contain noisy, redundant, or irrelevant attributes that hinder performance. To address this, we introduce a partitioning scheme that simulates unseen conditions in an inductive setting (which is the most challenging), allowing attribute relevance to be assessed without access to semantic information from unseen classes. Within this framework, we study two complementary feature-selection strategies and assess their generalisation. The first adapts embedded feature selection to the particular demands of ZSL, turning model-driven rankings into meaningful semantic pruning; the second leverages evolutionary computation to directly explore the space of attribute subsets more broadly. Experiments on five benchmark datasets (AWA2, CUB, SUN, aPY, FLO) show that both methods consistently improve accuracy on unseen classes by reducing redundancy, but in complementary ways: RFS is efficient and competitive though dependent on critical hyperparameters, whereas GA is more costly yet explores the search space more broadly and avoids such dependence. These results confirm that semantic spaces are inherently redundant and highlight the proposed partitioning scheme as an effective tool to refine them under inductive conditions. 

**Abstract (ZH)**: 零样本学习是通用人工智能系统中一个重要的范式，特别是在开放世界场景中，系统需要动态适应新任务。语义空间在这种情境下发挥关键作用，它们连接了已知和未知类别，但无论是由人类标注还是通过机器学习模型生成，语义空间往往包含噪声、冗余或无关的属性，从而妨碍性能。为了解决这一问题，我们提出了一种分区方案，在归纳设置中模拟未知条件（最具挑战性的情况），允许在不访问未知类别语义信息的情况下评估属性的相关性。在此框架内，我们研究了两种互补的特征选择策略，并评估了它们的一般性。第一种方法针对零样本学习的具体需求，将模型驱动的排名转化为有意义的语义修剪；第二种方法利用进化计算更广泛地探索属性子集的空间。实验结果显示，在五个基准数据集（AWA2、CUB、SUN、aPY、FLO）上，两种方法都能通过减少冗余性在未知类别上提高准确性，但方式互补：特征选择（RFS）高效且具有竞争力，但依赖于关键超参数；而遗传算法（GA）虽然成本更高，但能更广泛地探索搜索空间并避免这种依赖性。这些结果证实了语义空间本身存在冗余性，并强调了所提出的分区方案作为在归纳条件下完善语义空间的有效工具。 

---
# Meta-Awareness Enhances Reasoning Models: Self-Alignment Reinforcement Learning 

**Title (ZH)**: 元意识提升推理模型：自我对齐强化学习 

**Authors**: Yoonjeon Kim, Doohyuk Jang, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03259)  

**Abstract**: Recent studies on reasoning models explore the meta-awareness of language models, the ability to know how to think by itself. We argue that large reasoning models lack this meta-awareness property by proving severe misalignment between true rollouts and predicted meta information. We posit that aligning meta-prediction with true rollouts will lead to significant performance gains. To verify this hypothesis, we design a training pipeline that boosts Meta-Awareness via Self-Alignment (MASA), and prove that enhanced meta-awareness directly translates to improved accuracy. Unlike existing meta-cognitive reasoning models, our method does not require external training sources but leverages self-generated signals to train meta-awareness. Moreover, our method enables efficient training by i) filtering out zero-variance prompts that are either trivial or unsolvable and ii) cutting off lengthy rollouts when they are unlikely to lead to correct answers. The results are inspiring: our strategy yields significant improvements in both accuracy and training efficiency on in-domain tasks and shows strong generalization to out-of-domain benchmarks. More specifically, our method can speed up GRPO training by over 1.28x to reach the same performance, and achieve a 19.3% gain in accuracy on AIME25, and a 6.2 % average gain over six mathematics benchmarks. Training with meta-cognitive guidance enhances out-of-domain generalization, giving a 3.87 % boost on GPQA-Diamond and a 2.08 % overall accuracy gain across 13 benchmarks spanning logical, scientific, and coding domains. 

**Abstract (ZH)**: 最近关于推理模型的研究探索了语言模型的元意识能力，即自我了解如何思考的能力。我们通过证明真实的展开与预测的元信息之间存在严重的错位，认为大型推理模型缺乏这种元意识属性。我们提出，使元预测与真实展开一致将带来显著的性能提升。为了验证这一假设，我们设计了一种通过自我对齐提升元意识（MASA）的训练管道，并证明增强的元意识可以直接转化为更高的准确性。与现有元认知推理模型不同，我们的方法不需要外部训练来源，而是利用自动生成的信号来训练元意识。此外，我们的方法通过以下方式实现高效的训练：i) 过滤掉那些要么无聊要么无法解决的零方差提示，ii) 当展开不太可能得出正确答案时，中断过长的展开。结果令人鼓舞：我们的策略在领域内任务中显著提高了准确性和训练效率，并在领域外基准测试中展现出强大的泛化能力。具体来说，我们的方法可以将GRPO的训练速度提高1.28倍以上，达到相同表现，同时在AIME25上获得19.3%的准确性提升，并在六个数学基准上平均获得6.2%的提升。使用元认知指导进行训练增强了领域外泛化能力，在GPQA-Diamond上获得3.87%的提升，并在覆盖逻辑、科学和编程领域的13个基准测试中平均提高2.08%的总体准确率。 

---
# POEM: Explore Unexplored Reliable Samples to Enhance Test-Time Adaptation 

**Title (ZH)**: POEM: 探索未探索的可靠样本以增强测试时适应性 

**Authors**: Chang'an Yi, Xiaohui Deng, Shuaicheng Niu, Yan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03258)  

**Abstract**: Test-time adaptation (TTA) aims to transfer knowledge from a source model to unknown test data with potential distribution shifts in an online manner. Many existing TTA methods rely on entropy as a confidence metric to optimize the model. However, these approaches are sensitive to the predefined entropy threshold, influencing which samples are chosen for model adaptation. Consequently, potentially reliable target samples are often overlooked and underutilized. For instance, a sample's entropy might slightly exceed the threshold initially, but fall below it after the model is updated. Such samples can provide stable supervised information and offer a normal range of gradients to guide model adaptation. In this paper, we propose a general approach, \underline{POEM}, to promote TTA via ex\underline{\textbf{p}}loring the previously unexpl\underline{\textbf{o}}red reliabl\underline{\textbf{e}} sa\underline{\textbf{m}}ples. Additionally, we introduce an extra Adapt Branch network to strike a balance between extracting domain-agnostic representations and achieving high performance on target data. Comprehensive experiments across multiple architectures demonstrate that POEM consistently outperforms existing TTA methods in both challenging scenarios and real-world domain shifts, while remaining computationally efficient. The effectiveness of POEM is evaluated through extensive analyses and thorough ablation studies. Moreover, the core idea behind POEM can be employed as an augmentation strategy to boost the performance of existing TTA approaches. The source code is publicly available at \emph{this https URL} 

**Abstract (ZH)**: Test-time adaptation (TTA)旨在在线方式将知识从源模型转移到具有潜在分布偏移的未知测试数据中。许多现有的TTA方法依赖于熵作为置信度度量来优化模型。然而，这些方法对预定义的熵阈值敏感，影响了哪些样本被选中进行模型适应。因此，许多潜在可靠的目标样本往往被忽视和未充分利用。例如，一个样本的熵最初可能略微超过阈值，但在模型更新后又低于阈值。这种样本可以提供稳定的监督信息，并提供指导模型适应的正常范围梯度。在本文中，我们提出了一种通用方法POEM，通过探索之前未探索的可靠样本来促进TTA。此外，我们引入了一个额外的Adapt Branch网络，以平衡提取领域无关表示和在目标数据上实现高性能之间的关系。在多种架构上的综合实验表明，在具有挑战性的场景和实际领域偏移中，POEM始终优于现有的TTA方法，同时保持计算效率。POEM的有效性通过广泛分析和彻底的消融研究进行了评估。此外，POEM的核心思想可以作为一种增强策略，以提高现有TTA方法的性能。源代码可在\emph{this https URL}公开获取。 

---
# Triple-BERT: Do We Really Need MARL for Order Dispatch on Ride-Sharing Platforms? 

**Title (ZH)**: Triple-BERT: 我们真的需要多智能体强化学习来解决拼车平台的订单分配问题吗？ 

**Authors**: Zijian Zhao, Sen Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03257)  

**Abstract**: On-demand ride-sharing platforms, such as Uber and Lyft, face the intricate real-time challenge of bundling and matching passengers-each with distinct origins and destinations-to available vehicles, all while navigating significant system uncertainties. Due to the extensive observation space arising from the large number of drivers and orders, order dispatching, though fundamentally a centralized task, is often addressed using Multi-Agent Reinforcement Learning (MARL). However, independent MARL methods fail to capture global information and exhibit poor cooperation among workers, while Centralized Training Decentralized Execution (CTDE) MARL methods suffer from the curse of dimensionality. To overcome these challenges, we propose Triple-BERT, a centralized Single Agent Reinforcement Learning (MARL) method designed specifically for large-scale order dispatching on ride-sharing platforms. Built on a variant TD3, our approach addresses the vast action space through an action decomposition strategy that breaks down the joint action probability into individual driver action probabilities. To handle the extensive observation space, we introduce a novel BERT-based network, where parameter reuse mitigates parameter growth as the number of drivers and orders increases, and the attention mechanism effectively captures the complex relationships among the large pool of driver and orders. We validate our method using a real-world ride-hailing dataset from Manhattan. Triple-BERT achieves approximately an 11.95% improvement over current state-of-the-art methods, with a 4.26% increase in served orders and a 22.25% reduction in pickup times. Our code, trained model parameters, and processed data are publicly available at the repository this https URL . 

**Abstract (ZH)**: 基于需求的共享出行平台（如Uber和Lyft）面临实时挑战，即如何将具有不同出发地和目的地的乘客与可用车辆进行有效的组合和匹配，同时还要应对系统中的大量不确定性。由于驾驶员和订单数量庞大导致观测空间广泛，尽管本质上是集中式任务，但订单调度通常采用多智能体强化学习（MARL）方法。然而，独立的MARL方法无法捕捉全局信息，智能体间合作效果差，而集中式训练分布式执行（CTDE）的MARL方法则遭受维度灾难。为克服这些挑战，我们提出了Triple-BERT方法，这是一种专为共享出行平台大规模订单调度设计的集中式单智能体强化学习方法。基于TD3的变体，该方法通过动作分解策略解决庞大的动作空间问题，将联合动作概率分解为单个驾驶员的动作概率。为应对广泛的观测空间，我们引入了一种新型的基于BERT的网络，其中参数重用在驾驶员和订单数量增加时减少了参数的增长，并且注意力机制有效地捕捉了大量驾驶员和订单间的复杂关系。我们在曼哈顿的真实打车数据集上验证了该方法。Triple-BERT在现有最先进的方法上实现了约11.95%的性能提升，服务订单数增加了4.26%，接客时间减少了22.25%。我们的代码、训练模型参数和处理后的数据可在以下仓库公开访问：this https URL。 

---
# SciTS: Scientific Time Series Understanding and Generation with LLMs 

**Title (ZH)**: SciTS：借助大规模语言模型的科学时间序列理解和生成 

**Authors**: Wen Wu, Ziyang Zhang, Liwei Liu, Xuenan Xu, Junlin Liu, Ke Fan, Qitan Lv, Jimin Zhuang, Chen Zhang, Zheqi Yuan, Siyuan Hou, Tianyi Lin, Kai Chen, Bowen Zhou, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03255)  

**Abstract**: The scientific reasoning ability of large language models (LLMs) has recently attracted significant attention. Time series, as a fundamental modality in scientific data, presents unique challenges that are often overlooked in current multimodal LLMs, which either encode numerical sequences as text or convert them into images. Such approaches may be insufficient for comprehensive scientific time series understanding and generation. Existing unified time series models typically specialise in either forecasting or analysis, and their effectiveness on non-periodic, heterogeneous scientific signals remains unclear. To address these gaps, we introduce SciTS, a benchmark spanning 12 scientific domains and 43 tasks, with over 50k+ instances, both univariate and multivariate signals ranging from $10^0$ to $10^7$ in length and up to 10~MHz in frequency. We benchmark 17 models, including text-only LLMs, multimodal LLMs, and unified time series models, and find that general-purpose LLMs exhibit stronger generalisability than specialised time series models, while representing time series as text or images limits their performance due to excessively long sequences and loss of numerical precision, respectively. We then introduce TimeOmni, a framework that equips LLMs with the ability to understand and generate time series while remaining compatible with general-purpose LLM training. This work fills a gap in both dedicated benchmarks and modelling frameworks for scientific time series, paving the way for LLMs to understand and generate complex temporal scientific data. 

**Abstract (ZH)**: 大型语言模型的科学推理能力最近引起了广泛关注。作为科学数据的基本模态，时间序列面临着当前多模态大型语言模型中常被忽视的独特挑战，这些模型要么将数值序列编码为文本，要么将它们转换为图像。这些方法可能不足以实现全面的时间序列科学理解和生成。现有的统一时间序列模型通常专注于预测或分析，它们在非周期性、异质科学信号上的有效性仍不清楚。为了解决这些差距，我们引入了 SciTS，这是一个覆盖 12 个科学领域和 43 项任务的基准，包括超过 50000 个实例，单变量和多变量信号长度从 \(10^0\) 到 \(10^7\)，频率高达 10 MHz。我们对 17 个模型进行了基准测试，包括仅文本的大规模语言模型、多模态大规模语言模型和统一时间序列模型，并发现通用目的的大规模语言模型在泛化能力上优于专门的时间序列模型，而将时间序列表示为文本或图像则因序列过长和数值精度的损失限制了其性能。我们随后引入了 TimeOmni，这是一种框架，使大规模语言模型能够理解和生成时间序列数据，同时保持与通用目的大规模语言模型训练的兼容性。这项工作填补了专门的时间序列基准和建模框架的空白，为大规模语言模型理解和生成复杂的时序科学数据铺平了道路。 

---
# Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents 

**Title (ZH)**: 解决粒度不匹配：面向长_horizon LLM代理的层次化偏好学习 

**Authors**: Heyang Gao, Zexu Sun, Erxue Min, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03253)  

**Abstract**: Large Language Models (LLMs) as autonomous agents are increasingly tasked with solving complex, long-horizon problems. Aligning these agents via preference-based offline methods like Direct Preference Optimization (DPO) is a promising direction, yet it faces a critical granularity mismatch. Trajectory-level DPO provides a signal that is too coarse for precise credit assignment, while step-level DPO is often too myopic to capture the value of multi-step behaviors. To resolve this challenge, we introduce Hierarchical Preference Learning (HPL), a hierarchical framework that optimizes LLM agents by leveraging preference signals at multiple, synergistic granularities. While HPL incorporates trajectory- and step-level DPO for global and local policy stability, its core innovation lies in group-level preference optimization guided by a dual-layer curriculum. Our approach first decomposes expert trajectories into semantically coherent action groups and then generates contrasting suboptimal groups to enable preference learning at a fine-grained, sub-task level. Then, instead of treating all preference pairs equally, HPL introduces a curriculum scheduler that organizes the learning process from simple to complex. This curriculum is structured along two axes: the group length, representing sub-task complexity, and the sample difficulty, defined by the reward gap between preferred and dispreferred action groups. Experiments on three challenging agent benchmarks show that HPL outperforms existing state-of-the-art methods. Our analyses demonstrate that the hierarchical DPO loss effectively integrates preference signals across multiple granularities, while the dual-layer curriculum is crucial for enabling the agent to solve a wide range of tasks, from simple behaviors to complex multi-step sequences. 

**Abstract (ZH)**: 大型语言模型（LLMs）作为自主代理，正逐渐被赋予解决复杂、长期问题的任务。通过基于偏好的离线方法如直接偏好优化（DPO）进行对齐是一种有前途的方向，但会遇到关键的粒度不匹配问题。轨迹级DPO提供的信号过粗，不适合精确的信用分配，而步骤级DPO往往过于短视，无法捕捉多步行为的价值。为了解决这一挑战，我们引入了层次偏好学习（HPL），这是一种利用多级协同粒度的偏好信号来优化LLM代理的层次框架。虽然HPL结合了轨迹级和步骤级DPO以实现全局和局部策略的稳定性，其核心创新在于由双层课程引导的群体级偏好优化。我们的方法首先将专家轨迹分解为语义上连贯的动作组，然后生成对比的次优组以在细粒度的任务子级上实现偏好学习。然后，HPL并未平等对待所有的偏好对，而是引入了一种课程调度器从简单到复杂组织学习过程。该课程沿两个轴结构化：组长度，代表子任务复杂性，以及样本难度，由偏好和不偏好动作组的奖励差距定义。在三个具有挑战性的代理基准测试中，HPL优于现有最先进的方法。我们的分析表明，层次化的DPO损失有效整合了多粒度的偏好信号，而双层课程对于使代理能够解决从简单行为到复杂多步序列的各种任务至关重要。 

---
# Universal Multi-Domain Translation via Diffusion Routers 

**Title (ZH)**: 通过扩散路由器实现的通用多域翻译 

**Authors**: Duc Kieu, Kien Do, Tuan Hoang, Thao Minh Le, Tung Kieu, Dang Nguyen, Thin Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03252)  

**Abstract**: Multi-domain translation (MDT) aims to learn translations between multiple domains, yet existing approaches either require fully aligned tuples or can only handle domain pairs seen in training, limiting their practicality and excluding many cross-domain mappings. We introduce universal MDT (UMDT), a generalization of MDT that seeks to translate between any pair of $K$ domains using only $K-1$ paired datasets with a central domain. To tackle this problem, we propose Diffusion Router (DR), a unified diffusion-based framework that models all central$\leftrightarrow$non-central translations with a single noise predictor conditioned on the source and target domain labels. DR enables indirect non-central translations by routing through the central domain. We further introduce a novel scalable learning strategy with a variational-bound objective and an efficient Tweedie refinement procedure to support direct non-central mappings. Through evaluation on three large-scale UMDT benchmarks, DR achieves state-of-the-art results for both indirect and direct translations, while lowering sampling cost and unlocking novel tasks such as sketch$\leftrightarrow$segmentation. These results establish DR as a scalable and versatile framework for universal translation across multiple domains. 

**Abstract (ZH)**: 多域翻译（MDT）旨在学习多个域之间的翻译，但现有方法要么需要完全对齐的元组，要么只能处理训练中出现的域对，这限制了它们的实际应用并排除了许多跨域映射。我们引入了一种多域翻译的通用化方法（UMDT），该方法仅使用一个中心域和$K-1$个中心域与非中心域的配对数据集，即可在任意一对$K$域之间进行翻译。为了解决这一问题，我们提出了统一的扩散路由器（DR），这是一种基于扩散的统一框架，它通过在源域和目标域标签的条件下条件化噪声预测器来建模所有中心$\leftrightarrow$非中心的翻译。DR通过路由到中心域来实现间接的非中心翻译。我们还引入了一种新颖的可扩展学习策略，该策略具有变分界线目标和高效的Tweedie精炼过程，以支持直接的非中心映射。通过在三个大规模UMDT基准上的评估，DR在间接和直接翻译中均实现了最优结果，同时降低了采样成本并解锁了如草图$\leftrightarrow$分割等新任务。这些结果确立了DR作为一种适用于多域通用翻译的可扩展和多功能框架的地位。 

---
# Numerion: A Multi-Hypercomplex Model for Time Series Forecasting 

**Title (ZH)**: Numerion：一种用于时间序列预测的多超复数模型 

**Authors**: Hanzhong Cao, Wenbo Yan, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03251)  

**Abstract**: Many methods aim to enhance time series forecasting by decomposing the series through intricate model structures and prior knowledge, yet they are inevitably limited by computational complexity and the robustness of the assumptions. Our research uncovers that in the complex domain and higher-order hypercomplex spaces, the characteristic frequencies of time series naturally decrease. Leveraging this insight, we propose Numerion, a time series forecasting model based on multiple hypercomplex spaces. Specifically, grounded in theoretical support, we generalize linear layers and activation functions to hypercomplex spaces of arbitrary power-of-two dimensions and introduce a novel Real-Hypercomplex-Real Domain Multi-Layer Perceptron (RHR-MLP) architecture. Numerion utilizes multiple RHR-MLPs to map time series into hypercomplex spaces of varying dimensions, naturally decomposing and independently modeling the series, and adaptively fuses the latent patterns exhibited in different spaces through a dynamic fusion mechanism. Experiments validate the model`s performance, achieving state-of-the-art results on multiple public datasets. Visualizations and quantitative analyses comprehensively demonstrate the ability of multi-dimensional RHR-MLPs to naturally decompose time series and reveal the tendency of higher dimensional hypercomplex spaces to capture lower frequency features. 

**Abstract (ZH)**: 基于超复数空间的Numerion时间序列forecasting模型 

---
# Real-Time Brain Biomechanics Prediction with Neural Operators: Toward Clinically Deployable Traumatic Brain Injury Models 

**Title (ZH)**: 基于神经算子的实时脑 biomechanics 预测：迈向临床可用的创伤性脑损伤模型 

**Authors**: Anusha Agarwal, Dibakar Roy Sarkar, Somdatta Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2510.03248)  

**Abstract**: Traumatic brain injury (TBI) remains a major public health concern, with over 69 million cases annually worldwide. Finite element (FE) models offer high-fidelity predictions of brain deformation but are computationally expensive, requiring hours per simulation and limiting their clinical utility for rapid decision-making. This study benchmarks state-of-the-art neural operator (NO) architectures for rapid, patient-specific prediction of brain displacement fields, aiming to enable real-time TBI modeling in clinical and translational settings. We formulated TBI modeling as an operator learning problem, mapping subject-specific anatomical MRI, magnetic resonance elastography (MRE) stiffness maps, and demographic features to full-field 3D brain displacement predictions. Four architectures - Fourier Neural Operator (FNO), Factorized FNO (F-FNO), Multi-Grid FNO (MG-FNO), and Deep Operator Network (DeepONet) were trained and evaluated on 249 MRE datasets across physiologically relevant frequencies (20 - 90 Hz). MG-FNO achieved the highest accuracy (MSE = 0.0023, 94.3\% spatial fidelity) and preserved fine-scale features, while F-FNO converged 2$\times$ faster than standard FNO. DeepONet offered the fastest inference (14.5 iterations/s) with a 7$\times$ computational speed-up over MG-FNO, suggesting utility for embedded or edge computing applications. All NOs reduced computation time from hours to milliseconds without sacrificing anatomical realism. NOs provide an efficient, resolution-invariant approach for predicting brain deformation, opening the door to real-time, patient-specific TBI risk assessment, clinical triage support, and optimization of protective equipment. These results highlight the potential for NO-based digital twins of the human brain, enabling scalable, on-demand biomechanical modeling in both clinical and population health contexts. 

**Abstract (ZH)**: 创伤性脑损伤(TBI)仍然是一个主要的公共健康问题，每年全球病例超过6900万例。有限元(FE)模型能够高保真地预测脑部变形，但是计算成本高昂，每个模拟需要数小时的时间，限制了其在快速决策临床环境中的应用。本研究旨在通过先进的神经运算器( Neural Operator, NO)架构，实现快速的患者特定脑部位移场预测，以期在临床和转化研究中实现即时创伤性脑损伤(TBI)建模。我们将TBI建模定性为一个运算器学习问题，将个体化的解剖MRI、磁共振弹性图(MRE)刚度图以及人口统计特征映射到全领域的三维脑部位移预测。四种架构——傅里叶神经运算器(Fourier Neural Operator, FNO)、因子分解傅里叶神经运算器(Factorized FNO, F-FNO)、多重网格傅里叶神经运算器(Multi-Grid FNO, MG-FNO) 和深度运算器网络(Deep Operator Network, DeepONet)——在249个MRE数据集上进行了训练和评估，这些数据涵盖了生理相关频率（20-90 Hz）。多重网格傅里叶神经运算器(MG-FNO)达到了最高的准确性（均方误差MSE=0.0023，空间保真度94.3%）并保留了细尺度特征，同时因子分解傅里叶神经运算器(F-FNO)比标准傅里叶神经运算器(FNO)快两倍的收敛速度。深度运算器网络提供了最快推断速度（每秒14.5次迭代），相比多重网格傅里叶神经运算器有七倍的计算速度提升，表明其可能适用于嵌入式或边缘计算应用。所有神经运算器(NOs)将计算时间从数小时减少到毫秒级，同时保持了解剖学的真实感。神经运算器提供了一种高效、分辨率无关的方法来预测脑变形，为实现即时、患者特定的TBI风险评估、临床分诊支持和防护设备优化提供了可能。这些结果突显了基于神经运算器的类人脑数字孪生的潜力，能够在临床和人口健康领域实现扩展且按需的生物力学建模。 

---
# Towards Multimodal Active Learning: Efficient Learning with Limited Paired Data 

**Title (ZH)**: 面向多模态主动学习：在有限配对数据情况下高效学习 

**Authors**: Jiancheng Zhang, Yinglun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03247)  

**Abstract**: Active learning (AL) is a principled strategy to reduce annotation cost in data-hungry deep learning. However, existing AL algorithms focus almost exclusively on unimodal data, overlooking the substantial annotation burden in multimodal learning. We introduce the first framework for multimodal active learning with unaligned data, where the learner must actively acquire cross-modal alignments rather than labels on pre-aligned pairs. This setting captures the practical bottleneck in modern multimodal pipelines such as CLIP and SigLIP, where unimodal features are easy to obtain but high-quality alignment is costly. We develop a new algorithm that combines uncertainty and diversity principles in a modality-aware design, achieves linear-time acquisition, and applies seamlessly to both pool-based and streaming-based settings. Extensive experiments on benchmark datasets demonstrate that our approach consistently reduces multimodal annotation cost while preserving performance; for instance, on the ColorSwap dataset it cuts annotation requirements by up to $40\%$ without loss in accuracy. 

**Abstract (ZH)**: 多模态未对齐数据的主动学习框架 

---
# StructPrune: Structured Global Pruning asymptotics with $\mathcal{O}(\sqrt{N})$ GPU Memory 

**Title (ZH)**: StructPrune: 结构化全局剪枝的 $\mathcal{O}(\sqrt{N})$ GPU 内存复杂度 

**Authors**: Xinyuan Song, Guangji Bai, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03246)  

**Abstract**: Pruning is critical for scaling large language models (LLMs). Global pruning achieves strong performance but requires $\mathcal{O}(N)$ memory, which is infeasible for billion-parameter models. Local pruning reduces GPU memory usage to that of a single layer by pruning layers independently, but it neglects inter-layer dependencies and often leads to suboptimal performance in high-sparsity regimes. Unlike unstructured pruning, structured pruning produces regular sparsity patterns that align well with GPU kernels and library optimizations, making it more hardware-efficient. However, structured pruning typically relies on global pruning, since structured patterns are more prone to severe performance degradation under local optimization. To jointly achieve structured pruning and the memory efficiency of local pruning, we propose a divide-and-conquer strategy that decomposes the global pruning problem into coordinated subproblems across different modules, each of which fits within limited GPU memory. Building on this idea, we design \textbf{STRUPRUNE}, an ADMM-based framework that integrates structured sparsity into the pruning process, combining the memory efficiency of local pruning with the hardware compatibility of structured methods. We derive a closed-form analytical solution for structured pruning masks that provides an explicit rule for layer-wise sparsity allocation, and further develop an energy-based asymptotic framework yielding a softmax-form allocation scheme that simplifies optimization while adapting to heterogeneous layer importance. Experiments demonstrate that STRUPRUNE matches the perplexity of global structured pruning while reducing memory cost from $\mathcal{O}(N)$ to $\mathcal{O}(\sqrt{N})$, enabling practical deployment at the billion-parameter scale. 

**Abstract (ZH)**: 剪枝对于扩展大规模语言模型（LLMs）至关重要。全局剪枝性能强大，但需要 $\mathcal{O}(N)$ 内存，这对十亿参数模型而言是不可行的。局部剪枝通过独立剪枝层将GPU内存使用量降低到单层水平，但忽略了层间的依赖关系，在高稀疏性区间通常会导致性能不佳。与无结构剪枝不同，结构剪枝生成与GPU内核和库优化相兼容的规律稀疏模式，使其更具硬件效率。然而，结构剪枝通常依赖于全局剪枝，因为局部优化下的结构模式更容易导致严重的性能下降。为同时实现结构剪枝和局部剪枝的内存效率，我们提出了一种分而治之的策略，该策略将全局剪枝问题分解为跨不同模块协调的子问题，每个子问题都能在有限的GPU内存内运行。基于这一理念，我们设计了STRUPRUNE框架，这是一个基于ADMM的框架，将结构稀疏性整合到剪枝过程中，结合了局部剪枝的内存效率和结构方法的硬件兼容性。我们推导出结构剪枝掩码的闭式解析解，提供了一种逐层稀疏性分配的显式规则，并进一步开发了一种基于能量的渐近框架，提供了一种softmax形式的分配方案，简化了优化过程并适应不同层的重要性。实验表明，STRUPRUNE在减少内存成本从 $\mathcal{O}(N)$ 到 $\mathcal{O}(\sqrt{N})$ 的同时，匹配全局结构剪枝的困惑度，使其在十亿参数规模下具备实际部署能力。 

---
# Frequency-Aware Model Parameter Explorer: A new attribution method for improving explainability 

**Title (ZH)**: 频率感知模型参数探索者：一种提高可解释性的归因方法 

**Authors**: Ali Yavari, Alireza Mohamadi, Elham Beydaghi, Rainer A. Leitgeb  

**Link**: [PDF](https://arxiv.org/pdf/2510.03245)  

**Abstract**: Ensuring the reliability of deep neural networks (DNNs) in the presence of real world noise and intentional perturbations remains a significant challenge. To address this, attribution methods have been proposed, though their efficacy remains suboptimal and necessitates further refinement. In this paper, we propose a novel category of transferable adversarial attacks, called transferable frequency-aware attacks, enabling frequency-aware exploration via both high-and low-frequency components. Based on this type of attacks, we also propose a novel attribution method, named Frequency-Aware Model Parameter Explorer (FAMPE), which improves the explainability for DNNs. Relative to the current state-of-the-art method AttEXplore, our FAMPE attains an average gain of 13.02% in Insertion Score, thereby outperforming existing approaches. Through detailed ablation studies, we also investigate the role of both high- and low-frequency components in explainability. 

**Abstract (ZH)**: 确保深度神经网络在现实世界噪声和故意干扰下的可靠性依然是一项重大挑战。为此，已经提出了一些归因方法，但其效果仍然不尽如人意，需要进一步优化。本文提出了一种新型可移植频率感知攻击，称为传输频率感知攻击，能够通过高频和低频成分进行频率感知探索。基于此类攻击，我们还提出了一种新型归因方法——频率感知模型参数探索器（FAMPE），该方法提高了深度神经网络的可解释性。与当前最先进的方法AttEXplore相比，我们的FAMPE在插入分数上平均提高了13.02%，从而优于现有方法。通过详细的消融研究表明，高频和低频成分在可解释性中均发挥着重要作用。 

---
# VIFO: Visual Feature Empowered Multivariate Time Series Forecasting with Cross-Modal Fusion 

**Title (ZH)**: VIFO：视觉特征增强的跨模态融合多变量时间序列预测 

**Authors**: Yanlong Wang, Hang Yu, Jian Xu, Fei Ma, Hongkang Zhang, Tongtong Feng, Zijian Zhang, Shao-Lun Huang, Danny Dongning Sun, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03244)  

**Abstract**: Large time series foundation models often adopt channel-independent architectures to handle varying data dimensions, but this design ignores crucial cross-channel dependencies. Concurrently, existing multimodal approaches have not fully exploited the power of large vision models (LVMs) to interpret spatiotemporal data. Additionally, there remains significant unexplored potential in leveraging the advantages of information extraction from different modalities to enhance time series forecasting performance. To address these gaps, we propose the VIFO, a cross-modal forecasting model. VIFO uniquely renders multivariate time series into image, enabling pre-trained LVM to extract complex cross-channel patterns that are invisible to channel-independent models. These visual features are then aligned and fused with representations from the time series modality. By freezing the LVM and training only 7.45% of its parameters, VIFO achieves competitive performance on multiple benchmarks, offering an efficient and effective solution for capturing cross-variable relationships in 

**Abstract (ZH)**: 跨模态时间序列预测模型VIFO 

---
# PARS: Low-Latency LLM Serving via Pairwise Learning-to-Rank 

**Title (ZH)**: PARS:通过成对学习到排序实现低延迟的大语言模型服务 

**Authors**: Yiheng Tao, Yihe Zhang, Matthew T. Dearing, Xin Wang, Yuping Fan, Zhiling Lan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03243)  

**Abstract**: Efficient scheduling of LLM inference tasks is essential for achieving low latency and high throughput, particularly with the growing use of reasoning-capable LLMs. Traditional strategies like First-Come-First-Serve (FCFS) often suffer from Head-of-Line (HOL) blocking, where long-running tasks delay shorter ones queued behind them. In this paper, we introduce PARS, a prompt-aware LLM task scheduler that improves serving efficiency by approximating shortest-job-first (SJF) scheduling through pairwise ranking with margin ranking loss. PARS focuses on impactful scheduling decisions and is seamlessly integrated into the state-of-the-art LLM serving system vLLM. It effectively predicts response-length-based task ordering, reducing latency with minimal overhead. Extensive experiments across multiple LLMs and real-world inference datasets show that PARS significantly improves performance, including for reasoning workloads. Furthermore, our cross-model evaluations demonstrate that the design generalizes well, enabling effective scheduling even when predictors are trained on different LLMs. 

**Abstract (ZH)**: 一种基于提示感知的LLM任务调度器PARS：通过成对排名与边际排名损失近似最短任务优先调度以提高服务效率 

---
# ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization 

**Title (ZH)**: ReplaceMe：通过深度剪枝和Transformer块线性化实现网络简化 

**Authors**: Dmitriy Shopkhoev, Ammar Ali, Magauiya Zhussip, Valentin Malykh, Stamatios Lefkimmiatis, Nikos Komodakis, Sergey Zagoruyko  

**Link**: [PDF](https://arxiv.org/pdf/2505.02819)  

**Abstract**: We introduce ReplaceMe, a generalized training-free depth pruning method that effectively replaces transformer blocks with a linear operation, while maintaining high performance for low compression ratios. In contrast to conventional pruning approaches that require additional training or fine-tuning, our approach requires only a small calibration dataset that is used to estimate a linear transformation, which approximates the pruned blocks. The estimated linear mapping can be seamlessly merged with the remaining transformer blocks, eliminating the need for any additional network parameters. Our experiments show that ReplaceMe consistently outperforms other training-free approaches and remains highly competitive with state-of-the-art pruning methods that involve extensive retraining/fine-tuning and architectural modifications. Applied to several large language models (LLMs), ReplaceMe achieves up to 25% pruning while retaining approximately 90% of the original model's performance on open benchmarks - without any training or healing steps, resulting in minimal computational overhead (see Fig.1). We provide an open-source library implementing ReplaceMe alongside several state-of-the-art depth pruning techniques, available at this https URL. 

**Abstract (ZH)**: ReplaceMe：一种有效的无需训练的深度剪枝方法，通过线性操作替换变压器块，保持低压缩比下的高性能 

---
# Textured Gaussians for Enhanced 3D Scene Appearance Modeling 

**Title (ZH)**: 纹理高斯函数用于增强的3D场景外观建模 

**Authors**: Brian Chao, Hung-Yu Tseng, Lorenzo Porzi, Chen Gao, Tuotuo Li, Qinbo Li, Ayush Saraf, Jia-Bin Huang, Johannes Kopf, Gordon Wetzstein, Changil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2411.18625)  

**Abstract**: 3D Gaussian Splatting (3DGS) has recently emerged as a state-of-the-art 3D reconstruction and rendering technique due to its high-quality results and fast training and rendering time. However, pixels covered by the same Gaussian are always shaded in the same color up to a Gaussian falloff scaling factor. Furthermore, the finest geometric detail any individual Gaussian can represent is a simple ellipsoid. These properties of 3DGS greatly limit the expressivity of individual Gaussian primitives. To address these issues, we draw inspiration from texture and alpha mapping in traditional graphics and integrate it with 3DGS. Specifically, we propose a new generalized Gaussian appearance representation that augments each Gaussian with alpha~(A), RGB, or RGBA texture maps to model spatially varying color and opacity across the extent of each Gaussian. As such, each Gaussian can represent a richer set of texture patterns and geometric structures, instead of just a single color and ellipsoid as in naive Gaussian Splatting. Surprisingly, we found that the expressivity of Gaussians can be greatly improved by using alpha-only texture maps, and further augmenting Gaussians with RGB texture maps achieves the highest expressivity. We validate our method on a wide variety of standard benchmark datasets and our own custom captures at both the object and scene levels. We demonstrate image quality improvements over existing methods while using a similar or lower number of Gaussians. 

**Abstract (ZH)**: 基于3D高斯体的增强纹理映射技术：一种改进的3D重建与渲染方法 

---
# A Modular Conditional Diffusion Framework for Image Reconstruction 

**Title (ZH)**: 一种模块化的条件扩散框架用于图像重建 

**Authors**: Magauiya Zhussip, Iaroslav Koshelev, Stamatis Lefkimmiatis  

**Link**: [PDF](https://arxiv.org/pdf/2411.05993)  

**Abstract**: Diffusion Probabilistic Models (DPMs) have been recently utilized to deal with various blind image restoration (IR) tasks, where they have demonstrated outstanding performance in terms of perceptual quality. However, the task-specific nature of existing solutions and the excessive computational costs related to their training, make such models impractical and challenging to use for different IR tasks than those that were initially trained for. This hinders their wider adoption, especially by those who lack access to powerful computational resources and vast amount of training data. In this work we aim to address the above issues and enable the successful adoption of DPMs in practical IR-related applications. Towards this goal, we propose a modular diffusion probabilistic IR framework (DP-IR), which allows us to combine the performance benefits of existing pre-trained state-of-the-art IR networks and generative DPMs, while it requires only the additional training of a relatively small module (0.7M params) related to the particular IR task of interest. Moreover, the architecture of the proposed framework allows for a sampling strategy that leads to at least four times reduction of neural function evaluations without suffering any performance loss, while it can also be combined with existing acceleration techniques such as DDIM. We evaluate our model on four benchmarks for the tasks of burst JDD-SR, dynamic scene deblurring, and super-resolution. Our method outperforms existing approaches in terms of perceptual quality while it retains a competitive performance with respect to fidelity metrics. 

**Abstract (ZH)**: 基于扩散概率模型的盲图像恢复框架：模块化设计与高效应用 

---
