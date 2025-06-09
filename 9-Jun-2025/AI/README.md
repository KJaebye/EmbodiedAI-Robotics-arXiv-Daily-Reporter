# Reflect-then-Plan: Offline Model-Based Planning through a Doubly Bayesian Lens 

**Title (ZH)**: 反演再计划：双重贝叶斯视角下的离线模型驱动规划 

**Authors**: Jihwan Jeong, Xiaoyu Wang, Jingmin Wang, Scott Sanner, Pascal Poupart  

**Link**: [PDF](https://arxiv.org/pdf/2506.06261)  

**Abstract**: Offline reinforcement learning (RL) is crucial when online exploration is costly or unsafe but often struggles with high epistemic uncertainty due to limited data. Existing methods rely on fixed conservative policies, restricting adaptivity and generalization. To address this, we propose Reflect-then-Plan (RefPlan), a novel doubly Bayesian offline model-based (MB) planning approach. RefPlan unifies uncertainty modeling and MB planning by recasting planning as Bayesian posterior estimation. At deployment, it updates a belief over environment dynamics using real-time observations, incorporating uncertainty into MB planning via marginalization. Empirical results on standard benchmarks show that RefPlan significantly improves the performance of conservative offline RL policies. In particular, RefPlan maintains robust performance under high epistemic uncertainty and limited data, while demonstrating resilience to changing environment dynamics, improving the flexibility, generalizability, and robustness of offline-learned policies. 

**Abstract (ZH)**: 离线强化学习（RL）在在线探索成本高或不安全时至关重要，但由于数据有限，往往难以应对高的认识性不确定性。现有方法依赖固定保守策略，限制了适应性和泛化能力。为解决这一问题，我们提出了一种新颖的双层贝叶斯离线模型导向（MB）规划方法RefPlan。RefPlan通过将规划重新定义为贝叶斯后验估计来统一不确定性建模和MB规划。在部署时，RefPlan使用实时观察更新对环境动力学的信念，并通过边缘化将不确定性纳入MB规划中。实验结果表明，RefPlan显著提高了保守的离线RL策略的性能。特别是在高认识性不确定性与有限数据下，RefPlan保持了稳定的性能，并展示了对环境动态变化的抗扰性，从而增强了离线学习策略的灵活性、泛化能力和 robustness。 

---
# PersonaAgent: When Large Language Model Agents Meet Personalization at Test Time 

**Title (ZH)**: PersonaAgent：当大型语言模型代理在测试时遇到个性化 

**Authors**: Weizhi Zhang, Xinyang Zhang, Chenwei Zhang, Liangwei Yang, Jingbo Shang, Zhepei Wei, Henry Peng Zou, Zijie Huang, Zhengyang Wang, Yifan Gao, Xiaoman Pan, Lian Xiong, Jingguo Liu, Philip S. Yu, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06254)  

**Abstract**: Large Language Model (LLM) empowered agents have recently emerged as advanced paradigms that exhibit impressive capabilities in a wide range of domains and tasks. Despite their potential, current LLM agents often adopt a one-size-fits-all approach, lacking the flexibility to respond to users' varying needs and preferences. This limitation motivates us to develop PersonaAgent, the first personalized LLM agent framework designed to address versatile personalization tasks. Specifically, PersonaAgent integrates two complementary components - a personalized memory module that includes episodic and semantic memory mechanisms; a personalized action module that enables the agent to perform tool actions tailored to the user. At the core, the persona (defined as unique system prompt for each user) functions as an intermediary: it leverages insights from personalized memory to control agent actions, while the outcomes of these actions in turn refine the memory. Based on the framework, we propose a test-time user-preference alignment strategy that simulate the latest n interactions to optimize the persona prompt, ensuring real-time user preference alignment through textual loss feedback between simulated and ground-truth responses. Experimental evaluations demonstrate that PersonaAgent significantly outperforms other baseline methods by not only personalizing the action space effectively but also scaling during test-time real-world applications. These results underscore the feasibility and potential of our approach in delivering tailored, dynamic user experiences. 

**Abstract (ZH)**: LLM赋能的个性化代理：一种解决多样化个性化任务的先进框架 

---
# Integer Linear Programming Preprocessing for Maximum Satisfiability 

**Title (ZH)**: 整数线性规划预处理 for 最大满足性问题 

**Authors**: Jialu Zhang, Chu-Min Li, Sami Cherif, Shuolin Li, Zhifei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06216)  

**Abstract**: The Maximum Satisfiability problem (MaxSAT) is a major optimization challenge with numerous practical applications. In recent MaxSAT evaluations, most MaxSAT solvers have adopted an ILP solver as part of their portfolios. This paper investigates the impact of Integer Linear Programming (ILP) preprocessing techniques on MaxSAT solving. Experimental results show that ILP preprocessing techniques help WMaxCDCL-OpenWbo1200, the winner of the MaxSAT evaluation 2024 in the unweighted track, solve 15 additional instances. Moreover, current state-of-the-art MaxSAT solvers heavily use an ILP solver in their portfolios, while our proposed approach reduces the need to call an ILP solver in a portfolio including WMaxCDCL or MaxCDCL. 

**Abstract (ZH)**: 最大满意度问题（MaxSAT）是一个重要优化挑战，具有多种实际应用。在最近的MaxSAT评估中，大多数MaxSAT求解器将整数线性规划（ILP）求解器作为其组合中的一部分。本文探讨了整数线性规划（ILP）预处理技术对MaxSAT求解的影响。实验结果表明，ILP预处理技术帮助WMaxCDCL-OpenWbo1200，在无权重轨道2024年MaxSAT评估中的获胜者，解决了额外的15个实例。此外，当前最前沿的MaxSAT求解器在组合中大量使用整数线性规划（ILP）求解器，而我们提出的方法则减少了包含WMaxCDCL或MaxCDCL的组合中调用整数线性规划（ILP）求解器的需求。 

---
# Decomposability-Guaranteed Cooperative Coevolution for Large-Scale Itinerary Planning 

**Title (ZH)**: 可分解性保证的协同共进化大规模行程规划 

**Authors**: Ziyu Zhang, Peilan Xu, Yuetong Sun, Yuhui Shi, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06121)  

**Abstract**: Large-scale itinerary planning is a variant of the traveling salesman problem, aiming to determine an optimal path that maximizes the collected points of interest (POIs) scores while minimizing travel time and cost, subject to travel duration constraints. This paper analyzes the decomposability of large-scale itinerary planning, proving that strict decomposability is difficult to satisfy, and introduces a weak decomposability definition based on a necessary condition, deriving the corresponding graph structures that fulfill this property. With decomposability guaranteed, we propose a novel multi-objective cooperative coevolutionary algorithm for large-scale itinerary planning, addressing the challenges of component imbalance and interactions. Specifically, we design a dynamic decomposition strategy based on the normalized fitness within each component, define optimization potential considering component scale and contribution, and develop a computational resource allocation strategy. Finally, we evaluate the proposed algorithm on a set of real-world datasets. Comparative experiments with state-of-the-art multi-objective itinerary planning algorithms demonstrate the superiority of our approach, with performance advantages increasing as the problem scale grows. 

**Abstract (ZH)**: 大规模行程规划是旅行商问题的一个变体，旨在确定一条最优路径，该路径在满足旅行时长约束的前提下，最大化收集的兴趣点（POIs）分数，同时尽量减少旅行时间和成本。本文分析了大规模行程规划的可分性，证明了严格的可分性难以满足，并引入了一种基于必要条件的弱可分性定义，推导出能够满足该性质的相应图结构。在保证可分性的前提下，我们提出了一种新颖的多目标协同进化算法，针对组件不平衡和交互挑战。具体而言，我们设计了一种基于各个组件标准化适应度的动态分解策略，定义了考虑组件规模和贡献的优化潜力，并开发了计算资源分配策略。最后，我们在一组真实世界的数据集上评估了所提出的算法。与最先进的多目标行程规划算法的比较实验证明了我们方法的优势，性能优势随问题规模的增大而增强。 

---
# CP-Bench: Evaluating Large Language Models for Constraint Modelling 

**Title (ZH)**: CP-Bench: 评估约束建模的大语言模型 

**Authors**: Kostis Michailidis, Dimos Tsouros, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2506.06052)  

**Abstract**: Combinatorial problems are present in a wide range of industries. Constraint Programming (CP) is a well-suited problem-solving paradigm, but its core process, namely constraint modelling, is a bottleneck for wider adoption. Aiming to alleviate this bottleneck, recent studies have explored using Large Language Models (LLMs) as modelling assistants, transforming combinatorial problem descriptions to executable constraint models, similar to coding assistants. However, the existing evaluation datasets for constraint modelling are often limited to small, homogeneous, or domain-specific instances, which do not capture the diversity of real-world scenarios. This work addresses this gap by introducing CP-Bench, a novel benchmark dataset that includes a diverse set of well-known combinatorial problem classes sourced from the CP community, structured explicitly for evaluating LLM-driven CP modelling. With this dataset, and given the variety of constraint modelling frameworks, we compare and evaluate the modelling capabilities of LLMs for three distinct constraint modelling systems, which vary in abstraction level and underlying syntax: the high-level MiniZinc language and Python-based CPMpy library, and the lower-level Python interface of the OR-Tools CP-SAT solver. In order to enhance the ability of LLMs to produce valid constraint models, we systematically evaluate the use of prompt-based and inference-time compute methods adapted from existing LLM-based code generation research. Our results underscore the modelling convenience provided by Python-based frameworks, as well as the effectiveness of documentation-rich system prompts, which, augmented with repeated sampling and self-verification, achieve further improvements, reaching up to 70\% accuracy on this new, highly challenging benchmark. 

**Abstract (ZH)**: 组合优化问题是许多行业的共同挑战。约束编程（CP）是一种合适的问题求解范式，但其核心过程，即约束建模，成为了更广泛采用的瓶颈。为了减轻这一瓶颈，近期研究探索了使用大规模语言模型（LLMs）作为建模助手，将组合优化问题描述转换为可执行的约束模型，类似于代码助手的功能。然而，现有的约束建模评估数据集往往局限于小规模、同质或领域特定的实例，无法捕捉现实场景的多样性。本工作通过引入CP-Bench这一新的基准数据集来填补这一空白，该数据集包含来自CP社区的多样化组合优化问题类，专门用于评估LLM驱动的CP建模。借助该数据集和不同的约束建模框架，我们比较和评估了LLM在三个具有不同抽象级别和底层语法的约束建模系统中的建模能力：MiniZinc高级语言、基于Python的CPMpy库以及OR-Tools CP-SAT求解器的Python接口。为了增强LLMs生成有效约束模型的能力，我们系统性地评估了从现有LLM基于代码生成研究中适应而来的基于提示和推理时计算方法的有效性。我们的结果强调了基于Python的框架提供的建模便利性，以及文档丰富的系统提示的有效性，这些提示在重复采样和自我验证的增强下，实现了进一步的提升，在这一新且极具挑战性的基准中达到高达70%的准确性。 

---
# CrimeMind: Simulating Urban Crime with Multi-Modal LLM Agents 

**Title (ZH)**: CrimeMind: 用多模态LLM代理模拟城市犯罪 

**Authors**: Qingbin Zeng, Ruotong Zhao, Jinzhu Mao, Haoyang Li, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05981)  

**Abstract**: Modeling urban crime is an important yet challenging task that requires understanding the subtle visual, social, and cultural cues embedded in urban environments. Previous work has predominantly focused on rule-based agent-based modeling (ABM) and deep learning methods. ABMs offer interpretability of internal mechanisms but exhibit limited predictive this http URL contrast, deep learning methods are often effective in prediction but are less interpretable and require extensive training data. Moreover, both lines of work lack the cognitive flexibility to adapt to changing environments. Leveraging the capabilities of large language models (LLMs), we propose CrimeMind, a novel LLM-driven ABM framework for simulating urban crime within a multi-modal urban context.A key innovation of our design is the integration of the Routine Activity Theory (RAT) into the agentic workflow of CrimeMind, enabling it to process rich multi-modal urban features and reason about criminal this http URL, RAT requires LLM agents to infer subtle cues in evaluating environmental safety as part of assessing guardianship, which can be challenging for LLMs. To address this, we collect a small-scale human-annotated dataset and align CrimeMind's perception with human judgment via a training-free textual gradient this http URL across four major U.S. cities demonstrate that CrimeMind outperforms both traditional ABMs and deep learning baselines in crime hotspot prediction and spatial distribution accuracy, achieving up to a 24% improvement over the strongest this http URL, we conduct counterfactual simulations of external incidents and policy interventions and it successfully captures the expected changes in crime patterns, demonstrating its ability to reflect counterfactual this http URL, CrimeMind enables fine-grained modeling of individual behaviors and facilitates evaluation of real-world interventions. 

**Abstract (ZH)**: 基于大语言模型的城市犯罪建模：一种多模态城市环境中的犯罪模拟新颖框架 

---
# Preference Learning for AI Alignment: a Causal Perspective 

**Title (ZH)**: 基于因果视角的AI对齐偏好学习 

**Authors**: Katarzyna Kobalczyk, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05967)  

**Abstract**: Reward modelling from preference data is a crucial step in aligning large language models (LLMs) with human values, requiring robust generalisation to novel prompt-response pairs. In this work, we propose to frame this problem in a causal paradigm, providing the rich toolbox of causality to identify the persistent challenges, such as causal misidentification, preference heterogeneity, and confounding due to user-specific factors. Inheriting from the literature of causal inference, we identify key assumptions necessary for reliable generalisation and contrast them with common data collection practices. We illustrate failure modes of naive reward models and demonstrate how causally-inspired approaches can improve model robustness. Finally, we outline desiderata for future research and practices, advocating targeted interventions to address inherent limitations of observational data. 

**Abstract (ZH)**: 从偏好数据中构建奖励模型是使大型语言模型（LLMs）与人类价值观对齐的关键步骤，要求在新的提示-响应对上具备稳固的泛化能力。我们在本文中提出将该问题置于因果框架中，利用因果推断丰富的工具箱来识别持续的挑战，如因果误识别、偏好异质性以及由于用户特定因素引起的混杂。继承因果推断领域的文献，我们识别出可靠泛化所必需的关键假设，并将其与常见的数据收集实践进行对比。我们展示了朴素奖励模型的失败模式，并演示了如何基于因果启发的方法提高模型的鲁棒性。最后，我们概述了未来研究和实践的所需条件，提倡针对性的干预措施以解决观察数据固有的局限性。 

---
# Proactive Assistant Dialogue Generation from Streaming Egocentric Videos 

**Title (ZH)**: 基于流式第一人称视频的主动助理对话生成 

**Authors**: Yichi Zhang, Xin Luna Dong, Zhaojiang Lin, Andrea Madotto, Anuj Kumar, Babak Damavandi, Joyce Chai, Seungwhan Moon  

**Link**: [PDF](https://arxiv.org/pdf/2506.05904)  

**Abstract**: Recent advances in conversational AI have been substantial, but developing real-time systems for perceptual task guidance remains challenging. These systems must provide interactive, proactive assistance based on streaming visual inputs, yet their development is constrained by the costly and labor-intensive process of data collection and system evaluation. To address these limitations, we present a comprehensive framework with three key contributions. First, we introduce a novel data curation pipeline that synthesizes dialogues from annotated egocentric videos, resulting in \dataset, a large-scale synthetic dialogue dataset spanning multiple domains. Second, we develop a suite of automatic evaluation metrics, validated through extensive human studies. Third, we propose an end-to-end model that processes streaming video inputs to generate contextually appropriate responses, incorporating novel techniques for handling data imbalance and long-duration videos. This work lays the foundation for developing real-time, proactive AI assistants capable of guiding users through diverse tasks. Project page: this https URL 

**Abstract (ZH)**: 近期对话人工智能的发展取得了显著进展，但开发基于实时视觉输入的感知任务指导系统仍具挑战性。这些系统必须提供基于流式视觉输入的互动和主动协助，然而其发展受限于数据收集和系统评估的成本高昂和劳动密集。为解决这些限制，我们提出了一种综合框架，包含三个关键贡献。首先，我们引入了一种新颖的数据编纂管道，从标注的自中心视频中合成对话，形成了\( \dataset \)，一个涵盖多个领域的大型合成对话数据集。其次，我们开发了一套自动评估指标，并通过广泛的用户研究进行了验证。第三，我们提出了一种端到端模型，用于处理流式视频输入以生成上下文适当的回答，结合了处理数据不平衡和长时视频的新技术。本工作为开发能够引导用户完成多样化任务的实时主动AI助手奠定了基础。项目页面：this https URL。 

---
# Explainability in Context: A Multilevel Framework Aligning AI Explanations with Stakeholder with LLMs 

**Title (ZH)**: 上下文中的可解释性：一种将AI解释与利益相关者对LLM对齐的多层次框架 

**Authors**: Marilyn Bello, Rafael Bello, Maria-Matilde García, Ann Nowé, Iván Sevillano-García, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2506.05887)  

**Abstract**: The growing application of artificial intelligence in sensitive domains has intensified the demand for systems that are not only accurate but also explainable and trustworthy. Although explainable AI (XAI) methods have proliferated, many do not consider the diverse audiences that interact with AI systems: from developers and domain experts to end-users and society. This paper addresses how trust in AI is influenced by the design and delivery of explanations and proposes a multilevel framework that aligns explanations with the epistemic, contextual, and ethical expectations of different stakeholders. The framework consists of three layers: algorithmic and domain-based, human-centered, and social explainability. We highlight the emerging role of Large Language Models (LLMs) in enhancing the social layer by generating accessible, natural language explanations. Through illustrative case studies, we demonstrate how this approach facilitates technical fidelity, user engagement, and societal accountability, reframing XAI as a dynamic, trust-building process. 

**Abstract (ZH)**: 人工智能在敏感领域应用的增长加剧了对不仅准确而且可解释和可信赖的系统的需求。尽管可解释人工智能（XAI）方法层出不穷，但许多方法并未考虑到与人工智能系统互动的多元受众：从开发者和领域专家到最终用户和社会。本文探讨了设计和传递解释如何影响人们对AI的信任，并提出了一种多层框架，该框架将解释与不同利益相关者的认知、情境和伦理期望相一致。该框架由三个层次构成：算法和领域基础层、以人为本层和社会可解释性层。我们强调大型语言模型（LLMs）在增强社会层方面的作用，通过生成易于理解的自然语言解释。通过示例案例研究，我们展示了这种方法如何促进技术准确度、用户参与和社会问责，并重新定义XAI为一个动态的信任建立过程。 

---
# Trajectory Entropy: Modeling Game State Stability from Multimodality Trajectory Prediction 

**Title (ZH)**: 轨迹熵：基于多模态轨迹预测的游戏状态稳定性建模 

**Authors**: Yesheng Zhang, Wenjian Sun, Yuheng Chen, Qingwei Liu, Qi Lin, Rui Zhang, Xu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05810)  

**Abstract**: Complex interactions among agents present a significant challenge for autonomous driving in real-world scenarios. Recently, a promising approach has emerged, which formulates the interactions of agents as a level-k game framework. It effectively decouples agent policies by hierarchical game levels. However, this framework ignores both the varying driving complexities among agents and the dynamic changes in agent states across game levels, instead treating them uniformly. Consequently, redundant and error-prone computations are introduced into this framework. To tackle the issue, this paper proposes a metric, termed as Trajectory Entropy, to reveal the game status of agents within the level-k game framework. The key insight stems from recognizing the inherit relationship between agent policy uncertainty and the associated driving complexity. Specifically, Trajectory Entropy extracts statistical signals representing uncertainty from the multimodality trajectory prediction results of agents in the game. Then, the signal-to-noise ratio of this signal is utilized to quantify the game status of agents. Based on the proposed Trajectory Entropy, we refine the current level-k game framework through a simple gating mechanism, significantly improving overall accuracy while reducing computational costs. Our method is evaluated on the Waymo and nuPlan datasets, in terms of trajectory prediction, open-loop and closed-loop planning tasks. The results demonstrate the state-of-the-art performance of our method, with precision improved by up to 19.89% for prediction and up to 16.48% for planning. 

**Abstract (ZH)**: 基于级联博弈框架的轨迹熵在自主驾驶中的应用 

---
# Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective 

**Title (ZH)**: 受限采样对于语言模型应当易于实现：从MCMC的角度看他 

**Authors**: Emmanuel Anaya Gonzalez, Sairam Vaidya, Kanghee Park, Ruyi Ji, Taylor Berg-Kirkpatrick, Loris D'Antoni  

**Link**: [PDF](https://arxiv.org/pdf/2506.05754)  

**Abstract**: Constrained decoding enables Language Models (LMs) to produce samples that provably satisfy hard constraints. However, existing constrained-decoding approaches often distort the underlying model distribution, a limitation that is especially problematic in applications like program fuzzing, where one wants to generate diverse and valid program inputs for testing purposes. We propose a new constrained sampling framework based on Markov Chain Monte Carlo (MCMC) that simultaneously satisfies three core desiderata: constraint satisfying (every sample satisfies the constraint), monotonically converging (the sampling process converges to the true conditional distribution), and efficient (high-quality samples emerge in few steps). Our method constructs a proposal distribution over valid outputs and applies a Metropolis-Hastings acceptance criterion based on the LM's likelihood, ensuring principled and efficient exploration of the constrained space. Empirically, our sampler outperforms existing methods on both synthetic benchmarks and real-world program fuzzing tasks. 

**Abstract (ZH)**: 约束解码使语言模型能够生成满足硬约束的样本。然而，现有的约束解码方法 often 常常会扭曲基础模型的概率分布，这一局限性尤其在程序 fuzzing 等应用中问题明显，因为这类应用的目标是生成多样且有效的程序输入以进行测试。我们提出了一种基于马尔可夫链蒙特卡洛（MCMC）的新约束采样框架，该框架同时满足三项核心需求：约束满足性（每个样本都满足约束）、单调收敛性（采样过程收敛到真实的条件分布）和高效性（高质量样本在少量步骤内产生）。我们的方法在有效输出上构建了建议分布，并基于语言模型的似然性应用了梅特罗波利斯-哈斯廷斯接受准则，确保了在约束空间内的原则性和高效性探索。实验证明，我们的采样器在合成基准和实际程序 fuzzing 任务中均优于现有方法。 

---
# SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models 

**Title (ZH)**: SPRINT: 启发式 interleaved 规划与并行执行在推理模型中的实现 

**Authors**: Emil Biju, Shayan Talaei, Zhemin Huang, Mohammadreza Pourreza, Azalia Mirhoseini, Amin Saberi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05745)  

**Abstract**: Large reasoning models (LRMs) excel at complex reasoning tasks but typically generate lengthy sequential chains-of-thought, resulting in long inference times before arriving at the final answer. To address this challenge, we introduce SPRINT, a novel post-training and inference-time framework designed to enable LRMs to dynamically identify and exploit opportunities for parallelization during their reasoning process. SPRINT incorporates an innovative data curation pipeline that reorganizes natural language reasoning trajectories into structured rounds of long-horizon planning and parallel execution. By fine-tuning LRMs on a small amount of such curated data, the models learn to dynamically identify independent subtasks within extended reasoning processes and effectively execute them in parallel. Through extensive evaluations, we show that the models fine-tuned with the SPRINT framework match the performance of reasoning models on complex domains such as mathematics while generating up to ~39% fewer sequential tokens on problems requiring more than 8000 output tokens. Finally, we observe consistent results transferred to two out-of-distribution tasks of GPQA and Countdown with up to 45% and 65% reduction in average sequential tokens for longer reasoning trajectories, while achieving the performance of the fine-tuned reasoning model. 

**Abstract (ZH)**: 大型推理模型 (LRMs) 在复杂推理任务中表现出色，但通常生成较长的顺序性推理链，导致在得出最终答案前需要较长的推断时间。为解决这一挑战，我们提出了SPRINT，一种新颖的后训练及推理时框架，旨在使LRMs能够在推理过程中动态识别和利用并行化的机遇。SPRINT 包含一个创新的数据整理管道，将自然语言推理轨迹重新组织为长时规划和并行执行的结构化轮次。通过在少量此类整理过的数据上微调LRMs，模型能够动态识别扩展推理过程中的独立子任务并有效并行执行。通过广泛评估，我们展示了使用SPRINT框架微调的模型在数学等复杂领域中的性能与推理模型相当，同时在需要超过8000个输出令牌的问题上生成的顺序性令牌最多可减少39%。最后，我们在GPQA和Countdown的两个出分布任务中观察到一致的结果，对于较长的推理轨迹，平均顺序性令牌数分别减少了45%和65%，同时保持了微调推理模型的性能。 

---
# Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties 

**Title (ZH)**: 推理的拓扑结构：通过推理图属性理解大型推理模型 

**Authors**: Gouki Minegishi, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05744)  

**Abstract**: Recent large-scale reasoning models have achieved state-of-the-art performance on challenging mathematical benchmarks, yet the internal mechanisms underlying their success remain poorly understood. In this work, we introduce the notion of a reasoning graph, extracted by clustering hidden-state representations at each reasoning step, and systematically analyze three key graph-theoretic properties: cyclicity, diameter, and small-world index, across multiple tasks (GSM8K, MATH500, AIME 2024). Our findings reveal that distilled reasoning models (e.g., DeepSeek-R1-Distill-Qwen-32B) exhibit significantly more recurrent cycles (about 5 per sample), substantially larger graph diameters, and pronounced small-world characteristics (about 6x) compared to their base counterparts. Notably, these structural advantages grow with task difficulty and model capacity, with cycle detection peaking at the 14B scale and exploration diameter maximized in the 32B variant, correlating positively with accuracy. Furthermore, we show that supervised fine-tuning on an improved dataset systematically expands reasoning graph diameters in tandem with performance gains, offering concrete guidelines for dataset design aimed at boosting reasoning capabilities. By bridging theoretical insights into reasoning graph structures with practical recommendations for data construction, our work advances both the interpretability and the efficacy of large reasoning models. 

**Abstract (ZH)**: 近期大规模推理模型在具有挑战性的数学基准测试中取得了最先进的性能，但其成功背后的内部机制仍 poorly understood。在本工作中，我们引入了推理图的概念，通过在每个推理步骤中聚类隐藏状态表示提取而来，并系统分析了三个关键的图论属性：环路性、直径和小世界指数，跨多个任务（GSM8K、MATH500、AIME 2024）。我们的研究发现，蒸馏推理模型（如DeepSeek-R1-Distill-Qwen-32B）相比其基线模型，表现出显著更多的循环回路（每样本约5次）、更大的图直径以及更为明显的小区间特性（约6倍）。值得注意的是，这些结构优势随着任务难度和模型容量的增加而增强，环检测在14B规模下达到峰值，探索直径在32B变体中最大化，并与准确性呈正相关。此外，我们展示了在改进的数据集上进行监督微调系统地扩展了推理图的直径，并伴随着性能的提升，为设计增强推理能力的数据集提供了具体的指导。通过将对推理图结构的理论见解与数据构建的实践建议相结合，我们的工作不仅提高了大型推理模型的可解释性，还提升了其有效性。 

---
# Population-Proportional Preference Learning from Human Feedback: An Axiomatic Approach 

**Title (ZH)**: 基于人类反馈的人口比例偏好学习：一种公理化方法 

**Authors**: Kihyun Kim, Jiawei Zhang, Asuman Ozdaglar, Pablo A. Parrilo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05619)  

**Abstract**: Conventional preference learning methods often prioritize opinions held more widely when aggregating preferences from multiple evaluators. This may result in policies that are biased in favor of some types of opinions or groups. The objective of this paper is to develop a novel preference learning framework capable of aligning aggregate opinions and policies proportionally with the true population distribution of evaluator preferences. Our approach infers the feasible set of evaluator population distributions directly from pairwise comparison data. Using these estimates, the algorithm constructs a policy that satisfies foundational axioms from social choice theory, namely monotonicity and Pareto efficiency, as well as our newly-introduced axioms of population-proportional representation and population-bounded robustness. We propose a soft-max relaxation method that smoothly trade-offs population-proportional representation with the selection of the Condorcet winner (which beats all other options in pairwise comparisons). Finally, we validate the effectiveness and scalability of our approach through experiments on both tabular recommendation tasks and large-scale language model alignment. 

**Abstract (ZH)**: 传统的偏好学习方法在聚合多个评价者偏好时往往优先考虑更为广泛持有的观点，这可能导致政策偏向某些类型的观点或群体。本文旨在开发一种新的偏好学习框架，使其能够将聚合的偏好和政策按比例与评价者真实人群的偏好分布相一致。我们的方法直接从成对比较数据中推断出评价者人群分布的可行集。利用这些估计值，算法构建一个满足社会选择理论基础公理（即单调性和帕累托效率）以及我们新引入的按人口比例代表性和人口有界稳健性的政策。我们提出了一种软最大松弛方法，该方法平滑地在选择康多塞获胜者（即在成对比较中击败所有其他选项）与按人口比例代表性之间进行权衡。最后，我们通过针对表格式推荐任务和大规模语言模型对齐的实验，验证了该方法的有效性和可扩展性。 

---
# Toward Greater Autonomy in Materials Discovery Agents: Unifying Planning, Physics, and Scientists 

**Title (ZH)**: 向着材料发现代理更大的自主权：统一规划、物理和科学家 

**Authors**: Lianhao Zhou, Hongyi Ling, Keqiang Yan, Kaiji Zhao, Xiaoning Qian, Raymundo Arróyave, Xiaofeng Qian, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.05616)  

**Abstract**: We aim at designing language agents with greater autonomy for crystal materials discovery. While most of existing studies restrict the agents to perform specific tasks within predefined workflows, we aim to automate workflow planning given high-level goals and scientist intuition. To this end, we propose Materials Agent unifying Planning, Physics, and Scientists, known as MAPPS. MAPPS consists of a Workflow Planner, a Tool Code Generator, and a Scientific Mediator. The Workflow Planner uses large language models (LLMs) to generate structured and multi-step workflows. The Tool Code Generator synthesizes executable Python code for various tasks, including invoking a force field foundation model that encodes physics. The Scientific Mediator coordinates communications, facilitates scientist feedback, and ensures robustness through error reflection and recovery. By unifying planning, physics, and scientists, MAPPS enables flexible and reliable materials discovery with greater autonomy, achieving a five-fold improvement in stability, uniqueness, and novelty rates compared with prior generative models when evaluated on the MP-20 data. We provide extensive experiments across diverse tasks to show that MAPPS is a promising framework for autonomous materials discovery. 

**Abstract (ZH)**: 我们旨在设计更具自主性的语言代理以发现晶体材料。现有大多数研究限制代理在预定义工作流中执行特定任务，而我们的目标是在高层目标和科学家直觉的指导下自动化工作流规划。为此，我们提出了一种统合规划、物理和科学家的Materials Agent，简称MAPPS。MAPPS包括工作流规划器、工具代码生成器和科学调解器。工作流规划器使用大规模语言模型（LLMs）生成结构化和多步工作流。工具代码生成器合成可执行的Python代码以执行各种任务，包括调用编码物理学的力场基础模型。科学调解器协调沟通、促进科学家反馈，并通过错误反思和恢复确保鲁棒性。通过统合规划、物理和科学家，MAPPS能够实现更具自主性的灵活且可靠的材料发现，并在MP-20数据上评估时，与先前的生成模型相比，在稳定性和新颖性方面取得了五倍的改进。我们提供了涵盖多种任务的广泛实验，证明了MAPPS是自主材料发现的一个有前景的框架。 

---
# MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark 

**Title (ZH)**: MMTu：大规模多任务表格理解与推理基准 

**Authors**: Junjie Xing, Yeye He, Mengyu Zhou, Haoyu Dong, Shi Han, Lingjiao Chen, Dongmei Zhang, Surajit Chaudhuri, H. V. Jagadish  

**Link**: [PDF](https://arxiv.org/pdf/2506.05587)  

**Abstract**: Tables and table-based use cases play a crucial role in many important real-world applications, such as spreadsheets, databases, and computational notebooks, which traditionally require expert-level users like data engineers, data analysts, and database administrators to operate. Although LLMs have shown remarkable progress in working with tables (e.g., in spreadsheet and database copilot scenarios), comprehensive benchmarking of such capabilities remains limited. In contrast to an extensive and growing list of NLP benchmarks, evaluations of table-related tasks are scarce, and narrowly focus on tasks like NL-to-SQL and Table-QA, overlooking the broader spectrum of real-world tasks that professional users face. This gap limits our understanding and model progress in this important area.
In this work, we introduce MMTU, a large-scale benchmark with over 30K questions across 25 real-world table tasks, designed to comprehensively evaluate models ability to understand, reason, and manipulate real tables at the expert-level. These tasks are drawn from decades' worth of computer science research on tabular data, with a focus on complex table tasks faced by professional users. We show that MMTU require a combination of skills -- including table understanding, reasoning, and coding -- that remain challenging for today's frontier models, where even frontier reasoning models like OpenAI o4-mini and DeepSeek R1 score only around 60%, suggesting significant room for improvement. We highlight key findings in our evaluation using MMTU and hope that this benchmark drives further advances in understanding and developing foundation models for structured data processing and analysis. Our code and data are available at this https URL and this https URL. 

**Abstract (ZH)**: 大规模表格理解基准（MMTU）在众多现实世界任务中的专家级评估 

---
# When Models Know More Than They Can Explain: Quantifying Knowledge Transfer in Human-AI Collaboration 

**Title (ZH)**: 当模型知道超出其解释能力的内容：量化人类-AI协作中的知识迁移 

**Authors**: Quan Shi, Carlos E. Jimenez, Shunyu Yao, Nick Haber, Diyi Yang, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.05579)  

**Abstract**: Recent advancements in AI reasoning have driven substantial improvements across diverse tasks. A critical open question is whether these improvements also yields better knowledge transfer: the ability of models to communicate reasoning in ways humans can understand, apply, and learn from. To investigate this, we introduce Knowledge Integration and Transfer Evaluation (KITE), a conceptual and experimental framework for Human-AI knowledge transfer capabilities and conduct the first large-scale human study (N=118) explicitly designed to measure it. In our two-phase setup, humans first ideate with an AI on problem-solving strategies, then independently implement solutions, isolating model explanations' influence on human understanding. Our findings reveal that although model benchmark performance correlates with collaborative outcomes, this relationship is notably inconsistent, featuring significant outliers, indicating that knowledge transfer requires dedicated optimization. Our analysis identifies behavioral and strategic factors mediating successful knowledge transfer. We release our code, dataset, and evaluation framework to support future work on communicatively aligned models. 

**Abstract (ZH)**: 最近在AI推理方面的进展推动了各类任务上的显著改进。一个关键的开放问题是这些改进是否也带来了更好的知识迁移：模型以人类能够理解、应用和学习的方式进行推理的能力。为了探究这个问题，我们引入了知识整合与转移评估（KITE）的概念和实验框架，并开展了一个第一阶段的大规模人类研究（N=118），以明确测量知识迁移能力。在我们的两阶段设置中，人类首先与AI探讨解决问题的策略，然后独立实施解决方案，以隔离模型解释对人类理解的影响。我们的研究发现表明，尽管模型基准性能与合作结果相关，但这种关系显著不一致，存在大量异常值，表明知识迁移需要专门的优化。我们分析了促进成功知识迁移的行为和战略因素。我们发布了我们的代码、数据集和评估框架，以支持未来通信对齐模型的研究。 

---
# Avoiding Death through Fear Intrinsic Conditioning 

**Title (ZH)**: 通过内在条件作用避免死亡 

**Authors**: Rodney Sanchez, Ferat Sahin, Alexander Ororbia, Jamison Heard  

**Link**: [PDF](https://arxiv.org/pdf/2506.05529)  

**Abstract**: Biological and psychological concepts have inspired reinforcement learning algorithms to create new complex behaviors that expand agents' capacity. These behaviors can be seen in the rise of techniques like goal decomposition, curriculum, and intrinsic rewards, which have paved the way for these complex behaviors. One limitation in evaluating these methods is the requirement for engineered extrinsic for realistic environments. A central challenge in engineering the necessary reward function(s) comes from these environments containing states that carry high negative rewards, but provide no feedback to the agent. Death is one such stimuli that fails to provide direct feedback to the agent. In this work, we introduce an intrinsic reward function inspired by early amygdala development and produce this intrinsic reward through a novel memory-augmented neural network (MANN) architecture. We show how this intrinsic motivation serves to deter exploration of terminal states and results in avoidance behavior similar to fear conditioning observed in animals. Furthermore, we demonstrate how modifying a threshold where the fear response is active produces a range of behaviors that are described under the paradigm of general anxiety disorders (GADs). We demonstrate this behavior in the Miniworld Sidewalk environment, which provides a partially observable Markov decision process (POMDP) and a sparse reward with a non-descriptive terminal condition, i.e., death. In effect, this study results in a biologically-inspired neural architecture and framework for fear conditioning paradigms; we empirically demonstrate avoidance behavior in a constructed agent that is able to solve environments with non-descriptive terminal conditions. 

**Abstract (ZH)**: 生物学和心理学概念启发了强化学习算法创造出新的复杂行为，扩展了代理的能力。这些行为可以体现在目标分解、 Curriculum、内在奖励等技术的兴起中，这些技术为实现这些复杂行为铺平了道路。评估这些方法的一个限制是需要为现实环境工程化设计外在奖励。工程化必要的奖励函数的核心挑战之一在于环境中存在高负向奖励状态，但对代理没有任何反馈。死亡即是这样的刺激之一，未能直接向代理提供反馈。在本工作中，我们提出了一种受早期杏仁体发育启发的内在奖励函数，并通过一种新颖的神经网络增强记忆架构（MANN）生成这种内在奖励。我们展示了这种内在动机如何阻止探索终态，并导致与动物恐惧条件反射相似的逃避行为。我们还展示了通过调整恐惧响应激活的阈值，可以产生一系列描述一般焦虑障碍（GADs）范式的不同行为。我们在Miniworld Sidewalk环境中展示了这种行为，该环境提供了一部分可观测的马尔可夫决策过程（POMDP）和稀疏奖励，带有非描述性的终态条件，即死亡。因此，本研究结果生成了一种生物启发的神经架构和框架，用于恐惧条件反射范式；我们实证展示了能够在解决具有非描述性终态条件的环境中展示逃避行为的代理。 

---
# Towards Data Systems That Are Business Semantic-Centric and AI Agents-Assisted 

**Title (ZH)**: 面向业务语义中心且有AI代理辅助的数据系统 

**Authors**: Cecil Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05520)  

**Abstract**: Contemporary businesses operate in dynamic environments requiring rapid adaptation to achieve goals and maintain competitiveness. Existing data platforms often fall short by emphasizing tools over alignment with business needs, resulting in inefficiencies and delays. To address this gap, I propose the Business Semantics Centric, AI Agents Assisted Data System (BSDS), a holistic system that integrates architecture, workflows, and team organization to ensure data systems are tailored to business priorities rather than dictated by technical constraints. BSDS redefines data systems as dynamic enablers of business success, transforming them from passive tools into active drivers of organizational growth. BSDS has a modular architecture that comprises curated data linked to business entities, a knowledge base for context-aware AI agents, and efficient data pipelines. AI agents play a pivotal role in assisting with data access and system management, reducing human effort, and improving scalability. Complementing this architecture, BSDS incorporates workflows optimized for both exploratory data analysis and production requirements, balancing speed of delivery with quality assurance. A key innovation of BSDS is its incorporation of the human factor. By aligning data team expertise with business semantics, BSDS bridges the gap between technical capabilities and business needs. Validated through real-world implementation, BSDS accelerates time-to-market for data-driven initiatives, enhances cross-functional collaboration, and provides a scalable blueprint for businesses of all sizes. Future research can build on BSDS to explore optimization strategies using complex systems and adaptive network theories, as well as developing autonomous data systems leveraging AI agents. 

**Abstract (ZH)**: 以业务语义为中心、由AI代理辅助的数据系统（BSDS） 

---
# Constructive Symbolic Reinforcement Learning via Intuitionistic Logic and Goal-Chaining Inference 

**Title (ZH)**: 基于直觉逻辑和目标链推理的建设性符号强化学习 

**Authors**: Andrei T. Patrascu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05422)  

**Abstract**: We introduce a novel learning and planning framework that replaces traditional reward-based optimisation with constructive logical inference. In our model, actions, transitions, and goals are represented as logical propositions, and decision-making proceeds by building constructive proofs under intuitionistic logic. This method ensures that state transitions and policies are accepted only when supported by verifiable preconditions -- eschewing probabilistic trial-and-error in favour of guaranteed logical validity. We implement a symbolic agent operating in a structured gridworld, where reaching a goal requires satisfying a chain of intermediate subgoals (e.g., collecting keys to open doors), each governed by logical constraints. Unlike conventional reinforcement learning agents, which require extensive exploration and suffer from unsafe or invalid transitions, our constructive agent builds a provably correct plan through goal chaining, condition tracking, and knowledge accumulation. Empirical comparison with Q-learning demonstrates that our method achieves perfect safety, interpretable behaviour, and efficient convergence with no invalid actions, highlighting its potential for safe planning, symbolic cognition, and trustworthy AI. This work presents a new direction for reinforcement learning grounded not in numeric optimisation, but in constructive logic and proof theory. 

**Abstract (ZH)**: 一种基于构造逻辑推断的新型学习与规划框架：从数值优化到证明理论 

---
# Contextual Memory Intelligence -- A Foundational Paradigm for Human-AI Collaboration and Reflective Generative AI Systems 

**Title (ZH)**: 情境记忆智能：人类与人工智能协作及反思型生成人工智能系统的基石范式 

**Authors**: Kristy Wedel  

**Link**: [PDF](https://arxiv.org/pdf/2506.05370)  

**Abstract**: A critical challenge remains unresolved as generative AI systems are quickly implemented in various organizational settings. Despite significant advances in memory components such as RAG, vector stores, and LLM agents, these systems still have substantial memory limitations. Gen AI workflows rarely store or reflect on the full context in which decisions are made. This leads to repeated errors and a general lack of clarity. This paper introduces Contextual Memory Intelligence (CMI) as a new foundational paradigm for building intelligent systems. It repositions memory as an adaptive infrastructure necessary for longitudinal coherence, explainability, and responsible decision-making rather than passive data. Drawing on cognitive science, organizational theory, human-computer interaction, and AI governance, CMI formalizes the structured capture, inference, and regeneration of context as a fundamental system capability. The Insight Layer is presented in this paper to operationalize this vision. This modular architecture uses human-in-the-loop reflection, drift detection, and rationale preservation to incorporate contextual memory into systems. The paper argues that CMI allows systems to reason with data, history, judgment, and changing context, thereby addressing a foundational blind spot in current AI architectures and governance efforts. A framework for creating intelligent systems that are effective, reflective, auditable, and socially responsible is presented through CMI. This enhances human-AI collaboration, generative AI design, and the resilience of the institutions. 

**Abstract (ZH)**: 生成式AI系统在各种组织环境中快速部署仍面临一个关键挑战。尽管记忆组件（如RAG、向量存储和LLM代理）取得了显著进展，这些系统仍然存在显著的记忆限制。生成式AI的工作流程很少保存或反思决策所处的完整上下文，这导致重复错误和整体缺乏清晰度。本文引入了上下文记忆智能(CMI)作为构建智能系统的新型基础范式。CMI重新定位记忆作为长期连贯性、可解释性和负责任决策所必需的适应性基础设施，而不仅仅是被动的数据。结合认知科学、组织理论、人机交互和AI治理，CMI正式化了结构化捕获、推理和再生上下文作为基本系统能力。本文提出了洞察层（Insight Layer）来实现这一愿景。该模块化架构采用人工介入的反思、漂移检测和理由保留机制，将上下文记忆融入系统。本文认为，CMI使系统能够与数据、历史、判断和变化的上下文进行推理，从而解决当前AI架构和治理努力中的根本性盲点。CMI为创造有效的、反思性的、可审计的和社会负责任的智能系统提供了框架，这增强了人机协作、生成式AI设计和机构的韧性。 

---
# A Path to Loving 

**Title (ZH)**: 一条通往爱的路径 

**Authors**: John Beverley, Regina Hurley  

**Link**: [PDF](https://arxiv.org/pdf/2506.05352)  

**Abstract**: This work lays the foundations for a rigorous ontological characterization of love, addressing its philosophical complexity and scientific relevance, with particular emphasis on psychology and sociology, as well as highlighting ways in which such characterization enhances relevant AI based applications. The position defended here is that love is best understood as a concatenation of passive sensations (e.g., emotional arousal) and active evaluative judgments (e.g., perceiving the beloved as valuable), in the interest of balancing the involuntary aspects of love with its rational accountability. To provide a structured foundation, the paper draws on Basic Formal Ontology (BFO) and other applied ontological methods to differentiate various senses of love. This work engages with objections to the understanding of love as concatenation, particularly concerning the relationship between sensation and judgment. A causal correlation model is defended, ensuring that the affective and cognitive components are linked. By offering a precise and scalable ontological account, this work lays the foundation for future interdisciplinary applications, making love a subject of formal inquiry in ontology engineering, artificial intelligence, and the sciences. 

**Abstract (ZH)**: 本研究为严谨的情爱本体特征化奠定了基础，探讨了情爱的哲学复杂性和科学意义，特别强调心理学和 sociology，并突出了这种特征化如何增强相关的基于 AI 的应用。本文的观点是，情爱最好理解为被动感觉（例如，情绪唤醒）和主动评价判断（例如，感知所爱之人具有价值）的结合，以平衡情爱的非自愿方面和其理性的可问责性。为了提供一个结构化的基础，论文借助基本形式本体论（BFO）和其他应用本体论方法来区分各种情爱的意义。本文回应了将情爱理解为组合所面临的一些反对意见，特别是关于感受和判断之间关系的问题。通过维护因果关联模型，确保情感和认知成分之间的联系。通过提供一个精确且可扩展的本体论解释，本研究为未来跨学科应用奠定了基础，使情爱成为本体工程、人工智能和科学中的形式研究对象。 

---
# Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias 

**Title (ZH)**: 无AspectRatio偏见的神经网络特征谱分析 

**Authors**: Yuanzhe Hu, Kinshuk Goel, Vlad Killiakov, Yaoqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06280)  

**Abstract**: Diagnosing deep neural networks (DNNs) through the eigenspectrum of weight matrices has been an active area of research in recent years. At a high level, eigenspectrum analysis of DNNs involves measuring the heavytailness of the empirical spectral densities (ESD) of weight matrices. It provides insight into how well a model is trained and can guide decisions on assigning better layer-wise training hyperparameters. In this paper, we address a challenge associated with such eigenspectrum methods: the impact of the aspect ratio of weight matrices on estimated heavytailness metrics. We demonstrate that matrices of varying sizes (and aspect ratios) introduce a non-negligible bias in estimating heavytailness metrics, leading to inaccurate model diagnosis and layer-wise hyperparameter assignment. To overcome this challenge, we propose FARMS (Fixed-Aspect-Ratio Matrix Subsampling), a method that normalizes the weight matrices by subsampling submatrices with a fixed aspect ratio. Instead of measuring the heavytailness of the original ESD, we measure the average ESD of these subsampled submatrices. We show that measuring the heavytailness of these submatrices with the fixed aspect ratio can effectively mitigate the aspect ratio bias. We validate our approach across various optimization techniques and application domains that involve eigenspectrum analysis of weights, including image classification in computer vision (CV) models, scientific machine learning (SciML) model training, and large language model (LLM) pruning. Our results show that despite its simplicity, FARMS uniformly improves the accuracy of eigenspectrum analysis while enabling more effective layer-wise hyperparameter assignment in these application domains. In one of the LLM pruning experiments, FARMS reduces the perplexity of the LLaMA-7B model by 17.3% when compared with the state-of-the-art method. 

**Abstract (ZH)**: 基于权值矩阵特征值谱的深度神经网络诊断中方面比的影响及FARMS方法 

---
# Distillation Robustifies Unlearning 

**Title (ZH)**: 蒸馏增强正学习 

**Authors**: Bruce W. Lee, Addie Foote, Alex Infanger, Leni Shor, Harish Kamath, Jacob Goldman-Wetzler, Bryce Woodworth, Alex Cloud, Alexander Matt Turner  

**Link**: [PDF](https://arxiv.org/pdf/2506.06278)  

**Abstract**: Current LLM unlearning methods are not robust: they can be reverted easily with a few steps of finetuning. This is true even for the idealized unlearning method of training to imitate an oracle model that was never exposed to unwanted information, suggesting that output-based finetuning is insufficient to achieve robust unlearning. In a similar vein, we find that training a randomly initialized student to imitate an unlearned model transfers desired behaviors while leaving undesired capabilities behind. In other words, distillation robustifies unlearning. Building on this insight, we propose Unlearn-Noise-Distill-on-Outputs (UNDO), a scalable method that distills an unlearned model into a partially noised copy of itself. UNDO introduces a tunable tradeoff between compute cost and robustness, establishing a new Pareto frontier on synthetic language and arithmetic tasks. At its strongest setting, UNDO matches the robustness of a model retrained from scratch with perfect data filtering while using only 60-80% of the compute and requiring only 0.01% of the pretraining data to be labeled. We also show that UNDO robustifies unlearning on the more realistic Weapons of Mass Destruction Proxy (WMDP) benchmark. Since distillation is widely used in practice, incorporating an unlearning step beforehand offers a convenient path to robust capability removal. 

**Abstract (ZH)**: 当前的LLM去学习方法不够 robust：通过 few steps 的 fine-tuning 就可以轻松恢复。即使对于从未接触过不需要信息的理想去学习方法，训练去学习模型模仿一个 oracle 模型也是如此，这表明基于输出的 fine-tuning 无法实现 robust 去学习。类似地，我们发现，初始化为随机的学生模型去模仿一个未学习模型能够传递所需的行为，同时保留不必要的能力。换句话说，知识蒸馏增强了去学习的 robust 性。基于这一见解，我们提出了去学习-加入噪声-知识蒸馏（UNDO）方法，这是一种可扩展的方法，将一个未学习模型蒸馏为一个部分噪声版本的自己。UNDO 引入了可调节的计算成本与 robust 性之间的权衡，建立了合成语言和算术任务上的新的帕累托前沿。在最强设置下，UNDO 仅需使用 60-80% 的计算量和不到 0.01% 的预训练数据标签，就能达到从头开始重新训练模型且完美数据过滤的 robust 性。我们还展示了 UNDO 在更现实的大规模毁灭性武器代理（WMDP）基准测试中增强了去学习 robust 性。由于知识蒸馏在实践中广泛应用，因此在先前引入一个去学习步骤提供了一条实现 robust 能力去除的便捷途径。 

---
# Cartridges: Lightweight and general-purpose long context representations via self-study 

**Title (ZH)**: cartridges: 自我学习驱动的轻量级通用长期上下文表示 

**Authors**: Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, Christopher Re  

**Link**: [PDF](https://arxiv.org/pdf/2506.06266)  

**Abstract**: Large language models are often used to answer queries grounded in large text corpora (e.g. codebases, legal documents, or chat histories) by placing the entire corpus in the context window and leveraging in-context learning (ICL). Although current models support contexts of 100K-1M tokens, this setup is costly to serve because the memory consumption of the KV cache scales with input length. We explore an alternative: training a smaller KV cache offline on each corpus. At inference time, we load this trained KV cache, which we call a Cartridge, and decode a response. Critically, the cost of training a Cartridge can be amortized across all the queries referencing the same corpus. However, we find that the naive approach of training the Cartridge with next-token prediction on the corpus is not competitive with ICL. Instead, we propose self-study, a training recipe in which we generate synthetic conversations about the corpus and train the Cartridge with a context-distillation objective. We find that Cartridges trained with self-study replicate the functionality of ICL, while being significantly cheaper to serve. On challenging long-context benchmarks, Cartridges trained with self-study match ICL performance while using 38.6x less memory and enabling 26.4x higher throughput. Self-study also extends the model's effective context length (e.g. from 128k to 484k tokens on MTOB) and surprisingly, leads to Cartridges that can be composed at inference time without retraining. 

**Abstract (ZH)**: 一种小型缓存训练方法：通过自学习生成对话来扩展大型语言模型的有效上下文长度并降低成本 

---
# DesignBench: A Comprehensive Benchmark for MLLM-based Front-end Code Generation 

**Title (ZH)**: DesignBench: 基于MLLM的前端代码生成综合基准 

**Authors**: Jingyu Xiao, Ming Wang, Man Ho Lam, Yuxuan Wan, Junliang Liu, Yintong Huo, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06251)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in automated front-end engineering, e.g., generating UI code from visual designs. However, existing front-end UI code generation benchmarks have the following limitations: (1) While framework-based development becomes predominant in modern front-end programming, current benchmarks fail to incorporate mainstream development frameworks. (2) Existing evaluations focus solely on the UI code generation task, whereas practical UI development involves several iterations, including refining editing, and repairing issues. (3) Current benchmarks employ unidimensional evaluation, lacking investigation into influencing factors like task difficulty, input context variations, and in-depth code-level analysis. To bridge these gaps, we introduce DesignBench, a multi-framework, multi-task evaluation benchmark for assessing MLLMs' capabilities in automated front-end engineering. DesignBench encompasses three widely-used UI frameworks (React, Vue, and Angular) alongside vanilla HTML/CSS, and evaluates on three essential front-end tasks (generation, edit, and repair) in real-world development workflows. DesignBench contains 900 webpage samples spanning over 11 topics, 9 edit types, and 6 issue categories, enabling detailed analysis of MLLM performance across multiple dimensions. Our systematic evaluation reveals critical insights into MLLMs' framework-specific limitations, task-related bottlenecks, and performance variations under different conditions, providing guidance for future research in automated front-end development. Our code and data are available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在自动化前端工程中的表现令人瞩目，例如从视觉设计生成UI代码。然而，现有的前端UI代码生成基准存在以下局限性：（1）在现代前端编程中，基于框架的开发已成为主流，当前的基准却未包含主流开发框架。（2）现有评估主要集中在UI代码生成任务上，而实际的UI开发涉及多个迭代，包括编辑细化和问题修复。（3）当前的基准采用单一维度评估，缺乏对任务难度、输入上下文变化以及深入代码层面分析的调查。为弥补这些差距，我们提出了DesignBench，这是一个多框架、多任务评估基准，用于评估MLLMs在自动化前端工程中的能力。DesignBench涵盖了广泛使用的三种UI框架（React、Vue和Angular）以及纯HTML/CSS，并在实际开发工作流中评估三个基本前端任务（生成、编辑和修复）。DesignBench包含900个网页样本，横跨11个主题、9种编辑类型和6类问题类别，使我们能够从多个维度详细分析MLLM的表现。我们的系统性评估揭示了MLLM在框架特定限制、任务相关瓶颈以及在不同条件下的性能变化，为未来自动化前端开发的研究提供了指导。我们的代码和数据可在以下链接获取。 

---
# Visual Graph Arena: Evaluating Visual Conceptualization of Vision and Multimodal Large Language Models 

**Title (ZH)**: 视觉图腾 arena：评估视觉概念化能力的视觉和多模态大型语言模型 

**Authors**: Zahra Babaiee, Peyman M. Kiasari, Daniela Rus, Radu Grosu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06242)  

**Abstract**: Recent advancements in multimodal large language models have driven breakthroughs in visual question answering. Yet, a critical gap persists, `conceptualization'-the ability to recognize and reason about the same concept despite variations in visual form, a basic ability of human reasoning. To address this challenge, we introduce the Visual Graph Arena (VGA), a dataset featuring six graph-based tasks designed to evaluate and improve AI systems' capacity for visual abstraction. VGA uses diverse graph layouts (e.g., Kamada-Kawai vs. planar) to test reasoning independent of visual form. Experiments with state-of-the-art vision models and multimodal LLMs reveal a striking divide: humans achieved near-perfect accuracy across tasks, while models totally failed on isomorphism detection and showed limited success in path/cycle tasks. We further identify behavioral anomalies suggesting pseudo-intelligent pattern matching rather than genuine understanding. These findings underscore fundamental limitations in current AI models for visual understanding. By isolating the challenge of representation-invariant reasoning, the VGA provides a framework to drive progress toward human-like conceptualization in AI visual models. The Visual Graph Arena is available at: \href{this https URL}{this http URL} 

**Abstract (ZH)**: 近期多模态大型语言模型的进展推动了视觉问答领域的突破。然而，仍存在一个关键缺口，即“概念化”——识别和推理同一概念的能力，尽管其视觉形式存在差异，这是人类推理的基本能力。为应对这一挑战，我们提出了视觉图场（VGA），一个包含六项基于图的任务的数据集，旨在评估和提高AI系统在视觉抽象方面的能力。VGA使用多样化的图布局（例如，Kamada-Kawai vs. 平面布局）来测试独立于视觉形式的推理能力。使用最先进的视觉模型和多模态大语言模型的实验揭示了一个明显的鸿沟：人类在所有任务中几乎实现了完美的准确性，而模型在同构性检测上完全失败，并且在路径/环路任务上表现出有限的成功。我们进一步识别了行为异常，表明伪智能的模式匹配而不是真正的理解。这些发现强调了当前AI模型在视觉理解方面的根本局限性。通过将表示不变的推理挑战隔离出来，VGA提供了一个框架，以促进开发出类似人类的概念化能力的AI视觉模型。视觉图场数据集可在以下链接获取：\href{this https URL}{this http URL}。 

---
# Towards an Explainable Comparison and Alignment of Feature Embeddings 

**Title (ZH)**: 可解释的特征嵌入比较与对齐研究 

**Authors**: Mohammad Jalali, Bahar Dibaei Nia, Farzan Farnia  

**Link**: [PDF](https://arxiv.org/pdf/2506.06231)  

**Abstract**: While several feature embedding models have been developed in the literature, comparisons of these embeddings have largely focused on their numerical performance in classification-related downstream applications. However, an interpretable comparison of different embeddings requires identifying and analyzing mismatches between sample groups clustered within the embedding spaces. In this work, we propose the \emph{Spectral Pairwise Embedding Comparison (SPEC)} framework to compare embeddings and identify their differences in clustering a reference dataset. Our approach examines the kernel matrices derived from two embeddings and leverages the eigendecomposition of the difference kernel matrix to detect sample clusters that are captured differently by the two embeddings. We present a scalable implementation of this kernel-based approach, with computational complexity that grows linearly with the sample size. Furthermore, we introduce an optimization problem using this framework to align two embeddings, ensuring that clusters identified in one embedding are also captured in the other model. We provide numerical results demonstrating the SPEC's application to compare and align embeddings on large-scale datasets such as ImageNet and MS-COCO. The code is available at [this https URL](this http URL). 

**Abstract (ZH)**: 谱成对嵌入比较（SPEC）框架：嵌入比较及聚类差异分析 

---
# "We need to avail ourselves of GenAI to enhance knowledge distribution": Empowering Older Adults through GenAI Literacy 

**Title (ZH)**: 我们需要利用生成式AI提升知识传播能力：通过生成式AI素养赋能老年人 

**Authors**: Eunhye Grace Ko, Shaini Nanayakkara, Earl W. Huff Jr  

**Link**: [PDF](https://arxiv.org/pdf/2506.06225)  

**Abstract**: As generative AI (GenAI) becomes increasingly widespread, it is crucial to equip users, particularly vulnerable populations such as older adults (65 and older), with the knowledge to understand its benefits and potential risks. Older adults often exhibit greater reservations about adopting emerging technologies and require tailored literacy support. Using a mixed methods approach, this study examines strategies for delivering GenAI literacy to older adults through a chatbot named Litti, evaluating its impact on their AI literacy (knowledge, safety, and ethical use). The quantitative data indicated a trend toward improved AI literacy, though the results were not statistically significant. However, qualitative interviews revealed diverse levels of familiarity with generative AI and a strong desire to learn more. Findings also show that while Litti provided a positive learning experience, it did not significantly enhance participants' trust or sense of safety regarding GenAI. This exploratory case study highlights the challenges and opportunities in designing AI literacy education for the rapidly growing older adult population. 

**Abstract (ZH)**: 随着生成式人工智能（GenAI）的广泛应用，为用户，特别是老年人（65岁及以上）等脆弱群体提供相关知识以理解其益处和潜在风险变得至关重要。老年人对采用新兴技术往往表现出更大的顾虑，并需要个性化的信息素养支持。采用混合方法，本研究通过名为Litti的聊天机器人探讨向老年人传授GenAI信息素养的策略，并评估其对其AI信息素养（知识、安全和伦理使用）的影响。定量数据分析显示AI信息素养有所提高，但结果不具备统计显著性。然而，定性访谈揭示了人们对生成式AI的不同熟悉程度，并表达了强烈的学习愿望。研究结果还表明，虽然Litti提供了积极的学习体验，但并未显著增强参与者对GenAI的信任感或安全感。该探索性案例研究突显了为迅速增长的老年群体设计AI信息素养教育面临的挑战和机遇。 

---
# GenIR: Generative Visual Feedback for Mental Image Retrieval 

**Title (ZH)**: GenIR: 生成式视觉反馈的心理图像检索 

**Authors**: Diji Yang, Minghao Liu, Chung-Hsiang Lo, Yi Zhang, James Davis  

**Link**: [PDF](https://arxiv.org/pdf/2506.06220)  

**Abstract**: Vision-language models (VLMs) have shown strong performance on text-to-image retrieval benchmarks. However, bridging this success to real-world applications remains a challenge. In practice, human search behavior is rarely a one-shot action. Instead, it is often a multi-round process guided by clues in mind, that is, a mental image ranging from vague recollections to vivid mental representations of the target image. Motivated by this gap, we study the task of Mental Image Retrieval (MIR), which targets the realistic yet underexplored setting where users refine their search for a mentally envisioned image through multi-round interactions with an image search engine. Central to successful interactive retrieval is the capability of machines to provide users with clear, actionable feedback; however, existing methods rely on indirect or abstract verbal feedback, which can be ambiguous, misleading, or ineffective for users to refine the query. To overcome this, we propose GenIR, a generative multi-round retrieval paradigm leveraging diffusion-based image generation to explicitly reify the AI system's understanding at each round. These synthetic visual representations provide clear, interpretable feedback, enabling users to refine their queries intuitively and effectively. We further introduce a fully automated pipeline to generate a high-quality multi-round MIR dataset. Experimental results demonstrate that GenIR significantly outperforms existing interactive methods in the MIR scenario. This work establishes a new task with a dataset and an effective generative retrieval method, providing a foundation for future research in this direction. 

**Abstract (ZH)**: 基于视觉-语言模型的思维图像检索（Mental Image Retrieval, MIR）：一种生成式的多轮检索范式 

---
# Can Theoretical Physics Research Benefit from Language Agents? 

**Title (ZH)**: 理论物理学研究能够从语言代理中受益吗？ 

**Authors**: Sirui Lu, Zhijing Jin, Terry Jingchen Zhang, Pavel Kos, J. Ignacio Cirac, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2506.06214)  

**Abstract**: Large Language Models (LLMs) are rapidly advancing across diverse domains, yet their application in theoretical physics research is not yet mature. This position paper argues that LLM agents can potentially help accelerate theoretical, computational, and applied physics when properly integrated with domain knowledge and toolbox. We analyze current LLM capabilities for physics -- from mathematical reasoning to code generation -- identifying critical gaps in physical intuition, constraint satisfaction, and reliable reasoning. We envision future physics-specialized LLMs that could handle multimodal data, propose testable hypotheses, and design experiments. Realizing this vision requires addressing fundamental challenges: ensuring physical consistency, and developing robust verification methods. We call for collaborative efforts between physics and AI communities to help advance scientific discovery in physics. 

**Abstract (ZH)**: 大型语言模型在理论物理学研究中的应用尚不成熟，但正迅速迈向多元化领域。本文立场论文认为，当适当结合领域知识和工具箱时，大型语言模型代理有望加速理论物理、计算物理和应用物理的发展。我们分析了当前物理领域的大型语言模型能力——从数学推理到代码生成，并识别出物理直觉、约束满足和可靠推理方面的关键差距。我们展望未来专门化于物理学的大规模语言模型，它们能够处理多模态数据、提出可测试的假设并设计实验。要实现这一愿景，需要解决根本性的挑战：确保物理一致性和开发稳健的验证方法。我们呼吁物理学与人工智能社区之间的合作，以推动物理学中的科学发现。 

---
# PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts 

**Title (ZH)**: PuzzleWorld: 一款用于谜题 hunt 多模态开放性推理的基准测试 

**Authors**: Hengzhi Li, Brendon Jiang, Alexander Naehu, Regan Song, Justin Zhang, Megan Tjandrasuwita, Chanakya Ekbote, Steven-Shine Chen, Adithya Balachandran, Wei Dai, Rebecca Chang, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06211)  

**Abstract**: Puzzlehunts are a genre of complex, multi-step puzzles lacking well-defined problem definitions. In contrast to conventional reasoning benchmarks consisting of tasks with clear instructions, puzzlehunts require models to discover the underlying problem structure from multimodal evidence and iterative reasoning, mirroring real-world domains such as scientific discovery, exploratory data analysis, or investigative problem-solving. Despite recent progress in foundation models, their performance on such open-ended settings remains largely untested. In this paper, we introduce PuzzleWorld, a large-scale benchmark of 667 puzzlehunt-style problems designed to assess step-by-step, open-ended, and creative multimodal reasoning. Each puzzle is annotated with the final solution, detailed reasoning traces, and cognitive skill labels, enabling holistic benchmarking and fine-grained diagnostic analysis. Most state-of-the-art models achieve only 1-2% final answer accuracy, with the best model solving only 14% of puzzles and reaching 40% stepwise accuracy. To demonstrate the value of our reasoning annotations, we show that fine-tuning a small model on reasoning traces improves stepwise reasoning from 4% to 11%, while training on final answers alone degrades performance to near zero. Our error analysis reveals that current models exhibit myopic reasoning, are bottlenecked by the limitations of language-based inference, and lack sketching capabilities crucial for visual and spatial reasoning. We release PuzzleWorld at this https URL to support future work on building more general, open-ended, and creative reasoning systems. 

**Abstract (ZH)**: 谜题搜寻是一种缺乏明确问题定义的复杂多步谜题类型。与传统推理基准中任务明确指示的模式不同，谜题搜寻要求模型从多模态证据和迭代推理中发现潜在的问题结构，这与科学发现、探索性数据分析或调查性问题解决等现实世界领域相映射。尽管基础模型的进展取得了进步，但在这种开放式设置中的性能仍然很少被测试。本文我们引入了PuzzleWorld，这是一个包含667个谜题搜寻样问题的大规模基准，旨在评估逐步、开放式和创造性的多模态推理能力。每个谜题都带有最终解决方案、详细的推理轨迹和认知技能标签，支持全面基准测试和细粒度诊断分析。大多数最先进的模型仅达到1-2%的最终答案准确性，最佳模型也只能解决14%的谜题，并且逐步准确性达到40%。为了展示我们推理注释的价值，我们展示了在推理轨迹上微调一个小模型能将逐步推理从4%提升到11%，而仅使用最终答案进行训练则会将性能降低到接近零。我们的错误分析揭示了当前模型表现出短视推理、受限于基于语言的推理的局限性，并且缺乏对于视觉和空间推理至关重要的绘图能力。我们在此提供PuzzleWorld以支持未来构建更通用、开放式和创造性推理系统的研究。 

---
# Building Models of Neurological Language 

**Title (ZH)**: 构建神经语言模型 

**Authors**: Henry Watkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.06208)  

**Abstract**: This report documents the development and evaluation of domain-specific language models for neurology. Initially focused on building a bespoke model, the project adapted to rapid advances in open-source and commercial medical LLMs, shifting toward leveraging retrieval-augmented generation (RAG) and representational models for secure, local deployment. Key contributions include the creation of neurology-specific datasets (case reports, QA sets, textbook-derived data), tools for multi-word expression extraction, and graph-based analyses of medical terminology. The project also produced scripts and Docker containers for local hosting. Performance metrics and graph community results are reported, with future possible work open for multimodal models using open-source architectures like phi-4. 

**Abstract (ZH)**: 本报告记录了神经学领域特定语言模型的发展与评估。项目最初专注于构建定制模型，后根据开源和商业医疗LLM的快速进步，转向利用检索增强生成（RAG）和表示模型实现安全的本地部署。关键贡献包括创建神经学专用数据集（病例报告、问答集、教材衍生数据）、多词表达提取工具以及医学术语的图谱分析。项目还生成了本地托管的脚本和Docker容器。报告了性能指标和图社区分析结果，并开放了使用开源架构如phi-4的多模态模型未来工作。 

---
# Astra: Toward General-Purpose Mobile Robots via Hierarchical Multimodal Learning 

**Title (ZH)**: Astra: 基于层次多模态学习的通用型移动机器人研究 

**Authors**: Sheng Chen, Peiyu He, Jiaxin Hu, Ziyang Liu, Yansheng Wang, Tao Xu, Chi Zhang, Chongchong Zhang, Chao An, Shiyu Cai, Duo Cao, Kangping Chen, Shuai Chu, Tianwei Chu, Mingdi Dan, Min Du, Weiwei Fang, Pengyou Fu, Junkai Hu, Xiaowei Jiang, Zhaodi Jiang, Fuxuan Li, Jun Li, Minghui Li, Mingyao Li, Yanchang Li, Zhibin Li, Guangming Liu, Kairui Liu, Lihao Liu, Weizhi Liu, Xiaoshun Liu, Yufei Liu, Yunfei Liu, Qiang Lu, Yuanfei Luo, Xiang Lv, Hongying Ma, Sai Ma, Lingxian Mi, Sha Sa, Hongxiang Shu, Lei Tian, Chengzhi Wang, Jiayu Wang, Kaijie Wang, Qingyi Wang, Renwen Wang, Tao Wang, Wei Wang, Xirui Wang, Chao Wei, Xuguang Wei, Zijun Xia, Zhaohao Xiao, Tingshuai Yan, Liyan Yang, Yifan Yang, Zhikai Yang, Zhong Yin, Li Yuan, Liuchun Yuan, Chi Zhang, Jinyang Zhang, Junhui Zhang, Linge Zhang, Zhenyi Zhang, Zheyu Zhang, Dongjie Zhu, Hang Li, Yangang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06205)  

**Abstract**: Modern robot navigation systems encounter difficulties in diverse and complex indoor environments. Traditional approaches rely on multiple modules with small models or rule-based systems and thus lack adaptability to new environments. To address this, we developed Astra, a comprehensive dual-model architecture, Astra-Global and Astra-Local, for mobile robot navigation. Astra-Global, a multimodal LLM, processes vision and language inputs to perform self and goal localization using a hybrid topological-semantic graph as the global map, and outperforms traditional visual place recognition methods. Astra-Local, a multitask network, handles local path planning and odometry estimation. Its 4D spatial-temporal encoder, trained through self-supervised learning, generates robust 4D features for downstream tasks. The planning head utilizes flow matching and a novel masked ESDF loss to minimize collision risks for generating local trajectories, and the odometry head integrates multi-sensor inputs via a transformer encoder to predict the relative pose of the robot. Deployed on real in-house mobile robots, Astra achieves high end-to-end mission success rate across diverse indoor environments. 

**Abstract (ZH)**: 现代机器人导航系统在多样复杂的室内环境中面临挑战。传统的 approach 依赖多个模块和小规模模型或基于规则的系统，因而缺乏对新环境的适应性。为了解决这一问题，我们开发了 Astra，一种综合双模型架构，包括 Astra-Global 和 Astra-Local，用于移动机器人导航。Astra-Global 是一个多模态大语言模型，处理视觉和语言输入，使用混合拓扑-语义图作为全局地图进行自我和目标定位，并优于传统视觉场所识别方法。Astra-Local 是一个多任务网络，处理局部路径规划和里程计估计。其通过自监督学习训练的 4D 空间-时间编码器生成鲁棒的 4D 特征以供下游任务使用。规划头利用流匹配和一种新颖的掩码 ESDF 损失来最小化碰撞风险以生成局部轨迹，里程计头通过变压器编码整合多传感器输入以预测机器人的相对姿态。在实际室内移动机器人上部署的 Astra 实现了高端到端任务成功率，适用于多种室内环境。 

---
# MLOps with Microservices: A Case Study on the Maritime Domain 

**Title (ZH)**: 微服务中的MLOps： Maritime领域案例研究 

**Authors**: Renato Cordeiro Ferreira, Rowanne Trapmann, Willem-Jan van den Heuvel  

**Link**: [PDF](https://arxiv.org/pdf/2506.06202)  

**Abstract**: This case study describes challenges and lessons learned on building Ocean Guard: a Machine Learning-Enabled System (MLES) for anomaly detection in the maritime domain. First, the paper presents the system's specification, and architecture. Ocean Guard was designed with a microservices' architecture to enable multiple teams to work on the project in parallel. Then, the paper discusses how the developers adapted contract-based design to MLOps for achieving that goal. As a MLES, Ocean Guard employs code, model, and data contracts to establish guidelines between its services. This case study hopes to inspire software engineers, machine learning engineers, and data scientists to leverage similar approaches for their systems. 

**Abstract (ZH)**: 基于机器学习的海事异常检测系统Ocean Guard的案例研究：挑战与经验教训 

---
# semantic-features: A User-Friendly Tool for Studying Contextual Word Embeddings in Interpretable Semantic Spaces 

**Title (ZH)**: 语义特征：一个用于研究可解释语义空间中上下文词嵌入的用户友好工具 

**Authors**: Jwalanthi Ranganathan, Rohan Jha, Kanishka Misra, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2506.06169)  

**Abstract**: We introduce semantic-features, an extensible, easy-to-use library based on Chronis et al. (2023) for studying contextualized word embeddings of LMs by projecting them into interpretable spaces. We apply this tool in an experiment where we measure the contextual effect of the choice of dative construction (prepositional or double object) on the semantic interpretation of utterances (Bresnan, 2007). Specifically, we test whether "London" in "I sent London the letter." is more likely to be interpreted as an animate referent (e.g., as the name of a person) than in "I sent the letter to London." To this end, we devise a dataset of 450 sentence pairs, one in each dative construction, with recipients being ambiguous with respect to person-hood vs. place-hood. By applying semantic-features, we show that the contextualized word embeddings of three masked language models show the expected sensitivities. This leaves us optimistic about the usefulness of our tool. 

**Abstract (ZH)**: 我们介绍了基于Chronis等人（2023）的可扩展且易于使用的semantic-features库，用于通过将LM的上下文化词嵌入投影到可解释的空间中来研究这些词嵌入。我们在一项实验中应用了该工具，该实验测量了供词结构（介词或双宾语）选择对其语义解释的影响（Bresnan，2007）。具体来说，我们测试了在句子“I sent London the letter.”中，“London”是否比在“I sent the letter to London.”中更有可能被解释为有生命的事物（例如，某个人的名字）。为此，我们设计了一个包含450个句子配对的数据集，每个配对句子使用不同的供词结构，且接收者在是否为人物或地点方面存在歧义。通过应用semantic-features库，我们展示了三个掩码语言模型的上下文化词嵌入显示出预期的敏感性。这使我们对工具的实用性感到乐观。 

---
# The Lock-in Hypothesis: Stagnation by Algorithm 

**Title (ZH)**: 锁定假设：算法导致的停滞 

**Authors**: Tianyi Alex Qiu, Zhonghao He, Tejasveer Chugh, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2506.06166)  

**Abstract**: The training and deployment of large language models (LLMs) create a feedback loop with human users: models learn human beliefs from data, reinforce these beliefs with generated content, reabsorb the reinforced beliefs, and feed them back to users again and again. This dynamic resembles an echo chamber. We hypothesize that this feedback loop entrenches the existing values and beliefs of users, leading to a loss of diversity and potentially the lock-in of false beliefs. We formalize this hypothesis and test it empirically with agent-based LLM simulations and real-world GPT usage data. Analysis reveals sudden but sustained drops in diversity after the release of new GPT iterations, consistent with the hypothesized human-AI feedback loop. Code and data available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）的训练与部署与人类用户形成一个反馈循环：模型从数据中学习人类的信念，通过生成内容强化这些信念，重新吸收强化后的信念，并再次反馈给用户。这一动态类似于回音室效应。我们假设这个反馈循环巩固了用户的现有价值观和信念，导致多样性的损失，并可能锁定错误的信念。我们形式化这一假设，并通过基于代理的LLM模拟和实际使用的GPT数据进行实证测试。分析发现，在新GPT迭代发布后，多样性突然但持续地下降，与假设的人工智能-人类反馈循环一致。代码和数据可在以下链接获取：this https URL 

---
# (AI peers) are people learning from the same standpoint: Perception of AI characters in a Collaborative Science Investigation 

**Title (ZH)**: AI同伴是从相同视角学习的人：合作科学调查中关于AI角色的感知 

**Authors**: Eunhye Grace Ko, Soo Hyoung Joo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06165)  

**Abstract**: While the complexity of 21st-century demands has promoted pedagogical approaches to foster complex competencies, a persistent gap remains between in-class learning activities and individualized learning or assessment practices. To address this, studies have explored the use of AI-generated characters in learning and assessment. One attempt is scenario-based assessment (SBA), a technique that not only measures but also fosters the development of competencies throughout the assessment process. SBA introduces simulated agents to provide an authentic social-interactional context, allowing for the assessment of competency-based constructs while mitigating the unpredictability of real-life interactions. Recent advancements in multimodal AI, such as text-to-video technology, allow these agents to be enhanced into AI-generated characters. This mixed-method study investigates how learners perceive AI characters taking the role of mentor and teammates in an SBA mirroring the context of a collaborative science investigation. Specifically, we examined the Likert scale responses of 56 high schoolers regarding trust, social presence, and effectiveness. We analyzed the relationships between these factors and their impact on the intention to adopt AI characters through PLS-SEM. Our findings indicated that learners' trust shaped their sense of social presence with the AI characters, enhancing perceived effectiveness. Qualitative analysis further highlighted factors that foster trust, such as material credibility and alignment with learning goals, as well as the pivotal role of social presence in creating a collaborative context.
This paper was accepted as an full paper for AIED 2025. 

**Abstract (ZH)**: 21世纪需求复杂性促进教学方法以培养复杂能力，但课堂学习活动与个性化学习或评估实践之间仍存在差距。为解决这一问题，研究探索了使用AI生成角色在学习和评估中的应用。其中一项尝试是情景化评估（SBA），这种方法不仅能评估，还能在评估过程中促进能力的发展。SBA引入了模拟代理，提供了一个真实的社会互动背景，允许评估基于能力的构建体，同时减轻了实时互动的不确定性。最新的人工智能多模态技术，如文本转视频技术，使这些代理能够升级为AI生成的角色。本混合方法研究探讨了高中生对担任导师和队友角色的AI角色在模拟科学协作情境中的感知，具体分析了关于信任、社会临在感和有效性的李克特量表反应。通过PLS-SEM分析了这些因素之间的关系及其对采用AI角色意愿的影响。研究发现，学习者的信任影响了他们对AI角色的社会临在感感知，增强了感知的有效性。定性分析进一步强调了促进信任的因素，如材料可信度和与学习目标的一致性，以及社会临在感在创造协作情境中的关键作用。本文已被接受为AIED 2025的全文论文。 

---
# Recommender systems, stigmergy, and the tyranny of popularity 

**Title (ZH)**: 推荐系统、群体信息素机制与流行性的霸权 

**Authors**: Zackary Okun Dunivin, Paul E. Smaldino  

**Link**: [PDF](https://arxiv.org/pdf/2506.06162)  

**Abstract**: Scientific recommender systems, such as Google Scholar and Web of Science, are essential tools for discovery. Search algorithms that power work through stigmergy, a collective intelligence mechanism that surfaces useful paths through repeated engagement. While generally effective, this ``rich-get-richer'' dynamic results in a small number of high-profile papers that dominate visibility. This essay argues argue that these algorithm over-reliance on popularity fosters intellectual homogeneity and exacerbates structural inequities, stifling innovative and diverse perspectives critical for scientific progress. We propose an overhaul of search platforms to incorporate user-specific calibration, allowing researchers to manually adjust the weights of factors like popularity, recency, and relevance. We also advise platform developers on how word embeddings and LLMs could be implemented in ways that increase user autonomy. While our suggestions are particularly pertinent to aligning recommender systems with scientific values, these ideas are broadly applicable to information access systems in general. Designing platforms that increase user autonomy is an important step toward more robust and dynamic information 

**Abstract (ZH)**: 科学型推荐系统，如Google Scholar和Web of Science，是发现的重要工具。基于集体智能机制的搜索算法通过共鸣效应揭示有用路径，虽然通常有效，但这种“富者愈富”的动态导致少数高影响力论文主导了可见度。本文认为，推荐系统过度依赖流行度促进了学术同质性，并加剧了结构性不平等，扼杀了对于科学进步至关重要的创新性和多样性视角。我们建议对搜索平台进行改革，纳入用户特定校准，使研究者能够手动调整流行度、新颖性和相关性等因素的权重。我们还建议平台开发者如何通过引入词嵌入和大语言模型来增强用户自主权。虽然我们的建议特别适用于使推荐系统与科学价值观相符，但这些想法对一般的信息访问系统也有广泛的应用价值。设计增强用户自主权的平台是朝着更具弹性和动态信息获取方向迈出的重要一步。 

---
# Joint-GCG: Unified Gradient-Based Poisoning Attacks on Retrieval-Augmented Generation Systems 

**Title (ZH)**: Joint-GCG：统一基于梯度的检索增强生成系统中毒攻击 

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06151)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by retrieving relevant documents from external corpora before generating responses. This approach significantly expands LLM capabilities by leveraging vast, up-to-date external knowledge. However, this reliance on external knowledge makes RAG systems vulnerable to corpus poisoning attacks that manipulate generated outputs via poisoned document injection. Existing poisoning attack strategies typically treat the retrieval and generation stages as disjointed, limiting their effectiveness. We propose Joint-GCG, the first framework to unify gradient-based attacks across both retriever and generator models through three innovations: (1) Cross-Vocabulary Projection for aligning embedding spaces, (2) Gradient Tokenization Alignment for synchronizing token-level gradient signals, and (3) Adaptive Weighted Fusion for dynamically balancing attacking objectives. Evaluations demonstrate that Joint-GCG achieves at most 25% and an average of 5% higher attack success rate than previous methods across multiple retrievers and generators. While optimized under a white-box assumption, the generated poisons show unprecedented transferability to unseen models. Joint-GCG's innovative unification of gradient-based attacks across retrieval and generation stages fundamentally reshapes our understanding of vulnerabilities within RAG systems. Our code is available at this https URL. 

**Abstract (ZH)**: 基于检索的生成(RAG)系统通过在生成响应之前从外部语料库检索相关文档，增强了大型语言模型(LLM)的功能。这种方法通过利用大量的最新外部知识，显著扩展了LLM的能力。然而，对外部知识的依赖使得RAG系统容易受到通过注入污染文档进行操纵的语料库污染攻击。现有的污染攻击策略通常将检索和生成阶段视为不相关的，限制了它们的效果。我们提出了一种名为Joint-GCG的新框架，它是第一个通过三项创新统一跨检索器和生成器的梯度攻击的框架：(1) 跨词汇投影以对齐嵌入空间，(2) 梯度标记同步以同步标记级别梯度信号，(3) 自适应加权融合以动态平衡攻击目标。评估结果显示，Joint-GCG在多个检索器和生成器上分别获得最高25%和平均5%更高的攻击成功率，而优化条件下生成的污染剂对未见过的模型显示出前所未有的迁移性。Joint-GCG从根本上重塑了我们对RAG系统中的潜在风险的理解。我们的代码可在以下链接获取。 

---
# Phonetically-Augmented Discriminative Rescoring for Voice Search Error Correction 

**Title (ZH)**: Phonetic增强判别重评分在语音搜索错误纠正中的应用 

**Authors**: Christophe Van Gysel, Maggie Wu, Lyan Verwimp, Caglar Tirkaz, Marco Bertola, Zhihong Lei, Youssef Oualil  

**Link**: [PDF](https://arxiv.org/pdf/2506.06117)  

**Abstract**: End-to-end (E2E) Automatic Speech Recognition (ASR) models are trained using paired audio-text samples that are expensive to obtain, since high-quality ground-truth data requires human annotators. Voice search applications, such as digital media players, leverage ASR to allow users to search by voice as opposed to an on-screen keyboard. However, recent or infrequent movie titles may not be sufficiently represented in the E2E ASR system's training data, and hence, may suffer poor recognition.
In this paper, we propose a phonetic correction system that consists of (a) a phonetic search based on the ASR model's output that generates phonetic alternatives that may not be considered by the E2E system, and (b) a rescorer component that combines the ASR model recognition and the phonetic alternatives, and select a final system output.
We find that our approach improves word error rate between 4.4 and 7.6% relative on benchmarks of popular movie titles over a series of competitive baselines. 

**Abstract (ZH)**: 端到端自动语音识别模型中的 phonetic 修正系统及其应用改进词错误率 4.4% 至 7.6% 

---
# Towards Lifecycle Unlearning Commitment Management: Measuring Sample-level Unlearning Completeness 

**Title (ZH)**: 面向生命周期的遗忘承诺管理：样本级别遗忘完成度度量 

**Authors**: Cheng-Long Wang, Qi Li, Zihang Xiang, Yinzhi Cao, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06112)  

**Abstract**: Growing concerns over data privacy and security highlight the importance of machine unlearning--removing specific data influences from trained models without full retraining. Techniques like Membership Inference Attacks (MIAs) are widely used to externally assess successful unlearning. However, existing methods face two key limitations: (1) maximizing MIA effectiveness (e.g., via online attacks) requires prohibitive computational resources, often exceeding retraining costs; (2) MIAs, designed for binary inclusion tests, struggle to capture granular changes in approximate unlearning. To address these challenges, we propose the Interpolated Approximate Measurement (IAM), a framework natively designed for unlearning inference. IAM quantifies sample-level unlearning completeness by interpolating the model's generalization-fitting behavior gap on queried samples. IAM achieves strong performance in binary inclusion tests for exact unlearning and high correlation for approximate unlearning--scalable to LLMs using just one pre-trained shadow model. We theoretically analyze how IAM's scoring mechanism maintains performance efficiently. We then apply IAM to recent approximate unlearning algorithms, revealing general risks of both over-unlearning and under-unlearning, underscoring the need for stronger safeguards in approximate unlearning systems. The code is available at this https URL. 

**Abstract (ZH)**: Growing Concerns over Data Privacy and Security Highlight the Importance of Machine Unlearning: A Framework for Efficient Approximate Unlearning Evaluation 

---
# Text-to-LoRA: Instant Transformer Adaption 

**Title (ZH)**: Text-to-LoRA：即时Transformer适配 

**Authors**: Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, Robert Tjarko Lange  

**Link**: [PDF](https://arxiv.org/pdf/2506.06105)  

**Abstract**: While Foundation Models provide a general tool for rapid content creation, they regularly require task-specific adaptation. Traditionally, this exercise involves careful curation of datasets and repeated fine-tuning of the underlying model. Fine-tuning techniques enable practitioners to adapt foundation models for many new applications but require expensive and lengthy training while being notably sensitive to hyper-parameter choices. To overcome these limitations, we introduce Text-to-LoRA (T2L), a model capable of adapting Large Language Models on the fly solely based on a natural language description of the target task. T2L is a hypernetwork trained to construct LoRAs in a single inexpensive forward pass. After training T2L on a suite of 9 pre-trained LoRA adapters (GSM8K, Arc, etc.), we show that the ad-hoc reconstructed LoRA instances match the performance of task-specific adapters across the corresponding test sets. Furthermore, T2L can compress hundreds of LoRA instances and zero-shot generalize to entirely unseen tasks. This approach provides a significant step towards democratizing the specialization of foundation models and enables language-based adaptation with minimal compute requirements. Our code is available at this https URL 

**Abstract (ZH)**: 基于文本的LoRA (T2L): 一种仅依赖目标任务自然语言描述即可实时适配大规模语言模型的方法 

---
# Simple Yet Effective: Extracting Private Data Across Clients in Federated Fine-Tuning of Large Language Models 

**Title (ZH)**: 简单而有效的 federated 大型语言模型微调中跨客户端提取私人数据方法 

**Authors**: Yingqi Hu, Zhuo Zhang, Jingyuan Zhang, Lizhen Qu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06060)  

**Abstract**: Federated fine-tuning of large language models (FedLLMs) presents a promising approach for achieving strong model performance while preserving data privacy in sensitive domains. However, the inherent memorization ability of LLMs makes them vulnerable to training data extraction attacks. To investigate this risk, we introduce simple yet effective extraction attack algorithms specifically designed for FedLLMs. In contrast to prior "verbatim" extraction attacks, which assume access to fragments from all training data, our approach operates under a more realistic threat model, where the attacker only has access to a single client's data and aims to extract previously unseen personally identifiable information (PII) from other clients. This requires leveraging contextual prefixes held by the attacker to generalize across clients. To evaluate the effectiveness of our approaches, we propose two rigorous metrics-coverage rate and efficiency-and extend a real-world legal dataset with PII annotations aligned with CPIS, GDPR, and CCPA standards, achieving 89.9% human-verified precision. Experimental results show that our method can extract up to 56.57% of victim-exclusive PII, with "Address," "Birthday," and "Name" being the most vulnerable categories. Our findings underscore the pressing need for robust defense strategies and contribute a new benchmark and evaluation framework for future research in privacy-preserving federated learning. 

**Abstract (ZH)**: 联邦微调大型语言模型（FedLLMs）为在敏感领域实现强大的模型性能同时保持数据隐私提供了一种有前景的方法。然而，大型语言模型固有的记忆能力使它们容易受到训练数据提取攻击。为了研究这一风险，我们引入了一种简单有效的提取攻击算法，专门针对FedLLMs。相比之下，先前的“逐字”提取攻击假设可以访问所有训练数据的片段，而我们的方法则在攻击者只能访问单个客户端数据并试图从其他客户端提取未见过的个人可识别信息（PII）的更现实威胁模型下运作。这需要利用攻击者持有的上下文前缀来跨客户端进行泛化。为了评估我们方法的有效性，我们提出了两个严格的度量标准——覆盖率和效率，并扩展了一个符合CPIS、GDPR和CCPA标准的现实世界法律数据集，实现89.9%的人工验证精度。实验结果表明，我们的方法可以提取高达56.57%的受害者独有的PII，“地址”、“生日”和“姓名”是最容易受损的类别。我们的研究结果突显了制定 robust 防御策略的紧迫性，并为未来隐私保护联邦学习的研究提供了新的基准和评估框架。 

---
# Microgrids Coalitions for Energy Market Balancing 

**Title (ZH)**: 微电网联盟在能源市场平衡中的应用 

**Authors**: Viorica Chifu, Cristina Bianca Pop, Tudor Cioara, Ionut Anghel  

**Link**: [PDF](https://arxiv.org/pdf/2506.06058)  

**Abstract**: With the integration of renewable sources in electricity distribution networks, the need to develop intelligent mechanisms for balancing the energy market has arisen. In the absence of such mechanisms, the energy market may face imbalances that can lead to power outages, financial losses or instability at the grid level. In this context, the grouping of microgrids into optimal coalitions that can absorb energy from the market during periods of surplus or supply energy to the market during periods of is a key aspect in the efficient management of distribution networks. In this article, we propose a method that identify an optimal microgrids coalition capable of addressing the dynamics of the energy market. The proposed method models the problem of identifying the optimal coalition as an optimization problem that it solves by combining a strategy inspired by cooperative game theory with a memetic algorithm. An individual is represented as a coalition of microgrids and the evolution of population of individuals over generations is assured by recombination and mutation. The fitness function is defined as the difference between the total value generated by the coalition and a penalty applied to the coalition when the energy traded by coalition exceeds the energy available/demanded on/by the energy market. The value generated by the coalition is calculated based on the profit obtained by the collation if it sells energy on the market during periods of deficit or the savings obtained by the coalition if it buys energy on the market during periods of surplus and the costs associated with the trading process. This value is divided equitably among the coalition members, according to the Shapley value, which considers the contribution of each one to the formation of collective value. 

**Abstract (ZH)**: 随着可再生能源在电力 distribution 网络中的整合，开发智能机制以平衡能源市场的需求变得日益重要。在缺乏此类机制的情况下，能源市场可能面临失衡，这可能导致停电、经济损失或电网级的不稳定。在此背景下，将微电网分组为能够吸收市场过剩能量或在短缺时向市场供电的最优联盟，是高效管理 distribution 网络的关键方面。本文提出了一种方法，以识别一个能够应对能源市场动态的最优微电网联盟。所提出的方法将识别最优联盟的问题建模为一个优化问题，并通过结合合作博弈论启发的策略与遗传算法来求解。个体被表示为一个微电网联盟，种群的进化通过重组和变异得以保证。适应度函数定义为联盟总价值与当联盟交易的能量超过市场可用/需求的能量时所施加的惩罚之差。联盟的价值基于在短缺期间向市场出售能量所得利润，或在过剩期间从市场购买能量所节省的成本及交易过程中的相关成本进行计算。该价值根据 Shapley 值公平地分配给联盟成员，Shapley 值考虑了每个成员对集体价值形成所作的贡献。 

---
# Hey, That's My Data! Label-Only Dataset Inference in Large Language Models 

**Title (ZH)**: 嘿，这是我的数据！基于大型语言模型的标签唯一数据集推理 

**Authors**: Chen Xiong, Zihao Wang, Rui Zhu, Tsung-Yi Ho, Pin-Yu Chen, Jingwei Xiong, Haixu Tang, Lucila Ohno-Machado  

**Link**: [PDF](https://arxiv.org/pdf/2506.06057)  

**Abstract**: Large Language Models (LLMs) have revolutionized Natural Language Processing by excelling at interpreting, reasoning about, and generating human language. However, their reliance on large-scale, often proprietary datasets poses a critical challenge: unauthorized usage of such data can lead to copyright infringement and significant financial harm. Existing dataset-inference methods typically depend on log probabilities to detect suspicious training material, yet many leading LLMs have begun withholding or obfuscating these signals. This reality underscores the pressing need for label-only approaches capable of identifying dataset membership without relying on internal model logits.
We address this gap by introducing CatShift, a label-only dataset-inference framework that capitalizes on catastrophic forgetting: the tendency of an LLM to overwrite previously learned knowledge when exposed to new data. If a suspicious dataset was previously seen by the model, fine-tuning on a portion of it triggers a pronounced post-tuning shift in the model's outputs; conversely, truly novel data elicits more modest changes. By comparing the model's output shifts for a suspicious dataset against those for a known non-member validation set, we statistically determine whether the suspicious set is likely to have been part of the model's original training corpus. Extensive experiments on both open-source and API-based LLMs validate CatShift's effectiveness in logit-inaccessible settings, offering a robust and practical solution for safeguarding proprietary data. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过在解释、推理和生成人类语言方面表现出色，彻底改变了自然语言处理。然而，它们对大规模、经常是专有数据集的依赖提出了一个关键挑战：未经授权使用这些数据可能导致版权侵权和重大经济损失。现有的数据集推断方法通常依赖于对数概率来检测可疑训练材料，而许多领先的LLM已经开始隐藏或模糊这些信号。这一现实凸显了急需一种仅标签的方法，它能够在不依赖内部模型逻辑值的情况下识别数据集成员身份的需求。我们通过引入CatShift——一种利用灾难性遗忘的数据集推断框架来填补这一空白：灾难性遗忘是指当LLM接触到新数据时，会覆盖之前学习的知识的倾向。如果可疑数据集之前被模型见过，对其进行部分微调会导致模型输出显着变化；相反，真正新颖的数据只会引起适度的变化。通过将可疑数据集的模型输出变化与已知非成员验证集的变化进行统计比较，我们能够确定可疑数据集很可能是模型原始训练语料的一部分。广泛的实验在开源和基于API的LLM上验证了CatShift在对数概率不可访问环境中的有效性，提供了一种 robust 和实用的解决方案来保护专有数据。 

---
# FPDANet: A Multi-Section Classification Model for Intelligent Screening of Fetal Ultrasound 

**Title (ZH)**: FPDANet：一种用于胎儿超声智能化筛查的多段分类模型 

**Authors**: Minglang Chen, Jie He, Caixu Xu, Bocheng Liang, Shengli Li, Guannan He, Xiongjie Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06054)  

**Abstract**: ResNet has been widely used in image classification tasks due to its ability to model the residual dependence of constant mappings for linear computation. However, the ResNet method adopts a unidirectional transfer of features and lacks an effective method to correlate contextual information, which is not effective in classifying fetal ultrasound images in the classification task, and fetal ultrasound images have problems such as low contrast, high similarity, and high noise. Therefore, we propose a bilateral multi-scale information fusion network-based FPDANet to address the above challenges. Specifically, we design the positional attention mechanism (DAN) module, which utilizes the similarity of features to establish the dependency of different spatial positional features and enhance the feature representation. In addition, we design a bilateral multi-scale (FPAN) information fusion module to capture contextual and global feature dependencies at different feature scales, thereby further improving the model representation. FPDANet classification results obtained 91.05\% and 100\% in Top-1 and Top-5 metrics, respectively, and the experimental results proved the effectiveness and robustness of FPDANet. 

**Abstract (ZH)**: 带双向多尺度信息融合网络的FPDANet在胎儿超声图像分类中的应用 

---
# TRUST: Test-time Resource Utilization for Superior Trustworthiness 

**Title (ZH)**: TRUST: 测试时资源利用以提高可信度 

**Authors**: Haripriya Harikumar, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2506.06048)  

**Abstract**: Standard uncertainty estimation techniques, such as dropout, often struggle to clearly distinguish reliable predictions from unreliable ones. We attribute this limitation to noisy classifier weights, which, while not impairing overall class-level predictions, render finer-level statistics less informative. To address this, we propose a novel test-time optimization method that accounts for the impact of such noise to produce more reliable confidence estimates. This score defines a monotonic subset-selection function, where population accuracy consistently increases as samples with lower scores are removed, and it demonstrates superior performance in standard risk-based metrics such as AUSE and AURC. Additionally, our method effectively identifies discrepancies between training and test distributions, reliably differentiates in-distribution from out-of-distribution samples, and elucidates key differences between CNN and ViT classifiers across various vision datasets. 

**Abstract (ZH)**: 标准不确定性估计技术，如丢弃法，往往难以清晰地区分可靠预测与不可靠预测。我们将其局限性归因于噪声分类器权重，虽然这些噪声并未损害整体类别的预测，但降低了细粒度统计数据的信息性。为此，我们提出了一种新的测试时优化方法，该方法考虑了此类噪声的影响，以生成更可靠的置信度估计。该评分定义了一个单调的子集选择函数，随着分数较低的样本被移除，总体准确率一直提升，并且在标准风险度量指标如AUSE和AURC中具有优越性能。此外，我们的方法能够有效识别训练分布与测试分布之间的差异，可靠地区分在分布样本与离分布样本，并阐明了在各种视觉数据集上CNN与ViT分类器的关键差异。 

---
# HAVIR: HierArchical Vision to Image Reconstruction using CLIP-Guided Versatile Diffusion 

**Title (ZH)**: HAVIR: 分层视觉引导的CLIP指导可变扩散图像重建 

**Authors**: Shiyi Zhang, Dong Liang, Hairong Zheng, Yihang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06035)  

**Abstract**: Reconstructing visual information from brain activity bridges the gap between neuroscience and computer vision. Even though progress has been made in decoding images from fMRI using generative models, a challenge remains in accurately recovering highly complex visual stimuli. This difficulty stems from their elemental density and diversity, sophisticated spatial structures, and multifaceted semantic information.
To address these challenges, we propose HAVIR that contains two adapters: (1) The AutoKL Adapter transforms fMRI voxels into a latent diffusion prior, capturing topological structures; (2) The CLIP Adapter converts the voxels to CLIP text and image embeddings, containing semantic information. These complementary representations are fused by Versatile Diffusion to generate the final reconstructed image. To extract the most essential semantic information from complex scenarios, the CLIP Adapter is trained with text captions describing the visual stimuli and their corresponding semantic images synthesized from these captions. The experimental results demonstrate that HAVIR effectively reconstructs both structural features and semantic information of visual stimuli even in complex scenarios, outperforming existing models. 

**Abstract (ZH)**: 从大脑活动重建视觉信息跨越了神经科学与计算机视觉的鸿沟。尽管已经使用生成模型从fMRI解码图像取得了进展，但在准确恢复高度复杂的视觉刺激方面仍面临挑战。这种困难源于其基本密度和多样性、复杂的空间结构以及多方面的语义信息。

为应对这些挑战，我们提出了HAVIR，其包含两个适配器：(1) AutoKL适配器将fMRI体素转换为潜在扩散先验，捕捉拓扑结构；(2) CLIP适配器将体素转换为CLIP文本和图像嵌入，包含语义信息。这些互补的表示由通用扩散融合生成最终重建图像。为了从复杂场景中提取最核心的语义信息，CLIP适配器基于描述视觉刺激及其对应语义图像的文本说明进行训练。实验结果表明，HAVIR在复杂场景中有效地重建了视觉刺激的结构特征和语义信息，优于现有模型。 

---
# End-to-End Framework for Robot Lawnmower Coverage Path Planning using Cellular Decomposition 

**Title (ZH)**: 基于细胞分解的自主草坪修剪机器人全覆盖路径规划端到端框架 

**Authors**: Nikunj Shah, Utsav Dey, Kenji Nishimiya  

**Link**: [PDF](https://arxiv.org/pdf/2506.06028)  

**Abstract**: Efficient Coverage Path Planning (CPP) is necessary for autonomous robotic lawnmowers to effectively navigate and maintain lawns with diverse and irregular shapes. This paper introduces a comprehensive end-to-end pipeline for CPP, designed to convert user-defined boundaries on an aerial map into optimized coverage paths seamlessly. The pipeline includes user input extraction, coordinate transformation, area decomposition and path generation using our novel AdaptiveDecompositionCPP algorithm, preview and customization through an interactive coverage path visualizer, and conversion to actionable GPS waypoints. The AdaptiveDecompositionCPP algorithm combines cellular decomposition with an adaptive merging strategy to reduce non-mowing travel thereby enhancing operational efficiency. Experimental evaluations, encompassing both simulations and real-world lawnmower tests, demonstrate the effectiveness of the framework in coverage completeness and mowing efficiency. 

**Abstract (ZH)**: 高效覆盖路径规划（CPP）对于自主 robotic 前后驱动割草机有效导航和维护各种不规则形状的草坪是必要的。本文介绍了一套完整的端到端 CPP 管道，旨在将用户在空中地图上定义的边界无缝转换为优化的覆盖路径。该管道包含用户输入提取、坐标转换、区域分解和路径生成（使用我们新颖的 AdaptiveDecompositionCPP 算法）、通过交互式的覆盖路径可视化器进行预览和自定义，以及转换为可操作的GPS航点。AdaptiveDecompositionCPP 算法结合了细胞分解与自适应合并策略，以减少非割草行进距离从而提高操作效率。实验评估，包括模拟和实际草坪割草机测试，展示了该框架在覆盖完整性和割草效率方面的有效性。 

---
# When to Trust Context: Self-Reflective Debates for Context Reliability 

**Title (ZH)**: 何时信任背景：关于背景可靠性的自我反思辩论 

**Authors**: Zeqi Zhou, Fang Wu, Shayan Talaei, Haokai Zhao, Cheng Meixin, Tinson Xu, Amin Saberi, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06020)  

**Abstract**: Large language models frequently encounter conflicts between their parametric knowledge and contextual input, often resulting in factual inconsistencies or hallucinations. We propose Self-Reflective Debate for Contextual Reliability (SR-DCR), a lightweight framework that integrates token-level self-confidence with an asymmetric multi-agent debate to adjudicate such conflicts. A critic, deprived of context, challenges a defender who argues from the given passage; a judge model evaluates the debate and determines the context's reliability. The final answer is selected by combining the verdict with model confidence. Experiments on the ClashEval benchmark demonstrate that SR-DCR consistently enhances robustness to misleading context while maintaining accuracy on trustworthy inputs, outperforming both classical debate and confidence-only baselines with minimal computational overhead. The code is available at this https URL. 

**Abstract (ZH)**: 自省辩论以提升上下文可靠性（SR-DCR）：一种轻量级框架 

---
# Optimization-Free Universal Watermark Forgery with Regenerative Diffusion Models 

**Title (ZH)**: 基于再生扩散模型的无优化通用水印伪造 

**Authors**: Chaoyi Zhu, Zaitang Li, Renyi Yang, Robert Birke, Pin-Yu Chen, Tsung-Yi Ho, Lydia Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06018)  

**Abstract**: Watermarking becomes one of the pivotal solutions to trace and verify the origin of synthetic images generated by artificial intelligence models, but it is not free of risks. Recent studies demonstrate the capability to forge watermarks from a target image onto cover images via adversarial optimization without knowledge of the target generative model and watermark schemes. In this paper, we uncover a greater risk of an optimization-free and universal watermark forgery that harnesses existing regenerative diffusion models. Our proposed forgery attack, PnP (Plug-and-Plant), seamlessly extracts and integrates the target watermark via regenerating the image, without needing any additional optimization routine. It allows for universal watermark forgery that works independently of the target image's origin or the watermarking model used. We explore the watermarked latent extracted from the target image and visual-textual context of cover images as priors to guide sampling of the regenerative process. Extensive evaluation on 24 scenarios of model-data-watermark combinations demonstrates that PnP can successfully forge the watermark (up to 100% detectability and user attribution), and maintain the best visual perception. By bypassing model retraining and enabling adaptability to any image, our approach significantly broadens the scope of forgery attacks, presenting a greater challenge to the security of current watermarking techniques for diffusion models and the authority of watermarking schemes in synthetic data generation and governance. 

**Abstract (ZH)**: Optimization-Free and Universal Watermark Forgery Using Regenerative Diffusion Models: PnP (Plug-and-Plant) Attack 

---
# Unlocking Recursive Thinking of LLMs: Alignment via Refinement 

**Title (ZH)**: 解锁LLMs的递归思考：通过精炼实现对齐 

**Authors**: Haoke Zhang, Xiaobo Liang, Cunxiang Wang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06009)  

**Abstract**: The OpenAI o1-series models have demonstrated that leveraging long-form Chain of Thought (CoT) can substantially enhance performance. However, the recursive thinking capabilities of Large Language Models (LLMs) remain limited, particularly in the absence of expert-curated data for distillation. In this paper, we propose \textbf{AvR}: \textbf{Alignment via Refinement}, a novel method aimed at unlocking the potential of LLMs for recursive reasoning through long-form CoT. AvR introduces a refinement process that integrates criticism and improvement actions, guided by differentiable learning techniques to optimize \textbf{refinement-aware rewards}. As a result, the synthesized multi-round data can be organized as a long refinement thought, further enabling test-time scaling. Experimental results show that AvR significantly outperforms conventional preference optimization methods. Notably, with only 3k synthetic samples, our method boosts the performance of the LLaMA-3-8B-Instruct model by over 20\% in win rate on AlpacaEval 2.0. Our code is available at Github (this https URL). 

**Abstract (ZH)**: AvR：通过长形式链式思考优化的递归对齐方法 

---
# Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models 

**Title (ZH)**: .token Signature: 通过Token解码特征预测大型语言模型中的chain-of-thought收益 

**Authors**: Peijie Liu, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06008)  

**Abstract**: Chain-of-Thought (CoT) technique has proven effective in improving the performance of large language models (LLMs) on complex reasoning tasks. However, the performance gains are inconsistent across different tasks, and the underlying mechanism remains a long-standing research question. In this work, we make a preliminary observation that the monotonicity of token probability distributions may be correlated with the gains achieved through CoT reasoning. Leveraging this insight, we propose two indicators based on the token probability distribution to assess CoT effectiveness across different tasks. By combining instance-level indicators with logistic regression model, we introduce Dynamic CoT, a method that dynamically select between CoT and direct answer. Furthermore, we extend Dynamic CoT to closed-source models by transferring decision strategies learned from open-source models. Our indicators for assessing CoT effectiveness achieve an accuracy of 89.2\%, and Dynamic CoT reduces token consumption by more than 35\% while maintaining high accuracy. Overall, our work offers a novel perspective on the underlying mechanisms of CoT reasoning and provides a framework for its more efficient deployment. 

**Abstract (ZH)**: Chain-of-Thought技术在提升大型语言模型复杂推理任务性能方面的初步观察及其动态应用研究 

---
# Enhancing Orthopox Image Classification Using Hybrid Machine Learning and Deep Learning Models 

**Title (ZH)**: 使用混合机器学习与深度学习模型增强正痘病毒图像分类 

**Authors**: Alejandro Puente-Castro, Enrique Fernandez-Blanco, Daniel Rivero, Andres Molares-Ulloa  

**Link**: [PDF](https://arxiv.org/pdf/2506.06007)  

**Abstract**: Orthopoxvirus infections must be accurately classified from medical pictures for an easy and early diagnosis and epidemic prevention. The necessity for automated and scalable solutions is highlighted by the fact that traditional diagnostic techniques can be time-consuming and require expert interpretation and there are few and biased data sets of the different types of Orthopox. In order to improve classification performance and lower computational costs, a hybrid strategy is put forth in this paper that uses Machine Learning models combined with pretrained Deep Learning models to extract deep feature representations without the need for augmented data. The findings show that this feature extraction method, when paired with other methods in the state-of-the-art, produces excellent classification outcomes while preserving training and inference efficiency. The proposed approach demonstrates strong generalization and robustness across multiple evaluation settings, offering a scalable and interpretable solution for real-world clinical deployment. 

**Abstract (ZH)**: 正痘病毒感染需要通过医学影像准确分类以便实现早期诊断和疫情预防。本研究强调了自动化和可扩展解决方案的必要性，因为传统诊断技术耗时且需要专家解释，而正痘病毒的不同类型数据集较少且带有偏见。为了提高分类性能并降低计算成本，本文提出了一种结合机器学习模型和预训练深度学习模型的混合策略，以提取深层特征表示，无需增加数据。研究结果表明，这种方法与其他先进的方法结合使用时，能够产生出色的分类效果，同时保持训练和推理效率。所提出的建模方法在多种评估设置中展现出良好的泛化能力和鲁棒性，提供了一种可扩展且可解释的临床部署解决方案。 

---
# Bootstrapping World Models from Dynamics Models in Multimodal Foundation Models 

**Title (ZH)**: 从多模态基础模型的动态模型中bootstrapping世界模型 

**Authors**: Yifu Qiu, Yftah Ziser, Anna Korhonen, Shay B. Cohen, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2506.06006)  

**Abstract**: To what extent do vision-and-language foundation models possess a realistic world model (observation $\times$ action $\rightarrow$ observation) and a dynamics model (observation $\times$ observation $\rightarrow$ action), when actions are expressed through language? While open-source foundation models struggle with both, we find that fine-tuning them to acquire a dynamics model through supervision is significantly easier than acquiring a world model. In turn, dynamics models can be used to bootstrap world models through two main strategies: 1) weakly supervised learning from synthetic data and 2) inference time verification. Firstly, the dynamics model can annotate actions for unlabelled pairs of video frame observations to expand the training data. We further propose a new objective, where image tokens in observation pairs are weighted by their importance, as predicted by a recognition model. Secondly, the dynamics models can assign rewards to multiple samples of the world model to score them, effectively guiding search at inference time. We evaluate the world models resulting from both strategies through the task of action-centric image editing on Aurora-Bench. Our best model achieves a performance competitive with state-of-the-art image editing models, improving on them by a margin of $15\%$ on real-world subsets according to GPT4o-as-judge, and achieving the best average human evaluation across all subsets of Aurora-Bench. 

**Abstract (ZH)**: 视觉和语言基础模型在多模态观察（观察×动作→观察）和动力学模型（观察×观察→动作）方面具备多现实世界的程度，尤其是在动作通过语言表达时？尽管开源基础模型在两者方面都存在挑战，我们发现通过监督细调它们以获得动力学模型比获得世界模型要容易得多。反过来，动力学模型可以通过两种主要策略来增强世界模型：1) 从合成数据中进行弱监督学习；2) 推断时间验证。首先，动力学模型可以为未标记的视频帧观察对添加动作标签，以扩展训练数据。我们进一步提出一个新的目标，其中在观察对中的图像标记根据识别模型的预测重要性加权。其次，动力学模型可以为世界模型的多个样本分配奖励，以评估它们，在推理时间有效引导搜索。我们通过奥罗拉平台上的以动作为中心的图像编辑任务评估通过这两种策略生成的世界模型。我们的最佳模型在GPT4o-as-judge评测上与最先进的图像编辑模型具有竞争力，在现实世界的子集上提高了15%的性能，并在奥罗拉平台的所有子集上实现了最高的平均人类评估得分。 

---
# Leveraging Generative AI for Enhancing Automated Assessment in Programming Education Contests 

**Title (ZH)**: 利用生成式AI提升编程教育竞赛中的自动化评估 

**Authors**: Stefan Dascalescu, Adrian Marius Dumitran, Mihai Alexandru Vasiluta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05990)  

**Abstract**: Competitive programming contests play a crucial role in cultivating computational thinking and algorithmic skills among learners. However, generating comprehensive test cases to effectively assess programming solutions remains resource-intensive and challenging for educators. This paper introduces an innovative NLP-driven method leveraging generative AI (large language models) to automate the creation of high-quality test cases for competitive programming assessments. We extensively evaluated our approach on diverse datasets, including 25 years of Romanian Informatics Olympiad (OJI) data for 5th graders, recent competitions hosted on the this http URL platform, and the International Informatics Olympiad in Teams (IIOT). Our results demonstrate that AI-generated test cases substantially enhanced assessments, notably identifying previously undetected errors in 67% of the OJI 5th grade programming problems. These improvements underscore the complementary educational value of our technique in formative assessment contexts. By openly sharing our prompts, translated datasets, and methodologies, we offer practical NLP-based tools that educators and contest organizers can readily integrate to enhance assessment quality, reduce workload, and deepen insights into learner performance. 

**Abstract (ZH)**: 基于NLP的生成式AI方法在编程竞赛评估中自动创建高质量测试案例的研究 

---
# Audio-Aware Large Language Models as Judges for Speaking Styles 

**Title (ZH)**: 具有音频意识的大语言模型作为演讲风格的评判者 

**Authors**: Cheng-Han Chiang, Xiaofei Wang, Chung-Ching Lin, Kevin Lin, Linjie Li, Radu Kopetz, Yao Qian, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05984)  

**Abstract**: Audio-aware large language models (ALLMs) can understand the textual and non-textual information in the audio input. In this paper, we explore using ALLMs as an automatic judge to assess the speaking styles of speeches. We use ALLM judges to evaluate the speeches generated by SLMs on two tasks: voice style instruction following and role-playing. The speaking style we consider includes emotion, volume, speaking pace, word emphasis, pitch control, and non-verbal elements. We use four spoken language models (SLMs) to complete the two tasks and use humans and ALLMs to judge the SLMs' responses. We compare two ALLM judges, GPT-4o-audio and Gemini-2.5-pro, with human evaluation results and show that the agreement between Gemini and human judges is comparable to the agreement between human evaluators. These promising results show that ALLMs can be used as a judge to evaluate SLMs. Our results also reveal that current SLMs, even GPT-4o-audio, still have room for improvement in controlling the speaking style and generating natural dialogues. 

**Abstract (ZH)**: Audio-aware大型语言模型（ALLMs）能够理解音频输入中的文本和非文本信息。本文探索使用ALLMs作为自动评委评估演讲的演讲风格。我们使用ALLM评委评估由四个口语模型（SLMs）完成的两种任务（语音风格指令跟随和角色扮演）所产生的演讲，使用人类评委和ALLM评委对SLMs的响应进行评估。我们比较了两种ALLM评委，GPT-4o-audio和Gemini-2.5-pro，与人类评估结果，并展示了Gemini和人类评委之间的共识与人类评估者之间的共识相当。这些有希望的结果表明，ALLMs可以作为评委来评估SLMs。我们的结果还表明，当前的SLMs，即使包括GPT-4o-audio，仍然在控制演讲风格和生成自然对话方面存在改进空间。 

---
# AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification 

**Title (ZH)**: AMPED：自适应多目标投影以平衡探索与 skill 多样化 

**Authors**: Geonwoo Cho, Jaemoon Lee, Jaegyun Im, Subi Lee, Jihwan Lee, Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05980)  

**Abstract**: Skill-based reinforcement learning (SBRL) enables rapid adaptation in environments with sparse rewards by pretraining a skill-conditioned policy. Effective skill learning requires jointly maximizing both exploration and skill diversity. However, existing methods often face challenges in simultaneously optimizing for these two conflicting objectives. In this work, we propose a new method, Adaptive Multi-objective Projection for balancing Exploration and skill Diversification (AMPED), which explicitly addresses both exploration and skill diversification. We begin by conducting extensive ablation studies to identify and define a set of objectives that effectively capture the aspects of exploration and skill diversity, respectively. During the skill pretraining phase, AMPED introduces a gradient surgery technique to balance the objectives of exploration and skill diversity, mitigating conflicts and reducing reliance on heuristic tuning. In the subsequent fine-tuning phase, AMPED incorporates a skill selector module that dynamically selects suitable skills for downstream tasks, based on task-specific performance signals. Our approach achieves performance that surpasses SBRL baselines across various benchmarks. These results highlight the importance of explicitly harmonizing exploration and diversity and demonstrate the effectiveness of AMPED in enabling robust and generalizable skill learning. Project Page: this https URL 

**Abstract (ZH)**: 基于技能的强化学习（SBRL）通过预先训练一个技能条件策略，能够在稀疏奖励环境中实现快速适应。有效的技能学习需要同时最大化探索和技能多样性。然而，现有方法往往难以同时优化这两个相互排斥的目标。在本文中，我们提出了一种新方法，适应性多目标投影以平衡探索和技能多样化（AMPED），该方法明确地同时处理探索和技能多样性的目标。我们通过广泛的消融研究来识别和定义一套能够有效捕捉探索和技能多样性方面目标。在技能预训练阶段，AMPED引入了一种梯度手术技术来平衡探索和技能多样性的目标，缓解冲突并减少对启发式调谐的依赖。在后续的微调阶段，AMPED引入了一个技能选择模块，根据特定任务的性能信号动态选择适合的技能用于下游任务。我们的方法在各种基准测试中优于SBRL基线。这些结果强调了明确协调探索和多样性的关键性，并展示了AMPED在促进鲁棒性和通用化的技能学习方面的有效性。项目页面：this https URL。 

---
# On Measuring Long-Range Interactions in Graph Neural Networks 

**Title (ZH)**: 测量图神经网络中的长范围交互 

**Authors**: Jacob Bamberger, Benjamin Gutteridge, Scott le Roux, Michael M. Bronstein, Xiaowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05971)  

**Abstract**: Long-range graph tasks -- those dependent on interactions between distant nodes -- are an open problem in graph neural network research. Real-world benchmark tasks, especially the Long Range Graph Benchmark, have become popular for validating the long-range capability of proposed architectures. However, this is an empirical approach that lacks both robustness and theoretical underpinning; a more principled characterization of the long-range problem is required. To bridge this gap, we formalize long-range interactions in graph tasks, introduce a range measure for operators on graphs, and validate it with synthetic experiments. We then leverage our measure to examine commonly used tasks and architectures, and discuss to what extent they are, in fact, long-range. We believe our work advances efforts to define and address the long-range problem on graphs, and that our range measure will aid evaluation of new datasets and architectures. 

**Abstract (ZH)**: 长范围图任务：图神经网络研究中的开放问题与普适刻画 

---
# Let's Put Ourselves in Sally's Shoes: Shoes-of-Others Prefixing Improves Theory of Mind in Large Language Models 

**Title (ZH)**: 让我们换位思考萨利的处境：他人鞋类前缀提高大规模语言模型的理论思维能力 

**Authors**: Kazutoshi Shinoda, Nobukatsu Hojo, Kyosuke Nishida, Yoshihiro Yamazaki, Keita Suzuki, Hiroaki Sugiyama, Kuniko Saito  

**Link**: [PDF](https://arxiv.org/pdf/2506.05970)  

**Abstract**: Recent studies have shown that Theory of Mind (ToM) in large language models (LLMs) has not reached human-level performance yet. Since fine-tuning LLMs on ToM datasets often degrades their generalization, several inference-time methods have been proposed to enhance ToM in LLMs. However, existing inference-time methods for ToM are specialized for inferring beliefs from contexts involving changes in the world state. In this study, we present a new inference-time method for ToM, Shoes-of-Others (SoO) prefixing, which makes fewer assumptions about contexts and is applicable to broader scenarios. SoO prefixing simply specifies the beginning of LLM outputs with ``Let's put ourselves in A's shoes.'', where A denotes the target character's name. We evaluate SoO prefixing on two benchmarks that assess ToM in conversational and narrative contexts without changes in the world state and find that it consistently improves ToM across five categories of mental states. Our analysis suggests that SoO prefixing elicits faithful thoughts, thereby improving the ToM performance. 

**Abstract (ZH)**: Recent studies have shown that大语言模型（LLM）中的心智理论（ToM）尚未达到人类水平的表现。由于在ToM数据集上微调LLM往往会损害其泛化能力，因此已经提出了几种推理时的方法来增强LLM中的ToM。然而，现有的ToM推理时方法专门用于推断涉及世界状态变化的情景下的信念。在本研究中，我们提出了一种新的ToM推理时方法——他人的鞋子前缀（Shoes-of-Others, SoO prefixing），该方法对背景的假设较少，并适用于更广泛的情景。SoO前缀简单地规定LLM输出的开始部分为“让我们站在A的立场上看问题。”，其中A表示目标人物的名称。我们在两个评估对话和叙述背景下ToM而无需世界状态变化的基准上评估了SoO前缀，发现它在五类心理状态中都能一致地提升ToM性能。我们的分析表明，SoO前缀激发了忠实的想法，从而提高了ToM性能。 

---
# Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning 

**Title (ZH)**: 在线强化学习中贝尔曼最优性算子向贝尔曼算子的渐进过渡 

**Authors**: Motoki Omura, Kazuki Ota, Takayuki Osa, Yusuke Mukuta, Tatsuya Harada  

**Link**: [PDF](https://arxiv.org/pdf/2506.05968)  

**Abstract**: For continuous action spaces, actor-critic methods are widely used in online reinforcement learning (RL). However, unlike RL algorithms for discrete actions, which generally model the optimal value function using the Bellman optimality operator, RL algorithms for continuous actions typically model Q-values for the current policy using the Bellman operator. These algorithms for continuous actions rely exclusively on policy updates for improvement, which often results in low sample efficiency. This study examines the effectiveness of incorporating the Bellman optimality operator into actor-critic frameworks. Experiments in a simple environment show that modeling optimal values accelerates learning but leads to overestimation bias. To address this, we propose an annealing approach that gradually transitions from the Bellman optimality operator to the Bellman operator, thereby accelerating learning while mitigating bias. Our method, combined with TD3 and SAC, significantly outperforms existing approaches across various locomotion and manipulation tasks, demonstrating improved performance and robustness to hyperparameters related to optimality. 

**Abstract (ZH)**: 将贝尔曼最优算子引入actor-critic框架的有效性研究：加速学习并减轻偏差 

---
# MOGO: Residual Quantized Hierarchical Causal Transformer for High-Quality and Real-Time 3D Human Motion Generation 

**Title (ZH)**: MOGO：残差量化分层因果变换器，用于高保真实时三维人体运动生成 

**Authors**: Dongjie Fu, Tengjiao Sun, Pengcheng Fang, Xiaohao Cai, Hansung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05952)  

**Abstract**: Recent advances in transformer-based text-to-motion generation have led to impressive progress in synthesizing high-quality human motion. Nevertheless, jointly achieving high fidelity, streaming capability, real-time responsiveness, and scalability remains a fundamental challenge. In this paper, we propose MOGO (Motion Generation with One-pass), a novel autoregressive framework tailored for efficient and real-time 3D motion generation. MOGO comprises two key components: (1) MoSA-VQ, a motion scale-adaptive residual vector quantization module that hierarchically discretizes motion sequences with learnable scaling to produce compact yet expressive representations; and (2) RQHC-Transformer, a residual quantized hierarchical causal transformer that generates multi-layer motion tokens in a single forward pass, significantly reducing inference latency. To enhance semantic fidelity, we further introduce a text condition alignment mechanism that improves motion decoding under textual control. Extensive experiments on benchmark datasets including HumanML3D, KIT-ML, and CMP demonstrate that MOGO achieves competitive or superior generation quality compared to state-of-the-art transformer-based methods, while offering substantial improvements in real-time performance, streaming generation, and generalization under zero-shot settings. 

**Abstract (ZH)**: 近期基于变压器的文字到运动生成技术取得了显著进展，极大地促进了高质量人类运动合成。然而，同时实现高保真度、流式传输能力、实时响应性和可扩展性仍是一项基本挑战。本文提出了一种新型自回归框架MOGO（运动生成一站式），旨在高效地进行实时三维运动生成。MOGO包含两个关键组成部分：（1）MoSA-VQ，一种运动尺度自适应残差矢量量化模块，该模块通过可学习的缩放对运动序列进行分层离散化，生成紧凑而富有表现力的表示；（2）RQHC-Transformer，一种残差量化层次因原因子变压器，在单向前传播中生成多层运动标记，显著减少了推理延迟。为进一步提升语义保真度，我们引入了一种文本条件对齐机制，该机制在文本控制下改善了运动解码。在包括HumanML3D、KIT-ML和CMP基准数据集上的广泛实验表明，MOGO在实时性能、流式生成和零样本设置下的泛化方面取得了显著改进，同时生成质量与最先进的基于变压器的方法相当或更优。 

---
# IntentionESC: An Intention-Centered Framework for Enhancing Emotional Support in Dialogue Systems 

**Title (ZH)**: 意图导向的情感支持对话系统框架：一种基于意图的框架 

**Authors**: Xinjie Zhang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.05947)  

**Abstract**: In emotional support conversations, unclear intentions can lead supporters to employ inappropriate strategies, inadvertently imposing their expectations or solutions on the seeker. Clearly defined intentions are essential for guiding both the supporter's motivations and the overall emotional support process. In this paper, we propose the Intention-centered Emotional Support Conversation (IntentionESC) framework, which defines the possible intentions of supporters in emotional support conversations, identifies key emotional state aspects for inferring these intentions, and maps them to appropriate support strategies. While Large Language Models (LLMs) excel in text generating, they fundamentally operate as probabilistic models trained on extensive datasets, lacking a true understanding of human thought processes and intentions. To address this limitation, we introduce the Intention Centric Chain-of-Thought (ICECoT) mechanism. ICECoT enables LLMs to mimic human reasoning by analyzing emotional states, inferring intentions, and selecting suitable support strategies, thereby generating more effective emotional support responses. To train the model with ICECoT and integrate expert knowledge, we design an automated annotation pipeline that produces high-quality training data. Furthermore, we develop a comprehensive evaluation scheme to assess emotional support efficacy and conduct extensive experiments to validate our framework. Our data and code are available at this https URL. 

**Abstract (ZH)**: 基于意图的情感支持对话框架（IntentionESC）及其生成机制（ICECoT）的研究 

---
# Comparative Analysis of Modern Machine Learning Models for Retail Sales Forecasting 

**Title (ZH)**: 现代机器学习模型在零售销售预测中的 comparative analysis 

**Authors**: Luka Hobor, Mario Brcic, Lidija Polutnik, Ante Kapetanovic  

**Link**: [PDF](https://arxiv.org/pdf/2506.05941)  

**Abstract**: Accurate forecasting is key for all business planning. When estimated sales are too high, brick-and-mortar retailers may incur higher costs due to unsold inventories, higher labor and storage space costs, etc. On the other hand, when forecasts underestimate the level of sales, firms experience lost sales, shortages, and impact on the reputation of the retailer in their relevant market. Accurate forecasting presents a competitive advantage for companies. It facilitates the achievement of revenue and profit goals and execution of pricing strategy and tactics. In this study, we provide an exhaustive assessment of the forecasting models applied to a high-resolution brick-and-mortar retail dataset. Our forecasting framework addresses the problems found in retail environments, including intermittent demand, missing values, and frequent product turnover. We compare tree-based ensembles (such as XGBoost and LightGBM) and state-of-the-art neural network architectures (including N-BEATS, NHITS, and the Temporal Fusion Transformer) across various experimental settings. Our results show that localized modeling strategies especially those using tree-based models on individual groups with non-imputed data, consistently deliver superior forecasting accuracy and computational efficiency. In contrast, neural models benefit from advanced imputation methods, yet still fall short in handling the irregularities typical of physical retail data. These results further practical understanding for model selection in retail environment and highlight the significance of data preprocessing to improve forecast performance. 

**Abstract (ZH)**: 准确的预测对于所有商业规划至关重要。当估计销售额过高时，实体零售商会因未售出的库存、更高的劳动和储存空间成本等问题产生更高的成本。另一方面，当预测低估了销售水平时，企业会面临销售额损失、短缺以及对零售商声誉的负面影响。准确的预测为公司提供了竞争优势。它有助于实现收入和利润目标，并执行定价策略。在本研究中，我们对应用于高分辨率实体零售数据集的预测模型进行了全面评估。我们的预测框架解决了零售环境中遇到的问题，包括间歇性需求、缺失值和频繁的产品更换。我们将基于树的集成模型（如XGBoost和LightGBM）与先进神经网络架构（包括N-BEATS、NHITS和时空融合转换器）进行了比较，涵盖了各种实验设置。我们的结果显示，本地化建模策略，特别是使用非插补数据的基于树的模型在各个组别中，始终能提供更高的预测准确性和计算效率。相比之下，神经网络模型虽然受益于高级插补方法，但在处理实体零售数据中的异常波动方面仍存在不足。这些结果进一步增强了在零售环境中选择模型的实用理解，并突显了数据预处理对于提高预测性能的重要性。 

---
# Quantifying Adversarial Uncertainty in Evidential Deep Learning using Conflict Resolution 

**Title (ZH)**: 使用冲突解析度量证据深度学习中的对抗不确定性 

**Authors**: Charmaine Barker, Daniel Bethell, Simos Gerasimou  

**Link**: [PDF](https://arxiv.org/pdf/2506.05937)  

**Abstract**: Reliability of deep learning models is critical for deployment in high-stakes applications, where out-of-distribution or adversarial inputs may lead to detrimental outcomes. Evidential Deep Learning, an efficient paradigm for uncertainty quantification, models predictions as Dirichlet distributions of a single forward pass. However, EDL is particularly vulnerable to adversarially perturbed inputs, making overconfident errors. Conflict-aware Evidential Deep Learning (C-EDL) is a lightweight post-hoc uncertainty quantification approach that mitigates these issues, enhancing adversarial and OOD robustness without retraining. C-EDL generates diverse, task-preserving transformations per input and quantifies representational disagreement to calibrate uncertainty estimates when needed. C-EDL's conflict-aware prediction adjustment improves detection of OOD and adversarial inputs, maintaining high in-distribution accuracy and low computational overhead. Our experimental evaluation shows that C-EDL significantly outperforms state-of-the-art EDL variants and competitive baselines, achieving substantial reductions in coverage for OOD data (up to 55%) and adversarial data (up to 90%), across a range of datasets, attack types, and uncertainty metrics. 

**Abstract (ZH)**: 基于冲突感知的证据深度学习在高危应用中的可靠性对于模型部署至关重要，它能够缓解对抗性和离分布输入导致的问题，提升模型的鲁棒性而不需重新训练。C-EDL通过为每个输入生成多样性的任务保留变换，并在需要时通过量化表征分歧来校准不确定性估计，从而提高检测离分布和对抗输入的能力，同时保持较高的内分布准确性和低的计算开销。实验评估表明，C-EDL明显优于最先进的EDL变体和竞争性基线，在多种数据集、攻击类型和不确定性度量下，离分布数据和对抗数据的覆盖范围分别减少了最高达55%和90%。 

---
# DynamicMind: A Tri-Mode Thinking System for Large Language Models 

**Title (ZH)**: DynamicMind: 大型语言模型的三模式思考系统 

**Authors**: Wei Li, Yanbin Wei, Qiushi Huang, Jiangyue Yan, Yang Chen, James T. Kwok, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05936)  

**Abstract**: Modern large language models (LLMs) often struggle to dynamically adapt their reasoning depth to varying task complexities, leading to suboptimal performance or inefficient resource utilization. To address this, we introduce DynamicMind, a novel tri-mode thinking system. DynamicMind empowers LLMs to autonomously select between Fast, Normal, and Slow thinking modes for zero-shot question answering (ZSQA) tasks through cognitive-inspired prompt engineering. Our framework's core innovations include: (1) expanding the established dual-process framework of fast and slow thinking into a tri-mode thinking system involving a normal thinking mode to preserve the intrinsic capabilities of LLM; (2) proposing the Thinking Density metric, which aligns computational resource allocation with problem complexity; and (3) developing the Thinking Mode Capacity (TMC) dataset and a lightweight Mind Router to predict the optimal thinking mode. Extensive experiments across diverse mathematical, commonsense, and scientific QA benchmarks demonstrate that DynamicMind achieves superior ZSQA capabilities while establishing an effective trade-off between performance and computational efficiency. 

**Abstract (ZH)**: 现代大规模语言模型（LLMs）往往难以动态适应不同任务复杂性的推理深度，导致性能不佳或资源利用效率低下。为解决这一问题，我们引入了DynamicMind，一个新颖的三模思考系统。DynamicMind通过认知启发式的提示工程，使LLMs自主选择快速、正常和慢速思考模式，以应对零样本问答（ZSQA）任务。该框架的核心创新包括：（1）将快速和慢速思考的双过程框架扩展为包含正常思考模式的三模思考系统，以保留LLM的核心能力；（2）提出了思考密度度量，该度量将计算资源分配与问题复杂性对齐；（3）开发了思考模式容量（TMC）数据集和轻量级Mind Router，以预测最优思考模式。广泛的实验跨不同领域的数学、常识和科学问答基准表明，DynamicMind在提高零样本问答能力的同时，有效地在性能和计算效率之间找到了平衡。 

---
# FADE: Frequency-Aware Diffusion Model Factorization for Video Editing 

**Title (ZH)**: 频率 Awareness 下的扩散模型因子分解方法及其在视频编辑中的应用 

**Authors**: Yixuan Zhu, Haolin Wang, Shilin Ma, Wenliang Zhao, Yansong Tang, Lei Chen, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.05934)  

**Abstract**: Recent advancements in diffusion frameworks have significantly enhanced video editing, achieving high fidelity and strong alignment with textual prompts. However, conventional approaches using image diffusion models fall short in handling video dynamics, particularly for challenging temporal edits like motion adjustments. While current video diffusion models produce high-quality results, adapting them for efficient editing remains difficult due to the heavy computational demands that prevent the direct application of previous image editing techniques. To overcome these limitations, we introduce FADE, a training-free yet highly effective video editing approach that fully leverages the inherent priors from pre-trained video diffusion models via frequency-aware factorization. Rather than simply using these models, we first analyze the attention patterns within the video model to reveal how video priors are distributed across different components. Building on these insights, we propose a factorization strategy to optimize each component's specialized role. Furthermore, we devise spectrum-guided modulation to refine the sampling trajectory with frequency domain cues, preventing information leakage and supporting efficient, versatile edits while preserving the basic spatial and temporal structure. Extensive experiments on real-world videos demonstrate that our method consistently delivers high-quality, realistic and temporally coherent editing results both qualitatively and quantitatively. Code is available at this https URL . 

**Abstract (ZH)**: 近期在扩散框架方面的进展显著提升了视频编辑效果，实现了高度逼真和与文本提示的强烈对齐。然而，传统的使用图像扩散模型的方法在处理视频动态方面仍然不足，特别是在处理如运动调整等具有挑战性的时间编辑时。尽管当前的视频扩散模型能够生成高质量的结果，但由于繁重的计算需求使得难以直接应用先前的图像编辑技术，进而将其用于高效的编辑操作。为了解决这些局限性，我们提出了FADE，这是一种无需训练即可高效实现视频编辑的方法，通过频率感知的因子分解充分利用预训练的视频扩散模型固有的先验知识。我们不仅使用这些模型，而是首先分析视频模型中的注意力模式，揭示视频先验在不同组件中的分布情况。基于这些洞见，我们提出了一种因子分解策略来优化每个组件的专业角色。此外，我们设计了频谱导向的调制，通过频域线索改进采样轨迹，防止信息泄露，同时支持高效和多样的编辑操作，并保持基本的空间和时间结构。在实际视频上的广泛实验表明，我们的方法在定性和定量上都能够持续产出高质量、逼真且具有时序一致性的编辑结果。代码已发布于此 https URL 。 

---
# MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models 

**Title (ZH)**: MoA：参数高效调整大型语言模型的异质适配器混合 

**Authors**: Jie Cao, Tianwei Lin, Hongyang He, Rolan Yan, Wenqiao Zhang, Juncheng Li, Dongping Zhang, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05928)  

**Abstract**: Recent studies integrate Low-Rank Adaptation (LoRA) and Mixture-of-Experts (MoE) to further enhance the performance of parameter-efficient fine-tuning (PEFT) methods in Large Language Model (LLM) applications. Existing methods employ \emph{homogeneous} MoE-LoRA architectures composed of LoRA experts with either similar or identical structures and capacities. However, these approaches often suffer from representation collapse and expert load imbalance, which negatively impact the potential of LLMs. To address these challenges, we propose a \emph{heterogeneous} \textbf{Mixture-of-Adapters (MoA)} approach. This method dynamically integrates PEFT adapter experts with diverse structures, leveraging their complementary representational capabilities to foster expert specialization, thereby enhancing the effective transfer of pre-trained knowledge to downstream tasks. MoA supports two variants: \textbf{(i)} \textit{Soft MoA} achieves fine-grained integration by performing a weighted fusion of all expert outputs; \textbf{(ii)} \textit{Sparse MoA} activates adapter experts sparsely based on their contribution, achieving this with negligible performance degradation. Experimental results demonstrate that heterogeneous MoA outperforms homogeneous MoE-LoRA methods in both performance and parameter efficiency. Our project is available at this https URL. 

**Abstract (ZH)**: 近期研究表明，将低秩适应（LoRA）与专家混排（MoE） integrates 与进一步增强大规模语言模型（LLM）应用中参数高效微调（PEFT）方法的性能。现有方法采用同质的MoE-LoRA架构，由具有相似或相同结构和容量的LoRA专家组成。然而，这些方法往往会导致表示坍塌和专家负载不平衡，从而负面影响LLM的潜力。为解决这些挑战，我们提出了一种异质的Mixture-of-Adapters（MoA）方法。该方法通过动态集成具有不同结构的PEFT适配器专家，利用它们互补的表示能力促进专家的专业化，从而增强预训练知识向下游任务的有效转移。MoA支持两种变体：（i）软MoA通过加权融合所有专家输出实现精细集成；（ii）稀疏MoA基于贡献稀疏激活适配器专家，实现接近无性能下降的效果。实验结果表明，异质MoA在性能和参数效率方面均优于同质MoE-LoRA方法。我们的项目可在以下链接访问：this https URL。 

---
# LengClaro2023: A Dataset of Administrative Texts in Spanish with Plain Language adaptations 

**Title (ZH)**: LengClaro2023: 带有plain language适应的西班牙语行政文本数据集 

**Authors**: Belén Agüera-Marco, Itziar Gonzalez-Dios  

**Link**: [PDF](https://arxiv.org/pdf/2506.05927)  

**Abstract**: In this work, we present LengClaro2023, a dataset of legal-administrative texts in Spanish. Based on the most frequently used procedures from the Spanish Social Security website, we have created for each text two simplified equivalents. The first version follows the recommendations provided by arText claro. The second version incorporates additional recommendations from plain language guidelines to explore further potential improvements in the system. The linguistic resource created in this work can be used for evaluating automatic text simplification (ATS) systems in Spanish. 

**Abstract (ZH)**: 本研究介绍了LengClaro2023，一个西班牙语法律行政文本数据集。基于西班牙社会保障网站上最常用的程序，我们为每篇文本创建了两个简化的等价版本。第一个版本遵循arText claro提供的建议。第二个版本结合了简洁语言指南的其他建议，以进一步探索系统改进的潜力。本研究创建的语言资源可以用于评估西班牙语自动文本简化系统（ATS）。 

---
# Small Models, Big Support: A Local LLM Framework for Teacher-Centric Content Creation and Assessment using RAG and CAG 

**Title (ZH)**: 小模型，大支持：基于RAG和CAG的以教师为中心的内容创编与评估本地LLM框架 

**Authors**: Zarreen Reza, Alexander Mazur, Michael T. Dugdale, Robin Ray-Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.05925)  

**Abstract**: While Large Language Models (LLMs) are increasingly utilized as student-facing educational aids, their potential to directly support educators, particularly through locally deployable and customizable open-source solutions, remains significantly underexplored. Many existing educational solutions rely on cloud-based infrastructure or proprietary tools, which are costly and may raise privacy concerns. Regulated industries with limited budgets require affordable, self-hosted solutions. We introduce an end-to-end, open-source framework leveraging small (3B-7B parameters), locally deployed LLMs for customized teaching material generation and assessment. Our system uniquely incorporates an interactive loop crucial for effective small-model refinement, and an auxiliary LLM verifier to mitigate jailbreaking risks, enhancing output reliability and safety. Utilizing Retrieval and Context Augmented Generation (RAG/CAG), it produces factually accurate, customized pedagogically-styled content. Deployed on-premises for data privacy and validated through an evaluation pipeline and a college physics pilot, our findings show that carefully engineered small LLM systems can offer robust, affordable, practical, and safe educator support, achieving utility comparable to larger models for targeted tasks. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs） increasingly被用作面向学生的教育辅助工具，它们直接支持教育工作者的潜力，尤其是在通过本地部署和可定制的开源解决方案方面，仍然被显著低估。许多现有的教育解决方案依赖于基于云的基础设施或专有工具，这往往成本高昂且可能引发隐私担忧。受到预算限制的受监管行业需要负担得起的、可自行托管的解决方案。我们引入了一个端到端的开源框架，利用本地部署的小型（3B-7B参数）LLM来生成和评估个性化的教学材料。我们的系统独特地集成了一个对于有效微调小型模型至关重要的互动循环，并包含一个辅助LLM验证器以减轻突破限制的风险，从而提高输出可靠性和安全性。利用检索和上下文增强生成（RAG/CAG）技术，它能够生成事实准确且符合教学风格的内容。该系统内置数据隐私保护措施，并通过评估管道和大学物理试点项目进行了验证，研究发现精心设计的小型LLM系统可以提供稳健、经济、实用且安全的教育支持，即使对于特定任务，其功能效用也与大型模型相当。 

---
# Rethinking Semi-supervised Segmentation Beyond Accuracy: Reliability and Robustness 

**Title (ZH)**: 超越准确性：重新思考半监督分割的可靠性和鲁棒性 

**Authors**: Steven Landgraf, Markus Hillemann, Markus Ulrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.05917)  

**Abstract**: Semantic segmentation is critical for scene understanding but demands costly pixel-wise annotations, attracting increasing attention to semi-supervised approaches to leverage abundant unlabeled data. While semi-supervised segmentation is often promoted as a path toward scalable, real-world deployment, it is astonishing that current evaluation protocols exclusively focus on segmentation accuracy, entirely overlooking reliability and robustness. These qualities, which ensure consistent performance under diverse conditions (robustness) and well-calibrated model confidences as well as meaningful uncertainties (reliability), are essential for safety-critical applications like autonomous driving, where models must handle unpredictable environments and avoid sudden failures at all costs. To address this gap, we introduce the Reliable Segmentation Score (RSS), a novel metric that combines predictive accuracy, calibration, and uncertainty quality measures via a harmonic mean. RSS penalizes deficiencies in any of its components, providing an easy and intuitive way of holistically judging segmentation models. Comprehensive evaluations of UniMatchV2 against its predecessor and a supervised baseline show that semi-supervised methods often trade reliability for accuracy. While out-of-domain evaluations demonstrate UniMatchV2's robustness, they further expose persistent reliability shortcomings. We advocate for a shift in evaluation protocols toward more holistic metrics like RSS to better align semi-supervised learning research with real-world deployment needs. 

**Abstract (ZH)**: 可靠的分割评分：一种综合预测准确性、校准和不确定性质量的新评价指标 

---
# Route-and-Reason: Scaling Large Language Model Reasoning with Reinforced Model Router 

**Title (ZH)**: 路线与理由：强化模型路由扩展大型语言模型推理 

**Authors**: Chenyang Shao, Xinyang Liu, Yutang Lin, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05901)  

**Abstract**: Multi-step reasoning has proven essential for enhancing the problem-solving capabilities of Large Language Models (LLMs) by decomposing complex tasks into intermediate steps, either explicitly or implicitly. Extending the reasoning chain at test time through deeper thought processes or broader exploration, can furthur improve performance, but often incurs substantial costs due to the explosion in token usage. Yet, many reasoning steps are relatively simple and can be handled by more efficient smaller-scale language models (SLMs). This motivates hybrid approaches that allocate subtasks across models of varying capacities. However, realizing such collaboration requires accurate task decomposition and difficulty-aware subtask allocation, which is challenging. To address this, we propose R2-Reasoner, a novel framework that enables collaborative reasoning across heterogeneous LLMs by dynamically routing sub-tasks based on estimated complexity. At the core of our framework is a Reinforced Model Router, composed of a task decomposer and a subtask allocator. The task decomposer segments complex input queries into logically ordered subtasks, while the subtask allocator assigns each subtask to the most appropriate model, ranging from lightweight SLMs to powerful LLMs, balancing accuracy and efficiency. To train this router, we introduce a staged pipeline that combines supervised fine-tuning on task-specific datasets with Group Relative Policy Optimization algorithm, enabling self-supervised refinement through iterative reinforcement learning. Extensive experiments across four challenging benchmarks demonstrate that R2-Reasoner reduces API costs by 86.85% while maintaining or surpassing baseline accuracy. Our framework paves the way for more cost-effective and adaptive LLM reasoning. The code is open-source at this https URL . 

**Abstract (ZH)**: 多步推理已被证明对于通过将复杂任务分解为中间步骤（显式或隐式）来增强大型语言模型（LLMs）的问题解决能力是必不可少的。通过在测试时进行更深入的思考过程或更广泛的探索来延伸推理链，可以进一步提高性能，但由于标记使用量的爆炸性增长，往往会带来显著的成本。然而，许多推理步骤相对简单，可以由更高效的小规模语言模型（SLMs）处理。这促成了跨不同容量模型分配子任务的混合方法。然而，实现这种协作需要准确的任务分解和难度感知的子任务分配，这是具有挑战性的。为了解决这一问题，我们提出了一种新的R2-Reasoner框架，该框架允许通过基于估计复杂性的动态路由在异构LLMs之间进行协作推理。该框架的核心是一个强化模型路由器，由一个任务分解器和一个子任务分配器组成。任务分解器将复杂的输入查询分割成逻辑有序的子任务，而子任务分配器将每个子任务分配给最适合的模型，从轻量级SLMs到强大的LLMs，平衡准确性和效率。为了训练这个路由器，我们引入了一种分阶段管道，结合特定任务数据集上的监督微调和群组相对策略优化算法，通过迭代强化学习实现自我监督的精炼。在四个具有挑战性的基准上进行的广泛实验表明，R2-Reasoner将API成本降低了86.85%，同时保持或超越了基线准确性。该框架为更经济高效和适应性的LLM推理铺平了道路。代码在该网址处开源。 

---
# WhisQ: Cross-Modal Representation Learning for Text-to-Music MOS Prediction 

**Title (ZH)**: WhisQ: 跨模态表示学习用于文本到音乐主观质量预测 

**Authors**: Jakaria Islam Emon, Kazi Tamanna Alam, Md. Abu Salek  

**Link**: [PDF](https://arxiv.org/pdf/2506.05899)  

**Abstract**: Mean Opinion Score (MOS) prediction for text to music systems requires evaluating both overall musical quality and text prompt alignment. This paper introduces WhisQ, a multimodal architecture that addresses this dual-assessment challenge through sequence level co-attention and optimal transport regularization. WhisQ employs the Whisper Base pretrained model for temporal audio encoding and Qwen 3, a 0.6B Small Language Model (SLM), for text encoding, with both maintaining sequence structure for fine grained cross-modal modeling. The architecture features specialized prediction pathways: OMQ is predicted from pooled audio embeddings, while TA leverages bidirectional sequence co-attention between audio and text. Sinkhorn optimal transport loss further enforce semantic alignment in the shared embedding space. On the MusicEval Track-1 dataset, WhisQ achieves substantial improvements over the baseline: 7% improvement in Spearman correlation for OMQ and 14% for TA. Ablation studies reveal that optimal transport regularization provides the largest performance gain (10% SRCC improvement), demonstrating the importance of explicit cross-modal alignment for text-to-music evaluation. 

**Abstract (ZH)**: 基于文本到音乐系统的Mean Opinion Score (MOS)预测需要评估整体音乐质量和文本提示对齐。本文介绍了WhisQ，这是一种多模态架构，通过序列级别共注意力和最优传输正则化来应对这种双重评估挑战。WhisQ 使用 Whisper Base 预训练模型进行时序音频编码，并使用 Qwen 3（一个 0.6B 小型语言模型）进行文本编码，两者都保持了序列结构以进行精细粒度的跨模态建模。该架构具有专门的预测路径：OMQ 从聚合的音频嵌入中预测，而 TA 则利用音频与文本之间的双向序列共注意力。最优传输损失进一步在共享嵌入空间中强制执行语义对齐。在 MusicEval Track-1 数据集上，WhisQ 在 OMQ 和 TA 方面均实现了显著提升：OMQ 的 Spearman 相关系数提高了 7%，TA 提高了 14%。消融研究结果显示，最优传输正则化提供了最大的性能增益（Spearman 相关系数提高了 10%），这表明明确的跨模态对齐对于文本到音乐评估的重要性。 

---
# Object Navigation with Structure-Semantic Reasoning-Based Multi-level Map and Multimodal Decision-Making LLM 

**Title (ZH)**: 基于结构语义推理的多层级地图与多模态决策导航 

**Authors**: Chongshang Yan, Jiaxuan He, Delun Li, Yi Yang, Wenjie Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.05896)  

**Abstract**: The zero-shot object navigation (ZSON) in unknown open-ended environments coupled with semantically novel target often suffers from the significant decline in performance due to the neglect of high-dimensional implicit scene information and the long-range target searching task. To address this, we proposed an active object navigation framework with Environmental Attributes Map (EAM) and MLLM Hierarchical Reasoning module (MHR) to improve its success rate and efficiency. EAM is constructed by reasoning observed environments with SBERT and predicting unobserved ones with Diffusion, utilizing human space regularities that underlie object-room correlations and area adjacencies. MHR is inspired by EAM to perform frontier exploration decision-making, avoiding the circuitous trajectories in long-range scenarios to improve path efficiency. Experimental results demonstrate that the EAM module achieves 64.5\% scene mapping accuracy on MP3D dataset, while the navigation task attains SPLs of 28.4\% and 26.3\% on HM3D and MP3D benchmarks respectively - representing absolute improvements of 21.4\% and 46.0\% over baseline methods. 

**Abstract (ZH)**: 基于环境属性图和多级逻辑记忆模块的零样本物体导航 

---
# HMVLM: Multistage Reasoning-Enhanced Vision-Language Model for Long-Tailed Driving Scenarios 

**Title (ZH)**: HMVLM：多阶段推理增强的视觉-语言模型用于长尾驾驶场景 

**Authors**: Daming Wang, Yuhao Song, Zijian He, Kangliang Chen, Xing Pan, Lu Deng, Weihao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05883)  

**Abstract**: We present HaoMo Vision-Language Model (HMVLM), an end-to-end driving framework that implements the slow branch of a cognitively inspired fast-slow architecture. A fast controller outputs low-level steering, throttle, and brake commands, while a slow planner-a large vision-language model-generates high-level intents such as "yield to pedestrian" or "merge after the truck" without compromising latency. HMVLM introduces three upgrades: (1) selective five-view prompting with an embedded 4s history of ego kinematics, (2) multi-stage chain-of-thought (CoT) prompting that enforces a Scene Understanding -> Driving Decision -> Trajectory Inference reasoning flow, and (3) spline-based trajectory post-processing that removes late-stage jitter and sharp turns. Trained on the Waymo Open Dataset, these upgrades enable HMVLM to achieve a Rater Feedback Score (RFS) of 7.7367, securing 2nd place in the 2025 Waymo Vision-based End-to-End (E2E) Driving Challenge and surpassing the public baseline by 2.77%. 

**Abstract (ZH)**: 基于视觉-语言模型的 HaoMo 驱动框架：一种认知启发式快速-缓慢架构的端到端驾驶框架 

---
# Bayesian Persuasion as a Bargaining Game 

**Title (ZH)**: 贝叶斯劝说作为一种讨价还价博弈 

**Authors**: Yue Lin, Shuhui Zhu, William A Cunningham, Wenhao Li, Pascal Poupart, Hongyuan Zha, Baoxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05876)  

**Abstract**: Bayesian persuasion, an extension of cheap-talk communication, involves an informed sender committing to a signaling scheme to influence a receiver's actions. Compared to cheap talk, this sender's commitment enables the receiver to verify the incentive compatibility of signals beforehand, facilitating cooperation. While effective in one-shot scenarios, Bayesian persuasion faces computational complexity (NP-hardness) when extended to long-term interactions, where the receiver may adopt dynamic strategies conditional on past outcomes and future expectations. To address this complexity, we introduce the bargaining perspective, which allows: (1) a unified framework and well-structured solution concept for long-term persuasion, with desirable properties such as fairness and Pareto efficiency; (2) a clear distinction between two previously conflated advantages: the sender's informational advantage and first-proposer advantage. With only modest modifications to the standard setting, this perspective makes explicit the common knowledge of the game structure and grants the receiver comparable commitment capabilities, thereby reinterpreting classic one-sided persuasion as a balanced information bargaining framework. The framework is validated through a two-stage validation-and-inference paradigm: We first demonstrate that GPT-o3 and DeepSeek-R1, out of publicly available LLMs, reliably handle standard tasks; We then apply them to persuasion scenarios to test that the outcomes align with what our information-bargaining framework suggests. All code, results, and terminal logs are publicly available at this http URL. 

**Abstract (ZH)**: 贝叶斯说服是一种扩展了廉价交谈的沟通方式，涉及知情的发送者承诺实施一种信号方案以影响接收者的行动。与廉价交谈相比，发送者的这种承诺使接收者能够在接收信号之前验证其激励相容性，从而促进合作。虽然在单一互动中有效，但当扩展到长期互动时，贝叶斯说服会遇到计算复杂性（NP难问题）的问题，因为在长期互动中，接收者可能会根据过去的成果和未来预期采取动态策略。为了应对这种复杂性，我们引入了讨价还价的视角，该视角允许：（1）为长期说服提供一个统一的框架和具有良好性质的解概念，如公平性和帕累托效率；（2）明确区分之前混淆的两种优势：发送者的信息优势和首次提议者优势。通过对标准设置进行适度修改，这种视角使游戏结构的共同知识得以明确，并赋予接收者相当的承诺能力，从而将经典的单方面说服重新解释为平衡信息讨价还价框架。该框架通过两阶段验证和推理范式进行验证：首先我们证明，在可公开获取的语言模型中，GPT-o3和DeepSeek-R1可靠地处理标准任务；然后我们将其应用于说服情景，测试其结果是否与我们的信息讨价还价框架的预测一致。所有代码、结果和终端日志均可通过此http网址公开访问。 

---
# Research on Personalized Financial Product Recommendation by Integrating Large Language Models and Graph Neural Networks 

**Title (ZH)**: 基于大规模语言模型和图神经网络的个性化金融产品推荐研究 

**Authors**: Yushang Zhao, Yike Peng, Dannier Li, Yuxin Yang, Chengrui Zhou, Jing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05873)  

**Abstract**: With the rapid growth of fintech, personalized financial product recommendations have become increasingly important. Traditional methods like collaborative filtering or content-based models often fail to capture users' latent preferences and complex relationships. We propose a hybrid framework integrating large language models (LLMs) and graph neural networks (GNNs). A pre-trained LLM encodes text data (e.g., user reviews) into rich feature vectors, while a heterogeneous user-product graph models interactions and social ties. Through a tailored message-passing mechanism, text and graph information are fused within the GNN to jointly optimize embeddings. Experiments on public and real-world financial datasets show our model outperforms standalone LLM or GNN in accuracy, recall, and NDCG, with strong interpretability. This work offers new insights for personalized financial recommendations and cross-modal fusion in broader recommendation tasks. 

**Abstract (ZH)**: 金融科技的快速成长使得个性化金融产品推荐愈发重要。传统方法如协作过滤或基于内容的模型往往难以捕捉用户的潜在偏好和复杂关系。我们提出一种结合大型语言模型（LLMs）和图神经网络（GNNs）的混合框架。预训练的LLM将文本数据（如用户评论）编码为丰富的特征向量，而异质用户-产品图则用于建模交互和社会关系。通过定制的消息传递机制，文本和图信息在GNN中融合以协同优化嵌入表示。在公共和真实世界的金融数据集上的实验表明，我们的模型在准确率、召回率和NDCG方面优于独立的LLM或GNN，并具有较强的可解释性。本工作为个性化金融推荐和更广泛的推荐任务中的跨模态融合提供了新的见解。 

---
# Loss Functions for Predictor-based Neural Architecture Search 

**Title (ZH)**: 基于预测器的神经架构搜索的损失函数 

**Authors**: Han Ji, Yuqi Feng, Jiahao Fan, Yanan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.05869)  

**Abstract**: Evaluation is a critical but costly procedure in neural architecture search (NAS). Performance predictors have been widely adopted to reduce evaluation costs by directly estimating architecture performance. The effectiveness of predictors is heavily influenced by the choice of loss functions. While traditional predictors employ regression loss functions to evaluate the absolute accuracy of architectures, recent approaches have explored various ranking-based loss functions, such as pairwise and listwise ranking losses, to focus on the ranking of architecture performance. Despite their success in NAS, the effectiveness and characteristics of these loss functions have not been thoroughly investigated. In this paper, we conduct the first comprehensive study on loss functions in performance predictors, categorizing them into three main types: regression, ranking, and weighted loss functions. Specifically, we assess eight loss functions using a range of NAS-relevant metrics on 13 tasks across five search spaces. Our results reveal that specific categories of loss functions can be effectively combined to enhance predictor-based NAS. Furthermore, our findings could provide practical guidance for selecting appropriate loss functions for various tasks. We hope this work provides meaningful insights to guide the development of loss functions for predictor-based methods in the NAS community. 

**Abstract (ZH)**: 性能评估是神经架构搜索(NAS)中的一个关键但成本高昂的程序。性能预测器已被广泛采用，通过直接估计架构性能来降低评估成本。预测器的有效性受到损失函数选择的影响。虽然传统预测器使用回归损失函数来评估架构的绝对准确性，但最近的方法已经探索了各种基于排名的损失函数，如成对和列表排名损失，以便重点关注架构性能的排名。尽管这些方法在NAS中取得了成功，但这些损失函数的有效性和特性尚未得到充分研究。本文首次对性能预测器中的损失函数进行全面研究，将它们分为三类：回归、排名和加权损失函数。具体而言，我们在涵盖五个搜索空间的13个任务上使用多种与NAS相关的指标评估八种损失函数。我们的结果表明，特定类别的损失函数可以有效组合以增强基于预测器的NAS。此外，我们的发现可以为各种任务选择合适的损失函数提供实用指导。我们希望这项工作能为NAS社区中预测器方法的损失函数开发提供有意义的指导。 

---
# Cross-View Multi-Modal Segmentation @ Ego-Exo4D Challenges 2025 

**Title (ZH)**: Ego-Exo4D 挑战赛 2025 多模态跨视角分割 

**Authors**: Yuqian Fu, Runze Wang, Yanwei Fu, Danda Pani Paudel, Luc Van Gool  

**Link**: [PDF](https://arxiv.org/pdf/2506.05856)  

**Abstract**: In this report, we present a cross-view multi-modal object segmentation approach for the object correspondence task in the Ego-Exo4D Correspondence Challenges 2025. Given object queries from one perspective (e.g., ego view), the goal is to predict the corresponding object masks in another perspective (e.g., exo view). To tackle this task, we propose a multimodal condition fusion module that enhances object localization by leveraging both visual masks and textual descriptions as segmentation conditions. Furthermore, to address the visual domain gap between ego and exo views, we introduce a cross-view object alignment module that enforces object-level consistency across perspectives, thereby improving the model's robustness to viewpoint changes. Our proposed method ranked second on the leaderboard of the large-scale Ego-Exo4D object correspondence benchmark. Code will be made available at this https URL. 

**Abstract (ZH)**: 在本报告中，我们提出了一个用于Ego-Exo4D对应挑战2025中的对象对应任务的跨视角多模态对象分割方法。给定一个视角的对象查询（例如，第一人称视图），目标是在另一个视角（例如，第三人称视图）中预测相应的对象掩码。为了解决这一任务，我们提出了一种多模态条件融合模块，通过利用视觉掩码和文本描述作为分割条件来增强对象定位。此外，为了应对第一人称视图和第三人称视图之间的视觉域差距，我们引入了一种跨视角对象对齐模块，确保各视角之间的一致性，从而提高模型对视角变化的鲁棒性。我们的提出的方法在大型Ego-Exo4D对象对应基准测试的排行榜上排名第二。代码将在以下链接处提供：这个 https URL。 

---
# DeepFake Doctor: Diagnosing and Treating Audio-Video Fake Detection 

**Title (ZH)**: DeepFake医生：音频-视频伪造检测的诊断与治疗 

**Authors**: Marcel Klemt, Carlotta Segna, Anna Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2506.05851)  

**Abstract**: Generative AI advances rapidly, allowing the creation of very realistic manipulated video and audio. This progress presents a significant security and ethical threat, as malicious users can exploit DeepFake techniques to spread misinformation. Recent DeepFake detection approaches explore the multimodal (audio-video) threat scenario. In particular, there is a lack of reproducibility and critical issues with existing datasets - such as the recently uncovered silence shortcut in the widely used FakeAVCeleb dataset. Considering the importance of this topic, we aim to gain a deeper understanding of the key issues affecting benchmarking in audio-video DeepFake detection. We examine these challenges through the lens of the three core benchmarking pillars: datasets, detection methods, and evaluation protocols. To address these issues, we spotlight the recent DeepSpeak v1 dataset and are the first to propose an evaluation protocol and benchmark it using SOTA models. We introduce SImple Multimodal BAseline (SIMBA), a competitive yet minimalistic approach that enables the exploration of diverse design choices. We also deepen insights into the issue of audio shortcuts and present a promising mitigation strategy. Finally, we analyze and enhance the evaluation scheme on the widely used FakeAVCeleb dataset. Our findings offer a way forward in the complex area of audio-video DeepFake detection. 

**Abstract (ZH)**: Generative AI在音频视频DeepFake检测基准测试中的挑战与对策：从数据集、检测方法和评估协议三个核心方面深入探究 

---
# Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models 

**Title (ZH)**: 跨语言坍缩：语言中心基础模型如何塑造大型语言模型的推理过程 

**Authors**: Cheonbok Park, Jeonghoon Kim, Joosung Lee, Sanghwan Bae, Jaegul Choo, Kangmin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05850)  

**Abstract**: We identify \textbf{Cross-lingual Collapse}, a systematic drift in which the chain-of-thought (CoT) of a multilingual language model reverts to its dominant pre-training language even when the prompt is expressed in a different language. Recent large language models (LLMs) with reinforcement learning with verifiable reward (RLVR) have achieved strong logical reasoning performances by exposing their intermediate reasoning traces, giving rise to large reasoning models (LRMs). However, the mechanism behind multilingual reasoning in LRMs is not yet fully explored. To investigate the issue, we fine-tune multilingual LRMs with Group-Relative Policy Optimization (GRPO) on translated versions of the GSM$8$K and SimpleRL-Zoo datasets in three different languages: Chinese, Korean, and Ukrainian. During training, we monitor both task accuracy and language consistency of the reasoning chains. Our experiments reveal three key findings: (i) GRPO rapidly amplifies pre-training language imbalances, leading to the erosion of low-resource languages within just a few hundred updates; (ii) language consistency reward mitigates this drift but does so at the expense of an almost 5 - 10 pp drop in accuracy. and (iii) the resulting language collapse is severely damaging and largely irreversible, as subsequent fine-tuning struggles to steer the model back toward its original target-language reasoning capabilities. Together, these findings point to a remarkable conclusion: \textit{not all languages are trained equally for reasoning}. Furthermore, our paper sheds light on the roles of reward shaping, data difficulty, and pre-training priors in eliciting multilingual reasoning. 

**Abstract (ZH)**: 跨语言坍缩：多语言语言模型链式思考的系统性漂移 

---
# Regional, Lattice and Logical Representations of Neural Networks 

**Title (ZH)**: 区域、晶格和逻辑表示的神经网络 

**Authors**: Sandro Preto, Marcelo Finger  

**Link**: [PDF](https://arxiv.org/pdf/2506.05834)  

**Abstract**: A possible path to the interpretability of neural networks is to (approximately) represent them in the regional format of piecewise linear functions, where regions of inputs are associated to linear functions computing the network outputs. We present an algorithm for the translation of feedforward neural networks with ReLU activation functions in hidden layers and truncated identity activation functions in the output layer. We also empirically investigate the complexity of regional representations outputted by our method for neural networks with varying sizes. Lattice and logical representations of neural networks are straightforward from regional representations as long as they satisfy a specific property. So we empirically investigate to what extent the translations by our algorithm satisfy such property. 

**Abstract (ZH)**: 一种可能实现神经网络可解释性的途径是将它们（近似地）表示为分段线性函数的区域格式，其中输入的区域与计算网络输出的线性函数相关联。我们提出了一种算法，用于将具有ReLU激活函数的隐藏层和截断恒.identity激活函数的输出层的前向神经网络进行转换。我们还 empirical 地研究了由我们方法输出的具有不同大小的神经网络的区域表示的复杂性。只要满足特定属性，神经网络的格子表示和逻辑表示直接从区域表示中得出。因此，我们 empirical 地研究了我们算法的转换到何种程度满足该属性。 

---
# Fuzzy Lattice-based Description Logic 

**Title (ZH)**: 基于模糊格的描述逻辑 

**Authors**: Yiwen Ding, Krishna Manoorkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05833)  

**Abstract**: Recently, description logic LE-ALC was introduced for reasoning in the semantic environment of enriched formal contexts, and a polynomial-time tableaux algorithm was developed to check the consistency of knowledge bases with acyclic TBoxes. In this work, we introduce a fuzzy generalization of LE-ALC  called  LE-FALC which provides a description logic counterpart of many-valued normal non-distributive logic a.k.a. many-valued LE-logic. This description logic can be used to represent and reason about knowledge in the formal framework  of fuzzy formal contexts and fuzzy formal concepts. We provide a tableaux algorithm that provides a complete and sound polynomial-time decision procedure to check the consistency of  LE-FALC  ABoxes. As a result, we also obtain an exponential-time decision procedure for checking the consistency of  LE-FALC  with acyclic TBoxes by unraveling. 

**Abstract (ZH)**: 最近，引入了描述逻辑LE-ALC以处理丰富形式背景下的语义环境中的推理，并开发了一种多项式时间表心算法来检查具有无环TBox的知识库的一致性。在此项工作中，我们介绍了LE-ALC的模糊推广LE-FALC，这是一种描述逻辑，为其提供了许多值正常非分配逻辑（即，许多值LE-逻辑）的形式背景和模糊形式概念框架中的知识表示和推理提供了一个对应物。我们提供了一种表心算法，该算法提供了一种完全且多项式时间的决策程序来检查LE-FALC ABoxes的一致性。因此，我们还通过展开法获得了检查LE-FALC与无环TBox一致性的一种指数时间决策程序。 

---
# Heartcare Suite: Multi-dimensional Understanding of ECG with Raw Multi-lead Signal Modeling 

**Title (ZH)**: Heartcare Suite: 多维度理解心电图的原始多导联信号建模 

**Authors**: Yihan Xie, Sijing Li, Tianwei Lin, Zhuonan Wang, Chenglin Yang, Yu Zhong, Wenqiao Zhang, Haoyuan Li, Hao Jiang, Fengda Zhang, Qishan Chen, Jun Xiao, Yueting Zhuang, Beng Chin Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05831)  

**Abstract**: We present Heartcare Suite, a multimodal comprehensive framework for finegrained electrocardiogram (ECG) understanding. It comprises three key components: (i) Heartcare-220K, a high-quality, structured, and comprehensive multimodal ECG dataset covering essential tasks such as disease diagnosis, waveform morphology analysis, and rhythm interpretation. (ii) Heartcare-Bench, a systematic and multi-dimensional benchmark designed to evaluate diagnostic intelligence and guide the optimization of Medical Multimodal Large Language Models (Med-MLLMs) in ECG scenarios. and (iii) HeartcareGPT with a tailored tokenizer Bidirectional ECG Abstract Tokenization (Beat), which compresses raw multi-lead signals into semantically rich discrete tokens via duallevel vector quantization and query-guided bidirectional diffusion mechanism. Built upon Heartcare-220K, HeartcareGPT achieves strong generalization and SoTA performance across multiple clinically meaningful tasks. Extensive experiments demonstrate that Heartcare Suite is highly effective in advancing ECGspecific multimodal understanding and evaluation. Our project is available at this https URL . 

**Abstract (ZH)**: 我们提出Heartcare Suite，这是一个多模态综合框架，用于精细的心电图（ECG）理解。它包含三个关键组成部分：(i) Heartcare-220K，一个高质量、结构化且全面的多模态ECG数据集，涵盖了疾病诊断、波形形态分析和节律解读等基本任务。(ii) Heartcare-Bench，一个系统且多维度的基准测试，用于评估诊断智能并指导多模态医疗大型语言模型（Med-MLLMs）在ECG场景中的优化。(iii) HeartcareGPT，一个配备定制化双向ECG摘要标记器（Beat）的模型，通过双层向量量化和查询指导的双向扩散机制，将原始多导联信号压缩为语义丰富且离散的标记。基于Heartcare-220K，HeartcareGPT在多个临床相关任务中实现了强大的泛化能力和SOTA性能。大量实验表明，Heartcare Suite在推动特定于心电图的多模态理解和评估方面非常有效。我们的项目可在以下链接访问：this https URL。 

---
# FuseUNet: A Multi-Scale Feature Fusion Method for U-like Networks 

**Title (ZH)**: FuseUNet：U形网络的多尺度特征融合方法 

**Authors**: Quansong He, Xiangde Min, Kaishen Wang, Tao He  

**Link**: [PDF](https://arxiv.org/pdf/2506.05821)  

**Abstract**: Medical image segmentation is a critical task in computer vision, with UNet serving as a milestone architecture. The typical component of UNet family is the skip connection, however, their skip connections face two significant limitations: (1) they lack effective interaction between features at different scales, and (2) they rely on simple concatenation or addition operations, which constrain efficient information integration. While recent improvements to UNet have focused on enhancing encoder and decoder capabilities, these limitations remain overlooked. To overcome these challenges, we propose a novel multi-scale feature fusion method that reimagines the UNet decoding process as solving an initial value problem (IVP), treating skip connections as discrete nodes. By leveraging principles from the linear multistep method, we propose an adaptive ordinary differential equation method to enable effective multi-scale feature fusion. Our approach is independent of the encoder and decoder architectures, making it adaptable to various U-Net-like networks. Experiments on ACDC, KiTS2023, MSD brain tumor, and ISIC2017/2018 skin lesion segmentation datasets demonstrate improved feature utilization, reduced network parameters, and maintained high performance. The code is available at this https URL. 

**Abstract (ZH)**: 医学图像分割是计算机视觉中的一个关键任务，UNet作为一种里程碑式的架构起到了重要作用。UNet家族的典型组件是跳跃连接，然而，这些跳跃连接面临两大显著限制：（1）它们在不同尺度特征之间的有效交互不足；（2）它们依赖于简单的连接或加法操作，限制了高效信息整合。尽管最近对UNet的改进主要集中在增强编码器和解码器的能力上，但这些限制仍被忽视。为克服这些挑战，我们提出了一种新颖的多尺度特征融合方法，将UNet的解码过程重新构想为求解初值问题（IVP），并将跳跃连接视为离散节点。通过利用线性多步法原理，我们提出了一个自适应常微分方程方法，以实现有效的多尺度特征融合。该方法独立于编码器和解码器架构，使其适用于各种U-Net类型的网络。在ACDC、KiTS2023、MSD脑肿瘤和ISIC2017/2018皮肤病变分割数据集上的实验表明，该方法能提高特征利用效率、减少网络参数数量，同时保持高性能。代码可在以下链接获取。 

---
# Positional Encoding meets Persistent Homology on Graphs 

**Title (ZH)**: Positional Encoding 结合 Persistent Homology 在图上的应用 

**Authors**: Yogesh Verma, Amauri H. Souza, Vikas Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.05814)  

**Abstract**: The local inductive bias of message-passing graph neural networks (GNNs) hampers their ability to exploit key structural information (e.g., connectivity and cycles). Positional encoding (PE) and Persistent Homology (PH) have emerged as two promising approaches to mitigate this issue. PE schemes endow GNNs with location-aware features, while PH methods enhance GNNs with multiresolution topological features. However, a rigorous theoretical characterization of the relative merits and shortcomings of PE and PH has remained elusive. We bridge this gap by establishing that neither paradigm is more expressive than the other, providing novel constructions where one approach fails but the other succeeds. Our insights inform the design of a novel learnable method, PiPE (Persistence-informed Positional Encoding), which is provably more expressive than both PH and PE. PiPE demonstrates strong performance across a variety of tasks (e.g., molecule property prediction, graph classification, and out-of-distribution generalization), thereby advancing the frontiers of graph representation learning. Code is available at this https URL. 

**Abstract (ZH)**: 消息传递图神经网络（GNNs）的局部归纳偏置阻碍了其利用关键结构信息（如连通性和循环）的能力。位置编码（PE）和持续同调（PH）方法已 emerged 作为两种有前景的解决方法。位置编码方案赋予 GNNs 位置感知特征，而持续同调方法则增强 GNNs 的多分辨率拓扑特征。然而，关于位置编码和持续同调的相对优缺点的严格的理论表征仍难以捉摸。我们通过建立两者在表达能力上互不占优的关系，填补了这一空白，并提供了新的构造，其中一个方法失败而另一个方法成功。我们的见解启发设计了一种新型可学习方法 PiPE（基于持续同调的位置编码），该方法在表达能力上被证明优于位置编码和持续同调。PiPE 在多种任务（如分子性质预测、图分类和离分布泛化）中表现出色，从而推进了图表示学习的前沿。代码可在以下网址获取：this https URL。 

---
# Robust sensor fusion against on-vehicle sensor staleness 

**Title (ZH)**: 针对车载传感器陈旧性的鲁棒传感器融合 

**Authors**: Meng Fan, Yifan Zuo, Patrick Blaes, Harley Montgomery, Subhasis Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.05780)  

**Abstract**: Sensor fusion is crucial for a performant and robust Perception system in autonomous vehicles, but sensor staleness, where data from different sensors arrives with varying delays, poses significant challenges. Temporal misalignment between sensor modalities leads to inconsistent object state estimates, severely degrading the quality of trajectory predictions that are critical for safety. We present a novel and model-agnostic approach to address this problem via (1) a per-point timestamp offset feature (for LiDAR and radar both relative to camera) that enables fine-grained temporal awareness in sensor fusion, and (2) a data augmentation strategy that simulates realistic sensor staleness patterns observed in deployed vehicles. Our method is integrated into a perspective-view detection model that consumes sensor data from multiple LiDARs, radars and cameras. We demonstrate that while a conventional model shows significant regressions when one sensor modality is stale, our approach reaches consistently good performance across both synchronized and stale conditions. 

**Abstract (ZH)**: 传感器融合对于自主车辆高性能和稳健的感知系统至关重要，但传感器数据延迟不一致带来的挑战显著。不同传感器数据的时间错位导致目标状态估计不一致，严重影响了对安全至关重要的轨迹预测质量。我们提出了一种模型无关的新方法，通过（1）一种针对点的时间戳偏移特征（LiDAR和雷达相对于摄像头），实现传感器融合中的精细时间感知，以及（2）一种模拟部署车辆中观察到的传感器延迟模式的数据增强策略。该方法集成到一个多传感器视图检测模型中，该模型消耗来自多个LiDAR、雷达和摄像头的数据。我们证明，尽管传统模型在某一种传感器模式延迟时表现显著下降，但我们的方法在同步和延迟条件下都能保持一致的良好性能。 

---
# dots.llm1 Technical Report 

**Title (ZH)**: dots.llm1 技术报告 

**Authors**: Bi Huo, Bin Tu, Cheng Qin, Da Zheng, Debing Zhang, Dongjie Zhang, En Li, Fu Guo, Jian Yao, Jie Lou, Junfeng Tian, Li Hu, Ran Zhu, Shengdong Chen, Shuo Liu, Su Guang, Te Wo, Weijun Zhang, Xiaoming Shi, Xinxin Peng, Xing Wu, Yawen Liu, Yuqiu Ji, Ze Wen, Zhenhai Liu, Zichao Li, Zilong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05767)  

**Abstract**: Mixture of Experts (MoE) models have emerged as a promising paradigm for scaling language models efficiently by activating only a subset of parameters for each input token. In this report, we present dots.llm1, a large-scale MoE model that activates 14B parameters out of a total of 142B parameters, delivering performance on par with state-of-the-art models while reducing training and inference costs. Leveraging our meticulously crafted and efficient data processing pipeline, dots.llm1 achieves performance comparable to Qwen2.5-72B after pretraining on 11.2T high-quality tokens and post-training to fully unlock its capabilities. Notably, no synthetic data is used during pretraining. To foster further research, we open-source intermediate training checkpoints at every one trillion tokens, providing valuable insights into the learning dynamics of large language models. 

**Abstract (ZH)**: 混合专家模型（MoE）作为一种高效扩展语言模型的 paradigm，通过为每个输入令牌仅激活部分参数而得以发展。本报告介绍了 dots.llm1，一个大型 MoE 模型，激活了总计 142B 参数中的 14B 参数，其性能与最先进的模型相当，同时降低了训练和推理成本。借助我们精心设计且高效的數據处理管道，dots.llm1 在预训练 11.2T 高质量令牌后，经过进一步训练以充分解锁其能力，其性能与 Qwen2.5-72B 相当。值得注意的是，预训练过程中未使用合成数据。为了促进进一步研究，我们开源了每隔一万亿令牌的中间训练检查点，提供了关于大型语言模型学习动态的宝贵见解。 

---
# Revealing hidden correlations from complex spatial distributions: Adjacent Correlation Analysis 

**Title (ZH)**: 揭示复杂空间分布中的隐含相关性：相邻相关性分析 

**Authors**: Guang-Xing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05759)  

**Abstract**: Physics has been transforming our view of nature for centuries. While combining physical knowledge with computational approaches has enabled detailed modeling of physical systems' evolution, understanding the emergence of patterns and structures remains limited. Correlations between quantities are the most reliable approach to describe relationships between different variables. However, for complex patterns, directly searching for correlations is often impractical, as complexity and spatial inhomogeneity can obscure correlations. We discovered that the key is to search for correlations in local regions and developed a new method, adjacent correlation analysis, to extract such correlations and represent them in phase space. When multiple observations are available, a useful way to study a system is to analyze distributions in phase space using the Probability Density Function (PDF). Adjacent correlation analysis evaluates vectors representing local correlations, which can be overlaid on the PDF plot to form the adjacent correlation plot. These correlation vectors often exhibit remarkably regular patterns and may lead to the discovery of new laws. The vectors we derive are equivalent to the vector field in dynamical systems on the attracting manifold. By efficiently representing spatial patterns as correlation vectors in phase space, our approach opens avenues for classification, prediction, parameter fitting, and forecasting. 

**Abstract (ZH)**: 物理学一直在改变我们对自然界的看法。将物理知识与计算方法相结合虽然能够详细模拟物理系统的演化，但对模式和结构的涌现的理解仍然有限。不同变量之间的相关性是最可靠的描述不同变量关系的方法。然而，对于复杂的模式，直接寻找相关性往往不切实际，因为复杂性和空间不均匀性可能会掩盖相关性。我们发现关键是在局部区域寻找相关性，并开发了一种新方法——邻近相关分析，以提取这些相关性并在相空间中表示。当有多次观测时，研究系统的一个有用方法是使用概率密度函数（PDF）分析相空间中的分布。邻近相关分析评估代表局部相关性的向量，并可以在PDF图上叠加形成邻近相关图。这些相关向量通常表现出显著的规律性，可能引领发现新的定律。我们推导出的向量等同于吸引流形上的动力系统向量场。通过在相空间中有效表示空间模式为相关向量，我们的方法为分类、预测、参数拟合和预报开辟了途径。 

---
# FlowOE: Imitation Learning with Flow Policy from Ensemble RL Experts for Optimal Execution under Heston Volatility and Concave Market Impacts 

**Title (ZH)**: FlowOE：基于组合_rl专家流动策略的heston波动率和凹市场影响下的最优执行imitation学习 

**Authors**: Yang Li, Zhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05755)  

**Abstract**: Optimal execution in financial markets refers to the process of strategically transacting a large volume of assets over a period to achieve the best possible outcome by balancing the trade-off between market impact costs and timing or volatility risks. Traditional optimal execution strategies, such as static Almgren-Chriss models, often prove suboptimal in dynamic financial markets. This paper propose flowOE, a novel imitation learning framework based on flow matching models, to address these limitations. FlowOE learns from a diverse set of expert traditional strategies and adaptively selects the most suitable expert behavior for prevailing market conditions. A key innovation is the incorporation of a refining loss function during the imitation process, enabling flowOE not only to mimic but also to improve upon the learned expert actions. To the best of our knowledge, this work is the first to apply flow matching models in a stochastic optimal execution problem. Empirical evaluations across various market conditions demonstrate that flowOE significantly outperforms both the specifically calibrated expert models and other traditional benchmarks, achieving higher profits with reduced risk. These results underscore the practical applicability and potential of flowOE to enhance adaptive optimal execution. 

**Abstract (ZH)**: 基于流匹配模型的流OE：一类新颖的模仿学习框架在金融市场最优执行中的应用 

---
# Integrating Spatiotemporal Features in LSTM for Spatially Informed COVID-19 Hospitalization Forecasting 

**Title (ZH)**: 基于空间信息的COVID-19住院情况时空特征集成LSTM预报 

**Authors**: Zhongying Wang, Thoai D. Ngo, Hamidreza Zoraghein, Benjamin Lucas, Morteza Karimzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.05752)  

**Abstract**: The COVID-19 pandemic's severe impact highlighted the need for accurate, timely hospitalization forecasting to support effective healthcare planning. However, most forecasting models struggled, especially during variant surges, when they were needed most. This study introduces a novel Long Short-Term Memory (LSTM) framework for forecasting daily state-level incident hospitalizations in the United States. We present a spatiotemporal feature, Social Proximity to Hospitalizations (SPH), derived from Facebook's Social Connectedness Index to improve forecasts. SPH serves as a proxy for interstate population interaction, capturing transmission dynamics across space and time. Our parallel LSTM architecture captures both short- and long-term temporal dependencies, and our multi-horizon ensembling strategy balances consistency and forecasting error. Evaluation against COVID-19 Forecast Hub ensemble models during the Delta and Omicron surges reveals superiority of our model. On average, our model surpasses the ensemble by 27, 42, 54, and 69 hospitalizations per state on the $7^{th}$, $14^{th}$, $21^{st}$, and $28^{th}$ forecast days, respectively, during the Omicron surge. Data-ablation experiments confirm SPH's predictive power, highlighting its effectiveness in enhancing forecasting models. This research not only advances hospitalization forecasting but also underscores the significance of spatiotemporal features, such as SPH, in refining predictive performance in modeling the complex dynamics of infectious disease spread. 

**Abstract (ZH)**: COVID-19大流行严重的影响突显了准确及时的住院预测支持有效医疗规划的必要性。然而，在需要这些预测的变异激增期间，大多数预测模型都遇到了困难。本研究介绍了一种新型的长短期记忆（LSTM）框架，用于预测美国每日州级新增住院情况。我们提出了一种时空特征，即基于Facebook社交连通性指数衍生的社交亲近度到住院情况（SPH），以改进预测。SPH 作为州际人口互动的代理指标，捕捉了空间和时间上的传播动态。我们的并行LSTM架构捕获了短期和长期的时序依赖关系，我们的多视线组合策略平衡了一致性和预测误差。在Delta和Omicron变异激增期间与COVID-19预测联合体模型的评估显示了我们模型的优势。在Omicron激增期间，我们的模型在第7、14、21和28天预测日分别比联合体模型多出27、42、54和69名住院病例。数据消融实验确认了SPH 的预测能力，突显了其在增强预测模型中的有效性。这项研究不仅推进了住院预测，还强调了如SPH这类时空特征在建模传染病传播复杂动态中改进预测性能的重要性。 

---
# An Ontology for Representing Curriculum and Learning Material 

**Title (ZH)**: 一种表示课程和学习材料的本体论 

**Authors**: Antrea Christou, Chris Davis Jaldi, Joseph Zalewski, Hande Küçük McGinty, Pascal Hitzler, Cogan Shimizu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05751)  

**Abstract**: Educational, learning, and training materials have become extremely commonplace across the Internet. Yet, they frequently remain disconnected from each other, fall into platform silos, and so on. One way to overcome this is to provide a mechanism to integrate the material and provide cross-links across topics.
In this paper, we present the Curriculum KG Ontology, which we use as a framework for the dense interlinking of educational materials, by first starting with organizational and broad pedagogical principles. We provide a materialized graph for the Prototype Open Knowledge Network use-case, and validate it using competency questions sourced from domain experts and educators. 

**Abstract (ZH)**: 教育、学习和培训材料在网络空间中变得极为常见，但它们往往缺乏连接，处于不同的平台孤岛之中。为了解决这一问题，我们提供了一种机制来整合这些材料并在不同主题之间提供跨链接。本文介绍了课程KG本体，我们将其用作密集互联教育材料的框架，首先基于组织和广泛的教学原则。我们为原型开放知识网络用例提供了实现的图，并通过领域专家和教育者来源的能力问题进行了验证。 

---
# Efficient Online RFT with Plug-and-Play LLM Judges: Unlocking State-of-the-Art Performance 

**Title (ZH)**: 高效的在线RFT与插拔式LLM裁判：解锁最先进的性能 

**Authors**: Rudransh Agnihotri, Ananya Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2506.05748)  

**Abstract**: Reward-model training is the cost bottleneck in modern Reinforcement Learning Human Feedback (RLHF) pipelines, often requiring tens of billions of parameters and an offline preference-tuning phase. In the proposed method, a frozen, instruction-tuned 7B LLM is augmented with only a one line JSON rubric and a rank-16 LoRA adapter (affecting just 0.8% of the model's parameters), enabling it to serve as a complete substitute for the previously used heavyweight evaluation models. The plug-and-play judge achieves 96.2% accuracy on RewardBench, outperforming specialized reward networks ranging from 27B to 70B parameters. Additionally, it allows a 7B actor to outperform the top 70B DPO baseline, which scores 61.8%, by achieving 92% exact match accuracy on GSM-8K utilizing online PPO. Thorough ablations indicate that (i) six in context demonstrations deliver the majority of the zero-to-few-shot improvements (+2pp), and (ii) the LoRA effectively addresses the remaining disparity, particularly in the safety and adversarial Chat-Hard segments. The proposed model introduces HH-Rationales, a subset of 10,000 pairs from Anthropic HH-RLHF, to examine interpretability, accompanied by human generated justifications. GPT-4 scoring indicates that our LoRA judge attains approximately = 9/10 in similarity to human explanations, while zero-shot judges score around =5/10. These results indicate that the combination of prompt engineering and tiny LoRA produces a cost effective, transparent, and easily adjustable reward function, removing the offline phase while achieving new state-of-the-art outcomes for both static evaluation and online RLHF. 

**Abstract (ZH)**: 基于插拔式裁判的高效RLHF方法：结合提示工程与小型LoRA生成成本 Effective, Transparent, and Easily Adjustable Reward Function via Plug-and-Play Judge and Prompt Engineering with Small LoRA 

---
# When Better Features Mean Greater Risks: The Performance-Privacy Trade-Off in Contrastive Learning 

**Title (ZH)**: 当更好的特征意味着更大的风险：对比学习的性能-隐私权衡 

**Authors**: Ruining Sun, Hongsheng Hu, Wei Luo, Zhaoxi Zhang, Yanjun Zhang, Haizhuan Yuan, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05743)  

**Abstract**: With the rapid advancement of deep learning technology, pre-trained encoder models have demonstrated exceptional feature extraction capabilities, playing a pivotal role in the research and application of deep learning. However, their widespread use has raised significant concerns about the risk of training data privacy leakage. This paper systematically investigates the privacy threats posed by membership inference attacks (MIAs) targeting encoder models, focusing on contrastive learning frameworks. Through experimental analysis, we reveal the significant impact of model architecture complexity on membership privacy leakage: As more advanced encoder frameworks improve feature-extraction performance, they simultaneously exacerbate privacy-leakage risks. Furthermore, this paper proposes a novel membership inference attack method based on the p-norm of feature vectors, termed the Embedding Lp-Norm Likelihood Attack (LpLA). This method infers membership status, by leveraging the statistical distribution characteristics of the p-norm of feature vectors. Experimental results across multiple datasets and model architectures demonstrate that LpLA outperforms existing methods in attack performance and robustness, particularly under limited attack knowledge and query volumes. This study not only uncovers the potential risks of privacy leakage in contrastive learning frameworks, but also provides a practical basis for privacy protection research in encoder models. We hope that this work will draw greater attention to the privacy risks associated with self-supervised learning models and shed light on the importance of a balance between model utility and training data privacy. Our code is publicly available at: this https URL. 

**Abstract (ZH)**: 随着深度学习技术的迅速发展，预训练编码器模型展示了卓越的特征提取能力，在深度学习的研究与应用中发挥了关键作用。然而，它们的广泛应用也引发了关于训练数据隐私泄露风险的重大担忧。本文系统地探讨了针对编码器模型的会员推理攻击（MIA）所造成的数据隐私威胁，重点关注对比学习框架。通过实验分析，我们揭示了模型架构复杂性对会员隐私泄露的重大影响：随着更先进的编码器框架提升特征提取性能，它们同时加剧了隐私泄露风险。此外，本文提出了一种基于特征向量p-范数的新颖会员推理攻击方法，称为嵌入Lp-范数似然攻击（LpLA）。该方法通过利用特征向量p-范数的统计分布特性来推断会员状态。在多个数据集和模型架构上的实验结果表明，LpLA在攻击性能和鲁棒性方面优于现有方法，尤其是在有限的攻击知识和查询量条件下。本研究不仅揭示了对比学习框架中隐私泄露的潜在风险，还为编码器模型的隐私保护研究提供了实用基础。我们期望这项工作能引起对自监督学习模型隐私风险的关注，并强调模型实用性和训练数据隐私之间的平衡的重要性。我们的代码已公开发布在：this https URL。 

---
# To Protect the LLM Agent Against the Prompt Injection Attack with Polymorphic Prompt 

**Title (ZH)**: 使用多态提示来保护LLM代理免受提示注入攻击 

**Authors**: Zhilong Wang, Neha Nagaraja, Lan Zhang, Hayretdin Bahsi, Pawan Patil, Peng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05739)  

**Abstract**: LLM agents are widely used as agents for customer support, content generation, and code assistance. However, they are vulnerable to prompt injection attacks, where adversarial inputs manipulate the model's behavior. Traditional defenses like input sanitization, guard models, and guardrails are either cumbersome or ineffective. In this paper, we propose a novel, lightweight defense mechanism called Polymorphic Prompt Assembling (PPA), which protects against prompt injection with near-zero overhead. The approach is based on the insight that prompt injection requires guessing and breaking the structure of the system prompt. By dynamically varying the structure of system prompts, PPA prevents attackers from predicting the prompt structure, thereby enhancing security without compromising performance. We conducted experiments to evaluate the effectiveness of PPA against existing attacks and compared it with other defense methods. 

**Abstract (ZH)**: LLM代理广泛用于客户支持、内容生成和代码辅助。然而，它们容易受到提示注入攻击的影响，即攻击者通过操纵输入来改变模型的行为。传统的防御方法如输入 sanitization、防护模型和防护栏要么复杂难以实施，要么效果不佳。本文提出了一种新颖的轻量级防御机制——多态性提示组装（PPA），该机制能够以接近零的性能开销来抵御提示注入攻击。该方法基于这样一个洞察：提示注入需要猜测和破解系统提示的结构。通过动态变化系统提示的结构，PPA 阻止攻击者预测提示结构，从而在不牺牲性能的情况下增强安全性。我们进行了实验评估了 PPA 对现有攻击的有效性，并将其与其它防御方法进行了比较。 

---
# Generalized Incremental Learning under Concept Drift across Evolving Data Streams 

**Title (ZH)**: 概念漂移下 evolving 数据流的广义增量学习 

**Authors**: En Yu, Jie Lu, Guangquan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05736)  

**Abstract**: Real-world data streams exhibit inherent non-stationarity characterized by concept drift, posing significant challenges for adaptive learning systems. While existing methods address isolated distribution shifts, they overlook the critical co-evolution of label spaces and distributions under limited supervision and persistent uncertainty. To address this, we formalize Generalized Incremental Learning under Concept Drift (GILCD), characterizing the joint evolution of distributions and label spaces in open-environment streaming contexts, and propose a novel framework called Calibrated Source-Free Adaptation (CSFA). First, CSFA introduces a training-free prototype calibration mechanism that dynamically fuses emerging prototypes with base representations, enabling stable new-class identification without optimization overhead. Second, we design a novel source-free adaptation algorithm, i.e., Reliable Surrogate Gap Sharpness-aware (RSGS) minimization. It integrates sharpness-aware perturbation loss optimization with surrogate gap minimization, while employing entropy-based uncertainty filtering to discard unreliable samples. This mechanism ensures robust distribution alignment and mitigates generalization degradation caused by uncertainties. Therefore, CSFA establishes a unified framework for stable adaptation to evolving semantics and distributions in open-world streaming scenarios. Extensive experiments validate the superior performance and effectiveness of CSFA compared to state-of-the-art approaches. 

**Abstract (ZH)**: 实时时序数据表现出由概念漂移特征化的固有非平稳性，为自适应学习系统带来了重大挑战。现有方法虽能应对孤立的分布转移，但忽视了在有限监督和持续不确定性条件下标签空间和分布的共同演化。为解决这一问题，我们正式定义了一种概念漂移下的广义增量学习框架（GILCD），描述了开放环境流式数据中分布和标签空间的联合演化，并提出了一种名为校准无源适应（CSFA）的新框架。首先，CSFA引入了一种无训练原型校准机制，能够动态融合新兴原型与基础表示，实现稳定的新生类别识别且无优化开销。其次，我们设计了一种名为可靠替代缺口敏锐度aware（RSGS）最小化的新无源适应算法。该算法结合了敏锐度aware扰动损失优化和替代缺口最小化，并利用基于熵的不确定性过滤剔除不可靠样本。此机制确保稳健的分布对齐并减轻由不确定性引起的泛化退化。因此，CSFA建立了一个统一框架，以在开放世界的流式场景中稳定地适应演变的概念和分布。大量实验证明了CSFA在性能和效果上的优越性，超过了最先进的方法。 

---
# Large Language Models are Good Relational Learners 

**Title (ZH)**: 大型语言模型是良好的关系学习者 

**Authors**: Fang Wu, Vijay Prakash Dwivedi, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2506.05725)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various domains, yet their application to relational deep learning (RDL) remains underexplored. Existing approaches adapt LLMs by traversing relational links between entities in a database and converting the structured data into flat text documents. Still, this text-based serialization disregards critical relational structures, introduces redundancy, and often exceeds standard LLM context lengths. We introduce Rel-LLM, a novel architecture that utilizes a graph neural network (GNN)- based encoder to generate structured relational prompts for LLMs within a retrieval-augmented generation (RAG) framework. Unlike traditional text-based serialization approaches, our method preserves the inherent relational structure of databases while enabling LLMs to effectively process and reason over complex entity relationships. Specifically, the GNN encoder extracts a local subgraph around an entity to build feature representations that contain relevant entity relationships and temporal dependencies. These representations are transformed into structured prompts using a denormalization process, effectively allowing the LLM to reason over relational structures. Through extensive experiments, we demonstrate that Rel-LLM outperforms existing methods on key RDL tasks, offering a scalable and efficient approach to integrating LLMs with structured data sources. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域展现了卓越的能力，但其在关系深度学习（RDL）中的应用仍处于探索之中。现有方法通过遍历数据库实体间的关系链接并转换结构化数据为扁平文本文件来适应LLMs，但这种基于文本的序列化方式忽视了关键的关系结构，引入了冗余，并且常超过标准LLM的上下文长度。我们提出Rel-LLM，这是一种新颖的架构，利用基于图神经网络（GNN）的编码器在检索增强生成（RAG）框架中为LLMs生成结构化的关系提示。与传统的基于文本的序列化方法不同，我们的方法保留了数据库的固有关系结构，同时使LLMs能够有效处理和推理复杂的实体关系。具体而言，GNN编码器提取实体周围的局部子图，构建包含相关实体关系和时间依赖性的特征表示。这些表示通过反规范化过程转换为结构化提示，有效地使LLMs能够在关系结构上进行推理。通过广泛的实验，我们证明Rel-LLM在关键的RDL任务上优于现有方法，提供了一种将LLMs与结构化数据源集成的可扩展和高效的方法。代码可访问：这个链接。 

---
# Any-Class Presence Likelihood for Robust Multi-Label Classification with Abundant Negative Data 

**Title (ZH)**: 面对丰富负样本的鲁棒多标签分类中的任意类别存在概率 

**Authors**: Dumindu Tissera, Omar Awadallah, Muhammad Umair Danish, Ayan Sadhu, Katarina Grolinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.05721)  

**Abstract**: Multi-label Classification (MLC) assigns an instance to one or more non-exclusive classes. A challenge arises when the dataset contains a large proportion of instances with no assigned class, referred to as negative data, which can overwhelm the learning process and hinder the accurate identification and classification of positive instances. Nevertheless, it is common in MLC applications such as industrial defect detection, agricultural disease identification, and healthcare diagnosis to encounter large amounts of negative data. Assigning a separate negative class to these instances further complicates the learning objective and introduces unnecessary redundancies. To address this challenge, we redesign standard MLC loss functions by deriving a likelihood of any class being present, formulated by a normalized weighted geometric mean of the predicted class probabilities. We introduce a regularization parameter that controls the relative contribution of the absent class probabilities to the any-class presence likelihood in positive instances. The any-class presence likelihood complements the multi-label learning by encouraging the network to become more aware of implicit positive instances and improve the label classification within those positive instances. Experiments on large-scale datasets with negative data: SewerML, modified COCO, and ChestX-ray14, across various networks and base loss functions show that our loss functions consistently improve MLC performance of their standard loss counterparts, achieving gains of up to 6.01 percentage points in F1, 8.06 in F2, and 3.11 in mean average precision, all without additional parameters or computational complexity. Code available at: this https URL 

**Abstract (ZH)**: 多标签分类中负数据处理的损失函数设计：基于归一化加权几何均值的概率构建方法 

---
# Grokking Beyond the Euclidean Norm of Model Parameters 

**Title (ZH)**: 超越模型参数欧几里得范数的理解 

**Authors**: Pascal Jr Tikeng Notsawo, Guillaume Dumas, Guillaume Rabusseau  

**Link**: [PDF](https://arxiv.org/pdf/2506.05718)  

**Abstract**: Grokking refers to a delayed generalization following overfitting when optimizing artificial neural networks with gradient-based methods. In this work, we demonstrate that grokking can be induced by regularization, either explicit or implicit. More precisely, we show that when there exists a model with a property $P$ (e.g., sparse or low-rank weights) that generalizes on the problem of interest, gradient descent with a small but non-zero regularization of $P$ (e.g., $\ell_1$ or nuclear norm regularization) results in grokking. This extends previous work showing that small non-zero weight decay induces grokking. Moreover, our analysis shows that over-parameterization by adding depth makes it possible to grok or ungrok without explicitly using regularization, which is impossible in shallow cases. We further show that the $\ell_2$ norm is not a reliable proxy for generalization when the model is regularized toward a different property $P$, as the $\ell_2$ norm grows in many cases where no weight decay is used, but the model generalizes anyway. We also show that grokking can be amplified solely through data selection, with any other hyperparameter fixed. 

**Abstract (ZH)**: 过度拟合后的推迟泛化可以通过正则化诱导：从显式或隐式正则化到属性泛化的扩展研究 

---
# Ensemble Elastic DQN: A novel multi-step ensemble approach to address overestimation in deep value-based reinforcement learning 

**Title (ZH)**: 集成弹性DQN：一种解决深度值基于强化学习中过度估计的新型多步集成方法 

**Authors**: Adrian Ly, Richard Dazeley, Peter Vamplew, Francisco Cruz, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2506.05716)  

**Abstract**: While many algorithmic extensions to Deep Q-Networks (DQN) have been proposed, there remains limited understanding of how different improvements interact. In particular, multi-step and ensemble style extensions have shown promise in reducing overestimation bias, thereby improving sample efficiency and algorithmic stability. In this paper, we introduce a novel algorithm called Ensemble Elastic Step DQN (EEDQN), which unifies ensembles with elastic step updates to stabilise algorithmic performance. EEDQN is designed to address two major challenges in deep reinforcement learning: overestimation bias and sample efficiency. We evaluated EEDQN against standard and ensemble DQN variants across the MinAtar benchmark, a set of environments that emphasise behavioral learning while reducing representational complexity. Our results show that EEDQN achieves consistently robust performance across all tested environments, outperforming baseline DQN methods and matching or exceeding state-of-the-art ensemble DQNs in final returns on most of the MinAtar environments. These findings highlight the potential of systematically combining algorithmic improvements and provide evidence that ensemble and multi-step methods, when carefully integrated, can yield substantial gains. 

**Abstract (ZH)**: 深度Q网络的集成弹性步长DQN：一种统一算法稳定性的新方法 

---
# Action-Adaptive Continual Learning: Enabling Policy Generalization under Dynamic Action Spaces 

**Title (ZH)**: 动态动作空间下的行动自适应连续学习：促进策略泛化 

**Authors**: Chaofan Pan, Jiafen Liu, Yanhua Li, Linbo Xiong, Fan Min, Wei Wei, Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05702)  

**Abstract**: Continual Learning (CL) is a powerful tool that enables agents to learn a sequence of tasks, accumulating knowledge learned in the past and using it for problem-solving or future task learning. However, existing CL methods often assume that the agent's capabilities remain static within dynamic environments, which doesn't reflect real-world scenarios where capabilities dynamically change. This paper introduces a new and realistic problem: Continual Learning with Dynamic Capabilities (CL-DC), posing a significant challenge for CL agents: How can policy generalization across different action spaces be achieved? Inspired by the cortical functions, we propose an Action-Adaptive Continual Learning framework (AACL) to address this challenge. Our framework decouples the agent's policy from the specific action space by building an action representation space. For a new action space, the encoder-decoder of action representations is adaptively fine-tuned to maintain a balance between stability and plasticity. Furthermore, we release a benchmark based on three environments to validate the effectiveness of methods for CL-DC. Experimental results demonstrate that our framework outperforms popular methods by generalizing the policy across action spaces. 

**Abstract (ZH)**: 连续学习中的动态能力（Continual Learning with Dynamic Capabilities, CL-DC） 

---
# RKEFino1: A Regulation Knowledge-Enhanced Large Language Model 

**Title (ZH)**: RKEFino1：一种调控知识增强的大语言模型 

**Authors**: Yan Wang, Yueru He, Ruoyu Xiang, Jeff Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05700)  

**Abstract**: Recent advances in large language models (LLMs) hold great promise for financial applications but introduce critical accuracy and compliance challenges in Digital Regulatory Reporting (DRR). To address these issues, we propose RKEFino1, a regulation knowledge-enhanced financial reasoning model built upon Fino1, fine-tuned with domain knowledge from XBRL, CDM, and MOF. We formulate two QA tasks-knowledge-based and mathematical reasoning-and introduce a novel Numerical NER task covering financial entities in both sentences and tables. Experimental results demonstrate the effectiveness and generalization capacity of RKEFino1 in compliance-critical financial tasks. We have released our model on Hugging Face. 

**Abstract (ZH)**: Recent Advances in Large Language Models for Digital Regulatory Reporting: Introducing RKEFino1, a Regulation Knowledge-Enhanced Financial Reasoning Model 

---
# Evaluating AI-Powered Learning Assistants in Engineering Higher Education: Student Engagement, Ethical Challenges, and Policy Implications 

**Title (ZH)**: 基于人工智能的学习助手在工程高等教育中的评估：学生参与、伦理挑战及政策意义 

**Authors**: Ramteja Sajja, Yusuf Sermet, Brian Fodale, Ibrahim Demir  

**Link**: [PDF](https://arxiv.org/pdf/2506.05699)  

**Abstract**: As generative AI tools become increasingly integrated into higher education, understanding how students interact with and perceive these technologies is essential for responsible and effective adoption. This study evaluates the use of the Educational AI Hub, an AI-powered learning framework, in undergraduate civil and environmental engineering courses at a large R1 public university. Using a mixed-methods approach that combines pre- and post-surveys, system usage logs, and qualitative analysis of the open-ended prompts and questions students posed to the AI chatbot, the research explores students' perceptions of trust, ethical concerns, usability, and learning outcomes. Findings reveal that students appreciated the AI assistant for its convenience and comfort, with nearly half reporting greater ease in using the AI tool compared to seeking help from instructors or teaching assistants. The tool was seen as most helpful for completing homework and understanding course concepts, though perceptions of its instructional quality were mixed. Ethical concerns emerged as a key barrier to full engagement: while most students viewed AI use as ethically acceptable, many expressed uncertainties about institutional policies and apprehension about potential academic misconduct. This study contributes to the growing body of research on AI in education by highlighting the importance of usability, policy clarity, and faculty guidance in fostering meaningful AI engagement. The findings suggest that while students are ready to embrace AI as a supplement to human instruction, thoughtful integration and transparent institutional frameworks are critical for ensuring student confidence, trust, and learning effectiveness. 

**Abstract (ZH)**: 随着生成式AI工具在高等教育中的日益集成，理解学生与这些技术的互动及其感知对于负责任和有效采用这些技术至关重要。本研究评估了在一所大型R1公立大学的土木与环境工程本科课程中使用Educational AI Hub（一种基于AI的学习框架）的情况。通过结合预调查、后调查、系统使用日志以及对学生提出给AI聊天机器人的开放性问题和提示的定性分析，本研究探讨了学生对信任、伦理关注、易用性和学习成果的感知。研究发现，学生高度评价AI助手的便利性和舒适性，近半数学生认为使用AI工具比向教师或助教求助更容易。该工具被认为在完成作业和理解课程概念方面最有帮助，尽管对学生教学质量的感知存在差异。伦理关注成为全面参与的关键障碍：虽然大多数学生认为使用AI在伦理上是可以接受的，但许多人对机构政策表示不确定，并担心潜在的学术不当行为。本研究通过强调易用性、政策清晰度和教师指导的重要性，为教育领域中AI的研究增添了新的认识。研究结果表明，虽然学生准备好将AI作为人教补充来接受，但谨慎的集成和透明的机构框架对于确保学生信心、信任和学习效果至关重要。 

---
# SafeGenBench: A Benchmark Framework for Security Vulnerability Detection in LLM-Generated Code 

**Title (ZH)**: SafeGenBench: 一个用于检测LLM生成代码中的安全漏洞的基准框架 

**Authors**: Xinghang Li, Jingzhe Ding, Chao Peng, Bing Zhao, Xiang Gao, Hongwan Gao, Xinchen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05692)  

**Abstract**: The code generation capabilities of large language models(LLMs) have emerged as a critical dimension in evaluating their overall performance. However, prior research has largely overlooked the security risks inherent in the generated code. In this work, we introduce \benchmark, a benchmark specifically designed to assess the security of LLM-generated code. The dataset encompasses a wide range of common software development scenarios and vulnerability types. Building upon this benchmark, we develop an automatic evaluation framework that leverages both static application security testing(SAST) and LLM-based judging to assess the presence of security vulnerabilities in model-generated code. Through the empirical evaluation of state-of-the-art LLMs on \benchmark, we reveal notable deficiencies in their ability to produce vulnerability-free code. Our findings highlight pressing challenges and offer actionable insights for future advancements in the secure code generation performance of LLMs. The data and code will be released soon. 

**Abstract (ZH)**: 大型语言模型(LLMs)的代码生成能力已成为评估其整体性能的关键维度。然而，先前的研究大多忽视了生成代码中固有的安全风险。在本文中，我们引入了benchmark，一个专门用于评估LLM生成代码安全性基准。该数据集涵盖了广泛常见的软件开发场景和漏洞类型。基于此基准，我们开发了一种自动评估框架，该框架结合了静态应用安全测试(SAST)和基于LLM的评估，以评估模型生成代码中是否存在安全漏洞。通过对benchmark上的先进LLM进行实证评估，我们揭示了它们在产生无漏洞代码方面存在的显著缺陷。我们的研究结果突显了紧迫的挑战，并为未来提升LLM安全代码生成性能提供了可操作的见解。数据和代码将于不久后发布。 

---
# Multi-Modal Multi-Task Federated Foundation Models for Next-Generation Extended Reality Systems: Towards Privacy-Preserving Distributed Intelligence in AR/VR/MR 

**Title (ZH)**: 面向AR/VR/MR的下一代扩展现实系统的大规模多模态多任务联邦基础模型：迈向隐私保护分布式智能 

**Authors**: Fardis Nadimi, Payam Abdisarabshali, Kasra Borazjani, Jacob Chakareski, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2506.05683)  

**Abstract**: Extended reality (XR) systems, which consist of virtual reality (VR), augmented reality (AR), and mixed reality (XR), offer a transformative interface for immersive, multi-modal, and embodied human-computer interaction. In this paper, we envision that multi-modal multi-task (M3T) federated foundation models (FedFMs) can offer transformative capabilities for XR systems through integrating the representational strength of M3T foundation models (FMs) with the privacy-preserving model training principles of federated learning (FL). We present a modular architecture for FedFMs, which entails different coordination paradigms for model training and aggregations. Central to our vision is the codification of XR challenges that affect the implementation of FedFMs under the SHIFT dimensions: (1) Sensor and modality diversity, (2) Hardware heterogeneity and system-level constraints, (3) Interactivity and embodied personalization, (4) Functional/task variability, and (5) Temporality and environmental variability. We illustrate the manifestation of these dimensions across a set of emerging and anticipated applications of XR systems. Finally, we propose evaluation metrics, dataset requirements, and design tradeoffs necessary for the development of resource-aware FedFMs in XR. This perspective aims to chart the technical and conceptual foundations for context-aware privacy-preserving intelligence in the next generation of XR systems. 

**Abstract (ZH)**: 扩展现实(XR)系统，包括虚拟现实(VR)、增强现实(AR)和混合现实(MXR)，为沉浸式、多模态和具身的人机交互提供了一个变革性接口。本文设想，通过将多模态多任务(M3T)联邦基础模型(FedFMs)的表示能力与联邦学习(FL)的隐私保护模型训练原则相结合，M3T基础模型可以为XR系统提供变革性的能力。我们提出了一种模块化架构，该架构涉及不同的协调范式用于模型训练和聚合。我们愿景的核心在于，将影响FedFMs实现的XR挑战编码到SHIFT维度下：(1) 传感器和模态多样性，(2) 硬件异质性和系统级约束，(3) 交互性和具身个性化，(4) 功能/任务多样性，以及(5) 时间性和环境多样性。我们展示了这些维度在一组新兴和预期的XR系统应用中的表现形式。最后，我们提出评估指标、数据集需求和设计权衡，以促进资源感知的FedFMs在XR中的发展。本视角旨在为下一代XR系统的上下文感知隐私保护智能奠定技术和概念基础。 

---
# Learning Design-Score Manifold to Guide Diffusion Models for Offline Optimization 

**Title (ZH)**: 学习设计-评分流形以指导离线优化的扩散模型 

**Authors**: Tailin Zhou, Zhilin Chen, Wenlong Lyu, Zhitang Chen, Danny H.K. Tsang, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05680)  

**Abstract**: Optimizing complex systems, from discovering therapeutic drugs to designing high-performance materials, remains a fundamental challenge across science and engineering, as the underlying rules are often unknown and costly to evaluate. Offline optimization aims to optimize designs for target scores using pre-collected datasets without system interaction. However, conventional approaches may fail beyond training data, predicting inaccurate scores and generating inferior designs. This paper introduces ManGO, a diffusion-based framework that learns the design-score manifold, capturing the design-score interdependencies holistically. Unlike existing methods that treat design and score spaces in isolation, ManGO unifies forward prediction and backward generation, attaining generalization beyond training data. Key to this is its derivative-free guidance for conditional generation, coupled with adaptive inference-time scaling that dynamically optimizes denoising paths. Extensive evaluations demonstrate that ManGO outperforms 24 single- and 10 multi-objective optimization methods across diverse domains, including synthetic tasks, robot control, material design, DNA sequence, and real-world engineering optimization. 

**Abstract (ZH)**: 基于扩散的ManGO框架：全面捕捉设计-评分相互依赖关系以优化复杂系统 

---
# Peer-Ranked Precision: Creating a Foundational Dataset for Fine-Tuning Vision Models from DataSeeds' Annotated Imagery 

**Title (ZH)**: 基于同伴排序精度：创建一个用于从DataSeeds标注图像fine-tune视觉模型的基础数据集 

**Authors**: Sajjad Abdoli, Freeman Lewin, Gediminas Vasiliauskas, Fabian Schonholz  

**Link**: [PDF](https://arxiv.org/pdf/2506.05673)  

**Abstract**: The development of modern Artificial Intelligence (AI) models, particularly diffusion-based models employed in computer vision and image generation tasks, is undergoing a paradigmatic shift in development methodologies. Traditionally dominated by a "Model Centric" approach, in which performance gains were primarily pursued through increasingly complex model architectures and hyperparameter optimization, the field is now recognizing a more nuanced "Data-Centric" approach. This emergent framework foregrounds the quality, structure, and relevance of training data as the principal driver of model performance. To operationalize this paradigm shift, we introduce the this http URL sample dataset (the "DSD"), initially comprised of approximately 10,610 high-quality human peer-ranked photography images accompanied by extensive multi-tier annotations. The DSD is a foundational computer vision dataset designed to usher in a new standard for commercial image datasets. Representing a small fraction of this http URL's 100 million-plus image catalog, the DSD provides a scalable foundation necessary for robust commercial and multimodal AI development. Through this in-depth exploratory analysis, we document the quantitative improvements generated by the DSD on specific models against known benchmarks and make the code and the trained models used in our evaluation publicly available. 

**Abstract (ZH)**: 现代人工 Intelligence (AI) 模型的发展，特别是用于计算机视觉和图像生成任务的扩散基于模型的发展，正在经历一种开发方法范式的转变。传统上，该领域主要受到以“模型为中心”方法的影响，通过构建越来越复杂的数据架构和超参数优化来追求性能提升，但现正开始认识到一种更精细的“数据为中心”方法。这种新兴框架强调训练数据的质量、结构和相关性是模型性能的主要驱动因素。为实现这一范式转变，我们介绍了这个 http://thisurl.com 样本数据集（“DSD”），初始包含约 10,610 张高质量的人类同行评分的摄影作品及其详尽的多级注释。DSD 是一个基础的计算机视觉数据集，旨在推动新的商用图像数据集标准。作为这个 http://thisurl.com 百万以上图像目录的小部分，DSD 提供了一个可扩展的基础，对于稳健的商用和多模态 AI 开发至关重要。通过这项深入的探索性分析，我们记录了 DSD 对特定模型在已知基准上的量化改进，并向公众提供了我们在评估中使用的代码和训练模型。 

---
# DriveAction: A Benchmark for Exploring Human-like Driving Decisions in VLA Models 

**Title (ZH)**: DriveAction: 一种探索VLA模型中人类-like 驾驶决策的标准基准 

**Authors**: Yuhan Hao, Zhengning Li, Lei Sun, Weilong Wang, Naixin Yi, Sheng Song, Caihong Qin, Mofan Zhou, Yifei Zhan, Peng Jia, Xianpeng Lang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05667)  

**Abstract**: Vision-Language-Action (VLA) models have advanced autonomous driving, but existing benchmarks still lack scenario diversity, reliable action-level annotation, and evaluation protocols aligned with human preferences. To address these limitations, we introduce DriveAction, the first action-driven benchmark specifically designed for VLA models, comprising 16,185 QA pairs generated from 2,610 driving scenarios. DriveAction leverages real-world driving data proactively collected by users of production-level autonomous vehicles to ensure broad and representative scenario coverage, offers high-level discrete action labels collected directly from users' actual driving operations, and implements an action-rooted tree-structured evaluation framework that explicitly links vision, language, and action tasks, supporting both comprehensive and task-specific assessment. Our experiments demonstrate that state-of-the-art vision-language models (VLMs) require both vision and language guidance for accurate action prediction: on average, accuracy drops by 3.3% without vision input, by 4.1% without language input, and by 8.0% without either. Our evaluation supports precise identification of model bottlenecks with robust and consistent results, thus providing new insights and a rigorous foundation for advancing human-like decisions in autonomous driving. 

**Abstract (ZH)**: 基于视觉-语言-行动的DriveAction基准：面向自主驾驶的行动驱动评测 

---
# TissUnet: Improved Extracranial Tissue and Cranium Segmentation for Children through Adulthood 

**Title (ZH)**: TissUnet: 用于从童年至成年的 Extracranial 组织和颅骨分割改进方法 

**Authors**: Markian Mandzak, Elvira Yang, Anna Zapaishchykova, Yu-Hui Chen, Lucas Heilbroner, John Zielke, Divyanshu Tak, Reza Mojahed-Yazdi, Francesca Romana Mussa, Zezhong Ye, Sridhar Vajapeyam, Viviana Benitez, Ralph Salloum, Susan N. Chi, Houman Sotoudeh, Jakob Seidlitz, Sabine Mueller, Hugo J.W.L. Aerts, Tina Y. Poussaint, Benjamin H. Kann  

**Link**: [PDF](https://arxiv.org/pdf/2506.05660)  

**Abstract**: Extracranial tissues visible on brain magnetic resonance imaging (MRI) may hold significant value for characterizing health conditions and clinical decision-making, yet they are rarely quantified. Current tools have not been widely validated, particularly in settings of developing brains or underlying pathology. We present TissUnet, a deep learning model that segments skull bone, subcutaneous fat, and muscle from routine three-dimensional T1-weighted MRI, with or without contrast enhancement. The model was trained on 155 paired MRI-computed tomography (CT) scans and validated across nine datasets covering a wide age range and including individuals with brain tumors. In comparison to AI-CT-derived labels from 37 MRI-CT pairs, TissUnet achieved a median Dice coefficient of 0.79 [IQR: 0.77-0.81] in a healthy adult cohort. In a second validation using expert manual annotations, median Dice was 0.83 [IQR: 0.83-0.84] in healthy individuals and 0.81 [IQR: 0.78-0.83] in tumor cases, outperforming previous state-of-the-art method. Acceptability testing resulted in an 89% acceptance rate after adjudication by a tie-breaker(N=108 MRIs), and TissUnet demonstrated excellent performance in the blinded comparative review (N=45 MRIs), including both healthy and tumor cases in pediatric populations. TissUnet enables fast, accurate, and reproducible segmentation of extracranial tissues, supporting large-scale studies on craniofacial morphology, treatment effects, and cardiometabolic risk using standard brain T1w MRI. 

**Abstract (ZH)**: Extracranial 组织在脑磁共振成像(MRI)中的可视化：对于表征健康状况和临床决策具有重要意义，但鲜有量化。目前的工具在发育脑或潜在病理情况下未广泛验证。我们提出了一种深度学习模型 TissUnet，可以从常规三维 T1 加权 MRI（有或无对比增强）中分割颅骨骨、皮下脂肪和肌肉。该模型使用 155 对 MRI-CT 扫描进行训练，并在九个涵盖广泛年龄范围且包括脑肿瘤患者的数据库中进行验证。与来自 37 对 MRI-CT 的 AI-CT 提取标签相比，在健康成人组中，TissUnet 的中位 Dice 系数为 0.79 [IQR: 0.77-0.81]。在使用专家手动注释进行的第二次验证中，在健康个体中，中位 Dice 为 0.83 [IQR: 0.83-0.84]，在肿瘤病例中为 0.81 [IQR: 0.78-0.83]，优于之前的最先进方法。接受性测试后，在决选裁定者的评估下，接受率为 89%（N=108），而在盲法比较审查中，TissUnet 在包括儿童患者在内的健康和肿瘤病例中表现出色。TissUnet 使 Extracranial 组织的快速、准确和可重复分割成为可能，支持使用标准脑 T1 加权 MRI 对颅面形态、治疗效果和心血管代谢风险进行大规模研究。 

---
# Bayesian Inference for Correlated Human Experts and Classifiers 

**Title (ZH)**: 关联的人类专家和分类器的贝叶斯推断 

**Authors**: Markelle Kelly, Alex Boyd, Sam Showalter, Mark Steyvers, Padhraic Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2506.05636)  

**Abstract**: Applications of machine learning often involve making predictions based on both model outputs and the opinions of human experts. In this context, we investigate the problem of querying experts for class label predictions, using as few human queries as possible, and leveraging the class probability estimates of pre-trained classifiers. We develop a general Bayesian framework for this problem, modeling expert correlation via a joint latent representation, enabling simulation-based inference about the utility of additional expert queries, as well as inference of posterior distributions over unobserved expert labels. We apply our approach to two real-world medical classification problems, as well as to CIFAR-10H and ImageNet-16H, demonstrating substantial reductions relative to baselines in the cost of querying human experts while maintaining high prediction accuracy. 

**Abstract (ZH)**: 机器学习的应用通常涉及基于模型输出和人类专家意见进行预测。在这一背景下，我们研究了使用最少的人类查询来查询专家进行类标签预测的问题，并利用预训练分类器的类概率估计。我们为该问题开发了一个通用的贝叶斯框架，通过联合潜在表示建模专家的相关性，从而基于模拟对额外专家查询的效用进行推理，并推断未观察到的专家标签的后验分布。我们将该方法应用于两个实际医疗分类问题以及CIFAR-10H和ImageNet-16H，相对于基线方法在查询人类专家的成本上实现了显著降低，同时保持了高预测准确性。 

---
# AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization 

**Title (ZH)**: AutoQD: 基于质量多样性优化的多样化行为自动发现 

**Authors**: Saeed Hedayatian, Stefanos Nikolaidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.05634)  

**Abstract**: Quality-Diversity (QD) algorithms have shown remarkable success in discovering diverse, high-performing solutions, but rely heavily on hand-crafted behavioral descriptors that constrain exploration to predefined notions of diversity. Leveraging the equivalence between policies and occupancy measures, we present a theoretically grounded approach to automatically generate behavioral descriptors by embedding the occupancy measures of policies in Markov Decision Processes. Our method, AutoQD, leverages random Fourier features to approximate the Maximum Mean Discrepancy (MMD) between policy occupancy measures, creating embeddings whose distances reflect meaningful behavioral differences. A low-dimensional projection of these embeddings that captures the most behaviorally significant dimensions is then used as behavioral descriptors for off-the-shelf QD methods. We prove that our embeddings converge to true MMD distances between occupancy measures as the number of sampled trajectories and embedding dimensions increase. Through experiments in multiple continuous control tasks we demonstrate AutoQD's ability in discovering diverse policies without predefined behavioral descriptors, presenting a well-motivated alternative to prior methods in unsupervised Reinforcement Learning and QD optimization. Our approach opens new possibilities for open-ended learning and automated behavior discovery in sequential decision making settings without requiring domain-specific knowledge. 

**Abstract (ZH)**: 基于自动生成行为描述符的质量多样性算法：理论上支持的方法 

---
# GP-MoLFormer-Sim: Test Time Molecular Optimization through Contextual Similarity Guidance 

**Title (ZH)**: GP-MoLFormer-Sim: 基于上下文相似性指导的测试时分子优化 

**Authors**: Jiri Navratil, Jarret Ross, Payel Das, Youssef Mroueh, Samuel C Hoffman, Vijil Chenthamarakshan, Brian Belgodere  

**Link**: [PDF](https://arxiv.org/pdf/2506.05628)  

**Abstract**: The ability to design molecules while preserving similarity to a target molecule and/or property is crucial for various applications in drug discovery, chemical design, and biology. We introduce in this paper an efficient training-free method for navigating and sampling from the molecular space with a generative Chemical Language Model (CLM), while using the molecular similarity to the target as a guide. Our method leverages the contextual representations learned from the CLM itself to estimate the molecular similarity, which is then used to adjust the autoregressive sampling strategy of the CLM. At each step of the decoding process, the method tracks the distance of the current generations from the target and updates the logits to encourage the preservation of similarity in generations. We implement the method using a recently proposed $\sim$47M parameter SMILES-based CLM, GP-MoLFormer, and therefore refer to the method as GP-MoLFormer-Sim, which enables a test-time update of the deep generative policy to reflect the contextual similarity to a set of guide molecules. The method is further integrated into a genetic algorithm (GA) and tested on a set of standard molecular optimization benchmarks involving property optimization, molecular rediscovery, and structure-based drug design. Results show that, GP-MoLFormer-Sim, combined with GA (GP-MoLFormer-Sim+GA) outperforms existing training-free baseline methods, when the oracle remains black-box. The findings in this work are a step forward in understanding and guiding the generative mechanisms of CLMs. 

**Abstract (ZH)**: 在保留目标分子和/or属性相似性的前提下设计分子的能力对于药物发现、化学设计和生物学等多个应用至关重要。本文介绍了一种基于生成化学语言模型（CLM）且无需训练的高效方法，用于导航和采样分子空间，并以分子目标相似性作为引导。该方法利用CLM本身学习到的上下文表示来估计分子相似性，并据此调整CLM的自回归采样策略。在解码过程的每一步中，该方法跟踪当前生成的分子与目标之间的距离，并更新logits以鼓励保持相似性。我们使用最近提出的一个约47M参数的基于SMILES的CLM，GP-MoLFormer，实现该方法，并将其命名为GP-MoLFormer-Sim，该方法可以在测试时更新深层生成策略以反映一组引导分子的上下文相似性。该方法进一步集成到遗传算法（GA）中，并在涉及属性优化、分子重新发现和结构基药物设计的一系列标准分子优化基准测试上进行测试。结果表明，GP-MoLFormer-Sim与GA结合（GP-MoLFormer-Sim+GA）在oracle保持黑盒的情况下优于现有无需训练的基本方法。本文的研究成果为理解和指导CLMs的生成机制迈进了一步。 

---
# Deployability-Centric Infrastructure-as-Code Generation: An LLM-based Iterative Framework 

**Title (ZH)**: 基于LLM的面向部署性的基础设施即代码生成迭代框架 

**Authors**: Tianyi Zhang, Shidong Pan, Zejun Zhang, Zhenchang Xing, Xiaoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.05623)  

**Abstract**: Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions, but current evaluation focuses on syntactic correctness while ignoring deployability, the fatal measure of IaC template utility. We address this gap through two contributions: (1) IaCGen, an LLM-based deployability-centric framework that uses iterative feedback mechanism to generate IaC templates, and (2) DPIaC-Eval, a deployability-centric IaC template benchmark consists of 153 real-world scenarios that can evaluate syntax, deployment, user intent, and security. Our evaluation reveals that state-of-the-art LLMs initially performed poorly, with Claude-3.5 and Claude-3.7 achieving only 30.2% and 26.8% deployment success on the first attempt respectively. However, IaCGen transforms this performance dramatically: all evaluated models reach over 90% passItr@25, with Claude-3.5 and Claude-3.7 achieving 98% success rate. Despite these improvements, critical challenges remain in user intent alignment (25.2% accuracy) and security compliance (8.4% pass rate), highlighting areas requiring continued research. Our work provides the first comprehensive assessment of deployability-centric IaC template generation and establishes a foundation for future research. 

**Abstract (ZH)**: 基于语言模型的基础设施即代码生成及其可部署性评估 

---
# LFA applied to CNNs: Efficient Singular Value Decomposition of Convolutional Mappings by Local Fourier Analysis 

**Title (ZH)**: LFA应用于CNNs：局部傅里叶分析下的卷积映射高效奇异值分解 

**Authors**: Antonia van Betteray, Matthias Rottmann, Karsten Kahl  

**Link**: [PDF](https://arxiv.org/pdf/2506.05617)  

**Abstract**: The singular values of convolutional mappings encode interesting spectral properties, which can be used, e.g., to improve generalization and robustness of convolutional neural networks as well as to facilitate model compression. However, the computation of singular values is typically very resource-intensive. The naive approach involves unrolling the convolutional mapping along the input and channel dimensions into a large and sparse two-dimensional matrix, making the exact calculation of all singular values infeasible due to hardware limitations. In particular, this is true for matrices that represent convolutional mappings with large inputs and a high number of channels. Existing efficient methods leverage the Fast Fourier transformation (FFT) to transform convolutional mappings into the frequency domain, enabling the computation of singular values for matrices representing convolutions with larger input and channel dimensions. For a constant number of channels in a given convolution, an FFT can compute N singular values in O(N log N) complexity. In this work, we propose an approach of complexity O(N) based on local Fourier analysis, which additionally exploits the shift invariance of convolutional operators. We provide a theoretical analysis of our algorithm's runtime and validate its efficiency through numerical experiments. Our results demonstrate that our proposed method is scalable and offers a practical solution to calculate the entire set of singular values - along with the corresponding singular vectors if needed - for high-dimensional convolutional mappings. 

**Abstract (ZH)**: 卷积映射的奇异值编码了有趣的谱性质，这些性质可以用于改进卷积神经网络的泛化能力和鲁棒性，以及促进模型压缩。然而，奇异值的计算通常非常耗费资源。朴素的方法是将卷积映射沿输入和通道维度展开成一个大的稀疏二维矩阵，由于硬件限制，这使得所有奇异值的精确计算变得不可行，尤其是对于具有大量输入和高通道数的卷积映射矩阵而言。现有的高效方法利用快速傅里叶变换（FFT）将卷积映射转换到频域中，从而能够计算表示大输入和通道尺寸卷积的矩阵的奇异值。对于给定卷积中通道数不变的情况，FFT可以在O(N log N)复杂度下计算N个奇异值。本文我们提出了一种基于局部傅里叶分析的复杂度为O(N)的方法，并且该方法还利用了卷积算子的移不变性。我们对算法的运行时间进行了理论分析，并通过数值实验验证了其效率。我们的结果表明，所提出的方法具有可扩展性，并提供了一种计算高维卷积映射全部奇异值（如果需要，还包括相应的奇异向量）的实用解决方案。 

---
# When Maximum Entropy Misleads Policy Optimization 

**Title (ZH)**: 当最大熵误导政策优化 

**Authors**: Ruipeng Zhang, Ya-Chien Chang, Sicun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05615)  

**Abstract**: The Maximum Entropy Reinforcement Learning (MaxEnt RL) framework is a leading approach for achieving efficient learning and robust performance across many RL tasks. However, MaxEnt methods have also been shown to struggle with performance-critical control problems in practice, where non-MaxEnt algorithms can successfully learn. In this work, we analyze how the trade-off between robustness and optimality affects the performance of MaxEnt algorithms in complex control tasks: while entropy maximization enhances exploration and robustness, it can also mislead policy optimization, leading to failure in tasks that require precise, low-entropy policies. Through experiments on a variety of control problems, we concretely demonstrate this misleading effect. Our analysis leads to better understanding of how to balance reward design and entropy maximization in challenging control problems. 

**Abstract (ZH)**: 最大熵强化学习框架（MaxEnt RL）是实现多种RL任务高效学习和稳健性能的领先方法。然而，MaxEnt方法在实践中也显示出在绩效关键控制问题上表现不佳，而非MaxEnt算法在这种情况下可以成功学习。在本工作中，我们分析了在复杂控制任务中稳健性和最优性之间的权衡如何影响MaxEnt算法的性能：虽然熵最大化增强了探索性和稳健性，但也可能误导策略优化，在需要精确、低熵策略的任务中导致失败。通过在一系列控制问题上的实验，我们具体展示了这种误导性效应。我们的分析有助于更好地理解在具有挑战性的控制问题中如何平衡奖励设计和熵最大化。 

---
# Scenarios in Computing Research: A Systematic Review of the Use of Scenario Methods for Exploring the Future of Computing Technologies in Society 

**Title (ZH)**: 计算研究中的情景分析：探索计算技术在社会未来发展中的情景方法系统综述 

**Authors**: Julia Barnett, Kimon Kieslich, Jasmine Sinchai, Nicholas Diakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.05605)  

**Abstract**: Scenario building is an established method to anticipate the future of emerging technologies. Its primary goal is to use narratives to map future trajectories of technology development and sociotechnical adoption. Following this process, risks and benefits can be identified early on, and strategies can be developed that strive for desirable futures. In recent years, computer science has adopted this method and applied it to various technologies, including Artificial Intelligence (AI). Because computing technologies play such an important role in shaping modern societies, it is worth exploring how scenarios are being used as an anticipatory tool in the field -- and what possible traditional uses of scenarios are not yet covered but have the potential to enrich the field. We address this gap by conducting a systematic literature review on the use of scenario building methods in computer science over the last decade (n = 59). We guide the review along two main questions. First, we aim to uncover how scenarios are used in computing literature, focusing especially on the rationale for why scenarios are used. Second, in following the potential of scenario building to enhance inclusivity in research, we dive deeper into the participatory element of the existing scenario building literature in computer science. 

**Abstract (ZH)**: 情景构建是一种成熟的方法，用于预测新兴技术的未来。其主要目标是通过叙事来映射技术发展的未来轨迹和社会技术采纳路径。在这一过程中，可以早期识别风险和利益，并开发策略以追求理想未来的愿景。近年来，计算机科学已采纳了这一方法，并将其应用于各种技术，包括人工智能（AI）。由于计算技术在塑造现代社会中发挥着如此重要的作用，值得探讨情景是如何作为一种预测工具在该领域被使用，以及可能尚未覆盖但有潜力丰富该领域的传统情景使用方法。我们通过系统文献综述，探讨了过去十年（n=59）计算机科学中情景构建方法的应用，围绕两个主要问题进行指导。首先，我们旨在揭示情景在计算文献中的使用方式，特别是探讨为什么使用情景的原因。其次，沿着情景构建增强研究包容性的潜力，深入探讨计算机科学中现有情景构建文献中的参与元素。 

---
# SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs 

**Title (ZH)**: SynthesizeMe！基于人设引导提示的个性化奖励模型合成 

**Authors**: Michael J Ryan, Omar Shaikh, Aditri Bhagirath, Daniel Frees, William Held, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05598)  

**Abstract**: Recent calls for pluralistic alignment of Large Language Models (LLMs) encourage adapting models to diverse user preferences. However, most prior work on personalized reward models heavily rely on additional identity information, such as demographic details or a predefined set of preference categories. To this end, we introduce SynthesizeMe, an approach to inducing synthetic user personas from user interactions for personalized reward modeling. SynthesizeMe first generates and verifies reasoning to explain user preferences, then induces synthetic user personas from that reasoning, and finally filters to informative prior user interactions in order to build personalized prompts for a particular user. We show that using SynthesizeMe induced prompts improves personalized LLM-as-a-judge accuracy by 4.4% on Chatbot Arena. Combining SynthesizeMe derived prompts with a reward model achieves top performance on PersonalRewardBench: a new curation of user-stratified interactions with chatbots collected from 854 users of Chatbot Arena and PRISM. 

**Abstract (ZH)**: Recent呼吁多样性对齐的大语言模型鼓励模型适应多元用户偏好。然而，大多数关于个性化奖励模型的先前工作严重依赖额外的身份信息，如人口统计细节或预定义的偏好类别。为此，我们提出SynthesizeMe，一种从用户互动中诱导合成用户人设的个性化奖励建模方法。SynthesizeMe首先生成并验证解释用户偏好的原因，然后从中诱导合成用户人设，并最终筛选有益的用户先前互动以构建特定用户的个性化提示。实验结果显示，使用SynthesizeMe诱导的提示在Chatbot Arena中提高了个性化LLM作为评判者的准确率4.4%。将SynthesizeMe推断出的提示与奖励模型结合，在一个新收集的854名Chatbot Arena和PRISM用户的分层交互数据集PersonalRewardBench上达到了最佳性能。 

---
# Zero-shot protein stability prediction by inverse folding models: a free energy interpretation 

**Title (ZH)**: 基于逆折叠模型的零样本蛋白质稳定性预测：自由能解释 

**Authors**: Jes Frellsen, Maher M. Kassem, Tone Bengtsen, Lars Olsen, Kresten Lindorff-Larsen, Jesper Ferkinghoff-Borg, Wouter Boomsma  

**Link**: [PDF](https://arxiv.org/pdf/2506.05596)  

**Abstract**: Inverse folding models have proven to be highly effective zero-shot predictors of protein stability. Despite this success, the link between the amino acid preferences of an inverse folding model and the free-energy considerations underlying thermodynamic stability remains incompletely understood. A better understanding would be of interest not only from a theoretical perspective, but also potentially provide the basis for stronger zero-shot stability prediction. In this paper, we take steps to clarify the free-energy foundations of inverse folding models. Our derivation reveals the standard practice of likelihood ratios as a simplistic approximation and suggests several paths towards better estimates of the relative stability. We empirically assess these approaches and demonstrate that considerable gains in zero-shot performance can be achieved with fairly simple means. 

**Abstract (ZH)**: 逆折叠模型已被证明是蛋白质稳定性零样本预测的高度有效工具。尽管取得了这一成功，逆折叠模型的氨基酸偏好与其热力学稳定性的自由能考虑之间的联系仍不完全清楚。更好地理解这一点不仅在理论上富有意义，而且可能为基础更强的零样本稳定性预测提供基础。在本文中，我们采取步骤澄清逆折叠模型的自由能基础。我们的推导揭示了likelihood ratios的常规做法是一种简单的近似，并建议了几条更好地估计相对稳定性的途径。我们通过实证评估了这些方法，并证明了相当大的零样本性能提升可以通过相对简单的手段实现。 

---
# Improving Neural Diarization through Speaker Attribute Attractors and Local Dependency Modeling 

**Title (ZH)**: 通过说话人口语属性吸引子和局部依赖建模改进神经语音分离 

**Authors**: David Palzer, Matthew Maciejewski, Eric Fosler-Lussier  

**Link**: [PDF](https://arxiv.org/pdf/2506.05593)  

**Abstract**: In recent years, end-to-end approaches have made notable progress in addressing the challenge of speaker diarization, which involves segmenting and identifying speakers in multi-talker recordings. One such approach, Encoder-Decoder Attractors (EDA), has been proposed to handle variable speaker counts as well as better guide the network during training. In this study, we extend the attractor paradigm by moving beyond direct speaker modeling and instead focus on representing more detailed `speaker attributes' through a multi-stage process of intermediate representations. Additionally, we enhance the architecture by replacing transformers with conformers, a convolution-augmented transformer, to model local dependencies. Experiments demonstrate improved diarization performance on the CALLHOME dataset. 

**Abstract (ZH)**: 近年来，端到端方法在应对说话人分割挑战方面取得了显著进展，说话人分割涉及多说话人录音中的说话人分割和识别。一种这样的方法，编码器-解码器吸引子（EDA），已被提出以处理变化的说话人数，并且在训练过程中更好地引导网络。在本研究中，我们通过超越直接说话人建模，而是通过多阶段中间表示来表示更详细的“说话人属性”，扩展了吸引子范式。此外，我们通过使用卷积增强的变压器（Conformer）代替变压器来增强架构，以建模局部依赖性。实验结果显示，在CALLHOME数据集上的说话人分割性能有所提高。 

---
# CoFrNets: Interpretable Neural Architecture Inspired by Continued Fractions 

**Title (ZH)**: CoFrNets：基于连续分数的可解释神经架构 

**Authors**: Isha Puri, Amit Dhurandhar, Tejaswini Pedapati, Kartikeyan Shanmugam, Dennis Wei, Kush R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2506.05586)  

**Abstract**: In recent years there has been a considerable amount of research on local post hoc explanations for neural networks. However, work on building interpretable neural architectures has been relatively sparse. In this paper, we present a novel neural architecture, CoFrNet, inspired by the form of continued fractions which are known to have many attractive properties in number theory, such as fast convergence of approximations to real numbers. We show that CoFrNets can be efficiently trained as well as interpreted leveraging their particular functional form. Moreover, we prove that such architectures are universal approximators based on a proof strategy that is different than the typical strategy used to prove universal approximation results for neural networks based on infinite width (or depth), which is likely to be of independent interest. We experiment on nonlinear synthetic functions and are able to accurately model as well as estimate feature attributions and even higher order terms in some cases, which is a testament to the representational power as well as interpretability of such architectures. To further showcase the power of CoFrNets, we experiment on seven real datasets spanning tabular, text and image modalities, and show that they are either comparable or significantly better than other interpretable models and multilayer perceptrons, sometimes approaching the accuracies of state-of-the-art models. 

**Abstract (ZH)**: 近年来，关于神经网络的局部后验解释研究取得了 considerable 的进展，但构建可解释神经架构的工作相对较少。在本文中，我们提出了一种受连分数形式启发的新颖神经架构 CoFrNet。连分数在数论中因其快速逼近实数的许多吸引性质而闻名。我们展示了可以通过利用其特殊的函数形式高效地训练和解释 CoFrNets。此外，我们基于不同于通常用于证明神经网络无限宽度（或深度）泛化能力的策略，证明了此类架构是通用逼近器，这一证明策略可能具有独立的研究兴趣。我们在非线性合成函数上的实验表明，这些架构能够准确建模并估计特征贡献，在某些情况下甚至能够估计高阶项，这一结果证明了此类架构的表示能力和可解释性。为了进一步展示 CoFrNets 的能力，我们在七个涵盖表格、文本和图像模态的真实数据集上进行了实验，并展示了它们要么与可解释模型和多层感知机相当，要么显著更好，有时甚至接近最佳模型的准确率。 

---
# Conformal Prediction Adaptive to Unknown Subpopulation Shifts 

**Title (ZH)**: 自适应未知亚人群体转移的齐性预测 

**Authors**: Nien-Shao Wang, Duygu Nur Yaldiz, Yavuz Faruk Bakman, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.05583)  

**Abstract**: Conformal prediction is widely used to equip black-box machine learning models with uncertainty quantification enjoying formal coverage guarantees. However, these guarantees typically break down in the presence of distribution shifts, where the data distribution at test time differs from the training (or calibration-time) distribution. In this work, we address subpopulation shifts, where the test environment exhibits an unknown and differing mixture of subpopulations compared to the calibration data. We propose new methods that provably adapt conformal prediction to such shifts, ensuring valid coverage without requiring explicit knowledge of subpopulation structure. Our algorithms scale to high-dimensional settings and perform effectively in realistic machine learning tasks. Extensive experiments on vision (with vision transformers) and language (with large language models) benchmarks demonstrate that our methods reliably maintain coverage and controls risk in scenarios where standard conformal prediction fails. 

**Abstract (ZH)**: 基于一致预测的方法在数据分布转移情况下的子人群转移适应研究 

---
# Combating Misinformation in the Arab World: Challenges & Opportunities 

**Title (ZH)**: 阿拉伯世界打击虚假信息：挑战与机遇 

**Authors**: Azza Abouzied, Firoj Alam, Raian Ali, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.05582)  

**Abstract**: Misinformation and disinformation pose significant risks globally, with the Arab region facing unique vulnerabilities due to geopolitical instabilities, linguistic diversity, and cultural nuances. We explore these challenges through the key facets of combating misinformation: detection, tracking, mitigation and community-engagement. We shed light on how connecting with grass-roots fact-checking organizations, understanding cultural norms, promoting social correction, and creating strong collaborative information networks can create opportunities for a more resilient information ecosystem in the Arab world. 

**Abstract (ZH)**: misinformation和disinformation在全球范围内构成重大风险，阿拉伯地区由于地缘政治不稳定性、语言多样性和文化差异而面临独特的脆弱性。我们通过打击 misinformation的关键方面——检测、跟踪、缓解和社区参与，探讨这些挑战。我们强调与基层事实核查组织建立联系、理解文化规范、推动社会矫正以及创建强有力的信息协作网络的重要性，以在阿拉伯世界构建更具韧性的信息生态系统。 

---
# Collaborative Learning in Agentic Systems: A Collective AI is Greater Than the Sum of Its Parts 

**Title (ZH)**: 代理系统中的协同学习：集体人工智能大于各部分之和 

**Authors**: Saptarshi Nath, Christos Peridis, Eseoghene Benjamin, Xinran Liu, Soheil Kolouri, Peter Kinnell, Zexin Li, Cong Liu, Shirin Dora, Andrea Soltoggio  

**Link**: [PDF](https://arxiv.org/pdf/2506.05577)  

**Abstract**: Agentic AI has gained significant interest as a research paradigm focused on autonomy, self-directed learning, and long-term reliability of decision making. Real-world agentic systems operate in decentralized settings on a large set of tasks or data distributions with constraints such as limited bandwidth, asynchronous execution, and the absence of a centralized model or even common objectives. We posit that exploiting previously learned skills, task similarities, and communication capabilities in a collective of agentic AI are challenging but essential elements to enabling scalability, open-endedness, and beneficial collaborative learning dynamics. In this paper, we introduce Modular Sharing and Composition in Collective Learning (MOSAIC), an agentic algorithm that allows multiple agents to independently solve different tasks while also identifying, sharing, and reusing useful machine-learned knowledge, without coordination, synchronization, or centralized control. MOSAIC combines three mechanisms: (1) modular policy composition via neural network masks, (2) cosine similarity estimation using Wasserstein embeddings for knowledge selection, and (3) asynchronous communication and policy integration. Results on a set of RL benchmarks show that MOSAIC has a greater sample efficiency than isolated learners, i.e., it learns significantly faster, and in some cases, finds solutions to tasks that cannot be solved by isolated learners. The collaborative learning and sharing dynamics are also observed to result in the emergence of ideal curricula of tasks, from easy to hard. These findings support the case for collaborative learning in agentic systems to achieve better and continuously evolving performance both at the individual and collective levels. 

**Abstract (ZH)**: 基于代理的AI：一种关注自主性、自我导向学习和决策长期可靠性的研究范式 

---
# Ravan: Multi-Head Low-Rank Adaptation for Federated Fine-Tuning 

**Title (ZH)**: Ravan: 多头低秩适应的联邦微调 

**Authors**: Arian Raje, Baris Askin, Divyansh Jhunjhunwala, Gauri Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05568)  

**Abstract**: Large language models (LLMs) have not yet effectively leveraged the vast amounts of edge-device data, and federated learning (FL) offers a promising paradigm to collaboratively fine-tune LLMs without transferring private edge data to the cloud. To operate within the computation and communication constraints of edge devices, recent literature on federated fine-tuning of LLMs proposes the use of low-rank adaptation (LoRA) and similar parameter-efficient methods. However, LoRA-based methods suffer from accuracy degradation in FL settings, primarily because of data and computational heterogeneity across clients. We propose \textsc{Ravan}, an adaptive multi-head LoRA method that balances parameter efficiency and model expressivity by reparameterizing the weight updates as the sum of multiple LoRA heads $s_i\textbf{B}_i\textbf{H}_i\textbf{A}_i$ in which only the core matrices $\textbf{H}_i$ and their lightweight scaling factors $s_i$ are trained. These trainable scaling factors let the optimization focus on the most useful heads, recovering a higher-rank approximation of the full update without increasing the number of communicated parameters since clients upload $s_i\textbf{H}_i$ directly. Experiments on vision and language benchmarks show that \textsc{Ravan} improves test accuracy by 2-8\% over prior parameter-efficient baselines, making it a robust and scalable solution for federated fine-tuning of LLMs. 

**Abstract (ZH)**: Ravan：一种自适应多头LoRA方法用于联邦微调大语言模型 

---
# ScaleRTL: Scaling LLMs with Reasoning Data and Test-Time Compute for Accurate RTL Code Generation 

**Title (ZH)**: ScaleRTL：使用推理数据和测试时计算进行准确RTL代码生成 

**Authors**: Chenhui Deng, Yun-Da Tsai, Guan-Ting Liu, Zhongzhi Yu, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.05566)  

**Abstract**: Recent advances in large language models (LLMs) have enabled near-human performance on software coding benchmarks, but their effectiveness in RTL code generation remains limited due to the scarcity of high-quality training data. While prior efforts have fine-tuned LLMs for RTL tasks, they do not fundamentally overcome the data bottleneck and lack support for test-time scaling due to their non-reasoning nature. In this work, we introduce ScaleRTL, the first reasoning LLM for RTL coding that scales up both high-quality reasoning data and test-time compute. Specifically, we curate a diverse set of long chain-of-thought reasoning traces averaging 56K tokens each, resulting in a dataset of 3.5B tokens that captures rich RTL knowledge. Fine-tuning a general-purpose reasoning model on this corpus yields ScaleRTL that is capable of deep RTL reasoning. Subsequently, we further enhance the performance of ScaleRTL through a novel test-time scaling strategy that extends the reasoning process via iteratively reflecting on and self-correcting previous reasoning steps. Experimental results show that ScaleRTL achieves state-of-the-art performance on VerilogEval and RTLLM, outperforming 18 competitive baselines by up to 18.4% on VerilogEval and 12.7% on RTLLM. 

**Abstract (ZH)**: recent 进展在大规模语言模型（LLMs）已在软件编码基准上实现了接近人类的表现，但在 RTL 代码生成方面的有效性受限于高质量训练数据的稀缺性。尽管先前的努力已经在 RTL 任务上微调了 LLMs，但它们仍然无法从根本上克服数据瓶颈，并且由于其非推理特性，缺乏测试时缩放的支持。在本文中，我们介绍了 ScaleRTL，这是首个能够扩展高质量推理数据和测试时计算的用于 RTL 编码的推理 LLM。具体地，我们精心编纂了一组长度平均为 56K 令牌的多样化长推理链记录，最终形成一个包含 3.5B 令牌的数据集，捕捉了丰富的 RTL 知识。在该语料库上微调一个通用推理模型产生了具备深度 RTL 推理能力的 ScaleRTL。随后，我们通过一种新颖的测试时缩放策略进一步增强了 ScaleRTL 的性能，该策略通过迭代地反思和自我纠正先前的推理步骤来扩展推理过程。实验结果表明，ScaleRTL 在 VerilogEval 和 RTLLM 上实现了最先进的性能，在 VerilogEval 上优于 18 个竞争基线最高达 18.4%，在 RTLLM 上优于 12.7%。 

---
# Applying Informer for Option Pricing: A Transformer-Based Approach 

**Title (ZH)**: 基于Transformer的Informer在期权定价中的应用 

**Authors**: Feliks Bańka, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2506.05565)  

**Abstract**: Accurate option pricing is essential for effective trading and risk management in financial markets, yet it remains challenging due to market volatility and the limitations of traditional models like Black-Scholes. In this paper, we investigate the application of the Informer neural network for option pricing, leveraging its ability to capture long-term dependencies and dynamically adjust to market fluctuations. This research contributes to the field of financial forecasting by introducing Informer's efficient architecture to enhance prediction accuracy and provide a more adaptable and resilient framework compared to existing methods. Our results demonstrate that Informer outperforms traditional approaches in option pricing, advancing the capabilities of data-driven financial forecasting in this domain. 

**Abstract (ZH)**: 基于Informer神经网络的期权定价研究：提高预测准确性和适应市场波动的能力 

---
# MORSE-500: A Programmatically Controllable Video Benchmark to Stress-Test Multimodal Reasoning 

**Title (ZH)**: MORSE-500：一个可编程控制的视频基准测试，用于压力测试多模态推理能力 

**Authors**: Zikui Cai, Andrew Wang, Anirudh Satheesh, Ankit Nakhawa, Hyunwoo Jae, Keenan Powell, Minghui Liu, Neel Jay, Sungbin Oh, Xiyao Wang, Yongyuan Liang, Tom Goldstein, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05523)  

**Abstract**: Despite rapid advances in vision-language models (VLMs), current benchmarks for multimodal reasoning fall short in three key dimensions. First, they overwhelmingly rely on static images, failing to capture the temporal complexity of real-world environments. Second, they narrowly focus on mathematical problem-solving, neglecting the broader spectrum of reasoning skills -- including abstract, physical, planning, spatial, and temporal capabilities -- required for robust multimodal intelligence. Third, many benchmarks quickly saturate, offering limited headroom for diagnosing failure modes or measuring continued progress. We introduce MORSE-500 (Multimodal Reasoning Stress-test Environment), a video benchmark composed of 500 fully scripted clips with embedded questions spanning six complementary reasoning categories. Each instance is programmatically generated using deterministic Python scripts (via Manim, Matplotlib, MoviePy), generative video models, and curated real footage. This script-driven design allows fine-grained control over visual complexity, distractor density, and temporal dynamics -- enabling difficulty to be scaled systematically as models improve. Unlike static benchmarks that become obsolete once saturated, MORSE-500 is built to evolve: its controllable generation pipeline supports the creation of arbitrarily challenging new instances, making it ideally suited for stress-testing next-generation models. Initial experiments with state-of-the-art systems -- including various Gemini 2.5 Pro and OpenAI o3 which represent the strongest available at the time, alongside strong open-source models -- reveal substantial performance gaps across all categories, with particularly large deficits in abstract and planning tasks. We release the full dataset, generation scripts, and evaluation harness to support transparent, reproducible, and forward-looking multimodal reasoning research. 

**Abstract (ZH)**: 尽管视觉语言模型取得了快速进步，当前的多模态推理基准在三个关键维度上仍存在不足。首先，它们主要依赖静态图像，未能捕捉到现实世界环境的时间复杂性。其次，它们狭隘地集中在数学问题解决上，忽视了实现稳健的多模态智能所需更广泛的推理技能范围，包括抽象、物理、规划、空间和时间能力。第三，许多基准很快达到饱和，提供有限的空间来诊断失败模式或衡量持续进步。我们引入了MORSE-500（多模态推理压力测试环境），这是一个由500个完全脚本化的视频片段组成的数据集，涵盖六个互补的推理类别，并嵌入了问题。每一实例都是通过确定性的Python脚本（使用Manim、Matplotlib、MoviePy）、生成性视频模型和精选的真实素材程序化生成的。这种脚本驱动的设计允许对视觉复杂性、干扰密度和时间动态性的精确控制——从而使难度可以随着模型的进步而系统地调整。不同于一旦饱和就会变得过时的静态基准，MORSE-500 是为了进化而构建的：其可控的生成管道支持创建任意具有挑战性的新实例，使其非常适合测试下一代模型的压力。初始实验证明，最新的顶级系统——包括各种Gemini 2.5 Pro和OpenAI o3以及强大的开源模型——在所有类别中均表现出显著的性能差距，尤其是在抽象和规划任务中的差距尤为显著。我们发布了完整的数据集、生成脚本和评估框架，以支持透明、可重复和前瞻性的多模态推理研究。 

---
# Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for Fallen Robots 

**Title (ZH)**: 学习恢复：基于轮腿协调的跌倒机器人动态奖励塑形 

**Authors**: Boyuan Deng, Luca Rossini, Jin Wang, Weijie Wang, Nikolaos Tsagarakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.05516)  

**Abstract**: Adaptive recovery from fall incidents are essential skills for the practical deployment of wheeled-legged robots, which uniquely combine the agility of legs with the speed of wheels for rapid recovery. However, traditional methods relying on preplanned recovery motions, simplified dynamics or sparse rewards often fail to produce robust recovery policies. This paper presents a learning-based framework integrating Episode-based Dynamic Reward Shaping and curriculum learning, which dynamically balances exploration of diverse recovery maneuvers with precise posture refinement. An asymmetric actor-critic architecture accelerates training by leveraging privileged information in simulation, while noise-injected observations enhance robustness against uncertainties. We further demonstrate that synergistic wheel-leg coordination reduces joint torque consumption by 15.8% and 26.2% and improves stabilization through energy transfer mechanisms. Extensive evaluations on two distinct quadruped platforms achieve recovery success rates up to 99.1% and 97.8% without platform-specific tuning. The supplementary material is available at this https URL 

**Abstract (ZH)**: 基于episode动态奖励塑形和课程学习的适应性跌倒恢复对于轮腿机器人实用部署是 essential 技能，这种机器人独特地结合了腿的灵活性和轮的速度以实现快速恢复。然而，依赖于预先规划的恢复动作、简化的动力学模型或稀疏奖励的传统方法往往无法产生 robust 的恢复策略。本文介绍了一种基于学习的框架，结合了基于episode的动态奖励塑形和课程学习，动态平衡多样恢复动作的探索与精确姿态优化。不对称的Actor-Critic架构通过利用仿真中的优先信息加速训练，同时注入噪声的观测增强其对不确定性的稳健性。进一步的实验表明，协同的轮-腿协调减少了关节扭矩消耗15.8%和26.2%，并通过能量转移机制提高了稳定性能。在两个不同的四足平台上的广泛评估表明，恢复成功率分别达到了99.1%和97.8%，无需针对特定平台进行调整。更多补充材料请参见此链接：this https URL。 

---
# Winner-takes-all for Multivariate Probabilistic Time Series Forecasting 

**Title (ZH)**: 赢家全拿：多变量概率时间序列预测 

**Authors**: Adrien Cortés, Rémi Rehm, Victor Letzelter  

**Link**: [PDF](https://arxiv.org/pdf/2506.05515)  

**Abstract**: We introduce TimeMCL, a method leveraging the Multiple Choice Learning (MCL) paradigm to forecast multiple plausible time series futures. Our approach employs a neural network with multiple heads and utilizes the Winner-Takes-All (WTA) loss to promote diversity among predictions. MCL has recently gained attention due to its simplicity and ability to address ill-posed and ambiguous tasks. We propose an adaptation of this framework for time-series forecasting, presenting it as an efficient method to predict diverse futures, which we relate to its implicit quantization objective. We provide insights into our approach using synthetic data and evaluate it on real-world time series, demonstrating its promising performance at a light computational cost. 

**Abstract (ZH)**: TimeMCL：一种利用Multiple Choice Learning范式进行多可能时间序列未来预测的方法 

---
# Beyond the Buzz: A Pragmatic Take on Inference Disaggregation 

**Title (ZH)**: 超越热度：推断拆解的务实视角 

**Authors**: Tiyasa Mitra, Ritika Borkar, Nidhi Bhatia, Ramon Matas, Shivam Raj, Dheevatsa Mudigere, Ritchie Zhao, Maximilian Golub, Arpan Dutta, Sailaja Madduri, Dharmesh Jani, Brian Pharris, Bita Darvish Rouhani  

**Link**: [PDF](https://arxiv.org/pdf/2506.05508)  

**Abstract**: As inference scales to multi-node deployments, disaggregation - splitting inference into distinct phases - offers a promising path to improving the throughput-interactivity Pareto frontier. Despite growing enthusiasm and a surge of open-source efforts, practical deployment of disaggregated serving remains limited due to the complexity of the optimization search space and system-level coordination. In this paper, we present the first systematic study of disaggregated inference at scale, evaluating hundreds of thousands of design points across diverse workloads and hardware configurations. We find that disaggregation is most effective for prefill-heavy traffic patterns and larger models. Our results highlight the critical role of dynamic rate matching and elastic scaling in achieving Pareto-optimal performance. Our findings offer actionable insights for efficient disaggregated deployments to navigate the trade-off between system throughput and interactivity. 

**Abstract (ZH)**: 随着推理部署扩展到多节点，拆分推理过程为不同的阶段有望改善吞吐量-交互性帕累托前沿。尽管出现了越来越多的热情和开源努力，但由于优化搜索空间复杂性和系统级协调的复杂性，拆分推理的实际部署仍然有限。在本文中，我们首次系统地研究了拆分推理的扩展问题，评估了不同类型工作负载和硬件配置下的数十万个设计点。我们发现，拆分推理对预填充型流量模式和大型模型最有效。我们的结果强调了动态速率匹配和弹性扩展在实现帕累托最优性能中的关键作用。我们的发现为在系统吞吐量和交互性之间权衡高效拆分推理部署提供了可操作的洞察。 

---
# StealthInk: A Multi-bit and Stealthy Watermark for Large Language Models 

**Title (ZH)**: StealthInk: 一种针对大规模语言模型的多比特隐形水印 

**Authors**: Ya Jiang, Chuxiong Wu, Massieh Kordi Boroujeny, Brian Mark, Kai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05502)  

**Abstract**: Watermarking for large language models (LLMs) offers a promising approach to identifying AI-generated text. Existing approaches, however, either compromise the distribution of original generated text by LLMs or are limited to embedding zero-bit information that only allows for watermark detection but ignores identification. We present StealthInk, a stealthy multi-bit watermarking scheme that preserves the original text distribution while enabling the embedding of provenance data, such as userID, TimeStamp, and modelID, within LLM-generated text. This enhances fast traceability without requiring access to the language model's API or prompts. We derive a lower bound on the number of tokens necessary for watermark detection at a fixed equal error rate, which provides insights on how to enhance the capacity. Comprehensive empirical evaluations across diverse tasks highlight the stealthiness, detectability, and resilience of StealthInk, establishing it as an effective solution for LLM watermarking applications. 

**Abstract (ZH)**: 面向大规模语言模型的隐蔽多比特水印方案：保持原始文本分布的同时嵌入来源数据 

---
# Conformal Prediction Beyond the Seen: A Missing Mass Perspective for Uncertainty Quantification in Generative Models 

**Title (ZH)**: 超越已见：生成模型中不确定性量化的一种缺失质量视角 

**Authors**: Sima Noorani, Shayan Kiyani, George Pappas, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2506.05497)  

**Abstract**: Uncertainty quantification (UQ) is essential for safe deployment of generative AI models such as large language models (LLMs), especially in high stakes applications. Conformal prediction (CP) offers a principled uncertainty quantification framework, but classical methods focus on regression and classification, relying on geometric distances or softmax scores: tools that presuppose structured outputs. We depart from this paradigm by studying CP in a query only setting, where prediction sets must be constructed solely from finite queries to a black box generative model, introducing a new trade off between coverage, test time query budget, and informativeness. We introduce Conformal Prediction with Query Oracle (CPQ), a framework characterizing the optimal interplay between these objectives. Our finite sample algorithm is built on two core principles: one governs the optimal query policy, and the other defines the optimal mapping from queried samples to prediction sets. Remarkably, both are rooted in the classical missing mass problem in statistics. Specifically, the optimal query policy depends on the rate of decay, or the derivative, of the missing mass, for which we develop a novel estimator. Meanwhile, the optimal mapping hinges on the missing mass itself, which we estimate using Good Turing estimators. We then turn our focus to implementing our method for language models, where outputs are vast, variable, and often under specified. Fine grained experiments on three real world open ended tasks and two LLMs, show CPQ applicability to any black box LLM and highlight: (1) individual contribution of each principle to CPQ performance, and (2) CPQ ability to yield significantly more informative prediction sets than existing conformal methods for language uncertainty quantification. 

**Abstract (ZH)**: 生成人工智能模型如大型语言模型（LLMs）的安全部署需要不确定性量化（UQ），尤其是在高风险应用中。形式预测（CP）提供了一种原则性的不确定性量化框架，但经典方法主要关注回归和分类，依赖几何距离或softmax分数：这些工具假设了结构化的输出。我们摒弃这一范式，在仅查询设置中研究CP，其中预测集必须仅从对黑盒生成模型的有限查询中构建，引入了覆盖、测试时间查询预算和信息量之间的新权衡。我们引入了基于查询 oracle 的形式预测（CPQ）框架，该框架刻画了这些目标之间最佳交互。我们的有限样本算法基于两个核心原则：一个管理最优查询策略，另一个定义从查询样本到预测集的最优映射。令人惊讶的是，两者都根植于统计中的经典缺失质量问题。具体而言，最优查询策略取决于缺失质量衰减率，或其导数，为此我们开发了一种新的估计器。同时，最优映射依赖于缺失质量本身，我们使用Good Turing估计器进行估计。然后，我们将注意力转向在语言模型中实施该方法，其中输出量大、变化且往往不够明确。针对三个实际开放任务和两个LLMs的精细实验表明，CPQ在任何黑盒LLMs中的适用性，并突出显示：（1）每个原则对CPQ性能的独立贡献，以及（2）CPQ在语言不确定性量化中比现有形式预测方法生成显著更具信息量的预测集的能力。 

---
# Sentiment Analysis in Learning Management Systems Understanding Student Feedback at Scale 

**Title (ZH)**: 学习管理系统中的情感分析：大规模理解学生反馈 

**Authors**: Mohammed Almutairi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05490)  

**Abstract**: During the wake of the Covid-19 pandemic, the educational paradigm has experienced a major change from in person learning traditional to online platforms. The change of learning convention has impacted the teacher-student especially in non-verbal communication. The absent of non-verbal communication has led to a reliance on verbal feedback which diminished the efficacy of the educational experience. This paper explores the integration of sentiment analysis into learning management systems (LMS) to bridge the student-teacher's gap by offering an alternative approach to interpreting student feedback beyond its verbal context. The research involves data preparation, feature selection, and the development of a deep neural network model encompassing word embedding, LSTM, and attention mechanisms. This model is compared against a logistic regression baseline to evaluate its efficacy in understanding student feedback. The study aims to bridge the communication gap between instructors and students in online learning environments, offering insights into the emotional context of student feedback and ultimately improving the quality of online education. 

**Abstract (ZH)**: COVID-19 pandemic期间教育范式的转变及其对师生非言语沟通的影响：通过情感分析集成到学习管理系统中的解决方案 

---
# Zeroth-Order Optimization Finds Flat Minima 

**Title (ZH)**: 零阶优化找到平坦的最小值 

**Authors**: Liang Zhang, Bingcong Li, Kiran Koshy Thekumparampil, Sewoong Oh, Michael Muehlebach, Niao He  

**Link**: [PDF](https://arxiv.org/pdf/2506.05454)  

**Abstract**: Zeroth-order methods are extensively used in machine learning applications where gradients are infeasible or expensive to compute, such as black-box attacks, reinforcement learning, and language model fine-tuning. Existing optimization theory focuses on convergence to an arbitrary stationary point, but less is known on the implicit regularization that provides a fine-grained characterization on which particular solutions are finally reached. We show that zeroth-order optimization with the standard two-point estimator favors solutions with small trace of Hessian, which is widely used in previous work to distinguish between sharp and flat minima. We further provide convergence rates of zeroth-order optimization to approximate flat minima for convex and sufficiently smooth functions, where flat minima are defined as the minimizers that achieve the smallest trace of Hessian among all optimal solutions. Experiments on binary classification tasks with convex losses and language model fine-tuning support our theoretical findings. 

**Abstract (ZH)**: 零阶方法在计算梯度不可行或昂贵的应用场景下（如黑盒攻击、强化学习和语言模型微调）得到了广泛使用。现有的优化理论主要关注于收敛到任意稳定点，而对于提供对最终达到的具体解的细致表征的隐式正则化知之甚少。我们证明，使用标准的两点 estimator 的零阶优化倾向于选择Hessian矩阵迹较小的解，这一特性在先前的工作中用于区分尖锐极小值和平坦极小值。进一步地，我们为凸函数和充分光滑函数提供了零阶优化收敛到近似平坦极小值的速率，其中平坦极小值被定义为所有最优解中Hessian矩阵迹最小的解。实验结果在带凸损失的二分类任务和语言模型微调中支持了我们的理论发现。 

---
# MLLM-CL: Continual Learning for Multimodal Large Language Models 

**Title (ZH)**: MLLM-CL: 多模态大型语言模型的持续学习 

**Authors**: Hongbo Zhao, Fei Zhu, Rundong Wang, Gaofeng Meng, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05453)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) excel in vision-language understanding but face challenges in adapting to dynamic real-world scenarios that require continuous integration of new knowledge and skills. While continual learning (CL) offers a potential solution, existing benchmarks and methods suffer from critical limitations. In this paper, we introduce MLLM-CL, a novel benchmark encompassing domain and ability continual learning, where the former focuses on independently and identically distributed (IID) evaluation across evolving mainstream domains, whereas the latter evaluates on non-IID scenarios with emerging model ability. Methodologically, we propose preventing catastrophic interference through parameter isolation, along with an MLLM-based routing mechanism. Extensive experiments demonstrate that our approach can integrate domain-specific knowledge and functional abilities with minimal forgetting, significantly outperforming existing methods. 

**Abstract (ZH)**: 最近的多模态大型语言模型在视觉-语言理解方面表现出色，但在应对需要持续集成新知识和技能的动态现实场景方面面临挑战。尽管持续学习提供了一种潜在的解决方案，但现有基准和方法存在重大局限。本文介绍了一种新的基准MLLM-CL，涵盖了领域和能力的持续学习，前者关注独立同分布（IID）评估在不断演变的主要领域的独立性，后者则评估在新兴模型能力非IID场景下的表现。方法上，我们提出了通过参数隔离防止灾难性干扰，并设计了一种基于多模态大型语言模型的路由机制。广泛实验证明，我们的方法能够以最小的遗忘整合领域特定知识和功能能力，显著优于现有方法。 

---
# Interpretation Meets Safety: A Survey on Interpretation Methods and Tools for Improving LLM Safety 

**Title (ZH)**: 解读与安全并重：改进大型语言模型安全性的解释方法与工具综述 

**Authors**: Seongmin Lee, Aeree Cho, Grace C. Kim, ShengYun Peng, Mansi Phute, Duen Horng Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.05451)  

**Abstract**: As large language models (LLMs) see wider real-world use, understanding and mitigating their unsafe behaviors is critical. Interpretation techniques can reveal causes of unsafe outputs and guide safety, but such connections with safety are often overlooked in prior surveys. We present the first survey that bridges this gap, introducing a unified framework that connects safety-focused interpretation methods, the safety enhancements they inform, and the tools that operationalize them. Our novel taxonomy, organized by LLM workflow stages, summarizes nearly 70 works at their intersections. We conclude with open challenges and future directions. This timely survey helps researchers and practitioners navigate key advancements for safer, more interpretable LLMs. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在更广泛的实际应用中被采用，理解和缓解其不安全行为至关重要。解释技术可以揭示不安全输出的原因并指导安全措施，但在之前的综述中，这种与安全的联系往往被忽视。我们首次提出了填补这一空白的综述，介绍了一个统一框架，将安全导向的解释方法、它们所指导的安全增强措施以及实施这些措施的工具联系起来。我们按照LLM工作流程阶段组织的创新分类法总结了近70项相关工作。最后，我们提出了开放性挑战和未来方向。这项及时的综述有助于研究人员和实践者导航更安全、更具可解释性的LLM的关键进展。 

---
# Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning 

**Title (ZH)**: 语言模型规模规律背后的训练动力学：损失减速与零和学习 

**Authors**: Andrei Mircea, Supriyo Chakraborty, Nima Chitsazan, Irina Rish, Ekaterina Lobacheva  

**Link**: [PDF](https://arxiv.org/pdf/2506.05447)  

**Abstract**: This work aims to understand how scaling improves language models, specifically in terms of training dynamics. We find that language models undergo loss deceleration early in training; an abrupt slowdown in the rate of loss improvement, resulting in piecewise linear behaviour of the loss curve in log-log space. Scaling up the model mitigates this transition by (1) decreasing the loss at which deceleration occurs, and (2) improving the log-log rate of loss improvement after deceleration. We attribute loss deceleration to a type of degenerate training dynamics we term zero-sum learning (ZSL). In ZSL, per-example gradients become systematically opposed, leading to destructive interference in per-example changes in loss. As a result, improving loss on one subset of examples degrades it on another, bottlenecking overall progress. Loss deceleration and ZSL provide new insights into the training dynamics underlying language model scaling laws, and could potentially be targeted directly to improve language models independent of scale. We make our code and artefacts available at: this https URL 

**Abstract (ZH)**: 本项工作旨在理解扩增如何改善语言模型，特别是在训练动力学方面的表现。我们发现语言模型在训练初期会出现损失减速现象；损失改进速率的突然减缓，在对数-对数空间中导致损失曲线呈现分段线性行为。通过扩增模型，这种过渡得以缓解，具体表现为：（1）损失减速发生的损失值降低，（2）在损失减速后，损失改进的对数-对数速率得到提升。我们将损失减速归因于一种我们称之为零和学习（ZSL）的退化训练动态。在ZSL中，每个样本的梯度变得系统性地对立，导致损失的样本变化出现破坏性干涉。因此，在一个样本子集上改进损失会损害另一个子集上的损失，从而阻碍整体进展。损失减速和ZSL为语言模型缩放定律背后的训练动力学提供了新的见解，并有可能直接针对这些现象来独立于规模地改进语言模型。我们已在以下链接提供了我们的代码和 artefacts: this https URL。 

---
# Sentinel: SOTA model to protect against prompt injections 

**Title (ZH)**: 哨兵：当前最佳模型，用于抵御提示注入攻击 

**Authors**: Dror Ivry, Oran Nahum  

**Link**: [PDF](https://arxiv.org/pdf/2506.05446)  

**Abstract**: Large Language Models (LLMs) are increasingly powerful but remain vulnerable to prompt injection attacks, where malicious inputs cause the model to deviate from its intended instructions. This paper introduces Sentinel, a novel detection model, qualifire/prompt-injection-sentinel, based on the \answerdotai/ModernBERT-large architecture. By leveraging ModernBERT's advanced features and fine-tuning on an extensive and diverse dataset comprising a few open-source and private collections, Sentinel achieves state-of-the-art performance. This dataset amalgamates varied attack types, from role-playing and instruction hijacking to attempts to generate biased content, alongside a broad spectrum of benign instructions, with private datasets specifically targeting nuanced error correction and real-world misclassifications. On a comprehensive, unseen internal test set, Sentinel demonstrates an average accuracy of 0.987 and an F1-score of 0.980. Furthermore, when evaluated on public benchmarks, it consistently outperforms strong baselines like protectai/deberta-v3-base-prompt-injection-v2. This work details Sentinel's architecture, its meticulous dataset curation, its training methodology, and a thorough evaluation, highlighting its superior detection capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益强大但仍易受提示注入攻击的影响，恶意输入会导致模型偏离其预期指令。本文引入了基于\answerdotai/ModernBERT-large架构的Sentinel，一种新颖的检测模型，qualifire/prompt-injection-sentinel。通过利用ModernBERT的高级特性并在涵盖多种开源和私有数据集合的广泛且多样化的数据集上进行微调，Sentinel实现了最先进的性能。该数据集整合了各种攻击类型，包括角色扮演、指令劫持以及生成偏见内容的尝试，同时还包括广泛的良性指令，其中包含针对细微错误修正和现实世界误分类的私有数据集。在全面的未见过的内部测试集上，Sentinel的平均准确率为0.987，F1分为0.980。此外，在公开基准测试中，Sentinel一致地优于强大的基线模型，如protectai/deberta-v3-base-prompt-injection-v2。本文详细介绍了Sentinel的架构、精心的数据集编纂、训练方法以及全面评估，突显了其卓越的检测能力。 

---
# Causal Policy Learning in Reinforcement Learning: Backdoor-Adjusted Soft Actor-Critic 

**Title (ZH)**: 因果策略学习在强化学习中的调整后软 actor-critic 方法 

**Authors**: Thanh Vinh Vo, Young Lee, Haozhe Ma, Chien Lu, Tze-Yun Leong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05445)  

**Abstract**: Hidden confounders that influence both states and actions can bias policy learning in reinforcement learning (RL), leading to suboptimal or non-generalizable behavior. Most RL algorithms ignore this issue, learning policies from observational trajectories based solely on statistical associations rather than causal effects. We propose DoSAC (Do-Calculus Soft Actor-Critic with Backdoor Adjustment), a principled extension of the SAC algorithm that corrects for hidden confounding via causal intervention estimation. DoSAC estimates the interventional policy $\pi(a | \mathrm{do}(s))$ using the backdoor criterion, without requiring access to true confounders or causal labels. To achieve this, we introduce a learnable Backdoor Reconstructor that infers pseudo-past variables (previous state and action) from the current state to enable backdoor adjustment from observational data. This module is integrated into a soft actor-critic framework to compute both the interventional policy and its entropy. Empirical results on continuous control benchmarks show that DoSAC outperforms baselines under confounded settings, with improved robustness, generalization, and policy reliability. 

**Abstract (ZH)**: 隐变量影响状态和行动均会导致强化学习中的策略学习产生偏差，进而导致次优或非泛化的行为。大多数强化学习算法忽视这一问题，仅基于统计关联而非因果效应从观测轨迹中学习策略。我们提出了DoSAC（因果反事实软actor-critic算法，基于后门调整），这是一种利用因果干预估计纠正隐变量混杂效应的SAC算法的原理性扩展。DoSAC通过后门准则估计干预策略$\pi(a | \mathrm{do}(s))$，无需访问真实混杂变量或因果标签。为此，我们引入了一个可学习的后门重构器，从当前状态推断伪历史变量（上一状态和行动），以从观测数据中实现后门调整。该模块被整合到软actor-critic框架中，用于计算干预策略及其熵。在连续控制基准上的实验结果显示，DoSAC在混杂环境下优于基线方法，具有更好的鲁棒性、泛化能力和策略可靠性。 

---
# UniPTMs: The First Unified Multi-type PTM Site Prediction Model via Master-Slave Architecture-Based Multi-Stage Fusion Strategy and Hierarchical Contrastive Loss 

**Title (ZH)**: UniPTMs：基于主从架构多阶段融合策略和层次对比损失的首个统一多类型PTM位点预测模型 

**Authors**: Yiyu Lin, Yan Wang, You Zhou, Xinye Ni, Jiahui Wu, Sen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05443)  

**Abstract**: As a core mechanism of epigenetic regulation in eukaryotes, protein post-translational modifications (PTMs) require precise prediction to decipher dynamic life activity networks. To address the limitations of existing deep learning models in cross-modal feature fusion, domain generalization, and architectural optimization, this study proposes UniPTMs: the first unified framework for multi-type PTM prediction. The framework innovatively establishes a "Master-Slave" dual-path collaborative architecture: The master path dynamically integrates high-dimensional representations of protein sequences, structures, and evolutionary information through a Bidirectional Gated Cross-Attention (BGCA) module, while the slave path optimizes feature discrepancies and recalibration between structural and traditional features using a Low-Dimensional Fusion Network (LDFN). Complemented by a Multi-scale Adaptive convolutional Pyramid (MACP) for capturing local feature patterns and a Bidirectional Hierarchical Gated Fusion Network (BHGFN) enabling multi-level feature integration across paths, the framework employs a Hierarchical Dynamic Weighting Fusion (HDWF) mechanism to intelligently aggregate multimodal features. Enhanced by a novel Hierarchical Contrastive loss function for feature consistency optimization, UniPTMs demonstrates significant performance improvements (3.2%-11.4% MCC and 4.2%-14.3% AP increases) over state-of-the-art models across five modification types and transcends the Single-Type Prediction Paradigm. To strike a balance between model complexity and performance, we have also developed a lightweight variant named UniPTMs-mini. 

**Abstract (ZH)**: 统一蛋白多类型PTM预测框架：UniPTMs 

---
# Structured Labeling Enables Faster Vision-Language Models for End-to-End Autonomous Driving 

**Title (ZH)**: 结构化标签使端到端自主驾驶中的视觉语言模型加速训练成为可能 

**Authors**: Hao Jiang, Chuan Hu, Yukang Shi, Yuan He, Ke Wang, Xi Zhang, Zhipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05442)  

**Abstract**: Vision-Language Models (VLMs) offer a promising approach to end-to-end autonomous driving due to their human-like reasoning capabilities. However, troublesome gaps remains between current VLMs and real-world autonomous driving applications. One major limitation is that existing datasets with loosely formatted language descriptions are not machine-friendly and may introduce redundancy. Additionally, high computational cost and massive scale of VLMs hinder the inference speed and real-world deployment. To bridge the gap, this paper introduces a structured and concise benchmark dataset, NuScenes-S, which is derived from the NuScenes dataset and contains machine-friendly structured representations. Moreover, we present FastDrive, a compact VLM baseline with 0.9B parameters. In contrast to existing VLMs with over 7B parameters and unstructured language processing(e.g., LLaVA-1.5), FastDrive understands structured and concise descriptions and generates machine-friendly driving decisions with high efficiency. Extensive experiments show that FastDrive achieves competitive performance on structured dataset, with approximately 20% accuracy improvement on decision-making tasks, while surpassing massive parameter baseline in inference speed with over 10x speedup. Additionally, ablation studies further focus on the impact of scene annotations (e.g., weather, time of day) on decision-making tasks, demonstrating their importance on decision-making tasks in autonomous driving. 

**Abstract (ZH)**: Vision-Language模型（VLMs）提供了端到端自动驾驶的一种有前景的方法，由于其类似人类的推理能力。然而，当前的VLMs与实际自动驾驶应用之间仍存在显著差距。一个主要限制是现有的、松散格式的语言描述数据集不便于机器处理，并可能引入冗余。此外，高计算成本和大规模的VLMs也阻碍了推理速度和实际部署。为解决这些问题，本文介绍了一个结构化且简洁的基准数据集NuScenes-S，该数据集源自NuScenes数据集，并包含便于机器处理的结构化表示。同时，我们呈现了一个紧凑的VLM基线FastDrive，其参数量仅为0.9B。与现有的超过7B参数和非结构化语言处理的VLMs（如LLaVA-1.5）相比，FastDrive能够理解结构化和简洁的描述，并以高效率生成便于机器处理的驾驶决策。 extensive实验表明，FastDrive在结构化数据集上实现了竞争力的表现，决策任务准确率提高了约20%，同时在推理速度上超过大规模参数基线，快了超过10倍。此外，消融研究进一步关注场景标注（如天气、时间段）对决策任务的影响，证明了它们在自动驾驶决策任务中的重要性。 

---
# BYO-Eval: Build Your Own Dataset for Fine-Grained Visual Assessment of Multimodal Language Models 

**Title (ZH)**: BYO-Eval: 自建数据集以实现多模态语言模型细粒度视觉评估 

**Authors**: Ludovic Arnould, Salim Khazem, Hugues Ali Mehenni  

**Link**: [PDF](https://arxiv.org/pdf/2506.05440)  

**Abstract**: Visual Language Models (VLMs) are now sufficiently advanced to support a broad range of applications, including answering complex visual questions, and are increasingly expected to interact with images in varied ways. To evaluate them, current benchmarks often focus on specific domains (e.g., reading charts), constructing datasets of annotated real images paired with pre-defined Multiple Choice Questions (MCQs) to report aggregate accuracy scores. However, such benchmarks entail high annotation costs, risk information leakage, and do not clarify whether failures stem from limitations in visual perception, reasoning, or general knowledge. We propose a new evaluation methodology, inspired by ophthalmologic diagnostics, leveraging procedural generation of synthetic images to obtain control over visual attributes and precisely reveal perception failures in VLMs. Specifically, we build collections of images with gradually more challenging variations in the content of interest (e.g., number of objects in a counting task) while holding other visual parameters constant. This diagnostic allows systematic stress testing and fine-grained failure analysis, shifting the focus from coarse benchmarking toward targeted and interpretable assessment of VLM capabilities. Our code is available at this https URL. 

**Abstract (ZH)**: 视觉语言模型（VLMs）现在足够先进，可以支持一系列应用，包括回答复杂的视觉问题，并且越来越被期待以各种方式与图像互动。为了评估它们，当前的基准测试通常集中在特定领域（例如，阅读图表），通过构建带有预定义多项选择题（MCQs）的标注真实图像数据集来报告总体准确率分数。然而，这种基准测试涉及高昂的标注成本，存在信息泄漏的风险，并不能明确区分失败是源于视觉感知、推理还是通用知识的局限性。我们提出了一种新的评估方法，受眼科诊断的启发，利用程序生成合成图像以控制视觉属性并精确揭示VLMs的感知失败。具体而言，我们构建了具有不同挑战性内容变化的图集（例如，计数任务中的对象数量），同时保持其他视觉参数不变。这种诊断方法可进行系统性的压力测试和精细的失败分析，将重点从粗略的基准测试转向针对和可解释性强的评估VLM能力。我们的代码可在以下链接获取：this https URL。 

---
# LLMs Can Compensate for Deficiencies in Visual Representations 

**Title (ZH)**: LLMs可以在视觉表示的不足之处进行补偿。 

**Authors**: Sho Takishita, Jay Gala, Abdelrahman Mohamed, Kentaro Inui, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2506.05439)  

**Abstract**: Many vision-language models (VLMs) that prove very effective at a range of multimodal task, build on CLIP-based vision encoders, which are known to have various limitations. We investigate the hypothesis that the strong language backbone in VLMs compensates for possibly weak visual features by contextualizing or enriching them. Using three CLIP-based VLMs, we perform controlled self-attention ablations on a carefully designed probing task. Our findings show that despite known limitations, CLIP visual representations offer ready-to-read semantic information to the language decoder. However, in scenarios of reduced contextualization in the visual representations, the language decoder can largely compensate for the deficiency and recover performance. This suggests a dynamic division of labor in VLMs and motivates future architectures that offload more visual processing to the language decoder. 

**Abstract (ZH)**: 许多证明在多种多模态任务中非常有效的vision-language模型（VLMs）基于CLIP的视觉编码器，但这些编码器已知存在各种局限。我们研究了VLMs中的强大语言骨干能够通过上下文化或丰富视觉特征来补偿可能较弱的视觉特征这一假设。使用三种基于CLIP的VLMs，我们在精心设计的探针任务上进行了受控的自注意力消融实验。我们的研究结果表明，尽管CLIP的视觉表示存在已知的局限性，但它们仍为语言解码器提供了易于读取的语义信息。然而，在视觉表示中上下文化减弱的情况下，语言解码器可以大量补偿不足并恢复性能。这表明VLMs中存在动态的工作分工，并激发了未来将更多视觉处理卸载到语言解码器的架构设计。 

---
# An Unsupervised Framework for Dynamic Health Indicator Construction and Its Application in Rolling Bearing Prognostics 

**Title (ZH)**: 一种动态健康指标构建的无监督框架及其在滚动轴承预测中的应用 

**Authors**: Tongda Sun, Chen Yin, Huailiang Zheng, Yining Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05438)  

**Abstract**: Health indicator (HI) plays a key role in degradation assessment and prognostics of rolling bearings. Although various HI construction methods have been investigated, most of them rely on expert knowledge for feature extraction and overlook capturing dynamic information hidden in sequential degradation processes, which limits the ability of the constructed HI for degradation trend representation and prognostics. To address these concerns, a novel dynamic HI that considers HI-level temporal dependence is constructed through an unsupervised framework. Specifically, a degradation feature learning module composed of a skip-connection-based autoencoder first maps raw signals to a representative degradation feature space (DFS) to automatically extract essential degradation features without the need for expert knowledge. Subsequently, in this DFS, a new HI-generating module embedded with an inner HI-prediction block is proposed for dynamic HI construction, where the temporal dependence between past and current HI states is guaranteed and modeled explicitly. On this basis, the dynamic HI captures the inherent dynamic contents of the degradation process, ensuring its effectiveness for degradation tendency modeling and future degradation prognostics. The experiment results on two bearing lifecycle datasets demonstrate that the proposed HI construction method outperforms comparison methods, and the constructed dynamic HI is superior for prognostic tasks. 

**Abstract (ZH)**: 基于动态感知的滚动轴承健康指标构建方法 

---
# A MARL-based Approach for Easing MAS Organization Engineering 

**Title (ZH)**: 基于MARL的方法用于简化MAS组织工程 

**Authors**: Julien Soulé, Jean-Paul Jamont, Michel Occello, Louis-Marie Traonouez, Paul Théron  

**Link**: [PDF](https://arxiv.org/pdf/2506.05437)  

**Abstract**: Multi-Agent Systems (MAS) have been successfully applied in industry for their ability to address complex, distributed problems, especially in IoT-based systems. Their efficiency in achieving given objectives and meeting design requirements is strongly dependent on the MAS organization during the engineering process of an application-specific MAS. To design a MAS that can achieve given goals, available methods rely on the designer's knowledge of the deployment environment. However, high complexity and low readability in some deployment environments make the application of these methods to be costly or raise safety concerns. In order to ease the MAS organization design regarding those concerns, we introduce an original Assisted MAS Organization Engineering Approach (AOMEA). AOMEA relies on combining a Multi-Agent Reinforcement Learning (MARL) process with an organizational model to suggest relevant organizational specifications to help in MAS engineering. 

**Abstract (ZH)**: 基于多Agent系统组织辅助工程的方法（AOMEA）：一种结合多Agent强化学习的过程 

---
# Event Classification of Accelerometer Data for Industrial Package Monitoring with Embedded Deep Learning 

**Title (ZH)**: 基于嵌入式深度学习的工业包装监测加速度计数据事件分类 

**Authors**: Manon Renault, Hamoud Younes, Hugo Tessier, Ronan Le Roy, Bastien Pasdeloup, Mathieu Léonardon  

**Link**: [PDF](https://arxiv.org/pdf/2506.05435)  

**Abstract**: Package monitoring is an important topic in industrial applications, with significant implications for operational efficiency and ecological sustainability. In this study, we propose an approach that employs an embedded system, placed on reusable packages, to detect their state (on a Forklift, in a Truck, or in an undetermined location). We aim to design a system with a lifespan of several years, corresponding to the lifespan of reusable packages. Our analysis demonstrates that maximizing device lifespan requires minimizing wake time. We propose a pipeline that includes data processing, training, and evaluation of the deep learning model designed for imbalanced, multiclass time series data collected from an embedded sensor. The method uses a one-dimensional Convolutional Neural Network architecture to classify accelerometer data from the IoT device. Before training, two data augmentation techniques are tested to solve the imbalance problem of the dataset: the Synthetic Minority Oversampling TEchnique and the ADAptive SYNthetic sampling approach. After training, compression techniques are implemented to have a small model size. On the considered twoclass problem, the methodology yields a precision of 94.54% for the first class and 95.83% for the second class, while compression techniques reduce the model size by a factor of four. The trained model is deployed on the IoT device, where it operates with a power consumption of 316 mW during inference. 

**Abstract (ZH)**: 嵌入式系统在可重复使用包装上的状态监测：一种长期可持续的工业应用方法 

---
# Efficient Robust Conformal Prediction via Lipschitz-Bounded Networks 

**Title (ZH)**: 基于Lipschitz有界网络的高效稳健拟合预测 

**Authors**: Thomas Massena, Léo andéol, Thibaut Boissin, Franck Mamalet, Corentin Friedrich, Mathieu Serrurier, Sébastien Gerchinovitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.05434)  

**Abstract**: Conformal Prediction (CP) has proven to be an effective post-hoc method for improving the trustworthiness of neural networks by providing prediction sets with finite-sample guarantees. However, under adversarial attacks, classical conformal guarantees do not hold anymore: this problem is addressed in the field of Robust Conformal Prediction. Several methods have been proposed to provide robust CP sets with guarantees under adversarial perturbations, but, for large scale problems, these sets are either too large or the methods are too computationally demanding to be deployed in real life scenarios. In this work, we propose a new method that leverages Lipschitz-bounded networks to precisely and efficiently estimate robust CP sets. When combined with a 1-Lipschitz robust network, we demonstrate that our lip-rcp method outperforms state-of-the-art results in both the size of the robust CP sets and computational efficiency in medium and large-scale scenarios such as ImageNet. Taking a different angle, we also study vanilla CP under attack, and derive new worst-case coverage bounds of vanilla CP sets, which are valid simultaneously for all adversarial attack levels. Our lip-rcp method makes this second approach as efficient as vanilla CP while also allowing robustness guarantees. 

**Abstract (ZH)**: 利用Lipschitz有界网络进行精确高效估计鲁棒同构预测集 

---
# Prefix Grouper: Efficient GRPO Training through Shared-Prefix Forward 

**Title (ZH)**: 前缀分组器：通过共享前缀前向传播实现高效的GRPO训练 

**Authors**: Zikang Liu, Tongtian Yue, Yepeng Tang, Longteng Guo, Junxian Cai, Qingbin Liu, Xi Chen, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05433)  

**Abstract**: Group Relative Policy Optimization (GRPO) enhances policy learning by computing gradients from relative comparisons among candidate outputs that share a common input prefix. Despite its effectiveness, GRPO introduces substantial computational overhead when processing long shared prefixes, which must be redundantly encoded for each group member. This inefficiency becomes a major scalability bottleneck in long-context learning scenarios. We propose Prefix Grouper, an efficient GRPO training algorithm that eliminates redundant prefix computation via a Shared-Prefix Forward strategy. In particular, by restructuring self-attention into two parts, our method enables the shared prefix to be encoded only once, while preserving full differentiability and compatibility with end-to-end training. We provide both theoretical and empirical evidence that Prefix Grouper is training-equivalent to standard GRPO: it yields identical forward outputs and backward gradients, ensuring that the optimization dynamics and final policy performance remain unchanged. Empirically, our experiments confirm that Prefix Grouper achieves consistent results while significantly reducing the computational cost of training, particularly in long-prefix scenarios. The proposed method is fully plug-and-play: it is compatible with existing GRPO-based architectures and can be seamlessly integrated into current training pipelines as a drop-in replacement, requiring no structural modifications and only minimal changes to input construction and attention computation. Prefix Grouper enables the use of larger group sizes under the same computational budget, thereby improving the scalability of GRPO to more complex tasks and larger models. Code is now available at this https URL 

**Abstract (ZH)**: Prefix Grouper: An Efficient Training Algorithm for Group Relative Policy Optimization 

---
# PCDVQ: Enhancing Vector Quantization for Large Language Models via Polar Coordinate Decoupling 

**Title (ZH)**: PCDVQ：通过极坐标解耦提升大规模语言模型的向量量化 

**Authors**: Yuxuan Yue, Zukang Xu, Zhihang Yuan, Dawei Yang, Jianglong Wu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.05432)  

**Abstract**: Large Language Models (LLMs) face significant challenges in edge deployment due to their massive parameter scale. Vector Quantization (VQ), a clustering-based quantization method, serves as a prevalent solution to this issue for its extremely low-bit (even at 2-bit) and considerable accuracy. Since a vector is a quantity in mathematics and physics that has both direction and magnitude, existing VQ works typically quantize them in a coupled manner. However, we find that direction exhibits significantly greater sensitivity to quantization compared to the magnitude. For instance, when separately clustering the directions and magnitudes of weight vectors in LLaMA-2-7B, the accuracy drop of zero-shot tasks are 46.5\% and 2.3\%, respectively. This gap even increases with the reduction of clustering centers. Further, Euclidean distance, a common metric to access vector similarities in current VQ works, places greater emphasis on reducing the magnitude error. This property is contrary to the above finding, unavoidably leading to larger quantization errors. To these ends, this paper proposes Polar Coordinate Decoupled Vector Quantization (PCDVQ), an effective and efficient VQ framework consisting of two key modules: 1) Polar Coordinate Decoupling (PCD), which transforms vectors into their polar coordinate representations and perform independent quantization of the direction and magnitude parameters.2) Distribution Aligned Codebook Construction (DACC), which optimizes the direction and magnitude codebooks in accordance with the source distribution. Experimental results show that PCDVQ outperforms baseline methods at 2-bit level by at least 1.5\% zero-shot accuracy, establishing a novel paradigm for accurate and highly compressed LLMs. 

**Abstract (ZH)**: 基于极坐标解耦的矢量量化（Polar Coordinate Decoupled Vector Quantization） 

---
# Robustness Evaluation for Video Models with Reinforcement Learning 

**Title (ZH)**: 基于强化学习的视频模型稳健性评估 

**Authors**: Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Antonio Guillen, Ricardo Luna Gutierrez, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05431)  

**Abstract**: Evaluating the robustness of Video classification models is very challenging, specifically when compared to image-based models. With their increased temporal dimension, there is a significant increase in complexity and computational cost. One of the key challenges is to keep the perturbations to a minimum to induce misclassification. In this work, we propose a multi-agent reinforcement learning approach (spatial and temporal) that cooperatively learns to identify the given video's sensitive spatial and temporal regions. The agents consider temporal coherence in generating fine perturbations, leading to a more effective and visually imperceptible attack. Our method outperforms the state-of-the-art solutions on the Lp metric and the average queries. Our method enables custom distortion types, making the robustness evaluation more relevant to the use case. We extensively evaluate 4 popular models for video action recognition on two popular datasets, HMDB-51 and UCF-101. 

**Abstract (ZH)**: 评估视频分类模型的鲁棒性非常具有挑战性，特别是在与基于图像的模型相比时。随着其增加的时间维度，复杂性和计算成本显著增加。其中一个关键挑战是将扰动保持在最低限度以诱导误分类。在本工作中，我们提出了一种时空多智能体强化学习方法，该方法协同学习以识别给定视频的敏感时空区域。智能体在生成细微扰动时考虑时间一致性，从而产生更有效且视觉上不可感知的攻击。我们的方法在Lp度量和平均查询上优于现有最佳解决方案。我们的方法支持自定义失真类型，使鲁棒性评估更符合实际应用。我们对HMDB-51和UCF-101两个流行数据集上的4种流行视频动作识别模型进行了广泛评估。 

---
# Explainer-guided Targeted Adversarial Attacks against Binary Code Similarity Detection Models 

**Title (ZH)**: 由解释器引导的目标导向对抗攻击针对二进制代码相似性检测模型 

**Authors**: Mingjie Chen, Tiancheng Zhu, Mingxue Zhang, Yiling He, Minghao Lin, Penghui Li, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.05430)  

**Abstract**: Binary code similarity detection (BCSD) serves as a fundamental technique for various software engineering tasks, e.g., vulnerability detection and classification. Attacks against such models have therefore drawn extensive attention, aiming at misleading the models to generate erroneous predictions. Prior works have explored various approaches to generating semantic-preserving variants, i.e., adversarial samples, to evaluate the robustness of the models against adversarial attacks. However, they have mainly relied on heuristic criteria or iterative greedy algorithms to locate salient code influencing the model output, failing to operate on a solid theoretical basis. Moreover, when processing programs with high complexities, such attacks tend to be time-consuming.
In this work, we propose a novel optimization for adversarial attacks against BCSD models. In particular, we aim to improve the attacks in a challenging scenario, where the attack goal is to limit the model predictions to a specific range, i.e., the targeted attacks. Our attack leverages the superior capability of black-box, model-agnostic explainers in interpreting the model decision boundaries, thereby pinpointing the critical code snippet to apply semantic-preserving perturbations. The evaluation results demonstrate that compared with the state-of-the-art attacks, the proposed attacks achieve higher attack success rate in almost all scenarios, while also improving the efficiency and transferability. Our real-world case studies on vulnerability detection and classification further demonstrate the security implications of our attacks, highlighting the urgent need to further enhance the robustness of existing BCSD models. 

**Abstract (ZH)**: 二进制代码相似性检测模型的对抗攻击优化：针对特定范围目标的新型方法 

---
# Coordinated Robustness Evaluation Framework for Vision-Language Models 

**Title (ZH)**: 视觉-语言模型协调稳健性评估框架 

**Authors**: Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Antonio Guillen, Ricardo Luna Gutierrez, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05429)  

**Abstract**: Vision-language models, which integrate computer vision and natural language processing capabilities, have demonstrated significant advancements in tasks such as image captioning and visual question and answering. However, similar to traditional models, they are susceptible to small perturbations, posing a challenge to their robustness, particularly in deployment scenarios. Evaluating the robustness of these models requires perturbations in both the vision and language modalities to learn their inter-modal dependencies. In this work, we train a generic surrogate model that can take both image and text as input and generate joint representation which is further used to generate adversarial perturbations for both the text and image modalities. This coordinated attack strategy is evaluated on the visual question and answering and visual reasoning datasets using various state-of-the-art vision-language models. Our results indicate that the proposed strategy outperforms other multi-modal attacks and single-modality attacks from the recent literature. Our results demonstrate their effectiveness in compromising the robustness of several state-of-the-art pre-trained multi-modal models such as instruct-BLIP, ViLT and others. 

**Abstract (ZH)**: 视觉语言模型综合了计算机视觉和自然语言处理能力，在图像字幕和视觉问答等任务上取得了显著进展。然而，类似传统模型，它们对小规模扰动敏感，这给其鲁棒性带来了挑战，尤其是在部署场景中。评估这些模型的鲁棒性需要同时在视觉和语言模态上施加扰动，以学习其跨模态依赖关系。在此工作中，我们训练了一个通用的替代模型，该模型可以接受图像和文本作为输入，并生成联合表示，进一步用于为文本和图像模态生成对抗性扰动。这种协调攻击策略在视觉问答和视觉推理数据集上使用了多种最先进的视觉语言模型进行了评估。我们的结果显示，所提出的策略在应对多模态攻击和近期文献中的单模态攻击方面表现更优。我们的结果表明，该策略在多种最新的预训练多模态模型如instruct-BLIP、ViLT等上削弱了其鲁棒性。 

---
# Diffusion with a Linguistic Compass: Steering the Generation of Clinically Plausible Future sMRI Representations for Early MCI Conversion Prediction 

**Title (ZH)**: 以语言为指南的扩散：引导临床合理未来sMRI表示以预测早期MCI转换 

**Authors**: Zhihao Tang, Chaozhuo Li, Litian Zhang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05428)  

**Abstract**: Early prediction of Mild Cognitive Impairment (MCI) conversion is hampered by a trade-off between immediacy--making fast predictions from a single baseline sMRI--and accuracy--leveraging longitudinal scans to capture disease progression. We propose MCI-Diff, a diffusion-based framework that synthesizes clinically plausible future sMRI representations directly from baseline data, achieving both real-time risk assessment and high predictive performance. First, a multi-task sequence reconstruction strategy trains a shared denoising network on interpolation and extrapolation tasks to handle irregular follow-up sampling and learn robust latent trajectories. Second, an LLM-driven "linguistic compass" is introduced for clinical plausibility sampling: generated feature candidates are quantized, tokenized, and scored by a fine-tuned language model conditioned on expected structural biomarkers, guiding autoregressive generation toward realistic disease patterns. Experiments on ADNI and AIBL cohorts show that MCI-Diff outperforms state-of-the-art baselines, improving early conversion accuracy by 5-12%. 

**Abstract (ZH)**: 基于扩散的MCI转换早期预测框架MCI-Diff：实现实时风险评估与高预测性能 

---
# MTPNet: Multi-Grained Target Perception for Unified Activity Cliff Prediction 

**Title (ZH)**: MTPNet：多层次目标感知统一活动悬崖预测 

**Authors**: Zishan Shu, Yufan Deng, Hongyu Zhang, Zhiwei Nie, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.05427)  

**Abstract**: Activity cliff prediction is a critical task in drug discovery and material design. Existing computational methods are limited to handling single binding targets, which restricts the applicability of these prediction models. In this paper, we present the Multi-Grained Target Perception network (MTPNet) to incorporate the prior knowledge of interactions between the molecules and their target proteins. Specifically, MTPNet is a unified framework for activity cliff prediction, which consists of two components: Macro-level Target Semantic (MTS) guidance and Micro-level Pocket Semantic (MPS) guidance. By this way, MTPNet dynamically optimizes molecular representations through multi-grained protein semantic conditions. To our knowledge, it is the first time to employ the receptor proteins as guiding information to effectively capture critical interaction details. Extensive experiments on 30 representative activity cliff datasets demonstrate that MTPNet significantly outperforms previous approaches, achieving an average RMSE improvement of 18.95% on top of several mainstream GNN architectures. Overall, MTPNet internalizes interaction patterns through conditional deep learning to achieve unified predictions of activity cliffs, helping to accelerate compound optimization and design. Codes are available at: this https URL. 

**Abstract (ZH)**: 多粒度目标感知网络在药物发现和材料设计中的活性断崖预测 

---
# Mixture-of-Experts Meets In-Context Reinforcement Learning 

**Title (ZH)**: Experts混合体遇上了基于文本的强化学习 

**Authors**: Wenhao Wu, Fuhong Liu, Haoru Li, Zican Hu, Daoyi Dong, Chunlin Chen, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05426)  

**Abstract**: In-context reinforcement learning (ICRL) has emerged as a promising paradigm for adapting RL agents to downstream tasks through prompt conditioning. However, two notable challenges remain in fully harnessing in-context learning within RL domains: the intrinsic multi-modality of the state-action-reward data and the diverse, heterogeneous nature of decision tasks. To tackle these challenges, we propose \textbf{T2MIR} (\textbf{T}oken- and \textbf{T}ask-wise \textbf{M}oE for \textbf{I}n-context \textbf{R}L), an innovative framework that introduces architectural advances of mixture-of-experts (MoE) into transformer-based decision models. T2MIR substitutes the feedforward layer with two parallel layers: a token-wise MoE that captures distinct semantics of input tokens across multiple modalities, and a task-wise MoE that routes diverse tasks to specialized experts for managing a broad task distribution with alleviated gradient conflicts. To enhance task-wise routing, we introduce a contrastive learning method that maximizes the mutual information between the task and its router representation, enabling more precise capture of task-relevant information. The outputs of two MoE components are concatenated and fed into the next layer. Comprehensive experiments show that T2MIR significantly facilitates in-context learning capacity and outperforms various types of baselines. We bring the potential and promise of MoE to ICRL, offering a simple and scalable architectural enhancement to advance ICRL one step closer toward achievements in language and vision communities. Our code is available at this https URL. 

**Abstract (ZH)**: 基于令牌和任务的MoE的上下文强化学习（T2MIR）：一种将MoE架构引入基于变换器的决策模型的创新框架 

---
# SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning 

**Title (ZH)**: SIV-Bench: 用于社会互动理解与推理的视频基准 

**Authors**: Fanqi Kong, Weiqin Zu, Xinyu Chen, Yaodong Yang, Song-Chun Zhu, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05425)  

**Abstract**: The rich and multifaceted nature of human social interaction, encompassing multimodal cues, unobservable relations and mental states, and dynamical behavior, presents a formidable challenge for artificial intelligence. To advance research in this area, we introduce SIV-Bench, a novel video benchmark for rigorously evaluating the capabilities of Multimodal Large Language Models (MLLMs) across Social Scene Understanding (SSU), Social State Reasoning (SSR), and Social Dynamics Prediction (SDP). SIV-Bench features 2,792 video clips and 8,792 meticulously generated question-answer pairs derived from a human-LLM collaborative pipeline. It is originally collected from TikTok and YouTube, covering a wide range of video genres, presentation styles, and linguistic and cultural backgrounds. It also includes a dedicated setup for analyzing the impact of different textual cues-original on-screen text, added dialogue, or no text. Our comprehensive experiments on leading MLLMs reveal that while models adeptly handle SSU, they significantly struggle with SSR and SDP, where Relation Inference (RI) is an acute bottleneck, as further examined in our analysis. Our study also confirms the critical role of transcribed dialogue in aiding comprehension of complex social interactions. By systematically identifying current MLLMs' strengths and limitations, SIV-Bench offers crucial insights to steer the development of more socially intelligent AI. The dataset and code are available at this https URL. 

**Abstract (ZH)**: 人类社会互动的丰富多样性，涵盖多模态线索、不可观察的关系和心理状态以及动态行为，给人工智能带来了巨大的挑战。为了推进这一领域的研究，我们引入了SIV-Bench，这是一个新的视频基准，用于严格评估多模态大型语言模型（MLLMs）在社会场景理解（SSU）、社会状态推理（SSR）和社会动力预测（SDP）方面的能力。SIV-Bench 包含 2,792 个视频片段和 8,792 个精心生成的问题-答案对，这些数据源自人-LLM 合作管道。这些数据最初是从 TikTok 和 YouTube 收集的，涵盖了广泛的视频类型、呈现风格以及语言和文化背景。此外，它还包括一种专门设置，用于分析不同文本线索（原始屏幕文本、添加对话或无文本）的影响。我们在全面实验中发现，虽然模型在社会场景理解方面表现良好，但在社会状态推理和社会动力预测方面却面临巨大挑战，其中关系推理（RI）是一个关键瓶颈，我们在分析中进一步进行了探讨。我们的研究还证实，转录的对话在理解复杂社会互动中起着关键作用。通过系统地识别当前 MLLMs 的优势和局限性，SIV-Bench 提供了重要的见解，以引导更加社交智能的 AI 的开发。该数据集和代码可通过此链接访问。 

---
# Dream to Generalize: Zero-Shot Model-Based Reinforcement Learning for Unseen Visual Distractions 

**Title (ZH)**: 梦中泛化：针对未见过的视觉干扰的零样本模型驱动 reinforcement 学习 

**Authors**: Jeongsoo Ha, Kyungsoo Kim, Yusung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05419)  

**Abstract**: Model-based reinforcement learning (MBRL) has been used to efficiently solve vision-based control tasks in highdimensional image observations. Although recent MBRL algorithms perform well in trained observations, they fail when faced with visual distractions in observations. These task-irrelevant distractions (e.g., clouds, shadows, and light) may be constantly present in real-world scenarios. In this study, we propose a novel self-supervised method, Dream to Generalize (Dr. G), for zero-shot MBRL. Dr. G trains its encoder and world model with dual contrastive learning which efficiently captures task-relevant features among multi-view data augmentations. We also introduce a recurrent state inverse dynamics model that helps the world model to better understand the temporal structure. The proposed methods can enhance the robustness of the world model against visual distractions. To evaluate the generalization performance, we first train Dr. G on simple backgrounds and then test it on complex natural video backgrounds in the DeepMind Control suite, and the randomizing environments in Robosuite. Dr. G yields a performance improvement of 117% and 14% over prior works, respectively. Our code is open-sourced and available at this https URL 

**Abstract (ZH)**: 基于模型的强化学习（MBRL）在高维图像观察下的视觉控制任务中已被用于高效求解。尽管近期的MBRL算法在训练观察中表现良好，但在面对观察中的视觉干扰时却会失效。这些与任务无关的干扰（如云、阴影和光线）在现实世界场景中可能会持续存在。在本研究中，我们提出了一种新颖的自监督方法Dr. G，用于零样本MBRL。Dr. G 使用双对比学习训练其编码器和世界模型，该方法能有效地在多视图数据增强中捕获任务相关特征。我们还引入了一个递归状态逆动力学模型，帮助世界模型更好地理解时间结构。所提出的方法可以提高世界模型对视觉干扰的鲁棒性。为了评估泛化性能，我们首先在简单背景上训练Dr. G，然后在DeepMind Control套件和Robosuite中的随机环境下的复杂自然视频背景上进行测试。与先前的工作相比，Dr. G 的性能分别提高了117%和14%。我们的代码已开源，可在以下链接获取。 

---
# Self-Predictive Dynamics for Generalization of Vision-based Reinforcement Learning 

**Title (ZH)**: 基于视觉的强化学习的自我预测动力学泛化方法 

**Authors**: Kyungsoo Kim, Jeongsoo Ha, Yusung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.05418)  

**Abstract**: Vision-based reinforcement learning requires efficient and robust representations of image-based observations, especially when the images contain distracting (task-irrelevant) elements such as shadows, clouds, and light. It becomes more important if those distractions are not exposed during training. We design a Self-Predictive Dynamics (SPD) method to extract task-relevant features efficiently, even in unseen observations after training. SPD uses weak and strong augmentations in parallel, and learns representations by predicting inverse and forward transitions across the two-way augmented versions. In a set of MuJoCo visual control tasks and an autonomous driving task (CARLA), SPD outperforms previous studies in complex observations, and significantly improves the generalization performance for unseen observations. Our code is available at this https URL. 

**Abstract (ZH)**: 基于视觉的强化学习需要在图像包含阴影、云朵和光线等干扰元素时，尤其是在这些干扰元素在训练中未被暴露的情况下，有效地提取相关的特征表示。我们设计了一种自我预测动力学（SPD）方法，在训练后的未见 observation 中也能高效提取任务相关特征。SPD 并行使用弱增强和强增强，并通过预测双向增强版本的逆向和正向转换来学习表示。在一系列 MuJoCo 视觉控制任务和自主驾驶任务（CARLA）中，SPD 在复杂 observation 中优于先前研究，并显著提高了未见 observation 的泛化性能。相关代码已发布在以下链接：此 https URL。 

---
# FERRET: Private Deep Learning Faster And Better Than DPSGD 

**Title (ZH)**: FERRET: 私有深度学习更快更好于DPSGD 

**Authors**: David Zagardo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05416)  

**Abstract**: We revisit 1-bit gradient compression through the lens of mutual-information differential privacy (MI-DP). Building on signSGD, we propose FERRET--Fast and Effective Restricted Release for Ethical Training--which transmits at most one sign bit per parameter group with Bernoulli masking.
Theory: We prove each fired group leaks at most ln 2 nats; after subsampling with rate s, the total privacy loss of G groups trained for T steps with firing probability p is epsilon = G * T * s * p * ln 2. Thus FERRET achieves MI-DP for epsilon in [0.1, 2] without additive noise.
Practice: We evaluate three granularities--FERRET-MAX (finest), FERRET-EIGHTH (medium), and FERRET-2 (coarsest)--on five LLMs (137M-1.8B parameters) against DPSGD and Non-DP baselines. All methods trained for 1, 3, and 5 epochs.
Utility: Across all settings, FERRET-MAX/EIGHTH beat DPSGD's perplexity. At epsilon=0.5, 5 epochs: FERRET-EIGHTH achieves 3.98 perplexity vs DPSGD's 11.61 (2.9x better), within 23% of Non-DP (3.25).
Privacy: MI-AUC stays at chance for FERRET-MAX/EIGHTH (~0.51), matching DPSGD vs Non-DP's 0.76-0.99. FERRET-2 shows higher leakage (~0.55) due to lower headroom.
Efficiency: Stricter budgets fire fewer signs, so FERRET uses 19-33% of DPSGD's training time and only 34-36% of Non-DP training time.
Take-away: Sign-based MI-DP gets closer to achieving all three qualities of the privacy, utility, performance trilemma: FERRET trains up to 5x faster, achieves 3x lower perplexity compared to DPSGD and 1.2x greater than Non-DP, all while providing formal, mathematically provable privacy guarantees using zero additive noise. The results also show that, in certain instances, masked 1-bit updates can match non-private training utility while safeguarding data. 

**Abstract (ZH)**: 我们通过互信息差分隐私（MI-DP）的视角重新审视1比特梯度压缩。基于signSGD，我们提出FERRET——快速有效的受限制发布伦理训练方法——每参数组最多传输一个符号位，并使用伯努利蒙版。 

---
# SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing 

**Title (ZH)**: SAVVY: 通过视听LLM观察与聆听的空间意识 

**Authors**: Mingfei Chen, Zijun Cui, Xiulong Liu, Jinlin Xiang, Caleb Zheng, Jingyuan Li, Eli Shlizerman  

**Link**: [PDF](https://arxiv.org/pdf/2506.05414)  

**Abstract**: 3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs. 

**Abstract (ZH)**: 三维动态环境中的视听空间推理是人类认知的一个基石，但现有视听大型语言模型（AV-LLMs）及其基准测试大多集中于静态或二维场景，尚未充分探索。我们引入了SAVVY-Bench，这是首个用于动态场景中同步空间音频的精细三维空间推理基准测试，包括成千上万涉及静止和移动物体的关系，要求精细的时间对接、一致的三维定位和多模态注释。为应对这一挑战，我们提出了SAVVY，一种无需训练的新型推理管道，包含两个阶段：(i) 自我中心空间轨迹估计，利用AV-LLMs及其他视听方法，结合视觉和空间音频线索跟踪与查询相关的关键物体的轨迹，以及(ii) 动态全局地图构建，聚合多模态查询物体轨迹并转化为统一的全球动态地图。通过构建的地图，最终的问答答案通过坐标转换获得，使全球地图与查询视角对齐。实证评估表明，SAVVY显著提升了现有最佳视听大型语言模型的性能，为AV-LLMs中的动态三维空间推理设定了新的标准和起点。 

---
# SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs 

**Title (ZH)**: SmoothRot: 结合通道级缩放和旋转以实现量化友好的大语言模型 

**Authors**: Patrik Czakó, Gábor Kertész, Sándor Szénási  

**Link**: [PDF](https://arxiv.org/pdf/2506.05413)  

**Abstract**: We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at this https URL. 

**Abstract (ZH)**: SmoothRot：一种改进大型语言模型4比特量化效率的新颖后训练量化技术 

---
# AD-EE: Early Exiting for Fast and Reliable Vision-Language Models in Autonomous Driving 

**Title (ZH)**: AD-EE: 自动驾驶中快速可靠视觉语言模型的早期退出方法 

**Authors**: Lianming Huang, Haibo Hu, Yufei Cui, Jiacheng Zuo, Shangyu Wu, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2506.05404)  

**Abstract**: With the rapid advancement of autonomous driving, deploying Vision-Language Models (VLMs) to enhance perception and decision-making has become increasingly common. However, the real-time application of VLMs is hindered by high latency and computational overhead, limiting their effectiveness in time-critical driving scenarios. This challenge is particularly evident when VLMs exhibit over-inference, continuing to process unnecessary layers even after confident predictions have been reached. To address this inefficiency, we propose AD-EE, an Early Exit framework that incorporates domain characteristics of autonomous driving and leverages causal inference to identify optimal exit layers. We evaluate our method on large-scale real-world autonomous driving datasets, including Waymo and the corner-case-focused CODA, as well as on a real vehicle running the Autoware Universe platform. Extensive experiments across multiple VLMs show that our method significantly reduces latency, with maximum improvements reaching up to 57.58%, and enhances object detection accuracy, with maximum gains of up to 44%. 

**Abstract (ZH)**: 随着自主驾驶技术的rapid advancement, 将视觉-语言模型(Vision-Language Models, VLMs)部署以增强感知和决策的应用越来越普遍。然而,VLMs的实时应用受到高延迟和计算开销的限制, 在时间敏感的驾驶场景中限制了其有效性。特别是在VLMs表现出过度推断的情况下, 即使在达成自信预测后仍持续处理不必要的层, 这一挑战尤为明显。为解决这一低效率问题, 我们提出AD-EE, 一种Early Exit框架, 结合自主驾驶领域的特性并利用因果推理来识别最优退出层。我们在大规模的实际自主驾驶数据集上评估了我们的方法, 包括Waymo和以边缘情况为重点的CODA, 以及在使用Autoware Universe平台的真实车辆上进行了评估。多个视觉-语言模型的广泛实验结果显示, 我们的方法显著减少了延迟, 最大改善幅度达到57.58%, 并提升了对象检测准确性, 最大增益达到44%。 

---
# Attention-based transformer models for image captioning across languages: An in-depth survey and evaluation 

**Title (ZH)**: 基于注意力的变换器模型在跨语言图像 Captioning 中的应用：一种深入的综述与评估 

**Authors**: Israa A. Albadarneh, Bassam H. Hammo, Omar S. Al-Kadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05399)  

**Abstract**: Image captioning involves generating textual descriptions from input images, bridging the gap between computer vision and natural language processing. Recent advancements in transformer-based models have significantly improved caption generation by leveraging attention mechanisms for better scene understanding. While various surveys have explored deep learning-based approaches for image captioning, few have comprehensively analyzed attention-based transformer models across multiple languages. This survey reviews attention-based image captioning models, categorizing them into transformer-based, deep learning-based, and hybrid approaches. It explores benchmark datasets, discusses evaluation metrics such as BLEU, METEOR, CIDEr, and ROUGE, and highlights challenges in multilingual captioning. Additionally, this paper identifies key limitations in current models, including semantic inconsistencies, data scarcity in non-English languages, and limitations in reasoning ability. Finally, we outline future research directions, such as multimodal learning, real-time applications in AI-powered assistants, healthcare, and forensic analysis. This survey serves as a comprehensive reference for researchers aiming to advance the field of attention-based image captioning. 

**Abstract (ZH)**: 基于注意力的图像字幕模型综述 

---
# Gen4D: Synthesizing Humans and Scenes in the Wild 

**Title (ZH)**: Gen4D：在自然场景中合成人类和场景 

**Authors**: Jerrin Bright, Zhibo Wang, Yuhao Chen, Sirisha Rambhatla, John Zelek, David Clausi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05397)  

**Abstract**: Lack of input data for in-the-wild activities often results in low performance across various computer vision tasks. This challenge is particularly pronounced in uncommon human-centric domains like sports, where real-world data collection is complex and impractical. While synthetic datasets offer a promising alternative, existing approaches typically suffer from limited diversity in human appearance, motion, and scene composition due to their reliance on rigid asset libraries and hand-crafted rendering pipelines. To address this, we introduce Gen4D, a fully automated pipeline for generating diverse and photorealistic 4D human animations. Gen4D integrates expert-driven motion encoding, prompt-guided avatar generation using diffusion-based Gaussian splatting, and human-aware background synthesis to produce highly varied and lifelike human sequences. Based on Gen4D, we present SportPAL, a large-scale synthetic dataset spanning three sports: baseball, icehockey, and soccer. Together, Gen4D and SportPAL provide a scalable foundation for constructing synthetic datasets tailored to in-the-wild human-centric vision tasks, with no need for manual 3D modeling or scene design. 

**Abstract (ZH)**: 野生活动输入数据的缺乏常常导致各种计算机视觉任务性能低下。这一挑战在复杂的非常见人类中心领域（如体育）表现尤为明显，因为在这些领域中，现实世界数据的收集既复杂又不现实。虽然合成数据集提供了有前景的替代方案，但现有方法通常由于依赖于刚体资产库和手工编写的渲染管线，而在人类外观、动作和场景构成的多样性方面存在局限。为了解决这一问题，我们介绍了Gen4D，这是一个全自动的生成多样化和逼真4D人体动画的pipeline。Gen4D结合了专家驱动的动作编码、提示引导的基于扩散的Gaussian溅射avatar生成以及人体意识背景合成，以产生高度多样且逼真的人类序列。基于Gen4D，我们介绍了SportPAL，这是一个涵盖三类运动（棒球、冰球和足球）的大规模合成数据集。结合Gen4D和SportPAL，我们提供了一个可扩展的基础，用于构建针对野生人类中心视觉任务的定制合成数据集，无需手动3D建模或场景设计。 

---
# Advancing Decoding Strategies: Enhancements in Locally Typical Sampling for LLMs 

**Title (ZH)**: 改进解码策略：增强局部典型采样方法在大语言模型中的应用 

**Authors**: Jaydip Sen, Saptarshi Sengupta. Subhasis Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05387)  

**Abstract**: This chapter explores advancements in decoding strategies for large language models (LLMs), focusing on enhancing the Locally Typical Sampling (LTS) algorithm. Traditional decoding methods, such as top-k and nucleus sampling, often struggle to balance fluency, diversity, and coherence in text generation. To address these challenges, Adaptive Semantic-Aware Typicality Sampling (ASTS) is proposed as an improved version of LTS, incorporating dynamic entropy thresholding, multi-objective scoring, and reward-penalty adjustments. ASTS ensures contextually coherent and diverse text generation while maintaining computational efficiency. Its performance is evaluated across multiple benchmarks, including story generation and abstractive summarization, using metrics such as perplexity, MAUVE, and diversity scores. Experimental results demonstrate that ASTS outperforms existing sampling techniques by reducing repetition, enhancing semantic alignment, and improving fluency. 

**Abstract (ZH)**: 本章探讨了大规模语言模型（LLM）解码策略的进展，重点关注增强局部典型采样（LTS）算法。传统的解码方法，如top-k和nucleus采样，往往在文本生成中难以平衡流畅性、多样性和一致性。为此，提出了一种改进的自适应语义感知典型性采样（ASTS）算法，该算法结合了动态熵阈值、多目标评分和奖惩调整。ASTS能够保证上下文一致性和多样性的同时保持计算效率。其性能通过故事生成和抽象总结等基准测试进行评估，采用困惑度、MAUVE和多样性评分等指标。实验结果表明，ASTS在减少重复、增强语义对齐和改善流畅性方面优于现有采样技术。 

---
# Beyond RAG: Reinforced Reasoning Augmented Generation for Clinical Notes 

**Title (ZH)**: 超越RAG：强化推理增强生成在临床笔记中的应用 

**Authors**: Lo Pang-Yun Ting, Chengshuai Zhao, Yu-Hua Zeng, Yuan Jee Lim, Kun-Ta Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05386)  

**Abstract**: Clinical note generation aims to automatically produce free-text summaries of a patient's condition and diagnostic process, with discharge instructions being a representative long-form example. While recent large language model (LLM)-based methods pre-trained on general clinical corpora show promise in clinical text generation, they fall short in producing long-form notes from limited patient information. In this paper, we propose R2AG, the first reinforced retriever for long-form discharge instruction generation based on pre-admission data. R2AG is trained with reinforcement learning to retrieve reasoning paths from a medical knowledge graph, providing explicit semantic guidance to the LLM. To bridge the information gap, we propose Group-Based Retriever Optimization (GRO) which improves retrieval quality with group-relative rewards, encouraging reasoning leaps for deeper inference by the LLM. Comprehensive experiments on the MIMIC-IV-Note dataset show that R2AG outperforms baselines in both clinical efficacy and natural language generation metrics. Further analysis reveals that R2AG fills semantic gaps in sparse input scenarios, and retrieved reasoning paths help LLMs avoid clinical misinterpretation by focusing on key evidence and following coherent reasoning. 

**Abstract (ZH)**: 临床笔记生成旨在自动生成患者状况和诊断过程的自由文本总结，出院指示是典型的长文例证。虽然基于通用临床语料库预训练的大规模语言模型 (LLM) 在临床文本生成方面展现出希望，但在有限的患者信息下生成长文笔记方面仍然不足。本文提出 R2AG，这是一种基于预住院数据的第一个强化检索器，用于长文出院指示生成。R2AG 通过强化学习训练，从医学知识图谱中检索推理路径，为LLM 提供显式的语义指导。为弥补信息缺口，我们提出基于组别优化的检索器（GRO），通过组内相对奖励提高检索质量，鼓励LLM 进行更深层次的推理。在 MIMIC-IV-Note 数据集上的全面实验表明，R2AG 在临床效果和自然语言生成指标上均优于基线方法。进一步的分析表明，R2AG 在稀疏输入场景中填补了语义空缺，并检索到的推理路径有助于LLM 避免临床误读，通过关注关键证据并遵循连贯的推理过程。 

---
# Q-Ponder: A Unified Training Pipeline for Reasoning-based Visual Quality Assessment 

**Title (ZH)**: Q-Ponder：一种基于推理的视觉质量评估统一训练流程 

**Authors**: Zhuoxuan Cai, Jian Zhang, Xinbin Yuan, Pengtao Jiang, Wenxiang Chen, Bowen Tang, Lujian Yao, Qiyuan Wang, Jinwen Chen, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05384)  

**Abstract**: Recent studies demonstrate that multimodal large language models (MLLMs) can proficiently evaluate visual quality through interpretable assessments. However, existing approaches typically treat quality scoring and reasoning descriptions as separate tasks with disjoint optimization objectives, leading to a trade-off: models adept at quality reasoning descriptions struggle with precise score regression, while score-focused models lack interpretability. This limitation hinders the full potential of MLLMs in visual quality assessment, where accuracy and interpretability should be mutually reinforcing. To address this, we propose a unified two-stage training framework comprising a cold-start stage and a reinforcement learning-based fine-tuning stage. Specifically, in the first stage, we distill high-quality data from a teacher model through expert-designed prompts, initializing reasoning capabilities via cross-entropy loss supervision. In the second stage, we introduce a novel reward with Group Relative Policy Optimization (GRPO) to jointly optimize scoring accuracy and reasoning consistency. We designate the models derived from these two stages as Q-Ponder-CI and Q-Ponder. Extensive experiments show that Q-Ponder achieves state-of-the-art (SOTA) performance on quality score regression benchmarks, delivering up to 6.5% higher SRCC on cross-domain datasets. Furthermore, Q-Ponder significantly outperforms description-based SOTA models, including its teacher model Qwen-2.5-VL-72B, particularly in description accuracy and reasonableness, demonstrating the generalization potential over diverse tasks. 

**Abstract (ZH)**: 近期研究显示，多模态大型语言模型（MLLMs）能够通过可解释的评估来熟练地评价图像质量。然而，现有方法通常将质量评分和理由描述视为分离的任务，各自优化目标不一致，导致权衡：擅长理由描述的模型在精确评分回归方面表现不佳，而注重评分的模型缺乏解释性。这一限制阻碍了MLLMs在图像质量评估中的全部潜力，其中准确性与解释性应该是相互促进的关系。为了解决这一问题，我们提出了一种统一的两阶段训练框架，包括一个冷启动阶段和一个基于强化学习的微调阶段。在第一阶段，我们通过专家设计的提示从教师模型中提取高质量数据，并通过交叉熵损失监督初始化推理能力。在第二阶段，我们引入了组相对策略优化（GRPO）的新型奖励，以联合优化评分准确性和理由一致性。我们从这两个阶段得出的模型分别命名为Q-Ponder-CI和Q-Ponder。 extensive实验表明，Q-Ponder在质量评分回归基准测试中达到了现有最佳性能（SOTA），在跨领域数据集上SRCC可提高6.5%。此外，Q-Ponder显著优于基于描述的SOTA模型，包括其教师模型Qwen-2.5-VL-72B，在描述准确性和合理性方面表现尤为出色，展示了其在多样化任务中的泛化潜力。 

---
# How stealthy is stealthy? Studying the Efficacy of Black-Box Adversarial Attacks in the Real World 

**Title (ZH)**: 隐形的有多隐形？探究黑盒对抗攻击在实际环境中的有效性 

**Authors**: Francesco Panebianco, Mario D'Onghia, Stefano Zanero aand Michele Carminati  

**Link**: [PDF](https://arxiv.org/pdf/2506.05382)  

**Abstract**: Deep learning systems, critical in domains like autonomous vehicles, are vulnerable to adversarial examples (crafted inputs designed to mislead classifiers). This study investigates black-box adversarial attacks in computer vision. This is a realistic scenario, where attackers have query-only access to the target model. Three properties are introduced to evaluate attack feasibility: robustness to compression, stealthiness to automatic detection, and stealthiness to human inspection. State-of-the-Art methods tend to prioritize one criterion at the expense of others. We propose ECLIPSE, a novel attack method employing Gaussian blurring on sampled gradients and a local surrogate model. Comprehensive experiments on a public dataset highlight ECLIPSE's advantages, demonstrating its contribution to the trade-off between the three properties. 

**Abstract (ZH)**: 深度学习系统在自动驾驶等领域的应用极易受到对抗样本的攻击（特制的输入旨在误导分类器）。本文研究了计算机视觉中的黑盒对抗攻击。攻击者仅对目标模型具有查询访问权限，是一种现实场景。本文引入了评估攻击可行性的三个属性：对压缩的鲁棒性、自动检测中的隐形性和人工检查中的隐形性。现有先进方法往往在这些标准之间权衡取舍。我们提出了一种新颖的攻击方法ECLIPSE，该方法使用高斯模糊处理采样梯度并采用局部代理模型。在公共数据集上的全面实验突显了ECLIPSE的优势，展示了其在三个属性之间的权衡中的贡献。 

---
# Designing DSIC Mechanisms for Data Sharing in the Era of Large Language Models 

**Title (ZH)**: 大型语言模型时代的数据共享DSIC机制设计 

**Authors**: Seyed Moein Ayyoubzadeh, Kourosh Shahnazari, Mohammmadali Keshtparvar, MohammadAmin Fazli  

**Link**: [PDF](https://arxiv.org/pdf/2506.05379)  

**Abstract**: Training large language models (LLMs) requires vast amounts of high-quality data from institutions that face legal, privacy, and strategic constraints. Existing data procurement methods often rely on unverifiable trust or ignore heterogeneous provider costs. We introduce a mechanism-design framework for truthful, trust-minimized data sharing that ensures dominant-strategy incentive compatibility (DSIC), individual rationality, and weak budget balance, while rewarding data based on both quality and learning utility. We formalize a model where providers privately know their data cost and quality, and value arises solely from the data's contribution to model performance. Based on this, we propose the Quality-Weighted Marginal-Incentive Auction (Q-MIA), which ranks providers using a virtual cost metric and uses Myerson-style payments to ensure DSIC and budget feasibility. To support settings with limited liquidity or long-term incentives, we introduce the Marginal Utility Token (MUT), which allocates future rights based on marginal contributions. We unify these in Mixed-MIA, a hybrid mechanism balancing upfront payments and deferred rewards. All mechanisms support verifiable, privacy-preserving implementation. Theoretically and empirically, they outperform volume-based and trust-based baselines, eliciting higher-quality data under budget constraints while remaining robust to misreporting and collusion. This establishes a principled foundation for sustainable and fair data markets for future LLMs. 

**Abstract (ZH)**: 一种面向大规模语言模型的机制设计框架：最小化信任的数据共享机制 

---
# A Red Teaming Roadmap Towards System-Level Safety 

**Title (ZH)**: 面向系统级安全的红队演练路线图 

**Authors**: Zifan Wang, Christina Q. Knight, Jeremy Kritz, Willow E. Primack, Julian Michael  

**Link**: [PDF](https://arxiv.org/pdf/2506.05376)  

**Abstract**: Large Language Model (LLM) safeguards, which implement request refusals, have become a widely adopted mitigation strategy against misuse. At the intersection of adversarial machine learning and AI safety, safeguard red teaming has effectively identified critical vulnerabilities in state-of-the-art refusal-trained LLMs. However, in our view the many conference submissions on LLM red teaming do not, in aggregate, prioritize the right research problems. First, testing against clear product safety specifications should take a higher priority than abstract social biases or ethical principles. Second, red teaming should prioritize realistic threat models that represent the expanding risk landscape and what real attackers might do. Finally, we contend that system-level safety is a necessary step to move red teaming research forward, as AI models present new threats as well as affordances for threat mitigation (e.g., detection and banning of malicious users) once placed in a deployment context. Adopting these priorities will be necessary in order for red teaming research to adequately address the slate of new threats that rapid AI advances present today and will present in the very near future. 

**Abstract (ZH)**: 大型语言模型（LLM）防护措施，通过实施请求拒绝，已成为应对滥用的广泛应用的缓解策略。在对抗机器学习和人工智能安全的交叉点上，防护红队活动有效地识别了先进拒绝训练的LLM中的关键漏洞。然而，我们认为，关于LLM红队的众多会议投稿在整体上没有优先解决正确的研究问题。首先，针对清晰的产品安全规范进行测试应高于抽象的社会偏见或伦理原则。其次，红队活动应优先考虑现实威胁模型，以代表不断扩大的风险景观和实际攻击者可能的行为。最后，我们认为，系统级安全是推动红队研究向前发展的必要步骤，因为一旦在部署环境中存在，AI模型既带来了新的威胁，也提供了威胁缓解的可能性（如检测和封禁恶意用户）。为了使红队研究能够充分应对当前及非常近未来快速发展的AI技术带来的新威胁，采纳这些优先事项是必要的。 

---
# Speaking images. A novel framework for the automated self-description of artworks 

**Title (ZH)**: 描绘图像：一种自动化艺术作品自我描述的新框架 

**Authors**: Valentine Bernasconi, Gustavo Marfia  

**Link**: [PDF](https://arxiv.org/pdf/2506.05368)  

**Abstract**: Recent breakthroughs in generative AI have opened the door to new research perspectives in the domain of art and cultural heritage, where a large number of artifacts have been digitized. There is a need for innovation to ease the access and highlight the content of digital collections. Such innovations develop into creative explorations of the digital image in relation to its malleability and contemporary interpretation, in confrontation to the original historical object. Based on the concept of the autonomous image, we propose a new framework towards the production of self-explaining cultural artifacts using open-source large-language, face detection, text-to-speech and audio-to-animation models. The goal is to start from a digitized artwork and to automatically assemble a short video of the latter where the main character animates to explain its content. The whole process questions cultural biases encapsulated in large-language models, the potential of digital images and deepfakes of artworks for educational purposes, along with concerns of the field of art history regarding such creative diversions. 

**Abstract (ZH)**: 近期生成式人工智能的突破为艺术和文化遗产领域的新研究视角打开了大门，其中大量文物已被数字化。需要创新以简化访问并突出数字收藏的内容。这些创新发展成为对数字图像及其可塑性和当代诠释的创造性探索，与原始历史物体相对。基于自主图像的概念，我们提出了一种新的框架，用于生成自我解释的文化艺术品，利用开源的大型语言模型、面部检测、文本转语音和音频到动画模型。目标是从一件数字化的艺术作品开始，自动组装一个简短视频，其中的主要角色会解释其内容。整个过程质疑大型语言模型中嵌入的文化偏见，数字图像和艺术品的深伪技术在教育目的上的潜力，以及艺术史领域对此类创造性的关切。 

---
# Can ChatGPT Perform Image Splicing Detection? A Preliminary Study 

**Title (ZH)**: ChatGPT能否进行图像拼接检测？一项初步研究 

**Authors**: Souradip Nath  

**Link**: [PDF](https://arxiv.org/pdf/2506.05358)  

**Abstract**: Multimodal Large Language Models (MLLMs) like GPT-4V are capable of reasoning across text and image modalities, showing promise in a variety of complex vision-language tasks. In this preliminary study, we investigate the out-of-the-box capabilities of GPT-4V in the domain of image forensics, specifically, in detecting image splicing manipulations. Without any task-specific fine-tuning, we evaluate GPT-4V using three prompting strategies: Zero-Shot (ZS), Few-Shot (FS), and Chain-of-Thought (CoT), applied over a curated subset of the CASIA v2.0 splicing dataset.
Our results show that GPT-4V achieves competitive detection performance in zero-shot settings (more than 85% accuracy), with CoT prompting yielding the most balanced trade-off across authentic and spliced images. Qualitative analysis further reveals that the model not only detects low-level visual artifacts but also draws upon real-world contextual knowledge such as object scale, semantic consistency, and architectural facts, to identify implausible composites. While GPT-4V lags behind specialized state-of-the-art splicing detection models, its generalizability, interpretability, and encyclopedic reasoning highlight its potential as a flexible tool in image forensics. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）如GPT-4V能够在文本和图像模态间进行推理，在复杂视觉语言任务中展现出巨大潜力。在这一初步研究中，我们探讨了GPT-4V在图像取证领域的通用能力，特别是在检测图像拼接篡改方面的能力。未经任何任务特定微调，我们通过三种提示策略（零样本、少量样本和逐步推理）评估了GPT-4V在CASIA v2.0拼接数据集子集上的表现。结果显示，GPT-4V在零样本设置下实现了竞争力的检测性能（超过85%的准确率），逐步推理提示策略提供了在真伪图像间最为均衡的权衡。进一步的定性分析表明，该模型不仅检测低级别视觉伪影，还利用现实世界上下文知识如物体尺度、语义一致性及建筑事实来识别不合逻辑的复合图像。虽然GPT-4V在专门的拼接检测模型面前略逊一筹，但其泛化能力、可解释性和百科全书式推理展示了其在图像取证领域作为灵活工具的潜力。 

---
# Infinite Time Turing Machines and their Applications 

**Title (ZH)**: 无限时间图灵机及其应用 

**Authors**: Rukmal Weerawarana, Maxwell Braun  

**Link**: [PDF](https://arxiv.org/pdf/2506.05351)  

**Abstract**: This work establishes a rigorous theoretical foundation for analyzing deep learning systems by leveraging Infinite Time Turing Machines (ITTMs), which extend classical computation into transfinite ordinal steps. Using ITTMs, we reinterpret modern architectures like Transformers, revealing fundamental limitations in scalability, efficiency, and interpretability. Building on these insights, we propose the Universal State Machine (USM), a novel computational paradigm designed from first principles. The USM employs a dynamic, queryable computation graph that evolves in real time, enabling modular, interpretable, and resource-efficient computation. This framework not only overcomes the inefficiencies and rigidity of current models but also lays the groundwork for scalable, generalizable artificial intelligence systems. 

**Abstract (ZH)**: 利用无限时间图灵机建立深学习系统分析的严格理论基础：通用状态机的提出及其高效可解释计算图 Paradigm 

---
# Towards provable probabilistic safety for scalable embodied AI systems 

**Title (ZH)**: 可验证的概率安全性朝着可扩展的具身AI系统的方向研究 

**Authors**: Linxuan He, Qing-Shan Jia, Ang Li, Hongyan Sang, Ling Wang, Jiwen Lu, Tao Zhang, Jie Zhou, Yi Zhang, Yisen Wang, Peng Wei, Zhongyuan Wang, Henry X. Liu, Shuo Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05171)  

**Abstract**: Embodied AI systems, comprising AI models and physical plants, are increasingly prevalent across various applications. Due to the rarity of system failures, ensuring their safety in complex operating environments remains a major challenge, which severely hinders their large-scale deployment in safety-critical domains, such as autonomous vehicles, medical devices, and robotics. While achieving provable deterministic safety--verifying system safety across all possible scenarios--remains theoretically ideal, the rarity and complexity of corner cases make this approach impractical for scalable embodied AI systems. To address this challenge, we introduce provable probabilistic safety, which aims to ensure that the residual risk of large-scale deployment remains below a predefined threshold. Instead of attempting exhaustive safety proof across all corner cases, this paradigm establishes a probabilistic safety boundary on overall system performance, leveraging statistical methods to enhance feasibility and scalability. A well-defined probabilistic safety boundary enables embodied AI systems to be deployed at scale while allowing for continuous refinement of safety guarantees. Our work focuses on three core questions: what is provable probabilistic safety, how to prove the probabilistic safety, and how to achieve the provable probabilistic safety. By bridging the gap between theoretical safety assurance and practical deployment, our work offers a pathway toward safer, large-scale adoption of embodied AI systems in safety-critical applications. 

**Abstract (ZH)**: 具身AI系统的可验证概率安全：理论与实践的桥梁 

---
# Category Query Learning for Human-Object Interaction Classification 

**Title (ZH)**: 人类对象交互分类的类别查询学习 

**Authors**: Chi Xie, Fangao Zeng, Yue Hu, Shuang Liang, Yichen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2303.14005)  

**Abstract**: Unlike most previous HOI methods that focus on learning better human-object features, we propose a novel and complementary approach called category query learning. Such queries are explicitly associated to interaction categories, converted to image specific category representation via a transformer decoder, and learnt via an auxiliary image-level classification task. This idea is motivated by an earlier multi-label image classification method, but is for the first time applied for the challenging human-object interaction classification task. Our method is simple, general and effective. It is validated on three representative HOI baselines and achieves new state-of-the-art results on two benchmarks. 

**Abstract (ZH)**: 不同于大多数以往的人机对象方法侧重于学习更好的人-物特征，我们提出了一种新颖且互补的方法，称为类别查询学习。此类别查询明确与交互类别相关联，通过变压器解码器转换为图像特定的类别表示，并通过辅助的图像级分类任务进行学习。这一想法受到早期的多标签图像分类方法的启发，但首次应用于具有挑战性的交互分类任务。我们的方法简单、通用且有效，并在三个代表性的HOI基准上进行了验证，在两个基准上取得了新的最佳结果。 

---
