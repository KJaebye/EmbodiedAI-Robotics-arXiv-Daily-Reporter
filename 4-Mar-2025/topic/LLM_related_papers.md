# LLM-Advisor: An LLM Benchmark for Cost-efficient Path Planning across Multiple Terrains 

**Title (ZH)**: LLM-Advisor: 一种跨多种地形高效成本路径规划的LLM基准 

**Authors**: Ling Xiao, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.01236)  

**Abstract**: Multi-terrain cost-efficient path planning is a crucial task in robot navigation, requiring the identification of a path from the start to the goal that not only avoids obstacles but also minimizes travel costs. This is especially crucial for real-world applications where robots need to navigate diverse terrains in outdoor environments, where recharging or refueling is difficult. However, there is very limited research on this topic. In this paper, we develop a prompt-based approach, LLM-Advisor, which leverages large language models (LLMs) as effective advisors for path planning. The LLM-Advisor selectively provides suggestions, demonstrating its ability to recognize when no modifications are necessary. When suggestions are made, 70.59% of the paths suggested for the A* algorithm, 69.47% for the RRT* algorithm, and 78.70% for the LLM-A* algorithm achieve greater cost efficiency. Since LLM-Advisor may occasionally lack common sense in their suggestions, we propose two hallucination-mitigation strategies. Furthermore, we experimentally verified that GPT-4o performs poorly in zero-shot path planning, even when terrain descriptions are clearly provided, demonstrating its low spatial awareness. We also experimentally demonstrate that using an LLM as an advisor is more effective than directly integrating it into the path-planning loop. Since LLMs may generate hallucinations, using LLMs in the loop of a search-based method (such as A*) may lead to a higher number of failed paths, demonstrating that our proposed LLM-Advisor is a better choice. 

**Abstract (ZH)**: 多地形低成本路径规划是机器人导航中的关键任务，要求识别一条既避免障碍物又最小化旅行成本的路径。特别是在室外环境中需要导航多种地形的实际情况中，充电或加油都很困难。然而，这方面的研究非常有限。在本文中，我们开发了一种基于提示的方法LLM-Advisor，利用大型语言模型（LLMs）作为路径规划的有效顾问。LLM-Advisor有选择性地提供建议，展示了其在必要时不需要修改路径的辨识能力。在提出建议的情况下，70.59%的路径对于A*算法、69.47%的路径对于RRT*算法以及78.70%的路径对于LLM-A*算法实现了更高的成本效率。由于LLM-Advisor偶尔可能在其建议中缺乏常识，我们提出了两种减轻幻觉的策略。此外，我们实验证明，即使提供了明确的地形描述，GPT-4o在零样本路径规划中表现不佳，显示出其低空间意识。我们还实验证明，使用LLM作为顾问比直接将其集成到路径规划循环中更有效。由于LLM可能会生成幻觉，在基于搜索的方法（如A*）的循环中使用LLM可能会导致更多失败的路径，这进一步证明了我们提出的LLM-Advisor是一个更好的选择。 

---
# Never too Prim to Swim: An LLM-Enhanced RL-based Adaptive S-Surface Controller for AUVs under Extreme Sea Conditions 

**Title (ZH)**: 永不过时的泳趣：一种增强学习模型辅助的适应性S-表面控制器用于极端海况下的自治 underwater 机器人 

**Authors**: Guanwen Xie, Jingzehua Xu, Yimian Ding, Zhi Zhang, Shuai Zhang, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00527)  

**Abstract**: The adaptivity and maneuvering capabilities of Autonomous Underwater Vehicles (AUVs) have drawn significant attention in oceanic research, due to the unpredictable disturbances and strong coupling among the AUV's degrees of freedom. In this paper, we developed large language model (LLM)-enhanced reinforcement learning (RL)-based adaptive S-surface controller for AUVs. Specifically, LLMs are introduced for the joint optimization of controller parameters and reward functions in RL training. Using multi-modal and structured explicit task feedback, LLMs enable joint adjustments, balance multiple objectives, and enhance task-oriented performance and adaptability. In the proposed controller, the RL policy focuses on upper-level tasks, outputting task-oriented high-level commands that the S-surface controller then converts into control signals, ensuring cancellation of nonlinear effects and unpredictable external disturbances in extreme sea conditions. Under extreme sea conditions involving complex terrain, waves, and currents, the proposed controller demonstrates superior performance and adaptability in high-level tasks such as underwater target tracking and data collection, outperforming traditional PID and SMC controllers. 

**Abstract (ZH)**: 自主 underwater 车辆 (AUVs) 的适应性和机动能力由于海洋环境中的不可预测干扰和自由度间的强耦合而受到广泛关注。本文提出了一种大型语言模型 (LLM) 增强的强化学习 (RL) 基准自适应 S-表面控制器。具体而言，LLM 用于在 RL 训练过程中联合优化控制器参数和奖励函数。利用多模态和结构化的显式任务反馈，LLM 使参数联合调整、平衡多个目标，并提高任务导向的性能和适应性。在所提出的控制器中，RL 策略关注高层任务，输出任务导向的高层命令，S-表面控制器将其转换为控制信号，以确保在极端海况下消除非线性影响和不可预测的外部干扰。在涉及复杂地形、波浪和流速的极端海况下，所提出的控制器在水下目标跟踪和数据采集等高层任务中表现出色，优于传统的 PID 和 SMC 控制器。 

---
# Interact, Instruct to Improve: A LLM-Driven Parallel Actor-Reasoner Framework for Enhancing Autonomous Vehicle Interactions 

**Title (ZH)**: 交互，指令以提高：一个由LLM驱动的并行行动-推理框架，用于增强自动驾驶车辆交互 

**Authors**: Shiyu Fang, Jiaqi Liu, Chengkai Xu, Chen Lv, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00502)  

**Abstract**: Autonomous Vehicles (AVs) have entered the commercialization stage, but their limited ability to interact and express intentions still poses challenges in interactions with Human-driven Vehicles (HVs). Recent advances in large language models (LLMs) enable bidirectional human-machine communication, but the conflict between slow inference speed and the need for real-time decision-making challenges practical deployment. To address these issues, this paper introduces a parallel Actor-Reasoner framework designed to enable explicit bidirectional AV-HV interactions across multiple scenarios. First, by facilitating interactions between the LLM-driven Reasoner and heterogeneous simulated HVs during training, an interaction memory database, referred to as the Actor, is established. Then, by introducing the memory partition module and the two-layer memory retrieval module, the Actor's ability to handle heterogeneous HVs is significantly enhanced. Ablation studies and comparisons with other decision-making methods demonstrate that the proposed Actor-Reasoner framework significantly improves safety and efficiency. Finally, with the combination of the external Human-Machine Interface (eHMI) information derived from Reasoner's reasoning and the feasible action solutions retrieved from the Actor, the effectiveness of the proposed Actor-Reasoner is confirmed in multi-scenario field interactions. Our code is available at this https URL. 

**Abstract (ZH)**: 自主驾驶车辆（AVs）已进入商业化阶段，但其有限的交互和意图表达能力仍给与人类驾驶车辆（HVs）的交互带来挑战。大型语言模型（LLMs）的 Recent 进展使双向人机通信成为可能，但推理速度缓慢与实时决策需求之间的冲突挑战其实际部署。为解决这些问题，本文引入一种并行 Actor-Reasoner 框架，旨在实现多种场景下AV-HV的明确双向交互。首先，在训练过程中通过LLM驱动的Reasoner与异构模拟HV的交互建立一个交互记忆数据库，称为Actor。然后，通过引入记忆分区模块和两层记忆检索模块，显著增强了Actor处理异构HV的能力。消融研究和与其他决策方法的对比表明，提出的钱Actor-Reasoner框架显著提高了安全性和效率。最后，结合来自Reasoner推理的外部人机接口（eHMI）信息和从Actor检索到的可行动作解决方案，在多种场景下的实地交互中验证了所提出的钱Actor-Reasoner的有效性。代码可在以下链接获取。 

---
# SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models 

**Title (ZH)**: SafeAuto：基于多模态基础模型的安全自主驾驶知识增强方法 

**Authors**: Jiawei Zhang, Xuan Yang, Taiqi Wang, Yu Yao, Aleksandr Petiushko, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00211)  

**Abstract**: Traditional autonomous driving systems often struggle to integrate high-level reasoning with low-level control, resulting in suboptimal and sometimes unsafe driving behaviors. The emergence of Multimodal Large Language Models (MLLMs), which can process both visual and textual data, presents an opportunity to unify perception and reasoning tasks within a single framework. However, effectively embedding precise safety knowledge into MLLMs for autonomous driving remains a significant challenge. To address this, we propose SafeAuto, a novel framework that enhances MLLM-based autonomous driving systems by incorporating both unstructured and structured knowledge. Specifically, we first introduce the Position-Dependent Cross-Entropy (PDCE) loss function, designed to improve the accuracy of low-level control signal predictions when numerical values are represented as text. Second, to ensure safe autonomous driving by explicitly integrating precise safety knowledge into the MLLM, we develop a reasoning component for SafeAuto. This component translates driving safety regulations into first-order logic rules (e.g., "red light => stop") and incorporates these rules into a probabilistic graphical model, such as a Markov Logic Network (MLN). The MLN is trained to verify the predicted next actions using environmental attributes identified by attribute recognition models (e.g., detecting a red light) to form the predicates. Additionally, we construct a Multimodal RAG model that leverages video data, control signals, and environmental attributes to learn more effectively from past similar driving experiences. By integrating PDCE, MLN, and Multimodal RAG, SafeAuto significantly outperforms existing baselines across multiple datasets. This advancement enables more accurate, reliable, and safer autonomous driving systems that learn from experience, obey traffic laws, and perform precise control actions. 

**Abstract (ZH)**: 传统自动驾驶系统往往难以将高层次推理与低层次控制相结合，导致行为效果不佳甚至存在安全隐患。多模态大型语言模型（MLLM）能够处理视觉和文本数据的出现，为统一感知和推理任务提供了一种可能。然而，将精确的安全知识有效地嵌入到MLLM中以实现自动驾驶仍是一个重大挑战。为解决这一问题，我们提出了SafeAuto框架，该框架通过结合未结构化和结构化知识来增强基于MLLM的自动驾驶系统。具体而言，我们首先引入了位置依赖交叉熵（PDCE）损失函数，旨在提高当数值以文本形式表示时低层次控制信号预测的准确性。其次，为确保通过明确整合精确的安全知识实现安全的自动驾驶，我们为SafeAuto开发了一个推理组件。该组件将驾驶安全规定转化为一阶逻辑规则（例如，“红灯=>停止”），并将这些规则融入概率图模型，如马尔科夫逻辑网络（MLN）。MLN利用属性识别模型识别的环境属性（例如，检测红灯）来构建谓词，并通过这些属性验证预测的下一步动作。此外，我们构建了一个多模态RAG模型，该模型利用视频数据、控制信号和环境属性从过去的类似驾驶经历中更有效地学习。通过整合PDCE、MLN和多模态RAG，SafeAuto在多个数据集上显著优于现有基线。这一进展使得能够实现更准确、可靠和安全的自动驾驶系统，这些系统能够从经验中学习、遵守交通法规并执行精确的控制动作。 

---
# AI and Semantic Communication for Infrastructure Monitoring in 6G-Driven Drone Swarms 

**Title (ZH)**: AI和语义通信在6G驱动的无人机群基础设施监控中的应用 

**Authors**: Tasnim Ahmed, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2503.00053)  

**Abstract**: The adoption of unmanned aerial vehicles to monitor critical infrastructure is gaining momentum in various industrial domains. Organizational imperatives drive this progression to minimize expenses, accelerate processes, and mitigate hazards faced by inspection personnel. However, traditional infrastructure monitoring systems face critical bottlenecks-5G networks lack the latency and reliability for large-scale drone coordination, while manual inspections remain costly and slow. We propose a 6G-enabled drone swarm system that integrates ultra-reliable, low-latency communications, edge AI, and semantic communication to automate inspections. By adopting LLMs for structured output and report generation, our framework is hypothesized to reduce inspection costs and improve fault detection speed compared to existing methods. 

**Abstract (ZH)**: 6G使能的无人机群系统：集成超可靠低延迟通信、边缘AI和语义通信的智能巡检技术 

---
# Do GFlowNets Transfer? Case Study on the Game of 24/42 

**Title (ZH)**: Do GFlowNets 转移？关于 24/42 游戏的案例研究 

**Authors**: Adesh Gupta, Abhinav Kumar, Mansi Gupta, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2503.01819)  

**Abstract**: Generating diverse solutions is key to human-like reasoning, yet autoregressive language models focus on single accurate responses, limiting creativity. GFlowNets optimize solution generation as a flow network, promising greater diversity. Our case study shows their limited zero-shot transferability by fine-tuning small and medium-sized large language models on the Game of 24 and testing them on the Game of 42 datasets. Results revealed that GFlowNets struggle to maintain solution diversity and accuracy, highlighting key limitations in their cross-task generalization and the need for future research in improved transfer learning capabilities. 

**Abstract (ZH)**: 生成多样化的解决方案是实现类人推理的关键，而自回归语言模型专注于单一准确响应，限制了其创造性。GFlowNets通过优化解决方案生成为流网络，有望实现更高的多样性。我们的案例研究通过在24点游戏和42点游戏数据集上微调小型和中型语言模型，并测试它们的表现，展示了GFlowNets在零样本迁移上的局限性。结果表明，GFlowNets难以保持解决方案的多样性和准确性，突显了其跨任务泛化中的关键局限，并强调了未来研究以提高迁移学习能力的重要性。 

---
# SAKE: Steering Activations for Knowledge Editing 

**Title (ZH)**: SAKE: 引导激活以进行知识编辑 

**Authors**: Marco Scialanga, Thibault Laugel, Vincent Grari, Marcin Detyniecki  

**Link**: [PDF](https://arxiv.org/pdf/2503.01751)  

**Abstract**: As Large Langue Models have been shown to memorize real-world facts, the need to update this knowledge in a controlled and efficient manner arises. Designed with these constraints in mind, Knowledge Editing (KE) approaches propose to alter specific facts in pretrained models. However, they have been shown to suffer from several limitations, including their lack of contextual robustness and their failure to generalize to logical implications related to the fact. To overcome these issues, we propose SAKE, a steering activation method that models a fact to be edited as a distribution rather than a single prompt. Leveraging Optimal Transport, SAKE alters the LLM behavior over a whole fact-related distribution, defined as paraphrases and logical implications. Several numerical experiments demonstrate the effectiveness of this method: SAKE is thus able to perform more robust edits than its existing counterparts. 

**Abstract (ZH)**: 基于知识编辑的方法在大型语言模型中的应用：利用最优运输方法进行事实导向的调整 

---
# Position: Don't use the CLT in LLM evals with fewer than a few hundred datapoints 

**Title (ZH)**: 位置：不要在数据点少于几百个的LLM评估中使用CLT。 

**Authors**: Sam Bowyer, Laurence Aitchison, Desi R. Ivanova  

**Link**: [PDF](https://arxiv.org/pdf/2503.01747)  

**Abstract**: Rigorous statistical evaluations of large language models (LLMs), including valid error bars and significance testing, are essential for meaningful and reliable performance assessment. Currently, when such statistical measures are reported, they typically rely on the Central Limit Theorem (CLT). In this position paper, we argue that while CLT-based methods for uncertainty quantification are appropriate when benchmarks consist of thousands of examples, they fail to provide adequate uncertainty estimates for LLM evaluations that rely on smaller, highly specialized benchmarks. In these small-data settings, we demonstrate that CLT-based methods perform very poorly, usually dramatically underestimating uncertainty (i.e. producing error bars that are too small). We give recommendations for alternative frequentist and Bayesian methods that are both easy to implement and more appropriate in these increasingly common scenarios. We provide a simple Python library for these Bayesian methods at this https URL . 

**Abstract (ZH)**: 大规模语言模型的严格统计评价，包括有效的误差范围和显著性测试，对于有意义和可靠的性能评估是必不可少的。目前，当报告此类统计度量时，通常依赖中心极限定理（CLT）。在本文中，我们argue认为，在基准数据包含数千个样本的情况下，基于CLT的不确定性量化方法是合适的，但对于依赖于较小且高度专门化基准的大型语言模型评估而言，它们无法提供充分的不确定性估计。在这些小样本数据设置中，我们证明基于CLT的方法表现得很差，通常极大地低估了不确定性（即产生过小的误差范围）。我们提出了替代的频繁主义和贝叶斯方法的建议，这些方法易于实施且更适用于这些越来越常见的场景。我们在此提供的简单Python库包含了这些贝叶斯方法。 

---
# Graph-Augmented Reasoning: Evolving Step-by-Step Knowledge Graph Retrieval for LLM Reasoning 

**Title (ZH)**: 图增强推理：逐步知识图谱检索在大模型推理中的 evolving 

**Authors**: Wenjie Wu, Yongcheng Jing, Yingjie Wang, Wenbin Hu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01642)  

**Abstract**: Recent large language model (LLM) reasoning, despite its success, suffers from limited domain knowledge, susceptibility to hallucinations, and constrained reasoning depth, particularly in small-scale models deployed in resource-constrained environments. This paper presents the first investigation into integrating step-wise knowledge graph retrieval with step-wise reasoning to address these challenges, introducing a novel paradigm termed as graph-augmented reasoning. Our goal is to enable frozen, small-scale LLMs to retrieve and process relevant mathematical knowledge in a step-wise manner, enhancing their problem-solving abilities without additional training. To this end, we propose KG-RAR, a framework centered on process-oriented knowledge graph construction, a hierarchical retrieval strategy, and a universal post-retrieval processing and reward model (PRP-RM) that refines retrieved information and evaluates each reasoning step. Experiments on the Math500 and GSM8K benchmarks across six models demonstrate that KG-RAR yields encouraging results, achieving a 20.73\% relative improvement with Llama-3B on Math500. 

**Abstract (ZH)**: 近期大型语言模型（LLM）的推理尽管取得成功，但仍面临领域知识有限、容易产生幻觉以及推理深度受限等问题，特别是在资源受限环境中部署的小规模模型中更为明显。本文首次探讨将逐步知识图谱检索与逐步推理相结合以应对这些挑战，提出了一个名为图增强推理的新范式。我们的目标是使冻结的小规模LLM能够逐步检索和处理相关数学知识，增强其解决问题的能力而无需额外训练。为此，我们提出了KG-RAR框架，该框架以过程导向的知识图谱构建、层次化检索策略以及一种通用的检索后处理和奖励模型（PRP-RM）为核心，该模型对检索到的信息进行精炼并评估每一步推理。实验结果显示，KG-RAR在Math500和GSM8K基准测试中的六种模型上取得了令人鼓舞的结果，相较于Llama-3B在Math500上的表现提高了20.73%。 

---
# OptMetaOpenFOAM: Large Language Model Driven Chain of Thought for Sensitivity Analysis and Parameter Optimization based on CFD 

**Title (ZH)**: OptMetaOpenFOAM：基于CFD的大型语言模型驱动的灵敏度分析与参数优化思维链方法 

**Authors**: Yuxuan Chen, Long Zhang, Xu Zhu, Hua Zhou, Zhuyin Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.01273)  

**Abstract**: Merging natural language interfaces with computational fluid dynamics (CFD) workflows presents transformative opportunities for both industry and research. In this study, we introduce OptMetaOpenFOAM - a novel framework that bridges MetaOpenFOAM with external analysis and optimization tool libraries through a large language model (LLM)-driven chain-of-thought (COT) methodology. By automating complex CFD tasks via natural language inputs, the framework empowers non-expert users to perform sensitivity analyses and parameter optimizations with markedly improved efficiency. The test dataset comprises 11 distinct CFD analysis or optimization tasks, including a baseline simulation task derived from an OpenFOAM tutorial covering fluid dynamics, combustion, and heat transfer. Results confirm that OptMetaOpenFOAM can accurately interpret user requirements expressed in natural language and effectively invoke external tool libraries alongside MetaOpenFOAM to complete the tasks. Furthermore, validation on a non-OpenFOAM tutorial case - namely, a hydrogen combustion chamber - demonstrates that a mere 200-character natural language input can trigger a sequence of simulation, postprocessing, analysis, and optimization tasks spanning over 2,000 lines of code. These findings underscore the transformative potential of LLM-driven COT methodologies in linking external tool for advanced analysis and optimization, positioning OptMetaOpenFOAM as an effective tool that streamlines CFD simulations and enhances their convenience and efficiency for both industrial and research applications. Code is available at this https URL. 

**Abstract (ZH)**: 将自然语言接口与计算流体动力学（CFD）工作流融合为工业和研究带来了变革性的机会。本研究介绍了OptMetaOpenFOAM ——一种通过大型语言模型（LLM）驱动的思维链（COT）方法将MetaOpenFOAM与外部分析和优化工具库连接起来的新型框架。通过自然语言输入自动化复杂CFD任务，该框架使非专家用户能够进行敏感性分析和参数优化，显著提高效率。测试数据集包含11项不同的CFD分析或优化任务，包括一个源自OpenFOAM教程的基本模拟任务，涵盖流体力学、燃烧和传热。结果证实，OptMetaOpenFOAM能够准确解释用户用自然语言表达的需求，并有效调用与MetaOpenFOAM协同工作的外部工具库来完成任务。此外，对一个非OpenFOAM教程案例——氢燃烧室——的验证表明，仅200字符的自然语言输入可以触发超过2000行代码的模拟、后处理、分析和优化任务序列。这些发现强调了LLM驱动的COT方法在连接外部工具以进行高级分析和优化方面的变革潜力，将OptMetaOpenFOAM定位为一种有效工具，可简化CFD模拟并提高其在工业和研究应用中的便利性和效率。代码可在以下链接获取。 

---
# Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers 

**Title (ZH)**: 基于多臂老虎机的提示设计策略选择改进提示优化器 

**Authors**: Rin Ashizawa, Yoichi Hirose, Nozomu Yoshinari, Kento Uchida, Shinichi Shirakawa  

**Link**: [PDF](https://arxiv.org/pdf/2503.01163)  

**Abstract**: Prompt optimization aims to search for effective prompts that enhance the performance of large language models (LLMs). Although existing prompt optimization methods have discovered effective prompts, they often differ from sophisticated prompts carefully designed by human experts. Prompt design strategies, representing best practices for improving prompt performance, can be key to improving prompt optimization. Recently, a method termed the Autonomous Prompt Engineering Toolbox (APET) has incorporated various prompt design strategies into the prompt optimization process. In APET, the LLM is needed to implicitly select and apply the appropriate strategies because prompt design strategies can have negative effects. This implicit selection may be suboptimal due to the limited optimization capabilities of LLMs. This paper introduces Optimizing Prompts with sTrategy Selection (OPTS), which implements explicit selection mechanisms for prompt design. We propose three mechanisms, including a Thompson sampling-based approach, and integrate them into EvoPrompt, a well-known prompt optimizer. Experiments optimizing prompts for two LLMs, Llama-3-8B-Instruct and GPT-4o mini, were conducted using BIG-Bench Hard. Our results show that the selection of prompt design strategies improves the performance of EvoPrompt, and the Thompson sampling-based mechanism achieves the best overall results. Our experimental code is provided at this https URL . 

**Abstract (ZH)**: 优化提示策略选择以提升大型语言模型性能 

---
# Can Large Language Models Help Experimental Design for Causal Discovery? 

**Title (ZH)**: 大语言模型能否辅助实验设计以进行因果发现？ 

**Authors**: Junyi Li, Yongqiang Chen, Chenxi Liu, Qianyi Cai, Tongliang Liu, Bo Han, Kun Zhang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01139)  

**Abstract**: Designing proper experiments and selecting optimal intervention targets is a longstanding problem in scientific or causal discovery. Identifying the underlying causal structure from observational data alone is inherently this http URL interventional data, on the other hand, is crucial to causal discovery, yet it is usually expensive and time-consuming to gather sufficient interventional data to facilitate causal this http URL approaches commonly utilize uncertainty or gradient signals to determine the intervention targets. However, numerical-based approaches may yield suboptimal results due to the inaccurate estimation of the guiding signals at the beginning when with limited interventional data. In this work, we investigate a different approach, whether we can leverage Large Language Models (LLMs) to assist with the intervention targeting in causal discovery by making use of the rich world knowledge about the experimental design in this http URL, we present \oursfull (\ours) -- a robust framework that effectively incorporates LLMs to augment existing numerical approaches for the intervention targeting in causal discovery. Across $4$ realistic benchmark scales, \ours demonstrates significant improvements and robustness over existing methods and even surpasses humans, which demonstrates the usefulness of LLMs in assisting with experimental design for scientific discovery. 

**Abstract (ZH)**: 设计合适的实验并选择最优的干预目标是科学发现或因果发现中的长期问题。仅从观察数据中识别潜在的因果结构本身具有挑战性，而干预数据对于因果发现至关重要，但由于收集足够的干预数据通常代价高昂且耗时，因此常见的方法通常利用不确定性或梯度信号来确定干预目标。然而，基于数值的方法可能会因为有限的干预数据及初始时引导信号的不准确估计而产生次优的结果。在本工作中，我们探讨了一种不同的方法，即我们是否可以利用大语言模型（LLMs）通过利用丰富的实验设计世界知识来协助因果发现中的干预目标选择。为此，我们提出了\oursfull（\ours）——一个稳健的框架，该框架有效结合了LLMs以增强现有的数值方法在因果发现中的干预目标选择。在4个现实基准规模中，\ours 在与现有方法相比时表现出显著的改进和鲁棒性，甚至超过了人类，这表明了LLMs在协助科学发现中的实验设计方面的有效性。 

---
# Evidence of conceptual mastery in the application of rules by Large Language Models 

**Title (ZH)**: 大型语言模型在规则应用中概念掌握的证据 

**Authors**: José Luiz Nunes, Guilherme FCF Almeida, Brian Flanagan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00992)  

**Abstract**: In this paper we leverage psychological methods to investigate LLMs' conceptual mastery in applying rules. We introduce a novel procedure to match the diversity of thought generated by LLMs to that observed in a human sample. We then conducted two experiments comparing rule-based decision-making in humans and LLMs. Study 1 found that all investigated LLMs replicated human patterns regardless of whether they are prompted with scenarios created before or after their training cut-off. Moreover, we found unanticipated differences between the two sets of scenarios among humans. Surprisingly, even these differences were replicated in LLM responses. Study 2 turned to a contextual feature of human rule application: under forced time delay, human samples rely more heavily on a rule's text than on other considerations such as a rule's purpose.. Our results revealed that some models (Gemini Pro and Claude 3) responded in a human-like manner to a prompt describing either forced delay or time pressure, while others (GPT-4o and Llama 3.2 90b) did not. We argue that the evidence gathered suggests that LLMs have mastery over the concept of rule, with implications for both legal decision making and philosophical inquiry. 

**Abstract (ZH)**: 本文利用心理学方法探讨大语言模型在应用规则方面的概念掌握程度。我们引入了一种新的程序，将大语言模型生成的多元想法与人类样本中的观察多样性进行匹配。随后，我们进行了两项实验，比较了人类和大语言模型基于规则的决策制定。研究1发现，所有研究的大语言模型在处理训练截止前后的场景描述时，均再现了人类的模式。此外，我们还在人类两组场景中发现了未预期的差异，这些差异甚至在大语言模型的响应中也得到了再现。研究2关注了人类应用规则的上下文特征：在被迫的时间延迟条件下，人类样本更依赖规则的文字内容，而非其他考虑因素如规则的目的。我们的结果表明，一些模型（Gemini Pro和Claude 3）对描述被迫延迟或时间压力的提示响应得像人类一样，而其他模型（GPT-4o和Llama 3.2 90b）则不然。我们认为收集的证据表明大语言模型对规则的概念有掌握，这对法律决策和哲学探究均有重要意义。 

---
# A Law Reasoning Benchmark for LLM with Tree-Organized Structures including Factum Probandum, Evidence and Experiences 

**Title (ZH)**: 基于树组织结构的证据推理基准测试：包括事证、证据和经验的法律推理 

**Authors**: Jiaxin Shen, Jinan Xu, Huiqi Hu, Luyi Lin, Fei Zheng, Guoyang Ma, Fandong Meng, Jie Zhou, Wenjuan Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.00841)  

**Abstract**: While progress has been made in legal applications, law reasoning, crucial for fair adjudication, remains unexplored. We propose a transparent law reasoning schema enriched with hierarchical factum probandum, evidence, and implicit experience, enabling public scrutiny and preventing bias. Inspired by this schema, we introduce the challenging task, which takes a textual case description and outputs a hierarchical structure justifying the final decision. We also create the first crowd-sourced dataset for this task, enabling comprehensive evaluation. Simultaneously, we propose an agent framework that employs a comprehensive suite of legal analysis tools to address the challenge task. This benchmark paves the way for transparent and accountable AI-assisted law reasoning in the ``Intelligent Court''. 

**Abstract (ZH)**: 虽然在法律应用方面取得了一定进展，但对于公平判决至关重要的法律推理仍有待探索。我们提出了一种透明的法律推理框架，其中包含层次化的事实主张、证据和隐含经验，以供公众审查并防止偏见。受此框架启发，我们引入了一个具有挑战性的任务，该任务接受文本案件描述，并输出一个支持最终判决的层次结构。我们还为此任务创建了首个众包数据集，以实现全面评估。同时，我们提出了一种代理框架，利用全面的法律分析工具来应对这一挑战任务。该基准为“智能法庭”中的透明和问责制AI辅助法律推理铺平了道路。 

---
# Instructor-Worker Large Language Model System for Policy Recommendation: a Case Study on Air Quality Analysis of the January 2025 Los Angeles Wildfires 

**Title (ZH)**: 面向政策推荐的教师-工人大型语言模型系统：以2025年1月洛杉矶野火空气质量分析案例如谈 

**Authors**: Kyle Gao, Dening Lu, Liangzhi Li, Nan Chen, Hongjie He, Linlin Xu, Jonathan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00566)  

**Abstract**: The Los Angeles wildfires of January 2025 caused more than 250 billion dollars in damage and lasted for nearly an entire month before containment. Following our previous work, the Digital Twin Building, we modify and leverage the multi-agent large language model framework as well as the cloud-mapping integration to study the air quality during the Los Angeles wildfires. Recent advances in large language models have allowed for out-of-the-box automated large-scale data analysis. We use a multi-agent large language system comprised of an Instructor agent and Worker agents. Upon receiving the users' instructions, the Instructor agent retrieves the data from the cloud platform and produces instruction prompts to the Worker agents. The Worker agents then analyze the data and provide summaries. The summaries are finally input back into the Instructor agent, which then provides the final data analysis. We test this system's capability for data-based policy recommendation by assessing our Instructor-Worker LLM system's health recommendations based on air quality during the Los Angeles wildfires. 

**Abstract (ZH)**: 2025年1月洛杉矶野火造成的超过2500亿美元损失并持续近一个月才得到控制。在我们之前工作的基础上，我们修改并利用多代理大型语言模型框架以及云地图集成来研究洛杉矶野火期间的空气质量。大型语言模型的最新进展使自动化大规模数据分析成为可能。我们使用由Instructor代理和Worker代理组成的多代理大型语言系统。在收到用户指令后，Instructor代理从云平台检索数据并生成指令提示给Worker代理。Worker代理随后分析数据并提供总结。总结最终输入回到Instructor代理，后者提供最终的数据分析。我们通过评估Instructor-Worker LLM系统在洛杉矶野火期间空气质量基础上的健康建议来测试该系统基于数据的政策建议能力。 

---
# Jailbreaking Safeguarded Text-to-Image Models via Large Language Models 

**Title (ZH)**: 通过大语言模型保障文本到图像模型的安全破解 

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01839)  

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks. 

**Abstract (ZH)**: 基于文本到图像模型的安全防护突破方法：PromptTune 

---
# CrowdSelect: Synthetic Instruction Data Selection with Multi-LLM Wisdom 

**Title (ZH)**: CrowdSelect: 多大语言模型智慧合成指令数据选择 

**Authors**: Yisen Li, Lingfeng Yang, Wenxuan Shen, Pan Zhou, Yao Wan, Weiwei Lin, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01836)  

**Abstract**: Distilling advanced Large Language Models' instruction-following capabilities into smaller models using a selected subset has become a mainstream approach in model training. While existing synthetic instruction data selection strategies rely mainly on single-dimensional signals (i.e., reward scores, model perplexity), they fail to capture the complexity of instruction-following across diverse fields. Therefore, we investigate more diverse signals to capture comprehensive instruction-response pair characteristics and propose three foundational metrics that leverage Multi-LLM wisdom, informed by (1) diverse LLM responses and (2) reward model assessment. Building upon base metrics, we propose CrowdSelect, an integrated metric incorporating a clustering-based approach to maintain response diversity. Our comprehensive experiments demonstrate that our foundation metrics consistently improve performance across 4 base models on MT-bench and Arena-Hard. CrowdSelect, efficiently incorporating all metrics, achieves state-of-the-art performance in both Full and LoRA fine-tuning, showing improvements of 4.81% on Arena-Hard and 11.1% on MT-bench with Llama-3.2-3b-instruct. We hope our findings will bring valuable insights for future research in this direction. Code are available at this https URL. 

**Abstract (ZH)**: 使用选定子集将先进大型语言模型的指令遵循能力精简到较小模型已成为模型训练中的主流方法。现有的合成指令数据选择策略主要依赖单一维度的信号（如奖励分数、模型困惑度），但无法捕捉不同领域指令遵循的复杂性。因此，我们研究了更加多样的信号以捕捉全面的指令-响应对特征，并提出了三种基于多LLM智慧的基本度量，这些度量受到了（1）多样化LLM响应和（2）奖励模型评估的启发。在此基础上，我们提出了CrowdSelect综合度量，该度量结合了基于聚类的方法以保持响应多样性。我们的全面实验表明，我们的基础度量在4个基模型上均能跨MT-bench和Arena-Hard任务提升性能。CrowdSelect通过高效结合所有度量，在全量和LoRA微调中均达到了最佳性能，分别在Llama-3.2-3b-instruct上取得了Arena-Hard任务4.81%和MT-bench任务11.1%的性能提升。希望我们的发现能为未来的研究提供有价值的见解。代码可在以下链接获取。 

---
# Persuade Me if You Can: A Framework for Evaluating Persuasion Effectiveness and Susceptibility Among Large Language Models 

**Title (ZH)**: 说服我如果可以：评估大型语言模型说服效果与易感性框架 

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2503.01829)  

**Abstract**: Large Language Models (LLMs) demonstrate persuasive capabilities that rival human-level persuasion. While these capabilities can be used for social good, they also present risks of potential misuse. Moreover, LLMs' susceptibility to persuasion raises concerns about alignment with ethical principles. To study these dynamics, we introduce Persuade Me If You Can (PMIYC), an automated framework for evaluating persuasion through multi-agent interactions. Here, Persuader agents engage in multi-turn conversations with the Persuadee agents, allowing us to measure LLMs' persuasive effectiveness and their susceptibility to persuasion. We conduct comprehensive evaluations across diverse LLMs, ensuring each model is assessed against others in both subjective and misinformation contexts. We validate the efficacy of our framework through human evaluations and show alignment with prior work. PMIYC offers a scalable alternative to human annotation for studying persuasion in LLMs. Through PMIYC, we find that Llama-3.3-70B and GPT-4o exhibit similar persuasive effectiveness, outperforming Claude 3 Haiku by 30%. However, GPT-4o demonstrates over 50% greater resistance to persuasion for misinformation compared to Llama-3.3-70B. These findings provide empirical insights into the persuasive dynamics of LLMs and contribute to the development of safer AI systems. 

**Abstract (ZH)**: Large Language Models (LLMs)展现的人类级别的说服能力与其潜在的滥用风险并存：一个自动化的多代理交互框架（PMIYC）用于评估说服力及其抗说服性 

---
# RSQ: Learning from Important Tokens Leads to Better Quantized LLMs 

**Title (ZH)**: RSQ: 通过学习重要 token 提升量化大语言模型性能 

**Authors**: Yi-Lin Sung, Prateek Yadav, Jialu Li, Jaehong Yoon, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.01820)  

**Abstract**: Layer-wise quantization is a key technique for efficiently compressing large models without expensive retraining. Previous methods typically quantize the weights of each layer by "uniformly" optimizing the layer reconstruction loss across all output tokens. However, in this paper, we demonstrate that better-quantized models can be obtained by prioritizing learning from important tokens (e.g. which have large attention scores). Building on this finding, we propose RSQ (Rotate, Scale, then Quantize), which (1) applies rotations (orthogonal transformation) to the model to mitigate outliers (those with exceptionally large magnitude), (2) scales the token feature based on its importance, and (3) quantizes the model using the GPTQ framework with the second-order statistics computed by scaled tokens. To compute token importance, we explore both heuristic and dynamic strategies. Based on a thorough analysis of all approaches, we adopt attention concentration, which uses attention scores of each token as its importance, as the best approach. We demonstrate that RSQ consistently outperforms baseline methods across multiple downstream tasks and three model families: LLaMA3, Mistral, and Qwen2.5. Additionally, models quantized with RSQ achieve superior performance on long-context tasks, further highlighting its effectiveness. Lastly, RSQ demonstrates generalizability across various setups, including different model sizes, calibration datasets, bit precisions, and quantization methods. 

**Abstract (ZH)**: 逐层量化是一种高效压缩大型模型的关键技术，无需昂贵的重新训练。传统方法通常通过在所有输出标记上均匀优化层重构损失来量化每一层的权重。然而，本文表明，通过优先学习重要标记（例如，具有较大注意力分数的标记）可以获得更好的量化模型。在此基础上，我们提出了RSQ（Rotate, Scale, then Quantize），它包括：（1）应用旋转（正交变换）以减轻异常值的影响（那些具有异常大 magnitude 的标记），（2）根据标记的重要性对其进行标定，（3）使用带有按标定标记计算的二阶统计量的 GPTQ 框架对模型进行量化。为了计算标记的重要性，我们探索了启发式和动态策略。根据所有方法的综合分析，我们采用基于注意力集中度的方法，将每个标记的注意力分数作为其重要性，作为最佳方法。我们证明，RSQ 在多个下游任务和三种模型家族（LLaMA3、Mistral 和 Qwen2.5）中始终优于基线方法。此外，使用 RSQ 量化后的模型在长语境任务中表现出更优的性能，进一步突显其有效性。最后，RSQ 在不同设置下展示了泛化能力，包括不同的模型大小、校准数据集、位精度和量化方法。 

---
# LLMInit: A Free Lunch from Large Language Models for Selective Initialization of Recommendation 

**Title (ZH)**: LLMInit：从大型语言模型获得的免费午餐——用于推荐的选择性初始化 

**Authors**: Weizhi Zhang, Liangwei Yang, Wooseong Yang, Henry Peng Zou, Yuqing Liu, Ke Xu, Sourav Medya, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01814)  

**Abstract**: Collaborative filtering models, particularly graph-based approaches, have demonstrated strong performance in capturing user-item interactions for recommendation systems. However, they continue to struggle in cold-start and data-sparse scenarios. The emergence of large language models (LLMs) like GPT and LLaMA presents new possibilities for enhancing recommendation performance, especially in cold-start settings. Despite their promise, LLMs pose challenges related to scalability and efficiency due to their high computational demands and limited ability to model complex user-item relationships effectively. In this work, we introduce a novel perspective on leveraging LLMs for CF model initialization. Through experiments, we uncover an embedding collapse issue when scaling CF models to larger embedding dimensions. To effectively harness large-scale LLM embeddings, we propose innovative selective initialization strategies utilizing random, uniform, and variance-based index sampling. Our comprehensive evaluation on multiple real-world datasets demonstrates significant performance gains across various CF models while maintaining a lower computational cost compared to existing LLM-based recommendation approaches. 

**Abstract (ZH)**: 基于图的合作过滤模型在捕捉用户项交互方面表现出强大的性能，但在冷启动和数据稀疏场景中仍面临挑战。大型语言模型（LLMs）如GPT和LLaMA的出现为提升推荐性能提供了新可能，尤其是在冷启动设置中。尽管LLMs前景广阔，但高计算需求和建模复杂用户项关系能力有限导致的可扩展性和效率问题依然存在。在本文中，我们提出了一种创新的利用LLMs初始化CF模型的新视角。通过实验，我们揭示了当将CF模型扩展到更高嵌入维度时存在的嵌入塌缩问题。为了有效利用大规模LLM嵌入，我们提出了基于随机、均匀和方差采样的创新性选择性初始化策略。我们的全面评估在多个真实世界数据集上显示，在保持较低计算成本的同时，CF模型的性能显著提升。 

---
# AutoAdvExBench: Benchmarking autonomous exploitation of adversarial example defenses 

**Title (ZH)**: AutoAdvExBench: 自动化利用 adversarial example 防御的技术评估 

**Authors**: Nicholas Carlini, Javier Rando, Edoardo Debenedetti, Milad Nasr, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2503.01811)  

**Abstract**: We introduce AutoAdvExBench, a benchmark to evaluate if large language models (LLMs) can autonomously exploit defenses to adversarial examples. Unlike existing security benchmarks that often serve as proxies for real-world tasks, bench directly measures LLMs' success on tasks regularly performed by machine learning security experts. This approach offers a significant advantage: if a LLM could solve the challenges presented in bench, it would immediately present practical utility for adversarial machine learning researchers. We then design a strong agent that is capable of breaking 75% of CTF-like ("homework exercise") adversarial example defenses. However, we show that this agent is only able to succeed on 13% of the real-world defenses in our benchmark, indicating the large gap between difficulty in attacking "real" code, and CTF-like code. In contrast, a stronger LLM that can attack 21% of real defenses only succeeds on 54% of CTF-like defenses. We make this benchmark available at this https URL. 

**Abstract (ZH)**: 我们介绍AutoAdvExBench，这是一个基准，用于评估大型语言模型（LLMs）是否能够自主利用对抗例子的防御措施。 

---
# $\texttt{SEM-CTRL}$: Semantically Controlled Decoding 

**Title (ZH)**: SEM-CTRL: 语义控制解码 

**Authors**: Mohammad Albinhassan, Pranava Madhyastha, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01804)  

**Abstract**: Ensuring both syntactic and semantic correctness in Large Language Model (LLM) outputs remains a significant challenge, despite being critical for real-world deployment. In this paper, we introduce $\texttt{SEM-CTRL}$, a unified approach that enforces rich context-sensitive constraints and task- and instance-specific semantics directly on an LLM decoder. Our approach integrates token-level MCTS, which is guided by specific syntactic and semantic constraints. The constraints over the desired outputs are expressed using Answer Set Grammars -- a logic-based formalism that generalizes context-sensitive grammars while incorporating background knowledge to represent task-specific semantics. We show that our approach guarantees correct completions for any off-the-shelf LLM without the need for fine-tuning. We evaluate $\texttt{SEM-CTRL}$ on a range of tasks, including synthetic grammar synthesis, combinatorial reasoning, and planning. Our results demonstrate that $\texttt{SEM-CTRL}$ allows small pre-trained LLMs to efficiently outperform larger variants and state-of-the-art reasoning models (e.g., o1-preview) while simultaneously guaranteeing solution correctness. 

**Abstract (ZH)**: 确保大型语言模型（LLM）输出在句法和语义上的正确性仍然是一个重大挑战，尽管这对于实际部署至关重要。本文介绍了一种统一的方法$\texttt{SEM-CTRL}$，该方法直接在LLM解码器上施加丰富的上下文敏感约束和任务及实例特定的语义。我们的方法结合了受特定句法和语义约束指导的令牌级别MCTS。所需的输出约束使用回答集文法表达——这是一种基于逻辑的形式主义，可以泛化上下文敏感文法并结合背景知识来表示任务特定的语义。我们证明了我们的方法可以在不需要微调的情况下保证任何现成的LLM生成正确的完成。我们在合成语法合成、组合推理和规划等多种任务上评估了$\texttt{SEM-CTRL}$。结果表明，$\texttt{SEM-CTRL}$允许小型预训练LLM高效地超越更大规模的变体和最先进的推理模型（例如o1-preview），同时同时保证解决方案的正确性。 

---
# Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models 

**Title (ZH)**: 检索模型不是工具高手：大型语言模型工具检索基准研究 

**Authors**: Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.01763)  

**Abstract**: Tool learning aims to augment large language models (LLMs) with diverse tools, enabling them to act as agents for solving practical tasks. Due to the limited context length of tool-using LLMs, adopting information retrieval (IR) models to select useful tools from large toolsets is a critical initial step. However, the performance of IR models in tool retrieval tasks remains underexplored and unclear. Most tool-use benchmarks simplify this step by manually pre-annotating a small set of relevant tools for each task, which is far from the real-world scenarios. In this paper, we propose ToolRet, a heterogeneous tool retrieval benchmark comprising 7.6k diverse retrieval tasks, and a corpus of 43k tools, collected from existing datasets. We benchmark six types of models on ToolRet. Surprisingly, even the models with strong performance in conventional IR benchmarks, exhibit poor performance on ToolRet. This low retrieval quality degrades the task pass rate of tool-use LLMs. As a further step, we contribute a large-scale training dataset with over 200k instances, which substantially optimizes the tool retrieval ability of IR models. 

**Abstract (ZH)**: 工具学习旨在通过多样化工具增强大型语言模型（LLMs），使其能够作为解决实际任务的代理。由于工具使用型LLMs的上下文长度限制，采用信息检索（IR）模型从大型工具集选择有用工具是至关重要的初步步骤。然而，IR模型在工具检索任务中的性能仍缺乏探索和明确说明。大多数工具使用基准通过手动预标注每个任务的小数量相关工具来简化这一步骤，这远不符合真实世界的情景。在本文中，我们提出了一个异构工具检索基准ToolRet，包含了7600个多样化的检索任务和一个包含43000个工具的语料库，这些工具是从现有数据集中收集的。我们在ToolRet上对六种类型的模型进行了基准测试。令人惊讶的是，即使在传统信息检索基准中表现出色的模型，在ToolRet上的表现也较差。这种低质量的检索降低了工具使用LLMs的任务通过率。为进一步改进，我们贡献了一个包含超过20万个实例的大规模训练数据集，显著优化了IR模型的工具检索能力。 

---
# Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs 

**Title (ZH)**: Phi-4-Mini 技术报告：通过混合小型LoRA实现紧凑而强大的多模态语言模型 

**Authors**: Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi-ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, Mei Gao, Min Gao, Amit Garg, Abhishek Goswami, Junheng Hao, Amr Hendy, Yuxuan Hu, Xin Jin, Mahmoud Khademi, Dongwoo Kim, Young Jin Kim, Gina Lee, Jinyu Li, Yunsheng Li, Chen Liang, Xihui Lin, Zeqi Lin, Mengchen Liu, Yang Liu, Gilsinia Lopez, Chong Luo, Piyush Madan, Vadim Mazalov, Ali Mousavi, Anh Nguyen, Jing Pan, Daniel Perez-Becker, Jacob Platin, Thomas Portet, Kai Qiu, Bo Ren, Liliang Ren, Sambuddha Roy, Ning Shang, Yelong Shen, Saksham Singhal, Subhojit Som, Xia Song, Tetyana Sych, Praneetha Vaddamanu, Shuohang Wang, Yiming Wang, Zhenghao Wang, Haibin Wu, Haoran Xu, Weijian Xu, Yifan Yang, Ziyi Yang, Donghan Yu, Ishmam Zabir, Jianwen Zhang, Li Lyna Zhang, Yunan Zhang, Xiren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01743)  

**Abstract**: We introduce Phi-4-Mini and Phi-4-Multimodal, compact yet highly capable language and multimodal models. Phi-4-Mini is a 3.8-billion-parameter language model trained on high-quality web and synthetic data, significantly outperforming recent open-source models of similar size and matching the performance of models twice its size on math and coding tasks requiring complex reasoning. This achievement is driven by a carefully curated synthetic data recipe emphasizing high-quality math and coding datasets. Compared to its predecessor, Phi-3.5-Mini, Phi-4-Mini features an expanded vocabulary size of 200K tokens to better support multilingual applications, as well as group query attention for more efficient long-sequence generation. Phi-4-Multimodal is a multimodal model that integrates text, vision, and speech/audio input modalities into a single model. Its novel modality extension approach leverages LoRA adapters and modality-specific routers to allow multiple inference modes combining various modalities without interference. For example, it now ranks first in the OpenASR leaderboard to date, although the LoRA component of the speech/audio modality has just 460 million parameters. Phi-4-Multimodal supports scenarios involving (vision + language), (vision + speech), and (speech/audio) inputs, outperforming larger vision-language and speech-language models on a wide range of tasks. Additionally, we experiment to further train Phi-4-Mini to enhance its reasoning capabilities. Despite its compact 3.8-billion-parameter size, this experimental version achieves reasoning performance on par with or surpassing significantly larger models, including DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B. 

**Abstract (ZH)**: 我们介绍了Phi-4-Mini和Phi-4-Multimodal，这两种模型虽然紧凑但功能强大，具备语言和多模态能力。Phi-4-Mini是基于高质量网络和合成数据训练的38亿参数语言模型，显著优于同类规模的开源模型，并在需要复杂推理的数学和编程任务上达到与两倍规模模型相当的性能。这一成就得益于精心策划的合成数据配方，强调高质量的数学和编程数据集。与 predecessor Phi-3.5-Mini 相比，Phi-4-Mini 的词汇量扩大到 20 万 tokens，以更好地支持多语言应用，并集成了组查询注意机制，提高长序列生成的效率。Phi-4-Multimodal 是一个整合了文本、视觉和语音/音频输入模态的多模态模型。其新颖的模态扩展方法利用 LoRA 调用器和特定模态路由器，允许多种结合不同模态的推理模式而无干扰。例如，它目前在 OpenASR 领导板上排名第一，尽管语音/音频模态的 LoRA 组件只有 4.6 亿参数。Phi-4-Multimodal 支持涉及 (视觉 + 语言)、(视觉 + 语音) 和 (语音/音频) 输入的场景，在多种任务上超越更大规模的视觉语言和语音语言模型。此外，我们尝试进一步训练 Phi-4-Mini 以增强其推理能力。尽管其紧凑的 38 亿参数规模，此实验版本在推理性能上与或超过了显著更大的模型，包括 DeepSeek-R1-Distill-Qwen-7B 和 DeepSeek-R1-Distill-Llama-8B。 

---
# Word Form Matters: LLMs' Semantic Reconstruction under Typoglycemia 

**Title (ZH)**: 字形 Matters： Typoglycemia 下 LLMs 的语义重构 

**Authors**: Chenxi Wang, Tianle Gu, Zhongyu Wei, Lang Gao, Zirui Song, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01714)  

**Abstract**: Human readers can efficiently comprehend scrambled words, a phenomenon known as Typoglycemia, primarily by relying on word form; if word form alone is insufficient, they further utilize contextual cues for interpretation. While advanced large language models (LLMs) exhibit similar abilities, the underlying mechanisms remain unclear. To investigate this, we conduct controlled experiments to analyze the roles of word form and contextual information in semantic reconstruction and examine LLM attention patterns. Specifically, we first propose SemRecScore, a reliable metric to quantify the degree of semantic reconstruction, and validate its effectiveness. Using this metric, we study how word form and contextual information influence LLMs' semantic reconstruction ability, identifying word form as the core factor in this process. Furthermore, we analyze how LLMs utilize word form and find that they rely on specialized attention heads to extract and process word form information, with this mechanism remaining stable across varying levels of word scrambling. This distinction between LLMs' fixed attention patterns primarily focused on word form and human readers' adaptive strategy in balancing word form and contextual information provides insights into enhancing LLM performance by incorporating human-like, context-aware mechanisms. 

**Abstract (ZH)**: 人类读者可以通过单词形式高效理解乱序单词，这一现象称为Typoglycemia；若仅依靠单词形式不足以进行解释，则会利用上下文线索。虽然高级大型语言模型（LLMs）展现出类似能力，但其内在机制尚不清楚。为探究此问题，我们进行了控制实验，分析单词形式和上下文信息在语义重建中的作用，并考察LLM的注意力模式。具体而言，我们首先提出了一种可靠的SemRecScore度量方法来量化语义重建的程度，并验证其有效性。利用该度量方法，我们研究了单词形式和上下文信息如何影响LLM的语义重建能力，发现单词形式是这一过程的核心因素。此外，我们分析了LLMs如何利用单词形式，并发现它们依赖于特定的注意力头来提取和处理单词形式信息，这一机制在不同水平的单词乱序中保持稳定。LLMs固定关注单词形式的注意力模式与人类读者适应性地平衡单词形式和上下文信息的策略之间的差异，为我们提供了通过引入类似人类的、上下文感知的机制来提升LLM性能的见解。 

---
# SAGE: A Framework of Precise Retrieval for RAG 

**Title (ZH)**: SAGE: 一种精确检索框架用于RAG 

**Authors**: Jintao Zhang, Guoliang Li, Jinyang Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.01713)  

**Abstract**: Retrieval-augmented generation (RAG) has demonstrated significant proficiency in conducting question-answering (QA) tasks within a specified corpus. Nonetheless, numerous failure instances of RAG in QA still exist. These failures are not solely attributable to the limitations of Large Language Models (LLMs); instead, they predominantly arise from the retrieval of inaccurate information for LLMs due to two limitations: (1) Current RAG methods segment the corpus without considering semantics, making it difficult to find relevant context due to impaired correlation between questions and the segments. (2) There is a trade-off between missing essential context with fewer context retrieved and getting irrelevant context with more context retrieved.
In this paper, we introduce a RAG framework (SAGE), to overcome these limitations. First, to address the segmentation issue without considering semantics, we propose to train a semantic segmentation model. This model is trained to segment the corpus into semantically complete chunks. Second, to ensure that only the most relevant chunks are retrieved while the irrelevant ones are ignored, we design a chunk selection algorithm to dynamically select chunks based on the decreasing speed of the relevance score, leading to a more relevant selection. Third, to further ensure the precision of the retrieved chunks, we propose letting LLMs assess whether retrieved chunks are excessive or lacking and then adjust the amount of context accordingly. Experiments show that SAGE outperforms baselines by 61.25% in the quality of QA on average. Moreover, by avoiding retrieving noisy context, SAGE lowers the cost of the tokens consumed in LLM inference and achieves a 49.41% enhancement in cost efficiency on average. Additionally, our work offers valuable insights for boosting RAG. 

**Abstract (ZH)**: 检索增强生成（RAG）已经在特定语料库内的问答（QA）任务中展现了显著的能力。然而，RAG在QA中的众多失败实例依然存在。这些失败不仅归因于大型语言模型（LLMs）的局限性，还主要源于检索不准确信息导致的两个限制：（1）当前的RAG方法在分段时未考虑语义，导致问题与分段之间的关联性受损，难以找到相关背景。（2）在获取较少背景时可能会遗漏重要信息，在获取更多背景时可能会获取到无关信息。在本文中，我们提出了一个名为SAGE的RAG框架，克服了这些限制。首先，针对未考虑语义的分段问题，我们提出训练一个语义分割模型，该模型用于将语料库分割成语义完整的片段。其次，为确保仅检索最具相关性的片段而忽略无关片段，我们设计了一种分段选择算法，根据相关性评分下降速度动态选择片段，从而实现更有针对性的选择。第三，为了进一步确保检索片段的精准性，我们建议让LLMs评估检索到的片段是否过多或不足，并据此调整背景的量。实验结果显示，SAGE在问答质量方面平均优于基线模型61.25%。此外，通过避免检索噪声背景信息，SAGE降低了LLM推理中消耗的 token 成本，并在平均成本效率方面提高了49.41%。此外，我们的工作为提升RAG提供了有价值的见解。 

---
# Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens 

**Title (ZH)**: Spark-TTS：一种基于单流解耦语音 token 的高效 LLMA 文本转语音模型 

**Authors**: Xinsheng Wang, Mingqi Jiang, Ziyang Ma, Ziyu Zhang, Songxiang Liu, Linqin Li, Zheng Liang, Qixi Zheng, Rui Wang, Xiaoqin Feng, Weizhen Bian, Zhen Ye, Sitong Cheng, Ruibin Yuan, Zhixian Zhao, Xinfa Zhu, Jiahao Pan, Liumeng Xue, Pengcheng Zhu, Yunlin Chen, Zhifei Li, Xie Chen, Lei Xie, Yike Guo, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.01710)  

**Abstract**: Recent advancements in large language models (LLMs) have driven significant progress in zero-shot text-to-speech (TTS) synthesis. However, existing foundation models rely on multi-stage processing or complex architectures for predicting multiple codebooks, limiting efficiency and integration flexibility. To overcome these challenges, we introduce Spark-TTS, a novel system powered by BiCodec, a single-stream speech codec that decomposes speech into two complementary token types: low-bitrate semantic tokens for linguistic content and fixed-length global tokens for speaker attributes. This disentangled representation, combined with the Qwen2.5 LLM and a chain-of-thought (CoT) generation approach, enables both coarse-grained control (e.g., gender, speaking style) and fine-grained adjustments (e.g., precise pitch values, speaking rate). To facilitate research in controllable TTS, we introduce VoxBox, a meticulously curated 100,000-hour dataset with comprehensive attribute annotations. Extensive experiments demonstrate that Spark-TTS not only achieves state-of-the-art zero-shot voice cloning but also generates highly customizable voices that surpass the limitations of reference-based synthesis. Source code, pre-trained models, and audio samples are available at this https URL. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）在零-shot 文本转语音（TTS）合成中的最新进展已推动了显著的进步。然而，现有的基础模型依赖多阶段处理或复杂的架构来预测多个码本，限制了效率和集成灵活性。为克服这些挑战，我们介绍了由单一流程编码器（BiCodec）驱动的Spark-TTS，该编码器将语音分解为两种互补的标记类型：低比特率语义标记用于语言内容和固定长度的全局标记用于说话人属性。这种分离表示，结合Qwen2.5大语言模型和链式思考（CoT）生成方法，既支持粗粒度控制（如性别、发音风格），也支持细粒度调整（如精确的音调值、语速）。为促进可控TTS的研究，我们引入了VoxBox，这是一个精心挑选的100,000小时数据集，具有全面的属性注释。大量实验表明，Spark-TTS不仅实现了最先进的零-shot 语音克隆，而且还生成了高度可定制的声音，超越了参考基于合成的限制。更多代码、预训练模型和音频样本可在以下链接获得。 

---
# Evaluating LLMs' Assessment of Mixed-Context Hallucination Through the Lens of Summarization 

**Title (ZH)**: 通过摘要视角评价LLM对混合上下文幻觉的评估 

**Authors**: Siya Qi, Rui Cao, Yulan He, Zheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01670)  

**Abstract**: With the rapid development of large language models (LLMs), LLM-as-a-judge has emerged as a widely adopted approach for text quality evaluation, including hallucination evaluation. While previous studies have focused exclusively on single-context evaluation (e.g., discourse faithfulness or world factuality), real-world hallucinations typically involve mixed contexts, which remains inadequately evaluated. In this study, we use summarization as a representative task to comprehensively evaluate LLMs' capability in detecting mixed-context hallucinations, specifically distinguishing between factual and non-factual hallucinations. Through extensive experiments across direct generation and retrieval-based models of varying scales, our main observations are: (1) LLMs' intrinsic knowledge introduces inherent biases in hallucination evaluation; (2) These biases particularly impact the detection of factual hallucinations, yielding a significant performance bottleneck; (3) The fundamental challenge lies in effective knowledge utilization, balancing between LLMs' intrinsic knowledge and external context for accurate mixed-context hallucination evaluation. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的迅速发展，LLM-as-a-judge已广泛应用于文本质量评估，包括虚构性评估。尽管以往的研究主要集中于单一语境评估（如语篇忠实性或世界事实性），但现实生活中的虚构性通常涉及混合语境，而这些语境的评估仍显不足。本研究使用摘要作为代表性任务，全面评估LLMs在检测混合语境虚构性方面的能力，特别是区分事实性和非事实性虚构性。通过对不同规模的直接生成和检索模型进行广泛的实验，我们的主要观察结果是：（1）LLMs固有的知识在虚构性评估中引入了固有的偏见；（2）这些偏见特别影响了事实性虚构性的检测，导致了显著的性能瓶颈；（3）根本挑战在于有效利用知识，平衡LLMs固有的知识与外部语境，以实现准确的混合语境虚构性评估。 

---
# CoPL: Collaborative Preference Learning for Personalizing LLMs 

**Title (ZH)**: CoPL：协作偏好学习以个性化LLMs 

**Authors**: Youngbin Choi, Seunghyuk Cho, Minjong Lee, MoonJeong Park, Yesong Ko, Jungseul Ok, Dongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.01658)  

**Abstract**: Personalizing large language models (LLMs) is important for aligning outputs with diverse user preferences, yet existing methods struggle with flexibility and generalization. We propose CoPL (Collaborative Preference Learning), a graph-based collaborative filtering framework that models user-response relationships to enhance preference estimation, particularly in sparse annotation settings. By integrating a mixture of LoRA experts, CoPL efficiently fine-tunes LLMs while dynamically balancing shared and user-specific preferences. Additionally, an optimization-free adaptation strategy enables generalization to unseen users without fine-tuning. Experiments on UltraFeedback-P demonstrate that CoPL outperforms existing personalized reward models, effectively capturing both common and controversial preferences, making it a scalable solution for personalized LLM alignment. 

**Abstract (ZH)**: 基于图的合作偏好学习（CoPL）：一种用于个性化大型语言模型的框架 

---
# Distilled Prompt Learning for Incomplete Multimodal Survival Prediction 

**Title (ZH)**: Incomplete多模态生存预测的蒸馏提示学习 

**Authors**: Yingxue Xu, Fengtao Zhou, Chenyu Zhao, Yihui Wang, Can Yang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01653)  

**Abstract**: The integration of multimodal data including pathology images and gene profiles is widely applied in precise survival prediction. Despite recent advances in multimodal survival models, collecting complete modalities for multimodal fusion still poses a significant challenge, hindering their application in clinical settings. Current approaches tackling incomplete modalities often fall short, as they typically compensate for only a limited part of the knowledge of missing modalities. To address this issue, we propose a Distilled Prompt Learning framework (DisPro) to utilize the strong robustness of Large Language Models (LLMs) to missing modalities, which employs two-stage prompting for compensation of comprehensive information for missing modalities. In the first stage, Unimodal Prompting (UniPro) distills the knowledge distribution of each modality, preparing for supplementing modality-specific knowledge of the missing modality in the subsequent stage. In the second stage, Multimodal Prompting (MultiPro) leverages available modalities as prompts for LLMs to infer the missing modality, which provides modality-common information. Simultaneously, the unimodal knowledge acquired in the first stage is injected into multimodal inference to compensate for the modality-specific knowledge of the missing modality. Extensive experiments covering various missing scenarios demonstrated the superiority of the proposed method. The code is available at this https URL. 

**Abstract (ZH)**: 多模态数据（包括病理图像和基因特征）在精确生存预测中的集成应用广泛。尽管最近在多模态生存模型方面取得了进展，但收集完整模态以进行多模态融合仍然面临重大挑战，阻碍了其在临床环境中的应用。当前处理不完整模态的方法往往不尽如人意，因为它们通常只能填补缺失模态知识的一小部分。为了解决这个问题，我们提出了一种蒸馏提示学习框架（DisPro），利用大型语言模型（LLMs）对缺失模态的强鲁棒性，采用两阶段提示来为缺失模态提供全面信息的补偿。在第一阶段，单模态提示（UniPro）提取每个模态的知识分布，为后续阶段补充缺失模态的模态特定知识做准备。在第二阶段，多模态提示（MultiPro）利用可用模态作为提示引导LLMs推断缺失模态，提供模态共通信息，同时将第一阶段获得的单模态知识注入多模态推理，以填补缺失模态的模态特定知识。广泛的实验覆盖了各种缺失场景，证明了所提出方法的优越性。代码可在以下链接获取。 

---
# Machine Learners Should Acknowledge the Legal Implications of Large Language Models as Personal Data 

**Title (ZH)**: 机器学习者应承认大规模语言模型作为个人数据的法律影响 

**Authors**: Henrik Nolte, Michèle Finck, Kristof Meding  

**Link**: [PDF](https://arxiv.org/pdf/2503.01630)  

**Abstract**: Does GPT know you? The answer depends on your level of public recognition; however, if your information was available on a website, the answer is probably yes. All Large Language Models (LLMs) memorize training data to some extent. If an LLM training corpus includes personal data, it also memorizes personal data. Developing an LLM typically involves processing personal data, which falls directly within the scope of data protection laws. If a person is identified or identifiable, the implications are far-reaching: the AI system is subject to EU General Data Protection Regulation requirements even after the training phase is concluded. To back our arguments: (1.) We reiterate that LLMs output training data at inference time, be it verbatim or in generalized form. (2.) We show that some LLMs can thus be considered personal data on their own. This triggers a cascade of data protection implications such as data subject rights, including rights to access, rectification, or erasure. These rights extend to the information embedded with-in the AI model. (3.) This paper argues that machine learning researchers must acknowledge the legal implications of LLMs as personal data throughout the full ML development lifecycle, from data collection and curation to model provision on, e.g., GitHub or Hugging Face. (4.) We propose different ways for the ML research community to deal with these legal implications. Our paper serves as a starting point for improving the alignment between data protection law and the technical capabilities of LLMs. Our findings underscore the need for more interaction between the legal domain and the ML community. 

**Abstract (ZH)**: GPT了解你吗？这取决于你的公众知名度；然而，如果你的信息出现在网站上，答案可能是肯定的。所有的大型语言模型（LLMs）在一定程度上会记忆训练数据。如果LLM的训练语料库包含个人数据，它也会记忆个人数据。开发LLM通常涉及处理个人数据，这直接处于数据保护法律的管辖范围之内。如果个人可以被识别或可被识别，后果深远：即使在训练阶段结束后，AI系统也需要遵守欧盟通用数据保护条例的要求。为了支持我们的论点：（1）我们重申LLMs在推理时输出训练数据，无论是直接引用还是概括形式。 （2）我们证明一些LLMs可以被视为个人数据本身，这引发了数据保护的一系列影响，如数据主体权利，包括访问权、更正权或删除权，这些权利延伸到嵌入在AI模型中的信息。 （3）本文 argued 机器学习研究人员必须在整个机器学习开发生命周期中承认LLMs作为个人数据的法律含义，从数据收集和整理到模型提供，例如在GitHub或Hugging Face。 （4）我们提出了研究社区处理这些法律影响的不同方法。我们的论文为改善数据保护法与LLMs技术能力之间的契合度提供了起点。我们的研究结果突显了法律领域与机器学习社区之间加强互动的必要性。 

---
# Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering 

**Title (ZH)**: 超越提示：一种高效的开放域问答嵌入框架 

**Authors**: Zhanghao Hu, Hanqi Yan, Qingling Zhu, Zhenyi Shen, Yulan He, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2503.01606)  

**Abstract**: Large language models have recently pushed open domain question answering (ODQA) to new frontiers. However, prevailing retriever-reader pipelines often depend on multiple rounds of prompt level instructions, leading to high computational overhead, instability, and suboptimal retrieval coverage. In this paper, we propose EmbQA, an embedding-level framework that alleviates these shortcomings by enhancing both the retriever and the reader. Specifically, we refine query representations via lightweight linear layers under an unsupervised contrastive learning objective, thereby reordering retrieved passages to highlight those most likely to contain correct answers. Additionally, we introduce an exploratory embedding that broadens the model's latent semantic space to diversify candidate generation and employs an entropy-based selection mechanism to choose the most confident answer automatically. Extensive experiments across three open-source LLMs, three retrieval methods, and four ODQA benchmarks demonstrate that EmbQA substantially outperforms recent baselines in both accuracy and efficiency. 

**Abstract (ZH)**: 大型语言模型最近推动了开放域问答（ODQA）达到了新的前沿。然而，占主导地位的检索-阅读器管道经常依赖多轮提示级指令，导致高计算开销、不稳定性和检索覆盖率不足。本文提出EmbQA，这是一种嵌入级框架，通过增强检索器和阅读器来缓解这些问题。具体而言，我们通过轻量级线性层在无监督对比学习目标下精炼查询表示，从而重新排序检索段落以突出最有可能包含正确答案的部分。此外，我们引入了一种探索性嵌入，扩展了模型的潜在语义空间以多样化候选生成，并采用基于熵的选择机制以自动选择最自信的答案。在三个开源LLM、三种检索方法和四个ODQA基准上进行的广泛实验表明，EmbQA在准确性和效率上显著优于最近的基线。 

---
# EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection 

**Title (ZH)**: 精英KV：通过RoPE频率选择和联合低秩投影的可扩展键值缓存压缩 

**Authors**: Yuhao Zhou, Sirui Song, Boyang Liu, Zhiheng Xi, Senjie Jin, Xiaoran Fan, Zhihao Zhang, Wei Li, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01586)  

**Abstract**: Rotary Position Embedding (RoPE) enables each attention head to capture multi-frequency information along the sequence dimension and is widely applied in foundation models. However, the nonlinearity introduced by RoPE complicates optimization of the key state in the Key-Value (KV) cache for RoPE-based attention. Existing KV cache compression methods typically store key state before rotation and apply the transformation during decoding, introducing additional computational overhead. This paper introduces EliteKV, a flexible modification framework for RoPE-based models supporting variable KV cache compression ratios. EliteKV first identifies the intrinsic frequency preference of each head using RoPElite, selectively restoring linearity to certain dimensions of key within attention computation. Building on this, joint low-rank compression of key and value enables partial cache sharing. Experimental results show that with minimal uptraining on only $0.6\%$ of the original training data, RoPE-based models achieve a $75\%$ reduction in KV cache size while preserving performance within a negligible margin. Furthermore, EliteKV consistently performs well across models of different scales within the same family. 

**Abstract (ZH)**: Rotary Position Embedding (RoPE)的Rotaryelite enablers Each Attention Head to Capture Multi-Frequency Information along the Sequence Dimension and is Widely Applied in Foundation Models 

---
# Revisiting Large Language Model Pruning using Neuron Semantic Attribution 

**Title (ZH)**: revisit 大型语言模型剪枝利用神经语义归因 

**Authors**: Yizhuo Ding, Xinwei Sun, Yanwei Fu, Guosheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01542)  

**Abstract**: Model pruning technique is vital for accelerating large language models by reducing their size and computational requirements. However, the generalizability of existing pruning methods across diverse datasets and tasks remains unclear. Thus, we conduct extensive evaluations on 24 datasets and 4 tasks using popular pruning methods. Based on these evaluations, we find and then investigate that calibration set greatly affect the performance of pruning methods. In addition, we surprisingly find a significant performance drop of existing pruning methods in sentiment classification tasks. To understand the link between performance drop and pruned neurons, we propose Neuron Semantic Attribution, which learns to associate each neuron with specific semantics. This method first makes the unpruned neurons of LLMs explainable. 

**Abstract (ZH)**: 现有的剪枝方法在不同数据集和任务上的泛化能力仍不明确：一种针对情感分类任务的显著性能下降的解释——基于神经元语义归因的研究 

---
# Pragmatic Inference Chain (PIC) Improving LLMs' Reasoning of Authentic Implicit Toxic Language 

**Title (ZH)**: Pragmatic 推理链 (PIC) 改进大语言模型对真实隐含有毒语言的推理能力 

**Authors**: Xi Chen, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01539)  

**Abstract**: The rapid development of large language models (LLMs) gives rise to ethical concerns about their performance, while opening new avenues for developing toxic language detection techniques. However, LLMs' unethical output and their capability of detecting toxicity have primarily been tested on language data that do not demand complex meaning inference, such as the biased associations of 'he' with programmer and 'she' with household. Nowadays toxic language adopts a much more creative range of implicit forms, thanks to advanced censorship. In this study, we collect authentic toxic interactions that evade online censorship and that are verified by human annotators as inference intensive. To evaluate and improve LLMs' reasoning of the authentic implicit toxic language, we propose a new prompting method, Pragmatic Inference Chain (PIC), drawn on interdisciplinary findings from cognitive science and linguistics. The PIC prompting significantly improves the success rate of GPT-4o, Llama-3.1-70B-Instruct, and DeepSeek-v2.5 in identifying implicit toxic language, compared to both direct prompting and Chain-of-Thought. In addition, it also facilitates the models to produce more explicit and coherent reasoning processes, hence can potentially be generalized to other inference-intensive tasks, e.g., understanding humour and metaphors. 

**Abstract (ZH)**: 大型语言模型的迅速发展引发了对其性能的伦理关切，同时也为开发有毒语言检测技术开辟了新途径。然而，大型语言模型的不当输出及其检测毒性的能力主要是在不需要复杂意义推理的语言数据上进行测试的，例如“他”与程序员的偏见关联，“她”与家庭的关联。如今，由于高级审查技术的应用，有毒语言采取了更为创意的形式。在本研究中，我们收集了逃避在线审查的真实有毒互动，并由人工标注员验证这些互动需要复杂的推理。为了评估和提高大型语言模型对真实隐含有毒语言的理解能力，我们提出了一种新的提示方法——实用推理链（PIC），这一方法借鉴了认知科学和语言学的跨学科研究成果。与直接提示和思维链条相比，PIC提示显著提高了GPT-4o、Llama-3.1-70B-Instruct和DeepSeek-v2.5在识别隐含有毒语言方面的成功率。此外，它还使模型产生更加明确和连贯的推理过程，从而有可能应用于其他需要复杂推理的任务，例如理解幽默和隐喻。 

---
# Liger: Linearizing Large Language Models to Gated Recurrent Structures 

**Title (ZH)**: Liger: 将大型语言模型线性化为门控循环结构 

**Authors**: Disen Lan, Weigao Sun, Jiaxi Hu, Jusen Du, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01496)  

**Abstract**: Transformers with linear recurrent modeling offer linear-time training and constant-memory inference. Despite their demonstrated efficiency and performance, pretraining such non-standard architectures from scratch remains costly and risky. The linearization of large language models (LLMs) transforms pretrained standard models into linear recurrent structures, enabling more efficient deployment. However, current linearization methods typically introduce additional feature map modules that require extensive fine-tuning and overlook the gating mechanisms used in state-of-the-art linear recurrent models. To address these issues, this paper presents Liger, short for Linearizing LLMs to gated recurrent structures. Liger is a novel approach for converting pretrained LLMs into gated linear recurrent models without adding extra parameters. It repurposes the pretrained key matrix weights to construct diverse gating mechanisms, facilitating the formation of various gated recurrent structures while avoiding the need to train additional components from scratch. Using lightweight fine-tuning with Low-Rank Adaptation (LoRA), Liger restores the performance of the linearized gated recurrent models to match that of the original LLMs. Additionally, we introduce Liger Attention, an intra-layer hybrid attention mechanism, which significantly recovers 93\% of the Transformer-based LLM at 0.02\% pre-training tokens during the linearization process, achieving competitive results across multiple benchmarks, as validated on models ranging from 1B to 8B parameters. Code is available at this https URL. 

**Abstract (ZH)**: 将大型语言模型线性化为门控递归结构的Liger：无额外参数的线性递归模型高效部署 

---
# SePer: Measure Retrieval Utility Through The Lens Of Semantic Perplexity Reduction 

**Title (ZH)**: SePer: 通过语义困惑度降低的角度衡量检索效能 

**Authors**: Lu Dai, Yijie Xu, Jinhui Ye, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01478)  

**Abstract**: Large Language Models (LLMs) have demonstrated improved generation performance by incorporating externally retrieved knowledge, a process known as retrieval-augmented generation (RAG). Despite the potential of this approach, existing studies evaluate RAG effectiveness by 1) assessing retrieval and generation components jointly, which obscures retrieval's distinct contribution, or 2) examining retrievers using traditional metrics such as NDCG, which creates a gap in understanding retrieval's true utility in the overall generation process. To address the above limitations, in this work, we introduce an automatic evaluation method that measures retrieval quality through the lens of information gain within the RAG framework. Specifically, we propose Semantic Perplexity (SePer), a metric that captures the LLM's internal belief about the correctness of the retrieved information. We quantify the utility of retrieval by the extent to which it reduces semantic perplexity post-retrieval. Extensive experiments demonstrate that SePer not only aligns closely with human preferences but also offers a more precise and efficient evaluation of retrieval utility across diverse RAG scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过融合外部检索知识展示了增强的生成性能，这一过程称为检索增强生成（RAG）。尽管这种方法具有潜力，现有研究通过1）联合评估检索和生成组件，从而使检索的独特贡献模糊化，或2）使用传统指标如NDCG评估检索器，导致对检索在整个生成过程中的真正作用理解不足。为解决上述局限性，本文提出了一种自动评估方法，通过RAG框架内的信息增益视角测量检索质量。具体而言，我们提出了一种语义困惑度（SePer）度量，该度量捕捉了LLM对检索信息正确性的内部信念。我们通过检索后减少语义困惑度的程度来量化检索的实用性。广泛的实验表明，SePer不仅与人类偏好高度一致，而且能够在多种RAG场景中提供更精确和高效的检索实用性评估。 

---
# Rethinking Data: Towards Better Performing Domain-Specific Small Language Models 

**Title (ZH)**: 重新审视数据：朝着更好的领域专用小语言模型方向努力 

**Authors**: Boris Nazarov, Darya Frolova, Yackov Lubarsky, Alexei Gaissinski, Pavel Kisilev  

**Link**: [PDF](https://arxiv.org/pdf/2503.01464)  

**Abstract**: Fine-tuning of Large Language Models (LLMs) for downstream tasks, performed on domain-specific data has shown significant promise. However, commercial use of such LLMs is limited by the high computational cost required for their deployment at scale. On the other hand, small Language Models (LMs) are much more cost effective but have subpar performance in a similar setup. This paper presents our approach to finetuning a small LM, that reaches high accuracy in multiple choice question answering task. We achieve this by improving data quality at each stage of the LM training pipeline. In particular, we start with data structuring resulting in extraction of compact, semantically meaningful text chunks used by a retriever. This allows more efficient knowledge digestion by the LM. Further, we improve the retrieved context by training a lightweight Chunk Re-Ranker (CRR) that generates more accurate relative relevance chunk scores. Finally, we improve the model generalization ability by merging the models fine-tuned with different parameters on different data subsets. We present detailed procedure descriptions, and corresponding experimental findings that show the improvements of each one of the proposed techniques. 

**Abstract (ZH)**: 小型语言模型（LMs）在下游任务上的微调研究：通过提升数据质量实现高准确率的选择题回答任务 

---
# Leveraging LLMs for Mental Health: Detection and Recommendations from Social Discussions 

**Title (ZH)**: 利用大语言模型进行心理健康检测与建议：从社交讨论中获取 Insights 

**Authors**: Vaishali Aggarwal, Sachin Thukral, Krushil Patel, Arnab Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.01442)  

**Abstract**: Textual data from social platforms captures various aspects of mental health through discussions around and across issues, while users reach out for help and others sympathize and offer support. We propose a comprehensive framework that leverages Natural Language Processing (NLP) and Generative AI techniques to identify and assess mental health disorders, detect their severity, and create recommendations for behavior change and therapeutic interventions based on users' posts on Reddit.
To classify the disorders, we use rule-based labeling methods as well as advanced pre-trained NLP models to extract nuanced semantic features from the data. We fine-tune domain-adapted and generic pre-trained NLP models based on predictions from specialized Large Language Models (LLMs) to improve classification accuracy. Our hybrid approach combines the generalization capabilities of pre-trained models with the domain-specific insights captured by LLMs, providing an improved understanding of mental health discourse. Our findings highlight the strengths and limitations of each model, offering valuable insights into their practical applicability.
This research potentially facilitates early detection and personalized care to aid practitioners and aims to facilitate timely interventions and improve overall well-being, thereby contributing to the broader field of mental health surveillance and digital health analytics. 

**Abstract (ZH)**: 社交媒体平台上的文本数据通过围绕各种问题展开的讨论捕捉到心理健康的不同方面，用户寻求帮助，其他人则给予同情和支持。我们提出一种综合框架，利用自然语言处理（NLP）和生成型AI技术来识别和评估心理健康障碍，检测其严重程度，并基于Reddit上的用户帖子提出行为改变和治疗干预的建议。 

---
# Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding 

**Title (ZH)**: 采样效率的测试时扩展：早期解码中自估计的最佳采样数量 

**Authors**: Yiming Wang, Pei Zhang, Siyuan Huang, Baosong Yang, Zhuosheng Zhang, Fei Huang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01422)  

**Abstract**: Test-time scaling improves large language model performance by adding extra compute during decoding. Best-of-N (BoN) sampling serves as a common scaling technique, broadening the search space for finding better solutions from the model distribution. However, traditional BoN requires N full generations, leading to high GPU memory overhead and time latency. Moreover, some methods depend on reward models, adding computational cost and limiting domain generalization.
In this paper, we propose Self-Truncation Best-of-N (ST-BoN), a novel decoding method that avoids fully generating all samplings and eliminates the need for reward models. ST-BoN introduces early sampling consistency to estimate the most promising sample, truncating suboptimal ones to free memory and accelerate inference. This pushes the sampling-efficient test-time scaling. Compared to traditional BoN, ST-BoN can reduce dynamic GPU memory overhead by over 90% and time latency by 50%, while achieving comparable or even better performance across reasoning and open-ended domains. 

**Abstract (ZH)**: Test-time Scaling via Self-Truncation Best-of-N Improves Large Language Model Performance 

---
# Parameter-Efficient Fine-Tuning of Large Language Models via Deconvolution in Subspace 

**Title (ZH)**: 子空间中反卷积的参数高效微调大规模语言模型 

**Authors**: Jia-Chen Zhang, Yu-Jie Xiong, Chun-Ming Xia, Dong-Hai Zhu, Xi-He Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01419)  

**Abstract**: Large language model (LLM) is considered a milestone towards achieving Artificial General Intelligence (AGI). With its advanced emergent capabilities, it adapt to a wide range of specific applications. Fine-tuning LLMs for various downstream tasks has become a new paradigm. Low-Rank Adaptation (LoRA) is well-known for its parameter efficiency. It can reduce the number of parameters needed to fine-tune LLMs by several orders of magnitude. However, LoRA-based approaches encounter a significant limitation due to the bottleneck imposed by rank one decomposition. As the parameters count in LLMs increase, even rank one decomposition might surpass the number of parameters truly necessary for handling more downstream tasks. In this paper, we propose a new method for Parameter-Efficient Fine-Tuning (PEFT) via deconvolution in subspace, dubbed as DCFT. We innovatively use deconvolution to complete details and enhance knowledge in subspace incremental matrices, and dynamically control parameters by adjusting the kernel size, unconstrained by rank-one decomposition. Extensive experiments are conducted to validate the effectiveness of DCFT. Results show that compared to LoRA, DCFT achieve an 8$\times$ reduction in parameters, and still achieves highly impressive performance. Our code is available here: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLM）被认为是实现人工通用智能（AGI）的关键里程碑。通过其先进的涌现能力，它能够适应广泛的具体应用。针对各种下游任务对大型语言模型进行微调已成为一种新的范式。低秩适应（LoRA）因其实参效率而广为人知，它可以将需要微调的参数数量减少几个数量级。然而，基于LoRA的方法遇到了显著的限制，这是因为低秩分解所施加的瓶颈。随着大型语言模型参数数量的增加，即使低秩分解的数量也可能超过处理更多下游任务所需的真正必要参数数量。在本文中，我们提出了一种新的参数高效微调（PEFT）方法——称为时空反卷积参数化高效微调（DCFT）。我们创新性地利用反卷积在子空间增量矩阵中完成细节和增强知识，并通过调整核大小动态控制参数，不受低秩分解的限制。进行了大量的实验以验证DCFT的有效性。结果表明，与LoRA相比，DCFT在参数数量上实现了8倍的减少，同时仍实现了令人印象深刻的表现。我们的代码可以在以下链接获取：this https URL。 

---
# SwiLTra-Bench: The Swiss Legal Translation Benchmark 

**Title (ZH)**: SwiLTra-Bench: 瑞士法律翻译基准 

**Authors**: Joel Niklaus, Jakob Merane, Luka Nenadic, Sina Ahmadi, Yingqiang Gao, Cyrill A. H. Chevalley, Claude Humbel, Christophe Gösken, Lorenzo Tanzi, Thomas Lüthi, Stefan Palombo, Spencer Poff, Boling Yang, Nan Wu, Matthew Guillod, Robin Mamié, Daniel Brunner, Julio Pereyra, Niko Grupen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01372)  

**Abstract**: In Switzerland legal translation is uniquely important due to the country's four official languages and requirements for multilingual legal documentation. However, this process traditionally relies on professionals who must be both legal experts and skilled translators -- creating bottlenecks and impacting effective access to justice. To address this challenge, we introduce SwiLTra-Bench, a comprehensive multilingual benchmark of over 180K aligned Swiss legal translation pairs comprising laws, headnotes, and press releases across all Swiss languages along with English, designed to evaluate LLM-based translation systems. Our systematic evaluation reveals that frontier models achieve superior translation performance across all document types, while specialized translation systems excel specifically in laws but under-perform in headnotes. Through rigorous testing and human expert validation, we demonstrate that while fine-tuning open SLMs significantly improves their translation quality, they still lag behind the best zero-shot prompted frontier models such as Claude-3.5-Sonnet. Additionally, we present SwiLTra-Judge, a specialized LLM evaluation system that aligns best with human expert assessments. 

**Abstract (ZH)**: 在瑞士，由于国家有四种官方语言和多语言法律文件的要求，法律翻译具有独特的重要性。然而，这一过程传统上依赖于既是法律专家又是熟练翻译的专业人员——这造成了瓶颈并影响了有效获取司法服务。为应对这一挑战，我们引入了SwiLTra-Bench，这是一个包含超过180,000对对齐的瑞士法律翻译样本的综合多语言基准，涵盖了所有瑞士语言及其英语版本的法律条文、简短说明和新闻稿，旨在评估基于LLM的翻译系统。我们的系统性评估表明，前沿模型在所有文档类型上的翻译性能优于其他系统，而专门的翻译系统在法律条文中表现出色但在简短说明中则表现不佳。通过严格的测试和人类专家验证，我们发现，尽管微调开放的SLM显着提高了其翻译质量，但它们仍然落后于Claude-3.5-Sonnet等最佳零样本提示前沿模型。此外，我们还介绍了SwiLTra-Judge，这是一种与人类专家评估高度一致的专门LLM评估系统。 

---
# Same Question, Different Words: A Latent Adversarial Framework for Prompt Robustness 

**Title (ZH)**: 同一个问题，不同的表达：一种潜在对抗框架以提升提示 robustness 

**Authors**: Tingchen Fu, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2503.01345)  

**Abstract**: Insensitivity to semantically-preserving variations of prompts (paraphrases) is crucial for reliable behavior and real-world deployment of large language models. However, language models exhibit significant performance degradation when faced with semantically equivalent but differently phrased prompts, and existing solutions either depend on trial-and-error prompt engineering or require computationally expensive inference-time algorithms. In this study, built on the key insight that worst-case prompts exhibit a drift in embedding space, we present Latent Adversarial Paraphrasing (LAP), a dual-loop adversarial framework: the inner loop trains a learnable perturbation to serve as a "latent continuous paraphrase" while preserving semantics through Lagrangian regulation, and the outer loop optimizes the language model parameters on these perturbations. We conduct extensive experiments to demonstrate the effectiveness of LAP across multiple LLM architectures on the RobustAlpaca benchmark with a 0.5%-4% absolution improvement on worst-case win-rate compared with vanilla supervised fine-tuning. 

**Abstract (ZH)**: 对语义保持变化的提示（重述）的不敏感性是大型语言模型可靠行为和实际部署的关键。然而，当面对语义等价但表述不同的提示时，语言模型会表现出显著的性能下降，现有解决方案要么依赖于试错式的提示工程，要么需要计算成本高昂的推理时算法。在此研究中，基于最坏情况提示在嵌入空间中表现出漂移的关键洞察，我们提出了隐含对抗重述（LAP），一个双环对抗框架：内环训练可学习的扰动以充当“隐含连续重述”，并通过Lagrange调节保持语义，外环在这些扰动上优化语言模型参数。我们在RobustAlpaca基准上进行了广泛实验，证明了LAP在多个LLM架构上的有效性，与vanilla监督微调相比，在最坏情况胜率上实现了0.5%-4%的绝对改善。 

---
# Answer, Refuse, or Guess? Investigating Risk-Aware Decision Making in Language Models 

**Title (ZH)**: 回答、拒绝或猜测？探究语言模型中的风险意识决策-making 

**Authors**: Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.01332)  

**Abstract**: Knowing when to answer or refuse is crucial for safe and reliable decision-making language agents. Although prior work has introduced refusal strategies to boost LMs' reliability, how these models adapt their decisions to different risk levels remains underexplored. We formalize the task of risk-aware decision-making, expose critical weaknesses in existing LMs, and propose skill-decomposition solutions to mitigate them. Our findings show that even cutting-edge LMs--both regular and reasoning models--still require explicit prompt chaining to handle the task effectively, revealing the challenges that must be overcome to achieve truly autonomous decision-making agents. 

**Abstract (ZH)**: 适应不同风险级别的回答与拒绝决策对于安全可靠的语言代理至关重要：现有模型的风险感知决策任务分析与解决方案 

---
# Neural ODE Transformers: Analyzing Internal Dynamics and Adaptive Fine-tuning 

**Title (ZH)**: 神经ODE变换器：分析内部动力学和自适应微调 

**Authors**: Anh Tong, Thanh Nguyen-Tang, Dongeun Lee, Duc Nguyen, Toan Tran, David Hall, Cheongwoong Kang, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.01329)  

**Abstract**: Recent advancements in large language models (LLMs) based on transformer architectures have sparked significant interest in understanding their inner workings. In this paper, we introduce a novel approach to modeling transformer architectures using highly flexible non-autonomous neural ordinary differential equations (ODEs). Our proposed model parameterizes all weights of attention and feed-forward blocks through neural networks, expressing these weights as functions of a continuous layer index. Through spectral analysis of the model's dynamics, we uncover an increase in eigenvalue magnitude that challenges the weight-sharing assumption prevalent in existing theoretical studies. We also leverage the Lyapunov exponent to examine token-level sensitivity, enhancing model interpretability. Our neural ODE transformer demonstrates performance comparable to or better than vanilla transformers across various configurations and datasets, while offering flexible fine-tuning capabilities that can adapt to different architectural constraints. 

**Abstract (ZH)**: 基于变换器架构的大语言模型 Recent 进展：使用高度灵活的非自治神经常微分方程建模 

---
# Scaling Law Phenomena Across Regression Paradigms: Multiple and Kernel Approaches 

**Title (ZH)**: 不同回归范式下标度定律现象：多元和核方法研究 

**Authors**: Yifang Chen, Xuyang Guo, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.01314)  

**Abstract**: Recently, Large Language Models (LLMs) have achieved remarkable success. A key factor behind this success is the scaling law observed by OpenAI. Specifically, for models with Transformer architecture, the test loss exhibits a power-law relationship with model size, dataset size, and the amount of computation used in training, demonstrating trends that span more than seven orders of magnitude. This scaling law challenges traditional machine learning wisdom, notably the Oscar Scissors principle, which suggests that an overparametrized algorithm will overfit the training datasets, resulting in poor test performance. Recent research has also identified the scaling law in simpler machine learning contexts, such as linear regression. However, fully explaining the scaling law in large practical models remains an elusive goal. In this work, we advance our understanding by demonstrating that the scaling law phenomenon extends to multiple regression and kernel regression settings, which are significantly more expressive and powerful than linear methods. Our analysis provides deeper insights into the scaling law, potentially enhancing our understanding of LLMs. 

**Abstract (ZH)**: Recent大规模语言模型（LLMs）取得了显著成功。OpenAI观察到的规模律是其成功的关键因素之一。特别是对于具有Transformer架构的模型，测试损失与模型大小、数据集大小以及训练中使用的计算量之间呈现出幂律关系，显示了跨越七个数量级的趋势。这一规模律挑战了传统的机器学习智慧，特别是Oscar Scissors原则，该原则认为过参数化的算法会在训练数据集上过拟合，导致较差的测试性能。最近的研究还在更简单的机器学习上下文中（如线性回归）发现了这一规模律。然而，完全解释大型实际模型中的规模律仍然是一个难以捉摸的目标。在本文中，我们通过证明这一现象扩展到了多个回归和核回归设置中，加深了我们对规模律的理解，这些设置比线性方法更为表达性和强大，我们的分析可能有助于增强我们对LLMs的理解。 

---
# ReaderLM-v2: Small Language Model for HTML to Markdown and JSON 

**Title (ZH)**: ReaderLM-v2: 小型语言模型用于HTML到Markdown和JSON的转换 

**Authors**: Feng Wang, Zesheng Shi, Bo Wang, Nan Wang, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01151)  

**Abstract**: We present ReaderLM-v2, a compact 1.5 billion parameter language model designed for efficient web content extraction. Our model processes documents up to 512K tokens, transforming messy HTML into clean Markdown or JSON formats with high accuracy -- making it an ideal tool for grounding large language models. The model's effectiveness results from two key innovations: (1) a three-stage data synthesis pipeline that generates high quality, diverse training data by iteratively drafting, refining, and critiquing web content extraction; and (2) a unified training framework combining continuous pre-training with multi-objective optimization. Intensive evaluation demonstrates that ReaderLM-v2 outperforms GPT-4o-2024-08-06 and other larger models by 15-20\% on carefully curated benchmarks, particularly excelling at documents exceeding 100K tokens, while maintaining significantly lower computational requirements. 

**Abstract (ZH)**: ReaderLM-v2: 一种高效的1.5亿参数语言模型，用于web内容提取 

---
# How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach 

**Title (ZH)**: LLMs压缩自身推理过程的能力：一种基于tokens复杂性的方法 

**Authors**: Ayeong Lee, Ethan Che, Tianyi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01141)  

**Abstract**: Chain-of-thought prompting has emerged as a powerful technique for enabling large language models (LLMs) to solve complex reasoning tasks. However, these reasoning chains can be verbose, raising concerns about efficiency. In response, recent works have sought to decrease response lengths through simple prompting strategies (e.g. 'be concise'). In this work, we conduct the first systematic study of the relationship between reasoning length and model performance across a diverse range of compression instructions (e.g. 'use 10 words or less' or 'remove all punctuation'). In doing so, we discover a universal tradeoff between reasoning length and accuracy that persists across even very distinct reasoning chains. We demonstrate that this tradeoff emerges from a sharp threshold behavior at the question level: each task has an intrinsic 'token complexity' - a minimal number of tokens required for successful problem-solving. We show how token complexity enables us to compute information-theoretic limits on the accuracy-compression tradeoff, and find that prompt-based compression strategies operate far from these theoretical limits. This suggests there may be significant room for improvement and our framework provides a benchmark to help researchers evaluate progress in reasoning efficiency. Our work also highlights the importance of adaptive compression -- giving shorter responses for easier questions -- and we show that token complexity is a useful tool for measuring this capability. 

**Abstract (ZH)**: 关于推理长度与模型性能之间关系的系统研究：从压缩指令探索推理效率与准确性的权衡 

---
# Beyond QA Pairs: Assessing Parameter-Efficient Fine-Tuning for Fact Embedding in LLMs 

**Title (ZH)**: 超越问答对：评估参数高效微调在LLMs中事实嵌入的效果 

**Authors**: Shivam Ratnakar, Abhiroop Talasila, Raghav Chamadiya, Nikhil Agarwal, Vinayak K Doifode  

**Link**: [PDF](https://arxiv.org/pdf/2503.01131)  

**Abstract**: This paper presents an extensive examination of Parameter-Efficient Fine-Tuning (PEFT) for embedding domain specific facts into Large Language Models (LLMs), focusing on improving the fine-tuning process by categorizing question-answer (QA) pairs into Factual and Conceptual classes using a BERT-based classifier. Two distinct Llama-2 models are fine-tuned based on these classifications and evaluated using larger models like GPT-3.5 Turbo and Gemini. Our results indicate that models trained on conceptual datasets outperform those trained on factual datasets. Additionally, we compare the efficiency of two synthetic fine-tuning dataset generation techniques, D-RAG and D-Naive, with D-Naive demonstrating superior performance. Although PEFT has shown effectiveness, our research indicates that it may not be the most optimal method for embedding facts into LLMs. However, it has demonstrated exceptional performance in instruction-based tasks. Our findings are reinforced by a 1000-sample dataset in the data center domain, where the fine-tuned Llama-2 7B model significantly outperforms the baseline model in generating product recommendations. Our study highlights the importance of QA pair categorization and synthetic dataset generation techniques in enhancing the performance of LLMs in specific domains. 

**Abstract (ZH)**: 本研究对Parameter-Efficient Fine-Tuning (PEFT) 用于将领域特定事实嵌入大型语言模型 (LLMs) 进行了全面探讨，重点关注通过使用基于BERT的分类器将问题-答案 (QA) 对分类为事实性和概念性类别的方法来改进微调过程。基于这些分类微调了两个不同的Llama-2模型，并使用GPT-3.5 Turbo和Gemini等较大模型进行评估。结果显示，使用概念性数据集训练的模型优于使用事实性数据集训练的模型。此外，我们比较了两种合成微调数据集生成技术D-RAG和D-Naive的效率，发现D-Naive表现出更好的性能。尽管PEFT显示出有效性，但研究显示它可能不是将事实嵌入LLMs的最佳方法。然而，它在基于指令的任务上表现出色。我们的发现得到了数据中心领域的1000样本数据集的支持，其中微调的Llama-2 7B模型在生成产品推荐方面显著优于基线模型。我们的研究强调了在特定领域增强LLMs性能时QA对分类和合成数据集生成技术的重要性。 

---
# Scientific Reasoning: Assessment of Multimodal Generative LLMs 

**Title (ZH)**: 多模态生成性大语言模型的科学推理评估 

**Authors**: Florian Dreyer, Ekaterina Kolos, Daria Matiash  

**Link**: [PDF](https://arxiv.org/pdf/2503.01064)  

**Abstract**: Large language models (LLMs) can answer questions and reason about complex tasks, also from the scientific domain. We assess several multimodal LLMs (MLLMs) on ScienceQA and find that Gemini models show the highest accuracy with little context, and the highest textual similarity to human explanations with richer context. Adapter-tuning of smaller MLLMs did not lead to any reliable performance. Training from Gemini outputs consistently underperformed training from the original data. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以回答问题并处理复杂的任务，包括科学领域的问题。我们评估了几种多模态LLMs（MLLMs）在ScienceQA上的表现，发现Gemini模型在少量上下文情况下准确性最高，并且在丰富上下文中与人类解释的 textual 相似度最高。对较小的MLLMs进行适应性微调并没有提高可靠性能。从Gemini输出进行训练的表现始终不如使用原始数据进行训练。 

---
# SFO: Piloting VLM Feedback for Offline RL 

**Title (ZH)**: SFO: 指导大规模语言模型反馈的离线 reinforcement learning 

**Authors**: Jacob Beck  

**Link**: [PDF](https://arxiv.org/pdf/2503.01062)  

**Abstract**: While internet-scale image and textual data have enabled strong generalization in Vision-Language Models (VLMs), the absence of internet-scale control data has impeded the development of similar generalization in standard reinforcement learning (RL) agents. Although VLMs are fundamentally limited in their ability to solve control tasks due to their lack of action-conditioned training data, their capacity for image understanding allows them to provide valuable feedback in RL tasks by recognizing successful outcomes. A key challenge in Reinforcement Learning from AI Feedback (RLAIF) is determining how best to integrate VLM-derived signals into the learning process. We explore this question in the context of offline RL and introduce a class of methods called sub-trajectory filtered optimization. We identify three key insights. First, trajectory length plays a crucial role in offline RL, as full-trajectory preference learning exacerbates the stitching problem, necessitating the use of sub-trajectories. Second, even in Markovian environments, a non-Markovian reward signal from a sequence of images is required to assess trajectory improvement, as VLMs do not interpret control actions and must rely on visual cues over time. Third, a simple yet effective approach--filtered and weighted behavior cloning--consistently outperforms more complex reinforcement learning from human feedback-based methods. We propose sub-trajectory filtered behavior cloning, a method that leverages VLM feedback on sub-trajectories while incorporating a retrospective filtering mechanism that removes sub-trajectories preceding failures to improve robustness and prevent turbulence. This study is preliminary; we provide initial evidence through evaluations on a toy control domain. Please enjoy our airport puns. 

**Abstract (ZH)**: 虽然互联网规模的图像和文本数据使得视觉语言模型（VLMs）具备较强的泛化能力，但由于缺乏互联网规模的控制数据，标准强化学习（RL）代理的发展受到了限制。尽管VLMs在解决控制任务方面由于缺少基于动作的训练数据而受到根本性的限制，但它们在图像理解方面的能力可以通过识别成功的结果来为RL任务提供宝贵的反馈。从AI反馈进行强化学习（RLAIF）的一个关键挑战是如何最佳地将VLM衍生的信号整合到学习过程中。我们从离线RL的角度探讨了这一问题，并引入了一类称为子轨迹筛选优化的方法。我们识别出三个关键见解。首先，轨迹长度在离线RL中扮演着至关重要的角色，因为全长轨迹偏好学习加剧了缝合问题，迫使我们使用子轨迹。其次，即使在马尔可夫环境里，对于评估轨迹改进仍需要来自图像序列的非马尔可夫奖励信号，因为VLMs无法解释控制动作，而是依赖于时间上的视觉线索。第三，一种简单而有效的方法——筛选和加权行为克隆——在复杂的人类反馈强化学习方法中表现得更为出色。我们提出了一种子轨迹筛选行为克隆方法，该方法利用VLM在子轨迹上的反馈，并结合一种回顾性的筛选机制，该机制会移除失败前的子轨迹，以提高鲁棒性并防止振动。这项研究是初步的；我们通过在玩具控制域上的评估提供了初步证据。请欣赏我们的机场双关语。 

---
# Language Models Predict Empathy Gaps Between Social In-groups and Out-groups 

**Title (ZH)**: 语言模型预测社交内群体与外群体之间的共情差距 

**Authors**: Yu Hou, Hal Daumé III, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.01030)  

**Abstract**: Studies of human psychology have demonstrated that people are more motivated to extend empathy to in-group members than out-group members (Cikara et al., 2011). In this study, we investigate how this aspect of intergroup relations in humans is replicated by LLMs in an emotion intensity prediction task. In this task, the LLM is given a short description of an experience a person had that caused them to feel a particular emotion; the LLM is then prompted to predict the intensity of the emotion the person experienced on a numerical scale. By manipulating the group identities assigned to the LLM's persona (the "perceiver") and the person in the narrative (the "experiencer"), we measure how predicted emotion intensities differ between in-group and out-group settings. We observe that LLMs assign higher emotion intensity scores to in-group members than out-group members. This pattern holds across all three types of social groupings we tested: race/ethnicity, nationality, and religion. We perform an in-depth analysis on Llama-3.1-8B, the model which exhibited strongest intergroup bias among those tested. 

**Abstract (ZH)**: 人类心理学的研究表明，人们在向群体内成员而非群体外成员施予同情方面更有动机（Cikara等，2011）。在本研究中，我们探讨这种人类群体关系方面的特征在情感强度预测任务中如何被LLM复制。在这个任务中，LLM被给出一个描述某人经历而导致其产生特定情绪的简短描述；然后被提示预测该人经历的情绪强度在数值比例尺上的数值。通过操控分配给LLM的人格（“观察者”）和叙述中的人物（“体验者”）的社会群体身份，我们测量了在群体内和群体外情景中预测的情绪强度差异。我们观察到，LLM对群体内成员的情感强度评分高于群体外成员。这一模式在我们测试的所有三种社会群体类型中都成立：种族/ ethnicity、国籍和宗教。我们对表现最明显的群体间偏见的Llama-3.1-8B模型进行了深入分析。 

---
# LLM-Fusion: A Novel Multimodal Fusion Model for Accelerated Material Discovery 

**Title (ZH)**: LLM-融合：一种加速材料发现的新型多模态融合模型 

**Authors**: Onur Boyar, Indra Priyadarsini, Seiji Takeda, Lisa Hamada  

**Link**: [PDF](https://arxiv.org/pdf/2503.01022)  

**Abstract**: Discovering materials with desirable properties in an efficient way remains a significant problem in materials science. Many studies have tackled this problem by using different sets of information available about the materials. Among them, multimodal approaches have been found to be promising because of their ability to combine different sources of information. However, fusion algorithms to date remain simple, lacking a mechanism to provide a rich representation of multiple modalities. This paper presents LLM-Fusion, a novel multimodal fusion model that leverages large language models (LLMs) to integrate diverse representations, such as SMILES, SELFIES, text descriptions, and molecular fingerprints, for accurate property prediction. Our approach introduces a flexible LLM-based architecture that supports multimodal input processing and enables material property prediction with higher accuracy than traditional methods. We validate our model on two datasets across five prediction tasks and demonstrate its effectiveness compared to unimodal and naive concatenation baselines. 

**Abstract (ZH)**: 高效发现具有 desirable 性质的材料仍然是材料科学中的一个重大问题。许多研究通过利用关于材料的不同信息集来解决这一问题。其中，多模态方法因其结合不同信息源的能力而显示出前景。然而，到目前为止的融合算法仍较为简单，缺乏为多种模态提供丰富表示的机制。本文提出了一个名为 LLM-Fusion 的新颖多模态融合模型，该模型利用大型语言模型（LLMs）整合诸如 SMILES、SELFIES、文本描述和分子指纹等多样化的表示，以实现准确的性质预测。我们的方法引入了一个灵活的基于 LLM 的架构，支持多种模态的输入处理，并能够通过更高精度实现材料性质预测。我们在两个数据集中针对五个预测任务验证了该模型，并证明了其有效性，优于单一模态和简单的串联基线方法。 

---
# Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs 

**Title (ZH)**: 无界对话：LLMs中扩展响应的恒定大小KV缓存 

**Authors**: Ravi Ghadia, Avinash Kumar, Gaurav Jain, Prashant Nair, Poulami Das  

**Link**: [PDF](https://arxiv.org/pdf/2503.00979)  

**Abstract**: Autoregressive Transformers rely on Key-Value (KV) caching to accelerate inference. However, the linear growth of the KV cache with context length leads to excessive memory consumption and bandwidth constraints. This bottleneck is particularly problematic in real-time applications -- such as chatbots and interactive assistants -- where low latency and high memory efficiency are critical. Existing methods drop distant tokens or compress states in a lossy manner, sacrificing accuracy by discarding vital context or introducing bias.
We propose MorphKV, an inference-time technique that maintains a constant-sized KV cache while preserving accuracy. MorphKV balances long-range dependencies and local coherence during text generation. It eliminates early-token bias while retaining high-fidelity context by adaptively ranking tokens through correlation-aware selection. Unlike heuristic retention or lossy compression, MorphKV iteratively refines the KV cache via lightweight updates guided by attention patterns of recent tokens. This approach captures inter-token correlation with greater accuracy, crucial for tasks like content creation and code generation. Our studies on long-response tasks show 52.9$\%$ memory savings and 18.2$\%$ higher accuracy on average compared to state-of-the-art prior works, enabling efficient real-world deployment. 

**Abstract (ZH)**: MorphKV：一种保持恒定大小的键值缓存并保持准确性的推理时技术 

---
# SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking 

**Title (ZH)**: SemViQA: 一种用于越南语信息事实核查的语义问答系统 

**Authors**: Nam V. Nguyen, Dien X. Tran, Thanh T. Tran, Anh T. Hoang, Tai V. Duong, Di T. Le, Phuc-Lu Le  

**Link**: [PDF](https://arxiv.org/pdf/2503.00955)  

**Abstract**: The rise of misinformation, exacerbated by Large Language Models (LLMs) like GPT and Gemini, demands robust fact-checking solutions, especially for low-resource languages like Vietnamese. Existing methods struggle with semantic ambiguity, homonyms, and complex linguistic structures, often trading accuracy for efficiency. We introduce SemViQA, a novel Vietnamese fact-checking framework integrating Semantic-based Evidence Retrieval (SER) and Two-step Verdict Classification (TVC). Our approach balances precision and speed, achieving state-of-the-art results with 78.97\% strict accuracy on ISE-DSC01 and 80.82\% on ViWikiFC, securing 1st place in the UIT Data Science Challenge. Additionally, SemViQA Faster improves inference speed 7x while maintaining competitive accuracy. SemViQA sets a new benchmark for Vietnamese fact verification, advancing the fight against misinformation. The source code is available at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT和Gemini加剧了虚假信息的传播，特别是对于越南语等低资源语言而言，急需 robust 的事实核查解决方案。现有的方法在语义模糊性、同音词以及复杂语言结构方面存在困难，常常在准确性和效率之间做出权衡。我们提出了SemViQA，这是一种新颖的越南语事实核查框架，结合了基于语义的证据检索（SER）和两步验证分类（TVC）。我们的方法在精确度和速度之间取得了平衡，并在ISE-DSC01上取得了78.97%的严格准确率，在ViWikiFC上取得了80.82%的准确率，获得UIT数据科学挑战赛冠军。此外，SemViQA Faster 将推理速度提高了7倍，同时保持了竞争力的准确率。SemViQA 为越南语事实核查设立了新的基准，并推动了对抗虚假信息的斗争。源代码可在以下链接获取：this https URL。 

---
# HiBench: Benchmarking LLMs Capability on Hierarchical Structure Reasoning 

**Title (ZH)**: HiBench: LLMs在层次结构推理能力评估 

**Authors**: Zhuohang Jiang, Pangjing Wu, Ziran Liang, Peter Q. Chen, Xu Yuan, Ye Jia, Jiancheng Tu, Chen Li, Peter H.F. Ng, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00912)  

**Abstract**: Structure reasoning is a fundamental capability of large language models (LLMs), enabling them to reason about structured commonsense and answer multi-hop questions. However, existing benchmarks for structure reasoning mainly focus on horizontal and coordinate structures (\emph{e.g.} graphs), overlooking the hierarchical relationships within them. Hierarchical structure reasoning is crucial for human cognition, particularly in memory organization and problem-solving. It also plays a key role in various real-world tasks, such as information extraction and decision-making. To address this gap, we propose HiBench, the first framework spanning from initial structure generation to final proficiency assessment, designed to benchmark the hierarchical reasoning capabilities of LLMs systematically. HiBench encompasses six representative scenarios, covering both fundamental and practical aspects, and consists of 30 tasks with varying hierarchical complexity, totaling 39,519 queries. To evaluate LLMs comprehensively, we develop five capability dimensions that depict different facets of hierarchical structure understanding. Through extensive evaluation of 20 LLMs from 10 model families, we reveal key insights into their capabilities and limitations: 1) existing LLMs show proficiency in basic hierarchical reasoning tasks; 2) they still struggle with more complex structures and implicit hierarchical representations, especially in structural modification and textual reasoning. Based on these findings, we create a small yet well-designed instruction dataset, which enhances LLMs' performance on HiBench by an average of 88.84\% (Llama-3.1-8B) and 31.38\% (Qwen2.5-7B) across all tasks. The HiBench dataset and toolkit are available here, this https URL, to encourage evaluation. 

**Abstract (ZH)**: 大型语言模型的结构推理能力是其基本能力之一，使得它们能够对结构化常识进行推理并回答多跳问题。然而，现有的结构推理基准主要关注水平和坐标结构（例如，图形），忽略了它们内部的层级关系。层级结构推理对于人类认知至关重要，特别是在记忆组织和问题解决中。它在信息提取和决策制定等多种实际任务中也起着关键作用。为解决这一问题，我们提出了HiBench，这是第一个从初始结构生成到最终能力评估的框架，旨在系统地基准大型语言模型的层级推理能力。HiBench涵盖了六个代表性场景，既包括基础层面也包括实际应用层面，并包含30个具有不同层级复杂性的任务，共计39,519个查询。为了全面评估大型语言模型，我们开发了五个能力维度，描绘了层级结构理解的不同方面。通过对10个模型家族中的20个大型语言模型进行广泛评估，我们揭示了关于其能力和局限性的关键见解：1）现有大型语言模型在基本层级推理任务上表现出色；2）它们在更复杂的结构和隐含的层级表示方面仍然面临挑战，尤其是在结构修改和文本推理方面。基于这些发现，我们创建了一个小型但设计精良的指令数据集，通过提升Llama-3.1-8B和Qwen2.5-7B在所有任务上的平均表现，分别为88.84%和31.38%。HiBench数据集和工具包在此处提供，以鼓励评估：this https URL。 

---
# A Simple and Effective Reinforcement Learning Method for Text-to-Image Diffusion Fine-tuning 

**Title (ZH)**: 一种简单而有效的强化学习方法用于文本到图像扩散模型微调 

**Authors**: Shashank Gupta, Chaitanya Ahuja, Tsung-Yu Lin, Sreya Dutta Roy, Harrie Oosterhuis, Maarten de Rijke, Satya Narayan Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2503.00897)  

**Abstract**: Reinforcement learning ( RL)-based fine-tuning has emerged as a powerful approach for aligning diffusion models with black-box objectives. Proximal policy optimization (PPO) is the most popular choice of method for policy optimization. While effective in terms of performance, PPO is highly sensitive to hyper-parameters and involves substantial computational overhead. REINFORCE, on the other hand, mitigates some computational complexities such as high memory overhead and sensitive hyper-parameter tuning, but has suboptimal performance due to high-variance and sample inefficiency. While the variance of the REINFORCE can be reduced by sampling multiple actions per input prompt and using a baseline correction term, it still suffers from sample inefficiency. To address these challenges, we systematically analyze the efficiency-effectiveness trade-off between REINFORCE and PPO, and propose leave-one-out PPO ( LOOP), a novel RL for diffusion fine-tuning method. LOOP combines variance reduction techniques from REINFORCE, such as sampling multiple actions per input prompt and a baseline correction term, with the robustness and sample efficiency of PPO via clipping and importance sampling. Our results demonstrate that LOOP effectively improves diffusion models on various black-box objectives, and achieves a better balance between computational efficiency and performance. 

**Abstract (ZH)**: 基于强化学习的微调方法：proximal策略优化在与黑盒目标对齐扩散模型中的应用及其效率效果权衡分析——leave-one-out PPO（LOOP）方法在扩散模型微调中的应用 

---
# Babel: Open Multilingual Large Language Models Serving Over 90% of Global Speakers 

**Title (ZH)**: Babel: 开放多语言大型语言模型服务全球90%以上的Speech者 

**Authors**: Yiran Zhao, Chaoqun Liu, Yue Deng, Jiahao Ying, Mahani Aljunied, Zhaodonghui Li, Lidong Bing, Hou Pong Chan, Yu Rong, Deli Zhao, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00865)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing (NLP), yet open-source multilingual LLMs remain scarce, with existing models often limited in language coverage. Such models typically prioritize well-resourced languages, while widely spoken but under-resourced languages are often overlooked. To address this disparity, we introduce $\texttt{Babel}$, an open multilingual LLM that covers the top 25 languages by number of speakers, supports over 90% of the global population, and includes many languages neglected by other open multilingual LLMs. Unlike traditional continue pretraining approaches, Babel expands its parameter count through a layer extension technique that elevates Babel's performance ceiling. We introduce two variants: $\texttt{Babel-9B}$, designed for efficient inference and fine-tuning, and $\texttt{Babel-83B}$, which sets a new standard for open multilingual LLMs. Extensive evaluations on multilingual tasks demonstrate its superior performance compared to open LLMs of comparable size. In addition, using open-source supervised fine-tuning datasets, Babel achieves remarkable performance, with Babel-9B-Chat leading among 10B-sized LLMs and Babel-83B-Chat setting a new standard for multilingual tasks, reaching the same level of commercial models. 

**Abstract (ZH)**: 大型语言模型（LLMs）已revolutionized自然语言处理（NLP），但开源多语言LLMs依然稀缺，现有模型往往在语言覆盖上有限制。这类模型通常优先考虑资源丰富的语言，而广泛使用但资源不足的语言常常被忽视。为解决这一问题，我们介绍了$\texttt{Babel}$，一个开源多语言LLM，覆盖了说人口最多的前25种语言，支持全球超过90%的人口，并包含许多其他开源多语言LLM忽视的语言。与传统的连续预训练方法不同，Babel通过层扩展技术扩展了其参数数量，提升了Babel的性能上限。我们引入了两个变体：$\texttt{Babel-9B}$，用于高效推理和微调，以及$\texttt{Babel-83B}$，后者为开源多语言LLM设定新的标准。广泛的任务评估表明，其在性能上优于同类规模的开源LLM。此外，借助开源监督微调数据集，Babel取得了显著性能，其中$\texttt{Babel-9B-Chat}$在10B规模的LLM中名列前茅，而$\texttt{Babel-83B-Chat}$为多语言任务设定了新标准，达到商业模型的水平。 

---
# Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners 

**Title (ZH)**: 奖励图推理过程使大语言模型成为更通用的推理者 

**Authors**: Miao Peng, Nuo Chen, Zongrui Suo, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00845)  

**Abstract**: Despite significant advancements in Large Language Models (LLMs), developing advanced reasoning capabilities in LLMs remains a key challenge. Process Reward Models (PRMs) have demonstrated exceptional promise in enhancing reasoning by providing step-wise feedback, particularly in the context of mathematical reasoning. However, their application to broader reasoning domains remains understudied, largely due to the high costs associated with manually creating step-level supervision. In this work, we explore the potential of PRMs in graph reasoning problems - a domain that demands sophisticated multi-step reasoning and offers opportunities for automated step-level data generation using established graph algorithms. We introduce GraphSILO, the largest dataset for graph reasoning problems with fine-grained step-wise labels, built using automated Task-oriented Trajectories and Monte Carlo Tree Search (MCTS) to generate detailed reasoning steps with step-wise labels. Building upon this dataset, we train GraphPRM, the first PRM designed for graph reasoning problems, and evaluate its effectiveness in two key settings: inference-time scaling and reinforcement learning via Direct Preference Optimization (DPO). Experimental results show that GraphPRM significantly improves LLM performance across 13 graph reasoning tasks, delivering a 9% gain for Qwen2.5-7B and demonstrating transferability to new graph reasoning datasets and new reasoning domains like mathematical problem-solving. Notably, GraphPRM enhances LLM performance on GSM8K and Math500, underscoring the cross-domain applicability of graph-based reasoning rewards. Our findings highlight the potential of PRMs in advancing reasoning across diverse domains, paving the way for more versatile and effective LLMs. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）取得了显著进展，但在LLMs中开发高级推理能力仍然是一个关键挑战。过程奖励模型（PRMs）通过提供逐步反馈，在增强推理能力方面展现了出色潜力，特别是在数学推理方面。然而，由于手动创建逐步监督的成本高昂，PRMs在更广泛推理领域的应用仍然研究不足。在本工作中，我们探讨了PRMs在图推理问题中的潜力——一个需要复杂多步骤推理且可通过现有图算法自动生成逐步数据的领域。我们引入了GraphSILO，这是迄今为止最大的图推理问题数据集，附带精细的步骤级标签，该数据集基于自动化任务导向轨迹和蒙特卡洛树搜索（MCTS）构建，用于生成详细的推理步骤和逐步标签。基于此数据集，我们训练了GraphPRM，这是首个为图推理问题设计的PRM，并在其在推理时间扩展和直接偏好优化（DPO）强化学习两种关键场景中的有效性进行了评估。实验结果表明，GraphPRM在13个图推理任务中显著提高了LLM的性能，为Qwen2.5-7B带来了9%的提升，并且能够转移到新的图推理数据集和新的推理领域如数学问题求解中。值得注意的是，GraphPRM在GSM8K和Math500上的应用提升了LLM的性能，突显了基于图的推理奖励在跨领域中的适用性。我们的研究结果强调了PRMs在推动跨领域推理方面潜力，为进一步提升更具灵活性和有效性的大语言模型铺平了道路。 

---
# AI Agents for Ground-Based Gamma Astronomy 

**Title (ZH)**: 基于地面伽马射线天文观测的AI代理 

**Authors**: D. Kostunin, V. Sotnikov, S. Golovachev, A. Strube  

**Link**: [PDF](https://arxiv.org/pdf/2503.00821)  

**Abstract**: Next-generation instruments for ground-based gamma-ray astronomy are marked by a substantial increase in complexity, featuring dozens of telescopes. This leap in scale introduces significant challenges in managing system operations and offline data analysis. Methods, which depend on advanced personnel training and sophisticated software, become increasingly strained as system complexity grows, making it more challenging to effectively support users in such a multifaceted environment. To address these challenges, we propose the development of AI agents based on instruction-finetuned large language models (LLMs). These agents align with specific documentation and codebases, understand the environmental context, operate with external APIs, and communicate with humans in natural language. Leveraging the advanced capabilities of modern LLMs, which can process and retain vast amounts of information, these AI agents offer a transformative approach to system management and data analysis by automating complex tasks and providing intelligent assistance. We present two prototypes that integrate with the Cherenkov Telescope Array Observatory pipelines for operations and offline data analysis. The first prototype automates data model implementation and maintenance for the Configuration Database of the Array Control and Data Acquisition (ACADA). The second prototype is an open-access code generation application tailored for data analysis based on the Gammapy framework. 

**Abstract (ZH)**: 基于高级语言模型指令微调的AI代理在地基伽马射线天文学中的系统管理和数据分析中的应用 

---
# Towards Reliable LLM-Driven Fuzz Testing: Vision and Road Ahead 

**Title (ZH)**: 面向可靠的大规模语言模型驱动模糊测试：愿景与前行之路 

**Authors**: Yiran Cheng, Hong Jin Kang, Lwin Khin Shar, Chaopeng Dong, Zhiqiang Shi, Shichao Lv, Limin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00795)  

**Abstract**: Fuzz testing is a crucial component of software security assessment, yet its effectiveness heavily relies on valid fuzz drivers and diverse seed inputs. Recent advancements in Large Language Models (LLMs) offer transformative potential for automating fuzz testing (LLM4Fuzz), particularly in generating drivers and seeds. However, current LLM4Fuzz solutions face critical reliability challenges, including low driver validity rates and seed quality trade-offs, hindering their practical adoption.
This paper aims to examine the reliability bottlenecks of LLM-driven fuzzing and explores potential research directions to address these limitations. It begins with an overview of the current development of LLM4SE and emphasizes the necessity for developing reliable LLM4Fuzz solutions. Following this, the paper envisions a vision where reliable LLM4Fuzz transforms the landscape of software testing and security for industry, software development practitioners, and economic accessibility. It then outlines a road ahead for future research, identifying key challenges and offering specific suggestions for the researchers to consider. This work strives to spark innovation in the field, positioning reliable LLM4Fuzz as a fundamental component of modern software testing. 

**Abstract (ZH)**: LLM驱动的模糊测试可靠性瓶颈及其研究方向 

---
# Towards Efficient Educational Chatbots: Benchmarking RAG Frameworks 

**Title (ZH)**: 面向高效的教育聊天机器人：RAG框架的基准测试 

**Authors**: Umar Ali Khan, Ekram Khan, Fiza Khan, Athar Ali Moinuddin  

**Link**: [PDF](https://arxiv.org/pdf/2503.00781)  

**Abstract**: Large Language Models (LLMs) have proven immensely beneficial in education by capturing vast amounts of literature-based information, allowing them to generate context without relying on external sources. In this paper, we propose a generative AI-powered GATE question-answering framework (GATE stands for Graduate Aptitude Test in Engineering) that leverages LLMs to explain GATE solutions and support students in their exam preparation. We conducted extensive benchmarking to select the optimal embedding model and LLM, evaluating our framework based on criteria such as latency, faithfulness, and relevance, with additional validation through human evaluation. Our chatbot integrates state-of-the-art embedding models and LLMs to deliver accurate, context-aware responses. Through rigorous experimentation, we identified configurations that balance performance and computational efficiency, ensuring a reliable chatbot to serve students' needs. Additionally, we discuss the challenges faced in data processing and modeling and implemented solutions. Our work explores the application of Retrieval-Augmented Generation (RAG) for GATE Q/A explanation tasks, and our findings demonstrate significant improvements in retrieval accuracy and response quality. This research offers practical insights for developing effective AI-driven educational tools while highlighting areas for future enhancement in usability and scalability. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在教育中通过捕获大量的文献信息，证明了其极大的益处，使其能够在不依赖外部来源的情况下生成上下文。本文提出了一种基于生成式AI的GATE问答框架（GATE代表工程研究生入学考试），该框架利用LLMs解释GATE解决方案并支持学生备考。我们进行了广泛的基准测试以选择最优的嵌入模型和LLM，并基于延迟、忠实度和相关性等标准评估框架，通过人工评估进行了额外验证。我们的聊天机器人整合了最新的嵌入模型和LLM，以提供准确且上下文相关的响应。通过严格的实验，我们确定了平衡性能和计算效率的配置，确保了可靠的服务，以满足学生的需求。此外，我们讨论了数据处理和建模面临的挑战并实现了相应的解决方案。本文探讨了在GATE问答解释任务中应用检索增强生成（RAG）的应用，并发现检索准确性和响应质量都得到了显著提升。这项研究为开发有效的AI驱动教育工具提供了实用见解，并指出了未来在易用性和可扩展性方面需要改进的领域。 

---
# LLMs are everywhere: Ubiquitous Utilization of AI Models through Air Computing 

**Title (ZH)**: LLMs无处不在：通过空气计算广泛应用AI模型 

**Authors**: Baris Yamansavascilar, Atay Ozgovde, Cem Ersoy  

**Link**: [PDF](https://arxiv.org/pdf/2503.00767)  

**Abstract**: We are witnessing a new era where problem-solving and cognitive tasks are being increasingly delegated to Large Language Models (LLMs) across diverse domains, ranging from code generation to holiday planning. This trend also creates a demand for the ubiquitous execution of LLM-powered applications in a wide variety of environments in which traditional terrestrial 2D networking infrastructures may prove insufficient. A promising solution in this context is to extend edge computing into a 3D setting to include aerial platforms organized in multiple layers, a paradigm we refer to as air computing, to augment local devices for running LLM and Generative AI (GenAI) applications. This approach alleviates the strain on existing infrastructure while enhancing service efficiency by offloading computational tasks to the corresponding air units such as UAVs. Furthermore, the coordinated deployment of various air units can significantly improve the Quality of Experience (QoE) by ensuring seamless, adaptive, and resilient task execution. In this study, we investigate the synergy between LLM-based applications and air computing, exploring their potential across various use cases. Additionally, we present a disaster response case study demonstrating how the collaborative utilization of LLMs and air computing can significantly improve outcomes in critical situations. 

**Abstract (ZH)**: 我们正处在一个新的时代，其中问题解决和认知任务越来越多地被大型语言模型（LLMs）分配到各个领域，从代码生成到假日规划。这一趋势也引发了对基于LLM的应用程序在各种环境中的广泛执行的需求，在这些环境中，传统的地面2D网络基础设施可能不够用。在这个背景下，一种有前景的解决方案是将边缘计算扩展到三维设置，包括多层组织的空中平台，我们称之为空计算（air computing），以增强本地设备运行LLM和生成式AI（GenAI）应用的能力。这种方法可以缓解现有基础设施的压力，通过将计算任务卸载到相应的空中单元（如无人机）来提升服务质量。此外，各种空中单元的协调部署可以显著提高用户体验质量（QoE），通过确保无缝、适应性和弹性的任务执行。在本研究中，我们探讨了基于LLM的应用程序与空计算之间的协同作用，并探索它们在各种用例中的潜在应用。此外，我们还提出了一次灾难响应案例研究，展示了LLM与空计算协作利用如何在关键时刻显著提高结果。 

---
# LADDER: Self-Improving LLMs Through Recursive Problem Decomposition 

**Title (ZH)**: 梯子：通过递归问题分解实现自我提升的大语言模型 

**Authors**: Toby Simonds, Akira Yoshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2503.00735)  

**Abstract**: We introduce LADDER (Learning through Autonomous Difficulty-Driven Example Recursion), a framework enabling LLMs to autonomously improve their problem-solving capabilities through self-guided learning. By recursively generating and solving progressively simpler variants of complex problems, LADDER enables models to progressively learn through reinforcement learning how to solve harder problems. This self-improvement process is guided by verifiable reward signals, allowing the model to assess its solutions. Unlike prior approaches requiring curated datasets or human feedback, LADDER leverages the model's own capabilities to easier variants of sample questions. We demonstrate LADDER's effectiveness on mathematical integration tasks, where it improves a Llama 3B model's accuracy from 1\% to 82\% on undergraduate-level problems and enables a 7B parameter model to achieve state-of-the-art performance (70\%) on the MIT Integration Bee examination for it's model size. We also introduce TTRL (Test-Time Reinforcement Learning), a method that generates variants of test problems at inference time and applies reinforcement learning to further improve performance. By further creating and solving related problems during testing, TTRL enables the 7B model to achieve a score of 85\%, surpassing o1. These results showcase how strategic self-directed learning can achieve significant capability improvements without relying on architectural scaling or human supervision. 

**Abstract (ZH)**: 我们介绍了LADDER（基于自主难度驱动示例递归的学习），这是一种框架，使大型语言模型能够通过自我引导学习自主提升其问题解决能力。通过递归生成并解决日益简单的复杂问题变体，LADDER使得模型能够通过强化学习逐步学习如何解决更难的问题。这一自我提升过程受到可验证的奖励信号的指导，允许模型评估其解决方案。与先前需要精心策划的数据集或人类反馈的方法不同，LADDER利用模型自身的能力解决示例问题的较简单变体。我们通过数学积分任务展示了LADDER的有效性，其中LADDER将一个3B参数Llama模型在本科水平问题上的准确性从1%提高到82%，并使一个7B参数模型在MIT Integration Bee考试中取得了与其模型规模相匹配的最佳性能（70%）。我们还介绍了TTRL（推理时的强化学习），这是一种方法，能够在推理时生成测试问题的变体，并应用强化学习进一步提高性能。通过在测试过程中进一步创建和解决问题，TTRL使7B参数模型取得了85%的成绩，超越了o1。这些结果展示了战略性自我导向学习如何在无需依赖架构扩展或人工监督的情况下实现显著的能力提升。 

---
# LLMDR: LLM-Driven Deadlock Detection and Resolution in Multi-Agent Pathfinding 

**Title (ZH)**: LLMDR：基于大规模语言模型的多智能体路径规划中的死锁检测与解决 

**Authors**: Seungbae Seo, Junghwan Kim, Minjeong Shin, Bongwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00717)  

**Abstract**: Multi-Agent Pathfinding (MAPF) is a core challenge in multi-agent systems. Existing learning-based MAPF methods often struggle with scalability, particularly when addressing complex scenarios that are prone to deadlocks. To address these challenges, we introduce LLMDR (LLM-Driven Deadlock Detection and Resolution), an approach designed to resolve deadlocks and improve the performance of learnt MAPF models. LLMDR integrates the inference capabilities of large language models (LLMs) with learnt MAPF models and prioritized planning, enabling it to detect deadlocks and provide customized resolution strategies. We evaluate LLMDR on standard MAPF benchmark maps with varying agent numbers, measuring its performance when combined with several base models. The results demonstrate that LLMDR improves the performance of learnt MAPF models, particularly in deadlock-prone scenarios, with notable improvements in success rates. These findings show the potential of integrating LLMs to improve the scalability of learning-based MAPF methods.
The source code for LLMDR is available at: this https URL 

**Abstract (ZH)**: 多智能体路径finding (MAPF) 是多智能体系统中的核心挑战。现有的基于学习的MAPF方法通常在处理易产生死锁的复杂场景时面临可扩展性问题。为了解决这些挑战，我们引入了LLMDR（LLM驱动的死锁检测与解决），这是一种旨在解决死锁并提高学习到的MAPF模型性能的方法。LLMDR将大型语言模型（LLM）的推理能力与学习到的MAPF模型和优先规划相结合，使其能够检测死锁并提供定制化的解决策略。我们在不同智能体数量的标准MAPF基准地图上评估了LLMDR，测量其与几种基本模型结合时的性能。结果表明，LLMDR能够提高学习到的MAPF模型的性能，特别是在易产生死锁的场景中，成功率有显著提升。这些发现展示了将LLM整合以提高基于学习的MAPF方法可扩展性的潜力。 

---
# Speculative Ad-hoc Querying 

**Title (ZH)**: 投机性即席查询 

**Authors**: Haoyu Li, Srikanth Kandula, Maria Angels de Luis Balaguer, Aditya Akella, Venkat Arun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00714)  

**Abstract**: Analyzing large datasets requires responsive query execution, but executing SQL queries on massive datasets can be slow. This paper explores whether query execution can begin even before the user has finished typing, allowing results to appear almost instantly. We propose SpeQL, a system that leverages Large Language Models (LLMs) to predict likely queries based on the database schema, the user's past queries, and their incomplete query. Since exact query prediction is infeasible, SpeQL speculates on partial queries in two ways: 1) it predicts the query structure to compile and plan queries in advance, and 2) it precomputes smaller temporary tables that are much smaller than the original database, but are still predicted to contain all information necessary to answer the user's final query. Additionally, SpeQL continuously displays results for speculated queries and subqueries in real time, aiding exploratory analysis. A utility/user study showed that SpeQL improved task completion time, and participants reported that its speculative display of results helped them discover patterns in the data more quickly. In the study, SpeQL improves user's query latency by up to $289\times$ and kept the overhead reasonable, at $\$4$ per hour. 

**Abstract (ZH)**: 基于大语言模型的预测查询执行系统SpeQL：几乎即时的结果显示 

---
# How Diversely Can Language Models Solve Problems? Exploring the Algorithmic Diversity of Model-Generated Code 

**Title (ZH)**: 语言模型能够多样化地解决问题吗？探索模型生成代码的算法多样性 

**Authors**: Seonghyeon Lee, Heejae Chon, Joonwon Jang, Dongha Lee, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00691)  

**Abstract**: Language models (LMs) have exhibited impressive abilities in generating code from natural language requirements. In this work, we highlight the diversity of code generated by LMs as a critical criterion for evaluating their code generation capabilities. There is a lack of studies focused on assessing the diversity of generated code, which overlooks its importance in code LMs. Therefore, we propose a systematic approach to evaluate code diversity, introducing various metrics with inter-code similarity. Specifically, we introduce code clustering methods that leverages LMs' capabilities in code understanding and reasoning, resulting in a set of metrics that represent the number of algorithms in model-generated solutions. We extensively investigate the property of model-generated solutions by contrasting them with human-written ones and quantifying the impact of various factors on code diversity: model size, temperature, instruction tuning, and problem complexity. Our analysis demonstrates that model-generated solutions exhibit low algorithmic diversity, which was neglected by the research community. Moreover, we explore methods to increase code diversity by combining solutions from different models and increasing sampling temperatures. Our findings highlight that code diversity can be enhanced with the help of heterogeneous models and setting temperature beyond 1.0 that has not been fully explored due to the functional correctness degradation. To facilitate our research direction, we publicly share our code and datasets through open-source repositories. 

**Abstract (ZH)**: 语言模型生成的代码多样性及其评估：一种系统的视角 

---
# GPIoT: Tailoring Small Language Models for IoT Program Synthesis and Development 

**Title (ZH)**: GPIoT: 优化小型语言模型以适应物联网程序合成与开发 

**Authors**: Leming Shen, Qiang Yang, Xinyu Huang, Zijing Ma, Yuanqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00686)  

**Abstract**: Code Large Language Models (LLMs) enhance software development efficiency by automatically generating code and documentation in response to user requirements. However, code LLMs cannot synthesize specialized programs when tasked with IoT applications that require domain knowledge. While Retrieval-Augmented Generation (RAG) offers a promising solution by fetching relevant domain knowledge, it necessitates powerful cloud LLMs (e.g., GPT-4) to process user requirements and retrieved contents, which raises significant privacy concerns. This approach also suffers from unstable networks and prohibitive LLM query costs. Moreover, it is challenging to ensure the correctness and relevance of the fetched contents. To address these issues, we propose GPIoT, a code generation system for IoT applications by fine-tuning locally deployable Small Language Models (SLMs) on IoT-specialized datasets. SLMs have smaller model sizes, allowing efficient local deployment and execution to mitigate privacy concerns and network uncertainty. Furthermore, by fine-tuning the SLMs with our IoT-specialized datasets, the SLMs' ability to synthesize IoT-related programs can be substantially improved. To evaluate GPIoT's capability in synthesizing programs for IoT applications, we develop a benchmark, IoTBench. Extensive experiments and user trials demonstrate the effectiveness of GPIoT in generating IoT-specialized code, outperforming state-of-the-art code LLMs with an average task accuracy increment of 64.7% and significant improvements in user satisfaction. 

**Abstract (ZH)**: Code生成系统GPIoT通过细调本地部署的小型语言模型提升物联网应用开发效率 

---
# Efficiently Editing Mixture-of-Experts Models with Compressed Experts 

**Title (ZH)**: 高效编辑混合专家模型中的压缩专家 

**Authors**: Yifei He, Yang Liu, Chen Liang, Hany Hassan Awadalla  

**Link**: [PDF](https://arxiv.org/pdf/2503.00634)  

**Abstract**: Mixture-of-Experts (MoE) models have become a key approach for scaling large language models efficiently by activating only a subset of experts during training and inference. Typically, the number of activated experts presents a trade-off: fewer experts reduce computational costs, while more experts improve performance. Recent studies reveal that not all activated experts contribute equally to model performance, with some providing minimal utility, particularly when finetuning pretrained MoE models for specialized downstream tasks. The co-existence of significant and redundant parameters in experts provides us an opportunity to reduce the number of activated experts while maintaining model performance. In this work, we propose the concept of compressed experts, lightweight modules that serve as compact representations of full experts. Our approach preserves the most important experts while replacing other auxiliary activated experts with compressed experts. The reduction of active parameters significantly lowers inference costs while achieving comparable performance. Extensive experiments on models including Phi-MoE and OLMoE demonstrate that compressed experts recover over 90% of full expert performance across various tasks while reducing more than 30% active parameters and saving 20% in inference costs. This approach enables efficient deployment of MoE models in resource-constrained settings and facilitates scaling to larger models with manageable overhead. Our code is available at this https URL. 

**Abstract (ZH)**: MoE模型的压缩专家：在保持性能的同时减少激活专家的数量 

---
# An evaluation of DeepSeek Models in Biomedical Natural Language Processing 

**Title (ZH)**: DeepSeek模型在生物医学自然语言处理中的评估 

**Authors**: Zaifu Zhan, Shuang Zhou, Huixue Zhou, Jiawen Deng, Yu Hou, Jeremy Yeung, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00624)  

**Abstract**: The advancement of Large Language Models (LLMs) has significantly impacted biomedical Natural Language Processing (NLP), enhancing tasks such as named entity recognition, relation extraction, event extraction, and text classification. In this context, the DeepSeek series of models have shown promising potential in general NLP tasks, yet their capabilities in the biomedical domain remain underexplored. This study evaluates multiple DeepSeek models (Distilled-DeepSeek-R1 series and Deepseek-LLMs) across four key biomedical NLP tasks using 12 datasets, benchmarking them against state-of-the-art alternatives (Llama3-8B, Qwen2.5-7B, Mistral-7B, Phi-4-14B, Gemma-2-9B). Our results reveal that while DeepSeek models perform competitively in named entity recognition and text classification, challenges persist in event and relation extraction due to precision-recall trade-offs. We provide task-specific model recommendations and highlight future research directions. This evaluation underscores the strengths and limitations of DeepSeek models in biomedical NLP, guiding their future deployment and optimization. 

**Abstract (ZH)**: 大型语言模型的进步显著影响了生物医学自然语言处理，增强了命名实体识别、关系抽取、事件抽取和文本分类等任务。在此背景下，DeepSeek系列模型在通用自然语言处理任务中展现了潜在能力，但在生物医学领域的能力尚未充分探索。本研究使用12个数据集，在四种关键的生物医学自然语言处理任务中评估了多个DeepSeek模型（Distilled-DeepSeek-R1系列和DeepSeek-LLMs），并将它们与最先进的替代方案（Llama3-8B、Qwen2.5-7B、Mistral-7B、Phi-4-14B、Gemma-2-9B）进行基准比较。研究结果表明，虽然DeepSeek模型在命名实体识别和文本分类任务中表现出竞争力，但在事件和关系抽取任务中存在精确度与召回率之间的权衡问题。我们提供了特定任务的模型推荐，并指出了未来的研究方向。该评估突显了DeepSeek模型在生物医学自然语言处理中的优势与局限性，指导其未来的部署与优化。 

---
# Urban Safety Perception Through the Lens of Large Multimodal Models: A Persona-based Approach 

**Title (ZH)**: 基于个性化的视角通过大型多模态模型探索城市安全感知 

**Authors**: Ciro Beneduce, Bruno Lepri, Massimiliano Luca  

**Link**: [PDF](https://arxiv.org/pdf/2503.00610)  

**Abstract**: Understanding how urban environments are perceived in terms of safety is crucial for urban planning and policymaking. Traditional methods like surveys are limited by high cost, required time, and scalability issues. To overcome these challenges, this study introduces Large Multimodal Models (LMMs), specifically Llava 1.6 7B, as a novel approach to assess safety perceptions of urban spaces using street-view images. In addition, the research investigated how this task is affected by different socio-demographic perspectives, simulated by the model through Persona-based prompts. Without additional fine-tuning, the model achieved an average F1-score of 59.21% in classifying urban scenarios as safe or unsafe, identifying three key drivers of perceived unsafety: isolation, physical decay, and urban infrastructural challenges. Moreover, incorporating Persona-based prompts revealed significant variations in safety perceptions across the socio-demographic groups of age, gender, and nationality. Elder and female Personas consistently perceive higher levels of unsafety than younger or male Personas. Similarly, nationality-specific differences were evident in the proportion of unsafe classifications ranging from 19.71% in Singapore to 40.15% in Botswana. Notably, the model's default configuration aligned most closely with a middle-aged, male Persona. These findings highlight the potential of LMMs as a scalable and cost-effective alternative to traditional methods for urban safety perceptions. While the sensitivity of these models to socio-demographic factors underscores the need for thoughtful deployment, their ability to provide nuanced perspectives makes them a promising tool for AI-driven urban planning. 

**Abstract (ZH)**: 理解城市环境在安全感知方面的认知对于城市规划和政策制定至关重要。传统的调查方法受限于高成本、时间消耗和可扩展性问题。为克服这些挑战，本研究引入了大型多模态模型（LMMs），具体为Llava 1.6 7B，借助街道视角图片评估城市空间的安全感知。此外，研究还探讨了不同社会人口统计学视角对这一任务的影响，模拟这一过程使用基于人设的提示。未经额外微调，该模型在分类城市场景为安全或不安全方面平均F1分数为59.21%，并识别出不安全感的三个主要驱动因素：隔离、物理衰退和城市基础设施挑战。同时，加入基于人设的提示揭示了不同社会人口统计学群体（年龄、性别和国籍）在安全感知方面的显著差异。老年人和女性人设的一贯感知到更高水平的不安全感，而年轻人和男性感知到较低水平的不安全感。同样，国籍在不安全分类的比例上也有显著差异，从新加坡的19.71%到博茨瓦纳的40.15%不等。值得注意的是，模型的默认配置最接近中年男性人设。这些发现强调了LMMs作为可扩展且成本效益高的替代传统方法评估城市安全感知的潜力。尽管这些模型对社会人口统计学因素的敏感性突显了慎重部署的需要，但它们提供细致视角的能力使得它们成为AI驱动城市规划的有前景工具。 

---
# Semantic Integrity Constraints: Declarative Guardrails for AI-Augmented Data Processing Systems 

**Title (ZH)**: 语义完整约束：面向AI增强数据处理系统的声明式护栏 

**Authors**: Alexander W. Lee, Justin Chan, Michael Fu, Nicolas Kim, Akshay Mehta, Deepti Raghavan, Ugur Cetintemel  

**Link**: [PDF](https://arxiv.org/pdf/2503.00600)  

**Abstract**: The emergence of AI-augmented Data Processing Systems (DPSs) has introduced powerful semantic operators that extend traditional data management capabilities with LLM-based processing. However, these systems face fundamental reliability (a.k.a. trust) challenges, as LLMs can generate erroneous outputs, limiting their adoption in critical domains. Existing approaches to LLM constraints--ranging from user-defined functions to constrained decoding--are fragmented, imperative, and lack semantics-aware integration into query execution. To address this gap, we introduce Semantic Integrity Constraints (SICs), a novel declarative abstraction that extends traditional database integrity constraints to govern and optimize semantic operators within DPSs. SICs integrate seamlessly into the relational model, allowing users to specify common classes of constraints (e.g., grounding and soundness) while enabling query-aware enforcement and optimization strategies.
In this paper, we present the core design of SICs, describe their formal integration into query execution, and detail our conception of grounding constraints, a key SIC class that ensures factual consistency of generated outputs. In addition, we explore novel enforcement mechanisms, combining proactive (constrained decoding) and reactive (validation and recovery) techniques to optimize efficiency and reliability. Our work establishes SICs as a foundational framework for trustworthy, high-performance AI-augmented data processing, paving the way for future research in constraint-driven optimizations, adaptive enforcement, and enterprise-scale deployments. 

**Abstract (ZH)**: AI增强数据处理系统中的语义完整性约束（SICs）：可靠高效的语义数据处理基础框架 

---
# Zero-Shot Keyphrase Generation: Investigating Specialized Instructions and Multi-Sample Aggregation on Large Language Models 

**Title (ZH)**: 零样本关键短语生成：探究专门指令和多样本聚合在大型语言模型中的应用 

**Authors**: Jayanth Mohan, Jishnu Ray Chowdhury, Tomas Malik, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2503.00597)  

**Abstract**: Keyphrases are the essential topical phrases that summarize a document. Keyphrase generation is a long-standing NLP task for automatically generating keyphrases for a given document. While the task has been comprehensively explored in the past via various models, only a few works perform some preliminary analysis of Large Language Models (LLMs) for the task. Given the impact of LLMs in the field of NLP, it is important to conduct a more thorough examination of their potential for keyphrase generation. In this paper, we attempt to meet this demand with our research agenda. Specifically, we focus on the zero-shot capabilities of open-source instruction-tuned LLMs (Phi-3, Llama-3) and the closed-source GPT-4o for this task. We systematically investigate the effect of providing task-relevant specialized instructions in the prompt. Moreover, we design task-specific counterparts to self-consistency-style strategies for LLMs and show significant benefits from our proposals over the baselines. 

**Abstract (ZH)**: 基于大规模语言模型的关键短语生成研究 

---
# BadJudge: Backdoor Vulnerabilities of LLM-as-a-Judge 

**Title (ZH)**: BadJudge：作为法官的LLM的后门易感性 

**Authors**: Terry Tong, Fei Wang, Zhe Zhao, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00596)  

**Abstract**: This paper proposes a novel backdoor threat attacking the LLM-as-a-Judge evaluation regime, where the adversary controls both the candidate and evaluator model. The backdoored evaluator victimizes benign users by unfairly assigning inflated scores to adversary. A trivial single token backdoor poisoning 1% of the evaluator training data triples the adversary's score with respect to their legitimate score. We systematically categorize levels of data access corresponding to three real-world settings, (1) web poisoning, (2) malicious annotator, and (3) weight poisoning. These regimes reflect a weak to strong escalation of data access that highly correlates with attack severity. Under the weakest assumptions - web poisoning (1), the adversary still induces a 20% score inflation. Likewise, in the (3) weight poisoning regime, the stronger assumptions enable the adversary to inflate their scores from 1.5/5 to 4.9/5. The backdoor threat generalizes across different evaluator architectures, trigger designs, evaluation tasks, and poisoning rates. By poisoning 10% of the evaluator training data, we control toxicity judges (Guardrails) to misclassify toxic prompts as non-toxic 89% of the time, and document reranker judges in RAG to rank the poisoned document first 97% of the time. LLM-as-a-Judge is uniquely positioned at the intersection of ethics and technology, where social implications of mislead model selection and evaluation constrain the available defensive tools. Amidst these challenges, model merging emerges as a principled tool to offset the backdoor, reducing ASR to near 0% whilst maintaining SOTA performance. Model merging's low computational cost and convenient integration into the current LLM Judge training pipeline position it as a promising avenue for backdoor mitigation in the LLM-as-a-Judge setting. 

**Abstract (ZH)**: 本论文提出了一种针对LLM-as-a-Judge评估机制的新颖后门威胁，攻击者控制候选模型和评估模型。后门评估模型通过不公平地给予攻击者夸大分数来损害无辜用户。单一令牌后门污染评估模型训练数据的1%，可以使攻击者的分数相对于其合法分数几乎翻三倍。我们系统地将数据访问级别分类为三种现实场景：（1）网络污染、（2）恶意标注者和（3）权重污染。这些机制反映出从弱到强的数据访问提升，与攻击严重性高度相关。即使在最弱假设下（网络污染），攻击者仍能引发20%的分数膨胀。同样，在权重污染机制（第三种情况）下，更强的假设使攻击者能够将分数从1.5/5提升到4.9/5。后门威胁在不同的评估模型架构、触发设计、评估任务和污染率下具有通用性。通过污染评估模型训练数据的10%，我们能够控制毒性裁判（Guardrails）将有毒提示误分类为无毒的89%时间，并使文档重排序裁判在RAG中将污染文档排在首位的几率达到97%。LLM-as-a-Judge在伦理和技术的交叉点上独具优势，社会模型选择和评估误导的潜在影响限制了可用的防御工具。在这些挑战中，模型合并作为一种原则性的工具，能够缓解后门威胁，同时保持SOTA性能并将ASR降至几乎为零。模型合并的低计算成本和容易整合到当前LLM Judge训练管道中，使其成为LLM-as-a-Judge环境中后门减轻的一种有前途的途径。 

---
# LoR2C : Low-Rank Residual Connection Adaptation for Parameter-Efficient Fine-Tuning 

**Title (ZH)**: LoR2C：低秩残差连接适应性调整用于参数高效微调 

**Authors**: Jiancheng Zhao, Xingda Yu, Yuxiang Zhang, Zhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00572)  

**Abstract**: In recent years, pretrained large language models have demonstrated outstanding performance across various natural language processing tasks. However, full-parameter fine-tuning methods require adjusting all model parameters, leading to immense computational resource demands. Although parameter-efficient fine-tuning methods like LoRA have significantly reduced the number of parameters, they still face challenges such as gradient vanishing and the potential for further parameter reduction. To address these issues, this paper proposes a novel parameter-efficient fine-tuning method called LoR2C (Low-Rank Residual Connection Adaptation). LoR2C introduces residual connections with low-rank matrices within the model layers, which not only reduces the number of fine-tuning parameters but also effectively alleviates the gradient vanishing problem. Additionally, this paper presents three optimization variants of LoR2C: ShareLoR2C, MergeLoR2C, and InjectLoR2C. These variants further improve parameter efficiency and model performance through parameter sharing, module merging, and injection mechanisms, respectively. Experimental results on multiple natural language understanding and natural language generation tasks demonstrate that LoR2C and its optimized variants significantly reduce parameter overhead while maintaining or even improving performance, outperforming existing mainstream parameter-efficient fine-tuning this http URL code is publicly available at this https URL. 

**Abstract (ZH)**: 近年来，预训练大型语言模型在各种自然语言处理任务中展现了出色的性能。然而，全参数微调方法需要调整所有模型参数，导致巨大的计算资源需求。尽管LoRA等参数高效微调方法显著减少了参数数量，但仍面临梯度消失和其他参数进一步减少的挑战。为解决这些问题，本文提出了一种新的参数高效微调方法——LoR2C（低秩残差连接适应）。LoR2C在模型层内引入低秩矩阵的残差连接，不仅减少了需要微调的参数数量，还有效地缓解了梯度消失问题。此外，本文还提出了LoR2C的三种优化变体：ShareLoR2C、MergeLoR2C和InjectLoR2C。这些变体分别通过参数共享、模块合并和注入机制进一步提高了参数效率和模型性能。多项自然语言理解与自然语言生成任务的实验结果表明，LoR2C及其优化变体在显著减少参数开销的同时，能够保持或甚至提高性能，并在现有主流参数高效微调方法中表现出色。代码已公开，可在以下链接获取：this https URL 

---
# Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable 

**Title (ZH)**: 安全税：安全对齐使你的大型推理模型变得不够合理 

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Zachary Yahn, Yichang Xu, Ling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00555)  

**Abstract**: Safety alignment is an important procedure before the official deployment of a Large Language Model (LLM). While safety alignment has been extensively studied for LLM, there is still a large research gap for Large Reasoning Models (LRMs) that equip with improved reasoning capability. We in this paper systematically examine a simplified pipeline for producing safety aligned LRMs. With our evaluation of various LRMs, we deliver two main findings: i) Safety alignment can be done upon the LRM to restore its safety capability. ii) Safety alignment leads to a degradation of the reasoning capability of LRMs. The two findings show that there exists a trade-off between reasoning and safety capability with the sequential LRM production pipeline. The discovered trade-off, which we name Safety Tax, should shed light on future endeavors of safety research on LRMs. As a by-product, we curate a dataset called DirectRefusal, which might serve as an alternative dataset for safety alignment. Our source code is available at this https URL. 

**Abstract (ZH)**: 大规模推理模型的安全对齐是其正式部署前的重要程序。虽然已经对大规模语言模型（LLM）进行了广泛的安全对齐研究，但对于具备增强推理能力的大规模推理模型（LRMs）来说，安全对齐研究仍然存在巨大的研究空白。本文系统地研究了一个简化的生成安全对齐LRMs的管线。通过对多种LRMs的评估，我们得到了两个主要发现：i) 可以对LRM进行安全对齐以恢复其安全能力；ii) 安全对齐会导致LRMs推理能力的下降。这两个发现表明，在顺序生成LRM管线中存在推理能力和安全能力之间的权衡。我们发现的这种权衡现象，我们称之为“安全税”，应该会为未来关于LRMs安全研究的方向带来启示。作为副产品，我们收集了一个名为DirectRefusal的数据集，该数据集可能作为安全对齐的替代数据集。我们的源代码可在以下链接获取。 

---
# Distributionally Robust Reinforcement Learning with Human Feedback 

**Title (ZH)**: 基于人类反馈的分布鲁棒强化学习 

**Authors**: Debmalya Mandal, Paulius Sasnauskas, Goran Radanovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.00539)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has evolved to be one of the main methods for fine-tuning large language models (LLMs). However, existing RLHF methods are non-robust, and their performance deteriorates if the downstream task differs significantly from the preference dataset used in fine-tuning. In order to mitigate this problem, we introduce a distributionally robust RLHF for fine-tuning LLMs. In particular, our goal is to ensure that a fine-tuned model retains its performance even when the distribution of prompts significantly differs from the distribution encountered during fine-tuning. We formulate distributionally robust optimization (DRO) version of two popular fine-tuning methods -- (1) reward-based RLHF and (2) reward-free DPO (direct preference optimization). We propose a minibatch gradient descent based algorithms for both of them, and theoretically prove convergence guarantees for the algorithms. Subsequently, we evaluate our algorithms on an out-of-distribution (OOD) task by first training the model on the Unified-Feedback dataset and evaluating its performance on two different datasets. The experimental results show that our robust training improves the accuracy of the learned reward models on average, and markedly on some tasks, such as reasoning. Furthermore, we show that the robust versions of policy optimization methods, similarly improve performance on OOD tasks. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）已发展成为细调大型语言模型（LLMs）的主要方法之一。然而，现有的RLHF方法不够健壮，如果下游任务与用于细调的偏好数据集显著不同，其性能会下降。为了缓解这一问题，我们提出了一个分布鲁棒的RLHF方法以细调LLMs。具体而言，我们的目标是在提示分布与细调过程中遇到的分布显著不同的情况下，确保细调后的模型仍能保持其性能。我们为两种流行的细调方法——基于奖励的RLHF和无奖励的DPO（直接偏好优化）——制定了分布鲁棒优化（DRO）版本，并提出基于小批量梯度下降的算法，理论上证明了算法的收敛性保证。随后，我们在一个离分布（OOD）任务中评估了这些算法：首先在统一反馈数据集中训练模型，然后在两个不同的数据集上评估其性能。实验结果表明，我们的鲁棒训练方法在平均情况下提高了学习奖励模型的准确性，并且在某些任务（如推理）上显著提高了准确性。此外，我们展示了用于策略优化的鲁棒版本方法同样能够提高离分布任务的性能。 

---
# LLaSE-G1: Incentivizing Generalization Capability for LLaMA-based Speech Enhancement 

**Title (ZH)**: LLaSE-G1: 激励基于LLaMA的语音增强的一般化能力 

**Authors**: Boyi Kang, Xinfa Zhu, Zihan Zhang, Zhen Ye, Mingshuai Liu, Ziqian Wang, Yike Zhu, Guobin Ma, Jun Chen, Longshuai Xiao, Chao Weng, Wei Xue, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.00493)  

**Abstract**: Recent advancements in language models (LMs) have demonstrated strong capabilities in semantic understanding and contextual modeling, which have flourished in generative speech enhancement (SE). However, many LM-based SE approaches primarily focus on semantic information, often neglecting the critical role of acoustic information, which leads to acoustic inconsistency after enhancement and limited generalization across diverse SE tasks. In this paper, we introduce LLaSE-G1, a LLaMA-based language model that incentivizes generalization capabilities for speech enhancement. LLaSE-G1 offers the following key contributions: First, to mitigate acoustic inconsistency, LLaSE-G1 employs continuous representations from WavLM as input and predicts speech tokens from X-Codec2, maximizing acoustic preservation. Second, to promote generalization capability, LLaSE-G1 introduces dual-channel inputs and outputs, unifying multiple SE tasks without requiring task-specific IDs. Third, LLaSE-G1 outperforms prior task-specific discriminative and generative SE models, demonstrating scaling effects at test time and emerging capabilities for unseen SE tasks. Additionally, we release our code and models to support further research in this area. 

**Abstract (ZH)**: 最近在语言模型方面的进展展示了强大的语义理解和上下文建模能力，这些能力在生成性语音增强中得到了发展。然而，许多基于语言模型的语音增强方法主要关注语义信息，往往忽视了声学信息的关键作用，导致增强后声学不一致，并限制了在不同语音增强任务中的泛化能力。本文介绍了一种基于LLaMA的语言模型LLaSE-G1，该模型旨在提高语音增强的泛化能力。LLaSE-G1的主要贡献如下：首先，为缓解声学不一致问题，LLaSE-G1采用WavLM的连续表示作为输入，并从X-Codec2预测语音标记，最大化声学信息的保留。其次，为促进泛化能力，LLaSE-G1引入了双通道输入和输出，能够统一多种语音增强任务而无需特定任务标识符。第三，LLaSE-G1在测试时展现了优于先前任务特定的判别和生成语音增强模型的效果，并展示了对未见过的语音增强任务的新兴能力。此外，我们还公开了我们的代码和模型，以支持该领域的进一步研究。 

---
# Challenges in Testing Large Language Model Based Software: A Faceted Taxonomy 

**Title (ZH)**: 基于大型语言模型的软件测试挑战：一个多维度分类体系 

**Authors**: Felix Dobslaw, Robert Feldt, Juyeon Yoon, Shin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00481)  

**Abstract**: Large Language Models (LLMs) and Multi-Agent LLMs (MALLMs) introduce non-determinism unlike traditional or machine learning software, requiring new approaches to verifying correctness beyond simple output comparisons or statistical accuracy over test datasets.
This paper presents a taxonomy for LLM test case design, informed by both the research literature, our experience, and open-source tools that represent the state of practice. We identify key variation points that impact test correctness and highlight open challenges that the research, industry, and open-source communities must address as LLMs become integral to software systems.
Our taxonomy defines four facets of LLM test case design, addressing ambiguity in both inputs and outputs while establishing best practices. It distinguishes variability in goals, the system under test, and inputs, and introduces two key oracle types: atomic and aggregated. Our mapping indicates that current tools insufficiently account for these variability points, highlighting the need for closer collaboration between academia and practitioners to improve the reliability and reproducibility of LLM testing. 

**Abstract (ZH)**: 大型语言模型（LLMs）和多代理大型语言模型（MALLMs）引入了不同于传统或机器学习软件的不确定性，需要超越简单输出比对或测试数据集上统计准确性的新验证方法。本文提出了一种LLM测试案例设计的分类体系，参考了研究文献、我们的经验和开源工具代表的最佳实践。我们识别出影响测试正确性的关键变化点，并强调随着LLMs成为软件系统的核心组成部分，研究界、工业界和开源社区需要共同解决的关键挑战。本文的分类体系定义了四种LLM测试案例设计的维度，解决了输入和输出的模糊性，并建立了最佳实践。它区分了目标、被测系统和输入的变异性，并引入了两种关键的判别方法：原子判别法和聚合判别法。我们的映射表明，当前工具未能充分考虑到这些变异性点，突显了学术界和实践者之间需要更紧密合作以提高LLM测试的可靠性和可重复性的重要性。 

---
# Leveraging Compute-in-Memory for Efficient Generative Model Inference in TPUs 

**Title (ZH)**: 基于计算集成内存的TPU高效生成模型推理 

**Authors**: Zhantong Zhu, Hongou Li, Wenjie Ren, Meng Wu, Le Ye, Ru Huang, Tianyu Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.00461)  

**Abstract**: With the rapid advent of generative models, efficiently deploying these models on specialized hardware has become critical. Tensor Processing Units (TPUs) are designed to accelerate AI workloads, but their high power consumption necessitates innovations for improving efficiency. Compute-in-memory (CIM) has emerged as a promising paradigm with superior area and energy efficiency. In this work, we present a TPU architecture that integrates digital CIM to replace conventional digital systolic arrays in matrix multiply units (MXUs). We first establish a CIM-based TPU architecture model and simulator to evaluate the benefits of CIM for diverse generative model inference. Building upon the observed design insights, we further explore various CIM-based TPU architectural design choices. Up to 44.2% and 33.8% performance improvement for large language model and diffusion transformer inference, and 27.3x reduction in MXU energy consumption can be achieved with different design choices, compared to the baseline TPUv4i architecture. 

**Abstract (ZH)**: 随着生成模型的迅速发展，高效地将这些模型部署在专用硬件上变得至关重要。Tensor Processing Units (TPU) 设计用于加速AI工作负载，但由于其高功耗，需要创新来提高效率。计算存储一体（CIM）作为一种前景广阔的范式，具有卓越的面积和能量效率。在本工作中，我们提出了一种TPU架构，将数字CIM集成到矩阵乘法单元（MXU）中以替代传统的数字 systolic 数组。首先，我们建立了一种基于CIM的TPU架构模型和模拟器，以评估CIM在不同生成模型推理中的优势。基于观察到的设计见解，我们进一步探索了各种基于CIM的TPU架构设计选择。与基线TPUv4i架构相比，不同的设计选择可实现高达44.2%的大语言模型和扩散变换器推理性能提升，33.8%的MXU能量消耗减少，以及27.3倍的MXU能量消耗降低。 

---
# PodAgent: A Comprehensive Framework for Podcast Generation 

**Title (ZH)**: PodAgent：播客生成的综合框架 

**Authors**: Yujia Xiao, Lei He, Haohan Guo, Fenglong Xie, Tan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.00455)  

**Abstract**: Existing Existing automatic audio generation methods struggle to generate podcast-like audio programs effectively. The key challenges lie in in-depth content generation, appropriate and expressive voice production. This paper proposed PodAgent, a comprehensive framework for creating audio programs. PodAgent 1) generates informative topic-discussion content by designing a Host-Guest-Writer multi-agent collaboration system, 2) builds a voice pool for suitable voice-role matching and 3) utilizes LLM-enhanced speech synthesis method to generate expressive conversational speech. Given the absence of standardized evaluation criteria for podcast-like audio generation, we developed comprehensive assessment guidelines to effectively evaluate the model's performance. Experimental results demonstrate PodAgent's effectiveness, significantly surpassing direct GPT-4 generation in topic-discussion dialogue content, achieving an 87.4% voice-matching accuracy, and producing more expressive speech through LLM-guided synthesis. Demo page: this https URL. Source code: this https URL. 

**Abstract (ZH)**: 现有自动音频生成方法在生成播客风格的音频节目时面临诸多挑战，特别是在深入内容生成和恰当有表现力的语音生产方面。本文提出了PodAgent，一个全面的音频节目创作框架。PodAgent通过设计主持人-嘉宾-撰稿人多代理协作系统来生成信息性主题讨论内容，构建语音池以实现合适的声音角色匹配，并利用增强语言模型的语音合成方法生成富有表现力的对话语音。鉴于缺乏播客风格音频生成的标准化评估标准，我们制定了全面的评估指南以有效评估模型性能。实验结果表明，PodAgent在主题讨论对话内容生成方面优于直接使用GPT-4生成方法，语音匹配准确率为87.4%，并通过语言模型引导合成生成更具表现力的语音。演示页面：this https URL. 代码仓库：this https URL。 

---
# Rehearse With User: Personalized Opinion Summarization via Role-Playing based on Large Language Models 

**Title (ZH)**: 基于大型语言模型的角色扮演进行个性化意见总结训练 

**Authors**: Yanyue Zhang, Yulan He, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00449)  

**Abstract**: Personalized opinion summarization is crucial as it considers individual user interests while generating product summaries. Recent studies show that although large language models demonstrate powerful text summarization and evaluation capabilities without the need for training data, they face difficulties in personalized tasks involving long texts. To address this, \textbf{Rehearsal}, a personalized opinion summarization framework via LLMs-based role-playing is proposed. Having the model act as the user, the model can better understand the user's personalized needs. Additionally, a role-playing supervisor and practice process are introduced to improve the role-playing ability of the LLMs, leading to a better expression of user needs. Furthermore, through suggestions from virtual users, the summary generation is intervened, ensuring that the generated summary includes information of interest to the user, thus achieving personalized summary generation. Experiment results demonstrate that our method can effectively improve the level of personalization in large model-generated summaries. 

**Abstract (ZH)**: 个性化意见总结至关重要，因为它能在生成产品总结时考虑个人用户兴趣。近期研究表明，尽管大语言模型无需训练数据即可表现出强大的文本总结和评估能力，但在涉及长文本的个性化任务中仍面临困难。为了解决这一问题，提出了一种基于大语言模型角色扮演的个性化意见总结框架——\textbf{Rehearsal}。让模型扮演用户的角色，可以使模型更好地理解用户的个性化需求。此外，引入了角色扮演监督者和练习过程，以提高大语言模型的角色扮演能力，并更好地表达用户需求。通过虚拟用户的建议，干预摘要生成过程，确保生成的摘要包含用户的兴趣信息，从而实现个性化摘要生成。实验结果表明，我们的方法可以有效提高大模型生成摘要的个性化水平。 

---
# Language Model Mapping in Multimodal Music Learning: A Grand Challenge Proposal 

**Title (ZH)**: 多模态音乐学习中的语言模型映射：一个重大挑战建议 

**Authors**: Daniel Chin, Gus Xia  

**Link**: [PDF](https://arxiv.org/pdf/2503.00427)  

**Abstract**: We have seen remarkable success in representation learning and language models (LMs) using deep neural networks. Many studies aim to build the underlying connections among different modalities via the alignment and mappings at the token or embedding level, but so far, most methods are very data-hungry, limiting their performance in domains such as music where paired data are less abundant. We argue that the embedding alignment is only at the surface level of multimodal alignment. In this paper, we propose a grand challenge of \textit{language model mapping} (LMM), i.e., how to map the essence implied in the LM of one domain to the LM of another domain under the assumption that LMs of different modalities are tracking the same underlying phenomena. We first introduce a basic setup of LMM, highlighting the goal to unveil a deeper aspect of cross-modal alignment as well as to achieve more sample-efficiency learning. We then discuss why music is an ideal domain in which to conduct LMM research. After that, we connect LMM in music with a more general and challenging scientific problem of \textit{learning to take actions based on both sensory input and abstract symbols}, and in the end, present an advanced version of the challenge problem setup. 

**Abstract (ZH)**: 语言模型映射：跨域语言模型之间的本质映射 

---
# Breaking the Loop: Detecting and Mitigating Denial-of-Service Vulnerabilities in Large Language Models 

**Title (ZH)**: 打破循环：检测和缓解大型语言模型中的拒绝服务漏洞 

**Authors**: Junzhe Yu, Yi Liu, Huijia Sun, Ling Shi, Yuqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00416)  

**Abstract**: Large Language Models (LLMs) have significantly advanced text understanding and generation, becoming integral to applications across education, software development, healthcare, entertainment, and legal services. Despite considerable progress in improving model reliability, latency remains under-explored, particularly through recurrent generation, where models repeatedly produce similar or identical outputs, causing increased latency and potential Denial-of-Service (DoS) vulnerabilities.
We propose RecurrentGenerator, a black-box evolutionary algorithm that efficiently identifies recurrent generation scenarios in prominent LLMs like LLama-3 and GPT-4o. Additionally, we introduce RecurrentDetector, a lightweight real-time classifier trained on activation patterns, achieving 95.24% accuracy and an F1 score of 0.87 in detecting recurrent loops. Our methods provide practical solutions to mitigate latency-related vulnerabilities, and we publicly share our tools and data to support further research. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本理解和生成方面取得了显著进步，成为教育、软件开发、医疗保健、娱乐和法律服务等领域的关键组成部分。尽管在提高模型可靠性方面取得了显着进展，但延迟问题仍被忽视，尤其是在循环生成中，模型反复生成相似或相同的输出，导致延迟增加和潜在的服务中断（DoS）漏洞。

我们提出了一种名为RecurrentGenerator的黑盒进化算法，能够有效地识别出LLaMA-3和GPT-4o等 prominent LLMs 中的循环生成场景。此外，我们还引入了一种轻量级的实时分类器RecurrentDetector，基于激活模式进行训练，其检测循环循环的准确率为95.24%，F1分为0.87。我们的方法提供了缓解延迟相关漏洞的实际解决方案，并且我们公开分享了我们的工具和数据以支持进一步的研究。 

---
# Smoothing Grounding and Reasoning for MLLM-Powered GUI Agents with Query-Oriented Pivot Tasks 

**Title (ZH)**: 面向查询导向枢纽任务的LLM驱动GUI代理的平滑对接与推理 

**Authors**: Zongru Wu, Pengzhou Cheng, Zheng Wu, Tianjie Ju, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00401)  

**Abstract**: Perception-enhanced pre-training, particularly through grounding techniques, is widely adopted to enhance the performance of graphical user interface (GUI) agents. However, in resource-constrained scenarios, the format discrepancy between coordinate-oriented grounding and action-oriented reasoning limits the effectiveness of grounding for reasoning tasks. To address this challenge, we propose a query-oriented pivot approach called query inference, which serves as a bridge between GUI grounding and reasoning. By inferring potential user queries from a screenshot and its associated element coordinates, query inference improves the understanding of coordinates while aligning more closely with reasoning tasks. Experimental results show that query inference outperforms previous grounding techniques under the same training data scale. Notably, query inference achieves comparable or even better performance to large-scale grounding-enhanced OS-Atlas with less than 0.1% of training data. Furthermore, we explore the impact of reasoning formats and demonstrate that integrating additional semantic information into the input further boosts reasoning performance. The code is publicly available athttps://github.com/ZrW00/GUIPivot. 

**Abstract (ZH)**: 基于查询导向的中间表示在图形用户界面代理中的应用：通过接地技术和增强预训练提升性能 

---
# Progressive Sparse Attention: Algorithm and System Co-design for Efficient Attention in LLM Serving 

**Title (ZH)**: 渐进稀疏注意力：高效语言模型服务中算法与系统协同设计的稀疏注意力机制 

**Authors**: Qihui Zhou, Peiqi Yin, Pengfei Zuo, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00392)  

**Abstract**: Processing long contexts has become a critical capability for modern large language models (LLMs). However, serving long-context LLMs comes with significant inference costs due to the high memory overhead of the key-value (KV) cache. Existing work leverages dynamic sparse attention algorithms (DSAes) to mitigate the KV cache overhead, but these algorithms rely on top-$k$ KV cache selection, which results in a trade-off between accuracy and efficiency. A larger $k$ improves accuracy but decreases efficiency, while a smaller $k$ boosts efficiency but compromises accuracy. To overcome this trade-off, this paper presents PSA, a $\underline{P}$rogressive $\underline{S}$parse $\underline{A}$ttention mechanism that integrates algorithmic innovations with system co-design to achieve both high inference accuracy and improved efficiency in LLM serving. The PSA algorithm adaptively adjusts the KV cache budget of different tokens and layers according to their real attention weight distributions, rather than relying on a fixed budget $k$. This enables high accuracy while minimizing KV cache usage. To further enhance execution efficiency, we introduce a pipelined iteration scheme that reduces CPU-GPU interleaving and synchronization overhead during PSA computation. Additionally, we implement unified GPU memory management that optimizes PSA's memory utilization by accounting for uneven memory requirements across different model layers. Extensive experimental results demonstrate that PSA reduces KV cache usage for attention computation by up to 2.4$\times$ and 8.8$\times$, and increases end-to-end serving throughput by up to 1.4$\times$ and 2.0$\times$, compared to state-of-the-art DSAes and systems without sparse attention, respectively. 

**Abstract (ZH)**: 渐进稀疏注意力机制：兼顾高效与高精度的长期上下文处理 

---
# Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding 

**Title (ZH)**: 八脚章鱼：通过动态对比解码减轻幻觉效应 

**Authors**: Wei Suo, Lijun Zhang, Mengyang Sun, Lin Yuanbo Wu, Peng Wang, Yanning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00361)  

**Abstract**: Large Vision-Language Models (LVLMs) have obtained impressive performance in visual content understanding and multi-modal reasoning. Unfortunately, these large models suffer from serious hallucination problems and tend to generate fabricated responses. Recently, several Contrastive Decoding (CD) strategies have been proposed to alleviate hallucination by introducing disturbed inputs. Although great progress has been made, these CD strategies mostly apply a one-size-fits-all approach for all input conditions. In this paper, we revisit this process through extensive experiments. Related results show that hallucination causes are hybrid and each generative step faces a unique hallucination challenge. Leveraging these meaningful insights, we introduce a simple yet effective Octopus-like framework that enables the model to adaptively identify hallucination types and create a dynamic CD workflow. Our Octopus framework not only outperforms existing methods across four benchmarks but also demonstrates excellent deployability and expansibility. Code is available at this https URL. 

**Abstract (ZH)**: 大型多模态模型（LVLMs）在视觉内容理解和多模态推理方面取得了显著性能。不幸的是，这些大型模型面临严重的幻觉问题，倾向于生成虚构的响应。近期，提出了一些对比解码（CD）策略，通过引入扰动输入来缓解幻觉问题。尽管取得了显著进展，但这些CD策略大多采用一刀切的方法适用于所有输入条件。在本文中，我们通过广泛的实验重新审视了这一过程。相关结果表明，幻觉的成因是多重的，每个生成步骤面临着独特的幻觉挑战。借助这些有价值的认识，我们引入了一个简单而有效的类似八爪鱼的框架，使模型能够自适应地识别幻觉类型并创建动态CD工作流。我们的Octopus框架不仅在四个基准测试中优于现有方法，而且还展示了出色的可部署性和扩展性。代码可在以下链接获取：this https URL。 

---
# BERT-based model for Vietnamese Fact Verification Dataset 

**Title (ZH)**: 基于BERT的越南语事实验证数据集模型 

**Authors**: Bao Tran, T. N. Khanh, Khang Nguyen Tuong, Thien Dang, Quang Nguyen, Nguyen T. Thinh, Vo T. Hung  

**Link**: [PDF](https://arxiv.org/pdf/2503.00356)  

**Abstract**: The rapid advancement of information and communication technology has facilitated easier access to information. However, this progress has also necessitated more stringent verification measures to ensure the accuracy of information, particularly within the context of Vietnam. This paper introduces an approach to address the challenges of Fact Verification using the Vietnamese dataset by integrating both sentence selection and classification modules into a unified network architecture. The proposed approach leverages the power of large language models by utilizing pre-trained PhoBERT and XLM-RoBERTa as the backbone of the network. The proposed model was trained on a Vietnamese dataset, named ISE-DSC01, and demonstrated superior performance compared to the baseline model across all three metrics. Notably, we achieved a Strict Accuracy level of 75.11\%, indicating a remarkable 28.83\% improvement over the baseline model. 

**Abstract (ZH)**: 信息和通信技术的快速进步促进了信息的更容易获取，但这一进展也 necessitated 更严格的验证措施以确保信息的准确性，特别是在越南的背景下。本文提出了一种方法，通过将句子选择和分类模块集成到统一的网络架构中来应对事实验证挑战，该方法利用越南语数据集进行研究。所提出的方法利用预训练的 PhoBERT 和 XLM-RoBERTa 作为网络的骨干，该模型在名为 ISE-DSC01 的越南语数据集上进行训练，并在所有三个指标上均显示出了优于基线模型的性能。值得注意的是，我们实现了严格的准确率75.11%，比基线模型提高了28.83%。 

---
# Structured Reasoning for Fairness: A Multi-Agent Approach to Bias Detection in Textual Data 

**Title (ZH)**: 结构化推理以公平性为目标：文本数据中偏见检测的多_agent方法 

**Authors**: Tianyi Huang, Elsa Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00355)  

**Abstract**: From disinformation spread by AI chatbots to AI recommendations that inadvertently reinforce stereotypes, textual bias poses a significant challenge to the trustworthiness of large language models (LLMs). In this paper, we propose a multi-agent framework that systematically identifies biases by disentangling each statement as fact or opinion, assigning a bias intensity score, and providing concise, factual justifications. Evaluated on 1,500 samples from the WikiNPOV dataset, the framework achieves 84.9% accuracy$\unicode{x2014}$an improvement of 13.0% over the zero-shot baseline$\unicode{x2014}$demonstrating the efficacy of explicitly modeling fact versus opinion prior to quantifying bias intensity. By combining enhanced detection accuracy with interpretable explanations, this approach sets a foundation for promoting fairness and accountability in modern language models. 

**Abstract (ZH)**: 从AI聊天机器人传播的假信息到AI推荐无意中强化刻板印象，文本偏见对大型语言模型（LLMs）的信任度构成了重大挑战。本文提出了一种多智能体框架，该框架系统地通过将每个陈述区分为事实或意见、分配偏见强度评分并提供简洁的事实依据来识别偏见。在WikiNPOV数据集的1,500个样本上评估，该框架的准确率达到了84.9%，比零样本基线提高了13.0%，展示了在量化偏见强度之前明确建模事实与意见的有效性。通过结合增强的检测准确性和可解释的解释，该方法为促进现代语言模型中的公平性和可问责性奠定了基础。 

---
# More of the Same: Persistent Representational Harms Under Increased Representation 

**Title (ZH)**: 更多的同一性：在增加代表性的背景下持久存在的表现性伤害 

**Authors**: Jennifer Mickel, Maria De-Arteaga, Leqi Liu, Kevin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.00333)  

**Abstract**: To recognize and mitigate the harms of generative AI systems, it is crucial to consider who is represented in the outputs of generative AI systems and how people are represented. A critical gap emerges when naively improving who is represented, as this does not imply bias mitigation efforts have been applied to address how people are represented. We critically examined this by investigating gender representation in occupation across state-of-the-art large language models. We first show evidence suggesting that over time there have been interventions to models altering the resulting gender distribution, and we find that women are more represented than men when models are prompted to generate biographies or personas. We then demonstrate that representational biases persist in how different genders are represented by examining statistically significant word differences across genders. This results in a proliferation of representational harms, stereotypes, and neoliberalism ideals that, despite existing interventions to increase female representation, reinforce existing systems of oppression. 

**Abstract (ZH)**: 为了识别并减轻生成性AI系统的危害，必须考虑生成性AI系统输出中谁得到了体现以及人们是如何被体现的。当天真地改进谁得到了体现时，这并不意味着已经应用了针对如何体现人们的问题进行的偏见缓解努力。我们通过调查最先进的大型语言模型中的职业性别代表性来严格审视这一点。我们首先展示了证据表明，随着时间的推移，对模型进行了干预以改变结果中的性别分布，并发现当模型被提示生成简历或角色时，女性比男性更加体现。然后我们通过检查不同性别间显著的词频差异，展示了代表性偏见依然存在。这导致了更多的代表性危害、刻板印象和新自由主义理念的盛行，即便已经存在增加女性代表性的干预措施，它们也巩固了现有的压迫体系。 

---
# Shifting Power: Leveraging LLMs to Simulate Human Aversion in ABMs of Bilateral Financial Exchanges, A bond market study 

**Title (ZH)**: 权力转移：利用大语言模型在双边金融交换ABM中模拟人类厌恶情绪——以债券市场研究为例 

**Authors**: Alicia Vidler, Toby Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00320)  

**Abstract**: Bilateral markets, such as those for government bonds, involve decentralized and opaque transactions between market makers (MMs) and clients, posing significant challenges for traditional modeling approaches. To address these complexities, we introduce TRIBE an agent-based model augmented with a large language model (LLM) to simulate human-like decision-making in trading environments. TRIBE leverages publicly available data and stylized facts to capture realistic trading dynamics, integrating human biases like risk aversion and ambiguity sensitivity into the decision-making processes of agents. Our research yields three key contributions: first, we demonstrate that integrating LLMs into agent-based models to enhance client agency is feasible and enriches the simulation of agent behaviors in complex markets; second, we find that even slight trade aversion encoded within the LLM leads to a complete cessation of trading activity, highlighting the sensitivity of market dynamics to agents' risk profiles; third, we show that incorporating human-like variability shifts power dynamics towards clients and can disproportionately affect the entire system, often resulting in systemic agent collapse across simulations. These findings underscore the emergent properties that arise when introducing stochastic, human-like decision processes, revealing new system behaviors that enhance the realism and complexity of artificial societies. 

**Abstract (ZH)**: 双边市场，如政府债券市场，涉及市场maker与客户之间的分散且不透明的交易，对传统建模方法提出了重大挑战。为应对这些复杂性，我们引入了TRIBE模型，该模型结合了大型语言模型（LLM），以模拟交易环境中类人的决策制定过程。TRIBE利用公开数据和统计事实来捕捉真实的交易动态，将人类偏差如风险厌恶和模糊性敏感性整合到代理决策过程之中。我们的研究贡献包括：首先，我们展示了将LLM整合到基于代理的模型中以增强客户自主性是可行的，并丰富了复杂市场的代理行为模拟；其次，我们发现编码在LLM中的轻微交易厌恶会导致完全停止交易活动，突显了市场动态对代理风险轮廓的敏感性；第三，我们证明了引入人类行为的变异性将权力动态向客户转移，并可能不成比例地影响整个系统，经常导致模拟中的系统性代理崩溃。这些发现强调了引入随机的、类人的决策过程时所出现的涌现性质，揭示了增强人造社会的真实性和复杂性的新系统行为。 

---
# Pseudo-Knowledge Graph: Meta-Path Guided Retrieval and In-Graph Text for RAG-Equipped LLM 

**Title (ZH)**: 伪知识图谱：元路径引导的检索与图内文本结合用于RAG装备的大语言模型 

**Authors**: Yuxin Yang, Haoyang Wu, Tao Wang, Jia Yang, Hao Ma, Guojie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00309)  

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized natural language processing. However, these models face challenges in retrieving precise information from vast datasets. Retrieval-Augmented Generation (RAG) was developed to combining LLMs with external information retrieval systems to enhance the accuracy and context of responses. Despite improvements, RAG still struggles with comprehensive retrieval in high-volume, low-information-density databases and lacks relational awareness, leading to fragmented answers.
To address this, this paper introduces the Pseudo-Knowledge Graph (PKG) framework, designed to overcome these limitations by integrating Meta-path Retrieval, In-graph Text and Vector Retrieval into LLMs. By preserving natural language text and leveraging various retrieval techniques, the PKG offers a richer knowledge representation and improves accuracy in information retrieval. Extensive evaluations using Open Compass and MultiHop-RAG benchmarks demonstrate the framework's effectiveness in managing large volumes of data and complex relationships. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现已经革新了自然语言处理。然而，这些模型在从庞大数据库中检索精确信息时面临挑战。检索增强生成（RAG）被开发出来，旨在将LLMs与外部信息检索系统结合以提高响应的准确性和上下文。尽管取得了进步，RAG在高体积低信息密度数据库中的全面检索仍然存在问题，并缺乏关系意识，导致答案碎片化。

为了解决这一问题，本文引入了伪知识图谱（PKG）框架，旨在通过将元路径检索、图内文本和向量检索整合到LLMs中来克服这些局限性。通过保留自然语言文本并利用多种检索技术，PKG提供了更丰富的知识表示，并在信息检索准确性方面有所提升。使用Open Compass和MultiHop-RAG基准的广泛评估证明了该框架在处理大量数据和复杂关系方面的有效性。 

---
# Reducing Large Language Model Safety Risks in Women's Health using Semantic Entropy 

**Title (ZH)**: 使用语义熵减少大型语言模型在女性健康方面安全风险 

**Authors**: Jahan C. Penny-Dimri, Magdalena Bachmann, William R. Cooke, Sam Mathewlynn, Samuel Dockree, John Tolladay, Jannik Kossen, Lin Li, Yarin Gal, Gabriel Davis Jones  

**Link**: [PDF](https://arxiv.org/pdf/2503.00269)  

**Abstract**: Large language models (LLMs) hold substantial promise for clinical decision support. However, their widespread adoption in medicine, particularly in healthcare, is hindered by their propensity to generate false or misleading outputs, known as hallucinations. In high-stakes domains such as women's health (obstetrics & gynaecology), where errors in clinical reasoning can have profound consequences for maternal and neonatal outcomes, ensuring the reliability of AI-generated responses is critical. Traditional methods for quantifying uncertainty, such as perplexity, fail to capture meaning-level inconsistencies that lead to misinformation. Here, we evaluate semantic entropy (SE), a novel uncertainty metric that assesses meaning-level variation, to detect hallucinations in AI-generated medical content. Using a clinically validated dataset derived from UK RCOG MRCOG examinations, we compared SE with perplexity in identifying uncertain responses. SE demonstrated superior performance, achieving an AUROC of 0.76 (95% CI: 0.75-0.78), compared to 0.62 (0.60-0.65) for perplexity. Clinical expert validation further confirmed its effectiveness, with SE achieving near-perfect uncertainty discrimination (AUROC: 0.97). While semantic clustering was successful in only 30% of cases, SE remains a valuable tool for improving AI safety in women's health. These findings suggest that SE could enable more reliable AI integration into clinical practice, particularly in resource-limited settings where LLMs could augment care. This study highlights the potential of SE as a key safeguard in the responsible deployment of AI-driven tools in women's health, leading to safer and more effective digital health interventions. 

**Abstract (ZH)**: 大型语言模型（LLMs）在临床决策支持方面展现出巨大的潜力。然而，它们在医学领域尤其是医疗健康中的广泛采用受到其生成虚假或误导性输出（称为幻觉）的倾向的阻碍。在如妇产科等高风险领域，临床推理中的错误可能会对母婴结果产生深远影响，因此确保AI生成响应的可靠性至关重要。传统的不确定性量化方法，如困惑度，无法捕捉导致错误信息的语义层面不一致性。我们评估了语义熵（SE），这是一种新型不确定性指标，可以评估语义层面的变化，用于检测AI生成的医疗内容中的幻觉。我们使用来自英国RCOG MRCOG考试的临床验证数据集进行了比较，发现SE在识别不确定响应方面优于困惑度，AUC ROC为0.76（95% CI：0.75-0.78），而困惑度为0.62（0.60-0.65）。临床专家验证进一步证实了其有效性，SE的AUC ROC达到0.97。尽管语义聚类仅在30%的情况下成功，SE仍然是提高妇产科领域AI安全性的有价值工具。这些发现表明，SE可能有助于使更可靠的AI集成到临床实践中，尤其是在资源受限的环境中，LLM可以增强护理。本研究表明，SE作为AI驱动工具负责任部署的关键保障之一，具有潜在价值，有助于实现更安全和更有效的数字健康干预。 

---
# Decoupling Content and Expression: Two-Dimensional Detection of AI-Generated Text 

**Title (ZH)**: 内容与表达解耦：二维检测AI生成文本 

**Authors**: Guangsheng Bao, Lihua Rong, Yanbin Zhao, Qiji Zhou, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00258)  

**Abstract**: The wide usage of LLMs raises critical requirements on detecting AI participation in texts. Existing studies investigate these detections in scattered contexts, leaving a systematic and unified approach unexplored. In this paper, we present HART, a hierarchical framework of AI risk levels, each corresponding to a detection task. To address these tasks, we propose a novel 2D Detection Method, decoupling a text into content and language expression. Our findings show that content is resistant to surface-level changes, which can serve as a key feature for detection. Experiments demonstrate that 2D method significantly outperforms existing detectors, achieving an AUROC improvement from 0.705 to 0.849 for level-2 detection and from 0.807 to 0.886 for RAID. We release our data and code at this https URL. 

**Abstract (ZH)**: 大规模语言模型的广泛使用对检测文本中的AI参与提出了关键要求。现有研究在分散的背景下探讨了这些检测，但缺乏一个系统性和统一的方法。本文提出HART，一种层次化的AI风险水平框架，每个级别对应一个检测任务。为了解决这些任务，我们提出了一个新颖的2D检测方法，将文本分解为内容和语言表达。研究发现内容对抗表面级变化具有抵抗力，这可以作为检测的关键特征。实验结果显示，2D方法显著优于现有检测器，在级别2检测中AUROC从0.705提高到0.849，在RAID中从0.807提高到0.886。我们在此httpsURL发布了我们的数据和代码。 

---
# Jawaher: A Multidialectal Dataset of Arabic Proverbs for LLM Benchmarking 

**Title (ZH)**: Jawaher：阿拉伯谚语多方言数据集用于LLM基准测试 

**Authors**: Samar M. Magdy, Sang Yun Kwon, Fakhraddin Alwajih, Safaa Abdelfadil, Shady Shehata, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2503.00231)  

**Abstract**: Recent advancements in instruction fine-tuning, alignment methods such as reinforcement learning from human feedback (RLHF), and optimization techniques like direct preference optimization (DPO) have significantly enhanced the adaptability of large language models (LLMs) to user preferences. However, despite these innovations, many LLMs continue to exhibit biases toward Western, Anglo-centric, or American cultures, with performance on English data consistently surpassing that of other languages. This reveals a persistent cultural gap in LLMs, which complicates their ability to accurately process culturally rich and diverse figurative language such as proverbs. To address this, we introduce Jawaher, a benchmark designed to assess LLMs' capacity to comprehend and interpret Arabic proverbs. Jawaher includes proverbs from various Arabic dialects, along with idiomatic translations and explanations. Through extensive evaluations of both open- and closed-source models, we find that while LLMs can generate idiomatically accurate translations, they struggle with producing culturally nuanced and contextually relevant explanations. These findings highlight the need for ongoing model refinement and dataset expansion to bridge the cultural gap in figurative language processing. 

**Abstract (ZH)**: 近期对指令微调、基于人类反馈的强化学习（RLHF）对齐方法以及直接偏好优化（DPO）等优化技术的发展显著提升了大规模语言模型（LLMs）对用户偏好的适应性。尽管如此，许多LLMs仍然倾向于西方、盎格鲁中心或美国文化，并且在英语数据上的表现持续优于其他语言。这揭示了LLMs中长期存在的文化差距，使得它们难以准确处理富含文化丰富性和多样化隐喻语言的能力受到复杂化。为了解决这一问题，我们引入了Jawaher基准，旨在评估LLMs理解和解释阿拉伯谚语的能力。Jawaher包含各种阿拉伯方言的谚语，以及直译和解释。通过广泛评估开源和闭源模型，我们发现，虽然LLMs能够生成语用上准确的翻译，但在提供具有文化内涵和语境相关性的解释方面存在困难。这些发现强调了持续模型精炼和数据集扩展的必要性，以缩小隐喻语言处理中的文化差距。 

---
# Zero-Shot and Efficient Clarification Need Prediction in Conversational Search 

**Title (ZH)**: 零样本且高效的澄清需求预测在会话搜索中 

**Authors**: Lili Lu, Chuan Meng, Federico Ravenda, Mohammad Aliannejadi, Fabio Crestani  

**Link**: [PDF](https://arxiv.org/pdf/2503.00179)  

**Abstract**: Clarification need prediction (CNP) is a key task in conversational search, aiming to predict whether to ask a clarifying question or give an answer to the current user query. However, current research on CNP suffers from the issues of limited CNP training data and low efficiency. In this paper, we propose a zero-shot and efficient CNP framework (Zef-CNP), in which we first prompt large language models (LLMs) in a zero-shot manner to generate two sets of synthetic queries: ambiguous and specific (unambiguous) queries. We then use the generated queries to train efficient CNP models. Zef-CNP eliminates the need for human-annotated clarification-need labels during training and avoids the use of LLMs with high query latency at query time. To further improve the generation quality of synthetic queries, we devise a topic-, information-need-, and query-aware chain-of-thought (CoT) prompting strategy (TIQ-CoT). Moreover, we enhance TIQ-CoT with counterfactual query generation (CoQu), which guides LLMs first to generate a specific/ambiguous query and then sequentially generate its corresponding ambiguous/specific query. Experimental results show that Zef-CNP achieves superior CNP effectiveness and efficiency compared with zero- and few-shot LLM-based CNP predictors. 

**Abstract (ZH)**: 对话澄清需求预测（CNP）是对话搜索中的一个重要任务，旨在预测是否需要提出一个澄清问题或者直接给出当前用户查询的答案。然而，当前CNP研究面临着训练数据有限和效率低的问题。本文提出了一种零样本和高效CNP框架（Zef-CNP），该框架首先以零样本方式提示大型语言模型（LLMs）生成两组合成查询：模糊和具体的查询。然后使用生成的查询来训练高效的CNP模型。Zef-CNP在训练过程中消除了对人类标注的澄清需求标签的需求，并避免了在查询时使用具有高查询延迟的LLMs。为进一步提高合成查询的生成质量，我们设计了一种主题、信息需求和查询感知的思考链提示策略（TIQ-CoT）。并且，我们通过引入反事实查询生成（CoQu）增强了TIQ-CoT，该策略首先引导LLMs生成一个具体的/模糊的查询，然后依次生成对应的模糊/具体的查询。实验结果表明，Zef-CNP在CNP效果和效率方面均优于基于零样本和少样本LLMs的CNP预测器。 

---
# Steering Large Language Model Activations in Sparse Spaces 

**Title (ZH)**: 在稀疏空间中引导大规模语言模型激活 

**Authors**: Reza Bayat, Ali Rahimi-Kalahroudi, Mohammad Pezeshki, Sarath Chandar, Pascal Vincent  

**Link**: [PDF](https://arxiv.org/pdf/2503.00177)  

**Abstract**: A key challenge in AI alignment is guiding large language models (LLMs) to follow desired behaviors at test time. Activation steering, which modifies internal model activations during inference, offers a potential solution. However, prior work in dense activation spaces struggles with superposition, wherein multiple features become entangled, limiting interpretability and precise control. In contrast, sparse representations provide an untapped opportunity for more interpretable behavior modulation. In this work, we introduce sparse activation steering (SAS), a method that leverages sparse autoencoders (SAEs) to steer LLM behavior in sparse spaces. By isolating behavior-specific features through a contrastive prompt-pairing approach, we define a set of features that can selectively reinforce or suppress behaviors. Experiments on Gemma 2 LLMs show that SAS vectors enable nuanced behavioral modulation and finer-grained control. Furthermore, scaling SAEs improves monosemanticity of SAS vectors, suggesting more reliable and interpretable interventions. 

**Abstract (ZH)**: AI对齐中的一个关键挑战是指导大型语言模型在测试时遵循期望的行为。稀疏激活导向（Sparse Activation Steering, SAS）通过利用稀疏自编码器在稀疏空间中引导LLM行为，提供了一种潜在的解决方案。通过对比提示对齐方法隔离行为特定特征，我们定义了一组可选择性地增强或抑制行为的特征。实验表明，SAS向量能够实现细腻的行为调节和更精细的控制。此外，稀疏自编码器的扩展提高了SAS向量的单一语义性，暗示了更可靠和可解释的干预措施。 

---
# Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs 

**Title (ZH)**: Palm：面向阿拉伯语言大语言模型的跨文化包容性和多语言多样性数据集 

**Authors**: Fakhraddin Alwajih, Abdellah El Mekki, Samar Mohamed Magdy, Abdelrahim A. Elmadany, Omer Nacar, El Moatez Billah Nagoudi, Reem Abdel-Salam, Hanin Atwany, Youssef Nafea, Abdulfattah Mohammed Yahya, Rahaf Alhamouri, Hamzah A. Alsayadi, Hiba Zayed, Sara Shatnawi, Serry Sibaee, Yasir Ech-Chammakhy, Walid Al-Dhabyani, Marwa Mohamed Ali, Imen Jarraya, Ahmed Oumar El-Shangiti, Aisha Alraeesi, Mohammed Anwar Al-Ghrawi, Abdulrahman S. Al-Batati, Elgizouli Mohamed, Noha Taha Elgindi, Muhammed Saeed, Houdaifa Atou, Issam Ait Yahia, Abdelhak Bouayad, Mohammed Machrouh, Amal Makouar, Dania Alkawi, Mukhtar Mohamed, Safaa Taher Abdelfadil, Amine Ziad Ounnoughene, Rouabhia Anfel, Rwaa Assi, Ahmed Sorkatti, Mohamedou Cheikh Tourad, Anis Koubaa, Ismail Berrada, Mustafa Jarrar, Shady Shehata, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2503.00151)  

**Abstract**: As large language models (LLMs) become increasingly integrated into daily life, ensuring their cultural sensitivity and inclusivity is paramount. We introduce our dataset, a year-long community-driven project covering all 22 Arab countries. The dataset includes instructions (input, response pairs) in both Modern Standard Arabic (MSA) and dialectal Arabic (DA), spanning 20 diverse topics. Built by a team of 44 researchers across the Arab world, all of whom are authors of this paper, our dataset offers a broad, inclusive perspective. We use our dataset to evaluate the cultural and dialectal capabilities of several frontier LLMs, revealing notable limitations. For instance, while closed-source LLMs generally exhibit strong performance, they are not without flaws, and smaller open-source models face greater challenges. Moreover, certain countries (e.g., Egypt, the UAE) appear better represented than others (e.g., Iraq, Mauritania, Yemen). Our annotation guidelines, code, and data for reproducibility are publicly available. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）日益融入日常生活，确保其文化敏感性和包容性至关重要。我们介绍了一个历时一年、由阿拉伯22个国家社区驱动的项目，涵盖了20个多样化的主题。该数据集包括现代标准阿拉伯语（MSA）和方言阿拉伯语（DA）的指令（输入-响应对）。由来自阿拉伯世界的44名研究人员组成的工作团队（均为本文作者）构建了该数据集，提供了宽泛且包容的视角。我们使用该数据集评估了几种前沿LLM的文化和方言能力，揭示了一些显著的局限性。例如，虽然闭源LLM通常表现出色，但它们并非完美无缺，而开源小模型面临的挑战更大。此外，某些国家（如埃及、阿联酋）似乎比其他国家（如伊拉克、毛里塔尼亚、也门）更好地被代表。我们提供的注释指南、代码和数据均对外开放，以确保可重复性。 

---
# Evaluation of LLMs-based Hidden States as Author Representations for Psychological Human-Centered NLP Tasks 

**Title (ZH)**: 基于LLMs的隐藏状态作为心理人本导向NLP任务的作者表示评价 

**Authors**: Nikita Soni, Pranav Chitale, Khushboo Singh, Niranjan Balasubramanian, H. Andrew Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2503.00124)  

**Abstract**: Like most of NLP, models for human-centered NLP tasks -- tasks attempting to assess author-level information -- predominantly use representations derived from hidden states of Transformer-based LLMs. However, what component of the LM is used for the representation varies widely. Moreover, there is a need for Human Language Models (HuLMs) that implicitly model the author and provide a user-level hidden state. Here, we systematically evaluate different ways of representing documents and users using different LM and HuLM architectures to predict task outcomes as both dynamically changing states and averaged trait-like user-level attributes of valence, arousal, empathy, and distress. We find that representing documents as an average of the token hidden states performs the best generally. Further, while a user-level hidden state itself is rarely the best representation, we find its inclusion in the model strengthens token or document embeddings used to derive document- and user-level representations resulting in best performances. 

**Abstract (ZH)**: 像大多数NLP任务一样，针对人类中心的NLP任务——旨在评估作者级别的信息的任务——主要使用基于Transformer大型语言模型（LLM）隐藏状态的表示。然而，用于表示的LM组件差异较大。此外，需要隐含建模作者的人类语言模型（HuLM），并提供用户级别的隐藏状态。在此，我们系统地评估了使用不同LM和HuLM架构表示文档和用户的不同方式，以预测任务结果作为动态变化的状态和平均的情感、唤醒度、共情和困扰等用户级别特质属性。我们发现，将文档表示为token隐藏状态的平均值通常性能最佳。此外，虽然用户级别的隐藏状态本身很少是最优表示，但我们发现其包含在模型中增强了用于生成文档级和用户级表示的token或文档嵌入，从而实现了最佳性能。 

---
# BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology 

**Title (ZH)**: BixBench: 一个全面的基于计算生物学的LLM代理基准测试 

**Authors**: Ludovico Mitchener, Jon M Laurent, Benjamin Tenmann, Siddharth Narayanan, Geemi P Wellawatte, Andrew White, Lorenzo Sani, Samuel G Rodriques  

**Link**: [PDF](https://arxiv.org/pdf/2503.00096)  

**Abstract**: Large Language Models (LLMs) and LLM-based agents show great promise in accelerating scientific research. Existing benchmarks for measuring this potential and guiding future development continue to evolve from pure recall and rote knowledge tasks, towards more practical work such as literature review and experimental planning. Bioinformatics is a domain where fully autonomous AI-driven discovery may be near, but no extensive benchmarks for measuring progress have been introduced to date. We therefore present the Bioinformatics Benchmark (BixBench), a dataset comprising over 50 real-world scenarios of practical biological data analysis with nearly 300 associated open-answer questions designed to measure the ability of LLM-based agents to explore biological datasets, perform long, multi-step analytical trajectories, and interpret the nuanced results of those analyses. We evaluate the performance of two frontier LLMs (GPT-4o and Claude 3.5 Sonnet) using a custom agent framework we open source. We find that even the latest frontier models only achieve 17% accuracy in the open-answer regime, and no better than random in a multiple-choice setting. By exposing the current limitations of frontier models, we hope BixBench can spur the development of agents capable of conducting rigorous bioinformatic analysis and accelerate scientific discovery. 

**Abstract (ZH)**: 大型语言模型及其基于模型的代理在加速科学研究方面展现出巨大潜力。Bioinformatics基准（BixBench）：评估大型语言模型代理在生物数据分析中的能力 

---
# Rethinking LLM Bias Probing Using Lessons from the Social Sciences 

**Title (ZH)**: 重新思考大语言模型偏差探究：从社会科学中汲取教训 

**Authors**: Kirsten N. Morehouse, Siddharth Swaroop, Weiwei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00093)  

**Abstract**: The proliferation of LLM bias probes introduces three significant challenges: (1) we lack principled criteria for choosing appropriate probes, (2) we lack a system for reconciling conflicting results across probes, and (3) we lack formal frameworks for reasoning about when (and why) probe results will generalize to real user behavior. We address these challenges by systematizing LLM social bias probing using actionable insights from social sciences. We then introduce EcoLevels - a framework that helps (a) determine appropriate bias probes, (b) reconcile conflicting findings across probes, and (c) generate predictions about bias generalization. Overall, we ground our analysis in social science research because many LLM probes are direct applications of human probes, and these fields have faced similar challenges when studying social bias in humans. Based on our work, we suggest how the next generation of LLM bias probing can (and should) benefit from decades of social science research. 

**Abstract (ZH)**: LLM 社会偏见探针的普及引入了三个重大挑战：（1）缺乏选择合适探针的原理性标准，（2）缺乏综合跨探针冲突结果的系统方法，（3）缺乏关于何时以及为何探针结果能够泛化到实际用户行为的正式框架。我们通过结合社会科学的实用启示系统化 LLM 社会偏见探针来应对这些挑战，进而介绍 EcoLevels - 一个帮助（a）确定合适偏见探针，（b）综合跨探针的冲突发现，以及（c）生成偏见泛化的预测的框架。总体而言，我们基于社会科学研究来开展分析，因为许多 LLM 探针直接借鉴了针对人类的探针，而这些领域在研究人类社会偏见时也面临类似的挑战。基于我们的研究，我们建议下一代 LLM 偏见探针可以从数十年的社会科学研究中获益并从中受益。 

---
# EdgeAIGuard: Agentic LLMs for Minor Protection in Digital Spaces 

**Title (ZH)**: EdgeAIGuard：数字空间中保护minor的代理型LLM 

**Authors**: Ghulam Mujtaba, Sunder Ali Khowaja, Kapal Dev  

**Link**: [PDF](https://arxiv.org/pdf/2503.00092)  

**Abstract**: Social media has become integral to minors' daily lives and is used for various purposes, such as making friends, exploring shared interests, and engaging in educational activities. However, the increase in screen time has also led to heightened challenges, including cyberbullying, online grooming, and exploitations posed by malicious actors. Traditional content moderation techniques have proven ineffective against exploiters' evolving tactics. To address these growing challenges, we propose the EdgeAIGuard content moderation approach that is designed to protect minors from online grooming and various forms of digital exploitation. The proposed method comprises a multi-agent architecture deployed strategically at the network edge to enable rapid detection with low latency and prevent harmful content targeting minors. The experimental results show the proposed method is significantly more effective than the existing approaches. 

**Abstract (ZH)**: 社交媒体已成为未成年人日常生活中不可或缺的一部分，被用于交友、探索共同兴趣和参与教育活动等多种目的。然而，屏幕时间的增加也带来了新的挑战，包括网络欺凌、在线诱骗和恶意行为者的利用。传统的内容审核技术对于行为者不断变化的策略证明效果不佳。为应对这些 growing 挑战，我们提出了一种名为 EdgeAIGuard 的内容审核方法，旨在保护未成年人免受在线诱骗和多种形式的数字利用。该方法包含一个部署在网络边缘的多代理结构，以实现快速检测和低延迟，防止针对未成年人的有害内容。实验结果表明，所提出的方法显著优于现有方法。 

---
# Societal Alignment Frameworks Can Improve LLM Alignment 

**Title (ZH)**: 社会对齐框架可以提高LLM对齐程度 

**Authors**: Karolina Stańczak, Nicholas Meade, Mehar Bhatia, Hattie Zhou, Konstantin Böttinger, Jeremy Barnes, Jason Stanley, Jessica Montgomery, Richard Zemel, Nicolas Papernot, Nicolas Chapados, Denis Therien, Timothy P. Lillicrap, Ana Marasović, Sylvie Delacroix, Gillian K. Hadfield, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.00069)  

**Abstract**: Recent progress in large language models (LLMs) has focused on producing responses that meet human expectations and align with shared values - a process coined alignment. However, aligning LLMs remains challenging due to the inherent disconnect between the complexity of human values and the narrow nature of the technological approaches designed to address them. Current alignment methods often lead to misspecified objectives, reflecting the broader issue of incomplete contracts, the impracticality of specifying a contract between a model developer, and the model that accounts for every scenario in LLM alignment. In this paper, we argue that improving LLM alignment requires incorporating insights from societal alignment frameworks, including social, economic, and contractual alignment, and discuss potential solutions drawn from these domains. Given the role of uncertainty within societal alignment frameworks, we then investigate how it manifests in LLM alignment. We end our discussion by offering an alternative view on LLM alignment, framing the underspecified nature of its objectives as an opportunity rather than perfect their specification. Beyond technical improvements in LLM alignment, we discuss the need for participatory alignment interface designs. 

**Abstract (ZH)**: Recent progress in大型语言模型（LLMs）的Recent进展集中在生成符合人类期望和共享价值观的回应——这一过程被称为对齐。然而，由于人类价值观的复杂性与技术方法的狭窄性质之间固有的不匹配，LLMs对齐仍然具有挑战性。当前的对齐方法往往导致目标定义不准确，反映出更广泛的问题即不完全合同的存在，模型开发人员与模型之间的合同无法涵盖所有LLMs对齐场景。本文认为，提高LLMs对齐需要借鉴社会对齐框架的见解，包括社会、经济和合同对齐，并讨论这些领域中潜在的解决方案。鉴于社会对齐框架中的不确定性，我们探讨了它如何在LLMs对齐中显现。在讨论的结尾，我们提出了对LLMs对齐的一种替代观点，将目标定义不明确视为机遇而非完善其定义的机会。超越LLMs对齐的技术改进，我们讨论了参与式对齐接口设计的需求。 

---
# Leveraging Large Models for Evaluating Novel Content: A Case Study on Advertisement Creativity 

**Title (ZH)**: 利用大型模型评估新颖内容：广告创意案例研究 

**Authors**: Zhaoyi Joey Hou, Adriana Kovashka, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00046)  

**Abstract**: Evaluating creativity is challenging, even for humans, not only because of its subjectivity but also because it involves complex cognitive processes. Inspired by work in marketing, we attempt to break down visual advertisement creativity into atypicality and originality. With fine-grained human annotations on these dimensions, we propose a suit of tasks specifically for such a subjective problem. We also evaluate the alignment between state-of-the-art (SoTA) vision language models (VLM) and humans on our proposed benchmark, demonstrating both the promises and challenges of using VLMs for automatic creativity assessment. 

**Abstract (ZH)**: 评估创造力具有挑战性，即使是对于人类而言，不仅由于其主观性，还由于其中涉及的复杂认知过程。受营销领域工作的启发，我们尝试将视觉广告创意分解为非典型性和原创性。通过在这些维度上进行细致的人工标注，我们提出了一系列专门针对此类主观问题的任务。我们还评估了最新视觉语言模型（VLM）与人类在我们提出的基准上的对齐情况，展示了使用视觉语言模型进行自动创造力评估的潜力与挑战。 

---
# VOILA: Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning 

**Title (ZH)**: VOILA：评估MLLMs在感知理解和类比推理方面的表现 

**Authors**: Nilay Yilmaz, Maitreya Patel, Yiran Lawrence Luo, Tejas Gokhale, Chitta Baral, Suren Jayasuriya, Yezhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00043)  

**Abstract**: Multimodal Large Language Models (MLLMs) have become a powerful tool for integrating visual and textual information. Despite their exceptional performance on visual understanding benchmarks, measuring their ability to reason abstractly across multiple images remains a significant challenge. To address this, we introduce VOILA, a large-scale, open-ended, dynamic benchmark designed to evaluate MLLMs' perceptual understanding and abstract relational reasoning. VOILA employs an analogical mapping approach in the visual domain, requiring models to generate an image that completes an analogy between two given image pairs, reference and application, without relying on predefined choices. Our experiments demonstrate that the analogical reasoning tasks in VOILA present a challenge to MLLMs. Through multi-step analysis, we reveal that current MLLMs struggle to comprehend inter-image relationships and exhibit limited capabilities in high-level relational reasoning. Notably, we observe that performance improves when following a multi-step strategy of least-to-most prompting. Comprehensive evaluations on open-source models and GPT-4o show that on text-based answers, the best accuracy for challenging scenarios is 13% (LLaMa 3.2) and even for simpler tasks is only 29% (GPT-4o), while human performance is significantly higher at 70% across both difficulty levels. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）已成为整合视觉和文本信息的强大力量。尽管它们在视觉理解基准测试中表现出色，但评估其在多个图像之间进行抽象推理的能力仍然是一个重大挑战。为了解决这一问题，我们引入了VOILA，一个大规模、开放式、动态基准，旨在评估MLLMs的感知理解和抽象关系推理能力。VOILA采用视觉领域的类比映射方法，要求模型生成一个完成给定图像对（参考和应用）之间类比关系的图像，而无需依赖预定义的选择。我们的实验表明，VOILA中的类比推理任务对MLLMs构成了挑战。通过多层次分析，我们发现当前的MLLMs在理解跨图像关系以及在高层次关系推理方面能力有限。值得注意的是，我们观察到，遵循最少到最详细提示的多步策略可以提高性能。在开源模型和GPT-4o的全面评估中，对于基于文本的答案，在挑战性场景中最佳准确率为13%（LLaMa 3.2），即使是对于更简单的任务，准确率也只有29%（GPT-4o），而人类的表现在这两种难度级别上明显更高，达到70%。 

---
# from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors 

**Title (ZH)**: 从良性转化为恶意：通过对抗隐喻突破语言模型 

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Qi Li, Jiangyu Lei  

**Link**: [PDF](https://arxiv.org/pdf/2503.00038)  

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. 

**Abstract (ZH)**: 当前的研究揭示了大型语言模型（LLMs）通过破解攻击生成有害内容的风险。然而，这些研究忽视了从头生成有害内容比诱导LLMs将良性内容转变为有害形式更为困难。在我们的研究中，我们引入了一个新的攻击框架，利用Adversarial MetaPHOR (AVATAR) 诱导LLMs生成恶意隐喻以实现破解。具体而言，为了回答有害查询，AVATAR会自适应地识别一组良性但逻辑相关的隐喻作为初始种子。然后，在这些隐喻的驱动下，目标LLM被诱导进行隐喻内容的推理和校准，从而通过直接输出有害回复或校准隐喻与专业有害内容之间的残差来实现破解。实验结果表明，AVATAR能够有效且可移植地破解LLMs，并在多个高级LLM上实现了最先进的攻击成功率。 

---
# Constraining Sequential Model Editing with Editing Anchor Compression 

**Title (ZH)**: 基于编辑锚压缩的序列模型编辑约束 

**Authors**: Hao-Xiang Xu, Jun-Yu Ma, Zhen-Hua Ling, Ningyu Zhang, Jia-Chen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00035)  

**Abstract**: Large language models (LLMs) struggle with hallucinations due to false or outdated knowledge. Given the high resource demands of retraining these models, there is an increasing focus on developing model editing. However, the general abilities of LLMs across downstream tasks are prone to significant degradation during sequential editing. This paper statistically observes that the parameter matrix after editing exhibits a significant deviation compared to its previous state as the number of edits increases. This serious deviation affects the original knowledge associations within LLMs and leads to the degradation of their general abilities. To this end, a framework termed Editing Anchor Compression (EAC) is proposed to constrain the deviation of the parameter matrix during sequential editing. It compresses the editing information by selecting editing anchors that are important in encoding new relations without deviating too much from the original matrix, thereby preserving the general abilities. Experiments of applying EAC to two popular editing methods on three LLMs across four tasks are conducted. Evaluation results show that EAC effectively minimizes unreasonable deviations caused by model editing, preserving over 70% of the general abilities while better retaining the editing knowledge compared to the original counterpart methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）因虚假或过时的知识而导致幻觉问题。鉴于重新训练这些模型所需资源的高需求，越来越多的研究集中在开发模型编辑方法上。然而，在顺序编辑过程中，LLMs在下游任务上的普遍能力容易出现显著下降。本文通过统计观察发现，在编辑次数增加时，参数矩阵与之前的状态相比表现出显著的偏差。这种严重的偏差影响了LLMs中的原始知识关联，导致其普遍能力下降。为此，提出了一种称为编辑锚压缩（EAC）的框架，在顺序编辑过程中限制参数矩阵的偏差。EAC通过选择在编码新关系方面重要的编辑锚点来压缩编辑信息，同时不偏离原始矩阵太多，从而保留普遍能力。在三个LLM上针对四个任务对两种流行编辑方法应用EAC的实验结果显示，EAC有效地最小化了由模型编辑引起的不合理偏差，同时保留了超过70%的普遍能力，并且相比原始方法更好地保留了编辑知识。 

---
# MergeIT: From Selection to Merging for Efficient Instruction Tuning 

**Title (ZH)**: MergeIT：从选择到合并的高效指令调优 

**Authors**: Hongyi Cai, Yuqian Fu, Hongming Fu, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00034)  

**Abstract**: Instruction tuning is crucial for optimizing Large Language Models (LLMs), yet mainstream data selection methods heavily rely on LLMs as instruction quality scorers, leading to high computational costs and reduced data diversity. To address these limitations, we propose MergeIT, a novel LLM-based Merging strategy for better Instruction Tuning that shifts the focus from selection to synthesis. MergeIT operates in two stages: first, topic-aware filtering clusters and refines the dataset, preserving diversity while eliminating redundancy without relying on LLM-based scoring. Second, LLM-based merging synthesizes semantically similar instructions into more informative and compact training data, enhancing data richness while further reducing dataset size. Experimental results demonstrate that MergeIT enables efficient, diverse, and scalable instruction selection and synthesis, establishing LLM-based merging as a promising alternative to conventional scoring-based selection methods for instruction tuning. Our source code and datasets are now available at this https URL 

**Abstract (ZH)**: 基于LLM的合成策略MergeIT：一种用于更好的指令调优的数据合并方法 

---
# Detecting LLM-Generated Korean Text through Linguistic Feature Analysis 

**Title (ZH)**: 通过语言特征分析检测LLM生成的韩文文本 

**Authors**: Shinwoo Park, Shubin Kim, Do-Kyung Kim, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.00032)  

**Abstract**: The rapid advancement of large language models (LLMs) increases the difficulty of distinguishing between human-written and LLM-generated text. Detecting LLM-generated text is crucial for upholding academic integrity, preventing plagiarism, protecting copyrights, and ensuring ethical research practices. Most prior studies on detecting LLM-generated text focus primarily on English text. However, languages with distinct morphological and syntactic characteristics require specialized detection approaches. Their unique structures and usage patterns can hinder the direct application of methods primarily designed for English. Among such languages, we focus on Korean, which has relatively flexible spacing rules, a rich morphological system, and less frequent comma usage compared to English. We introduce KatFish, the first benchmark dataset for detecting LLM-generated Korean text. The dataset consists of text written by humans and generated by four LLMs across three genres.
By examining spacing patterns, part-of-speech diversity, and comma usage, we illuminate the linguistic differences between human-written and LLM-generated Korean text. Building on these observations, we propose KatFishNet, a detection method specifically designed for the Korean language. KatFishNet achieves an average of 19.78% higher AUROC compared to the best-performing existing detection method. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速发展增加了区分人类撰写和LLM生成文本的难度。检测LLM生成的文本对于维护学术诚信、防止学术剽窃、保护版权和确保伦理研究实践至关重要。大多数关于检测LLM生成文本的研究主要集中在英语文本上。然而，具有不同词形和句法特征的语言需要专门的检测方法。由于它们独特的结构和使用模式，这些语言的方法往往难以直接应用于英语主要设计的方法。在这些语言中，我们重点关注韩语，韩语有相对较灵活的空格规则、丰富的形态系统以及比英语更少的逗号使用。我们介绍了KatFish，这是首个用于检测LLM生成韩语文本的标准数据集。该数据集包括人类撰写的文本和由四种LLM在三个体裁中生成的文本。通过分析空格模式、词性多样性和逗号使用，我们揭示了人类撰写和LLM生成韩语文本之间的语言差异。在此基础上，我们提出了KatFishNet，这是一种专门针对韩语设计的检测方法。KatFishNet在平均AUROC方面比现有最佳检测方法高出19.78%。 

---
# Efficient Test-Time Scaling via Self-Calibration 

**Title (ZH)**: 高效的测试时缩放通过自校准 

**Authors**: Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00031)  

**Abstract**: Increasing test-time computation is a straightforward approach to enhancing the quality of responses in Large Language Models (LLMs). While Best-of-N sampling and Self-Consistency with majority voting are simple and effective, they require a fixed number of sampling responses for each query, regardless of its complexity. This could result in wasted computation for simpler questions and insufficient exploration for more challenging ones. In this work, we argue that model confidence of responses can be used for improving the efficiency of test-time scaling. Unfortunately, LLMs are known to be overconfident and provide unreliable confidence estimation. To address this limitation, we introduce Self-Calibration by distilling Self-Consistency-derived confidence into the model itself. This enables reliable confidence estimation at test time with one forward pass. We then design confidence-based efficient test-time scaling methods to handle queries of various difficulty, such as Early-Stopping for Best-of-N and Self-Consistency with calibrated confidence. Experiments on three LLMs across six datasets demonstrate the effectiveness of our approach. Specifically, applying confidence-based Early Stopping to Best-of-N improves MathQA accuracy from 81.0 to 83.6 with a sample budget of 16 responses, indicating the efficacy of confidence-based sampling strategy at inference time. 

**Abstract (ZH)**: 使用模型响应的信心提高大型语言模型测试时放规模效率 

---
# Game-Theoretic Regularized Self-Play Alignment of Large Language Models 

**Title (ZH)**: 基于博弈的正则化自我对弈大型语言模型对齐 

**Authors**: Xiaohang Tang, Sangwoong Yoon, Seongho Son, Huizhuo Yuan, Quanquan Gu, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.00030)  

**Abstract**: Self-play alignment algorithms have been developed as effective methods for fine-tuning large language models (LLMs), formulating preference optimization as a two-player game. However, the regularization with respect to the reference policy, which is crucial for mitigating over-optimization, has been insufficiently investigated in self-play alignment. In this paper, we show that our regularization method can improve the unregularized self-play significantly. To study the impact of different regularizations in self-play alignment, we propose Regularized Self-Play Policy Optimization (RSPO). This generalized framework regularizes the self-play by simply adding a chosen regularization term into the loss while maintaining provable last-iterate convergence to the Nash Equilibrium of the corresponding regularized game. Surprisingly, empirical evaluations using the Mistral-7B-Instruct base model reveal that forward KL divergence regularization reduces response length in RSPO, whereas reverse KL divergence markedly improves raw win rates. RSPO with a linear combination of forward and reverse KL divergence regularization substantially increases the length-controlled win rate in AlpacaEval-2, elevating the unregularized self-play alignment method (SPPO) from $28.53\%$ to $35.44\%$. Finally, we show that RSPO also improves the response diversity. 

**Abstract (ZH)**: 自游戏对齐算法中的正则化方法对大型语言模型微调的影响研究：Regularized Self-Play Policy Optimization 

---
# Evaluating Large Language Models on the Spanish Medical Intern Resident (MIR) Examination 2024/2025:A Comparative Analysis of Clinical Reasoning and Knowledge Application 

**Title (ZH)**: 评价大型语言模型在2024/2025年西班牙医学实习居民（MIR）考试中的临床推理和知识应用能力：一种比较分析 

**Authors**: Carlos Luengo Vera, Ignacio Ferro Picon, M. Teresa del Val Nunez, Jose Andres Gomez Gandia, Antonio de Lucas Ancillo, Victor Ramos Arroyo, Carlos Milan Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00025)  

**Abstract**: This study presents a comparative evaluation of 22 large language models LLMs on the Spanish Medical Intern Resident MIR examinations for 2024 and 2025 with a focus on clinical reasoning domain specific expertise and multimodal processing capabilities The MIR exam consisting of 210 multiple choice questions some requiring image interpretation serves as a stringent benchmark for assessing both factual recall and complex clinical problem solving skills Our investigation encompasses general purpose models such as GPT4 Claude LLaMA and Gemini as well as specialized fine tuned systems like Miri Pro which leverages proprietary Spanish healthcare data to excel in medical contexts
Recent market entries Deepseek and Grok have further enriched the evaluation landscape particularly for tasks that demand advanced visual and semantic analysis The findings indicate that while general purpose LLMs perform robustly overall fine tuned models consistently achieve superior accuracy especially in addressing nuanced domain specific challenges A modest performance decline observed between the two exam cycles appears attributable to the implementation of modified questions designed to mitigate reliance on memorization
The results underscore the transformative potential of domain specific fine tuning and multimodal integration in advancing medical AI applications They also highlight critical implications for the future integration of LLMs into medical education training and clinical decision making emphasizing the importance of balancing automated reasoning with ethical and context aware judgment 

**Abstract (ZH)**: 本研究对22个大型语言模型在2024年和2025年西班牙医学住院医师MIR考试中的表现进行了比较评估，重点考察了临床推理领域的专业能力和多模态处理能力。MIR考试由210道多项选择题组成，其中一些题目要求进行图像解读，作为衡量事实回忆和复杂临床问题解决能力的严格基准。本调查涵盖了通用模型（如GPT4、Claude、LLaMA和Gemini）以及像Miri Pro这样的专门微调系统，后者利用专有西班牙医疗服务数据在医疗场景中表现出色。

近期市场上的Deepseek和Grok进一步丰富了评估景观，特别是在需要高级视觉和语义分析的任务中。研究发现，虽然通用大型语言模型整体表现稳健，但专门微调模型在应对细微的专业领域挑战时始终表现出更优的准确性。观察到的两次考试周期之间的轻微性能下降可归因于设计了修改后的题目，旨在减少对记忆的依赖。

这些结果突显了特定领域微调和多模态整合在推动医疗人工智能应用方面的潜力，也强调了在未来将大型语言模型整合到医疗教育、培训和临床决策中时平衡自动推理与伦理和情境意识判断的重要性。 

---
# KVCrush: Key value cache size-reduction using similarity in head-behaviour 

**Title (ZH)**: KVCrush: 基于头部行为相似性的键值缓存大小缩减 

**Authors**: Gopi Krishna Jha, Sameh Gobriel, Liubov Talamanova, Alexander Kozlov, Nilesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2503.00022)  

**Abstract**: Key-value (KV) caching has emerged as a crucial optimization technique for accelerating inference in large language models (LLMs). By allowing the attention operation to scale linearly rather than quadratically with the total sequence length, KV caching significantly enhances generation throughput. However, due to large context lengths in the modern LLMs, the memory footprint of the KV is a huge bottleneck for model deployment directly impacting the model's batch size, hindering its ability to deliver high-throughput. Existing research addresses this challenge using several techniques, such as discarding low-attention tokens, quantization, and matrix approximation which typically lead to a negative impact on the model accuracy.
In this paper, We propose KVCrush technology which can be combined with many KV compression technologies to improve the model accuracy at a much smaller memory. KVCrush provides an alternate representation scheme for key-value states, along with a low-overhead token pruning algorithm that accounts for the token distribution in the KV cache, which in turn allows for a a smaller footprint while maintaining the accuracy of the model. Based on our results, KVCrush reduces LongBench KV Cache size by 4x with less than 1% accuracy drop and achieves state-of-the-art average accuracy with minimal overhead, incurring less than 0.5% total inference latency. KVCrush not only outperforms the accuracy of state-of-the-art importance-based token retention schemes but is also compatible with typical practical LLM deployments using KV cache paging schemes such as vLLM and mixed precision quantization. 

**Abstract (ZH)**: KVCrush技术：在更小内存 footprint 下提升模型精度的新方法 

---
# Eeyore: Realistic Depression Simulation via Supervised and Preference Optimization 

**Title (ZH)**: Eeyore: 基于监督和偏好优化的现实抑郁症模拟 

**Authors**: Siyang Liu, Bianca Brie, Wenda Li, Laura Biester, Andrew Lee, James Pennebaker, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2503.00018)  

**Abstract**: Large Language Models (LLMs) have been previously explored for mental healthcare training and therapy client simulation, but they still fall short in authentically capturing diverse client traits and psychological conditions. We introduce \textbf{Eeyore}, an 8B model optimized for realistic depression simulation through a structured alignment framework, incorporating expert input at every stage. First, we systematically curate real-world depression-related conversations, extracting depressive traits to guide data filtering and psychological profile construction, and use this dataset to instruction-tune Eeyore for profile adherence. Next, to further enhance realism, Eeyore undergoes iterative preference optimization -- first leveraging model-generated preferences and then calibrating with a small set of expert-annotated preferences. Throughout the entire pipeline, we actively collaborate with domain experts, developing interactive interfaces to validate trait extraction and iteratively refine structured psychological profiles for clinically meaningful role-play customization. Despite its smaller model size, the Eeyore depression simulation outperforms GPT-4o with SOTA prompting strategies, both in linguistic authenticity and profile adherence. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已在心理健康培训和治疗模拟中得到探索，但仍无法真实地捕捉多样化的患者特质和心理状态。我们引入了\textbf{Eeyore}，一个通过结构化对齐框架优化、并结合每个阶段的专家输入以实现真实的抑郁模拟的8B模型。首先，我们系统地整理了与抑郁相关的现实对话，提取抑郁特质以指导数据过滤和心理档案构建，并使用该数据集对Eeyore进行指令调优，使其符合心理档案。然后，为了进一步增强真实感，Eeyore经历了迭代的偏好优化——首先利用模型生成的偏好，然后与少量专家标注的偏好进行校准。在整个流程中，我们积极与领域专家合作，开发交互界面以验证特质提取，并迭代细化结构化心理档案以实现临床意义的角色扮演定制。尽管模型规模较小，但Eeyore的抑郁模拟在语言的真实性及心理档案贴合度上均优于具有最新技术策略的GPT-4o。 

---
